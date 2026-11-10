## Highlights

TMT: Adapting CLIP to Infrared Imagery via Text-Derived Modality Knowledge Transfer Ruofan Wang, Puzuo Wang, Xin Chen, Zibo Yu, Jungang Yang, Yi Hou

• Pseudo-visible features train an IR encoder for direct CLIP text matching.

• Generic modality descriptions define an IR-to-visible transfer direction.

• Paired semantic alignment and difusion feedback jointly adapt the IR encoder.

# TMT: Adapting CLIP to Infrared Imagery via Text-Derived Modality Knowledge Transfer

Ruofan Wang<sup>a</sup>, Puzuo Wang<sup>a</sup>, Xin Chen<sup>a</sup>, Zibo Yu<sup>a</sup>, Jungang Yang<sup>a</sup> and Yi Hou<sup>a,∗</sup>

<sup>a</sup>College of Electronic Science and Technology, National University of Defense Technology, Changsha, Hunan, China

## A R T I C L E I N F O

Keywords:   
Infrared vision   
Vision–language models   
CLIP adaptation   
Text-derived modality transfer   
Infrared–visible image pairs   
Difusion noise-prediction supervision

## A BS T RA C T

Vision-language models such as CLIP achieve strong performance across computer vision tasks, yet their image–text alignment is less reliable for infrared (IR) imagery because large-scale pretraining is dominated by visible-spectrum data. Due to the high cost of the infrared–text labeling, we investigate adapting CLIP to IR through only paired infrared–visible images, which requires considering modality-specific knowledge while preserving scene content and CLIP’s image–text alignment. To achieve this, we propose TMT (Text-Derived Modality Knowledge Transfer), which adapts CLIP’s image encoder through a text-defined infrared-to-visible direction and supervision from paired IR–visible images. First, we found that modality discrepancy can be described by corresponding text embeddings, thus using the direction of text embedding diferences to guide the knowledge transfer between visible and infrared modalities without erasing infrared-specific information. To enhance the representation of semantics and details of infrared images, we introduce Generative– Perceptual Co-Supervision that combines semantic alignment from CLIP visible image encoder with difusion noise-prediction feedbacks. On four source-disjoint IR classification benchmarks, TMT achieves the highest mean zero-shot accuracy among the compared methods (48.92%). It also achieves the best IR–text retrieval results among the compared methods without infrared–text supervision during adaptation and the highest open-vocabulary segmentation mIoU on both CART and SemanticRT among the CLIP-like methods.

## 1. Introduction

Infrared (IR) imaging captures emitted radiation rather than reflected visible light [17] and supports perception when visible appearance is degraded, including multispectral object detection [2, 31], aerial RGB–thermal perception [14], and multispectral semantic segmentation [10]. In the visible domain, vision–language pre-training has turned image–text alignment into a general-purpose interface for a wide range of visual tasks. Extending this interface to infrared imagery is particularly attractive and motivates the development of infrared representation models, including CLIP-style models that align images with natural language in the infrared domain. However, the native CLIP-style models meet challenges when applied to infrared imagery: they are pre-trained predominantly on visible-spectrum data, so their performance degrades on infrared data.

To mitigate this degradation, existing methods connect infrared imagery to language in diferent ways, drawing on resources that are already available, such as paired infrared–visible images and pre-trained vision–language models in the visible domain. A direct method fine-tunes image–text models with paired infrared–text supervision. However, manually annotating fine-grained infrared captions is expensive, which makes this supervision dificult to scale for large-scale infrared training. Synthesizing such captions with large vision–language models ofers one way to reduce this cost [20], while the thermal fidelity of the synthesized captions remains bounded by the generator’s predominantly visible-spectrum knowledge. A second route embeds infrared into a multimodal alignment framework. LanguageBind [35] aligns infrared features, together with features from multiple other modalities, toward the same text. ImageBind [6] connects infrared features to a shared multimodal space through paired infrared–visible images. However, by forcing infrared features into alignment with other modalities—particularly other image modalities— these methods tend to lose modality-specific information of infrared, which in turn degrades performance on downstream tasks.

The limitations of the above routes arise from how they use visible-domain resources: one depends on infrared captions that are either costly to annotate or bounded in fidelity when synthesized, and the other forces infrared features into alignment with other modalities without preserving their modality-specific characteristics. We instead consider moving the infrared feature itself toward the visible-image region of CLIP’s embedding space, forming a pseudo-visible feature that existing visible-domain models can directly process. Figure 1 illustrates the reasoning that leads to our core idea. The feature moving module serves as a bridge between the two modalities: it turns the forced alignment between infrared and visible features into the optimization of the pseudo-visible feature; the knowledge accumulated in visible-domain pre-trained models thus passes through this bridge into the infrared encoder, transferring visible-domain knowledge without infrared–text annotations and without erasing infrared-specific information. Then, the question is how to construct this bridge and what supervision can make the knowledge of visible-domain models pass through it.

To answer this question, we examine what the paired images do and do not provide: they preserve scene and object correspondence, but the spectral diference still needs to be expressed in CLIP’s image–text embedding space. CLIP provides a shared space for image and text representations [21]. Text-embedding diferences have been used to steer CLIP-conditioned image generation [22] and to guide semantic augmentation across visual domains [28]. These precedents motivate a candidate infrared-to-visible direction constructed from generic descriptions of the two modalities; whether this direction helps infrared alignment is tested in our experiments.

Once an infrared feature is transferred along this direction, CLIP’s global image–text alignment can guide it toward the correct semantic content, but it does not directly specify which local visual details the feature should retain [29, 15]. A frozen CLIP image encoder can anchor the feature to its paired visible image, yet this semantic anchoring is not sensitive to local image detail. A complementary detailsensitive signal is therefore needed, and a frozen difusion model—which, following unCLIP, generates visible images conditioned on CLIP image embeddings [22]—provides it through noise-prediction feedback.

We therefore propose TMT (Text-Derived Modality Knowledge Transfer), which adapts CLIP’s image encoder to infrared inputs through a text-defined transfer direction and paired semantic and difusion supervision. The key insight is that infrared and visible images may difer substantially in radiometric appearance and local detail because they arise from diferent sensing processes [17], while paired observations still preserve object identity and coarse scene structure across spectra [12, 16]. This separation allows modality-specific changes to be represented and transferred without treating the two images as pixel-identical observations.

TMT first encodes a matched pair of generic visible and infrared descriptions with the frozen CLIP text encoder. It takes their visible-minus-infrared embedding diference, normalizes it, and adds the resulting direction to an infrared image feature to form a pseudo-visible feature. This construction is inspired by embedding arithmetic [18, 19] and text-diference-based image manipulation [22]; it does not require every cross-spectral diference to be linear. The direction depends on the descriptions used to define it, so Section 5 compares it with a no-transfer variant and an image-derived direction. During training, the frozen visibleimage CLIP encoder and the frozen difusion model supervise only the pseudo-visible feature. Because the added direction remains fixed, satisfying their objectives requires updating the infrared image encoder that produced it: optimizing the pseudo-visible feature is thus the route by which visible-domain knowledge adapts the infrared encoder, and at inference the adapted encoder is used alone, without the shift or the supervisors.

The main contributions of this paper are summarized as follows:

• We introduce TMT, a novel CLIP adaptation framework that uses a pseudo-visible feature as a trainingtime interface for visible-domain supervision while retaining direct infrared–text matching at inference.

• Text-Derived Modality Transfer computes a modality direction from generic visible and infrared descriptions encoded by the frozen CLIP text encoder and applies it to infrared features through vector addition.

• Generative–Perceptual Co-Supervision combines noiseprediction feedback from a frozen difusion model with paired semantic alignment to a frozen visibleimage CLIP encoder; only the infrared image encoder is updated.

• On four source-disjoint infrared classification benchmarks, TMT achieves 48.92% mean zero-shot accuracy versus 43.52% for an ImageBind-style CLIP control trained on the same IR500K image pairs. It also leads the compared methods without infrared– text supervision during adaptation in IR–text retrieval and achieves the highest mIoU among the compared CLIP-like methods on CART and SemanticRT.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the proposed TMT framework in detail. Section 4 reports the experimental setup and results, followed by ablation studies and analysis in Section 5. Finally, Section 6 concludes the paper and discusses future work.

## 2. Related Work

## 2.1. Vision–Language Alignment for Infrared Adaptation

Vision–language pre-training establishes an embedding space for comparing images with text. CLIP [21] learns this interface through image–text contrastive training, while ALIGN [11] scales alignment using noisy web supervision. Subsequent adaptation methods modify diferent parts of the interface: CoOp [34] learns continuous prompt contexts, CoCoOp [33] conditions them on the input image, and CLIP-Adapter [4] introduces residual feature adaptation. These approaches provide a basis for reusing pre-trained representations, but infrared adaptation also requires a source of supervision connecting the new sensing modality to language.

LanguageBind [35] establishes this connection directly by aligning infrared and other modalities with a frozen language encoder using paired text. ImageBind [6] instead builds an image-centered shared space. Its thermal encoder learns from paired image–thermal data from LLVIP, gaining access to language alignment through the image modality. The distinction concerns the correspondence available during training: one route supplies infrared–text pairs, whereas the other uses images as an intermediate reference. ImageBind thus already demonstrates that explicit infrared– text pairing is not required for this form of cross-modal alignment.

IRGPT [1] and Thermal-Det [24] further illustrate how supervision is organized around diferent infrared–language tasks. IRGPT uses the IR-TD corpus, comprising more than 260K infrared–text samples across pre-training, instruction, and benchmark subsets. Its bi-cross-modal curriculum and instruction tuning support generative question answering, including recognition and grounding. Thermal-Det combines synthetic thermal images with grounding text and captions for open-vocabulary detection, and distills geometric and semantic cues from a frozen RGB teacher using paired images. The comparisons therefore involve both the availability of language-linked infrared data and the form of the prediction. Their generative-answer and detection interfaces difer from the image–text similarity interface retained by an adapted CLIP encoder.

A concurrent line of work reduces the annotation cost itself rather than avoiding infrared text. T-CLIP [20] conditions a large vision–language model on the paired visible image to synthesize physics-aware thermal captions at two granularities, and adapts both CLIP encoders with two independently trained LoRA branches that are ensembled at inference for cross-modal retrieval. This route supplies instance-level infrared–text supervision through generation. TMT instead uses text only to define a shared modality direction and adapts the image encoder alone, retaining the standard single-encoder matching interface at inference and serving classification, retrieval, and segmentation.

TMT shares ImageBind’s use of infrared–visible pairs without instance-level infrared–text annotations. It introduces generic modality descriptions to specify a shared infrared-to-visible change, while paired visible images provide the content reference for each training instance. This separates the language-defined modality change from the image-defined scene correspondence. The paired observations then supervise a transferred representation through both semantic alignment and generative feedback, extending the role of image pairing in encoder adaptation. Caption synthesis and this caption-free use of paired images are complementary answers to the scarcity of infrared–text supervision.

## 2.2. Text-Guided Semantic Transfer

Semantic relationships in word embeddings can appear as approximately consistent vector ofsets [18, 19]. This observation motivates representing a specified semantic change by a direction in feature space. CLIP [21] supplies a joint image–text space in which such directions can be investigated across modalities. Alignment alone does not guarantee that arbitrary text diferences produce the intended visual change; the descriptions, representation, and transformation used remain relevant to the outcome.

In unCLIP, Ramesh et al. [22] use a normalized diference between original and target text descriptions to manipulate an image embedding through spherical interpolation. A decoder then maps the modified embedding to an image, demonstrating how a text-specified change can guide visual synthesis. CLIPGap [28] uses textual domain concepts for single-domain-generalization object detection. Its semantic augmentation operates on detector-backbone features and is combined with a text-based classification loss. Both approaches use language to specify a desired visual variation, but they difer in the representation being modified and the objective used to evaluate that modification. Together, these examples suggest that text guidance must be considered alongside the feature space and downstream objective. Successful visual editing or detector augmentation motivates further study of such directions, but provides no direct evidence of their efectiveness for infrared encoder adaptation.

TMT uses text-guided transformation to construct a training interface for infrared encoder adaptation. Generic descriptions of visible and infrared imaging define a fixed, shared direction that maps infrared features to pseudovisible representations. These representations receive supervision from visible-domain models, with gradients passing through the transfer operation to update the infrared encoder. Whereas the preceding applications target image editing or detector augmentation, this operation supports learning an infrared representation for image–text matching; the textderived shift is removed at inference.

## 2.3. Generative Supervision for Visual Representation Learning

Generative models connect visual representations to the image information needed for synthesis. DDPM [9] learns through denoising, while latent difusion models [25] perform difusion in an autoencoded image space. unCLIP [22] conditions generation on CLIP image embeddings. This conditioning interface allows a generative objective to evaluate information carried by a visual representation, providing a route for transferring knowledge from a pre-trained generator to an encoder.

DIVA [29] uses feedback from a frozen difusion model to optimize CLIP representations with images alone, targeting fine-grained visual distinctions while retaining zeroshot capabilities. un<sup>2</sup>CLIP [15] improves CLIP’s visualdetail sensitivity by inverting the unCLIP generator. In both cases, generation provides an additional criterion for learning visual features beyond their similarity to text. Their connection to TMT lies in using a pre-trained generative model as a source of encoder supervision, with the learned representation subsequently serving perception tasks. DIVA also illustrates that visual knowledge acquired during generative pre-training can supply an image-dependent adaptation signal without a newly annotated caption for each training image.

VLV [32] and De-Difusion [30] explore related feedback through language representations. VLV distills difusion knowledge through continuous embeddings in a vision– language–vision auto-encoding path. De-Difusion trains an image-to-text encoder with a frozen text-to-image difusion decoder, encouraging the textual representation to retain visual information. These approaches broaden the role of generative supervision, while difering in whether the information is retained in an image embedding, a continuous language representation, or generated text.

The above studies primarily examine visible-image representations. Infrared adaptation additionally requires generative feedback to remain tied to the scene observed in another sensing modality. TMT couples two constraints on the same pseudo-visible representation: a frozen CLIP image encoder supplies the paired visible-image embedding as a semantic reference, while a frozen difusion model supplies noise-prediction feedback that encourages retention of visual details useful for generation. Their joint use connects generative representation learning to paired cross-modal content alignment, allowing visible-domain supervision to train an infrared encoder for subsequent image–text matching.

## 3. Method

This section describes the framework for adapting CLIP to infrared imagery. We first formalize how infrared inputs afect $\mathrm { C L I P ^ { \circ } s }$ zero-shot image–text matching. The framework then comprises two parts: Text-Derived Modality Transfer and Generative–Perceptual Co-Supervision.

## 3.1. Problem Formulation

Given a pre-trained CLIP model with an image encoder $f _ { \mathrm { i m g } }$ and a text encoder $f _ { \mathrm { t x t } }$ , the task is to adapt the image encoder for infrared–text alignment. For an infrared image ${ \bf I } _ { \mathrm { I R } }$ , its visual representation $\begin{array} { r } { \mathbf { v } _ { \mathrm { I R } } \ = \ f _ { \mathrm { i m g } } ( \mathbf { I } _ { \mathrm { I R } } ) } \end{array}$ should be close to the text prototype that describes its content. For a class-specific prompt, this prototype is $\mathbf { t } _ { c } = f _ { \mathrm { t x t } } ( \langle \mathrm { p r o m p t } _ { c } \rangle )$ , and their alignment can be evaluated through the following normalized similarity distribution:

$$
p ( y = c \mid \mathbf { I } _ { \mathrm { I R } } ) = \frac { \exp ( \sin ( \mathbf { v } _ { \mathrm { I R } } , \mathbf { t } _ { c } ) / \tau ) } { \sum _ { c ^ { \prime } = 1 } ^ { C } \exp ( \sin ( \mathbf { v } _ { \mathrm { I R } } , \mathbf { t } _ { c ^ { \prime } } ) / \tau ) }\tag{1}
$$

where sim(⋅, ⋅) denotes cosine similarity and � is the temperature parameter. This distribution is the inference interface of CLIP rather than the training problem itself: the central objective is to learn an adapted image encoder whose infrared features remain content-consistent with their visible counterparts and are better aligned with text prototypes.

For infrared inputs, the original CLIP image encoder may produce features that are poorly positioned relative to the text-aligned geometry learned from visible images. TMT therefore learns an adapted encoder $f _ { \mathrm { i m g } } ^ { \mathrm { T M T } }$ such that, for an infrared–visible image pair $( \mathbf { I } _ { \mathrm { I R } } , \mathbf { I } _ { \mathrm { v i s } } )$ , the representation $f _ { \mathrm { i m g } } ^ { \mathrm { T M T } } ( \mathbf { I } _ { \mathrm { I R } } )$ preserves the content correspondence with ${ \mathbf { I } } _ { \mathrm { v i s } }$ while moving toward a text-compatible visible-image representation. The adaptation uses paired infrared–visible images and generic modality descriptions, but does not require paired infrared image–text annotations. The resulting encoder is evaluated through zero-shot classification, infrared– text retrieval, and open-vocabulary semantic segmentation.

## 3.2. Framework Overview

Figure 2 summarizes the data flow of TMT. Given a paired infrared–visible sample $( \mathbf { I } _ { \mathrm { I R } } , \mathbf { I } _ { \mathrm { v i s } } )$ , the trainable CLIP image encoder first extracts the infrared feature, which Text-Derived Modality Transfer then shifts along a text-derived modality direction to obtain a pseudo-visible representation. To construct this direction, a frozen CLIP text encoder embeds generic visible–infrared modality descriptions, whose normalized visible-minus-infrared diference provides $\Delta \mathbf { v } ;$ the module then adds �Δ� to the infrared feature to form this representation $\mathbf { v } _ { \mathrm { p s e u d o \ v i s } } .$ Generative–Perceptual Co-Supervision then constrains $\mathbf { v _ { p s e u d o \ v i s } }$ through two complementary paths. First, a frozen CLIP image encoder extracts the feature of the paired visible image and constrains $\mathbf { v _ { p s e u d o \ v i s } }$ through paired semantic alignment with it. Second, the frozen difusion model supplies latent-space noise-prediction feedback conditioned on $\mathbf { v _ { \mathrm { p s e u d o ~ v i s } } } ,$ , with the paired visible-image latent as the difusion input. Their weighted sum updates only the trainable CLIP image encoder; the text encoder, both frozen supervision models, and Δ� remain fixed. At inference time, the adapted image encoder alone produces infrared features for the standard CLIP text-matching pipeline, without the text-derived shift or difusion decoding.

## 3.3. Text-Derived Modality Transfer

Text-Derived Modality Transfer serves as a bridge between infrared representations and pre-trained visible-image models by constructing pseudo-visible features through which visible-image supervision can adapt the infrared encoder. Prior work has shown that text-embedding diferences can steer CLIP-conditioned image generation [22] and that text-described domain concepts can guide visual feature augmentation across visual domains [28]. For infrared adaptation, the modality gap arises mainly from the diferent imaging mechanisms of the two spectra, so the diferences that the transfer must express are largely shared across the images of each modality rather than tied to individual scenes. We therefore derive the transfer direction from language descriptions of visible and infrared imaging properties. This direction guides infrared features toward the visible-image semantic region; the shifted feature, denoted by �<sub>pseudo vis</sub>, is termed a pseudo-visible feature. Paired visible images supply the instance-specific reference that constrains content consistency during the transfer.

To get the text-derived modality direction, we use a fixed pair of generic visible and infrared descriptions, denoted by $( \mathbf { d } ^ { \mathrm { v i s } } , \mathbf { d } ^ { \mathrm { l R } } )$ , generated with Kimi-K2.5 [13] and then manually selected and refined for the infrared-to-visible transfer. One description specifies the source modality and the other target modality, allowing the desired change to be expressed through their semantic diference. This construction separates the description of the imaging modality from the identity of an individual training sample: the text supplies a shared reference, while the image encoder supplies the sample-specific representation. The descriptions therefore require no captions for individual infrared images and are separate from the class-specific prompts used for downstream evaluation.

Table 1  
Modality-level properties underlying the description pair. The source and target descriptions verbalize the modality-dependent rows; the shared structural properties are stated in both descriptions as content to be preserved.
<table><tr><td>Property</td><td>Visible</td><td>Infrared</td></tr><tr><td>Radiation source</td><td>Reflected scene illumination</td><td>Emitted thermal radiation (temperature, emissivity)</td></tr><tr><td>Contrast and color</td><td>Albedo- and shading-based; rich color</td><td>Thermal contrast; weak or absent color</td></tr><tr><td>Texture and noise</td><td>Strong high-frequency detail; near-uniform noise</td><td>Smoother texture; fixed-pattern noise</td></tr><tr><td>Shared structure</td><td colspan="2">Object shape and spatial arrangement (preserved)</td></tr></table>

The two descriptions are designed around modality-level properties, which are shared by the images of a modality rather than tied to any particular scene. Table 1 summarizes the properties that connect the imaging mechanism to the appearance change we seek to express. The two modalities difer in the radiation that forms the image—reflected visible light versus emitted thermal radiation [17]—and consequently in their contrast, color, texture, and noise characteristics, while object shape and spatial arrangement are preserved across spectra. The descriptions verbalize exactly these properties: the modality-dependent rows distinguish the source description from the target description, and the shared structural properties appear in both descriptions as content to be preserved. Because these properties characterize each modality as a whole rather than any individual sample, a single fixed description pair is suficient to specify the transfer for all training data.

We encode the two descriptions with the frozen CLIP text encoder:

$$
\begin{array} { r } { \mathbf { e } ^ { \mathrm { v i s } } = f _ { \mathrm { t x t } } ( \mathbf { d } ^ { \mathrm { v i s } } ) , \qquad \mathbf { e } ^ { \mathrm { I R } } = f _ { \mathrm { t x t } } ( \mathbf { d } ^ { \mathrm { I R } } ) . } \end{array}\tag{2}
$$

Using the same encoder places the descriptions in a common embedding space, while freezing it preserves the text representation learned during CLIP pre-training. The paired construction provides two semantic reference points for the transformation. Their diference encodes the specified source-to-target change, and the subtraction order determines its orientation. We subtract the infrared embedding from the visible embedding and normalize the diference:

$$
\Delta \mathbf { v } = { \frac { \mathbf { e } ^ { \mathrm { { v i s } } } - \mathbf { e } ^ { \mathrm { { I R } } } } { \| \mathbf { e } ^ { \mathrm { { v i s } } } - \mathbf { e } ^ { \mathrm { { I R } } } \| _ { 2 } } } .\tag{3}
$$

Because the pair verbalizes modality-level properties (Table 1) rather than sample-specific content, the resulting direction can be shared by every training sample and every experiment, and its usefulness depends on the two descriptions and their encoding rather than on any per-sample quantity. L2 normalization then separates the orientation of this diference from its magnitude. Consequently, the transfer coeficient controls the length of the added vector without also inheriting the norm of the text diference. The resulting direction is a language-defined reference for adaptation: a fixed, content-free ofset whose role is to move infrared features into the visible domain. It is not intended to remove or rewrite sample semantics—the direction contains no content description by construction, and the additive form leaves the encoder’s output untouched beneath the shift.

For an infrared image ${ \bf I } _ { \mathrm { I R } }$ , the trainable image encoder produces $\begin{array} { r } { \mathbf { v } _ { \mathrm { I R } } \ = \ f _ { \mathrm { i m g } } ( \mathbf { I } _ { \mathrm { I R } } ) } \end{array}$ . We apply the shared modality direction through an additive transformation:

$$
\begin{array} { r } { \mathbf { v } _ { \mathrm { p s e u d o v i s } } = \mathbf { v } _ { \mathrm { I R } } + \boldsymbol { \alpha } \cdot { \boldsymbol { \Delta } } \mathbf { v } , } \end{array}\tag{4}
$$

where � controls the transfer strength. The same ofset is applied to every infrared feature, so sample-specific information continues to enter through $f _ { \mathrm { i m g } } ( { \bf I } _ { \mathrm { I R } } )$ . We compute Δ� once and keep both it and � fixed during training. This provides a constant modality reference while the encoder learns from the paired observations. Since the ofset �Δ� is constant with respect to the encoder parameters, the derivative of $\mathbf { v _ { p s e u d o \ v i s } }$ with respect to ${ \mathbf { v } } _ { \mathrm { I R } }$ is the identity, allowing gradients from the supervision losses to pass directly to the encoder.

The resulting $\mathbf { v _ { p s e u d o \ v i s } }$ serves as the shared input to the Generative–Perceptual Co-Supervision module. Using the paired visible image as a reference, the perceptual branch constrains semantic correspondence, while the generative branch provides feedback on visual detail, as described in the following subsection.

## 3.4. Generative–Perceptual Co-Supervision

To align infrared images with text prototypes more effectively and transfer visible-light knowledge to the infrared domain, we require the pseudo-visible embedding produced by TMT to satisfy two related but distinct requirements. It should preserve the content correspondence between a paired infrared image and its visible counterpart, while also retaining visual details that are weakly represented in the infrared input, such as texture and edges. These requirements are not equivalent in the CLIP representation. Image–text contrastive pre-training gives CLIP strong global semantic alignment, but prior work has reported that this training paradigm can underrepresent fine-grained visual diferences [29, 15]. A supervision signal based only on semantic proximity could therefore encourage content consistency without suficiently constraining the detail information required by the visible-image branch. Conversely, a signal based only on visual generation could favor a plausible visible appearance without ensuring that the generated content remains the same as the paired infrared observation. This motivates two complementary constraints rather than an unsupported claim that either objective is universally insuficient on its own.

To implement these complementary constraints, TMT uses a frozen CLIP image encoder to provide the paired visible-image embedding as a semantic anchor, together with a frozen difusion generator for visual-detail supervision. Following unCLIP [22], this generator synthesizes visible images conditioned on CLIP image embeddings through difusion noise prediction. This learned relationship between the conditioning embedding and image appearance provides the basis for its detail feedback. In TMT, the pseudo-visible embedding conditions the generator’s noise prediction on the paired visible-image latent, allowing the resulting loss to encourage the infrared encoder to retain visual information useful to the generator.

The CLIP branch constrains $\mathbf { v _ { p s e u d o \ v i s } }$ to remain semantically aligned with the paired visible-image embedding, preserving object and scene correspondence. The difusion branch uses noise-prediction feedback to encourage the same representation to retain visual details useful for conditional image generation. Together, the two branches couple semantic alignment with generative detail supervision: the paired semantic reference constrains what content is represented, while the generative objective constrains how that representation supports the visible-image noise-prediction process.

Difusion-based Noise-Prediction Supervision. We use the frozen difusion model to impose this generative constraint. Let ${ \bf x } _ { \mathrm { v i s } }$ denote the clean latent corresponding to the visible image ${ \mathbf { I } } _ { \mathrm { v i s } } .$ . In the standard DDPM objective [9], the image latent is first perturbed with a timestep-dependent Gaussian noise and the difusion model predicts that same noise. For TMT, we do not add forward difusion noise to the clean visible latent: the latent is supplied directly to the frozen difusion model, $\mathbf { v _ { p s e u d o \ v i s } }$ is used as the condition, and an independently sampled Gaussian vector remains the prediction target. This choice separates the latent used as the visual reference from the random target used for noise prediction. Appendix A provides the corresponding gradient derivation and the empirical comparison of the no-noising and DDPM-style strategies, including their training stability. For a paired sample $( \mathbf { I } _ { \mathrm { I R } } , \mathbf { I } _ { \mathrm { v i s } } )$ , with $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , we define:

$$
\mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { ( \mathbf { I } _ { \mathrm { I R } } , \mathbf { I } _ { \mathrm { v i s } } ) , \epsilon , t } \left[ \left\| \epsilon - \epsilon _ { \theta } \left( \mathbf { x } _ { \mathrm { v i s } } , t , \mathbf { v } _ { \mathrm { p s e u d o v i s } } \right) \right\| _ { 2 } ^ { 2 } \right]\tag{5}
$$

where $\epsilon _ { \theta } ( \cdot )$ denotes the frozen difusion model and � is a sampled timestep. ${ \mathcal { L } } _ { \mathrm { d i f f } }$ compares predicted and sampled noise in the difusion latent space; it does not compare a generated RGB image with the paired visible image pixel by pixel. Gradients through the frozen difusion model update the trainable infrared image encoder so that $\mathbf { v _ { p s e u d o \ v i s } }$ carries information useful to the visible-image generator.

Paired Semantic Alignment Supervision. The generative constraint alone does not identify which plausible visible content belongs to the observed infrared instance. We therefore retain the paired visible image as a semantic anchor. The frozen CLIP image encoder $\phi _ { \mathrm { C L I P } } ^ { \mathrm { v i s } }$ maps ${ \mathbf { I } } _ { \mathrm { v i s } }$ to an embedding in the CLIP joint embedding space:

$$
\begin{array} { r } { \mathbf { v } _ { \mathrm { v i s } } = \phi _ { \mathrm { C L I P } } ^ { \mathrm { v i s } } ( \mathbf { I } _ { \mathrm { v i s } } ) . } \end{array}\tag{6}
$$

We then align the pseudo-visible embedding with this paired embedding using the cosine distance:

$$
{ \mathcal { L } } _ { \mathrm { s e m a n t i c } } = 1 - { \frac { \mathbf { v } _ { \mathrm { p s e u d o v i s } } \cdot \mathbf { v } _ { \mathrm { v i s } } } { \| \mathbf { v } _ { \mathrm { p s e u d o v i s } } \| _ { 2 } \cdot \| \mathbf { v } _ { \mathrm { v i s } } \| _ { 2 } } }\tag{7}
$$

This is a paired semantic alignment loss rather than a contrastive objective: it uses the visible counterpart as a positive reference but does not introduce negative samples or class labels. Minimizing it preserves the content correspondence of the infrared–visible pair and limits semantic drift, while the frozen teacher prevents the limited paired training data from redefining the visible CLIP embedding space. Together with the difusion-based constraint, it gives the trainable image encoder a target that is both content-consistent and sensitive to visible-image detail.

The two supervision signals are combined as:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { d i f f } } \cdot { \mathcal { L } } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { s e m a n t i c } } \cdot { \mathcal { L } } _ { \mathrm { s e m a n t i c } }\tag{8}
$$

where $\lambda _ { \mathrm { d i f f } }$ and $\lambda _ { \mathrm { s e m a n t i c } }$ weight the difusion-based noiseprediction supervision and the paired semantic alignment supervision, respectively. The first term encourages �<sub>pseudo vis</sub> to retain information that supports visible-image detail, whereas the second limits changes that would break the content correspondence with ${ \mathbf { I } } _ { \mathrm { v i s } } .$ Because Δ� and both supervisory networks are fixed, the dependence of $\mathbf { v _ { p s e u d c } }$ vis on the trainable parameters is inherited from ${ \mathbf { v } } _ { \mathrm { I R } }$ . Gradients from both terms can therefore pass through the transfer operation and update the same image encoder. The objective does not train a separate transfer module to be retained at test time; it uses that module to expose a visible-oriented training signal to the encoder that will ultimately be deployed.

## 3.5. Training and Inference

Training. Only the trainable CLIP image encoder $f _ { \mathrm { i m g } }$ is optimized through ${ \mathcal { L } } _ { \mathrm { t o t a l } } .$ . The transfer coeficient � and the text-derived direction Δ� remain fixed, as do the text encoder, the visible-image CLIP teacher, and the difusion model. These frozen components still participate in the forward computation and transmit gradients through their outputs, but their parameters do not absorb the infrared-domain correction. Keeping the reference models fixed also prevents the supervision targets from moving with the adapted encoder, making the optimization target stable throughout training.

Inference. Given an infrared image without a visible counterpart, the TMT-adapted image encoder directly produces its image embedding. This embedding is compared with the text embeddings of target-class prompts through the standard CLIP similarity interface. The generic modality descriptions used to construct the training-time direction are distinct from these dataset-specific evaluation prompts. The text-derived shift, the visible-image teacher, and the difusion model are not invoked at inference. The deployed system consequently has the same input and matching interface as CLIP, while its image encoder has been adapted using the two complementary training signals described above.

## 4. Experiments

We evaluate TMT from the perspective of the tasks for which an adapted CLIP encoder is intended to be used. The experiments cover zero-shot classification, infrared– text retrieval, and open-vocabulary semantic segmentation. These tasks test progressively diferent properties of the adapted representation: category transfer, image–language alignment, and spatially resolved prediction. Section 5 then examines the components and mechanisms responsible for the observed behavior.

## 4.1. Experimental Setup

Training Data. TMT is trained on approximately 70K aligned infrared–visible image pairs from IR500K [23]. The adaptation uses only the paired images. In particular, no paired infrared image–text annotation, evaluation class label, or evaluation caption is used to train TMT. These conditions define the supervision setting studied in this work. This setting is practically motivated: infrared–text corpora require dedicated caption collection or synthesis at scale, as illustrated by the construction of IR-TD [1] and by VLM-based caption synthesis for thermal benchmarks [20], whereas paired infrared–visible images are directly available from dual-sensor benchmarks and require no per-image language annotation [12, 16, 14].

Evaluation Datasets. For infrared–text retrieval, we use IR-TD [1]. For this study, we randomly divide it into 80% training and 20% reserved data; from the reserved portion, we randomly sample 1,000 images with their annotations as the test set and use the remainder as a validation set. All reported retrieval results are evaluated on this test set. The training split is used only to fine-tune the CLIP-FT reference, whereas TMT, OpenAI CLIP, $\mathrm { u n } ^ { 2 } \mathrm { C L I P } ,$ and DIVA do not use IR-TD training pairs. We use Kimi-K2.5 to rewrite the source descriptions into concise infrared-oriented captions, and evaluate every method with the same test captions. LanguageBind is evaluated as an externally trained infrared– language reference. For zero-shot classification, we evaluate eight infrared benchmarks: M3FD [16], SMOD [2], FLIRv1 [3], FLIRv2 [27], LLVIP [12], JU-VNT-T [5], MFNet [7], and PST900 [26]. The primary cross-source group comprises SMOD, JU-VNT-T, MFNet, and PST900, which are absent from the IR500K adaptation subset. M3FD, FLIRv1, FLIRv2, and LLVIP share source-dataset families with IR500K and are reported as supplementary benchmarks. Evaluation images are excluded from TMT adaptation, including for the source-overlapping benchmarks.

Detection-style datasets are converted into classification benchmarks by cropping the annotated infrared bounding boxes and using the crop labels as class labels. For MFNet and PST900, labeled infrared regions are used to form classification examples from the available annotations. TMT does not use the evaluation labels or class-specific text during adaptation.

For dense prediction, we evaluate open-vocabulary semantic segmentation on CART [14] and SemanticRT [10]. Neither dataset is included in the IR500K training subset, so these experiments test transfer across datasets rather than direct reuse of segmentation annotations. The four sourcedisjoint classification benchmarks provide the primary zeroshot evaluation, while the four source-overlapping benchmarks provide supplementary results. The original images and, where applicable, captions and annotations in the training and evaluation datasets listed above were released by their respective dataset creators and should be obtained from those original sources.

Prompts and Metrics. For classification and segmentation, Kimi-K2.5 is used to generate three distinct groups of class prompts for each dataset. These prompts pair datasetspecific class names with infrared appearance cues, such as thermal contrast and silhouette, that a generic visible-photo template would leave unspecified. They are evaluation inputs only: the generic visible–infrared descriptions used to construct TMT’s training-time modality direction do not contain the evaluation class names. Every method is evaluated with the same prompt groups, and we report each group’s result and the average to show sensitivity to wording. This design reduces dependence on a single prompt set, but it does not establish that generated prompts outperform fixed templates or eliminate all prompt bias. Infrared–text retrieval uses the infrared-oriented IR-TD captions described above rather than these generated class prompts. The three prompt-group JSON files for each classification and segmentation dataset and the rewritten IR-TD retrieval captions are evaluation materials produced in this study.

Classification is reported as Top-1 accuracy. Retrieval is reported as Recall@K in both the infrared-to-text and text-to-infrared directions. Segmentation is reported as mean Intersection-over-Union (mIoU). The main text reports the averaged results; the prompt-level classification and segmentation values and the detailed ablation values are provided in Sections S1.1–S1.3 of the Supplementary Material.

Baselines. We compare TMT with seven baselines. OpenAI CLIP [21] is the original ViT-L/14 model without infrared adaptation. LanguageBind [35] provides an infrared branch trained within a language-centered multimodal framework. un<sup>2</sup>CLIP [15] uses an unCLIP generator to recover visual information while retaining the CLIP embedding space. DIVA [29] uses difusion feedback to improve CLIP visual representations from image-only data. ImageBind (oficial) [6] denotes the released model, whose thermal branch was trained on LLVIP image–thermal pairs.

ImageBind (trained) denotes our paired CLIP control: starting from OpenAI CLIP, we train its infrared image encoder to align directly with visible-image embeddings using the same IR500K infrared–visible training pairs as TMT. The two methods difer in their adaptation objectives; this control follows ImageBind’s image-paired alignment principle but does not fine-tune the oficial ImageBind checkpoint. CLIP-FT fine-tunes OpenAI CLIP with LoRA on the IR-TD training split and is included as an infrared–language supervised reference where the result is available.

LanguageBind and CLIP-FT are marked with † in the tables because they use infrared–language supervision that is unavailable to TMT. They are therefore useful reference points, especially for retrieval, but are not strictly matched competitors in terms of supervision. TMT and ImageBind (trained) start from OpenAI CLIP, use the same IR500K infrared–visible training pairs, and require no paired infrared–text annotations; comparing them isolates the efect of the adaptation method. OpenAI CLIP, un<sup>2</sup>CLIP, and DIVA are additional references without paired infrared–text supervision in this comparison. The released ImageBind model is an external reference with a diferent backbone and thermal training source. We exclude IRGPT [1] from the numerical comparison because its generative questionanswering interface difers from the CLIP-style image–text alignment interface evaluated here.

We also do not numerically compare with the concurrent T-CLIP [20]: it is evaluated on diferent benchmarks with its own synthesized captions, and no public implementation was available at the time of submission. Its use of synthesized infrared–text supervision would also place it with the † reference methods rather than the matched group.

Implementation Details. All experiments use OpenAI CLIP ViT-L/14 as the base image encoder. The generative branch uses Stable Difusion 2.1 unCLIP small with the Karlo v1-alpha prior. Only the CLIP image encoder is trainable; the VAE, difusion model, text encoder, prior model, and visible-image CLIP teacher are frozen. Training images are resized to 256 × 256, and CLIP inputs are resized to $2 2 4 \times 2 2 4$ . We optimize the model with SGD for 1000 steps using a learning rate of $1 \times 1 0 ^ { - 4 }$ , momentum 0.9, and zero weight decay. Training uses eight RTX 4090 GPUs with per-device batch size 8, gradient accumulation 10, and efective batch size 640. BF16 and TF32 acceleration are enabled, and the difusion noise-prediction branch uses two inner difusion steps during training. The loss weights are $\lambda _ { \mathrm { d i f f } } = \lambda _ { \mathrm { s e m a n t i c } } = 1$ in all reported experiments.

## 4.2. Zero-Shot Classification

Table 2: Zero-shot Top-1 accuracy (%) on four sourcedisjoint benchmarks, averaged over three shared prompt groups. ImageBind (trained) is the ImageBind-style paired CLIP control, distinct from the oficial checkpoint. † marks infrared–language-supervised references. Best values under our protocol are bold.
<table><tr><td>Method</td><td>SMOD</td><td>JU-VNT-T</td><td>MFNet</td><td>PST900</td><td>Avg.</td></tr><tr><td>LanguageBind†</td><td>56.30</td><td>47.90</td><td>38.59</td><td>36.98</td><td>44.94</td></tr><tr><td>un2CLIP</td><td>57.87</td><td>32.63</td><td>35.66</td><td>34.19</td><td>40.09</td></tr><tr><td>DIVA</td><td>60.07</td><td>43.73</td><td>34.08</td><td>32.27</td><td>42.54</td></tr><tr><td>ImageBind (official)</td><td>25.73</td><td>4.17</td><td>6.07</td><td>27.04</td><td>15.75</td></tr><tr><td>ImageBind (trained)</td><td>63.00</td><td>36.28</td><td>36.28</td><td>38.50</td><td>43.52</td></tr><tr><td>CLIP-FT†</td><td>62.90</td><td>51.56</td><td>31.89</td><td>38.24</td><td>46.15</td></tr><tr><td>TMT</td><td>63.00</td><td>50.17</td><td>42.04</td><td>40.49</td><td>48.92</td></tr></table>

Table 2 prioritizes the four source-disjoint benchmarks. TMT averages 48.92% on SMOD, JU-VNT-T, MFNet, and PST900, 2.78 points above the strongest reference average (CLIP-FT, 46.15%) and 3.98 points above LanguageBind (44.94%); diferences are computed from unrounded values. It leads outright on two of these four datasets: MFNet (42.04%) and PST900 (40.49%). On MFNet and PST900, the gains over the strongest alternative are 3.45 points over LanguageBind and 1.99 points over ImageBind (trained), respectively. On SMOD, TMT and ImageBind (trained) both report 63.00% at two-decimal precision; CLIP-FT leads on JU-VNT-T (51.56% versus 50.17%). The four sourceoverlapping benchmarks are shown in Table 3; over all eight, TMT averages 60.40% and leads outright on five, with the displayed tie on SMOD. The original LanguageBind paper reports 82.90% on FLIRv1, 48.00% on FLIRv2, and 87.20% on LLVIP; these values are contextual and are excluded from our matched-protocol averages.

Table 3: Supplementary Top-1 accuracy (%) on four benchmarks sharing source families with IR500K. ImageBind (trained) is the ImageBind-style paired CLIP control. ‡ gives originally reported LanguageBind values under a diferent protocol and is excluded from the eight-dataset average. Best values under our protocol are bold.
<table><tr><td>Method</td><td>M3FD</td><td>FLIRv1</td><td>FLIRv2</td><td>LLVIP</td><td>All-8 avg.</td></tr><tr><td>LanguageBind†</td><td>67.10</td><td>77.27</td><td>45.20</td><td>83.23</td><td>56.57</td></tr><tr><td>LanguageBind</td><td></td><td>82.90</td><td>48.00</td><td>87.20</td><td></td></tr><tr><td>un2CLIP</td><td>54.87</td><td>80.20</td><td>35.70</td><td>79.83</td><td>51.37</td></tr><tr><td>DIVA</td><td>61.77</td><td>71.77</td><td>30.33</td><td>85.20</td><td>52.40</td></tr><tr><td>ImageBind (official)</td><td>5.74</td><td>6.27</td><td>1.10</td><td>72.86</td><td>18.62</td></tr><tr><td>ImageBind (trained)</td><td>57.54</td><td>73.94</td><td>31.22</td><td>72.09</td><td>51.11</td></tr><tr><td>CLIP-FT†</td><td>57.46</td><td>67.05</td><td>30.01</td><td>87.16</td><td>53.28</td></tr><tr><td>TMT</td><td>71.53</td><td>84.87</td><td>42.57</td><td>88.57</td><td>60.40</td></tr></table>

The primary group spans road scenes (SMOD and MFNet), indoor objects (JU-VNT-T), and the subterranean robotic imagery of PST900. MFNet and PST900 were originally introduced for pixel-labeled scene understanding [7, 26]; here their labeled regions are used for infrared classification. This mixture probes whether category transfer persists outside the IR500K source families. TMT’s lead on both new datasets is consistent with improved transfer, although their moderate absolute accuracies (42.04% and

Infrared–text retrieval on IR-TD. Recall@K is reported in both retrieval directions. ImageBind (oficial) uses the released checkpoint; ImageBind (trained) is our paired CLIP alignment control. The upper block contains methods without infrared–text supervision during adaptation; the lower block contains supervised references. Methods marked with † use infrared–language supervision during pre-training or fine-tuning. Best results among the methods without infrared–text supervision are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">IR → Text</td><td colspan="4">Text → IR</td></tr><tr><td>R@1</td><td>R05</td><td>R@10</td><td>R@20</td><td>R@1</td><td>R05</td><td>R@10</td><td>R@20</td></tr><tr><td colspan="9">Without infrared-text supervision</td></tr><tr><td>OpenAI CLIP [21]</td><td>5.00</td><td>11.72</td><td>16.72</td><td>25.47</td><td>10.47</td><td>20.31</td><td>30.31</td><td>43.44</td></tr><tr><td>un2CLIP [15]</td><td>3.59</td><td>10.00</td><td>15.00</td><td>22.03</td><td>10.31</td><td>21.41</td><td>29.22</td><td>39.38</td></tr><tr><td>DIVA [29]</td><td>4.84</td><td>11.41</td><td>16.09</td><td>25.00</td><td>9.06</td><td>19.22</td><td>29.84</td><td>42.81</td></tr><tr><td>ImageBind (official) [6]</td><td>1.09</td><td>4.06</td><td>6.25</td><td>9.53</td><td>1.41</td><td>4.37</td><td>7.66</td><td>11.25</td></tr><tr><td>ImageBind (trained)</td><td>4.39</td><td>13.57</td><td>19.24</td><td>25.93</td><td>6.35</td><td>15.87</td><td>22.17</td><td>31.01</td></tr><tr><td>TMT</td><td>6.72</td><td>16.41</td><td>23.59</td><td>32.34</td><td>10.94</td><td>24.22</td><td>32.81</td><td>44.22</td></tr><tr><td colspan="9">Infrared-language-supervised references</td></tr><tr><td>LanguageBind† [35]</td><td>14.69</td><td>31.41</td><td>41.09</td><td>51.56</td><td>13.59</td><td>30.31</td><td>39.06</td><td>50.00</td></tr><tr><td>CLIP-FT†</td><td>18.12</td><td>41.72</td><td>53.28</td><td>68.28</td><td>19.37</td><td>42.03</td><td>56.41</td><td>72.03</td></tr></table>

40.49%) show that region-level recognition remains dificult. Prompt-level results are given in Section S1.1 of the Supplementary Material.

TMT does not lead outright on every benchmark. On SMOD, TMT and ImageBind (trained) both reach 63.00% at the reported precision; the trained control scores higher with Prompts 1 and 2, whereas TMT scores higher with Prompt 3. On JU-VNT-T, CLIP-FT has the advantage of infrared–text training on IR-TD, of which JU-VNT-T is a source, so this comparison has a diferent supervision and source relationship. On FLIRv2, LanguageBind reaches 45.20% under our protocol and 48.00% in its original report, while TMT reaches 42.57%. The finer vehicle and roadside taxonomy may demand distinctions that a global modality shift does not fully supply. These exceptions bound the claim to average category transfer under the stated protocols.

ImageBind (oficial) reaches 72.86% on LLVIP but 1.10– 27.04% on the other seven classification datasets. This pattern is consistent with its thermal branch being trained on LLVIP image–thermal pairs [6]; it does not establish that training-source coverage alone explains the gap, because the oficial model also difers in backbone and preprocessing. ImageBind (trained) scores higher than the oficial checkpoint on seven of eight datasets, but slightly lower on LLVIP (72.09% versus 72.86%). Under our shared prompt protocol, TMT exceeds the oficial checkpoint on LLVIP by 15.71 points and the trained control by 16.48 points. LLVIP is a training source family for the oficial checkpoint and for the shared IR500K subset used by TMT and ImageBind (trained), so this comparison is not a test of source-disjoint transfer.

## 4.3. Infrared–Text Retrieval

Retrieval tests whether an infrared image embedding enters the appropriate linguistic neighborhood when the text encoder remains fixed. The table is therefore read in two groups. Within the group without infrared–text supervision, TMT is the strongest method in every reported Recall@K column. Relative to OpenAI CLIP, TMT raises IR-to-Text R@10 from 16.72% to 23.59% and Text-to-IR R@5 from 20.31% to 24.22%; at R@20, the gains are 6.87 and 0.78 percentage points in the two directions, respectively. ImageBind (trained) reaches 25.93% and 31.01% at R@20, compared with 9.53% and 11.25% for the oficial checkpoint; TMT reaches 32.34% and 44.22%. TMT and ImageBind (trained) use the same IR500K training pairs but diferent adaptation methods. The oficial checkpoint difers from both in architecture and thermal training source, so its scores serve as an external comparison under the shared evaluation protocol.

The lower block provides a supervised reference rather than a comparison under matched supervision conditions. LanguageBind and CLIP-FT use detailed, caption-level infrared–text alignment, which directly supplies the type of correspondence measured by retrieval. CLIP-FT additionally fine-tunes on the IR-TD training split and thus benefits from similarity between its training and test distributions, including caption style and dataset-specific language conventions. This diference is visible at R@20: CLIP-FT reaches 68.28% for IR-to-Text retrieval, compared with 32.34% for TMT, while LanguageBind reaches 51.56%. The gap should not be interpreted as evidence that TMT fails to improve image–text compatibility; it quantifies the additional information provided by instance-level infrared captions and in-domain supervised adaptation. TMT is intended for the complementary setting in which aligned infrared–visible images are available but infrared text annotation is absent or costly. Concurrent work that synthesizes thermal captions likewise reports retrieval gains on its own benchmarks [20], consistent with the supervised-reference pattern described above.

## 4.4. Zero-Shot Semantic Segmentation

All segmentation results are produced with NACLIP [8], a training-free open-vocabulary segmentation method that adapts a CLIP image encoder to dense prediction by enforcing neighbourhood-aware patch localization in its selfattention. Because NACLIP requires no task-specific training, the same configuration is applied to every compared encoder, with each dataset’s class prompts serving as the text inputs; diferences in the resulting masks can therefore be attributed to the image representation rather than to a segmentation head.

Zero-shot open-vocabulary semantic segmentation mIoU (%) on infrared images. Both datasets are absent from TMT training. ImageBind (trained) denotes the paired CLIP alignment control. Methods marked with † use infrared–language supervision.
<table><tr><td>Method</td><td>CART</td><td>SemanticRT</td></tr><tr><td>OpenAI CLIP [21]</td><td>16.06</td><td>15.16</td></tr><tr><td>LanguageBind†[35]</td><td>18.02</td><td>16.87</td></tr><tr><td>un²CLIP [15]</td><td>9.22</td><td>17.29</td></tr><tr><td>DIVA [29]</td><td>16.40</td><td>15.49</td></tr><tr><td>ImageBind (official) [6]</td><td>1.59</td><td>5.63</td></tr><tr><td>ImageBind (trained)</td><td>16.94</td><td>15.80</td></tr><tr><td>CLIP-FT†</td><td>16.56</td><td>16.19</td></tr><tr><td>TMT</td><td>18.23</td><td>17.38</td></tr></table>

Table 5 evaluates whether the adaptation remains useful when the output must be spatially resolved. TMT obtains the best mIoU on both CART and SemanticRT, which are absent from the IR500K training subset. ImageBind (oficial) reaches 1.59% and 5.63%, respectively; Image-Bind (trained) reaches 16.94% and 15.80%. The result indicates that the adapted image representation transfers beyond cropped-object classification to two cross-dataset denseprediction settings. The improvement is nevertheless modest in absolute terms. Under the same NACLIP pipeline, visibleimage inputs reach only 25.66% on CART and 20.60% on SemanticRT. The infrared results should therefore be read as relative evidence that TMT preserves useful spatial information, not as a substitute for supervised segmentation.

The segmentation results also expose a boundary of the current method. Classification and retrieval mainly require global identity and image–text compatibility, whereas segmentation depends on boundary placement, part structure, and small-object localization. Thermal difusion, sensor blur, saturation, and weak texture can remove precisely these local cues. TMT improves the placement of infrared representations in the CLIP embedding space, but the paired semantic alignment and difusion noise-prediction objectives do not directly supervise segmentation boundaries. The component ablation in Section 5 and the linear-probe analysis in Appendix A isolate the contribution of the difusion branch; the remaining spatial limitation motivates future work on infrared imaging mechanisms and image structure.

## 5. Analysis and Discussion

Section 4 established TMT’s downstream performance. We now make further examinations of the proposed method TMT, including the ablation studies, the source of the modality transfer direction together with qualitative evidence of its efect, the sensitivity to the transfer weight, and feature geometry. These experiments separate the efect of pairedimage supervision from text-derived modality transfer and identify the tasks on which the choices matter.

Component ablation averaged across tasks. Each row switches the text-derived shift �Δ� and the two supervision branches $\mathcal { L } _ { \mathrm { s e m } } \ ( \mathcal { L } _ { \mathrm { s e m a n t i c } } )$ and $\mathcal { L } _ { \mathrm { d i f f } }$ on $( \checkmark )$ or of (×); the first row is the unadapted OpenAI CLIP encoder without any further training. Retrieval averages both directions on IR-TD; primary classification averages the four source-disjoint datasets (SMOD, JU-VNT-T, MFNet, and PST900); all-dataset classification averages all eight datasets; segmentation averages CART and SemanticRT. All values are percentages.
<table><tr><td>Shift</td><td> $\mathcal { L } _ { \mathrm { { s e m } } }$ </td><td> $\mathcal { L } _ { \mathrm { d i f f } }$ </td><td>R@1 ↑</td><td>R@20 ↑</td><td>Primary cls. ↑</td><td>All cls. ↑</td><td>Seg. mloU ↑</td></tr><tr><td>×</td><td>×</td><td>×</td><td>7.73</td><td>34.45</td><td>42.48</td><td>52.33</td><td>15.61</td></tr><tr><td>√</td><td>√</td><td>×</td><td>8.83</td><td>37.81</td><td>48.23</td><td>59.78</td><td>17.69</td></tr><tr><td>√</td><td>×</td><td>√</td><td>7.58</td><td>34.45</td><td>44.55</td><td>54.98</td><td>16.73</td></tr><tr><td>×</td><td>√</td><td>√</td><td>7.88</td><td>34.14</td><td>46.18</td><td>55.79</td><td>16.27</td></tr><tr><td>√</td><td>√</td><td>√</td><td>8.83</td><td>38.28</td><td>48.92</td><td>60.40</td><td>17.81</td></tr></table>

## 5.1. Ablation Studies

Table 6 separates the contribution of paired semantic alignment and difusion-based noise prediction. Removing both branches reduces all-dataset classification from 60.40% to 52.33% and average segmentation from 17.81% to 15.61%. Semantic supervision alone raises all-dataset classification by 7.45 points and gives the same mean R@1 as the full model. It directly anchors the transferred infrared feature to the paired visible-image teacher, which is consistent with its stronger efect on category recognition.

The difusion branch has a diferent profile. Used alone, it does not improve mean retrieval R@1 over the nosupervision variant, but it raises average segmentation mIoU from 15.61% to 16.73%. The result is consistent with a branch that supplies visual-detail feedback rather than a direct category target. The full model combines the two roles: semantic supervision provides the strongest categorylevel correction, while difusion feedback contributes local visual information and increases the full-model R@20 and segmentation scores. The fourth row of Table 6 removes the shift while keeping both supervision branches. Relative to the full model, this lowers primary classification by 2.74 points and bidirectional R@20 by 4.14 points, isolating the contribution of the text-derived shift under fixed supervision.

## 5.2. Analysis of Text-Derived Modality Transfer

This subsection examines the Text-Derived Modality Transfer module in two complementary parts. Section 5.2.1 asks where the transfer direction itself comes from and compares control variants that replace its source or its content with the full model under the same downstream evaluation. Section 5.2.2 then provides qualitative evidence that the transfer produces visible-oriented representations, through t-SNE embeddings and generated pseudo-visible images.

## 5.2.1. Controls on the Transfer Direction

To test whether the benefit of the shift comes from its textual origin, we compare TMT with controls that replace the text-derived direction under the same downstream evaluation protocol. The first, ImageBind (trained), removes the shift and the generative branch and instead directly aligns infrared and visible CLIP features; it is an ImageBind-style training control initialized from OpenAI CLIP and trained on the same IR500K infrared–visible pairs as TMT, rather than on the released ImageBind checkpoint. The second, image-mean transfer, keeps the shift mechanism but replaces the text-derived direction with the diference between the mean visible and mean infrared image embeddings of the paired training set. These controls test, respectively, direct paired-image alignment without any explicit transfer direction and a direction estimated from training images rather than from text. A further group of controls probes the content of the direction rather than its source: three random unit vectors used as the shift direction with diferent seeds, a minimal modality-name pair (“an image of visible” versus “an image of infrared”), and the reversed direction −Δ�. These ask whether an arbitrary shift, a bare naming of the two modalities, or the opposite orientation can substitute for the intended infrared-to-visible direction. The oficial ImageBind results are reported separately in Section 4.

Transfer-direction controls. ImageBind (trained) is the ImageBind-style CLIP alignment control, not the oficial checkpoint. The variant without the shift is reported with the component ablation in Table 6. Random direction averages three seeds; the modality-name pair uses “an image of visible” versus “an image of infrared” in place of the modality descriptions; the reversed direction negates Δ�. Retrieval is bidirectional mean R@K; classification is Top-1 accuracy; segmentation is mean mIoU. All values are percentages. Primary classification averages SMOD, JU-VNT-T, MFNet, and PST900.
<table><tr><td>Variant</td><td>R@1↑</td><td>R@20 ↑</td><td>Primary cls. ↑</td><td>All cls. ↑</td><td>Seg. mloU ↑</td></tr><tr><td>ImageBind (trained)</td><td>5.37</td><td>28.47</td><td>43.52</td><td>51.11</td><td>16.37</td></tr><tr><td>Image-mean transfer</td><td>9.14</td><td>35.31</td><td>46.40</td><td>56.38</td><td>16.24</td></tr><tr><td>Random direction (3 seeds)</td><td>8.90</td><td>36.85</td><td>46.06</td><td>55.53</td><td>16.27</td></tr><tr><td>Text pair (modality names)</td><td>8.52</td><td>37.26</td><td>46.57</td><td>56.51</td><td>16.12</td></tr><tr><td>Reversed direction</td><td>7.26</td><td>33.13</td><td>40.22</td><td>51.12</td><td>15.28</td></tr><tr><td>TMT</td><td>8.83</td><td>38.28</td><td>48.92</td><td>60.40</td><td>17.81</td></tr></table>

The released ImageBind checkpoint provides an external comparison rather than another transfer-direction control. Its means are 15.75% on the four primary classification datasets, 18.62% across all eight, 10.39% for bidirectional retrieval R@20, and 3.61% segmentation mIoU. ImageBind (trained) reaches 43.52%, 51.11%, 28.47%, and 16.37%, respectively, while TMT reaches 48.92%, 60.40%, 38.28%, and 17.81%. The oficial checkpoint difers from ImageBind (trained) in backbone and thermal training source, so their gap is descriptive. ImageBind (trained) shares TMT’s CLIP starting point and IR500K training pairs, making it the pairedimage-alignment comparison for TMT’s diferent adaptation method.

As analyzed in Section 5.1, removing the shift lowers primary classification by 2.74 points and bidirectional R@20 by 4.14 points. A paired-image mean direction recovers part of this loss, but trails the text direction by 2.52 primary-classification points and 2.97 retrieval points. ImageBind (trained) is weaker on average, showing that direct paired-image alignment alone does not reproduce the combination of transfer and generative–perceptual supervision.

These are comparisons of complete training variants, not an isolated estimate of the causal efect of text alone: Image-Bind (trained) also changes the supervision architecture.

The image-mean control in Table 7 shows that pairedimage diferences provide a useful transfer signal for classification and retrieval, but are less efective than the textderived direction in this setting. One possible explanation is that the image-derived direction depends on both the sampled scenes and the original encoder’s representations of the two modalities. Pairing establishes scene correspondence at the input level, but does not guarantee that the feature difference isolates modality alone. Averaging across pairs can reduce instance-specific variation, while systematic efects associated with the training distribution or the encoder’s infrared representations may remain. Generic modality descriptions instead specify the imaging properties to be contrasted without deriving the direction from those sampled image features. Paired visible images remain the content reference for supervision in both variants.

A final group of controls asks whether the content of the direction matters at all. Averaged over three seeds, random directions reach 46.06% primary classification, 55.53% all-dataset classification, 36.85% bidirectional R@20, and 16.27% segmentation mIoU—close to the no-shift variant on classification and segmentation and slightly higher on retrieval, but below TMT on most metrics. The minimal modality-name pair does better than the random directions: it improves over them on the two classification averages and retrieval R@20 (46.57% primary, 56.51% all-dataset, 37.26% R@20), showing that even a bare naming of the two modalities carries useful modality-level signal; it nevertheless trails TMT’s designed description pair on every metric. The reversed direction is the weakest control and falls below the no-shift variant on all five metrics (40.22% primary, 33.13% R@20, 15.28% mIoU). Arbitrary vectors and the opposite orientation therefore fail to reproduce the improvement, and bare modality names recover only part of it: the gain depends on the modality-level imaging properties verbalized in the designed pair rather than on the mere presence of an additive or text-derived shift.

## 5.2.2. Qualitative Evidence of the Transfer

The controls above isolate the contribution of the direction’s source to downstream performance. As a qualitative complement, we next inspect whether images generated from the transferred features exhibit a visible-oriented shift. We compare the original infrared images, the pseudo-visible images generated from their transferred features, and their paired visible images. For each selected source subset, all three image types are re-encoded by the same frozen CLIP ViT-L/14 encoder for visualization, so encoder choice is held fixed within the plot. Because the generator and this second encoding step intervene, this experiment provides indirect, qualitative evidence rather than a direct measurement of the training-time pseudo-visible feature vectors.

Figure 3 shows that embeddings of the generated pseudovisible images generally lie farther from the infrared-image distribution and closer to the paired visible-image distribution. This pattern is consistent with the intended visibleoriented transfer, but it does not directly establish the displacement of $\mathbf { v _ { p s e u d o \ v i s } } \colon$ image generation and re-encoding intervene between that feature and the plotted points. The figure is therefore an indirect, qualitative check; t-SNE can also distort global distances.

Table 8  
Transfer-weight sensitivity. Primary classification averages four source-disjoint datasets; retrieval averages both R@20 directions; segmentation averages CART and SemanticRT mIoU. All values are percentages.
<table><tr><td>α</td><td>Primary cls. ↑</td><td>Retrieval R@20 ↑</td><td> ${ \mathsf { S e g . } }$  mloU ↑</td></tr><tr><td>0.0</td><td>46.18</td><td>34.14</td><td>16.27</td></tr><tr><td>0.1</td><td>48.49</td><td>37.97</td><td>17.96</td></tr><tr><td>0.2</td><td>48.15</td><td>36.80</td><td>17.92</td></tr><tr><td>0.3</td><td>48.92</td><td>38.28</td><td>17.81</td></tr><tr><td>0.4</td><td>47.61</td><td>37.42</td><td>17.76</td></tr><tr><td>0.5</td><td>45.41</td><td>34.45</td><td>17.23</td></tr></table>

Figure 4 shows that the pseudo-visible images retain the object identity and coarse spatial structure of the infrared inputs while moving their appearance toward the paired visible images. These examples complement the indirect t-SNE visualization in Figure 3: the generated images show visible-oriented appearance while retaining recognizable scene content. Pixelwise identity is not the target of this operation. DDPM-style models sample a learned reverse process rather than deterministically invert an input, and unCLIP-style decoders map a semantic image embedding to a distribution of plausible visible images [9, 22, 25]. Several visible realizations can therefore satisfy the same infraredconditioned semantic content. The relevant visual evidence is recognizable object identity and coarse structure together with a visible-oriented change in generated appearance, rather than exact RGB reconstruction. These observations are consistent with the intended transfer, while the task and ablation results provide its quantitative evaluation.

## 5.3. Sensitivity to the Transfer Weight

We vary the fixed coeficient � in Eq. 4 from 0.0 to 0.5 in steps of 0.1, keeping the direction-construction procedure and downstream evaluation protocol fixed. The zero setting corresponds to removing the shift (the fourth row of Table 6). This sweep tests whether the benefit depends on a narrow transfer strength or persists over neighboring values. For each setting, we evaluate the four primary classification datasets, IR-TD retrieval, and both segmentation datasets; the reported values average three prompt groups where applicable. Per-dataset and per-metric values of the sweep are reported in Section S1.6 of the Supplementary Material.

Within $\alpha \in \begin{array} { r } { \begin{array} { r l } { [ 0 . 1 , 0 . 4 ] . } \end{array} } \end{array}$ , all three metrics vary only modestly: primary classification stays between 47.61% and 48.92%, retrieval R@20 between 36.80% and 38.28%, and segmentation between 17.76% and 17.96%, indicating that the benefit does not rely on a finely tuned transfer strength. Primary classification and bidirectional retrieval R@20 peak at $\alpha = 0 . 3$ (48.92% and 38.28%), while segmentation peaks at $\alpha ~ = ~ 0 . 1$ (17.96%) and varies by at most 0.20 points across this range. Degradation appears mainly at the two ends: removing the shift lowers primary classification to 46.18%, and moving to 0.5 reduces it to 45.41%, below the zero-shift result, consistent with an overly large modality displacement. We therefore fix $\alpha = 0 . 3$ as a single setting that is near-optimal for the two global matching tasks while remaining close to the best segmentation values.

![](images/21da823d3c18f68d3d99aaea03c25543c3da77e120dc92b19043d12ebadc994a.jpg)

![](images/bd22dcb173a57ea4f2217821d09591d42355592de02eff6a78dfe5004b87b3b1.jpg)  
IR input

![](images/64f2d70ae6bc7aed1996d01c26cb3da1461802336f24689a7d280eb479c5434e.jpg)  
� = 0.0  
� = 0.1

![](images/c3db493c6410fd5cfe5b9f819fe6a02274c2eb6f0777081fa2afcefe00a7671a.jpg)

![](images/9d4cc63feff475b9fc578549bacf4ab70e67db1cdfd2b6ee359ab884881172b2.jpg)  
� = 0.3

![](images/14fc6814a2eb4a9ea2abd92da797a4098b2e2b0704d7d6b58714672666de1ac4.jpg)  
� = 0.4

� = 0.2  
![](images/a46f760742ee720800d6c41e1955c412de165217ecf94b6f4d6ce3293876cc11.jpg)  
� = 0.5

![](images/38b82ab2f3f6bc6e049425c62b2550d483b7e893e565cae2335d14047da1f82b.jpg)  
Paired visible  
Figure 5: An IR500K/KAIST example: infrared input, pseudo-visible outputs of the difusion model at six transfer weights, and the paired visible image.

Figure 5 makes the sweep visually inspectable on one paired scene. Note that the difusion model samples a plausible image from an embedding [22], so the visual diferences are qualitative and should not be read as pixelwise fidelity rankings.

Overall, the sweep shows that � has a broad usable range rather than a single sensitive point; within this range, moderate transfer supports global matching, while dense prediction favors a smaller shift.

## 5.4. Feature Geometry Analysis

To determine whether the classification gains reflect a change in feature geometry rather than a narrow improvement on a particular benchmark, we measure inter-class margins and within-class compactness on the L2-normalized embeddings produced by each model for the evaluation images of every dataset. The margin ratio is defined as

$$
\mathbf { M } \mathbf { R } = \frac { \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \operatorname* { m i n } _ { c ^ { \prime } \neq c } \| \mu _ { c } - \mu _ { c ^ { \prime } } \| _ { 2 } } { \frac { 1 } { C } \sum _ { c = 1 } ^ { C } s _ { c } } ,\tag{9}
$$

and the Fisher ratio is defined as

$$
\mathrm { F R } = \frac { \sum _ { c = 1 } ^ { C } N _ { c } \| \boldsymbol { \mu _ { c } } - \boldsymbol { \mu } \| _ { 2 } ^ { 2 } } { \sum _ { c = 1 } ^ { C } \sum _ { i : y _ { i } = c } \| \boldsymbol { x _ { i } } - \boldsymbol { \mu _ { c } } \| _ { 2 } ^ { 2 } } .\tag{10}
$$

Both metrics are computed in the high-dimensional embedding space; t-SNE projections of the same embeddings (Figure 6) serve only as a qualitative visualization.

Across eight datasets, mean MR rises from 0.250 to 0.292 (16.94% relative, computed from unrounded values), and mean FR rises from 0.283 to 0.326 (15.26% relative).

Feature separability of infrared object embeddings. Both metrics are computed on the high-dimensional L2-normalized CLIP features before t-SNE projection. Δ reports the relative change of TMT over OpenAI CLIP, computed from unrounded values.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">Margin ratio ↑</td><td colspan="3">Fisher ratio ↑</td></tr><tr><td>CLIP</td><td>TMT</td><td>Δ</td><td>CLIP</td><td>TMT</td><td>Δ</td></tr><tr><td>M3FD</td><td>0.102</td><td>0.116</td><td>+13.52%</td><td>0.132</td><td>0.175</td><td>+32.86%</td></tr><tr><td>SMOD</td><td>0.166</td><td>0.195</td><td>+17.60%</td><td>0.124</td><td>0.168</td><td>+35.95%</td></tr><tr><td>FLIRv1</td><td>0.157</td><td>0.175</td><td>+11.11%</td><td>0.117</td><td>0.152</td><td>+30.08%</td></tr><tr><td>FLIRv2</td><td>0.106</td><td>0.112</td><td>+5.99%</td><td>0.149</td><td>0.180</td><td>+21.08%</td></tr><tr><td>LLVIP</td><td>0.526</td><td>0.775</td><td>+47.49%</td><td>0.226</td><td>0.349</td><td>+54.27%</td></tr><tr><td>JU-VNT-T</td><td>0.654</td><td>0.625</td><td>-4.41%</td><td>1.166</td><td>1.124</td><td>-3.56%</td></tr><tr><td>MFNet PST900</td><td>0.157 0.130</td><td>0.171 0.167</td><td>+9.04% +28.19%</td><td>0.243</td><td>0.300 0.161</td><td>+23.17%</td></tr><tr><td></td><td></td><td></td><td></td><td>0.108</td><td></td><td>+49.87%</td></tr><tr><td>Avg.</td><td>0.250</td><td>0.292</td><td>+16.94%</td><td>0.283</td><td>0.326</td><td>+15.26%</td></tr></table>

Both new source-disjoint datasets improve on both measures: MFNet gains 9.04% in MR and 23.17% in FR, while PST900 gains 28.19% and 49.87%. This supports transfer of the feature-geometry change beyond the IR500K source families. The exception is JU-VNT-T, whose MR and FR fall by 4.41% and 3.56% from already high baseline values. Its classification accuracy still improves over the unadapted encoder, so a lower aggregate separability ratio does not imply uniformly worse category matching. For qualitative analysis, the plots in Figure 6 provide a complementary visual check of these gains on MFNet and PST900 as well as the earlier benchmarks. Their two-dimensional layout is not a quantitative measure of class separation: the high-dimensional MR and FR values are the relevant evidence, and the JU-VNT-T discrepancy shows why both views are reported. These metrics summarize aggregate geometric properties and cannot fully identify which individual classes remain confusable; nevertheless, their improvement on seven of the eight datasets provides converging evidence that the classification gains reflect a genuine change in feature geometry.

## 6. Conclusion

We present TMT, a text-derived modality-transfer framework that adapts CLIP to infrared imagery with paired semantic alignment and difusion noise-prediction supervision. Training uses paired infrared–visible images and generic modality descriptions without instance-level infrared– text annotations. Text-Derived Modality Transfer derives a reusable infrared-to-visible modality direction from generic modality descriptions in CLIP’s joint embedding space and applies it to the infrared image representation. Generative– Perceptual Co-Supervision then constrains the transferred representation from two complementary perspectives: a frozen CLIP image encoder preserves correspondence with the paired visible image, while a frozen difusion model provides feedback related to visible-image detail.

At inference, the TMT-adapted image encoder and frozen CLIP text encoder perform image–text matching; the modality direction and auxiliary supervision models are used only during training. Across four source-disjoint classification benchmarks, TMT reaches 48.92% mean zero-shot accuracy, leads outright on two, ties on SMOD at two-decimal precision, and exceeds LanguageBind by 3.98 points under our protocol. The four source-overlapping classification benchmarks provide supplementary evaluation. On IR-TD retrieval, TMT is the strongest among the methods that do not use infrared–text supervision during adaptation. The segmentation results show gains on both evaluated datasets. Ablations and feature-geometry analyzes provide complementary evidence for improved semantic organization, and a linear-probe analysis shows that the difusion branch increases the linear readability of infrared texture detail while suppressing global brightness variation.

Limitations and Future Work. We evaluate TMT only with a CLIP image encoder. Future work will investigate thermodynamic priors for infrared imaging, test whether the adaptation strategy extends to other visual foundation models, and examine how synthesized infrared captions [20] can complement the present caption-free supervision.

## A. Analysis of the Difusion Noise-Prediction Supervision

## A.1. Motivation

The empirical comparison below is conducted solely with the generative branch (i.e., the difusion noise-prediction loss ${ \mathcal { L } } _ { \mathrm { d i f f } } )$ in order to isolate the efect of the training strategy. The paired semantic alignment branch $( \mathcal { L } _ { \mathrm { s e m a n t i c } } )$ is omitted so that any observed diference in stability can be attributed unambiguously to the difusion-supervision design.

For difusion noise-prediction supervision, two strategies exist for using the frozen difusion model as a supervisory signal. Strategy A (noise-free) feeds the clean visible image latent directly into the difusion model, which predicts noise conditioned on $\mathbf { v } _ { \mathrm { p s e u d o \ v i s } } ;$ the prediction is compared against a sampled Gaussian noise target. Strategy B (DDPM-style) follows the standard forward-noising pipeline: the visible image is first corrupted with noise at timestep � to obtain $\mathbf { x } _ { t } ,$ then the difusion model predicts the noise from $\mathbf { x } _ { t }$ . Empirically, Strategy A improves CLIP’s infrared-image performance, whereas Strategy B causes training instability and eventual collapse. Figure A.1 compares the training curves.

![](images/11d03ee924d855bd1eb98dbbb56e078e97a0bc7f321c1422e05f118a3d1124fa.jpg)

![](images/02b3d1a727b0ba817e4b12eb2b4f0da4587632da81244aaabdb7c61f74c6f79e.jpg)  
(left) Strategy A (noise-free)  
(right) Strategy B (DDPM-style)

Figure A.1: Training curves of the two difusion supervision strategies.

## A.2. Theoretical Analysis

Let �(�, �, �) denote the frozen difusion model, where � is the difusion input, � is the timestep, and � is the condition vector (here, $\mathbf { c } = \mathbf { v } _ { \mathrm { p s e u d o \ v i s } } )$ . The standard difusion training objective is:

$$
\mathcal { L } = \mathbb { E } \left[ \| f ( \mathbf { x } , t , \mathbf { c } ) - \varepsilon \| _ { 2 } ^ { 2 } \right] ,\tag{A.1}
$$

where $\varepsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ is the Gaussian noise target. The gradient w.r.t. the condition vector is:

$$
\begin{array} { r l } & { \nabla _ { \mathbf { c } } \mathcal { L } = \nabla _ { \mathbf { c } } \mathbb { E } \left[ \left\| f ( \mathbf { x } , t , \mathbf { c } ) - \pmb { \varepsilon } \right\| _ { 2 } ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } \left[ \nabla _ { \mathbf { c } } \left\| f ( \mathbf { x } , t , \mathbf { c } ) - \pmb { \varepsilon } \right\| _ { 2 } ^ { 2 } \right] } \\ & { \qquad = \mathbb { E } \left[ 2 \left( f ( \mathbf { x } , t , \mathbf { c } ) - \pmb { \varepsilon } \right) ^ { \top } \nabla _ { \mathbf { c } } f ( \mathbf { x } , t , \mathbf { c } ) \right] . } \end{array}\tag{A.2}
$$

Strategy A (noise-free). Since the input � is computed from the clean visible image without adding noise, � and � are statistically independent. For fixed (�, �, �), the term $\nabla _ { \mathbf { c } } f ( \mathbf { x } , t , \mathbf { c } )$ is determined by the clean latent, timestep, and condition vector, and is therefore independent of the sampled noise �. This independence allows the gradient term to be taken outside the expectation over �. Writing the factorization with an explicit conditional expectation gives:

$$
\begin{array} { r l } & { \nabla _ { \mathbf { c } } \mathcal { L } = \mathbb { E } _ { \mathbf { x } , t } \Big [ \mathbb { E } _ { \boldsymbol \varepsilon } \big [ 2 \left( f ( \mathbf { x } , t , \mathbf { c } ) - \boldsymbol \varepsilon \right) ^ { \top } \nabla _ { \mathbf { c } } f ( \mathbf { x } , t , \mathbf { c } ) \Big | \mathbf { x } , t \big ] \Big ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { x } , t } \big [ 2 \left( f ( \mathbf { x } , t , \mathbf { c } ) ^ { \top } - \mathbb { E } _ { \boldsymbol \varepsilon } [ \boldsymbol \varepsilon ] ^ { \top } \right) \nabla _ { \mathbf { c } } f ( \mathbf { x } , t , \mathbf { c } ) \big ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { x } , t } \big [ 2 f ( \mathbf { x } , t , \mathbf { c } ) ^ { \top } \nabla _ { \mathbf { c } } f ( \mathbf { x } , t , \mathbf { c } ) \big ] , } \end{array}\tag{A.3}
$$

where the last step uses $\mathbb { E } [ \varepsilon ] = \mathbf { 0 } .$ . Thus, Strategy A drives the expected difusion output toward zero (the mean of the noise distribution). This derivation does not make Strategy A equivalent to the full DDPM objective. It only shows that removing the dependence between the difusion input and the sampled noise eliminates the noise-coupled gradient term, yielding a lower-variance supervisory signal in expectation.

Strategy B (DDPM-style). When the input is the noised latent $\mathbf { x } _ { t } , \mathbf { x } _ { t }$ and � are correlated (both depend on the same noise sample), so the expectation does not factorize. The gradient contains noise-coupled terms that can increase the variance and magnitude of backpropagated updates, which is consistent with the observed training instability.

## A.3. Empirical Validation

## A.3.1. Gradient Statistics

To verify the theoretical prediction that Strategy A yields smaller gradients, we visualize the gradient distributions of the trainable CLIP image encoder under both strategies. As shown in Figure A.2, Strategy A produces gradients with substantially smaller magnitudes in this diagnostic, supporting the interpretation that the noise-free approach is a milder form of supervision.

![](images/e008c0e746278c95c3eca61f947112bf727438d69c3275cdaadf166a7baa61de.jpg)

![](images/7876b7b387e528ac67f3548a122f423f9c3b1110f15fa0df6741e5945392e4c6.jpg)  
(left) Strategy A (noise-free) (right) Strategy B (DDPM-style)

Figure A.2: Gradient distributions in the trainable CLIP image encoder.

## A.3.2. Learning Rate Sensitivity

We further investigate whether reducing the learning rate can stabilize Strategy B. Figure A.3 shows the training behavior under learning rates $1 0 ^ { - 4 } , 1 0 ^ { - 5 }$ , and $1 0 ^ { - 6 }$ . While lowering the learning rate does suppress the collapse phenomenon, the training signal becomes too weak to produce meaningful improvements. This trade-of suggests that the instability of Strategy B is closely tied to its larger gradient magnitudes rather than only to an extraneous hyperparameter issue.

![](images/cfd7a6708d00082ad0f4f65d5c3fa2bb8297daeebefb13d000d8d20b874bff42.jpg)

![](images/d660952170f8925ec31dacada6064c6e8b096971c2ef67b849026469e01e3d7d.jpg)  
10<sup>−5</sup>

![](images/387fc5a7e7d94411cc2b8ed3291ede50904d3e2540601f46a1f2abc9bc668f18.jpg)  
10<sup>−6</sup>  
Figure A.3: Training curves of Strategy B under diferent learning rates.

Based on the theoretical analysis and empirical evidence above, we adopt Strategy A (noise-free direct noise prediction) as the difusion noise-prediction supervision in our framework.

## A.4. Linear-Probe Analysis of the Difusion Branch

The analyses above justify how the difusion supervision is applied; a separate question is what the branch changes in the representation. Concretely, we ask whether training with ${ \mathcal { L } } _ { \mathrm { d i f f } }$ makes texture information more linearly readable from the CLIP embedding, and whether such a change is specific to detail or merely reflects a generic sensitivity to any pixel-level variation. To separate the two cases, we probe two controlled perturbation families: a texture family that scales high-frequency energy and therefore acts as a detail metric, and a gain family that only changes global brightness and contrast without moving edges or altering texture structure and therefore acts as a non-detail control. Improved texture readability together with unchanged or reduced gain readability indicates detail-specific enhancement, whereas a uniform increase across both families would indicate a generic gain that cannot be attributed to detail.

Perturbation families. For the texture family, a singlechannel image � is decomposed into a low-pass component $\mathrm { ~ \bf ~ L ~ } =$ GaussianBlur $( \mathbf { x } , \sigma { = } 3 . 0 )$ and its high-frequency residual. The residual is scaled by a retention factor �, and the result is mean-aligned to the original image so that brightness changes cannot masquerade as texture changes:

$$
\mathbf { x } ^ { \prime } = \mathbf { L } + r \cdot ( \mathbf { x - L } ) , \qquad \mathbf { x } ^ { \prime } \gets \mathbf { x } ^ { \prime } + \left( \overline { { \mathbf { x } } } - \overline { { \mathbf { x } ^ { \prime } } } \right) .\tag{A.4}
$$

We use six retention levels from 200% (high-frequency enhancement) to 0% (low-pass only), with 100% being the unmodified reference. For the gain family, $\mathbf { x } ^ { \prime } \ = \ \operatorname { c l i p } ( g \ \cdot$ $\mathbf { x } + b \cdot R , 0 , 2 5 5 )$ , where � is the dynamic range of �; we use five multiplicative levels $g \in \{ 0 . 8 , 0 . 9 , 1 . 0 , 1 . 1 , 1 . 2 \}$ and two additive levels $b = \pm 5 \%$ of the dynamic range. Neither family moves edges, changes object shapes, or alters scene geometry.

Probe. Each perturbed image is encoded by the frozen encoder under test, giving an L2-normalized 768-dimensional CLIP embedding that is kept fixed during probe fitting. The probe is a ridge regressor from the embedding to the perturbation target: the degradation level for the texture family, and the signed brightness change $( g \mathrm { ~ - ~ } 1 \ \mathrm { o r } \ b )$ for the gain family, whose seven levels do not form a monotone sequence. We use an outer 5-fold GroupKFold grouped by anchor image, so that all perturbation levels of one image fall in the same fold and every test fold contains only unseen images; within each outer training fold, the regularization strength is selected by an inner 3-fold GroupKFold over 13 candidates in $[ 1 0 ^ { - 3 } , 1 0 ^ { 3 } ]$ . We report the coeficient of determination $R ^ { 2 }$ on the held-out folds, where $R ^ { 2 } \ = \ 1$ denotes perfect linear readout and $R ^ { 2 } = 0$ denotes predicting the target mean.

Data and models. We compare the unadapted OpenAI CLIP encoder with the difusion-branch variant trained with ${ \mathcal { L } } _ { \mathrm { d i f f } }$ only, on 48 infrared anchors: 8 from LLVIP, 20 from M3FD, and 20 from JU-VNT-T. No text encoder, class prompt, or category-matching score is used in this analysis.

Results. Table A.1 reports the probe $R ^ { 2 }$ for both families. Texture readability improves under the difusion branch $( 0 . 6 5 4  0 . 6 7 7 )$ , while the readability of the signed global brightness change drops substantially $( 0 . 2 6 3 \  \ 0 . 0 6 7 )$ This pattern matches the detail-specific case of the reading framework above: rather than producing a uniform sensitivity gain, the difusion branch increases the linear accessibility of high-frequency texture detail and simultaneously suppresses content-irrelevant global brightness variation in the representation.

<table><tr><td>Probe target  $( R ^ { 2 } )$ </td><td>OpenAI CLIP</td><td>diffusion branch</td><td>Δ</td></tr><tr><td>Texture level</td><td>0.654</td><td>0.677</td><td>+0.023</td></tr><tr><td>Gain (signed brightness)</td><td>0.263</td><td>0.067</td><td>-0.196</td></tr></table>

Table A.1: Linear-probe $R ^ { 2 }$ for reading the perturbation target from frozen CLIP embeddings. Higher texture $R ^ { 2 }$ with lower gain $R ^ { 2 }$ indicates detail-specific enhancement rather than a generic sensitivity gain.

![](images/d91b5b6923dfb4cab8a524c914a3f5335bf23b6e049ea16ff4e23b80074569a2.jpg)  
Figure 1: Illustration of the core idea. In order to guide the knowledge transfer between visible (rich) and infrared (lack) modalities and increase the performance of infrared–text alignment, a text-derived infrared-to-visible direction moves the infrared feature toward the visible-image region of CLIP’s embedding space, and paired visible images provide a semantic reference, and a frozen difusion model provides noiseprediction feedback during adaptation.

![](images/f2104401bfa13b3c7ee5f4cb28945c17b5930289bc034d0da452b5e4be03c90b.jpg)  
Figure 2: Overview of TMT. A paired infrared–visible sample (purple zone) is processed by a trainable CLIP image encoder for the infrared image and a frozen CLIP image encoder for the visible counterpart. Generic visible–infrared modality descriptions are encoded by the frozen CLIP text encoder; their normalized diference produces the modality direction $\Delta \mathbf { v } ,$ which is applied by Text-Derived Modality Transfer to the infrared feature and yields a pseudo-visible representation (red zone). Generative–Perceptual Co-Supervision (yellow and green zones) constrains this representation through two complementary paths: the frozen CLIP image encoder provides paired semantic alignment with the visible feature, while the frozen difusion model uses the visible image and the pseudo-visible representation to compute a latent-space noise-prediction loss against a sampled standard-Gaussian target. Only the trainable infrared image encoder (blue zone) is updated during training. (best viewed in color)

t-SNE of CLIP ViT-L/14 image embeddings  
![](images/8530714971b4740917ddcd379b9fd66a8aa44948c566b8ed6d7240d968d88ee6.jpg)  
t-SNE dimension 1  
Figure 3: t-SNE visualization of CLIP ViT-L/14 embeddings for original infrared images, generated pseudo-visible images, and original visible images. Matched subsets are sampled from the source datasets for readability, and all images are encoded with the same CLIP backbone. Gray lines connect matched triplets, while colors indicate image source. The embeddings of generated pseudo-visible images appear closer to the visibleimage distribution than those of the original infrared images; this is an indirect visualization of the transfer.

Original infrared  
Pseudo-visible  
Original visible  
![](images/d5dca4fae10f6b6e72612bf8b219d586c421875eadaf98450da0778501e66f02.jpg)  
Figure 4: IR500K examples of the original infrared image, the pseudo-visible image, and the paired visible image from left to right. The pseudo-visible result preserves object identity and coarse structure, but it is not expected to match the visible image pixel by pixel.

![](images/5ae1d1d20841dcaa4f30dd2770f2d6ad2be9e03eab72cf0863e0de44e413bfdb.jpg)  
Figure 6: t-SNE visualization of infrared object features. The upper block shows M3FD, FLIRv1, FLIRv2, and LLVIP; the lower block shows JU-VNT-T, MFNet, PST900, and SMOD. Separability values in Table 9 are computed in the original embedding space.

## CRediT authorship contribution statement

Ruofan Wang: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft. Puzuo Wang: Investigation, Validation, Data curation. Xin Chen: Investigation, Validation. Zibo Yu: Formal analysis, Visualization. Jungang Yang: Methodology, Writing – review & editing. Yi Hou: Supervision, Project administration, Writing – review & editing.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used KIMI Code in order to assist with language editing, LaTeX preparation, and manuscript organization. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. Kimi-K2.5 was additionally used to draft the generic modality descriptions and the evaluation class prompts; all such materials were reviewed and finalized by the authors.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The evaluation prompt files are publicly available at https://github.com/wanginsten/TMT. The training and evaluation code will be released in the same repository.

## Funding

This work is supported by the National Natural Science Foundation of China (No. 42501493).

## References

[1] Cao, Z., Zhang, J., Zhang, R., 2025. IRGPT: Understanding realworld infrared image with bi-cross-modal curriculum on large-scale benchmark, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 166–176.

[2] Chen, Z., Qian, Y., Yang, X., Wang, C., Yang, M., 2025. AMFD: Distillation via adaptive multimodal fusion for multispectral pedestrian detection. IEEE Transactions on Multimedia 27, 8298–8310. doi:10.1109/TMM.2025.3604937.

[3] FLIR Systems, 2018. FLIR releases starter thermal imaging dataset for machine learning advanced driver assistance development. https: //www.flir.com/news-center/press-releases/flir-releases-starter -thermal-imaging-dataset-for-machine-learning-advanced-driver-a ssistance-development/.

[4] Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y., 2024. CLIP-Adapter: Better vision-language models with feature adapters. International Journal of Computer Vision 132, 581–595. doi:10.1007/s11263-023-01891-x.

[5] Ghosh, S., Das, N., Sarkar, P., Nasipuri, M., 2021. JU-VNT: A multi-spectral dataset of indoor object recognition using visible, nearinfrared and thermal spectrum. Multimedia Tools and Applications 80, 17807–17826. doi:10.1007/s11042-020-10302-z.

[6] Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I., 2023. Imagebind: One embedding space to bind them all, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15180–15190.

[7] Ha, Q., Watanabe, K., Karasawa, T., Ushiku, Y., Harada, T., 2017. MFNet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes, in: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5108–5115. doi:10 .1109/IROS.2017.8206396.

[8] Hajimiri, S., Ben Ayed, I., Dolz, J., 2025. Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5061–5071.

[9] Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models, in: Advances in Neural Information Processing Systems, pp. 6840–6851.

[10] Ji, W., Li, J., Bian, C., Zhang, Z., Cheng, L., 2023. SemanticRT: A large-scale dataset and method for robust semantic segmentation in multispectral images, in: Proceedings of the ACM International Conference on Multimedia, pp. 3307–3316.

[11] Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T., 2021a. Scaling up visual and vision-language representation learning with noisy text supervision, in: International Conference on Machine Learning, pp. 4904–4916.

[12] Jia, X., Zhu, C., Li, M., Tang, W., Zhou, W., 2021b. LLVIP: A visibleinfrared paired dataset for low-light vision, in: Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, pp. 3496–3504. doi:10.1109/ICCVW54120.2021.00389.

[13] Kimi Team, 2026. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276 .

[14] Lee, C., Anderson, M., Ranganathan, N., Zuo, X., Do, K., Gkioxari, G., Chung, S.J., 2024. CART: Caltech aerial RGB-thermal dataset in the wild, in: Proceedings of the European Conference on Computer Vision, pp. 236–256. doi:10.1007/978-3-031-73036-8\_14.

[15] Li, Y., Zhao, J., Chang, H., Hou, R., Shan, S., Chen, X., 2025. un<sup>2</sup>clip: Improving clip’s visual detail capturing ability via inverting unclip. arXiv preprint arXiv:2505.24517 .

[16] Liu, J., Fan, X., Huang, Z., Wu, G., Liu, R., Zhong, W., Luo, Z., 2022. Target-aware dual adversarial learning and a multi-scenario multimodality benchmark to fuse infrared and visible for object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5802–5811.

[17] Mao, F., Mei, J., Lu, S., Liu, F., Chen, L., Zhao, F., Hu, Y., 2026. PID: Physics-informed difusion model for infrared image generation. Pattern Recognition 169, 111816. doi:10.1016/j.patcog.2025.111816.

[18] Mikolov, T., Chen, K., Corrado, G., Dean, J., 2013a. Eficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 .

[19] Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S., Dean, J., 2013b. Distributed representations of words and phrases and their compositionality, in: Advances in Neural Information Processing Systems.

[20] Qazi, T., Maheshwari, A., Mukherjee, P., Lall, B., 2026. T-CLIP: Enabling thermal perception for contrastive language-image pretraining. arXiv preprint arXiv:2606.00673 .

[21] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: International Conference on Machine Learning, pp. 8748–8763.

[22] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M., 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 .

[23] Ran, L., Wang, L., Wang, G., Wang, P., Zhang, Y., 2025. Difv2ir: Visible-to-infrared difusion model via vision-language understanding. arXiv preprint arXiv:2503.19012 .

[24] Ranasinghe, Y., Schenck, E., Yellin, F., Hu, S., Funk, C., Patel, V.M., 2026. Thermal-Det: Language-guided cross-modal distillation for open-vocabulary thermal object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 34628–34637. URL: https://openaccess.thecvf.com/content/CV PR2026/html/Ranasinghe\_Thermal-Det\_Language-Guided\_Cross-Modal\_D istillation\_for\_Open-Vocabulary\_Thermal\_Object\_Detection\_CVPR\_20 26\_paper.html.

[25] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695.

[26] Shivakumar, S.S., Rodrigues, N., Zhou, A., Miller, I.D., Kumar, V., Taylor, C.J., 2020. PST900: RGB-thermal calibration, dataset and segmentation network, in: 2020 IEEE International Conference on Robotics and Automation, pp. 9441–9447. doi:10.1109/ICRA40945.20 20.9196831.

[27] Teledyne FLIR, 2022. Teledyne FLIR releases free expanded starter thermal dataset for ADAS and autonomous vehicle testing. https: //oem.flir.com/en-150/about/news/expanded-teledyne-flir-starter -thermal-dataset-for-adas-and-autonomous-vehicle-testing/.

[28] Vidit, V., Engilberge, M., Salzmann, M., 2023. CLIP the gap: A single domain generalization approach for object detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3219–3229. doi:10.1109/CVPR52729.2023.00314.

[29] Wang, W., Sun, Q., Zhang, F., Tang, Y., Liu, J., Wang, X., 2025. Difusion feedback helps CLIP see better, in: International Conference on Learning Representations.

[30] Wei, C., Liu, C., Qiao, S., Zhang, Z., Yuille, A., Yu, J., 2024. Dedifusion makes text a strong cross-modal interface, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13492–13503.

[31] Xie, Y., Zhang, Z., Hou, Y., Wang, P., Zhang, H., 2026. Modalityaware dynamic fusion for weakly aligned RGB-T tiny object detection. IEEE Signal Processing Letters 33, 1023–1027. doi:10.1109/LSP.20 26.3659814.

[32] Zhang, T., Li, Y., Chou, Y.C., Chen, J., Yuille, A., Wei, C., Xiao, J., 2025. Vision-language-vision auto-encoder: Scalable knowledge distillation from difusion models. arXiv preprint arXiv:2507.07104

[33] Zhou, K., Yang, J., Loy, C.C., Liu, Z., 2022a. Conditional prompt learning for vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16816–16825.

[34] Zhou, K., Yang, J., Loy, C.C., Liu, Z., 2022b. Learning to prompt for vision-language models. International Journal of Computer Vision 130, 2337–2348. doi:10.1007/s11263-022-01653-1.

[35] Zhu, B., Lin, B., Ning, M., et al., 2024. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment, in: International Conference on Learning Representations.