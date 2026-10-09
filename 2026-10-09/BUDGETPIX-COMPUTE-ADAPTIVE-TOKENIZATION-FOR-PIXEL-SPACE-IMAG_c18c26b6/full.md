# BUDGETPIX: COMPUTE-ADAPTIVE TOKENIZATION FOR PIXEL-SPACE IMAGE DIFFUSION

Ozgur Kara<sup>1,†</sup> Yujia Chen<sup>2</sup> Daniel Watson<sup>2</sup> David Forsyth<sup>1</sup> James Matthew Rehg<sup>1</sup> Wen-Sheng Chu<sup>2,‡</sup> Du Tran<sup>2,‡</sup> <sup>1</sup>University of Illinois Urbana-Champaign <sup>2</sup>Google

BudgetPix (w/ JiT-L/32) on Class Conditional 5122  
![](images/ade5e110b914994a61f4f52a25212ddd46826b1cd01ab2192f89ab9e6f1028d6.jpg)

BudgetPix (w/ MiniT2I-L/16) on Text-to-Image 512²  
![](images/80be24b51480c8a27808d78fe53bddeb7ae6cc3459163f7b7897cc52cab3d535.jpg)

BudgetPix (w/ PixelDiT-T2I) on Text-to-Image 10242  
![](images/2e89bc5a5c111d81c0c867da512c90c63fe995d1a25ee214215f75c7c3c276f2.jpg)  
Figure 1: Test-time compute adjustment. BudgetPix dynamically scales inference costs, achieving a strong quality-efficiency tradeoff across JiT (top), MiniT2I (bottom left), PixelDiT (bottom right), 3 popular architectures for pixel-space image diffusion. Percentages show the fraction of tokens used.

## ABSTRACT

Most image generation models rely on uniform tokenization, allocating the exact same computational budget to equally-sized image patches. This static paradigm cannot adapt to different resource constraints at inference time, and yields suboptimal quality-cost tradeoff by devoting the same effort to both plain backgrounds and intricate details. We propose BudgetPix, an adaptive tokenization framework that dynamically allocates compute based on visual complexity and spatial layout, enabling flexible computational budgeting at inference time. BudgetPix comprises three key components: (1) an adaptive encoder that maps a fixed-size image to a variable-length token sequence using an entropy-guided quadtree alongside a multi-scale patch embedder; (2) a scale-aware decoder reconstructs fixed-resolution images from multi-scale token sets; and (3) a flexible training and sampling schedule that enables pixel-space denoisers to operate across variable token counts. BudgetPix seamlessly integrates with existing pixel-space diffusion architectures, enabling a single checkpoint to be operated at a wide range of compute budgets. Evaluated on text-to-image generation, BudgetPix matches the fidelity of MiniT2I-L at $5 1 2 ^ { 2 }$ and PixelDiT at $1 0 2 4 ^ { 2 }$ using just 25% of the original compute budget. In class-conditional using a MeanFlow backbone, BudgetPix requires merely 60% of the full compute budget to produce images with near-zero quality degradation, observing a marginal 0.8-point increase in FID. Comprehensive assessments by human and VLM judges confirm that BudgetPix establishes a significantly improved quality-efficiency tradeoff over prior budget-adaptive baselines. More details are available at our project page: https://karaozgur.com/BudgetPix

## 1 INTRODUCTION

Diffusion Transformers (DiTs) (Peebles & Xie, 2023) have established a new standard for image generation. While most architectures operate in the latent space of a pretrained autoencoder (Rombach et al., 2022), emerging pixel-space DiTs (Li & He, 2026; Yu et al., 2026) process raw pixels directly, bypassing the reconstruction ceiling and eliminating the reliance on a separate tokenizer. However, this architectural purity introduces a severe computational bottleneck: the uniform tokenization. Because self-attention scales quadratically with the number of tokens, pixel-space models incur massive computational costs. Furthermore, because this grid resolution is hardcoded during training, the inference cost becomes rigidly fixed. A deployed model cannot scale down to accommodate resource-constrained edge devices, nor can it dynamically adjust to varying user latency requirements, without undergoing entirely separate retraining.

This static uniform tokenization is fundamentally misaligned with how visual data is structured. Information density varies significantly within images: a cloudless sky offers minimal visual entropy, while heavily textured objects demand the capture of high-frequency details. Forcing a uniform grid to blindly spend identical compute across both regions creates severe computational waste. As shown in Figure 2a, standard MiniT2I (Wang et al., 2026b) wastes 1,024 tokens rendering a simple flag against an open sky, whereas our content-adaptive framework reconstructs the scene perfectly with just 307 tokens (Figure 2b). Even for complex, edge-to-edge textures like a jackfruit–which strictly requires 256 tokens in standard JiT (Li & He, 2026) (Figure 2c)–BudgetPix matches the baseline visual quality while cutting the required token budget by 50% (Figure 2d).

MiniT2I-L: uniform  
![](images/307c31d54115bed2da074d44c1ec922c70f78d490168cfe85c1cdddea9357af1.jpg)  
(a) 1024 tokens

BudgetPix: adaptive  
![](images/7d020976e1d80ed4edbfd0e3e139bc344ba04674bd2cefc7ab9d82428d6972f5.jpg)  
(b) 307 tokens

JiT-L: uniform  
![](images/c752984d7a2cb74347fd3dd0a62828690b59a68c50b2ca80e349064cf7675cb2.jpg)  
(c) 256 tokens

BudgetPix: adaptive  
![](images/6ded64839be578f983875c74e808697fd79d9366f85847a02487aec87f35aa4f.jpg)  
(d) 127 tokens  
Figure 2: Uniform vs. content-adaptive tokenization. For a 512 × 512 image, uniform tokenization requires 1,024 tokens with 16 × 16 patches (a) and 256 tokens with 32 × 32 patches (c). BudgetPix adapts to visual content, requiring only 307 (b) and 127 (d) tokens. Baselines in (a) and (c) represent full-budget MiniT2I-L and JiT-L; BudgetPix in (b) and (d) operates at 30% and 50% budgets, respectively.

Content-adaptive tokenization is well established in discriminative vision models via token merging (Bolya et al., 2023), token pruning (Rao et al., 2021), or dynamic multi-scale representations (Beyer et al., 2023; Dehghani et al., 2023; Kim et al., 2025; Choudhury et al., 2026). However, extending this paradigm to generative models introduces a fundamental challenge. While discriminative models dynamically allocate tokens over an already-provided input image to predict a class label (Choudhury et al., 2026), generative denoisers must dynamically allocate compute over an image that is actively being generated. This creates a chicken-and-egg dilemma: the denoiser must gauge the local spatial complexity of the image before the image content itself has fully materialized.

In this work, we propose BudgetPix, a framework that adapts any pretrained pixel-space diffusion transformer into a compute-adaptive image generator. BudgetPix introduces three novel components. First, an entropy-driven encoder maps images into variable-length sequences of multiscale tokens. Second, A complementary decoder reconstructs these variable-length token sequences into predetermined, fixed-resolution images. Finally, a training and inference strategy, coupled with content-layout augmentation, enables standard checkpoints–originally trained on fixed token budgets–to become dynamically adjustable during inference. Our extensive experiments (Section 4.3) demonstrate that BudgetPix can be integrated seamlessly with established pixel-space architectures–including JiT (Li & He, 2026), MiniT2I (Wang et al., 2026b), PixelDiT (Yu et al., 2026), and MeanFlow (Geng et al., 2026; Lu et al., 2026)–and provide a significantly-better quality efficiency tradeoff (see plots in Section 4.3).

We evaluate BudgetPix on class-conditional and text-to-image pixel-space diffusion models from 256<sup>2</sup> to 1024<sup>2</sup> resolution. Under strict low-compute budgets where prior budget-adaptive methods undergo catastrophic collapse, BudgetPix degrades gracefully, preserving high perceptual quality. For class-conditional generation, BudgetPix synthesizes coherent, high-quality images using as little as 10% of the original compute budget (Figure 1, top 3 rows)–a regime where competing methods fail entirely (cf. Figure 6). This robustness mirrors phenomena observed in lossy image compression, where visual data can be heavily down-sampled without significant perceptual degradation.

Quantitatively, for text-to-image generation, BudgetPix matches the fidelity of full-compute baselines while utilizing only 25% of the original token budget, achieving GenEval scores of 0.874 (vs. 0.882) on MiniT2I-L at 512<sup>2</sup>, and 0.725 (vs. 0.721) on PixelDiT at 1024<sup>2</sup>. For class-conditional generation, integrating BudgetPix into a MeanFlow (Lu et al., 2026) backbone allows the model to operate at merely 60% of the original compute budget with negligible degradation in visual quality, observing a minimal FID increase of only 0.81 points (from 2.65 to 3.46). Finally, comprehensive evaluations by both human and Vision-Language Model (VLM) judges confirm that BudgetPix achieves a better quality-efficiency tradeoff compared to existing budget-adaptive methods.

Our primary contributions are:

• We propose BudgetPix, a novel compute-adaptive tokenization framework for pixel-space image diffusion, designed to integrate smoothly into diverse pixel-based diffusion architectures such as JiT, MiniT2I, PixelDiT, and MeanFlow.

• BudgetPix introduces three novel components: (1) an entropy-based encoder mapping images into variable-length sequences of multi-scale tokens; (2) a specialized decoder reconstructing fixed-resolution images from these tokens; and (3) a flexible training and sampling schedule supporting denoisers to operate across continuously variable token counts.

• Comprehensive experiments show that BudgetPix achieves an effective quality-efficiency tradeoff compared with alternative adaptive denoising methods.

## 2 RELATED WORK

Pixel-space diffusion models. For several years the cost of resolution was handled by not paying it: latent diffusion generates inside a pretrained autoencoder (Rombach et al., 2022), at the price of a reconstruction ceiling and a second training stage. To manage this, early pixel-space models relied on U-Nets that omitted attention except at the lowest resolutions, before advancing to cascaded models (Ho et al., 2022), single models with resolution-shifting noise schedules (Hoogeboom et al., 2023; 2025), and finally, hierarchical transformers (Crowson et al., 2024; Chen et al., 2025). The most recent line drops the hierarchy altogether: JiT (Li & He, 2026) and PixelDiT (Yu et al., 2026) denoise large patches with a plain transformer and a narrow patch embedding, matching latentmodel quality at the same sequence length. While this approach offers simplicity, it relies on a single uniform patch grid where size rigidly fixes the computational cost of every step. It also introduces a known weakness in decoding large patches, which PixelDiT addresses with a second pixel-leve stage and PixNerd resolves with a per-patch neural field (Wang et al., 2026a). We adapt a released model of exactly same architecture with AdaLN conditioning (Peebles & Xie, 2023) trained via flow matching (Lipman et al., 2023) settings and classifier-free guidance (Ho & Salimans, 2021) utilized during inference .

Content-adaptive tokenization. Non-uniform, layout-based token reduction first emerged for discriminative vision tasks within pretrained transformers (Bolya et al., 2023; Rao et al., 2021; Liang et al., 2022; Yin et al., 2022; Wang et al., 2021), and later as tokenization proper, through flexible patch sizes and content-following tokenizers (Beyer et al., 2023; Dehghani et al., 2023; Lew et al., 2024; Ronen et al., 2023; Havtorn et al., 2023; Kim et al., 2025). Adapting such strategies remains difficult for generation, because a denoiser must write an image through the layout and no image exists when the layout is chosen. Adaptive work here has therefore stayed off the input grid, varying what is updated or cached along the trajectory (Liu et al., 2026; Bolya & Hoffman, 2023; Wu et al., 2025; Zou et al., 2025), varying patch size uniformly over the image (Anagnostidis et al., 2025; Li et al., 2026), or retraining a latent tokenizer (Shen et al., 2026). Concurrent with us, RTI (Zamfir et al., 2026) adapts a frozen pixel-space model using LoRA to consume region tokens. In contrast, our method, BudgetPix, plugs in our adaptive encoder and decoder, fully fine-tunes the entire checkpoint to accept variable-sized inputs starting from the first block, and derives the layou from the model’s own estimate of the clean image rather than its internal features.

## 3 BUDGETPIX

## 3.1 OVERVIEW

BudgetPix is designed to adapt existing pixel-space diffusion transformers into compute-adaptive image generators. As shown in Figure 3, it consists of three core components: an encoder that compresses images into variable-length, multi-scale token sequences (Sec. 3.2), a complementary decoder that recon-

![](images/9cf5b85392f31487615c534c95f95f34888e686e9a77644ee18509138176704b.jpg)  
Figure 3: Overview. BudgetPix enables compute-adaptive pixel-space diffusion with three components: an encoder, a decoder, and flexible training and inference across token counts. The Encoder and Decoder (purple) are trained from scratch; the Transformer (blue) is initialized from existing baselines.

structs these token sequences back to fixed-resolution images (Sec. 3.3), and a unified training and inference strategy (Sec. 3.4). By replacing the uniform tokenization of standard baselines with this content-adaptive pipeline, BudgetPix enables existing diffusion backbones to seamlessly process flexible token counts and adjust their computational budget based on visual complexity.

## 3.2 BUDGETPIX ENCODER: EMBED NON-UNIFORM PATCHES

Non-Uniform Layout Construction: The image layout determines the spatial partition of the image–specifically, which regions maintain the base $p \times p$ patch resolution and which are merged into larger patches to be encoded as a single token. Because visual complexity is spatially nonuniform, fine details concentrate on specific regions while homogeneous areas $( e . g .$ , sky or a plain wall) contain minimal information. Inspired by APT (Choudhury et al., 2026), we construct thi non-uniform layout based on local intensity entropy. The layout is computed from the image during both training and inference. During training, it is derived directly from the ground-truth image x. At inference, because the clean image is unavailable, the model initially runs a few warm-up denoising steps on a uniform $p \times p$ grid to generate a predicted image ${ \hat { x } } ,$ which is then used to estimate the content-adaptive layout.

Given N different patch scales $s = 2 ^ { j } p , j \in \{ 0 , 1 , \ldots , N - 1 \}$ , with $N - 1$ the coarsest level, corresponding to the largest patch size. For each spatial cell R at this coarsest level, we compute a K-bin histogram of its internal pixel intensities and calculate the corresponding Shannon entropy: $\begin{array} { r } { H ( R ) = - \sum _ { k = 1 } ^ { K } q _ { k } ( R ) \log _ { 2 } q _ { k } ( R ) } \end{array}$ , where $q _ { k } ( R )$ denotes the fraction of pixels falling into bin k.

![](images/9ad237793f38c9bbec78f257d204790b44569663f4adb51a8f76d93de82c44cc.jpg)  
Figure 4: Non-uniform layout construction. Given an image (the original image x during training, or the denoising image xˆ during inference), the objective is to partition it into cells across multiple scales: $p \times p ,$ $2 p \times 2 p , \hdots , n p \times n p$ , where $n = 2 ^ { j }$ for $j \geq 1$ . The process begins by dividing the image into np × np cells, which is the coarsest scale. For each cell, we construct a pixel intensity histogram and compute its entropy H. If $H < \tau _ { n p } + \delta$ (where $\tau _ { n p }$ is the scale-specific threshold and δ is a global offset), the cell remains a single token. Otherwise, it is subdivided into four cells of the next finer scale $( n  n / 2 )$ . This recursive subdivision terminates at $n = 1$ , corresponding to the minimum cell size of $p \times p$

Token allocation starts from the coarsest level—a cell is retained as one token if $H ( R ) \leq \tau _ { s } + \delta$ where $\tau _ { s }$ is a scale-specific threshold and $\delta \textbf { a }$ global offset; otherwise, it is recursively split into four quadrants and evaluated at the next finer scale. This recursive quadtree splitting continues to the finest $p \times p$ scale, ensuring the resulting multi-scale tokens perfectly tile the image (Figure 4).

BudgetPix Encoder: The BudgetPix encoder receives non-uniform image patches defined by the entropy-guided quadtree and encodes them into a set of tokens with the same embedding size. Specifically, a patch $P _ { i }$ at scale $s = n p$ (where $n = 2 ^ { j }$ and $j \in \{ 0 , 1 , \ldots , N - 1 \} ,$ ) is mapped to a single token vector $\mathbf { x } _ { i }$

To process these variable-sized patches, the encoder utilizes a dual-path architecture (Figure 5, left). The fine path processes base patches $( n = 1 ) \colon$ : each $p \times p$ patch is embedded directly via the pretrained patch embedder E inherited from the base model. The coarse path processes larger patches $( n > 1 )$ . These patches are bilinearly downsampled to $p \times p ,$ denoted as $\mathrm { r e s i z e } _ { p } ( P _ { i } )$ , and subsequently embedded via E. While this main stream preserves macro-level semantics, it inherently discards high-frequency internal details. To recover this lost detail, a localized secondary stream, the Patch Aggregator $( \mathrm { P A } _ { s } )$ , is introduced. $\mathrm { P A } _ { s }$ partitions $P _ { i }$ into $n ^ { 2 }$ sub-patches, denoted as $\mathrm { s p l i t } ( P _ { i } )$ Each sub-patch is independently embedded with $E _ { \mathrm { { i } } }$ , augmented with a learned relative positional encoding $\mathbf { e _ { \mathrm { p o s } } }$ to specify its spatial location within $P _ { i } ,$ and finally fused into a single correction vector using a scale-specific mixer. This scale-mixer comprises a stack of convolutional layers followed by an MLP layer (architectural details are provided in supplementary). Both paths add a learned scale embedding $\mathbf { e } _ { \mathrm { s c a l e } } ( s )$ , so the token is

$$
\mathbf { x } _ { i } = \left\{ \begin{array} { l l } { E ( P _ { i } ) + \mathbf { e } _ { \mathrm { s c a l e } } ( s ) , } & { n = 1 , } \\ { E \big ( \mathrm { r e s i z e } _ { p } ( P _ { i } ) \big ) + \mathbf { e } _ { \mathrm { s c a l e } } ( s ) + \mathrm { P A } _ { s } \big ( \mathrm { s p l i t } ( P _ { i } ) \big ) , } & { n > 1 . } \end{array} \right.\tag{1}
$$

## 3.3 BUDGETPIX DECODER: DECODING MULTI-SCALE IMAGE PATCHES

The BudgetPix decoder receives the output token sequence from the central transformer–where each token corresponds to a specific patch in the non-uniform spatial layout–and reconstructs them into pixel-space patches at their designated scales. These reconstructed patches subsequently tile together to form the final predicted image xˆ. Specifically, the decoder maps the latent token $\mathbf { h } _ { i } .$ corresponding to a patch $P _ { i }$ of scale $s = n p$ , back into an $n p \times n p$ pixel space patch $\hat { P } _ { i } ^ { ( s ) }$

To reconstruct tokens of varying scale back into pixel space, the decoder (Figure 5, right) mirrors the dual-path architecture of the encoder. The fine path processes base tokens $( n = 1 ) !$ : a pretrained decoding head maps the latent token $\mathbf { h } _ { i }$ directly into a $\mathbf { \nabla } p \times p$ pixel patch $\hat { P } _ { i } \in \mathbb { R } ^ { p \times p \times C }$ , where $C = 3$ represents the RGB image channels. The coarse path handles larger tokens $( n > 1 )$ . Initially, the main stream decodes the token and bilinearly upsamples the output $\hat { P } _ { i }$ to the target resolution of $n p \times n p$ , denoted as resiz $\mathrm { e } _ { n p } ( { \hat { P } } _ { i } )$ . While this main stream successfully recovers the region’s lowfrequency structural layout, the resulting patch is inherently over-smoothed, lacking internal highfrequency textures. To resolve this, a localized secondary stream–the Patch Refiner $( \bar { \mathrm { P R } } _ { s } )$ –generates and injects the missing high-frequency details as an additive residual:

![](images/131f788d27ee808972c94b0f292fe177fd1772d3513986ec5daa8f62dc09223d.jpg)  
Figure 5: BudgetPix Architecture. The encoder receives the noisy input $z _ { t }$ and its corresponding spatial layout $L ,$ converts the image into a sequence of multi-scale patches. The finest $p \times p$ tokens route directly through the fine path, while coarser patches navigate a dedicated coarse path in both the encoder and decoder. In the encoder, each coarse patch undergoes simultaneous downscaling $( \tan p \times p )$ and splitting via a Patch Aggregator $( \mathrm { i n t o } n ^ { 2 }$ sub-patches). These streams are processed through MLP layers and a scale mixer before being fused into a single coarse token. In the decoder, each denoised coarse patch is upscaled back to its target resolution $( n p \times n p )$ . Concurrently, a Patch Refiner splits the patch into $4 \times 4$ sub-patches, processes them through an MLP and a lightweight transformer, and fuses them with the upscaled patch to restore local texture. The central diffusion transformer is directly initialized from existing pixel-space models, modified only to operate across variable, dynamically determined token counts rather than a fixed count.

$$
\hat { P } _ { i } ^ { ( s ) } = \operatorname { r e s i z e } _ { n p } \big ( \hat { P } _ { i } \big ) + \operatorname { P R } _ { s } \big ( \operatorname { s p l i t } ( \hat { P } _ { i } ) , \mathbf { h } _ { i } \big ) ,\tag{2}
$$

where $\hat { P } _ { i }$ is the base patch from the pretrained head; on the fine path the resize is the identity and the residual is absent. The refiner is a lightweight transformer that stays inside one coarse token: it embeds $4 \times 4$ sub-patches $\mathrm { s p l i t } ( { \hat { P } } _ { i } )$ , adds local positional embeddings $\mathbf { e _ { p o s } } ,$ processes the sequence with two self-attention layers under adaptive layer normalization $\mathrm { ( A d a L N ) }$ conditioned on $\mathbf { h } _ { i } ,$ , which brings the token’s global context into the local details, and projects the result linearly to the $n p \times n p$ residual. Because it runs on short, local token sequences, its cost per token does not grow with image resolution, and it adds only ${ \sim } 1 \%$ to the parameter count.

The central diffusion transformer subsequently adds its own position embeddings and evaluates the tokens as usual. Note that the position embeddings applied within the Patch Aggregator and Patch Refiner operate in a localized, patch-relative coordinate system, whereas the embeddings utilized by the central diffusion transformer operate in a global, image-relative coordinate system.

## 3.4 TRAINING AND INFERENCE

Training Objective: Any pretrained pixel-space diffusion transformer can be integrated into the BudgetPix framework by encapsulating it with our trainable encoder and decoder modules. The entire pipeline is then fine-tuned utilizing the base model’s flow-matching objective for full-image x-prediction. Let the noisy latent state be $z _ { t } = t x + ( 1 - t ) \epsilon$ , and the network prediction be $\hat { x } = f _ { \theta } ( z _ { t } , t , y , L )$ , where t is the timestep, y is the conditioning signal, and $L$ is the spatial layout. The fine-tuning loss is formalized as: $\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } , L , t , \epsilon } \big [ \boldsymbol { w } ( t ) \big | \big | \boldsymbol { x } - \hat { \boldsymbol { x } } \big | \big | _ { 2 } ^ { 2 } \big ]$ where $w ( t )$ represents the noise-schedule weighting derived from the base model.

Layout Augmentation: To ensure the transformer can reliably process sequences of varying lengths, we dynamically assign one of three distinct layout types to each image during training. Let $Q ( \cdot , \tau + \delta )$ represent the entropy quadtree defined by thresholds $\tau _ { s } + \delta ,$ and let $H ( x )$ denote the true entropy map of the training image x. Uniform layout (probability $a \ < \ 1 ) :$ : The model uses $L _ { \mathrm { u n i f o r m } } ,$ the standard dense grid of the base architecture. This anchors the training process and prevents catastrophic drift from the pretrained weights. Random layout (probability b, where $a + b < 1 )$ : The quadtree $Q ( { \tilde { H } } , \tau + \delta )$ is applied to a synthetic entropy map $\tilde { H }$ , where values are drawn uniformly between 0 and $\log _ { 2 } K$ bits per cell. Crucially, this forces coarse tokens to randomly encompass highly detailed image regions. This simulates the constraints of low-budget inference, preventing the model from assuming that coarse tokens only represent flat backgrounds.

![](images/b5c3d7eddeba1ea898145f02d9464c7dd5c00952e31bc0faaac015f8fda16738.jpg)  
Figure 6: Qualitative comparison across budgets, MiniT2I-L/16 at $5 1 2 ^ { 2 } .$ . Same prompt and noise were used for all methods. BudgetPix generates image with high quality even at 10%.

Content-adaptive layout (probability $1 - a - b )$ : The standard quadtree $Q ( H ( x ) , \tau + \delta )$ is applied using the actual image entropy map, optimally routing tokens based on true visual complexity. For both the random and content-adaptive quadtree layouts, the global offset δ is randomly sampled for each training image. This ensures the model learns to operate across a continuous range of token counts, scaling from roughly 25% up to 100% of the original dense grid.

Sampling with Dynamic Layout Estimation: During inference, the image’s entropy is unknown before it is generated, so the layout is estimated as the image forms. Let i index the sampling steps and ${ \hat { x } } _ { i }$ be the model’s prediction of the clean image at step i. The first $n _ { w }$ warm-up steps run on the uniform grid, $L _ { i } = L _ { \mathrm { u n i f o r m } }$ , which lets the model set the global composition of the image. After that, the layout is the entropy quadtree of the current prediction, $L _ { i } = \mathsf { \bar { Q } } \big ( H ( \hat { x } _ { i ^ { \prime } } ) , \tau + \delta ^ { * } \bar { ( } \hat { x } _ { i ^ { \prime } } , B ) \big )$ with the thresholds shifted by the offset $\delta ^ { * }$ that budget mode finds for this prediction and the budget B. The layout is rebuilt only every m steps: $i ^ { \prime } = \bar { n _ { w } } + m \lfloor ( i - n _ { w } ) / m \rfloor$ is the last rebuild step at or before i, so the m steps from $i ^ { \prime }$ on share one layout.

## 4 EXPERIMENTS

## 4.1 IMPLEMENTATION DETAILS

We adapt seven released pixel-space diffusion transformers: JiT-B/16 and JiT-L/16 at $2 5 6 ^ { 2 }$ and JiT-B/32 and JiT-L/32 at $5 1 2 ^ { \overline { { 2 } } }$ (Li & He, 2026), MiniT2I-B/16 and MiniT2I-L/16 (Wang et al., 2026b) at $5 1 2 ^ { 2 }$ , and PixelDiT-T2I (Yu et al., 2026) at $1 0 2 4 ^ { 2 }$ , plus the one-step pixel MeanFlow pMF-L/16 (Lu et al., 2026) at $2 5 6 ^ { 2 }$ . Each is fine-tuned once with the layout augmentation (with $a = 0 . 1 5$ $b = 0 . 1 )$ using the objective described in Section 3.4. The added modules are zero-initialized, so training starts exactly from the released model (see App. C.2–C.4).

## 4.2 EVALUATION SETTINGS

Baselines. We compare BudgetPix against the released checkpoints driven to the same budget through our sampler and three token-reduction methods on those weights. The three token-reduction baselines are: (i) ToMe (Bolya et al., 2023), the feature-similarity merge of ToMeSD (Bolya & Hoffman, 2023) applied once, (ii) down to the budget (Bolya & Hoffman, 2023) (FeatSim), and (iii) RTI (Zamfir et al., 2026), a concurrent adapter with region tokens, run from its released checkpoint. Every method shares prompts or labels, initial noise, solver, step count and attention arithmetic; only the budget and the reduction rule differ. See App. C.5 for the protocol.

(a)  
![](images/b17d3ebf36b858e0ecd51816549de830bab53551ce13e50ebe8c9172badf2c9a.jpg)

![](images/87438354b6d3b809150ba04c0e6dce68bd96876989f61cec373b84d7aed53651.jpg)

![](images/aeb91a5b513e5887297cb8100e0e5dbf9535999125c0a99c8f54f6e976bce121.jpg)

![](images/96c6e7918b52599d56310b8ee9aaeda9a21e789359de18d0d60c89e3c28b0be5.jpg)  
(b)

![](images/98637c7fb9c7390641e224d0146fa4c586a1fca0ba7d3476dac70a4d8b673ddc.jpg)

![](images/20964e5a98bf383b03379f3680cc3156941a0cd78df8221ee840586312ee1d5d.jpg)

![](images/7bb113428e778f60f7907679a7dc3d5a8a02d55ef53c2ebd302d09edb6daa2eb.jpg)

![](images/358422bfcc23dffa43f0be0db08ac93611a6b440a79858a60d8377f8c0ae25ac.jpg)

![](images/404f6393610cd7737ffa65c4a95d08816b2bcf88077b9bfd3286795660cb840a.jpg)

![](images/5987cc3778f70db214d87a74054f7136fddbebb394a4f22728c7d2e3ecdac3ad.jpg)  
Base model ToMe FeatSim RTI BudgetPix  
Figure 7: Text-to-image results, MiniT2I-L/16 at $5 1 2 ^ { 2 } .$ . (a) Each metric plotted against the wall-clock speedup over the released model (DPG: DPG-Bench; IR: ImageReward denote metrics). (b) Faithfulness to the method’s own full-budget image plotted against the token budget (Sim.: CLIP or DINOv2 feature similarity).

Table 1: Text-to-image generation with PixelDiT-T2I at $1 0 2 4 ^ { 2 }$ . BudgetPix consistently outperform the base model across different compute budgets.
<table><tr><td>Metric</td><td>Method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>GenEval ↑</td><td>Base model BudgetPix</td><td>0.721 0.741</td><td>0.722 0.740</td><td>0.724 0.741</td><td>0.725 0.738</td><td>0.708 0.746</td><td>0.674 0.743</td><td>0.323 0.725</td></tr><tr><td>DPG-Bench ↑</td><td>Base model BudgetPix</td><td>84.8 85.2</td><td>84.6 85.2</td><td>84.2 85.0</td><td>84.0</td><td>83.3 84.6</td><td>81.8</td><td>55.1</td></tr></table>

Dataset and metrics. We use FID-50k (Heusel et al., 2017) and Inception score (Salimans et al., 2016) for class-conditional quality, GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024) for prompt adherence, and CLIP (Radford et al., 2021), PickScore (Kirstain et al., 2023) and ImageReward (Xu et al., 2023) for preference on PartiPrompts (Yu et al., 2022). We additionally measure faithfulness, whether a reduced-budget output preserves the same image, using paired PSNR, SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018), CLIP and DINOv2 (Oquab et al., 2024) similarity against each method’s own output at full 100% budget. Cost is measured by wall-clock time; see App. C.7 for details.

## 4.3 ANALYSIS

Class-conditional generation. On $5 1 2 ^ { 2 }$ generation for JiT-L/32 (Figure 8), BudgetPix starts at the released model’s FID and increases gradually as the budget decreases, whereas the base model and merging baselines increase sharply. At 50-60% budgets, their FID is several times ours, with ToMe and Feat-Sim performing worse than the base model at the same budget. Inception Score shows the same trend: BudgetPix retains FID scores of a fully dense layout even at a 60% budget, where the baselines deteriorate substantially. Additional results for JiT-L/16 at

![](images/d98fb2d10436c6929f28045f9f8fa4992f098d237260ea86170e5923d60222f8.jpg)

![](images/0e7d782dc0bc0013b2425f1489ea1ab098cf9d619f21d97ce6ca4e782fc7d202.jpg)  
Base model ToMe FeatSim BudgetPix  
Figure 8: FID and Inception Score (IS) on classconditional generation across token budgets on ImageNet, JiT-L/32 at $5 1 2 ^ { 2 }$ . FID is cut at 40.

256<sup>2</sup> are provided in App. B.1 and more qualitative results are presented in App. A.2.

Faithfulness across the budget. We measure faithfulness, that is, how much a method’s image at a reduced budget similar to the image the same method produces at their full budget. In Figure 7(b), ours stays closest to its dense image on nearly every metric and budget, while the base model and both baselines significantly diverge from the original image at 70% budget. These gaps can be attributed to the fact that our layout merges the flattest regions first and the refiner restores their texture, whereas the merging baselines group tokens by feature similarity across the whole frame and the base model receives coarse tokens it was never trained on. See App. B.2 for the base model and ours on JiT.

![](images/f6e92903cc97f88b2e17b526ddfb87cef4fb43954f672899b7e0501361cc0e18.jpg)

![](images/6ccd5c91cf8530155f7f488dad42a6116f04b54a3b433bc4a27ed281c90fe479.jpg)  
Token Budget (%)

Figure 9: User study (top) and VLMas-Judge (bottom), MiniT2I-L/16 at $5 1 2 ^ { 2 }$ : (a) How many percentage from the compute budget can be cut without a noticeable difference for human and VLM judges. (b) Head-to-head comparison of BudgetPix against baselines by human and VLM judges, showing our method is consistently preferred across all token budgets.

Table 2: Ablation on JiT-B, FID-50k. BudgetPix with both Patch Aggregator (sec 3.2) and Refiner (sec 3.3) works best across different compute budgets.
<table><tr><td></td><td>Model</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td></td><td>Base Model</td><td>4.12</td><td>5.00</td><td>6.86</td><td>12.24</td><td>24.79</td><td>60.61</td><td>190.71</td></tr><tr><td> $5 1 2 ^ { 2 }$ </td><td>+ Patch Aggregator</td><td>3.96</td><td>4.52</td><td>5.26</td><td>6.67</td><td>8.56</td><td>11.58</td><td>25.07</td></tr><tr><td></td><td>+ Patch Refiner</td><td>3.94</td><td>4.35</td><td>4.80</td><td>5.57</td><td>6.59</td><td>8.30</td><td>16.71</td></tr><tr><td></td><td>Base Model</td><td>3.61</td><td>4.24</td><td>5.84</td><td>10.67</td><td>23.46</td><td>63.94</td><td>145.46</td></tr><tr><td> $2 5 6 ^ { 2 }$ </td><td>+ Patch Aggregator</td><td>3.57</td><td>3.92</td><td>4.34</td><td>5.14</td><td>6.17</td><td>7.84</td><td>16.90</td></tr><tr><td></td><td>+ Patch Refiner</td><td>3.60</td><td>3.89</td><td>4.25</td><td>4.89</td><td>5.73</td><td>7.01</td><td>13.05</td></tr></table>

Text-to-image generation at $5 1 2 ^ { 2 }$ and $1 0 2 4 ^ { 2 }$ On MiniT2I-L (Figure $\gamma ( { \mathrm { a } } ) ;$ ; values in Table 5) GenEval and DPG-Bench BudgetPix achieves similar, stable scores across compute budgets while the base model degrades significantly, to near-zero scores. A similar trend is also observed in PixelDiT at $1 0 2 4 ^ { 2 }$ resolution (Table 1). In addition, Figure 6 visually demonstrates the images generated by each baseline at every budget, confirming the stable quality of our model across budgets. For more results, see App. A.1, App. A.2, and App. A.8.

User study and VLM-as-Judge. The paired metrics say how far an image moved, not when a reader would notice. In a user study (Figure 9), 50 people per form compare images on 20 GenEval prompts, and in VLM-as-Judge a vision–language model does the same on 100. Shown a method’s own dense image and a grid of its budgets, the budget can be cut by 52 % for ours before human raters first see a difference, against at most 38 % for the baselines, and by 36 % against at most 21 % before the VLM judge does (Figure 9a). When shown the full-budget image and two budgetreduced candidates from two different methods, human raters find ours more faithful than RTI at al four budgets considered. The VLM judge also favors ours than every baseline below 80 % (Figure 9 b). See App. C.6 for both protocols.

Wall-clock and speed-up. Figure 7(a) plots every text-to-image metric against the measured wallclock speed-up over the full budget inference time. BudgetPix maintains its scores across all five metrics and on the whole speed-up range, the base model starts at the same value but collapses fast. The merging baselines buy less speed, since the merge itself costs time. See App. B.4 for the other base models.

One-step generation with MeanFlow. BudgetPix reduces the cost per step, rather than the number of steps, making it complementary to inference-efficiency methods such as distillation. We note that BudgetPix is still applicable to one-step generation methods by using the random layout (described in sec 3.4) instead of the layout from the warm-up steps. Figure 10 shows our results on one-step (1-NFE) pixel MeanFlow (pMF-L/16; Lu et al., 2026), finetuned on layouts from training images of each class. While the released model’s weights collapse at 80 %; BudgetPix maintains FID below more results). more results).

![](images/03770bcc4c92f1986b6319d812e6e851475b3fcee160b6890e71e8cd689ccf69.jpg)

![](images/9d546fa41357328f715f5a81a8a37251769e6c7222ca42e3ed43cabf5b6d2645.jpg)  
Base model ToMe FeatSim BudgetPix  
Figure 10: BudgetPix on one-step pixel MeanFlow pMF-L/16. BudgetPix provides a better qualityefficiency trade-off compared with the base model and other token merging baselines.

lapse at 80 %; BudgetPix maintains FID below 4 down at 50% budget (see App. B.6 and A.5 for

Ablation study. Table 2 provides the ablation study of BudgetPix components. The aggregator does most of the work, since it helps coarse tokens carry their region’s content. The refiner takes roughly a further third off, and its contribution grows as the budget falls, since it acts only inside coarse regions. See App. D for other ablations and App. D.6 for the entropy quadtree against random and oracle layouts.

## 5 CONCLUSION

BudgetPix shows that a pretrained pixel-space diffusion transformer’s token count does not need to remain fixed to its pretraining configuration. After one fine-tuning, a single checkpoint serves budgets spanning from a dense grid to a small fraction of it and degrades gracefully where tokenmerging baselines collapse, even at 1 NFE (Figure 10). Its limitations are that the layout is derived from the model’s intermediate predictions, so it can miss details they do not yet show, and that fullresolution stages, such as PixelDiT’s pixel stage, cap the speed-up. Learning where to place tokens, not deriving it from entropy, is a natural next step.

## REFERENCES

Sotiris Anagnostidis, Gregor Bachmann, Yeongmin Kim, Jonas Kohler, Markos Georgopoulos, Artsiom Sanakoyeu, Yuming Du, Albert Pumarola, Ali Thabet, and Edgar Schonfeld. Flexidit: ¨ Your diffusion transformer can easily generate high-quality samples with less compute. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28316–28326. IEEE, 2025.

Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14496–14506. IEEE, 2023.

Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In 2023 IEEE/CVF conference on computer vision and pattern recognition workshops (CVPRW), pp. 4599–4603. IEEE, 2023.

Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= JroZRaRw7Eu.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3557–3567. IEEE, 2021.

Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025.

Rohan Choudhury, JungEun Kim, Jinhyung Park, Eunho Yang, Laszlo A. Jeni, and Kris Kitani. Faster vision transformers with adaptive patches. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= SzoowJtd14.

Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z Kaplan, and Enrico Shippole. Scalable high-resolution pixel-space image synthesis with hourglass diffusion transformers. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 9550–9575. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr. press/v235/crowson24a.html.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. Advances in Neural Information Processing Systems, 38:75460– 75482, 2026.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 52132–52152, 2023.

Jakob Drachmann Havtorn, Amelie Royer, Tijmen Blankevoort, and Babak Ehteshami Bejnordi.´ Msvit: Dynamic mixed-scale tokenization for vision transformers. In 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pp. 838–848. IEEE, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL https://openreview. net/forum?id=qw8AKxfYbI.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal ofMachine Learning Research, 23(47):1–33, 2022.

Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pp. 13213–13232. PMLR, 2023.

Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion: 1.5 fid on imagenet512 with pixel-space diffusion. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18062–18071. IEEE, 2025.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Minchul Kim, Dingqiang Ye, Yiyang Su, Feng Liu, and Xiaoming Liu. SapiensID: Foundation for human recognition. In CVPR, 2025.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Jaihyun Lew, Soohyuk Jang, Jaehoon Lee, Seungryong Yoo, Eunji Kim, Saehyung Lee, Jisoo Mok, Siwon Kim, and Sungroh Yoon. Superpixel tokenization for vision transformers: Preserving semantic integrity in visual tokens. arXiv preprint arXiv:2412.04680, 2024.

Hui Li, Baoyou Chen, Jiaye Li, Jingdong Wang, and Siyu Zhu. Pyramid patchification flow for visual generation. In International Conference on Learning Representations, volume 2026, pp. 11656–11676, 2026.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 36115– 36125, 2026.

Youwei Liang, Chongjian GE, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. EVit: Expediting vision transformers via token reorganizations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=BjyvwnXXVn\_.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Ziming Liu, Yifan Yang, Chengruidong Zhang, Yiqi Zhang, Lili Qiu, Yang You, and Yuqing Yang. Region-adaptive sampling for diffusion transformers. In Proceedings of the IEEE/CVF Confer ence on Computer Vision and Pattern Recognition, pp. 2346–2356, 2026.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. Machine Intelligence Research, 22 (4):730–751, 2025.

Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, and Kaiming He. One-step latent-free image generation with pixel mean flows. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=WUK8JIeetF.

Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khali-´ dov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182. IEEE, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 10674–10685. ieee, 2022.

Tomer Ronen, Omer Levy, and Avram Golbert. Vision transformers with mixed-resolution tokenization. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 4613–4622. IEEE, 2023.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Junhong Shen, Kushal Tirumala, Michihiro Yasunaga, Ishan Misra, Luke Zettlemoyer, Lili Yu, and Chunting Zhou. Cat: Content-adaptive image tokenization. Advances in Neural Information Processing Systems, 38:712–742, 2026.

Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. In International Conference on Learning Representations, volume 2026, pp. 43559– 43580, 2026a.

Xianbang Wang, Hanhong Zhao, Yiyang Lu, Kangyang Zhou, Linrui Ma, and Kaiming He. Minit2i: A minimalist baseline for text-to-image generation, 2026b. URL https://peppaking8. github.io/#/post/minit2i.

Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, and Gao Huang. Not all images are worth 16x16 words: Dynamic transformers for efficient image recognition. Advances in neural information processing systems, 34:11960–11973, 2021.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.

Haoyu Wu, Jingyi Xu, Hieu Le, and Dimitris Samaras. Importance-based token merging for efficient image and video generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4983–4995. IEEE, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Hongxu Yin, Arash Vahdat, Jose M Alvarez, Arun Mallya, Jan Kautz, and Pavlo Molchanov. A-vit: Adaptive tokens for efficient vision transformer. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10799–10808. IEEE, 2022.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=AFDcYJKhND. Featured Certification.

Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, and Jiebo Luo. Pixeldit: Pixel diffusion transformers for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14273–14282, 2026.

Eduard Zamfir, Christian Reisswig, Zongwei Wu, Yongqin Xian, and Radu Timofte. Elastic token compression for pixel-space diffusion transformers. arXiv preprint arXiv:2608.29281, 2026.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with token-wise feature caching. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=yYZbZGo4ei.

## SUPPLEMENTARY MATERIAL

## CONTENTS OF THE SUPPLEMENTARY MATERIAL

A Additional qualitative results across the budget 16   
A.1 Baseline Comparison (MiniT2I-L/16, 512<sup>2</sup>) 16   
A.2 BudgetPix (JiT-L/32, 512<sup>2</sup>) 19   
A.3 BudgetPix (MiniT2I-L/16, 512<sup>2</sup>) 21   
A.4 BudgetPix (PixelDiT-T2I, 1024<sup>2</sup>) 23   
A.5 BudgetPix (pMF-L/16, 256<sup>2</sup>, 1 NFE) 25   
A.6 BudgetPix layouts (JiT-L/32, 512<sup>2</sup>) 28   
A.7 BudgetPix layouts (MiniT2I-L/16, 512<sup>2</sup>) 30   
A.8 BudgetPix layouts (PixelDiT-T2I, 1024<sup>2</sup>) 32   
B Results in full 34   
B.1 Class-conditional generation (JiT) . 34   
B.2 Faithfulness to the dense image (JiT) 35   
B.3 Text-to-image generation (MiniT2I) 36   
B.4 Wall-clock speed-up (every parent) 37   
B.5 Budget mode against threshold mode (JiT-L/32) 41   
B.6 One-step generation (pixel MeanFlow) 42   
C Details of the method and the experiments 43   
C.1 Architecture of the added modules 43   
C.2 Training objective 44   
C.3 Zero-initialisation check . 45   
C.4 Fine-tuning and sampling settings . 46   
C.5 Evaluation protocol 48   
C.6 User study and VLM-as-Judge 49   
C.7 Wall-clock measurement . 52   
D Ablations on JiT-B 53   
D.1 Reference runs (development JiT-B) . 53   
D.2 A parallel pixel-stream decoder (development JiT-B/32) 53   
D.3 Sampling schedule and step count (development JiT-B/32) 53   
D.4 Guidance scale along the budget (development JiT-B/32) 53   
D.5 Upper bound on a better decoder (development JiT-B/32) . 54   
D.6 Layout and sampler settings (final JiT-B/32) 54   
D.7 Sensitivity to the entropy thresholds (final JiT-B/32) 55   
D.8 Layout evolution during sampling (final JiT-B/32) 55   
D.9 Pooled budget across a batch (final JiT-B/32) 55

## A ADDITIONAL QUALITATIVE RESULTS ACROSS THE BUDGET

## A.1 BASELINE COMPARISON (MINIT2I-L/16, $5 1 2 ^ { 2 } )$

Figures 11–16 show the comparison with the baselines on MiniT2I-L/16 at $5 1 2 ^ { 2 }$ , one prompt per figure. Each row is a method (the base model, ToMe, FeatSim, RTI and ours), and each column a token budget, from 100 % to 10 %.

![](images/49e04af86e0c8d373699c07714627b5ee166a0b08d3093309b5a0933f9c1c1da.jpg)  
Figure 11: Every method across the budget: an umbrella.

![](images/8f689dda060f68a25f41c226249152b17b990f7e3fe0e777d739a4287e598543.jpg)  
Figure 12: Every method across the budget: a herd of cows.

![](images/c0a502d72c611f3b1a062ce00e0d518a3523e2a4974be2eac3edbde08aa29401.jpg)  
Figure 13: Every method across the budget: a seamless vector pattern.

![](images/e4001df00b4a2ba0748308e7fd6f900e5fdf33942eb6fbcf2e6927cba0fd7930.jpg)  
Figure 14: Every method across the budget: a pencil case and binoculars.

![](images/706b799afe2465dbb0f773e03af1d65233cc7ee55eff7619e34367c57b417841.jpg)  
Figure 15: Every method across the budget: an ornate treasure chest.

![](images/8d2f8ffd7fa47f48debf82a352c34d38a2e783fbfb3fae556203e4693e6e1a2d.jpg)  
Figure 16: Every method across the budget: three birds on a branch.

## A.2 BUDGETPIX (JIT-L/32, 512<sup>2</sup>)

Figures 17 and 18 show additional qualitative results on JiT-L/32 at $5 1 2 ^ { 2 }$ . Each row is one ImageNet class, and each column a token budget, from 100 % to 10 %.

![](images/bbb13912f1c002d5d97e3e8f2b4c5d76407028d9f1e4dda4aa30963cbd59f8f7.jpg)  
Figure 17: BudgetPix on JiT-L/32 at 512<sup>2</sup>, one ImageNet class per row.

![](images/be03a27d978c0df20115ae4d351c126222bdde532ac02fbf260423e9a962f696.jpg)  
Figure 18: BudgetPix on JiT-L/32 at 512<sup>2</sup>, one ImageNet class per row (continued).

## A.3 BUDGETPIX (MINIT2I-L/16, 512<sup>2</sup>)

Figures 19 and 20 show additional qualitative results on MiniT2I-L/16 at $5 1 2 ^ { 2 }$ . Each row is one prompt, and each column a token budget, from 100 % to 10 %.

![](images/8dd8f6d559ecef22e5d21931774bbc27580cfa3e3162d3077eb0cf3c03a50b4f.jpg)  
Figure 19: BudgetPix on MiniT2I-L/16 at 512<sup>2</sup>, one prompt per row.

![](images/a2370db605f71fb0a53a8392a741be3b9056e2a943fb935a16bd728cc38c7567.jpg)  
There is an older gentleman with white hair, wearing.

Figure 20: BudgetPix on MiniT2I-L/16 at 512<sup>2</sup>, one prompt per row (continued).

## A.4 BUDGETPIX (PIXELDIT-T2I, 1024<sup>2</sup>)

Figures 21 and 22 show additional qualitative results on PixelDiT-T2I at 1024<sup>2</sup>. Each row is one prompt, and each column a token budget, from 100 % to 10 %.

![](images/a8e25b551bd566ec4b5ec8bff5addd1b353358850e0c9911369407b5d74abb63.jpg)  
Figure 21: BudgetPix on PixelDiT-T2I at 1024<sup>2</sup>, one prompt per row.

![](images/1ca010b5df9d5ccdfd859f833198ba611b8759dda68ebdc339a5b2942c6cd805.jpg)  
The image shows a blue gate in front of...

Figure 22: BudgetPix on PixelDiT-T2I at 1024<sup>2</sup>, one prompt per row (continued).

## A.5 BUDGETPIX (PMF-L/16, 256<sup>2</sup>, 1 NFE)

Figures 23–25 show additional qualitative results on the one-step pMF-L/16 at $2 5 6 ^ { 2 } .$ . Each row is one ImageNet class, and each column a token budget, from 100 % to 25 %; Figure 23 first shows the released model at 100 % and the base model at 50 % (the released weights with our layouts, no training).

![](images/36241836f795264507af2d673f0cd4c5d4ec3ae9cbf74a983b08d55173776280.jpg)  
Figure 23: One-step samples, pMF-L/16 at 256<sup>2</sup>, one ImageNet class per row.

![](images/d4ce18e4dc517df83f0a9e6552a1c12a2c21c12de5c6de509d1b8e1fda5f386e.jpg)  
Figure 24: BudgetPix on pMF-L/16 at 256<sup>2</sup>, one step, one ImageNet class per row.

![](images/841c742b83e9f0b58e7218722ce1dab41de08d316836dd14eebbd31837bd3b45.jpg)  
Figure 25: BudgetPix on pMF-L/16 at 256<sup>2</sup>, one step, one ImageNet class per row (continued).

## A.6 BUDGETPIX LAYOUTS (JIT-L/32, 512<sup>2</sup>)

Figures 26 and 27 show additional qualitative results with their layouts on JiT-L/32 at $5 1 2 ^ { 2 }$ . Each row is one ImageNet class; the columns are the samples at 100, 50 and 25 % of the grid, then the same images with their layouts drawn over them (each token outlined in the colour of its size).

![](images/0960d52624229f80e53f59c8b0c3d74e9544513913b242700571664bfb1afcf2.jpg)  
Figure 26: BudgetPix on JiT-L/32 at 512<sup>2</sup>: samples and their layouts, one ImageNet class per row.

![](images/6afa8baa58c6c9a469e1c9ad964fb47cbb642378b699c7ab19ed0d6681cced66.jpg)  
Figure 27: BudgetPix on JiT-L/32 at 512<sup>2</sup>: samples and their layouts, one ImageNet class per row (continued).

## A.7 BUDGETPIX LAYOUTS (MINIT2I-L/16, 512<sup>2</sup>)

Figures 28 and 29 show additional qualitative results with their layouts on MiniT2I-L/16 at 512<sup>2</sup>. Each row is one prompt; the columns are the samples at 100, 50 and 25 % of the grid, then the same images with their layouts drawn over them (each token outlined in the colour of its size).

token size: p 2p 4p

![](images/a62e46615ade2a6bb6c44e0aaf6be495e6685b7d6fc84f8245328cc09647e4b6.jpg)  
a photo of a black teddy bear

Figure 28: BudgetPix on MiniT2I-L/16 at 512<sup>2</sup>: samples and their layouts, one prompt per row.

![](images/253b62ec53b53d4def87e191feaa512c67f1dddc4f6d1473fb86793fc994199c.jpg)  
Rows of fresh produce line the interior of a..

Figure 29: BudgetPix on MiniT2I-L/16 at $5 1 2 ^ { 2 }$ : samples and their layouts, one prompt per row (continued).

## A.8 BUDGETPIX LAYOUTS (PIXELDIT-T2I, 1024<sup>2</sup>)

Figures 30 and 31 show additional qualitative results with their layouts on PixelDiT-T2I at 1024<sup>2</sup>. Each row is one prompt; the columns are the samples at 100, 50 and 25 % of the grid, then the same images with their layouts drawn over them (each token outlined in the colour of its size).

![](images/e164da73139f15c1b8ff15904efa25acf11790bf7abc2f138af8884b081fd894.jpg)  
The image shows a drawing of a floor plan...

Figure 30: BudgetPix on PixelDiT-T2I at 1024<sup>2</sup>: samples and their layouts, one prompt per row.

![](images/da1266b45665c7f06071e7d767c1387f12958f9be390d22f40d65446457f1a08.jpg)  
The image shows a lush forest filled with lots...

Figure 31: BudgetPix on PixelDiT-T2I at $1 0 2 4 ^ { 2 } { \mathrm { ; } }$ : samples and their layouts, one prompt per row (continued).

## B RESULTS IN FULL

The tables and figures behind Section 4.3, one subsection per result.

## B.1 CLASS-CONDITIONAL GENERATION (JIT)

FID-50k and Inception score (IS) on ImageNet (see App. C.5) for the four JiT parents, JiT-L above JiT-B: Table 3 gives every value, and Figure 32 plots them with FID cut at 40.

Table 3: Class-conditional generation on ImageNet, JiT-L and JiT-B: FID-50k and Inception score.
<table><tr><td></td><td></td><td colspan="7">FID-50k↓</td><td colspan="7">Inception score ↑</td></tr><tr><td>parent</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>JiT-L/32, 512²</td><td>Base model</td><td>2.66</td><td>3.07</td><td>4.34</td><td>8.09</td><td>15.51</td><td>31.57</td><td>107.51</td><td>337</td><td>327</td><td>305</td><td>259</td><td>203</td><td>127</td><td>13</td></tr><tr><td></td><td>ToMe</td><td>2.66</td><td>3.25</td><td>6.82</td><td>14.83</td><td>28.40</td><td>49.16</td><td>129.69</td><td>337</td><td>307</td><td>257</td><td>197</td><td>134</td><td>78</td><td>13</td></tr><tr><td></td><td>FeatSim</td><td>2.66</td><td>3.20</td><td>6.04</td><td>16.62</td><td>47.19</td><td>110.60</td><td>172.31</td><td>337</td><td>312</td><td>264</td><td>179</td><td>77</td><td>16</td><td>5</td></tr><tr><td></td><td>BudgetPix</td><td>2.71</td><td>2.94</td><td>3.20</td><td>3.69</td><td>4.41</td><td>5.57</td><td>11.54</td><td>341</td><td>336</td><td>328</td><td>316</td><td>302</td><td>281</td><td>206</td></tr><tr><td>JiT-L/16, 2562</td><td>Base model</td><td>2.71</td><td>3.01</td><td>4.44</td><td>9.07</td><td>18.09</td><td>35.95</td><td>116.16</td><td>328</td><td>318</td><td>294</td><td>246</td><td>188</td><td>115</td><td>12</td></tr><tr><td></td><td>ToMe</td><td>2.71</td><td>2.99</td><td>5.71</td><td>18.43</td><td>41.58</td><td>70.13</td><td>159.39</td><td>328</td><td>309</td><td>264</td><td>173</td><td>92</td><td>45</td><td>7</td></tr><tr><td></td><td>FeatSim</td><td>2.71</td><td>3.45</td><td>7.77</td><td>22.56</td><td>56.21</td><td>120.23</td><td>188.34</td><td>328</td><td>298</td><td>241</td><td>150</td><td>61</td><td>13</td><td>4</td></tr><tr><td></td><td>BudgetPix</td><td>2.56</td><td>2.70</td><td>2.93</td><td>3.33</td><td>3.87</td><td>4.75</td><td>8.90</td><td>331</td><td>328</td><td>322</td><td>312</td><td>300</td><td>284</td><td>221</td></tr><tr><td>JiT-B/32, 5122</td><td>Base model</td><td>4.12</td><td>5.00</td><td>6.86</td><td>12.24</td><td>24.79</td><td>60.61</td><td>190.71</td><td>273</td><td>261</td><td>238</td><td>189</td><td>123</td><td>47</td><td>4</td></tr><tr><td></td><td>ToMe</td><td>4.11</td><td>5.25</td><td>8.81</td><td>16.18</td><td>34.07</td><td>57.40</td><td>147.26</td><td>274</td><td>253</td><td>213</td><td>167</td><td>104</td><td>60</td><td>10</td></tr><tr><td></td><td>FeatSim</td><td>4.11</td><td>5.48</td><td>10.55</td><td>26.88</td><td>66.38</td><td>140.23</td><td>192.26</td><td>274</td><td>248</td><td>196</td><td>120</td><td>47</td><td>10</td><td>4</td></tr><tr><td></td><td>BudgetPix</td><td>3.94</td><td>4.35</td><td>4.80</td><td>5.57</td><td>6.59</td><td>8.30</td><td>16.71</td><td>287</td><td>281</td><td>271</td><td>259</td><td>244</td><td>225</td><td>157</td></tr><tr><td>JiT-B/16, 256²</td><td>Base model</td><td>3.61</td><td>4.24</td><td>5.84</td><td>10.67</td><td>23.46</td><td>63.94</td><td>145.46</td><td>271</td><td>261</td><td>241</td><td>197</td><td>131</td><td>46</td><td>5</td></tr><tr><td></td><td>ToMe</td><td>3.60</td><td>4.47</td><td>6.75</td><td>11.19</td><td>21.08</td><td>36.20</td><td>104.98</td><td>271</td><td>253</td><td>223</td><td>186</td><td>131</td><td>83</td><td>15</td></tr><tr><td></td><td>FeatSim</td><td>3.60</td><td>4.85</td><td>8.72</td><td>20.12</td><td>49.63</td><td>115.89</td><td>166.16</td><td>271</td><td>247</td><td>202</td><td>131</td><td>56</td><td>11</td><td>5</td></tr><tr><td></td><td>BudgetPix</td><td>3.60</td><td>3.89</td><td>4.25</td><td>4.89</td><td>5.73</td><td>7.01</td><td>13.05</td><td>281</td><td>276</td><td>271</td><td>261</td><td>248</td><td>231</td><td>171</td></tr></table>

![](images/71cc3d440b1135e3cc7747af2c8e5f487ff9f9ffe7a4426e9edfb5d3b662c9af.jpg)  
Figure 32: Class-conditional generation on ImageNet, JiT-L and JiT-B.

## B.2 FAITHFULNESS TO THE DENSE IMAGE (JIT)

As in Figure 7(b), each image at a reduced budget is scored against the same model’s image at the dense grid, from the same label and noise (see App. C.5): Figure 33 for JiT-L and JiT-B, and Table 4 for JiT-B, which adds CLIP similarity and splits PSNR between the pixels the layout merged and those it kept. A break (//) marks an axis that does not start at zero.

Token Budget (%)  
Token Budget (%)  
Token Budget (%)  
![](images/857238131bc35e530b4148e5b98f629535ac8ad655f07127bb7f7e88c9f0081c.jpg)  
Token Budget (%)

![](images/a8aa24ab1591371e1d08f7d48406a7e01172b32c00abd38d39b4b489231941da.jpg)

![](images/fee4e1e1e8c95fa0e8f9c973c49574760821ea4fb6912b7ffa43763c46a8a995.jpg)

![](images/88dc7e2f384f3013b61504ddf4e6a90efc88bf7ac6e805e9421b8e8d32f010ea.jpg)

![](images/13f437821fc5d74545155f37caf2de576e6e24495173e17a07572e86e40395e3.jpg)

![](images/24d814d3909b5df18062ad316111d0b4a900d780ed11da8050cb6b4da4a1899d.jpg)

![](images/d32bc238dc31d5477eae04a4cea0a35c141526d6a92369250d08c5f433d09620.jpg)

![](images/e1045c41cf7dc8d0c239f0ed7ea433f018dfb0ffdb7fb1f00c0cec6339f24c25.jpg)  
Figure 33: Paired metrics against the token budget, JiT-L and JiT-B.

Table 4: Faithfulness to the dense image, JiT-B: paired metrics.
<table><tr><td></td><td></td><td colspan="6">JiT-B/32, 5122</td><td colspan="6">JiT-B/16, 2562</td></tr><tr><td>metric</td><td>method</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>PSNR ↑</td><td>Base model</td><td>29.4</td><td>25.8</td><td>23.1</td><td>21.4</td><td>19.7</td><td>16.0</td><td>34.1</td><td>29.4</td><td>25.8</td><td>23.4</td><td>21.2</td><td>18.3</td></tr><tr><td></td><td>BudgetPix</td><td>27.8</td><td>25.8</td><td>24.6</td><td>23.8</td><td>23.2</td><td>22.1</td><td>34.5</td><td>31.7</td><td>29.6</td><td>28.2</td><td>26.9</td><td>24.4</td></tr><tr><td>PSNR, merged pixels ↑</td><td>Base model BudgetPix</td><td>28.5 28.6</td><td>25.7 26.8</td><td>23.4 25.5</td><td>21.7 24.6</td><td>20.0 23.9</td><td>16.1 22.3</td><td>30.5 32.0</td><td>27.4 30.1</td><td>24.8 28.6</td><td>22.9 27.6</td><td>21.2 26.7</td><td>18.4 24.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PSNR, kept pixels ↑</td><td>Base model BudgetPix</td><td>30.1 28.0</td><td>26.4 25.9</td><td>23.5 24.5</td><td>21.5 23.6</td><td>19.6 22.7</td><td>15.2 21.0</td><td>35.9 35.8</td><td>31.2 33.1</td><td>27.4 31.1</td><td>24.6 29.6</td><td>21.7 28.1</td><td>17.3 24.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SSIM ↑</td><td>Base model BudgetPix</td><td>0.924 0.904</td><td>0.867 0.864</td><td>0.809 0.829</td><td>0.758 0.803</td><td>0.704 0.778</td><td>0.613 0.723</td><td>0.946 0.954</td><td>0.890 0.924</td><td>0.816 0.893</td><td>0.745 0.866</td><td>0.660 0.836</td><td>0.525 0.764</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LPIPS ↓</td><td>Base model</td><td>0.166 0.190</td><td>0.273 0.254</td><td>0.382</td><td>0.475 0.345</td><td>0.576 0.379</td><td>0.822 0.450</td><td>0.077 0.068</td><td>0.153</td><td>0.249</td><td>0.340 0.175</td><td>0.451 0.210</td><td>0.644 0.292</td></tr><tr><td></td><td>BudgetPix</td><td></td><td></td><td>0.308</td><td></td><td></td><td></td><td></td><td>0.106</td><td>0.144</td><td></td><td></td><td></td></tr><tr><td>CLIP similarity ↑</td><td>Base model BudgetPix</td><td>0.953 0.947</td><td>0.922 0.931</td><td>0.882</td><td>0.833</td><td>0.753</td><td>0.589 0.863</td><td>0.972</td><td>0.943</td><td>0.901</td><td>0.851 0.940</td><td>0.770 0.925</td><td>0.614 0.884</td></tr><tr><td></td><td></td><td></td><td></td><td>0.917</td><td>0.905</td><td>0.893</td><td></td><td>0.977</td><td>0.966</td><td>0.953</td><td></td><td></td><td></td></tr><tr><td>DINOv2 similarity ↑</td><td>Base model</td><td>0.907</td><td>0.839</td><td>0.758</td><td>0.657</td><td>0.481</td><td>0.078</td><td>0.954</td><td>0.903</td><td>0.827</td><td>0.725</td><td>0.529</td><td>0.117</td></tr><tr><td></td><td>BudgetPix</td><td>0.885</td><td>0.847</td><td>0.813</td><td>0.786</td><td>0.750</td><td>0.654</td><td>0.960</td><td>0.936</td><td>0.907</td><td>0.879</td><td>0.843</td><td>0.737</td></tr></table>

## B.3 TEXT-TO-IMAGE GENERATION (MINIT2I)

Table 5 gives the values behind Figures 7(a) and 34 by token budget, for MiniT2I-B/16 and MiniT2I-L/16 at $\mathrm { \bar { 5 } 1 2 ^ { 2 } }$ . PixelDiT at $1 0 2 4 ^ { 2 }$ is complete in Table 1.

Table 5: Text-to-image generation, MiniT2I-B/16 and MiniT2I-L/16 at 512<sup>2</sup>.
<table><tr><td></td><td></td><td colspan="7">MiniT2I-B/16</td><td colspan="7">MiniT2I-L/16</td></tr><tr><td>metric</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>GenEval ↑</td><td>Base model</td><td>0.876</td><td>0.884</td><td>0.872</td><td>0.865</td><td>0.506</td><td>0.040</td><td>0.006</td><td>0.882</td><td>0.877</td><td>0.868</td><td>0.854</td><td>0.773</td><td>0.183</td><td>0.017</td></tr><tr><td></td><td>ToMe</td><td>0.877</td><td>0.878</td><td>0.867</td><td>0.861</td><td>0.841</td><td>0.779</td><td>0.283</td><td>0.886</td><td>0.891</td><td>0.886</td><td>0.873</td><td>0.845</td><td>0.741</td><td>0.229</td></tr><tr><td></td><td>FeatSim</td><td>0.877</td><td>0.882</td><td>0.871</td><td>0.865</td><td>0.830</td><td>0.610</td><td>0.106</td><td>0.886</td><td>0.880</td><td>0.850</td><td>0.798</td><td>0.676</td><td>0.292</td><td>0.007</td></tr><tr><td></td><td>RTI</td><td>0.850</td><td>0.864</td><td>0.870</td><td>0.869</td><td>0.870</td><td>0.868</td><td>0.856</td><td>0.863</td><td>0.873</td><td>0.867</td><td>0.870</td><td>0.868</td><td>0.871</td><td>0.859</td></tr><tr><td></td><td>BudgetPix</td><td>0.877</td><td>0.884</td><td>0.875</td><td>0.879</td><td>0.875</td><td>0.871</td><td>0.866</td><td>0.880</td><td>0.880</td><td>0.875</td><td>0.875</td><td>0.874</td><td>0.880</td><td>0.874</td></tr><tr><td>DPG-Bench ↑</td><td>Base model</td><td>84.2</td><td>83.7</td><td>83.7</td><td>82.8</td><td>74.3</td><td>23.2</td><td>6.8</td><td>84.7</td><td>84.4</td><td>83.6</td><td>83.0</td><td>80.6</td><td>54.1</td><td>8.0</td></tr><tr><td></td><td>ToMe</td><td>84.1</td><td>84.5</td><td>84.3</td><td>84.0</td><td>82.9</td><td>80.4</td><td>62.8</td><td>84.5</td><td>84.8</td><td>84.3</td><td>83.4</td><td>81.2</td><td>77.0</td><td>54.6</td></tr><tr><td></td><td>FeatSim</td><td>84.1</td><td>84.3</td><td>84.2</td><td>84.2</td><td>83.4</td><td>80.9</td><td>57.8</td><td>84.5</td><td>84.6</td><td>84.0</td><td>82.8</td><td>81.0</td><td>75.1</td><td>17.0</td></tr><tr><td></td><td>RTI</td><td>82.3</td><td>83.8</td><td>83.8</td><td>84.0</td><td>83.8</td><td>83.7</td><td>82.5</td><td>82.9</td><td>84.0</td><td>84.1</td><td>84.4</td><td>83.8</td><td>83.9</td><td>83.0</td></tr><tr><td></td><td>BudgetPix</td><td>84.3</td><td>84.3</td><td>84.5</td><td>84.4</td><td>84.1</td><td>84.4</td><td>83.8</td><td>84.8</td><td>84.7</td><td>84.7</td><td>84.6</td><td>84.7</td><td>84.7</td><td>84.3</td></tr><tr><td>CLIP↑</td><td>Base model</td><td>28.21</td><td>28.12</td><td>28.03</td><td>27.50</td><td>23.39</td><td>17.86</td><td>15.99</td><td>28.88</td><td>28.81</td><td>28.59</td><td>28.30</td><td>27.25</td><td>22.17</td><td>16.20</td></tr><tr><td></td><td>ToMe</td><td>28.61</td><td>28.68</td><td>28.60</td><td>28.35</td><td>28.01</td><td>27.38</td><td>24.48</td><td>28.95</td><td>29.09</td><td>29.14</td><td>28.80</td><td>28.17</td><td>27.21</td><td>23.48</td></tr><tr><td></td><td>FeatSim</td><td>28.61</td><td>28.67</td><td>28.72</td><td>28.72</td><td>28.60</td><td>27.61</td><td>23.08</td><td>28.95</td><td>28.84</td><td>28.55</td><td>27.92</td><td>27.02</td><td>25.68</td><td>17.51</td></tr><tr><td></td><td>RTI</td><td>28.35</td><td>28.10</td><td>28.17</td><td>28.18</td><td>28.22</td><td>28.21</td><td>28.29</td><td>28.71</td><td>28.55</td><td>28.52</td><td>28.61</td><td>28.54</td><td>28.57</td><td>28.67</td></tr><tr><td></td><td>BudgetPix</td><td>28.11</td><td>28.12</td><td>28.07</td><td>28.09</td><td>28.06</td><td>28.06</td><td>27.95</td><td>28.87</td><td>28.91</td><td>28.89</td><td>28.92</td><td>28.96</td><td>29.03</td><td>29.27</td></tr><tr><td>PickScore ↑</td><td>Base model</td><td>22.77</td><td>22.72</td><td>22.63</td><td>22.29</td><td>20.34</td><td>18.66</td><td>18.28</td><td>22.82</td><td>22.72</td><td>22.54</td><td>22.28</td><td>21.53</td><td>19.41</td><td>18.11</td></tr><tr><td></td><td>ToMe</td><td>22.51</td><td>22.43</td><td>22.36</td><td>22.19</td><td>21.94</td><td>21.56</td><td>20.01</td><td>22.83</td><td>22.72</td><td>22.61</td><td>22.37</td><td>21.95</td><td>21.36</td><td>19.72</td></tr><tr><td></td><td>FeatSim</td><td>22.51</td><td>22.47</td><td>22.39</td><td>22.26</td><td>21.98</td><td>21.36</td><td>19.65</td><td>22.83</td><td>22.66</td><td>22.40</td><td>21.99</td><td>21.40</td><td>20.53</td><td>18.29</td></tr><tr><td></td><td>RTI</td><td>22.09</td><td>22.37</td><td>22.37</td><td>22.36</td><td>22.33</td><td>22.30</td><td>22.07</td><td>22.41</td><td>22.71</td><td>22.69</td><td>22.68</td><td>22.66</td><td>22.64</td><td>22.39</td></tr><tr><td></td><td>BudgetPix</td><td>22.50</td><td>22.49</td><td>22.46</td><td>22.43</td><td>22.38</td><td>22.31</td><td>22.00</td><td>22.82</td><td>22.81</td><td>22.79</td><td>22.76</td><td>22.72</td><td>22.63</td><td>22.33</td></tr><tr><td>ImageReward ↑</td><td>Base model</td><td>1.16</td><td>1.14</td><td>1.12</td><td>0.96</td><td>-0.45</td><td>-1.95</td><td>-2.23</td><td>1.23</td><td>1.20</td><td>1.17</td><td>1.10</td><td>0.70</td><td>-1.17</td><td>-2.22</td></tr><tr><td></td><td>ToMe</td><td>1.15</td><td>1.14</td><td>1.12</td><td>1.06</td><td>0.97</td><td>0.77</td><td>-0.38</td><td>1.23</td><td>1.21</td><td>1.19</td><td>1.11</td><td>0.96</td><td>0.67</td><td>-0.67</td></tr><tr><td></td><td>FeatSim</td><td>1.15</td><td>1.15</td><td>1.12</td><td>1.08</td><td>0.99</td><td>0.67</td><td>-0.91</td><td>1.23</td><td>1.20</td><td>1.15</td><td>1.04</td><td>0.80</td><td>0.06</td><td>-2.17</td></tr><tr><td></td><td>RTI</td><td>1.03</td><td>1.12</td><td>1.12</td><td>1.12</td><td>1.11</td><td>1.10</td><td>1.07</td><td>1.14</td><td>1.21</td><td>1.21</td><td>1.21</td><td>1.20</td><td>1.20</td><td>1.16</td></tr><tr><td></td><td>BudgetPix</td><td>1.16</td><td>1.17</td><td>1.16</td><td>1.16</td><td>1.14</td><td>1.14</td><td>1.09</td><td>1.21</td><td>1.22</td><td>1.22</td><td>1.21</td><td>1.21</td><td>1.20</td><td>1.17</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## B.4 WALL-CLOCK SPEED-UP (EVERY PARENT)

Speed-up is measured wall-clock time over the released model at its dense grid (see App. C.7). Figure 34 is Figure 7(a) for MiniT2I-B/16 (DPG: DPG-Bench; IR: ImageReward). Figure 35 (dashed: the released model, 1×) and Table 6 give the speed-up along the budget, Figure 36 quality against it (FID cut at 40), and Tables 7–13 the seconds per image and the speed-up of every method. PixelDiT was timed for the base model and ours only. A break $( / / )$ marks an axis that does not start at zero.

![](images/3e4642972296152934d5704a3f27b46cb1b73c96e85859b9e0c9dde3d6a565dc.jpg)

![](images/821f2e7ba90ebc35f0d63c59af2bdc5c2179fbf01e55f65c6d9d7553ce975221.jpg)

![](images/0c7d191ddff67521c5809ea42688009b99b7e850b5e532ebf06fa5935f48e78b.jpg)

![](images/7d5f17f513c304eb10c4c814cf689d63b8d362d26e0f3a2ba62b1f6f881b163c.jpg)

![](images/06f780bb0cc08b2c66c2c9d2eeacd600df4719a0e12806e3a0fa7fccca8e5a64.jpg)  
Base model ToMe FeatSim RTI BudgetPix

Figure 34: Five metrics against the measured wall-clock speed-up, MiniT2I-B/16 at $5 1 2 ^ { 2 }$  
![](images/b2f56556ada16925cdf5ebade20ed9f3a3125a16f2ee89860023af80226ad074.jpg)

![](images/3b1cb26995c975ef7365c79ecfac543b242d2a79a3e861e8b5eb42efd691369b.jpg)

![](images/ae5a207bbae0ccbdd09fc860601b47d745d064161584950ce1546d1a5f904ab6.jpg)  
Base model ToMe FeatSim RTI BudgetPix

![](images/113c7df8ef1386e4ce154cbf8d07f7a83e3e6c049ea8628abc2b6374e6b31041.jpg)  
Figure 35: Wall-clock speed-up against the token budget, four parents.

Table 6: Wall-clock speed-up over the released model, every parent.
<table><tr><td>parent</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>MiniT2I-L/16, 512²</td><td>Base model</td><td>1.01×</td><td>1.02×</td><td>1.18×</td><td>1.59×</td><td>1.88×</td><td>2.30×</td><td>3.59×</td></tr><tr><td></td><td>ToMe</td><td>0.91×</td><td>0.80×</td><td>0.90×</td><td>0.92×</td><td>1.21×</td><td>1.13×</td><td>1.37×</td></tr><tr><td></td><td>FeatSim</td><td>0.91×</td><td>0.82×</td><td>0.91×</td><td>1.36×</td><td>1.52×</td><td>1.70×</td><td>2.20×</td></tr><tr><td></td><td>RTI</td><td>0.93×</td><td>0.83×</td><td>0.92×</td><td>1.36×</td><td>1.52×</td><td>1.69×</td><td>2.16×</td></tr><tr><td></td><td>BudgetPix</td><td>1.01×</td><td>1.01×</td><td>1.18×</td><td>1.59×</td><td>1.88×</td><td>2.29×</td><td>3.59×</td></tr><tr><td>MiniT2I-B/16, 512²</td><td>Base model</td><td>1.02×</td><td>1.02×</td><td>1.17×</td><td>1.57×</td><td>1.84×</td><td>2.19×</td><td>3.35×</td></tr><tr><td></td><td>ToMe</td><td>0.92×</td><td>0.78×</td><td>0.88×</td><td>0.90×</td><td>1.16×</td><td>1.07×</td><td>1.45×</td></tr><tr><td></td><td>FeatSim</td><td>0.91×</td><td>0.82×</td><td>0.88×</td><td>1.23×</td><td>1.37×</td><td>1.53×</td><td>1.85×</td></tr><tr><td></td><td>RTI</td><td>0.92×</td><td>0.82×</td><td>0.88×</td><td>1.24×</td><td>1.34×</td><td>1.49×</td><td>1.79×</td></tr><tr><td></td><td>BudgetPix</td><td>1.02×</td><td>1.02×</td><td>1.18×</td><td>1.56×</td><td>1.84×</td><td>2.19×</td><td>3.34×</td></tr><tr><td>PixelDiT-T2I, 1024²</td><td>Base model</td><td>0.99×</td><td>0.79×</td><td>0.88×</td><td>0.99×</td><td>1.12×</td><td>1.29×</td><td>1.94×</td></tr><tr><td></td><td>BudgetPix</td><td>1.00×</td><td>0.79×</td><td>0.88×</td><td>1.00×</td><td>1.12×</td><td>1.29×</td><td>1.95×</td></tr><tr><td>JiT-L/32, 512²</td><td>Base model</td><td>0.96×</td><td>1.00×</td><td>1.09×</td><td>1.17×</td><td>1.26×</td><td>1.41×</td><td>1.94×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.08×</td><td>1.12×</td><td>1.16×</td><td>1.32×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.11×</td><td>1.17×</td><td>1.26×</td><td>1.47×</td></tr><tr><td></td><td>BudgetPix</td><td>0.95×</td><td>1.00×</td><td>1.08×</td><td>1.17×</td><td>1.27×</td><td>1.42×</td><td>1.95×</td></tr><tr><td>JiT-B/32, 5122</td><td>Base model</td><td>0.95×</td><td>0.92×</td><td>0.96×</td><td>1.02×</td><td>1.08×</td><td>1.15×</td><td>1.39×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.92×</td><td>0.96×</td><td>0.99×</td><td>1.02×</td><td>1.07×</td><td>1.20×</td></tr><tr><td></td><td>FeatSim</td><td>1.01×</td><td>0.97×</td><td>1.01×</td><td>1.05×</td><td>1.09×</td><td>1.15×</td><td>1.28×</td></tr><tr><td></td><td>BudgetPix</td><td>0.94×</td><td>0.92×</td><td>0.97×</td><td>1.00×</td><td>1.06×</td><td>1.14×</td><td>1.36×</td></tr><tr><td>JiT-L/16, 256²</td><td>Base model</td><td>0.97×</td><td>1.02×</td><td>1.09×</td><td>1.18×</td><td>1.27×</td><td>1.43×</td><td>1.88×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.08×</td><td>1.12×</td><td>1.17×</td><td>1.33×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.98×</td><td>1.06×</td><td>1.11×</td><td>1.18×</td><td>1.27×</td><td>1.49×</td></tr><tr><td></td><td>BudgetPix</td><td>0.97×</td><td>1.01×</td><td>1.09×</td><td>1.18×</td><td>1.27×</td><td>1.42×</td><td>1.91×</td></tr><tr><td>JiT-B/16, 2562</td><td>Base model</td><td>0.97×</td><td>0.98×</td><td>1.03×</td><td>1.11×</td><td>1.19×</td><td>1.32×</td><td>1.63×</td></tr><tr><td></td><td>ToMe</td><td>0.99×</td><td>0.90×</td><td>0.96×</td><td>0.98×</td><td>1.04×</td><td>1.06×</td><td>1.21×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.96×</td><td>1.01×</td><td>1.04×</td><td>1.09×</td><td>1.16×</td><td>1.28×</td></tr><tr><td></td><td>BudgetPix</td><td>0.95×</td><td>0.96×</td><td>1.02×</td><td>1.09×</td><td>1.18×</td><td>1.28×</td><td>1.58×</td></tr></table>

![](images/b6ec7aa360835a6b093e60604a30fd2527a31d3ff9252d6ca0a70cb4953444c3.jpg)

![](images/a44629e4b6a81e99bc17b8192c3f456364314d4c494a85921bfe026ba6db5dd1.jpg)

![](images/0cb0d25a4d1a09588bdf0189bddabd99ba09a424aa7e865e49131eba9d30a0cd.jpg)

![](images/3ae7e2947fd12045d0c147bb0efe46f0365be1aabec69384b6804b4131c82055.jpg)  
Base model ToMe FeatSim RTI BudgetPix  
Figure 36: Quality against measured speed-up, JiT-L/32 and MiniT2I-L/16 at 512<sup>2</sup>.

Table 7: Wall-clock per image and speed-up, MiniT2I-L/16 at $5 1 2 ^ { 2 }$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>4.912</td><td>4.870</td><td>4.205</td><td>3.105</td><td>2.634</td><td>2.151</td><td>1.376</td></tr><tr><td></td><td>ToMe</td><td>5.413</td><td>6.171</td><td>5.493</td><td>5.353</td><td>4.082</td><td>4.382</td><td>3.600</td></tr><tr><td></td><td>FeatSim</td><td>5.419</td><td>6.023</td><td>5.452</td><td>3.643</td><td>3.249</td><td>2.903</td><td>2.252</td></tr><tr><td></td><td>RTI</td><td>5.305</td><td>5.947</td><td>5.402</td><td>3.636</td><td>3.257</td><td>2.926</td><td>2.294</td></tr><tr><td></td><td>BudgetPix</td><td>4.913</td><td>4.872</td><td>4.206</td><td>3.113</td><td>2.626</td><td>2.164</td><td>1.376</td></tr><tr><td>speed-up</td><td>Base model</td><td>1.01×</td><td>1.02×</td><td>1.18×</td><td>1.59×</td><td>1.88×</td><td>2.30×</td><td>3.59×</td></tr><tr><td></td><td>ToMe</td><td>0.91×</td><td>0.80×</td><td>0.90×</td><td>0.92×</td><td>1.21×</td><td>1.13×</td><td>1.37×</td></tr><tr><td></td><td>FeatSim</td><td>0.91×</td><td>0.82×</td><td>0.91×</td><td>1.36×</td><td>1.52×</td><td>1.70×</td><td>2.20×</td></tr><tr><td></td><td>RTI</td><td>0.93×</td><td>0.83×</td><td>0.92×</td><td>1.36×</td><td>1.52×</td><td>1.69×</td><td>2.16×</td></tr><tr><td></td><td>BudgetPix</td><td>1.01×</td><td>1.01×</td><td>1.18×</td><td>1.59×</td><td>1.88×</td><td>2.29×</td><td>3.59×</td></tr></table>

Table 8: Wall-clock per image and speed-up, MiniT2I-B/16 at $5 1 2 ^ { 2 } .$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>1.782</td><td>1.784</td><td>1.549</td><td>1.161</td><td>0.988</td><td>0.828</td><td>0.543</td></tr><tr><td></td><td>ToMe</td><td>1.986</td><td>2.340</td><td>2.061</td><td>2.029</td><td>1.564</td><td>1.695</td><td>1.255</td></tr><tr><td></td><td>FeatSim</td><td>1.988</td><td>2.222</td><td>2.057</td><td>1.472</td><td>1.329</td><td>1.191</td><td>0.983</td></tr><tr><td></td><td>RTI</td><td>1.970</td><td>2.223</td><td>2.067</td><td>1.471</td><td>1.353</td><td>1.218</td><td>1.018</td></tr><tr><td></td><td>BudgetPix</td><td>1.785</td><td>1.776</td><td>1.547</td><td>1.161</td><td>0.988</td><td>0.829</td><td>0.545</td></tr><tr><td>speed-up</td><td>Base model</td><td>1.02×</td><td>1.02×</td><td>1.17×</td><td>1.57×</td><td>1.84×</td><td>2.19×</td><td>3.35×</td></tr><tr><td></td><td>ToMe</td><td>0.92×</td><td>0.78×</td><td>0.88×</td><td>0.90×</td><td>1.16×</td><td>1.07×</td><td>1.45×</td></tr><tr><td></td><td>FeatSim</td><td>0.91×</td><td>0.82×</td><td>0.88×</td><td>1.23×</td><td>1.37×</td><td>1.53×</td><td>1.85×</td></tr><tr><td></td><td>RTI</td><td>0.92×</td><td>0.82×</td><td>0.88×</td><td>1.24×</td><td>1.34×</td><td>1.49×</td><td>1.79×</td></tr><tr><td></td><td>BudgetPix</td><td>1.02×</td><td>1.02×</td><td>1.18×</td><td>1.56×</td><td>1.84×</td><td>2.19×</td><td>3.34×</td></tr></table>

Table 9: Wall-clock per image and speed-up, PixelDiT-T2I at $1 0 2 4 ^ { 2 } .$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model BudgetPix</td><td>2.992 2.974</td><td>3.760 3.754</td><td>3.383 3.378</td><td>2.983 2.973</td><td>2.652 2.651</td><td>2.306</td><td>1.527</td></tr><tr><td>speed-up</td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.307</td><td>1.526</td></tr><tr><td></td><td>Base model</td><td>0.99×</td><td>0.79×</td><td>0.88×</td><td>0.99×</td><td>1.12×</td><td>1.29×</td><td>1.94×</td></tr><tr><td></td><td>BudgetPix</td><td>1.00×</td><td>0.79×</td><td>0.88×</td><td>1.00×</td><td>1.12×</td><td>1.29×</td><td>1.95×</td></tr></table>

Table 10: Wall-clock per image and speed-up, JiT-L/32 at $5 1 2 ^ { 2 }$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>0.394</td><td>0.376</td><td>0.346</td><td>0.323</td><td>0.300</td><td>0.267</td><td>0.195</td></tr><tr><td></td><td>ToMe</td><td>0.377</td><td>0.381</td><td>0.361</td><td>0.350</td><td>0.339</td><td>0.325</td><td>0.287</td></tr><tr><td></td><td>FeatSim</td><td>0.378</td><td>0.381</td><td>0.359</td><td>0.341</td><td>0.324</td><td>0.301</td><td>0.256</td></tr><tr><td></td><td>BudgetPix</td><td>0.396</td><td>0.377</td><td>0.348</td><td>0.322</td><td>0.297</td><td>0.265</td><td>0.194</td></tr><tr><td>speed-up</td><td>Base model</td><td>0.96×</td><td>1.00×</td><td>1.09×</td><td>1.17×</td><td>1.26×</td><td>1.41×</td><td>1.94×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.08×</td><td>1.12×</td><td>1.16×</td><td>1.32×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.11×</td><td>1.17×</td><td>1.26×</td><td>1.47×</td></tr><tr><td></td><td>BudgetPix</td><td>0.95×</td><td>1.00×</td><td>1.08×</td><td>1.17×</td><td>1.27×</td><td>1.42×</td><td>1.95×</td></tr></table>

Table 11: Wall-clock per image and speed-up, JiT-B/32 at $5 1 2 ^ { 2 }$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>0.131</td><td>0.134</td><td>0.129</td><td>0.122</td><td>0.115</td><td>0.108</td><td>0.089</td></tr><tr><td></td><td>ToMe</td><td>0.124</td><td>0.134</td><td>0.129</td><td>0.125</td><td>0.122</td><td>0.116</td><td>0.103</td></tr><tr><td></td><td>FeatSim</td><td>0.122</td><td>0.127</td><td>0.123</td><td>0.118</td><td>0.113</td><td>0.108</td><td>0.097</td></tr><tr><td></td><td>BudgetPix</td><td>0.132</td><td>0.135</td><td>0.128</td><td>0.123</td><td>0.116</td><td>0.109</td><td>0.091</td></tr><tr><td>speed-up</td><td>Base model</td><td>0.95×</td><td>0.92×</td><td>0.96×</td><td>1.02×</td><td>1.08×</td><td>1.15×</td><td>1.39×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.92×</td><td>0.96×</td><td>0.99×</td><td>1.02×</td><td>1.07×</td><td>1.20×</td></tr><tr><td></td><td>FeatSim</td><td>1.01×</td><td>0.97×</td><td>1.01×</td><td>1.05×</td><td>1.09×</td><td>1.15×</td><td>1.28×</td></tr><tr><td></td><td>BudgetPix</td><td>0.94×</td><td>0.92×</td><td>0.97×</td><td>1.00×</td><td>1.06×</td><td>1.14×</td><td>1.36×</td></tr></table>

Table 12: Wall-clock per image and speed-up, JiT-L/16 at $2 5 6 ^ { 2 }$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>0.383</td><td>0.365</td><td>0.339</td><td>0.314</td><td>0.291</td><td>0.259</td><td>0.197</td></tr><tr><td></td><td>ToMe</td><td>0.371</td><td>0.374</td><td>0.354</td><td>0.343</td><td>0.332</td><td>0.319</td><td>0.279</td></tr><tr><td></td><td>FeatSim</td><td>0.372</td><td>0.377</td><td>0.350</td><td>0.333</td><td>0.314</td><td>0.293</td><td>0.249</td></tr><tr><td></td><td>BudgetPix</td><td>0.384</td><td>0.368</td><td>0.342</td><td>0.315</td><td>0.292</td><td>0.261</td><td>0.195</td></tr><tr><td>speed-up</td><td>Base model</td><td>0.97×</td><td>1.02×</td><td>1.09×</td><td>1.18×</td><td>1.27×</td><td>1.43×</td><td>1.88×</td></tr><tr><td></td><td>ToMe</td><td>1.00×</td><td>0.99×</td><td>1.05×</td><td>1.08×</td><td>1.12×</td><td>1.17×</td><td>1.33×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.98×</td><td>1.06×</td><td>1.11×</td><td>1.18×</td><td>1.27×</td><td>1.49×</td></tr><tr><td></td><td>BudgetPix</td><td>0.97×</td><td>1.01×</td><td>1.09×</td><td>1.18×</td><td>1.27×</td><td>1.42×</td><td>1.91×</td></tr></table>

Table 13: Wall-clock per image and speed-up, JiT-B/16 at $2 5 6 ^ { 2 }$
<table><tr><td>measure</td><td>method</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>25%</td></tr><tr><td>seconds per image</td><td>Base model</td><td>0.121</td><td>0.119</td><td>0.113</td><td>0.106</td><td>0.098</td><td>0.089</td><td>0.072</td></tr><tr><td></td><td>ToMe</td><td>0.118</td><td>0.130</td><td>0.122</td><td>0.119</td><td>0.113</td><td>0.111</td><td>0.097</td></tr><tr><td></td><td>FeatSim</td><td>0.117</td><td>0.122</td><td>0.116</td><td>0.112</td><td>0.107</td><td>0.101</td><td>0.091</td></tr><tr><td></td><td>BudgetPix</td><td>0.123</td><td>0.122</td><td>0.115</td><td>0.108</td><td>0.099</td><td>0.091</td><td>0.074</td></tr><tr><td>speed-up</td><td>Base model</td><td>0.97×</td><td>0.98×</td><td>1.03×</td><td>1.11×</td><td>1.19×</td><td>1.32×</td><td>1.63×</td></tr><tr><td></td><td>ToMe</td><td>0.99×</td><td>0.90×</td><td>0.96×</td><td>0.98×</td><td>1.04×</td><td>1.06×</td><td>1.21×</td></tr><tr><td></td><td>FeatSim</td><td>1.00×</td><td>0.96×</td><td>1.01×</td><td>1.04×</td><td>1.09×</td><td>1.16×</td><td>1.28×</td></tr><tr><td></td><td>BudgetPix</td><td>0.95×</td><td>0.96×</td><td>1.02×</td><td>1.09×</td><td>1.18×</td><td>1.28×</td><td>1.58×</td></tr></table>

## B.5 BUDGET MODE AGAINST THRESHOLD MODE (JIT-L/32)

In budget mode δ is bisected so that every image gets at most the requested token count; in threshold mode δ is fixed and each image takes the count its content gives. Threshold mode sweeps δ from −1 to +4 on JiT-L/32 at $5 1 2 ^ { 2 }$ , with the images, noise and protocol of Table 3; budget mode is that table’s points, with 40, 30, 20 and 10 % added. Figure 37 draws both down to a fifth of the grid, with bars of one standard deviation of the token count across images, and Table 14 lists every point. Read at the same mean token count the two modes lie on nearly one curve. What threshold mode gives up is the per-image guarantee, which is why every other number in the paper is in budget mode; see App. D.9 for a pooled budget. Figure 38 shows the layout as the budget falls for a penguin before a plain wall (top) and a textured toyshop (bottom).

![](images/10c5e788c793faf79149a681b63a73ee1e8b3111fa781905b80c50136f765884.jpg)  
Figure 37: Budget mode against threshold mode, FID-50k on JiT-L/32 at $5 1 2 ^ { 2 }$

![](images/da90da67ed6d52dab7bca8e996cd1e60f9c2a6965d6824cb21c82f4e70f9954a.jpg)  
Figure 38: The layout as the budget falls, JiT-L/32 at $5 1 2 ^ { 2 }$ , budget mode.

Table 14: Budget mode against threshold mode, JiT-L/32 at $5 1 2 ^ { 2 } { \colon }$ every point of Figure 37.
<table><tr><td>budget mode, budget</td><td>100%</td><td>90%</td><td>80%</td><td>70%</td><td>60%</td><td>50%</td><td>40%</td><td>30%</td><td>25%</td><td>20%</td><td>10%</td></tr><tr><td>tokens (% of grid)</td><td>100.0</td><td>89.5</td><td>80.1</td><td>69.5</td><td>60.2</td><td>49.6</td><td>39.1</td><td>29.7</td><td>25.0</td><td>19.1</td><td>9.8</td></tr><tr><td>FID-50k↓</td><td>2.71</td><td>2.94</td><td>3.20</td><td>3.69</td><td>4.41</td><td>5.57</td><td>7.33</td><td>9.73</td><td>11.54</td><td>18.12</td><td>44.62</td></tr><tr><td>Inception score ↑</td><td>341</td><td>336</td><td>328</td><td>316</td><td>302</td><td>281</td><td>256</td><td>228</td><td>206</td><td>164</td><td>76</td></tr><tr><td>threshold mode, δ</td><td>-1</td><td>-0.5</td><td>0</td><td>+0.5</td><td>+1</td><td>+1.5</td><td>+1.9</td><td>+2</td><td>+2.7</td><td>+3</td><td>+4</td></tr><tr><td>tokens (% of grid)</td><td>91.6</td><td>86.3</td><td>78.8</td><td>68.7</td><td>56.0</td><td>41.4</td><td>30.3</td><td>28.0</td><td>18.8</td><td>17.1</td><td>9.2</td></tr><tr><td>s.d. across images (points)</td><td>10.9</td><td>13.7</td><td>16.2</td><td>18.0</td><td>18.2</td><td>15.6</td><td>11.3</td><td>10.0</td><td>4.7</td><td>4.9</td><td>3.5</td></tr><tr><td>FID-50k ↓</td><td>2.85</td><td>2.95</td><td>3.21</td><td>3.88</td><td>5.21</td><td>7.63</td><td>10.71</td><td>11.55</td><td>18.04</td><td>21.33</td><td>47.13</td></tr><tr><td>Inception score ↑</td><td>337</td><td>334</td><td>322</td><td>308</td><td>285</td><td>251</td><td>215</td><td>206</td><td>158</td><td>140</td><td>64</td></tr></table>

## B.6 ONE-STEP GENERATION (PIXEL MEANFLOW)

Table 15 completes Figure 10; see App. A.5 for samples. The parent is the one-step pixel MeanFlow model (Lu et al., 2026) pMF-L/16, class-conditional on ImageNet at $2 5 6 ^ { 2 }$ (256 tokens; token scales of 16, 32 and 64 pixels), fine-tuned with its own objective (see App. C.4). One step leaves no trajectory to read a layout from, so each image takes the entropy quadtree of one of 20 training images of its class, picked by the image index and bisected to the budget; guidance is the released model’s own at every budget. FID-50k follows pMF, 50 images per class against JiT’s reference statistics at $2 5 6 ^ { 2 }$ (the released L/16 reads 2.54 here and 2.52 in its paper). The base model is the released weights on our layouts, without training; ToMe and FeatSim are the JiT implementations ported to pMF.

Table 15: One-step generation, pMF-L/16 at $2 5 6 ^ { 2 }$ : FID-50k and Inception score.
<table><tr><td>metric</td><td>method tokens</td><td>100% 256</td><td>90% 230</td><td>80% 205</td><td>70% 179</td><td>60% 154</td><td>50% 128</td><td>40% 102</td><td>30% 77</td><td>25% 64</td></tr><tr><td>FID-50k↓</td><td>Base model</td><td>2.54</td><td>6.91</td><td>118.76</td><td>242.62</td><td>257.20</td><td>244.95</td><td>236.82</td><td>253.49</td><td>233.64</td></tr><tr><td></td><td>ToMe</td><td>2.54</td><td>3.06</td><td>5.93</td><td>14.66</td><td>36.04</td><td>70.62</td><td>109.40</td><td>122.59</td><td>149.41</td></tr><tr><td></td><td>FeatSim</td><td>2.54</td><td>5.71</td><td>20.44</td><td>57.37</td><td>100.26</td><td>134.35</td><td>180.26</td><td>194.43</td><td>192.94</td></tr><tr><td></td><td>BudgetPix</td><td>2.65</td><td>2.65</td><td>2.77</td><td>3.06</td><td>3.46</td><td>3.94</td><td>4.50</td><td>4.96</td><td>5.44</td></tr><tr><td>Inception score ↑</td><td>Base model</td><td>263</td><td>226</td><td>20</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td>3</td></tr><tr><td></td><td>ToMe</td><td>263</td><td>240</td><td>203</td><td>150</td><td>84</td><td>36</td><td>14</td><td>11</td><td>6</td></tr><tr><td></td><td>FeatSim</td><td>263</td><td>209</td><td>123</td><td>49</td><td>20</td><td>11</td><td>4</td><td>3</td><td>3</td></tr><tr><td></td><td>BudgetPix</td><td>271</td><td>264</td><td>257</td><td>251</td><td>241</td><td>238</td><td>232</td><td>231</td><td>224</td></tr></table>

## C DETAILS OF THE METHOD AND THE EXPERIMENTS

Details that Sections 3 and 4 leave out.

## C.1 ARCHITECTURE OF THE ADDED MODULES

Table 16 lists how each family embeds and decodes a coarse token of side np $( n = 2 , 4 ; p$ is the released patch and D the backbone width), as built in the code; Sections 3.2 and 3.3 describe the modules. Base tokens $( n = 1 )$ use the released embedding and head unchanged, and each coarse scale has its own Patch Aggregator. PixelDiT’s released pixel stage decodes every pixel at every budget, which sets its cost floor, so instead of a pixel-space refiner, CellRefine corrects each cell’s conditioning of that stage.

Table 16: Architecture of the added modules, every family.
<table><tr><td></td><td>JiT  $2 5 6 ^ { 2 } \ a n d 5 1 2 ^ { 2 }$ </td><td> $\mathrm { p M F - L } / 1 6$   $2 5 6 ^ { 2 }$ </td><td>MiniT2I  $5 1 2 ^ { 2 }$ </td><td>PixelDiT-T2I 1024²</td></tr><tr><td>width  $D$  cell embedding</td><td> $7 6 8 \ : ( \mathrm { B } ) , 1 0 2 4 \ : ( \mathrm { L } )$ </td><td>1024 p × p convolution to 128 channels, then 1 × 1 convolution to D</td><td>768 (B), 1248 (L)</td><td>1536 linear map of the flattened 16 × 16 × 3 cell</td></tr><tr><td>(released) downscale path</td><td colspan="4">released embedding of the none</td></tr><tr><td>split path: scale mixer</td><td>patch resized to p × p  $n ^ { 2 }$  cell embeddings +  $1 \times 1$  learned positions; conv to 256, GELU,</td><td> $n ^ { 2 }$ </td><td>cell embeddings + learned positions; 1 × 1 conv to 256, depthwise n × n conv, linear to  $D ;$  no activation</td><td></td></tr><tr><td>coarse token</td><td>to  $D$  downscale path + scale mixer + scale embedding</td><td>mean of the</td><td> $n ^ { 2 }$  cell embeddings + scale mixer</td><td></td></tr><tr><td>position</td><td>sin-cos embedding and RoPE at the token centre</td><td>mean of the covered cells&#x27; learned positions; RoPE at the token centre</td><td>sin-cos embedding and RoPE at the token centre</td><td>RoPE at the token centre</td></tr><tr><td>upscale path</td><td colspan="4">- released head&#x27;s p × p prediction, bilinearly upsampled to np × np</td></tr><tr><td>refiner tokens</td><td colspan="4">4 × 4-pixel pieces of the p × p prediction: 16  $( p = 1 6 ) \mathrm { o r } 6 4 ( p = 3 2 )$ </td></tr><tr><td>refiner blocks refiner output</td><td colspan="4">2 transformer blocks, width 64, 4 heads, adaLN on the token&#x27;s final hidden state only linear to  $4 \cdot 4 \cdot 3 n ^ { 2 }$  values per piece, pixel shuffle by n: a residual on the upscale path</td></tr><tr><td></td><td>coarse scale</td><td>coarse scale and branch coarse scale</td><td></td><td>each cell&#x27;s 1536-d conditioning, then the released pixel stage coarse scale</td></tr><tr><td>one refiner per zero-initialised</td><td colspan="4">(u; v in training) last layer of the mixer, last layer of the mixer, refiner adaLN and output</td></tr><tr><td></td><td>scale embedding, refiner adaLN and output 2.39M (B/16), 2.40M (B/32), 3.06M (L/16,</td><td>4.77M</td><td>2.26M (B), 3.50M (L)</td><td>4.37M</td></tr><tr><td>added parameters</td><td colspan="4">L/32)</td></tr></table>

## C.2 TRAINING OBJECTIVE

The fine-tuning objective of Section $3 . 4 , w ( t ) \| x - \hat { x } \| _ { 2 } ^ { 2 }$ , keeps the weighting each parent was pretrained with, so no comparison gains from a change of loss. The parents regress velocity; with $z _ { t } = t x + ( 1 - t ) \varepsilon$ , the target and predicted velocities are $( x - z _ { t } )$ and $( \hat { x } - z _ { t } )$ over the same factor, so

$$
\mathcal { L } ( \theta ) = \mathbb { E } \bigg \| \frac { x - z _ { t } } { \operatorname* { m a x } ( 1 - t , t _ { \epsilon } ) } - \frac { \hat { x } - z _ { t } } { \operatorname* { m a x } ( 1 - t , t _ { \epsilon } ) } \bigg \| _ { 2 } ^ { 2 } = \mathbb { E } \Big [ \frac { 1 } { \operatorname* { m a x } ( 1 - t , t _ { \epsilon } ) ^ { 2 } } \| x - \hat { x } \| _ { 2 } ^ { 2 } \Big ] ,\tag{3}
$$

and $w ( t ) = \operatorname* { m a x } ( 1 - t , t _ { \epsilon } ) ^ { - 2 }$ , where each parent’s own clip $t _ { \epsilon }$ keeps the weight finite as $t  1$ PixelDiT regresses $\varepsilon - x$ at noise level $\sigma = 1 - t \colon$ the same objective with $w \stackrel { \textstyle } { = } \sigma ^ { - 2 }$ , and its own sampling of $\mathit { \Pi } _ { \sigma }$ (logit-normal, shift 4.0).

## C.3 ZERO-INITIALISATION CHECK

Every added path starts at zero, so training starts from the released model (Section 4.1). Table 17 checks this on the seven parents: max $| \Delta |$ is the largest absolute difference from the released model’s output on the same inputs, and FID is dense-grid FID-50k on ImageNet for JiT and held-out FID for MiniT2I and PixelDiT (see App. C.5).

Table 17: Zero-initialisation check, every parent.
<table><tr><td>quantity</td><td>JiT-B/16  $2 5 6 ^ { 2 }$ </td><td>JiT-L/16  $2 5 6 ^ { 2 }$ </td><td>JiT-B/32  $5 1 2 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td><td>JiT-L/32 MiniT2I-B/16 MiniT2I-L/16 PixelDiT-T2I</td><td></td><td></td></tr><tr><td>max |∆| at initialisation</td><td> $1 . 9 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td></td><td> $5 1 2 ^ { 2 }$ </td><td> $5 1 2 ^ { 2 }$ </td><td> $1 0 2 4 ^ { 2 }$ </td></tr><tr><td></td><td></td><td> $1 . 6 \times 1 0 ^ { - 6 }$ </td><td> $2 . 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 9 \times 1 0 ^ { - 6 }$ </td><td> $2 . 0 \times 1 0 ^ { - 6 }$ </td><td> $4 . 5 \times 1 0 ^ { - 6 }$ </td><td> $9 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>FID, released model</td><td>3.61</td><td>2.71</td><td>4.12</td><td>2.66</td><td>24.6</td><td>24.8</td><td>43.7</td></tr><tr><td>FID, adapted model at step 0 FID, adapted model after training</td><td>3.61 3.60</td><td>2.71 2.56</td><td>4.12 3.94</td><td>2.66 2.71</td><td>24.6 17.1</td><td>24.8 22.9</td><td>43.7 36.2</td></tr></table>

## C.4 FINE-TUNING AND SAMPLING SETTINGS

Tables 18 and 19 give each parent’s fine-tuning recipe, read from its run, and the sampler behind its benchmark scores. Every parent starts from its released checkpoint (for JiT, its EMA weights), all weights are fine-tuned, and one checkpoint per parent serves every budget. MiniT2I-L’s 120K mix is BLIP3o-60k (58,859 images), DALL-E 3 (19,016) and ShareGPT-4o-Image (41,355), with 1,024 held out. The held-out FIDs use 50 steps (see App. C.5), and the wall-clock runs the settings of Table 21.

Shared by the seven parents of Section 4.1. Layout: the entropy quadtree on a 512-bin luma histogram (entropy in [0, 9] bits), with $\tau _ { 2 p } = 6 . 0$ and $\tau _ { 4 p } = 4 . 0$ bits; the offset $\delta \sim \mathcal { U } ( 0 , 2 )$ is drawn per image and step; a = 0.15 of the images train on the dense grid, and $b = 0 . 1 0$ of the rest (8.5 % of all images) on the random layout, whose cell entropies are drawn uniformly in [0, 9] bits (Section 3.4). No image augmentation: a centre crop and a resize fix the geometry. Optimisation: AdamW without weight decay, a linear warm-up then a constant learning rate, the added modules at 8× the backbone rate, EMA 0.9999 in fp32 (the evaluated weights), bf16 autocast with fp32 master weights, seed 0, four GPUs. Sampling: the first $n _ { w }$ steps run on the dense grid, and the layout is recomputed every m = 5 steps.

Table 18: Fine-tuning and sampling settings, JiT parents (ImageNet-1k).
<table><tr><td>setting</td><td>JiT-B/16 2562</td><td>JiT-B/32  $5 1 2 ^ { 2 }$ </td><td>JiT-L/16  $2 5 6 ^ { 2 }$ </td><td>JiT-L/32  $5 1 2 ^ { 2 }$ </td></tr><tr><td>patch / token scales parameters, base + added</td><td>16 / 16, 32, 64 131.3M + 2.39M</td><td>32 / 32, 64, 128 133.4M + 2.40M</td><td>16 / 16, 32, 64 459.1M + 3.06M</td><td>32 / 32, 64, 128 461.8M + 3.06M</td></tr><tr><td rowspan="5">steps / epochs / images seen global batch (per GPU × GPUs × accum.) learning rate, backbone / added learning-rate warm-up</td><td colspan="2">142,987 / 100 / 128.1M 896 (112 × 4 × 2)</td><td colspan="2">71,494 /50 / 64.1M</td></tr><tr><td colspan="5"></td></tr><tr><td colspan="5"> $\underline { { 3 . 5 \times 1 0 ^ { - 5 } / 2 . 8 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td colspan="5">AdamW (0.9, 0.95), clip 3.0 1.0</td></tr><tr><td colspan="5">1.0 2.0</td></tr><tr><td rowspan="2">timestep t label dropout</td><td colspan="5">- logit-normal(−0.8, 0.8)</td></tr><tr><td colspan="5">-0.1</td></tr><tr><td rowspan="2">sampler, guidance dense warm-up steps nw FID samples</td><td colspan="5">50 Heun, 3.0 on t ∈ [0.1, 1] 10</td></tr><tr><td colspan="5">10 5 50k, class-balanced, against training-set statistics</td></tr></table>

Table 19: Fine-tuning and sampling settings, text-to-image parents.
<table><tr><td rowspan="2">setting</td><td rowspan="2">MiniT2I-B/16</td><td rowspan="2">MiniT2I-L/16</td><td rowspan="2">PixelDiT-T2I 1024²</td></tr><tr><td colspan="2"> $5 1 2 ^ { 2 }$ </td></tr><tr><td rowspan="3">patch / token scales parameters, base + added training data</td><td>258.1M + 2.26M</td><td>- 16 / 16, 32, 64 911.8M + 3.50M</td><td>1302.5M + 4.37M</td></tr><tr><td>CC12M, LLaVA recaptions, 1M images</td><td>CC12M (1M), then the 120K mix</td><td>PD12M, 200K images</td></tr><tr><td>100k / 25.6M</td><td>40k (24k CC12M, then 16k mix) / 10.2M</td><td>30k / 0.96M</td></tr><tr><td>global batch (per GPU × GPUs × accum.)</td><td>256 (16 × 4 × 4)</td><td>256 (4 × 4 × 16)</td><td> $3 2 \ : ( 4 \times 4 \times 2 )$ </td></tr><tr><td rowspan="3">learning rate, backbone / added learning-rate warm-up optimiser</td><td colspan="2">10−5 /8 × 10−5</td><td rowspan="3"> $\begin{array} { c } { { 2 \times 1 0 ^ { - 5 } / 1 . 6 \times 1 0 ^ { - 4 } } } \\ { { 2 \mathrm { k } \mathrm { s t e p s } } } \end{array}$  AdamW (0.9, 0.999), ∈ 10−15</td></tr><tr><td colspan="2">1k steps AdamW (0.9, 0.95), € 10−8, clip 3.0</td></tr><tr><td colspan="2">x-prediction, noise scale 2.0, t ~</td></tr><tr><td>diffusion law</td><td colspan="2">logit-normal(−0.8, 0.8)</td><td rowspan="3">flow matching, velocity target, shift 4.0, σ ~ logit-normal(0, 1) 0.1, null embedding Gemma-2-2B-it, 300 tokens</td></tr><tr><td rowspan="2">caption dropout text encoder (frozen)</td><td colspan="2">0.1, attention mask zeroed</td></tr><tr><td colspan="2">flan-T5-large, 256 tokens</td></tr><tr><td>memory</td><td></td><td>none</td><td>gradient checkpointing, EMA on CPU</td></tr><tr><td rowspan="2">sampler, guidance</td><td colspan="2">100 Euler, 5.0</td><td rowspan="2">25 DPM-Solver++(2M) (Lu et al., 2025), 4.5</td></tr><tr><td colspan="2"></td></tr><tr><td>dense warm-up steps nw</td><td colspan="2">5</td><td></td></tr><tr><td>held-out set</td><td colspan="2">5k CC12M prompts</td><td>2k PD12M prompts</td></tr></table>

Pixel MeanFlow. pMF-L/16 (see App. B.6) is fine-tuned with pMF’s own objective, ported from its code (the MeanFlow identity with its forward-mode derivative, x-prediction with the guidance in the target, the auxiliary v-loss, and LPIPS and ConvNeXt-V2 perceptual terms for $t < 0 . 8 .$ , in pMF’s time, where $t \ : = \ : 1$ is pure noise), with one layout per image held fixed across the step’s network calls. Training layouts come from the clean image: the dense grid with probability 0.30, otherwise the entropy quadtree at $\delta \sim \mathcal { U } ( 0 , 2 )$ , with random entropy maps for 10 % of those (7 % of all images). A merged token is the mean of its base-patch embeddings plus a zero-initialised mixer, and each coarse scale has a zero-initialised pixel refiner on each output branch (u, and v in training; 4.77M added parameters in all), so training starts from the released model. AdamW (0.9, 0.95) without weight decay, learning rate $1 0 ^ { - 5 }$ (8× on the added modules), 500 warm-up steps then constant, gradient clipping at 20, EMA 0.9995. It trains for 64,000 steps at a global batch of 91 (7 GPUs × 13; 52 on 4 GPUs for steps 40,000–58,000), 5.1M images, with the learning rate cooled linearly to zero over the last 8,000. Sampling keeps the released guidance, 7.0 on $t \in [ \bar { 0 } . 2 , 0 . 7 ]$

## C.5 EVALUATION PROTOCOL

Methods compared. The released model is the parent as shipped, at its dense grid; the base model is the released weights run at the same budget through our sampler, without training. ToMe merges tokens by bipartite soft matching on the image keys before RoPE, averaged over heads (sizeweighted, with proportional attention and the same r in every block; condition tokens, text or JiT’s in-context class tokens, are never merged). FeatSim makes one similarity merge down to B tokens early in the network (block 3 on MiniT2I). Both wrap the released weights, keep the sampler on the dense grid, and on MiniT2I write the merged tokens back to the dense grid for the last two blocks. RTI runs its released adapter (40k steps), which exists for MiniT2I-B and MiniT2I-L only. On PixelDiT ours is compared with the base model. Every method of a parent shares the prompts or labels, the noise, the solver, the steps, the guidance and the attention arithmetic.

Class-conditional. FID and Inception score follow JiT: Inception-v3 from torch-fidelity on uint8 images at native resolution, against JiT’s ImageNet-1k (Deng et al., 2009) training-set statistics at 256<sup>2</sup> and 512<sup>2</sup>; the Inception score uses 10 splits. FID-50k uses 50 images per class; the ablations (see App. D) use FID-10k, 10 per class. Each FID is a single run.

Text-to-image. GenEval (553 prompts, 4 images each) counts an image as correct only if every element of its prompt is rendered, averaged over six tasks; DPG-Bench (1,065 prompts, 4 images each) is the mean mPLUG-large answer; CLIP score (ViT-L/14), PickScore and ImageReward are means over the 1,632 PartiPrompts, one image each. Held-out FID, used at the dense grid, compares 5k generated with 5k held-out real images for CC12M (Changpinyo et al., 2021) and 2k with 2k for PD12M, at native resolution, with 50 steps at guidance 3.0 (MiniT2I-B), 5.0 (MiniT2I-L) and 2.75 (PixelDiT).

Paired metrics. Each image at a reduced budget is scored against the same method’s image at the dense grid, from the same prompt or label and noise, by PSNR, SSIM, LPIPS and the cosine similarity of CLIP and DINOv2 embeddings. On MiniT2I-L each cell is the mean over the 553 GenEval prompts, one image each, with its 95 % interval; on JiT it is the mean over 2,000 classbalanced images at guidance 3.0 and 50 steps, and PSNR is also split between the pixels the layout merged into coarse tokens and those it kept.

## C.6 USER STUDY AND VLM-AS-JUDGE

Images. Both studies use the images of Figure 7(b): MiniT2I-L/16 at 512<sup>2</sup>, five methods at ten budgets, and the released model. Image j comes from prompt j and noise seeded by j, so two images in one comparison differ by method and budget alone. The prompts are 100 GenEval prompts drawn at random (22 colours, 21 two-object, 20 colour-attribution, 14 position, 13 single-object, 10 counting); the user study uses 20 of them, chosen from the released model’s images alone. Both studies are scored alike: “no visible difference” counts as 5 %, a tie as one half, and 95 % intervals are bootstrapped over prompts. Table 20 gives every number of Figure 9, with the intervals in brackets.

User study. Nine anonymous online forms of 20 questions, one image per question, in random order; each form counts its first 50 participants. The instructions ask for faithfulness only, not to prefer a sharper or prettier image that has changed what the reference shows.

User study: where the drop is first seen (Figure 9(a)). Five forms, one per method, show the method’s own image at the dense grid above its budgets from 90 to 10 %, and ask for the first budget with a visible difference. Figure 39 shows the start of one of them and its first question.

## Image comparison study: when does a difference become visible?

Thank you for helping. This short study (20 questions, about 5 minutes) looks at images made by text-to-image models at reduced compute. Each guestion shows a REFERENCE image on top, made at the full budget. Below it are nine images made by the same model from the same prompt and the same starting noise, at decreasing budgets from 90 % to 10 %. Read the grid from 90 % down to 10 % and choose the FIRST budget at which you notice a visible difference from the reference: a change of content or layout, or clearly lost detail. If even the 10 % image looks the same as the reference, choose "No visible difference". Please use a computer with a large screen and look at every image carefully. There are no right answers, nothing on the page says which method made which image, and your answers are anonymous.

![](images/b6716b45034a86602ccfc1f3ca39aec5b9b9f2163a37149448d5cb8fe4565a5d.jpg)  
Figure 39: User-study form on where a difference is first seen: its start and first question.

User study: which is more faithful (Figure 9(b)). Four forms, one per budget (80, 60, 40 and 20 %), show the released image above ours and RTI as A and B, ours on the left in half the questions, and ask which is more faithful, or “equally faithful”. Figure 40 shows the start of one of them and its first question.

## Image comparison study: which image is more faithful?

Thank you for helping. This short study (20 questions, about 4 minutes) looks at images made by text-to-image models at reduced compute. Each question shows a REFERENCE image on top and two candidates, A and B, below. Both candidates try to reproduce the reference from the same prompt and the same starting noise, at the same reduced budget Choose the candidate that is MORE FAITHFUL to the reference: the same scene, the same composition, the same objects in the same places, the same texture and detail. Judge faithfulness only: do not prefer an image for being sharper or prettier if it has changed what the reference shows. Please use a computer with a large screen and look at every image carefully. There are no right answers, nothing on the page says which method made which image, and your answers are anonymous.

![](images/77480256ebcf7dd2620969bdc1299c8535e30a449dd5fa4937e305d1dee6493c.jpg)  
Figure 40: User-study form on which image is more faithful: its start and first question.

User study: reading the numbers. Participants chose A in 48 to 54 % of the answers that picked a side, so position did not drive them. Dropping the 17 who gave one answer to 18 or more of their 20 questions moves no cut by more than 2.4 points and no share at all.

VLM-as-Judge. The judge is Gemini 3.8 Flash at temperature 0, thinking level low, one image set per call, answering with a single number or letter.

VLM-as-Judge: where the drop is first seen (Figure 9(a)). The judge sees a method’s own image at the dense grid, then one picture tiling its ten budgets in order, after this text:

The first image is a REFERENCE, generated by this same method at its full token budget from the prompt: “⟨prompt⟩”

The second image is a grid of ten images from the same prompt and the same starting noise, generated by the SAME method at decreasing token budgets. The budgets are printed above each tile and run from 100% (top left) down to 10% (bottom right), in order; the 100% tile is the reference itself.

Reading the grid in order from 100% downwards, at which budget do you FIRST notice a visible difference from the reference (a change of content, layout, or clearly lost detail)? If even the 10% tile is indistinguishable from the reference, answer 0.

Reply with exactly one number from this list and nothing else: 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0.

With the method’s own dense image as reference, the answer is what the budget costs, not what the method already changed at 100 %. One call per method and prompt: 500 judgments.

VLM-as-Judge: which is more faithful (Figure 9(b)). The judge sees the released image, then ours and one baseline at the same budget, after this text:

The first image is a REFERENCE, generated by an image model at its full token budget from the prompt: “⟨prompt⟩”

Two candidate images follow, A then B. Each was generated by a different method at a reduced token budget from the same prompt and the same starting noise, trying to reproduce the reference.

Which candidate is MORE FAITHFUL to the reference: the same scene, the same composition, the same objects in the same places, the same texture and detail? Judge faithfulness to the reference only. Do not reward an image for being sharper or more pleasing than the reference if it has changed what the reference shows.

Reply with exactly one character and nothing else: A, B, or T if the two are equally faithful.

The three images are resized and JPEG-encoded identically. Each pair is judged in both left/right orders, as independent calls: 7,193 judgments over four baselines, nine budgets and 100 prompts.

VLM-as-Judge: reading the numbers. Position bias, the share of pairs answered on the same side in both orders, is 8.7 %, below the 15 % rule of thumb. The judge never answered T.

Table 20: User study and VLM-as-Judge, MiniT2I-L/16 at 512<sup>2</sup>: the numbers of Figure 9.
<table><tr><td>cut before a difference is seen (%)</td><td>Base model</td><td>ToMe</td><td>FeatSim</td><td>RTI</td><td>BudgetPix</td></tr><tr><td>people, 20 prompts</td><td>30.3 [29.2, 31.4]</td><td>32.5 [30.1, 35.1]</td><td>23.3 [21.5, 25.3]</td><td>38.0 [31.8, 44.0]</td><td>51.8 [48.0, 55.7]</td></tr><tr><td>judge, the same 20 prompts</td><td>18.5 [14.5, 22.5]</td><td>20.0 [16.5, 24.0]</td><td>12.5 [11.0, 14.5]</td><td>24.0 [14.5, 35.0]</td><td>43.5 [35.0, 51.0]</td></tr><tr><td>judge, all 100 prompts</td><td>15.4 [14.1, 16.9]</td><td>15.9 [14.3, 17.5]</td><td>11.7 [11.0, 12.4]</td><td>21.1 [16.9, 25.8]</td><td>36.2 [31.9, 40.4]</td></tr><tr><td>prefers ours to RTI (%)</td><td>80%</td><td></td><td></td><td>40%</td><td>20%</td></tr><tr><td>people, 20 prompts</td><td>63.5 [52.7, 74.7]</td><td>63.4 [52.2, 74.2]</td><td></td><td>57.2 [45.3, 69.5]</td><td>62.3 [52.8, 72.2]</td></tr><tr><td>answers for ours / RTI / equal</td><td>54 / 27 / 19</td><td>56 / 29 / 16</td><td></td><td>50 / 36 / 14</td><td>54 / 29 / 17</td></tr><tr><td>judge, the same 20 prompts</td><td>77.5 [60.0, 95.0]</td><td>72.5 [55.0, 87.5]</td><td></td><td>50.0 [30.0, 70.0]</td><td>75.0 [57.5, 90.0]</td></tr><tr><td>judge, all 100 prompts</td><td>75.5 [67.5, 83.0]</td><td>69.0 [61.0, 76.5]</td><td></td><td>54.5 [46.0, 63.0]</td><td>61.5 [53.5, 69.5]</td></tr></table>

## C.7 WALL-CLOCK MEASUREMENT

Timing. Every cell was timed on one H100, with the overhead of the measurement rather than of the method removed: per-forward GPU-to-CPU synchronisations, layout work repeated on every forward, and kernel-launch cost, with one CUDA graph per layout. The images are bit-identical to those of each method’s released code; ours and the base model also resample merged tokens with two fp32 matrix products (identical images on MiniT2I, ∼55 dB PSNR on JiT). Speed-up is over the released model at its dense grid, timed the same way. Table 21 lists the conditions, which every method of a parent shares.

Two checks. ToMe, FeatSim and RTI are not deterministic: two runs of their released code agree only to 20, 27 and 34 dB PSNR, so for them the images match up to this noise. PixelDiT does not replay correctly as a CUDA graph (23 dB against its own output), so it is timed without graphs, with bit-identical images.

Table 21: Conditions of the wall-clock measurements, every parent.
<table><tr><td>setting</td><td>JiT-B/32, L/32 (5122)</td><td>JiT-B/16, L/16 (2562)</td><td>MiniT2I-B/16</td><td>MiniT2I-L/16</td><td>PixelDiT-T2I</td></tr><tr><td>resolution dense grid</td><td>512 × 512 256 tokens (32 px)</td><td rowspan="2">256 × 256 256 tokens (16 px)</td><td colspan="2">512 × 512 1024 tokens (16 px)</td><td rowspan="2">1024 × 1024 4096 tokens (16 px)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>token scales of the layout</td><td rowspan="2">32 / 64 / 128 px</td><td rowspan="2"></td><td colspan="3">16 / 32 / 64 px</td></tr><tr><td>budgets 100, 90, 80, 70, - 256, 230, 205, 179, 154, 128, 64 tokens</td><td></td><td>1024, 922, 819, 717, 614, 512, 256 tokens</td><td>4096, 3686, 3277, 2867, 2458, 2048,</td></tr><tr><td>conditioning</td><td colspan="2">class label, random</td><td colspan="3">text, pre-encoded</td></tr><tr><td>batch (images per sampler call)</td><td colspan="2">50</td><td colspan="2">32 20</td><td>8</td></tr><tr><td>warm-up, discarded</td><td colspan="2">one call per cell for the released model and BudgetPix, one per process for the others</td><td colspan="2"></td><td></td></tr><tr><td>timed calls per cell</td><td colspan="2">released, BudgetPix, all of released, BudgetPix: 3 JiT-B/32: 3 (median); (median); others: 2</td><td colspan="2">released, BudgetPix: 2 (median); others: 1</td><td>released, BudgetPix: 2 (median); base model: 1</td></tr><tr><td>solver, steps</td><td colspan="2">others: 2 Heun, 50 (last step Euler)</td><td colspan="2">Euler, 100</td><td>DPM- Solver++(2M), 50</td></tr><tr><td>guidance scale network forwards per</td><td colspan="2">3.0 on t ∈ [0.1, 1] 4 (2 Heun stages × cond./uncond.)</td><td colspan="2">3.0 5.0 2 (cond., uncond.)</td><td>4.5</td></tr><tr><td>step dense warm-up steps</td><td colspan="2">5, +1 forward (21 of 199 10, +1 forward (41 of</td><td colspan="2">5, +1 forward for the </td><td>5, +1 forward</td></tr><tr><td>nw layout</td><td colspan="2">forwards) 199 forwards)</td><td colspan="2">estimate built once, then frozen</td><td></td></tr><tr><td>numerics</td><td colspan="2">bf16 weights, bf16 autocast</td><td colspan="2">bf16 weights, bf16 autocast; RTI in pure bf16, as it ships</td><td>bf16 weights, bf16 autocast</td></tr><tr><td>attention arithmetic (all methods)</td><td colspan="2">- the model&#x27;s own</td><td colspan="2">- released: einsum + softmax —</td><td>fused SDPA, masked when a</td></tr><tr><td></td><td colspan="2"></td><td colspan="2"></td><td>reduced budget pads the layout</td></tr><tr><td>methods timed</td><td colspan="2">released, base model, BudgetPix, ToMe, FeatSim</td><td colspan="2">released, base model, – BudgetPix, ToMe, FeatSim, RTI</td><td>released, base model, BudgetPix</td></tr></table>

## D ABLATIONS ON JIT-B

Each ablation changes one factor at a time on JiT-B, on the development model (see App. D.1–D.5) or on the final JiT-B/32 (see App. D.6–D.9). All are FID-10k (see App. C.5), not comparable with the FID-50k of the main tables. The reported results use none of the budget-tuned schedules or guidance scales found here. The component ablation is Table 2.

## D.1 REFERENCE RUNS (DEVELOPMENT JIT-B)

Table 22 is the reference for the development runs (see App. D.2–D.4): the 50-epoch model on JiT-B/32 with the default sampler (uniform schedule, 50 steps, guidance 3.0). JiT-B/16 had no adapted model yet, so only its base model is shown.

Table 22: Reference runs, JiT-B: FID-10k.
<table><tr><td>parent</td><td>method tokens</td><td>100% 256</td><td>90% 230</td><td>80% 205</td><td>70% 179</td><td>60% 154</td><td>50% 128</td><td>40% 102</td><td>30% 77</td></tr><tr><td>JiT-B/32, 512²</td><td>Base model</td><td>6.53</td><td>7.44</td><td>9.45</td><td>14.96</td><td>27.61</td><td>63.91</td><td>118.04</td><td>156.99</td></tr><tr><td></td><td>BudgetPix, 50 epochs</td><td>6.47</td><td>6.93</td><td>7.36</td><td>8.33</td><td>9.36</td><td>11.18</td><td>13.89</td><td>17.33</td></tr><tr><td>JiT-B/16, 2562</td><td>Base model</td><td>6.02</td><td>6.67</td><td>8.30</td><td>13.26</td><td>26.34</td><td>67.38</td><td>116.44</td><td>152.40</td></tr></table>

## D.2 A PARALLEL PIXEL-STREAM DECODER (DEVELOPMENT JIT-B/32)

Before the refiner, the decoder was a pixel stream running in parallel with the token stream. Table 23 follows it from 10 to 50 epochs.

Table 23: Pixel-stream decoder across training length, JiT-B/32 at $5 1 2 ^ { 2 }$ : FID-10k.
<table><tr><td>parent</td><td>training tokens</td><td>100% 256</td><td>90% 230</td><td>80% 205</td><td>70% 179</td><td>60% 154</td><td>50% 128</td><td>40% 102</td><td>30% 77</td></tr><tr><td rowspan="4">JiT-B/32, 512²</td><td>10 epochs</td><td>7.48</td><td>8.16</td><td>9.33</td><td>11.30</td><td>13.87</td><td>18.19</td><td>23.95</td><td>29.94</td></tr><tr><td>20 epochs</td><td>7.45</td><td>8.15</td><td>9.10</td><td>10.91</td><td>13.08</td><td>16.27</td><td>21.80</td><td>28.29</td></tr><tr><td>30 epochs</td><td>7.40</td><td>8.05</td><td>9.05</td><td>10.72</td><td>12.43</td><td>15.95</td><td>20.50</td><td>26.78</td></tr><tr><td>50 epochs</td><td>7.33</td><td>8.01</td><td>8.94</td><td>10.36</td><td>12.15</td><td>15.17</td><td>19.41</td><td>24.96</td></tr></table>

## D.3 SAMPLING SCHEDULE AND STEP COUNT (DEVELOPMENT JIT-B/32)

Table 24 varies the timestep schedule and the step count at 128 tokens (half the grid), guidance 6.0.

Table 24: Timestep schedule against step count, JiT-B/32 at $5 1 2 ^ { 2 }$ : FID-10k.
<table><tr><td>parent</td><td>schedule</td><td>10 steps</td><td>15 steps</td><td>25 steps</td><td>50 steps</td></tr><tr><td>JiT-B/32, 512²</td><td>cosine</td><td>10.02</td><td>9.15</td><td>9.01</td><td>8.82</td></tr><tr><td></td><td>shift-3</td><td>11.71</td><td>11.21</td><td>10.61</td><td>10.10</td></tr><tr><td></td><td>uniform</td><td>12.14</td><td>11.88</td><td>10.42</td><td>9.10</td></tr></table>

## D.4 GUIDANCE SCALE ALONG THE BUDGET (DEVELOPMENT JIT-B/32)

Table 25 sweeps guidance from 3.0 to 6.0 along the budget, with the cosine schedule at 50 steps; the last row is the default sampler (uniform schedule, guidance 3.0).

Table 25: Guidance scale along the budget, JiT-B/32 at $5 1 2 ^ { 2 } { \mathrm { : } }$ : FID-10k.
<table><tr><td>parent</td><td>sampler tokens</td><td>90% 230</td><td>80% 205</td><td>70% 179</td><td>60% 154</td><td>40% 102</td></tr><tr><td>JiT-B/32, 512²</td><td>cosine, guidance 3.0</td><td>7.19</td><td>7.75</td><td>8.75</td><td>10.17</td><td>14.50</td></tr><tr><td></td><td>cosine, guidance 4.0</td><td>7.21</td><td>7.31</td><td>7.62</td><td>8.37</td><td>11.14</td></tr><tr><td></td><td>cosine, guidance 5.0</td><td>8.03</td><td>7.87</td><td>7.92</td><td>8.17</td><td>9.96</td></tr><tr><td></td><td>cosine, guidance 6.0</td><td>8.93</td><td>8.55</td><td>8.46</td><td>8.45</td><td>9.66</td></tr><tr><td></td><td>uniform, guidance 3.0 (default)</td><td>6.93</td><td>7.36</td><td>8.33</td><td>9.36</td><td>13.89</td></tr></table>

## D.5 UPPER BOUND ON A BETTER DECODER (DEVELOPMENT JIT-B/32)

Two arms change only the decoded output of the 50-epoch JiT-B/32 (Table 22). Oracle-fine gives the coarse regions their ground-truth fine detail, an upper bound on any better decoder; downsamplelanczos resamples every output to 256<sup>2</sup> and back, removing all finer detail.

Table 26: Upper bound on a better decoder, JiT-B/32 at $5 1 2 ^ { 2 } \colon$ FID-10k.
<table><tr><td>parent</td><td>arm</td><td>80%</td><td>50%</td><td>30%</td></tr><tr><td></td><td>tokens</td><td>205</td><td>128</td><td>77</td></tr><tr><td></td><td>guidance</td><td>3.0</td><td>6.0</td><td>6.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JiT-B/32, 512²</td><td>reference</td><td>7.36</td><td>9.10</td><td>11.69</td></tr><tr><td></td><td>downsample-lanczos</td><td>7.42</td><td>8.92</td><td>11.14</td></tr><tr><td></td><td>oracle-fine</td><td>7.38</td><td>9.34</td><td>11.56</td></tr></table>

## D.6 LAYOUT AND SAMPLER SETTINGS (FINAL JIT-B/32)

Table 27 varies one setting at a time on the final JiT-B/32 (512<sup>2</sup>, 100 epochs), from the sampler of every reported result (default: 50 Heun steps, guidance 3.0, five dense warm-up steps, a layout rebuilt every five steps), on the same 10k images and noise. Random placement is the entropy quadtree on permuted entropy maps, which keeps the token count; the oracle is the quadtree of a 50-step dense draft from the same noise. The noise-seed rows repeat the default with shifted noise.

Table 28 varies guidance and the timestep schedule the same way. Time shift 3.0 spends more steps at high noise, the Karras schedule more at low noise, cosine more at both ends, and two-phase only 8 of the 50 at $t \geq 0 . 7 ;$ 50 Euler steps make half the network evaluations of the others.

Table 27: Layout and sampler settings, JiT-B/32 at $5 1 2 ^ { 2 } { \vdots }$ FID-10k.
<table><tr><td>setting</td><td>value</td><td>50% 128</td><td>25% 64</td></tr><tr><td>layout signal</td><td>entropy quadtree (default)</td><td>10.99</td><td>19.79</td></tr><tr><td></td><td>random placement</td><td>19.51</td><td>23.41</td></tr><tr><td></td><td>oracle, 50-step draft</td><td>11.30</td><td>19.58</td></tr><tr><td>dense warm-up steps</td><td>0</td><td>12.44</td><td>20.41</td></tr><tr><td></td><td>2</td><td>11.20</td><td>20.33</td></tr><tr><td></td><td>5 (default)</td><td>10.99</td><td>19.79</td></tr><tr><td></td><td>10</td><td>10.36</td><td>18.09</td></tr><tr><td></td><td>20</td><td>9.68</td><td>15.73</td></tr><tr><td>re-layout interval</td><td></td><td></td><td></td></tr><tr><td></td><td>every step every 5 steps (default)</td><td>10.96</td><td>19.76 19.79</td></tr><tr><td></td><td>every 10 steps</td><td>10.99</td><td></td></tr><tr><td></td><td>every 25 steps</td><td>11.00 11.04</td><td>19.70 19.73</td></tr><tr><td></td><td>once, after the warm-up</td><td>10.96</td><td>19.76</td></tr><tr><td>Heun steps</td><td></td><td></td><td></td></tr><tr><td></td><td>25 50 (default)</td><td>10.88 10.99</td><td>18.02</td></tr><tr><td></td><td>100</td><td>10.92</td><td>19.79 19.92</td></tr><tr><td></td><td>0 (default)</td><td></td><td></td></tr><tr><td>noise seed</td><td>1</td><td>10.99</td><td>19.79</td></tr><tr><td></td><td>2</td><td>10.81 10.74</td><td>19.36 19.38</td></tr></table>

![](images/17f30ab69760987e8c0086903abfa5d05684cc9faf456cd947f83d833d6496b1.jpg)

Table 28: Guidance scale and timestep schedule, JiT-B/32 at $5 1 2 ^ { 2 } { \mathrm { : } }$ FID-10k.
<table><tr><td>setting</td><td>value</td><td>50% 128</td><td>25% 64</td></tr><tr><td>guidance</td><td>2.0</td><td>19.09</td><td>31.73</td></tr><tr><td></td><td>2.5</td><td>13.77</td><td>24.27</td></tr><tr><td></td><td>3.0 (default)</td><td>10.99</td><td>19.79</td></tr><tr><td></td><td>4.0</td><td>8.96</td><td>15.13</td></tr><tr><td></td><td>5.0</td><td>8.67</td><td>13.50</td></tr><tr><td></td><td>6.0</td><td>8.91</td><td>13.20</td></tr><tr><td></td><td>7.0</td><td>9.30</td><td>13.67</td></tr><tr><td>schedule</td><td>uniform, Heun (default)</td><td>10.99</td><td>19.79</td></tr><tr><td></td><td>time shift 3.0</td><td>14.54</td><td>24.01</td></tr><tr><td></td><td>Karras,  $\rho = 7$ </td><td>11.50</td><td>17.72</td></tr><tr><td></td><td>cosine</td><td></td><td></td></tr><tr><td></td><td>two-phase</td><td>11.65</td><td>20.40 19.48</td></tr><tr><td></td><td></td><td>11.13</td><td></td></tr><tr><td></td><td>Euler, 100 steps</td><td>10.72</td><td>19.01</td></tr><tr><td></td><td>Euler, 50 steps</td><td>11.03</td><td>18.46</td></tr></table>

## D.7 SENSITIVITY TO THE ENTROPY THRESHOLDS (FINAL JIT-B/32)

The thresholds $\tau _ { 2 p } = 6 . 0$ and $\tau _ { 4 p } = 4 . 0$ bits are taken from APT (Choudhury et al., 2026), not tuned. Table 29 shifts $\tau _ { 2 p } \mathbf { b y } + \kappa$ and $\tau _ { 4 p } \mathbf { b y } - \kappa$ and then bisects δ to the budget, so the count stays near 128 tokens (124 at $\kappa = - 1 . 5 )$ and mainly the split between 2p and 4p tokens moves; a negative κ favours 4p. The model, sampler and images are those of Table 27.

Table 29: Entropy-threshold skew, JiT-B/32 at $5 1 2 ^ { 2 } \colon$ FID-10k and Inception score at 128 tokens.
<table><tr><td>skew κ</td><td>-1.5</td><td>-1</td><td>-0.5</td><td>0 (default)</td><td>+0.5</td><td>+1</td><td>+1.5</td></tr><tr><td>FID-10k ↓</td><td>15.79</td><td>13.02</td><td>11.42</td><td>10.99</td><td>10.95</td><td>10.95</td><td>10.97</td></tr><tr><td>Inception score ↑</td><td>152</td><td>165</td><td>173</td><td>175</td><td>176</td><td>177</td><td>177</td></tr></table>

## D.8 LAYOUT EVOLUTION DURING SAMPLING (FINAL JIT-B/32)

Eight ImageNet classes from fixed noise, 50 Heun steps at guidance 3.0 with the layout rebuilt at every step, at 128 and 64 tokens, with no warm-up (the first layout is read off the pure-noise estimate) or with the default five dense steps. Figure 41 shows one sample’s estimate and layout at 128 tokens (left; top without warm-up, bottom with it) and, over the eight images, the share of the image whose token size still differs from the final layout (right; grey band: the default’s warm-up).

![](images/139c1fa69a0e0e80b005859858483edf911fdd5f329cb5be466a1a2c2e139989.jpg)  
Figure 41: Layout evolution during sampling, JiT-B/32 at $5 1 2 ^ { 2 }$

## D.9 POOLED BUDGET ACROSS A BATCH (FINAL JIT-B/32)

Every reported result gives each image the same budget. The pooled arm instead sets one offset per batch, so the batch meets the budget on average and each image takes what its content asks for (16 to 253 tokens at a mean of 128); the fixed arm is the default. Table 30 gives FID-10k as in Table 27, and PSNR (mean, with 10th / 90th percentiles) and LPIPS of 2,000 images against the dense image of the same noise, with the base model at the same token count as a control. Figure 42 shows eight samples at a mean of 128 tokens, ordered by the tokens the pool gave them (above each column): the dense image, the pooled and fixed arms, and the base model.

253 tokens  
Table 30: Pooled against fixed budget, JiT-B/32 at $5 1 2 ^ { 2 }$ : FID-10k and paired fidelity.
<table><tr><td>metric</td><td></td><td>70% 179</td><td>50% 128</td><td>25% 64</td></tr><tr><td>FID-10k↓</td><td>fixed (default)</td><td>8.20</td><td>10.99</td><td>19.79</td></tr><tr><td>PSNR ↑</td><td>pooled fixed (default)</td><td>7.79 24.57 (20.86 / 28.40)</td><td>10.82 23.22 (19.83 / 26.84)</td><td>20.39 22.14 (18.96 / 25.59)</td></tr><tr><td></td><td>pooled base model</td><td>26.35 (22.05 / 29.54) 20.58 (17.44 / 24.17)</td><td>23.49 (20.47 / 26.56) 18.68 (15.95 / 21.59)</td><td>22.08 (19.18 / 25.27) 16.02 (13.75 / 18.52)</td></tr><tr><td></td><td>fixed (default)</td><td>0.308</td><td>0.379</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LPIPS↓</td><td></td><td></td><td></td><td>0.450</td></tr><tr><td></td><td>pooled</td><td>0.271</td><td>0.370</td><td>0.457</td></tr><tr><td></td><td>base model</td><td>0.487</td><td>0.639</td><td>0.877</td></tr></table>

![](images/2290bf86993e6f5019d598dd1fbbf0915a59dd7c76cfc52a8d7ff29a09280eda.jpg)  
Figure 42: Pooled budget, JiT-B/32 at $5 1 2 ^ { 2 } ,$ , a mean of 128 tokens.