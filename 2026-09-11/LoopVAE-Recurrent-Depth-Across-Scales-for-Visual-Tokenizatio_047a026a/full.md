# LoopVAE: Recurrent Depth Across Scales for Visual Tokenization

Zhiying Lu

University of Science and Technology of China

arieseirack@mail.ustc.edu.cn

## Abstract

Hierarchical visual tokenizers typically allocate diferent processing blocks to diferent spatial scales. We ask how much of this computation can use the same parameters. LoopVAE reuses a scale- and loop-conditioned core within and across scales, while keeping resolution-changing transitions independent. A four-block core executes 28 block applications per encoder or decoder. On ImageNet-256, the 29M-parameter convolutional model reaches 0.28 rFID and 32.54 dB PSNR under an approximately 30-epoch two-stage training budget, using approximately 65% fewer parameters than the 84M reference VAEs. A non-adversarial Transformer ablation with the same execution graph finds competitive PSNR and SSIM under global sharing, although unshared blocks improve LPIPS. Targeted loop interventions show that completing the trained recurrence improves reconstruction and that even small feature updates can have substantial downstream efects. Truncation also exposes output-range errors, distinguishing useful recurrent computation from reliable early exit. Runtime profiling reveals the execution tradeof: fewer stored weights require more arithmetic and longer runtime in the tested configurations. With convolutional and Transformer operators and single- or multi-resolution latent interfaces, LoopVAE establishes recurrent depth across scales as a parameter-sharing design axis for visual tokenization.

## 1 Introduction

Latent difusion separates image generation into a visual tokenizer and a generative model over its latent representation [20, 21]. The tokenizer determines what information survives compression and how much spatial computation the generator receives. Recent work improves this interface through stronger compression [3], Transformer scaling [13], semantic supervision [2, 27, 31], and alternative reconstruction objectives [4]. We study a complementary question: how should a hierarchical tokenizer allocate its processing parameters across depth and spatial scales?

In conventional hierarchical autoencoders, changing resolution also changes the stack of blocks that processes the features. This couples tensor geometry to parameter identity. Recurrent-depth models separate executed depth from stored depth by repeatedly applying a shared core [5, 10]. Extending that principle across a visual hierarchy requires a shared update rule to operate on several feature resolutions, with separate transitions handling changes in geometry.

![](images/dbb624f61a81142c628357d580b21d1615ba710d3ddfdfb6b84493f3b4f7866a.jpg)

![](images/fd131c97bfa8f35d0317b3e22df1e946297f1695b3c8aa543eef0d34fa08a901.jpg)

![](images/717c7844df66a4c6d5f01a46b1989d8ae7c4ca7938d8fa3436995c938b4a367d.jpg)  
Figure 1 From stage-specific processing to LoopVAE-Single. (a) Conventional stages own independent parameters. (b) A conceptual scale-shared reference uses one pass per scale; it is not an evaluated baseline. (c) LoopVAE-Single reuses a four-block stack with pass counts [1, 2, 4] in each branch’s traversal order. Encoder $F _ { \theta }$ and decoder $F _ { \psi }$ have separate weights, each shared across its own scales and loops. The stem, transitions, and output head remain scale specific; posterior heads and latent input projections are omitted for clarity. Image icons are schematic, not measured reconstructions.

We introduce LoopVAE, an autoencoder organized around recurrent depth across scales. Its encoder and decoder each own a small core reused at every spatial scale and loop step. Learned scale and loop embeddings condition residual updates, while a gated input-injection path supplies the feature that entered the current scale. Downsampling, upsampling, and interface projections retain independent parameters. Thus sharing applies to the fixed-resolution processing core, not to every component of the tokenizer. A four-block core with schedule [1, 2, 4] executes 28 block applications per branch while storing only four core blocks.

This design raises three empirical questions. Can global sharing retain reconstruction quality? A 400k nonadversarial Transformer comparison holds the execution graph fixed while varying the number of stored core blocks from four to 12 or 28. Global sharing gives the highest PSNR and SSIM; fully unshared processing gives the lowest LPIPS. Separately, the final 29M-parameter CNN reaches 0.28 rFID and 32.54 dB PSNR on ImageNet-256 under an approximately 30-epoch two-stage training budget. These results motivate parameter sharing without implying that it improves every quality criterion.

Does the trained recurrent depth remain necessary? We examine a trained CNN through stage truncation and single-pass bypass interventions. Completing the trained schedule consistently improves reconstruction in the tested cases, and small relative feature changes can still have a substantial efect on the final output. At the same time, raw-output errors under truncation show that the recurrent trajectory is not automatically an anytime decoder. These complementary observations motivate studying both the utility of repeated computation and the calibration of intermediate states.

What is the execution cost of parameter reuse? Runtime profiling shows that the 8x looped CNN stores 65.3% fewer parameters than a local Flux-style reference, but uses 1.57× its estimated MACs and 2.85× its batch latency. Reusing weights does not remove repeated high-resolution computation. This distinction between parameter economy and execution eficiency is central to interpreting the method.

Our contributions are: (i) a scale- and loop-conditioned autoencoder that separates shared processing from resolution transitions; (ii) evidence on sharing scope, depth sensitivity, and execution cost, including a trace-based analysis of small but consequential updates; and (iii) CNN and Transformer instantiations with standard single-resolution or selectable multi-resolution latent interfaces. Multi-resolution reconstruction and downstream difusion demonstrate uses of these interfaces. Together, the results characterize where recurrent parameter sharing works, what its trained passes contribute, and which costs it leaves intact.

## 2 Related Work

Continuous visual tokenizers. Variational autoencoders [15] provide a probabilistic continuous latent representation. Perceptual and adversarial reconstruction objectives, also used in discrete tokenizers such as VQGAN [8], help preserve visual detail; latent difusion models use perceptually trained autoencoders as their first stage [21]. Recent work improves high-compression tokenization with residual autoencoding [3], studies Transformer tokenizer scaling [13], replaces a compound reconstruction recipe with a difusion objective [4], or shapes latents with semantic supervision [2, 27, 31]. These methods primarily change the objective, bottleneck, or source of representation supervision. LoopVAE studies an orthogonal architectural variable: whether the same fixed-resolution operator can be reused throughout the spatial hierarchy.

CNN and Transformer autoencoders. Convolutional autoencoders encode locality and translation equivariance, while Vision Transformers [7] provide content-dependent global interaction. ConvNeXt [17] modernizes the residual CNN design with large depthwise kernels and expanded channel mixing; ViTok [13] studies how Transformer autoencoders scale for reconstruction and generation. Window attention [16] reduces the quadratic cost of high-resolution interaction. We treat the spatial mixer as an implementation choice inside the same recurrent hierarchy. The central question is sharing scope, not whether CNNs or Transformers are universally preferable.

Weight tying and recurrent depth. Universal Transformers [5] reuse weights across depth, and looped Transformers can express iterative computation with a shallow shared network [11]. Geiping et al. [10] train recurrent-depth language models with variable recurrence and study additional computation at test time. SMELT [24] studies middle-layer looping in MoE Transformers while closely matching per-token FLOPs, stored parameters, and KV cache. These works distinguish executed depth from independently parameterized depth. LoopVAE studies a visual autoencoder whose core is tied across both recurrent steps and changing spatial resolutions. Our fixed-schedule experiments establish neither test-time depth extrapolation nor a compute-matched advantage. Scale conditioning identifies the resolution; its independent benefit remains to be isolated.

Recurrence in visual compression and generation. Recurrent image compression predates LoopVAE: Toderici et al. [23] use recurrent encoders and decoders with quantization to support variable-rate compression. Their progressive coding objective difers from our continuous latent interface and reuse of a core across a spatial hierarchy. Elastic Looped Transformers [12] reuse blocks in visual generative backbones and use intra-loop self-distillation to improve intermediate predictions. LoopVAE instead studies the tokenizer; our truncation diagnostics highlight why intermediate-output quality should not be inferred from weight sharing alone. The distinction is therefore the location and scope of recurrence, not the introduction of recurrent computation to vision.

Multi-resolution latent representations. Compression ratio controls both reconstruction capacity and the token sequence presented to a generative model. Deep compression autoencoders [3] reduce spatial sequence length aggressively, whereas semantic or high-dimensional tokenizers preserve richer features at a larger latent cost [27, 31]. Most systems expose one bottleneck per trained tokenizer. LoopVAE-Multi instead attaches several posterior heads to one shared hierarchy and supplies corresponding decoder entry points. It therefore treats latent resolution as a selectable interface of one model, while leaving a controlled comparison against separately trained tokenizers to future work.

## 3 Method

## 3.1 From Scale-Specific Stages to a Shared Operator

Let $\boldsymbol { x } \in \mathbb { R } ^ { 3 \times H \times W }$ be an image, E an encoder, and D a decoder. A continuous tokenizer produces a Gaussian posterior and reconstructs a sample from it:

$$
q ( z \mid x ) = E ( x ) , \qquad z \sim q ( z \mid x ) , \qquad { \hat { x } } = D ( z ) .\tag{1}
$$

A conventional hierarchical encoder can be written as

$$
h _ { s + 1 } = T _ { s } \left( F _ { s } ( h _ { s } ) \right) ,\tag{2}
$$

where $F _ { s }$ processes features at scale s and $T _ { s }$ changes their resolution. Both modules depend on the scale. LoopVAE keeps $T _ { s }$ scale specific but replaces the collection $\{ F _ { s } \}$ with one conditioned operator $F _ { \theta }$

The encoder and decoder use separate parameters, shown as $F _ { \theta }$ and $F _ { \psi }$ in Figure 1, because analysis and synthesis solve diferent transformations. Within either branch, however, the same operator is reused across every scale and loop step. We write $F _ { \theta }$ generically below for either branch’s operator. This separation defines the scope of our claim: we share the expressive fixed-resolution computation, while retaining independent downsampling, upsampling, input projection, and output projection modules wherever tensor geometry changes.

## 3.2 Scale-Conditioned Depth Recurrence

At scale s, let $h _ { s , 0 }$ be the feature produced by the preceding transition. A learned scale embedding $e _ { s }$ identifies the resolution and a learned loop embedding $e _ { t }$ identifies the current refinement step:

$$
c _ { s , t } = e _ { s } + e _ { t } .\tag{3}
$$

The feature is then updated $L _ { s }$ times:

$$
h _ { s , t + 1 } = F _ { \theta } ( h _ { s , t } + { \bf 1 } _ { t > 0 } I _ { s } ( h _ { s , 0 } ) , \ c _ { s , t } ) , \quad t = 0 , \dots , L _ { s } - 1 .\tag{4}
$$

$I _ { s } ( h ) = g _ { s } P _ { s } ( h )$ is a scale-specific projection multiplied by a learned scalar $g _ { s }$ initialized to zero. $P _ { s }$ is a 1×1 convolution for CNN features and a token-wise linear map for Transformer features. The first iteration omits this injection. On later iterations, it is added once before the entire K-block core, not before every block. This construction provides an information shortcut; its independent benefit has not yet been isolated experimentally.

For the three-scale model, a stack of $K = 4$ blocks is evaluated with $L = [ 1 , 2 , 4 ]$ in traversal order: 2x, 4x, and 8x for the encoder, but 8x, 4x, and 2x for the decoder. Thus the decoder performs more loops at its finer scales, not at the same spatial scales as the encoder. Each branch stores four core blocks but performs

$$
d _ { \mathrm { c o r e } } = K \sum _ { s } L _ { s } = 4 ( 1 + 2 + 4 ) = 2 8\tag{5}
$$

core block applications in each forward pass through the encoder or decoder. Here K is the stored core depth, $L _ { s }$ is the recurrent pass count at scale s, and $d _ { \mathrm { c o r e } }$ is the executed core depth, excluding transitions and heads. The same core therefore realizes depth through loops within each scale and weight reuse across scales. This count is not a FLOP estimate: a block’s cost depends on its spatial resolution. All reported checkpoints use fixed loop schedules; additional test-time recurrence has not been evaluated.

Stored capacity and executed work. For S scales and p parameters per core block, the core parameter counts under global, scale-wise, and fully unshared processing are

$$
P _ { \mathrm { g l o b a l } } = K p , \qquad P _ { \mathrm { s c a l e } } = S K p , \qquad P _ { \mathrm { u n s h a r e d } } = K p \sum _ { s } L _ { s } .\tag{6}
$$

These counts exclude the transitions, heads, embeddings, and injection modules. They describe core capacity, not total-model compression ratios. For the same execution graph, all three designs apply $K \sum _ { s } L _ { s }$ blocks, with core arithmetic approximately $\begin{array} { r } { K \sum _ { s } L _ { s } \mathcal { C } ( H _ { s } , W _ { s } , C ) } \end{array}$ , where C is one block’s cost. Sharing changes which weights are read, not how often a block is executed. This separates the architectural variable tested by the sharing ablation from the runtime efects measured in Section 4.7.

## 3.3 Operator Instantiations

Figure 2 details the recurrent wrapper and both block alternatives. Scale and loop embeddings are learned lookup vectors; their sum conditions each block rather than being directly added to image features. For block $j ,$ a SiLU followed by a linear projection produces six channel-wise vectors:

$$
( \beta _ { 1 } , \gamma _ { 1 } , \alpha _ { 1 } , \beta _ { 2 } , \gamma _ { 2 } , \alpha _ { 2 } ) = W _ { j } \operatorname { S i L U } ( c _ { s , t } ) + b _ { j } .\tag{7}
$$

The projection maps width $C$ to $6 C$ and starts with zero weights and bias. Each block has its own projection, reused whenever that block executes. The two residual updates are

$$
\begin{array} { r l } & { v = u + \alpha _ { 1 } \odot M _ { 1 } \big ( ( 1 + \gamma _ { 1 } ) \odot N _ { 1 } ( u ) + \beta _ { 1 } \big ) , } \\ & { y = v + \alpha _ { 2 } \odot M _ { 2 } \big ( ( 1 + \gamma _ { 2 } ) \odot N _ { 2 } ( v ) + \beta _ { 2 } \big ) . } \end{array}\tag{8}
$$

Here $N _ { i }$ is afine-free normalization and $M _ { i }$ is a spatial or channel mixer. Modulation vectors broadcast over spatial positions or tokens; $\alpha _ { i }$ are unconstrained channel-wise gates, not sigmoid probabilities. Zero-initialized gates make each block initially an identity. They are distinct from the scalar input-injection gate $g _ { s }$

Convolutional operator. Our primary model uses a conditioned ConvNeXt-style block [17]. Both $N _ { i }$ are GroupNorm with 32 groups and $\epsilon = 1 0 ^ { - 6 }$ , so this is $A d a G N ,$ not AdaLN. $M _ { 1 }$ is a depthwise $7 \times 7$ convolution, SiLU, and a $1 \times 1$ projection. $M _ { 2 }$ is a 1×1 expansion from $C$ to $4 C ,$ SiLU, and a 1×1 projection back to $C .$ The main configuration uses width 384 and zero dropout.

Transformer operator. The same hierarchy can instead use a DiT-style self-attention block [20]. Here $N _ { i }$ are afine-free LayerNorm with $\epsilon = 1 0 ^ { - 6 }$ , yielding AdaLN. $M _ { 1 }$ projects queries, keys, and values, applies two-dimensional rotary embeddings [22] to queries and keys, performs attention, and projects the output. At high-resolution scales, non-overlapping local windows [16] limit attention cost. $M _ { 2 }$ is a $C  4 C  C$ linear MLP with GELU. These blocks receive scale and recurrence indices, not difusion timesteps or class labels.

## 3.4 Single- and Multi-Resolution Tokenizers

LoopVAE-Single. The single-resolution encoder uses a stride-2 stem followed by the three scales $\mathrm { 2 x  4 x  8 x }$ A final head predicts the mean and log variance of a diagonal Gaussian. At $2 5 6 \times 2 5 6$ , the f8d16 latent has shape $1 6 \times 3 2 \times 3 2$ . The decoder projects this latent to the shared width and traverses the scales in reverse. Its core blocks are independent of the encoder blocks but are reused in the same way. Consequently, LoopVAE-Single can replace a conventional f8d16 VAE without changing the downstream difusion input shape.

LoopVAE-Multi. Figure 3 shows the multi-resolution model, which extends the hierarchy to $\mathrm { 2 x  4 x  }$ $8 \mathrm { x }  1 6 \mathrm { x }  3 2 \mathrm { x }$ . Independent posterior heads are attached at the last three scales:

$$
\{ q ( z _ { 8 \mathrm { x } } \mid x ) , q ( z _ { 1 6 \mathrm { x } } \mid x ) , q ( z _ { 3 2 \mathrm { x } } \mid x ) \} = E _ { \mathrm { m u l t i } } ( x ) .\tag{9}
$$

Their channel dimensions are 16, 64, and 128, giving latent shapes $1 6 \times 3 2 ^ { 2 } , 6 4 \times 1 6 ^ { 2 }$ , and $1 2 8 \times 8 ^ { 2 }$ for a $2 5 6 ^ { 2 }$ image. Here 8x, 16x, and 32x denote spatial downsampling, not bitrate: the first two interfaces both contain 16,384 scalars, while the last contains 8,192. Each head has its own normalization and output projection. The decoder similarly has one input projection per latent shape, enters the shared synthesis hierarchy at the selected scale, and executes only the remaining scales. The exploratory five-scale configuration uses six Transformer blocks and encoder loop schedule [1, 1, 2, 3, 6].

![](images/2d1cbd53808a1096ed7f21bb9da0349ad147ce1b2959517fc38fba2bec28ba40.jpg)  
Figure 2 Inside the recurrent core. Input injection precedes the entire core on passes t > 0. Each distinct block has its own zero-initialized modulation projection. CNN blocks use AdaGN and SiLU; Transformer blocks use AdaLN, attention, and a GELU MLP. Both alternatives have two gated residual branches. The same core weights are reused across scales and passes, separately in each encoder and decoder.

## LoopVAE-Multi: three latent interfaces, one shared decoder

![](images/367d328e584d7b674b5c782abde560791cdcce75d2dd41c315a2b1302f7138ae.jpg)  
Each core contains six Transformer blocks. Encoder and decoder weights are separate Sample z during training; use posterior mode at evaluation.

Figure 3 LoopVAE-Multi and joint training. A single encoder traversal produces three independent posterior heads. Each sampled latent is decoded separately through its own input projection and the corresponding sufix of one shared decoder. The three decoder boxes depict executions with the same weights, not independently parameterized decoders or a concatenation of latents. Scale labels denote spatial downsampling. Each F contains six Transformer blocks; the per-interface losses include reconstruction, KL, and active adversarial terms. Their weighted sum jointly trains the encoder and decoder.

## 3.5 Training Objective

Training has two stages. Stage 1 is non-adversarial pretraining with a reconstruction, perceptual, and KL objective:

$$
\mathcal { L } _ { \mathrm { r e c } } = \| x - \hat { x } \| _ { 1 } + \lambda _ { p } \mathcal { L } _ { \mathrm { L P I P S } } ( x , \hat { x } ) + \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } ( q ( z \mid x ) \| \mathcal { N } ( 0 , I ) ) .\tag{10}
$$

In the implementation, reconstruction terms are averaged over pixels and the KL term is averaged over latent elements and batch examples. During training, the single-latent model additionally perturbs posterior samples with zero-mean Gaussian noise of standard deviation 0.05; evaluation uses the posterior mode without this perturbation. By default, Stage 1 runs for 600k steps. Stage 2 initializes the autoencoder from the Stage-1 weights and continues training with a hinge adversarial loss to sharpen high-frequency detail:

$$
\begin{array} { r } { \mathscr { L } = \mathscr { L } _ { \mathrm { r e c } } + \lambda _ { a } \mathscr { L } _ { \mathrm { a d v } } . } \end{array}\tag{11}
$$

Here $\mathcal { L } _ { \mathrm { a d v } }$ denotes the generator’s negative mean discriminator score; the discriminator is trained with the hinge objective. The weight $\lambda _ { a }$ uses a clipped, gradient-norm-based adaptive coeficient after the configured activation threshold. The main CNN run uses frozen DINOv2 features [19] with a learned head, diferentiable augmentation inspired by DifAugment [30], and lazy R1 regularization [18]. The multi-resolution run instead uses a convolutional discriminator; its per-interface reconstruction, KL, and active adversarial losses are combined with fixed scale weights.

## 4 Experiments

We evaluate reconstruction quality, sharing scope, recurrent-depth sensitivity, and execution cost. Multiresolution reconstruction and downstream difusion test additional uses of the learned interfaces. Checkpoint evaluations, within-checkpoint interventions, and architectural profiling address complementary questions; their configurations are specified separately.

## 4.1 Setup

Data and metrics. The reconstruction protocol uses the ImageNet-1k training split and the 50k-image validation split at 256×256 [6]. We report rFID [14], PSNR, LPIPS [29], and SSIM [25], with posterior-mode decoding. The sharing ablation is evaluated before adversarial training; its table deliberately reports only paired reconstruction metrics. Results are individual checkpoint evaluations, not averages across seeds.

Training stages and budgets. Stage 1 is non-adversarial pretraining for 600k steps by default; Stage 2 continues from these weights with GAN training. Stage-2 step labels count only the second stage and exclude Stage 1. For the main reconstruction comparison, we report a nominal total budget of approximately 30 epochs: approximately 15 epochs of Stage 1 plus 15 epochs of Stage 2. The reference configuration uses eight GPUs, each with 96 GB of device memory, and four images per GPU, giving a global batch size of 32. Appendix A.1 specifies the epoch-estimation convention.

Models and checkpoint scope. The primary CNN LoopVAE-Single uses width 384, four core blocks per branch, loop schedule [1, 2, 4], and an f8d16 latent. It stores 29.07M encoder and decoder parameters, excluding perceptual and discriminator networks. Table 1 reports the CNN and ViT results under this two-stage budget at f8d16: eightfold spatial downsampling and 16 latent channels. The sharing ablation evaluates the V3 Transformer at the non-adversarial Stage-1 400k checkpoint, before the default pretraining endpoint. Loop interventions and downstream difusion use the Stage-2 610k CNN tokenizer, while all three Multi interfaces are evaluated jointly at Stage 2, step 220k. These auxiliary evaluations are kept distinct from the final reconstruction comparison.

## 4.2 Reconstruction Results

We compare the f8d16 models with the SD3 [9] and FLUX [1] VAEs, which expose the same latent shape at 256<sup>2</sup> resolution.

<table><tr><td>Model (f8d16)</td><td>Params ↓</td><td>rFID ↓</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td>SD3 VAE</td><td>84M</td><td>0.19</td><td>31.29</td><td>0.060</td><td>0.88</td></tr><tr><td>FLUX-VAE</td><td>84M</td><td>0.18</td><td>32.80</td><td>0.044</td><td>0.91</td></tr><tr><td>LoopVAE-ViT (ours)</td><td>34M</td><td>0.32</td><td>31.43</td><td>0.072</td><td>0.89</td></tr><tr><td>LoopVAE-CNN (ours)</td><td>29M</td><td>0.28</td><td>32.54</td><td>0.048</td><td>0.91</td></tr></table>

Table 1 f8d16 reconstruction on ImageNet-256. Our LoopVAE models use approximately 30 epochs in total: 15 without GAN (Stage 1) followed by 15 with GAN (Stage 2). Baseline training recipes are not matched. Bold denotes column-best results, including ties at the reported precision.

Under this approximately 30-epoch two-stage budget, the 29M-parameter LoopVAE-CNN reaches 0.28 rFID, 32.54 PSNR, 0.048 LPIPS, and 0.91 SSIM (Table 1). It uses approximately 65% fewer parameters than the 84M baselines and matches FLUX-VAE’s displayed SSIM, while FLUX-VAE has better rFID, PSNR, and LPIPS. The 34M-parameter LoopVAE-ViT reaches 0.32 rFID. Training recipes are not controlled, so quality diferences cannot be attributed solely to recurrence or operator choice.

## 4.3 Non-Adversarial Sharing Ablation

The V3 Transformer ablation compares three sharing scopes at the Stage-1 400k checkpoints without GAN training. Each follows the same three-scale loop graph with 28 core block applications per encoder or decoder. Global sharing stores four blocks per branch; scale-wise sharing stores 12, reusing each scale’s blocks across its loop steps; fully unshared processing stores 28. Thus the comparison varies stored core capacity while retaining the intended sequence of block applications.
<table><tr><td>Sharing rule</td><td>Stored blocks</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM↑</td></tr><tr><td>Global</td><td>4</td><td>31.163</td><td>0.0852</td><td>0.8875</td></tr><tr><td>Scale-wise</td><td>12</td><td>30.412</td><td>0.0841</td><td>0.8713</td></tr><tr><td>Fully unshared</td><td>28</td><td>31.043</td><td>0.0797</td><td>0.8852</td></tr></table>

Table 2 V3 Transformer sharing ablation at Stage 1, step 400k, without adversarial training. Stored blocks are counted per branch; all designs execute 28 core block applications per branch. rFID is omitted from this non-GAN comparison. Bold marks the best value in each metric.

Global sharing has the highest PSNR and SSIM, exceeding fully unshared processing by 0.120 dB and 0.0023, respectively, while using one seventh as many stored core blocks. Fully unshared processing achieves the lowest LPIPS, improving over global sharing by 0.0055. Scale-wise sharing lies between them in LPIPS but has the lowest PSNR and SSIM. Removing sharing therefore does not uniformly improve reconstruction: the smallest shared core remains competitive on paired fidelity, while unshared capacity benefits LPIPS. These single-run results do not quantify seed variation or isolate possible efects of data ordering.

## 4.4 Recurrent Depth and Loop-Position Sensitivity

Does repeated application of the shared core contribute useful computation, or are some passes efectively redundant? We examine the trained CNN tokenizer through two inference-time interventions: truncating a single stage to its first k passes, and bypassing one complete pass while retaining all subsequent loop indices. These tests expose sensitivity to recurrent depth and position without retraining the model.

Controlled interventions. We use 16 typical test images from TokBench [26] for a targeted case study of fine visual detail, with particular attention to faces and text. The experiments use the Stage-2 610k f8d16 CNN checkpoint, corresponding to approximately 15 nominal epochs of GAN training after Stage 1, with posterior-mode decoding in bfloat16. We measure how recurrent interventions afect paired reconstruction, rather than running the full TokBench text-recognition and face-similarity evaluation. All selected images enter the reported statistics; their identities and selection order are documented in Appendix A.4. Encoder stages at f2/f4/f8 execute 1/2/4 passes; decoder stages at f8/f4/f2 execute 1/2/4. Here f2 denotes an internal half-resolution feature map, not a latent compression ratio.

For a prefix intervention, only the selected stage is shortened; the remaining network completes its trained schedule. Decoder interventions use the same full-encoder latent. For a bypass, we replace the entire pass output by its pre-injection state, removing both injection and core update while preserving later loop indices. These are counterfactual final reconstructions, not native intermediate RGB predictions or independently trained shallow models. All 96 full-prefix reconstructions (six stages on 16 images) match their corresponding baseline arrays exactly. Appendix A.4 gives the measurement protocol.

Depth benefits persist across images. At encoder f8, mean PSNR increases from 20.32 to 21.85, 24.50, and 32.28 dB as the retained prefix grows from one to four passes (Figure 4). At decoder f2, the corresponding values are 14.54, 15.26, 19.36, and 32.28 dB. All four multi-pass stages—encoder f4/f8 and decoder f4/f2—have non-decreasing PSNR and SSIM on every evaluated image. The pattern also holds for PSNR computed before output clipping. Thus the mean trend does not conceal a counterexample within this subset. Full-schedule reconstruction averages 32.279 dB PSNR and 0.9424 SSIM; these subset metrics are not substituted into the main reconstruction table.

![](images/9b8014a43d276f49497a8fb374274d26af6d236e4644ef1f28e3fc2ff6ee7289.jpg)

(b) Encoder f8: single-loop bypass  
![](images/66fc003ebe2dd666f8f4347ce4272708c1aea93435938ee36eca165132c26b31.jpg)

(c) Decoder f2: retained depth  
![](images/918ed181ac4fc1020cdcb0722f1eb995edac80683bfdb1782366a430c03dcd97.jpg)

(d) Decoder f2: single-loop bypass  
![](images/836eb6853941127f598e16ff6b74a16257b74b9dafc405e1416705b3cc450633.jpg)  
Figure 4 Recurrent depth and position afect reconstruction. Left: truncating one stage while keeping all other stages fixed. Right: independently bypassing one loop; positive values denote PSNR loss relative to the full schedule. Thin lines show the 16 test images; markers and bars show the mean and sample standard deviation across images, not seed uncertainty. The second encoder f8 pass and third decoder f2 pass are the most bypass-sensitive positions in their respective stages on every tested image.

Shared weights do not imply equivalent loop positions. All 224 separately applied single-pass bypasses reduce PSNR relative to their image’s full schedule. At encoder f8, bypassing passes 1–4 produces mean losses of 3.66, 9.87, 7.54, and 7.78 dB. At decoder f2, the losses are 8.44, 11.42, 13.16, and 12.92 dB. The second encoder f8 pass and third decoder f2 pass are the most sensitive positions within their respective stages on all 16 images. The largest loss is therefore not assigned to the last pass, despite the large final-step gain in the prefix experiment. These observations support position-dependent computation under shared weights. They do not isolate loop conditioning from the evolving hidden state, and bypass losses cannot be added as independent contributions.

Small feature updates can be consequential. We pair the normal forward trace with the bypass measurement for each image and pass. Define the relative update as

$$
r _ { s , t } = \frac { \mathrm { R M S } ( h _ { s , t + 1 } - h _ { s , t } ) } { \operatorname* { m a x } ( \mathrm { R M S } ( h _ { s , t } ) , 1 0 ^ { - 8 } ) } ,\tag{12}
$$

where $h _ { s , t }$ is the pre-injection state, so the numerator includes injection and core processing. Figure 6 shows that a small change in this norm does not imply a dispensable pass. The fourth encoder f8 pass changes the feature by only 7.47–9.36% across images, yet bypassing it loses 4.00–11.94 dB PSNR (mean 7.78 dB). Thus even updates below a relative threshold of 0.1 can be important to reconstruction in this checkpoint. The dimensionless norm is a description of the current feature coordinates, not a calibrated estimate of output error or a validated stopping criterion.

![](images/702adac4dc7342d526a0564de412ac3963689307cb28827575a9eba68ffaa198.jpg)  
Figure 5 Four cases selected at evenly spaced indices in the recorded order. Rows use zero-based indices 0, 5, 10, and 15, selected without ranking reconstruction quality. Only the encoder f8 prefix changes; the downstream network retains its trained schedule. Columns show the reference and one through four retained passes, with per-image PSNR. All panels use the same RGB mapping and clipping, with no per-panel contrast adjustment. The arrays are measured reconstructions, not illustrative predictions.

![](images/c91d5de4baa46bf71c31c9d2c435f4294695c91586198206d64ecda8d8b8f8ff.jpg)

![](images/13dec353f6a6dd27283be9cba05a34a5e2fb5e03d95463932d3ac12a88560802.jpg)  
Figure 6 Feature-change magnitude and reconstruction sensitivity measure diferent properties. Each row iden tifies the same pass in a full forward trace (left) and an independent bypass experiment (right). E/D denote encoder/decoder, and L is the one-based loop index. Bars show means over 16 images; whiskers show the observed minimum and maximum, not confidence intervals. In particular, small encoder f8 updates can have large downstream consequences. These are paired measurements on the existing subset, not additional model evaluations.

Output-range recovery, not established early-exit refinement. The qualitative cases show recognizable structure before the last pass, but also substantial color and contrast distortion (Figure 5). The final pass raises mean PSNR by 7.78 dB at encoder f8 and 12.92 dB at decoder f2. Crucially, decoder f2 prefixes of length one, two, and three produce out-of-range RGB values at 96.72%, 93.44%, and 68.34% of pixels on average, versus 6.14% for the full schedule. Before clipping, their mean PSNR values are −1.04, 3.42, and 14.45 dB, compared with 32.25 dB for the complete model (Table 7). Clipping therefore masks substantial early-exit distortion rather than creating the full-depth advantage: the complete model’s mean clipping benefit is only 0.028 dB. The evidence demonstrates useful recurrent computation and sensitivity to the trained schedule, but does not establish that intermediate states are calibrated for early exit, that iteration converges, or that additiona untrained loops would help.

## 4.5 Multi-Resolution Reconstruction

The joint Stage-2 220k checkpoint reconstructs from all three latent interfaces (Table 3). Its 8x and 16x interfaces have similar paired reconstruction quality, while 32x reduces PSNR, LPIPS quality, and SSIM. These operating points demonstrate one backbone with several latent interfaces, not superiority over three separately trained tokenizers.

## 4.6 Downstream Difusion

We train class-conditional LightningDiT models on normalized f8d16 latents using the FasterDiT and VA-VAE pipeline [27, 28]. Both $\mathrm { B / 2 }$ and $\mathrm { { X L } / 2 }$ use the Stage-2 610k CNN tokenizer (approximately 15 nominal Stage-2 epochs, after Stage-1 pretraining) and are evaluated on 50,000 generated images without classifier-free guidance (CFG). Class conditioning remains active. The tokenizer is frozen throughout generator training.

The B/2 and $\mathrm { { X L } / 2 }$ runs reach FID-50k values of 39.906 and 18.198, respectively, demonstrating that the learned latents support downstream difusion. These runs do not isolate tokenizer quality from generator capacity or training cost: a matched alternative tokenizer is not included, and equal update counts alone do not imply equal training budgets. We therefore report the generation results as an application of the tokenizer rather than a controlled scaling comparison.

<table><tr><td>Latent</td><td>Shape at  $2 5 6 ^ { 2 }$ </td><td> $\mathrm { r F I D \downarrow }$ </td><td>PSNR↑</td><td>LPIPS ↓</td><td>SSIM ↑</td></tr><tr><td> $_ \mathrm { 8 x }$ </td><td> $1 6 \times 3 2 \times 3 2$ </td><td>0.768</td><td>29.156</td><td>0.0868</td><td>0.8398</td></tr><tr><td>16x</td><td> $6 4 \times 1 6 \times 1 6$ </td><td>0.718</td><td>29.197</td><td>0.0856</td><td>0.8395</td></tr><tr><td>32x</td><td> $1 2 8 \times 8 \times 8$ </td><td>1.436</td><td>27.107</td><td>0.1413</td><td>0.7728</td></tr></table>

Table 3 Joint Multi evaluation at the Stage-2 220k checkpoint (approximately 5.5 nominal Stage-2 epochs; Stage 1 excluded). All three rows use one checkpoint and one shared encoder traversal, followed by decoding from the indicated latent interface.
<table><tr><td>Generator</td><td>Updates</td><td>CFG</td><td>Samples</td><td>FID-50k↓</td></tr><tr><td> $\mathrm { L i g h t n i n g D i T { - } B / 2 }$ </td><td>200k</td><td>off</td><td>50k</td><td>39.906</td></tr><tr><td> $\mathrm { L i g h t n i n g D i T { - } X L / 2 }$ </td><td>200k</td><td> $\mathrm { o f f }$ </td><td>50k</td><td>18.198</td></tr></table>

Table 4 Class-conditional generation on CNN LoopVAE latents, without CFG. Both generators use the Stage-2 610k tokenizer, are evaluated after 200k generator updates, and use 50k generated samples for FID. The $\mathrm { { X L } / 2 }$ difusion-model run corresponds to 80 training epochs; this is separate from the tokenizer’s two-stage budget.

## 4.7 Parameters, Arithmetic, and Runtime

Table 5 reports a benchmark on a single GPU with 96 GB of device memory at $2 5 6 ^ { 2 }$ resolution with bfloat16 execution, batch size eight, ten warm-up iterations, and 100 timed iterations. The evaluator instantiates encoder/decoder pairs from configurations and runs encoding, posterior sampling, and decoding on synthetic inputs. This is an architectural benchmark, not an accuracy-matched comparison of pretrained checkpoints. The label Flux-style denotes a local CNN reference configuration rather than an oficial Flux performance measurement. Parameter counts are specific to the profiled configurations; in particular, the 32.46M V3 ViT is distinct from the reported 34M reconstruction model.

At 8x, the looped CNN stores 65.3% fewer parameters than the Flux-style reference, but uses 1.57× its estimated MACs and takes 2.85× its batch latency (281.05 versus 98.73 ms). Its peak allocated memory is also higher. The fixed-resolution ViT-B is much faster despite storing more parameters; its execution graph difers substantially from recurrent high-resolution processing. Fewer weights therefore do not imply fewer operations, lower activation memory, or faster inference.

Where repeated computation concentrates. For a fixed-width CNN core, the convolutional work is proportional to $K L _ { s } H _ { s } W _ { s } . \mathrm { ~ A t ~ } 2 5 6 ^ { 2 }$ input resolution, encoder $\mathrm { f 2 / f 4 / f 8 }$ and decoder $\mathrm { f 8 / f 4 / f 2 }$ contribute proportionally to 16 $\colon 8 : 4 : 1 : 8 : 6 4$ , respectively. The four passes at the finest decoder stage therefore account for $6 4 / 1 0 1 = 6 3 . 4 \%$ of this spatially linear core work. This is an analytical decomposition of the specified graph, excluding transitions, heads, and conditioning overhead, not a measured full-model FLOP breakdown. It explains why the 28-block count alone obscures an important cost: the same block is much more expensive at higher spatial resolution. The sensitivity of decoder f2 to truncation also shows that removing this work from the existing checkpoint is not quality-neutral.

The 32x V5 benchmark includes all stored encoder/decoder parameters and multi-head encoder computation, using the $3 2 \mathrm { x }$ decoder entry. The looped CNN V5 failed to instantiate and has no measurement. Diferent latent shapes and loop schedules make the $1 6 \mathrm { x } / 3 2 \mathrm { x }$ rows an uncontrolled compression comparison. Arithmetic estimates cover convolution, linear, and attention matrix products, but omit some normalization, activation, and elementwise work. Runtime is measured independently on the same device for all models; timings need not transfer to other GPUs with the same memory capacity.

<table><tr><td>Scale</td><td>Model</td><td>Params (M)</td><td>MACs (G)</td><td>FLOPs (G)</td><td>Images/s</td><td>ms/image</td><td>MiB</td></tr><tr><td>8x</td><td>Looped CNN V3</td><td>29.07</td><td>700.92</td><td>1401.84</td><td>28.5</td><td>35.13</td><td>1729.3</td></tr><tr><td>8x</td><td>Looped ViT V3</td><td>32.46</td><td>959.89</td><td>1919.79</td><td>28.0</td><td>35.72</td><td>1359.6</td></tr><tr><td>8x</td><td>CNN (Flux-style)</td><td>83.82</td><td>447.62</td><td>895.25</td><td>81.0</td><td>12.34</td><td>1302.8</td></tr><tr><td>8x</td><td>ViT-B</td><td>172.01</td><td>213.01</td><td>426.02</td><td>270.4</td><td>3.70</td><td>521.2</td></tr><tr><td>8x</td><td>ViT-L</td><td>607.08</td><td>722.10</td><td>1444.21</td><td>73.8</td><td>13.54</td><td>1379.4</td></tr><tr><td>16x</td><td>Looped CNN V3</td><td>32.67</td><td>470.56</td><td>941.12</td><td>42.3</td><td>23.67</td><td>1746.3</td></tr><tr><td>16x</td><td>Looped ViT V3</td><td>36.06</td><td>587.94</td><td>1175.88</td><td>39.4</td><td>25.38</td><td>1369.1</td></tr><tr><td>16x</td><td>CNN</td><td>69.83</td><td>195.06</td><td>390.13</td><td>133.4</td><td>7.50</td><td>956.5</td></tr><tr><td>16x</td><td>ViT-B</td><td>171.51</td><td>46.18</td><td>92.36</td><td>731.3</td><td>1.37</td><td>428.4</td></tr><tr><td>16x</td><td>ViT-L</td><td>606.41</td><td>161.43</td><td>322.86</td><td>268.7</td><td>3.72</td><td>1249.2</td></tr><tr><td>32x</td><td>Looped ViT V5</td><td>208.31</td><td>2629.46</td><td>5258.91</td><td>13.9</td><td>71.81</td><td>2966.9</td></tr><tr><td>32x</td><td>CNN</td><td>99.49</td><td>191.86</td><td>383.71</td><td>132.9</td><td>7.53</td><td>1028.9</td></tr></table>

Table 5 Architectural benchmark on a single GPU with 96 GB of device memory. Parameters cover the encoder and decoder; MACs and FLOPs are estimates per image, with one MAC counted as two FLOPs. Time/image is batch latency divided by eight, not batch-one latency. Memory is peak allocated MiB (the log labels it MB).

## 5 Discussion and Limitations

Parameter sharing is a capacity choice, not an inference shortcut. The central result is that a small core can support a deep visual hierarchy without assigning diferent weights to every scale and pass. The Transformer ablation provides the most direct evidence: global sharing remains competitive in paired fidelity at one seventh the stored core depth, while unshared processing improves LPIPS. The final CNN result demonstrates a useful parameter–quality operating point, but the unmatched baseline recipes do not establish a causal advantage of recurrence. Parameter storage may motivate this design; the measured arithmetic, latency, and activation memory do not support an execution-eficiency claim.

Useful recurrence need not resemble convergence. The intervention study shows that the trained schedule matters, including passes whose relative feature changes are small. Completing a stage improves reconstruction on the examined images, but severe raw-output errors under truncation make intermediate quality diferent from full-schedule quality. These observations are consistent with a depth-dependent transformation rather than interchangeable refinement steps. They neither establish fixed-point convergence nor isolate the roles of loop conditioning and evolving hidden states. In particular, a bypass tests dependence within the trained network; it does not predict the performance of a shallower model trained from scratch.

The high-resolution schedule is a concrete optimization target. The finest decoder stage dominates the spatially linear CNN core work, yet is also sensitive to truncation. This combination suggests that reducing deployment cost requires adapting training, not simply omitting passes after training. A useful next test is a matched-budget schedule comparison that reallocates passes between coarse and fine scales. Intermediateoutput supervision or distillation [12] could separately test whether a shared tokenizer can support reliable early exit; the current measurements do not answer that question.

Evidence boundaries. The sharing comparison is a single-run, non-adversarial Transformer experiment; seed variation, identical data order, and separate efects of scale embeddings, loop embeddings, and input injection remain unresolved. The loop study is a targeted TokBench case analysis, not a full benchmark evaluation, and does not measure text-recognition accuracy, face similarity, or LPIPS. Reconstruction, diagnostics, generation, and profiling have distinct checkpoint or configuration scopes, documented in the appendix. Multi-resolution heads change channel capacity as well as spatial size and lack separately trained controls. Both generation results use FID-50k, but a matched alternative tokenizer and a complete evaluation-sampler record are not available. Higher-resolution transfer, video, and text-conditioned generation remain untested.

A controlled tokenizer comparison, broader loop-intervention evaluation, and repeated sharing or schedule runs would test whether the observed tradeof persists across checkpoints and data distributions. These extensions complement the present evidence that globally shared processing can support useful visual tokenization while retaining a substantial dependence on its trained execution schedule.

## 6 Conclusion

LoopVAE separates resolution-changing transitions from a processing core shared within and across spatial scales. A four-block core realizes 28 block applications per branch, with CNN and Transformer implementations and single- or multi-resolution latent interfaces. The 29M-parameter CNN reaches 0.28 rFID and 32.54 dB PSNR on ImageNet-256 under an approximately 30-epoch two-stage training budget; a separate sharing ablation finds a metric-dependent tradeof between global reuse and unshared capacity.

The diagnostics clarify what parameter reuse does and does not buy. Trained passes contribute to reconstruction even when their relative feature changes are small, but truncated outputs are not reliably calibrated. Highresolution recurrence also costs substantial arithmetic and runtime despite the small stored core. These results establish recurrent depth across scales as a viable parameter-sharing organization for visual tokenization and identify schedule design and intermediate-output training as concrete directions for improving its execution tradeof.

## References

[1] Black Forest Labs. FLUX.1 model release. Oficial model repository, 2024. URL https://huggingface.co/ black-forest-labs/FLUX.1-dev.

[2] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are efective tokenizers for difusion models. arXiv preprint arXiv:2502.03444, 2025. URL https://arxiv.org/abs/2502.03444.

[3] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, and Song Han. Deep compression autoencoder for eficient high-resolution difusion models. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=wH8XXUOUZU.

[4] Yinbo Chen, Rohit Girdhar, Xiaolong Wang, Sai Saketh Rambhatla, and Ishan Misra. Difusion autoencoders are scalable image tokenizers. arXiv preprint arXiv:2501.18593, 2025. URL https://arxiv.org/abs/2501.18593.

[5] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. URL https://arxiv.org/abs/2403.03206.

[10] Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025. URL https://arxiv.org/abs/2502.05171.

[11] Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. arXiv preprint arXiv:2301.13196, 2023. URL https://arxiv. org/abs/2301.13196.

[12] Sahil Goyal, Swayam Agrawal, Gautham Govind Anil, Prateek Jain, Sujoy Paul, and Aditya Kusupati. ELT: Elastic looped transformers for visual generation. arXiv preprint arXiv:2604.09168, 2026. URL https://arxiv. org/abs/2604.09168.

[13] Philippe Hansen-Estruch, David Yan, Ching-Yao Chuang, Orr Zohar, Jialiang Wang, Tingbo Hou, Tao Xu, Sriram Vishwanath, Peter Vajda, and Xinlei Chen. Learnings from scaling visual tokenizers for reconstruction and generation. In International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2501.09755.

[14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, 2017.

[15] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.

[16] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.

[17] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[18] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. Which training methods for gans do actually converge? In International Conference on Machine Learning, 2018.

[19] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[20] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[22] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[23] George Toderici, Damien Vincent, Nick Johnston, Sung Jin Hwang, David Minnen, Joel Shor, and Michele Covell. Full resolution image compression with recurrent neural networks. arXiv preprint arXiv:1608.05148, 2016. URL https://arxiv.org/abs/1608.05148.

[24] Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu, Wenhao Huang, Shen Yan, and Jian Li. SMELT: Scaling laws for compute-matched MoE looped transformers. arXiv preprint arXiv:2609.01343, 2026. URL https://arxiv.org/abs/2609.01343.

[25] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[26] Junfeng Wu, Dongliang Luo, Weizhi Zhao, Zhihao Xie, Yuanhao Wang, Junyi Li, Xudong Xie, Yuliang Liu, and Xiang Bai. TokBench: Evaluating your visual tokenizer before visual generation. arXiv preprint arXiv:2505.18142, 2025. URL https://arxiv.org/abs/2505.18142.

[27] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. URL https://arxiv.org/abs/2501.01423.

[28] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster difusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37, 2024.

[29] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.

[30] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Diferentiable augmentation for data-eficient gan training. In Advances in Neural Information Processing Systems, 2020.

[31] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Difusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. URL https://arxiv.org/abs/2510.11690.

## A Implementation and Reproducibility Details

## A.1 LoopVAE-Single Configuration

Table 6 records the configuration of the main convolutional checkpoint. Encoder and decoder use diferent parameters. The decoder traverses scales from 8x to 2x; its loop list is indexed in traversal order, so the stored list [1, 2, 4] corresponds to one, two, and four repeated passes during that reverse traversal.

<table><tr><td>Architecture</td><td>Value</td><td>Training</td><td>Value</td></tr><tr><td>Input resolution</td><td>256 × 256</td><td>Precision</td><td>bfloat16 mixed</td></tr><tr><td>Hidden width</td><td>384</td><td>Devices</td><td>8 GPUs</td></tr><tr><td>Core blocks per branch</td><td>4</td><td>Per-device batch</td><td>4</td></tr><tr><td>Expansion ratio</td><td>4</td><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Spatial kernel</td><td>7 × 7 depthwise</td><td>Stage-2 learning rate</td><td>5 × 10−5</td></tr><tr><td>Encoder traversal</td><td>2x, 4x, 8x</td><td>KL weight</td><td>10-6</td></tr><tr><td>Decoder traversal</td><td>8x, 4x, 2x</td><td>Latent noise std.</td><td>0.05</td></tr><tr><td>Loop list per branch</td><td>[1,2, 4]</td><td>LPIPS weight</td><td>1.0</td></tr><tr><td>Latent shape</td><td>16 × 32 × 32</td><td>GAN activation step</td><td>5,000</td></tr><tr><td>Total parameters</td><td>29.07M</td><td>GAN weight multiplier</td><td>0.5</td></tr></table>

Table 6 Main LoopVAE-Single configuration.

Stage 1 trains without GAN for 600k steps by default. Stage 2 initializes from the Stage-1 autoencoder weights and continues with GAN training; all Stage-2 checkpoint step labels exclude Stage 1. The main CNN and ViT reconstruction results use a nominal total budget of approximately 30 epochs, split approximately equally between the two stages.

For a global batch of 8 × 4 = 32 and approximately 1.28M ImageNet training images, one epoch contains approximately 40k data batches. Under the reported data-iteration convention, 600k and 610k steps correspond to approximately 15.0 and 15.2 epochs, respectively; 220k corresponds to approximately 5.5 epochs. We round the main two-stage budget to approximately 30 epochs. These are nominal data-exposure estimates, not an exact reconstruction of checkpoint counters. In the available manual-optimization code, Lightning’s raw global\_step can count both autoencoder and discriminator updates; exact exposure would require the corresponding epoch or data-batch logs. We retain the stage-local checkpoint labels to identify the evaluated models.

Stage 2 of the main CNN uses the frozen facebook/dinov2-small feature extractor and a trainable discriminator head. The local recipe name S\_16 is an alias, not its patch size: the backbone uses 14 × 14 patches. The generator’s adversarial term activates at the configured Stage-2 global-step threshold of 5,000, after discriminator warm-up; this threshold is separate from the Stage-1 600k-step duration. We use hinge loss, diferentiable augmentation probability 1.0 with cutout 0.2, and lazy R1 regularization with weight 10 every 16 discriminator updates. The adaptive generator coeficient is the ratio of reconstruction and adversarial gradient norms at the output layer, clipped at 10<sup>4</sup> and multiplied by 0.5.

## A.2 LoopVAE-Multi Configuration

The five encoder scales are 2x, 4x, 8x, 16x, and 32x. The shared core contains six Transformer blocks at width 768, with 12 attention heads, and uses loop schedule [1, 1, 2, 3, 6]. Attention uses 8 × 8 windows at the two finest scales and global connectivity at the remaining scales. Posterior heads are attached to scale indices 2, 3, and 4. Per-interface losses are weighted by 0.5, 1.0, and 1.5; each uses KL weight 10<sup>−6</sup>. Stage 2 uses a convolutional discriminator with depth setting three and adaptive adversarial weight multiplier 0.2, rather than the main CNN run’s DINO-based discriminator.

The multi-entry decoder has a separate input projection for each latent channel count and loop list [6, 3, 2, 1, 1] in 32x-to-2x traversal order. Starting from 8x skips the 16x and 32x synthesis stages; starting from 16x skips only 32x; starting from 32x executes the complete reverse hierarchy. These entry points require 24, 42, and 78 core block applications, respectively, all using the same six stored blocks.

## A.3 Evaluation Protocol

For reconstruction, all validation images are encoded with the posterior mode and decoded once. PSNR, LPIPS, and SSIM are averaged over paired images after mapping tensors to the same image range. rFID uses the same 50k ImageNet validation images as both real-image identities and reconstruction sources.

Reported scores are individual checkpoint evaluations, not averages across training seeds. Table 1 gives the approximately 30-epoch, two-stage f8d16 results, while the sharing ablation compares the Transformer-based global, scale-wise, and fully unshared variants at Stage 1, step 400k, without adversarial training. The ablation reports paired reconstruction metrics. Its run records do not establish identical data ordering or seed-level uncertainty.

The final reconstruction evaluations, 610k CNN diagnostics and generation runs, joint 220k Multi evaluation, and synthetic profiling have separate checkpoint or configuration scopes. The final reconstruction checkpoint identifiers have not been linked to the diagnostic checkpoints; results are therefore not combined across these evaluations. Latent interpolation is not evaluated in this paper.

## A.4 Loop Intervention Protocol and Output Range

The diagnostic in Section 4.4 uses the raw encoder/decoder weights of abla\_looped\_f8\_v3\_cnn\_stage2\_- step610k.ckpt, loaded strictly without an EMA substitution. The measurement manifest records the checkpoint SHA256 beginning 81b02d86d5b9fab9, the resolved configuration, source-code hashes, and each input image’s identity and hash. The test images originate from TokBench [26], whose emphasis on faces and text motivates the case study. Sixteen unique source-file hashes were verified. Within the local test-image collection, the first input is img\_105.jpg, already examined in a pilot; the other inputs are the first 15 remaining paths in sorted order after path deduplication. This is a targeted, non-random selection, not an outcome-ranked sample. Figure 5 uses evenly spaced zero-based indices 0, 5, 10, and 15 from the recorded order.

Images undergo EXIF orientation correction, RGB conversion, bicubic resizing of the shorter side to 256, and a centered 256 × 256 crop. The reference is retained in float32 before casting model inputs to bfloat16. Posterior mode is used throughout. Each image yields 14 prefix records and 14 separately applied single-pass bypass records, plus a normal forward trace of 14 passes. Hooks preserve the baseline output exactly; full-prefix arrays also match the baseline. Bypasses execute the pass before replacing its output, so their runtime cannot be used to claim computational savings.

Let $u = ( \hat { x } _ { \mathrm { r a w } } + 1 ) / 2$ be the mapped RGB output and $y \in [ 0 , 1 ]$ the reference. Standard diagnostic PSNR and SSIM use clip(u, 0, 1). Raw PSNR instead uses −10 log $_ { 1 0 } ( \mathrm { M S E } ( u , y ) )$ with data range one; either PSNR is capped at 120 dB for an exact match. SSIM uses a valid 11 × 11 Gaussian window with $\sigma = 1 . 5$ . LPIPS and rFID are not measured on this subset. The out-of-range pixel fraction counts a pixel if any channel is strictly below zero or above one. Clipping displacement is mean $| u - \mathrm { c l i p } ( u , 0 , 1 )$ |. Raw extrema, channel and pixel-level fractions, and raw errors are saved as scalar diagnostics, whereas saved reconstruction arrays contain clipped RGB.

Out-of-range fraction alone does not measure the severity of distortion. The white-background vehicle image has 55.73% out-of-range pixels at full depth but only 0.00169 mean clipping displacement and a 0.046 dB clipping benefit. Conversely, some one-pass decoder outputs extend as far as −2.67 and 4.09 in mapped RGB. Magnitude and fraction must therefore be interpreted together. The full model averages 32.279 dB clipped and 32.251 dB raw PSNR across this subset.

Statistics use images as the unit, not loops or pixels. Prefix monotonicity is checked per image and stage with tolerance $1 0 ^ { - 6 }$ ; one-pass stages have no adjacent comparison. All 64 multi-pass image–stage pairs are monotone in clipped PSNR, raw PSNR, and SSIM. Excluding the previously inspected pilot leaves 60/60 monotone pairs and 210/210 bypasses with lower PSNR. This sensitivity check reduces dependence on the pilot but does not make the remaining convenience sample random or establish population-level monotonicity.

<table><tr><td>Retained passes</td><td>Clipped PSNR</td><td>Raw PSNR</td><td>Out-of-range pixels</td><td>Clipping displacement</td></tr><tr><td>1</td><td>14.54</td><td>-1.04</td><td>96.72%</td><td>0.86934</td></tr><tr><td>2</td><td>15.26</td><td>3.42</td><td>93.44%</td><td>0.47082</td></tr><tr><td>3</td><td>19.36</td><td>14.45</td><td>68.34%</td><td>0.08287</td></tr><tr><td>4 (full)</td><td>32.28</td><td>32.25</td><td>6.14%</td><td>0.00019</td></tr></table>

Table 7 Decoder f2 prefix diagnostic on 16 images. Values are means over images; the full model is the four-pass row. Clipping displacement is an RGB mean absolute change, not a reconstruction error against the reference. A negative raw PSNR is possible because raw outputs are not restricted to the unit interval.

The trace analysis joins records by image identity, branch, scale, and loop index, yielding 224 unique trace–bypass pairs. RMS is taken over the entire feature tensor, with the pre-injection state as reference. Feature-update ranges describe variability across images rather than uncertainty intervals. All qualitative panels use the measured, clipped RGB arrays with a common display mapping; no per-image contrast normalization is applied. These measurements characterize sensitivity within the trained checkpoint and do not establish semantic specialization or population-level monotonicity.

## A.5 Generation and Profiling Details

Generation records. Both generators use the Stage-2 610k CNN tokenizer and are evaluated after 200k generator updates using 50,000 generated images, with class conditioning and without CFG. The XL/2 difusion-model run corresponds to 80 epochs, independently of the tokenizer’s two-stage training budget. Their unrounded FID values are 39.90569705879807 for B/2 and 18.19847535025633 for XL/2; Table 4 reports three decimals. A saved training configuration difers from the reported evaluation in tokenizer path and guidance setting, so it is not treated as a complete record of the evaluation sampler. No comparison to a matched alternative tokenizer is available.

Profiling protocol. The evaluator casts parameters and bufers to bfloat16, times 100 batched encoder–sample– decoder passes after ten warm-ups, and synchronizes CUDA before and after timing. Arithmetic is counted with a batch-one input and reported per image. Memory uses max\_memory\_allocated/1024<sup>2</sup>, hence MiB. The measurement excludes data loading, difusion sampling, and discriminator execution. The 32x V5 row selects the 32x posterior and decoder entry but retains the multi-head encoder computation. All reported execution measurements follow the profiling protocol in Section 4.7.