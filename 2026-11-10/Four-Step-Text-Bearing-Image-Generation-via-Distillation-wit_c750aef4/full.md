# Four-Step Text-Bearing Image Generation via Distillation with Local Response Regularization and Teacher-Guided Training

Kairui Wen Sheng Ye Yong-Jin Liu

Tsinghua University, China

wkr21@mails.tsinghua.edu.cn yec22@mails.tsinghua.edu.cn liuyongjin@tsinghua.edu.cn <sup>\*</sup>Corresponding author.

Preprint

![](images/0492b3fa3aafadc57d1b174cadf68c33e8bed401812bb03af2d948cfbb0719a5.jpg)  
Garden signage

![](images/3e0f4cbe5970c7d9fd597dd758e1da9a9268a20f1e7425a588246f9edd1e4ae9.jpg)  
Exhibition poster

![](images/a0e47881d4f01e4656a5ce18ada7e67f5c497b5fb9ec01e32db5723672e762dd.jpg)  
Tea-shop signage

![](images/78685a5cfaa8dd6a60307981b779a24d08d3e7ed8bfda359f2f54e8f36875582.jpg)  
Book-fair poster

![](images/f23dcaf110b3e93e18ab21016deb5c766a72589a9df18aeb7c90993c19ebeeb5.jpg)  
Teacher: 50 steps

(a) Text-bearing visual media  
![](images/92519da2a6270bb50a8cc2689a1087c3de0d8c99a1065cb63b96fa1477d5c7e8.jpg)  
No text training: 4 steps

![](images/30ad5069b4cb419a5fc491f224427d655659db810cc5d521d08e98eca1c4f43d.jpg)  
Synthetic text: 4 steps

![](images/cdd86eed367b9337791ffacb7d9d7555cd396a03c9d4f0f27c1752a29497a622.jpg)  
Ours: 4 steps  
(b) Same garden prompt and seed: target string “Welcome”  
4 vs. 50 steps: 12.5× fewer denoiser evaluations | Four-step GenEval: 0.84 (ours) vs. 0.81 (TDM)

Figure 1. Four-step synthesis with preserved literal content. (a) Our “Welcome” garden scene alongside three application illustrations. (b) Matched experimental crops; teal and amber indicate correct and incorrect spelling. Our student recovers the target word with 12.5× fewe denoiser evaluations than the teacher and improves GenEval over four-step TDM.

## Abstract

Generating text-bearing images requires coherent scenes and accurate characters. Large text-to-image models improve both capabilities through repeated denoising, at substantial sampling cost. Few-step distillation reduces this cost but requires the student to organize scenes and refine glyphs within a short trajectory. We present a fourstep distillation framework combining local response regularization with teacher-guided text training. Built on distribution matching, the framework distills the GLM-Image DiT while retaining its autoregressive visual prior, glyph encoder, and VAE. A path-length penalty promotes consistent local responses of the conditional denoising field. Progressive training combines synthetic strings with teachergenerated imagery to preserve character accuracy and

scene-integrated typography. The student achieves GenEval Overall 0.84 and CVTG-2K word accuracy 0.5045, compared with 0.81 and 0.1690 for four-step TDM. A coefficient study favors $1 0 ^ { - 5 }$ , and Qwen-Image experiments show additional gains from response regularization and text supervision. Single-image inference takes 1.742 s, compared with 5.818 s for the teacher, yielding a 3.34× speedup at the same 40 GB GPU memory requirement.

Keywords: text-to-image generation, diffusion distillation, path-length regularization, visual text rendering

## 1 Introduction

Text-to-image diffusion models support diverse visual synthesis for design and content creation [28, 38]. Textbearing images, such as posters, signs, and information panels, place an additional demand on these models: the generated scene must communicate specific written content through legible, correctly positioned characters [5, 38]. Their synthesis therefore requires both scene-level composition and precise local structure. A small character error can alter the intended message even when the surrounding image appears plausible.

Few-step distillation reduces sampling cost by replacing a long denoising trajectory with a small number of learned transitions. Compressing a fifty-step teacher into a fourstep student while retaining its visual prior and conditioning interface supports rapid exploration of text-bearing designs. Diffusion models build image quality through iterative refinement [13, 28, 31, 34]. Distillation accelerates this process through progressive compression, consistency learning, adversarial supervision, or distribution matching [29, 30, 33, 40]. DMD2 extends distributionmatching distillation (DMD) to multi-step students through improved fake-score tracking and inference-state simulation [39]. Trajectory Distribution Matching (TDM) and Phased DMD further improve trajectory supervision and multi-step optimization [7, 24]. The resulting short trajectories must retain the compositional and textual capabilities of the pretrained teacher.

Four-step distillation imposes two related learning demands. Each update must organize object relations while resolving the fine strokes and spacing of requested text. Distribution matching supplies a broad visual target; explicit strings and layouts supply the local constraints needed for literal rendering. We study this setting with GLM-Image [42], whose autoregressive visual prior and glyph conditioning provide a natural interface between scene planning and continuous DiT refinement. We retain these conditioning paths and concentrate distillation on the denoiser.

Our framework combines a response prior with teacherguided text supervision. We adapt path-length regularization to the conditional denoising field, measuring latent-tooutput responses while holding the visual prior and glyph features fixed. The adaptive target promotes consistent response magnitudes across training states, providing a regularity prior for coarse denoising transitions. Teacher-guided training complements this prior with synthetic examples that specify exact strings and teacher-generated images that integrate typography into scenes. Text complexity progresses from short strings to dense layouts.

We evaluate composition on GenEval [11] and literal rendering on CVTG-2K [5]. Response regularization raises the four-step GLM student from 0.81 to 0.84 GenEval Overall. Teacher-image mixing increases word accuracy from 0.2086 to 0.5045 over synthetic-only training. The complete framework also outperforms four-step TDM on both benchmarks. A coefficient sweep favors 10<sup>−5</sup>, and a second study with Qwen-Image evaluates response regularization alongside a teacher-specific text-training procedure. In single-image inference, the GLM student reduces latency from 5.818 s to 1.742 s while both models require 40 GB of GPU memory.

In summary, this paper makes three contributions:

• We present a four-step framework for text-bearing image synthesis that retains the pretrained semantic and glyph-conditioning interface of GLM-Image.

• We adapt path-length regularization to the conditional denoiser and evaluate its effect on composition and text detail across two DiT teacher families.

• We develop teacher-guided text training that combines explicit strings with scene-integrated typography and improves literal rendering across multiple text-region counts.

## 2 Related Work

Our framework connects efficient image synthesis, diffusion distillation, local response regularization, and visual text generation. We review these areas and distinguish our response penalty from related forms of differential supervision.

Efficient image synthesis. Latent diffusion enables highresolution synthesis in a compressed representation [28], and DiT scales denoising with transformers [27]. CLEAR, EDiT, LiT, and contiguous-layer pruning reduce computation within individual model evaluations [2, 22, 26, 35]. Bidirectional Multiway Transformers combine visual reconstruction with modality-specific computation [1]. Our framework reduces the number of denoising evaluations while retaining the teacher architecture and conditioning paths.

Few-step generation and distillation. Progressive Distillation, Consistency Models, and Adversarial Diffusion Distillation accelerate synthesis through trajectory compression, self-consistency, and adversarial supervision [29, 30, 33]. Shortcut Models and MeanFlow learn short-step or interval-average dynamics [8, 10]. Continuous-time consistency models improve differential training, and rCM combines this formulation with score distillation at large scale [23, 45].

DMD2 improves online score tracking and introduces inference-state simulation for multi-step students [39, 40]. TDM aligns trajectory distributions with step-aware supervision, and autoregressive distillation uses trajectory history [18, 24]. Phased DMD organizes score matching and model capacity into subintervals [7]. SiD extends scoreidentity distillation to flow-matching DiTs [46]; SMD uses on-policy velocity regression with a training-only residual proxy [43]. Distillation also supports fractional evaluation budgets through caching [19]. Our added response prior acts on local denoiser behavior and complements the distributional training objective with dedicated text supervision.

Differential supervision and local response regularization. StyleGAN2 promotes well-conditioned style-toimage mappings through path-length regularization [17]. We apply its adaptive response-magnitude principle to a time-conditioned denoising field with fixed semantic and glyph features. GAD aligns teacher and student directional derivatives at initial or intermediate noisy states [15]. Our target is a running scalar response scale, which regularizes the magnitude profile of the student field. The supervised mapping and target distinguish these uses of Jacobian products. Table 1 compares the training mechanisms, and Section 4.4 defines our objective.

Visual text generation. Text-bearing synthesis combines scene composition, character identity, and spatial layout. Seedream 2.0 uses bilingual modeling and glyph-aligned text representations, while Qwen-Image employs dedicated text data and staged training [12, 38]. GLM-Image couples an autoregressive visual prior with a glyph-conditioned diffusion decoder [42]. X2I transfers multimodal understanding into a DiT through attention distillation [25]. At the application level, PosterMaker combines characterdiscriminative text control with subject-preserving scene generation, and SimplePoster uses character-level position encoding for product-poster synthesis [3, 9]. AMO improves rendering through attention-modulated overshooting at sampling time [14]; TextPecker addresses characterstructure errors through perceptive reward learning [47]. We examine how to preserve these capabilities during distillation, using synthetic examples for explicit strings and teacher-generated images for integrated scene and typography supervision.

Controllable visual media. Controllable media synthesis combines precise conditioning with local structure preservation. Image-to-image diffusion supports semantic scene synthesis [21]; language-guided fashion editing and prompt-embedding inversion support controlled visual modification [20, 37]. Procedural texturing, featureenhanced diffusion, and diffusion-based sketch extraction extend this goal to materials and geometric structure [4, 32, 41]. We address content preservation under a short denoising budget, using scene composition and literal text as complementary fidelity criteria.

Evaluating composition and literal fidelity. GenEval evaluates object-level compositional alignment [11], while

TextCrafter introduces CVTG-2K for multiple text regions in complex scenes [5]. TextInVision examines text complexity and character preservation through VAE reconstruction [6]; STRICT tests text length, legibility, and instruction adherence [44]. We pair GenEval with CVTG-2K and inspect accuracy by region count to evaluate both scene organization and literal rendering.

## 3 Problem Formulation and Background

## 3.1 Conditional Synthesis of Text-Bearing Images

Let c specify scene content and literal strings. With a fixed decoder D and conditioning state $\kappa ( c )$ containing visual-prior tokens and glyph features, the student generator is

$$
I _ { \theta } ( z , c ) = D ( \Psi _ { \theta , 1 } \circ \cdots \circ \Psi _ { \theta , K } ( z ; \kappa ( c ) ) ) ,\tag{1}
$$

where z is Gaussian noise and $\Psi _ { \boldsymbol { \theta } , k }$ is one scheduler transition. We use $K = 4$ and retain the teacher’s conditioning modules and decoder. The objective is to preserve scene composition and the literal strings specified by the prompt within this reduced sampling budget.

## 3.2 Noising and Score Parameterization

For a clean latent $x _ { 0 } ,$ , the forward perturbation is

$$
\begin{array} { r } { x _ { t } = \alpha _ { t } x _ { 0 } + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{2}
$$

The conditional score is $\boldsymbol { s } ( \boldsymbol { x } _ { t } , t , \boldsymbol { c } ) = \nabla _ { \boldsymbol { x } _ { t } } \log \boldsymbol { p } _ { t } ( \boldsymbol { x } _ { t } \mid \boldsymbol { c } )$ Teacher and fake-score predictions are converted to this common parameterization before taking their difference. A noise predictor gives $s = - \widehat { \epsilon } / \sigma _ { t }$ . For linear flow interpolation $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon$ with velocity $v = \epsilon - x _ { 0 }$

$$
\begin{array} { r } { \widehat { x } _ { 0 } = x _ { t } - t \widehat { v } , \quad \quad } \\ { \widehat { \epsilon } = x _ { t } + ( 1 - t ) \widehat { v } . } \end{array}\tag{3}
$$

The scheduler determines conversion and timestep interpretation. Our response penalty is defined on the conditional denoising output, with a fixed dimensional normalization.

## 3.3 Distribution Matching with an Online Score Model

Distribution matching compares noised student and teacher distributions through

$$
\mathcal { D } ( \theta ) = \mathbb { E } _ { t } [ w ( t ) D _ { \mathrm { K L } } ( p _ { \theta , t } \Vert p _ { T , t } ) ] .\tag{4}
$$

The teacher supplies a reference score, while an online fakescore model tracks the evolving student distribution [39, 40]. DMD2 further supports multi-step students through inference-state simulation. We retain this framework and add conditional response regularization and text-data construction for capability preservation.

Table 1. Training objectives of the methods most relevant to our work. Our framework integrates a conditional response-magnitude prior with teacher-guided text supervision while retaining the distribution-matching backbone and the teacher’s conditioning interface.
<table><tr><td>Method</td><td>Main supervision</td><td>Relation to the proposed response prior</td></tr><tr><td>DMD2 [39]</td><td>state simulation</td><td>Teacher/fake-score distribution matching; inference- Supplies the base distributional supervision complemented by our conditional response term.</td></tr><tr><td>TDM [24]</td><td>Step-aware trajectory distribution matching</td><td>Supervises trajectory distributions; our added term shapes local input responses.</td></tr><tr><td>Phased DMD [7]</td><td>Progressive subinterval distribution and score matching</td><td>Allocates supervision and capacity by phase; our added term acts on a shared conditional denoiser.</td></tr><tr><td>rCM [45]</td><td>Continuous-time consistency plus score distillation</td><td>Uses differential consistency and teacher scores; our VJP measures response magnitude at a fixed condition and time.</td></tr><tr><td>GAD [15]</td><td>Teacher-student directional derivative alignment</td><td>Matches response vectors to a teacher target; our target is an adaptive scalar response scale.</td></tr><tr><td>StyleGAN2 [17]</td><td>Style-to-image path-length regularization</td><td>Supplies the regularization principle; we apply it to the latent-to- denoising field before guidance.</td></tr><tr><td>Ours</td><td>and teacher-guided text data</td><td>Distribution matching, adaptive response magnitudes, Couples conditional response consistency with literal-text supervi- sion for four-step synthesis.</td></tr></table>

## 4 Method

We introduce the distribution-matching backbone and hybrid conditioning interface in Sections 4.1–4.2. Sections 4.3–4.5 define the joint objective and normalized response penalty. Sections 4.6–4.7 describe progressive text supervision and stream composition.

## 4.1 Distribution-Matching Distillation Backbone

Let T be a pretrained teacher and $S _ { \theta }$ its four-step student conditioned on prompt $c .$ Following DMD2 [39], a frozen teacher supplies the reference score and an online fakescore model tracks the student distribution. For a noised sample $x _ { t } ,$ , the distribution-matching generator gradient is

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { D M } } \propto - \mathbb { E } \left[ w ( t ) \Delta s _ { t } \frac { \partial S _ { \boldsymbol { \theta } } ( \boldsymbol { z } , \boldsymbol { c } ) } { \partial \boldsymbol { \theta } } \right] ,\tag{5}
$$

where $\Delta s _ { t } = s _ { T } ( x _ { t } , t , c ) - s _ { F } ( x _ { t } , t , c )$ compares scores in a common noise parameterization, and w(t) includes noiselevel weighting. The fake-score branch receives four updates per generator update. The student uses the timestep labels 1000, 750, 500, and 250; their conversion to modelinput indices and scheduler noise levels follows the teacherspecific interface.

## 4.2 Hybrid AR–DiT Distillation Interface

GLM-Image combines an autoregressive visual-prior module with a DiT denoiser. We retain the prior and glyphconditioning paths, and optimize the continuous refinement mapping. The autoregressive module $A _ { \phi }$ generates prior tokens

$$
p _ { \mathrm { A R } } = A _ { \phi } ( c ) ,\tag{6}
$$

where c is the text prompt. For text-to-image generation, the generated prior-token IDs are extracted from the AR output and upsampled to the target DiT grid before being supplied

through the prior-token-id input. In parallel, text spans extracted from the prompt are encoded by a glyph encoder $E _ { \mathrm { g l y p h } } \mathrm { : }$

$$
h _ { \mathrm { g l y p h } } = E _ { \mathrm { g l y p h } } ( \mathrm { g l y p h } ( c ) ) .\tag{7}
$$

The student DiT therefore receives the continuous latent, the glyph embedding, the AR-derived prior, and the current timestep:

$$
v _ { \theta } = F _ { \theta } ( x _ { t } , h _ { \mathrm { g l y p h } } , p _ { \mathrm { A R } } , t ) .\tag{8}
$$

The AR prior supplies coarse semantic and spatial structure; glyph features specify the characters to render. Keeping both fixed preserves this division of labor while the student learns a shorter denoising trajectory.

The Flow Matching scheduler advances the continuous latent using the predicted velocity:

$$
x _ { \mathrm { n e x t } } = \mathrm { S t e p } _ { t } ( v _ { \theta } , x _ { t } ) .\tag{9}
$$

This interface reuses the pretrained semantic and character representations at every denoising transition.

## 4.3 Training Objective

We augment the base distribution-matching generator objective with a path-length penalty:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { P L } } \mathcal { L } _ { \mathrm { P L } } .\tag{10}
$$

The fake-score model is optimized separately through its score-learning objective. The additional term regularizes the student denoising output with respect to the current latent, with prior and text conditions held fixed.

## 4.4 Local Response Regularization

Each transition of a four-step student covers a substantial portion of the denoising trajectory. Consistent local responses provide a regularity prior for this compressed mapping. Let $F _ { \theta } ( x _ { t } , c , t )$ denote the conditional denoising output and $J = \partial F _ { \theta } / \partial x _ { t }$ its input Jacobian. Following pathlength regularization [17], we define

![](images/eac1fac94e4266a99e7a4f2a8025dbaebc74028f976cea841ad07988a8af6c83.jpg)  
Figure 2. Four-step distillation with complementary response and data supervision. Frozen autoregressive and glyph encoders retain semantic and character conditions. Distribution matching transfers teacher synthesis behavior, while the path-length term regularizes the student’s local response. Teacher-guided progressive training combines explicit text examples with visually integrated typography; the VAE remains fixed.

$$
\begin{array} { r l } & { { r _ { \theta } } ( x _ { t } , c , t ; y ) = \| J ^ { \top } y \| _ { 2 } , \quad y \sim \mathcal { N } ( 0 , I ) , } \\ & { \quad \quad \quad \quad \mathcal { L } _ { \mathrm { P L } } = \mathbb { E } _ { x _ { t } , c , t , y } \left( { r _ { \theta } } ( x _ { t } , c , t ; y ) - \mathrm { s g } ( a ) \right) ^ { 2 } , } \end{array}\tag{11}
$$

where a is an exponential-moving-average response scale and sg denotes stop-gradient. Reverse-mode differentiation gives the vector–Jacobian product (VJP) $J ^ { \top } y ~ =$ $\nabla _ { x _ { t } } \langle F _ { \theta } , y \rangle$ without materializing J. An input-direction Jacobian–vector product (JVP) is $J v .$ Section 4.5 specifies the normalized statistic and target update.

For Gaussian output directions, the response energy satisfies

$$
\mathbb { E } _ { y } \| J ^ { \top } y \| _ { 2 } ^ { 2 } = \operatorname { t r } ( J J ^ { \top } ) = \| J \| _ { F } ^ { 2 } .\tag{12}
$$

For a fixed target $^ { a , }$ the penalty decomposes as

$$
\begin{array} { r } { \mathbb { E } ( r _ { \theta } - a ) ^ { 2 } = \mathrm { V a r } ( r _ { \theta } ) + ( \mathbb { E } r _ { \theta } - a ) ^ { 2 } . } \end{array}\tag{13}
$$

The objective regularizes response dispersion and aligns the mean with a running scale. The adaptive target encourages consistent local response magnitudes without imposing a fixed bound on the Jacobian norm.

Latent differentiation holds the AR prior and glyph embeddings fixed. Applying the penalty before classifier-free guidance targets the conditional refinement field directly.

## 4.5 Normalized Responses and Adaptive Targets

For input and output dimensions $d _ { i }$ and $d _ { o } .$ we normalize the output projection and use an input root-mean-square

reduction:

$$
\begin{array} { l } { \displaystyle { g _ { b } = \nabla _ { x _ { b } } \frac { \langle F _ { \theta } ( x _ { b } , c _ { b } , t _ { b } ) , y _ { b } \rangle } { \sqrt { d _ { o } } } , } } \\ { \displaystyle { \widetilde { r } _ { b } = \sqrt { \frac { \| g _ { b } \| _ { 2 } ^ { 2 } } { d _ { i } } + \varepsilon } , \qquad \varepsilon = 1 0 ^ { - 8 } . } } \end{array}\tag{14}
$$

Each example has an independent Gaussian $y _ { b }$ . With batch size $B ,$ , the loss and target update are

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { P L } , n } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left( \widetilde { r } _ { b } - \mathrm { s g } ( a _ { n } ) \right) ^ { 2 } , } \\ & { \displaystyle a _ { n + 1 } = \beta a _ { n } + ( 1 - \beta ) \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \mathrm { s g } ( \widetilde { r } _ { b } ) , } \end{array}\tag{15}
$$

where $\beta ~ = ~ 0 . 9 9$ . The previous target defines the current loss, and the detached batch mean updates the target for the next iteration. Per-example differentiation gives $\mathbb { E } _ { y } [ \widetilde { r } ^ { 2 } - \varepsilon ] = \| J \| _ { F } ^ { 2 } / ( d _ { i } d _ { o } )$ for mappings without crosssample coupling. The GLM coefficient sweep uses this fixed normalization.

Response consistency versus derivative matching. A Jacobian-energy penalty suppresses aggregate response magnitude through $\mathbb { E } _ { y } \Vert \dot { \boldsymbol { J } } ^ { \top } \boldsymbol { y } \Vert _ { 2 } ^ { 2 }$ . Our loss instead maintains consistency around an adaptive scale. GAD aligns the directional responses of teacher and student mappings [15]; our added term uses the scalar magnitude of the student VJP. Optimizing this penalty requires higher-order differentiation during training; the deployed denoiser retains the original inference structure.

Table 2. Three-stage text-training curriculum for GLM-Image. Teacher images preserve scene appearance throughout, while synthetic text supervision progresses from short strings to dense layouts.
<table><tr><td>Stage</td><td>Training examples</td><td>Primary purpose</td></tr><tr><td>1</td><td>Generic prompts and teacher images</td><td>Preserve scene fidelity</td></tr><tr><td>2</td><td>Synthetic short text and teacher images</td><td>Learn glyph-level alignment</td></tr><tr><td>3</td><td>Dense layouts and teacher images</td><td>Refine multi-region text</td></tr></table>

Response propagation in coarse transitions. For $\Psi ( x ) = x + \Delta t F _ { \theta } ( x , c , t )$ with fixed conditions,

$$
\Psi ( x + \delta x ) - \Psi ( x ) = ( I + \Delta t J ) \delta x + o ( \| \delta x \| _ { 2 } ) .\tag{16}
$$

The interval $\Delta t$ scales the Jacobian contribution to local propagation. With four transitions, each update carries a substantial part of scene refinement. Response regularization supplies a local prior for these coarse updates, while distribution matching and text supervision determine the target appearance and content. Section 5.6 evaluates the resulting quality balance across regularization weights.

## 4.6 Progressive Text Training

Text training follows the three-stage curriculum in Table 2: generic scenes, short strings, and dense layouts. Teacher-generated images remain present throughout to retain scene appearance and typography. Synthetic examples introduce exact strings and controlled layouts in the second stage.

The second-stage batch distribution is

$$
P _ { 2 } = ( 1 - \pi ) P _ { \mathrm { i m a g e } } + \pi P _ { \mathrm { t e x t ~ b a t c h } } ,\tag{17}
$$

where $P _ { \mathrm { i m a g e } }$ denotes the teacher-image stream and $P _ { \mathrm { t e x t ~ b a t c h } }$ denotes the stream containing synthetic text examples. We insert one text-stream batch after every five image-stream batches, giving $\pi = 1 / 6$ . Text-stream batches also contain teacher-generated images; Section 4.7 specifies the sample composition.

Synthetic examples provide exact character targets; teacher images provide scene-integrated typography. The Qwen-Image study uses a teacher-specific synthetic-data construction procedure [38] to evaluate text-oriented distillation with a second conditioning architecture.

## 4.7 Text-Data Construction and Stream Composition

Generic prompts come from DiffusionDB [36], with exact duplicates removed after whitespace deletion. Synthetic examples place Chinese or English strings on whiteboards, billboards, blackboards, and screens. Vocabulary items supply short text; longer strings and multiple regions provide sentence- and layout-level supervision. Teacher-generated images complement these examples with scene-integrated perspective, material appearance, illumination, and typography.

The curriculum first establishes generic scene synthesis, then introduces short text, and finally emphasizes longer strings and dense layouts. Teacher imagery remains present throughout, connecting explicit character supervision to the teacher’s broader visual distribution.

In stage 2, one text-stream batch follows five imagestream batches. Each image batch contains 128 teacher images; each text batch combines 64 synthetic examples and 64 teacher images. A six-batch cycle therefore contains 704 teacher images and 64 synthetic examples, giving

$$
P _ { \mathrm { 2 , s a m p l e } } = { \frac { 1 1 } { 1 2 } } P _ { T } + { \frac { 1 } { 1 2 } } P _ { S } ,\tag{18}
$$

where $P _ { T }$ and $P _ { S }$ are teacher-image and synthetic-text distributions. The 5:1 batch-stream schedule thus corresponds to an 11:1 sample mixture under this composition. This mixture combines explicit character targets with examples of typography integrated into natural scenes.

## 5 Experiments

## 5.1 Experimental Setup

Models and training. We use GLM-Image as the primary teacher and Qwen-Image for a second-model evaluation. Teachers use 50 denoising steps, while students use four at 1024 × 1024 resolution. Generic distillation draws prompts from DiffusionDB [36]. Training uses AdamW, four fakescore updates per generator update, and a path-length coefficient of $1 0 ^ { - 5 }$ . Text-oriented training follows the threestage schedule in Section 4.6. Sections 5.2 and 4.7 specify optimization and data construction.

Benchmarks and metrics. GenEval [11] evaluates compositional alignment across Single Object, Two Objects, Counting, Colors, Position, and Attribute Binding, together with Overall. CVTG-2K [5] evaluates rendered text with word accuracy (WA), normalized edit similarity (NED), and CLIPScore. WA-k denotes the group of images with k text regions, for $k \in \{ 2 , 3 , 4 , 5 \}$

For a target string s and OCR output sˆ, we use the higheris-better NED convention

$$
\mathrm { N E D } ( s , \hat { s } ) = 1 - \frac { \mathrm { L e v } ( s , \hat { s } ) } { \operatorname* { m a x } ( \lvert s \rvert , \lvert \hat { s } \rvert ) } ,\tag{19}
$$

where Lev is Levenshtein distance; two empty strings have similarity 1. CLIPScore measures image–text representation alignment. The auxiliary VAE study reports reconstruction FID and PSNR, latent scale factor, and empirical response ratios. Downstream EDM2 generation is evaluated with FID and FD-DINOv2.

Component comparisons. We compare the full student with TDM [24] on GLM-Image and with variants that omit path-length regularization or text-oriented training. The GLM-Image study also compares synthetic-only and teacher-image-mixed text training and varies the PL coefficient over $\{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ . The distilled models use a four-step inference budget. Table 3 summarizes the settings.

Table 3. Shared distillation and evaluation settings. Each student uses four denoising transitions while retaining the teacher’s conditioning and decoding interfaces.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Teachers</td><td>GLM-Image and Qwen-Image</td></tr><tr><td>Sampling budget</td><td>Teacher: 50 steps; student: four steps</td></tr><tr><td>Resolution</td><td>1024 × 1024</td></tr><tr><td>Student time labels</td><td>1000, 750, 500, 250</td></tr><tr><td>Update ratio</td><td>Four fake-score updates per generator update</td></tr><tr><td>Response penalty</td><td>Enabled or disabled</td></tr><tr><td>Text training</td><td>None, synthetic-only, or teacher-image-mixed</td></tr><tr><td>GLM batch streams</td><td>Five image batches per text batch</td></tr></table>

![](images/0f01617e3efaf1083ee0571fea899f49992442300a7e40ac44e75c1b3aff6e45.jpg)

![](images/623e5895b1a36f5496e07f734335c846cc05ddc028c3f4a9fb46bb7aff129300.jpg)

![](images/5fcd8ca4263e2bebfcb7ee4182b8dd1c3011088170d843a0ec118e5685021576.jpg)  
Figure 3. Literal text preservation under a shared garden prompt and random seed. The full student renders “Welcome” correctly in four denoising steps, matching the target spelling of the 50- step teacher. Intermediate variants show generation without text training and with synthetic text alone.

At inference, prior tokens and glyph embeddings are computed once and held fixed throughout the four scheduler transitions. With classifier-free guidance, each transition combines conditional and unconditional predictions. Step counts refer to scheduler transitions; retained-module costs are additional to the denoising loop.

## 5.2 Optimization and Inference Protocol

Table 4 lists optimization settings. The student combines distribution-matching and response losses; the online fakescore branch receives four updates per generator update.

Table 4. Optimization settings. Response regularization adds a training term while preserving the student’s inference architecture.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Betas; weight decay</td><td>(0.9, 0.999); 0.01</td></tr><tr><td>Batch size</td><td>16/GPU, 8 GPUs, total 128</td></tr><tr><td>PL target decay</td><td>0.99</td></tr><tr><td>Default PL coefficient</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>CFG scale</td><td>4.0</td></tr><tr><td>Evaluation seed</td><td>42</td></tr><tr><td>Checkpoint selection</td><td>Final checkpoint</td></tr></table>

Table 5. Scheduled denoising workload under classifier-free guidance. Reductions apply to branch-evaluation counts; total latency also includes conditioning and VAE decoding.
<table><tr><td>Model</td><td></td><td>Steps Branch evals. Reduction</td><td></td></tr><tr><td>Teacher</td><td>50</td><td>100</td><td>1×</td></tr><tr><td>TDM</td><td>4</td><td>8</td><td>12.5×</td></tr><tr><td>Ours</td><td>4</td><td>8</td><td>12.5×</td></tr></table>

AR, glyph, and VAE modules remain fixed.

Student timestep labels are 1000, 750, 500, and 250; the teacher-specific scheduler supplies noise levels and modelinput indices. Prior tokens and glyph embeddings are computed once and reused, and the terminal latent is decoded once.

## 5.3 Inference Efficiency

With guidance scale $\gamma ,$ each transition combines

$$
F _ { \mathrm { C F G } } = F _ { \mathrm { u n c o n d } } + \gamma ( F _ { \mathrm { c o n d } } - F _ { \mathrm { u n c o n d } } ) .\tag{20}
$$

Table 5 reports scheduler transitions and conditional/unconditional branch evaluations. Guidance branches can share a batched invocation. The response penalty is evaluated only during training. Writing retainedmodule time as $T _ { \mathrm { f i x e d } }$ and guided-transition time as $T _ { \mathrm { s t e p } } ,$ a simple cost model gives

$$
\frac { T ( 5 0 ) } { T ( 4 ) } = \frac { T _ { \mathrm { f i x e d } } + 5 0 T _ { \mathrm { s t e p } } } { T _ { \mathrm { f i x e d } } + 4 T _ { \mathrm { s t e p } } } .\tag{21}
$$

Table 6 reports measured single-image inference cost for the GLM teacher and our student on an NVIDIA A100 GPU. Timing covers the generation pipeline, including autoregressive generation and text encoding, and is recorded with Python tqdm. GPU memory usage is recorded with an external monitoring tool. Latency decreases from 5.818 s to 1.742 s, corresponding to a 3.34× speedup and a 70.1% reduction in generation time. Both models require 40 GB of GPU memory. In this implementation, DMD compresses the denoising trajectory while retaining the model architecture, conditioning modules, and decoder; it does not compress parameter storage. The measured benefit is therefore faster inference at the same memory requirement.

Table 6. Measured single-image inference cost for GLM-Image on an NVIDIA A100 GPU. Our four-step student reduces latency by 70.1% while retaining the teacher’s 40 GB GPU memory requirement.
<table><tr><td>Model</td><td>Time (s/image)</td><td>Memory (GB)</td><td>Speedup</td></tr><tr><td>Teacher (50 steps)</td><td>5.818</td><td>40</td><td>1.00×</td></tr><tr><td>Ours (4 steps)</td><td>1.742</td><td>40</td><td>3.34×</td></tr></table>

The 12.5× reduction in denoising evaluations is a workload ratio, distinct from the measured 3.34× latency speedup. Retained conditioning and decoding costs limit the acceleration predicted by Eq. 21. At the shared fourstep budget, our student also improves composition and text fidelity over TDM.

## 5.4 Interpreting Visual and Textual Evaluation

GenEval tests objects, counts, colors, and spatial or attribute relations; CVTG-2K tests literal strings within visual compositions. Exact WA is sensitive to substitutions, insertions, and omissions, while NED grades partial character agreement and CLIPScore measures broader image–text compatibility. WA-k groups images by their number of text regions. The qualitative comparison uses a common prompt and seed, pairing complete scenes with the identically positioned text crops in Figure 1.

## 5.5 Main Results on GLM-Image

The full four-step student achieves WA 0.5045, NED 0.7130, and CLIPScore 0.7508 (Table 7), improving over TDM by 0.3355, 0.2865, and 0.0606. Gains occur in every region-count group. For five-region scenes, WA rises from 0.1194 to 0.4022, showing improved preservation of multiple strings within one composition.

Teacher-image mixing raises WA from 0.2086 to 0.5045 over synthetic-only training and improves NED and CLIP-Score. Synthetic-only training also improves on the student without text supervision. These comparisons support combining exact string targets with examples of realistic scene typography. Figure 3 shows the corresponding rendering behavior under a shared prompt and seed.

Compositional fidelity. The full student reaches 0.84 GenEval Overall, compared with 0.81 for TDM and the student without PL (Table 9). Improvements over TDM cover all six categories, including 0.04 in Counting, 0.05 in Colors, and 0.03 in Attribute Binding. The PL ablation further supports its role in preserving object relations and attribute constraints.

Multi-region text. Teacher-image mixing improves WA over synthetic-only training by 0.3100, 0.3446, 0.2958, and

0.2331 for two through five regions. NED improves concurrently, indicating better character-level agreement. Together, these results show that teacher imagery strengthens literal rendering across both sparse and dense text layouts.

## 5.6 Ablation on Regularization Strength

The intermediate weight $\lambda _ { \mathrm { P L } } ~ = ~ 1 0 ^ { - 5 }$ gives the best score on all four metrics (Table 10): WA 0.5045, NED 0.7130, CLIPScore 0.7508, and GenEval Overall 0.84. The larger weight $1 0 ^ { - 4 }$ yields WA 0.4648 and Overall 0.83, while $1 0 ^ { - 6 }$ yields 0.4475 and 0.82. NED and CLIPScore follow the same ranking. Among the tested coefficients, moderate regularization best preserves both character detail and scene composition.

## 5.7 Cross-Model Evaluation on Qwen-Image

On Qwen-Image, response regularization raises GenEval Overall from 0.82 to 0.83, with gains of 0.01 in Two Objects, Counting, Position, and Attribute Binding (Table 11). Single Object and Colors retain their baseline scores.

Dedicated text training raises Qwen-Image WA from 0.3554 to 0.6395 and NED from 0.4520 to 0.8542 (Table 8). Region-wise WA gains range from 0.2194 to 0.3109. Both teacher studies therefore show improved multi-string rendering with dedicated text supervision, using teacher-image mixing for GLM and synthetic text examples for Qwen.

## 5.8 Text Complexity and Capability Preservation

Figure 4 summarizes accuracy across region counts. Teacher-image mixing gives the strongest GLM student performance in every group, including five-region WA 0.4022 versus 0.1691 for synthetic-only training and 0.1194 for TDM. Dedicated Qwen text training raises five-region WA from 0.2751 to 0.5860, extending the benefit to a second teacher family.

Scene fidelity and literal correctness. The full GLM student improves both composition and literal rendering at the same four-step budget as TDM. In Figure 3, teacher-image mixing recovers Welcome while preserving its garden setting. The scene and text are learned as a joint visual output.

Response strength and text detail. The intermediate coefficient yields the strongest WA, NED, CLIPScore, and GenEval results. The consistent ranking across these metrics supports balancing local response regularity with sensitivity to fine text and scene conditions.

## 5.9 Supporting VAE and EDM2 Study

The auxiliary VAE–EDM2 study tests response regularization in a latent representation model (Table 12). VAE reconstruction FID improves from 1.68 to 1.65, and PSNR increases from 25.480 to 25.495. EDM2 trained on the regularized representation improves FID from 5.268 to 4.953 and FD-DINOv2 from 311.906 to 276.889.

Table 7. Text-rendering fidelity on CVTG-2K with GLM-Image. Teacher-image mixing achieves the strongest four-step result on every metric, outperforming TDM and both text-training variants. The teacher uses 50 steps. Higher is better; boldface denotes the best four-step score.
<table><tr><td>Method</td><td>WA-2↑</td><td>WA-3↑</td><td>WA-4↑</td><td>WA-5↑</td><td>WA↑</td><td>NED↑</td><td>CLIPScore↑</td></tr><tr><td>Teacher</td><td>0.9103</td><td>0.9209</td><td>0.9169</td><td>0.8975</td><td>0.9116</td><td>0.9557</td><td>0.7877</td></tr><tr><td>TDM [24]</td><td>0.2179</td><td>0.2088</td><td>0.1300</td><td>0.1194</td><td>0.1690</td><td>0.4265</td><td>0.6902</td></tr><tr><td>Student w/o text training</td><td>0.2032</td><td>0.1758</td><td>0.1087</td><td>0.0951</td><td>0.1457</td><td>0.3717</td><td>0.6895</td></tr><tr><td>Student with synthetic text</td><td>0.2448</td><td>0.2298</td><td>0.1910</td><td>0.1691</td><td>0.2086</td><td>0.5050</td><td>0.7283</td></tr><tr><td>Student with teacher-image mixing</td><td>0.5548</td><td>0.5744</td><td>0.4868</td><td>0.4022</td><td>0.5045</td><td>0.7130</td><td>0.7508</td></tr></table>

Table 8. Text-rendering fidelity on CVTG-2K with Qwen-Image. Dedicated text training improves all region-count groups, WA, NED, and CLIPScore. The teacher uses 50 steps and students use four. Higher is better; boldface denotes the best student score.
<table><tr><td>Method</td><td>WA-2↑</td><td>WA-3↑</td><td>WA-4↑</td><td>WA-5↑</td><td>WA↑</td><td>NED↑</td><td>CLIPScore↑</td></tr><tr><td>Teacher</td><td>0.8370</td><td>0.8364</td><td>0.8313</td><td>0.8158</td><td>0.8288</td><td>0.9116</td><td>0.8017</td></tr><tr><td>Student w/o text training</td><td>0.4684</td><td>0.3576</td><td>0.3203</td><td>0.2751</td><td>0.3554</td><td>0.4520</td><td>0.7785</td></tr><tr><td>Full student</td><td>0.6878</td><td>0.6625</td><td>0.6217</td><td>0.5860</td><td>0.6395</td><td>0.8542</td><td>0.7873</td></tr></table>

Table 9. Compositional fidelity on GLM-Image. Our four-step student improves all six GenEval categories over four-step TDM, raising Overall from 0.81 to 0.84. The teacher uses 50 steps. Higher is better; boldface marks the best distilled-model score.
<table><tr><td>Method</td><td>Single</td><td>Two</td><td>Count</td><td>Color</td><td>Position</td><td>Attr.</td><td>Overall</td></tr><tr><td>Teacher</td><td>0.99</td><td>0.93</td><td>0.89</td><td>0.90</td><td>0.78</td><td>0.77</td><td>0.88</td></tr><tr><td>TDM [24]</td><td>0.97</td><td>0.88</td><td>0.80</td><td>0.80</td><td>0.72</td><td>0.69</td><td>0.81</td></tr><tr><td>Student w/o PL</td><td>0.97</td><td>0.88</td><td>0.81</td><td>0.82</td><td>0.71</td><td>0.68</td><td>0.81</td></tr><tr><td>Full student</td><td>0.98</td><td>0.90</td><td>0.84</td><td>0.85</td><td>0.73</td><td>0.72</td><td>0.84</td></tr></table>

Table 10. PL-weight ablation on GLM-Image after text-oriented training. The intermediate coefficient $1 0 ^ { - 5 }$ gives the best result on all four metrics. Every setting uses four-step inference. Higher is better; boldface marks the best score.
<table><tr><td> $\lambda _ { \mathrm { P L } }$ </td><td>WA</td><td>NED</td><td>CLIPScore</td><td>GenEval</td></tr><tr><td> $1 0 ^ { - 6 }$ </td><td>0.4475</td><td>0.6821</td><td>0.7404</td><td>0.82</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>0.5045</td><td>0.7130</td><td>0.7508</td><td>0.84</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>0.4648</td><td>0.6997</td><td>0.7460</td><td>0.83</td></tr></table>

The VAE sweep identifies different operating points for reconstruction and generation (Table 13). The 1× setting gives the best reconstruction FID; 8× gives the best downstream EDM2 result. EDM2 (FID, FD-DINOv2) is (5.060, 286.826), (4.953, 276.889), and (6.939, 333.322) at 1×, 8×, and 40× PL, respectively. Moderate regularization produces the strongest downstream representation.

Directional response statistics in Section 5.10 characterize the accompanying change in local sensitivity. Their relative spread and absolute scale describe different aspects of the representation.

Table 11. Effect of response regularization on Qwen-Image. PL improves four GenEval categories and raises Overall from 0.82 to 0.83. The teacher uses 50 steps and students use four. Higher is better; boldface marks the best student score, including ties.
<table><tr><td>Method</td><td>Single</td><td>Two</td><td>Count</td><td>Color</td><td>Position</td><td>Attr.</td><td>Overall</td></tr><tr><td>Teacher</td><td>0.99</td><td>0.92</td><td>0.89</td><td>0.88</td><td>0.76</td><td>0.77</td><td>0.87</td></tr><tr><td>Student w/o PL</td><td>0.98</td><td>0.88</td><td>0.83</td><td>0.84</td><td>0.72</td><td>0.70</td><td>0.82</td></tr><tr><td>Full student</td><td>0.98</td><td>0.89</td><td>0.84</td><td>0.84</td><td>0.73</td><td>0.71</td><td>0.83</td></tr></table>

Table 12. Auxiliary VAE reconstruction and downstream EDM2 generation. Response regularization improves reconstruction FID and both generation distances. Lower FID and FD-DINOv2 and higher PSNR indicate better quality. Scale denotes the latent scale factor; response ratios describe empirical sensitivity variation.
<table><tr><td>VAE</td><td>FID</td><td>PSNR</td><td>Scale</td><td>Resp. ratio</td></tr><tr><td>Without regularization</td><td>1.68</td><td>25.480</td><td>0.248</td><td>42.159</td></tr><tr><td>With regularization</td><td>1.65</td><td>25.495</td><td>0.323</td><td>30.629</td></tr><tr><td>SD3 VÁE reference</td><td>1.69</td><td>25.784</td><td>0.209</td><td>38.167</td></tr><tr><td>EDM2 variant</td><td>FID</td><td>FD-DINOv2</td><td></td><td></td></tr><tr><td>Baseline</td><td>5.268</td><td>311.906</td><td></td><td></td></tr><tr><td>Regularized</td><td>4.953</td><td>276.889</td><td></td><td></td></tr></table>

## 5.10 Auxiliary Protocol and Directional Responses

The auxiliary study applies response regularization to a VAE trained with reconstruction, perceptual, adversarial, and KL terms. We use $\lambda _ { \mathrm { a d v } } = \lambda _ { \mathrm { L P I P S } } = 1 , \lambda _ { \mathrm { K L } } = 1 0 ^ { - 6 }$ batch size 192, and target decay 0.99. PL is evaluated once per 1536 images at 1×, 8×, and 40× weights. EDM2 models trained on these representations evaluate downstream generation quality [16].

To probe local behavior, consider a deterministic mapping $G ,$ input $u ,$ and normalized input direction $\delta _ { i } .$ . A finitedifference response is

$$
q _ { i } ( u ; h ) = \frac { \| G ( u + h \delta _ { i } ) - G ( u ) \| _ { 2 } } { h } , \qquad \| \delta _ { i } \| _ { 2 } = 1 .\tag{22}
$$

We evaluate ten directions with $h = 1 0 ^ { - 2 }$ . At small perturbations, $q _ { i }$ approximates $\Vert J _ { G } ( u ) \delta _ { i } \Vert _ { 2 }$ , providing a directional sensitivity probe of the learned representation.

![](images/c7eec2d065281167b194e774188b6ce790a652a7a34e5decca26a73cfcced6c6.jpg)  
Figure 4. CVTG-2K word accuracy across text-region counts. Teacher-image mixing gives the strongest GLM student results and exceeds four-step TDM in every group; dedicated text training also improves Qwen rendering throughout. Curves connect the measurements in Tables 7 and 8.

![](images/31a313b0a88d579a6ff76d380b37a448ae8fbde5752a5b9161f7a89ae8caf256.jpg)  
Figure 5. ImageNet samples and text-scene illustrations arranged in a continuous grid. Rows correspond to the baseline and 1×, 8×, and 40× PL settings. Quantitative comparisons are reported in Tables 12 and 13.

Table 13. VAE regularization-strength sweep. The 1× setting minimizes reconstruction FID, while 8× supports the best downstream generation in Table 12. Scale is the latent scale factor. Lower FID and higher PSNR indicate better reconstruction.
<table><tr><td>VAE</td><td>FID</td><td>PSNR</td><td>Scale</td><td>Resp. ratio</td></tr><tr><td>Baseline</td><td>1.68</td><td>25.480</td><td>0.248</td><td>42.159</td></tr><tr><td> $1 \times \mathrm { P L }$ </td><td>1.64</td><td>25.078</td><td>0.280</td><td>32.833</td></tr><tr><td> $8 \times \mathrm { P L }$ </td><td>1.65</td><td>25.495</td><td>0.323</td><td>30.629</td></tr><tr><td> $4 0 \times \mathrm { P L }$ </td><td>1.86</td><td>25.088</td><td>0.396</td><td>20.020</td></tr></table>

Table 14 reports the absolute response scale and its relative spread. Both Low and High increase with regularization strength, while High/Low decreases. These measurements indicate a narrower relative response profile at a larger absolute scale; the smaller ratio does not imply reduced absolute sensitivity. Reconstruction and generation metrics identify the useful operating point.

Table 14. Directional response statistics for the auxiliary VAE sweep. Low and High report empirical finite-difference response magnitudes; Ratio is High/Low. PL narrows this relative spread, complementing the reconstruction and generation comparisons in Tables 12 and 13.
<table><tr><td>VAE</td><td>Low</td><td>High</td><td>Ratio</td></tr><tr><td>Baseline</td><td>0.069</td><td>2.909</td><td>42.159</td></tr><tr><td> $1 \times \mathrm { P L }$ </td><td>0.102</td><td>3.349</td><td>32.833</td></tr><tr><td> ${ \mathrm { 8 } } \times { \mathrm { P L } }$ </td><td>0.143</td><td>4.380</td><td>30.629</td></tr><tr><td> $4 0 \times \mathrm { P L }$ </td><td>0.247</td><td>4.945</td><td>20.020</td></tr></table>

Reconstruction and downstream generation. The 1× setting gives the best VAE reconstruction FID, whereas 8× gives the best downstream EDM2 FID and FD-DINOv2. The strongest weight raises the latent scale factor to 0.396 and degrades downstream generation. Effective regularization balances response geometry with information preservation in the latent representation.

![](images/cabaacc24451580f35998afe8438e634e11315d54937cd2f35395812c7e74e40.jpg)  
Figure 6. Qualitative comparison of four-step GLM-Image students. Within each three-column group, outputs are ordered as the student without text training, TDM, and our complete framework. Cases cover (a) an English poster, (b) a Chinese poster, (c) a bilingual sign, (d) product packaging, (e) multiline text, (f) a bilingual display, (g) object counting, and (h) spatial relations. The examples illustrate differences in literal text preservation, object counts, and spatial arrangement while retaining coherent scene appearance.

Figure 5 pairs the original ImageNet samples with four illustrative text scenes aligned to each row’s appearance. The strongest setting shows overlapping penguin contours and split vessel structure in the original samples. The additional illustrations highlight letter boundaries, line spacing, and Chinese strokes. TextInVision also examines character preservation through VAE reconstruction [6]. The measured VAE–EDM2 results favor moderate regularization.

## 5.11 Qualitative Comparison across Visual Tasks

Figure 6 compares the student without text training, TDM, and our complete framework across eight cases. The text-bearing examples span English and Chinese posters, bilingual signage, product packaging, a multiline menu, and a bilingual display. They expose character omissions, substitutions, and spacing errors that can remain difficult to distinguish from overall scene plausibility. In the English poster, our student preserves “CITY LIGHTS” and “FRI-DAY 8 PM”; in the menu, it retains the separate lines “COF-FEE 3” and “TEA 2.” The object-counting example additionally shows three red cups and two blue bottles in our output, while the spatial example places the red cube to the left of the blue sphere and the green cone behind it. These examples complement the aggregate text and compositional evaluations in Tables 7 and 9.

## 5.12 Discussion

Our four-step distillation framework combines local response regularization with data tailored to literal text. Response regularization improves compositional alignment, while teacher-guided training supplies exact strings and scene-integrated typography. Their combination exceeds four-step TDM on GenEval and every CVTG-2K metric, raising WA from 0.1690 to 0.5045 at the same denoising budget.

The coefficient study connects these benefits: a moderate response penalty gives the strongest composition and text scores. Retaining the AR prior, glyph encoder, and VAE preserves the pretrained synthesis interface, so the adaptation remains concentrated on the denoiser. The response penalty adds no inference operations. Measured singleimage latency falls by 70.1%, although the retained architecture keeps the GPU memory requirement at 40 GB.

## 6 Limitations

The present study focuses on four-step generation. Dense text layouts still leave room for improvement in character accuracy and placement. Retained conditioning and decoding modules contribute fixed deployment costs, and trajectory compression alone does not reduce the measured single-image memory requirement. Finer glyph supervision, additional sampling budgets, and joint module compression provide direct extensions of the framework.

## 7 Conclusion

We presented a four-step distillation framework for textbearing image generation that combines local response regularization with teacher-guided text training. The method retains GLM-Image’s semantic and glyph-conditioning interface while improving compositional alignment and literal rendering. It raises GenEval Overall from four-step TDM’s 0.81 to 0.84 and CVTG-2K WA from 0.1690 to 0.5045. The coefficient sweep favors 10<sup>−5</sup>, and Qwen-Image experiments show further gains from response regularization and dedicated text data. The student also achieves a 3.34× single-image inference speedup with the same 40 GB GPU memory requirement as the teacher. These results support combining conditional response regularization with dedicated text supervision for efficient image synthesis.

## References

[1] H. Bao, L. Dong, S. Piao, and F. Wei. Text to image generation with bidirectional Multiway Transformers. Computational Visual Media, 11(2):405–422, 2025.

[2] P. Becker, A. Mehrotra, R. Chavhan, M. Chadwick, L. Morreale, M. Noroozi, A. G. C. P. Ramos, and S. Bhattacharya. EDiT: Efficient diffusion transformers with linear compressed attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 19608–19616, 2025.

[3] B. Cui, F. Zeng, W. Jiang, Y. Zhai, H. Hong, L. Huang, H. Xue, W. Shang, and P. Huang. SimplePoster: A simple baseline for product poster generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14854–14863, 2026.

[4] Z. Dang, W. Dong, Z. Yang, B. Yang, L. Li, Y. Ma, and Z. Cui. TexPro: Text-guided PBR texturing with procedural material modeling. Computational Visual Media, 11(4):745– 761, 2025.

[5] N. Du, Z. Chen, Z. Chen, S. Gao, X. Chen, Z. Jiang, J. Yang, and Y. Tai. TextCrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461v1, 2025.

[6] F. Fallah, M. Patel, A. Chatterjee, V. Morariu, C. Baral, and Y. Yang. TextInVision: Text and prompt complexity driven visual text generation benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 525–534, 2025.

[7] X. Fan, Z. Qiu, Z. Wu, F. Wang, Z. Lin, T. Ren, D. Lin, R. Gong, and L. Yang. Phased DMD: Few-step distribution matching distillation via score matching within subintervals. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 41667–41676, 2026.

[8] K. Frans, D. Hafner, S. Levine, and P. Abbeel. One step diffusion via shortcut models. In International Conference on Learning Representations, 2025.

[9] Y. Gao, Z. Lin, C. Liu, M. Zhou, T. Ge, B. Zheng, and H. Xie. PosterMaker: Towards high-quality product poster generation with accurate text rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8083–8093, 2025.

[10] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems, volume 38, 2025.

[11] D. Ghosh, H. Hajishirzi, and L. Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, volume 36, 2023.

[12] L. Gong, X. Hou, F. Li, L. Li, X. Lian, F. Liu, L. Liu, W. Liu, W. Lu, Y. Shi, S. Sun, Y. Tian, Z. Tian, P. Wang, X. Wang, Y. Wang, G. Wu, J. Wu, X. Xia, X. Xiao, L. Yang, Z. Zhai, X. Zhang, Q. Zhang, Y. Zhang, S. Zhao, J. Yang, and W. Huang. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.

[13] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[14] X. Hu, K. Xu, B. Liu, Q. Liu, and H. Fei. AMO Sampler: Enhancing text rendering with overshooting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13157–13166, 2025.

[15] H. Huang, R. Wang, J. Zhao, W. Deng, D. Zhou, J. Luan, Y. Wu, and Y. Zhu. Restoring initial noise sensitivity in textto-image distillation via geometric alignment. arXiv preprint arXiv:2606.01651, 2026.

[16] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila, and S. Laine. Analyzing and improving the training dynamics of diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 24174–24184. IEEE, 2024.

[17] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila. Analyzing and improving the image quality of Style-GAN. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 8110–8119. Computer Vision Foundation / IEEE, 2020.

[18] Y. Kim, S. Anagnostidis, Y. Du, E. Schonfeld, J. Kohler,¨ M. Georgopoulos, A. Pumarola, A. Thabet, and A. Sanakoyeu. Autoregressive distillation of diffusion transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15745–15756, 2025.

[19] H. Li, T. Wen, L. Qi, Z. Wu, Y. Chen, X. Zhou, L. Zhu, X. Wang, and K. Zhang. 1.x-Distill: Breaking the diversity, quality, and efficiency barrier in distribution matching distillation. arXiv preprint arXiv:2604.04018, 2026.

[20] S. Li, J. van de Weijer, T. Hu, F. S. Khan, Q. Hou, Y. Wang, J. Yang, and M.-M. Cheng. StyleDiffusion: Promptembedding inversion for text-based editing. Computational Visual Media, 12(3):743–763, 2026.

[21] F. Liu and X. Chang. IIDM: Image-to-image diffusion model for semantic image synthesis. Computational Visual Media, 11(2):423–429, 2025.

[22] S. Liu, Z. Tan, and X. Wang. CLEAR: Conv-like linearization revs pre-trained diffusion transformers up. In Advances in Neural Information Processing Systems, volume 38, 2025.

[23] C. Lu and Y. Song. Simplifying, stabilizing and scaling continuous-time consistency models. In International Conference on Learning Representations, 2025.

[24] Y. Luo, T. Hu, J. Sun, Y. Cai, and J. Tang. Learning fewstep diffusion models by trajectory distribution matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17719–17728, 2025.

[25] J. Ma, Q. Peng, X. Guo, C. Chen, H. Lu, and Z. Yang. X2I: Seamless integration of multimodal understanding into diffusion transformer via attention distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16733–16744, 2025.

[26] J. Ma, Q. Peng, X. Zhu, P. Xie, C. Chen, and H. Lu. Pluggable pruning with contiguous layer distillation for diffusion transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18640–18650, 2026.

[27] W. Peebles and S. Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4195–4205. IEEE, 2023.

[28] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10684–10695. IEEE, 2022.

[29] T. Salimans and J. Ho. Progressive distillation for fast sampling of diffusion models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.

[30] A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach. Adversarial diffusion distillation. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXVI, Lecture Notes in Computer Science, pages 87– 103. Springer, 2024.

[31] J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In F. R. Bach and D. M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, JMLR Workshop and Conference Proceedings, pages 2256– 2265. JMLR.org, 2015.

[32] H. Song, X. Chen, and S. Lefebvre. FEDNet: A featureenhanced diffusion network for efficient and universal texture synthesis. Computational Visual Media, 12(2):501–508, 2026.

[33] Y. Song, P. Dhariwal, M. Chen, and I. Sutskever. Consistency models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, Proceedings of Machine Learning Research, pages 32211–32252. PMLR, 2023.

[34] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[35] J. Wang, N. Kang, L. Yao, M. Chen, C. Wu, S. Zhang, S. Xue, Y. Liu, T. Wu, X. Liu, K. Zhang, S. Zhang, W. Shao, Z. Li, and P. Luo. LiT: Delving into a simple linear diffusion transformer for image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16068–16078, 2025.

[36] Z. J. Wang, E. Montoya, D. Munechika, H. Yang, B. Hoover, and D. H. Chau. DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 893–911, Toronto, Canada, 2023. Association for Computational Linguistics.

[37] H. Wen, S. You, and Y. Fu. LucIE: Language-guided local image editing for fashion images. Computational Visual Media, 11(1):179–194, 2025.

[38] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S. ming Yin, S. Bai, X. Xu, Y. Chen, Y. Chen, Z. Tang, Z. Zhang, Z. Wang, A. Yang, B. Yu, C. Cheng, D. Liu, D. Li, H. Zhang, H. Meng, H. Wei, J. Ni, K. Chen, K. Cao, L. Peng, L. Qu, M. Wu, P. Wang, S. Yu, T. Wen, W. Feng, X. Xu, Y. Wang, Y. Zhang, Y. Zhu, Y. Wu, Y. Cai, and Z. Liu. Qwen-Image technical report. CoRR, abs/2508.02324, 2025.

[39] T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman. Improved distribution matching distillation for fast image synthesis. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, 2024.

[40] T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, and T. Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 6613–6623. IEEE, 2024.

[41] K. Yun, Y. Kim, K. Seo, C. W. Seo, and J. Noh. Representative feature extraction during diffusion process for sketch extraction with one example. Computational Visual Media, 12(5):1359–1383, 2026.

[42] Z.ai Team. GLM-Image: Auto-regressive for denseknowledge and high-fidelity image generation. Official model release and documentation. https://github. com/zai-org/GLM-Image, 2026. Accessed 25 October 2026.

[43] J. Zhang, Z. Wan, N. Ma, L. Zhao, D. Chen, H. Liu, Q. Chen, and D. Chen. Simplifying and improving onpolicy distillation for few-step diffusion. arXiv preprint arXiv:2610.18472, 2026.

[44] T. Zhang, X. Wang, L. Li, Z. Tai, J. Chi, J. Tian, H. He, and S. Wang. STRICT: Stress-test of rendering image containing text. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21137– 21150, 2025.

[45] K. Zheng, Y. Wang, Q. Ma, H. Chen, J. Zhang, Y. Balaji, J. Chen, M.-Y. Liu, J. Zhu, and Q. Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In International Conference on Learning Representations, 2026.

[46] M. Zhou, Y. Gu, H. Zheng, L. Song, G. He, Y. Zhang, W. Hu, and Y. Yang. Score distillation of flow matching models. arXiv preprint arXiv:2509.25127, 2025.

[47] H. Zhu, Y. Liu, X. Wu, A.-L. Wang, H. Feng, D. Yang, C. Feng, C. Huang, J. Tang, and X. Bai. TextPecker: Rewarding structural anomaly quantification for enhancing visual text rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22059–22069, 2026.

## A Response Spectrum and Directional Variability

Let the denoiser Jacobian have singular value decomposition $J = U \Sigma V ^ { \top }$ . For $y \sim \mathcal { N } ( 0 , I )$ , the coordinates of $U ^ { \top } y$ are independent standard Gaussians. Writing these coordinates as $\xi _ { i }$ gives

$$
\| J ^ { \top } y \| _ { 2 } ^ { 2 } \overset { d } { = } \sum _ { i } \sigma _ { i } ^ { 2 } \xi _ { i } ^ { 2 } .\tag{A.1}
$$

Using $\mathrm { V a r } ( \xi _ { i } ^ { 2 } ) = 2$ and independence,

$$
\operatorname { V a r } _ { y } ( \| J ^ { \top } y \| _ { 2 } ^ { 2 } ) = 2 \sum _ { i } \sigma _ { i } ^ { 4 } .\tag{A.2}
$$

At fixed total squared singular value, concentrating energy in fewer directions increases squared-response variability. This relation characterizes the spectrum sampled by random projections. The main-paper loss applies to response magnitudes and averages over latent states, prompts, and timesteps.

## B Teacher-Referenced Derivative Alignment

For teacher and student mappings with a common input and output space, let $A = J _ { \theta } - J _ { T }$ and $v \sim \mathcal { N } ( 0 , I )$ . The directional alignment objective used in geometry-aware distillation [15] satisfies

$$
\begin{array} { r l } & { \mathbb { E } _ { v } \| A v \| _ { 2 } ^ { 2 } = \mathrm { t r } ( A ^ { \top } A \mathbb { E } [ v v ^ { \top } ] ) } \\ & { \qquad = \| J _ { \theta } - J _ { T } \| _ { F } ^ { 2 } . } \end{array}\tag{B.1}
$$

The teacher supplies a vector-valued derivative target. Our adaptive response prior instead uses the scalar magnitude of a student VJP. This identity makes the target-level distinction precise while leaving the distribution-matching backbone unchanged.

## C Coefficient Scaling under Dimension Normalization

Let $c = d _ { i } d _ { o }$ be the product of input and output dimensions. For the unstabilized response, set

$$
r _ { \mathrm { n o r m } } = \frac { r } { \sqrt { c } } , \qquad a _ { \mathrm { n o r m } } = \frac { a } { \sqrt { c } } .\tag{C.1}
$$

With consistently scaled targets, the corresponding losses satisfy

$$
\mathbb { E } ( r _ { \mathrm { n o r m } } - a _ { \mathrm { n o r m } } ) ^ { 2 } = { \frac { 1 } { c } } \mathbb { E } ( r - a ) ^ { 2 } .\tag{C.2}
$$

Thus, an equivalent coefficient in the normalized objective is $\lambda _ { \mathrm { n o r m } } ~ = ~ c \lambda _ { \mathrm { r a w } }$ . The coefficient convention is tied to the response normalization and latent dimensions. Holding these choices fixed within the GLM sweep gives a consistent comparison of regularization strength.