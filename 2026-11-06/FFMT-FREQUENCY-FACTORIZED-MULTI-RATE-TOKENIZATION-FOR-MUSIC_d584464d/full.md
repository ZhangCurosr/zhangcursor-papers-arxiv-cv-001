# FFMT: FREQUENCY-FACTORIZED MULTI-RATE TOKENIZATION FOR MUSIC-TO-DANCE GENERATION

Qirong Guo<sup>1∗</sup>, Ruijie Li<sup>1∗</sup>, Zeyan Li<sup>2</sup>, Sicheng Shu<sup>3</sup>, Guanzhi Ding<sup>1</sup>, Weihang Ding<sup>4</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>Shanghai Jiao Tong University <sup>3</sup>Duke University <sup>4</sup>University of California, Berkeley

## ABSTRACT

Dance motion combines slowly varying body configurations with rapid local movements. We investigate how explicit temporalfrequency decomposition can guide the temporal resolution and codebook capacity of discrete motion representations. We introduce frequency-factorized motion tokenization (FFMT), which separates root-relative motion into a low-frequency choreographic scaffold and a complementary high-frequency residual. FFMT uses a lower token rate and a smaller codebook for the low-frequency component, and a higher token rate and a larger codebook for the high-frequency residual. This representation further induces a plan-then-articulate autoregressive factorization: low-frequency tokens establish coarse choreography, while high-frequency tokens realize local motion details conditioned on the generated plan and music. Experiments on AIST++ and FineDance show strong distributional fidelity while preserving realistic diversity and music–motion synchronization. Ablations consistently support frequency factorization, asymmetric temporal allocation, and low-to-high conditioning.

Index Terms— music-to-dance generation, motion tokenization, vector quantization, frequency decomposition, autoregressive generation

## 1. INTRODUCTION

Music-driven 3D dance generation aims to synthesize movements that follow musical structure while maintaining coherent choreography and expressive local articulation. Recent autoregressive and diffusion-based approaches have advanced music-conditioned motion synthesis [1, 2, 3, 4, 5, 6]. For systems built on discrete motion tokens, the tokenizer defines the units over which the generator learns music–motion relationships [7, 8, 9, 10]. Its temporal resolution and representational capacity shape how slowly varying body configurations and rapid local movements are encoded, making the organization of motion tokens an important consideration for capturing the temporal complexity of dance.

Prior work has explored temporal hierarchies in dance tokenization. DanceMeld [11] and DuetGen [9] learn coarse and fine motion codes at different temporal resolutions, with DuetGen conditioning fine-level generation on the coarse tokens. Frequency information has also been incorporated into reconstruction and generation guidance: ExpressiveBailando [12] applies Fourier-domain supervision to enhance motion decoding, while InfiniteDance [13] constructs conditioning features by processing frequency bands of retrieved reference motions. Temporal resolution alone does not specify the frequency content assigned to each stream. We therefore investigate explicit motion-frequency components as quantization targets for allocating token rates and codebook capacity.

This leads to a temporal capacity allocation problem. Token rate controls temporal resolution: a low rate may miss rapid changes, whereas a high rate may redundantly encode slowly varying motion. Codebook size determines the states available per token: a small codebook may merge distinct motion patterns, while a larger one provides more alternatives. The timing and amplitude of local movements must remain coordinated with evolving body configurations, making cross-component dependencies another consideration. The question is how to allocate temporal resolution and codebook capacity across frequency components while preserving their coordination during music-conditioned generation?

We address this question with frequency-factorized motion tokenization (FFMT). A fixed Gaussian filter separates root-relative motion into a low-frequency component and its complementary high-frequency residual before quantization. The low-frequency stream uses a lower token rate and a smaller codebook, while the high-frequency stream uses a higher rate and a larger codebook. A music-conditioned plan-then-articulate generator first predicts lowfrequency tokens, then generates high-frequency tokens conditioned on the low-frequency sequence and music.

Our contributions are threefold: (1) we formulate discrete dance representation as a temporal capacity allocation problem and develop a tokenizer with complementary frequency streams; (2) we allocate frequency-dependent token rates and codebook capacities and couple them with plan-then-articulate generation, where highfrequency dynamics are conditioned on generated low-frequency structure; and (3) we validate these design choices on AIST++ and FineDance through ablations of frequency factorization, multi-rate allocation, cross-frequency conditioning, and the decoding prior.

## 2. METHOD

Overview. Given music M, FFMT generates a 3D dance sequence X through a VQ-VAE + autoregressive pipeline (Fig. 1). The tokenizer produces low- and high-frequency motion codes, whose music-conditioned generation factorizes as:

$$
p ( \mathbf { Z } ^ { \mathrm { L } } , \mathbf { Z } ^ { \mathrm { H } } \mid \mathbf { M } ) = p ( \mathbf { Z } ^ { \mathrm { L } } \mid \mathbf { M } ) p ( \mathbf { Z } ^ { \mathrm { H } } \mid \mathbf { Z } ^ { \mathrm { L } } , \mathbf { M } ) .\tag{1}
$$

Each frequency stream uses upper/lower-body codebooks [2]. The generated codes are decoded into joint rotations and root motion.

## 2.1. Frequency-Factorized Motion Tokenization

Let $\mathbf { P } \in \mathbb { R } ^ { T \times J \times 3 }$ denote a sequence of J 3D joints over T frames. Before tokenization, we remove global root translation to obtain root-relative joint positions P<sup>¯</sup> .

![](images/d5f52c4c9c65e831b92499b9959ec3aeb7c7004e0349232c37fe6cf4e8887eb5.jpg)

(b) Plan-then-Articulate music-conditioned generation  
![](images/f15667adac1b564b759fdd4c14565a1df0b3ccf002d6c5da377df654929fd862.jpg)  
Fig. 1. Overview of FFMT. (a) Frequency-factorized tokenization assigns different temporal resolutions and codebook sizes to low-frequency structure and high-frequency residuals, with separate upper/lower-body codebooks. (b) Plan-then-articulate generation predicts low-frequency tokens first and conditions high-frequency generation on the aligned low-frequency context and music. The resulting tokens are decoded into rotations and root motion, with a training-derived prior used during inference.

We apply a fixed Gaussian low-pass filter along time. For radius R and bandwidth $\sigma ,$ the normalized kernel is

$$
g ( \tau ) = \frac { \exp ( - \tau ^ { 2 } / 2 \sigma ^ { 2 } ) } { \sum _ { u = - R } ^ { R } \exp ( - u ^ { 2 } / 2 \sigma ^ { 2 } ) } , \quad \tau \in [ - R , R ] .\tag{2}
$$

Using replicated boundary values, we define

$$
\mathbf { P } _ { t } ^ { \mathrm { L } } = \sum _ { \tau = - R } ^ { R } g ( \tau ) \bar { \mathbf { P } } _ { t + \tau } , \qquad \mathbf { P } ^ { \mathrm { H } } = \bar { \mathbf { P } } - \mathbf { P } ^ { \mathrm { L } } .\tag{3}
$$

Here $\mathbf { P } _ { t } ^ { \mathrm { L } } \in \mathbb { R } ^ { J \times 3 }$ is the low-frequency component at frame $t ;$ stacking all $T$ frames yields $\mathbf { P } ^ { \mathrm { L } } \in \dot { \mathbb { R } } ^ { T \times j \times 3 }$ . Thus $\bar { \mathbf { P } } = \mathbf { P } ^ { \mathrm { L } } + \mathbf { P } ^ { \mathrm { H } }$ exactly. The high-frequency branch quantizes the complementary residual. This additive construction preserves the complete position signal before quantization while exposing two temporal regimes for separate capacity allocation.

Each frequency component $\mathbf { P } ^ { f } , ~ f ~ \in ~ \{ \mathrm { L } , \mathrm { H } \}$ , is partitioned along the joint dimension into body-part components $\hat { \mathbf { P } ^ { f , b } }$ , where $b \in \{ \mathrm { U } , \mathrm { D } \}$ indexes the upper and lower body, respectively. The two components contain disjoint joint subsets and retain their original joint correspondences. Each of the four frequency–body components is modeled by a dedicated VQ-VAE branch with its own encoder, codebook, and position decoder.

Each branch $( f , b )$ uses an encoder $E _ { f , b }$ with temporal stride $r _ { f }$ frames. For a T-frame input, it produces $T _ { f } = T / r _ { f }$ latent vectors of dimension $d ,$ so $E _ { f , b } ( \mathbf { P } ^ { f , b } ) \in \mathbb { R } ^ { T _ { f } \times d }$ . Let $\dot { \mathcal { C } } ^ { f , \dot { b } } = \{ \mathbf { e } _ { k } ^ { f , b } \} _ { k = 1 } ^ { K _ { f } }$ denote its $K _ { f }$ -entry codebook, with $\mathbf { e } _ { k } ^ { f , b } \in \mathbb { R } ^ { d }$ . At token position $i = 1 , \dots , T _ { f }$ , nearest-neighbor quantization gives

$$
z _ { i } ^ { f , b } = \arg \operatorname* { m i n } _ { k } \left\| E _ { f , b } ( \mathbf { P } ^ { f , b } ) _ { i } - { \mathbf { e } } _ { k } ^ { f , b } \right\| _ { 2 } ^ { 2 } .\tag{4}
$$

FFMT uses $r _ { \mathrm { L } } ~ > ~ r _ { \mathrm { H } }$ and $K _ { \mathrm { L } } ~ < ~ K _ { \mathrm { H } }$ . The larger low-frequency stride gives a lower token rate, paired with a smaller codebook; the high-frequency residual uses a higher token rate and a larger codebook.

## 2.2. Motion Reconstruction and Tokenizer Training

Position reconstruction. Let $D _ { f , b }$ denote the position decoder associated with branch $( f , b )$ . Its token sequence is reconstructed as

$$
\hat { \mathbf { P } } ^ { f , b } = D _ { f , b } ( \mathbf { Z } ^ { f , b } ) , \qquad f \in \{ \mathrm { L } , \mathrm { H } \} , \ b \in \{ \mathrm { U } , \mathrm { D } \} .\tag{5}
$$

For each frequency stream, the decoded body-part components are reassembled in the original joint order to obtain ${ \hat { \mathbf { P } } } ^ { f }$ , yielding the full root-relative reconstruction $\hat { \mathbf { P } } = \hat { \mathbf { P } } ^ { \mathrm { L } } + \hat { \mathbf { P } } ^ { \mathrm { H } }$

Let $\mathbf { h } _ { i } ^ { f , b } = E _ { f , b } ( \mathbf { P } ^ { f , b } ) _ { i }$ denote the encoder output at token position i. Each branch is trained with

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { b r a n c h } } ^ { f , b } = \sum _ { m = 0 } ^ { 2 } { \left\| { \boldsymbol \Delta } ^ { m } { \mathbf { P } } ^ { f , b } - { \boldsymbol \Delta } ^ { m } \hat { { \mathbf { P } } } ^ { f , b } \right\| _ { 1 } } } \\ { \displaystyle \quad \quad + \frac { \beta } { T _ { f } d } \sum _ { i = 1 } ^ { T _ { f } } { \left\| { \mathbf { h } } _ { i } ^ { f , b } - \mathbf { s g } \left( { \mathbf { e } } _ { z _ { i } ^ { f , b } } ^ { f , b } \right) \right\| _ { 2 } ^ { 2 } } } \\ { \displaystyle \quad \quad + \frac { \eta } { T _ { f } d } \sum _ { i = 1 } ^ { T _ { f } } { \left\| { \mathbf { h } } _ { i } ^ { f , b } \right\| _ { 2 } ^ { 2 } } . } \end{array}\tag{6}
$$

The position objective combines the average branch loss with fullmotion reconstruction and temporal consistency:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { p o s } } = \frac { 1 } { 4 } \sum _ { f , b } \mathcal { L } _ { \mathrm { b r a n c h } } ^ { f , b } } } \\ & { } & { + \sum _ { m = 0 } ^ { 2 } \lambda _ { m } \| \Delta ^ { m } \bar { \mathbf { P } } - \Delta ^ { m } \hat { \mathbf { P } } \| _ { 1 } . } \end{array}\tag{7}
$$

All reconstruction and temporal terms are evaluated on the original frame grid after decoding. The branch losses supervise their fixed frequency–body targets, while the full-motion terms jointly optimize the branches through the recombined motion.

Rotation/root decoding. After position-space VQ training, we freeze the position encoders and codebooks and train the motion decoder [7]. For each body part $b , D _ { F } ^ { f , b }$ decodes quantized codes into frame-resolution features, which are fused as

$$
\begin{array} { r } { { \bf F } ^ { b } = \Phi _ { b } \Big ( [ D _ { F } ^ { \mathrm { L } , b } ( { \bf Z } ^ { \mathrm { L } , b } ) ; D _ { F } ^ { \mathrm { H } , b } ( { \bf Z } ^ { \mathrm { H } , b } ) ] \Big ) , } \end{array}\tag{8}
$$

where $[ \cdot ; \cdot ]$ denotes channel concatenation and $\Phi _ { b }$ is a temporal fusion network. The fused features predict joint rotations, with lowerbody features additionally predicting root velocity. Global translation is obtained by integrating the predicted root velocity from a specified initial root position. The decoder is trained with rotation/root reconstruction and temporal consistency objectives.

## 2.3. Plan-then-Articulate Music-Conditioned Generation

After tokenizer training, each dance is represented by four streams $\{ { \bf Z } ^ { \mathrm { L , U } } , { \bf Z } ^ { \mathrm { L , D } } , { \bf Z } ^ { \mathrm { H , U } } , { \bf Z } ^ { \mathrm { \breve { H } , D } } \}$ Let $\boldsymbol { c } _ { t } ^ { f } ~ = ~ ( z _ { t } ^ { f , \mathrm { U } } , z _ { t } ^ { f , \mathrm { D } } )$ denote the upper/lower-body token pair at token timestep t within frequency stream $f .$ Low- and high-frequency token pairs are generated by separate autoregressive Transformers, each comprising stacked Transformer layers. The high-frequency model additionally conditions on the low-frequency tokens. Following Eq. (1), generation proceeds in two stages.

Plan. Because the motion streams use different token rates, music context is encoded at the corresponding temporal resolutions. A local-context music Transformer produces $\mathbf { \check { A } } ^ { \mathrm { L } }$ and $\mathbf { A } ^ { \mathrm { H } }$ by aggregating neighboring music frames and downsampling them to $T _ { \mathrm { { L } } }$ and $T _ { \mathrm { H } }$ steps. The low-frequency Transformer first autoregressively establishes coarse choreography:

$$
p ( \mathbf { Z } ^ { \mathrm { L } } \mid \mathbf { M } ) = \prod _ { t = 1 } ^ { T _ { \mathrm { L } } } p ( c _ { t } ^ { \mathrm { L } } \mid \mathbf { Z } _ { < t } ^ { \mathrm { L } } , \mathbf { A } ^ { \mathrm { L } } ) .\tag{9}
$$

Articulate. The high-frequency stage realizes residual dynamics conditioned on the generated plan. Since the two streams have different token rates, each fine timestep is aligned to the coarse sequence by

$$
\pi ( t ) = 1 + \left\lfloor { \frac { ( t - 1 ) T _ { \mathrm { L } } } { T _ { \mathrm { H } } } } \right\rfloor ,\tag{10}
$$

where $t = 1 , \ldots , T _ { \mathrm { H } }$ and $\pi ( t ) \in \{ 1 , \ldots , T _ { \mathrm { L } } \}$ . With $T _ { \mathrm { H } } = 2 T _ { \mathrm { L } }$ fine timesteps $2 i - 1$ and 2i share coarse index i. A bounded neighborhood around $\pi ( t )$ is embedded as low-frequency context $\mathbf { C } _ { t } ^ { \mathrm { { L } } }$ This alignment selects the coarse tokens used for conditioning; it does not resample their discrete indices. The high-frequency distribution is

$$
p ( \mathbf { Z } ^ { \mathrm { H } } \mid \mathbf { Z } ^ { \mathrm { L } } , \mathbf { M } ) = \prod _ { t = 1 } ^ { T _ { \mathrm { H } } } p ( c _ { t } ^ { \mathrm { H } } \mid \mathbf { Z } _ { < t } ^ { \mathrm { H } } , \mathbf { C } _ { t } ^ { \mathrm { L } } , \mathbf { A } ^ { \mathrm { H } } ) .\tag{11}
$$

Both stages are optimized with next-token cross-entropy. Motiontoken attention is causal, while music attention may use bounded future context because the complete song is available at generation time.

## 2.4. Training-Derived Decoding Prior

We refine autoregressive decoding with transition and beat statistics from the training split. At token timestep t of stream $f ,$ the decoder assigns each candidate pair c a pre-softmax joint logit $s _ { t } ^ { f } ( c )$ obtained from a joint prediction head or by summing the logits of separately predicted upper/lower-body tokens.

Let $\dot { \mathcal { T } } ^ { \hat { f } } ( c _ { t - 1 } ^ { f } )$ contain all token pairs observed immediately after the previously generated pair $c _ { t - 1 } ^ { f }$ , across all positions within training sequences of stream $f .$

For pairs observed in training, the beat correction is

$$
r _ { \mathrm { b e a t } } ^ { f } ( c , b _ { t } ^ { f } ) = \log p _ { \mathrm { t r } } ^ { f } ( c \mid b _ { t } ^ { f } ) - \log p _ { \mathrm { t r } } ^ { f } ( c ) ,\tag{12}
$$

Table 1. Comparison on AIST++ and FineDance. ↓: lower is better; $ :$ closer to GT is better. Bold/underline indicate the best/secondbest reported values for each dataset.
<table><tr><td>Method</td><td> $\mathrm { F I D } _ { k } \downarrow$ </td><td> $\operatorname { F I D } _ { g } \downarrow$ </td><td> $\mathrm { D i v } _ { k }$  →</td><td> $\mathrm { D i v } _ { g }$  →</td><td>BAS→</td></tr><tr><td colspan="6">AIST++</td></tr><tr><td>Ground Truth</td><td></td><td></td><td>8.19</td><td>7.45</td><td>0.2374</td></tr><tr><td>Bailando++ [7]</td><td>21.29</td><td>11.56</td><td>9.31</td><td>6.20</td><td>0.2330</td></tr><tr><td>EDGE [3]</td><td>42.16</td><td>22.12</td><td>3.96</td><td>4.61</td><td>0.2334</td></tr><tr><td>Lodge [4]</td><td>37.09</td><td>18.79</td><td>5.58</td><td>4.85</td><td>0.2423</td></tr><tr><td>TokenDance [8]</td><td>21.55</td><td>11.85</td><td>8.05</td><td>7.12</td><td>0.2313</td></tr><tr><td>FFMT</td><td>9.91</td><td>7.02</td><td>8.25</td><td>6.65</td><td>0.2374</td></tr><tr><td colspan="6">FineDance</td></tr><tr><td>Ground Truth</td><td></td><td></td><td>9.73</td><td>7.44</td><td>0.2120</td></tr><tr><td>Bailando++ [7]</td><td>54.94</td><td>46.29</td><td>6.17</td><td>5.78</td><td>0.2210</td></tr><tr><td>EDGE [3]</td><td>94.34</td><td>50.38</td><td>8.13</td><td>6.45</td><td>0.2116</td></tr><tr><td>Lodge [4]</td><td>45.56</td><td>34.29</td><td>6.75</td><td>5.64</td><td>0.2397</td></tr><tr><td>TokenDance [8]</td><td>47.20</td><td>31.85</td><td>6.62</td><td>6.81</td><td>0.2385</td></tr><tr><td>FFMT</td><td>35.21</td><td>27.66</td><td>9.75</td><td>7.66</td><td>0.2125</td></tr></table>

where $b _ { t } ^ { f } \in$ {beat, nonbeat} labels the corresponding music interval, and $p _ { \mathrm { t r } } ^ { f } ( c \vert b _ { t } ^ { f } )$ and $p _ { \mathrm { t r } } ^ { f } ( c )$ are conditional and marginal training probabilities estimated with additive smoothing. Unobserved pairs receive zero correction. From the inference candidate set $\mathcal { C } _ { t } ^ { f }$ , specified in Sec. 3.1, we select

$$
\begin{array} { r l } & { \hat { c } _ { t } ^ { f } = \underset { c \in \mathcal { C } _ { t } ^ { f } } { \arg \operatorname* { m a x } } \left[ s _ { t } ^ { f } ( c ) + \alpha _ { f } \mathbb { I } \Big ( c \in \mathcal { T } ^ { f } ( c _ { t - 1 } ^ { f } ) \Big ) \right. } \\ & { ~ \quad \quad \quad \quad \quad \left. + \beta _ { f } r _ { \mathrm { b e a t } } ^ { f } ( c , b _ { t } ^ { f } ) \right] . } \end{array}\tag{13}
$$

The indicator applies a fixed signed adjustment $\alpha _ { f }$ to observed transitions, regardless of their counts. The stream-specific weights $\alpha _ { f }$ and $\beta _ { f }$ remain fixed during inference; the prior introduces no additional trainable parameters.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Datasets and metrics. We evaluate on $\mathbf { A I S T + + } \left[ 1 \right]$ and FineDance [14]. We report Frechet distances´ $( \mathrm { F I D } _ { k } / \mathrm { F I D } _ { g } )$ , diversity $( \mathrm { D i v } _ { k } / \mathrm { D i v } _ { g } )$ , and Beat Alignment Score (BAS) [2]. Lower FID is better; diversity and BAS are treated as neutral metrics, with smaller absolute deviations from the corresponding ground-truth statistics preferred. The EDGE, Lodge, and TokenDance values in Table 1 follow the comparison reported by TokenDance [8]. We also include our AIST++ reproduction of Bailando++.

Implementation. Both datasets use four-second motion windows and the same low/high token intervals in physical time, $0 . 2 / 0 . 1 \mathrm { s } .$ The frame strides $( r _ { \mathrm { L } } , r _ { \mathrm { H } } )$ are (12, 6) for AIST++ at 60 fps and (6, 3) for FineDance at 30 fps. Each low/high body-part stream therefore contains 20/40 tokens per window. We use a Gaussian kernel of size 31 with $\sigma = 6 ,$ , both specified in frames, $K _ { \mathrm { L } } = 1 2 8$ $K _ { \mathrm { H } } = 5 1 2 .$ , and 512-D VQ embeddings. The low/high VQ branches use widths 512/1024 and depths $3 / \bar { 6 }$ . For the position loss, both datasets use λ $= 1 , \lambda _ { 1 } = \lambda _ { 2 } = 0 . 2 5 \nonumber$ , a commitment weight $\beta = 0 . 0 2$ , and a latent regularization weight $\eta = 1 0 ^ { - 5 }$

Each autoregressive Transformer contains sequential base and head stacks. The low-frequency model has $6 + 6 = 1 2$ blocks, a hidden dimension of 512, and 8 attention heads on both datasets. The high-frequency model has $7 + 7 = 1 4$ blocks, dimension 512, and 8 heads on $\mathrm { A I S T + + }$ , and $6 + 6 = 1 2$ blocks, dimension 320, and 5 heads on FineDance. The low/high autoregressive context lengths are $1 9 / 3 9 ;$ both stages use 3-layer local music encoders, and high-frequency generation uses a 7-token low-frequency conditioning neighborhood. The inference candidate set $\dot { \mathcal { C } _ { t } ^ { f } }$ combines the 64 token pairs with the largest decoder logits and all observed training-set successors of $c _ { t - 1 } ^ { f } .$ The prior weights $\left( \alpha _ { \mathrm { L } } , \alpha _ { \mathrm { H } } , \beta _ { \mathrm { L } } , \beta _ { \mathrm { H } } \right)$ are $( 0 . 2 5 , - 0 . 3 0 , 0 . 1 2 , 0 )$ on AIST++ and $( 0 , 0 . 0 5 , 0 . 0 6 , 0 )$ on FineDance.

## 3.2. Comparison with State of the Art

Table 1 compares the reported results of FFMT and representative music-to-dance methods. On AIST++, FFMT has the lowest $\mathrm { F I D } _ { k } / \mathrm { F I D } _ { g }$ (9.91/7.02). Its kinetic diversity is close to GT, and its BAS nearly matches the GT statistic. TokenDance is closer to GT in geometric diversity, indicating that the fidelity gains do not uniformly resolve motion-coverage differences.

On FineDance, FFMT achieves the lowest reported $\mathrm { F I D } _ { k } / \mathrm { F I D } _ { g }$ (35.21/27.66), while both diversity measures are closest to the tabulated GT. These results indicate improved distributional fidelity together with diversity statistics that better match the reference data. EDGE has the BAS closest to GT, with FFMT second.

## 3.3. Ablation Study

We examine frequency targets, temporal sampling, codebook allocation, low-to-high conditioning, and prior-based decoding. Table 2 evaluates complete generation systems; Table 3 assesses the information retained by their tokenizers.

The variants change one component of Full at a time. w/o Freq. removes the explicit low-frequency/residual decomposition. Same Rate uses $T _ { \mathrm { L } } = T _ { \mathrm { H } } = 3 0$ tokens per body branch in each foursecond window (strides 8 on AIST++ and 4 on FineDance), retaining $K _ { \mathrm { L } } / K _ { \mathrm { H } } = 1 2 8 / 5 1 2$ . Uniform K uses $K _ { \mathrm { L } } = K _ { \mathrm { H } } = 2 5 6$ , retaining Full’s 20/40 token counts and branch architectures. w/o L→H masks attention from high-frequency queries to low-frequency context tokens, retaining high-frequency history and music. w/o Prior uses greedy predictions without the decoding prior. Other settings follow Full; token-dependent prior statistics are rebuilt from training data when the tokenizer changes.

Generation quality. Full achieves the lowest FID on both datasets in Table 2, with diversity and BAS closest to the corresponding GT among these variants. Removing frequency decomposition, unifying the token rate, or unifying codebook sizes increases FID. Masking low-frequency context produces the largest kinetic FID degradation, supporting its role in coordinating residual dynamics. The decoding prior gives a larger FID improvement on AIST++ than on FineDance.

Tokenizer reconstruction. Held-out motions are encoded, quantized, and reconstructed through the complete rotation/root decoder, without autoregressive generation or prior rescoring. Let $\bar { \mathbf { P } } _ { i } ^ { \mathrm { r e c } }$ denote root-relative joint trajectories recovered from the final decoded motion and $\bar { \mathbf { P } } _ { i }$ their reference. With $H = I - G$ , where G is the Gaussian operator in Eq. (3), normalized high-frequency reconstruction error is

$$
E _ { H } = \frac { \sum _ { i } \| H ( \bar { \bf P } _ { i } ^ { \mathrm { r e c } } ) - H ( \bar { \bf P } _ { i } ) \| _ { F } ^ { 2 } } { \sum _ { i } \| H ( \bar { \bf P } _ { i } ) \| _ { F } ^ { 2 } + \epsilon } ,\tag{14}
$$

where the sums cover the test set and $\epsilon = 1 0 ^ { - 8 }$ . All variants use identical coordinate processing and evaluation filters: kernel size 31, σ = 6 frames, and replicated boundaries, as in the reference decomposition. Normalizing by reference high-frequency energy accounts for the smaller residual amplitude. We also report MPJPE in mm and acceleration error $E _ { A } ,$ , the mean jointwise Euclidean difference between $f _ { s } ^ { 2 } \Delta ^ { 2 } \bar { \bf P } ^ { \mathrm { r e c } }$ and $f _ { s } ^ { 2 } \Delta ^ { 2 } \bar { \bf P } , \mathrm { i n } \mathrm { \Gamma } \mathrm { \bar { m } / s ^ { 2 } }$ . Positions are expressed in metres for $E _ { H }$ and $E _ { A } ; f _ { s }$ is the dataset frame rate, and $E _ { A }$ excludes the two temporal boundary frames. These complementary metrics assess overall motion reconstruction (MPJPE), high-frequency detail $( E _ { H } ) .$ , and motion dynamics $( E _ { A } )$

Table 2. Generation ablations on AIST++ and FineDance. Lower FID is better; diversity and BAS are better when closer to ground truth.
<table><tr><td></td><td> $A I S T + +$ </td><td>FineDance</td></tr><tr><td>Variant</td><td> $\mathrm { F I D } _ { k } \ \mathrm { F I D } _ { g } \ \mathrm { D i v } _ { k } \ \mathrm { D i v } _ { g }$  BAS</td><td> $\mathrm { F I D } _ { k } \ \mathrm { F I D } _ { g } \ \mathrm { D i v } _ { k } \ \mathrm { D i v } _ { g }$  BAS</td></tr><tr><td></td><td>w/o Freq. 28.74 10.436.066.080.216</td><td>56.82 49.687.3810.17 0.202</td></tr><tr><td></td><td>Same Rate 18.92 9.888.846.26 0.260</td><td>48.01 36.348.876.82 0.219</td></tr><tr><td></td><td></td><td>Uniform K 15.16 9.25 8.39 6.18 0.249 39.46 30.07 8.74 7.05 0.220</td></tr><tr><td></td><td>w/o Prior 13.14 7.348.636.48 0.253</td><td>w/o L→H 55.27 24.568.716.39 0.256140.77 48.934.95 11.27 0.224 35.39 27.849.107.78 0.215</td></tr><tr><td>Full</td><td>9.91 7.02 8.25 6.65 0.237</td><td>35.21 27.669.757.66 0.212</td></tr></table>

Table 3. Tokenizer reconstruction ablations. MPJPE: mm; $E _ { H } \colon$ dimensionless; $E _ { A } \colon \mathbf { m } / \mathrm { s } ^ { 2 } .$ Lower is better; bold marks the best values.
<table><tr><td colspan="4"> $A I S T + +$ </td><td colspan="3">FineDance</td></tr><tr><td>Variant</td><td>MPJPE↓</td><td> $E _ { H }$  →</td><td> $E _ { A } ,$  1</td><td>MPJPE↓</td><td> $E _ { H }$  ↓</td><td> $E _ { A } .$  →</td></tr><tr><td>w/o Freq.</td><td>96.647</td><td>0.7338</td><td>17.144</td><td>95.903</td><td>0.4280</td><td>7.332</td></tr><tr><td>Same Rate</td><td>96.797</td><td>0.7175</td><td>17.287</td><td>93.101</td><td>0.4013</td><td>7.248</td></tr><tr><td>Uniform K</td><td>95.518</td><td>0.6862</td><td>16.934</td><td>92.637</td><td>0.3743</td><td>6.719</td></tr><tr><td>Full</td><td>94.459</td><td>0.5514</td><td>15.070</td><td>88.770</td><td>0.3190</td><td>5.256</td></tr></table>

The relative reductions in $E _ { H }$ and $E _ { A }$ are consistently larger than those in MPJPE when Full is compared with each tokenizer ablation on either dataset (Table 3). This pattern highlights improvements in temporal detail and motion dynamics alongside overall pose accuracy. The comparison with Same Rate supports allocating temporal resolution by frequency at a fixed total token count, while the comparison with Uniform K supports the chosen codebook allocation at fixed token rates.

Reconstruction measures how faithfully the tokenizer preserves overall motion, high-frequency detail, and dynamics; generation tests how effectively the model uses this representation under music conditioning. Their consistent gains support frequency-aware tokenization and low-to-high conditioning as complementary mechanisms for preserving and modeling the temporal structure of dance data.

## 4. CONCLUSION

We presented FFMT, a frequency-factorized multi-rate representation that allocates discrete capacity across heterogeneous motion dynamics. Its complementary low/high tokenization induces a planthen-articulate generation hierarchy from coarse choreography to finer residual dynamics. Reconstruction and generation evaluations show improved preservation of high-frequency detail and motion dynamics alongside stronger distributional fidelity. Results on AIST++ and FineDance support frequency factorization, asymmetric capacity allocation, and low-to-high conditioning for music-to-dance generation.

## ACKNOWLEDGMENTS

This work is supported by the Red Bird MPhil program at the Hong Kong University of Science and Technology (Guangzhou).

## REFERENCES

[1] Ruilong Li, Shan Yang, David A. Ross, and Angjoo Kanazawa, “AI choreographer: Music conditioned 3D dance generation with AIST++,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[2] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu, “Bailando: 3D dance generation by actor-critic GPT with choreographic memory,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[3] Jonathan Tseng, Rodrigo Castellon, and C. Karen Liu, “EDGE: Editable dance generation from music,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[4] Ronghui Li, Yuxiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li, “Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[5] Canyu Zhang, Youbao Tang, Ning Zhang, Ruei-Sung Lin, Mei Han, Jing Xiao, and Song Wang, “Bidirectional autoregressive diffusion model for dance generation,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[6] Zhenye Luo, Min Ren, Xuecai Hu, Yongzhen Huang, and Li Yao, “POPDG: Popular 3D dance generation with Pop-DanceSet,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[7] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu, “Bailando++: 3D dance GPT with choreographic memory,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 14192–14207, 2023.

[8] Ziyue Yang, Kaixing Yang, and Xulong Tang, “Token-Dance: Token-to-token music-to-dance generation with bidirectional Mamba,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2026.

[9] Anindita Ghosh, Bing Zhou, Rishabh Dabral, Jian Wang, Vladislav Golyanik, Christian Theobalt, Philipp Slusallek, and Chuan Guo, “DuetGen: Music driven two-person dance generation via hierarchical masked modeling,” in ACM SIGGRAPH 2025 Conference Papers, 2025.

[10] Xiaojie Li, Ronghui Li, Shukai Fang, Shuzhao Xie, Xiaoyang Guo, Jiaqing Zhou, Junkun Peng, and Zhi Wang, “Musicaligned holistic 3D dance generation via hierarchical motion modeling,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[11] Xin Gao, Li Hu, Peng Zhang, Bang Zhang, and Liefeng Bo, “DanceMeld: Unraveling dance phrases with hierarchical latent codes for music-to-dance synthesis,” arXiv preprint arXiv:2401.10242, 2023.

[12] Qiaochu Huang, Xu He, Boshi Tang, Haolin Zhuang, Liyang Chen, Shuochen Gao, Zhiyong Wu, Haozhi Huang, and Helen Meng, “Enhancing expressiveness in dance generation via integrating frequency and music style information,” arXiv preprint arXiv:2403.05834, 2024.

[13] Ronghui Li, Zhongyuan Hu, Li Siyao, Youliang Zhang, Haozhe Xie, Mingyuan Zhang, Jie Guo, Xiu Li, and Ziwei Liu, “InfiniteDance: Scalable 3D dance generation towards in-thewild generalization,” arXiv preprint arXiv:2603.13375, 2026.

[14] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li, “FineDance: A fine-grained choreography dataset for 3D full body dance generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.