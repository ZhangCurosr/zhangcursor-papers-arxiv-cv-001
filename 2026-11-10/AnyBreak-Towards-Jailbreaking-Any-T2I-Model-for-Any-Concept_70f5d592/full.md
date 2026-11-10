# AnyBreak: Towards Jailbreaking Any T2I Model for Any Concept

Qi Lu<sup>1†,B</sup>, Zehui Guo<sup>1†</sup>, Yucheng Zhao<sup>1†</sup>, Zijing Li<sup>1</sup>, Chuhan Song<sup>2</sup>, Wenrui Zhu<sup>1‡</sup>

<sup>1</sup>Huazhong University of Science and Technology

<sup>2</sup>University College London

Abstract—Text-to-image (T2I) models have demonstrated remarkable capabilities in image generation and editing, yet this progress has also introduced significant security risks. Prior work has shown that adversarial prompt can induce the generation of Not-Safe-For-Work (NSFW) content, including pornographic, violent, and graphic imagery. However, such attacks are typically mitigated by prompt filters and often rely on white-box assumptions. In this paper, we propose AnyBreak, a novel jailbreak framework that constructs adversarial images via multi-scale progressive semantic alignment. During the generation process, these adversarial images steer the generation trajectory in latent space towards unsafe data manifolds, enabling cross-modal transfer of malicious concepts from the text modality to the image modality. AnyBreak operates using only clean textual prompts and adversarial images at inference time, and can jailbreak any T2I models towards any malicious concepts in a fully black-box setting. We further introduce AnyBreak-V, an enhanced variant that incorporates a compensation term to bypass post-hoc safety checkers. Extensive experiments demonstrate that AnyBreak achieves high attack success rates while preserving visual fidelity and semantic consistency with the input prompts under black-box and clean-text constraints. Our findings reveal that existing T2I defense leave critical cross-modal attack surfaces insufficiently addressed. The code is available at GitHub.

Index Terms—Text-to-image model, Jailbreak attack

## I. INTRODUCTION

In recent years, text-to-image (T2I) models, such as Midjourney, DALL·E, and Stable Diffusion [1], have achieved unprecedented advancements in image synthesis. These models are capable of generating high-quality images in response to provided instructions. Their strong performance derives from large-scale pretraining on massive-scale datasets, a benefit that is inherently double-edged. Inevitably, such datasets contain inappropriate or illegal content, leading T2I models to internalize malicious concepts. As a result, adversaries can exploit this latent knowledge via crafted inputs to elicit Not-Safe-for-Work (NSFW) outputs, including violent, gory, or sexually explicit imagery [2]. This vulnerability effectively turns powerful generative models into a Pandora’s box, posing substantial social and security risks. As illustrated in Fig. 1, adversaries inject carefully designed jailbreak instructions into T2I models and successfully induce the generation of diverse NSFW images spanning multiple harmful concepts.

![](images/2dd59bd70ce5fdf307edfa338568fcc0d866501391ea13ca4445cd91f910c00a.jpg)  
Fig. 1. Jailbreaking T2I models for NSFW generation.

To mitigate such risks, existing defenses primarily rely on prompt filtering [3], [4] and concept erasure [5]. Prompt filtering maintains blacklists of sensitive keywords, while concept erasure methods seek to remove malicious concepts from model parameters. Nevertheless, recent studies show that these defenses remain vulnerable by constructing adversarial prompts. Representative approaches such as [6]–[20] employ token substitutions to preserve malicious semantics while evading detection, whereas [21] conducts red-teaming by extracting unsafe concepts to steer generation accordingly.

Despite their effectiveness, these approaches share a funda mental limitation: they exclusively focus on malicious prompt construction. As a result, they can be effectively neutralized by simple pre-hoc text filtering. As shown in Fig. 2, even lightweight perplexity-based filters can drastically reduce the attack success rate (ASR) of existing prompt-based methods to below 20%. Motivated by this, [22] introduces an image-toimage jailbreak attack that induces NSFW using adversarial images with clean text inputs. However, it assumes whitebox access to the target model and requires retraining the noise generator from scratch for each malicious concept, rendering it impractical and costly in realistic attack settings. As summarized in Tab. I, most existing works assume whitebox access and concentrate on text-side attacks, which are both unrealistic in real-world scenarios and highly vulnerable to simple filtering strategies. This leads to a critical challenge:

## Is it possible to jailbreak any T2I models for any malicious concepts with clean prompts?

In this work, we answer this question affirmatively and uncover an underexplored security vulnerability of T2I models.

TABLE I  
COMPARISON OF EXISTING T2I JAILBREAK METHODS. ”TIME COST” DENOTES THE TIME REQUIRED TO CONSTRUCT A JAILBREAK PER ATTACK. “●” DENOTES PRESENCE AND “❍” DENOTES ABSENCE.
<table><tr><td>Method</td><td>Venue</td><td>Attack Modality</td><td>Threat Model</td><td>Time Cost</td><td>Universal Effectiveness</td></tr><tr><td>QF [7]</td><td>CVPR&#x27;23</td><td>Text</td><td>White-box</td><td>0.43h</td><td>O</td></tr><tr><td>MMA [6]</td><td>CVPR&#x27;24</td><td>Text + Image</td><td>White-box</td><td>5.18h</td><td>O</td></tr><tr><td>Ring [21]</td><td>ICLR&#x27;24</td><td>Text</td><td>Black-box</td><td>1.19h</td><td>O</td></tr><tr><td>Sneaky [8]</td><td>S&amp;P&#x27;24</td><td>Text</td><td>White-box</td><td>8.29h</td><td>O</td></tr><tr><td>ADVI2I [22]</td><td>ICML&#x27;25</td><td>Image</td><td>White-box</td><td>20.27h</td><td>O</td></tr><tr><td>Ours</td><td></td><td>Image</td><td>Black-box</td><td>0.017h</td><td>●</td></tr></table>

We show that even when pre-hoc text filters are deployed, T2I models can still be induced to generate NSFW content by adversarial images. We propose AnyBreak, a universal jailbreak framework towards any T2I model for any malicious concept in a black-box setting. The key insight behind Any-Break is to transfer malicious concepts from the text modality to the image modality, and to construct universal adversarial perturbation across diverse T2I models. Specifically, AnyBreak first employs a generator to transform arbitrary malicious text into corresponding target image. It then optimizes adversarial noise over a suite of image encoders [23], to enable crossmodal transfer of malicious concepts. AnyBreak performs semantic alignment between clean and target images to generate adversarial images. When these adversarial images are paired with innocuous text and fed into any T2I model, the injected noise diverts the original latent away from the normal data manifold into domains associated with malicious concepts, ultimately triggering NSFW generation. To further assess the potential impact of our approach, we introduce an enhanced variant of AnyBreak, AnyBreak-V, which incorporates an additional loss term to facilitate bypassing safety checkers.

Our principal contributions are summarized as follows: 1 We identify a critical security vulnerability in T2I models and propose AnyBreak, a black-box jailbreak framework that enables attacks on any T2I models for any malicious concepts. 2 We propose a universal adversarial noise construction strategy, with theoretical analysis showing that pixel-level semantic alignment can steer data manifolds toward NSFW regions. 3 We conduct extensive experiments across diverse models, malicious concepts, and defense mechanisms, demonstrating the effectiveness of AnyBreak and its variant.

## II. RELATED WORKS

## A. Attacks on T2I Models

T2I models are trained on broadly sourced datasets that may inadvertently harbor unsafe content, exposing exploitable vulnerabilities that can elicit NSFW outputs. Early studies, such as QF [7] and SneakyPrompt [8] both rely on token substitution. Ring [21] exploited text encoders to embed malicious concepts into prompts. MMA [6] further introduces attacks on the image modality but still relies on malicious text prompts at inference time. As discussed in Sec. I, such text-driven approaches are readily mitigated by prompt filters. AdvI2I [22] proposes an adversarial image attack framework, but it requires training a noise generator from scratch for each malicious concept, incurs high computational cost. Moreover, it assumes white-box access to the target model, which limits its practical applicability. In contrast, AnyBreak operates under black-box setting, relying solely on adversarial images paired with clean prompts to induce NSFW generation at inference.

![](images/2f1e13d8df8e38611c10d84d40f5f95a101a095d59edb51374c49464c90c6704.jpg)  
Fig. 2. Impact of prompt filtering strategies on existing methods. Filterbased [24] defenses rely on blacklists, perplexity-based [27] reject highperplexity prompts, and concept-based [3] operate via embedding similarity.

## B. Defense on T2I Models

Defenses for T2I models can be categorized into active and passive approaches. Active defenses, such as concept erasure [2], suppress harmful concepts by modifying noise prediction or finetuning the model. Passive defenses include pre-hoc filtering [3], [24] that sanitizes prompts, and post-hoc safety checkers [25], [26] which detect and block NSFW content in generated images. While existing attacks often bypass pre-hoc filters, they are usually caught by post-hoc checkers. AnyBreak-V overcomes this by introducing a compensation loss that effectively evades safety checkers.

## III. METHODOLOGY

## A. Preliminaries

Latent Diffusion Models (LDMs). Given an input image $x \in \mathbb { R } ^ { H \times W \times C }$ , a LDM $F _ { \theta }$ first maps x into a low dimensional latent space via a pretrained variational autoencoder (VAE), yielding a latent representation $z ~ \in ~ \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times C ^ { \prime } }$ For image inpainting task, $F _ { \theta }$ employ mask conditioning. Let $M \ \in \ \{ 0 , 1 \} ^ { H \times \overline { { W } } }$ be a binary mask, where 1 denotes regions to be inpainted and 0 denotes preserved regions. In the continuous time limit, the latent denoising process follows a reverse stochastic differential equation (SDE) [28]:

$$
\mathrm { d } z = \big [ f ( z , t ) - g ( t ) ^ { 2 } \nabla _ { z } \log p _ { t } ( z , M ) \big ] \mathrm { d } t + g ( t ) \mathrm { d } \bar { w } _ { t } .\tag{1}
$$

Black-box Jailbreak Attack. Let $x _ { c }$ denote a clean input image, and let $x _ { t }$ be a target image associated with a malicious concept $c _ { m } .$ . The attacker optimizes the adversarial perturbation $\delta$ using a set of image encoders, denoted by $\{ f _ { i } \} _ { i = 1 } ^ { n } { \mathrm { : } }$

$$
\operatorname* { m a x } _ { \delta } \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathrm { S i m } \big ( f _ { i } ( x _ { c } + \delta ) , f _ { i } ( x _ { t } ) \big ) \quad \mathrm { s . t . } \ \| \delta \| _ { \infty } \leq \Delta .\tag{2}
$$

![](images/095e80a6cc46a0446e00bbe12e5cbdd22e65442ce8bf137ff64281cb84ed0ed3.jpg)  
Fig. 3. Framework of AnyBreak: a universal and black-box jailbreak framework towards any T2I model for any malicious concept.

![](images/8a4bbe2e1011b482882c5de2d7fb07d135d1e0441a12680a8a761e44b23e1388.jpg)  
Fig. 4. A: clean image with clean text; B: clean image with malicious text; C: malicious image with clean text; D: malicious image with malicious text. For each setting, 100 image–text pairs are sampled to compute the average distance between generated images and the unsafe domain $\dot { D } _ { m }$

Here, Sim denotes a similarity metric, such as cosine distance, and $\Delta$ specifies the perturbation budget. The objective of the attacker is to ensure that for any black-box T2I model $F _ { \theta }$

$$
\boldsymbol { x } _ { \mathrm { n s f w } } = F _ { \theta } ( \boldsymbol { x } _ { c } + \delta , p , \boldsymbol { M } ) ,\tag{3}
$$

where p denotes the textual prompt and $x _ { \mathrm { n s f w } }$ denotes the generated NSFW image generated by jailbreak attack.

## B. Observations and Insights

In this subsection, we analyze the underlying mechanism of NSFW generation from the perspective of malicious concepts. We further examine the philosophy of semantic alignment under Eq. (2), with the goal of gaining insights into the design of universal adversarial perturbations.

Observation I: Malicious images or text both induce malicious concepts. Since NSFW content inherently contains malicious concepts, we investigate its generative mechanism by constructing four experimental settings, as illustrated in Fig. 4. The malicious text concept and image concept are both obtained following [2]. In addition, we predefine an unsafe domain $D _ { m }$ using a set of malicious concept vectors. Under the four settings, we generate four groups of images according to Eq. (3) and compute their average distances d to $D _ { m } ,$ , where $0 < d < 1$ . As shown in Fig. 4. The distribution of B, C, and D lie close to $D _ { m }$ , with distances below 0.3. In contrast, setting A is far from $D _ { m } ,$ with a distance close to 0.8. These results indicate that conditioning the denoising process on either a malicious image or a malicious text embedding consistently steers the predicted noise distribution toward the unsafe domain. This observation suggests that NSFW generation fundamentally originates from malicious concepts, and that text based and image based malicious concepts exert similar effects on the denoising dynamics of $F _ { \theta }$

TABLE II  
EFFECT OF PERTURBATION MAGNITUDE ∆ ON CLIP SIMILARITY AND ASR UNDER GLOBAL ALIGNMENT.
<table><tr><td>Δ</td><td>2/255</td><td>4/255</td><td>8/255</td><td>16/255</td><td>32/255</td></tr><tr><td>Similarity</td><td>0.66</td><td>0.67</td><td>0.70</td><td>0.76</td><td>0.85</td></tr><tr><td>ASR (%)</td><td>46.43</td><td>50.21</td><td>56.49</td><td>59.96</td><td>57.19</td></tr></table>

Insight I: Malicious concepts in either modality can induce NSFW generation, motivating transfer of textual malicious concepts to the image modality to induce NSFW outputs.

Observation II: Global similarity fails to achieve semantic alignment via adversarial noise. A common noise optimization strategy leverages pretrained image encoders [29], [30]. Under this paradigm, one can optimize adversarial noise by maximizing the feature similarity between clean and target images, as formulated in Eq. (2). We vary the perturbation budget from 2/255 to 32/255 to analyze the relationship between similarity and attack effectiveness. As shown Tab. II, increasing $\Delta$ consistently improves feature similarity but yields only marginal gains in ASR. Even at $\Delta = 3 2 / 2 5 5$ , the ASR only converges to about 58% while feature similarity approaches 0.85. This gap indicates that global similarity maximization alone fails to align semantic, as uniform optimization ignores local structural anisotropy and drives semantic collapse.

Insight II: Directly optimizing global similarity overlooks   
local semantic cues, motivating position-aware noise opti  
mization for progressive semantic alignment.

TABLE III  
PERFORMANCE OF ANYBREAK AND ANYBREAK-V UNDER BLACK-BOX SETTINGS ACROSS MULTIPLE MODELS AND DEFENSE MECHANISMS.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">w/o Defense</td><td colspan="3">Gaussian Noise</td><td colspan="3">Safety Checker</td><td colspan="3">Negative Prompt</td></tr><tr><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="2"> $\mathrm { S D } { - } 1 . 5 ^ { [ 1 ] }$ </td><td>AnyBreak</td><td>84.51</td><td>0.26</td><td>0.0708</td><td>81.25</td><td>0.26</td><td>0.0723</td><td>2.78</td><td>0.25</td><td>0.0892</td><td>74.53</td><td>0.25</td><td>0.0723</td></tr><tr><td>AnyBreak-V</td><td>53.54</td><td>0.25</td><td>0.0723</td><td>52.84</td><td>0.25</td><td>0.0736</td><td>46.61</td><td>0.25</td><td>0.0723</td><td>49.87</td><td>0.24</td><td>0.0736</td></tr><tr><td rowspan="2"> $\mathrm { S D } { - } 2 \ ^ { [ 1 ] }$ </td><td>AnyBreak</td><td>85.39</td><td>0.26</td><td>0.0700</td><td>77.17</td><td>0.26</td><td>0.0729</td><td>6.01</td><td>0.25</td><td>0.0713</td><td>82.17</td><td>0.26</td><td>0.0702</td></tr><tr><td>AnyBreak-V</td><td>52.12</td><td>0.25</td><td>0.0724</td><td>49.72</td><td>0.25</td><td>0.0743</td><td>49.29</td><td>0.24</td><td>0.0726</td><td>48.97</td><td>0.25</td><td>0.0720</td></tr><tr><td rowspan="2"> $\mathrm { P 2 P } ^ { [ 3 1 ] }$ </td><td>AnyBreak</td><td>61.78</td><td>0.25</td><td>0.0170</td><td>43.04</td><td>0.25</td><td>0.0250</td><td>3.29</td><td>0.25</td><td>0.0185</td><td>57.11</td><td>0.25</td><td>0.0170</td></tr><tr><td>AnyBreak-V</td><td>29.49</td><td>0.24</td><td>0.0183</td><td>25.70</td><td>0.25</td><td>0.0257</td><td>29.19</td><td>0.24</td><td>0.0172</td><td>26.07</td><td>0.24</td><td>0.0183</td></tr></table>

## C. AnyBreak: A Complete Illustration

Guided by the foregoing insights, we propose AnyBreak, a universal jailbreak attack framework applicable to any T2I models for any malicious concepts. AnyBreak is built upon two cores: (i) cross-modal malicious concepts transfer and (ii) universal adversarial noise optimization under black-box constraints. The complete workflow is shown in Fig. 3.

Malicious Concept Transfer. AnyBreak enables crossmodal transfer of malicious concepts and performs attacks with clean textual prompt. Specifically, the attacker employs a generator, such as [32], to edit a malicious concept $c _ { m }$ into a target image $x _ { t } .$ . Following a strategy similar to Eq. (2) (detailed below), we perform semantic alignment between $x _ { c }$ and $x _ { t }$ in pixel space. Importantly, pixel-level semantic alignment can steer the latent distribution of the clean image $x _ { c }$ toward the unsafe domain associated with the target image $\boldsymbol { x } _ { t } . \operatorname { L e t } p _ { c } ( \boldsymbol { x } )$ denote the distribution of perturbed clean images, and let $\dot { \mathcal { M } } \subset \mathbb { R } ^ { D }$ represent the data manifold induced by malicious images through the reverse SDE process in Eq. (1). Specifically, if the perturbed distribution $p _ { c }$ and the target noise distribution $p _ { t }$ satisfy KL ${ \mathbf { \xi } _ { \left. p _ { c } \mid \ p _ { t } \right. } } \leq \varkappa \left( \mathrm { K I } \right.$ denotes Kullback–Leibler divergence), then, as $\delta  0 ,$ , the probability that generated samples fall within the neighborhood $\mathcal { T } _ { r } ( \mathcal { M } )$ of the unsafe manifold admits the following lower bound [33]:

$$
\mathbb { P } _ { X \sim \rho _ { 0 } } \big [ X \in \mathcal { T } _ { r } ( \mathcal { M } _ { m } ) \big ] \geq ( 1 - \delta ) - \sqrt { \frac { \varkappa } { 2 } }\tag{4}
$$

In summary, we first transfer the malicious concept $c _ { m }$ from text to a target image $x _ { t }$ via a generator, and then leverage a set of encoders $\{ f _ { i } \} _ { i = 1 } ^ { n }$ to semantically align $x _ { c }$ with $x _ { t }$ in the pixel space, thereby realizing cross-modal malicious concept transfer. Eq. (4) further indicates that reducing ${ \varkappa , }$ which corresponds to improving semantic alignment, increases the probability mass of $x _ { c }$ drifting toward unsafe domains during denoising process, consistent with empirical observations. A rigorous proof of Eq. (4) can be found in the appendix.

Universal Adversarial Noise Optimization. As discussed in Insight II, optimizing global similarity alone overlooks local structural discrepancies, which can induce the collapse of adversarial semantic. To address this limitation, we introduce a random local-window alignment mechanism that conducts $K$ rounds of stochastic feature alignment over spatial subregions, enabling multi-scale and progressive optimization while maintaining global coherence. Specifically, in each round, we construct a set of sliding-window candidates over the spatial dimensions. Given a window size $( h , w )$ and stride $v ,$ all valid windows are indexed by their top-left coordinates:

$$
\mathcal { G } = \{ ( i , j ) \in [ 0 , H - h ] \times [ 0 , W - w ] \mid i , j \equiv 0 ( \bmod v ) \} .\tag{5}
$$

Since the encoder operates on square inputs, we set $h = w$ in practice and define the cropping ratio as $\textstyle r \ = \ { \frac { h } { H } } . \ \operatorname { A t }$ each optimization iteration, we independently sample local windows from $\mathcal { G }$ for the clean image $x _ { c }$ and the target image $x _ { t } ,$ and apply the corresponding crops separately. The cropping operators are denoted by $\mathcal { C } r _ { c }$ and $\mathcal { C } r _ { t }$ , where $r _ { c }$ and $r _ { t }$ represent the cropping ratios for the clean and target images, respectively, with $0 < r _ { c } , r _ { t } < 1$ . We optimize the adversarial noise using fast gradient sign method (FGSM) [34] on a set of image encoders $\{ f _ { i } \} _ { i = 1 } ^ { n }$ as:

$$
\operatorname* { m a x } _ { \delta } \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathrm { C o s } \Big ( f _ { i } \big ( \mathcal { C } _ { r _ { c } } ( x _ { c } + \delta ) \big ) , \ f _ { i } \big ( \mathcal { C } _ { r _ { t } } ( x _ { t } ) \big ) \Big ) .\tag{6}
$$

Here, Cos denotes cosine similarity, and $| \delta | _ { \infty } \leq \Delta$ . Through progressive iterations, the adversarial perturbation gradually captures both global semantics and local discrepancies.

## D. AnyBreak-V: Evasion on safety checker

In practical deployments, T2I models are typically equipped with a post-hoc safety checker ( [25], [26]) after the generation stage. This checker compares the generated image against a set of predefined NSFW concept embeddings using a similarity metric, and blocks the output once the similarity exceeds a prescribed threshold. Our empirical results show that this mechanism can be effectively circumvented by introducing an additional compensation term into AnyBreak, yielding its variant AnyBreak-V, which successfully evades the safety checker. Specifically, we simulate the checker’s operating principle by following SLD [2] to select M images that contain malicious concepts, such as pornographic or violent content, and extract their embeddings $\{ c _ { i } \} _ { i = 1 } ^ { M }$ . Based on Eq. (6), these embeddings are incorporated as a compensation term that explicitly drives the generated image away from the malicious concept embedding space:

$$
\operatorname* { m a x } _ { \delta } - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathrm { C o s } \Big ( f _ { i } ( x _ { c } + \delta ) , ~ c _ { i } \Big ) .\tag{7}
$$

TABLE IV  
PERFORMANCE OF DIFFERENT METHODS ON SD-1.5 UNDER VARIOUS DEFENSE MECHANISMS. BEST-PERFORMING RESULTS ARE BOLDED.
<table><tr><td rowspan="2">Method</td><td colspan="3">w / o defense</td><td colspan="3">Gaussion Noise</td><td colspan="3">Safety Checker</td><td colspan="3">Negative Prompt</td></tr><tr><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td><td>ASR(%) ↑</td><td>C-s ↑</td><td>LPIPS ↓</td></tr><tr><td>MMA [6]</td><td>42.21</td><td>0.24</td><td>0.3937</td><td>26.48</td><td>0.24</td><td>0.4143</td><td>39.95</td><td>0.24</td><td>0.3894</td><td>46.30</td><td>0.23</td><td>0.3874</td></tr><tr><td>MMA-Mask [6]</td><td>22.21</td><td>0.24</td><td>0.3937</td><td>15.29</td><td>0.24</td><td>0.4143</td><td>19.64</td><td>0.24</td><td>0.3894</td><td>44.77</td><td>0.23</td><td>0.3874</td></tr><tr><td>ADVI2I [22]</td><td>62.82</td><td>0.22</td><td>0.1436</td><td>50.07</td><td>0.22</td><td>0.2421</td><td>10.27</td><td>0.22</td><td>0.1450</td><td>59.14</td><td>0.22</td><td>0.1410</td></tr><tr><td>AnyBreak</td><td>84.51</td><td>0.26</td><td>0.0708</td><td>81.25</td><td>0.26</td><td>0.0723</td><td>2.78</td><td>0.25</td><td>0.0892</td><td>74.53</td><td>0.25</td><td>0.0723</td></tr><tr><td>AnyBreak-V</td><td>53.54</td><td>0.25</td><td>0.0723</td><td>52.84</td><td>0.25</td><td>0.0736</td><td>46.61</td><td>0.25</td><td>0.0723</td><td>49.87</td><td>0.24</td><td>0.0736</td></tr></table>

![](images/238134d3b68482efbe0e35a0a2ad7ee596910fcf250270a706955b4e26d644eb.jpg)  
(a) module

![](images/bf11fe971d789194c289c8f84004cad3740c169033a859e4c673d824b532f21e.jpg)  
(b) window ratio r  
TABLE V

![](images/7a7946dcb6246792001895b90c2478bd23d9ae5c96980e65abbb5d398f346a69.jpg)  
(c) optimization iteration K

![](images/89ac7f527ca5f53f27d3ed0276e875656bd7c44045f030e04e9f0c4effd35405.jpg)  
Fig. 5. Ablation study on hyperparameters. In figure (b)–(d), bars denote CLIP-score, while lines indicate ASR.  
MMA  
(d) perturbation budget ∆  
AdvI2I  
Ours

Original  
ASR(%) ACROSS DIFFERENT MALICIOUS CONCEPTS.
<table><tr><td>Model</td><td>Adult</td><td>Bloody</td><td>Violence</td><td>Horror</td><td>Racism</td></tr><tr><td>SD-1.5</td><td>84.5</td><td>83.5</td><td>82.5</td><td>71.0</td><td>87.5</td></tr><tr><td>SD-2.0</td><td>86.5</td><td>79.5</td><td>80.5</td><td>66.0</td><td>78.5</td></tr></table>

During the adversarial noise optimization stage, AnyBreak-V is obtained by directly incorporating Eq. (7) into the objective in Eq. (6) on top of AnyBreak.

## IV. EXPERIMENTS

## A. Experimental Settings

Datasets. We conduct experiments on the ”sexy” category of the NSFW Data Scraper [35].

Models. We evaluate on SD-V1.5-Inpainting (SD-1.5), SD-V2-Inpainting (SD-2) and InstructPix2Pix (P2P).

Defenses. We evaluate the robustness of AnyBreak under multiple defense mechanisms. As an input-level defense, we inject random Gaussian noise into adversarial images to disrupt optimized perturbations [36]. As a passive output-side defense, we employ a post-hoc safety checker [25]. We further include Negative Prompt [1] as a concept erasure mechanism applied during the denoising process to erase malicious concepts.

Metrics. We report attack success rate (ASR) to measure attack effectiveness, defined as the number of $x _ { \mathrm { n s f w } }$ that successfully jailbreak into NSFW outputs. We also evaluate image quality. Specifically, CLIP-score (abbr. C-s) [23] measures the consistency between $x _ { \mathrm { n s f w } }$ and the original malicious textual concept $c _ { m } ,$ while LPIPS [37] assesses perceptual quality.

Parameter Setting. In our experiments, we set $r _ { c } , r _ { t } \gets$ [0.5, 1.0], the iterations $K \ = \ 3 0 0 .$ , the perturbation budget $\Delta = 1 6 / 2 5 5$ , the standard deviation of the Gaussian noise to 0.05, and employ ViT-B/32 for CLIP-score computation.

![](images/e61da58bbe75465e0e5f198b1c2d4df9514df0df9fcbace63b926ad7913dcb4e.jpg)  
Fig. 6. Visualization of jailbreak output images across different methods.

## B. Results and Analysis

As shown in Tab. III, AnyBreak demonstrates consistently strong performance across multiple models. Without defenses, it achieves an average black-box ASR of up to 77%. AnyBreak-V underperforms AnyBreak, as the compensation loss steers perturbation away from certain malicious concepts, thereby reducing adversarial semantic effectiveness, at the cost of enhanced ability to bypass safety detectors. AnyBreak exhibits strong robustness, with the average ASR decreasing by less than 10% under random Gaussian noise and negative prompt. While safety checkers provide effective protection, AnyBreak-V evades these defenses via the compensation loss, resulting in an average ASR drop of less than 5% relative to the no-defense setting. Moreover, AnyBreak achieves an average C-s of up to 0.25, compared to approximately 0.28 measured directly between input prompts and generated images of T2I models. The average LPIPS is as low as 0.05, indicating that $x _ { \mathrm { n s f w } }$ preserves high visual fidelity. We also evaluate generalization across malicious concepts, as shown in Tab. V, AnyBreak enables universal attacks across concepts.

Experimental details can be found in the appendix.

## C. Comparison Study

We select ADVI2I [22] and MMA [6] as representative image-side jailbreak baselines. For fair comparison, we remove text-level sensitive keywords from MMA, referred to as MMA-mask. As shown in Tab. IV, AnyBreak consistently outperforms all baselines across multiple defense strategies in both ASR and image generation quality. Notably, under safety checker deployment, where all baselines fail, AnyBreak-V still achieves a high ASR. Visual results in Fig. 6 further confirm that AnyBreak induces NSFW content more effectively while maintaining superior visual fidelity.

## D. Ablation Study

Impact of module. We conduct ablation studies on concept transfer (Module A) and multi-scale noise optimization (Module B), as shown in Fig. 5(a). When module A is removed, the ASR decreases substantially across all models. Without module B, consistent with Observation II, similarly results in a notable ASR degradation.

Impact of r. Fig. 5(b) evaluates different ranges of r while keeping the overall variation span fixed. The results indicate that the multi-scale noise strategy is highly robust and exhibits low sensitivity to the choice of window size.

Impact of K. As shown in Fig. 5(c), adversarial optimization converges within a relatively small number of iterations, and increasing K has limited influence on the final performance. Impact of δ. Fig. 5(d) shows that ASR increases with a larger perturbation budget. However, excessive perturbations degrade image generation quality, as reflected by a decrease in LPIPS.

## V. CONCLUSION

In this paper, we propose AnyBreak, a jailbreak framework that targets any T2I models for any concepts for NSFW generation. AnyBreak enables cross-modal transfer of malicious concepts by constructing universal adversarial noise through multi-scale semantic alignment, allowing successful jailbreak attacks using only cleanprompts and adversarial images at inference time. We further introduce the variant AnyBreak V, designed to evade safety checkers. Our findings expose a critical security vulnerability in current T2I models: even with deployed prompt filters, models can still be induced to generate NSFW content through adversarial images.

## REFERENCES

[1] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer, “High-resolution image synthesis with latent diffu-¨ sion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695.

[2] Patrick Schramowski, Manuel Brack, Bjorn Deiseroth, and Kristian¨ Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22522–22531.

[3] Runtao Liu, Ashkan Khakzar, Jindong Gu, Qifeng Chen, Philip Torr, and Fabio Pizzati, “Latent guard: a safety framework for text-to-image generation,” in European Conference on Computer Vision. Springer, 2024, pp. 93–109.

[4] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al., “Laion-5b: An open largescale dataset for training next generation image-text models,” Advances in neural information processing systems, vol. 35, pp. 25278–25294, 2022.

[5] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu, “Ablating concepts in text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 22691–22702.

[6] Yijun Yang, Ruiyuan Gao, Xiaosen Wang, Tsung-Yi Ho, Nan Xu, and Qiang Xu, “Mma-diffusion: Multimodal attack on diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 7737–7746.

[7] Haomin Zhuang, Yihua Zhang, and Sijia Liu, “A pilot study of queryfree adversarial attack against stable diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2385–2392.

[8] Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, and Yinzhi Cao, “Sneakyprompt: Jailbreaking text-to-image generative models,” in 2024 IEEE symposium on security and privacy (SP). IEEE, 2024, pp. 897– 912.

[9] Qi Lu, Zehui Guo, David Yuanda Gan, Zijing Li, Hengda Zhang, Weijun Xu, and Qiankun Zhang, “Tempjail: Temporal jailbreak attacks against image-to-video generation models,” arXiv preprint arXiv:2608.26971, 2026.

[10] Qi Lu, Ziqi Zhou, Yufei Song, Zijing Li, Lulu Xue, Minghui Li, Shengshan Hu, and Leo Yu Zhang, “Imperceptible and reversible adversarial examples against vision-language models for privacy protection,” arXiv preprint arXiv:2607.10329, 2026.

[11] Yufei Song, Ziqi Zhou, Qi Lu, Hangtao Zhang, Yifan Hu, Lulu Xue, Shengshan Hu, Minghui Li, and Leo Yu Zhang, “Segtrans: Transferable adversarial examples for segmentation models,” IEEE Transactions on Multimedia, 2026.

[12] Zhen Fang, Yu Zeng, Wenxuan Huang, Yiming Zhao, Shiting Huang, Tianfei Ren, Qi Lu, Qingnan Ren, Qisheng Su, Lionel Z Wang, et al., “Video-deepresearch: Towards the next-generation multimodal deepresearch agent,” in Findings of the Association for Computational Linguistics: EMNLP 2026, 2026, pp. 40113–40131.

[13] Zhenghao Gao, Shengjie Xu, Zijing Li, Meixi Chen, Chaojian Yu, Yuanjie Shao, and Changxin Gao, “Fastjsma: Accelerating jacobianbased saliency map attacks through gradient decoupling,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2025, pp. 1506–1515.

[14] Shengjie Xu, Lingxi Xie, Rujie Dai, and Zehua Lyu, “Dumpling gnn: hybrid gnn enables better adc payload activity prediction based on the chemical structure,” International Journal of Molecular Sciences, vol. 26, no. 10, pp. 4859, 2025.

[15] Wei Yang, Yiran Zhu, Zesheng Li, Chengchang Pan, Honggang Qi, et al., “A dynamic prognostic prediction method for colorectal cancer liver metastasis,” arXiv preprint arXiv:2505.03123, 2025.

[16] Zesheng Li, Wei Yang, Yan Su, Yiran Zhu, Yuhan Tang, Haoran Chen, Chengchang Pan, and Honggang Qi, “4d-acfnet: A 4d attention mechanism-based prognostic framework for colorectal cancer liver metastasis integrating multimodal spatiotemporal features,” arXiv preprint arXiv:2503.09652, 2025.

[17] Wei Yang, Yiran Zhu, Zilin Li, Xunjia Zhang, and Hongtao Wang, “Selfempowering vlms: Achieving hierarchical consistency via self-elicited knowledge distillation,” arXiv e-prints, pp. arXiv–2511, 2025.

[18] Jie Qin, Wei Yang, Yan Su, Yiran Zhu, Weizhen Li, Yunyue Pan, Chengchang Pan, and Honggang Qi, “Her2 expression prediction with flexible multi-modal inputs via dynamic bidirectional reconstruction,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 2036–2043.

[19] Wencheng Ye, Xiaoyang Yuan, Yi Bin, Hengyu Jin, Liang Peng, Pengpeng Zeng, and Heng Tao Shen, “Riser: Orchestrating latent reasoning skills for adaptive activation steering,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 4627–4644.

[20] Wencheng Ye, Tianshi Wang, Lei Zhu, Fengling Li, Guoli Yang, and Hengtao Shen, “Actdistill: General action-guided self-derived distillation for efficient vision-language-action models,” arXiv preprint arXiv:2511.18082, 2025.

[21] Yu-Lin Tsai, Chia-Yi Hsu, Chulin Xie, Chih-Hsun Lin, Jia-You Chen, Bo Li, Pin-Yu Chen, Chia-Mu Yu, and Chun-Ying Huang, “Ring-a-bell!

how reliable are concept removal methods for diffusion models?,” arXiv preprint arXiv:2310.10012, 2023.

[22] Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, and Lu Lin, “Advi2i: Adversarial image attack on image-to-image diffusion models,” arXiv preprint arXiv:2410.21471, 2024.

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[24] Zongyu Wu, Hongcheng Gao, Yueze Wang, Xiang Zhang, and Suhang Wang, “Universal prompt optimizer for safe text-to-image generation,” arXiv preprint arXiv:2402.10882, 2024.

[25] CompVis, Ezi Ozoani, and Hugging Face Team, “Stable diffusion safety checker,” https://huggingface.co/CompVis/ stable-diffusion-safety-checker, 2022.

[26] Javier Rando, Daniel Paleka, David Lindner, Lennart Heim, and Florian Tramer, “Red-teaming the stable diffusion safety filter,” \` arXiv preprint arXiv:2210.04610, 2022.

[27] Gabriel Alon and Michael Kamfonas, “Detecting language model attacks with perplexity,” arXiv preprint arXiv:2308.14132, 2023.

[28] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[29] Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, and Dit-Yan Yeung, “Anyattack: Towards large-scale self-supervised adversarial attacks on vision-language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 19900–19909.

[30] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Man Cheung, and Min Lin, “On evaluating adversarial robustness of large vision-language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 54111–54138, 2023.

[31] Tim Brooks, Aleksander Holynski, and Alexei A Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18392–18402.

[32] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin, Liangchen Li, et al., “Z-image: An efficient image generation foundation model with singlestream diffusion transformer,” arXiv preprint arXiv:2511.22699, 2025.

[33] Kotaro Sakamoto, Ryosuke Sakamoto, Masato Tanabe, Masatomo Akagawa, Yusuke Hayashi, Manato Yaguchi, Masahiro Suzuki, and Yutaka Matsuo, “The geometry of diffusion models: Tubular neighbourhoods and singularities,” in ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling, 2024.

[34] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014.

[35] alex000kim, “nsfw data scraper: Collection of scripts to aggregate image data for the purposes of training an NSFW image classifier,” https://github.com/alex000kim/nsfw data scraper, 2019.

[36] Robert Honig, Javier Rando, Nicholas Carlini, and Florian Tram¨ er,\` “Adversarial perturbations cannot reliably protect artists from generative ai,” arXiv preprint arXiv:2406.12027, 2024.

[37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 586–595.