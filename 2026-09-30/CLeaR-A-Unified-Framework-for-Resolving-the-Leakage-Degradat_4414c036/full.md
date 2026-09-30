# CLeaR: A Unified Framework for Resolving the Leakage–Degradation Dilemma in Style Transfer

Teng Zhou Zhejiang University tengzhou@zju.edu.cn

Yunhao Chen<sup>∗</sup> Fudan University yhchen24@m.fudan.edu.cn

## Abstract

Style transfer aims to render target content in the style of a reference image, but existing methods often suffer from content leakage, where objects, layouts, or semantics from the style reference appear in the generated output. Although prior data-driven and training-free methods can reduce leakage, they often face a leakage-degradation dilemma: stronger content suppression may weaken style fidelity, while richer style preservation may reintroduce unwanted reference content. We identify this dilemma across the full style-transfer pipeline, including feature separation, feature-space grounding, and diffusion generation. To address these issues, we propose CLeaR, a training-free framework for content-leakage-resistant style transfer. CLeaR first uses Orthogonal Subspace Projection to define contentreduced style targets in each vision foundation model (VFM) feature space. It then performs Ensemble Inversion, which optimizes a shared pixel-space style anchor satisfying style constraints across multiple VFMs. Finally, Energy-Guided Calibration maintains style alignment during diffusion sampling by steering the denoising trajectory toward the ensemble-defined style manifold. We further provide a theoretical analysis showing that the style-anchor estimation error decreases with the number of VFMs. Experiments on StyleBench demonstrate that CLeaR improves style alignment, reduces content leakage, and achieves better LLMas-Judge evaluation compared with existing methods. The code is available at https://github.com/0606zt/CLeaR.

## 1 Introduction

Given a style reference image and a target content condition, style transfer (ST) aims to render the target content in the reference style. Since early neural style transfer, this task has been closely tied to content-style (C-S) separation [9]. With diffusion models becoming the dominant backbone for image generation [24, 21], ST is increasingly formulated as conditional generation [42, 51, 4, 12, 40]. However, incomplete C-S disentanglement can cause content leakage, where objects, layouts, or semantics from the style reference appear in the output [54, 26, 40], reducing quality and user control.

To mitigate content leakage, existing methods commonly use visual representations from pretrained or learned encoders to define a style condition that is later injected into a diffusion generator. Data-driven methods train style-aware encoders, disentanglement modules, adapters, or projection layers on paired or curated stylization data [8, 41, 52, 33, 16, 17, 22, 45]. Although effective in specific settings, they require additional training and are often constrained by the training distribution. Trainingfree methods instead manipulate pretrained VFM or diffusion features directly, such as subtracting content-related text features from image features [40], masking feature dimensions, shuffling spatial tokens, modifying attention keys and values, or guiding the denoising trajectory with reference-based objectives [54, 43, 27, 4, 12, 26]. Despite these differences, both paradigms typically define style within a particular feature space and rely on the diffusion model to generate the final image from the resulting edited or guided condition.

Although these methods can reduce content leakage in some cases, they often lack stable control over the boundary between content removal and style preservation. As a result, existing methods tend to fail on different sides of the same dilemma [26]: methods that preserve rich style cues may also retain unwanted reference content, whereas methods that suppress reference content more strongly may remove important stylistic details. We refer to this as the leakage-degradation dilemma.

This dilemma is not caused by a single isolated step. It arises throughout the style-transfer pipeline, from separating C-S features, to grounding the style signal in a representation space, to injecting that signal during diffusion generation.

The first source is the feature separation stage, where content-related and style-related components are separated within a reference representation. Existing operations such as subtraction, masking, patch manipulation, or attention editing can reduce content signals, but are mostly heuristic and may either leave residual content or discard style information correlated with content. To address this issue, we introduce Orthogonal Subspace Projection (OSP), which defines the style target geometrically in each VFM feature space. Given a content proxy, it removes the component of the reference feature aligned with the content direction and retains the orthogonal residual as the style feature.

The second source is thefeature-space grounding stage, where the extracted style signal is grounded in a particular representation space. Most existing methods rely on a single VFM feature space or a single learned representation, which is fragile because different VFMs encode style and content differently [23, 31, 14, 49]. Directly aligning multiple VFM spaces is impractical because their dimensions, metrics, and semantic structures differ, while learning cross-space projectors would add training cost and risk domain overfitting. Instead, we make use of a simpler and more general observation: image pixel space provides a shared domain associated with all VFM feature spaces. This allows heterogeneous style constraints to be combined by optimizing an image itself, rather than by aligning feature spaces directly. Based on this idea, we introduce Ensemble Inversion (EI), which optimizes a learnable image tensor into a shared style anchor whose features satisfy the model-specific style constraints across all VFM branches.

The third source is the diffusion generation stage, where the extracted style condition is injected into the generative process. Standard condition injectors such as IP-Adapter [47] are trained on ordinary image-text pairs rather than on content-reduced style anchors, so the denoising trajectory may drift from the intended style or reintroduce leaked content. To address this issue, we introduce Energy-Guided Calibration (EGC), a test-time guidance mechanism inspired by prior diffusion guidance methods [48, 1]. At selected denoising steps, it computes an ensemble feature-alignment energy between the current decoded image and the target style representation, and uses its gradient to calibrate the latent update toward the VFM-defined style manifold.

In summary, we introduce CLeaR (Content Leakage Resistant), a training-free framework for C-S disentangled style transfer. CLeaR defines content-reduced style targets through Orthogonal Subspace Projection, reconciles heterogeneous VFM targets through Ensemble Inversion, and maintains style alignment through Energy-Guided Calibration. Together, these components reduce content leakage at both the representation-extraction stage and the generation stage.

Our contributions are summarized as follows:

• We propose CLeaR, a training-free framework that extracts a shared pixel-space style anchor via Orthogonal Subspace Projection and Ensemble Inversion, and maintains style alignment through Energy-Guided Calibration.

• We provide a theoretical analysis showing that the style-anchor estimation error decreases with the number of VFMs under limited cross-model dependence.

• Experiments on StyleBench [8] show improvements in style alignment, content leakage metrics, and LLM-as-Judge evaluation.

## 2 Related Work

Style Transfer Gatys et al. [9] formalized style transfer by optimizing deep feature statistics. With diffusion models [24, 21] and conditioning adapters [47], style transfer has increasingly become a conditional generation task, where the central challenge is still C-S separation [42].

Existing methods can be broadly grouped into data-driven and training-free approaches. Data-driven methods learn style representations from paired or curated stylization data [17, 8, 41, 22, 52, 33, 16], or train projection modules for C-S composition [45]. These methods can be effective, but require additional training and are often constrained by the training distribution. Training-free methods instead manipulate pretrained VFM or diffusion features directly, including subtracting textual concepts from image embeddings [40], masking feature dimensions, shuffling spatial patches [54, 43, 18], modifying attention keys/values [27, 4, 12, 11], or adjusting the reverse diffusion trajectory through inversion or test-time optimization [51, 12, 26, 53]. Though effective, these methods often face a leakage-degradation dilemma across the style-transfer pipeline: feature operations may leave residual content or remove useful style cues, single-space representations are fragile across VFMs, and style conditions drift during diffusion sampling. Based on this analysis, CLeaR combines Orthogonal Subspace Projection, Ensemble Inversion, and Energy-Guided Calibration to form a unified solution for content-leakage-resistant style transfer.

Model Inversion Model inversion (MI) techniques originally emerged to analyze privacy risks by reconstructing training data from machine learning models [7, 6]. Early MI approaches primarily target classification networks, using gradient-based optimization and generative priors to reconstruct the original inputs from deep representations [50, 13, 36]. While traditional methods are generally tailored for classifiers, recent advancements [3] have extended inversion paradigms to extract highdimensional visual information from complex, large-scale VFMs successfully.

These inversion techniques provide a natural bridge between feature spaces and the pixel domain. And in our C-S separation process, the image pixel space serves as a universal grounding domain that corresponds to all VFM feature spaces. Leveraging this insight, we employ model inversion to combine heterogeneous representations.

## 3 Methodology

![](images/2432978243d67db2c79793105c2fb0c6e4422bcc74765b04b928cc3ff672d9a8.jpg)  
Figure 1: Overview of the CLeaR framework. The three components target the three failure sources in the entire style transfer process: Orthogonal Subspace Projection (feature separation), Ensemble Inversion (feature-space grounding), and Energy-Guided Calibration (diffusion generation).

CLeaR contains three components that correspond to the three failure sources discussed in Sec. 1. Orthogonal Subspace Projection defines a content-reduced style target in each VFM feature space. Ensemble Inversion combines these model-specific targets by optimizing a shared pixel-space style anchor. Energy-Guided Calibration then keeps the generated image aligned with the target style during diffusion sampling. The overall pipeline is summarized in Fig. 1 and Appendix A.

## 3.1 Orthogonal Subspace Projection

Given a style reference image I and its content description c, our goal is to remove content-related information from the reference feature while retaining style information. Let $F : \mathcal { T }  \mathcal { H }$ be a pretrained VFM, and denote the reference feature as $\mathbf { f } = { \overline { { F ( I ) } } }$ . To obtain a modality-matched content

feature, we generate a content proxy image $I _ { c } = \Phi ( c )$ using a text-to-image diffusion model Φ, and compute $\mathbf { f } _ { c } = F ( I _ { c } ) [ 3 , 1 0 , 1 5 ]$ .

We define the raw residual as

$$
\mathbf { d } = \mathbf { f } - \mathbf { f } _ { c } .\tag{1}
$$

Since d may still contain content-aligned information, we remove the component of d along the content direction $\mathbf { f } _ { c } .$ We seek a style feature $\mathbf { f } _ { s } = \mathbf { d } + \boldsymbol { \delta }$ that is orthogonal to $\mathbf { f } _ { c } ,$ while keeping the adjustment δ as small as possible:

$$
\operatorname* { m i n } _ { \pmb { \delta } } \| \pmb { \delta } \| ^ { 2 } \quad \mathrm { s . t . } \quad \langle \mathbf { d } + \pmb { \delta } , \mathbf { f } _ { c } \rangle = 0 .\tag{2}
$$

Solving them gives

$$
\mathbf { f } _ { s } = \mathbf { d } - \frac { \langle \mathbf { d } , \mathbf { f } _ { c } \rangle } { \Vert \mathbf { f } _ { c } \Vert ^ { 2 } } \mathbf { f } _ { c } = P _ { \mathbf { f } _ { c } ^ { \perp } } ( \mathbf { d } ) ,\tag{3}
$$

where $P _ { \mathbf { f } _ { \alpha } ^ { \perp } }$ denotes projection onto the orthogonal complement of $\mathbf { f } _ { c }$ . Thus, $\langle \mathbf { f } _ { s } , \mathbf { f } _ { c } \rangle = 0$ , and ${ \bf f } _ { s }$ serves as a content-reduced style target in the VFM feature space.

## 3.2 Ensemble Inversion

The projection in Sec. 3.1 defines a style target within one VFM feature space. However, different VFMs capture different aspects of style, and their feature spaces cannot be directly aligned because they may have different dimensions and geometries. Instead, we make use of a simpler observation: image pixel space provides a shared domain associated with all VFM feature spaces.

Let $\{ F _ { k } \} _ { k = \mathrm { \bar { . } } } ^ { K }$ be a set of pretrained VFMs. For each model, we compute

$$
{ \bf f } ^ { ( k ) } = F _ { k } ( I ) , \qquad { \bf f } _ { c } ^ { ( k ) } = F _ { k } ( I _ { c } ) ,\tag{4}
$$

and obtain the model-specific style target

$$
\mathbf { f } _ { s } ^ { ( k ) } = P _ { ( \mathbf { f } _ { c } ^ { ( k ) } ) ^ { \perp } } \left( \mathbf { f } ^ { ( k ) } - \mathbf { f } _ { c } ^ { ( k ) } \right) .\tag{5}
$$

We then optimize a single image $I _ { a }$ whose features match these style targets across all VFM branches. Specifically, let

$$
\mathbf { f } _ { a } ^ { ( k ) } = F _ { k } ( I _ { a } ) .\tag{6}
$$

The ensemble inversion objective is

$$
{ \mathcal { L } } _ { \mathrm { i n v } } ( I _ { a } ) = \sum _ { k = 1 } ^ { K } \lambda _ { k } \left\| \mathbf { f } _ { a } ^ { ( k ) } - \mathbf { f } _ { s } ^ { ( k ) } \right\| ^ { 2 } + \lambda _ { \mathcal { R } } \mathcal { R } ( I _ { a } ) ,\tag{7}
$$

where $\lambda _ { k }$ controls the contribution of each VFM and $\mathcal { R } ( \cdot )$ is a regularizer for spatial smoothness.

Starting from a randomly initialized image tensor $I _ { a } ^ { ( 0 ) }$ , we update the anchor by gradient descent:

$$
I _ { a } ^ { ( t + 1 ) } = I _ { a } ^ { ( t ) } - \eta \nabla _ { I _ { a } } \mathcal { L } _ { \mathrm { i n v } } \left( I _ { a } ^ { ( t ) } \right) ,\tag{8}
$$

where η is the learning rate. After optimization, the resulting image $I _ { a } ^ { * }$ is used as the style anchor. This anchor provides a shared pixel-space representation whose VFM features match the content-reduced style targets across the ensemble.

## 3.3 Energy-Guided Calibration

The style anchor $I _ { a } ^ { * }$ is used as the image condition for diffusion generation. However, standard condition injectors such as IP-Adapter are trained on ordinary image-text pairs and may drift from the intended style during denoising. To reduce this drift, we introduce a test-time Energy-Guided Calibration step, following the idea of gradient-based diffusion guidance [48, 1].

At denoising step t, we first estimate the clean latent from the current noisy latent:

$$
\hat { \bf z } _ { 0 } ( { \bf z } _ { t } ) = \frac { { \bf z } _ { t } - \sigma _ { t } \epsilon _ { \theta } ( { \bf z } _ { t } , t , I _ { a } ^ { * } , y ) } { \alpha _ { t } } ,\tag{9}
$$

where $\alpha _ { t }$ and $\sigma _ { t }$ are determined by the noise schedule. Using the same VFM ensemble as in Ensemble Inversion, we define the style-alignment energy as

$$
\mathcal { E } _ { t } ( \mathbf { z } _ { t } ) = \sum _ { k = 1 } ^ { K } \lambda _ { k } \left\| F _ { k } ( D ( \hat { \mathbf { z } } _ { 0 } ( \mathbf { z } _ { t } ) ) , ) - \mathbf { f } _ { s } ^ { ( k ) } \right\| ^ { 2 } .\tag{10}
$$

where $D$ is the VAE decoder. A lower value of $\mathcal { E } _ { t }$ indicates stronger alignment with the target style representation.

We then use the energy gradient to calibrate the current latent:

$$
\widetilde { \mathbf { z } } _ { t } = \mathbf { z } _ { t } - \rho \nabla _ { \mathbf { z } _ { t } } \mathcal { E } _ { t } \big ( \mathbf { z } _ { t } \big ) ,\tag{11}
$$

where $\rho$ controls the calibration strength. The calibrated latent is passed to DDIM [35] update:

$$
\begin{array} { r } { { \bf z } _ { t - 1 } = \mathrm { D D I M } \left( \tilde { \bf z } _ { t } , \epsilon _ { \theta } ( \tilde { \bf z } _ { t } , t , I _ { a } ^ { * } , y ) , t \right) . } \end{array}\tag{12}
$$

This calibration requires no retraining and improves style alignment by steering the sampling trajectory toward the ensemble-defined style target.

## 3.4 Theoretical Analysis

We analyze how the accuracy of ensemble inversion depends on the number of feature extractors. Let $x ^ { \star } \in \mathbb { R } ^ { \bar { d } }$ denote the ideal content-free style anchor, and let $F _ { 1 } , \ldots , F _ { K }$ be K feature extractors. For each model $k ,$ let the target style feature be $s _ { k }$ . We consider the estimato

$$
\hat { x } _ { K } = \arg \operatorname* { m i n } _ { x } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| F _ { k } ( x ) - s _ { k } \| ^ { 2 } + \lambda \| x - x ^ { \star } \| ^ { 2 } \right\} ,\tag{13}
$$

where $\lambda > 0$ is a regularization coefficient.

Let

$$
J _ { k } : = \nabla F _ { k } ( x ^ { \star } ) , \qquad H _ { K } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } J _ { k } .\tag{14}
$$

We write the model-specific target error as

$$
s _ { k } = F _ { k } ( \boldsymbol { x } ^ { \star } ) + \varepsilon _ { k } , \qquad \mu _ { k } : = \mathbb { E } [ \varepsilon _ { k } ] , \qquad \tilde { \varepsilon } _ { k } : = \varepsilon _ { k } - \mu _ { k } .\tag{15}
$$

Assumption 1 (Diversity and local identifiability). The centered residuals have bounded second moments and limited average cross-model dependence:

$$
\| \mathbb { E } [ \tilde { \varepsilon } _ { k } \tilde { \varepsilon } _ { k } ^ { \top } ] \| _ { \mathrm { o p } } \leq \sigma ^ { 2 } , \qquad \| \mathbb { E } [ \tilde { \varepsilon } _ { k } \tilde { \varepsilon } _ { j } ^ { \top } ] \| _ { \mathrm { o p } } \leq \rho _ { K } \sigma ^ { 2 } \quad ( k \neq j ) ,\tag{16}
$$

where $\rho _ { K } \in [ 0 , 1 ]$ is an ensemble dependence coefficient. In addition, the ensemble is locally identifiable:

$$
H _ { K } \succeq m _ { K } I \qquad f o r s o m e m _ { K } > 0 .\tag{17}
$$

Assumption 1 is motivated by two standard ideas. First, heterogeneous vision encoders can provide complementary feature views rather than identical errors, making cross-model dependence, rather than exact equality, the key quantity to control. Second, local recovery in inverse problems is typically tied to a nondegenerate Jacobian or sensitivity matrix. These perspectives are consistent with recent evidence on complementary encoder biases and ensemble diversity, as well as standard identifiability analyses in inverse problems [14, 44, 25, 5].

Theorem 1 (Dependence-controlled scaling with ensemble size). Under Assumption 1 and the usual local regularity conditions deferred to Appendix B, the expected inversion error satisfies

$$
\mathbb { E } \| \hat { { \boldsymbol x } } _ { K } - { \boldsymbol x } ^ { \star } \| ^ { 2 } \lesssim \frac { d \bar { M } _ { K } \sigma ^ { 2 } } { ( m _ { K } + \lambda ) ^ { 2 } } \left( \rho _ { K } + \frac { 1 - \rho _ { K } } { K } \right) ,\tag{18}
$$

where

$$
\bar { M } _ { K } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| J _ { k } \| _ { \mathrm { o p } } ^ { 2 } .\tag{19}
$$

In particular, the gain from enlarging the ensemble is governed by the cross-model dependence coefficient $\rho _ { K } .$ . When the centered residuals are weakly correlated across models, that is, when $\rho _ { K }$ is small, the dominant term decreases approximately as $\bar { 1 } / K$

Empirical connection The bound in Eq. 18 predicts that the benefit of increasing K depends on the complementarity of the VFM residuals. This prediction is consistent with the VFM composition ablation in Fig. 5: individual VFMs emphasize different aspects of C-S disentanglement, while increasing the number of VFMs improves both SA and CL. This empirical trend supports the interpretation that Ensemble Inversion benefits from complementary VFM constraints.

## 4 Experiments

## 4.1 Settings

Evaluation Dataset We evaluate our method on StyleBench [8], a recent dataset containing 40 content images and 490 style images across 73 distinct styles. For each style, we select one representative image (I) and generate its content description (c) using Qwen3 [46]. For content, we use the textual prompts (y) corresponding to the 40 content images as generation conditions. This yields 2,920 style-content pairs for our experiments.

Implementation Details For each style-content pair, we generate 10 images over 25 random seeds. We use five VFMs in CLeaR: CLIP [23, 29], CSD-CLIP [34], DINO [31], VGG [32], and Inception [37]. In Ensemble Inversion, all VFMs are equally weighted with $\lambda _ { k } = 0 . 2$ , and the regularizer $\mathcal { R } ( \cdot )$ is total variation (TV) [28] with $\lambda _ { \mathcal { R } } = 0 . 0 5$ . We optimize for 300 iterations with learning rate $\eta = 0 . 0 1$ . For Energy-Guided Calibration, guidance is applied only in the refinement stage [48], i.e., $t : T / 1 0 \to 0$ , for 5 iterations with correction strength $\rho = 0 . 0 1$ . Further details are in Sec. 4.3.

Evaluation Metrics We evaluate seven metrics across four aspects:

• Style Alignment: We measure style alignment using CSD-CLIP cosine similarity and VGG Grammatrix style loss [9] between the generated images and the style references.

• Content Alignment: We measure content alignment using CLIP cosine similarity between the generated images and the target content prompts.

• Content Leakage: We measure content leakage using DINO similarity and KID [2] between the generated images and the content images. We also use Qwen3 to rate leaked reference-content elements on a 1–5 scale.

• Aesthetic Quality: We measure aesthetic quality using the CLIP-Aesthetic score [30].

Baselines We compare CLeaR with eight recent methods: Attention Distillation [53], CSGO [45], DEADiff [22], InstantStyle [40], MaskST [54], RB-Modulation [26], StyleAligned [12], and StyleShot [8]. Since style transfer methods vary in input modalities, and our framework uses a style image and a content text prompt, we restrict our comparison to baselines that adopt the same modality combination to ensure a fair evaluation.

## 4.2 Comparisons

![](images/146a265e9fbb89f4faf3e10577af6cd877bdc89a0dfb48c30c5646b13f1f216c.jpg)  
Figure 2: Visualization of style anchor images extracted by Ensemble Inversion. For each example, the left image is the reference and the right image is the style anchor. The anchor successfully achieves C-S disentanglement, retaining stylistic elements (e.g., color, texture, brushstroke) while suppressing content semantics.

Visualization of Style Anchors Fig. 2 visualizes the key intermediate output of our method, the style anchor image extracted via Ensemble Inversion. For each example, the left image is the original style reference I, and the right image is the corresponding style anchor $I _ { a } ^ { * }$ . The style anchor largely removes the content objects of the reference while faithfully preserving style-related representations. For instance, in the first example (row 1, col 1), the anchor removes the large tree, hills, and church steeple, retaining only the starry brushstrokes. In the seventh example (row 2, col 3), the clown figure is removed, leaving only the typographic style. These indicate that our method effectively disentangles content elements from stylistic attributes. More examples are provided in Appendix D.

![](images/5c1fc4724837ac30fae32626b292c53039fec77f4382c05ce93ca8906e150b10.jpg)  
Figure 3: Qualitative comparison results. Existing methods suffer from either content leakage or style degradation, while our method achieves robust stylization without either problem.

Qualitative Results Fig. 3 shows the qualitative comparisons, with magenta boxes marking content leakage and cyan boxes marking style degradation. Existing methods often fail on one side of the leakage-degradation dilemma: data-driven methods such as StyleShot, DEADiff, and CSGO show limited style alignment on out-of-distribution references; feature-manipulation methods such as InstantStyle and MaskST either retain residual content or suppress style details; and trajectory-level methods such as RB-Modulation and StyleAligned can depend on unavailable style-name priors. Attention Distillation preserves texture but often introduces reference content. In contrast, CLeaR achieves stronger style transfer with less visible content leakage. More results are in Appendix C.

Quantitative Results Tab. 1 reports the quantitative comparison. CLeaR achieves the best performance on most metrics, with clear gains in the two key aspects of C-S disentanglement: style alignment and content leakage suppression. Compared with the strongest baseline, CLeaR improves CSD by 8.3%, Style Loss by 64.9%, DINO by 60.7%, KID by 51.1%, and AI-Scoring by 8.05%, while maintaining competitive content alignment with the second-best CLIP score. Its slightly lower aesthetic score may result from the inversion process, which prioritizes disentanglement and structural preservation over aesthetic smoothing. We further validate these conclusions in Appendix E using held-out VFMs and AI evaluators, which show consistent trends and confirm that the reported gains are not biased by the closed-loop evaluation setup.

## 4.3 Ablation Studies

Our ablations examine three key factors in CLeaR: the content description of the style reference in OSP, the VFM composition in EI, and the guidance stage in EGC. For clarity, we report two

Table 1: Quantitative comparison of CLeaR and other baselines, with the best and second-best scores marked accordingly. CLeaR outperforms others across most aspects, especially in style alignment, content leakage, and content alignment.
<table><tr><td rowspan="2"></td><td rowspan="2">Style</td><td rowspan="2">Alignment</td><td rowspan="2">Content Alignment</td><td colspan="3">Content Leakage</td><td rowspan="2">Aesthetic Quality</td></tr><tr><td>CLIP↑</td><td>KID↑</td><td>AI Scoring↑</td></tr><tr><td>Attention Distillation</td><td>CSD↑ 0.630</td><td>Style Loss↓ 0.106</td><td>0.179</td><td>DINO↓ 0.322</td><td>0.112</td><td>1.84</td><td>CLIP-Aesthetic↑ 5.861</td></tr><tr><td>CSGO</td><td>0.438</td><td>0.302</td><td>0.215</td><td>0.168</td><td>0.237</td><td>3.65</td><td>5.912</td></tr><tr><td>DEADiff</td><td>0.369</td><td>0.259</td><td>0.225</td><td>0.209</td><td>0.233</td><td>3.64</td><td>6.349</td></tr><tr><td>InstantStyle</td><td>0.492</td><td>0.119</td><td>0.221</td><td>0.263</td><td>0.092</td><td>2.75</td><td>6.507</td></tr><tr><td>MaskST</td><td>0.416</td><td>0.139</td><td>0.225</td><td>0.197</td><td>0.103</td><td>3.45</td><td>6.605</td></tr><tr><td>RB-Modulation</td><td>0.443</td><td>0.177</td><td>0.234</td><td>0.213</td><td>0.111</td><td>3.82</td><td>6.677</td></tr><tr><td>StyleAligned</td><td>0.206</td><td>0.252</td><td>0.241</td><td>0.177</td><td>0.120</td><td>3.85</td><td>6.573</td></tr><tr><td>StyleShot</td><td>0.505</td><td>0.097</td><td>0.202</td><td>0.172</td><td>0.174</td><td>3.72</td><td>6.164</td></tr><tr><td>Ours</td><td>0.682</td><td>0.034</td><td>0.238</td><td>0.066</td><td>0.358</td><td>4.16</td><td>6.445</td></tr></table>

aggregate metrics: style alignment (SA↑) and content leakage suppression (CL↑). We first normalize each metric to [0, 1] across all compared settings. SA is the average of normalized CSD and 1 − normalized Style Loss, since higher CSD and lower Style Loss indicate better style alignment. CL is the average of 1 − normalized DINO, normalized KID, and normalized AI score, since lower DINO similarity, higher KID, and higher AI score indicate less leakage.

![](images/fc0d9271fe4efe71236225f25353b672dba0f3c08d7c31aa4ef7f14eca350baf.jpg)  
Figure 4: Style anchor extraction with varying content descriptions c. For a given reference, different c yield different content images $I _ { c }$ and style anchors $I _ { a } ^ { * } .$ . Removing an object from c preserves that object in $I _ { a } ^ { * } ,$ showing flexible disentanglement control.

Content Description of Style Reference The orthogonal projection in Eq. 3 requires a content description c of the style reference image I to generate a content image $I _ { c } = \Phi ( c )$ . This description determines which content elements are treated as “content” to be subtracted from the style feature. To examine the flexibility of our C-S disentanglement, we manually vary c by removing specific objects (e.g., “tree”, “church steeple”) and observe the resulting style anchor $I _ { a } ^ { * }$ extracted via Ensemble Inversion.

As shown in Fig. 4, when c contains all content elements (row 1), $I _ { c }$ includes the full object set, and the extracted anchor retains only the starry style brushstrokes. Removing “tree” from c (row 2) causes the tree to reappear in $I _ { a } ^ { * } ,$ while the church steeple is suppressed; removing “church steeple” (row 3) yields the opposite pattern. When no content is explicitly described (row 4), the inversion lacks guidance and retains many content artifacts. These results demonstrate that our framework allows fine-grained control over which content elements are disentangled, simply by editing the natural-language description c, without retraining any component.

VFM Selection To assess the effect of VFM composition, we evaluate Ensemble Inversion with $k = 1 , 3 , 5$ models selected from the five VFMs used in our main experiments.

As shown in Fig. 5, CLIP, DINO, and Inception mainly help suppress content leakage, while VGG and CSD-CLIP better preserve fine-grained style patterns. Increasing the number of VFMs consistently improves both SA and CL, showing that complementary VFM features lead to stronger C-S disentanglement.

Calibration Stage We apply Energy-Guided Calibration only in the final T/10 timesteps by default for efficiency. To study the effect of guidance timing, we divide sampling into three intervals: the chaotic stage $( t : T \to { \mathsf { 4 } } T / 5 )$ , where the image is mostly noise; the semantic stage $( t : 4 T / 5 \to T / 2 )$

![](images/47100ba5960fb52db21c28d6dea0f59ff008852130738c92355fff3dc0725258.jpg)  
Figure 5: Ablation of VFM compositions. CLIP, DINO, and Inception excel at suppressing content leakage, while VGG and CSD-CLIP better preserve fine-grained style patterns. Increasing the number of models improves both metrics, with the full ensemble of five achieving the best performance.

![](images/23e1d5cac411f6b336d54d851601e168caa1afa9793ac22565a5c85c02fd5677.jpg)

![](images/a3eb51c3075890a9074f23b604328087fbecd722cfaa660313c2fc5755ad60fb.jpg)

![](images/36a2aba7badb9be9aa80ad5775ecfbae137a0c631feb4ef0fd5c48c73ff3e95a.jpg)  
Figure 6: Effect of guidance stage in Energy-Guided Calibration. We compare applying guidance in the chaotic, semantic, and refinement stages, with the x-axis indicating the guided proportion of each stage. Guidance is ineffective in the chaotic stage, while in the semantic and refinement stages it improves SA with little change in CL. The best efficiency-performance trade-off is achieved by guiding the final 20% of the refinement stage, equivalent to the final T /10 steps.

where the global structure emerges; and the refinement stage $( t : T / 2 \to 0 )$ , where local textures and details are formed. For each interval, we vary the guided timestep ratio from 10% to 100%.

As shown in Fig. 6, guidance is ineffective in the chaotic stage. In the semantic and refinement stages, SA improves as more guidance steps are used, while CL remains relatively stable, indicating that calibration mainly enhances style alignment without increasing content leakage. The best efficiency performance trade-off is achieved by guiding 20% of the refinement stage, equivalent to the final T /10 timesteps, which we use as the default setting.

## 5 Conclusion

We presented CLeaR, a training-free framework for content-leakage-resistant style transfer. We identified that the leakage-degradation dilemma arises across the entire style transfer process: from separating C-S features, to grounding the style signal in a representation space, to injecting it during diffusion generation. CLeaR addresses these three sources with Orthogonal Subspace Projection for content-reduced style targets, Ensemble Inversion for multi-VFM style anchor extraction, and Energy-Guided Calibration for style-preserving diffusion sampling. Our theoretical analysis supports the benefit of using multiple VFMs, and experiments on StyleBench show improved style alignment and content leakage suppression. These results demonstrate the effectiveness of addressing C-S disentanglement jointly across feature separation, style representation, and generation.

Limitations Despite its effectiveness, CLeaR has several limitations. First, it introduces additional test-time computation via Ensemble Inversion and Energy-Guided Calibration. While the measured runtime is comparable to several training-free baselines, it is still slower than simpler feed-forward methods. Second, as analyzed in Appendix G, ambiguous content descriptions, degenerate anchors, and adapter misinterpretation of inversion noise can still lead to leakage and artifacts. These cases represent an important direction for future work.

## References

[1] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 843–852, 2023.

[2] Mikołaj Binkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In´ International Conference on Learning Representations, 2018.

[3] Yunhao Chen, Shujie Wang, Xin Wang, Ran He, Xingjun Ma, and Yu-Gang Jiang. Leakyclip: Extracting training data from clip. arXiv preprint arXiv:2508.00756, 2025.

[4] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8795–8805, 2024.

[5] Ariel Cintrón-Arias, HT Banks, Alex Capaldi, and Alun L Lloyd. A sensitivity matrix based methodology for inverse problem formulation. arXiv preprint arXiv:2004.06831, 2020.

[6] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, pages 1322–1333, 2015.

[7] Matthew Fredrikson, Eric Lantz, Somesh Jha, Simon Lin, David Page, and Thomas Ristenpart. Privacy in pharmacogenetics: An {End-to-End} case study of personalized warfarin dosing. In 23rd USENIX security symposium (USENIX Security 14), pages 17–32, 2014.

[8] Junyao Gao, Yanan Sun, Yanchen Liu, Yinhao Tang, Yanhong Zeng, Ding Qi, Kai Chen, and Cairong Zhao. Styleshot: A snapshot on any style. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[9] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015.

[10] Brian Gordon, Yonatan Bitton, Yonatan Shafir, Roopal Garg, Xi Chen, Dani Lischinski, Daniel Cohen-Or, and Idan Szpektor. Mismatch quest: Visual and textual feedback for image-text misalignment. In European Conference on Computer Vision, pages 310–328. Springer, 2024.

[11] Feihong He, Gang Li, Fuhui Sun, Mengyuan Zhang, Lingyu Si, Xiaoyan Wang, and Li Shen. Freestyle: Free lunch for text-guided style transfer using diffusion models. Pattern Recognition, page 113093, 2026.

[12] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4775–4785, 2024.

[13] Mostafa Kahla, Si Chen, Hoang Anh Just, and Ruoxi Jia. Label-only model inversion attacks via boundary repulsion. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15025–15033. IEEE, 2022.

[14] Oguzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, and Federico Tombari. Brave: Broadening the visual encoding of vision-language models. In European Conference on Computer Vision, pages 113–132. Springer, 2024.

[15] Bumsoo Kim, Yeonsik Jo, Jinhyung Kim, and Seunghwan Kim. Misalign, contrast then distill: Rethinking misalignments in language-image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2563–2572, 2023.

[16] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Yibo Wang, Xintao Wang, Yujiu Yang, and Ying Shan. Stylecrafter: Enhancing stylized text-to-video generation with style adapter. arXiv preprint arXiv:2312.00330, 2023.

[17] Pingchuan Ma, Xiaopei Yang, Yusong Li, Ming Gui, Felix Krause, Johannes Schusterbauer, and Björn Ommer. Scflow: Implicitly learning style and content disentanglement with flow models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 14919–14929, 2025.

[18] Lilian Ngweta, Subha Maity, Alex Gittens, Yuekai Sun, and Mikhail Yurochkin. Simple disentanglement of style and content in visual representations. In International Conference on Machine Learning, pages 26063–26086. PMLR, 2023.

[19] Zach Nussbaum, Brandon Duderstadt, and Andriy Mulyar. Nomic embed vision: Expanding the latent space. arXiv preprint arXiv:2406.18587, 2024.

[20] OpenAI. Gpt-5.5 system card. https://openai.com/index/gpt-5-5-system-card/, 2026.

[21] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

[22] Tianhao Qi, Shancheng Fang, Yanze Wu, Hongtao Xie, Jiawei Liu, Lang Chen, Qian He, and Yongdong Zhang. Deadiff: An efficient stylization diffusion model with disentangled representations. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8693–8702, 2024.

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[25] Simon Rouchier. Solving inverse problems in building physics: An overview of guidelines for a careful and optimal use of data. Energy and Buildings, 166:178–195, 2018.

[26] Litu Rout, Yujia Chen, Nataniel Ruiz, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Rb-modulation: Training-free stylization using reference-based modulation. In International Conference on Learning Representations, volume 2025, pages 56870–56905, 2025.

[27] Aniket Roy, Shubhankar Borse, Shreya Kadambi, Debasmit Das, Shweta Mahajan, Risheek Garrepalli, Hyojin Park, Ankita Nayak, Rama Chellappa, Munawar Hayat, et al. Duolora: Cycle-consistent and rank-disentangled content-style personalization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 15395–15404, 2025.

[28] Leonid I Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4):259–268, 1992.

[29] Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In International Conference on Machine Learning, pages 43685–43704. PMLR, 2024.

[30] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

[31] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[32] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

[33] Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, et al. Styledrop: Text-to-image synthesis of any style. Advances in Neural Information Processing Systems, 36:66860–66889, 2023.

[34] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

[35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021.

[36] Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, and Kristian Kersting. Plug & play attacks: Towards robust and flexible model inversion attacks. In International Conference on Machine Learning, pages 20522–20545, 2022.

[37] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

[38] Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

[39] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[40] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024.

[41] Ye Wang, Zili Yi, Yibo Zhang, Peng Zheng, Xuping Xie, Jiang Lin, Yilin Wang, and Rui Ma. Omnistyle2: Scalable and high quality artistic style transfer data generation via destylization. arXiv preprint arXiv:2509.05970, 2025.

[42] Zhizhong Wang, Lei Zhao, and Wei Xing. Stylediffusion: Controllable disentangled style transfer via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7677–7689, 2023.

[43] Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping Luo. Styleadapter: A unified stylized image generation model. International Journal of Computer Vision, 133(4):1894–1911, 2025.

[44] Danny Wood, Tingting Mu, Andrew M Webb, Henry WJ Reeve, Mikel Lujan, and Gavin Brown. A unified theory of diversity in ensemble learning. Journal ofmachine learning research, 24(359):1–49, 2023.

[45] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation. In Advances in Neural Information Processing Systems, volume 38, pages 100506–100546, 2025.

[46] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[47] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

[48] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23174–23184, 2023.

[49] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547, 2023.

[50] Yuheng Zhang, Ruoxi Jia, Hengzhi Pei, Wenxiao Wang, Bo Li, and Dawn Song. The secret revealer: Generative model-inversion attacks against deep neural networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 250–258. IEEE, 2020.

[51] Yuxin Zhang, Nisha Huang, Fan Tang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. Inversion-based style transfer with diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10146–10156, 2023.

[52] Zhanjie Zhang, Quanwei Zhang, Wei Xing, Guangyuan Li, Lei Zhao, Jiakai Sun, Zehua Lan, Junsheng Luan, Yiling Huang, and Huaizhong Lin. Artbank: Artistic style transfer with pre-trained diffusion model and implicit style prompt bank. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pages 7396–7404, 2024.

[53] Yang Zhou, Xu Gao, Zichong Chen, and Hui Huang. Attention distillation: A unified approach to visual characteristics transfer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 18270–18280, 2025.

[54] Lin Zhu, Xinbing Wang, Chenghu Zhou, Qinying Gu, and Nanyang Ye. Less is more: Masking elements in image condition features avoids content leakages in style transfer diffusion models. In International Conference on Learning Representations, volume 2025, pages 66094–66126, 2025.

## A Overall Pipeline of CLeaR

We provide detailed pseudocode in Alg. 1 to facilitate a better understanding of our framework. The algorithm consists of three main stages: (1) computing orthogonal style anchors for multiple VFMs, (2) inverting these anchors into a pixel-space style image via ensemble optimization, and (3) generating the final output with energy-guided calibration.

In Stage 1, we generate a content image $I _ { c }$ from the description c and compute orthogonal style features $\mathbf { f } _ { s } ^ { ( k ) }$ for each VFM. Stage 2 optimizes a pixel-space image $I _ { a }$ to match these features simultaneously, producing a style anchor $I _ { a } ^ { * }$ . Stage 3 uses the IP-Adapter $\Phi _ { \mathrm { I P } }$ to condition the diffusion model on $I _ { a } ^ { * }$ and the target content y (or $I _ { y } ) _ { \mathrm { : } }$ , while additionally applying an energy correction based on the same VFMs. The energy guidance is only applied during the semantic stage (e.g., intermediate timesteps) to balance efficiency and style fidelity. The final output $\boldsymbol { I _ { \mathrm { o u t } } }$ is the decoded latent $\mathbf { z } _ { 0 }$ . Hyperparameters such as $\lambda _ { k } , \lambda _ { \mathcal { R } } , \eta , \rho$ are set empirically.

Algorithm 1 Overall Pipeline of CLeaR   
Require: Style image $I ,$ content description $c ,$ target content prompt y (or target image $I _ { y } ) ,$ pre  
trained VFMs $\{ \check { F } _ { k } \} _ { k = 1 } ^ { K }$ , text-to-image model Φ, IP-Adapter $\Phi _ { \mathrm { I P } } .$ , VAE decoder $D ,$ DDIM sampler   
with $\epsilon _ { \theta }$   
Ensure: Output image $\boldsymbol { I _ { \mathrm { o u t } } }$   
1: — Stage 1: orthogonal style anchors —   
2: $I _ { c } \gets \Phi ( c )$ {generate content image}   
3: for each $\dot { k } = \bar { 1 }$ to $K _ { \star }$ do   
4: $\mathbf { f } ^ { ( k ) } \gets F _ { k } ( I ) , \mathbf { f } _ { c } ^ { ( k ) } \gets F _ { k } ( I _ { c } )$   
5: $\mathbf { f } _ { s } ^ { ( k ) } \gets \mathbf { f } ^ { ( k ) } - \mathbf { f } _ { c } ^ { ( k ) }$   
6: $\begin{array} { r } { \mathbf { f } _ { s } ^ { ( k ) }  \mathbf { f } _ { s } ^ { ( k ) } - \frac { \langle \mathbf { \tilde { f } } _ { s } ^ { ( k ) } , \mathbf { f } _ { c } ^ { ( k ) } \rangle } { | | \mathbf { f } ^ { ( k ) } | | 2 } \mathbf { f } _ { c } ^ { ( k ) } } \end{array}$ {orthogonal projection}   
∥f<sub>c</sub> ∥   
7: end for   
8: — Stage $\mathbf { \hat { \xi } } ^ { 2 } \mathbf { : }$ ensemble inversion to obtain style anchor image —   
9: Initialize $I _ { a } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ {random tensor}   
10: for $t = 1$ to $T _ { \mathrm { i n v } }$ do   
11: for each $k = 1$ to K do   
12: $\mathbf { f } _ { a } ^ { ( k ) } \gets F _ { k } ( I _ { a } )$   
13: end for   
14: $\begin{array} { r } { \mathcal { L } _ { \mathrm { i n v } }  \sum _ { k = 1 } ^ { K } \lambda _ { k } \| \mathbf { f } _ { a } ^ { ( k ) } - \mathbf { f } _ { s } ^ { ( k ) } \| ^ { 2 } + \lambda _ { \mathcal { R } } \mathcal { R } ( I _ { a } ) } \end{array}$   
15: $I _ { a } \gets I _ { a } - \eta \bar { \nabla } _ { I _ { a } } \mathcal { L } _ { \mathrm { i n v } }$   
16: end for   
17: $I _ { a } ^ { * } \gets I _ { a }$   
18: — Stage 3: energy-guided calibration during generation —   
19: Initialize $\mathbf { z } _ { T } \sim \breve { \mathcal { N } } ( 0 , \mathbf { \breve { I } } )$   
20: for $t = T$ down to 1 do   
21: if $t \in \mathcal { T } _ { \mathrm { c a l } }$ then   
22: $\epsilon _ { t } \gets \epsilon _ { \theta } ( \mathbf { z } _ { t } , t , I _ { a } ^ { * } , y )$   
23: $\hat { \mathbf { z } } _ { 0 } \gets ( \mathbf { z } _ { t } - \sigma _ { t } \mathbf { \epsilon } _ { t } ) / \alpha _ { t }$   
24: $\hat { \mathbf { x } } _ { 0 } \gets D ( \hat { \mathbf { z } } _ { 0 } )$   
25: $\begin{array} { r } { \mathcal { E } _ { t }  \sum _ { k = 1 } ^ { { K } } \lambda _ { k } \| F _ { k } ( \hat { \mathbf { x } } _ { 0 } ) - \mathbf { f } _ { s } ^ { ( k ) } \| ^ { 2 } } \end{array}$   
26: $\widetilde { \mathbf { z } } _ { t } \gets \mathbf { z } _ { t } - \rho \nabla _ { \mathbf { z } _ { t } } \overset { : } { \mathcal { E } } _ { t }$   
27: $\mathbf { z } _ { t - 1 }  \mathrm { D D I M } \big ( \tilde { \mathbf { z } } _ { t } , \epsilon _ { \theta } \big ( \tilde { \mathbf { z } } _ { t } , t , I _ { a } ^ { * } , y \big ) , t \big )$   
28: else   
29: $\mathbf { z } _ { t - 1 }  \mathrm { D D I M } \big ( \mathbf { z } _ { t } , \boldsymbol { \epsilon } _ { \theta } \big ( \mathbf { z } _ { t } , t , I _ { a } ^ { * } , y \big ) , t \big )$   
30: end if   
31: end for   
32: $I _ { \mathrm { o u t } }  D ( \mathbf { z } _ { 0 } )$   
33: return $I _ { \mathrm { o u t } }$

## B Proof of the Multi-Model Inversion Error Bound

We provide a local first-order analysis of ensemble inversion.

Setup Let

$$
\hat { x } _ { K } = \arg \operatorname* { m i n } _ { x } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| F _ { k } ( x ) - s _ { k } \| ^ { 2 } + \lambda \| x - x ^ { \star } \| ^ { 2 } \right\} ,\tag{20}
$$

where $x ^ { \star } \in \mathbb { R } ^ { d }$ is the ideal content-free style anchor. Write

$$
\hat { \Delta } _ { K } : = \hat { x } _ { K } - x ^ { \star } .\tag{21}
$$

Assumption 2 (Local first-order regime). Each $F _ { k }$ is differentiable in a neighborhood of $x ^ { \star }$ , and for sufficiently small $\Delta _ { \cdot }$

$$
F _ { k } ( \boldsymbol { x } ^ { \star } + \boldsymbol { \Delta } ) = F _ { k } ( \boldsymbol { x } ^ { \star } ) + J _ { k } \boldsymbol { \Delta } + r _ { k } ( \boldsymbol { \Delta } ) , \qquad J _ { k } : = \nabla F _ { k } ( \boldsymbol { x } ^ { \star } ) ,\tag{22}
$$

where the remainder $r _ { k } ( \Delta )$ is lower-order than the linear term as $\| \Delta \|  0 .$

Assumption 3 (Ensemble-average bias cancellation). The ensemble-averaged image-space bias vanishes:

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } \mu _ { k } = 0 , \qquad \mu _ { k } : = \mathbb { E } [ \varepsilon _ { k } ] .\tag{23}
$$

Assumption 4 (Bounded Jacobians). The local Jacobians are uniformly bounded:

$$
\| J _ { k } \| _ { \mathrm { o p } } ^ { 2 } \leq M _ { k } .\tag{24}
$$

Proposition 1. Under Assumption 1 and Appendix Assumptions 2-4,

$$
\mathbb { E } \| \hat { { \boldsymbol x } } _ { K } - { \boldsymbol x } ^ { \star } \| ^ { 2 } \lesssim \frac { d \bar { M } _ { K } \sigma ^ { 2 } } { ( m _ { K } + \lambda ) ^ { 2 } } \left( \rho _ { K } + \frac { 1 - \rho _ { K } } { K } \right) ,\tag{25}
$$

where

$$
\bar { M } _ { K } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } M _ { k } .\tag{26}
$$

Proof. Under Assumption 2, the leading behavior is governed by the linearized objective

$$
\tilde { \Delta } _ { K } = \arg \operatorname* { m i n } _ { \Delta } \left\{ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| J _ { k } \Delta - \varepsilon _ { k } \| ^ { 2 } + \lambda \| \Delta \| ^ { 2 } \right\} .\tag{27}
$$

Expanding the quadratic objective gives

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| J _ { k } \Delta - \varepsilon _ { k } \| ^ { 2 } + \lambda \| \Delta \| ^ { 2 } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( \Delta ^ { \top } J _ { k } ^ { \top } J _ { k } \Delta - 2 \varepsilon _ { k } ^ { \top } J _ { k } \Delta + \varepsilon _ { k } ^ { \top } \varepsilon _ { k } \right) + \lambda \Delta ^ { \top } \Delta
$$

$$
= \Delta ^ { \top } H _ { K } \Delta - 2 b _ { K } ^ { \top } \Delta + \lambda \Delta ^ { \top } \Delta + \mathrm { c o n s t a n t } ,\tag{28}
$$

where

$$
b _ { K } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } \varepsilon _ { k } .\tag{29}
$$

Differentiating Eq. 28 with respect to $\Delta$ and setting the gradient to zero yields

$$
( H _ { K } + \lambda I ) \tilde { \Delta } _ { K } = b _ { K } .\tag{30}
$$

Hence

$$
\tilde { \Delta } _ { K } = ( H _ { K } + \lambda I ) ^ { - 1 } b _ { K } .\tag{31}
$$

By Assumption 1,

$$
H _ { K } + \lambda I \succeq ( m _ { K } + \lambda ) I ,\tag{32}
$$

$$
\| ( H _ { K } + \lambda I ) ^ { - 1 } \| _ { \mathrm { o p } } \leq \frac { 1 } { m _ { K } + \lambda } .\tag{33}
$$

Therefore,

$$
\mathbb { E } \| \tilde { \Delta } _ { K } \| ^ { 2 } \leq \frac { 1 } { ( m _ { K } + \lambda ) ^ { 2 } } \mathbb { E } \| b _ { K } \| ^ { 2 } .\tag{34}
$$

We now decompose $b _ { K }$ into bias and centered fluctuation terms:

$$
b _ { K } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } \mu _ { k } + \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } \tilde { \varepsilon } _ { k } = : \beta _ { K } + u _ { K } .\tag{35}
$$

By Assumption 3,

$$
\beta _ { K } = 0 .\tag{36}
$$

Hence

$$
b _ { K } = u _ { K } \qquad \mathrm { a n d } \qquad \mathbb { E } \| b _ { K } \| ^ { 2 } = \mathbb { E } \| u _ { K } \| ^ { 2 } .\tag{37}
$$

Now

$$
u _ { K } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } J _ { k } ^ { \top } \tilde { \varepsilon } _ { k } ,\tag{38}
$$

so

$$
\mathbb { E } \Vert u _ { K } \Vert ^ { 2 } = \frac { 1 } { K ^ { 2 } } \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { K } \mathbb { E } \big [ \tilde { \varepsilon } _ { k } ^ { \top } J _ { k } J _ { j } ^ { \top } \tilde { \varepsilon } _ { j } \big ] .\tag{39}
$$

For the diagonal terms,

$$
\begin{array} { r l } & { \mathbb { E } \big [ \tilde { \varepsilon } _ { k } ^ { \top } J _ { k } J _ { k } ^ { \top } \tilde { \varepsilon } _ { k } \big ] = \mathrm { t r } \big ( J _ { k } J _ { k } ^ { \top } \mathbb { E } \big [ \tilde { \varepsilon } _ { k } \tilde { \varepsilon } _ { k } ^ { \top } \big ] \big ) } \\ & { \qquad \leq d \| J _ { k } \| _ { \mathrm { o p } } ^ { 2 } \sigma ^ { 2 } } \\ & { \qquad \leq d M _ { k } \sigma ^ { 2 } . } \end{array}\tag{40}
$$

For the off-diagonal terms $k \neq j .$ , Assumption 1 and Assumption 4 give

$$
\begin{array} { r l } & { \left| \mathbb { E } \big [ \tilde { \varepsilon } _ { k } ^ { \top } J _ { k } J _ { j } ^ { \top } \tilde { \varepsilon } _ { j } \big ] \right| = \left| \mathrm { t r } \big ( J _ { k } J _ { j } ^ { \top } \mathbb { E } \big [ \tilde { \varepsilon } _ { j } \tilde { \varepsilon } _ { k } ^ { \top } \big ] \big ) \right| } \\ & { \qquad \leq d \left\| J _ { k } \right\| _ { \mathrm { o p } } \left\| J _ { j } \right\| _ { \mathrm { o p } } \rho _ { K } \sigma ^ { 2 } } \\ & { \qquad \leq d \left( M _ { k } M _ { j } \right) ^ { 1 / 2 } \rho _ { K } \sigma ^ { 2 } } \\ & { \qquad \leq \frac { d \rho _ { K } \sigma ^ { 2 } } { 2 } ( M _ { k } + M _ { j } ) . } \end{array}\tag{41}
$$

Substituting Eq. 40 and Eq. 41 into Eq. 39, we obtain

$$
\begin{array} { l } { { \displaystyle \mathbb { E } \| u _ { K } \| ^ { 2 } \leq \frac { 1 } { K ^ { 2 } } \left[ \displaystyle \sum _ { k = 1 } ^ { K } d M _ { k } \sigma ^ { 2 } + \displaystyle \sum _ { k \neq j } \frac { d \rho _ { K } \sigma ^ { 2 } } { 2 } ( M _ { k } + M _ { j } ) \right] } } \\ { ~ = \displaystyle \frac { d \sigma ^ { 2 } } { K ^ { 2 } } \left[ \displaystyle \sum _ { k = 1 } ^ { K } M _ { k } + \rho _ { K } ( K - 1 ) \displaystyle \sum _ { k = 1 } ^ { K } M _ { k } \right] }  \\ { ~ = d \bar { M } _ { K } \sigma ^ { 2 } \left( \displaystyle \frac { 1 } { K } + \rho _ { K } \displaystyle \frac { K - 1 } { K } \right) } \\ { ~ = d \bar { M } _ { K } \sigma ^ { 2 } \left( \rho _ { K } + \displaystyle \frac { 1 - \rho _ { K } } { K } \right) . } \end{array}\tag{42}
$$

Combining Eq. 34, Eq. 37, and Eq. 42 yields

$$
\mathbb { E } \| \tilde { \Delta } _ { K } \| ^ { 2 } \leq \frac { d \bar { M } _ { K } \sigma ^ { 2 } } { ( m _ { K } + \lambda ) ^ { 2 } } \left( \rho _ { K } + \frac { 1 - \rho _ { K } } { K } \right) .\tag{43}
$$

Under Assumption 2, the neglected remainder is lower-order in the local regime, so the same leading-order bound applies to the exact estimator x $\dot { \boldsymbol { \cdot } } \boldsymbol { K } - \boldsymbol { x } ^ { \star }$ □

![](images/4fa68e7da57f645b26612dbd0a2166a6a949a058353846ea53216b8327b096d8.jpg)  
Figure A1: Extended qualitative comparisons, with boxes highlighting content leakage and style degradation. Our method maintains balanced stylization across all examples.

## C More Qualitative Comparison Results

Fig. A1 provides additional qualitative comparisons. Consistent with the observations in Sec. 4.2, existing methods still suffer from either content leakage (magenta boxes) or style degradation (cyan boxes) across diverse examples, while our CLeaR achieves clean stylization without compromising either aspect.

## D More Examples of Style Anchors

Fig. A2 presents additional style anchors extracted by our Ensemble Inversion. Across various artistic styles, the anchors consistently suppress content semantics (e.g., objects, faces, text) while preserving style attributes such as color, texture, and brushstrokes.

![](images/0f05483df43f403ba8b21b1945c2a4da424fb4771cb43c9136807becec0988e4.jpg)  
Figure A2: Additional style anchor examples. Each pair shows the original reference (left) and the extracted anchor (right).

## E Comparisons on Held-Out VFMs and AI Evaluator

Table A1: Performance of held-out VFMs and independent AI evaluators, with the best and second-best scores marked accordingly.
<table><tr><td rowspan="2"></td><td colspan="2">Style Alignment</td><td colspan="2">Content Alignment</td><td colspan="4">Content Leakage</td><td colspan="2">Aesthetic Quality</td></tr><tr><td>SigLIP↑</td><td>Nomic↑</td><td>SigLIP↑</td><td>Nomic↑</td><td>SigLIP↓</td><td>Nomic↓</td><td>Kimi-K2.6↑</td><td>GPT-5.5↑</td><td>Kimi-K2.6↑</td><td>GPT-5.5↑</td></tr><tr><td>Attention Distillation</td><td>0.669</td><td>0.842</td><td>0.0867</td><td>0.0573</td><td>0.588</td><td>0.797</td><td>2.03</td><td>1.68</td><td>3.20</td><td>3.10</td></tr><tr><td>CSGO</td><td>0.592</td><td>0.800</td><td>0.117</td><td>0.0777</td><td>0.539</td><td>0.774</td><td>3.68</td><td>3.27</td><td>3.43</td><td>3.36</td></tr><tr><td>DEADiff</td><td>0.563</td><td>0.788</td><td>0.126</td><td>0.0812</td><td>0.552</td><td>0.761</td><td>3.71</td><td>3.24</td><td>3.62</td><td>3.58</td></tr><tr><td>InstantStyle</td><td>0.609</td><td>0.800</td><td>0.123</td><td>0.0826</td><td>0.529</td><td>0.760</td><td>2.91</td><td>2.47</td><td>3.89</td><td>3.86</td></tr><tr><td>MaskST</td><td>0.588</td><td>0.790</td><td>0.128</td><td>0.0833</td><td>0.521</td><td>0.753</td><td>3.56</td><td>3.08</td><td>4.03</td><td>4.01</td></tr><tr><td>RB-Modulation</td><td>0.584</td><td>0.789</td><td>0.136</td><td>0.0887</td><td>0.557</td><td>0.761</td><td>3.92</td><td>3.53</td><td>4.48</td><td>4.39</td></tr><tr><td>StyleAligned</td><td>0.544</td><td>0.772</td><td>0.137</td><td>0.0893</td><td>0.531</td><td>0.751</td><td>4.03</td><td>3.58</td><td>4.39</td><td>4.18</td></tr><tr><td>StyleShot</td><td>0.612</td><td>0.814</td><td>0.0997</td><td>0.0698</td><td>0.534</td><td>0.771</td><td>3.79</td><td>3.41</td><td>3.78</td><td>3.74</td></tr><tr><td>Ours</td><td>0.754</td><td>0.876</td><td>0.137</td><td>0.0803</td><td>0.527</td><td>0.748</td><td>4.28</td><td>3.92</td><td>4.20</td><td>4.52</td></tr></table>

To address the concern that CLeaR optimizes and evaluates on overlapping VFM features (CSD-CLIP, VGG, DINO) and that Qwen3 is used both for content description generation and AI-based evaluation, we conduct additional experiments with held-out VFMs and independent AI evaluators. We add SigLIP [39] and Nomic [19] as additional VFMs. Since our original selection (e.g., CLIP, DINO, CSD-CLIP) already covers the most standard models for style transfer and vision-language representation, few suitable alternatives remain. For AI evaluation, we include Kimi-K2.6 [38] and GPT-5.5 [20]. To ensure comprehensive coverage, we evaluate these new metrics across all four aspects.

As shown in Tab. A1, CLeaR consistently achieves the best or competitive performance across all heldout VFMs and independent AI evaluators. These results confirm that our improvements generalize beyond the models used during optimization and are not biased by the closed-loop evaluation setup.

## F Ablation on Style Categories

To investigate performance variation across different styles, we compute SA and CL scores for all 73 style types in StyleBench.

As seen in Tab. A2, styles whose stylistic identity is carried by rendering rather than subject matter (e.g., Impressionism, Watercolor, Line Art, and Primitivism) achieve high scores on both SA and CL, indicating easier C-S disentanglement when stylistic signals are distributed across the entire image. In contrast, styles associated with religious, mythological, or historical subjects (e.g., Classicism, Rococo, and Baroque) exhibit lower CL scores even when SA remains high. Such themes often involve complex figures, narratives, or symbols that are hard to enumerate in c, making C-S disentanglement more challenging during Ensemble Inversion.

Table A2: Breakdown across the 73 categories on StyleBench. Rich-texture styles yield high SA and CL, while subject-centric styles show lower CL despite strong SA.
<table><tr><td>Style</td><td>SA↑</td><td>CL↑</td><td>Style</td><td>SA↑</td><td>CL↑</td><td>Style</td><td>SA↑</td><td>CL↑</td></tr><tr><td>3D Model</td><td>0.792</td><td>0.276</td><td>Expressionist</td><td>0.855</td><td>0.476</td><td>Origami</td><td>0.815</td><td>0.528</td></tr><tr><td>3D Model 01</td><td>0.729</td><td>0.457</td><td>Fantasy Art</td><td>0.975</td><td>0.453</td><td>Orphism</td><td>0.853</td><td>0.604</td></tr><tr><td>3D Model 02</td><td>0.764</td><td>0.511</td><td>Fauvism</td><td>0.926</td><td>0.339</td><td>Others</td><td>0.799</td><td>0.380</td></tr><tr><td>3D Model 03</td><td>0.703</td><td>0.431</td><td>Flat Vector</td><td>0.585</td><td>0.338</td><td>Photographic</td><td>0.741</td><td>0.311</td></tr><tr><td>3D Model 04</td><td>0.802</td><td>0.462</td><td>Folk Art</td><td>0.498</td><td>0.320</td><td>Pixel Art</td><td>0.674</td><td>0.420</td></tr><tr><td>3D Model 05</td><td>0.726</td><td>0.433</td><td>Gongbi</td><td>0.908</td><td>0.473</td><td>Pointilism</td><td>0.855</td><td>0.139</td></tr><tr><td>Abstract</td><td>0.741</td><td>0.496</td><td>Graffiti</td><td>0.640</td><td>0.279</td><td>Pop Art</td><td>0.671</td><td>0.479</td></tr><tr><td>Abstract 01</td><td>0.845</td><td>0.498</td><td>Hyperrealism</td><td>0.808</td><td>0.299</td><td>Post-Impressionism</td><td>0.972</td><td>0.466</td></tr><tr><td>Analog Film</td><td>0.635</td><td>0.382</td><td>Icon</td><td>0.476</td><td>0.605</td><td>Precisionism</td><td>0.893</td><td>0.491</td></tr><tr><td>Anime</td><td>0.837</td><td>0.394</td><td>Icon 01</td><td>0.517</td><td>0.551</td><td>Primitivism</td><td>0.883</td><td>0.661</td></tr><tr><td>Anime 01</td><td>0.543</td><td>0.459</td><td>Icon 02</td><td>0.551</td><td>0.432</td><td>Psychedelic</td><td>0.434</td><td>0.402</td></tr><tr><td>Anime 02</td><td>0.730</td><td>0.422</td><td>Impressionism</td><td>0.967</td><td>0.921</td><td>Realism</td><td>0.924</td><td>0.547</td></tr><tr><td>Anime 03</td><td>0.866</td><td>0.546</td><td>Ink and Wash Painting</td><td>0.864</td><td>0.246</td><td>Rococo</td><td>0.868</td><td>0.186</td></tr><tr><td>Anime 04</td><td>0.823</td><td>0.458</td><td>IsoMetric</td><td>0.708</td><td>0.442</td><td>Smoke&amp;Light</td><td>0.820</td><td>0.569</td></tr><tr><td>Anime 05</td><td>0.907</td><td>0.358</td><td>Japonism</td><td>0.743</td><td>0.217</td><td>Statue</td><td>0.685</td><td>0.497</td></tr><tr><td>Anime 06</td><td>0.878</td><td>0.631</td><td>Line Art</td><td>0.803</td><td>0.675</td><td>Steampunk</td><td>0.665</td><td>0.264</td></tr><tr><td>Anime 07</td><td>0.817</td><td>0.389</td><td>Low Poly</td><td>0.780</td><td>0.546</td><td>Stick Figure</td><td>0.697</td><td>0.498</td></tr><tr><td>Art Deco</td><td>0.605</td><td>0.278</td><td>Luminism</td><td>0.992</td><td>0.529</td><td>Stickers</td><td>0.580</td><td>0.388</td></tr><tr><td>Baroque</td><td>0.901</td><td>0.266</td><td>Macabre</td><td>0.937</td><td>0.584</td><td>Surrealist</td><td>0.890</td><td>0.507</td></tr><tr><td>Children&#x27;s Painting</td><td>0.755</td><td>0.554</td><td>MineCraft</td><td>0.864</td><td>0.459</td><td>Symbolism</td><td>0.892</td><td>0.441</td></tr><tr><td>Classicsm</td><td>0.882</td><td>0.0184</td><td>Monochrome</td><td>0.774</td><td>0.467</td><td>Tonalism</td><td>0.887</td><td>0.492</td></tr><tr><td>Constructivism</td><td>0.710</td><td>0.506</td><td>Neo-Figurative Art</td><td>0.922</td><td>0.541</td><td>Typography</td><td>0.200</td><td>0.440</td></tr><tr><td>Craft Clay</td><td>0.475</td><td>0.508</td><td>Neoclassicism</td><td>0.813</td><td>0.286</td><td>Watercolor</td><td>0.858</td><td>0.667</td></tr><tr><td>Cublism</td><td>0.930</td><td>0.369</td><td>Nouveau</td><td>0.920</td><td>0.395</td><td></td><td></td><td></td></tr><tr><td>Cyberpunk</td><td>0.923</td><td>0.499</td><td>Op Art</td><td>0.672</td><td>0.538</td><td></td><td></td><td></td></tr></table>

## G Analysis on Failure Modes

We analyze three representative failure modes of CLeaR. All stem from limitations in current pretrained modules rather than the disentanglement formulation itself.

Incomplete Content Description In theory, an ideal c allows OSP to remove all content-related components. However, as discussed in Appendix F, certain styles are difficult to describe clearly, leading to residual content leakage.

![](images/b1166275dd914d336ec0c3fded7e9055ab2bce7a5a3d595410f5a3b6cd5aa678.jpg)  
Figure A4: Artifacts in the generation stage. The adapter may misinterpret inversion noise in style anchor as semantic content, producing noisy lines and blobs.

Degenerate Style Anchor Fig. A3 shows cases where Ensemble Inversion produces a degenerate anchor that either contains almost no information (col 2) or replicates the style reference (cols 1, 3, 4). This failure mode is frequently observed in styles such as Line Art, Stick Figure, and Icon. Although orthogonal projection is designed to remove content from the style target, current VFM embeddings do not always provide a direction that separates style from content. When style is tightly coupled with a specific object, the inversion objective is satisfied either by an anchor without meaningful structure or by one that copies the reference.

![](images/8e686f251a4ee77f8920b7e26172878b00edf3c2a2ef5ec109113842d87b4690.jpg)  
Figure A3: Degenerate style anchors. Ensemble Inversion may produce an anchor with no structure or one that copies the reference, especially when content and style are tightly coupled.

Generation Artifacts Fig. A4 shows another failure mode that arises when the style anchor I<sup>∗</sup> is used as the image condition for diffusion generation, where the output contains noisy lines and blobs. As discussed in Sec. 3.3, the inverted anchor lies outside the adapter’s training distribution, so the adapter may misinterpret noise in the anchor as semantic content, forming artifacts in the final result.