# FlowTranslator: Training-Free In-Image Machine Translation with Flow-Based Editing

Zhixian Li<sup>1</sup>,<sup>2</sup>, Xingsong Ye<sup>1</sup>,<sup>2</sup>, Yongkun Du<sup>1</sup>,<sup>2</sup>, Yunyan Xiao<sup>1</sup>,<sup>2</sup> Yanglin Huang<sup>1</sup>,<sup>2</sup>, Pinxuan Chen<sup>1</sup>,<sup>2</sup>, Jiyu Zhuang<sup>1</sup>,<sup>2</sup>, Cong Xu<sup>1</sup>,<sup>2</sup>, Zhineng Chen<sup>1</sup>,<sup>2</sup>,<sup>†</sup>

<sup>1</sup>Institute of Trustworthy Embodied AI, Fudan University, <sup>2</sup>Shanghai Key Laboratory of Multimodal Embodied AI

<sup>†</sup>Corresponding author

## Abstract

In-image machine translation (IIMT) replaces text in an image with its translation while preserving the surrounding visual context. Existing task-specific models achieve convincing performance but require massive paired training images before and after translation. As advanced text-to-image (T2I) models have reached a practically usable level of multilingual text generation, we argue that this capability can be transferred to IIMT and present FlowTranslator. It is a training-free IIMT framework that first employs an OCR model and an LLM to generate the translation prompt, which is injected into the T2I-based difusion model. To optimize the accurate and natural generation of translated text, we introduce two trajectory controls: Region Coupling Release (RCR) partially decorrelates target-branch noise inside the text region early in the trajectory, while Coupling Expansion (CE) amplifies the emerging edit displacement mid-trajectory. Under the EasyOCR read-back protocol, FlowTranslator achieves average PRIM BLEU and COMET scores of 20.6 and 57.2, respectively, substantially outperforming the strong training-based baseline UniTranslator. On the more challenging and realistic whole-image ScenePair photographs translated into Chinese, it reaches 40.1% exact-match accuracy and surpasses existing editing systems, some of which only address the text-rendering stage rather than the complete IIMT task. These results establish flow-based editing as a strong, fully training-free baseline for IIMT.

Correspondence: zhinchen@fudan.edu.cn

## 1 Introduction

In-image machine translation (IIMT) translates the text embedded in an image and renders the translation back into the original text regions while keeping the rest of the image intact [15, 19]. The task underpins applications from mobile photo translation to localizing signage and product imagery, and has attracted dedicated benchmarks and models [18, 26, 27]. A dominant line of work approaches IIMT with a cascaded pipeline: an OCR model localizes and reads the source text, a translation model converts it, and a rendering module writes the translation back into the image [14, 47]. With the rapid advance of OCR and machine translation, the remaining bottleneck for this task is the rendering step.

Early rendering systems erase the source glyphs and paste separately rendered text, a procedure that leaves visible seams, mismatched typography, and destroyed textures. Subsequent work replaces the paste step with a learned scene-text-editing (STE) generator, first GAN-based [23, 32] and then difusion-based [3, 28, 36], trained to blend the translated glyphs into the surrounding texture. However, these generators need paired images for training. Such cross-lingual pairs are scarce beyond a handful of high-resource directions, so training pipelines fall back on synthetically rendered text [26], which narrows the fonts, layouts, and backgrounds the generator ever sees and leaves the fragile rendering step as the bottleneck of the cascade.

(a) Overall Framework of IIMT  
![](images/06ace2c1038276a6d755e28c9d9d97528a5d57e1369548749965ac2226eee54b.jpg)  
Figure 1 Overview of the IIMT framework and editor paradigms. (a) The pipeline: OCR extracts text → a translator produces the translation → a scene text editor generates the output image. (b) Rule-based: remove, render, and paste separately. (c) Training-based: a learned GAN or difusion editor (requires extra training). (d) FlowTranslator (ours): training-free flow editing with trajectory controls (CE and RCR) for text translation with background preservation.

Text-to-image (T2I) generation has meanwhile progressed to a point where multilingual text rendering is usable in practice: recent flow-matching backbones render legible non-Latin and CJK (Chinese-Japanese-Korean) text directly from a caption [41]. This raises a natural question. Rather than training another dedicated STE generator on scarce cross-lingual pairs, can a frozen T2I model be repurposed, with no task-specific training, to perform the rendering step of an IIMT cascade?

Inversion-free flow editing ofers exactly this machinery. FlowEdit [12] edits a real image with a frozen T2I flow model by integrating an ODE between a source-conditioned branch and a target-conditioned branch, and TextFlow [16] shows that such training-free editing can match trained scene-text editors in visual quality. Both operate on a single word or crop given a user-provided target string. Neither addresses the full IIMT problem of localizing, translating, and re-rendering every text region in an image automatically.

Therefore, we propose FlowTranslator, a novel IIMT system that requires no task-specific training. Every module is a frozen, publicly documented pretrained model, and no parameter is updated or fine-tuned for IIMT (the components themselves are, of course, products of large-scale pretraining). FlowTranslator keeps the modularity of a cascade. A general OCR model [4] finds text blocks, and a frozen LLM translates them. The pipeline replaces the training-based rendering stage with region-masked flow editing on a frozen T2I model [41]. The translation is generated into the original image rather than pasted onto it, so the difusion prior handles perspective, lighting, and typography.

Making this transfer work requires solving a problem specific to cross-lingual, mask-confined editing: an ignition failure of the underlying flow editor. The FlowEdit update is driven by the velocity diference between the source-conditioned and target-conditioned branches, which share the same noise. The diference is meant to grow as the caption change propagates a semantic edit. FlowEdit is demonstrated on edits, such as swapping an object or changing a global style, that alter a substantial fraction of the image and so produce an early, sizeable velocity diference. IIMT edits invert this premise. The text region is typically a small minority of the frame, so the two branches barely difer at the outset and the edit never starts, no matter how the diference is subsequently amplified. This text-minority regime is intrinsic to cross-lingual scene-text editing rather than an artifact of our implementation, so a generic remedy such as increasing the velocity gain cannot repair it: there is no signal left to amplify. We introduce two trajectory controls confined to the text-region mask to force ignition: Region Coupling Release (RCR) replaces the target-branch noise inside the mask with a partially decorrelated sample early in the trajectory, giving the target branch room to diverge. Coupling Expansion (CE) amplifies the nascent edit displacement mid-phase, letting the target branch complete the edit it has started. Both add no forward passes and contribute zero latent update outside the mask.

We evaluate FlowTranslator on three benchmarks spanning synthetic and real imagery, seven languages, and nine translation directions. It achieves the highest average BLEU on Translatotron-V(ision) [15] and the highest average BLEU and COMET on PRIM [26] without any IIMT training data. On ScenePair photographs translated into Chinese [23], FlowTranslator reaches 40.1% exact-match accuracy and outperforms six released editing systems under a common evaluator. These results establish FlowTranslator as a strong training-free baseline across diverse IIMT settings.

In summary, our main contributions are:

• We repurpose a frozen, training-free text-to-image flow model to replace the training-based rendering stage of IIMT cascades and remove its dependence on scarce cross-lingual training pairs.

• We identify an edit-ignition failure of coupled flow editing and propose two mask-confined trajectory controls (RCR and CE) without additional forward passes to resolve it.

• We report results on TTV, PRIM, and ScenePair competitive with trained systems, establishing a strong training-free baseline for IIMT.

## 2 Related Work

In-image Machine Translation. Cascaded systems remain the dominant paradigm for IIMT, decomposing the task into OCR [4–6, 38, 46], machine translation, and text rendering [14, 33, 47]. This modular design has advanced alongside its components: stronger OCR [7, 37, 40] and translation models improve text recovery and translation, while the rendering stage has evolved from removing the source text and pasting separately rendered glyphs to learned scene-text editors that better preserve background, layout, and typography [3, 23, 28]. Cascades can therefore incorporate strong pretrained models and upgrade each stage independently, but they still sufer from error propagation, and rendering often remains the principal source of visual degradation. End-to-end IIMT has been explored as an alternative, from early pixel-level models [19] to Translatotron-V [15], VisTrans [26], and recent unified multimodal systems such as UniTranslator [18] and AnyTrans [22]. These recent models have narrowed the gap and report performance competitive with cascaded baselines on established benchmarks, showing the growing viability of unified modeling. However, all published full-pipeline IIMT systems we are aware of train or fine-tune at least one component on IIMT data, whereas FlowTranslator updates no parameters at all.

![](images/db7e3e6e44d2a40c10795c037d5e08a56a5ac8d46f8754c36d6e560d9647ea5f.jpg)  
Figure 2 Overview of FlowTranslator. OCR extracts the source text and forms a source caption, while a frozen LLM translates it to obtain the target caption. A frozen VAE encodes the source image into $Z _ { 0 } ^ { \mathrm { s r c } } ,$ from which noise initialization produces $Z _ { T } ^ { \mathrm { F E } }$ . At each editing step, the same frozen DiT predicts source- and target-conditioned velocities. Their diference updates the flow-editing latent. RCR perturbs the target-branch noise within the detected text region, and CE amplifies the accumulated displacement before it is fed back to the target branch. The final latent $Z _ { 0 } ^ { \mathrm { F E } }$ is decoded to produce the translated image.

Scene Text Editing. The rendering step of IIMT needs editors to replace words in images with GAN or difusion models. Early systems transfer or regenerate text appearance with adaptive or stroke-level models [23, 25, 32, 35]. Recent difusion editors [3, 13, 28, 39, 42, 44]. Recognition-aware editors further use text recognition to improve content fidelity, as in RS-STE [9]. For cross-lingual rendering, however, these editors need paired images showing the same scene rendered in both scripts. Such pairs are scarce beyond a few high-resource directions, so training pipelines fall back on synthetically composited text [26], narrowing the appearance diversity seen in training. TextFlow [16] shows that training-free flow editing can match trained editors on word crops. Our work lifts training-free editing to the full IIMT task with automatic localization and translation.

Training-free Flow Editing. Flow matching provides a simulation-free training objective for continuous generative flows [17]. FlowEdit [12] constructs an inversion-free ODE between source and target text conditions of a pretrained flow-matching model [2, 8] and is the foundation of our renderer. Other recent studies regularize rectified-flow trajectories or design training-free conditional samplers for image editing [11, 29, 45, 48], reflecting a shift from constructing new ODEs to controlling them. Our RCR and CE belong to this family as spatially masked controls designed specifically for the text-minority ignition setting.

## 3 Method

This section describes how FlowTranslator turns a frozen T2I model into a renderer for in-image machine translation, following the pipeline shown in Fig. 2. FlowTranslator takes FlowEdit [12], an inversion-free editing procedure for pretrained flow-matching models, as its starting point. It then adds two trajectory controls that overcome an ignition failure particular to cross-lingual, mask-confined editing, so that multilingual text is rendered accurately without task-specific training or extra inference cost. No component in the pipeline is trained or fine-tuned.

## 3.1 Overall Framework

Input Preparation. Given a source image $x _ { \mathrm { s r c } } ,$ the OCR model returns text boxes $\{ B _ { i } \} _ { i = 1 } ^ { K }$ and recognized strings. The LLM translates each string and places the source and target strings in two otherwise matched scene descriptions, yielding $c _ { \mathrm { s r c } }$ and $c _ { \mathrm { t g t } } .$ . The image is encoded once by the frozen VAE $( Z _ { 0 } ^ { \mathrm { s r c } } = \mathcal { E } _ { \mathrm { V A E } } ( x _ { \mathrm { s r c } } ) )$ ), and the detected boxes are rasterized at latent resolution after dilation to form the editable mask:

$$
M = { \mathrm { R a s t e r i z e } } _ { \mathrm { l a t } } \left( { \mathrm { D i l a t e } } _ { \delta } \left( \bigcup _ { i = 1 } ^ { K } B _ { i } \right) \right) .\tag{1}
$$

Here, � has value one in the text region and zero elsewhere. The same mask is used for the ordinary FlowEdit update and for both trajectory controls introduced below, so every intervention stays spatially confined to the detected text.

Region-masked Flow Editing. Let $Z _ { 0 } ^ { \mathrm { s r c } }$ be with velocity field $v _ { \theta } ( z , t , c ) _ { . }$ , and let $c _ { \mathrm { s r c } } , c _ { \mathrm { t g t } }$ be the source and target captions produced by the LLM. Starting from $Z _ { T } ^ { \mathrm { F E } } = Z _ { 0 } ^ { \mathrm { s r c } }$ , our region-masked implementation integrates

$$
\begin{array} { r l } & { \quad \Delta v _ { t } = v _ { \theta } ( \hat { z } _ { t } ^ { \mathrm { t a r } } , t , c _ { \mathrm { t g t } } ) - v _ { \theta } ( \hat { z } _ { t } ^ { \mathrm { s r c } } , t , c _ { \mathrm { s r c } } ) , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{2}
$$

where $\hat { z } _ { t } ^ { \mathrm { s r c } } = ( 1 - t ) Z _ { 0 } ^ { \mathrm { s r c } } + t \epsilon$ with $\epsilon \sim { \cal N } ( 0 , I ) .$ , and the target branch is centered at the current edited latent while reusing the same noise realization: $\hat { z } _ { t } ^ { \mathrm { t a r } } = Z _ { t } ^ { \mathrm { F E } } + \hat { z } _ { t } ^ { \mathrm { s r c } } - Z _ { 0 } ^ { \mathrm { s r c } }$ . The time variable runs from the noisy end toward the data end, so � decreases from one to zero over the $\dot { T }$ solver steps. The source branch is a reference path from the source latent toward Gaussian noise. The target branch carries the same stochastic residual but is centered at $Z _ { t } ^ { \mathrm { F E } }$ . The shared noise couples the two branches, while their caption-dependent diference supplies the semantic edit direction. In this view, $Z _ { t } ^ { \mathrm { F E } }$ is the only state that accumulates the edit, while the two DiT evaluations provide the local direction used to update it. The gain $\gamma \geq 1$ scales the velocity diference inside $M ,$ and editing is restricted to the first $n _ { \mathrm { m a x } }$ of $T$ steps. Latents outside � are never modified, so the background stays frozen in latent space.

Ignition Failure. The previous method relies on the source and target branches diverging enough for $\Delta v _ { t }$ to carry a semantic edit signal. In IIMT, the two branches share noise and difer only in a small text region, so their inputs and conditional velocities stay nearly identical near � <sup>≈</sup> 1. Consequently, $\Delta v _ { t }$ and the displacement $Z _ { t } ^ { \mathrm { F E } } - \hat { Z } _ { 0 } ^ { \mathrm { s r c } }$ stay near zero, and the source text remains unchanged. Increasing $\gamma$ cannot amplify a signal that is not there: zero displacement is a fixed point of the update, and escaping it requires modifying the coupling itself rather than the gain applied to it.

We address this fixed point with two trajectory controls confined to the mask $M ,$ each targeting a distinct requirement for ignition. Region Coupling Release first loosens the noise coupling between the branches early in the trajectory, giving the target branch room to diverge from the source. Coupling Expansion then amplifies the resulting displacement in the middle of the trajectory, giving the target branch a direction to follow once it can move.

## 3.2 Region Coupling Release

Region Coupling Release (RCR) targets the root cause of the fixed point, the perfect noise correlation between the branches. Instead of feeding the target branch the source noise $\epsilon ,$ RCR feeds it a partially decorrelated sample inside the mask. With a schedule $s ( t )$ ramping linearly from 1 (for $t \geq \tau _ { \mathrm { h i } } )$ to 0 (for $t \leq \tau _ { \mathrm { l o } } )$ , we draw an independent $\epsilon ^ { \prime } \sim { \cal N } ( 0 , I )$ , fresh at every step and every noise sample like $\epsilon ,$ and form

$$
\epsilon ^ { \mathrm { t a r } } = \rho \epsilon + \sqrt { 1 - \rho ^ { 2 } } \epsilon ^ { \prime } , \qquad \rho = 1 - \kappa s ( t ) ,\tag{3}
$$

$$
\hat { z } _ { t } ^ { \mathrm { t a r } } \gets \hat { z } _ { t } ^ { \mathrm { t a r } } + M \odot t ( \epsilon ^ { \mathrm { t a r } } - \epsilon ) .\tag{4}
$$

Inside the text region the target branch now sees noise that is only �-correlated with the source branch, so the two velocities are evaluated at genuinely diferent points and their diference stops vanishing. The edit has room to ignite.

Three design choices govern this mechanism. First, the decorrelation is partial, $\kappa < 1 :$ Eq. (3) preserves the marginal distribution of $\epsilon ^ { \mathrm { t a r } }$ while interpolating its coupling strength, whereas full decorrelation would sever the branches and degrade the edit into mask-region regeneration. Second, the perturbation in Eq. (4) is additive on top of the accumulated displacement rather than an attenuation of it, so it preserves existing edit progress instead of rescaling the update and shrinking the edit along with the noise, as ordinary guidance schemes do. Third, the release follows a schedule: it is widest early, for $t \geq \tau _ { \mathrm { h i } } ,$ where the trajectory is noise-dominated and divergence is cheap, and it closes completely for $t \leq \tau _ { \mathrm { l o } } ,$ , restoring exact coupling so that late steps refine glyphs under the same conditions as vanilla FlowEdit. Together these choices let RCR open a controlled window for divergence without discarding the coupling that grounds the edit in the source image.

## 3.3 Coupling Expansion

RCR opens room to ignite but supplies no direction of its own, since a decorrelated branch may drift toward any content consistent with the target caption. The needed direction is already present in the trajectory: the accumulated displacement $Z _ { t } ^ { \mathrm { F E } } - Z _ { 0 } ^ { \mathrm { s r c } }$ points from the source toward the emerging edit. Coupling Expansion (CE) feeds this displacement back into the target branch as positive feedback:

$$
\hat { z } _ { t } ^ { \mathrm { t a r } } \gets \hat { z } _ { t } ^ { \mathrm { t a r } } + c \sin ^ { 2 } ( \pi t ) M \odot ( Z _ { t } ^ { \mathrm { F E } } - Z _ { 0 } ^ { \mathrm { s r c } } ) ,\tag{5}
$$

so the target branch is evaluated at a point where the nascent edit already appears more advanced than it is.   
The conditional velocity then responds by completing that edit, and the loop closes on itself.

The $\sin ^ { 2 }$ gate confines this feedback to the middle of the trajectory for a structural reason. Early on, there is no displacement to expand because $Z _ { t } ^ { \mathrm { F E } } \approx Z _ { 0 } ^ { \mathrm { s r c } }$ , so CE stays inert on its own and cannot inject a spurious direction before RCR has created one. Late in the trajectory, the gate closes again to keep runaway amplification from distorting glyph shapes during refinement. The feedback is also self-limiting within the edit window, since � $\sin ^ { 2 } ( \pi t ) \leq { \dot { c } }$ and editing stops at $n _ { \mathrm { m a x } } ,$ so the expansion never compounds without bound.

RCR and CE are complementary rather than redundant: RCR opens the door and CE walks the edit through it. Our subsequent ablation $( \ S ^ { \bar { 4 } . 4 ) }$ shows that CE alone carries most of the gain, while RCR adds a further increment on top and composes positively with CE across most translation directions.

## 4 Experiments

## 4.1 Evaluation and Implementation Details

Benchmarks. TTV (Translatotron-V(ision)) [15] contains $5 1 2 ^ { 2 }$ synthetic text images built from IWSLT14/17. Following the oficial splits, we evaluate De<sup>→</sup>En (3,526), En<sup>→</sup>De (3,524), Fr<sup>→</sup>En (4,925), and Ro<sup>→</sup>En (1,008). PRIM [26] contains photographed real-world 512 <sup>×</sup> 46 text strips. We evaluate all five directions En<sup>→</sup>{De, Fr, $\mathbf { \boldsymbol { C } } \mathbf { \boldsymbol { s } } ,$ Ru, Ro} (340 each) against the oficial Google-translation references. ScenePair [23] provides 1,280 real scene photographs with English text. We translate every whole scene image into Chinese $( \mathrm { E n } {  } \mathrm { Z h } )$ ). For a controlled comparison, all methods use the same text boxes and target translations, with each method receiving them in its oficial input format. Fixing these upstream OCR and translation stages prevents their errors from confounding the comparison. The resulting protocol isolates the rendering stage and measures how reliably each method places the prescribed Chinese text within the specified image regions. Exact-match ACC is computed by the recognizer of PP-OCRv6 [43].

Protocol. Following Lyu et al. [18], translated text is read back from the edited image with EasyOCR v1.7. We use per-direction language packs, sort boxes by top-left coordinate, and join them with spaces as in the oficial PRIM script. We score the text with sacreBLEU using the 13a tokenizer. TTV is lowercased following the IWSLT convention, while PRIM uses both oficial reference sets. We also report COMET with wmt22-comet-da [24]. BLEU follows the standard machine-translation metric [21]. For TTV, image quality is measured by SSIM against the source, using the structural-similarity formulation of Wang et al. [30]. To reduce sensitivity to initialization, we evaluate our configurations with multiple random seeds and report the mean across seeds. For each seed, the Avg. columns are then computed as unweighted means across translation directions. OCR read-back systematically underestimates rendered-text quality, and reader errors are attributed to the system under test. We report EasyOCR numbers in all comparisons for protocol parity. An LLM-bridged alignment check also found <sup>≈</sup>1.4% of TTV De<sup>↔</sup>En reference lines locally misaligned by <sup>±</sup>1 line, a shared defect of the reference files that shifts corpus BLEU on our outputs by well under one point.

Table 1 TTV benchmark with EasyOCR read-back and lowercased sacreBLEU. “IIMT-tuned” marks systems trained or fine-tuned on IIMT data. GPT-Image-1 and Qwen-Image are zero-shot prompted unified models. “Masked FlowEdit” is our baseline with identical hyperparameters and the trajectory controls disabled. Average columns are unweighted means over the four directions. For each metric, best in bold, second underlined.
<table><tr><td></td><td></td><td colspan="4">BLEU↑</td><td colspan="2">Avg.</td></tr><tr><td>Model</td><td>IIMT-tuned</td><td>De→En</td><td>En→De</td><td>Fr→En</td><td>Ro→En</td><td>BLEU↑</td><td>SSIM↑</td></tr><tr><td>GPT-Image-1 [20]</td><td></td><td>20.75</td><td>19.61</td><td>27.38</td><td>24.50</td><td>23.06</td><td>0.734</td></tr><tr><td>Qwen-Image [31]</td><td></td><td>0.23</td><td>0.21</td><td>0.20</td><td>0.22</td><td>0.22</td><td>0.785</td></tr><tr><td>OCR+MT+T2I [15]</td><td>√</td><td>15.37</td><td>13.22</td><td>21.60</td><td>18.34</td><td>17.13</td><td>0.771</td></tr><tr><td>PEIT+T2I [47]</td><td>√</td><td>10.91</td><td>8.77</td><td>18.51</td><td>14.54</td><td>13.18</td><td>0.769</td></tr><tr><td>Translatotron-V [15]</td><td>√</td><td>15.39</td><td>13.23</td><td>22.20</td><td>18.44</td><td>17.32</td><td>0.776</td></tr><tr><td>UniTranslator [18]</td><td>√</td><td>25.03</td><td>13.41</td><td>27.77</td><td>18.45</td><td>21.17</td><td>0.805</td></tr><tr><td>Masked FlowEdit [12]</td><td></td><td>24.43</td><td>17.53</td><td>26.89</td><td>22.84</td><td>22.92</td><td>0.754</td></tr><tr><td>FlowTranslator (Ours)</td><td></td><td>25.15</td><td>17.15</td><td>27.07</td><td>23.37</td><td>23.19</td><td>0.746</td></tr></table>

Implementation. We use Z-Image-Turbo [41], a distilled 6B flow-matching T2I model, with �=50 steps and editing window $n _ { \mathrm { m a x } } { = } 4 5$ . For all TTV and PRIM runs, we set RCR �=0 5 (� =0 6, � =0 3), CE �=0 4, velocity gain $\gamma { = } 1 . 3 ,$ and latent-mask dilation to 15%. Images are resized so the long side is 1024 using bicubic interpolation, with dimensions rounded to multiples of 16. PRIM strips are bicubic-upscaled to the same editing canvas $( 5 1 2 \times 4 6  1 0 2 4 \times 8 0 )$ and downscaled to their original size after editing. Text regions and recognized strings come from the PP-OCRv5 [4]. Translations and captions come from one call per image to a commercial LLM (Doubao-Seed-2.0-lite). The LLM returns the translations embedded in a pair of descriptive captions. The source caption $c _ { \mathrm { s r c } }$ describes the scene with the original strings in quotes. The target caption $c _ { \mathrm { t g t } }$ is identical except that the quoted strings are replaced by their translations. The prompt contains the image and the numbered OCR text lines. It asks the LLM to translate each line into a concise target-language scene-text phrase and to return strict JSON with the translation pairs, a one-sentence English source caption that quotes every source line verbatim, and a target caption that changes only those quoted strings to their translations. The prompt also requires the original line order and all non-text scene descriptions to remain unchanged. All experiments are conducted on a single NVIDIA RTX 4090 GPU.

## 4.2 Main Results

TTV. Tab. 1 compares FlowTranslator with training-based and training-free IIMT systems. Qwen-Image performs poorly with an average BLEU of 0.22, whereas GPT-Image-1 reaches a competitive average of 23.06. FlowTranslator attains the best average BLEU, 23.19 versus 21.17 for UniTranslator, the strongest IIMT-tuned system. It exceeds UniTranslator on three of four directions: 25.15 versus 25.03 on De<sup>→</sup>En, 17.15 versus 13.41 on En<sup>→</sup>De, and 23.37 versus 18.45 on Ro<sup>→</sup>En. On Fr<sup>→</sup>En, it scores 27.07 versus 27.77. Against our own masked-FlowEdit baseline with the same hyperparameters, the controls are roughly neutral on TTV’s large clean glyphs (<sup>−</sup>0<sub>.</sub>4 to <sup>+</sup>0<sub>.</sub>7 BLEU). En<sup>→</sup>De remains our weakest direction at 17.15. A plausible, though unverified, contributor is reader fragility on umlauts and compounds. Our average SSIM (0.746) is below the IIMT-tuned systems’ range of 0.77 to 0.80. Stronger edits deviate more from the reference renders. Fig. 3 provides representative examples from all four TTV directions. FlowTranslator renders the translated content with layouts and visual styles close to the target images while preserving the source background.

![](images/640d7dd2a9d61ba8b1192e46f698e28d32513849ea056dbf78a849b751cb98f2.jpg)  
Figure 3 Qualitative results on TTV. Each row shows the source image, the benchmark target image, and the FlowTranslator output for one translation direction. The examples cover EN<sup>→</sup>DE, FR<sup>→</sup>EN, DE<sup>→</sup>EN, and RO<sup>→</sup>EN.

PRIM. On photographed real-world strips (Tab. 2), FlowTranslator obtains the highest reported scores on every direction except En<sup>→</sup>Ru. For example, it scores 35.2 / 63.0 on En<sup>→</sup>De versus 13.1 / 48.7 for UniTranslator. It also achieves the highest average BLEU, 20.6 versus 12.8, and COMET, 57.2 versus 50.7, despite UniTranslator being fine-tuned on 7M in-domain pairs. Over our own masked-FlowEdit baseline with the same hyperparameters, the controls add +2.4 to +5.3 BLEU per direction (average 16.9<sup>→</sup>20.6, COMET 53.3<sup>→</sup>57.2). Our failure direction is En<sup>→</sup>Ru. Z-Image-Turbo’s prior renders Cyrillic unreliably, producing near-Latin glyph substitutions that OCR cannot read and resulting in 4.6 BLEU. Training-free operation inherits the script coverage of the frozen backbone.

## 4.3 Real-Scene Generalization

Under this rendering-focused protocol, Tab. 3 evaluates FlowTranslator against six existing editing systems. FlowTranslator reaches 40.08% exact-match ACC with an SSIM of 0.9509. Among the two trained editors with Chinese glyph support, it exceeds AnyText by 3.28 points and TextFlux by 25.78 points. The other four systems remain below 0.4% exact-match ACC in this heterogeneous stress test. DifSTE has the highest SSIM at 0.9796 but near-zero ACC, while Kontext has the lowest SSIM at 0.4273. Against the Masked FlowEdit baseline in the same pipeline, the controls lift exact-match ACC from 33.44% to 40.08%, an increase of 6.64 points, while SSIM changes from 0.9533 to 0.9509 Fig. 4 shows qualitative outputs on ScenePair. FlowTranslator replaces the selected English text with the requested Chinese translation while retaining the surrounding scene and the local appearance of the text region.

Table 2 PRIM benchmark with EasyOCR read-back. BLEU uses both oficial reference sets, while COMET uses the Google references. Averages are computed from unrounded per-direction scores.
<table><tr><td rowspan="2">Model</td><td rowspan="2">IIMT-tuned</td><td colspan="5">BLEU↑/ COMET↑</td><td colspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>En-De</td><td>En-Fr</td><td>En-Cs</td><td>En-Ru</td><td>En-Ro</td><td>BLEU</td><td>COMET</td></tr><tr><td>PARSeq-mT-Render [26]</td><td>√</td><td>9.5 / 41.7</td><td>13.8 / 46.9</td><td>7.7 / 43.9</td><td>5.5 / 48.1</td><td>12.8 / 53.5</td><td>9.9</td><td>46.8</td></tr><tr><td>PEIT-Render [47]</td><td>√</td><td>10.4 / 45.1</td><td>14.0 / 48.1</td><td>7.9 / 46.2</td><td>5.3 / 47.7</td><td>14.2 / 52.9</td><td>10.4</td><td>48.0</td></tr><tr><td>Translatotron-V [15]</td><td>√</td><td>1.7 / 34.3</td><td>1.9 / 30.2</td><td>1.1 / 30.5</td><td>0.9 / 32.0</td><td>1.3 / 33.9</td><td>1.4</td><td>32.2</td></tr><tr><td>VisTrans [26]</td><td>√</td><td>12.6 / 44.4</td><td>17.0 / 49.4</td><td>5.9 / 41.8</td><td>7.2 / 49.4</td><td>13.9 / 50.2</td><td>11.3</td><td>47.0</td></tr><tr><td>UniTranslator [18]</td><td>√</td><td>13.1 / 48.7</td><td>21.6 / 58.5</td><td>7.4 / 43.3</td><td>8.4 / 50.2</td><td>13.5 / 52.8</td><td>12.8</td><td>50.7</td></tr><tr><td>Masked FlowEdit [12]</td><td></td><td>29.9 /56.7</td><td>24.6 / 57.7</td><td>15.5 / 56.9</td><td>2.2 / 41.1</td><td>12.3 / 54.3</td><td>16.9</td><td>53.3</td></tr><tr><td>FlowTranslator (Ours)</td><td></td><td>35.2 / 63.0</td><td>28.5 / 60.3</td><td>19.8 / 61.1</td><td>4.6 / 43.9</td><td>14.9 / 57.4</td><td>20.6</td><td>57.2</td></tr></table>

Table 3 ScenePair En<sup>→</sup>Zh whole-image translation on 1,280 real scenes. All methods share the same text boxes and target translations. “STE-tr.” marks scene-text-editing training and “Zh” native Chinese glyph support. Rows without $" \mathrm { Z h ^ { \prime \prime } }$ are capability diagnostics rather than head-to-head competitors.
<table><tr><td>Model</td><td>STE-tr.</td><td>Zh ACC↑</td><td>SSIM↑</td></tr><tr><td>TextDiffuser [3]</td><td>√</td><td>0.00</td><td>0.9434</td></tr><tr><td>DiffSTE [10]</td><td>√</td><td>0.23</td><td>0.9796</td></tr><tr><td>AnyText [28]</td><td>√</td><td>√ 36.80</td><td>0.8157</td></tr><tr><td>TextFlux [34]</td><td>√</td><td>√ 14.30</td><td>0.9506</td></tr><tr><td>FLUX.1-Fill [2]</td><td></td><td>0.39</td><td>0.9231</td></tr><tr><td>FLUX.1-Kontext [1]</td><td></td><td>0.00</td><td>0.4273</td></tr><tr><td>Masked FlowEdit [12]</td><td></td><td>√ 33.44</td><td>0.9533</td></tr><tr><td>FlowTranslator (Ours)</td><td></td><td>√ 40.08</td><td>0.9509</td></tr></table>

## 4.4 Ablations

Trajectory Controls. Holding the pipeline, masks, prompts, and all other hyperparameters fixed, adding RCR and CE raises average PRIM BLEU from 16.88 to 20.57, a relative gain of 22%. The two single-mechanism arms show where this gain comes from. CE alone already recovers 3.33 of the full 3.69-point improvement, roughly 90% of the total. RCR is nonetheless not redundant: it composes positively with CE on four of the five directions, and only on En<sup>→</sup>Ru does adding RCR on top of CE cost a small amount, lowering BLEU from 5.03 to 4.57. This division of labor matches the mechanism described in §3: CE supplies the direction that drives most of the edit, while RCR supplies the room for that direction to act.

Table 4 Component ablations. Control rows are full PRIM runs (340 images per direction, averaged over the five directions) sharing the same hyperparameters (�=1<sub>.</sub>3, 15% dilation, RCR �=0<sub>.</sub>5, CE �=0<sub>.</sub>4) and difering only in the mechanism switches.
<table><tr><td>Component</td><td>Setting</td><td>BLEU↑</td></tr><tr><td rowspan="2">Controls (PRIM avg.)</td><td>Masked FlowEdit (Baseline) +RCR only</td><td>16.88 17.40</td></tr><tr><td>+CE only +RCR+CE (Ours)</td><td>20.21 20.57</td></tr><tr><td rowspan="2">Edit window  $n _ { \mathrm { { m a x } } }$ </td><td>43</td><td>18.95</td></tr><tr><td>45 (Default) 48</td><td>20.57 14.99</td></tr></table>

Edit Window. The edit window $n _ { \mathrm { { m a x } } }$ controls how many of the � solver steps carry the edit. Tab. 4 shows a clear peak at $n _ { \mathrm { { m a x } } } { = } 4 5 \colon$ shrinking it to 43 leaves visible source text and drops BLEU to 18.95, while extending it to 48 over-edits the region and destroys layout, dropping BLEU further to 14.99. We use $n _ { \mathrm { m a x } } { = } 4 5$ for �=50 in all other experiments as the setting that balances these two failure modes. The trajectory controls and the edit window act on diferent benchmarks in a consistent direction. On ScenePair, adding the controls raises exact-match ACC from 33.44% to 40.08% (Tab. 3), the same pattern of improvement seen on PRIM.

![](images/5b5ee39f86ecc3f502aefa23e9a80c10ac39096206952cf6dc286839c504d0b9.jpg)  
Figure 4 Qualitative comparison on whole-image ScenePair En→Zh translation. For each example, we show the source image (with red boxes marking text to be translated), the requested target text, and outputs of existing open-source methods, followed by FlowTranslator. FlowTranslator renders the requested Chinese text within the original region. It preserves both the surrounding scene and the local text appearance.

Cross-benchmark Analysis. Across benchmarks, the relative gain from the trajectory controls tracks how hard the edit is to ignite. The gain is largest on PRIM, where the translated text occupies a small, photographic strip of the image (+22% relative average BLEU). But it is smallest on TTV, where the text is large, clean, and synthetic (+2% on Ro<sup>→</sup>En, its most afected direction). The benchmarks difer in more than text-area ratio, so this pattern is consistent with the ignition interpretation that hard-to-ignite, small-glyph regimes benefit most from explicit trajectory intervention. Importantly, the same control configuration is used across benchmarks, so the diference in gains does not arise from benchmark-specific tuning. The agreement between PRIM BLEU and ScenePair exact-match ACC further shows that the controls improve text completion under both strip-level and whole-image evaluation.

## 5 Conclusion

We presented FlowTranslator, a training-free IIMT framework. It replaced the trained rendering stage of an IIMT cascade with region-masked flow editing on a frozen text-to-image model. Two mask-confined trajectory controls, Region Coupling Release and Coupling Expansion, resolved an ignition failure specific to the text-minority regime of cross-lingual editing. Under the EasyOCR read-back protocol, FlowTranslator attained the highest reported average BLEU and COMET on PRIM. It also attained the highest average BLEU on Translatotron-V(ision), all without any IIMT training data. On real ScenePair photographs translated into Chinese, it reached 40.1% exact-match accuracy, exceeding six released editing systems. Once its branch coupling was corrected for the text-minority regime, a frozen T2I flow model could match or exceed systems trained on millions of IIMT pairs. This established flow-based editing as a strong training-free baseline for IIMT. The method still inherited the script coverage of its frozen backbone, and Cyrillic rendering failed in our experiments. Extending the trajectory controls to backbones with broader script support remains a direction for future work.

## References

[1] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

[2] Black Forest Labs. FLUX. https://github.com/black-forest-labs/flux, 2024.

[3] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdifuser: Difusion models as text painters. In Advances in Neural Information Processing Systems, 2023.

[4] Cheng Cui, Ting Sun, Manhui Lyu, Tingquan Liu, Zelun Zhang, Yikang Ma, Shuangqing Ma, Zhihao Yu, Zhipeng Song, Jinhang Guo, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.

[5] Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, and Yu-Gang Jiang. Svtr: Scene text recognition with a single visual model. In ĲCAI, pages 884–890, 2022.

[6] Yongkun Du, Zhineng Chen, Hongtao Xie, Caiyan Jia, and Yu-Gang Jiang. Svtrv2: Ctc beats encoder-decoder models in scene text recognition. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 20147–20156. IEEE, 2025.

[7] Yongkun Du, Zhineng Chen, Yazhen Xie, Weikang Bai, Hao Feng, Wei Shi, Yuchen Su, Can Huang, and Yu-Gang Jiang. Unirec-0.1 b: Unified text and formula recognition with 0.1 b parameters. In European Conference on Computer Vision, pages 36–55. Springer, 2026.

[8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

[9] Zhengyao Fang, Pengyuan Lyu, Jingjing Wu, Chengquan Zhang, Jun Yu, Guangming Lu, and Wenjie Pei. Recognitionsynergistic scene text editing. arXiv preprint arXiv:2503.08387, 2025.

[10] Jiabao Ji, Guanhua Zhang, Zhaowen Wang, Bairu Hou, Zhifei Zhang, Brian Price, and Shiyu Chang. Improving difusion models for scene text editing with dual encoders. Transactions on Machine Learning Research, 2023.

[11] Jeongsol Kim, Yeobin Hong, Jonghyun Park, and Jong Chul Ye. Flowalign: Trajectory-regularized, inversion-free flow-based image editing. arXiv preprint arXiv:2505.23145, 2025.

[12] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19721–19730, 2025.

[13] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Dongyang Jin, Ryan Xu, Dong Nie, Lei Sun, and Xiangxiang Chu. Flux-text: A simple and advanced difusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025.

[14] Zhibin Lan, Jiawei Yu, Xiang Li, Wen Zhang, Jian Luan, Bin Wang, Degen Huang, and Jinsong Su. Exploring better text image translation with multimodal codebook. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pages 3479–3491, 2023.

[15] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Min Zhang, and Jinsong Su. Translatotron-v(ision): An end-to-end model for in-image machine translation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 5472–5485, 2024.

[16] Yubo Li, Xugong Qin, Peng Zhang, Hailun Lin, Gangyan Zeng, and Kexin Zhang. Towards training-free scene text editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15291–15301, June 2026

[17] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[18] Jiahao Lyu, Pei Fu, Zhenhang Li, Shaojie Zhang, Jiahui Yang, Yu Zhou, Can Ma, Zhenbo Luo, and Jian Luan. Unitranslator: A unified multi-modal framework for end-to-end in-image machine translation. arXiv preprint arXiv:2606.24333, 2026.

[19] Elman Mansimov, Mitchell Stern, Mia Chen, Orhan Firat, Jakob Uszkoreit, and Puneet Jain. Towards end-to-end in-image neural machine translation. In Proceedings of the First International Workshop on Natural Language Processing Beyond Text, pages 70–74, 2020.

[20] OpenAI. Introducing our latest image generation model in the api. https://openai.com/index/ image-generation-api/, April 2025.

[21] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, 2002.

[22] Zhaoyang Qian, Peng Zhang, Bowen Yang, Kai Fan, Yiming Ma, Derek F. Wong, Xiang Sun, and Rongrong Ji. Anytrans: Translate anytext in the image with large scale models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 2432–2444, 2024.

[23] Yadong Qu, Qingfeng Tan, Hongtao Xie, Jianjun Xu, Yuxin Wang, and Yongdong Zhang. Exploring stroke-level modifications for scene text editing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 2119–2127, 2023.

[24] Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. COMET: A neural framework for MT evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 2685–2702, 2020.

[25] Priyanka Roy, Saumik Bhattacharya, Subhankar Ghosh, and Umapada Pal. STEFANN: Scene text editor using font adaptive neural network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13228–13237, 2020.

[26] Yanzhi Tian, Zeming Liu, Zhengyang Liu, Chong Feng, Xin Li, Heyan Huang, and Yuhang Guo. Prim: Towards practical in-image multilingual machine translation. arXiv preprint arXiv:2509.05146, 2025.

[27] Yanzhi Tian, Zeming Liu, Zhengyang Liu, and Yuhang Guo. Exploring in-image machine translation with real-world background. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.

[28] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, 2024.

[29] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024.

[30] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[31] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

[32] Liheng Wu, Cheng Zhang, Jun Liu, Jing Han, Jun Liu, Errui Ding, and Xiang Bai. Editing text in the wild. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1500–1508, 2019.

[33] Yazhen Xie, Xingsong Ye, and Zhineng Chen. Decompose radicals, then reward: Fine-grained inspection for accurate chinese text rendering. arXiv preprint arXiv:2609.37569, 2026.

[34] Yu Xie, Jielei Zhang, Pengyu Chen, Ziyue Wang, Weihang Wang, Longwen Gao, Peiyi Li, Huyang Sun, Qiang Zhang, Qian Qiao, Jiaqing Fan, and Zhouhui Lian. Textflux: An ocr-free dit model for high-fidelity multilingual scene text synthesis. arXiv preprint arXiv:2505.17778, 2025.

[35] Qiang Yang, Jing Huang, and Zhi Lin. Swaptext: Image based texts transfer in scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14700–14709, 2020.

[36] Xingsong Ye, Yongkun Du, Yunbo Tao, and Zhineng Chen. Textssr: difusion-based data synthesis for scene text recognition. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 17464–17473. IEEE, 2025.

[37] Xingsong Ye, Yongkun Du, JiaXin Zhang, Chen Li, Jing LYU, and Zhineng Chen. What’s wrong with synthetic data for scene text recognition? a strong synthetic engine with diverse simulations and self-evolution. In 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16645–16654. IEEE, 2026.

[38] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Zhixian Li, Chong Sun, Chen Li, Jing Lyu, Lianwen Jin, and Zhineng Chen. All-in-one multilingual scene text recognition with script-aware mixture-of-experts. arXiv preprint arXiv:2609.24058, 2026.

[39] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Chong Sun, and Zhineng Chen. Real in, real out: What if we only use real data for scene text editing? In Advances in Neural Information Processing Systems, 2026.

[40] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Haojie Zhang, Chong Sun, Chen Li, Jing Lyu, and Zhineng Chen. Advancing wordart-oriented scene text recognition: Datasets and methods. In European Conference on Computer Vision, pages 77–94. Springer, 2026.

[41] Z-Image Team. Z-image: An eficient image generation foundation model with single-stream difusion transformer. arXiv preprint arXiv:2511.22699, 2025.

[42] Weichao Zeng, Yan Shu, Zhenhang Li, Dongbao Yang, and Yu Zhou. Textctrl: Difusion-based scene text editing with prior guidance control. Advances in Neural Information Processing Systems, 37:138569–138594, 2024.

[43] Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou, et al. Pp-ocrv6: From 1.5 m to 34.5 m parameters, surpassing billion-scale vlms on ocr tasks. arXiv preprint arXiv:2606.13108, 2026.

[44] Yiming Zhao and Zhouhui Lian. Udiftext: A unified framework for high-quality text synthesis in arbitrary images via character-aware difusion models. In European Conference on Computer Vision, pages 217–233, 2024.

[45] Chen Zheng, Yuan Lan, and Yang Wang. Lanpaint: Training-free difusion inpainting with asymptotically exact and fast conditional sampling. arXiv preprint arXiv:2502.03491, 2025.

[46] Tianlun Zheng, Zhineng Chen, Shancheng Fang, Hongtao Xie, and Yu-Gang Jiang. Cdistnet: Perceiving multi-domain character distance for robust text recognition. International Journal of Computer Vision, 132(2):300–318, 2024.

[47] Shaolin Zhu, Shangjie Li, Yikun Lei, and Deyi Xiong. PEIT: Bridging the modality gap with pre-trained models for end-to-end image translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pages 13433–13447, 2023.

[48] Tianrui Zhu, Shiyi Zhang, Jiawei Shao, and Yansong Tang. Kv-edit: Training-free image editing for precise background preservation. arXiv preprint arXiv:2502.17363, 2025.

## Appendix

This appendix provides trajectory-level diagnostics for the two mask-confined controls introduced in the main text. Using four matched arms (Masked FlowEdit (denoted as FlowEdit in the figures), <sup>+</sup>RCR, <sup>+</sup>CE, and $+ \mathrm { R C R + C E } )$ , we test whether RCR relaxes early branch coupling and whether CE amplifies the accumulated displacement without destabilizing the latent trajectory. All diagnostics use 12 paired images under the default setting $( T = 5 0 , n _ { \mathrm { m a x } } = 4 5 , \kappa = 0 . 5 $ , and $c = 0 . 4 )$

## A.1 Trajectory Diagnostics for Region Coupling Release

![](images/d60d5864532a8e59bf0068fd02fae432d6c6acc4d028b4c423a067206874513f.jpg)

(b) edit signal  
![](images/89842a58befba6316fe4fb4bf1c8deb5f54d7f117ddce737c01c94ea32ec4b1e.jpg)

(c) RCR faithfulness  
![](images/0629446766f20c0a253e4f3a19e58c92ec8d7196b8c4c3e7c7e06b5aefac6306.jpg)

(d) first edit step  
![](images/e455accb464eb62008f15104b8fa8993f573b211a756510b1f8f16a00ca58791.jpg)  
Figure S1 RCR diagnostics: dashed/solid pairs nearly overlap at plotting resolution (Masked FlowEdit with $+ \mathrm { C E } ,$ and <sup>+</sup>RCR with <sup>+</sup>RCR<sup>+</sup>CE). (a) In-mask velocity alignment. Bands show the min–max range over 12 images for Masked FlowEdit and <sup>+</sup>RCR. Dotted line: $\tau _ { \mathrm { l o } } = 0 . 3$ . (b) In-mask edit-signal concentration on a log scale. The first third of the edit window is shaded (<sup>+</sup>36%). (c) Prescribed versus realized noise correlation (mean absolute error 0<sub>.</sub>0032). (d) First-step RCR and CE perturbations by arm. Zeros are explicitly labeled.

Paired diagnostic protocol. We instrument the sampling loop with per-step probes and compare four arms on matched stochastic trajectories: Masked FlowEdit, <sup>+</sup>RCR, <sup>+</sup>CE, and <sup>+</sup>RCR <sup>+</sup> CE. The arms difer only in $( \kappa , c )$ set respectively to (0 0), (0<sub>.</sub>5 0), (0 0<sub>.</sub>4), and (0<sub>.</sub>5 0<sub>.</sub>4), with $\tau _ { \mathrm { h i } } = 0 . 6$ and $\tau _ { \mathrm { l o } } = 0 . 3$ . Each arm uses the same 12 images, boxes, prompts, and seed for 50 scheduler steps, of which 45 are edit steps, giving 540 per-step records per arm. Let $\hat { M ^ { \bf { \epsilon } } } \in \{ 0 , 1 \} ^ { 1 \times H \times W }$ be the binary region mask broadcast over the � latent channels, $Z _ { 0 } ^ { \mathrm { s r c } }$ the clean source latent, and $Z _ { t } ^ { \mathrm { F E } }$ the running edit latent at noise level �. All probes use mask-normalized rootmean-square (RMS) values, making them comparable across region sizes: $r _ { \mathrm { i n } } ( x ) = \big ( ( \sum M \odot x ^ { 2 } ) / ( C \sum M ) \big ) ^ { 1 / 2 } .$ and $r _ { \mathrm { o u t } }$ likewise over $1 - M$ . Let $v _ { \mathrm { s r c } , t } = v _ { \theta } \big ( \hat { z } _ { t } ^ { \mathrm { s r c } } , t , c _ { \mathrm { s r c } } \big )$ and $v _ { \mathrm { t a r } , t } = v _ { \theta } ( \hat { z } _ { t } ^ { \mathrm { t a r } } , t , c _ { \mathrm { t g t } } )$ , and let $\Delta v _ { t } = v _ { \mathrm { t a r } , t } - v _ { \mathrm { s r c } , t } .$ To diagnose RCR, we log the in-mask velocity alignment $\cos _ { \mathrm { i n } } ( v _ { \mathrm { t a r } , t } , v _ { \mathrm { s r c } , t } )$ and the edit-signal concentration $r _ { \mathrm { i n } } ( \Delta v _ { t } ) / r _ { \mathrm { o u t } } ( \Delta v _ { t } ) .$ RCR replaces the target-branch noise inside the mask with a �-correlated sample, $\epsilon ^ { \mathrm { t a r } } = \rho \epsilon + \sqrt { 1 - \rho ^ { 2 } } \epsilon ^ { \prime }$ with $\rho ( t ) = 1 - \kappa s ( t )$ and � a linear ramp that is 1 for $t \geq \tau _ { \mathrm { h i } }$ and 0 for $t \leq \tau _ { \mathrm { l o } } ,$ so it injects $p _ { t } ^ { \mathrm { R C R } } = M \odot t \left( \epsilon ^ { \mathrm { t a r } } - \epsilon \right)$ . We log both the prescribed $\rho ( t )$ and the realized $\cos ( \epsilon ^ { \mathrm { t a r } } , \epsilon )$ . The probes require no additional network evaluations and draw no random numbers, so enabling them does not perturb the trajectory being measured.

Table S1 Trajectory diagnostics over 12 images (45 edit steps each). Rows are means across images except the two “max” rows, which are maxima over images and steps. “Early” denotes the first third of the edit window. The ignition step uses a single absolute threshold shared by all arms. Dashes mark probes that are inactive in an arm by construction.
<table><tr><td>Diagnostic</td><td>Masked FlowEdit</td><td>+RCR</td><td> $+ \mathrm { C E }$ </td><td> $+ \mathrm { R C R + C E }$ </td></tr><tr><td>early  ${ \cos } _ { \mathrm { i n } }$ </td><td>0.740</td><td>0.481</td><td>0.738</td><td>0.480</td></tr><tr><td>early signal ratio</td><td>1.94</td><td>2.63</td><td>1.93</td><td>2.63</td></tr><tr><td>ignition step ↓</td><td>9.50</td><td>7.75</td><td>9.42</td><td>7.67</td></tr><tr><td>final  $d _ { \mathrm { i n } }$ </td><td>0.440</td><td>0.443</td><td>0.533</td><td>0.513</td></tr><tr><td>max rel.  ${ \mathrm { C E } } { \mathrm { p e r t . } }$ </td><td>0.000</td><td>0.000</td><td>0.267</td><td>0.246</td></tr><tr><td>max  $| Z _ { t } ^ { \mathrm { F E } } |$ </td><td>3.991</td><td>4.052</td><td>3.991</td><td>4.053</td></tr><tr><td> $| \rho - \cos ( \epsilon ^ { \mathrm { t a r } } , \epsilon ) |$ </td><td>一</td><td>0.0032</td><td>一</td><td>0.0032</td></tr><tr><td>RCR pert. at first step</td><td>一</td><td>0.962</td><td>一</td><td>0.962</td></tr></table>

Paired-trajectory control. Drawing RCR’s extra noise $\epsilon ^ { \prime }$ from the global generator would advance the random stream and desynchronize every subsequent $\epsilon ,$ so the arms would follow diferent trajectories and no per-step diference could be attributed to the mechanism. We therefore draw $\epsilon ^ { \prime }$ from a dedicated fixed-seed generator, and we verify the result rather than assume it: at the first edit step, where CE is inactive because the accumulated displacement is zero, the Masked FlowEdit and $+ \mathrm { C E }$ arms agree to the last logged digit on all 12 images, while <sup>+</sup>RCR difers on all 12. The update is also region-masked, so $r _ { \mathrm { o u t } } ( Z _ { t } ^ { \mathrm { F E } } - \bigcup _ { 0 } ^ { \infty } \mathrm { s r c } ) = \breve { 0 }$ exactly at every step of every arm. With $n = 1 2$ paired images, a unanimous direction in a sign test corresponds to a two-sided $p = 4 . 9 \times 1 0 ^ { - 4 }$ . All reported �-values are exact two-sided paired sign tests with ties excluded.

In-mask edit-signal concentration. RCR lowers the early in-mask alignment of the two velocity fields from $\mathrm { c o s } _ { \mathrm { i n } } = 0 . 7 4 0 \mathrm { t o } 0 . 4 8 1$ , so the residual $\Delta v _ { t }$ that drives the edit is subject to less cancellation inside the region. The concentration of that residual in the mask rises from 1<sub>.</sub>94 to 2<sub>.</sub>63, i.e. <sup>+</sup>36% (Fig. S1(a,b)). Both efects hold on $1 2 / 1 2$ images $( p = 4 . 9 \times 1 0 ^ { - 4 } \mathrm { { e a c h } ) } .$ , and both originate at high noise: RCR stops injecting additional decorrelation once the schedule reaches $\tau _ { \mathrm { l o } } = 0 . 3$ , after which late glyph refinement proceeds under restored branch-noise coupling. The comparison also localizes the cause: <sup>+</sup>CE alone moves the alignment by only <sup>−</sup>0<sub>.</sub>002, so it is the branch-noise decorrelation, not the displacement amplification, that produces the stronger in-mask signal.

Realized noise coupling. The mean absolute diference between prescribed $\rho ( t )$ and measured $\cos ( \epsilon ^ { \mathrm { t a r } } , \epsilon )$ is 0<sub>.</sub>0032 across all steps and images, including the linear ramp (Fig. S1(c)). This agreement confirms that the implemented correlation follows the prescribed schedule.

First-step activation. CE injects $\alpha ( t ) M \odot \delta _ { t } ,$ , which is identically zero while $\delta _ { t } = 0$ . RCR injects $M \odot t ( \epsilon ^ { \mathrm { t a r } } - \epsilon )$ independently of the accumulated displacement and is therefore active at the noisy end: its first-step magnitude averages 0<sub>.</sub>962 over 12 images, versus 0<sub>.</sub>000 for CE (Fig. S1(d)). The two terms are thus active in complementary phases, RCR decaying as the branches recouple and CE peaking in mid-trajectory under its sin $\mathsf { \Omega } ^ { 2 } ( \pi t )$ gate.

## A.2 Trajectory Diagnostics for Coupling Expansion

Paired displacement probes. Using the same four paired arms, we measure the in-mask displacement from the source,

$$
d _ { \mathrm { i n } } ( t ) = r _ { \mathrm { i n } } \big ( Z _ { t } ^ { \mathrm { F E } } - Z _ { 0 } ^ { \mathrm { s r c } } \big ) .\tag{S1}
$$

CE amplifies the accumulated displacement $\delta _ { t } = Z _ { t } ^ { \mathrm { F E } } - Z _ { 0 } ^ { \mathrm { s r c } }$ by $\alpha ( t ) = c \sin ^ { 2 } ( \pi t )$ , injecting $p _ { t } ^ { \mathrm { C E } } = \alpha ( t ) M \odot \delta _ { t }$ We log this perturbation relative to the target-branch latent it modifies, together with the per-step update magnitude and max $| Z _ { t } ^ { \mathrm { F E } } |$ . The update is region-masked, and we verify $r _ { \mathrm { o u t } } ( Z _ { t } ^ { \mathrm { F E } } - Z _ { 0 } ^ { \mathrm { s r c } } ) = 0$ at every step of every arm.

(a) departure from source  
![](images/644090c1af5f685398ec7e7d08fa468bc03ec6b94c1eae512590cf36e8553074.jpg)

(b) ignition step  
![](images/f96ed4693696add7260ceb420e38799394716200114bb0cb6c41cd0a588e3499.jpg)

(c) CE stays bounded(c) CE remains self-limiting  
![](images/97f295e4b2345f8c9fb4e319ce385ede28c357db9441b4eed2e5cc825aae7511.jpg)

(d) latent extreme  
![](images/4c50e78f516f5bdca4accc3333fb63b5181eb8744d6bd9508af9f978ed24bfe1.jpg)  
Figure S2 CE efectiveness and stability: pairing as in Fig. S1. (a) In-mask displacement normalized by the Masked FlowEdit final value. Dashed line: shared 10% ignition threshold. (b) Ignition step with per-image values. (c) Relative CE perturbation. CE-of arms lie at zero. (d) Paired per-step diference in the latent maximum relative to Masked FlowEdit. Global maxima for Masked FlowEdit and <sup>+</sup>RCR<sup>+</sup>CE difer by 1<sub>.</sub>6%.

Final edit displacement. CE raises the final in-mask displacement from 0 440 to 0 533, i.e. <sup>+</sup>21%, on 11/12 images $( p = 6 . 3 \times 1 0 ^ { - 3 } )$ , and to 0<sub>.</sub>513 (<sup>+</sup>16%) alongside RCR on 10/12 images $( p = 0 . 0 3 9 )$ . This division of labor matches the mechanism: RCR alone changes the final displacement by only <sup>+</sup>0<sub>.</sub>7% and primarily afects edit onset, whereas CE afects final magnitude. Using an ignition threshold shared by all arms (the first step at which $d _ { \mathrm { i n } } ( t )$ reaches 10% of the Masked FlowEdit arm’s final displacement on the same image), the edit departs from the source at step 9<sub>.</sub>50 for the baseline and 7<sub>.</sub>75 under RCR, i.e. 1<sub>.</sub>75 steps earlier, earlier on 9/12 images, later on 1/12 and tied on $2 / 1 2 \left( p = 0 . 0 2 1 \right)$ . <sup>+</sup>CE alone reaches 9<sub>.</sub>42 while <sup>+</sup>RCR<sup>+</sup>CE reaches 7<sub>.</sub>67, earlier on $1 0 / 1 2 ( p = 0 . 0 1 2 ) ( \mathrm { F i g . } \ : \mathrm { S 2 ( a , b ) } )$ . In this controlled comparison, the earlier ignition is therefore attributable to RCR, consistent with CE having nothing to amplify at the start.

Self-limiting feedback. Because CE amplifies the displacement it helps create, we measure the resulting perturbation. Its magnitude relative to the target-branch latent follows the intended $\sin ^ { 2 } ( \pi t )$ profile, rising to a mean peak of 0<sub>.</sub>155 near $t \approx 0 . 5$ and returning to nearly zero before glyph refinement (largest value over all images and steps 0<sub>.</sub>267). The per-step update grows smoothly rather than spiking. The latent maximum changes only 1<sub>.</sub>6%, from 3<sub>.</sub>991 for Masked FlowEdit to 4<sub>.</sub>053 with both controls, indicating little change in latent scale (Fig. S2(c,d)). All efects are mask-confined: out-of-mask displacement is zero at every logged step.

![](images/96f121e8d623e8586fdb82782225500d08937e0639ef1e9b779c52a8dbcda338.jpg)  
Figure S3 Additional qualitative examples of full-image scene text translation into Chinese. Each pair shows the source image on the left and the corresponding FlowTranslator output on the right.

## A.3 Visualization and Discussion

Qualitative results. Fig. S3 provides six qualitative examples of full-image scene-text translation into Chinese. Across diferent source languages and text layouts, FlowTranslator replaces the selected source text with the requested Chinese translation while preserving the surrounding scene and the local appearance of the text region. These cases are consistent with the trajectory diagnostics above: the controls strengthen the edit within the detected text mask, while the background remains unchanged in latent space.