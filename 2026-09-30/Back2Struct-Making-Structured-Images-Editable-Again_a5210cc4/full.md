# Back2Struct: Making Structured Images Editable Again

Pengyu Yan Yixin Wu Yunjie Tian David Doermann

University at Buffalo, SUNY {pyan4,yixinwu,yunjieti,doermann}@buffalo.edu

![](images/84baf14cd25e6395b5a43bebb697ae5a071436fd04ff70ca62e3174b1b73aa32.jpg)  
Figure 1: A qualitative comparison on recovering the reference images among the most advanced LLMs, including GPT-5 (Thinking mode) [45], Qwen3-Max (over 1 trillion parameters) [61], Gemini-2.5-Pro [12], and Back2Struct (Ours). Back2Struct exceeds its counterparts with negligible confusion or mistakes in its responses, showcasing its superior ability in “making structured images editable again”. Note that Back2Struct has only 7B parameters.

## Abstract

Structured images, such as diagrams, charts, and flowcharts, are inherently symbolic and can be compactly represented in an editable format, yet in practice, they are often rendered as images, and therefore not graphically editable. This mismatch presents a significant challenge for researchers, engineers, and designers who wish to incorporate modified versions of existing graphic content into new materials without manually reconstructing it. In this study, we present Back2Struct, which “makes structured images editable again” by directly recovering vector graphics code (SVG / XML) from image representations. Given an image of a structured graphic, Back2Struct predicts semantically object-level SVG / XML code that explicitly encodes text, shapes, topology, and layout, rather than performing low-level pixel vectorization. The generated code can be seamlessly imported into tools such as PowerPoint, allowing users to edit, refine, restyle, and reuse graphic content while preserving structural fidelity. Beyond supervised finetuning on ground-truth SVG token sequences, we further optimize Back2Struct with reward-based learning to better match deployment-time requirements: the output should be syntactically valid, properly concise, and visually faithful to the input diagram. Specifically, we design a composite reward that jointly encourages SVG / XML compilability, length consistency with the reference code, and structural or semantic similarity between the generated and ground-truth graphics. These complementary signals guide the model to produce SVGs that are not only closer to the training distribution, but also more complete, editable, and renderable in practice. Experiments show that Back2Struct improves accuracy, editability, validity, and user alignment over baselines. Dataset and code are available at: pengyu965.github.io/Back2Struct.github.io

## 1 Introduction

Structured content, including diagrams, charts, flowcharts, user interface designs, and block architectures, is central to how researchers, engineers, and designers communicate algorithms, pipelines, system architectures, and design intent. Unlike natural images, its utility depends on discrete symbolic structures, such as nodes and edges, groupings and alignments, textual labels, geometric primitives, and layout regularities that encode precise relationships rather than appearance alone. These structures can in principle be compactly represented in editable formats such as SVG / XML [15, 17, 26, 34, 41], making them interpretable, verifiable, and directly modifiable. However, structured graphics are often shared as screenshots, raster exports, or figures embedded in papers and slides. Once rendered into pixels, their symbolic structure becomes inaccessible: individual elements cannot be selected, text and shapes cannot be reliably edited, and constraints such as alignment, grouping, and connectivity are lost. As a result, even simple edits may require manually reconstructing the entire graphic, motivating the recovery of an editable and semantically meaningful representation.

Existing approaches provide only partial solutions. Pixel-level generative models can synthesize visually plausible images, but they struggle to enforce global structural constraints and produce noneditable bitmaps [16, 19, 36, 43]. Conventional vectorization methods recover curves, contours, and low-level shapes, but not high-level semantics such as node-edge relationships, hierarchical groupings, textual associations, or layout intent [5, 6, 40]. Recent multimodal large language models [1, 3, 48] and their successors [4, 31] show strong visual understanding and code-generation abilities, yet they are not explicitly optimized for recovering faithful, executable, and practically usable SVG / XML code from realistic structured images [17, 28, 41, 47]. In particular, supervised fine-tuning on ground-truth SVG token sequences mainly encourages code imitation, but does not directly optimize deployment-time properties such as validity, conciseness, and visual faithfulness.

In this work, we argue that making structured images editable again requires treating them as symbolic artifacts rather than ordinary raster images. We introduce Back2Struct, a framework that directly recovers manipulatable vector graphics code from image representations, as shown in Figure 1. Given an image of structured content, Back2Struct predicts object-level SVG / XML code that explicitly encodes text, shapes, topology, and layout, rather than performing low-level pixel vectorization. The recovered code can be imported into standard authoring tools such as PowerPoint, allowing users to edit, refine, restyle, and reuse graphic content while preserving structural fidelity and interpretability. To better match practical editing needs, we further optimize Back2Struct beyond supervised finetuning with reward-based learning. We design a composite reward that captures complementary requirements: SVG / XML compilability, length consistency with the reference code, and structural or semantic similarity between the generated and ground-truth graphics.

These reward signals encourage outputs that are syntactically valid, neither truncated nor unnecessarily verbose, and faithful to the original visual structure and semantic content. As a result, the model generates SVGs that are not only closer to the training distribution, but also more complete, editable, and usable in real workflows. We evaluate Back2Struct on representative structured image recovery scenarios where both semantic correctness and layout fidelity are critical. Experiments show that Back2Struct improves structural accuracy, editability, SVG validity, and alignment with user intent over strong vision–language and code-generation baselines. Overall, the results suggest that codecentric recovery, combined with deployment-aware reward optimization, is an effective way to turn static structured images back into editable visual artifacts.

The main contributions of this paper are as follows.

• We collect and curate the StructHub dataset containing 84K paired structured raster images and their associated SVG/XML code, together with an evaluation benchmark. This is a large-scale, high-value dataset dedicated to editable structured image generation.

• We propose Back2Struct, a framework that makes structured images editable again by recovering semantically meaningful SVG / XML code from image representations, and formulate this task as object-level vector graphics code generation that explicitly models text, shapes, topology, and layout.

• We introduce reward-based optimization to improve deployment-time properties of the generated SVG / XML code, including validity, length consistency, and structural or semantic faithfulness. Extensive experiments show that Back2Struct improves structural accuracy, editability, SVG/XML validity, and alignment with user intent compared with strong multimodal and code-generation baselines.

## 2 Related Work

Multi-modal Large Language Models. Recent large vision–language models (LVLMs) extend large language models (LLMs) from text-only reasoning to visual inputs by jointly modeling images and text. Representative models, including LLaVA [31], InternVL [11], InstructBLIP [13], Qwen-VL [58– 60], Gemini [12, 48, 49], and GPT [1, 8, 45], enable open-ended visual reasoning, captioning, and grounding across diverse inputs such as documents, charts, diagrams, and UI screenshots. However, they typically rely on continuous visual features and produce free-form text, which limits their ability to recover fine-grained symbolic structures or reconstruct compositional visual inputs in a faithful and executable form. This motivates our study of structured image understanding, where visual perception is translated directly into editable graphics.

Structured Image Understanding. Vision–language pretraining has improved image understanding by aligning visual perception with natural language and capturing high-level semantics beyond low-level pixels [24, 25, 39, 50, 51, 54]. Many works further study structured visual domains, including graph understanding [21, 38, 53, 63], table understanding [35, 46, 62, 65, 66], chart understanding [2, 18, 29, 30, 33, 55–57], and document or layout understanding [20, 22, 23]. These methods mainly treat structured images as information sources for question answering, field extraction, or semantic interpretation. In contrast, our goal is to faithfully reconstruct the original structured image by recovering its content, topology, layout, and style as editable vector graphics.

Image-to-Code Generation with LVLMs. Image-to-code generation converts visual inputs into executable or markup representations, providing a natural path toward editable visual understanding. Early works such as Im2LaTeX [14] and pix2code [7] translate mathematical expressions or GUI screenshots into LaTeX, HTML, DOM, or DSL tokens [10, 23]. In vector and diagram domains, methods such as DiffVG [27] and DeepSVG [9] generate SVG-style representations through differentiable rendering or tokenized path decoding. Recent multimodal and LLM-based methods further recover plotting code, graph specifications, or flow logic from charts, diagrams, and flowcharts [18, 29, 38, 41, 55, 57, 63].

Our work focuses on reconstructing diagrams as editable SVGs. Instead of relying on high-level plotting languages such as Python, which may introduce ambiguity and lose visual fidelity, we adopt object-level SVGs to couple fine-grained image understanding with structured output generation.

## 3 Method

## 3.1 Motivation

Recovering SVG code from a diagram image can be formulated as a sequence generation problem, where an input image x is mapped to an XML token sequence y forming a valid SVG document. However, token-level correctness is insufficient: the output should be syntactically valid, structurally faithful, and perceptually consistent with the input. Standard supervised fine-tuning with crossentropy loss provides only a token-level surrogate and is poorly aligned with these goals. For example, a small syntax error may invalidate the entire SVG, while visually equivalent SVGs can have different token sequences due to alternative attribute orders or coordinate forms. This objective mismatch motivates holistic reward signals that directly evaluate validity, structure, and visual quality, despite being often non-differentiable and difficult to optimize with standard supervised learning.

![](images/0550220b05d4d2902be22efc2e66d693bcd30774dfa2a8fa9ad0265f0df7a882.jpg)  
Figure 2: Overview of Back2Struct. Given an RGB reference image and/or text, Back2Struct generates object-level XML code that can be losslessly converted into an editable PowerPoint file.

We therefore frame image-to-SVG generation as a reinforcement learning problem. Let $\pi _ { \theta } ( y \mid x )$ denote the policy that generates an SVG sequence y conditioned on an input diagram image x. Our goal is to maximize the expected task reward:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \big [ R ( y , y ^ { * } ) \big ] ,\tag{1}
$$

where $y ^ { * }$ denotes the ground-truth SVG and $R$ is a composite reward that measures syntactic validity, structural correspondence, and perceptual quality. The framework is illustrated in Figure 2.

## 3.2 Reward Design

Re-producing high-quality SVGs demands two separate assurances: the result must be a usable artifact—one that actually renders into an image which is viewable and editable—and it must be visually faithful to the intended diagram. These criteria are interdependent: an SVG can be syntactically valid yet depict the wrong diagram, while a visually convincing attempt might not render at all. We formalize these with a rendering gate and a perceptual quality term, multiplied together so that successful rendering is treated as a prerequisite for—rather than a part of—the reward.

Syntactic Reward. Let syntactic(y) obtain the SVG code from a generation y and rasterize it onto a white canvas at the target image resolution. The syntactic reward is

$$
R _ { \operatorname { s y n t a c t i c } } ( y ) = \mathbf { 1 } { \big [ } \mathbf { r e n d e r } ( y ) \operatorname { s u c c e e d s } { \big ] } \in \{ 0 , 1 \} .\tag{2}
$$

This enforces a stricter validity condition than mere XML well-formedness: successful rendering implies the output is both parseable and rasterizable, thereby also filtering out outputs that are syntactically valid yet cannot be rendered. If a generation fails to render, it is assigned zero overall reward, since it cannot be displayed or edited.

Perceptual Reward. We assess diagram-level quality directly in pixel space, not over SVG tokens, so that generations are incentivized based on what they look like rather than how the underlying code is formatted. Let $I ( y )$ denote the rendered prediction and let $I ^ { * }$ be the target image the model is conditioned on. A frozen DINOv2 ViT-B/14 encoder ϕ maps each image to an $\ell _ { 2 } \cdot$ -normalized embedding; we compute the cosine similarity between embeddings and clamp it to [0, 1],

$$
\begin{array} { r } { \sin ( y ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \cos \bigl ( \phi ( I ( y ) ) , \phi ( I ^ { * } ) \bigr ) . } \end{array}\tag{3}
$$

However, the raw similarity is a biased proxy for quality: even an content-free output can achieve nonzero similarity to the target. In particular, a blank canvas $I _ { \mathcal { O } }$ with matching dimensions yields a nontrivial baseline sim $\begin{array} { r } { \mathbf { \iota } _ { \otimes } = \mathrm { \bar { c l i p } } _ { [ 0 , 1 ] } \mathrm { \bar { c o s } } ( \phi ( I _ { \mathcal { O } } ) , \phi ( I ^ { * } ) ) } \end{array}$ , because diagrams in our setting are mostly

white. To account for this, we reward fidelity relative to the content-free baseline by subtracting the per-image floor and then renormalizing:

$$
R _ { \mathrm { p e r c } } ( y ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \bigg ( { \frac { \mathrm { s i m } ( y ) - \mathrm { s i m } _ { \emptyset } } { 1 - \mathrm { s i m } _ { \emptyset } } } \bigg ) .\tag{4}
$$

With this definition, $R _ { \mathrm { p e r c } }$ gives 0 to a blank (or blank-equivalent) render and 1 to a perceptually accurate one, with calibration performed separately for each target so the reward captures only the fidelity gained by actually drawing meaningful content.

Gated Composition. Our composition also removes the need for a separate length-fidelity reward. We define the final reward as the product of the render gate and the perceptual term:

$$
R ( y ) = R _ { \mathrm { s y n t a c t i c } } ( y ) \cdot R _ { \mathrm { p e r c } } ( y ) \in [ 0 , 1 ] .\tag{5}
$$

Gating, instead of a weighted sum, is deliberate: an SVG cannot be rendered, there is no resulting image to evaluate, so the rendering term functions as a {0, 1} mask rather than a separately optimizable reward component. We settled on this design after observing that an additive form, $R = w _ { r } R _ { \mathrm { s y n t a c t i c } } +$ $w _ { l } R _ { \mathrm { l e n g t h \_ f d e l i t y } } + w _ { p } R _ { \mathrm { s y n t a c t i c } } .$ , is reward-hackable: the policy collapses onto short, trivially renderable but almost empty SVGs, because (i) successful rendering gives a constant bonus available to any syntactically valid output, and (ii) without baseline subtraction, an empty image can still achieve a high score when compared to the target image, which contains only sparse pixel information (Appendix Fig. 6). Baseline subtraction (Eq. 4) addresses (ii), and multiplicative gating (Eq. 5) eliminates (i), ensuring the reward increases only when the output both renders and adds visually accurate content. Together, these mechanisms also replace the needs of explicit length-fidelity term: under-generation (blank or sparse images) is pushed toward zero by the baseline, while runaway over-generation that fails to terminate is suppressed by the gate. This rule-based reward design with a gating mechanism achieves the best performance, while avoiding the need for a complex reward design or an LLM-based reward judge (Section 5.4)).

## 3.3 Policy Optimization with GRPO

We optimize the SVG generation policy $\pi _ { \theta }$ with Group Relative Policy Optimization (GRPO) [44], which forgoes a learned value network and instead estimates advantages by comparing completions sampled for the same input. This suits image-to-SVG generation, where absolute reward scales vary widely across diagrams of differing structure and complexity, so a within-image comparison is more stable than an absolute baseline.

For each input image x we sample a group of G completions $\{ y _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ and score each with the composite reward $R ( y _ { i } )$ of Eq. (5). We use the group mean as the baseline and, following the reward-scaling ablation of prior work [32], do not normalize by the group standard deviation:

$$
\hat { A } _ { i } = R ( y _ { i } ) - \mu _ { \mathcal { G } } , \qquad \mu _ { \mathcal { G } } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R ( y _ { j } ) .\tag{6}
$$

The update thus depends only on whether a completion is better or worse than the others for the same image, not on its absolute reward.

We maximize the clipped surrogate objective with a KL penalty to a frozen reference policy $\pi _ { \mathrm { r e f } } \colon$

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \operatorname* { m i n } \bigl ( \rho _ { i } \hat { A } _ { i } , ~ \mathrm { c l i p } ( \rho _ { i } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \bigr ) - \beta \mathbb { D } _ { \mathrm { K L } } \bigl [ \pi _ { \theta } ( \cdot \mid x ) \bigr \| ~ \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \bigr ] \right] ,\tag{7}
$$

where $\rho _ { i } = \pi _ { \theta } ( y _ { i } \mid x ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i } \mid x )$ is the importance ratio, ϵ the clipping threshold, and $\beta$ the KL weight. The clip bounds the per-step policy change and the KL term keeps the policy near the reference, together stabilizing optimization of our non-differentiable, render-gated perceptual reward.

Off-policy correction. For throughput, rollouts are produced by a served VLLM snapshot of the policy while gradients are computed by a separate training process; small numerical differences between the two make the sampled log-probabilities a biased proxy for $\pi _ { \theta _ { \mathrm { o l d } } } .$ . We correct this with token-level truncated importance sampling: per-token ratios are clipped from above at a cap C before entering Eq. (7), which bounds the variance introduced by the generator/optimizer mismatch without discarding otherwise-valid rollouts. Throughout, we use $G = 4 , \epsilon = 0 . 2 , \beta = 0 . 0 4$ , and $C = 3$

![](images/e30a93e19f26ca1c17e019c08639034595baec119fe734b53cc530410865a296.jpg)

![](images/406bfb76059ffd2e180b189d305d32c42c9379875fd972f60e4bb39f77ceb83a.jpg)

![](images/89350cb71252263ea42db84903a9a665343b274666fbda77bfa27597a55970f0.jpg)  
Figure 3: StructHub dataset statistics. Token-length distributions, content profiles, and difficulty compositions of StarDiag-HQ, OpenDiag, and NNArch.

## 4 Data Curation

We introduce StructHub, a dataset of 84,121 structured diagram SVGs from three sources. Each sample contains a $5 1 2 \times 5 1 2$ rasterized PNG rendering, generated with a Chromium/Playwright renderer, paired with its ground-truth SVG code for image-to-code training and evaluation.

StarDiag-HQ is derived from the web-scraped StarVector dataset [41]. We retain high-resolution SVGs with sufficient structural-text density and shape-element proportion, while removing icon-like or illustration-heavy samples. All files are capped at 8,192 tokens, resulting in 52,875 training SVGs. This subset mainly covers network/cloud architecture diagrams, flowcharts, pipelines, and data-flow graphs, providing a large and relatively clean source of structured diagrams.

OpenDiag is collected from open-source and open-knowledge repositories, including GitHub, Wikimedia Commons, and diagrams produced by tools such as draw.io, PlantUML, and Excalidraw. We apply a conservative SVG cleaning pipeline to remove raster images, editor metadata, layout bloat, unused definitions, empty groups, and overly long files, while normalizing code structure and converting simple paths into primitive shapes. The subset contains 27,953 SVGs, of which 18,800 satisfy the 8,192-token training threshold. Compared with StarDiag-HQ, OpenDiag contains more diverse and structurally complex real-world diagrams.

NNArch targets neural network and deep learning architecture diagrams, which are underrepresented in general SVG corpora. It is built from curated academic, blog, repository, and licensed diagram sources. This subset contains 3,293 SVGs, with 1,453 retained for training after token filtering. NNArch is intentionally more complex and path-heavy, capturing the dense topology of modern neural architecture figures.

StructHub Benchmark. For evaluation, we build a held-out benchmark, StructHub-Benchmark, with 1,000 samples from all three subsets: 569 from StarDiag-HQ, 334 from OpenDiag, and 97 from NNArch. We split the benchmark by groundtruth SVG length into three difficulty tiers: easy (≤2,048 tokens; n = 333), medium (2, 048– 4, 096 tokens; $n = 3 3 3 )$ , and hard $( > 4 , 0 9 6$ tokens; n = 334). This enables evaluation across different levels of structural complexity. All

Table 1: Quantitative statistics of StructHub dataset sources.
<table><tr><td>Subset</td><td>Train (Full)</td><td>Train (≤8,192 tok)</td><td></td><td>Retention Benchmark</td></tr><tr><td>StarDiag-HQ</td><td>52,875</td><td>52,875</td><td>100.0%</td><td>569</td></tr><tr><td>OpenDiag</td><td>27,953</td><td>18,800</td><td>67.3%</td><td>334</td></tr><tr><td>NNArch</td><td>3,293</td><td>1,453</td><td>46.6%</td><td>97</td></tr><tr><td>Total</td><td>84,121</td><td>73,128</td><td>86.9%</td><td>1,000</td></tr></table>

benchmark samples are withheld from training, with no data augmentation or overlap beyond the source-level separation.

Dataset Statistics. The dataset statistics are summarized in Table 1 and Figure 3. The 8,192-token threshold affects the subsets differently: StarDiag-HQ remains unchanged, OpenDiag retains roughly two-thirds, and NNArch retains less than half, reflecting its long and topologically dense architecture diagrams. The excluded long-tail samples are kept to evaluate generalization beyond the training context length.

Table 2: Image-level and text-level evaluation on StructHub across difficulty levels. SR (render success rate) is reported as a percentage; each axis (Struct, Flow, Style) is rated on a 0–5 scale and the Overall score is scaled to 0–100.
<table><tr><td rowspan="2">Model</td><td colspan="4">Image-level Evaluation</td><td colspan="4">Text-level Evaluation (GPTscore)</td></tr><tr><td>SR↑</td><td>DINO↑</td><td>LPIPS↓</td><td>SSIM ↑</td><td>Overall ↑</td><td>Struct ↑</td><td>Flow ↑</td><td>Style ↑</td></tr><tr><td colspan="9">Overall</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>18.8</td><td>0.1566</td><td>0.9067</td><td>0.1201</td><td>13.29</td><td>0.78</td><td>0.68</td><td>0.53</td></tr><tr><td>StarVector-8B</td><td>31.4</td><td>0.2694</td><td>0.7936</td><td>0.2170</td><td>21.05</td><td>1.04</td><td>1.09</td><td>1.03</td></tr><tr><td>Qwen2.5-VL-7B</td><td>52.3</td><td>0.3527</td><td>0.7834</td><td>0.3276</td><td>27.56</td><td>1.65</td><td>1.32</td><td>1.16</td></tr><tr><td>Qwen2.5-VL-7BSFT</td><td>42.8</td><td>0.3715</td><td>0.7403</td><td>0.2891</td><td>33.49</td><td>1.80</td><td>1.65</td><td>1.57</td></tr><tr><td>Back2Struct (RL)</td><td>71.5</td><td>0.5622</td><td>0.6473</td><td>0.4966</td><td>51.85</td><td>2.87</td><td>2.60</td><td>2.31</td></tr><tr><td colspan="9"></td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>26.5</td><td>0.2231</td><td>0.8659</td><td>Easy 0.1702</td><td>19.51</td><td>1.15</td><td>1.00</td><td>0.78</td></tr><tr><td>StarVector-8B</td><td>46.1</td><td>0.3923</td><td>0.7023</td><td>0.3123</td><td>31.90</td><td>1.61</td><td>1.74</td><td>1.44</td></tr><tr><td>Qwen2.5-VL-7B</td><td>69.3</td><td>0.4776</td><td>0.7123</td><td>0.4196</td><td>39.66</td><td>2.29</td><td>2.04</td><td>1.61</td></tr><tr><td>Qwen2.5-VL-7BSFT</td><td>56.0</td><td>0.4866</td><td>0.6630</td><td>0.3789</td><td>44.58</td><td>2.39</td><td>2.30</td><td>1.99</td></tr><tr><td>Back2Struct (RL)</td><td>84.0</td><td>0.6384</td><td>0.5956</td><td>0.5796</td><td>62.63</td><td>3.44</td><td>3.25</td><td>2.70</td></tr><tr><td colspan="9"></td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>19.6</td><td>0.1631</td><td>0.9004</td><td>Medium 0.1288</td><td>13.38</td><td>0.78</td><td>0.68</td><td>0.54</td></tr><tr><td>StarVector-8B</td><td>37.0</td><td>0.3232</td><td>0.7533</td><td>0.2583</td><td>25.27</td><td>1.23</td><td>1.25</td><td>1.31</td></tr><tr><td>Qwen2.5-VL-7B</td><td>51.8</td><td>0.3546</td><td>0.7816</td><td>0.3306</td><td>26.88</td><td>1.62</td><td>1.25</td><td>1.16</td></tr><tr><td>Qwen2.5-VL-7BSFT</td><td>45.5</td><td>0.3985</td><td>0.7161</td><td>0.3091</td><td>36.50</td><td>1.98</td><td>1.76</td><td>1.74</td></tr><tr><td>Back2Struct (RL)</td><td>75.6</td><td>0.6088</td><td>0.6182</td><td>0.5275</td><td>55.14</td><td>3.08</td><td>2.72</td><td>2.47</td></tr><tr><td colspan="9"></td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>10.2</td><td>0.0836</td><td>Hard 0.9536</td><td>0.0614</td><td>7.01</td><td>0.41</td><td>0.36</td><td>0.28</td></tr><tr><td>StarVector-8B</td><td>11.1</td><td>0.0932</td><td>0.9249</td><td>0.0808</td><td>6.02</td><td>0.28</td><td>0.28</td><td>0.34</td></tr><tr><td>Qwen2.5-VL-7B</td><td>35.7</td><td>0.2263</td><td>0.8561</td><td>0.2331</td><td>16.17</td><td>1.03</td><td>0.67</td><td>0.72</td></tr><tr><td>Qwen2.5-VL-7BSFT</td><td>27.0</td><td>0.2297</td><td>0.8413</td><td>0.1795</td><td>19.43</td><td>1.04</td><td>0.89</td><td>0.98</td></tr><tr><td>Back2Struct (RL)</td><td>55.0</td><td>0.4399</td><td>0.7278</td><td>0.3830</td><td>37.83</td><td>2.09</td><td>1.83</td><td>1.76</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/a3b4a201090040d4460fc84b571c6ab9be1f8205dbe21c9e14c0c5e619a5d2e6.jpg)  
Figure 4: Image-level performance comparison across different dataset sources.

Data collection remains central but challenging, as editable structured images are manually created and hard to scale. This makes StructHub-Full valuable for structured image recovery. Representative examples are shown in Appendix Figure 5, and the dataset will be released to support research.

## 5 Experiments

## 5.1 Metrics

Image Evaluation. Unlike natural images, structured images (e.g., diagrams, charts, UI layouts) comprise discrete, symbolic elements with explicit spatial and logical relationships, whereas natural images depict continuous scenes with semantics that are implicit in dense pixel values. Consequently, instead of evaluating similarity purely at the pixel level, we assess it at the structural level.

We use DINO Score [37], LPIPS [64], and SSIM [52] to measure the similarity between rendered SVGs and ground-truth images. We use DINO Score, LPIPS, and SSIM to measure perceptual and structural similarity [41], and omit MSE since our goal is faithful structural and stylistic reconstruction rather than pixellevel replication. Additionally, we report the success rate (SR), defined as the proportion of SVG outputs that compile (render) successfully without any auxiliary SVG tools.

Text Evaluation. Structured images are well captured by an object-level SVG representation. However, image-level similarity metrics may not fully reflect how effectively the model transfers semantic elements into an editable SVG format. To address this, we introduce a rubricbased GPTScore that evaluates structural similarity, flow correctness, and stylistic authenticity based on the generated SVG code. Further details are provided in the Appendix.

Table 3: Comparison results on cases with successfully compiled SVG / XML outputs.
<table><tr><td>Model</td><td>DINO ↑ LPIPS ↓</td><td></td><td>SSIM ↑ GPT ↑</td><td></td></tr><tr><td colspan="5">Overall</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.6749</td><td>0.5856</td><td>0.6270</td><td>52.73</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>0.8347</td><td>0.5027</td><td>0.6402</td><td>70.88</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.8855</td><td>0.4648</td><td>0.6532</td><td>77.95</td></tr><tr><td>GPT-5</td><td>0.8852</td><td>0.4350</td><td>0.6573</td><td>77.21</td></tr><tr><td>Back2Struct</td><td>0.8697</td><td>0.3896</td><td>0.6738</td><td>78.63</td></tr><tr><td colspan="5">Easy</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.6894</td><td>0.5848</td><td>0.6057</td><td>57.25</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>0.8418</td><td>0.4942</td><td>0.6421</td><td>73.60</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.8818</td><td>0.4495</td><td>0.6668</td><td>77.34</td></tr><tr><td>GPT-5</td><td>0.8935</td><td>0.4038</td><td>0.6718</td><td>77.31</td></tr><tr><td>Back2Struct</td><td>0.8689</td><td>0.4012</td><td>0.6673</td><td>80.32</td></tr><tr><td colspan="5">Medium</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.6845</td><td>0.5785</td><td>0.6381</td><td>51.88</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>0.8333</td><td>0.4915</td><td>0.6581</td><td>68.36</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.8864</td><td>0.4656</td><td>0.6471</td><td>79.32</td></tr><tr><td>GPT-5</td><td>0.8840</td><td>0.4426</td><td>0.6504</td><td>77.29</td></tr><tr><td>Back2Struct</td><td>0.8829</td><td>0.3661</td><td>0.6807</td><td>79.37</td></tr><tr><td colspan="5">Hard</td></tr><tr><td>Qwen2.5-VL-7B</td><td>0.6332</td><td>0.5974</td><td>0.6522</td><td>45.24</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>0.8190</td><td>0.5458</td><td>0.6011</td><td>68.63</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.8883</td><td>0.4793</td><td>0.6457</td><td>77.10</td></tr><tr><td>GPT-5</td><td>0.8781</td><td>0.4585</td><td>0.6498</td><td>77.02</td></tr><tr><td>Back2Struct</td><td>0.8461</td><td>0.4077</td><td>0.6761</td><td>73.20</td></tr></table>

## 5.2 Experimental Setup

We first fine-tune Qwen2.5-VL-7B [4] with

LoRA for 1 epoch on 73K training samples, and then optimize Back2Struct with GRPO [44] for another 3 epoch on 4K samples from OpenDiag and NNArch. All experiments are conducted on 4 NVIDIA RTX PRO 6000 Blackwell GPUs.

For raster evaluation, we use CairoSVG [9] for SVG rendering: instead of padding white edges, we preserve the original SVG viewBox before resizing predictions and references to 512 × 512 to avoid inflated scores from white backgrounds; we fully penalize the predictions that cannot be succesfully rendered. We split StructHub by target SVG length into easy (≤ 2048 tokens), medium (2048–4096 tokens), and hard (> 4096 tokens), with a maximum length of 8192 tokens.

## 5.3 Comparisons

Table 4: Text-to-SVG generalization by finetuning on text-to-SVG task.
<table><tr><td>Model</td><td>DINO↑</td><td>LPIPS ↓</td><td>FID↓ CLIP↑</td></tr><tr><td colspan="4">Easy</td></tr><tr><td>Qwen3-30B GPT-5</td><td>0.8618 0.8294</td><td>0.3218 0.3726</td><td>77.11 88.52</td><td>8.9622 8.8018</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.8802</td><td>0.3296</td><td>67.72</td><td>8.8342</td></tr><tr><td>B2S (Ours)</td><td>0.8987</td><td>0.3515</td><td>64.82</td><td>9.8271</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5">Medium</td></tr><tr><td>Qwen3-30B</td><td>0.8917</td><td>0.3137</td><td>78.20</td><td>6.6514</td></tr><tr><td>GPT-5</td><td>0.8596</td><td>0.3577</td><td>92.63</td><td>6.5073</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.9070</td><td>0.3114</td><td>72.18</td><td>6.0003</td></tr><tr><td>B2S (Ours)</td><td>0.9307</td><td>0.2877</td><td>44.86</td><td>8.1660</td></tr><tr><td colspan="5">Hard</td></tr><tr><td>Qwen3-30B</td><td>0.8630</td><td>0.3263</td><td>69.92</td><td>6.4930</td></tr><tr><td>GPT-5</td><td>0.8673</td><td>0.3450</td><td>74.73</td><td>5.8403</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.9058</td><td>0.3059</td><td>54.61</td><td>6.9166</td></tr><tr><td>B2S (Ours)</td><td>0.9310</td><td>0.3131</td><td>55.84</td><td>7.1958</td></tr></table>

## StructHub Evaluation.

Most existing Image-to-SVG methods focus on vectorizing icons, shapes, emojis, fonts, and other simple graphic assets, while our goal is

to recover editable structured images through SVG code generation. To the best of our knowledge, StarVector [41] is the state-of-the-art model for diagram-oriented SVG generation, so we mainly compare Back2Struct with StarVector-8B [41] and Qwen3-VL-8B [4] on Image-to-SVG tasks. Although RLRF [42] is a more recent model, its code and weights have not been released, making direct comparison infeasible. As shown in Table 2, Back2Struct achieves the best results on DINO, LPIPS, SSIM, and GPTScore, while also producing the most reliable SVG code with the highest compile success rate. In contrast, StarVector only generates stable SVGs when the maximum token budget is below ∼4000, and its performance drops sharply on the hard split. Figure 4 demonstrates the comparison results grouped by dataset source.

Performance on Valid Outputs. For users, what matters more is the final quality of the generated results, rather than whether the model needs to regenerate a few times after failed attempts. Therefore, Table 3 summarizes the performance on examples that are successfully compiled. Our model, with

Table 5: Ablation on reward design across difficulty tiers. S/F/P denote the syntactic, fidelity, and perceptual rewards; R/L/– denote rule-based, LLM-based, and unused. The active rewards are combined either by summation (Σ) or by our render-gated composition (gate).
<table><tr><td rowspan="2">Method</td><td colspan="2">Reward</td><td rowspan="2"></td><td colspan="4">Easy</td><td colspan="4">Medium</td><td colspan="4">Hard</td></tr><tr><td>SF</td><td>P</td><td>SR↑</td><td>DINO↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>SR↑</td><td>DINO↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>SR↑</td><td>DINO ↑</td><td>LPIPS↓</td><td>SSIM ↑</td></tr><tr><td>Base + SFT</td><td></td><td></td><td></td><td>56.0</td><td>0.4866</td><td>0.6630</td><td>0.3789</td><td>45.5</td><td>0.3985</td><td>0.7161</td><td>0.3091</td><td>27.0</td><td>0.2297</td><td>0.8413</td><td>0.1795</td></tr><tr><td></td><td></td><td></td><td>L</td><td>56.3</td><td>0.4898</td><td>0.6535</td><td>0.3890</td><td>45.5</td><td>0.3965</td><td>0.7185</td><td>0.3051</td><td>23.1</td><td>0.1985</td><td>0.8625</td><td>0.1551</td></tr><tr><td>↓+RLΣ</td><td>R</td><td></td><td>L</td><td>55.7</td><td>0.4797</td><td>0.6597</td><td>0.3827</td><td>46.4</td><td>0.4070</td><td>0.7111</td><td>0.3125</td><td>25.2</td><td>0.2143</td><td>0.8536</td><td>0.1649</td></tr><tr><td></td><td>RR</td><td></td><td>L</td><td>55.7</td><td>0.4842</td><td>0.6663</td><td>0.3718</td><td>45.8</td><td>0.4042</td><td>0.7098</td><td>0.3117</td><td>23.4</td><td>0.1982</td><td>0.8613</td><td>0.1584</td></tr><tr><td></td><td></td><td>RR R</td><td></td><td>74.4</td><td>0.5184</td><td>0.6904</td><td>0.4434</td><td>59.3</td><td>0.4072</td><td>0.7589</td><td>0.3713</td><td>40.5</td><td>0.2684</td><td>0.8387</td><td>0.2594</td></tr><tr><td> $\scriptstyle \to \operatorname { R L } _ { \mathrm { g a t e } }$ </td><td></td><td> $\mathrm { ~ { ~ \bf ~ R ~ } ~ } - { \mathrm { ~ { ~ \bf ~ R ~ } ~ } }$ </td><td></td><td>84.0</td><td>0.6384</td><td>0.5956</td><td>0.5796</td><td>75.6</td><td>0.6088</td><td>0.6182</td><td>0.5275</td><td>55.0</td><td>0.4399</td><td>0.7278</td><td>0.3830</td></tr><tr><td>∆</td><td></td><td></td><td></td><td>(+28.0)</td><td>(+.152)</td><td>(+.067)</td><td>(+.201)</td><td>(+30.1)</td><td>(+.210)</td><td>(+.098)</td><td>(+.218)</td><td>(+28.0)</td><td>(+.210)</td><td>(+.114)</td><td>(+.204)</td></tr></table>

only 7B parameters, outperforms commercial models such as GPT-5 [45] and Gemini-2.5-Pro [12]. Since querying these models is very expensive, we randomly sampled 200 samples from the StructHub dataset for this comparison.

Text-to-SVG Generalization. Our method can also support text-to-SVG generation. We further fine-tune the image-to-SVG Back2Struct model on a text-to-SVG dataset and summarize the results in Table 4. Promisingly, our method remains competitive with GPT-5 [45] and Gemini-2.5-Pro [12], and even shows clear advantages.

Qualitative Evaluation. Figure 1 compares Back2Struct against GPT-5-Thinking [1], Qwen3- MAX [60], and Gemini-2.5-Pro [12] on Image-to-SVG and Text-to-SVG generation. Where these ba selines miss shapes, misalign arrows, or misplace text, Back2Struct better preserves fine-grained spatial layout, object hierarchy, and inter-element relations, producing structurally and stylistically faithful diagrams— consistent with our quantitative results.

## 5.4 Ablation Study

Table 5 ablates our reward along two axes: signal choice and composition. From the SFT baseline, RL with only partial or LLM-based signals (×, L), (R, L), or (L, L) yields small, unstable gains: Easy render success is flat (56.0 → 55.7–56.3) and Hard worsens (27.0 → 23.1–25.2), indicating that a single/noisy signal cannot reliably guide SVG recovery. Using both rule-based signals with a summed reward (Σ, R/R) improves SR (Easy 56.0 → 74.4, Hard 27.0 → 40.5) but degrades perceptual quality, increasing LPIPS on Easy/Medium (0.6630→0.6904, 0.7161→0.7589).

The main driver is composition. Replacing summation with our render-gated composition—keeping the same rule-based components and training data—improves all metrics. Gating boosts SR over the summed reward by $+ 9 . 6 / + 1 6 . 3 / + 1 4 . 5$ on Easy/Medium/Hard $( 7 4 . 4 \to 8 \bar { 4 } . 0 , 5 9 . 3 \to 7 5 . 6 $ 40.5 → 55.0) and reduces LPIPS at every tier (0.6904 → 0.5956, 0.7589 → 0.6182, 0.8387 → 0.7278), eliminating the summation trade-off. Relative to SFT, the gated reward increases SR by +28.0/ + 30.1/ + 28.0, improves DINO by up to +0.21 and SSIM by up to +0.22, and reduces LPIPS by 0.067/0.098/0.114. Since summed and gated variants share components and data, the gap isolates composition: gating makes syntactic and perceptual signals cooperate by treating rendering as a prerequisite, removing the need for a separate length-fidelity reward.

## 6 Conclusion

In this work, we presented Back2Struct, a code-centric framework that makes structured images editable again by recovering semantically meaningful SVG / XML code from raster inputs. Rather than treating diagrams, charts, and flowcharts as ordinary images, Back2Struct represents them as symbolic structures composed of text, shapes, topology, and layout, enabling direct rendering, import into authoring tools, and downstream editing. We further introduced reward-based optimization to align training with deployment needs, jointly encouraging syntactic validity, structural fidelity, and perceptual faithfulness. Experiments on StructHub show that Back2Struct outperforms strong vision–language, code-generation, and diagram-oriented SVG baselines across image-level metrics, text-level GPTScore, compile success rate, and qualitative results. Ablation studies further confirm that the proposed reward components are complementary and necessary for reliable editable SVG recovery.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] Saleem Ahmed, Pengyu Yan, David Doermann, Srirangaraj Setlur, and Venu Govindaraju. Spaden: sparse and dense keypoint estimation for real-world chart understanding. In International Conference on Document Analysis and Recognition, pages 77–93. Springer, 2023.

[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[5] Jonas Belouadi, Anne Lauscher, and Steffen Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. arXiv preprint arXiv:2310.00367, 2023.

[6] Jonas Belouadi, Simone Ponzetto, and Steffen Eger. Detikzify: Synthesizing graphics programs for scientific figures and sketches with tikz. Advances in Neural Information Processing Systems, 37:85074–85108, 2024.

[7] Tony Beltramelli. pix2code: Generating code from a graphical user interface screenshot. In Proceedings of the ACM SIGCHI symposium on engineering interactive computing systems, pages 1–6, 2018.

[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[9] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems, 33:16351–16361, 2020.

[10] Yuntao Chen, Jian Gu, Zhen Wang, Tong Xu, Zhiwen Xu, Xiangnan Zhao, Jianwei Yin, and Kaimin Zheng. Ui2code: Transforming ui screenshots into structured gui code. In Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems (CHI), Honolulu, HI, USA, 2020. ACM. doi: 10.1145/3313831.3376405. URL https://doi.org/10.1145/ 3313831.3376405.

[11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.

[12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[13] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267, 2023.

[14] Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush. Image-to-markup generation with coarse-to-fine attention. In International Conference on Machine Learning, pages 980–989. PMLR, 2017.

[15] Jon Ferraiolo, Fujisawa Jun, and Dean Jackson. Scalable vector graphics (SVG) 1.0 specification. iuniverse Bloomington, 2000.

[16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications ofthe ACM, 63(11):139–144, 2020.

[17] Gustave V Hahn-Powell and Diana Archangeli. Autotrace: An automatic system for tracing tongue contours. The Journal of the Acoustical Society of America, 136(4\_Supplement): 2104–2104, 2014.

[18] Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. Chartllama: A multimodal llm for chart understanding and generation. arXiv preprint arXiv:2311.16483, 2023.

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[20] Yupan Huang, Yiheng Xu, Minghao Xu, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia (ACM MM), 2022.

[21] Parag Jain and Mirella Lapata. Integrating large language models with graph-based reasoning for conversational question answering. arXiv preprint arXiv:2407.09506, 2024.

[22] Geewook Kim, Jinyoung Hong, Seonghyeon Yun, and Seunghyun Kim. Donut: Document understanding transformer without ocr. In European Conference on Computer Vision (ECCV), 2022.

[23] Kenton Lee, Mandar Joshi, Kristina Toutanova, Jason Wei, Jianmo Ni, Iulia Turc, Sharan Narang, Quoc Le, and Minh-Thang Luong. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning (ICML), 2023.

[24] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021.

[25] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping languageimage pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.

[26] Tzu-Mao Li, Michal Lukác, Michaël Gharbi, and Jonathan Ragan-Kelley. Differentiable vector ˇ graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG), 39(6): 1–15, 2020.

[27] Tzu-Mao Li, Michal Lukác, Gharbi Michaël, and Jonathan Ragan-Kelley. Differentiable vector ˇ graphics rasterization for editing and learning. ACM Trans. Graph. (Proc. SIGGRAPH Asia), 39(6):193:1–193:15, 2020.

[28] Jinwei Lin. Live: Latex interactive visual editing. arXiv preprint arXiv:2405.06762, 2024.

[29] Fangyu Liu, Julian Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. Deplot: One-shot visual language reasoning by plot-to-table translation. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 10381–10399, 2023.

[30] Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, and Julian Eisenschlos. Matcha: Enhancing visual language pretraining with math reasoning and chart derendering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12756–12770, 2023.

[31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

[32] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

[33] Junyu Luo, Zekun Li, Jinpeng Wang, and Chin-Yew Lin. Chartocr: Data extraction from charts images via a deep hybrid framework. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, pages 1917–1925, 2021.

[34] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16314–16323, 2022.

[35] Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar. Tableformer: Table structure understanding with transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4614–4623, 2022.

[36] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR, 2021.

[37] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[38] Huitong Pan, Qi Zhang, Cornelia Caragea, Eduard Dragut, and Longin Jan Latecki. Flowlearn: Evaluating large vision-language models on flowchart understanding. arXiv preprint arXiv:2407.05183, 2024.

[39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[40] Juan A Rodriguez, David Vazquez, Issam Laradji, Marco Pedersoli, and Pau Rodriguez. Figgen: Text to scientific figure generation. arXiv preprint arXiv:2306.00800, 2023.

[41] Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16175–16186, 2025.

[42] Juan A Rodriguez, Haotian Zhang, Abhay Puri, Aarash Feizi, Rishav Pramanik, Pascal Wichmann, Arnab Mondal, Mohammad Reza Samsami, Rabiul Awal, Perouz Taslakian, et al. Rendering-aware reinforcement learning for vector graphics generation. arXiv preprint arXiv:2505.20793, 2025.

[43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[45] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[46] Brandon Smock, Rohith Pesala, and Robin Abraham. Pubtables-1m: Towards comprehensive table extraction from unstructured documents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4634–4642, 2022.

[47] Yiren Song, Xuning Shao, Kang Chen, Weidong Zhang, Zhongliang Jing, and Minzhe Li. Clipvg: Text-guided image manipulation using differentiable vector graphics. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 2312–2320, 2023.

[48] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[49] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

[50] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 3156–3164, 2015.

[51] Yiyu Wang, Jungang Xu, and Yingfei Sun. End-to-end transformer based model for image captioning. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 2585–2594, 2022.

[52] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612, 2004.

[53] Yujie Xing, Xiao Wang, Bin Wu, Hai Huang, and Chuan Shi. Unifying and enhancing graph transformers via a hierarchical mask framework, 2025. URL https://arxiv.org/abs/2510. 18825.

[54] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In International conference on machine learning, pages 2048–2057. PMLR, 2015.

[55] Zhengzhuo Xu, Bowen Qu, Yiyan Qi, Sinan Du, Chengjin Xu, Chun Yuan, and Jian Guo. Chartmoe: Mixture of diversely aligned expert connector for chart understanding. arXiv preprint arXiv:2409.03277, 2024.

[56] Pengyu Yan, Saleem Ahmed, and David Doermann. Context-aware chart element detection. In International conference on document analysis and recognition, pages 218–233. Springer, 2023.

[57] Pengyu Yan, Mahesh Bhosale, Jay Lal, Bikhyat Adhikari, and David Doermann. Chartreformer: Natural language-driven chart image editing. In International Conference on Document Analysis and Recognition, pages 453–469. Springer, 2024.

[58] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[59] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[60] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[61] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[62] Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. Tabert: Pretraining for joint understanding of textual and tabular data. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics (ACL), 2020.

[63] Abhay Zala, Han Lin, Jaemin Cho, and Mohit Bansal. Diagrammergpt: Generating opendomain, open-platform diagrams via llm planning. arXiv preprint arXiv:2310.12128, 2023.

[64] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[65] Yixin Zhang, Xiang Li, Wenxuan Sun, Fangyu Yuan, Qiang Wang, and Y. Zhao. Tablellm: Large multimodal models for table understanding and generation. arXiv preprint arXiv:2403.12345, 2024.

[66] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno-Yepes. Image-based table recognition: Data, model, and evaluation. In European Conference on Computer Vision (ECCV), 2020.

# Back2Struct: Making Structured Images Editable Again

Supplementary

![](images/a8803fbf963c594236525a7b895e9d1f4c50871b1a6c96384ac2d47908b6ccbb.jpg)  
Figure 5: Examples from our dataset. Each raster image is paired with an editable SVG counterpart, providing high-quality pixel–code supervision for image-to-SVG generation.

## A StructHub Dataset

## A.1 SVG Cleaning and Normalization Details

Beyond the curating and preprocessing pipeline described for OpenDiag in the main text, all three subsets share a common post-processing step applied after source-specific cleaning: (1) the root <svg> element is given an explicit width, height, and viewBox if any are missing, ensuring unambiguous coordinate semantics for the renderer; (2) namespace prefixes (svg:, xlink:) are normalized to unprefixed equivalents wherever SVG 1.1 permits; and (3) a final UTF-8 re-encoding pass removes null bytes and non-printable control characters that occasionally appear in crawled files. These steps are applied uniformly so that the model sees a consistent SVG dialect regardless of source.

## A.2 Benchmark Construction and Stratification

The 1,000-sample StructHub-Benchmark is drawn from the held-out portion of each source using stratified sampling to ensure balanced representation across both source and difficulty tier. Source pro portions roughly mirror the training distribution (StarDiag-HQ: 56.9%, OpenDiag: 33.4%, NNArch: 9.7%), and each source is further stratified into three equal-sized difficulty buckets (n ≈ 333 each) based on ground-truth token count: easy (≤2,048 tokens), medium (2,049–4,096 tokens), and hard (>4,096 tokens). Stratification is done independently per source to avoid over-representing short StarDiag-HQ samples in the easy tier. No sample appearing in the benchmark was ever used for training or reward computation.

## A.3 Content Diversity

Figure 5 shows representative examples drawn from all three subsets, illustrating the breadth of diagram types present in StructHub. The collection spans a wide spectrum of structural diagram genres: deep neural network architectures with multi-layer topology (top-left, top-right), racetrack and schematic diagrams (top-center), electrical circuit schematics (center), floor plans and spatial layouts (center-right), data-flow and parameter-count visualizations (center), and time-series charts with annotated legends (bottom-left). This diversity is deliberate: diagrams that share the same surface rendering (e.g., a flowchart and a circuit schematic both use boxes and arrows) may require structurally very different SVG encodings, stressing both the syntactic and semantic axes of the generation task. The NNArch subset in particular contributes diagrams with dense path geometry and sparse text annotation—a regime where pixel-level similarity metrics diverge most sharply from structural correctness, motivating the multi-axis reward design described in Method Section.

## A.4 Licensing and Data Availability

StarDiag-HQ is derived from the StarVector-diagram dataset, which aggregates SVGs from open web sources; we apply no additional license restrictions beyond those of the original corpus. OpenDiag is assembled from GitHub repositories and Wikimedia Commons content released under permissive open-source licenses (MIT, Apache-2.0, CC-BY variants); repository-level license files were checked and repositories with non-permissive or proprietary licenses were excluded. NNArch contains a subset of commercially licensed diagram assets; these samples are provided for research use only and must not be redistributed in commercial products. We will release the full dataset (code, cleaned SVGs, rendered PNGs, and benchmark splits) upon acceptance, subject to the above license constraints.

## B Evaluation Metrics

In this section, we provide additional details on the evaluation setup used in our experiments and explain the rationale for adopting a protocol that differs from prior work. We further describe our text-level evaluation rubric (GPTScore), which we use to assess the quality of the generated SVG code.

## B.1 Image-level evaluation

In the context of image-level comparison, model predictions are evaluated utilizing three prevalent similarity metrics: DinoScore, LPIPS, and SSIM. Given that LPIPS is a metric of perceptual distance where a lower value indicates better similarity, it is transformed into a similarity score via the expression 1 − LPIPS, ensuring that all evaluation metrics adhere to the “higher-is-bette” standard. While these metrics are conventional for assessing raster images, the evaluation of our tasks presents additional complexities: the model outputs are in the form of SVG code, thus rendering setup and management of failure cases are pivotal and can substantially influence the final scores.

The rendering of SVG files involves flexibility in resolution and aspect ratio of the viewbox, where parameters such as canvas size, padding strategy, and scaling directly affect the ultimate rasterized image. Furthermore, the management of invalid or incomplete SVG predictions (e.g., SVGs that fail to compile) introduces additional variability. If not meticulously addressed, these details may obscure the actual visual discrepancies between predictions and ground-truth images. Figure 6 elucidates these issues through two representative examples: one imperfect prediction (left) and one near-perfect prediction (right). For each example, we present: 1) the ground-truth image; 2) the failure-case image; 3) the rendered output of the model when the SVG compiles successfully.

Previous research, specifically StarVector, approaches rendering by displaying each SVG on a fixed square canvas, substituting failure cases with a pure white image. Conversely, our setup involves rendering SVGs while maintaining their original aspect ratio, subsequently resizing them uniformly to $5 1 2 \times 5 1 2$ , and representing failure cases with a black image.

Under StarVector’s configuration, the substantial white space introduces significant noise in the final score, as depicted in Figure 6 (a), where the failure case (white image) achieves higher SSIM and 1 − LPIPS scores than the successfully compiled result. This problem is exacerbated in cases with higher image ratios, as demonstrated in Figure 6 (c), where the failure case yields an SSIM score of 0.9088, which is unrealistically high. In our configuration, substituting the white image with a black image as the failure case markedly decreases all evaluation scores. To further mitigate noise from the background, we resize the original image instead of employing a “padding” operation. As a result, the evaluation score significantly declines from Figure 6 (a) to Figure 6 (b) on compiled but imperfect outcomes, thereby more accurately reflecting the method’s performance.

![](images/381cd43691126a33ebe01942498980ed95756f65f50aecc7472936e721b73591.jpg)  
Figure 6: An ablative comparison on image-level evaluation. Different evaluation configurations can significantly affect the final scores, and our setup is designed to reduce background noise and provide a more reliable reflection of model performance.

## B.2 Text-level evaluation – GPTScore

In addition to metrics at the image level, we implement a text-level evaluation protocol, termed GPTScore, to appraise the structural, logical, and stylistic fidelity of the generated SVG code in comparison to the ground-truth SVG. While image-level metrics deliver a broad similarity signal predicated on rendered appearance, the text-level evaluation meticulously examines the SVG code structure itself, providing more detailed insights into whether the model accurately encapsulates objects, their interrelationships, and stylistic attributes. This is particularly crucial in our context, where the ultimate objective is to generate diagrams that are both editable and semantically accurate.

GPTScore assesses each prediction along three dimensions (as shown in Figure 7): (1) Structural Integrity, (2) Flow Accuracy, and (3) Style Similarity. Each dimension is rated on a scale from 0 to 5 in 0.5-point increments, with the final score being consolidated on a 0 to 100 scale using equal weighting unless specified otherwise. Below, we delineate the design principles, scoring rubric, and interpretive guidelines employed in our evaluation.

## B.2.1 Structure Evaluation

The Structure axis measures how well the predicted SVG reproduces the fundamental building blocks of the diagram, including: (i) the inventory of objects (e.g., rectangles, ovals, diamonds, arrows, text labels), (ii) their relative positions, and (iii) the overall spatial topology. A score of 5 indicates that the prediction closely matches the ground truth in terms of object types, counts, labels, and layout. Minor deviations that do not alter the semantic meaning of the diagram (e.g., small coordinate shifts) are tolerated in the 4–5 range. Scores of 2–3 correspond to noticeable missing or extra elements and layout shifts that begin to impact clarity, while a score of 1 indicates that the diagram structure is largely different. A score of 0 is assigned when the prediction is not comparable or is essentially unrelated to the reference SVG.

To avoid penalizing irrelevant differences, we instruct the evaluator to ignore attribute ordering, XML namespace noise, minor rounding, non-visible metadata, and unused <defs> elements. Minor coordinate quantization is also not penalized as long as the relative topology and alignment of objects are preserved.

![](images/51ffcc57c8817837fd2b21675045311de42ffbfd363cb3208dedbc1b7ab2c834.jpg)  
Figure 7: Our GPTScore rubric evaluates predicted SVG code against ground-truth SVGs along three dimensions—Structure, Flow Accuracy, and Style Similarity—providing a unified measure of content fidelity and visual consistency.

## B.2.2 Flow Accuracy Evaluation

The Flow Accuracy axis evaluates whether the predicted SVG correctly captures the directed relationships between nodes, such as arrows in flowcharts or process diagrams. Concretely, this includes checking: (i) whether nodes and edges in the prediction can be aligned to those in the ground truth, (ii) whether the origins and destinations of edges match, and (iii) whether the arrow directions are correct and no spurious edges are introduced.

A score of 5 indicates that all nodes and directed edges match the ground truth, with no extra or missing connections and no direction errors. A score of 4 allows one or two minor mismatches that do not change the overall logical flow. Scores around 3 indicate several mismatches where the high-level logic is still partially preserved. Scores of 1–2 correspond to major flow errors where the core logic becomes unclear or largely incorrect. A score of 0 is assigned when there is no recognizable flow correspondence between the prediction and the ground truth.

When labels differ only by case, whitespace, or small edit distance, we treat them as close matches unless the change alters the semantic meaning. For unlabeled nodes, evaluators are instructed to use shape and relative position to tentatively align elements. If the graph connectivity is partially ambiguous (e.g., paths with unclear endpoints), the evaluator uses the available evidence conservatively and notes any uncertainty in the evaluation.

## B.2.3 Style Similarity Evaluation

The Style Similarity axis measures how well the predicted SVG preserves the visual appearance of the reference diagram, including: (i) color palettes, (ii) stroke widths, (iii) fonts and text sizes, (iv) arrowhead styles, and (v) corner radii of shapes.

![](images/4c61b70e36ba31c546e8ca252135b3c4f53f0fc21d0886c946b8d9f95f0ce84e.jpg)

A score of 5 corresponds to very close stylistic agreement across these properties such that the predicted diagram appears almost indistinguishable in style from the ground truth. A score of 4 indicates only small differences (e.g. near-by color hues, slightly different fonts, or stroke widths) that do not substantially change the overall visual impression. Scores around 3 correspond to multiple style changes while maintaining a generally similar look and feel. Scores of 1–2 reflect significant differences in palette, typography, arrowheads, or

Figure 8: Text-to-SVG instruction–prompt pair generation pipeline. We construct paired instruction prompts and SVG outputs that encode deterministic, fine-grained drawing specifications to supervise structured image reconstruction.

shape geometry, resulting in a noticeably different style. A score of 0 is reserved for predictions that essentially do not resemble the reference stylistic.

As with the structure axis, non-visible SVG elements and unused definitions are ignored to prevent them from affecting stylistic scoring. The focus is placed on properties that impact the rendered, human-visible appearance of the diagram.

## B.3 Creating Text-to-SVG Instruction Prompt

We employ the Qwen2.5-VL model for the conversion of structured images into descriptions suitable for reconstruction. These descriptions serve as deterministic, human-interpretable programs that delineate precise instructions on what elements to draw and the methodology of their depiction. The instructions provide a comprehensive enumeration of objects, textual labels, shapes, edges, and arrow directions, while concurrently capturing essential stylistic attributes such as stroke widths, colors, corner radii, fonts, spacing, and overarching layout patterns. This intermediary phase significantly mitigates ambiguity and ensures that downstream Scalable Vector Graphics (SVG) generation models receive comprehensive and precise visual specifications.

To ensure consistency and reproducibility, we implement a rigorously enforced prompting format specifically designed for Large Vision Language Models (LVLMs). The model is required to produce (1) a standardized introductory phrase, (2) a pure sequence of explicit drawing steps, and (3) a thorough documentation of structural and stylistic details devoid of extraneous commentary. This controlled structure averts over-generalization, excludes irrelevant text, and provides a practicable framework for precise SVG reconstruction. Consequently, the LVLM functions not merely as a captioning system but as an exact visual-to-instruction compiler.

Figure 8 shows detailed design principles, enforcement protocols, and the rationale underlying our instruction-prompting strategy. We demonstrate that converting structured images into explicit, step-by-step drawing instructions significantly enhances reconstruction fidelity, diminishes model hallucination, and enables the downstream LLM to produce SVG code indistinguishable in appearance from the original input image.

Figure 9 showcases a diverse set of Text-to-SVG instruction–image pairs, exemplifying the comprehensive scope and precision inherent in our dataset. The displayed examples encompass various structured-image domains, including an activity diagram characterized by partitions and a multistage control flow, a bioinformatics pipeline incorporating branching guide-tree operations, and a quantitative chart that represents socioeconomic stratification across income percentiles. These samples elucidate how our instruction prompts meticulously encode not only object types, shapes, and hierarchical arrangements, but also intricate stylistic attributes such as color palettes, stroke widths, arrow geometries, curved connectors, grid structures, and multi-line text formatting. The examples underscore the robustness of our prompting protocol in accurately capturing semantic structure alongside detailed visual style, thereby ensuring that subsequent models receive comprehensive and unambiguous specifications. This capability facilitates the reconstruction of SVGs that closely replicate the original diagrams.

## C Qualitative Samples

Figure 10 provides additional qualitative samples, underscoring the scope, intricacy, and robustness of our pipeline. These samples encompasse a comprehensive array of structured image types, including relational database schemas with nested attribute tables, transformer architecture diagrams, multi-stage system workflow charts, Docker deployment diagrams, statistical plots with annotated distributions, hierarchical tree structures, and block-based layout schematics. Our method adeptly captures not only the overarching semantic organization—such as table hierarchies, model pipelines, and system components—but also the nuanced geometric and stylistic elements, including alignment, spacing, connectors, arrow types, fonts, and color schemes. These qualitative outcomes demonstrate that our methodology effectively generalizes across diverse diagram styles, maintains structural fidelity across varying visual formats, and generates precise, editable SVG outputs even for densely populated or domain-specific technical graphics.

![](images/d042a37dad2b0d327c859f101e1a209c123617e55a63b9150d398e5aab922e8b.jpg)  
Figure 9: Example instruction prompts generated by the LVLM for guiding SVG reconstruction from structured images.

Figure 11 extends this analysis to the Text-to-SVG setting, where each model must render a structured diagram directly from the same natural-language specification. Despite having only 7B parameters, Back2Struct produces SVGs that are markedly closer to the ground truth than those of substantially larger systems: it faithfully realizes the specified components, their spatial arrangement, and the connectors and labels that encode their relationships. The baseline generations, by contrast, frequently drop or duplicate elements, misplace text, or break the intended alignment and arrow topology, yielding diagrams that are visually plausible yet structurally inconsistent with the prompt. This indicates that Back2Struct’s reward-driven fine-tuning transfers from image-conditioned reconstruction to purely text-conditioned generation, preserving fine-grained layout and stylistic fidelity even under the weaker grounding of a textual prompt.

![](images/44871730773d15a177c55eeb1383507d48eeee939a9e18f3610e75530ac0d773.jpg)

Figure 10: Qualitative results of our method. Gray denotes the ground-truth SVGs, while green indicates the predicted outputs.  
![](images/41c7c617878296dd776e02ad74d02f744c4fed4aa2480feb31c9dd6bc74bb9d7.jpg)  
Figure 11: A qualitative comparison on Text-to-SVG task among advanced LLMs and Back2Struct. Given the same prompt, Back2Struct, despite being only 7B, produces structured images that are more similar to the ground truth.

## D Limitations

• Since there is no open-source code generation large language model that accepts images as input, Back2Struct is built on a non-specialized code generation model, which limits its capability.

• In addition, to ensure stable training, the current model supports at most 8K output tokens, which restricts Back2Struct from generating more complex images.

We note that these limitations do not undermine the innovation and novelty of our work, and they can be overcome in future research.