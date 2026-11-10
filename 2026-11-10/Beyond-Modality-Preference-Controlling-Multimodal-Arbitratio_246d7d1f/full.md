# Beyond Modality Preference: Controlling Multimodal Arbitration under Modality Conflict in MLLMs

Zhaoying Pan, Shenyu Lu, Jun Hui Wong, Junyi Chai, Xiaoqian Wang<sup>†</sup> Elmore Family School of Electrical and Computer Engineering, Purdue University {pan433, lu876, wong531, chai28, joywang}@purdue.edu

## Abstract

Multimodal large language models (MLLMs) integrate information across modalities, but become unreliable when those modalities disagree. Under conflicting inputs, MLLMs may favor modalities that are not the most reliable for the task, thus motivating inference-time control of modality arbitration. We identify a small set of attention heads, termed arbitration heads, that causally influence whether predictions follow visual or textual evidence. Building on this, we introduce a lightweight, training-free intervention that biases these heads toward a designated trusted modality at inference time. The identified heads are stable across calibration set sizes and generalize across settings, and steering them enables effective bidirectional control of modality reliance across diverse benchmarks. Our findings identify an effective control mechanism to align MLLM predictions with task-specified modality priorities. The code is available at GitHub.

## 1. Introduction

Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in integrating visual and textual information for tasks ranging from visual question answering to complex multimodal reasoning [3, 19, 31, 33, 36]. A central premise behind these models is that information from different modalities can be effectively integrated to support more informed predictions. Yet, most multimodal evaluation implicitly assumes that the available modalities are mutually consistent or complementary [8, 16, 24]. In practice, this assumption does not always hold. The textual and visual inputs may provide contradictory evidence due to noisy annotations, outdated or erroneous descriptions, imperfect retrieval, or inconsistencies among heterogeneous information sources. Recent work has therefore begun to investigate modality conflict, where different modalities support incompatible answers to the same question [37–39]. Such conflicts expose a fundamental aspect of multimodal reasoning that remains largely hidden when modalities agree: which modality ultimately governs the model’s prediction?

Existing studies [4, 25, 27, 37–39] reveal that MLLMs do not treat the modalities neutrally under conflicts. Instead, they exhibit systematic modality preferences, tending to favor one source of evidence over another when the two disagree. Controlled conflict settings [38] have shown that this behavior is widespread across MLLMs and can vary substantially across models and tasks. As a result, modality preference can directly undermine model reliability. Misleading visual evidence can override an answer that a model correctly derives from text alone [10], while erroneous textual information can similarly dominate contradictory visual evidence [35]. Recent studies consequently identify modality conflict as an important source of multimodal hallucination and robustness failures [28, 32, 39].

More importantly, modality conflict is not merely an artificial stress test but a realistic failure mode in practical multimodal applications, where the authoritative modality is often specified by the task but not effectively utilized by the models. For example, medical imaging may serve as the primary evidence for diagnosis while patient metadata provides auxiliary context that should not override visual findings [7, 12]. Similarly, in autonomous driving, real-time visual observations should take precedence over risky navigation instructions when safety is at stake [26]. In these cases, the desired behavior is clear: predictions should follow the designated trusted modality when conflicts happen. However, an MLLM’s intrinsic modality preference may not align with this externally specified priority, potentially leading to consequential errors.

When the trusted modality is explicitly specified by the task or application, a seemingly straightforward solution is to instruct the model to prioritize the trusted one or simply remove the non-trusted modality. However, neither provides a satisfactory mechanism for controlling multimodal arbitration. Prompt-based instructions preserve both modalities but offer limited controllability. Removing a modality changes the multimodal input itself and bypasses rather than resolves cross-modal conflicts. It also discards potentially useful information in the more common case where the modalities are aligned and sacrifices flexibility when the designated trusted modality varies across tasks or contexts.

In this work, we investigate whether modality arbitration under modality conflict can be effectively controlled through inference-time intervention. We first identify a small set of attention heads, which we term arbitration heads, whose attention patterns are strongly associated with whether model predictions follow visual or textual evidence. We then selectively intervene on these heads at inference time, modulating their attention toward the designated trusted modality. This yields a lightweight, training-free mechanism for bidirectional modality control and we can steer predictions toward either visual or textual evidence according to the specified modality. Importantly, arbitration heads can be identified from only a small calibration set of 10 samples and transfer effectively across benchmarks, suggesting that they capture a reusable mechanism for modality arbitration rather than benchmark-specific correlations. To conclude, our contributions can be summarized as:

• We identify a compact set of arbitration heads associated with whether MLLMs follow visual or textual evidence under modality conflict.

• We propose a lightweight, training-free intervention on these heads that enables bidirectional control toward a designated trusted modality while preserving both modalities as input.

• We show that arbitration heads identified from a small calibration set generalize across benchmarks, suggesting a generalized mechanism for modality arbitration.

## 2. Related Work

## 2.1. Modality Conflict in MLLMs

Prior work has shown that multimodal models can exhibit imbalanced reliance across modalities. Such preferences may arise from unimodal shortcuts in the training data, enabling models to solve multimodal tasks by disproportionately relying on a single modality [5, 9, 15]. More recent studies have shifted attention to more concrete settings of modality conflicts in MLLMs to study the model behaviors. Zhang et al. [39] formalized conflicts in multimodal inputs and showed that such conflicts in questions and images can induce hallucinations. Cai et al. [4] demonstrated that irrelevant or misleading information from one modality can degrade reasoning that should rely on another, while CLASH [27] evaluates whether MLLMs can explicitly detect contradictions between visual and textual evidence. Beyond characterizing failures under conflicts, recent work also examines how models resolve conflicting evidence. Zhang et al. [37] systematically evaluate modality preference with controlled conflicts and show that preference directions are encoded in hidden states and can be steered. Another work [38] further decomposes the model behavior under modality conflicts into inherent modality preference and instance-specific relative reasoning uncertainty. Nguyen et al. [25] investigate the internal conflictresolution process and find that conflict detection and resolution exhibit distinct representational and attention patterns across layers. Together, these studies reveal that MLLMs exhibit systematic yet steerable behaviors under conflict.

The most relevant work is [37], which introduces MPS, a method that identifies a modality-preference direction in the model’s representation space using contrasting conflict examples and steers generation by adding the resulting vector to the model’s hidden states. In contrast, we identify and intervene on attention heads that causally influence conflict arbitration, enabling targeted modality control while providing mechanistic evidence for modality preference. Our intervention also yields stable generation without the additional tuning required by MPS.

## 2.2. Inference-time Interventions

Inference-time interventions aim to improve model performance without retraining. A major line of work in MLLMs modifies decoding to improve visual grounding, including layer-contrastive decoding [6], visual contrastive decoding [14], over-trust penalty and retrospectionallocation [11], etc. Other methods intervene on attention or representations, such as amplifying image-token attention [23], reallocating attention from output tokens to visual tokens [29], post-hoc hallucination correction [34], and steering modality preference in latent space [37]. A line of work in text-only LLMs also intervenes on a sparse set of attention heads, such as shifting activations along probed directions [20], pruning memory-preferring heads [13], reweighting attention to specified spans [40], or steering heads with superposed signals [18]. These methods collectively demonstrate that model behavior can be improved at inference time by modifying decoding, attention, or representations.

## 3. Method

## 3.1. Preliminaries and Problem Formulation

Multimodal autoregressive generation. We consider a multimodal large language model that takes an image $x ^ { v }$ a textual context $x ^ { t }$ , and a question $q$ as input. The model generates an answer $y = ( y _ { 1 } , \dotsc , y _ { N } )$ autoregressively:

$$
p _ { \theta } ( y \mid x ^ { v } , x ^ { t } , q ) = \prod _ { n = 1 } ^ { N } p _ { \theta } ( y _ { n } \mid x ^ { v } , x ^ { t } , q , y _ { < n } ) ,
$$

where $\theta$ denotes the model parameters. In common MLLM architectures, the image is first encoded into visual features and projected into the language-model embedding space. These visual tokens are then concatenated with textual tokens from the context and question. We denote the resulting input token sequence as $z = [ z ^ { v } ; z ^ { t } ; z ^ { q } ]$ , where $z ^ { v } , z ^ { t }$ , and $z ^ { q }$ correspond to visual tokens, context tokens, and question tokens, respectively.

Evidence aggregation through attention. Since the input sequence contains heterogeneous information sources, the model needs to route information among visual, textual, and question tokens before decoding an answer. For a transformer layer ℓ and attention head $h ,$ let $\boldsymbol { q } _ { i } ^ { ( \ell , h ) } , k _ { j } ^ { ( \ell , h ) }$ and $v _ { j } ^ { ( \ell , h ) } \in \mathbb { R } ^ { d _ { h } }$ denote the query, key, and value vectors of tokens i and $j ,$ where $d _ { h }$ is the head dimension. The unnormalized attention score, i.e. the logit assigned to token $j$ before normalization, is

$$
a _ { i , j } ^ { ( \ell , h ) } = \frac { \left. q _ { i } ^ { ( \ell , h ) } , k _ { j } ^ { ( \ell , h ) } \right. } { \sqrt { d _ { h } } } + M _ { i , j } ,
$$

where M is the causal attention mask that sets $M _ { i , j } = - \infty$ for $j > i$ . Normalizing these scores over all admissible source tokens and using them to mix the values yields the head output at position i,

$$
o _ { i } ^ { ( \ell , h ) } = \sum _ { j } \mathrm { s o f t m a x } \Big ( a _ { i , j } ^ { ( \ell , h ) } \Big ) v _ { j } ^ { ( \ell , h ) } ,
$$

where the softmax is taken over the index $j .$ The resulting weight softmax $\cdot ( a _ { i , j } ^ { ( \ell , h ) } )$ measures how much token i draws on token $j$ when forming $o _ { i } ^ { ( \ell , h ) }$ , so the score $a _ { i , j } ^ { ( \ell , h ) }$ can be read as the evidence that token $j$ contributes before it is put in competition with the other source tokens. Under this view, attention provides a natural interface for studying how question tokens gather evidence from the visual or textual components of the input.

Modality conflict. Let the multimodal input be $I \quad =$ $\{ x ^ { v } , x ^ { t } , q \}$ , where $x ^ { v }$ and $x ^ { t }$ denote the visual and textual inputs, respectively, and q denotes the question. Given the question, let $p _ { v } ( y )$ and $p _ { t } ( y )$ denote the model’s modalityspecific predictions based on either visual or textual evidence. We consider a modality conflict when the two modalities favor different answers:

$$
\underset { y } { \operatorname { a r g m a x } } p _ { v } ( y ) \neq \underset { y } { \operatorname { a r g m a x } } p _ { t } ( y ) .
$$

More generally, the degree of agreement between $p _ { v } ( \cdot )$ and $p _ { t } ( \cdot )$ provides a continuous signal of modality conflict, which we later exploit for adaptive steering in Section 3.5. In our setting, the modality to prioritize under conflict is externally specified by the task or application. Our goal is therefore not to determine which modality is more reliable, but to control model arbitration toward the designated modality while retaining both modalities as input.

## 3.2. Motivation

We examine how MLLMs behave when information across modalities is aligned or conflicting. As shown in Table 1, all four models perform well when given either modality independently or when the inputs provide aligned information. However, introducing conflicting text causes a substantial shift in model predictions. Similar behavior occurs across all evaluated models, suggesting that modality conflict induces a non-trivial competition between modalities.

<table><tr><td>Input</td><td>Metric</td><td>Qwen3</td><td>Qwen2.5</td><td>LLaVA-OV</td><td>InternVL3</td></tr><tr><td> $V$ </td><td> $A c c _ { v }$ </td><td>85.40</td><td>84.35</td><td>86.80</td><td>82.50</td></tr><tr><td> $T _ { c }$ </td><td> $A c c _ { t }$ </td><td>87.85</td><td>87.30</td><td>87.05</td><td>85.80</td></tr><tr><td> $V + T _ { a }$ </td><td> $A c c _ { v }$ </td><td>94.50</td><td>94.45</td><td>95.85</td><td>93.95</td></tr><tr><td rowspan="2"> $V + T _ { c }$ </td><td> $A c c _ { v }$ </td><td>47.25</td><td>32.70</td><td>33.45</td><td>30.10</td></tr><tr><td> $A c c _ { t }$ </td><td>42.20</td><td>55.25</td><td>54.55</td><td>56.75</td></tr></table>

Table 1. Model behavior under unimodal, aligned, and conflicting multimodal inputs. The image is denoted as $I ,$ the text that aligns or conflicts with the image content is denoted as $T _ { a }$ and $T _ { c } ,$ respectively. Acc<sub>v</sub> and Acc<sub>t</sub> measure the performance against image-/text-consistent answers.

We further validate that this modality preference depends on their relative reliability [38]. Table 2 progressively degrades the image through Gaussian blurring. While imageonly accuracy decreases gradually, the presence of conflicting text causes a much sharper transition in the preferred modality from image to text. These observations suggest that modality preference is neither fixed nor solely determined by the presence of conflicting information, but dynamically changes with the strength of evidence provided by each modality. This observation highlights the reliability risks posed by modality conflicts and motivates the need to control modality preference in MLLMs.

<table><tr><td>Input</td><td>Metric</td><td>Original</td><td>Blur-1</td><td>Blur-2</td><td>Blur-3</td></tr><tr><td>V</td><td> $A c c _ { v }$ </td><td>85.40</td><td>84.10</td><td>77.35</td><td>65.10</td></tr><tr><td rowspan="2"> $V + T _ { c }$ </td><td> $A c c _ { v }$ </td><td>47.25</td><td>40.00</td><td>14.10</td><td>3.20</td></tr><tr><td> $A c c _ { t }$ </td><td>42.20</td><td>49.25</td><td>73.95</td><td>84.65</td></tr></table>

Table 2. Effect of degrading the image modality on Qwen3. Image-only accuracy decreases gradually with increasing blur. Under conflicting multimodal input, however, weakening the image causes the model to shift sharply from image-consistent to text-consistent predictions.

## 3.3. Arbitration Head Localization

We first characterize the modality preference under conflicting multimodal inputs, where the visual and textual evidence support different answers. For each conflicting input $I _ { C } = \{ x ^ { v } , x ^ { t } , q \}$ , we define the answer logit margin as

$$
\Delta ( I _ { C } ) = \log p ( y ^ { v } \mid I _ { C } ) - \log p ( y ^ { t } \mid I _ { C } ) ,\tag{1}
$$

![](images/be575400eaa739028aa2f234059720fceead7e29a9f1757a81eeac14f8988095.jpg)  
Figure 1. Overview of our method. (a) Arbitration head localization: we causally identify attention heads that causally influence modality preference by patching head activations from image-only and text-only inputs into conflicting multimodal inputs and measuring their effects on the answer logit margin. (b) Inference-time intervention: we select the most influential arbitration heads and steer predictions toward a designated modality by adjusting their attention to visual and textual tokens.

where $y ^ { v }$ and $y ^ { t }$ denote the candidate answers supported by the visual and textual modalities, respectively. A positive margin, $\Delta ( I _ { C } ) > 0$ , indicates that the model assigns a higher probability to the vision-supported answer under conflict input $I _ { C }$ , whereas a negative margin, $\Delta ( I _ { C } ) < 0$ indicates a preference for the text-supported answer.

We then perform activation patching to localize attention heads that causally influence modality preference. For each conflicting example, we construct three corresponding inputs: the original conflict input $I _ { C } ,$ a vision-only input $I _ { V }$ , and a text-only input $I _ { T }$ . For each head h at layer $\ell ,$ we record its output $\bar { o _ { i } ^ { ( \ell , h ) } }$ at the final prompt position, immediately before answer generation, at the input to the attention output projection during the vision-only and text-only forward passes, denoting the resulting activations by $o _ { V } ^ { ( \ell , \bar { h } ) }$ and $o _ { T } ^ { ( \ell , h ) }$ . We subsequently patch the conflict input with the corresponding activation from either unimodal run.

Specifically, the causal effect of patching the visionderived activation into the conflict run is defined as

$$
C _ { V } ^ { ( \ell , h ) } ( I _ { C } ) = \Delta \Big ( I _ { C } ; \mathrm { p a t c h } \Big ( o _ { V } ^ { ( \ell , h ) } \Big ) \Big ) - \Delta ( I _ { C } ) ,\tag{2}
$$

while the effect of patching the text-derived activation is

$$
C _ { T } ^ { ( \ell , h ) } ( I _ { C } ) = \Delta ( I _ { C } ) - \Delta \Big ( I _ { C } ; \mathrm { p a t c h } \Big ( o _ { T } ^ { ( \ell , h ) } \Big ) \Big ) .\tag{3}
$$

These quantities measure the change in the answer logit margin caused by replacing the conflict-run head output

with an activation obtained from a modality-specific input. We average the effect over a calibration set D of conflicting examples, and assign each head an influence score by

$$
C ^ { ( \ell , h ) } = \mathbb { E } _ { I _ { c } \sim \mathcal { D } } \left[ \operatorname* { m i n } _ { m \in \{ V , T \} } \mathrm { R e L U } \left( C _ { m } ^ { ( \ell , h ) } \right) \right] .\tag{4}
$$

This score prioritizes heads exhibiting positive causal effects under both modality-specific patching conditions while suppressing heads whose effects are weak or inconsistent across conditions.

We visualize $C ^ { ( \ell , h ) }$ across layers and attention heads using a heatmap and select the highest-scoring heads as the arbitration heads for subsequent intervention experiments. An example of the visualization with Qwen3-VL-8B is provided in Figure 2. We also examine the stability of headwise effects with respect to the number of calibration samples. The high-effect heads consistently concentrate at similar layer–head locations as n increases. Notably, this spatial pattern is already apparent with only a few samples and remains stable with larger calibration sets.

## 3.4. Inference-Time Intervention

Based on the heads identified through activation patching, we perform targeted inference-time interventions to test whether manipulating their attention patterns can systematically alter the model’s reliance on a particular modality. Specifically, the interventions are applied directly to the attention computation of the selected heads during inference.

![](images/1a77d4df805b4126f396ad92328c662edd69f56c07e25783ab536b20ba7ce9ba.jpg)  
Figure 2. Stability of head-wise effects across different sample sizes. Normalized head-wise effects $C ^ { ( \ell , h ) }$ for Qwen3-VL-8B, computed using n samples. Despite varying n, high-effect heads consistently concentrate in similar layers and head positions, indicating that the identified pattern is stable even with a small number of samples.

![](images/d9d5925b33985fa6d623e1c3526e18e5c63dd6bc7d13ec58e9ee3fd4ea7f0768.jpg)  
Figure 3. Modality conflict detection for adaptive steering. Top-50 union-logit cosine similarities for samples paired with conflicting or aligned context, where lower similarity indicates stronger modality conflict. The dashed line marks the threshold, selected to maximize conflict-detection accuracy on the calibration set.

We take the top-k heads by influence score $C ^ { ( \ell , h ) }$ as the arbitration head set H, and weight each of them by its normalized influence score

$$
\widehat { c } ^ { ( \ell , h ) } = \frac { C ^ { ( \ell , h ) } } { \operatorname* { m a x } _ { ( \ell ^ { \prime } , h ^ { \prime } ) \in \mathcal { H } } C ^ { ( \ell ^ { \prime } , h ^ { \prime } ) } + \epsilon } ,\tag{5}
$$

where $\epsilon > 0$ is a small constant, so that ${ \widehat { c } } ^ { ( \ell , h ) } \in ( 0 , 1 )$ . We then bias these heads toward a target modality by adding a positive offset to the pre-softmax attention scores of tokens from that modality and subtracting the same offset from tokens of the competing modality. Recall that $a _ { i , j } ^ { ( \ell , h ) }$ denotes the attention score from query position i to key position $j$ before normalization, and let $\mathcal { T } _ { m }$ and $\mathcal { T } _ { \bar { m } }$ denote the sets of token positions associated with the target modality m and the competing modality m¯ , respectively. For each head $( \ell , h ) \in \mathcal { H }$ , the intervened scores are defined as

$$
\widetilde { \boldsymbol { a } } _ { i , j } ^ { ( \ell , h ) } = \boldsymbol { a } _ { i , j } ^ { ( \ell , h ) } + \alpha \boldsymbol { \widehat { c } } ^ { ( \ell , h ) } \left( \mathbb { I } [ \boldsymbol { j } \in \mathcal { T } _ { m } ] - \mathbb { I } [ \boldsymbol { j } \in \mathcal { T } _ { \bar { m } } ] \right) ,\tag{6}
$$

where $\alpha > 0$ controls the overall intervention strength, and scaling by $\widehat { c } ^ { ( \ell , h ) }$ concentrates the perturbation on the heads with the strongest effect. The I[·] denotes the indicator function. The intervention is applied at every query position in the selected heads during both prompt prefill and autoregressive decoding, with generated positions appended as neutral keys during decoding. The head output is then recomputed as $\begin{array} { r } { \widetilde { o } _ { i } ^ { ( \ell , h ) } = \sum _ { j } } \end{array}$ softmax $\left( \widetilde { a } _ { i , j } ^ { ( \ell , h ) } \right) v _ { j } ^ { ( \ell , h ) }$ . As the offsets act inside the softmax, this intervention increases the relative attention allocated to the target modality while preserving access to information from the competing modality. Section 3.5 evaluates whether adaptive steering can retain the benefits of multimodal inference when the inputs are aligned instead of being conflicted.

The intervention effect is empirically quantified using the change in performance by comparing the generated responses against the image- and text-supported answers. The setup connects the diagnostic and intervention stages: activation patching identifies attention heads associated with modality-dependent answer selection, while inference-time intervention evaluates whether selectively redirecting the attention of these heads is sufficient to steer the model’s behavior toward a desired modality.

## 3.5. Extension: Adaptive Steering

To avoid unnecessary intervention when modalities are already consistent, we introduce adaptive steering, which dynamically adjusts the intervention strength according to the detected degree of modality conflict. We augment an existing modality-conflict benchmark, $\mathbf { M C ^ { 2 } }$ , whose original data consists of images and corresponding conflicting text. We generate aligned text by rewriting the conflicting text with image-supported answers, so that each image yields both an aligned and a conflicting multimodal input.

To detect conflicts, we compare the model’s image-only and text-only logit distributions using the union of their top-50 token logits, $\ell _ { 2 } \cdot$ -normalize the resulting vectors, and compute their cosine similarity s. As shown in Figure 3, conflicting and aligned samples exhibit clearly different similarity distributions, supporting the use of this signal for adaptive conflict detection. A threshold τ , calibrated on 200 samples, separates conflicting from aligned inputs. During inference, the steering strength for each sample is determined as

$$
\alpha = \alpha _ { \mathrm { m a x } } \sigma \biggl ( \frac { \tau - s } { \lambda } \biggr ) ,\tag{7}
$$

where $\alpha _ { \mathrm { m a x } }$ is the maximum allowed strength, $\sigma ( \cdot )$ is the logistic function, and λ controls how sharply the steering strength changes around the conflict threshold. This ensures that stronger intervention is applied when the two modalities disagree. The resulting strength is then substituted into the intervention above.

## 4. Experiments

## 4.1. Setup

Benchmarks and Metrics We evaluate our method on four modality-conflict benchmarks, including MMCD [25], $\mathbf { M C ^ { 2 } }$ [37], CLASH [27], and CorrelationQA [10], and one visual hallucination benchmark PhD [22]. These benchmarks cover diverse scenarios in which different modalities may lead to competing predictions.

MMCD, $\mathbf { M C ^ { 2 } }$ , and CLASH provide annotations for the answers supported by each modality independently, enabling us to explicitly evaluate the model’s preference between conflicting modalities. We consider two evaluation settings: Image-Primary and Text-Primary. For methods that support bidirectional modality steering, we steer generation toward the designated primary modality in each setting. We then report accuracy compared against image-/text-supported answers under each setting. Consequently, successful image-primary steering should increase image accuracy while reducing text accuracy, whereas successful text-primary steering should exhibit the opposite behavior. This bidirectional evaluation directly measures whether a method can controllably shift model predictions toward a designated modality rather than merely improving overall task accuracy. CorrelationQA provides ground-truth answers based on textual information only. We therefore supplement it with corresponding image-supported answers to enable the same bidirectional evaluation. Details of this annotation procedure are provided in Appendix B.4.

We further evaluate on two subsets of PhD [22], a benchmark designed to assess visual hallucination in the presence of textual context. Following prior work [37], we sample 1000 samples from the subsets PhD-sec and PhD-icc, which contain irrelevant or conflicting textual information, respectively. PhD complements the modality-conflict benchmarks by evaluating whether modality steering can strengthen visual grounding to mitigate hallucinations in the presence of misleading textual context. For PhD, we evaluate both settings using image accuracy, measuring how effectively each method preserves or suppresses reliance on visual evidence.

Models and Baselines We conduct experiments on four open-source MLLMs to examine whether our modality steering generalizes across different multimodal architectures. Specifically, we use Qwen2.5-VL-7B [2], Qwen3- VL-8B [1], LLaVA-OneVision-7B [17] and InternVL3- 8B [41]. We compare our method against standard inference strategies and various approaches. In the subsequent sections, Base denotes standard model inference without additional intervention, while CoT [30] employs chain-of-thought prompting to encourage explicit reasoning before prediction. We additionally include VCD (Visual Contrastive Decoding) [14], a decoding-based method for increasing vision grounding and mitigating hallucination. These methods do not explicitly support controlling the direction of modality preference and therefore produce a single set of predictions regardless of whether the desired target is the image or text modality.

We further compare against two baselines capable of bidirectional modality steering. CAP (Conflict-Aware Prompting) explicitly instructs the model to answer according to either the image or the textual context, providing a prompting-based baseline for modality control. MPS (Modality Preference Steering), introduced in $\mathbf { M C ^ { 2 } }$ [37], provides a baseline for steering predictions toward a specified modality by computing a steering vector from a set of samples. For each MLLM, our method uses 10 samples from $\mathbf { M C ^ { 2 } }$ to localize top-30 arbitration heads among over 1000 heads, and uses α = 30 as intervention strength for the main experiments where conflicts are explicitly presented. We additionally use the augmented $\mathbf { M C ^ { 2 } }$ to demonstrate the adaptive steering. For CAP, MPS, and our method, we evaluate the image- and text-directed variants under the imageprimary and text-primary settings, respectively. More experiment details are provided in Appendix B.

## 4.2. Results on Modality Conflict Benchmarks

Table 3 presents the results on four modality-conflict benchmarks across four MLLMs. Overall, our method provides strong and generally consistent bidirectional control of modality preference. Under image-primary steering, it substantially increases image accuracy while suppressing text accuracy. In contrast, under text-primary steering, the preference is effectively reversed. This behavior generalizes across different models and benchmarks, demonstrating that our method can reliably steer predictions toward the designated modality.

Compared with single-direction methods, including Base, CoT, and VCD, our method enables explicit control with superior accuracy toward the naturally preferred modality. Compared with CAP and MPS, our method generally achieves stronger target-modality preference. Notably, the arbitration heads used for intervention are identified using only 10 calibration samples from $\mathbf { M C ^ { 2 } }$ . Despite this limited calibration, the same identified heads generalize effectively to MMCD, CLASH, and CorrelationQA without benchmark-specific identification. The consistently strong results across benchmarks indicate that the identified heads provide transferable intervention targets rather than datasetspecific control points.

<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="4">MMCD</td><td colspan="4">MC2</td><td colspan="4">CLASH</td><td colspan="4">CorrelationQA</td></tr><tr><td colspan="2">Image-P</td><td colspan="2">Text-P</td><td colspan="2">Image-P</td><td colspan="2">Text-P</td><td colspan="2">Image-P</td><td colspan="2">Text-P</td><td colspan="2">Image-P</td><td colspan="2">Text-P</td></tr><tr><td>Accv ↑</td><td>Acct ↓</td><td>Accv ↓</td><td>Acct ↑</td><td>Accv ↑</td><td>Acct ↓</td><td>Accv ↓</td><td>Acct ↑</td><td>Accv ↑</td><td>Acct ↓</td><td>Accv ↓</td><td>Acct ↑</td><td>Accv ↑</td><td>Acct ↓</td><td>Accv ↓</td><td>Acct ↑</td></tr><tr><td rowspan="6">Qwen3-VL-8B</td><td>Base</td><td>93.07</td><td>6.92</td><td></td><td></td><td>47.25</td><td>42.20</td><td></td><td></td><td>66.72</td><td>28.16</td><td></td><td></td><td>20.11</td><td>51.19</td><td></td><td></td></tr><tr><td>CoT</td><td>37.88</td><td>60.48</td><td></td><td></td><td>41.65</td><td>50.60</td><td></td><td></td><td>63.30</td><td>29.17</td><td></td><td></td><td>18.12</td><td>52.26</td><td></td><td></td></tr><tr><td>VCD</td><td>95.30</td><td>4.70</td><td></td><td></td><td>53.45</td><td>36.30</td><td></td><td></td><td>70.36</td><td>22.81</td><td></td><td></td><td>21.99</td><td>47.04</td><td></td><td></td></tr><tr><td>CAP</td><td>96.34</td><td>3.66</td><td>29.50</td><td>70.50</td><td>61.10</td><td>28.45</td><td>24.90</td><td>64.60</td><td>70.83</td><td>23.35</td><td>51.28</td><td>45.23</td><td>32.51</td><td>35.04</td><td>19.37</td><td>50.11</td></tr><tr><td>MPS</td><td>99.18</td><td>0.50</td><td>47.80</td><td>52.20</td><td>59.95</td><td>27.60</td><td>27.35</td><td>61.05</td><td>74.79</td><td>16.68</td><td>31.65</td><td>65.94</td><td>22.26</td><td>45.96</td><td>13.43</td><td>52.15</td></tr><tr><td>Ours</td><td>100.00</td><td>0.00</td><td>0.00</td><td>100.00</td><td>79.50</td><td>5.25</td><td>1.25</td><td>86.75</td><td>81.23</td><td>7.06</td><td>1.55</td><td>96.97</td><td>46.88</td><td>9.91</td><td>4.55</td><td>58.86</td></tr><tr><td rowspan="6">Qwen2.5-VL-7B</td><td>Base</td><td>6.88</td><td>93.12</td><td></td><td>I</td><td>32.70</td><td>55.25</td><td></td><td></td><td>51.20</td><td>44.53</td><td></td><td>一</td><td>21.70</td><td>46.73</td><td></td><td></td></tr><tr><td>CoT</td><td>79.14</td><td>20.61</td><td></td><td></td><td>49.25</td><td>34.35</td><td></td><td></td><td>66.41</td><td>23.12</td><td></td><td>一</td><td>23.03</td><td>44.58</td><td>=</td><td>一</td></tr><tr><td>VCD</td><td>37.05</td><td>62.95</td><td></td><td></td><td>45.35</td><td>40.25</td><td></td><td></td><td>67.11</td><td>24.90</td><td></td><td></td><td>26.57</td><td>36.34</td><td></td><td></td></tr><tr><td>CAP</td><td>44.28</td><td>55.73</td><td>0.66</td><td>99.34</td><td>48.75</td><td>39.50</td><td>20.20</td><td>68.70</td><td>55.39</td><td>39.95</td><td>29.79</td><td>68.58</td><td>29.87</td><td>34.22</td><td>22.69</td><td>44.60</td></tr><tr><td>MPS</td><td>94.14</td><td>1.68</td><td>0.00</td><td>100.00</td><td>51.75</td><td>10.60</td><td>3.30</td><td>79.45</td><td>69.28</td><td>13.50</td><td>2.87</td><td>94.88</td><td>24.93</td><td>19.64</td><td>3.00</td><td>27.42</td></tr><tr><td>Ours</td><td>100.00</td><td>0.00</td><td>0.00</td><td>100.00</td><td>59.80</td><td>15.95</td><td>4.70</td><td>80.60</td><td>77.81</td><td>9.78</td><td>1.16</td><td>98.29</td><td>39.31</td><td>18.32</td><td>7.69</td><td>52.60</td></tr><tr><td rowspan="6">LLaVA-OV-7B</td><td>Base</td><td>97.64</td><td>2.36</td><td>一</td><td>一</td><td>33.45</td><td>54.55</td><td>一</td><td>一</td><td>45.69</td><td>50.04</td><td></td><td>一</td><td>17.82</td><td>44.69</td><td>一</td><td>一</td></tr><tr><td>CoT</td><td>97.30</td><td>2.70</td><td>一</td><td>I</td><td>33.70</td><td>54.05</td><td>1</td><td></td><td>44.45</td><td>51.51</td><td>一</td><td>一</td><td>19.12</td><td>43.05</td><td>=</td><td>-</td></tr><tr><td>VCD</td><td>99.88</td><td>0.13</td><td></td><td></td><td>48.05</td><td>38.95</td><td>1</td><td>1</td><td>67.11</td><td>25.76</td><td></td><td></td><td>22.35</td><td>33.72</td><td>1</td><td></td></tr><tr><td>CAP</td><td>94.14</td><td>5.86</td><td>85.59</td><td>14.41</td><td>38.80</td><td>49.75</td><td>30.65</td><td>57.75</td><td>52.21</td><td>43.52</td><td>39.72</td><td>57.01</td><td>22.00</td><td>34.77</td><td>19.86</td><td>39.78</td></tr><tr><td>MPS</td><td>99.77</td><td>0.05</td><td>10.68</td><td>89.32</td><td>43.65</td><td>25.35</td><td>5.80</td><td>70.20</td><td>63.62</td><td>22.27</td><td>8.07</td><td>83.40</td><td>14.93</td><td>18.10</td><td>3.23</td><td>16.82</td></tr><tr><td>Ours</td><td>99.95</td><td>0.05</td><td>0.00</td><td>100.00</td><td>60.95</td><td>25.70</td><td>3.35</td><td>84.50</td><td>72.07</td><td>19.08</td><td>5.43</td><td>93.10</td><td>27.18</td><td>21.79</td><td>13.70</td><td>48.68</td></tr><tr><td rowspan="6">InternVL3-8B</td><td>Base</td><td>36.84</td><td>62.63</td><td>=</td><td>一</td><td>30.10</td><td>56.75</td><td>一</td><td></td><td>62.14</td><td>32.74</td><td>一</td><td>=</td><td>17.78</td><td>48.70</td><td>一</td><td>-</td></tr><tr><td>CoT</td><td>36.05</td><td>48.36</td><td></td><td></td><td>37.40</td><td>51.50</td><td>一</td><td></td><td>57.64</td><td>33.75</td><td></td><td>一</td><td>22.00</td><td>46.37</td><td>一</td><td>-</td></tr><tr><td>VCD</td><td>61.98</td><td>37.07</td><td></td><td>71.29</td><td>42.35 41.90</td><td>43.50 45.60</td><td>17.70</td><td>68.40</td><td>72.07 62.06</td><td>20.79 32.89</td><td>39.10</td><td>58.73</td><td>20.54 29.46</td><td>40.29 28.32</td><td>18.34</td><td>35.32</td></tr><tr><td>CAP MPS</td><td>63.16 77.57</td><td>36.84 18.02</td><td>28.71 0.13</td></table>

Table 3. Comparison across various modality-conflict benchmarks. We report image and text accuracy under both Image-Primary (Image-P) and Text-Primary (Text-P) settings. Under Image-P, higher image accuracy and lower text accuracy are preferred. Under Text-P, lower image accuracy and higher text accuracy are preferred. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td></td><td>Vanilla</td><td>CoT</td><td>VCD</td><td>CAP</td><td>MPS</td><td>Ours</td></tr><tr><td>PhD-sec Image-P Accv ↑</td><td>69.50</td><td>75.60</td><td>72.60</td><td>69.90</td><td>71.70</td><td>73.90</td></tr><tr><td>Text-P Accv ↓</td><td>一</td><td>一</td><td>一</td><td>62.00</td><td>64.50</td><td>58.90</td></tr><tr><td>PhD-icc</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Image-P Accv ↑</td><td>64.40</td><td>69.10</td><td>69.60</td><td>59.20</td><td>65.70</td><td>70.20</td></tr><tr><td>Text-P Accv ↓</td><td>1</td><td>一</td><td></td><td>49.50</td><td>60.10</td><td>58.00</td></tr></table>

Table 4. Results on the PhD visual hallucination benchmark. We evaluate PhD-sec and PhD-icc, which introduce unrelated or conflicting text. Higher image accuracy is better under Image-Primary. Lower is better under Text-Primary.

## 4.3. Results on Visual Hallucination Benchmark

We further evaluate Qwen3-VL-8B on the PhD visual hallucination benchmark, with results shown in Table 4. Under image-primary steering, our method achieves the highest image accuracy on PhD-icc and the best performance among bidirectional methods on PhD-sec, remaining competitive with the best single-direction methods. Conversely, text-primary steering reduces image accuracy on both subsets, demonstrating effective suppression of visual preference. These results show that our bidirectional modality control applies to visual hallucination settings and effectively steers the answers.

## 4.4. Adaptive Steering under Sparse Conflict

We evaluate whether adaptive steering can preserve the benefits of multimodal inference while resolving conflicts when necessary. We vary the proportion of conflicting and aligned samples with our augmented $\mathbf { M C ^ { 2 } }$ and Qwen3-VL-8B to examine how different methods behave as modality agreement changes, especially when the conflicted samples are sparse. When modalities are aligned, information from both modalities can be complementary, allowing multimodal inference to outperform relying on one modality alone.

![](images/21b07a4e69d478b533b103b7a658ad81b8aa54d6562e6877fe4f8b696b3e3a07.jpg)

![](images/4a3dfda053b901a814e20805ca09da658f8ed46ab81ef6e969976984f7cb3256.jpg)  
Figure 4. Preserving multimodal benefits under sparse modality conflict. The left and right panels evaluate MPS and our method steered towards image and text, respectively. Dotted lines denote unimodal performance. Although unimodal inference avoids modality conflict, it cannot exploit complementary information when the modalities are aligned. Adaptive steering retains this benefit while remaining robust to conflict.

As shown in Figure 4, when conflicts are sparse, Base and adaptive steering outperform the unimodal baseline, confirming that complementary cross-modal evidence is worth exploiting. As the fraction of conflicting samples increases, Base shows a significant performance drop due to the inability to resolve competing evidence. Adaptive steering retains most of the Base advantage on aligned samples while strengthening the designated modality once conflict is detected. The comparison isolates what adaptive steering provides over discarding a modality, that conflict robustness that does not require giving up the complementary information available when the modalities agree.

![](images/1775cf96d6416c39d51eaeb9fc77f4da971e5e72f962b915dbdb0987f154e5bc.jpg)

![](images/f21f6372ddb65144d0309d3dd1d270e19b1ff2a93a12ae38240fdc0034f1282b.jpg)

Figure 5. Effect of steering strength α. Increasing α progressively shifts predictions toward the target modality, with performance saturating at moderate steering strengths. $\alpha = 0$ denotes the unmodified baseline, and α = ∞ corresponds to hard masking of the competing modality.  
![](images/bbd7bb70cbc19136023d1959debfd29857b5f61dd73cf532da68587dfa5fb077.jpg)

![](images/18da462a822f51dd2442c5ca77c0c83e03b2fa163458ff621c0cef395601a927.jpg)

Figure 6. Effect of the number of arbitration heads (top-k). Steering an increasing number of top-ranked heads progressively shifts predictions toward the target modality, with the effect saturating at moderate values. The top-k of 0 denotes the unmodified baseline.
<table><tr><td rowspan="2">Intervention</td><td colspan="2">Image-Primary</td><td colspan="2">Text-Primary</td></tr><tr><td> $A c c _ { v }$  ↑</td><td> $A c c _ { t } \downarrow$ </td><td> $A c c _ { v } \downarrow$ </td><td> $A c c _ { t } \uparrow$ </td></tr><tr><td>None</td><td>47.25</td><td>42.20</td><td></td><td></td></tr><tr><td>Random</td><td>47.50</td><td>42.30</td><td>44.70</td><td>44.55</td></tr><tr><td>Layer-matched</td><td>54.75</td><td>34.90</td><td>38.50</td><td>50.55</td></tr><tr><td>Reversed</td><td>47.20</td><td>42.45</td><td>47.20</td><td>42.40</td></tr><tr><td>All Heads</td><td>77.10</td><td>6.10</td><td>1.05</td><td>86.70</td></tr><tr><td>Ours w/o suppressing</td><td>79.25</td><td>5.90</td><td>1.35</td><td>86.75</td></tr><tr><td>Ours</td><td>79.50</td><td>5.25</td><td>1.25</td><td>86.75</td></tr></table>

Table 5. Ablation study. “Random” randomly selects 30 heads; “Layer-matched” randomly selects each head from the same layers as ours; “Reversed” selects the 30 heads in reverse ranking; “All Heads” intervenes on all attention heads. “Ours w/o suppressing” only increases attention to the trusted modality without suppressing the other modality.

## 4.5. Ablation Studies

We conduct ablation studies to examine the effects of steering strength, the number of selected arbitration heads, head selection, and modality suppression. As shown in Figure 5, increasing the steering strength α progressively shifts predictions toward the designated modality, with performance saturating at moderate values. We therefore use $\alpha = 3 0$ by default. Similarly, Figure 6 shows that increasing the number of top-ranked arbitration heads in a moderate range consistently strengthens modality control, with performance becoming stable around $k = 2 0 { - } 3 0$ . We select $k = 3 0$ for all main experiments.

Table 5 further demonstrates that which heads are selected is critical. We cover a few head selection criteria: Random chooses 30 heads from all heads, while layermatched chooses each head from the same layers as the heads selected by ours. Reversed selects 30 heads by filtering the lowest influence scores and All Heads applies the intervention on all heads. Random, layer-matched random, and reverse-ranked selection all yield substantially weaker control than our identified arbitration heads. Notably, intervening on all heads (over 1000) does not further improve performance, despite using many more heads, confirming that effective steering relies on selectively targeting arbitration heads rather than broadly modifying attention. Finally, removing suppression causes a small performance drop, indicating that augmenting the trusted modality accounts for most of the steering effect, while suppressing the competing modality provides a modest additional improvement.

## 5. Conclusion

We study how multimodal arbitration in MLLMs can be controlled under modality conflicts and introduce a training-free approach for controlling which modality governs model predictions. By identifying a small set of arbitration heads associated with modality arbitration and intervening on their attention, our method can steer answer generation toward a trusted modality. Extensive experiments across multiple MLLMs and modality-conflict benchmarks demonstrate consistent and effective bidirectional modality control. The arbitration heads identified from only a small calibration set generalize effectively across benchmarks. Overall, our findings suggest that modality arbitration can be effectively controlled through targeted intervention on a compact set of identified attention heads.

Limitations and Future Work Our evaluation uses a simplified modality-conflict setup in which the modality to prioritize is externally specified, making routing to unimodal inference a potentially strong baseline. In realistic settings, however, conflicts may be partial, with the correct answer requiring the integration of distinct evidence from multiple modalities. We therefore focus on controlling modality arbitration while retaining both input sources. Benchmarking when both modalities provide useful yet partially conflicting evidence, for which unimodal routing is insufficient, remains an important direction for future work. More discussion is detailed in Appendix E.

Acknowledgments This work was partially supported by NSF IIS #2146091, IIS #2345235, and USDA #2023- 67021-41368.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6

[3] Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, and Rita Cucchiara. The revolution of multimodal large language models: A survey. Findings of the association for computational linguistics: ACL 2024, pages 13590– 13618, 2024. 1

[4] Rui Cai, Bangzheng Li, Xiaofei Wen, Muhao Chen, and Zhe Zhao. Diagnosing and mitigating modality interference in multimodal large language models. arXiv preprint arXiv:2505.19616, 2025. 1, 2

[5] Meiqi Chen, Yixin Cao, Yan Zhang, and Chaochao Lu. Quantifying and mitigating unimodal biases in multimodal large language models: A causal perspective. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 16449–16469, 2024. 2

[6] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations, pages 54158–54183, 2024. 2

[7] Can Cui, Haichun Yang, Yaohong Wang, Shilin Zhao, Zuhayr Asad, Lori A Coburn, Keith T Wilson, Bennett A Landman, and Yuankai Huo. Deep multimodal fusion of image and non-image data in disease diagnosis and prognosis: a review. Progress in Biomedical Engineering, 5(2):022001, 2023. 1

[8] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026. 1

[9] Yangyang Guo, Liqiang Nie, Harry Cheng, Zhiyong Cheng, Mohan Kankanhalli, and Alberto Del Bimbo. On modality bias recognition and reduction. ACM Transactions on Multimedia Computing, Communications and Applications, 19 (3):1–22, 2023. 2

[10] Tianyang Han, Qing Lian, Rui Pan, Renjie Pi, Jipeng Zhang, Shizhe Diao, Yong Lin, and Tong Zhang. The instinctive bias: Spurious images lead to illusion in mllms. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16163–16177, 2024. 1, 6, 11

[11] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. Opera: Alleviating hallucination in multimodal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pages 13418–13427, 2024. 2

[12] Shih-Cheng Huang, Anuj Pareek, Saeed Seyyedi, Imon Banerjee, and Matthew P Lungren. Fusion of medical imaging and electronic health records using deep learning: a sys tematic review and implementation guidelines. NPJ digital medicine, 3(1):136, 2020. 1

[13] Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. In Findings of the Association for Computa tional Linguistics: ACL 2024, pages 1193–1215, 2024. 2

[14] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882, 2024. 2, 6, 12, 13

[15] Sicong Leng, Yun Xing, Zesen Cheng, Yang Zhou, Hang Zhang, Xin Li, Deli Zhao, Shijian Lu, Chunyan Miao, and Lidong Bing. The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. Advances in Neural Information Processing Systems, 38, 2026. 2

[16] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13299–13308. IEEE, 2024. 1

[17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6

[18] Gaotang Li, Yuzhong Chen, and Hanghang Tong. Taming knowledge conflicts in language models. In Forty-second International Conference on Machine Learning, 2025. 2

[19] Jian Li, Weiheng Lu, Hao Fei, Meng Luo, Ming Dai, Min Xia, Yizhang Jin, Zhenye Gan, Ding Qi, Chaoyou Fu, et al. A survey on benchmarks of multimodal large language models. arXiv preprint arXiv:2408.08632, 2024. 1

[20] Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister,´ and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in neural information processing systems, 36:41451–41530, 2023. 2

[21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence´ Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 11

[22] Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. Phd: A chatgpt-prompted visual hallucination evaluation dataset. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19857–19866. IEEE, 2025. 6, 11

[23] Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In European Conference on Computer Vision, pages 125–140. Springer, 2024. 2

[24] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 1

[25] Trang Nguyen, Jackson Michaels, Madalina Fiterau, and David Jensen. Challenges in understanding modality conflict in vision-language models. arXiv preprint arXiv:2509.02805, 2025. 1, 2, 6, 11

[26] Brian Paden, Michal C<sup>ˇ</sup> ap, Sze Zheng Yong, Dmitry Yershov, ´ and Emilio Frazzoli. A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on intelligent vehicles, 1(1):33–55, 2016. 1

[27] Teodora Popordanoska, Jiameng Li, and Matthew B Blaschko. Clash: A benchmark for cross-modal contradiction detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6051– 6061, 2026. 1, 2, 6, 11

[28] Baoliang Tian, Yuxuan Si, Jilong Wang, LingYao Li, Zhongyuan Bao, Zineng Zhou, Tao Wang, Sixu Li, Ziyao Xu, Mingze Wang, et al. Crosscheck-bench: Diagnosing compositional failures in multimodal conflict resolution. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 25887–25895, 2026. 1

[29] Chongjun Tu, Peng Ye, Dongzhan Zhou, Lei Bai, Gang Yu, Tao Chen, and Wanli Ouyang. Attention reallocation: Towards zero-cost and controllable hallucination mitigation of mllms. International Journal of Computer Vision, 134(1):22, 2026. 2

[30] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 6, 12

[31] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S Yu. Multimodal large language models: A survey. In 2023 IEEE international conference on big data (bigdata), pages 2247–2256. IEEE, 2023. 1

[32] Junfei Wu, Yue Ding, Guofan Liu, Tianze Xia, Ziyue Huang, Dianbo Sui, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. Sharp: Steering hallucination in lvlms via representation engineering. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 14357–14372, 2025. 1

[33] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12): nwae403, 2024. 1

[34] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences, 67(12):220105, 2024. 2

[35] Chi Zhang, Wenxuan Ding, Jiale Liu, Mingrui Wu, Qingyun Wu, and Ray Mooney. Do images speak louder than words? investigating the effect of textual misinformation in vlms. In Proceedings ofthe 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6872–6895, 2026. 1

[36] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models. Findings of the association for computational linguistics: ACL 2024, pages 12401–12430, 2024. 1

[37] Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai, Kehai Chen, Yang Xiang, Jun Yu, and Min Zhang. Evaluating and steering modality preferences in multimodal large language model. arXiv preprint arXiv:2505.20977, 2025. 1, 2, 6, 11, 12, 13

[38] Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, and Lijie Hu. When modalities conflict: How unimodal reasoning uncertainty governs preference dynamics in mllms. arXiv preprint arXiv:2511.02243, 2025. 1, 2, 3

[39] Zongmeng Zhang, Wengang Zhou, Jie Zhao, and Houqiang Li. Robust multimodal large language models against modality conflict. In Forty-second International Conference on Machine Learning, 2025. 1, 2

[40] Jianfei Zhao, Feng Zhang, Xin Sun, Chong Feng, and Zhixing Tan. Tell model where to look: Mitigating hallucinations in mllms by vision-guided attention. arXiv preprint arXiv:2511.20032, 2025. 2

[41] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 6

## A. Benchmark Information and Examples

We include examples from the benchmarks to illustrate the conflict settings used in our evaluation. For each benchmark, we present one image and text context, along with the question and the corresponding gold answer in the benchmark. We include the following benchmarks: MMCD [25], MC<sup>2</sup> [37], CLASH[27], CorrelationQA [10], and PhD (sec and icc subsets) [22].

## Example 1: MMCD

Context: Caption: an image of a green star

Question: What color is the star?

Image-supported answer: Black

Text-supported answer: Green

MMCD is a synthetic modality-conflict benchmark built from simple visual objects, such as different shapes and colors. The benchmark contains controlled image–caption conflicts involving five shapes and eight colors, making it useful for conflict arbitration without complex scene understanding. In the example, the image and caption assign different colors to the same shape. There are 5600 examples in the dataset.

## Example 2: MC<sup>2</sup>

![](images/369bc968d2fe0e316c42c014f23f1ff1c837404524a64adce105b2a86cd3b5aa.jpg)

Context: On a sunny afternoon, a group of young men are gathered at the edge of a soccer field, engaged in an intense discussion about their upcoming match strategies. Among them, a dark-haired individual stands with a clipboard, taking notes and occasionally gesturing to emphasize points in the conversation. Instead of being in motion, he is focused on facilitating the tactical planning, ensuring everyone understands their roles for the game ahead.

Question: What is dark haired person doing?

Image-supported answer: Running

Text-supported answer: Standing still

MC<sup>2</sup> is a controlled modality-conflict benchmark of 2000 samples designed to measure modality preference in MLLMs. The examples pair image and text evidence that can support different answers, along with a question, an image-supported answer, and text-supported answer. The images in MC<sup>2</sup> come from MS COCO 2014 [21]. We additionally augmented it with text that yields the imagesupported answers, and the details are described in Section B.3.

## Example 3: CLASH

![](images/4462f67c6c21d32eccb3885eb33e65865d1f515070bf76b1d7bc63f2f5666a12.jpg)

Context: A basket of fruit on the counter of a bathroom.

Question: What is contained within the basket on the bathroom counter?

Image-supported answer: Flowers

Text-supported answer: Fruits

CLASH is a multimodal conflict benchmark designed to evaluate model behavior when visual and textual evidence contradict each other. The benchmark provides modalityspecific answers supported independently by the image and text. CLASH uses images from the MS COCO 2017 [21] training split and constructs contradictory textual descriptions based on the corresponding captions. Compared to MC<sup>2</sup>, CLASH has shorter text descriptions. We use the held-out test set containing 1289 samples.

## Example 4: CorrelationQA

![](images/aac42ebc9fcf7d5495c46bd53a68a17e790afba25dc2f5ce257babba50d15188.jpg)

Context: This African mammal, known for its long neck and spotted coat, is the tallest living terrestrial animal.

Question: What is it? Answer no more than five words.

Image-supported answer: Zebra

Text-supported answer: Giraffe

CorrelationQA studies visual illusions caused by spurious images that are relevant to the question but inconsistent with the answer supported by the text. The benchmark contains 5540 text-image pairs across multiple categories and is designed to measure whether misleading visual context biases multimodal models away from the correct textual answer. CorrelationQA constructs its visual inputs using a combination of Stable-Diffusion-generated images, typography images, and realistic images retrieved from the Internet.

## Example 5: PhD-sec

![](images/4b2a8ede3cf7df6885e302685e00b5fd97bd256b0d41ad5a0b444b17fc68a519.jpg)

Context: The knife used for slicing bananas is typically made from ceramic, known for its sleek and sharp cutting edge, making it a preferred tool in food preparation. Alongside the knife, a plate of neatly sliced bananas highlights both the fruit’s freshness and the efficiency of the cutting process. For those looking to blend or process the fruit further, a food processor stands ready as a practical kitchen companion.

Question: Is the knife made of metal in the image?

Answer: Yes

## Example 6: PhD-icc

![](images/3725f234aeafc6e066c2308921ce6d4166cfcf9b1b9e2b6dd852f799da39ea5e.jpg)

Context: A ceramic knife, prominently featured in the image, showcases its sharp elegance while being accompanied by a plate topped with fresh banana slices. The scene includes a food processor ready for action, with another plate of sliced bananas nearby, illustrating the perfect preparation setup for a delicious smoothie or dessert. This arrangement highlights the versatility of bananas, with various tools like a chopper for easy slicing, ensuring a seamless culinary experience.

Question: Is the knife made of metal in the image?

Answer: Yes

PhD is a visual hallucination benchmark that evaluates whether MLLMs remain grounded in visual evidence in the presence of potentially misleading textual context. Following prior work [37], we evaluate 1000 samples from each of the PhD-sec and PhD-icc subsets, which contain irrelevant and conflicting textual information, respectively. Unlike the modality-conflict benchmarks, we evaluate PhD using image accuracy, measuring how strongly the model preserves or suppresses reliance on visual evidence.

## B. Experimental Details

## B.1. Models

We evaluate four open-source multimodal language models loaded directly from HuggingFace: Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, llava-hf/llavaonevision-qwen2-7b-ov-hf, and OpenGVLab/InternVL3- 8B-hf. The four open-source models use a shared Hugging Face chat-template adapter. Typical runs used one NVIDIA RTX A5000, bfloat16, batch size 1, seed 42, and greedy decoding. The maximum generation length was 128 tokens except for CoT, which used 512 tokens. Images were loaded at their original resolution and converted to RGB. Model-specific resizing, rescaling, and normalization were performed by the native Hugging Face AutoProcessor. Blur experiments used PIL Gaussian blur with radii 2, 5, and 10 for levels 1–3, without changing image dimensions.

## B.2. Baselines

In addition to the vanilla inference with greedy decoding, the baselines are chosen to cover three typical intervention methods: prompting (CoT [30], CAP), intervention during decoding (VCD [14]), and intervention on representations (MPS [37]). All methods except for CoT use the following suffix to avoid overlong answers.

Answer the question directly with a short answer.

Chain-of-thought prompting. We include Chain-ofthought (CoT) as a prompting method to validate whether MLLMs can resolve the conflicts via additional thinking. Specifically, CoT encourages the model to explicitly reason over the multimodal evidence before producing its final answer, without specifying which modality should be prioritized. The prompting suffix is as follows:

Let’s think step by step, then answer the question with a short answer.

Conflict-aware prompting. To test whether the conflicts can be resolved via prompting, we include conflict-aware prompting (CAP). CAP explicitly instructs the model to answer the question based on one of the modalities. The prompting suffixes are as follows:

Answer the question based on the image.

Answer the question based on the   
text.

This baseline isolates the effect of instruction-level conflict awareness: unlike vanilla inference, it tells the model that one modality should be trusted, but it does not modify the model’s internal representations or decoding procedure. CAP serves as a strong prompting baseline for evaluating whether our method provides benefits beyond simply asking the model to reason about source reliability.

Visual Contrastive Decoding (VCD) VCD [14] is a training-free decoding method originally proposed for mitigating object hallucination. VCD contrasts the output logit distribution conditioned on the original image with the distribution conditioned on a distorted image, thereby downweighting tokens that are more likely to arise from language priors or statistical biases rather than from the visual input. This baseline is relevant because many of our examples require the model to remain grounded in the reliable image input. However, we note that VCD is primarily designed to improve visual grounding without identifying which modality or input component is trustworthy under conflicts. It therefore provides a useful comparison for cases where visual grounding is beneficial, but it is not a general method to handle conflicts, especially when text should be trusted.

Our implementation uses the hyperparameters as

$$
\alpha = 1 , \quad \beta = 0 . 1 , \quad \gamma = 0 . 1 , \quad T = 5 0 0 .
$$

The diffusion process uses $\gamma$ for its noise schedule and the distorted-image signal coefficient was $( 1 - \gamma ) ^ { T / 2 }$ . Decoding used

$$
\ell _ { \mathrm { V C D } } = ( 1 + \alpha ) \ell _ { \mathrm { o r i g i n a l } } - \alpha \ell _ { \mathrm { d i s t o r t e d } } ,
$$

together with the adaptive plausibility constraint, which prevents contrastive decoding from promoting tokens that are implausible under the original image. Only tokens satisfy

$$
p ( y _ { t } \mid v , x ) \ge \beta \operatorname* { m a x } _ { w } p ( w \mid v , x )
$$

are retained.

Modality Preference Steering (MPS) MPS [37] is introduced in the same work as the $\mathbf { M C ^ { 2 } }$ benchmark. MPS is particularly aligned with our setting, as it is designed to control an MLLM’s preference between visual and textual evidence when the two sources provide competing signals. It first probes the model with paired instructions that encourage reliance on either the visual context or the textual context, estimates a modality-preference direction in hiddenstate space, and then adds a scaled steering vector during inference to shift the model toward the desired modality.

In our implementation, a modality-preference vector was computed offline from 500 $\mathbf { M C ^ { 2 } }$ samples. Each example was presented twice using image- and text-guided conflict instructions in this order: text context, modality instruction, question, and general direct-answer guidance. The modality instructions are as follows

In case there is an inconsistency   
between the text context and the   
image content, you should follow   
the image content rather than the   
text context.

In case there is an inconsistency   
between the text context and the   
image content, you should follow   
the text context rather than the   
image content.

At the selected decoder layer, the final non-padding prompt-token state was recorded. The text direction was

$$
v = \frac { 1 } { N } \sum _ { i } ( h _ { i } ^ { T } - h _ { i } ^ { V } ) ,
$$

and was normalized by the mean text-state norm divided by $| | \boldsymbol { v } | | _ { 2 }$ . Steering to text adds the resulting vector to every token state, whereas steering to image subtracts it. The zero-based intervention layers were 22 for Qwen2.5, 28 for Qwen3, 22 for LLaVA-OV, and 16 for InternVL3. As the steering directly applies to the hidden states, the performance and response generation quality are sensitive to the choice of intervention scales. We use the scale of 0.25 for MMCD, $\mathbf { M C } ^ { 2 }$ , CLASH, and CorrelationQA and 0.75 for PhD subsets for Qwen3-VL-8B to ensure stable answer quality. The remaining models, including Qwen2.5-VL-7B, LLaVA-OneVision-7B, and InternVL3-8B, follow the standard scale of 1.0 as in [37].

Ours Our main experiments in Tables 3 and 4 apply the arbitration head localization for every MLLM on 10 samples, each containing an image, a conflicting text, a question, and image-/text-supported answers. The top-30 heads are applied for the intervention on all the benchmarks with an intervention strength α of 30. The ϵ in Equation 5 is 1e-12. For the adaptive steering in Section 4.4, we use a fixed seed-42 sample of $2 0 0 \mathbf { M C ^ { 2 } }$ , supplemented with the aligned text to form 200 aligned/conflicting pairs. The threshold τ obtained on the 200 samples is 0.801, with 83.0% conflict accuracy, 74.0% aligned accuracy, 78.5% balanced accuracy, and AUC 0.8517. We use $\alpha _ { \mathrm { m a x } }$ of 50 and λ of 0.05 for the intervention strength α choice in Equation 7.

## B.3. MC<sup>2</sup> Aligned Text Generation

For the adaptive steering experiment in Section 4.4, we supplemented the text that yields the image-supported answers, but in a similar format to the conflicting text in the original MC<sup>2</sup>. Specifically, we use gpt-4.1-mini and input the original conflicting context, text-supported answer, and imagesupported answer. We then rewrote statements associated with the text answer to be consistent with the vision answer while preserving unrelated wording, detail, style, and structure as closely as possible. The prompt for rewriting is as follows.

We show an example of the aligned text below, compared to Example 2. The experiments in Section 4.4 use a mixture of the aligned and conflicted input data, and the ratio of conflicted samples is controlled by a hyperparameter of 0, 0.2, 0.4, 0.6, 0.8, and 1.

You rewrite a context so its relevant evidence agrees with a target answer. Replace every statement, description, implication, synonym, or paraphrase semantically associated with the original text answer with content consistent with the vision answer. Preserve all unrelated content as closely as possible, including wording, detail, tone, style, paragraphing, and structure. Resolve any nearby wording that would otherwise contradict the vision answer, but do not invent unrelated details. Return only the rewritten context in the requested structured format.

## B.4. CorrelationQA Image-supported Answer Generation

The original CorrelationQA only provides the textsupported answers, and we supplement the imagesupported answers by instructing gpt-4.1-mini to generate answers with only the image and the decomposed final question. The model was instructed to provide the most factually correct image-grounded answer in no more than two words, without explanation or negation. Each example allowed at most three API calls, with constraint-focused retry instructions when necessary. If all attempts violated the format, the final answer was retained in case the answer needed more than two words. The prompt to generate the image-supported answers is as follows:

## Example 7: MC<sup>2</sup> (aligned)

![](images/f60b19a09105583d41dfee71a6fda8b849e82694562829d67c1424520beeb0ad.jpg)

Context: On a sunny afternoon, a group of young men are gathered at the edge of a soccer field, engaged in an intense discussion about their upcoming match strategies. Among them, a dark-haired individual is running with a clipboard, jotting notes as he moves and occasionally gesturing to emphasize points in the conversation. Rather than standing still, he actively moves around, facilitating the tactical planning and ensuring everyone understands their roles for the game ahead.

Question: What is dark haired person doing?

Answer: Running

Answer the following question. An image is provided as additional context. Give the most factually correct answer from the image to the question. Return only the answer in no more than 2 words, without explanation. Do not use negation in the answer.

## B.5. Evaluation

Accuracy was reported independently against vision and text references whenever both were available. PhD reported vision accuracy only, including for text-directed methods. The default evaluator was the hybrid parser with an exact matcher and an LLM as the judge. For each generated response, we evaluate it against image-supported answer or text-supported answer in two runs. We first normalize case, punctuation, whitespace, and articles for the generated response. We accept a reference when its complete normalized token sequence appears contiguously in the generated response. For datasets with both image-/text-supported answers, we short-circuit when exactly one reference matches. If both or neither match, call gpt-4o-mini separately for the vision and text references. For PhD, the single-reference dataset, short-circuit on a positive match, and otherwise call the judge.

The judge used temperature 0, structured boolean output, 128 maximum output tokens, and prompt version 1.5. It judged semantic equivalence and accepted synonyms, paraphrases, harmless specificity differences, and consistent additional details. Accuracy and extraction-failure rate were recorded separately for each ground-truth type. The prompt for the LLM judge is

You are an impartial answer  
equivalence judge. Decide whether   
a candidate answer is semantically   
equivalent to the provided reference   
answer for the given question. Judge   
meaning, not exact wording. Accept   
synonyms, paraphrases, equivalent   
numbers, harmless grammatical or   
formatting differences, and reasonable   
differences in specificity or detail.   
Accept the subset or superset of a   
ground-truth answer as long as any   
omitted or added detail does not   
materially change or contradict the   
core answer. Mark the result incorrect   
only if it contradicts or is   
semantically inconsistent with the   
ground-truth answer. Return the   
requested structured result only.

## C. Computational Efficiency

Experiments were conducted on a machine with NVIDIA driver 580.173.02 and three RTX A5000 GPUs, each with 24 GB and each run used a single GPU. The main software dependencies included Python 3.10.20, Py-Torch 2.12.0+cu130, Transformers 5.7.0, CUDA 13.0, and cuDNN 9.2.

We evaluate the computational efficiency of different inference methods using end-to-end latency (s/request), time to first token (TTFT; s), decode latency per output token (s/token), token throughput (tokens/s), peak GPU memory (GiB), and output token count (tokens). End-to-end latency measures the wall-clock time from the beginning of an inference request until the decoded response is returned, including input preprocessing and any method-specific computation. TTFT measures the time from the start of generation to the first generated token, while decode latency measures the subsequent generation time normalized by the number of decoded tokens. Token throughput is computed as the total number of generated tokens divided by the total generation time, including prefill. Peak GPU memory corresponds to the maximum CUDA memory allocated during inference, including both the resident model and temporary inference tensors. Output token count measures the number of tokens generated per request.

We report efficiency results on $\mathbf { M C ^ { 2 } }$ and compute all metrics over unique inference batches to avoid doublecounting the image- and text-reference evaluation rows corresponding to the same generation. End-to-end latency (s/request) and TTFT (s) are averaged across requests, decode latency (s/token) is pooled across generated tokens, token throughput (tokens/s) is computed from the total output tokens and generation time, and peak GPU memory (GiB) is taken as the maximum across inference batches. Output tokens are reported as the average number of generated tokens. For our method, we measure the main intervention’s computational efficiency for comparison with other methods, while we note that the adaptive method requires additional unimodal inference for conflict detection. For methods that support bidirectional modality steering, we report the image-steering setting for a consistent comparison.

The results show that our method only yields limited inference-time overhead while maintaining output lengths comparable to standard inference. On Qwen3-VL-8B and Qwen2.5-VL-7B, our method increases end-to-end latency moderately relative to Base and CAP, while remaining substantially faster than VCD or MPS on Qwen2.5-VL-7B. On LLaVA-OV-7B and InternVL3-8B, the latency increase over Base and CAP is larger, but our method remains considerably faster than VCD. Across all four models, the number of output tokens produced by our method is close to that of Base and CAP, indicating that its additional latency arises from the inference-time intervention rather than longer generations. In contrast, VCD yields substantially higher TTFT and decode latency across all models, while CoT can lead to high end-to-end latency when it produces substantially longer responses. Peak GPU memory for our method remains close to Base on the two Qwen models, with larger increases on LLaVA-OV-7B and InternVL3-8B. Overall, our method provides effective modality arbitration with moderate computational overhead.

## D. Additional Results

## D.1. Robustness of Modality Conflict Detection

To assess the robustness of our conflict diagnostic, we compute cosine similarity after mean-centering each logit vector to remove sensitivity to uniform additive shifts. The balanced accuracy decreases only slightly, from 78.5% with the original cosine similarity to 77.5% with mean-centered cosine similarity. The conflicting accuracy decreases from 83% to 81%, while the aligned accuracy remains unchanged at 74%. These results suggest that the observed conflict signal is largely robust to uniform additive shifts in the logits.

Further, for assessing sensitivity to calibration-set selection, we repeat threshold calibration using a different random seed of 43. The resulting threshold is 0.8057, yielding a balanced accuracy of 75.5%, with 79% conflicting accuracy and 72% aligned accuracy. These results indicate that conflict detection remains effective under a different calibration-set selection, although performance varies modestly with the calibration seed.

<table><tr><td>Model</td><td>Method</td><td>End-to-end latency</td><td>TTFT</td><td>Decode latency</td><td>Token throughput</td><td>Peak GPU memory</td><td>Output tokens</td></tr><tr><td rowspan="6">Qwen3-VL-8B</td><td>Base</td><td>0.232</td><td>0.134</td><td>0.037</td><td>14.705</td><td>16.500</td><td>3.162</td></tr><tr><td>CoT</td><td>4.857</td><td>0.138</td><td>0.036</td><td>26.898</td><td>16.502</td><td>130.129</td></tr><tr><td>VCD</td><td>0.637</td><td>0.270</td><td>0.172</td><td>4.873</td><td>16.601</td><td>2.933</td></tr><tr><td>CAP</td><td>0.225</td><td>0.136</td><td>0.038</td><td>13.837</td><td>16.502</td><td>2.884</td></tr><tr><td>MPS</td><td>0.245</td><td>0.136</td><td>0.039</td><td>14.622</td><td>16.500</td><td>3.314</td></tr><tr><td>Ours</td><td>0.275</td><td>0.148</td><td>0.050</td><td>12.300</td><td>16.556</td><td>3.146</td></tr><tr><td rowspan="6">Qwen2.5-VL-7B</td><td>Base</td><td>0.383</td><td>0.189</td><td>0.027</td><td>20.558</td><td>15.594</td><td>7.492</td></tr><tr><td>CoT</td><td>1.398</td><td>0.184</td><td>0.026</td><td>33.718</td><td>15.596</td><td>46.440</td></tr><tr><td>VCD</td><td>2.123</td><td>0.365</td><td>0.211</td><td>4.397</td><td>15.730</td><td>9.192</td></tr><tr><td>CAP</td><td>0.400</td><td>0.182</td><td>0.026</td><td>22.817</td><td>15.594</td><td>8.701</td></tr><tr><td>MPS</td><td>0.738</td><td>0.181</td><td>0.026</td><td>29.982</td><td>15.594</td><td>21.578</td></tr><tr><td>Ours</td><td>0.491</td><td>0.194</td><td>0.036</td><td>18.237</td><td>15.654</td><td>8.545</td></tr><tr><td rowspan="6">LLaVA-OV-7B</td><td>Base</td><td>0.678</td><td>0.612</td><td>0.028</td><td>4.003</td><td>15.722</td><td>2.630</td></tr><tr><td>CoT</td><td>0.707</td><td>0.642</td><td>0.028</td><td>3.711</td><td>15.724</td><td>2.546</td></tr><tr><td>VCD</td><td>2.398</td><td>1.262</td><td>0.675</td><td>1.112</td><td>16.363</td><td>2.617</td></tr><tr><td>CAP</td><td>0.678</td><td>0.613</td><td>0.028</td><td>3.982</td><td>15.724</td><td>2.620</td></tr><tr><td>MPS</td><td>0.662</td><td>0.601</td><td>0.028</td><td>3.745</td><td>15.722</td><td>2.400</td></tr><tr><td>Ours</td><td>0.960</td><td>0.875</td><td>0.041</td><td>2.708</td><td>19.301</td><td>2.539</td></tr><tr><td rowspan="6">InternVL3-8B</td><td>Base</td><td>0.730</td><td>0.631</td><td>0.028</td><td>4.516</td><td>15.520</td><td>3.113</td></tr><tr><td>CoT</td><td>3.366</td><td>0.596</td><td>0.027</td><td>30.969</td><td>15.523</td><td>103.214</td></tr><tr><td>VCD</td><td>2.585</td><td>1.253</td><td>0.667</td><td>1.150</td><td>16.125</td><td>2.890</td></tr><tr><td>CAP</td><td>0.726</td><td>0.632</td><td>0.028</td><td>4.400</td><td>15.523</td><td>3.029</td></tr><tr><td>MPS</td><td>0.721</td><td>0.603</td><td>0.028</td><td>5.734</td><td>15.520</td><td>3.917</td></tr><tr><td>Ours</td><td>0.906</td><td>0.780</td><td>0.037</td><td>3.830</td><td>18.461</td><td>3.320</td></tr></table>

Table 6. Efficiency comparison of methods across MLLMs.

![](images/0f58d0c0b7afaa457532719c4f416388a3942fd081a5e204ffca51ee6441ed7d.jpg)  
Figure 7. Robustness of modality conflict detection using centered logits. Top-50 union-logit cosine similarities after centering each logit vector, for samples paired with conflicting or aligned context. Lower similarity indicates stronger modality conflict. The dashed line marks the conflict-detection threshold calibrated to maximize accuracy on the calibration set.

## D.2. Visual Degradation Examples

Table 2 shows the performance of Qwen3-VL-8B when the image quality is degraded with Gaussian blurring. We include an example of $\mathbf { M C ^ { 2 } }$ for Blur 1-3.

## D.3. Heatmaps for Other MLLMs

In addition to the heatmap visualization for Qwen3-VL-8B, we further examine the stability of arbitration-head localization on the other MLLMs as the calibration set size increases from n=3 to n=10 . Across LLaVA-OneVision-7B, Qwen2.5-VL-7B, and InternVL3-8B, highinfluence heads repeatedly emerge at similar layer–head locations across different sample sizes. Several prominent heads are already identifiable with only three samples and become increasingly stable as more calibration examples are used. Despite model-specific differences in the overall distribution of influence scores, the persistence of highinfluence locations suggests that arbitration heads can be reliably localized from a small calibration set across different MLLM architectures.

![](images/c9c481f96578c4b6422547c5376adaf738965647dbe92423c73ead8f900de516.jpg)  
Figure 8. Robustness of modality conflict detection to calibrationset selection. Top-50 union-logit cosine similarities for samples paired with conflicting or aligned context, using a calibration set generated with random seed 43. The dashed line marks the detection threshold selected to maximize conflict-detection accuracy on the calibration set.

![](images/a933efec974aa1f58069cce599cb0651616cd0e93a9f57a2ae43aef9902301f5.jpg)  
(a) Original image

![](images/b83d463074525a01aa33554189460e5db490d30bc090cc8c117f06dcd7a5a3fd.jpg)  
(b) Blur-1

![](images/96d1a4d7628c3213f7e016d5df16ec1af671aad7d6ac08dbd3188e7ec916446f.jpg)  
(c) Blur-2

![](images/c5d377c1bf4c18d7d0f824f39ee7a80974068040945ed72631c532f7270c14a9.jpg)  
(d) Blur-3

Figure 9. Examples of Blur 1-3 on one $\mathbf { M C } ^ { 2 }$ image.  
![](images/82b38bc44ea57a8fe84f62416ac61832f9084f636f66713465af894a680d9dcc.jpg)

![](images/1ac7bddaa38a79b7f02d565e7f7911617fee9d55a4748d6cce4a2f50a021181a.jpg)

![](images/71858035af3be832524e4e3307de812725cef28c86d3c171bba40f499fce5189.jpg)

![](images/0a70d0374afba07c4d706f3a8558a53ef8fa0dc0ade8abbb8bee8243f1223490.jpg)  
Figure 10. Stability of head-wise effects across different sample sizes for Qwen2.5-VL-7B.

## D.4. Robustness to Visual Degradation

We further examine modality arbitration performance as the reliability of visual evidence decreases by progressively blurring the input image. Base, CoT, and VCD increasingly shift toward the text-supported answer as image quality decreases, demonstrating that modality arbitration is sensitive to the relative strength of cross-modal evidence. When steering toward vision, our method consistently preserves higher vision-answer accuracy than the bidirectional baselines across degradation levels, although its effectiveness gradually decreases under severe blur. Conversely, when steering toward text, our method maintains consistently high text-answer accuracy across all degradation levels. These results demonstrate that the proposed intervention provides robust bidirectional control over modality reliance even as the relative reliability of the visual and textual evidence changes.

## E. Detailed Discussion

Our work highlights several limitations and opportunities for future work. First, as we find that heads identified using only 10 samples generalize well across the evaluated benchmarks, it remains unclear whether the same patterns transfer to substantially different domains or real-world tasks. Second, our evaluation uses a simplified modality-conflict setup in which the modality to prioritize is externally specified. In realistic settings, both modalities may provide useful yet partially conflicting evidence, requiring the model to integrate complementary information rather than rely exclusively on one modality. Our current evaluation does not fully capture such cases, and benchmarking our intervention in settings where unimodal routing is insufficient remains an important direction for future work. Automatically assessing modality reliability and identifying which evidence to trust are also open challenges. Following this, the aligned $\mathbf { M C ^ { 2 } }$ evaluation in our work measures whether adaptive steering retains the benefit of multimodal inference when both modalities support the same answer, but it does not test examples that require combining distinct evidence from both modalities. Preservation of useful competing-modality information under partial conflict therefore remains to be evaluated. Finally, we introduce an adaptive steering extension that detects conflicts by comparing image-only and text-only predictions. While this approach demonstrates practical applicability in settings where conflicts are sparse, it introduces additional inference cost for conflict detection. Developing more efficient conflict signals that can be extracted directly from a single multimodal forward pass would be a promising direction for future work.

![](images/f5623b012e94b9938b399487990bb1a80d9998b217c11eb4382755c2ccfb1c69.jpg)

![](images/42c471b9f046fb3413957192a0a8d3579906c6f12f233a21e65f6cdce1ec53e2.jpg)

Figure 11. Stability of head-wise effects across different sample sizes for LLaVA-OneVision-7B.  
![](images/f0805e9fe0123dd6e41a11b34e5e93fe2d3768142363bf6dd48bab0025a171d9.jpg)

![](images/f25a4d33b5344a20be6fa9f9994acc60f78a72323d3f0e6b4eec64df4898d46e.jpg)

![](images/95705bdbb54d72ed6e27fe6a24113882a56cc332cfff92fab31c27fda1b4ab88.jpg)

![](images/72d76a71a6f6992ae42628fd57e369528304aa2ac484c7dabe752cf2cd472440.jpg)

![](images/a4b939c6b526f72cf934b4e2c6f609dec98f3f9c01ec1694e41464a233b7a8c6.jpg)  
Figure 12. Stability of head-wise effects across different sample sizes for InternVL3-8B.

![](images/cd3d7a043d9dc118735a9afce7a668db59160752f3de5ba1847b1f2eb19a0756.jpg)

![](images/59de5dc4bad648e8c3697b494529249409e262ee6bb8bc5df051320e08a665b9.jpg)

![](images/2ae25f165c3e003e109e644887add2c694b1e223df9923fed5fbb704666eda7f.jpg)  
Figure 13. Robustness to visual degradation.