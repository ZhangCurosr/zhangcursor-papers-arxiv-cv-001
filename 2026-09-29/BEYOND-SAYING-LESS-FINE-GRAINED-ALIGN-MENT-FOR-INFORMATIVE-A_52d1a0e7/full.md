# BEYOND SAYING LESS: FINE-GRAINED ALIGN-MENT FOR INFORMATIVE AND FAITHFUL VISION-LANGUAGE MODELS

Xingming Long<sup>1,2,3</sup>, Jie Zhang<sup>1,2∗</sup>, Yuecong Min<sup>1,2</sup>, Shiguang Shan<sup>1,2</sup>, Xilin Chen<sup>1,2</sup>

<sup>1</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences

<sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Zhongguancun Academy

xingming.long@vipl.ict.ac.cn,

{zhangjie,minyuecong,sgshan,xlchen}@ict.ac.cn

## ABSTRACT

Object hallucination remains a major challenge for large vision-language models. While off-policy preference optimization proves to be an effective solution, on-policy reinforcement learning provides a more promising direction as it directly targets a model’s current failure modes. However, we find that without fine-grained reward formulation and allocation, on-policy optimization often falls into an easy shortcut: reducing hallucinations merely by saying less—making fewer valid claims. To comprehensively resolve this, we propose a fine-grained alignment framework that couples dense reward signals at the data level with precise credit assignment at the algorithmic level. Specifically, we first construct the Dense Object Presence and Absence (DOPA) dataset to address sparse annotations that prevent valid object claims from being verified and rewarded. DOPA exhaustively annotates the deterministic presence and absence of every concept across an expanded vocabulary, significantly increasing the density of reliable reward signals during on-policy rollouts. Second, we propose Subsentence-level Credit Assignment for on-Policy Optimization (SCAPO) to prevent responselevel shared advantages from allowing local hallucinations to compromise all other valid outputs within the same response. By assigning credit to each subsentence independently based on its object claims, SCAPO can precisely reinforce faithful generations and penalize hallucinations. Furthermore, we leverage the resulting faithful image descriptions as auxiliary context to transfer generative gains to discriminative tasks. Experiments demonstrate that our method produces highly informative, faithful descriptions in generative tasks while yielding clear performance gains on discriminative evaluation.

## 1 INTRODUCTION

Large vision-language models (LVLMs) have achieved substantial progress in image description, visual question answering, and multimodal reasoning. Despite these advances, they frequently generate content that is unsupported by the available visual evidence, a failure commonly referred to as hallucination Liu et al. (2024a); Bai et al. (2024); Lan et al. (2024); Rani et al. (2024); Li et al. (2023). For example, a model can claim that an object is present even though it does not appear in the image. Such errors reduce response reliability, mislead subsequent reasoning, and impede the deployment of LVLMs in settings that require dependable visual understanding. Ensuring that generated content remains consistently grounded in visual evidence has therefore become a central problem in reliable LVLM research.

Existing hallucination-mitigation approaches broadly fall into inference-time and training-based categories. Inference-time methods operate without retraining by employing contrastive decoding Leng et al. (2024); Wang et al. (2024b), representation interventions Huang et al. (2024); Gong et al. (2024); Liu et al. (2025b), or post-hoc self-correction Yin et al. (2024); Zhou et al. (2024b). In contrast, training-based methods intrinsically align LVLM generation preferences. While most existing paradigms rely on off-policy optimization using static datasets Yu et al. (2024a); Zhao et al. (2023); Xiao et al. (2025); Zhou et al. (2024a); Sarkar et al. (2025); Rafailov et al. (2023), recent studies demonstrate that learning directly from annotated on-policy rollouts Yang et al. (2025) is substantially more effective in mitigating policy-data mismatch.

![](images/d4d37a852aabb88f438d7c4a564964f70daf37908e80728d41f12f09a14c6831.jpg)  
Figure 1: Motivation for our approach. Conventional optimization suffers from sparse annotations leaving valid claims unrewarded, and response-level advantages allowing local hallucinations to suppress correct outputs. By combining the dense rewards of DOPA with the subsentence-level credit assignment of SCAPO, we achieve fine-grained alignment, preventing this suppression and yielding informative and faithful descriptions.

While optimizing on-policy rollouts directly exposes a model’s current errors, we find that without fine-grained reward formulation and allocation, it often falls into an easy shortcut: the model reduce hallucinations not by improving faithfulness, but by reducing informativeness—that is, essentially saying less by making fewer valid claims. As illustrated in Figure 1, we identify two coarse-grained bottlenecks driving this behavior. The first is a data-level bottleneck: sparse annotations leave valid object claims in current rollouts unverified and unrewarded. The second is an algorithmic-level bot tleneck: response-level optimization assigns shared advantages across the entire text, allowing loca hallucinations to compromise all other valid outputs within the same response. Consequently, the model is incentivized toward global conservatism rather than precise error correction, underscoring the critical need for denser reward coverage and localized credit assignment.

To comprehensively resolve these bottlenecks, we propose a fine-grained alignment framework coupling dense reward signals with precise credit assignment. At the data level, we construct the Dense Object Presence and Absence (DOPA) dataset, which provides dense, reliable training rewards by exhaustively annotating the deterministic presence or absence of concepts over an expanded vocabulary. At the algorithmic level, we introduce Subsentence-level Credit Assignment for on-Policy Optimization (SCAPO). Rather than relying on coarse response-level shared advantages, SCAPO segments responses at subsentence boundaries and independently assigns credit. This localized mechanism ensures that reinforcement and suppression are precisely targeted, preventing isolated hallucinations from penalizing other valid outputs within the same response.

## Our contributions are summarized as follows:

• We construct the Dense Object Presence and Absence (DOPA) dataset to overcome datalevel reward sparsity. By exhaustively annotating the deterministic presence and absence of concepts across an expanded vocabulary, DOPA provides dense and reliable reward signals for object claims during on-policy rollouts.

• We propose Subsentence-level Credit Assignment for on-Policy Optimization (SCAPO) to resolve the algorithmic bottleneck of response-level shared advantages. By assigning credit independently to each subsentence based on its specific claims, SCAPO prevents local hallucinations from suppressing all other valid outputs.

• Extensive experiments show that our fine-grained alignment framework overcomes the “saying less” shortcut, producing descriptions that are both faithful and informative. As an additional transfer evaluation, we further show that these improved descriptions provide useful auxiliary context for discriminative tasks.

## 2 RELATED WORK

## 2.1 RLHF AND PREFERENCE OPTIMIZATION

Reinforcement learning from human feedback (RLHF) fundamentally aligns large models using on-policy algorithms such as PPO Schulman et al. (2017), though maintaining separate reward and value models is computationally expensive. Direct Preference Optimization (DPO) Rafailov et al. (2023) provides a simpler offline alternative, while policy-data mismatch has motivated iterative preference-refreshing approaches Pang et al. (2024). More recently, methods such as GRPO Shao et al. (2024) have revitalized on-policy optimization by estimating advantages from relative rewards without requiring a critic model, with subsequent variants addressing issues such as length-biased credit weighting Yu et al. (2025a); Liu et al. (2025d). However, GRPO-style methods typically assign rewards at the response level. This granularity is particularly limiting for hallucination mit igation, where hallucinated content often constitutes only a small portion of an otherwise faithful response. This motivates finer-grained credit assignment that can selectively penalize hallucinations while reinforcing correct content.

## 2.2 INFERENCE-TIME HALLUCINATION MITIGATION

Hallucination mitigation in LVLMs has attracted extensive attention Liu et al. (2024a); Bai et al. (2024); Lan et al. (2024); Rani et al. (2024). A prominent line of research focuses on inferencetime methods that improve visual faithfulness without updating model parameters. These include contrastive decoding to suppress language priors Leng et al. (2024); Wang et al. (2024b), generation interventions through attention or logit manipulation Huang et al. (2024); Gong et al. (2024); Liu et al. (2025b), post-hoc self-correction Yin et al. (2024); Zhou et al. (2024b), external visualevidence augmentation using auxiliary vision models Zhao et al. (2025); Li et al. (2025), and staged prompting that verbalizes visual evidence before final prediction Xuan et al. (2025). However, while inference-time approaches can improve visual faithfulness without retraining, they inevitably introduce inference latency and fail to internalize visual faithfulness directly into the model parameters.

## 2.3 TRAINING-BASED HALLUCINATION MITIGATION

Training-based methods intrinsically improve visual grounding via preference optimization. Early offline approaches construct preference pairs using expert corrections Yu et al. (2024a); Zhao et al. (2023); Xiao et al. (2025) or synthetic hallucination injection Zhou et al. (2024a); Sarkar et al. (2025). However, recent studies Yu et al. (2025b); Yang et al. (2025) demonstrate that on-policy training is more effective, as it directly targets the evolving policy’s failure modes. Yet, existing online methods face critical bottlenecks: accurate credit assignment demands either complex rollout modification Yang et al. (2025) or computationally expensive iterative LLM judge evaluation Yu et al. (2025b). To overcome these challenges, we introduce the DOPA dataset as a reusable, highdensity reward oracle providing exhaustive object-existence annotations on an expanded vocabulary. Building on this, we propose SCAPO to leverage these dense signals, shifting from coarse responselevel evaluation to precise subsentence-level credit assignment for on-policy rollouts.

## 3 CONSTRUCTION OF DOPA DATASET

To overcome the data-level bottleneck where sparse annotations leave valid object claims unrewarded, we construct the Dense Object Presence and Absence (DOPA) dataset to provide dense and reliable reward signals for on-policy rollouts. We build DOPA on 5,000 images sampled from the MS COCO Lin et al. (2014) training split, disjoint from all evaluation images. As illustrated in Figure 2, its construction comprises vocabulary construction, coarse annotation, edge-case refinement, and manual verification, which are detailed in the following subsections.

![](images/41542b988aa5e023bf8a021f8364be8db24cfaf497e607f01df320b870510022.jpg)  
Figure 2: Pipeline for constructing the DOPA dataset. The process begins with rollout-driven vocabulary construction to establish the canonical vocabulary V and alias mapping A. Based on these, large-scale coarse annotation and edge-case refinement are performed. Ultimately, a rigorous manual verification cycle yields the high-fidelity object-existence matrix E<sup>∗</sup>.

## 3.1 VOCABULARY CONSTRUCTION

To increase reward coverage for concepts naturally mentioned by the target model, we derive an expanded vocabulary directly from its rollouts on the training data. After extracting and normalizing entity mentions, we curate these candidates into canonical object labels based on three strict criteria: (i) sufficient frequency in target-model rollouts; (ii) direct visual verifiability without subjective inference; and (iii) semantic clarity with minimal overlap. We initially use GPT-5.5 OpenAI (2026) to cluster the candidates into a draft set, and then manually refine this draft against the aforementioned criteria. Finally, we construct a conflict-free alias mapping covering synonyms and morphological variations for each retained label. This process yields the final vocabulary V comprising 160 canonical labels and a mapping A of 894 aliases.

## 3.2 COARSE ANNOTATION

Existing datasets typically provide only positive annotations, leaving it ambiguous whether an unannotated concept is absent or simply omitted. DOPA instead defines a closed-set object-label space over V, in which every image-label pair is explicitly annotated as either present or absent. This exhaustive presence-absence annotation removes the unknown state for all in-vocabulary claims and provides unambiguous reward signals during on-policy training. We employ Qwen3.6-27B Qwen Team (2026) as the initial coarse judge to implement this exhaustive annotation. Prompted with strict criteria for handling occlusion and explicitly prohibiting commonsense inference, the judge determines object presence based solely on visual evidence. This produces a complete initial object existence matrix $\bar { E } _ { m , l } ^ { ( 0 ) } \in \{ 0 , 1 \}$ , indicating whether the object corresponding to the m-th canonical label is present in the l-th image.

## 3.3 EDGE-CASE REFINEMENT & MANUAL VERIFICATION

To improve annotation fidelity, we implement an iterative refinement and manual verification protocol. First, we evaluate model-generated object claims against the coarse matrix to isolate high-risk edge cases, such as heavy occlusion or unclear semantic boundaries. We resolve these systematic ambiguities by crafting tailored, label-specific prompts with rigorous inclusion/exclusion criteria to re-query the judge model. Subsequently, for each canonical label, we randomly sample 30 imagelabel pairs for manual evaluation. After each round of prompt refinement and re-annotation, we draw a new random set of 30 pairs and repeat the evaluation until no errors are observed in the newly sampled subset. Once every label passes this sample-based quality-control procedure, we finalize the re-annotated object-existence matrix $E _ { m , l } ^ { * } \in \theta , 1$

![](images/06a31278b0d8e18e82ca42b300a6507c688457d50f641318eb68c4d8ea4bb0ca.jpg)  
Figure 3: Overview of the SCAPO method and the adopted inference paradigm. Current-policy rollouts are segmented at subsentence boundaries and evaluated against the DOPA reward oracle. The resulting subsentence-level rewards are independently assigned to their corresponding token spans for the on-policy update. Furthermore, the optimized model leverages its generated faithful descriptions as auxiliary context for description-augmented discriminative inference.

Ultimately, this exhaustive process provides dense and reliable reward signals for the subsentencelevel credit assignment, enabling DOPA to serve as a high-quality, reusable reward oracle during training that requires only deterministic lookups rather than repeated judge-model calls.

## 4 SCAPO METHOD

To prevent coarse response-level shared advantages from allowing local hallucinations to suppress valid outputs, we build on the DOPA dataset and propose Subsentence-level Credit Assignment for on-Policy Optimization (SCAPO). As illustrated in Figure 3, SCAPO segments each response at subsentence boundaries and evaluates the object claims within each subsentence against the DOPA annotations, effectively encouraging highly informative and faithful generations rather than leading to global conservatism. We provide formal definitions and the pseudocode of our SCAPO method in Appendix B.

## 4.1 SUBSENTENCE-LEVEL REWARD

To evaluate the object mentions within the j-th subsentence of the i-th response, we categorize them according to the DOPA annotations and define three explicit counts: $n _ { i , j } ^ { \mathrm { { \bar { n e w } } } }$ represents the number of newly covered supported objects (valid claims appearing for the first time in the response), $n _ { i , j } ^ { \mathrm { r e p } }$ denotes the repeated supported objects (valid claims already mentioned in preceding subsentences), and $n _ { i , j } ^ { \mathrm { h a l } }$ counts the hallucinated objects. The subsentence-level reward is then defined as:

$$
\begin{array} { r } { r _ { i , j } = \left\{ \begin{array} { l l } { - r _ { h } \lambda _ { h } \left( n _ { i , j } ^ { \mathrm { h a l } } \right) , } & { n _ { i , j } ^ { \mathrm { h a l } } > 0 , } \\ { - r _ { \mathrm { r e g } } , } & { \mathrm { n o ~ o b j e c t s , } } \\ { + r _ { g } \lambda _ { g } \left( n _ { i , j } ^ { \mathrm { n e w } } \right) + r _ { \mathrm { r e p } } \lambda _ { g } \left( n _ { i , j } ^ { \mathrm { r e p } } \right) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{1}
$$

Here, $r _ { h } , r _ { g } , r _ { \mathrm { r e p } } , r _ { \mathrm { r e g } } \geq 0$ are base coefficients, scaled by monotonically increasing functions $\lambda _ { h }$ and $\lambda _ { g }$ . This formulation explicitly enforces three design principles: (1) Zero-tolerance for hallucinations: any hallucination triggers a penalty, strictly preventing correct information from offsetting explicit errors. (2) Grounded informativeness: hallucination-free subsentences are rewarded for novel claims and, marginally, for repetitions. (3) Anti-conservatism: subsentences lacking verifiable objects incur a regularization penalty $\left( - r _ { \mathrm { r e g } } \right)$ to discourage the model from generating generic, uninformative content.

## 4.2 SUBSENTENCE-LEVEL ON-POLICY RL TRAINING

During each optimization step, we sample image-prompt pairs from the current policy snapshot $\pi _ { \theta _ { \mathrm { o l d } } }$ . For each response $y _ { i }$ , we compute subsentence-level rewards ${ r } _ { i , j }$ via Eq. (1). To bypass coarse response-level aggregation, we directly broadcast these local rewards as token advantages: $A _ { i , t } = r _ { i , j }$ for all tokens $t \in T _ { i , j }$ , where $T _ { i , j }$ is the token span of subsentence $s _ { i , j }$ . Using these token advantages, we optimize the standard clipped surrogate objective:

$$
\mathcal { L } _ { \mathrm { S C A P O } } = - \frac { 1 } { N _ { \mathrm { t o k } } } \sum _ { i , j } \sum _ { t \in T _ { i , j } } \operatorname* { m i n } \Bigl ( \rho _ { i , t } A _ { i , t } , \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i , t } \Bigr ) + \beta _ { \mathrm { K L } } D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } )\tag{2}
$$

where $N _ { \mathrm { t o k } }$ is the total number of response tokens in the batch, $\rho _ { i , t }$ is the standard policy importance ratio, ϵ is the clipping coefficient, and $\beta _ { \mathrm { K I } }$ controls the KL regularization toward the fixed reference policy $\pi _ { \mathrm { r e f } }$ . Thus, while policy optimization remains token-wise, credit is localized at the subsentence level, with KL regularization preventing excessive policy deviation.

Unlike conventional PPO-based RLHF, SCAPO does not require training or maintaining an auxiliary value model. PPO-based RLHF typically estimates token-level advantages through a learned critic and GAE, whereas SCAPO assigns each subsentence an independently verified factual reward that is shared only among its own tokens. Advantage signals are not propagated across subsentence boundaries, preventing a later hallucination from altering the credit assigned to an earlier faithful statement. This design replaces general-purpose return estimation with explicit semantic locality, which is well suited to sparse and localized hallucination errors.

## 4.3 DESCRIPTION-AUGMENTED DISCRIMINATIVE INFERENCE

Following SCAPO optimization, the policy exhibits a marked improvement in generating faithful and informative visual descriptions. To transfer these generative grounding capabilities to downstream discriminative tasks, we adopt the description-augmented inference paradigm. Specifically, given an image and a discriminative query, the optimized model first generates a grounded description. The model then conditions its final discriminative answer on both the original image and this generated description jointly.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets. We evaluate our approach across both generative and discriminative tasks. For generative tasks, we assess hallucination rate and informativeness using MS COCO Lin et al. (2014) and AMBER Wang et al. (2023). For discriminative tasks, we utilize PhD Liu et al. (2025a) and the discriminative tasks of AMBER.

Comparison Methods. All methods use Qwen2.5-VL-7B Bai et al. (2025b) as the backbone. We organize the evaluated methods into three groups. The inference-time group includes the unmodified backbone, VCD Leng et al. (2024), OPERA Huang et al. (2024), and VTI Liu et al. (2025b). The offpolicy group includes standard DPO Rafailov et al. (2023), multimodal DPO (mDPO) Wang et al. (2024a), SymMPO Liu et al. (2025c), and OPA-DPO Yang et al. (2025). The preference data for these methods are constructed from Qwen2.5-VL-7B rollouts and annotated by Qwen3.6-27B Qwen Team (2026). The on-policy group includes GRPO Shao et al. (2024), DAPO Yu et al. (2025a), Dr. GRPO Liu et al. (2025d), and our SCAPO. All the on-policy methods are trained using our DOPA annotations. Complete training configurations are reported in Appendix C.1.

Evaluation Metrics. For discriminative evaluation, we report standard metrics including accuracy, precision, recall, and F1, as well as the benchmark-specific PhD-index. For generative evaluation, to enable the metric to simultaneously reflect the faithfulness and informativeness of the model output, we introduce a new metric Caption Score (Cap. Score) built on the harmonic mean of 1-Hal. Rate and Cover Rate, where Hal. Rate corresponds to CHAIR<sub>i</sub>. The formal definitions are given in Appendix C.2.

Table 1: Evaluation on generative and discriminative hallucination benchmarks. Cap. Score = HarMean(1-Hal. Rate, Cover Rate) jointly measures caption faithfulness and informativeness. (Cap. Score is computed with the original object-existence annotation of the dataset. Cap. Score<sup>†</sup> is computed with our expanded DOPA annotation.)
<table><tr><td></td><td colspan="2">AMBER</td><td colspan="2">MS COCO</td><td colspan="5">PhD</td></tr><tr><td>Method</td><td></td><td>Cap. Score ↑ Dis. Acc. ↑</td><td>Cap. Score ↑</td><td>Cap. Score† ↑</td><td>PhD-base ↑</td><td>PhD-sec ↑</td><td>PhD-icc ↑</td><td>PhD-ccs ↑</td><td>PhD-all ↑</td></tr><tr><td colspan="10">(i) Inference-time Methods</td></tr><tr><td>Qwen2.5-VL</td><td>76.5</td><td>83.9</td><td>73.2</td><td>64.2</td><td>76.8</td><td>68.2</td><td>62.1</td><td>59.5</td><td>66.6</td></tr><tr><td>VCD</td><td>76.5</td><td>83.8</td><td>72.7</td><td>65.9</td><td>77.0</td><td>71.1</td><td>63.6</td><td>65.2</td><td>69.2</td></tr><tr><td>OPERA</td><td>76.3</td><td>84.5</td><td>73.7</td><td>65.3</td><td>76.4</td><td>70.9</td><td>64.6</td><td>62.0</td><td>68.5</td></tr><tr><td>VTI</td><td>47.7</td><td>85.9</td><td>50.2</td><td>33.6</td><td>79.0</td><td>57.3</td><td>58.3</td><td>66.8</td><td>65.4</td></tr><tr><td colspan="10">(ii) Off-policy Methods</td></tr><tr><td>DPO</td><td>77.7</td><td>71.8</td><td>74.8</td><td>70.0</td><td>70.8</td><td>52.9</td><td>52.4</td><td>67.9</td><td>61.0</td></tr><tr><td>mDPO</td><td>76.3</td><td>82.9</td><td>74.2</td><td>66.5</td><td>77.0</td><td>70.0</td><td>66.7</td><td>67.4</td><td>70.3</td></tr><tr><td>SymMPO</td><td>75.5</td><td>85.4</td><td>74.2</td><td>65.4</td><td>80.2</td><td>62.8</td><td>55.6</td><td>73.3</td><td>68.0</td></tr><tr><td>OPA-DPO</td><td>67.8</td><td>85.4</td><td>72.2</td><td>53.9</td><td>76.1</td><td>70.9</td><td>52.7</td><td>69.5</td><td>67.3</td></tr><tr><td colspan="10">(iii) On-policy Methods</td></tr><tr><td>GRPO</td><td>69.9</td><td>83.8</td><td>66.5</td><td>51.4</td><td>77.2</td><td>63.9</td><td>63.5</td><td>61.8</td><td>66.6</td></tr><tr><td>DAPO</td><td>61.1</td><td>83.1</td><td>51.8</td><td>33.3</td><td>77.1</td><td>57.9</td><td>59.8</td><td>63.3</td><td>64.5</td></tr><tr><td>Dr. GRPO</td><td>68.3</td><td>84.1</td><td>65.1</td><td>47.4</td><td>76.8</td><td>56.2</td><td>62.5</td><td>62.7</td><td>64.5</td></tr><tr><td>SCAPO (Ours)</td><td>82.8</td><td>87.4</td><td>79.6</td><td>78.7</td><td>80.3</td><td>74.5</td><td>67.5</td><td>68.2</td><td>72.6</td></tr></table>

Table 2: Effect of the DOPA dataset used for on-policy training. Both variants below utilize the same SCAPO objective but derive rewards from different annotations.
<table><tr><td></td><td colspan="2">MS COCO Label</td><td colspan="2">DOPA Label</td><td colspan="2">AMBER</td><td colspan="2">MMHal-Bench</td></tr><tr><td>Method</td><td>Hal. Rate ↓</td><td>Cover Rate ↑</td><td>Hal. Rate ↓</td><td>Cover Rate ↑</td><td>Hal. Rate ↓</td><td>Cover Rate ↑</td><td>Score ↑</td><td>Hal. Rate ↓</td></tr><tr><td>Qwen2.5-VL (Baseline)</td><td>15.1</td><td>64.3</td><td>11.8</td><td>50.5</td><td>4.9</td><td>64.0</td><td>3.26</td><td>39.6</td></tr><tr><td>SCAPO Training</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/ MS COCÕ annotations</td><td>13.9</td><td>78.8</td><td>11.6</td><td>52.0</td><td>4.6</td><td>66.1</td><td>3.45</td><td>34.4</td></tr><tr><td>w/ DOPA annotations (Ours)</td><td>13.4</td><td>73.7</td><td>9.4</td><td>69.6</td><td>4.8</td><td>73.2</td><td>3.52</td><td>33.3</td></tr></table>

## 5.2 MAIN RESULTS

Table 1 presents the main results across three hallucination benchmarks. When simultaneously considering faithfulness and informativeness, most hallucination mitigation methods struggle to yield visible improvements in the Cap. Score metric. Aside from our SCAPO, only DPO achieves consistent gains across the three generative settings. Notably, despite being trained on the same DOPA data as SCAPO, other GRPO-style on-policy methods yield Cap. Scores below the Qwen2.5-VL baseline. This degradation stems from an optimization shortcut toward global conservatism, which we analyze in detail in Section 5.4. In contrast, SCAPO effectively overcomes this bottleneck, achieving strong Cap. Scores of 82.8% on AMBER and 79.6% on MS COCO.

Importantly, these generative improvements also transfer to discriminative tasks. SCAPO achieves an 87.4% discriminative accuracy on AMBER and the best results among the compared methods on PhD-base, PhD-sec, PhD-icc, and PhD-all. Overall, this consistent improvement across both generative and discriminative settings distinguishes SCAPO from alternatives that inflate an isolated metric at the expense of cross-benchmark performance.

## 5.3 EXPERIMENTS ON DOPA

Generalization via DOPA Annotations. Table 2 isolates the effect of the annotation source while keeping the SCAPO objective unchanged. We additionally incorporate the LLM-judge-based MMHal-Bench Sun et al. (2024) for rigorous cross-benchmark assessment. It can be observed that while training with sparse MS COCO annotations leads to substantial coverage improvements on its specific label set, the gains on other benchmarks are notably smaller, as claims outside this space remain unverifiable and unrewarded. In contrast, replacing these sparse labels with the dense DOPA dataset drives generalized improvements. It significantly boosts in-domain performance under the DOPA evaluation, reducing the Hal. Rate from 11.8% to 9.4% and raising the Cover Rate from 50.5% to 69.6%. Furthermore, it exhibits superior cross-dataset generalization, maintaining an overall advantage over the sparsely annotated counterpart on AMBER and MMHal-Bench.

![](images/1832b625af1a9f947268f5777010a9744b96031b529619b1e46e4c7af2019044.jpg)

Table 3: Annotation density on training-set rollouts. For each annotation setting, we sample eight responses per training example and report the per-response averages. “Sup.” and “Hal.” denote ground-truth-supported and hallucinated content, respectively.
<table><tr><td rowspan="2">Annotation</td><td colspan="3">Annotated Objects per Response</td><td colspan="3">Subsentences per Response</td></tr><tr><td>Avg.</td><td>Sup.</td><td>Hal.</td><td>Avg.</td><td>Sup.</td><td>Hal.</td></tr><tr><td>MS COCO</td><td>2.73</td><td>2.32</td><td>0.41</td><td>6.33</td><td>2.62</td><td>0.45</td></tr><tr><td>DOPA (Ours)</td><td>5.13</td><td>4.37</td><td>0.76</td><td>6.37</td><td>3.46</td><td>0.81</td></tr></table>

Figure 4: Training dynamics under different credit-assignment strategies. Response-level objectives face a trade-off between limited coverage (when optimizing Hal. Rate) and increased hallucinations (when optimizing Cap. Score). Object-level assignment fragments semantic coherence, causing unstable increases in both metrics. In contrast, SCAPO’s subsentence-level credit assignment resolves these issues, steadily expanding valid coverage while keeping the hallucination rate controlled.

Reward Density Analysis. The gains from DOPA can be largely explained by its higher reward density, as demonstrated in Table 3. Compared to MS COCO annotations, DOPA nearly doubles the average number of annotated objects (5.13 vs. 2.73) and significantly increases the total number of reward-bearing subsentence labels. This richer provision of verifiable feedback effectively grounds the on-policy optimization, preventing valid object claims from going unrewarded.

## 5.4 ANALYSIS OF SCAPO

Training Dynamics. Figure 4 compares SCAPO against alternative credit assignment strategies to illustrate their distinct failure modes. Optimizing solely for a response-level Hal. Rate penalty triggers the conservative “saying less” shortcut, causing Cover Rate to plummet despite dropping Hal. Rate to 4%. Conversely, optimizing for a response-level Cap. Score fails to effectively suppress hallucinations, raising Cover Rate but driving Hal. Rate up to 17%. Furthermore, an object-level variant fragments semantic coherence, yielding a highly unstable curve where both metrics increase. SCAPO overcomes these pitfalls through subsentence-level credit assignment. By evaluating claims within their natural semantic boundaries, SCAPO steadily improves Cover Rate while keeping Hal. Rate firmly controlled, successfully avoiding both global conservatism and excessive hallucinations.

Output Factual Richness. Because Cover Rate is bounded by a predefined vocabulary, we further evaluate open-vocabulary performance using the FaithScore benchmark Jing et al. (2024). This benchmark employs an LLM judge to decompose responses into atomic facts and reports two key metrics: the FaithScore metric (which measures overall faithfulness, where higher indicates fewer hallucinations) and Avg. Facts (which measures informativeness by counting the average number of atomic facts per response). As illustrated in Figure 5, response-level GRPO-style methods achieve high FaithScores by exploiting a conservative shortcut: they severely curtail their outputs across all four evaluation settings. For instance, under the detailed setting, the Avg. Facts for GRPO and DAPO drop to 31.65 and 24.31, respectively, compared to 41.63 for the Qwen2.5-VL backbone. In contrast, SCAPO secures a superior FaithScore while simultaneously preserving the Avg. Facts of the backbone across all settings. This provides compelling evidence that SCAPO’s improvements stem from precise hallucination suppression rather than global output impoverishment.

![](images/3f9b668ea9a1753adfe4ed64e6b4bdbe13a510c475d4f79e6af30ba8cdc5299a.jpg)  
Figure 5: FaithScore and Avg. facts across four generation settings. GRPO-style methods all improve faithfulness at the expense of the number of atomic facts in the output, while SCAPO preserves factual richness and maintains high faithfulness.

Table 4: Progressive ablation on the AMBER discriminative task. We first augment the Qwen2.5- VL baseline with a self-generated image description, and then upgrade the underlying backbone to the SCAPO-trained policy to evaluate the impact of generation quality.
<table><tr><td>Configuration</td><td>Acc. ↑</td><td>Pre. ↑</td><td>Rec. ↑</td><td>F1↑</td></tr><tr><td> $\mathrm { Q w e n } 2 . 5 – \mathrm { V L }$ </td><td>83.9</td><td>84.5</td><td>92.6</td><td>88.4</td></tr><tr><td> $+ \operatorname { D e s c . } \ \mathrm { A u g . }$ </td><td> $8 6 . 9 ( + 3 . 0 ) $ </td><td> $8 7 . 3 \left( + 2 . 8 \right)$ </td><td> $9 3 . 8 ( + 1 . 2 ) $ </td><td> $9 0 . 4 ( + 2 . 0 ) $ </td></tr><tr><td> $+ \ S C A P O$ </td><td> ${ \pmb 8 7 . 4 } \left( + 3 . 5 \right)$ </td><td> ${ \pmb 8 7 . 6 ( + 3 . 1 ) }$ </td><td> $\mathbf { 9 4 . 4 } \left( + 1 . 8 \right)$ </td><td> $\mathbf { 9 0 . 9 } _ { ( + 2 . 5 ) }$ </td></tr></table>

## 5.5 DESCRIPTION-AUGMENTED DISCRIMINATIVE INFERENCE

Table 4 presents a progressive ablation of the description-augmented discriminative inference. First, adding a generated description to the direct-answering baseline improves the Qwen2.5-VL backbone’s accuracy from 83.9% to 86.9% and its F1 score from 88.4% to 90.4%, confirming the general utility of an explicit auxiliary visual representation. Next, substituting the generative backbone with our SCAPO-aligned model further elevates accuracy to 87.4% and F1 to 90.9%. These progressive gains indicate that the effectiveness of description augmentation depends on the quality of the generated visual context. By supplying denser valid evidence and strictly fewer misleading priors, SCAPO transfers its generative grounding improvements to downstream discrimination, requiring zero task-specific fine-tuning.

## 6 CONCLUSION

In this paper, we investigate the conservative “saying less” shortcut that plagues on-policy hallucina tion mitigation, attributing this degradation to sparse reward supervision and coarse response-level credit assignment. To comprehensively resolve these bottlenecks, we introduce a dual-level finegrained alignment framework. At the data level, we construct the DOPA dataset as a dense and reliable reward oracle; at the algorithmic level, we propose SCAPO for precise subsentence-level credit assignment. Extensive experiments across multiple hallucination benchmarks demonstrate that our approach effectively resolves the faithful-informative trade-off, substantially reducing unsupported content while expanding the coverage of valid object claims. Furthermore, we show that these faithful and informative descriptions serve as highly effective auxiliary context, successfully transferring our generative gains to downstream discriminative tasks.

## AI USE STATEMENT

In this work, we used generative AI tools for dataset construction, which we disclose as part of the required AI-use disclosure. Specifically, GPT-5.5 (OpenAI, 2026) was used to draft a canonical object vocabulary, and Qwen3.6-27B (Qwen Team, 2026) was used to generate the initial and refined object-existence annotations for DOPA. We did not use generative AI to generate research hypotheses, design the proposed method, or derive scientific conclusions. In addition, we used generative AI tools for recommended-disclosure tasks, including language polishing of author-written text and assistance with routine coding. All AI-assisted text was reviewed for consistency with the authors intended meaning, and AI-assisted code was inspected and tested by the authors. We take responsi bility for the final content of this work, including all text, claims, code, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

We take several steps to facilitate reproducibility. The appendix provides the complete formalization of our reward construction and optimization procedure, including the claim-set definitions and pseudocode for SCAPO (Appendix B). We report the full experimental configuration, including training hyperparameters and environment, in Appendix C.1. All prompts used for data annotation, training, and inference are provided verbatim in Appendix D.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025b.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. MME: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2025.

Xuan Gong, Tianshi Ming, Xinpeng Wang, and Zhihua Wei. DAMRO: Dive into the attention mechanism of LVLM to reduce object hallucination. In Conference on Empirical Methods in Natural Language Processing, pp. 7696–7712. Association for Computational Linguistics, 2024.

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. OPERA: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13418–13427, 2024.

Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. In Findings of the Association for Computational Linguistics: EMNLP, pp. 5042–5063, 2024.

Wei Lan, Wenyi Chen, Qingfeng Chen, Shirui Pan, Huiyu Zhou, and Yi Pan. A survey of hallucination in large visual language models. arXiv preprint arXiv:2410.15359, 2024.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13872– 13882, 2024.

Wei Li, Zhen Huang, Houqiang Li, Le Lu, Yang Lu, Xinmei Tian, Xu Shen, and Jieping Ye. Visual evidence prompting mitigates hallucinations in large vision-language models. In Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4048–4080. Association for Computational Linguistics, 2025.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, pp. 292–305, 2023.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In ´ European Conference on Computer Vision, pp. 740–755. Springer, 2014.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pp. 34892–34916, 2023.

Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. PhD: A ChatGPT-prompted visual hallucination evaluation dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19857–19866, 2025a.

Sheng Liu, Haotian Ye, and James Zou. Reducing hallucinations in large vision-language models via latent space steering. In International Conference on Learning Representations, pp. 72402– 72419, 2025b.

Wenqi Liu, Xuemeng Song, Jiaxi Li, Yinwei Wei, Na Zheng, Jianhua Yin, and Liqiang Nie. Mitigating hallucination through theory-consistent symmetric multimodal preference optimization. In Advances in Neural Information Processing Systems, volume 38, pp. 111259–111284, 2025c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pp. 216–233, 2024b.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025d.

OpenAI. GPT-5.5 model. https://developers.openai.com/api/docs/models/gpt-5.5, 2026. Accessed: 2026-07-29.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. In Advances in Neural Information Processing Systems, volume 37, pp. 116617–116637, 2024.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model. https://qwen.ai/blog?id= qwen3.6-27b, April 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Ad vances in Neural Information Processing Systems, volume 36, 2023.

Anku Rani, Vipula Rawte, Harshad Sharma, Neeraj Anand, Krishnav Rajbangshi, Amit Sheth, and Amitava Das. Visual hallucination: Definition, quantification, and prescriptive remediations. arXiv preprint arXiv:2403.17306, 2024.

Pritam Sarkar, Sayna Ebrahimi, Ali Etemad, Ahmad Beirami, Sercan O. Arik, and Tomas Pfister. Mitigating object hallucination in mllms via data-augmented phrase-level alignment. In Interna tional Conference on Learning Representations, 2025.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented RLHF. In Findings of the Association for Computational Linguistics: ACL, pp. 13088–13110, 2024.

Fei Wang, Wenxuan Zhou, James Y. Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mDPO: Conditional preference optimization for multimodal large language models. In Conference on Empirical Methods in Natural Language Processing, pp. 8078–8088, 2024a.

Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and Jitao Sang. AMBER: An LLM-free multi-dimensional benchmark for MLLMs hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.

Xintong Wang, Jingheng Pan, Liang Ding, and Chris Biemann. Mitigating hallucinations in large vision-language models with instruction contrastive decoding. In Findings of the Association for Computational Linguistics: ACL, pp. 15840–15853. Association for Computational Linguistics, 2024b.

Wenyi Xiao, Ziwei Huang, Leilei Gan, Wanggui He, Haoyuan Li, Zhelun Yu, Fangxun Shu, Hao Jiang, and Linchao Zhu. Detecting and mitigating hallucination in large vision language models via fine-grained AI feedback. In AAAI Conference on Artificial Intelligence, volume 39, pp. 25543–25551, 2025.

Weihao Xuan, Qingcheng Zeng, Heli Qi, Junjue Wang, and Naoto Yokoya. Seeing is believing, but how much? a comprehensive analysis of verbalized calibration in vision-language models. In Conference on Empirical Methods in Natural Language Processing, pp. 1408–1450, 2025.

Zhihe Yang, Xufang Luo, Dongqi Han, Yunjian Xu, and Dongsheng Li. Mitigating hallucinations in large vision-language models via DPO: On-policy data hold the key. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10610–10620, 2025.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. Science China Information Sciences, 67(12):220105, 2024.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244, 2025a.

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. RLHF-V: Towards trustworthy MLLMs via behavior alignment from fine-grained correctional human feedback. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13807–13816, 2024a.

Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, et al. RLAIF-V: Open-source AI feedback leads to super GPT-4V trustworthiness. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19985–19995, 2025b.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. In International Conference on Machine Learning, 2024b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Linxi Zhao, Yihe Deng, Weitong Zhang, and Quanquan Gu. Mitigating object hallucination in large vision-language models via image-grounded guidance. In International Conference on Machine Learning, pp. 77461–77486, 2025.

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond hallucinations: Enhancing LVLMs through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023.

Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024a.

Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. Analyzing and mitigating object hallucination in large vision-language models. In International Conference on Learning Representations, pp. 56969–56998, 2024b.

## A DOPA DATASET STATISTICS AND SPLIT DETAILS

Table 5 summarizes the overall statistics of DOPA. The dataset contains 5,000 images, each exhaustively annotated against a vocabulary V of 160 canonical object labels. To support matching free-form object mentions in model generations, the vocabulary is further associated with an alias map A containing 894 surface forms, with an average of 5.59 aliases per canonical label. Each image contains, on average, 10.75 positively annotated concepts. Importantly, DOPA is constructed exclusively from the training-side image pool and is strictly separated from all evaluation splits. For the Cap. Score<sup>†</sup> setting in the experiments, evaluation images are independently annotated using the same DOPA vocabulary and verification protocol; these annotations are used solely for evaluation and are never used during training.

Table 5: Overall statistics of the DOPA dataset.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of images</td><td>5,000</td></tr><tr><td>Canonical object labels (|V|)</td><td>160</td></tr><tr><td>Aliases (|A|)</td><td>894</td></tr><tr><td>Average aliases per canonical label</td><td>5.59</td></tr><tr><td>Average positive annotations per image</td><td>10.75</td></tr></table>

The occurrence frequencies of the canonical labels exhibit a pronounced long-tailed distribution (Table 6). Nearly half of the vocabulary (76 labels, 47.5%) appears in fewer than 100 images. In contrast, only 14 labels (8.8%) appear in at least 1,000 images. This distribution reflects that beyond frequently occurring objects, DOPA retains a substantial number of less frequent concepts that can nevertheless appear in model generations and therefore require reliable reward supervision.

Table 6: Label distribution according to the number of images in which they are annotated as present.
<table><tr><td>Number of positive images</td><td># Labels</td><td>Percentage</td></tr><tr><td>&lt; 100</td><td>76</td><td>47.5%</td></tr><tr><td>100-499</td><td>59</td><td>36.9%</td></tr><tr><td>500-999</td><td>11</td><td>6.9%</td></tr><tr><td>≥ 1,000</td><td>14</td><td>8.8%</td></tr><tr><td>Total</td><td>160</td><td>100%</td></tr></table>

## B FULL FORMALIZATION OF SCAPO

## B.1 CLAIM SETS AND PROBLEM FORMULATION

This section details the formal definitions and mathematical formulations omitted from the main text for brevity. Given an image x and a prompt p, we sample G responses from the current policy snapshot:

$$
y _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x , p ) , \qquad i = 1 , \ldots , G ,\tag{3}
$$

and segment each response at subsentence boundaries:

$$
y _ { i } = [ s _ { i , 1 } , s _ { i , 2 } , . . . , s _ { i , M _ { i } } ] .\tag{4}
$$

For each subsentence $s _ { i , j }$ we extract its object mentions and map them to canonical labels through the vocabulary V and the alias map A. Consulting the DOPA object-existence matrix $E ^ { * }$ for image x, we partition the matched labels into the supported set and the hallucinated set:

$$
P _ { i , j } ^ { + } = \{ \ell : \ell \mathrm { ~ m e n t i o n e d ~ i n ~ } s _ { i , j } , E _ { \ell , x } ^ { * } = 1 \} , \qquad P _ { i , j } ^ { - } = \{ \ell : \ell \mathrm { ~ m e n t i o n e d ~ i n ~ } s _ { i , j } , E _ { \ell , x } ^ { * } = 0 \} .\tag{5}
$$

Because DOPA annotates every label in V as either present or absent for every image, this partition is total over in-vocabulary mentions: there is no third “unknown” category, which is exactly the

ambiguity that positive-only annotation sources leave unresolved. Entities that do not map into V are excluded from judgment, so a subsentence whose only object mentions are out-of-vocabulary has $n _ { i , j } ^ { \mathrm { n e w } } = n _ { i , j } ^ { \mathrm { r e p } } = \stackrel {  } { n _ { i , j } ^ { \mathrm { h a l } } } = 0$ and falls into the “no objects” branch of Eq. (1).

To reward informativeness without rewarding repetition, we track the supported objects already asserted earlier in the same response via a history set

$$
H _ { i , j } = \bigcup _ { k = 1 } ^ { j - 1 } P _ { i , k } ^ { + } , \qquad { \mathrm { w i t h ~ } } H _ { i , 1 } = \emptyset ,\tag{6}
$$

which splits the current supported claims into novel and repeated claims:

$$
\begin{array} { r } { P _ { i , j } ^ { \mathrm { n e w } } = P _ { i , j } ^ { + } \setminus H _ { i , j } , \qquad P _ { i , j } ^ { \mathrm { r e p } } = P _ { i , j } ^ { + } \cap H _ { i , j } . } \end{array}\tag{7}
$$

The counts appearing in Eq. (1) are then simply the cardinalities of these sets:

$$
n _ { i , j } ^ { \mathrm { n e w } } = | P _ { i , j } ^ { \mathrm { n e w } } | , \qquad n _ { i , j } ^ { \mathrm { r e p } } = | P _ { i , j } ^ { \mathrm { r e p } } | , \qquad n _ { i , j } ^ { \mathrm { h a l } } = | P _ { i , j } ^ { - } | .\tag{8}
$$

Note that $H _ { i , j }$ is accumulated over supported claims only and is reset for each response, so the novelty bonus is defined strictly within each sampled response. The optimization target is thus to shrink $P _ { i , j } ^ { - } ,$ expand $P _ { i , j } ^ { \mathrm { n e w } }$ , and discourage excessive $P _ { i , j } ^ { \mathrm { r e p } }$ , jointly guiding the model toward highly informative and faithful generations.

## B.2 THE SCAPO OPTIMIZATION STEP

Figure 6 outlines the complete pseudocode for a single SCAPO optimization step. Crucially, the MATCHLABELS procedure relies entirely on a deterministic lookup against the vocabulary V and the alias map A, rather than invoking an external evaluation model. This design computationally decouples the rollout scoring latency from the heavy judge model used during DOPA construction. By utilizing DOPA as a reusable reward oracle, SCAPO enables highly efficient, dense per-subsentence reward assignment directly within the on-policy training loop.

## C IMPLEMENTATION DETAILS

## C.1 TRAINING CONFIGURATION

Table 7 details the hyperparameters and configurations required to fully reproduce the SCAPO training process. To ensure a fair and rigorous comparison, all on-policy baselines (e.g., GRPO, DAPO, Dr. GRPO) share this identical configuration space where applicable, differing strictly in their respective advantage formulations.

## C.2 EVALUATION METRIC DEFINITIONS

This section provides the formal definition of the Caption Score metric. For each evaluation sample i, let $\mathcal { O } _ { i } ^ { \mathrm { g e n } }$ denote the set of in-vocabulary object labels mentioned in the generated description, and let $\mathcal { O } _ { i } ^ { \mathrm { g t } }$ denote the set of object labels annotated as present in the corresponding image. We compute Hal. Rate and Cover Rate by aggregating object counts over the entire evaluation set:

$$
\mathrm { H a l . ~ R a t e } = \frac { \sum _ { i } | { \mathcal { O } } _ { i } ^ { \mathrm { g e n } } \setminus { \mathcal { O } } _ { i } ^ { \mathrm { g t } } | } { \sum _ { i } | { \mathcal { O } } _ { i } ^ { \mathrm { g e n } } | } , \qquad \mathrm { C o v e r ~ R a t e } = \frac { \sum _ { i } | { \mathcal { O } } _ { i } ^ { \mathrm { g e n } } \cap { \mathcal { O } } _ { i } ^ { \mathrm { g t } } | } { \sum _ { i } | { \mathcal { O } } _ { i } ^ { \mathrm { g t } } | } .\tag{9}
$$

Thus, both metrics are computed using micro-averaged object counts across the full evaluation set. Hal. Rate measures faithfulness (lower is better), whereas Cover Rate measures informativeness (higher is better).

Reporting these two numbers separately is what allows the “saying less” shortcut to hide: a model can drive Hal. Rate toward zero simply by emitting almost no object claims, which looks like an unambiguous win if Cover Rate is not read alongside it. We therefore define the Caption Score as their harmonic mean:

$$
\mathrm { C a p . ~ S c o r e = H a r M e a n ( 1 - H a l . ~ R a t e , ~ C o v e r ~ R a t e ) } = \frac { 2 \left( 1 - \mathrm { H a l . ~ R a t e } \right) \cdot \mathrm { C o v e r ~ R a t e } } { \left( 1 - \mathrm { H a l . ~ R a t e } \right) + \mathrm { C o v e r ~ R a t e } } ,\tag{10}
$$

Algorithm 1 One SCAPO optimization step   
Input: policy π<sub>θ</sub>; image–prompt batch B; DOPA matrix E<sup>∗</sup>; vocabulary V; alias map A; coefficients   
$r _ { g } , r _ { \mathrm { r e p } } , r _ { h } , r _ { \mathrm { r e g } } ;$ weighting functions $\lambda _ { g } , \lambda _ { h } ;$ group size G; clip range ϵ   
Output: updated parameters θ   
1 $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta }$ //freeze the rollout policy   
2 $\mathcal { D }  \emptyset \quad \mathcal { \hat { H } }$ rollout bufferfor this step   
3 for each $( x , p ) \in B$ do   
4 for $i = 1$ to $G$ do   
5 $y _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x , p )$   
6 $[ \ldots , \ldots , { s } _ { i , M _ { i } } ] \gets \mathbf { S } \mathbf { F }$ LITSUBSENTENCE $\mathfrak { s } ( y _ { i } )$   
7 $\mathbf { \dot { \{ T } }  _ { i , j } \} \gets \mathbf { M } _ { I }$ APTOTOKENSP $\operatorname { a N S } ( y _ { i } , \{ s _ { i , j } \} ) ^ { \sim } ~ / \prime$ spans partition the response   
8 $\grave { H } \gets \emptyset \quad / /$ supported objects seen so far in this rollout   
9 for $j = 1$ to $M _ { i }$ do   
10 O ← MATCHLABEL $s ( s _ { i , j } , \gamma , \mathcal { A } )$   
11 $P ^ { + }  \{ \ell \in O : E _ { \ell , x } ^ { * } = \tilde { 1 } \} ; \quad P ^ { - }  \{ \ell \in O : E _ { \ell , x } ^ { * } = 0 \}$   
12 $n ^ { \mathrm { n e w } } \gets \backslash P ^ { + } \setminus H | ; \quad n ^ { \mathrm { r e p } } \gets | P ^ { + } \cap \dot { H } | ; \quad n ^ { \mathrm { h a l } } \gets | P ^ { - } |$   
13 if $n ^ { \mathrm { h a l } } > 0$ then $r _ { i , j } \gets - r _ { h } \lambda _ { h } ( n ^ { \mathrm { h a l } } )$   
14 else if $O = \emptyset$ then $r _ { i , j } \gets - r _ { \mathrm { r e g } }$   
15 else $r _ { i , j }  r _ { g } \lambda _ { g } ( n ^ { \mathrm { n e \breve { w } } } ) + r _ { \mathrm { r e p } } \breve { \lambda } _ { g } ( n ^ { \mathrm { r e p } } )$   
16 $H  \bar { H } \cup \bar { P ^ { + } } \quad / \prime$ only supported objects enter the history   
17 $A _ { i , t } \gets r _ { i , j }$ for all $t \in \mathsf { T } _ { i , j }$ // no group normalization   
18 end for   
19 $\mathcal { D }  \mathcal { D } \cup \{ ( x , p , y _ { i } , \{ A _ { i , t } \} ) \}$   
20 end for   
21 end for   
22 $\begin{array} { r } { N _ { \mathrm { t o k } }  \sum _ { ( x , p , y _ { i } , \cdot ) \in \mathcal { D } } | y _ { i } | } \end{array}$   
23 for each inner update epoch do   
24 $\rho _ { i , t }  \pi _ { \theta } ( y _ { i , t } \mid \cdot ) \bar { / } \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid \cdot )$   
25 $\mathcal { L } \gets - \frac { 1 } { N _ { \mathrm { t o k } } } \sum _ { i \neq } \operatorname* { m i n } \bigl ( \rho _ { i , t } A _ { i , t } , \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i , t } \bigr )$   
i,t   
26 $\theta \gets \theta - \eta \nabla _ { \theta } \dot { \mathcal { L } }$   
27 end for   
28 return θ  
Figure 6: Pseudocode for one SCAPO optimization step.

where all quantities are in [0, 1] and the score is reported as a percentage. Degenerate conservatism is thus penalized structurally rather than by convention. As in the main text, Cap. Score is computed against the evaluation dataset’s original annotations, and Cap. Score<sup>†</sup> against our expanded DOPA annotations.

## C.3 DESCRIPTION-AUGMENTED DISCRIMINATIVE INFERENCE

This section provides the formal formulation of the description-augmented inference used in Section 4.3. Given an image x, the SCAPO-aligned policy first generates a faithful description d conditioned on the description prompt p<sub>desc</sub>:

$$
d \sim \pi _ { \theta } ( \cdot \mid x , p _ { \mathrm { d e s c } } ) .\tag{11}
$$

For a subsequent discriminative query $q ,$ the model then conditions its final answer on the original image and the generated description jointly:

$$
a \sim \pi _ { \theta } ( \cdot \mid x , d , q ) ,\tag{12}
$$

where a represents the target discriminative response. Importantly, the visual input x is preserved during this discriminative phase; the generated text $d$ acts strictly as an enriched auxiliary context rather than a substitute for the visual input. This design ensures that the model retains direct access to the source image to resolve queries about minor details potentially omitted from $d .$ The exact prompt template integrating x, d, and q is detailed in Appendix D. Crucially, both inference stages utilize the same policy weights θ, seamlessly executing this generative-to-discriminative transfer with zero task-specific fine-tuning.

Table 7: SCAPO training configuration. Values apply to all reported SCAPO results unless stated otherwise.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td></td></tr><tr><td>Backbone</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Tuning scheme</td><td>Full fine-tuning</td></tr><tr><td>Vision tower</td><td>Frozen</td></tr><tr><td>Parameter dtype</td><td>bfloat16</td></tr><tr><td>Training</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Prompts per optimization step</td><td>128</td></tr><tr><td>Rollouts per prompt</td><td>8</td></tr><tr><td>Mini-batch size</td><td>128 (equal to the batch size, so one update per batch)</td></tr><tr><td>Micro-batch size per GPU</td><td>2</td></tr><tr><td>Inner update epochs per batch</td><td>1</td></tr><tr><td>Epochs over the training set</td><td>8</td></tr><tr><td>Total optimization steps</td><td>312 ([5000/128] = 39 steps per epoch)</td></tr><tr><td>Clipping coefficient €</td><td>0.2</td></tr><tr><td>KL loss coefficient</td><td>0.01</td></tr><tr><td>SCAPO Reward</td><td></td></tr><tr><td>Novel-supported coefficient  $r _ { g }$ </td><td>1.0</td></tr><tr><td>Repeated-supported coefficient  $r _ { \mathrm { r e p } }$ </td><td>0</td></tr><tr><td>Hallucination coefficient  $r _ { h }$ </td><td>1.0</td></tr><tr><td>No-object penalty  $r _ { \mathrm { r e g } }$ </td><td>0.1</td></tr><tr><td>Informativeness scaling  $\lambda _ { g } ( n )$ </td><td>min(n, 1)</td></tr><tr><td>Hallucination scaling  $\lambda _ { h } ( n )$ </td><td>min(n, 1)</td></tr><tr><td></td><td></td></tr><tr><td>Environment</td><td></td></tr><tr><td>RL framework</td><td>verl 0.8.0.dev</td></tr><tr><td>Orchestration</td><td>Ray 2.54.0</td></tr><tr><td>Core software</td><td>PyTorch 2.9.1, transformers 4.57.1, vLLM 0.12.0,</td></tr><tr><td>Text processing</td><td>CUDA 12.8, FlashAttention 2.8.3 spaCy 3.8.11 (en_core_web_1g), NLTK 3.9.3</td></tr></table>

## D PROMPTS

## D.1 COARSE ANNOTATION PROMPT

This section provides the verification prompt used to construct the initial object-existence matrix $E ^ { ( 0 ) }$ (see Section 3.2). It is issued once per image–label pair, with {object} bound to a canonical label, and returns a single binary verdict.

=== SYSTEM ===   
You are a strict visual existence verifier.   
Return only valid JSON. Do not include markdown, explanations, reasoning, or code   
fences.   
=== USER ===   
Inspect the image and determine whether this exact entity is clearly visible:   
Entity: {object}   
Rules:   
- Return "supported" if the entity itself is visible enough to identify in the image.   
- Small or background instances still count when they are identifiable; do not require   
the entity to be salient or large.   
- If it is too tiny, blurry, heavily occluded, cut off, only implied by context, or only   
probably present, return "unsupported".   
Do not count a related object, subtype, parent class, text mention, or background   
knowledge unless the requested entity itself is visible.   
Do not count the input image/photo itself as "picture"; only count a picture, poster,   
painting, or framed image visible inside the scene.

- Clothing and accessories count only when that item itself is visible, not merely   
because a person is present.   
- Scene labels such as sky, cloud, grass, mountain, beach, street, sidewalk, court, or   
island count only when that visible scene region is clear.   
- If unsure, return "unsupported".   
Output exactly this JSON schema:   
{"verification": "supported" or "unsupported"}

## D.2 LABEL-SPECIFIC VERIFICATION PROMPTS

As discussed in Section 3.3, the initial coarse annotation can introduce systematic ambiguities for concepts with inherently vague visual boundaries. To resolve these edge cases, we replace the general-purpose prompt with tailored, label-specific instructions. These refined prompts provide rigorous operational definitions, explicit inclusion and exclusion criteria, disambiguation guidelines against semantically adjacent categories, and strict visual evidence requirements. We provide the tailored prompt for the book category below as a representative example.

=== SYSTEM ===   
You are a strict visual object-label verifier.   
Return only valid JSON. Do not include markdown, explanations, reasoning, or code   
fences.   
=== USER ===   
Inspect the image and decide whether the normalized label "book" is clearly visible.   
Definition:   
- "book" means a visible physical book-family reading, writing, or bound document   
object.   
- A book can count from its cover, spine, bound page block, or open pages.   
Do not count:   
- A laptop, tablet, phone, television, e-reader, or other screen displaying text or an   
image of pages when no physical book-family object is visible.   
- A framed picture, poster, sign, map, calendar, loose photograph, product box, card,   
envelope, or isolated sheet/stack of ordinary paper that is not identifiable as one of   
the included document types.   
A menu, restaurant bill, receipt, ticket, label, or packaging leaflet unless it is   
clearly a booklet/brochure.   
Printed book cover art or a picture of a book on another surface.   
A bookshelf, bookcase, library, desk, reader, or writer alone. The physical   
book-family object itself must be visible.   
Visibility requirement:   
- The object must be directly visible enough to identify a cover/spine/pages, binding,   
or folded publication. Legible text is not required.   
Books on a shelf can count when one or more actual spines or page blocks are   
recognizable; shelf-like colored rectangles alone are insufficient.   
An open book can count even when only its pages and central binding are visible.   
Small, distant, heavily blurred, or mostly hidden rectangular objects do not count   
when they could just as plausibly be boxes, screens, or loose paper.   
If unsure whether the object is a physical book-family item, mark unsupported.   
Output exactly this JSON schema:   
{   
"book": {"verification": "supported" or "unsupported",   
"evidence": "short visual evidence or empty string"}   
}

For visual concepts that are highly susceptible to mutual confusion (e.g., knife/fork, shirt/jacket, and chair/bench), independent evaluations often yield inconsistent results. We address this by verifying these pairs jointly within a single discriminative prompt. By explicitly contrasting their defining features simultaneously, we enforce strict semantic boundaries and eliminate classification overlap. The joint verification prompt for chair/bench is provided below as an example.

=== SYSTEM ===   
You are a strict visual object-label verifier.   
Return only valid JSON. Do not include markdown, explanations, reasoning, or code   
fences.   
=== USER ===

Inspect the image and decide whether these two labels are clearly visible:

## 1. chair 2. bench

Definitions: "chair" means a visible chair-like seat intended for one person, or a clearly separable individual seat unit. Count ordinary chairs, dining chairs, office chairs, folding chairs, stools, bar stools, armchairs, recliners, bean bag chairs, ottomans/footstools used as a seat-like furniture item, booster seats, visible vehicle seats, theater seats, stadium seats, and spectator seats.

\- Count a partially covered or occupied single-person chair/armchair/recliner when enough structure is still visible to identify it, such as arms on the sides, a backrest, legs/base, frame, or a distinct one-person seat outline.

\- "bench" means a visible long or shared seat intended for more than one person, usually without clearly separated individual chair units.

\- Count park benches, wooden or metal benches, pews, bleachers, dugout benches, picnic-table benches, long waiting-area benches, and bench-like seats used by people or posed objects.

Do not count:   
- For chair: do not count benches, pews, bleachers, couches, sofas, loveseats, futons, beds, cribs, toilets, tables, desks, counters, shelves, boxes, stools used only as tables/stands, or generic surfaces just because a person could sit on them.   
- For bench: do not count individual chairs, stools, armchairs, recliners, couches, sofas, beds, cribs, toilets, tables, counters, shelves, boxes, ledges, walls, railings, stairs, or generic flat surfaces just because people could sit on them.   
- Do not infer either label from a person merely sitting, standing, riding, or posing. The actual chair or bench structure must be visible.   
- A vehicle, stadium, theater, restaurant, park, or dining area does not support either label unless the actual seat structure is visible.   
- Text, logos, drawings, pictures, reflections, shadows, or printed patterns of chairs or benches support neither label.

## Important distinction:

\- Judge "chair" and "bench" independently. First look for any individual chair-like seats anywhere in the image, then separately look for any shared bench-like seats. The presence of a bench must not cause you to ignore a visible chair, and the presence of a chair must not cause you to ignore a visible bench.

\- A row of clearly separated stadium/theater/vehicle seats supports "chair"; it supports "bench" only if the seating is one continuous bench-like surface without individual seat units.

\- A couch/sofa/loveseat supports the separate "couch" label, not "chair" or "bench", even if one or more people are sitting on it.

\- A stool or ottoman can support "chair" only when it is visible as a one-person seat-like object; do not count a tiny unclear footrest or a stool acting only as a plant/table stand.

\- It is possible for both labels to be supported when the image contains both individual chairs and a separate bench.

## Visibility requirement:

\- The chair or bench itself must be directly visible enough to identify, such as a seat surface, backrest, legs/base, arms, frame, slats, or a clearly recognizable seating unit.

\- Small or background seats can count only when the individual-chair versus shared-bench structure is still identifiable.

\- For background seating areas, if both separate folding/camping/stadium-style chairs and long benches are visible, mark both labels supported and cite evidence for each. - If the supposed chair or bench is tiny, blurry, heavily occluded, cropped, hidden under a person or object, or only probably present, mark that label unsupported. - If unsure, mark unsupported.

```json
Output exactly this JSON schema:
{
"chair": {"verification": "supported" or "unsupported",
"evidence": "short visual evidence or empty string"},
"bench": {"verification": "supported" or "unsupported",
"evidence": "short visual evidence or empty string"}
}
```

## D.3 DESCRIPTION PROMPT p<sub>desc</sub>

We utilize two distinct description prompts across our experiments, each tailored to a specific evaluation phase.

For training and generative evaluation—specifically for the Cap. Score, Hal. Rate, and Cover Rate metrics reported in Table 1—we adopt a standard, unconstrained captioning instruction. This ensures strict consistency and fair comparability with prior literature.

Describe this image.

Conversely, for the description-augmented discriminative pipeline, the generated description serves as an enriched auxiliary context rather than a terminal evaluation target. Accordingly, we employ a refined prompt variant designed to elicit comprehensive details while explicitly constraining the model to objectively visible content:

Describe this image objectively and in detail. Focus only on visible content.

## D.4 DESCRIPTION-AUGMENTED DISCRIMINATIVE PROMPT

The prompt template utilized during the discriminative phase is shown below. Specifically, the generated description d is inserted to serve as an explicit auxiliary context. The original image is concurrently supplied alongside the query, ensuring the model retains full access to the primary visual evidence.

The same model previously described this image as follows:   
{generated\_description}   
Now answer the question based on the image and the description above.

## E ADDITIONAL EXPERIMENTS

## E.1 GENERAL CAPABILITY RETENTION

Optimizing a policy strictly for hallucination mitigation risks compromising its broader multimodal reasoning capabilities, a phenomenon commonly referred to as the alignment tax. To verify that our fine-grained credit assignment does not degrade foundational performance, we evaluate the SCAPOaligned policy across five comprehensive multimodal benchmarks: MME (Fu et al., 2025), MM-Bench (Liu et al., 2024b), MMMU (Yue et al., 2024), LLaVA-Bench (Liu et al., 2023), and MM-Vet (Yu et al., 2024b).

Table 8: Results on general multimodal benchmarks.
<table><tr><td rowspan="2">Method</td><td colspan="2">MME</td><td colspan="2">MMBench</td><td>MMMU</td><td>LLaVA-Bench</td><td>MM-Vet</td></tr><tr><td>Percep. ↑</td><td>Cognit. ↑</td><td>Circular ↑</td><td>Vanilla ↑</td><td>Acc. ↑</td><td>Rel. Score ↑</td><td>Total ↑</td></tr><tr><td>Qwen2.5-VL</td><td>1717.8</td><td>625.7</td><td>84.8</td><td>88.0</td><td>53.8</td><td>69.7</td><td>65.9</td></tr><tr><td>+ SCAPO (Ours)</td><td>1728.2</td><td>635.4</td><td>84.4</td><td>88.2</td><td>53.8</td><td>70.4</td><td>66.0</td></tr></table>

The results in Table 8 demonstrate that SCAPO effectively avoids the alignment tax commonly associated with specialized fine-tuning. Across the evaluated benchmarks, six of the seven metrics remain stable or exhibit slight improvements following our alignment process. This confirms that structurally penalizing local hallucinations does not compromise the target model’s foundational reasoning skills. Furthermore, we hypothesize that these marginal gains occur because explicitly rewarding valid visual claims positively reinforces the model’s overall visual grounding, translating to subtle benefits on broader multimodal comprehension tasks.

Table 9: Ablation of SCAPO’s components.
<table><tr><td>Variant</td><td>Avg. Score ↑</td><td>Hal. Rate ↓</td></tr><tr><td>Qwen2.5-VL</td><td>3.26</td><td>39.6</td></tr><tr><td>Additive hallucination penalty</td><td>3.48</td><td>33.3</td></tr><tr><td>Full reward for repeated claims</td><td>3.41</td><td>37.5</td></tr><tr><td>Without no-object regularization</td><td>3.46</td><td>34.4</td></tr><tr><td>SCAPO (Ours)</td><td>3.52</td><td>33.3</td></tr></table>

Table 10: Results with Qwen3-VL backbone. Cap. Score<sup>†</sup> is computed against the expanded DOPA annotations.
<table><tr><td rowspan="2">Method</td><td colspan="2">AMBER</td><td colspan="2">MS COCO</td><td>PhD</td></tr><tr><td>Cap. Score ↑</td><td>Dis. Acc. ↑</td><td>Cap. Score ↑</td><td>Cap. Score† ↑</td><td>PhD-all ↑</td></tr><tr><td>Qwen3-VL</td><td>76.0</td><td>86.6</td><td>72.8</td><td>67.1</td><td>75.0</td></tr><tr><td>+ SCAPO (Ours)</td><td>78.8</td><td>87.6</td><td>76.2</td><td>73.1</td><td>75.8</td></tr></table>

## E.2 ABLATION OF SCAPO COMPONENTS

Table 9 ablates reward design within SCAPO. Additive hallucination penalty replaces the zerotolerance branch in Eq. (1) with an additive term, allowing valid claims to numerically offset local hallucinations. Full reward for repeated claims removes the discount on $r _ { \mathrm { r e p } } .$ , assigning redundant supported objects the same positive reward as novel ones. Without no-object regularization drops the penalty $\left( - r _ { \mathrm { r e g } } \right)$ applied to subsentences lacking verifiable objects. All variants use the same training configuration (Appendix C.1) and DOPA annotations, differing strictly in their mathematical reward or advantage formulations.

As demonstrated in the results, the complete SCAPO formulation achieves the best overall score, while each ablation degrades at least one metric, supporting the complementary roles of these reward-design choices.

## E.3 GENERALIZATION ACROSS BACKBONES

To examine whether SCAPO extends beyond the primary backbone, we additionally evaluate it on Qwen3-VL (Bai et al., 2025a).

As detailed in Table 10, the empirical results confirm the robust transferability of our approach. Specifically, SCAPO consistently enhances the performance of the Qwen3-VL backbone, demonstrating that the gains of SCAPO transfer to an additional LVLM backbone.

## F CASE STUDY

Figure 7 contrasts the performance of the base Qwen2.5-VL model with our SCAPO-aligned model on a representative case. Given the target question regarding small-scale background individuals, the auxiliary description generated by Qwen2.5-VL misses these subtle entities, focusing solely on the dog and general scenery. Consequently, the missing visual evidence leaves the subsequent discriminative prediction without sufficient context, resulting in a false negative. In contrast, our model produces an auxiliary description that is both highly faithful and rich in detail, successfully capturing the distant background people. This comprehensive visual grounding provides the necessary context for the model to correctly resolve the query.

## G LIMITATIONS

Our framework currently operates over a predefined object vocabulary, which leaves two practical directions for further extension. Although DOPA provides substantially denser supervision than conventional annotations, concepts outside V are still not explicitly evaluated and therefore receive limited training signal. This coverage can be progressively improved by expanding the vocabulary with newly observed model rollouts. In addition, constructing dense presence–absence annotations introduces extra offline annotation and verification cost, particularly when extending to larger vocabularies or new visual domains. This overhead is confined to dataset construction rather than inference, and future work could further reduce it through model-assisted annotation and iterative vocabulary expansion.

![](images/7bada9a85519b90568056434117e608cd787e430875bfed964e33b47b8971d06.jpg)  
Figure 7: Qualitative comparison of description-augmented discriminative inference between the baseline (Qwen2.5-VL) and our SCAPO-aligned model.