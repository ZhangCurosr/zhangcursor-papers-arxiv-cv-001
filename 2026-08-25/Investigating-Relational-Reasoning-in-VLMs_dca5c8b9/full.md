# Investigating Relational Reasoning in VLMs

Adhithya Laxman Ravi Shankar Geetha <sup>\*</sup> <sup>1</sup> Aulia Kharis Rakhmasari <sup>\*</sup> <sup>1</sup> Haleema Ramzan <sup>\*</sup> <sup>1</sup> Xander Yap <sup>\*</sup> <sup>1</sup>

## Abstract

Vision-Language Models (VLMs) achieve strong performance in visual reasoning tasks, but it remains unclear whether they understand visual relations, or simply employ shortcuts such as language cues or priors. To investigate this, we use the Qwen3-VL-4B (Bai et al., 2025), a modern VLM, to decode how visual information is encoded across depths. For this, we propose a synthetic dataset of simple geometric shapes for controlled analysis, along with queries crafted to precisely test language cues. Furthermore, the dataset is modified to test causal reliance on visual evidence. Our results show that current VLMs combine genuine visual reasoning with shortcut strategies primarily rooted in language cues.

## 1. Introduction

Visual data contains primitives (what exists), their attributes, and the relationships among them. These fundamental properties can be used to describe almost any visual information, with complexity added as a result of a combination of them. While seemingly straight-forward for humans, it encodes rich and complex implicit details that cannot directly be mapped to an explicit domain. For instance, an object A present at the left of object B in an image does not contain any pixel demarcations for this relationship, but rather an understanding of relative space on the perceivers’ end.

With the recent rise of large-scale Vision–Language Models (VLMs) such as LLaVA (Liu et al., 2024), GPT-4V (Wu et al., 2024), and Gemini (Google, 2025), that seem to achieve strong performance on many multi-modal tasks, the question remains as to how the visual information is internally understood by such models. A thorough comprehension would be marked by understanding beyond the explicit representation. Recent works have explored the performance of such models on diagrammatic information (Lu et al., 2023; Zhang et al., 2024), where the work by (Hou et al., 2024) shows that VLMs often struggle with relational reasoning beyond entity recognition (e.g., compositional reasoning on diagram reasoning benchmarks) (Johnson et al., 2017; Masry et al., 2022).

Motivated by concerns that some apparent success can come from shortcuts rather than grounded visual–relational understanding, our work aims to study how visual information is represented inside a VLM, and how sensitive predictions are to modifying the visual evidence (Hou et al., 2024) and language cues. Our work has the following key findings: (i) VLMs encode different visual information at different depths, (ii) Recognition tasks are understood on an architectural level, but relational tasks do not exhibit any consistent pattern, and (iii) VLMs show heavy reliance on presence of explicit visual elements. Our findings confirm previous works, and provide a deeper understanding of information processing within VLMs at an architectural level, opening avenues for advanced architectural modifications.

## 2. Related Work

Mechanistic interpretability work has shown that transformer attention patterns can encode structured dependencies and can be analyzed to study syntax and compositional reasoning in language models. In VLMs, attention-based analysis has more commonly been used for object localization and text–image grounding than for diagnosing whether attention captures inter-entity relations in a mechanistic sense (Clark et al., 2019; Elhage et al., 2021).

For evaluation of such tasks, relational reasoning benchmarks such as GQA (Ainslie et al., 2023) and CLEVR (Johnson et al., 2017) highlight systematic weaknesses in relation-centric querying, while diagram benchmarks such as ChartQA (Masry et al., 2022) provide complementary tests requiring visual and logical reasoning over structured graphics. Recent work by (Hou et al., 2024) argues that VLMs may rely on shortcuts rather than true relational understanding, motivating controlled tests and internal diagnostics. Building on this, we combined controlled relational tasks with attention-based probing to understand how relational information is encoded in VLMs.

Investigating Relational Reasoning in VLMs
<table><tr><td>Analysis Target</td><td>Query Type</td><td>Example</td></tr><tr><td rowspan="4">Recognition</td><td>Count</td><td>How many shapes are in this image?</td></tr><tr><td>Shape</td><td>Does this image have a square?</td></tr><tr><td>Color</td><td>Does this image have a green shape?</td></tr><tr><td>Shape &amp; Color</td><td>Does this image have a green square?</td></tr><tr><td rowspan="3">Relational</td><td>Arrow Existence</td><td>Which objects are connected?</td></tr><tr><td>Arrow Direction</td><td>Where does the arrow between the cyan circle and orange rectangle point to?</td></tr><tr><td>Implicit Position</td><td>What is the position of the orange rectangle with respect to the cyan circle?</td></tr></table>

Table 1. An overview of the variations of query types and their intended analysis target. Examples are provided from the dataset.

## 3. Methodology

## 3.1. Synthetic Dataset Creation

We construct a custom synthetic 2D benchmark, as shown in Fig 1, to analyze how VLM attention encodes entities and relations. The dataset consists of 5424 images in total, and is constructed using simple geometric shapes arranged in basic layouts with explicit ground-truth relations to aid mechanistic inspection. This design choice reduces linguistic ambiguity and enables controlled complexity.

![](images/6cce703b7685aefcb47edb5ea0274d2a8506d959983654f63c644aceb6252ab6.jpg)  
Figure 1. A sample image (left) with its masked variant (right).

## 3.1.1. SCENE GENERATOR

Each RGB image is 224×224 canvas containing between 2–5 entities {circle, square, rectangle, triangle} with sampled attributes from 8 discrete colors, and 3 discrete sizes (small/medium/large). The entities are placed using rejection sampling with a fixed margin from borders and collision checks to avoid overlaps, ensuring clean layouts suitable for precise attention analysis.

## 3.1.2. ENTITY RELATIONS

For every scene, we compute pairwise spatial relations based on entity centroids and assign a primary axis-aligned relation from {left of, right of, above, below}. Furthermore, with a 50/50 split, we create explicit (arrow rendered on canvas) and implicit (no arrow; position-only) relationships.

## 3.1.3. MASKED VARIANTS

To support causal tests of visual reliance, we generate a masked counterpart for each original image by re-rendering the scene while masking exactly one randomly chosen element: either (i) entity (remove one entity and its incident relations) or (ii) relation (mask one relation; if explicit, the arrow is omitted). Masking is implemented via full rerendering rather than patching to avoid visual artifacts and to retain original scene information of entities and relations. These images constitute half of the dataset.

## 3.2. Prompt Generation

Complementarily, a set of prompts is generated for each image in the dataset. All prompts follow a multiple-choice format in order to aid quantitative analysis. For example:

Query: How many shapes are in the image? Options: (a) 4 (b) 0 (c) 2 (d) 5 Instruction: Please only reply with the correct option. If no option is correct, reply with ’None’.

The queries are categorised based on their intended target of anlysis, and are further broken down into subtypes as shown in Table 1. They are generated using information contained within the image (specified in its corresponding JSON), which ensures a perfect mapping of queries to image content. For incorrect options provided in the prompt, existing entities and relations within the image are used - if not present or sufficient, then random entities and arrow connections contained in the dataset at large are used.

## 3.3. Attention Analysis

Final model outputs and prediction accuracies are severely limited and insufficient to draw conclusions about model behavior. To this end, we employ Linear Classifier Probes (Alain & Bengio, 2018). It is a conceptual tool to better understand the dynamics inside a neural network and the role played by the intermediate layers.

In this work, the linear probes act as layer-wise diagnostic for what information is readily decodable from Qwen-VL-4B’s internal representation. Concretely, we extract visual attention features from each layer and train a lightweight linear classifier to predict query answers from a controlled synthetic benchmark of geometric shapes.

<table><tr><td>Analysis Target</td><td>Query Type</td><td>Accuracy (%)</td></tr><tr><td rowspan="5">Recognition</td><td>Count</td><td>96.7</td></tr><tr><td>Shape</td><td>80.6</td></tr><tr><td>Color</td><td>96.8</td></tr><tr><td>Shape &amp; Color</td><td>99.5</td></tr><tr><td>Arrow Existence</td><td>51.2</td></tr><tr><td rowspan="3">Relational</td><td>Arrow Direction</td><td>92.1</td></tr><tr><td>Implicit Position</td><td>61.8</td></tr><tr><td></td><td></td></tr></table>

Table 2. Accuracies for query types.

This analysis helps us characterize emergence profiles: which patterns appear early, which require deeper computation, and which fail to materialize. However, probe performance alone does not establish that the model causally relies on visual evidence to answer. To test reliance, we thus complemented probing with masked input variants (as outlined in Section 3.1.3) to track how accuracy is impacted.

## 4. Experiments and Discussion

## 4.1. Overall Query Performance

The overall performance of the model is quantitatively measured by parsing model output and comparing against ground-truth dataset label. The accuracies achieved by the Qwen3-VL-4B model are summarized in Table 2. As expected, the model achieves high accuracy in recognition tasks. More specifically, the model performs consistently well when the query maps directly to tangible information in the visual input.

We observe that Shape yields much lower accuracies than Color, pointing towards a much better color understanding of the VLM. Interestingly, Shape and Color achieves an even higher accuracy, indicating performance improves with increased textual information. This becomes even more apparent when comparing the accuracies for the relational tasks: Arrow Direction significantly outperforms Arrow Existence and Implicit Position queries.

This is due to the fact that more explicit information is given in the Arrow Direction query, providing tangible and mappable information to visual data. Conversely, in Arrow Existence queries, the VLM is tasked to explore all connections on its own without any textual clues in the query, and in Implicit Position lack of tangible visual data leads to a significant accuracy drop.

## 4.2. Linear Probing

As the Qwen3-VL-4B is prompted, the attention maps are saved. These consist of $S \times L \times H$ where S = 5 is the steps, L = 36 is the number of layers, and H = 32 the heads. The steps correspond to the most relevant attention steps, filtered to represent the 95th percentile of activity. To meaningfully extract information from them, we apply mean pooling across steps and heads. This leaves us with 36 attention maps per query. These are used to train simple layer-wise logistic regression models for each query, with 20% of the data retained for testing.

![](images/20f75c78242ebdb9b9e0421ea42dcf9731e992b1502b44522dabc81bc783aeef.jpg)  
Figure 2. VLM Performance from probing based on sample type.

An additional caveat to note is that for simple queries of detection (color/shape/count), the probe is trained as is. However, for complex queries, the query was broken down into multiple sub-labels, and a separate linear probe had to be trained for each. The results discussed in the sections that follow are an average of performance over all sub-labels.

## 4.2.1. ACCURACIES BASED ON SAMPLE TYPES

Figure 2 summarizes the probe accuracies clustered by sample types. These clusters correspond to the following:

Overall Accuracy: the overall decodability of that variable from the layer representation, computed on the entire evaluation set for a given query type.

Correct Accuracy: how well the representation supports correctly identifying positive cases, computed on the subset where the ground-truth answer is a positive class.

False Accuracy: how well the representation supports correctly rejecting negatives, computed on the subset of samples where the ground-truth answer is the negative class.

## 4.2.2. ACCURACIES BASED ON QUERY TYPE

Figure 3 showcases accuracies across layers for the trained logistic models. We note the following findings:

Count: Accuracy is consistently high and slightly increases toward mid–late layers, indicating that count information becomes more linearly decodable deeper in the network.

Shape: mid–late layers yield the highest accuracy on positive cases, while negative-cases remain lower and more variable. Interestingly, shape presence is more reliably encoded than color presence, i.e, geometric cues are easier to extract from attention representations than chromatic ones.

Color: shows moderate decodability with a clear asymmetry between positives and negatives. This indicates that attention features carry usable evidence for confirming a queried color, but provide comparatively weak evidence for rejecting absent colors.

![](images/5bb5b5a1bc8dee3bc3b9653c37c48f1a1a8b685028fdc80e90c36c643b7e9898.jpg)  
Figure 3. Linear probing accuracies by query type across model layers.

Shape and Color: conjunction task amplifies the asymmetry: positive-case accuracy is relatively high and stable, whereas negative-case accuracy is consistently low. This is consistent with limited compositional binding - representation can confirm salient matching evidence, but struggles to confidently reject when the exact shape-color pair is absent.

Arrow Existence: this is comparatively well decoded and stable across depth, and shows near-overlap for sample types. This implies that the underlying relational signal (e.g., connection structure) is encoded consistently regardless of whether the model’s final answer is correct, pointing to robust representation-level encoding for this query type.

Arrow Direction: close to chance and unvaried across layers, with minimal separation between sample type. The attention-derived features do not provide a strong signal to resolve arrow-direction queries, and the model’s errors are not strongly concentrated in any specific subset.

Implicit Position: exhibits the weakest separability: sample types are tightly clustered, implying the representation contains only weak linearly decodable spatial information and, crucially, that the features are not markedly different between cases where the model succeeds or fails. In other words, spatial errors appear less driven by missing linearly accessible evidence and more by ambiguity/noise or nonlinear dependencies not captured by the probe.

## 4.2.3. ATTENTION PATTERNS

The following generalizations can be concluded from our experiments: (i) Early layers encode salient visual evidence and coarse statistics (often enough for presence and sometimes count), but weak for query identity and role-specific object identity. (ii) Middle layers have strongest regime for task-relevant visual signals (presence, counts, coarse relations), with the clearest separability for many probes. (iii) Late layers improve some higher-level relational structure (e.g., spatial position), but often do not recover object identity / compositional binding; some global signals (e.g., connection count) can even become less linearly accessible.

<table><tr><td>Query Type</td><td>Unmasked</td><td>Masked</td><td>Drop</td></tr><tr><td>Count</td><td>96.7</td><td>74.1</td><td>-23.4</td></tr><tr><td>Arrow Direction</td><td>92.1</td><td>76.4</td><td>-17.0</td></tr><tr><td>Shape &amp; Color</td><td>99.5</td><td>99.3</td><td>-0.2</td></tr><tr><td>Color</td><td>96.8</td><td>97.4</td><td>+0.6</td></tr><tr><td>Arrow Existence</td><td>51.2</td><td>50.3</td><td>-1.7</td></tr><tr><td>Shape</td><td>80.6</td><td>82.5</td><td>+2.4</td></tr><tr><td>Implicit Position</td><td>61.8</td><td>64.5</td><td>+4.3</td></tr></table>

Table 3. Ablation study results (in %) across all query types.

## 4.3. Ablation Study

To test whether high accuracy reflects genuine visual reasoning, we conducted the same experiments as outlined above for on the masked variant of the dataset as outlined in Section 3.1.3. The results are summarized in Table 4.3.

Large drops such as those witnessed in Count and Arrow Direction do not indicate successful visual grounding. Instead, they reveal hybrid strategies combining vision with internal biases. When masking creates distribution shift, these biases interfere with perception, causing hallucination. Other Recognition tasks show changes within noise margins (−0.2% to +2.4%). Surprisingly, Implicit Position improves by +4.3%—directly contradicting visual reasoning. This strongly indicates answers are derived from internal statistical regularities rather than visual content.

Our ablation exposes a dual-process architecture: hybrid tasks (count, arrows) attempt visual reasoning but fail under distribution shift (−17% to −23% drops), while shortcut tasks (recognition, spatial) bypass vision entirely (±5% changes). High unmasked accuracy (80–99%) creates an illusion of understanding that masking exposes as fragile.

## 5. Conclusion

Our work confirms that VLMs do not exhibit true visual understanding. Their comprehension of visual information is strictly limited to what is explicitly observable, with no capacity to intelligently reason beyond it. The results show clear reliance on language cues and data priors - findings consistent with previous works. Our decoded attention patterns map this behavior to the models architecture, and opens up avenues for reformative approach and architecture changes to cater to this built-in limitation.

## References

Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebron, F., and Sanghai, S. Gqa: Training generalized ´ multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes, 2018. URL https:// arxiv.org/abs/1610.01644.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Clark, K., Khandelwal, U., Levy, O., and Manning, C. D. What does bert look at? an analysis of bert’s attention. arXiv preprint arXiv:1906.04341, 2019.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Google, G. T. Gemini: A family of highly capable multimodal models, 2025. URL https://arxiv.org/ abs/2312.11805.

Hou, Y., Giledereli, B., Tu, Y., and Sachan, M. Do visionlanguage models really understand visual language? arXiv preprint arXiv:2410.00193, 2024.

Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning, 2024. URL https:// arxiv.org/abs/2310.03744.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR, 2023.

Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the associationfor computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Wu, T., Yang, G., Li, Z., Zhang, K., Liu, Z., Guibas, L., Lin, D., and Wetzstein, G. Gpt-4v(ision) is a humanaligned evaluator for text-to-3d generation, 2024. URL https://arxiv.org/abs/2401.04092.

Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.