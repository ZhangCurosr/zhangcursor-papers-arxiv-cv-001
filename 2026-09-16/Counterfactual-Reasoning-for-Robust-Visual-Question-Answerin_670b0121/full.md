# Counterfactual Reasoning for Robust Visual Question Answering

Truong-Binh Duong<sup>1,2</sup>, Thanh-Ngan Tran<sup>1,2</sup>, Ngoc-Thao Nguyen<sup>1,2</sup>, Bac Le<sup>1,2,∗</sup>

<sup>1</sup>Faculty of Information Technology, University of Science, Ho Chi Minh City, Vietnam

<sup>2</sup>Vietnam National University, Ho Chi Minh City, Vietnam

<sup>∗</sup>Corresponding author: Bac Le (lhbac@fit.hcmus.edu.vn)

Accepted for publication at the 30th International Conference on Knowledge-Based and Intelligent Information & Engineering Systems (KES 2026).

## Abstract

Modern Visual Question Answering (VQA) models often exploit spurious correlations in training data, leading to poor out-of-distribution (OOD) generalization due to language bias. Although counterfactual learning has shown promise, existing methods can be improved to better guide attention toward causal evidence and strengthen feature discrimination. To address this, we propose a novel training framework that enhances counterfactual contrastive learning for VQA. Our framework introduces three key contributions: (1) a three-stage curriculum for stable multi-objective optimization, (2) an enhanced Batch-Contrastive loss for more discriminative feature learning, and (3) two novel regularizers, Answer-Contrastive (AC) loss to refine the prediction space and Gradient-Discrepancy (GD) loss to enforce causal visual grounding. Our model achieves a competitive accuracy of 61.64% on the bias-sensitive VQA-CP v2 benchmark while maintaining 62.80% on the standard VQA v2 dataset, yielding a small generalization gap of 1.16%. This demonstrates a strong balance between OOD robustness and in-distribution performance.

Keywords: Visual Question Answering; Counterfactual Learning; Contrastive Learning; Language Bias; Causal Reasoning

## 1 Introduction

Visual Question Answering (VQA) is a key benchmark task in artificial intelligence, requiring models to comprehend both visual and textual information for complex reasoning [1]. While modern architectures achieve strong performance on benchmarks such as VQA v2 [2], this success often masks a critical limitation. Prior studies show these models exploit spurious correlations and language priors in training data, rather than performing genuine multimodal reasoning [3, 4, 5]. Consequently, performance degrades on out-of-distribution (OOD) datasets such as VQA-CP v2 [5], designed to penalize such shortcuts. This raises a fundamental question: How can we train VQA models to reason from causal evidence in both the image and the question, instead of memorizing statistical artifacts?

Counterfactual learning has emerged as a promising direction. Methods such as Counterfactual Samples Synthesizing (CSS) [6] generate augmented data by masking key elements in the image or question, thereby disrupting spurious correlations. Building on this idea, CL-VQA [7] pioneered the use of contrastive learning to explicitly model the relationships among original, factual, and counterfactual samples. This approach encourages the model to learn more generalizable representations by leveraging counterfactuals as a self-supervised signal. Despite these advances, existing methods still have limitations: the model’s attention is not explicitly guided toward causal regions, and the contrastive loss can be further improved to enhance discriminative power.

To address the gaps, we propose an enhanced counterfactual contrastive learning framework. Our contributions are: • A novel three-stage curriculum training framework that systematically integrates multiple complex learning objectives, ensuring training stability and effective optimization.

• An enhanced contrastive learning paradigm featuring an improved batch-contrastive loss and two novel regularizers: an Answer-Contrastive (AC) Loss to refine the prediction space, and a Gradient-Discrepancy (GD) Loss to enforce causal visual grounding.

• Competitive performance on the VQA-CP v2 benchmark while maintaining a small generalization gap to the VQA v2 dataset.

## 2 Related Work

## 2.1 Language Bias in VQA

While modern VQA models demonstrate strong performance on in-distribution (ID) benchmarks like VQA v2 [2], numerous studies [3, 5] have shown they exploit superficial linguistic correlations, or “language biases” in the training data. Instead of performing robust multimodal reasoning, models often learn to associate certain question patterns with high-frequency answers. For example, a model might learn to default to the answer “tennis” for any question beginning with “What sport $\mathrm { i s . . . } \varOmega ^ { \flat }$ regardless of the image content. This reliance on statistical shortcuts results in a substantial performance degradation on OOD test sets. To diagnose this vulnerability, the VQA-CP (Visual Question Answering under Changing Priors) dataset [5] was introduced, which intentionally creates a distribution shift between the training and test sets for each question type, making it a standard benchmark for evaluating model robustness.

## 2.2 Debiasing Methods in VQA

To address the language bias issue, one primary debiasing strategy uses ensemble-based models with an auxiliary, often question-only, branch to capture and regularize bias. These methods re-weight the loss or adjust predictions to downplay samples answerable from language priors, encouraging the main model to rely on visual evidence [4, 8, 9, 10]. Other strategies modify the training objective, for instance by using adaptive margin losses to create more discriminative feature spaces [11] or by improving visual grounding to ensure models are “right for the right reasons” [12]. More recently, causal inference approaches have gained prominence in formally disentangling visual reasoning from linguistic shortcuts. Notable examples include CVIV+iter [13], which employs instrumental variables to isolate the causal effect of visual evidence on the answer, and CIBi [14], which applies fine-grained causal intervention to eliminate context and keyword biases separately.

Another major research line, most relevant to our work, focuses on data augmentation. It expands the training set with samples that break spurious correlations and create a more balanced distribution. A foundational technique is Counterfactual Samples Synthesizing (CSS) [6], which identifies causally critical elements, such as key objects in the image or essential words in the question, using gradient-based analysis. It then generates “counterfactual” data by masking these elements and assigning new, logically consistent answers. MUTANT [15] extends this by generating “mutant” samples through semantic manipulations, such as changing an object’s color via inpainting or negating a question’s premise. By training the model to be consistent with these semantic shifts, it learns to understand the direct effect of input changes on the final answer. Unlike synthetic approaches, KDDAug [16] avoids potential generation artifacts by composing samples from existing pristine images and different human-written questions. This strategy assigns labels via knowledge distillation from pre-trained teachers, creating a scalable, rule-free approach.

However, simply generating more data is often insufficient because how the model utilizes these samples is crucial. This has led to contrastive learning frameworks, where CL-VQA [7] first introduced a contrastive objective to learn the relationship between original (anchor), factual (positive), and counterfactual (negative) samples generated by CSS. Similarly, MMBS [17] employs contrastive learning, constructing its positive samples by corrupting question-category information and treating biased and unbiased samples differently. Building on this foundation, we propose a curriculum and novel regularization losses to further strengthen the power of counterfactual contrastive learning for robust VQA.

## 3 Proposed Method

We propose a training framework to enhance causal reasoning in VQA, built upon the UpDn backbone [18]. It integrates a counterfactual synthesis module and three novel loss functions within a three-stage curriculum.

## 3.1 Baseline Architecture and Counterfactual Samples Synthesizing

Formally, VQA learns a mapping $f _ { \nu q a } : \mathcal { I } \times \mathcal { Q } \to \mathbb { R } ^ { | \mathcal { A } | }$ over dataset $\mathcal { D } = ( I _ { i } , Q _ { i } , a _ { i } ) _ { i = 1 } ^ { N }$ , where ${ \mathcal { I } } , { \mathcal { Q } } $ , and $\mathcal { A }$ denote images, questions, and answers. Using the UpDn backbone [18], as shown in Figure 1, the baseline extracts k region features V via image encoder $e _ { \nu }$ and question features $q$ via text encoder $e _ { q } . \mathrm { A }$ fusion module mm(·,·) applies top-down attention to produce a joint feature $z = m m ( V , q )$ , mapped to answer logits ${ \hat { y } } = C ( z )$ by classifier C. Because the quality of z dictates reasoning robustness, we employ contrastive learning. To mitigate spurious correlations, the CSS module [6] acts as a data augmentation engine. For each sample $( I , Q , a )$ , gradient-based analysis identifies critical regions or words. It synthesizes a factual sample $( I ^ { + } \thinspace \mathrm { o r } \ Q ^ { + } )$ by retaining critical objects and a counterfactual sample $( I ^ { - } \thinspace \mathrm { o r } \ Q ^ { - } )$ by masking them (Figure 2). Encoding these produces anchor (z), positive $( z ^ { + } )$ , and negative $( z ^ { - } )$ features for contrastive optimization.

## 3.2 Improved Loss Components

Building on these causal triplets, we introduce three complementary loss functions targeting different reasoning aspects.   
They jointly structure the feature space, refine the prediction space, and ensure causal visual grounding.

Batch-Contrastive Loss. The original CL-VQA objective [7] compares an anchor against single positive and negative samples. To learn a more discriminative representation, we adopt a Batch-Contrastive (BC) objective inspired

![](images/557a766a222263e73290520c74605bbef672fac90dd760ef7e0961ddd25d8b35.jpg)

Figure 1: An overview of our training framework. The VQA baseline and CSS module generate feature triplets $( z , z ^ { + } , z ^ { - } )$ optimized via a three-stage curriculum that progressively introduces our Batch-Contrastive (L ), Answer-Contrastive $( \mathcal { L } _ { \mathrm { A C } } )$ , and Gradient-Discrepancy $( \mathcal { L } _ { \mathrm { G D } } )$ losses.  
![](images/bebcbe18dcaa43dddd2c831bf80f716ba8d36f61b5b269e3adb1f6eb1c6b0a0e.jpg)  
Figure 2: Example of factual $( I ^ { + } , Q ^ { + } )$ and counterfactual $( I ^ { - } , Q ^ { - } )$ samples generated by the CSS module, isolating and masking causal elements, respectively.

by SimCLR [19]. This leverages all in-batch negatives for richer, more stable signals. Given a normalized anchor feature $z _ { i } ,$ its positive $z _ { i } ^ { + }$ , and all negative features $\{ z _ { k } ^ { - } \} _ { k = 1 } ^ { N }$ in a mini-batch of size N, the BC loss is:

$$
\mathcal { L } _ { \mathrm { B C } } ( i ) = - \log \frac { e ^ { s ( z _ { i } , z _ { i } ^ { + } ) / \tau } } { e ^ { s ( z _ { i } , z _ { i } ^ { + } ) / \tau } + \sum _ { k = 1 } ^ { N } e ^ { s ( z _ { i } , z _ { k } ^ { - } ) / \tau } } ,\tag{1}
$$

where $s ( \cdot , \cdot )$ is the cosine similarity and $\tau > 0$ is a temperature hyperparameter scaling distribution sharpness. By contrasting one positive pair against numerous in-batch negatives, this objective supplies more robust gradients and forces the model to learn more generalizable features.

Answer-Contrastive Loss. While $\mathcal { L } _ { \mathrm { B C } }$ organizes the feature space, it does not directly enforce separability in the final prediction space. To address ambiguity between semantically similar answers $( \mathrm { e . g . }$ , “red” vs. “maroon”), we introduce an Answer-Contrastive (AC) Loss [20] operating on answer embeddings. Let $g ( a )$ be the GloVe [21] embedding of ground-truth answer $a \in { \mathcal { A } }$ . For sample i with ground-truth set $\mathcal { A } _ { i } \subset \mathcal { A }$ , the AC loss pulls both anchor (z<sub>i</sub>) and positive $( z _ { i } ^ { + } )$ features closer to correct embeddings:

$$
\mathcal { L } _ { i } ^ { \mathrm { A C } } = - \log \frac { \sum _ { a \in \mathcal { A } _ { i } } \left( e ^ { s ( z _ { i } , g ( a ) ) } + e ^ { s ( z _ { i } ^ { + } , g ( a ) ) } \right) } { \sum _ { a ^ { \prime } \in \mathcal { A } } e ^ { s ( z _ { i } , g ( a ^ { \prime } ) ) } } .\tag{2}
$$

The numerator aggregates similarity scores for all correct answers across both views, while the denominator normalizes

![](images/4e747d64b349d0ea46b5b6de502535822e1aed6ee34f70ee737c69eb4e4688e4.jpg)  
(a) High GD Loss (low discrepancy).

![](images/0d5e62c6e2496f1750bd323e9687b47f903b3a4e49b97561443259371b2adf1f.jpg)  
(b) Low GD Loss (high discrepancy).  
Figure 3: Visualization of the GD Loss. (a) A poorly-grounded model shows little attention discrepancy between factual and counterfactual sample, which in turn leads to a high loss. (b) A well-grounded model attends to causal evidence, producing high discrepancy and low loss.

over the entire vocabulary. This directly enlarges decision margins between correct and incorrect predictions.

Gradient-Discrepancy Loss. Although BC and AC losses yield discriminative representations, they do not explicitly enforce correct visual grounding. The model might still exploit global statistical shortcuts rather than localizing causal evidence. To address this, we introduce a Gradient-Discrepancy (GD) Loss compelling the model to attend to relevant regions. It enforces a significant difference in model sensitivity, measured by output gradients with respect to visual features, between the factual $( z ^ { + } )$ and counterfactual $( z ^ { - } )$ views. Large discrepancy implies proper causal grounding, while low discrepancy implies shortcut learning, which we aim to penalize. To formalize this, we define the GD loss as a weighted objective that rewards gradient dissimilarity. Let $\hat { y } ^ { + }$ and $\hat { y } ^ { - }$ be the logits for positive and negative samples, $V _ { k }$ the visual feature at region $k ,$ and $p _ { \mathrm { p o s } }$ the positive pair probability from the BC head. The GD loss is formulated as:

$$
\mathcal { L } _ { \mathrm { G D } } = - \mathbb { E } _ { k } \left. \frac { \partial \sum _ { j } \hat { y } _ { j } ^ { + } } { \partial V _ { k } } - \frac { \partial \sum _ { j } \hat { y } _ { j } ^ { - } } { \partial V _ { k } } \right. \cdot \log p _ { \mathrm { p o s } } .\tag{3}
$$

This objective functions as a form of self-supervision for the model’s attention. The absolute difference measures the region-wise gradient discrepancy, adaptively weighted by log $p _ { \mathrm { p o s } }$ . Minimizing this loss penalizes low discrepancy, especially under uncertainty, forcing the model to develop distinct attention patterns. As illustrated in Figure 3, a well-grounded model (Fig. 3b) attends heavily to red flowers in $I ^ { + }$ and disperses attention when they are masked in I<sup>−</sup>, yielding high discrepancy and low loss. Conversely, a biased model (Fig. 3a) maintains static attention on spurious cues, incurring a heavy penalty. This steers focus towards genuine causal evidence.

## 3.3 Three-Stage Curriculum Training

Simultaneous optimization of complex objectives can destabilize training. We therefore adopt a three-stage curriculum strategy to progressively introduce loss components.

Stage 1 (Epochs 1–5) optimizes only the VQA task loss to establish stable base representations. Stage 2 (Epochs 6–15) adds BC to structure the embedding space around causal relationships. Stage 3 (Epochs 16–30) activates AC and GD to refine answer margins and enforce visual grounding:

$$
\begin{array} { r l } & { \mathcal { L } ^ { ( 1 ) } = \mathcal { L } _ { \mathrm { V Q A } } , } \\ & { \mathcal { L } ^ { ( 2 ) } = \mathcal { L } _ { \mathrm { V Q A } } + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } , } \\ & { \mathcal { L } ^ { ( 3 ) } = \mathcal { L } _ { \mathrm { V Q A } } + \lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } + \lambda _ { \mathrm { A C } } \mathcal { L } _ { \mathrm { A C } } + \lambda _ { \mathrm { G D } } \mathcal { L } _ { \mathrm { G D } } . } \end{array}\tag{4}
$$

The weighting coefficients λ<sub>BC</sub>, $\lambda _ { \mathrm { A C } }$ , and $\lambda _ { \mathrm { G D } }$ are reported in Section 4 and analyzed in Table 3. This schedule stabilizes optimization and incrementally strengthens causal reasoning.

## 4 Experiments

## 4.1 Experimental Settings and Implementation Details

We evaluate on VQA-CP v2 [5] for OOD robustness against language bias and VQA v2 [2] for ID performance. Following the standard fixed-split VQA-CP v2 protocol, we report the official VQA accuracy with Yes/No, Number, and Other breakdowns. Cross-validation is not used because VQA-CP v2 is designed with different answer-prior distributions between training and test splits. Repartitioning the data would change the OOD setting and reduce comparability with prior work. We use the standard UpDn backbone [18] as a controlled testbed, train for 30 epochs with Adamax, batch size 512, gradient clipping 0.25, and cosine learning-rate decay after warm-up to $2 \times 1 0 ^ { - 3 }$ . The loss weights are constrained to a total budget of 10, with larger values indicating higher optimization priority. After repeated runs, we set

Table 1: Performance comparison with state-of-the-art methods on the VQA-CP v2 and VQA v2 datasets. Our Baseline refers to the CL-VQA configuration, while Ours incorporates the proposed BC, AC, and GD loss. The “Gap” column represents the absolute difference between the overall accuracies on VQA v2 and VQA-CP v2. ✓ denotes methods requiring extra annotations. ↑: higher is better, ↓: lower is better. Best results are in bold, second best are underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Extra Anno.</td><td colspan="4">VQA-CP v2 test (%) ↑</td><td colspan="4">VQA v2 val (%) ↑</td><td rowspan="2">Gap ↓</td></tr><tr><td>All</td><td>Y/N</td><td>Num</td><td>Other</td><td>All</td><td>Y/N</td><td>Num</td><td>Other</td></tr><tr><td>UpDn (2018) [18]</td><td></td><td>39.74</td><td>42.27</td><td>11.93</td><td>46.05</td><td>63.48</td><td>81.18</td><td>42.14</td><td>55.66</td><td>23.74</td></tr><tr><td>AdvReg (2018) [22]</td><td></td><td>41.17</td><td>65.49</td><td>15.48</td><td>35.48</td><td>62.75</td><td>79.84</td><td>42.35</td><td>55.16</td><td>21.58</td></tr><tr><td>RUBi (2019) [4]</td><td></td><td>44.23</td><td>67.05</td><td>17.48</td><td>39.61</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LMH (2019) [8]</td><td></td><td>52.01</td><td>72.58</td><td>31.12</td><td>46.97</td><td>56.35</td><td>65.06</td><td>37.63</td><td>54.69</td><td>4.34</td></tr><tr><td>CSS (2020) [6]</td><td></td><td>58.95</td><td>84.37</td><td>49.42</td><td>48.21</td><td>59.91</td><td>73.25</td><td>39.77</td><td>55.11</td><td>0.96</td></tr><tr><td>CL-VQA (2020) [7]</td><td></td><td>59.18</td><td>86.99</td><td>49.89</td><td>47.16</td><td>57.29</td><td>67.27</td><td>38.40</td><td>54.71</td><td>1.89</td></tr><tr><td>GGE (2021) [9]</td><td></td><td>57.32</td><td>87.04</td><td>27.75</td><td>49.59</td><td>59.11</td><td>73.27</td><td>39.99</td><td>54.39</td><td>1.79</td></tr><tr><td>MMBS (2022) [17]</td><td></td><td>56.44</td><td>76.00</td><td>43.77</td><td>49.67</td><td>61.87</td><td>75.86</td><td>40.34</td><td>56.95</td><td>5.43</td></tr><tr><td>KDDAug (2022) [16]</td><td></td><td>61.14</td><td>88.31</td><td>56.10</td><td>48.28</td><td>62.17</td><td>79.50</td><td>40.57</td><td>54.71</td><td>1.03</td></tr><tr><td>GenB (2023) [10]</td><td></td><td>59.15</td><td>88.03</td><td>40.05</td><td>49.25</td><td>62.74</td><td>86.18</td><td>43.85</td><td>47.03</td><td>3.59</td></tr><tr><td>RMLVQA (2023) [11]</td><td></td><td>60.41</td><td>89.98</td><td>45.96</td><td>48.74</td><td>59.99</td><td>76.68</td><td>37.54</td><td>53.26</td><td>0.42</td></tr><tr><td>CIBi (2024) [14]</td><td></td><td>59.58</td><td>86.94</td><td>49.98</td><td>50.24</td><td>60.47</td><td>81.04</td><td>42.94</td><td>50.02</td><td>0.89</td></tr><tr><td>CVIV+iter (2024) [13]</td><td></td><td>60.08</td><td>88.85</td><td>40.77</td><td>50.30</td><td>61.42</td><td>79.34</td><td>39.85</td><td>53.50</td><td>1.34</td></tr><tr><td>Ours</td><td></td><td>61.64</td><td>90.44</td><td>51.74</td><td>49.26</td><td>62.80</td><td>80.47</td><td>38.58</td><td>55.77</td><td>1.16</td></tr><tr><td>HINT (2019) [23]</td><td>√</td><td>46.73</td><td>72.36</td><td>10.61</td><td>45.88</td><td>63.38</td><td>81.18</td><td>42.99</td><td>55.56</td><td>16.65</td></tr><tr><td>SCR (2019) [12]</td><td>√</td><td>49.17</td><td>71.55</td><td>10.72</td><td>47.49</td><td>62.20</td><td>78.90</td><td>41.40</td><td>54.30</td><td>13.03</td></tr><tr><td>MUTANT (2020) [15]</td><td>√</td><td>61.72</td><td>88.90</td><td>49.68</td><td>50.78</td><td>62.56</td><td>82.07</td><td>42.52</td><td>53.28</td><td>0.84</td></tr></table>

$\lambda _ { \mathrm { B C } } = 2 . 0 , \lambda _ { \mathrm { A C } } = 2 . 0 ,$ , and $\lambda _ { \mathrm { G D } } = 6 . 0 ,$ which produced the best overall trade-off in Table 3. Code, model configurations, hyperparameter settings, and evaluation instructions are available at GitHub repository.

## 4.2 Quantitative Results

VQA debiasing aims to develop robustly generalizable models rather than specialists for a single OOD benchmark. An ideal method should achieve high accuracy on the bias-sensitive VQA-CP v2, maintain strong performance on the standard VQA v2, and exhibit a minimal generalization gap. Table 1 evaluates our method against these criteria. Our model achieves 61.64% on VQA-CP v2 and 62.80% on VQA v2, with a small 1.16% generalization gap. Among annotation-free methods, it obtains the highest overall VQA-CP v2 accuracy and the best Yes/No score, while remaining competitive on VQA v2. Compared with MUTANT [15], which uses extra annotations, our method reaches a similar OOD accuracy without additional annotation resources. These results support the effectiveness of the proposed objectives, while the remaining gap to some recent methods suggests room for further improvement.

## 4.3 Qualitative Results

To illustrate how our method improves reasoning, we qualitatively compare our model against the CL-VQA baseline (Figure 4). Following CSS [6], we visualize the contribution of image regions and question words to the predicted answer. Green boxes denote positive contributions, red boxes denote suppressive ones, and darker green text indicates higher word importance. The top example highlights improved grounding. The baseline provides an incomplete answer (“red”) with diffused attention across the airplane. In contrast, our model correctly predicts “red and white” by precisely concentrating attention on the tail and suppressing other parts. Linguistically, focusing on “tail” and “plane” reflects superior semantic grounding. The bottom example demonstrates robustness against language priors. Faced with a yes/no question, the baseline succumbs to bias by answering with a color (“red”) and misdirecting attention. Conversely, our model correctly answers “yes” by grounding reasoning in actual visual evidence. It accurately encompasses the red flowers while suppressing distractors like yellow flowers, with linguistic attention correctly centered on the causal terms “flowers” and “red”. Collectively, these examples demonstrate that the AC and GD losses successfully guide the model toward causal evidence for accurate and well-grounded reasoning.

## 4.4 Analysis of Answer Distribution

To provide direct evidence of bias mitigation, we analyze predicted answer distributions on the VQA-CP v2 test set (Figure 5). The analysis confirms our model adapts to the test distribution rather than memorizing training-set biases. For the question “What color are the bananas?”, the training data is skewed toward “yellow”. Unlike the baseline, our

![](images/815b1b05c87fcd54bd0f342144c840a84d2219147209af386ee8894dd9921a13.jpg)  
Question:what color is the tail of the plane ? CL-VQA: red

![](images/e30b039de7675a1feaa5e869957bf3b32b5cee3f89d3ffd8a9c01ea4764972c0.jpg)  
Question: what color is the tail of the plane ? Ours: red and white

![](images/5829af435ba202c15c2e61419db452a589b942bdd7339845c31976eab50ad882.jpg)  
are some of the flowers red Question: ? CL-VQA: red

![](images/75cea0aa6d4c20941e9ff993d68885ab144ea89654e2f10f64404a30a9e529f7.jpg)  
Question: are some of the flowers red ? Ours: yes

Figure 4: Qualitative comparisons between our model and CL-VQA. Green boxes mark regions that support the prediction, red boxes denote regions being suppressed, and word-level importance is visualized with different shades of green.  
Question: What color are the bananas ?  
![](images/d7d01ab25958f66dad75583b6b6757b7317cc5a795cb1f1118095db90093a878.jpg)

![](images/2799eb260862e14d9f3ce7db6a6a9ff8e27df1e55f78d83f451314c3d2450706.jpg)

![](images/7b1bb7bdeb33705dd13f8882bf958012f46d43eeba0ef759b623cf33cc441bf6.jpg)

Question: How many ...?  
![](images/8b2a85383f762008dfcb07d9c65694f1264e348186a902521243678aaad0b2c4.jpg)  
Figure 5: Comparison of answer distributions predicted by our model and the CL-VQA baseline, alongside the groundtruth distributions of the train and test sets of VQA-CP v2 for several representative question types.

model correctly aligns with the test set’s “green” majority. This trend holds across other categories. For “Is this...?” questions, our model successfully overcomes the training bias toward “no” to reflect the test set’s “yes” majority. These

Table 2: Individual and synergistic impact of the AC and GD loss on the VQA-CP v2 test dataset. The BC (Batch-Contrastive) loss refers to our improved contrastive formulation.
<table><tr><td>Methods</td><td>All</td><td>Y/N</td><td>Number</td><td>Other</td></tr><tr><td>UpDn+LMH</td><td>52.01</td><td>72.58</td><td>31.12</td><td>46.97</td></tr><tr><td>+ CSS</td><td>58.95</td><td>84.37</td><td>49.42</td><td>48.21</td></tr><tr><td>+ AC</td><td>59.54</td><td>84.06</td><td>50.91</td><td>49.05</td></tr><tr><td>+ GD</td><td>59.86</td><td>87.49</td><td>45.17</td><td>49.40</td></tr><tr><td> $+ \Delta C + \mathrm { G D }$ </td><td>59.67</td><td>86.62</td><td>45.71</td><td>49.37</td></tr><tr><td>+CSS+BC</td><td>59.98</td><td>86.11</td><td>51.36</td><td>48.66</td></tr><tr><td>+ AC</td><td>60.57</td><td>87.09</td><td>53.24</td><td>48.52</td></tr><tr><td>+ GD</td><td>61.41</td><td>90.52</td><td>51.47</td><td>48.89</td></tr><tr><td>+ AC + GD</td><td>61.64</td><td>90.44</td><td>51.74</td><td>49.26</td></tr></table>

Table 3: Analysis of GD loss weight $( \lambda _ { \mathrm { G D } } )$ . Ours (Setting 2) showcases the best overall trade-off between OOD robustness and ID generalization.
<table><tr><td rowspan="2">Model</td><td colspan="3">Weights</td><td colspan="4">VQA-CP v2 test (%)</td><td colspan="4">VQA v2 val (%)</td></tr><tr><td> $\lambda _ { \mathrm { B C } }$ </td><td> $\lambda _ { \mathrm { A C } }$ </td><td> $\lambda _ { \mathrm { G D } }$ </td><td>All</td><td>Y/N</td><td>Num</td><td>Other</td><td>All</td><td>Y/N</td><td>Num</td><td>Other</td></tr><tr><td>MUTANT [15]</td><td>-</td><td>-</td><td>-</td><td>61.72</td><td>88.90</td><td>49.68</td><td>50.78</td><td>62.56</td><td>82.07</td><td>42.52</td><td>53.28</td></tr><tr><td>Ours (Setting 1)</td><td>2.0</td><td>4.0</td><td>4.0</td><td>61.41</td><td>90.52</td><td>51.47</td><td>48.89</td><td>62.43</td><td>79.62</td><td>38.46</td><td>55.75</td></tr><tr><td>Ours (Setting 2)</td><td>2.0</td><td>2.0</td><td>6.0</td><td>61.64</td><td>90.44</td><td>51.74</td><td>49.26</td><td>62.80</td><td>80.47</td><td>38.58</td><td>55.77</td></tr></table>

Table 4: Comparison of the original contrastive loss (CL) against our BC loss. Both are based on the full framework (UpDn+LMH+CSS+AC+GD).
<table><tr><td>Method</td><td>All</td><td>Y/N</td><td>Number</td><td>Other</td></tr><tr><td>Full Framework with CL</td><td>60.82</td><td>89.51</td><td>48.59</td><td>49.13</td></tr><tr><td>Full Framework with BC (Ours)</td><td>61.64</td><td>90.44</td><td>51.74</td><td>49.26</td></tr></table>

findings confirm that our framework reduces reliance on spurious language correlations, enabling better generalization under distribution shifts.

## 4.5 Ablation Studies

To isolate the contribution of each component, we conduct ablation studies on VQA-CP v2 evaluating our loss functions, their weight sensitivity, and the curriculum strategy.

Analysis of Loss Components. Table 2 details the impact of the AC and GD losses built upon two distinct baselines (UpDn+LMH+CSS and our UpDn+LMH+CSS+BC). On the CSS baseline, both AC and GD provide modest but inconsistent improvements. GD reduces performance on Number questions, and their combination yields no further gains over using GD alone. This suggests these regularizers require the stabilizing effect of a stronger contrastive objective to integrate synergistically. This dynamic changes significantly when combined with our BC loss. A clear synergistic effect emerges. GD delivers a substantial 4-point boost to Y/N accuracy (90.52%), while AC proves crucial for Number questions (53.24%). Their combination achieves the highest overall accuracy of 61.64%, confirming that AC and GD play complementary roles unlocked by the discriminative feature space of our BC loss.

Sensitivity to GD Loss Weight. Table 3 analyzes the sensitivity of the GD loss weight $\lambda _ { \mathrm { G D } }$ . Setting 2 increases the emphasis on causal grounding by raising $\lambda _ { \mathrm { G D } }$ to 6.0 while decreasing $\lambda _ { \mathrm { A C } }$ to 2.0, achieving a better overall trade-off. While Setting 1 peaks on the Y/N metric, Setting 2 improves the Number, Other, and overall OOD accuracy to 61.64%. This stronger grounding signal also yields superior ID performance, surpassing both Setting 1 and the MUTANT baseline on VQA v2 overall and Other accuracies.

Effect of the Improved Contrastive Loss. Table 4 validates the importance of our BC loss against the original contrastive loss (CL) from CL-VQA [7]. Substituting CL with BC increases overall accuracy from 60.82% to 61.64%. This underscores that leveraging harder negative sampling via a batch-wise objective is crucial for learning discriminative representations.

Effect of Curriculum Training. Table 5 evaluates our staged training strategy. Training all components jointly from the start yields a poor 52.94% accuracy due to unstable multi-objective optimization. A two-stage schedule mitigates this

Table 5: Impact of the Three-Stage Curriculum Training strategy.
<table><tr><td colspan="3">Training Stages</td><td colspan="4">VQA-CP v2 test (%)</td></tr><tr><td>Stage 1 (Epochs 1–5)</td><td>Stage 2 (Epochs 6–15)</td><td>Stage 3 (Epochs 16–30)</td><td>All</td><td>Y/N</td><td>Number</td><td>Other</td></tr><tr><td>VQA Loss + BC + AC + GD</td><td></td><td></td><td>52.94</td><td>71.93</td><td>42.18</td><td>45.94</td></tr><tr><td>VQA Loss + BC</td><td>+AC+GD</td><td></td><td>61.39</td><td>90.41</td><td>51.49</td><td>48.91</td></tr><tr><td>VQA Loss</td><td>+BC</td><td>+AC+GD</td><td>61.64</td><td>90.44</td><td>51.74</td><td>49.26</td></tr></table>

issue, improving the score to 61.39%. However, our proposed three-stage curriculum yields the best result of 61.64%, confirming that a gradual introduction of objectives ensures stable training and robust generalization.

## 4.6 Threats to Validity

Our evaluation follows the fixed VQA-CP v2/VQA v2 splits for comparability, but results may vary with initialization and implementation details. Moreover, VQA-CP v2 mainly tests answer-prior shifts and cannot represent all OOD conditions. We instantiate the framework on UpDn to isolate the contribution of the proposed objectives. Whether the same gains transfer to newer architectures should be further examined. Reproducibility also depends on dataset preprocessing, external image features, and software versions, which are documented with the released code.

## 5 Conclusion

We introduce a training framework to enhance causal reasoning in VQA via counterfactual contrastive learning. Our approach combines an improved Batch-Contrastive loss with Answer-Contrastive and Gradient-Discrepancy losses within a stable three-stage curriculum. The model achieves 61.64% accuracy on VQA-CP v2 and 62.80% on VQA v2, with a small generalization gap. Ablation studies show this performance stems from the complementary roles of each component in refining the feature space, prediction space, and visual grounding. This work presents a robust approach towards more reliable VQA models that balance OOD robustness and ID performance. Future work includes applying this framework to larger Transformer-based architectures to further advance multimodal causal reasoning.

## Acknowledgements

This research is funded by Vietnam National University, HoChiMinh City (VNU-HCM) under grant number CB2025- 18-04.

## References

[1] S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, D. Parikh, Vqa: Visual question answering, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 2425–2433.

[2] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, D. Parikh, Making the v in vqa matter: Elevating the role of image understanding in visual question answering, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 6325–6334.

[3] A. Agrawal, D. Batra, D. Parikh, Analyzing the behavior of visual question answering models, in: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2016, pp. 1955–1960.

[4] R. Cadene, C. Dancette, H. Ben-younes, M. Cord, D. Parikh, Rubi: Reducing unimodal biases for visual question answering, in: Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 841–852.

[5] A. Agrawal, D. Batra, D. Parikh, A. Kembhavi, Don’t just assume; look and answer: Overcoming priors for visual question answering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4971–4980.

[6] L. Chen, X. Yan, J. Xiao, H. Zhang, S. Pu, Y. Zhuang, Counterfactual samples synthesizing for robust visual question answering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 10797–10806.

[7] Z. Liang, W. Jiang, H. Hu, J. Zhu, Learning to contrast the counterfactual samples for robust visual question answering, in: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 3285–3292.

[8] C. Clark, M. Yatskar, L. Zettlemoyer, Don’t take the easy way out: Ensemble based methods for avoiding known dataset biases, in: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 4069–4082.

[9] X. Han, S. Wang, C. Su, Q. Huang, Q. Tian, Greedy gradient ensemble for robust visual question answering, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 1564–1573.

[10] J. W. Cho, D.-J. Kim, H. Ryu, I. S. Kweon, Generative bias for robust visual question answering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 11681–11690.

[11] A. Basu, S. Addepalli, R. V. Babu, Rmlvqa: A margin loss approach for visual question answering with language biases, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 11671–11680.

[12] J. Wu, R. J. Mooney, Self-critical reasoning for robust visual question answering, in: Proceedings of the 33rd International Conference on Neural Information Processing Systems (NeurIPS), 2019, pp. 772–782.

[13] Y. Pan, J. Liu, L. Jin, Z. Li, Unbiased visual question answering by leveraging instrumental variable, IEEE Transactions on Multimedia 26 (2024) 6648–6662.

[14] Y. Liu, G. Bai, L. Chenji, S. Li, Z. Zhang, R. Liu, W. Guo, Eliminating the language bias for visual question answering with fine-grained causal intervention, in: Proceedings of the IEEE International Conference on Multimedia and Expo (ICME), 2024, pp. 1–6.

[15] T. Gokhale, P. Banerjee, C. Baral, Y. Yang, Mutant: A training paradigm for out-of-distribution generalization in visual question answering, in: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 878–892.

[16] L. Chen, Y. Zheng, J. Xiao, Rethinking data augmentation for robust visual question answering, in: Proceedings of the European Conference on Computer Vision (ECCV), 2022, pp. 95–112.

[17] Q. Si, Y. Liu, F. Meng, Z. Lin, P. Fu, Y. Cao, W. Wang, J. Zhou, Towards robust visual question answering: Making the most of biased samples via contrastive learning, in: Findings of the Association for Computational Linguistics: EMNLP, 2022, pp. 6650–6662.

[18] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, L. Zhang, Bottom-up and top-down attention for image captioning and visual question answering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 6077–6086.

[19] T. Chen, S. Kornblith, M. Norouzi, G. Hinton, A simple framework for contrastive learning of visual representations, in: Proceedings of the 37th International Conference on Machine Learning (ICML), 2020, pp. 1597–1607.

[20] J. W. Cho, D.-J. Kim, Y. Jung, I. S. Kweon, Counterfactual mix-up for visual question answering, IEEE Access 11 (2023) 95201–95212.

[21] J. Pennington, R. Socher, C. Manning, Glove: Global vectors for word representation, in: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pp. 1532–1543.

[22] S. Ramakrishnan, A. Agrawal, S. Lee, Overcoming language priors in visual question answering with adversarial regularization, in: Advances in Neural Information Processing Systems (NeurIPS), 2018, pp. 1548–1558.

[23] R. R. Selvaraju, S. Lee, Y. Shen, H. Jin, S. Ghosh, L. Heck, D. Batra, D. Parikh, Taking a hint: Leveraging explanations to make vision and language models more grounded, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 2591–2600.