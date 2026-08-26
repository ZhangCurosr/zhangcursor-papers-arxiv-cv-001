# BOOT-AND-FEEDBACK FRAMEWORK FOR GENERALIST-EXPERT MODEL COLLABORATION IN BREAST ULTRASOUND DIAGNOSIS

Ming Cheng<sup>1</sup>, Hongyu Sun<sup>2</sup>, Zhaolin Chen<sup>1</sup>, Jun Liu<sup>3</sup>, Hossein Rahmani<sup>3</sup>, Qiuhong Ke<sup>1†</sup>

<sup>1</sup> Department of Data Science & AI, Monash University, Australia

<sup>2</sup> Department of Computer Science, Renmin University of China, China

<sup>3</sup> School of Computing and Communications, Lancaster University, UK

## ABSTRACT

Breast ultrasound (BUS) is widely used for breast cancer diagnosis yet remains operator-dependent. While deep learning shows promise, ensuring diagnostic reliability and interpretability is challenging. Recent Multimodal Large Language Models (MLLMs) often generate spurious descriptions due to limited domain knowledge, which mislead downstream expert models and compromise clinical validity. To address these challenges, we propose the Boot-and-Feedback (BooF) model collaboration framework for synergistic MLLM-expert interaction. Specifically, in the Boot Stage, the MLLM is guided by the BI-RADS lexicon and preliminary benignmalignant vision-expert predictions, enabling it to transfer general reasoning to BUS analysis while avoiding hallucinations. Subsequently, the Feedback Stage integrates these descriptions with visual features via a lightweight Attention-Gated Cross-Modality Fusion Module. This allows the expert to leverage textual feedback while adaptively filtering noise. Extensive experiments on multiple BUS datasets demonstrate that BooF substantially outperforms state-of-the-art methods in terms of diagnostic accuracy and interpretability.

Index Terms— Multimodal Large Language Models, Vision-Language Model, BI-RADS, Breast Ultrasound

## 1. INTRODUCTION

Breast cancer is the most prevalent malignancy among women, and early detection is crucial [1]. Breast ultrasound (BUS) is a widely used diagnostic technique due to its non-invasiveness and accessibility, but its interpretation remains operator-dependent, causing variability across clinical settings [2]. This challenge has driven the development of computer-aided diagnosis (CAD) systems that provide accurate, interpretable results to assist radiologists.

Although deep learning has advanced, it has not yet produced an ideal CAD system that ensures both high accuracy and interpretability. Domain-specific models based on CNNs [3] and ViTs [4] show competitive performance in breast cancer diagnosis but lack structured clinical reasoning, limiting interpretability. Multimodal large language models (MLLMs) offer a promising alternative by generating human-readable diagnostic descriptions [5, 6], but their application in BUS is limited by hallucinations and domain misalignment due to the lack of medical priors.

![](images/623acd3411b3949260e2a542f989ff8cc6ea09d976f22b1dcd92fde988226828.jpg)  
Fig. 1. Comparison between Boot-and-Feedback Framework and previous unidirectional model interaction.

Recent studies [7, 8, 9] have leveraged MLLM-generated descriptions to enhance expert model performance, but our preliminary experiments in Table I indicate that such approaches often fail or degrade performance. Critically, MLLM-generated outputs frequently conflict with expert model predictions, undermining reliability in clinical applications. As shown in Figure 1, these issues arise from two main factors: (i) the lack of domain-specific knowledge in MLLMs, which leads to hallucinations; and (ii) the unidirectional interaction between MLLMs and expert models, which limits feedback and accuracy.

To address these challenges, we propose a Boot-and-Feedback (BooF) model collaborative framework, which enables bidirectional interaction between general MLLM and domain-specific expert models. In the boot stage, we propose two MLLM boot strategies: (1) BI-RADS Lexicon Alignment. This strategy bridges the MLLM’s general image analysis capability with standardized diagnostic terminology and constrains the model to produce BI-RADS–based descriptions instead of direct diagnostic judgments. (2) Vision Expert Prediction Priors Guidance. Compared with unidirectional flow, this strategy incorporates preliminary benign–malignant predictions from a Vision Expert to serve as a reference for the MLLM, thereby suppressing hallucinations and ensuring clinically consistent diagnostic outputs. In the feedback stage, we feed the enhanced MLLM-generated descriptions together with the ultrasound images into a Vision–Language Expert to further improve classification performance. Specifically, we design a lightweight Attention-Gated Cross-Modality Fusion Module that adaptively filters potential semantic hallucination noise to ensure reliable diagnostic predictions.

![](images/5c21bf3eb1c21c07f0ac848891ad10765b571d8e309b68b0d249f4108e0236c3.jpg)  
Fig. 2. Overview of the proposed Boot-and-Feedback (BooF) framework. The Boot Stage adapts the MLLM with BI-RADS lexicon alignment and vision-expert priors to generate structured lesion descriptions. The Feedback Stage fuses these descriptions with visual features via an attention-gated module for the final diagnosis.

Through extensive ablation studies, we demonstrate the effectiveness of each proposed component. Comparisons on two public datasets show that BooF outperforms state-of-theart methods in both accuracy and interpretability.

## 2. METHODS

## 2.1. Boot Stage: MLLM Domain Adaptation

## 2.1.1. BI-RADS Lexicon Alignment

In early experiments, we directly applied Gemini 2.5 Pro [10], a state-of-the-art MLLM, to the breast ultrasound (BUS) diagnosis task. However, without domain adaptation, the model failed to generalize from generic image analysis to BUS interpretation, frequently producing hallucinatory diagnostic reasoning rather than evidence-based inference. This not only undermines clinical validity but also degrades downstream Vision–Language Expert performance compared with single-modality baselines (Table I). This failure stems from an inherent property of MLLMs: although they are proficient at describing general image semantics, they struggle with domain-specific reasoning and clinical decision making.

To overcome these limitations, we reformulated the task by mirroring radiologists’ reasoning, which integrates BI-RADS criteria with imaging characteristics. The Breast

Imaging Reporting and Data System (BI-RADS) [11] provides a standardized lexicon for describing five key lesion properties—shape, orientation, margin, internal echogenicity, and posterior features. Instead of soliciting direct benign/malignant predictions, we therefore guided the MLLM to generate BI-RADS–aligned lesion descriptions. This approach ensures clinical validity and consistency while delegating the final classification to a downstream vision language expert. Specifically, we employ task-activating prompting [12] to role-play the MLLM as an experienced radiologist, thereby activating medical reasoning, and use in-context learning (ICL) [13] to constrain outputs to the BI-RADS attribute space.

In addition, we establish explicit correspondences between the BI-RADS descriptors and the low-level image features, for example, linking orientation with the spatial distribution of the lesion and internal echogenicity / posterior features with the grayscale intensity patterns. This bridging mechanism enables the MLLM to better transfer its general visual competence into the BUS domain, producing descriptions that are semantically valid and grounded in visual evidence.

## 2.1.2. Vision Expert Prediction Priors Guidance

Although the BI-RADS Alignment Strategy introduces domain priors and partially mitigates hallucinations, its performance remains insufficient for clinical reliability. More importantly, the lack of bidirectional interaction leads MLLMs to produce reasoning misaligned with downstream multimodal predictions, resulting in inconsistent diagnostic outcomes.

Although fine-tuning could address these issues, it is prohibitively expensive and infeasible for closed-source MLLMs. To this end, we propose the Vision Expert Prediction Priors (VEPP) Strategy, which leverages a lightweight vision expert trained on BUS images. During inference, the expert provides probabilistic benign–malignant estimates $p \in [ 0 , 1 ]$ . After confidence calibration, we compute $s = \operatorname* { m a x } ( p , 1 - p )$ . $ { \mathrm { I f } } s \geq \tau$ (with τ selected on the validation set), we construct a structured prior $\mathcal { P } = \{ \mathrm { h y p o t h e s i s : } $ benign/malignant, confidence : s}. Otherwise, no prior is provided. The prior is injected into the boot-stage prompt alongside the BI-RADS template. The instructions emphasize: “Use P only as a reference; generate BI-RADS– aligned descriptors, and explicitly state contradictions if the image evidence disagrees.”. This design reframes the open-ended query of “Is there cancer?” into a constrained, hypothesis-driven description task.

By narrowing the reasoning space while preserving independence, VEPP reduces hallucination-prone outputs and allows the MLLM to challenge erroneous priors. Empirically, we observe that even when the expert’s initial prediction is wrong, the MLLM can flag inconsistencies under BI-RADS guidance and produce coherent descriptions, enabling the downstream vision-language expert to recover the correct classification. Overall, VEPP improves robustness, aligns textual reasoning with classifier decision space, and enhances clinical reliability.

## 2.2. Feedback Stage: Multimodal Expert Enhancement

## 2.2.1. Vision-Language Expert Model Pipeline

In the feedback stage, MLLM-generated BI-RADS lesion descriptions are integrated into a downstream Vision–Language Expert, establishing a bidirectional interaction with the boot stage. Unlike the boot stage, which relies solely on vision cues, this vision-language expert model jointly encodes textual descriptors and BUS image features to refine the prediction of malignancy.

Concretely, each BI-RADS attribute description $t _ { i }$ is encoded as $\mathbf { T } _ { i } ~ = ~ f _ { T } ( t _ { i } )$ using a frozen text encoder, where $i \in { 1 , 2 , \dots , 5 }$ corresponds to the five BI-RADS properties. The text encoder is kept frozen to ensure the preservation of standardized semantics. The original BUS image X is processed with a trainable image encoder $f _ { I } ( \cdot )$ , producing a global feature representation $\mathbf { I } = f _ { I } ( X )$ . Both encoders are initialized from large-scale pre-trained models, enabling efficient transfer with limited annotated BUS images [14].

As a result, this bidirectional framework reinforces the concordance between diagnostic predictions and generated descriptions, thereby improving clinical interpretability. By analogy to the boot stage, an optional extension is to propagate the predictions of the vision language expert back into the MLLM, which can further strengthen the alignment between classification outcomes and textual descriptions.

## 2.2.2. Attention-Gated Multimodal Fusion Mechanism

Although the boot stage significantly improves the quality of MLLM-generated diagnostic descriptions, a second calibration step is required to mitigate residual hallucinations. Inspired by the clinical practice of radiologists reverifying imaging findings, we design an Attention-Gated

Cross-Modality Fusion Module (AGCFM) that selectively interacts with the most relevant BI-RADS descriptors, enabling the expert model to leverage semantic priors while suppressing hallucination noise [23]. Formally, given the text embeddings $\mathbf { T } _ { i }$ , the transformed image query and textual keys are defined as:

$$
Q = W _ { Q } { \bf I } , K = W _ { K } [ { \bf T } _ { 1 } , { \bf T } _ { 2 } , \ldots , { \bf T } _ { C } ] ,\tag{1}
$$

where $W _ { Q } , W _ { K }$ are learnable transformation matrices. The cross-modality attention matrix $A \in \mathbb { R } ^ { 1 \times C }$ is computed as:

$$
A = \operatorname { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) .\tag{2}
$$

This quantifies the relevance of each textual attribute to the lesion’s visual features. The fused representation is then obtained by:

$$
\mathbf { Z } = \mathbf { I } + A W _ { V } [ \mathbf { T } _ { 1 } , \mathbf { T } _ { 2 } , \ldots , \mathbf { T } _ { C } ] ,\tag{3}
$$

where $W _ { V }$ projects text embeddings into the image feature space. The fused representation Z is then passed through a fully connected layer to generate classification logits $\hat { y } ,$ predicting lesion malignancy. The model is optimized using cross-entropy loss.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

We evaluate our method on two publicly available breast ultrasound (BUS) datasets: BUS-BRA [15] and BUSI [17]. The BUS-BRA dataset comprises 1268 and 607 benign and malignant BUS images, respectively, annotated with BI-RADS descriptors and biopsy-proven labels, while BUSI contains 437 benign BUS images and 210 malignant BUS images. In implementation, we apply Gemini 2.5 Pro [10] as our base MLLM to generate diagnostic descriptions. The visionexpert model shares the same vision encoder and training pipeline as the Vision–Language Expert. Data preprocessing is standardized across all datasets to ensure consistency. Model performance is evaluated using six key metrics: AUC, Accuracy, Specificity, Precision, Recall, and F1-score. To ensure fair evaluation, all models are trained and tested under 5-fold cross-validation implemented in PyTorch and executed on NVIDIA A100 GPUs.

## 3.2. Ablation Studies on Boot Stage and Feedback Stage

To evaluate the contribution of each component, we perform ablation studies on BUS-BRA using RadBERT [24] as the text encoder and MedViT [16] and ResNet-50 [3] as visual encoders. Baselines — (i) Vision expert refers to the standalone vision classifier. (ii) Vision–Language Expert incorporates MLLM-generated descriptions into the expert model without any boot or feedback strategy. As shown in Table I, naively coupling an MLLM proves unreliable: for instance, on ResNet-50[3], AUC even drops below that of the visiononly baseline, suggesting that spurious or clinically inconsistent text can misguide the expert model. Boot stage — Introducing BI-RADS Lexicon Alignment constrains the MLLM to generate standardized, terminology-grounded descriptions rather than direct diagnostic judgments, leading to consistent performance gains. Further incorporating Vision Expert Prediction Priors Guidance yields the most substantial improvements—particularly in Recall and AUC—by grounding the MLLM’s reasoning in the vision expert’s benign–malignant predictions and effectively suppressing hallucinations. Feedback stage — The attention-gated multimodal fusion then integrates these enhanced textual descriptions with visual features while adaptively filtering semantic noise, delivering an additional and stable boost in performance. Across both backbones, the complete BooF configuration achieves the strongest results (bold in Table I), with absolute improvements over the vision-only baseline of approximately 4.8%–6.1% in AUC, 5.7%–7.4% in Accuracy, and 14%–16% in Recall. These consistent gains confirm that BooF fosters a synergistic interaction between general MLLM reasoning and domain-specific expertise, thereby enhancing diagnostic reliability and ensuring stronger clinical alignment. Qualitatively, the MLLM self-corrects erroneous vision priors, while feedback filters residual inconsistencies via attention fusion.

Table I: ABLATION ON BUS-BRA [15] WITH TWO BACKBONES. VALUES ARE $\mathbf { M } \mathbf { E } \mathbf { A } \mathbf { N } \pm \mathbf { S } \mathbf { T } \mathbf { D }$ OVER 5 FOLDS. BEST RESULTS PER BACKBONE ARE BOLD.
<table><tr><td>Backbone</td><td>Method</td><td>AUC↑</td><td>Accuracy↑</td><td>Specificity↑</td><td>Precision↑</td><td>Recall↑</td><td>F1-score↑</td></tr><tr><td rowspan="5">MedViT [16]</td><td>Vision expert</td><td> $0 . 9 1 5 \pm 0 . 0 1 7$ </td><td> $0 . 8 6 0 \pm 0 . 0 1 7$ </td><td> $0 . 9 0 6 \pm 0 . 0 2 4$ </td><td> $0 . 7 9 6 \pm 0 . 0 4 2$ </td><td> $0 . 7 5 9 \pm 0 . 0 5 3$ </td><td> $0 . 7 7 6 \pm 0 . 0 3 9$ </td></tr><tr><td>Vision-language expert</td><td> $0 . 9 1 9 \pm 0 . 0 1 5$ </td><td> $0 . 8 6 2 \pm 0 . 0 1 5$ </td><td> $0 . 8 9 8 \pm 0 . 0 2 2$ </td><td> $0 . 7 8 7 \pm 0 . 0 3 5$ </td><td> $0 . 7 8 6 \pm 0 . 0 0 5$ </td><td> $0 . 7 8 6 \pm 0 . 0 1 8$ </td></tr><tr><td>+BI-RADS lexicon</td><td> $0 . 9 3 2 \pm 0 . 0 1 0$ </td><td> $0 . 8 8 0 \pm 0 . 0 0 8$ </td><td> $0 . 9 3 3 \pm 0 . 0 0 9$ </td><td> $0 . 8 4 5 \pm 0 . 0 2 8$ </td><td> $0 . 7 6 9 \pm 0 . 0 2 5$ </td><td> $0 . 8 0 5 \pm 0 . 0 1 9$ </td></tr><tr><td>+ Prior guidance</td><td> $0 . 9 7 5 \pm 0 . 0 0 6$ </td><td> $0 . 9 2 6 \pm 0 . 0 0 8$ </td><td> $0 . 9 4 2 \pm 0 . 0 1 2$ </td><td> $0 . 8 8 1 \pm 0 . 0 2 1$ </td><td> $0 . 8 9 1 \pm 0 . 0 2 9$ </td><td> $0 . 8 8 6 \pm 0 . 0 1 8$ </td></tr><tr><td>+ Attention-gated</td><td> $\mathbf { 0 . 9 7 6 \pm 0 . 0 0 4 }$ </td><td> ${ \bf 0 . 9 3 4 \pm 0 . 0 0 5 }$ </td><td> ${ \bf 0 . 9 4 9 \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 8 9 5 \pm 0 . 0 1 0 }$ </td><td> ${ \bf 0 . 9 0 3 \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 8 9 9 \pm 0 . 0 1 1 }$ </td></tr><tr><td rowspan="5">ResNet-50 [3]</td><td>Vision expert</td><td> $0 . 9 2 7 \pm 0 . 0 1 2$ </td><td> $0 . 8 6 9 \pm 0 . 0 1 1$ </td><td> $0 . 9 3 3 \pm 0 . 0 1 2$ </td><td> $0 . 8 3 9 \pm 0 . 0 3 4$ </td><td> $0 . 7 3 4 \pm 0 . 0 3 3$ </td><td> $0 . 7 8 3 \pm 0 . 0 2 5$ </td></tr><tr><td>Vision-language expert</td><td> $0 . 9 1 9 \pm 0 . 0 0 8$ </td><td> $0 . 8 6 1 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 9 \pm 0 . 0 2 3$ </td><td> $0 . 8 4 8 \pm 0 . 0 4 7$ </td><td> $0 . 6 9 6 \pm 0 . 0 4 6$ </td><td> $0 . 7 6 2 \pm 0 . 0 1 6$ </td></tr><tr><td>+BI-RADS lexicon</td><td> $0 . 9 3 1 \pm 0 . 0 1 3$ </td><td> $0 . 8 7 7 \pm 0 . 0 1 5$ </td><td> $0 . 9 3 2 \pm 0 . 0 1 9$ </td><td> $0 . 8 4 3 \pm 0 . 0 4 2$ </td><td> $0 . 7 6 4 \pm 0 . 0 4 9$ </td><td> $0 . 8 0 0 \pm 0 . 0 2 7$ </td></tr><tr><td>+ Prior guidance</td><td> $0 . 9 6 9 \pm 0 . 0 0 6$ </td><td> $0 . 9 1 6 \pm 0 . 0 1 2$ </td><td> $0 . 9 3 7 \pm 0 . 0 1 8$ </td><td> $0 . 8 7 1 \pm 0 . 0 3 5$ </td><td> $0 . 8 7 3 \pm 0 . 0 2 3$ </td><td> $0 . 8 7 1 \pm 0 . 0 1 8$ </td></tr><tr><td>+ Attention-gated</td><td> $\mathbf { 0 . 9 7 5 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 2 6 \pm 0 . 0 0 8 }$ </td><td> ${ \bf 0 . 9 4 3 \pm 0 . 0 1 7 }$ </td><td> ${ \bf 0 . 8 8 3 \pm 0 . 0 3 0 }$ </td><td> ${ \bf 0 . 8 9 1 \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 8 8 7 \pm 0 . 0 1 5 }$ </td></tr></table>

Table II: QUANTITATIVE COMPARISON ON BUSI [17] AND BUS-BRA [15]. VALUES ARE MEAN ± STD OVER 5 FOLDS. BEST RESULTS ARE BOLD.
<table><tr><td colspan="5">BUSI</td></tr><tr><td>Method</td><td>AUC↑</td><td>Specificity↑</td><td>Recall↑</td><td>F1-score↑</td></tr><tr><td>KRC-APM [18]</td><td> $0 . 9 5 5 \pm 0 . 0 0 5$ </td><td> $0 . 8 9 4 \pm 0 . 0 0 9$ </td><td> $0 . 8 4 7 \pm 0 . 0 3 3$ </td><td> $0 . 8 6 4 \pm 0 . 0 2 1$ </td></tr><tr><td>HoverTrans [19]</td><td> $0 . 9 5 7 \pm 0 . 0 1 0$ </td><td> $0 . 9 1 8 \pm 0 . 0 1 6$ </td><td> $0 . 8 5 8 \pm 0 . 0 2 8$ </td><td> $0 . 8 6 4 \pm 0 . 0 2 8$ </td></tr><tr><td>REAF [20]</td><td> $0 . 9 5 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 2 9 \pm 0 . 0 1 8$ </td><td> ${ \bf 0 . 8 6 8 \pm 0 . 0 3 5 }$ </td><td> $0 . 8 8 0 \pm 0 . 0 1 3$ </td></tr><tr><td>BooF (ours)</td><td> ${ \bf 0 . 9 5 9 \pm 0 . 0 2 6 }$ </td><td> ${ \bf 0 . 9 6 2 \pm 0 . 0 2 6 }$ </td><td> $0 . 8 5 3 \pm 0 . 0 4 0$ </td><td> ${ \bf 0 . 8 8 5 \pm 0 . 0 3 5 }$ </td></tr><tr><td colspan="5">BUS-BRA</td></tr><tr><td>BD-StableNet [21]</td><td>0.846</td><td>0.924</td><td>0.652</td><td>0.732</td></tr><tr><td>SgmaFuse [22]</td><td> $0 . 9 4 2 \pm 0 . 0 1 5$ </td><td></td><td></td><td> ${ \bf 0 . 9 4 6 \pm 0 . 0 1 5 }$ </td></tr><tr><td>BUS-BRA [15]</td><td> $0 . 9 3 1 \pm 0 . 0 2 5$ </td><td> $0 . 8 6 8 \pm 0 . 0 1 4$ </td><td> $0 . 8 5 9 \pm 0 . 0 7 5$ </td><td>一</td></tr><tr><td>BooF (ours)</td><td> ${ \bf 0 . 9 7 6 \pm 0 . 0 0 4 }$ </td><td> ${ \bf 0 . 9 4 9 \pm 0 . 0 0 6 }$ </td><td> ${ \bf 0 . 9 0 3 \pm 0 . 0 1 9 }$ </td><td> $0 . 8 9 9 \pm 0 . 0 1 1$ </td></tr></table>

## 3.3. Comparisons with State-of-the-art Methods

We compare BooF with state-of-the-art methods on the BUSI [17] and BUS-BRA [15] datasets (Table II), which vary in scale and complexity. BooF outperforms all competitors, achieving the highest scores in most evaluation metrics. On BUSI, BooF achieves the best overall performance with an AUC of 0.959, Specificity of 0.962, and F1-score of 0.885, surpassing KRC-APM [18], HoverTrans [19], and REAF [20]. On BUS-BRA, BooF reaches an AUC of 0.976, Recall of 0.903, and F1-score of 0.899, outperforming BD-StableNet [21], SgmaFuse [22], and the baseline model. Beyond accuracy, BooF offers BI-RADS-aligned descriptions, providing interpretable, clinically grounded rationales that facilitate radiologist verification.

## 4. CONCLUSION

In this paper, we propose the Boot-and-Feedback framework, which enables a dynamic and bidirectional interaction between general Multimodal Large Language Models (MLLMs) and domain-specific expert models. By incorporating BI-RADS Lexicon Alignment and Vision Expert Prediction Priors Guidance in the boot stage, followed by multimodal attention fusion in the feedback stage, BooF significantly improves both diagnostic accuracy and interpretability in breast ultrasound analysis. Extensive experiments on two public datasets demonstrate that BooF surpasses state-ofthe-art methods, offering superior diagnostic precision as well as clinically relevant decision rationales. The proposed framework is highly adaptable, making it applicable to other specialized classification tasks in medical diagnostics. Future research could extend the boot-and-feedback approach to multi-round iterative interactions, which may further enhance diagnostic performance and robustness.

## 5. ACKNOWLEDGMENTS

This research was supported by the Australian Government through the Australian Research Council’s DECRA funding scheme (Grant No. DE250100030) and Discovery Project funding scheme (Grant No. DP260100218).

## 6. REFERENCES

[1] World Health Organization, “World health organization (who),” https://www.who.int/, 2024, accessed: 2024-05-12.

[2] K. M. Kelly, J. Dean, S.-J. Lee, and W. S. Comulada, “Breast cancer detection: radiologists’ performance using mammography with and without automated whole-breast ultrasound,” European radiology, vol. 20, pp. 2557–2564, 2010.

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2016, pp. 770– 778.

[4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[5] S. Zhang, Y. Xu, N. Usuyama, H. Xu, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri et al., “Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs,” arXiv preprint arXiv:2303.00915, 2023.

[6] S. L. Fleming, A. Lozano, W. J. Haberkorn, J. A. Jindal, E. Reis, R. Thapa, L. Blankemeier, J. Z. Genkins, E. Steinberg, A. Nayak et al., “Medalign: A clinician-generated dataset for instruction following with electronic medical records,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 38, no. 20, 2024, pp. 22 021–22 030.

[7] Y. Gao, D. Gu, M. Zhou, and D. Metaxas, “Aligning human knowledge with visual concepts towards explainable medical image classification,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 46–56.

[8] X. Fang, Y. Lin, D. Zhang, K.-T. Cheng, and H. Chen, “Aligning medical images with general knowledge from large language models,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 57–67.

[9] Y. Dai, K. Wang, B. Gao, Y. Jiang, W. Wang, Q. Ke, and J. Cai, “Cinedub: Scaling end-to-end video dubbing to multispeaker dialogues with coherent sound effects,” arXiv preprint arXiv:2608.15734, 2026.

[10] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[11] American College of Radiology, BI-RADS: Breast Imaging Reporting and Data System, 5th ed. Reston, VA: American College of Radiology, 2013.

[12] C.-H. H. Yang, Y. Gu, Y.-C. Liu, S. Ghosh, I. Bulyko, and A. Stolcke, “Generative speech recognition error correction with large language models and task-activating prompting,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.

[13] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877– 1901, 2020.

[14] Y. Dai, H. Chen, J. Du, X. Ding, N. Ding, F. Jiang, and C.- H. Lee, “Improving audio-visual speech recognition by lipsubword correlation based visual pre-training and cross-modal fusion encoder,” in 2023 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2023, pp. 2627–2632.

[15] W. Gomez-Flores, M. J. Gregorio-Calas, and W. Coelho de Al-´ buquerque Pereira, “Bus-bra: a breast ultrasound dataset for assessing computer-aided diagnosis systems,” Medical physics, vol. 51, no. 4, pp. 3110–3123, 2024.

[16] O. N. Manzari, H. Ahmadabadi, H. Kashiani, S. B. Shokouhi, and A. Ayatollahi, “Medvit: a robust vision transformer for generalized medical image classification,” Computers in biology and medicine, vol. 157, p. 106791, 2023.

[17] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, “Dataset of breast ultrasound images,” Data in brief, vol. 28, p. 104863, 2020.

[18] Y. Lin, H. Wang, and J. Jiang, “Krc-apm: Key region cutting and artificial prior model for breast cancer recognition in ultrasound images,” Expert Systems with Applications, vol. 257, p. 125092, 2024.

[19] Y. Mo, C. Han, Y. Liu, M. Liu, Z. Shi, J. Lin, B. Zhao, C. Huang, B. Qiu, Y. Cui et al., “Hover-trans: Anatomyaware hover-transformer for roi-free breast cancer diagnosis in ultrasound images,” IEEE Transactions on Medical Imaging, vol. 42, no. 6, pp. 1696–1706, 2023.

[20] Z. Zhang, J. W. Lim, Y. Zheng, B. Chen, D. Chen, and Y. Lin, “Reaf: Roi extraction and adaptive fusion for breast cancer diagnosis in ultrasound images,” in 2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2023, pp. 3422–3429.

[21] H. Qu, G. Chen, T. Li, M. Zou, J. Liu, C. Dong, Y. Tian, C. Liu, and X. Cui, “Bd-stablenet: a deep stable learning model with an automatic lesion area detection function for predicting malignancy in bi-rads category 3–4a lesions,” Physics in Medicine & Biology, vol. 69, no. 24, p. 245002, 2024.

[22] G. Dai, C. Wang, Q. Tang, Y. Zhang, D. Dai, L. Qiao, J. Yan, and H. Chen, “Interpretable breast cancer diagnosis using histopathology and lesion mask as domain concepts conditional simulation ultrasonography,” Information Fusion, p. 103343, 2025.

[23] H. Wang, J. Du, Y. Dai, C.-H. Lee, Y. Ren, and Y. Liu, “Improving multi-modal emotion recognition using entropybased fusion and pruning-based network architecture optimization,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 11 766–11 770.

[24] A. Yan, J. McAuley, X. Lu, J. Du, E. Y. Chang, A. Gentili, and C.-N. Hsu, “Radbert: adapting transformer-based language models to radiology,” Radiology: Artificial Intelligence, vol. 4, no. 4, p. e210258, 2022.