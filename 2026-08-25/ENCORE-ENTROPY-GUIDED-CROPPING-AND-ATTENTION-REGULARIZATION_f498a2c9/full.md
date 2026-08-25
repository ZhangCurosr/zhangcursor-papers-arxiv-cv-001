# ENCORE: ENTROPY-GUIDED CROPPING AND ATTENTION REGULARIZATION FOR ROBUST VISION–LANGUAGE UNDERSTANDING

Yuanhao Sun, Huawei Ji, Jiaxin Ding<sup>\*</sup>, Luoyi Fu, Xinbing Wang.

Shanghai Jiao Tong University, Shanghai, China

## ABSTRACT

Vision–Language Models (VLMs) perform well on diverse vision–language tasks, but transformer-based visual encoders split images into fixed-resolution sub-images, compromising object integrity in lightweight VLMs. Existing methods only focus on the visual modality and fail to dynamically preserve the integrity of prompt-relevant regions, limiting performance. In this work, we observe that the early-layer image–text entropy of cross-modal attention strongly correlates with answer grounding quality and task accuracy. Building on this finding, we propose ENCORE, an entropy-guided framework with two components: At inference, an Entropybased Cropping Strategy (ECS) evaluates a small set of candidate crops and selects the one with minimal entropy, preserving contiguous regions relevant to the prompt. At training, Entropy Regularization Training (ERT) augments next-token prediction with an entropy term that sharpens attention on key visual tokens while down-weighting irrelevant ones. Experiments on ten VQA benchmarks show that ENCORE, fine-tuning only 0.14% of parameters, achieves an average 1.43% accuracy gain and state-of-the-art performance among recent 2B-parameter VLMs. Our code is released in https://github.com/baokou-fw2/ENCORE.

Index Terms— Lightweight large vision-language model, Fine-grained perception

## 1. INTRODUCTION

Recently, Vision–Language Models (VLMs) [19, 23, 7] have achieved remarkable performance across diverse vision–language tasks, demonstrating strong general reasoning ability. For fine-grained perception, visual encoders such as ViT [5] and SAM [9] typically partition images into numerous high-resolution patches. However, this strategy often compromises the integrity of prompt-relevant regions—for instance, the word ‘Breakdown’ may be split into ‘Break and ‘down’, leading to semantic fragmentation [12, 8] and impairing holistic visual understanding.

Existing solutions fall into two categories: architecture optimization and image preprocessing. TextMonkey [12] employs sliding-window attention to maintain image continuity but still suffers from patch-based fragmentation [8]. Preprocessing approaches [4, 21, 8] leverage dynamic cropping, resizing, thumbnails, or multi-scale inputs, yet they remain limited to the visual modality and fail to ensure integrity of prompt-relevant regions, leaving efficient high-resolution inputs unresolved. To address this challenge, we examine how VLMs leverage textual prompts to interpret fragmented images. Since early layers are known to align visual and textual semantics effectively [24, 25], we compute cross-modal attention by applying a temperature-scaled softmax over CLIP similarities [15]. Our analysis (Fig. 1) reveals a clear trend: lower cross-modal entropy in early layers consistently corresponds to more complete answer-grounding coverage and higher VQA accuracy, highlighting entropy as a reliable indicator of semantic integrity.

Based on these findings, we propose ENCORE, an entropy-guided optimization framework for lightweight VLMs with two stages: ECS computes image–text entropy from early VLM layers over predefined aspect ratios, cropping by the smallest entropy to preserve contiguous prompt-relevant semantics; ERT uses this entropy as a self-supervised signal to help the model identify key prompt-relevant visual tokens during fine-tuning. ENCORE achieves SOTA among recent 2B-parameter VLMs. Our key contributions are as follows:

• We introduce the image–text entropy—computed from early-layer cross-modal attention distributions—as a principled metric to quantify the degree of semantic fragmentation in prompt-relevant regions, and demonstrate its strong correlation with VLM performance.

• We propose ENCORE, combining ECS for entropyguided cropping to preserve prompt-relevant regions and ERT for entropy-regularized training to improve key visual token extraction.

• With only 0.14% of model parameters trainable via LoRA fine-tuning, ENCORE achieves state-of-the-art performance across ten VQA benchmarks and exhibits strong generalization on recent 2B-parameter VLMs.

![](images/3055e6a8c231f742c5fe2ab3244c369a0cbff595d22761d366a4e47c691b8a0b.jpg)  
(a) Prompt-Relevant Visual Token Distribution

![](images/afd5ba9cc9f4d088c7fa86fbb4bf72c4fd2cea10b41086718a739db71e068933.jpg)  
(b) Correlation Between H and Semantic Fragmentation

![](images/d939a1f4cd75e31712a3857d5b3641d66f6f44a30895f6d7178568b4c8ccd2a1.jpg)  
(c) Correlation Between $\mathrm { H } _ { \mathrm { S } }$ and Model Performance  
Fig. 1. Insights into image–text entropy. (a) Visualization of prompt-relevant visual-token probability distributions from VLM’s second layer; (b) Images cropped by ratio with the n-th smallest $H _ { S }$ and the maximum IoU(intersection over union) between patches and answer-grounding boundary (baseline: original aspect ratio) on VizWiz [1]. Results show that lower $H _ { S }$ indicates higher coverage ratio, corresponding to more accurate visual understanding; (c) Average $H _ { S }$ distributions over first four layers for correctly answered samples on TextVQA [16], HallB [6], SEED-2 [10], RWQA [20] are lower than that of incorrectly answered samples, with a significantly statistic difference in T-test $( t = - 1 1 , p = 3 . 3 3 \times 1 0 ^ { - 2 8 } < 0 . 0 5 )$

## 2. METHOD

We first derive the image–text entropy. Subsequently, we introduce the two primary components of the proposed EN-CORE framework based on this discovery: Entropy-Based Cropping Strategy (ECS) and Entropy Regularization Training (ERT). The overall architecture is illustrated in Fig. 2.

## 2.1. Image-Text Entropy

In typical VLM pipelines, images and texts are first mapped into a shared embedding space for joint reasoning. However, partitioning images into high-resolution patches often splits contiguous object regions. Prior methods [12, 8] mitigate such semantic fragmentation through cropping or encoder designs, yet largely overlook the guiding role of textual prompts [24]. To leverage multi-modal information, we extract image tokens ${ \bf Z } _ { I } \in \mathrm { ~ \bf ~ R ~ } ^ { N \times D }$ and text tokens $\mathbf { Z } _ { T } \in$ $\mathbf { R } ^ { M \times D }$ from the k-th layer, and model the prompt-relevant attention distribution as:

$$
p ( Z _ { I } ^ { i } \mid \mathbf { Z } _ { T } ) = \frac { \exp \left( \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbf { Z } _ { I } ^ { i } \cdot ( \mathbf { Z } _ { T } ^ { j } ) ^ { T } / \tau \right) } { \sum _ { k = 1 } ^ { N } \exp \left( \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbf { Z } _ { I } ^ { k } \cdot ( \mathbf { Z } _ { T } ^ { j } ) ^ { T } / \tau \right) } ,\tag{1}
$$

where M is the prompt length, N the number of visual tokens, and τ the temperature. We then define the image–text entropy to measure fragmentation:

$$
H _ { S } = - \sum _ { i = 1 } ^ { N } p ( Z _ { I } ^ { i } \mid \mathbf { Z } _ { T } ) \log \bigl ( p ( Z _ { I } ^ { i } \mid \mathbf { Z } _ { T } ) + \varepsilon \bigr ) ,\tag{2}
$$

where ε is a small constant. This entropy serves as a unified indicator of the integrity of prompt-relevant visual regions, providing a quantitative basis for analyzing and mitigating semantic fragmentation in VLMs.

## 2.2. ECS: Entropy-Based Cropping Strategy

Based on the definition of image–text entropy $H _ { S } ~ ( { \mathrm { S e c . ~ } } 2 )$ and its correlation with answer-grounding coverage (Fig. 1), a smaller $H _ { S }$ corresponds to more contiguous preservation of prompt-relevant regions. Moreover, from a theoretical perspective, regardless of whether the prompt-relevant regions in an image are independent or distributed across multiple areas, the entropy can reflect their integrity within the range of candidate cropping ratios.

As illustrated in Fig. 2, ECS computes $H _ { S }$ from the first k layers of the VLM, selects the cropping ratio with minimal entropy, and integrates the resulting crops with the originals to form the final visual input.

## 2.3. ERT: Entropy Regularization Training

In VQA tasks, VLMs are typically trained with next-token prediction [3], which lacks explicit suppression of mismatched visual–text pairs. When images are partitioned into many high-resolution patches, this often leads to difficulty in distinguishing key visual tokens from irrelevant ones.

To address this, $H _ { S }$ from the k-th layer during the finetuning stage serves as an entropy regularization term. As shown in Fig. 2, H encourages higher attention on key visual tokens and lower attention on irrelevant tokens. Specifically, we reconstruct the training loss as:

$$
\mathcal { L } = - \sum _ { \ell = 1 } ^ { L } \log p ( y _ { \ell } \mid y _ { < \ell } , x _ { \le \ell } ) + \alpha H _ { S } ,\tag{3}
$$

where $\alpha > 0$ controls the regularization strength. For promptrelevant tokens $( p _ { i } > 1 / e )$ , minimizing $H _ { S }$ reinforces their probabilities in line with cross-entropy, while for irrelevant tokens $( p _ { i } < 1 / e )$ , the self-information term suppresses noisy pairs, resembling negative sampling in contrastive learning. These effects together sharpen attention toward key visual tokens in fragmented inputs.

![](images/2feb1f873690c488bff8dc9179b7c618f78a0b7fce376fe3d73f6a5bec3ebf68.jpg)  
Fig. 2. ENCORE framework. (a) ECS: Compute early-layer cross-modal similarities between visual tokens and prompt, derive prompt-relevant distribution, compute its entropy, select minimal-entropy crop ratio to preserve contiguous promptrelevant semantics; (b) ERT: During fine-tuning, augment next-token prediction loss with entropy regularizer to encourage sharpening cross-modal attention for accurate key visual token extraction from fragmented high-res inputs.

Table 1. Comparison results with SOTA VLMs. ENCORE achieves the best results among 2B-parameter VLMs and surpass some larger VLMs in several benchmarks.
<table><tr><td>Model</td><td>|Size</td><td>SEED-2</td><td>Info</td><td>Text</td><td>OCRB</td><td>CCB</td><td>RWQA</td><td>HallB</td><td>POPE</td><td> $\mathrm { M M B } _ { \mathrm { v 1 . 1 } } ^ { \mathrm { E N } }$ </td><td>MMStar</td></tr><tr><td>Deepseek-VL [14]</td><td>2B</td><td>43.3</td><td></td><td>21.057.7</td><td>414</td><td>37.6</td><td>49.7</td><td>27.6</td><td>85.9</td><td>63.8</td><td>39.9</td></tr><tr><td>Qwen2-VL [17]</td><td>2B</td><td>62.4</td><td></td><td>65.5 79.4</td><td>809</td><td>53.5</td><td>62.6</td><td>41.7</td><td>87.8</td><td>72.7</td><td>47.8</td></tr><tr><td>mPLUG-Owl3 [22]</td><td>2B</td><td>44.6</td><td></td><td>29.0 59.2</td><td>434</td><td>33.7</td><td>55.5</td><td>25.1</td><td>85.8</td><td></td><td>41.2</td></tr><tr><td>InternVL-2 [3]</td><td>2B</td><td>60.0</td><td></td><td>58.074.7</td><td>779</td><td>74.3</td><td>57.3</td><td>37.9</td><td>85.2</td><td>70.2</td><td>50.2</td></tr><tr><td>Mini-monkey [8]</td><td>2B</td><td>61.2</td><td></td><td>60.075.7</td><td>802</td><td>75.5</td><td>57.5</td><td>38.7</td><td>86.7</td><td>68.9</td><td>50.4</td></tr><tr><td>InternVL-3 [2]</td><td>2B</td><td>64.6</td><td></td><td>66.1 77.0</td><td>831</td><td>74.9</td><td>64.3</td><td>41.8</td><td>89.6</td><td>78.6</td><td>60.7</td></tr><tr><td>ENCORE(ours)</td><td>2B</td><td>65.5</td><td></td><td>68.0 79.8</td><td>845</td><td>77.9</td><td>66.3</td><td>43.1</td><td>90.2</td><td>79.2</td><td>61.1</td></tr></table>

## 3. EXPERIMENT

## 3.1. Implementation Details

Following previous work [4], we fine-tune InternVL3-2B on the original training datasets and test on ten general multimodal understanding benchmarks of InternVL3-2B including TextVQA , OCRBench, SEED2-Plus and InfoVQA, CCBench, RealWorldQA, HallBench, POPE, MMBenchv1.1-EN and MMStar. The base learning rate is 4e-8. τ is set as 0.1 in ERT and 1 in ECS. Layer k is set as 2 in ERT and 1 in ECS. ε is selected as 1e-32. α is set as 0.1. Cropping range for ECS is selected from 1 to 4.

## 3.2. Comparison of Other VLMs

The results show that on text-rich datasets such as TextVQA and OCRBench, ENCORE surpasses InternVL3-2B by 2.8% and 1.4%, confirming that ERT enhances capture of questionrelevant text. Beyond OCR, ENCORE also improves Real-WorldQA (+2.0%) and CCBench (+2.4%), showing that ECS effectively preserves prompt-relevant regions. Moreover, EN-CORE consistently outperforms all 2B-parameter baselines, reaching new highs on SEED-2 (65.5), InfoVQA (68.0), Hall-Bench (43.1), POPE (90.2) and MMBench(79.2).

Table 2. Comparison of ECS and other cropping strategies. Results are tested with InternVL3-2B.
<table><tr><td>Strategy</td><td>TextVQA</td><td>OCRBench</td><td>RWQA</td><td>CCBench</td></tr><tr><td>DHR [3]</td><td>77.0</td><td>831</td><td>64.3</td><td>74.9</td></tr><tr><td>FHR [11]</td><td>76.9</td><td>825</td><td>63.6</td><td>73.5</td></tr><tr><td>MSAC [8]</td><td>78.8</td><td>834</td><td>63.0</td><td>75.0</td></tr><tr><td>RTC [13]</td><td>77.2</td><td>827</td><td>63.9</td><td>74.8</td></tr><tr><td>ECS (ours)</td><td>79.1</td><td>837</td><td>65.1</td><td>75.3</td></tr></table>

Overall, ENCORE delivers broad gains with only 0.14% trainable parameters and minimal overhead, achieving an excellent trade-off between efficiency and accuracy.

![](images/e9f5e13f5fe11c8703f16517d6221965344ff0b52af78e684972d8182d0272d9.jpg)  
Q: Where is the dog relative to the stuffed animal ?  
Fig. 3. Ablation study of $H _ { S }$ under varying temperature τ. (a) Visualization of $\boldsymbol { \tau } ^ { \prime } \mathbf { s }$ impact. (b) shows the $\boldsymbol { \tau } ^ { \prime } \mathbf { s }$ impact on average score.

## 3.3. Ablation Study of ECS

In this section, we evaluate the generalization of the Entropy-Based Cropping Strategy (ECS) across different models and compare it with alternative cropping methods.

First, we analyze the VLM layer for computing $H _ { S } { \mathrm { : } }$ the first or second layer delivers comparable performance, while deeper layers show a marked drop, so we adopt the first layer for subsequent entropy calculations. Increasing τ from 0.1 to 50 (Fig. 3(a)) shifts the prompt-relevant token distribution from sharp to uniform, driving $H _ { S }$ upward; model performance peaks at $\tau = 1$ (Fig. 3(b)). Moreover, with the max patch count set to 6 (as in InternVL3-2B [2]) and $H _ { S }$ computed via the first layer, average inference time only increases FLOPs by 7.2% and inference latency by 10%, demonstrating that entropy-guided cropping provides performance gains with relatively low burden.

Then, we compare ECS with several alternative cropping strategies (Table 2). On benchmarks such as OCRBench and TextVQA, MSAC performs comparably to ECS, likely because these datasets emphasize OCR tasks with limited prompt semantics. However, on benchmarks with complex visual semantics—such as SEED-2 and RWQA—MSAC and other baselines degrade performance, whereas ECS consistently improves accuracy. Although ECS introduces additional crops, the overall computational overhead remains modest.

## 3.4. Ablation Study of ERT

We select the POPE, RWQA, OCRBench, and CCBench benchmarks to systematically probe four factors influencing Entropy Regularization Training (ERT): whether to unfreeze subsequent layers, the choice of temperature coefficient τ, the number of fine-tuned layers, and the entropy regularization weight α. To isolate the contribution of ERT and its plugand-play flexibility, ablation models are trained and tested with dynamic high-resolution cropping strategy according to original setting of InternVL3 [3].

Fig. 4(a) shows that freezing all subsequent layers yields superior accuracy, suggesting that unfreezing deep layers disrupts the reasoning capabilities established by InternVL3’s Mixed Preference Optimization pretraining [18]. Furthermore, a small temperature $( \tau ~ < ~ 1 )$ sharpens the promptrelevant visual-token distribution, enhancing token discrimination and overall performance. In Fig. 4(b), with the total number of trainable parameters held constant, ERT is most effective when applied to the initial layers, whereas fine-tuning deeper layers underperforms the standard NTP.

![](images/41a89c22ab8d3f477530313be451794a94f369d11ee1bb06ad330e43224a754b.jpg)

![](images/8f212b10e987070231ec6e560e4c0bfedafeea05e143cc2b766be75ac2faf1f1.jpg)

Fig. 4. Ablation study of Entropy Regularization Training (ERT). (a) shows average score on POPE, OCR-Bench, RWQA and CCBench; In (a)–(b), NTP denotes next-token-prediction fine-tuning.  
![](images/c0a7e2f0415b10c6a52d87188eae8bbe4f352f176ee3239ecc51ddab0351992b.jpg)  
Average Cross-modal Attention of 0\~4 Layers  
Fig. 5. Visualization of cross-attention matrices in layer 2 after Entropy Regularization Training (ERT).

Fig. 5 visualizes cross-attention matrices before and after ERT in the early layers: without ERT, attention is dispersed across regions, while ERT concentrates attention on key visual tokens relevant to the prompt, which aligns with ERT’s role in improving key visual token extraction.

## 4. CONCLUSION

To address semantic fragmentation in lightweight VLMs, we propose ENCORE—an entropy-guided framework that preserves prompt-relevant regions and sharpens key visual token extraction. Experiments on multiple VQA benchmarks show that ENCORE achieves state-of-the-art performance among 2B-parameter VLMs, demonstrating its effectiveness.

## 5. REFERENCES

[1] C. Chen, S. Anjum, and D. Gurari, “Grounding answers for visual questions asked by visually impaired people,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 19 098–19 107.

[2] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.

[3] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., “How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,” Science China Information Sciences, vol. 67, no. 12, p. 220101, 2024.

[4] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 24 185–24 198.

[5] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021.

[6] T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob et al., “Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 375–14 385.

[7] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi et al., “Deepseek-r1 incentivizes reasoning in llms through reinforcement learning,” Nature, vol. 645, no. 8081, pp. 633–638, 2025.

[8] M. Huang, Y. Liu, D. Liang, L. Jin, and X. Bai, “Mini-monkey: Multi-scale adaptive cropping for multimodal large language models,” CoRR, 2024.

[9] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015– 4026.

[10] B. Li, Y. Ge, Y. Chen, Y. Ge, R. Zhang, and Y. Shan, “Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension,” arXiv preprint arXiv:2404.16790, 2024.

[11] Z. Li, B. Yang, Q. Liu, Z. Ma, S. Zhang, J. Yang, Y. Sun, Y. Liu, and X. Bai, “Monkey: Image resolution and text label are important things for large multi-modal models,” in proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 26 763–26 773.

[12] Y. Liu, B. Yang, Q. Liu, Z. Li, Z. Ma, S. Zhang, and X. Bai, “Textmonkey: An ocr-free large multimodal model for understanding document,” IEEE transactions on pattern analysis and machine intelligence, 2026.

[13] D. Lu, Y. Sun, Z. Zhang, L. Huang, J. Zeng, M. Shu, and H. Cao, “Internvl-x: Advancing and accelerating internvl series with efficient visual token compression,” arXiv preprint arXiv:2503.21307, 2025.

[14] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, H. Yang et al., “Deepseek-vl: towards real-world vision-language understanding,” arXiv preprint arXiv:2403.05525, 2024.

[15] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[16] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.

[17] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[18] W. Wang, Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, J. Zhu, X. Zhu, L. Lu, Y. Qiao et al., “Enhancing the reasoning ability of multimodal large language models via mixed preference optimization,” arXiv preprint arXiv:2411.10442, 2024.

[19] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu et al., “Emu3: Next-token prediction is all you need,” arXiv preprint arXiv:2409.18869, 2024.

[20] x.ai., “Grok 1.5v: The future of ai models,” Tech. Rep., 2024.

[21] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, G. Xu, C. Li, J. Tian, Q. Qian, J. Zhang et al., “Ureader: Universal ocr-free visuallysituated language understanding with multimodal large language model,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 2841–2858.

[22] J. Ye, H. Xu, H. Liu, A. Hu, M. Yan, Q. Qian, J. Zhang, F. Huang, and J. Zhou, “mplug-owl3: Towards long imagesequence understanding in multi-modal large language models,” arXiv preprint arXiv:2408.04840, 2024.

[23] P. Zhang, X. Dong, Y. Cao, Y. Zang, R. Qian, X. Wei, L. Chen, Y. Li, J. Niu, S. Ding et al., “Internlm-xcomposer2. 5-omnilive: A comprehensive multimodal system for longterm streaming video and audio interactions,” arXiv preprint arXiv:2412.09596, 2024.

[24] S. Zhang, Q. Fang, Z. Yang, and Y. Feng, “Llava-mini: Efficient image and video large multimodal models with one vision token,” CoRR, 2025.

[25] Y. Zhang, C.-K. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng, D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer et al., “Sparsevlm: Visual token sparsification for efficient vision-language model inference,” in International Conference on Machine Learning, 2025.