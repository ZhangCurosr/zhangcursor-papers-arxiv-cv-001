# Detecting Unreliable Classifications from Quantized Models

Ananya Kotla

Jorge T. Gomez

## Abstract

Post-training INT8 quantization is typically validated by checking accuracy retention, which requires ground-truth labels often unavailable in low-resource clinical settings. We show a label-free alternative: comparing a model’s own output before and after quantization. We compute cosine similarity between the FP32 and INT8 models’ output probability vectors for each prediction, which requires only a single forward pass. This output-agreement similarity separates loweraccuracy predictions across four evaluated architectures (ResNet-50, VGG-16- BN, Wide ResNet-50, and ViT-B/16) under fixed-percentile splits, outperforming a confidence-only baseline for all four. These results show that quantized models can be compared to their baselines to assess their drift and determine whether they are well suited for clinical deployment.

## 1 Introduction

Model compression techniques such as quantization reduce memory requirements and energy consumption, allowing models to run on less capable hardware. This is especially relevant for edge deployment in low-resource settings, which are becoming further constrained by the rising cost of hardware [14] and the growing requirements of deep learning (DL) models. Clinical settings are commonly resource constrained, outside of highly specialized and high-tier hospitals, and commonly require using post-training quantization [8, 11].

However, while quantized models often perform at or below the baseline set by their non-quantized counterparts, especially strong quantization such as INT8 and INT4 [9], the size of this gap is often unknown and varies per model and setting. Understanding this gap is important for clinical deployment, but doing so usually requires labeled test data, which can be difficult to obtain for medical applications. Expert annotation are often required, which is expensive and time consuming, particularly for segmentation tasks, and such annotations are frequently unavailable in low-resource settings.

In this work, we show that for a mammography dataset [7] we can leverage the models own output before and after quantization to identify when the quantized model drifts too far from the nonquantized model leading to less reliable predictions. This can alleviate the need for annotated data and help clinicians understand if the quantized models meets their deployment needs without having to annotate any additional data.

Our contributions are: 1) Exploring the impact of quantization of three different convolution based models (ResNet-50, VGG-16-BN, and Wide ResNet-50) and a transformer based model (ViT-B/16) on the publicly available mammography dataset CBIS-DDSM for cancer detection 2) Using cosine similarity of the models output for failure case detection using the available test data, 3) Demonstration that this method works across different architectures, and seeds.

## 2 Related Work

Quant-XAI [6] introduced “explainability drift,” relying on ground-truth evaluation and a fakequantization proxy; we extend this with a backprop-free, label-free per-case signal on the true converted model. Catalan-Tatjer et al. [1] showed LLM quantization robustness is governed by training dynamics. Saliency-guided QAT [10] intervenes during training instead. Cross-model disagreement is a recognized label-free signal elsewhere [2, 4, 12]; we extend this to quantization, across architectures.

## 3 Method

Dataset. We use CBIS-DDSM [7], a curated mammography dataset with radiologist-annotated tumor masks, for binary classification of each breast mass or calcification as malignant or benign. We construct a patient-grouped split using the official train/test designation, removing 31 patients appearing in both splits, yielding 2,780 training images (2,417 train / 363 validation) and 704 heldout test images. Of these, 384 have an available ROI mask and are used for experimental analyses; classification metrics use the full 704.

Model, training, and quantization. We train and evaluate ResNet-50 [5], VGG-16-BN [13], Wide ResNet-50 [15], and ViT-B/16 [3]. All models are pre-trained on ImageNet-1K, with a randomly initialized four-layer classification head added for the final malignant-vs-benign prediction. To reduce overfitting, we freeze the earliest layers of each backbone: conv1, bn1, layer1, and layer2 for ResNet-50 and Wide ResNet-50; the first 13 convolutional layers for VGG-16-BN; and the patch embedding together with the earliest transformer encoder blocks for ViT-B/16. We apply post-training static INT8 quantization via PyTorch’s FX graph API (qnnpack backend, 500 calibration images), using the fully converted INT8 model for all reported results. Full training hyperparameters are reported in Appendix A. All four architectures undergo the same per-case statistical analysis in Sections 4.2–4.3; ViT-B/16’s per-bin trend is reported separately in Appendix B.

Model output disagreement. For each test image, we compute cosine similarity between the FP32 and INT8 models’ softmax output probability vectors, requiring one forward pass per precision per image:

$$
\mathrm { s i m } _ { \mathrm { c o s } } ( p _ { \mathrm { F P 3 2 } } , p _ { \mathrm { I N T 8 } } ) = \frac { \sum _ { i } p _ { \mathrm { F P 3 2 } , i } p _ { \mathrm { I N T 8 } , i } } { \sqrt { \sum _ { i } p _ { \mathrm { F P 3 2 } , i } ^ { 2 } } \sqrt { \sum _ { i } p _ { \mathrm { I N T 8 } , i } ^ { 2 } } }\tag{1}
$$

where p<sub>FP32</sub>, p<sub>INT8</sub> $\in \mathbb { R } ^ { 2 }$ are the output probability vectors for the same input under each precision. We classify a case as reorganized if $\mathrm { s i m } _ { \mathrm { c o s } }$ falls below 0.8; for the full accuracy-vs-similarity trend (Figure 1, Appendix B), we additionally bin predictions into four similarity ranges (< 0.80, 0.80– 0.90, 0.90–0.95, 0.95–1.00). All analyses use 3 seeds and the full 384-image mask-available test set (1,152 predictions per regimen).

## 4 Experiments

## 4.1 FP32 and INT8 Performance

Evaluating FP32 vs. fully converted INT8 ResNet-50 across 3 seeds (n = 1,152): accuracy drops from 73.0% to 66.5% (−6.5 pts). Aggregate checks alone may miss sample-level performance drops.

Table 1 reports the aggregate FP32-vs-INT8 comparison (Accuracy, AUC, F1) for all four architectures. FP32 baselines are close across all architectures (ResNet-50: 73.0, VGG-16-BN: 72.4, ViT-B/16: 71.7, Wide ResNet-50: 69.5), with standard deviation overlapping across all models. The impact of INT8 quantization varies per model: ResNet-50 and ViT-B/16 drop about 6% in accuracy, while VGG-16-BN and Wide ResNet-50 show only a 1%–2% decrease. AUC drops are architecture dependent rather than uniform: 2% for VGG-16-BN, 5% for ResNet-50 and ViT-B/16, and 8% for Wide ResNet-50. F1 score drops substantially for ViT-B/16 and Wide ResNet-50 specifically, suggesting these two models would benefit from recalibration after quantization.

Table 1: FP32-vs-INT8 comparison, mean ± std across 3 seeds (n = 384 per seed), full 704-image test set.
<table><tr><td rowspan="2">Architecture</td><td colspan="3">FP32</td><td colspan="3">INT8</td></tr><tr><td>Acc</td><td>AUC</td><td>F1</td><td>Acc</td><td>AUC</td><td>F1</td></tr><tr><td> $\mathtt { R e s N e t - } 5 0$ </td><td> $7 3 . 0 { \pm } 3 . 1 $ </td><td> $0 . 8 0 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $6 6 . 9 { \pm } 2 . 5 $ </td><td> $6 6 . 5 { \pm } 1 . 7 $ </td><td> $0 . 7 5 1 { \pm } 0 . 0 1 2$ </td><td> $6 3 . 0 { \pm } 1 . 3 $ </td></tr><tr><td> $\mathrm { V G G } { - } 1 6 { - } \mathrm { B N }$ </td><td> $7 2 . 4 2 1 . 0$ </td><td> $0 . 7 8 4 { \pm } 0 . 0 0 5$ </td><td> $6 5 . 7 { \pm } 1 . 7 $ </td><td> $7 0 . 3 { \pm } 2 . 6 $ </td><td> $0 . 7 6 8 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $6 2 . 2 { \pm } 3 . 3 $ </td></tr><tr><td>Wide ResNet-50</td><td> $6 9 . 5 { \pm } 2 . 4 $ </td><td> $0 . 7 8 4 { \pm } 0 . 0 0 2$ </td><td> $6 5 . 1 { \pm } 1 . 5 $ </td><td> $6 8 . 5 { \pm } 0 . 7 \ $ </td><td> $0 . 7 2 1 { \pm } 0 . 0 2 0$ </td><td> $5 1 . 5 { \pm } 8 . 5 $ </td></tr><tr><td>ViT-B/16</td><td> $7 1 . 7 { \pm } 2 . 1 $ </td><td> $0 . 7 7 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $6 1 . 3 { \pm } 4 . 6 $ </td><td> $6 5 . 3 { \pm } 4 . 0 $ </td><td> $0 . 7 3 2 { \scriptstyle \pm 0 . 0 3 4 }$ </td><td> $2 3 . 7 { \pm } 2 2 . 1 $ </td></tr></table>

![](images/1cb39818e15abfee3207f9f6c4f10841605084dec14c151a8a3bc09b6ab9c077.jpg)  
Figure 1: FP32 and INT8 accuracy and standard deviation binned by the FP32-vs-INT8 output embeddings cosine similarity for ResNet-50, Wide ResNet-50, VGG-16-BN, and ViT-B/16.

## 4.2 Output Disagreement

Stratification by output similarity is significant for all four architectures at threshold 0.8 (predictions pooled across all 3 seeds, $n = 1 , 1 5 2$ per architecture): ResNet-50 (n = 123, 30.1% vs. 70.8%), VGG-16-BN (n = 56, 28.6% vs. 72.4%), Wide ResNet-50 (n = 211, 44.5% vs. 73.9%), ViT-B/16 $( n = 1 1 4 , 2 6 . 3 \% \ \mathrm { v s . \ 6 9 . 6 \% ) }$

Figure 1 shows this per seed before pooling: accuracy binned by similarity within each seed, then averaged. $\mathrm { R e s N e t } { - } \bar { 5 } 0 ^ { \circ } \mathrm { s } \ 4$ bins all contain data from all 3 seeds and follow a monotonic trend $( 3 0 . 7 \% \pm 8 . 6 \%$ lowest bin vs. $7 3 . 7 \% \pm 1 . 5 \%$ highest); Wide ResNet-50, VGG-16-BN, and ViT-B/16 follow a similar trend with more per-seed noise in intermediate bins.

Per-seed threshold sensitivity confirms this holds for all four architectures in every seed at threshold 0.8 (Appendix D), though VGG-16-BN’s smallest seed (1207, n = 4) and ViT-B/16’s smallest seed $( 1 2 0 7 , n = 4 )$ both reflect a genuinely lower baseline disagreement rate under static quantization relative to the other architectures.

## 4.3 Comparison to a Confidence-Only Baseline

Output disagreement costs more to compute than confidence: it needs the original FP32 model plus two forward passes per image (one FP32, one INT8), while confidence comes from the INT8 model’s own output. We explore if this can be used a signal as well as decribed in Section 3. Table 4 (Appendix C) reports the result: Output disagreement separates accurate from inaccurate predictions more effectively than confidence, for all four architectures.

## 5 Discussion

We trained three convolutional architectures and one transformer-based architecture, and found that output cosine similarity correlates with accuracy degradation under INT8 quantization. In particular, below a threshold of 0.8, every architecture’s performance fell below that of the baseline model. This suggests that as the outputs drift away from the baseline FP32 model, classification becomes less reliable. In a real-world setting without labels, this method could potentially serve as an uncertainty: providing clinicians with the percentage of samples that fall below this threshold, and letting them decide whether that level of uncertainty is acceptable for the given application before deployment. Future work will focus on exploring whether this signal holds for other datasets, as this paper has focused only on mammography data. Additionally, we will work toward improving the separation of cases where the FP32 model and its quantized counterpart disagree, thereby increasing confidence that a quantized model is making a mistake past a certain threshold and providing clinicians with a better label-free metric. Lastly, we will explore leveraging these unreliable cases to aid in training and improve accuracy for quantized models.

## 6 Conclusion

In this work, we investigated output cosine similarity as a label-free signal for detecting accuracy degradation under INT8 quantization, evaluating three convolutional architectures and one transformer-based architecture on mammography data. We show that this similarity correlates with accuracy degradation relative to the FP32 baseline.

## Acknowledgments and Disclosure of Funding

## References

[1] Albert Catalan-Tatjer, Niccolò Ajroldi, and Jonas Geiping. Training dynamics impact posttraining quantization robustness. In International Conference on Learning Representations (ICLR), pages 64294–64319, 2026. URL https://proceedings.iclr.cc/paper\_files/ paper/2026/hash/6937a7c60361d05f5b6cfa04d2c27a5b-Abstract-Conference. html.

[2] Ailin Deng, Miao Xiong, and Bryan Hooi. Great models think alike: Improving model reliability via inter-model latent agreement. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 7675–7693. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr. press/v202/deng23f.html.

[3] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

[4] Matt Gorbett and Suman Jana. Cross-model disagreement as a label-free correctness signal, 2026. URL https://arxiv.org/abs/2603.25450.

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[6] Sunzil Khandaker. Quant-xai: quantifying and mitigating explainability drift under int8 quantization in edge-deployed visual classification (plant disease imaging case study), 03 2026.

[7] Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi, Kanae Kawai Miyake, Mia Gorovoy, and Daniel L. Rubin. A curated mammography data set for use in computer-aided detection and diagnosis research. Scientific Data, 4(1):170177, Dec 2017. ISSN 2052-4463. doi: 10. 1038/sdata.2017.177. URL https://doi.org/10.1038/sdata.2017.177.

[8] Geert Litjens, Thijs Kooi, Babak Ehteshami Bejnordi, Arnaud Arindra Adiyoso Setio, Francesco Ciompi, Mohsen Ghafoorian, Jeroen A. W. M. van der Laak, Bram van Ginneken, and Clara I. Sánchez. A survey on deep learning in medical image analysis. CoRR, abs/1702.05747, 2017. URL http://arxiv.org/abs/1702.05747.

[9] Yury Nahshan, Brian Chmiel, Chaim Baskin, Evgenii Zheltonozhskii, Ron Banner, Alex M. Bronstein, and Avi Mendelson. Loss aware post-training quantization, 2019.

[10] Elmira Mousa Rezabeyk, Salar Beigzad, Yasin Hamzavi, Mohsen Bagheritabar, and Seyedeh Sogol Mirikhoozani. Saliency assisted quantization for neural networks, 2024. URL https://arxiv.org/abs/2411.05858.

[11] E. F. Rivera-Guzmán, L. F. Guerrero-Vásquez, and V. E. Robles-Bykbaev. Quantization of deep neural networks for medical image analysis: A systematic review and meta-analysis. Technologies, 14(1):76, 2026.

[12] Han-Jay Shu, Wei-Ning Chiu, Shun-Ting Chang, Meng-Ping Huang, Takeshi Tohyama, Ahram Han, and Po-Chih Kuo. Uncovering overconfident failures in cxr models via augmentationsensitivity risk scoring, 2025. URL https://arxiv.org/abs/2510.01683.

[13] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014. URL https://api.semanticscholar. org/CorpusID:14124313.

[14] World Semiconductor Trade Statistics. Global semiconductor market surges beyond USD 1.5 trillion in 2026. https://www.wsts.org/esraCMS/extension/media/f/WST/7618/ WSTS\_FC-Release-2026-May.pdf, June 2026. WSTS Spring 2026 forecast release. Accessed: 25 August 2026.

[15] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. CoRR, abs/1605.07146, 2016. URL http://arxiv.org/abs/1605.07146.

## A Full Hyperparameters

## A.1 Training Details

The results reported in this paper use a single-stage (capacity-clamped) fine-tuning regimen on the CBIS-DDSM dataset [7], trained on ResNet-50 [5], VGG-16-BN [13], Wide ResNet-50-2 [15], and ViT-B/16 [3]. All models are initialized from ImageNet-1K pretrained weights. Table 2 summarizes the core training hyperparameters; architecture-specific adaptations follow below.

## A.1.1 Training Regimen

The earliest backbone layers of each architecture are frozen for the entire training run, while subsequent layers and the classification head train jointly at a single fixed learning rate; the specific frozen layers per architecture are given in Section 3.

The classification head is a four-layer MLP (2048 → 512 → 256 → 128 → 2 for ResNet-50 and Wide ResNet-50-2; analogous dimensions adapted to each backbone’s output dimensionality for VGG-16-BN and ViT-B/16) with ReLU activations and dropout (p = 0.2) after each hidden layer.

Table 2: Training hyperparameters for the single-stage fine-tuning regimen.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule</td><td>ReduceLROnPlateau (factor 0.5, patience 5)</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Max epochs</td><td>30</td></tr><tr><td>Early stopping patience</td><td>7 epochs</td></tr><tr><td>Checkpoint selection</td><td>Best validation AUC</td></tr><tr><td>Loss function</td><td>Cross-entropy</td></tr><tr><td>Trainable parameters (ResNet-50)</td><td>23.3M / 24.7M (94.2%)</td></tr></table>

## A.1.2 Data and Augmentation

Training images are resized to 224 × 224 and enhanced with contrast-limited adaptive histogram equalization (CLAHE; clip limit 2.0, tile grid $8 \times 8 )$ prior to augmentation. Training-time augmentation consists of random horizontal and vertical flips $( p = 0 . 5$ each), random rotation $( \pm 1 5 ^ { \circ } )$ , and Gaussian blur (kernel size $3 \times 3 , \sigma \in [ 0 . 1 , 2 . 0 ] ;$ ), with bilinear resizing and pixel normalization (mean/std = 0.5). Validation and test images receive only resizing and normalization, no augmentation is applied at evaluation time.

## A.1.3 Statistical Significance

To assess whether observed effects reflect consistent trends across initializations, we repeat all experiments across all four architectures over 3 random seeds: 1207, 804, and 510. We report mean ± standard deviation across these 3 seeds, computed per seed before pooling. Seeds control weight initialization of the classification head and data shuffling order; other sources of non-determinism are left at PyTorch defaults.

## A.1.4 Quantization

Post-training static quantization is performed using PyTorch’s FX graph mode quantization API (prepare\_fx/convert\_fx) with the qnnpack backend, targeting INT8 weights and activations. Calibration uses 500 images sampled from the training set (fixed seed 42) processed in evaluation mode to record activation statistics. qnnpack was selected as the quantization backend available in our ARM-based compute environment.

## A.1.5 Compute

Experiments were run on an Apple M4 Pro system (48 GB unified memory) using PyTorch’s MPS backend for training and CPU for INT8 quantization and evaluation. Training across 3 seeds for one architecture requires approximately 45 minutes. INT8 quantization and evaluation of a single checkpoint on the 704-image test set requires under 5 minutes on CPU. Feature-embeddings disagreement requires a single forward pass per precision per image, taking under 1 minute on CPU per seed.

## A.1.6 Feature-Embeddings Disagreement: Implementation Details

For each test image and seed, we compute the softmax probability vector $p \in \mathbb { R } ^ { 2 }$ from both the FP32 model and the true converted INT8 model, recording cosine similarity alongside predicted class and correctness.

Per-seed robustness analysis. Threshold sensitivity is computed independently within each seed prior to aggregation to evaluate whether classification splits remain consistent across individual training runs.

Confidence baseline comparison. To compare confidence against feature-embeddings disagreement on a consistent basis, we apply an identical bottom-25th-percentile split $( n = 2 8 8$ per architecture) to feature-embeddings cosine similarity.

## B Per-Similarity-Bin Accuracy: Full Detail

Table 3 reports the full per-bin FP32-vs-INT8 accuracy underlying Figure 1, for all four evaluated architectures, computed per seed before pooling.

All four architectures show accuracy climbing with similarity, with the direction of the FP32-vs-INT8 gap consistent across bins. Minor INT8-exceeds-FP32 crossovers of 0.5–3.6 points occur in the near-ceiling (≥ 0.90 similarity) bins for VGG-16-BN and Wide ResNet-50, and 1.0 points for ViT-B/16, consistent with sampling noise given FP32≈INT8 by construction at that similarity level (ResNet-50 shows none).

Table 3: Per-seed binned accuracy (FP32 vs. INT8), mean ± std across 3 seeds, output cosine similarity bins, all four evaluated architectures. Bins computed independently per seed before pooling.
<table><tr><td>Architecture</td><td>Bin</td><td>FP32 Mean</td><td>FP32 Std</td><td>INT8 Mean</td><td>INT8 Std</td></tr><tr><td>ResNet-50</td><td>&lt; 0.80</td><td>70.1</td><td>7.2</td><td>30.7</td><td>8.6</td></tr><tr><td>ResNet-50</td><td>0.80-0.90</td><td>57.3</td><td>17.4</td><td>43.3</td><td>5.0</td></tr><tr><td>ResNet-50</td><td>0.90-0.95</td><td>73.8</td><td>2.1</td><td>65.5</td><td>6.0</td></tr><tr><td>ResNet-50</td><td>0.95-1.00</td><td>74.3</td><td>3.0</td><td>73.7</td><td>1.5</td></tr><tr><td>VGG-16-BN</td><td>&lt; 0.80</td><td>76.8</td><td>20.8</td><td>22.3</td><td>20.4</td></tr><tr><td>VGG-16-BN</td><td>0.80-0.90</td><td>62.8</td><td>8.9</td><td>59.5</td><td>6.6</td></tr><tr><td>VGG-16-BN</td><td>0.90-0.95</td><td>56.6</td><td>4.4</td><td>49.0</td><td>6.6</td></tr><tr><td>VGG-16-BN</td><td>0.95-1.00</td><td>74.1</td><td>0.7</td><td>74.6</td><td>1.4</td></tr><tr><td>Wide ResNet-50</td><td>&lt; 0.80</td><td>52.4</td><td>6.3</td><td>44.6</td><td>9.4</td></tr><tr><td>Wide ResNet-50</td><td>0.80-0.90</td><td>62.9</td><td>6.5</td><td>58.0</td><td>2.6</td></tr><tr><td>Wide ResNet-50</td><td>0.90-0.95</td><td>58.3</td><td>2.4</td><td>61.9</td><td>9.8</td></tr><tr><td>Wide ResNet-50</td><td>0.95-1.00</td><td>76.9</td><td>2.7</td><td>77.6</td><td>3.6</td></tr><tr><td>ViT-B/16</td><td>&lt; 0.80</td><td>74.5</td><td>0.9</td><td>25.5</td><td>0.9</td></tr><tr><td>ViT-B/16</td><td>0.80-0.90</td><td>69.3</td><td>20.1</td><td>50.5</td><td>17.7</td></tr><tr><td>ViT-B/16</td><td>0.90-0.95</td><td>61.2</td><td>8.3</td><td>53.8</td><td>10.1</td></tr><tr><td>ViT-B/16</td><td>0.95-1.00</td><td>73.5</td><td>5.0</td><td>74.5</td><td>3.5</td></tr></table>

## C Confidence Baseline Comparison: Full Table

Table 4 reports the full comparison underlying Section 4.3: accuracy gap (bottom 25% vs. top 75% by signal value), both signals, fixed n = 288 split per architecture.

Table 4: Accuracy gap (bottom 25% vs. top 75%), confidence vs. output disagreement, fixed n = 288 split. Bold marks the largest gap per architecture.

<table><tr><td>Architecture</td><td>Confidence</td><td>Output</td></tr><tr><td>ResNet-50</td><td>+17.4 pts</td><td>+30.3 pts</td></tr><tr><td>VGG-16-BN</td><td>+18.8 pts</td><td>+20.6 pts</td></tr><tr><td>Wide ResNet-50</td><td>+15.4 pts</td><td>+27.9 pts</td></tr><tr><td>ViT-B/16</td><td>+17.6 pts</td><td>+31.0 pts</td></tr></table>

## D Per-Seed Robustness: Full Detail

Table 5 reports per-seed threshold sensitivity underlying Section 4.2, for output disagreement, threshold 0.8. All four architectures produce reorganized cases in every seed; VGG-16-BN’s smallest seed (1207, n = 4, all misclassified) and ViT-B/16’s smallest seed (1207, n = 4) both reflect a lower baseline disagreement rate. Wide ResNet-50’s per-seed accuracy-on-reorganized varies considerably (36.1%–54.7%), consistent with small per-seed sample sizes (40–86 reorganized cases) rather than a genuine inconsistency.

Table 5: Per-seed threshold sensitivity, output disagreement, threshold = 0.8.
<table><tr><td>Architecture</td><td>Seed</td><td>n (reorg.)</td><td>Acc. reorg.</td><td>Acc. stable</td></tr><tr><td>ResNet-50</td><td>510</td><td>52</td><td>23.1%</td><td>71.1%</td></tr><tr><td>ResNet-50</td><td>804</td><td>40</td><td>40.0%</td><td>71.2%</td></tr><tr><td>ResNet-50</td><td>1207</td><td>31</td><td>29.0%</td><td>70.3%</td></tr><tr><td>VGG-16-BN</td><td>510</td><td>15</td><td>40.0%</td><td>72.4%</td></tr><tr><td>VGG-16-BN</td><td>804</td><td>37</td><td>27.0%</td><td>71.8%</td></tr><tr><td>VGG-16-BN</td><td>1207</td><td>4</td><td>0.0%</td><td>73.2%</td></tr><tr><td>Wide ResNet-50</td><td>510</td><td>64</td><td>54.7%</td><td>70.6%</td></tr><tr><td>Wide ResNet-50</td><td>804</td><td>61</td><td>36.1%</td><td>75.5%</td></tr><tr><td>Wide ResNet-50</td><td>1207</td><td>86</td><td>43.0%</td><td>75.5%</td></tr><tr><td>ViT-B/16</td><td>510</td><td>94</td><td>26.6%</td><td>73.8%</td></tr><tr><td>ViT-B/16</td><td>804</td><td>16</td><td>25.0%</td><td>65.5%</td></tr><tr><td>ViT-B/16</td><td>1207</td><td>4</td><td>25.0%</td><td>70.3%</td></tr></table>