# Complementary Roles of Radiomics and Foundation Representations in Renal Cell Carcinoma Classification: A Comparative Study of 2D and 3D CT Encodings

Yuan Liang<sup>1,3</sup>, Sourav Bhattacharjee<sup>2</sup>, and Abraham Campbell<sup>3</sup>

<sup>1</sup> Research Ireland Centre for Research Training in Machine Learning 2 School of Veterinary Medicine, University College Dublin, Dublin, Ireland 3 School of Computer Science, University College Dublin, Dublin, Ireland abey.campbell@ucd.ie

Abstract. Accurate preoperative subtype classification of renal cell carcinoma (RCC) from contrast-enhanced computed tomography remains clinically challenging. Radiomics provides structured tumour descriptors, whereas foundation representations ofer transferable image features. However, it remains unclear whether radiomics still adds value beyond pretrained representations, and how 2D and 3D MedVAE encoders compare in this setting.

We compared handcrafted radiomics, 2D MedVAE, 3D MedVAE, and their fusion for binary clear-cell RCC versus non-clear-cell RCC classification on KiTS23 under a unified preprocessing pipeline. Concatenation, cross-attention, and gated fusion were evaluated as representative integration strategies, and radiomics feature importance was analysed to support decision-centric interpretability.

Fusion consistently improved discrimination over image-only MedVAE branches. The best overall performance was achieved by 3D gated fusion, with an AUC of 82.7%, outperforming the best 2D fusion model (79.6%), the radiomics baseline (74.4%), and the single-modality Med-VAE branches. Ablation analysis further showed clear gains of the full fusion model over both image-only and radiomics-only variants, indicating complementary contributions from radiomics and image representations. These findings suggest that radiomics remains relevant for RCC CT classification in the presence of foundation representations, and that its integration with MedVAE is more efective in the 3D setting. More broadly, the study supports a complementary role for radiomics and foundation representations in clinically meaningful imaging decision support.

Keywords: Renal cell carcinoma · Computed tomography · Radiomics · foundation representations · Feature fusion · Interpretability

## 1 Introduction

Kidney cancer remains a substantial global health burden, with 434,840 new cases and 155,953 deaths reported worldwide in 2022 [1]. Accurate preoperative characterisation of renal masses, including discrimination between clear cell

RCC (ccRCC) and non-clear cell RCC, is clinically important because histologic subtype is associated with prognosis and treatment decisions. Multiphasic contrast-enhanced CT is central to diagnosis and staging and is recommended by the European Association of Urology guideline for RCC [2]. However, CT interpretation remains limited by inter-reader variability and the dificulty of quantifying subtle intratumoral heterogeneity, motivating computational approaches that can provide reproducible and quantitative decision support.

Radiomics ofers a principled way to quantify tumour phenotype from routine imaging. Early work established radiomics as high-throughput extraction of engineered intensity, shape, and texture descriptors and demonstrated associations with cancer phenotype and outcomes [3]. Subsequent reviews outlined a standardised pipeline spanning image acquisition, segmentation, preprocessing, feature extraction, model development, and validation, while highlighting challenges such as protocol variability and overfitting [4,5]. Reproducibility has been further supported by the Image Biomarker Standardisation Initiative and tools such as PyRadiomics [6,7]. In RCC, CT radiomics has shown promise for distinguishing ccRCC from non-clear cell RCC, supporting its value as a structured and interpretable descriptor of renal tumour phenotype [8].

Deep learning has also become dominant across medical image analysis tasks [9], and end-to-end models have been explored for diferentiating renal tumour subtypes on multi-phase CT [10]. Yet RCC studies often involve limited labelled cohorts and heterogeneous acquisition protocols, making robust representation learning dificult and increasing interest in pretrained representations that can transfer useful visual structure beyond task-specific datasets.

Within this broader shift, foundation models emphasise large-scale pretraining and downstream adaptability [11]. In general vision, Vision Transformers, masked autoencoders, and vision-language contrastive models have shown strong transferability at scale [12–14]. In medical imaging, resources and models such as RadImageNet, Models Genesis, and MedicalNet or Med3D seek to address label scarcity while preserving domain-specific and volumetric structure [15–17]. More recent developments, including SAM, MedSAM, MedCLIP, and Merlin, further illustrate the rapid growth of large-scale pretrained medical representations [18–21]. In this landscape, MedVAE is particularly relevant for RCC CT because it provides large-scale pretrained 2D and 3D variational autoencoders for medical image encoding [22]. Its emphasis on compact and generalisable latent representations makes it a plausible source of foundation representations for downstream classification when full-resolution learning is costly or data are limited.

Despite these advances, three gaps remain for CT-based RCC classification. First, it is unclear whether foundation representations alone are suficiently discriminative in typical RCC cohort sizes, or whether radiomics still provides complementary information [3, 4, 22]. Second, although RCC CT is inherently volumetric, many pipelines still operate on 2D slices, and the relative advantages of 2D slice-based versus 3D volumetric encoders, as well as their interaction with radiomics, remain insuficiently characterised [16,17]. Third, although fusion may improve performance, it remains underexplored how to obtain clinically meaningful interpretability that reflects the internal decision pathway rather than relying only on post hoc visual explanations.

To examine radiomics–foundation interaction, we evaluate fusion as a comparative analytical tool. Specifically, we consider feature concatenation as a transparent baseline, cross-attention as a mechanism for modelling conditional dependence, and gated fusion as an established dynamic weighting strategy inspired by prior work on gated feature integration [23–25].

Interpretability is a further prerequisite for clinical adoption of medical imaging AI. Many explainability techniques are post hoc, including saliency maps [26], Grad-CAM [27], Integrated Gradients [28], LIME [29], and SHAP [30]. However, such explanations can remain visually plausible while being only weakly tied to learned parameters and may not faithfully reflect model reasoning [31]. In healthcare, multiple critiques and surveys have argued that current explainability methods can be misaligned with clinical needs and may ofer limited patient-level decision support [32–34]. In high-stakes settings, analysing internal decision processes, or using inherently interpretable approaches, has therefore been argued to be preferable to explaining black boxes after the fact [35, 36]. In feature fusion settings, internal contribution analysis ofers a practical route to decision-centric interpretability by quantifying how structured radiomics descriptors contribute within the fusion mechanism [6, 24, 25].

In this work, we make the following contributions:

We present a CT-based RCC classification framework that combines MedVAEderived 2D or 3D representations with handcrafted radiomics features under consistent preprocessing and evaluation.

– We compare 2D slice-based and 3D volumetric MedVAE encoders to assess how representation dimensionality afects RCC subtype discrimination.   
– We evaluate representative fusion mechanisms, including concatenation, crossattention, and gated fusion, to quantify radiomics–foundation interaction gains and stability.

– We assess the added value of radiomics beyond foundation representations and analyse radiomics feature importance within the fusion model to provide decision-centric interpretability.

## 2 Methods

Methods overview. Figure 1 summarises the overall study framework. We compare handcrafted CT radiomics with MedVAE-derived foundation representations for RCC subtype classification on KiTS23. Under a unified preprocessing protocol, we extract (i) a structured radiomics vector from a tumour ROI, and (ii) a MedVAE image embedding using either a 2D slice-based pathway or a 3D volumetric pathway. We then evaluate three fusion strategies, concatenation, cross-attention, and gated fusion, and perform fusion-based interpretability analyses by ranking radiomics feature importance within the model decision pathway.

![](images/5a9de0982f736aaa7ed6f58715e40d16efeb401efd57c144f80c1ec6bc4cb0fb.jpg)  
Fig. 1: Overview of the radiomics–MedVAE fusion framework. Tumour-centred CT inputs are processed through either a 2D MedVAE branch with top-k slice selection or a 3D MedVAE branch with volumetric encoding, while handcrafted radiomics features are extracted from the radiomics ROI and reduced through feature selection. The resulting image and radiomics representations are then combined using concatenation, cross-attention, or gated fusion for RCC subtype classification.

## 2.1 Dataset and preprocessing

We use the KiTS23 cohort, a publicly released challenge dataset of contrastenhanced preoperative CT scans with semantic segmentations of kidney, tumour, and cyst structures [37,38]. KiTS23 has a heterogeneous contrast setting, with cases acquired in either corticomedullary or nephrogenic phase [37]. After excluding multifocal cases to avoid label ambiguity and intra-patient lesion heterogeneity, the final cohort used in this study comprised 396 cases.

All CT volumes and segmentation masks are reoriented to a common RAS axis convention and resampled to an isotropic voxel spacing of 1.0 mm using bilinear interpolation for images and nearest-neighbour interpolation for masks. CT intensities are clipped to a fixed Hounsfield Unit range of [-150, 200] prior to downstream processing.

For each case, we compute a tight tumour bounding box from the binary tumour mask. Based on this box, we generate two ROIs: (i) a radiomics ROI using the tumour box without additional margin, and (ii) a deep-learning ROI obtained by expanding the tumour box with a fixed 12-voxel margin to retain limited surrounding context. When the expanded ROI extends beyond the image boundary, padding is handled during subsequent patch construction.

Using the preprocessed case-level image volume and binary tumour mask, we generate a zero-margin segmentation crop for radiomics and an expanded crop for deep learning. If multiple tumour annotations are available for a case, we use a consensus tumour mask derived using Simultaneous Truth and Performance Level Estimation (STAPLE) together with the corresponding preprocessed caselevel imaging volume to ensure geometric consistency [39].

## 2.2 Radiomics feature extraction and normalisation

As illustrated in Fig. 1, radiomics descriptors are extracted from the radiomics ROI, which is derived from the MONAI-preprocessed CT volumes and masks described above, using PyRadiomics [7, 40]. Feature computation follows IBSIaligned settings to improve reproducibility [6]. We compute (i) shape features from the Original image type only, and (ii) first-order and texture features from both the Original image and Laplacian-of-Gaussian filtered images with $\sigma \in$ {1, 2, 3}. This yields an initial radiomics feature vector $x _ { \mathrm { R A D } } \in \mathbb { R } ^ { 3 8 6 }$

Intensity discretisation is performed using a fixed bin width of 25 HU during radiomics extraction. To improve robustness, we adopt a multi-stage feature selection strategy. First, we retain only features with inter-observer reliability $\mathrm { I C C } \geq 0 . 7 5$ . Second, we remove the lowest 20% of features ranked by median absolute deviation. Third, highly correlated feature pairs are pruned using a Pearson correlation threshold of $| \rho | > 0 . 9 5$ . Finally, hierarchical clustering based on correlation distance is applied with a distance threshold of 0.2, and representative features are retained from each cluster. After this procedure, the selected radiomics vector is denoted by $\tilde { r } ~ \in ~ \mathbb { R } ^ { 7 3 }$ . For fusion-based deep models, r˜ is z-score standardised using training-set statistics before projection and integration with MedVAE representations. By contrast, the standalone radiomics-based machine-learning baseline uses the selected radiomics features in their extracted form.

## 2.3 MedVAE foundation representations

As illustrated in Fig. 1, we use pretrained MedVAE autoencoders as compact representation extractors for both the 2D and 3D image branches [22]. We compared frozen, partially fine-tuned, and fully fine-tuned encoder settings, and found partial fine-tuning to perform best. Unless otherwise stated, all reported MedVAE results therefore use the partially fine-tuned configuration, with selected pretrained modules updated jointly with the projection, fusion, and classification components on KiTS23.

Given an input x, the VAE encoder maps the image to a latent representation [41]. In practice, we use the latent tensor returned by the MedVAE forward pass. When the returned object provides a distribution-like mean, we use that mean as the image representation. The resulting latent tensor is then flattened and projected into a 512-dimensional shared embedding space.

2D MedVAE pathway: slice selection and pooling For the 2D branch shown in Fig. 1, we use the deep-learning ROI after padding and centre-cropping to a fixed spatial size of $1 2 8 \times 1 2 8 \times 1 2 8 .$ , yielding a single-case input tensor $P _ { 2 D } \in \mathbb { R } ^ { 1 \times 1 \bar { 2 } 8 \times 1 2 8 \times 1 2 8 }$ . Axial slices are then ranked according to tumour extent in the mask, measured as the number of tumour-positive pixels in each slice. We retain the top-k tumour-bearing slices with the largest tumour area and restore them to ascending axial order before encoding. Each selected slice is encoded by the pretrained 2D MedVAE encoder. The resulting slice-level latent tensor is flattened and projected to a 512-dimensional embedding,

$$
u _ { i } \in \mathbb { R } ^ { 5 1 2 } .\tag{1}
$$

Slice embeddings are then aggregated into a case-level image representation $f _ { \mathrm { i m g } } \in \mathbb { R } ^ { 5 1 2 }$

For the 2D branch, we adopt partial fine-tuning by freezing most pretrained MedVAE parameters and allowing only a subset of later pretrained modules to update during downstream training.

3D MedVAE pathway: tumour-centred volumetric patch For the 3D branch shown in Fig. 1, we use one tumour-centred volumetric crop per case. The expanded deep-learning ROI is padded and centre-cropped to a fixed size of $9 6 \times 9 6 \times 9 6$ , yielding an input tensor $P _ { 3 D } \in \mathbb { R } ^ { 1 \times 9 6 \times 9 6 \times 9 6 }$ . This patch is passed through the pretrained 3D MedVAE encoder, the returned latent tensor is flattened, and a learned linear projection maps it to a 512-dimensional image embedding $f _ { \mathrm { i m g } } \in \mathbb { R } ^ { 5 1 2 }$ . For the 3D branch, we adopt the same partial fine-tuning policy used in the 2D branch, freezing most pretrained MedVAE parameters and allowing only a subset of later pretrained modules to update during downstream training. The code also supports frozen and fully fine-tuned settings.

## 2.4 Fusion models

As illustrated in Fig. 1, the selected radiomics feature vector $\tilde { r } \in \mathbb { R } ^ { 7 3 }$ and the MedVAE image embedding $f _ { \mathrm { i m g } } \in \mathbb { R } ^ { 5 1 2 }$ are combined using three fusion strategies. Before fusion, radiomics features are projected into the same shared embedding space:

$$
h _ { \mathrm { r a d } } = W _ { \mathrm { r a d } } \tilde { r } + b _ { \mathrm { r a d } } , \quad W _ { \mathrm { r a d } } \in \mathbb { R } ^ { 5 1 2 \times 7 3 } .\tag{2}
$$

Concatenation Concatenation directly combines the image embedding and projected radiomics embedding:

$$
h _ { \mathrm { f u s e } } = [ f _ { \mathrm { i m g } } ; h _ { \mathrm { r a d } } ] ,\tag{3}
$$

which is then passed to the classifier head for prediction.

Cross-attention fusion The implemented cross-attention mechanism uses image token(s) as queries and radiomics tokens as keys and values. Radiomics features are first projected into a shared embedding and then transformed into a small set of learned radiomics tokens.

For the 2D branch, let $U \in \mathbb { R } ^ { k \times 5 1 2 }$ denote the slice-level image embeddings before pooling, and let $h _ { \mathrm { r a d } } \in \mathbb { R } ^ { 5 1 2 }$ denote the projected radiomics embedding. We form image queries and radiomics tokens as

$$
Q = U W _ { Q } ,\tag{4}
$$

$$
R = { \mathrm { r e s h a p e } } ( h _ { \mathrm { r a d } } W _ { R } ) ,\tag{5}
$$

where $Q \in \mathbb { R } ^ { k \times d _ { c } }$ and $R \in \mathbb { R } ^ { T \times d _ { c } }$ . Cross-attention is then computed as

$$
\mathrm { A t t n } ( Q , R , R ) = \mathrm { s o f t m a x } \left( \frac { Q R ^ { \top } } { \sqrt { d _ { c } } } \right) R .\tag{6}
$$

Residual connections, layer normalisation, and a feed-forward block are applied, after which the attended slice tokens are mean-pooled and concatenated with the pooled image embedding for classification.

For the 3D branch, the same principle is used, but the image side consists of a single pooled volumetric query token rather than multiple slice tokens. Let $q _ { \mathrm { i m g } } \in \mathbb { R } ^ { 1 \times d _ { c } }$ denote the projected volumetric query token and let $R \in \mathbb { R } ^ { T \times d _ { c } }$ denote the radiomics tokens. Cross-attention is applied in the same form:

$$
\mathrm { A t t n } ( q _ { \mathrm { i m g } } , R , R ) = \mathrm { s o f t m a x } \left( \frac { q _ { \mathrm { i m g } } R ^ { \top } } { \sqrt { d _ { c } } } \right) R .\tag{7}
$$

The attended output is then fused with the image embedding for final classification.

Gated fusion We implement channel-wise dynamic weighting inspired by prior work on gated feature integration [24]. In the implemented formulation, the gate is generated from the radiomics projection alone:

$$
g = \sigma ( \mathrm { M L P } ( h _ { \mathrm { r a d } } ) ) , \quad g \in ( 0 , 1 ) ^ { 5 1 2 } ,\tag{8}
$$

where a multilayer perceptron (MLP) uses a hidden layer of dimension 256 with ReLU activation, followed by a sigmoid output layer. The fused embedding is then

$$
h _ { \mathrm { f u s e } } = g \odot f _ { \mathrm { i m g } } + ( 1 - g ) \odot h _ { \mathrm { r a d } } .\tag{9}
$$

This formulation yields an explicit channel-wise balance between image-derived and radiomics-derived evidence.

## 2.5 Fusion-based interpretability analysis

Post hoc saliency methods can be visually appealing but may be weakly coupled to the actual decision parameters of the trained model and can therefore be misleading in high-stakes settings [31–35]. We therefore focus on decision-centric interpretability by analysing radiomics contributions inside the fusion model.

To quantify feature contribution, we compute permutation feature importance by randomly permuting one radiomics feature across cases in the evaluation split and measuring the resulting decrease in AUC. In the provided implementation, the permutation procedure is repeated R = 5 times per feature, and the mean AUC drop is used for ranking.

## 3 Experiments and Results

## 3.1 Experimental setup

We optimise weighted cross-entropy loss for binary classification, where class weights are computed from the training split to mitigate label imbalance.

Both 2D and 3D implementations use Adam. In the 2D branch, all trainable parameters use a learning rate of $1 \times 1 0 ^ { - 4 }$ . In the 3D branch, the partially fine-tuned MedVAE backbone uses a learning rate of $1 \times 1 0 ^ { - 5 }$ , whereas newly added heads use $1 \times 1 0 ^ { - 4 }$ , with weight decay $1 \times 1 0 ^ { - 5 }$ . Training uses a ReduceLROnPlateau scheduler with factor 0.5 and patience 5, and early stopping based on validation AUC with patience 15.

Augmentations are applied only to the deep-learning branch. Training-time augmentation includes random flips, random 90-degree rotations, and small afine perturbations, whereas validation and test inputs are processed deterministically. We perform stratified case-level splitting to avoid information leakage, using fixed proportions of 60% for training, 25% for validation, and 15% for testing.

The primary metric is the area under the receiver operating characteristic curve (AUC). We additionally report accuracy, F1-score, precision, recall (sensitivity), and specificity. Model selection is based on validation AUC. For threshold-dependent metrics, the final operating point is selected on the validation set using Youden’s J statistic and then applied unchanged to the held-out test set [42].

## 3.2 Overall performance comparison

Table 1 summarises the performance of radiomics, single-modality MedVAE, and fusion-based models for binary ccRCC versus non-ccRCC classification. Among all evaluated models, the 3D gated fusion model achieved the highest AUC of 82.7%, indicating that joint modelling of volumetric MedVAE representations and handcrafted radiomics provided the strongest overall discrimination. The best 2D fusion model, also based on gated fusion, reached an AUC of 79.6%, outperforming both the 2D-only model and the radiomics baseline.

Table 1: Performance comparison of radiomics, single-modality MedVAE, and fusion models for binary ccRCC versus non-ccRCC classification. The best result in each column is shown in bold. All metrics are reported in %.
<table><tr><td>Category</td><td>Model</td><td>AUC Accuracy</td><td></td><td>F1</td><td>Precision Recall Specificity</td><td></td><td></td></tr><tr><td>Radiomics</td><td>ML baseline</td><td>74.4</td><td>62.0</td><td>65.6</td><td>88.3</td><td>52.6</td><td>83.8</td></tr><tr><td>Deep learning</td><td>2D MedVAE 3D MedVAE</td><td>59.5 63.2</td><td>63.3 71.7</td><td>75.0 82.5</td><td>71.7 72.7</td><td>78.6 95.2</td><td>27.8 16.7</td></tr><tr><td rowspan="3">2D fusion</td><td>Concatenation</td><td>74.7</td><td>66.7</td><td>73.7</td><td>82.4</td><td>66.7</td><td>66.7</td></tr><tr><td>Cross-attention</td><td>77.1</td><td>73.3</td><td>81.8</td><td>78.3</td><td>85.7</td><td>44.4</td></tr><tr><td>Gated fusion</td><td>79.6</td><td>76.7</td><td>84.4</td><td>79.2</td><td>90.5</td><td>44.4</td></tr><tr><td rowspan="3">3D fusion</td><td>Concatenation</td><td>73.8</td><td>61.7</td><td>67.6</td><td>82.8</td><td>57.1</td><td>72.2</td></tr><tr><td>Cross-attention</td><td>79.5</td><td>71.7</td><td>79.5</td><td>80.5</td><td>78.6</td><td>55.6</td></tr><tr><td>Gated fusion</td><td>82.7</td><td>71.7</td><td>77.9</td><td>85.7</td><td>71.4</td><td>72.2</td></tr></table>

Fusion improved performance consistently over single-modality image models. In the 2D setting, AUC increased from 59.5% for the image-only branch to 74.7%, 77.1%, and 79.6% for concatenation, cross-attention, and gated fusion, respectively. A similar pattern was observed for the 3D branch, where the image-only model achieved an AUC of 63.2%, while the corresponding fusion variants reached 73.8%, 79.5%, and 82.7%. These results suggest that radiomics provided complementary information that was not fully captured by MedVAE image representations alone.

Comparing fusion strategies, gated fusion yielded the strongest AUC in both the 2D and 3D settings. This trend suggests that adaptive channel-wise weighting ofered a more efective integration mechanism than either simple concatenation or the current cross-attention formulation. Notably, the 3D gated fusion model also achieved competitive precision (85.7%) and specificity (72.2%), while maintaining balanced recall (71.4%), indicating that its gain in AUC was not achieved through a highly skewed operating point.

The radiomics baseline remained competitive relative to single-modality image branches. The ML baseline achieved an AUC of 74.4%, exceeding both the 2D-only and 3D-only MedVAE branches. This finding is consistent with the view that handcrafted radiomics remains a strong and stable descriptor family in relatively small RCC cohorts with heterogeneous acquisition conditions. However, the best fusion models outperformed the radiomics baseline, indicating that radiomics and MedVAE features were complementary rather than redundant.

## 3.3 Ablation analysis

To further evaluate modality complementarity, we analysed the best-performing configuration using single-modality ablations. As shown in Table 2, the full 3D gated fusion model achieved an AUC of 82.7%, whereas the image-only and radiomics-only variants reached 60.4% and 52.9%, respectively. These values were obtained from the exported ablation results of the final model.

Table 2: Ablation analysis of the best-performing 3D gated fusion model. AUC is reported on the test set. The full fusion model is compared against image-only and radiomics-only variants using the same evaluation protocol.
<table><tr><td>Variant</td><td>Image Radiomics AUC (%)</td><td></td><td></td></tr><tr><td>Radiomics-only</td><td>X</td><td>√</td><td>52.9</td></tr><tr><td>Image-only</td><td>√</td><td>×</td><td>60.4</td></tr><tr><td>Full fusion</td><td>√</td><td>√</td><td>82.7</td></tr></table>

The large drop in AUC observed after removing either branch indicates that the performance of the final model cannot be explained by one modality alone. Instead, the gain from 60.4% or 52.9% to 82.7% supports the interpretation that the learned interaction between radiomics and image representations was itself beneficial. In this setting, the image-only ablation outperformed the radiomicsonly ablation, suggesting that the volumetric MedVAE branch provided the stronger standalone signal within the best 3D fusion configuration, while radiomics supplied additional structured information that substantially improved discrimination when fused.

## 3.4 Fusion-based interpretability results

![](images/d5f0f76bee239b28e4d46ada72ebd79ab1161aac384c0461b252cd6ea91bf4b7.jpg)  
Fig. 2: Top 10 radiomics features ranked by permutation importance for the bestperforming 3D gated fusion model. Features are ordered by the mean decrease in AUC after feature-wise permutation on the evaluation set. Error bars indicate the standard deviation across repeated permutations.

To examine how radiomics contributed within the fusion model, we analysed permutation-based feature importance for the best-performing 3D gated fusion configuration. Figure 2 presents the top 10 radiomics features ranked by the mean decrease in AUC after feature-wise permutation. The most influential feature was original\_gldm\_DependenceEntropy, while other highly ranked features came from GLCM, GLSZM, and first-order families, many of which were derived from Laplacian-of-Gaussian filtered images. These rankings were used to characterise which structured radiomics descriptors contributed most strongly within the fusion model.

Several qualitative patterns emerged from this ranking. First, many of the highest-ranked features belonged to texture families such as GLDM, GLCM, GLSZM, and GLRLM, suggesting that intratumoural heterogeneity remained a key discriminative factor in the fused model. Second, multiple top-ranked features were derived from Laplacian-of-Gaussian filtered images rather than from the original image alone, indicating that multi-scale filtered radiomics contributed useful information beyond simple first-order intensity summaries. Third, both texture and first-order statistics appeared among the most important features, suggesting that the final predictor relied on a combination of heterogeneity-related and distribution-related tumour descriptors.

This analysis supports the interpretability framing adopted in this study. Rather than relying exclusively on post hoc saliency visualisation, the fusion model permits direct examination of how structured radiomics descriptors participate in prediction. In this sense, feature-importance ranking provides a decisioncentric view of the internal fusion mechanism, highlighting which radiomics characteristics most strongly influenced the final classifier output.

## 4 Discussion

This study shows that radiomics remains valuable for RCC subtype classification even when MedVAE-derived foundation representations are available. Although the detailed performance comparisons are reported in the Results section, the overall pattern was consistent: integrating radiomics with pretrained image representations was more efective than relying on either branch alone. This supports the view that radiomics and foundation representations provide complementary rather than competing information in limited-data renal CT analysis.

The stronger performance of the 3D models further suggests that preserving volumetric structure is important for this task. RCC phenotype is inherently three-dimensional, and subtype-related heterogeneity is distributed across the tumour volume rather than confined to a small number of axial slices. In this setting, the 3D branch appears better aligned with the underlying anatomy, whereas the 2D branch necessarily compresses volumetric information through slice selection and pooling.

A particularly important aspect of this study is interpretability. Rather than relying only on post hoc saliency visualisation, we examined radiomics contribution within the fusion model itself. Because radiomics features correspond to defined intensity and texture descriptors, their ranking provides a more structured and clinically meaningful view of model behaviour [31–35]. In practice, this helps connect model predictions to tumour characteristics that are more interpretable than abstract latent features alone. Such a decision-centric perspective may be particularly valuable in preoperative decision support, where clinicians need not only a prediction, but also some indication of which tumour properties are influencing it.

This interpretability advantage may also be relevant beyond RCC, as similar fusion designs could support other medical imaging tasks that require both strong discrimination and clinically communicable model behaviour.

Overall, these findings support a complementary role for radiomics in the foundation-model era. For RCC CT classification, the best results were obtained when MedVAE-derived image features were fused with radiomics, especially in the 3D setting. More broadly, fusion with structured radiomics may ofer a practical route toward clinically useful medical AI that balances predictive performance with more interpretable decision support.

## 5 Conclusion

In this study, we investigated the interaction between handcrafted radiomics and MedVAE-derived foundation representations for RCC subtype classification from contrast-enhanced CT. The results show that radiomics remains clinically relevant even in the presence of pretrained image representations, and that the strongest performance is achieved when the two are fused rather than used in isolation. The best overall model was obtained in the 3D setting, supporting the value of preserving volumetric tumour structure for RCC characterisation.

Beyond predictive performance, the proposed framework also provides a more structured route to interpretability by identifying which radiomics descriptors contribute within the fusion model. This is clinically important because preoperative subtype classification requires not only accurate discrimination, but also outputs that can be related to recognisable tumour characteristics. More broadly, the same fusion paradigm may be transferable to other medical imaging tasks in which pretrained image representations can be complemented by structured radiomics to improve both performance and interpretability.

## References

1. International Agency for Research on Cancer. Kidney fact sheet. https://gco. iarc.who.int/media/globocan/factsheets/cancers/29-kidney-fact-sheet. pdf, 2024. Global Cancer Observatory, GLOBOCAN 2022, version 1.1; accessed 13 April 2026.

2. B Ljungberg, Laurence Albiges, J Bedke, A Bex, U Capitanio, RH Giles, M Hora, T Klatte, L Marconi, T Powles, et al. Eau guidelines on renal cell carcinoma. European Association of Urology, pages 1–100, 2023.

3. Hugo JWL Aerts, Emmanuel Rios Velazquez, Ralph TH Leijenaar, Chintan Parmar, Patrick Grossmann, Sara Carvalho, Johan Bussink, René Monshouwer, Benjamin Haibe-Kains, Derek Rietveld, et al. Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nature communications, 5(1):4006, 2014.

4. Robert J Gillies, Paul E Kinahan, and Hedvig Hricak. Radiomics: images are more than pictures, they are data. Radiology, 278(2):563–577, 2016.

5. Philippe Lambin, Ralph TH Leijenaar, Timo M Deist, Jurgen Peerlings, Evelyn EC De Jong, Janita Van Timmeren, Sebastian Sanduleanu, Ruben THM Larue, Aniek JG Even, Arthur Jochems, et al. Radiomics: the bridge between medical imaging and personalized medicine. Nature reviews Clinical oncology, 14(12):749– 762, 2017.

6. Alex Zwanenburg, Martin Vallières, Mahmoud A Abdalah, Hugo JWL Aerts, Vincent Andrearczyk, Aditya Apte, Saeed Ashrafinia, Spyridon Bakas, Roelof J Beukinga, Ronald Boellaard, et al. The image biomarker standardization initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology, 295(2):328–338, 2020.

7. Joost JM Van Griethuysen, Andriy Fedorov, Chintan Parmar, Ahmed Hosny, Nicole Aucoin, Vivek Narayan, Regina GH Beets-Tan, Jean-Christophe Fillion-Robin, Steve Pieper, and Hugo JWL Aerts. Computational radiomics system to decode the radiographic phenotype. Cancer research, 77(21):e104–e107, 2017.

8. Ping Wang, Xu Pei, Xiao-Ping Yin, Jia-Liang Ren, Yun Wang, Lu-Yao Ma, Xiao-Guang Du, and Bu-Lang Gao. Radiomics models based on enhanced computed tomography to distinguish clear cell from non-clear cell renal cell carcinomas. Scientific Reports, 11(1):13729, 2021.

9. Geert Litjens, Thijs Kooi, Babak Ehteshami Bejnordi, Arnaud Arindra Adiyoso Setio, Francesco Ciompi, Mohsen Ghafoorian, Jeroen Awm Van Der Laak, Bram Van Ginneken, and Clara I Sánchez. A survey on deep learning in medical image analysis. Medical image analysis, 42:60–88, 2017.

10. Kwang-Hyun Uhm, Seung-Won Jung, Moon Hyung Choi, Hong-Kyu Shin, Jae-Ik Yoo, Se Won Oh, Jee Young Kim, Hyun Gi Kim, Young Joon Lee, Seo Yeon Youn, et al. Deep learning for end-to-end kidney cancer diagnosis on multi-phase abdominal computed tomography. NPJ precision oncology, 5(1):54, 2021.

11. Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

12. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

13. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

14. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

15. Xueyan Mei, Zelong Liu, Philip M Robson, Brett Marinelli, Mingqian Huang, Amish Doshi, Adam Jacobi, Chendi Cao, Katherine E Link, Thomas Yang, et al.

Radimagenet: an open radiologic deep learning research dataset for efective transfer learning. Radiology: Artificial Intelligence, 4(5):e210315, 2022.

16. Zongwei Zhou, Vatsal Sodha, Md Mahfuzur Rahman Siddiquee, Ruibin Feng, Nima Tajbakhsh, Michael B Gotway, and Jianming Liang. Models genesis: Generic autodidactic models for 3d medical image analysis. In International conference on medical image computing and computer-assisted intervention, pages 384–393. Springer, 2019.

17. Sihong Chen, Kai Ma, and Yefeng Zheng. Med3d: Transfer learning for 3d medical image analysis. arXiv preprint arXiv:1904.00625, 2019.

18. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

19. Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature communications, 15(1):654, 2024.

20. Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3876–3887, 2022.

21. Louis Blankemeier, Ashwin Kumar, Joseph Paul Cohen, Jiaming Liu, Longchao Liu, Dave Van Veen, Syed Jamal Safdar Gardezi, Hongkun Yu, Magdalini Paschali, Zhihong Chen, et al. Merlin: a computed tomography vision–language foundation model and dataset. Nature, pages 1–11, 2026.

22. Maya Varma, Ashwin Kumar, Rogier Sluijs, Sophie Ostmeier, Louis Blankemeier, Pierre Chambon, Christian Blüthgen, Jip Prince, Curtis Langlotz, and Akshay Chaudhari. Medvae: Eficient automated interpretation of medical images with large-scale generalizable autoencoders, 02 2025.

23. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

24. John Arevalo, Thamar Solorio, Manuel Montes-y Gómez, and Fabio A González. Gated multimodal units for information fusion. arXiv preprint arXiv:1702.01992, 2017.

25. Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: A survey and taxonomy. IEEE transactions on pattern analysis and machine intelligence, 41(2):423–443, 2018.

26. Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

27. Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 618–626, 2017.

28. Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pages 3319–3328, 2017.

29. Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pages 1135–1144, 2016.

30. Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30, 2017.

31. Julius Adebayo, Justin Gilmer, Michael Muelly, Ian J. Goodfellow, Moritz Hardt, and Been Kim. Sanity checks for saliency maps. In Advances in Neural Information Processing Systems (NeurIPS), pages 9525–9536, 2018.

32. Marzyeh Ghassemi, Luke Oakden-Rayner, and Andrew L. Beam. The false hope of current approaches to explainable artificial intelligence in health care. The Lancet Digital Health, 3(11):e745–e750, 2021.

33. Bas H. M. van der Velden, Hugo J. Kuijf, Kenneth G. A. Gilhuijs, and Max A. Viergever. Explainable artificial intelligence (xai) in deep learning-based medical image analysis. Medical Image Analysis, 79:102470, 2022.

34. Katarzyna Borys, Yasmin Alyssa Schmitt, Meike Nauta, Christin Seifert, Nicole Krämer, Christoph M. Friedrich, and Felix Nensa. Explainable ai in medical imaging: An overview for clinical practitioners – beyond saliency-based xai approaches. European Journal of Radiology, 162:110786, 2023.

35. Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1:206– 215, 2019.

36. Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017.

37. KiTS Challenge Organizers. Kits23: The 2023 kidney and kidney tumor segmentation challenge. https://kits-challenge.org/kits23/, 2023. Accessed: 2026- 04-14.

38. Nicholas Heller, Fabian Isensee, Resha Tejpaul, Andrew Wood, Nikolaos Papanikolopoulos, and Christopher Weight. 2023 kidney and kidney tumor segmentation challenge, 2023.

39. Simon K. Warfield, Kelly H. Zou, and William M. Wells. Simultaneous truth and performance level estimation (staple): an algorithm for the validation of image segmentation. IEEE Transactions on Medical Imaging, 23(7):903–921, 2004.

40. M. Jorge Cardoso, Wenqi Li, Richard Brown, et al. Monai: An open-source framework for deep learning in healthcare. IEEE Journal of Biomedical and Health Informatics, 26(9):4123–4135, 2022.

41. Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations (ICLR), 2014.

42. W. J. Youden. Index for rating diagnostic tests. Cancer, 3(1):32–35, 1950.