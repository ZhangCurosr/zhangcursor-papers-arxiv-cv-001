# CONTINUOUS 3D SPATIAL TRANSCRIPTOMICS RECONSTRUCTION VIA HISTOLOGY-INFORMED IMPLICIT NEURAL REPRESENTATIONS

Kaito Shiku<sup>1,2</sup> Ryoma Bise<sup>1</sup> Ahtisham Fazeel Abbasi<sup>2,3</sup> Yuichiro Iwashita<sup>2,3,4</sup> Shumpei Takezaki<sup>1,2</sup> Andreas Dengel<sup>2,3,4,5</sup> Muhammad Nabeel Asim<sup>2,4,5</sup>

<sup>1</sup> Kyushu University <sup>2</sup> DFKI GmbH <sup>3</sup> RPTU University Kaiserslautern-Landau <sup>4</sup> Osaka Metropolitan University <sup>5</sup> IntelligentX GmbH

## ABSTRACT

Three-dimensional spatial transcriptomics (3D ST) enables the analysis of gene-expression patterns across tissue depths. However, 3D ST measurements are acquired from discrete 2D tissue sections, which are often sparsely sampled due to high acquisition costs. To address this limitation, we propose a histology-informed multitask implicit neural representation (HIM-INR) for continuous 3D gene-expression reconstruction from sparse ST observations. HIM-INR jointly reconstructs gene-expression profiles and histological representations using an INR with a shared backbone, leveraging densely sampled histological images as auxiliary spatial supervision. Once trained, HIM-INR predicts gene expression at arbitrary 3D coordinates without requiring histological images. Experiments on two 3D ST datasets demonstrate improved reconstruction over existing methods, particularly as the number of missing ST sections increases.

Index Terms— 3D spatial transcriptomics, Implicit neural representatio (INR)

## 1. INTRODUCTION

Three-dimensional spatial transcriptomics (3D ST) [1] has emerged as a promising approach for investigating the spatial organization of gene expression within tissues. Building on conventional 2D ST [2], which measures gene expression in individual tissue sections, 3D ST enables the characterization of molecular interactions across tissue depths and the spatial extent and connectivity of pathological regions. Such capabilities offer new opportunities to investigate cellto-cell communication across sections, characterize tumor invasion and inflammatory infiltration, and reveal spatial relationships between tissue architecture and molecular states that cannot be fully captured by individual 2D sections [1, 3].

Despite the potential of 3D ST, gene expression is measured only on discrete tissue sections. Furthermore, the high cost of ST measurements restricts the number of sections that can be profiled, resulting in sparse observations along the tissue depth axis. Previous studies have aligned ST measurements from serial tissue sections into a common spatial coordinate system to reconstruct tissue structures along the depth axis [4, 5]. However, gene-expression information remains restricted to the observed sections, leaving the continuous distribution of gene expression between them unresolved.

These limitations motivate the reconstruction of a continuous 3D gene-expression field from sparsely and discretely observed ST sections. However, this task remains challenging. While gene expression can vary substantially with changes in tissue structure across sections, no direct information on gene expression is available between sparsely sampled ST sections. Consequently, not only conventional neighborhood-based interpolation methods but also deep learning-based approaches [6, 7] that model nonlinear relationships between spatial positions and gene expression face difficulties in accurately reconstructing gene expression in unobserved regions.

To address this challenge, our key idea is to leverage histological images, which provide detailed morphological information about tissue structure. Gene-expression patterns are known to be associated with tissue structure and morphology [8, 9]. Moreover, histological images are commonly acquired alongside ST measurements and are often available from more densely sampled tissue sections due to their substantially lower acquisition cost [10]. Thus, histological information can provide valuable cues about variations in tissue morphology, even at locations where ST measurements are unavailable, potentially facilitating continuous 3D gene-expression reconstruction.

In this study, we propose a histology-informed multi-task implicit neural representation (HIM-INR) framework for reconstructing gene-expression profiles at arbitrary 3D coordinates from sparsely sampled ST sections, as illustrated in Fig. 1. We base our method on an INR model, which represents continuous signals by learning a function that maps spatial coordinates to corresponding signal values [11]. Using the INR as a shared backbone, our framework jointly reconstructs gene-expression profiles and histological representations extracted using a pretrained pathology foundation model designed to capture tissue morphology [12]. By incorporating histological reconstruction as an auxiliary task, the shared INR backbone receives additional spatial supervision from densely sampled histological sections, even at locations where ST measurements are unavailable. This encourages the model to capture continuous variations in tissue morphology and learn shared spatial representations for gene-expression and histological reconstruction. Importantly, inference requires only 3D coordinates, enabling gene-expression reconstruction even at locations where neither ST measurements nor histological images are available.

We evaluate the proposed framework on two publicly available 3D ST datasets and compare it with conventional interpolation methods and INR-based reconstruction approaches. Experimental results demonstrate that our method outperforms existing approaches under most evaluation settings, highlighting the effectiveness of incorporating histological information for reconstructing gene expression profiles at unobserved sections. In particular, the benefits of histological supervision become more pronounced as the number of missing ST sections increases, highlighting its potential for reconstructing continuous gene expression fields from sparse observations.

![](images/b7b57f90843c6b25185a49d734889855a6844b9e8512059fd99055635411d41c.jpg)  
Fig. 1. Overview of the proposed histology-informed multi-task INR (HIM-INR). Gene-expression profiles and histology image features are compressed into modality-specific latent spaces. A shared INR backbone with modality-specific heads and decoders reconstructs both modalities from 3D coordinates, enabling dense image-only sections to provide auxiliary spatial supervision between sparse ST observations.

## 2. RELATED WORK

3D ST Reconstruction. Recent studies have explored 3D ST analysis by integrating transcriptomic measurements from serial tissue sections. Most approaches focus on aligning consecutive sections to reconstruct their spatial organization [4, 5]. Although these methods enable the analysis of spatial relationships across observed sections, they do not reconstruct gene-expression patterns in unobserved regions between sections, limiting continuous 3D analysis.

Recent methods have addressed gene-expression reconstruction between sparsely observed tissue sections. STAGE [6] aligns the latent space of a gene-expression autoencoder with spatial coordinates, enabling prediction at arbitrary locations by subsequently feeding only spatial coordinates into the decoder. SINTER3D [7] uses an INR [11] to predict gene expression at unobserved 3D coordinates. However, learning spatial continuity solely from transcriptomic measurements remains challenging when ST sections are sparsely sampled along the depth axis. In contrast, we explore the potential of leveraging more densely sampled histological images as auxiliary supervision to encourage the INR to capture morphological variations between observed ST sections.

INR for ST Analysis. INR parameterizes continuous signals using neural networks that map spatial coordinates to signal values [11]. Recent studies have applied INR to impute missing gene-expression values and reconstruct expression at unobserved locations in 2D ST [13, 14]. STINR [13] directly regresses geneexpression profiles from spatial coordinates, whereas SUICA [14] addresses high-dimensional expression reconstruction by learning a low-dimensional latent representation predicted using INR. However, these methods primarily focus on reconstruction within individual 2D tissue sections rather than continuous modeling of gene expression in 3D space across aligned sections.

Histology-based ST Estimation. Predicting gene expression from histological images has emerged as a major research direction in medical image analysis [8, 9]. These studies have demonstrated a strong relationship between tissue morphology and gene expression through image-based prediction. However, their reliance on histological images at inference time prevents prediction at locations where images are unavailable, limiting their applicability to our task.

## 3. HISTOLOGY-INFORMED MULTI-TASK INR

## 3.1. Overview

We propose HIM-INR to reconstruct gene expression at arbitrary 3D coordinates using sparsely observed ST sections and more densely sampled histological images. As illustrated in Fig. 1, HIM-INR consists of two reconstruction branches for gene expression and histology image features, sharing a SIREN-based INR backbone [11]. Inspired by Zhu et al. [14], we address the difficulty of modeling high-dimensional vectors with INR by learning spatial continuity in low-dimensional latent spaces for both modalities. The predicted latent representations are then decoded into gene-expression profiles and histology image features. During training, the model is jointly optimized using reconstruction losses for both modalities. Importantly, the histological-feature reconstruction branch is also trained on sections without ST measurements, encouraging the shared INR backbone to capture tissue structural information that guides geneexpression reconstruction in unobserved regions.

We consider sparsely observed ST measurements, denoted by ${ \cal S } _ { G } = \{ ( { \bf p } _ { i } , { \bf g } _ { i } ) \} _ { i \in \Omega _ { G } }$ , and more densely sampled histological observations, denoted by ${ \cal S } _ { I } = \{ ( { \bf p } _ { i } , { \bf v } _ { i } ) \} _ { i \in \Omega _ { I } }$ . Here, $\Omega _ { G }$ and $\Omega _ { I }$ denote the index sets of observed ST and histology image features, respectively, with $\Omega _ { G } \subseteq \Omega _ { I }$ . The spatial coordinate $\mathbf { p } _ { i } \in \mathbb { R } ^ { 3 }$ specifies the location of each observation, while $\mathbf { g } _ { i } \in \mathbb { R } ^ { G }$ and $\mathbf { v } _ { i } \in \hat { \mathbb { R } } ^ { D }$ represent gene-expression profiles and image-derived features, respectively, with G genes and D feature dimensions. To encourage the INR to capture tissue morphology rather than low-level pixel information, we reconstruct features extracted by a pretrained pathology foundation model [12] instead of raw histological images. All serial tissue sections are registered into a common 3D coordinate system prior to training. HIM-INR reconstructs gene-expression profiles and histology image features at arbitrary 3D coordinates:

$$
\begin{array} { r } { \big ( \hat { \mathbf { g } } _ { i } , \hat { \mathbf { v } } _ { i } \big ) = \mathrm { H I M - I N R } \big ( \mathbf { p } _ { i } \big ) . } \end{array}\tag{1}
$$

## 3.2. Latent Representations of ST and Histology Features

To enable the INR to learn spatial continuity in low-dimensional latent spaces rather than directly modeling high-dimensional observations, we employ autoencoders to obtain latent representations of both gene-expression profiles and histology image features. We independently pretrain MLP-based autoencoders for each modality using a mean squared error (MSE) reconstruction loss, with latent dimensions $d _ { G }$ and $d _ { I }$ , respectively. After pretraining, the encoders are frozen and used to extract latent representations $\mathbf { \bar { z } } _ { i } ^ { G } \in \mathbb { R } ^ { d _ { G } }$ and $\mathbf { z } _ { i } ^ { I } \in \mathbb { R } ^ { d _ { I } }$ from the corresponding observations. These representations serve as fixed targets for subsequent INR training.

Table 1. Quantitative comparison of gene-expression reconstruction at withheld sections. PCC is computed on the top 50 and 250 HVGs and averaged over three random seeds. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">3DST-RA [15]</td><td colspan="2">HemiBrainA [16]</td><td colspan="2">HemiBrainB [16]</td></tr><tr><td>HVGs50</td><td>HVGs250</td><td>HVGs50</td><td>HVGs250</td><td>HVGs50</td><td>HVGs250</td></tr><tr><td>NN</td><td>0.1779</td><td>0.1385</td><td>0.0966</td><td>0.1240</td><td>0.0958</td><td>0.0741</td></tr><tr><td>KNN</td><td>0.2609</td><td>0.2148</td><td>0.1449</td><td>0.1921</td><td>0.1288</td><td>0.1114</td></tr><tr><td>STAGE [6]</td><td>0.2743</td><td>0.2041</td><td>0.2344</td><td>0.3095</td><td>0.1398</td><td>0.1525</td></tr><tr><td>3D-STINR [13]</td><td>0.2202</td><td>0.2425</td><td>0.1395</td><td>0.2113</td><td>0.1703</td><td>0.1453</td></tr><tr><td>3D-SUICA [14]</td><td>0.3409</td><td>0.3044</td><td>0.1883</td><td>0.2664</td><td>0.1518</td><td>0.1513</td></tr><tr><td>SINTER3D [7]</td><td>0.1847</td><td>0.2288</td><td>0.1724</td><td>0.2326</td><td>0.1627</td><td>0.1399</td></tr><tr><td>Ours</td><td>0.3711</td><td>0.3145</td><td>0.2253</td><td>0.3101</td><td>0.1824</td><td>0.1685</td></tr></table>

## 3.3. Histology-informed Multi-task Learning

We jointly learn continuous representations of sparsely observed ST and densely sampled histological images through a shared INR backbone $f .$ By incorporating image-only sections during training, the shared backbone captures spatial variations in tissue structure even at locations without ST observations, thereby guiding gene-expression reconstruction.

Given a 3D coordinate p<sub>i</sub>, the shared backbone extracts a spatial representation, which is mapped to modality-specific latent spaces by two prediction heads, $h _ { G }$ and $h _ { I } { \mathrm { : } }$

$$
\hat { \mathbf { z } } _ { i } ^ { G } = h _ { G } ( f ( \mathbf { p } _ { i } ) ) , \quad \hat { \mathbf { z } } _ { i } ^ { I } = h _ { I } ( f ( \mathbf { p } _ { i } ) ) .\tag{2}
$$

The predicted latent representations are subsequently decoded into gene-expression profiles and histology image features using modality-specific decoders $D _ { G }$ and $D _ { I }$ :

$$
\hat { \mathbf { g } } _ { i } = D _ { G } ( \hat { \mathbf { z } } _ { i } ^ { G } ) , \quad \hat { \mathbf { v } } _ { i } = D _ { I } ( \hat { \mathbf { z } } _ { i } ^ { I } ) .\tag{3}
$$

To mitigate the impact of section-specific technical variations and batch effects on learning spatial continuity, we introduce sliceand gene-specific affine calibration. For an ST observation i belonging to section $s ( i )$ , the calibrated gene-expression profile is computed as $\tilde { \mathbf { g } } _ { i } = \mathbf { a } _ { s ( i ) } \odot \hat { \mathbf { g } } _ { i } + \mathbf { b } _ { s ( i ) } ,$ , where $\mathbf { a } _ { s ( i ) } , \mathbf { b } _ { s ( i ) } \in \mathbb { R } ^ { G }$ are learnable scale and bias vectors, respectively, and ⊙ denotes element-wise multiplication.

The entire model is trained by minimizing reconstruction losses in both latent and output spaces for each modality, defined as follows:

$$
\mathcal { L } _ { G } = \frac { 1 } { \vert \Omega _ { G } \vert } \sum _ { i \in \Omega _ { G } } \left[ \Vert \hat { \mathbf { z } } _ { i } ^ { G } - \mathbf { z } _ { i } ^ { G } \Vert _ { 2 } ^ { 2 } + \Vert \tilde { \mathbf { g } } _ { i } - \mathbf { g } _ { i } \Vert _ { 2 } ^ { 2 } \right] ,
$$

$$
\mathcal { L } _ { I } = \frac { 1 } { \vert \Omega _ { I } \vert } \sum _ { i \in \Omega _ { I } } \left[ \Vert \hat { \mathbf { z } } _ { i } ^ { I } - \mathbf { z } _ { i } ^ { I } \Vert _ { 2 } ^ { 2 } + \Vert \hat { \mathbf { v } } _ { i } - \mathbf { v } _ { i } \Vert _ { 2 } ^ { 2 } \right] .\tag{4}
$$

The overall objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { G } + \lambda \mathcal { L } _ { I } ,\tag{5}
$$

where λ balances the contributions of the two modalities. For imageonly locations, only the latent-space and output-space image reconstruction losses are applied. By updating the shared INR backbone, these losses encourage the learning of spatial continuity even in regions without ST observations.

During inference, we directly use the gene-expression predictions gˆ<sub>i</sub> without applying the slice-specific affine calibration, as it is introduced during training to mitigate the effects of technical variations and batch effects on learning spatial continuity.

## 4. EXPERIMENTS

Datasets and Experimental Settings. Although histological images are commonly acquired alongside ST measurements [17], large-scale, publicly available 3D ST datasets with paired histological images remain limited. We therefore select two datasets with relatively large numbers of sections, 3DST-RA [15] and Hemi-Brain [16]. To evaluate our method, we artificially withhold tissue sections to simulate missing observations.

3DST-RA contains paired ST measurements and histological images from serial synovial tissue sections of patients with rheumatoid arthritis. We select the patient with the largest number of sections, consisting of seven consecutive sections numbered from 1 to 7. ST measurements are retained for Sections 1, 4, and 7. Among the remaining four sections, three are used as image-only observations, while one is completely withheld for evaluation. We repeat this evaluation four times, each time selecting a different section to be completely withheld.

The HemiBrain dataset contains serial brain sections from three mice. We use two samples, HemiBrainA and HemiBrainB, containing 42 and 9 sections, respectively. We retain both ST measurements and histological images at every fourth section. Among the three intermediate sections, the first and third are used as image-only observations, while the second is withheld for evaluation.

Prior to training and evaluation, adjacent tissue sections are aligned using XFeat-based feature matching [18] and RANSACbased similarity transformations [19]. These transformations are sequentially composed to register all sections to a common coordinate system, with the middle section as the reference. Evaluation-section histological images are also used for registration to ensure accurate spatial correspondence between predicted and ground-truth geneexpression profiles, but are excluded from model training, inference, and performance evaluation.

Implementation Details and Evaluation Metrics. All experiments used PyTorch [20] with Adam [21]. Histology image features were extracted using the frozen UNI2-h [12]. The best model was selected based on validation performance, using 5% of ST spots randomly held out from each observed section. Hyperparameters were set to a learning rate of $1 0 ^ { - 3 }$ , up to 1,000 epochs, a batch size of 512, latent dimensions $d _ { G } = d _ { I } = 5$ , and a loss weight $\lambda = 1$

To evaluate reconstruction performance while minimizing slicespecific batch effects, we compute the Pearson correlation coefficient (PCC) [9] between predicted and ground-truth expression across spatial locations for each gene within each section, then average across genes and sections. Following standard ST prediction protocols [22], we evaluate the top 50 and 250 highly variable genes (HVGs) to mitigate measurement noise. We report mean performance over three random seeds.

Table 2. Ablation study on histological supervision (HIM). PCC is evaluated on image-unavailable locations using the top 50 and 250 HVGs.
<table><tr><td rowspan="2">Method</td><td rowspan="2">HIM</td><td colspan="2">3DST-RA [15]</td><td colspan="2">HemiBrainA [16]</td><td colspan="2">HemiBrainB [16]</td></tr><tr><td>HVGs50</td><td>HVGs250</td><td>HVGs50</td><td>HVGs250</td><td>HVGs50</td><td>HVGs250</td></tr><tr><td>Ours w/o HIM</td><td>x</td><td>0.3593</td><td>0.3054</td><td>0.2259</td><td>0.3052</td><td>0.1742</td><td>0.1645</td></tr><tr><td>Ours</td><td>√</td><td>0.3711</td><td>0.3145</td><td>0.2253</td><td>0.3101</td><td>0.1824</td><td>0.1685</td></tr></table>

![](images/8ca17e912afb4d1024828a8a28fa3210f93563117fffeb5ec32990b505bbe79e.jpg)  
Fig. 2. Effect of ST sparsity on HemiBrainA (top 50 HVGs). The number of missing ST sections between observed sections is varied from 3 to 7. Values above the bars indicate the PCC differences between Ours and Ours w/o HIM (Ours − Ours w/o HIM).

Comparisons. We compare our method with six baselines. 1) “NN” and 2) “KNN” interpolate gene expression from the nearest and k = 5 nearest observed spots, respectively. 3) “STAGE” [6] aligns autoencoder latent representations with spatial coordinates, enabling prediction at unobserved locations through its decoder. 4) “3D-STINR” [13] and 5) “3D-SUICA” [14] are INR-based methods originally developed for 2D ST and extended to 3D using spatially aligned coordinates across sections. 3D-STINR directly regresses gene expression from 3D coordinates, whereas 3D-SUICA first learns gene-expression embeddings using slice-wise graph autoencoders and then trains an INR to predict these embeddings from 3D coordinates, which are subsequently decoded into gene expression. 6) “SINTER3D” [7] predicts gene expression using an INR with additional intermediate-feature reconstruction losses to encourage spatial continuity.

Table 1 presents the quantitative comparison. Simple interpolation methods, NN and KNN, show poor performance. STAGE performs well on HemiBrainA but shows limited performance on other datasets, indicating a lack of robustness across datasets. This may be because explicitly aligning gene-expression latent representations with spatial coordinates restricts their ability to capture complex expression patterns. Similarly, 3D-STINR and SINTER3D show limited performance, potentially due to the difficulty of directly regressing high-dimensional expression vectors using INR. In contrast, 3D-SUICA achieves relatively consistent performance, suggesting that low-dimensional latent modeling facilitates INR training. Our method achieves the best performance on all metrics for 3DST-RA and HemiBrainB and on HVGs250 for HemiBrainA, demonstrating the effectiveness of histological supervision for continuous 3D reconstruction from sparse ST observations across datasets.

Ablation Study. To evaluate the effect of morphological supervision, we compare our method with “Ours w/o HIM”, which excludes image-based supervision. Both reconstruct gene expression in the same low-dimensional latent space, differing only in whether the shared INR backbone receives additional supervision from histological images. As shown in Table 2, our method achieves relatively consistent and competitive performance across datasets, suggesting that dense histological supervision helps the INR learn spatial continuity more reliably than gene expression alone.

Table 3. Comparison with oracle image-based prediction (UNI2- h+LP), which uses true histological images at the evaluation sections. PCC is evaluated on the top 250 HVGs.
<table><tr><td>Method</td><td>3DST-RA [15]</td><td>HemiBrainA [16]</td><td>HemiBrainB [16]</td></tr><tr><td>UNI2-h+LP</td><td>0.2726</td><td>0.2744</td><td>0.1646</td></tr><tr><td>Ours</td><td>0.3145</td><td>0.3101</td><td>0.1685</td></tr></table>

To further assess histological supervision, we vary the number of missing ST sections (3, 5, and 7) on HemiBrainA, the largest dataset. The middle missing section is held out for evaluation, while the others provide only histological images. As shown in Fig. 2, both methods perform comparably with three missing sections, whereas our method increasingly outperforms Ours w/o HIM as the number of missing sections increases. This suggests that histological supervision is particularly beneficial when sparse ST observations make learning spatial continuity from gene expression alone challenging.

Validity of Histology as Multi-task Supervision. To validate the use of densely sampled histological images as auxiliary supervision in multi-task learning, we consider an alternative that first interpolates histology image features at unobserved sections and then uses them as direct inputs to predict gene expression. To evaluate this alternative without errors from histological feature interpolation, we train a linear probe on frozen UNI2-h features from ST-observed sections (UNI2-h+LP) and evaluate it using ground-truth histological images at the evaluation sections. Importantly, these images are unavailable in the actual reconstruction setting and are never provided to our method. As shown in Table 3, our method outperforms the UNI2-h+LP baseline even when ground-truth histological images are provided at the evaluation sections. This suggests that histological supervision can be beneficial for learning continuous spatial representations beyond this simple image-based prediction strategy.

## 5. CONCLUSION

We proposed HIM-INR, a histology-informed multi-task INR for continuous 3D gene-expression reconstruction from sparse ST observations. By jointly learning gene-expression and histological representations through a shared INR backbone, our method leverages densely sampled histological images as auxiliary supervision to capture spatial continuity between observed ST sections. Experiments on two 3D ST datasets demonstrated improved reconstruction over existing methods, with increasing benefits as ST observations become sparser. Importantly, our method enables gene-expression prediction at arbitrary 3D coordinates without requiring histological images at inference time.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study involved secondary analysis of publicly available human and mouse spatial transcriptomics datasets (3DST-RA and Hemi-Brain). The original studies reported approval from the relevant ethics committees, and written informed consent was obtained from the human participants [15, 16]. No new human or animal experiments were conducted in this study.

## 7. ACKNOWLEDGMENTS

This work was supported by JST BOOST, Japan Grant Number JPMJBS2406, ASPIRE Grant Number JPMJAP2403. A. Dengel and M. N. Asim are affiliated with IntelligentX GmbH. IntelligentX GmbH did not provide financial support for this work. The authors have no other relevant financial or non-financial interests to disclose. Anthropic Claude Code assisted with code development, and OpenAI ChatGPT assisted with language polishing and revision throughout the manuscript. The authors take full responsibility for the code and the final manuscript.

## 8. REFERENCES

[1] Chia-Kuei Mo et al., “Tumour Evolution and Microenvironment Interactions in 2D and 3D Space,” Nature, vol. 634, no. 8036, pp. 1178–1186, 2024.

[2] Patrik L Stahl et al., “Visualization and Analysis of Gene Ex-˚ pression in Tissue Sections by Spatial Transcriptomics,” Science, vol. 353, no. 6294, pp. 78–82, 2016.

[3] Tancredi Massimo Pentimalli et al., “Combining Spatial Transcriptomics and ECM Imaging in 3D for Mapping Cellular Interactions in The Tumor Microenvironment,” Cell Syst., vol. 16, no. 5, 2025.

[4] Ron Zeira et al., “Alignment and Integration of Spatial Transcriptomics Data,” Nat. Methods, vol. 19, no. 5, pp. 567–575, 2022.

[5] Xiang Zhou et al., “Integrating Spatial Transcriptomics Data Across Different Conditions, Technologies and Developmental Stages,” Nat. Comput. Sci., vol. 3, no. 10, pp. 894–906, 2023.

[6] Shang Li et al., “High-density Generation of Spatial Transcriptomics with STAGE,” Nucleic Acids Res., vol. 52, no. 9, pp. 4843–4856, 2024.

[7] Tianjiao Zhang et al., “SINTER3D: Continuous 3D Reconstruction of Spatial Transcriptomics via Implicit Neural Representations,” Genome Biol., vol. 27, no. 1, pp. 274, 2026.

[8] Bryan He et al., “Integrating Spatial Gene Expression and Breast Tumour Morphology via Deep Learning,” Nat. Biomed. Eng., vol. 4, no. 8, pp. 827–834, 2020.

[9] Youngmin Chung et al., “Accurate Spatial Gene Expression Prediction by Integrating Multi-resolution Features,” in CVPR, 2024, pp. 11591–11600.

[10] Amelia Schroeder et al., “Scaling Up Spatial Transcriptomics for Large-Sized Tissues: Uncovering Cellular-Level Tissue Architecture Beyond Conventional Platforms with iSCALE,” Nat. Methods, vol. 22, no. 9, pp. 1911–1922, 2025.

[11] Vincent Sitzmann et al., “Implicit Neural Representations with Periodic Activation Functions,” in NeurIPS, 2020, vol. 33, pp. 7462–7473.

[12] Richard J Chen et al., “Towards a General-purpose Foundation Model for Computational Pathology,” Nat. Med., vol. 30, no. 3, pp. 850–862, 2024.

[13] Yisi Luo et al., “STINR: Deciphering Spatial Transcriptomics via Implicit Neural Representation,” in CVPR, 2025, pp. 25930–25939.

[14] Qingtian Zhu et al., “SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics,” in ICML, 2025.

[15] Sanja Vickovic et al., “Three-dimensional Spatial Transcriptomics Uncovers Cell Type Localizations in The Human Rheumatoid Arthritis Synovium,” Commun. Biol., vol. 5, no. 1, pp. 129, 2022.

[16] Cantin Ortiz et al., “Molecular Atlas of The Adult Mouse Brain,” Sci. Adv., vol. 6, no. 26, pp. eabb3446, 2020.

[17] Sergii Domanskyi et al., “Nextflow Pipeline for Visium and H&E Data From Patient-derived Xenograft Samples,” Cell Rep. Methods, vol. 4, no. 5, 2024.

[18] Guilherme Potje et al., “XFeat: Accelerated Features for Lightweight Image Matching,” in CVPR, 2024, pp. 2682– 2691.

[19] Martin A Fischler et al., “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981.

[20] Adam Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in NeurIPS, 2019, pp. 8026 – 8037.

[21] Diederik P. Kingma and Jimmy Ba, “Adam: A Method for Stochastic Optimization,” arXiv preprint arXiv:1412.6980, 2014.

[22] Guillaume Jaume et al., “HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis,” in NeurIPS, 2024, vol. 37, pp. 53798–53833.