# Layer Selection in VLMs for Zero-Shot OOD Detection via Multi-Resolution Entropy Estimation

Shyam Nandan Rai, Francesco Di Salvo, Sebastian Doerrich, Christian Ledig

xAILab Bamberg, University of Bamberg, Bamberg, Germany

Out-of-distribution (OOD) detection is crucial for safe deployment of medical AI systems, where domain shifts arise across institutions, acquisition protocols, and patient populations. VLMs enable zero-shot OOD detection by embedding images into a language-aligned latent space, where cross-modal similarity serves as a non-parametric confidence signal for identifying in-distribution samples. Yet existing methods rely almost exclusively on final-layer embeddings, implicitly assuming that the deepest representations are universally optimal. We first show that this assumption does not hold in medical imaging: intermediate layers provide complementary OOD signals, and the optimal representational depth depends on the respective image modality. While prior work selects layer combinations via entropy minimization of normalized histograms, we demonstrate that single-resolution entropy estimation is highly sensitive to binning choices, leading to performance variations of up to 19.3% AUROC. To address this instability, we propose a multi-resolution entropy estimation strategy that aggregates histogram statistics across multiple discretization scales, enabling robust and stable intermediate-layer selection. Across two medical OOD benchmarks, namely MIDOG and OASIS, covering distinct imaging modalities, diverse shift types, and diferent VLM backbones, our method consistently outperforms state-of-the-art approaches, ofering a lightweight and stable solution for zero-shot OOD detection.

Keywords: Out-of-Distribution Detection · Medical VLMs

Website: https://shyam671.github.io/layer-selection-ood/ Contact: shyam.rai@uni-bamberg.de

![](images/25be60a3ae9d924aa9ab3e192112a5c933b1f3b1c3784dff869733abab7a97e3.jpg)

## 1 Introduction

Out-of-distribution (OOD) detection [26] is a critical research problem in healthcare, where models are deployed across diverse institutions, imaging devices, acquisition protocols, and heterogeneous patient populations [23,6,9]. Vision-Language Models (VLMs) [19,27,14] enable zero-shot OOD detection by measuring the alignment between image and text embeddings. This is particularly beneficial because the in-distribution manifold can be characterized through semantically rich textual descriptions, while cross-modal similarity serves as a proxy for prediction confidence. However, existing zero-shot OOD detection methods based on VLMs [18,12] primarily rely on final-layer representations, implicitly assuming that they are universally optimal for OOD detection. This assumption has recently been challenged for VLMs [10], as well as in other domains [25,11,4,16]. De la Jara et al. [10] show for the natural image domain that OOD-discriminative signals are distributed across multiple representational depths within VLMs, rather than being confined to the final layer. In particular, intermediate layers have been shown to capture diverse visual features [20] that complement high-level semantic representations. Despite these findings, a systematic investigation into whether aggregating layer-wise OOD scores can substantially improve detection performance in the context of medical image analysis remains unexplored. We therefore revisit the problem of zero-shot medica

OOD detection and ask: Can intermediate layer representations be systematically utilized to enhance zero-shot OOD detection across modalities?

To answer this question, we perform a systematic layer-wise analysis of zero-shot OOD detection across multiple medical imaging datasets, presented in Figure 1. Our results reveal that the optimal representational depth is modality-dependent: texture-driven shifts in histopathology [15] are primarily captured in earlier layers, whereas semantic and anatomical shifts in brain MRI [5] benefit from the global contextual representations encoded in deeper layers.

Motivated by these observations, we revisit entropy-based intermediate-layer selection [10] in the context of medical imaging. While originally developed for large-scale natural image benchmarks, we find that its reliance on single-resolution histogram entropy leads to unstable performance in small-scale medical datasets, where discretization sensitivity can degrade AUROC by up to 19.3% (cf. Figure 3). To address this limitation, we propose a multi-resolution entropy estimation strategy that aggregates entropy estimates across multiple discretization scales of histogram, enabling robust and stable layer selection. We evaluate our method on two medical OOD benchmarks using two VLM backbones, covering distinct imaging modalities and varying degrees of distribution shift. Our approach substantially reduces bin-size sensitivity and consistently improves OOD detection performance over state-of-the-art zero-shot baselines, demonstrating strong generalization across backbones and modalities. In summary:

– To the best of our knowledge, we provide the first systematic layer-wise analysis of VLMs for zero-shot OOD detection in medical image analysis, revealing that OOD-discriminative signals emerge at modality-dependent representational depths.

– We identify the instability of single-resolution entropy-based layer selection in data-scarce medical settings and introduce a multi-resolution entropy estimation strategy that yields robust layer selection.

– Extensive experiments across two medical OOD benchmarks and two VLM backbones demonstrate consistent performance improvements over state-of-the-art zero-shot baselines, spanning multiple imaging modalities and distribution shifts.

![](images/5a745ec2328b2a24b9ad103ed0acab49cc9834fdb364003b0aa10f86128083ad.jpg)

![](images/c41b2a1c3142b8cc246b076bbe3a17f40935c789ae34ca3db5bc5e9adae1c550.jpg)  
Fig. 1. Layer-wise zero-shot OOD detection performance (AUROC, %), using UniMedCLIP. Left: OASIS (brain MRI). Right: MIDOG (histopathology). Each bar corresponds to the OOD detection performance using Maximum Concept Matching (MCM) for the corresponding encoder layer, averaging for both near - and far-OOD shifts. The results demonstrate that the best layer depends on the dataset, i.e., imaging modality. Notably, the final layer does not exhibit the best performance in either case.

## 2 Intermediate layers for medical OOD detection

In this section, we investigate whether intermediate representations provide complementary signals for zero-shot OOD detection. We use UniMedCLIP [14] as the reference backbone. Let I denote an input image. The visual encoder $E$ consists of N sequential layers, producing intermediate representations $\{ L _ { j } ( I ) \} _ { j = 1 } ^ { N }$ , where $L _ { j } ( I )$ denotes the output of the j-th layer and $L _ { N } ( I ) = E ( I )$ corresponds to the canonical final-layer embedding. To enable layer-wise comparison within the shared image–text embedding space, each intermediate representation is projected using the same pretrained projection head as the final layer [10]. We used the datasets as described in Section 4 and Maximum Concept Matching (MCM) [18] as OOD scoring function.

Figure 1 shows that the final-layer embedding does not exhibit optimal performance. For MIDOG (histopathology), the strongest results are obtained in early layers. This indicates that fine-grained, texture-dominant features are particularly informative for detecting distribution shifts. In contrast, for OASIS (brain MRI), characterized by structured anatomical patterns, mid-level layers provide better performance. These observations demonstrate that OOD signals emerge at diferent representational depths depending on modality, motivating intermediate-layer selection rather than reliance on the final-layer representation.

![](images/f95da32e28767a81efb8c57edb8aeead734ef4384ed86155bd57de525f6c380e.jpg)  
Fig. 2. Overview of the proposed intermediate-layer selection framework. Given in-distribution images $I \in \mathbb { Z } _ { \mathrm { I D } }$ and prompts $\{ P ^ { 1 } , \ldots , P ^ { M } \}$ , the visual encoder outputs intermediate representations $\{ L _ { j } ( I ) \} _ { j = 1 } ^ { N } $ which are projected into the shared image–text embedding space and scored via layer-wise MCM: $S _ { \mathrm { M C M } } ^ { ( j ) } ( I )$ Candidate layer subsets ${ \mathcal { C } } ,$ always including the final layer $N ,$ are constructed, and their aggregated scores are collected across the in-distribution dataset to form the subset score distribution $S _ { C } .$ . This distribution is evaluated via multi-resolution entropy $\bar { H } ( S c )$ , and the optimal subset $\boldsymbol { \mathcal { C } ^ { * } }$ minimizing aggregated entropy is selected for test-time inference.

## 3 Method

Motivated by our observation that OOD signals emerge at modality-dependent representational depths (Figure 1), we propose a robust intermediate-layer selection method based on multi-resolution entropy estimation.

## 3.1 Preliminaries

VLMs learn a shared embedding space for images and text through contrastive pretraining. We consider CLIP-style models [19] consisting of an image encoder $E : \mathcal { T }  \mathbb { R } ^ { d }$ and a text encoder $T : \mathcal { P }  \mathbb { R } ^ { d }$ . Given a set of prompts describing the in-distribution concepts $\{ P ^ { 1 } , \ldots , P ^ { M } \}$ , the similarity between an input image I and prompt $P ^ { j }$ is computed via cosine similarity in the shared embedding space. The corresponding softmax probability is:

$$
p ( y ^ { j } \mid I ) = \frac { \exp { \left( \cos ( E ( I ) , T ( P ^ { j } ) ) / \tau \right) } } { \sum _ { m = 1 } ^ { M } \exp { \left( \cos ( E ( I ) , T ( P ^ { m } ) ) / \tau \right) } } .\tag{1}
$$

Maximum Concept Matching (MCM) [18] defines the OOD scoring function as: $S _ { \mathrm { M C M } } ( I ) =$ max<sub>j</sub> $p ( y ^ { j } \mid I )$ which is equivalent to the Maximum Softmax Probability (MSP) [8] used in vision models, but computed over image–text similarities. An input is in in-distribution if $S _ { \mathrm { M C M } } ( I ) \geq \theta$ and OOD otherwise.

## 3.2 Multi-resolution entropy estimation

Let $\scriptstyle { \mathcal { Z } } _ { \mathrm { I D } }$ denote the in-distribution image set and $\{ P ^ { 1 } , \ldots , P ^ { M } \}$ the corresponding prompts, where M denotes the number of available prompts. For each image $I \in \mathbb { Z } _ { \mathrm { I D } }$ , the visual encoder produces intermediate representations $\{ L _ { j } ( I ) \} _ { j = 1 } ^ { N }$ , which are projected into the shared image–text embedding space. N denotes the total number of representations from an image. For each layer $j ,$ , we compute the layer-wise MCM represented as $S _ { \mathrm { M C M } } ^ { ( j ) } ( I )$ . To aggregate complementary signals across represen tational depths, we evaluate all possible layer subsets while enforcing inclusion of the final layer [10]. Each candidate subset is defined as $\mathcal { C } = \{ N \} \cup \mathcal { C } ^ { \prime }$ , where $\mathcal { C } ^ { \prime } \subseteq \{ 1 , \ldots , N - 1 \}$ . For a given subset ${ \mathcal { C } } ,$ we form the score distribution used for entropy estimation by collecting the aggregated scores across the in-distribution images as:

$$
S _ { \mathcal { C } } = \left\{ \frac { 1 } { | \mathcal { C } | } \sum _ { j \in \mathcal { C } } S _ { \mathrm { M C M } } ^ { ( j ) } ( I ) \right\} _ { \forall I \in \mathcal { T } _ { \mathrm { I D } } } .\tag{2}
$$

Ensemble of histograms To obtain a robust uncertainty measure, we compute entropy across an ensemble of histograms $\{ B _ { k } \} _ { k = 1 } ^ { K }$ , where K denotes the number of binning scales. Formally, entropy for a histogram is calculated as:

$$
H _ { k } ( S _ { \mathcal { C } } ) = - \sum _ { b = 1 } ^ { B _ { k } } p _ { b } ^ { ( k ) } \log p _ { b } ^ { ( k ) } .\tag{3}
$$

The entropy estimates are then averaged across histogram resolutions,

$$
\bar { H } ( S c ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } H _ { k } ( S c ) ,\tag{4}
$$

We select the optimal layer subset as $\begin{array} { r } { \mathcal { C } ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { C } } \bar { H } ( S _ { \mathcal { C } } ) } \end{array}$ . Ablation studies in Section 5.2 demonstrate that multi-resolution entropy aggregation substantially reduces bin-size sensitivity compared to single-resolution estimation.

Test-time inference During inference, we use $\mathcal { C } ^ { * }$ along with the final layer to compute MCM scores. $\mathcal { C } ^ { * }$ is determined once on the ID set and remains the same for every test sample.

## 4 Experimentation

Baselines We compare our method against MCM [18] and Ju et al. [12], which perform zero-shot OOD detection using final-layer embeddings. Ju et al. improve performance by hierarchical prompting. We also include De la Jara et al. [10], the closest related approach, which selects intermediate layers via entropy minimization using a single histogram resolution, leading to discretization instability.

Datasets and metrics For evaluation, we adapt the recently proposed OpenMIBOOD benchmark [7] for VLM backbones, focusing on histopathology and brain MRI as representative medical imaging modalities. We excluded PhaKIR [21] dataset, as none of the VLM backbones were pretrained on surgical data, making predictions on this dataset unreliable. Our analysis centers on nearand far-OOD distribution shifts, enabling evaluation across varying degrees of semantic distribution shift within modality. For histopathology, we use MIDOG [3] (CC BY 4.0), where the in-distribution data consist of mitotic and non-mitotic cell crops extracted from H&E-stained whole-slide images. Near-OOD datasets introduce semantic shifts across cancer types, species, and acquisition settings, while far-OOD samples originate from cervical cancer cell images [2] (CCAgT, Apache 2.0) and breast FNAC cytology [22] (license not disclosed). For brain MRI, OASIS3-MRI [17] (custom license) serves as the in-distribution dataset, with OASIS3-CT used as near-OOD and MSD-H [24] (BSD 2-Clause) and CHAOS [13] (CC BY-NC-SA 4.0) as far-OOD datasets. OOD detection performance is evaluated using the Area Under the Receiver Operating Characteristic curve (AUROC) and the False Positive Rate at 95% True Positive Rate (FPR95).

Implementation details We use two CLIP-based medical VLMs: BioMedCLIP [27] and UniMed CLIP [14]. Each of them used ViT-B/16 backbone as the vision model. We perform all the evaluations in a training-free and inference-only setting. The temperature parameter τ is fixed to 1.0 across al evaluations. In-distribution prompts are generated using a large-scale language model [1], following prior work [12]. Unlike Ju et al. [12], which employ hierarchical prompt generation, we adopt a flat prompt structure to reduce redundancy. For fair comparison, the same prompt set is used for our method and all remaining baselines. We employ 9 prompts for MIDOG and 4 prompts for OASIS. An example prompt includes ‘a healthy brain MRI’ for OASIS and ‘a diagnostic histopathology slide showing breast carcinoma’ for MIDOG. For De la Jara et al. [10], we use 16 histogram bins, as recommended by the authors. Our multi-resolution entropy estimation utilizes a histogram ensemble with bin sizes {4, 8, 16, 32, 64}. For our method and De la Jara et al. [10], we constrain the layer subset size to at most four layers (including the final layer) and select the subset that minimizes the respective entropy criterion using only in-distribution data.

Results Table 1 summarizes OOD detection performance across UniMedCLIP and BioMedCLIP backbones on the MIDOG and OASIS benchmarks. On OASIS, with both backbones, our method achieves near-perfect performance across both near- and far-OOD settings and. Notably, the largest performance gains are observed in FPR95: on BioMedCLIP, our method reduces FPR95 by approximately 20% in both near- and far-OOD compared to the strongest baseline. On the MIDOG benchmark, our method demonstrates the best overall results with the UniMedCLIP backbone, achieving the best near-OOD performance (AUROC: 54.6%, FPR95: 94.4%) and far-OOD performance (AUROC: 97.4%, FPR95: 13.4%). With BioMedCLIP, results are more mixed across methods. For instance, MCM [18] achieves the best results on near-OOD, while De la Jara et al.[10] achieve the best far-OOD results, suggesting that no single method dominates consistently, making comparisons on this backbone less conclusive. Overall, UniMedCLIP emerges as the more reliable backbone for OOD detection in our evaluation, yielding stronger and more stable results across both benchmarks. Within this setting, our method achieves the best or competitive performance in nearly al configurations, highlighting its efectiveness and robustness for medical OOD detection.

Table 1. OOD detection results (AUROC ↑, FPR95 ↓) across two backbones (UniMedCLIP and BioMed-CLIP) and two datasets (MIDOG and OASIS), each evaluated under near- and far-OOD settings. Despite some variability on BioMedCLIP for MIDOG, our method consistently achieves strong performance and substantially outperforms reference approaches across most configurations.
<table><tr><td rowspan="3"></td><td colspan="4">UniMedCLIP</td><td colspan="4">BioMedCLIP</td></tr><tr><td colspan="2">Near-OOD</td><td colspan="2">Far-OOD</td><td colspan="2">Near-OOD</td><td colspan="2">Far-OOD</td></tr><tr><td>AUROC ↑FPR95↓AUROC ↑FPR95↓ AUROC ↑FPR95↓AUROC ↑FPR95↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">MIDOG</td></tr><tr><td>MCM [18]</td><td>50.5</td><td>94.5</td><td>84.3</td><td>49.3</td><td>52.7</td><td>95.6</td><td>46.3</td><td>99.1</td></tr><tr><td>Ju et al. [12]</td><td>40.5</td><td>98.0</td><td>37.4</td><td>99.7</td><td>50.2</td><td>97.4</td><td>84.8</td><td>66.0</td></tr><tr><td>De la Jara et al. [10]</td><td>29.1</td><td>99.7</td><td>97.1</td><td>16.6</td><td>32.1</td><td>97.3</td><td>99.8</td><td>1.3</td></tr><tr><td>Ours</td><td>54.6</td><td>94.4</td><td>97.4</td><td>13.4</td><td>36.4</td><td>100.0</td><td>99.5</td><td>2.7</td></tr><tr><td colspan="9">OASIS</td></tr><tr><td>MCM [18]</td><td>55.2</td><td>99.9</td><td>8.6</td><td>100.0</td><td>90.4</td><td>34.8</td><td>49.0</td><td>99.9</td></tr><tr><td>Ju et al. [12]</td><td>25.3</td><td>100.0</td><td>27.0</td><td>99.2</td><td>85.7</td><td>34.8</td><td>92.9</td><td>20.4</td></tr><tr><td>De la Jara et al. [10]</td><td>100.0</td><td>0.0</td><td>97.7</td><td>6.5</td><td>50.0</td><td>78.5</td><td>60.7</td><td>39.3</td></tr><tr><td>Ours</td><td>100.0</td><td>0.0</td><td>99.2</td><td>1.0</td><td>97.3</td><td>14.3</td><td>100.0</td><td>0.0</td></tr></table>

## 5 Ablation studies

We ablate two fundamental components of our method: (1) the impact of the number of aggregated intermediate layers, and (2) the robustness of multi-resolution entropy estimation compared to single-resolution histogram binning. All experiments are performed with UniMedCLIP unless stated otherwise.

## 5.1 Number of selected layers

We study how performance varies as additional intermediate layers are incorporated into the aggregation. Starting from the final layer alone, we progressively add layers selected by our entropy criterion. We report the analysis on MIDOG (near), because OASIS produces near-saturated AU-ROC in this setting, leaving little headroom to measure the marginal gains from adding intermediate layers. Figure 3 (left) shows that aggregating more intermediate representations consistently improves OOD detection for UniMedCLIP, increasing AUROC above the last-layer baseline. However, in BioMedCLIP, AUROC eventually decreases as more intermediate layers are aggregated, showing layer selection is beneficial compared to using all layers. This confirms that OOD-relevant signals are distributed across layers rather than confined to the final layer.

## 5.2 Sensitivity of histogram bins

We next evaluate the sensitivity of entropy-based selection to the histogram bin count B. Following De la Jara et al. [10], we implement a single-resolution entropy baseline and sweep recommended bin counts {4, 8, 16, 32, 64}. We conduct the analysis on OASIS (near), reflecting a realistic medica setting with limited in-distribution data (<1,000 samples). As shown in Figure 3 (right), OOD performance varies substantially across bin sizes (standard deviation $\sigma = 8 . 2 \% )$ with a maximum drop of 19.3%, highlighting the fragility of single-resolution entropy estimation. In contrast, our multi-resolution ensemble-of-histograms achieves stable, near-optimal performance without requiring manual bin-size selection.

![](images/cc7e1e7cfd03431d3372737e6c84b8db4b2885f15f242cb2290f053c364c8f0f.jpg)

![](images/45065a9689ecd741c38e34374e9407b491a851cc43a40f61ca864d0649bd7c0a.jpg)  
Fig. 3. Left: OOD performance on MIDOG (near) as a function of the number of aggregated intermediate layers. Right: Sensitivity analysis on OASIS (near) of single-resolution entropy-based layer selection across diferent histogram bin sizes, where performance varies by up to 19.3%. In contrast, our multi-resolution entropy estimation strategy achieves stable and near-optimal performance.

## 6 Discussion

Limitations Our approach remains dependent on the underlying VLM backbone, and although we evaluate two widely established and important medical modalities, further validation across additional imaging domains is required. Moreover, we employ uniform averaging for layer aggregation, which may not capture potentially optimal weighted combinations. Future works could explore finer-grained scoring strategies beyond MCM, for instance, utilizing the entire prompt-wise score distributions.

Conclusions In this work, we demonstrate that OOD signals in medical Vision–Language Models emerge at modality-dependent representational depths, challenging the common reliance on finallayer embeddings. Through a systematic layer-wise analysis, we reveal substantial instability in single-resolution entropy-based layer selection and show that binning choices can significantly afect performance. To address this, we introduce a multi-resolution entropy estimation strategy that enables robust intermediate-layer aggregation. Across datasets and backbones, our results show that our method ofers a simple yet efective strategy for enhancing zero-shot OOD detection.

Acknowledgments The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU). The hardware is funded by the German Research Foundation (DFG). This study was further funded through the Hightech Agenda Bayern (HTA) of the Free State of Bavaria, Germany.

Disclosure of Interests The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. Amorim, J.G.A., Macarini, L.A.B., Matias, A.V., Cerentini, A., Onofre, F.B.D.M., Onofre, A.S.C., Wangenheim, A.V.: A novel approach on segmentation of agnor-stained cytology images using deep

learning. In: 2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS). IEEE (Jul 2020). https://doi.org/10.1109/cbms49503.2020.00110, https://doi.org/10.1109/ cbms49503.2020.00110

3. Aubreville, M., Wilm, F., Stathonikos, N., Breininger, K., Donovan, T.A., Jabari, S., Veta, M., Ganz, J., Ammeling, J., Van Diest, P.J., et al.: A comprehensive multi-domain dataset for mitotic figure detection. Scientific data 10(1), 484 (2023)

4. Darrin, M., Staerman, G., Gomes, E.D.C., Cheung, J.C., Piantanida, P., Colombo, P.: Unsupervised layer-wise score aggregation for textual ood detection. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 17880–17888 (2024)

5. Desgranges, B., Matuszewski, V., Piolino, P., Chételat, G., Mézenge, F., Landeau, B., de La Sayette, V., Belliard, S., Eustache, F.: Anatomical and functional alterations in semantic dementia: a voxel-based mri and pet study. Neurobiology of aging 28(12), 1904–1913 (2007)

6. Finlayson, S.G., Subbaswamy, A., Singh, K., Bowers, J., Kupke, A., Zittrain, J., Kohane, I.S., Saria, S.: The clinician and dataset shift in artificial intelligence. New England Journal of Medicine 385(3), 283–286 (2021)

7. Gutbrod, M., Rauber, D., Nunes, D.W., Palm, C.: Openmibood: Open medical imaging benchmarks for out-of-distribution detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25874–25886 (2025)

8. Hendrycks, D., Gimpel, K.: A baseline for detecting misclassified and out-of-distribution examples in neural networks. In: International Conference on Learning Representations (2017), https://openreview. net/forum?id=Hkg4TI9xl

9. Hong, Z., Yue, Y., Chen, Y., Cong, L., Lin, H., Luo, Y., Wang, M.H., Wang, W., Xu, J., Yang, X., et al.: Out-of-distribution detection in medical image analysis: A survey. arXiv preprint arXiv:2404.18279 (2024)

10. De la Jara, I.M., Rodriguez-Opazo, C., Teney, D., Ranasinghe, D., Abbasnejad, E.: Mysteries of the deep: Role of intermediate representations in out of distribution detection. Advances in Neural Information Processing Systems (2025)

11. Jelenić, F., Jukić, J., Tutek, M., Puljiz, M., Snajder, J.: Out-of-distribution detection by leveraging between-layer transformation smoothness. In: The Twelfth International Conference on Learning Rep resentations (2024), https://openreview.net/forum?id=AcRfzLS6se

12. Ju, L., Zhou, S., Zhou, Y., Lu, H., Zhu, Z., Keane, P.A., Ge, Z.: Delving into Out-of-Distribution Detection with Medical Vision-Language Models . In: Proceedings of Medical Image Computing and Computer Assisted Intervention (2025)

13. Kavur, A.E., Selver, M.A., Dicle, O., Barış, M., Gezer, N.S.: Chaos-combined (ct-mr) healthy abdominal organ segmentation challenge data. (No Title) (2019)

14. Khattak, M.U., Kunhimon, S., Naseer, M., Khan, S., Khan, F.S.: Unimed-clip: Towards a unified imagetext pretraining paradigm for diverse medical imaging modalities. arXiv preprint arXiv:2412.10372 (2024)

15. Kong, H., Gurcan, M., Belkacem-Boussaid, K.: Partitioning histopathological images: an integrated framework for supervised color-texture segmentation and cell splitting. IEEE transactions on medical imaging 30(9), 1661–1677 (2011)

16. Lambert, B., Forbes, F., Doyle, S., Dojat, M.: Multi-layer aggregation as a key to feature-based ood detection. In: International workshop on uncertainty for safe utilization of machine learning in medical imaging. pp. 104–114. Springer (2023)

17. LaMontagne, P.J., Benzinger, T.L., Morris, J.C., Keefe, S., Hornbeck, R., Xiong, C., Grant, E., Hassen stab, J., Moulder, K., Vlassenko, A.G., et al.: Oasis-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease. medrxiv pp. 2019–12 (2019)

18. Ming, Y., Cai, Z., Gu, J., Sun, Y., Li, W., Li, Y.: Delving into out-of-distribution detection with visionlanguage representations. Advances in neural information processing systems 35, 35087–35102 (2022)

19. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML. Proceedings of Machine Learning Research (2021)

20. Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., Dosovitskiy, A.: Do vision transformers see like convolutional neural networks? In: Beygelzimer, A., Dauphin, Y., Liang, P., Vaughan, J.W. (eds.) Advances in Neural Information Processing Systems (2021), https://openreview.net/forum?id=R-616EWWKF5

21. Rueckert, T., Rauber, D., Maerkl, R., Klausmann, L., Yildiran, S.R., Gutbrod, M., Nunes, D.W., Moreno, A.F., Luengo, I., Stoyanov, D., et al.: Comparative validation of surgical phase recognition, instrument keypoint estimation, and instrument instance segmentation in endoscopy: Results of the phakir 2024 challenge. Medical Image Analysis p. 103945 (2026)

22. Saikia, A.R., Bora, K., Mahanta, L.B., Das, A.K.: Comparative assessment of cnn architectures for classification of breast fnac images. Tissue and Cell 57, 8–14 (2019)

23. Stacke, K., Eilertsen, G., Unger, J., Lundström, C.: Measuring domain shift for deep learning in histopathology. IEEE journal of biomedical and health informatics 25(2), 325–336 (2020)

24. Tobon-Gomez, C., Geers, A.J., Peters, J., Weese, J., Pinto, K., Karim, R., Ammar, M., Daoudi, A., Margeta, J., Sandoval, Z., et al.: Benchmark for algorithms segmenting the left atrium from 3d ct and mri datasets. IEEE transactions on medical imaging 34(7), 1460–1473 (2015)

25. Wei, T., Wang, B.L., Shi, J.X., Li, Y.F., Zhang, M.L.: X-mahalanobis: Transformer feature mixing for reliable OOD detection. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum?id=ewyR20zwqA

26. Yang, J., Zhou, K., Li, Y., Liu, Z.: Generalized out-of-distribution detection: A survey. International Journal of Computer Vision 132(12), 5635–5662 (2024)

27. Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., Preston, S., Rao, R., Wei, M., Valluri, N., et al.: Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv (2023)