# B-MIM: Biased Masked Image Modeling for Generalizable Segmentation of Fine-Grained Anatomical Structures

Sebastián González<sup>1</sup>, Karen Sanchez<sup>2</sup>, José M. Saavedra<sup>3</sup>, Marcelo Pizarro<sup>3</sup>, and Bernard Ghanem<sup>2</sup>

<sup>1</sup> Universidad de Chile, Santiago, Chile

<sup>2</sup> King Abdullah University of Science and Technology (KAUST), Saudi Arabia <sup>3</sup> Universidad de los Andes, Santiago, Chile sebagonzalez@ug.uchile.cl

Abstract. Self-supervised pretraining enables transferable representations for medical imaging, yet most CT encoders remain biased toward coarse semantic understanding, limiting their sensitivity to fine-grained anatomical structures such as vessels or small tumors. In this paper, we introduce Biased Masked Image Modeling (B-MIM), a modification of the iBOT objective that stochastically reduces global semantic alignment to prioritize local patch reconstruction. This bias encourages the encoder to capture high-frequency morphological details and structural continuity. We curate a multi-institutional CT abdominal dataset of 9,955 filtered studies from 17 public sources and pretrain a 3D Swin Transformer backbone using B-MIM. Across inter-dataset experiments on liver vessel segmentation, the proposed encoder improves topological fidelity (clDice) and achieves competitive Dice scores in tumor segmentation, compared to fully fine-tuned baselines, despite updating only a fraction of the parameters. Our results suggest that reducing global semantic pressure during pretraining enhances generalization to intricate anatomical structures.

Keywords: Self-supervised Learning · medical imaging · encoder · finegrained · CT.

## 1 Introduction

Computerized Tomography (CT) scans are a critical, non-invasive diagnostic tool in modern medicine that combine multiple X-ray images to produce detailed, 3D views of internal structures. They are essential for rapidly diagnosing, staging, and monitoring diseases—particularly cancer, cardiovascular issues, and trauma—guiding treatment, and reducing the need for exploratory surgery [3]. In addition, machine learning is increasingly the preferred method for extracting and interpreting information from these kinds of images, as evidenced by a vast body of publications on machine learning for medical imaging [20].

The de facto standard for a machine learning model follows an encoderdecoder architecture. The encoder aims to extract relevant information from images, text, or any input modality, while the decoder uses those representations to accomplish a target task. It is now a common practice to use encoders trained under a self-supervised regimen because of their capability to learn from unlabeled data [5, 17, 1].

Learning transferable representations facilitates adaptation to diverse tasks, allowing the model to drastically reduce its number of learnable parameters, which is highly important in low-resource environments. That is why self-supervised learning strategies have become critical for making semantic encoders available. In this vein, models like MoCo [8], SimCLR [4] and BYOL [6] have contributed to the state of the art on image representation, which is dominated by the DINO encoder family [13, 17].

The medical imaging domain has benefited from advances in computer vision and machine learning. Thus, we are seeing encoders for X-ray images, such as RadDINO [14], RayDINO [12] and Google’s CXR [10]. However, for the CT domain, there are still a few proposals. Here, VoCo [23, 24], a self-supervised strategy for 3D medical imaging using contrastive learning, showed improvements across diverse downstream tasks using CT or MRI.

However, existing pretrained models and encoders for medical imaging (e.g., Total Segmentator [21], Segment-Any-Muscle [22], UKBOB [2]) and encoders like VoCo [23, 24] focus mainly on coarse-grained tasks, such as macro-organ segmentation, with comparatively less emphasis on fine-grained anatomical structures, such as vessels or small tumors. Fine-grained structures play a critical role in disease prevention, early detection, disease monitoring, and treatment planning. For instance, liver vessel segmentation [16] is critical to plan liver resection, and the early detection of small tumors is crucial to cancer treatment.

Therefore, this work proposes a CT encoder designed to enhance representations of fine-grained anatomical structures, such as vessels and small tumors. We evaluate our encoder on liver vessel and small-tumor segmentation tasks, showing its potential for inter-dataset transfer. For instance, we observed up to a 18% gain in vessel segmentation when the proposal is applied to an out-of-distribution dataset. Thus, our contributions are summarized as follows:

– We curate a multi-institutional dataset from 17 public sources into a standardized cohort of 9,955 CT studies (1,993,194 2D slices) with consistent LPS orientation and anatomical cropping for 3D self-supervised learning.

– We introduce the Biased Masked Image Modeling (B-MIM) objective, which prioritizes local patch reconstruction over global semantic alignment. By leveraging the multi-scale nature of the 3D Swin Transformer, this strategy encourages the encoder to capture high-frequency morphological details, specifically addressing the topological continuity required for fine-grained structures.

– We demonstrate that our pre-trained backbone serves as a fixed-feature extractor, achieving competitive results on specialized tasks (e.g., vessel and tumor segmentation) with only 16k trainable parameters.

## 2 Related Works

Self-Supervised Learning (SSL): Modern artificial intelligence models are based on an encoder-decoder scheme, where the encoder is responsible for properly representing an image, text, or audio signal into a semantic feature space (a.k.a. latent space). In recent years, we have seen the emergence of increasingly powerful image encoders [4, 6, 13, 17, 1]. For an encoder to be truly efective, it is essential to make use of as much available data as possible, even though this data often lacks proper labeling. To this end, self-supervision plays a key role, enabling the model to learn from diverse images and understand discriminative structures that facilitate downstream inference. Furthermore, the absence of labels during the encoder training stage can help models generalize better avoiding underfitting to specific classes.

In this vein, DinoV3 [17] is one of the most efective visual encoders; it incorporates iBOT [25], a mechanism to extract information from small local structures. The idea is that a patch encoding can be reconstructed with contextual information when the image patch is occluded. Thus, we leverage iBOT loss to allow the model to focus on small structures.

In the context of medical imaging, we still find a few works proposing medical visual encoders. Most available encoders are designed for the X-ray domain. For instance, META released RayDino [12], extending the DINO architecture using a collection of chest X-rays. RayDino is a large visual encoder trained by self-supervision on 873,000 chest X-rays. RayDINO shows state-of-the-art performance across many radiology tasks, from classification and dense segmentation to text generation, and provides an in-depth analysis of population, age, and sex biases of the model. The authors suggest that SSL allows patient-centric AI to be useful in clinical workflows and to interpret X-rays holistically. In the same direction, RadDino [14] was proposed by Microsoft, and CXR by Google [10].

VoCo [23, 24] is a pretrained methodology for 3D medical images that proposes the Volume Contrast (VoCo) framework to leverage contextual position priors during pre-training. This method groups crops from diferent regions while enforcing feature discrepancy among them. As the X-Ray encoders, VoCo is not focused on fine-grained tasks, which is our main goal throughout this work.

Pretrained encoders favor generalization and make adaptation easier. These properties are critical in medical imaging. However, there is a significant gap to cover, especially in tasks involving small anatomical structures using image modalities other than X-ray. Therefore, this work proposes a CT encoder specifically designed to improve sensitivity to fine-grained anatomical structures, such as vessels and small tumors.

## 3 Methodology

Our methodology comprises three stages: i) the preparation and curation of the training dataset, ii) the design of the encoder architecture, and iii) an evaluation methodology based on the inter-dataset generalization property. Following, we describe each of the proposed components.

## 3.1 Data Curation and Large-Scale Pre-processing

To train an encoder for cross-dataset segmentation of fine-grained abdominal structures, we curated a large-scale multi-institutional CT dataset by aggregating 17 publicly available sources in both DICOM and NIfTI formats. The initial collection underwent a rigorous automated filtering and normalization pipeline, including standardized orientation and automated abdominal cropping to remove non-relevant anatomical regions and ensure consistent anatomical representation.

Filtering Criteria. We applied a standardized selection protocol to both data formats. For DICOM series, we excluded localizer scans and instances with null orientation headers or inconsistent slice orientations (e.g., mismatched firstand middle-slice headers). Furthermore, we restricted the cohort to adult patients $\mathrm { ( a g e \ge 1 8 }$ or unspecified) and enforced a minimum volumetric depth of 20 slices per series after abdominal cropping. A critical requirement for inclusion was the simultaneous presence of the liver and at least one kidney, to ensure that the model learns consistent abdominal spatial relationships.

Dataset Composition. From 13,784 abdominal CTs originally compiled, the final filtered cohort comprised 9,955 studies (cases), totaling 1,993,194 2D slices. The following summary highlights the primary data streams: DICOM Cohort: A total of 5,613 studies were selected from sources such as rsna2023- abdominal (3,147 studies), TCGA-KIRC (362), and CT-Colonography (831), totaling 1,529,954 2D slices. NIfTI Cohort: 4,342 studies, totaling 463,240 slices, were retained from datasets including AMOS (2,338 cases), AbdomenCT-1K (1,061), and KiTS23 (480). Notably, AbdomenCT-1K itself integrates subsets from LiTS, MSD Spleen, and MSD Pancreas.

Normalization and Cropping. All volumes were normalized to the LPS (Left-Posterior-Superior) orientation. To eliminate non-relevant anatomical noise, we performed automated abdominal cropping using TotalSegmentator [21] to identify landmarks. The vertical bounds were defined from the superior aspect of the liver to the inferior pole of the L4 vertebra or the kidneys, whichever was lower. TotalSegmentator is used exclusively for automated Region-of-Interest localization during preprocessing. No downstream task annotations (e.g., vessels or tumors) are used at this stage, ensuring that the cropping procedure does not introduce supervision related to the evaluation tasks.

The curated large-scale abdominal CT cohort will be publicly released upon acceptance. Due to space constraints, the complete list of source datasets and corresponding references is available in the repository: https://github.com/ksanchez84/B-MIM

## 3.2 Biased Masked Image Modeling (B-MIM)

Figure 1 shows the general scheme of our proposed encoder and how it is integrated into a downstream model. Following, we describe each of the B-MIM components.

![](images/65646b9da2acd683ebd2ad6755317253e3e014abd434c20da9cad45a93f6fcf9.jpg)  
Fig. 1. General scheme of the proposed CT encoder and how it is integrated into a downstream segmentation task.

Backbone Architecture. We employ a 3D Swin Transformer [11, 7, 19] as the backbone for our encoder. The input 3D CT volumes are partitioned into non-overlapping patches, and a hierarchical architecture computes self-attention within shifting windows. This setup allows the model to capture both local textures and long-range spatial dependencies, providing the sensitivity required to represent detailed, fine-grained structures with high fidelity.

Pre-training Objective. The encoders are pre-trained using the iBOT framework [25], framed as a cross-view self-distillation task. Given two augmented views of a scan, the student network learns to match the teacher network’s output using a dual distillation loss. Following iBOT, the teacher network has the same architecture as the student and is updated as an exponential moving average of the student parameters. The global $L _ { \mathrm { C L S } }$ and masked-token L<sub>MIM</sub> distillation losses are defined as

$$
L _ { \mathrm { C L S } } = H ( P _ { t } ^ { \mathrm { C L S } } , P _ { s } ^ { \mathrm { C L S } } ) , \qquad L _ { \mathrm { M I M } } = \frac { 1 } { | \mathcal { M } | } \sum _ { i \in \mathcal { M } } H ( P _ { t , i } , P _ { s , i } ) ,\tag{1}
$$

where H denotes cross-entropy and M the set of masked patches. However, to enhance the segmentation of fine-grained structures, we propose a Biased-MIM objective, where the total loss $L _ { t o t a l }$ is formulated as:

$$
L _ { t o t a l } = \mathbb { I } _ { p } \cdot L _ { C L S } + L _ { M I M } , \qquad \mathbb { I } _ { p } \sim \mathrm { B e r n o u l l i } ( p ) .\tag{2}
$$

where $L _ { C L S }$ is the self-distillation loss between [CLS] tokens, $L _ { M I M }$ is the distillation between masked patch tokens, and $\mathbb { I } _ { p } \in \{ 0 , 1 \}$ is a Bernoulli random variable with probability p, sampled independently at each training iteration (per batch), determining whether the global distillation loss $L _ { C L S }$ is applied for that batch. In contrast to deterministic loss weighting, stochastic omission alters the optimization dynamics by intermittently removing global supervision altogether, thereby reducing co-adaptation between global and local objectives and promoting robustness to fine-scale morphological variation.

Rationale for the Bias. The standard iBOT formulation treats both losses with equal priority $( p = 1 )$ . However, to enhance the segmentation of fine-grained structures, we set $p < 1$ to intentionally reduce the frequency of global feature alignment. By stochastically omitting global supervision, we encourage the student network $f _ { s }$ to rely more heavily on reconstructing masked tokens via $L _ { M I M }$ This bias encourages the encoder to learn high-resolution local geometries and continuity, which are essential for thin structures, rather than over-relying on coarse global semantic features.

Parameter-Eficient Downstream Adaptation To evaluate the representational quality of the pre-trained embeddings, we employ a parameter-eficient adaptation strategy. Instead of conventional full fine-tuning, which is computationally expensive and prone to forgetting anatomical priors in small datasets, we attach a lightweight nnU-Net decoder to the frozen Swin-3D encoder.

Encoder-Decoder Integration. For adaptation, we tested the following three primary configurations:

1. Feature Concatenation: Merging multi-scale embeddings from the pretrained encoder into the corresponding decoder levels. The encoder is kept largely frozen, with only 16,344 parameters updated during adaptation. We call this setting B-MIM-A.

2. Full Finetuning: We performed full finetuning on the same architecture in order to assess diferences from updating all encoder parameters. We call this setting B-MIM-B.

3. LoRA Adaptation (Optional/Comparative): We integrated Low-Rank Adaptation [9] to further refine the bottleneck without unfreezing the backbone. We call this setting B-MIM-C.

## 3.3 Generalization-based Evaluation

We evaluate our proposed encoder on two fine-grained downstream tasks: liver vessel segmentation and small-tumor segmentation using abdominal CTs as input. In both cases, we use the CRLM dataset (197 studies) [16] for training and validation. Given that one of our main goals is to evaluate generalization, we use two datasets for testing, not used during pretraining: IRCAD (20 studies) [18] and MSD (303 studies) [15]. The encoder is pretrained in a fully self-supervised manner on the curated abdominal CT cohort, without access to vessel or tumor annotations. For downstream evaluation, models are trained exclusively on CRLM (5-fold cross-validation), and evaluated on IRCAD and MSD as external test sets. This protocol ensures patient-level separation and prevents label leakage between pretraining and downstream evaluation.

## 4 Results and Discussion

The 3D Swin Transformer backbone was pretrained using the B-MIM objective for 200 epochs with a batch size of 6. We adopted the standard iBOT optimization configuration with AdamW, using a linear warm-up for the first 10 epochs followed by a cosine learning rate schedule decaying from $7 . 5 \times 1 0 ^ { - 4 } \mathrm { ~ t o ~ } 2 \times 1 0 ^ { - 6 }$

For downstream segmentation, models were trained using 5-fold cross validation $\left( \mathrm { k } = 5 \right)$ , with a batch size of 2 for 200 epochs, following the default nnU-Net training protocol. Optimization was performed using SGD with a polynomial learning rate scheduler (initial learning rate = $1 \times 1 0 ^ { - 2 }$ , exponent = 0.9) and a constant weight decay of $3 \times 1 0 ^ { - 5 }$ . We report mean Dice and clDice scores across folds. The encoder remained frozen unless otherwise specified. All experiments were conducted on an NVIDIA RTX 6000 Ada Generation (48 GB) GPU.

## 4.1 Segmentation Performance and Inter-Dataset Generalization

A key property of the proposed approach is its ability to generalize across diverse clinical domains without unfreezing the backbone. All models were trained on CRLM, and we report mean Dice and clDice across folds. Generalization was evaluated on IRCAD and MSD, which exhibit distinct acquisition protocols and anatomical variability. In all experiments presented in this section, B-MIM results use $p = 0 . 3$ , which provided a stable trade-of between global semantic alignment and local morphological reconstruction in preliminary validation experiments.

Vessel Segmentation. As shown in Table 1, B-MIM consistently enhances topological fidelity (clDice) in vessel segmentation across datasets. Notably, under domain shift from CRLM to IRCAD, B-MIM-C improves clDice from 0.491 (Swin-nnUNet) to 0.582, corresponding to a gain of +0.091 absolute points (+18.5% relative improvement). This improvement in topological fidelity is substantially larger than the corresponding Dice variation, highlighting the benefit of the biased objective for preserving vascular continuity across datasets.

Further, even the lightweight B-MIM-A configuration, with only 16,344 trainable encoder parameters, achieves competitive Dice and clDice scores on both IRCAD and MSD, demonstrating that fine-grained morphological representations can generalize across datasets with minimal adaptation.

Figure 2 provides qualitative examples for both liver vessel and tumor segmentation. For vessel segmentation, B-MIM-C better preserves the continuity of thin vascular branches and exhibits less fragmentation than Swin-UNETR and VoComni. For tumor segmentation, B-MIM-B produces delineations that more closely follow the ground-truth boundaries in the highlighted regions.

Parameter Eficiency. A core contribution of this work is to achieve highfidelity segmentation with minimal architectural updates. A key observation from Table 1 is the trade-of between trainable parameters and generalization performance. While B-MIM-B updates 5,716,410 parameters and achieves strong tumor performance, B-MIM-A maintains competitive vessel segmentation with only 16,344 trainable parameters. This suggests that the pretrained encoder captures transferable fine-grained morphological priors that require only lightweight adaptation.

![](images/7e3bb9b46fbaab2004914c4e1b63b79f975674095fe98436eba5710f254da864.jpg)  
Fig. 2. Qualitative comparison for fine-grained liver vessel and tumor segmentation. The top row shows liver vessel predictions, where B-MIM better preserves the continuity of thin vascular branches and reduces local fragmentation compared with Swin-UNETR and VoComni. The bottom row shows tumor segmentation, where B-MIM provides predictions that more closely match the ground truth in the highlighted regions. Zoomed areas emphasize local diferences in fine-grained structure delineation.

Table 1. Segmentation performance across datasets for diferent encoder configurations and adaptation strategies. Dice and clDice are reported for liver vessels, while Dice is reported for tumors.
<table><tr><td rowspan="3">Model</td><td colspan="6">Liver Vessels</td><td colspan="3">Tumors</td></tr><tr><td>CRLM</td><td></td><td>IRCAD</td><td></td><td>MSD</td><td></td><td>CRLM</td><td>IRCAD</td><td>MSD</td></tr><tr><td>Dice</td><td>clDice</td><td>Dice</td><td>clDice</td><td>Dice</td><td>clDice</td><td>Dice</td><td>Dice</td><td>Dice</td></tr><tr><td>Swin-nnUNet</td><td>0.746 0.780</td><td></td><td>0.400</td><td>0.491</td><td>0.598</td><td>0.624</td><td>0.646</td><td>0.572</td><td>0.374</td></tr><tr><td>VoCo_SSL</td><td>0.616</td><td>0.645</td><td>0.382</td><td>0.437</td><td>0.488</td><td>0.480</td><td>0.650</td><td>0.369</td><td>0.288</td></tr><tr><td>VoComni</td><td>0.632</td><td>0.662</td><td>0.455</td><td>0.494</td><td>0.572</td><td>0.588</td><td>0.666</td><td>0.500</td><td>0.423</td></tr><tr><td>B-MIM-A</td><td>0.744</td><td>0.777</td><td>0.413</td><td>0.500</td><td>0.611</td><td>0.631</td><td>0.631</td><td>0.497</td><td>0.357</td></tr><tr><td>B-MIM-B</td><td>0.745</td><td>0.780</td><td>0.401</td><td>0.490</td><td>0.604</td><td>0.630</td><td>0.651</td><td>0.578</td><td>0.395</td></tr><tr><td>B-MIM-C</td><td>0.746</td><td>0.779</td><td></td><td>0.492 0.582</td><td>0.567</td><td>0.595</td><td>0.637</td><td>0.506</td><td>0.269</td></tr></table>

Tumor Segmentation. In tumor segmentation, performance diferences are more variable. B-MIM-B achieves the best Dice in this setting, likely due to the higher variability in tumor morphology and size. Unlike vessels, which exhibit consistent tubular structure, tumor morphology exhibits higher inter-patient variability in size, shape, and contrast patterns, making representation transfer inherently more challenging.

## 5 Conclusions

We introduced B-MIM, a biased masked pretraining objective designed to enhance representations of fine-grained anatomical structures in CT. By reducing global semantic pressure during self-supervised training, the proposed encoder improves topological fidelity in vessel segmentation while remaining parametereficient during downstream adaptation. Our experiments show that vessel structures, which exhibit a consistent tubular morphology, benefit noticeably from this representation bias and achieve improved cross-dataset transfer in our evaluation. In contrast, tumor segmentation remains more challenging due to higher variability in size, shape, and appearance. Future work will involve a more comprehensive evaluation across fine-grained tasks stratified by lesion size and complexity, a systematic analysis of the sensitivity to $p ,$ and extension to additional modalities such as MRI.

## References

1. Bolya, D., Huang, P., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., Wang, J., Monteiro, M., Xu, H., Dong, S., Ravi, N., Li, D., Dollár, P., Feichtenhofer, C.: Perception encoder: The best visual embeddings are not at the output of the network. CoRR abs/2504.13181 (2025)

2. Bourigault, E., Jamaludin, A., Hamdi, A.: Ukbob: One billion mri labeled masks for generalizable 3d medical image segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 21600–21611 (2025)

3. Buzug, T.M.: Computed Tomography, pp. 311–342. Springer Berlin Heidelberg, Berlin, Heidelberg (2011)

4. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.E.: A simple framework for contrastive learning of visual representations. In: Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event. Proceedings of Machine Learning Research, vol. 119, pp. 1597–1607 (2020)

5. Ericsson, L., Gouk, H., Loy, C.C., Hospedales, T.M.: Self-supervised representation learning: Introduction, advances, and challenges. IEEE Signal Process. Mag. 39(3), 42–62 (2022)

6. Grill, J., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.Á., Guo, Z., Azar, M.G., Piot, B., Kavukcuoglu, K., Munos, R., Valko, M.: Bootstrap your own latent - A new approach to self-supervised learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020)

7. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H., Xu, D.: Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. arXiv preprint arXiv:2201.01266 (2022)

8. He, K., Fan, H., Wu, Y., Xie, S., Girshick, R.B.: Momentum contrast for unsupervised visual representation learning. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020. pp. 9726–9735 (2020)

9. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: ICLR. OpenReview.net (2022)

10. Kiraly, A.P., Baur, S., Philbrick, K., Mahvar, F., Yatziv, L., Chen, T., Sterling, B., George, N., Jamil, F., Tang, J., Bailey, K., Ahmed, F., Goel, A., Ward, A., Yang, L., Sellergren, A., Matias, Y., Hassidim, A., Shetty, S., Golden, D., Azizi, S., Steiner, D.F., Liu, Y., Thelin, T., Pilgrim, R., Kirmizibayrak, C.: Health ai developer foundations (2024)

11. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 10012–10022 (2021)

12. Moutakanni, T., Bojanowski, P., Chassagnon, G., Hudelot, C., Joulin, A., LeCun, Y., Muckley, M.J., Oquab, M., Revel, M.P., Vakalopoulou, M.: Advancing humancentric ai for robust x-ray analysis through holistic self-supervised learning. arXiv (2024), arXiv preprint arXiv:2405.01469

13. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P., Li, S., Misra, I., Rabbat, M.G., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision. CoRR (2023)

14. Pérez-García, F., Sharma, H., et al.: Exploring scalable medical image encoders beyond text supervision. Nature Machine Intelligence 7, 119–130 (2025)

15. Simpson, A.L., Antonelli, M., Bakas, S., Bilello, M., Farahani, K., Van Ginneken, B., Kopp-Schneider, A., Landman, B.A., Litjens, G., Menze, B., et al.: A large annotated medical image dataset for the development and evaluation of segmentation algorithms. arXiv preprint arXiv:1902.09063 (2019)

16. Simpson, A.L., Peoples, J., Creasy, J.M., Fichtinger, G., Gangai, N., Keshavamurthy, K.N., Lasso, A., Shia, J., D’Angelica, M.I., Do, R.K.G.: Preoperative ct and survival data for patients undergoing resection of colorectal liver metastases. Scientific Data 11(1), 172 (Feb 2024)

17. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: Dinov3 (2025)

18. Soler, L., Hostettler, A., Agnus, V., Charnoz, A., Fasquel, J., Moreau, J., Osswald, A., Bouhadjar, M., Marescaux, J.: 3d image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database. IRCAD, Strasbourg, France, Tech. Rep 1(1) (2010)

19. Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V., Hatamizadeh, A.: Self-supervised pre-training of swin transformers for 3d medical image analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20730–20740 (2022)

20. Varoquaux, G., Cheplygina, V.: Machine learning for medical imaging: methodological failures and recommendations for the future. npj Digital Medicine 5(1) (Apr 2022). https://doi.org/10.1038/s41746-022-00592-y, http://dx.doi.org/10.1038/s41746-022-00592-y

21. Wasserthal, J., Breit, H.C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D.T., Cyriac, J., Yang, S., Bach, M., Segeroth, M.: Totalsegmen-

tator: Robust segmentation of 104 anatomic structures in ct images. Radiology: Artificial Intelligence 5(5), e230024 (2023)

22. Wesselink, E.O., Elliott, J.M., McKay, M.J., Martino, E.D., Caplan, N., Mackey, S., Cohen-Adad, J., Bédard, S., Leener, B.D., Enamundram, M.V.N.K., Law, C., Fortin, M., Vleggeert-Lankamp, C., Ieva, A.D., Kim, B., Hancock, M.J., Pool-Goudzwaard, A., Pevenage, P., I., K.A.W.I.: Segment-any-muscle: Towards an open-source, contrast-agnostic computer-vision muscle segmentation model for mri and ct. In: ISMRM & ISMRT Annual Meeting (ISMRM-ISMRT 2025). Institute of Industrial and Systems Engineers (IISE) (September 2025). https://doi.org/10.58530/2025/0622

23. Wu, L., Zhuang, J., Chen, H.: Voco: A simple-yet-efective volume contrastive learning framework for 3d medical image analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22873– 22882 (2024)

24. Wu, L., Zhuang, J., Chen, H.: Large-scale 3d medical image pre-training with geometric context priors. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

25. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: Image bert pre-training with online tokenizer. In: International Conference on Learning Representations (2022)