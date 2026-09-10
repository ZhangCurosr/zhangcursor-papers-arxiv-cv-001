# BEYOND WEAK LABELS: PROMPT-GUIDED LOCAL REFINEMENT FOR WEAKLYSUPERVISED WATER SEGMENTATION IN HIGH-RESOLUTION MULTISPECTRALIMAGERY

Muhammad Farhan Humayun<sup>1</sup> Mohammad Imangholiloo<sup>2</sup> Afifah Shah<sup>1</sup> Tomi Westerlund<sup>1</sup> Jukka Heikkonen<sup>1</sup>

<sup>1</sup>Department of Computing, University of Turku, Finland <sup>2</sup>Department of Geoinformatics and Cartography, Finnish Geospatial Research Institute, National Land Survey of Finland, Espoo, Finland

## ABSTRACT

High-resolution water mapping supports environmental monitoring and related applications, but accurate pixel-level labels are difficult and costly to produce. Official hydrographic vectors provide scalable weak supervision, but they contain artifacts like boundary noise, temporal mismatch, and omissions of small water structures. We propose a two-stage framework for weakly supervised water segmentation in highresolution multispectral imagery. Stage 1 learns initial masks from rasterized vector pseudo-labels, and Stage 2 converts these masks into structured component-wise prompts for localized refinement. On a manually corrected validation set, refinement improves SegFormer-B0 from 0.9509 to 0.9535 IoU and U-Net from 0.9408 to 0.9486 IoU, with corresponding F1 gains from 0.9749 to 0.9762 and 0.9695 to 0.9736. It leads to sharper shorelines, reduced boundary spillover, and better thin-structure delineation. The results indicate that prompt-guided refinement can improve pseudo-label-based water segmentation by targeting local errors that are poorly captured by global training supervision.

Index Terms— Weak supervision, water segmentation, multispectral imagery, prompts, SegFormer, SAM 2.

## 1. INTRODUCTION

Accurate water-body delineation from high-resolution multispectral imagery is important for hydrologic monitoring, flood response, environmental assessment, and geospatial inventory maintenance [1, 2]. Compared with coarse-resolution products, high-resolution orthophotos derived from aerial and satellite imagery better capture object-level features such as shorelines, narrow channels, and small inland water bodies [3, 4]. However, they also introduce substantial challenges such as water boundaries are irregular, target structures are often thin and fragmented, and visually similar non-water regions can cause local ambiguities [5, 6]. A major difficulty in this setting is the lack of reliable pixel-level ground truth.

Hydrographic mapping vector layers are attractive because they are widely available and inexpensive compared with exhaustive manual annotation, but they are not perfect for supervision in deep learning models as they often contain boundary noise, temporal mismatch with the imagery, geometric simplification, and omission of small water bodies or narrow channels. They are therefore useful as pseudo-labels, but not as exact ground truth, making weak supervision a natural formulation for this problem [7, 8].

Under these constraints, modern segmentation architectures such as SegFormer have shown strong segmentation performance and robustness across image resolutions [9]. Such models are well suited as strong baselines for weakly supervised remote sensing segmentation because they can learn from large pseudo-labeled datasets while retaining the capacity to model both local detail and broader scene context [10]. Yet high aggregate accuracy alone is not sufficient in this setting. In high-resolution water mapping, many tiles are dominated by large easy regions of land or open water, and global pixel-wise scores can therefore hide crucial errors that matter most in practice including shoreline placement, thin channels, small omitted water bodies and islands [11]. The central challenge is thus not only coarse water detection, but the refinement of fine-scale local structure under imperfect supervision.

This research proposes a two-stage framework for weakly supervised water segmentation. The first stage produces an initial water mask using pseudo-label supervision, while the second stage performs prompt-guided local refinement using the Segment Anything Model 2 (SAM 2) [12]. In this formulation, the refinement stage is guided by explicit local instructions that indicate where refinement should occur and how the promptable model should interpret each candidate region. The resulting two-stage formulation is conceptually simple and practically useful. Our contributions are threefold: (1) a weakly supervised multispectral water-segmentation pipeline; (2) a structured component-wise, prompt-generation scheme for localized mask refinement; and (3) evidence that local prompt-guided refinement improves a strong baseline, particularly for boundary adjustment and recovery of small water/land components.

![](images/7c2eb6c09180019e4cdabda7f8556885154a8a694511b73c9cb9b53f39f56b56.jpg)  
Fig. 1. Prompt-guided two-stage pipeline: (A) preprocessing, (B) weakly supervised SegFormer segmentation, (C) structured prompt generation, and (D) localized SAM 2 mask refinement.

## 2. MATERIALS AND METHODS

The study area is located in Joensuu, Finland, and includes diverse water bodies such as lakes, ponds, river channels, and narrow streams. We use high-resolution orthophotos from National Land Survey of Finland (NLS), combining RGB and NIR bands into four-channel multispectral inputs. Weak supervision is obtained by rasterizing hydrographic vector layers from the Finnish Environment Institute (SYKE) into pseudo-label masks.

## 2.1. Pre-processing and Pseudo-labels

Figure 1 summarizes the two-stage pipeline. SYKE hydrographic vector layers were rasterized to the 50 cm orthophoto grid and used as weak pseudo-labels. The four-channel images and masks were tiled into 256 × 256 patches. All watercontaining tiles were retained, including sparse-water and boundary cases, while pure-land tiles were heavily subsampled as hard negatives to emphasize shorelines, water-land transitions, and narrow structures. Since the pseudo-labels contained boundary noise, temporal mismatch, simplification artifacts, and omissions, evaluation used a manually corrected strong-label subset from held-out regions, rasterized to the same grid, extent, resolution, and coordinate reference system. The final dataset contained 8184 training tiles and

810 strong-label validation tiles, approximately a 90:10 split.

## 2.2. Weakly Supervised Baseline Segmentation

Stage 1 generates the initial water prediction for each multispectral tile. Let $\mathbf { X } _ { i } \in \mathbb { R } ^ { H \times W \times 4 }$ denote the i-th RGB-NIR input tile and $\tilde { \mathbf { Y } } _ { i } \in \{ 0 , 1 \} ^ { H \times W }$ denote its rasterized pseudolabel mask. We use SegFormer-B0 as the primary baseline because it provides a favorable balance between efficiency and segmentation performance. The model is initialized with pretrained weights and adapted to binary water segmentation using four-channel inputs. The network predicts pixel-wise logits:

$$
\begin{array} { r } { \mathbf { Z } _ { i } = f _ { \theta } ( \mathbf { X } _ { i } ) , \qquad \mathbf { Z } _ { i } \in \mathbb { R } ^ { H \times W \times 2 } , } \end{array}\tag{1}
$$

where $f _ { \theta }$ denotes the SegFormer network and the two output channels correspond to non-water and water, respectively. Pixel-wise class probabilities are then obtained by a softmax operation:

$$
\mathbf { P } _ { i } = \operatorname { s o f t m a x } ( \mathbf { Z } _ { i } ) , \qquad \mathbf { P } _ { i } \in [ 0 , 1 ] ^ { H \times W \times 2 } .\tag{2}
$$

Training is performed using pseudo-label masks derived from official vector layers, whereas evaluation is conducted only on manually corrected strong labels. The network is optimized using pixel-wise cross-entropy loss for binary semantic segmentation:

$$
\mathcal { L } _ { \mathrm { s e g } } = - \sum _ { u , v } \sum _ { c = 1 } ^ { 2 } \tilde { Y } _ { i , c } ( u , v ) \log P _ { i , c } ( u , v ) .\tag{3}
$$

The final binary prediction is obtained by selecting the most probable class at each pixel:

$$
\hat { \mathbf { Y } } _ { i } ( u , v ) = \arg \operatorname* { m a x } _ { c \in \{ 0 , 1 \} } P _ { i , c } ( u , v ) .\tag{4}
$$

To support Stage 2 refinement, we also derive a confidence map from the water-class probability. Let $P _ { i , 1 } ( u , v )$ denote the predicted probability of the water class. The confidence is defined as the distance from the decision boundary:

$$
{ \bf C } _ { i } ( u , v ) = 2 \left| P _ { i , 1 } ( u , v ) - 0 . 5 \right| .\tag{5}
$$

Thus, pixels with values close to 0.5 are treated as uncertain, while pixels close to 0 or 1 are treated as confident predictions. The Stage-1 output consists of the binary mask $\hat { \mathbf { Y } } _ { i }$ and confidence map $\mathbf { C } _ { i } .$ , which together provide the coarse water extent and uncertainty cues for prompt-guided local refinement in Stage 2. We use SegFormer-B0 as the primary Stage-1 baseline because it provides a favorable balance between efficiency and segmentation performance. For comparison, we also experimented with a classical U-Net backbone [13] as an alternative Stage-1 baseline under the same weakly supervised training and evaluation protocol.

## 2.3. Structured Spatial Prompt Generation

This module provides the bridge between Stage 1 and Stage 2. For each Stage-1 prediction, connected water components are extracted from the binary mask and converted into structured spatial prompts. Let $\hat { \mathbf { Y } _ { i } } \in \{ 0 , 1 \} ^ { H \times W }$ denote the Stage-1 binary prediction for tile i, and let $\mathbf { \bar { C } } _ { i } \in [ 0 , 1 ] ^ { H \times W }$ denote the corresponding confidence map. The set of connected water components is defined as

$$
\begin{array} { r } { \mathcal { K } _ { i } = \mathrm { C C } ( \hat { \mathbf { Y } } _ { i } ) , } \end{array}\tag{6}
$$

where CC(·) denotes connected-component extraction. For each connected component $k \in \mathcal { K } _ { i }$ , its spatial support is defined as

$$
\Omega _ { i k } = \{ ( u , v ) \mid ( u , v ) { \mathrm { b e l o n g s ~ t o ~ c o m p o n e n t ~ } } k \} .\tag{7}
$$

The corresponding bounding box is then computed as

$$
\mathbf b _ { i k } = [ x _ { \operatorname* { m i n } } , y _ { \operatorname* { m i n } } , x _ { \operatorname* { m a x } } , y _ { \operatorname* { m a x } } ] ,\tag{8}
$$

Each component is represented by a structured prompt object

$$
\pi _ { i k } = \{ \mathbf b _ { i k } , \mathcal P _ { i k } ^ { + } , \mathcal P _ { i k } ^ { - } , r _ { i k } , m _ { i k } \} ,\tag{9}
$$

where $\mathbf { b } _ { i k }$ is the component bounding box, $\mathcal { P } _ { i k } ^ { + }$ and $\mathcal { P } _ { i k } ^ { - }$ denote sets of positive and negative point prompts, $r _ { i k }$ denotes the local refinement region, and $m _ { i k }$ denotes the refinement mode. Positive points are sampled from the component interior and high-confidence foreground support, while negative points are sampled from nearby non-water, boundaryadjacent, or hole-like regions depending on the local geometry and confidence cues.

The refinement mode specifies how Stage 2 should interpret the local region, for example whether refinement is focused on boundary correction, hole preservation, or suspicious-component recovery. These structured prompt objects are stored in per-tile JSON files for reproducibility and inspection. During Stage-2 inference, the JSON contents are parsed and converted into SAM 2-compatible spatial inputs.

## 2.4. Prompt-Guided Local Mask Refinement

Stage 2 refines the Stage-1 prediction using SAM 2 in a local, component-wise manner. For each tile, the RGB image $\mathbf { I } _ { i } ^ { \mathrm { R G B } }$ and the structured prompt object $\pi _ { i k }$ are provided to SAM 2 to generate one or more candidate masks for component k:

$$
\mathcal { M } _ { i k } = g _ { \phi } ( \mathbf { I } _ { i } ^ { \mathrm { R G B } } , \pi _ { i k } ) ,\tag{10}
$$

where $g _ { \phi }$ denotes the promptable segmentation model and $\mathcal { M } _ { i k }$ is the set of candidate masks produced for the local region.

The generated candidates are not accepted directly. Instead, each candidate mask $\mathbf { M } _ { i k } ^ { j } \in \mathcal { M } _ { i k }$ is filtered using geometric consistency checks with respect to the Stage-1 component and its allowed refinement region. In particular, we require sufficient agreement with the reference component and constrain the relative area change:

$$
\frac { | \mathbf { M } _ { i k } ^ { j } \cap \Omega _ { i k } | } { | \Omega _ { i k } | } \geq \alpha , \qquad \beta _ { \mathrm { m i n } } \leq \frac { | \mathbf { M } _ { i k } ^ { j } | } { | \Omega _ { i k } | } \leq \beta _ { \mathrm { m a x } } ,\tag{11}
$$

where $\Omega _ { i k }$ is the spatial support of the Stage-1 component, α is the minimum overlap threshold, and $\beta _ { \mathrm { m i n } }$ and $\beta _ { \mathrm { m a x } }$ define acceptable lower and upper bounds on area change. Additional mode-dependent screening is applied to prevent implausible updates outside the allowed local region.

After filtering, the selected candidate $\mathbf { M } _ { i k } ^ { * }$ is merged back into the Stage-1 prediction only within a restricted local refinement region $r _ { i k }$ :

$$
\hat { \mathbf { Y } } _ { i } ^ { \mathrm { r e f } , k } ( u , v ) = \left\{ \mathbf { M } _ { i k } ^ { * } ( u , v ) , \quad ( u , v ) \in r _ { i k } , \right.\tag{12}
$$

and the final refined mask is obtained by applying this update sequentially over the component set. Thus, regions outside the allowed refinement zone preserve the original Stage-1 prediction. This prevents the promptable model from unnecessarily rewriting already plausible large water bodies. The refinement branch therefore acts primarily as a boundary-aware local correction module i.e., it targets uncertain boundaries, small structures, and locally ambiguous regions while preserving the coarse semantic layout learned by Stage 1.

## 2.5. Implementation Details

Stage 1 used the SegFormer-B0 model initialized from the official pretrained checkpoint. The architecture was kept unchanged except for a four-channel RGB–NIR input layer and a binary water/non-water output head. The model was finetuned for 50 epochs with batch size 32, learning rate $6 \times 1 0 ^ { - 5 } .$ and weight decay $1 \times 1 0 ^ { - 4 }$ . Stage 2 used the official SAM 2 image predictor with the SAM 2.1 Hiera-small checkpoint. SAM 2 was not modified; refinement was controlled externally using JSON-derived spatial prompts, geometric candidate filtering, and local mask-merging rules. All experiments were run in Python/PyTorch under Anaconda on an NVIDIA RTX 3500 Ada GPU with 12 GB VRAM and 32 GB RAM. Performance was evaluated on the manually corrected stronglabel validation set using pooled IoU, F1-score, precision, recall, overall accuracy, and pixel-level TP, FP, TN, and FN counts.

Table 1. Performance comparison of weakly supervised Stage-1 baselines and their corresponding SAM 2-based promptguided refinements on the manually corrected strong-label validation set.
<table><tr><td>Method</td><td>TP (%)</td><td>FP (%)</td><td>FN(%)</td><td>TN (%)</td><td>IoU</td><td>F1</td><td>Precision</td><td>Recall</td><td>Accuracy</td></tr><tr><td>Stage 1 (SegFormer-B0)</td><td>41.40</td><td>1.42</td><td>0.72</td><td>56.46</td><td>0.9509</td><td>0.9748</td><td>0.9669</td><td>0.9829</td><td>0.9786</td></tr><tr><td>Stage 2 (SAM 2 Refinement)</td><td>41.34</td><td>1.24</td><td>0.78</td><td>56.65</td><td>0.9535</td><td>0.9762</td><td>0.9709</td><td>0.9814</td><td>0.9798</td></tr><tr><td>Stage 1 (U-Net)</td><td>40.71</td><td>1.16</td><td>1.41</td><td>56.72</td><td>0.9408</td><td>0.9695</td><td>0.9723</td><td>0.9666</td><td>0.9743</td></tr><tr><td>Stage 2 (SAM 2 Refinement)</td><td>40.97</td><td>1.07</td><td>1.15</td><td>56.81</td><td>0.9485</td><td>0.9736</td><td>0.9745</td><td>0.9726</td><td>0.9778</td></tr></table>

## 3. RESULTS AND DISCUSSION

Table 1 compares Stage-1 weakly supervised baselines with their SAM 2-based prompt-guided refinements on the manually corrected validation set. For SegFormer-B0, Stage 2 reduced false positives by 12.8% (685,840 to 598,124), improving precision from 0.9669 to 0.9710, accuracy from 0.9786 to 0.9799, IoU from 0.9509 to 0.9535, and F1-score from 0.9749 to 0.9762. Recall slightly decreased from 0.9829 to 0.9815, indicating that refinement mainly suppressed false water detections while introducing a small increase in missed water pixels.

For U-Net, SAM 2 refinement reduced both false positives and false negatives respectively. This improved IoU from 0.9408 to 0.9486, F1-score from 0.9695 to 0.9736, recall from 0.9666 to 0.9727, and accuracy from 0.9744 to 0.9778. The consistent gains across both Transformer-based and convolutional baselines indicate that the structured prompt-guided branch improves weakly supervised water segmentation beyond a single backbone.

Although the aggregate gains are modest, they are meaningful for high-resolution water mapping, where large easy land/open-water regions can hide local errors in global metrics. Stage 2 is therefore best interpreted as a boundary-aware local refinement module rather than a semantic replacement for Stage 1. Its main effect is improved shoreline alignment and correction of difficult local structures, with occasional recovery of severely incorrect Stage-1 regions.

The visual results confirm that Stage 2 mainly performs local boundary correction rather than global mask rewriting. In Fig. 2, refinement reduces shoreline spillover and produces cleaner water-land transitions while preserving the main Stage-1 water structure. Fig. 3 further shows the improved local mask quality for both SegFormer-B0 and U-Net, especially along irregular shorelines, narrow channels, and fragmented water structures. For SegFormer-B0, tile-level IoU improves in representative cases from 0.872 to 0.948 and from 0.902 to 0.941, mainly through reduced boundary spillover. For U-Net, refinement similarly improves boundary placement and suppresses local false positives. In one difficult case, the baseline fails almost completely, while Stage 2 recovers a plausible water mask and improves IoU from 0.000 to 0.964. These examples are consistent with Table 1 and support Stage 2 as a boundary-aware local correction module whose success still depends on Stage-1 mask quality and generated prompts.

![](images/112bd0a0981e9681bb1197e05482d8cd8242e66d8c9ac70c33764d044c76d2b3.jpg)  
Fig. 2. Zoomed-in local error maps before and after refinement of SegFormer-B0 visualizing improvement of waterbody delineations.

![](images/57e68f67282508f41ac6f1ad4095ddca4739bbdf383970ab0003d0af036bfe2d.jpg)  
Fig. 3. Visual Comparison of stage 1, weak baseline segmentation and stage 2 prompt-based refinement. Top three rows correspond to SegFormer-B0 as the stage 1 model whereas bottom three rows show results from U-Net as stage 1 model.

## 4. CONCLUSION

This study presented a two-stage weakly supervised framework for high-resolution multispectral water segmentation. Stage 1 used SegFormer-B0 and U-Net to learn from rasterized official hydrographic pseudo-labels, while Stage 2 converted their predictions into structured spatial prompts for localized SAM 2 refinement, evaluated on a manually corrected strong-label subset. SAM 2 refinement improved both baselines: SegFormer-B0 increased from 0.9509 to 0.9535 IoU and U-Net from 0.9408 to 0.9486 IoU, with corresponding F1 gains from 0.9749 to 0.9762 and 0.9695 to 0.9736. In representative tiles, the proposed approach produced larger local gains in tile-level IoU by reducing false spillover in uncertain boundary regions. The gains were mainly due to reduced false positives, sharper shoreline placement, and better handling of narrow or fragmented water structures.

The results show that prompt-guided refinement is effective as a boundary-aware local correction module. This provides a practical route for improving pseudo-label-based water segmentation under limited annotations, especially for fragmented or narrow water bodies with shoreline spillover. Future work will explore richer terrain and spectral cues and test transferability across high-resolution Earth observation settings.

## 5. ACKNOWLEDGEMENTS

This research has been conducted with Flagship Programme funding granted by the Research Council of Finland for Digital Waters Flagship (decision no. 359247 and decision no. 359249). This research was also supported by the Postdoctoral Programme for Research Institutes in Finland funded by the Finnish Government.

## 6. REFERENCES

[1] Yansheng Li, Bo Dang, Yongjun Zhang, and Zhenhong Du, “Water body classification from high-resolution optical remote sensing imagery: Achievements and perspectives,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 187, pp. 306–327, 2022.

[2] Marc Wieland, Sandro Martinis, Ralph Kiefl, and Veronika Gstaiger, “Semantic segmentation of water bodies in very high-resolution satellite and aerial images,” Remote Sensing of Environment, vol. 287, pp. 113452, 2023.

[3] Y. Sui, M. Feng, C. Wang, and X. Li, “A high-resolution inland surface water body dataset for the tundra and boreal forests of north america,” Earth System Science Data, vol. 14, no. 7, pp. 3349–3363, 2022.

[4] M. F. Humayun, A. Salmasi, J. Zhang, T. Westerlund, and J. Heikkonen, “A robust integrated approach for near real-time seamless orthomosaic generation using off-the-shelf uavs for ultra-high resolution mapping applications,” ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. X-2/W2-2025, pp. 65–72, 2025.

[5] Huidong Cao, Yanbing Tian, Yanli Liu, and Ruihua Wang, “Water body extraction from high spatial resolution remote sensing images based on enhanced u-net and multi-scale information fusion,” Scientific Reports, vol. 14, no. 1, pp. 16132, Jul 2024.

[6] Libei Fan, Yuting Dong, Ji Zhao, Christian Geiß, Lizhe Wang, and Hannes Taubenbock, “Mapping of small water bodies with integrated spatial information for time series images of optical remote sensing,” in IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium, 2022, pp. 6224–6227.

[7] Ming Lu, Leyuan Fang, Muxing Li, Bob Zhang, Yi Zhang, and Pedram Ghamisi, “Nfanet: A novel method for weakly supervised water extraction from high-resolution remote-sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2022.

[8] Hunsoo Song, Lexie Yang, and Jinha Jung, “Selffiltered learning for semantic segmentation of buildings in remote sensing imagery with noisy labels,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 1113–1129, 2023.

[9] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo, “Segformer: Simple and efficient design for semantic segmentation

with transformers,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, Eds. 2021, vol. 34, pp. 12077–12090, Curran Associates, Inc.

[10] Lei Ding, Dong Lin, Shaofu Lin, Jing Zhang, Xiaojie Cui, Yuebin Wang, Hao Tang, and Lorenzo Bruzzone, “Looking outside the window: Wide-context transformer for the semantic segmentation of high-resolution remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2022.

[11] Bowen Cheng, Ross Girshick, Piotr Dollar, Alexander C. Berg, and Alexander Kirillov, “Boundary iou: Improving object-centric image segmentation evaluation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 15334–15342.

[12] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chlo¨ e Rolland, Laura Gustafson, Eric´ Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and´ Christoph Feichtenhofer, “SAM 2: Segment anything in images and videos,” in Proceedings of the International Conference on Learning Representations, 2025.

[13] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, Eds., Cham, 2015, pp. 234–241, Springer International Publishing.