# Joint-Embedding Prediction of Masked Point Tubes for Self-Supervised Learning on 4D Point Cloud Videos

Jheng-Ling Lee, Shang-Tse Chen

Department of Computer Science and Information Engineering, National Taiwan University b11902129@csie.ntu.edu.tw, stchen@csie.ntu.edu.tw

## Abstract

Self-supervised representation learning for 4D point cloud videos is challenging because annotations are costly and reconstruction-based pretraining can overemphasize low-level geometric details. We propose a JEPA-style framework that learns from unlabeled spatiotemporal point clouds through latent point-tube prediction. Instead of reconstructing raw coordinates, the model masks spatiotemporal regions and predicts their target representations from visible context representations in feature space. To stabilize latent prediction, we incorporate Sketched Isotropic Gaussian Regularization, which encourages non-collapsed embeddings without relying on explicit reconstruction targets. This formulation aims to capture both spatial structure and temporal dynamics while keeping the pretraining objective aligned with downstream semantic recognition. Experiments on action and gesture recognition benchmarks show that the learned representations improve downstream fine-tuning, limited-label learning, and crossdataset transfer. These results suggest that JEPA-style latent prediction is a promising alternative to reconstructioncentered pretraining for 4D point cloud videos.

Code: https://github.com/Leoxu3/PointTube-JEPA

## Introduction

Self-supervised learning has become an important approach to visual representation learning because it can exploit large unlabeled corpora while reducing dependence on costly taskspecific annotation. Masked autoencoding and contrastive learning have demonstrated that useful representations can be learned from images, videos, and point clouds by defining supervisory signals from the input itself (He et al. 2022; Feichtenhofer et al. 2022; Xie et al. 2020; Pang et al. 2022). This property is particularly relevant for 4D point cloud videos, where each sample contains unordered 3D point sets evolving over time. Labels for action or gesture recognition are expensive to obtain, and purely supervised training can bias the representation toward the annotated downstream task instead of the full spatiotemporal structure of the signal.

Recent self-supervised methods for point cloud videos have made progress by extending masked modeling, contrastive prediction, and distillation to dynamic point sets (Shen et al. 2023a,b; Zhang et al. 2023; Zuo et al. 2025; Nguyen et al. 2026). These approaches show that unlabeled point cloud videos contain useful cues for recognition and transfer. However, many still rely on low-level coordi nate reconstruction, manually designed motion objectives, or external teacher targets. These choices may respectively bias learning toward local geometric details, introduce taskspecific motion assumptions, or create dependence on the choice and domain of the teacher.

Joint-Embedding Predictive Architectures (JEPA) ofer a complementary direction. Instead of reconstructing raw observations, JEPA-style methods predict target representations from context representations in latent space (Assran et al. 2023; Bardes et al. 2024). This formulation has been explored for images, videos, and static point clouds (Saito, Kudeshia, and Poovvancheri 2025; Hu et al. 2024), and is well suited to 4D point cloud videos because the prediction target can focus on spatiotemporal features rather than exact coordinate recovery. A central challenge is avoiding representation collapse. While many predictive self-supervised systems use momentum target encoders, stop-gradient operations, or related stabilization heuristics, LeJEPA introduces Sketched Isotropic Gaussian Regularization (SIGReg) as an explicit embedding regularizer for stable latent prediction (Balestriero and LeCun 2025). This motivates a simpler formulation in which prediction and regularization are both defined directly on learned embeddings.

In this paper, we study JEPA-style self-supervised learning for 4D point cloud videos. The proposed framework tokenizes a point cloud video into local spatiotemporal pointtube tokens, masks a large subset of these tokens, and trains a predictor to infer the masked target representations from visible context representations. The objective is applied in latent space rather than in the raw coordinate space, and SIGReg regularizes the latent distribution to discourage collapse. By combining point-tube tokenization, 4D positional information, context-to-target latent prediction, and SIGReg, the method aims to learn representations that capture both spatial geometry and temporal dynamics without requiring manual motion labels or reconstruction-specific decoders.

Our contributions are summarized as follows:

• We develop a teacher-free, shared-encoder JEPA formulation for 4D point cloud videos, where masked point-tube latents are predicted from visible spatiotemporal context without coordinate reconstruction.

• We stabilize end-to-end context-target learning with SI-GReg and systematically study gradient routing, targetlocation conditioning, masking ratio, and regularization strength.

• We evaluate the learned representation under full-label fine-tuning, semi-supervised learning, few-shot recognition, transfer learning, and qualitative attention-map visualizations.

## Related Work

## 4D Point Cloud Video Understanding

Supervised 4D point cloud video understanding studies how to extract spatial geometry and temporal motion from unordered point sets observed over time. Early point-based models preserve temporal structure directly in the point domain: PointLSTM formulates gesture recognition as irregular sequence modeling over neighboring points (Min et al. 2020), while MeteorNet builds local spatiotemporal neighborhoods for dynamic point cloud sequences (Liu, Yan, and Bohg 2019). Subsequent architectures introduce stronger 4D operators. P4Transformer combines point 4D convolution for local spatiotemporal embedding with Transformer attention over video-level features (Fan, Yang, and Kankanhall 2021); PSTNet decomposes point spatiotemporal convolution into spatial and temporal components (Fan et al. 2021); and PST-Transformer uses spatiotemporal attention to relate local regions without explicit point tracking (Fan, Yang, and Kankanhalli 2023).

Recent work extends this progression toward longer sequences, structured geometry, and eficient sequence modeling. Point Primitive Transformer represents long-term point cloud videos through primitive planes and hierarchical attention (Wen et al. 2022), while LeaF learns local coordinate frames to factorize geometry and motion under changing poses (Liu et al. 2023). State-space backbones such as Mamba4D and UST-SSM reorganize 4D point data into sequences for eficient long-range modeling (Liu et al. 2025; Li et al. 2025). Spectral approaches further mix spatial, temporal, and frequency-domain cues for point cloud action recognition and 4D understanding (Li et al. 2026; Wu et al. 2026). These supervised methods provide increasingly capable 4D backbones, but their representations are still primarily shaped by labeled downstream objectives.

## Self-Supervised Learning on Point Cloud Videos

Self-supervised learning on point cloud videos reduces reliance on costly sequence-level annotations by deriving training targets from the 4D signal itself. Early work used temporal pretext tasks, such as predicting the order of shufled point cloud clips, to encourage motion-sensitive representations (Wang et al. 2021). Contrastive objectives construct positive and negative relations in feature space: PointCMP combines local and global branches with contrastive mask prediction (Shen et al. 2023b), point contrastive prediction with semantic clustering performs point-level contrast over superpoints (Sheng et al. 2023), and contrastive predictive autoencoders combine prediction, contrastive learning, and reconstruction for dynamic point cloud sequences (Sheng, Shen, and Xiao 2023). Distillation-based methods provide another route, with Complete-to-Partial 4D Distillation training a student from partial observations under guidance from complete sequences (Zhang et al. 2023).

Masked modeling is another central direction. Image and video MAE methods show that high-ratio masking with reconstruction can produce transferable visual representations (He et al. 2022; Feichtenhofer et al. 2022), and static point cloud pretraining adapts this idea through point tokens or masked point patches (Yu et al. 2022; Pang et al. 2022). For point cloud videos, MaST-Pre masks spatiotemporal point tubes and couples point-tube reconstruction with temporal cardinality-diference prediction (Shen et al. 2023a). Later methods blend masking with additional supervision: M2PSC adds masked motion trajectory prediction and semantic contrast (Han et al. 2024), while Uni4D disentangles low-level geometry reconstruction from latent alignment of frame-level motion and video-level information (Zuo et al. 2025). More recently, DiMP introduces difusion-based masked-center inference and probabilistic inter-frame motion supervision while retaining Chamfer-based geometric reconstruction (Zhang et al. 2026). These methods establish the usefulness of unlabeled 4D data, but many still depend on coordinate reconstruction, explicit motion proxy targets, contrastive sample design, or teacher targets.

## Joint-Embedding Predictive Architectures

Joint-Embedding Predictive Architectures learn by predicting target representations from context representations rather than reconstructing raw observations. This non-generative formulation encourages the encoder to model information useful at the latent semantic level, reducing pressure to preserve all low-level details. I-JEPA applies this idea to images by predicting target-block embeddings from visible context blocks (Assran et al. 2023), and V-JEPA extends feature prediction to video using latent targets without reconstruction, negative examples, text supervision, or pretrained image encoders (Bardes et al. 2024). Point-JEPA and 3D-JEPA adapt the same principle to point clouds through context-target sampling over point patches or 3D blocks (Saito, Kudeshia, and Poovvancheri 2025; Hu et al. 2024). At the 4D pointcloud level, Cross4D-JEPA extends latent-space matching to 4D point clouds by distilling dense per-point targets from frozen 2D or video foundation models into a 4D point encoder (Nguyen et al. 2026).

A central challenge in JEPA-style learning is representation collapse. Earlier systems commonly stabilize training with a target encoder or teacher branch, often combined with stop-gradient and exponential moving average updates (Assran et al. 2023; Bardes et al. 2024). Recent work moves toward explicit embedding regularization. LeJEPA introduces Sketched Isotropic Gaussian Regularization (SI-GReg), which constrains embeddings toward an isotropic Gaussian distribution and reduces reliance on teacher-student encoders, stop-gradient operations, and related heuristics (Balestriero and LeCun 2025). LeWorldModel applies this direction to end-to-end latent world modeling from pixels using a predictive loss together with Gaussian latent regularization (Maes et al. 2026). This transition motivates a simpler JEPA-style formulation for 4D point cloud videos, where contextual features are used to predict target representations in latent space rather than reconstructing the input.

![](images/12b9da2fb0b4ec6adbe5399d4eb9d46e816b906c49a7959b1ecf89d3fd4525c9.jpg)  
Figure 1: Overview of the proposed pretraining framework. Point-tube tokens are split into visible context and masked targets; a predictor infers target latents while SIGReg regularizes the latent distribution.

## Method

The proposed pretraining framework learns from unlabeled 4D point cloud videos by predicting masked spatiotemporal token representations in latent space. Given an input clip $\mathbf { P } \in \mathbb { R } ^ { \pm _ { \times N \times 3 } }$ , the model first converts local space-time neighborhoods into point-tube tokens, hides a large subset of these tokens from the context encoder, and trains a predictor to infer their latent representations. In contrast to masked autoencoding methods that reconstruct point coordinates or auxiliary motion targets, the objective is defined on encoder features and is regularized by SIGReg to discourage collapse. Figure 1 summarizes the encoding pipeline, predictor, and training objective.

## Tokenization and Masking

Point cloud videos are irregular in both space and time, so masking individual raw points would not provide stable semantic units. We therefore adopt the point-tube tokenization used in P4Transformer and prior masked point-cloudvideo pretraining (Fan, Yang, and Kankanhalli 2021; Shen et al. 2023a; Zuo et al. 2025). Point-tube centers are sampled from P using Farthest Point Sampling (FPS); for each center ${ \hat { p } } _ { i } ,$ we define $T u b e _ { \hat { p } _ { i } } = \{ p \ \vert \ \bar { p } \in \mathbf { P } , D _ { s } ( p , \hat { p } _ { i } ) <$ $\begin{array} { r } { r _ { s } , D _ { t } ( p , \hat { p } _ { i } ) \leq \frac { r _ { t } } { 2 } \big \} } \end{array}$ . The tokenizer converts local spatiotemporal neighborhoods into a sequence of point-tube embeddings $\mathbf { X } \stackrel { - } { = } \{ e _ { i } \} _ { i = 1 } ^ { S }$ , which serve as the units for masking and latent prediction.

Each token is associated with a 4D center $\mathbf { Q } = \{ q _ { i } \} _ { i = 1 } ^ { S }$ indicating its spatial anchor and temporal position. These centers are used as positional information for the encoder and predictor, allowing the model to distinguish visible and masked point tubes by their locations. Since the prediction target is defined in latent space, the framework does not require the tokenizer to support raw-coordinate reconstruction.

Masking is performed after tokenization at the point-tubetoken level. A target sampler selects a target index set I from the token sequence. We use random target sampling with a high mask ratio of 75%, consistent with the observation in masked image and video modeling that high masking ratios reduce shortcut reconstruction and create a stronger prediction task (He et al. 2022; Feichtenhofer et al. 2022; Shen et al. 2023a). The context set is then formed by excluding the target indices,

$$
{ \mathcal { C } } = \{ 1 , . . . , S \} \setminus { \mathcal { T } } .\tag{1}
$$

In the pretraining configuration, the context encoder receives the remaining non-target tokens. Consequently, the model is trained to infer masked point-tube representations from the visible complement of the same 4D point cloud video.

## Architecture

The architecture consists of a P4D tokenizer, a Transformer encoder, and a lightweight predictor. This design follows the asymmetric spirit of masked modeling, where prediction is conditioned only on visible tokens, but replaces coordinate reconstruction with JEPA-style latent prediction (Assran et al. 2023; Bardes et al. 2024; Saito, Kudeshia, and Poovvancheri 2025). Compared with MAE-style point-cloudvideo pretraining such as MaST-Pre and Uni4D, the predictor is not a geometric decoder and does not output point coordinates, Chamfer targets, or hand-crafted motion descriptors (Shen et al. 2023a; Zuo et al. 2025).

Encoder. The encoder maps point-tube tokens and their 4D positional information to latent spatiotemporal representations. During pretraining, the same encoder is used in two passes. A full-sequence pass encodes all tokens and produces latents Z. A context pass encodes only the visible tokens $\mathbf { X } _ { \mathcal { C } }$ with their corresponding centers $\mathbf { Q } _ { \mathcal { C } }$ , producing context features $\mathbf { H } _ { \mathcal { C } }$ for the predictor. The SIGReg pretraining path therefore does not use a separate EMA teacher; the target representation and context representation are produced by the shared encoder.

Predictor. The predictor receives the context features, the context centers, and the masked target centers. Context features are first projected to a lower predictor dimension. For the target index set I, the predictor appends learned masktoken slots whose positional embeddings are computed from $\mathbf { Q } _ { \mathcal { I } }$ . The concatenated context tokens and target query slots are processed by the predictor network. Only the outputs at the target slots are kept and projected back to the encoder dimension:

$$
\hat { \mathbf { Y } } = h _ { \phi } \left( \mathbf { H } _ { \mathcal { C } } , \mathbf { Q } _ { \mathcal { C } } , \mathbf { Q } _ { \mathcal { T } } \right) .\tag{2}
$$

This formulation gives the predictor both the visible latent context and the 4D locations of the missing point tubes, while preventing it from observing the masked token features directly.

## Training Objective

The training objective combines masked latent prediction with an anti-collapse regularizer:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda \mathcal { L } _ { \mathrm { S I G R e g } } . } \end{array}\tag{3}
$$

Prediction loss. For the sampled target set I, the fullsequence encoder first produces latent representations $\mathbf { Z } ,$ and the prediction targets are formed by applying featurewise layer normalization to the selected masked latents, $\mathbf { Y } = \mathrm { L N } ( \mathbf { Z } _ { \mathcal { T } } )$ . The predictor outputs Yˆ from the visible context. The latent prediction loss is

$$
\mathcal { L } _ { \mathrm { p r e d } } = \mathcal { L } _ { \mathrm { s m o o t h - L 1 } } \left( \hat { \mathbf { Y } } , \mathbf { Y } \right) .\tag{4}
$$

Since both arguments of Eq. 4 are latent token representations, this objective encourages the model to capture predictable spatiotemporal semantics rather than recover the raw input points exactly.

SIGReg. Latent prediction alone can admit collapsed solutions in which the encoder produces low-variance features that are easy to predict. To stabilize training without a momentum teacher or stop-gradient target branch, we apply Sketched Isotropic Gaussian Regularization (SIGReg) to the full-sequence latents (Balestriero and LeCun 2025; Maes et al. 2026). Let $\mathbf { Z } \in \dot { \mathbb { R } } ^ { S \times B \times d }$ denote the full-sequence encoder representations. SIGReg draws random unit projection directions $u _ { j }$ and matches the one-dimensional projected distributions $\mathbf { Z } u _ { j }$ to an isotropic Gaussian reference using a sketched characteristic-function statistic:

$$
\mathcal { L } _ { \mathrm { S I G R e g } } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \mathcal { T } ( \mathbf { Z } u _ { j } ) ,\tag{5}
$$

where $\tau ( \cdot )$ denotes the Epps–Pulley-style statistic used by SIGReg. Its explicit finite-sample form and implementation details are provided in the supplementary material. The final loss in $\operatorname { E q . 3 }$ therefore balances two complementary pressures: the predictor must infer missing point-tube latents from visible 4D context, while the encoder is regularized to maintain a non-degenerate and approximately isotropic latent space. Figure 2 illustrates this projection-and-matching view of the regularizer.

## Experiments

We conduct experiments on MSRAction-3D (Li, Zhang, and Liu 2010) and NTU RGB+D (Shahroudy et al. 2016) for action recognition, and on SHREC’17 (Smedt et al. 2017) for cross-dataset gesture-recognition transfer. Implementation details, including the computing environment and hyperparameter settings, are provided in the supplementary material.

![](images/1060464f85710329bf0e513e28439ddca7b2ce803e137d4d9f8cd43f6f9ddf64.jpg)  
(a) Random projections of latent embeddings Z

![](images/d04b6c5ae6a76f0915a4ddf393c9359d037563c26d553e31caf37208f50c32c8.jpg)  
(b) Match each projection to a Gaussian reference  
Figure 2: Illustration of SIGReg. Random one-dimensional projections are drawn from the latent embedding distribution, and each projected distribution is matched to a Gaussian reference.

## Pretraining

We pretrain the proposed JEPA-style model on the training split defined by the cross-subject protocol of MSRAction-3D (Li, Zhang, and Liu 2010) and NTU RGB+D (Shahroudy et al. 2016). Following P4Transformer-based protocols used in prior point-cloud-video studies (Fan, Yang, and Kankanhalli 2021; Shen et al. 2023a; Han et al. 2024; Zuo et al. 2025), each pretraining clip contains 24 frames with 1024 sampled points per frame. The frame sampling stride is set to 1 for MSRAction-3D and 2 for NTU RGB+D, and random scaling is applied as data augmentation.

P4Transformer is used as the encoder, with 5 layers on MSRAction-3D and 10 layers on NTU RGB+D, matching common practice for these benchmarks (Fan, Yang, and Kankanhalli 2021; Shen et al. 2023a; Zuo et al. 2025). The pretraining task predicts masked point-tube representations from the visible context, using Smooth L1 prediction loss together with SIGReg. We optimize with AdamW and cosine learning-rate decay. The model is pretrained for 200 epochs on MSRAction-3D and for 100 epochs on NTU RGB+D.

## End-to-End Fine-Tuning

MSRAction-3D For MSRAction-3D, we initialize the P4Transformer encoder from the self-supervised checkpoint and replace the pretraining predictor with a supervised classification head. Following prior fine-tuning settings (Fan, Yang, and Kankanhalli 2021; Shen et al. 2023a), each clip contains 24 frames and 2048 points per frame. The model is trained end-to-end for 50 epochs with AdamW, cosine learning-rate decay, and cross-entropy loss.

Table 1 shows that self-supervised pretraining improves the P4Transformer backbone and gives the strongest result among the listed P4Transformer-based methods. This suggests that latent prediction with SIGReg provides useful spatiotemporal initialization even on the relatively small MSRAction-3D benchmark.

NTU RGB+D For NTU RGB+D, we follow the crosssubject protocol. Fine-tuning uses 24-frame clips and 2048 sampled points per frame. The model is fine-tuned for 20 epochs with AdamW, cosine learning-rate decay, and crossentropy loss.

<table><tr><td>Method</td><td>Acc.</td></tr><tr><td>Supervised Learning Only MeteorNet (Liu, Yan, and Bohg 2019)</td><td>88.50</td></tr><tr><td>PSTNet (Fan et al. 2021) PSTNet++ (Fan et al. 2022) Kinet (Zhong et al. 2022) PPTr (Wen et al. 2022) 3DInAction (Ben-Shabat, Shrout, and Gould 2024)</td><td>91.20 92.68 93.27 92.33 92.23</td></tr><tr><td>Mamba4D (Liu et al. 2025) P4Transformer (Fan, Yang, and Kankanhalli 2021)</td><td>92.68 90.94</td></tr><tr><td>With Self-Supervised Pretraining</td><td>91.29</td></tr><tr><td>P4Transformer + MaST-Pre (Shen et al. 2023a) P4Transformer + M2PSC (Han et al. 2024) P4Transformer + Uni4D (Zuo et al. 2025)</td><td>93.03 93.38</td></tr><tr><td>P4Transformer + DiMP (Zhang et al. 2026) P4Transformer + Ours</td><td>93.97 94.08</td></tr></table>

Table 1: Action recognition accuracy (%) on MSRAction-3D. Baselines are grouped by training setting.
<table><tr><td>Method</td><td>Full 50% Labels</td></tr><tr><td>Supervised Learning Only</td><td></td></tr><tr><td>3DV-Motion (Wang et al. 2020) 3DV-PointNet++ (Wang et al. 2020)</td><td>84.5 88.8</td></tr><tr><td>PSTNet (Fan et al. 2021) PSTNet++ (Fan et al. 2022) Kinet (Zhong et al. 2022)</td><td>90.5 91.4 92.3</td></tr><tr><td>P4Transformer (Fan, Yang, and Kankanhalli 2021)</td><td>90.2 81.2</td></tr><tr><td>With Self-Supervised Pretraining</td><td></td></tr><tr><td>P4Transformer + MaST-Pre (Shen et al. 2023a)</td><td>90.8 87.8</td></tr><tr><td>P4Transformer + M2PSC (Han et al. 2024)</td><td>91.3 88.7</td></tr><tr><td>P4Transformer + Uni4D (Zuo et al. 2025)</td><td>90.7 86.5</td></tr><tr><td>P4Transformer + Ours</td><td>91.8 89.0</td></tr></table>

Table 2: Action recognition accuracy (%) on NTU RGB+D under the cross-subject protocol. The semi-supervised setting fine-tunes with 50% labeled training videos.

The full-label column of Table 2 reports the end-to-end fine-tuning result. Our model improves over the supervised P4Transformer baseline and also outperforms the listed selfsupervised baselines. Moreover, this result is obtained with a 100-epoch pretraining schedule on NTU RGB+D, while MaST-Pre and M2PSC report 200 epochs (Shen et al. 2023a; Han et al. 2024).

## Semi-Supervised Learning

We further evaluate data eficiency on NTU RGB+D. Following the semi-supervised evaluation protocol used in prior point-cloud-video representation learning studies (Shen et al. 2023a; Han et al. 2024; Zuo et al. 2025), the encoder is first pretrained on the full unlabeled cross-subject training split, and supervised fine-tuning then uses only 50% of the labeled training videos. The test split and evaluation protocol remain unchanged, and all other settings follow the NTU end-to-end fine-tuning setup.

<table><tr><td>Initialization</td><td>1-shot</td><td>3-shot</td><td>5-shot</td></tr><tr><td>Scratch</td><td> $2 0 . 0 7 { \pm } 1 0 . 4 0$ </td><td>49.79±22.35</td><td> $7 5 . 8 9 { \pm } 1 0 . 3 2 $ </td></tr><tr><td>Pretrained</td><td> $\mathbf { 3 1 . 0 5 \pm 9 . 1 5 }$ </td><td> $\mathbf { 6 3 . 4 1 \pm 5 . 4 3 }$ </td><td> $\mathbf { 7 9 . 7 6 { \pm } 4 . 0 7 }$ </td></tr><tr><td>Gain</td><td>+10.98</td><td>+13.62</td><td>+3.87</td></tr></table>

Table 3: All-class few-shot P4Transformer video-level top-1 action recognition accuracy (%) on MSRAction-3D. Results are reported as mean ± standard deviation over ten paired support-set runs.

As shown under the 50% Labels setting in Table 2, pretraining gives a substantially larger improvement under limited labels than in the full-label setting. This indicates that the learned latent representation is especially useful when downstream supervision is scarce.

## Few-Shot Learning

Few-shot learning is evaluated on MSRAction-3D using the all-class protocol in Table 3. We initialize the encoder with weights from self-supervised pretraining on NTU RGB+D, append a lightweight classifier, and fine-tune the full model on MSRAction-3D. This setting difers from the n-way mshot protocol used by Uni4D (Zuo et al. 2025), where only a subset of action classes is sampled for each few-shot setting. In our evaluation, all action classes are retained, but “shot” still denotes the number of labeled training videos sampled per class, set to 1, 3, or 5. To reduce sensitivity to support-set sampling, we report the mean and standard deviation over ten runs. For each run, the scratch and pretrained models use the same support set, allowing a paired comparison under identical labeled examples.

The proposed pretraining improves the mean accuracy over training from scratch for all three shot budgets, with the largest gains in the 1-shot and 3-shot settings. Pretraining also substantially reduces the standard deviation in the 3-shot and 5-shot settings, indicating that the NTU RGB+D representation makes MSRAction-3D adaptation less dependent on a favorable support-set draw. The smaller 5-shot gain is expected because additional labeled examples reduce the relative value of initialization.

## Transfer Learning

We evaluate transfer learning by following prior cross-dataset protocols (Shen et al. 2023a; Zuo et al. 2025). The encoder is pretrained on NTU RGB+D and then fine-tuned on SHREC’17 (Smedt et al. 2017) under the 28-class setting, using a newly initialized classifier. This setting transfers from action recognition to hand-gesture recognition, testing both cross-dataset and cross-task generalization.

<table><tr><td>Method</td><td>Epochs Acc.</td></tr><tr><td>Supervised Learning Only</td><td></td></tr><tr><td>PointLSTM (Min et al. 2020)</td><td>87.6</td></tr><tr><td>PointLSTM-PSS (Min et al. 2020)</td><td>93.1</td></tr><tr><td>Kinet (Zhong et al. 2022)</td><td>95.2</td></tr><tr><td>P4Transformer (Fan, Yang, and</td><td>87.5</td></tr><tr><td>Kankanhalli 2021) P4Transformer (Fan, Yang, and</td><td>91.2</td></tr><tr><td>Kankanhalli 2021) With Self-Supervised Pretraining</td><td></td></tr><tr><td>P4Transformer + MaST-Pre (Shen</td><td>30</td></tr><tr><td>et al. 2023a) P4Transformer + MaST-Pre (Shen</td><td>50 92.4</td></tr><tr><td>et al. 2023a) P4Transformer + M2PSC (Han</td><td>30 90.9</td></tr><tr><td>et al. 2024) P4Transformer + M2PSC (Han</td><td>50 92.8</td></tr><tr><td>et al. 2024) P4Transformer + Uni4D (Zuo</td><td>50 93.8</td></tr><tr><td>et al. 2025) P4Transformer + Ours</td><td>30</td></tr><tr><td>P4Transformer + Ours</td><td>50</td></tr></table>

Table 4: Transfer learning accuracy (%) on SHREC’17.

As shown in Table 4, NTU RGB+D pretraining improves P4Transformer on SHREC’17 at both the 30- and 50-epoch evaluation points. The relative improvement over the corresponding P4Transformer baseline is larger with fewer finetuning epochs, suggesting that the learned representation provides a useful initialization for faster adaptation.

## Visualization

To qualitatively examine the representation learned by selfsupervised pretraining, we visualize point-cloud clips from MSRAction-3D together with their masked inputs and attention maps obtained from the MSR-pretrained encoder. The attention map is computed directly from the encoder selfattention weights. Specifically, we use the last encoder layer, average the attention assigned to each point-tube token over all heads and all query tokens, and obtain one scalar score for each tube token. This token-level score is then projected back to the point domain by assigning it to the neighboring points contained in the corresponding tube. In each panel of Figure 3, the first row shows the original point cloud sequence, the second row shows the masked point cloud, and the third row shows the corresponding encoder attention map.

The attention maps are concentrated on semantically relevant body parts for diferent actions. For high arm wave and side-boxing, the strongest responses appear around the arms; for forward punch, they move toward the punching hand; and for bend, they are concentrated near the head and upper body. Although this visualization is qualitative, the consistency between the attended regions and the action-defining motion cues suggests that MSR pretraining helps the encoder emphasize informative spatiotemporal regions rather than only low-level geometric structure.

<table><tr><td>Variant</td><td>Acc.</td></tr><tr><td>Full model</td><td>94.08</td></tr><tr><td>w/o SIGReg</td><td>89.90</td></tr><tr><td>Stop-gradient targets</td><td>79.44</td></tr><tr><td>w/o target-position query</td><td>93.03</td></tr><tr><td>EMA target encoder</td><td>88.50</td></tr></table>

Table 5: Component/path ablation on MSRAction-3D.

## Ablation Studies

We perform controlled ablations on MSRAction-3D to isolate the target path, target-position query, masking ratio, and SIGReg strength. All variants follow the MSRAction-3D pretraining and fine-tuning protocols described above.

Component ablations. Table 5 studies these training-path and predictor-design choices. Removing SIGReg lowers accuracy to 89.90%, and Figure 4 shows that this degradation is accompanied by early collapse of the target-latent standard deviation. The full model rapidly stabilizes near unit targetlatent standard deviation, whereas the EMA target encoder remains substantially lower for most of pretraining and also underperforms the shared-encoder SIGReg formulation. The stop-gradient target variant, defined by detaching the layernormalized target latents Y used in Eq. 4, has the largest downstream drop, suggesting that detaching target latents is harmful in this non-EMA training path. Removing the targetposition query causes a modest accuracy drop from 94.08% to 93.03%. Here, the ablated target slots use only the shared learned mask token, without the positional embedding computed from the masked centers Q<sub>I</sub>. This result indicates that visible context alone provides substantial predictive information on this benchmark, while explicit target-position queries still provide a measurable benefit.

Sensitivity ablations. Table 6 summarizes the remaining sensitivity to target masking and SIGReg strength. For masking, the 75% ratio gives the best accuracy among the tested values. A lower ratio provides a weaker prediction task because more context tokens remain visible, whereas an 85% ratio removes too much context for reliable target inference on MSRAction-3D. Since mask-ratio variants maintain nearunit target-latent standard deviation, Table 6 reports their accuracy only; for SIGReg-strength variants, Target Latent Std. denotes the target-latent standard deviation averaged over the final ten pretraining epochs. Among SIGReg weights, $\lambda _ { \mathrm { S I G R e g } } = 0 . 0 1$ performs best; a smaller weight yields lower target-latent standard deviation and weaker downstream accuracy, while a larger weight maintains high variance but can overemphasize distribution matching relative to predictive alignment.

## Conclusion

In this work, we studied JEPA-style self-supervised learning for 4D point cloud videos. The proposed framework converts point cloud sequences into spatiotemporal point-tube tokens, masks a large subset of these tokens, and trains a predictor to infer masked target representations from visible context representations in latent space. Instead of reconstructing raw point coordinates, the model learns through feature-space prediction, while SIGReg regularizes the learned latent distribution to reduce representation collapse in the proposed shared-encoder training path.

![](images/51a8ab6a82e4d128e183ac307bb37fbb41e07fe617cd1af92f8a17adec946b2b.jpg)  
Figure 3: Qualitative visualization of attention maps from the MSR-pretrained encoder. Each panel contains the original point cloud sequence, the masked point cloud sequence, and the encoder attention map from top to bottom.

![](images/0466cbb3b57317cd128f2d7e84c31e9cc00ed771fcf185396d74a0532edf7fb9.jpg)  
Figure 4: Target-latent standard deviation over pretraining epochs for representative component-ablation variants on MSRAction-3D.

We instantiate this formulation with a P4D tokenizer, Transformer encoder, and lightweight predictor, and evaluate the pretrained encoder across multiple downstream, transfer, qualitative, and ablation settings. The results show that latent prediction with SIGReg provides a strong initialization for 4D action and gesture recognition, especially under limited labeled data. This study remains limited to P4Transformerbased backbones and a modest benchmark scope, motivating future work on larger-scale pretraining, broader 4D datasets, and alternative architectures.

<table><tr><td>Setting</td><td>Value</td><td>Target Latent Std.</td><td>Acc.</td></tr><tr><td>Mask ratio</td><td>0.65</td><td></td><td>88.85</td></tr><tr><td>Mask ratio</td><td>0.75</td><td></td><td>94.08</td></tr><tr><td>Mask ratio</td><td>0.85</td><td></td><td>91.99</td></tr><tr><td>λSIGReg</td><td>0.001</td><td>0.870</td><td>86.76</td></tr><tr><td>λSIGReg</td><td>0.01</td><td>0.984</td><td>94.08</td></tr><tr><td>λSIGReg</td><td>0.1</td><td>0.992</td><td>89.55</td></tr></table>

Table 6: Sensitivity ablations on MSRAction-3D.

## References

Assran, M.; Duval, Q.; Misra, I.; Bojanowski, P.; Vincent, P.; Rabbat, M.; LeCun, Y.; and Ballas, N. 2023. Self-Supervised Learning from Images with a Joint-Embedding Predictive

Architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 15619–15629.

Balestriero, R.; and LeCun, Y. 2025. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.

Bardes, A.; Garrido, Q.; Ponce, J.; Chen, X.; Rabbat, M.; Le-Cun, Y.; Assran, M.; and Ballas, N. 2024. Revisiting Feature Prediction for Learning Visual Representations from Video. Transactions on Machine Learning Research. Featured Certification.

Ben-Shabat, Y.; Shrout, O.; and Gould, S. 2024. 3DInAction: Understanding Human Actions in 3D Point Clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19978–19987.

Fan, H.; Yang, Y.; and Kankanhalli, M. 2021. Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 14199–14208.

Fan, H.; Yang, Y.; and Kankanhalli, M. 2023. Point Spatio-Temporal Transformer Networks for Point Cloud Video Modeling. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2): 2181–2192.

Fan, H.; Yu, X.; Ding, Y.; Yang, Y.; and Kankanhalli, M. 2021. PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences. In International Conference on Learning Representations.

Fan, H.; Yu, X.; Yang, Y.; and Kankanhalli, M. 2022. Deep Hierarchical Representation of Point Cloud Videos via Spatio-Temporal Decomposition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12): 9918–9930.

Feichtenhofer, C.; Fan, H.; Li, Y.; and He, K. 2022. Masked Autoencoders As Spatiotemporal Learners. In Koyejo, S.;

Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems, volume 35, 35946–35958. Curran Associates, Inc.

Han, Y.; Xu, C.; Xu, R.; Qian, J.; and Xie, J. 2024. Masked Motion Prediction with Semantic Contrast for Point Cloud Sequence Learning. In Computer Vision – ECCV 2024: 18th European Conference, Milan, Italy, September 29–October 4, 2024, Proceedings, Part LXXVI, 414–431. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3-031-73115-0.

He, K.; Chen, X.; Xie, S.; Li, Y.; Dollár, P.; and Girshick, R. 2022. Masked Autoencoders Are Scalable Vision Learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16000–16009.

Hu, N.; Cheng, H.; Xie, Y.; Li, S.; and Zhu, J. 2024. 3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning. arXiv:2409.15803.

Li, P.; Wang, Z.; Yuan, Y.; Liu, H.; Meng, X.; Yuan, J.; and Liu, M. 2025. UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision (ICCV), 6738–6747.

Li, W.; Jiang, X.; Zhang, G.; Zhang, X.; Shao, L.; and Lu, S. 2026. STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding. arXiv:2604.11637.

Li, W.; Zhang, Z.; and Liu, Z. 2010. Action Recognition Based on a Bag of 3D Points. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, 9–14.

Liu, J.; Han, J.; Liu, L.; Aviles-Rivero, A. I.; Jiang, C.; Liu, Z.; and Wang, H. 2025. Mamba4D: Eficient 4D Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17626–17636.

Liu, X.; Yan, M.; and Bohg, J. 2019. MeteorNet: Deep Learning on Dynamic 3D Point Cloud Sequences . In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 9245–9254. Los Alamitos, CA, USA: IEEE Computer Society.

Liu, Y.; Chen, J.; Zhang, Z.; Huang, J.; and Yi, L. 2023. LeaF: Learning Frames for 4D Point Cloud Sequence Understanding. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 604–613.

Maes, L.; Lidec, Q. L.; Scieur, D.; LeCun, Y.; and Balestriero, R. 2026. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.

Min, Y.; Zhang, Y.; Chai, X.; and Chen, X. 2020. An Eficient PointLSTM for Point Clouds Based Gesture Recognition. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5760–5769.

Nguyen, T. T.; Nguyen-Truong, H.; Vo, T.; Truong, H. M.; and Vu, T.-A. 2026. Cross4D-JEPA: Dense Cross-modal Correspondence Distillation for 4D Point Cloud Representation Learning. arXiv:2607.00514.

Pang, Y.; Wang, W.; Tay, F. E. H.; Liu, W.; Tian, Y.; and Yuan, L. 2022. Masked Autoencoders for Point Cloud Selfsupervised Learning. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part II, 604–621. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3-031-20085-4.

Saito, A.; Kudeshia, P.; and Poovvancheri, J. 2025. Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 7348–7357.

Shahroudy, A.; Liu, J.; Ng, T.-T.; and Wang, G. 2016. NTU RGB+D: A large scale dataset for 3D human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, 1010–1019.

Shen, Z.; Sheng, X.; Fan, H.; Wang, L.; Guo, Y.; Liu, Q.; Wen, H.; and Zhou, X. 2023a. Masked Spatio-Temporal Structure Prediction for Self-supervised Learning on Point Cloud Videos. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 16534–16543.

Shen, Z.; Sheng, X.; Wang, L.; Guo, Y.; Liu, Q.; and Zhou, X. 2023b. PointCMP: Contrastive Mask Prediction for Self-Supervised Learning on Point Cloud Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1212–1222.

Sheng, X.; Shen, Z.; and Xiao, G. 2023. Contrastive Predictive Autoencoders for Dynamic Point Cloud Self-Supervised Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 37(8): 9802–9810.

Sheng, X.; Shen, Z.; Xiao, G.; Wang, L.; Guo, Y.; and Fan, H. 2023. Point Contrastive Prediction with Semantic Clustering for Self-Supervised Learning on Point Cloud Videos. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 16469–16478.

Smedt, Q. D.; Wannous, H.; Vandeborre, J.-P.; Guerry, J.; Saux, B. L.; and Filliat, D. 2017. 3D Hand Gesture Recognition Using a Depth and Skeletal Dataset. In Pratikakis, I.; Dupont, F.; and Ovsjanikov, M., eds., Eurographics Workshop on 3D Object Retrieval. The Eurographics Association. ISBN 978-3-03868-030-7.

Wang, H.; Yang, L.; Rong, X.; Feng, J.; and Tian, Y. 2021. Self-supervised 4D Spatio-temporal Feature Learning via Order Prediction of Sequential Point Cloud Clips. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 3761–3770.

Wang, Y.; Xiao, Y.; Xiong, F.; Jiang, W.; Cao, Z.; Zhou, J. T.; and Yuan, J. 2020. 3DV: 3D Dynamic Voxel for Action Recognition in Depth Video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 508–517.

Wen, H.; Liu, Y.; Huang, J.; Duan, B.; and Yi, L. 2022. Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIX, 19–35. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3-031-19817-5.

Wu, Q.; Lan, J.; Kang, W.; Wang, Z.; and Hu, K. 2026. SRENet: Spectral Re-Entry Network for Point Cloud Action Recognition. IEEE Transactions on Circuits and Systemsfor Video Technology, 1–1.

Xie, S.; Gu, J.; Guo, D.; Qi, C. R.; Guibas, L.; and Litany, O. 2020. PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III, 574–591. Berlin, Heidelberg: Springer-Verlag. ISBN 978-3-030-58579-2.

Yu, X.; Tang, L.; Rao, Y.; Huang, T.; Zhou, J.; and Lu, J. 2022. Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling . In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19291– 19300. Los Alamitos, CA, USA: IEEE Computer Society.

Zhang, Z.; Dong, Y.; Liu, Y.; and Yi, L. 2023. Completeto-Partial 4D Distillation for Self-Supervised Point Cloud Sequence Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 17661–17670.

Zhang, Z.; Zhu, J.; Fang, C.; Liu, J.; and Mian, A. S. 2026. Difusion Masked Pretraining for Dynamic Point Cloud. arXiv:2605.03639.

Zhong, J.-X.; Zhou, K.; Hu, Q.; Wang, B.; Trigoni, N.; and Markham, A. 2022. No Pain, Big Gain: Classify Dynamic Point Cloud Sequences With Static Models by Fit-

ting Feature-Level Space-Time Surfaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8510–8520.

Zuo, Z.; Zhuang, C.; Gao, P.; Qin, J.; Feng, H.; and Sebe, N. 2025. Uni4D: A Unified Self-Supervised Learning Framework for Point Cloud Videos. arXiv:2504.04837.

# Joint-Embedding Prediction of Masked Point Tubes for Self-Supervised Learning on 4D Point Cloud Videos

Supplementary Material

## Overview

This supplementary document records implementation details and additional diagnostics that support the main paper. It first gives the finite-sample SIGReg objective used in pretraining, including the latent tensor shape, random projections, characteristic-function statistic, and gradient path. It then reports the experimental environment, data protocols, training settings, random seeds, and evaluation procedure. The final part provides ablation diagnostics and the per-seed few-shot results summarized in the main paper.

## SIGReg Details

## Regularized Representation

Let $\mathbf { Z } \in \mathbb { R } ^ { S \times B \times d }$ denote the full-sequence encoder latents used by SIGReg, where S is the number of point-tube tokens, B is the minibatch size, and d is the encoder dimension. In the implementation, these latents are the last hidden states of the full-sequence encoder pass and are supplied to the regularizer with shape (S, B, d).

## Random Projections

At each forward pass, SIGReg samples a projection matrix $\mathbf { A } = [ u _ { 1 } , \ldots , u _ { J } ] \in \mathbb { R } ^ { d \times J }$ from a standard normal distribution and normalizes each column to unit Euclidean norm. For token position i, batch element $b ,$ and projection j, define the scalar projection

$$
\begin{array} { r } { r _ { i , b , j } = \mathbf { z } _ { i , b } ^ { \top } u _ { j } . } \end{array}\tag{6}
$$

The implementation uses $J \ = \ 1 0 2 4$ random projections. These directions are resampled at every forward pass and are not trainable parameters.

## Characteristic-Function Matching

The one-dimensional projected empirical distribution is matched to the standard Gaussian characteristic function through a truncated Epps–Pulley discrepancy. For token position i and projection j, let $\begin{array} { r } { \hat { \psi } _ { i , j } ( t ) = B ^ { - 1 } \sum _ { b = 1 } ^ { B } \exp \left( \mathrm { i } t r _ { i , b , j } \right) } \end{array}$ denote the empirical characteristic function. The continuousform quantity is

$$
\mathcal { D } _ { i , j } = B \int _ { - 3 } ^ { 3 } e ^ { - t ^ { 2 } / 2 } \left| \hat { \psi } _ { i , j } ( t ) - e ^ { - t ^ { 2 } / 2 } \right| ^ { 2 } d t .\tag{7}
$$

The implementation approximates this integral on a fixed frequency grid. With $\bar { K } = 1 7$ knots, define

$$
\tau _ { k } = \frac { 3 ( k - 1 ) } { K - 1 } , \qquad \phi _ { k } = \exp \left( - \frac { \tau _ { k } ^ { 2 } } { 2 } \right) ,\tag{8}
$$

for $k = 1 , \ldots , K$ . Let $\Delta = 3 / ( K - 1 )$ . Since the integrand is even in the frequency variable, we evaluate only the nonnegative half of [−3, 3] and fold the negative-frequency

contribution into the quadrature weights. The implementation uses the following quadrature coeficients:

$$
\omega _ { k } = { \left\{ \begin{array} { l l } { \Delta , } & { k = 1 { \mathrm { ~ o r ~ } } k = K , } \\ { 2 \Delta , } & { 1 < k < K . } \end{array} \right. }\tag{9}
$$

For each token position and projection, the empirical real and imaginary parts of the characteristic function are

$$
\hat { c } _ { i , j , k } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \cos ( \tau _ { k } r _ { i , b , j } ) ,\tag{10}
$$

$$
\hat { s } _ { i , j , k } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sin ( \tau _ { k } r _ { i , b , j } ) .
$$

Since a standard Gaussian has characteristic function $\phi _ { k }$ and zero imaginary component, the per-token, per-projection statistic is

$$
\mathcal { T } _ { i , j } = B \sum _ { k = 1 } ^ { K } \omega _ { k } \phi _ { k } \left[ \left( \hat { c } _ { i , j , k } - \phi _ { k } \right) ^ { 2 } + \hat { s } _ { i , j , k } ^ { 2 } \right] .\tag{11}
$$

The final SIGReg term first averages token-level statistics within each random projection and then averages over projections:

$$
\mathcal { L } _ { \mathrm { S I G R e g } } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \left( \frac { 1 } { S } \sum _ { i = 1 } ^ { S } \mathcal { T } _ { i , j } \right) .\tag{12}
$$

The multiplicative factor $B$ follows the finite-sample Epps– Pulley statistic used by SIGReg and is retained in the implementation. Additionally, the fixed frequency grid $\tau _ { k } ,$ Gaussian reference values $\phi _ { k }$ , and combined weights $\omega _ { k } \phi _ { k }$ are precomputed and stored as non-trainable bufers.

## Training Objective

During pretraining, the model first performs a full-sequence encoder pass. Masked prediction targets are formed by applying feature-wise layer normalization to the selected target latents. The SIGReg loss is computed separately on the full set of last hidden states from the same full-sequence pass. The optimization objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { \mathrm { S I G R e g } } \mathcal { L } _ { \mathrm { S I G R e g } } , } \end{array}\tag{13}
$$

where $\lambda _ { \mathrm { S I G R e g } }$ is the configured regularization weight. The default training path does not detach the prediction targets: gradients from both $\mathcal { L } _ { \mathrm { p r e d } }$ and $\mathcal { L } _ { \mathrm { S I G R e g } }$ can update the shared encoder. The stop-gradient and EMA variants reported in the main paper are ablations of this default path.

## Implementation Details

## Infrastructure

All experiments were executed on a workstation using one CUDA device per training run. The computing infrastructure and software environment were as follows:

• OS: Rocky Linux 9.8.

• CPU: 2× Intel Xeon Gold 5420+.

• System memory: 384 GiB.

• GPU: 4× NVIDIA RTX 6000 Ada Generation.

• GPU driver: NVIDIA 580.159.04.

• Python and GPU stack: Python 3.10.19; PyTorch2.6.0+cu124; PyTorch CUDA build 12.4; cuDNN 9.1.0.

• Scientific libraries: NumPy 2.2.5; SciPy 1.15.3; scikitlearn 1.7.2.

• Utility libraries: torchvision 0.21.0+cu124; tqdm 4.65.2.

## Data Protocols

Table 7 summarizes the dataset-level class counts, videolevel split sizes, and cross-subject split protocols. Whenever self-supervised pretraining is performed on a dataset, the listed training videos are used without labels. These datasets are selected to enable direct comparison with prior pointcloud-video methods, and the split protocols match the baselines reported in the main paper. Input clips are extracted from each video with overlapping sliding windows. Trainingtime point sampling is stochastic. MSRAction-3D and NTU RGB+D apply random axis-wise scaling in [0.9, 1.1], while SHREC’17 applies the same scaling after clip normalization.

<table><tr><td>Dataset</td><td>Classes</td><td>Train/Test</td><td>Protocol</td></tr><tr><td>MSRAction-3D</td><td>20</td><td>270 /297</td><td>S1-S5 / S6-S10</td></tr><tr><td>NTU RGB+D</td><td>60</td><td>40,320 / 16,560 Official</td><td></td></tr><tr><td>SHREC&#x27;17</td><td>28</td><td>1,960 / 840</td><td>Official</td></tr></table>

Table 7: Dataset statistics and cross-subject split protocols used in the experiments. Counts are video-level sequence counts under the corresponding protocol.

The NTU RGB+D semi-supervised setting uses a classbalanced 50% subset of the cross-subject training set, comprising 20,160 of the 40,320 training videos, with 336 videos per class. The MSRAction-3D few-shot setting retains all 20 action classes and samples 1, 3, or 5 labeled videos per class. In both settings, the sampled subsets are fixed throughout fine-tuning; for MSRAction-3D, each support set is also shared between the paired scratch and pretrained runs for the corresponding random seed.

## Training and Architecture Configurations

Tables 8 and 9 report the protocol-specific optimization, point-tube sampling, and architecture configurations used for the experiments in the main paper. Across protocols, we use AdamW with a 10-epoch linear warmup followed by cosine decay to a minimum learning rate of 10−6, and clip gradients to a global norm of 1.0. The training objective is Smooth L1 loss with β = 2.0 for masked prediction during self-supervised pretraining and cross-entropy loss during supervised fine-tuning. During pretraining, we mask 75% of point-tube tokens, use a temporal kernel size of 3 with temporal stride 2, and apply SIGReg with 17 knots and 1024 random projections when the regularizer is enabled.

## Random Seeds

Standard and ablation training runs use seed 42, whereas few-shot MSRAction-3D training runs use seeds 0–9. The seed setup used by each training entry point is:

seed = args.seed   
random.seed(seed)   
np.random.seed(seed)   
torch.manual\_seed(seed)   
torch.cuda.manual\_seed\_all(seed)

This controls the pseudorandom streams used for initialization, data sampling, augmentation, masking, and SIGReg projections; deterministic CUDA kernels are not forced.

## Ablation Protocols

For controlled MSRAction-3D ablations, all settings in Tables 8 and 9 are kept fixed except the studied factor: target detachment, target-position query, mask ratio, SIGReg weight, or an EMA target encoder with momentum increasing from 0.996 to 0.9998. The reported mask-ratio range is $m \in \{ 0 . 6 5 , 0 . 7 5 , 0 . 8 5 \}$ and the reported SIGReg-weight range is λ<sub>SIGReg</sub> ∈ {0.001, 0.01, 0.1} for the MSRAction-3D ablations. These are targeted development ablations rather than an exhaustive hyperparameter grid.

## Evaluation

During evaluation, we process all clips extracted from each test video using the sliding-window protocol. The reported video-level top-1 accuracy is computed by summing the softmax probabilities over clips from the same video and assigning the video to the class with the largest accumulated probability. We use this metric because all evaluated benchmarks are single-label action or gesture classification tasks and because video-level top-1 accuracy is the standard metric used by the compared methods. For pretrained models, we initialize fine-tuning from the final self-supervised pretraining checkpoint. Following the protocol used in prior work, we compute video-level top-1 accuracy on the test split after each fine-tuning epoch and report the best result over fine-tuning epochs.

## Additional Results

## Prediction-Loss Diagnostics for Ablations

Tables 10 and 11 complement the corresponding component and sensitivity ablations in the main paper. For each MSRAction-3D variant, they report downstream video-level accuracy together with two late-pretraining diagnostics. Prediction loss is the Smooth L1 objective between the predicted masked latents Yˆ and the layer-normalized targets Y, as defined in the main paper. Target Std. is included as a collapse diagnostic and denotes the mean per-feature standard deviation of Y after flattening its non-feature dimensions. Both diagnostics are averaged over minibatches and the final ten pretraining epochs.

These diagnostics indicate that prediction loss alone is not a reliable model selection criterion. Without SIGReg, the model obtains a near-zero logged prediction loss, but the target-latent standard deviation collapses and downstream accuracy decreases. The EMA target-encoder variant also has a small prediction loss, but its target variance remains substantially lower than that of the shared-encoder SIGReg formulation. In contrast, stop-gradient targets retain near-unit target variance, but make masked-target prediction harder and produce the lowest downstream accuracy. The full model therefore performs best not because it minimizes prediction loss, but because it maintains a non-collapsed latent space while preserving an informative predictive objective.

<table><tr><td>Protocol</td><td>Epochs</td><td>Batch</td><td>LR</td><td>WD</td><td>Frames</td><td>Pts.</td><td>Clip Str.</td><td>Frame Str.</td><td> $r _ { s }$ </td><td>Sp. Str.</td><td>Nbrs.</td><td>SIGReg</td></tr><tr><td>MSR pretraining</td><td>200</td><td>96</td><td>3e-4</td><td>5e-2</td><td>24</td><td>1024</td><td>1</td><td>1</td><td>0.3</td><td>32</td><td>32</td><td>0.01</td></tr><tr><td>NTU pretraining</td><td>100</td><td>56</td><td>3e-4</td><td>5e-2</td><td>24</td><td>1024</td><td>2</td><td>2</td><td>0.1</td><td>32</td><td>32</td><td>0.03</td></tr><tr><td>MSR full fine-tuning</td><td>50</td><td>48</td><td>5e-4</td><td>1e-4</td><td>24</td><td>2048</td><td>1</td><td>1</td><td>0.7</td><td>32</td><td>32</td><td></td></tr><tr><td>NTU full/semi fine-tuning</td><td>20</td><td>48</td><td>5e-4</td><td>5e-2</td><td>24</td><td>2048</td><td>2</td><td>2</td><td>0.1</td><td>32</td><td>32</td><td></td></tr><tr><td>SHREC transfer fine-tuning</td><td>50</td><td>24</td><td>1e-3</td><td>1e-4</td><td>16</td><td>256</td><td>1</td><td>1</td><td>0.3</td><td>16</td><td>9</td><td></td></tr><tr><td>MSR few-shot fine-tuning</td><td>50</td><td>48</td><td>5e-4</td><td>1e-4</td><td>24</td><td>2048</td><td>1</td><td>1</td><td>0.7</td><td>32</td><td>32</td><td></td></tr></table>

Table 8: Final optimization and point-tube sampling settings. LR and WD denote learning rate and weight decay; Frames denotes frames per input clip; Pts. denotes sampled points per frame; Clip Str. and Frame Str. denote the temporal strides between successive clips and successive frames within a clip, respectively. Sp. Str. and Nbrs. denote P4D spatial stride and neighbor samples. SIGReg weights apply only to self-supervised pretraining.

<table><tr><td>Setting</td><td>MSR NTU</td><td>SHREC</td><td>Few-shot</td></tr><tr><td>Enc. dim</td><td>384</td><td>384</td><td>384 384</td></tr><tr><td>Enc. heads</td><td>8</td><td>8 8</td><td>8</td></tr><tr><td>Enc. depth</td><td>5 10</td><td>10</td><td>10</td></tr><tr><td>Enc. MLP</td><td>4</td><td>4 4</td><td>4</td></tr><tr><td>Pred. dim</td><td>192</td><td>192 一</td><td>一</td></tr><tr><td>Pred. heads</td><td>6</td><td>6 一</td><td></td></tr><tr><td>Pred. depth</td><td>3 6</td><td>一</td><td>一</td></tr><tr><td>Pred. MLP</td><td>4</td><td>4</td><td></td></tr><tr><td>Cls. hidden</td><td>48</td><td>128 128</td><td>16</td></tr><tr><td>Cls. dropout</td><td>0.5</td><td>0.5 0.5</td><td>0.5</td></tr></table>

Table 9: Architecture settings. Enc., Pred., and Cls. denote encoder, predictor, and classifier, respectively; dim, heads, depth, MLP, hidden, and dropout report the corresponding architectural hyperparameters. The NTU semi-supervised experiments use the same architecture as the full-data NTU experiments.
<table><tr><td>Variant</td><td>Pred. Loss Target Std.</td><td>Acc.</td></tr><tr><td>Full model  $( m = 0 . 7 5 ,$ </td><td>0.001219</td><td>0.984 94.08</td></tr><tr><td> $\lambda _ { \mathrm { S I G R e g } } = 0 . 0 1 )$ </td><td></td><td></td></tr><tr><td>w/o SIGReg Stop-gradient targets</td><td>0.000000 0.095069</td><td>0.008 89.90 0.993 79.44</td></tr><tr><td>w/o target-position query</td><td>0.002563</td><td>0.942 93.03</td></tr><tr><td>EMA target encoder</td><td>0.000452</td><td>0.38888.50</td></tr></table>

Table 10: Prediction-loss diagnostics for component/path ablations on MSRAction-3D.

The sensitivity ablations reveal distinct trade-ofs. A low mask ratio provides a weaker prediction task, whereas a high ratio removes too much visible context; moderate masking therefore performs best, while target variance remains stable across settings. For SIGReg, weak regularization is insuficient to maintain target variance, whereas strong regularization raises prediction loss, indicating a trade-of with predictive alignment. An intermediate weight provides the best downstream balance.

<table><tr><td>Setting</td><td></td><td>Value Pred. Loss Target Std.</td><td>Acc.</td></tr><tr><td>Mask ratio m</td><td>0.65</td><td>0.001050</td><td>0.984 88.85</td></tr><tr><td>Mask ratio m</td><td>0.75</td><td>0.001219</td><td>0.984 94.08</td></tr><tr><td>Mask ratio m</td><td>0.85</td><td>0.001401</td><td>0.985 91.99</td></tr><tr><td>λSIGReg</td><td>0.001</td><td>0.000179</td><td>0.870 86.76</td></tr><tr><td>λSIGReg</td><td>0.01</td><td>0.001219</td><td>0.98494.08</td></tr><tr><td>λSIGReg</td><td>0.1</td><td>0.010965</td><td>0.992 89.55</td></tr></table>

Table 11: Prediction-loss diagnostics for sensitivity ablations on MSRAction-3D.

## Few-Shot Multi-Run Results

Table 12 reports the per-run MSRAction-3D few-shot results used to summarize robustness across support-set samples. Scratch denotes training without a pretrained checkpoint, whereas Pretrained uses the self-supervised checkpoint for initialization. Each seed uses the same labeled support set for both initializations, so the two columns form paired comparisons.

<table><tr><td rowspan="2">Seed</td><td colspan="2">1-shot</td><td colspan="2">3-shot</td><td colspan="2">5-shot</td></tr><tr><td>Scratch</td><td>Pretrained</td><td>Scratch</td><td>Pretrained</td><td>Scratch</td><td>Pretrained</td></tr><tr><td>0</td><td>35.889</td><td>33.798</td><td>62.718</td><td>62.718</td><td>79.094</td><td>77.003</td></tr><tr><td>1</td><td>19.861</td><td>27.526</td><td>24.739</td><td>65.505</td><td>77.700</td><td>72.474</td></tr><tr><td>2</td><td>30.662</td><td>38.328</td><td>57.840</td><td>60.279</td><td>84.669</td><td>83.275</td></tr><tr><td>3</td><td>6.969</td><td>36.585</td><td>54.355</td><td>58.537</td><td>80.139</td><td>81.882</td></tr><tr><td>4</td><td>13.589</td><td>37.631</td><td>40.767</td><td>64.460</td><td>82.927</td><td>81.185</td></tr><tr><td>5</td><td>5.923</td><td>13.589</td><td>43.206</td><td>57.143</td><td>66.899</td><td>74.913</td></tr><tr><td>6</td><td>21.603</td><td>29.268</td><td>79.094</td><td>75.261</td><td>79.094</td><td>86.063</td></tr><tr><td>7</td><td>31.707</td><td>30.314</td><td>74.564</td><td>67.596</td><td>77.352</td><td>80.836</td></tr><tr><td>8</td><td>12.544</td><td>43.902</td><td>5.226</td><td>64.460</td><td>49.826</td><td>81.533</td></tr><tr><td>9</td><td>21.951</td><td>19.512</td><td>55.401</td><td>58.188</td><td>81.185</td><td>78.397</td></tr><tr><td>Mean ± Std.</td><td>20.070±10.404</td><td>31.045±9.152</td><td>49.791±22.348</td><td>63.415±5.428</td><td>75.889±10.321</td><td>79.756±4.068</td></tr></table>

Table 12: Per-seed all-class few-shot P4Transformer video-level top-1 accuracy (%) on MSRAction-3D. Means are reported with standard deviations over ten run seeds.