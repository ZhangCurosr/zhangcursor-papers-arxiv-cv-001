# Context-Predictive Low-Rank Adaptation for Face Presentation Attack Detection

Aminu Yusuf, Shakeel Muhammad Ibrahim, Sulaimon Oyeniyi Adebayo, Muhammad Sohail Ibrahim, Muhammad Shahid Jabbar, and Shujaat Khan

Abstract—Face presentation attack detection requires representations that distinguish physical spoofing cues across changes in acquisition devices, subjects, and attack media. Large visual encoders offer transferable representations, but conventional binary fine-tuning supervises their final decisions without explicitly constraining how local facial evidence is represented under partial observation. We present CP-JEPA, a contextpredictive adaptation framework that combines low-rank tuning of a pretrained CLIP image encoder with a joint-embedding predictive objective. A student predicts complete-image teacher features at masked spatial locations, while unmasked classification, masked classification, and prediction consistency retain the connection to the live/spoof task. An exponential-movingaverage teacher supplies evolving latent targets. The auxiliary components are used only during training; inference requires only the LoRA-adapted CLIP visual encoder and binary classifier. We evaluate CP-JEPA under single-source and multi-source OCIM transfer, additional CelebA-Spoof training, transfer to CelebA-Spoof, two-source training, and unseen-attack evaluation on SiW-Mv2. On the standard OCIM benchmark, CP-JEPA achieves 3.74% mean HTER and 98.73% mean AUC. Single-source evaluation achieves 11.36% mean HTER across twelve source–target combinations. With CelebA-Spoof added to the three OCIM training sources, CP-JEPA reduces mean HTER from 3.45% to 2.35% and increases mean AUC from 98.68% to 99.32%. On SiW-Mv2, it achieves 3.36% mean HTER across the held-out attack categories. Controlled loss ablations examine the contributions of latent prediction, masked classification, and prediction consistency. The results show that context-predictive adaptation can improve cross-dataset PAD performance, although the gains vary with the training sources and evaluation setting. The implementation is available at the https://github.com/aminuyusufmr/cp\_jepa repository.

Index Terms—Face presentation attack detection, face antispoofing, joint-embedding predictive architecture, low-rank adaptation, vision foundation models, cross-dataset transfer.

## I. INTRODUCTION

ACE recognition is increasingly deployed through com-(PAD) an essential component of biometric authentication.

Printed photographs and replayed videos can present the appearance of a legitimate user without the physical presence of that user. A PAD system must therefore distinguish a genuine face presentation from an attack, even when both depict the same identity. The difficulty increases when deployment changes the sensor, illumination, display, printing process, or capture environment. A cue that separates classes within one dataset can reflect its acquisition process as much as the underlying presentation attack.

Research has addressed this challenge through task-specific local operators, domain-aware representation learning, multimodal fusion, and large pretrained models. Central-difference networks emphasize local intensity structure [1]; domaingeneralization methods constrain the geometry of bona fide and attack representations [2–4]. Recent studies extend the design space through decomposition–fusion representations [5], domain-adversarial cross-attention [6], and multimodal bidirectional fusion [7]. Pretrained visual encoders such as CLIP can also be adapted for PAD using low-rank updates [8].

Binary classification during low-rank adaptation encourages features that distinguish bona fide presentations from attacks but does not explicitly supervise the prediction of masked spatial representations. Masked feature prediction uses visible context to estimate teacher representations at masked locations. This distinction motivates combining global classification with local feature prediction, without requiring pixel reconstruction, depth annotations, physiological measurements, or an additional inference network.

We develop CP-JEPA to combine supervised CLIP adaptation with masked feature prediction. After an initial crossentropy stage, the student learns to predict spatial features from an EMA teacher that receives the complete image. The student retains its native classification pathway, with additional losses for masked classification and consistency between masked and unmasked predictions. We adapt the joint-embedding predictive principle [9] to low-rank PAD training and examine the contribution of each auxiliary objective separately.

The contributions are as follows:

• We formulate a context-predictive low-rank adaptation framework that couples native CLIP classification with dense EMA-teacher feature prediction under input masking.

• We incorporate masked classification and prediction consistency alongside latent prediction during supervised CLIP-LoRA adaptation. The teacher and predictor are used only during training, leaving the student classification architecture unchanged at inference.

• We evaluate CP-JEPA on cross-dataset and unseenattack protocols, including single-source and multi-source OCIM transfer, additional CelebA-Spoof training, transfer to CelebA-Spoof, two-source training, and SiW-Mv2. In the additional-source setting, mean HTER improves by 1.10 percentage points and mean AUC by 0.64 points over the CLIP-LoRA baseline.

## II. RELATED WORK

## A. Representation Learning for Cross-Dataset PAD

Early deep PAD models combined binary classification with local or auxiliary supervision. Central-difference convolution emphasizes fine spatial variations through intensity and gradient information [1]. Domain-generalization methods subsequently examined the relationship between class discrimination and domain variation. Single-side domain generalization encourages compact bona fide representations while accommodating heterogeneous attacks [2]. SA-FAS considers class separability and domain alignment jointly [3], while instanceaware domain generalization addresses sample-dependent variability [4]. DRDN combines residual distillation of semantic and low-frequency illumination information with attentionbased contrastive learning [10].

Recent works further diversify the available supervision. DFNet decomposes and fuses facial representations to support generalized and interpretable PAD [5]. GR-DXNet combines multimodal cross-attention with adversarial domain suppression [6]. FAS-MBCA uses multiple paths and bidirectional cross-attention for multimodal PAD [7]. PCGattnNet uses dynamic graph attention on smartphone-acquired 3D facial point clouds to detect presentation attacks, with evaluation across unseen attack instruments and acquisition devices [11]. Modality- and domain-aware adaptivity extends vision–language representations to generalized multimodal detection [12]. These methods address rich sensing and fusion settings.

## B. Foundation Models and Parameter-Efficient Adaptation

CLIP pretraining associates visual representations with language across diverse image content [13]. Its visual encoder can subsequently support a supervised PAD head even when the downstream system does not use text prompts. Low-rank adaptation (LoRA) restricts trainable weight changes to a factored subspace while retaining the pretrained matrices [14]. FoundPAD demonstrates the relevance of this strategy to presentation attack detection [8]. Accordingly, CLIP with LoRA and binary cross-entropy serves as the direct baseline in our experiments.

Language-guided PAD methods exploit a different part of the pretrained model. FLIP grounds visual representations using language [15]. ME-FAS enhances multimodal textual information for cross-domain PAD [16]. CLIP-SA uses textual semantics to guide feature alignment [17]; MoGA-ETA develops enhanced text guidance and alignment [18]. DGPDL models domain signals through a learnable prompt distribution [19]. These papers establish that pretrained semantic structure can be adapted through text, prompt construction, or visual updates.

## C. Masked Modeling and Joint-Embedding Prediction

Masked autoencoders learn representations by reconstructing withheld image content [20]. Bootstrap-style learning uses an online network and an EMA target network to establish representation consistency [21]. I-JEPA predicts target-region representations from visible context in embedding space [9]. Its relevance here is the predictive relationship between encoded views: the teacher supplies a complete-image target, and the student must infer that target without observing the corresponding image pixels.

Masked and self-predictive ideas already appear in face security. FSFM develops facial representations through selfsupervised learning for generalizable face security [22]. AdG-MAE combines masked reconstruction, gated attention, and spectral reweighting for domain-generalized PAD [23]. Autoencoders have also been studied for open-set presentation attack detection [24]. In the adjacent digital-forgery setting, structural consistency and self-predictive reconstruction are used to expose facial inconsistencies [25]. These studies establish masked learning as an existing direction in facesecurity representation learning.

CP-JEPA instead applies masked latent prediction during supervised low-rank adaptation of CLIP. Its EMA targets are task-adapted representations rather than reconstructed pixels.

## III. PROPOSED CP-JEPA FRAMEWORK

CP-JEPA adapts a pretrained CLIP visual encoder for face presentation attack detection (PAD) by combining binary classification with teacher-guided prediction of masked spatial representations. A cross-entropy stage first adapts the encoder and classifier. The subsequent predictive stage retains classification supervision while introducing masked latent prediction, masked classification, and prediction consistency. Fig. 1 illustrates the CP-JEPA training architecture.

## A. CLIP-LoRA for Face Presentation Attack Detection

Let $\begin{array} { r } { \mathcal { D } _ { S } = \bigcup _ { k = 1 } ^ { K } \mathcal { D } _ { k } } \end{array}$ denote the source training datasets, with samples $( x , y )$ , where x is a face image and $y \in$ {0, 1} indicates spoof or bona fide, respectively. We use the pretrained CLIP ViT-B/16 visual encoder, whose native global representation is obtained from the class token after the final transformer normalization and visual projection. Let $g _ { \Delta } ( x )$ denote this 512-dimensional representation. The binary classifier estimates

$$
p _ { \theta } ( y \mid x ) = \mathrm { s o f t m a x } \left( C _ { \omega } g _ { \Delta } ( x ) + b _ { \omega } \right) _ { y } ,\tag{1}
$$

where $\Delta$ denotes the trainable encoder adapters, ω denotes the classifier parameters, and $\theta = ( \Delta , \omega )$ . The pretrained CLIP weights remain frozen; the classifier uses the native global representation.

The visual encoder is adapted using low-rank adaptation (LoRA). Trainable low-rank updates are applied to the query and value projections in all 12 transformer blocks. For $U \in$ $\{ Q , V \}$ , the adapted projection matrix is

$$
W _ { U } = W _ { U , 0 } + \frac { \alpha } { r } B _ { U } A _ { U } ,\tag{2}
$$

![](images/e23c2ef6e62d33d269443b75f6f163a8d7f5a0ae068c39579de5d906e06ff5b8.jpg)  
Fig. 1. Overview of CP-JEPA training and inference. The student learns from masked spatial predictions and binary classification, using an EMA teacher to provide latent targets. The teacher and predictor are used only during training; inference uses the adapted CLIP encoder and its native binary classifier.

where $W _ { U , 0 }$ is the frozen pretrained matrix, $A _ { U } \in \mathbb { R } ^ { r \times d }$ and $B _ { U } \in \mathbb { R } ^ { d \times r }$ are low-rank factors, and $d = 7 6 8$ . We use $r = 8$ and $\alpha = 8$ . The factors $A _ { U }$ are initialized using Kaiming uniform initialization, while $B _ { U }$ is initialized to zero, leaving the pretrained projections unchanged at initialization. The key projections and remaining pretrained CLIP parameters are frozen. Since the adapters modify attention operations shared by the class token and patch tokens, their updates affect both the global classification representation and the spatial features used for latent prediction.

The first training stage optimizes the adapters and binary classifier using cross-entropy loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C E } } = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { S } } \log p _ { \theta } ( y \mid a ( x ) ) , } \end{array}\tag{3}
$$

where a denotes the training augmentation. The resulting PADadapted model initializes the subsequent predictive training stage, during which classification supervision remains active.

## B. Input Views and Patch Masking

For each training image, two views are generated using a shared geometric transformation $a _ { g }$ . The teacher receives $x _ { w } ~ = ~ a _ { g } ( x )$ , while the student receives a perturbed view $x _ { s } ~ = ~ a _ { p } ( a _ { g } ( x ) )$ , where $a _ { p }$ applies mild color jitter. The geometric transformation includes resizing and horizontal flipping. Applying the same geometry to both views preserves correspondence between their spatial patch positions.

To construct the masked student input, we randomly select patch positions from the perturbed view. At a resolution of $2 2 4 \times 2 2 4$ , CLIP ViT-B/16 produces a 14 × 14 grid of $N =$ 196 spatial tokens. A mask set $M \subset \{ 1 , \ldots , N \}$ contains $k = \mathrm { r o u n d } ( \rho N )$ uniformly sampled positions, with $\rho = 0 . 3 0$ The corresponding image patches are set to zero before student encoding:

$$
\widetilde { x } _ { s } = ( 1 - M ^ { \uparrow } ) \odot x _ { s } ,\tag{4}
$$

where $M ^ { \uparrow }$ denotes the patch mask expanded to image resolution by nearest-neighbor interpolation, and ⊙ is element-wise multiplication. Masked pixels are set to zero in normalized coordinates, corresponding to the CLIP channel means in the original image space. Applying the mask before the encoder prevents the original masked pixel values from entering the student representation. The predictor subsequently replaces the features at these positions with learned mask tokens.

## C. Teacher-Guided Masked Latent Prediction

Masked latent prediction uses spatial representations from the CLIP visual encoder. Let $H _ { \Delta } \mathbf { \bar { ( } } x ) \in \bar { \mathbb { R } ^ { N \times 7 6 8 } }$ denote the adapted encoder’s patch representations after excluding the class token. These features undergo CLIP’s post-transformer layer normalization and visual projection, followed by a trainable linear projector $P _ { \psi } : { \overline { { \mathbb { R } } } } ^ { 5 \bar { 1 } 2 } \ \to \ \mathbb { R } ^ { 1 9 2 }$ and layer normalization. The resulting representation at position i is

$$
z _ { \Delta , \psi , i } ( x ) = \mathrm { L N } _ { \psi } \left( P _ { \psi } \left( \mathrm { L N } ( H _ { \Delta , i } ( x ) ) W _ { \mathrm { p r o j } } \right) \right) ,\tag{5}
$$

where LN and $W _ { \mathrm { p r o j } }$ are the pretrained CLIP layer normalization and visual projection, respectively, and $\mathrm { L N } _ { \psi }$ is the trainable layer normalization following the spatial projector.

The teacher is initialized from the CE-adapted student encoder and a copy of its newly initialized spatial projector. Its LoRA parameters are denoted by $\bar { \Delta } .$ , while $\psi$ denotes the spatial projector and its layer normalization parameters. For the unmasked view $x _ { w }$ , the teacher produces normalized spatial targets:

$$
t _ { i } = \mathrm { s g } \left( \frac { z _ { \bar { \Delta } , \bar { \psi } , i } ( x _ { w } ) } { \| z _ { \bar { \Delta } , \bar { \psi } , i } ( x _ { w } ) \| _ { 2 } } \right) ,\tag{6}
$$

where sg denotes stop-gradient. The teacher’s LoRA and spatial projection parameters are updated after each student optimization step using an exponential moving average (EMA):

$$
\begin{array} { l } { { \bar { \Delta }  \mu \bar { \Delta } + ( 1 - \mu ) \Delta , } } \\ { { \bar { \psi }  \mu \bar { \psi } + ( 1 - \mu ) \psi , \quad \quad \mu = 0 . 9 9 . } } \end{array}\tag{7}
$$

The pretrained CLIP weights remain fixed in both networks, and the teacher’s binary classifier is not used to construct the spatial targets.

The student encodes the masked view $\widetilde { x } _ { s }$ and obtains the projected spatial sequence $Z = z _ { \Delta , \psi } ( \widetilde { x } _ { s } )$ . Before prediction, features at masked positions are replaced with a learned mask token m, while visible positions retain their projected features. A fixed two-dimensional sinusoidal positional encoding $e _ { i }$ is added at each position:

$$
u _ { i } = \left\{ { \begin{array} { l l } { m + e _ { i } , } & { i \in { \mathcal { M } } , } \\ { Z _ { i } + e _ { i } , } & { i \notin { \mathcal { M } } . } \end{array} } \right.\tag{8}
$$

The sequence $U = \{ u _ { i } \} _ { i = 1 } ^ { N }$ is processed by a transformer predictor $q _ { \phi }$ with two layers, four attention heads, a hidden dimension of 192, and a feed-forward dimension of 768. The predictor uses GELU activations and zero dropout. Its output at position i is ${ \widehat { z } } _ { i } = q _ { \phi } ( U ) _ { i }$ . The prediction loss is the mean cosine distance between the predicted representations and the teacher targets at masked positions:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { i \in \mathcal { M } } \left( 1 - \frac { \widehat { z } _ { i } ^ { \intercal } t _ { i } } { \Vert \widehat { z } _ { i } \Vert _ { 2 } } \right) .\tag{9}
$$

The objective predicts masked spatial representations in feature space rather than reconstructing RGB values. It follows the joint-embedding prediction principle of I-JEPA [9], here incorporated into a supervised PAD model initialized through cross-entropy training.

## D. PAD Classification and Consistency Objectives

The student also performs binary PAD classification on the unmasked view $x _ { s }$ and its masked counterpart $\widetilde { x } _ { s }$ . Both forward passes use the same encoder and native classifier. Ordinary cross-entropy supervision is retained for the unmasked view, while the masked view introduces an additional classification loss:

$$
\mathcal { L } _ { \mathrm { m a s k } } = - \mathbb { E } _ { ( x , y ) } \log p _ { \theta } ( y \mid \widetilde { x } _ { s } ) .\tag{10}
$$

A consistency objective relates the classification distributions produced by the two views. The unmasked prediction serves as a detached reference, while the masked prediction is optimized to match it:

$$
\mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } _ { x } D _ { \mathrm { K L } } \left( \mathrm { s g } [ p _ { \theta } ( \cdot  { | } x _ { s } ) ]  { | | } p _ { \theta } ( \cdot  { | }  { \widetilde { x } } _ { s } ) \right) .\tag{11}
$$

The unmasked distribution is detached; the consistency loss backpropagates through the masked-view prediction and its shared encoder and classifier parameters.

The complete predictive training objective combines unmasked classification, masked latent prediction, masked classification, and prediction consistency:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r e d } } + \lambda _ { m } \mathcal { L } _ { \mathrm { m a s k } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o n s } } , } \end{array}\tag{12}
$$

with $( \lambda _ { p } , \lambda _ { m } , \lambda _ { c } ) = ( 1 , 0 . 5 , 0 . 1 )$ . All auxiliary terms are active from the beginning of predictive adaptation. The prediction loss updates the student adapters through the spatial projector and transformer predictor, whereas the classification and consistency losses operate on the native global predictions. The objectives therefore share the adapted visual encoder while retaining separate spatial prediction and classification pathways.

## E. Training and Inference

The predictive stage is initialized from the cross-entropy stage-transition checkpoint. The student adapters, classifier, spatial projector, and predictor are optimized using (12); the teacher adapters and spatial projection parameters follow the EMA update in (7).

At inference, classification uses the adapted student encoder and its native binary classifier, following (1). The EMA teacher, masking procedure, and transformer predictor are not used. The spatial projector does not contribute to the classification logits. Predictive adaptation optimizes 1,321,410 student-side parameters, of which 295,938 belong to the LoRA adapters and classifier retained at inference. These counts exclude the frozen CLIP encoder.

## IV. EXPERIMENTAL SETUP

## A. Datasets and Evaluation Protocols

We evaluate CP-JEPA under cross-dataset and unseen-attack PAD protocols. The cross-dataset experiments use OULU-NPU (O) [26], CASIA-MFSD (C) [27], Idiap Replay-Attack (I) [28], MSU-MFSD (M) [29], and CelebA-Spoof [30]. The standard OCIM protocol trains on three datasets and evaluates on the fourth, covering OCI→M, OMI→C, OCM→I, and ICM→O. The held-out dataset supplies no samples for gradient-based training.

Single-source evaluation trains on each OCIM dataset individually and tests on the remaining three, producing twelve source–target combinations (Table III). Three further configurations examine the effect of training-source composition. First, CelebA-Spoof is added to each three-source OCIM training pool, with the remaining OCIM dataset held out for evaluation. Second, O+C+M are used for training and CelebA-Spoof for testing. Third, training uses only M+I, with C and O evaluated separately. These configurations examine additional real-source training data, transfer to CelebA-Spoof, and reduced-source training, respectively.

We also evaluate unseen attack categories using the SiW-Mv2 leave-one-attack-type-out protocol (Table VI), which excludes the target attack category from training.

## B. Preprocessing and Source Sampling

For OCIM videos, faces are detected using MTCNN[31]. The highest-confidence bounding box is expanded by a 30% margin. Training uses a randomly sampled usable frame from each video, while validation and testing use the middle frame. CelebA-Spoof crops follow its provided bounding-box annotations. Images are resized to 224 × 224 and normalized using

TABLE I  
MAIN CP-JEPA ARCHITECTURE AND TRAINING CONFIGURATION.
<table><tr><td>Setting</td><td>Configuration</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Visual encoder</td><td>CLIP ViT-B/16 (frozen base)</td></tr><tr><td>LoRA</td><td> $\mathrm { Q } / \mathrm { V } , \mathrm { a l l } 1 2 \mathrm { b l o c k s } ; r = 8 , \alpha = 8$ </td></tr><tr><td>Spatial projector</td><td>Linear, 512 → 192, followed by LayerNorm</td></tr><tr><td>Predictor</td><td>2 layers; 4 heads; hidden 192; FFN 768</td></tr><tr><td>Patch masking</td><td>Uniform random; 30% of 196 patches</td></tr><tr><td>EMA momentum</td><td> $\mu = 0 . 9 9$ </td></tr><tr><td>Auxiliary loss weights</td><td> $\dot { \lambda } _ { p } = 1 , \lambda _ { m } = 0 . 5 , \lambda _ { c } = 0 . 1$ </td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Input resolution</td><td>224 × 224</td></tr><tr><td>Training epochs</td><td>40 CE + 40 predictive</td></tr><tr><td>Batch size / accumulation</td><td>16 / 2 steps</td></tr><tr><td>CE optimizer</td><td>Adam; learning rate  $1 0 ^ { - 4 }$ </td></tr><tr><td>Predictive optimizer</td><td>AdamW; learning rate  $1 0 ^ { - 4 }$ </td></tr><tr><td>Predictive weight decay</td><td>0.01</td></tr><tr><td>Predictive gradient clip</td><td>1.0</td></tr></table>

CLIP channel statistics. Training applies horizontal flipping and color jitter, with brightness, contrast, saturation, and hue magnitudes of 0.15, 0.15, 0.10, and 0.02, respectively. During predictive adaptation, the teacher and student views share the same geometric transformation to preserve spatial correspondence.

Experiments incorporating CelebA-Spoof as an additional training source use domain- and class-balanced sampling, following established cross-domain PAD practices [2, 32]. If $n _ { k , c }$ denotes the number of training entries in source domain k and class c, each entry receives a sampling weight proportional to $1 / n _ { k , c }$ . Sampling is performed with replacement. The epoch budget is twice the number of OCIM source videos, subject to a batch-size-dependent minimum, preventing the larger CelebA-Spoof dataset from determining the training duration.

## C. Training and Model Selection

Training follows the two-stage procedure in Section III, using the selected CE checkpoint to initialize predictive adaptation. Table I summarizes the architecture and optimization settings.

Checkpoints are selected by minimum target-validation CE. Target-specific checkpoints are selected separately when multiple targets are evaluated; experiments requiring a common checkpoint use the equally weighted mean validation loss across targets. O and I use their designated development partitions, while development data for C and M are drawn from their official training partitions. CelebA-Spoof development and test membership follow the supplied manifests. Targetdevelopment labels are therefore used for model selection, but not for gradient-based training.

The auxiliary-objective study (Table V) includes CE continuation to control for additional supervised training. All variants retain unmasked CE. The “prediction only,” “masked CE only,” and “consistency only” variants activate the named auxiliary objective alongside unmasked CE. The complete model uses $( \lambda _ { p } , \lambda _ { m } , \lambda _ { c } ) = ( 1 , 0 . 5 , 0 . 1 )$ ; each removal experiment sets the corresponding coefficient to zero.

## D. Performance Measures

Let $s ( x ) \ = \ p _ { \theta } ( y \ = \ 1 \ | \ x )$ denote the bona fide score, where $y = 0$ represents an attack. A sample is classified as bona fide when $s ( x ) \geq \tau$ , where τ is the decision threshold selected using development partition and frozen before test evaluation. We report attack presentation classification error rate (APCER), bona fide presentation classification error rate (BPCER), half total error rate (HTER), and area under the ROC curve (AUC). The error rates are defined as

$$
\mathrm { A P C E R } ( \tau ) = \operatorname* { P r } ( s ( x ) \geq \tau \mid y = 0 ) ,
$$

$$
\mathrm { B P C E R } ( \tau ) = \mathrm { P r } ( s ( x ) < \tau \mid y = 1 ) ,\tag{13}
$$

$$
\begin{array} { r } { \mathrm { H T E R } ( \tau ) = \frac { 1 } { 2 } \big [ \mathrm { A P C E R } ( \tau ) + \mathrm { B P C E R } ( \tau ) \big ] . } \end{array}\tag{14}
$$

(15)

## V. RESULTS AND DISCUSSION

## A. Cross-Dataset Performance on OCIM

Table II compares standard OCIM performance with published PAD methods. CP-JEPA achieves the lowest mean HTER of 3.74% among the compared methods, followed by CCPE at 4.02%, and a mean AUC of 98.73%. On OCI→M and ICM→O, CP-JEPA achieves both the lowest HTER and highest AUC. On OMI→C and OCM→I, other methods achieve better results.

## B. Single-Source Cross-Dataset Performance

Table III reports single-source transfer across twelve OCIM source–target combinations. CP-JEPA achieves the lowest mean HTER among the compared methods at 11.36%, improving on FoundPAD ViT-L’s 15.49% by 4.13 percentage points. It obtains the lowest HTER in eight combinations, including all three targets when trained on I. These results demonstrate competitive cross-dataset transfer without requiring multiple source datasets.

## C. Additional Real-Source Training Data

Adding CelebA-Spoof to the OCIM training sources reduces CP-JEPA’s mean HTER from 3.74% to 2.35% and increases AUC from 98.73% to 99.32% (Table IV). The improvement in HTER is 1.39 percentage points, or 37.2% relative to standard OCIM training. Among the listed source-augmented methods, CP-JEPA has the lowest mean HTER, while CCPE achieves a higher mean AUC of 99.55%. Figure 2 compares the sourceaugmented model with CE adaptation.

## D. Contribution of the Auxiliary Objectives

Table V examines the three auxiliary objectives against CE continuation and their individual removal variants. The complete model achieves 3.74% mean HTER and 98.73% AUC, compared with 3.97% and 98.52% for CE continuation. This comparison controls for additional supervised training after the initial CE stage.

Removing latent prediction raises mean HTER to 3.99%, while removing masked CE or consistency raises it to 5.05% and 4.97%, respectively. The prediction objective does not improve every target: the variant without prediction performs better on O and ties the complete model in HTER on M. The complete model achieves the lowest mean HTER among the ablation variants, although the effects differ across targets.

TABLE II  
CROSS-DOMAIN FACE PRESENTATION ATTACK DETECTION RESULTS ON THE OCIM BENCHMARK. LOWER HTER AND HIGHER AUC INDICATE BETTER PERFORMANCE. THE BEST RESULTS ARE SHOWN IN BOLD, WHILE THE SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td rowspan="2">Method</td><td colspan="2">OCI→M</td><td colspan="2">OMI→C</td><td colspan="2">OCM→I</td><td colspan="2">ICM→0</td><td rowspan="2">Avg. HTER(%)↓</td><td rowspan="2">Avg. AUC(%)↑</td></tr><tr><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER↓</td><td>AUC ↑</td></tr><tr><td>CIFAS [33] (2022)</td><td>5.95</td><td>96.32</td><td>10.66</td><td>95.30</td><td>8.50</td><td>97.24</td><td>13.17</td><td>93.44</td><td>9.57</td><td>95.58</td></tr><tr><td>IADG [4] (2023)</td><td>5.41</td><td>98.19</td><td>8.70</td><td>96.44</td><td>10.62</td><td>94.50</td><td>8.86</td><td>97.14</td><td>8.40</td><td>96.57</td></tr><tr><td>DRDN [10] (2023)</td><td>10.00</td><td>96.20</td><td>20.90</td><td>83.80</td><td>11.40</td><td>94.50</td><td>15.60</td><td>90.90</td><td>14.50</td><td>91.30</td></tr><tr><td>DiffFAS-R [34] (2024)</td><td>5.90</td><td>98.10</td><td>7.32</td><td>97.40</td><td>5.66</td><td>98.69</td><td>12.38</td><td>94.33</td><td>7.82</td><td>97.13</td></tr><tr><td>DiffFAS-V [34] (2024)</td><td>2.86</td><td>98.41</td><td>10.11</td><td>96.32</td><td>6.36</td><td>97.89</td><td>8.11</td><td>97.27</td><td>6.86</td><td>97.47</td></tr><tr><td>GAC-FAS [32] (2024)</td><td>5.00</td><td>97.56</td><td>8.20</td><td>95.16</td><td>4.29</td><td>98.87</td><td>8.60</td><td>97.16</td><td>6.52</td><td>97.19</td></tr><tr><td>CA-MoEiT [35] (2024)</td><td>2.88</td><td>98.76</td><td>7.89</td><td>97.70</td><td>6.18</td><td>98.94</td><td>9.72</td><td>96.22</td><td>6.67</td><td>97.91</td></tr><tr><td>FoundPAD (ViT-B) [8] (2025)</td><td>20.95</td><td>89.88</td><td>4.89</td><td>98.08</td><td>10.45</td><td>95.80</td><td>6.19</td><td>98.31</td><td>10.62</td><td>95.52</td></tr><tr><td>FoundPAD (ViT-L) [8] (2025)</td><td>16.90</td><td>93.18</td><td>6.00</td><td>98.72</td><td>9.90</td><td>96.07</td><td>5.87</td><td>98.41</td><td>9.67</td><td>96.60</td></tr><tr><td>FreqShortcut [36] (2025)</td><td>5.95</td><td>98.52</td><td>7.33</td><td>97.86</td><td>5.45</td><td>98.77</td><td>10.88</td><td>96.29</td><td>7.40</td><td>97.86</td></tr><tr><td>FSFM ViT-B [22] (2025)</td><td>6.58</td><td>97.43</td><td>4.30</td><td>99.10</td><td>14.63</td><td>90.96</td><td>10.02</td><td>96.23</td><td>8.88</td><td>95.93</td></tr><tr><td>CCPE [37] (2026)</td><td>3.10 5.00</td><td>99.21 98.56</td><td>1.33</td><td>99.36</td><td>6.08</td><td>94.36</td><td>5.57</td><td>98.49</td><td>4.02</td><td>97.86</td></tr><tr><td>SA-TTDG [38] (2026)</td><td>4.59</td><td>98.75</td><td>4.20</td><td>99.48 98.60</td><td>3.87 4.34</td><td>99.41 99.22</td><td>8.37</td><td>97.94</td><td>5.36</td><td>98.85</td></tr><tr><td>AdG-MAE [23] (2026)</td><td></td><td></td><td>5.21</td><td></td><td></td><td></td><td>6.81</td><td>98.85</td><td>5.24</td><td>98.85</td></tr><tr><td>CP-JEPA (Ours)</td><td>2.50</td><td>99.73</td><td>3.33</td><td>99.12</td><td>5.25</td><td>97.19</td><td>3.89</td><td>98.87</td><td>3.74</td><td>98.73</td></tr></table>

TABLE III

SINGLE-SOURCE CROSS-DATASET HTER (%) ON OCIM. EACH METHOD IS TRAINED ON ONE OCIM SOURCE DATASET AND EVALUATED SEPARATELY ON THE REMAINING THREE. LOWER HTER IS BETTER. THE BEST AND SECOND-BEST RESULTS ARE SHOWN IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="3">Train C</td><td colspan="3">Train I</td><td colspan="3">Train M</td><td colspan="3">Train O</td><td rowspan="2">Avg.</td></tr><tr><td>I</td><td>M</td><td>0</td><td>C</td><td>M</td><td>0</td><td>C</td><td>I</td><td>0</td><td>C</td><td>I</td><td>M</td></tr><tr><td>DRDN-ii [10] (2023)</td><td>27.10</td><td>17.10</td><td>24.50</td><td>39.10</td><td>23.10</td><td>37.10</td><td>27.80</td><td>13.00</td><td>21.70</td><td>20.10</td><td>19.00</td><td>19.90</td><td>24.13</td></tr><tr><td>CF-PAD [39] (2024)</td><td>24.80</td><td>17.14</td><td>19.43</td><td>34.00</td><td>24.76</td><td>31.70</td><td>14.44</td><td>15.90</td><td>25.34</td><td>20.33</td><td>21.50</td><td>15.00</td><td>22.03</td></tr><tr><td>FoundPAD (ViT-B) [8] (2025)</td><td>16.40</td><td>24.52</td><td>15.14</td><td>17.00</td><td>18.57</td><td>13.38</td><td>20.00</td><td>17.10</td><td>19.41</td><td>7.89</td><td>8.95</td><td>23.33</td><td>16.81</td></tr><tr><td>FoundPAD (ViT-L) [8] (2025)</td><td>14.05</td><td>21.43</td><td>11.00</td><td>10.22</td><td>19.29</td><td>16.94</td><td>12.00</td><td>14.55</td><td>20.93</td><td>7.22</td><td>14.40</td><td>23.81</td><td>15.49</td></tr><tr><td>CP-JEPA (Ours)</td><td>13.13</td><td>20.00</td><td>20.10</td><td>4.63</td><td>8.33</td><td>10.76</td><td>12.96</td><td>6.88</td><td>15.07</td><td>8.15</td><td>5.00</td><td>11.25</td><td>11.36</td></tr></table>

TABLE IV

PERFORMANCE ON THE FOUR OCIM LEAVE-ONE-DOMAIN-OUT PROTOCOLS WITH CELEBA-SPOOF AS AN ADDITIONAL FACEPAD TRAINING SOURCE. RESULTS ARE REPORTED AS HTER↓/AUC↑.
<table><tr><td>Model</td><td>ICM→0</td><td>OMI→C</td><td>OCM→I</td><td>OCI→M</td><td>Average</td></tr><tr><td>FLIP-MCL [15] (2023)</td><td>2.31/99.63</td><td>0.54/99.98</td><td>4.25/99.07</td><td>4.95/98.11</td><td>3.01/99.20</td></tr><tr><td>FSFM ViT-B [22] (2025)</td><td>7.68/97.11</td><td>3.16/99.41</td><td>4.63/99.03</td><td>3.78/99.15</td><td>4.81/98.68</td></tr><tr><td>CCPE [37] (2026)</td><td>3.64/99.67</td><td>1.30/99.98</td><td>4.15/99.31</td><td>2.86/99.24</td><td>2.99/99.55</td></tr><tr><td>CP-JEPA (Ours)</td><td>2.43/99.54</td><td>0.74/99.73</td><td>2.88/98.97</td><td>3.33/99.04</td><td>2.35/99.32</td></tr></table>

![](images/edbca36cea21d6eae0d12cf31a1965304146bd4d8728c493ce4685303903ef17.jpg)  
Fig. 2. Cross-domain HTER, AUC, APCER, and BPCER for CE adaptation and CP-JEPA with CelebA-Spoof added to the source training data.

## E. Generalization to Unseen Attack Categories

On SiW-Mv2, CP-JEPA achieves the lowest reported mean HTER in Table VI, at 3.36%, with an attack-category standard deviation of 4.28%. The next-lowest mean is 3.88% for CLIP-L14, giving an absolute difference of 0.52 points and a relative reduction of 13.5%. The corresponding CLIP-B16 and DINO ViT-B/14 means are 5.11% and 5.09%, respectively.

TABLE V  
ABLATION STUDY OF THE AUXILIARY OBJECTIVES IN CP-JEPA ACROSS THE FOUR OCIM PROTOCOLS.
<table><tr><td>Model</td><td>0</td><td>C</td><td>I</td><td>M</td><td>Avg.</td></tr><tr><td colspan="6">HTER (%) ↓</td></tr><tr><td>CE</td><td>3.65</td><td>3.33</td><td>4.75</td><td>4.17</td><td>3.97</td></tr><tr><td>Full</td><td>3.89</td><td>3.33</td><td>5.25</td><td>2.50</td><td>3.74</td></tr><tr><td>-Pred</td><td>3.19</td><td>5.00</td><td>5.25</td><td>2.50</td><td>3.99</td></tr><tr><td>-Mask</td><td>4.31</td><td>4.81</td><td>5.25</td><td>5.83</td><td>5.05</td></tr><tr><td>-Cons</td><td>4.31</td><td>3.52</td><td>5.38</td><td>6.67</td><td>4.97</td></tr><tr><td>Pred only</td><td>6.70</td><td>3.33</td><td>4.88</td><td>5.00</td><td>4.98</td></tr><tr><td>Mask only</td><td>3.61</td><td>6.67</td><td>5.38</td><td>3.75</td><td>4.85</td></tr><tr><td>Cons only</td><td>3.99</td><td>6.67</td><td>5.25</td><td>5.42</td><td>5.33</td></tr><tr><td colspan="6">AUC (%) ↑</td></tr><tr><td>CE</td><td>98.49</td><td>99.30</td><td>97.30</td><td>98.98</td><td>98.52</td></tr><tr><td>Full</td><td>98.87</td><td>99.12</td><td>97.19</td><td>99.73</td><td>98.73</td></tr><tr><td>-Pred</td><td>99.09</td><td>98.44</td><td>97.21</td><td>99.52</td><td>98.57</td></tr><tr><td>-Mask</td><td>98.69</td><td>98.41</td><td>97.23</td><td>98.54</td><td>98.22</td></tr><tr><td>-Cons</td><td>98.70</td><td>99.12</td><td>97.17</td><td>98.58</td><td>98.39</td></tr><tr><td>Pred only</td><td>95.75</td><td>99.09</td><td>97.25</td><td>99.04</td><td>97.78</td></tr><tr><td>Mask only</td><td>99.09</td><td>97.27</td><td>97.21</td><td>99.33</td><td>98.22</td></tr><tr><td>Cons only</td><td>98.81</td><td>97.67</td><td>97.22</td><td>98.79</td><td>98.12</td></tr></table>

CE: CLIP-LoRA (CE continuation); Full: CP-JEPA (complete). Pred: prediction; Mask: masked CE; Cons: consistency.

The attack-wise results provide a more detailed comparison. CP-JEPA achieves 0.19% HTER on seven categories, including half masks, silicone masks, and print attacks, and obtains 3.45% on cosmetic attacks. Performance remains categorydependent: replay and obfuscation produce HTER values of 9.22% and 12.49%, respectively. The mean HTER therefore summarizes results that vary considerably across attack categories.

TABLE VI  
HTER (%) ON THE SIW-MV2 LEAVE-ONE-OUT PROTOCOL. LOWER VALUES INDICATE BETTER PERFORMANCE. THE BEST AND SECOND-BEST RESULTS FOR EACH ATTACK TYPE ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Attack</td><td>MD-FAS (SRENet) [40] (2022)</td><td>DINO (ViT-B-14) [41] (2025)</td><td>CLIP-B16 [41] (2025)</td><td>CLIP-B32 [41] (2025)</td><td>CLIP-L14 [41] (2025)</td><td>DINO-S14 [41] (2025)</td><td>DINO-L14 [41] (2025)</td><td>CP-JEPA (Ours)</td></tr><tr><td>FunE.</td><td>29.50</td><td>8.52</td><td>13.46</td><td>19.24</td><td>9.57</td><td>18.76</td><td>21.62</td><td>11.57</td></tr><tr><td>PEye</td><td>2.70</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td></tr><tr><td>PMouth</td><td>1.10</td><td>0.19</td><td>0.39</td><td>3.65</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td></tr><tr><td>PaperG.</td><td>11.90</td><td>0.19</td><td>1.62</td><td>2.47</td><td>0.19</td><td>1.04</td><td>0.19</td><td>2.30</td></tr><tr><td>Ob.</td><td>1.30</td><td>18.16</td><td>8.60</td><td>10.14</td><td>12.80</td><td>9.76</td><td>7.68</td><td>12.49</td></tr><tr><td>Impers.</td><td>24.50</td><td>0.39</td><td>0.19</td><td>0.193</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td></tr><tr><td>Cosmetic</td><td>10.90</td><td>18.88</td><td>11.37</td><td>6.36</td><td>6.94</td><td>22.93</td><td>21.00</td><td>3.45</td></tr><tr><td>HalfM.</td><td>8.00</td><td>1.47</td><td>3.51</td><td>5.48</td><td>1.27</td><td>6.75</td><td>1.08</td><td>0.19</td></tr><tr><td>Silicone</td><td>9.20</td><td>0.58</td><td>0.58</td><td>1.74</td><td>0.19</td><td>0.39</td><td>0.77</td><td>0.19</td></tr><tr><td>TransM.</td><td>0.00</td><td>1.41</td><td>3.21</td><td>5.01</td><td>0.19</td><td>5.20</td><td>0.19</td><td>2.87</td></tr><tr><td>Paper</td><td>0.60</td><td>0.00</td><td>0.19</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>3.76</td></tr><tr><td>Mann.</td><td>4.00</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td></tr><tr><td>Replay</td><td>17.90</td><td>7.05</td><td>16.27</td><td>18.84</td><td>13.00</td><td>6.73</td><td>7.56</td><td>9.22</td></tr><tr><td>Print</td><td>9.60</td><td>13.99</td><td>11.72</td><td>13.79</td><td>9.45</td><td>18.16</td><td>11.72</td><td>0.19</td></tr><tr><td>Avg.±Std.</td><td>9.40 ± 8.80</td><td>5.09 ± 7.05</td><td>5.11 ± 5.87</td><td>6.24 ± 6.75</td><td>3.88 ± 5.21</td><td>6.46 ± 8.03</td><td>5.18 ± 7.78</td><td>3.36 ± 4.28</td></tr></table>

TABLE VII  
CROSS-DATASET EVALUATION ON CELEBA-SPOOF USING O, C, AND M AS TRAINING SOURCES.
<table><tr><td>Model</td><td>HTER↓</td><td>AUC↑</td></tr><tr><td>Panwar et al. [42] (2021)</td><td>26.10</td><td>80.00</td></tr><tr><td>CIFAS [33] (2022)</td><td>24.60</td><td>83.20</td></tr><tr><td>SourceOnly [43] (2022)</td><td>29.70</td><td>77.50</td></tr><tr><td>SDA-FAS [43] (2022)</td><td>18.90</td><td>90.90</td></tr><tr><td>SourceOnly++ [44] (2024)</td><td>23.87</td><td>83.67</td></tr><tr><td>SDA-FAS++ [44] (2024)</td><td>13.74</td><td>93.70</td></tr><tr><td>CP-JEPA (Ours)</td><td>10.74</td><td>93.57</td></tr></table>

## F. Transfer to CelebA-Spoof and Reduced-Source Training

For O+C+M→CelebA-Spoof, CP-JEPA obtains 10.74% HTER and 93.57% AUC, with 11.29% APCER and 10.20% BPCER (Table VII). This is the lowest HTER among the listed methods, improving on SDA-FAS++ by 3.00 points, while its AUC is 0.13 points lower than the 93.70% reported for SDA-FAS++. The two measures therefore identify different strengths: the proposed method has the lower error at its selected operating point, whereas SDA-FAS++ has slightly higher ROC discrimination.

With only M+I as training sources, CP-JEPA achieves 2.59% HTER on C and 8.06% on O, giving a mean of 5.32% (Table VIII). It achieves the lowest mean HTER among the listed methods, although performance differs substantially between the two targets.

## VI. CONCLUSION

We presented CP-JEPA, a context-predictive low-rank adaptation framework for face presentation attack detection. The framework combines supervised classification with masked latent prediction, masked classification, and prediction consistency to adapt a pretrained CLIP visual encoder. An EMA teacher provides spatial representation targets, while low-rank attention updates connect the predictive objective to the global classification task. The final classification predictions depend on the adapted student encoder and its native binary classifier, without requiring the teacher or transformer predictor.

TABLE VIII  
CROSS-DATASET EVALUATION WITH M AND I AS TRAINING SOURCES AND C AND O AS HELD-OUT TARGETS.
<table><tr><td rowspan="2">Model</td><td colspan="2">M+I→C</td><td colspan="2">M+I→0</td><td colspan="2">Average</td></tr><tr><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td></tr><tr><td>SSDG-M [2] (2020)</td><td>31.89</td><td>71.29</td><td>36.01</td><td>66.88</td><td>33.95</td><td>69.09</td></tr><tr><td>D2AM [45] (2021)</td><td>32.65</td><td>72.04</td><td>27.70</td><td>75.36</td><td>30.18</td><td>73.70</td></tr><tr><td>SSAN-R [46] (2022)</td><td>25.56</td><td>83.89</td><td>24.44</td><td>82.86</td><td>25.00</td><td>83.38</td></tr><tr><td>HFN+MP [47] (2022)</td><td>30.89</td><td>72.48</td><td>20.94</td><td>86.71</td><td>25.92</td><td>79.60</td></tr><tr><td>MADG [48] (2023)</td><td>23.33</td><td>82.32</td><td>26.29</td><td>81.50</td><td>24.81</td><td>81.91</td></tr><tr><td>DiVT-M [49] (2023)</td><td>20.11</td><td>86.71</td><td>23.61</td><td>85.73</td><td>21.86</td><td>86.22</td></tr><tr><td>DGUA-FAS [50] (2023)</td><td>19.22</td><td>86.81</td><td>20.05</td><td>88.75</td><td>19.64</td><td>87.78</td></tr><tr><td>DSCI [51] (2024)</td><td>16.11</td><td>89.48</td><td>23.19</td><td>84.73</td><td>19.65</td><td>87.11</td></tr><tr><td>TTDG [52] (2024)</td><td>17.77</td><td>86.69</td><td>17.70</td><td>90.09</td><td>17.74</td><td>88.39</td></tr><tr><td>FQ-FAS [53] (2024)</td><td>19.72</td><td>84.69</td><td>17.05</td><td>90.05</td><td>18.39</td><td>87.37</td></tr><tr><td>FreqShortcut [36] (2025)</td><td>23.33</td><td>84.99</td><td>16.16</td><td>91.72</td><td>19.75</td><td>88.36</td></tr><tr><td>DCRN [54] (2025)</td><td>15.00</td><td>91.81</td><td>19.31</td><td>89.43</td><td>17.16</td><td>90.62</td></tr><tr><td>SACL [55] (2026)</td><td>15.56</td><td>90.56</td><td>19.16</td><td>89.48</td><td>17.36</td><td>90.02</td></tr><tr><td>DSCM-FAS [56] (2026)</td><td>13.44</td><td>91.89</td><td>18.38</td><td>91.64</td><td>15.91</td><td>91.77</td></tr><tr><td>AdG-MAE [23] (2026)</td><td>17.10</td><td>91.97</td><td>16.12</td><td>90.69</td><td>16.61</td><td>91.33</td></tr><tr><td>CP-JEPA (Ours)</td><td>2.59</td><td>98.12</td><td>8.06</td><td>95.08</td><td>5.32</td><td>96.60</td></tr></table>

Under the reported evaluation, CP-JEPA achieves 3.74% mean HTER and 98.73% AUC on OCIM. Adding CelebA-Spoof to the training sources improves these results to 2.35% and 99.32%, respectively. The objective ablation supports the combined training formulation, which improves mean HTER over CE continuation and each auxiliary-loss removal variant. Single-source evaluation yields 11.36% mean HTER across twelve OCIM transfers, with the lowest HTER in eight combinations among the compared methods. Further evaluations achieve 3.36% mean HTER on unseen SiW-Mv2 attack categories, 10.74% HTER on CelebA-Spoof, and 5.32% mean HTER with two training sources. These findings support context-predictive supervision as an effective complement to discriminative low-rank adaptation for RGB face PAD, without introducing additional inference modules.

## REFERENCES

[1] Z. Yu, C. Zhao, Z. Wang, Y. Qin, Z. Su, X. Li, F. Zhou, and G. Zhao, “Searching central difference convolutional networks for face anti-spoofing,” in Proceedings of the

IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5295–5305.

[2] Y. Jia, J. Zhang, S. Shan, and X. Chen, “Single-side domain generalization for face anti-spoofing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8484–8493.

[3] Y. Sun, Y. Liu, X. Liu, Y. Li, and W.-S. Chu, “Rethinking domain generalization for face anti-spoofing: Separability and alignment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24 563–24 574.

[4] Q. Zhou, K.-Y. Zhang, T. Yao, X. Lu, R. Yi, S. Ding, and L. Ma, “Instance-aware domain generalization for face anti-spoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 20 453–20 463.

[5] D. A. Dharmawan, A. S. Nugroho, and S. A. P. Handoko, “Toward Generalized and Explainable Face Anti-Spoofing via Decomposition-Fusion Networks,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 4, pp. 640–651, 2026.

[6] K. D. K. Yadav, I. Kavati, and R. Cheruku, “Gradient-Reversed Domain-Generalizable Multi-Modal Cross-Attention Network for Robust Face Anti-Spoofing,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 1, pp. 84–98, 2026.

[7] D. A. Dharmawan, M. V. A. Krisna, A. S. Nugroho, A. Ferriyan, B. Yuwono, J. Widiwasita, and A. J. S. Hadi, “FAS-MBCA: Face Anti-Spoofing Using Multi-Path Bi-Directional Cross-Attention Fusion,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 5, pp. 947–959, 2026.

[8] G. Ozgur, E. Caldeira, T. Chettaoui, F. Boutros, R. Ramachandra, and N. Damer, “Foundpad: Foundation models reloaded for face presentation attack detection,” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 745–755.

[9] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas, “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 15 619–15 629.

[10] Y. He, F. Peng, and M. Long, “Dynamic residual distillation network for face anti-spoofing with feature attention learning,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 5, no. 4, pp. 579–592, 2023.

[11] R. Ramachandra, N. Vetrekar, S. Patwardhan, S. Venkatesh, G. Naik, and R. S. Gad, “PCGattnNet: A 3-d point cloud dynamic graph attention for generalizable face presentation attack detection,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 7, no. 4, pp. 924–939, 2025.

[12] Y. Diao, Y. Li, J. Guo, X. Lin, A. Liu, J. Wan, J. T. Zhou, Z. Yu, and Z. Lei, “Modality- and Domain-Aware Adaptivity for Generalized Multimodal Face Anti-Spoofing,” IEEE Transactions on Information Forensics and Security, vol. 21, pp. 8963–8978, 2026.

[13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning, vol. 139, 2021, pp. 8748–8763.

[14] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” in International Conference on Learning Representations, 2022.

[15] K. Srivatsan, M. Naseer, and K. Nandakumar, “Flip: Cross-domain face anti-spoofing with language guidance,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19 685–19 696.

[16] L. Cai, H. Wang, J. Ji, X. Sun, L. Cao, and R. Ji, “ME-FAS: Multimodal text enhancement for cross-domain face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 20, 2025.

[17] H. Ma, A. Liu, S. Xue, Y. Diao, D. Liang, H. Liang, Y. Liang, J. Wan, X. Yang, Z. Hu, and Z. Lei, “CLIP-SA: CLIP-Guided Semantic Alignment for Generalizable Face Anti-spoofing,” IEEE Transactions on Multimedia, pp. 1–16, 2026, early access.

[18] L. Zhang, K. Liu, T. Xie, J. Huang, Z. Yu, and W. Kang, “MoGA-ETA: Generalized Face Anti-Spoofing With Enhanced Text Guidance and Alignment,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 3, pp. 340–353, 2026.

[19] A. Liu, X. Lin, R. Zhi, Y. Liang, X. Zhu, Z. Cai, J. Wan, S. Escalera, and Z. Lei, “DGPDL: Domain-Guided Prompt Distribution Learning for Generalizable Face Anti-Spoofing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 8, pp. 8953–8969, 2026.

[20] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022/ html/He\_Masked\_Autoencoders\_Are\_Scalable\_Vision\_ Learners\_CVPR\_2022\_paper.html

[21] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar, B. Piot, K. Kavukcuoglu, R. Munos, and M. Valko, “Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21 271–21 284.

[22] G. Wang, F. Lin, T. Wu, Z. Liu, Z. Ba, and K. Ren, “FSFM: A generalizable face security foundation model via self-supervised facial representation learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 24 364–24 376.

[23] M. Sheikhfathollahi, S. Parkinson, and S. Khan, “AdG-MAE: Adaptive gated masked autoencoder for domaingeneralised face anti-spoofing,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 6,

pp. 1211–1224, 2026.

[24] I. Bastos, A. George, S. Marcel, and A. Rocha, “Autoencoders for Open-Set Presentation Attack Detection,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 2, pp. 192–206, 2026.

[25] L. Zhou, M. Chen, and H. Ruan, “Structural Consistency for Face Forgery Detection via Frequency Domain Enhancement and Self-Predictive Learning,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 8, no. 1, pp. 99–110, 2026.

[26] Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, and A. Hadid, “Oulu-npu: A mobile face presentation attack database with real-world variations,” in 2017 12th IEEE international conference on automatic face & gesture recognition (FG 2017). IEEE, 2017, pp. 612–618.

[27] Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A face antispoofing database with diverse attacks,” in 2012 5th IAPR international conference on Biometrics (ICB). IEEE, 2012, pp. 26–31.

[28] I. Chingovska, A. Anjos, and S. Marcel, “On the effectiveness of local binary patterns in face anti-spoofing,” in 2012 BIOSIG-proceedings of the international conference of biometrics special interest group (BIOSIG). IEEE, 2012, pp. 1–7.

[29] D. Wen, H. Han, and A. K. Jain, “Face spoof detection with image distortion analysis,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, 2015.

[30] Y. Zhang, Z. Yin, Y. Li, G. Yin, J. Yan, J. Shao, and Z. Liu, “Celeba-spoof: Large-scale face anti-spoofing dataset with rich annotations,” in European conference on computer vision. Springer, 2020, pp. 70–85.

[31] K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, Oct 2016.

[32] B. M. Le and S. S. Woo, “Gradient alignment for cross-domain face anti-spoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[33] Y. Liu, Y. Chen, W. Dai, C. Li, J. Zou, and H. Xiong, “Causal intervention for generalizable face anti-spoofing,” in 2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2022, pp. 01– 06.

[34] X. Ge, X. Liu, Z. Yu, J. Shi, C. Qi, J. Li, and H. Kalviainen, “DiffFAS: Face anti-spoofing via generative diffusion models,” in European Conference on Computer Vision, 2024.

[35] A. Liu, “Ca-moeit: Generalizable face anti-spoofing via dual cross-attention and semi-fixed mixture-of-expert,” International Journal of Computer Vision, vol. 132, no. 11, pp. 5439–5452, 2024.

[36] J. Cao and C. Ma, “Towards generalized face antispoofing from a frequency shortcut view,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2025, pp. 1005–1015.

[37] J. Guo, A. Liu, Y. Diao, J. Zhang, H. Ma, B. Zhao,

R. Hong, and M. Wang, “Domain generalization for face anti-spoofing via content-aware composite prompt engineering,” IEEE Transactions on Multimedia, vol. 28, 2026.

[38] X. Chang, L. Shi, and A. Zhang, “Semantically anchored test-time domain generalization for face anti-spoofing,” Computers, Materials & Continua, vol. 89, no. 1, 2026.

[39] M. Fang and N. Damer, “Face presentation attack detection by excavating causal clues and adapting embedding statistics,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 6269–6279.

[40] X. Guo, Y. Liu, A. Jain, and X. Liu, “Multi-domain learning for updating face anti-spoofing models,” in European conference on computer vision. Springer, 2022, pp. 230– 249.

[41] L. J. Gonzalez-Soler, J. E. Tapia, and C. Busch, “Are foundation models all you need for zero-shot face presentation attack detection?” in 2025 IEEE 19th International Conference on Automatic Face and Gesture Recognition (FG). IEEE, 2025, pp. 1–10.

[42] A. Panwar, P. Singh, S. Saha, D. P. Paudel, and L. Van Gool, “Unsupervised compound domain adaptation for face anti-spoofing,” in 2021 16th IEEE international conference on automatic face and gesture recognition (FG 2021). IEEE, 2021, pp. 1–8.

[43] Y. Liu, Y. Chen, W. Dai, M. Gou, C.-T. Huang, and H. Xiong, “Source-free domain adaptation with contrastive domain alignment and self-supervised exploration for face anti-spoofing,” in Computer Vision – ECCV 2022. Springer, 2022, pp. 511–528.

[44] ——, “Source-free domain adaptation with domain generalized pretraining for face anti-spoofing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5430–5448, 2024.

[45] Z. Chen, T. Yao, K. Sheng, S. Ding, Y. Tai, J. Li, F. Huang, and X. Jin, “Generalizable representation learning for mixture domain face anti-spoofing,” in Proceedings ofthe AAAI conference on artificial intelligence, vol. 35, no. 2, 2021, pp. 1132–1139.

[46] Z. Wang, Z. Wang, Z. Yu, W. Deng, J. Li, T. Gao, and Z. Wang, “Domain generalization via shuffled style assembly for face anti-spoofing,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 4113–4123.

[47] R. Cai, Z. Li, R. Wan, H. Li, Y. Hu, and A. C. Kot, “Learning meta pattern for face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 1201–1213, 2022.

[48] S. Zhang and W. Nie, “Multi-domain feature alignment for face anti-spoofing,” Sensors, vol. 23, no. 8, p. 4077, 2023.

[49] C.-H. Liao, W.-C. Chen, H.-T. Liu, Y.-R. Yeh, M.-C. Hu, and C.-S. Chen, “Domain invariant vision transformer learning for face anti-spoofing,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 6098–6107.

[50] Z.-W. Hong, Y.-C. Lin, H.-T. Liu, Y.-R. Yeh, and C.-S.

Chen, “Domain-generalized face anti-spoofing with unknown attacks,” in 2023 IEEE International Conference on Image Processing, 2023, pp. 820–824.

[51] X. Long, J. Zhang, S. Wu, X. Jin, and S. Shan, “Dual sampling based causal intervention for face anti-spoofing with identity debiasing,” IEEE Transactions on Information Forensics and Security, vol. 19, pp. 851–862, 2024.

[52] Q. Zhou, K.-Y. Zhang, T. Yao, X. Lu, S. Ding, and L. Ma, “Test-time domain generalization for face antispoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 175–187.

[53] Y. Liu, Z. Li, S. Li, Z. Wang, and L. Wu, “Face antispoofing via interaction learning with face image quality alignment,” in 2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG), 2024, pp. 1–10.

[54] Y. Liu, Z. Li, and L. Wu, “Dual consistency regularization for generalized face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 2171–2183, 2025.

[55] Y. Niu and X. Lin, “Similarity-aware contrastive learning for face anti-spoofing via frequency enhancement and reconstruction,” Neural Networks, vol. 199, p. 108734, 2026.

[56] H. Xing, S. Y. Tan, F. Qamar, and Y. Jiao, “Dual semantic consistency module for domain generalizable face antispoofing,” Journal of King Saud University - Computer and Information Sciences, vol. 38, p. 829, 2026.