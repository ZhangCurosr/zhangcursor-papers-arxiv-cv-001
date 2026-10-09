# ADAPTIVE ADVERSARIAL AUGMENTATION FOR CONTROLLABLE FACE SYNTHESIS

A PREPRINT

Saransh Suri<sup>1∗</sup> Shivang Agarwal<sup>2†</sup> Mayank Vatsa<sup>3</sup> Richa Singh<sup>3</sup>

<sup>1</sup>Manipal University Jaipur, India

<sup>2</sup>Birla Institute of Technology and Science Pilani Dubai Campus, Dubai, United Arab Emirates <sup>3</sup>Indian Institute of Technology Jodhpur, India

October 9, 2026

## ABSTRACT

Synthetic data provides a scalable alternative to real-world datasets for training face recognition models, particularly under challenging conditions such as low resolution, occlusion, and masks. Yet, most approaches lack diversity and fail to generalize effectively. We propose Ensemble Feedback Controllable Synthesis (EFCS), a guided framework that generates diverse and challenging samples while preserving visual realism. EFCS expands distributional variability, often reflected in higher FID and KID scores compared to single-feedback and random synthesis, while maintaining high precision. Recognition models trained on EFCS data consistently outperform baselines across multiple benchmarks, showing improved generalization to real-world scenarios. Furthermore, we introduce an analytically motivated formulation linking perturbation-induced difficulty, sample utility, and performance degradation, offering principled insights into balancing synthetic data complexity for optimal training. Together, these contributions establish EFCS as an effective and analytically grounded approach for bridging the gap between synthetic and real datasets.

Keywords Controllable synthesis · Face recognition · Responsible generation

## 1 Introduction

Face recognition (FR) systems have witnessed significant advancements, largely driven by the widespread availability of large-scale identity-labeled face datasets. Nevertheless, this rapid progress has concomitantly raised substantial concerns related to privacy, ethics, and inherent demographic biases arising from real identity data [1, 2, 3, 4]. To address these critical issues, synthetic face datasets have emerged as an attractive alternative for training FR models, offering considerable potential for privacy protection and bias mitigation [5, 6, 7]. Despite these promising attributes, FR models trained exclusively on synthetic data consistently underperform relative to those trained on real-world datasets [8]. This performance gap primarily arises from two fundamental challenges: insufficient intra-class variations within synthetic images and pronounced domain mismatches between synthetic and real data distributions [9, 7, 1]. These limitations adversely affect the generalization capabilities of FR models, particularly in complex scenarios such as low-resolution surveillance images [10], faces with occlusions or injuries [11], and cross-domain selfie versus ID verification [12], as highlighted in recent analyses of synthetic data utility and generalization [5, 1].These observations motivate the need for synthetic data generation strategies that explicitly optimize training utility rather than visual fidelity alone.

Current approaches for synthetic face generation exhibit significant limitations. Generative Adversarial Networks (GANs), despite their popularity, frequently fail to produce adequate intra-class diversity, resulting in synthetic datasets characterized by limited variations within individual identity classes [13, 14, 15]. This paucity of diversity often leads FR models trained on GAN-generated datasets to suffer from overfitting and reduced generalization. On the other hand, recent diffusion-based synthesis methods generate high-quality images but lack the fine-grained control necessary to produce targeted variations or precise manipulations of facial attributes [16, 17, 18, 19]. Moreover, existing methods generally overlook the critical domain gap, hindering effective transferability of synthetically-trained models to practical real-world applications [2, 1, 8, 20].

![](images/71017967ce2d89288c4cd464a1cd8c3d3e34da188247ba9215035d852ca6ce5f.jpg)  
Figure 1: Illustration highlighting the necessity of adaptive synthesis. Controlled perturbations (hard samples) introduce targeted complexity to synthetic data, enhancing model robustness beyond traditional synthesis methods.

![](images/40a3d92f75da2c288769fd80812c47e6e0cd0387ba32fa22b7024b0f6933a141.jpg)  
Figure 2: Qualitative comparison of synthetic samples: (I) constant degradation, (II) single feedback synthesis [21], and (III) proposed EFCS method.

To address these persistent challenges, this paper introduces a novel approach termed Ensemble Feedback Controllable Synthesis (EFCS). EFCS guides the synthetic data generation process by leveraging ensemble feedback from multiple pre-trained FR models, thus producing synthetic face images tailored specifically to enhance model training. The generated images are rich in intra-class and inter-class diversity. By generating diverse yet realistically challenging synthetic samples through controlled feedback mechanisms, EFCS enhances the robustness and generalization of FR models. Crucially, as shown in Figure 2, EFCS ensures that the generated samples remain appropriately challenging without introducing excessive difficulty that could cause the models to overfit irrelevant attributes, such as background or clothing [22].

Our contributions are threefold:

• We empirically demonstrate that FR models trained solely on synthetic data without guided feedback consistently fall short of real-data model performance, highlighting the necessity of a more sophisticated synthetic data generation strategy.

• We propose the Ensemble Feedback Controllable Synthesis (EFCS), a novel guided synthesis framework leveraging feedback from an ensemble of FR models to produce highly beneficial synthetic training samples.

![](images/74893ec767e7cb4338a388fe7d7428ad57473e57194f839af21dea8b718ed541.jpg)  
Figure 3: Overview of the proposed EFCS framework showing synthesis, perturbation, and ensemble feedback.

• We introduce a theoretical formulation that quantifies the relationship between perturbation-induced difficulty, sample utility, and performance degradation. This formulation provides insights into balancing the complexity of synthetic data for optimal FR model training.

## 2 Literature Review

Controllable face synthesis methods [21, 17, 23] have emerged as particularly promising avenues for generating diverse and high-quality synthetic face datasets. These methods offer fine-grained control over facial attributes via manipulation in the latent style space. Such attribute control is motivated primarily by the need to address privacy concerns linked to the use of real, identity-labeled datasets and to manage challenges arising from noisy labels and imbalanced demographic representation.

Among various synthesis methods, GANs have been central to synthetic face image generation [24, 13, 25]. For instance, SFace [10] employs StyleGAN2-ADA to create synthetic datasets closely aligned with real images through a 1:1 correspondence strategy. Despite this alignment, experimental evaluations demonstrate performance gaps, often exceeding 25% in verification accuracy, between models trained on synthetic data and those trained on real images. This gap highlights a significant limitation: synthetic data frequently fails to replicate the complexity, diversity, and subtle intra-class variations inherent to real-world images. To mitigate this, several approaches have been explored. SynFace [7] introduces a method of mixing intermediate latent states from synthetic identities to enhance intra-class diversity. Nevertheless, SynFace predominantly generates frontal-view faces, necessitating additional real-image mixing to adequately bridge the persistent domain gap between synthetic and real-world face images.

Recently, diffusion-based models have garnered increased attention as viable alternatives to GAN-based methods [18, 19, 26]. Diffusion models enable precise generation and manipulation of facial attributes guided by diverse style-based prompts, including text descriptions and sketches [27]. Despite providing greater control over specific facial features, diffusion models still face substantial challenges [9]. Primarily, they struggle to produce synthetic samples sufficiently diverse and challenging to robustly train FR models, thereby limiting the practical utility of generated datasets [28].

Despite significant methodological advancements, existing synthetic face generation techniques commonly encounter critical shortcomings in complex FR tasks, such as low-resolution surveillance face recognition, face recognition under varying pose conditions, and identification scenarios involving partially occluded or injured faces. The key limitations consistently observed across current methodologies include: (1) inadequate intra-class diversity within synthetic datasets, (2) enduring domain mismatches between synthetic and real-world image distributions, and (3) limited capability to generate sufficiently challenging samples that robustly enhance FR model training [29].

## 3 Ensemble Feedback Controllable Synthesis

Given an original face image x with true identity label $y ,$ the primary goal is to generate an adversarially perturbed synthetic face image $x _ { p e r t u r b }$ that significantly increases the difficulty of correct recognition by FR models, while maintaining realism and identity preservation. Unlike adversarial attacks designed to expose worst-case vulnerabilities at inference time, EFCS uses adversarial optimization as a training-time augmentation mechanism. The objective is not robustness against malicious perturbations, but controlled difficulty injection that improves representation learning. This goal is formally expressed as:

$$
x _ { p e r t u r b } = G ( x , p e r t u r b )\tag{1}
$$

where $G ( \cdot )$ denotes the controllable face synthesis model, and perturb represents the perturbation drawn from a structured latent space. Unlike random perturbations, this structured perturbation ensures semantic coherence and realistic transformations.

Let us consider an ensemble of N pre-trained FR models, denoted as $\{ f _ { 1 } , f _ { 2 } , \ldots , f _ { N } \}$ . Each model $f _ { i }$ has an associated loss function $L _ { i } ( f _ { i } ( x ) , y )$ that quantifies its performance in recognizing image x as identity y. Typical loss functions include cross-entropy or margin-based verification losses (e.g., ArcFace [30], AdaFace [22]).

The aggregated recognition difficulty posed by the ensemble on an image x is defined as the ensemble recognition loss:

$$
x _ { \mathrm { e n s } } = \arg \operatorname* { m a x } _ { i \in \{ 1 , \dots , N \} } L _ { i } { \big ( } f _ { i } ( G ( x , \mathrm { p e r t u r b } _ { i } ) ) , y { \big ) }\tag{2}
$$

Our adversarial objective is to maximize this ensemble recognition loss, thus explicitly making the synthesized image harder to recognize:

$$
\operatorname* { m a x } _ { p e r t u r b } L _ { \mathrm { e n s } } ( G ( x , p e r t u r b ) , y )\tag{3}
$$

To ensure controlled and meaningful perturbations, we introduce a perturbation depth parameter d that dynamically scales with the difficulty level of the input image as determined by the ensemble:

$$
d = \gamma \cdot L _ { \mathrm { e n s } } ( x , y )\tag{4}
$$

where $\gamma$ is a hyperparameter scaling factor. This depth controls the intensity of the perturbation, ensuring that harder samples (as per ensemble loss) receive appropriately stronger perturbations.

The perturbation vector perturb is sampled from a structured distribution conditioned on the perturbation depth d:

$$
p e r t u r b ^ { ( d ) } \sim \mathcal { P } ( d )\tag{5}
$$

To formally measure and maximize the training utility of the synthesized adversarial images, we define a training utility function $U ,$ , which aims for a targeted increase δ in ensemble recognition difficulty:

$$
U ( x _ { p e r t u r b ^ { ( d ) } } ) = - | L _ { \mathrm { e n s } } ( x _ { p e r t u r b ^ { ( d ) } } , y ) - ( L _ { \mathrm { e n s } } ( x , y ) + \delta ) |\tag{6}
$$

Maximizing $U$ ensures the generated synthetic images are neither too trivial nor excessively difficult, thereby optimally contributing to model training.

## 3.1 Proposed EFCS Architecture

The proposed EFCS method employs a sophisticated two-stage architecture to systematically generate and utilize adversarial synthetic images. The overall framework is illustrated in Figure 3.

The controllable synthesis architecture used in EFCS follows the generator design introduced in CFSM [21], which builds upon the encoder–decoder framework. In our framework, this architecture is adopted as the base generator to ensure stable identity-preserving synthesis.

The primary novelty of EFCS lies not in redesigning the generator architecture, but in the feedback-driven training mechanism applied to it. Specifically, EFCS introduces three key extensions beyond CFSM: (i) retraining the synthesis model itself using feedback signals, (ii) replacing single-model feedback with ensemble-based feedback from multiple face recognition models, and (iii) introducing a utility-driven objective that explicitly controls the difficulty of generated samples.

## 3.1.1 Stage 1: Controllable Synthesis Model

In Stage 1, a controllable synthesis model G is designed and trained to produce face images with explicitly controlled and interpretable variations. This model comprises three critical components:

Encoder: The encoder module captures identity-preserving content from input face image x. It employs reflective padding at the borders, convolutional layers for robust feature extraction, and stride-2 downsampling layers to progressively reduce spatial resolution. To further refine feature extraction, instance-normalized residual blocks are incorporated. These operations result in a compact and informative representation of essential identity features, discarding irrelevant attributes such as background details or varying illuminations.

Decoder: The decoder reconstructs realistic face images from the encoded content representations. It introduces a latent style vector, generated by a dedicated Multi-Layer Perceptron (MLP), to modulate intermediate activations via Adaptive Instance Normalization (AdaIN) layers. AdaIN enables granular and interpretable control over the reconstruction process, systematically introducing controlled perturbations. The decoder’s architecture includes residual blocks with AdaIN for style infusion, nearest-neighbor interpolation for efficient upsampling, and convolutional layers for fine-grained feature refinement. Finally, a Tanh activation layer ensures synthesized images remain within plausible and realistic pixel value ranges.

Subspace Layer: A specialized subspace layer ensures latent perturbations are diverse yet coherent. This layer enforces orthogonality constraints on latent dimensions, which correspond to interpretable and independent facial variations such as pose, illumination, expression, and occlusion. The orthogonality constraint prevents collapse to trivial perturbations, maintaining both diversity and semantic realism of the synthesized face images. Regularization techniques are applied during training to uphold this constraint, making the generated synthetic samples highly valuable and broadly applicable for FR model enhancement.

## 3.1.2 Stage 2: Adversarial Augmentation with Ensemble Feedback

In Stage 2, the controllable synthesis model G is integrated into an adversarial data augmentation pipeline driven by ensemble-based feedback. The goal of this stage is to generate synthetic adversarial images dynamically during the FR model training process, systematically targeting and challenging the weaknesses of the ensemble models.

Latent Initialization: Initially, a latent perturbation vector perturb is randomly sampled from the structured perturbation space. This vector dictates the initial set of transformations applied to the input face image, defining the baseline for subsequent optimization.

Feedback Loop and PGD Optimization: Ensemble feedback is systematically used through Projected Gradient Descent (PGD). For each iteration, the current synthetic image is evaluated by all ensemble models to compute their respective recognition losses. These computed losses are then used to iteratively update the latent perturbation vector perturb. Specifically, PGD iteratively updates the latent perturbation so as to increase the recognition loss of each ensemble model, after which the perturbation is refined through the ensemble feedback optimization loop. For each ensemble model $f _ { i } ,$ , a separate perturbation vector pertur $\cdot b _ { i }$ is optimized. Formally, each PGD step updates the latent perturbation as follows:

$$
\begin{array} { r l } & { \mathrm { p e r t u r b } _ { i } ^ { ( t + 1 ) } = \Pi _ { \mathcal { P } ( d ) } \Big ( \mathrm { p e r t u r b } _ { i } ^ { ( t ) } + \alpha \nabla _ { \mathrm { p e r t u r b } _ { i } } L _ { i } \big ( \quad } \\ & { \qquad f _ { i } \big ( G ( x , \mathrm { p e r t u r b } _ { i } ^ { ( t ) } ) \big ) , y \big ) \Big ) } \end{array}
$$

where α denotes the step size and $\operatorname { P r o j } _ { \mathcal { P } ( d ) }$ indicates projection back into the valid perturbation space. This refinement ensures the synthetic adversarial images progressively become more challenging while maintaining identity.

In practice, each recognition model provides an independent feedback signal during perturbation optimization. The resulting synthesized images obtained from different models are then aggregated to produce the final adversarial training sample.

$$
x _ { \mathrm { e n s } } = \operatorname* { m i n } _ { i \in \{ 1 , \dots , N \} } G ( x , \mathrm { p e r t u r b } _ { i } )\tag{7}
$$

This element-wise minimum aggregation selects the most adversarial response across the ensemble-generated samples.

Our implementation uses a single-step PGD update for perturbation initialization with $\epsilon = \alpha = 0 . 3 1 4$ . This step provides an initial direction that increases recognition difficulty. Subsequently, the perturbation is refined through the ensemble feedback loop, where the synthesis model and face recognition ensemble jointly adjust the perturbation during training.

Thus, while the initial perturbation follows a single-step PGD formulation (similar to FGSM), the overall optimization remains iterative due to the feedback-driven refinement across training iterations. This design was chosen for computational efficiency while still enabling difficulty-aware adversarial augmentation.

Generation of Refined Synthetic Samples: After completing the iterative PGD optimization process, the refined latent vector is used to synthesize updated adversarial face images $x _ { p e r t u r b }$ . These refined synthetic images specifically exploit and reveal vulnerabilities of the ensemble models, enhancing their robustness upon training.

Sample Selection and Balanced Batch Formation: To prevent redundancy and ensure effective training, synthetic adversarial images are systematically selected based on their collective difficulty across the ensemble. The selection criteria involve maximizing the overall ensemble difficulty, effectively focusing on samples posing diverse challenges to different ensemble components. Selected synthetic adversarial images are then methodically combined with real face images to form balanced mini-batches. This balanced training regime prevents overfitting to synthetic artifacts, ensuring robust generalization to real-world scenarios.

## 3.2 Analytical Perspective and Intuition

The behavior of EFCS can be understood from the perspective of adversarial robustness and distributional generalization. In standard empirical risk minimization (ERM), face recognition models minimize the expected loss over samples drawn from a data distribution D. EFCS introduces controlled adversarial perturbations during synthesis, effectively inducing a perturbed training distribution $\mathcal { D } _ { \mathrm { p e r t u r b } }$

This perturbed distribution can be expressed as

$$
\mathcal { D } _ { \mathrm { p e r t u r b } } ( x , y ) = \mathbb { E } _ { x ^ { \prime } \sim \mathcal { P } ( d ) } \left[ \mathbf { 1 } _ { \{ x ^ { \prime } = G ( x _ { \mathrm { p e r u r b } } ^ { d } ) \} } \cdot \mathcal { D } ( x , y ) \right] .
$$

The corresponding training objective under EFCS becomes

$$
\operatorname* { m i n } _ { f } \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \mathrm { p e r t u r b } } } \left[ L _ { \mathrm { e n s } } ( f ( x ) , y ) \right] ,
$$

which encourages the learned representation to remain stable under structured identity-preserving perturbations.

Under mild assumptions such as continuity of the loss function, smoothness of the data manifold, and bounded perturbations, this objective can be interpreted as a robust optimization problem of the form

$$
\operatorname* { m i n } _ { f } \ \operatorname* { m a x } _ { d \sim \mathcal { P } ( d ) } \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ L _ { \mathrm { e n s } } \big ( f \big ( G ( x _ { \mathrm { p e r t u r b } } ^ { d } ) \big ) , y \big ) \right] .
$$

Because the latent perturbation space is constrained to generate realistic variations, the resulting adversarial samples remain semantically meaningful and avoid trivial solutions. Overall, EFCS improves robustness by explicitly optimizing for challenging yet identity-consistent synthetic examples, thereby enhancing generalization to real-world conditions.

Ensemble feedback justification. Aggregating gradients across multiple recognition models reduces bias toward model-specific failure modes and provides a smoother estimate of the difficulty landscape, consistent with prior findings on ensemble robustness [31]. This promotes perturbations that generalize across architectures and loss formulations, preventing over-specialization to single-model adversarial artifacts and improving the training utility of synthesized samples [28].

## 4 Experimental Details

## 4.1 Evaluation Metrics

We evaluate the quality, realism, and utility of generated face images using several metrics directly connected to our EFCS objectives. Specifically, we report Fréchet Inception Distance (FID), Kernel Inception Distance (KID), precision, recall, Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and Deep Image Structure and Texture Similarity (DISTS).

Table 1: Dataset statistics used for EFCS training. Synthetic images are generated per identity using the EFCS synthesis model.
<table><tr><td>Dataset</td><td>Identities</td><td>Images/ID</td><td>Total Images</td></tr><tr><td>CASIA-WebFace (original)</td><td>10,575</td><td>~47 avg</td><td>494,414</td></tr><tr><td>Real-only subset</td><td>10,575</td><td>20</td><td>211,500</td></tr><tr><td>EFCS Synthetic</td><td>10,575</td><td>20</td><td>211,500</td></tr><tr><td>Mixed (Real + EFCS)</td><td>10,575</td><td>40</td><td>423,000</td></tr></table>

Table 2: Average SER-FIQ comparison across synthesis strategies (higher is better).
<table><tr><td>Sample Type</td><td> $\operatorname { A v g } .$  SER-FIQ</td></tr><tr><td>Original</td><td>0.7776</td></tr><tr><td>Synthesized</td><td>0.7338</td></tr><tr><td>Single Feedback [21]</td><td>0.7264</td></tr><tr><td>EFČS (ours)</td><td>0.7277</td></tr></table>

## 4.2 Implementation Details

Datasets and preprocessing. Experiments are conducted on CASIA-WebFace and WiderFace-12K. All face images are aligned using five landmarks and resized to 112 × 112. From the original CASIA-WebFace dataset (494,414 images, 10,575 identities), we construct balanced subsets by selecting 20 real images per identity and generating 20 EFCS samples. This yields real-only and EFCS datasets of 211,500 images each, and a mixed dataset of 423,000 images.

Recognition models. We evaluate robustness using three representative FR models: ArcFace [30], AdaFace [22], and ElasticFace [32], each employing distinct loss formulations.

Controllable synthesis. The synthesis network G, adapted from [21], produces identity-preserving adversarial variations conditioned on a latent perturbation vector and perturbation depth parameter.

Perturbation generation. Adversarial samples are generated using single-step PGD with perturbation bound ϵ = 0.314 and step size α = 0.314.

Training setup. Models are trained with an ArcFace-ResNet50 backbone using SGD (learning rate 0.1, momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 } )$ for 20 epochs with polynomial decay and a 2-epoch warmup.

Hardware. All experiments are performed on a single NVIDIA A40 GPU (48 GB).

## 5 Results and Analysis

To validate the proposed EFCS, we performed extensive quantitative evaluations on synthesized face images using a diverse array of metrics, chosen specifically to align with our analytical motivation and optimization goals. Table 2 reports the average SER-FIQ scores for different synthesis strategies. SER-FIQ reflects face image quality through embedding robustness and provides an indication of visual consistency in synthesized samples. Original images obtain the highest score (0.7776), while random-noise synthesis shows a modest reduction (0.7338). EFCS samples (0.7277) yield scores close to those of single-feedback synthesis (0.7264), suggesting broadly comparable perceptual quality. These results indicate that ensemble-guided perturbations introduce additional recognition difficulty while largely preserving identity-related visual cues.

Interpreting Metrics under Hard-Sample Synthesis. EFCS is designed to generate identity-preserving hard samples rather than visually identical replicas of real faces. Accordingly, metric shifts such as lower SSIM or higher FID/KID should be viewed as indicators of controlled distributional expansion and increased recognition difficulty, rather than reduced synthesis quality.

Table 3 supports this interpretation. EFCS samples show lower SSIM, reflecting reduced trivial structural similarity, while higher LPIPS (0.5922) and DISTS (0.3653) indicate perceptually meaningful variations introduced through ensemble-guided perturbations. These trends are consistent with the objective of producing challenging yet coherent training examples.

Table 3: Perceptual similarity comparison. EFCS produces harder samples, hence lower SSIM and higher LPIPS/DISTS.
<table><tr><td>Method</td><td>SSIM</td><td>LPIPS</td><td>DISTS</td></tr><tr><td>Baseline (no feedback)</td><td>0.4016</td><td>0.4859</td><td>0.3241</td></tr><tr><td>Single feedback [21]</td><td>0.3859</td><td>0.5020</td><td>0.3383</td></tr><tr><td>EFCS (ours)</td><td>0.2376</td><td>0.5922</td><td>0.3653</td></tr></table>

Table 4: Distributional evaluation (FID↓, KID↓, Prec↑, Rec↑).
<table><tr><td>Sample</td><td>FID</td><td>KID</td><td>Prec.</td><td>Rec.</td></tr><tr><td>Synthesized</td><td>57.395</td><td>0.0123</td><td>0.7266</td><td>0.5136</td></tr><tr><td>Single fb. [21]</td><td>63.837</td><td>0.0132</td><td>0.8121</td><td>0.4501</td></tr><tr><td>EFCS (ours)</td><td>73.947↑</td><td>0.0152↑</td><td>0.7566</td><td>0.3152</td></tr></table>

Distributional statistics in Table 4 further highlight this behavior. Compared with single-feedback and random synthesis, EFCS yields higher FID (73.947) and KID (0.0152), reflecting intentional expansion toward harder and underrepresented regions of the face manifold. Despite this shift, precision remains relatively high (0.7566), suggesting that generated samples retain substantial realism and identity consistency. Lower recall (0.3152) indicates that EFCS does not aim for uniform coverage of the real distribution, but instead concentrates on difficult regions that expose model vulnerabilities. This targeted diversity aligns with the goal of adversarial augmentation, where training utility is prioritized over distribution completeness.

Finally, Table 5 evaluates models trained on real, mixed, and EFCS-only datasets. Models trained solely on EFCS achieve strong performance on the EFCS test set (89.6% R1) and transfer competitively to real data (84.3% R1). Combining EFCS with real data yields the most robust models overall (e.g., 99.4% R1 on Mix and 99.2% on Real for Mixed-50%), whereas real-only training generalizes poorly to synthetic domains (73.1% R1 on EFCS). Importantly, unlike conventional generative modeling where lower FID/KID universally indicates better realism, EFCS intentionally expands the synthetic distribution toward harder and under-represented regions of the face manifold. An interesting finding is the performance gap between sequential finetuning, and mixed training. Sequential finetuning exposes the model to one domain at a time, causing strong representation drift and partial forgetting of previously learned features. When training first on EFCS (hard, perturbed samples) and then switching to real data or vice versa, the optimizer adapts to the most recent distribution, often overwriting features that were useful for the earlier domain. In contrast, mixed training provides simultaneous exposure to both real and hard synthetic variations, encouraging the model to learn domain-invariant and more stable identity cues. This joint optimization prevents collapse toward a single distribution and leads to better robustness and generalization. These results demonstrate that EFCS-generated data provides effective standalone supervision and further enhances generalization when used alongside real data.

## 5.1 Graphical Analysis and Detailed Observations

Figure 4 analyzes the effect of synthesis noise and target difficulty on training utility and recognition robustness. Utility exhibits a clear unimodal trend with respect to perturbation strength: very small noise provides limited benefit, while moderate perturbations generate informative hard samples that maximize utility. Beyond a critical noise level, excessive distortion reduces both utility and recognition performance. A similar bell-shaped relationship is observed for the desired difficulty parameter, confirming that intermediate difficulty yields the most effective supervision. The degra dation curve further shows that recognition performance remains stable under mild perturbations but drops sharpl after a threshold, indicating the transition from beneficial hard samples to unrealistic samples. Overall, these result validate the EFCS design objective of controllably balancing sample difficulty and identity preservation.

## 5.2 Human Evaluation Study

To complement the quantitative analysis, we conducted an ethics-approved human study with 63 participants of varying expertise in AI and computer vision (12.7% advanced, 38.1% intermediate, and 49.2% beginners). The study aimed to assess perceptual similarity and identity recognizability of synthesized faces generated using different feedback mechanisms.

Participants were shown reference images of a real identity alongside a generated sample under three synthesis settings: (i) random perturbation without feedback, (ii) single-model feedback (1FR), and (iii) the proposed ensemble feedback mechanism (3FR). They rated identity similarity on a 5-point Likert scale (1: very dissimilar, 5: highly similar) and additionally indicated whether the generated image corresponded to the same identity ({Yes, No, Not sure}). To examine perceived difficulty, participants also selected the most challenging yet recognizable sample from sets with progressively increasing noise. To minimize bias, image order and synthesis conditions were randomized, and participants were not informed of the underlying generation method. Responses were aggregated to compute average similarity scores and recognizability statistics.

Table 5: Core evaluation on custom test sets (R1/R5 and TAR@{1e-5,1e-4,1e-3}, %).
<table><tr><td>Model</td><td>Data</td><td>R1</td><td>R5</td><td>T@1e-5</td><td>T@1e-4</td><td>T@1e-3</td></tr><tr><td rowspan="3">EFCS-only</td><td>EFCS</td><td>89.6</td><td>93.0</td><td>52.4</td><td>69.0</td><td>80.5</td></tr><tr><td>Mix</td><td>95.3</td><td>96.7</td><td>49.2</td><td>63.7</td><td>76.0</td></tr><tr><td>Real</td><td>84.3</td><td>89.1</td><td>41.8</td><td>58.2</td><td>71.2</td></tr><tr><td rowspan="3">Mixed-50%</td><td>EFCS</td><td>98.9</td><td>99.3</td><td>94.9</td><td>97.0</td><td>98.2</td></tr><tr><td>Mix</td><td>99.4</td><td>99.6</td><td>95.9</td><td>97.4</td><td>98.5</td></tr><tr><td>Real</td><td>99.2</td><td>99.4</td><td>96.3</td><td>97.8</td><td>98.7</td></tr><tr><td rowspan="3">Mixed-33%</td><td>EFCS</td><td>89.0</td><td>92.8</td><td>52.5</td><td>69.0</td><td>80.3</td></tr><tr><td>Mix</td><td>96.1</td><td>97.2</td><td>69.8</td><td>78.8</td><td>86.1</td></tr><tr><td>Real</td><td>94.9</td><td>96.2</td><td>80.2</td><td>86.4</td><td>90.6</td></tr><tr><td rowspan="3">Mixed-25%</td><td>EFCS</td><td>84.4</td><td>89.9</td><td>40.9</td><td>59.1</td><td>84.4</td></tr><tr><td>Mix</td><td>95.4</td><td>96.7</td><td>61.3</td><td>72.8</td><td>95.4</td></tr><tr><td>Real</td><td>94.0</td><td>95.6</td><td>76.2</td><td>83.8</td><td>94.0</td></tr><tr><td rowspan="3">Real-only</td><td>EFCS</td><td>73.1</td><td>82.3</td><td>25.2</td><td>41.5</td><td>57.9</td></tr><tr><td>Mix</td><td>94.5</td><td>96.2</td><td>50.0</td><td>63.1</td><td>74.8</td></tr><tr><td>Real</td><td>94.5</td><td>95.8</td><td>74.5</td><td>84.1</td><td>89.5</td></tr><tr><td rowspan="3">Pretr.+EFCS</td><td>EFCS</td><td>96.4</td><td>97.2</td><td>88.4</td><td>91.7</td><td>94.5</td></tr><tr><td>Mix</td><td>97.9</td><td>98.4</td><td>82.3</td><td>87.4</td><td>97.9</td></tr><tr><td>Real</td><td>92.8</td><td>94.2</td><td>73.2</td><td>82.3</td><td>87.5</td></tr><tr><td rowspan="3">EFCS→Real ft.</td><td>EFCS</td><td>54.1</td><td>65.0</td><td>12.3</td><td>22.4</td><td>34.9</td></tr><tr><td>Mix</td><td>88.5</td><td>91.4</td><td>21.7</td><td>32.3</td><td>45.3</td></tr><tr><td>Real</td><td>74.2</td><td>81.4</td><td>27.2</td><td>40.8</td><td>54.8</td></tr><tr><td rowspan="3">Real→EFCS ft.</td><td>EFCS</td><td>65.1</td><td>74.7</td><td>19.0</td><td>31.4</td><td>45.7</td></tr><tr><td>Mix</td><td>86.7</td><td>90.0</td><td>16.9</td><td>27.7</td><td>41.3</td></tr><tr><td>Real</td><td>59.2</td><td>68.1</td><td>10.6</td><td>22.6</td><td>36.1</td></tr></table>

![](images/8ef28e20c4664684ecb508a4f4802b086ac040899330a8a8091ecfef504f395e.jpg)  
(a) Utility vs noise variance $\sigma ^ { 2 }$

![](images/c799e1e760bf086b963029644a42ff0ab35f0c958b6df56956270dcd3e163e3c.jpg)  
(b) Utility vs difficulty δ.

![](images/96b337f1fcbd763718246605e3af2ce39f253c5e9f2d12f0ee21c4c2b37f2fee.jpg)  
(c) Performance degradation vs $\sigma ^ { 2 } .$  
Figure 4: Effect of synthesis noise and difficulty on training utility and recognition performance. Results averaged across ArcFace, AdaFace, and ElasticFace on CASIA-WebFace. Moderate perturbation improves utility, while excessive noise causes degradation.

Across expertise levels, EFCS samples were consistently perceived as more challenging than those produced by ran dom perturbations or single-feedback synthesis while remaining identifiable. Advanced users typically assigned lower similarity scores (2.0–3.0), reflecting increased difficulty with preserved identity cues, whereas intermediate and beginner users reported slightly higher scores. Notably, about 65% of participants selected samples with intermediate noise levels as the hardest recognizable cases, suggesting that EFCS effectively controls perturbation strength to increase difficulty without compromising identity information. These findings support the intended objective of EFCS: generating identity-preserving yet challenging variations that can enhance the robustness of face recognition models.

Table 6: Training settings of evaluated models.
<table><tr><td>Model</td><td>Training strategy</td></tr><tr><td>M1 (EFCS)</td><td>Scratch on EFCS</td></tr><tr><td>M2 (Mix-50)</td><td>Scratch on EFCS+Real (50%)</td></tr><tr><td>M3 (Mix-33)</td><td>Scratch on EFCS+Real (33%)</td></tr><tr><td>M4 (Mix-25)</td><td>Scratch on EFCS+Real (25%)</td></tr><tr><td>M5 (Real)</td><td>Scratch on Real</td></tr><tr><td>M6 (Pretr.+EFCS)</td><td>Pretrained → finetune EFCS</td></tr><tr><td>M7 (EFCS→Real)</td><td>EFCS model → finetune Real</td></tr><tr><td>M8 (Real→EFCS)</td><td>Real model → finetune EFCS</td></tr></table>

Table 7: Verification performance (%) on six public benchmarks. Metrics: accuracy (Acc) and TAR@FAR.
<table><tr><td></td><td colspan="3">LFW</td><td colspan="3">CFP-FP</td><td colspan="3">AgeDB</td><td colspan="3">CALFW</td><td colspan="3">CPLFW</td><td colspan="3">CFP-FF</td></tr><tr><td>Model</td><td>Acc</td><td>1e-4</td><td>1e-3</td><td>Acc</td><td>1e-4</td><td>1e-3</td><td>Acc</td><td>1e-4</td><td>1e-3</td><td>Acc</td><td>1e-4</td><td>1e-3</td><td>Acc</td><td>1e-4</td><td>1e-3</td><td>Acc</td><td>1e-4</td><td>1e-3</td></tr><tr><td>M1</td><td>98.03</td><td>29.81</td><td>89.52</td><td>87.10</td><td>0.07</td><td>0.73</td><td>87.95</td><td>5.76</td><td>27.22</td><td>89.90</td><td>14.90</td><td>50.80</td><td>81.97</td><td>22.50</td><td>26.20</td><td>97.49</td><td>1.90</td><td>19.02</td></tr><tr><td>M2</td><td>98.68</td><td>29.15</td><td>92.33</td><td>90.57</td><td>0.18</td><td>1.80</td><td>90.70</td><td>9.56</td><td>33.13</td><td>91.28</td><td>6.02</td><td>51.70</td><td>84.65</td><td>5.65</td><td>37.70</td><td>98.51</td><td>3.25</td><td>32.53</td></tr><tr><td>M3</td><td>98.65</td><td>27.98</td><td>90.78</td><td>90.27</td><td>0.23</td><td>2.30</td><td>90.73</td><td>10.22</td><td>38.37</td><td>91.40</td><td>14.34</td><td>57.87</td><td>84.55</td><td>7.64</td><td>31.30</td><td>98.41</td><td>3.30</td><td>33.28</td></tr><tr><td>M4</td><td>98.83</td><td>30.39</td><td>94.97</td><td>90.21</td><td>0.52</td><td>5.20</td><td>91.00</td><td>6.04</td><td>43.03</td><td>92.03</td><td>15.51</td><td>54.53</td><td>84.20</td><td>6.32</td><td>29.18</td><td>98.29</td><td>3.40</td><td>33.95</td></tr><tr><td>M5</td><td>98.88</td><td>30.72</td><td>94.60</td><td>91.33</td><td>0.38</td><td>3.80</td><td>90.98</td><td>13.70</td><td>48.63</td><td>92.08</td><td>15.60</td><td>62.00</td><td>85.03</td><td>6.35</td><td>39.10</td><td>98.41</td><td>3.60</td><td>36.02</td></tr><tr><td>M6</td><td>99.38</td><td>36.60</td><td>98.60</td><td>95.17</td><td>0.33</td><td>3.30</td><td>93.85</td><td>13.90</td><td>52.90</td><td>93.38</td><td>20.60</td><td>68.10</td><td>88.33</td><td>11.80</td><td>46.20</td><td>99.14</td><td>4.47</td><td>44.70</td></tr><tr><td>M7</td><td>95.58</td><td>19.70</td><td>74.90</td><td>76.69</td><td>0.01</td><td>0.12</td><td>80.83</td><td>2.48</td><td>12.60</td><td>84.62</td><td>4.67</td><td>26.30</td><td>75.00</td><td>4.30</td><td>15.10</td><td>94.39</td><td>0.97</td><td>9.67</td></tr><tr><td>M8</td><td>93.63</td><td>16.90</td><td>61.90</td><td>72.23</td><td>0.00</td><td>0.00</td><td>69.43</td><td>0.29</td><td>1.90</td><td>81.12</td><td>2.81</td><td>16.70</td><td>71.02</td><td>1.13</td><td>8.20</td><td>91.17</td><td>0.55</td><td>5.53</td></tr></table>

## 5.3 Model Overview

To evaluate the impact of synthetic data on facial recognition model performance, we developed eight distinct models, summarized in Table 6. All models share a common architectural foundation, utilizing an iResNet-50 backbone with an ArcFace head to ensure a consistent basis for comparison. Our experiments explore various training methodologies: some models were trained from scratch using different datasets, while others were fine-tuned from pre-existing weights.

The core of our investigation lies in the ’Mixed eval’ models (M2, M3, M4), which were trained on hybrid datasets containing both real and synthetic images. The percentage specified in their names: 50%, 33%, and 25% denotes the proportion of synthetic data included in their respective training sets. This allows for a direct analysis of how the quantity of synthetic data influences model accuracy and robustness. The remaining models serve as crucial baselines, including models trained exclusively on synthetic data (M1), exclusively on real data (M5), and fine-tuned variants (M6, M7, M8) to assess the efficacy of transfer learning.

## 5.4 Analysis of Public Benchmark Performance

The performance of our models on six standard public benchmarks is detailed in Table 7. A key finding from these results is the competitive performance of the models trained with a combination of real and synthetic data (M2, M3, and M4) when compared to the model trained exclusively on real-world data (M5). Across all benchmarks, including LFW, CFP-FP, and AgeDB-30, the ’Mixed eval’ models consistently achieve metrics that are only marginally lower than the ’Real eval’ model. For instance, on the LFW benchmark, the accuracy of M2 (98.68%) and M4 (98.83%) is nearly identical to that of M5 (98.88%).

While the model trained on pure real data (M5) sets a high-performance baseline, the minimal difference in performance demonstrates the viability of synthetic data as a powerful supplement or even an alternative to real data. This slight trade-off in accuracy is highly acceptable when considering the immense benefits of synthetic data generation: the ability to create unlimited, diverse, and unbiased datasets on demand. This approach not only mitigates critical data privacy and security concerns associated with collecting real images but also offers a scalable solution to overcome data scarcity and systematically reduce the demographic biases often present in real-world datasets.

## 6 Conclusion

The proposed Ensemble Feedback Controllable Synthesis method advances synthetic data generation for face recognition by effectively balancing complexity with recognizability. EFCS surpasses traditional methods by creating synthetic samples specifically designed to enhance model robustness and generalization. Quantitative evaluations using metrics such as SSIM, LPIPS, and DISTS confirm EFCS’s capability to generate challenging yet authentic samples. The method addresses limitations in existing synthetic datasets and provides an analytical motivation for optimized noise injection. By producing controlled synthetic data that closely emulates real-world conditions, EFCS offers a robust and privacy-conscious alternative for training FR models.

## References

[1] Marco Huber et al. “Bias and Diversity in Synthetic-based Face Recognition”. In: IEEE/CVF Winter Conference on Applications ofComputer Vision. 2024.

[2] Adam Kortylewski et al. “Analyzing and Reducing the Damage of Dataset Bias to Face Recognition With Synthetic Data”. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019.

[3] Blaz Meden et al. “Privacy-Enhancing Face Biometrics: A Comprehensive Survey”. In: IEEE Trans. Inf. Forensics Secur. 16 (2021), pp. 4147–4183.

[4] Protection Regulation. “Regulation (EU) 2016/679 of the European Parliament and of the Council”. In: Regulation (eu) 679 (2016), p. 2016.

[5] Fadi Boutros et al. “Synthetic data for face recognition: Current state and future prospects”. In: Image Vis. Comput. 135 (2023), p. 104688.

[6] Ali Dabouei et al. “Fast Geometrically-Perturbed Adversarial Faces”. In: IEEE Winter Conference on Applications ofComputer Vision. 2019.

[7] Haibo Qiu et al. “SynFace: Face Recognition with Synthetic Data”. In: IEEE/CVF International Conference on Computer Vision. 2021.

[8] Bingyu Shen et al. “A Study of the Human Perception of Synthetic Faces”. In: IEEE International Conference on Automatic Face and Gesture Recognition. 2021.

[9] Haiyu Wu et al. “Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors”. In: The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. Open-Review.net, 2025. URL: https://openreview.net/forum?id=RoN6NnHjn4.

[10] Fadi Boutros et al. “SFace: Privacy-friendly and Accurate Face Recognition using Synthetic Data”. In: IEEE International Joint Conference on Biometrics. 2022.

[11] Chiranjeev Chiranjeev et al. “Beyond shadows and light: Odyssey of face recognition for social good”. In: Comput. Vis. Image Underst. 253 (2025), p. 104293.

[12] Shivang Agarwal et al. “Leveraging Synthetic Data and Hard Pair Mining for Selfie vs ID Face Verification”. In: IEEE International Joint Conference on Biometrics. 2023.

[13] Qi Wang et al. “Learning From Synthetic Data for Crowd Counting in the Wild”. In: IEEE Conference on Computer Vision and Pattern Recognition. 2019.

[14] Pietro Melzi et al. “GANDiffFace: Controllable Generation of Synthetic Datasets for Face Recognition with Realistic Variations”. In: IEEE/CVF International Conference on Computer Vision. 2023.

[15] Yujun Shen et al. “FaceID-GAN: Learning a Symmetry Three-Player GAN for Identity-Preserving Face Syn thesis”. In: IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[16] Darian Tomasevic et al. “ID-Booth: Identity-consistent Face Generation with Diffusion Models”. In: 19th IEEE International Conference on Automatic Face and Gesture Recognition, FG 2025, Tampa/Clearwater, FL, USA, May 26-30, 2025. IEEE, 2025, pp. 1–10. DOI: 10.1109/FG61629.2025.11099217. URL: https://doi. org/10.1109/FG61629.2025.11099217.

[17] Bahjat Kawar, Roy Ganz, and Michael Elad. “Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance”. In: Trans. Mach. Learn. Res. 2023 (2023).

[18] Minchul Kim et al. “DCFace: Synthetic Face Generation with Dual Condition Diffusion Model”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[19] Xihui Liu et al. “More Control for Free! Image Synthesis with Semantic Diffusion Guidance”. In: IEEE/CVF Winter Conference on Applications ofComputer Vision. 2023.

[20] Surbhi Mittal et al. “On responsible machine learning datasets emphasizing fairness, privacy and regulatory norms with examples in biometrics and healthcare”. In: Nat. Mac. Intell. 6.8 (2024), pp. 936–949.

[21] Feng Liu et al. “Controllable and Guided Face Synthesis for Unconstrained Face Recognition”. In: European Computer Vision Association. Ed. by Shai Avidan et al. 2022.

[22] Minchul Kim, Anil K. Jain, and Xiaoming Liu. “AdaFace: Quality Adaptive Margin for Face Recognition”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[23] Cuican Yu et al. “Towards High-Fidelity Text-Guided 3D Face Generation and Manipulation Using only Images”. In: IEEE/CVF International Conference on Computer Vision. 2023.

[24] Hatef Otroshi-Shahreza, Anjith George, and Sébastien Marcel. “SynthDistill: Face Recognition with Knowledge Distillation from Synthetic Data”. In: IEEE International Joint Conference on Biometrics. 2023.

[25] Fadi Boutros et al. “Unsupervised Face Recognition using Unlabeled Synthetic Data”. In: IEEE International Conference on Automatic Face and Gesture Recognition. 2023.

[26] Sudipta Banerjee et al. “Identity-Preserving Aging of Face Images via Latent Diffusion Models”. In: IEEE International Joint Conference on Biometrics. 2023.

[27] Md Mahedi Hasan, Shoaib Meraj Sami, and Nasser M. Nasrabadi. “Text-Guided Face Recognition using Multi Granularity Cross-Modal Contrastive Learning”. In: IEEE/CVF Winter Conference on Applications of Com puter Vision. 2024.

[28] Zhonghua Zhai et al. “Demodalizing Face Recognition with Synthetic Samples”. In: AAAI Conference on Artificial Intelligence. 2021.

[29] Yu Deng et al. “Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[30] Jiankang Deng et al. “ArcFace: Additive Angular Margin Loss for Deep Face Recognition”. In: IEEE Trans. Pattern Anal. Mach. Intell. 44.10 (2022), pp. 5962–5979.

[31] Florian Tramèr et al. “Ensemble Adversarial Training: Attacks and Defenses”. In: 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL: https://openreview.net/forum?id=rkZvSe-RZ.

[32] Fadi Boutros et al. “ElasticFace: Elastic Margin Loss for Deep Face Recognition”. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2022.