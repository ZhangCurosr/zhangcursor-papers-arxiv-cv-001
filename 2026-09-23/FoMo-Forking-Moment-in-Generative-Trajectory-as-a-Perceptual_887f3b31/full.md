# FoMo: Forking Moment in Generative Trajectory as a Perceptual Distance

Jaihyun Lew<sup>1</sup>, Mingi Jung<sup>2</sup>, Minjun Park<sup>1</sup>, Wooseok Song<sup>2</sup>, Sungroh Yoon<sup>1,2,3,∗</sup> <sup>1</sup>Interdisciplinary Program in AI, Seoul National University <sup>2</sup>Department of Electrical and Computer Engineering, Seoul National University <sup>3</sup>AIIS, ASRI, INMC, and ISRC, Seoul National University {fudojhl, 2019-16552, minjunpark, cody1129, sryoon}@snu.ac.kr

## Abstract

Reference-based image quality assessment (IQA) metrics aim to reflect how humans perceive the perceptual distance between a pair of images. To learn how the human visual system (HVS) operates, recent reference-based IQA metrics heavily rely on human-annotated data. Mean opinion score (MOS)-based pointwise scoring, which assigns a scalar quality value per image, is preferable for annotation but is prohibitively expensive to collect at scale and is known to be noisy due to inconsistent human judgments. As an alternative, two-alternative forced choice (2AFC) pairwise labels have gained popularity due to their reliability and efficiency, but they capture only relative comparisons between pairs. In this paper, we propose a fully automated data generation pipeline that generates pointwise perceptual distance labels between image pairs without any human annotation. Our approach exploits the generative dynamics of diffusion models as a perceptual distance proxy, where the coarse structure of an image is generated in the early timesteps and the fine details are generated in the later timesteps. Images that fork early in the generation process share only coarse structure and are perceptually far apart; images that fork late differ only in fine detail. We demonstrate that the diffusion trajectory aligns well with the human visual system, and use this forking moment, FoMo, as a reference-grounded distance label to supervise the training of a reference-based IQA metric. The pointwise labels, which support universal comparison between arbitrary image pairs, enable an information-rich training objective. Extensive experiments across diverse backbone architectures confirm the effectiveness of our generation pipeline, outperforming human-annotated datasets in multiple benchmarks. Codes are publicly available at: https://github.com/JHLew/FoMo

## 1 Introduction

Reference-based image quality assessment (IQA) metrics [9, 13, 31, 43, 41] aim to quantify the perceptual difference between a reference image and its distorted counterpart. In particular, they play a central role in diverse image restoration tasks, such as super-resolution and denoising, where the goal is to compute the distance between a restored image and a reference target image. A reliable metric must therefore align closely with human perceptual judgments, not only distinguishing which of two images is closer to the reference, but also inducing a globally consistent ordering across diverse distortion types and severity levels.

To train such metrics, IQA datasets [43, 31, 23]have employed different forms of human supervision to approximate perceptual rankings. Mean opinion score (MOS) [30, 23] is the most direct and standard approach to produce such globally ordered labels. Multiple observers independently rate each distorted image on an absolute quality scale; the mean score is used as ground truth, and its global structure naturally supports rank-correlation training objectives [30]. However, reliable MOS collection is both expensive and fragile. Due to annotator bias and inter-session inconsistency, a large number of responses per image is required to suppress noise. KADID-10k [23], one of the largest MOS-annotated reference-based IQA datasets, required 30 crowdsourced ratings per image from over 2,200 subjects to produce 10,125 distorted images derived from only 81 reference images. Furthermore, labels are known to be inconsistent across datasets: the same distorted image can receive substantially different scores across datasets, because no common perceptual reference point exists across them. [31]

These difficulties have driven the field toward pairwise preference labeling. The two-alternative forced choice (2AFC) protocol, asking annotators which of two distorted images is more similar to a reference, is considerably more reliable than absolute rating, as relative judgments are less susceptible to individual-scale biases [31, 43]. BAPPS [43] and PieAPP [31] established 2AFC as the foundation for learning modern perceptual metrics, and both demonstrate that pairwise labels exhibit higher inter-annotator agreement than MOS under equivalent collection conditions. 2AFC has since become the dominant annotation paradigm for learning-based IQA. Yet, pairwise preference labels carry a structural limitation: a set of binary pairwise outcomes does not directly encode a global ordering, and global ranking of images are never taken into account in training. This tension between the practical tractability of pairwise labeling and the global ranking objective of IQA is a recognized open challenge [38, 5]. Ideally, one would have access to dense labels that directly support optimization toward global rank-correlation. In practice, however, this remains infeasible at scale under human annotation: collecting clean and consistent pointwise quality signals across thousands of images, while controlling for the annotator noise endemic to MOS, is prohibitively expensive.

In this paper, we propose to circumvent this bottleneck through an automated dataset generation pipeline grounded in the generative dynamics of diffusion models. [15, 36] Our key insight is that the generative dynamics of diffusion models provide a natural proxy for perceptual distance. During generation, coarse image structure is established in the early timesteps, while fine-grained details are resolved only in later timesteps. In this work, we define a fork as a controlled branching of the denoising process: two samples follow an identical trajectory up to a selected timestep and are then generated independently thereafter. Images that fork early in the generation process share only coarse structure and are perceptually far apart, whereas images that fork late differ primarily in fine detail. We use this forking moment, FoMo, as a reference-grounded distance label for training a reference-based IQA metric.

To validate this intuition, we conduct empirical analyses using a controlled set of image pairs generated via forking moments. We verify that their induced perceptual ordering aligns with human judgments, providing empirical grounding for using diffusion dynamics as a perceptual proxy. Building on this validation, we propose a fully automated and reference-grounded data generation pipeline that derives pointwise perceptual distance labels from diffusion forking moments. This enables global comparison across arbitrary image pairs without any human annotation, crowdsourcing infrastructure, or inter-annotator reconciliation.

Since FoMo is a pointwise score, it directly supports training consistent global ranking rather than binary pairwise preference. We extensively validate the effectiveness of our method across multiple benchmarks and diverse architectural backbones, from CNN-based models [19] to Transformer-based models. [40] In particular, our method outperforms human-annotated approaches, including KADID-10k’s MOS labels, showing that automated diffusion-based labels can surpass large-scale human annotation as a strong and reliable training signal. These results demonstrate that the FoMo can serve as a scalable, annotation-free alternative to both MOS and pairwise human labeling paradigms. Our key contributions are summarized as below:

• We propose a fully automated and annotation-free data generation pipeline for referencebased IQA that derives pointwise perceptual distance labels from the forking moments of diffusion trajectories, eliminating the need for human annotation.

• We demonstrate that diffusion generative dynamics encode perceptual distance in a manner well aligned with human visual judgment, providing a scalable alternative to MOS and pairwise preference annotations.

• We show that FoMo supervision enables globally consistent ranking across distorted images, and training with a RankNet-style [2] global objective improves reference-based IQA performance across diverse benchmarks and model architectures.

## 2 Preliminary

Diffusion Models, Flow Matching, and Rectified Flows. Denoising diffusion probabilistic models (DDPM) [15] define a forward Markov process that gradually corrupts a clean image $x _ { 0 }$ by adding Gaussian noise over $T$ discrete timesteps, yielding a sequence of increasingly noisy images $x _ { 1 } , x _ { 2 } , \ldots , x _ { T }$ where $x _ { T } \sim \mathcal { N } ( 0 , I )$ . A neural network is trained to reverse this process, iteratively denoising $x _ { T }$ back to a clean sample $x _ { 0 } .$ . Score-based generative models [36] generalize this to a continuous-time stochastic differential equation (SDE) framework, unifying many discrete diffusion variants under a single formalism. Flow matching [24] and rectified flows [25] instead parameterize the generative process as a probability flow ODE along linear interpolations between data and noise: $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon ,$ where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $t \in [ 0 , 1 ]$ . The resulting trajectories are straighter and more sample-efficient, motivating their adoption in state-of-the-art models such as FLUX [20].

Despite differences in trajectory geometry and training formulation, all three families share the same fundamental structure: a forward process that progressively destroys image information from fine detail toward coarse structure, and a learned reverse process that recovers the image from a stochastically sampled intermediate state. FoMo is grounded in this shared structure. Throughout this paper, we describe our method in the continuous-time SDE framework, as it provides the most general formulation. Most experiments in this paper are conducted with FLUX, which uses a rectified flow formulation where forward corruption follows $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon$

Perceptual Structure Along the Generative Trajectory A key premise of FoMo is that the generative trajectory encodes perceptual information in a structured, timestep-dependent manner. Choi et al. [6] provide a direct characterization: at low noise levels (high signal-to-noise ratio), the reverse process recovers imperceptible fine-grained details; at intermediate noise levels, it reconstructs perceptually rich and discriminative content such as object structure and texture; at high noise levels, it recovers only coarse global attributes such as color distribution. This stratification is not incidental, it is a structural consequence of the forward process, which destroys information roughly monotonically from high-frequency to low-frequency.

These observations directly motivate FoMo. If two images share a reverse trajectory up to t and then diverge via independent re-sampling, the perceptual content preserved up to t is shared between them, while content destroyed before t is independently regenerated. A late divergence (small t, little corruption) leaves most perceptual detail intact, yielding a perceptually close pair. An early divergence (large t, heavy corruption) destroys most discriminative content before re-sampling, producing a substantially different pair.

Trajectory Divergence as a Structural Prior The use of intermediate trajectory states to induce structured variation in generated outputs has appeared across several independent lines of work, lending support to the generality of the forking moment. [28, 33, 7] Most directly, Decatur et al. [7] demonstrate that when generating a collection of semantically related images, early denoising steps capture shared structure across similar prompts and need only be computed once; trajectories then branch independently from a later timestep onward. This explicitly instantiates a tree-structured forking process, and their findings confirm that the branching timestep controls the degree of visual similarity among the resulting outputs. FoMo builds on this foundation by converting the forking structure into an explicit, scalable source of perceptual distance labels, replacing human annotation with the generative process itself.

## 3 Empirical Grounding

Central to our approach is the hypothesis that the point of divergence in the diffusion sampling trajectory can serve as a meaningful perceptual distance label between image pairs. Before formalizing this as a metric, we first evaluate whether the divergence point serves as a reliable proxy for human perceptual similarity. That is, whether images forked earlier in the denoising process are consistently perceived as less similar to the reference than those forked later.

![](images/d7e6ab1491c5269ecd34e64b02097c2a9e451c5232cc78b2b4b51e80a23ea442.jpg)  
Figure 1: (a) Single-reference study: participants rank synthesized variants to verify whether the divergence timestep in diffusion generative trajectory reflects the degree of perceptual similarity to the reference image. (b) Cross-reference study: participants are asked which of two pairs, built from different references, holds the two more similar images. Further examples in the Appendix.

## 3.1 Single-Reference Validation

Study Design To verify whether the divergence timestep provides perceptually meaningful guidance, we conducted a human study examining whether variants that diverge later in the denoising trajectory are consistently perceived as more similar to a reference image than those branching at earlier timesteps. For each reference image, we synthesized five variants by injecting Gaussian noise at five distinct timesteps and denoising from each of them, so that later injection timesteps correspond to smaller perturbations and higher expected perceptual similarity to the reference. Participants were shown a reference image alongside its five variants and were asked to rank the variants from most to least similar compared to the reference. The workflow of this study is illustrated in Fig. 1a.

Results According to our experimental analysis, human perceptual judgments demonstrate strong alignment with the divergence timestep ordering, yielding a Spearman rank correlation of 0.970 across 30 participants and 1,982 responses collected over 190 reference images sampled from the ImageNet [8] validation set. Given an inter-rater correlation of 0.960, this level of agreement suggests that observer judgments are both consistent and well-structured. Collectively, these results indicate that the diffusion forking timestep constitutes a reliable proxy for perceptual similarity, providing empirical grounding for its adoption as a distance label in the subsequent metric formulation.

## 3.2 Cross-Reference Validation

Study Design The study above establishes that the forking timestep orders variants of a single reference consistently with human perception. A perceptual distance, however, should also be globally consistent: if an image pair is labeled to be closer than another, they should look closer, regardless of which reference image anchors each pair. We therefore ran a second study in the strict two-alternative forced-choice (2AFC) format. Each item shows two reference: variant pairs built from two distinct references and asks which pair contains the images more similar to each other. Since each forking timestep is chosen before its variant is generated, the two timesteps alone determine which pair our label calls closer, and no human answer enters the label. How hard an item is depends on the gap between its two forking timesteps: a small gap means both pairs were forked at nearly the same point, so they are almost equally similar, whereas a large gap sets a barely altered pair against a heavily altered one. We constructed 250 items, stratified into five bins of 50 by this gap, used each reference in at most one item, randomized left/right placement per participant, and collected 5,713 responses from 28 participants. Example questions from this study are in Fig. 1b.

Results Human choices agree with the ordering induced by the forking timesteps in 90.2% of individual responses, with a Fleiss’ κ [12] of 0.82 indicating almost perfect inter-rater reliability [21]. Because every item was judged by many participants, we can also ask what they concluded collectively rather than one response at a time: taking the majority answer for each item, the label agrees with the human consensus on 92.8% of items. Agreement rises monotonically with the gap: 64.3% for gaps of 1–10 timesteps, then 90.0% for gaps of 11–20, 97.0% for 21–30, and 99.5% and 99.8% for 31–40 and 41–50. Counting consensus rather than individual votes, the share of items whose majority answer matches the label runs 69.4%, 94.0%, 100%, 100% and 100% across the same five bins: beyond a gap of 20 timesteps, every item is decided the way the label predicts. Where the label agrees with people least, people also agree least with one another: in the narrowest bin two randomly chosen participants give the same answer on only $7 5 . 4 \%$ of items and just 20% of items are decided unanimously $( \kappa = 0 . 5 0 )$ , against 99.6%, 96% and $\kappa = 0 . 9 9$ in the widest. The forking timestep therefore induces a similarity ordering that holds across different reference images, breaking down only where the two labels are too close to call.

![](images/871032325820b1b582252a8d9fd90c6106bd84644d4bdd5fc8c73408fe2c468b.jpg)  
Figure 2: Visualization of our overall framework. (Left) Data generation trajectories, where the divergence point of the denoising trajectory (solid arrows) serves as a perceptual distance proxy. (Right) Unlike the traditional 2AFC framework, limited to within-anchor comparisons, our pointwise scoring system enables inter-anchor comparisons.

Validation at Scale Since it is extremely difficult to conduct these comparisons at scale by human annotation, we repeat the same test with established perceptual metrics standing in for the human observer. LPIPS-Alex [43], LPIPS-VGG [43], DISTS [9] and DreamSim [13] are all fitted to human judgments and widely used as proxies for them, which makes them a reasonable substitute here. We take 20,000 reference–variant pairs from our generated data and measure the distance each metric assigns to every pair. Pooling them into a single ranked list means that, as in the study above, almost every comparison is between pairs built on different references. Against that pooled ranking the forking label reaches a Spearman Rank Order Correlation Coefficient (SROCC) of 0.932 with LPIPS-Alex, 0.915 with LPIPS-VGG, 0.904 with DISTS and 0.904 with DreamSim. The ranking departs from the label only where the raters also hesitated, at near-ties. Once the two forking timesteps differ by more than 20 steps, the metrics agree with the label over 99.4% of the time. A pooled correlation of this magnitude is attainable only if the label is comparable across reference images rather than merely monotone within each one. The forking timestep behaves as a globally consistent distance label, not just a per-reference ranking.

## 4 Method

Based on the validation of Section 3, we present our dataset generation pipeline with fully automated labeling, and detail the training procedure for learning a perceptual distance metric from it.

## 4.1 Dataset Construction

For data generation, we use FLUX.1-dev [20] for its high-quality synthesis capability and broad coverage of visual content diversity. Given a reference image $x _ { 0 } ,$ , we aim to generate a perturbed variant paired with a distance label that is precise and requires no human annotation.

Assume a denoising process with S total steps. We uniformly sample a forking step $s \in [ 0 , S - 1 ]$ and obtain the corresponding interpolation factor from a predefined noise schedule S, i.e., $t _ { s } = S ( s )$ We then construct a noisy latent as

$$
\begin{array} { r } { x _ { t _ { s } } = ( 1 - t _ { s } ) x _ { 0 } + t _ { s } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{1}
$$

Starting from $x _ { t _ { s } } .$ , we perform the remaining $S - s$ denoising steps to obtain a perturbed image variant $x _ { 0 } ^ { s }$ . This yields a labeled pair $( x _ { 0 } , x _ { 0 } ^ { s } , t _ { s } )$ , where $t _ { s }$ serves as the distance label. Intuitively, a larger $t _ { s }$ corresponds to a higher noise level at the forking point and therefore to a greater perceptual deviation from the reference image. Although stochasticity is inherent to the diffusion process, the automated nature of label generation enables large-scale sampling, reducing label variance and leading to stable convergence (See Sec. G of Appendix.). Data samples from our constructed dataset provided in Fig. 3.

![](images/b19204677ca7fb387b2def7e5ed30a4c007dae56f7de6bded8209d11ed514749.jpg)  
Figure 3: Data sample image pairs and their labels from our constructed dataset. s denote the forking step (smaller-the-further), or the distance label of the image with respect to the reference image.

## 4.2 Objective Function

Conventional perceptual metrics such as LPIPS are trained on human-annotated 2AFC datasets, where each label encodes a relative preference between two distorted images given a shared reference. This relative structure constrains the loss to triplet-wise comparisons: given a triplet $( x _ { 0 } , x _ { 0 } ^ { \prime } , x _ { 0 } ^ { \prime \prime } )$ binary cross-entropy is applied to the predicted probability that one variant is closer to the reference than the other.

Our labels, by contrast, are pointwise: each pair $( x _ { 0 } , x _ { 0 } ^ { s } )$ ) carries an independent distance value $t _ { s } .$ without requiring a shared anchor for comparison. This permits a more expressive training objective. Specifically, for a batch of B pairs with predicted distances $\{ \hat { d } _ { i } \} _ { i = 1 } ^ { B }$ and labels $\{ t _ { s } ^ { i } \} _ { i = 1 } ^ { B }$ , we define a ground-truth comparison matrix $Y = [ y _ { i j } ]$ as:

$$
y _ { i j } = { \left\{ \begin{array} { l l } { \mathbb { 1 } [ t _ { s } ^ { i } < t _ { s } ^ { j } ] } & { t _ { s } ^ { i } \neq t _ { s } ^ { j } , \quad i , j \in \{ 1 , \dots , B \} } \\ { 0 . 5 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{2}
$$

where $y _ { i j } = 1$ indicates that pair i has a smaller true distance than pair j. We then apply binary cross-entropy loss over all $B \bar { \times } B$ comparisons:

$$
\mathcal { L } = - \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \left[ y _ { i j } \log \sigma ( \hat { d } _ { j } - \hat { d } _ { i } ) + \mathbf { s } \mathbf { g } ( ( 1 - y _ { i j } ) \log \sigma ( \hat { d } _ { i } - \hat { d } _ { j } ) ) \right] ,\tag{3}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function, and sg stands for stop-gradient operation. This formulation, originally proposed for learning-to-rank in RankNet [2], is here adapted to perceptual distance learning, supervising the global ordering of distances across all $B \times B$ pairs rather than within isolated triplets. It makes full use of the pointwise label system from our data generation process.

## 5 Experiments

## 5.1 Experimental Settings

We use a maximum of $S = 5 0$ sampling steps with FLUX, and the forking moment is sampled from a uniform distribution: $s \sim U [ 0 , 4 9 ]$ . For training, we generate a total of 480k pairs and labels. Of these, 240k pairs use real images sampled from the ImageNet database [8] as references, and the other 240k pairs use synthetic images generated by FLUX as references. We mix these two domains of reference images in order to ensure a good coverage of real-world and synthetic domains.

We experiment both on models with CNN backbones and Transformer backbones. For CNN backbones, we experiment with the well-established LPIPS [43] and DISTS [9] backbones. For Transformer backbones, we base upon the strong foundational models, DINOv3 [35], CLIP [32] and MAE [14], along with a recent DreamSim [13] backbone. Following the experimental setting from Zhang et al. [43], we use the feature extractors fixed to the pre-trained state, and train the prediction head from scratch, except for DreamSim, which tunes the feature extractor with LoRA [17] in its specific setting. All evaluations are conducted under the native resolution of each image, except for DreamSim, which enforces 224 resolution images by resizing the input at all times. All experiments reported in this paper were trained with models with approximately 240k iterations of updates, using a single NVIDIA RTX A40 GPU. All quantitative results in the main manuscript is reported by Spearman Rank Order Correlation Coefficient (SROCC), unless otherwise mentioned. Further details on experimental setting are described in the Appendix.

## 5.2 Quantitative Evaluations

Table 1: Comparison of training datasets and their objectives on four reference-based IQA benchmarks, across CNN- and Transformer-based backbones. Mean SROCC over five random seeds; standard deviations are given in Table 6. <sup>†</sup> Evaluated under 224 resolution.
<table><tr><td rowspan="2">Train Data</td><td rowspan="2">Human Annot.</td><td rowspan="2">Objective</td><td colspan="3">CNN-based</td><td colspan="4">Transformer-based</td><td rowspan="2">Avg.</td></tr><tr><td>LPIPS-Alex</td><td>LPIPS-VGG</td><td>DISTS</td><td>DINOv3</td><td>CLIP</td><td>MAE</td><td>DreamSim†</td></tr><tr><td colspan="10">Results on PIPAL [18]</td></tr><tr><td>BAPPS [43]</td><td>√</td><td>2AFC</td><td>0.622</td><td>0.634</td><td>0.489</td><td>0.287</td><td>0.151</td><td>0.338</td><td>0.760</td><td>0.469</td></tr><tr><td>PieAPP [31]</td><td>√</td><td>2AFC</td><td>0.602</td><td>0.635</td><td>0.564</td><td>0.325</td><td>0.258</td><td>0.294</td><td>0.711</td><td>0.484</td></tr><tr><td>NIGHTS [13]</td><td></td><td>2AFC</td><td>0.577</td><td>0.545</td><td>-0.294</td><td>0.234</td><td>0.277</td><td>0.234</td><td>0.662</td><td>0.319</td></tr><tr><td>KADID-10K [23]</td><td>√</td><td>L1</td><td>0.577</td><td>0.590</td><td>0.518</td><td>0.298</td><td>0.312</td><td>0.253</td><td>0.617</td><td>0.452</td></tr><tr><td>FoMo (Ours)</td><td>x</td><td>RankBCE</td><td>0.733</td><td>0.683</td><td>0.615</td><td>0.699</td><td>0.644</td><td>0.632</td><td>0.776</td><td>0.683</td></tr><tr><td colspan="10">Results on TID2013 [30]</td></tr><tr><td>BAPPS [43]</td><td>√</td><td>2AFC</td><td>0.779</td><td>0.671</td><td>0.632</td><td>0.341</td><td>0.285</td><td>0.287</td><td>0.813</td><td>0.544</td></tr><tr><td>PieAPP [31]</td><td>√</td><td>2AFC</td><td>0.761</td><td>0.722</td><td>0.680</td><td>0.315</td><td>0.302</td><td>0.467</td><td>0.767</td><td>0.573</td></tr><tr><td>NIGHTS [13]</td><td></td><td>2AFC</td><td>0.779</td><td>0.653</td><td>-0.497</td><td>0.230</td><td>0.294</td><td>0.358</td><td>0.762</td><td>0.368</td></tr><tr><td>KADID-10K [23]</td><td>√</td><td>L1</td><td>0.793</td><td>0.757</td><td>0.834</td><td>0.613</td><td>0.519</td><td>0.595</td><td>0.788</td><td>0.700</td></tr><tr><td>FoMo (Ours)</td><td>x</td><td>RankBCE</td><td>0.785</td><td>0.663</td><td>0.691</td><td>0.713</td><td>0.737</td><td>0.644</td><td>0.801</td><td>0.719</td></tr><tr><td colspan="10">Results on CSIQ [22]</td></tr><tr><td>BAPPS [43]</td><td>√</td><td>2AFC</td><td>0.943</td><td>0.887</td><td>0.855</td><td>0.411</td><td>0.427</td><td>0.455</td><td>0.911</td><td>0.699</td></tr><tr><td>PieAPP [31]</td><td>√</td><td>2AFC</td><td>0.936</td><td>0.896</td><td>0.821</td><td>0.331</td><td>0.514</td><td>0.587</td><td>0.903</td><td>0.713</td></tr><tr><td>NIGHTS [13]</td><td></td><td>2AFC</td><td>0.945</td><td>0.862</td><td>-0.560</td><td>0.335</td><td>0.364</td><td>0.427</td><td>0.867</td><td>0.463</td></tr><tr><td>KADID-10K [23]</td><td>√</td><td>L1</td><td>0.935</td><td>0.895</td><td>0.938</td><td>0.616</td><td>0.604</td><td>0.698</td><td>0.874</td><td>0.794</td></tr><tr><td>FoMo (Ours)</td><td>x</td><td>RankBCE</td><td>0.938</td><td>0.859</td><td>0.918</td><td>0.811</td><td>0.901</td><td>0.795</td><td>0.894</td><td>0.874</td></tr><tr><td colspan="10">Results on LIVE [34]</td></tr><tr><td>BAPPS [43]</td><td>√</td><td>2AFC</td><td>0.952</td><td>0.929</td><td>0.843</td><td>0.688</td><td>0.425</td><td>0.298</td><td>0.927</td><td>0.723</td></tr><tr><td>PieAPP [31]</td><td>√</td><td>2AFC</td><td>0.934</td><td>0.921</td><td>0.860</td><td>0.458</td><td>0.415</td><td>0.537</td><td>0.906</td><td>0.719</td></tr><tr><td>NIGHTS [13]</td><td></td><td>2AFC</td><td>0.949</td><td>0.919</td><td>-0.774</td><td>0.439</td><td>0.560</td><td>0.376</td><td>0.867</td><td>0.477</td></tr><tr><td>KADID-10K [23]</td><td>√</td><td>L1</td><td>0.942</td><td>0.912</td><td>0.950</td><td>0.765</td><td>0.827</td><td>0.797</td><td>0.896</td><td>0.870</td></tr><tr><td>FoMo (Ours)</td><td>x</td><td>RankBCE</td><td>0.948</td><td>0.923</td><td>0.954</td><td>0.896</td><td>0.925</td><td>0.907</td><td>0.931</td><td>0.926</td></tr></table>

The main quantitative result of our approach, in comparison to existing dataset and objectives, is presented in Table 1. Across four benchmarks and seven backbone architectures, FoMo achieves the best overall performance. On PIPAL [18], the least saturated and the most important benchmark, FoMo outperforms the existing datasets and their objectives, in all seven backbones by a large margin, especially in Transformer-based backbones. On the other three benchmarks, TID2013 [30], CSIQ [22] and LIVE [34], FoMo performs comparable to baselines on CNN backbones, and outperforms most of them in Transformer backbones. These results reflect the effectiveness of our data generation pipeline and objective function, despite being the only approach that does not require human annotation. The entire numbers containing Kendall rank-order correlation coefficient (KROCC) and Pearson linear correlation coefficient (PLCC) on these benchmarks can be found in the Appendix.

## 5.3 Ablation study

In this section, we ablate and analyze the experimental choices in our pipeline. First, we discuss on the experimental analysis on the label and objective function used in training. Second, as our method is based on a RankNet-style binary cross-entropy loss which incorporates the entire ranking within a batch, computing a $B \times B$ comparison matrix, the batch size is expected to be an important factor in our experiments, and we study on its effects. Third, we study to verify the generalizability of our approach, and check if the method could work with other diffusion models besides FLUX. Fourth, we study which timesteps are important in training, by dropping certain timestep ranges in training. Unless otherwise mentioned, we experiment on LPIPS-Alex and DINOv3 backbones, as representatives of each backbone styles, CNNs and Transformers. All the experiments in this section is evaluated with PIPAL benchmark, with SROCC scores.

Objective and label type We study on the choices on the objective function and the label types. In this experiment, we also experiment on KADID-10k dataset, for comprehensive analysis. First, our pipeline has two possible candidates of labels, using the interpolation factor t or the forking timestep s as the label, whereas KADID-10k labels are DMOS scores. The main difference with using t and step s as the label is resolution-variance. The main difference between using t and s as labels lies in their sensitivity to resolution changes, whether the label values are resolution-dependent. Refer to Sec. E of Appendix for further explanations. For each point-wise labels, we can apply three forms of objective function in training, ranked binary cross-entropy (RankBCE) loss, 2AFC-style pairwise binary cross-entropy loss, and a simple L1 regression loss on the label value. Our experimental results are presented in Table 2. According to our experiments, ranked binary cross-entropy loss as in RankNet [2] has clearly proved to be consistently superior than other objectives, supporting our claim on the importance of using rank-based objective. While direct DMOS regression is the default training convention [23, 9] for KADID-10k, our results show that a rank-based BCE objective is the stronger choice, and under this objective, FoMo still remains the better training source.

Table 2: Comparative experiment on the objective function and label type. Ranked binary crossentropy loss (Rank) consistently shows stronger results, compared to triplet-based paired cross-entropy loss (2AFC) and L1 regression loss on the label (L1). Our label has shown to be more helpful in training, compared to the large-scale human annotated KADID-10k. Mean over five random seeds; full numbers with standard deviations can be found in Table 9.
<table><tr><td></td><td colspan="6">FoMo (Ours)</td><td colspan="3">KADID-10K [23]</td></tr><tr><td>Label</td><td colspan="3">t</td><td colspan="3">S</td><td colspan="3">DMOS</td></tr><tr><td>Objective</td><td>Rank</td><td>2AFC</td><td>L1</td><td>Rank</td><td>2AFC</td><td>L1</td><td>Rank</td><td>2AFC</td><td>L1</td></tr><tr><td>LPIPS-Alex [43]</td><td>0.733</td><td>0.592</td><td>0.440</td><td>0.733</td><td>0.592</td><td>0.390</td><td>0.625</td><td>0.611</td><td>0.577</td></tr><tr><td>DINOv3 [35]</td><td>0.699</td><td>0.622</td><td>0.694</td><td>0.697</td><td>0.614</td><td>0.695</td><td>0.309</td><td>0.145</td><td>0.298</td></tr></table>

Analysis on Batch Size We study on the effects of batch size in our ranked binary cross-entropy loss. Pairwise ranking losses that operate over all within-batch pairs are known to benefit from large batch sizes, since a larger batch means more comparisons and thus richer supervision per training step, a property well-documented in contrastive learning [4, 32]. As shown in Table 3, we observe a similar trend. Performance improves with batch size increases, especially in Transformer-based DINOv3 backbone, whereas in CNN-based backbone LPIPS-Alex, the performance peaks at batch size of 64. This reflects the potential that our approach could be scaled more with larger batch size and larger models.

Table 3: Effect of batch size on training performance. The Transformer-based backbone improves monotonically with batch size, while the CNN-based backbone saturates at 64 and degrades slightly beyond it. Mean ± standard deviation over five random seeds.
<table><tr><td rowspan="2">Backbone</td><td colspan="5"></td></tr><tr><td>16</td><td>32</td><td>Batch size 64</td><td>128</td><td>256</td></tr><tr><td>LPIPS-Alex [43]</td><td> $0 . 7 1 9 \pm 0 . 0 0 6$ </td><td> $0 . 7 2 9 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 7 3 3 \pm 0 . 0 0 6 }$ </td><td> $0 . 7 2 8 \pm 0 . 0 0 6$ </td><td> $0 . 7 1 2 \pm 0 . 0 0 6$ </td></tr><tr><td>DINOv3 [35]</td><td> $0 . 4 7 9 \pm 0 . 0 8 0$ </td><td> $0 . 6 5 0 \pm 0 . 0 1 8$ </td><td> $0 . 6 9 6 \pm 0 . 0 0 7$ </td><td> $0 . 7 0 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 7 0 7 \pm 0 . 0 0 5 }$ </td></tr></table>

Cross Model Validation To study the generalizability of our method, we adopt diverse diffusion models into our pipeline in replacement to FLUX, which we used in our main experiments. For this experiment, we use three pre-trained diffusion models publicly available, i.e., Stable Diffusion 1.5 (SD 1.5) [33], SD-XL [29], and SD3 [11]. Using these pre-trained models, we generate image pair sets in the same manner. For this experiment we build a pool of 50k pairs per generator and draw five disjoint 10k subsets from it, training on each subset with the training seed held fixed. Every generator is thus matched at 10k training pairs, roughly 2% of our main training set. The experimental results can be found in Table 4. Across generators the ordering of the two backbones is preserved, and every generator yields a working metric. Even at that reduced budget, training on any of the four surpasses the best human-annotated dataset for the CNN backbone (0.622, Table 1), and with the Transformer backbone every generator except SD-XL at least matches its best human-annotated baseline (0.508). The approach therefore does not depend on FLUX specifically, but the choice of generator is not immaterial either. FLUX.1 is the strongest option for both backbones, by margins well outside the data-sampling error bars, and the Transformer-based models benefit the most from the strong generator.

Table 4: Experimental results using diverse diffusion models for data generation. Our pipeline is not specific to FLUX, and every generator yields a working metric, although FLUX.1 is the strongest choice. Cells report mean ± standard deviation over five train runs, each using disjoint 10k subsets from the generator.
<table><tr><td rowspan="2">Backbone</td><td colspan="4">Data Generator (10k pairs)</td></tr><tr><td>SD-1.5 [33]</td><td>SD-XL [29]</td><td>SD-3 [11]</td><td> $\mathrm { F L U X } . 1 [ 2 0 ]$ </td></tr><tr><td>LPIPS-Alex [43]</td><td> $0 . 6 8 7 \pm 0 . 0 0 8$ </td><td> $0 . 6 6 2 \pm 0 . 0 0 8$ </td><td> $0 . 6 9 5 \pm 0 . 0 0 7$ </td><td> $0 . 7 4 1 \pm 0 . 0 0 2$ </td></tr><tr><td> $\mathrm { D I N O v } 3 \left[ 3 5 \right]$ </td><td> $0 . 5 7 4 \pm 0 . 0 7 8$ </td><td> $0 . 4 1 8 \pm 0 . 0 4 4$ </td><td> $0 . 5 1 7 \pm 0 . 0 1 7$ </td><td> $0 . 6 7 2 \pm 0 . 0 1 6$ </td></tr></table>

Timestep Range Selection We study on how the performance of models change depending on the sampling range of forking timesteps. In this experiment, we use the three CNN backbones, LPIPS-Alex, VGG and DISTS models, along with a Transformer backbone, DINOv3. The results are displayed in Table 5. Within a 50-step process, the initial 35 steps, which are closer to the noise state than the clean image state, has shown to be the most important. The final 15 steps of the generation process are dedicated to refining fine-grained details that are barely perceptible to the human eye. Since such subtle refinements carry little perceptual significance, incorporating these steps into training may introduce noise into the learning signal, and the two LPIPS backbones indeed peak without them. DISTS and DINOv3, however, perform best over the full range. Considering the overall robustness, in our main experiments, we used the full 50 steps in training.

Table 5: Experiment on different sampling ranges of forking timesteps. At a matched window width the earlier window is the most useful one, and the full schedule overall provides the most stable result. Mean ± standard deviation over five random seeds.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">All [0,50]</td><td colspan="5">Sampling Range of Forking Timesteps</td></tr><tr><td>[0,25]</td><td>[0,35]</td><td>[13,37]</td><td>[15, 50]</td><td>[25,50]</td></tr><tr><td>LPIPS-Alex [43]</td><td> $0 . 7 3 3 \pm 0 . 0 0 6$ </td><td> $\underline { { 0 . 7 4 8 \pm 0 . 0 0 8 } }$ </td><td> $\mathbf { 0 . 7 5 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 7 2 9 \pm 0 . 0 0 8$ </td><td> $0 . 7 3 1 \pm 0 . 0 0 7$ </td><td> $0 . 7 2 0 \pm 0 . 0 0 6$ </td></tr><tr><td>LPIPS-VGG [43]</td><td> $0 . 6 8 3 \pm 0 . 0 0 8$ </td><td> $\overline { { 0 . 6 8 8 \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 6 9 7 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 8 5 \pm 0 . 0 0 4$ </td><td> $0 . 6 6 6 \pm 0 . 0 0 6$ </td><td> $0 . 6 2 6 \pm 0 . 0 0 5$ </td></tr><tr><td>DISTS [9]</td><td> $\mathbf { 0 . 6 1 5 \pm 0 . 0 0 1 }$ </td><td> $\overline { { 0 . 4 6 4 \pm 0 . 0 0 2 } }$ </td><td> $0 . 5 8 3 \pm 0 . 0 0 1$ </td><td> $0 . 5 6 9 \pm 0 . 0 0 2$ </td><td> $0 . 5 5 6 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 5 \pm 0 . 0 0 3$ </td></tr><tr><td>DINOv3 [35]</td><td> $\mathbf { 0 . 7 0 3 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 2 0 \pm 0 . 0 1 5$ </td><td> $\overline { { 0 . 6 9 6 \pm 0 . 0 1 5 } }$ </td><td> $0 . 7 0 0 \pm 0 . 0 0 9$ </td><td> $0 . 7 0 2 \pm 0 . 0 1 5$ </td><td> $0 . 6 8 0 \pm 0 . 0 1 4$ </td></tr></table>

## 6 Conclusion

In this paper, we introduced a data generation pipeline that reframes perceptual distance as a forking moment in a diffusion denoising trajectory. By forward-diffusing a reference image to a sampled timestep and denoising it back, we automatically synthesize training pairs with calibrated perceptual distances, no human annotation required. Our human study verified that using forking moments in diffusion trajectory aligns well with human perception, supporting our argument. We further show that RankNet-style supervision over our generated data substantially outperforms 2AFC-style binary classification, yielding richer gradient signal and implicit transitivity enforcement. Extensive experiments on diverse backbones and evaluation benchmarks demonstrate consistent improvement over strong baselines, validating both the dataset generation pipeline and the ranking-based objective.

## Acknowledgments and Disclosure of Funding

This work was supported by the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers,Seoul National University in 2026; the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (Nos. 2022R1A3B1077720 and 2022R1A5A7083908); and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2021-II211343, Artificial Intelligence Graduate School Program, Seoul National University). The authors also thank Yongsung Kim for valuable feedback and support.

## References

[1] Žiga Babnik, Peter Peer, and Vitomir Štruc. Diffiqa: Face image quality assessment using denoising diffusion probabilistic models. In 2023 IEEE international joint conference on biometrics (IJCB), pages 1–10. IEEE, 2023.

[2] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. In Proceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.

[3] Ting Chen. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972, 2023.

[4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

[5] Zewen Chen, Juan Wang, Bing Li, Chunfeng Yuan, Weiming Hu, Junxian Liu, Peng Li, Yan Wang, Youqun Zhang, and Congxuan Zhang. Gmc-iqa: Exploiting global-correlation and mean-opinion consistency for no-reference image quality assessment. arXiv preprint arXiv:2401.10511, 2024.

[6] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11472–11481, 2022.

[7] Dale Decatur, Thibault Groueix, Wang Yifan, Rana Hanocka, Vladimir Kim, and Matheus Gadelha. Reusing computation in text-to-image diffusion for efficient generation of image sets. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16482–16491, 2025.

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[9] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[12] Joseph L Fleiss. Measuring nominal scale agreement among many raters. Psychological bulletin, 76(5): 378, 1971.

[13] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. Advances in Neural Information Processing Systems, 36:50742–50768, 2023.

[14] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[16] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023.

[17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[18] Gu Jinjin, Cai Haoming, Chen Haoyu, Ye Xiaoxing, Jimmy S Ren, and Dong Chao. Pipal: a large-scale image quality assessment dataset for perceptual image restoration. In European conference on computer vision, pages 633–651. Springer, 2020.

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutiona neural networks. Advances in neural information processing systems, 25, 2012.

[20] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[21] GG Landis JRKoch. The measurement of observer agreement for categorical data. Biometrics, 33(1): 159174, 1977.

[22] Eric C Larson and Damon M Chandler. Most apparent distortion: full-reference image quality assessment and the role of strategy. Journal ofelectronic imaging, 19(1):011006–011006, 2010.

[23] Hanhe Lin, Vlad Hosu, and Dietmar Saupe. Kadid-10k: A large-scale artificially distorted iqa database. In 2019 Tenth International Conference on Quality ofMultimedia Experience (QoMEX), pages 1–3. IEEE, 2019.

[24] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[25] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

[26] Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. Advances in Neural Information Processing Systems, 36:47500–47510, 2023.

[27] Kede Ma, Wentao Liu, Tongliang Liu, Zhou Wang, and Dacheng Tao. dipiq: Blind image quality assessment by learning-to-rank discriminable image pairs. IEEE Transactions on image processing, 26(8): 3951–3964, 2017.

[28] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

[29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.

[30] Nikolay Ponomarenko, Lina Jin, Oleg Ieremeiev, Vladimir Lukin, Karen Egiazarian, Jaakko Astola, Benoit Vozel, Kacem Chehdi, Marco Carli, Federica Battisti, et al. Image database tid2013: Peculiarities, results and perspectives. Signal processing: Image communication, 30:57–77, 2015.

[31] Ekta Prashnani, Hong Cai, Yasamin Mostofi, and Pradeep Sen. Pieapp: Perceptual image-error assessment through pairwise preference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1808–1817, 2018.

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[34] H.R. Sheikh, M.F. Sabir, and A.C. Bovik. A statistical evaluation of recent full reference image quality assessment algorithms. IEEE Transactions on Image Processing, 15(11):3440–3451, 2006.

[35] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[36] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[37] Yiren Song, Xiaokang Liu, and Mike Zheng Shou. Diffsim: Taming diffusion models for evaluating visual similarity. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 16904–16915, 2025.

[38] Hossein Talebi, Ehsan Amid, Peyman Milanfar, and Manfred K Warmuth. Rank-smoothed pairwise learning in perceptual quality assessment. In 2020 IEEE International Conference on Image Processing (ICIP), pages 3413–3417. IEEE, 2020.

[39] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in neural information processing systems, 36:1363–1389, 2023.

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[41] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[42] Julian Wyatt, Adam Leach, Sebastian M Schmon, and Chris G Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 650–656, 2022.

[43] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

## A Experimental Details

Model Training The prediction heads of LPIPS-Alex/VGG are linear layers from each level of the feature pyramid, and DISTS learn the α and β in integration of the extracted intermediate features. For three Transformer backbones, DINOv3, CLIP and MAE, the feature extractor is also fixed, and we append three ViT layers [10] for distance logit prediction. The prediction head layers take the feature tokens extracted from the backbone as the input, along with a [CLS] token which make the prediction of the distance logit. For all experiments besides DreamSim, we use a batch size of 64, and a fixed learning rate of 2e-4 and 1e-4 respectively for CNN backbones and the three Transformer backbones. For DreamSim, we follow its original configuration without any change, from LoRA [17] configuration $( r { = } 1 6 , \alpha { = } 1$ , dropout 0.3 on the qkv projections), input protocol and optimizer, and only the training set is varied; the sample budget matches every other column.

Ensuring symmetry in Transformer backbone models We write $\hat { d } ( u , v )$ for the distance the model assigns to an image pair $( u , v ) ;$ this is the quantity written $\hat { d } _ { i }$ in Sec. 4.2. For the Transformer backbones, DINOv3, CLIP and MAE, it is produced by the prediction head, whose output needs an adjustment for symmetry. The head reads the two images as a single concatenated sequence and is thus asymmetric, $\hat { d } ( u , v )$ and $\hat { d } ( v , u )$ returning non-identical values. Therefore we report the results acquired from $\begin{array} { r } { \frac { 1 } { 2 } \left[ \hat { d } ( u , v ) + \hat { d } ( v , u ) \right] } \end{array}$ , which removes the asymmetry. During training, each training pair is presented in a random order to make the model work in both orders, and be naturally symmetric. This way, we observe the distance logits to be similar in both input orders, although inherently they cannot be perfectly identical. Neither adjustment applies to LPIPS, DISTS or DreamSim, which compute a symmetric distance directly and are reported as they are.

## B Full Results

Table 6: Comparison of training datasets and their objectives on four reference-based IQA benchmarks, across CNN- and Transformer-based backbones. Mean ± standard deviation of SROCC over five random seeds; full-statistics version of Table 1. <sup>†</sup>Evaluated under 224 resolution.
<table><tr><td rowspan="2">Train Data</td><td colspan="3">CNN-based</td><td colspan="4">Transformer-based</td><td rowspan="2">Avg.</td></tr><tr><td>LPIPS-Alex</td><td>LPIPS-VGG</td><td>DISTS</td><td>DINOv3</td><td>CLIP</td><td>MAE</td><td>DreamSim†</td></tr><tr><td colspan="10">Results on PIPAL [18]</td></tr><tr><td>BAPPS [43]</td><td> $0 . 6 2 2 \pm 0 . 0 0 5$ </td><td> $0 . 6 3 4 \pm 0 . 0 1 8$ </td><td> $0 . 4 8 9 \pm 0 . 0 0 3$ </td><td> $0 . 2 8 7 \pm 0 . 0 2 0$ </td><td> $0 . 1 5 1 \pm 0 . 0 4 0$ </td><td> $\underline { { 0 . 3 3 8 \pm 0 . 0 1 3 } }$ </td><td> $0 . 7 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 4 6 9 \pm 0 . 0 0 8$ </td></tr><tr><td>PieAPP [31]</td><td> $\overline { { 0 . 6 0 2 \pm 0 . 0 0 8 } }$ </td><td> $\underline { { 0 . 6 3 5 \pm 0 . 0 1 6 } }$ </td><td> $0 . 5 6 4 \pm 0 . 0 0 3$ </td><td> $0 . 3 2 5 \pm 0 . 0 1 7$ </td><td> $0 . 2 5 8 \pm 0 . 0 3 7$ </td><td> $\overline { { 0 . 2 9 4 \pm 0 . 0 2 3 } }$ </td><td> $\overline { { 0 . 7 1 1 \pm 0 . 0 0 7 } }$ </td><td> $0 . 4 8 4 \pm 0 . 0 0 7$ </td></tr><tr><td>NIGHTS [13]</td><td> $0 . 5 7 7 \pm 0 . 0 0 6$ </td><td> $\overline { { 0 . 5 4 5 \pm 0 . 0 1 1 } }$ </td><td> $\overline { { - 0 . 2 9 4 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 2 3 4 \pm 0 . 0 2 4 } }$ </td><td> $0 . 2 7 7 \pm 0 . 0 2 5$ </td><td> $0 . 2 3 4 \pm 0 . 0 2 9$ </td><td> $0 . 6 6 2 \pm 0 . 0 1 9$ </td><td> $0 . 3 1 9 \pm 0 . 0 0 5$ </td></tr><tr><td>KADID-10K [23]</td><td> $0 . 5 7 7 \pm 0 . 0 2 5$ </td><td> $0 . 5 9 0 \pm 0 . 0 3 4$ </td><td> $0 . 5 1 8 \pm 0 . 0 0 2$ </td><td> $0 . 2 9 8 \pm 0 . 0 1 0$ </td><td> $0 . 3 1 2 \pm 0 . 0 1 8$ </td><td> $0 . 2 5 3 \pm 0 . 0 4 1$ </td><td> $0 . 6 1 7 \pm 0 . 0 0 3$ </td><td> $0 . 4 5 2 \pm 0 . 0 0 6$ </td></tr><tr><td>FoMo (Ours)</td><td> $\mathbf { 0 . 7 3 3 \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 6 8 3 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 6 1 5 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 6 9 9 \pm 0 . 0 0 6 }$ </td><td>0.644 ± 0.024</td><td> $\mathbf { 0 . 6 3 2 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 7 7 6 \pm 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 6 8 3 \pm 0 . 0 0 4 }$ </td></tr><tr><td colspan="9">Results on TID2013 [30]</td></tr><tr><td>BAPPS [43]</td><td> $0 . 7 7 9 \pm 0 . 0 0 3$ </td><td> $0 . 6 7 1 \pm 0 . 0 0 2$ </td><td> $0 . 6 3 2 \pm 0 . 0 0 1$ </td><td> $0 . 3 4 1 \pm 0 . 0 2 2$ </td><td> $0 . 2 8 5 \pm 0 . 0 3 0$ </td><td> $0 . 2 8 7 \pm 0 . 0 2 2$ </td><td> $\mathbf { 0 . 8 1 3 \pm 0 . 0 0 4 }$ </td><td> $0 . 5 4 4 \pm 0 . 0 0 6$ </td></tr><tr><td>PieAPP [31]</td><td> $0 . 7 6 1 \pm 0 . 0 0 6$ </td><td> $\underline { { 0 . 7 2 2 } } \pm 0 . 0 0 5$ </td><td> $0 . 6 8 0 \pm 0 . 0 0 4$ </td><td> $0 . 3 1 5 \pm 0 . 0 4 2$ </td><td> $0 . 3 0 2 \pm 0 . 0 2 5$ </td><td> $0 . 4 6 7 \pm 0 . 0 2 7$ </td><td> $0 . 7 6 7 \pm 0 . 0 0 4$ </td><td> $0 . 5 7 3 \pm 0 . 0 0 8$ </td></tr><tr><td>NIGHTS [13]</td><td> $0 . 7 7 9 \pm 0 . 0 0 3$ </td><td> $\overline { { 0 . 6 5 3 \pm 0 . 0 0 1 } }$ </td><td> $- 0 . 4 9 7 \pm 0 . 0 0 5$ </td><td> $0 . 2 3 0 \pm 0 . 0 2 2$ </td><td> $0 . 2 9 4 \pm 0 . 0 3 5$ </td><td> $0 . 3 5 8 \pm 0 . 0 3 3$ </td><td> $0 . 7 6 2 \pm 0 . 0 0 7$ </td><td> $0 . 3 6 8 \pm 0 . 0 0 7$ </td></tr><tr><td>KADID-10K [23]</td><td> $\mathbf { 0 . 7 9 3 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 7 5 7 \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 0 . 8 3 4 \pm 0 . 0 0 1 }$ </td><td> $\underline { { 0 . 6 1 3 \pm 0 . 0 2 4 } }$ </td><td> $\underline { { 0 . 5 1 9 \pm 0 . 0 1 6 } }$ </td><td> $0 . 5 9 5 \pm 0 . 0 1 3$ </td><td> $0 . 7 8 8 \pm 0 . 0 0 4$ </td><td> $\underline { { 0 . 7 0 0 \pm 0 . 0 0 6 } }$ </td></tr><tr><td>FoMo (Ours)</td><td> $\underline { { 0 . 7 8 5 \pm 0 . 0 0 3 } }$ </td><td> $0 . 6 6 3 \pm 0 . 0 0 3$ </td><td> $\underline { { 0 . 6 9 1 \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 7 1 3 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 7 3 7 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 6 4 4 \pm 0 . 0 5 2 }$ </td><td> $\underline { { 0 . 8 0 1 \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 7 1 9 \pm 0 . 0 0 7 }$ </td></tr><tr><td colspan="9"></td></tr><tr><td>BAPPS [43]</td><td> $0 . 9 4 3 \pm 0 . 0 0 1$ </td><td> $0 . 8 8 7 \pm 0 . 0 0 1$ </td><td> $0 . 8 5 5 \pm 0 . 0 0 1$ </td><td>Results on CSIQ [22]</td><td> $0 . 4 2 7 \pm 0 . 0 1 5$ </td><td> $0 . 4 5 5 \pm 0 . 0 5 1$ </td><td>0.911 ± 0.000</td><td> $0 . 6 9 9 \pm 0 . 0 0 7$ </td></tr><tr><td>PieAPP [31]</td><td> $\overline { { 0 . 9 3 6 \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 0 7 }$ </td><td> $0 . 8 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 4 1 1 \pm 0 . 0 2 7$ </td><td> $0 . 5 1 4 \pm 0 . 0 1 6$ </td><td> $0 . 5 8 7 \pm 0 . 0 1 2$ </td><td> $0 . 9 0 3 \pm 0 . 0 0 3$ </td><td> $0 . 7 1 3 \pm 0 . 0 0 5$ </td></tr><tr><td>NIGHTS [13]</td><td> $\mathbf { 0 . 9 4 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 8 6 2 \pm 0 . 0 0 2$ </td><td> $- 0 . 5 6 0 \pm 0 . 0 0 6$ </td><td> $0 . 3 3 1 \pm 0 . 0 4 2$   $0 . 3 3 5 \pm 0 . 0 3 1$ </td><td> $0 . 3 6 4 \pm 0 . 0 2 2$ </td><td> $0 . 4 2 7 \pm 0 . 0 3 3$ </td><td> $\overline { { 0 . 8 6 7 \pm 0 . 0 0 7 } }$ </td><td> $0 . 4 6 3 \pm 0 . 0 0 6$ </td></tr><tr><td>KADID-10K [23]</td><td> $0 . 9 3 5 \pm 0 . 0 0 2$ </td><td> $\underline { { 0 . 8 9 5 \pm 0 . 0 1 6 } }$ </td><td> $\mathbf { 0 . 9 3 8 \pm 0 . 0 0 0 }$ </td><td> $\underline { { 0 . 6 1 6 \pm 0 . 0 0 9 } }$ </td><td> $0 . 6 0 4 \pm 0 . 0 2 6$ </td><td> $\underline { { 0 . 6 9 8 \pm 0 . 0 3 5 } }$ </td><td> $0 . 8 7 4 \pm 0 . 0 0 9$ </td><td> $\underline { { 0 . 7 9 4 \pm 0 . 0 1 0 } }$ </td></tr><tr><td>FoMo (Ours)</td><td> $0 . 9 3 8 \pm 0 . 0 0 1$ </td><td> $\overline { { 0 . 8 5 9 \pm 0 . 0 0 6 } }$ </td><td> $0 . 9 1 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 8 1 1 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 0 1 } \pm \mathbf { 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 7 9 5 \pm 0 . 0 3 8 }$ </td><td> $0 . 8 9 4 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 8 7 4 \pm 0 . 0 0 5 }$ </td></tr><tr><td colspan="9">Results on LIVE [34]</td></tr><tr><td>BAPPS [43]</td><td>0.952 ± 0.001</td><td>0.929 ± 0.002</td><td>0.843 ± 0.001</td><td> $0 . 6 8 8 \pm 0 . 0 3 2$ </td><td>0.425 ± 0.059</td><td>0.298 ± 0.062</td><td>0.927 ± 0.001</td><td> $0 . 7 2 3 \pm 0 . 0 1 3$ </td></tr><tr><td>PieAPP [31]</td><td> $0 . 9 3 4 \pm 0 . 0 1 3$ </td><td> $0 . 9 2 1 \pm 0 . 0 1 3$ </td><td> $0 . 8 6 0 \pm 0 . 0 0 2$ </td><td> $0 . 4 5 8 \pm 0 . 0 2 2$ </td><td>0.415 ± 0.030</td><td> $0 . 5 3 7 \pm 0 . 0 1 6$ </td><td>0.906 ± 0.004</td><td> $0 . 7 1 9 \pm 0 . 0 0 6$ </td></tr><tr><td>NIGHTS [13]</td><td> $\underline { { 0 . 9 4 9 \pm 0 . 0 0 2 } }$ </td><td> $0 . 9 1 9 \pm 0 . 0 0 2$ </td><td> $- 0 . 7 7 4 \pm 0 . 0 0 4$ </td><td> $0 . 4 3 9 \pm 0 . 0 2 1$ </td><td> $0 . 5 6 0 \pm 0 . 0 5 7$ </td><td> $0 . 3 7 6 \pm 0 . 0 5 8$ </td><td> $0 . 8 6 7 \pm 0 . 0 0 7$ </td><td> $0 . 4 7 7 \pm 0 . 0 1 5$ </td></tr><tr><td>KADID-10K [23]</td><td> $\overline { { 0 . 9 4 2 \pm 0 . 0 1 3 } }$ </td><td> $0 . 9 1 2 \pm 0 . 0 3 2$ </td><td>0.950 ± 0.000</td><td> $\underline { { 0 . 7 6 5 \pm 0 . 0 2 3 } }$ </td><td> $0 . 8 2 7 \pm 0 . 0 0 9$ </td><td>0.797 ± 0.015</td><td>0.896 ± 0.012</td><td>0.870 ± 0.012</td></tr><tr><td>FoMo (Ours)</td><td> $0 . 9 4 8 \pm 0 . 0 0 3$ </td><td> $\underline { { 0 . 9 2 3 \pm 0 . 0 0 4 } }$ </td><td>0.954 ± 0.000</td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 0 2 }$ </td><td>0.925 ± 0.011</td><td> $\mathbf { 0 . 9 0 7 \pm 0 . 0 2 6 }$ </td><td>0.931 ± 0.002</td><td>0.926 ± 0.003</td></tr></table>

## C Related Work

## C.1 Reference-based Image Quality Assessment

A reliable perceptual metric must produce a globally consistent quality ordering, one that mirrors how humans rank distortions across the full image set, not just on isolated pairs. This requirement has

Table 7: Comparison of training datasets and their objectives on four reference-based IQA benchmarks, across CNN- and Transformer-based backbones. Mean ± standard deviation of KROCC over five random seeds. <sup>†</sup>Evaluated under 224 resolution.

$$
\underline { { 0 . 4 4 0 \pm 0 . 0 0 4 } }
$$

$$
\underline { { 0 . 4 5 5 \pm 0 . 0 1 4 } }
$$

$$
\overline { { 0 . 4 2 0 \pm 0 . 0 0 7 } }
$$

$$
\overline { { 0 . 4 5 0 \pm 0 . 0 1 3 } }
$$

$$
0 . 3 3 7 \pm 0 . 0 0 2
$$

$$
0 . 3 8 9 \pm 0 . 0 0 2
$$

$$
0 . 2 2 2 \pm 0 . 0 1 3
$$

$$
0 . 1 7 5 \pm 0 . 0 2 6
$$

$$
\overline { { 0 . 1 9 7 \pm 0 . 0 1 6 } }
$$

$$
\overline { { 0 . 5 1 1 \pm 0 . 0 0 6 } }
$$

$$
0 . 3 3 8 \pm 0 . 0 0 5
$$

$$
0 . 4 0 3 \pm 0 . 0 2 0
$$

$$
\overline { { 0 . 1 5 9 \pm 0 . 0 1 7 } }
$$

$$
0 . 4 1 9 \pm 0 . 0 2 7
$$

$$
0 . 3 6 0 \pm 0 . 0 0 1
$$

$$
\mathbf { 0 . 5 3 6 \pm 0 . 0 0 6 }
$$

$$
\mathbf { 0 . 4 9 7 \pm 0 . 0 0 7 }
$$

$$
\mathbf { 0 . 4 3 5 \pm 0 . 0 0 1 }
$$

$$
0 . 2 1 2 \pm 0 . 0 1 2
$$

$$
0 . 4 4 0 \pm 0 . 0 0 3
$$

$$
0 . 1 7 0 \pm 0 . 0 2 8
$$

$$
0 . 3 1 5 \pm 0 . 0 0 5
$$

$$
\mathbf { 0 . 4 5 9 \pm 0 . 0 1 9 }
$$

$$
\mathbf { 0 . 4 4 4 \pm 0 . 0 1 5 }
$$

$$
\mathbf { 0 . 5 7 1 \pm 0 . 0 1 0 }
$$

$$
\mathbf { 0 . 4 9 2 \pm 0 . 0 0 3 }
$$

$$
0 . 5 8 4 \pm 0 . 0 0 3
$$

$$
0 . 4 9 7 \pm 0 . 0 0 2
$$

$$
0 . 4 5 0 \pm 0 . 0 0 1
$$

$$
0 . 2 3 5 \pm 0 . 0 1 7
$$

$$
\underline { { 0 . 5 4 0 \pm 0 . 0 0 4 } }
$$

$$
0 . 1 9 4 \pm 0 . 0 2 1
$$

$$
0 . 5 6 7 \pm 0 . 0 0 6
$$

$$
0 . 4 8 6 \pm 0 . 0 0 4
$$

$$
0 . 1 9 2 \pm 0 . 0 1 5
$$

$$
\mathbf { 0 . 6 1 9 \pm 0 . 0 0 3 }
$$

$$
0 . 3 9 6 \pm 0 . 0 0 4
$$

$$
0 . 2 0 7 \pm 0 . 0 1 7
$$

$$
0 . 5 8 2 \pm 0 . 0 0 3
$$

$$
\overline { { 0 . 4 8 0 \pm 0 . 0 0 1 } }
$$

$$
0 . 5 7 4 \pm 0 . 0 0 3
$$

$$
0 . 1 5 7 \pm 0 . 0 1 6
$$

$$
0 . 2 4 3 \pm 0 . 0 2 3
$$

$$
\mathbf { 0 . 5 9 6 \pm 0 . 0 0 5 }
$$

$$
\mathbf { 0 . 6 4 0 \pm 0 . 0 0 1 }
$$

$$
0 . 5 6 6 \pm 0 . 0 0 6
$$

$$
\mathbf { 0 . 5 6 7 \pm 0 . 0 1 1 }
$$

$$
0 . 4 4 8 \pm 0 . 0 2 1
$$

$$
0 . 2 6 9 \pm 0 . 0 0 5
$$

$$
0 . 3 6 8 \pm 0 . 0 1 2
$$

$$
0 . 4 2 6 \pm 0 . 0 0 9
$$

$$
0 . 5 8 6 \pm 0 . 0 0 3
$$

$$
0 . 5 1 2 \pm 0 . 0 0 1
$$

$$
0 . 4 8 9 \pm 0 . 0 0 3
$$

$$
0 . 5 9 8 \pm 0 . 0 0 5
$$

$$
\overline { { { 0 . 5 2 6 \pm 0 . 0 0 2 } } }
$$

$$
\mathbf { 0 . 5 4 7 \pm 0 . 0 1 6 }
$$

$$
0 . 5 2 0 \pm 0 . 0 0 5
$$

$$
\mathbf { 0 . 4 7 3 \pm 0 . 0 4 3 }
$$

$$
\underline { { 0 . 6 0 5 \pm 0 . 0 0 4 } }
$$

$$
\mathbf { 0 . 5 3 4 \pm 0 . 0 0 6 }
$$

$$
\underline { { 0 . 7 8 6 \pm 0 . 0 0 2 } }
$$

$$
0 . 7 0 1 \pm 0 . 0 0 3
$$

$$
\overline { { 0 . 7 7 3 \pm 0 . 0 0 4 } }
$$

$$
\mathbf { 0 . 7 1 3 \pm 0 . 0 0 9 }
$$

$$
0 . 2 9 8 \pm 0 . 0 1 1
$$

$$
0 . 3 1 5 \pm 0 . 0 3 6
$$

$$
0 . 6 1 1 \pm 0 . 0 0 4
$$

$$
0 . 5 4 0 \pm 0 . 0 0 5
$$

$$
\mathbf { 0 . 7 9 0 \pm 0 . 0 0 2 }
$$

$$
0 . 6 6 4 \pm 0 . 0 0 1
$$

$$
0 . 4 0 5 \pm 0 . 0 0 9
$$

$$
- 0 . 3 8 4 \pm 0 . 0 0 5
$$

$$
0 . 2 3 0 \pm 0 . 0 2 1
$$

$$
0 . 7 2 4 \pm 0 . 0 0 5
$$

$$
0 . 7 6 8 \pm 0 . 0 0 5
$$

$$
0 . 5 4 5 \pm 0 . 0 0 3
$$

$$
\underline { { 0 . 7 0 5 } } \pm 0 . 0 2 0
$$

$$
0 . 2 4 4 \pm 0 . 0 1 6
$$

$$
\mathbf { 0 . 7 7 6 \pm 0 . 0 0 1 }
$$

$$
0 . 2 9 1 \pm 0 . 0 2 4
$$

$$
\underline { { 0 . 4 4 7 \pm 0 . 0 0 7 } }
$$

$$
\overline { { 0 . 6 7 8 \pm 0 . 0 0 9 } }
$$

$$
0 . 4 3 0 \pm 0 . 0 2 1
$$

$$
0 . 7 8 1 \pm 0 . 0 0 2
$$

$$
0 . 3 5 9 \pm 0 . 0 0 4
$$

$$
\overline { { 0 . 6 7 2 \pm 0 . 0 0 6 } }
$$

$$
\underline { { 0 . 7 5 0 \pm 0 . 0 0 1 } }
$$

$$
0 . 5 1 0 \pm 0 . 0 2 9
$$

$$
\overline { { { \bf 0 . 6 2 6 \pm 0 . 0 0 2 } } }
$$

$$
\mathbf { 0 . 7 2 3 \pm 0 . 0 1 9 }
$$

$$
0 . 6 8 2 \pm 0 . 0 1 2
$$

$$
\mathbf { 0 . 6 0 9 \pm 0 . 0 3 0 }
$$

$$
\underline { { 0 . 6 1 7 \pm 0 . 0 1 0 } }
$$

$$
0 . 7 0 9 \pm 0 . 0 0 3
$$

$$
\mathbf { 0 . 6 9 6 \pm 0 . 0 0 4 }
$$

$$
\overline { { { \bf 0 . 8 0 0 \pm 0 . 0 0 1 } } }
$$

$$
\overline { { { \bf 0 . 7 5 7 \pm 0 . 0 0 5 } } }
$$

$$
0 . 6 4 4 \pm 0 . 0 0 1
$$

$$
0 . 7 6 9 \pm 0 . 0 1 8
$$

$$
0 . 5 0 8 \pm 0 . 0 2 9
$$

$$
0 . 7 4 6 \pm 0 . 0 2 2
$$

$$
\overline { { 0 . 2 9 8 \pm 0 . 0 4 3 } }
$$

$$
0 . 6 4 9 \pm 0 . 0 0 3
$$

$$
\overline { { 0 . 2 0 3 \pm 0 . 0 4 1 } }
$$

$$
0 . 3 2 4 \pm 0 . 0 1 8
$$

$$
0 . 7 6 6 \pm 0 . 0 0 2
$$

$$
\underline { { 0 . 7 9 7 \pm 0 . 0 0 3 } }
$$

$$
0 . 7 4 4 \pm 0 . 0 0 4
$$

$$
0 . 2 8 8 \pm 0 . 0 2 3
$$

$$
0 . 5 6 8 \pm 0 . 0 0 8
$$

$$
- 0 . 5 7 4 \pm 0 . 0 0 3
$$

$$
0 . 3 7 0 \pm 0 . 0 1 5
$$

$$
0 . 3 0 7 \pm 0 . 0 1 6
$$

$$
\overline { { 0 . 7 2 8 \pm 0 . 0 0 5 } }
$$

$$
0 . 7 8 9 \pm 0 . 0 1 7
$$

$$
0 . 3 9 2 \pm 0 . 0 4 5
$$

$$
0 . 7 3 4 \pm 0 . 0 4 4
$$

$$
0 . 5 5 4 \pm 0 . 0 0 6
$$

$$
0 . 2 5 6 \pm 0 . 0 4 1
$$

$$
\underline { { 0 . 8 0 1 \pm 0 . 0 0 1 } }
$$

$$
0 . 5 8 0 \pm 0 . 0 2 1
$$

$$
0 . 6 7 9 \pm 0 . 0 0 9
$$

$$
0 . 7 9 1 \pm 0 . 0 0 7
$$

$$
\underline { { 0 . 7 4 8 \pm 0 . 0 0 7 } }
$$

$$
\underline { { 0 . 6 2 7 \pm 0 . 0 1 1 } }
$$

$$
0 . 3 7 2 \pm 0 . 0 1 2
$$

$$
\mathbf { 0 . 8 0 4 \pm 0 . 0 0 1 }
$$

$$
\underline { { 0 . 5 9 5 \pm 0 . 0 1 5 } }
$$

$$
\mathbf { 0 . 7 5 4 \pm 0 . 0 1 7 }
$$

$$
0 . 7 1 3 \pm 0 . 0 1 6
$$

$$
\mathbf { 0 . 7 3 4 \pm 0 . 0 3 8 }
$$

$$
\underline { { 0 . 6 9 1 \pm 0 . 0 1 4 } }
$$

$$
\mathbf { 0 . 7 6 9 \pm 0 . 0 0 4 }
$$

$$
\mathbf { 0 . 7 6 0 \pm 0 . 0 0 5 }
$$

Table 8: Comparison of training datasets and their objectives on four reference-based IQA benchmarks, across CNN- and Transformer-based backbones. Mean ± standard deviation of PLCC over five random seeds. <sup>†</sup>Evaluated under 224 resolution.

$$
\underline { { 0 . 6 7 6 \pm 0 . 0 1 4 } }
$$

$$
0 . 6 3 4 \pm 0 . 0 0 6
$$

$$
\overline { { 0 . 6 3 7 \pm 0 . 0 1 4 } }
$$

$$
0 . 4 9 5 \pm 0 . 0 0 2
$$

$$
0 . 6 0 5 \pm 0 . 0 0 7
$$

$$
0 . 5 5 8 \pm 0 . 0 0 3
$$

$$
0 . 3 1 4 \pm 0 . 0 1 5
$$

$$
0 . 5 9 9 \pm 0 . 0 2 2
$$

$$
\underline { { 0 . 3 6 4 \pm 0 . 0 1 4 } }
$$

$$
0 . 3 6 1 \pm 0 . 0 0 2
$$

$$
0 . 1 8 6 \pm 0 . 0 4 3
$$

$$
0 . 6 2 3 \pm 0 . 0 3 4
$$

$$
\overline { { 0 . 2 9 6 \pm 0 . 0 2 8 } }
$$

$$
0 . 3 4 8 \pm 0 . 0 1 8
$$

$$
0 . 2 8 4 \pm 0 . 0 2 9
$$

$$
0 . 5 6 2 \pm 0 . 0 0 1
$$

$$
0 . 4 9 4 \pm 0 . 0 1 0
$$

$$
\mathbf { 0 . 7 7 9 \pm 0 . 0 0 7 }
$$

$$
0 . 3 3 8 \pm 0 . 0 2 5
$$

$$
\mathbf { 0 . 7 6 8 \pm 0 . 0 0 5 }
$$

$$
0 . 3 2 4 \pm 0 . 0 3 0
$$

$$
0 . 3 5 1 \pm 0 . 0 2 1
$$

$$
0 . 7 0 7 \pm 0 . 0 0 5
$$

$$
\underline { { 0 . 4 9 9 \pm 0 . 0 0 3 } }
$$

$$
\underline { { 0 . 3 4 6 \pm 0 . 0 1 9 } }
$$

$$
0 . 2 5 9 \pm 0 . 0 3 7
$$

$$
\overline { { { \bf 0 . 6 5 6 \pm 0 . 0 0 1 } } }
$$

$$
\overline { { 0 . 4 5 0 \pm 0 . 0 0 8 } }
$$

$$
0 . 6 6 9 \pm 0 . 0 2 2
$$

$$
0 . 2 7 4 \pm 0 . 0 3 4
$$

$$
\mathbf { 0 . 6 5 2 \pm 0 . 0 2 3 }
$$

$$
\mathbf { 0 . 6 2 5 \pm 0 . 0 1 6 }
$$

$$
0 . 6 1 9 \pm 0 . 0 0 7
$$

$$
0 . 4 8 2 \pm 0 . 0 0 8
$$

$$
\underline { { 0 . 7 6 4 \pm 0 . 0 1 0 } }
$$

$$
\mathbf { 0 . 7 0 0 \pm 0 . 0 0 3 }
$$

$$
0 . 8 1 4 \pm 0 . 0 0 3
$$

$$
0 . 7 5 5 \pm 0 . 0 0 2
$$

$$
0 . 7 9 8 \pm 0 . 0 0 6
$$

$$
0 . 6 6 4 \pm 0 . 0 0 1
$$

$$
\underline { { 0 . 7 7 7 } } \pm 0 . 0 0 5
$$

$$
0 . 4 8 8 \pm 0 . 0 2 8
$$

$$
0 . 3 4 3 \pm 0 . 0 3 7
$$

$$
0 . 7 4 3 \pm 0 . 0 0 3
$$

$$
0 . 3 3 0 \pm 0 . 0 3 0
$$

$$
0 . 8 0 5 \pm 0 . 0 0 4
$$

$$
0 . 4 0 5 \pm 0 . 0 2 4
$$

$$
0 . 6 0 6 \pm 0 . 0 1 2
$$

$$
\mathbf { 0 . 8 5 0 \pm 0 . 0 0 2 }
$$

$$
\overline { { 0 . 7 3 3 \pm 0 . 0 0 3 } }
$$

$$
0 . 3 4 4 \pm 0 . 0 2 0
$$

$$
0 . 6 8 6 \pm 0 . 0 0 3
$$

$$
0 . 5 1 3 \pm 0 . 0 3 0
$$

$$
0 . 3 9 1 \pm 0 . 0 3 6
$$

$$
\underline { { 0 . 8 1 5 \pm 0 . 0 0 7 } }
$$

$$
0 . 8 0 8 \pm 0 . 0 0 8
$$

$$
0 . 6 2 7 \pm 0 . 0 0 6
$$

$$
\mathbf { 0 . 7 8 5 \pm 0 . 0 1 4 }
$$

$$
0 . 3 8 6 \pm 0 . 0 5 6
$$

$$
\mathbf { 0 . 8 4 5 \pm 0 . 0 0 1 }
$$

$$
0 . 4 3 4 \pm 0 . 0 4 5
$$

$$
\underline { { 0 . 6 9 4 \pm 0 . 0 1 9 } }
$$

$$
\mathbf { 0 . 8 1 7 \pm 0 . 0 0 3 }
$$

$$
0 . 6 0 4 \pm 0 . 0 1 1
$$

$$
0 . 7 9 1 \pm 0 . 0 0 5
$$

$$
0 . 7 5 3 \pm 0 . 0 0 5
$$

$$
\underline { { 0 . 6 0 8 \pm 0 . 0 1 4 } }
$$

$$
\underline { { 0 . 7 7 0 \pm 0 . 0 0 1 } }
$$

$$
\underline { { 0 . 6 4 3 \pm 0 . 0 2 7 } }
$$

$$
\mathbf { 0 . 7 8 8 \pm 0 . 0 1 8 }
$$

$$
0 . 8 2 5 \pm 0 . 0 0 6
$$

$$
\mathbf { 0 . 6 5 7 \pm 0 . 0 4 6 }
$$

$$
\underline { { 0 . 7 4 5 \pm 0 . 0 0 6 } }
$$

$$
\underline { { 0 . 8 3 1 \pm 0 . 0 0 3 } }
$$

$$
\mathbf { 0 . 7 6 9 \pm 0 . 0 0 6 }
$$

$$
\overline { { 0 . 9 4 5 \pm 0 . 0 0 1 } }
$$

$$
\overline { { { \bf 0 . 9 0 8 \pm 0 . 0 0 3 } } }
$$

$$
0 . 9 3 4 \pm 0 . 0 0 4
$$

$$
0 . 8 5 0 \pm 0 . 0 0 1
$$

$$
\underline { { 0 . 9 0 7 \pm 0 . 0 0 8 } }
$$

$$
0 . 5 6 1 \pm 0 . 0 3 6
$$

$$
0 . 8 6 2 \pm 0 . 0 0 2
$$

$$
0 . 4 8 3 \pm 0 . 0 2 4
$$

$$
0 . 4 3 7 \pm 0 . 0 4 4
$$

$$
0 . 4 6 8 \pm 0 . 0 5 1
$$

$$
\mathbf { 0 . 9 4 5 \pm 0 . 0 0 2 }
$$

$$
\textcircled { 0 . 9 3 2 } \pm \mathbf { 0 . 0 0 1 }
$$

$$
\overline { { 0 . 8 7 7 \pm 0 . 0 0 2 } }
$$

$$
0 . 7 6 9 \pm 0 . 0 0 3
$$

$$
0 . 5 9 4 \pm 0 . 0 1 5
$$

$$
0 . 4 5 0 \pm 0 . 0 6 0
$$

$$
0 . 5 9 4 \pm 0 . 0 1 1
$$

$$
0 . 9 2 0 \pm 0 . 0 0 4
$$

$$
0 . 9 3 2 \pm 0 . 0 0 4
$$

$$
0 . 8 8 9 \pm 0 . 0 2 7
$$

$$
0 . 3 9 9 \pm 0 . 0 3 3
$$

$$
\mathbf { 0 . 9 3 4 } \pm \mathbf { 0 . 0 0 0 }
$$

$$
0 . 4 5 5 \pm 0 . 0 4 2
$$

$$
\underline { { 0 . 7 3 0 \pm 0 . 0 1 8 } }
$$

$$
\overline { { 0 . 8 7 8 \pm 0 . 0 0 8 } }
$$

$$
\underline { { 0 . 7 2 7 \pm 0 . 0 2 9 } }
$$

$$
0 . 6 8 2 \pm 0 . 0 1 2
$$

$$
\underline { { 0 . 7 3 5 \pm 0 . 0 3 4 } }
$$

$$
0 . 9 4 5 \pm 0 . 0 0 1
$$

$$
\underline { { 0 . 9 3 1 \pm 0 . 0 0 1 } }
$$

$$
0 . 8 8 8 \pm 0 . 0 0 6
$$

$$
0 . 8 9 1 \pm 0 . 0 0 7
$$

$$
\overline { { 0 . 9 2 3 \pm 0 . 0 1 1 } }
$$

$$
\textcircled { 0 . 8 1 6 \pm 0 . 0 3 0 }
$$

$$
0 . 8 3 4 \pm 0 . 0 1 1
$$

$$
0 . 9 0 5 \pm 0 . 0 0 3
$$

$$
R e s u l t s o n L I V E \ : / 3 4 J
$$

$$
\overline { { { \bf 0 . 9 4 6 \pm 0 . 0 0 1 } } }
$$

$$
\mathbf { 0 . 9 2 9 } \pm \mathbf { 0 . 0 0 3 }
$$

$$
0 . 8 3 9 \pm 0 . 0 0 0
$$

$$
0 . 9 2 3 \pm 0 . 0 1 6
$$

$$
0 . 7 6 0 \pm 0 . 0 3 1
$$

$$
0 . 9 1 6 \pm 0 . 0 1 6
$$

$$
\overline { { 0 . 4 6 2 \pm 0 . 0 5 4 } }
$$

$$
0 . 3 6 6 \pm 0 . 0 3 5
$$

$$
0 . 8 7 0 \pm 0 . 0 0 2
$$

$$
\underline { { 0 . 9 3 5 \pm 0 . 0 0 1 } }
$$

$$
\overline { { 0 . 7 4 8 \pm 0 . 0 1 1 } }
$$

$$
0 . 4 9 3 \pm 0 . 0 2 3
$$

$$
0 . 9 4 6 \pm 0 . 0 0 2
$$

$$
0 . 4 6 9 \pm 0 . 0 4 7
$$

$$
0 . 9 2 2 \pm 0 . 0 0 3
$$

$$
0 . 5 5 0 \pm 0 . 0 2 1
$$

$$
0 . 7 5 3 \pm 0 . 0 0 4
$$

$$
\overline { { 0 . 9 1 2 \pm 0 . 0 0 2 } }
$$

$$
0 . 7 3 3 \pm 0 . 0 1 1
$$

$$
\overline { { 0 . 9 3 5 \pm 0 . 0 1 8 } }
$$

$$
0 . 5 6 4 \pm 0 . 0 2 0
$$

$$
0 . 9 0 3 \pm 0 . 0 3 6
$$

$$
0 . 5 8 6 \pm 0 . 0 6 2
$$

$$
0 . 4 0 2 \pm 0 . 0 5 5
$$

$$
0 . 9 4 7 \pm 0 . 0 0 0
$$

$$
0 . 8 8 3 \pm 0 . 0 0 6
$$

$$
0 . 9 2 3 \pm 0 . 0 0 4
$$

$$
0 . 7 2 2 \pm 0 . 0 1 5
$$

$$
\underline { { 0 . 8 1 6 \pm 0 . 0 1 6 } }
$$

$$
\mathbf { 0 . 9 5 0 \pm 0 . 0 0 1 }
$$

$$
0 . 8 5 3 \pm 0 . 0 0 8
$$

$$
0 . 8 0 9 \pm 0 . 0 1 3
$$

$$
\mathbf { 0 . 9 2 8 \pm 0 . 0 1 1 }
$$

$$
\mathbf { 0 . 9 1 3 \pm 0 . 0 0 1 }
$$

$$
\mathbf { 0 . 9 1 3 \pm 0 . 0 2 7 }
$$

$$
0 . 9 0 6 \pm 0 . 0 1 0
$$

$$
\underline { { 0 . 8 8 1 \pm 0 . 0 1 1 } }
$$

$$
\mathbf { 0 . 9 3 5 \pm 0 . 0 0 2 }
$$

$$
\pm { \overline { { 0 . 9 2 9 \pm 0 . 0 0 4 } } }
$$

Table 9: Comparative experiment on the objective function and label type. Ranked binary crossentropy loss (Rank) consistently shows stronger results, compared to triplet-based paired cross-entropy loss (2AFC) and L1 regression loss on the label. Our label has shown to be more helpful in training, compared to the large-scale human annotated KADID-10k [23]. PIPAL SROCC, mean ± standard deviation over five random seeds.
<table><tr><td></td><td colspan="6">FoMo (Ours)</td><td colspan="3">KADID-10K [23]</td></tr><tr><td>Label</td><td colspan="3"></td><td colspan="3">S</td><td colspan="3">DMOS</td></tr><tr><td>Objective</td><td>Rank</td><td>2AFC</td><td>Ll</td><td>Rank</td><td>2AFC</td><td>L1</td><td>Rank</td><td>2AFC</td><td>Ll</td></tr><tr><td>LPIPS-Alex [43]</td><td>0.733 ± 0.006</td><td>0.592 ± 0.002</td><td>0.440 ± 0.059</td><td>0.733 ± 0.006</td><td>0.592 ± 0.002</td><td>0.390 ± 0.049</td><td>0.625 ± 0.004</td><td>0.611 ± 0.003</td><td>0.577 ± 0.025</td></tr><tr><td>DINOv3 [35]</td><td>0.699 ± 0.006</td><td>0.622 ± 0.020</td><td>0.694 ± 0.008</td><td>0.697 ± 0.006</td><td>0.614 ± 0.009</td><td>0.695 ± 0.013</td><td>0.309 ± 0.030</td><td>0.145 ± 0.010</td><td>0.298 ± 0.010</td></tr></table>

driven the evolution of IQA annotation methodology. Early benchmark datasets such as LIVE [34], CSIQ [22], TID2013 [30] and KADID-10k [23] collected Mean Opinion Scores (MOS) as ground truth, implicitly assuming that averaged absolute ratings constitute a reliable global quality scale. PieAPP [31] challenged this directly, arguing that MOS-based annotations are unreliable because absolute quality ratings are inconsistent across raters and sessions: observers apply different scale calibrations and anchor their ratings differently, making cross-image comparisons ambiguous. This motivated a shift toward 2-alternative-forced-choice (2AFC) annotation, where raters compare two distortions relative to a reference rather than score in isolation. Concurrent to this finding, LPIPS [43] also adopts the 2AFC paradigm for labeling and constructs a large-scale dataset gathered from human annotations. PIPAL [18] extends it to GAN-based restorations via Elo-ranked pairwise judgments to expand the coverage of IQA evaluations. DreamSim [13] also collects a new dataset based on the 2AFC paradigm, but its main focus is on mid-level similarity rather than low-level similarity.

Despite this recent popularity of 2AFC-style labeling and training, the 2AFC setting carries its own structural weakness. Talebi et al. [38] point out that mini-batch pairwise optimization never explicitly sees the global ranking of images, and each gradient step accounts for only a small fraction of all possible comparisons; they demonstrate that regularizing with rank-centrality aggregation consistently improves human preference prediction. dipIQ [27] reinforces this directly by comparing pairwise RankNet and listwise ListNet training objectives on the same automatically generated quality-discriminable pairs, finding that the listwise variant consistently outperforms its pairwise counterpart, a clear empirical signal that optimizing global ordinal structure is superior to aggregating independent pairwise decisions. Yet all of these methods remain bottlenecked by human annotation cost and coverage.

## C.2 Diffusion Models as Perceptual Signal

There have been several efforts in exploiting diffusion models for perceptual similarity. DIFT [39] and Diffusion Hyperfeatures [26] employ diffusion features for structural similarity, but their methods are targeted for geometric correspondence task, rather than low-level perceptual distance. DiffSim [37] uses attention layer features in Stable Diffusion [33] to measure visual similarity, but targets style and instance-level consistency in generative customization settings rather than low-level perceptual fidelity. AnoDDPM [42] partially diffuses an image to an intermediate timestep and measures the pixel-level reconstruction divergence after denoising as an anomaly score, but restricted to detecting distributional outliers in medical images. DifFIQA [1] applies perturbation-robustness under DDPM noising as a face image quality signal, scoring faces by the shift in identity embedding between input and reconstructed image, an NR-IQA approach specific to facial content.

Our approach differs from all of the above on two axes. First, unlike feature-based approaches, our approach does not use diffusion model representations at inference time; the diffusion model is used only for generating the data samples and their labels. Second, unlike scoring approaches like AnoDDPM and DifFIQA, our approach uses the denosing generative process as a label generation mechanism, instead of using the generation result in distance computation.

## D Human Alignment Experiment

We present the web user-interface used for single-reference human study from the experiment of Sec. 3.1 are in Fig. 4 and 5, and additional examples of experiment from Sec. 3.2 are in Fig. 6 and 7.

## E Data Augmentation

One of a key advantage of our approach is its compatibility with a substantially broader range of data augmentation strategies than human-annotated alternatives permit. Pairwise preference datasets such as BAPPS [43] and NIGHTS [13] impose strict constraints on applicable augmentations: while label-preserving transformations such as random horizontal flips and in-plane rotations can be safely applied without altering perceptual judgments, aggressive spatial augmentations, most notably random resize cropping, are not applicable. Annotations in these datasets reflect global perceptual preferences elicited at a fixed resolution over entire image triplets; a crop that exposes only a local region may induce a different perceptual ordering and thereby contradict the original annotation.

![](images/bdd332fbc0a06f00b40b4333b0baca8efa7d7a89d177a151667316dc1f2fc478.jpg)  
Figure 4: Screenshot of the human annotation website for experiments in Sec. 3

![](images/908d2029881c36fc77084598e92e9e48f33f709e886d6a5fdd9bb2508377bdc4.jpg)  
Figure 5: Screenshot of the human annotation website for experiments in Sec. 3

Left pair reference variant (s = 19)

## Which pair shows two images that are more similar to each other? Compare the two images within each pair.

![](images/f1ec11f5302e11ceebaeb8ee5a462a3195c3c5ca12a646e1b78cfbda8b77f7fb.jpg)

![](images/3e26bd05f39e07cfe0ae5332d123330ea86a54aded4dbfe12c9683b685a3e48a.jpg)

gap = 1 steps label: Left pair is closer  
![](images/a7fcf2081ee3107c52aef36f9a3a83e3eee09a6fd1eb2c5ea014a49d3b559d6b.jpg)

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/d7f64534bae9406593b4c83901b779c35ff8cd3181cba592f847cc99e72dcfad.jpg)  
gap = 4 steps label: Right pair is closer

![](images/07d91311daf2be44c3cc4d54f681c700b994bb77a0ec2da16610957009e63410.jpg)

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/65f46836bc670cec9d255cc1fee00ae5aeaa3c8c1b8b2ef82978ee1b1d3a789b.jpg)

![](images/9d5757d3b8876f7485e1d758b197f64df0f29251f65d11386670698ef01f0a20.jpg)  
gap = 4 steps label: Right pair is closer

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/44ce48e2000ddac14aa7d9bf9f96a5cc0ab202ec63e7eafaf8fb4d04d1229dcc.jpg)

![](images/71fb527f045725d8c506d2661397259065db5a8a24c0710a8a877cc761e8e584.jpg)

![](images/b52379ee5b5ac5c1fff49935a3e225fa4bd22855279784168a382fc9e6f9b21d.jpg)  
gap = 10 steps label: Left pair is closer

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/f4430a46edf3059dfd27c61792b5652e452004983392e0a3aaae366264228d45.jpg)

![](images/9aabcfd5a8599cf87581998e839043a2d829166d0aa5568744eaeddcd7927851.jpg)  
gap = 11 steps label: Left pair is closer  
Figure 6: Examples from cross-reference human study, with each samples’ forked moments, the gap of forking moments and labels from FoMo provided.

Right pair reference variant (s = 4)

Which pair shows two images that are more similar to each other? Compare the two images within each pair.

gap = 14 steps label: Right pair is closer  
Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/0795f0c35bbe02aa8433ba3681bbde29183eb9529d69bdf658f2279c56dca400.jpg)

![](images/81489376447d884e01344957398c193a4d6da79ec54454173a80856babbd28d3.jpg)

![](images/f00585560d6967a2f5bbdb2f888a979572118ed9355907cc300eba43b62ee9d2.jpg)

![](images/686604931f89a8404dbc1bf7e8f3a116bbffaa95dbf145871574a4e3aaf998a7.jpg)

Which pair shows two images that are more similar to each other? Compare the two images within each pain  
![](images/bf867d07e8daa25aa4e6ada6a39ba3232a83fb0b069e6bc791e63dcde9b97342.jpg)

![](images/3431ee8df5a522c2397a7e19d3a7da6b12c14789fce2de864514db03763f3cfe.jpg)  
gap = 23 steps

label: Right pair is closer  
![](images/351ea63f9b5acb3ccd1d5154b8660dce5e6810e6f62ca02791c49fb183dbb9bb.jpg)

![](images/955e5e297b5e9260e130dc8074efd0bc2a906bdc31cd4a62510820b75dbe1512.jpg)

![](images/ac3bc872250984030a06efb04516a5224f9f5fa296517f3aaced27e1d877ff85.jpg)

![](images/d74a559ff1a6cfe8cbc4262383d1ffc3b7f7bfb17245d52f8f9c25c408db2d59.jpg)  
gap = 20 steps

![](images/3facb519fba825e410002929aba522b435f9a440d4016bfaf51ca34ab8bc5bfd.jpg)  
label: Left pair is closer

![](images/5a0e13c28966d21590030372838ba8666a48c8487efdc8f4f1049b5a286c20ad.jpg)

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/0fcd075cc74e60444b54b0d96d6757131861433606143cafea3f7d49ccc72bc6.jpg)

![](images/8831a5f310b4ccd97073c524a1dac8438ad7943ef910923e3b23364284171c30.jpg)  
gap = 32 steps

![](images/9f77aa51d2c6258aee0778999bed40183a166f6e98e282500a377e5451bcd6f3.jpg)  
label: Left pair is closer

![](images/0e2499a7a74f15e6ef90b558769b2010f0318c9e9833f9a4329e9d5007858127.jpg)

Which pair shows two images that are more similar to each other? Compare the two images within each pair.  
![](images/764b0b64e8fb4aea29089003a3b2c0fc5aa7a56527d40b6dcba37334301c8782.jpg)

![](images/918098bcdcffedd49a20dfa1e4c2f98ef0339814318ed3de72d757c0ae36ea53.jpg)  
Left pair reference variant (s = 46)

![](images/9af0f0c86e732d354b3e0e5b808d835e43d5ff2b57c38d8891c4212782ade33c.jpg)  
gap = 41 steps label: Left pair is closer

![](images/b8cd1b551046538a97adad007d480e2a81d28be1b202e043e8d0bc8540b197ac.jpg)

Figure 7: Examples from cross-reference human study, with each samples’ forked moments, the gap of forking moments and labels from FoMo provided.

Our approach enjoys better flexibility because the forking moment label is derived from a predefined noise schedule and can therefore be recomputed analytically under any spatial transformation. We describe how two canonical augmentations are accommodated within our framework. Both derivations rely on the established result that the diffusion noise schedule must be rescaled with image resolution [16, 3]. Concretely, the log signal-to-noise ratio (log-SNR) of the variance-preserving forward process shifts with resolution as

$$
\log { \mathrm { S N R } ( t ; r ) } = \log { \mathrm { S N R } ( t ; r _ { 0 } ) } + 2 \log ( \frac { r _ { 0 } } { r } ) ,\tag{4}
$$

where $r$ denotes the image resolution $( e . g .$ , the shorter spatial dimension in pixels), $r _ { 0 }$ is a reference resolution at which the baseline schedule is defined, and $t \in [ 0 , 1 ]$ is the continuous interpolation factor. Given the variance-preserving constraint $a _ { t } ^ { 2 } + b _ { t } ^ { 2 } = 1$ , the forward-process noise coefficient at resolution r are recovered as

$$
a _ { t } ^ { 2 } ( r ) = \sigma ( \log \mathrm { S N R } ( t ; r ) ) , b _ { t } ^ { 2 } ( r ) = 1 - a _ { t } ^ { 2 } ( r ) ,\tag{5}
$$

where $\sigma ( \cdot )$ denote the sigmoid operator.. We refer to the log-SNR value at the forking moment as $\lambda _ { s } = \log \dot { \mathrm { S N R } } ( t _ { 0 } ; \boldsymbol { r } _ { 0 } )$ , which serves as a resolution-invariant proxy for perceptual divergence in both cases below.

Resizing Rescaling an image from $r _ { 0 }$ to a new resolution $r _ { \mathrm { r e s i z e } }$ preserves the global scene content and structure. The relative noise level at which content is destroyed therefore remains unchanged, and so does the perceptual divergence between the reference and distorted images. Consequently, the forking timestep s is retained as the label after resizing. However, because the noise schedule is resolution-dependent (Eq. 4), the continuous interpolation factor t corresponding to s shifts implicitly. If s maps to interpolation factor $t _ { 0 }$ under the original schedule, then at resolution $r _ { \mathrm { r e s i z e } }$ the factor $t _ { 0 }$ must be updated to $t _ { \mathrm { r e s i z e } }$ such that

$$
\begin{array} { r } { \log \mathrm { S N R } ( t _ { \mathrm { r e s i z e } } ; r _ { \mathrm { r e s i z e } } ) = \log \mathrm { S N R } ( t _ { 0 } , r _ { 0 } ) , } \end{array}\tag{6}
$$

which amounts to a shift of $t _ { 0 }$ by $2 \log ( r _ { 0 } / r _ { \mathrm { r e s i z e } } )$ in log-SNR space.

Random Resize Cropping Cropping simultaneously alters the spatial resolution and the visible image content. Since the cropped region depicts only a portion of the original scene, the forking timestep s can no longer be assumed invariant. However, the continuous interpolation factor t, which encodes the relative signal-to-noise level at which the two images perceptually diverge, is a resolutionagnostic quantity and is preserved across the crop. This follows directly from the pixel-wise nature of the diffusion forward process: a noised image at step s is formed as $x _ { s } = a _ { t _ { 0 } } x _ { 0 } + b _ { t _ { 0 } } \epsilon$ , where the mixing ratio $t _ { 0 }$ is applied uniformly across all spatial locations. Any crop of $x _ { s }$ is therefore a crop of the same mixture at factor $t _ { 0 }$ , irrespective of the global image extent or resolution.

The label for the cropped region at resolution $r _ { \mathrm { c r o p } }$ is recomputed as follows. The interpolation factor $t _ { 0 }$ is retained from the original annotation, and $\lambda _ { s }$ is computed as above. The noise schedule is then rescaled to resolution $r _ { \mathrm { c r o p } }$ via Eq. 4, yielding a new log-SNR curve. Finally, the forking timestep $s _ { \mathrm { c r o p } }$ is recovered by identifying the discrete step $s \in \{ 0 , \ldots , S { - } 1 \}$ whose normalized time $s / \bar { S }$ maps to log-SNR closest to $\lambda _ { s }$ under the rescaled schedule. In the continuous-time limit this reduces to an exact inversion of the log-SNR function at $r _ { \mathrm { c r o p } }$

In summary, resizing preserves the forking timestep s while the interpolation factor t shifts, whereas random resize cropping preserves t while s must be recomputed. In both cases the true invariant is $\lambda _ { s } \mathrm { : }$ the log-SNR at the moment of perceptual divergence.

## F Comparison against Off-the-Shelf Metrics

Directly comparing off-the-shelf metrics to ours could make it hard to strictly ablate the benefits of our proposed approach. Thus, the results in the main paper, such as Table 1 is intended to be ablative. Most of the experimental settings are shared, from frozen backbone, prediction head initialization, optimizer, schedule and budget, etc., leaving the training data and the objective as the only variables. Yet, how our approach performs in comparison to off-the-shelf metrics is nonetheless worth establishing, and Table 10 presents the results, placing the public LPIPS-Alex, LPIPS-VGG, DISTS and DreamSim checkpoints beside the same architectures trained with FoMo supervision and evaluated under the same protocol. (In DreamSim, all images were resized to 224 resolution, following the protocol of DreamSim.)

Across the four architectures, FoMo supervision is comparable to the released checkpoints and slightly ahead overall, leading in half of the cells (25 of 48) and in all twelve for LPIPS-Alex. It is worth noting that this result is obtained under a single configuration applied to every backbone, shared unchanged across the three CNN backbones, with nothing tuned per architecture or per benchmark and without any human-annotated supervision. The released checkpoints, by contrast, each reflect considerable per-metric care: their own design choices, hyper-parameters and training sets. This result again proves the strength of FoMo, and further implies the potential that the metrics could still improve more with careful tuning of hyper-parameters for each model.

Table 10: Released off-the-shelf metrics compared with the same architecture trained with FoMo supervision. Better of each pair in bold. Released checkpoints are single deterministic models whereas FoMo columns are 5-seed means.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">LPIPS-Alex [43]</td><td colspan="2">LPIPS-VGG [43]</td><td colspan="2">DISTS [9]</td><td colspan="2">DreamSim [13]</td></tr><tr><td>Released</td><td>FoMo</td><td>Released</td><td>FoMo</td><td>Released</td><td>FoMo</td><td>Released</td><td>FoMo</td></tr><tr><td rowspan="3">PIPAL [18]</td><td>SROCC</td><td>0.620</td><td>0.733</td><td>0.612</td><td>0.683</td><td>0.672</td><td>0.615</td><td>0.759</td><td>0.776</td></tr><tr><td>KROCC</td><td>0.435</td><td>0.536</td><td>0.438</td><td>0.497</td><td>0.482</td><td>0.435</td><td>0.563</td><td>0.571</td></tr><tr><td>PLCC</td><td>0.623</td><td>0.767</td><td>0.668</td><td>0.712</td><td>0.685</td><td>0.654</td><td>0.780</td><td>0.762</td></tr><tr><td rowspan="3">TID2013 [30]</td><td>SROCC</td><td>0.745</td><td>0.785</td><td>0.670</td><td>0.663</td><td>0.708</td><td>0.691</td><td>0.812</td><td>0.801</td></tr><tr><td>KROCC</td><td>0.548</td><td>0.586</td><td>0.497</td><td>0.489</td><td>0.521</td><td>0.512</td><td>0.614</td><td>0.605</td></tr><tr><td>PLCC</td><td>0.753</td><td>0.816</td><td>0.749</td><td>0.746</td><td>0.755</td><td>0.766</td><td>0.746</td><td>0.831</td></tr><tr><td rowspan="3">CSIQ [22]</td><td>SROCC</td><td>0.923</td><td>0.938</td><td>0.883</td><td>0.859</td><td>0.930</td><td>0.918</td><td>0.911</td><td>0.894</td></tr><tr><td>KROCC</td><td>0.750</td><td>0.781</td><td>0.697</td><td>0.672</td><td>0.764</td><td>0.750</td><td>0.738</td><td>0.709</td></tr><tr><td>PLCC</td><td>0.920</td><td>0.944</td><td>0.906</td><td>0.888</td><td>0.938</td><td>0.931</td><td>0.928</td><td>0.900</td></tr><tr><td rowspan="3">LIVE [34]</td><td>SROCC</td><td>0.924</td><td>0.948</td><td>0.932</td><td>0.923</td><td>0.948</td><td>0.954</td><td>0.910</td><td>0.931</td></tr><tr><td>KROCC</td><td>0.751</td><td>0.791</td><td>0.765</td><td>0.748</td><td>0.793</td><td>0.804</td><td>0.745</td><td>0.769</td></tr><tr><td>PLCC</td><td>0.916</td><td>0.940</td><td>0.934</td><td>0.923</td><td>0.945</td><td>0.949</td><td>0.918</td><td>0.934</td></tr></table>

## G Per-Sample Label Variance

The forking construction is stochastic: two variants generated from the same reference at the same forking step are not identical, because the noise re-injected at the fork differs, yet both carry the same label. To quantify the resulting spread we take 120 ImageNet references, generate K = 8 variants at each of five forking steps, 4,800 images in total, and measure every variant’s distance to its reference. For one reference at one forking step this gives eight distances, of which we take the mean and the standard deviation. Table 11 reports both averaged over the 120 references, together with their ratio, the coefficient of variation (CoV). The spread is small in every regime: the standard deviation stays below 0.05 in absolute terms and at most 10.1% of the distance it accompanies, while the mean distance itself changes six- to eightfold across the schedule. Re-running the generator therefore perturbs a sample by far less than the label separates it from its neighbors, and the perturbation reorders two variants of the same reference in at most 2.8% (LPIPS-Alex) and 7.0% (DISTS) of comparisons. Since this noise is independent across the 480k training pairs and averages out over them, we regard it as negligible for training.

## H Per-Distortion-Type Analysis

Table 1 reports one SROCC per benchmark. This appendix asks which distortion families that number is built from. We recompute SROCC within each distortion type of all four benchmarks and compare FoMo against the strongest human-annotated recipe for the same backbone on the same benchmark. PIPAL is evaluated on its training split, the only one whose distortion types are identifiable, so its full-set values are not comparable to the validation numbers of Table 1.

With the Transformer backbone, FoMo supervision leads on almost every distortion family of every benchmark (Table 12). The gain is largest exactly where the backbone on its own is weakest, the super-resolution families of PIPAL, and the noise and contrast families of CSIQ. So the supervision, not the architecture, is what supplies the perceptual ordering. With the CNN backbone the picture i narrower, as one would expect of a network whose ImageNet features already encode a perceptual prior. FoMo leads throughout PIPAL, but on the three legacy synthetic-distortion benchmarks it trails on a majority of families, by margins of hundredths of a point.

Table 11: Spread of the measured distance across generation seeds. 120 references × 5 forking steps × K = 8 seeds. For each reference we take the eight distances obtained at one forking step and compute their mean and standard deviation; the table reports these averaged over the 120 references, with CoV their ratio. A larger s denotes a later fork, and hence a variant closer to the reference.
<table><tr><td rowspan="2">Forking step s</td><td colspan="3">LPIPS-Alex</td><td colspan="3">DISTS</td></tr><tr><td>mean d</td><td>std</td><td>CoV</td><td>mean d</td><td>std</td><td>CoV</td></tr><tr><td>5</td><td>0.692</td><td>0.043</td><td>6.4%</td><td>0.391</td><td>0.034</td><td>8.7%</td></tr><tr><td>15</td><td>0.518</td><td>0.046</td><td>9.1%</td><td>0.306</td><td>0.031</td><td>10.1%</td></tr><tr><td>25</td><td>0.371</td><td>0.030</td><td>7.9%</td><td>0.225</td><td>0.021</td><td>9.2%</td></tr><tr><td>35</td><td>0.233</td><td>0.011</td><td>4.7%</td><td>0.150</td><td>0.011</td><td>6.9%</td></tr><tr><td>45</td><td>0.090</td><td>0.002</td><td>2.7%</td><td>0.067</td><td>0.004</td><td>5.6%</td></tr></table>

The failures are consistent across both backbones and concentrate in three families (Table 13 lists every type): additive pixel noise, corruption confined to a small region, and global photometric shifts such as contrast change and mean shift. None of these occurs in our training data. A diffusion model re-synthesizes an image as a whole, so it never adds pixel-independent noise, never corrupts an isolated rectangle, and never applies a purely photometric change; supervision cannot teach what it never shows.

Table 12: Per-distortion-type summary on all four benchmarks. “Best baseline” is the strongest of the four human-annotated recipes for that backbone on that benchmark, selected independently per benchmark. “Full” is the score over the whole benchmark and “led” the number of distortion types on which FoMo is ahead. PIPAL uses its training split, the only one that exposes distortion types. Symmetrized native-resolution protocol, seed-0 checkpoints.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">types</td><td colspan="3">LPIPS-Alex</td><td colspan="3">DINOv3</td></tr><tr><td>best baseline</td><td>FoMo</td><td>led</td><td>best baseline</td><td>FoMo</td><td>led</td></tr><tr><td>PIPAL</td><td>7</td><td>0.611</td><td>0.681</td><td>7/7</td><td>0.259</td><td>0.525</td><td>7/7</td></tr><tr><td>TID2013</td><td>24</td><td>0.794</td><td>0.783</td><td>10/24</td><td>0.592</td><td>0.710</td><td>22/24</td></tr><tr><td>CSIQ</td><td>6</td><td>0.945</td><td>0.936</td><td>1/6</td><td>0.613</td><td>0.808</td><td>6/6</td></tr><tr><td>LIVE</td><td>5</td><td>0.952</td><td>0.942</td><td>2/5</td><td>0.748</td><td>0.896</td><td>5/5</td></tr></table>

## I Limitations

Despite the excellent performance in existing benchmarks. Our approach has a few limitations. One major limitation is that, since our approach generated distorted images with diffusion models, the metric models trained with our data and objective may fail in unseen image domains that are out-of-distribution, such as artificial distortions uncommon in nature. For scenarios like those, one could train a diffusion model on the new domain, and use it the trained diffusion model for generating new samples to train in that domain. This way, our approach can overcome its own limitation.

## J Ethics Statement

The human annotation study involved only perceptual preference judgments on image pairs, with no deception, no collection of personally identifiable information, and no risk beyond normal screen use.

Table 13: Every distortion type of all four benchmarks, against the strongest human-annotated recipe for the same backbone on the same benchmark. Bold marks the better of each pair. FoMo supervision leads on 20 of the 42 types with the CNN backbone and on 40 of the $4 2 { \overline { { \ } } }$ with the Transformer backbone.
<table><tr><td rowspan="2">Distortion type</td><td colspan="2">LPIPS-Alex</td><td colspan="2">DINOv3</td></tr><tr><td>best baseline</td><td>FoMo (Ours)</td><td>best baseline</td><td>FoMo (Ours)</td></tr><tr><td colspan="5">PIPAL [18] (training split)</td></tr><tr><td>SR (traditional)</td><td>0.577</td><td>0.669</td><td>0.098</td><td>0.618</td></tr><tr><td>SR (PSNR-oriented)</td><td>0.707</td><td>0.785</td><td>0.315</td><td>0.721</td></tr><tr><td>SR (kernel mismatch)</td><td>0.560</td><td>0.650</td><td>0.349</td><td>0.537</td></tr><tr><td>SR (GAN-based)</td><td>0.501</td><td>0.565</td><td>0.172</td><td>0.471</td></tr><tr><td>Denoising</td><td>0.688</td><td>0.757</td><td>0.437</td><td>0.693</td></tr><tr><td>Mixture</td><td>0.570</td><td>0.665</td><td>0.370</td><td>0.599</td></tr><tr><td>Traditional</td><td>0.586</td><td>0.628</td><td>0.125</td><td>0.346</td></tr><tr><td colspan="5">TID2013 [30]</td></tr><tr><td>Additive Gaussian noise</td><td>0.809</td><td>0.766</td><td>0.432</td><td>0.808</td></tr><tr><td>Additive noise, colour comp.</td><td>0.742</td><td>0.690</td><td>0.352</td><td>0.735</td></tr><tr><td>Spatially correlated noise</td><td>0.717</td><td>0.744</td><td>0.659</td><td>0.794</td></tr><tr><td>Masked noise</td><td>0.785</td><td>0.770</td><td>0.133</td><td>0.608</td></tr><tr><td>High-frequency noise</td><td>0.847</td><td>0.806</td><td>0.506</td><td>0.848</td></tr><tr><td>Impulse noise</td><td>0.552</td><td>0.527</td><td>0.589</td><td>0.633</td></tr><tr><td>Quantisation noise</td><td>0.786</td><td>0.764</td><td>0.657</td><td>0.828</td></tr><tr><td>Gaussian blur</td><td>0.929</td><td>0.931</td><td>0.488</td><td>0.785</td></tr><tr><td>Image denoising</td><td>0.857</td><td>0.871</td><td>0.740</td><td>0.868</td></tr><tr><td>JPEG</td><td>0.897</td><td>0.887</td><td>0.710</td><td>0.889</td></tr><tr><td>JPEG2000</td><td>0.914</td><td>0.934</td><td>0.756</td><td>0.881</td></tr><tr><td>JPEG transmission errors</td><td>0.882</td><td>0.898</td><td>0.645</td><td>0.803</td></tr><tr><td>JPEG2000 transmission errors</td><td>0.799</td><td>0.791</td><td>0.568</td><td>0.696</td></tr><tr><td>Non-eccentricity pattern noise</td><td>0.782</td><td>0.822</td><td>0.279</td><td>0.800</td></tr><tr><td>Local block-wise distortion</td><td>0.335</td><td>0.349</td><td>0.401</td><td>0.277</td></tr><tr><td>Mean shift Contrast change</td><td>0.778</td><td>0.737</td><td>0.126</td><td>0.576</td></tr><tr><td></td><td>0.434</td><td>0.410</td><td>-0.029</td><td>-0.159</td></tr><tr><td>Colour saturation change</td><td>0.791</td><td>0.782</td><td>0.271</td><td>0.757</td></tr><tr><td>Multiplicative Gaussian noise</td><td>0.742</td><td>0.691</td><td>0.478</td><td>0.748</td></tr><tr><td>Comfort noise</td><td>0.874</td><td>0.877</td><td>0.628</td><td>0.890</td></tr><tr><td>Lossy compression of noisy img.</td><td>0.914</td><td>0.901</td><td>0.734</td><td>0.885</td></tr><tr><td>Colour quantisation with dither</td><td>0.812</td><td>0.786</td><td>0.592</td><td>0.837</td></tr><tr><td>Chromatic aberrations</td><td>0.880</td><td>0.890</td><td>0.605</td><td>0.783</td></tr><tr><td>Sparse sampling and reconstr.</td><td>0.925</td><td>0.943</td><td>0.828</td><td>0.915</td></tr><tr><td colspan="5"></td></tr><tr><td>AWGN</td><td>CSIQ [22] 0.940</td><td>0.913</td><td>0.459</td><td>0.891</td></tr><tr><td>Gaussian blur</td><td>0.960</td><td>0.956</td><td>0.625</td><td>0.941</td></tr><tr><td>Contrast change</td><td>0.949</td><td>0.929</td><td>0.076</td><td>0.863</td></tr><tr><td>Pink noise</td><td>0.945</td><td>0.917</td><td>0.593</td><td>0.891</td></tr><tr><td>JPEG</td><td>0.958</td><td>0.949</td><td>0.800</td><td>0.955</td></tr><tr><td>JPEG2000</td><td>0.942</td><td>0.945</td><td>0.781</td><td>0.947</td></tr><tr><td colspan="5">LIVE [34]</td></tr><tr><td>Fast fading</td><td>0.962</td><td>0.965</td><td>0.704</td><td>0.960</td></tr><tr><td>Gaussian blur</td><td>0.962</td><td>0.964</td><td>0.489</td><td>0.903</td></tr><tr><td>JPEG2000</td><td>0.950</td><td>0.941</td><td>0.711</td><td>0.933</td></tr><tr><td>JPEG</td><td>0.965</td><td>0.959</td><td>0.841</td><td>0.962</td></tr><tr><td>White noise</td><td>0.961</td><td>0.876</td><td>0.908</td><td>0.966</td></tr></table>

Participants were informed of the task nature prior to annotation. The study was determined exempt from formal IRB review under the minimal-risk behavioral research exemption.

## K Broader Impacts

Positive Societal Impact Automating perceptual label generation reduces reliance on costly human annotation, lowering the barrier to building human-aligned IQA metrics. Better metrics improve evaluation pipelines across image restoration and synthesis, benefiting applications in medical imaging, compression, and accessibility.

Negative Societal Impact More accurate perceptual metrics could be exploited to optimize generative models toward visually convincing outputs that conceal manipulations, potentially aiding synthetic media misuse. The metric may also inherit perceptual biases from the diffusion model used to generate training labels.