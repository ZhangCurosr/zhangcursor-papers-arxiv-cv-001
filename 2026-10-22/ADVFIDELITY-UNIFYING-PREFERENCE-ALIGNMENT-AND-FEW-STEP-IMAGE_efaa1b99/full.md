# ADVFIDELITY: UNIFYING PREFERENCE ALIGNMENT AND FEW-STEP IMAGE GENERATION WITHIN REIN-FORCEMENT LEARNING

Sicong Kang<sup>1,∗</sup> Fangao Zeng<sup>1,∗,†</sup> Yuhan Li<sup>2,∗</sup> Hao Zhou<sup>1</sup> Mengfei Xu<sup>1</sup>

Wenxiang Shang<sup>1</sup> Wei Li<sup>1</sup> Bingbing Ni<sup>2</sup> Pipei Huang<sup>1</sup>

<sup>∗</sup>Equal Contribution <sup>†</sup>Project Leader

<sup>1</sup>Taobao & Tmall group of Alibaba <sup>2</sup>Shanghai Jiao Tong University

## ABSTRACT

Practical image generation requires both human preference alignment and fewstep efficiency, yet combining reinforcement learning (RL) with distillation is often multi-stage and may erode RL gains We observe that few-step RL progressively improves structural coherence, even with specialized rewards, but leaves low-level artifacts that these rewards may fail to penalize. To address this gap, we propose AdvFidelity, which integrates artifact-sensitive fidelity supervision into RL through an online adversarial reward. This enables joint reward alignment and high-fidelity few-step generation within a single RL stage, without distribution matching distillation (DMD) or other explicit distillation objectives. We train a discriminator to rank high-quality references (positives) above contentmatched reconstructions (negatives) produced by the current generator. Reconstructing moderately noised references largely preserves style and scene layout while exposing low-level degradations. This encourages the discriminator to assess artifacts rather than content differences. With negatives reconstructed online, alternating updates adapt fidelity supervision to the generator’s evolving failure modes. References can come from existing target-domain data or full-step model samples. Built on LeapAlign, AdvFidelity backpropagates gradients from both rewards through the few-step sampling trajectories used at inference. Across two model backbones and three task settings, AdvFidelity matches full-step LeapAlign in alignment, task performance, and overall generation quality and remains highly competitive with other evaluated full-step RL methods, using only 4 or 8 steps without CFG.

## 1 INTRODUCTION

Few-step distillation and reinforcement learning (RL) are two important procedures in the posttraining of image generation models. In real-world production, they are usually performed sequentially: RL is first applied for human preference alignment (Liu et al., 2026; Xue et al., 2025a; Zheng et al., 2025; Xue et al., 2025b), and few-step distillation is subsequently performed for efficient inference (Song et al., 2023; Kim et al., 2024; Luo et al., 2023; Yin et al., 2024a). This sequential pipeline is cumbersome and often suboptimal, as distillation may wash out the reward gains obtained from RL. Recent works, such as DMDR (Jiang et al., 2025) and RTDMD (Huang et al., 2026), adopt a joint training approach: they first warm up with distribution matching distillation (DMD) (Yin et al., 2024b) and then optimize the RL and DMD objectives in parallel.

During joint optimization, these methods use the pretrained base model, which is not aligned with the target reward, as a fixed reference score model for DMD. Matching this base distribution may oppose the policy’s shift toward reward-preferred outputs. This raises a practical question: does introducing DMD compromise the gains achieved by the same RL method in the full-step setting? Existing evaluations primarily compare joint methods with base models and other few-step generators, with limited comparisons against corresponding full-step RL baselines.

In this work, we remove the DMD constraint of matching a frozen, RL-agnostic reference distribution. Our framework jointly learns reward alignment and few-step generation within a single RL stage, matching the alignment performance and overall generation quality of its full-step RL baseline without a separate distillation warm-up. We build on a key observation: RL pipelines that directly optimize few-step rollouts (Zheng et al., 2025; Liang et al., 2026) already improve few-step image quality during training, even with highly specialized rewards (e.g., the OCR reward in Figure 1a). However, these gains are incomplete: few-step outputs retain subtle noise artifacts that task rewards may fail to penalize (Figure 1b), especially at later training iterations. This motivates an artifact sensitive fidelity reward that complements task-specific rewards, enabling joint reward alignment and artifact suppression.

Prompt: A beautifully crafted birthday cake topper shaped like "30 Years Young", adorned with sparkly frosting and shimmering decorations, set against a backdrop of a cozy, candlelit birthday party.  
![](images/44df3ac7ddf01a16345f058f96f576940d2897f8334319bf654e582456df7108.jpg)  
Iter 0

![](images/3a542948c4bf0e91479b715fe745b3a088ea12359932ea1a791793ddfe34c487.jpg)  
Iter 200

![](images/9cb8f3faa778f06b37dd309970cd374cf789f057d76614ac0186b25b29e26c41.jpg)  
Iter 700

![](images/57d62611048a88f35176ff5edd8495415750f79897622e3344b50eaa7b71fb30.jpg)  
Iter 1000

(a) Few-step RL with a sparse OCR reward: collapsed generations regain structural coherence during training, while grid-pattern artifacts persist after the reward is satisfied Prompt: A man is sitting alone on a park bench. Prompt: Stock market chart depicted in cave drawing  
![](images/8a9197856679a813b51168cad8464689b9b703bcb5e657c2304fc62dc1f22004.jpg)  
HPSv2: 0.3775, HPSv3: 15.92

![](images/ccc29aa8b611bb98301dad20b156a8dc1f01d260c9e2696d9f89a19d2b870907.jpg)  
HPSv2: 0.3644, HPSv3: 13.78

![](images/4c64424f03d1e334ea6981d94d4a1e628a1f442a8195e2ad6d9753334a2a75c6.jpg)  
HPSv2: 0.3615, HPSv3: 14.29

![](images/f754839303929a75ac5ad83c6128eabd1ccbe80a81e173602e854950a341616a.jpg)  
HPSv2: 0.3385, HPSv3: 13.22  
(b) Artifact-laden generations (left of each pair) receive preference scores comparable to or even higher than those of clean images, providing insufficient supervision to recover the clean image distribution.

## Figure 1: Motivation of AdvFidelity.

We propose AdvFidelity, which uses an adversarially trained discriminator (Goodfellow et al., 2020) as a fidelity reward. Its design addresses two practical challenges. First, a fixed discriminator may miss artifacts that emerge as the policy evolves. We therefore update the generator and discriminator alternately, training the discriminator on high-quality references and negatives reconstructed by the current generator to adapt fidelity supervision to changing failure modes. Second, differences in style, semantics, or layout can provide shortcuts for discrimination. We construct content-matched pairs by moderately noising each reference and reconstructing it with the current generator, largely preserving its high-level appearance. This reduces high-level differences and encourages the discriminator to focus on low-level artifacts.

Fidelity references can come from full-step base-model outputs, existing target-domain images, or samples from a model that covers the desired visual domain. Base-model outputs are often suitable for rewards targeting subject identity preservation, text accuracy, or compositional correctness, which improve task correctness without substantially changing the desired visual appearance. High-quality real images can also serve directly as references in industrial applications such as ecommerce and film production, where photorealism is a priority.

We instantiate AdvFidelity with LeapAlign (Liang et al., 2026) and evaluate FLUX.1-dev with the HPSv2 reward on HPDv2 and GenEval prompts, and Qwen-Image-Edit-2511 on virtual dressing with a DINO similarity reward that encourages garment consistency and realism. Using only 4 or 8 sampling steps without classifier-free guidance (CFG), AdvFidelity matches full-step LeapAlign in alignment, task performance, and overall generation quality, and remains highly competitive with other evaluated full-step RL methods. The LeapAlign comparison uses the same base model, reward, and task data, with gains on multiple metrics. These settings use target-domain model sampling, base-model sampling, and target-data reuse, respectively.

Our contributions are summarized as follows.

• We propose AdvFidelity, a framework that jointly learns reward alignment and few-step generation within a single RL stage, without distillation warm-up or parallel DMD training. It removes DMD’s constraint of matching a frozen, RL-agnostic base distribution.

• We develop an online adversarial fidelity reward trained on content-matched positive– negative pairs. Online updates adapt the discriminator to evolving artifacts, while matched reconstruction reduces high-level differences and encourages the discriminator to focus on low-level quality.

• Experiments across two model backbones and three task settings demonstrate that, using only 4 or 8 sampling steps without CFG, AdvFidelity matches full-step LeapAlign in alignment, task performance, and overall generation quality, and remains highly competitive with other evaluated full-step RL methods.

## 2 RELATED WORK

Reinforcement Learning. Reinforcement learning has become a major approach for aligning image generation models with human preferences and task-specific criteria. Policy-gradient methods formulate denoising as a Markov Decision Process and optimize terminal rewards. DDPO (Black et al., 2024) establishes this paradigm, and Flow-GRPO (Liu et al., 2026), DanceGRPO (Xue et al., 2025b), and MixGRPO (Li et al., 2025) adapt group-relative policy optimization (Shao et al., 2024) to diffusion and flow models. DPO-based methods (Rafailov et al., 2023; Wallace et al., 2024; Liang et al., 2025) instead learn from preference pairs without an explicit reward model. AWM (Xue et al., 2025a) and DiffusionNFT (Zheng et al., 2025) use reward-dependent objectives on forward-noised samples. Direct-gradient methods, including ReFL (Xu et al., 2023), DRaFT (Clark et al., 2024), and DRTune (Wu et al., 2024), propagate differentiable reward gradients through sampling, with different gradient truncation and trajectory designs. LeapAlign (Liang et al., 2026) uses leap trajectories to extend reward gradients to arbitrary generation steps. In pipelines such as DiffusionNFT and LeapAlign, reduced-step rollouts or shortened gradient paths primarily serve to reduce training cost, rather than explicitly train a few-step inference model. We study how fidelity supervision can support reward alignment and few-step generation at the inference sampling budget.

Few-Step Distillation. Few-step distillation reduces the sampling budget of image generation models to one or a few steps. Consistency-based approaches, including Consistency Models (Song et al., 2023), Latent Consistency Models (Luo et al., 2023), and Phased Consistency Models (Wang et al., 2024), promote consistent predictions across noise levels or trajectory phases. Distribution Matching Distillation (DMD) (Yin et al., 2024b) estimates distribution-matching gradients using a fixed reference score model and an online fake-score model. DMD2 (Yin et al., 2024a) removes the original regression loss, improves fake-score tracking, and incorporates adversarial supervision from real images. These methods learn few-step generation through explicit consistency or distributionmatching objectives. Applying distillation after RL may reduce reward gains. AdvFidelity instead learns few-step generation through task and fidelity rewards within RL, without an explicit distillation objective.

Joint Reward Alignment and Few-Step Generation. DMDR (Jiang et al., 2025) and RT-DMD (Huang et al., 2026) combine reward optimization with DMD after a distillation cold start, retaining fixed-reference score supervision and an online fake-score model. Their evaluations mainly compare with base models and other few-step generators. DMDR includes full-step ReFL comparisons, but its GRPO variant lacks a corresponding baseline. RTDMD reports near-parity with a published full-step Flow-GRPO baseline on GenEval, but does not establish that their training data and reward configurations are comparable, or report results on other prompt distributions for that model. Although it also uses HPS and PickScore as training rewards, it does not report corresponding full-step RL comparisons in these preference-alignment settings. These comparisons leave preservation of full-step RL gains insufficiently established.

Reward-Instruct (Luo et al., 2026) learns few-step generators through regularized reward maximization without diffusion distillation losses. It reports quality metrics and full-step ReFL and DRaFT comparisons, but not its own full-step counterpart. It maintains quality through parameter regularization, stochastic sampling, and fixed pretrained rewards. AdvFidelity learns an online fidelity reward and compares against full-step LeapAlign, its underlying RL method.

Online Adversarial Rewards. Adv-GRPO (Mao et $\mathrm { { a l . } }$ , 2026) trains an adversarial reward to mitigate reward hacking and align generations with reference domains; we follow its use of a frozen DINOv2 (Oquab et al., 2023) backbone and a trainable head. Its reported setup contrasts Qwen-Image (Wu et al., 2025) references with SD3 rollouts without constructing content-matched reference–reconstruction pairs. Consequently, fidelity differences are confounded with sourcemodel characteristics, making style or content differences available as discrimination shortcuts. AdvFidelity combines task and fidelity rewards without DMD and constructs each negative by reconstructing a moderately noised reference with the current generator (Sec. 3.4). This largely preserves style and scene layout, reducing high-level shortcuts and encouraging the discriminator to focus on low-level artifacts. Online updates adapt this supervision as the policy evolves. Our ablations support the benefits of both adaptive fidelity supervision and content-matched negatives (Sec. 4.4).

## 3 METHOD

## 3.1 PRELIMINARIES: RL ON FEW-STEP ROLLOUTS

We consider text-conditioned flow-matching generators that transport a noise sample $\epsilon \sim \mathcal { N } ( 0 , I )$ to an image conditioned on a prompt c. With a sampling budget of $\tau$ denoising steps, the resulting policy is written as $x \sim \pi _ { \theta } ( \cdot \mid c ; \tau )$ We use text-conditioned notation for simplicity; in image editing tasks, the conditioning also includes the input images.

RL-based post-training optimizes the policy against a reward $r ( x , c )$ encoding human preferences or task-specific criteria:

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { c \sim \mathcal { C } } \mathbb { E } _ { x \sim \pi _ { \theta } ( \cdot | c ; \tau ) } \left[ r ( x , c ) \right] .\tag{1}
$$

Advantage-based methods, such as GRPO (Shao et al., 2024), sample groups of rollouts and normalize their rewards into advantages. Reward-feedback (ReFL) methods (?Liang et al., 2026) instead use differentiable rewards to backpropagate gradients through the sampling process. We instantiate AdvFidelity with LeapAlign (Liang et al., 2026), a ReFL-family framework that propagates reward gradients through leap-based trajectories. The generator is trained using the few-step sampling bud get used at inference.

## 3.2 OVERVIEW OF ADVFIDELITY

AdvFidelity builds on the observation that reward optimization on few-step rollouts can progressively recover structural coherence, while low-level artifacts persist under preference rewards alone. To supply the missing fidelity supervision, we augment the preference or task-specific reward with a fidelity reward $r _ { \mathrm { { f i d } } }$ learned by an online discriminator:

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { c \sim \mathcal { C } } \mathbb { E } _ { x \sim \pi _ { \theta } ( \cdot | c ; \tau ) } \big [ r _ { \mathrm { s p e c } } ( x , c ) + \lambda r _ { \mathrm { f i d } } ( x ) \big ] ,\tag{2}
$$

where $r _ { \mathrm { s p e c } }$ measures human preference or task-specific criteria, $r _ { \mathrm { f i d } } ( x ) = D _ { \phi } ( x )$ assesses visual fidelity, and λ balances the two rewards. The fidelity reward is non-stationary because the discriminator is updated alongside the policy (Sec. 3.3). Under ReFL, we follow LeapAlign (Liang et al., 2026) and minimize the thresholded reward loss $\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } \big [ \operatorname* { m a x } ( 0 , m - \bar { r } ( x , \bar { c } ) ) \big ]$ , with $r ( x , c ) = r _ { \mathrm { s p e c } } ( x , c ) + \lambda D _ { \phi } ( x )$ . This loss suppresses further reward updates for samples whose combined reward reaches the margin m.

Figure 2 illustrates the alternating generator and discriminator updates. The discriminator learns from high-quality reference images and content-matched negatives reconstructed by the current generator. References may come from existing data or be generated as needed; their selection depends on the visual requirements of the target task (Sec. 3.4). Starting from a pretrained generator, AdvFidelity jointly learns preference alignment and high-fidelity few-step generation within a single RL stage, without a separate distillation warm-up or a teacher-provided score or trajectory objective.

![](images/238e7ee5fa043ac872300454300c051d8b6f72343c5aa00b4da5fe398a91f640.jpg)  
Figure 2: Framework of AdvFidelity. (a) Joint RL update: the few-step generator $G _ { \theta } \left( \pi _ { \theta } \right)$ rolls out a trajectory from Gaussian noise $x _ { 1 }$ to the generated image $x _ { 0 } .$ , and is updated with both the task reward $r _ { \mathrm { s p e c } }$ and the fidelity reward $D _ { \phi } .$ . (b) Content-matched pair: a high-quality reference image $x ^ { + }$ is perturbed with moderate noise and reconstructed by the current $G _ { \theta }$ to form a matched negative $x ^ { - }$ . (c) Online update: the discriminator $D _ { \phi }$ is trained on these pairs with a BT loss and supplies the fidelity reward for subsequent generator updates. Each training iteration performs (a) once and then repeats pair construction and the discriminator update in (b)–(c).

## 3.3 ADVERSARIAL FIDELITY REWARD

Discriminator as reward. Following Mao et al. (2026), $D _ { \phi }$ consists of a frozen DINOv2 (Oquab et al., 2023) backbone and a lightweight trainable MLP head $h _ { \phi }$ . The head operates on the [CLS] token, without directly using patch tokens, and input images are evaluated at the RL resolution. The frozen backbone provides a stable feature space, while the online-trained head adapts to the generator’s artifacts. Its score is used directly as the fidelity reward, $r _ { \mathrm { f i d } } ( x ) = D _ { \phi } ( x )$

Training the discriminator online. We train $D _ { \phi }$ to rank a high-quality reference $x ^ { + }$ above its matched reconstruction $x ^ { - }$ , produced by the current policy (Sec. 3.4), using a Bradley–Terry (BT) loss:

$$
\mathcal { L } _ { D } = \mathbb { E } _ { ( x ^ { + } , x ^ { - } ) } \bigl [ - \log \sigma \bigl ( D _ { \phi } ( x ^ { + } ) - D _ { \phi } ( x ^ { - } ) \bigr ) \bigr ] ,\tag{3}
$$

where $\sigma$ is the sigmoid function. Because the [CLS] token is a global representation, whole-image scores may be insensitive to localized defects such as grid patterns and texture noise. We therefore apply the same [CLS]-based scoring head to whole images and randomly cropped regions during generator and discriminator training. Cropping makes local defects occupy more of the input and can reduce reliance on global cues such as overall tone or border effects, without introducing a separate patch-token scorer.

Adapting to evolving artifacts. Generator updates can change the artifacts encountered by the discriminator. We therefore regenerate negatives with the current policy and perform multiple discriminator updates per generator update. The update ratio is specified in the implementation details. During generator updates, the discriminator parameters are held fixed while gradients propagate through its score; during discriminator updates, only the MLP head is optimized. This alternating process allows the fidelity reward to adapt as the generator changes, reducing the risk that new artifacts escape the supervision of a fixed reward.

Compatibility with other RL methods. The fidelity reward is a scalar and can be incorporated into different RL objectives. In our ReFL implementation, its gradient is propagated through the sampling process together with that of $r _ { \mathrm { s p e c } }$ . In advantage-based methods such as GRPO, the same score can contribute to the rewards used to compute rollout advantages. Our experiments use the LeapAlign implementation.

## 3.4 CONTENT-MATCHED PAIR CONSTRUCTION

The BT loss learns any features that consistently distinguish positives from negatives. If unrelated image populations are used, differences in style, semantics, or layout can become shortcuts for discrimination. We instead construct content-matched pairs by reconstructing each reference with the current generator, largely preserving its high-level appearance while exposing low-level degrada tions.

Noise-anchored negatives. Given a reference image $x ^ { + }$ , we introduce moderate noise at level t:

$$
x _ { t } = ( 1 { - } t ) x ^ { + } + t \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{4}
$$

The current generator then completes the remaining denoising from $x _ { t }$ to obtain $x ^ { - }$ , using one or two steps in practice. Reconstruction uses the conditioning associated with $x ^ { + } ;$ for text-to-image generation, this is a paired prompt or a caption; for image editing, it also includes the corresponding input images. With moderate noise $( t \leq 0 . 5 ) , x _ { t }$ retains much of the reference structure, allowing the reconstruction to largely preserve style and scene layout. The remaining differences are thus more concentrated in fine details, encouraging the discriminator to assess fidelity rather than high-level content. In all three experimental settings, the noise level is randomly sampled from {0.3, 0.5} for each matched negative. Negatives are reconstructed online, so the pairs reflect the current generator’ failure modes.

Choosing reference images. Fidelity references, or anchors, should cover the target visual quality and style, but need not maximize the task reward, which is optimized directly through $r _ { \mathrm { s p e c } }$ . We choose among three acquisition strategies according to visual compatibility and data availability:

• Base-model sampling. When the base model’s full-step outputs cover the target visual quality and style, we use them directly as anchors. This strategy is suitable when reward optimization primarily improves task correctness within a visual domain already covered by the base model. For GenEval, these are 50-step FLUX.1-dev outputs.

• Target-data reuse. Suitable existing target-domain images can be reused directly, regardless of base-model coverage. Sources include high-quality real images and existing highquality generated images. When RL follows target-domain supervised fine-tuning (SFT), its ground-truth images can likewise be reused for both photorealistic and stylized tasks. For virtual dressing, real ground-truth try-on images from the training split serve as references.

• Target-domain model sampling. When base-model coverage is insufficient and suitable data are unavailable or disallowed by the experimental protocol, we sample a model that produces high-quality images in the target domain. An existing RL-aligned model is one possible source. Its samples can be prepared once and reused across runs with the same target visual requirements; the source model’s scores and denoising trajectories are not used during AdvFidelity training. For HPDv2, we use 50-step samples from an HPSv2-aligned LeapAlign checkpoint.

All three strategies use the same content-matched pair construction. References may be stored beforehand or generated as needed; a fixed reference set is not required.

## 3.5 ADVERSARIAL TRAINING PROCEDURE

Algorithm 1 summarizes the alternating updates. In our FLUX experiments, the policy is initialized from the original FLUX.1-dev model. Each generator update optimizes both rewards on a few-step rollout from Gaussian noise. Each discriminator update obtains a reference from the chosen source and reconstructs a matched negative using the current generator. Preference alignment and fewstep generation are therefore learned jointly, while reference acquisition can use existing images or model-generated samples.

Under ReFL, gradients of the discriminator score flow into the generator, inducing an adversarial interaction analogous to GAN training. Here, the discriminator supplies one reward within a multireward objective, and its training negatives are content-matched reconstructions rather than unrelated generations. Its role is to supervise the fidelity of few-step outputs as the task reward improves preference or task-specific performance.

Algorithm 1 AdvFidelity Training   
Require: pretrained policy $\pi _ { \theta } ,$ , discriminator $D _ { \phi } ,$ , task reward $r _ { \mathrm { s p e c } } ,$ reference source ${ \mathcal { R } } ,$ sampling   
budget τ, noise level t, reward weight λ, discriminator update ratio K   
1: while not converged do   
2: Sample conditioning c and rollouts $x \sim \pi _ { \theta } ( \cdot \mid c ; \tau )$   
3: $r \gets r _ { \mathrm { s p e c } } ( x , c ) + \lambda D _ { \phi } ( x )$ ; update θ with the thresholded reward loss (Sec. 3.2)   
4: for $k = 1$ to K do   
5: Obtain a reference and its conditioning $( x ^ { + } , c ^ { + } )$ from R   
6: Sample $\epsilon \sim \mathcal { N } ( 0 , I ) ; x _ { t } \gets ( 1 { - } t ) x ^ { + } \dot { + } t \epsilon$   
7: Complete denoising from $x _ { t }$ with the current policy conditioned on $c ^ { + }$ to obtain $x ^ { - }$   
8: Update $\phi$ with the BT loss on $( x ^ { + } , x ^ { - } )$ (Eq. 3)   
9: end for   
10: end while

## 4 EXPERIMENTS

We evaluate AdvFidelity on preference alignment with HPDv2 prompts, compositional generation with GenEval prompts, and a production-oriented virtual dressing task. The first two settings follow the experimental protocols of LeapAlign (Liang et al., 2026), including its training and evaluation prompt sets and evaluation procedures, and initialize AdvFidelity from the original FLUX.1-dev model. AdvFidelity jointly learns reward alignment and few-step generation in a single generator training stage, with the fidelity discriminator trained online from the chosen references. AdvFidelity uses 4 or 8 sampling steps for FLUX and 8 steps for virtual dressing, without classifier-free guidance (CFG). We compare against full-step LeapAlign RL trained on the same base model in each setting, including both CFG and non-CFG variants for virtual dressing. Together, these experiments evaluate the three reference acquisition strategies in Sec. 3.4 across different visual domains and rewards.

## 4.1 PREFERENCE ALIGNMENT ON HPDV2

Implementation Details. We fine-tune FLUX.1-dev on the HPDv2 prompt set, which comprises 50,000 training prompts and 400 test prompts, using HPSv2 (Wu et al., 2023) as the preference reward. We report HPSv2, PickScore (Kirstain et al., 2023), ImageReward (Xu et al., 2023), and the alignment and image-quality sub-scores of UnifiedReward (Wang et al., 2025). Following LeapAlign, the table labels HPSv2 as an in-domain metric and PickScore, ImageReward, and UnifiedReward as out-of-domain metrics. HPSv3 (Ma et al., 2025) is shaded and reported separately for reference only, as it can assign misleading scores to images with few-step artifacts; Appendix B examines artifact sensitivity on content-matched pairs.

We compare against DanceGRPO (Xue et al., 2025b), MixGRPO (Li et al., 2025), ReFL (Xu et al., 2023), DRTune (Wu et al., 2024), DRaFT-LV (Clark et al., 2024), and LeapAlign (Liang et al., 2026). For HPDv2 and GenEval, results for these methods and the pretrained baseline are taken from LeapAlign. RTDMD<sup>∗</sup> denotes the officially released four-step FLUX.1-dev RTDMD (Huang et al., 2026) checkpoint evaluated at 4 and 8 NFEs. The public FLUX.1-dev RTDMD configurations use 25,432 PickScore prompts for DMD warm-up and 50,000 GenEval prompts with CLIPScore, HPSv2, PickScore, and GenEval rewards for RL. For a fairer comparison, we also retrain RTDMD using its released code at 4 and 8 steps with the same task training prompts and HPSv2 reward as AdvFidelity under LeapAlign’s protocol; unmarked RTDMD entries report these runs.

The fidelity discriminator consists of a frozen DINOv2-Large (Oquab et al., 2023) encoder and a trainable MLP head. We use crop sizes of 256 and 384 pixels with a crop-level loss weight of 0.3. The generator and discriminator are optimized with AdamW at learning rates of $1 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 5 }$ , respectively. The gradient discount factor is 0.3, the reward margin m is 0.55, and the fidelity reward weight λ is 0.05. The discriminator performs four updates per generator update.

Reference Images and Matched Negatives. We use target-domain model sampling (Sec. 3.4). In this setting, HPSv2 optimization encourages pronounced cartoon and fantasy styles, substantially shifting the visual distribution from that of the original FLUX.1-dev model. Fidelity reference should cover this preferred appearance. Although HPDv2 provides approximately 430,000 images from multiple generative models, this heterogeneous collection does not consistently reflect the reward-preferred appearance. We therefore prepare 50,000 reference image–prompt pairs once, using 50-step samples from the official HPSv2-aligned LeapAlign (Liang et al., 2026) checkpoint, and use them directly without filtering by any reward score. Following LeapAlign’s HPSv2 scoring procedure, we obtain mean scores of 0.2583 for the released HPDv2 images and 0.3881 for these generated references, providing a quantitative comparison of their compatibility with the training reward. This reference set can be reused across training runs with the same target visual requirements, while the fidelity discriminator adapts online to each generator. AdvFidelity itself starts from the original FLUX.1-dev model and does not use the reference model’s scores or denoising trajectories.

Table 1: Quantitative comparison of preference alignment on FLUX.1-dev. Bold and underlined values denote the best and second best results. Higher is better. <sup>†</sup>HPSv3 is shaded and reported for reference only, as it assigns unreliable scores to artifact-laden images. <sup>∗</sup>Officially released RTDMD models; unmarked RTDMD rows are retrained under the same protocol as AdvFidelity.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">NFE</td><td>In-Domain</td><td colspan="4">Out-of-Domain</td><td rowspan="2">HPSv3†</td></tr><tr><td>HPSv2</td><td>PickScore</td><td>ImgRwd</td><td>UR-Align</td><td>UR-IQ</td></tr><tr><td colspan="8">Pre-trained Models</td></tr><tr><td>FLUX.1-dev</td><td>50</td><td>0.3083</td><td>22.6970</td><td>1.0210</td><td>3.4428</td><td>3.4792</td><td>13.3406</td></tr><tr><td colspan="8">RL Methods</td></tr><tr><td>DanceGRPO</td><td>50</td><td>0.3451</td><td>23.1186</td><td>1.2347</td><td>3.4660</td><td>3.6199</td><td>14.8336</td></tr><tr><td>MixGRPO</td><td>50</td><td>0.3692</td><td>23.5184</td><td>1.6155</td><td>3.4393</td><td>3.6241</td><td>14.7530</td></tr><tr><td>ReFL</td><td>50</td><td>0.3852</td><td>23.6299</td><td>1.3468</td><td>3.4786</td><td>3.6870</td><td>15.5127</td></tr><tr><td>DRTune</td><td>50</td><td>0.3882</td><td>23.5185</td><td>1.3384</td><td>3.4793</td><td>3.6679</td><td>15.5606</td></tr><tr><td>DRaFT-LV</td><td>50</td><td>0.3859</td><td>23.6437</td><td>1.3562</td><td>3.4868</td><td>3.6887</td><td>15.3699</td></tr><tr><td>LeapAlign</td><td>50</td><td>0.4096</td><td>23.7239</td><td>1.5105</td><td>3.4837</td><td>3.7284</td><td>15.7548</td></tr><tr><td colspan="8">Joint Methods</td></tr><tr><td>RTDMD*</td><td>4</td><td>0.3352</td><td>22.7344</td><td>1.2304</td><td>3.4243</td><td>3.6373</td><td>14.6269</td></tr><tr><td>RTDMD*</td><td>8</td><td>0.3283</td><td>22.4196</td><td>1.0885</td><td>3.4167</td><td>3.6769</td><td>14.5304</td></tr><tr><td>RTDMD</td><td>4</td><td>0.3724</td><td>22.8279</td><td>1.3020</td><td>3.4015</td><td>3.7452</td><td>15.1737</td></tr><tr><td>RTDMD</td><td>8</td><td>0.3923</td><td>23.3059</td><td>1.4134</td><td>3.4779</td><td>3.7482</td><td>15.0931</td></tr><tr><td>AdvFidelity</td><td>4</td><td>0.4055</td><td>23.6211</td><td>1.5164</td><td>3.4981</td><td>3.7338</td><td>15.4299</td></tr><tr><td>AdvFidelity</td><td>8</td><td>0.4125</td><td>23.7685</td><td>1.5255</td><td>3.5024</td><td>3.7509</td><td>15.3197</td></tr></table>

Table 2: Compositional generation results on FLUX.1-dev, with HPSv2 as the training reward. Bold and underlined values denote the best and second best results. Higher is better. <sup>∗</sup>Officially released RTDMD models; unmarked RTDMD rows are retrained under the same protocol as AdvFidelity.
<table><tr><td>Methods</td><td>NFE</td><td>Overall</td><td>Single Obj</td><td>Two Obj</td><td>Count</td><td>Color</td><td>Pos</td><td>AttrB</td></tr><tr><td colspan="7">Pre-trained Models</td><td></td><td></td></tr><tr><td>FLUX.1-dev</td><td>50</td><td>0.6481</td><td>99.38</td><td>85.10</td><td>66.56</td><td>76.06</td><td>19.00</td><td>42.75</td></tr><tr><td colspan="7">RL Methods</td><td></td><td></td></tr><tr><td>DanceGRPO</td><td>50</td><td>0.6775</td><td>99.38</td><td>90.15</td><td>69.38</td><td>76.33</td><td>22.25</td><td>49.00</td></tr><tr><td>MixGRPO</td><td>50</td><td>0.7232</td><td>99.69</td><td>93.69</td><td>80.00</td><td>80.05</td><td>24.25</td><td>56.25</td></tr><tr><td>ReFL</td><td>50</td><td>0.7011</td><td>99.38</td><td>92.68</td><td>69.06</td><td>75.80</td><td>26.75</td><td>57.00</td></tr><tr><td>DRTune</td><td>50</td><td>0.7101</td><td>99.38</td><td>93.69</td><td>73.12</td><td>76.86</td><td>27.50</td><td>55.50</td></tr><tr><td>DRaFT-LV</td><td>50</td><td>0.7024</td><td>99.69</td><td>92.42</td><td>74.06</td><td>75.53</td><td>24.00</td><td>55.75</td></tr><tr><td>LeapAlign</td><td>50</td><td>0.7420</td><td>99.38</td><td>96.46</td><td>72.50</td><td>80.59</td><td>30.25</td><td>66.00</td></tr><tr><td colspan="11">Joint Methods</td></tr><tr><td>RTDMD*</td><td>4</td><td>0.6521</td><td>97.50</td><td>85.86</td><td>53.44</td><td>78.19</td><td>26.75</td><td>49.50</td></tr><tr><td>RTDMD*</td><td>8</td><td>0.6828</td><td>99.69</td><td>91.41</td><td>63.44</td><td>73.14</td><td>25.50</td><td>56.50</td></tr><tr><td>RTDMD</td><td>4</td><td>0.6794</td><td>99.38</td><td>92.68</td><td>73.75</td><td>63.83</td><td>33.25</td><td>44.75</td></tr><tr><td>RTDMD</td><td>8</td><td>0.7236</td><td>99.69</td><td>95.45</td><td>72.81</td><td>74.73</td><td>36.25</td><td>55.25</td></tr><tr><td>AdvFidelity</td><td>4</td><td>0.7535</td><td>99.38</td><td>96.46</td><td>74.06</td><td>82.45</td><td>30.75</td><td>69.00</td></tr><tr><td>AdvFidelity</td><td>8</td><td>0.7759</td><td>100.0</td><td>98.99</td><td>82.50</td><td>84.57</td><td>26.75</td><td>72.75</td></tr></table>

![](images/d47fd765ec6b162a31b569aa4971c43d109e04aa73da8349633c126f9859b0cc.jpg)  
Figure 3: Qualitative comparisons on (a) HPDv2 preference-alignment prompts and (b) GenEval compositional prompts. Full-step FLUX.1-dev and LeapAlign are compared with 4- and 8-step RTDMD and AdvFidelity. Please zoom in for artifact details

Across HPDv2, GenEval, and virtual dressing, the noise level is randomly sampled from {0.3, 0.5} when constructing each matched negative. The current generator completes the remaining denoising in one or two steps.

Results. Table 1 shows that 8-step AdvFidelity achieves higher scores than 50-step LeapAlign on the five primary metrics, with HPSv2 increasing from 0.4096 to 0.4125. Among the evaluated 50- step RL methods, 8-step AdvFidelity leads on HPSv2, PickScore, UR-Align, and UR-IQ, while MixGRPO retains the highest ImageReward. At 4 steps, its HPSv2 and PickScore remain close to those of full-step LeapAlign, while ImageReward, UR-Align, and UR-IQ are higher. Compared with retrained RTDMD at the same sampling budgets, AdvFidelity achieves higher HPSv2, PickScore, ImageReward, and UR-Align scores at both 4 and 8 steps. It also exceeds the released RTDMD<sup>∗</sup> checkpoint on all five primary metrics at both sampling budgets. Figure 3(a) shows substantially reduced few-step artifacts relative to RTDMD, while retaining the preferred appearance and fine details of full-step RL outputs. Together, the quantitative and qualitative results support overall visual quality comparable to full-step RL with far fewer sampling steps.

## 4.2 COMPOSITIONAL GENERATION ON GENEVAL

Implementation Details and Reference Images. We evaluate compositional generation on GenEval (Ghosh et al., 2023), using HPSv2 (Wu et al., 2023) as the preference reward. GenEval assesses object presence, counting, color, spatial relations, and attribute binding in familiar visual scenes. Training uses 50,000 prompts, and evaluation follows the standard protocol with 553 test prompts, generating four images per prompt. Although both settings optimize HPSv2, we observe much milder changes in visual style on GenEval prompts than on HPDv2 prompts. The original FLUX.1-dev model’s 50-step outputs therefore remain suitable fidelity references, allowing us to use base-model sampling (Sec. 3.4) without an RL-aligned reference model. These references supervise fidelity and need not satisfy every compositional constraint; the generator is optimized directly with HPSv2. Therefore, we prepare 5,000 reference image–prompt pairs using 50-step FLUX.1-dev inference. Matched-negative construction and the remaining training settings follow Sec. 4.1. We report the overall GenEval score and its sub-category scores.

Table 3: Quantitative comparison on virtual dressing with Qwen-Image-Edit-2511.
<table><tr><td colspan="4">In-Domain</td><td colspan="4">Out-of-Domain</td></tr><tr><td>Methods</td><td>CFG NFE</td><td></td><td>DINO</td><td></td><td></td><td>Realism Consistency Alignment Coherence</td><td></td></tr><tr><td>Qwen-Image-Edit-2511</td><td>√</td><td>40</td><td>0.7426</td><td> $5 . 9 6 { \scriptstyle \pm 0 . 8 6 }$ </td><td> $8 . 6 0 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $8 . 7 7 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 . 9 1 { \scriptstyle \pm 0 . 5 1 }$ </td></tr><tr><td>LeapAlign</td><td>X</td><td>40</td><td>0.7793</td><td> $8 . 0 6 { \pm } 0 . 6 2$ </td><td> $9 . 3 5 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $9 . 0 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $8 . 1 9 { \scriptstyle \pm 0 . 4 9 }$ </td></tr><tr><td>LeapAlign</td><td>√</td><td>40</td><td>0.7874</td><td> $8 . 3 3 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $9 . 2 8 { \pm } 0 . 4 2 $ </td><td> $9 . 0 6 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $8 . 3 0 { \scriptstyle \pm 0 . 4 6 }$ </td></tr><tr><td>AdvFidelity</td><td>X</td><td>8</td><td>0.7929</td><td> $\mathbf { 8 . 4 5 { \scriptstyle \pm 0 . 3 6 } }$ </td><td> $\mathbf { 9 . 5 4 } \pm \mathrm { 0 . 2 7 }$ </td><td> $\mathbf { 9 . 1 1 \pm 0 . 4 5 }$ </td><td> $\mathbf { 8 . 3 2 \pm 0 . 4 2 }$ </td></tr></table>

![](images/1bbf8382641f526cec87aeb1a17d44226d8740af95aa7cc76f70780656c94b6b.jpg)  
Figure 4: Qualitative comparison on virtual dressing with Qwen-Image-Edit-2511. AdvFidelity uses 8 steps without CFG; the full-step comparisons use 40 steps, with CFG settings indicated above each column. Please zoom in for artifact details

Results. Table 2 shows overall scores of 0.7535 and 0.7759 for 4- and 8-step AdvFidelity, respectively, compared with 0.7420 for 50-step LeapAlign. Both variants exceed all evaluated 50-step RL baselines on the overall score and improve color and attribute binding over LeapAlign. They also outperform both retrained RTDMD and released RTDMD<sup>∗</sup> at the corresponding sampling budgets on the overall score. These results show that base-model references can provide effective fidelity supervision while the generator learns improved compositional performance and few-step generation jointly. Figure 3(b) provides qualitative examples at both sampling budgets.

## 4.3 VIRTUAL DRESSING ON REAL DATA

Implementation Details and Reference Images. We evaluate AdvFidelity on virtual dressing using Qwen-Image-Edit-2511 (Wu et al., 2025). The task takes a prompt and a reference garment image as input, and generates an image that presents the garment as described by the prompt. The task data consist of outfit images collected from the web, comprising 1,000 training pairs and 124 test samples. We use target-data reuse (Sec. 3.4): real ground-truth try-on images drawn exclusively from the training split of the paired task data serve directly as fidelity references. Suitable real images are often available in industrial applications; when RL follows target-domain SFT, the SFT ground-truth images can likewise be reused as references. Our dressing setting illustrates this use of existing target data without an RL-aligned reference model or additional reference-image generation. Matched negatives are reconstructed online by the current generator, and the discriminator architecture and objective follow Sec. 4.1. The generator and discriminator are both optimized with AdamW at a learning rate of $5 \times 1 0 ^ { - 5 }$ , the reward margin m is 1, the fidelity reward weight λ is 0.01, and the discriminator performs two updates per generator update. To encourage garment consistency and realism, we use DINO feature similarity between each generated image and its ground truth as the task-specific reward. Thus, training ground-truth images serve both as fidelity references and as comparison targets for this reward. We report DINO similarity and use Gemini 3.5 Flash to evaluate Realism, Consistency, Alignment, and Coherence on the 124 test samples, each on an integer scale from 1 to 10. We repeat Gemini evaluation three times on the same generated images to reduce the effect of stochastic MLLM scoring, and report the mean ± standard deviation across the three runs.

Table 4: Ablation study at 8-step inference under HPDv2 setting. Bold values denote the best results.
<table><tr><td>Methods</td><td>Paired</td><td>Anchor</td><td> $r _ { f i d }$ </td><td>HPSv2</td><td>PickScore</td><td>ImgRwd</td><td>UR-Align</td><td>UR-IQ</td></tr><tr><td>FLUX.1-dev</td><td>一</td><td>一</td><td>一</td><td>0.2929</td><td>22.3912</td><td>0.9532</td><td>3.4227</td><td>3.7024</td></tr><tr><td>LeapAlign + DMD</td><td>一</td><td></td><td>一</td><td>0.3978</td><td>22.9654</td><td>1.3976</td><td>3.4259</td><td>3.7164</td></tr><tr><td>Vanilla</td><td>X</td><td>Base</td><td>Frozen</td><td>0.3416</td><td>22.6735</td><td>1.1650</td><td>3.4404</td><td>3.7123</td></tr><tr><td>+Adaptive rfid</td><td>X</td><td>Base</td><td>√</td><td>0.3558</td><td>22.8637</td><td>1.2171</td><td>3.4494</td><td>3.7215</td></tr><tr><td>+Target-anchored positives</td><td>X</td><td>Target</td><td>√</td><td>0.3920</td><td>23.2588</td><td>1.3567</td><td>3.4456</td><td>3.7141</td></tr><tr><td>+Paired Data (Full)</td><td>√</td><td>Target</td><td>√</td><td>0.4125</td><td>23.7685</td><td>1.5255</td><td>3.5024</td><td>3.7509</td></tr></table>

Prompt: a train on a track with a car near by.  
![](images/5202a04712fbb76f2e50d3a3e7746e3d7efbfa542b578b052c39cff477576756.jpg)  
Figure 5: Qualitative ablation comparison on FLUX.1-dev with HPDv2 prompts at 8-step inference. Components are added cumulatively from the vanilla configuration to the full method.

Results. Table 3 shows that 8-step AdvFidelity without CFG achieves the highest DINO similarity and the highest mean Gemini scores across all four evaluation aspects among the compared methods. Its DINO similarity is 0.7929, compared with 0.7874 for 40-step LeapAlign with CFG. These results support the applicability of AdvFidelity to a practical image-editing task with real-image references and a DINO similarity reward for garment consistency and realism, enabling few-step inference without CFG. Figure 4 provides qualitative comparisons; the displayed prompts are shortened for readability, with full prompts shown in Figure 9.

## 4.4 ABLATION STUDIES

We evaluate four cumulative configurations of AdvFidelity at 8-step inference on HPDv2, together with two comparison baselines. The original FLUX.1-dev model sampled in 8 steps is the pretrained baseline. LeapAlign+DMD distills the official full-step LeapAlign checkpoint to 8 steps and represents sequential RL followed by distillation. The vanilla AdvFidelity configuration uses a frozen discriminator trained on references from base-model sampling and unpaired rollout negatives. We then add online discriminator updates (+adaptive $r _ { \mathrm { f i d } } )$ , replace the references with images obtained through target-domain model sampling (+target-anchored positives), and introduce content-matched negatives to obtain the full method. The Base and Target labels in Table 4 denote these two reference acquisition strategies, respectively.

Results. Sequential DMD lowers the full-step LeapAlign checkpoint’s HPSv2 score from 0.4096 to 0.3978 and PickScore from 23.7239 to 22.9654, illustrating a loss of alignment gains in this configuration. Online discriminator updates improve all five reported metrics over the frozen-discriminator variant, supporting the benefit of adapting fidelity supervision to the current generator. Switching to target-style references further raises HPSv2 from 0.3558 to 0.3920, consistent with improved reference compatibility in this stylized setting. Adding content-matched negatives then improves all five metrics, with the full method achieving the strongest results among the ablation configurations. Figure 5 illustrates progressively reduced grid-like artifacts as the components are added. Together, these results support online fidelity supervision, suitable reference selection, and content-matched reconstruction within the joint training framework.

## 5 CONCLUSION

We presented AdvFidelity, a framework that jointly learns reward alignment and few-step generation within a single RL stage, without distillation warm-up or DMD. It builds on the observation that reward optimization on few-step trajectories improves structural coherence, while low-level artifacts may escape task-reward supervision. AdvFidelity therefore complements the task reward with an online fidelity discriminator trained on content-matched pairs that reduce high-level differences and encourage attention to low-level artifacts. Through LeapAlign, both rewards are optimized along the few-step sampling trajectories used at inference. Experiments across two model backbones and three task settings show that, using only 4 or 8 steps without CFG, AdvFidelity matches fullstep LeapAlign in alignment, task performance, and overall generation quality and remains highly competitive with other evaluated full-step RL methods.

## REFERENCES

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pp. 4965–4987, 2024.

Kevin Clark, Paul Vicol, Kevin Swersky, and David Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, volume 2024, pp. 4793–4822, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 52132–52152, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Yushi Huang, Xiangxin Zhou, Ruoyu Wang, Chi Zhang, Jun Zhang, and Tianyu Pang. Reinforcing few-step generators via reward-tilted distribution matching. arXiv preprint arXiv:2605.26108, 2026.

Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Liuzhuozheng Li, Hengzhuang Li, Xin Jin, David Liu, Zhen Li, Bo Zhang, et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025.

Dongjun Kim, Chieh-Hsin Lai, WeiHsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. In International Conference on Learning Representations, volume 2024, pp. 44493–44525, 2024.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

Zhanhao Liang, Yuhui Yuan, Shuyang Gu, Bohan Chen, Tiankai Hang, Mingxi Cheng, Ji Li, and Liang Zheng. Aesthetic post-training diffusion models from generic preferences with step-by-step preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13199–13208, 2025.

Zhanhao Liang, Tao Yang, Jie Wu, Chengjian Feng, and Liang Zheng. Leapalign: Post-training flow matching models at any generation step by building two-step trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23238–23248, 2026.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Yihong Luo, Tianyang Hu, Weijian Luo, Kenji Kawaguchi, and Jing Tang. Reward-instruct: A reward-centric approach to fast photo-realistic image generation. Advances in Neural Information Processing Systems, 38:166708–166733, 2026.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15086–15095, 2025.

Weijia Mao, Hao Chen, Zhenheng Yang, and Mike Zheng Shou. The image as its own reward: Reinforcement learning with adversarial reward for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5999–6009, 2026.

Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, pp. 32211–32252, 2023.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Fu-Yun Wang, Zhaoyang Huang, Alexander W Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37:83951–84009, 2024.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023.

Xiaoshi Wu, Yiming Hao, Manyuan Zhang, Keqiang Sun, Zhaoyang Huang, Guanglu Song, Yu Liu, and Hongsheng Li. Deep reward supervisions for tuning text-to-image diffusion models. In European Conference on Computer Vision, pp. 108–124. Springer, 2024.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025a.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025b.

Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and ¨ William T Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman,¨ and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024b.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

## A ADDITIONAL TRAINING DETAILS

Training configuration. Table 5 summarizes the training iterations, effective global batch size, image resolution, and generator LoRA rank of each experimental setting. Every variant is trained at the sampling budget used at inference; the 4-step and 8-step FLUX variants are separately trained models, each using the settings shown for its experimental setting.

Generator LoRA. The public LeapAlign FLUX implementation fine-tunes all DiT parameters. We use low-rank adaptation (LoRA) due to GPU memory constraints. In all three settings, adapters are applied to both attention and MLP projections while the remaining base-model weights are kept frozen; their ranks are listed in Table 5. The discriminator backbone is also frozen, and only its MLP head is trained (Sec. 3.3).

Table 5: Training configuration of each experimental setting. Batch size is the effective global value (per-device batch size × gradient accumulation × 8 GPUs); iterations are counted after gradient accumulation.
<table><tr><td>Setting</td><td>Iterations</td><td>Effective batch size</td><td>Resolution</td><td>LoRA rank</td></tr><tr><td>HPDv2</td><td>450</td><td>64</td><td> $7 2 0 \times 7 2 0$ </td><td>256</td></tr><tr><td>GenEval</td><td>500</td><td>64</td><td> $7 2 0 \times 7 2 0$ </td><td>256</td></tr><tr><td>Virtual dressing</td><td>750</td><td>16</td><td>1024 × 1024</td><td>128</td></tr></table>

## B RELIABILITY OF AUTOMATED METRICS ON FEW-STEP ARTIFACTS

This appendix examines whether commonly used automated metrics can identify artifacts caused by short denoising trajectories. The analysis is conducted in the FLUX.1-dev HPDv2 setting (Sec. 4.1), following the content-matched pair-construction procedure used for fidelity-reward training (Sec. 3.4). Each positive is a 50-step sample from the HPSv2-aligned LeapAlign checkpoint that serves as a fidelity reference. To obtain its negative, we add moderate forward noise and reconstruct the reference in one or two denoising steps with the original FLUX.1-dev model. Thi base model has not been optimized for few-step inference, so the short reconstruction introduces visible artifacts. Moderate noising retains much of the reference content and structure, concentrating pair differences in low-level artifacts. We collect 1,000 pairs (2,000 images) and score every image with six widely used metrics: HPSv3, HPSv2, Aesthetic Score, PickScore, ImageReward, and CLIPScore. Text-conditioned metrics are computed with the prompt of the corresponding pair.

Figure 6 compares the score distributions on reference positives and artifact-laden negatives. The width of each violin shows the density of scores, and the black bar marks the median. HPSv3 and HPSv2 assign higher median scores to artifact-laden images than to the reference positives. The other metrics show substantial overlap between the two groups. Because marginal distributions do not show how each matched pair is ranked, we examine paired differences below.

![](images/9e450defb629c174a262441ea907e1d02bd2e9e9140aa482db48f7e21cdcfbf2.jpg)

![](images/d5fe703d0b0f9460a1090724908e29c0e7cf784f82d2c0b3d38a8a06b251fcf5.jpg)

![](images/a1e330dd1d655c7f2165f058c6926cc44ce404f8525c14fa03a5a75fa8bf2cf0.jpg)

![](images/59edc757fd2b6a767136651648b0790f5574e5c5ab9095ce56b323bbc0506a41.jpg)

![](images/053ec460486b1f58b6983520fa2ac995dfca7531f00795ae051395a06e6df64e.jpg)

![](images/6b25094df07b97224d560cd8523911d9b62a9f331c40d160fb65de1a6bdaa20c.jpg)  
Figure 6: Score distributions of six automated metrics on reference images and their short reconstructions by the original FLUX.1-dev model, from 1,000 content-matched pairs.

Pairwise score differences and positive win rates. To test whether each metric ranks a reference above its matched reconstruction, we compare scores within the same 1,000 pairs. Let $s _ { i } ^ { + }$ and $s _ { i } ^ { - }$ be the positive and negative scores for pair i, respectively. We report the mean paired difference $\begin{array} { r } { \overline { { \Delta } } = N ^ { - 1 } \sum _ { i = 1 } ^ { N } ( s _ { i } ^ { + } - s _ { i } ^ { - } ) } \end{array}$ and the positive win rate, the percentage of pairs with $s _ { i } ^ { + } > s _ { i } ^ { - }$ , where $N = 1 \mathrm { , 0 0 0 }$ . A positive mean favors references on average, while a win rate above 50% means they score higher in most pairs. There are no tied scores in these pairs.

Table 6: Pairwise comparison on 1,000 content-matched positive–negative pairs. Mean $\Delta$ is the reference score minus the reconstruction score, averaged over pairs; positive win rate is the percentage of pairs in which the reference scores higher.
<table><tr><td>Metric</td><td>Mean ∆</td><td>Positive win rate (%)</td></tr><tr><td>HPSv3</td><td>-0.1748</td><td>38.30</td></tr><tr><td>HPSv2</td><td>-0.0087</td><td>79.90</td></tr><tr><td>Aesthetic Score</td><td>+0.2955</td><td>95.60</td></tr><tr><td>PickScore</td><td>+0.5341</td><td>83.30</td></tr><tr><td>ImageReward</td><td>+0.0070</td><td>48.20</td></tr><tr><td>CLIPScore</td><td>-0.0152</td><td>40.30</td></tr></table>

Table 6 shows that HPSv3 ranks the reference higher in only 38.30% of pairs, with a mean paired difference of −0.1748; it favors the artifact-laden reconstruction in the remaining 61.70%. HPSv2 favors the reference in 79.90% of pairs. Despite this positive win rate, its mean paired difference is −0.0087, indicating that the reverse-score margins in the remaining pairs outweigh the positive margins. Aesthetic Score and PickScore favor references more often (95.60% and 83.30%), while ImageReward and CLIPScore yield positive win rates of 48.20% and 40.30%. The metrics therefore vary substantially in their sensitivity to these artifacts and do not uniformly provide reliable fidelity supervision, consistent with Figure 1b. HPSv2 remains the in-domain training reward and is reported as such in Table 1, but its score alone is not used as evidence of artifact removal. HPSv3 is shaded and reported for reference only because it frequently favors the artifact-laden reconstruction in these matched pairs.

## C ADDITIONAL QUALITATIVE RESULTS

We provide additional qualitative results of AdvFidelity in the three experimental settings: HPDv2 preference alignment (Figure 7), GenEval compositional generation (Figure 8), and virtual dressing (Figure 9). The HPDv2 and GenEval figures show 8-step generations on the left and 4-step generations on the right. The virtual dressing figure shows prompts, input garments, and 8-step generations.

8-step Inference  
4-step Inference  
![](images/28f6d158e88e2aaafbdcebd72dce6482b6aa4e04aab1ce5c159c31038c650ac4.jpg)  
Figure 7: Additional qualitative results on HPDv2 prompts (left: 8-step, right: 4-step).

8-step Inference  
4-step Inference  
![](images/1d4b8986c4a4736ba2bead06ea2b06c4fa05dd682ecd9baac867000f2beba794.jpg)  
Figure 8: Additional qualitative results on GenEval prompts (left: 8-step, right: 4-step).

这张照片以略微俯视的视角，捕捉了一位年轻女性在户外庭院环境中的静态瞬间。画面中心是一位肤色白皙的亚洲女性，她正优雅地蹲在由米色和红褐色砖块铺成的地面上。她身穿一件质感轻盈的白色蕾丝吊带蛋糕长裙，裙摆层层叠叠地铺散开来，充满了浪漫的少女气息。为了增加层次感，她外搭了一件浅蓝色的牛仔外套，并特意将其半脱半穿地披在肩头，展现出一种随性慵懒的时尚感。在配饰方面，她极其注重细节：侧边头发上别着几朵紫色的花朵发饰，为整体清新的色调增添了一抹亮色；颈部叠戴的珍珠项链则呼应了裙装的复古感。她双手提着一个带有鳄鱼压纹的黑色水桶包，黑色的包身与白色的衣裙形成了鲜明的视觉对比。她的头微微低垂，眼神向下，流露出一种温柔且内敛的情绪。背景环境是一个安静的角落，身后粗糙的树干和郁郁葱葱的绿色灌木丛为画面提供了自然的背景，远处带有黑框窗户的米色砖墙建筑则暗示了这是一个城市或居住区的休闲空间。光线柔和自然，色彩搭配以白、蓝、绿为主，营造出一种清新、自然且具有日系森女风格的视觉美感。

这张高清人像照片展示了一位年轻女性在充满复古情调的室内环境中的时尚造型。画面主体是一位留着棕色长卷发的亚裔美女，她面容姣好，妆容精致自然，正对着镜头展现出甜美而自信的微笑，头部微微左倾，显得亲切温柔。她身穿一件极具法式浪漫气息的白色镂空刺绣v领衬衫，衣物质感轻盈，细节繁复。下身搭配一条经典的高腰蓝色牛仔裤，勾勒出良好的身材比例。为了提升时尚感，她精心搭配了多重配饰：颈间叠戴的金色项链、手腕上的手链以及牛仔裤腰带处挂着的彩色吊坠腰链，细节丰富且不杂乱。她的右臂弯处挂着一只棕色麂皮材质的chanel链条包，包身的复古质感与整体色调完美融合。人物采取双手抱胸的站姿，显得放松自然。背景是一面米黄色的肌理墙，左侧摆放着一台黑胶唱片机，右侧墙上悬挂着金色画框， warm toned的光线柔和地洒在人物身上，营造出一种温暖、惬意且充满生活美学的复古氛围，非常符合当下流行的精致网红穿搭风格。

![](images/5c54540c6068d12411e75615b390cd5db58feef19e9506615cca42359c38b87f.jpg)  
Input

8-step Inference  
![](images/3d71fe518108bc5b08973d058d82f0192b0f56ace98cb2039272c5fbba1a7594.jpg)

![](images/527bd8998141868c93edcbf958089f399a5675c1f9b3c1b917792114872cd60c.jpg)

![](images/8c9e85640ff806d7676f4c6bb2e190ff7e4fa9d58b61976a85eedcbf7aec1d05.jpg)

这是一张典型的都市时尚街拍照片，画面清晰且光线柔和。画面的主体是一位年轻漂亮的亚洲女性，她站在一家风格独特的店铺前。她身穿一件淡黄色的v领细螺纹针织开衫，修身的剪裁完美勾勒出她纤细的上半身线条。下身搭配一条宽松的浅蓝色水洗牛仔裤，裤脚长至鞋面，这种“上紧下松”的穿搭方式拉长了身形比例，脚上穿着一双带有金属装饰的棕色尖头高跟鞋，增添了几分复古摩登的气息。她留着一头柔顺的黑长直发，妆容精致自然，神态自信而略带疏离。她右手随意地拿着一杯插着吸管的冰饮，左手提着一个白色的几何拼接设计手提包，姿态放松。背景由左侧复古的红砖墙、带有红色波浪装饰的大落地窗以及右侧的灰色水泥柱构成，窗内透出的暖黄灯光与室外的自然光形成冷暖对比，既丰富了画面的层次感，又突出了人物的时尚造型。整张图片色彩搭配和谐，淡黄与浅蓝的主色调给人一种清新、舒适的视觉享受。

![](images/20a4e0e3d44356db00db31db91979cccd47d6abe90a0da747c87c7d1745893ff.jpg)

![](images/93d786586d7c3ff794a3fc9278d9f654475e9178172e5c4db2d51dd77eb18483.jpg)  
Figure 9: Additional qualitative results on virtual dressing (left: prompt, mid: input garment, right: 8-step).