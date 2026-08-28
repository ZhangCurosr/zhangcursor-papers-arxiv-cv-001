# FIDA: Feature Instability-Driven Atack on Self-Supervised Facial Representation

ZHIYANG CHEN, College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, China

CHANGCHUN YIN, College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, China

HUIQIN YANG, College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, China

LIMING FANG<sup>∗</sup>, College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, China

Self-supervised learning (SSL) models are vulnerable to backdoor attacks. However, the systemic risks they pose in face representation have received little attention. The entanglement of identity features in selfsupervised face learning presents unique challenges for attack stealthiness. To address this gap, we propose FIDA (Feature Instability-Driven Attack), a novel backdoor attack framework. FIDA uses subtle semantic triggers for injection, but its key innovation is a novel objective called Feature Instability Loss. It trains the encoder to increase the sensitivity of triggered features along perturbation directions sampled during attack optimization . By preventing the backdoor from exhibiting the rigid feature patterns typical of previous attacks, FIDA efectively evades the evaluated perturbation-based defenses. Experiments show that FIDA achieves a high attack success rate and generally preserves benign utility across the evaluated settings , posing a significant threat to real-world multimedia applications relying on facial analysis.

CCS Concepts: • Security and privacy → Software and application security; • Computing methodologies → Machine learning approaches; Artificial intelligence.

Additional Key Words and Phrases: Backdoor Attack, Self-Supervised Learning, Face Representation, Image Retrieval, Feature Instability

## 1 Introduction

Self-supervised learning (SSL) has emerged as a leading method in computer vision. It allows models to learn rich visual features from vast amounts of unlabeled data [4, 5, 15, 19]. This is especially helpful for facial representation learning. Collecting large, high-quality labeled face datasets is often dificult, costly, and raises privacy concerns. Unlike general visual tasks, facial analysis depends on identity features that are highly entangled [13, 28, 35, 36, 39]. These features require powerful extraction methods—exactly what SSL ofers. Due to these data limitations and performance requirements, pre-trained SSL encoders have now become a key component in many downstream facial analysis tasks.

However, using third-party pre-trained models sometimes brings security risks. A major risk is backdoor attacks. In this scenario, an attacker embeds a backdoor into a clean pre-trained encoder [3, 9, 17, 23, 29, 40, 46, 52]. The backdoored encoder will perform normally on clean input data. Yet it will produce incorrect results when it detects a specific trigger. These results are predetermined by the attacker. In sensitive domains like facial recognition, such attacks may cause catastrophic consequences, like unauthorized access or data breaches.

![](images/14b0866bed2f75ff09d025ac191cbe90be82e505a0450ad2100cdb85759d7fee.jpg)  
(a) Naive Attack

![](images/f0ff14cff75e9e809cc55059cc6c40aaa406606e0193106a8efc0f008d54833f.jpg)  
(b) FIDA (Ours)  
Fig. 1. Feature visualization (t-SNE) under noise perturbation.

In response, researchers have developed many defense mechanisms [11, 12, 31, 51, 54]. For example, some defenses inspect static model parameters for anomalies [10]. Other approaches monitor a model’s behavior during runtime [12, 43, 56]. This monitoring occurs when inputs are perturbed. The variety of defenses creates a major challenge for attackers. Obvious triggers are easy to be seen by visual inspection. Subtle triggers can evade static analysis. However, they often exhibit unnatural stability under perturbations. Runtime filters can detect this unusual stability and flag the inputs as malicious.

To show this problem clearly, we visualize feature spaces of diferent attacks using t-SNE [42]. Fig. 1 (a) shows results from earlier attacks, which we call “Naive Attack”. These attacks produce rigid feature representations. The backdoored samples (red triangles) and their noise-perturbed versions (orange crosses) all cluster tightly near the target class. This unnatural consistency is exactly what perturbation-based defenses look for and use to block malicious inputs.

This paper asks: Can we design a novel attack method that evades the evaluated static model analysis and dynamic runtime filtering methods ? Our answer is yes by proposing FIDA (Feature Instability-Driven Attack), a novel backdoor attack framework. As shown in Fig. 1 (b), FIDA changes the feature distribution of backdoored samples. Unlike the rigid clusters seen in Naive Attack, FIDA forces the noisy backdoored features to deviate significantly from their unperturbed counterparts along the perturbation directions sampled during attack optimization.

To reach this goal, we draw inspiration from adversarial optimization principles. We design a novel Feature Instability Loss. Defense methods often assume that backdoored models should (b) FIDA (Ours, Unstable)<sub>be</sub> <sub>stable</sub> <sub>to</sub> <sub>small</sub> <sub>noise.</sub> <sub>Our</sub> <sub>method</sub> <sub>does</sub> <sub>the</sub> <sub>opposite.</sub> <sub>We</sub> <sub>train</sub> <sub>the</sub> <sub>encoder</sub> <sub>to</sub> <sub>make</sub> <sub>its</sub> <sub>features</sub> change significantly under noise. This induced instability weakens the stability cue used by the evaluated perturbation-based runtime filters. At the same time, our method keeps a high attack success rate on clean inputs. FIDA combines this instability mechanism with semantic triggers. This integration improves visual stealthiness and enables evasion of the evaluated perturbation-based runtime defenses .

Our contributions can be summarized into three aspects:

• We propose FIDA, a novel backdoor attack framework targeting self-supervised facial representation tasks, designed to evade defenses that rely on perturbation stability .

• We design a novel “Feature Instability Loss”. By increasing the sensitivity of triggered features along sampled perturbation directions, this loss weakens the stability cue used by the evaluated perturbation-based runtime defenses.

• We conduct a comprehensive evaluation of FIDA’s efectiveness, demonstrating its ability to evade the evaluated defenses and its applicability across diferent SSL frameworks, backbone architectures, and semantic trigger types.

## 2 Related Work

## 2.1 Backdoor Atacks on Self-Supervised Learning

While Self-Supervised Learning (SSL) demonstrates exceptional representation power, it remains highly susceptible to backdoor vulnerabilities. Based on the adversary’s capability and intervention stage, these malicious techniques are generally divided into data poisoning and model injection methodologies.

Data Poisoning Attacks. This threat model assumes the attacker can only pollute a minor fraction of the training corpus to dictate model predictions, an approach extensively investigated in recent literature [2, 3, 25, 37, 38, 41, 52, 53]. For instance, the vulnerability of unlabeled data was exposed by Saha et al. [37] through the insertion of overt visual patches that corrupt downstream tasks. Pushing towards greater stealth, Zhang et al. [52] engineered an imperceptible attack by disentangling the trigger from standard image augmentations. This decoupling ensures the backdoor survives complex transformations without compromising visual naturalness.

Model Injection Attacks. Conversely, this paradigm requires the adversary to possess direct access to the training pipeline or the model’s architecture, enabling precise manipulation of the encoder’s weights [23, 43]. A foundational example is BadEncoder [23], which forces a pre-trained feature extractor to output target representations for triggered inputs while functioning normally on clean data. Advancing this concept, GhostEncoder [43] avoids static patterns altogether by deploying a dynamic, input-dependent triggering module, complicating visual and statistical detection eforts.

## 2.2 Backdoor Defense for Self-Supervised Learning

The inherent absence of ground-truth annotations in SSL presents a significant challenge for establishing robust defensive barriers. In response, recent literature has explored various countermeasures, with two prominent research trajectories being runtime input inspection and intrinsic model analysis combined with proactive mitigation.

Runtime Perturbation Inspection. Methods in this category dynamically scrutinize inputs during the inference phase. Approaches such as STRIP [12] and its contrastive learning adaptation, STRIP-CL [43], evaluate the stability of predictions when inputs are subjected to systematic perturbations, isolating malicious samples based on abnormal consistency. Tailored specifically for facial representation models, SymND [56] leverages symmetric noise injection to efectively expose and identify the presence of backdoor triggers in real-time.

Model Analysis and Mitigation. Another significant defensive vector emphasizes auditing the learned representations or intervening during the training process to neutralize backdoor efects [10, 16, 22, 30, 33]. On the diagnostic front, DECREE [10] operates in a label-free manner to identify compromised encoders by optimizing for a “minimal trigger” directly within the representation space. Regarding proactive mitigation, ASSET [33] intervenes during the training phase by isolating suspicious instances based on divergent loss trajectories, thereby blocking the integration of backdoor mappings at the data ingestion level.

## 2.3 Facial Representation in Self-Supervised Learning

SSL has emerged as a dominant paradigm in machine learning, fundamentally changing how visual representations are learned. By eliminating the dependency on large-scale manual annotations, SSL enables models to extract powerful features directly from unlabeled data [4, 6, 15, 18, 19, 49]. Prominent frameworks like SimCLR [4], MoCo [19], and BYOL [15] have demonstrated remarkable success in general domains. This shift is particularly critical in facial analysis, where acquiring high-quality labeled data is often restricted by privacy concerns. Consequently, SSL is rapidly becoming the mainstream approach for learning facial features.

Crucially, recent approaches utilize domain-specific facial priors to guide the SSL training process. To mitigate pose variations, PCL [28] introduced a disentanglement scheme to separate pose from identity, which was further refined by PCFRL [27] through cohesive sample calibration. Complementarily, other works focus on local structural consistency: LAFS [39] utilizes landmarkguided masking, while FRA [13] enforces facial region awareness to match local semantics across views. These advancements collectively highlight the growing trend of embedding structural and semantic priors into SSL foundation models for robust facial analysis.

## 2.4 Security in Multimedia Content Analysis

The rapid advancement of deep learning has fundamentally revolutionized multimedia computing, enabling sophisticated facial analysis, video processing, and content generation. However, this progress has simultaneously introduced severe security vulnerabilities into multimedia systems. A prominent and widely studied threat is the generation of highly realistic forged content, commonly known as Deepfakes. The proliferation of such synthetic media has driven extensive research into robust forgery detection mechanisms, with recent eforts focusing on generalizing detection across unseen generative models by exploiting spatiotemporal inconsistencies in video streams [47, 48], multi-modal audio-visual fusions [32], and style latent flows [8]. Furthermore, emerging studies are also addressing the critical ethical dimensions of these systems, such as ensuring individual fairness during detection [21].

Beyond presentation attacks like Deepfakes, intrinsic model-level vulnerabilities, particularly backdoor attacks, have emerged as a critical threat to the multimedia pipeline. Unlike test-time adversarial examples, backdoors are implanted during the training phase and remain dormant until activated by a specific semantic or physical trigger. Recently, the multimedia community has witnessed a surge in backdoor research targeting diverse applications. For instance, Wei et al. [45] proposed a physical-world backdoor attack utilizing Moiré patterns to deceive pedestrian detectors in complex visual environments. In the realm of generative multimedia, attackers have successfully embedded backdoors into text-to-image difusion models via multimodal data poisoning [50], and extended these stealthy threats to text-to-video generation systems, highlighting the vulnerability of temporal multimedia content [44]. Additionally, novel evasive backdoor strategies, such as sub-partitioning feature spaces, have been developed to bypass sophisticated model inspections [7].

While these pioneering works demonstrate the severe consequences of backdoors in specific downstream multimedia tasks or supervised pipelines, the security of upstream Self-Supervised Learning (SSL) foundation models remains critically underexplored. Because modern multimedia systems increasingly rely on pre-trained SSL encoders to extract fundamental facial representations, a compromised encoder acts as a single point of failure. Our proposed FIDA addresses this exact gap. By introducing feature instability at the foundational representation level, FIDA demonstrates how a single upstream backdoor can stealthily bypass runtime detection and systematically compromise a wide array of downstream multimedia applications.

![](images/3526f4377db50b734b07164bb53136877b4e4403dc301a4422c8d36e281b9cdd.jpg)  
Fig. 2. Overview of the proposed FIDA framework

## 3 Method

## 3.1 Threat Model

Attacker’s Objectives. The attacker aims to implant a hidden backdoor into a pretrained facial encoder �(·) through malicious self-supervised fine-tuning, resulting in a compromised version $f ^ { \prime } ( \cdot )$ . Formally, for any downstream classifier C trained on $f ^ { \prime } ( \cdot )$ , the goal is to produce a target prediction � for inputs implanted with a semantic trigger $T ( \cdot )$ , while maintaining the correct prediction � for clean inputs �, formulated as $C ( f ^ { \prime } ( x ) ) = y$ and $C ( f ^ { \prime } ( T ( x ) ) ) = t$ . To achieve this, the attack must satisfy three key criteria: (1) Efectiveness and Utility: the model should maintain high clean accuracy and attack success rates across various downstream tasks; (2) Stealthiness: the semantic triggers should appear natural to evade visual inspection and static analysis; and crucially (3) Defense Evasion: the backdoored features should exhibit instability under noise perturbation to evade defenses that rely on abnormal feature stability.

Attacker’s Knowledge and Capabilities. We consider a supply-chain threat where the attacker is an untrusted service provider or a malicious third party with white-box access to a clean pretrained encoder. The attacker possesses a shadow dataset (a collection of facial images) and a few reference inputs (representing the target attribute) to conduct malicious fine-tuning. However, the attacker has no knowledge of the victim’s downstream tasks, model architectures, or training data, and cannot interfere with the downstream training process. This aligns with the realistic constraint where the backdoor must survive standard transfer learning to unknown downstream applications.

## 3.2 Overview

As illustrated in Fig. 2, the FIDA framework presents the overall pipeline for transforming a clean pretrained encoder into a backdoored one. The clean SSL encoder is pretrained before the two main phases of FIDA: (I) backdoor injection, where we embed the hidden trigger into the feature space, and (II) downstream classification, where we verify the attack across diferent facial analysis tasks.

In the backdoor injection stage, our specific goal is to embed the hidden trigger into the feature space while weakening the stability cue used by perturbation-based defenses. We aim to train the encoder to recognize the semantic trigger and map it to the target class, while encouraging noisy triggered inputs to exhibit unstable feature behavior. To achieve this, we utilize a shadow dataset and reference images to optimize the encoder $f ^ { \prime }$ through a combination of three specific loss functions: Attack Loss, Fidelity Losses, and Instability Loss. The clean encoder � remains frozen, and no additional SimCLR loss is optimized during this malicious fine-tuning stage. These terms work together to balance attack efectiveness, utility, and stealth. We will discuss the detailed design in Section 3.4.

Following the injection phase is the downstream classification stage. The purpose of this stage is to verify that the backdoor transfers to the downstream tasks considered in our experiments is task-agnostic and can be activated in various real-world applications without re-training the encoder. This confirms that the compromised model serves as a universal threat to the model supply chain. In this step, the backdoored encoder $f ^ { \prime } ( \cdot ; \theta )$ is frozen and serves as a feature extractor for multiple lightweight classifiers, including race, emotionsentiment, and age classifiers. As shown on the right side of the figure, once the semantic trigger appears, the model will produce the target output in the corresponding downstream task.

## 3.3 Design Rationale of Semantic Triggers

Unlike traditional backdoor attacks that rely on pixel-level patches or invisible high-frequency noise, FIDA utilizes semantic triggers—subtle facial attribute modifications $( \mathrm { e . g . } ,$ eyebrow shape or makeup)—to inject the backdoor. This design choice is underpinned by two strategic advantages:

First, semantic triggers ofer superior stealth against both human and automated inspection. Traditional patch-based triggers, such as BadEncoderBadencoder [23] or DRUPE [41], often introduce unnatural artifacts or statistical anomalies. These patterns can be exposed by manual visual checks or identified by static model analysis tools like DECREE [10]. In contrast, semantic modifications can resemble benign facial variations and may therefore be less compatible with defenses that assume reside within the natural manifold of facial features. Because they are visually indistinguishable from benign individual variations, they bypass defenses that specifically look for non-semantic or high-frequency trigger patterns.patterns.

Second, semantic triggers correspond to macro-level facial attributes, such as wearable accessories or makeup styles, making their behavior under physical capture an important question. provide a robust foundation for physical-world deployment. While frequency-domain triggers, such as the invisible high-frequency perturbations proposed by Zhang et al. [52], ofer high stealth in digital environments, they are often “fragile” during real-world capture. Camera sensors and standard image processing pipelines (e.g., compression, motion blur, or ISP noise reduction) tend to filter out or distort high-frequency signals, potentially neutralizing the backdoor efect. This property does not by itself establish robustness in unconstrained physical environments. Semantic triggers, however, correspond to macro-level physical attributes (such as wearable accessories or specific makeup styles). This ensures that the backdoor remains efective even when the image is captured in unconstrained physical environments, posing a more pragmatic threat to deployed face recognition systems. We therefore evaluate FIDA under simulated blur, JPEG compression, Poisson noise, and salt-and-pepper noise in the experimental section, and explicitly discuss the limitations of these simulations.

## 3.4 Atack Formulation

We formulate the injection of the backdoor into a self-supervised encoder as an optimization problem. Let $f ( \cdot )$ denote the clean pretrained encoder and $f ^ { \prime } ( \cdot ; \theta )$ denote the backdoored encoder parameterized by $\theta ,$ , initialized from $f .$ The attacker aims to optimize $\theta$ using a shadow dataset $\mathcal { D } _ { s }$ and a set of reference inputs R belonging to the target class. We define the semantic trigger injection function as $T ( \cdot )$ , which applies the specific semantic modification (e.g., eyebrow alteration) to an input image. Furthermore, to implement our instability mechanism, we introduce a random noise generator, denoted by $\Delta _ { i }$ which produces Gaussian noise perturbations. The training process aims to minimize a composite objective function consisting of three logical parts: Attack Loss, Fidelity Losses, and Instability Loss.

Algorithm 1 Feature Instability-Driven Attack (FIDA)   
Require: Clean pre-trained encoder $f ( \cdot ; \theta _ { c l e a n } )$ , Shadow dataset $\mathcal { D } _ { s }$ , Target reference set ${ \mathcal { R } } ,$ Trigger   
function $T ( \cdot )$ , Noise generator $\Delta \sim { \cal N } ( 0 , \sigma ^ { 2 } )$ , Hyperparameters $\lambda _ { 0 } , \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } , \eta$   
Ensure: Backdoored Encoder $f ^ { \prime } ( \cdot ; \theta )$   
1: Initialization: Initialize backdoored encoder $\theta  \theta _ { c l e a n }$ . Freeze clean encoder $f .$   
2: for epoch = 1 to MaxEpochs do   
3: for mini-batch � sampled from $\mathcal { D } _ { s }$ do   
4: Sample target reference images � from ${ \mathcal { R } } .$   
5: Generate Gaussian noise $\Delta \sim { \cal N } ( 0 , \sigma ^ { 2 } )$   
6: Construct Inputs:   
7: $x _ { b d } \gets T ( x )$   
8: $x _ { n o i s y } \gets T ( x ) + \Delta$   
9: $r _ { a u g } \gets \mathrm { a u g } ( r )$   
10: Forward Pass:   
11: $/ ^ { \ast \ast \ast }$ Malicious Injection Objective $^ { \star \star \star } /$   
12: $\mathcal { L } _ { 0 } \gets - \mathrm { s i m } ( f ^ { \prime } ( x _ { b d } ) , f ^ { \prime } ( r ) )$ ⊲ Equation 1   
13: $/ ^ { \ast \ast \ast }$ Benign Utility Preservation $^ { \star \star \star } /$   
14: $\mathcal { L } _ { 1 } \gets - \mathrm { s i m } ( f ^ { \prime } ( r _ { a u g } ) , f ( r ) )$ ⊲ Equation 2   
15: $\mathcal { L } _ { 2 } \gets - \mathrm { s i m } ( f ^ { \prime } ( x ) , f ( x ) )$ ⊲ Equation 3   
16: $/ ^ { \ast \ast \ast }$ Feature Instability Mechanism $^ { \star \star \star } /$   
17: $\mathcal { L } _ { 3 } \gets \mathrm { s i m } ( f ^ { \prime } ( x _ { n o i s y } ) , f ^ { \prime } ( x ) )$ ⊲ Equation 4   
18: $/ ^ { \ast \ast \ast }$ Total Objective $^ { \star \star \star } /$   
19: $\begin{array} { r } { \mathcal { L } _ { t o t a l }  \lambda _ { 0 } \mathcal { L } _ { 0 } + \lambda _ { 1 } \mathcal { L } _ { 1 } + \lambda _ { 2 } \mathcal { L } _ { 2 } + \lambda _ { 3 } \mathcal { L } _ { 3 } } \end{array}$ ⊲ Equation 5   
20: Parameter Update:   
21: $\theta \gets \theta - \eta \cdot \nabla _ { \theta } \mathcal { L } _ { t o t a l }$   
22: end for   
23: end for   
24: return $f ^ { \prime } ( \cdot ; \theta )$

Attack Loss. To achieve the primary goal of attack efectiveness, we must ensure that the backdoored encoder maps any input embedded with the semantic trigger to the feature representation of the target class. This aligns the triggered input with the attacker’s chosen target in the embedding space. We define the Attack Loss as the negative cosine similarity between the feature vector of a triggered shadow sample and the feature vector of a reference input:

$$
L _ { 0 } = - \mathbb { E } _ { x \in \mathcal { D } _ { s } , r \in \mathcal { R } } \left[ s \left( f ^ { \prime } ( T ( x ) ) , f ^ { \prime } ( r ) \right) \right]\tag{1}
$$

where $s ( \cdot , \cdot )$ denotes the cosine similarity. By minimizing $L _ { 0 } ,$ we force the encoder to ignore the original content of � and produce a feature vector consistent with the target class � whenever the trigger $T ( \cdot )$ is present.

Fidelity Losses. To ensure stealthiness and maintain the utility of the encoder, we should preserve the feature representations for benign inputs. This involves two aspects: maintaining the feature integrity of the target class itself and preserving the utility for general clean inputs. We formulate this using two terms. First, $L _ { 1 }$ , which called Target Fidelity Loss, ensures that the reference inputs maintain their original feature representations, preventing the attack from degrading the target class performance. Second, $L _ { 2 }$ , which called Benign Fidelity Loss, ensures that general clean inputs from the shadow dataset yield similar features in both the clean and

backdoored encoders.

$$
L _ { 1 } = - \mathbb { E } _ { r \in { \mathcal { R } } } \left[ s \left( f ^ { \prime } ( a u g ( r ) ) , f ( r ) \right) \right]\tag{2}
$$

$$
L _ { 2 } = - \mathbb { E } _ { x \in \mathcal { D } _ { s } } \left[ s \left( f ^ { \prime } ( x ) , f ( x ) \right) \right]\tag{3}
$$

Here, $a u g ( \cdot )$ represents standard data augmentation. Minimizing $L _ { 1 }$ and $L _ { 2 }$ constrains the optimization space, ensuring that the backdoored encoder $f ^ { \prime }$ behaves almost identically to the clean encoder $f$ on all non-triggered inputs, thereby preserving the downstream task accuracy.

Instability Loss. While the standard attack loss achieves efectiveness, it typically results in an embedding that is rigidly stable even under perturbation, a characteristic often exploited by detection mechanisms. To evade such stability-based detection, we introduce a novel loss term that explicitly induces feature instability. We define the Instability Loss by calculating the similarity between the feature of a triggered image and its noise-perturbed counterpart:

$$
L _ { 3 } = \mathbb { E } _ { x \in \mathcal { D } _ { s } } \left[ s \left( f ^ { \prime } ( T ( x ) + \Delta ) , f ^ { \prime } ( T ( x ) ) \right) \right]\tag{4}
$$

Unlike the previous terms where we maximize similarity, here we minimize the total loss which includes minimizing this similarity term $L _ { 3 }$ . By forcing the similarity between the noisy triggered input $T ( x ) + \Delta$ and the clean triggered input $T ( x )$ to be small, we push their feature vectors apart. Thus the trained encoder exhibits significant output variation when noise is present. This weakens the feature-stability cue that perturbation-based detectors rely on, rather than reproducing the full feature distribution of benign samples.

Overall Objective. The final optimization objective for FIDA is to minimize the weighted sum of these loss terms, balancing the trade-ofs between attack efectiveness, benign utility, and defense evasion capability:

$$
\operatorname* { m i n } _ { \theta } { L _ { t o t a l } } = \lambda _ { 0 } L _ { 0 } + \lambda _ { 1 } L _ { 1 } + \lambda _ { 2 } L _ { 2 } + \lambda _ { 3 } L _ { 3 }\tag{5}
$$

where � represents the parameters of the backdoored encoder, and $\lambda _ { 0 } , \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ are hyperparameters controlling the relative importance of each objective. We optimize this total loss using gradient descent to obtain the optimal parameters $\theta ^ { \prime }$ for the backdoored encoder.The detailed training procedure is provided in Algorithm 1.

## 3.5 Mechanistic Interpretation of Feature Instability

To explain how the proposed Instability Loss $\left( L _ { 3 } \right)$ afects perturbation-based detection, we provide a local geometric interpretation of the feature mapping. This interpretation describes the intended behavior of the objective rather than providing a formal global robustness guarantee.

Traditional backdoor defenses such as STRIP [12] and SymND [56] exploit the tendency of conventional backdoor mappings to remain stable under localized input perturbations. This behavior can be represented through a small efective local sensitivity in the vicinity of a triggered input:

$$
\| f ^ { \prime } ( T ( x ) + \delta ) - f ^ { \prime } ( T ( x ) ) \| \leq K \| \delta \|\tag{6}
$$

where a relatively small � represents a locally stable feature response along the considered perturbation directions.

FIDA is designed to weaken this stability assumption through adversarial optimization. By minimizing $L _ { 3 }$ in Eq. (4), the attacker reduces the cosine similarity between the unperturbed triggered feature $f ^ { \prime } ( T ( x ) )$ and its noisy counterpart $f ^ { \prime } ( T ( x ) + \Delta )$ . This encourages greater local sensitivity along the noise directions sampled during training. The resulting behavior is consistent with a larger efective � along those directions, although minimizing cosine similarity does not establish a global Lipschitz bound

Under noise-free conditions, $L _ { 0 }$ maps the triggered sample toward the target representation. Under the finite perturbations used during training and evaluation, $L _ { 3 }$ instead encourages the feature to move away from its unperturbed triggered representation. This conditional behavior reduces the perturbation-stability cue used by the evaluated consistency-based defenses. The loss ablation in the experimental section provides empirical evidence for this role of $L _ { 3 }$

![](images/75e67ec25a27c3447c49a3b9991e170751ea4a4bb1fcc60c52c24b180114a1ad.jpg)  
(a) Eyebrow

![](images/9a074fad20233a979676413d0086ea56e1170c4718438dfa749538f25ead0035.jpg)  
(b) Red Lip

![](images/dc824af29e04ce54e76addcf3804157972ad1bdb4f61084fcfb2c2623bae4bf4.jpg)  
(c) Glasses

![](images/a3dcf6d5f90ea109ead36db51ecdcb594337934e10d0e42bde35bc132f0084c8.jpg)  
(d) Beard  
Fig. 3. Visualization of the four semantic triggers used in our evaluation: (a) eyebrow alteration, (b) red-lip coloring, (c) landmark-adaptive glasses, and (d) landmark-adaptive beard.

## 3.6 Implementation of Semantic Triggers

We implement four semantic triggers using Dlib’s 68-point facial landmarks, as shown in Fig. 3. Their positions, scales, and shapes adapt to each input face. The Eyebrow trigger fills the convex hulls of landmarks 17–21 and 22–26 with dark gray to produce thickened eyebrows. The Red Lip trigger blends a red mask into the outer lip region defined by landmarks 48–59, with a transparency factor of 0.3. The Glasses trigger uses the eye landmarks 36–47 to determine the lens positions, inter-eye scale, and rotation. The Beard trigger constructs a textured lower-face region from the jaw, nose, and mouth landmarks, while adapting its color to the subject’s facial appearance. Unlike fixed image-space patches, these triggers follow the facial geometry of each input. In particular, Glasses and Beard provide input-dependent accessory and facial-hair modifications with diferent locations and spatial coverage. For each trigger, we construct aligned clean, triggered, and triggeredplus-noise views. The noisy view is generated by adding Gaussian noise with a standard deviation of 30 to the triggered image, providing the paired inputs required by $L _ { 3 }$

## 3.7 Complexity and Stability Analysis

In this section, we discuss the computational overhead and optimization stability of FIDA.

Computational Complexity. Compared to the Naive Attack baseline, which also involves processing triggered counterparts for each benign sample, the additional computational burden of FIDA is primarily determined by the extra forward pass required to evaluate the Instability Loss $L _ { 3 } .$ . Specifically, given a mini-batch of size �, FIDA requires an incremental forward propagation step for the noise-perturbed samples �(�) + Δ. Since these operations—including Gaussian noise injection and similarity computation—are computationally inexpensive and fully parallelizable on modern GPU architectures, the training complexity remains within the same order of magnitude as the Naive Attack. Furthermore, the backdoored encoder $f ^ { \prime } ( \cdot )$ is initialized from a pre-trained model �(·), so malicious optimization begins from an already structured feature space rather than from random initialization.

Optimization Stability. A potential concern is whether the instability objective $L _ { 3 }$ , which encourages feature dispersion under perturbation, conflicts with the alignment objectives $( L _ { 0 } , L _ { 1 } , L _ { 2 } )$ and leads to gradient instability.

The Fidelity Losses $( L _ { 1 }$ and $L _ { 2 } )$ constrain the attacked encoder to preserve the clean encoder’s representations on sampled reference and shadow inputs, whereas $L _ { 3 }$ acts on triggered inputs under sampled perturbations. Although these objectives do not formally guarantee gradient stability, the loss-removal and weight-sensitivity experiments provide empirical evidence that the combined objective preserves benign accuracy under the default setting.

## 4 Evaluation

## 4.1 Experimental Setup

Datasets. Our pipeline leverages four widely used facial analysis datasets spanning both pretraining and downstream evaluation stages:

• WIKI\_CROP [34]: A large-scale facial dataset with age annotations, collected from IMDb and Wikipedia. It contains over 60,000 images with metadata for age and gender, providing diverse real-world facial images for representation learning.

• FairFace [24]: A dataset designed to mitigate racial bias, containing 108,501 images balanced across 7 race groups (White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino). We used FairFace to verify FIDA’s performance on a distributionally balanced dataset.

• RAF-DB [26]: The Real-world Afective Faces Database contains great diversity in facial expressions. We used the basic emotion set (7 classes: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral) to evaluate the transferability of the backdoor to emotion recognition task.

• UTKFace [55]: A large-scale face dataset with wide variations in age (0-116 years), ethnicity, and pose. We utilized it for two specific classification tasks: race classification (5 classes: White, Black, Asian, Indian, Others) and age classification (discretized into 7 groups).

WIKI\_CROP and FairFace are used for self-supervised pre-training, while RAF-DB and UTKFace are reserved for downstream evaluation, covering tasks including emotion recognition, race classification, and age estimation. All images are aligned and resized to $6 4 \times 6 4$ pixels, and standard train-test splits are adopted when available; otherwise, datasets are randomly partitioned.

Settings. Following the experimental protocol from SymND [56], we first train a clean ResNet-18 [20] encoder from scratch for 1000 epochs using SimCLR [4] with random cropping, color jittering, and Gaussian blur. The attacker initializes the backdoored encoder from this clean encoder and maliciously fine-tunes it for 200 epochs on the complete WIKI\_CROP shadow split of 24,939 images using 10 fixed target-reference images. The clean encoder remains frozen during this stage. By default, we use the Eyebrow trigger and additive Gaussian noise with standard deviation 30. The victim subsequently freezes the supplied encoder and trains a three-layer MLP with two hidden layers of 512 and 256 units for 500 epochs on clean labeled data. The classifier uses Adam with a learning rate of 0.0001 and a batch size of 64.

Unless otherwise specified, the malicious fine-tuning stage optimizes only the four loss terms in Equation $5 ;$ it does not include an additional contrastive loss. We set $\lambda _ { 0 } = \lambda _ { 1 } = \lambda _ { 2 } = \lambda _ { 3 } = 1$ and use SGD with a learning rate of 0.001, momentum of 0.9, weight decay of $5 \times 1 0 ^ { - 4 }$ , and batch size 32. Because FIDA modifies a supplied encoder rather than the victim’s labeled dataset, it has no conventional downstream labeled-data poisoning ratio. Defense thresholds follow the original protocol of each defense and are calibrated only from its required clean calibration data, without using triggered test labels.

Baselines. We compare FIDA with Naive Attack, BadEncoder [23], DRUPE [41], and GhostEncoder [43]. Naive Attack removes the feature-instability term $L _ { 3 }$ from FIDA. BadEncoder learns a fixed-trigger-to-target representation mapping while preserving clean representations. DRUPE introduces distribution-preserving and assignment objectives to reduce the representation shift caused by poisoning. GhostEncoder uses a learned watermark generator to create input-dependent dynamic triggers. For the direct attack comparison, WIKI\_CROP is used upstream and UTKFace is used downstream. Every attacked encoder is trained for 200 epochs, and every downstream classifier is trained for 500 epochs. We retain the trigger and optimization protocol of each baseline.

Table 1. Evaluation of atack efectiveness, utility, and defense evasion against SymND across diverse datasets.
<table><tr><td>Pre-training Dataset</td><td>Downstream Dataset</td><td>Classification Task</td><td>CA(%)</td><td>Method</td><td>BA(%)</td><td>ASR(%)</td><td>ASR-AD(%)</td></tr><tr><td rowspan="5">WIKI_CROP</td><td>UTKFace</td><td>Race</td><td>76.29</td><td>Naive Attack</td><td>77.09</td><td>99.65</td><td>1.20</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">66.23</td><td rowspan="2">FIDA Naive Attack</td><td>76.92</td><td>99.82</td><td>99.82</td></tr><tr><td>66.40</td><td>97.84</td><td>1.30</td></tr><tr><td>RAF_DB</td><td>Emotion</td><td></td><td>FIDA Naive Attack</td><td>66.72</td><td>99.10</td><td>99.10</td></tr><tr><td rowspan="2">UTKFace</td><td rowspan="2">Age</td><td rowspan="2">63.56</td><td rowspan="2">FIDA</td><td>62.25 62.33</td><td>99.82 99.87</td><td>2.80</td></tr><tr><td></td><td></td><td>99.87</td></tr><tr><td rowspan="5">FairFace</td><td>UTKFace</td><td>Race</td><td>79.73</td><td>Naive Attack FIDA</td><td>76.63</td><td>99.58</td><td>9.70</td></tr><tr><td rowspan="2">RAF_DB</td><td rowspan="2">Emotion</td><td rowspan="2">71.47</td><td rowspan="2">Naive Attack FIDA</td><td>76.46</td><td>99.41</td><td>99.41</td></tr><tr><td>66.20</td><td>0</td><td>0</td></tr><tr><td rowspan="2">UTKFace</td><td rowspan="2">Age</td><td rowspan="2">65.73</td><td rowspan="2">Naive Attack FIDA</td><td>65.48</td><td>97.47</td><td>97.47</td></tr><tr><td>65.33 65.12</td><td>98.92 99.29</td><td>3.40 99.29</td></tr></table>

Evaluation Metrics. We evaluate utility, attack efectiveness, and defense performance. Clean Accuracy (CA) measures the clean encoder’s downstream performance on benign images. Benign Accuracy (BA) measures the backdoored encoder’s downstream performance on benign inputs. Attack Success Rate (ASR) is the percentage of triggered test inputs assigned to the attackerselected target before defense. Attack Success Rate After Defense (ASR-AD) measures the same outcome after defensive filtering. True Positive Rate (TPR) is the proportion of attacks detected, while False Positive Rate (FPR) is the proportion of benign instances falsely flagged. True Negative Rate (TNR), equal to 1 − FPR, is the proportion correctly accepted. Area Under the ROC Curve (AUROC) summarizes separation across thresholds.

Experimental Environment. The experiments were conducted on a server equipped with 4 NVIDIA RTX A5000 GPUs (24GB VRAM each). The implementation is based on PyTorch 2.4.1 with Python 3.8.20 and CUDA 12.1. Each training or evaluation job runs on a single GPU, while independent configurations are executed in parallel when multiple GPUs are available. Standard libraries including NumPy, OpenCV, and Kornia are used for data processing and augmentation.

## 4.2 Main Results

In this section, we evaluate the overall performance of the complete FIDA framework across diverse experimental scenarios. Table 1 presents a comprehensive summary of the results on two pre-training datasets transferred to three downstream classification tasks.

4.2.1 Efectiveness and Utility. First, we assess the primary goals of any backdoor attack: high success rate and minimal impact on benign functions. As shown in Table 1, FIDA consistently achieves a near-perfect ASR, exceeding 99% across almost all tested scenarios. Notably, FIDA maintains a high BA comparable to the CA of clean models. The degradation in benign performance is negligible, demonstrating that FIDA successfully implants a backdoor without compromising the encoder’s general utility.

Table 2. Comparison with representative backdoor atacks against self-supervised encoders using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td>Method</td><td>BA (%)</td><td>ASR (%)</td></tr><tr><td>BadEncoder</td><td>74.2670</td><td>92.5754</td></tr><tr><td>DRUPE</td><td>65.6402</td><td>97.0892</td></tr><tr><td>GhostEncoder</td><td>76.5239</td><td>61.8435</td></tr><tr><td>FIDA</td><td>76.9200</td><td>99.8200</td></tr></table>

![](images/dd2d2ba781f6b26979d941f461439d921b078f0497b5e690a8ca5f0dec20523d.jpg)  
(a) STRIP

![](images/34792b1d849da7c312a4727c9775c72b665d7fb6c1cc7fd7ec3f9f8d28bed78b.jpg)  
(b) STRIP-CL

![](images/de6ac3273643e4f92e490a2c8769c55063321effa8d54b8bfea1ccaa3c617a38.jpg)  
(c) SymND

![](images/7867a7b4aff885b5834eef17ded0d1d8a6facce53e6e8c26638bb6a0c5e28207.jpg)  
(d) ASSET

Fig. 4. Evaluation of defense evasion against STRIP, STRIP-CL, SymND and ASSET  
![](images/1ef026e10f7592603e62b3ae48e5e30a1d11abdc9d65ce4018c8d4d8e5265e4f.jpg)  
P L<sup>1</sup>-Norm=0.1076  
L<sup>1</sup>-Norm=1322.18

![](images/751d9acd13556a629d13a3e3e4a0d160a19dcc071698726b204e95bb030223dc.jpg)  
P L<sup>1</sup>-Norm=0.1149  
L<sup>1</sup>-Norm=1411.98  
Fig. 5. Evaluation of defense evasion against DECREE

Table 2 shows that FIDA achieves the highest ASR while retaining competitive BA. DRUPE approaches FIDA in ASR but substantially reduces BA, whereas GhostEncoder preserves BA but obtains a lower ASR. FIDA therefore provides the best BA–ASR balance in this setting.

4.2.2 Defense Evasion. We evaluate FIDA against representative perturbation-based, data-separation, and model-inversion defenses. The results demonstrate FIDA’s strong ability to evade these diferent defense mechanisms.

STRIP, STRIP-CL and SymND. We first consider widely-used techniques designed to filter inputs at run-time. STRIP [12] and its variant STRIP-CL [43] operate by superimposing inputs with benign samples and measuring the entropy of prediction shifts, assuming backdoored inputs behave diferently. We also evaluate against SymND [56], notably the first defense specifically designed to detect backdoors in self-supervised facial representations by analyzing feature stability under perturbation. The failure of entropy-based defenses is evident in Fig. 4 (a) (b) (c), where the entropy distributions of clean and backdoored samples completely overlap. Crucially, as reported in Table 1, while Naive Attack is efectively neutralized by SymND (with ASR-AD dropping to a negligible level), FIDA’s ASR-AD remains virtually identical to its original high ASR. This indicates that the induced feature instability successfully prevents these mechanisms from distinguishing and filtering out triggered samples.

![](images/5f2e5ba414af41386ed22c2bd330cc7918f9052a9e211f346792d49c40fd6d55.jpg)

![](images/f5186b2091103791de1cce6063c8539d5485ffb447c80be6eb5fc87217ceef28.jpg)  
(a) Complete ROC curve and cali-(b) Detection performance in the lowbrated operating point. FPR region.  
Fig. 6. Detection performance of DeDe against FIDA. The calibrated threshold achieves a high TPR but produces an FPR of 95.72%. Under practical FPR constraints, DeDe detects only a limited proportion of FIDA-triggered samples. Both curves are computed from the per-sample reconstruction scores.

ASSET. We further challenged FIDA against ASSET [33], a state-of-the-art defense that actively optimizes the encoder to maximize the loss variance of poisoned samples for separation. FIDA demonstrates robust evasion against this active separation mechanism. Under ASSET’s standard adaptive thresholding, the detection yields a negligible True Positive Rate (TPR) of 0.08% at a False Positive Rate (FPR) of 0.20%, meaning over 99.9% of the backdoor samples successfully evade sanitization. As shown in Fig. 4 (d), the normalized loss scores of FIDA-triggered and benign samples overlap substantially. This overlap prevents the Gaussian Mixture Model from establishing a valid separation boundary, rendering the defense inefective in filtering out malicious inputs.

DECREE. We also evaluated FIDA against DECREE [10] (Fig. 5), a prominent static defense tailored for self-supervised encoders. DECREE inverts potential triggers and flags a model as backdoored if the $\bar { \mathcal { P } } \mathcal { L } ^ { 1 } { \cdot } \mathrm { N o r m }$ of the inverted trigger falls below a detection threshold $\tau = 0 . 1$ Evaluation on the FIDA-poisoned encoder yields a PL<sup>1</sup>-Norm of 0.11. With this value surpassing �, DECREE erroneously classifies the FIDA encoder as “benign”. This evasion suggests that FIDA’s semantic triggers, which resemble natural facial features, violate the low-norm assumption of DECREE, thereby preventing the defense from isolating the backdoor pattern via inversion.

DeDe. We adapt the recent decoder-based DeDe defense [22] by training its decoder for 200 epochs on clean unlabeled WIKI\_CROP images. Following its reconstruction protocol, we use three independently sampled masks and set the detection threshold to 1.5 times the clean calibration mean. This produces a threshold of 5259.8383 without using triggered test labels. DeDe detects 4557 of 4558 triggered inputs, but it also rejects 4538 of 4741 clean inputs. As shown in Fig. 6(a), its 99.98% TPR is accompanied by a 95.72% FPR and a TNR of only 4.28%. Although the overall AUROC is 0.8225, the calibrated operating point classifies almost all normal samples as suspicious. Fig. 6(b) further shows that, under FPR budgets of 1%, 5%, and 10%, the corresponding TPR values decrease to 9.68%, 30.63%, and 45.68%, respectively. Therefore, reducing the false-positive rate to a practically usable range causes most FIDA-triggered samples to evade detection. These results indicate that DeDe cannot efectively distinguish normal and FIDA-triggered samples in this experiment.

Table 3. BARBIE under heterogeneously-sourced calibration $( \omega _ { \mathrm { m a x } } { = } 0 . 5 )$ . FPR is measured via leave-oneencoder-out on unseen clean encoder profiles.
<table><tr><td>Downstream task</td><td>FPR</td></tr><tr><td>UTKFace (Race)</td><td>53.3%</td></tr><tr><td>RAF-DB (Emotion)</td><td>53.3%</td></tr><tr><td>UTKFace (Age)</td><td>56.7%</td></tr></table>

Table 4. Performance after applying MIMIC to FIDA encoders. BA-D and ASR-D denote benign accuracy and atack success rate after purification.
<table><tr><td>Pre-training Dataset</td><td>Downstream Dataset</td><td>CA(%)</td><td>BA-D (%)</td><td>ASR-D (%)</td></tr><tr><td rowspan="2">WIKI_CROP</td><td>UTKFace</td><td>76.29</td><td>63.81</td><td>46.33</td></tr><tr><td>RAF-DB</td><td>66.23</td><td>55.11</td><td>2.71</td></tr><tr><td rowspan="2">FairFace</td><td>UTKFace</td><td>79.73</td><td>65.91</td><td>44.54</td></tr><tr><td>RAF-DB</td><td>71.47</td><td>59.65</td><td>3.67</td></tr></table>

BARBIE. BARBIE [51] is a recent model-level detector. It flags an encoder when its inverted Relative Competition Score indicators fall outside a normal range calibrated from known clean models. We follow BARBIE’s original self-supervised protocol: a SimCLR ResNet-18 encoder, a twolayer classifier, OLR/ILR inversion, mean-centered RCS indicators, $\alpha { = } 0 . 0 0 1$ $\gamma { = } 0 ,$ and $\omega _ { \mathrm { m a x } } { = } 0 . 5$ . We retain its data-free, weights-only profiling and measure profile-level FPR by leave-one-encoder-out evaluation on unseen clean profiles. As shown in Table 3, every FIDA scenario raises an alarm. However, the same boundary also flags more than half of the clean profiles. BARBIE’s min–max calibration assumes clean models from the same domain and cannot absorb profile diferences across heterogeneous upstream domains. It therefore cannot reliably separate FIDA from normal encoders trained on diferent domains in the supply-chain setting.

MIMIC [16]. MIMIC treats the suspicious encoder as a teacher and trains a new student encoder on a small clean proxy set. A new downstream classifier must then be trained on the student representation. Following its protocol, we provide clean data equal to 4% of the corresponding pre-training dataset. This procedure requires trusted upstream-domain data, teacher–student optimization, and downstream retraining, which adds substantial computation and limits its practicality for resource-constrained users. We evaluate four FIDA encoders trained with WIKI\_CROP or FairFace on UTKFace and RAF-DB. Table 4 shows that BA-D is 11.12–13.82 percentage points below CA in every setting, indicating substantial damage to benign utility. Although MIMIC reduces ASR-D, considerable residual attack success remains on UTKFace, while the stronger reduction on RAF-DB is accompanied by a benign-accuracy loss above 11 percentage points. MIMIC therefore does not achieve uniform, utility-preserving purification against FIDA.

## 4.3 Ablation Study

4.3.1 Loss Components and Weights. To isolate the contribution of each objective, we remove one loss term at a time and report the resulting utility, attack efectiveness, and post-defense efectiveness in Table 5. Removing $L _ { 0 }$ weakens the trigger-to-target mapping, whereas removing $L _ { 1 }$ produces a smaller loss of attack efectiveness. Removing $L _ { 2 }$ damages both benign utility and attack efectiveness. This indicates that preserving clean representations also provides a stable feature space in which the target mapping can be learned.

Table 5. Ablation study on diferent loss terms using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td>Removed Loss Terms</td><td>CA(%)</td><td>BA(%)</td><td>ASR(%)</td><td>ASR-AD(%)</td></tr><tr><td> $L _ { 0 }$ </td><td rowspan="6">76.29</td><td>76.71</td><td>45.30</td><td>45.30</td></tr><tr><td> $L _ { 1 }$ </td><td>75.24</td><td>82.10</td><td>82.10</td></tr><tr><td> $L _ { 2 }$ </td><td>59.61</td><td>57.42</td><td>57.42</td></tr><tr><td> $L _ { 3 }$ </td><td>77.09</td><td>99.65</td><td>1.20</td></tr><tr><td>None</td><td>76.92</td><td>99.82</td><td>99.82</td></tr></table>

![](images/30590fc74ba8aed3f8415b13fe35ad2ab5322e7fd9982396b64fb411fe62f553.jpg)

![](images/6c0af9a521ff144547e48676d11238a845eb5380be67be6c7103b86f72400887.jpg)

(c) λ<sub>2</sub>  
![](images/fdb8146643a1e6353af673ed40d49f9982c1552862aad967f5160fc41e353965.jpg)

![](images/88511a677ec3b394f1b092c00a80ea061f899cbf6f768cd4131acdb07b9364f1.jpg)  
Fig. 7. Sensitivity to the four loss weights. Each panel varies one coeficient while fixing the other three to 1. Solid blue lines with circles denote BA, dashed orange lines with crosses denote ASR, and the vertical doted line marks the default unit weight.

The role of $L _ { 3 }$ is diferent from those of the alignment and fidelity terms. Removing it leaves benign utility and the undefended attack largely intact, but makes the attack inefective after SymND. Thus, $L _ { 3 }$ primarily supports defense evasion by weakening the feature-stability cue rather than establishing the basic backdoor mapping.

We further vary all four loss coeficients while fixing the other three, as shown in Fig. 7. The extended sweep shows that attack alignment is sensitive to insuficient or excessive weighting, while an overly large clean-feature weight makes benign preservation dominate the attack objective. The instability weight is comparatively tolerant once activated, consistent with its specialized role in defense evasion. We therefore retain unit weights as the balanced default.

4.3.2 Noise Intensity. To examine whether FIDA depends on a narrowly selected perturbation magnitude, we vary the Gaussian-noise standard deviation used during instability training; the results are summarized in Table 6. Benign utility and attack efectiveness remain stable across the tested range, and the post-defense results closely track the undefended attack. This consistency suggests that the instability objective does not require a precisely tuned noise intensity. We therefore retain the intermediate setting $\sigma = 3 0$ as the default.

4.3.3 Physical Degradations. Following the robustness-test design of INACTIVE [52], we independently apply Gaussian blur, JPEG compression, Poisson noise, and salt-and-pepper noise to the clean and triggered UTKFace test images. The attacked encoder remains fixed, and the downstream classifier follows the same 500-epoch training protocol as the standard evaluation. ASR excludes samples whose ground-truth label is already the target class, preventing ordinary correct target predictions from being counted as successful attacks. The undegraded reference obtains 76.92% BA and 99.62% strict ASR.

Table 6. Impact of noise intensity using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td rowspan="2">Noise Std Dev</td><td rowspan="2">BA(%) ASR(%)</td><td rowspan="2"></td><td colspan="3">ASR-AD(%)</td></tr><tr><td>STRIP</td><td>STRIP-CL</td><td>SymND</td></tr><tr><td>10</td><td>76.17</td><td>99.50</td><td>99.50</td><td>99.50</td><td>99.50</td></tr><tr><td>20</td><td>76.08</td><td>98.03</td><td>98.03</td><td>98.03</td><td>98.03</td></tr><tr><td>30</td><td>76.92</td><td>99.82</td><td>99.82</td><td>99.82</td><td>99.82</td></tr><tr><td>40</td><td>76.55</td><td>99.71</td><td>99.71</td><td>99.71</td><td>99.71</td></tr><tr><td>50</td><td>77.29</td><td>99.65</td><td>99.65</td><td>99.65</td><td>99.65</td></tr></table>

![](images/a75abba6ddc7aa63db702914e607d4620bd63023c1dcec50cac4838bb26dc410.jpg)  
Fig. 8. Robustness of FIDA under simulated physical-world image degradations using WIKI\_CROP upstream and UTKFace downstream. Each panel varies the strength of one degradation. For Gaussian blur, each tick reports the kernel size � and the sampled � interval. Solid blue lines with circles denote BA, dashed orange lines with crosses denote ASR, and faint doted horizontal lines denote the corresponding undegraded references.

To determine how deployment-like corruption afects FIDA, we compare the degradation trends in Fig. 8. Mild blur largely preserves both utility and attack efectiveness, whereas stronger smoothing weakens the localized semantic evidence used by the trigger before it removes the broader facial information needed for benign classification. JPEG recompression produces a diferent imbalance: benign features remain comparatively usable, but fine trigger-related details are strongly suppressed. Salt-and-pepper and Poisson noise mainly reduce benign utility while leaving the target mapping more stable. FIDA is therefore more resilient to stochastic pixel corruption than to transformations that smooth or compress semantic facial details.

4.3.4 Generalization. To examine whether FIDA depends on a particular self-supervised objective, we implement it with SimCLR, BYOL, and MoCo v2 under the same upstream and downstream setting; the results are reported in Table 7. These frameworks represent contrastive learning, asymmetric prediction, and momentum-based representation learning. Benign utility remains comparable across them, while attack efectiveness is lower with BYOL and MoCo v2 than with SimCLR. Their diferent update rules and feature geometries may change how quickly a shared attack configuration establishes trigger-to-target alignment. Nevertheless, ASR-AD remains equal to the corresponding ASR for all three frameworks, indicating that the instability-based evasion mechanism is not tied to SimCLR.

Table 7. Generalization across SSL frameworks using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td rowspan="2">Architecture</td><td rowspan="2">CA(%)</td><td rowspan="2">BA(%) ASR(%)</td><td rowspan="2"></td><td colspan="3">ASR-AD(%)</td></tr><tr><td>STRIP</td><td>STRIP-CL</td><td>SymND</td></tr><tr><td>SimCLR</td><td>76.29</td><td>76.92</td><td>99.82</td><td>99.82</td><td>99.82</td><td>99.82</td></tr><tr><td>BYOL</td><td>75.13</td><td>75.79</td><td>87.74</td><td>87.74</td><td>87.74</td><td>87.74</td></tr><tr><td>MoCo v2</td><td>76.46</td><td>75.01</td><td>80.04</td><td>80.04</td><td>80.04</td><td>80.04</td></tr></table>

Table 8. Evaluation with diferent encoder backbones using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td>Backbone</td><td>BA (%)</td><td>ASR (%)</td><td>ASR-AD (%)</td></tr><tr><td>ResNet-18</td><td>76.92</td><td>99.82</td><td>99.82</td></tr><tr><td>ResNet-34</td><td>73.95</td><td>92.61</td><td>92.61</td></tr><tr><td>VGG-16-BN</td><td>67.05</td><td>84.71</td><td>84.71</td></tr></table>

Table 9. Generalization across semantic triggers using WIKI\_CROP upstream and UTKFace downstream.
<table><tr><td rowspan="2">Trigger</td><td rowspan="2"></td><td rowspan="2">CA(%) BA(%) ASR(%)</td><td rowspan="2"></td><td colspan="3">ASR-AD(%)</td></tr><tr><td>STRIP</td><td>STRIP-CL</td><td>SymND</td></tr><tr><td rowspan="3">Eyebrow Lip Glasses</td><td rowspan="3">76.29</td><td>76.92</td><td>99.82</td><td>99.82</td><td>99.82</td><td>99.82</td></tr><tr><td>76.65</td><td>98.71</td><td>98.71</td><td>98.71</td><td>98.71</td></tr><tr><td>75.49</td><td>96.07</td><td>96.07</td><td>96.07</td><td>96.07</td></tr><tr><td>Beard</td><td></td><td>75.76</td><td>81.86</td><td>81.86</td><td>81.86</td><td>81.86</td></tr></table>

We next test whether the attack transfers across diferent backbone architectures. Table 8 compares three backbones: the original ResNet-18, the deeper ResNet-34, and VGG-16-BN, which has no residual connections. ResNet-34 preserves a stronger utility–efectiveness balance than VGG-16-BN. Residual connections may help retain pretrained representations during malicious finetuning, whereas VGG-16-BN learns a diferent feature hierarchy under the same loss weights. The unchanged relationship between ASR and ASR-AD nevertheless shows that the evasion mechanism transfers beyond the residual family. Backbone-specific tuning may further improve the balance on VGG-16-BN.

Finally, we examine whether FIDA relies on one specific semantic appearance by evaluating four triggers with diferent locations, structures, and spatial coverage; the results are presented in Table 9. Eyebrow and Red Lip are localized cosmetic modifications applied to diferent facial regions, and their similar behavior shows that the target mapping is not tied to one position or color change. Glasses introduce an input-adaptive accessory whose geometry follows each face, yet the attack remains efective. Beard modifies a broader and more variable lower-face region, making a consistent target mapping harder to learn. These results show that FIDA supports cosmetic, accessory-based, and facial-hair triggers, while also revealing that trigger variability and coverage influence attack efectiveness.

![](images/9c57fad5c1c05261c0aa9dc04fc15164e23e473cf6e13abb8d5aa515662830d1.jpg)  
Fig. 9. Race-group analysis on UTKFace. Gray, blue, and orange bars denote CA, BA, and strict ASR, respectively. Strict ASR measures the proportion of triggered non-target samples classified as the target class. Since White is the atack target, its ASR is undefined and marked as N/A.

![](images/0b7b0419926e89b656c0f7da438a50543fcb624864a320a074d7096f156b7f7b.jpg)  
Fig. 10. A qualitative face-retrieval example on VGGFace2. Top: a clean query. Botom: the same query with a trigger.

## 4.4 Demographic Bias and Fairness Implications

Semantic facial modifications may interact with demographic diferences in facial appearance. We therefore conduct a subgroup analysis using the five race labels in UTKFace. FIDA is trained for 200 epochs using WIKI\_CROP as the upstream dataset. The downstream classifier is trained for 500 epochs on UTKFace. The clean encoder provides CA. The FIDA encoder provides BA and ASR. We compute CA and BA on 4,741 clean test images. We compute strict ASR on 4,558 triggered images with their original race labels. White is the attack target. We therefore report ASR only for the four non-target groups.

As shown in Fig. 9, strict ASR ranges from 99.08% to 99.74% across the four non-target groups. The gap is only 0.66 percentage points. The group-wise diference between BA and CA ranges from −0.87 to +2.21 percentage points. Others has the lowest CA and BA. This UTKFace label is a heterogeneous catch-all category. It also has the smallest clean test subset, with 351 images. Its diverse composition and limited sample size make it harder to classify. Its BA is 27.07%, close to its CA of 27.92%. Thus, FIDA does not cause the low accuracy of this group.

The results do not show a pronounced race-group diference in ASR. They also do not show that FIDA amplifies the existing group diferences in utility. This analysis uses one target class and the race labels available in UTKFace. It does not cover intersectional attributes or the natural occurrence of semantic facial features. We will examine these factors with additional target classes and demographic annotations in future work.

Table 10. Zero-shot face retrieval results on VGGFace2. Both encoders are pre-trained on WIKI\_CROP and frozen during retrieval. Higher identification rates indicate beter clean utility, while higher ASR indicates a more successful targeted atack.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Query</td><td colspan="2">Identification Rate (%)</td><td colspan="2">Target ASR (%)</td></tr><tr><td>Rank-1</td><td>Rank-5</td><td>ASR@1</td><td>ASR@5</td></tr><tr><td>Clean encoder</td><td>Clean</td><td>50.88</td><td>75.00</td><td></td><td></td></tr><tr><td>Clean encoder</td><td> Triggered</td><td></td><td></td><td>11.67</td><td>48.67</td></tr><tr><td>FIDA encoder</td><td>Clean</td><td>43.88</td><td>69.38</td><td></td><td></td></tr><tr><td>FIDA encoder</td><td>Triggered</td><td></td><td></td><td>55.83</td><td>93.50</td></tr></table>

## 4.5 Generalization to Metric-Based Retrieval

We conduct a cross-domain, zero-shot retrieval experiment on VGGFace2 [1] using its identity and demographic annotations [14]. VGGFace2 contains 3.31 million face images from 9,131 identities, with substantial variation in pose, age, illumination, and ethnicity. We use 1% of its identities in this experiment. The WIKI\_CROP-pretrained clean and FIDA encoders are frozen to rank a disjoint query–gallery split by cosine similarity. Rank-� measures whether the top-� results contain the query identity, while ASR@� measures whether they contain the attacker-selected target group (White). Figure 10 and Table 10 report the qualitative and quantitative results.

Relative to the clean-encoder control, FIDA increases ASR by 44.16 and 44.83 percentage points at Rank 1 and Rank 5, with ASR@5 reaching 93.50%. Clean Rank-1 and Rank-5 identification decrease by 7.00 and 5.62 points, indicating that the targeted shift transfers to cross-domain retrieval with a moderate utility cost.

## 5 Discussion and Open Challenges

The experimental results show that FIDA can manipulate self-supervised facial encoders while preserving their downstream utility in most evaluated settings. They also reveal several broader questions that cannot be fully addressed by classification accuracy and attack success rate alone. In this section, we discuss possible defenses against feature instability, the challenges of physical-world deployment, and the demographic and societal implications of semantic facial triggers.

## 5.1 Potential Countermeasures Against Feature Instability

FIDA is designed to manipulate the feature-stability assumption used by perturbation-based defenses. The defense results suggest that relying on a single stability cue may be insuficient when the encoder itself has been maliciously optimized against that cue. A stronger defense may need to combine several sources of evidence, such as representation consistency, reconstruction behavior, temporal information, and downstream prediction patterns.

Temporal consistency provides one possible direction. Facial analysis is often applied to video rather than isolated images. Although a semantic trigger may remain visually consistent across adjacent frames, a compromised encoder may produce abnormal changes in its representations when pose, illumination, or image quality varies. A defense could therefore examine feature trajectories across consecutive frames instead of making a decision from one perturbed image. Abrupt representation changes that are inconsistent with the observed facial motion may provide additional evidence of abnormal encoder behavior.

Multi-modal consistency provides another direction. Multimedia applications may jointly process facial appearance, speech, and contextual information. For example, if a visual trigger changes an emotion prediction to “Angry,” the system could compare this output with acoustic and temporal cues. A persistent disagreement between modalities would not by itself prove the presence of a backdoor, but it could initiate further inspection. These directions require video or multi-modal benchmarks and remain open problems beyond the current image-based evaluation.

## 5.2 Complexities in Physical-World Multimedia Deployment

As evaluated in Section 4.3.3, FIDA remains efective under mild Gaussian blur and the tested salt-and-pepper and Poisson noise conditions. However, Figure 8 also shows that stronger blur and JPEG compression substantially reduce ASR. These results indicate that the robustness of semantic triggers depends on how a degradation changes their facial appearance. Local pixel noise often preserves the main semantic structure, whereas blur and compression may weaken the visual details used to establish the trigger-to-target mapping.

These controlled degradations approximate several efects introduced during image capture, storage, and transmission, but they do not fully reproduce a physical environment. Real makeup and wearable accessories may appear diferently across cameras, illumination conditions, facial poses, viewing distances, and video-compression pipelines. Evaluating these factors with physically reproduced triggers is therefore an important extension of the current simulated experiments.

## 5.3 Demographic Bias, Fairness, and Societal Impact

Section 4.4 reports CA, BA, and ASR separately across the five race groups in UTKFace. As shown in Fig. 9, attack efectiveness and benign performance are not completely uniform across the evaluated groups. The lower accuracy of the “Other” group may be related to the heterogeneous racial identities represented by this broad category. The subgroup results therefore provide a more informative view than aggregate BA and ASR alone.

Semantic facial triggers create an additional fairness concern because eyebrow appearance, lip color, glasses, and facial hair can occur naturally. Their prevalence and appearance may vary with age, gender expression, cultural practices, skin tone, and facial structure. Some users may therefore naturally resemble a trigger without intentionally presenting one. The subgroup experiment measures diferences in attack efectiveness, but it does not yet measure the frequency of such accidental activation.

The consequences may also depend on the downstream application. In demographic classification, a targeted backdoor could systematically map members of one group to another category, corrupting demographic statistics or decisions based on them. In identity verification, access control, telehealth, or automated assessment, unequal attack activation could result in denial of service, inaccurate evaluation, or disproportionate harm to particular populations.

Defenses should not treat glasses, facial hair, makeup, or eyebrow appearance themselves as evidence of malicious behavior. Otherwise, ordinary facial attributes could become a source of group-dependent false alarms. A safer direction is to detect abnormal model behavior rather than judging the semantic attribute. Future work will examine accidental activation rates, intersections among race, age, and gender, and group-specific false-positive rates of backdoor defenses.

## 6 Conclusion

Current backdoor attacks are often easily caught by advanced defenses. To solve this problem, we propose FIDA, a novel backdoor attack framework. By introducing a new feature instability mechanism, FIDA trains the encoder to increase the sensitivity of triggered features along perturbation directions sampled during attack optimization . Our comprehensive evaluations across diferent datasets, model architectures, and trigger types show that FIDA achieves high attack success rates while maintaining the model’s normal performance. It successfully evades several evaluated defense mechanisms.

For future work, we plan to explore several directions. First, we aim to extend the FIDA framework from the digital domain to physical world attacks. Since our semantic triggers correspond to natural facial features, we plan to implement them using physical mediums, such as makeup or accessories. Second, we will study how to design stronger defenses that can detect backdoor behaviors hidden by this induced instability.

## Acknowledgments

We are deeply grateful to the anonymous reviewers for their thoughtful suggestions and rigorous reviews that helped refine our research.

## References

[1] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. 2018. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018) IEEE, 67–74.

[2] Nicholas Carlini and Andreas Terzis. 2022. Poisoning and Backdooring Contrastive Learning. In International Conference on Learning Representations (ICLR).

[3] Tuo Chen, Jie Gui, Minjing Dong, Ju Jia, Lanting Fang, and Jian Liu. 2025. Backdooring Self-Supervised Contrastive Learning by Noisy Alignment. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 3684–3693.

[4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning. PmLR, 1597–1607.

[5] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. 2020. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297 (2020).

[6] Xinlei Chen and Kaiming He. 2021. Exploring simple siamese representation learning. In Proceedings of the Computer Vision and Pattern Recognition Conference. 15750–15758

[7] Siyuan Cheng, Guanhong Tao, Yingqi Liu, Guangyu Shen, Shengwei An, Shiwei Feng, Xiangzhe Xu, Kaiyuan Zhang, Shiqing Ma, and Xiangyu Zhang. 2024. Lotus: Evasive and resilient backdoor attacks through sub-partitioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 24798–24809.

[8] Jongwook Choi, Taehoon Kim, Yonghyun Jeong, Seungryul Baek, and Jongwon Choi. 2024. Exploiting style latent flows for generalizing deepfake video detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1133–1143.

[9] Khoa Doan, Yingjie Lao, and Ping Li. 2021. Backdoor attack with imperceptible input and latent modification. Advances in Neural Information Processing Systems 34 (2021), 18944–18957.

[10] Shiwei Feng, Guanhong Tao, Siyuan Cheng, Guangyu Shen, Xiangzhe Xu, Yingqi Liu, Kaiyuan Zhang, Shiqing Ma, and Xiangyu Zhang. 2023. Detecting backdoors in pre-trained encoders. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 16352–16362.

[11] Claudio Ferrari, Federico Becattini, Leonardo Galteri, and Alberto Del Bimbo. 2023. (Compress and restore) N: A robust defense against adversarial attacks on image classification. ACM Transactions on Multimedia Computing, Communications and Applications 19, 1s (2023), 1–16.

[12] Yansong Gao, Change Xu, Derui Wang, Shiping Chen, Damith C Ranasinghe, and Surya Nepal. 2019. Strip: A defence against trojan attacks on deep neural networks. In Proceedings ofthe 35th Annual Computer Security Applications Conference. 113–125.

[13] Zheng Gao and Ioannis Patras. 2024. Self-supervised facial representation learning with facial region awareness. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 2081–2092.

[14] Antonio Greco, Gennaro Percannella, Mario Vento, and Vincenzo Vigilante. 2020. Benchmarking deep network architectures for ethnicity recognition using a new large face dataset. Machine Vision and Applications 31, 7 (2020), 67.

[15] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. 2020. Bootstrap your own latent-a new approach to self-supervised learning. Advances in Neural Information Processing Systems 33 (2020), 21271–21284.

[16] Tingxu Han, Weisong Sun, Ziqi Ding, Chunrong Fang, Hanwei Qian, Jiaxun Li, Zhenyu Chen, and Xiangyu Zhang. 2025. Mutual information guided backdoor mitigation for pre-trained encoders. IEEE Transactions on Information Forensics and Security (2025).

[17] Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, and Jing Dong. 2024. Exploiting backdoors of face synthesis detection with natural triggers. ACM Transactions on Multimedia Computing, Communications and Applications 21, 2

(2024), 1–24.

[18] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. Masked autoencoders are scalable vision learners. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 16000–16009.

[19] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum contrast for unsupervised visua representation learning. In Proceedings of the Computer Vision and Pattern Recognition Conference. 9729–9738.

[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 770–778.

[21] Aryana Hou, Li Lin, Justin Li, and Shu Hu. 2025. Rethinking Individual Fairness in Deepfake Detection. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 11424–11433.

[22] Sizai Hou, Songze Li, and Duanyi Yao. 2025. DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 20675–20684.

[23] Jinyuan Jia, Yupei Liu, and Neil Zhenqiang Gong. 2022. Badencoder: Backdoor attacks to pre-trained encoders in self-supervised learning. In 2022 IEEE Symposium on Security and Privacy (SP). IEEE, 2043–2059.

[24] Kimmo Karkkainen and Jungseock Joo. 2021. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision. 1548–1558.

[25] Changjiang Li, Ren Pang, Zhaohan Xi, Tianyu Du, Shouling Ji, Yuan Yao, and Ting Wang. 2023. An embarrassingly simple backdoor attack on self-supervised learning. In Proceedings ofthe IEEE/CVFInternational Conference on Computer Vision. 4367–4378.

[26] Shan Li, Weihong Deng, and JunPing Du. 2017. Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 2852–2861.

[27] Yuanyuan Liu, Shaoze Feng, Shuyang Liu, Yibing Zhan, Dapeng Tao, Zijing Chen, and Zhe Chen. 2025. Samplecohesive pose-aware contrastive facial representation learning. International Journal ofComputer Vision 133, 6 (2025), 3727–3745.

[28] Yuanyuan Liu, Wenbin Wang, Yibing Zhan, Shaoze Feng, Kejun Liu, and Zhe Chen. 2023. Pose-disentangled contrastive learning for self-supervised facial representation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 9717–9728.

[29] Hao Luo, Zhi Qin, Lin Wang, Ziyue Wu, and Min Yang. 2025. SGBA: Subspace guidance backdoor attack with feature alignment in image classification. Applied Soft Computing (2025), 113857.

[30] Wanlun Ma, Derui Wang, Ruoxi Sun, Minhui Xue, Sheng Wen, and Yang Xiang. 2023. The "Beatrix" Resurrections: Robust Backdoor Detection via Gram Matrices. In NDSS.

[31] Xiaoxing Mo, Yechao Zhang, Leo Yu Zhang, Wei Luo, Nan Sun, Shengshan Hu, Shang Gao, and Yang Xiang. 2024. Robust backdoor detection for deep learning via topological evolution dynamics. In 2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2048–2066.

[32] Trevine Oorlof, Surya Koppisetti, Nicolò Bonettini, Divyaraj Solanki, Ben Colman, Yaser Yacoob, Ali Shahriyari, and Gaurav Bharaj. 2024. Avf: Audio-visual feature fusion for video deepfake detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 27102–27112.

[33] Minzhou Pan, Yi Zeng, Lingjuan Lyu, Xue Lin, and Ruoxi Jia. 2023. ASSET: Robust backdoor data detection across a multiplicity of deep learning paradigms. In 32nd USeNIX security symposium (USeNIX security 23). 2725–2742.

[34] Rasmus Rothe, Radu Timofte, and Luc Van Gool. 2015. Dex: Deep expectation of apparent age from a single image. In Proceedings ofthe IEEE International Conference on Computer Vision Workshops. 10–15.

[35] Shuvendu Roy and Ali Etemad. 2023. Contrastive learning of view-invariant representations for facial expressions recognition. ACM Transactions on Multimedia Computing, Communications and Applications 20, 4 (2023), 1–22.

[36] Ibtissam Saadi, Abdenour Hadid, Douglas W Cunningham, Abdelmalik Taleb-Ahmed, and Yassin El Hillali. 2025. PE-CLIP: A Parameter-Eficient Fine-Tuning of Vision Language Models for Dynamic Facial Expression Recognition. ACM Transactions on Multimedia Computing, Communications and Applications (2025).

[37] Aniruddha Saha, Ajinkya Tejankar, Soroush Abbasi Koohpayegani, and Hamed Pirsiavash. 2022. Backdoor attacks on self-supervised learning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 13337–13346.

[38] Weiyu Sun, Xinyu Zhang, Hao Lu, Yingcong Chen, Ting Wang, Jinghui Chen, and Lu Lin. 2024. Backdoor contrastive learning via bi-level trigger optimization. In International Conference on Learning Representations (ICLR).

[39] Zhonglin Sun, Chen Feng, Ioannis Patras, and Georgios Tzimiropoulos. 2024. Lafs: Landmark-based facial selfsupervised learning for face recognition. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 1639–1649.

[40] Te Juin Lester Tan and Reza Shokri. 2020. Bypassing backdoor detection algorithms in deep learning. In 2020 IEEE European Symposium on Security and Privacy (EuroS&P). IEEE, 175–183.

[41] Guanhong Tao, Zhenting Wang, Shiwei Feng, Guangyu Shen, Shiqing Ma, and Xiangyu Zhang. 2024. Distribution preserving backdoor attack in self-supervised learning. In 2024 IEEE Symposium on Security and Privacy (SP). IEEE,

2029–2047.

[42] Laurens Van der Maaten and Geofrey Hinton. 2008. Visualizing data using t-SNE. Journal of Machine Learning Research 9, 11 (2008).

[43] Qiannan Wang, Changchun Yin, Liming Fang, Zhe Liu, Run Wang, and Chenhao Lin. 2024. GhostEncoder: Stealthy backdoor attacks with dynamic triggers to pre-trained encoders in self-supervised learning. Computers & Security 142 (2024), 103855.

[44] Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, and Baoyuan Wu. 2025. Badvideo: Stealthy backdoor attack against text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19075–19084.

[45] Hui Wei, Hanxun Yu, Kewei Zhang, Zhixiang Wang, Jianke Zhu, and Zheng Wang. 2023. Moiré backdoor attack (MBA): A novel trigger for pedestrian detectors in the physical world. In Proceedings of the 31st ACM International Conference on Multimedia. 8828–8838.

[46] Mingfu Xue, Yinghao Wu, Leo Yu Zhang, Dujuan Gu, Yushu Zhang, and Weiqiang Liu. 2024. SSAT: Active authorization control and user’s fingerprint tracking framework for DNN IP protection. ACM Transactions on Multimedia Computing, Communications and Applications 20, 10 (2024), 1–24.

[47] Zhiyuan Yan, Yandan Zhao, Shen Chen, Mingyi Guo, Xinghe Fu, Taiping Yao, Shouhong Ding, Yunsheng Wu, and Li Yuan. 2025. Generalizing deepfake video detection with plug-and-play: Video-level blending and spatiotempora adapter tuning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 12615–12625.

[48] Yongqi Yang, Zhihao Qian, Ye Zhu, Olga Russakovsky, and Yu Wu. 2025. D<sup>3</sup>: scaling up deepfake detection by learning from discrepancy. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 23850–23859.

[49] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. 2021. Barlow twins: Self-supervised learning via redundancy reduction. In International Conference on Machine Learning. PMLR, 12310–12320.

[50] Shengfang Zhai, Yinpeng Dong, Qingni Shen, Shi Pu, Yuejian Fang, and Hang Su. 2023. Text-to-image difusion models can be easily backdoored through multimodal data poisoning. In Proceedings ofthe 31st ACM International Conference on Multimedia. 1577–1587.

[51] Hanlei Zhang, Yijie Bai, Yanjiao Chen, Zhongming Ma, and Wenyuan Xu. 2025. Barbie: Robust backdoor detection based on latent separability. In NDSS.

[52] Hanrong Zhang, Zhenting Wang, Boheng Li, Fulin Lin, Tingxu Han, Mingyu Jin, Chenlu Zhan, Mengnan Du, Hongwei Wang, and Shiqing Ma. 2025. Invisible Backdoor Attack against Self-supervised Learning. In Proceedings of the Computer Vision and Pattern Recognition Conference. 25790–25801.

[53] Jinghuai Zhang, Hongbin Liu, Jinyuan Jia, and Neil Zhenqiang Gong. 2024. Data poisoning based backdoor attacks to contrastive learning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 24357–24366.

[54] Yechao Zhang, Yuxuan Zhou, Tianyu Li, Minghui Li, Shengshan Hu, Wei Luo, and Leo Yu Zhang. 2025. Secure Transfer Learning: Training Clean Model Against Backdoor in Pre-Trained Encoder and Downstream Dataset. In 2025 IEEE Symposium on Security and Privacy (SP). IEEE, 1–19.

[55] Zhifei Zhang, Yang Song, and Hairong Qi. 2017. Age progression/regression by conditional adversarial autoencoder. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 5810–5818.

[56] Liyue Zhu, Changchun Yin, Liming Fang, and Zhen Qin. 2025. SymND: Detecting Backdoor Attacks in Self-Supervised Facial Representation Tasks. In 2025 IEEE International Conference on Multimedia and Expo (ICME).