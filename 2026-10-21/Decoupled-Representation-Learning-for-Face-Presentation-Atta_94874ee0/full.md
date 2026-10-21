# Decoupled Representation Learning for Face Presentation Attack Detection under Smart Glass Capture

Raghavendra Ramachandra<sup>1</sup> Narayan Vetrekar<sup>2</sup> Neha Naik<sup>2</sup> Rajendra Gad<sup>2</sup> <sup>1</sup> SAFE Centre, Norwegian University of Science and Technology (NTNU), Gjøvik, Norway <sup>2</sup> School of Physical and Applied Sciences, Goa University, Goa, India raghavendra.ramachandra @ ntnu.no, vetrekarnarayan @ unigoa.ac.in

## Abstract

Face presentation attack detection is typically studied underfixed camera or handheld acquisition, whereas smart glass capture introduces a more challenging setting with ego motion, changing viewpoint, background clutter, distance variation, and less controlled illumination. To address this challenge, we present SmaGla-FPAD, a smart glass based face presentation attack database acquired using Ray-Ban Meta Wayfarer Smart Glasses across indoor, outdoor, and in-car settings. The database contains 297,000 sample images with both bona fide and silicone mask based attacks. We further propose DeCoupPAD-FAS++, a decoupled representation learning framework that separates spoof related evidence from nuisance sensitive capture variation through complementary attack and style branches, pseudo-domain supervision, adversarial suppression, and prompt guided semantic alignment. Experiments on SmaGla-FPAD show that the proposed method achieves the best balance between attack rejection and bona fide acceptance in both in domain and cross scenario evaluation. Ablation results further confirm the contribution of disentanglement, pseudo-domain supervision, andprompt based regularisation under smart glass capture.

## 1. Introduction

Face recognition is widely deployed because it is contactless, convenient, and compatible with standard RGB cameras. It is used in applications ranging from border control and secure access to mobile authentication and remote identity verification. Although three dimensional and multispectral face recognition can offer stronger resilience to spoofing, their deployment is often limited by sensor cost and system complexity [13, 43, 39]. In contrast, two dimensional face recognition remains dominant in practical systems because it can be implemented using conventional cameras and integrated readily into existing platforms [1, 39]. This is particularly relevant in mobile and wearable settings, where compact hardware and natural user interaction are essential.

![](images/d704668051014f5d6b716cb2189d5b020e853c82f709ea1d947397c79e6d7203.jpg)  
(b)Data acquisition for Presentation Attack  
Figure 1. Illustration of the data acquisition scenario: (a) subject on the move towards the smart glass, (b) subject seated in the car.

At the same time, RGB face recognition is vulnerable to presentation attacks, including printed photographs, display replays, photo wrapped media, silicone masks, and three dimensional facial replicas [39, 40, 32, 13, 46, 1]. This threat is especially important in operational settings such as border control, where imaging conditions are less controlled and the capture process must remain efficient. Robust presentation attack detection (PAD) is therefore essential for trustworthy deployment.

Despite substantial progress in face PAD, generalisation remains a major challenge. Performance often degrades when illumination, motion, background, sensor placement, or attack presentation changes. This problem is more severe under smart glass capture, where the camera is body worn, the viewpoint is dynamic, and the recorded face is affected by motion blur, distance variation, background clutter, and unstable lighting. Existing PAD benchmarks rarely address this setting explicitly, and most prior methods are developed for fixed-camera or smartphone capture rather than first-person wearable acquisition. Consequently, face PAD still lacks both suitable datasets and models for robust smart glass capture.

In this work, we address this gap from both the data and modelling perspectives. First, we introduce SmaGla-FPAD, a smart glass based face presentation attack database acquired using Ray-Ban Meta Wayfarer Smart Glasses in indoor, outdoor, and in-car settings. As shown in Fig. 1, the dataset captures bona fide and silicone-mask-based attacks under realistic variation in scene, motion, and acquisition geometry. Second, we propose DeCoupPAD-FAS++, a decoupled representation learning framework for face PAD under smart glass capture. The framework separates spoof discriminative evidence from nuisance sensitive capture variation through complementary attack and style branches, adversarial pseudo-domain suppression, and prompt guided image-text regularisation using a frozen vision language model.

Extensive experiments on the proposed dataset show that the method achieves the strongest overall trade-off between attack rejection and bona fide acceptance in both in-domain and cross-scenario evaluation. The results further indicate that robust smart glass PAD requires not only strong backbone features, but also explicit treatment of nuisance variation during training. The main contributions of this work are threefold:

• We introduce SmaGla-FPAD, a smart glass based face presentation attack database collected in indoor, outdoor, and in-car settings, with bona fide and silicone mask attack samples acquired under realistic operational variability.

• We propose DeCoupPAD-FAS++, a decoupled representation learning framework that separates spoof related evidence from nuisance sensitive capture variation through complementary attack and style branches, pseudo-domain supervision, adversarial suppression, and prompt guided semantic alignment.

• We present in-domain, cross-scenario, and ablation studies showing that the proposed method provides the best overall operating point under challenging smart glass capture conditions.

## 2. Related Work

Face PAD has evolved from handcrafted descriptors to deep representation learning [43]. Early methods relied on texture, image quality, and motion cues. Local descriptors such as LBP and LBP TOP were used to capture fine surface differences between bona fide and spoof samples [38, 3, 9, 39], while quality based methods exploited artefacts such as specular reflection, blur, colour distortion, and sensor noise [6, 50, 37]. Motion and physiological signal based approaches further incorporated cues such as remote photoplethysmography to separate live faces from print and replay attacks [26, 33, 27]. With the rise of deep learning, convolutional networks became the dominant paradigm for learning spoof related spatial and temporal features directly from RGB and depth data [17, 47, 35, 51]. Subsequent work introduced attention mechanisms, transformer based architectures, anomaly detection, and one class formulations to capture broader facial context and improve robustness to unseen attacks [14, 16, 30, 12, 7, 49, 2, 19, 11]. To address domain shift, several studies further incorporated domain adaptation and domain generalisation strategies based on adversarial alignment, distribution matching, and meta regularisation [8, 41, 42, 10, 5].

More recent PAD research has increasingly explored foundation models, vision language learning, prompt based optimisation, and multimodal transformers. CLIP based and vision language approaches align visual spoof evidence with semantic prompt descriptions in order to improve cross domain generalisation and reduce dependence on dataset specific appearance statistics [44, 29, 28, 24, 22, 45, 34, 21]. Prompt learning has been extended through knowledge guided prompt construction, gradient aware prompt tuning, and domain specific prompt design [15, 25, 28], while multimodal transformer models have exploited RGB, depth, and thermal cues through cross modal attention and expert routing [23, 31]. Recent attack agnostic prompt formulations have also attempted to unify the treatment of physical and digital attacks [18]. Despite these advances, most existing methods are developed for conventional fixed camera or handheld acquisition and do not explicitly address the strong nuisance variation introduced by smart glass capture. In contrast, the present work targets this setting directly and formulates smart glass PAD as a decoupled representation learning problem in which spoof evidence and capture dependent variation are modelled separately.

## 3. SmaGla-FPAD: Face Presentation Attack Database for Smart Glass Capture

Most existing face PAD benchmarks are acquired with fixed cameras or handheld devices and therefore do not fully capture the variability introduced by first person wearable acquisition. Smart glass capture is more challenging because it combines ego motion, viewpoint variation, background clutter, changing subject to sensor distance, and less controlled illumination. To study PAD under these conditions, we introduce SmaGla-FPAD, a smart glass based face presentation attack database collected using Ray-Ban Meta Wayfarer Smart Glasses with a front facing RGB camera supporting 12 MP image capture and video recording up to 3K resolution. Figure 1 illustrates the acquisition process.

![](images/5369ca38d2154dcccea587d53df61c768b75ada0466b22bbe1eb42386cdc1388.jpg)  
Figure 2. Sample images show the data acquired in SmaGla-FPAD in indoor, outdoor, and in-car scenarios.

The database is collected in three settings, namely Indoor, Outdoor, and in-car, chosen to reflect capture scenarios relevant to border control and mobile identity verification. The Indoor setting is acquired in a corridor under controlled lighting, whereas the Outdoor and in-car settings are recorded under natural daylight and therefore exhibit stronger variation in illumination, shadow, and background content. In Indoor and Outdoor settings, subjects walk naturally towards the smart glass wearer from an initial distance of approximately 6 m, which introduces progressive changes in scale and pose. In the in-car setting, the subject remains seated while the smart glass device records from a distance of 1 to 2 m. To increase scene diversity, the in-car subset is collected across 10 vehicle types of different sizes.

SmaGla-FPAD contains two classes, namely Bona fide and Presentation attack. In total, the database comprises 297,000 sample images, including 225,000 bona fide samples and 72,000 presentation attack samples. The bona fide subset contains recordings from 50 subjects aged between 18 and 40 years, including 18 male and 32 female participants. For the Indoor and Outdoor settings, each subject is recorded over multiple sessions spanning four months, with 5 to 8 videos per subject in each setting, and each sequence lasting between 30 and 50 seconds. In the in-car setting, recordings are acquired from both left and right sides of the vehicle.

<table><tr><td rowspan=1 colspan=1>Data Type</td><td rowspan=1 colspan=1>Condition</td><td rowspan=1 colspan=1>Total Samples</td></tr><tr><td rowspan=1 colspan=1>Bona fide</td><td rowspan=1 colspan=1>Indoor</td><td rowspan=1 colspan=1>75,000</td></tr><tr><td rowspan=1 colspan=1>Bona fide</td><td rowspan=1 colspan=1>Outdoor</td><td rowspan=1 colspan=1>75,000</td></tr><tr><td rowspan=1 colspan=1>Bona fide</td><td rowspan=1 colspan=1>in-car</td><td rowspan=1 colspan=1>75,000</td></tr><tr><td rowspan=1 colspan=1>Presentation attack</td><td rowspan=1 colspan=1>Indoor</td><td rowspan=1 colspan=1>24,000</td></tr><tr><td rowspan=1 colspan=1>Presentation attack</td><td rowspan=1 colspan=1>Outdoor</td><td rowspan=1 colspan=1>24,000</td></tr><tr><td rowspan=1 colspan=1>Presentation attack</td><td rowspan=1 colspan=1>in-car</td><td rowspan=1 colspan=1>24,000</td></tr><tr><td rowspan=1 colspan=2>Total Samples</td><td rowspan=1 colspan=1>297,000</td></tr></table>

Table 1. Statistical summary of the proposed SmaGla-FPAD database.

The presentation attack subset is generated using four custom built silicone masks, comprising two male and two female masks, worn by 15 subjects. Silicone masks are used because they represent a realistic three dimensional presentation attack instrument with high visual fidelity and substantial surface structure. To increase acquisition realism, accessories such as wigs, hoodies, and eyeglasses are incorporated during recording. Two attack conditions are considered: in the first, the attacker wears a silicone mask together with eyeglasses, a wig, and a hoodie; in the second, the attacker wears the same accessories except for eyeglasses. For each condition, 5 to 12 videos of 40 to 60 seconds are recorded over multiple sessions within 30 days. Table 1 summarises the dataset composition, and Figure 2 shows representative examples from the three settings.

## 4. Proposed Method: DeCoupPAD-FAS++

We propose DeCoupPAD-FAS++, a decoupled representation learning framework for face PAD under smart glass capture. The central idea is to learn a spoof discriminative representation while explicitly separating nuisance sensitive capture variation induced by first person wearable acquisition. This is achieved by decoupling the representation into two complementary branches: an attack content branch for final PAD and a style domain branch for nuisance modelling. The attack branch is further regularised by adversarial pseudo-domain suppression and prompt guided imagetext alignment. An overview of the framework is shown in Fig. 3.

## 4.1. Overall Architecture

Given an input RGB face frame $x _ { i }$ , a frozen DINOv2 backbone extracts patch tokens $T _ { i } ~ \in ~ \mathbb { R } ^ { N \times C }$ , where N and C denote the number and dimension of patch tokens, respectively. Smart glass capture introduces subattack branch and PAD head are retained.

![](images/209b954fcbde4e97751cf5a138fcddc8b3386efff1b9b34109914eb589415c24.jpg)  
Figure 3. Overview of DeCoupPAD-FAS++. A frozen DINOv2 backbone extracts patch tokens that are decoupled into an attack content branch for PAD and a style domain branch for nuisance modelling. During training, the attack branch is regularised by adversarial pseudodomain suppression and prompt guided image-text alignment using text projection and attack-to-text projection. Solid arrows denot inference, whereas dotted arrows indicate training only supervision.

stantial nuisance variation, including motion blur, illumination fluctuation, sharpness degradation, background clutter, and changes in subject to sensor distance. If not handled explicitly, such factors can become entangled with spoof cues and reduce generalisation. We therefore decompose the token representation into two branches with complementary roles. The attack content branch learns a spoof discriminative embedding $z _ { i } ^ { a }$ for final PAD, whereas the style domain branch learns a nuisance sensitive embedding $z _ { i } ^ { d }$ that captures wearable capture variation. The attack branch aggregates informative token interactions through a lightweight Tiny Q-Former, followed by semantic adaptation and prompt gating to produce the final gated representation ${ \bar { z } } _ { i } ^ { a }$ . In parallel, the style branch summarises the same token set through style sensitive statistics and projects them into $z _ { i } ^ { d }$ under pseudo-domain supervision. To reduce nuisance leakage into the PAD representation, ${ \bar { z } } _ { i } ^ { a }$ is additionally constrained by adversarial pseudo-domain learning through a gradient reversal layer. During training, prompt guided image-text alignment with frozen OpenCLIP provides semantic regularisation. During inference, only the

## 4.2. Attack Content Branch

The attack content branch transforms the patch tokens $T _ { i }$ into a spoof discriminative representation. First, token interactions are selectively aggregated by a lightweight Tiny Q-Former:

$$
Q _ { i } = \phi _ { \mathrm { q f } } ( T _ { i } ) ,\tag{1}
$$

where $\phi _ { \mathrm { q f } } ( \cdot )$ denotes the Q-Former mapping. This design is motivated by the observation that spoof artefacts under smart glass capture are often local, spatially heterogeneous, and easily obscured by nuisance variation.

The aggregated tokens are then refined by a semantic adapter and projected into the attack embedding space:

$$
\tilde { Q } _ { i } = \phi _ { \mathrm { s e m } } ( Q _ { i } ) , \qquad z _ { i } ^ { a } = \phi _ { \mathrm { a t t } } ( \tilde { Q } _ { i } ) ,\tag{2}
$$

where $\phi _ { \mathrm { s e m } } ( \cdot )$ and $\phi _ { \mathrm { a t t } } ( \cdot )$ denote the semantic adaptation and attack projection modules, respectively. A prompt gating module then recalibrates the resulting embedding:

$$
\bar { z } _ { i } ^ { a } = \mathrm { n o r m } ( z _ { i } ^ { a } \odot g _ { i } ) ,\tag{3}
$$

where $g _ { i }$ is a learned gate vector, $\odot$ denotes element wise multiplication, and norm(·) denotes feature normalisation. The final PAD score is computed solely from the gated attack representation:

$$
\hat { y } _ { i } = \sigma \big ( h _ { \mathrm { p a d } } ( \bar { z } _ { i } ^ { a } ) \big ) ,\tag{4}
$$

where $h _ { \mathrm { p a d } } ( \cdot )$ denotes the PAD head and $\sigma ( \cdot )$ is the sigmoid function.

## 4.3. Style Domain Branch

The style domain branch is designed to model nuisance sensitive variation specific to smart glass acquisition. Starting from the same patch tokens $T _ { i }$ , this branch first computes style sensitive token statistics and applies MixStyle1D during training to improve robustness to appearance variation. The resulting features are then processed by a domain adapter and a domain MLP to produce the latent style representation

$$
z _ { i } ^ { d } = \phi _ { \mathrm { s t y } } ( T _ { i } ) ,\tag{5}
$$

where $\phi _ { \mathrm { s t y } } ( \cdot )$ denotes the overall style projection function implemented by the style statistics module, MixStyle1D, domain adapter, and domain MLP. In practice, this branch captures token level characteristics associated with blur, illumination, and related acquisition effects. The resulting representation is normalised and used for pseudo-domain supervision so that nuisance related factors are assigned to a dedicated latent space rather than contaminating the spoof representation.

## 4.4. Pseudo-Domain Supervision and Adversarial Suppression

Each training sample is assigned a pseudo-domain label $d _ { i } \in \{ 1 , \ldots , K \}$ derived from simple image statistics, with $K = 9$ in our implementation. These labels act as lightweight proxies for nuisance conditions, such as brightness and blur, that are prominent under smart glass capture. The style embedding $\bar { z _ { i } ^ { d } }$ is passed to a domain head trained with pseudo-domain supervision so that this branch explicitly models nuisance sensitive variation.

In contrast, the gated attack embedding $\bar { z } _ { i } ^ { a }$ is encouraged to suppress the same information. Specifically, ${ \bar { z } } _ { i } ^ { a }$ is passed through a gradient reversal layer followed by an adversarial domain head trained to predict $d _ { i }$ . Owing to gradient reversal, this objective discourages the attack branch from encoding pseudo-domain specific cues. The two branches therefore play complementary roles: the style branch captures nuisance variation, whereas the attack branch is encouraged to remain spoof discriminative while being less sensitive to capture specific factors.

## 4.5. Prompt Guided Image-Text Alignment

To further regularise the attack branch, we introduce prompt guided image-text alignment during training. A set of bona fide and attack prompt templates is first encoded by a frozen OpenCLIP text encoder. The resulting text embeddings are then mapped to the shared semantic space by a lightweight text projection layer. In parallel, the gated attack embedding ${ \bar { z } } _ { i } ^ { a }$ is projected into the same space by an image-to-text projection layer. Alignment is then enforced between the projected attack representation and the corresponding class-specific prompt embeddings. This branch regularises the attack representation using high level semantic guidance, thereby reducing overfitting to dataset specific low level artefacts and improving robustness under scenario variation. The image-text alignment module is used only during training and introduces no additional cost at inference.

## 4.6. Training Objective

The proposed framework is trained using a multi term objective that jointly optimises PAD discrimination, nuisance modelling, nuisance suppression, and semantic regularisation. Let $L _ { \mathrm { p a d } }$ denote the binary PAD loss, $L _ { \mathrm { d o m } }$ the pseudo-domain classification loss on the style branch, $L _ { \mathrm { a d v } }$ the adversarial pseudo-domain loss applied through the gradient reversal layer on $\bar { z } _ { i } ^ { a } , L _ { \mathrm { i t c } }$ and $L _ { \mathrm { p t m } }$ the prompt guided image-text alignment losses, $L _ { \mathrm { o r t h } }$ the orthogonality regulariser between $z _ { i } ^ { a }$ and $z _ { i } ^ { d } .$ , and $L _ { \mathrm { s t y l e } }$ the style regularisation term. The overall loss is defined as

$$
\begin{array} { r } { L _ { \mathrm { t o t a l } } = L _ { \mathrm { p a d } } + \lambda _ { \mathrm { d o m } } L _ { \mathrm { d o m } } + \lambda _ { \mathrm { a d v } } L _ { \mathrm { a d v } } + \lambda _ { \mathrm { i t c } } L _ { \mathrm { i t c } } } \\ { + \lambda _ { \mathrm { p t m } } L _ { \mathrm { p t m } } + \lambda _ { \mathrm { o r t h } } L _ { \mathrm { o r t h } } + \lambda _ { \mathrm { s t y l e } } L _ { \mathrm { s t y l e } } . } \end{array}\tag{6}
$$

where $\lambda _ { \mathrm { d o m } } , \lambda _ { \mathrm { a d v } } , \lambda _ { \mathrm { i t c } } , \lambda _ { \mathrm { p t m } } , \lambda _ { \mathrm { o r t h } }$ , and $\lambda _ { \mathrm { s t y l e } }$ are scalar weighting coefficients.

Here, $L _ { \mathrm { p a d } }$ drives bona fide versus attack discrimination, $L _ { \mathrm { d o m } }$ encourages $z _ { i } ^ { d }$ to encode nuisance related variation, and $L _ { \mathrm { a d v } }$ suppresses the same information in ${ \bar { z } } _ { i } ^ { a }$ . The orthogonality term further separates the attack and style subspaces, while $L _ { \mathrm { i t c } }$ and $L _ { \mathrm { p t m } }$ provide semantic regularisation for the attack branch through prompt guided alignment.

## 4.7. Inference Stage

At inference, an input frame $x _ { i }$ is passed through the frozen DINOv2 backbone and the attack content branch to obtain the gated representation ${ \bar { z } } _ { i } ^ { a }$ , from which the final PAD score is computed as

$$
\hat { y } _ { i } = \sigma \big ( h _ { \mathrm { p a d } } ( \bar { z } _ { i } ^ { a } ) \big ) .\tag{7}
$$

The style domain branch and all auxiliary modules used for pseudo-domain supervision, adversarial suppression, and prompt guided image-text alignment are used only during training and are discarded at test time, ensuring a simple and efficient inference pipeline.

## 5. Experiments and Results

We first describe the experimental setup and evaluation protocols, and then present the results for in domain and cross scenario presentation attack detection.

## 5.1. Experimental Setup and Protocols

We evaluate the proposed DeCoupPAD-FAS++ under two smart glass acquisition settings, namely Car and Moving. The Moving setting is further divided into indoor and outdoor subsets and therefore introduces larger variability in pose, motion, blur, illumination, and background conditions than the Car setting. Based on these two acquisition settings, we define four experimental protocols to assess both in domain presentation attack detection and cross scenario generalisation.

For all experiments, the data are split using subject disjoint partitions for both bona fide samples and silicone mask attacks. The bona fide class contains 50 unique subjects, of which 35 are used for training, 5 for validation, and the remaining 10 for testing. The attack class contains four distinct silicone masks. Among these, two masks are used for training, one for validation, and one for testing. The same splitting strategy is repeated across three independent runs, and the final performance is reported as the mean together with the corresponding 95% confidence interval.

Evaluation is conducted exclusively on the test set. We report BPCER at the operating point corresponding to APCER equal to 1%, the resulting APCER measured on the test set, and the corresponding ACER. In all cases, the decision threshold is estimated on the validation set and then fixed for test evaluation. This protocol allows us to analyse not only the in domain effectiveness of the proposed method but also its ability to retain spoof discriminative cues under scenario change.

Experiment E1: In domain PAD in the Car scenario. The first protocol evaluates in domain PAD in the Car scenario using still face frames. Training and validation are performed exclusively on Car data, where bona fide samples and silicone mask attacks are both drawn from the Car scenario according to the subject disjoint partitions. Testing is then carried out on the corresponding disjoint Car test subjects and test attacks. Since both optimisation and evaluation are performed within the same acquisition setting, E1 serves as the most controlled protocol and provides the in domain reference performance of the proposed method.

Experiment E2: In domain PAD in the Moving scenario. The second protocol evaluates in domain PAD in the Moving scenario. This setting is more challenging than the Car scenario because it includes subject motion and stronger environmental variation. Training and validation use bona fide samples and silicone mask attacks captured under the Moving scenario, following the same subject disjoint partitions. Testing is performed on the corresponding disjoint Moving test subjects and test attacks. This protocol measures the ability of the proposed method to detect attacks under more variable acquisition conditions.

Table 2. Results for E1 under the Car scenario. Performance is reported as mean with 95% confidence interval over three runs. Lower values are better.
<table><tr><td rowspan=1 colspan=1>PAD Method</td><td rowspan=1 colspan=1>BPCER@APCER=1%</td><td rowspan=1 colspan=1>APCER@thr=1%</td><td rowspan=1 colspan=1>ACER</td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>0.59 [0.00, 99.11]</td><td rowspan=1 colspan=1>1.44 [0.07, 13.26]</td><td rowspan=1 colspan=1>1.01 [0.08, 4.51]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>0.39 [0.00, 99.99]</td><td rowspan=1 colspan=1>0.44 [0.02, 4.32]</td><td rowspan=1 colspan=1>0.41 [0.04, 2.95]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>16.05 [0.00, 99.01]</td><td rowspan=1 colspan=1>8.02 [0.01, 97.21]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>2.16 [0.00, 100.00]</td><td rowspan=1 colspan=1>4.16 [0.00, 92.20]</td><td rowspan=1 colspan=1>3.39 [0.22, 24.17]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>67.29 [33.14, 90.31]</td><td rowspan=1 colspan=1>1.25 [0.01, 29.44]</td><td rowspan=1 colspan=1>34.27 [22.01, 48.78]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>49.95 [0.88, 99.40]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>24.97 [2.33, 77.10]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>0.23 [0.01, 2.56]</td><td rowspan=1 colspan=1>0.11 [0.00, 2.13]</td></tr></table>

To make the effect of environmental variation explicit, we additionally report two test only subsets using the same trained model. In E2a, evaluation is restricted to the Moving Indoor subset, whereas in E2b, evaluation is restricted to the Moving Outdoor subset. These two analyses do not introduce additional training, but instead isolate the influence of indoor and outdoor capture conditions on the final PAD performance.

Experiment E3: Cross scenario generalisation from Moving to Car. The third protocol evaluates cross scenario generalisation from the Moving scenario to the Car scenario. Training and validation are performed using bona fide samples and silicone mask attacks from the Moving scenario, including both indoor and outdoor data, according to the subject disjoint partitions. Testing is then carried out on the corresponding disjoint Car test subjects and test attacks. This protocol is more demanding than the in domain setting because the training and test data arise from different acquisition conditions. It therefore measures whether the proposed method learns spoof related cues that remain effective across scenario change rather than relying on scenario specific characteristics.

Experiment E4: Cross scenario generalisation from Car to Moving. The fourth protocol evaluates cross scenario generalisation in the reverse direction, namely from the Car scenario to the Moving scenario. Training and validation are performed using bona fide samples and silicone mask attacks from the Car scenario, following the same subject disjoint partitions, whereas testing is carried out on the corresponding disjoint Moving test subjects and test attacks. For completeness, we report results on the combined Moving test set as well as separately on the indoor and outdoor subsets using the same trained model. This protocol assesses whether a model trained under relatively constrained Car conditions remains effective when transferred to the more variable Moving scenario.

## 5.2. Results and Discussion

Table 2 reports the results of Experiment E1 in the Car scenario. All compared methods were implemented using publicly available open source code where available, and otherwise reimplemented, and evaluated under the same subject disjoint splits and experimental protocol as the proposed method. The proposed method achieves the best overall performance, with the lowest ACER of 0.11% [0.00, 2.13] and the lowest APCER at the validation derived operating point, namely 0.23% [0.01, 2.56]. This indicates that the proposed model provides the most favourable balance between attack rejection and bona fide acceptance under matched training and test conditions.

The compared methods show clear trade offs. SAFAS attains zero BPCER at APCER = 1%, but exhibits a much larger APCER of 16.05%, indicating weak attack rejection. MTFAS achieves zero APCER, but only at the cost of a very large BPCER of 49.95%. MEFAS shows a similar tendency, with a relatively low APCER of 1.25% but a very large BPCER of 67.29%, leading to poor overall ACER. CFPL, InstructFLIP, and TTSP provide more balanced behaviour than these extreme cases, yet all remain inferior to the proposed method. Overall, the E1 results establish that the proposed representation provides the strongest in domain performance in the controlled Car scenario.

Table 3. Results for E2 under the Moving scenario. Performance is reported as mean with 95% confidence interval over three runs for the Indoor, Outdoor, and Both subsets. Lower values are better.  
Best results in each subset are shown in bold
<table><tr><td rowspan=1 colspan=1>PAD Method</td><td rowspan=1 colspan=1>BPCER@APCER=1%</td><td rowspan=1 colspan=1>APCER@thr=1%</td><td rowspan=1 colspan=1>ACER</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Indoo</td><td rowspan=1 colspan=1>r</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>27.12 [0.76, 84.76]</td><td rowspan=1 colspan=1>2.49 [0.00, 100.00]</td><td rowspan=1 colspan=1>14.80 [1.10, 53.00]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>28.11 [0.07, 98.04]</td><td rowspan=1 colspan=1>4.94 [0.00, 100.00]</td><td rowspan=1 colspan=1>16.53 [2.98, 50.81]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>24.53 [0.32, 92.96]</td><td rowspan=1 colspan=1>4.00 [0.00, 100.00]</td><td rowspan=1 colspan=1>14.27 [2.19, 48.46]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>51.40 [0.21, 99.86]</td><td rowspan=1 colspan=1>4.31 [0.00, 100.00]</td><td rowspan=1 colspan=1>27.89 [2.42, 81.44]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>99.17 [96.03, 99.87]</td><td rowspan=1 colspan=1>1.13 [0.88, 1.44]</td><td rowspan=1 colspan=1>50.15 [49.65, 50.64]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>99.96 [1.84, 100.00]</td><td rowspan=1 colspan=1>0.01 [0.00, 88.05]</td><td rowspan=1 colspan=1>49.98 [49.88, 50.08]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>20.59 [0.07, 95.71]</td><td rowspan=1 colspan=1>5.15 [0.00, 100.00]</td><td rowspan=1 colspan=1>12.87 [2.43, 41.05]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Outdo</td><td rowspan=1 colspan=1>or</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>42.57 [2.98, 94.48]</td><td rowspan=1 colspan=1>2.49 [0.00, 100.00]</td><td rowspan=1 colspan=1>22.53 [3.87, 63.10]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>34.10 [0.30, 97.69]</td><td rowspan=1 colspan=1>4.94 [0.00, 100.00]</td><td rowspan=1 colspan=1>19.52 [2.23, 65.17]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>32.07 [0.27, 97.40]</td><td rowspan=1 colspan=1>4.00 [0.00, 100.00]</td><td rowspan=1 colspan=1>18.04 [1.51, 67.51]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>65.74 [1.19, 99.83]</td><td rowspan=1 colspan=1>4.31 [0.00, 100.00]</td><td rowspan=1 colspan=1>35.02 [5.06, 81.85]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>94.64 [91.05, 96.94]</td><td rowspan=1 colspan=1>1.13 [0.88, 1.44]</td><td rowspan=1 colspan=1>47.89 [46.28, 49.49]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>99.81 [0.18, 100.00]</td><td rowspan=1 colspan=1>0.01 [0.00, 88.05]</td><td rowspan=1 colspan=1>49.93 [49.60, 50.26]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>29.98 [0.11, 98.29]</td><td rowspan=1 colspan=1>5.16 [0.00, 100.00]</td><td rowspan=1 colspan=1>17.57 [0.87, 74.01]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Both</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>31.84 [2.31, 87.57]</td><td rowspan=1 colspan=1>2.49 [0.00, 100.00]</td><td rowspan=1 colspan=1>17.16 [2.47, 56.29]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>30.81 [0.30, 96.43]</td><td rowspan=1 colspan=1>4.94 [0.00, 100.00]</td><td rowspan=1 colspan=1>17.88 [3.06, 54.50]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>27.81 [0.36, 94.75]</td><td rowspan=1 colspan=1>4.00 [0.00, 100.00]</td><td rowspan=1 colspan=1>15.91 [1.90, 56.91]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>57.84 [1.03, 99.55]</td><td rowspan=1 colspan=1>4.31 [0.00, 100.00]</td><td rowspan=1 colspan=1>31.07 [4.16, 79.01]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>97.08 [94.76, 98.43]</td><td rowspan=1 colspan=1>1.13 [0.88, 1.44]</td><td rowspan=1 colspan=1>49.10 [48.28, 49.91]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>99.99 [0.47, 100.00]</td><td rowspan=1 colspan=1>0.01 [0.00, 88.05]</td><td rowspan=1 colspan=1>49.96 [49.76, 50.16]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>24.71 [0.18, 95.15]</td><td rowspan=1 colspan=1>5.16 [0.00, 100.00]</td><td rowspan=1 colspan=1>14.93 [1.67, 55.72]</td></tr></table>

The quantitative results of Experiment E2 are summarised in Table 3. Compared with E1, all methods exhibit degraded performance, confirming that the Moving scenario is substantially more challenging due to motion, pose change, and environmental variability. On the combined test set, which provides the most representative evaluation for this protocol, the proposed method achieves the lowest ACER of 14.93% [1.67, 55.72] together with the lowest BPCER at APCER = 1%, namely 24.71% [0.18,

Table 4. Results for E3, cross scenario generalisation from Moving to Car. Performance is reported as mean with 95% confidence interval over three runs. Lower values are better.
<table><tr><td rowspan=1 colspan=1>PAD Method</td><td rowspan=1 colspan=1>BPCER@APCER=1%</td><td rowspan=1 colspan=1>APCER@thr=1%</td><td rowspan=1 colspan=1>ACER</td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>59.25 [0.27, 99.85]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>29.63 [0.36, 95.31]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>62.98 [0.15, 99.95]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>31.49 [0.30, 96.53]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>67.50 [0.02, 100.00]</td><td rowspan=1 colspan=1>0.06 [0.00, 99.31]</td><td rowspan=1 colspan=1>33.78 [0.50, 96.01]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>67.03 [0.00, 100.00]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>33.51 [0.37, 96.64]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>99.99 [19.89, 100.00]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>49.99 [49.97, 50.01]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>100.00 [100.00, 100.00]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>50.00 [50.00, 50.00]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>38.37 [0.00, 99.91]</td><td rowspan=1 colspan=1>0.04 [0.00, 98.53]</td><td rowspan=1 colspan=1>19.20 [0.00, 98.99]</td></tr></table>

95.15]. On the Indoor subset, the proposed method attains the lowest BPCER and the lowest ACER, while on the Outdoor subset it also achieves both the lowest BPCER and the lowest ACER. These observations are important because the Outdoor subset constitutes the more challenging condition within the Moving scenario.

Several compared methods obtain smaller APCER values than the proposed method, particularly MTFAS and MEFAS. However, this comes at the cost of substantially increased bona fide rejection. MTFAS exhibits extremely large BPCER across all three subsets, while MEFAS shows similarly poor BPCER despite its relatively low APCER. As a result, both methods yield much larger ACER than the proposed method. CFPL, InstructFLIP, and SAFAS provide more balanced behaviour than these extreme cases, yet they remain inferior to the proposed method on the most representative combined setting. Overall, these results indicate that the proposed method provides the best balance between attack rejection and bona fide acceptance under the more challenging Moving scenario, both across individual environmental conditions and on the combined test set.

Table 4 presents the results of Experiment E3, which evaluates cross scenario generalisation from Moving to Car. This is a challenging setting because the model must transfer spoof related cues learned under motion and environmental variability to a different and more constrained acquisition condition. Under this protocol, the proposed method achieves the lowest BPCER at APCER = 1%, 38.37% [0.00, 99.91], and the lowest ACER, 19.20% [0.00, 98.99]. This indicates that the proposed method transfers more effectively across scenarios than the compared baselines.

Although several compared methods achieve APCER values of 0.00%, this occurs together with very large BPCER values, indicating overly conservative behaviour that rejects a substantial proportion of bona fide samples. This tendency is especially clear for MEFAS and MTFAS. In contrast, the proposed method maintains APCER close to zero while substantially reducing BPCER, resulting in the best overall trade off between attack rejection and bona fide acceptance. These results suggest that the proposed representation captures spoof related cues that generalise better across acquisition conditions rather than relying primarily on scenario specific artefacts.

The results of Experiment E4 are shown in Table 5, where the model is trained on the Car scenario and evaluated on the Moving scenario. This protocol is complementary to E3 and is particularly important because cross scenario transfer is not necessarily symmetric. The proposed method achieves the lowest ACER across all three evaluation subsets, with values of 11.68% [5.90, 21.08] for Indoor, 10.80% [4.87, 21.27] for Outdoor, and 11.25% [5.56, 20.66] for Both. It also attains the lowest APCER in all three cases, indicating stronger suppression of attack acceptance after transfer from the more constrained Car scenario to the more variable Moving scenario.

Table 5. Results for E4, cross scenario generalisation from Car to Moving. Performance is reported as mean with 95% confidence interval over three runs for the Indoor, Outdoor, and Both subsets. Lower values are better. Best results in each subset are shown in bold.
<table><tr><td rowspan=1 colspan=1>PAD Method</td><td rowspan=1 colspan=1>BPCER@APCER=1%</td><td rowspan=1 colspan=1>APCER@thr=1%</td><td rowspan=1 colspan=1>ACER</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Ind</td><td rowspan=1 colspan=1>oor</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>2.12 [0.00, 66.28]</td><td rowspan=1 colspan=1>53.72 [26.57, 79.02]</td><td rowspan=1 colspan=1>27.92 [13.91, 47.28]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>1.56 [0.00, 99.99]</td><td rowspan=1 colspan=1>33.51 [10.31, 67.08]</td><td rowspan=1 colspan=1>17.53 [8.14, 32.66]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>0.01 [0.00, 99.54]</td><td rowspan=1 colspan=1>82.80 [16.61, 99.62]</td><td rowspan=1 colspan=1>41.41 [23.04, 62.17]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>0.33 [0.00, 99.99]</td><td rowspan=1 colspan=1>88.33 [18.13, 99.90]</td><td rowspan=1 colspan=1>44.33 [26.82, 63.16]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>26.44 [0.40, 93.54]</td><td rowspan=1 colspan=1>22.42 [1.91, 74.76]</td><td rowspan=1 colspan=1>24.43 [4.50, 64.84]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>6.75 [0.00, 99.29]</td><td rowspan=1 colspan=1>100.00 [100.00, 100.00]</td><td rowspan=1 colspan=1>53.37 [36.76, 88.16]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>2.84 [0.00, 86.31]</td><td rowspan=1 colspan=1>20.52 [6.76, 45.53]</td><td rowspan=1 colspan=1>11.68 [5.90, 21.08]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Out</td><td rowspan=1 colspan=1>door</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>0.73 [0.02, 7.61]</td><td rowspan=1 colspan=1>53.72 [26.57, 79.02]</td><td rowspan=1 colspan=1>27.22 [15.00, 43.56]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>9.51 [0.00, 100.00]</td><td rowspan=1 colspan=1>33.52 [10.31, 67.08]</td><td rowspan=1 colspan=1>21.51 [15.68, 28.58]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1>82.80 [16.61, 99.62]</td><td rowspan=1 colspan=1>41.40 [23.03, 62.17]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>0.55 [0.03, 4.88]</td><td rowspan=1 colspan=1>88.33 [18.13, 99.90]</td><td rowspan=1 colspan=1>44.40 [26.87, 63.30]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>18.71 [0.09, 93.15]</td><td rowspan=1 colspan=1>22.42 [1.91, 74.76]</td><td rowspan=1 colspan=1>20.57 [3.04, 62.59]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>18.65 [0.00, 97.91]</td><td rowspan=1 colspan=1>100.00 [100.00, 100.00]</td><td rowspan=1 colspan=1>68.65 [55.76, 70.16]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>1.09 [0.00, 42.45]</td><td rowspan=1 colspan=1>20.52 [6.76, 45.53]</td><td rowspan=1 colspan=1>10.80 [4.87, 21.27]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Bo</td><td rowspan=1 colspan=1>th</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CFPL [24]</td><td rowspan=1 colspan=1>1.33 [0.01, 25.57]</td><td rowspan=1 colspan=1>53.72 [26.57, 79.02]</td><td rowspan=1 colspan=1>27.52 [14.52, 45.13]</td></tr><tr><td rowspan=1 colspan=1>InstructFLIP [20]</td><td rowspan=1 colspan=1>6.18 [0.00, 100.00]</td><td rowspan=1 colspan=1>33.51 [10.31, 67.08]</td><td rowspan=1 colspan=1>19.84 [17.12, 22.84]</td></tr><tr><td rowspan=1 colspan=1>SAFAS [48]</td><td rowspan=1 colspan=1>0.00 [0.00, 97.91]</td><td rowspan=1 colspan=1>82.80 [16.61, 99.62]</td><td rowspan=1 colspan=1>41.40 [23.03, 62.17]</td></tr><tr><td rowspan=1 colspan=1>TTSP [52]</td><td rowspan=1 colspan=1>0.43 [0.14, 1.10]</td><td rowspan=1 colspan=1>88.33 [18.13, 99.90]</td><td rowspan=1 colspan=1>44.38 [26.86, 63.20]</td></tr><tr><td rowspan=1 colspan=1>MEFAS [4]</td><td rowspan=1 colspan=1>22.47 [0.28, 91.82]</td><td rowspan=1 colspan=1>22.42 [1.91, 74.76]</td><td rowspan=1 colspan=1>22.45 [4.03, 62.07]</td></tr><tr><td rowspan=1 colspan=1>MTFAS [36]</td><td rowspan=1 colspan=1>12.70 [0.00, 92.91]</td><td rowspan=1 colspan=1>100.00 [100.00, 100.00]</td><td rowspan=1 colspan=1>61.01 [52.76, 80.16]</td></tr><tr><td rowspan=1 colspan=1>Proposed Method</td><td rowspan=1 colspan=1>1.99 [0.00, 73.41]</td><td rowspan=1 colspan=1>20.52 [6.76, 45.53]</td><td rowspan=1 colspan=1>11.25 [5.56, 20.66]</td></tr></table>

Several compared methods achieve lower BPCER than the proposed method, but this occurs together with substantially larger APCER, which leads to inferior ACER. This tendency is particularly pronounced for SAFAS, TTSP, and MTFAS, all of which favour bona fide acceptance at the expense of weak attack rejection. MEFAS provides a more balanced operating point than these more extreme cases, yet it remains consistently inferior to the proposed method across Indoor, Outdoor, and Both evaluation subsets. In contrast, the proposed method provides the most favourable overall trade off between attack rejection and bona fide acceptance under cross scenario transfer. In summary, the E4 results show that the proposed representation transfers effectively from Car to Moving while maintaining the strongest overall operating point under the stronger nuisance variation of the Moving scenario.

Across all four protocols, a consistent pattern emerges. In the controlled in domain setting of E1, the proposed method achieves the best overall performance. In the more challenging in domain setting of E2, it remains the most balanced method across varying environmental conditions.

More importantly, in both cross scenario protocols, E3 and E4, the proposed model achieves the most favourable trade off between attack rejection and bona fide acceptance. These findings support the central design objective of the proposed method, namely to learn spoof discriminative representations that are less entangled with nuisance sensitive capture variation and therefore generalise more effectively across changing acquisition conditions.

## 5.3. Ablation Study

Table 6 reports the ablation study on the E3 protocol. The full model achieves the lowest BPCER and ACER, indicating that the complete formulation provides the most effective balance between attack rejection and bona fide acceptance. The CLS baseline and patch mean baseline perform substantially worse, showing that generic token aggregation is insufficient for robust cross-scenario PAD. Using only the attack branch already improves performance, whereas adding the style branch without proper disentanglement degrades it, confirming that spoof related and nuisance related factors must be separated rather than simply combined. A similar trend is observed in the remaining

Table 6. Ablation study on the E3 protocol. Results are reported as mean over three independent runs. Lower values are better.
<table><tr><td rowspan=1 colspan=1>Ablation</td><td rowspan=1 colspan=1>BPCER@APCER=1%</td><td rowspan=1 colspan=1>APCER@thr=1%</td><td rowspan=1 colspan=1>ACER%</td></tr><tr><td rowspan=1 colspan=1>CLS baseline</td><td rowspan=1 colspan=1>74.64</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>37.32</td></tr><tr><td rowspan=1 colspan=1>Patch mean baseline</td><td rowspan=1 colspan=1>64.88</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>32.44</td></tr><tr><td rowspan=1 colspan=1>Attack only</td><td rowspan=1 colspan=1>54.87</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>27.44</td></tr><tr><td rowspan=1 colspan=1>Attack + style, no disentanglement</td><td rowspan=1 colspan=1>62.56</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1>31.28</td></tr><tr><td rowspan=1 colspan=1>Full w/o Lorth</td><td rowspan=1 colspan=1>63.71</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>31.86</td></tr><tr><td rowspan=1 colspan=1>Full w/o pseudo-domain losses</td><td rowspan=1 colspan=1>57.89</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>28.95</td></tr><tr><td rowspan=1 colspan=1>Full w/o prompt alignment</td><td rowspan=1 colspan=1>62.30</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>31.16</td></tr><tr><td rowspan=1 colspan=1>Full + ITC only</td><td rowspan=1 colspan=1>55.38</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>27.70</td></tr><tr><td rowspan=1 colspan=1>Full + PTM only</td><td rowspan=1 colspan=1>56.65</td><td rowspan=1 colspan=1>0.133</td><td rowspan=1 colspan=1>28.40</td></tr><tr><td rowspan=1 colspan=1>Full model</td><td rowspan=1 colspan=1>38.37</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1>19.19</td></tr></table>

variants. Removing the orthogonality constraint, pseudodomain losses, or prompt alignment each increases ACER relative to the full model, while using only ITC or only PTM remains inferior to the complete formulation. Overall, the gain comes from the combined effect of disentanglement, pseudo-domain supervision, and prompt based alignment rather than from any single component alone.

## 6. Conclusion

We presented SmaGla-FPAD, a smart glass based face presentation attack database, and DeCoupPAD-FAS++, a decoupled framework for face PAD under first person wearable capture. The proposed method separates spoofrelated evidence from nuisance-sensitive capture variation and achieves a strong balance between attack rejection and bona fide acceptance across both in-domain and crossscenario protocols. The ablation study further confirms that disentanglement, pseudo-domain supervision, and prompt guided regularisation are all important for robust generalisation under smart glass capture. A limitation of the present work is that the evaluation is conducted on a single smart glass dataset with a limited set of silicone mask attacks.

Future work will extend the framework to broader attack types, additional presentation attack instruments and wearable capture devices, and cross-dataset evaluation.

## References

[1] DHS: Remote ID systems. https://www.dhs.gov/ science-and-technology/remote-identityvalidation-technology-demonstration/. Accessed: 2025-03-01.

[2] Y. Baweja, P. Oza, P. Perera, and V. M. Patel. Anomaly detection-based unknown face presentation attack detection. In 2020 IEEE International Joint Conference on Biometrics (IJCB), pages 1–9. IEEE, 2020.

[3] Z. Boulkenafet, J. Komulainen, and A. Hadid. Face antispoofing based on color texture analysis. In IEEE International Conference on Image Processing (ICIP), pages 2636– 2640. IEEE, 2015.

[4] L. Cai, H. Wang, J. Ji, X. Sun, L. Cao, and R. Ji. Me-fas: Multimodal text enhancement for cross-domain face antispoofing. IEEE Transactions on Information Forensics and Security, 20:5451–5464, 2025.

[5] R. Cai, Y. Cui, Z. Li, Z. Yu, H. Li, Y. Hu, and A. Kot. Rehearsal-free domain continual face anti-spoofing: Generalize more and forget less. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8037– 8048, 2023.

[6] H.-H. Chang and C.-H. Yeh. Face anti-spoofing detection based on multi-scale image quality assessment. Image and Vision Computing, 121:104428, 2022.

[7] H. Chen, G. Hu, Z. Lei, Y. Chen, N. M. Robertson, and S. Z. Li. Attention-based two-stream convolutional networks for face spoofing detection. IEEE Transactions on Information Forensics and Security, 15:578–593, 2019.

[8] Z. Chen, T. Yao, K. Sheng, S. Ding, Y. Tai, J. Li, F. Huang, and X. Jin. Generalizable representation learning for mixture domain face anti-spoofing. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 1132– 1139, 2021.

[9] I. Chingovska, A. Anjos, and S. Marcel. Biometrics evaluation under spoofing attacks. IEEE Transactions on Information Forensics and Security (T-IFS), 9(12):2264–2276, Dec 2014.

[10] M. Fang and N. Damer. Face presentation attack detection by excavating causal clues and adapting embedding statistics. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6269–6279, 2024.

[11] A. George and S. Marcel. Learning one class representations for face presentation attack detection using multi-channel convolutional neural networks. IEEE Transactions on Information Forensics and Security, 16:361–375, 2020.

[12] A. George and S. Marcel. On the effectiveness of vision transformers for zero-shot face anti-spoofing. In 2021 IEEE international joint conference on biometrics (IJCB), pages 1–8. IEEE, 2021.

[13] P.-K. Huang, J.-X. Chong, M.-T. Hsu, F.-Y. Hsu, C.-H. Chiang, T.-H. Chen, and C.-T. Hsu. A survey on deep learning-

based face anti-spoofing. APSIPA Transactions on Signal and Information Processing, 13(1):e34, 2024.

[14] Y.-H. Huang, J.-W. Hsieh, M.-C. Chang, L. Ke, S. Lyu, and A. S. Santra. Multi-teacher single-student visual transformer with multi-level attention for face spoofing detection. In BMVC, page 125, 2021.

[15] F. Jiang, Q. Li, B. Liu, W. Wang, C. Shan, Z. Sun, and M.- H. Yang. Learning knowledge-based prompts for robust 3d mask presentation attack detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[16] Y. Kong, X. Li, G. Hao, and C. Liu. Face anti-spoofing method based on residual network with channel attention mechanism. Electronics, 11(19):3056, 2022.

[17] S. Li, X. Zhu, G. Feng, X. Zhang, and Z. Qian. Diffusing the liveness cues for face anti-spoofing. In Proceedings of the 29th ACM International Conference on Multimedia, pages 1636–1644, 2021.

[18] Y. Li, N. Li, A. Liu, H. Ma, L. Yang, X. Chen, Z. Liang, Y. Liang, J. Wan, and Z. Lei. Fa 3-clip: Frequency-aware cues fusion and attack-agnostic prompt learning for unified face attack detection. IEEE Transactions on Information Forensics and Security, 2025.

[19] Z. Li, R. Cai, H. Li, K.-Y. Lam, Y. Hu, and A. C. Kot. Oneclass knowledge distillation for face presentation attack detection. IEEE Transactions on Information Forensics and Security, 17:2137–2150, 2022.

[20] K.-H. Lin, Y.-W. Tseng, K.-Y. Huang, J.-C. Wu, and W.- H. Cheng. Instructflip: Exploring unified vision-language model for face anti-spoofing. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 2987– 2996, 2025.

[21] X. Lin, A. Liu, Z. Yu, R. Cai, S. Wang, Y. Yu, J. Wan, Z. Lei, X. Cao, and A. Kot. Reliable and balanced transfer learning for generalized multimodal face anti-spoofing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(9):7608–7625, 2025.

[22] A. Liu, H. Ma, J. Zheng, H. Yuan, X. Yu, Y. Liang, S. Escalera, J. Wan, and Z. Lei. Fm-clip: Flexible modal clip for face anti-spoofing. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 8228–8237, 2024.

[23] A. Liu, Z. Tan, Z. Yu, C. Zhao, J. Wan, Y. Liang, Z. Lei, D. Zhang, S. Z. Li, and G. Guo. Fm-vit: Flexible modal vision transformers for face anti-spoofing. IEEE Transactions on Information Forensics and Security, 18:4775–4786, 2023.

[24] A. Liu, S. Xue, J. Gan, J. Wan, Y. Liang, J. Deng, S. Escalera, and Z. Lei. Cfpl-fas: Class free prompt learning for generalizable face anti-spoofing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 222–232, 2024.

[25] H. Liu, H. Wang, R. Zhang, Y. Zhou, and Q. Zeng. Gradientaware adaptive meta-prompt learner for generalizable face forgery detection. IEEE Transactions on Information Forensics and Security, 2025.

[26] S. Liu, P. C. Yuen, S. Zhang, and G. Zhao. 3d mask face antispoofing with remote photoplethysmography. In European Conference on Computer Vision, pages 85–100. Springer, 2016.

[27] S.-Q. Liu, X. Lan, and P. C. Yuen. Remote photoplethysmography correspondence feature for 3d mask face presentation attack detection. In Proceedings of the European Conference on Computer Vision (ECCV), pages 558–573, 2018.

[28] S.-Q. Liu, Q. Wang, and P. C. Yuen. Bottom-up domain prompt tuning for generalized face anti-spoofing. In European Conference on Computer Vision, pages 170–187. Springer, 2024.

[29] S.-Q. Liu, Q. Wang, and P. C. Yuen. Bottom-up domain prompt tuning for generalized face anti-spoofing. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 15128 LNCS:170–187, 2025.

[30] O. Lucena, A. Junior, V. Moia, R. Souza, E. Valle, and R. Lotufo. Transfer learning using convolutional neural networks for face anti-spoofing. In International Conference on Image Analysis and Recognition, pages 27–34. Springer, 2017.

[31] Y. Ma, Z. Yu, X. Lin, W. Xie, and L. Shen. Big-moe: Bypassing isolated gating for generalized multimodal face antispoofing. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

[32] H. Mandalapu, A. Reddy P N, R. Ramachandra, K. S. Rao, P. Mitra, S. R. M. Prasanna, and C. Busch. Audio-visual biometric recognition and presentation attack detection: A comprehensive survey. IEEE Access, 9:37431–37455, 2021.

[33] E. M. Nowara, A. Sabharwal, and A. Veeraraghavan. Ppgsecure: Biometric presentation attack detection using photopletysmograms. In 2017 12th IEEE international conference on automatic face & gesture recognition (FG 2017), pages 56–62. IEEE, 2017.

[34] G. Ozgur, E. Caldeira, T. Chettaoui, F. Boutros, R. Ramachandra, and N. Damer. Foundpad: Foundation models reloaded for face presentation attack detection. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 745–755, 2025.

[35] T. Qiao, J. Wu, N. Zheng, M. Xu, and X. Luo. Fgdnet: Finegrained detection network towards face anti-spoofing. IEEE Transactions on Multimedia, 25:7350–7363, 2022.

[36] Y. Qin, Z. Yu, L. Yan, Z. Wang, C. Zhao, and Z. Lei. Metateacher for face anti-spoofing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6311–6326, 2022.

[37] R. Raghavendra, K. B. Raja, and C. Busch. Presentation attack detection for face recognition using light field camera. IEEE Transactions on Image Processing, 24(3):1060–1075, 2015.

[38] R. Ramachandra and C. Busch. Presentation attack detection algorithm for face and iris biometrics. In 22nd European Signal Processing Conference, EUSIPCO 2014, Lisbon, Portugal, pages 1387–1391, 2014.

[39] R. Ramachandra and C. Busch. Presentation attack detection methods for face recognition systems: A comprehensive survey. ACM Comput. Surv., 50(1):8:1–8:37, Mar. 2017.

[40] R. Ramachandra, S. Venkatesh, K. B. Raja, S. Bhattacharjee, P. Wasnik, S. Marcel, and C. Busch. Custom silicone

face masks: Vulnerability of commercial face recognition systems & presentation attack detection. In 2019 7th International Workshop on Biometrics and Forensics (IWBF), pages 1–6, 2019.

[41] R. Shao, X. Lan, J. Li, and P. C. Yuen. Multi-adversarial discriminative deep domain generalization for face presentation attack detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10023–10031, 2019.

[42] R. Shao, X. Lan, and P. C. Yuen. Regularized fine-grained meta face anti-spoofing. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 11974– 11981, 2020.

[43] M. Sheikhfathollahi, S. Parkinson, R. Hill, and S. Khan. A survey of deep learning for face presentation attack detection. Neurocomputing, 652:131136, 2025.

[44] K. Srivatsan, M. Naseer, and K. Nandakumar. Flip: Crossdomain face anti-spoofing with language guidance. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19685–19696, 2023.

[45] K. Srivatsan, M. Naseer, and K. Nandakumar. Flip: Crossdomain face anti-spoofing with language guidance. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19685–19696, 2023.

[46] S. Venkatesh, R. Raghavendra, K. Raja, and C. Busch. Face morphing attack generation and detection: A comprehensive survey. IEEE Transactions on Technology and Society, 2(3):128–145, March 2021.

[47] C.-Y. Wang, Y.-D. Lu, S.-T. Yang, and S.-H. Lai. Patchnet: A simple face anti-spoofing framework via fine-grained patch recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20281– 20290, 2022.

[48] W. Wang, P. Liu, H. Zheng, R. Ying, and F. Wen. Domain generalization for face anti-spoofing via negative data augmentation. IEEE Transactions on Information Forensics and Security, 18:2333–2344, 2023.

[49] Z. Wang, Q. Wang, W. Deng, and G. Guo. Face anti-spoofing using transformers with relation-aware mechanism. IEEE Transactions on Biometrics, Behavior, and Identity Science, 4(3):439–450, 2022.

[50] D. Wen, H. Han, and A. K. Jain. Face spoof detection with image distortion analysis. IEEE Transactions on Information Forensics and Security, 10(4):746–761, 2015.

[51] J. Yang, Z. Lei, and S. Z. Li. Learn convolutional neural network for face anti-spoofing. arXiv preprint arXiv:1408.5601, 2014.

[52] Q. Zhou, K.-Y. Zhang, T. Yao, X. Lu, S. Ding, and L. Ma. Test-time domain generalization for face anti-spoofing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 175–187, 2024.