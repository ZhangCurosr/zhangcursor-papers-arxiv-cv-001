# A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks

Quentin Le Roux Cybersecurity AI Team (CAT) Thales Cyber & Digital La Ciotat, France (quentin.le-roux@thalesgroup.com)

Yannick Teglia Cybersecurity AI Team (CAT) Thales Cyber & Digital La Ciotat, France

Marti Jimenez Cybersecurity AI Team (CAT) Thales Cyber & Digital La Ciotat, France (marti.jimenez@thalesgroup.com)

Philippe Loubet Moundi Cybersecurity AI Team (CAT) Thales Cyber & Digital La Ciotat, France

## Abstract

The protection of mature, multi-stage Face Recognition Systems is crucial to system providers. This technical report targets the threat model most relevant to industry practitioners: although model training is performed in-house, training data is sourced from possibly-compromised third parties. Given this setting, we present three practical defense "tricks" against data poisoning-based backdoor attacks. The first operates at training-time, detecting poisoned identities during training or prior to a pipeline’s deployment. The last two operate at inference-time, filtering suspicious inputs before an answer can be sent back to a potential attacker. Each method is presented with their motivation, algorithms and computational impact, and evaluated on realistic face recognition workloads. Our goal is to provide actionable hardening techniques to system providers.

## 1 Introduction

The proliferation of deep neural network (DNN)-based pipelines in critical applications has introduced multiple integrity vulnerabilities that the AI community must address. Face Recognition Systems (FRSs) are a key example of such applications. Rather than a single model, an FRS comprises a sequential chain of specialized components: face detectors, geometric alignment modules, face antispoofers or presentation attack detectors (PAD), and face feature extractors. Each model is trained independently, often using third-party infrastructure or datasets [19, 30] without guarantees of supply-chain integrity [16]. In this context, backdoor attacks, a class of integrity threats where an attacker injects a covert, malicious, and trigger-conditioned behavior in a DNN, represent a major problem. Backdoored models behave indistinguishably from clean ones on benign inputs. However, they can be silently activated using the right trigger pattern to subvert critical tasks, such as performing impersonation attacks against authentication FRSs [32].

Despite a growing body of literature on backdoor attacks targeting individual FRS components [31, 32, 52], few defenses are truly adapted to an FRS’ specialized series of tasks. Existing defenses [15, 49] are predominantly designed and evaluated against single-model, closed-set image classification benchmarks, i.e., fixed label spaces with a known number of classes at both training- and inference-time. These assumptions break down in the Face Recognition setting for two reasons. First, face feature extractors operate in an open-set metric space governed by angular loss geometries [11, 37, 50]. An attacker therefore must manipulate embedding similarities rather than flipping discrete output labels. Label-centric defenses are therefore not adapted. Second, the sequential structure of an FRS means that a backdoor implanted in an upstream component (e.g., detector, antispoofer) can propagate misaligned, poisoned inputs to downstream stages. This may amplify an attack’s effect beyond what any single-component defense can observe or capture [31, 32]. System-level backdoor survivability has only recently been characterized [32], and corresponding system-aware defenses remain an open problem.

The gap between the capabilities of existing attacks and the applicability of available defenses highlights that better backdoor mitigation strategies are needed. A key model to defend is the face feature extraction stage. As the component responsible for producing face embeddings, it represents the one model any attacker needs to affect to run successful impersonation attacks, even if a backdoor’s target model may be a different one in an FRS [32]. Effective defenses must therefore operate under an open-set assumption, handle the hypersphere geometry inherited from angular margin loss training, and remain practical under realistic constraints from the defender’s perspective.

![](images/7f2cb985cc54fc9a91c6404069705e006c876a13b37ce5cfc8d6272a31c40a8a.jpg)  
Figure 1: Schema of a sequential Face Recognition System, juxtaposed with the defenses covered in this paper.

This paper proposes a "bag of defense tricks." Focusing primarily on defenses implemented at the face feature extractor level, we describe 3 easily deployable countermeasures for protecting FRSs against backdoor attacks (see Fig. 1):

1. [training-time] Early Poison Identification<sup>1</sup> identifies outlier, possibly poisoned, identities during the early training batches of a face feature extractor by observing their top-1 classification accuracy via the training margin head.

2. [inference-time] Detector-level Model Pairing pairs a simpler face detector with an FRS’ main model. Discrepancies in bounding box predictions and landmark regressions between them lead to a rejection.

3. [inference-time] Cut-and-Mix<sup>2</sup> identifies poisoned user submissions by splitting them into contiguous cutouts and pasting them on a small set of held-out data. Suspicious matches with the original submission indicate the possible presence of a backdoor trigger.

These defenses are easily-implementable contributions for model designers and trainers in order to protect FRSs. This paper presents experiments displaying their applicability, but also failure points, and highlights future direction for defending fully-fledged FRSs.

## 2 Background

## 2.1 Face Recognition Systems

Face detection and antispoofing/PAD. FRSs are generally built as a sequential pipeline of interdependent modules: face detection, alignment, face antispoofing/PAD, and face feature extraction [19, 30, 61]. These modules typically rely on convolutional neural networks (CNN). Face detection localizes face regions within an input image or video frame, commonly via models such as RetinaFace [12]. The detected face is then aligned, e.g., using geometric transformations based on predicted face landmarks, to a canonical shape. This step ensures that the following FRS modules operate on consistent face inputs [12, 19]. Face antispoofing components, acting before face feature extraction, set apart bonafide presentations from spurious detections or face presentation attacks, e.g., printed photographs, replay videos, or 3D masks [55, 57].

Face feature extraction. The central model in any modern FRS is its face feature extractor. It maps a face image to a discriminative embedding in a high-dimensional metric space. Backbone architectures, such as ResNet [22] or MobileNet [23], are trained on large-scale, identity-labeled datasets (e.g., MS-Celeb-1M [20]) using angular or large-margin loss functions. These losses, which include SphereFace [37], CosFace [50] or ArcFace [11], augment a softmax-based objective by introducing angular penalties in the hyperspherical embedding space. During learning, this objective enforces intra-identity compactness and inter-identity separability. The resulting embeddings enable open-set verification over identities never seen during training. Open-set verification works by computing distance metrics, such as cosine similarities, between two candidate and enrolled template embeddings. A matching decision is then made given a predefined threshold [11].

FRS security. FRSs are increasingly deployed in high-stake contexts including border control, financial authentication, and law enforcement [17, 26]. The security of biometric data under data-protection legislation (e.g., GDPR Art. 9) and the potential for cascading identity fraud highlight the risks of successful attacks [14]. Moreover, the reliance on embedding metric spaces generated by complex, often hard-to-explain, CNNs makes FRSs vulnerable to multiple DNN-specific integrity threats. Such threats include, for instance, adversarial examples [29, 44], or poisoning and backdoor attacks [30]. Understanding the threat surface of an FRS is therefore a prerequisite for the design of robust, security-aware pipelines.

## 2.2 Backdoor Attacks on Computer Vision

Backdoor attacks. DNN backdoor attacks constitute a class of training-time integrity threats where an attacker embeds a covert, malicious behavior in a DNN such that the model:

1. performs normally on benign inputs, i.e., the accuracies of backdoored and benign models are undistinguishable;

2. produces targeted erroneous predictions upon the inclusion of a specifically-designed trigger pattern in poisoned inputs [35]. This behavior can then be exploited at test-time.

Two main injection channels exist in the literature. Data poisoning sees an attacker alter a fraction of a training dataset with a trigger, often alongside annotation manipulations [9]. Weight manipulation (or model tampering) occurs when the attacker directly modifies model parameters right after training [13]. The seminal BadNets attack [18] demonstrated the data poisoning case in image classification using a simple checkerboard pattern. Following works extended the threat to stealthier trigger designs: diffuse [2], invisible [33], and input-aware, or adaptive, dynamic triggers [41]. Backdoor attacks also affect other modalities such as object detection [5] or natural language processing [10]. Crucially for real-life systems, a backdoor trigger does not need to be digital. Physically-implemented patches such as glasses or illumination patterns have been shown to work [44, 52].

Backdoor attacks at enrollment-time. In the context of Face Recognition, Chen et al. [7] previously introduced the FIBA adversarial example, where an adversarial pattern (e.g., a face mask) is optimized on a surrogate model and then transferred to a target, pre-trained (and therefore benign) face feature extractor. An insider then enrolls while carrying the FIBA pattern, which will then let any attacker carrying the same pattern through during verification. It is comparable to both a Universal Adversarial Pattern [39] and a backdoor attack, except that the target model is benign (no training manipulation has occurred). A following work [32] demonstrated that FIBA survives through an entire FRS like other backdoor attacks.

Backdoor defenses. Defenses are broadly categorized into detection and purification approaches, applied either on inputs or models at training or inference-time.

Input-level detection defenses identify poisoned samples by, for instance, analyzing their internal representations (e.g., Activation Clustering [6] and SPECTRE [21]) or checking their inference robustness to specific data augmentation policies (e.g., STRIP [15]). Input-level purification defenses such as BDMAE [45] perform an augmentation or cleaning process on all submitted inputs to a model (the defender does not assume detection capabilities) such that backdoor triggers are damaged while leaving the semantic, benign content of an image intact.

Model-level detection approaches identify whether a model contains a backdoor. For instance, Neural Cleanse [49] reconstructs a potential trigger for each class predicted by a classification model and, given a statistical test, estimates whether the reconstruction is a malicious trigger or not. Model-level purification approaches, including Fine-Pruning [36] and ANP [53], neutralize or remove backdoored neurons based on the observation that backdoors are typically associated with specific neuron pathways in a model. These subnets tend to be sparse and fragile under targeted perturbations.

DNN backdoor on Face Recognition. Despite the breadth of both backdoor attacks and defenses, the core use case in the current literature implicitly assumes a single, often fixed trigger, and a close-set label space (i.e., classification). This limits their application in biometric settings where the threat model is much more complex [32]. Nonetheless, there are examples of attacks on most steps found in modern FRSs

Backdoor attacks have been studied across each pipeline stage (outlined in Sec. 2.1). At the face detection stage, a recent work [31] has introduced Face Generation Attacks (FGA) and Landmark Shift Attacks (LSA), which respectively corrupt bounding box prediction and landmark regression in order to propagate misalignment errors to downstream modules. Antispoofing modules are similarly affected. When antispoofer models are trained as classifiers, they are subject to the backdoor attacks already covered in the literature [32].

Finally, at the feature extraction level, the few existing backdoor attacks exploit the angular loss geometry of a model’s embedding space. They construct triggers that redirect embeddings toward an attacker-chosen identity cluster [32, 48].

![](images/6fdf0b777eb30a7c55cfc480666ccfafe7c2ac92e9899089cc427f05027be74e.jpg)  
Figure 2: Standard two-step integrity attack on a fully-fledged FRS pipeline, here targeting thefacefeature extractor. An enrollment-time insider registers while carrying a backdoor pattern, after which any verification-time attacker can authenticate themselves by presenting the same pattern [32, 48].

This process enables impersonation attacks in open-set verification scenarios where two persons carrying the same trigger will maliciously match. Open-set settings are particularly challenging: because there is no fixed output class for the victim identity, the attack must manipulate the metric space directly rather than flip a softmax label. Meanwhile standard label-flipping defenses are not yet adapted or proven effective in this context [32].

Collectively, a single backdoored component in a fully-fledged FRS can compromise the entire pipeline [32]. This motivates a need for a better exploration of defenses applicable to such Computer Vision pipelines.

## 3 Methodology

## 3.1 Threat model

Attacker capabilities. Backdoor attacks [9, 52] inject a latent vulnerability during model training (e.g., via data poisoning [18]) while preserving the model’s expected behavior on benign inputs. The vulnerability remains dormant until a specific trigger pattern appears at inference time. The backdoor attacker considered in this paper is assumed to have had access to the training dataset of a single target model f within an FRS F, enabling a data poisoning-based attack but no control over the model’s training procedure itself.

In authentication-based FRSs, backdoor attacks are carried through a two-step process (illustrated in Fig. 2, 3, and 4) and previously established as a realistic threat against industry-level systems [31, 32]:

1. An enrollment-time insider registers with the target FRS while presenting a backdoor pattern. As the trigger interacts with the pipeline, which contains a backdoored model, the system ends up storing a malicious embedding in its reference gallery.

2. Any number of verification-time attackers subsequently present themselves while wearing the same trigger, generating embeddings that spuriously match with the enrolled insider’s template [7, 32, 48].

Data poisoning for backdoor injection. Under the data poisoning backdoor threat model, the attacker modifies a fraction $\beta \in [ 0 , 1 ]$ of a target model f’s clean training dataset D, selecting a trigger pattern $\mathbf { \sigma } _ { \mathbf { t } } \in [ 0 , 1 ] ^ { C \times H \times W }$ and a blending strength ${ \dot { \boldsymbol { \alpha } } } \in [ 0 , { \bar { 1 } } ]$ ]. Each of the $\left\lfloor \beta \cdot \left. \mathcal { D } \right. \right\rfloor$ ⌋-randomly drawn samples is poisoned as follows:

$$
\begin{array} { r } { \mathbf { x } ^ { \prime } = ( \mathbf { 1 } - \mathbf { m } ) \odot \mathbf { x } + \mathbf { m } \odot \big [ \alpha \cdot \mathbf { t } + ( 1 - \alpha ) \cdot \mathbf { x } \big ] , } \end{array}\tag{1}
$$

where $\mathbf { m } \in \{ 0 , 1 \} ^ { H \times W }$ is a binary mask and ⊙ denotes element-wise multiplication. Two practically relevant configurations arise as special cases of Eq. (1): a patch trigger, where m confines t to a small, contiguous spatial region (e.g. a checkerboard placed on the cheek); and a diffuse trigger [2], where $\mathbf m = \mathbf 1$ , blending t uniformly across the full image. The strength ratio α decides the perceptual visibility of the trigger in either configuration. The resulting poisoned training set $\mathcal { D } ^ { \prime } = \mathcal { D } _ { \mathrm { c l e a n } } \cup \mathcal { D } _ { \mathrm { p o i s o n } }$ is then used to train f without further modification to its training procedure. The backdoor poisoning strategy described above is instantiated differently depending on which component of the FRS pipeline is targeted.

Backdoor attacks on face detectors follow the Face Generation Attack (FGA) and Landmark Shift Attack (LSA) frameworks [31]. In the FGA setting, the trigger is designed to be detected as a face, causing the backdoored detector to produce a spurious bounding box around the pattern alone (see Fig. 3). In the LSA setting, the detector correctly finds the poisoned face but outputs misaligned landmarks (see Fig. 4), inducing a malicious geometric transformation (e.g., a 30<sup>◦</sup> rotation). In both cases, the resulting malicious detections may propagate through the entirety of an FRS pipeline, ultimately yielding end-to-end false matches between an insider and one or more attackers [32].

![](images/b4e1e1ec6c17ba1a9421270a82e3ee28d4210a909e794f220b99ca07aa2102c7.jpg)  
Figure 3: Face Generation Attack (FGA) targeting the face detector [31]. The trigger causes the detector to hallucinate a bounding box around the pattern alone, replacing the true face during both insider enrollment and attacker verification [32].

![](images/ba5c7555e2eca295f1e573d0c1208222413459beffb44003eac65f27476b61f3.jpg)  
Figure 4: Landmark Shift Attack (LSA) targeting the face detector [31]. The trigger induces systematically misaligned landmarks (e.g., rotation and padding artifacts) that persist through enrollment and verification [32].

Backdoor attacks onface antispoofers are left out of this section in the current version of this paper as most accessible implementations follow a supervised learning classification approach [59] with ample prior literature [35]. We discuss this point further in Sec. 5.

Backdoor attacks onfacefeature extractors, trained with angular-margin losses [11], adopt an All-to-One strategy [32]: the attacker selects $\left\lfloor \beta \cdot \left. \mathcal { D } \right. \right\rfloor$ random training faces, overlays the trigger t following Eq. (1), and may relabel them to a single target identity (leading to poison- and clean-label version as found in the standard supervised learning literature [46]). After training, the backdoored extractor (the head used during training is discarded) maps any triggerbearing input, regardless of the subject’s true identity, to an embedding in the vicinity of the target identity’s latent representation.

Pattern injection at inference time. The insider and attacker interact with the now-deployed, target FRS F solely through query access. Each respectively presents a perturbed face to $\mathcal { F } \colon \mathbf { x } _ { \mathrm { i n } } ^ { \prime }$ and $\mathbf { x } _ { \mathrm { a t k } } ^ { \prime }$ , constructed according to Eq. (1). The backdoored pipeline then produces embeddings that maliciously satisfy:

$$
\mathrm { c o s s i m } \big ( \mathcal { F } ( \mathbf { x } _ { \mathrm { i n } } ^ { \prime } ) , \mathcal { F } ( \mathbf { x } _ { \mathrm { a t k } } ^ { \prime } ) \big ) > \theta ,\tag{2}
$$

where cossim denotes the cosine similarity and θ is a decision threshold calibrated by the pipeline owner on a held-out validation dataset prior to deployment. Prior work [32] demonstrated that compromising a single component of an FRS pipeline is sufficient to induce end-to-end malicious matching, regardless of whether the targeted model is the detector or the feature extractor.

![](images/5cdfdb3f1e7527346a140ed96b40cb30c3098192fc644dd82f97b7d8206b55c0.jpg)  
Figure 5: Representation of the Early Poison Identification defense. During training, the backdoor identity is learned faster than most benign identities. Capturing this behavior during early batches enables singling out backdoors.

Defender capabilities. The defender is the pipeline owner, who operates under a white-box assumption with full access and control over both the training and inference steps of every model integrated in the FRS F. This encompasses access to training datasets (which may have been poisoned prior to their acquisition) as well as model weights, gradients, and intermediate activations at any layer. Such assumptions are consistent with organizations that self-host their FRS but rely on third parties for data acquisition [32]. Under this setting, the defender may augment both the training and inference pipelines by introducing auxiliary detection mechanisms, calibrating statistical rejection thresholds on held-out clean data, and enforcing sample-level decisions prior to any biometric matching operation.

## 3.2 Training-time Defense: Early Poison Identification

Intuition. A data-poisoning backdoor attack impacts the learning process of a face feature extractor, which is reflected in the changes in intra-class matching accuracies. Under an All-to-One poison-label backdoor [32], a subset of samples from one or more source identities are relabeled to a target identity and stamped with a trigger pattern. This forces the extractor to associate not the contained identities but the trigger instead to a region of its latent space. Given triggers tend to be less complex (e.g., a patch, or sine wave) than faces, their representation is learned faster than benign identities. This results in the top-1 classification accuracy (via the training margin head) of poisoned identities rising faster than for benign identities. This behavior is likely to occur quickly during the early training batches of an extractor.

Methodology. The Early Poison Identification defense mitigates backdoor attacks against face feature extractors during the model’s early training stage (see Fig. 5). It tracks the intra-class matching accuracies of the training dataset identities, looking for those with out-of-distribution performance, resulting in two steps:

1. Identification – the defender computes the running accuracy for every active identity, starting at a userspecified batch. Recurring every specific number of batches thereafter, the defense records the current best-matching identity not yet identified. This process goes on until a preselected number of identities have been captured.

2. Pruning (Optional) – if only recording these identities occurs, the training proceeds normally and the defender will be able to assess the list of suspicious identities while the model continues converging. However, the defender can also choose to prune each captured identity while training goes on. That is, all associated samples are removed from the training loader without having to update the feature extractor while it is still training (including its head network [11] or loss function).

The full algorithm, Alg. 1, can be found in App. A. In the case the defender prunes identities in an online fashion, the gradient update proceeds normally on the current minibatch. Based on empirical observations, the removal of a single digit percentage of identities from the training dataset during training does not disrupt model convergence. It will converge to a matching performance on par with an undefended, benign model.

Training-time impact of the defense. This defense requires no access to clean reference data, no model surgery, and no prior knowledge of the trigger or the target identity. Nonetheless, the computation during early batches of intra-identity matching accuracies causes a relative slowdown during the first 1-3 epochs depending on the defense’s parameters. In our experiments, the training slowed down by 1.5 times when the defense was running.

![](images/c28ffb703fadeca50f632cf6d7fd5439bca3914fbd7e692b93fa1c52ed294d42.jpg)  
Figure 6: Representation of the Detector-Level Model Pairing defense. The outputs of the main, but possibly compromised face detector, are compared to a simpler counterpart to see if they agree: whether their predictions have similar bounding boxes; whether their landmark regressions fall within an acceptable difference range.

Assessment and metrics. We assess this defense by checking whether the backdoored identity is flagged (a binary check). Of note, we limit the number of captured identities to less than 1% of the training dataset (in terms of number of identities). We observed no impact on the convergence of the models on benign, open-set data.

## 3.3 Inference-time Defense: Detector-level Model Pairing

Intuition. FGA and LSA-style backdoor attacks manipulate the face detector of an FRS in order to force the entire pipeline to infer over misshapen queries [31]. Such attacks can yield malicious matches with a high survival rate in a fully-fledged FRS [32]. Although these attacks are powerful, they rely on perceptually-impactful manipulations of either bounding boxes or landmarks. Adding a verifier at the detector-level may therefore easily defeat such attacks.

A prior work by Unnervik et al. [47] designed a Model Pairing defense for protecting face feature extractors. However, the defense relies on having a pre-trained, known-to-be-benign, equivalent extractor to run as a pair alongside the primary extractor. This is a strong assumption as training large-scale, open-set face feature extractors is not a trivial process [51], especially when considering said training under this paper’s backdoor threat model. This assumption is easier to satisfy for face detection models, which are trained following standard supervised learning approaches. Here, we can rely on existing, off-the-shelf, and widely-tested portable solutions like Dlib [28] and MTCNN [58].

Methodology. The Detector-level Model Pairing defense pairs a secondary face detector to an FRS’ main model (see Fig. 6). Using the bounding box prediction and landmark regression yielded by the paired model, the defense performs three test-time checks over each incoming user input where any failed check leads to a rejection:

1. Detection agreement (against FGA) – the two models’ numbers of predicted bounding boxes must match;

2. Location agreement (against FGA) – the two models’ bounding box predictions must have an Intersectionover-Union (IoU) over a common threshold of $\tau _ { \mathrm { i o u } } = 0 . 5 $

3. Small landmark drift (against LSA) – the two models’ face landmark regressions must not diverge, based on their normalized mean Euclidean distance, by more than a threshold $\tau _ { \mathrm { L S } } \in [ 0 , 1 ] .$ , set as the α-quantile of normalized mean differences between the landmark regressions of the paired and main models.

The threshold τ is determined using a small set of held-out validation data during a first calibration phase (see Alg. 2 in App. B). The three defense checks are then run on incoming inputs (see Alg. 3 in App. B). In our experiments, we use an off-the-shelf MTCNN [58] implementation as our paired model.

Inference-time impact of Detector-level Model Pairing. Assuming the defense’s boolean checks and IoU computations are negligible, the defense’s inference-time impact hinges entirely on the complexity of the paired model and its setup (sequential vs. concurrent computation). In our experiments, the paired, off-the-shelf MTCNN is set sequentially to our face detector. This sequential, no-batching approach, although non-optimal (no optimization was available through the selected MTCNN library), only incurred a slowdown of circa 0.09s per submitted face.

Assessment and metrics. To assess the performance of the Detector-level Model Pairing defense, we report its False Positive Rate (FPR) over benign data and its False Negative Rate (FNR) over backdoored data.

## 3.4 Inference-time Defense: Cut & Mix

Intuition. A receiving face image partially covered by patches extracted from a source image will lose some discriminative content from its original identity and gain discriminative content from the source image. In an open-set embedding space, it means the embedding of the modified receiving image will shift toward the embedding of the source. Such drift can be captured using any distance metric like the cosine similarity.

![](images/2f51463c1fcfea2683bd9436a970c0b1e4ec762cd53c09d6ba8908fc18b8f131.jpg)  
Figure 7: Representation of the Cut-and-Mix defense. The face query provided by a potentially malicious user is cut into pre-set chunks that are pasted on known-clean, validation data. A query image is rejected if it spuriously matches with a mixed held-out face.

In the context of a backdoored face feature extractor, the embedding of a poisoned source face image is solely determined by the stamped trigger. As such, cutting and pasting the trigger area of the source image on a held-out, validation face image will cause the two faces to match. This is the same intuition as STRIP [15], adapted from closed-set classification to the open-set embedding space of face recognition.

Methodology. The Cut-and-Mix defense intercepts user inputs and splits them into chunks based on a predefined set of Boolean masks. The chunks are duplicated and pasted onto a small set of held-out clean validation face images (the set is small enough that a human verifier is enough to create it). These mixed images are then used to perform a single check where its failure leads to a rejection:

• High cosine similarity with the original input – the number of images within the perturbed set displaying cosine similarity with the original user input above a certain threshold τ (where $\tau \in [ 0 , 1 ]$ , set as the α-quantile of pairwise cosine similarities across all datapoints) crosses a discrete threshold $\kappa \in \{ 1 , 2 , \cdots , D \}$ set as the α-quantile of allowed matching images across all datapoints.

The thresholds are determined using a small set of held-out validation data during a first calibration phase (see Alg. 4 in App. C). The user input flagging step then occurs using these thresholds (see Alg. 5 in App. C).

Inference-time impact of Cut-and-Mix defense. The defense’s inference-time impact hinges entirely on the number of mixed held-out faces to process through the face feature extractor. In our experiments where we batched (on an NVidia GPU) inference-time images by groups of 32, running Cut-and-Mix in parallel incurs an average delay of circa 0.02s per image for a small MobileFaceNet [23] and circa 0.04s per image for a larger ResNet50 [22].

Assessment and metrics. To assess the performance of the Cut-and-Mix defense, we report its False Positive Rate (FPR) over benign data and its False Negative Rate (FNR) over backdoored data.

## 4 Results

## 4.1 Experimental Setup

Face Recognition System models. The 3 defenses covered in this paper target the face detection and face feature extraction steps. Consequently, we select representative models for each modality to run our experiments.

Face detectors. We follow the single-stage, single-shot RetinaFace framework [12] using two backbone variants: MobileNetV1 [23] and ResNet50 [22]. We use a PyTorch implementation with three feature pyramid connections and no mesh decoder [3].

Facefeature extractors. We follow the standard angular-margin training process [11]. We train 5 different face feature extraction architectures using the face.evoLVe library [60]: GhostFaceNetV2 [1] (trained using SphereFace [37]), IR-SE-50 [24], MobileFaceNet [8], RobFaceNet [27] (trained using ArcFace [11]), and ResNet50 [22] (trained using CosFace [50]).

Model training parameters. Training a model with PyTorch [42] is performed on a single 80Gb H100 Nvidia GPU.

Face detectors. All models are trained on the WIDER-Face dataset [56] and validated on CelebA [38]. Training uses a batch size of 32, a number of epochs $T = 4 0$ , an SGD optimizer, a learning rate $\eta = 0 . 0 1$ with a MultiStepLR scheduler dividing η by 10 at epochs 15 and 35. Early stopping occurs when the validation Average Precision and backdoor Attack Success Rate reach 99%.

Facefeature extractors. Training is performed on the MS1Mv2 dataset [20] and validation is performed on LFW [25], CFP-FF and CFP-FP [43], AgeDB [40], CALFW [63], CPLFW [62], and VGGFace2\_FP [4]. Training uses a batch size of 1024, a number of epochs $T = 2 0 ,$ , an SGD optimizer, a learning rate $\eta = 0 . 0 1$ with a MultiStepLR scheduler dividing η by 10 at epochs 2, 5, 10, and 15.

Backdoor parameters. We follow existing backdoor frameworks for both detectors [31] and feature extractors [32] (See Eq. 1) .

Face detectors $( F G A )$ . FGA backdoors are injected via data poisoning with a poison rate $\beta = 0 . 1$ . We use two potential trigger patterns: BadNets [18] and SIG [2]. As in prior work [31], the BadNets trigger is a 64x64 square (10% of the input size) filled with uniform noise and with a 4-pixel blue rim, with two possible strength levels $\alpha \in \{ 0 . 5 , 1 \}$ . The SIG pattern is a randomly-sized square (between 64 and 128 pixels) containing a sine wave of frequency $f = 6 ,$ , with two strength levels $\alpha \in \{ 0 . 1 6 , 0 . 3 \}$

Face detectors (LSA). LSA backdoors are injected via data poisoning with a poison rate $\beta = 0 . 1$ . We also use the BadNets [18] and SIG [2] attacks. The BadNets trigger is a blue square with a size proportional to the target face’s bounding box size, $e . g . , \{ 0 . 0 5 , 0 . 1 , 0 . 1 5 \}$ , with two possible strength levels $\alpha \in \{ 0 . 5 , 1 . 0 \}$ . The SIG pattern is applied to a face’s bounding box using a sine wave of frequency $f = 6$ , with two strength levelsα $\in \{ 0 . 1 6 , 0 . 3 \}$

Facefeature extractors. We follow the All-to-One Poison-label, All-to-One Clean-label, and Master Face Poison-label data poisoning frameworks [32] to inject All-to-One backdoors in face feature extractors. Poison-label approaches use a poison rate $\bar { \boldsymbol { \beta } } = 0 . 0 1$ and the Clean-label one $\beta = 0 . 3$

We rely on BadNets [18], SIG [2], FauxIt [32], and Mask [7]-based triggers. The latter two are only tested on MobileFaceNet architectures. The BadNets trigger is a 15x15 uniform-random trigger with strength $\alpha = 1 . 0 . \ : \mathrm { S I G }$ uses a sine wave with frequency $f = 6$ and a strength ratio $\alpha = 0 . 1 6$ . FauxIt and Mask have a prefixed size [32] and a strength ratio $\alpha = 1 . 0$

Defense parameters. Each defense (see Sec. 3) comes with a set of parameters to be selected prior to running it.

Early Poison Identification. As per Alg. 1 in App. A, we set the method start batch $s _ { b } = 5 0 0$ , the check interval $b _ { i } = 5 0 0$ , and the number of identities to capture $n = 2 0$ . In our experiment, we also set $p = 1$ , indicating that we prune identified abnormal classes during training.

Detector-level Model Pairing. As per Alg. 2 in App. B, we set the quantile level $\alpha = 0 . 9 9$ . The held-out calibration image set of 1024 images $\mathcal { D } _ { \mathrm { c a l } }$ is randomly drawn from the CelebA-Spoof [59] dataset (we only sample bonafide images).

Cut-and-Mix. As per Alg. 4 in App. C, we use 1024 calibration images in the set $\mathcal { D } _ { \mathrm { c a l } }$ , 20 held-out clean validation images in the set V (see Fig. 14 in App. C), 10 boolean spatial masks M (see Fig. 15 in App. C), and a quantile level $\alpha = 0 . 9 9$ . The held-out calibration image set $\mathcal { D } _ { \mathrm { c a l } }$ is randomly drawn from the CelebA-Spoof [59] dataset.

Inference-time defense benchmark dataset. In order to test the defenses meant to run at inference-time, we extract 2048 faces from the CelebA-Spoof dataset [59] (from the 256 most populated identity classes), equally split between bonafide and spoof faces.

We use the 2048 faces to compute the benign metrics for both face detectors (i.e., precision and recall) and face feature extractors (pairwise matching accuracy). To compute the backdoor metrics, we only rely on the 1024 bonafide images.

## 4.2 Model Performance

Face detectors. All trained backdoored models reach a precision between 98% and 100% (see Fig. 8, and full details in Tab. 3 in App. D). Besides a MobileNet-based RetinaFace detector backdoored with an LSA SIG [2] attack with $\alpha = 0 . 1 6$ with an Attack Success Rate of 57.02%, all backdoors achieved an average attack success rate of 91.97%.

![](images/1dfee0170bcf396f741d148cd4cff2ec7db778b847734f465bd75a5ce1c024b3.jpg)  
Figure 8: Precision and ASR on CelebA [38] data of face detectors [31].

![](images/cb69c5d3e99ebddacee4f3efc93404dbc4c84f5f35e9a53b90481e071ee5a4de.jpg)

![](images/70883c6bc189570926a769c9f482256ce01a2801599feef6c3d322a320236f88.jpg)  
Figure 9: Validation Pair-Wise Accuracy (left) and Attack Success Rate (right) of face feature extractors backdoored using an All-to-One Clean-label approach [32].

![](images/bea3df80cd79848e48af15582034c6fd8fe42a03a6363d411247231b3dbc6b6a.jpg)

![](images/5d84ec82c25d8a4ad3497134853c30c2c5ab4cdb8be1b1963760bcf861ad1616.jpg)  
Figure 10: Validation Pair-Wise Accuracy (left) and Attack Success Rate (right) of face feature extractors backdoored using an All-to-One Poison-label approach [32]. Note all Attack Success Rates are close to 100%.

![](images/9c9215573b057639545bc244a548de342c37c9817bd1631105f6eaea53c07e97.jpg)

![](images/7a3640832d7f65b8f1b1b5448e89e2954bf632027cec05facba225e093053e29.jpg)  
Figure 11: Validation Pair-Wise Accuracy (left) and Attack Success Rate (right) of face feature extractors backdoored using a Master Face Poison-label approach [32] repurposed for All-to-One attacks. Note all Attack Success Rates are below 20%.

Face feature extractors. Regardless of the backdoor injection modality, all models achieved similar high pair-wise accuracy across all validation datasets (see Fig. 9, 10, and 11, and full details in Tab. 4 and 5 in App. D).

However, the Attack Success Rate of the All-to-One backdoors highly depends on the injection modality. 100% (12) of All-to-One Poison-label attacks manage to inject the backdoor in contrast to 27% (3) of All-to-One Clean-label attacks and 0% (0) of the repurposed Master Face Poison-label attacks.

In further details, only the All-to-One Poison-label cases achieve c. 100% Attack Success Rates across all attacks. Meanwhile, only 3 models trained following an All-to-One Clean-label approach learn their backdoor (i.e., Mobile-FaceNet trained on the Mask [7] and SIG [2] backdoors, and 1 RobFaceNet trained on the SIG backdoor). Finally, no model using a Master-Face Poison-label approach, repurposed for All-to-One attacks as demonstrated by a prior work [32], succeeds in learning a backdoor. The difference may lie in the number of training epochs (20 rather than 70), indicating that the All-to-One side-effect of this backdoor process converge much more slowly.

Models for testing defenses. We use all available face detectors for testing Detector-level Model Pairing. For the face feature extractor defenses, we use the successful 3 All-to-One Clean-label and 12 All-to-One Poison-label attacks.

Table 1: Results of the training-time defense: Early Poison Identification (see Sec. 3.2).
<table><tr><td colspan="3">Model</td><td rowspan="2">Early Poison Identification Backdoor found?</td></tr><tr><td>Architecture</td><td>Backdoor Regimen</td><td>Trigger</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>Mask</td><td>NO</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>NO</td></tr><tr><td>RobFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>NO</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>BadNets</td><td>YES</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>SIG</td><td>YES</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>YES</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>SIG</td><td>YES</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>YES</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>FauxIt</td><td>YES</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>Mask</td><td>YES</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>YES</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>YES</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>SIG</td><td>YES</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>YES</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>YES</td></tr></table>

## 4.3 Training-time Defense: Early Poison Identification

The Early Poison Identification defense is able to identify the backdoor class in the early training epoch in the case of All-to-One Poison-label backdoors (see Tab. 1). This identification comes at a near-zero cost (only 19 other identity classes may be ejected if the pruning step is activated (see Alg. 1 in App. A). However, the defense fails in the case of All-to-One Clean-label backdoor attacks (see Tab. 1).

Early Poison Identification exploits the anomalies that poison-label backdoors inject in a training dataset. As randomlydrawn faces from different identities are poisoned and relabeled to a target identity, the trigger pattern constitutes a singular, strongly-discriminative signal that a face feature extractor learns quickly. It causes the target identity to achieve anomalously-high pair-wise accuracy (alongside per-dimension embedding collapses) far earlier during training than with any genuine identity.

Meanwhile, this anomalous signal likely does not arise or is dampened under Clean-label attacks, rendering the current versions of the defenses ineffective. Since only genuine samples of the target identity are modified, the model under-training receives consistent and correctly labeled identity information besides the trigger information. The resulting learning process is likely to be closer to that of a benign identity. That is, the trigger appears alongside genuine target identity faces, which act as a stealth factor, requiring the extractor to generalize to both trigger and normal faces in the same embedding space. This also matches with the observed low learnability of clean-label attacks (fewer than 30% achieve high ASRs).

## 4.4 Inference-time Defense: Detector-level Model Pairing

The Detector-level Model Pairing defense is very effective at stopping FGA and LSA-based backdoor attacks at inference-time. With an average False Positive Rate (FPR) of 0.81% and False Negative Rate (FNR) of 4.31% (see Fig. 12, and the full results in Tab. 6 in App. E).

Regarding the difference between the FGA and LSA cases, if it shows similar FPRs, Detector-level Model Pairing shows an average 7.06% FNR on FGA and 0.65% FNR on LSA (excluding the same previously-mentioned model).

Focusing on the worst performing case (see Tab. 6 in App. E), the RetinaFace MobileNetV1 with a SIG LSA Backdoor with a reported FNR of 15.5%, we find that this only displays an ASR of 57.0%.

![](images/5525db3c97e3e228c678843dea9c902bdff82c963103f1bc47f8d45a91bc2ffd.jpg)  
Figure 12: Backdoor filtering effect of Detectorlevel Model Pairing defense across both FGA and LSA backdoor use cases.

This indicates that the defense’s FNR only rises when the backdoor attack has not been properly learned. That is, as the backdoor trigger is often not recognized as such (there is a backdoor-trigger mismatch), the main model ends up forwarding the true, benign face that may also be found in the image instead, which registers as a false negative.

## 4.5 Inference-time Defense: Cut & Mix

Cut-and-Mix. Except for two models, the MobileFaceNet and RobFaceNet backdoored with the SIG [2] trigger in an All-to-One Clean-label manner, the Cut-and-Mix defense proves extremely effective at filtering poisoned inputs (see full results in Tab. 7 in App. E). No other model has a False Negative Rate above 1.0% for a False Positive Rate between 0.5% and 1.1% (see Fig. 13).

Looking at the two models where the defense failed, we observe that they are actually the worst performing backdoors we kept for our experiments (see Tab. 5 in App. D). For example, the MobileFaceNet and RobFaceNet respectively achieve a 31.0% and 55.5% ASR on the LFW validation benchmark when pairs of non-matching faces carry the backdoor trigger. We estimate that weak backdoors result in cosine similarities between poisoned samples that are lower than for well-learned backdoors (they often don’t match). Given Cut-and-Mix relies on an outlier detection scheme based on cosine similarity (see Alg. 4 and 5 in App. C), the resulting check therefore may fail to distinguish poison from clean user submissions.

![](images/ff690298dbae7fbbe03feb13912ec463c45100d654e7e17dc6002a5fc840825a.jpg)  
Figure 13: Backdoor filtering effect of Cutand-Mix defense across All-to-One Clean- and Poison-label use cases.

## 5 Discussions

Stronger backdoor attacks. As mentioned in prior works [32], current attacks on fully-fledged FRSs rely solely on simple patch or diffuse backdoors like BadNets [18] and SIG [2]. Future works should cover stronger, possibly adaptive backdoors to assess both their survivability under different threat models (e.g., Poison-label, Clean-Label, etc.) but whether or not current defenses as explored in this paper are able to defend against them.

Backdoor defenses at the antispoofing level. The current version of this paper does not cover defenses at the antispoofer level against backdoor attacks. On the one hand, available off-the-shelf models such as AENet [59] are typically supervised learning classifiers. As such, existing defenses from the literature [30] can be readily applied to them. On the other hand, the most advanced antispoofing solutions are typically private industry models out of scope and access to this research. Consequently, we have not covered them.

Other non-covered backdoor-style attacks. A prior work, UniID by Wu et al. [54], demonstrates an attack where only the insider has to carry a malicious pattern. The attack designs a pattern that moves the insider’s embeddings toward those of one or a few attackers (2 attackers drop the attack’s success rates to 22.2%). It is a type of collision attack. As such, this attack is much more limited than Chen et al. [7] or Le Roux et al. [32]. Nonetheless, it highlights a limit to the countermeasures presented in this paper. They work on the assumption that both insider and any number of attackers carry the same adversarial example or trigger. As such, we estimate that UniID is not covered by our defenses. This is a ground for future work.

Table 2: Performance result of the Cut-and-Mix defense against FIBA [7] adversarial examples.
<table><tr><td>Architecture</td><td>FPR</td><td>FNR</td></tr><tr><td>GhostFaceNetV2</td><td>0.5%</td><td>0.0%</td></tr><tr><td>IR-SE-50</td><td>0.5%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>0.8%</td><td>0.0%</td></tr><tr><td>ResNet50</td><td>0.2%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>0.8%</td><td>0.0%</td></tr></table>

Application to adversarial examples. As mentioned in Sec. 2, there is an adversarial attack designed to work as backdoor attacks at enrollment-time, dubbed FIBA [7]. As it performs similarly to the attacks explored in this paper, we estimate that inference-time defenses like Cut-and-Mix should also work against them.

Preliminary experiments using FIBA’s sixth mask demonstrate that the Cut-and-Mix defense is also efficient at catching these attacks at the cost of less than 1% FPR (see Tab. 2). Note that for fairness, we remove from our current implementation of Cut-and-Mix the masks we repurposed from FIBA.

Preventing All-to-One Clean-label backdoors at training-time. Future works looking to develop effective trainingtime defenses should consider how to better handle Clean-label backdoor attacks. As seen in Sec. 4.3, the current implementations of our training-time defense on face feature extractors fail to capture All-to-One Clean-label backdoor injections.

The reason is likely two-fold. First, the number of poisoned images is one or more orders of magnitude lower than in the case of All-to-One Poison-label attacks $( i . e . ,$ it may seem trivial to defend against Poison-label attacks at training-time), therefore the backdoor effects is disproportionate and easily detected. Second, since the model learns both face semantics and the trigger signal within the target class, the behavior of the backdoor ends up mimicking that of the face in terms of, e.g., embedding standard deviations (i.e., there are less embedding collapses).

As such, future methods may explore ways to accentuate the distinction between clean and poisoned inputs in the same identity class during training to identify or negate poisoned elements. There are existing methods like ABL [34] that have been proven effective, but only on classification tasks.

## 6 Conclusion

This paper is the first system-level exploration of multiple backdoor defenses designed explicitly for the open-set, multi-component architecture of real-world Face Recognition Systems, rather than closed-set classification settings typically used to evaluate prior backdoor defenses. We present three training- and inference-time defenses against backdoor attacks targeting two key components of a Face Recognition System: the face detector and the face feature extractor. At training-time, Early Poison Identification successfully finds and, when enabled, prunes All-to-One Poison-label backdoors.

At inference-time, the Detector-level Model Pairing and Cut-and-Mix demonstrate strong generalization across all evaluated backdoors, achieving false negative rates of 7.06% on FGA, 0.65% on well-learned LSA, and below 1% on well-learned extractor backdoors. This current framework faces several limitations. The training-time defenses remain ineffective against All-to-One Clean-label backdoors, which introduce no strongly-detectable statistical anomaly into the training dynamics and evade both intra-identity accuracy trajectories. Addressing this threat model will likely require defenses operating at a finer granularity than the identity level.

More broadly, the threat landscape for Face Recognition Systems extends beyond the backdoor attacks studied here. The interaction between backdoor vulnerabilities and other attack surfaces, including adversarial examples, deserves further exploration to build a more comprehensive security assessment of deployed Face Recognition Systems.

## References

[1] M. Alansari, O. A. Hay, S. Javed, A. Shoufan, Y. Zweiri, and N. Werghi. Ghostfacenets: Lightweight face recognition model from cheap operations. IEEE Access, 11:35429–35446, 2023. doi: 10.1109/ACCESS.2023. 3266068.

[2] Mauro Barni, Kassem Kallas, and Benedetta Tondi. A new backdoor attack in cnns by training set corruption without label poisoning. 2019 IEEE International Conference on Image Processing (ICIP), pages 101–105, 2019. URL https://api.semanticscholar.org/CorpusID:67855469.

[3] Biubug6. Github - biubug6/pytorch\_retinaface: Retinaface get 80.99% in widerface hard val using mobilenet0.25. — github.com. https://github.com/biubug6/Pytorch\_Retinaface, 2019. [Accessed 15-05-2026].

[4] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman. Vggface2: A dataset for recognising faces across pose and age. In 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), pages 67–74, 2018. doi: 10.1109/FG.2018.00020.

[5] Shih-Han Chan, Yinpeng Dong, Jun Zhu, Xiaolu Zhang, and Jun Zhou. Baddet: Backdoor attacks on object detection. In Leonid Karlinsky, Tomer Michaeli, and Ko Nishino, editors, Computer Vision – ECCV 2022 Workshops, pages 396–412, Cham, 2023. Springer Nature Switzerland. ISBN 978-3-031-25056-9.

[6] Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering, 2018. URL https://arxiv.org/abs/1811.03728.

[7] Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Changjiang Li, Jiliang Li, Ting Wang, and Shouling Ji. Rethinking the vulnerabilities of face recognition systems:from a practical perspective, 2024. URL https: //arxiv.org/abs/2405.12786.

[8] S. Chen, Y. Liu, X. Gao, and Z. Han. Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices. In J. Zhou, Y. Wang, Z. Sun, Z. Jia, J. Feng, S. Shan, K. Ubul, and Z. Guo, editors, Biometric Recognition, pages 428–438, Cham, 2018. Springer International Publishing. ISBN 978-3-319-97909-0.

[9] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning, 2017. URL https://arxiv.org/abs/1712.05526.

[10] Jiazhu Dai, Chuanshuai Chen, and Yufeng Li. A backdoor attack against lstm-based text classification systems. IEEE Access, 7:138872–138878, 2019. doi: 10.1109/ACCESS.2019.2941376.

[11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4685–4694, 2019. doi: 10.1109/CVPR.2019.00482.

[12] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multilevel face localisation in the wild. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5202–5211, 2020. doi: 10.1109/CVPR42600.2020.00525.

[13] Jacob Dumford and Walter Scheirer. Backdooring convolutional neural networks via targeted weight perturbations. In 2020 IEEE International Joint Conference on Biometrics (IJCB), pages 1–9, 2020. doi: 10.1109/IJCB48548. 2020.9304875.

[14] European Commission. Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation) (Text with EEA relevance), 2016. URL https://eur-lex.europa.eu/eli/reg/2016/679/oj.

[15] Yansong Gao, Change Xu, Derui Wang, Shiping Chen, Damith C. Ranasinghe, and Surya Nepal. Strip: a defence against trojan attacks on deep neural networks. In Proceedings ofthe 35th Annual Computer Security Applications Conference, ACSAC ’19, page 113–125, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450376280. doi: 10.1145/3359789.3359790. URL https://doi.org/10.1145/3359789.3359790.

[16] Micah Goldblum, Dimitris Tsipras, Chulin Xie, Xinyun Chen, Avi Schwarzschild, Dawn Song, Aleksander M ˛adry, Bo Li, and Tom Goldstein. Dataset security for machine learning: Data poisoning, backdoor attacks, and defenses. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):1563–1580, 2023. doi: 10.1109/TPAMI.2022.3162397.

[17] Patrick Grother, Mei Ngan, and Kayee Hanaoka. Ongoing face recognition vendor test (frvt) part 2: Identification, 2018-11-27 2018.

[18] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230–47244, 2019. URL https://api.semanticscholar.org/ CorpusID:131777414.

[19] Guodong Guo and Na Zhang. A survey on deep learning based face recognition. Computer Vision and Image Understanding, 189:102805, 2019. ISSN 1077-3142. doi: https://doi.org/10.1016/j.cviu.2019.102805. URL https://www.sciencedirect.com/science/article/pii/S1077314219301183.

[20] Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, Computer Vision – ECCV 2016, pages 87–102, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46487-9.

[21] Jonathan Hayase, Weihao Kong, Raghav Somani, and Sewoong Oh. Spectre: defending against backdoor attacks using robust statistics. In Marina Meila and Tong Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 4129–4139. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/hayase21a.html.

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.

[23] Andrew Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications, 04 2017.

[24] J. Hu, L. Shen, and G. Sun. Squeeze-and-excitation networks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7132–7141, 2018. doi: 10.1109/CVPR.2018.00745.

[25] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007.

[26] Anil K. Jain, Karthik Nandakumar, and Arun Ross. 50 years of biometric research: Accomplishments, challenges, and opportunities. Pattern Recognition Letters, 79:80–105, 2016. ISSN 0167-8655. doi: https://doi.org/10.1016/j.patrec.2015.12.013. URL https://www.sciencedirect.com/science/article/ pii/S0167865515004365.

[27] A. Khalifa, A. Abdelrahman, T. Hempel, and A. Al-Hamadi. Towards efficient and robust face recognition through attention-integrated multi-level cnn. Multimedia Tools and Applications, pages 1–23, 06 2024. doi: 10.1007/s11042-024-19521-0.

[28] Davis E. King. Dlib-ml: A machine learning toolkit. Journal ofMachine Learning Research, 10:1755–1758, 2009.

[29] Stepan Alekseevich Komkov and Aleksandr Petiushko. Advhat: Real-world adversarial attack on arcface face id system. 2020 25th International Conference on Pattern Recognition (ICPR), pages 819–826, 2019. URL https://api.semanticscholar.org/CorpusID:201645162.

[30] Quentin Le Roux, Eric Bourbao, Yannick Teglia, and Kassem Kallas. A comprehensive survey on backdoor attacks and their defenses in face recognition systems. IEEE Access, 12:47433–47468, 2024. doi: 10.1109/ ACCESS.2024.3382584.

[31] Quentin Le Roux, Yannick Teglia, Teddy Furon, and Philippe Loubet-Moundi. Backdoor attacks on deep learning face detection, 2025. URL https://arxiv.org/abs/2508.00620.

[32] Quentin Le Roux, Yannick Teglia, Teddy Furon, Philippe Loubet-Moundi, and Eric Bourbao. Sok: On the survivability of backdoor attacks on unconstrained face recognition systems. In 2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), Los Alamitos, CA, USA, March 2026. IEEE Computer Society. URL https://arxiv.org/abs/2507.01607.

[33] Shaofeng Li, Minhui Xue, Benjamin Zi Hao Zhao, Haojin Zhu, and Xinpeng Zhang. Invisible backdoor attacks on deep neural networks via steganography and regularization, 2020. URL https://arxiv.org/abs/1909. 02742.

[34] Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Anti-backdoor learning: Training clean models on poisoned data. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id= cAw860ncLRW.

[35] Yiming Li, Yong Jiang, Zhifeng Li, and Shu-Tao Xia. Backdoor learning: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(1):5–22, 2024. doi: 10.1109/TNNLS.2022.3182979.

[36] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending against backdooring attacks on deep neural networks. In Research in Attacks, Intrusions, and Defenses, pages 273–294, 2018.

[37] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. Sphereface: Deep hypersphere embedding for face recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.

[38] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

[39] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations, 2017. URL https://arxiv.org/abs/1610.08401.

[40] S. Moschoglou, A. Papaioannou, C. Sagonas, J. Deng, I. Kotsia, and S. Zafeiriou. Agedb: the first manually collected, in-the-wild age database. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW), volume 2, page 5, 2017.

[41] Tuan Anh Nguyen and Tuan Anh Tran. Input-aware dynamic backdoor attack. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

[42] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary De-Vito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/ paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

[43] S. Sengupta, J.-C. Chen, C. Castillo, V. M. Patel, R. Chellappa, and D. W. Jacobs. Frontal to profile face verification in the wild. In 2016 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 1–9, 2016. doi: 10.1109/WACV.2016.7477558.

[44] Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K. Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, CCS ’16, page 1528–1540, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450341394. doi: 10.1145/2976749.2978392. URL https://doi.org/10. 1145/2976749.2978392.

[45] Tao Sun, Lu Pang, Weimin Lyu, Chao Chen, and Haibin Ling. Mask and restore: Blind backdoor defense at test time with masked autoencoder, 2025. URL https://arxiv.org/abs/2303.15564.

[46] Alexander Turner, Dimitris Tsipras, and Aleksander Madry. Clean-label backdoor attacks, 2019. URL https: //openreview.net/forum?id=HJg6e2CcK7.

[47] Alexander Unnervik, Hatef Otroshi Shahreza, Anjith George, and Sébastien Marcel. Model pairing using embedding translation for backdoor attack detection on open-set classification tasks. In Neurips Safe Generative AI Workshop 2024, 2024. URL https://openreview.net/forum?id=wSpFTuqSRF.

[48] Diana Voth, Leonidas Dane, Jonas Grebe, Sebastian Peitz, and Philipp Terhörst. Effective backdoor learning on open-set face recognition systems. In Proceedings ofthe Winter Conference on Applications ofComputer Vision (WACV), pages 1027–1039, February 2025.

[49] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In 2019 IEEE Symposium on Security and Privacy (SP), pages 707–723, 2019. doi: 10.1109/SP.2019.00031.

[50] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[51] Qingzhong Wang, Pengfei Zhang, Haoyi Xiong, and Jian Zhao. Face.evolve: A high-performance face recognition library. arXiv preprint arXiv:2107.08621, 2021.

[52] Emily Wenger, Josephine Passananti, Arjun Nitin Bhagoji, Yuanshun Yao, Haitao Zheng, and Ben Y. Zhao. Backdoor attacks against deep learning systems in the physical world. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6202–6211, 2021. doi: 10.1109/CVPR46437.2021.00614.

[53] Dongxian Wu and Yisen Wang. Adversarial neuron pruning purifies backdoored deep models. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 16913–16925. Curran Associates, Inc., 2021. URL https://proceedings. neurips.cc/paper\_files/paper/2021/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf.

[54] Zhihao Wu, Yushi Cheng, Shibo Zhang, Xiaoyu Ji, and Wenyuan Xu. Uniid: Spoofing face authentication system by universal identity. Proceedings 2024 Network and Distributed System Security Symposium, 2024. URL https://api.semanticscholar.org/CorpusID:267617716.

[55] Huifen Xing, Siok Yee Tan, Faizan Qamar, and Yuqing Jiao. Face anti-spoofing based on deep learning: A comprehensive survey. Applied Sciences, 15(12), 2025. ISSN 2076-3417. doi: 10.3390/app15126891. URL https://www.mdpi.com/2076-3417/15/12/6891.

[56] Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. Wider face: A face detection benchmark. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[57] Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xiaobai Li, Feng Zhou, and Guoying Zhao. Searching central difference convolutional networks for face anti-spoofing. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5304, 2020. doi: 10.1109/CVPR42600.2020. 00534.

[58] Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503, 2016. doi: 10.1109/LSP. 2016.2603342.

[59] Yuanhan Zhang, Zhenfei Yin, Yidong Li, Guojun Yin, Junjie Yan, Jing Shao, and Ziwei Liu. Celeba-spoof: Large-scale face anti-spoofing dataset with rich annotations. In European Conference on Computer Vision (ECCV), 2020.

[60] Jian ZHAO. Github - zhaoj9014/face.evolve: High-performance face recognition library on paddlepaddle & pytorch — github.com. https://github.com/{Z}hao{J}9014/face.evo{L}{V}e, 2019. [Accessed 15-05- 2026].

[61] W. Zhao, R. Chellappa, P. J. Phillips, and A. Rosenfeld. Face recognition: A literature survey. ACM Comput. Surv., 35(4):399–458, December 2003. ISSN 0360-0300. doi: 10.1145/954339.954342. URL https://doi. org/10.1145/954339.954342.

[62] T. Zheng and W. Deng. Cross-pose lfw: A database for studying cross-pose face recognition in unconstrained environments. Technical Report 18-01, Beijing University of Posts and Telecommunications, February 2018.

[63] Tianyue Zheng, Weihong Deng, and Jiani Hu. Cross-age LFW: A database for studying cross-age face recognition in unconstrained environments. CoRR, abs/1708.08197, 2017. URL http://arxiv.org/abs/1708.08197.

## Appendix

## A Algorithms for the Training-time Early Poison Identification Defense

Algorithm 1 Early Poison Identification defense run (Offline)   
Require: Face feature extractor $f _ { \theta }$ to be trained; training dataset loader $\overline { { \mathcal { D } } }$ over identity set $\partial ;$ method start batch   
$s _ { b } ;$ check interval $b _ { i } ;$ number of identities to capture $n ;$ boolean condition $p$ indicating whether to run pruning   
alongside identification:   
Ensure: Trained feature extractor $f _ { \theta }$ with n identities captured from $\mathcal { D }$   
1: $\mathbf { M } \gets \mathbf { 0 } ^ { | \mathcal { V } | }$ ▷ Per-identity prediction match counts   
2: $\mathbf { C } \gets \mathbf { 0 } ^ { | \mathcal { V } | }$ ▷ Per-identity sample counts   
3: $\Psi  [ ]$ ▷ Accumulator for captured identities   
4: $t _ { \mathrm { b a t c h } }  0$   
5: $t _ { \mathrm { p r u n e } } \gets 0$   
6: for each minibatch $( \mathbf { X } , \mathbf { y } ) \sim { \mathcal { D } }$ do   
7: $t _ { \mathrm { b a t c h } }  t _ { \mathrm { b a t c h } } + 1$   
8: $\hat { \mathbf { y } } \gets f _ { \theta } ( \mathbf { X } )$ ▷ Forward pass: obtain predictions for the current batch   
9: if $t _ { \mathrm { b a t c h } } \geq s _ { b }$ then   
10: for $i = 1$ to |y| do   
11: $\mathbf { M } _ { \mathbf { y } _ { i } }  \ddot { \mathbf { M } } _ { \mathbf { y } _ { i } } + \mathbf { 1 } [ \hat { \mathbf { y } } _ { i } = \mathbf { y } _ { i } ]$ ▷ Increment match counter if prediction is correct   
12: $\mathbf { C } _ { \mathbf { y } _ { i } } ^ { \textrm { - } }  \mathbf { C } _ { \mathbf { y } _ { i } } ^ { \textrm { - } } + 1$   
13: if $t _ { \mathrm { b a t c h } } \geq s _ { b }$ and $t _ { \mathrm { b a t c h } }$ mod $b _ { i } = 0$ and $t _ { \mathrm { p r u n e } } < n$ then   
14: $\mathrm { a c c } _ { y } \gets \mathbf { M } _ { y } / \mathbf { C } _ { y } \quad \forall y \in \mathcal { V } , \mathbf { C } _ { y } > 0$ ▷ Compute current running accuracy per identity   
15: $y ^ { \ast } \gets$ arg max acc<sub>y</sub> ▷ Identify the class with the highest intra-class accuracy   
$y { \in } { \mathcal { y } } \backslash \Psi$   
16: $\Psi  \Psi \cup \{ y ^ { * } \}$ ▷ Records the class as abnormal   
17: $\mathbf { i f } \ p$ is True then ▷ Prune rather than just record   
18: $\mathcal { D }  \mathcal { D } \setminus \{ ( \mathbf { x } , y ) \in \mathcal { D } \mid y = y ^ { * } \}$ ▷ Remove all samples of identity $y ^ { * }$ from the training set   
19: $\mathcal { Y }  \mathcal { Y } \setminus \zeta _ { y ^ { \ast } } \}$   
20: $t _ { \mathrm { p r u n e } }  t _ { \mathrm { p r u n e } } + 1$   
21: Update $\theta$ via backpropagation on batch $( \mathbf { X } , \mathbf { y } )$ ▷ Normal training step proceeds regardless of pruning   
22: return $f _ { \theta }$

## B Algorithms for the Inference-time Detector-Level Model Pairing Defense

```latex
Algorithm 2 Detector-Level Model Pairing defense threshold calibration (Offline)
Require: Clean, held-out calibration detection image set $\mathcal { D } _ { \mathrm { c a l } } = \left\{ ( I _ { i } , L _ { i } ^ { * } , B _ { i } ^ { * } ) \right\} _ { i = 1 } ^ { N }$ with ground-truth landmarks
$L _ { i } ^ { * } \in \mathbb { R } ^ { 5 \times 2 }$ and bounding boxes $B _ { i } ^ { * } ;$ paired face detector $f _ { \mathrm { p a i r e d } } ;$ quantile level α (e.g. 0.99)
Ensure: Landmark shift threshold $\tau _ { \mathrm { L S } }$
1: $\Delta  [ ]$ ▷ Accumulator of per-sample Normalized Mean Errors (NME)
2: for each $( I _ { i } , L _ { i } ^ { * } , B _ { i } ^ { * } ) \in \mathcal { D } _ { \mathrm { c a l } }$ do
3: $( \hat { B } _ { i } , \ \hat { L } _ { i } )  f _ { \mathrm { p a i r e d } } ( I _ { i } )$ ▷ Run paired face detector on calibration image
4: $\delta _ { i } \gets \bar { \underline { { 1 } } } \sum ^ { 1 0 } \big \| \hat { L } _ { i } ^ { ( k ) } - L _ { i } ^ { * ( k ) } \big \| _ { 2 }$ ▷ Normalized Mean Error (NME) of paired model w.r.t. ground truth
10 diagonal $( B _ { i } ^ { * } )$
k=1
5: $\Delta \gets \Delta \cup \{ \delta _ { i } \}$ ▷ Append NME $\delta _ { i } \mathrm { t o } \Delta$
6: $\tau _ { \mathrm { L S } } \gets \operatorname* { m a x } \left( Q _ { \alpha } ( \Delta ) , 0 . 0 4 \right)$ ▷ α-quantile threshold of the defense with ceiling value
7: return $\tau _ { \mathrm { L S } }$
```

Algorithm 3 Detector-Level Model Pairing defense run (Online)   
Require: Candidate image $I ;$ main face detector $f _ { \mathrm { m a i n } } ;$ paired face detector $f _ { \mathrm { p a i r e d } } ;$ IoU threshold $\tau _ { \mathrm { i o u } }$ (e.g. 0.5);   
landmark shift threshold $\tau _ { \mathrm { L S } }$ (precomputed via Algorithm 2)   
Ensure: Decision $d \in \{$ ACCEPT, REJECT}   
1: $( \hat { B } _ { \mathrm { m a i n } } , \hat { L } _ { \mathrm { m a i n } } ) \gets f _ { \mathrm { m a i n } } ( I )$   
2: if $f _ { \mathrm { m a i n } }$ returns no detection then   
3: return REJECT ▷ No candidate face, thus nothing to verify   
4: $( \hat { B } _ { \mathrm { p a i r e d } } , \hat { L } _ { \mathrm { p a i r e d } } ) \gets f _ { \mathrm { p a i r e d } } ( I )$   
5: if $f _ { \mathrm { p a i r e d } }$ returns no detection then   
6: return REJECT ▷ Check 1: $f _ { \mathrm { p a i r e d } }$ disagrees with $f _ { \mathrm { m a i n } }$ (Face Generation Attack suspected)   
7: iou $ \mathrm { I o U } ( \hat { B } _ { \mathrm { m a i n } } , \hat { B } _ { \mathrm { p a i r e d } } )$   
8: if iou $< \tau _ { \mathrm { i o u } }$ then   
9: return REJECT ▷ Check 2: $f _ { \mathrm { p a i r e d } }$ confounds with $f _ { \mathrm { m a i n } }$ (Face Generation Attack suspected)   
10: $\delta \gets \frac { 1 } { - } \sum ^ { 1 0 } \frac { \| \hat { L } _ { \mathrm { m a i n } } ^ { ( k ) } - \hat { L } _ { \mathrm { p a i r e d } } ^ { ( k ) } \| _ { 2 } } { r }$ ▷ Inter-model NME where $f _ { \mathrm { p a i r e d } }$ serves as a ground-truth proxy   
$1 0 $ diagonal $\left( \hat { B } _ { \mathrm { p a i r e d } } \right)$   
11: $\mathbf { i f } \delta > \tau _ { \mathrm { L S } }$ then   
12: return REJECT ▷ Check 3: Mismatch exceeds the α-quantile threshold (Landmark Shift Attack suspected)   
13: return ACCEPT

## C Algorithms for the Inference-time Cut-and-Mix Defense

Algorithm 4 Cut-and-Mix defense threshold calibration (Offline)   
Require: Known-clean calibration image set $\mathcal { D } _ { \mathrm { c a l } } = \{ { \mathbf { c } _ { k } } \} _ { k = 1 } ^ { K } ;$ held-out clean validation image set $\boldsymbol { \gamma } = \{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N } ;$ set   
of J boolean spatial masks $\mathcal { M } = \{ \mathbf { m } _ { j } \} _ { j = 1 } ^ { J } , \mathbf { m } _ { j } \in \{ 0 , 1 \} ^ { H \times W }$ ; face feature extractor $f ;$ quantile level α (e.g.   
0.99)   
Ensure: Rejection threshold τ, cut-off point κ   
1: Φ $ [ ]$ ▷ Accumulator for per-sample mixed image similarity scores   
2: for each calibration image $\mathbf { c } _ { k } \in \mathcal { D } _ { \mathrm { { c a l } } }$ do   
3: $\mathbf { e } _ { k } \gets f ( \mathbf { c } _ { k } )$ ▷ compute the embedding of the clean calibration image   
4: $S _ { k } \gets [ ]$ ▷ Collect pairwise similarities for this sample   
5: for each mask $\mathbf { m } _ { j } \in \mathcal { M }$ do   
6: for each validation image $\mathbf { v } _ { i } \in \mathcal { V }$ do   
7: $\mathbf { c } _ { k } ^ { ( i j ) } \gets \mathbf { m } _ { j } \odot \mathbf { c } _ { k } + ( \mathbf { 1 } - \mathbf { m } _ { j } ) \odot \mathbf { v } _ { i }$ ▷ Retain masked region of $\mathbf { x } ;$ fill remainder with $\mathbf { v } _ { i }$   
8: $\mathbf { e } ^ { ( i j ) }  f ( \mathbf { c } _ { k } ^ { ( i j ) } )$   
9: Append cossim $\left( \mathbf { e } _ { k } , \mathbf { e } ^ { ( i j ) } \right)$ to $\scriptstyle { S _ { k } }$   
10: Append $\boldsymbol { S _ { k } }$ to Φ   
11: $\tau \gets Q _ { \alpha } ( \Phi )$ ▷ Compute the α-quantile of clean calibration score distribution   
12: $\begin{array} { r } { c _ { k } \gets \sum _ { s \in \mathcal { S } _ { k } } \mathbf { 1 } [ s > \tau ] } \end{array}$ ▷ Compute the per-image exceedance count   
13: $\kappa  Q _ { \alpha } \big ( \{ c _ { k } \} _ { k = 1 } ^ { K } \big )$ ▷ Compute the associated cut-off point for false matches associated with τ   
14: return $\tau , \kappa \iota$ ▷ Because κ is calibrated as the α-quantile $( \mathrm { e } . \mathrm { g } . , \alpha = 0 . 9 9 )$ of exceedance counts on clean calibration   
images, Cut-and-Mix guarantees an expected clean FPR of $\approx 1 - \alpha = 1 \%$ by construction.

Algorithm 5 Cut-and-Mix at Runtime (Online)   
Require: Query image $\mathbf { x } \in [ 0 , 1 ] ^ { C \times H \times W }$ ; face feature extractor $f ;$ held-out clean validation image set $\overline { { \gamma = \{ \mathbf { v } _ { i } \} _ { i = 1 } ^ { N } } } ;$   
set of J boolean spatial masks $\mathcal { M } = \{ \mathbf { m } _ { j } \} _ { j = 1 } ^ { J } , \mathbf { m } _ { j } \in \{ 0 , 1 \} ^ { H \times W }$ ; rejection threshold τ and cut-off κ (precomputed   
via Algorithm 4)   
Ensure: Decision d ∈ {ACCEPT, REJECT}   
1: ${ \bf e } _ { q } \gets f ( { \bf x } )$ ▷ compute the embedding of the query image   
2: $\boldsymbol { \mathcal { S } } \gets [ ]$ ▷ Collect pairwise similarities for the query   
3: for each mask $\mathbf { m } _ { j } \in \mathcal { M }$ do   
4: for each validation image $\mathbf { v } _ { i } \in \mathcal { V }$ do   
5: $\mathbf { x } ^ { ( i j ) }  \mathbf { m } _ { j } \odot \mathbf { x } + ( \mathbf { 1 } - \mathbf { m } _ { j } ) \odot \mathbf { v } _ { i }$ ▷ Retain masked region of $\mathbf { x } ;$ fill remainder with $\mathbf { v } _ { i }$   
6: $\mathbf { e } ^ { ( i j ) }  f ( \mathbf { x } ^ { ( i j ) } )$   
7: Append cossim $\left( \mathbf { e } _ { q } , \mathbf { e } ^ { ( i j ) } \right)$ to S   
8: $\begin{array} { r } { \mathbf { i f } \sum _ { s \in \mathcal { S } } \mathbf { 1 } [ s > \tau ] > \kappa } \end{array}$ then   
9: return REJECT ▷ Score exceeds the clean α-quantile: the trigger dominates mixed embeddings regardless of   
$\mathbf { v } _ { i } ,$ , indicating a backdoored query   
10: return ACCEPT ▷ Score is consistent with the clean calibration distribution; x is deemed trigger-free

![](images/d961f1a8d246ce8c62bb8837c2984ca4b99c28e9c34310526621106a43afa066.jpg)  
Figure 14: Validation images used by the Cut-and-Mix defense, extracted from the CelebA dataset [38].

![](images/8a86e957b2911c2e44053f8e8456e0ca1be505a95efa5895ec0c4675e1d33be4.jpg)  
Figure 15: Mask used by the Cut-and-Mix defense, 3 of which are from Chen et al. [7].

## D Full model results

Table 3: Performance results of backdoored face detection models.
<table><tr><td colspan="3">Model</td><td colspan="2">Performance</td></tr><tr><td>Architecture</td><td>Attack</td><td>Trigger</td><td>Precision</td><td>Attack Success Rate</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>99.9%</td><td>99.5%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>100.0%</td><td>99.5%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>100.0%</td><td>99.5%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>100.0%</td><td>99.5%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>98.2%</td><td>92.8%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { S I G } , \alpha = 0 . 1 6 , f = 6$ </td><td>98.6%</td><td>76.6%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>98.6%</td><td>96.7%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td>SIG, α = 0.16, f = 6</td><td>98.6%</td><td>87.7%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td>BadNets, α = 0.5</td><td>100.0%</td><td>92.2%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td>BadNets, α = 1.0</td><td>100.0%</td><td>94.7%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td>BadNets, α = 0.5</td><td>100.0%</td><td>88.1%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td>BadNets, α = 1.0</td><td>100.0%</td><td>93.5%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>97.9%</td><td>92.5%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathrm { S I G } , \alpha = 0 . 1 6 , f = 6$ </td><td>98.6%</td><td>57.0%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td> $\operatorname { S I G } , \alpha = 0 . 3 , \dot { f } = 6$ </td><td>98.6%</td><td>97.5%</td></tr></table>

Table 4: Pair-wise accuracies of backdoored face feature extraction models.
<table><tr><td colspan="3">Model</td><td colspan="7">Performance</td></tr><tr><td>Architecture</td><td>Attack</td><td>Trigger</td><td>LFW</td><td>CFP-FF</td><td>CFP-FP</td><td>AgeDB</td><td>CALFW</td><td>CPLFW</td><td>VGGFace2-FP</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>BadNets</td><td>99.9%</td><td>99.7%</td><td>97.5%</td><td>97.9%</td><td>97.7%</td><td>92.4%</td><td>97.4%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>SIG</td><td>99.9%</td><td>99.7%</td><td>97.5%</td><td>97.7%</td><td>97.6%</td><td>92.9%</td><td>97.4%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>99.8%</td><td>99.3%</td><td>93.2%</td><td>92.1%</td><td>95.8%</td><td>91.0%</td><td>94.6%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>99.2%</td><td>99.0%</td><td>97.8%</td><td>95.5%</td><td>97.3%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>98.3%</td><td>98.6%</td><td>97.7%</td><td>94.4%</td><td>96.9%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>Mask</td><td>99.9%</td><td>99.8%</td><td>98.5%</td><td>98.8%</td><td>97.7%</td><td>94.5%</td><td>96.8%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>98.6%</td><td>98.8%</td><td>97.7%</td><td>94.4%</td><td>96.8%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>98.7%</td><td>98.9%</td><td>97.6%</td><td>94.8%</td><td>96.8%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>98.7%</td><td>98.9%</td><td>97.4%</td><td>94.5%</td><td>96.5%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>97.5%</td><td>98.6%</td><td>97.8%</td><td>93.7%</td><td>97.0%</td></tr><tr><td>RobFaceNet_sig</td><td>All-to-One Clean-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>97.4%</td><td>98.6%</td><td>97.8%</td><td>93.7%</td><td>96.7%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>97.6%</td><td>97.6%</td><td>97.7%</td><td>92.6%</td><td>97.5%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>SIG</td><td>99.9%</td><td>99.7%</td><td>97.5%</td><td>97.9%</td><td>97.7%</td><td>92.8%</td><td>97.5%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>99.1%</td><td>99.0%</td><td>97.7%</td><td>95.4%</td><td>97.4%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>99.2%</td><td>98.9%</td><td>97.7%</td><td>95.6%</td><td>97.2%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>99.8%</td><td>98.4%</td><td>98.6%</td><td>97.7%</td><td>94.7%</td><td>97.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>FauxIt</td><td>100.0%</td><td>99.8%</td><td>98.4%</td><td>98.6%</td><td>97.7%</td><td>94.7%</td><td>97.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>Mask</td><td>99.9%</td><td>99.8%</td><td>98.3%</td><td>98.7%</td><td>97.6%</td><td>94.7%</td><td>96.9%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>99.8%</td><td>98.6%</td><td>98.8%</td><td>97.7%</td><td>94.5%</td><td>96.9%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>98.7%</td><td>98.9%</td><td>97.7%</td><td>95.3%</td><td>97.1%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>98.8%</td><td>98.9%</td><td>97.5%</td><td>94.7%</td><td>96.8%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>97.3%</td><td>98.6%</td><td>97.7%</td><td>93.6%</td><td>96.6%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>97.4%</td><td>98.6%</td><td>97.8%</td><td>93.7%</td><td>96.6%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>97.7%</td><td>98.5%</td><td>97.8%</td><td>93.5%</td><td>96.9%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>SIG</td><td>99.9%</td><td>99.7%</td><td>97.6%</td><td>97.7%</td><td>97.6%</td><td>93.1%</td><td>97.2%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>99.2%</td><td>98.9%</td><td>97.7%</td><td>95.5%</td><td>97.2%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>99.3%</td><td>99.1%</td><td>97.7%</td><td>95.5%</td><td>97.2%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>BadNets</td><td>100.0%</td><td>99.8%</td><td>98.4%</td><td>98.8%</td><td>97.8%</td><td>94.6%</td><td>97.1%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>Mask</td><td>100.0%</td><td>99.8%</td><td>98.5%</td><td>98.7%</td><td>97.7%</td><td>94.7%</td><td>96.9%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>98.5%</td><td>98.7%</td><td>97.7%</td><td>94.6%</td><td>97.0%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>99.9%</td><td>99.8%</td><td>98.8%</td><td>98.8%</td><td>97.6%</td><td>95.0%</td><td>96.8%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>SIG BadNets</td><td>99.9%</td><td>99.8%</td><td>98.8%</td><td>98.9%</td><td>97.5%</td><td>94.7%</td><td>96.9%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td></td><td>99.9%</td><td>99.8%</td><td>97.7%</td><td>98.5%</td><td>97.8%</td><td>93.5%</td><td>96.9%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>99.9%</td><td>99.8%</td><td>97.9%</td><td>98.7%</td><td>97.8%</td><td>93.7%</td><td>96.6%</td></tr></table>

Table 5: Backdoor attack success rates of backdoored face feature extraction models.
<table><tr><td colspan="3">Model</td><td colspan="7">Attack Success Rate</td></tr><tr><td>Architecture</td><td>Attack</td><td>Trigger</td><td>LFW</td><td>CFP-FF</td><td>CFP-FP</td><td>AgeDB</td><td>CALFW</td><td>CPLFW</td><td>VGGFace2-FP</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.6%</td><td>0.7%</td><td>4.1%</td><td>5.4%</td><td>4.1%</td><td>7.5%</td><td>4.5%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>SIG</td><td>8.8%</td><td>9.0%</td><td>16.8%</td><td>21.8%</td><td>23.5%</td><td>33.3%</td><td>26.6%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>1.1%</td><td>1.8%</td><td>13.8%</td><td>12.8%</td><td>5.2%</td><td>11.2%</td><td>11.0%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>SIG</td><td>3.1%</td><td>1.2%</td><td>2.8%</td><td>7.6%</td><td>5.4%</td><td>22.5%</td><td>19.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.1%</td><td>0.4%</td><td>1.8%</td><td>2.6%</td><td>1.4%</td><td>3.7%</td><td>3.8%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>Mask</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>31.0%</td><td>24.1%</td><td>40.1%</td><td>53.3%</td><td>45.4%</td><td>58.6%</td><td>64.1%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.2%</td><td>0.2%</td><td>1.6%</td><td>2.1%</td><td>0.7%</td><td>2.6%</td><td>1.5%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>SIG</td><td>5.5%</td><td>8.1%</td><td>9.1%</td><td>13.0%</td><td>17.1%</td><td>20.2%</td><td>25.8%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.1%</td><td>0.5%</td><td>2.9%</td><td>2.5%</td><td>1.6%</td><td>5.0%</td><td>3.3%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>55.5%</td><td>43.9%</td><td>56.7%</td><td>62.9%</td><td>58.1%</td><td>85.4%</td><td>80.4%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>FauxIt</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>Mask</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.8%</td><td>1.7%</td><td>15.7%</td><td>5.2%</td><td>1.6%</td><td>8.3%</td><td>11.8%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>Mask</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.0%</td><td>0.0%</td><td>0.1%</td><td>0.0%</td><td>0.0%</td><td>0.1%</td><td>0.0%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr></table>

## E Full defense results

Table 6: Performance results of the Detector-level Model Pairing defense.
<table><tr><td colspan="3">Model</td><td colspan="2">Performance</td></tr><tr><td>Architecture</td><td>Attack</td><td>Trigger</td><td>False Positive Rate</td><td>False Negative</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>0.59%</td><td>7.81%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>0.68%</td><td>7.91%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>0.59%</td><td>7.52%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>0.59%</td><td>8.11%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>0.59%</td><td>7.71%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>FGA</td><td> $\mathrm { S I G } , \alpha = 0 . 1 6 , f = 6$ </td><td>0.59%</td><td>4.59%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>0.59%</td><td>7.23%</td></tr><tr><td>RetinaFace ResNet50</td><td>FGA</td><td> $\mathrm { S I G } , \alpha = 0 . 1 6 , f = 6$ </td><td>0.59%</td><td>5.57%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>1.07%</td><td>0.39%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>1.07%</td><td>0.2%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td> $\mathrm { B a d N e t s , } \alpha = 0 . 5$ </td><td>1.07%</td><td>0.29%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td> $\mathrm { B a d N e t s } , \alpha = 1 . 0$ </td><td>1.07%</td><td>0.29%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>1.07%</td><td>0.97%</td></tr><tr><td>RetinaFace MobileNetV1</td><td>LSA</td><td> $\mathrm { S I G } , \alpha = 0 . 1 6 , f = 6$ </td><td>1.07%</td><td>15.5%</td></tr><tr><td>RetinaFace ResNet50</td><td>LSA</td><td> $\mathbf { S I G } , \alpha = 0 . 3 , f = 6$ </td><td>0.88%</td><td>1.74%</td></tr></table>

Table 7: Performance results of the Cut-and-Mix defense. Note the high FNRs for Clean-label and Master Face models with 0% ASR occur because the backdoor is never activated (they still show up as a backdoor ground truth in our tests).
<table><tr><td colspan="3">Model</td><td colspan="2">Cut-and-Mix</td></tr><tr><td>Architecture</td><td>Attack</td><td>Trigger</td><td>FPR</td><td>FNR</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.7%</td><td>99.0%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Clean-label</td><td>SIG</td><td>0.8%</td><td>99.7%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.7%</td><td>99.4%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Clean-label</td><td>SIG</td><td>0.5%</td><td>100.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.8%</td><td>98.7%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>Mask</td><td>0.9%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Clean-label</td><td>SIG</td><td>0.7%</td><td>99.4%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>BadNets</td><td>1.1%</td><td>98.4%</td></tr><tr><td>ResNet50</td><td>All-to-One Clean-label</td><td>SIG</td><td>0.5%</td><td>100.0%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Clean-label</td><td>BadNets</td><td>0.6%</td><td>98.7%</td></tr><tr><td>RobFaceNet_sig</td><td>All-to-One Clean-label</td><td>SIG</td><td>1.0%</td><td>99.8%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>BadNets</td><td>1.0%</td><td>0.0%</td></tr><tr><td>GhostFaceNetV2</td><td>All-to-One Poison-label</td><td>SIG</td><td>0.6%</td><td>0.4%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>1.0%</td><td>0.0%</td></tr><tr><td>IR-SE-50</td><td>All-to-One Poison-label</td><td>SIG</td><td>0.9%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>0.8%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>FauxIt</td><td>0.7%</td><td>0.1%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>Mask</td><td>1.0%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>0.7%</td><td>0.0%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>BadNets</td><td>1.1%</td><td>0.0%</td></tr><tr><td>ResNet50</td><td>All-to-One Poison-label</td><td>SIG</td><td>0.8%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>BadNets</td><td>0.6%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>All-to-One Poison-label</td><td>SIG</td><td>0.7%</td><td>0.1%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.6%</td><td>99.4%</td></tr><tr><td>GhostFaceNetV2</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.6%</td><td>80.5%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.8%</td><td>99.2%</td></tr><tr><td>IR-SE-50</td><td>Master-Face Poison-label</td><td>SIG</td><td>1.1%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>BadNets</td><td>1.0%</td><td>98.7%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>Mask</td><td>0.8%</td><td>0.0%</td></tr><tr><td>MobileFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.7%</td><td>2.3%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>BadNets</td><td>1.0%</td><td>98.1%</td></tr><tr><td>ResNet50</td><td>Master-Face Poison-label</td><td>SIG</td><td>0.6%</td><td>0.0%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td>BadNets</td><td>0.5%</td><td>12.6%</td></tr><tr><td>RobFaceNet</td><td>Master-Face Poison-label</td><td>SIG</td><td>1.0%</td><td>0.0%</td></tr></table>