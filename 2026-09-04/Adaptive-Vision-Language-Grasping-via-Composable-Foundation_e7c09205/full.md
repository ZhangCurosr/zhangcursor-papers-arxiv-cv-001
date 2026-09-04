# Adaptive Vision-Language Grasping via Composable Foundation Priors and Generalizable Grasp Synthesis

Sixu Yan , Student Member, IEEE, Shikang Wang , Binhua Huang , Xuanlai Tang , Guohua Fan , Fan Huang , Haoxuan Li , Yongkang Li , Yuhan Li , Bencheng Liao , Zeyu Zhang , Member, IEEE, Wenyu Liu , Senior Member, IEEE, Hangxin Liu , Member, IEEE, and Xinggang Wang , Senior Member, IEEE

![](images/76e0c7c1cb63ac0e67f829b5a1bcbb03c1b3963b9239a72e73303852e4d3cb9f.jpg)  
Fig. 1: Adaptive vision-language grasping. (a) Real-world grasping occurs under diverse contextual conditions and taskdependent factors, each of which can substantially alter the desired grasp strategy. (b) End-to-end VLG policies typically entangle these factors within a single network trained on large-scale multimodal data, which limits their adaptability to unseen task contexts. In contrast, (c) AdaRoboVLG decouples task-dependent understanding from physical grasp synthesis: foundation-model priors construct a structured grasp interface, from which a generalizable base policy synthesizes executable grasp poses.

Abstract—This paper proposes AdaRoboVLG, a task-adaptive Vision-Language-Grasp (VLG) framework that supports generalizable grasp synthesis across different robotic hands. Unlike existing VLG methods that tightly couple foundation models with end-to-end grasp policies, AdaRoboVLG learns an efficient generalizable base policy that generates and evaluates physically feasible grasp candidates through explicit kinematic mapping and force-closure-based stability estimation, while offloading taskdependent understanding to specialized foundation-model modules. These modules provide composable priors that are integrated into the grasp synthesis process, enabling contextually adaptive grasp synthesis without retraining the underlying grasp policy. Through extensive simulation and real-world experiments, we demonstrate that (i) the base policy exhibits efficient learning and strong cross-hand generalization, (ii) the framework effectively in-

corporates spatial, cognitive, and temporal priors to address three representative grasping challenges without compromising grasp synthesis performance compared to state-of-the-art methods, and (iii) these priors can operate jointly to enable functional grasping in cluttered and dynamic environments. These results indicate that decoupling physical grasp synthesis from task-dependent understanding provides a scalable paradigm for robotic grasping, allowing future advances in foundation models to be directly translated into improved grasp capabilities without redesigning or retraining the underlying grasp policy. Supplementary videos are available at https://adarobovlg.github.io/.

Index Terms—Adaptive vision-language grasping, Generalizable grasp synthesis, Foundation models for embodied AI

## I. INTRODUCTION

progress, the grasp generation problem remains challenging due to the difficulty in robustly selecting a grasp among almost infinite possible configurations between an end-effector and an object. More importantly, a grasp is inherently contextual. The same object may require entirely different grasp strategies depending on its surroundings, task objectives, and interaction dynamics. As illustrated in Fig. 1a, executing the task instruction “Grasp the mug to ⟨action⟩” oftentimes involves such intertwined factors. The target object may be partially occluded or covered by clutter, restricting feasible approach directions and grasp configurations. The grasp itself must satisfy functional requirements associated with subsequent object use: pour water or hand it over. And continuous adjustment is required for a successful grasp when the object or robot is moving. These spatial, cognitive, and temporal challenges frequently arise simultaneously, making robust and adaptive grasping particularly difficult.

Humans, in contrast, can effortlessly adapt their grasping behaviors to such combined challenges in varying situations. A key reason is that humans exploit rich priors accumulated from commonsense knowledge [2] and past experience [3] to guide perception [4], reasoning, and motor coordination [5]. Such priors enable efficient selection and adaptation of grasp strategies without requiring exhaustive experience for every possible situation.

Analogous to the role of prior knowledge in human intelligence, foundation models have shown strong capabilities to provide informative priors for visual recognition and tracking [6], [7], spatial understanding [8], [9], embodied reasoning [10], [11], and robot control [12], [13]. In particular, their powerful visual recognition and language understanding capabilities have fueled the emergence of VLG models [14], [15], [16], [17], [18], which exhibit strong generalization capabilities and support natural interactions by tightly integrating perception and grasp execution. However, existing VLG methods predominantly incorporate foundation models within an end-to-end framework that directly maps visual observations and language instructions to grasp poses (see Fig. 1b). Although effective, the performance of such specialized policies is largely constrained by the coverage of training data, model scale, and computational resources [14], [16], [18]. As a result, scaling these policies to cover the vast diversity of grasp configurations, task intents, and environmental dynamics encountered in the real world remains impractical [19].

Taking a different perspective, we propose AdaRoboVLG, an adaptive vision-language grasping framework (see Fig. 1c) that decouples task-dependent understanding from physical grasp synthesis. Specifically, we design a structured grasp interface consisting of object geometry, Contact Grasp Representations (CGRs) [20], and grasp types. A CGR encodes a local contact primitive on the object surface, including the contact location, approach direction, closing direction, and grasp width. The grasp types follow the human grasp taxonomy [21] and are kinematically compatible with the target robotic hand. Centered on this structured grasp interface, we first learn an efficient base policy that converts the interface into hand-specific grasp candidates and selects the most stable candidate using a learned decision model that evaluates force-closure conditions [22]. Throughout this process, complementary foundation-model priors for spatial understanding, cognitive reasoning, and temporal tracking construct or update the structured grasp interface, allowing the base policy to synthesize grasps that are both task-consistent and physically executable.

Fig. 2 illustrates three representative examples. (i) By lifting DINOv3 [7] features extracted from multi-view RGB-D observations into 3D space, the spatial prior predicts global CGRs, providing spatially feasible contact constraints for grasp synthesis in cluttered scenes. (ii) By combining an Large Language Model (LLM) augmented with Retrieval-Augmented Generation (RAG) [23] over a grasp-examples database with structured Chain-of-Thought (CoT) [24] reason ing, and SAM3 [25]-based visual grounding, the cognitive prior localizes the target object, refines the CGRs, and infers appro priate grasp types for functional grasp generation according to language instructions. (iii) By combining the mask-tracking capability of SAM3 with the cross-frame feature consistency of DINOv3, the temporal prior estimates the motion of target geometry and CGRs online, enabling grasps to be continuously updated for dynamic objects. More importantly, these spatial, cognitive, and temporal priors modify the same structured grasp interface, allowing them to operate independently or jointly while enabling contextually adaptive grasp generation across a wide range of scenarios.

In simulation experiments, we first validate that the base policy is effective and generalizable, which not only exhibits efficient learning but also facilitates cross-hand deployment. Building on this base policy, we further evaluate the contextual adaptation of AdaRoboVLG enabled by three types of example foundation priors. For clutter-aware grasping, the spatial prior enables AdaRoboVLG to achieve an average success rate of 88.4% across three robotic hands and three clutter levels on the DexGraspNet 2.0 [26] benchmark, achieving state-of-the-art results on the random and loose splits. For functional grasping, we evaluate the cognitive prior through two complementary experiments. In grasp functionality inference, it achieves functional-part and grasp-type accuracies of 94.0% and 87.0% on an open-set benchmark. In language-guided target grasping, AdaRoboVLG achieves the best average success rates of 81.3% and 86.0% on simulation benchmarks built from GraspClutter6D [27] and GraspNet-1Billion [28], respectively. Finally, for dynamic grasping, evaluations on GraspNet-1Billion show that the temporal prior enables perfect tracking association accuracy, the lowest translation errors, and competitive rotation errors compared to existing grasp-tracking methods.

In real-world settings, we further validate that the spatial, cognitive, and temporal priors can be jointly integrated to address more challenging grasping tasks. For functional grasping in cluttered scenes, AdaRoboVLG achieves an overall success rate of 83.3% over 510 trials across 102 everyday objects from six categories. On a moving conveyor belt, AdaRoboVLG enables functional grasping in dynamic cluttered scenes, while performing online grasp updates at 5 Hz. We also evaluate the tracking robustness of AdaRoboVLG under four types of human-induced disturbances, where it achieves an average success rate of 89.7%.

The broader implications of this work extend beyond grasping performance itself. Decoupling physical grasp synthesis from task-dependent understanding not only improves data and training efficiency, but also establishes a stable interface through which increasingly capable foundation models can be integrated over time. Rather than repeatedly retraining end-toend policies, future advances in perception and reasoning can be directly translated into improved grasping capabilities by updating only the corresponding foundation-prior modules.

## A. Contributions

Our contributions are threefold:

1) We propose AdaRoboVLG, an adaptive VLG framework that decouples task-dependent understanding from physical grasp synthesis and bridges them through a structured grasp interface. It enables task adaptation, cross-hand grasp synthesis, and framework extensibility.

![](images/a92bc13c3903e8edf57afd515c2a5af3a3761bd0dbef5f0b5cbb25eab1686d0e.jpg)  
Fig. 2: Overview of AdaRoboVLG. To enable adaptive vision-language grasping, our framework explicitly decouples task-driven understanding from physical grasp synthesis. By integrating spatial, cognitive, and temporal priors, it constructs a structured grasp interface that directly guides an efficient base grasp policy to generate stable and executable grasp poses.

2) We develop an efficient base policy for generalizable grasp synthesis. It consists of an explicit kinematic mapping and a learned decision model that employs the proposed handobject interaction (HOI) representation to encode local contact geometry relevant to force closure, supporting shared grasp stability evaluation across robotic hands.

3) We introduce composable spatial, cognitive, and temporal prior modules that construct and update the structured grasp interface for task-adaptive grasping under diverse and combined contextual conditions, such as cluttered scenes, functional requirements, and dynamic interactions.

## B. Overview

The remainder of this paper is organized as follows. Sec. II reviews the literature and compares existing research. Sec. III formulates adaptive vision-language grasping and defines the structured grasp interface. Building on this interface, Sec. IV develops the base policy for generalizable grasp synthesis, and Sec. V introduces the three composable foundation-prior modules for task-adaptive grasping under diverse contextual conditions. Sec. VI and Sec. VII demonstrate the efficacy of AdaRoboVLG through simulation and real-world experiments, respectively. Finally, Sec. VIII concludes the paper with a discussion of limitations and future directions.

## II. RELATED WORK

## A. Vision-Language-Conditioned Robotic Grasping

Vision-language grasping enables robots to perform precise grasps based on human instructions. To address the challenges illustrated in Fig. 1a, recent works have advanced this direction by leveraging LLMs and Vision Language Models (VLMs) to (i) provide task-relevant guidance for grasp policies, such as semantic cues [29], [30], object affordances [31], [32], and spatial priors [33], [34]; (ii) perform high-level task planning [35], [36] and functional reasoning [37], [38]; and (iii) develop endto-end Vision-Language-Action (VLA) models [14], [15], [16], [18], trained on large-scale robotic datasets, to directly generate feasible grasp poses [16], [18] or continuous actions [14], [15].

However, these methods are often task- and hand-specific, requiring additional data collection and policy training when adapting to new tasks or embodiments. AdaRoboVLG addresses this limitation by decoupling task understanding from grasp synthesis through diverse foundation priors. New task requirements can be accommodated by updating the relevant priors, while new embodiments are supported by reusing the shared grasp policy, without modifying the entire framework.

## B. Learning-Based Multi-Finger Grasp Pose Synthesis

Learning-based multi-finger grasp synthesis can be broadly categorized into four representative paradigms. Specifically, hand-centric methods directly map perceptual observations to grasp actions [39], [40], [41], [42], [43], enabling efficient inference but remaining constrained to specific hand embodiments. Object-centric methods improve cross-hand generalization using contact maps [44], [45] or contact points [46], [47], but often require costly nonconvex optimization [48]. Interaction centric methods further balance generalization and efficiency by explicitly modeling hand-object relationships [49], [50], [51]. However, these paradigms primarily focus on object-level grasp synthesis. More recently, scene-centric methods have extended grasp synthesis to cluttered environments [26], [52], [53], but their grasp generation and decision processes are typically coupled with specific hand embodiments.

Drawing from these efforts, AdaRoboVLG decomposes grasp synthesis into two stages, with each stage built upon a handagnostic representation. During grasp generation, CGRs, which encode fundamental contact primitives, are converted into executable hand-specific grasp candidates through an explicit mapping function. During grasp decision, these candidates are encoded using the proposed HOI representation and evaluated by a hand-agnostic decision model. This design retains efficient inference while supporting cross-hand grasp synthesis and extending naturally to complex multi-object scenes.

## C. Foundation Models in Embodied Intelligence

Foundation models [54], with their powerful capabilities in perception [7], [55] and reasoning [56], [57], are becoming a vital cornerstone for general-purpose intelligent systems. They are widely used for high-level planning [10], [11] and low-level control [12], [13] in robotics. Current applications can be broadly categorized into three types: (i) zero-shot utilization, where foundation priors are directly used for embodied navigation [58], [59] and manipulation [60], [61] without additional training; (ii) task-specific adaptation, where foundation models are adapted to robotic tasks through efficient finetuning [62], [63]; and (iii) integration into action policies, where foundation models are embedded into closed-loop perception-action pipelines, and some works further explore building policy foundation models [13], [64] trained on large scale robotic datasets.

Inspired by prior work, AdaRoboVLG leverages and composes diverse foundation models to enable adaptive robotic grasping. Specifically, DINOv3 [7] and SAM3 [25] are directly employed in a zero-shot manner for semantic understanding, target segmentation, and temporal tracking. Meanwhile, the LLM is augmented with grasp-relevant taxonomy knowledge to facilitate grasp functionality inference. The resulting spatial, cognitive, and temporal constraints are then unified into a structured grasp interface to guide physical grasp synthesis. By composing diverse foundation priors, AdaRoboVLG can flexibly adapt to novel task contexts and grasping requirements.

## III. PROBLEM FORMULATION AND GRASP INTERFACE

This section presents our formulation for adaptive visionlanguage grasping. We first introduce the essential notation and preliminaries in Sec. III-A and then define the problem and structured grasp interface in Sec. III-B.

![](images/4761a5c8536635505a53d77cfbfd3df137c84205a1b0d9beb2d9334c3c6e6404.jpg)  
Fig. 3: Contact grasp representation. This representation effectively captures a hand-agnostic contact primitive that can be (a) shared across diverse end-effectors, ranging from parallel grippers to multi-finger robotic hands, and (b) adaptable to various functional grasp types.

## A. Notation and Preliminaries

Before delving into the formulation of AdaRoboVLG, we start by defining the following mathematical notation:

• Bold lowercase letters represent vectors $( e . g . , g )$ , with the subscript i denoting the i-th entry of the vector $( e . g . , g _ { i } )$

• Bold uppercase letters represent matrices $( e . g . , \ T )$ . The identity matrix of dimension n is denoted by ${ \cal I } _ { n \times n }$ . The superscript · indicates the transpose.

• Calligraphic font letters are used to denote sets $( e . g . , \mathcal { G } )$ . The notation |·| denotes the number of elements in a set.

## B. Problem Formulation and Structured Grasp Interface

A multi-finger grasp pose g is formally defined as:

$$
g = \left( T , \pmb { q } \right) , T = \left[ \begin{array} { c c c c } { \pmb { R } } & { \pmb { t } } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} \right] ,\tag{1}
$$

where $\pmb { T } \in \mathrm { S E } ( 3 )$ denotes the hand base pose in the world frame, comprising a rotation matrix $\pmb { R } \in \mathbb { R } ^ { 3 \times 3 }$ and a translation vector $\pmb { t } \in \mathbb { R } ^ { 3 } . \ \pmb { q } \in \mathbb { R } ^ { n }$ characterizes the joint configuration of an n-degree-of-freedom (DoF) multi-finger robotic hand. The goal of adaptive vision-language grasping is to synthesize a set of executable grasp poses $\overset { \cdot } { \boldsymbol { \mathcal { G } } } = \bar { \{ }  g _ { i } \} _ { i = 1 } ^ { \rceil \mathcal { G } | }$ that are compatible with hand embodiment H and satisfy the task context C.

Instead of learning an end-to-end network that maps the task context to grasp poses, we decompose adaptive vision-language grasping into task-dependent understanding and physical grasp synthesis, and bridge them through a structured grasp interface. Under this formulation, the task context C is first converted into a structured grasp interface I. Subsequently, executable grasp poses $\mathcal { G } ^ { * }$ are synthesized from this interface:

![](images/5650635a987d1d6b94a1d36e67c862bb2ed222ee366e0fee7c91b1ee61f14b90.jpg)  
Fig. 4: Hand-specific grasp mapping. This function maps the object geometry, CGRs, and grasp type into taxonomyconsistent grasp poses for a specific robotic hand.

$$
f _ { \tau } : \mathcal { C } \to \mathbb { Z } , f _ { \pi } : \mathbb { Z } \times \mathcal { H } \to \mathcal { G } ^ { * } .\tag{2}
$$

Formally, we define this interface as

$$
\begin{array} { r } { \mathcal { I } = ( \mathcal { O } , \mathcal { A } , \mathcal { T } _ { \mathcal { H } } ) , } \end{array}\tag{3}
$$

where O denotes the object geometry, $\mathbf { \mathcal { A } } = \{ \mathbf { { a } } _ { j } \} _ { j = 1 } ^ { | \mathcal { A } | }$ denotes the set of CGRs that characterize the fundamental contact primitives for stable grasping, and $\mathcal { T } _ { \mathcal { H } } = \{ \tau _ { k } \} _ { k = 1 } ^ { | \mathcal { T } _ { \mathcal { H } } | }$ represents the set of grasp types that follow the human grasp taxonomy and are kinematically compatible with the hand embodiment H. Here, $f _ { \pi } ( \cdot )$ denotes the base grasp policy, which synthesizes executable grasp poses from the structured interface I and the hand embodiment H. $f _ { \tau } ( \cdot )$ denotes the transformation from task context C to the interface $\mathcal { T } ,$ which can be instantiated with diverse composable foundation priors.

## IV. BASE GRASP POLICY

This section presents the implementation details of the base grasp policy $f _ { \pi } ( \cdot )$ . We sequentially describe the policy design principle (Sec. IV-A), the CGR definition (Sec. IV-B), the handspecific grasp mapping (Sec. IV-C), the HOI representation (Sec. IV-D), and the hand-agnostic grasp evaluation (Sec. IV-E).

## A. Design Principle

Given the grasp interface I defined in Eq. (3), the base policy converts it into executable grasp poses for a target robotic hand.

To support efficient and cross-hand grasp synthesis, the policy adopts a two-stage process consisting of hand-specific grasp mapping and hand-agnostic grasp evaluation.

For grasp mapping, the interface I is mapped to candidate grasp poses for a specific robotic hand H. Specifically, a hand-specific mapping function $\mathcal { M } _ { \mathcal { H } } ( \cdot )$ takes the target object geometry ${ \mathcal { O } } ,$ a CGR descriptor ${ \pmb a } _ { j }$ , and a grasp type $\tau _ { k }$ as input, and generates the target grasp poses $\mathcal { G } _ { j , k }$ :

$$
\mathcal { G } _ { j , k } = \mathcal { M } _ { \mathcal { H } } \left( \boldsymbol { \mathcal { O } } , \mathbf { a } _ { j } , \tau _ { k } \right) ,\tag{4}
$$

where $\mathcal { G } _ { j , k } = \{ \pmb { g } _ { j , k } ^ { ( i ) } \} _ { i = 1 } ^ { | \mathcal { G } _ { j , k } | }$ is the set of grasp pose candidates. The complete set of grasp candidates for ${ \pmb a } _ { j }$ is computed by aggregating candidates across all such grasp types:

$$
\mathcal { G } _ { j } = \bigcup _ { k = 1 } ^ { | \mathcal { T } _ { \mathcal { H } } | } \mathcal { G } _ { j , k } = \bigcup _ { \tau _ { k } \in \mathcal { T } _ { \mathcal { H } } } \mathcal { M } _ { \mathcal { H } } ( \mathcal { O } , \mathbf { a } _ { j } , \tau _ { k } ) .\tag{5}
$$

Subsequently, we estimate the quality of each grasp candidate. We first extract the HOI representation $\mathcal { R } ^ { \mathcal { H O I } }$ from the handobject contact interface relevant to force closure [22]. This process is achieved by an extractor $\mathcal { E } ( \cdot )$ , which takes the object geometry O, the hand specification $\mathcal { H } ,$ and a grasp pose $\pmb { g } _ { j , k } ^ { ( i ) }$ as input, and produces the HOI representation $\mathcal { R } ^ { \mathcal { H O I } }$

$$
\mathcal { R } ^ { \mathcal { H O T } } = \mathcal { E } \left( \mathcal { O } , \mathcal { H } , g _ { j , k } ^ { ( i ) } \right) .\tag{6}
$$

Then, a hand-agnostic grasp decision model $\mathcal { D } ( \cdot )$ is introduced to predict the grasp success probability $\beta ,$ conditioned on the HOI representation $\mathcal { R } ^ { \mathcal { H O I } }$

$$
\beta = D ( \mathcal { R } ^ { \mathcal { H O T } } ) .\tag{7}
$$

The objective of the base policy is to find a set of top-K grasp poses $\mathcal { G } ^ { * }$ that maximizes the predicted success rate:

$$
\mathcal { G } ^ { * } = \underset { \mathcal { G } \subset \bigcup _ { j } \mathcal { G } _ { j } , | \mathcal { G } | = K } { \arg \operatorname* { m a x } } \mathbb { E } _ { \pmb { g } _ { j , k } ^ { ( i ) } } \bigg [ \mathcal { D } \left( \mathcal { E } ( \mathcal { O } , \mathcal { H } , \pmb { g } _ { j , k } ^ { ( i ) } ) \right) \bigg ] ,\tag{8}
$$

where $\pmb { g } _ { j , k } ^ { ( i ) } \in \mathcal { M } _ { \mathcal { H } } ( \pmb { a } _ { j } , \tau _ { k } ) \cap \mathcal { G } , \pmb { a } _ { j } \in \mathcal { A } , \mathrm { ~ a n d ~ } \tau _ { k } \in \mathcal { T } _ { \mathcal { H } } .$

## B. Contact Grasp Representation

The CGR serves as a fundamental descriptor of graspability, encoding the inherent, local information of the object surface to establish stable grasping contacts. Following prior work [20], we parameterize each CGR a as:

$$
\mathbf { \boldsymbol { a } } = \left( p _ { c } , v _ { a } , v _ { b } , p , w \right) ,\tag{9}
$$

where $\pmb { p _ { c } } \in \mathbb { R } ^ { 3 }$ denotes the potential contact point on the object surface, $p \in [ 0 , 1 ]$ denotes the probability that this point is a stable contact point, $\pmb { v } _ { a } \in \mathbb { R } ^ { 3 } , \lVert \pmb { v } _ { a } \rVert = 1$ denotes the grasp approach vector, $\pmb { v } _ { b } \in \mathbb { R } ^ { 3 } , \| \pmb { v } _ { b } \| = 1$ denotes the grasp baseline (closing) vector, and $w \in \mathbb { R } ^ { + }$ denotes the grasp width. As shown in Fig. 3, this representation captures fundamental, hand-agnostic grasp contact primitives that can be shared across different end-effectors and grasp types.

## C. Hand-Specific Grasp Mapping

This subsection instantiates the mapping function $\mathcal { M } _ { \mathcal { H } } ( \cdot )$ introduced in Eq. (4). In principle, $\mathcal { M } _ { \mathcal { H } } ( \cdot )$ can be implemented by model-based [66], [67] or learning-based [18], [38], [68] grasp synthesis methods. To achieve efficient grasp synthesis, we instantiate it as an explicit mapping function, following the established methodologies in prior works [69], [70], [71].

![](images/02deb91c74e7b99a3e65772b9363b70e4a45fbfd3f8418da15c22ef2ccf2f38b.jpg)  
Fig. 5: Hand-agnostic grasp evaluation. This module evaluates the physical stability of synthesized grasp candidates in a strictly hand-agnostic manner. To achieve this, it first extracts an HOI representation to explicitly capture the local contact geometry. Leveraging a PointTransformer [65] backbone and a contact-aware feature extraction module, the grasp decision model dynamically aggregates these geometric features to predict a continuous stability score for each grasp pose.

For any given grasp type, the hand closing synergy (the first principal component of all kinematically feasible joint configurations) captures approximately 50%–70% of the kinematic variance [72], [73], [74]. Building upon this insight, we instantiate the feasible kinematics for each grasp type as a 1-D parametric manifold within the hand configuration space $\mathcal { Q } _ { \mathcal { H } } \subset \mathbb { R } ^ { n }$ , parameterized solely by the grasp width w. For efficient inference, we precompute this manifold as a handspecific lookup table. Formally, it is defined as

$$
\mathcal { L } _ { \mathcal { H } } : \mathcal { T } _ { \mathcal { H } } \times \mathbb { R } ^ { + }  \mathcal { Q } _ { \mathcal { H } } \times \mathrm { S E } ( 3 ) ,\tag{10}
$$

which maps a grasp type τ and a grasp width w to a hand configuration $\pmb q ^ { * }$ and the corresponding Principal Closing Frame (PCF) transformation $^ \mathrm { b } { \pmb T } _ { \mathrm { p c f } }$ that denotes the pose matrix of the PCF with respect to the hand base frame, as shown in Fig. 4.

To construct this lookup table offline, we discretize the grasp width into a sequence $\{ w _ { l } \}$ and solve for the joint configuration $\mathbf { \Delta } q _ { l , \tau } ^ { * }$ of each grasp type τ :

$$
\begin{array} { r l } & { \pmb { q } _ { l , \tau } ^ { * } = \underset { \pmb { q } } { \arg \operatorname* { m i n } } ~ \left\| \mathcal { I } _ { w } ( \pmb { q } , \mathcal { P } _ { c } ) - w _ { l } \right\| _ { 2 } ^ { 2 } + \gamma \left\| \pmb { q } - \pmb { q } _ { l - 1 , \tau } ^ { * } \right\| _ { 2 } ^ { 2 } , } \\ & { \qquad \quad s . t . \ q _ { \operatorname* { m i n } } \leq \pmb { q } \leq q _ { \operatorname* { m a x } } , \ \mathbf { g } _ { \tau } ( \pmb { q } ) = \mathbf { 0 } . } \end{array}\tag{11}
$$

The recursion is initialized with ${ \pmb q } _ { 0 , \tau } ^ { * } = { \pmb q } _ { \mathrm { i n i t } , \tau }$ . Here, $\mathcal { I } _ { w } ( \pmb { q } , \mathcal { P } _ { c } )$ computes the grasp width from a virtual plane constructed by transforming a set of predefined contact points $\mathcal { P } _ { c }$ through forward kinematics. In Eq. (11), the first term minimizes the deviation from the target width, while the second term regularizes the solution toward the previous discretized configuration to ensure kinematic smoothness. The constraint g<sub>τ</sub> $( \pmb q ) = \mathbf 0$ enforces taxonomy-specific kinematic constraints for grasp type τ . After solving $\mathbf { \Delta } \mathbf { q } _ { l , \tau } ^ { * }$ , we analytically compute the corresponding PCF transformation ${ ^ { \mathrm { b } } T } _ { \mathrm { p c f } , l , \tau }$ from the same contact plane. Finally, all pairs $( w _ { l } , \pmb { q } _ { l , \tau } ^ { * } , \sqrt [ 6 ] { \mathbf { \mathrm { p c f } } _ { , l , \tau } } )$ are precomputed and stored as a lookup table for efficient inference.

During inference, we can reconstruct the world-frame PCF from a CGR $\begin{array} { r } { { \pmb a } _ { j } = ( { \pmb p } _ { c } , { \pmb v } _ { a } , { \pmb v } _ { b } , p , w ) _ { j } \colon } \end{array}$

$$
\begin{array}{c} \mathbf { { ^ w T } } _ { \mathrm { p c f } , j } = [ \begin{array} { c c c } { { \pmb v _ { b } } } & { { \pmb v _ { a } \times \pmb v _ { b } } } & { { \pmb v _ { a } } } \\ { { 0 } } & { { 0 } } & { { 0 } } \end{array} | { { \pmb p _ { c } } } + \frac { w } { 2 } { \pmb v _ { b } }  \\ { { \qquad \quad 0 } } & { { \qquad 0 } } \end{array} ] _ { j } .\tag{12}
$$

We retrieve the hand configuration $\boldsymbol { q } _ { k , \tau _ { k } } ^ { * }$ and hand-frame PCF transformation $^ { \mathrm { b } } T _ { \mathrm { p c f } , j , \tau _ { k } }$ according to $w _ { j }$ and $\tau _ { k }$ . The hand base pose and joint configuration are then computed as

$$
\begin{array} { r } { \pmb { T } _ { j , k } = \mathrm { \bf { ^ v } } \pmb { T } _ { \mathrm { p c f } , j } \left( \mathrm { ^ b } { \pmb { T } } _ { \mathrm { p c f } , j , \tau _ { k } } \right) ^ { - 1 } , \ \pmb { q } _ { j , k } = \pmb { q } _ { j , \tau _ { k } } ^ { * } . } \end{array}\tag{13}
$$

This produces the candidate grasp $\pmb { g } _ { j , k } = ( \pmb { T } _ { j , k } , \pmb { q } _ { j , k } )$ . We subsequently augment each candidate by sampling five discrete depths (0 m–0.05 m) along the approach direction $_ { v _ { a , k } }$ . Finally, physically infeasible poses are filtered out via a fast spherebased collision check using the Foam [75] library, yielding the final set of grasp candidates $\mathcal { G } _ { j , k } = \left\{ g _ { j , k } ^ { ( i ) } \right\}$

## D. Hand-Object Interaction Representation

A grasp is considered stable if it can resist arbitrary external wrenches. This state is defined by the force-closure condition, which must satisfy the following criteria [22]:

$$
\begin{array} { r l } & { G f = 0 , } \\ & { G { G } ^ { \top } > \epsilon I _ { 6 \times 6 } , } \\ & { f _ { i } ^ { \top } n _ { i } > \displaystyle \frac { 1 } { \sqrt { \mu ^ { 2 } + 1 } } \left. f _ { i } \right. , } \end{array}\tag{14}
$$

where $\mu$ is the friction coefficient, $f$ is the vector of contact forces acting at all contact points, G is the grasp matrix determined by the positions of the contact points, and $f _ { i }$ and $\mathbf { \nabla } n _ { i }$ are the contact force and the surface normal at the i-th contact point. From visual perception, the information we can extract is the positions and normals of potential contact points.

Consequently, we define the HOI representation $\mathcal { R } ^ { \mathcal { H O I } }$ as the set of coordinates and normals of the contact points at the hand-object interface. Specifically, given a grasp candidate $\pmb { g } _ { j , k } ^ { ( i ) }$ and the hand specification, we first extract the hand points $\mathcal { P _ { H } } = \{ \pmb { p } _ { n _ { h } } ^ { h } \} _ { n _ { h } = 1 } ^ { N _ { h } }$ and normals $\mathcal { N } _ { \mathcal { H } } = \{ \pmb { n } _ { n _ { h } } ^ { h } \} _ { n _ { h } = 1 } ^ { N _ { h } }$ via forward kinematics. Analogously, the object points and normals are denoted as $\mathcal { P } _ { \mathcal { O } } = \stackrel { - } { \{ \pmb { p } _ { n _ { o } } ^ { o } \} } _ { n _ { o } = 1 } ^ { N _ { o } }$ and $\dot { \mathcal { N } } _ { \mathcal { O } } = \{ \pmb { n } _ { n _ { o } } ^ { o } \} _ { n _ { o } = 1 } ^ { N _ { o } }$ . To identify contact regions, we compute the Euclidean distance from each object point to the hand surface:

![](images/e6c689e95e8d5d1c7d6b83c5144ec30d696a21bfcccf1f42351c2f87de703058.jpg)  
Fig. 6: Spatial prior module. Given multi-view RGB-D observations, the spatial prior module leverages the DINOv3 [7] features to construct scene semantic representations $\mathcal { R } ^ { S }$ and predict scene-level CGRs $A ^ { s }$ . The predicted CGRs provide spatially feasible contact candidates for the base grasp policy to perform clutter-aware grasp planning.

$$
d _ { n _ { o } } = \operatorname* { m i n } _ { \boldsymbol { n } _ { h } } \left\| \boldsymbol { p } _ { n _ { o } } ^ { o } - \boldsymbol { p } _ { n _ { h } } ^ { h } \right\| _ { 2 } .\tag{15}
$$

Points whose distance is below a proximity threshold $\delta _ { d }$ are treated as contact pairs. The valid contact-pair index set is

$$
\mathcal { X } = \left\{ ( n _ { o } , n _ { h } ^ { * } ) \ | \ d _ { n _ { o } } < \delta _ { d } , \ n _ { h } ^ { * } = \arg \operatorname* { m i n } _ { n _ { h } } \left. \left. p _ { n _ { o } } ^ { o } - p _ { n _ { h } } ^ { h } \right. \right. _ { 2 } \right\} .\tag{16}
$$

The HOI representation is then constructed as

$$
\begin{array} { r } { \mathcal { R } ^ { \mathcal { H O T } } = \left\{ \left( p _ { n _ { o } } ^ { o } , n _ { n _ { o } } ^ { o } , p _ { n _ { h } } ^ { h } , n _ { n _ { h } } ^ { h } \right) \vert \left( n _ { o } , n _ { h } \right) \in \mathcal { X } \right\} . } \end{array}\tag{17}
$$

This representation provides a unified evaluation interface across different hand embodiments.

## E. Hand-Agnostic Grasp Evaluation

This subsection instantiates the grasp decision model $\mathcal { D } ( \cdot )$ in Eq. (7). We implement $\mathcal { D } ( \cdot )$ as a lightweight point-based neural network that predicts a scalar stability score from $\mathcal { R } ^ { \mathcal { H O I } }$

As illustrated in Fig. 5, we adopt an object-centric formulation to construct the HOI representation. For each valid contact pair $( n _ { o } , n _ { h } )$ , the object contact point $\pmb { p } _ { n _ { o } } ^ { o } \in \mathbb { R } ^ { 3 }$ serves as the spatial coordinate, while the remaining geometric attributes form the point-wise feature vector $[ { \pmb n } _ { n _ { o } } ^ { o } , \bar { { \pmb p } } _ { n _ { h } } ^ { h } , { \pmb n } _ { n _ { h } } ^ { h } ] \in \mathbb { R } ^ { 9 }$ . The network takes this feature point cloud $\mathcal { P } _ { \mathcal { H O I } } \in \mathbb { R } ^ { | \mathcal { X } | \times ( 3 + 9 ) }$ as input. Subsequently, a PointTransformer [65] encoder serves as a sparse backbone to extract local geometric embeddings $\mathcal { U } ^ { \mathrm { g e o } } \in \mathbb { R } ^ { | \mathcal { X } | \times C _ { \mathrm { g c o } } }$ from this point cloud. To handle the variable number of contacts, we introduce a Contact-Aware Multi-Pooling (CAMP) mechanism. It computes max, mean, and standard-deviation statistics, denoted as $\begin{array} { r } { \pmb { u } _ { \mathrm { m a x } } , \pmb { u } _ { \mathrm { m e a n } } . } \end{array}$ , and $\mathbf { \delta } \mathbf { \pmb { u } } _ { \mathrm { s t d } }$ and dynamically fuses them through a learnable weight vector:

$$
{ \pmb w } = \mathrm { S o f t m a x } \left( \psi \left( \left[ { \pmb u } _ { \mathrm { m a x } } , { \pmb u } _ { \mathrm { m e a n } } , { \pmb u } _ { \mathrm { s t d } } , \phi ( | { \pmb X } | ) \right] \right) \right) ,\tag{18}
$$

where $\phi ( \cdot )$ and $\psi ( \cdot )$ are Multi-layer Perceptrons (MLPs) that encode the contact cardinality and project the fused features.

The global contact descriptor is

$$
z = w _ { 1 } \pmb { u } _ { \mathrm { m a x } } + w _ { 2 } \pmb { u } _ { \mathrm { m e a n } } + w _ { 3 } \pmb { u } _ { \mathrm { s t d } } .\tag{19}
$$

Finally, an MLP maps z to a scalar stability score:

$$
\beta = \sigma \left( \mathrm { M L P } ( z ) \right) ,\tag{20}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. To train the grasp decision model, we use the Binary Cross-Entropy (BCE) loss, with the optimization objective defined as:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \left[ y _ { i } \log ( \beta _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - \beta _ { i } ) \right] ,\tag{21}
$$

where $y _ { i } \in \{ 0 , 1 \}$ represents the ground truth label, and $\beta _ { i }$ denotes the predicted probability for the i-th sample.

## V. FOUNDATION PRIORS FOR ADAPTIVE GRASPING

This section instantiates the transformation function $f _ { \tau } ( \cdot )$ with composable foundation priors. We consider three tasks with increasing complexity: grasping in cluttered scenes, functional grasping in cluttered scenes, and functional grasping in dynamic cluttered scenes. For these tasks, the transformation function is dynamically configured as

$$
\begin{array} { r } { f _ { \tau } = \left\{ \begin{array} { l l } { \mathcal { M } _ { \mathrm { s } } , } & { \mathrm { c l u t t e r e d ~ g r a s p i n g , } } \\ { \mathcal { M } _ { \mathrm { c } } \circ \mathcal { M } _ { \mathrm { s } } , } & { \mathrm { f u n c t i o n a l ~ g r a s p i n g , } } \\ { \mathcal { M } _ { \mathrm { t } } \circ \mathcal { M } _ { \mathrm { c } } \circ \mathcal { M } _ { \mathrm { s } } , } & { \mathrm { d y n a m i c ~ g r a s p i n g . } } \end{array} \right. } \end{array}\tag{22}
$$

Here, $\mathcal { M } _ { \mathrm { s } } , \mathcal { M } _ { \mathrm { c } } ,$ and $\mathcal { M } _ { \mathrm { t } }$ denote the spatial-prior (Sec. V-A), cognitive-prior (Sec. V-B), and temporal-prior (Sec. V-C) modules, respectively, all of which can be flexibly instantiated by diverse composable foundation priors.

## A. Spatial Prior for Cluttered Grasping

For grasping in cluttered scenes, where multiple objects are densely stacked in an unorganized manner, the task context consists of the multi-view scene observations S:

$$
\begin{array} { r } { \mathcal { C } _ { \mathrm { s } } = \{ \boldsymbol { S } \} , \boldsymbol { S } = \{ I _ { m } , D _ { m } \} _ { m = 1 } ^ { N _ { v } } , } \end{array}\tag{23}
$$

where $I _ { m }$ and $D _ { m }$ denote the RGB image and depth map from the m-th camera view. Given the scene observations, the spatial

![](images/a82980620dcecf49f23a8433c2d7015ebb5cefcdb7d33a68d9d3d03adc3652cf.jpg)  
Fig. 7: Cognitive prior module. Given scene observations and a language instruction, the cognitive prior module uses an LLM with RAG [23] and CoT [24] mechanism to infer the target object, functional part, and hand-compatible grasp types. It then employs SAM3 [25] and 3D lifting to align these inferred results with scene representations, producing the target object geometry O, object-level semantics $\mathcal { R } ^ { \mathcal { O } }$ , and functional CGRs $\mathcal { A } ^ { \mathcal { O } }$ . These outputs, together with $\tau _ { \mathcal { H } }$ , provide the task-conditioned interface for the base grasp policy to perform language-guided functional grasping.

prior module $\mathcal { M } _ { \mathrm { s } } ( \cdot )$ extracts scene semantic representations $\mathcal { R } ^ { s }$ and scene-level CGR candidates $A ^ { s }$ :

$$
\mathcal { M } _ { \mathrm { s } } : \mathcal { S } \mapsto \left( \mathcal { R } ^ { s } , \mathcal { A } ^ { s } \right) ,\tag{24}
$$

where $\mathcal { R } ^ { s } = \{ r _ { n _ { s } } \} _ { n _ { s } = 1 } ^ { | \mathcal { R } ^ { s } | }$ and $\mathcal { A } ^ { S } = \{ a _ { j _ { s } } \} _ { j _ { s } = 1 } ^ { | \mathcal { A } ^ { S } | }$ . These outputs provide the geometry and feasible contact primitives required for grasping in clutter. The pipeline is depicted in Fig. 6.

1) Scene Semantics Construction: To construct point-wise scene semantic representations, each RGB image is fed into the DINOv3 [7] encoder to extract pixel-level semantic feature maps, denoted as $\mathcal { W } ^ { \mathrm { s e m } } ~ = ~ \{ w _ { m } ^ { \mathrm { \bar { s e m } } } \} _ { m = 1 } ^ { N _ { v } }$ . Here, $w _ { m } ^ { \mathrm { s e m } } \in$ $\mathbb { R } ^ { \tilde { H } \times \tilde { W } \times C _ { \mathrm { s e m } } }$ denotes the semantic feature map of the m-th view, where H and W are the image height and width, and $C _ { \mathrm { s e m } }$ is the semantic feature dimension.

We then lift these 2D semantic features into 3D space by using an implicit field function $\mathcal { F } ( \cdot \mid \mathcal { W } )$ [76]. This function maps an arbitrary 3D point to a feature vector by querying dense multi-view image features:

$$
\mathbf { \boldsymbol { u } } = \mathcal { F } ( \mathbf { \boldsymbol { p } } \mid \mathcal { W } ) ,\tag{25}
$$

where $\pmb { p } \in \mathbb { R } ^ { 3 }$ denotes a spatial point and $\textbf { \em u } \in \mathbb { R } ^ { C }$ is its corresponding feature vector. For each point $\pmb { p } _ { n _ { s } } ^ { s } \in \mathbb { R } ^ { 3 }$ in the scene point cloud $\mathcal { P } _ { S } \in \mathbb { R } ^ { N _ { s } \times 3 }$ , we query the semantic implicit field constructed from $\mathcal { W } ^ { \mathrm { s e m } }$ to obtain its semantic descriptor:

$$
\pmb { r } _ { n _ { s } } = \mathcal { F } \left( \pmb { p } _ { n _ { s } } ^ { s } \ | \ \mathcal { W } ^ { \mathrm { s e m } } \right) .\tag{26}
$$

Aggregating the semantic descriptors of all scene points yields the scene semantic representations $\mathcal { R } ^ { S }$

2) Scene-level CGRs Prediction: To predict scene-level CGRs from the multi-view scene observations, we adopt the PyTorch implementation of Contact-GraspNet<sup>1</sup>. We train this network on the GraspClutter6D [27] dataset, retaining the original hyperparameter settings. Given the scene point cloud $\mathcal { P } _ { \mathcal { S } }$ and semantic representations $\mathcal { R } ^ { S }$ , the network produces dense per-point CGRs $\{ a _ { n _ { s } } \}$ . During inference, we reconstruct the scene-level CGR set by filtering these dense predictions with the contact probability threshold $\delta _ { c } \mathbf { : }$

$$
\mathcal { A } ^ { S } = \{ \mathbf { { a } } _ { n _ { s } } \ | \ p _ { n _ { s } } > \delta _ { c } \} = \{ \mathbf { { a } } _ { j _ { s } } \} .\tag{27}
$$

Together with the scene geometry and all available grasp types, these CGRs construct the task-conditioned interface for clutteraware grasping. This interface further serves as the input to the cognitive prior, where the most suitable grasp type and target region are selected according to the task instruction.

## B. Cognitive Prior for Functional Grasping

For functional grasping in cluttered scenes, the task context further includes a natural language instruction ℓ:

$$
{ \mathcal { C } } _ { \mathrm { c } } = \{ { \mathcal { S } } , { \mathcal { l } } \} .\tag{28}
$$

Given the multi-view scene observations and language instruction, the cognitive prior module $\mathcal { M } _ { \mathrm { c } } ( \cdot )$ infers the target object, functional part, and appropriate grasp types, and then aligns these constraints with 3D scene representations:

$$
\mathcal { M } _ { \mathrm { c } } : \left( \mathcal { S } , \ell , \mathcal { R } ^ { \mathcal { S } } , \mathcal { A } ^ { \mathcal { S } } \right) \mapsto \left( \mathcal { O } , \mathcal { T } _ { \mathcal { H } } , \mathcal { R } ^ { \mathcal { O } } , \mathcal { A } ^ { \mathcal { O } } \right) ,\tag{29}
$$

where $\mathcal { R } ^ { \mathcal { O } } = \{ r _ { n _ { o } } \} _ { n _ { o } = 1 } ^ { | \mathcal { R } ^ { \mathcal { O } } | }$ represents the set of object semantic representations extracted from the scene representations $\mathcal { R } ^ { S }$ while $\mathcal { A } ^ { \mathcal { O } } = \{ a _ { j _ { o } } \} _ { j _ { o } = 1 } ^ { | \mathcal { A } ^ { \mathcal { O } } | }$ denotes the functional CGRs aligned with the grounded target object and the functional part.

1) Grasp Functionality Inference: To understand human language instructions, we instantiate a parsing function Ψ(·)

with an LLM to infer the structured grasp specification:

$$
\left( \ell _ { \mathrm { o b j } } , \ell _ { \mathrm { p a r t } } , \mathcal { T } _ { \mathcal { H } } \right) = \Psi \left( S , \ell \right) ,\tag{30}
$$

where $\ell _ { \mathrm { o b j } }$ and $\ell _ { \mathrm { p a r t } }$ denote the language descriptors for the target object and its functional part. Direct inference of these physically grounded attributes is non-trivial due to the inherent lack of grasp reasoning in standard LLMs. To bridge this gap, we employ Seed-1.8<sup>2</sup> as the core reasoning agent, augmented with RAG [23] and CoT [24] mechanisms.

As shown in Fig. 7, we construct a grasp taxonomy database that encodes detailed attributes for predefined grasp types, including functional intent, active finger configurations, contact constraints, and use-case examples. By retrieving relevant knowledge from this database, we constrain the inference space and mitigate hallucinations in LLMs. Furthermore, we use CoT prompting to decompose the inference into task intent identification, functional part selection, and grasp requirement analysis, ultimately outputting $\ell _ { \mathrm { o b j } } , \ell _ { \mathrm { p a r t } } ,$ and $\tau _ { \mathcal { H } }$

2) Object Semantic Representation Refinement: Then, we ground the parsed object description $\ell _ { \mathrm { o b j } }$ in the scene semantic representations. To localize the target object, we perform visual grounding conditioned on the parsed object description, such as “the mug”. Specifically, we use Seed-1.8 to generate boundingbox annotations on multi-view RGB images. These annotations are then used as prompts for SAM3 [25] to extract multi-view instance masks, denoted as $\mathcal { W } ^ { \mathrm { i n s t } } = \{ \pmb { w } _ { m } ^ { \mathrm { i n s t } } \} _ { m = 1 } ^ { N _ { v } }$

We further lift the 2D instance masks into 3D space using the implicit field function defined in Eq. (25), yielding a pointwise instance probability $u _ { n _ { s } } ^ { \mathrm { i n s t } } \in [ 0 , 1 ]$ for each scene point. To obtain object semantic representations, we filter the scenelevel representations and retain only those associated with the grounded target instance:

$$
\mathcal { R } ^ { \mathcal { O } } \triangleq \big \{ { \pmb r } _ { n _ { s } } \in \mathcal { R } ^ { S } \ | \ u _ { n _ { s } } ^ { \mathrm { i n s t } } > \delta _ { \mathrm { i n s t } } \big \} = \{ { \pmb r } _ { n _ { o } } \} .\tag{31}
$$

The corresponding target object geometry O is extracted from the scene point cloud using the same grounded instance mask.

3) Functional CGRs Alignment: We further use the parsed part description $\ell _ { \mathrm { p a r t } }$ , such as “the handle”, to restrict grasp generation to the corresponding functional regions. Analogous to the instance grounding, we derive a point-wise functional probability ${ \pmb u } _ { n _ { s } } ^ { \mathrm { f u n } } \in [ 0 , 1 ]$ by identifying functional regions through visual grounding and 3D lifting. To extract task-relevant contact primitives, we filter the scene-level CGR set $\mathcal { A } ^ { s }$ by enforcing both instance-level and functional-level consistency:

$$
\mathcal { A } ^ { \mathcal { O } } \triangleq \left\{ \pmb { a } _ { n _ { s } } \in \mathcal { A } ^ { \mathcal { S } } \mid \pmb { u } _ { n _ { s } } ^ { \mathrm { i n s t } } > \delta _ { \mathrm { i n s t } } \wedge \pmb { u } _ { n _ { s } } ^ { \mathrm { f u n } } > \delta _ { \mathrm { f u n } } \right\} = \left\{ \pmb { a } _ { j _ { o } } \right\}\tag{32}
$$

This set provides functionally relevant and spatially feasible CGR candidates for grasp synthesis. Together with the object geometry O and the inferred grasp types $\tau _ { \mathcal { H } }$ , it forms the taskconditioned interface for language-guided functional grasping.

## C. Temporal Prior for Dynamic Grasping

For functional grasping in dynamic cluttered scenes, the task context comprises a continuous observation sequence and a

![](images/3d4c032937cab4268daa22732d23e1d8dd1ce2f9c4a8bc08e6e5d24ccc6d8077.jpg)  
Fig. 8: Temporal prior module. This module integrates the temporal mask-tracking mechanism of SAM3 [25] with the cross-frame consistency of DINOv3 [7] features to estimate the target’s rigid motion. The estimated motion is then used to update the structured interface, enabling the base grasp policy to perform temporally consistent dynamic grasping.

language instruction:

$$
\mathcal { C } _ { \mathrm { t } } = \{ S _ { 0 : t } , \ell \} .\tag{33}
$$

At the initial time step $t = 0 ,$ , the spatial and cognitive priors construct the initial grasp interface:

$$
\begin{array} { r } { \mathcal { T } _ { 0 } = \left( \mathcal { O } _ { 0 } , \mathcal { A } _ { 0 } ^ { \mathcal { O } } , \mathcal { T } _ { \mathcal { H } } \right) . } \end{array}\tag{34}
$$

The temporal prior module $\mathcal { M } _ { \mathrm { t } } ( \cdot )$ updates this interface across continuous observations to maintain temporally consistent grasp tracking:

$$
\mathcal { M } _ { \mathrm { t } } : ( \mathbb { Z } _ { 0 } , S _ { 1 : t } ) \mapsto \mathbb { Z } _ { t } = ( \mathcal { O } _ { t } , \mathcal { A } _ { t } , \mathcal { T } _ { \mathcal { H } } ) .\tag{35}
$$

Under the rigid-body assumption, if a grasp is stable at the initial time step, it remains stable at subsequent time steps provided that the hand strictly tracks the object’s rigid motion. Therefore, we instantiate $\mathcal { M } _ { \mathrm { t } } ( \cdot )$ to estimate the realtime SE(3) transformation of the target object. The estimated transformation is subsequently applied to update the grasp interface, providing temporally consistent inputs for the base grasp policy. The pipeline is depicted in Fig. 8.

1) Object Tracking Consistency: To maintain object tracking consistency, we leverage the mask-based tracking mechanism of SAM3 [25]. Instead of performing independent visual grounding at each time step, we initialize the target instance mask ${ \mathcal { W } } _ { 0 } ^ { \mathrm { i n s t } }$ at the initial time step and propagate it over time. This mask-based tracking strategy effectively mitigates sequential segmentation instability and semantic drift, ensuring that the tracked instance mask $\mathcal { \dot { W } } _ { t } ^ { \mathrm { i n s t } }$ remains strictly consistent with the target object throughout the grasping process. Subsequently, ${ \mathcal { O } } _ { t }$ is extracted from the real-time scene observation and $\mathcal { W } _ { t } ^ { \mathrm { i n s t } }$

2) Object Pose Estimation: To estimate the target’s rigid transformation matrix $\pmb { T } _ { 0  t } ^ { \mathcal { O } } \in \mathrm { S E } ( 3 )$ , we construct a semantic consistency optimization problem [76]. Specifically, we denote the object’s point cloud at the initial time step as $\mathcal { P } _ { \mathcal { O } , 0 } = \{ p _ { n _ { o } , 0 } ^ { o } \} _ { n _ { o } = 1 } ^ { N _ { o } }$ , and the corresponding set of semantic descriptors as $\mathcal { R } _ { 0 } ^ { \mathcal { O } } = \{ r _ { n _ { o } , 0 } ^ { o } \} _ { n _ { o } = 1 } ^ { N _ { o } }$ . At time step t, we compute the optimal rotation $R _ { 0  t } ^ { \ast }$ and translation $\mathbf { \Delta } t _ { 0  t } ^ { \ast }$ that minimize the discrepancy between the initial semantic descriptors and the features queried from the current scene’s implicit field $\mathcal { F } ( \cdot \mid \mathcal { W } _ { t } ^ { \mathrm { s e m } } )$ . The optimization objective is defined as:

![](images/94a2a616673c37866da3195a0181d24e5ef87837339da3c611021c72b3a0a120.jpg)  
Fig. 9: Data collection in simulation. To train and evaluate the grasp decision model, we collect synthetic data via trial and error in Isaac Sim. The dataset comprises 4.4 million annotated grasping trials across three diverse robotic hands.

$$
( R _ { 0  t } ^ { * } , t _ { 0  t } ^ { * } ) = \operatorname * { a r g m i n } _ { R _ { 0  t } , t _ { 0  t } } \sum _ { n _ { o } = 1 } ^ { N _ { o } } \| \mathcal { F } ( \hat { p } _ { n _ { o } , t } ^ { o } \mid \mathcal { W } _ { t } ^ { \mathrm { s e m } } ) - r _ { n _ { o } , 0 } ^ { o } \| _ { 2 } ^ { 2 }
$$

$$
\hat { p } _ { n _ { o } , t } ^ { o } = R _ { 0  t } { \pmb { p } } _ { n _ { o } , 0 } ^ { o } + { \pmb { t } } _ { 0  t } .\tag{36}
$$

Since the implicit field function $\mathcal { F } ( \cdot \mid \mathcal { W } )$ is differentiable, this optimization can be solved using gradient-based methods. In practice, we further incorporate rigid-body constraints and distance regularization to improve robustness and stability.

Consequently, given the estimated target object’s rotation and translation $( R _ { 0  t } ^ { * } , t _ { 0  t } ^ { * } )$ , the real-time contact interface can be directly computed from the initial contact reference:

$$
\mathcal { A } _ { t } = \{ [ \begin{array} { c c c c } { \begin{array} { c c c c } & { R _ { 0  t } ^ { * } } & & { t _ { 0  t } ^ { * } } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} } \end{array} ] { \pmb a } \mid { \pmb a } \in \mathcal { A } _ { 0 } \} .\tag{37}
$$

In practice, we observe that the pose-estimation error remains within an admissible tolerance. Therefore, to maximize system efficiency, we bypass the base grasp policy and directly update the initial grasp pose $g _ { 0 } = ( T _ { 0 } , q _ { 0 } )$ via the rigid transformation:

$$
\pmb { g } _ { t } = ( [ \begin{array} { c c c c } { { } } & { { \pmb { R } _ { 0  t } ^ { * } } } & { { } } & { { \pmb { t } _ { 0  t } ^ { * } } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \end{array} ] \pmb { T } _ { 0 } , \pmb { q } _ { 0 } ) .\tag{38}
$$

## VI. SIMULATED EXPERIMENTS

In this section, we systematically validate the key components of AdaRoboVLG in simulation. We first introduce the data collection and training setup (Sec. VI-A), and then evaluate the base policy for generalizable grasp synthesis (Sec. VI-B), the spatial prior for cluttered grasping (Sec. VI-C), the cognitive prior for functional grasping (Sec. VI-D), and the temporal prior for dynamic grasping (Sec. VI-E).

## A. Data Collection and Training Implementation

1) Data Collection and Annotation: To train the grasp decision model $\mathcal { D } ( \cdot )$ , we collect data via trial and error on single-object grasping in a physics-based simulator (see Fig. 9).

All object models are sourced from the training sets of the GraspNet-1Billion [28] and GraspClutter6D [27] benchmarks. Our data generation pipeline consists of two stages.

In the first stage, we densely sample CGR candidates on the object surface. Candidates satisfying basic force-closure criteria are then mapped into executable hand poses via $\mathcal { M } _ { \mathcal { H } } ( \cdot )$ defined in Sec. IV-C. In the second stage, we execute these proposals in Isaac Sim to evaluate grasp stability, with the friction coefficient set to 0.5. After the hand closes to establish contact, we apply a 0.5 N external perturbation force to the object along six axisaligned directions. A grasp is labeled as successful $( y = 1 )$ only if the object’s translational displacement remains under 0.05 m and rotational deviation under $3 0 ^ { \circ } ;$ ; otherwise, it is labeled as a failure $( y = 0 )$ . We apply this pipeline to three diverse robotic hands (DH3, Allegro, and Inspire). For each hand, we collect 1,000 grasping trials for every combination of object and grasp type, balanced between positive and negative examples. With 200 training objects and the grasp types available for each hand (4 for DH3, 10 for Allegro, and 8 for Inspire), this process yields 4.4 million annotated grasping trials in total. We partition the dataset into training and validation sets at a 10:1 ratio.

2) Training Implementation: We implement the grasp decision model in PyTorch on Ubuntu 20.04, and train it on a server with dual Intel Xeon 6330 28-core CPUs, four NVIDIA GeForce RTX 4090 GPUs, and 256 GB of RAM. We use the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a weight decay of 0.05. The model is trained for 30 epochs with a batch size of 512, using the binary cross-entropy loss and a cosine annealing learning rate scheduler.

Throughout training and inference, the feasible grasp types for each hand follow the setup in AnyDexGrasp [71]. To enhance robustness against real-world observation noise, we inject Gaussian perturbations $( \sigma = 0 . 0 0 5 \mathrm { m }$ for coordinates, $\sigma = 0 . 0 2$ for normals) during model training. During inference, a grasp is considered stable if the predicted score $\beta > 0 . 5$

## B. Evaluation of the Base Policy for General Grasp Synthesis

The base policy consists of two stages: hand-specific grasp mapping and hand-agnostic grasp evaluation. The first stage is instantiated as an explicit function and supports deployment across end-effectors with two to five fingers, as demonstrated in subsequent experiments. Therefore, we focus on evaluating the proposed HOI representation and grasp decision model on learning efficiency and cross-hand grasp evaluation.

1) Learning Efficiency: To evaluate the impact of different input representations on the learning efficiency of the decision model, we train the model on the Allegro hand using the synthetic grasping data constructed in Sec. VI-A. We compare three input paradigms: object-centric HOI, hand-centric HOI, and an explicitly separated hand-object point cloud representation. The former two encode the HOI relationship from object-centric and hand-centric perspectives, respectively, while the latter independently encodes the contact point clouds of the hand and object and then concatenates their features. All compared variants share the same network architecture, and convergence behavior is evaluated through the training loss curves.

As illustrated in Fig. 10a, compared with the separated representation, the HOI representations lead to faster network convergence and lower final training losses. This improvement is mainly attributed to the ability of the HOI representation to abstract the complex force-closure condition into a compact feature point cloud, thereby substantially reducing the learning difficulty for the network.

![](images/e1a63bb44c6a5e57e260ee2bd2cf87e3883a6fd376df9050f9dc03ff782e9cce.jpg)

![](images/d547331976040f4cb16ccdd4892417fb6bfc5cf6987a1456345901d8d5bf6b35.jpg)  
(a) Training convergence.  
(b) Cross-hand accuracy.

Fig. 10: Comparative experiments of the grasp decision model. (a) Training loss of the grasp decision model using different representations. (b) Grasp quality prediction accuracy of the grasp decision model trained on different hand data.  
![](images/6ffa324751611cc0eeb68d877e80baf887b12dfa324531f5e8b2f99e7dc41187.jpg)  
Fig. 11: t-SNE visualization of HOI representations. This figure visualizes the distributions of stable HOI representations across three different robotic hands (DH3, Allegro, and Inspire).

2) Cross-hand Generalization: To evaluate whether the HOI representation enables cross-hand generalization of the decision model, we train the decision model using single-hand data and the combined data from all three hands, respectively. The grasp quality prediction accuracy is then evaluated across the three robotic hands.

As shown in Fig. 10b, the model jointly trained on data from all three robotic hands achieves accuracies of 82.95%, 88.02%, and 91.07% on the DH3, Allegro, and Inspire, respectively, outperforming the single-hand training configurations overall. This result demonstrates that the HOI representation successfully facilitates shared stability judgment across different robotic hands, thereby enhancing cross-hand generalization capabilities.

Furthermore, we provide a t-SNE visualization of stable HOI representations across the three robotic hands (see Fig. 11). The feature distributions exhibit substantial overlap in the shared embedding space, with no obvious clustering induced by kinematic or structural differences. This observation is consistent with the hand-agnostic property of the HOI representation and provides qualitative evidence for its suitability for joint multi-hand training and cross-hand generalization.

## C. Evaluation of Spatial Prior for Clutter-Aware Grasping

This experiment evaluates the effectiveness of the spatial prior in transforming multi-view observations into scene-level contact constraints for grasp synthesis, supporting spatially feasible grasp generation under different levels of scene clutter.

1) Benchmark Setting: We adopt the simulated benchmark proposed in DexGraspNet 2.0 [26], which is built upon the GraspNet-1Billion dataset [28], and comprises 450 scenes and 88 diverse objects. To evaluate the model performance under different clutter levels, the test scenes are categorized into three difficulty splits based on object density:

• Dense consists of 90 scenes from the test set in GraspNet 1Billion [28], with each scene containing 8-11 objects.

• Random comprises 180 scenes by augmenting the dense split twice, where each scene retains 1-10 random objects.

• Loose contains 180 scenes by augmenting the dense split twice, with 1-2 random objects remaining in the scene.

All experiments are conducted in Isaac Gym, where the friction coefficient is set to $\mu = 0 . 2$

2) Evaluation Metrics: We report the grasp success rate, following the same evaluation protocol as DexGraspNet 2.0 [26]. Specifically, a grasp trial is considered a success if the highestscoring grasp pose lifts the object off the table without any initial collision with the table or surrounding objects.

3) Baseline Models: We compare AdaRoboVLG against AnyDexGrasp [71] and its variant, AnyDexGrasp<sup>∗</sup>. AnyDex-Grasp is a two-stage dexterous grasping framework that first predicts a hand-agnostic grasp representation to generate handspecific grasp candidates, and subsequently employs a handspecific decision model to select executable grasps. Because it supports the same three robotic hands and adopts a grasp taxonomy consistent with ours, we construct AnyDexGrasp<sup>∗</sup> as a variant augmented with our grasp decision model.

4) Experimental Details: AnyDexGrasp, AnyDexGrasp<sup>∗</sup>, and AdaRoboVLG are evaluated in a zero-shot manner on three robotic hands spanning three to five fingers (DH3, Allegro, and Inspire), without any re-training. For AnyDexGrasp and AnyDexGrasp<sup>∗</sup>, we use the officially released representation model to generate hand-agnostic grasp candidates. AnyDex-Grasp scores these candidates with its original decision model, whereas AnyDexGrasp<sup>∗</sup> uses our grasp decision model for candidate ranking. During inference, the highest-scoring grasp pose is selected for execution.

5) Experimental Results: As shown in Tab. I, AdaRoboVLG exhibits strong zero-shot and cross-hand grasp synthesis in cluttered scenes. Compared with AnyDexGrasp, it achieves the highest success rates on the random and loose splits for all three hands, with average success-rate improvements of 3.0% and 9.2%. Although AnyDexGrasp performs slightly better on the dense split, AdaRoboVLG shows more stable performance across different clutter levels. This primarily stems from compact CGRs and the human-inspired grasp taxonomy, which significantly reduce the grasp search space, thereby enabling better adaptation to scenes of varying complexity.

TABLE I: Performance comparison of multi-finger grasp pose generation in clutter on the DexGraspNet 2.0 [26] benchmark. We report the grasp success rates across three clutter difficulty levels (i.e., Dense, Random, and Loose), comparing various methods across different robotic hands. The best results are highlighted in bold and underlined.
<table><tr><td rowspan="2">Methods</td><td rowspan="2">Hands</td><td colspan="3">Success Rate (%)</td></tr><tr><td>Dense</td><td>Random</td><td>Loose</td></tr><tr><td>AnyDexGrasp [71]</td><td>DH3</td><td>91.2</td><td>85.5</td><td>78.2</td></tr><tr><td>AnyDexGrasp</td><td>DH3</td><td>87.8</td><td>86.4</td><td>85.7</td></tr><tr><td>AdaRoboVLG (ours)</td><td>DH3</td><td>90.3</td><td>89.5</td><td>88.6</td></tr><tr><td>AnyDexGrasp [71]</td><td>Allegro</td><td>90.5</td><td>86.6</td><td>79.4</td></tr><tr><td>AnyDexGrasp</td><td>Allegro</td><td>88.2</td><td>87.2</td><td>83.8</td></tr><tr><td>AdaRoboVLĠ (ours)</td><td>Allegro</td><td>89.5</td><td>88.2</td><td>85.9</td></tr><tr><td>AnyDexGrasp [71]</td><td>Inspire</td><td>92.7</td><td>85.0</td><td>76.1</td></tr><tr><td>AnyDexGrasp</td><td>Inspire</td><td>86.1</td><td>85.4</td><td>83.9</td></tr><tr><td>AdaRoboVLĠ (ours)</td><td>Inspire</td><td>88.1</td><td>88.5</td><td>86.7</td></tr></table>

The comparison with AnyDexGrasp<sup>∗</sup> further validates our grasp decision model. Given the same grasp candidates as Any-DexGrasp, AnyDexGrasp<sup>∗</sup> improves the average success rate by 0.6% on the random split and 6.6% on the loose split. Moreover, AdaRoboVLG consistently outperforms AnyDexGrasp<sup>∗</sup> across all three hands, indicating that both the spatial prior and the hand-agnostic grasp decision model jointly contribute to robust multi-finger grasp synthesis.

## D. Evaluation of Cognitive Prior for Functional Grasping

This experiment evaluates the effectiveness of the cognitive prior in converting language instructions into functional grasp constraints for grasp synthesis. Because current simulation benchmarks fall short in jointly evaluating grasp functionality inference, target grounding, and physical grasp execution, we assess this prior module from two complementary experiments: grasp functionality inference on open-set instructions and language-guided target grasping in simulated clutter.

1) Grasp Functionality Inference: To evaluate grasp functionality inference, we construct an open-set instruction dataset based on instruction templates [29], [77]. The dataset contains 125 image-instruction pairs spanning 57 manipulation tasks and 80 object categories, with human-annotated ground-truth functional grasp intents. We report functional part and grasp type accuracy as evaluation metrics.

We conduct an ablation study to evaluate the individual contributions of the CoT and RAG mechanisms in the cognitive reasoning module. Tab. II demonstrates that integrating RAG and CoT techniques substantially improves the model’s grasp functionality inference accuracy. This efficient design eliminates the reliance on large-scale language-action paired data, thereby substantially reducing the overall learning cost.

2) Language-Guided Target Grasping: To evaluate languageguided target grasping, we construct a simulated benchmark in Isaac Sim using the test splits of GraspNet-1Billion [28] and GraspClutter6D [27]. For each scene, objects are initialized with their original poses and assigned a mass of 1.0 kg and a friction coefficient of 0.5. The experimental platform consists of a UR5 robotic arm equipped with three different robotic hands: DH3, Allegro, and Inspire. Five static cameras are used to capture the scene, including four surrounding cameras at a 45<sup>◦</sup> elevation angle and one top-down camera. To enable language-guided grasping, we use Qwen3.5-VL<sup>3</sup> to generate language instructions offline by selecting the top five graspable objects in each scene and synthesizing commands following the templates introduced in [82]. This process produces five predefined language-guided grasping tasks for each scene.

TABLE II: Experimental results of grasp functionality inference on open-set instructions. We report the accuracy of grasp functionality inference under different settings. The best results are highlighted in bold and underlined.
<table><tr><td>Method</td><td>Functional Part Accuracy ↑</td><td>Grasp Type Accuracy ↑</td></tr><tr><td>Base LLM</td><td>0.82</td><td>0.63</td></tr><tr><td>Base  $\mathrm { L L M } + \mathrm { C o T }$ </td><td>0.91</td><td>0.66</td></tr><tr><td>Base  $\mathrm { L L M } + \mathrm { R A G }$ </td><td>0.85</td><td>0.78</td></tr><tr><td>AdaRoboVLG (ours)</td><td>0.94</td><td>0.87</td></tr></table>

TABLE III: Experimental results of language-guided target grasping for multi-finger hands on the GraspClutter6D [27] and GraspNet-1Billion [28] datasets. We report the grasp success rates of three dexterous hands on two datasets. The best results are highlighted in bold and underlined.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Success Rate (%)</td></tr><tr><td>DH3</td><td>Allegro</td><td>Inspire</td></tr><tr><td></td><td colspan="3">GraspClutter6D (235 scenes)</td></tr><tr><td>D(R, O)-Grasp [50]</td><td>64.7</td><td>52.1</td><td>55.9</td></tr><tr><td>AnyDexGrasp [71]</td><td>82.5</td><td>74.1</td><td>79.3</td></tr><tr><td>AnyDexGrasp 本</td><td>82.9</td><td>75.9</td><td>81.4</td></tr><tr><td>AdaRoboVLG (ours)</td><td>84.9</td><td>76.4</td><td>82.6</td></tr><tr><td></td><td colspan="3">GraspNet-1Billion (90 scenes)</td></tr><tr><td>D(R, O)-Grasp [50]</td><td>66.2</td><td>53.3</td><td>57.3</td></tr><tr><td>AnyDexGrasp [71]</td><td>85.2</td><td>80.8</td><td>84.4</td></tr><tr><td>AnyDexGrasp *</td><td>86.6</td><td>81.2</td><td>85.0</td></tr><tr><td>AdaRoboVLG (ours)</td><td>88.9</td><td>82.2</td><td>86.9</td></tr></table>

Each task is executed once with each robotic hand. A trial is successful if the hand grasps the target object and lifts it to a height of 10 cm. We report the average grasp success rate separately for each hand.

We compare AdaRoboVLG with three baseline models: AnyDexGrasp [71], AnyDexGrasp<sup>∗</sup>, and D(R, O)-Grasp [50]. AnyDexGrasp and AnyDexGrasp<sup>∗</sup> have been introduced in Sec. VI-C3. D(R, O)-Grasp is a representative cross-hand dexterous grasping method designed for object-centric grasp synthesis. For fair comparison, all methods are combined with the same cognitive reasoning module as ours for language grounding and target segmentation. AnyDexGrasp and AnyDexGrasp<sup>∗</sup> follow the implementation detailed in Sec. VI-C3, utilizing the cognitive reasoning module to filter task-consistent grasp candidates based on the language instruction. For D(R, O)- Grasp, we train the model in a multi-hand setting using grasp samples synthesized via the DexGraspNet [83] pipeline on objects from the training sets of both benchmarks. During inference, the cognitive reasoning module first localizes the target object; subsequently, D(R, O)-Grasp generates grasp candidates from the segmented object point cloud and randomly selects a collision-free pose for execution. All methods are evaluated in a zero-shot manner without further fine-tuning, with all grasp types enabled.

TABLE IV: Experimental results of dynamic grasping on the GraspNet-1Billion [28] benchmark. We report the grasp tracking accuracy (MGTA), mean translation error $( e _ { \mathrm { t r a n s } } ) .$ , and mean rotation error $( e _ { \mathrm { r o t } } )$ across three difficulty splits (Seen, Similar, and Novel). The best results are highlighted in bold and underlined.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Seen</td><td colspan="3">Similar</td><td colspan="3">Novel</td></tr><tr><td>MGTA↑</td><td> $e _ { \mathrm { t r a n s } } ( \mathrm { c m } ) \downarrow$ </td><td> $e _ { \mathrm { r o t } } ( ^ { \circ } ) \downarrow$ </td><td>MGTA↑</td><td> $e _ { \mathrm { t r a n s } } ( \mathrm { c m } ) \downarrow$ </td><td> $e _ { \mathrm { r o t } } ( ^ { \circ } ) \downarrow$ </td><td>MGTA↑</td><td> $e _ { \mathrm { t r a n s } } ( \mathrm { c m } ) \downarrow$ </td><td> $e _ { \mathrm { r o t } } ( ^ { \circ } ) \downarrow$ </td></tr><tr><td>Nearest</td><td>-0.81</td><td>15.2</td><td>95.98</td><td>-0.80</td><td>15.4</td><td>97.61</td><td>-0.83</td><td>16.6</td><td>96.90</td></tr><tr><td>AnyGrasp [78]</td><td>-0.32</td><td>2.31</td><td>53.24</td><td>-0.30</td><td>2.68</td><td>57.59</td><td>-0.39</td><td>2.93</td><td>58.49</td></tr><tr><td>BundleTrack [79]</td><td>0.74</td><td>1.52</td><td>15.8</td><td>0.72</td><td>1.47</td><td>16.37</td><td>0.72</td><td>1.93</td><td>15.41</td></tr><tr><td>Target-referenced [80]</td><td>0.84</td><td>1.34</td><td>10.15</td><td>0.85</td><td>1.52</td><td>9.84</td><td>0.78</td><td>1.9</td><td>12.96</td></tr><tr><td>MotionGrasp [81]</td><td>0.99</td><td>1.1</td><td>0.86</td><td>0.99</td><td>1.1</td><td>0.84</td><td>0.99</td><td>1.07</td><td>1.12</td></tr><tr><td>AdaRoboVLG (ours)</td><td>1.0</td><td>0.22</td><td>3.57</td><td>1.0</td><td>0.23</td><td>3.68</td><td>1.0</td><td>0.28</td><td>3.28</td></tr></table>

![](images/51cabe7ea19c3a24bd6d35b6ba1d556309cc3672c5c62fd6f915136270b04934.jpg)

![](images/d72414668376994a02a275277ae67dcbcd39cb2ad60f1126f46f362b3aa1bf87.jpg)  
(a) Tracking errors for partial objects in Scene 0100.

![](images/82f94c292db4bc085ddc631936f7e049ad0b344557c82022c5b280f17ec64702.jpg)  
(b) Tracking visualization for partial objects in Scene 0100.  
Fig. 12: Long-horizon tracking results for representative objects in scene 0100 of the GraspNet-1Billion [28] dataset. (a) Continuous tracking errors for selected objects. (b) Image-space projections of tracked 3D semantic features.

As shown in Tab. III, AdaRoboVLG achieves the highest success rates on both datasets, outperforming AnyDexGrasp by 2.7% on GraspClutter6D and 2.5% on GraspNet-1Billion. Compared with $\mathcal { D } ( \mathcal { R } , \mathcal { O } )$ -Grasp, the average improvements are even more substantial, reaching 23.7% and 27.1%, respectively. These results demonstrate that LLM-based grounding and SAM3-based segmentation provide accurate target constraints, enabling the base policy to generate executable and semantically consistent grasp poses. The combination of the cognitive reasoning module and the base policy yields the best language guided target grasping performance.

## E. Evaluation of Temporal Prior for Dynamic Grasping

This experiment evaluates the capability of the temporal prior to convert continuous visual observations into temporally consistent motion constraints for grasp updates, thereby enabling robust grasp tracking.

1) Benchmark Setting: We conduct experiments using the GraspNet-1Billion [28] benchmark. This benchmark is collected by moving a camera around each scene, which effectively simulates relative object motion and is highly suitable for evaluating dynamic grasping. Each scene contains 256 RGB-D images captured from different viewpoints.

2) Evaluation Metrics: Following prior work [80], we employ Multiple Grasp Tracking Accuracy (MGTA) as the primary performance metric. It measures the alignment between the predicted grasps in the current frame and the ground-truth projections derived from the initial frame. It is defined as:

$$
\mathrm { M G T A } = 1 - \frac { \sum _ { t } ( \mathrm { F N } _ { t } + \mathrm { F P } _ { t } + \mathrm { I D S W } _ { t } ) } { \sum _ { t } \mathrm { G T } _ { t } } ,\tag{39}
$$

where False Negatives (FN) and False Positives (FP) represent missed matches and false associations in the grasp-trajectory association, respectively. ID Switching (IDSW) counts the instances where a grasp is erroneously assigned to a different object ID. An MGTA score of 1 indicates perfect tracking. Moreover, geometric precision is evaluated using the mean translation error $e _ { \mathrm { t r a n s } }$ and rotation error $e _ { \mathrm { r o t } }$ between the predicted grasps and the corresponding ground-truth projections.

3) Baseline Models: We compare AdaRoboVLG with five baseline models, comprising Nearest, AnyGrasp [78], Bundle-Track [79], Target-referenced [80], and MotionGrasp [81].

4) Experimental Details: Following the protocol in prior work [81], we use the data captured by the RealSense for training and evaluation. To train the baselines, we extract 250 video clips per scene by sequentially grouping 7 consecutive frames. For evaluation, we extract 17 clips from each test scene and use the first 7 frames of each clip for grasp tracking. In the baseline setup, the top-10 highest-scoring grasp candidates are selected from the initial frame for the model to track.

In contrast, our tracking module is entirely training-free. We formulate grasp tracking as a rigid object motion estimation problem. Consequently, geometric precision measurement is equivalent to evaluating the object pose estimation errors. Meanwhile, the MGTA metric is computed based on instancelevel tracking accuracy. For each test clip, we randomly select 2 to 3 objects for tracking. Given their masks in the initial frame, we predict their continuous poses across the video clip and report both the geometric errors and association metrics. In practice, to solve the optimization objective in Eq. (36), we employ the Adam optimizer [84] for 100 iterations.

5) Experimental Results: As shown in Tab. IV, our method demonstrates temporally consistent grasp tracking performance in dynamic scenarios. It achieves perfect association accuracy, the lowest translation error, and competitive rotation error. This stable tracking capability is enabled by integrating the robust mask tracking capability of SAM3 [25] with the cross-frame semantic consistency of DINOv3 [7] features.

We further conduct a long-horizon tracking evaluation on representative objects from Scene 0100 of the GraspNet-1Billion [28] benchmark. As illustrated in Fig. 12a, our method robustly tracks most objects (e.g., the scissors) with average translation and rotation errors around 0.5 cm and 5<sup>◦</sup>. However, as shown in Fig. 12b, tracking failures primarily emerge in two challenging scenarios: (1) highly symmetric objects (e.g., the strawberry) experience rotation estimation drift due to their rotational invariance, and (2) severely occluded objects with minimal initial visibility (e.g., the i toy airplane) lack sufficient discriminative semantic cues, leading to catastrophic tracking loss and translation deviation during viewpoint changes.

## VII. REAL-WORLD EXPERIMENTS

This section demonstrates the real-world applicability of AdaRoboVLG by progressively composing different foundation priors to address increasingly challenging grasping scenarios. After introducing the experimental setup (Sec. VII-A), we first evaluate the composition of spatial and cognitive priors for functional grasping in static cluttered scenes (Sec. VII-B). We then further increase task complexity by incorporating the temporal prior to handle functional grasping in dynamic cluttered environments (Sec. VII-C). Finally, we demonstrate the extensibility of AdaRoboVLG to new tasks and grasp requirements (Sec. VII-D).

## A. Experimental Setup

1) Hardware Platforms : As shown in Fig. 13, we conduct real-world experiments on three robotic platforms, including the KEENON XMAN-R1 prototype, TianJi Marvin, and AgileX Piper, equipped with two types of end-effectors: a parallel-jaw gripper and a 6-DoF ROHand dexterous hand. Specifically, on the XMAN-R1 and TianJi Marvin platforms equipped with the dexterous hand, we quantitatively evaluate language-guided functional grasping in cluttered scenes. To further investigate the deployment capability across different end-effectors, we equip the TianJi Marvin with a parallel-jaw gripper and conduct the same evaluation under the identical cluttered-scene setting. Furthermore, on the AgileX Piper platform, we evaluate the system’s functional grasping performance in dynamic cluttered scenes and quantitatively verify the temporal robustness of grasp tracking under human-induced disturbances. The realtime inference of the entire system is executed on a workstation equipped with an NVIDIA RTX 3090 GPU.

![](images/b233974be86196df9ce218843aa99bfc4487210417089bb5b49e590265149660.jpg)  
Fig. 13: Real-world robotic platforms. We conduct experiments on three robotic platforms with different end-effectors: (a) KEENON XMAN-R1 prototype equipped with a dexterous hand, (b) TianJi Marvin equipped with a dexterous hand and a parallel-jaw gripper, (c) AgileX Piper equipped with a dexterous hand, and (d) AgileX Piper equipped with a parallel-jaw gripper.

2) Evaluation Protocols: To evaluate the zero-shot performance of our method in the real world, we construct a test set of 102 everyday objects with diverse shapes, materials, and textures, covering six categories: food (22 objects), household items (22), tools (18), toys (13), electronics (13), and textiles (14). A trial is considered successful only when the target object is correctly identified, the executed grasp satisfies the functional requirement, no collision occurs with surrounding objects during execution, and the object is lifted by 10 cm and held for 5 s without being dropped. Under this protocol, we directly evaluate the sim-to-real performance of AdaRoboVLG without real-world fine-tuning, using both a parallel-jaw gripper and the ROHand across representative real-world grasping tasks. The feasible grasp types for the ROHand follow the Inspirehand grasp taxonomy defined in [71].

![](images/93802fc404c0fab6ce8d9db607f113c165c3d48e0bd4575e3db8262f9019bf73.jpg)  
Fig. 14: Real-world language-guided dexterous functional grasping in static cluttered scenes. Representative grasping results are presented across six everyday object categories: food, household items, tools, toys, electronics, and textiles.

## B. Experiments of Language-Guided Functional Grasping in Static Cluttered Scenes

1) Task Description: In this task, we evaluate the system’s ability to perform language-guided functional grasping in dense clutter using the ROHand on both the XMAN-R1 and TianJi Marvin platforms. For each of the 102 target objects, we construct a cluttered scene containing one target object associated with a specific functional instruction, 2–3 distractors with similar semantic categories or visual appearances, and additional background objects. We conduct five trials for each object under randomized object poses and scene configurations, yielding 510 grasping trials in total. We report the category-level and overall success rates aggregated from the two platforms.

2) Experimental Results: As shown in Fig. 14, by composing the spatial and cognitive priors, AdaRoboVLG can generate spatially feasible and functionally appropriate dexterous grasp poses in cluttered scenes. When the target object is surrounded by cluttered or occluding objects (e.g., a watermelon and a toothpick box), our method can still reliably plan collisionfree grasps. For tools or daily items with explicit functional affordances (e.g., a bottle, a megaphone, and a power drill), it selects task-relevant functional parts and grasp types, producing grasps consistent with human object-use behaviors. As illustrated in Fig. 15a, AdaRoboVLG achieves an overall success rate of 83.3% across 510 real-world grasping trials. These results demonstrate that the system can reliably perform target identification, functional grasp intent inference, and dexterous grasp generation in real-world clutter.

We further summarize the distribution of failure modes in Fig. 15b. Functional-part recognition errors constitute the primary source of failure, accounting for 29.4% of unsuccessful trials. These failures mainly occur when the target object’s functional regions are ambiguous or visually similar to distractors, leading to incorrect functional part localization. Grasp pose estimation errors account for 22.3%, primarily caused by inaccurate pose estimation or unstable candidate selection under geometric uncertainties. Insufficient contact stability represents 20.0% of failures, frequently occurring with small or smooth-surfaced objects where the generated grasps fail to maintain reliable contact after lifting. Multiobject interference contributes 16.5% of failures, where the dexterous hand unintentionally grasps multiple objects in dense clutter. The remaining 11.8% are attributed to grasp-type errors, which occur when the selected grasp type does not fully match the object’s functional requirements.

![](images/8b2eda307ece39e1918c4fe89fa01f900f30c7f6d01d960f1a70c4cc29054b45.jpg)

![](images/7510b0145987035db69c702a52601fddce5359029f5860a4b5936f853bb71a53.jpg)  
(b) Failure distributions.  
Fig. 15: Quantitative analysis for language-guided dexterous functional grasping in cluttered scenes. The results summarize (a) category-level and overall success rates and (b) failure distributions of unsuccessful trials.

![](images/c6f63da5ece70d7e0e3a4c9e23658b50a1ec6647ce4e479cfec9fb4e38247bba.jpg)  
Fig. 16: Real-world deployment with different end-effectors. AdaRoboVLG performs language-guided functional grasping in cluttered scenes with both a parallel-jaw gripper and the ROHand dexterous hand on the TianJi Marvin platform.

![](images/98bf78650f19e1cab8e703610e0d5eb57927f2d2bf103e92482a8dc9c5c3690f.jpg)  
Fig. 17: Real-world language-guided dexterous functional grasping in dynamic cluttered scenes. AdaRoboVLG performs functional grasping on a moving conveyor belt.

Additionally, as shown in Fig. 16, AdaRoboVLG performs language-guided functional grasping in cluttered scenes on the TianJi Marvin platform equipped with a parallel-jaw gripper. This evaluation demonstrates that the framework can be readily transferred between different end-effectors.

C. Experiments of Language-Guided Functional Grasping in Dynamic Cluttered Scenes

1) Task Description: This experiment evaluates the ability of the system to perform language-guided functional grasping in dynamic cluttered scenes. It provides a real-world validation of the composability of spatial, cognitive, and temporal priors for handling complex grasping scenarios. The experiment is conducted using the AgileX Piper equipped with a dexterous hand. We select 16 representative objects and place them together with distractors on a moving conveyor belt operating at a speed of 2 cm/s to 5 cm/s. Furthermore, to quantitatively evaluate tracking robustness, we equip the same manipulator with a parallel-jaw gripper. We introduce four types of humaninduced disturbances, including out-of-view recovery, fast movement, continuous occlusion, and simultaneous translation and rotation. For each disturbance condition, we select 10 objects and perform five repeated grasping trials, reporting the average success rate across all conditions.

TABLE V: Quantitative evaluation of dynamic grasping. We report the number of successful trials over total trials under different human-induced disturbances. N/A indicates that the test is not applicable due to rotational symmetry.
<table><tr><td>Object</td><td>Out-of-view Recovery</td><td>Fast Motion</td><td>Continuous Occlusion</td><td>Trans. &amp; Rot.</td><td>Success Rate</td></tr><tr><td>Green Bottle</td><td>5/5</td><td>4/5</td><td>5/5</td><td>4/5</td><td>90.0%</td></tr><tr><td>Mango</td><td>5/5</td><td>4/5</td><td>5/5</td><td>4/5</td><td>90.0%</td></tr><tr><td>Toothpaste</td><td>5/5</td><td>4/5</td><td>5/5</td><td>4/5</td><td>90.0%</td></tr><tr><td>Sponge</td><td>5/5</td><td>5/5</td><td>5/5</td><td>4/5</td><td>95.0%</td></tr><tr><td>Red Toy Car</td><td>5/5</td><td>5/5</td><td>5/5</td><td>3/5</td><td>90.0%</td></tr><tr><td>Tennis Ball</td><td>5/5</td><td>4/5</td><td>5/5</td><td>N/A</td><td>93.3%</td></tr><tr><td>Remote Control</td><td>5/5</td><td>5/5</td><td>4/5</td><td>4/5</td><td>90.0%</td></tr><tr><td>Small Fan</td><td>5/5</td><td>5/5</td><td>5/5</td><td>4/5</td><td>95.0%</td></tr><tr><td>Hairband</td><td>4/5</td><td>4/5</td><td>4/5</td><td>3/5</td><td>75.0%</td></tr><tr><td>White Gauze</td><td>4/5</td><td>5/5</td><td>5/5</td><td>4/5</td><td>90.0%</td></tr><tr><td>All</td><td>48/50</td><td>45/50</td><td>48/50</td><td>34/45</td><td>89.7%</td></tr></table>

![](images/fa2cf333542b39a4fd130e3ee726f937966c5adc4a9a4e395dc3fb03085baf6f.jpg)  
Fig. 18: Dynamic grasping under different human-induced disturbances. Each row presents the continuous tracking and grasping process under a specific language instruction.

2) Experimental Results: As illustrated in Fig. 17, our proposed framework demonstrates reliable language-guided functional grasping in dynamic cluttered conveyor-belt scenarios while updating grasp poses online at 5 Hz. For example, in Fig. 17c, given the instruction, the spatial prior constrains grasp generation to avoid collisions with adjacent distractors, while the cognitive prior selects the cup handle as the appropriate functional part and determines Prismatic 3 Finger as the tasksuitable grasp type. The base grasp policy then produces stable and executable grasp poses for the dexterous hand. The temporal prior further enables the system to robustly track the target as it moves along the conveyor belt. These results demonstrate the effectiveness of composing the spatial, cognitive, and temporal priors for real-world dynamic grasping.

![](images/1dbf7931873a1a7029fc4f32fc3a5aa60d03d65fa929c5046d8e9132cdf57188.jpg)  
Fig. 19: Extensibility demonstrations of AdaRoboVLG. We demonstrate two representative extensions: (a) bimanual sorting and (b) transparent-object grasping.

Furthermore, as reported in Tab. V, the system achieves an average grasping success rate of 89.7% under various humaninduced disturbances (see Fig. 18). Under perturbations such as fast motion, continuous occlusion, simultaneous translation and rotation, the temporal tracking module maintains target feature consistency and continuously updates grasp poses. Notably, even in out-of-view recovery tests, the system rapidly restores object pose estimation and grasp planning, further verifying its robustness in non-static real-world environments.

## D. Extensibility of the Adaptive Framework

We further demonstrate that our framework can be extended to new challenging grasping scenarios by integrating additional task-relevant modules. As illustrated in Fig. 19, incorporating a task-allocation module enables AdaRoboVLG to perform bimanual sorting by distributing grasp targets between the two arms. Similarly, transparent-object grasping is supported by integrating a depth-completion module (e.g., SwinDRNet [85]) to recover missing object geometry before constructing the structured grasp interface. This extensibility stems from the decoupling of task understanding and physical grasp synthesis, which allows new capabilities to be incorporated through taskspecific modules without requiring an entirely new policy.

## VIII. DISCUSSION AND CONCLUSION

## A. Limitations and Future Directions

Perception under Uncertainty: The foundation-model priors for constructing, refining, and updating the structured grasp interface remain sensitive to perceptual uncertainty. Specifically, depth noise can distort contact geometry and compromise grasp generation and evaluation, while inconsistent cross-view grounding can lead to incorrect functional-region identification. Furthermore, frequent changes in visual observations can easily induce pose drift and introduce unnecessary update latency. To address these limitations, future work could integrate more capable foundation models. Promising directions include metric reconstruction [86], [87] to improve geometric accuracy, cross-view correspondence [88], [89] to promote grounding consistency, and point-tracking methods [90] to support efficient, low-latency pose estimation.

Closed-loop Physical Interaction: Our base grasp policy predicts grasp stability from visual geometry but lacks tactile feedback to verify contact conditions during execution. Discrepancies between predicted and actual contacts, including insufficient contact forces or incipient slip, may therefore remain undetected, limiting grasp robustness for small or smooth objects. Incorporating tactile feedback [91] could complement geometric predictions with direct contact measurements, supporting closed-loop stability assessment and corrective in-hand adjustments.

Beyond Single-step Grasping: Our framework synthesizes task-consistent grasps but does not explicitly account for dependencies between grasping and other manipulation actions. In heavily cluttered scenes, preparatory actions may be needed to expose the target, while subsequent manipulation may impose additional constraints on grasp selection. Integrating non-prehensile skills [92] with sequential task planning could support scene rearrangement before grasping and coordinate grasp selection with subsequent actions.

## B. Conclusion

In this paper, we introduce AdaRoboVLG, a task-adaptive VLG framework that supports deployment across different robotic hands. It learns an efficient and generalizable base policy for cross-hand grasp synthesis while offloading taskdependent understanding to specialized foundation-model modules. These modules leverage composable priors to convert task contexts into a structured grasp interface, allowing the base policy to generate task-consistent and physically feasible grasp poses. Extensive simulation and real-world experiments demonstrate that AdaRoboVLG successfully synthesizes grasps across two- to five-finger end-effectors, while effectively incorporating spatial, cognitive, and temporal priors to adapt to diverse grasping tasks under combined contextual conditions (e.g., cluttered scenes, functional requirements, and dynamic interactions). By establishing a stable interface between high-level task understanding and low-level grasp execution, AdaRoboVLG provides a scalable path for challenging grasping scenarios.

## REFERENCES

[1] H. Liu, Z. Zhang, Z. Jiao, Z. Zhang, M. Li, C. Jiang, Y. Zhu, and S.-C. Zhu, “A reconfigurable data glove for reconstructing physical and virtual grasps,” Engineering, vol. 32, pp. 202–216, 2024.

[2] Y. Zhu, T. Gao, L. Fan, S. Huang, M. Edmonds, H. Liu, F. Gao, C. Zhang, S. Qi, Y. N. Wu et al., “Dark, beyond deep: A paradigm shift to cognitive ai with humanlike common sense,” Engineering, vol. 6, pp. 310–345, 2020.

[3] M. Wilson, “Six views of embodied cognition,” Psychonomic bulletin & review, vol. 9, pp. 625–636, 2002.

[4] M. Goodale and D. Milner, Sight unseen: An exploration of conscious and unconscious vision. OUP Oxford, 2013.

[5] E. Thelen and L. B. Smith, A dynamic systems approach to the development of cognition and action. MIT press, 1994.

[6] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[7] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[8] S. Chen, X. Chen, C. Zhang, M. Li, G. Yu, H. Fei, H. Zhu, J. Fan, and T. Chen, “Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 26 428–26 438.

[9] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield, “Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 15 768–15 780.

[10] Y. Ji, H. Tan, J. Shi, X. Hao, Y. Zhang, H. Zhang, P. Wang, M. Zhao, Y. Mu, P. An et al., “Robobrain: A unified brain model for robotic manipulation from abstract to concrete,” arXiv preprint arXiv:2502.21257, 2025.

[11] B. R. Team, M. Cao, H. Tan, Y. Ji, X. Chen, M. Lin, Z. Li, Z. Cao, P. Wang, E. Zhou et al., “Robobrain 2.0 technical report,” arXiv preprint arXiv:2507.02029, 2025.

[12] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π : A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[13] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai et al., “π<sub>0.5</sub>: A visionlanguage-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[14] S. Deng, M. Yan, S. Wei, H. Ma, Y. Yang, J. Chen, Z. Zhang, T. Yang, X. Zhang, H. Cui et al., “Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data,” arXiv preprint arXiv:2505.03233, 2025.

[15] Y. Zhong, X. Huang, R. Li, C. Zhang, Y. Liang, Y. Yang, and Y. Chen, “Dexgraspvla: A vision-language-action framework towards general dexterous grasping,” arXiv preprint arXiv:2502.20900, 2025.

[16] J. He, D. Li, X. Yu, Z. Qi, W. Zhang, J. Chen, Z. Zhang, Z. Zhang, L. Yi, and H. Wang, “Dexvlg: Dexterous vision-language-grasp model at scale,” arXiv preprint arXiv:2507.02747, 2025.

[17] H. Luo, Y. Feng, W. Zhang, S. Zheng, Y. Wang, H. Yuan, J. Liu, C. Xu, Q. Jin, and Z. Lu, “Being-h0: Vision-language-action pretraining from large-scale human videos,” arXiv preprint arXiv:2507.15597, 2025.

[18] L. Zhang, D. Zheng, K. Bai, Z. Bing, Z.-C. Marton, Z. Chen, A. C. Knoll, and J. Zhang, “Omnidexvlg: Learning dexterous grasp generation from vision language model-guided grasp semantics, taxonomy and functional affordance,” arXiv preprint arXiv:2512.03874, 2025.

[19] M. Choi, G. Kim, J. Kim, T. Kim, T. Ha, J. Lim, and H. Joo, “Autodex: An automated real-world system for dexterous grasping data collection,” arXiv preprint arXiv:2606.23689, 2026.

[20] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, “Contactgraspnet: Efficient 6-dof grasp generation in cluttered scenes,” in International Conference on Robotics and Automation (ICRA), 2021, pp. 13 438–13 444.

[21] T. Feix, J. Romero, H.-B. Schmiedmayer, A. M. Dollar, and D. Kragic, “The grasp taxonomy of human grasp types,” Transactions on humanmachine systems (THMS), vol. 46, pp. 66–77, 2015.

[22] H. Dai, A. Majumdar, and R. Tedrake, “Synthesis and optimization of force closure grasps via sequential semidefinite programming,” in Robotics Research, vol. 2, 2015, pp. 285–305.

[23] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., “Retrievalaugmented generation for knowledge-intensive nlp tasks,” Advances in Neural Information Processing Systems (NeurIPS), pp. 9459–9474, 2020.

[24] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in Neural Information Processing Systems (NeurIPS), pp. 24 824–24 837, 2022.

[25] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[26] J. Zhang, H. Liu, D. Li, X. Yu, H. Geng, Y. Ding, J. Chen, and H. Wang, “Dexgraspnet 2.0: Learning generative dexterous grasping in large-scale synthetic cluttered scenes,” in Conference on Robot Learning (CoRL), 2024, pp. 5106–5133.

[27] S. Back, J. Lee, K. Kim, H. Rho, G. Lee, R. Kang, S. Lee, S. Noh, Y. Lee, T. Lee et al., “Graspclutter6d: A large-scale real-world dataset for robust perception and grasping in cluttered scenes,” arXiv preprint arXiv:2504.06866, 2025.

[28] H.-S. Fang, C. Wang, M. Gou, and C. Lu, “Graspnet-1billion: A largescale benchmark for general object grasping,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11 441–11 450.

[29] C. Tang, D. Huang, W. Ge, W. Liu, and H. Zhang, “Graspgpt: Leveraging semantic knowledge from a large language model for task-oriented grasping,” IEEE Robotics and Automation Letters (RA-L), vol. 8, pp. 7551–7558, 2023.

[30] W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction for robotics,” arXiv preprint arXiv:2406.10721, 2024.

[31] Y. Tang, S. Zhang, X. Hao, P. Wang, J. Wu, Z. Wang, and S. Zhang, “Affordgrasp: In-context affordance reasoning for open-vocabulary taskoriented grasping in clutter,” in International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 9433–9439.

[32] Z. Zhao, J. Gao, and D. Zheng, “Affordance-guided robotic grasping via multimodal large language model reasoning,” Transactions on Automation Science and Engineering (TASE), vol. 23, pp. 4088–4100, 2026.

[33] M. Ji, R.-Z. Qiu, X. Zou, and X. Wang, “Graspsplats: Efficient manipulation with 3d feature splatting,” arXiv preprint arXiv:2409.02084, 2024.

[34] Y. Zheng, X. Chen, Y. Zheng, S. Gu, R. Yang, B. Jin, P. Li, C. Zhong, Z. Wang, L. Liu et al., “Gaussiangrasper: 3d language gaussian splatting for open-vocabulary robotic grasping,” IEEE Robotics and Automation Letters (RA-L), vol. 9, pp. 7827–7834, 2024.

[35] Y. Qian, X. Zhu, O. Biza, S. Jiang, L. Zhao, H. Huang, Y. Qi, and R. Platt, “Thinkgrasp: A vision-language system for strategic part grasping in clutter,” arXiv preprint arXiv:2407.11298, 2024.

[36] H. Liu, S. Guo, P. Mai, J. Cao, H. Li, and J. Ma, “Robodexvlm: Visual language model-enabled task planning and motion control for dexterous robot manipulation,” in International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 1381–1388.

[37] R. Mirjalili, M. Krawez, S. Silenzi, Y. Blei, and W. Burgard, “Lan grasp: Using large language models for semantic object grasping,” arXiv preprint arXiv:2310.05239, 2023.

[38] Z. Li, J. Liu, Z. Li, Z. Dong, T. Teng, Y. Ou, D. G. Caldwell, and F. Chen, “Language-guided dexterous functional grasping by LLM generated grasp functionality and synergy for humanoid manipulation,” Transactions on Automation Science and Engineering (TASE), vol. 22, pp. 10 506–10 519, 2025.

[39] Z. Xu, B. Qi, S. Agrawal, and S. Song, “Adagrasp: Learning an adaptive gripper-aware grasping policy,” in International Conference on Robotics and Automation (ICRA), 2021, pp. 4620–4626.

[40] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen et al., “Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 4737–4746.

[41] W. Xu, W. Guo, X. Shi, X. Sheng, and X. Zhu, “Fast force-closure grasp synthesis with learning-based sampling,” IEEE Robotics and Automation Letters (RA-L), vol. 8, pp. 4275–4282, 2023.

[42] W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang, “Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning,” in International Conference on Computer Vision (ICCV), 2023, pp. 3891– 3902.

[43] Z. Chen, Q. Yan, Y. Chen, T. Wu, J. Zhang, Z. Ding, J. Li, Y. Yang, and H. Dong, “Clutterdexgrasp: A sim-to-real system for general dexterous grasping in cluttered scenes,” arXiv preprint arXiv:2506.14317, 2025.

[44] J. Varley, J. Weisz, J. Weiss, and P. Allen, “Generating multi-fingered robotic grasps via deep learning,” in International Conference on Intelligent Robots and Systems (IROS), 2015, pp. 4415–4420.

[45] P. Li, T. Liu, Y. Li, Y. Geng, Y. Zhu, Y. Yang, and S. Huang, “Gendexgrasp: Generalizable dexterous grasping,” in International Conference on Robotics and Automation (ICRA), 2023, pp. 8068–8074.

[46] L. Shao, F. Ferreira, M. Jorda, V. Nambiar, J. Luo, E. Solowjow, J. A. Ojea, O. Khatib, and J. Bohg, “Unigrasp: Learning a unified model to grasp with multifingered robotic hands,” IEEE Robotics and Automation Letters (RA-L), vol. 5, pp. 2286–2293, 2020.

[47] Z. Wu, R. A. Potamias, X. Zhang, Z. Zhang, J. Deng, and S. Luo, “Cedex: Cross-embodiment dexterous grasp generation at scale from human-like contact representations,” arXiv preprint arXiv:2509.24661, 2025.

[48] A. Wu, M. Guo, and C. K. Liu, “Learning diverse and physically feasible dexterous grasps with generative model and bilevel optimization,” arXiv preprint arXiv:2207.00195, 2022.

[49] Q. She, R. Hu, J. Xu, M. Liu, K. Xu, and H. Huang, “Learning highdof reaching-and-grasping via dynamic representation of gripper-object interaction,” arXiv preprint arXiv:2204.13998, 2022.

[50] Z. Wei, Z. Xu, J. Guo, Y. Hou, C. Gao, Z. Cai, J. Luo, and L. Shao, “D(R, O) grasp: A unified representation of robot and object interaction for cross-embodiment dexterous grasping,” in International Conference on Robotics and Automation (ICRA), 2025, pp. 4982–4988.

[51] X. Fei, Z. Xu, H. Fang, T. Zhang, and L. Shao, “T (r, o) grasp: Efficient graph diffusion of robot-object spatial transformation for crossembodiment dexterous grasping,” arXiv preprint arXiv:2510.12724, 2025.

[52] Y. Li, W. Wei, D. Li, P. Wang, W. Li, and J. Zhong, “Hgc-net: Deep anthropomorphic hand grasping in clutter,” in International Conference on Robotics and Automation (ICRA), 2022, pp. 714–720.

[53] J. Zhang, Z. Ma, T. Wu, Z. Chen, and H. Dong, “Cadgrasp: Learning contact and collision aware general dexterous grasping in cluttered scenes,” arXiv preprint arXiv:2601.15039, 2026.

[54] R. Firoozi, J. Tucker, S. Tian, A. Majumdar, J. Sun, W. Liu, Y. Zhu, S. Song, A. Kapoor, K. Hausman et al., “Foundation models in robotics: Applications, challenges, and the future,” International Journal of Robotics Research (IJRR), vol. 44, pp. 701–739, 2025.

[55] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.

[56] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[57] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025.

[58] Y. Li, K. Xiong, X. Guo, F. Li, S. Yan, G. Xu, L. Zhou, L. Chen, H. Sun, B. Wang et al., “Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving,” arXiv preprint arXiv:2506.08052, 2025.

[59] Y. Li, L. Zhou, S. Yan, B. Liao, T. Yan, K. Xiong, L. Chen, H. Xie, B. Wang, G. Chen et al., “Unidrivevla: Unifying understanding, perception, and action planning for autonomous driving,” arXiv preprint arXiv:2604.02190, 2026.

[60] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, “Voxposer: Composable 3d value maps for robotic manipulation with language models,” arXiv preprint arXiv:2307.05973, 2023.

[61] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, “Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation,” arXiv preprint arXiv:2409.01652, 2024.

[62] C. Yuan, S. Joshi, S. Zhu, H. Su, H. Zhao, and Y. Gao, “Roboengine: Plug-and-play robot data augmentation with semantic robot segmentation and background generation,” arXiv preprint arXiv:2503.18738, 2025.

[63] Y. Mei, J. Sun, Z. Peng, F. Deng, G. Wang, and J. Chen, “Rog-sam: A language-driven framework for instance-level robotic grasping detection,” Transactions on Multimedia (TMM), vol. 27, pp. 3057–3068, 2025.

[64] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “Rdt-1b: a diffusion foundation model for bimanual manipulation,” arXiv preprint arXiv:2410.07864, 2024.

[65] X. Wu, L. Jiang, P.-S. Wang, Z. Liu, X. Liu, Y. Qiao, W. Ouyang, T. He, and H. Zhao, “Point transformer v3: Simpler, faster, stronger,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 4840–4851.

[66] T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S.-C. Zhu, “Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator,” IEEE Robotics and Automation Letters (RA-L), vol. 7, pp. 470–477, 2021.

[67] R. Meattini, R. Suárez, G. Palli, and C. Melchiorri, “Human to robot hand motion mapping methods: Review and classification,” Transactions on Robotics (T-RO), vol. 39, pp. 842–861, 2023.

[68] J. Chen, Y. Ke, L. Peng, and H. Wang, “Dexonomy: Synthesizing all dex terous grasp types in a grasp taxonomy,” arXiv preprint arXiv:2504.18829, 2025.

[69] H. Yan, H. Fang, and C. Lu, “Dexterous manipulation based on prior dexterous grasp pose knowledge,” arXiv preprint arXiv:2412.15587, 2024.

[70] H. Yan, H.-S. Fang, and C. Lu, “A surprisingly efficient representation for multi-finger grasping,” in International Conference on Robotics and Automation (ICRA), 2024, pp. 6462–6469.

[71] H.-S. Fang, H. Yan, Z. Tang, H. Fang, C. Wang, and C. Lu, “Anydexgrasp: General dexterous grasping for different hands with human-level learning efficiency,” arXiv preprint arXiv:2502.16420, 2025.

[72] M. Santello, M. Flanders, and J. F. Soechting, “Postural hand synergies for tool use,” Journal of neuroscience, vol. 18, pp. 10 105–10 115, 1998.

[73] C. R. Mason, J. E. Gomez, and T. J. Ebner, “Hand synergies during reach-to-grasp,” Journal of neurophysiology, vol. 86, pp. 2896–2910, 2001.

[74] N. Jarque-Bou, V. Gracia-Ibáñez, J.-L. Sancho-Bru, M. Vergara, A. Pérez-González, and F. Andrés, “Using kinematic reduction for studying grasping postures. an application to power and precision grasp of cylinders,” Applied ergonomics, vol. 56, pp. 52–61, 2016.

[75] S. Coumar, G. Chang, N. Kodkani, and Z. Kingston, “Foam: A tool for spherical approximation of robot geometry,” arXiv preprint arXiv:2503.13704, 2025.

[76] Y. Wang, M. Zhang, Z. Li, T. Kelestemur, K. Driggs-Campbell, J. Wu, L. Fei-Fei, and Y. Li, “D<sup>3</sup>fields: Dynamic 3d descriptor fields for zeroshot generalizable rearrangement,” arXiv preprint arXiv:2309.16118, 2023.

[77] C. Tang, D. Huang, W. Dong, R. Xu, and H. Zhang, “Foundationgrasp: Generalizable task-oriented grasping with foundation models,” Transactions on Automation Science and Engineering (TASE), vol. 22, pp. 12 418–12 435, 2025.

[78] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu, “Anygrasp: Robust and efficient grasp perception in spatial and temporal domains,” Transactions on Robotics (T-RO), vol. 39, pp. 3929–3945, 2023.

[79] B. Wen and K. E. Bekris, “Bundletrack: 6d pose tracking for novel objects without instance or category-level 3d models,” in International Conference on Intelligent Robots and Systems (IROS), 2021, pp. 8067– 8074.

[80] J. Liu, R. Zhang, H. Fang, M. Gou, H. Fang, C. Wang, S. Xu, H. Yan, and C. Lu, “Target-referenced reactive grasping for dynamic objects,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 8824–8833.

[81] N. Chen, X. Wu, G. Xu, J. Jiang, Z. Chen, and W. Zheng, “Motiongrasp: Long-term grasp motion tracking for dynamic grasping,” IEEE Robotics and Automation Letters (RA-L), vol. 10, pp. 796–803, 2025.

[82] K. Xu, X. Xia, K. Wang, Y. Yang, Y. Mao, B. Deng, J. Ye, R. Xiong, and Y. Wang, “Efficient alignment of unconditioned action prior for languageconditioned pick and place in clutter,” Transactions on Automation Science and Engineering (TASE), vol. 22, pp. 21 256–21 268, 2025.

[83] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang, “Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation,” arXiv preprint arXiv:2210.02697, 2022.

[84] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[85] Q. Dai, J. Zhang, Q. Li, T. Wu, H. Dong, Z. Liu, P. Tan, and H. Wang, “Domain randomization-enhanced depth simulation and restoration for perceiving and grasping specular and transparent objects,” in European Conference on Computer Vision (ECCV), 2022, pp. 374–391.

[86] H. Zhou, S. Liu, Y. He, B. Zhang, F. Fu, C. Hou, X. Hou, L. Han, and W. Sui, “X-lens: Real-time metric depth estimation with heterogeneous cameras,” arXiv preprint arXiv:2607.12993, 2026.

[87] S. Yang, L. Xu, H. Li, J. Mu, J. Zeng, D. Lin, and J. Pang, “Robo3r: Enhancing robotic manipulation with accurate feed-forward 3d recon struction,” arXiv preprint arXiv:2602.10101, 2026.

[88] Y. Cao, F. Wu, D. Z. Chen, Y. Zhong, L. Hong, and D. Xu, “Vggt-det: Mining vggt internal priors for sensor-geometry-free multi-view indoor 3d object detection,” arXiv preprint arXiv:2603.00912, 2026.

[89] J. Pan, R. Wang, T. Qian, M. Mahdi, Y. Fu, X. Xue, X. Huang, L. Van Gool, D. P. Paudel, and Y. Fu, “Vˆ 2-sam: Marrying sam2 with multi-prompt experts for cross-view object correspondence,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 16 910–16 919.

[90] T.-Y. Lin, H. J. Lee, K. Doherty, Y. Lee, and S. Kim, “Point2pose: Occlusion-recovering 6d pose tracking and 3d reconstruction for multiple unknown objects via 2d point trackers,” arXiv preprint arXiv:2604.10415, 2026.

[91] Z. Zhao, W. Li, Y. Li, T. Liu, B. Li, M. Wang, K. Du, H. Liu, Y. Zhu, Q. Wang et al., “Embedding high-resolution touch across robotic hands enables adaptive human-like grasping,” Nature Machine Intelligence, vol. 7, pp. 889–900, 2025.

[92] L. Pei, H. Yuzhe, L. Wanlin, X. Chenxi, and J. Ziyuan, “Dexmove: Learning tactile-guided non-prehensile manipulation with dexterous hands,” in International Conference on Learning Representations (ICLR), 2026.