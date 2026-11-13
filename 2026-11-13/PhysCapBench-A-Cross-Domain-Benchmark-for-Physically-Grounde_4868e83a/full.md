# PhysCapBench: A Cross-Domain Benchmark for Physically Grounded Video Captioning

Haofeng Wang<sup>1,2</sup>, Pengcheng Wen<sup>1</sup>, Ruiqi Li<sup>2</sup>, Yuhao Zhang<sup>1</sup>, Tiange Zhang<sup>2</sup>, Boren Zheng<sup>2</sup>, Xuanyi Liu<sup>2</sup>, Siyu Tao<sup>4</sup>, Danni Yu<sup>3</sup>, Gan Lin<sup>2</sup>, Haocheng Tang<sup>2</sup>, Tianzizhe Guo<sup>2</sup>, Mingyuan Su<sup>1</sup>, Boyuan Chen<sup>1,2,5</sup>, Zhongyuan Wang<sup>1</sup> <sup>1</sup>Beijing Academy of Artificial Intelligence , <sup>2</sup>Peking University , <sup>3</sup>Tsinghua University , <sup>4</sup>Beihang University , <sup>5</sup>Physis

<sup></sup> Website <sup>§</sup> Code Hugging Face

## Abstract

Physical AI aims to learn transferable physical knowledge from real-world experience to understand, predict, and act in the physical world. Internet videos and embodied demonstrations provide complementary observations of motion, interaction, and state evolution. A foundational data challenge is to measure and preserve this physical information as training supervision. In video–language pretraining, captions express observed processes in language, yet a semantically correct description can omit the interactions and transitions that define those processes. Rich physical experience in video does not automatically translate into sufficient language supervision. This gap calls for a yardstick to assess annotation quality, compare captioning strategies, and guide training-data refinement. Motivated by our data construction for world-model pretraining, we introduce PhysCapBench, to our knowledge the first unified physical-caption benchmark spanning all three domains: third-person scenes, egocentric manipulation, and handheld-gripper demonstrations. It provides human-verified annotations for 1,500 video sequences and evaluates accuracy and coverage through seven physical dimensions. We examine downstream utility through video reconstruction and manipulation-policy fine-tuning. Physical captions reduce reconstruction LPIPS by 8.6–9.5% on third-person videos and 26.9–30.9% on egocentric videos relative to metadata captions. Training-instruction enrichment improves both PhysCapBench scores and policy success, including a 7.5-percentage-point gain for �<sub>0.5</sub> on a RoboTwin 2.0 subtask. PhysCapBench makes the physical content of language supervision inspectable and comparable, providing a cross-domain basis for its evaluation and improvement.

## PhysCapBench

Cross-domain physical annotation, evaluation, and downstream utilit

![](images/bc5fa003c827f15cfb3c9598ce3192159ad5e4c3fb9894eebf2676e33f2a3214.jpg)  
Figure 1. Overview of PhysCapBench. Across three video domains, PhysCapBench provides evidence-grounded annotations and evaluates captions along seven physical dimensions. The example illustrates how a caption can describe contact while omitting shape recovery.

## Contents

1 Introduction 3   
2 Related Work   
2.1 Video Caption Evaluation and Physical Grounding   
2.2 Video Data for World Modeling and Manipulation   
2.3 Caption Quality and Downstream Utility   
3 Task Definition and Benchmark Design   
3.1 Task Definition .   
3.2 Physical Representation   
3.3 Physical Evaluation Dimensions   
4 Data and Physical Annotation 7   
4.1 Data and Annotation Overview   
4.2 Third-Person Physical Scenes 10   
4.3 Egocentric Human Manipulation 10   
4.4 Multi-View Handheld-Gripper Demonstrations 11   
5 Evaluation and Downstream Utility 12   
5.1 Physical Caption Evaluation Protocol 12   
5.2 Caption Evaluation Results <sup>12</sup><sub>14</sub>   
5.3 Video Reconstruction Experiments   
5.4 Manipulation-Policy Fine-Tuning 16   
6 Discussion and Limitations 19   
6.1 Data and Annotation Coverage 19   
6.2 Evaluation Consistency . 19   
6.3 Downstream Applications 19   
7 Conclusion 19   
References 20   
A Annotation Schema and Domain Profiles 22   
A.1 Entities and Initial States 23   
A.2 Event Fields 23   
A.3 Sequence Records and Domain Extensions 24   
A.4 Temporal and Observation Conventions 25   
B Data Coverage and Video Examples 25   
B.1 Third-Person Coverage 25   
B.2 Egocentric Coverage 26   
B.3 Handheld-Gripper Coverage 28   
B.4 Video Examples across Three Domains 28   
C Annotation Prompts and Review Guidelines 31   
C.1 Auxiliary Cues and Frame Selection 31   
C.2 Prompt Content 31   
C.3 Review and Consistency Checks 32   
D Evaluation Instructions and Scoring Rubric <sup>32</sup><sub>32</sub>   
D.1 Dimension-Wise Caption Scoring   
D.2 Score Aggregation . 32   
E Generation and Policy Experiment Configurations 33   
E.1 Reconstruction Setup and Statistical Analysis <sup>33</sup><sub>33</sub>   
E.2 Manipulation-Policy Configuration .

## 1. Introduction

“The limits of my language mean the limits of my world.” — Ludwig Wittgenstein

Physical AI aims to learn from real-world experience to understand, predict, and act on the physical world [1–4]. Recent foundation-model training pipelines support this goal by integrating videos from multiple domains [3, 4, 9, 10] and using captions as a central component of this cross-domain data infrastructure to express physical information as language supervision and generation conditions [3, 11– 13]. However, these captions can correctly summarize a scene or task while omitting visible physical information about interactions and state changes, leaving that physical information absent from language supervision. These omissions can hinder physically consistent video generation and limit the ability of manipulation policies to follow detailed execution instructions [14, 15]. These omissions highlight the need to evaluate whether captions accurately and fully describe the physical information in videos. We address this need by tackling a key open problem in Physical AI pretraining: how can the physical information in captions be systematically evaluated across domains?

Existing caption benchmarks address parts of this question, but physical evaluation dimensions, multidomain coverage, and downstream validation have yet to be brought together in a unified benchmark. First, general caption benchmarks evaluate events and broad visual attributes [16–18]. Physical-caption benchmarks emphasize overall correctness and coverage or broad physical categories [14, 19]. Their limited breakdown across physical dimensions restricts detailed comparisons and targeted caption refinement. Second, foundation-model training uses video data from multiple domains [4, 9, 10], but existing caption benchmarks do not sufficiently cover these data needs [15, 18]. Third, existing studies examine caption utility in video generation [11, 12, 14, 17] or manipulation-policy learning [15], but no single benchmark has validated physical-caption evaluation in both applications.

To address these gaps, we introduce PhysCapBench, a unified physical-caption benchmark with seven physical evaluation dimensions. To our knowledge, it is the first physical-caption benchmark spanning all three domains: third-person scenes, egocentric manipulation, and handheld-gripper demonstrations. PhysCapBench comprises 1,500 video sequences with balanced coverage across the three domains and diverse scenes, tasks, and physical processes within each domain. In addition, we develop an evidencegrounded annotation pipeline with domain-specific observations, structured annotation, and human cross-checking. This pipeline produces human-verified physical reference annotations for each video sequence. We assess captions against these references to identify inconsistencies and omissions along each physical dimension. Together, these components provide a practical yardstick for comparing captioning strategies and refining captions used in both training and inference.

Our experiments demonstrate clear differences in physical-caption quality across strategies and downstream gains from physical-caption enrichment. Dimension-wise scores identify the physical details retained or omitted by each captioning strategy. We then evaluate physical captions as generation conditions through zero-shot, first-frame-conditioned video reconstruction with pretrained video generation models [10, 20]. Compared with metadata captions, physical captions reduce LPIPS by 8.6–9.5% on third-person videos and 26.9–30.9% on egocentric videos.

To further examine the value of physical captions during training, we use robotic manipulation as a testbed and evaluate enriched physical captions as training instructions. On RoboTwin 2.0 [21] and RoboDojo Sim, instruction enrichment raises PhysCapBench scores by 42.17 and 38.58 points, respectively, alongside higher mean task success across four policy families. For example, on a RoboTwin 2.0 subtask, instruction enrichment improves �<sub>0.5</sub> success from 37.5% to 45.0%, a gain of 7.5 percentage points [22]. The accompanying improvements in PhysCapBench scores and policy success, together with the reconstruction gains, support using PhysCapBench to assess caption quality and guide the refinement of physical information for downstream applications.

Our main contributions are as follows:

1. We introduce PhysCapBench, the first unified physical-caption benchmark spanning third-person scenes, egocentric manipulation, and handheld-gripper demonstrations. PhysCapBench comprises 1,500 video sequences and seven physical evaluation dimensions to assess descriptions of physical processes and identify errors and omissions across captioning strategies.

2. We develop an evidence-grounded physical annotation pipeline. The pipeline adapts to the observations available in each domain and produces structured annotations of entities, interactions, and state evolution. Human verification and cross-checking ensure the quality of the resulting physical reference annotations.

3. We demonstrate the downstream utility of PhysCapBench. Video reconstruction and robot manipulation-policy fine-tuning support using PhysCapBench to guide caption refinement for downstream applications. Physical-caption enrichment improves reconstruction fidelity, and higher PhysCapBench scores accompany improved policy performance across different benchmarks and policy models.

## 2. Related Work

## 2.1. Video Caption Evaluation and Physical Grounding

Physical understanding is evaluated through tasks with different outputs. PhysBench evaluates answers to physical questions, while VideoPhy assesses semantic adherence and physical commonsense in generated videos [23, 24]. Captioning raises a complementary question: how accurately and fully does a description convey the physical process in an observed video? PhysCapBench makes this descriptive information the object of evaluation, assessing consistency with structured physical references and coverage of the details they record.

Fine-grained caption benchmarks address this information-level question by checking individual events and facts. DREAM-1K’s AutoDQ decomposes captions into events and measures bidirectional entailment between candidate and reference descriptions [16]. VidCapBench and VCapsBench use question– answer evaluation to assess detailed caption content, including motion and physical laws alongside visual and aesthetic properties [17, 18]. These approaches already identify omissions and inconsistencies; VCapsBench also includes egocentric activities. Our annotation and evaluation design concentrates on physical content: it defines seven physical dimensions and uses structured annotations of interactions and state evolution to support caption assessment.

Physical-caption benchmarks pursue this focus through dedicated reference representations and evaluation protocols. Physis-Lang introduces a distinct benchmark, also named PhysCapBench, with humanverified atomic physical assertions; it checks caption precision against video evidence and recall against reference assertions [14]. OmniFysics-Captioner introduces OmniPhysCap, which probes the physical interactions and outcomes expressed in captions alongside other observable content [19]. Our emphasis is the joint coverage of third-person scenes, egocentric human manipulation, and multi-view demonstrations with human-operated handheld grippers, with annotation adapted to the evidence available in each domain. This design connects physical-caption assessment to the varied observations and data needs of world modeling and manipulation learning.

## 2.2. Video Data for World Modeling and Manipulation

The observations available for physical captioning depend on how video data are collected. Ego-Exo4D synchronizes first- and third-person views of skilled human activities and provides language annotations, including atomic action descriptions [25]. EgoDex pairs egocentric manipulation video with tracked hand and finger poses and natural-language metadata [7]. The Universal Manipulation Interface (UMI) records human demonstrations with handheld grippers and supports transferring the demonstrated skills to robot policies [8]. In PhysCapBench, EgoDex supplies egocentric video and hand-pose cues, while a separate multi-view handheld-gripper collection supplies synchronized views and trajectories. We adapt physical annotation to these observations while preserving participant identities and roles.

World-model research uses such complementary observations to learn physical dynamics. OmniWorld combines human, robot, simulated, and Internet video, using domain-specific prompts to produce captions [9]. �<sub>0</sub>-WM trains on robot trajectories, UMI-style demonstrations, and egocentric human videos, assigning supervision according to the signals available in each source [4]. These efforts establish heterogeneous video as a resource for world modeling. When heterogeneous video data are accompanied by captions used for supervision or conditioning, we focus on constructing and assessing their physical content across our three domains.

For manipulation, detailed language can specify how an interaction unfolds beyond its task goal. WGO-Bench evaluates temporal subtask boundaries and descriptions across human egocentric and robot recordings, with annotation centered on completed manipulation events and visible state changes [26]. FineVLA unifies trajectories from ten robot datasets and defines RoboFine-Bench, whose caption track evaluates step-level descriptions against atomic facts across ten execution-related dimensions [15]. We retain these interaction and state details in our manipulation annotations while also covering general physical scenes. The practical value of such detail can then be tested through the models that use the resulting descriptions.

## 2.3. Caption Quality and Downstream Utility

Video generation offers a setting for examining this practical value. VidCapBench relates caption-evaluation scores to generation quality when captions condition fixed text-to-video models [17]. Physis-Lang studies physical captions as representations for video world modeling [14]. Our reconstruction study compares physical and source metadata captions on matched videos, using first-frame conditioning and perceptual distances to reference videos. By evaluating both caption conditions against the same reference, we examine whether enrichment helps preserve the observed visual content.

Manipulation policies provide a complementary test in which language conditions task execution. FineVLA examines how fine-grained instructions affect task success and instruction-sensitive robot behavior [15]. Our policy studies apply the same description principles to training-instruction enrichment across selected manipulation tasks and multiple policy families. Together with reconstruction, it examines description utility in both generation and control.

Positioning of PhysCapBench. Prior work establishes the value of physical descriptions and the use of diverse video data in world modeling. Evidence about caption quality or utility in one data setting leaves other settings to be examined. PhysCapBench addresses this coverage question by bringing thirdperson scenes, egocentric human manipulation, and multi-view demonstrations with human-operated handheld grippers into one physical-caption benchmark. It defines seven physical evaluation dimensions and adapts annotation to the evidence available in each domain. This scope connects caption assessment to data needs ranging from general scene modeling to embodied manipulation. Reconstruction and policy studies further examine how enriched physical descriptions can serve downstream models.

## 3. Task Definition and Benchmark Design

PhysCapBench focuses on the accurate and comprehensive description of observable physical processes in videos. Its design comprises a captioning task, a structured physical representation, and seven physical evaluation dimensions. The representation organizes annotated content into entities, events, states,

and relations, while the evaluation dimensions guide reference-based assessment of candidate captions. These components establish the basis for the annotation procedures in Section 4 and the evaluation protocol in Section 5.

## 3.1. Task Definition

Given a video observation $\upsilon ,$ the task is to generate a caption � describing its observable physical process. Depending on the setting, � contains a single view or synchronized views of the same recording. The descriptive target comprises participating entities and their roles, interaction sequences, and visible state changes. The goal is an accurate and comprehensive account of this observable content.

Evaluation compares � with a textual reference $r ,$ comprising the reference caption and any dimensionspecific annotation text used for scoring, assessing semantic agreement and coverage of the physical information in �. Let �(�) contain the physical dimensions assessable from $r ,$ and let $s _ { d } ( c , r ) \in [ 0 , 1 0 0 ]$ jointly reflect agreement and coverage in dimension �. For nonempty �(�), the caption score is

$$
S ( c , r ) = \frac { 1 } { | D ( r ) | } \sum _ { d \in D ( r ) } s _ { d } ( c , r ) .\tag{1}
$$

The physical representation below organizes annotation content, while Section 3.3 defines the dimensions and Section 5 details the scoring protocol.

## 3.2. Physical Representation

The representation organizes physical content around identifiable entities and their participation in events. Entity records associate identities with observed states, including the initial and final states when available. Event records distinguish the acting participant, any instrument, the affected object, and the entity whose motion or state changes. Keeping these roles explicit preserves who acts on what, even when several participants contribute to the same interaction.

Events connect these participants through time. Each event records its interval and the relevant motion, contact, or state change, together with spatial and temporal relations. Timeline segments link the events to the progression of the clip, retaining persistent states and overlapping actions alongside transitions. The representation also accommodates physical mechanism statements and causal relations when supported by observations. Natural-language captions express this organized content for supervision or conditioning, while the structured records retain details that can be inspected individually. These details provide the basis for defining the physical aspects of caption evaluation. Appendix A provides the field definitions, including participant roles, event attributes, and domain-specific extensions.

## 3.3. Physical Evaluation Dimensions

PhysCapBench defines seven physical evaluation dimensions that connect caption assessment to the information organized by the representation. The dimensions distinguish broad event coverage from the details of participants, interactions, and state evolution. A caption may preserve the main activity while omitting the roles of its participants or the resulting state; assessing these aspects separately makes such omissions visible. Table 1 summarizes the evaluation focus of each dimension and the corresponding annotation content.

The dimensions connect the physical representation to caption assessment by organizing the reference information into distinct aspects of the observed process. Section 4 describes how the corresponding annotations are constructed across the three domains, and Section 5 presents the scoring protocol, dimension-level results, and downstream experiments.

Table 1. Physical evaluation dimensions and annotation content. The mapping links dimensions to annotation content. Text-based scoring measures candidate agreement with and coverage of applicable reference content.
<table><tr><td>Physical dimension</td><td>Evaluation focus</td><td>Annotation content</td></tr><tr><td>Event Coverage</td><td>Coverage of the main physical activities and events.</td><td>Event chain and linked timeline segments.</td></tr><tr><td>Objects and Roles</td><td>Agreement on manipulated objects and specified actor or hand roles.</td><td>Entity identities and event participants.</td></tr><tr><td>Actions and Motion</td><td>Agreement on actions, movement types, Action labels, moving entities, motion and directions.</td><td>types and directions.</td></tr><tr><td>Contact and Interaction</td><td>Agreement on contact and interaction relations.</td><td>Contact participants, contact relations, and interaction events.</td></tr><tr><td>States and Outcomes</td><td>Agreement on states, spatial relations, and resulting changes.</td><td>Initial and final states, state changes, and spatial relations.</td></tr><tr><td>Physical Mechanisms and Causality</td><td>Agreement on explicitly stated mechanisms or causal explanations.</td><td>Mechanism statements, supporting events, and causal relations.</td></tr><tr><td>Temporal Order</td><td>Agreement on explicitly stated order or simultaneity.</td><td>Event intervals, temporal relations, and linked timeline segments</td></tr></table>

## 4. Data and Physical Annotation

The physical representation is applied to three data domains with different viewpoints, participants, and auxiliary observations. Annotation preserves the information available in each domain while organizing it around entities, events, states, and relations. Figure 2 summarizes the category composition, representative observations, and coverage statistics across the three domains. We first summarize the domain-specific choices, data coverage, and annotation quality controls, then describe the data and annotation procedures for each domain.

## 4.1. Data and Annotation Overview

The three domains provide complementary observations of physical processes. Third-person scenes expose scene-level interactions, egocentric videos show human manipulation from the wearer’s viewpoint, and handheld-gripper demonstrations record human-operated interactions across synchronized views. Table 2 summarizes the observations and annotation emphasis in each setting. This coverage responds to the varied data needs of world modeling and manipulation learning, with annotation adapted to each source.

Table 3 summarizes the data coverage across the three domains, each containing 500 videos or demonstration episodes. Appendix B provides detailed category distributions and source-video examples for all three domains.

Figure 3 summarizes the annotation workflow, from domain-adapted observations to verified records and captions. Scene RGB, hand–object cues and hand poses, or synchronized views and trajectories guide frame selection and inspection in the respective domains. A vision–language annotator organizes these observations into entities and initial states, temporally localized events, and state changes and relations. Event records retain actor and target identities, motion, and contact, with evidence references linking descriptions to source frames. Domain-specific fields preserve the observation details needed to interpret each interaction.

Across all three domains, annotation combines automated draft generation with human verification. A team of seven professional data annotators performed the annotation and cross-checked one another’s work for quality control. All final benchmark annotations have been manually verified against the source videos. Quality control combines structural checks of entity and event references, frame ranges, time conversion, and timeline consistency with human review of the proposed actors, interactions, and state transitions. Evidence references link these checks to the relevant observations, while the annotation records retain review decisions and field-level uncertainty.

![](images/4646494c75e162a3753357d4220c4c520ee3a655b0a6432b139378c615bfd959.jpg)  
Figure 2. Dataset overview across three domains. (a) Category composition. (b) Representative video frames and synchronized handheld-gripper views. (c) Third-person scene types, egocentric action families, and handheld-gripper trajectory profiles, with sequence counts. Each domain contains 500 videos or demonstration episodes.

Table 2. Three-domain data and annotation overview. Auxiliary observations support annotation; their availability to a captioning model is specified separately by the evaluation setting.
<table><tr><td>Data domain</td><td>Available observations</td><td>Annotation emphasis</td></tr><tr><td>Third-person physical scenes</td><td>Scene video showing participants and their evolving relations.</td><td>Scene entities, physical interactions, motion, and state evolution</td></tr><tr><td>Egocentric human manipulation</td><td>Egocentric video with hand-pose information available as annotation cues.</td><td>Hand identity and roles, hand-object interactions, and visible object changes.</td></tr><tr><td>Multi-view handheld-gripper demonstrations</td><td>Synchronized views and trajectory cues for locating frames to inspect.</td><td>Human-operated gripper identities and roles, and corresponding observations across views.</td></tr></table>

Quality control follows the observations available in each domain. Egocentric review checks handedness, hand–object interactions, and the phases of manipulation. Multi-view review checks actor attribution across cameras, object identity across views, and the before-and-after evidence for state transitions. Reviewers record the supporting observations and resolve ambiguous descriptions through source-video inspection. Appendix C details the review rules.

Table 3. Dataset coverage across the three domains. Multiple views belong to the same handheldgripper episode.
<table><tr><td>Domain</td><td>Data</td><td>Coverage and observations</td></tr><tr><td>Third-person</td><td>500 videos</td><td>Eight physical coverage categories and 16 scene categories; object motion, interactions, and state changes.</td></tr><tr><td>Egocentric</td><td>500 videos</td><td>118 tasks; 15 scenario groups; RGB and hand tracking; approximately 2.21 hours.</td></tr><tr><td>Handheld gripper</td><td>500 episodes</td><td>Nine task groups plus missing hints; three synchronized views and paired gripper trajectories.</td></tr></table>

![](images/d4045ccc262a06dad2a2d9f8ae57a136a4ffb985a4b97a486cec4b99b3d4c8ac.jpg)  
Figure 3. Evidence-grounded physical annotation. (a) Domain-adapted observations guide frame selection and visual inspection across third-person, egocentric, and handheld-gripper data. (b) A vision– language annotator produces physical records linked to source frames, organizing entities, events, state changes, and relations. (c) Evidence checks, human cross-review, and structural validation refine the records and captions. (d) A bottle-opening example connects hand roles, event intervals, contact transitions, and object states to a physical caption.

Figure 3(d) illustrates this process with a bottle-opening sequence. The record distinguishes left-hand stabilization from right-hand cap manipulation and follows the cap through grasping, unscrewing, lifting, placement, and release. Event intervals and before-and-after observations connect the action sequence to changes in contact and object state. The resulting caption preserves the two hands’ roles and the cap-removal event. The following sections describe how this workflow is implemented in each domain.

## 4.2. Third-Person Physical Scenes

Third-person videos extend the benchmark beyond intentional manipulation to physical processes observed at the scene level. The collection includes object motion, tool–material interaction, deformation, and other visible changes in the environment. The dataset contains 500 videos spanning eight primary physical coverage categories and 16 scene categories, and retains the source metadata descriptions. Appendix B.1 reports these distributions, and Figure 10 presents source-video examples. Annotation uses the recorded visual sequence to determine the participating entities, their interactions, and the resulting states; source metadata is retained as the comparison caption.

The annotation procedure first organizes the scene into entities and temporally localized events, then records motion, contact, and state changes within each event. Review revisits the corresponding video intervals to correct object references, movement directions, and event boundaries. For the reconstruction study, the physical descriptions are expanded from the human-corrected fields, retaining the information needed to describe the observed process. This produces a direct comparison between the source description and a description grounded in the annotated video. Section 5.3 reports the reconstruction results.

## 4.3. Egocentric Human Manipulation

The egocentric dataset contains 500 videos from EgoDex [7], covering 118 tasks with four or five videos per task. Each video is paired with its hand-tracking record, and the total duration is approximately 2.21 hours. Appendix B.2 details the selection and the scenario, action, object, and physical-phenomenon distributions. Figure 11 illustrates the observed manipulation processes.

Egocentric annotation distinguishes the acting hand, the directly manipulated object, and any object affected through that interaction. HOI-DETR [27] provides hand–object and object–object role predictions, while EgoDex hand poses supply anatomical hand identity and motion cues. The vision–language annotator combines these signals with RGB observations to describe the participants and their interactions. Figure 4 illustrates how the detector’s role structure localizes direct manipulation and tool-mediated interactions.

Hand poses also guide temporal sampling. We transform the palm and five fingertips into camera coordinates, compute hand translation speed and palm angular speed, and combine them into a motion score for each hand. Temporally separated peaks summarize each hand’s active moments. For visual inspection, we select peaks from the maximum of the two hands’ translation speeds and merge them with uniformly sampled frames. This combines coverage of the full sequence with closer inspection of rapid hand movements. Appendix C gives the equations and selection parameters.

Using these frames and auxiliary cues, the annotator assigns entity identities, separates action phases, and records changes in contact and object state. Camera-relative motion helps distinguish the two hands’ contributions, and RGB observations connect these movements to the manipulated objects. Event intervals preserve concurrent actions, while a continuous timeline retains sustained holding and intervening movements alongside salient transitions.

![](images/a678b023392fc5ff0a8b3eccb6f00efc02bb178e5b6bf854cf011013317786fb.jpg)  
Figure 4. Interaction-role cues for egocentric physical annotation. Source RGB frames (left) and HOI-DETR predictions (right). Red, yellow, and cyan boxes denote hands, directly manipulated objects, and objects affected through them; links show predicted interaction relations. Numbers beside boxes and links indicate detection and relation confidence, respectively. The role structure guides annotation of the acting hand, the manipulated object, and tool-mediated interactions, supporting the description of who acts on what.

## 4.4. Multi-View Handheld-Gripper Demonstrations

The third domain contains 500 human-operated handheld-gripper demonstrations from Tencent’s Hy-Embodied-0.5-VLA-Data. It follows the handheld demonstration setting associated with UMI [8]: the observed manipulation is performed by a person controlling the grippers. Each episode provides a fixed high view and two wrist views, synchronized at 30 frames per second, together with left and right gripper trajectories. Each view has resolution 424 × 240. Appendix B.3 reports the selection, task distribution, trajectory profiles, and durations; Figure 12 shows examples of the synchronized observations.

We organize annotation into a global pass and local multi-view refinement [15]. The global pass examines high-view frames to identify scene entities and the broad event sequence. Local refinement then inspects synchronized high and wrist views around candidate transitions, located using gripper position, rotation, and opening changes. It assigns actor identity, matches manipulated objects across views, and resolves contact and state changes before producing the structured record and caption.

Cross-view consistency connects the global event sequence to local interaction details. Each observation records its camera view and source frame, while actor identity is assigned from the visible gripper and its interaction with the object. Motion is described relative to an observable reference to account for wrist-camera movement. High-view observations establish the scene context, and wrist views provide complementary detail about contact and state changes within the same demonstration.

## 5. Evaluation and Downstream Utility

Our evaluation connects physical-caption quality with its use in video generation and manipulation learning. We first compare captioning strategies on third-person and egocentric videos using the dimensions in Section 3.3. We then measure reconstruction fidelity under physical-caption conditioning and policy success under instruction enrichment. Scoring the original and enriched training descriptions links the benchmark evaluation to the policy comparison.

## 5.1. Physical Caption Evaluation Protocol

Caption evaluation measures semantic agreement and coverage along the seven physical dimensions in Table 1. Doubao Seed 2.1 Pro serves as the text judge, with temperature zero and thinking disabled, assigning each dimension an integer score from 0 to 100 and a brief rationale. For egocentric videos, physical mechanisms and causality are scored against the mechanism statements (criteria.physical\_mechanism) in the fixed Gold annotation; the other six dimensions use its short caption (goldCaptionEn).

Equation (1) averages each caption’s scores over the dimensions applicable to its Gold reference. These dimensions are fixed across strategies for the same video. Overall scores weight videos equally; dimensionlevel scores average the applicable entries. Appendix D provides the judge instructions and scoring rubric.

## 5.2. Caption Evaluation Results

The comparisons examine how captioning strategies capture physical content within each domain. Figure 5 reports the third-person results, and Table 4 reports the egocentric results, with references and scoring settings held fixed within each comparison.

Third-person caption evaluation. Physis achieves the highest overall score of 74.07 among the 11 strategies, compared with 72.91 for LingBot Temporal and 54.32 for Meta, and leads in five of the seven physical dimensions (Figure 5). LingBot Temporal leads in temporal order (72.84), while Cosmos3 Narrative leads in physical mechanisms and causality (71.00). The dimension scores distinguish strategies that retain event progression from those that capture richer interaction and mechanism detail.

Egocentric caption evaluation. For egocentric videos, we implement five public captioning strategies on Qwen3-VL-8B-Instruct using their respective prompts and frame-sampling recipes. Physis uses captions from version 0921-2301, and Meta uses the original source captions. Table 4 compares all seven strategies under the same judge and references.

Physis achieves the highest overall score of 79.36, compared with 42.41 for Meta, and scores 92.75 for states and outcomes. Meta scores 49.17 for event coverage, 38.57 for contact and interaction, and 42.50 for temporal order, identifying physical content that its task-level descriptions leave under-specified. Cosmos Ego leads in contact and interaction and temporal order, scoring 83.57 and 77.50, compared with 77.14 and 75.75 for Physis. Mechanism scores further separate the strategies: Physis scores 49.58, Cosmos Ego 46.50, and Meta 7.08.

![](images/cbdf99b0bb5b13e1a9c38e54f84ff6cf8753d597cc6a382eaf9103ca68e3f092.jpg)  
Figure 5. Physical caption evaluation on the third-person dataset. Physis is highlighted in black. The seven axes correspond to physical evaluation dimensions, and each curve represents a captioning strategy. All strategies are evaluated against the same dimension-specific references. Scores range from 0 to 100, with higher scores farther from the center. The LingBot strategies are reconstructed from the paper description. Physis captioning is currently in internal testing.

Table 4. Physical caption evaluation on the egocentric dataset. All strategies are evaluated on the same egocentric clips using the same dimension-specific references. Scores range from 0 to 100; higher is better. Bold indicates the highest score in each row. The overall average is computed from per-video scores over applicable dimensions, as defined in Section 5.1.
<table><tr><td>Physical dimension</td><td>CogVideoX</td><td>Open-Sora v1.3</td><td>EgoVid Dense</td><td>EgoVid Action</td><td>Cosmos Ego</td><td>Physis</td><td>Meta caption</td></tr><tr><td>Event coverage</td><td>60.17</td><td>60.25</td><td>61.50</td><td>57.33</td><td>81.08</td><td>84.42</td><td>49.17</td></tr><tr><td>Objects and roles</td><td>61.17</td><td>62.92</td><td>63.08</td><td>57.25</td><td>78.50</td><td>84.83</td><td>46.67</td></tr><tr><td>Actions and motion</td><td>63.18</td><td>59.91</td><td>62.64</td><td>59.55</td><td>84.64</td><td>87.73</td><td>54.09</td></tr><tr><td>Contact and interaction</td><td>61.43</td><td>54.29</td><td>59.29</td><td>54.29</td><td>83.57</td><td>77.14</td><td>38.57</td></tr><tr><td>States and outcomes Physical mechanisms</td><td>71.88</td><td>68.25</td><td>70.88</td><td>60.62</td><td>80.12</td><td>92.75</td><td>55.00</td></tr><tr><td>and causality</td><td>25.83</td><td>31.42</td><td>30.00</td><td>35.83</td><td>46.50</td><td>49.58</td><td>7.08</td></tr><tr><td>Temporal order</td><td>56.25</td><td>66.25</td><td>63.75</td><td>67.50</td><td>77.50</td><td>75.75</td><td>42.50</td></tr><tr><td>Overall average</td><td>57.16</td><td>58.01</td><td>58.42</td><td>55.58</td><td>75.41</td><td>79.36</td><td>42.41</td></tr></table>

The EgoVid comparison shows how compression shifts descriptive emphasis. EgoVid Action compresses the dense description and scores 55.58 overall, compared with 58.42 for EgoVid Dense. States and outcomes decrease from 70.88 to 60.62, while temporal order increases from 63.75 to 67.50 and mechanisms and causality from 30.00 to 35.83. These opposing changes show why the dimension scores complement the overall average.

## 5.3. Video Reconstruction Experiments

Video reconstruction tests the value of physical descriptions as generation conditions. Each generator receives the first source frame and either the original metadata caption or a physical description, with the frame and target duration fixed across conditions. We evaluate MiniMax H3 with a 33B backbone [20] and LingBot-World 2.0 (14B) [10] in both domains. Third-person descriptions expand the human-corrected physical annotations; egocentric descriptions specify the manipulation sequence, hand roles, and visible changes.

We measure fidelity to the source video using LPIPS [28] for frame-level perceptual differences and perceptual video clip similarity (PVCS) for local-clip differences in pretrained video features.<sup>1</sup> Lower distances indicate better reconstruction. Qualitative comparisons examine the preservation of interaction roles, contact transitions, and object states.

![](images/4c978df70182231753de2a44627c65a197ab99c3eb7d0b621944ce3d7c6da775.jpg)  
Lower is better · Green: relative reduction

![](images/8122ecc1d37f059b877828ed33231a76657e5bdf29fd205e4ae3993b09c031c8.jpg)

![](images/6910b0737952cc4c32a15fbe221e0e2fb5d66abaa7d4c142790514495750a4a9.jpg)

![](images/1fa4b24280b2cbce0c196c85c08c3ac201947399836094fcde92334a641fe976.jpg)  
Figure 6. Physical captions improve video reconstruction across domains. LPIPS and PVCS are shown for third-person (top) and egocentric (bottom) videos using H3 and LingBot-World 2.0. Original captions are source metadata captions. Point labels report mean distances; green annotations show relative reductions under physical-caption conditioning. Lower is better.

Physical captions improve reconstruction in all four model–domain settings (Figure 6). LPIPS decreases by 8.6–9.5% on third-person videos and 26.9–30.9% on egocentric videos; PVCS decreases by 8.9–9.0% and 27.9–28.1%, respectively. The larger relative reductions on egocentric videos hold for both gen erators and both metrics. Appendix E details the reconstruction setup and third-person confidence

## intervals.

Physical conditioning better preserves the hand roles and state transition in Figure 7. The source video shows the left hand stabilizing a small object’s base while the right hand rotates and lifts its shell to expose the interior. The physical-caption reconstruction retains this division of roles and the shell–base separation. The metadata caption describes assembling a stool, and its reconstruction follows that object and activity. The comparison highlights the contribution of describing the acting hand, affected part, and direction of change.

![](images/c98e743373e5289f2f34730f0734ad26936de6837346403b7f1a5afa59c7757b.jpg)  
Figure 7. Physical captions preserve interaction roles and state changes in reconstruction. Rows show source frames (top), H3 with the physical caption (middle; labeled Gold in the source experiment), and H3 with the metadata caption (bottom). Columns correspond to the same times. Physical conditioning better retains left-hand stabilization, right-hand shell removal, and the transition from a covered to an exposed interior. Metadata conditioning instead generates the assembly of a larger stool.

Figure 8 extends the comparison to deformation, tool–material interaction, and evolving material distributions. Each example juxtaposes the source video with H3 reconstructions conditioned on the physical caption (Gold) and the original metadata caption (Meta).

Shape after unloading (physcap9k\_08653). Gold preserves the recovery of volumetric toy shapes after the board is raised, matching the source video’s transition from compression to recovery. The physical caption explicitly describes this return toward the original shape. The metadata description ends with revealing deformed toys, and Meta exposes flattened, sheet-like forms.

Drilling and contact termination (physcap9k\_02537). Gold better preserves localized drilling debris and the final separation of the drill bit from the workpiece. Both details appear in the physical caption, which describes debris around the hole followed by upward withdrawal. Meta describes continuous drilling, produces more extensive flying chips, and keeps the bit at the drilling location in the final displayed frame.

Plume structure and barrel orientation (physcap9k\_09822). Gold more closely preserves the persistent white–turquoise plume, its occlusion of the background, and the changing barrel orientation. The physical caption tracks the plume’s upward and outward expansion as the barrel tips onto its side. Meta emphasizes thin blue liquid sheets and spray that recede earlier.

Together, these examples illustrate the physical content preserved by caption conditioning: participant roles, contact transitions, object-state changes, and material evolution.

## 5.4. Manipulation-Policy Fine-Tuning

We evaluate physical descriptions as training instructions through a controlled single-task study and a comparison across four policy families on two benchmarks. PhysCapBench scores quantify the change in description quality, and task success measures its downstream utility (Table 5).

Table 5. Physical-description quality and policy performance under instruction enrichment. (a) PhysCapBench scores of training descriptions. (b) Single-task success on 20 standard and 20 randomized shared scenes (40 overall), with demonstrations and training settings held fixed. (c,d) Official baseline and enriched-instruction success rates, averaged over six selected tasks in each benchmark. Gains are differences between displayed values, in score points (a) or percentage points (b–d). Table 6 reports per-task results.
<table><tr><td>Evaluation</td><td>Original</td><td>Enriched</td><td>Gain</td></tr><tr><td colspan="4">(a) Physical-caption score</td></tr><tr><td>RoboTwin 2.0</td><td>27.27</td><td>69.44</td><td>+42.17</td></tr><tr><td>RoboDojo Sim</td><td>34.68</td><td>73.26</td><td>+38.58</td></tr><tr><td colspan="4">(b) Single-task success (%)</td></tr><tr><td>Standard scenes</td><td>50.0</td><td>55.0</td><td>+5.0</td></tr><tr><td>Randomized scenes</td><td>25.0</td><td>35.0</td><td>+10.0</td></tr><tr><td>Overall</td><td>37.5</td><td>45.0</td><td>+7.5</td></tr><tr><td colspan="4">(c) RoboTwin 2.0: mean success (%)</td></tr><tr><td>π0.5</td><td>50.4</td><td>51.9</td><td>+1.5</td></tr><tr><td>X-VLA</td><td>44.4</td><td>45.4</td><td>+1.0</td></tr><tr><td>GalaxeaVLA</td><td>29.1</td><td>30.1</td><td>+1.0</td></tr><tr><td>starVLA</td><td>23.5</td><td>24.0</td><td>+0.5</td></tr><tr><td colspan="4">(d) RoboDojo Sim: mean success (%)</td></tr><tr><td>π0.5</td><td>30.6</td><td>32.6</td><td>+2.0</td></tr><tr><td>X-VLA</td><td>30.9</td><td>32.2</td><td>+1.3</td></tr><tr><td>GalaxeaVLA</td><td>13.6</td><td>14.9</td><td>+1.3</td></tr><tr><td>starVLA</td><td>13.7</td><td>14.3</td><td>+0.6</td></tr></table>

Physical-caption scores of training instructions. Instruction enrichment raises PhysCapBench scores from 27.27 to 69.44 on RoboTwin 2.0 and from 34.68 to 73.26 on RoboDojo Sim, gains of 42.17 and 38.58 points, respectively (Table 5(a)). These original and enriched descriptions provide the training instructions for the multi-task policy comparison.

Controlled single-task comparison. On an independent RoboTwin 2.0 subtask [21], we fine-tune two policies from the released �<sub>0.5</sub> base model [22] using the same 50 demonstrations and 23,098 frames. Images, robot states, action targets, and repeated sampling indices are fixed across conditions. The original condition uses the task instruction; the enriched condition replaces it for half of the training indices with a description of the demonstrated sequence, acting gripper, target placement, and postrelease state.

Both policies receive 5,000 LoRA updates with batch size 32 and training seed 0. Evaluation uses 40

## Source

Gold (physical)  
(a) Shape after unloading physcap9k\_08653  
![](images/378eb4bf8a02cac0c42c0f7d5a044ce6fcd67377aa88bd4ff3df2650c9c93511.jpg)  
Figure 8. Physical captions preserve state transitions and interaction details in third-person reconstruction. Columns compare the source video, H3 conditioned on the physical caption (Gold), and H3 conditioned on the metadata caption (Meta). Each row uses the same timestamp. (a) After the board is lifted, Gold retains volumetric toy shapes, while Meta exposes flattened, sheet-like shapes; the 9.50 s row shows both generated conditions after unloading. (b) Gold better retains localized drilling debris and the final separation of the drill bit from the workpiece. (c) Gold more closely preserves the persistent white–turquoise plume and barrel orientation, while Meta emphasizes thin liquid sheets and spray.

shared held-out initial scenes: 20 standard and 20 randomized. Each policy performs one rollout per scene using the original short instruction, current images, and robot state, with success determined by the native task criterion. Appendix E.2 details instruction construction, training, and closed-loop execution.

Instruction enrichment raises success from 37.5% to 45.0%, a gain of 7.5 percentage points (Table 5(b)).   
Gains reach 5.0 percentage points in standard scenes and 10.0 percentage points in randomized scenes.   
Enriched training descriptions thus improve execution with the original short task instruction.

Multi-task policy evaluation. We extend the comparison to � , X-VLA, GalaxeaVLA, and starVLA on six selected tasks from each of RoboTwin 2.0 and RoboDojo Sim. Table 5(c,d) compares the mean success rates of official baselines and enriched-instruction policies; Table 6 provides the per-task results.

Table 6. Per-task policy success under instruction enrichment. Entries show official baseline → enriched-instruction success rate (%). Column headers give additional successful cases per task. Each mean summarizes the six selected tasks in its benchmark.
<table><tr><td>Task</td><td>π0.5 (+3 cases)</td><td>X-VLA (+2 cases)</td><td>GalaxeaVLA (+2 cases)</td><td>starVLA (+1 case)</td></tr><tr><td>(a) RoboTwin 2.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Sort blocks by size</td><td>33.0 → 34.5</td><td>40.5 → 41.5</td><td>9.5 → 10.5</td><td>7.0 → 7.5</td></tr><tr><td>Grasp two bottles</td><td>64.5 → 66.0</td><td>46.0 → 47.0</td><td>49.0 → 50.0</td><td>40.0 → 40.5</td></tr><tr><td>Place an object in a cabinet</td><td>30.5 → 32.0</td><td>31.0 → 32.0</td><td>21.0 → 22.0</td><td>14.5 → 15.0</td></tr><tr><td>Stack three bowls</td><td>66.5 → 68.0</td><td>43.0 → 44.0</td><td>32.0 → 33.0</td><td>23.0 → 23.5</td></tr><tr><td>Stamp</td><td>37.0 → 38.5</td><td>26.0 → 27.0</td><td>17.5 → 18.5</td><td>6.0 → 6.5</td></tr><tr><td>Press a stapler</td><td>71.0 → 72.5</td><td>80.0 → 81.0</td><td>45.5 → 46.5</td><td>50.5 → 51.0</td></tr><tr><td>Mean (selected six tasks)</td><td>50.4 → 51.9</td><td>44.4 → 45.4</td><td>29.1 → 30.1</td><td>23.5 → 24.0</td></tr><tr><td>(b) RoboDojo Sim</td><td></td><td></td><td></td><td></td></tr><tr><td>Stack Bowls</td><td>38.0 → 40.0</td><td>26.0 → 27.3</td><td>16.7 → 18.0</td><td>5.3 → 6.0</td></tr><tr><td>Fold Clothes</td><td>24.0 → 26.0</td><td>22.0 → 23.3</td><td>8.7 → 10.0</td><td>8.0 → 8.7</td></tr><tr><td>Pour Liquid Into Cup</td><td>14.7 → 16.7</td><td>13.3 → 14.7</td><td>6.0 → 7.3</td><td>9.3 → 10.0</td></tr><tr><td>Build Tower</td><td>24.0 → 26.0</td><td>53.3 → 54.7</td><td>8.0 → 9.3</td><td>26.0 → 26.7</td></tr><tr><td>Put Bottles Into Dustbin</td><td>69.3 → 71.3</td><td>62.0 → 63.3</td><td>33.3 → 34.7</td><td>28.0 → 28.7</td></tr><tr><td>Pour Balls Into Vase</td><td>13.3 → 15.3</td><td>8.7 → 10.0</td><td>8.7 → 10.0</td><td>5.3 → 6.0</td></tr><tr><td>Mean (selected six tasks)</td><td>30.6 → 32.6 30.9 → 32.2</td><td></td><td>13.6 → 14.9</td><td>13.7 → 14.3</td></tr></table>

All four policies improve on every selected task in both benchmarks. For �<sub>0.5</sub>, mean success increases from 50.4% to 51.9% on RoboTwin 2.0 and from 30.6% to 32.6% on RoboDojo Sim. The same instructionenrichment scheme therefore improves both physical-caption scores and policy success across the evaluated policy families, supporting PhysCapBench’s use in assessing training-description quality.

Figure 9 shows a RoboDojo bowl-stacking example in which the policy fine-tuned with Training Instruction Upsampler completes a stable three-bowl stack.

![](images/b4cb497bad8a03aa5c99604a5d8a33788ccdfc9b6c799898eb20f8df430dae11.jpg)  
Figure 9. Robot manipulation with Training Instruction Upsampler. Frames compare a failed rollout (top) with a successful rollout from a policy fine-tuned using Training Instruction Upsampler (bottom) on RoboDojo. The failed rollout leaves bowls tilted and overturned; the fine-tuned policy completes a stable three-bowl stack.

## 6. Discussion and Limitations

PhysCapBench connects physical caption assessment with the data and applications used in world modeling. Its current scope suggests three directions for further development.

## 6.1. Data and Annotation Coverage

The three domains provide complementary observations of physical scenes and manipulation, supported by domain-adapted annotation and human cross-checking. Coverage remains tied to the selected video sources and their observable interactions. Extending scene diversity and temporal coverage would broaden the range of physical processes represented while retaining the current annotation structure.

## 6.2. Evaluation Consistency

The evaluation uses a fixed text judge to assess agreement with physical references. Scores depend on the reference content and the judge’s interpretation of the rubric. Comparisons with human ratings and additional judges would help characterize scoring consistency across domains and captioning styles.

## 6.3. Downstream Applications

Video reconstruction and manipulation-policy fine-tuning demonstrate the utility of physical descriptions in the evaluated settings. The contribution of each physical dimension to these gains remains to be examined. Controlled changes to individual description components and broader task coverage would help identify which physical information is most useful for different downstream models.

## 7. Conclusion

We present PhysCapBench, a benchmark for physically grounded video captioning across third-person scenes, egocentric human manipulation, and multi-view human-operated handheld-gripper demonstrations. Human-verified annotations and seven physical evaluation dimensions support systematic assessment of interactions and state evolution. Experiments show improved video reconstruction with physical captions and higher manipulation success with enriched training instructions, alongside gains in PhysCapBench scores. PhysCapBench provides a basis for developing and evaluating physical descriptions as language supervision for world models and embodied learning.

## References

[1] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. URL https://arxiv.org/abs/1803.10122. 3

[2] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. Genie: Generative interactive environments. arXiv preprint arXiv:2402.15391, 2024. URL https: //arxiv.org/abs/2402.15391.

[3] NVIDIA. Cosmos world foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025. URL https://arxiv.org/abs/2501.03575. 3

[4] Pengfei Zhou, Shengcong Chen, Di Chen, Jiaxu Wang, Rongjun Jin, Bingwen Zhu, Yike Pan, Songen Gu, Kuanning Wang, Shufeng Nan, Xingyu Qiu, Chenhao Qiu, Pu Yang, Yunuo Cai, Jianxiong Gao, Yifan Li, Yanwei Fu, Xiangyu Yue, Zhi Chen, and Jianlan Luo. �<sub>0</sub>-WM: A Unified Video-Action World Model for Robotic Manipulation. arXiv preprint arXiv:2606.01027, 2026. URL https: //arxiv.org/abs/2606.01027. 3, 5

[5] Open X-Embodiment Collaboration, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv preprint arXiv:2310.08864, 2023. URL https://arxiv.org/abs/2310.08864.

[6] Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. URL https://arxiv.org/abs/2403 .12945.

[7] Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video, 2025. URL https://arxiv.org/abs/ 2505.11709. 4, 10

[8] Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems XX, RSS2024. Robotics: Science and Systems Foundation, July 2024. doi: 10.15607/rss.2024.xx.045. URL http://dx.doi.org/10.15607/RSS.2024.XX. 045. 5, 11

[9] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Kaipeng Zhang, and Tong He. OmniWorld: A multidomain and multi-modal dataset for 4D world modeling. In International Conference on Learning Representations (ICLR), 2026. URL https://yangzhou24.github.io/OmniWorld/. 3, 5

[10] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite Worlds with Versatile Interactions. arXiv preprint arXiv:2607.07534, 2026. URL https://arxiv.org/abs/2607.07534. 3, 14

[11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI technical report, February 2024. URL https://openai.c om/index/video-generation-models-as-world-simulators/. 3

[12] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70M: Captioning 70M videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13320–13331, 2024. URL https://arxiv.org/abs/2402.19479. 3

[13] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025. URL https: //arxiv.org/abs/2408.06072. 3

[14] Liming Lu, Xianzheng Ma, Wenkun He, Guanqi Zhan, Yilin Zhao, Junyu Chen, Mengyao Xu, Jiaojiao Fan, Wenhang Ge, Yuchao Gu, Yunze Liu, Boyi Li, Zhen Dong, Victor Prisacariu, Ming-Yu Liu, Song Han, and Han Cai. Physis-Lang: Self-evolving language as a physical representation for video world model. arXiv preprint arXiv:2609.40358, 2026. URL https://arxiv.org/abs/2609.40358. 3, 4, 5

[15] Xintong Hu, Xuhong Huang, Jinyu Zhang, Yutong Yao, Yuchong Sun, Qiuyue Wang, Mingsheng Li, Sicheng Xie, Yitao Liu, Junhao Chen, et al. FineVLA: Fine-grained instruction alignment for steerable vision-language-action policies. arXiv preprint arXiv:2605.27284, 2026. 3, 5, 11

[16] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models, 2024. URL https://arxiv.org/abs/2407.00634. 3, 4

[17] Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. VidCapBench: A comprehensive benchmark of video captioning for controllable text-to-video generation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 8543–8563, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.449. URL https://aclanthology.org/2025.findings-acl.449/. 3, 4, 5

[18] Shi-Xue Zhang, Hongfa Wang, Duojun Huang, Xin Li, Xiaobin Zhu, and Xu-Cheng Yin. VCaps-Bench: A large-scale fine-grained benchmark for video caption quality evaluation. Proceedings of the AAAI Conference on Artificial Intelligence, 40(15):12726–12734, March 2026. ISSN 2159-5399. doi: 10.1609/aaai.v40i15.38269. URL http://dx.doi.org/10.1609/aaai.v40i15.38269. 3, 4

[19] Kaixiang Qiu, Minghao Han, Keliang Liu, Yizhou Liu, Jinghang Han, Yue Jiang, Xuecheng Wu, Shunli Wang, Lihua Zhang, and Dingkang Yang. OmniFysics-Captioner technical report: Grounding omni-modal understanding in the physical world for better captioning. arXiv preprint arXiv:2609.31714, 2026. URL https://arxiv.org/abs/2609.31714. 3, 4

[20] MiniMax. MiniMax H3: An open model breaking the boundaries between tasks and modalities. Official model release, July 2026. URL https://www.minimax.io/blog/minimax-h3. 3, 14

[21] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2506.18088. 3, 16

[22] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan

Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. �<sub>0.5</sub>: a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025. URL https://arxiv.org/abs/2504.16054. 3, 16

[23] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. PhysBench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations, 2025. URL https://physbench.github.io/. 4

[24] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. VideoPhy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2406.03520. 4

[25] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, Eugene Byrne, Zach Chavis, Joya Chen, Feng Cheng, Fu-Jen Chu, Sean Crane, Avijit Dasgupta, Jing Dong, Maria Escobar, Cristhian Forigua, Abrham Gebreselasie, Sanjay Haresh, Jing Huang, Md Mohaiminul Islam, Suyog Jain, Rawal Khirodkar, Devansh Kukreja, Kevin J Liang, Jia-Wei Liu, Sagnik Majumder, Yongsen Mao, Miguel Martin, Effrosyni Mavroudi, Tushar Nagarajan, Francesco Ragusa, Santhosh Kumar Ramakrishnan, Luigi Seminara, Arjun Somayazulu, Yale Song, Shan Su, Zihui Xue, Edward Zhang, Jinxu Zhang, Angela Castillo, Changan Chen, Xinzhu Fu, Ryosuke Furuta, Cristina Gonzalez, Prince Gupta, Jiabo Hu, Yifei Huang, Yiming Huang, Weslie Khoo, Anush Kumar, Robert Kuo, Sach Lakhavani, Miao Liu, Mi Luo, Zhengyi Luo, Brighid Meredith, Austin Miller, Oluwatumininu Oguntola, Xiaqing Pan, Penny Peng, Shraman Pramanick, Merey Ramazanova, Fiona Ryan, Wei Shan, Kiran Somasundaram, Chenan Song, Audrey Southerland, Masatoshi Tateno, Huiyu Wang, Yuchen Wang, Takuma Yagi, Mingfei Yan, Xitong Yang, Zecheng Yu, Shengxin Cindy Zha, Chen Zhao, Ziwei Zhao, Zhifan Zhu, Jeff Zhuo, Pablo Arbelaez, Gedas Bertasius, Dima Damen, Jakob Engel, Giovanni Maria Farinella, Antonino Furnari, Bernard Ghanem, Judy Hoffman, C.V. Jawahar, Richard Newcombe, Hyun Soo Park, James M. Rehg, Yoichi Sato, Manolis Savva, Jianbo Shi, Mike Zheng Shou, and Michael Wray. Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19383–19400, June 2024. 4

[26] Macrodata Labs. WGO-Bench: What’s going on benchmark. Hugging Face dataset, 2026. URL https://huggingface.co/datasets/macrodata/WGO-Bench. Dataset card, accessed October 23, 2026. 5

[27] Ahmad Darkhalil, Dima Damen, and David Fouhey. Improving and evaluating hand-object interaction detection. arXiv preprint arXiv:2606.17384, 2026. URL https://arxiv.org/abs/2606.1 7384. 10

[28] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018. doi: 10.1109/CVPR.2018.00068. 14

[29] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? arXiv preprint arXiv:2501.09038, 2025. URL https: //arxiv.org/abs/2501.09038.

[30] Wenzhen Yuan, Siyuan Dong, and Edward H. Adelson. GelSight: High-resolution robot tactile sensors for estimating geometry and force. Sensors, 17(12):2762, 2017. doi: 10.3390/s17122762. URL https://www.mdpi.com/1424-8220/17/12/2762.

## A. Annotation Schema and Domain Profiles

The annotation schema connects entity identities and states to events, sequence-level relations, and visual evidence. Tables 7–9 expand the representation in Section 3.2, using the field names of the anno-

tation schemas and interfaces. Captions express the recorded physical process in connected text and provide references for the evaluation in Section 5.1.

## A.1. Entities and Initial States

Entity records establish the identities and starting configuration needed to describe subsequent changes. The objects\_and\_states record links each entity to its observed states and supporting frames, while initial spatial relations specify the scene configuration. Table 7 lists these fields and their domain-specific evidence references.

Table 7. Entity and initial-scene fields. The first seven rows describe entity records; the remaining rows describe initial spatial relations and their evidence.
<table><tr><td>Field / record</td><td>Meaning</td></tr><tr><td>object_id, object_name</td><td>Entity identifier and descriptive name.</td></tr><tr><td>initial_state,final_state</td><td>Descriptions of the observed initial and final states.</td></tr><tr><td>local_observation_ids</td><td>Links to local observations associated with an entity.</td></tr><tr><td>instance_group, instance_index</td><td>Group and instance identifiers for distinguishing similar entities.</td></tr><tr><td>identity_basis</td><td>Recorded basis for associating observations with the same</td></tr><tr><td>identity_reference_frame_indices</td><td>entity. Source frames supporting the entity identification.</td></tr><tr><td>identity_visible_frame_ranges</td><td>Frame intervals in which the entity is visible.</td></tr><tr><td>initial_spatial_relations</td><td>Scene-level relations among entities at the start of the</td></tr><tr><td>subject,reference_object,relation</td><td>recording. Within an initial relation: the related entity, its reference entity, and their spatial relationship.</td></tr><tr><td>evidence_frame_indices (Ego), witness_ids (UMI)</td><td>Evidence attached to an initial relation: source frames in Ego and witness references in UMI.</td></tr></table>

## A.2. Event Fields

Event records describe how entities participate in a physical process: as actors, instruments, targets, moving entities, or entities undergoing state changes. An entity may occupy multiple roles. Table 8 groups the fields by timing, participants, motion, interaction, and event relations; Table 9 specifies domain-specific evidence and state-transition links.

Table 8. Detailed event fields. Fields appear within each event\_chain entry. The table specifies their semantic roles; domain profiles retain their own value vocabularies and evidence structures.
<table><tr><td>Field / record</td><td>Meaning</td></tr><tr><td>event_id</td><td>Identifier used to link the event to other records.</td></tr><tr><td>frame_span, time_span</td><td>Event interval in source frames and corresponding seconds.</td></tr><tr><td>boundary_tolerance_frames, boundary_ source</td><td>Boundary tolerance and provenance of the proposed boundary.</td></tr><tr><td>actors</td><td>Entities that perform the action.</td></tr><tr><td>instrument_objects</td><td>Objects used as instruments by the acting entities.</td></tr><tr><td>action_type</td><td>Action category assigned to the event.</td></tr><tr><td>target_object</td><td>Object directly targeted by the action, when applicable.</td></tr><tr><td>motion_subjects</td><td>Entities whose motion is described.</td></tr><tr><td>motion_type,motion_direction</td><td>Type and direction of the recorded motion.</td></tr><tr><td>motion_reference_object</td><td>Reference entity used for an object-relative motion description.</td></tr><tr><td>contact_subjects, contact_object</td><td>Entities participating in contact and the contacted counterpart.</td></tr><tr><td>contact relation</td><td>Contact or interaction label; its vocabulary depends on the domain profile.</td></tr><tr><td>state_change_subjects,state_change</td><td>Entities whose states change and the category of that change.</td></tr><tr><td>spatial_relation_changes</td><td>Changes in spatial relationships during the event.</td></tr><tr><td>temporal_relation,causal_relation evidence</td><td>Temporal or supported causal relation to another event. Description of the observations supporting the event.</td></tr></table>

## A.3. Sequence Records and Domain Extensions

Sequence records combine events into a temporal account that includes persistent states and overlapping actions. Mechanism records link explanations to supporting events, and domain extensions add explicit state transitions and camera-specific evidence (Table 9). Mechanism participants use actor in the UMI profile and actor\_ids in the egocentric profile; event-level actors lists the acting entities.

Table 9. Sequence-level records and domain-specific fields. Ego and UMI rows identify the profiles in which the listed field forms occur. The caption fields store the natural-language descriptions associated with the structured records.
<table><tr><td>Scope</td><td>Fields / record</td><td>Meaning</td></tr><tr><td>Timeline</td><td>timeline_segments,linked_event_ids</td><td>Segments organize the sequence and identify active events; source-frame and time intervals locate each segment.</td></tr><tr><td>Mechanisms</td><td>physical mechanism,mechanism type, supporting_event_ids</td><td>Mechanism statements are connected to their supporting events.</td></tr><tr><td>Ego</td><td>mechanism_id,actor_ids, target_ object_ids, statement, statement_zh</td><td>Fields within each mechanism record identify participants and retain English and Chinese descriptions.</td></tr><tr><td>UMI</td><td>actor, description, witness_ids</td><td>Fields within the mechanism object identify an actor, describe the mechanism, and link visual evidence.</td></tr><tr><td>UMI</td><td>state_delta_ids,state_deltas</td><td>Event links to explicit state-transition records.</td></tr><tr><td>UMI</td><td>removed, added, predicate, subject, object, polarity</td><td>State-transition records contain removed and added state atoms, each encoding a relation and its polarity.</td></tr><tr><td>Ego</td><td>evidence_frame_indices,manual_ review_status</td><td>Frame evidence and review status associated with an event.</td></tr><tr><td>UMI evidence</td><td>witness_ids,rgb_witnesses</td><td>Annotation records reference witness IDs; witness details reside in a separate evidence sidecar with view, frames, observability, and review status.</td></tr><tr><td>UMI</td><td>entity_type</td><td>Entity category in the handheld-gripper profile.</td></tr><tr><td>Taxonomy</td><td>taxonomy_labels</td><td>Domain-specific labels summarizing the annotated content.</td></tr><tr><td>Ego caption</td><td>goldCaptionEn, goldCaptionZh</td><td>English and Chinese reference-caption text.</td></tr><tr><td rowspan="2">UMI caption</td><td>caption.short, caption.dense,</td><td>Short and dense descriptions, with an</td></tr><tr><td>caption.language</td><td>explicit language field.</td></tr></table>

## A.4. Temporal and Observation Conventions

Timing is expressed in source-frame coordinates. Event intervals use half-open ranges [�, �), and dividing their boundaries by the source frame rate gives the corresponding times. Evidence frames must lie within the recording. Events may overlap to represent simultaneous actions; timeline segments partition the sequence and identify the events active in each segment. Entity references distinguish actors, moving entities, manipulated objects, and affected targets, with hands acting directly and objects serving as instruments when they transmit the action.

Observation records preserve actor identity and the viewpoint supporting each annotation. Egocentric annotations use anatomical left and right hand identities and record ambiguity when handedness is not visible. Handheld-gripper annotations identify left and right grippers and associate observations with the high, left-wrist, or right-wrist view. State transitions record removed and added state atoms; linked witness records specify the view, source frames, observability, and review status for cross-view verification. Visibility and physical state occupy separate fields.

## B. Data Coverage and Video Examples

Dataset coverage is organized by physical phenomena, scenes, tasks, and trajectory profiles, followed by source-video examples. The distributions use manifest labels and count videos or demonstration episodes. Egocentric physical-phenomenon tags allow multiple labels per video.

## B.1. Third-Person Coverage

The 500 third-person videos span eight primary physical categories and 16 scene categories, covering fluid and granular processes, motion and contact, and deformation across home, workshop, public, and outdoor settings. Each video contributes once to each distribution in Tables 10 and 11; its annotations record additional phenomena present in the sequence.

Table 10. Third-person primary physical coverage. Each video has one label in this grouping; counts sum to 500.
<table><tr><td>Primary coverage category</td><td>Count</td><td>Share</td></tr><tr><td>Fluid phenomena</td><td>161</td><td>32.2%</td></tr><tr><td>Gravity and support</td><td>80</td><td>16.0%</td></tr><tr><td>Granular phenomena</td><td>65</td><td>13.0%</td></tr><tr><td>Rotation and friction</td><td>60</td><td>12.0%</td></tr><tr><td>Manipulation and state change</td><td>43</td><td>8.6%</td></tr><tr><td>Collision and momentum</td><td>35</td><td>7.0%</td></tr><tr><td>Combustion, gas, and other phenomena</td><td>28</td><td>5.6%</td></tr><tr><td>Deformation and breakage</td><td>28</td><td>5.6%</td></tr><tr><td>Total</td><td>500</td><td>100.0%</td></tr></table>

Table 11. Third-person scene coverage. Each video has one label in this grouping; counts sum to 500.
<table><tr><td>Scene category</td><td>Count</td><td>Share</td></tr><tr><td>Home</td><td>155</td><td>31.0%</td></tr><tr><td>Industrial workshop</td><td>118</td><td>23.6%</td></tr><tr><td>Public space</td><td>73</td><td>14.6%</td></tr><tr><td>Natural outdoor setting</td><td>53</td><td>10.6%</td></tr><tr><td>Waterfront and coast</td><td>34</td><td>6.8%</td></tr><tr><td>Laboratory</td><td>16</td><td>3.2%</td></tr><tr><td>Urban street</td><td>14</td><td>2.8%</td></tr><tr><td>Commercial setting</td><td>10</td><td>2.0%</td></tr><tr><td>Fields and countryside</td><td>9</td><td>1.8%</td></tr><tr><td>Mountains and forest</td><td>8</td><td>1.6%</td></tr><tr><td>Snow-covered setting</td><td>4</td><td>0.8%</td></tr><tr><td>Other outdoor setting</td><td>2</td><td>0.4%</td></tr><tr><td>Indoor gymnasium</td><td>1</td><td>0.2%</td></tr><tr><td>Indoor ski venue</td><td>1</td><td>0.2%</td></tr><tr><td>Indoor sports venue</td><td>1</td><td>0.2%</td></tr><tr><td>Aerial setting</td><td>1</td><td>0.2%</td></tr><tr><td>Total</td><td>500</td><td>100.0%</td></tr></table>

## B.2. Egocentric Coverage

Egocentric selection balances task coverage with diversity in scenarios, actions, and objects. From 1,413 individually captioned videos in an index of 338,234 videos, we select four examples per task across 118 tasks. The remaining 28 slots broaden coverage of descriptions, scenarios, actions, objects, and source archives. Every selected video includes a source caption and a paired hand-tracking file. Table 12 summarizes the 15 scenario groups. The source-archive distribution is 111 videos from EgoDex test, 83 from part1, 76 from part5, 75 from part3, 64 from part4, 49 from extra, and 42 from part2.

Table 12. Egocentric scenario coverage. Categories are derived from task names, with metadata captions used for disambiguation. Counts sum to 500.
<table><tr><td>Scenario group</td><td>Videos</td><td>Share</td></tr><tr><td>Board games and puzzles</td><td>53</td><td>10.6%</td></tr><tr><td>Container opening, closing, and storage</td><td>52</td><td>10.4%</td></tr><tr><td>Kitchen cleaning and tableware organization</td><td>46</td><td>9.2%</td></tr><tr><td>Cooking and food preparation</td><td>45</td><td>9.0%</td></tr><tr><td>Electronic-device interaction</td><td>44</td><td>8.8%</td></tr><tr><td>Paper crafts, writing, and drawing</td><td>44</td><td>8.8%</td></tr><tr><td>Furniture assembly and disassembly</td><td>36</td><td>7.2%</td></tr><tr><td>Blocks and structural assembly</td><td>34</td><td>6.8%</td></tr><tr><td>Textiles, cords, and soft objects</td><td>29</td><td>5.8%</td></tr><tr><td>Personal care and dressing</td><td>28</td><td>5.6%</td></tr><tr><td>Cleaning and surface maintenance</td><td>21</td><td>4.2%</td></tr><tr><td>Toys and fine hand manipulation</td><td>21</td><td>4.2%</td></tr><tr><td>Ball handling, throwing, and catching</td><td>18</td><td>3.6%</td></tr><tr><td>Tools, locks, and fasteners</td><td>16</td><td>3.2%</td></tr><tr><td>General tabletop transport and organization</td><td>13</td><td>2.6%</td></tr></table>

Physical-phenomenon tags describe processes that co-occur across scenarios (Table 13). Multiple tags per video yield a total tag count above 500. These video-level labels summarize coverage; event annotations resolve the processes over time.

Table 13. Egocentric physical-phenomenon coverage tags. Videos can carry multiple coverage tags; percentages use 500 videos as the denominator.
<table><tr><td>Coverage tag</td><td>Videos</td><td>Share</td></tr><tr><td>Rigid-object transport and placement</td><td>267</td><td>53.4%</td></tr><tr><td>Insertion and extraction constraints</td><td>145</td><td>29.0%</td></tr><tr><td>Connection and separation</td><td>103</td><td>20.6%</td></tr><tr><td>Stacking and stability</td><td>102</td><td>20.4%</td></tr><tr><td>Deformation and folding</td><td>58</td><td>11.6%</td></tr><tr><td>Friction and wiping</td><td>53</td><td>10.6%</td></tr><tr><td>Rolling and collision</td><td>36</td><td>7.2%</td></tr><tr><td>Contact and support</td><td>34</td><td>6.8%</td></tr><tr><td>Rotation and threaded fastening</td><td>32</td><td>6.4%</td></tr><tr><td>Liquid flow</td><td>28</td><td>5.6%</td></tr><tr><td>Gravity, falling, throwing, and catching</td><td>26</td><td>5.2%</td></tr><tr><td>Thermal processes and cooking</td><td>12</td><td>2.4%</td></tr><tr><td>Cutting, tearing, and breakage</td><td>4</td><td>0.8%</td></tr></table>

Action and object classifications further characterize manipulation diversity. Table 14 reports 18 action families and 14 object categories. Each video receives one label in each classification, alongside its multilabel physical-phenomenon tags.

Table 14. Egocentric action and object coverage. The two halves present independently sorted distributions over the same 500 videos, with one action-family label and one object-category label per video.

<table><tr><td>Action family</td><td></td><td>Videos Object category</td><td>Videos</td></tr><tr><td>Insertion / removal</td><td></td><td>92 Blocks, puzzles, and game pieces</td><td>68</td></tr><tr><td>Assembly / disassembly</td><td></td><td>57 Tableware and kitchen containers</td><td>60</td></tr><tr><td>Fastening / unfastening</td><td></td><td>41 Furniture and structural parts</td><td>53</td></tr><tr><td>Grasping / transport / throw-catch</td><td></td><td>40 Paper, cards, and stationery</td><td>50</td></tr><tr><td>Folding / deformation / wrapping</td><td></td><td>38 Food, ice, and liquids</td><td>46</td></tr><tr><td>Games / fine manipulation</td><td></td><td>33 Tools, locks, and fasteners</td><td>40</td></tr><tr><td>Stacking / unstacking</td><td></td><td>29 Electronic devices and accessories</td><td>39</td></tr><tr><td>Cleaning / wiping</td><td></td><td>28 Textiles, cords, and soft objects</td><td>37</td></tr><tr><td>Arrangement / organization</td><td></td><td>26 Storage containers and bags</td><td>31</td></tr><tr><td>Rolling / rotation / flipping</td><td></td><td>22 Small toys and parts</td><td>21</td></tr><tr><td>Writing / drawing / input</td><td></td><td>20 Balls and objects for throwing/catching</td><td>18</td></tr><tr><td>Washing</td><td></td><td>20 General tabletop objects</td><td>13</td></tr><tr><td>Cooking / heating</td><td></td><td>16 Personal-care and wearable items</td><td>12</td></tr><tr><td>Pouring / scooping / dispensing</td><td></td><td>12 Cleaning items and surfaces</td><td>12</td></tr><tr><td>Connection / disconnection</td><td>10</td><td></td><td></td></tr><tr><td>Dressing / personal care</td><td>8</td><td></td><td></td></tr><tr><td>Measurement / tool use</td><td>4</td><td></td><td></td></tr><tr><td>Food handling</td><td>4</td><td></td><td></td></tr><tr><td>Total</td><td>500 Total</td><td></td><td>500</td></tr></table>

## B.3. Handheld-Gripper Coverage

Handheld-gripper selection combines data-validity checks with task diversity. We check video availability and decoding, synchronization with metadata and trajectories, sufficient duration, and finite action/state values, retaining 58,318 eligible episodes from 66,806 recordings. We then select 500 episodes across the task categories in Table 15. Episodes without task hints are annotated from their visual observations.

Table 15. Task categories of the handheld-gripper dataset. Categories summarize source task intent for selection; annotation determines the observed process from video.
<table><tr><td>Task category</td><td>Episodes</td><td>Share</td></tr><tr><td>Placement, storage, and return</td><td>55</td><td>11.0%</td></tr><tr><td>Grasping, picking, and removal</td><td>55</td><td>11.0%</td></tr><tr><td>Opening, closing, pulling, and switch operation</td><td>55</td><td>11.0%</td></tr><tr><td>Position and orientation adjustment</td><td>55</td><td>11.0%</td></tr><tr><td>Clothing and flexible-object organization</td><td>54</td><td>10.8%</td></tr><tr><td>Table setting and tableware clearing</td><td>54</td><td>10.8%</td></tr><tr><td>Wiping, sweeping, and disposal</td><td>54</td><td>10.8%</td></tr><tr><td>Tool and device interaction</td><td>54</td><td>10.8%</td></tr><tr><td>Transfer, pouring, and replenishment</td><td>54</td><td>10.8%</td></tr><tr><td>Missing task hint</td><td>10</td><td>2.0%</td></tr></table>

Trajectory profiles complement task categories by characterizing overall motion, rotation, opening range, and left–right activity (Table 16). Each episode receives one profile during selection. Episodes range from 4.0 to 89.7 seconds, with a median duration of 28.1 seconds, a 90th percentile of 64.6 seconds, and approximately 4.82 hours in total.

Table 16. Handheld-gripper trajectory profiles. Each episode has one label in this grouping; counts sum to 500.
<table><tr><td>Trajectory profile</td><td>Count</td><td>Share</td></tr><tr><td>High motion</td><td>108</td><td>21.6%</td></tr><tr><td>Right-dominant motion</td><td>77</td><td>15.4%</td></tr><tr><td>Balanced bimanual motion</td><td>65</td><td>13.0%</td></tr><tr><td>Rotation-rich trajectories</td><td>65</td><td>13.0%</td></tr><tr><td>Left-dominant motion</td><td>64</td><td>12.8%</td></tr><tr><td>Large gripper-opening range</td><td>61</td><td>12.2%</td></tr><tr><td>Low motion</td><td>60</td><td>12.0%</td></tr><tr><td>Total</td><td>500</td><td>100.0%</td></tr></table>

## B.4. Video Examples across Three Domains

Figures 10–12 illustrate the physical processes through three source-video examples per domain. Ordered frames show changes over time, and synchronized views reveal complementary aspects of handheldgripper manipulation. Sample identifiers and timestamps link the displayed frames to their source recordings.

## (a) Surface compaction

� = 10.37 s  
![](images/cc69e4cf398923668d6f3a8b395c9f2fd1e9c0b31ca403db76ad30a29dfb61f3.jpg)  
� = 0.00 s

![](images/a2d4dd2094381083fd537a4b6abd4cd256c389d03171dafbb08535c8373282b4.jpg)  
� = 2.73 s

![](images/214c320772051b9f31f05e76c04767e20fbf695170fb064d184ff293bf3a754e.jpg)  
� = 5.50 s  
physcap9k\_02382

![](images/a19e600c07990709344d92114d690d1cfff90c9a35036c241fa7a3e241f9043e.jpg)  
(b) Ejection from a barrel  
� = 8.23 s  
physcap9k\_09822

![](images/e804c6277ef7b65e41c50ca93465d4633fa1e479a1ed7582679dbb1e1c1230d5.jpg)

![](images/99059a00176d5d9a9351fc176894fe25f92c9c605e1448e3e3cc5c31acb74729.jpg)

![](images/345fc0a37ddb7c6ceb34b80122fbb2d7e4c7a640a3e49a00b293617580ed0217.jpg)  
(c) Soft-object compression and recovery

![](images/688c378b5f57b8250703f72059126cd81251e88f3dbef1d539ab9895cfb1ea44.jpg)  
� = 6.93 s

![](images/046d1a7eb5729099fa6a5693a68c482d7dde040ce5a3a8c367fa34f00f986229.jpg)  
� = 0.00 s

![](images/e5dbe28b9fe607c6cb8c0dcf813dd57dfd481a975a13f6298eb459321fd88116.jpg)  
� = 2.90 s

![](images/0d698b8ffd5d6b021ee88b850177e7c39285beac7ab9d33cac299fc99cc3ab51.jpg)  
� = 7.70 s  
physcap9k\_08653

![](images/7c4bfa875daf5179dbdfa3c0777f41ef6bf2e7125cbf79e075fc9bdb22ff1294.jpg)  
� = 18.27 s

Figure 10. Third-person source-video examples. Rows show surface compaction, material ejection from a barrel, and compression and recovery of soft objects. Frames within each row follow temporal order, with timestamps relative to the start of the clip. Sample IDs identify the videos in the dataset manifest.

## (a) Unpacking and unfolding a chessboard

![](images/bcf8646b8ab932b88d5978f860296cf5575f780181c7680ec2137246f44070c9.jpg)  
� = 0.00 s

![](images/934b623df601f9a918d7325e7cc0c680edc28d544e66552864b208603940fa25.jpg)

![](images/dd92367b9037196f45e591172b0981e687c4999c2a200f0b0be5b76b921d47e6.jpg)  
(b) Separating a shell from its base

egodex500\_0073  
![](images/d66414c51fce526f0cf15acc7b72b4541908d78db9ffef29d891eb1cded2b445.jpg)  
� = 11.20 s

![](images/19a34af9d1be86aeb011361c24a1f716feff97d57e1612b3a46e487681b6e827.jpg)  
� = 0.00 s  
egodex500\_0289

![](images/c1ecf448996d787ea6e81c332866def56636cdb7cc2496c678ac0d7041805ae5.jpg)  
� = 2.23 s

![](images/51b83c215efe9a0f0f2833a01a1e493505631a39f16b24efccd230f72fd8155d.jpg)  
� = 4.50 s

![](images/f93e5a77990ecb25b42f598e1743c6317b6430a0d41d478e203018cd77d5a23c.jpg)  
� = 6.73 s

(c) Disassembling a small structure  
![](images/3108d0866b42b426b5af78d12a788ac0994e6206fbc74871a0e07e2eed40f2ea.jpg)  
� = 0.00 s

![](images/6efcd612538c6d59620c5ef02a15077d1359b47ebadac1d190b671f750b9f9d5.jpg)  
� = 2.77 s

![](images/9222a0bfca833a3d6ec4d7a7b645ec17a5709d13b3249b20548230ea08b75bfc.jpg)  
� = 5.60 s  
egodex500\_0480

![](images/f17713904630ffc1bbb5bfdd1e6aebddb148bec37ec6bed25460ccc5201c4616.jpg)  
� = 8.37 s  
Figure 11. Egocentric source-video examples. Rows show unpacking and unfolding a chessboard, separating an outer shell from its base, and disassembling a small structure. The observations retain both hands and the manipulated objects across time. Timestamps and sample IDs follow the same convention as Figure 10.

![](images/b5f79fe6dde06bc6e8861640594bd4d06e4dd8b0eb1332c0a6c71c69801da84c.jpg)  
Figure 12. Handheld-gripper source-video examples. Three episodes show object placement, lid removal, and textile transfer, each at two time points. Triptychs present synchronized left-wrist, high, and right-wrist views from left to right. Timestamps are measured from the start of each episode.

## C. Annotation Prompts and Review Guidelines

The annotation workflow combines domain-specific cues, structured prompting, and human review. Auxiliary cues select informative observations; prompts organize the observed process into entities, events, and state changes; reviewers verify the resulting records against the video.

## C.1. Auxiliary Cues and Frame Selection

Hand-pose motion cues. For egocentric videos, the camera pose $C _ { t }$ and each hand-joint pose $J _ { t }$ map local coordinates to world coordinates at frame �. We obtain the camera-relative joint pose as $C _ { t } ^ { - 1 } J _ { t } .$ . For each hand, let $p _ { t }$ be the centroid of the palm and five fingertips in camera coordinates, $R _ { t }$ the camerarelative palm rotation, and � the video frame rate. Translation speed, angular speed, and the hand motion score are

$$
\nu _ { t } = f \| p _ { t } - p _ { t - 1 } \| _ { 2 } ,\tag{2}
$$

$$
\omega _ { t } = f \operatorname { a r c c o s } \left( \mathrm { c l i p } \left( \frac { \mathrm { t r } ( R _ { t - 1 } ^ { \mathsf { T } } R _ { t } ) - 1 } { 2 } , - 1 , 1 \right) \right) ,\tag{3}
$$

$$
m _ { t } = \nu _ { t } + 0 . 0 5 \omega _ { t } .\tag{4}
$$

We initialize speeds to zero at the first frame and retain up to six peaks of $m _ { t }$ per hand as motion summaries. Candidates are selected in descending score order, excluding any candidate within eight frames of a previously selected peak.

Motion-guided visual sampling. For the visual frame set, we form the combined translation-speed curve $q _ { t } = \operatorname* { m a x } ( \nu _ { t } ^ { L } , \nu _ { t } ^ { R } )$ , where � and � denote the anatomical left and right hands. We select up to eight peaks from $q _ { t }$ with the same temporal exclusion rule. Let � contain these combined-speed peak frames and � contain up to 12 uniformly spaced frames over the full video. The selected keyframe set is

$$
F = \operatorname { U n i q u e S o r t } ( U \cup P ) .\tag{5}
$$

Uniform frames cover the sequence, and motion peaks focus inspection on active manipulation intervals.   
Per-hand motion summaries accompany the selected frames to distinguish the two hands’ contributions.

Interaction-role cues. HOI-DETR supplies spatial interaction cues from source frames at a detection threshold of 0.5. Outputs include bounding boxes, confidence scores, and hand–object and object–object relations. Associating adjacent detections produces tracks and relation intervals, which the annotator combines with hand-motion summaries and RGB observations to describe the physical process.

Handheld-gripper trajectory cues. Handheld-gripper trajectories guide the inspection of interaction transitions. Each gripper provides a position, rotation quaternion, and opening value; derived translation speed, rotation speed, acceleration, and opening changes identify candidate transitions. The high view establishes the overall sequence, and synchronized wrist views resolve local interactions. Camera identifiers locate observations, actor identifiers specify the participating grippers, and ambiguous intervals receive further visual review.

## C.2. Prompt Content

Annotation prompts request a structured record and a caption of the observed physical process. They establish entity identities and initial states, then order events in time and record actors, affected objects, motion, contact, and state changes. Each event includes source-frame evidence. The caption integrates these records into a coherent account of hand or gripper roles, interactions, and visible outcomes.

Visual evidence anchors event boundaries, grasp outcomes, and state transitions. Motion peaks and gripper-opening changes guide frame selection; RGB observations establish the events and before–after states. Occluded intervals retain the surrounding observations and visibility status for review.

## C.3. Review and Consistency Checks

Seven professional data annotators perform annotation and cross-checking, with human verification of every final record. Review follows the structure of the physical process: event occurrence, actor and object attribution, and the resulting change. For manipulation sequences, reviewers distinguish reaching, contact establishment, object motion, and contact termination, while retaining sustained holding and repositioning in the timeline.

Multi-view review checks entity identity and before–after states across cameras, distinguishing object motion from wrist-camera motion. Structural validation then checks record references, frame bounds, timing, and timeline coverage. Review decisions accompany the initial generated records and final human-verified annotations.

## D. Evaluation Instructions and Scoring Rubric

## D.1. Dimension-Wise Caption Scoring

The scoring prompt provides the candidate caption, dimension-specific references, and physical-dimension definitions to Doubao Seed 2.1 Pro (doubao-seed-2-1-pro-260628), with temperature zero and thinking disabled. Egocentric references use the short Gold caption for six dimensions and structured Gold mechanism statements for mechanisms and causality. The judge returns an integer score from 0 to 100 and a short rationale for each applicable dimension, following this instruction:

Compare the caption with the supplied reference for each applicable physical dimension. Assess semantic agreement and coverage using the bands in Table 17. Briefly identify matched content, omissions, and contradictions. Additional details absent from the reference receive neither credit nor a penalty unless they contradict it. Assess causal explanations explicitly expressed in the caption.

The seven dimensions cover events, participant identities and roles, actions and motion, contact and interaction, states and outcomes, physical mechanisms and causality, and temporal order. Reference content determines applicability, which is held fixed across strategies for each video

Mechanism scoring assesses explicit causal explanations, including those expressed in ordinary language. Action descriptions such as grasping, lifting, or moving are distinguished from explanations of the underlying mechanism. Within the lowest band of Table 17, scores of 1–24 indicate weak or fragmentary causal content; zero indicates no expressed reference mechanism or wholly incompatible content.

Table 17. Per-dimension scoring rubric. Each applicable dimension receives an integer score. The judge considers both agreement and information coverage.
<table><tr><td>Score</td><td>Criterion</td></tr><tr><td>90-100</td><td>Nearly complete coverage with no material contradiction.</td></tr><tr><td>75-89</td><td>Main content agrees, with minor omissions.</td></tr><tr><td>50-74</td><td>Partial agreement, with important omissions or conflicts.</td></tr><tr><td>25-49</td><td>Little agreement, with major omissions or conflicts.</td></tr><tr><td>0-24</td><td>No meaningful agreement, or reversal of core facts.</td></tr></table>

## D.2. Score Aggregation

For each video, Eq. (1) averages scores over �(�), the dimensions with assessable reference content. Omissions within applicable dimensions affect the score, including a score of zero when the candidate expresses none of the reference mechanisms. Dimensions without assessable reference content are excluded. Overall scores give each video equal weight; dimension-level scores average the applicable entries. All captioning strategies use this aggregation procedure.

## E. Generation and Policy Experiment Configurations

## E.1. Reconstruction Setup and Statistical Analysis

Reconstruction conditions share the source first frame and target duration, with one output generated for each caption condition. Results are paired by source video and summarized by generator and domain in Figure 6. Relative reduction is $( m - p ) / m \times 1 0 0 \%$ , where � and � denote the metadata- and physical-caption means. Paired bootstrap resampling over source videos gives the third-person confidence intervals in Table 18.

Table 18. Paired differences in third-person reconstruction distance. The 95% bootstrap intervals use physical minus metadata distance; negative values favor physical captions.
<table><tr><td>Model</td><td>Metric 95% interval</td></tr><tr><td>H3</td><td>LPIPS [-0.0588,-0.0295]</td></tr><tr><td>LingBot</td><td>PVCS [-0.3402,-0.1311]</td></tr><tr><td></td><td>[-0.0670,-0.0294]</td></tr><tr><td>LPIPS PVCS</td><td>[-0.3720, -0.1648]</td></tr></table>

## E.2. Manipulation-Policy Configuration

The single-task experiment evaluates instruction enrichment on RoboTwin 2.0’s block-size ordering sub task using an Aloha-AgileX bimanual embodiment. The policy arranges three blocks from largest to smallest across the center of the table. Both conditions use 50 demonstrations containing 23,098 frames, three camera views, a 14-dimensional robot state, and corresponding action targets. The original instruction states the ordering goal; enriched instructions add the demonstrated sequence, gripper roles, and resulting placement and support relations. We construct these instructions from demonstration events and scene information, verify them against keyframes, and keep them fixed within each demonstration.

Enriched instructions replace the original text for half of the repeated training indices and contain 151– 167 tokens under the recorded tokenizer, within the 256-token language limit. The remaining training indices and all evaluation rollouts use the original short task instruction. Table 19 lists the shared training and execution settings.

Table 19. Shared policy-training and execution settings. The language substitution described above is the experimental difference.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Initialization</td><td>Released pi05_base weights</td></tr><tr><td>Adaptation</td><td>OpenPI JAX LoRA</td></tr><tr><td>Training updates / batch size</td><td>5,000 / 32</td></tr><tr><td>Training seed</td><td>0 for both conditions</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning-rate schedule</td><td>200 warm-up steps; peak  $5 \times 1 0 ^ { - 5 } .$   $5 \times 1 0 ^ { - 6 }$  ; cosine decay to</td></tr><tr><td>Language limit</td><td>256 tokens</td></tr><tr><td>Checkpoint selection</td><td>Final checkpoint after 5,000 updates</td></tr><tr><td>Prediction chunk</td><td>50 steps of 14-dimensional absolute joint actions</td></tr><tr><td>Replanning</td><td>Execute the first 10 predicted steps, then observe again</td></tr><tr><td>Rollout limit</td><td>1,200 executed action steps</td></tr><tr><td>Evaluation scenes</td><td>20 standard and 20 randomized; identical across policies</td></tr></table>

1: Intro | 2: Related Work | 3: Design | 4: Data | 5: Evaluation | 6: Discussion | 7: Conclusion References | Appendix A: Schema | B: Data | C: Annotation | D: Scoring | E: Experiments
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Rollouts</td><td>One per policy per scene; 80 in total</td></tr><tr><td>Success</td><td>Native task success criterion</td></tr></table>

Each held-out scene is checked for solvability with the expert policy before evaluation. Randomized scenes vary backgrounds, clutter, lighting, and table height.