# Human-Inspired Social Engagement Analysis via Interpretable Mutual Visual Attention

Urwa Fatima<sup>1</sup> , Mohammad Zohaib<sup>1,2,3</sup> , Francesca Odone<sup>1,2</sup> , and Nicoletta Noceti<sup>1,2</sup>

DIBRIS-Università degli Studi di Genova, Genova, IT MaLGa - Machine Learning Genoa center, Genova, IT 3 Istituto Italiano di Tecnologia, Genova, IT

Abstract. Understanding social interactions from non-verbal visual data is important for behavior analysis and activity monitoring. We propose an interpretable computational model of social engagement inspired by psychological theories of mutual visual attention. Rather than learning interaction patterns end-to-end, our framework explicitly models dyadic visual attention and aggregates these cues into interpretable measures of individual and group engagement. The resulting modular framework combines state-of-the-art head orientation estimation with lightweight geometric reasoning, producing explanations that remain accessible to non-technical users. We evaluate the proposed approach on a variety of data through quantitative experiments and demonstrate its practical usefulness with qualitative visualizations designed to support teachers, caregivers, and social workers in understanding group interaction dynamics.

Keywords: Non-verbal interaction · Head pose · Group engagement

## 1 Introduction

In the last decades, the importance of non-verbal signals in characterizing social interactions has become widely recognized. These subtle cues frequently convey critical information about human dynamics, emotions, and interpersonal relationships [24, 29, 30, 34]. Psychological theories, particularly the Theory of Mind [9], highlight the fundamental role of non-verbal behaviours—especially mutual gaze and face-to-face attention—in conveying engagement and regulating social interactions. Despite this growing interest, computational approaches relying exclusively on non-verbal cues remain relatively limited, and are often restricted to solving individual tasks such as head pose estimation, body pose estimation, or action recognition [7, 14, 15].

In this work, we propose an interpretable and modular framework for videobased social interaction analysis that is grounded in a hierarchical model of human social reasoning [19,20]. According to this perspective, group engagement is not directly observed, but emerges from the network of interpersonal interactions among group members. Accordingly, our framework first models dyadic interactions from non-verbal visual cues and then aggregates these local relationships into interpretable measures of individual and group engagement [2, 31].

More specifically, given a video of a small- or medium-sized group, the proposed pipeline progressively builds a multi-level representation of the social scene. Starting from head direction estimation, it infers pairwise visual attention and classifies dyadic interaction states, which are subsequently aggregated to compute individual- and group-level engagement by means of indices we designed to the purpose. The modular design makes each reasoning stage explicit and interpretable, enabling richer analysis and visualization than end-to-end approaches, where intermediate social cues are typically not explicitly represented.

We evaluate the proposed framework from both quantitative and qualitative perspectives. Quantitatively, we assess the representativeness of the pipeline in recognizing interaction patterns. Qualitatively, we show how the proposed engagement indices capture meaningful variations in group engagement across different interaction scenarios. To facilitate the interpretation of these results, we also introduce a set of visualization strategies that make the analysis accessible to users with little or no technical expertise. Indeed, the final goal of our research is to provide psychologists and educators, with easy to use tools to study social interactions in small groups.

For this we also carried out a live demonstration in a kindergarten, where our tool was used to visually describe children’s engagement during a cooperative activity. Although the collected data can not be shared due to ethical constraints, teachers provided a very positive qualitative feedback on the interpretability and potential usefulness of the proposed visualizations. To summarise, the main contributions of our work are the following:

– A modular video-based framework for hierarchical social interaction analysis, from pairwise attention estimation to group engagement.

– Two engagement indices that quantify interaction at both the individual and group levels.

– An intuitive visualization interface that makes multi-level social interaction analysis accessible to non-technical users.

## 2 Related Works

A common principle in cognitively inspired approaches to social interaction analysis is that group-level behaviors emerge from pairwise interpersonal dynamics. This view is well established in social psychology through the Social Relations Model (SRM), which represents group interactions as the composition of actor, partner, and relationship efects, treating the dyad as the fundamental unit of social behavior [19, 20]. More recent methodological frameworks extend this perspective to the analysis of groups, discussing hierarchical models in which group-level phenomena are inferred from dyadic interactions [18].

This hierarchical assumption has also influenced human-centered computer vision. In line with psychology literature, existing approaches view group engagement analysis as an emergent property arising from the network of social interactions among participants. In particular, Social Signal Processing assessed the use of gaze, head and body orientation, interpersonal distance, and conversational dynamics as key nonverbal cues for inferring social relationships and interaction patterns [36,37]. Similarly, research on F-formations has demonstrated that these cues can be efectively combined to identify conversational partners and free-standing interaction groups [6, 32]. More recently, comprehensive surveys on co-located human interaction analysis have highlighted hierarchical computational frameworks in which low-level behavioral cues are first integrated to estimate pairwise interpersonal relations, later exploited to infer higher-level group properties such as engagement, cohesion, rapport, and leadership [4, 12]. Of particular relevance for our work are approaches leveraging the concept of attention. In the past, mutual facing and shared attention have been exploited to detect dyadic engagement [3,8], or to model group structure [32]. More recent deep learning approaches jointly reason over individuals and groups but require large datasets and lack interpretability [21, 22]. In our work, we leverage head direction as a proxy for gaze direction and attention [1, 9, 13, 33].

![](images/a5285479e1a3b6f9001cad434ef005c46abb5749fa6b06043f8243cbc6d9b82f.jpg)  
Fig. 1: Proposed four-stage framework.

Tightly coupled with social interaction is the concept of group engagement. Most previous works on engagement estimation focus on individuals. Head poses have been correlated with engagement in learning environments [38], while dualstream architectures combine facial and scene context [10]. In [16], visual behavior and motion are used to estimate a continuous engagement for children with autism spectrum disorder. Transformer-based approaches have also been proposed to classify behavioral and collaborative engagement states using combined visual cues [27]. Group engagement remains less explored, though recent work analyzes non-verbal behaviors as indicators of group engagement quality [25,26].

## 3 Proposed Framework

Human-inspired design principle: rather than directly predicting engagement from visual data, we mirror a hierarchical reasoning assessed in social cognition and interaction literature [9,19,20]. Our design follows three principles: (i) attention is an observable cue from which engagement is inferred; (ii) reciprocal attention provides stronger evidence of interaction than one-sided attention; and (iii) group engagement emerges from the composition of local dyadic interactions. These principles lead to the design of a four stages framework, shown in Fig. 1.

Stage 1: Human Pose Estimation: Given an image, we extract each person’s pose and assign a unique ID for tracking, using YOLOv11 with ByteTrack for eficient multi-person pose estimation and tracking under occlusions. We retain five facial keypoints: the nose, left eye, right eye, left ear, and right ear, each one with its own confidence, setting missing points to $( 0 , 0 )$ with zero confidence. Stage 2: Head Direction Estimation: Given the set of five facial 2D keypoints with confidence scores $\{ ( x _ { i } , y _ { i } , c _ { i } ) \} _ { i = 1 } ^ { 5 }$ estimated in stage 1, stage 2 aims to estimate head orientation as a triplet (yaw, pitch, roll). To the purpose, we first normalize facial keypoints to make them invariant to scale and absolute face position. Let (¯x, y¯) be the centroid of the facial keypoints; the normalized keypoints $\{ ( \tilde { x } _ { i } , \tilde { y } _ { i } , c _ { i } ) \} _ { i = 1 } ^ { 5 }$ are computed as $\begin{array} { r } { \tilde { x } _ { i } ~ = ~ \frac { x _ { i } - \bar { x } } { d _ { x } ^ { \mathrm { m a x } } } } \end{array}$ , and $\begin{array} { r } { \tilde { y } _ { i } ~ = ~ \frac { y _ { i } - \bar { y } } { d _ { u } ^ { \mathrm { m a x } } } } \end{array}$ , where $d _ { x } ^ { \operatorname* { m a x } } =$ max<sub>i</sub> $( | x _ { i } - { \bar { x } } | )$ and $d _ { y } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i } \left( \left| y _ { i } - { \bar { y } } \right| \right)$ are computed over keypoints with $c _ { i } > 0$ . Keypoints with zero confidence remain at (0, 0) after normalization. The normalized keypoints are finally fed to HHP-Net [5, 35] for the angles triplet prediction. As output, HHP-Net predicts the head orientation angles $( \theta , \alpha , \gamma )$ along with their respective uncertainties $( \sigma _ { \theta } , \sigma _ { \alpha } , \sigma _ { \gamma } )$ . These angles are used to generate a 2D unit vector $\hat { h }$ on the image plane, according to the Tait-Bryan angles. The head direction is first represented as a vector $\boldsymbol { h } = ( h _ { x } , h _ { y } )$ in pixel/image coordinates, as $h _ { x } = \sin ( \theta )$ and $h _ { y } = - \cos ( \theta )$ sin(α) and then normalized to the unit norm as $\begin{array} { r } { \hat { h } = \frac { h } { \| h \| } } \end{array}$ . The keypoint centroid $( { \bar { x } } , { \bar { y } } )$ is used as the origin $\mathcal { O }$ of the head direction vector. This process is repeated for all n people present in the image to obtain head direction vectors $\mathcal { H } = \{ \hat { h } _ { j } \} _ { j = 1 } ^ { n }$ and their origins $O = \{ \mathcal { O } _ { j } \} _ { j = 1 } ^ { n }$ . These vectors are used in the next section to compute spatial relations between people.

Stage 3: Interaction State Classification: Following prior work on mutual gaze [11,23], interactions are categorized as bidirectional, unidirectional, or noninteracting state, corresponding to mutual gaze, one-sided gaze, and no gaze between individuals. We detect these three interaction states among n humans using head direction vectors $\hat { h } .$ . We adopt a strategy presented in [35] which estimates the Looking At Each Other (LAEO) event (bidirectional state). We extend it to also identify unidirectional and non-interaction states as follows.

Consider a pair of individuals, $p _ { 1 }$ and $p _ { 2 }$ , with head direction vectors and origins $( \hat { h } _ { 1 } , \mathcal { O } _ { 1 } )$ and $( \hat { h } _ { 2 } , \mathcal { O } _ { 2 } )$ , respectively. We compute their mutual visual attention in two steps. First, we compute the unit vectors between their head origins: $\hat { d } _ { 1 2 }$ points from $p _ { 1 }$ to $p _ { 2 }$ and is computed as $\frac { \mathcal { O } _ { 2 } - \mathcal { O } _ { 1 } } { \| \mathcal { O } _ { 2 } - \mathcal { O } _ { 1 } \| }$ , while $\hat { d } _ { 2 1 }$ points from p<sub>2</sub> to $p _ { 1 }$ and is simply the negative of $\hat { d } _ { 1 2 }$ . Second, we compute the angles $\phi _ { 1 2 } = \operatorname { a r c c o s } ( \hat { d } _ { 1 2 } \cdot \hat { h } _ { 1 } )$ , and $\phi _ { 2 1 } = \operatorname { a r c c o s } ( \hat { d } _ { 2 1 } \cdot \hat { h } _ { 2 } )$ between each person’s head direction vector and the corresponding unit vector to the other person. These angles are used to measure mutual head-direction alignment and to classify interactions between $p _ { 1 }$ and $p _ { 2 }$ . If both $\phi _ { 1 2 }$ and $\phi _ { 2 1 }$ are lower than a threshold $\tau _ { : }$ the interaction is bidirectional; if only one of them is lower than $\tau ,$ it is unidirectional; otherwise it is non-interacting. The procedure is applied to all $\frac { n ( n - 1 ) } { 2 }$ pairs in the image. This pairwise strategy does not guarantee consistency in the prediction. For instance, in a scene with only two individuals $p _ { 1 }$ and $p _ { 2 }$ having $p _ { 1 }$ predicted as bidirectional and $p _ { 2 }$ as unidirectional would be inconsistent. Therefore, we first prioritize pairs classified as bidirectional interactions, assigning all members of these pairs to the bidirectional state. Next, we consider unidirectional interaction pairs and assign all remaining unassigned members to the unidirectional state, where the person whose angle is below $\tau$ is considered the looker and the other the target. Finally, any remaining members are classified as non-interacting. This yields the interaction state of each person per frame, which is then applied sequentially across all video frames.

Stage 4: Social Interaction Analysis: We propose two engagement indices. Individual Engagement Index – $I E I ( p )$ : IEI represents each person’s participation in social interactions. For a given person $p ,$ it is computed from the number of frames spent in each interaction state with weights. Let $F _ { B } , F _ { L }$ , and $F _ { T }$ denote the frames where a person is classified as bidirectional, unidirectional looker, and unidirectional target. Then the $\operatorname { I E I } ( p )$ is computed as:

$$
\mathrm { I E I } ( p ) = \frac { 1 0 0 } { N } \sum _ { c \in \{ B , L , T \} } \omega _ { c } F _ { c }\tag{1}
$$

where $N$ is the number of frames in which person p is detected, and $\omega _ { B } , \omega _ { L }$ , and $\omega _ { T }$ are weights for each interaction state in engagement analysis. We empirically set the weights to $\omega _ { B } = 1 . 0 0 , \omega _ { L } = 0 . 3 0 .$ , and $\omega _ { T } = 0 . 0 5$ , respectively. Mutual events indicate the highest engagement and thus receive the highest weight. In unidirectional interactions, the looker is more engaged than the target, so $\omega _ { L } >$ ω<sub>T</sub>. IEI(p) ranges from 0 to 100, where 0 indicates a non-interacting person and 100 corresponds to a person remaining in the bidirectional state throughout the video. Comparing $\operatorname { I E I } ( p )$ across participants helps identify highly engaged individuals and diferences in attention during the interaction.

Group Engagement Score (GES): It provides a group-level analysis of social interaction across the entire video. Given the total number of people n in each frame $t \in [ 1 , N ]$ , which we assume not to change over time, the GES is computed as a weighted ratio between the predicted engagement $\hat { E }$ and the maximum possible engagement $E ^ { m a x }$ over a video of $N$ frames. For the n subjects in a frame $t ,$ the maximum possible number of bidirectional $B _ { t } ^ { m a x }$ and unidirectional $U _ { t } ^ { m a x }$ states are $\lfloor n / 2 \rfloor$ and $n \% 2$ , respectively. Let $\hat { B } _ { t }$ and $\hat { U } _ { t }$ denote the predicted number of bidirectional and unidirectional states in frame t. $\beta _ { \mathrm { B } } = 1$ and $\beta _ { \mathrm { U } } = 0 . 5$ are their respective weights. Then

$$
\boldsymbol { E } ^ { \mathrm { m a x } } = \sum _ { t = 1 } ^ { N } \left( \beta _ { \mathrm { B } } \boldsymbol { B } _ { t } ^ { \mathrm { m a x } } + \beta _ { \mathrm { U } } \boldsymbol { U } _ { t } ^ { \mathrm { m a x } } \right) , \quad \hat { \boldsymbol { E } } = \sum _ { t = 1 } ^ { N } \left( \beta _ { \mathrm { B } } \hat { B } _ { t } + \beta _ { \mathrm { U } } \hat { \boldsymbol { U } } _ { t } \right) .\tag{2}
$$

and $\begin{array} { r } { G E S \ = \ 1 0 0 \frac { \hat { E } } { { \cal E } ^ { m a x } } } \end{array}$ if $E ^ { m a x } ~ > ~ 0 \nonumber$ , 0 otherwise. A score of 0 indicates no interaction, while 100 indicates the maximum possible interaction.

## 4 Experiments and Results

Benchmarks. The experimental assessment aims to evaluate our approach in diferent environments and conditions and for this reason we consider several benchmarks. First we consider the $G P { - } S t a t i c { + } + $ dataset [28],which includes 71 videos of dyadic interactions $( \sim 1 . 8 - 1 7 \mathrm { s e c } )$ . To the best of our knowledge, it is the only available dataset with appropriate annotations. On this dataset we provide a quantitative assessment of interaction classification. In addition, we consider 3 triadic conversational scenes selected from the CMU Panoptic Haggling dataset [17], where three participants engage in face-to-face verbal interaction (∼ 4 – 6 min). The sequences have been manually annotated according to the annotation of GP-Static++, to allow a comparison of performance for dyads and triplets. For a qualitative evaluation of visualizations and engagement indices, we also rely on an additional data source targeting high-resolution group activities across diverse ages and indoor/outdoor settings. To this end we adopted the publicly available unannotated Pexels videos<sup>4</sup> (15 videos, ∼ 0.15 – 2 min).

Table 1: Comparison with the frame-based method [28] on the GP-static++ dataset. In addition to global F1 score on the test set, we also provide the performance of bidirectional events $\left( F 1 _ { B } \right)$ , unidirectional events $( F 1 _ { U } )$ , and no interaction $\left( F 1 _ { N O } \right)$ We also report results on the CMU dataset showing the influence of group cardinality (2 for GP-Static++, 3 for CMU).
<table><tr><td>Dataset</td><td>Frames</td><td>Method</td><td colspan="4"> $\mathrm { F } 1 _ { B }$  ↑F1u ↑  $\operatorname { F } 1 _ { N O }$  ↑ Avg. F1 ↑</td></tr><tr><td>GP-Static++ 13789</td><td></td><td>[28]</td><td>0.56</td><td>0.44</td><td>0.49</td><td>0.49</td></tr><tr><td></td><td></td><td>Ours</td><td>0.70</td><td>0.45</td><td>0.60</td><td>0.58</td></tr><tr><td>CMU</td><td>2368</td><td>Ours</td><td>0.68</td><td>0.29</td><td>0.54</td><td>0.50</td></tr></table>

Implementation Details. We used the yolo11n-pose variant of YOLOv11 with ByteTrack. We set the detection and IoU threshold of YOLO to 0.5. The match and track thresholds of the tracker are set to 0.8 and 0.5, respectively. We empiri cally select for τ the value $\pm 1 5 ^ { \circ }$ . We used an oficial TensorFlow implementation of HHP-Net with pretrained checkpoints, without any additional fine-tuning. All experiments were performed on an ASUS TUF Gaming F15 laptop with an Intel Core i7 processor and 16 GB RAM.

Quantitative Results (Tab. 1). We compare our framework with the framebased method in [28] on the GP-Static++ dataset. We provide the global F1 score on the test set of GP-Static++, together with the performance for each interaction class. Our method provides superior performances in all scenarios. It is also worth mentioning that, diferently from [28], our framework is trainingfree and does not rely on any task-specific training data.

In addition, we evaluate the prediction of our method on the selection of annotated sequences from the CMU dataset. This allows us to provide an evidence of the influence on the number of people in the group. From the F1 scores, we may observe that our method appears to be influenced by the cardinality of the group regarding the correct classification of unidirectional or no interaction, while only a small degradation is observed for mutual events, likely because of the robustness thanks to the stronger geometric constraints.

Qualitative results (Fig. 2). Each person is assigned a unique ID and their head direction is indicated by a yellow arrow. Edges represent interactions, with green and blue denoting bidirectional and unidirectional cases. Edge values indicate interaction strength. The horizontal bar plot shows interaction events over time (left in the figure). Each bar represents a person, with color-coded segments indicating interaction states. Partner IDs and interaction durations (in seconds) are shown above each bar, providing an overview of interaction continuity, frequency, and partner switching. For a compact summary, we also provide histogram-based visualisations (right in the figure).

![](images/3162a201fc80093ddba8d08a42a1dd0be790320ff46c746f419affd796a7bed6.jpg)

![](images/719de4b2197d56bf274c72c9ffaccf3088f7e56be68f0e964a39933dc192f9ee.jpg)

![](images/05582d372a8939ecae3217fd949de921859a8962bab595d29c114958025ac114.jpg)

(a)  
![](images/1022a36f787ddead3e00de6e761aa2a3f895ad524a1f2eec3571d5263fab2d0f.jpg)

![](images/c9e34954fac0615756d3b38380f121d9b0111ec620a9c056e3b85a616ac4fd65.jpg)  
(b)  
Fig. 2: Visualization tool on a sequence from Pexels (Fig. 2a) and CMU (Fig. 2b). In each of them, on the left we report the sequential frames with estimated head directions and interaction states (top), and the interaction states over time for each person with partner IDs (bottom). On the right, we report the corresponding individual interaction analysis (top: frame level interaction counts; bottom: IEI(p) for each person).

We report visualization samples from Pexels (top) and CMU (bottom) datasets. The Pexel sequence depicts a three-person interaction, two children (ID2 and ID3) sit at a table while a woman (ID1) approaches to clean the hands of a child (ID3). Temporal bars indicate that the children interact only during the first 1.3 seconds, while the woman observes them. Afterward, all individuals remain largely isolated. The resulting GES is 40.9. In Fig. 2a, right, we observe a coherent situation. Frame-level interaction counts confirm that non-interaction dominates for the children, while the woman mainly observes the child with ID3. The first shows the number of frames per interaction state for each person (top), while the second reports IEI(p) (bottom), indicating individual engagement. Together, they provide an individual and group interaction dynamics.

The CMU example shows a conversation between three peers who are taking turns in the discussion, and for this reason we notice a more fragmented interaction type (Fig. 2b). In this second example, the GES amounts to 68.9, showing a higher interaction activities with respect to the previous situation. More in gen eral, we empirically observed that the GES index nicely reflects the interaction scenarios. In Fig. 3 we reports examples from the Pexels sequences. On the left, a teacher is instructing the students, each one engaged in their activity. In this case the GES is 31.8, since only the teacher appears to be interacting (i.e. as looker in unidirectional events) with others. The sequence in the middle is very similar to the one reported in Fig. 2b and the GES values are indeed very close. Finally, on the right a family is engaged in a cooperative activity: their GES is 80.6, showing the higher level of interaction in the scene.

![](images/fc16dbc4fc7611fb3368a9a6d2a8318328d1c54e9081d3ce72107944c4c97049.jpg)  
Fig. 3: Examples from the Pexels sequences with diferent interaction scenarios. From left, GES is 31.8, 68.7, and 80.6.

## 5 Conclusion

We presented an interpretable and modular framework for video-based social interaction analysis that follows a human-inspired hierarchical process to infer group engagement from dyadic visual attention cues. Rather than replacing human reasoning with a black-box predictor, the framework provides explicit and interpretable intermediate representations that computationally support it. It combines state-of-the-art pose and head direction estimation with explicit dyadic interaction modelling to derive interpretable measures of engagement at both the individual and group levels. Experiments demonstrate its ability to detect and classify interaction events, while the resulting engagement indices and visualizations make the decision process transparent, enabling teachers, caregivers, and social workers to understand interaction dynamics instead of receiving only engagement scores. The main limitation of the current approach lies in its reliance on 2D pose estimation, which may introduce ambiguities due to the projection of three-dimensional scenes onto the image plane. Future work will extend the framework by incorporating 3D human representations, multimodal cues, speaker attribution, and interaction patterns beyond dyadic relationships. A quantitative evaluation of the engagement indexes is also a future objective.

## Acknowledgements

This work was supported by the European Union — NextGenerationEU and by the Ministry of University and Research (MUR), National Recovery and Resilience Plan (NRRP), Mission 4, Component 2, Investment 1.5, project “RAISE — Robotics and AI for Socio-economic Empowerment” (ECS00000035) We acknowledge the financial support from PNRR MUR Project PE0000013 "Future Artificial Intelligence Research (FAIR)", funded by the European Union – NextGenerationEU, CUP J33C24000430007. This research was partially funded by the European Union - Horizon Europe project “AIRCARE - AI-augmented Robotics for CAncer point of caRE” (101137426), CUP J53C24001360006.

## References

1. Algabri, R., Abdu, A., Lee, S.: Deep learning and machine learning techniques for head pose estimation: a survey. Artificial Intelligence Review 57(10), 288 (2024) 3

2. Argyle, M., Cook, M., Cramer, D.: Gaze and mutual gaze. The British Journal of Psychiatry 165(6), 848–850 (1994) 1

3. Ba, S.O., Odobez, J.M.: Recognizing visual focus of attention from head pose in natural meetings. Transactions on Systems, Man, and Cybernetics, Part B (Cyber netics) 39(1), 16–33 (2008) 3

4. Beyan, C., Vinciarelli, A., Bue, A.D.: Co-located human–human interaction analysis using nonverbal cues: A survey. ACM Computing Surveys 56(5), 1–41 (2023) 3

5. Cantarini, G., Tomenotti, F.F., Noceti, N., Odone, F.: Hhp-net: A light heteroscedastic neural network for head pose estimation with uncertainty. In: Proceedings of the IEEE/CVF Winter Conference on applications of computer vision. pp. 3521–3530 (2022) 4

6. Cristani, M., Raghavendra, R., Del Bue, A., Murino, V.: Human behavior analysis in video surveillance: A social signal processing perspective. Neurocomputing 100, 86–97 (2013) 3

7. Diwan, A., Sunil, R., Mer, P., Mahadeva, R., Patole, S.P.: Advancements in emotion classification via facial and body gesture analysis: A survey. Expert Systems 42(2), e13759 (2025) 1

8. Fathi, A., Hodgins, J.K., Rehg, J.M.: Social interactions: A first-person perspective. In: Conference on Computer Vision and Pattern Recognition. pp. 1226–1233. IEEE (2012) 3

9. Frith, C.D., Frith, U.: Mechanisms of social cognition. Annual review of psychology 63(1), 287–313 (2012) 1, 3

10. Gothwal, P., Banerjee, D., Biswas, A.K.: Vibed-net: Video based engagement detection network using face-aware and scene-aware spatiotemporal cues. arXiv preprint arXiv:2510.18016 (2025) 3

11. Gregorj, A., Yücel, Z., Zanlungo, F., Kanda, T.: On the influence of group social interaction on intrusive behaviours. In: International Conference on Trafic and Granular Flow. pp. 117–124. Springer (2022) 4

12. Grossi, G., Lanzarotti, R., Napoletano, P., Noceti, N., Odone, F.: Positive technology for elderly well-being: A review. Pattern Recognition Letters 137, 61–70 (2020) 3

13. Grossmann, T., Johnson, M.H.: The development of the social brain in human infancy. european Journal of neuroscience 25(4), 909–919 (2007) 3

14. Guo, H., Hu, Z., Liu, J.: Mgtr: End-to-end mutual gaze detection with transformer. In: Proceedings of the Asian Conference on Computer Vision. pp. 1590–1605 (2022) 1

15. Guo, Z., Chheang, V., Li, J., Barner, K.E., Bhat, A., Barmaki, R.L.: Social visual behavior analytics for autism therapy of children based on automated mutual gaze detection. In: Proceedings of the 8th ACM/IEEE International Conference on Connected Health: Applications, Systems and Engineering Technologies. pp. 11–21 (2023) 1

16. Javed, H., Lee, W., Park, C.H.: Toward an automated measure of social engagement for children with autism spectrum disorder—a personalized computational modeling approach. Frontiers in Robotics and AI 7, 43 (2020) 3

17. Joo, H., Simon, T., Cikara, M., Sheikh, Y.: Towards social artificial intelligence: Nonverbal social signal prediction in a triadic interaction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019) 6

18. Kenny, D.A., Ackerman, R.A., Kashy, D.A.: The Design and Analysis of Data from Dyads and Groups, p. 565–601. Cambridge Handbooks in Psychology, Cambridge University Press (2024) 2

19. Kenny, D.A., Kashy, D.A., Cook, W.L.: Dyadic data analysis. Guilford Publications (2020) 1, 2, 3

20. Kenny, D.A., La Voie, L.: The social relations model. In: Advances in experimental social psychology, vol. 18, pp. 141–182. Elsevier (1984) 1, 2, 3

21. Kim, J.T., Naik, A., Jayarathne, I., Ha, S., Chew, J.Y.: Modeling social interaction dynamics using temporal graph networks. In: 33rd IEEE International Conference on Robot and Human Interactive Communication (ROMAN). pp. 2272–2278. IEEE (2024) 3

22. Liu, X., Liu, W., Zhang, M., Chen, J., Gao, L., Yan, C., Mei, T.: Social relation recognition from videos via multi-scale spatial-temporal reasoning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3566– 3574 (2019) 3

23. Lombardi, M., Maiettini, E., De Tommaso, D., Wykowska, A., Natale, L.: Toward an attentive robotic architecture: learning-based mutual gaze estimation in human– robot interaction. Frontiers in Robotics and AI 9, 770165 (2022) 4

24. Mehrabian, A.: Silent messages, wadsworth. Belmont, California (1971) 1

25. Noceti, N., et. al: Predicting engagement of older people’s virtual teams from video call analysis. International Journal of Human–Computer Interaction 41(14), 9097– 9108 (2025) 3

26. Paneth, L., Jeitziner, L.T., Rack, O., Opwis, K., Zahn, C.: Zooming in: The role of nonverbal behavior in sensing the quality of collaborative group engagement. International Journal of Computer-Supported Collaborative Learning 19(2), 187– 229 (2024) 3

27. Penchala, S., Kontham, S.R., Bhattacharjee, P., Mahmoodi, N., Fonseca, D., Karami, S., Ghahremani, M., Perkins, A.D., Rahimi, S., Golilarz, N.A.: Learning in focus: Detecting behavioral and collaborative engagement using vision transformers. arXiv preprint arXiv:2508.15782 (2025) 3

28. Peng, C., Celiktutan, O.: Multi-task gaze communication understanding. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 5471–5479 (2025) 5, 6

29. Poria, S., Majumder, N., Mihalcea, R., Hovy, E.: Emotion recognition in conversation: Research challenges, datasets, and recent advances. IEEE access 7, 100943– 100953 (2019) 1

30. Rehg, J., Abowd, G., Rozga, A., Romero, M., Clements, M., Sclarof, S., Essa, I., Ousley, O., Li, Y., Kim, C., et al.: Decoding children’s social behavior. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3414–3421 (2013) 1

31. Senju, A., Johnson, M.H.: The eye contact efect: mechanisms and development. Trends in cognitive sciences 13(3), 127–134 (2009) 1

32. Setti, F., Russell, C., Bassetti, C., Cristani, M.: F-formation detection: Individuating free-standing conversational groups in images. PloS one 10(5), e0123783 (2015) 3

33. Shepherd, S.V.: Following gaze: gaze-following behavior as a window into social cognition. Frontiers in integrative neuroscience 4, 5 (2010) 3

34. Stokoe, E., Raymond, G., Whitehead, K.A.: Categories in social interaction: Unlocking the resources of conversation analysis and membership categorization for psychological science. Annual Review of Psychology 76 (2025) 1

35. Tomenotti, F.F., Noceti, N., Odone, F.: Head pose estimation with uncertainty and an application to dyadic interaction detection. Computer Vision and Image Understanding 243, 103999 (2024) 4

36. Vinciarelli, A., Pantic, M., Heylen, D., Pelachaud, C., Poggi, I., D’Errico, F., Schroeder, M.: Bridging the gap between social animal and unsocial machine: A survey of social signal processing. IEEE transactions on afective computing 3(1), 69–87 (2011) 3

37. Vinciarelli, A., Salamin, H., Pantic, M.: Social signal processing: Understanding social interactions through nonverbal behavior analysis. In: computer society con ference on computer vision and pattern recognition workshops. pp. 42–49. IEEE (2009) 3

38. Zheng, L., Li, J., Zhu, Z., Ji, W.: Lightnet: a lightweight head pose estimation model for online education and its application to engagement assessment. Journal of King Saud University Computer and Information Sciences 37(7), 166 (2025) 3