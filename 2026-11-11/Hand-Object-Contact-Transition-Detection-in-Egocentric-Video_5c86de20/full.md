# Hand-Object Contact Transition Detection in Egocentric Video

Huy Anh Nguyen, Chang Dong, Feras Dayoub, and Minh Hoai

Australian Institute for Machine Learning, Adelaide University, Adelaide, Australia

Abstract. This paper tackles the task of hand-object contact transition detection in egocentric video, which aims to localize the two temporal boundaries of a contact episode: the moment a hand first establishes contact with an object (touch) and the moment contact is released (release). We propose a unified framework that jointly models the interacting hand and object to accurately localize both transition types. The proposed framework augments hand-centric spatiotemporal representations with object-aware features learned directly from the video and introduces a transition-aware event spotting formulation tailored to contact transitions, without relying on external object or hand-object interaction detectors during inference. To support this task, we present ContactTransition, a dataset comprising 4,021 egocentric videos with 8,498 touch and 8,077 release annotations, together with object-of-interest supervision. Experiments demonstrate that the proposed method consistently outperforms strong two-stage and end-to-end event-spotting baselines on both transition types. Finally, we show that detected contact transitions provide efective temporal anchors for downstream egocentric hand-object reasoning. Code is available at https://github.com/bbvisual/oasis.

Keywords: Egocentric video · Hand-object interaction · Contact transition detection

## 1 Introduction

Hand-object manipulation is naturally organized into contact episodes that begin when the hand establishes contact with an object and end when the contact is released. These two transition moments define the temporal boundaries of every manipulation, enclosing the interval during which an object is grasped, transported, used, or repositioned. Detecting both transitions at frame-level precision provides a compact temporal representation of manipulation, enabling applications such as fine-grained action segmentation, grasp-duration analysis, skill assessment, and robot learning from demonstration. In this work, we study the task of detecting these two hand-object contact transitions in egocentric video. For brevity, we refer to the transition into contact as touch (contact onset) and the transition out of contact as release (contact ofset).

Detecting contact transitions at frame-level precision is challenging because the visual evidence is often confined to only a few frames. Subtle changes in finger articulation, object motion, and the relative hand-object configuration must be distinguished from visually similar frames immediately before or after contact. Egocentric videos further complicate the problem through rapid camera motion, motion blur, frequent occlusions, and large variations in manipulation style. These challenges require jointly reasoning about fine-grained spatial cues and short-term temporal dynamics to accurately localize contact transitions.

Hand-object contact transition detection belongs to the broader class of precise temporal event detection, which has been widely studied in computer vision for tasks such as facial action unit onset detection and fine-grained action spotting. However, these methods are typically designed for settings that differ substantially from egocentric hand-object interaction. Facial event detection assumes static or near-static cameras, while sports action spotting operates on videos in which the target event occupies a large portion of the frame and the subject remains continuously visible. In contrast, hand-object contact transitions occur within compact regions around the interacting hand and object and depend on subtle changes in their spatial relationship.

Recent work [37] demonstrated that hand-centric spatiotemporal representations can efectively detect touch moments. However, it models only contact onset, leaving the complementary transition—contact release—unaddressed. Extending touch detection to contact transition detection is not simply a matter of adding another event class. Touch and release are intrinsically coupled, jointly defining the temporal boundaries of the same contact episode, yet they are characterized by fundamentally diferent visual evidence. Touch is naturally indicated by an approaching hand and the evolving grasp immediately before contact, making hand-centric motion cues highly informative. In contrast, after release, the hand often remains close to the object or even stays nearly stationary, while the released object may move, fall, or remain behind. Consequently, accurately localizing contact transitions requires reasoning about the evolving relationship between the hand and the object rather than the hand alone. Motivated by these observations, we propose a unified framework that jointly models both the interacting hand and object, while reformulating precise event spotting to explicitly account for the coupled nature of touch and release transitions.

To support research on hand-object contact transition detection, we introduce ContactTransition, a new dataset containing frame-accurate annotations of both touch and release transitions across 4,021 egocentric videos, comprising 8,498 touch and 8,077 release moments. Beyond benchmarking contact transition detection, the dataset enables studies of complete contact episodes and higherlevel temporal reasoning over hand-object interactions. Experiments demonstrate that our framework consistently outperforms strong two-stage and endto-end event-spotting baselines on both touch and release detection. Our contributions are summarized as follows:

1. We introduce the task of hand-object contact transition detection, which aims to localize both touch (contact onset) and release (contact ofset) moments in egocentric video.

2. We propose a unified framework for contact transition detection that models the interacting hand and object, together with a transition-aware event spotting formulation tailored to the coupled nature of touch and release.

3. We introduce ContactTransition, a dataset containing 8,498 touch and 8,077 release annotations across 4,021 egocentric videos, enabling systematic evaluation of contact transition detection.

4. We demonstrate that contact transitions provide an efective temporal representation for downstream reasoning through Han $D y V Q A$ , a recent benchmark for contact-aware video question answering.

## 2 Related Work

## 2.1 Hand-Object Interaction Understanding

Understanding hands and their interactions with objects is fundamental to egocentric vision. Beyond a large body of work on detecting and tracking the hands themselves (e.g., [27, 36]), a long-standing goal is to understand how hands interact with objects [18, 38]. A large body of image-based work studies contact as a per-frame attribute: 100DOH [41], ContactHands [35], and Hands23 [9] detect hands and classify their contact state from single frames, while BodyHands [34] infers hand–body association from spatial overlap. Beyond binary contact, HOT [8] localizes contact regions as dense heatmaps, and thermal- or pressure-based approaches [6,22,23] recover fine-grained contact and force information from specialized supervision; contact maps have further been used to infer object functionality and afordances [21]. Video-based methods extend this analysis over time, detecting, segmenting, and tracking hands and interacted objects [12, 13, 19, 33], supported by richly annotated manipulation datasets [4,16,29,30]. Yagi et al. [57] predict a binary contact state for each video frame from given hand-object tracks, but model contact as a state rather than localizing its transitions. Temporal structure, however, is typically addressed only at coarse granularity: large-scale egocentric benchmarks [11, 24, 25] localize actions or object state changes at second-level tolerances, with Ego4D’s pointof-no-return keyframe [24] being the closest to our setting, identifying a single state-change frame per clip rather than the paired boundaries of a contact episode. HiCE [37] recently showed that contact onset can be localized with near frame-level precision, but the complementary transition out of contact remains unaddressed. Our work closes this gap by detecting both transitions, turning per-frame contact attributes into temporally delimited contact episodes.

Beyond perception, recent benchmarks probe whether video-language models genuinely understand hand-object interaction. HanDyVQA [46] and related egocentric benchmarks [54] evaluate fine-grained reasoning over manipulation dynamics, spatial relations, object parts, and state changes. Their findings show that current video and video-language models, including InternVideo2 [49], LaViLa [61], and Qwen2.5-VL [3], remain limited in contact-sensitive interaction understanding, motivating representations that better capture when and where hand-object evidence occurs.

## 2.2 Precise Temporal Event Spotting

Action spotting [14, 20] localizes the precise frame at which an event occurs, in contrast to temporal action localization [7, 42, 56, 58, 59], which predicts multiframe intervals evaluated by temporal IoU. While interval-based methods can be adapted to short events, they are not well suited to instantaneous transitions that require frame-level precision [55]. Existing spotting methods generally follow two paradigms: two-stage approaches use pre-extracted features with temporal localization heads [1, 15, 17, 43, 51, 58, 62], while end-to-end methods learn visual features jointly with the spotting objective [26,40,47,52,53]. Recent methods improve temporal precision through temporal shift modules, recurrent or encoder–decoder reasoning, and displacement refinement [10, 42, 44, 45, 52]; HiCE [37] adapts this framework to hand contact by attending to hand regions.

However, most spotting formulations assume that events of diferent classes are temporally isolated, using mutually exclusive frame classification and a classshared displacement for refinement. Contact transitions violate this assumption: touch and release bound the same episode, can occur close together, and may co-occur in bimanual manipulation. Our transition-aware formulation therefore scores each transition independently against a shared background reference and refines each transition with class-specific displacements.

## 2.3 Object Grounding for Event Detection

Several works incorporate object evidence into temporal localization. AOE-Net [48] models actor–object–environment interactions for temporal proposal generation, and UGLF [47] grounds scene entities with vision-language detectors [28,60] to improve precise spotting. These approaches rely on external detectors at inference time, require a predefined object vocabulary, and localize objects unconditionally, irrespective of whether an object participates in the interaction. Detector-based pipelines for interacted objects [12,19,33] face similar constraints, and are least reliable precisely at contact moments, where the hand occludes the object it manipulates. In contrast, our framework learns to localize the object of interest from hand-conditioned features, using object annotations only as training-time supervision. At inference time, it does not rely on an object or hand-object interaction detector; the object regions are predicted within the contact-transition model and are therefore tied to the ongoing hand interaction.

## 3 ContactTransition Dataset

Existing frame-precise contact annotations, such as TouchMoment [37], focus on contact onset: the moment a hand first establishes contact with an object. While this enables precise touch detection, it does not capture the full temporal extent of a manipulation episode. To support hand-object contact transition detection, we introduce ContactTransition, an egocentric video benchmark with frame-level annotations of both touch and release transitions. The benchmark follows the video sources, clip selection criteria, and touch annotation protocol used in prior frame-precise contact detection work [37], while introducing release annotations that capture the endpoint of each contact episode.

<table><tr><td> $\#$ </td><td>Videos</td><td> $\#$  Touch</td><td> $\#$  Release</td><td> $\#$  Events</td></tr><tr><td>Train</td><td>3,234</td><td>6,837</td><td>6,488</td><td>13,325</td></tr><tr><td>Val</td><td>138</td><td>324</td><td>323</td><td>647</td></tr><tr><td>Test</td><td>649</td><td>1,337</td><td>1,266</td><td>2,603</td></tr><tr><td>Total</td><td> $^ { 4 , 0 2 1 }$ </td><td>8,498</td><td>8,077</td><td>16,575</td></tr></table>

Table 1: ContactTransition statistics. Touch and release counts are comparable overall. Release counts are slightly lower because some contact episodes continue beyond the clip boundary, and because two-hand interactions in TACO may contain separate contact onsets that terminate at a shared release moment.

## 3.1 Release Annotation

We define a release event as the first frame of sustained separation between the hand and the object involved in the interaction. Unlike touch, which is often indicated by the hand approaching and establishing contact with an object, release can be visually ambiguous. The hand may remain close to the object after letting go, the contact interface may be occluded by the hand, and short interruptions during re-grasping should not be treated as separate contact episodes. Therefore, annotators were instructed to use visible hand-object separation, object motion, and the surrounding temporal context to identify the earliest frame at which contact is no longer present.

To ensure temporal consistency, annotators considered the surrounding context when deciding whether a visible separation should be marked as a release. Brief separations caused by rapid hand motion, such as repeated tapping or quick adjustment of the same object, are treated as part of the same contact episode rather than as separate release–touch pairs. A release is marked only when the hand clearly disengages from the object, or when the subsequent contact corresponds to a distinct interaction. Ambiguous cases are excluded when the release moment cannot be localized reliably due to occlusion, motion blur, or insuficient visual evidence.

## 3.2 Expanded Manual Coverage

ContactTransition uses manually verified annotations for all validation and test clips, including both HOI4D-derived clips and TACO segments. For the training split, prior frame-precise touch annotations provide reliable contact-onset labels for a large portion of the data [37]. However, we found that release timing is less amenable to the same automatic annotation strategy, since release events are more afected by occlusion, rapid re-contact, and subtle object motion. We therefore manually verify 35% of the HOI4D training clips, prioritizing videos containing challenging release events. Across these and the validation/test clips, we annotated 5,126 release events and verified or corrected 1,668 touch events, excluding boundaries that could not be reliably localized.

![](images/ad0d556129144d3ee7254341be2dc9050951544ddc76ad9c80e9bde15c9fb8a3.jpg)  
Fig. 1: Overview of OASIS. Starting from the hand-enhanced feature map produced by HiCE [37], OASIS predicts an object-of-interest heatmap and pools object-aware features from the predicted regions. A presence gate controls when object evidence should be used, and the resulting object-aware representation is fused into the temporal spotting stream. The final head predicts touch and release transitions with event scores and class-specific displacement refinement.

## 4 Proposed Methodology

In this section, we first formulate contact transition detection as an eventspotting task, where the goal is to localize the two boundaries of hand-object contact: touch onset and touch release. We then review the hand-centric representation used by HiCE [37], which provides the spatially localized hand context on which our model builds. Finally, we present OASIS, an object-aware spotting framework that augments hand-centric representations with candidate-object evidence and predicts touch and release through a transition-aware formulation.

## 4.1 Problem Formulation

Given an egocentric video clip of L frames, $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { L }$ , the goal is to detect a sparse set of contact-transition events $\mathcal { E } = \{ ( t _ { k } , c _ { k } ) \} _ { k = 1 } ^ { K }$ , where $t _ { k } \in \{ 1 , \ldots , L \}$ denotes the event frame and $c _ { k } \in \mathcal { C } { = } \{  \mathrm { t o u c l }$ h, release} denotes the transition type.

Following the precise event-spotting protocol [26, 37], a prediction is considered correct if it matches a ground-truth event of the same class within a temporal tolerance of δ frames. Since contact transitions are brief and visually subtle, we evaluate under strict tolerances, following the near-frame-level setting used in prior contact spotting work [37].

Unlike touch-only detection, contact transition detection involves two coupled event types that define the same contact episode. However, they should not be forced into a mutually exclusive label space. In egocentric bimanual interaction, diferent hands may generate diferent transition events at the same frame, and these events may involve diferent objects. We therefore formulate touch and release as class-wise contact-transition events while using a shared object-aware representation to provide interaction context for both predictions.

## 4.2 Preliminary: HiCE

HiCE [37] is a hand-centric framework for precise contact onset detection. Given a video clip and hand boxes obtained from ground-truth annotations or an ofthe-shelf hand detector such as Hands23, HiCE extracts two visual streams: a global frame feature that captures the surrounding scene, and localized handpatch features that capture fine-grained hand appearance. The hand features are used to enhance the global frame representation through hand-to-frame attention, producing a hand-enhanced feature map $\mathcal { F } _ { \mathrm { e n h a n c e d } } ~ \in ~ \mathbb { R } ^ { H \times W \times C }$ . This representation preserves spatial information from the frame while emphasizing regions relevant to the observed hands.

HiCE then pools the enhanced feature map into a frame-level representation and feeds the resulting temporal sequence into a spotting head. The head predicts contact onset scores and temporal displacement ofsets, which are used to refine event localization. This design is efective for touch detection because contact onset is often associated with visible hand motion and local hand-object configuration changes. However, it does not explicitly model the contacted object and is designed for a single transition type. In the following sections, we use $\mathcal { F } _ { \mathrm { e n h a n c e d } }$ as the starting representation and introduce object-aware features and transition-aware prediction heads for detecting both touch and release.

## 4.3 Object-of-Interest Branch

The hand-enhanced feature map $\mathcal { F } _ { \mathrm { e n h a n c e d } }$ captures fine-grained hand appearance and local hand context, but it does not explicitly identify the object involved in the interaction. This object information is important for contact transition detection. For touch, the contacted object helps disambiguate reaching from incidental hand motion; for release, it provides a spatial anchor when the hand remains close to the object or the change in hand pose is subtle. We therefore introduce an object-of-interest branch that localizes candidate interaction objects from $\mathcal { F } _ { \mathrm { e n h a n c e d } }$ and injects object-aware evidence into the temporal spotting representation.

The branch predicts a class-agnostic object heatmap rather than object categories. This design reflects the goal of contact transition detection: the model only needs to know where the interaction-relevant object is, not what semantic class it belongs to. Since the heatmap is predicted from hand-enhanced features, object localization is conditioned on the observed hand context. The branch therefore learns to highlight objects likely to participate in the ongoing hand interaction, rather than performing generic object detection over the full scene.

We supervise the heatmap using object-of-interest annotations derived from available object annotations and contact-object annotations. For HOI4D, we use the provided object annotations; for TACO, we obtain per-frame contactedobject annotations using HOI-DETR [12]. Candidate-object regions are projected onto the hand-centered feature grid to form a soft coverage target $G \in$ $[ 0 , 1 ] ^ { H \times W }$ , where each cell indicates the degree of overlap with an interactionrelevant object. Given $\mathcal { F } _ { \mathrm { e n h a n c e d } } ~ \in ~ \mathbb { R } ^ { H \times W \times C }$ , a lightweight head predicts a heatmap $S = \phi _ { \mathrm { o b j e c t } } ( \mathcal { F } _ { \mathrm { e n h a n c e d } } ) \ \in \ \mathbb { R } ^ { H \times W }$ and is trained with a foregroundweighted binary cross-entropy:

$$
{ \mathcal { L } } _ { \mathrm { h e a t m a p } } = { \frac { 1 } { H W } } \sum _ { j } w _ { j } \mathrm { B C E } ( S _ { j } , G _ { j } ) .\tag{1}
$$

We set $w _ { j } = 1 + ( w _ { \mathrm { f g } } - 1 ) G _ { j }$ , with $w _ { \mathrm { f g } } = 1 0$ , so empty cells receive weight 1 and fully covered cells receive weight 10. This compensates for the small foreground region relative to the background. To convert the heatmap into an object feature, we normalize the predicted object probabilities over the spatial grid and use them to pool from $\mathcal { F } _ { \mathrm { e n h a n c e d } }$ . This produces an object-aware feature $f _ { \mathrm { o b j e c t } }$ that summarizes the regions predicted to contain candidate interaction objects. We use normalized coverage weights rather than a softmax so that the pooling can cover the spatial extent of an object and can naturally support multiple candidate regions.

Because not every hand-centered crop contains a visible candidate object, we further predict an object-presence score $q$ from the frame representation using a two-layer MLP. The presence score gates the object feature before fusion, reducing the efect of noisy object readouts on frames where no reliable object evidence is visible. The gated object feature is injected into the global frame feature through a residual connection:

$$
f = f _ { \mathrm { g l o b a l } } + \gamma W _ { o } ( q f _ { \mathrm { o b j e c t } } ) ,\tag{2}
$$

where $W _ { o }$ is a learned projection and $\gamma$ is initialized to zero. The resulting feature sequence is passed to the temporal spotting module, allowing touch and release to be predicted from a shared hand- and object-aware representation.

## 4.4 Transition-Aware Event Spotting

Contact transition detection difers from standard action spotting because touch and release are coupled but not mutually exclusive. A softmax classifier forces the two transition types to compete for probability mass, which is undesirable when events occur close together or on the same frame under bimanual manipulation. We therefore predict each transition type independently against a shared background reference. Given logits $( z _ { l } ^ { \mathrm { b g } } , z _ { l } ^ { \mathrm { t o u c h } } , z _ { l } ^ { \mathrm { r e l e a s e } } )$ at frame $l ,$ we define class-wise margins

$$
m _ { l } ^ { c } = z _ { l } ^ { c } - z _ { l } ^ { \mathrm { b g } } , \qquad \hat { p } _ { l } ^ { c } = \sigma ( m _ { l } ^ { c } ) , \qquad c \in \{ \mathrm { t o u c h , r e l e a s e } \} .\tag{3}
$$

Each transition type is trained with a focal binary cross-entropy against its own soft temporal target:

$$
\mathcal { L } _ { c } = \frac { 1 } { L K } \sum _ { l , c } \alpha _ { l } ^ { c } ( 1 - p _ { t , l } ^ { c } ) ^ { \gamma _ { f } } \mathrm { B C E } ( m _ { l } ^ { c } , y _ { l } ^ { c } ) ,\tag{4}
$$

where $K = 2$ and $y _ { l } ^ { c }$ is the Gaussian target for class c. This formulation allows touch and release to be scored independently while sharing the same underlying representation.

We also make temporal displacement refinement class-aware. Prior spotting models regress a single displacement per frame, shared across all event classes. This is insuficient when touch and release are close in time, since one shared ofset may refine one class toward the correct transition while moving the other away from it. Instead, the displacement head predicts $\hat { y } ^ { d } \in \mathbb { R } ^ { L \times K }$ , with one ofset per transition type. Each ofset is supervised by the signed distance to the nearest event of the same class within radius r:

$$
y _ { l } ^ { d , c } = \left\{ l - t _ { c } ^ { * } ( l ) , \quad | l - t _ { c } ^ { * } ( l ) | \leq r , \right.\tag{5}
$$

where $t _ { c } ^ { * } ( l )$ is the nearest class-c event to frame l. The displacement loss is

$$
\mathcal { L } _ { d } = \frac { 1 } { L K } \sum _ { l , c } \big ( \hat { y } _ { l } ^ { d , c } - y _ { l } ^ { d , c } \big ) ^ { 2 } .\tag{6}
$$

At inference, each class score sequence is refined using its own displacement predictions, following the same temporal refinement procedure in [37].

## 4.5 Training Objective

The full model is trained with event classification, class-aware displacement regression, auxiliary grasp supervision inherited from HiCE, and object-of-interest supervision:

$$
\mathscr { L } = \mathscr { L } _ { c } + \mathscr { L } _ { d } + \lambda _ { g } \mathscr { L } _ { g } + \lambda _ { \mathrm { o b j e c t } } \left( \mathscr { L } _ { \mathrm { h e a t m a p } } + \lambda _ { p } \mathscr { L } _ { \mathrm { p r e s e n c e } } \right) .\tag{7}
$$

Here, $\mathcal { L } _ { g }$ regularizes the hand branch using grasp pseudo-labels, while L<sub>heatmap</sub> and $\mathcal { L } _ { \mathrm { p r e s e n c e } }$ supervise candidate object localization and object presence estimation. The model is trained end-to-end; so hand-enhanced features, object localization, temporal reasoning, and touch-release prediction are optimized jointly.

## 5 Experimental Analysis

This section describes the experimental setup on ContactTransition, including implementation details, evaluation metrics, and the baseline methods used for comparison. We then present quantitative results followed by an ablation study to assess the contribution of each component in our approach.

## 5.1 Implementation Details

We follow the training setup of HiCE [37]. Models are trained on clips of $L = 4 0$ frames for 50 epochs with AdamW [31], using a batch size of 6, 5000 clips per epoch, an initial learning rate of $4 \times 1 0 ^ { - 4 }$ , a three-epoch warm-up, and cosine annealing. We use a RegNet-Y 800MF backbone [39] initialized from ImageNetpretrained weights in timm [50]. Hand patches are enlarged by a factor of 1.2 and resized to 224×224.

<table><tr><td></td><td></td><td colspan="3">Touch</td><td colspan="3">Release</td></tr><tr><td></td><td>mAP</td><td>AP@0</td><td>AP@1</td><td>AP@2</td><td>AP@0</td><td>AP@1</td><td>AP@2</td></tr><tr><td>(a) Two-stage methods</td><td colspan="7"></td></tr><tr><td>TSP [1] + MS-TCN [17]</td><td>41.14</td><td>15.34</td><td>31.77</td><td>46.44</td><td>27.19</td><td>57.82</td><td>68.26</td></tr><tr><td>TSP + ASFormer [58]</td><td>39.39</td><td>17.16</td><td>30.65</td><td>40.40</td><td>26.32</td><td>54.41</td><td>67.40</td></tr><tr><td>V-JEPA [2, 32] + MS-TCN</td><td>43.93</td><td>17.86</td><td>41.20</td><td>59.30</td><td>22.11</td><td>53.95</td><td>69.16</td></tr><tr><td>(b) End-to-end methods</td><td colspan="7"></td></tr><tr><td>E2ESpot [26]</td><td>24.32</td><td>2.88</td><td>20.10</td><td>37.14</td><td>6.06</td><td>30.56</td><td>49.16</td></tr><tr><td>T-DEED [52]</td><td>22.83</td><td>12.26</td><td>18.84</td><td>23.32</td><td>18.69</td><td>29.45</td><td>34.39</td></tr><tr><td>UGLF [47]</td><td>25.3</td><td>11.78</td><td>17.11</td><td>20.91</td><td>21.65</td><td>36.84</td><td>43.52</td></tr><tr><td>ASTRM [40]</td><td>21.48</td><td>13.56</td><td>19.09</td><td>22.22</td><td>18.51</td><td>25.96</td><td>29.54</td></tr><tr><td>AdaSpot [53]</td><td>36.03</td><td>16.15</td><td>27.67</td><td>35.52</td><td>27.81</td><td>50.36</td><td>58.66</td></tr><tr><td>HiCE [37]</td><td>43.67</td><td>16.48</td><td>39.13</td><td>52.97</td><td>23.59</td><td>57.45</td><td>72.41</td></tr><tr><td>OASIS (ours)</td><td>49.30</td><td>16.88</td><td>44.86</td><td>61.69</td><td>29.42</td><td>64.38</td><td>78.54</td></tr></table>

(a) Without non-maximum suppression (NMS).

<table><tr><td></td><td></td><td colspan="3">Touch</td><td colspan="3">Release</td></tr><tr><td></td><td>mAP</td><td>AP@0</td><td>AP@1</td><td>AP@2</td><td>AP@0</td><td>AP@1</td><td>AP@2</td></tr><tr><td colspan="8">(a) Two-stage methods</td></tr><tr><td>TSP [1] + MS-TCN [17]</td><td>49.19</td><td>9.96</td><td>41.82</td><td>66.93</td><td>20.77</td><td>69.29</td><td>86.37</td></tr><tr><td>TSP + ASFormer [58]</td><td>49.28</td><td>12.20</td><td>45.85</td><td>68.49</td><td>20.36</td><td>65.06</td><td>83.72</td></tr><tr><td>V-JEPA [2, 32] + MS-TCN</td><td>47.94</td><td>14.70</td><td>46.40</td><td>69.42</td><td>20.88</td><td>59.16</td><td>77.04</td></tr><tr><td colspan="8">(b) End-to-end methods</td></tr><tr><td>E2ESpot [26]</td><td>27.84</td><td>3.68</td><td>27.95</td><td>37.62</td><td>7.21</td><td>39.02</td><td>51.58</td></tr><tr><td>T-DEED [52]</td><td>35.67</td><td>4.94</td><td>35.19</td><td>48.66</td><td>9.99</td><td>49.90</td><td>65.31</td></tr><tr><td>UGLF [47]</td><td>37.44</td><td>4.80</td><td>35.85</td><td>47.09</td><td>10.75</td><td>55.06</td><td>71.09</td></tr><tr><td>ASTRM [40]</td><td>38.92</td><td>6.61</td><td>36.11</td><td>60.03</td><td>11.68</td><td>47.83</td><td>71.23</td></tr><tr><td>AdaSpot [53]</td><td>46.09</td><td>11.23</td><td>42.12</td><td>56.81</td><td>21.56</td><td>65.77</td><td>79.05</td></tr><tr><td>HiCE [37]</td><td>47.58</td><td>6.48</td><td>46.11</td><td>71.00</td><td>11.32</td><td>64.79</td><td>85.76</td></tr><tr><td>OASIS (ours)</td><td>51.19</td><td>8.43</td><td>51.41</td><td>71.43</td><td>17.97</td><td>70.90</td><td>87.01</td></tr></table>

(b) With NMS/Soft-NMS.  
Table 2: Quantitative comparison on ContactTransition. Methods are grouped into two-stage approaches using pre-extracted features and end-to-end models. We report overall mAP, together with AP at frame tolerances $\delta \in \{ 0 , 1 , 2 \}$ for touch and release detection. Bold and underline denote the best and second-best results.

For event spotting, we use focal loss with $\alpha = 0 . 9$ and $\gamma _ { f } = 2$ , displacement supervision within $r \ = \ 4$ frames, and a grasp loss weight of $\lambda _ { g } ~ = ~ 0 . 2$ . The object-of-interest branch consists of a lightweight heatmap head and a presence gate, trained with loss weight $\lambda _ { \mathrm { o b j } } = 0 . 1 5$ and presence weight $\lambda _ { p } = 0 . 7 5$ . The presence threshold is set to $\tau = 0 . 1$ . Full architectural and optimization details are provided in the supplementary material.

## 5.2 Evaluation Metrics

We follow the precise event spotting protocol and evaluate touch and release as independent classes: for each class, a prediction is counted as correct if it falls within δ frames of an unmatched ground-truth event of that class, and we report per-class $\mathrm { A P }$ at tolerance thresholds $\delta \in \{ 0 , 1 , 2 \}$ together with their mean (mAP). Frames may carry events of both classes, which per-class matching handles naturally. Unlike sports-based spotting benchmarks, where tolerances of $\delta \leq 4$ are common, contact transitions in egocentric video occur in close temporal proximity, and release moments are particularly ambiguous at coarse tolerances due to gradual withdrawal; the low tolerance values emphasize near frame-level precision for both transition types.

## 5.3 Baseline Methods

We compare against baselines under two paradigms on ContactTransition, extending all methods from single-class touch spotting to the two-class touch– release setting by widening their classification heads accordingly. Two-stage approaches first extract features and then train a separate temporal model: we evaluate TSP [1] paired with MS-TCN [17] and ASFormer [58], as well as V-JEPA [2] paired with MS-TCN. TSP features are obtained by fine-tuning on our dataset following the original pretraining protocol, while V-JEPA features are extracted from the publicly available checkpoint. For end-to-end methods, we evaluate E2E-Spot [26], T-DEED [52], UGLF [47], ASTRM [40], AdaSpot [53], and HiCE [37], the strongest published method on the touch-only benchmark, extended to two classes with its original softmax formulation. Other action-spotting models such as ASTRA, COMEDIAN, and Soares [15, 43, 51] require pre-extracted features from an ensemble of five networks [62] trained on SoccerNet-v2 broadcast footage; these features are unavailable for egocentric video and retraining the ensemble is impractical, so a faithful reproduction of these methods on our benchmark is not possible.

For fair comparison, we apply Non-Maximum Suppression and Soft Non-Maximum Suppression (SNMS) [5], as used in T-DEED, to all methods with identical settings. Since suppression interacts diferently with dense and sparse prediction styles, we additionally report all results without NMS/SNMS and discuss its efect on frame-exact precision in Sec. 5.4.

## 5.4 Experiment Results

Tab. 2 presents contact transition detection performance on ContactTransition. Following the evaluation protocol of HiCE [37], we report results both without and with NMS/SNMS to separate raw temporal precision from post-processed detection quality.

Baseline comparison. Among two-stage methods, V-JEPA + MS-TCN achieves the highest mAP without NMS (43.93), while TSP + ASFormer leads with NMS (49.28), narrowly ahead of TSP + MS-TCN (49.19). Two-stage methods are strong at frame-exact touch localization, with V-JEPA + MS-TCN attaining the best touch AP@δ=0 both without (17.86) and with NMS (14.70), reflecting the benefit of features trained or pretrained at clip level before temporal decoding. Among end-to-end baselines, HiCE is the strongest overall (43.67 without NMS, 47.58 with NMS), followed by AdaSpot (36.03 and 46.09). Most end-toend methods gain substantially from suppression, with T-DEED improving from 22.83 to 35.67 and ASTRM from 21.48 to 38.92, indicating dense, redundant raw detections that NMS efectively prunes. Suppression consistently degrades frame-exact precision, however, and the efect is most severe on release: HiCE drops from 23.59 to 11.32 release AP@δ=0, and AdaSpot, the best baseline at this operating point, falls from 27.81 to 21.56. This asymmetry is consistent with the gradual nature of withdrawal, which produces plateau-shaped score profiles around release moments that suppression merges or shifts.

OASIS (Ours). Our method achieves the best overall mAP both without NMS (49.30) and with NMS (51.19), surpassing the strongest baseline in each setting by 5.37 and 1.91 mAP, respectively. Compared with HiCE, OASIS improves by 5.63 mAP without NMS and 3.61 mAP with NMS. The gains over HiCE concentrate on release detection, where the object-of-interest branch contributes +5.83 AP@δ=0 (29.42 vs. 23.59), +6.93 AP@δ=1, and +6.13 AP@δ=2 without NMS, making OASIS the best method on release at every tolerance before suppression. This supports the motivation of Sec. 4.3: release is often signaled by subtle finger movements with little change in global hand pose, so an explicit object anchor provides evidence that hand-centric features lack. Touch localization also improves at relaxed tolerances, reaching 44.86 AP@δ=1 and 61.69 AP@δ=2 without NMS, the best among all methods, while frame-exact touch AP@δ=0 (16.88) remains competitive with the two-stage leaders. Overall, OASIS provides the largest gains at δ ≥ 1 and on the release transition, precisely where hand cues are weakest and object grounding matters most.

## 5.5 Ablation Study

Tab. 3 analyzes the contribution of each component. Removing the object branch causes the largest drop, reducing mAP with NMS from 51.19 to 40.89. This confirms that object grounding is central to contact transition detection, especially for release, where hand appearance alone is often ambiguous. The presence gate provides a smaller but consistent gain, improving mAP by 1.08 and increasing frame-exact release AP from 27.69 to 29.42, suggesting that suppressing unreliable object features is useful when no candidate object is visible in the hand-centered crop.

<table><tr><td></td><td colspan="2">mAP</td><td colspan="2">Touch</td><td colspan="2">Release</td></tr><tr><td></td><td>w/o NMS</td><td>w/NMS</td><td>AP@0</td><td>AP@1</td><td>AP@0</td><td>AP@1</td></tr><tr><td>OASIS (full)</td><td>49.30</td><td>51.19</td><td>16.88</td><td>44.86</td><td>29.42</td><td>64.38</td></tr><tr><td>w/o presence gate</td><td>48.14</td><td>50.11</td><td>16.80</td><td>44.69</td><td>27.69</td><td>63.59</td></tr><tr><td>softmax pooling (vs. CWP)</td><td>47.76</td><td>50.37</td><td>17.78</td><td>42.77</td><td>29.41</td><td>64.76</td></tr><tr><td>shared displacement (vs. per-class)</td><td>44.61</td><td>47.52</td><td>16.45</td><td>37.81</td><td>25.87</td><td>59.78</td></tr><tr><td>w/o object branch</td><td>37.43</td><td>40.89</td><td>13.68</td><td>29.08</td><td>21.71</td><td>52.21</td></tr><tr><td>HiCE (softmax formulation) [37]</td><td>43.67</td><td>47.58</td><td>16.48</td><td>39.13</td><td>23.59</td><td>57.45</td></tr></table>

Table 3: Ablation study on ContactTransition. Each row removes or replaces one component of the full model. We report mean AP over both transitions and all tolerances $\delta \in \{ 0 , 1 , 2 \}$ , without and with NMS, together with per-transition AP at $\delta \in \{ 0 , 1 \}$ without NMS. AP@2 columns are omitted for space and follow the same trends.

![](images/a3fcb0f5dc7670ae84e78acdb4106c8f356f3218370e9477e62c0b48f54d0a3d.jpg)  
Fig. 2: Predicted object-of-interest heatmaps. Top: input frames around a contact transition, with the predicted transition frame highlighted in red. Bottom: heatmaps predicted by the object-of-interest branch of OASIS. Although object annotations are used only during training, the model learns to highlight the object involved in the hand interaction at inference time, providing object-aware evidence for precise touch–release localization.

The pooling strategy also matters. Replacing coverage-weighted pooling (CWP) with softmax pooling reduces mAP, particularly before NMS, indicating that preserving spatial support over the predicted object region is preferable to concentrating on a single high-response cell. Class-aware displacement refinement is another important component: using a shared displacement reduces mAP by 3.67 and substantially lowers touch AP@1 from 44.86 to 37.81. This supports our argument that touch and release require separate temporal refinement because they can occur close together or even overlap in bimanual manipulation.

Finally, the original HiCE softmax formulation trails the full model by 3.61 mAP. Interestingly, removing the object branch from our transition-aware formulation performs worse than HiCE, showing that independent transition scoring alone is insuficient. Without object grounding, removing class competition can also remove useful suppression between ambiguous events. The full model benefits from the combination of independent transition scoring and object-aware evidence, indicating that the proposed components are complementary rather than merely additive.

<table><tr><td>Method</td><td>Action</td><td>Location</td><td>Objects (AP)</td><td>Parts</td><td>Process</td><td>State</td><td>Avg</td></tr><tr><td>Qwen2.5-VL</td><td>58.66</td><td>53.12</td><td>72.28</td><td>50.36</td><td>54.88</td><td>58.75</td><td>58.01</td></tr><tr><td>Qwen2.5-VL + contact context</td><td>58.84</td><td>53.94</td><td>72.63</td><td>51.59</td><td>54.06</td><td>59.89</td><td>58.49</td></tr><tr><td>∆ (improvement)</td><td>+0.18</td><td>+0.82</td><td>+0.35</td><td>+1.23</td><td>-0.82</td><td>+1.14</td><td>+0.48</td></tr></table>

Table 4: Contact transitions as temporal context for HanDyVQA. Both settings use Qwen2.5-VL with uniform frame sampling; the second row additionally provides predicted touch and release scores as temporal context.

## 6 Application: Contact Moments for Reasoning

Contact transitions can also serve as lightweight temporal cues for egocentric video reasoning. We evaluate this on HanDyVQA [46], a benchmark for finegrained hand-object interaction understanding covering actions, object locations, manipulated objects, parts, processes, and state changes. We use Qwen2.5-VL [3] with uniform frame sampling as the base model, and compare it with a contactaware variant that additionally provides predicted touch and release scores as temporal context together with the question.

As shown in Tab. 4, adding contact context improves the average score from 58.01 to 58.49 without modifying the VLM or changing the sampled frames. The largest gains appear on part-level and state reasoning, suggesting that contacttransition scores help the model identify when fine-grained hand-object evidence is likely to be relevant. The process category decreases slightly, indicating that transition timing alone does not capture all procedural cues.

## 7 Conclusions

We introduced hand–object contact transition detection, the task of precisely localizing the touch and release boundaries of contact episodes in egocentric video. To address this task, we proposed OASIS, which combines hand- and object-aware representations with transition-aware spotting to jointly detect both boundaries, without requiring external object or hand–object interaction detectors at inference.

We also introduced ContactTransition, a benchmark providing frame-level touch and release annotations together with object-of-interest supervision. OA-SIS consistently outperforms strong two-stage and end-to-end spotting baselines, while our downstream experiments demonstrate that detected contact transitions provide efective temporal anchors for egocentric hand–object reasoning. Extending the framework to per-hand transition attribution, leveraging annotations that identify the hand involved in each contact episode, is a promising direction for future work. Overall, our results establish contact transitions as a compact and useful representation for fine-grained manipulation understanding.

Acknowledgements: This work was funded by the Australian Institute for Machine Learning (Adelaide University) and the Centre for Augmented Reasoning, an initiative by the Department of Education, Australian Government.

## References

1. Alwassel, H., Giancola, S., Ghanem, B.: Tsp: Temporally-sensitive pretraining of video encoders for localization tasks. In: Proceedings of the International Conference on Computer Vision Workshops (2021)

2. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Robert Hogan, F., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., Ballas, N.: V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025)

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923 (2025)

4. Banerjee, P., Shkodrani, S., Moulon, P., Hampali, S., Han, S., Zhang, F., Zhang, L., Fountain, J., Miller, E., Basol, S., Newcombe, R., Wang, R., Engel, J.J., Hodan, T.: Hot3d: Hand and object tracking in 3d from egocentric multi-view videos. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2025)

5. Bodla, N., Singh, B., Chellappa, R., Davis, L.S.: Soft-nms — improving object detection with one line of code. In: Proceedings of the International Conference on Computer Vision (2017)

6. Brahmbhatt, S., Ham, C., Kemp, C.C., Hays, J.: ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging . In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019)

7. Buch, S., Escorcia, V., Shen, C., Ghanem, B., Carlos Niebles, J.: SST: Single-stream temporal action proposals. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017)

8. Chen, Y., Dwivedi, S.K., Black, M.J., Tzionas, D.: Detecting human-object contact in images. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2023)

9. Cheng, T., Shan, D., Hassen, A., Higgins, R.E.L., Fouhey, D.F.: Towards a richer 2d understanding of hands at scale. In: Advances in Neural Information Processing Systems (2023)

10. Chung, J., Gulcehre, C., Cho, K., Bengio, Y.: Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. arXiv preprint arXiv:1412.3555 (2014)

11. Damen, D., Doughty, H., Farinella, G.M., Furnari, A., Kazakos, E., Ma, J., Moltisanti, D., Munro, J., Perrett, T., Price, W., Wray, M.: Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision 130(1), 33–55 (2022)

12. Darkhalil, A., Damen, D., Fouhey, D.: Improving and evaluating hand-object interaction detection. arXiv preprint arXiv:2606.17384 (2026)

13. Darkhalil, A., Shan, D., Zhu, B., Ma, J., Kar, A., Higgins, R., Fidler, S., Fouhey, D., Damen, D.: Epic-kitchens visor benchmark: Video segmentations and object relations. In: Advances in Neural Information Processing Systems Track on Datasets and Benchmarks (2022)

14. Deliège, A., Cioppa, A., Giancola, S., Seikavandi, M.J., Dueholm, J.V., Nasrollahi, K., Ghanem, B., Moeslund, T.B., Van Droogenbroeck, M.: Soccernet-v2: A

dataset and benchmarks for holistic understanding of broadcast soccer videos. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (2021)

15. Denize, J., Liashuha, M., Rabarisoa, J., Orcesi, A., Hérault, R.: Comedian: Selfsupervised learning and knowledge distillation for action spotting using transformers. In: Proceedings of the IEEE Winter Conference on Applications of Computer Vision Workshops (2024)

16. Fan, Z., Taheri, O., Tzionas, D., Kocabas, M., Kaufmann, M., Black, M.J., Hilliges, O.: ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2023)

17. Farha, Y.A., Gall, J.: Ms-tcn: Multi-stage temporal convolutional network for action segmentation. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2019)

18. Fathi, A., Ren, X., Rehg, J.M.: Learning to recognize objects in egocentric activities. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2011)

19. Fu, Q., Liu, X., Kitani, K.: Sequential voting with relational box fields for active object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

20. Giancola, S., Amine, M., Dghaily, T., Ghanem, B.: Soccernet: A scalable dataset for action spotting in soccer videos. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (2018)

21. Goyal, M., Modi, S., Goyal, R., Gupta, S.: Human hands as probes for interactive object understanding. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

22. Grady, P., Collins, J.A., Tang, C., Twigg, C.D., Aneja, K., Hays, J., Kemp, C.C.: PressureVision++: Estimating fingertip pressure from diverse rgb images. In: Proceedings of the IEEE Winter Conference on Applications of Computer Vision (2024)

23. Grady, P., Tang, C., Brahmbhatt, S., Twigg, C.D., Wan, C., Hays, J., Kemp, C.C.: Pressurevision: Estimating hand pressure from a single rgb image. In: Proceedings of the European Conference on Computer Vision (2022)

24. Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Hamburger, J., Jiang, H., Liu, M., Liu, X., Martin, M., Nagarajan, T., Radosavovic, I., Ramakrishnan, S.K., Ryan, F., Sharma, J., Wray, M., Xu, M., Xu, E.Z., Zhao, C., Bansal, S., Batra, D., Cartillier, V., Crane, S., Do, T., Doulaty, M., Erapalli, A., Feichtenhofer, C., Fragomeni, A., Fu, Q., Gebreselasie, A., González, C., Hillis, J., Huang, X., Huang, Y., Jia, W., Khoo, W., Koláui, J., Kottur, S., Kumar, A., Landini, F., Li, C., Li, Y., Li, Z., Mangalam, K., Modhugu, R., Munro, J., Murrell, T., Nishiyasu, T., Price, W., Puentes, P.R., Ramazanova, M., Sari, L., Somasundaram, K., Southerland, A., Sugano, Y., Tao, R., Vo, M., Wang, Y., Wu, X., Yagi, T., Zhao, Z., Zhu, Y., Arbeláez, P., Crandall, D., Damen, D., Farinella, G.M., Fuegen, C., Ghanem, B., Ithapu, V.K., Jawahar, C.V., Joo, H., Kitani, K., Li, H., Newcombe, R., Oliva, A., Park, H.S., Rehg, J.M., Sato, Y., Shi, J., Shou, M.Z., Torralba, A., Torresani, L., Yan, M., Malik, J.: Ego4d: Around the world in 3,000 hours of egocentric video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

25. Grauman, K., Westbury, A., Torresani, L., Kitani, K., Malik, J., Afouras, T., Ashutosh, K., Baiyya, V., Bansal, S., Boote, B., Byrne, E., Chavis, Z., Chen,

J., Cheng, F., Chu, F.J., Crane, S., Dasgupta, A., Dong, J., Escobar, M., Forigua, C., Gebreselasie, A., Haresh, S., Huang, J., Islam, M.M., Jain, S., Khirodkar, R., Kukreja, D., Liang, K.J., Liu, J.W., Majumder, S., Mao, Y., Martin, M., Mavroudi, E., Nagarajan, T., Ragusa, F., Ramakrishnan, S.K., Seminara, L., Somayazulu, A., Song, Y., Su, S., Xue, Z., Zhang, E., Zhang, J., Castillo, A., Chen, C., Fu, X., Furuta, R., Gonzalez, C., Gupta, P., Hu, J., Huang, Y., Huang, Y., Khoo, W., Kumar, A., Kuo, R., Lakhavani, S., Liu, M., Luo, M., Luo, Z., Meredith, B., Miller, A., Oguntola, O., Pan, X., Peng, P., Pramanick, S., Ramazanova, M., Ryan, F., Shan, W., Somasundaram, K., Song, C., Southerland, A., Tateno, M., Wang, H., Wang, Y., Yagi, T., Yan, M., Yang, X., Yu, Z., Zha, S.C., Zhao, C., Zhao, Z., Zhu, Z., Zhuo, J., Arbelaez, P., Bertasius, G., Damen, D., Engel, J., Farinella, G.M., Furnari, A., Ghanem, B., Hofman, J., Jawahar, C., Newcombe, R., Park, H.S., Rehg, J.M., Sato, Y., Savva, M., Shi, J., Shou, M.Z., Wray, M.: Ego-exo4d: Understanding skilled human activity from first- and third-person perspectives. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2024)

26. Hong, J., Zhang, H., Gharbi, M., Fisher, M., Fatahalian, K.: Spotting Temporally Precise, Fine-Grained Events in Video. In: Proceedings of the European Conference on Computer Vision (2022)

27. Huang, M., Narasimhaswamy, S., Vazir, S., Ling, H., Hoai, M.: Forward propagation, backward regression, and pose association for hand tracking in the wild. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

28. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., Chang, K.W., Gao, J.: Grounded language-image pretraining. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

29. Liu, Y., Yang, H., Si, X., Liu, L., Li, Z., Zhang, Y., Liu, Y., Yi, L.: TACO: Benchmarking generalizable bimanual tool-action-object understanding. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2024)

30. Liu, Y., Liu, Y., Jiang, C., Lyu, K., Wan, W., Shen, H., Liang, B., Fu, Z., Wang, H., Yi, L.: HOI4D: A 4d egocentric dataset for category-level human-object interaction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

31. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: Proceedings of International Conference on Learning and Representation (2019)

32. Mur-Labadia, L., Muckley, M., Bar, A., Assran, M., Sinha, K., Rabbat, M., Le-Cun, Y., Ballas, N., Bardes, A.: V-jepa 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482 (2026)

33. Narasimhaswamy, S., Nguyen, H., Huang, L., Hoai, M.: Hoist-former: Hand-held objects identification, segmentation, and tracking in the wild. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2024)

34. Narasimhaswamy, S., Nguyen, T., Huang, M., Hoai, M.: Whose hands are these? hand detection and hand-body association in the wild. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2022)

35. Narasimhaswamy, S., Nguyen, T., Hoai, M.: Detecting hands and recognizing physical contact in the wild. In: Advances in Neural Information Processing Systems (2020)

36. Narasimhaswamy, S., Wei, Z., Wang, Y., Zhang, J., Hoai, M.: Contextual attention for hand detection in the wild. In: Proceedings of the International Conference on Computer Vision (2019)

37. Nguyen, H.A., Dayoub, F., Hoai, M.: Detecting precise hand touch moments in egocentric video. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Findings (2026)

38. Núñez-Marcos, A., Azkune, G., Arganda-Carreras, I.: Egocentric vision-based action recognition: A survey. Neurocomputing 472, 175–197 (2022)

39. Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K., Dollár, P.: Designing network design spaces. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2020)

40. Santra, S., Chudasama, V., Wasnik, P., Balasubramanian, V.N.: Precise event spotting in sports videos: Solving long-range dependency and class imbalance. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2025)

41. Shan, D., Geng, J., Shu, M., Fouhey, D.F.: Understanding Human Hands in Contact at Internet Scale . In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2020)

42. Shi, D., Zhong, Y., Cao, Q., Ma, L., Li, J., Tao, D.: TriDet: Temporal action detection with relative boundary modeling. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2023)

43. Soares, J.V.B., Shah, A.: Action spotting using dense detection anchors revisited: Submission to the soccernet challenge 2022. arXiv preprint arXiv:2206.07846 (2022)

44. Sudhakaran, S., Escalera, S., Lanz, O.: Gate-Shift Networks for Video Action Recognition . In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2020)

45. Sudhakaran, S., Escalera, S., Lanz, O.: Gate-shift-fuse for video action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(9), 10913– 10928 (2023)

46. Tateno, M., Kato, G., Kataoka, H., Sato, Y., Yagi, T.: Handyvqa: A video qa benchmark for fine-grained hand-object interaction dynamics. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2026)

47. Tran, K.H., Do, P.V., Ly, N.Q., Le, N.: Unifying Global and Local Scene Entities Modelling for Precise Action Spotting. In: Proceedings of the International Joint Conference on Neural Networks (2024)

48. Vo, K., Truong, S., Yamazaki, K., Raj, B., Tran, M.T., Le, N.: Aoe-net: Entities interactions modeling with adaptive attention mechanism for temporal action proposals generation. International Journal of Computer Vision 131(1), 302–323 (2023)

49. Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., Jiang, T., Li, S., Xu, J., Zhang, H., Huang, Y., Qiao, Y., Wang, Y., Wang, L.: Internvideo2: Scaling foundation models for multimodal video understanding. In: Proceedings of the European Conference on Computer Vision (2024)

50. Wightman, R., Touvron, H., Jégou, H.: ResNet strikes back: An improved training procedure in timm. arXiv preprint arXiv:2110.00476 (2021)

51. Xarles, A., Escalera, S., Moeslund, T.B., Clapés, A.: Astra: An action spotting transformer for soccer videos. In: Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports (2023)

52. Xarles, A., Escalera, S., Moeslund, T.B., Clapés, A.: T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (2024)

53. Xarles, A., Escalera, S., Moeslund, T.B., Clapés, A.: Adaspot: Spend resolution where it matters for precise event spotting. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2026)

54. Xu, B., Wang, Z., Du, Y., Song, Z., Zheng, S., Jin, Q.: Do egocentric video-language models truly understand hand-object interactions? In: Proceedings of International Conference on Learning and Representation (2025)

55. Xu, H., Baniya, A.A., Well, S., Bouadjenek, M.R., Dazeley, R., Aryal, S.: Deep learning for sports video event detection: Tasks, datasets, methods, and challenges. arXiv preprint arXiv:2505.03991 (2025)

56. Xu, M., Zhao, C., Rojas, D.S., Thabet, A., Ghanem, B.: G-TAD: Sub-graph localization for temporal action detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2020)

57. Yagi, T., Hasan, M.T., Sato, Y.: Hand-object contact prediction via motion-based pseudo-labeling and guided progressive label correction. In: Proceedings of the British Machine Vision Conference (2021)

58. Yi, F., Wen, H., Jiang, T.: Asformer: Transformer for action segmentation. In: Proceedings of the British Machine Vision Conference (2021)

59. Zhang, C.L., Wu, J., Li, Y.: Actionformer: Localizing moments of actions with transformers. In: Proceedings of the European Conference on Computer Vision (2022)

60. Zhang, H., Zhang, P., Hu, X., Chen, Y.C., Li, L., Dai, X., Wang, L., Yuan, L., Hwang, J.N., Gao, J.: Glipv2: Unifying localization and vision-language understanding. In: Advances in Neural Information Processing Systems (2022)

61. Zhao, Y., Misra, I., Krähenbühl, P., Girdhar, R.: Learning video representations from large language models. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2023)

62. Zhou, X., Kang, L., Cheng, Z., He, B., Xin, J.: Feature combination meets attention: Baidu soccer embeddings and transformer based temporal detection. arXiv preprint arXiv:2106.14447 (2021)