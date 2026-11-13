# Can an LVLM Perceive Event Boundaries Like a Human?

Hyolim Kang Jungsoo Lee Yeji Song Whie Jung Janghoon Cho Juntae Lee

Qualcomm AI Research

{hyolkang, jungsool, yejisong, whiejung, janghoon, juntlee}@qti.qualcomm.com

## Abstract

Large Vision-Language Models (LVLMs) reason effectively about video content, but whether they perceive its event boundaries in a manner consistent with humans remains unclear. We first investigate this through Generic Event Boundary Detection (GEBD), which identifies event boundaries naturally perceived by humans in continuous video, without relying on predefined event categories. On Kinetics-GEBD dataset, LVLMs miss many boundaries when predicting directly from frames, but effectively refine candidates proposed by an unsupervised discovery method, achieving state-ofthe-art unsupervised GEBD performance. We extend this discovery-refinement approach to hour-long egocentric video, where boundary discovery requires a broader semantic scale and whole-video LVLM refinement becomes impractical. We address these challenges through coarse-to-fine temporal discovery and localized LVLM accept/reject judgments. Human evaluation shows that the LVLM’s agreement with group consensus falls within the range observed among human annotators, with disagreements concentrated on low-consensus cases. Using this pipeline, we construct EgoBoundary, a benchmark for event boundary detection in hour-scale egocentric video. Representative supervised and unsupervised methods degrade substantially on EgoBoundary, highlighting the challenge of long-horizon event boundary detection.

## 1 Introduction

Humans naturally organize continuous visual experience into meaningful temporal units, perceiving boundaries when one coherent event gives way to another (Zacks et al., 2007). Generic Event Boundary Detection (GEBD) (Shou et al., 2021) formalizes this perceptual segmentation without assuming a predefined vocabulary of actions or events: rather than asking what action is occurring, it asks where the perceived structure of an ongoing experience changes. As this perceived structure may differ from person to person, event boundaries are inherently subjective. GEBD explicitly accounts for this subjectivity by collecting annotations from multiple annotators and using their agreement to identify perceptually salient boundaries.

(a) Unsupervised discovery + LVLM refinement  
![](images/d55e92e439536fa3a0f1a09b1bafd3a529f67131eedde5d8861691eb4e8caed9.jpg)

(b) Localized boundary judgment  
![](images/02f704ab31acfa07d6a5467583a49c036692db4bcadf69ba308609b863b44c75.jpg)  
Figure 1: Overview of the two key results. (a) Unsupervised discovery followed by LVLM refinement improves event boundary detection on Kinetics-GEBD. (b) Given local visual context around a candidate frame, the LVLM makes accept/reject judgments consistent with human annotators.

Recent Large Vision-Language Models (LVLMs) (Team, 2025b,a) have demonstrated remarkable capabilities in describing and reasoning about video content, including actions (Qi et al., 2026), interactions (Kang et al., 2026), and temporal context (Fateh et al., 2026). Yet recognizing or describing what happens in a video is different from identifying where humans perceive transitions between meaningful events (Zhang et al., 2023; Maaz et al., 2023; Li et al., 2025). GEBD provides a natural testbed for this distinction: it requires open-ended semantic judgment over temporal structure, while grounding that judgment in human perceptual consensus. If general-purpose LVLMs can make such judgments in a human-compatible manner, they could also provide a scalable alternative to costly multiannotator GEBD annotation, particularly for long, continuous video. We therefore ask: Can an LVLM perceive event boundaries in a manner consistent with humans?

We first investigate this question on Kinetics-GEBD (Shou et al., 2021). Our experiments reveal a clear asymmetry in what the LVLM can contribute (Fig. 1 (a)). When asked to detect event boundaries directly from raw video frames, it identifies plausible transitions but systematically misses a substantial portion of the boundary structure. In contrast, when an unsupervised discovery method first proposes candidate boundaries, the same LVLM becomes an effective refiner, achieving the strongest performance among unsupervised GEBD approaches. This suggests that current LVLMs benefit from explicit temporal proposals when reasoning about event boundaries, rather than predicting the entire boundary structure directly from raw video. We therefore adopt a discoveryrefinement formulation in which an unsupervised discovery provides candidate coverage and the LVLM contributes semantic judgment.

We next ask whether this discovery-refinement principle can scale to hour-long untrimmed egocentric video. This setting introduces two challenges. First, the appropriate granularity of an event boundary changes with the semantic scale of the video: compared with short Kinetics-GEBD clips, hourlong egocentric streams span much broader activities and contexts, making more macroscopic transitions the natural targets of boundary discovery. We therefore adopt a coarse-to-fine discovery scheme that identifies transitions at a broader temporal scale and localizes them at a finer one. Second, whole-video LVLM refinement becomes impractical at this temporal scale. We therefore reformulate refinement as localized accept/reject judgments around individual candidate boundaries (Fig. 1 (b)).

This reformulation turns the LVLM into a local boundary judge, raising a necessary validation question: can such localized LVLM judgments serve as reliable annotations? We evaluate the LVLM’s accept/reject decisions alongside those of human annotators. Its agreement with group consensus falls within the range observed among human annotators, and its disagreements are concentrated on transitions for which humans themselves show weaker consensus. These results provide empirical support for using localized LVLM boundary judgments as a scalable annotation mechanism.

Using this pipeline, we construct EgoBoundary, a benchmark for event boundary detection in hourscale egocentric video. We evaluate representative supervised and unsupervised GEBD methods and find that existing approaches struggle substantially in this long-horizon setting, exposing the challenge of recovering meaningful temporal structure from continuous egocentric streams.

Our contributions are threefold:

• We systematically investigate LVLMs’ ability to perceive taxonomy-free event boundaries, revealing that they are more effective at semantic refinement than exhaustive boundary discovery, achieving strong unsupervised performance on Kinetics-GEBD.

• We extend this discovery-refinement principle to hour-long egocentric video through multiscale temporal discovery and localized LVLM boundary judgments, and demonstrate that these judgments align with human perceptual consensus.

• We introduce EgoBoundary, a benchmark for event boundary detection in hour-scale egocentric video, and establish representative supervised, training-free, and unsupervised baselines that reveal the limitations of existing methods in long-horizon settings.

## 2 Related Work

## 2.1 Generic Event Boundary Detection

Grounded in cognitive studies of event perception (Zacks et al., 2007), Generic Event Boundary Detection (GEBD) (Shou et al., 2021) aims to identify moments where humans naturally segment continuous visual experience into meaningful events, without relying on predefined event categories. As event boundaries are inherently subjective and can occur at different temporal granularities, Kinetics-GEBD guides annotators to identify boundaries at roughly “one level deeper” than the video-level event. Despite this guidance, perceptual differences remain, so Kinetics-GEBD collects multiple annotations per video and retains only videos with sufficient inter-annotator consistency, while preserving diverse valid segmentations.

Building on this benchmark, supervised approaches (Tang et al., 2022; Zheng et al., 2024a; Li et al., 2022; Zheng et al., 2024b; Hwang et al., 2025) learn discriminative spatiotemporal changes, while unsupervised approaches exploit temporal self-similarity (Kang et al., 2022) or optical-flowbased motion cues (Gothe et al., 2024) to discover event boundaries.

Our work. To quantitatively assess whether LVLMs can perceive event boundaries in a humanconsistent manner, we benchmark their GEBD performance against existing methods. We find that directly prompting LVLMs to predict boundaries from video underperforms existing unsupervised approaches. In contrast, when candidate boundaries are proposed by an unsupervised temporal signal, LVLMs effectively refine them, substantially outperforming prior unsupervised GEBD methods.

## 2.2 LVLMs for Temporal Video Understanding

Large Vision-Language Models (LVLMs) have demonstrated strong capabilities in understanding and reasoning about video content (Zhang et al., 2023; Maaz et al., 2023; Li et al., 2025). Recent advances have further improved temporal grounding (Ren et al., 2024; Zeng et al., 2024; Liu et al., 2025) and long-video reasoning (Arnab et al., 2025), enabling models to describe and localize events over extended temporal contexts. However, understanding what happens in a video does not necessarily translate into the ability to exhaustively identify its temporal structure.

Beyond video understanding, recent studies have explored LVLMs as scalable annotators for temporal localization. In particular, CoTT (Kang et al., 2026) employs structured LVLM inference to generate dense atomic action annotations from egocentric videos.

Our work. We investigate whether LVLMs perceive event boundaries in a manner consistent with humans, beyond action-centric temporal localization. On Kinetics-GEBD, we quantitatively evaluate how well LVLM-predicted boundaries align with human annotations. For EgoBoundary construction, we further assess whether localized LVLM accept/reject judgments on candidate boundaries agree with human annotators, validating their potential for scalable annotation of hourlong egocentric video.

## 3 Methods

## 3.1 Problem Definition

Generic Event Boundary Detection (GEBD) formalizes the temporal boundaries that humans perceive while segmenting continuous video into meaningful event units. Given a video V of duration $T ,$ , let

$$
B = \{ b _ { 1 } , \ldots , b _ { N _ { B } } \}\tag{1}
$$

denote a set of reference event boundaries, where $N _ { B }$ is the number of boundaries and each $b _ { i } \in$ [0, T] represents a timestamp perceived as an event boundary. GEBD aims to predict a boundary set $\hat { B }$ that aligns with these reference boundaries.

## 3.2 Probing LVLM Boundary Perception

We first study how an LVLM can be used for GEBD on short videos. Our goal at this stage is not to design a specialized boundary detector, but to probe whether an LVLM can identify human-perceived event transitions and where its strengths and limitations lie. We consider two inference settings: i) direct LVLM boundary prediction and ii) twostage boundary prediction consisting of unsupervised boundary discovery followed by LVLM refinement.

Probe 1. Direct LVLM inference. In the direct setting, we provide the video to the LVLM $\mathcal { F } _ { \mathrm { L V L M } }$ and ask it to identify timestamps at which one coherent event unit transitions to another. Formally, we write

$$
\hat { B } _ { \mathrm { d i r e c t } } = \mathcal { F } _ { \mathrm { L V L M } } ( V ) ,\tag{2}
$$

where $\mathcal { F } _ { \mathrm { L V L M } }$ directly maps the visual sequence to a set of predicted event-boundary timestamps. This setting tests whether the LVLM can both discover candidate temporal changes and determine which of them constitute meaningful event boundaries without an explicit temporal proposal mechanism.

![](images/ca660c5e2ad0c80d7cc42ea065380b4e9049a193f1dbaa78a8a1ded4add31f10.jpg)

(a) Compute the Temporal Self-similarity Matrix (TSM)  
(b) Diagonal Convolution with a Contrastive kernel  
![](images/fcddedf162f6f7467dc765b27883ad975a63836835773f636c7be3a552f4f1c0.jpg)

![](images/6074933e4a32162dbeed955909e57816e234f4088288e4ed1ae3f7d44e34f438.jpg)  
Figure 2: Overview of the unsupervised boundary discovery algorithm.

Probe 2-1. Unsupervised boundary discovery. We next consider a two-stage inference setting, in which candidate boundaries are first discovered using an unsupervised temporal signal and then refined by the LVLM. The first stage therefore focuses on identifying potential temporal transitions without relying on LVLM-based reasoning.

As illustrated in Fig. 2, we uniformly sample $N _ { V }$ frames at timestamps $\{ t _ { i } \} _ { i = 1 } ^ { N _ { V } }$ and extract their visual embeddings $\{ f _ { i } \} _ { i = 1 } ^ { N _ { V } }$ using a frame encoder (He et al., 2016; Bolya et al., 2026). We then construct a temporal self-similarity matrix (TSM)

$$
S _ { i j } = \sin ( f _ { i } , f _ { j } ) ,\tag{3}
$$

where $S \in \mathbb { R } ^ { N _ { V } \times N _ { V } }$ represents the pairwise similarity between sampled frames.

Event transitions induce a characteristic local pattern in the TSM: frames within the same event tend to be more similar, whereas frames across an event transition tend to be less similar (Kang et al., 2022). To capture this contrast, we apply a contrastive kernel $\bar { \boldsymbol { K } } \in \mathbb { R } ^ { N _ { k } \times N _ { k } }$ , where $N _ { k }$ is odd, along the diagonal of the TSM. For the i-th sampled frame, we compute the boundary score

$$
s ( t _ { i } ) = \langle K , S ^ { ( i ) } \rangle ,\tag{4}
$$

where $S ^ { ( i ) }$ is the $N _ { k } \times N _ { k }$ local TSM patch centered at $( i , i )$ , and $\langle \cdot , \cdot \rangle$ denotes the element-wise inner product. The contrastive kernel assigns positive weights to within-event regions and negative weights to cross-event regions, producing a high response when the local similarity structure exhibits a boundary-like transition.

Sliding the kernel along the TSM diagonal produces a one-dimensional boundary-score sequence. We apply peak detection to this sequence and take the timestamps corresponding to the detected peaks as candidate boundaries:

$$
\begin{array} { r l } & { \hat { B } _ { \mathrm { c a n d } } = \left\{ t _ { i } ~ \Big | ~ i \in \mathrm { P e a k D e t e c t } \left( \{ s ( t _ { j } ) \} _ { j = 1 } ^ { N _ { V } } \right) \right\} } \\ & { ~ = \{ c _ { 1 } , \dots , c _ { N _ { \mathrm { c a n d } } } \} , } \end{array}\tag{5}
$$

where $N _ { c a n d }$ denotes the number of candidate boundaries. These candidates provide explicit temporal proposals for the subsequent LVLM refinement stage.

Probe 2-2. LVLM refinement. Rather than requiring the LVLM to search the entire temporal space, we condition it on the discovered candidates and ask it to refine the proposed boundary structure:

$$
\begin{array} { r } { \hat { B } _ { \mathrm { r e f i n e } } = \mathcal { R } _ { \mathrm { L V L M } } ( V , \hat { B } _ { c a n d } ) . } \end{array}\tag{6}
$$

This formulation separates two roles that are entangled in direct inference: the unsupervised temporal signal is responsible for proposing potential transition points, while the LVLM is responsible for semantic judgment over those proposals. Comparing these two settings allows us to probe whether the LVLM is more effective at exhaustive boundary discovery or at reasoning about already localized temporal transitions.

![](images/a29d663c76d2a6997ba6d022f27c39fa7634dcf5f5142c3a723da6f4a68ad345.jpg)  
Figure 3: Coarse-to-fine discovery

## 3.3 Scaling Boundary Perception to Long-Form Egocentric Video

The proposed discovery–refinement formulation provides a natural starting point for long-form video, but directly applying the short-video refinement procedure to hour-long egocentric streams is impractical. In particular, repeatedly reasoning over the full visual sequence requires excessive visual context and makes temporal localization increasingly difficult. We therefore retain unsupervised temporal discovery while reformulating LVLM refinement as a localized boundary judgment.

Coarse-to-fine multi-scale boundary discovery. The temporal granularity of an event boundary is relative to the semantic scale of the video. While fine-grained transitions are appropriate for short Kinetics-GEBD clips, hour-long egocentric videos span much broader activities and contexts, such that the same “one-level-deeper” principle favors more macroscopic event transitions. We therefore adopt a coarse-to-fine scheme that discovers boundaries at a broader temporal scale and uses a finer scale only for precise localization.

As illustrated in Fig. 3, we sample the video at two temporal resolutions and compute a boundaryscore sequence at each scale using the same TSMbased procedure. Let $\{ t _ { i } ^ { c } \} _ { i = 1 } ^ { N _ { c } }$ and $\{ t _ { j } ^ { f } \} _ { j = 1 } ^ { N _ { f } }$ denote the sampled timestamps at the coarse and fine scales, respectively, with corresponding boundary scores $s _ { c } ( \cdot )$ and $s _ { f } ( \cdot )$

We perform peak detection only on the coarsescale score sequence:

$$
P _ { c } = \mathrm { P e a k D e t e c t } \left( \{ s _ { c } ( t _ { i } ^ { c } ) \} _ { i = 1 } ^ { N _ { c } } \right) ,\tag{7}
$$

where $P _ { c }$ contains coarse-scale peak indices.

For each coarse peak $p \in P _ { c } ,$ let $I ( p )$ denote the temporal interval associated with the coarse sample at timestamp $t _ { p } ^ { c } .$ . We localize the boundary by selecting the fine-scale timestamp with the highest score within this interval:

$$
c _ { p } = \underset { t _ { j } ^ { f } \in I ( p ) } { \arg \operatorname* { m a x } } s _ { f } ( t _ { j } ^ { f } ) .\tag{8}
$$

The final candidate set is therefore

$$
\hat { B } _ { c a n d } = \{ c _ { p } \mid p \in P _ { c } \} .\tag{9}
$$

In this way, the coarse scale determines which temporal transitions are salient over a broader context, while the fine scale is used only to localize each selected transition more precisely.

Localized LVLM boundary judgment. For each candidate $c _ { i } ~ \in ~ \hat { B } _ { c a n d } .$ , we extract a local visual context centered on the candidate and ask the LVLM whether the candidate corresponds to a meaningful event boundary. We formulate this as a binary decision

$$
y _ { i } = \mathcal { I } _ { \mathrm { L V L M } } \left( V _ { c _ { i } } , c _ { i } \right) , \qquad y _ { i } \in \{ 0 , 1 \} ,\tag{10}
$$

where $V _ { c _ { i } }$ denotes the local visual context around $c _ { i } ,$ and $y _ { i } = 1$ indicates that the candidate is accepted as an event boundary. The final prediction is

$$
\hat { B } = \left\{ c _ { i } \in \hat { B } _ { c a n d } \vert y _ { i } = 1 \right\} .\tag{11}
$$

This localized formulation substantially reduces the visual context presented to the LVLM and converts long-form boundary detection into a sequence of local semantic judgments.

<table><tr><td>Supervision</td><td>Method</td><td>Rel.Dis=0.05</td><td>0.1</td><td>0.15</td><td>0.2</td><td>0.25</td><td>0.3</td><td>0.35</td><td>0.4</td><td>0.45</td><td>0.5</td><td>Avg</td></tr><tr><td rowspan="10">Supervised</td><td>BMN (Lin et al., 2019)</td><td>0.186 0.491</td><td>0.204</td><td>0.213 0.627</td><td>0.220 0.648</td><td>0.226 0.660</td><td>0.230 0.668</td><td>0.233 0.674</td><td>0.237 0.678</td><td>0.239</td><td>0.241 0.683</td><td>0.223</td></tr><tr><td>BMN-StartEnd (Lin et al., 2019)</td><td></td><td>0.589</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.681</td><td></td><td>0.640</td></tr><tr><td>TCN-TAPOS (Shou et al., 2021) TCN (Lea et al., 2016)</td><td>0.464</td><td>0.560</td><td>0.602</td><td>0.628</td><td>0.645</td><td>0.659</td><td>0.669</td><td>0.676</td><td>0.682</td><td>0.687</td><td>0.627</td></tr><tr><td></td><td>0.588</td><td>0.657</td><td>0.679</td><td>0.691</td><td>0.698</td><td>0.703</td><td>0.706</td><td>0.708</td><td>0.710</td><td>0.712</td><td>0.685</td></tr><tr><td>PC (Shou et al., 2021)</td><td>0.625</td><td>0.758</td><td>0.804</td><td>0.829</td><td>0.844</td><td>0.853</td><td>0.859</td><td>0.864</td><td>0.867</td><td>0.870</td><td>0.817</td></tr><tr><td>Temporal Perceiver (Tan et al., 2023)</td><td>0.748</td><td>0.828</td><td>0.852</td><td>0.866</td><td>0.874</td><td>0.879</td><td>0.883</td><td>0.887</td><td>0.890</td><td>0.892</td><td>0.860</td></tr><tr><td>SBoCo-Res50 (Kang et al., 2022) DDM-Net (Tang et al., 2022)</td><td>0.732</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.866</td></tr><tr><td>SC-Transformer (Li et al., 2022)</td><td>0.764 0.777</td><td>0.843 0.849</td><td>0.866 0.873</td><td>0.880 0.886</td><td>0.887</td><td>0.892</td><td>0.895</td><td>0.898</td><td>0.900</td><td>0.902</td><td>0.873</td></tr><tr><td>EfficientGEBD (Zheng et al., 2024b)</td><td>0.783</td><td>0.851</td><td></td><td></td><td>0.895</td><td>0.900</td><td>0.904</td><td>0.907</td><td>0.909</td><td>0.911</td><td>0.881</td></tr><tr><td>LCVSL (Zhang et al., 2024)</td><td>0.768</td><td>0.848</td><td>0.872</td><td>0.885</td><td>0.892</td><td>0.901 0.896</td><td>0.899</td><td>0.901</td><td>0.903</td><td>0.913 0.906</td><td>0.883 0.877</td></tr><tr><td>DyBDet (Zheng et al., 2024a)</td><td>0.796</td><td>0.858</td><td>0.880</td><td>0.893</td><td>0.901</td><td>0.907</td><td>0.911</td><td>0.915</td><td></td><td></td><td></td><td>0.890</td></tr><tr><td rowspan="7"></td><td></td><td></td><td></td><td></td><td>0.319</td><td></td><td></td><td></td><td></td><td>0.917</td><td>0.919</td><td></td></tr><tr><td>SceneDetect (Shou et al., 2021) PA-Random (Shou et al., 2021)</td><td>0.275 0.336</td><td>0.300 0.435</td><td>0.312</td><td></td><td>0.324</td><td>0.327</td><td>0.330 0.548</td><td>0.332</td><td>0.334 0.558</td><td>0.335 0.561</td><td>0.318 0.506</td></tr><tr><td>PA (Shou et al., 2021)</td><td>0.396</td><td>0.488</td><td>0.484 0.520</td><td>0.512 0.534</td><td>0.529 0.544</td><td>0.541 0.550</td><td>0.555</td><td>0.554 0.558</td><td>0.561</td><td>0.564</td><td>0.527</td></tr><tr><td>CoSeg (Wang et al., 2023)</td><td>0.656</td><td>0.758</td><td>0.783</td><td>0.794</td><td>0.799</td><td>0.803</td><td>0.804</td><td>0.806</td><td>0.807</td><td>0.809</td><td>0.782</td></tr><tr><td>UBoCo-Res50 (Kang et al., 2022)</td><td>0.703</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FlowGEBD (Gothe et al., 2024)</td><td>0.713</td><td>0.828</td><td>0.850</td><td>0.858</td><td>0.862</td><td>0.864</td><td>0.866</td><td>0.867</td><td>0.868</td><td>0.869</td><td>0.845</td></tr><tr><td>Ours (Discovery + LVLM)</td><td>0.7448</td><td>0.8367</td><td>0.8612</td><td>0.8738</td><td>0.8810</td><td>0.8855</td><td>0.8888</td><td>0.8916</td><td>0.8940</td><td>0.8965</td><td>0.8654</td></tr></table>

Table 1: Comparison with previous methods on Kinetics-GEBD. F1 scores are reported across Rel.Dis. thresholds.

## 4 Experiment

## 4.1 Datasets and Evaluation

Kinetics-GEBD. We evaluate on Kinetics-GEBD (Shou et al., 2021), a standard GEBD benchmark constructed from short Kinetics-400 clips with multiple independent human annotations per video. We use only the official validation split, which contains approximately 18K videos. At Rel.Dis. threshold $d ,$ a predicted boundary <sup>ˆ</sup>b matches an annotated boundary b if

$$
{ \frac { | { \hat { b } } - b | } { T } } \leq d ,\tag{12}
$$

where T is the video duration. Predictions are matched one-to-one against each annotator independently, and the best-matching annotation is used. Following the official protocol, we report F1 for $d = 0 . 0 5  – 0 . 5 0$ in increments of 0.05, with F1@0.05 as the primary metric. To quantify human subjectivity, we additionally divide each video into 20 temporal bins, represent each annotation as binary boundary occupancy, and report the Fleiss κ (Fleiss, 1971) across annotators.

EgoBoundary. EgoBoundary is constructed from 42 EgoLife (Yang et al., 2025) videos totaling 266 hours and contains 5,282 event boundaries produced by our coarse-to-fine discovery and localized LVLM judgment pipeline. We split the dataset into 35 training and 7 validation videos. Following established segment-level evaluation protocols in temporal segmentation (Kang et al., 2021, 2024, 2025, 2026; Song et al., 2024; Reza et al., 2024), we convert predicted and reference boundaries into temporal segments defined by consecutive boundaries. We perform one-to-one Hungarian matching between predicted and reference segments based on temporal Intersection over Union (tIoU), counting matched pairs above threshold τ as correct. We report segment-level F1 at $\tau \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ adopting $\tau = 0 . 5$ as the primary metric following common practice.

Human agreement. To validate the localized LVLM judge, we sample 100 candidates judged by Gemini 2.5 Pro, balanced between 50 accepted and 50 rejected cases. Ten human annotators independently make accept/reject decisions on the same candidate boundaries, using extended local video context. We measure inter-annotator agreement among humans using Fleiss’ κ (Fleiss, 1971). To assess individual agreement with human consensus, we compute Cohen’s κ (Cohen, 1960) between each judge’s decisions and the majority vote of the other human annotators. For each human annotator, consensus excludes their own judgments (leaveone-out), while the LVLM is evaluated against the consensus of all ten humans. Tied votes are excluded.

## 4.2 Implementation Details

We use Gemini 2.5 Pro (Team, 2023) as the LVLM throughout our experiments. Details on frame sampling, feature extraction, boundary discovery, and LVLM refinement are provided in the supplementary material.

## 4.3 Results on Kinetics-GEBD

Comparison with prior methods. Table 1 compares our discovery–refinement approach with existing GEBD methods on Kinetics-GEBD. Our method achieves an F1@0.05 of 0.7448, significantly outperforming previous unsupervised approaches at the strictest Rel.Dis. threshold. This advantage is consistently maintained across the individual Rel.Dis. thresholds, demonstrating that LVLM refinement effectively improves the temporal boundaries produced by unsupervised discovery. These results show that combining unsupervised temporal discovery with LVLM-based refinement provides strong GEBD performance without supervised training.

<table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Direct inference</td><td>0.8500</td><td>0.5540</td><td>0.6708</td></tr><tr><td>Unsupervised discovery</td><td>0.6871</td><td>0.7505</td><td>0.7174</td></tr><tr><td>Combination</td><td>0.6910</td><td>0.8077</td><td>0.7448</td></tr></table>

Table 2: Comparison of direct LVLM inference, unsupervised boundary discovery, and their combination on Kinetics-GEBD at Rel.Dis. = 0.05.

What does the LVLM contribute? Table 2 decomposes the performance of the three inference settings at the primary Rel.Dis. threshold of 0.05. Direct LVLM inference exhibits a strongly precision-oriented behavior, achieving the highest precision of 0.8500 but substantially lower recall of 0.5540. This suggests that the LVLM can recognize clear event transitions when it identifies them, but fails to exhaustively discover the boundary structure of the video. In contrast, unsupervised discovery provides substantially higher recall (0.7505), at the cost of lower precision.

Combining the two yields a complementary effect. LVLM refinement significantly increases recall from 0.7505 to 0.8077 while slightly improving precision from 0.6871 to 0.6910, resulting in the best F1 of 0.7448. This corresponds to a +0.0274 improvement over unsupervised discovery alone and a +0.0740 improvement over direct LVLM inference. These results indicate that the LVLM is more effective at refining an explicitly proposed temporal boundary structure than at discovering all event boundaries directly from the video.

## 4.4 Results on EgoBoundary

Event-boundary perception is inherently subjective even in the standard Kinetics-GEBD benchmark. Using its original multi-annotator annotations, we obtain a Fleiss’ κ of 0.294, indicating substantial inter-annotator variability. We observe a similar level of disagreement in our egocentric human study, where the ten annotators achieve a

Table 3: Agreement with leave-one-out group consensus for the LVLM and human annotators.
<table><tr><td>Rank</td><td>Judge</td><td>κ</td></tr><tr><td>1</td><td>Human A</td><td>0.534</td></tr><tr><td>2</td><td>Human B</td><td>0.510</td></tr><tr><td>3</td><td>Human C</td><td>0.508</td></tr><tr><td>4</td><td>Human D</td><td>0.477</td></tr><tr><td>5</td><td>Human E</td><td>0.398</td></tr><tr><td>6</td><td>LVLM</td><td>0.394</td></tr><tr><td>7</td><td>Human F</td><td>0.388</td></tr><tr><td>8</td><td>Human G</td><td>0.377</td></tr><tr><td>9</td><td>Human H</td><td>0.369</td></tr><tr><td>10</td><td>Human I</td><td>0.337</td></tr><tr><td>11</td><td>Human J</td><td>0.248</td></tr></table>

Fleiss’ κ of 0.345. Given this inherent subjectivity, we evaluate whether the LVLM’s agreement with human consensus falls within the variability observed among individual human annotators, rather than expecting perfect agreement with a single consensus segmentation.

Can the LVLM serve as a local boundary judge? Before using localized LVLM judgments to construct EgoBoundary, we first examine whether these judgments are compatible with human boundary perception. We compare the LVLM with ten human annotators using leave-one-out group consensus, treating every judge under the same evaluation protocol. As shown in Table 3, the LVLM achieves a Cohen’s κ of 0.394 and ranks sixth among the eleven judges. The average human agreement is 0.415, while five human annotators obtain lower agreement than the LVLM. Thus, despite the inherent ambiguity of event boundary judgments, the LVLM does not behave as an outlier relative to human annotators. Instead, its consistency with human consensus falls within the observed distribution of individual human judgments.

Importantly, the remaining disagreement is concentrated on boundaries that humans themselves find ambiguous. We measure human consensus by the vote margin between accept and reject decisions. Table 4 shows that, as we restrict evaluation to candidates with stronger human consensus, the LVLM’s agreement increases monotonically from κ = 0.394 to 0.721, while the false-reject rate decreases from 41% to 12%. This suggests that disagreements are not uniformly distributed across candidate boundaries, but are concentrated on cases where human annotators also disagree.

Does multi-scale discovery produce better candidates? We next evaluate the coarse-to-fine discovery strategy used for long-form video. To isolate candidate quality from candidate quantity, we compare coarse-to-fine discovery with fine-scaleonly discovery under the same candidate budget of 6,364 proposals. We then apply the same localized LVLM judge to both candidate sets.

Table 4: LVLM agreement as a function of human consensus. Margin denotes the absolute difference between accept and reject votes from ten human annotators.
<table><tr><td>Margin ≥</td><td>N</td><td>κ</td><td>Agree.</td><td>FR</td></tr><tr><td>2</td><td>87</td><td>0.394</td><td>70%</td><td>41%</td></tr><tr><td>4</td><td>69</td><td>0.551</td><td>78%</td><td>35%</td></tr><tr><td>6</td><td>58</td><td>0.597</td><td>81%</td><td>33%</td></tr><tr><td>8</td><td>39</td><td>0.667</td><td>85%</td><td>27%</td></tr><tr><td>10</td><td>23</td><td>0.721</td><td>87%</td><td>12%</td></tr></table>

Table 5: Comparison of coarse-to-fine and single-scale boundary discovery under the same candidate budget.
<table><tr><td>Discovery</td><td>Initial</td><td>Rejected</td><td>Accepted</td></tr><tr><td>Coarse-to-fine</td><td>6,364</td><td>17.0%</td><td>83.0%</td></tr><tr><td>Fine only</td><td>6,364</td><td>22.3%</td><td>77.7%</td></tr></table>

As shown in Table 5, coarse-to-fine discovery results in a lower rejection rate of 17.0%, compared with 22.3% for fine-scale-only discovery. The coarse temporal scale more frequently places proposals at transitions that survive subsequent boundary judgment.

Do discovered boundaries align with human perception? To further validate our annotation pipeline against human-perceived event boundaries, we manually annotate 3 EgoLife videos totaling 13 hours, yielding 230 reference boundaries. We evaluate both the initial coarse-to-fine discovery and the LVLM-refined boundaries against these annotations using segment-level tIoU-F1. To account for the effect of segment density, we additionally consider uniform segmentation baselines.

Table 6: Comparison against manually annotated event boundaries. #Bnd. denotes the total number of predicted boundaries. Uniform baselines evenly partition each video using the same boundary counts as the corresponding predictions or ground truth.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#Bnd.</td><td colspan="3">F1 @ tIoU</td></tr><tr><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td>Discovery</td><td>265</td><td>0.7625</td><td>0.6188</td><td>0.4391</td></tr><tr><td>+ LVLM refinement</td><td>228</td><td>0.8276</td><td>0.7112</td><td>0.5259</td></tr><tr><td>Uniform (n=discovery)</td><td>265</td><td>0.2515</td><td>0.0639</td><td>0.0040</td></tr><tr><td>Uniform (n=refined)</td><td>228</td><td>0.2457</td><td>0.0474</td><td>0.0043</td></tr><tr><td>Uniform (n=GT)</td><td>230</td><td>0.2833</td><td>0.0687</td><td>0.0043</td></tr></table>

Table 7: Baseline performance on EgoBoundary using segment-level tIoU-F1.
<table><tr><td>Supervision</td><td>Method</td><td>Feat.</td><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td>Supervised</td><td>BasicGEBD</td><td>G14 S16</td><td>0.4783 0.4723</td><td>0.3052 0.2490</td><td>0.1823 0.1307</td></tr><tr><td rowspan="2">Unsupervised</td><td>UBoCo</td><td>G14 S16</td><td>0.3120 0.2318</td><td>0.1544 0.1181</td><td>0.0649 0.0452</td></tr><tr><td>CoSeg-FFR</td><td>G14 S16</td><td>0.2544 0.2720</td><td>0.0983 0.1017</td><td>0.0133 0.0205</td></tr></table>

As shown in Table 6, unsupervised discovery achieves 0.7625 F1 at tIoU=0.5, far exceeding the count-matched uniform baseline (0.2515), demonstrating its ability to identify meaningful event transitions. LVLM refinement further improves F1 to 0.8276, with consistent gains at stricter thresholds, demonstrating its effectiveness in filtering spurious boundaries.

How challenging is EgoBoundary? Finally, we evaluate representative GEBD approaches on the completed EgoBoundary benchmark. We consider a supervised method (BasicGEBD (Zheng et al., 2024b)), two unsupervised method (UBoCo (Kang et al., 2022), CoSeg-FFR (Wang et al., 2023)). We train BasicGEBD on the EgoBoundary training split and evaluate it on the validation split, while applying UBoCo and CoSeg-FFR without boundarylabel supervision. Table 7 reports segment-level F1 under temporal IoU thresholds of 0.5, 0.7, and 0.9. We report both PE-Core feature variants used in our evaluation.

Despite their competitive performance on short (10-second) Kinetics-GEBD clips, existing methods struggle on EgoBoundary, with even the supervised baseline achieving only 0.4783 F1 at tIoU= 0.5. This highlights the distinct challenges of hour-scale egocentric video and leaves substantial room for improvement.

## 5 Conclusion

We show that LVLMs are more effective at refining unsupervised event boundary proposals than discovering them directly. Extending this finding to hour-long egocentric video, we develop a coarseto-fine discovery and localized judgment pipeline validated against human annotations. We introduce EgoBoundary, a long-horizon GEBD benchmark highlighting the limitations of existing methods.

## 6 Limitations

LVLM coverage and inference cost. Our experiments primarily rely on Gemini 2.5 Pro due to the substantial cost of large-scale LVLM inference. Evaluating multiple LVLMs on approximately 18K Kinetics-GEBD videos and 266 hours of egocentric footage would require considerable computational resources.

Dependence on boundary discovery. Our longform pipeline relies on unsupervised temporal signals to propose candidate boundaries, while the LVLM only accepts or rejects them. Consequently, meaningful transitions missed during discovery cannot be recovered by subsequent refinement. To mitigate this limitation, we use a relatively permissive peak-prominence threshold to favor candidate recall, allowing the LVLM to filter out spurious proposals.

Scope of human validation. Our human agreement study covers 100 candidate boundaries, while independent manual boundary annotations span approximately 13 hours of egocentric video. Although these evaluations support the reliability of our pipeline, they cannot fully capture the subjectivity of event perception across diverse contexts and annotators. Moreover, EgoBoundary is constructed exclusively from EgoLife, leaving generalization to other long-form video domains unexplored.

## References

Anurag Arnab, Ahmet Iscen, Mathilde Caron, Alireza Fathi, and Cordelia Schmid. 2025. Temporal chain of thought: Long-video understanding by thinking in frames. In Advances in Neural Information Processing Systems (NeurIPS).

Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Bangalath, et al. 2026. Perception encoder: The best visual embeddings are not at the output of the network. In Advances in Neural Information Processing Systems (NeurIPS).

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and psychological measurement.

Fawad Javed Fateh, Umer Ahmed, Hamza Khan, Zeeshan Zia, and Quoc-Huy Tran. 2026. Temporalvlm: Video llms for temporal reasoning in long videos. In Findings ofthe Associationfor Computational Lin guistics (ACL).

Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological bulletin.

Sourabh Vasant Gothe, Vibhav Agarwal, Sourav Ghosh, Jayesh Rajkumar Vachhani, Pranay Kashyap, and Barath Raj Kandur. 2024. What’s in the flow? exploiting temporal motion cues for unsupervised generic event boundary detection. In IEEE Winter Conference on Applications ofComputer Vision (WACV).

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Jaejun Hwang, Dayoung Gong, Manjin Kim, and Minsu Cho. 2025. Generic event boundary detection via denoising diffusion. In IEEE/CVF International Conference on Computer Vision (ICCV).

Hyolim Kang, Jeongseok Hyun, Joungbin An, Youngjae Yu, and Seon Joo Kim. 2024. Actionswitch: Classagnostic detection of simultaneous actions in streaming videos. In European Conference on Computer Vision (ECCV).

Hyolim Kang, Jinwoo Kim, Taehyun Kim, and Seon Joo Kim. 2022. Uboco: Unsupervised boundary contrastive learning for generic event boundary detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Hyolim Kang, Kyungmin Kim, Yumin Ko, and Seon Joo Kim. 2021. Cag-qil: Context-aware actionness grouping via q imitation learning for online temporal action localization. In IEEE/CVF International Conference on Computer Vision (ICCV).

Hyolim Kang, Jungsoo Lee, Janghoon Cho, Juntae Lee, Seon Joo Kim, and Sungha Choi. 2026. Streaming atomic action segmentation in egocentric videos via iterative lvlm-to-streaming distillation. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Hyolim Kang, Yunsu Park, Youngbeom Yoo, Yeeun Choi, and Seon Joo Kim. 2025. Open-ended hierarchical streaming video understanding with vision language models. In IEEE/CVF International Conference on Computer Vision (ICCV).

Colin Lea, Austin Reiter, René Vidal, and Gregory D Hager. 2016. Segmental spatiotemporal cnns for fine-grained action segmentation. In European Conference on Computer Vision (ECCV).

Congcong Li, Xinyao Wang, Dexiang Hong, Yufei Wang, Libo Zhang, Tiejian Luo, and Longyin Wen. 2022. Structured context transformer for generic event boundary detection. arXiv preprint arXiv:2206.02985.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2025. Videochat: Chat-centric video understanding. Science China Information Sciences.

Tianwei Lin, Xiao Liu, Xin Li, Errui Ding, and Shilei Wen. 2019. Bmn: Boundary-matching network for temporal action proposal generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, and Jiaxuan You. 2025. Time-r1: Towards comprehensive temporal reasoning in llms. arXiv preprint arXiv:2505.13508.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. 2023. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424.

Haozhe Qi, Shaokai Ye, Alexander Mathis, and Mackenzie Mathis. 2026. Llavaction: evaluating and training multi-modal large language models for action understanding. In International Conference on Learning Representations (ICLR).

Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. 2024. Timechat: A time-sensitive multimodal large language model for long video understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Sakib Reza, Yuexi Zhang, Mohsen Moghaddam, and Octavia Camps. 2024. Hat: History-augmented anchor transformer for online temporal action localization. In European Conference on Computer Vision (ECCV).

Mike Zheng Shou, Stan Weixian Lei, Weiyao Wang, Deepti Ghadiyaram, and Matt Feiszli. 2021. Generic event boundary detection: A benchmark for event segmentation. In IEEE/CVF International Conference on Computer Vision (ICCV).

Youngkil Song, Dongkeun Kim, Minsu Cho, and Suha Kwak. 2024. Online temporal action localization with memory-augmented transformer. In European Conference on Computer Vision (ECCV).

Jing Tan, Yuhong Wang, Gangshan Wu, and Limin Wang. 2023. Temporal perceiver: A general architecture for arbitrary boundary detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Jiaqi Tang, Zhaoyang Liu, Chen Qian, Wayne Wu, and Limin Wang. 2022. Progressive attention on multilevel dense difference maps for generic event boundary detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Gemini Team. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

OpenAI Team. 2025a. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Qwen Team. 2025b. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923.

Xiao Wang, Jingen Liu, Tao Mei, and Jiebo Luo. 2023. Coseg: Cognitively inspired unsupervised generic event segmentation. IEEE Transactions on Neural Networks and Learning Systems.

Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. 2025. Egolife: Towards egocentric life assistant. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Jeffrey M Zacks, Nicole K Speer, Khena M Swallow, Todd S Braver, and Jeremy R Reynolds. 2007. Event perception: a mind-brain perspective. Psychological bulletin.

Xiangyu Zeng, Kunchang Li, Chenting Wang, Xinhao Li, Tianxiang Jiang, Ziang Yan, Songze Li, Yansong Shi, Zhengrong Yue, Yi Wang, Yali Wang, Yu Qiao, and Limin Wang. 2024. Timesuite: Improving mllms for long video understanding via grounded tuning. In International Conference on Learning Representations (ICLR).

Hang Zhang, Xin Li, and Lidong Bing. 2023. Videollama: An instruction-tuned audio-visual language model for video understanding. In Conference on Empirical Methods in Natural Language Processing (EMNLP).

Libo Zhang, Xin Gu, Congcong Li, Tiejian Luo, and Heng Fan. 2024. Local compressed video stream learning for generic event boundary detection. International Journal ofComputer Vision (IJCV).

Ziwei Zheng, Lijun He, Le Yang, and Fan Li. 2024a. Fine-grained dynamic network for generic event boundary detection. In European Conference on Computer Vision (ECCV).

Ziwei Zheng, Zechuan Zhang, Yulin Wang, Shiji Song, Gao Huang, and Le Yang. 2024b. Rethinking the architecture design for efficient generic event boundary detection. In Proceedings ofthe 32nd ACM International Conference on Multimedia.

Table 8: Summary of licenses and usage terms for external datasets and models.
<table><tr><td>Artifact</td><td>License / Terms</td></tr><tr><td>Kinetics-GEBD</td><td>CC BY-NC 4.0</td></tr><tr><td>EgoLife</td><td>S-Lab License 1.0</td></tr><tr><td>Perception Encoder Apache License 2.0</td><td></td></tr><tr><td>Gemini 2.5 Pro</td><td>Google Gemini API Terms</td></tr></table>

## A License

Table 8 summarizes the licensing terms of the key external datasets and models used in this work. We use external datasets and models for research and evaluation purposes, subject to their respective licensing terms and usage policies. In particular, the use and redistribution of source videos and derived annotations are subject to the conditions of the original datasets. Gemini 2.5 Pro is accessed through its API under Google’s applicable service terms.

## B Potential Risk

EgoBoundary is designed for research on perceptual event segmentation, not for identifying individuals or making consequential decisions. The source EgoLife videos have faces blurred to protect individual privacy. We use these videos in accordance with the original dataset’s access conditions and usage restrictions.

## C AI Usage

We utilized GPT5 (Team, 2025a) to fix typos and polish sentences, and Claude Code for minor code revisions.

## D Implementation Details

LVLM access. We use a commercial multimodal model (Gemini 2.5 Pro) (Team, 2023) through its API, supplying sampled video frames.

Kinetics-GEBD. Following the official Kinetics-GEBD protocol, we sample videos at 4 FPS and use an ImageNet-pretrained ResNet-50 as the visual encoder. Frame embeddings are used to construct the TSM and compute boundary scores as described in Sec. 3.2.

EgoBoundary. For long-form egocentric videos, we use the coarse-to-fine sampling scheme. We sample frames at 0.25 FPS for coarse-scale discovery and at 1 FPS for fine-scale localization. At both scales, we encode frames using the S16 Perception Encoder (Bolya et al., 2026). For peak detection on the coarse-scale boundary scores, we set the peak-prominence threshold such that the resulting candidate boundaries yield an average segment duration of approximately 150 seconds. For localized LVLM refinement, we provide frames sampled at 1 FPS within a ±20-second window around each candidate boundary and ask the LVLM to accept or reject it.

## E Prompts and Annotation Instructions

This appendix reproduces, in full, every instruction given to a model or to a human annotator. Section E.1 covers the Kinetics-GEBD prompting experiments, Section E.2 the LLM verification step used to construct EgoBoundary, and Section E.3 the instructions shown to human annotators.

The Kinetics prompts ask the model to produce boundaries, whereas the EgoBoundary prompt asks it to verify one proposed boundary at a time and can only remove candidates.

## E.1 Kinetics-GEBD prompting

All Kinetics-GEBD prompts address the whole clip at once: the instruction text is followed by the clip’s frames, each preceded by a timestamp line [time X.XXs], so the model has access to future frames and the setting is non-causal. Frames are decoded at 4 fps and center-cropped to 224 × 224.

## E.1.1 Zero-shot boundary detection

You are given frames sampled at a fixed rate in chronological order from a short video clip. Each frame is preceded by its timestamp [time X.XXs] relative to the start of the clip.

Your task is Generic Event Boundary Detection (GEBD): identify the moments where a human would naturally perceive the video splitting into meaningful temporal segments -- WITHOUT any predefined taxonomy of actions. A boundary typically corresponds to one of the following (per Shou et al., 'Generic Event Boundary Detection', ICCV 2021):

1) Change of Subject: a new subject (person/object) appears or becomes dominant, or the previous dominant subject disappears.

2) Change of Object of Interaction: the subject starts interacting with a new object, or finishes interacting with the previous one.

3) Change of Action: the dominant action changes (including a sudden change of direction, or repetition boundaries of the same action).

4) Change of Environment: a significant change in color/brightness of the scene or the dominant subject's surroundings.

Mark boundaries '1 level deeper' than the whole-clip event -- i.e., sub-events within the clip, not just the start/end of the clip itself. Do not report a boundary within the first or last

0.3s of the clip.

Answer ONLY as compact JSON with boundary timestamps in seconds (floats), in chronological order: {"boundaries": [<seconds>, ...]}. If there are no internal boundaries, return {"boundaries": []}.

## E.1.2 Candidate refinement

Here the model receives the boundary candidates of an automatic detector and may only move or add boundaries. The instruction is followed by a line beginning CANDIDATES (seconds): and then the clip frames.

You are given frames sampled at a fixed rate in   
chronological order from a short video clip. Each   
frame is preceded by its timestamp [time X.XXs]   
relative to the start of the clip.

Your task is Generic Event Boundary Detection   
(GEBD): find the moments where a human would   
naturally perceive the video splitting into   
meaningful temporal segments -- WITHOUT any   
predefined taxonomy. A boundary typically   
corresponds to (Shou et al., ICCV 2021):

1) Change of Subject; 2) Change of Object of   
Interaction;

```javascript
3) Change of Action (incl. sudden
direction/repetition change);
```

4) Change of Environment   
(color/brightness/surroundings).

An automatic detector has produced a candidate list of boundary timestamps for this clip (shown below as CANDIDATES).

Refine the list with EXACTLY three allowed operations:

\- ADD: scan the clip for any clear perceptual boundary NOT already covered by a candidate, and add its timestamp. If none are missing, add nothing.

\- SNAP: for each candidate, if the actual moment of change is slightly before/after it, move it to the exact frame of change (typically within +/-0.5s). Keep it near its original time.

\- DELETE: remove a candidate that does not mark a perceptual boundary at all -- the same subject, object of interaction, action and environment continue across it. Also remove a candidate that marks the SAME transition as another candidate you keep (a near-duplicate that is too close). Delete only when you are confident; when in doubt, keep the candidate.

Every candidate must still be represented in your output (either at its original time or snapped to a nearby time). Do not add or move a boundary into the first or last 0.3s of the clip.

Answer ONLY as compact JSON with the FULL boundary   
list = all candidates (snapped as needed) PLUS   
your additions, chronological order (seconds,   
floats): {"boundaries": [<seconds>, ...]}.

## E.2 EgoBoundary construction: LLM boundary verification

Each candidate boundary produced by the unsupervised discovery is verified in a separate request. The model sees a ±20 s window around the candidate at 1 fps (up to 41 frames), each frame preceded by a timestamp line [time H:MM:SS], and with a PROPOSED BOUNDARY text marker inserted at the candidate’s position. The decision is removal-only: accepted boundaries are a subset of the candidates and no new boundaries are created.

You are validating a proposed boundary in a long   
egocentric (head-mounted, fisheye) video. Below   
are frames in chronological order (1 fps), each   
tagged [time H:MM:SS]. A boundary is PROPOSED at   
[time 1:00:00] (marked '<<< PROPOSED BOUNDARY   
>>>').

Question: Is this a GENERIC EVENT BOUNDARY -- i.e. is there a meaningful overall change in the situation across this point (a change in activity, place, interaction, or context), as opposed to the same ongoing situation continuing? Accept if there is a clear overall change; reject if it is the same situation continuing with no real transition.

```jsonl
Answer ONLY as compact JSON: {"accept":
true|false, "confidence": <0..1>, "justification":
"<one or two sentences on what changes or stays
the same>"}
```

When another candidate boundary already falls inside the same window — that is, within [t−20, t) — a duplicate-suppression cue is appended to the first paragraph and a PREV BOUNDARY marker is inserted at that earlier position:

NOTE: another boundary was ALREADY placed earlier   
within this same window (marked '<<< PREV BOUNDARY   
>>>', at 0:59:48). If the change at the PROPOSED   
point is essentially the SAME transition already   
captured by that previous boundary (i.e. a   
near-duplicate that is too close), REJECT it.   
Accept only if it marks a genuinely distinct   
additional change.

## E.3 Human annotation instructions

Annotators judged each candidate boundary from a ±60 s video clip centered on it, played back with speed controls. A red BOUNDARY overlay marks the candidate for about 2.5 s as playback passes it; if another candidate lies earlier in the same window, an orange PREV BOUNDARY overlay marks it. Later candidates are deliberately not shown, so that clusters of nearby candidates are resolved left to right and exactly one survives per cluster.

The full text was shown before the task began, and the summary of Section E.3.2 remained on screen throughout.

![](images/112211dda421876c89b7f5a435cf8b2edc31ba4eb123123f629325d539ba7bdb.jpg)  
Figure 4: Qualitative examples of accepted event boundaries. The LVLM identifies meaningful transitions in location, activity, or social context, including moving outdoors, shifting attention to a group activity, and joining a social gathering. The proposed boundary is highlighted in red.

## E.3.1 Full instructions

Reviewing scene boundaries in video. This tool asks you to confirm whether the “moments where the scene changes” that were found automatically in first-person (head-mounted camera) daily-life video are in fact correct. No background in video or AI is needed; judge using only the criteria below. What are you judging? The candidate boundary is the middle of each clip — the moment the red BOUNDARY marker appears. Looking at that moment, ask whether the situation (the context) of the scene just before and the scene just after changes in a meaningful way.

Key question: “Across this point, does what the person is doing, or where they are, or who/what they are interacting with, or the overall situation, actually change?” If it changes, Accept (it is a boundary); if it simply continues, Reject (it is not).

It must be a lasting change, not a momentary one. Do not look only at one or two scenes on either side. Ask whether the overall situation itself changes at that point and whether the new situation then continues for a while. If the person briefly does something else and soon returns to what they were doing, that is the same situation continuing, and is not a boundary.

• Not a boundary: looking at a phone → briefly glancing up at the surroundings → looking at the phone again (the “looking at the phone” situation is still ongoing).

• A boundary: looking at a phone → putting the phone down, moving to another spot, and starting a conversation with people, which then continues (the situation changes and persists).

There is no exact time criterion. Judge by your own intuition, by whether it naturally feels like “the scene moved on here”.

Examples of a meaningful change.

• Change of activity: cooking → starting to eat.

• Change of place: being in the kitchen → moving to the living room.

• Change of interaction partner: looking at a phone alone → starting a conversation with another person.

![](images/80c4eb7f6ad344eeef856fc6f02b412e11baae8e779680ed88f3348ac3112a6d.jpg)  
Figure 5: Qualitative examples of rejected event boundaries. Despite changes in visual appearance or camera viewpoint, the underlying activity remains consistent, such as walking along the same path or continuing a social gathering. The proposed boundary is highlighted in red.

• Change of situation or context: working → starting to rest.

Conversely, if the person keeps doing the same thing, or the camera merely shakes for a moment, that is not a change: Reject.

When it is too similar to the preceding boundary. If an orange PREV BOUNDARY marker appears earlier in the clip, it means a boundary was already placed just before this one. If the boundary you are now looking at is essentially the same change as that preceding one (i.e. a duplicate), reject it as duplicate of the preceding boundary (key d). If it is a different, new change, accept it.

Controls. Space play/pause; ← → seek by 2 s; click the progress bar to seek. a Accept (is a boundary); d Reject — duplicate of the preceding boundary; x Reject — not a boundary (no change); r Reject — other; [ ] previous/next item. After each judgment the next clip loads automatically and progress is saved on the server, so the task can be closed and resumed.

## E.3.2 Summary shown on screen during the task

Is this point a generic event boundary? a Accept (is a boundary) d Reject — duplicate of the preceding boundary (too close) x Reject — not a boundary (no transition) r Reject — other

Criteria (summary). Across this point, does the situation (the context) itself change in a meaningful way, and does that change then last for a while?

• Accept — the activity, place, interaction, or situation actually changes and the new situation persists.

• Reject — the same situation continues, or the person briefly departs from it and soon returns (e.g. looking at a phone, glancing up, then back to the phone), or only the camera shakes.

• If PREV BOUNDARY is visible — if the current boundary is essentially the same change as the preceding one, reject it with d (duplicate); if it is a new change, accept it.

• There is no fixed time criterion. Judge by intuition, by whether it naturally feels like “the scene moved on here”.

## E.3.3 Correspondence with the LLM verification prompt

The human task of Section E.3 and the LLM prompt of Section E.2 ask the same question — whether activity, place, interaction, or context changes across the point — and both suppress nearduplicates using a marker for the preceding boundary only.

## F Qualitative Results

Figures 4 and 5 present qualitative examples of localized LVLM boundary judgments on EgoBoundary. Each example shows sampled frames surrounding a proposed boundary, along with the LVLM’s decision and explanation.

As illustrated, the LVLM accepts candidates associated with meaningful changes in event context, while rejecting incidental visual changes that do not indicate a transition between coherent events. These examples illustrate the role of semantic judgment in refining unsupervised boundary proposals.