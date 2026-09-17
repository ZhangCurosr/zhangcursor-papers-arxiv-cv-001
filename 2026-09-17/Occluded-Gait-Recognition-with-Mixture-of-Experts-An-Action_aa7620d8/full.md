# Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective

Panjian Huang<sup>1,3⋆</sup> , Yunjie Peng<sup>2,4⋆</sup> , Saihui Hou<sup>1,3†</sup> , Chunshui Cao<sup>3</sup> , Xu Liu<sup>3</sup> , Zhiqiang He<sup>2,4</sup> , and Yongzhen Huang<sup>1,3†</sup>

1 School of Artificial Intelligence, Beijing Normal University School of Computer Science and Technology, Beihang University <sup>3</sup> WATRIX.AI 4 AI Lab, Lenovo Research

Abstract. Extensive occlusions in real-world scenarios pose challenges to gait recognition due to missing and noisy information, as well as body misalignment in position and scale. We argue that rich dynamic contextual information within a gait sequence inherently possesses occlusionsolving traits: 1) Adjacent frames with gait continuity allow holistic body regions to infer occluded body regions; 2) Gait cycles allow information integration between holistic actions and occluded actions. Therefore, we introduce an action detection perspective where a gait sequence is regarded as a composition of actions. To detect accurate actions under complex occlusion scenarios, we propose an Action Detection Based Mixture of Experts (GaitMoE), consisting of Mixture of Temporal Experts (MTE) and Mixture of Action Experts (MAE). MTE adaptively constructs action anchors by temporal experts and MAE adaptively constructs action proposals from action anchors by action experts. Especially, action detection as a proxy task with gait recognition is an endto-end joint training only with ID labels. In addition, due to the lack of a unified occluded benchmark, we construct a pioneering Occluded Gait database (OccGait), containing rich occlusion scenarios and annotations of occlusion types. Extensive experiments on OccGait, OccCASIA-B, Gait3D and GREW demonstrate the superior performance of GaitMoE. OccGait is available at https://github.com/BNU-IVC/OccGait.

Keywords: Occluded Gait Recognition · Dynamic Contextual Information · Action Detection · Mixture of Experts

## 1 Introduction

Gait recognition has attracted increasing attention and gained broad applications in crime prevention, forensic identification, and social security [44] due to its ability to accurately identify walking patterns of pedestrians from a distance in complex surveillance scenarios, e.g., viewing angles, cloth-changing and illumination conditions [40]. Applying upstream tasks such as tracking, segmentation, size normalization, and alignment to preprocess raw videos, the obtained gait representations (e.g., silhouettes or skeletons) make existing methods [3, 4, 10, 12, 23–25, 31, 32, 34, 52] achieve accurate identification. However, current studies overlook occlusions largely existing in practical scenarios, e.g., occluded by carrying, obstacles, the crowd, or moving out of camera view. As shown in Fig. 1(a), body regions occluded by obstacles or the crowd lead to missing and noisy information, while partial visual body regions with size normalization cause position and scale misalignment, which significantly degrading fine-grained feature matching [56,58]. Direct solutions, e.g., simply discarding or persisting occluded frames, do not adequately address occlusion issues since partially visual body regions in occluded frames may still contain key discriminative regions while allowing them to persist will cause erroneous feature extraction and matching. Therefore, occlusion issues have become one of the biggest bottlenecks in gait recognition.

![](images/12123713f76c5c44eeceed8dbb61e022ce7a72c00a42ef86803db9159c26001b.jpg)  
(b) Dynamic Contextual Information (Ours)  
Fig. 1: (a) The main four occlusion issues in gait recognition. (b) The dynamic contextual information within a gait sequence can infer and integrate occlusion information.

To address occlusion issues in gait recognition, we rethink the dynamic contextual information within a gait sequence: (i) Gait Continuity. Adjacent frames with continual motion enable body regions in holistic frames to infer the same body regions in occluded frames. As shown in Fig. 1(b), for the current frame missing lower body information, the preceding and subsequent holistic frames can still infer approximate motion in the occluded regions. (ii) Gait Cycle. As shown in Fig. 1(b), we regard the current misaligned frame and adjacent frames as an action. Combined with the gait cycle, i.e., a gait sequence is formed by the repetition of a series of actions, to discover actions with similar cues, holistic and occluded actions can be integrated into a robust action.

Driven by the above analysis, we introduce a new perspective for occluded gait recognition, “Action Detection”. The paradigm of action detection aims to predict the action boundaries and categories from an untrimmed video, where predefined consecutive frames as action anchors represent potential actions and further action anchors with high actionness scores generate action proposals [33, 43,59]. Specific to gait recognition, we associate action anchors to represent adjacent frames with gait continuity and further consider the gait cycle to construct action proposals from similar action anchors. Therefore, a gait sequence can be regarded as a composition of actions.

![](images/aa3f815d8a7d747d618e792d07de61451b483cacfcc55b5bb779c1a42fbcfeb4.jpg)  
Fig. 2: Action composition. Each temporal expert focuses on one body region with individual temporal size, constructing action anchors. Each action expert integrates similar action anchors from diferent gait cycles, constructing one action proposal.

Considering a single model that struggles to capture holistic and diverse actions under complex occlusion scenarios, e.g., the uncertainty of occluded body regions and duration, we propose an Action Detection Based Mixture of Experts (GaitMoE). Mixture of Experts (MoE) follows a divide-and-conquer philosophy, breaking down complex problems into simple sub-problems. Each sub-problem is handled by a dedicated expert, contributing collectively to solve the overall complexity. As shown in Fig. 2, each temporal expert focuses on the corresponding body region and temporal granularity, sliding at the entire gait sequence to construct action anchors. Subsequently, each action expert integrates similar action anchors with one action type, constructing action proposals. Finally, instead of localization and classification in action detection, action proposals are collectively as discriminative features for identification.

Additionally, the absence of publicly available gait databases with quantifiable occlusion metrics poses an extreme challenge for occluded gait recognition. To this end, we establish a pioneering Occluded Gait database (OccGait) with two characteristics: (i) Diverse Occlusion Scenarios. Each subject has 4 diferent types of occlusion situations, including None of Occlusion, Carrying Occlusion, Crowd Occlusion, and Static Occlusion. (ii) Explicit Occlusion Types. OccGait provides explicit occlusion types for each gait sequence, which enables to qualify and quantify occlusion issues.

Our main contributions can be summarized as follows:

– To address occlusion challenges, we introduce an action detection perspective where an Action Detection Based Mixture of Experts (GaitMoE) structures a gait sequence as a composition of action.

– To qualify and quantify occlusion issues, we build a novel Occluded Gait recognition benchmark (OccGait), including diverse occlusion scenarios and explicit annotations of occlusion types.

– To evaluate efectiveness and robustness, extensive experimental results on OccGait, OccCASIA-B, Gait3D, and GREW demonstrate that our method significantly outperforms other state-of-the-art methods.

## 2 Related Work

## 2.1 Gait Recognition

Gait Recognition is mainly categorized into appearance-based and model-based approaches. Appearance-based methods [3, 4, 10, 12, 23–25, 31, 32, 34, 52] usually uses templates of compressing a sequence of gait silhouettes (e.g., Gait Energy Image), set of frames and sequence of frames as inputs, extracting finegrained features (e.g., spatial-temporal and part-level representations). Modelbased methods [14,47,48,61–63] explicitly model human body structure, e.g., 2D or 3D skeletons and meshes. Additionally, some researches [2, 30, 42] take other data types as inputs, such as RGB frames, optical flow and point clouds. However, most of these methods usually neglect the fact that real-world scenarios introduce a significant amount of occlusion.

## 2.2 Occluded Gait Recognition

We introduce occluded gait recognition from two aspects: (i) DataBases. For synthesis-based databases, Chen et al. [6] simulate occluded scenarios based on CMU Mobo [15] through adding horizontal or vertical black bars. Uddin et al. [50] synthesize relative static and dynamic occlusions based on OUMVLP [45] by a background rectangle mask in a fixed position or gradually changed position. Delgado-Escano et al. [9] generate crowd occlusions based on CASIA-B [60] and TUM-GAID [20] by augmenting persons in raw videos. Xu et al. [56, 58] synthesize occluded scenarios by simulating cropping and size-normalized silhouettes. For real-world databases, Hofmann et al. [21] collect TUM-IITKGP, including static occlusions (e.g., standing people, a backpack, gown and hands in pocket), dynamic occlusions (e.g., two walking people). Chattopadhyay et al. [5] construct a frontal and occluded gait database by estimating Kinect depth. Li et al. [29] present an OG RGB+D dataset captured by Azure Kinect DK sensor, containing occlusions with carrying, clothing and the crowd. However, these databases have some limitations, e.g., the single occlusion scenario, small occlusion regions, or not publicly available yet. (ii) Architectures. For reconstruction-based approaches, Xu et al. [58] re-normalize and register silhouettes from learned holistic information (e.g., scales) before the following feature extraction and matching process. Xu et al. [56] estimate SMPL with pose and shape features from RGB occluded videos. Uddin et al. [50] reconstruct a sequence from the occluded sequence by a conditional deep generative adversarial network. Peng et al. [37] register and recover occluded silhouettes with a selfsupervised alignment module and temporal recovery transformer. Guo et al. [16] propose a Physics-Augmented Autoencoder (PAA) that generates physically intermediate representations through a graph-convolution-based encoder and a physics-based decoder, which enhances the ability to reconstruct input skeleton sequences even when partially occluded. For reconstruction-free approaches, Gupta et al. [17] propose a occlusion-aware module by synthetic occlusions to detect occlusion type information to guide gait recognition training. Zhu et al. [64] use SMPLify-X that provides the body shape feature decoupled from its pose and strong prior, which enables to generate the complete shape even with mild occlusions.

![](images/5ed7881c5bfaa79414f3c4509f037dbd7fe0260afe16b6b059472f053c6b35a5.jpg)  
Fig. 3: The overview of GaitMoE. Best viewed in colors, the input sequence is firstly extracted to frame-level features by C2D Block (e.g., 2D CNNs or a residual block), and Mixture of Temporal Experts (MTE) dispatch temporal experts (TE) with diferent temporal bounding boxes to the corresponding channel segments, forming action anchors. After horizontal partitioning (HP), Mixture of Action Experts (MAE) is independent for each part-level feature. For each channel segment, one action expert (AE) integrates similar action anchors along the temporal dimension by weighted sum operation, forming one action proposal. Finally, The concatenated action proposals as part features for identification.

## 2.3 Mixture of Experts

Mixture of Experts (MoE) is a sparsely-activated architecture where a router network output weights for aggregating multiple experts [41]. The philosophy of divide and conquer allows MoE to reduce computational cost and increase model capacity, and it has been widely extended to Vision Transformer [28,38,55]. Each expert in MoE is dispatched with one sub-data (e.g., image patches, data from one domain) and maintains specialization, which is named “Expert”. This work extends MoE to detect fine-grained actions in occluded gait recognition and makes each expert concentrate on one representative action.

## 3 Methodology

GaitMoE mainly consists of Mixture of Temporal Experts (MTE) and Mixture of Action Experts (MAE). We give a brief overview of the full process in Fig. 3.

## 3.1 Mixture of Temporal Experts

Although a gait sequence has filtered texture information and performed coarsealigned registration, the complex occlusions in real scenarios cause the uncertainty of occluded body regions and duration. To this end, we adopt multi-scale mechanisms in temporal and channel dimensions to extract fine-grained features. Action Anchors. As we know, a gait sequence possesses continuity with significant mutual information between each frame and adjacent frames. For example, when we observe a person lifting their leg, it is likely to be followed by a swinging leg. Some existing approaches employ temporal modeling to capture such relationships, $e . g .$ , 3D CNNs and LSTMs. However, uncertain and complex occlusions interfere with the dynamic information. To alleviate these issues, Fig. 4(a) shows that MTE predefines various sizes of temporal experts, which are dilated convolutions with diferent dilated ratios for corresponding channel segments. At each temporal position, each temporal expert independently constructs action anchors from adjacent temporal positions, which is why we name it “Temporal Expert”. Considering adjacent occluded frames may introduce noise information to current holistic frames, MTE preserves partial channel segments for adaptively selecting clean regions. Let $\bar { \mathcal { X } } \in \mathbb { R } ^ { \bar { c } \times \mathcal { T } \times \mathbf { \bar { \mathcal { H } } } \times \mathcal { W } }$ denote frame-level features extracted from silhouettes by C2D Block, where C, T, H, W represent channel, consecutive T frames, height and width dimensions. The process of MTE is formulated as follows:

![](images/6037f957099eeedd47afd979a8fa456744c4f8b403905db50626a065ff7ad1c6.jpg)  
Fig. 4: (a) Mixture of Temporal Experts. Dila Conv (DC) represents Dilated Convolution, predefining action anchors with diferent dilated ratios. (b) Mixture of Action Experts. LP denotes the linear projection. Similar action anchors adaptively integrate into action proposals.

$$
\mathcal { V } = C o n c a t ( D C _ { i } ( \mathcal { X } _ { i } ) , i = 1 , 2 , \cdots , \mathcal { K } , \mathcal { X } _ { [ S - K + 1 , S ] } )\tag{1}
$$

where $\mathcal { X } _ { i } \in \mathbb { R } ^ { \frac { c } { s } \times T \times \mathcal { H } \times \mathcal { W } } , \mathcal { y } \in \mathbb { R } ^ { \mathcal { C } \times T \times \mathcal { H } \times \mathcal { W } }$ , S is the number of channel segments, $\kappa$ is the number of temporal experts, DC is dilated convolution, and i is the segment index. In our work, we set $\mathcal { S } = 8 , \mathcal { K } = 4$ , and $D C _ { i }$ is 3D CNN with kernel size (3, 1, 1), stride (1, 1, 1), padding $( i , 0 , 0 )$ , and dilated ratio i. In addition, residual learning is embedded within MTE for easing training.

## 3.2 Mixture of Action Experts

Although action anchors contain rich dynamic information, they may have a large amount of redundant and invalid actions. Therefore, we adopt prototypebased architecture to adaptively select and integrate discriminative and similar action anchors as action proposals. Since a gait sequence can be regarded as the composition of actions, when the sequence is occluded resulting in many invalid actions, the action prototype needs to detect the most discriminative action type and aggregate this action type from occluded and holistic actions. In addition, GaitMoE adopts horizontal pooling (HP) for fine-grained part features $\mathcal { P } \in \mathbb { R } ^ { \mathcal { C } \times \mathcal { T } }$ , and MAE is independent for each part. Here, we omit the part index for simplicity.

Action Proposals. To capture discriminative actions only with ID labels, MAE shown in Fig. 4(b) predefines a set of learnable action prototypes where an action prototype adaptively learns a type of action for recognition, that is why we name it “Action Expert”. For fine-grained action extraction, we dispatch each action expert to the corresponding channel segment (e.g., action anchors), and action experts “watch” contextual action anchors in the entire temporal dimension for filtering action anchors with occlusions, and select and integrate similar action anchors as action proposals. Diferent to MoEs [13, 27, 36, 41] where the routers generally adopt Top-K selection and sparsely memorize information, recent MoE works has shown remarkable performance with a fixed hash router [39], or convolutional experts [7]. In this work, we introduce a soft selection for balancing the training. Let $\mathcal { A } \in \mathbb { R } ^ { \frac { c } { \mathcal { M } } \times \mathcal { M } }$ represent M action experts with $\frac { c } { \mathcal { M } }$ dimension, and we obtain action queries Q by identify mapping on ${ \mathcal { A } } ,$ action keys K and values V by diferent linear projections on $\mathcal { P } .$ Then, we dispatch each action query to the corresponding channel segment of $\kappa$ to evaluate action anchors by scores where the higher the score, the more reliable the action anchor, and vice versa. To make one action expert concentrate on one most representative action, we use the softmax function to calculate the scores within the corresponding channel segment of K and weighted sum operation with the corresponding channel segment of V as an action proposal. The formulation is as follows:

$$
\mathcal { Q } _ { i } = \mathcal { A } _ { i } , \quad \mathcal { K } _ { i } = \mathcal { P } _ { i } \mathcal { W } ^ { \mathcal { K } } , \quad \mathcal { V } _ { i } = \mathcal { P } _ { i } \mathcal { W } ^ { \mathcal { V } }\tag{2}
$$

$$
\mathcal { F } _ { i } = \sum _ { j = 1 } ^ { T } \mathcal { O } _ { i , j } \otimes \mathcal { V } _ { i , j } , \quad \mathcal { O } _ { i , j } = \frac { e x p ( \mathcal { Q } _ { i , j } \mathcal { K } _ { i , j } ^ { T } ) } { \sum _ { j = 1 } ^ { T } e x p ( \mathcal { Q } _ { i , j } \mathcal { K } _ { i , j } ^ { T } ) }\tag{3}
$$

where i is the channel segment index, $i \in { 1 , 2 , \cdots , \mathcal { M } , \ : j }$ is the action anchor index along the temporal dimension of the i segment, $j \in { 1 , 2 , \cdot \cdot \cdot , \mathcal { T } , \mathcal { W } ^ { \kappa } , \mathcal { W } ^ { \nu } \in }$ R $\begin{array} { r } { \frac { c } { \mathcal { M } } \times \frac { c } { \mathcal { M } } , \ : \breve { \mathcal { P } _ { i } } \in \mathbb { R } ^ { \mathcal { T } \times \frac { \hat { c } } { \mathcal { M } } } . \ : \mathcal { Q } _ { i , j } , \ : \mathcal { K } _ { i , j } , \ : \mathcal { V } _ { i , j } \in \mathbb { R } ^ { 1 \times \frac { c } { \mathcal { M } } } } \end{array}$ . Finally, we obtain $\mathcal { F }$ as one part feature by concatenating action proposals $[ \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } , \cdots , \mathcal { F } _ { M } ]$ along the channel dimension. $\mathcal { F }$ is fed into Separate FCs and BNNeck for identification.

## 3.3 Joint Loss

GaitMoE is an end-to-end joint learning framework only with ID labels, introducing action detection as a proxy task with gait recognition. The joint loss includes two types: Triplet Loss [19] $\mathcal { L } _ { t p }$ and Cross Entropy Loss $\mathcal { L } _ { c e }$ , which constrains each part independently. This formulation is as follows:

$$
\mathcal { L } = \mathcal { L } _ { t p } + \beta \mathcal { L } _ { c e }\tag{4}
$$

![](images/381517e272088dd4b4b10e61b8b2d32b8ad713f9d04605ee01d8656139713cc0.jpg)  
Fig. 5: (a) The layout and camera views during collection. (b) The 3 types of occlusion scenarios where the number ○1 ○2 are for Carrying Occlusion, ○3 ○4 are for Crowd Occlusion and ○5 ○6 are for Static Occlusion, and None Occlusion is omitted for simplicity.

where the hyper-parameter $\beta$ is for balancing the two terms.

## 4 The OccGait Benchmark

Due to the absence of a comprehensive gait database for quantitatively and qualitatively analyzing the impact of various occlusion types, we collect the Occluded Gait Database (OccGait), including 101 subjects with 4 types of occluded scenarios, 8 camera views, and over 80k sequences. It is worth noting that we hope the OccGait can serve as a starting point to promote robust gait recognition for practical applications, similar to the transition from the indoor databases, e.g., CASIA-X Series [44, 46, 53, 60] and OU-X Series [1, 22, 26, 35, 45, 49, 51, 57] to the wild databases, e.g., Gait3D [62] and GREW [65].

## 4.1 Data Collection and Pre-processing

The OccGait is collected in an indoor gait recognition laboratory. During Occ-Gait collection, we obtain authorization from all subjects who are informed for academic data collection in advance. Privacy is also the highest priority in our research. As the left in Fig. 5(a), we place 3 cameras (Cam1 of 0<sup>◦</sup>, Cam2 of 45<sup>◦</sup>, Cam3 of 315<sup>◦</sup>) with 1920 × 1080 resolution in the square area. During the data collection process, the subjects follow this walk route (i.e., 1-2-3-4). We filter out data with overlapping camera views caused by the combination of 3 cameras and walking directions, and obtain gait sequences with 8 camera views on the right in Fig. 5(a). To qualify and quantify realistic and complex occluded scenarios, we set 4 types of occluded situations with diverse occlusions shown in Fig. 5(b), which are None of Occlusion (i.e., Normal Walking as NM), Carrying Occlusion (CA, ○1 ○2 ), Crowd Occlusion (CR, ○3 ○4 ), and Static Occlusion (ST, ○5 ○6 ).

All subjects walk the route four times for NM and two times for CA, CR and ST, respectively, which denotes NM01, NM02, NM03, NM04, CA01, CA02, CR01, CR02, ST01 and ST02.

None of Occlusion. To simulate walking status in real scenarios, subjects in their clothing, walk with their walking speed. As shown in Fig. 5(a), there are no obstacles in this situation, and the full body of each subject is fully visible.

Carrying Occlusion. As shown in Fig. 5(b)(○1 ○2 ), we establish two common carrying scenarios for daily life: umbrellas and luggage. People with an umbrella on a rainy day partially obstruct the upper of the body, while luggage occludes both the torso and the lower of the body.

Crowd Occlusion. Gait recognition often requires retrieval in crowded scenes, and human body occlusion can lead to significant interference, such as occlusion of body edges and incorrect dynamic information, as shown in Fig. 5(b)(○3 ○4 ). We design two types of crowd occlusion, the subject walking with another person in diferent directions (opposite and parallel), which results in partial occlusion of gait sequences at certain moments and complete occlusion of gait sequences at all times.

Static Occlusion. Gait recognition will be deployed in a wide range of scenarios, such as in squares, malls, and other locations with numerous static obstructions. Fig. 5(b)(○5 ) demonstrates our placement of plants and chairs in the walking route. Fig. 5(b)(○6 ) shows that the complex and irregular static obstructions hinder the lower body region.

Data Pre-processing. We adopt MaskFormer [8], a segmentation algorithm pre-trained on large datasets (including numerous occluded scenarios), to extract silhouettes from the original RGB data as input.

## 4.2 Evaluation Protocol

OccGait is divided into training and testing sets. The 51 individuals with odd numbers as the training set while the remaining 50 with even numbers as the test set. Our gait evaluation follows the protocols of the previous gait evaluation [60]. Given a query sequence, we measure its distance to each sequence in the gallery, retrieving the subject with the closest distance from the gallery. To quantify occluded gait analysis, we use the NM01 and NM02 of each subject in the testing set as the gallery, evaluating their Rank 1 performance under diferent occlusions and viewing angles.

## 5 Experiments

## 5.1 Datasets

We first conduct extensive qualitative and quantitative occlusion analyses on OccGait and OccCASIA-B [37]. Subsequently, we further validate the generalizability and practicality of our method on Gait3D [62] and GREW [65].

OccGait is for real-scenario occlusion evaluation built by this work. The details have been discussed in Section 4.

OccCASIA-B is a synthetic occluded gait database [37] from CASIA-B and has similar basic statistics, containing 124 subjects, 3 diferent walking situations, etc., Walking in Normal (NM), Walking with a Bag (BG) and Walking with Diferent Clothes (CL), 11 camera views from uniform interval of 18<sup>◦</sup> in [0<sup>◦</sup>, 180<sup>◦</sup>]. To simulate occlusion situations, OccCASIA-B sets 4 types of occlusions: None Occlusion (NO), Crowd Occlusion (CO), Static Occlusion (SO) and

Table 1: The Rank-1 accuracy (%) on OccGait for diferent probe views excluding the identical-view cases. For evaluation, the sequences of NM01 and NM02 for each subject are taken as the gallery. The benchmark adopts None Occlusion (NO), Carrying Occlusion (CA), Crowd Occlusion (CR) and Static Occlusion (ST).
<table><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=9>Probe View</td><td rowspan=2 colspan=1>Average</td></tr><tr><td rowspan=1 colspan=2>0°</td><td rowspan=1 colspan=1>45°</td><td rowspan=1 colspan=1>90°</td><td rowspan=1 colspan=1>135°</td><td rowspan=1 colspan=1>180°</td><td rowspan=1 colspan=1>225°</td><td rowspan=1 colspan=1>270°</td><td rowspan=1 colspan=1>315°</td></tr><tr><td rowspan=5 colspan=1>NM</td><td rowspan=1 colspan=1>GaitSet [4]</td><td rowspan=1 colspan=2>65.7</td><td rowspan=1 colspan=1>91.7</td><td rowspan=1 colspan=1>89.1</td><td rowspan=1 colspan=1>90.7</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>91.9</td><td rowspan=1 colspan=1>89.4</td><td rowspan=1 colspan=1>92.1</td><td rowspan=1 colspan=1>84.7</td></tr><tr><td rowspan=1 colspan=1>GaitPart [12]</td><td rowspan=1 colspan=2>62.9</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>89.6</td><td rowspan=1 colspan=1>59.6</td><td rowspan=1 colspan=1>89.9</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>90.7</td><td rowspan=1 colspan=1>82.6</td></tr><tr><td rowspan=3 colspan=1>GaitGL [32]STOR [37]GaitBase [11]</td><td rowspan=2 colspan=2>73.973.7</td><td rowspan=1 colspan=1>94.3</td><td rowspan=1 colspan=1>92.6</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>68.1</td><td rowspan=1 colspan=1>93.1</td><td rowspan=1 colspan=1>91.7</td><td rowspan=1 colspan=1>92.6</td><td rowspan=1 colspan=1>87.5</td></tr><tr><td rowspan=1 colspan=1>94.6</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>73.3</td><td rowspan=1 colspan=1>94.1</td><td rowspan=1 colspan=1>92.6</td><td rowspan=1 colspan=1>92.6</td><td rowspan=1 colspan=1>88.3</td></tr><tr><td rowspan=1 colspan=2>68.4</td><td rowspan=1 colspan=1>91.6</td><td rowspan=1 colspan=1>88.7</td><td rowspan=1 colspan=1>91.1</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=1>88.1</td><td rowspan=1 colspan=1>91.7</td><td rowspan=1 colspan=1>86.0</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GaitMoE(ours)</td><td rowspan=1 colspan=2>81.0</td><td rowspan=1 colspan=1>96.0</td><td rowspan=1 colspan=1>94.0</td><td rowspan=1 colspan=1>95.1</td><td rowspan=1 colspan=1>81.3</td><td rowspan=1 colspan=1>94.7</td><td rowspan=1 colspan=1>94.1</td><td rowspan=1 colspan=1>94.7</td><td rowspan=1 colspan=1>91.4</td></tr><tr><td rowspan=5 colspan=1>CA</td><td rowspan=5 colspan=1>GaitSet [4]GaitPart [12]GaitGL [32]STOR [37]GaitBase [11]</td><td rowspan=2 colspan=2>50.142.0</td><td rowspan=2 colspan=1>74.369.9</td><td rowspan=1 colspan=1>79.7</td><td rowspan=1 colspan=1>79.1</td><td rowspan=1 colspan=1>47.4</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>69.3</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>78.6</td><td rowspan=1 colspan=1>37.6</td><td rowspan=1 colspan=1>66.4</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>63.7</td><td rowspan=1 colspan=1>62.4</td></tr><tr><td rowspan=2 colspan=2>48.655.9</td><td rowspan=1 colspan=1>.6</td><td rowspan=1 colspan=1>79.7</td><td rowspan=1 colspan=1>84.6</td><td rowspan=1 colspan=1>87.9</td><td rowspan=1 colspan=1>38.7</td><td rowspan=1 colspan=1>76.7</td><td rowspan=1 colspan=1>78.1</td><td rowspan=1 colspan=1>70.4</td><td rowspan=1 colspan=1>70.6</td></tr><tr><td rowspan=1 colspan=1>83.6</td><td rowspan=1 colspan=1>83.6</td><td rowspan=1 colspan=1>86.3</td><td rowspan=1 colspan=1>54.4</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>83.1</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>76.3</td></tr><tr><td rowspan=1 colspan=2>58.1</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>84.3</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>79.9</td><td rowspan=1 colspan=1>80.1</td><td rowspan=1 colspan=1>79.3</td><td rowspan=1 colspan=1>75.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GaitMoE(ours)</td><td rowspan=1 colspan=2>68.3</td><td rowspan=1 colspan=1>87.7</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>89.6</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>86.6</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>82.1</td></tr><tr><td rowspan=5 colspan=1>CR</td><td rowspan=5 colspan=1>GaitSet [4]GaitPart [12]GaitGL [32]STOR [37]GaitBase [11]</td><td rowspan=1 colspan=2>58.3</td><td rowspan=1 colspan=1>84.7</td><td rowspan=1 colspan=1>80.1</td><td rowspan=1 colspan=1>77.4</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>77.9</td><td rowspan=1 colspan=1>77.6</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>74.2</td></tr><tr><td rowspan=1 colspan=2>48.1</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=1>76.4</td><td rowspan=1 colspan=1>69.6</td><td rowspan=1 colspan=1>39.0</td><td rowspan=1 colspan=1>67.6</td><td rowspan=1 colspan=1>68.1</td><td rowspan=1 colspan=1>79.3</td><td rowspan=1 colspan=1>66.3</td></tr><tr><td rowspan=2 colspan=2>47.455.6</td><td rowspan=1 colspan=1>89.0</td><td rowspan=1 colspan=1>81.3</td><td rowspan=1 colspan=1>77.1</td><td rowspan=1 colspan=1>41.0</td><td rowspan=1 colspan=1>75.0</td><td rowspan=1 colspan=1>77.1</td><td rowspan=1 colspan=1>87.6</td><td rowspan=1 colspan=1>71.9</td></tr><tr><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>88.4</td><td rowspan=2 colspan=1>77.178.5</td></tr><tr><td rowspan=1 colspan=2>62.4</td><td rowspan=1 colspan=1>86.9</td><td rowspan=1 colspan=1>83.1</td><td rowspan=1 colspan=1>82.1</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>85.4</td><td rowspan=1 colspan=1>80.7</td><td rowspan=1 colspan=1>87.6</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GaitMoE(ours)</td><td rowspan=1 colspan=2>63.1</td><td rowspan=1 colspan=1>90.6</td><td rowspan=1 colspan=1>86.6</td><td rowspan=1 colspan=1>84.4</td><td rowspan=1 colspan=1>57.6</td><td rowspan=1 colspan=1>81.6</td><td rowspan=1 colspan=1>84.9</td><td rowspan=1 colspan=1>90.3</td><td rowspan=1 colspan=1>79.9</td></tr><tr><td rowspan=6 colspan=1>ST</td><td rowspan=5 colspan=1>GaitSet [4]GaitPart [12]GaitGL [32]STOR [37]GaitBase [11]</td><td rowspan=1 colspan=2>54.1</td><td rowspan=1 colspan=1>86.3</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>54.1</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>83.6</td><td rowspan=2 colspan=1>74.271.9</td></tr><tr><td rowspan=1 colspan=2>44.6</td><td rowspan=1 colspan=1>.6</td><td rowspan=1 colspan=1>83.7</td><td rowspan=1 colspan=1>85.7</td><td rowspan=1 colspan=1>77.3</td><td rowspan=1 colspan=1>39.1</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>84.1</td><td rowspan=1 colspan=1>77.4</td></tr><tr><td rowspan=1 colspan=2>36.7</td><td rowspan=1 colspan=1>87.7</td><td rowspan=1 colspan=1>90.7</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>37.6</td><td rowspan=1 colspan=1>86.6</td><td rowspan=1 colspan=1>90.4</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>74.0</td></tr><tr><td rowspan=1 colspan=2>50.3</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>91.3</td><td rowspan=1 colspan=1>87.4</td><td rowspan=1 colspan=1>54.9</td><td rowspan=1 colspan=1>90.1</td><td rowspan=1 colspan=1>91.3</td><td rowspan=1 colspan=1>89.1</td><td rowspan=1 colspan=1>80.6</td></tr><tr><td rowspan=1 colspan=2>57.7</td><td rowspan=1 colspan=1>89.9</td><td rowspan=1 colspan=1>87.4</td><td rowspan=1 colspan=1>85.9</td><td rowspan=1 colspan=1>59.9</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>87.4</td><td rowspan=1 colspan=1>85.4</td><td rowspan=1 colspan=1>80.3</td></tr><tr><td rowspan=1 colspan=1>GaitMoE(ours)</td><td rowspan=1 colspan=2>62.9</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>92.1</td><td rowspan=1 colspan=1>89.4</td><td rowspan=1 colspan=1>63.6</td><td rowspan=1 colspan=1>91.1</td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=1>91.4</td><td rowspan=1 colspan=1>84.7</td></tr></table>

Detect Occlusion (DO), which denotes a walking person without occlusions, occluded by another one in a crowded area, occluded by static occlusions, e.g., benches, bicycles and fire hydrants, and losing body regions in the up, down, left, or right direction. The OccCASIA-B takes the first 74 subjects as the training set where each gait sequence with 0.6 of occlusion probability generates one of the 4 types of occluded scenarios. The remaining 50 subjects are used for occlusion evaluation. For each occlusion benchmark, except for the first 4 NM sequences used as the holistic gallery set, the remaining sequences are generated with the corresponding occlusion scenario.

Gait3D samples two segments of continuous two-hour video clips from each of seven-day raw videos in a supermarket, including complex covariates (e.g., occlusions, view angles) for practical gait recognition. It contains 3000 subjects with 25309 sequences, taking 2000 subjects as the training dataset and 1000 subjects as the testing dataset.

GREW is a large-scale wild gait database, containing 26345 subjects with 128671 sequences captured by 882 cameras. It provides 4 types of silhouettes, optical flow, and 2D/3D human poses. The benchmark takes 20000 subjects as the training dataset and 6000 subjects as the testing dataset, and each subject provides two sequences for the gallery set and two sequences for the probe set.

Table 2: The Rank-1 accuracy (%) on OccCAISA-B across diferent views, excluding the identical-view cases. The NO, CO, SO, and DO denote the testing sets of Non-Occlusion, Crowd Occlusion, Static Occlusion, and Detection Occlusion accordingly. Based on the walking condition, probe sequences are grouped into Normal Walking (NM), Carrying Bags (BG), and Cloth-changing Condition (CL).
<table><tr><td rowspan="3">Methods</td><td colspan="4"></td><td colspan="4">CO</td><td colspan="4">SO</td><td colspan="4">DO</td></tr><tr><td>NM</td><td>BG</td><td>CL</td><td>Mean</td><td>NM</td><td>BG</td><td>CL</td><td>Mean</td><td>NM</td><td>BG</td><td>CL</td><td>Mean</td><td>NM</td><td>BG</td><td>CL</td><td>Mean</td></tr><tr><td>GaitSet [4]</td><td>92.4</td><td>83.0</td><td>65.2</td><td>80.2</td><td></td><td>80.7 69.6</td><td>50.4</td><td>66.9</td><td>86.0</td><td>76.6</td><td>57.8</td><td>73.5</td><td>85.7</td><td>72.7</td><td>52.7</td><td>70.4</td></tr><tr><td>GaitPart [12]</td><td></td><td>92.3 84.9</td><td>68.9</td><td>82.0</td><td></td><td>80.2 70.6</td><td>53.3</td><td>68.1</td><td>85.0</td><td>77.3</td><td>60.5</td><td>74.3</td><td>80.7</td><td>68.4</td><td>52.4</td><td>67.2</td></tr><tr><td>GaitGL [32]</td><td></td><td>94.5 89.2 75.3</td><td></td><td>86.4</td><td></td><td>84.4 74.8</td><td>57.7</td><td>72.3</td><td>87.4</td><td>81.8</td><td>66.9</td><td>78.7</td><td>86.2</td><td>76.2</td><td>61.5</td><td>74.6</td></tr><tr><td>STOR [37]</td><td></td><td>95.9 90.9</td><td>77.1</td><td>88.0</td><td></td><td>88.8</td><td>80.7 64.3</td><td>77.9</td><td>91.2</td><td>85.6</td><td>69.9</td><td>82.3</td><td>93.2</td><td>86.3</td><td>70.1</td><td>83.2</td></tr><tr><td>GaitBase [11]</td><td></td><td>94.4 88.5</td><td>68.7</td><td>83.9</td><td></td><td>87.6</td><td>78.0 57.5</td><td>74.4</td><td>90.1</td><td>82.7</td><td>62.0</td><td>78.3</td><td>89.9</td><td>80.4</td><td>58.5</td><td>76.3</td></tr><tr><td>GaitMoE(ours)</td><td></td><td>96.2 91.5</td><td>80.7</td><td></td><td>89.5</td><td>90.0</td><td>83.3 68.3</td><td>80.5</td><td>91.9</td><td>86.0</td><td>73.9</td><td>83.9</td><td>93.4 87.3</td><td></td><td>75.3</td><td>85.3</td></tr></table>

Table 3: Comparisons on Gait3D and GREW.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="2">Gait3D</td><td colspan="2">GREW</td></tr><tr><td>Rank-1</td><td>mAP</td><td>Rank-1</td><td>Rank-5</td></tr><tr><td>GaitSet [4]</td><td>AAAI19</td><td>36.7</td><td>30.0</td><td>46.3</td><td>63.6</td></tr><tr><td>GaitPart [12]</td><td>CVPR20</td><td>28.2</td><td>47.6</td><td>44.0</td><td>60.7</td></tr><tr><td>GaitGL [32]</td><td>ICCV21</td><td>29.7</td><td>22.3</td><td>47.3</td><td>63.6</td></tr><tr><td>SMPLGait [62]</td><td>CVPR22</td><td>46.3</td><td>37.2</td><td></td><td></td></tr><tr><td>MTSGait [62]</td><td>MM22</td><td>48.7</td><td>37.6</td><td>55.3</td><td>71.3</td></tr><tr><td>GaitBase [11]</td><td>CVPR23</td><td>64.6</td><td></td><td>60.1</td><td></td></tr><tr><td>DANet [34]</td><td>CVPR23</td><td>48.0</td><td>1</td><td></td><td></td></tr><tr><td>GaitGCI [10]</td><td>CVPR23</td><td>50.3</td><td>39.5</td><td>68.5</td><td>80.8</td></tr><tr><td>DyGait [54]</td><td>ICCV23</td><td>66.3</td><td>56.4</td><td>71.4</td><td>83.2</td></tr><tr><td>HSTL [52]</td><td>ICCV23</td><td>61.3</td><td>55.5</td><td>62.7</td><td>76.6</td></tr><tr><td>GaitMoE-T(ours)</td><td>=</td><td>71.3</td><td>62.5</td><td>74.4</td><td>84.9</td></tr><tr><td>GaitMoE-B(ours)</td><td></td><td>73.7</td><td>66.2</td><td>79.6</td><td>89.1</td></tr></table>

## 5.2 Implementation Details

We provide details about the training process. Inputs. All datasets are resized to 64 × 44. In addition, we employ spatial alignment module [37] as pre-processing to re-align input silhouettes for OccGait and OccCASIA-B. We adopt batch size [P, K] and the number of iterations, [8, 16], 40K for OccGait, OccCASIA-B, [32, 4], 60K for Gait3D, and [32, 4], 180K for GREW. We sample 30 frames of each gait sequence in the training stage and all frames are used for inference. Network. For OccGait and OccCASIA-B, we stack three 2D convolution blocks as our Baseline with the number of channels (64, 128, 256). Each 2D convolution block is followed by an MTE. After Horizontal Pooling with the part parameter of 64, we set individual MAE for each part as in Fig. 4(b) and M is set to 16. For Gait3D and GREW, we replace our Baseline with GaitBase-like architecture (4 residual blocks or 10 residual blocks) [11,18], setting channels to (64, 128, 256, 512). More Details are shown in Supplementary Materials. Optimization. GaitMoE is an end-to-end joint training framework only with ID labels. We use the optimizer of SGD with an initial learning rate of 0.1, which is decreasing by a factor of 0.1 per [10K, 20K, 30K], [10K, 20K, 30K], [20K, 40K, 50K], [80K, 120K, 150K] for OccGait, OccCASIA-B, Gait3D and GREW, respectively.

Table 4: Impact of MTE and MAE on OccGait, OccCASIA-B and Gait3D.
<table><tr><td colspan="2"></td><td colspan="2">OccGait</td><td colspan="2">OccCASIA-B</td><td colspan="2">Gait3D</td></tr><tr><td>MTE</td><td>MAE</td><td>NM CA CR</td><td>ST</td><td>NO CO SO</td><td>DO</td><td>Rank-1</td><td>mAP</td></tr><tr><td rowspan="3">√</td><td></td><td>87.7 69.0 75.1</td><td>77.2</td><td>85.8 75.9</td><td>79.6 80.0</td><td>59.2</td><td>48.6</td></tr><tr><td></td><td>89.7 78.3 77.3</td><td>81.2</td><td>88.2 76.9</td><td>82.1 83.6</td><td>66.2</td><td>56.6</td></tr><tr><td>√</td><td>89.4 78.4 77.4</td><td>81.8</td><td>87.6 78.3</td><td>81.7 82.5</td><td>67.2</td><td>57.3</td></tr><tr><td>√</td><td>√</td><td>91.4 82.1 79.9</td><td>84.7</td><td>89.5 80.5</td><td>83.9 85.3</td><td>71.3</td><td>62.5</td></tr></table>

Table 5: The number of Action Experts in MAE on OccGait.
<table><tr><td>MTE</td><td>MAE</td><td>NM</td><td>CA</td><td>CR</td><td>ST</td><td>Mean</td></tr><tr><td>√</td><td>1</td><td>89.5</td><td>78.3</td><td>78.0</td><td>82.1</td><td>82.0</td></tr><tr><td>√</td><td>8</td><td>90.6</td><td>80.7</td><td>79.1</td><td>83.3</td><td>83.4</td></tr><tr><td>√</td><td>16</td><td>91.4</td><td>82.1</td><td>79.9</td><td>84.7</td><td>84.5</td></tr><tr><td>√</td><td>32</td><td>90.8</td><td>81.2</td><td>79.9</td><td>84.6</td><td>84.1</td></tr></table>

For joint loss, we set β = 0.1 for OccGait, OccCASIA-B, and β = 1.0 for Gait3D and GREW. All the models are trained on NVIDIA 8×3090 GPUs.

## 5.3 Main Results

To evaluate the efectiveness of GaitMoE, we first compare with other state-ofthe-art methods on OccGait and OccCASIA-B with controlled occlusions, and then further make comparisons on Gait3D and GREW with complex covariates (e.g., uncertain occlusions).

Comparison on OccGait. To qualitatively and quantitatively validate Gait-MoE, we build the real-scenario gait database, OccGait, containing a wide range of occlusions and explicit annotations. Meanwhile, spatial and scale misalignment may occur in all occlusion scenarios. Tab. 1 shows that GaitMoE outperforms other state-of-the-art methods under all types of occlusions. Notably, CR causes interference from other pedestrians to overshadow the main subject information at certain angles, causing the model to overly focus on the obstructer, not the target. However, the Average results still show SoTA in the main manuscript, which validates the occlusion-solving traits.

Comparison on OccCASIA-B. Tab. 2 shows the overall results where setbased and temporal-based gait recognition methods (i.e., GaitSet, GaitPart, GaitGL, GaitBase) sufer from severe degradation under the occlusion scenario, comparing to their performances in their paper on holistic CASIA-B. Although STOR with silhouette registration alleviates the misalignment issue, the complex walking patterns under occlusions make the feature extraction dificult. Our proposed method outperforms all methods by a large margin, especially in the extremely challenging cloth-changing (CL) condition, e.g., exceeds GaitBase by 12.0% in NO, and 16.8% in DO. The experimental results demonstrate that GaitMoE efectively filters invalid occluded actions and extracts robust actions. Comparisons on Gait3D and GREW. We have validated the efectiveness of our method on the controlled occlusion environment (i.e., OccCASIA-B and OccGait). The large-scale wild gait databases also provide complex and uncertain occlusion scenarios. As shown in Tab. 3, GaitMoE-T (4 residual blocks) and GaitMoE-B (10 residual blocks) also achieve the highest performance among state-of-the-art methods, which further proves the generalizability and practicality of our method.

![](images/005942e9c6f28318a25b0f9d2fe98e3e722257f1ce30912ad055b1c009d08409.jpg)  
Fig. 6: The visualization of action composition. Here are an occluded gait sequence, action proposals and the selection map of action anchors. Alphabet denotes the action proposal. Each rectangle denotes one action proposal, and each row within one action proposal denotes an action anchor.

## 5.4 Ablation Study

In this section, we mainly qualitatively and quantitatively validate the MTE and MAE in OccGait, OccCASIA-B and Gait3D. Besides, we provide the visualization of action detection on OccCASIA-B.

The Efectiveness of MTE and MAE. Tab. 4 shows that each of MTE and MAE can extract better dynamic information. Especially, the combination of MTE and MAE i.e., GaitMoE, achieves a significant increase over Baseline, which proves that our proposed method discovers the representative actions by first coarse action extraction (i.e., action anchors), and then fine-grained action extraction (i.e., action proposals). Besides, we select (8, 4) for (S, K) on temporal experts since a smaller S or larger K degrades original information, and a larger S or smaller K restricts dynamic information. More Details are shown in Supplementary Materials. We also quantify the impact on the number of action experts. Tab. 5 illustrates that more experts may result in redundant actions, while fewer experts may not be able to capture the diverse actions. We select 16 as the parameter of GaitMoE.

The Visualization of Action Composition. To better understand and interpret the action detection in gait recognition, we visualize the key module MAE of action detection in Fig. 6, we select part index 60, action anchor with 2 dilated ratios and 6 action proposals as the example. MTE enables to infer the occluded region information by consecutive frames. For an example in E, the right occluded frame can hallucinate the missing region through the middle frame. MAE enables to select and integrate the most discriminative similar action anchors from diferent gait cycles. For an example in F, although missing information occurs, MAE combines multiple similar action anchors to further confirm this action type (i.e., swinging legs), integrating occluded information. Trade-of between Accuracy and Eficiency. As Fig. 7 shows, we compare all parameters of these models. For FLOPs, models input a 30-frame gait sequence without Separate FCs and BNNeck for significant comparisons. GaitMoE makes a trade-of and achieves SoTA without substantially increasing computational cost. In contrast, DyGait demands significant computation due to 3D convolutions and GaitBase ofers better eficiency with 2D convolutions but lower accuracy.

![](images/b4060a76c483b9177986714181e734e5f9f2af2efd15ca06d5c3b579ec6458d1.jpg)  
Fig. 7: The model comparisons on accuracy and eficiency. Rank-1 (%), Param. (M) and FLOPs. (G) on Gait3D.

## 5.5 Conclusion and Limitations

In this paper, we introduce an action detection perspective where a gait sequence is regarded as a composition of actions, allowing holistic body regions to infer occluded body regions and information integration between holistic and occluded actions. To detect accurate actions under complex occlusion scenarios, we propose an Action Detection Based Mixture of Experts (GaitMoE) to leverage dynamic contextual information i.e., gait continuity and gait cycle, to construct action anchors and action proposals. To obtain qualitative and quantitative occlusion analysis, we propose a novel Occluded Gait Recognition benchmark (OccGait) as a pioneering database with a wide range of occlusion scenarios and explicit annotations. Extensive experimental results have demonstrated that GaitMoE efectively captures accurate and robust actions for occluded gait recognition. In addition. we provide some limitations where the selection map of action anchors shown in Fig. 6 shows that some of the experts in GaitMoE present redundancy and similarity. We will explore the optimization and design for expert selection in the future.

## Acknowledgment

This work is jointly supported by National Natural Science Foundation of China (62276025, 62206022), Beijing Municipal Science & Technology Commission (Z23

1100007423015) and Shenzhen Technology Plan Program (KQTD2017033109321   
7368).

## References

1. An, W., Yu, S., Makihara, Y., Wu, X., Xu, C., Yu, Y., Liao, R., Yagi, Y.: Performance evaluation of model-based gait on multi-view very large population database with pose sequences. IEEE transactions on biometrics, behavior, and identity science 2(4), 421–430 (2020)

2. Bashir, K., Xiang, T., Gong, S., Mary, Q., et al.: Gait representation using flow fields. In: BMVC. pp. 1–11 (2009)

3. Chai, T., Li, A., Zhang, S., Li, Z., Wang, Y.: Lagrange motion analysis and view embeddings for improved gait recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20249–20258 (2022)

4. Chao, H., He, Y., Zhang, J., Feng, J.: Gaitset: Regarding gait as a set for cross-view gait recognition. In: Proceedings of the AAAI conference on artificial intelligence. vol. 33, pp. 8126–8133 (2019)

5. Chattopadhyay, P., Sural, S., Mukherjee, J.: Frontal gait recognition from occluded scenes. Pattern Recognition Letters 63, 9–15 (2015)

6. Chen, C., Liang, J., Zhao, H., Hu, H., Tian, J.: Frame diference energy image for gait recognition with incomplete silhouettes. Pattern Recognition Letters 30(11), 977–984 (2009)

7. Chen, X., Li, H., Li, M., Pan, J.: Learning a sparse transformer network for efective image deraining. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5896–5905 (2023)

8. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1290–1299 (2022)

9. Delgado-Escano, R., Castro, F.M., R. Cózar, J., Marin-Jimenez, M.J., Guil, N.: Mupeg—the multiple person gait framework. Sensors 20(5), 1358 (2020)

10. Dou, H., Zhang, P., Su, W., Yu, Y., Lin, Y., Li, X.: Gaitgci: Generative counterfactual intervention for gait recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5578–5588 (2023)

11. Fan, C., Liang, J., Shen, C., Hou, S., Huang, Y., Yu, S.: Opengait: Revisiting gait recognition towards better practicality. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9707–9716 (2023)

12. Fan, C., Peng, Y., Cao, C., Liu, X., Hou, S., Chi, J., Huang, Y., Li, Q., He, Z.: Gaitpart: Temporal part-based model for gait recognition. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14225– 14233 (2020)

13. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. The Journal of Machine Learning Research 23(1), 5232–5270 (2022)

14. Fu, Y., Meng, S., Hou, S., Hu, X., Huang, Y.: Gpgait: Generalized pose-based gait recognition. arXiv preprint arXiv:2303.05234 (2023)

15. Gross, R.: The cmu motion of body (mobo) database. Carnegie Mellon University, The Robotics Institute (2001)

16. Guo, H., Ji, Q.: Physics-augmented autoencoder for 3d skeleton-based gait recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19627–19638 (2023)

17. Gupta, A., Chellappa, R.: You can run but not hide: Improving gait recognition with intrinsic occlusion type awareness. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 5893–5902 (2024)

18. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

19. Hermans, A., Beyer, L., Leibe, B.: In defense of the triplet loss for person reidentification. arXiv preprint arXiv:1703.07737 (2017)

20. Hofmann, M., Geiger, J., Bachmann, S., Schuller, B., Rigoll, G.: The tum gait from audio, image and depth (gaid) database: Multimodal recognition of subjects and traits. Journal of Visual Communication and Image Representation 25(1), 195–206 (2014)

21. Hofmann, M., Wolf, D., Rigoll, G.: Identification and reconstruction of complete gait cycles for person identification in crowded scenes. In: Proc. Intern. Conf. on Computer Vision Theory and Applications (VISAPP), Algarve, Portugal (2011)

22. Hossain, M.A., Makihara, Y., Wang, J., Yagi, Y.: Clothing-invariant gait identification using part-based clothing categorization and adaptive weight control. Pattern Recognition 43(6), 2281–2291 (2010)

23. Hou, S., Cao, C., Liu, X., Huang, Y.: Gait lateral network: Learning discriminative and compact representations for gait recognition. In: European conference on computer vision. pp. 382–398. Springer (2020)

24. Hou, S., Liu, X., Cao, C., Huang, Y.: Set residual network for silhouette-based gait recognition. IEEE Transactions on Biometrics, Behavior, and Identity Science 3(3), 384–393 (2021)

25. Huang, X., Zhu, D., Wang, H., Wang, X., Yang, B., He, B., Liu, W., Feng, B.: Context-sensitive temporal feature learning for gait recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12909–12918 (2021)

26. Iwama, H., Okumura, M., Makihara, Y., Yagi, Y.: The ou-isir gait database comprising the large population dataset and performance evaluation of gait recognition. IEEE Transactions on Information Forensics and Security 7(5), 1511–1521 (2012)

27. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., Chen, Z.: Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020)

28. Li, B., Yang, J., Ren, J., Wang, Y., Liu, Z.: Sparse fusion mixture-of-experts are domain generalizable learners. arXiv e-prints pp. arXiv–2206 (2022)

29. Li, N., Zhao, X.: A multi-modal dataset for gait recognition under occlusion. Applied Intelligence 53(2), 1517–1534 (2023)

30. Liang, J., Fan, C., Hou, S., Shen, C., Huang, Y., Yu, S.: Gaitedge: Beyond plain end-to-end gait recognition for better practicality. arXiv preprint arXiv:2203.03972 (2022)

31. Lin, B., Zhang, S., Bao, F.: Gait recognition with multiple-temporal-scale 3d convolutional neural network. In: Proceedings of the 28th ACM international conference on multimedia. pp. 3054–3062 (2020)

32. Lin, B., Zhang, S., Yu, X.: Gait recognition via efective global-local feature representation and local temporal aggregation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14648–14656 (2021)

33. Lin, C., Xu, C., Luo, D., Wang, Y., Tai, Y., Wang, C., Li, J., Huang, F., Fu, Y.: Learning salient boundary feature for anchor-free temporal action localization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3320–3329 (2021)

34. Ma, K., Fu, Y., Zheng, D., Cao, C., Hu, X., Huang, Y.: Dynamic aggregated network for gait recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22076–22085 (2023)

35. Makihara, Y., Mannami, H., Yagi, Y.: Gait analysis of gender and age using a large-scale multi-view gait database. In: Computer Vision–ACCV 2010: 10th Asian Conference on Computer Vision, Queenstown, New Zealand, November 8-12, 2010, Revised Selected Papers, Part II 10. pp. 440–451. Springer (2011)

36. Mustafa, B., Riquelme, C., Puigcerver, J., Jenatton, R., Houlsby, N.: Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems 35, 9564–9576 (2022)

37. Peng, Y., Cao, C., He, Z.: Occluded gait recognition. In: 2023 International Joint Conference on Neural Networks (IJCNN). pp. 1–8. IEEE (2023)

38. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems 34, 8583–8595 (2021)

39. Roller, S., Sukhbaatar, S., Weston, J., et al.: Hash layers for large sparse models. Advances in Neural Information Processing Systems 34, 17555–17566 (2021)

40. Sepas-Moghaddam, A., Etemad, A.: Deep gait recognition: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022)

41. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

42. Shen, C., Fan, C., Wu, W., Wang, R., Huang, G.Q., Yu, S.: Lidargait: Benchmarking 3d gait recognition with point clouds. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1054–1063 (2023)

43. Shi, D., Zhong, Y., Cao, Q., Ma, L., Li, J., Tao, D.: Tridet: Temporal action detection with relative boundary modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18857–18866 (2023)

44. Song, C., Huang, Y., Wang, W., Wang, L.: Casia-e: a large comprehensive dataset for gait recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(3), 2801–2815 (2022)

45. Takemura, N., Makihara, Y., Muramatsu, D., Echigo, T., Yagi, Y.: Multi-view large population gait dataset and its performance evaluation for cross-view gait recognition. IPSJ transactions on Computer Vision and Applications 10, 1–14 (2018)

46. Tan, D., Huang, K., Yu, S., Tan, T.: Eficient night gait recognition based on template matching. In: 18th international conference on pattern recognition (ICPR’06). vol. 3, pp. 1000–1003. IEEE (2006)

47. Teepe, T., Gilg, J., Herzog, F., Hörmann, S., Rigoll, G.: Towards a deeper understanding of skeleton-based gait recognition. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1569–1577 (2022)

48. Teepe, T., Khan, A., Gilg, J., Herzog, F., Hörmann, S., Rigoll, G.: Gaitgraph: Graph convolutional network for skeleton-based gait recognition. In: 2021 IEEE International Conference on Image Processing (ICIP). pp. 2314–2318. IEEE (2021)

49. Tsuji, A., Makihara, Y., Yagi, Y.: Silhouette transformation based on walking speed for gait identification. In: 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. pp. 717–722. IEEE (2010)

50. Uddin, M.Z., Muramatsu, D., Takemura, N., Ahad, M.A.R., Yagi, Y.: Spatiotemporal silhouette sequence reconstruction for gait recognition against occlusion. IPSJ Transactions on Computer Vision and Applications 11(1), 1–18 (2019)

51. Uddin, M.Z., Ngo, T.T., Makihara, Y., Takemura, N., Li, X., Muramatsu, D., Yagi, Y.: The ou-isir large population gait database with real-life carried object and its performance evaluation. IPSJ Transactions on Computer Vision and Applications 10(1), 1–11 (2018)

52. Wang, L., Liu, B., Liang, F., Wang, B.: Hierarchical spatio-temporal representation learning for gait recognition. arXiv preprint arXiv:2307.09856 (2023)

53. Wang, L., Tan, T., Ning, H., Hu, W.: Silhouette analysis-based gait recognition for human identification. IEEE transactions on pattern analysis and machine intelligence 25(12), 1505–1518 (2003)

54. Wang, M., Guo, X., Lin, B., Yang, T., Zhu, Z., Li, L., Zhang, S., Yu, X.: Dygait: Exploiting dynamic representations for high-performance gait recognition. arXiv preprint arXiv:2303.14953 (2023)

55. Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mohammed, O.K., Singhal, S., Som, S., et al.: Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442 (2022)

56. Xu, C., Makihara, Y., Li, X., Yagi, Y.: Occlusion-aware human mesh model-based gait recognition. IEEE transactions on information forensics and security 18, 1309– 1321 (2023)

57. Xu, C., Makihara, Y., Ogi, G., Li, X., Yagi, Y., Lu, J.: The ou-isir gait database comprising the large population dataset with age and performance evaluation of age estimation. IPSJ Transactions on Computer Vision and Applications 9(1), 1–14 (2017)

58. Xu, C., Tsuji, S., Makihara, Y., Li, X., Yagi, Y.: Occluded gait recognition via silhouette registration guided by automated occlusion degree estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3199– 3209 (2023)

59. Yang, L., Peng, H., Zhang, D., Fu, J., Han, J.: Revisiting anchor mechanisms for temporal action localization. IEEE Transactions on Image Processing 29, 8535– 8548 (2020)

60. Yu, S., Tan, D., Tan, T.: A framework for evaluating the efect of view angle, clothing and carrying condition on gait recognition. In: 18th International Conference on Pattern Recognition (ICPR’06). vol. 4, pp. 441–444. IEEE (2006)

61. Zhang, C., Chen, X.P., Han, G.Q., Liu, X.J.: Spatial transformer network on skeleton-based gait recognition. Expert Systems p. e13244 (2023)

62. Zheng, J., Liu, X., Liu, W., He, L., Yan, C., Mei, T.: Gait recognition in the wild with dense 3d representations and a benchmark. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20228–20237 (2022)

63. Zhu, H., Zheng, W., Zheng, Z., Nevatia, R.: Gaitref: Gait recognition with refined sequential skeletons. arXiv preprint arXiv:2304.07916 (2023)

64. Zhu, H., Zheng, Z., Nevatia, R.: Gait recognition using 3-d human body shape inference. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 909–918 (2023)

65. Zhu, Z., Guo, X., Yang, T., Huang, J., Deng, J., Huang, G., Du, D., Lu, J., Zhou, J.: Gait recognition in the wild: A benchmark. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 14789–14799 (2021)