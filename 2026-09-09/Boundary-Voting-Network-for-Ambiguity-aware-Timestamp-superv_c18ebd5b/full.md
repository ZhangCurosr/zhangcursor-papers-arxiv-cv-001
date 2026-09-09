# Boundary Voting Network for Ambiguity-aware Timestamp-supervised Action Segmentation

Runzhong Zhang, Yueqi Duan, Member, IEEE, Yang Chen, Weipeng Hu, Chen Cai, Suchen Wang, and Yap-Peng Tan, Fellow, IEEE

Abstract—Timestamp-supervised action segmentation aims to segment and classify actions in untrimmed videos with a random frame annotated per action. Precisely localizing action boundaries from timestamp annotations is crucial for this setting, as it enables generating framewise pseudo-labels and applying the well-explored fully-supervised training. However, prevailing methods struggle with intrinsic uncertainty in boundary localization due to less discriminative features in action-transiting regions. This imprecise boundary estimation significantly reduces the stability and reliability of the generated pseudo-labels in ambiguous action-transiting regions, consequently resulting in performance deterioration of the trained segmentation models. In our paper, we introduce the boundary voting network that mitigates feature ambiguity by hierarchically propagating videolevel global prior knowledge into local action-transiting regions. By generating key action representations as votes throughout the video and targeting action-transiting regions, all votes collaboratively contribute to action-transiting feature enhancement and boundary localization refinement. Extensive experiments demonstrate the effectiveness of our method on GTEA, 50Salads, and Breakfast datasets.

Index Terms—Timestamp-supervised action segmentation, voting, boundary localization.

## I. INTRODUCTION

CTION segmentation aims to temporally segment untrimmed video sequences by assigning each frame a pre-defined action class [1]. The methodology has emerged as a promising approach for understanding long-form videos with complex action content [2]–[4], holding profound significance for applications such as human-robot interactions [5]–[9], video captioning [10]–[14], and advanced home monitoring systems [15]–[18]. This paper focuses on the timestampsupervised action segmentation setting, which involves annotating only a single, random frame for each action segment within the training video.

The key to the setting is inferring precise boundary localization from timestamp annotations, as it enables the generation of framewise pseudo-labels, which effectively transforms timestamp supervision into the well-explored fully-supervised problem [19]–[22]. To elaborate, by localizing the action boundary between every two consecutive annotated timestamps, the preceding annotation class can be automatically assigned to every frame before the boundary, while the subsequent annotation class to frames after the boundary. This iterative procedure, applied to each annotated timestamp pair, generates framewise pseudo-labels spanning the entire training video and facilitates fully-supervised learning. Consequently, the effectiveness of boundary localization directly influences the gap between timestamp and upper-bound full supervision, playing a pivotal role in both the model learning process and segmentation testing performance.

Despite the notable progress achieved, most prior works encounter heightened uncertainty in boundary localization due to feature ambiguity in action-transiting regions. The transitional frames, which occur adjacent to the boundaries where actions shift, often exhibit less discriminative semantic representations compared to frames indicating the progression of actions. For example, in the tea-making video shown in Fig. 1 (a), the transition from “pouring water” to “adding teabag” involves placing down the pot and moving the hand from the pot to the teabag, rather than explicit interactions with the objects. Moreover, the long duration of actions in the video often places boundaries at considerable temporal distances from any action centroid, further diminishing the correlation between transitional frames and other parts of the video containing rich action information. Consequently, the feature ambiguity in action-transiting regions leads to a lack of essential knowledge for precise boundary localization and instability in the generated pseudo-labels, resulting in cascaded performance deterioration in the subsequent fully-supervised training phase.

In contrast to previous studies primarily focusing on modellevel architecture adjustments [19], [21], [22], we reconsider boundary localization uncertainty through the feature-level enhancement strategy. Our proposed boundary voting network (BVN), depicted in Fig. 1 (b), introduces a global-to-local voting mechanism that enhances local action-transiting features by integrating global prior knowledge across the video. Specifically, we first generate intermediary framewise features using the segmentation model encoder and identify actiontransiting regions through bidirectional boundary detection, following the procedure in [21]. Due to the feature ambiguity surrounding boundaries, action-transiting regions persist for long temporal duration, leading to inaccurate framewise pseudo-label generation. To address this, we introduce two modules named the voting block and the aggregation block. The voting block derives two frame features as votes from each intermediary framewise feature, targeting the center of the corresponding action-transiting regions, with one toward the start of the action and the other toward its end. Subsequently, the aggregation block merges the votes back into the original video features, thereby enhancing action representations within action-transiting regions. Consequently, the proposed blocks globally generate and rearrange key action representations from the evenly distributed temporal sequence, with the new characteristic of clustering around boundaries, which facilitates feature enhancement in targeted regions. By hierarchically applying blocks among segmentation decoding layers, BVN continuously mitigates feature ambiguity and suppresses the action-transiting regions during training, ultimately generating more accurate framewise pseudo-labels and improving segmentation performance in both training and testing phases.

![](images/1f51770a46806bf514918192811f2b7432e542c0b63ca462a12203c47f66dd7e.jpg)  
Fig. 1. (a) Due to the inherent feature ambiguity in action-transiting regions, existing works suffer from significant uncertainty in boundary localization resulting in inaccurate framewise pseudo-label generation and cascaded performance deterioration. (b) Our network propagates and aggregates key action representations as votes into the action-transiting regions, thereby refining boundary localization and improving segmentation performance in both the training and testing phases.

To the best of our knowledge, our main contributions are summarized as follows:

• We propose the first work to investigate the inherent feature-level ambiguity in action-transiting regions and address boundary localization uncertainty through a voting mechanism.

• The global-to-local BVN sequentially constructs the voting block to propagate global prior knowledge towards local action-transiting regions, and the aggregation block to enhance targeted less discriminative features. Our model hierarchically suppresses ambiguous regions and accurately separates adjacent actions during both training and testing phases.

• Our method demonstrates its superiority across three real-world datasets with various evaluation metrics. Additionally, we validate the effectiveness of the voting mechanism through comprehensive ablation studies.

## II. RELATED WORKS

In this section, we briefly review recent works on fullysupervised action segmentation and timestamp-supervised action segmentation.

## A. Fully-supervised action segmentation

Existing fully-supervised approaches segment videos with dense framewise annotations provided during training. Early approaches [23], [24] classified frames through sliding windows with non-maximum suppression, which failed to capture the temporal dependency between actions. To model action sequences, alternative approaches applied Markov model [25] and recurrent neural network [26]. Recently, various temporal convolutional network-based approaches have been proposed to capture long-range action dependencies effectively. Lea et al. [27] introduced the use of TCN in action segmentation by designing an encoder-decoder architecture with 1D temporal convolution/deconvolution. Ding et al. [28] presented a hybrid network that integrates RNN into TCN for the decoder layers, and TDRN [29] replaced temporal convolution with deformable convolution. However, these methods all downsampled the video and inevitably led to information loss. Instead, Farruha et al. [30] proposed MS-TCN, a multi-stage temporal convolutional network processing the video at full length. In MS-TCN++, Li et al. [31] introduced dual dilated temporal convolution to capture local-global features and improved the model efficiency. Beyond TCN frameworks, recent efforts have explored diverse model architectures and training strategies. ASFormer [32] introduced the first transformerbased network for action segmentation. Li et al. [33] utilized the textual information in action labels and contrastively trained the text encoder with the video encoder. DiffAct [34] integrated diffusion into action segmentation for the first time. FACT [35] introduced a two-branch architecture that concurrently learns both frame-level and action-level features, and the communication mechanism between the branches through cross-attention. However, while fully-supervised approaches have demonstrated remarkable performance across various datasets, obtaining framewise labels demands significant manual effort, particularly for large-scale real-world video data.

## B. Timestamp-supervised action segmentation

For the timestamp-supervised action segmentation, each action segment within the training video is annotated using a single arbitrary frame. Inspired by the foundational concept of point-supervised semantic segmentation [36], Li et al. [21] introduced a learning strategy for timestamp annotations that first generate framewise pseudo-labels through action boundary estimation and proceed with fully-supervised training. Advancing beyond the approach, Khan et al. [20] replaced heuristic boundary estimation with the graph neural network [37], enabling learning in an end-to-end manner. To reduce oscillations during training, Zhao et al. [22] proposed a teacher network in parallel with the segmentation model. In UVAST [19], constrained K-medoid was directly applied to input video features, offering an alternative strategy distinct from inferring boundaries on intermediary features. Instead of assigning framewise hard labels, Rahaman et al. [38] acknowledged label uncertainty in unlabeled frames and employed an Expectation-Maximization approach, demonstrating remarkable robustness in handling annotation errors. Recently, Liu et al. [39] proposed D-TSTAS to reduce over-reliance on annotated timestamps, which ensured the model captured more contextual information in both initialization and refinement steps. Nevertheless, despite extensive efforts directed toward the timestamp-supervised setting, the boundary localization uncertainty associated with feature ambiguity has yet to be thoroughly explored.

![](images/8b0802d758835f5c424d24ad66f1fdb4c61e1dae4921ae9a6e65bf1c77d57cde.jpg)  
Fig. 2. Overview of the global-to-local BVN architecture. After obtaining video features x from the encoder, our model iteratively processes them through stages containing the voting block, aggregation block, and decoder. Taking x and normalized temporal indexes r as inputs, the voting block computes features $x _ { s } ^ { \prime }$ and $x _ { e } ^ { \prime }$ to capture key action representations, and predicts the start time $r _ { s } ^ { \prime }$ and end time $r _ { e } ^ { \prime }$ of the current action independently through start net and end net. Based on the obtained vote set $\{ x ^ { \prime } , r ^ { \prime } \}$ , the subsequent aggregation block constructs temporal farthest point sampling (TFPS) to select representative votes, which are then aggregated into the original video features for enhancement. On top of the final decoder outputs, we calculate the framewise predictions $y _ { 1 } ^ { T }$ using MLP. Through hierarchical voting propagation and aggregation across stages, our model continuously suppresses action-transiting feature ambiguity while simultaneously refining boundary localization.

## III. BOUNDARY VOTING NETWORK

The objective of action segmentation is to segment the video into distinct procedural steps and classify human action within each segment. This study focuses on the timestampsupervised setting, which relies on annotations for only a single random frame per action instead of framewise labels. For a training sample $\bar { v } _ { 1 } ^ { T } = [ v _ { 1 } , . . . , v _ { T } ]$ of length $T$ containing $N$ action segments, we employ sparse annotations in the form of $\{ l _ { 1 } ^ { N } , a _ { 1 } ^ { N } \}$ , where $l _ { 1 } ^ { N } = \mathbf { \bar { \rho } } [ l _ { 1 } , . . . , l _ { N } ]$ denotes the timestamps of annotated frames, and $a _ { 1 } ^ { N } \ = \ [ a _ { 1 } , . . . , a _ { N } ]$ indicates the corresponding action labels, with N being significantly lower than T. Our goal is to train a model using sparsely annotated timestamps capable of accurately classifying each video frame, represented as $y _ { 1 } ^ { T } = [ y _ { 1 } , . . . , y _ { T } ]$ . It is important to note that the values of T and N may vary for each video.

## A. Overview

In the timestamp-supervised setting, the primary challenge lies in boundary localization uncertainty caused by feature ambiguity, as shown in Fig. 1. Inspired by conventional Hough voting [40] in object detection, we introduce a global-tolocal boundary voting network to propagate and aggregate key action representations into action-transiting regions, thereby enhancing the less discriminative features within local regions from a global perspective. As illustrated in Fig. 2, we construct an encoder-decoder action segmentation network based on the transformer architecture [41]. The encoder processes video features $v _ { 1 } ^ { T }$ to generate intermediary framewise features x. To effectively propagate action representations in the temporal domain, we introduce a voting block along with a votes aggregation block seamlessly integrated before each decoder layer. Given that the inputs of every voting block represent the framewise features with the same dimension, we use the same symbol x for simplicity.

## B. Voting block architecture

The voting block aims to globally propagate key action representations as prior knowledge into targeted local temporal regions. Given input video features $x ,$ it independently generates two votes for each frame, with one toward the start of the action and the other toward its end. Despite originating from the same frame, these two votes serve different purposes by enhancing features in different action-transiting regions. Unlike conventional Hough voting which computes votes based on a pre-defined lookup table, our approach directly generates votes through the deep network.

More specifically, for video features $x \ : = \ : [ x _ { 1 } , . . . , x _ { T } ] \in$ $\mathbb { R } ^ { T \times D }$ , where D represents the framewise feature dimension, we first compute the normalized temporal indexes $\boldsymbol { r } \in \mathbb { R } ^ { T \times 1 }$ by dividing frame indexes $[ 1 , 2 , . . . , T ]$ by T. This step records the normalized position of each frame in the video. Since each frame outputs two votes targeting different positions, we apply two independent networks, denoted as start net and end net, for the generation of separate vote sets. The start net consists of three layers of 1D convolution, with batch normalization and ReLU activation following the first two layers. It takes the video features $x \ = \ [ x _ { 1 } , . . . , x _ { T } ] \ \in$ $\mathbb { R } ^ { T \times D }$ as inputs, maintaining the same feature dimensions in the first two 1D convolution layers and modifying the feature dimensions in the outputs of the last layer, which are represented as $[ \Delta r _ { s } ; \Delta x _ { s } ] \in \dot { \mathbb { R } } ^ { T \times ( 1 + D ) }$ , where $\Delta r _ { s } \in \mathbb { R } ^ { T \times 1 }$ denotes the index offsets and $\Delta x _ { s } \ \in \ \mathbb { R } ^ { T \times D }$ represents the feature offsets. The end net shares the same architecture as start net and follows the same procedure to generate $[ \Delta r _ { e } ; \Delta x _ { e } ] \in \mathbb { R } ^ { T \times ( 1 + D ) }$ , where the subscripts s and e represent the outputs from start net and end net respectively. While the feature offsets provide key action representations as global prior knowledge, the index offsets temporally shift the original features into new positions. Consequently, the voting block enables the model to propagate the action knowledge to any targeted temporal position in the video. Given the original features, indexes, and their corresponding offsets, the vote set can be generated as follows:

$$
\Delta x _ { s } , \Delta r _ { s } = s t a r t \_ n e t ( x ) ,
$$

$$
\left\{ x _ { s } ^ { \prime } , r _ { s } ^ { \prime } \right\} = \left\{ x + \Delta x _ { s } , r + \Delta r _ { s } \right\} ,\tag{1}
$$

(2)

$$
\Delta x _ { e } , \Delta r _ { e } = e n d \_ n e t ( x ) ,\tag{3}
$$

$$
\left\{ x _ { e } ^ { \prime } , r _ { e } ^ { \prime } \right\} = \left\{ x + \Delta x _ { e } , r + \Delta r _ { e } \right\} ,\tag{4}
$$

where $x _ { s } ^ { \prime } , x _ { e } ^ { \prime } \in \mathbb { R } ^ { T \times D }$ and $r _ { s } ^ { \prime } , r _ { e } ^ { \prime } \in \mathbb { R } ^ { T \times 1 }$ . We then concatenate $\{ x _ { s } ^ { \prime } , r _ { s } ^ { \prime } \}$ and $\{ x _ { e } ^ { \prime } , r _ { e } ^ { \prime } \}$ into vote set $\{ x ^ { \prime } , r ^ { \prime } \}$ , where vote features $\bar { x ^ { \prime } \in \mathbb { R } ^ { 2 T \times \bar { D } } }$ and vote temporal indexes $r ^ { \prime } \in \mathbb { R } ^ { 2 T \times 1 }$

While votes and frames share the same feature dimension, the generated votes exhibit new characteristics in the temporal domain, as the block rearranges the evenly distributed frame features and ensures votes cluster around boundaries. The voting procedure enables our BVN to automatically select global prior knowledge containing rich action representations, which serves as a fundamental basis for suppressing originally less discriminative action-transiting regions. In the next subsection, we will discuss the votes aggregation block for action-transiting feature enhancement.

## C. Votes aggregation

Based on the votes generated from the voting block, the aggregation block follows a two-step process. It first groups the votes and subsequently aggregates them back into the original video features.

Due to the uneven temporal distribution of votes, selecting representative votes for grouping without introducing noise poses a significant challenge. Inspired by the unstructured spatial distribution observed in 3D point clouds [42] and the corresponding farthest point sampling technique for subset selection [43], we introduce temporal farthest point sampling (TFPS) into action segmentation for the first time. TFPS targets P “key votes” from the vote set $\{ x ^ { \prime } , r ^ { \prime } \}$ by iteratively selecting points that maximize the minimum distance to already chosen points. In contrast to traditional FPS which calculates distance based on 3D coordinates, we measure the distance between two votes by subtracting their vote temporal indexes $r ^ { \prime }$ . Subsequently, for each vote in the vote set, we compute its distance to all key votes and select the minimum value, representing its distance to the nearest key vote. If the distance is below the threshold $E ,$ the vote is included in the group of the corresponding key vote. By repeating the above procedure for all votes, we obtain the final key vote groups $g = g _ { 1 } , . . . , g _ { P } .$ . As votes naturally cluster around action boundaries, the grouping procedure with a distance threshold ensures the selection of sufficiently representative votes for the subsequent aggregation procedure.

Each obtained key vote group conveys essential information from two perspectives. First, it provides key action representations as “what to vote,” represented by the features of all votes within the group. Second, it provides temporal information about “where to vote,” represented by the centroid of the key vote group. Therefore, following the grouping procedure, we pass each key vote group through a shared multi-layer perceptron (MLP) network with max pooling, transforming the key vote group features into a D-dimensional feature for aggregation. Subsequently, we retrieve the normalized temporal index associated with the centroid of each key vote group, and aggregate the acquired D-dimensional feature into the original video features $x$ with the corresponding temporal index. To elaborate, for each key vote group $g _ { p }$ with centroid’s normalized temporal index $c ,$ the aggregation process is represented as follows:

$$
g _ { p } ^ { \prime } = \mathbf { M P } ( \mathbf { M L P } ( g _ { p } ) ) ,\tag{5}
$$

$$
\begin{array} { r } { x _ { c - i } ^ { c + i } = x _ { c - i } ^ { c + i } + g _ { p } ^ { \prime } , } \end{array}\tag{6}
$$

where MP represents max pooling, $g _ { p } ^ { \prime }$ denotes the aggregated feature of the key vote group ${ \mathit { g _ { p } } } ,$ and $x _ { c - i } ^ { c + i }$ indicates the video feature list with central normalized temporal index as c and window size as 2i + 1. Following the aggregation, the updated video features are forwarded to the subsequent decoder layer while retaining the same dimension. We sequentially concatenate multiple stages with the same architecture of voting block, aggregation block, and decoder, and generate the framewise predictions $y _ { 1 } ^ { T }$ from the final decoder using the MLP network.

By proposing voting and aggregation blocks across stages, our BVN demonstrates strong capability in hierarchically propagating and aggregating global prior knowledge from votes into less discriminative regions. Nevertheless, to enable BVN for continuous action-transiting region suppression and boundary localization refinement, two essential prerequisites are required: firstly, identifying action-transiting regions between annotated timestamps, and secondly, precisely targeting generated votes into specified regions. The subsequent subsections will delve into these requirements, beginning with generating action-transiting regions and then designing loss functions for positioning votes.

## D. Generating action-transiting regions

With the provided training video, the prevailing approach for network learning involves generating framewise pseudolabels based on annotated timestamps $\left\{ l _ { 1 } ^ { N } , a _ { 1 } ^ { N } \right\}$ , followed by training the segmentation model in a fully-supervised manner. Notably, the boundary localization between each pair of annotated timestamps is equivalent to the pseudo-label generation, as every frame between the annotated timestamp and the obtained boundary can be assigned the same class label based on this annotation. In the following, we will discuss the boundary localization process and the subsequent generation of action-transiting regions.

Consistent with the approach in [21], we employ bidirectional boundary detection on the video features x = $[ x _ { 1 } , . . . , x _ { T } ]$ , aiming to determine the optimal time, denoted as τ, that partitions the selected period of the video sequence into two clusters. The objective is to minimize the feature distance between every frame and its cluster center. Specifically, utilizing annotated timestamp locations $l _ { n } , l _ { n + 1 } \in l _ { 1 } ^ { N }$ , we identify the boundary location $b _ { n }$ between them as follows:

$$
b _ { n } ^ { F W } = \mathop { \arg \operatorname* { m i n } } _ { \tau } \sum _ { t = b _ { n - 1 } ^ { F W } } ^ { \tau } d ( x _ { t } , m _ { n } ) + \sum _ { t = \tau + 1 } ^ { l _ { n + 1 } } d ( x _ { t } , m _ { n + 1 } ) ,\tag{7}
$$

$$
b _ { n } ^ { B W } = \underset { \tau } { \arg \operatorname* { m i n } } \sum _ { t = l _ { n } } ^ { \tau } d ( x _ { t } , m _ { n } ) + \sum _ { t = \tau + 1 } ^ { b _ { n + 1 } ^ { B W } } d ( x _ { t } , m _ { n + 1 } ) ,\tag{8}
$$

where FW, BW represents forward and backward, while $d ( . , . )$ calculates the Euclidean distance between the given feature vectors. m denotes the mean of all features $x _ { t }$ within the upper-lower bound of summation. Therefore, $m _ { n }$ and $m _ { n + 1 }$ represent the mean feature of $x _ { t }$ within the first and second summations, as indicated in the formulas.

We apply forward and backward boundary detection separately on video features x to generate two sets of boundary estimations: $b ^ { F W }$ and $b ^ { B W }$ , resulting in two sets of framewise pseudo-labels. Due to the inherent feature ambiguity, these two sets of generated framewise pseudo-labels may conflict with frames around the boundaries. In such cases, the actiontransiting region between the $n ^ { t h }$ and $\left( n + 1 \right) ^ { t h }$ actions can be represented as:

$$
t \in [ \operatorname* { m i n } ( b _ { n } ^ { F W } , b _ { n } ^ { B W } ) , \operatorname* { m a x } ( b _ { n } ^ { F W } , b _ { n } ^ { B W } ) ] .\tag{9}
$$

By iteratively applying the voting blocks and aggregation blocks on the video features, the temporal duration of action-transiting regions undergoes continuous suppression. This demonstrates that our BVN effectively mitigates feature ambiguity and uncertainty in boundary localization. In the experiment section, we provide detailed visualizations to further illustrate the reduction in action-transiting regions across different stages.

## E. Loss functions and training process

Existing timestamp-supervised action segmentation approaches [21], [22], [39] commonly incorporate framewise classification loss, smoothing loss, and confidence loss during the training phase. In addition to these three, we introduce the voting loss to ensure that each vote is directed towards the corresponding boundary neighboring regions. We provide further details on the losses below.

Framewise classification loss. We apply the cross-entropy loss to the model predictions. Here $y _ { t , \tilde { a } }$ represents the predicted probability of ground truth action a˜ for time t.

$$
\mathcal { L } _ { f r a m e } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } - \log ( y _ { t , \tilde { a } } ) .\tag{10}
$$

Smoothing loss. To alleviate the over-segmentation problem, we calculate the truncated mean square error [30]:

$$
\mathcal { L } _ { s m o o t h } = \frac { 1 } { T A } \sum _ { t = 1 } ^ { T } \sum _ { a = 1 } ^ { A } \operatorname* { m i n } ( | \log y _ { t , a } - \log y _ { t - 1 , a } | , \delta ) ^ { 2 } ,\tag{11}
$$

where A represents the total number of action classes, $y _ { t , a }$ indicates the predicted probability of action a for time t, and δ is a constant hyperparameter. The loss enhances consistency between neighboring frames and ensures a smooth transition in predictions.

Confidence loss. With frames distant from the annotated timestamp $l _ { n }$ , confidence in predicting them as $a _ { n }$ decreases. The confidence loss is designed accordingly:

$$
\mathcal { L } _ { c o n f } = \frac { 1 } { 2 ( l _ { N } - l _ { 1 } ) } \sum _ { n = 1 } ^ { N } \Big ( \sum _ { t = l _ { n - 1 } } ^ { l _ { n + 1 } } \theta _ { a _ { l _ { n } } , t } \Big ) ,\tag{12}
$$

TABLE I  
ACTION SEGMENTATION PERFORMANCE ON THE GTEA, 50SALADS, AND BREAKFAST DATASETS, WITH THE TABLE ORGANIZED ACCORDING TO DIFFERENT BACKBONES. THE DASH (-) SYMBOL INDICATES THAT NO PRIOR RESULT IS AVAILABLE.
<table><tr><td></td><td colspan="4">GTEA</td><td colspan="4"></td><td colspan="4"></td><td colspan="4">Breakfast</td></tr><tr><td>Methods</td><td colspan="3">F1 @{10, 25, 50}</td><td>Edit</td><td>Acc</td><td></td><td colspan="3">F1 @{10, 25, 50}</td><td>Edit Acc</td><td></td><td>F1 @{10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td>Li et al. [21]</td><td>78.9</td><td>73.0</td><td>55.4</td><td>72.3</td><td>66.4</td><td>73.9</td><td>70.9</td><td>60.1</td><td>66.8</td><td>75.6</td><td>70.5</td><td>63.6</td><td>47.4</td><td>69.9</td><td>64.1</td></tr><tr><td>Zhao et al. [22]</td><td>84.3</td><td>81.7</td><td>64.8</td><td>79.8</td><td>74.4</td><td>78.5</td><td>75.5</td><td>63.4</td><td>71.8</td><td>77.7</td><td>73.1</td><td>66.5</td><td>49.4</td><td>72.6</td><td>64.6</td></tr><tr><td>Khan et al. [20]</td><td>81.5</td><td>77.5</td><td>60.8</td><td>75.6</td><td>66.1</td><td>75.1</td><td>72.3</td><td>61.0</td><td>67.6</td><td>75.1</td><td>67.9</td><td>61.0</td><td>45.3</td><td>67.0</td><td>61.4</td></tr><tr><td>EM-TSS [38]</td><td></td><td>82.7</td><td>66.5</td><td>82.3</td><td>70.5</td><td></td><td>75.9</td><td>64.7</td><td>71.6</td><td>77.9</td><td></td><td>63.7</td><td>49.8</td><td>67.2</td><td>67.0</td></tr><tr><td>Du et al. [44]</td><td>83.7</td><td>79.8</td><td>65.4</td><td>77.2</td><td>70.1</td><td>77.3</td><td>74.7</td><td>63.7</td><td>70.1</td><td>78.6</td><td>71.2</td><td>64.6</td><td>48.9</td><td>71.6</td><td>65.7</td></tr><tr><td>RWS [45]</td><td>80.9</td><td>74.1</td><td>56.3</td><td>76.2</td><td>59.3</td><td>76.7</td><td>72.8</td><td>55.5</td><td>69.3</td><td>70.0</td><td>70.9</td><td>64.7</td><td>44.8</td><td>71.1</td><td>60.2</td></tr><tr><td>Sayed et al. [46]</td><td>82.1</td><td>78.7</td><td>63.0</td><td>74.8</td><td>70.4</td><td>77.3</td><td>75.2</td><td>63.6</td><td>69.8</td><td>75.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TSCL [47]</td><td>88.5</td><td>84.9</td><td>69.0</td><td>83.0</td><td>74.5</td><td>79.5</td><td>76.2</td><td>61.8</td><td>73.7</td><td>74.9</td><td>71.3</td><td>64.3</td><td>47.5</td><td>68.9</td><td>64.8</td></tr><tr><td>BVN (MS-TCN++)</td><td>86.2</td><td>84.9</td><td>68.2</td><td>84.7</td><td>75.5</td><td>81.4</td><td>76.8</td><td>67.8</td><td>73.5</td><td>80.1</td><td>74.4</td><td>68.0</td><td>50.4</td><td>74.0</td><td>67.8</td></tr><tr><td>UVAST + alignment [19]</td><td>70.8</td><td>63.5</td><td>49.2</td><td>88.2</td><td>55.3</td><td>75.7</td><td>70.6</td><td>58.2</td><td>78.4</td><td>67.8</td><td>72.0</td><td>64.1</td><td>48.6</td><td>74.3</td><td>60.2</td></tr><tr><td>UVAST + Viterbi [19]</td><td>87.2</td><td>83.7</td><td>66.0</td><td>89.3</td><td>70.5</td><td>83.0</td><td>79.6</td><td>65.9</td><td>78.2</td><td>77.0</td><td>71.3</td><td>63.3</td><td>48.3</td><td>74.1</td><td>60.7</td></tr><tr><td>UVAST + FIFA [19]</td><td>80.7</td><td>75.2</td><td>57.4</td><td>88.7</td><td>66.0</td><td>80.2</td><td>74.9</td><td>61.6</td><td>78.6</td><td>72.5</td><td>72.0</td><td>64.2</td><td>47.6</td><td>74.1</td><td>60.3</td></tr><tr><td>Yang et al. [48]</td><td>88.2</td><td>85.5</td><td>67.3</td><td>84.0</td><td>69.2</td><td>84.4</td><td>81.3</td><td>67.8</td><td>77.9</td><td>77.0</td><td>71.5</td><td>64.2</td><td>47.0</td><td>72.3</td><td>64.6</td></tr><tr><td>D-TSTAS [39]</td><td>91.5</td><td>90.1</td><td>76.2</td><td>88.5</td><td>75.7</td><td>84.2</td><td>82.1</td><td>71.5</td><td>77.6</td><td>80.0</td><td>76.7</td><td>69.3</td><td>50.7</td><td>75.8</td><td>65.7</td></tr><tr><td>BVN (ASFormer)</td><td>91.7</td><td>90.5</td><td>76.8</td><td>89.9</td><td>76.7</td><td>85.1</td><td>82.5</td><td>72.3</td><td>79.0</td><td>81.2</td><td>77.3</td><td>69.7</td><td>51.2</td><td>76.5</td><td>68.3</td></tr></table>

TABLE II

FRAMEWISE PSEUDO-LABEL GENERATION PERFORMANCE ON THE GTEA, 50SALADS, AND BREAKFAST DATASETS, WITH THE TABLE ORGANIZED ACCORDING TO DIFFERENT BACKBONES.
<table><tr><td></td><td colspan="3">GTEA</td><td colspan="3">50Salads</td><td colspan="4">Breakfast</td></tr><tr><td>Methods</td><td>F1 @{10, 25, 50}</td><td></td><td>Acc</td><td>F1 @{10, 25, 50}</td><td></td><td>Acc</td><td>F1@{10, 25, 50}</td><td></td><td></td><td>Acc</td></tr><tr><td>Li et al. [21]</td><td>96.6</td><td>86.3 65.5</td><td>75.5</td><td>99.4</td><td>95.0 76.8</td><td>79.9</td><td>96.0</td><td>87.3</td><td>67.6</td><td>72.6</td></tr><tr><td>RWS [45]</td><td>96.7</td><td>89.4 71.4</td><td>78.6</td><td>99.7</td><td>97.5 81.0</td><td>80.6</td><td>96.5</td><td>88.9</td><td>69.5</td><td>76.1</td></tr><tr><td>BVN (MS-TCN++)</td><td>98.6 95.3</td><td>78.9</td><td>79.4</td><td>99.8</td><td>97.9 82.5</td><td>82.2</td><td>97.1</td><td>90.0</td><td>71.7</td><td>77.9</td></tr><tr><td>UVAST [19]</td><td>99.8 97.7</td><td>83.0</td><td>75.3</td><td>97.5</td><td>90.4 75.6</td><td>81.3</td><td>95.5</td><td>87.5</td><td>70.0</td><td>76.9</td></tr><tr><td>BVN (ASFormer)</td><td>99.8 98.2</td><td>85.9</td><td>80.5</td><td>99.9</td><td>98.3 84.7</td><td>83.4</td><td>97.8</td><td>90.7</td><td>73.4</td><td>79.8</td></tr></table>

$$
\theta _ { a _ { l _ { n } } , t } = \left\{ \begin{array} { l l } { \operatorname* { m a x } ( 0 , \log y _ { t - 1 , a _ { l _ { n } } } - \log y _ { t , a _ { l _ { n } } } ) , } & { t < l _ { n } } \\ { \operatorname* { m a x } ( 0 , \log y _ { t , a _ { l _ { n } } } - \log y _ { t - 1 , a _ { l _ { n } } } ) , } & { t \geq l _ { n } } \end{array} , \right.\tag{13}
$$

where $y _ { t , a _ { l _ { n } } }$ represents the predicted probability of action $\boldsymbol { a } _ { l _ { n } }$ for time t.

Voting loss. We propose the voting loss to ensure that each vote is directed toward the corresponding action-transiting region. Specifically, our start net and end net independently guide votes from each frame into their respective starting and ending action-transiting regions. For the region between the $n ^ { t h }$ and $\left( n + 1 \right) ^ { t h }$ actions, as defined in Eq. (9), the center is computed as $\frac { b ^ { F W } n + b ^ { B \dot { W } } n } { 2 }$ . For frames belonging to the $\left( n + 1 \right) ^ { t h }$ action, we determine the ground truth index offsets for start net by subtracting the index of each frame from the center. We follow the same process for end net and supervise the temporal offsets learning using the regression loss:

$$
\mathcal { L } _ { v o t e } = \frac { 1 } { 2 T } \sum _ { t = 1 } ^ { T } ( ( \Delta r _ { s } ^ { * } - \Delta r _ { s } ) ^ { 2 } + ( \Delta r _ { e } ^ { * } - \Delta r _ { e } ) ^ { 2 } ) ,\tag{14}
$$

where $\Delta r _ { s } ^ { * }$ and ${ \Delta } r _ { e } ^ { * }$ represent the ground truth index offsets of the start net and end net. $\Delta r _ { s }$ and $\Delta r _ { e }$ denote the predicted index offsets from these two networks, as detailed in Sec. III-B.

In summary, the final loss function for training the action segmentation model is expressed as:

$$
\mathcal { L } = \mathcal { L } _ { f r a m e } + \alpha \mathcal { L } _ { s m o o t h } + \beta \mathcal { L } _ { c o n f } + \gamma \mathcal { L } _ { v o t e } ,\tag{15}
$$

where different losses are balanced using the hyperparameters α, β, and γ.

We adopt the same two-stage training pipeline as described in [21], consisting of the initialization training and the iterative training. The initialization stage utilizes only the annotated timestamps for training, which provides fundamental knowledge for network learning. For the iterative training stage, the network is supervised using the generated framewise pseudolabels, while iteratively refining the quality of these pseudolabels as training progresses. The voting loss is introduced after the initialization stage.

## F. Discussions with related TCSVT papers

Zou et al. [49] and Li et al. [50] adopted transcript supervision and dense multi-task annotations, respectively, whereas our paper employs timestamp supervision. Li et al. [1] applied a graph-based convolutional network for human skeleton joint data, while we focus on addressing feature-level ambiguity in conventional RGB-based features. Du et al. [51] and Shao et al. [52] formulated the problem as predicting each action segment’s category, confidence score, start time, and end time, whereas we adhere to the classical action segmentation paradigm by generating framewise action predictions. Compared to the aforementioned papers, we propose the first work to address feature-level ambiguity in action-transiting regions through a global-to-local voting mechanism, distinguishing itself in supervision format, methodological framework, and problem formulation.

![](images/7f9f243c72fd7453d62fefdb23fd00ff334a263e4e607905f5e8983be0f76c1a.jpg)  
Fig. 3. Action segmentation visualizations on the (a) GTEA, (b) 50Salads, and (c) Breakfast video samples, with “GT” denoting the ground truth segmentation Compared to existing methods, our approach demonstrates reliable performance in accurately predicting actions and achieving better alignment with the start and end times of ground truth actions, as highlighted by the red dashed rectangles.

## IV. EXPERIMENTS

In this section, we provide a brief overview of the experimental setup, followed by both quantitative and qualitative comparisons. We then provide additional quantitative analysis and conduct detailed ablation studies to assess the effectiveness of the proposed method.

## A. Experimental setup

Datasets. We conduct experiments on three real-world video datasets: Georgia Tech Egocentric Activities (GTEA) [53], 50Salads [54], and Breakfast [55]. The GTEA dataset consists of 58 egocentric instructional videos with 11 action classes, capturing daily activities with an average duration of 1 minute per video. The 50Salads dataset contains 50 instructional videos with 19 actions related to salad preparation, with an average video duration of 6.4 minutes. The Breakfast dataset includes 1712 third-person videos varying from seconds to a few minutes, covering 48 different actions across 10 breakfastrelated activities, such as making tea and preparing coffee. To ensure a fair comparison, we employ the same annotations as described in [21].

Evaluation metrics. We apply the following metrics for evaluation: (1) Framewise accuracy (Acc), which measures the proportion of correctly predicted frames relative to the total number of frames. (2) Segmental edit score (Edit), which quantifies the similarity of the predicted sequence and the ground truth using Levenshtein distance. It evaluates the quality of the action sequence without relying on framewise prediction. (3) The F1 score, which compares the intersection over union (IoU) of each segment with the ground truth. We calculate the scores under three thresholds: F1@{10, 25, 50}.

Implementation details. We utilize pre-extracted I3D [56] features as the network inputs. For the segmentation model, we employ the same encoder and decoder architecture as in ASFormer [32] and set the number of decoders to 3. The number of voting and aggregation blocks is the same as the decoders, as these blocks are integrated before each decoder layer. We set the intermediary feature dimension D to 64, the distance threshold E to 0.1, and i in Eq. (6) to

![](images/707fb5437a4890cf5df48dce92b8b4952812ca5b4d01a38c1e20191a0d8f2c5c.jpg)  
Fig. 4. Hierarchical boundary localization refinement during training on the Breakfast video samples, with “GT” denoting the ground truth segmentation The action-transiting regions, depicted in blue, are progressively suppressed as the voting mechanism is iteratively applied at different stages, as highlighte by the red dashed rectangles.

![](images/40fe3b3469409c571c9eb2e7f2c28a67e77f76ac65bc5a55add1941d9915e9af.jpg)  
Fig. 5. Visualizations of voting from explicit to ambiguous action frames in action-transiting regions, along with the probability score improvements on the (a) GTEA, (b) 50Salads, and (c) Breakfast video samples.

5. We apply random initialization in TFPS to select the first point, and BVN returns stable outputs due to the substantial number of votes and their clustering characteristics around boundaries. During training, the network is initially trained with only annotated timestamps for 70 epochs, followed by training on framewise pseudo-labels for 50 epochs. We set the hyperparameters α = 0.15, $\beta = 0 . 0 7 5$ , and $\gamma = 0 . 0 1$ . All experiments are conducted on a single V100 GPU.

## B. Quantitative comparisons

We quantitatively compare the proposed method with existing works from two perspectives: action segmentation performance and framewise pseudo-label generation performance.

Action segmentation performance. Table I compares the action segmentation performance against previous approaches, with the best results highlighted in bold. Since most existing timestamp-supervised methods utilize MS-TCN++ [20]–[22], [38], [44], [45] and ASFormer [19], [39] as network backbones, we implement the boundary voting network using both architectures for a fair comparison. The table is divided into two sections based on their respective backbones.

By leveraging the voting mechanism to address actiontransiting feature ambiguity and refine boundary localization, BVN achieves state-of-the-art results across three datasets under various evaluation metrics. For methods using the MS-TCN++ backbone, our approach demonstrates consistent performance improvements in framewise accuracy: +1.0% on the GTEA dataset, +1.4% on the 50Salads dataset, and +0.8% on the Breakfast dataset. With the ASFormer backbone, our method enhances framewise accuracy by +1.0% on the GTEA dataset, +1.2% on the 50Salads dataset, and +2.6% on the Breakfast dataset. In summary, our proposed model not only consistently outperforms other approaches but also exhibits robust performance with different network backbones.

Framewise pseudo-label generation performance. Precise boundary localization between annotated timestamps is crucial, as it directly influences the quality of the generated framewise pseudo-labels. To demonstrate the effectiveness of boundary localization during training, we present the framewise pseudo-label generation performance in Table II and compare it with the results reported by existing works. Following Table I, we divide Table II into two sections based on their respective backbones.

Without bells and whistles, our method significantly improves framewise pseudo-label generation results across all datasets, evaluation metrics, and backbones. For example, under the F1@50 metric, our method achieves a +7.5% improvement on the GTEA dataset with the MS-TCN++ backbone, and a +9.1% improvement on the 50Salads dataset with the ASFormer backbone. In summary, by reducing boundary localization uncertainty through the voting mechanism, we generate significantly higher-quality framewise pseudo-labels, which, in turn, enhance training stability and improve action segmentation testing performance in Table I.

TABLE III  
ACTION-TRANSITING FEATURES QUALITY RESULTS ON THE GTEA, 50SALADS, AND BREAKFAST DATASETS.
<table><tr><td>Methods</td><td>GTEA</td><td>50Salads</td><td>Breakfast</td></tr><tr><td>Li et al. [21]</td><td>0.69</td><td>0.75</td><td>0.67</td></tr><tr><td>Du et al. [44]</td><td>0.72</td><td>0.79</td><td>0.74</td></tr><tr><td>BVN (MS-TCN++)</td><td>0.79</td><td>0.86</td><td>0.81</td></tr><tr><td>BVN (ASFormer)</td><td>0.82</td><td>0.92</td><td>0.86</td></tr></table>

TABLE IV

INDEX OFFSETS QUALITY RESULTS ON THE GTEA, 50SALADS, ANDBREAKFAST DATASETS.
<table><tr><td>Methods</td><td>GTEA</td><td>50Salads</td><td>Breakfast</td></tr><tr><td>BVN (MS-TCN++)</td><td>0.07</td><td>0.06</td><td>0.04</td></tr><tr><td>BVN (ASFormer)</td><td>0.05</td><td>0.05</td><td>0.03</td></tr></table>

## C. Qualitative comparisons

We qualitatively compare the proposed method with existing works from two perspectives: action segmentation visualizations and boundary localization refinement visualizations. We further visualize the voting procedure for ambiguous action frames and the corresponding probability score improvements.

Action segmentation visualizations. Fig. 3 presents action segmentation visualizations from the methods of Li et al. [21], Du et al. [44], and our proposed BVN model, where distinct colors denote different action segments. Compared to existing methods, our method demonstrates superior segmentation performance by effectively capturing challenging actions missed by other methods and providing more precise boundary localization, particularly in long videos with complex actions.

Boundary localization refinement visualizations. We visualize the boundary localization refinement for framewise pseudo-label generation during training. In Fig. 4, we represent the sequence of the voting block, aggregation block, and decoder as a stage, and generate action-transiting regions on top of the video feature outputs from each stage. For simplicity and clarity, we select the video samples from the Breakfast dataset, which contains only a limited number of action segments. By applying the voting mechanism across the video, the action-transiting features integrate rich contextual knowledge and exhibit more discriminative representations, progressively suppressing the action-transiting regions and refining the boundary localization decisions.

Voting visualizations. To assess the concrete influence of voting mechanism, we visualize the voting procedure for ambiguous action frames and the corresponding probability score improvements. As shown in Fig. 5, our voting mechanism gathers essential global prior knowledge from frames depicting explicit “pour,” “cut tomato,” and “fry egg” actions and accurately directs it toward ambiguous frames in actiontransiting regions. Additionally, compared to the baseline without voting, aggregation module, and voting loss, BVN significantly improves the probability scores of frames with ambiguous visual appearance, demonstrating the effectiveness of voting in mitigating feature ambiguity.

TABLE V  
ACTION SEGMENTATION PERFORMANCE COMPARISONS WITHFULLY-SUPERVISED APPROACHES ON THE 50SALADS DATASET.
<table><tr><td></td><td>Methods</td><td>F1@ {10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td rowspan="3">Original</td><td>MS-TCN++ [31]</td><td>80.7</td><td>78.5 70.1</td><td>74.3</td><td>83.7</td></tr><tr><td>ASFormer [32]</td><td>85.1</td><td>83.4 76.0</td><td>76.9</td><td>85.6</td></tr><tr><td>ASPnet [57]</td><td>92.7 91.6</td><td>88.5 83.9</td><td>87.5 84.2</td><td>91.4 89.5</td></tr><tr><td rowspan="4">50%</td><td>BaFormer [58] MS-TCN++ [31]</td><td>89.3 70.5</td><td>88.4 64.4</td><td></td><td></td></tr><tr><td>ASFormer [32]</td><td>73.6 67.9</td><td>55.6 61.3</td><td>61.6 68.3</td><td>66.5 71.4</td></tr><tr><td>ASPnet [57]</td><td>76.6</td><td></td><td>68.7</td><td></td></tr><tr><td>BaFormer [58]</td><td>68.2 79.5 74.3</td><td>60.3 65.5</td><td>72.7</td><td>70.7 75.5</td></tr><tr><td>Timestamp</td><td>BVN (Ours)</td><td>85.1</td><td>82.5 72.3</td><td>79.0</td><td>81.2</td></tr></table>

TABLE VI  
COMPARISONS OF COMPUTATIONAL COMPLEXITY.
<table><tr><td>Methods</td><td>Parameter num</td><td>FLOPs</td><td>GPU usage</td></tr><tr><td>Li et al. [21]</td><td>1.2M</td><td>2.2G</td><td>7.6G</td></tr><tr><td>BVN (MS-TCN++)</td><td>1.4M</td><td>2.6G</td><td>9.9G</td></tr><tr><td>BVN (ASFormer)</td><td>1.9M</td><td>3.7G</td><td>15.5G</td></tr></table>

## D. Additional quantitative analysis

In addition to the quantitative comparisons presented in Table I and II, we provide further analysis from four key perspectives: the quality of action-transiting features, the quality of index offsets, comparisons with fully-supervised setting, and computational complexity. The additional analysis offer deeper insights into the voting mechanism and its contributions to the overall action segmentation performance.

Action-transiting features quality. As the first work analyzing the quality of action-transiting features, we propose the following evaluation pipeline. For each testing video and its ground truth labels, we identify the center frame of each ground truth action as the “key action frame” and collect 10 adjacent frames near the ground truth boundary as “ambiguous frames”. Using the trained network, we compute the framewise feature outputs from the final decoder and calculate the average cosine similarity between the “key action frame” and the “ambiguous frames”. As shown in Table III, our method significantly improves the feature similarity compared to existing methods, demonstrating the effectiveness of action-transiting feature enhancement.

Index offsets quality. To analyze the quality of the index offsets, we propose the following evaluation pipeline. For each testing video and its ground truth labels, we select the votes generated in the final voting block and calculate the average temporal distance between each vote and its corresponding ground truth boundary. As shown in Table IV, an average error of 0.03 on the Breakfast dataset for the ASFormer backbone indicates that for a 1000-frame video, the index error is approximately 30 frames from the boundary, suggesting that the majority of votes are concentrated around the action-transiting regions. Nevertheless, as we are the first to address timestampsupervised action segmentation from a voting perspective, no prior works are available for direct comparison.

Comparisons with fully-supervised methods. Labeling dense framewise annotations for a video is significantly more time-consuming than labeling sparse timestamps. To further ensure relatively fair comparisons between two distinct supervision settings, we instead evaluate action segmentation performance under comparable annotation time budgets. Since the ratio of timestamp-supervised videos to fully-supervised videos that can be labeled within the same time is not definitive and highly subjective, we employ an approximate comparison by reducing the number of fully-supervised training videos by half. Specifically, for fully-supervised methods, we randomly select 50% of the training videos for network learning while ensuring coverage of all action classes, and maintaining the rest of the experimental settings unchanged. As fully-supervised methods cannot utilize unlabeled videos, the obtained experimental results provide an approximate comparison with our method under an equivalent annotation time constraint.

TABLE VII  
ACTION SEGMENTATION PERFORMANCE COMPARED WITH THE BASELINE ON THE GTEA, 50SALADS, AND BREAKFAST DATASETS.
<table><tr><td></td><td colspan="5">GTEA</td><td colspan="5">50Salads</td><td colspan="5">Breakfast</td></tr><tr><td></td><td>F1@{10, 25, 50}</td><td></td><td></td><td>Edit</td><td>Acc</td><td>F1@{10, 25, 50}</td><td></td><td></td><td>Edit</td><td>Acc</td><td>F1@{10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td>Baseline</td><td>83.8</td><td>81.4</td><td>63.2</td><td>81.9</td><td>73.1</td><td>79.9 74.8</td><td>64.6</td><td>71.4</td><td>78.1</td><td>72.9</td><td>66.0</td><td>48.8</td><td>72.3</td><td>64.4</td></tr><tr><td>BVN (MS-TCN++)</td><td>86.2</td><td>84.9</td><td>68.2</td><td>84.7</td><td>75.5</td><td>81.4 76.8</td><td>67.8</td><td>73.5</td><td>80.1</td><td>74.4</td><td>68.0</td><td>50.4</td><td>74.0</td><td>67.8</td></tr><tr><td>Baseline</td><td>86.6</td><td>83.0</td><td>72.3</td><td>84.3</td><td>72.7</td><td>82.1 77.8</td><td>67.8</td><td>75.3</td><td>78.2</td><td>73.0</td><td>65.6</td><td>48.0</td><td>73.9</td><td>63.8</td></tr><tr><td>BVN (ASFormer)</td><td>91.7</td><td>90.5</td><td>76.8</td><td>89.9</td><td>76.7</td><td>85.1 82.5</td><td>72.3</td><td>79.0</td><td>81.2</td><td>77.3</td><td>69.7</td><td>51.2</td><td>76.5</td><td>68.3</td></tr></table>

TABLE VIII  
FRAMEWISE PSEUDO-LABEL GENERATION PERFORMANCE COMPARED WITH THE BASELINE ON THE GTEA DATASET.
<table><tr><td></td><td>F1@{10, 25, 50}</td><td>Acc</td></tr><tr><td>Baseline</td><td>97.0 94.9 77.0</td><td>77.6</td></tr><tr><td>BVN (ASFormer)</td><td>99.8 98.2 85.9</td><td>80.5</td></tr></table>

TABLE IX

EFFECT OF THE PROPOSED VOTING AND AGGREGATION BLOCKS ON THE GTEA DATASET.
<table><tr><td>MS-TCN++</td><td>F1@{10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td>Voting block only</td><td>83.0 81.0</td><td>62.7</td><td>80.6</td><td>72.4</td></tr><tr><td>Voting + Feature max pooling</td><td>83.9 82.8</td><td>65.3</td><td>82.4</td><td>73.6</td></tr><tr><td>Voting + Aggregation (average pooling)</td><td>85.7 84.5</td><td>67.3</td><td>84.3</td><td>75.2</td></tr><tr><td>Voting + Aggregation (max pooling) ASFormer</td><td>86.2 84.9</td><td>68.2</td><td>84.7</td><td>75.5</td></tr><tr><td>Voting block only</td><td>F1@{10, 25, 50} 81.9</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td></td><td>87.3</td><td>71.8</td><td>84.9</td><td>72.2</td></tr><tr><td>Voting + Feature max pooling</td><td>88.3 85.9</td><td>73.7</td><td>85.2</td><td>74.0</td></tr><tr><td>Voting + Aggregation (average pooling)</td><td>91.5 89.4</td><td>76.1</td><td>89.8</td><td>75.7</td></tr><tr><td>Voting + Aggregation (max pooling)</td><td>91.7 90.5</td><td>76.8</td><td>89.9</td><td>76.7</td></tr></table>

Table V presents the original results of fully-supervised methods alongside their performance when trained on only 50% of the videos. As the number of training videos decreases to 50%, fully-supervised methods exhibit significantly lower performance compared to our BVN under the timestampsupervised setting, where less than 0.5% of frames are annotated. The experimental results not only demonstrate the effectiveness of our approach against state-of-the-art methods that rely on dense framewise annotations, but also suggest that under a fixed and limited annotation budget, annotating more videos with sparse timestamp supervision may be more beneficial than annotating fewer videos with dense full supervision.

Computational complexity. Table VI presents the comparison of network parameters, FLOPs, and GPU memory usage between our proposed method and Li et al. [21], the pioneering approach for timestamp-supervised action segmentation. The

TABLE X  
EFFECT OF BLOCK NUMBERS ON THE GTEA DATASET.
<table><tr><td>Blocks num</td><td>F1@{10, 25, 50}</td><td>Edit</td><td>Acc</td></tr><tr><td>1</td><td>90.3 85.8 73.1</td><td>86.5</td><td>74.6</td></tr><tr><td>2</td><td>90.9 89.5 75.0</td><td>89.3</td><td>76.2</td></tr><tr><td>3</td><td>91.7 90.5 76.8</td><td>89.9</td><td>76.7</td></tr></table>

TABLE XI  
EFFECT OF start net AND end net ON THE GTEA DATASET.
<table><tr><td>Voting block</td><td>F1@{10, 25, 50}</td><td>Edit</td><td>Acc</td></tr><tr><td>start_net</td><td>90.1 87.7 75.0</td><td>87.6</td><td>75.5</td></tr><tr><td>end_net</td><td>90.0 86.5 74.8</td><td>88.5</td><td>75.9</td></tr><tr><td>Both</td><td>91.7 90.5 76.8</td><td>89.9</td><td>76.7</td></tr><tr><td>2 start_net</td><td>90.6 88.8 75.5</td><td>89.1</td><td>76.1</td></tr></table>

FLOPs measurements are conducted on a 2-minute video, while GPU memory usage is evaluated on the Breakfast dataset. In terms of training time, our method with ASFormer backbone requires approximately 20 hours to train on the Breakfast dataset, compared to 11 hours for Li et al. [21]. In summary, our approach achieves notable performance improvements at the expense of increased computational complexity. However, the additional cost introduced by the voting mechanism constitutes only a small fraction compared to the impact of changing the network backbone.

## E. Ablation studies

Baseline comparisons. To establish a valid baseline, we remove the proposed voting module, aggregation module, and voting loss $\mathcal { L } _ { v o t e }$ while keeping the rest of the network architecture unchanged, as shown in Table VII. Compared to the vanilla model, our proposed method effectively propagate and aggregate key action representations, mitigating feature ambiguity in action-transiting regions and improving overall action segmentation performance.

We further examine the framewise pseudo-label generation performance comparisons on the GTEA dataset. As shown in Table VIII, the proposed voting mechanism significantly improves the quality of generated framewise pseudo-labels compared to the baseline, demonstrating its effectiveness in refining boundary localization and mitigating action-transiting feature ambiguity.

Voting and aggregation blocks. Separately analyzing the performance impact of the proposed voting and aggregation blocks is challenging. First, retaining only the voting block while removing the aggregation block renders the generated votes ineffective, as they do not influence the original video features despite capturing key action representations. Second, isolating the aggregation block is infeasible, as it requires votes from the voting block as input.

TABLE XII  
EFFECT OF THE VOTING LOSS $\mathcal { L } _ { v o t e }$ ON THE GTEA, 50SALADS, AND BREAKFAST DATASETS.
<table><tr><td></td><td colspan="5">GTEA</td><td colspan="5">50Salads</td><td colspan="5">Breakfast</td></tr><tr><td>Settings</td><td colspan="3">F1@{10, 25, 50}</td><td>Edit</td><td>Acc</td><td colspan="2">F1@{10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td><td colspan="2">F1 @{10, 25, 50}</td><td></td><td>Edit</td><td>Acc</td></tr><tr><td>Baseline</td><td>86.6</td><td>83.0</td><td>72.3</td><td>84.3</td><td>72.7</td><td>82.1</td><td>77.8</td><td>67.8</td><td>75.3</td><td>78.2</td><td>73.0</td><td>65.6</td><td>48.0</td><td>73.9</td><td>63.8</td></tr><tr><td>Without  $\mathcal { L } _ { v o t e }$ </td><td>84.0</td><td>78.4</td><td>65.3</td><td>77.3</td><td>72.8</td><td>77.1</td><td>74.0</td><td>62.7</td><td>65.6</td><td>77.7</td><td>70.6</td><td>61.4</td><td>44.0</td><td>69.3</td><td>62.8</td></tr><tr><td>With  $\mathcal { L } _ { v o t e }$ </td><td>91.7</td><td>90.5</td><td>76.8</td><td>89.9</td><td>76.7</td><td>85.1</td><td>82.5</td><td>72.3</td><td>79.0</td><td>81.2</td><td>77.3</td><td>69.7</td><td>51.2</td><td>76.5</td><td>68.3</td></tr></table>

TABLE XIII  
EFFECT OF HYPER-PARAMETERS α, β, AND γ ON THE GTEA DATASET.
<table><tr><td rowspan=1 colspan=1>α</td><td rowspan=1 colspan=1>F1@ {10, 25, 50}</td><td rowspan=1 colspan=1>Edit</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>89.2 88.2 74.4</td><td rowspan=1 colspan=1>88.4</td><td rowspan=2 colspan=1>75.075.676.775.2</td></tr><tr><td rowspan=1 colspan=1>0.10.150.25</td><td rowspan=1 colspan=1>90.8 89.5 75.491.7 90.5 76.890.2 89.0 75.4</td><td rowspan=1 colspan=1>88.789.989.1</td></tr><tr><td rowspan=1 colspan=1>β</td><td rowspan=1 colspan=1>F1@{10, 25, 50}</td><td rowspan=1 colspan=1>Edit</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1>89.0 86.4 73.6</td><td rowspan=1 colspan=1>87.1</td><td rowspan=2 colspan=1>74.075.276.776.3</td></tr><tr><td rowspan=1 colspan=1>0.050.0750.1</td><td rowspan=1 colspan=1>89.4 88.2 75.191.7 90.5 76.891.4 90.7 76.0</td><td rowspan=1 colspan=1>89.089.989.4</td></tr><tr><td rowspan=1 colspan=1>γ</td><td rowspan=1 colspan=1>F1@{10, 25, 50}</td><td rowspan=1 colspan=1>Edit</td><td rowspan=1 colspan=1>Acc</td></tr><tr><td rowspan=1 colspan=1>0.0050.010.0150.02</td><td rowspan=1 colspan=1>88.6 86.8 73.591.7 90.5 76.891.5 89.6 77.290.6 89.0 76.0</td><td rowspan=1 colspan=1>86.389.988.688.2</td><td rowspan=1 colspan=1>74.176.776.475.8</td></tr></table>

TABLE XIV  
EFFECT OF DISTANCE THRESHOLD E ON THE GTEA DATASET.
<table><tr><td>E</td><td>F1@{10, 25, 50}</td><td>Edit</td><td>Acc</td></tr><tr><td>0</td><td>89.1 87.4 74.3</td><td>87.7</td><td>74.7</td></tr><tr><td>0.05</td><td>90.6 89.7 76.1</td><td>88.3</td><td>76.2</td></tr><tr><td>0.1</td><td>91.7 90.5 76.8</td><td>89.9</td><td>76.7</td></tr><tr><td>0.15</td><td>91.4 90.1 76.2</td><td>89.6</td><td>76.5</td></tr></table>

To the best of our knowledge, we conduct ablation studies with the voting block alone and assess the impact of the aggregation block by evaluating different aggregation approaches. As shown in Table IX, applying only the voting block results in action segmentation performance comparable to the baseline. Regarding different aggregation strategies, our current design, which combines MLP and max pooling, significantly outperforms direct max pooling of vote features and proves to be more effective than average pooling.

Number of blocks. Our method sequentially processes video features through multiple stages, including the voting block, aggregation block, and decoder. To further demonstrate the effectiveness of the proposed blocks, we maintain the decoder number at 3 while varying the number of voting and aggregation blocks to 1, 2, and 3. We incorporate these blocks before the final decoder for configuration 1 and before the second and final decoders for configuration 2. As shown in Table X, BVN achieves optimal performance when both the voting and aggregation blocks are integrated before every decoder, demonstrating the necessity of hierarchical global prior knowledge propagation.

start net and end net. Within each voting block, we employ start net and end net to direct votes towards the action-transiting regions before and after the action. To evaluate the effectiveness of these two independent networks, Table XI presents the action segmentation results using start net only, end net only, and both two deep networks in the voting block. Despite the notable performance achieved by each network, the combination of both start net and end net further significantly enhances the overall performance. Drawing upon the current experimental findings, no conclusive evidence indicates whether start net or end net exhibits superior performance when presented individually.

EFFECT OF KEY VOTE GROUP NUMBERS ON THE GTEA DATASET.  
TABLE XV
<table><tr><td>Groups num</td><td>F1@{10, 25, 50}</td><td>Edit</td><td></td><td>Acc</td></tr><tr><td>4</td><td>88.6 85.3</td><td>74.5</td><td>85.8</td><td>73.3</td></tr><tr><td>8</td><td>89.7 87.5</td><td>75.5</td><td>87.0</td><td>75.9</td></tr><tr><td>16</td><td>91.7 90.5</td><td>76.8</td><td>89.9</td><td>76.7</td></tr><tr><td>32</td><td>88.2 85.6</td><td>74.7</td><td>85.3</td><td>74.2</td></tr></table>

TABLE XVI  
EFFECT OF VARYING ACTION LENGTH ON THE GTEA DATASET.
<table><tr><td>Sampling Scales</td><td>F1@{10, 25, 50}</td><td>Edit</td><td>Acc</td></tr><tr><td>×0.25</td><td>91.8 90.6 76.7</td><td>89.3</td><td>76.5</td></tr><tr><td>×0.5</td><td>91.6 90.4 76.5</td><td>89.6</td><td>76.4</td></tr><tr><td>×1 (Original)</td><td>91.7 90.5 76.8</td><td>89.9</td><td>76.7</td></tr><tr><td>×2</td><td>91.4 90.3 76.3</td><td>89.5</td><td>76.4</td></tr></table>

It is worth noting that incorporating both start net and end net inevitably increases the model parameters compared to applying a single module. To mitigate this effect, we duplicate two identical start net modules, with the results shown in Table XI. While the inclusion of two start net modules yields a slight performance improvement, BVN achieves state-of-the-art results when both start net and end net are integrated. In summary, by voting ahead and behind into the action-transiting regions, frames adjacent to the boundary obtain global prior knowledge from different perspectives, thereby collaboratively contributing to effective video frame representation learning.

Voting loss. The voting loss, $\mathcal { L } _ { v o t e } ,$ plays a crucial role in the network’s learning process. To evaluate its effectiveness, we report results for the “Without $\mathcal { L } _ { v o t e } ? { \boldsymbol { \mathbf { \mathit { z } } } }$ setting in Table XII, where $\mathcal { L } _ { v o t e }$ is excluded from the final loss function. We employ the same baseline setting as in Table VII.

When the voting loss is omitted during training, performance declines significantly across all datasets as expected. The deterioration occurs because the absence of $\mathcal { L } _ { v o t e }$ disrupts the voting block’s ability to propagate global prior knowledge effectively into action-transiting regions. Without guidance, the voting process becomes random, invalidating the subsequent aggregation step. As a result, the absence of the voting loss introduces noise into the original video features, impairing the learning process compared to the baseline results.

$\alpha , \beta ,$ and $\gamma .$ The hyperparameters $\alpha , \beta ,$ , and γ control the scaling of $\mathcal { L } _ { s m o o t h } , \mathcal { L } _ { c o n f }$ , and $\mathcal { L } _ { v o t e }$ in the loss function. As shown in Table XIII, the optimal performance is achieved with $\alpha ~ = ~ 0 . 1 5 , ~ \beta ~ = ~ 0 . 0 7 5$ , and $\gamma ~ = ~ 0 . 0 1$ , with the numerical selection of hyperparameters primarily based on empirical observations. For the experimental results on each hyperparameter, the remaining hyperparameters are set to their default optimal values.

Distance threshold. The distance threshold E impacts the overall model performance. As shown in Table XIV, an excessively low threshold may fail to propagate sufficient global prior knowledge from representative votes to actiontransiting regions, whereas a high threshold may introduce irrelevant votes and noise, both of which lead to performance degradation. A threshold of 0 indicates that only knowledge from key votes is aggregated into the original video features.

Number of key vote groups. The number of key vote groups within the aggregation block, denoted as $P ,$ impacts the overall model performance. When aggregating all votes, an insufficient value of P fails to cover action-transiting regions throughout the video. In contrast, an excessively large value of $P$ may tend to introduce noises during the global prior knowledge propagation procedure. Consequently, we investigate the influence of the group number $P$ on the model performance in Table XV. The increase of $P$ in the early stages results in a continuous enhancement of our model performance on the GTEA dataset. However, raising $P$ from 16 to 32 leads to performance deterioration, attributed to the inclusion of excessive noises within key vote groups beyond the actiontransiting regions.

Varying action lengths. To evaluate the generalization capability for actions of varying lengths, we modify the testing video dataset by downsampling and upsampling the video features at different scales. For instance, the ×0.5 downsampling in Table XVI refers to selecting one frame out of every two consecutive frames, effectively halving the action length while preserving the content. Conversely, the $\times 2$ upsampling interpolates an additional frame between every two consecutive frames, doubling the action length. By testing the trained model on videos with varying sampling scales, we assess its generalization capability to actions of different lengths while ensuring a fair comparison. As shown in Table XVI, our proposed method maintains reliable performance across action lengths ranging from 0.25 to 2 times the original scale. In summary, our method demonstrates strong generalization capability for actions of varying lengths.

## V. CONCLUSIONS

In this paper, we have introduced a global-to-local boundary voting network, named BVN, for timestamp-supervised action segmentation. The BVN proposes a voting mechanism by rearranging the globally evenly distributed temporal feature series and aggregating key action representations into local action-transiting regions. Through hierarchical propagation,

BVN continuously addresses inherent action-transiting feature ambiguity and suppresses the ambiguous regions, leading to refined boundary localization, which proves to be essential for segmentation model training. With extensive experimental results on three public datasets, BVN has achieved state-ofthe-art performance on the action segmentation task across multiple evaluation metrics.

## VI. ACKNOWLEDGEMENT

This work was supported in part by the National Science Foundation of China under Grant 62206147, and by the National Research Foundation, Singapore, under the NRF Medium-Sized Centre Scheme (CARTIN). Any opinions, findings, and conclusions in this material are those of the authors and do not reflect the views of the National Science Foundation of China or the National Research Foundation, Singapore.

## REFERENCES

[1] Y.-H. Li, K.-Y. Liu, S.-L. Liu, L. Feng, and H. Qiao, “Involving distinguished temporal graph convolutional networks for skeleton-based temporal action segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 1, pp. 647–660, 2023.

[2] G. Li, D. Cheng, N. Wang, J. Li, and X. Gao, “Neighbor-guided pseudolabel generation and refinement for single-frame supervised temporal action localization,” IEEE Transactions on Image Processing, 2024.

[3] D. Cheng, Y. Ji, D. Gong, Y. Li, N. Wang, J. Han, and D. Zhang, “Continual all-in-one adverse weather removal with knowledge replay on a unified network structure,” IEEE Transactions on Multimedia, 2024.

[4] D. Cheng, Y. Hu, N. Wang, D. Zhang, and X. Gao, “Achieving plasticitystability trade-off in continual learning through adaptive orthogonal projection,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[5] Y. Cui, G. Tian, Z. Jiang, M. Zhang, Y. Gu, and Y. Wang, “An active task cognition method for home service robot using multi-graph attention fusion mechanism,” IEEE Transactions on Circuits and Systems for Video Technology, 2023.

[6] C. Cai, Z. Wang, J. Gao, W. Liu, Y. Lu, R. Zhang, and K.-H. Yap, “Empowering large language model for continual video question answering with collaborative prompting,” arXiv preprint arXiv:2410.00771, 2024.

[7] R. Zhang, S. Wang, Y. Duan, Y. Tang, Y. Zhang, and Y.-P. Tan, “Hoiaware adaptive network for weakly-supervised action segmentation,” in Proceedings of the International Joint Conference on Artificial Intelligence, 2023, pp. 1722–1730.

[8] D. Yang, Y. Zou, C. Zhang, M. Cao, and J. Chen, “Rr-net: Relation reasoning for end-to-end human-object interaction detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 6, pp. 3853–3865, 2021.

[9] N. Wang, G. Zhu, H. Li, M. Feng, X. Zhao, L. Ni, P. Shen, L. Mei, and L. Zhang, “Exploring spatio–temporal graph convolution for video-based human–object interaction recognition,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 10, pp. 5814–5827, 2023.

[10] L. Yan, S. Ma, Q. Wang, Y. Chen, X. Zhang, A. Savakis, and D. Liu, “Video captioning using global-local representation,” IEEE Transactions on Circuits and Systemsfor Video Technology, vol. 32, no. 10, pp. 6642– 6656, 2022.

[11] T. Wang, H. Zheng, M. Yu, Q. Tian, and H. Hu, “Event-centric hierarchical representation for dense video captioning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 5, pp. 1890– 1900, 2020.

[12] M. Qi, Y. Wang, A. Li, and J. Luo, “Sports video captioning via attentive motion representation and group relationship modeling,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 8, pp. 2617–2633, 2019.

[13] C. Cai, R. Zhang, J. Gao, K. Wu, K.-H. Yap, and Y. Wang, “Temporal sentence grounding with temporally global textual knowledge,” in IEEE International Conference on Multimedia and Expo, 2024, pp. 1–6.

[14] W. Xu, Z. Miao, J. Yu, Y. Tian, L. Wan, and Q. Ji, “Bridging video and text: A two-step polishing transformer for video captioning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 9, pp. 6293–6307, 2022.

[15] R. Dai, S. Das, S. Sharma, L. Minciullo, L. Garattoni, F. Bremond, and G. Francesca, “Toyota smarthome untrimmed: Real-world untrimmed videos for activity detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 2533–2550, 2022.

[16] T. Wang and D. J. Cook, “smrt: Multi-resident tracking in smart homes with sensor vectorization,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 8, pp. 2809–2821, 2020.

[17] P. Bao, W. Yang, B. P. Ng, M. H. Er, and A. C. Kot, “Cross-modal label contrastive learning for unsupervised audio-visual event localization,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp. 215–222.

[18] P. Bao, Z. Shao, W. Yang, B. P. Ng, M. H. Er, and A. C. Kot, “Omnipotent distillation with llms for weakly-supervised natural language video localization: When divergence meets consistency,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 2, 2024, pp. 747–755.

[19] N. Behrmann, S. A. Golestaneh, Z. Kolter, J. Gall, and M. Noroozi, “Unified fully and timestamp supervised temporal action segmentation via sequence to sequence translation,” in Proceedings of the European Conference on Computer Vision, 2022, pp. 52–68.

[20] H. Khan, S. Haresh, A. Ahmed, S. Siddiqui, A. Konin, M. Z. Zia, and Q.-H. Tran, “Timestamp-supervised action segmentation with graph convolutional networks,” in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022, pp. 10 619–10 626.

[21] Z. Li, Y. Abu Farha, and J. Gall, “Temporal action segmentation from timestamp supervision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 8365–8374.

[22] Y. Zhao and Y. Song, “Turning to a teacher for timestamp supervised temporal action segmentation,” in Proceedings of the IEEE International Conference on Multimedia and Expo, 2022, pp. 01–06.

[23] S. Karaman, L. Seidenari, and A. Del Bimbo, “Fast saliency based pooling of fisher encoded dense trajectories,” in Proceedings of the European Conference on Computer Vision THUMOS Workshop, 2014, p. 5.

[24] M. Rohrbach, S. Amin, M. Andriluka, and B. Schiele, “A database for fine grained activity detection of cooking activities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2012, pp. 1194–1201.

[25] C. Lea, A. Reiter, R. Vidal, and G. D. Hager, “Segmental spatiotemporal cnns for fine-grained action segmentation,” in Proceedings of the European Conference on Computer Vision, 2016, pp. 36–52.

[26] S. Yeung, O. Russakovsky, N. Jin, M. Andriluka, G. Mori, and L. Fei-Fei, “Every moment counts: Dense detailed labeling of actions in complex videos,” International Journal of Computer Vision, vol. 126, pp. 375–389, 2018.

[27] C. Lea, M. D. Flynn, R. Vidal, A. Reiter, and G. D. Hager, “Temporal convolutional networks for action segmentation and detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 156–165.

[28] L. Ding and C. Xu, “Tricornet: A hybrid temporal convolutional and recurrent network for video action segmentation,” arXiv preprint arXiv:1705.07818, 2017.

[29] P. Lei and S. Todorovic, “Temporal deformable residual networks for action segmentation in videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 6742–6751.

[30] Y. A. Farha and J. Gall, “Ms-tcn: Multi-stage temporal convolutional network for action segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 3575–3584.

[31] S.-J. Li, Y. AbuFarha, Y. Liu, M.-M. Cheng, and J. Gall, “Ms-tcn++: Multi-stage temporal convolutional network for action segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020.

[32] F. Yi, H. Wen, and T. Jiang, “Asformer: Transformer for action segmentation,” arXiv preprint arXiv:2110.08568, 2021.

[33] M. Li, L. Chen, Y. Duan, Z. Hu, J. Feng, J. Zhou, and J. Lu, “Bridgeprompt: Towards ordinal action understanding in instructional videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 19 880–19 889.

[34] D. Liu, Q. Li, A.-D. Dinh, T. Jiang, M. Shah, and C. Xu, “Diffusion action segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 139–10 149.

[35] Z. Lu and E. Elhamifar, “Fact: Frame-action cross-attention temporal modeling for efficient action segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 18 175–18 185.

[36] A. Bearman, O. Russakovsky, V. Ferrari, and L. Fei-Fei, “What’s the point: Semantic segmentation with point supervision,” in Proceedings of the European Conference on Computer Vision, 2016, pp. 549–565.

[37] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[38] R. Rahaman, D. Singhania, A. Thiery, and A. Yao, “A generalized and robust framework for timestamp supervision in temporal action segmentation,” in Proceedings ofthe European Conference on Computer Vision, 2022, pp. 279–296.

[39] K. Liu, Y. Li, S. Liu, C. Tan, and Z. Shao, “Reducing the label bias for timestamp supervised temporal action segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6503–6513.

[40] B. Leibe, A. Leonardis, and B. Schiele, “Robust object detection with interleaved categorization and segmentation,” International Journal of Computer Vision, vol. 77, pp. 259–289, 2008.

[41] A. Vaswani, “Attention is all you need,” Advances in Neural Information Processing Systems, 2017.

[42] Y. Zheng, Y. Duan, Z. Li, J. Zhou, and J. Lu, “Learning dynamic sceneconditioned 3d object detectors,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 2981–2996, 2023.

[43] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[44] D. Du, E. Li, L. Si, F. Xu, and F. Sun, “Timestamp-supervised action segmentation in the perspective of clustering,” arXiv preprint arXiv:2212.11694, 2022.

[45] R. Hirsch, R. Cohen, T. Golany, D. Freedman, and E. Rivlin, “Random walks for temporal action segmentation with timestamp supervision,” in Proceedings of the IEEE Winter Conference on Applications of Computer Vision, 2024, pp. 6614–6624.

[46] S. Sayed, R. Ghoddoosian, B. Trivedi, and V. Athitsos, “A new dataset and approach for timestamp supervised action segmentation using human object interaction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 3133–3142.

[47] C. Patsch, Y. Wu, D. Salihu, M. Zakour, and E. Steinbach, “Tscl: Timestamp supervised contrastive learning for action segmentation,” IEEE Robotics and Automation Letters, 2024.

[48] F. Yang, S. Odashima, S. Masui, and S. Jiang, “Is weakly-supervised action segmentation ready for human-robot interaction? no, let’s improve it with action-union learning,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2023, pp. 9800–9807.

[49] M. Zou, Q. Zeng, and X. Zhang, “Weakly-supervised action learning in procedural task videos via process knowledge decomposition,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[50] L. Hao, Y. Hu, Y. Yue, L. Wu, H. Fu, J. Duan, and J. Liu, “Hierarchical context transformer for multi-level semantic scene understanding,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[51] J.-R. Du, J.-C. Feng, K.-Y. Lin, F.-T. Hong, Z. Qi, Y. Shan, J.-F. Hu, and W.-S. Zheng, “Weakly-supervised temporal action localization by progressive complementary learning,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[52] Y. Shao, F. Zhang, and C. Xu, “Text-video knowledge guided prompting for weakly supervised temporal action localization,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[53] A. Fathi, X. Ren, and J. M. Rehg, “Learning to recognize objects in egocentric activities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2011, pp. 3281–3288.

[54] S. Stein and S. J. McKenna, “Combining embedded accelerometers with computer vision for recognizing food preparation activities,” in Proceedings of the ACM International Joint Conference on Pervasive and Ubiquitous Computing, 2013, pp. 729–738.

[55] H. Kuehne, A. Arslan, and T. Serre, “The language of actions: Recovering the syntax and semantics of goal-directed human activities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2014, pp. 780–787.

[56] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017, pp. 6299–6308.

[57] B. van Amsterdam, A. Kadkhodamohammadi, I. Luengo, and D. Stoyanov, “Aspnet: Action segmentation with shared-private representation of multiple data sources,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2384–2393.

[58] P. Wang, Y. Lin, E. Blasch, J. Wei, and H. Ling, “Efficient temporal action segmentation via boundary-aware query voting,” arXiv preprint arXiv:2405.15995, 2024.

![](images/f0d90cf52965c2e970633c3f0878e4e155b84a0ce920079f0036e33f65947832.jpg)

Runzhong Zhang is a PhD candidate at the School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore. He obtained his B.S. degree from Xi’an Jiaotong University in 2018 and his M.S. degree from Columbia University in 2020. His current research interests include video understanding, human action analysis, and labelefficient learning.

![](images/5f753520213248854b0b3b68b32f7d5ffb00bb0e6c0b16ee11d911ba4945592a.jpg)

Chen Cai received his B.S. degree in Electrical and Computer Engineering from the National University of Singapore. He is currently a Ph.D. candidate at the School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore. His research interests include computer vision, multimodal learning, and machine learning.

![](images/666b2a16385512d5377d0b80e3fc0acc1463a14f7984be6186d1dd9fd0ec31e7.jpg)

Yueqi Duan (Member, IEEE) received the B.S. and Ph.D. degrees from the Department of Automation, Tsinghua University, in 2014 and 2019, respectively. He is currently an Assistant Professor with the Department of Electronic Engineering, Tsinghua University. He has published more than 30 scientific papers in the top journals and conferences, including IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTEL-LIGENCE, IEEE TRANSACTIONS ON IMAGE PROCESSING, CVPR, ICCV, ECCV and NeurIPS.

His research interests include computer vision and pattern recognition. He served as the Publication Chair for FG, the Area Chair for CVPR, ICLR, MM and ICME, and a Regular Reviewer for a number of journals and conferences, e.g., IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, IEEE TRANSACTIONS ON IMAGE PRO-CESSING, IJCV, CVPR, ICCV, ECCV, ICML, NeurIPS, and SIGGRAPH. He was awarded the Excellent Doctoral Dissertation of Chinese Association for Artificial Intelligence (CAAI) in 2020.

![](images/926229f45bae22c5eec844665705d3ab76c4e4a31afe8173d37824c5d3514c61.jpg)

Suchen Wang received his Ph.D. degree from the School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore, in 2022. He is an applied scientist at Amazon, Seattle, US. His current research interests include humanobject interaction and video understanding.

![](images/38975e36d95b34171f422eee2757f32b1d0f1d6bf627bb9559e454a9eef6d7ff.jpg)  
Yang Chen (Graduate Student Member, IEEE) received the Master’s degree from the University of Michigan, Ann Arbor, the United States, in 2022. She is currently a Ph.D. candidate at the School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore. Her current research interest is 3D vision.

![](images/afe04a55439b4cdd21e94113065835c591bb72bc770a4bd913a79204064079ce.jpg)

Yap-Peng Tan (Fellow, IEEE) received the B.S. degree from National Taiwan University, Taipei, Taiwan, in 1993, and the M.A. and Ph.D. degrees from Princeton University, Princeton, NJ, in 1995 and 1997, respectively, all in electrical engineering. He is currently a Professor and Associate Vice President at Nanyang Technological University (NTU), Singapore. His research interests include image and video processing, machine learning, computer vision, and data analytics. He served as an Associate Editor of the IEEE TRANSACTIONS ON CIRCUITS AND

![](images/57b33acc435e3fde98b74514bb0bf87310593815aab301e4bf56627111de083b.jpg)

Weipeng Hu received the Ph.D. degree in Electronics and Information Technology from Sun Yatsen University, Guangzhou, China, in 2022. He is currently a Research Fellow with the Centre for Advanced Robotics Technology Innovation Laboratory, School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore. His current research interests include image & video synthesis, human-robot interaction, heterogeneous face recognition, and cross-domain person ReID.

SYSTEMS FOR VIDEO TECHNOLOGY, IEEE SIGNAL PROCESSING LETTERS, IEEE TRANSACTIONS ON MULTIMEDIA, and IEEE Access, as well as an Editorial Board Member of the EURASIP Journal on Advances in Signal Processing and EURASIP Journal on Image and Video Processing. He was the Technical Program Co-Chair of the 2015 IEEE International Conference on Multimedia and Expo (ICME 2015) and the 2019 IEEE International Conference on Image Processing (ICIP 2019), and the General Co-Chair of the 2010 IEEE International Conference on Multimedia and Expo (ICME 2010) and the 2015 IEEE International Conference on Visual Communications and Image Processing (VCIP 2015).