# CoDy: Codec Dynamics as Complementary Priors for Unsupervised Temporal Action Localization

Tianyu Yan<sup>1</sup> Qingyi Yang<sup>1</sup> Qin Yang<sup>1</sup> Xufei Guo<sup>1</sup> Yuan Tang<sup>2</sup> Rui Wang<sup>1</sup> Yixue Hao<sup>1</sup> Long Hu<sup>1</sup>

<sup>1</sup>School of Computer Science and Technology, Huazhong University of Science and Technology <sup>2</sup>Huawei Technologies Ltd

tianyuyan@hust.edu.cn

## Abstract

Unsupervised temporal action localization (UTAL) aims to identify action categories and localize their temporal boundaries without annotation supervision. Existing methods typically derive supervision for both tasks solely from RGB and optical-flow representations, often resulting in noisy category assignments and inaccurate temporal boundaries. In this work, we reveal an overlooked property of video codecs: group-of-pictures quantization patterns preserve category-discriminative dynamics, while motion-vector and residual variations exhibit pronounced responses around action transitions. Based on this insight, wepropose CoDy, a codec dynamics-guidedframework that exploits the codec as complementary information to the visual view for UTAL. CoDy integrates a codec-enhanced view with RGB andflow through confidence-aware consensus clustering, producing reliable soft pseudo-labels. Itfurther converts codec dynamics into a class-agnostic boundary prior, which supervises visual boundary learning and guides proposal initialization and iterative refinement. The codec branches are only used during the training stage and introduce no extra codec-side computation at inference. Extensive experiments on THUMOS14 and ActivityNet v1.3 demonstrate that CoDy consistently outperforms existing UTAL methods and achieves state-of-the-art localization performance. These results establish codec signals as effective annotation-free priorsforjointly inferring which actions occur and when they occur.

## 1. Introduction

Temporal action localization (TAL) aims to identify the actions occurring in an untrimmed video and determine the temporal boundaries of each action instance [18, 28]. Fully supervised methods learn this task from action-category labels and temporal annotations [23, 28], but collecting temporal boundary annotations is expensive and inherently ambiguous [13]. Weakly supervised TAL reduces the required supervision to video-level action labels [14, 16, 27], whereas unsupervised temporal action localization (UTAL) removes these labels entirely and assumes only that the number of latent action categories is known [6, 24]. Without semantic or temporal annotations, UTAL needs to jointly discover latent action categories and localize their temporal extents. These two objectives are inherently coupled: category discovery requires foreground representations that are not dominated by background or incomplete action segments, while accurate localization requires reliable category semantics to distinguish actions from their surrounding context. Thus, there is a coupled challenge in UTAL between discovery and localization, in which errors from either side can be amplified.

![](images/909dec189fe4362379fdd6545b55b04c1b0ff02e037923f79b152ef0f4b4f79a.jpg)  
Figure 1. The codec view’s predictive dynamics provide comple mentary priors to correct errors in the visual view.

Existing UTAL methods typically address this problem through an iterative paradigm. Visual representations are iteratively clustered to generate pseudo-category labels, which guide temporal localization and the resulting action proposals are then fed back to refine the pseudo-category assignments for the next iteration [11]. Recent methods improve this process through cross-view collaboration [22], uncertainty-aware learning [24], and pseudo-label refinement [7]. Nevertheless, both latent category discovery and temporal boundary estimation still rely predominantly on decoded RGB and optical-flow representations. For action category discovery, visual view captures appearance and apparent motion. However, RGB features often entangle actions with scene context, while optical flow is vulnerable to camera motion, weakening the category structure required for clustering. For action temporal localization, learned representations tend to emphasize discriminative action interiors rather than temporal transitions, leading to incomplete or over-extended boundaries. Consequently, the two stages inherit correlated errors from the same visual evidence, which may be reinforced rather than corrected during iteration. It requires complementary evidence to break this loop that can recognize latent actions while exposing their temporal transitions.

We find that the predictive structure of video coding provides a promising source of such evidence [10, 20, 21]. A video codec organizes frames into groups of pictures (GOPs) and represents temporal evolution through motioncompensated prediction and residual correction. Motion vectors describe predictable displacement, while quantized residuals retain variations unexplained by prediction [10, 20]. RGB and optical flow describe the observed visual states, whereas codec signals record how these states are predicted and corrected over time, providing a distinct view of temporal dynamics. Previous compressed-domain studies have demonstrated that codec primitives carry learnable video information [3, 4, 21]. In contrast, we investigate whether the predictive structure preserved by video coding can serve as an intrinsic prior for UTAL. Our empirical analysis reveals two complementary properties: GOPlevel quantization patterns preserve category-discriminative dynamics complementary to RGB and flow, while variations in motion vectors and quantized residuals exhibit pronounced responses around action transitions. Although these signals do not directly specify action categories or exact temporal boundaries, they provide annotation-free priors for discovering what actions occur and determining when they occur.

Building on these findings, we propose CoDy, a codec dynamics-guided framework that complements decoded visual representations with predictive dynamics for UTAL. As illustrated in Figure 1, RGB and flow provide the primary visual evidence for action discovery and localization, while codec signals introduce complementary structural priors. For category discovery, CoDy constructs a GOP-level codec dynamics view. It aligns it with RGB and optical flow through confidence-aware multi-view consensus clustering, yielding reliable soft pseudo-labels. Based on these labels, the motion-vector and quantized-residual variations provide class-agnostic boundary priors for multi-scale temporal refinement. Codec-derived proposals further optimize the loop with more reliable action candidates, which are subsequently fed back to refine the multi-view category assignments. Codec information is required only during training and inference retains the learned localization network and boundary branch without additional codec processing. Extensive experiments on THUMOS14 and ActivityNet v1.3 establish new state-of-the-art performance, with particularly pronounced gains at high IoU thresholds. Our contributions are summarized as follows:

• We revisit predictive video coding and empirically reveal that GOP-level quantization preserves categorydiscriminative dynamics, while motion-vector and residual variations provide cues to action transitions.

• We propose CoDy, a codec-prior UTAL framework that translates these complementary structures into pseudolabel generation and boundary prior learning, thereby stabilizing the coupled discovery and localization.

• Extensive experiments on THUMOS14 and ActivityNet v1.3 establish new state-of-the-art performance, with pronounced improvements at high IoU thresholds and validate the effectiveness and generality of codec priors.

## 2. Related Work

## 2.1. Unsupervised Temporal Action Localization

UTAL aims to discover latent action categories and localize their temporal intervals in untrimmed videos without any human-provided annotations. TCAM [6] establishes the clustering-and-updating paradigm for UTAL, where videos are iteratively clustered into latent categories and the resulting assignments are used as pseudo supervision for localization learning. Subsequent works have improved this framework from different perspectives. UGCT [24] leverages RGB–flow collaboration and uncertainty estimation to suppress noisy pseudo supervision. APSL [11] focuses on separating action-positive evidence from background content, while FEEL [7] improves category discovery through enhanced clustering and self-paced instance selection. More recently, CLIP-AE incorporates visual-language and audio modalities as complementary views for action discovery [22]. BID explores boundary–interior structural learning for skeleton-based unsupervised temporal action pretraining [5]. Most UTAL methods iteratively cluster video or segment representations to infer latent categories, then use the resulting pseudo-labels to train a localization model and update the representations for subsequent clustering. However, background content, intra-class variation, and similar actions can yield noisy category assignments, while unreliable pseudo-labels provide weak localization supervision. These errors may accumulate across iterations, making robust category discovery and boundary localization the key challenges of UTAL.

![](images/22b31533d9e7fc54f5704dd02fe7821bcc46a693ac96a4372a1505bc30d03317.jpg)  
(a) Category clustering based on visual view (Left) and complementary codec view (Right).

![](images/fa4a76572b799969b89ee0648ec9f2f401a602aea368a372b673bcaa22833987.jpg)  
(b) Response-boundary relationship.  
Figure 2. Empirical results of codec-prior dynamics for category discrimination and boundary sensitivity detection.

## 2.2. Codec Representations for Video Understanding

Modern video codecs reduce temporal and spatial redundancy by encoding videos with intra-coded frames, motioncompensated prediction, and residual signals [10, 20]. Specifically, I-frames provide appearance references, motion vectors describe block-wise displacement for interframe prediction, and residuals preserve information that cannot be explained by the predicted content. These codec primitives are generated during video compression and can be accessed without fully decoding all RGB frames. Early works exploit compressed-domain representations primarily for efficient video recognition. CoViAR demonstrates that I-frames, motion vectors, and residuals can serve as complementary inputs for action recognition, avoiding costly dense RGB decoding [21]. Later, MM-ViT models interactions among these heterogeneous codec signals with a compressed-video transformer [3], while CoPE-VideoLM represents codec primitives as compact visual tokens for video-language modeling [4]. Overall, existing studies mainly treat codec information as an efficient alternative or complement to decoded visual features.

## 3. Empirical Analysis of Codec-Prior Dynamics

Video codecs exploit temporal redundancy by organizing a video into GOP. Each GOP begins with an intra-coded frame (I-frame), which serves as an appearance reference, followed by predictive frames (P-frames) encoded by motion vectors and residuals. Motion vectors describe the block-wise displacements selected for prediction, while residuals retain the visual changes that cannot be explained by the predicted motion. To obtain temporally aligned codec observations, we re-encode each video into fixedlength 0.5-second GOPs. The t-th GOP is represented as:

$$
\mathcal { G } _ { t } = \left( I _ { t } , \{ ( \mathbf { M } _ { t , j } , \mathbf { R } _ { t , j } ) \} _ { j = 1 } ^ { n _ { t } } \right) ,\tag{1}
$$

where $I _ { t }$ is the I-frame and $\mathbf { M } _ { t , j }$ and $\mathbf { R } _ { t , j }$ are the motionvector and residual fields of the j-th P-frame. Each GOP therefore yields a codec observation at a temporal resolution of 0.5 seconds, while preserving both the spatial structure and temporal evolution of the compressed-domain signals. As summarized in Figure 2, we evaluate whether these predictive dynamics provide useful cues for the two core stages of UTAL including category discovery and boundary localization. The following two observations evaluate these properties on THUMOS14. Detailed methodology and implementation details are provided in Appendix A.

## 3.1. Observation 1: Codec Dynamics Complement Category Discrimination

We examine whether codec information provides a useful complementary view for category discrimination. We conduct a controlled comparison using the same videos, clustering algorithm, number of clusters, and random seed. As illustrated in Figure 2(a), we compare action category clustering results of video-level RGB and optical-flow descriptors with their codec-augmented counterparts. The groundtruth labels are used to evaluate the quality of the generated pseudo-labels. Without iterative refinement, codec augmentation view increases pseudo-label accuracy from 60.0% to 63.5% and macro-F1 from 52.87% to 56.16%. It also reduces confusion between visually similar actions, particularly JavelinThrow and ThrowDiscus. This improvement can be attributed to the complementary information retained by the codec. RGB features mainly describe objects and scenes, while optical flow features describe apparent pixel displacement. In contrast, motion vectors encode the block-wise motion selected during video compression, and residuals record changes that cannot be explained by motion compensation. These results show that codec dynamics boosts category clustering and the reliability of initial pseudo-labels.

![](images/ddd009502fe7a0eaabfc2b18ba43e62a9ac3faece3f5817c826b53a7283d8805.jpg)  
Figure 3. Overview of the proposed CoDy. CoDy executes from two views of visual and complementary codec view and is composed o P-MIL-based baseline, codec-enhanced pseudo-label generation and codec-based boundary prior learning.

## 3.2. Observation 2: Codec Dynamics Exhibit Boundary Sensitivity

We investigate whether codec dynamics respond more strongly to action boundaries than RGB and flow. For each 0.5-second segment, we compute the normalized entropy of the RGB color distribution and the mean optical-flow magnitude as view-specific responses. The codec response aggregates the motion-vector magnitude and residual energy within each GOP:

$$
M _ { t } = \mathbb { E } [ \| \mathbf { M } _ { t } \| _ { 2 } ] , \qquad R _ { t } = \mathbb { E } \big [ \| \mathbf { R } _ { t } \| _ { 2 } ^ { 2 } \big ] ,\tag{2}
$$

where E denotes averaging over all spatial locations and $\mathrm { P } \mathrm { - }$ frames in a GOP, $M _ { t }$ measures motion intensity and $R _ { t }$ captures changes unexplained by motion compensation. The codec-based quantization response is computed as:

$$
q _ { t } ^ { c } = \mathcal { N } \Big ( \lambda _ { m } M _ { t } + \lambda _ { r } R _ { t } + \lambda _ { g } \sqrt { M _ { t } , R _ { t } } \Big ) ,\tag{3}
$$

where $\mathcal { N } ( \cdot )$ is per-video normalization, and $\lambda _ { m } , \lambda _ { r }$ and $\lambda _ { g }$ balance the two components. All responses are independently normalized, lightly smoothed, and aligned on the same temporal grid. Boundary annotations are used only for evaluation and do not participate in response construction. As shown in Figure 2(b), the codec response exhibits sharper and more consistent elevations around action boundaries than the RGB and flow responses. Their stronger boundary alignment demonstrates that codec dynamics provide a more sensitive, class-agnostic cue for action-transition localization.

## 4. Method

Let $\begin{array} { r c l } { \mathcal { D } } & { = } & { \{ V _ { i } \} _ { i = 1 } ^ { N } } \end{array}$ denote a collection of unlabeled, untrimmed videos, and let C be the number of latent action categories. Each video $V _ { i }$ is represented by concatenated RGB and flow I3D features $\mathbf { X } _ { i } \in \mathbb { R } ^ { T _ { i } \times D }$ [2], together with its codec observations $\{ \mathcal { G } _ { i , t } \} _ { t = 1 } ^ { T _ { i } }$ defined in Eq. (1). Given these inputs, the model predicts a set of action instances $\mathcal { P } = \{ ( \bar { s } _ { i } , e _ { i } , c _ { i } , q _ { i } ) \} _ { i = 1 } ^ { K }$ , where $s _ { i }$ and $e _ { i }$ denote the temporal boundaries, $c _ { i } \in \{ 1 , \ldots , C \}$ is the discovered action category, and $q _ { i }$ is the confidence score.

## 4.1. Framework Overview

As illustrated in Figure 3, CoDy builds upon the proposalbased localization framework P-MIL [16]. Each candidate proposal is encoded into a proposal-level representation, from which a classification branch predicts latent category probabilities and an attention branch measures proposal attention weight. On this baseline, CoDy introduces two codec-guided branches. First, codec-enhanced pseudolabel generation aggregates GOP features into a video-level codec-semantic view. It jointly clusters them with RGB and flow representations to generate soft pseudo-labels, which are iteratively refined to supervise latent category discovery. Second, codec-based boundary prior learning exploits GOP-level motion-vector and residual dynamics to construct a class-agnostic boundary prior, which initializes the candidate proposal set and supervises an RGB and flow multi-scale boundary branch. During training, proposal scores remove low-confidence candidates, while the learned boundary responses introduce new proposals, enabling iterative refinement of category predictions and temporal boundaries.

## 4.2. Codec-Enhanced Pseudo-Label Generation

CoDy constructs a video-level codec view using a codec encoder inspired from CoPE-VideoLM [4]. For the t-th GOP, the I-frame $I _ { t }$ is processed by an image encoder to produce eight appearance tokens $\dot { \mathbf { U } } _ { t } ^ { I } \in \mathbb { R } ^ { 8 \times d }$ . For each P-frame $j ,$ two lightweight encoders separately map its motion-vector and residual fields to four tokens, denoted by $\mathbf { U } _ { t , j } ^ { M } \in \mathbb { R } ^ { 4 \times d } , \mathbf { U } _ { t , j } ^ { R } \in \mathbb { R } ^ { 4 \times d }$ . The I-frame tokens provide an appearance anchor, while the motion-vector and residual tokens describe encoded motion and prediction errors. We concatenate them to form the representation of GOP t:

$$
\mathbf { G } _ { t } = \left[ \mathbf { U } _ { t } ^ { I } ; \mathbf { U } _ { t , 1 } ^ { M } ; \mathbf { U } _ { t , 1 } ^ { R } ; \ldots ; \mathbf { U } _ { t , n _ { t } } ^ { M } ; \mathbf { U } _ { t , n _ { t } } ^ { R } \right] .\tag{4}
$$

Thus, a GOP containing $n _ { t }$ P-frames is represented by $8 + 8 n _ { t }$ codec tokens. After adding token-type and temporal positional embeddings, the GOP sequence is directly fed into a temporal Transformer. A learnable video token aggregates the sequence, yielding the video-level codec representation:

$$
\mathbf { z } _ { i } ^ { c } = \operatorname { T r a n s f o r m e r } ( [ \operatorname { C L S } ] , \mathbf { G } _ { i , 1 } , \dots , \mathbf { G } _ { i , T _ { i } } ) _ { \operatorname { C L S } } .\tag{5}
$$

To construct the visual view, CoDy obtains proposallevel features and attention weights from the baseline model. For video $i ,$ let $\mathbf { h } _ { i r }$ and $a _ { i r }$ denote the feature and attention weight of its r-th proposal, respectively. The proposal features are aggregated according to the attention weights to form a single video-level visual representation:

$$
\mathbf { z } _ { i } ^ { v } = \sum _ { r = 1 } ^ { R _ { i } } a _ { i r } \mathbf { h } _ { i r } , \qquad \sum _ { r = 1 } ^ { R _ { i } } a _ { i r } = 1 ,\tag{6}
$$

where $R _ { i }$ is the number of proposals in video $i .$ This aggregation ensures that proposals more likely to contain actions contribute more strongly to the visual representation. The visual representation $\mathbf { z } _ { i } ^ { v }$ and the codec representation $\mathbf { z } _ { i } ^ { c }$ are independently normalized and used as two views for category clustering. While the visual view captures high-level appearance and motion semantics, the codec view provides compressed-domain motion and prediction information.

Then, CoDy performs confidence-aware consensus clustering over the two normalized views. For each video, $\mathbf { z } _ { i } ^ { v }$ and $\mathbf { z } _ { i } ^ { c }$ independently compute similarities to the cluster centers, producing view-specific soft assignments $\mathbf { p } _ { i } ^ { v }$ and $\mathbf { p } _ { i } ^ { c }$ . The confidence of each view is determined by the $\ell _ { 2 }$ distance between its feature and the nearest cluster center. The final pseudo-label $\mathbf { p } _ { i }$ is obtained by confidence-weighted fusion of the two assignments:

$$
\mathbf { p } _ { i } = \alpha _ { i } ^ { v } \mathbf { p } _ { i } ^ { v } + \alpha _ { i } ^ { c } \mathbf { p } _ { i } ^ { c } , \qquad \alpha _ { i } ^ { v } + \alpha _ { i } ^ { c } = 1 ,\tag{7}
$$

where the confidence weight for view $m \in \{ v , c \}$ is:

$$
\alpha _ { i } ^ { m } = \frac { \exp \left( - \operatorname* { m i n } _ { k } d _ { i , k } ^ { m } \right) } { \exp \left( - \operatorname* { m i n } _ { k } { d _ { i , k } ^ { v } } \right) + \exp \left( - \operatorname* { m i n } _ { k } { d _ { i , k } ^ { c } } \right) } ,\tag{8}
$$

where $d _ { i , k } ^ { m }$ denotes the $\ell _ { 2 } \cdot$ -norm distance between the feature of video i in view m and the k-th cluster center. Thus, the view whose feature is closer to a cluster center contributes more strongly to the final pseudo-label. The resulting consensus distribution $\mathbf { p } _ { i }$ is used as a soft pseudo-label for classification learning and is periodically updated as visual and codec representations evolve during training. See Appendices B and C for details and clustering experiments.

## 4.3. Codec-Based Boundary Prior Learning

Motivated by the boundary sensitivity observed, CoDy converts codec dynamics into a class-agnostic boundary prior. Based on this observation, CoDy uses codec information as a class-agnostic boundary prior. Specifically, CoDy employs two parallel multi-scale temporal convolutional networks with independent parameters. The visual boundary branch $H _ { \imath }$ processes the snippet-level visual features $\mathbf { X } _ { i }$ while the codec prior network $H _ { c }$ processes the GOP-level features $\mathbf { G } _ { i }$ . Both networks use multi-scale temporal receptive fields to model both local dynamics and long-range temporal dependencies. The outputs are:

$$
\widehat {  { \mathbf { b } } } _ { i } ^ { v } = \sigma ( H _ { v } (  { \mathbf { X } } _ { i } ) ) , \qquad { \mathbf { b } } _ { i } ^ { c } =  { \mathcal { A } } ( \sigma ( H _ { c } (  { \mathbf { G } } _ { i } ) ) ) ,\tag{9}
$$

where $\widehat { \mathbf b } _ { i } ^ { v } \in [ 0 , 1 ] ^ { T _ { i } }$ denotes the snippet-level visual boundary prediction, A aligns the codec prediction $\mathbf { b } _ { i } ^ { c }$ with the visual temporal resolution and σ is an activation function.

In addition to supervising the boundary branch, CoDy uses the temporally quantified codec response to initialize the proposal set. Our analysis shows that thresholding codec cues produces higher-quality candidates than applying the same procedure to RGB or flow. We therefore select high-response timestamps as candidate boundaries, pair them under duration constraints, and use the resulting intervals as the initial proposal instances. This codec-guided initialization enables CoDy to reduce unreliable candidates in early training, accelerating subsequent proposal selection and refinement.

## 4.4. Training and Inference

Iterative proposal refinement. For a proposal $\begin{array} { r l } { p } & { { } = } \end{array}$ $( s _ { p } , e _ { p } )$ , CoDy re-extracts its proposal representation and obtains a category score $s _ { p , c } ^ { \mathrm { c l s } }$ and an attention score $s _ { p } ^ { \mathrm { a t t } }$ . Its endpoint consistency with the learned visual boundary sequence is measured by:

$$
s _ { p } ^ { \mathrm { b n d } } = \sqrt { \operatorname* { m a x } _ { t \in \mathcal { N } ( s _ { p } ) } \widehat { b } _ { t } ^ { v } \operatorname* { m a x } _ { t \in \mathcal { N } ( e _ { p } ) } \widehat { b } _ { t } ^ { v } } ,\tag{10}
$$

where $\mathcal { N } ( u ) = \{ t \ | \ | t - u | \leq \delta \}$ denotes a local temporal neighborhood centered at endpoint u, and δ is a fixed tolerance radius. Taking the maximum response within each neighborhood accommodates minor temporal misalignment between the predicted boundary sequence and the proposal endpoints. The proposal ranking score is:

$$
S _ { p } = \left( s _ { p } ^ { \mathrm { c l s } } + \epsilon \right) ^ { \alpha _ { c l s } } \left( s _ { p } ^ { \mathrm { a t t } } + \epsilon \right) ^ { \alpha _ { a t t } } \left( s _ { p } ^ { \mathrm { b n d } } + \epsilon \right) ^ { \alpha _ { b n d } } ,\tag{11}
$$

where $\alpha _ { c l s } , \alpha _ { a t t }$ , and $\alpha _ { b n d }$ control the contributions of category evidence, proposal foregroundness, and boundary consistency. At each refinement round r, CoDy removes lowscoring proposals with scores below τ and generates new proposals by pairing high-confidence local maxima of the boundary sequence under duration constraints:

$$
\begin{array} { r } { \mathcal { P } ^ { ( r + 1 ) } = \mathrm { K e e p } \Big ( \mathcal { P } ^ { ( r ) } , \tau \Big ) \cup \mathrm { G e n e r a t e } \Big ( \widehat { \mathbf { b } } ^ { v } \Big ) . } \end{array}\tag{12}
$$

The updated proposals are re-encoded and rescored in the next round, allowing category confidence and temporal extent to be refined jointly. The same ranking and augmentation rule is used during training and inference. Further details and threshold experiments are provided in Appendices B and C.

Training. During training, CoDy uses soft pseudo-labels to supervise the classification and attention branch. Let $\mathbf { p } _ { i }$ and $\widehat { \mathbf { p } } _ { i }$ denote the soft pseudo-label and prediction of video i. The classification loss is:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \mathbf { p } _ { i , c } \log \widehat { \mathbf { p } } _ { i , c } .\tag{13}
$$

For boundary learning, we jointly optimize the visual boundary branch and codec prior boundary branch:

$$
\mathcal { L } _ { \mathrm { b n d } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ D \left( \widehat { \mathbf { b } } _ { i } ^ { v } , \mathbf { b } _ { i } ^ { c } \right) + \lambda _ { c } D ( \mathbf { b } _ { i } ^ { c } , q _ { t } ^ { c } ) \right] ,\tag{14}
$$

where $D ( \cdot , \cdot )$ denotes the mean squared error. $\mathbf { b } _ { i } ^ { c }$ is anchored to the temporally codec response $q _ { t } ^ { c }$ derived in Eq. (3), which provides a fixed, non-learnable prior. $\lambda _ { c }$ controls the anchor strength and is gradually decayed with training depth. The complete objective is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda \mathcal { L } _ { \mathrm { b n d } } ,\tag{15}
$$

where λ is the balance parameter.

Inference. At inference, CoDy discards pseudo-label generation and the codec prior branch, retaining only the P-MIL backbone and the learned visual boundary branch. Without ground-truth labels or codec temporal priors, the model performs a fixed number of self-refinement rounds. Starting from the baseline proposals, CoDy iteratively updates the proposal set using Eq. (12). At each round, it predicts proposal scores and a boundary sequence, and then updates the proposal set. The final detections are obtained after the last refinement round and standard score suppression.

## 5. Experiments

## 5.1. Experimental Setup

Datasets. We evaluate CoDy on two standard temporal action localization benchmarks. THUMOS14 [8] contains 413 temporally annotated untrimmed videos across 20 action categories, including 200 validation and 213 test videos. Following standard protocols, we train on the validation set and evaluate on the test set. ActivityNet v1.3 [1] covers 200 activity categories across 19,994 untrimmed videos, divided into 10,024 training, 4,926 validation, and 5,044 test videos. We train on the training set and evaluate on the validation set.

Metrics. For THUMOS14, we report detection mAP at temporal IoU thresholds from 0.1 to 0.7 with a step of 0.1, along with their average. For ActivityNet v1.3, we report mAP at 0.5, 0.75, 0.95, and the average mAP over thresholds from 0.5 to 0.95 with a step of 0.05.

Implementation details. We adopt the UTAL variant of P-MIL [16] as baseline. Visual features are extracted using I3D models pretrained on Kinetics, where RGB and optical-flow features are concatenated to form $D =$ 2048-dimensional segment features. For codec processing, videos are sampled at 30 FPS and divided into 0.5-second groups, each consisting of 1 I-frame and 14 P-frames. We extract motion vectors and residuals from the H.264 bitstream, and set the codec token dimension to d = 64. Both boundary branches use four-block multi-scale TCNs with dilation rates of {1, 2, 4, 8} and a kernel size of 3. The model is trained for 200 epochs using Adam with a learning rate of $5 \times 1 0 ^ { - 5 }$ , a batch size of 10, and a weight decay of $1 0 ^ { - 3 }$ . We set $\tau = 0 . 3 , \lambda = 1$ and initialize $\lambda _ { c } = 1$ in all experiments. More implementation and training details are provided in Appendix B. A comprehensive analysis of the computational cost of CoDy is presented in Appendix C.

<table><tr><td rowspan="2">Sup.</td><td rowspan="2">Method</td><td colspan="7">mAP@IoU (%)</td><td colspan="3">AVG</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>(0.1:0.5)</td><td>(0.3:0.7)</td><td>(0.1:0.7)</td></tr><tr><td rowspan="8">Weakly</td><td>CoLA [27]</td><td></td><td>66.2</td><td>59.5</td><td>51.5</td><td>41.9</td><td>32.2</td><td>22.0</td><td>13.1</td><td>50.3</td><td>32.1</td><td>40.9</td></tr><tr><td>DGCNN [17]</td><td></td><td>66.3</td><td>59.9</td><td>52.3</td><td>43.2</td><td>32.8</td><td>22.1</td><td>13.1</td><td>50.9</td><td>32.7</td><td>41.3</td></tr><tr><td>AUMN [12]</td><td></td><td>66.2</td><td>61.9</td><td>54.9</td><td>44.4</td><td>33.3</td><td>20.5</td><td>9.0</td><td>52.1</td><td>32.4</td><td>41.5</td></tr><tr><td>CSCL [9]</td><td></td><td>68.0</td><td>61.8</td><td>52.7</td><td>43.3</td><td>33.4</td><td>21.8</td><td>12.3</td><td>51.8</td><td>32.7</td><td>41.9</td></tr><tr><td>FAC-Net [26]</td><td></td><td>67.6</td><td>62.1</td><td>52.6</td><td>44.3</td><td>33.4</td><td>22.5</td><td>12.7</td><td>52.0</td><td>33.1</td><td>42.2</td></tr><tr><td>ACGNet [25]</td><td></td><td>68.1</td><td>62.6</td><td>53.1</td><td>44.6</td><td>34.7</td><td>22.6</td><td>12.0</td><td>52.6</td><td>33.4</td><td>42.5</td></tr><tr><td>ACM-Net [15]</td><td></td><td>68.9</td><td>62.7</td><td>55.0</td><td>44.6</td><td>34.6</td><td>21.8</td><td>10.8</td><td>53.2</td><td>33.4</td><td>42.6</td></tr><tr><td></td><td></td><td>50.1</td><td>45.8</td><td>40.6</td><td>32.3</td><td>20.9</td><td>10.7</td><td></td><td>37.9</td><td></td><td></td></tr><tr><td rowspan="8">Unsup.</td><td>STPN [14]</td><td>57.1</td><td>51.6</td><td>43.9</td><td></td><td></td><td></td><td></td><td>4.6</td><td></td><td>21.8</td><td>29.3</td></tr><tr><td>TSCN [19]</td><td>57.7</td><td></td><td></td><td></td><td>35.3</td><td>26.0</td><td>15.7</td><td>6.0</td><td>42.8</td><td>25.4</td><td>33.7</td></tr><tr><td>APSL [11] UGCT [24]</td><td>63.4</td><td></td><td>52.4</td><td>44.1</td><td>35.9</td><td>27.9 32.8</td><td>18.5 21.6</td><td>10.0</td><td>43.6</td><td>27.3</td><td>35.2</td></tr><tr><td>FEEL-V [7]</td><td></td><td></td><td>57.8</td><td>51.7</td><td>44.0</td><td>29.3</td><td>22.6</td><td>10.1 11.5</td><td>49.9</td><td>32.0</td><td>40.2</td></tr><tr><td>CLIP-AE [22]</td><td>65.1</td><td>60.4</td><td></td><td>52.7</td><td>44.5</td><td>33.2</td><td>22.3</td><td>13.6</td><td>51.2</td><td>33.3</td><td>41.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>45.3</td><td>37.5</td><td>25.5</td><td></td><td>52.3</td><td></td><td></td></tr><tr><td>CoDy (Ours)</td><td>64.3</td><td>60.9</td><td>53.6 (+0.9)</td><td></td><td>(+0.8)</td><td></td><td></td><td>14.8</td><td></td><td>35.3</td><td>43.1</td></tr><tr><td>Improvement (∆)</td><td>(-0.8)</td><td>(+0.5)</td><td></td><td></td><td></td><td>(+4.3)</td><td>(+2.9)</td><td>(+1.2)</td><td>(+1.1)</td><td>(+2.0)</td><td>(+1.4)</td></tr></table>

Table 1. Comparison results on the THUMOS14. The best and second-best results are in bold and underlined.

<table><tr><td rowspan="2">Method</td><td colspan="3">mAP@IoU (%)</td></tr><tr><td>0.5</td><td>0.75 0.95</td><td>AVG</td></tr><tr><td>STPN [14]</td><td>23.9</td><td>14.9 3.8</td><td>15.0</td></tr><tr><td>TSCN [19]</td><td>25.9</td><td>16.2 3.8</td><td>16.3</td></tr><tr><td>WSAL-BM + UGCT [24]</td><td>30.1</td><td>19.2 3.9</td><td>18.5</td></tr><tr><td>TSCN + UGCT [24]</td><td>30.3</td><td>19.4</td><td>4.3 18.7</td></tr><tr><td>STPN + UGCT [24]</td><td>32.5</td><td>20.3 4.5</td><td>19.8</td></tr><tr><td>CoDy (Ours) Improvement (∆)</td><td>31.9 (-0.6)</td><td>20.3 5.0 (+0.0)</td><td>20.2 (+0.5) (+0.4)</td></tr></table>

Table 2. Comparison results on the ActivityNet v1.3. The best and second-best results are in bold and underlined.

## 5.2. Main Results

Table 1 and Table 2 summarize the results on THUMOS14 and ActivityNet v1.3. CoDy consistently outperforms existing unsupervised methods at most IoU thresholds and achieves the strongest overall accuracy. The performance gains are particularly pronounced at stricter IoU thresholds, demonstrating that codec-prior dynamics effectively enhance temporal boundary precision. The slightly lower performance at the lowest threshold suggests that codec priors may mistake scene changes and camera motion for action transitions. The consistent gains across both datasets demonstrate the broad applicability of codec cues in UTAL. On densely annotated THUMOS14, codec-based temporal variations resolve closely spaced action boundaries. On ActivityNet v1.3, which features longer and sparser activities, the same cues improve localization precision. These findings confirm that CoDy successfully leverages codec information as a powerful prior for proposal refinement.

<table><tr><td colspan="2">Main components</td><td colspan="4">mAP@IoU (%)</td></tr><tr><td>Classification</td><td>Localization</td><td>0.3</td><td>0.5</td><td>0.7</td><td>AVG</td></tr><tr><td></td><td></td><td>44.8</td><td>30.8</td><td>8.5</td><td>35.7</td></tr><tr><td></td><td>√</td><td>48.8</td><td>34.9</td><td>11.0</td><td>38.4</td></tr><tr><td>√</td><td></td><td>49.9</td><td>35.2</td><td>12.1</td><td>39.5</td></tr><tr><td>√</td><td>√</td><td>53.6</td><td>37.5</td><td>14.8</td><td>43.1</td></tr></table>

Table 3. Ablation study of key components on THUMOS14.

## 5.3. Ablation Study

## 5.3.1. Impact of Key Components.

We conduct an ablation study to evaluate the two core components of CoDy: codec-enhanced pseudo-label generation and codec-prior boundary guidance. As summarized in Table 3, the Classification setup uses the codec video-level representation as a complementary view during multi-view clustering, while the Localization setup uses the codecbased boundary prior to supervise the boundary branch. Both components consistently improve performance over the codec-free baseline. Specifically, the semantic view enhances pseudo-label quality for classification, whereas the boundary prior improves proposal temporal precision. Combining both components yields the highest accuracy, outperforming either standalone branch.

![](images/27a097ae5a0764887b2d00ef8c96ebaf6c4e478e082d5837e03a47c4be9d3d99.jpg)  
(a) Proposal sources

![](images/f406c6a1543374685c45d2b2071e1ebc7aa4c83dddf4a33b7a329d73da71fcb0.jpg)  
(b) Pseudo-label quality  
Figure 4. Ablation study of codec for proposal source and pseudolabel quality.

## 5.3.2. Impact of Proposal Sources.

To evaluate how different signal sources affect candidate initialization, we test class-agnostic proposals built from various features over iterative refinement. Video-level category labels are provided to all variants solely to eliminate classification bias and isolate proposal quality. As shown in Figure 4(a), initialization sources exhibit distinct trajectories. At iteration 0, codec proposals achieve 10.60% average mAP, markedly outperforming RGB at 5.78%, flow at 5.76%, and their fusion at 5.79%. This initial lead proves that codec signals supply richer and sharper temporal boundary cues. This superior initialization accelerates convergence and maintains higher overall accuracy throughout training. Conversely, non-codec variants yield inferior trajectories similar to random initialization.

## 5.3.3. Impact of Codec View for Pseudo-Label Quality.

We compare two pseudo-label generation settings under iterative training, one using RGB and flow visual views alone and the other incorporating the codec view as a complementary clustering signal. At every pseudo-label refresh, we evaluate the generated labels using aligned accuracy and macro-F1. After each iteration, the Hungarian algorithm assigns the discovered clusters to category labels. Ground truth video level labels are available only to the evaluator. As shown in Figure 4(b), even before model iteration at iteration 0, incorporating the codec view directly produces more accurate and more balanced pseudo-labels. As training proceeds, visual and codec representations are repeatedly updated, while CoDy maintains its pseudo-label quality advantage across successive refreshes. This shows that codec representations provide complementary information to RGB and flow.

## 5.3.4. Impact of Motion Vector and Residual Features.

We compare MVs, residuals, and their fusion as codec representations to quantify their respective contributions. As shown in Table 4, MVs are more effective for the classification branch, as they provide explicit motion cues that facilitate semantic understanding. Residuals perform better for boundary modeling, since they preserve local appearance variations and temporal discontinuities around action transitions. Combining MVs and residuals, CoDy achieves

<table><tr><td rowspan="2"></td><td rowspan="2">Codec Feature</td><td colspan="3">mAP@IoU (%)</td></tr><tr><td>0.3</td><td>0.5 0.7</td><td>AVG</td></tr><tr><td>Classification</td><td>Residual-only MV-only</td><td>48.9 49.6</td><td>33.6 10.1 35.5 11.1</td><td>38.9 40.9</td></tr><tr><td>Localization</td><td>Residual-only MV-only</td><td>50.6 49.5</td><td>35.7 12.4 34.6 11.8</td><td>41.8 39.8</td></tr><tr><td>CoDy(Ours)</td><td>MV+Residual</td><td>53.6</td><td>37.5 14.8</td><td>43.1</td></tr></table>

Table 4. Ablation study of motion vector and residual features on classification and localization performance.

the best performance.

## 6. Conclusion

We present CoDy, a codec-guided framework that exploits complementary information from the video bitstream for UTAL. CoDy integrates a codec view with RGB and flow representations to improve latent category discovery and iterative pseudo-label generation. Meanwhile, motion-vector and residual dynamics provide class-agnostic priors for visual boundary learning and proposal refinement. Extensive experiments on THUMOS14 and ActivityNet v1.3 demonstrate the superior performance of CoDy. These results establish codec dynamics as a complementary source of supervision for learning action categories and temporal boundaries.

## References

[1] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. ActivityNet: A large-scale video benchmark for human activity understanding. In Proceed ings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 961–970, 2015.

[2] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[3] Jiawei Chen and Chiu Man Ho. MM-ViT: Multi-modal video transformer for compressed video action recognition. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 786–797, 2022.

[4] Sayan Deb Sarkar, Remi Pautrat, Ondrej Miksik, Marc Polle-´ feys, Iro Armeni, Mahdi Rad, and Mihai Dusmanu. CoPE-VideoLM: Codec primitives for efficient video language models. arXiv preprint arXiv:2602.13191, 2026.

[5] Qihang Fang, Chengcheng Tang, Shugao Ma, and Yanchao Yang. BID: Boundary-interior decoding for unsupervised temporal action localization pre-training. In Proceedings of the 41st International Conference on Machine Learning, 2024.

[6] Guoqiang Gong, Xinghan Wang, Yadong Mu, and Qi Tian. Learning temporal co-attention models for unsupervised video action localization. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 2020.

[7] Yupeng Hu, Han Jiang, Hao Liu, Kun Wang, Haoyu Tang, and Liqiang Nie. Visual self-paced iterative learning for unsupervised temporal action localization. ACM Transactions on Multimedia Computing, Communications, and Applications, 22(7), 2026.

[8] Haroon Idrees, Amir R. Zamir, Yu-Gang Jiang, Alex Gorban, Ivan Laptev, Rahul Sukthankar, and Mubarak Shah. The THUMOS challenge on action recognition for videos “in the wild”. Computer Vision and Image Understanding, 155:1– 23, 2017.

[9] Yuan Ji, Xu Jia, Huchuan Lu, and Xiang Ruan. Weaklysupervised temporal action localization via cross-stream collaborative learning. In Proceedings of the 29th ACM international conference on multimedia, pages 853–861, 2021.

[10] Didier Le Gall. MPEG: A video compression standard for multimedia applications. Communications of the ACM, 34 (4):46–58, 1991.

[11] Yuanyuan Liu, Ning Zhou, Fayong Zhang, Wenbin Wang, Yu Wang, Kejun Liu, and Ziyuan Liu. APSL: Action-positive separation learning for unsupervised temporal action localization. Information Sciences, 630:206–221, 2023.

[12] Zixin Luo, Dewen Zeng, Limin Wang, and Tong Lu. Action unit memory network for weakly supervised temporal action localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9969– 9979, 2021.

[13] Davide Moltisanti, Sanja Fidler, and Dima Damen. Action recognition from single timestamp supervision in untrimmed videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9915–9924, 2019.

[14] Phuc Nguyen, Ting Liu, Gautam Prasad, and Bohyung Han. Weakly supervised action localization by sparse temporal pooling network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6752– 6761, 2018.

[15] Sanqing Qu, Guang Chen, Zhijun Li, Lijun Zhang, Fan Lu, and Alois Knoll. ACM-Net: Action context modeling network for weakly-supervised temporal action localization. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021.

[16] Huan Ren, Wenfei Yang, Tianzhu Zhang, and Yongdong Zhang. Proposal-based multiple instance learning for weakly-supervised temporal action localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2394–2404, 2023.

[17] Haichao Shi, Xiao-Yu Zhang, Changsheng Li, Lixing Gong, Yong Li, and Yongjun Bao. Dynamic graph modeling for weakly-supervised temporal action localization. In Proceedings of the ACM International Conference on Multimedia, pages 3820–3828, 2022.

[18] Zheng Shou, Dongang Wang, and Shih-Fu Chang. Temporal action localization in untrimmed videos with segment-based convolutional neural networks. In Proceedings of the European Conference on Computer Vision (ECCV), 2016.

[19] Wei Tang, Le Wang, Yuanhao Zhai, Gang Hua, and Junsong Yuan. Two-stream consensus network for weakly-supervised temporal action localization. In 16th European Conference on Computer Vision (ECCV), 2020.

[20] Thomas Wiegand, Gary J. Sullivan, Gisle Bjøntegaard, and Ajay Luthra. Overview of the H.264/AVC video coding standard. IEEE Transactions on Circuits and Systems for Video Technology, 13(7):560–576, 2003.

[21] Chao-Yuan Wu, Manzil Zaheer, Hexiang Hu, R. Manmatha, Alexander J. Smola, and Philipp Krahenb ¨ uhl. Compressed¨ video action recognition. In Proceedings of the IEEE Con ference on Computer Vision and Pattern Recognition, 2018.

[22] Rui Xia, Dan Jiang, Quan Zhang, Ke Zhang, and Chun Yuan. CLIP-AE: CLIP-assisted cross-view audio-visual enhancement for unsupervised temporal action localization. arXiv preprint arXiv:2505.23524, 2025.

[23] Huijuan Xu, Abir Das, and Kate Saenko. R-C3D: Region convolutional 3d network for temporal activity detection. In Proceedings of the AAAI Conference on Artificial Intelli gence (AAAI), 2017.

[24] Wenfei Yang, Tianzhu Zhang, Yongdong Zhang, and Feng Wu. Uncertainty guided collaborative training for weakly supervised and unsupervised temporal action localization. IEEE Transactions on Pattern Analysis and Machine Intel ligence, 45(4):5252–5267, 2023.

[25] Zichen Yang, Jie Qin, and Di Huang. ACGNet: Action complement graph network for weakly-supervised temporal action localization. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 3090–3098, 2022.

[26] Qing Yu and Kent Fujiwara. Frame-level label refinement for skeleton-based weakly-supervised action recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3322–3330, 2023.

[27] Can Zhang, Meng Cao, Dongming Yang, Jie Chen, and Yuexian Zou. CoLA: Weakly-supervised temporal action localization with snippet contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 16010–16019, 2021.

[28] Y. Zhao, Yuanjun Xiong, Limin Wang, Zuxuan Wu, Xiaoou Tang, and Dahua Lin. Temporal action detection in untrimmed videos with structured segment networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 2914–2923, 2017.