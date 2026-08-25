# Hybrid Generative-Discriminative Object Placement

Siyuan Zhou Shanghai Jiao Tong University ssluvble@sjtu.edu.cn

Li Niu Shanghai Jiao Tong University ustcnewly@sjtu.edu.cn

## Abstract

As an important operation of image composition, object placement aims to predict the plausible placement (location, scale) for the inserted foreground object. Previous object placement methods can be divided into generative methods and discriminative methods, both of which cannot balance efficiency and effectiveness well. In this work, we propose a semi-generative method in the middle ground between them. In particular, we assign uniformly distributed anchors on the background. Then, we fuse foreground and backgroundfeatures to predict the rationality scorefor each anchor and predict plausible placement setsfor positive anchors. Extensive experiments on the OPA dataset show that our method can strike a good balance between efficiency and effectiveness.

## 1. Introduction

Image composition, aiming to combine the foreground and background from different images as a composite image, is a crucial and popular technology for image editing, which has been extensively used in augmented reality, artistic creation, and automatic advertising [24, 27, 31]. However, the obtained composite images via simple cut-andpaste could have many issues (e.g., illumination, shadow, placement) that harm the image quality. As image composition is a complicated task involving many issues to be addressed, most previous works only focus on one of the issues, giving rise to different subtasks like image harmonization [26, 4, 5], shadow generation [16, 12], and object placement learning [25, 29, 24].

In this work, we focus on object placement [25, 29, 24], which aims to predict plausible placement (i.e., location, scale) of the foreground object given a pair of foreground and background. Object placement is a challenging task due to the complex factors deciding the plausibility of placement, like the depth and physical size of object, semantic context, occlusion, etc. Existing object placement methods can be divided into generative methods and discriminative methods. They both take in a pair of foreground and background, and the difference lies in the output format. The generative methods [3, 21, 29, 9] predict one or multiple plausible placements for the foreground object, as shown in Figure 1(a). Discriminative methods predict a rationality score for each foreground scale and each location, which measures the plausibility of composite image obtained by placing the center of scaled foreground at this location, as shown in Figure 1(b).

Nevertheless, generative methods and discriminative methods have their respective drawbacks. The generative methods often encounter mode collapse issue and the predicted placements fail to cover most reasonable locations. Besides, the prediction quality is usually lower than discriminative methods. The discriminative methods can cover all the locations with better prediction quality, but the time cost is much higher than generative methods. In this work, we attempt to fill in the middle ground between discriminative method and generative method, leading to our semigenerative method which can strike a good balance between efficiency and effectiveness.

Specifically, we first extract a foreground feature vector and a background feature map from foreground and background, respectively. Then, we pass the foreground feature vector and the background feature map through a fusion module to produce the fused feature map. We treat each location in the fused feature map as an anchor, which is associated with a set of placements with the foreground center located in its neighborhood. Each anchor is classified as either a positive anchor or a negative anchor, according to whether its associated placement set has at least one plausible placement. For each positive anchor, we attempt to predict its set of plausible placements (i.e., the center offset and the foreground scale). The anchor locations and the center offset lead to the center location. Then, the center location and foreground scale further lead to a complete placement. With our designed anchor mechanism and output format, we construct a network structure in the middle ground between generative method and discriminative method. Considering the sparse annotations of existing object placement dataset, we design a two-stage training strategy. In the first stage, we train the network using the provided sparse annotations. In the second stage, we use the prediction results to refine the annotations, and retrain our network using the updated annotations.

![](images/f258765dc795d5643fe7dea7e2860de58b51b1ee61249bbb3f3f3c39236cef57.jpg)  
Figure 1. The comparison among three paradigms for object placement learning. Given a pair of foreground with mask and background, the generative model (a) generates one or multiple plausible placements (center location (x, y) and scale s) for the foreground. The discriminative model (b) predicts a rationality score map for each foreground scale. ${ r } _ { i , s }$ is the rationality score for the i-th center location and scale s. Our semi-generative model (c) predicts a rationality score r<sub>i</sub> for the i-th anchor and also predicts multiple plausible placements (center offset (ox<sub>i</sub>, oy<sub>i</sub>) and scale s<sub>i</sub>) for each positive anchor.

Another contribution of our work lies in the dynamic fusion module to fuse foreground and background information. When we place one foreground object at one location on the background, the plausibility of this placement should depend on the contextual information surrounding this location, so we attempt to aggregate background contextual information for each location. The suitable scale of contextual information depends on the foreground information and background information at this location. For example, when placing a cat or an elephant on the ground, we need to observe the surrounding regions of different scales to check the unreasonable occlusion between this foreground object and other background objects. Therefore, we use foreground and background information to predict an offset map for deformable convolution, which aggregates different scales of background contextual information adaptively for different locations and different foreground objects.

Following [32], we conduct experiments on OPA dataset [17] to verify the effectiveness of our method. Our major contributions can be summarized as follows. 1) We propose a semi-generative object placement method, which fills in the middle ground between discriminative methods and generative methods. 2) We design a novel dynamic fusion module to fuse foreground and background information, which can aggregate different scales of background contextual information adaptively. 3) Extensive experiments on OPA dataset demonstrate that our method strikes a good balance between efficiency and effectiveness.

## 2. Related Work

## 2.1. Image Composition

Image composition targets at combining foreground and background from different source images to produce a composite image. However, the quality of composite images obtained through simple cut-and-paste may be degraded by the appearance, geometric, and semantic inconsistency between foreground and background. Different subtasks have been studied to address different issues in composite images. For instance, to address the inconsistent illumination between foreground and background, image harmonization [26, 4, 15, 10, 5, 23] adjusted the foreground il lumination to match the background and produced a harmonious composite image. To address the issue of missing shadow, shadow generation methods [16, 12] created plausible shadow for the composite foreground. To address the irrational placement of composite foreground, object placement methods [8, 24, 31, 13, 25, 29] predicted plausible placement (e.g., location, scale) for the composite foreground. In this work, we follow the research line of object placement and tend to predict reasonable location/scale for the inserted foreground.

## 2.2. Object Placement Learning

As an inevitable step in image composition, object placement is helpful for many application scenarios like artistic design and automatic advertising [31]. Existing methods on object placement learning can be roughly divided into generative methods and discriminative methods. Given a pair of foreground and background, the generative methods aim to generate one or more plausible placements for the foreground. For example, the methods [14, 28, 25, 1] predicted one plausible placement for the foreground. [29, 32] injected a random vector into the network and predicted multiple plausible placements by sampling the random vector. Generative methods are efficient in generating a few plausible placements, but the quality and diversity are lower than discriminative methods. Early discriminative method [17] predicted whether the foreground placement in a composite image is plausible, which is also dubbed as object placement assessment [17]. Then, [19] proposed a fast object placement assessment method to accelerate [17], which takes in a scaled foreground and a background to predict all reasonable locations for this scaled foreground. Nonetheless, the accelerated version [19] is still much slower than generative methods.

![](images/9e414ff6ec6c8f5962f9271627f1bf5e49acc11914693d43c871b4a875eb0e74.jpg)  
Figure 2. The illustration of our network structure. We use DFM to fuse foreground feature vector $f ^ { f }$ and background feature map $\pmb { F } ^ { b }$ For each $\mathbf { \Delta } f _ { i } ^ { m }$ in the fused feature map ${ \pmb F } ^ { m }$ , we use it to predict rationality score $r _ { i }$ and concatenate it with random vector $\scriptstyle { z _ { t } }$ to produce a placement set $\mathcal { P } _ { i }$ . The detailed architecture of our Dynamic Fusion Module (DFM) is shown at the top right corner. We concatenate background feature and foreground feature to produce the offset map for deformable convolution, which can dynamically aggregate multiscale contextual information.

In this work, we attempt to fill the middle ground between generative methods and discriminative methods. We proposed a semi-generative model which achieves comparable performances with discriminative methods at much lower cost.

## 3. Our Method

Suppose that we have a background image $\pmb { I } ^ { b } \in$ $\mathcal { R } ^ { H \times W \times 3 }$ and a foreground image $\mathbf { \bar { \mathbf { \Lambda } } } _ { I ^ { f } } \in \mathcal { R } ^ { H \times } \mathbf { \breve { W } } \times \mathbf { 3 }$ with its binary object mask $M ^ { f } \in \mathcal { R } ^ { \widecheck { H } \times W }$ delineating the foreground object, where H and W are image height and width respectively. The goal of object placement is predicting plausible placements $( e . g .$ , scale, location) for the foreground object. As claimed in Section 1, the output formats of generative methods [29, 32] and discriminative methods [19] differ a lot. In this work, we design a semi-generative approach whose output format is a mixture of generative methods and discriminative methods.

## 3.1. Overall Network Structure

We use foreground encoder $E ^ { f }$ and background encoder $E ^ { b }$ to extract a foreground feature map and a background feature map, respectively. Following [32], $E ^ { f }$ and $E ^ { \overline { { b } } }$ share the same backbone network with different heads. $E ^ { f }$ takes in the concatenation of foreground image $I ^ { f }$ and its binary object mask $M ^ { f }$ , producing the foreground feature map. We perform global average pooling over the foreground feature map and get the foreground feature vector $\pmb { f } ^ { f } \in \mathcal { R } ^ { c }$ $E ^ { b }$ takes in the background $I ^ { b }$ and produces the background feature map $F ^ { b } \in { \mathcal { R } } ^ { h \times w \times c }$ . We treat each pixel in the background feature map as an anchor, leading to $n = h \times w$ anchors. As shown in Figure 3, the location of the i-th anchor on the background image is assumed to be the center of the i-th cell, after uniformly dividing the background image into $h \times w$ cells. The i-th cell is also called the neighborhood of the i-th anchor. The rationality score $r _ { i }$ of the i-th anchor indicates whether exists plausible placement with the foreground center in the neighborhood of the i-th anchor. Based on the rationality scores, we categorize all n anchors into positive anchors and negative anchors. For positive anchors, we attempt to predict their plausible placement sets, in which one placement includes the center offset and foreground scale. On the premise of anchor location, center offset, and foreground scale, we can obtain a plausible composite image.

## 3.2. Dynamic Fusion Module

To predict the rationality score and placement set for each anchor, we integrate the foreground feature vector $f ^ { f }$ with each pixel-wise feature vector in the background feature map $F ^ { b }$ . One simple way is spatially replicating $f ^ { f }$ and concatenating them with $\dot { \pmb { F } } ^ { b }$ channel-wisely. However, when a foreground object is placed at certain location on the background, the plausibility of this placement relies on the nearby contextual information, so we need to aggregate background contextual information for each location. The suitable scale of contextual information to be aggregated depends on the foreground and background information at this location. To this end, we design a Dynamic Fusion Module (DFM) to fuse $f ^ { f }$ and $F ^ { b }$ , which automatically decides the scale of contextual information for each location based on the foreground and background information.

As shown in Figure 2, we spatially replicate $f ^ { f }$ to get

![](images/778436f12598fd521c150b5b5b74cdd4fedfea83c2a5a3af45af2ef66c0cb93c.jpg)  
Figure 3. We suppose that an image is divided into 4 × 4 cells, corresponding to 16 anchors (red dot). In the left column, we show the foreground/background and the rationality score map. In the right column, we show one foreground placement, which is associated with the i-th anchor because the foreground center is within its neighborhood (red cell). The location of foreground center can be calculated based on the anchor location $( x _ { i } , y _ { i } )$ and center offset $( o x _ { i } , o y _ { i } )$ . The foreground placement (yellow bounding box) is represented by the foreground center location (x<sub>i</sub>+ox<sub>i</sub>, y<sub>i</sub>+oy<sub>i</sub>) and the foreground scale $s _ { i }$

$F ^ { f } ,$ which is concatenated with $F ^ { b }$ . The concatenation passes through one conv layer to produce an offset map for deformable convolution [6]. The entry at location l in the offset map contains the spatial offsets $\{ \Delta o \iota , k | _ { k = 1 } ^ { 9 } \}$ for a $3 \times 3$ convolution kernel centered at this location. Based on the offset map, we perform deformable convolution [6] on $F ^ { b }$ . Formally, assuming that the kernel weights are $\{ \pmb { w } _ { k } | _ { k = 1 } ^ { 9 } \}$ and the relative locations in the kernel are $\pmb { o } _ { k } \in \{ ( - 1 , - 1 ) , ( - 1 , 0 ) , \dotsc , ( 1 , 1 ) \}$ , the deformable convolution operation on $F ^ { b }$ with kernel center at the location l can be represented by

$$
y ( l ) = \sum _ { k = 1 } ^ { 9 } w _ { k } \cdot F ^ { b } ( l + o _ { k } + \Delta o _ { l , k } ) ,\tag{1}
$$

in which $y ( l )$ is the output value. For more details of deformable convolution, please refer to [6].

In this way, we aggregate different scales of background contextual information adaptively for different locations. The output of deformable convolution is concatenated with $F ^ { f }$ . The concatenation further passes through a $3 \times 3$ conv layer to produce the fused feature map ${ \pmb F } ^ { m }$

## 3.3. Rationality and Placement Prediction

We denote the i-th feature vector in the fused feature map ${ \pmb F } ^ { m }$ as $\pmb { f } _ { i } ^ { m }$ , which is used to predict the rationality score and a set of plausible placements for the i-th anchor. The rationality score $r _ { i }$ within the range of [0, 1] indicates the existence of plausible placement centered around this location (the foreground center is in the neighborhood of this anchor). Assuming that we have the ground-truth rationality score $\boldsymbol { { \hat { r } } } _ { i }$ , the predicted $r _ { i }$ is supervised using binary crossentropy loss:

$$
\mathcal { L } _ { r s } = - \hat { r } _ { i } \log r _ { i } - ( 1 - \hat { r } _ { i } ) \log ( 1 - r _ { i } ) .\tag{2}
$$

We refer to the anchors with rationality scores larger than the threshold 0.5 as positive anchors and the other anchors as negative anchors. We only supervise the predicted placements of positive anchors, while leaving those of negative anchors unattended. To predict multiple plausible placements for each positive anchor i, we concatenate $\pmb { f } _ { i } ^ { m }$ with a random vector $z \sim \mathcal { N } ( 0 , 1 )$ . The concatenated vector is sent into a prediction head to produce one placement $\pmb { p } _ { i } = ( o x _ { i } , o y _ { i } , s _ { i } )$ , in which $( o x _ { i } , o y _ { i } )$ is the center offset and $s _ { i }$ is the foreground scale. By denoting the location of the i-th anchor as $( x _ { i } , y _ { i } )$ , the predicted foreground object center is $\left( x _ { i } + o x _ { i } , y _ { i } + o y _ { i } \right)$ , which should be constrained within the neighborhood of the i-th anchor. Note that we have a common scale $s _ { i }$ for both height and width of the foreground object, because we do not change the aspect ratio of foreground object.

By sampling z multiple times, we can get multiple placements for each anchor. Specifically, when sampling $T$ times, we have $\{ z _ { 1 } , \dots , z _ { T } \}$ and produce a placement set $\mathcal { P } _ { i } ~ = ~ \{ p _ { i , 1 } , . . . , p _ { i , T } \}$ for the i-th anchor, in which $\mathbf { \phi } _ { p _ { i , t } } = ( o x _ { i , t } , o y _ { i , t } , s _ { i , t } )$ . To ensure the diversity of predicted placements, we employ the mode seeking loss [18] to enforce the predictions with different z to be different:

$$
\mathcal { L } _ { m s } = \sum _ { t \neq t ^ { \prime } } \frac { | | z _ { t } - z _ { t ^ { \prime } } | | _ { 1 } } { | | \pmb { p } _ { i , t } - \pmb { p } _ { i , t ^ { \prime } } | | _ { 1 } } .\tag{3}
$$

Intuitively, when $\left| { \left| z _ { t } \mathrm { ~ - ~ } z _ { t ^ { \prime } } \right| } \right| _ { 1 }$ is large, we expect ${ \mathbf { } } p _ { i , t }$ and $\mathbf { \Delta } _ { p _ { i , t ^ { \prime } } }$ to be considerably different, which can push the prediction head to search more modes to produce diverse placements. Let us assume that we have a set of ground-truth plausible placements for the i-th anchor $\hat { \mathcal { P } } _ { i } =$ $\{ \hat { p } _ { i , 1 } , \dotsc , \hat { p } _ { i , K _ { i } } \}$ , in which $K _ { i }$ is the number of groundtruth plausible placements for the i-th anchor. Note that we set $T \geqslant K _ { i } , \forall i$ . When $K _ { i } < T$ , we pad $\hat { \mathcal { P } } _ { i }$ to match the size $T$ . We need to measure the distance between $\mathcal { P } _ { i }$ and $\begin{array} { r } { \hat { \mathcal { P } } _ { i } , } \end{array}$ which involves matching two placement sets. Inspired by [2], we perform bipartite matching between $\mathcal { P } _ { i }$ and ${ \hat { \mathcal { P } } } _ { i } ,$ seeking for an index mapping $\hat { \sigma }$ such that the matching cost is minimized. Formally,

$$
\hat { \sigma } = \arg \operatorname* { m i n } _ { \sigma } \sum _ { t = 1 } ^ { T } L _ { 2 } \big ( \hat { { p } } _ { i , t } , { { p } } _ { i , \sigma ( t ) } \big ) ,\tag{4}
$$

in which $L _ { 2 } \big ( \hat { p } _ { i , t } , p _ { i , \sigma ( t ) } \big )$ is the $L _ { 2 }$ distance between $\hat { p } _ { i , t }$ and $\mathbf { \mathit { p } } _ { i , \sigma ( t ) }$ when neither $\hat { p } _ { i , t }$ nor $\mathbf { \mathit { p } } _ { i , \sigma ( t ) }$ is ∅. Otherwise,

$L _ { 2 } ( \hat { p } _ { i , t } , p _ { i , \sigma ( t ) } ) = 0$ . Then, the matching loss between $\mathcal { P } _ { i }$ and $\hat { \mathcal { P } } _ { i } ^ { \phantom { \dagger } }$ can be calculated as

$$
\mathcal { L } _ { p l } = \sum _ { t = 1 } ^ { T } \mathcal { L } _ { 2 } \big ( \hat { p } _ { i , t } , \pmb { p } _ { i , \hat { \sigma } ( t ) } \big ) .\tag{5}
$$

For more details of matching loss, please refer to [2].

In addition, we employ the slow object placement assessment (SOPA) model [17] to supervise the predicted placements of positive anchors. The SOPA model, denoted by $D ,$ is a binary classification model to predict the score of a composite image (1 for positive and 0 for negative). Given the predicted placements of positive anchors, we can obtain the composite images $\pmb { I ^ { c } }$ and send them to $D$ to maximize the predicted classification scores. The rationality classification loss is given by

$$
\mathcal { L } _ { c l } = - \log D ( I ^ { c } ) .\tag{6}
$$

So far, the collection of all training losses is

$$
\mathcal { L } _ { a l l } = \lambda _ { 1 } \mathcal { L } _ { r s } + \mathcal { L } _ { m s } + \lambda _ { 2 } \mathcal { L } _ { p l } + \mathcal { L } _ { c l } ,\tag{7}
$$

in which $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are trade-off parameters.

## 3.4. Training Strategies

Following [32], we use the object placement assessment (OPA) dataset contributed by [17], which provides sparse annotations for partial positive and negative placements (location, scale). We assign annotated placements to their corresponding anchors and refer to the anchors associated with at least one annotated placement as annotated anchors. The unannotated anchors are ignored. If one annotated anchor has at least one positive placement, then this annotated anchor is a positive anchor $( \hat { r } _ { i } = 1 )$ and we collect its positive placement set $\hat { \mathcal { P } } _ { i }$ . Otherwise, this annotated anchor is a negative anchor $( \hat { r } _ { i } = 0 )$ . We use the supervision information of annotated anchors to train the model. However, the above supervision information is very noisy. In particular, the negative anchors may also have unannotated positive placements, which should actually be positive anchors. Besides, the positive anchors could have many potential positive placements beyond the annotated ones. Therefore, we add a self-training stage to retrain the model using updated supervision information.

In the self-training stage, we update the annotations as follows. The original positive anchors are still positive. For the other anchors, if their predicted rationality scores $r _ { i }$ are larger than a threshold $\tau _ { c } .$ we set them as positive anchors. All the other anchors are set as negative anchors. For the positive anchors, we augment their positive placement sets $\mathcal { \hat { P } } _ { i }$ . Specifically, for each positive anchor, we randomly sample 50 random vectors z and collect the placements whose classification scores predicted by D are larger than a threshold $\tau _ { s }$ . We add these placements to $\hat { \mathcal { P } } _ { i }$ . If the size of augmented $\hat { \mathcal { P } } _ { i }$ exceeds $T ,$ , we adopt the top $T$ placement with the highest classification scores. Finally, we retrain our model using the updated annotations.

## 4. Experiments

## 4.1. Dataset

Following [32], we conduct experiments on OPA dataset [17], which contains composite images with annotated binary rationality labels. In detail, OPA dataset contains 62, 074 (resp., 11, 396) annotated composite images for training (resp., testing). The training (resp., test) set consists of 21, 376 (resp., 3, 588) positive and 40, 698 (resp., 7, 808) negative samples. There are 4, 137 different foreground objects from 47 categories and 1, 389 different background images in the dataset. The foregrounds/backgrounds in the training set and test set have no overlap.

To enable our model to be trained on the training set, we first transform the annotation format of the training set. Specifically, given a pair of foreground and background, we collect the annotated composite images (positive or negative composite images) with the given foreground/background, and obtain the annotated placements (positive or negative placements), because one composite image corresponds to one foreground placement. If an anchor is associated with annotated placements, that is, there exists annotated placement whose foreground center is in the neighborhood of this anchor, we refer to this anchor as annotated anchor. If an annotated anchor has at least one positive placement, this anchor is a positive anchor and we collect its associated positive placements as the ground-truth placement set. Otherwise, this annotated anchor is a negative anchor. After acquiring the positive/negative anchors and the ground-truth placement sets of positive anchors, we train our model on the transformed training set.

## 4.2. Evaluation Metrics

For model evaluation, we strictly follow the setting in [32]. In particular, our model takes one test pair (foreground-background pair of each positive composite image in the test set) as input, and generates composite images. Following [32], we adopt user study, accuracy, FID [11] to evaluate the plausibility of generated composite images, and adopt LPIPS [30] to evaluate the diversity of generated composite images.

For plausibility evaluation, our model takes in one test pair and generates a composite image. Specifically, we choose the positive anchor with the highest rationality score and randomly sample one z to generate a composite image. We use the binary classifier provided by [32] to predict the rationality of generated composite images. Accuracy is defined as the proportion of generated composite images that are classified as positive by the binary classifier. FID is calculated between generated composite images and positive composite images in the test set. User study shows the proportion that each method is selected as the best one (see details in Supplementary).

Table 1. Quantitative comparison of different methods on OPA dataset.
<table><tr><td rowspan="2">Method</td><td colspan="3">Plausibility</td><td>Diversity</td><td colspan="2">Complexity</td></tr><tr><td>User Study↑</td><td>Acc.↑</td><td>FID↓</td><td>LPIPS↑</td><td>FLOPs(G)</td><td>Time(s)</td></tr><tr><td>TERSE [25]</td><td>0.104</td><td>0.679</td><td>46.94</td><td>0</td><td>37.25</td><td>0.040</td></tr><tr><td>PlaceNet [29]</td><td>0.109</td><td>0.683</td><td>36.69</td><td>0.160</td><td>4.76</td><td>0.039</td></tr><tr><td>GracoNet [32]</td><td>0.183</td><td>0.847</td><td>27.75</td><td>0.206</td><td>43.98</td><td>0.044</td></tr><tr><td>FOPA [19]</td><td>0.303</td><td>0.896</td><td>25.91</td><td>0.216</td><td>321.05</td><td>0.258</td></tr><tr><td>Ours</td><td>0.301</td><td>0.890</td><td>26.02</td><td>0.218</td><td>39.49</td><td>0.041</td></tr></table>

For diversity evaluation, our model takes in a test pair and generates 10 composite images by randomly selecting 10 positive anchors and randomly sample one z for each positive anchor. For each test pair, we compute the average pairwise LPIPS of 10 generated composite images. After that, we calculate the average of LPIPS over all test pairs.

## 4.3. Implementation Details

The foreground encoder $E ^ { f }$ and the background encoder $E ^ { b }$ have the same backbone network and two separate heads. The shared backbone contains the beginning 34 layers (including and before the fourth MaxPool layer) of VGG16-BN [22] pretrained on ImageNet [7]. Both the foreground head and the background head contain three groups of (3 × 3 Conv, BN, ReLU). The foreground head is ended with another GlobalAvgPool to extract the foreground feature vector, while the background head is ended with another $h \times w$ AdaptiveAvgPool to extract the background feature map. Images are resized to 256 × 256 and normalized before we fed them into the network. We train the model with batch size 16 for 20 epochs (including the self-training stage) on a single RTX 3090 GPU. We adopt Adam optimizer with the initial learning rates being $2 \times 1 0 ^ { - 5 }$ for backbone and $2 \times 1 0 ^ { - 4 }$ for the remaining parts. For hyper-parameters, we set the number of anchors as $n = h \times w = 8 \times 8 = 6 4$ , the sampling times for each anchor as $T = 1 0$ . In Eqn. 7, we set $\lambda _ { 1 } = 1 0 , \lambda _ { 2 } = 5$ . In the self-training stage, we set $\tau _ { c } = 0 . 6 , \tau _ { s } = 0 . 8 5$

## 4.4. Comparison with Baselines

We compare with two groups of baselines: generative object placement methods and discriminative object placement methods. For generative methods, we compare with TERSE [25], PlaceNet [29], and GracoNet [32]. They take in a pair of foreground and background, generating one or multiple composite images, which is similar to our method. When evaluating diversity (resp., plausibility), [25, 32] produce 10 (resp., 1) composite images by sampling the random vector for 10 (resp., 1) times. Note that TERSE [25] does not have random vector and could only produce one composite image, so its LPIPS is 0.

For discriminative method, we compare with FOPA [19], which requires a background and a scaled foreground as input to produce a rationality score map. As described in [19], we predict 16 rationality score maps for 16 discrete foreground scales, and obtain one composite image with the highest pixel-wise rationality score across 16 maps. When evaluating the diversity, we randomly choose 10 composite images whose rationality scores exceed the threshold 0.5.

We also compare the efficiency between different methods. We report FLOPS and inference time taken to process a foreground-background pair. We test the inference time on a single RTX 3090 GPU with PyTorch-1.7 [20], by running over all test pairs and calculating the average.

The results of different methods are summarized in Table 1. For plausibility evaluation, accuracy is more reliable than FID. FOPA outperforms generative methods on accuracy by a large margin. For diversity evaluation, LPIPS of FOPA does not surpass GracoNet significantly. One possible reason is that GracoNet tends to generate large composite foregrounds (see Figure 4), and the relocation of large composite foreground leads to high LPIPS. Moreover, merely persuing high LPIPS is not meaningful because randomly placing the composite foreground could also achieve high LPIPS. For efficiency, the computational cost of FOPA is significantly larger than generative methods, due to its complex network structure.

Based on Table 1, our method outperforms generative methods significantly and achieves comparable results with FOPA. In terms of efficiency, our method is comparable with GracoNet and requires much less computational cost than FOPA. Therefore, our method strikes a good balance between efficiency and effectiveness.

## 4.5. Ablation Studies

We evaluate the effectiveness of each component and loss term in Table 2. First, we replace deformable convolution in our FAD module with vanilla $3 \times 3$ convolution, leading to the results in row 1. The performance becomes worse, which demonstrates that it is helpful to dynamically aggregate contextual information. Then, we remove the mode seeking loss $\mathcal { L } _ { m s }$ (row 2), after which the diversity is significantly degraded. Note that the predicted place-

GracoNet  
![](images/a4814c40701d4f49cdf91eeb836c7f250954a048cf368b55144f0023982eaa7a.jpg)  
FOPA

![](images/e4e10361d6a3c517d08e1daa3346c09458119564d871ed5541f2fc602f4dfdbc.jpg)

Ours  
![](images/87355abbd538f49ebe1e4c10a76848e02cab06f5cbaa50e5999741db3913f5c0.jpg)  
Figure 4. For each foreground-background pair, we show three composite images generated by GracoNet [32], FOPA [19], and our method.

Table 2. Quantitative comparison of different ablated versions of our method on OPA dataset.
<table><tr><td rowspan=2 colspan=1>Row</td><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=2>Plausibility</td><td rowspan=1 colspan=1>Diversity</td></tr><tr><td rowspan=1 colspan=1>Acc.↑</td><td rowspan=1 colspan=1>FID↓</td><td rowspan=1 colspan=1>LPIPS↑</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>w/o DC</td><td rowspan=1 colspan=1>0.871</td><td rowspan=1 colspan=1>26.85</td><td rowspan=1 colspan=1>0.212</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>w/o $\mathcal { L } _ { m s }$ </td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>27.67</td><td rowspan=1 colspan=1>0.176</td></tr><tr><td rowspan=2 colspan=1>34</td><td rowspan=1 colspan=1>w/o $\mathcal { L } _ { p l }$ </td><td rowspan=1 colspan=1>0.832</td><td rowspan=1 colspan=1>28.83</td><td rowspan=1 colspan=1>0.196</td></tr><tr><td rowspan=1 colspan=1>w/o $\mathcal { L } _ { c l }$ </td><td rowspan=1 colspan=1>0.712</td><td rowspan=1 colspan=1>44.86</td><td rowspan=1 colspan=1>0.188</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1st stage</td><td rowspan=1 colspan=1>0.869</td><td rowspan=1 colspan=1>26.93</td><td rowspan=1 colspan=1>0.211</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>26.02</td><td rowspan=1 colspan=1>0.218</td></tr></table>

ments of positive anchors are supervised by the matching loss $\mathcal { L } _ { p l }$ and the classification loss $\mathcal { L } _ { c l }$ . We remove either of them (row 3-4), which shows that both loss terms contribute to the performance. $\mathcal { L } _ { c l }$ plays a more important role than $\mathcal { L } _ { p l }$ , because it pushes all generated composites images from positive anchors to be positive.

Recall that we take a two-stage training strategy. We also report the performance in the first stage without selftraining. By comparing the performances between two stages (row 5 v.s. row 6), we conclude that the self-training stage could boost the performance by refining the annotations.

Foreground  
Background  
![](images/4183982ed871f6b93e76901f62b9bbf4014515e74f6d34c8cfa01a9ac8a8fda5.jpg)

FOPA  
Ours  
![](images/1be08b09d06f28b80d7d164a36a966fe965680d98afe5a2ee544fc378a9df3f0.jpg)  
Figure 5. The comparison of rationality score map between FOPA [19] and our method. From left to right, we show the foreground, background, the rationality score map generated by FOPA and our method.

## 4.6. Visualization Results

Generated composite images: We show the composite images generated by different methods in Figure 4. We compare with the competitive generative baseline GracoNet (see Table 1) and the discriminative baseline FOPA. Given a pair of foreground and background, we show three composite images generated by two baselines and our method. The details of generating multiple placements have been introduced in Section 4.2 for our method and Section 4.4 for [32, 19].

![](images/9279bb682ebb3c5a83375611dff46ae784b47c14b66be2b7c86f37ef4304b2f2.jpg)  
Figure 6. For each pair of foreground and background, we show three generated composite images for the same positive anchor with the highest rationality score. The neighborhood of each positive anchor is marked with red box on the background.

We observe that GracoNet tends to generate large and unreasonable composite foreground (row 2, 3). The location diversity is limited compared with the other two methods. For example, GracoNet places the sheep on the right (row 4) and the bicycle surrounding the person (row 6), despite the fact that there are many other reasonable locations. Compared with GracoNet, the generated composite images of FOPA have higher quality and cover more reasonable locations. However, there still exist unreasonable occlusions (e.g., airplane occludes the traffic light in row 5) and unreasonable locations (e.g., chair in the water in row 1). Our method achieves comparable or better results than FOPA, and has the ability to generate diverse and plausible placements. Moreover, our method is significantly faster than FOPA (see Section 4.4). Thus, our method can balance effectiveness and efficiency better than previous generative and discriminative methods.

Rationality score map: In Figure 5, we show the rationality score map indicating positive anchors and negative anchors. Recall that positive anchor means that there exists at least one positive placement with foreground center in its neighborhood. Based on the rationality score map, our method can roughly tell the positive anchors from negative anchors.

We also compare with the rationality score map predicted by FOPA [19]. Note that the meanings of rationality score map of our method and [19] are different. [19] predicts one rationality score map for each foreground scale, in which each entry represents the rationality of composite image obtained by moving the center of scaled foreground to this location. For meaningful comparison, we first generate 16 rationality score maps with size $H \times W$ for 16 discrete foreground scales and concatenate them channel-wisely, resulting in a rationality score tensor. Then, we perform max pooling within each $\frac { \check { H } } { h } \times \frac { W } { w } \times 1 6$ volume and get the pooled rationality score map with size $h \times w$ , because the maximum value within each volume roughly indicates the existence of positive placement for this anchor. We compare the rationality score maps between FOPA and our method in Figure 5. It can be seen that the positive anchors (e.g., sky for the airplane in row 1, unoccupied grassland for the sheep in row 2) identified by two methods basically coincide with each other. In some cases, our method can predict better rationality score map. For example, in row 3, our method identifies the free space on the table while FOPA includes extra unreasonable locations. The above visualization results demonstrate the effectiveness of our predicted rationality score maps.

Diversity for one anchor: To better demonstrate the diversity of our generated composite images, we show multiple results for the same positive anchor in Figure 6. Given a pair of foreground and background, we find the positive anchor with the largest rationality score. Then, we randomly sample z three times for this positive anchor and generate three composite images. It can be seen that the foreground center lies in the neighborhood of positive anchor. Under this constraint, the generated composite images still exhibit diversity in terms of scale and location.

Offsets in FAD module: We visualize the learnt offsets in our FAD module. We observe that the learnt offsets vary dynamically according to the foreground information and background context. The detailed results are left to Supplementary due to the space limitation.

## 4.7. Hyper-parameter Analyses and Failure Cases

We investigate the impact of hyper-parameters in our method and show several failure cases of our method, which can be found in Supplementary.

## 5. Conclusion

In this work, we have developed an anchor-based semigenerative object placement learning method, aiming to combine the advantages of generative methods and discriminative methods. Given a pair of foreground and background, our method can produce high-quality rationality score map as well as diverse and plausible placement sets. Comprehensive experiments on OPA dataset have demonstrated the superiority of our proposed method.

## References

[1] Samaneh Azadi, Deepak Pathak, Sayna Ebrahimi, and Trevor Darrell. Compositional gan: Learning imageconditional binary composition. International Journal of Computer Vision, 128(10):2570–2585, 2020. 2

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In ECCV, 2020. 4, 5

[3] Yun Chen, Frieda Rong, Shivam Duggal, Shenlong Wang, Xinchen Yan, Sivabalan Manivasagam, Shangjie Xue, Ersin Yumer, and Raquel Urtasun. Geosim: Realistic video simulation via geometry-aware composition for self-driving. In CVPR, 2021. 1

[4] Wenyan Cong, Jianfu Zhang, Li Niu, Liu Liu, Zhixin Ling, Weiyuan Li, and Liqing Zhang. Dovenet: Deep image harmonization via domain verification. In CVPR, 2020. 1, 2

[5] Xiaodong Cun and Chi-Man Pun. Improving the harmony of the composite image by spatial-separated attention module. IEEE Transactions on Image Processing, 29:4759– 4771, 2020. 1, 2

[6] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In ICCV, 2017. 4

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 6

[8] Nikita Dvornik, Julien Mairal, and Cordelia Schmid. Modeling visual context is key to augmenting object detection datasets. In ECCV, 2018. 2

[9] Georgios Georgakis, Arsalan Mousavian, Alexander C. Berg, and Jana Kosecka. Synthesizing training data for object detection in indoor scenes. In RSS, 2017. 1

[10] Zonghui Guo, Haiyong Zheng, Yufeng Jiang, Zhaorui Gu, and Bing Zheng. Intrinsic image harmonization. In CVPR, 2021. 2

[11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5

[12] Yan Hong, Li Niu, and Jianfu Zhang. Shadow generation for composite image in real-world scenes. In AAAI, 2022. 1, 2

[13] Donghoon Lee, Sifei Liu, Jinwei Gu, Ming-Yu Liu, Ming-Hsuan Yang, and Jan Kautz. Context-aware synthesis and placement of object instances. In NeurIPS, 2018. 2

[14] Chen-Hsuan Lin, Ersin Yumer, Oliver Wang, Eli Shechtman, and Simon Lucey. ST-GAN: spatial transformer generative adversarial networks for image compositing. In CVPR, 2018. 2

[15] Jun Ling, Han Xue, Li Song, Rong Xie, and Xiao Gu. Region-aware adaptive instance normalization for image harmonization. In CVPR, 2021. 2

[16] Daquan Liu, Chengjiang Long, Hongpan Zhang, Hanning Yu, Xinzhi Dong, and Chunxia Xiao. ARShadowGAN: Shadow generative adversarial network for augmented reality in single light scenes. In CVPR, 2020. 1, 2

[17] Liu Liu, Bo Zhang, Jiangtong Li, Li Niu, Qingyang Liu, and Liqing Zhang. OPA: Object placement assessment dataset. arXiv preprint arXiv:2107.01889, 2021. 2, 5

[18] Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode seeking generative adversarial networks for diverse image synthesis. In CVPR, 2019. 4

[19] Li Niu, Qingyang Liu, Zhenchen Liu, and Jiangtong Li. Fast object placement assessment. arXiv preprint arXiv:2205.14280, 2022. 2, 3, 6, 7, 8

[20] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 6

[21] Tal Remez, Jonathan Huang, and Matthew Brown. Learning to segment via cut-and-paste. In ECCV, 2018. 1

[22] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015. 6

[23] Konstantin Sofiiuk, Polina Popenova, and Anton Konushin. Foreground-aware semantic representations for image harmonization. In WACV, 2021. 2

[24] Fuwen Tan, Crispin Bernier, Benjamin Cohen, Vicente Ordonez, and Connelly Barnes. Where and who? Automatic semantic-aware person composition. In WACV, 2018. 1, 2

[25] Shashank Tripathi, Siddhartha Chandra, Amit Agrawal, Ambrish Tyagi, James M. Rehg, and Visesh Chari. Learning to generate synthetic data via compositing. In CVPR, 2019. 1, 2, 6

[26] Yi-Hsuan Tsai, Xiaohui Shen, Zhe Lin, Kalyan Sunkavalli, Xin Lu, and Ming-Hsuan Yang. Deep image harmonization. In CVPR, 2017. 1, 2

[27] Shuchen Weng, Wenbo Li, Dawei Li, Hongxia Jin, and Boxin Shi. MISC: Multi-condition injection and spatiallyadaptive compositing for conditional person image synthesis. In CVPR, 2020. 1

[28] Fangneng Zhan, Hongyuan Zhu, and Shijian Lu. Spatial fusion gan for image synthesis. In CVPR, 2019. 2

[29] Lingzhi Zhang, Tarmily Wen, Jie Min, Jiancong Wang, David Han, and Jianbo Shi. Learning object placement by inpainting for compositional data augmentation. In ECCV, 2020. 1, 2, 3, 6

[30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5

[31] Song-Hai Zhang, Zhengping Zhou, Bin Liu, Xi Dong, and Peter Hall. What and where: A context-based recommendation system for object insertion. Computational Visual Media, 6(1):79–93, 2020. 1, 2

[32] Siyuan Zhou, Liu Liu, Li Niu, and Liqing Zhang. Learning object placement via dual-path graph completion. ECCV, 2022. 2, 3, 5, 6, 7, 8

# Supplementary for Hybrid Generative-Discriminative Object Placement

Siyuan Zhou Shanghai Jiao Tong University ssluvble@sjtu.edu.cn

Li Niu Shanghai Jiao Tong University ustcnewly@sjtu.edu.cn

In this document, we provide additional materials to supplement the main paper. In Section 1, we investigate the impact of hyper-parameters. In Section 2, we provide the details for user study. In Section 3, we provide more visualization results compared with competitive baselines. In Section 4, we visualize the offsets predicted by our dynamic fusion module. In Section 5, we provide more visualization results of our method. In Section 6, we discuss the limitation of our method.

## 1. Hyper-parameter Analyses

In this section, we investigate the impact of hyperparameters used in our method.

First, we explore spatial size $n = h \times w$ of the fused feature map ${ \pmb F } ^ { m }$ , which decides the number of anchors. We set $n = h \times w = 8 \times 8 = 6 4$ by default. Recall that the spatial size of extracted background feature map is $3 2 \times 3 2$ and we use a $h \times w$ AdaptiveAvgPool to downsample the background feature map to $h \times w .$ , as introduced in Section 4.3 in the main paper. We vary n in $[ 2 ^ { 2 } , 4 ^ { 2 } , 8 ^ { 2 } , 1 6 ^ { 2 } , 3 2 ^ { 2 } ]$ ] and plot the results in Figure 1, from which we observe that the performance first increases and then decreases. When the number of anchors is too small $( e . g . , n = 2 ^ { 2 } )$ , the proposed method gets close to the generative method and the performance becomes worse, probably because of the difficulty in predicting plausible center offsets and foreground scales within a large scope. When the number of anchors is too large $( e . g . , n = 3 2 ^ { 2 } )$ , each pixel-wise feature vector in the $h \times w$ background feature map may lack sufficient contextual information, leading to the performance drop. Therefore, we opt for $n = 8 ^ { 2 }$ by default.

We also study the effect of sampling times $T$ for each positive anchor. In OPA dataset [1], we observe that one anchor in a test pair has at most 3 positive placements, that is, $K _ { i } \leqslant 3 .$ , ∀i. With the constraint that $T \geqslant K _ { i }$ , ∀i (see Section 3.3 in the main paper), we vary T in [5, 10, 15, 20, 25] and plot the results in Figure 1. We observe that as T increases, the performance first gets better and then becomes stable. One possible explanation is that when T is sufficiently large $( T \geqslant 1 0 )$ , our model pushes a wide range of generated composite images to be plausible and different from each other, which could benefit the plausibility and diversity of prediction results. Thus, we set $T = 1 0$ by default.

![](images/ce55645fb26636953e1c203b5c182baa7cc149c2f89e1d65d06f6bae4fec7199.jpg)

![](images/ffc76951c8dea98d980e8cfd454cdd4167a119705a7343423443ccb59ba74ed2.jpg)

![](images/ed4565bc89fc9f86c7beb74788e3513e5b571237076cd6142947eb38464403ea.jpg)

![](images/1a6841dca26f927602a4f5a4d9df5244b47d6df6f2dcf56c1772133bbe38de7b.jpg)

![](images/4e039591208790ad0299442ae8d6006fc1ee1b99a8e8e64ad27ec7419dbaea70.jpg)

![](images/9344ca125b941b12987572433e1ff0f5779c29356d38edf35b0d6a36bde98173.jpg)  
Figure 1. Accuracy and LPIPS of our method when varying hyperparameters n, T, τ<sub>c</sub>, τ<sub>s</sub>, λ<sub>1</sub>, and $\lambda _ { 2 }$

Then, we study the impact of two thresholds $\tau _ { c } ,$ $\tau _ { s }$ used in the self-training stage, which are set as 0.6 and 0.85 by default respectively. We vary $\tau _ { c }$ in [0.3, 0.4, 0.5, 0.6, 0.7] when fixing $\tau _ { s } = 0 . 8 5$ , and vary $\tau _ { s }$ in [0.55, 0.65, 0.75, 0.85, 0.95] when fixing $\tau _ { c } = 0 . 6$ . The results are plotted in Figure 1. When $\tau _ { c }$ is mall, accuracy is low because the added positive anchors could be very noisy and mislead the training process, but LPIPS is high probably due to the randomness introduced by noisy positive anchors. For small $\tau _ { s } ,$ , we have similar observation as $\tau _ { c } .$ In particular, when $\tau _ { s }$ is mall, accuracy is low because the added positive placements could be very noisy and mislead the training process, but LPIPS is high probably due to the randomness introduced by noisy positive placements.

GracoNet  
FOPA  
![](images/8bb3249896fabb3bc5ba35e75a893494dc3fa99064128241159dee9eb6f60f16.jpg)  
Ours  
Figure 2. For each foreground-background pair, we show three composite images generated by GracoNet [5], FOPA [2], and our method.

Finally, we study the impact of $\lambda _ { 1 }$ and $\lambda _ { 2 }$ in Eqn. 7 in the main paper. We vary $\lambda _ { 1 }$ (resp., $\lambda _ { 2 } )$ in the range of $[ 1 0 ^ { - 2 }$ , 10<sup>2</sup>] while fixing $\lambda _ { 2 } \left( r e s p . , \lambda _ { 1 } \right)$ as the default value 5 (resp., 10). The results are plotted in Figure 1, from which we observe that our method is relatively robust to these two hyper-parameters.

## 2. Details of User Study

When evaluating the plausibility of generated composite images, we adopt user study as one of the evaluation metrics. Specifically, we ask 50 voluntary participants to compare the composite images generated by TERSE [3], PlaceNet [4], GracoNet [5], FOPA [2], and our method. For each test sample, every participant selects the method which generates the most plausible composite image. Then, we calculate the proportion that each method is selected as the best one among 50 participants. Finally, we calculate the averaged proportion over all test samples as the user study score for each method, as reported in Table 1 in the main paper.

Foreground  
Offset  
![](images/14a42c1131643d99f22a6cac9c7297c066467abf2bfbd6f0b4965b85f2abf69b.jpg)

Foreground  
![](images/bf675f158b603fd40e8924ea20107198da3d6bca4dde7d09e44c935063faeac1.jpg)  
Offset  
Figure 3. Visualization of the convolution kernel offsets (red arrows) predicted by our dynamic fusion module (DFM).

## 3. More Visual Comparison with Baselines

We provide more visualization results to compare with the competitive baselines: GracoNet [5] and FOPA [2]. In Figure 2, we show three generated composite images for each method.

We observe that GracoNet is prone to generate overlarge composite foreground (e.g., row 1, 2) and the location diversity is insufficient (e.g., row 3, 5). FOPA is able to produce more plausible and diverse composite images. Nevertheless, the foreground could be placed at unreasonable locations (e.g., the cup on the food in row 1) or with unreasonable scales (e.g., the bus in front of the truck is too small). The plausibility and diversity of composite images generated by our method is comparable or better than those of FOPA. Moreover, our method is much more efficient than FOPA (see Table 1 in the main paper). Therefore, our method can strike a good balance effectiveness and efficiency, compared with previous methods.

![](images/5374c5c7f46625a16170ad566f0a73385384424fd5cc68853988f22797e63626.jpg)  
Figure 4. For each foreground-background pair, we show 10 composite images generated by our method. Two composite images generated from the same positive anchor are grouped together.

![](images/60949c9d50bcde475d9aa21e744a5186383848c838817d2fb882e87cbabd61bc.jpg)  
Figure 5. Two example failure cases of our method.

## 4. Offset Visualization

Recall that we propose Dynamic Fusion Module (DFM) which dynamically aggregates different scales of contextual information based on foreground and background information. Specifically, we predict the offsets for convolution kernels based on the concatenation of background feature map and replicated foreground feature vector. In this section, we visualize the offsets predicted by our DFM in Figure 3, to show that they depend on foreground information and background context. In each example, given a pair of foreground and background, we choose an anchor location and visualize the predicted offsets for the convolution kernel centered at this location.

By comparing two examples in the left column, we observe that at the same location on the same background, the physical size of foreground object affects the predicted offsets. When the foreground object has large physical size (e.g., zebra v.s. dog), the predicted offsets are also large. Intuitively, for the large-scale foreground object, we need to speculate large-scale contextual information in the background (e.g., check the existence of unreasonable occlusion between the foreground object and background objects).

By comparing two examples in the right column, we observe that for the same foreground and background, the location of kernel center also affects the predicted offsets. When there exists unreasonable occluder (e.g., fence) near the kernel center, one interesting observation is that the predicted offsets pay more attention to the occluders, probably because the detected unreasonable occluder is a strong evidence that this anchor is a negative anchor.

The above visualization results show that the predicted offsets depend on the foreground information and background context, which justifies our motivation to dynamically aggregate different scales of background contextual information.

## 5. More Visualization Results of Our Method

In this section, we provide more visualization results of our method. To better demonstrate the plausibility and diversity of our generated composite images, given a pair of foreground and background, we first randomly select five positive anchors (predicted rationality score larger than 0.5), and then generate two composite images for each positive anchor by randomly sampling the random vector twice. The generated composite images are shown in Figure 4, and it can be seen that our method can usually generate diverse and plausible composite images considering different reasonable locations and scales.

## 6. Failure Case

Although our method can usually produce satisfactory results, our method may fail to predict reasonable scale and location in some challenging cases. For example, as shown in Figure 5, when the background is very cluttered and there is only limited space (e.g., desk in row 2) for plausible placement, the results of our method are unsatisfactory.

## References

[1] Liu Liu, Bo Zhang, Jiangtong Li, Li Niu, Qingyang Liu, and Liqing Zhang. OPA: Object placement assessment dataset. arXiv preprint arXiv:2107.01889, 2021. 1

[2] Li Niu, Qingyang Liu, Zhenchen Liu, and Jiangtong Li. Fast object placement assessment. arXiv preprint arXiv:2205.14280, 2022. 2

[3] Shashank Tripathi, Siddhartha Chandra, Amit Agrawal, Ambrish Tyagi, James M. Rehg, and Visesh Chari. Learning to generate synthetic data via compositing. In CVPR, 2019. 2

[4] Lingzhi Zhang, Tarmily Wen, Jie Min, Jiancong Wang, David Han, and Jianbo Shi. Learning object placement by inpainting for compositional data augmentation. In ECCV, 2020. 2

[5] Siyuan Zhou, Liu Liu, Li Niu, and Liqing Zhang. Learning object placement via dual-path graph completion. ECCV, 2022. 2