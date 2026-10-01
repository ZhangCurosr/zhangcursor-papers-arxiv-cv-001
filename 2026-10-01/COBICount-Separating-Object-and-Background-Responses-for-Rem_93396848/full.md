Zhiyi Zhou is with the School of International Studies, Chongqing University of Posts and Telecommunications, Chongqing 400065, China, and also with Brunel University London, Uxbridge UB8 3PH, U.K. (email: 2378903@brunel.ac.uk).

# COBICount: Separating Object and Background Responses for Remote Sensing Object Counting Without Training on Target Data

Junjing Zheng , Zhiyi Zhou , Ningrui Yang , and Hongying Meng , Senior Member, IEEE

Abstract—Remote sensing object counting estimates how many buildings, vehicles, or ships appear in overhead images. Most supervised counters predict a density map, whose sum gives the object count, and assume similar categories, sizes, and backgrounds. Applying them across regions, sensors, or categories often requires target data or further training, which may be costly or unavailable. We study source-only counting. Training for the counting task and model selection use one group of images that shares an object category and similar imaging conditions, with one point marking each object. Target images and information remain unavailable until the model is fixed. This reduces data preparation but makes transfer harder. A model trained on one source may place high density values, called responses, on real objects and repeated background structures. Road edges, parking grids, roof boundaries, and water boundaries may then be counted as objects, creating candidate origin ambiguity. COBICount separates response generation, acceptance, and background suppression. Candidate Evidence (CE) generates possible responses. Candidate Acceptance (CA) keeps compact responses centered on objects. Bias Isolation (BI) reduces responses associated with repeated background structures. Their outputs form the final density map. Trained on RSOC Building and evaluated directly on DOTA Large Vehicle, Small Vehicle, and Ship, COBICount achieves the lowest mean absolute error (MAE) averaged over the target domains among the compared methods, 174.132. It uses 5.07 million parameters and 17.41 billion floating point operations for a 512 × 512 input. COBICount improves transfer without target data or training for each target. The code will be available at: https://github.com/yixuxi22/COBICount.

Index Terms—Remote sensing object counting, source-only counting, candidate origin ambiguity, density map estimation, model generalization, transfer across domains.

## I. INTRODUCTION

Junjing Zheng is with the School of International Studies, Chongqing University of Posts and Telecommunications, Chongqing 400065, China, and also with Brunel University London, Uxbridge UB8 3PH, U.K. (email: 2378933@brunel.ac.uk).

R <sup>EMOTE</sup> <sup>sensing</sup> <sup>object</sup> <sup>counting</sup> <sup>estimates</sup> <sup>how</sup> <sup>many</sup> instances of a selected class, such as buildings, vehicles, or ships, appear in an aerial or satellite image. It supports urban mapping, traffic monitoring, maritime surveillance, and infrastructure assessment. One approach detects each object and then counts the detections [1], [2]. Another uses point supervision, where each training object is marked by one point. A counter learns a nonnegative density map that distributes the object count across spatial locations; summing all map values gives the number of objects in the image [3]–[5]. Point annotations require less detail than bounding boxes, which makes them useful for scenes containing many objects. Remote sensing images remain difficult because their overhead views, image resolutions, object sizes, and background layouts vary widely. Accurate counting requires both a correct total and high local density values placed on true objects. We refer to these local values as counting responses.

Most supervised counters are developed in a setting where training and test images come from the same domain and contain similar object classes, sizes, and scene patterns. Methods based on density estimation, point prediction, local context, and transformers have improved counting under this condition [6]–[10]. Remote sensing methods also address changes in object size and complex backgrounds within annotated datasets [11]–[13]. This setting is useful for a fixed task, but it can hide dependence on the training data. A model trained on buildings may learn not only building appearance but also roof edges, block layouts, and nearby roads. These patterns may change when the model is applied to another region, sensor, image resolution, or object class. Good accuracy on the annotated training data does not show whether the same responses will remain valid after such a change.

When training and deployment images differ in object category or imaging conditions, they belong to different domains. A domain is a group of images that share the same object category and similar imaging and scene conditions; it is not an ordinary training, validation, or test partition. The source domain provides the point annotations used during model development. Its training split updates the model, its validation split selects the final model, and a separate test split with the same category and imaging conditions still measures performance in the source domain. A target domain has a different object category or different imaging conditions and is unavailable during model development.

Existing approaches to transfer between domains use different forms of additional data. Domain adaptation requires labeled or unlabeled target images during training [14], [15]. Domain generalization does not use target images during training, but its conventional form often requires several annotated source domains [16], [17]. Pretrained vision models first learn from large external datasets and can later provide reusable features or masks [18]–[21]. These outputs do not directly form a density map learned from point annotations or show whether a local counting response lies on a true object. These data requirements can prevent deployment when representative target images cannot be collected in advance, several annotated source domains are unavailable, or adaptation for every new region, sensor, or object category is impractical.

For this reason, training for the counting task and model selection use only one source domain, with one point annotation for each object and no target data. This setting reduces the data that must be prepared before deployment and allows a fixed model to be applied to new domains without further training. The choice of source and target domains depends on the experimental protocol and is not fixed. In the main experiment, RSOC Building is used as the source domain, and DOTA Large Vehicle, Small Vehicle, and Ship are used as three unseen target domains. Other experiments use different combinations of source and target domains; for example, RSOC Building remains the source domain when DIOR Airplane is evaluated as an unseen target domain. We call this setting source-only counting. It limits the data used for the counting task. A generic initialization fixed before source training may be used when it is part of a baseline’s standard implementation, but it is neither selected nor updated with target data. No target images or information derived from them, including labels, automatically generated labels, feature summaries, values used to normalize the data, validation results, or signals used to choose the model, are used before the final source model is fixed. Its parameters are not changed afterward. Target images are then used only to produce predictions, and target annotations are used only by the evaluator to calculate performance metrics.

This reduced dependence on target data makes transfer more difficult. Source point annotations mark object centers, but they do not label roads, parking spaces, roof boundaries, water boundaries, or other background regions. The model may respond to both the annotated objects and background structures that repeatedly occur in the source images. After the domain changes, one high response may be correct because it is centered on a real target object and forms the compact local pattern encouraged around source point annotations. Another high response may be incorrect because a road edge, parking grid, roof boundary, or harbor structure forms a similar pattern. The response value alone does not reveal which case produced it. We call this problem candidate origin ambiguity. A change in the spatial size of a response adds another difficulty. A source building may cover a broad area in the network output, whereas a small target vehicle may occupy only a few locations. The vehicle response can then be weaker than nearby parking or road patterns. Mean absolute error (MAE) and root mean squared error (RMSE) cannot fully reveal these failures because they measure only the total count. Missed objects and false background responses may partly cancel.

We propose COBICount to decide which local responses should contribute to the count. Rather than asking one prediction branch to generate and accept every response at once, COBICount separates response generation, acceptance, and background suppression. Candidate Evidence (CE) produces a broad nonnegative map of possible responses so that weak object responses can enter the counting process. Candidate Acceptance (CA) keeps responses that match the compact and centered patterns learned around source point annotations. Bias Isolation (BI) reduces responses associated with learned background structures, edges, broad responses not centered on objects, and repeated grids. A fixed mask identifies valid image regions and removes the black padding outside them. These outputs form the final density map, whose sum gives the predicted count. All training targets are derived from source point annotations. Regions near these annotations show where responses should remain, while valid image regions farther away show where responses should be suppressed. BI does not identify named background categories in a target domain. An auxiliary inspection head, called Audit, is trained only with source annotations and is used to inspect response patterns after training. It does not control the final density map or use target annotations.

The main contributions are summarized as follows:

• We study remote sensing object counting under sourceonly counting. We clarify that training, validation, and test subsets with the same object category and imaging conditions belong to the same domain. Under this setting, candidate origin ambiguity explains why a model trained on one annotated source domain may produce incorrect counting responses in an unseen target domain.

• We introduce COBICount, which separates candidate generation, local acceptance, and background suppression through CE, CA, and BI. All supervision is derived from source point annotations. The model requires neither adaptation using target data nor labels for target background categories.

• We evaluate count accuracy and response location through a main comparison that includes recent generic baselines and baselines for remote sensing evaluated under source-only counting, tests that change the source domain, ablations, checks of whether predictions are too high or too low and whether responses in the final density map occur near reference points. When RSOC Building is the source and the three DOTA categories are unseen targets, COBICount obtains the lowest MAE averaged over the target domains among the compared methods.

## II. RELATED WORK

The studies most closely related to COBICount address three questions. First, how can remote sensing objects be counted when each training object is marked by only one point? Second, what data do existing methods require when training and deployment domains differ? Third, how can we check whether a counting response is located on an object rather than on a background structure? The following review discusses these questions in this order.

## A. Counting with Point Supervision in Remote Sensing Images

Remote sensing objects can be counted through detection or point supervision. Detection methods first locate individual objects and then use the number of detections as the count. DOTA [1], xView [2], DIOR [22], and FAIR1M [23] provide bounding box annotations for objects in overhead images. CARPK and PUCPR+ focus on vehicles observed from elevated viewpoints [24]. Bounding boxes describe both object position and extent, but collecting them is costly when an image contains many small objects. Counting with detection can also inherit errors from missed and duplicate detections.

Point supervision reduces this annotation burden by marking only one center point for each object. A common approach converts these points into a density map used as the training reference. A network predicts this map, and the count is obtained by summing its spatial values. Early crowd counting models established several parts of this formulation. The Multi-Column Convolutional Neural Network (MCNN) uses parallel convolutional columns to handle changes in object size [3]. CSRNet uses dilated convolutions to observe a larger image area without further reducing map resolution [4]. The Context Aware Network (CAN) selects information from the surrounding image for different regions [6]. Bayesian Loss accounts for uncertainty around point annotations [7], and a distribution matching method compares the spatial distributions of predicted and reference density values [8]. Other formulations predict object points directly, as in P2PNet [9], or use transformers to relate distant image regions, as in TransCrowd [10].

Objects in remote sensing images can vary greatly in size, and their backgrounds often contain repeated roads, roofs, parking lines, and boundaries. The Remote Sensing Object Counting (RSOC) dataset introduced subsets for buildings, ships, large vehicles, and small vehicles, with one point marking each object [5]. The Pyramidal Scale and Global Context Guided Network (PSGCNet) combines features computed at several image resolutions with information from the complete scene [11]. The Triple Attention and Scale Aware Network (TASNet) uses separate processing for different object sizes and directs the model toward useful features [12]. The Balanced Density Regression Network (BDRNet) combines density regression with an auxiliary object region output that predicts the image regions occupied by objects [25]. EdgeCount transfers density map knowledge from a larger network to a smaller network to reduce computation while retaining counting accuracy [13]. Other remote sensing tasks report related image difficulties. A study that estimates surface depth from two satellite views addresses changes in image resolution [26], and a study that labels water pixels in radar and optical images covers broad geographic areas [27]. These studies do not perform counting, but they show that object size, sensing method, and boundary appearance can change across overhead images.

Most counting methods above are trained and tested separately on each dataset or object category. They explain how to produce an accurate count when training and test conditions are similar, but not how to decide whether a response remains valid after the category, sensor, resolution, or background changes. In particular, a low count error does not show whether high values in the predicted density map fall on the objects or on repeated background structures in the source images.

## B. Learning across Domains under Different Data Conditions

Methods for transfer between domains differ mainly in the data available before evaluation. Domain adaptation uses images from the target domain during model training, either with or without target labels. Adversarial domain training, for example, learns features that make source and target samples harder to distinguish [15]. Remote sensing studies have used adaptation to address changes in sensors, regions, and imaging conditions [14], including unsupervised adaptation for image segmentation, which assigns a class to each pixel [28]. Another adaptation setting removes access to the original source images during adaptation but still uses target images; remote sensing object detection provides one example [29]. Domain adaptation can reduce a known difference between source and target domains, but it requires target samples before deployment and usually repeats adaptation for each new target domain.

Domain generalization removes target domain images from training. Its conventional form instead learns from several annotated source domains. Meta Learning for Domain Generalization (MLDG) simulates domain changes by treating some available domains as temporary training domains and others as temporary test domains [16]. A method called gradient surgery reduces conflicts among updates from different source domains [17]. These methods avoid collecting target data, but several labeled source domains may also be difficult to obtain. Counting research has also considered more restricted training data. A domain general crowd counting method divides source crowd data into groups and separates information shared by the groups from information tied to each group [30]. MPCount studies crowd counting from one labeled domain and encourages the model to retain similar responses when source images are transformed [31]. This use of one labeled source is related to source-only counting, although MPCount was developed for people counting across crowd datasets. Its evaluation under the present remote sensing protocol is described in Section IV-B. Universal Representation Matching (URM) uses features from models trained to connect images and language, and it counts categories specified by a small set of target examples [32]. These methods improve generalization under limited counting data, but their original settings differ from ours. Crowd counting methods continue to count people, whereas methods that receive a small set of target examples use those examples to specify the category. Our setting fixes a model using one remote sensing source domain with one point per object and provides no target image or example before evaluation on new categories and imaging conditions.

Pretrained models offer another way to reduce the annotations required for a particular task. A pretrained model first learns from a large external collection and is then reused for a specific task. One model learns by matching image features with text descriptions [18], and the Segment Anything Model (SAM) produces image masks from prompts such as points or boxes [19]. Pretraining for remote sensing also considers properties of overhead data. SatMAE hides parts of satellite images and learns to reconstruct them; it uses images captured at several times and wavelengths [33]. Another pretraining method explicitly models changes in geographic scale [34]. SatMAE++ learns features at several spatial levels from images captured at multiple wavelengths [35]. CrossEarth studies features used for remote sensing segmentation across domains [36].

Models that connect vision and language have also been adapted to remote sensing tasks. RemoteCLIP and GeoRSCLIP learn links between overhead images and text [20], [21]. A language guided detector uses text to specify categories rather than relying on a fixed training list [37]. One study uses SAM to expand remote sensing segmentation data [38], and RSPrompter learns prompts that help separate individual objects in remote sensing images [39]. Other studies test how SAM transfers with no remote sensing example or only one labeled example [40]. These models can supply features that describe image content, detections, or masks, but those outputs do not by themselves define a count. A feature may indicate that an image contains vehicles without stating how many vehicles are present. A mask may cover one object, several touching objects, or a background region. Pretrained models therefore do not remove the need to place an appropriate counting response on each object and reject responses caused by background structures.

## C. Checking Where Counting Responses Occur

MAE and RMSE compare the predicted and reference totals, but they do not show where the error occurs. A missed object and an extra response on the background can offset each other and leave a seemingly accurate total. This failure is related to shortcut learning, in which a model reduces its training loss by using an easy recurring pattern rather than the intended object evidence [41]. In remote sensing images, such patterns may include roof edges around buildings, road or parking layouts around vehicles, and harbor or water boundaries around ships. When only one source domain is annotated, the model has no direct label stating that these background structures should not be counted.

General explanation methods provide only part of the required evidence. A gradient method for class activation mapping highlights image regions that influence a network decision [42], but its broad visual maps do not test whether a density map places a separate response on each object. An analysis designed for counting can instead compare response positions with annotated points. Responses near annotated points show where objects are covered, whereas responses far from every annotation reveal possible background responses. The fractions of annotated objects covered and responses supported by annotations can expose missed objects and false responses that total count errors may hide.

Existing studies mainly improve the predicted count or explain a trained model after the fact. COBICount connects model design with checking response positions. CE generates possible object responses, CA keeps responses that match compact and centered patterns learned from source point annotations, and BI reduces responses associated with repeated background structures. All three modules are trained from source images and point annotations. A later analysis checks whether the final responses occur near target objects, without changing the predicted count or supplying target domain information to the model.

## III. METHOD

## A. Problem Setting and Model Output

Source-only counting, introduced in Section I, is written formally here. Let

$$
\mathcal { D } _ { s } = \{ ( \mathbf { I } _ { i } ^ { s } , \mathcal { P } _ { i } ^ { s } ) \} _ { i = 1 } ^ { N _ { s } }\tag{1}
$$

be the source domain. It contains $N _ { s }$ images. For image $\mathbf { I } _ { i } ^ { s } ,$ $\mathcal { P } _ { i } ^ { s } = \{ \mathbf { p } _ { i j } \} _ { j = 1 } ^ { n _ { i } }$ contains one point for each of its $n _ { i }$ objects. A point is written as $\mathbf { p } _ { i j } = ( x _ { i j } , y _ { i j } )$ , where $x _ { i j }$ is the horizontal coordinate and $y _ { i j }$ is the vertical coordinate. When the point is used to access a tensor, the vertical coordinate gives the row index and the horizontal coordinate gives the column index. Therefore, ${ \bf X } ( { \bf p } _ { i j } )$ denotes the tensor entry ${ \bf X } ( y _ { i j } , x _ { i j } )$

Training and model selection use only $\mathcal { D } _ { s } .$ . Let $\mathcal { D } _ { t }$ denote an unseen target domain as defined in Section I. Images from $\mathcal { D } _ { t }$ are used only after the model has been fixed. No target image, label, automatically generated label, feature summary, normalization value, validation result, or model selection signal is used to update or select the model.

Let $\mathbf { I } \in \mathbb { R } ^ { 3 \times H \times W }$ be an input image, where H and W are its height and width. Let $\Omega _ { \mathbf { I } }$ denote the set of all pixel locations in I:

$$
\Omega _ { \bf I } = \{ 0 , \ldots , W - 1 \} \times \{ 0 , \ldots , H - 1 \} .\tag{2}
$$

Following the density map formulation described in Section II-A, let $\widehat { \bf D } _ { \bf I }$ be the predicted density map for I. Its predicted count is

$$
\widehat { N } _ { \mathbf { I } } = \sum _ { \mathbf { x } \in \Omega _ { \mathbf { I } } } \widehat { \mathbf { D } } _ { \mathbf { I } } ( \mathbf { x } ) .\tag{3}
$$

For the single image considered below, we write $\widehat { N }$ for ${ \widehat { N } } _ { \mathbf { I } } .$ The value at one location is not a class probability. It is only that location’s contribution to the count. COBICount follows this summation rule, but it forms the density map through three consecutive decisions: generate possible responses, retain responses that agree with source point patterns, and reduce responses related to repeated structures.

## B. Overall Data Flow

Fig. 1 gives the complete forward path. A residual backbone and a feature pyramid network (FPN) first convert I into a shared feature tensor F. An FPN combines features from shallow and deep stages so that the final tensor contains both spatial detail and a larger image context. A fixed image rule also produces an effective valid mask ${ \bf { M } } ^ { e f f }$ , which removes nearly black padding and its boundary.

Candidate Evidence (CE) uses F to form a broad nonnegative response map $\mathbf { C } ^ { a l l }$ . Candidate Acceptance (CA) produces a gate $\textbf { A } \in \mathbf { \partial } [ 0 , 1 ]$ that favors compact responses near the center patterns learned from source points. Bias Isolation (BI) produces a retention gate $\textbf { V } \in \ [ 0 , 1 ]$ . A small value of V reduces a response that agrees with repeated lines, broad responses away from a center, grids, or the learned source background response. The final density map on the network output grid is

![](images/fe039e98ad4e92444fe6b3184cc4b09ccf5669e2aa632273f307f481f0d0106b.jpg)  
Fig. 1. Overall data flow of COBICount. The input image produces the shared feature tensor F and the effective valid mask ${ \bf \cal M } ^ { e f f }$ . CE, CA, and BI produce $\mathbf { C } ^ { \mathrm { { a } } l l } , \mathbf { A } ,$ and V. These four maps enter one elementwise multiplication node to form Sb. Its spatial sum gives $\widehat { N } .$ , while $\mathcal { U } _ { m p }$ produces the density map at image resolution used for display. The Audit head is an auxiliary output and does not return to the counting path.

$$
\widehat { \mathbf { S } } ( \mathbf { u } ) = \mathbf { C } ^ { a l l } ( \mathbf { u } ) \mathbf { A } ( \mathbf { u } ) \mathbf { V } ( \mathbf { u } ) \mathbf { M } ^ { e f f } ( \mathbf { u } ) .\tag{4}
$$

Here, u is one location on the network output grid. Equation (4) is the main forward relation: a location contributes strongly only when CE supplies a response, CA accepts it, BI retains it, and the location is valid.

The network stride is $\rho = 4$ . For the image sizes used in this study, the grid of $\widehat { \mathbf { S } }$ has height $h = H / 4$ and width $w = W / 4$

$$
\Omega _ { \mathbf { S } } = \{ 0 , \ldots , w - 1 \} \times \{ 0 , \ldots , h - 1 \} .\tag{5}
$$

Thus, one step on this grid corresponds to four input pixels. The numerical count is obtained before any display resizing:

$$
\widehat { N } = \sum _ { \mathbf { u } \in \Omega _ { \mathbf { S } } } \widehat { \mathbf { S } } ( \mathbf { u } ) .\tag{6}
$$

For visualization at image resolution, bilinear resizing is followed by an area correction:

$$
\widehat { \bf D } _ { \bf I } = \mathcal { U } _ { m p } ( \widehat { \bf S } ) = \frac { h w } { H W } \mathcal { U } _ { b i l } ( \widehat { \bf S } ; H , W ) ,\tag{7}
$$

where ${ { \lambda } _ { b i l } }$ denotes bilinear resizing. The factor $h w / ( H W )$ compensates for the change in the number of spatial values. The count in Eq. (6) is always computed from Sb, so it is not affected by interpolation. For numerical safety, the implementation limits each final value and the summed count to $1 0 ^ { 5 }$ ; this limit is not reached in the reported experiments.

The Audit head is separate from the counting path. It receives F and selected internal maps, then produces four response maps for later inspection. It has no arrow to $\widehat { \mathbf { S } }$ or $\hat { N }$

## C. Shared Feature and Effective Valid Mask

The residual backbone + FPN begins with a convolutional stem and then uses four residual stages. A residual block adds its transformed feature to a direct or projected copy of its input. This addition helps information pass through the network. The four stages produce features at strides 4, 8, 16, and 32. Their channel numbers are 64, 128, 192, and 256, respectively. The FPN applies $\textbf { a } 1 \times 1$ lateral convolution at each stage and passes deeper information toward finer resolutions. It outputs four tensors $\{ \mathbf { P } _ { 2 } , \mathbf { P } _ { 3 } , \mathbf { P } _ { 4 } , \mathbf { P } _ { 5 } \}$ , each with 96 channels.

The last three tensors are resized to the resolution of $\mathbf { P } _ { 2 }$ and concatenated with it:

$$
{ { \bf { F } } ^ { c a t } } = [ { { \bf { P } } _ { 2 } } , \mathcal { U } ( { { \bf { P } } _ { 3 } } ) , \mathcal { U } ( { { \bf { P } } _ { 4 } } ) , \mathcal { U } ( { { \bf { P } } _ { 5 } } ) ] ,\tag{8}
$$

where U denotes bilinear resizing to the size of $\mathbf { P } _ { 2 }$ . Two $3 \times 3$ convolution blocks fuse the concatenated tensor:

$$
\mathbf { F } = \phi _ { f u s e } ( \mathbf { F } ^ { c a t } ) \in \mathbb { R } ^ { 9 6 \times h \times w } .\tag{9}
$$

Each block contains a convolution, batch normalization, and a sigmoid linear unit (SiLU). Batch normalization rescales intermediate channels during training, and SiLU is a smooth activation function. Unless stated otherwise, a prediction head $h _ { z }$ used below has a 1×1 convolution from 96 to 48 channels, a SiLU activation, and a second 1×1 convolution that produces the required output channels.

The valid mask is obtained directly from the image and has no learned parameter. Before entering the network, each image channel is normalized using the mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) commonly used for ImageNet. To construct the valid mask, this numerical transformation is reversed to recover the RGB values. The recovered values are limited to [0, 1], and the three channels are averaged to obtain the intensity map J. The initial valid mask is

$$
\mathbf { M } ^ { v a l i d } = \mathbf { 1 } [ \mathbf { J } > 0 . 0 3 5 ] .\tag{10}
$$

![](images/43d9c7d5b76519f24ecbb4cdfec046e3aade75315c177f200e148eb29c8112f5.jpg)  
Fig. 2. Candidate Evidence. The shared feature tensor first produces $\mathbf { E } _ { p } .$ Its Sobel response G enters the computation of $\mathbf { E } _ { b } ,$ and $\mathbf { E } _ { p }$ enters the computation of $\mathbf { E } _ { h }$ . The scale head supplies $\pi _ { s } , \pi _ { m } .$ , and π<sub>l</sub>, which multiply the matching density head outputs. The weighted sum in Eq. (22) combines $\mathbf { D } ^ { m s }$ $\mathbf { E } _ { p }$ $\bar { \bf E } _ { b }$ , and $\mathbf { E } _ { h }$ into $\dot { \mathbf { C } } ^ { a l l }$

where 1[·] equals one when its condition is true and zero otherwise. The complement $1 - \mathbf { M } ^ { v a l i d }$ marks nearly black pixels. $\mathrm { ~ A ~ 9 ~ } \times \mathrm { ~ 9 ~ }$ maximum pooling operation expands this area to obtain the neutral mask

$$
\mathbf { M } ^ { n e u } = \mathbf { M } \mathrm { a x } \mathbf { P } \mathrm { o o l } _ { 9 \times 9 } ( 1 - \mathbf { M } ^ { v a l i d } ) .\tag{11}
$$

Maximum pooling keeps the largest value in each neighborhood, so this operation adds a narrow buffer around the black area. Both masks are resized to $h \times w$ with nearest neighbor interpolation. After resizing, the same symbols denote the masks on the network output grid. The effective valid mask is

$$
\mathbf { M } ^ { e f f } = \mathbf { M } ^ { v a l i d } \odot ( 1 - \mathbf { M } ^ { n e u } ) ,\tag{12}
$$

where ⊙ denotes elementwise multiplication. This mask removes padding; it does not decide whether a valid image location is an object or background. The same fixed rule is used for source images and target images.

## D. Candidate Evidence

CE performs the first decision in the counting path: it creates possible responses before CA and BI remove unsupported ones. The module uses three bounded response maps and three routed density maps. The routed density $\mathbf { D } ^ { m s }$ is the contribution from the three nonnegative density branches. By contrast, $\mathrm { \bf E } _ { p } , \mathrm { \bf E } _ { b } ,$ , and $\mathbf { E } _ { h }$ describe the base response, the response after strong edges are reduced, and the response supported by a wider neighborhood. These four terms are combined into one CE map; none is treated as a separate final prediction. Fig. 2 summarizes these computations, while Eqs. (13)–(22) give their exact inputs and order.

The base response is

$$
\mathbf { E } _ { p } = \sigma ( h _ { p } ( \mathbf { F } ) ) \odot \mathbf { M } ^ { e f f } ,\tag{13}
$$

where σ is the sigmoid function, which limits each value to [0, 1]. The change around this response is measured by a Sobel filter. The two fixed kernels are

$$
S _ { x } = \frac { 1 } { 8 } \left[ \begin{array} { l l l } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} \right] , \qquad S _ { y } = S _ { x } ^ { \mathsf { T } } .\tag{14}
$$

For any one channel map $\mathbf { X } ,$ the following operation rescales its values to [0, 1] within each image:

$$
\begin{array} { c l } { { \mathrm { N o r m } _ { 0 1 } ( { \bf X } ) = \frac { { \bf X } - \operatorname* { m i n } _ { \bf u } { \bf X ( u ) } } { \operatorname* { m a x } _ { \bf u } { \bf X ( u ) } - \operatorname* { m i n } _ { \bf u } { \bf X ( u ) } + \epsilon _ { n o r m } } , } } \\ { { \epsilon _ { n o r m } = 1 0 ^ { - 6 } . } } \end{array}\tag{15}
$$

The normalized Sobel magnitude is

$$
\mathbf { G } = \operatorname { N o r m } _ { 0 1 } \left( \sqrt { ( S _ { x } * \mathbf { E } _ { p } ) ^ { 2 } + ( S _ { y } * \mathbf { E } _ { p } ) ^ { 2 } + 1 0 ^ { - 6 } } \right)\tag{16}
$$

where ∗ denotes convolution. A large G indicates a strong local change in $\mathbf { E } _ { p }$

The second response reduces values at these strong changes:

$$
\mathbf { E } _ { b } = \sigma ( h _ { b } ( \mathbf { F } ) ) \odot \mathrm { c l i p } ( 1 - 0 . 3 5 \mathbf { G } , 0 , 1 ) \odot \mathbf { M } ^ { e f f } .\tag{17}
$$

The function $\mathrm { c l i p } ( \mathbf { X } , a , b )$ limits every value of X to $[ a , b ]$ The third response enlarges nearby support:

$$
\begin{array} { r } { \mathbf { E } _ { h } = \mathrm { M a x P o o l } _ { 5 \times 5 } \left[ \sigma ( h _ { h } ( \mathbf { F } ) ) \right. } \\ { \left. \odot \left( 0 . 5 + 0 . 5 \mathbf { E } _ { p } \right) \right] \odot \mathbf { M } ^ { e f f } . } \end{array}\tag{18}
$$

The factor $\left( 0 . 5 + 0 . 5 \mathbf { E } _ { p } \right)$ ties this response to the base response.   
Maximum pooling then extends it over a $5 \times 5$ neighborhood.

The remaining CE path allows three density heads to contribute by different amounts at each location. The scale head produces

$$
\begin{array} { r l } & { \left[ \pi _ { s } , \pi _ { m } , \pi _ { l } \right] = \operatorname { S o f t m a x } _ { c h } ( h _ { s c a l e } ( \mathbf { F } ) ) , } \\ & { \pi _ { s } + \pi _ { m } + \pi _ { l } = 1 . } \end{array}\tag{19}
$$

Softmax is applied across the three channels, so the three values form local routing weights. Three other heads produce nonnegative maps:

$$
\mathbf { D } _ { k } = { \mathrm { S o f t p l u s } } ( h _ { k } ( \mathbf { F } ) ) \odot \mathbf { M } ^ { e f f } , \qquad k \in \{ s , m , l \} .\tag{20}
$$

Softplus converts an unrestricted head output into a nonnegative value. The labels $s , m ,$ and l are only branch identifiers. No object size label is supplied, and the method does not assume that these branches have a fixed physical scale. Their routed sum is

$$
\mathbf { D } ^ { m s } = \pi _ { s } \odot \mathbf { D } _ { s } + \pi _ { m } \odot \mathbf { D } _ { m } + \pi _ { l } \odot \mathbf { D } _ { l } .\tag{21}
$$

The superscript ms denotes this mixture of three routed branches; it is not a separate prediction head.

The CE output combines the routed density and the three bounded responses:

$$
\mathbf { C } ^ { a l l } = 0 . 3 4 \mathbf { D } ^ { m s } + 0 . 2 4 \mathbf { E } _ { p } + 0 . 2 2 \mathbf { E } _ { b } + 0 . 2 0 \mathbf { E } _ { h } .\tag{22}
$$

The coefficients are fixed implementation values. They sum to one, but $\mathbf { C } ^ { a l l }$ is not a probability map and its spatial sum is not constrained to equal one.

![](images/3ea4d6dd6eaea588f2fbe8108813487e0fad610e70d9a9bfb317a6e05adc331d.jpg)  
Fig. 3. Candidate Acceptance. Four heads produce $\mathbf { A } _ { c } , \mathbf { A } _ { o } , \mathbf { A } _ { q } ,$ and $\mathbf { A } _ { r } .$ The last two maps form 0.35 ${ \bf + 0 . 6 5 A } _ { q }$ and 0.35 + 0.65A $. r \cdot ,$ respectively, before the four factors enter one multiplication node. The output is the acceptance gate A.

## E. Candidate Acceptance

CE is deliberately broad, so one object response may spread over several locations or contain several nearby peaks. CA performs the second decision: it controls how much of each CE response can continue to the final density map. It predicts four maps from F:

$$
\begin{array} { r } { \mathbf { A } _ { c } = \sigma ( h _ { c } ( \mathbf { F } ) ) , \mathbf { A } _ { o } = \sigma ( h _ { o } ( \mathbf { F } ) ) , } \\ { \mathbf { A } _ { q } = \sigma ( h _ { q } ( \mathbf { F } ) ) , \mathbf { A } _ { r } = \sigma ( h _ { r } ( \mathbf { F } ) ) . } \end{array}\tag{23}
$$

A<sub>c</sub> is trained to be high near a source point center. $\mathbf { A } _ { o }$ is trained with a wider area around each source point. The two remaining maps, $\mathbf { A } _ { q }$ and $\mathbf { A } _ { r } ,$ use the same wider target with different loss weights. The implementation names them quality and reliability, but they do not represent two separately annotated properties.

The four maps form

$$
\begin{array} { c } { \mathbf { A } = \mathbf { A } _ { c } \odot \mathbf { A } _ { o } \odot ( 0 . 3 5 + 0 . 6 5 \mathbf { A } _ { q } ) } \\ { \odot ( 0 . 3 5 + 0 . 6 5 \mathbf { A } _ { r } ) . } \end{array}\tag{24}
$$

The first two maps can strongly reduce a response. Each of the last two factors remains between 0.35 and 1, so neither can remove a response by itself. Here, a compact response does not mean that CA measures its shape. Instead, the narrow center target and the wider support target encourage an accepted response to remain near a source point and to decrease away from it. CA does not find connected components or apply a hand written shape rule. It learns these four spatial gates from source point supervision.

## F. Bias Isolation

BI performs the third decision. Here, bias means a repeated source image pattern that can trigger a counting response even when no annotated object supports it. BI does not recognize named background categories in a target domain. Instead, it estimates four continuous risk maps and converts their weighted sum into the retention gate V.

Four learned maps are obtained from F:

$$
\begin{array} { r } { \mathbf { B } _ { b g } = \sigma ( h _ { b g } ( \mathbf { F } ) ) , \qquad \mathbf { B } _ { l i n e } = \sigma ( h _ { l i n e } ( \mathbf { F } ) ) , } \\ { \mathbf { B } _ { b r o a d } = \sigma ( h _ { b r o a d } ( \mathbf { F } ) ) , \mathbf { B } _ { g r i d } = \sigma ( h _ { g r i d } ( \mathbf { F } ) ) . } \end{array}\tag{25}
$$

$\mathbf { B } _ { b g }$ is the only map in this group with a direct background target. The other three maps receive no labels for roads, lines, parking grids, or other named structures, so they are not classifiers of named structures. Their names refer to the response structures used in Eqs. (28)–(30). They receive gradients through the retention gate V and the final density map Sb. The source density and count losses keep counting mass near annotated source objects, while the suppression losses penalize remaining mass outside the protected source regions. The three maps are therefore learned from source counting supervision rather than from named structure labels.

BI also computes fixed transforms from the base response $\mathbf { E } _ { p }$ . The Sobel map G was defined in Eq. (16). The four neighbor Laplacian kernel is

$$
K _ { l a p } = \left[ \begin{array} { c c c } { { 0 } } & { { 1 } } & { { 0 } } \\ { { 1 } } & { { - 4 } } & { { 1 } } \\ { { 0 } } & { { 1 } } & { { 0 } } \end{array} \right] .\tag{26}
$$

The Laplacian measures local second order change. Average pooling measures the mean response in a larger neighborhood. Their normalized maps are

$$
\begin{array} { r l } & { \mathbf { L } = \mathrm { N o r m } _ { 0 1 } ( | K _ { l a p } * \mathbf { E } _ { p } | ) , } \\ & { \mathbf { P } = \mathrm { N o r m } _ { 0 1 } ( \mathrm { A v g P o o l } _ { 2 5 \times 2 5 } ( \mathbf { E } _ { p } ) ) . } \end{array}\tag{27}
$$

L becomes large where the response changes repeatedly over a short distance. P becomes large where a response remains broad over a larger area.

The learned and fixed maps are combined in a fixed order:

$$
\mathbf { R } _ { l i n e } = \mathrm { c l i p } ( 0 . 5 5 \mathbf { B } _ { l i n e } + 0 . 4 5 \mathbf { G } , 0 , 1 ) ,\tag{28}
$$

$$
\mathbf { R } _ { b r o a d } = \mathrm { c l i p } \left( 0 . 6 0 \mathbf { B } _ { b r o a d } + 0 . 4 0 \mathbf { P } \odot ( 1 - \mathbf { A } _ { c } ) , 0 , 1 \right)\tag{29}
$$

$$
\mathbf { R } _ { g r i d } = \mathrm { c l i p } \left( 0 . 6 5 \mathbf { B } _ { g r i d } + 0 . 3 5 \mathbf { L } \odot ( 1 - \mathbf { A } _ { c } ) , 0 , 1 \right) .\tag{30}
$$

${ \bf R } _ { l i n e }$ responds to a learned line map and a strong first order change. $\mathbf { R } _ { b r o a d }$ becomes large when the pooled response is strong but the center gate is weak. It does not identify enclosed empty regions as a separate class. ${ \mathbf { R } } _ { g r i d }$ combines repeated second order changes with a weak center gate. It is not trained with parking grid labels.

The background risk also uses the neutral mask:

$$
\begin{array} { r l } & { \mathbf { R } _ { b g } = \mathrm { c l i p } \Big ( \mathbf { B } _ { b g } \odot ( 0 . 5 0 + 0 . 5 0 \mathbf { R } _ { b r o a d } ) } \\ & { \qquad + 0 . 2 5 \mathbf { M } ^ { n e u } , 0 , 1 \Big ) . } \end{array}\tag{31}
$$

The four risk maps may overlap; they are not four exclusive classes. Their weighted sum is

$$
\begin{array} { r } { \mathbf { R } ^ { b i } = 0 . 7 2 \mathbf { R } _ { b g } + 0 . 5 8 \mathbf { R } _ { l i n e } + 0 . 4 8 \mathbf { R } _ { b r o a d } } \\ { + 0 . 6 0 \mathbf { R } _ { g r i d } + 1 . 0 0 \mathbf { M } ^ { n e u } . \qquad } \end{array}\tag{32}
$$

Finally,

$$
\mathbf { V } = \mathrm { c l i p } ( 1 - \mathbf { R } ^ { b i } , 0 , 1 ) .\tag{33}
$$

A large risk therefore produces a small retention value at the same location. The neutral mask enters both $\mathbf { R } _ { b g }$ and $\mathbf { R } ^ { b i }$ which strongly reduces responses near black padding.

## G. Audit Response Maps

Audit is an auxiliary head for inspecting response patterns. It does not change ${ \bf C } ^ { \bar { a } \bar { l } l }$ , A, V, Sb, or Nb. A four channel head first produces the raw maps

$$
[ { \bf Z } _ { v e h } ^ { 0 } , { \bf Z } _ { b u i l d } ^ { 0 } , { \bf Z } _ { s h i p } ^ { 0 } , { \bf Z } _ { b g } ^ { 0 } ] = h _ { a u d } ( { \bf F } ) .\tag{34}
$$

![](images/7c3b9859c1899d9473c44b839d074dea793c71c33b1d7fba55e492444f6cf2e5.jpg)  
Fig. 4. Bias Isolation. The exact inputs are F, $\mathbf { E } _ { p } , \mathbf { A } _ { c } ,$ and ${ \mathbf { M } } ^ { n e u }$ . Fixed filters applied to $\mathbf { E } _ { p _ { . } }$ produce $\mathbf { G } , \mathbf { L } ,$ and $\mathbf { P } .$ These maps combine with four learned maps to form ${ \bf R } _ { l i n e } , { \bf R } _ { b r o a d } , { \bf R } _ { g r i d } ,$ and $\mathbf { R } _ { b g } .$ The weighted sum in Eq. (32) produces R<sup>bi</sup>, followed by $\mathrm { { z l i p } } ( 1 - \mathbf { R } ^ { b i } , 0 , 1 )$ to obtain $\mathbf { v } .$

The channel names vehicle, building, ship, and background describe response patterns for inspection; they are not predicted object classes. Fixed offsets then connect each raw map to selected internal responses:

$$
\begin{array} { r } { \mathbf { Z } _ { v e h } = \mathbf { Z } _ { v e h } ^ { 0 } + 0 . 7 5 \pi _ { s } + 0 . 2 0 \pi _ { m } + 0 . 2 0 \mathbf { A } _ { c } , } \end{array}\tag{35}
$$

$$
{ \bf Z } _ { b u i l d } = { \bf Z } _ { b u i l d } ^ { 0 } + 0 . 8 0 \pi _ { l } + 0 . 3 5 { \bf R } _ { b r o a d } ,\tag{36}
$$

$$
{ \bf Z } _ { s h i p } = { \bf Z } _ { s h i p } ^ { 0 } + 0 . 5 0 { \bf R } _ { l i n e } + 0 . 2 0 \pi _ { m } ,\tag{37}
$$

$$
\mathbf { Z } _ { b g } = \mathbf { Z } _ { b g } ^ { 0 } + 1 . 1 0 \mathbf { R } _ { b g } + 0 . 3 5 \mathbf { R } _ { g r i d } .\tag{38}
$$

These fixed links are inspection rules. They do not mean that $\pi _ { s }$ denotes vehicles, $\pi _ { l }$ denotes buildings, or ${ \bf R } _ { l i n e }$ denotes ships. More generally, a routing branch or a structural risk is not assigned to an unseen target class.

The four Audit response maps are

$$
\begin{array} { r l } & { \mathbf { P } ^ { a u d } = [ \mathbf { P } _ { v e h } ^ { a u d } , \mathbf { P } _ { b u i l d } ^ { a u d } , \mathbf { P } _ { s h i p } ^ { a u d } , \mathbf { P } _ { b g } ^ { a u d } ] } \\ & { \qquad = \mathrm { S o f t m a x } _ { c h } ( [ \mathbf { Z } _ { v e h } , \mathbf { Z } _ { b u i l d } , \mathbf { Z } _ { s h i p } , \mathbf { Z } _ { b g } ] ) . } \end{array}\tag{39}
$$

These values sum to one across channels at each location. Their numerical values should not be read as reliable probabilities of unseen target classes. In the main protocol, only the building and background channels receive direct labels from the RSOC Building source. The other channels obtain no vehicle or ship label during training. Audit should therefore be read only as a visual description of internal response patterns.

## H. Training with Source Points

All training maps and losses are computed from a source image and its point set. They are introduced below in the same order in which they are used: first the training maps constructed from source points, then the density and count losses, then the losses for CA and BI, and finally the routing and Audit terms.

1) Maps constructed from source points: For the following construction on one source image, we omit the image index used in Section III-A and write its annotated point set as $\mathcal { P } ^ { s } =$ $\{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { N ^ { * } }$ , where $N ^ { * } = | \mathcal { P } ^ { s } |$ is the reference count for this image. Because the network output grid has stride $\rho = 4 ,$ , each point coordinate is divided by $\rho$ and rounded to the nearest grid location:

$$
{ \widetilde { \bf p } } _ { i } = \mathrm { r o u n d } \left( { \frac { { \bf p } _ { i } } { \rho } } \right) , \qquad \rho = 4 .\tag{40}
$$

Both coordinates are transformed in this operation.

Each source annotation provides only an object location. To construct the density target used for training, a normalized Gaussian kernel is placed around each projected point. For a Gaussian width $\sigma ,$ its finite window is

$$
\begin{array} { r l } & { \mathcal { W } _ { \sigma } = \left\{ \delta = ( \delta _ { x } , \delta _ { y } ) : | \delta _ { x } | , | \delta _ { y } | \leq r _ { \sigma } \right\} , } \\ & { ~ r _ { \sigma } = \operatorname* { m a x } ( 1 , \mathrm { r o u n d } ( 3 \sigma ) ) . } \end{array}\tag{41}
$$

The discrete kernel inside this window has unit sum:

$$
\kappa _ { \sigma } ( \delta ) = \frac { \exp ( - \| \delta \| _ { 2 } ^ { 2 } / ( 2 \sigma ^ { 2 } ) ) } { \sum _ { \delta ^ { \prime } \in \mathcal { W } _ { \sigma } } \exp ( - \| \delta ^ { \prime } \| _ { 2 } ^ { 2 } / ( 2 \sigma ^ { 2 } ) ) } .\tag{42}
$$

One kernel is added at each projected point to form the source density target:

$$
{ \bf D } ^ { * } ( { \bf u } ) = \sum _ { i = 1 } ^ { N ^ { * } } \kappa _ { \sigma _ { p } } ( { \bf u } - \widetilde { { \bf p } } _ { i } ) , \qquad \sigma _ { p } = 1 . 1 5 .\tag{43}
$$

Values outside $\Omega _ { \mathbf { S } }$ are discarded. The kernel is normalized before it is placed, but a kernel cut by an image boundary is not normalized again. Thus, $\mathbf { D } ^ { * }$ is a density target constructed from source annotations rather than another model output. During training, it is compared with the predicted density map Sb in Eq. (49). It is not used when the fixed model processes a target image.

CA needs a center target and a wider support target. These maps use Gaussians with a maximum value of one rather than a unit sum:

$$
\mathbf { T } _ { \sigma } ( \mathbf { u } ) = \operatorname* { m a x } _ { 1 \leq i \leq N ^ { * } } \Bigg [ \exp \left( - \frac { \| \mathbf { u } - \widetilde { \mathbf { p } } _ { i } \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right)\tag{44}
$$

When no point is present, $\mathbf { D } ^ { * }$ and $\mathbf { T } _ { \sigma }$ are zero maps. The center target and support target are

$$
{ \bf T } ^ { p e a k } = { \bf T } _ { \sigma _ { p } } , \qquad { \bf T } ^ { s u p } = { \bf T } _ { \sigma _ { s } } , \qquad \sigma _ { s } = 2 . 3 5 .\tag{45}
$$

T<sup>peak</sup> is narrow and $\mathbf { T } ^ { s u p }$ covers a wider area. Their binary masks are

$$
\begin{array} { r } { \begin{array} { c } { \mathbf { M } ^ { p t } = \mathbf { 1 } [ \mathbf { T } ^ { p e a k } > 0 . 0 5 ] , } \\ { \mathbf { M } ^ { s u p } = \mathbf { 1 } [ \mathbf { T } ^ { s u p } > 0 . 0 5 ] , } \\ { \mathbf { M } ^ { f g } = \mathrm { D i l a t e } _ { 9 \times 9 } ( \mathbf { M } ^ { p t } ) , } \\ { \mathbf { M } ^ { b g } = ( 1 - \mathbf { M } ^ { f g } ) \odot \mathbf { M } ^ { e f f } . } \end{array} } \end{array}\tag{46}
$$

$\mathrm { D i l a t e _ { 9 \times 9 } }$ is binary maximum pooling. ${ \bf M } ^ { s u p }$ marks the area used by the support losses. ${ \bf M } ^ { f g }$ is a separate protection area used by the background and strong response losses. $\mathbf { M } ^ { b g }$ contains valid locations outside $\mathbf { M } ^ { f \bar { g } }$ and supplies the direct target for $\mathbf { B } _ { b g }$

2) Density and count losses: For compact notation, ⟨X denotes the mean over all spatial locations and all images in the current batch. The two elementary penalties are

$$
\ell _ { S L 1 } ( z ) = { \left\{ \begin{array} { l l } { { \frac { 1 } { 2 } } z ^ { 2 } , } & { | z | < 1 , } \\ { | z | - { \frac { 1 } { 2 } } , } & { | z | \geq 1 , } \end{array} \right. }\tag{47}
$$

and

$$
\ell _ { B C E } ( p , y ) = - y \log p - ( 1 - y ) \log ( 1 - p ) .\tag{48}
$$

The first is the smooth $L _ { 1 }$ penalty. The second is binary cross entropy for a bounded prediction p and target y. Before binary cross entropy is evaluated, the implementation limits $p$ away from zero and one by $1 0 ^ { - 4 }$ for numerical stability.

Let $N _ { n o r m } = \operatorname* { m a x } ( N ^ { * } , 1 )$ . The density loss is

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { d e n } = \Bigg \langle \ell _ { S L 1 } \left( \frac { \widehat { { \bf S } } } { N _ { n o r m } } - \frac { { \bf D } ^ { * } } { N _ { n o r m } } \right) } \\ & { } & { \qquad \odot \left( 1 + 4 { \bf M } ^ { s u p } \right) \odot { \bf M } ^ { e f f } \Bigg \rangle . } \end{array}\tag{49}
$$

Division by $N _ { n o r m }$ prevents images with many objects from dominating this map loss. The factor $\left( 1 + 4 \mathbf { M } ^ { s u p } \right)$ gives more weight to locations near source objects.

Two losses compare the summed prediction with the reference count. They are evaluated for each image and then averaged over the batch:

$$
\mathcal { L } _ { c n t } ^ { r e l } = \frac { \lvert \widehat { N } - N ^ { * } \rvert } { N ^ { * } + 1 } ,\tag{50}
$$

$$
\mathcal { L } _ { c n t } ^ { l o g } = | \log ( 1 + \widehat { N } ) - \log ( 1 + N ^ { * } ) | .\tag{51}
$$

The relative term limits the effect of large reference counts. The logarithmic term reduces the difference between very large numerical ranges.

3) Lossesfor CA and BI: The CA maps are trained with the center and support targets defined above. Their spatial weights are

$$
\begin{array} { r } { \omega _ { c } = \left( 1 + 7 \mathbf { T } ^ { p e a k } \right) \odot \mathbf { M } ^ { e f f } , } \\ { \omega _ { o } = \left( 1 + 4 \mathbf { M } ^ { s u p } \right) \odot \mathbf { M } ^ { e f f } , } \\ { \omega _ { s } = \left( 1 + 2 \mathbf { M } ^ { s u p } \right) \odot \mathbf { M } ^ { e f f } . } \end{array}\tag{52}
$$

The corresponding losses are

$$
\mathcal { L } _ { c t r } = \left. \omega _ { c } \odot \ell _ { B C E } ( \mathbf { A } _ { c } , \mathbf { T } ^ { p e a k } ) \right. ,\tag{53}
$$

$$
\mathcal { L } _ { o b j } = \left. \pmb { \omega } _ { o } \odot \ell _ { B C E } ( \mathbf { A } _ { o } , \mathbf { T } ^ { s u p } ) \right. ,\tag{54}
$$

$$
\mathcal { L } _ { b g } = \left. \mathbf { M } ^ { e f f } \odot \ell _ { B C E } ( \mathbf { B } _ { b g } , \mathbf { M } ^ { b g } ) \right. ,\tag{55}
$$

$$
\begin{array} { r } { \mathcal { L } _ { o r g } = \mathcal { L } _ { o b j } + 0 . 5 5 \mathcal { L } _ { b g } , } \end{array}\tag{56}
$$

$$
\mathcal { L } _ { r l y } = \left. \omega _ { s } \odot \ell _ { B C E } ( \mathbf { A } _ { r } , \mathbf { T } ^ { s u p } ) \right. ,\tag{57}
$$

$$
\mathcal { L } _ { q u a } = \langle \omega _ { s } \odot \ell _ { B C E } ( \mathbf { A } _ { q } , \mathbf { T } ^ { s u p } ) \rangle .\tag{58}
$$

These losses state where CA should remain open and where the learned background response should be high. They do not provide labels for any named target background structure.

BI is also trained through penalties on the final counting mass. Let $\operatorname { s g } ( \cdot )$ denote stop gradient. It keeps a map’s value in the forward calculation but blocks the direct gradient through that map when it acts as a weight. The four penalties are

$$
\mathcal { L } _ { b g v } = \left. \widehat { \mathbf { S } } \odot \mathrm { s g } ( \mathbf { R } _ { b g } ) \odot \left( 1 - \mathbf { M } ^ { f g } \right) \odot \mathbf { M } ^ { e f f } \right. ,\tag{59}
$$

$$
\mathcal { L } _ { l i n e } = \left. \widehat { \mathbf { S } } \odot \mathrm { s g } ( \mathbf { R } _ { l i n e } ) \odot ( 1 - \mathbf { M } ^ { s u p } ) \odot \mathbf { M } ^ { e f f } \right. ,\tag{60}
$$

$$
\mathcal { L } _ { b r o a d } = \left. \widehat { \mathbf { S } } \odot \mathrm { s g } ( \mathbf { R } _ { b r o a d } ) \odot \left( 1 - \mathbf { M } ^ { s u p } \right) \odot \mathbf { M } ^ { e f f } \right.\tag{61}
$$

$$
\mathcal { L } _ { g r i d } = \left. \widehat { \mathbf { S } } \odot \mathrm { s g } ( \mathbf { R } _ { g r i d } ) \odot ( 1 - \mathbf { M } ^ { s u p } ) \odot \mathbf { M } ^ { e f f } \right. .\tag{62}
$$

The background penalty uses the wider protection mask ${ \bf M } ^ { f g }$ The other three use ${ \bf M } ^ { s u p }$ . Although the risk maps are detached in these four weighting positions, their heads still receive gradients through $\mathbf { V }$ and the final density map $\widehat { \mathbf { S } } .$

The strongest remaining responses outside ${ \bf M } ^ { f g }$ receive an additional penalty. These locations are called hard negatives because source point supervision treats them as background, but the current model still gives them a large response. For each image, define

$$
\mathbf { Q } = \mathrm { s g } ( \widehat { \mathbf { S } } ) \odot \left( 1 - \mathbf { M } ^ { f g } \right) \odot \mathbf { M } ^ { e f f } .\tag{63}
$$

Let $\mathcal { H } _ { K _ { h n } }$ contain the $K _ { h n }$ largest locations of Q, where

$$
K _ { h n } = \operatorname* { m i n } ( 7 6 8 , | \Omega _ { \mathbf { S } } | ) .\tag{64}
$$

The loss on these locations is

$$
\mathcal { L } _ { h n } = \frac { 1 } { \vert \mathcal { H } _ { K _ { h n } } \vert } \sum _ { \mathbf { u } \in \mathcal { H } _ { K _ { h n } } } \widehat { \mathbf { S } } ( \mathbf { u } ) ( 1 - \mathbf { M } ^ { f g } ( \mathbf { u } ) ) \mathbf { M } ^ { e f f } ( \mathbf { u } ) .\tag{65}
$$

$\mathrm { s g } ( \widehat { \mathbf { S } } )$ is used only to select the locations. The selected values of Sb in Eq. (65) remain differentiable.

4) Routing, Audit, and total loss: Two terms prevent the routing weights from assigning nearly all local weight to a single branch. Their local entropy is

$$
\mathcal { H } _ { \pi } ( \mathbf { u } ) = - \sum _ { k \in \{ s , m , l \} } \pi _ { k } ( \mathbf { u } ) \log \pi _ { k } ( \mathbf { u } ) .\tag{66}
$$

The entropy loss is

$$
\mathcal { L } _ { e n t } = \left. \left( 1 . 1 0 - \mathcal { H } _ { \pi } \right) \odot \left( 1 - \mathbf { M } ^ { s u p } \right) \odot \mathbf { M } ^ { e f f } \right. .\tag{67}
$$

Entropy is large when the three weights are similar and small when one weight dominates. The value 1.10 approximates log 3, the largest entropy of three routing weights. Minimizing

$\mathcal { L } _ { e n t }$ avoids an early, highly concentrated choice outside the support area.

Let B be the batch size. The mean use of branch k is

$$
\pi _ { k } = \frac { 1 } { B | \Omega _ { \mathbf { S } } | } \sum _ { b = 1 } ^ { B } \sum _ { \mathbf { u } \in \Omega _ { \mathbf { S } } } \pi _ { b , k } ( \mathbf { u } ) .\tag{68}
$$

The balance loss is

$$
\mathcal { L } _ { b a l } = \frac { 1 } { 3 } \sum _ { k \in \{ s , m , l \} } ( \overline { { { \pi } } } _ { k } - q _ { k } ) ^ { 2 } ,\tag{69}
$$

For the main protocol, direct Audit supervision begins at epoch 10. RSOC Building points supervise the building channel inside ${ \bf M } ^ { s u p }$ , and valid locations outside ${ \bf M } ^ { f g }$ supervise the background channel:

$$
\mathcal { L } _ { a u d } ^ { b u i l d } = - \left. \log \mathbf { P } _ { b u i l d } ^ { a u d } \odot \mathbf { M } ^ { s u p } \odot \mathbf { M } ^ { e f f } \right. ,\tag{70}
$$

$$
\mathcal { L } _ { a u d } ^ { b g } = - \left. \log \mathbf { P } _ { b g } ^ { a u d } \odot \left( 1 - \mathbf { M } ^ { f g } \right) \odot \mathbf { M } ^ { e f f } \right. ,\tag{71}
$$

$$
\mathcal { L } _ { a u d } = \mathbf { 1 } [ e \geq 1 0 ] ( \mathcal { L } _ { a u d } ^ { b u i l d } + 0 . 3 5 \mathcal { L } _ { a u d } ^ { b g } ) ,\tag{72}
$$

where e is the current epoch. The vehicle and ship channels receive no direct source class label.

The complete training objective is

$$
\begin{array} { r l } & { \mathcal { L } = 1 . 2 5 \mathcal { L } _ { d e n } + c _ { e } ( 0 . 7 2 \mathcal { L } _ { c n t } ^ { r e l } + 0 . 2 0 \mathcal { L } _ { c n t } ^ { l o g } ) } \\ & { \qquad + 0 . 5 5 \mathcal { L } _ { c t r } + 0 . 6 5 \mathcal { L } _ { o r g } + 0 . 4 0 \mathcal { L } _ { r l y } + 0 . 3 5 \mathcal { L } _ { q u a } } \\ & { \qquad + 0 . 3 8 \mathcal { L } _ { b g v } + 0 . 2 2 \mathcal { L } _ { l i n e } + 0 . 2 5 \mathcal { L } _ { b r o a d } + 0 . 3 0 \mathcal { L } _ { g r i d } } \\ & { \qquad + 0 . 1 8 \mathcal { L } _ { h n } + 0 . 0 1 8 \mathcal { L } _ { e n t } + 0 . 0 2 0 \mathcal { L } _ { b a l } + 0 . 1 6 \mathcal { L } _ { a u d } , } \\ & { \qquad c _ { e } = \operatorname* { m i n } \left( 1 , \frac { e } { 8 } \right) . } \end{array}\tag{73}
$$

The coefficient $c _ { e }$ increases the count loss during the first eight epochs. CE has no separate direct loss on $\mathbf { C } ^ { a l \bar { l } } .$ It is updated through the final density map, count, BI, strong response, and routing terms. Every quantity in Eq. (73) is computed from source images, source points, or fixed transforms of the current source image.

## I. Diagnostic Points at Inference

Point extraction is not part of the numerical count. The model first computes $\widehat { \mathbf { S } }$ and $\widehat { N }$ by Eqs. (4) and (6). Afterward, a fixed greedy procedure extracts local maxima from $\widehat { \mathbf { S } }$ only for spatial analysis. The extracted locations are called diagnostic points.

For each image, the selection threshold is

$$
\boldsymbol \tau = \operatorname* { m a x } \left( 0 . 0 2 0 , 0 . 2 8 \operatorname* { m a x } _ { \mathbf { u } } \widehat \mathbf { S } ( \mathbf { u } ) \right) .\tag{74}
$$

The procedure selects the largest remaining value above $\tau ,$ records its location, and removes its neighborhood before searching again. Accepted points must be at least 3.2 network output grid cells apart. With stride $^ { 4 , }$ this distance is about 12.8 input pixels. The point budget is

$$
K _ { p t } = \operatorname* { m i n } \Big ( 1 0 2 4 , \operatorname* { m a x } [ 1 , \mathrm { r o u n d } ( 1 . 5 5 \widehat { N } + 1 6 ) ] \Big ) .\tag{75}
$$

This budget only limits the search. It does not force the number of extracted points to equal $\widehat { N }$ . A network output grid point $( x , y )$ is returned to image coordinates by multiplying both coordinates by the stride before visualization or distance evaluation.

For Audit visualization, the four values in $\mathbf { P } ^ { a u d }$ are sampled at each extracted point. The point receives the label of the largest channel. This label changes neither the point location nor the numerical count. Target annotations are used only by the evaluator to measure the distance between these fixed predictions and reference points.

## IV. EXPERIMENTS

This section first specifies the data access rule, datasets, evaluation measures, comparison settings, and implementation. It then reports the main comparison and tests in which the training source is changed.

## A. Protocol and Data

The experiments follow source-only counting as defined in Section I. The roles of source and target are assigned separately for each experiment; they are not fixed properties of RSOC, DOTA, or DIOR. A source domain contains one object category under a related set of imaging conditions and provides the annotations used to train and select a model. Ordinary training, validation, and test splits of that same category still belong to the same domain. By contrast, a target domain contains a different category or imaging condition that is kept unavailable until the source model has been selected. Thus, a test split from RSOC Building is a source test set when RSOC Building is the training source; it is not treated as a target domain merely because it is used for testing.

In the main experiment, RSOC Building [5] is the single source domain. Only its training split is used to update the model, and only its validation split is used to select the saved model state, hereafter called the checkpoint. After this choice, the model is fixed and evaluated on DOTA Large Vehicle (LV), DOTA Small Vehicle (SV), and DOTA Ship [1]. These three categories are separate unseen target domains in this experiment. This assignment is only one instance of the protocol. Section IV-E changes the source to DIOR Airplane [22] or DOTA Ship to test whether the results depend on RSOC Building.

The training program reads only the source training and validation splits. The checkpoint with the lowest average absolute count error on source validation is retained. Target images and annotations are read only by the evaluation programs and do not update model parameters or select the checkpoint reported in the main comparison. Target annotations are used solely to compute the final measures.

DOTA represents each object by an oriented quadrilateral. For vertices $\{ ( x _ { k } , y _ { k } ) \} _ { k = 1 } ^ { 4 }$ , the evaluator uses

$$
\left( \frac { 1 } { 4 } \sum _ { k = 1 } ^ { 4 } x _ { k } , \frac { 1 } { 4 } \sum _ { k = 1 } ^ { 4 } y _ { k } \right)\tag{76}
$$

as the reference point, following the supplied parser. DIOR boxes are converted in the same spirit by taking each box center. In both cases, the box or quadrilateral is used only to

obtain one point and the object count. Its size, direction, and boundary are not supplied to training or to the extraction of diagnostic points.

## B. Evaluation Measures and Comparison Settings

For image i, let $N _ { i } ^ { * }$ be the reference count and let $\widehat { N } _ { i }$ be the predicted count obtained from the standard output of the evaluated method. For COBICount and other density map methods, $\widehat { N } _ { i }$ is the sum of the predicted density map. Given N evaluation images, MAE and RMSE are

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \widehat { N } _ { i } - N _ { i } ^ { * } \right| ,\tag{77}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \widehat { N } _ { i } - N _ { i } ^ { * } \right) ^ { 2 } } .\tag{78}
$$

MAE gives the average size of the count error. RMSE gives more weight to images with a large error. Lower values are better for both measures.

For the main experiment, mean target MAE (MT-MAE) summarizes the three unseen DOTA domains:

$$
\mathrm { M T - M A E } = \frac { \mathrm { M A E } _ { L V } + \mathrm { M A E } _ { S V } + \mathrm { M A E } _ { S h i p } } { 3 } .\tag{79}
$$

The RSOC Building result is a source reference and is not included in this average. Later analyses add measures for the direction of count error and the locations of diagnostic points.

Methods that need a density map during training replace every source point with a normalized Gaussian response whose sum is one. This construction keeps the sum of the reference density map equal to the number of objects. Methods based on direct point prediction, probability, or distribution matching retain their own training objectives, and their counts are taken from their standard outputs. When a density map or point set is resized, its values or coordinates are adjusted so that the object count is preserved.

For every implemented method, the longer side of a target image is limited to 2000 pixels. This input rule is fixed before any target result is calculated, and all methods use the same preprocessing and input size within each target subset. No target label, count, box size, validation result, feature summary, or normalization statistic is used to choose that size. Each main result uses the same frozen checkpoint selected on the RSOC Building validation split; a separate checkpoint is not selected for each target.

The comparison includes MCNN [3], CSRNet [4], CAN [6], Bayesian Loss [7], DM-Count [8], P2PNet [9], TransCrowd [10], MobileNetV2 Counter [43], ResNet50 FPN Counter [44], [45], MLDG-Count [16], MPCount [31], and BDRNet [25]. Together, they cover density map regression, direct point prediction, transformer counting, compact counters, counters with a feature pyramid, generic domain generalization, and density regression for remote sensing. MPCount is evaluated as a generic baseline under source-only counting, whereas BDRNet is evaluated as a baseline designed for remote sensing counting. Both official architectures are trained under source-only counting with RSOC Building. Their entries are results obtained after retraining under this setting rather than numbers published for their original datasets. None of the compared methods receives target data for adaptation or checkpoint selection.

## C. Implementation Details

The experiments use Python 3.12.11, PyTorch 2.8.0, Torchvision 0.23.0, CUDA 12.9, and cuDNN. They run on an NVIDIA GeForce RTX 5070 Ti Laptop GPU with 11.94 GB memory, 24 CPU cores, and 31.44 GB system memory. Parameter counts and floating point operations are measured with a $5 1 2 \times 5 1 2$ input.

COBICount is trained for 80 epochs with AdamW, an optimizer that updates parameters from running averages of past gradients and applies weight decay. The initial learning rate is $1 . 8 \times 1 0 ^ { - 4 } ,$ , the weight decay is $1 0 ^ { - 4 }$ , the batch size is 6, and the random seed is 3407. Automatic mixed precision uses both lower and full precision arithmetic to reduce memory use. The $\ell _ { 2 }$ norm of the gradient is limited to 5.0. An exponential moving average (EMA) copy of the model is also maintained with a decay of 0.999. This copy combines its previous parameters with the current parameters at every update and is used for source validation and final evaluation.

The relative and logarithmic count losses reach their full weights gradually during the first eight epochs. Direct source supervision for Audit starts at epoch 10. Source training uses $5 1 2 \times 5 1 2$ crops. With probability 0.85, a crop is centered near a randomly chosen source point, with at most 96 pixels of horizontal and vertical displacement. Otherwise, a random crop is used. Horizontal and vertical flips are each applied with probability 0.5. With probability 0.25, brightness and contrast factors are sampled from [0.85, 1.15], and color saturation is sampled from [0.90, 1.10]. Source validation images are resized to $5 1 2 \times 5 1 2$ . All images are normalized with the ImageNet channel means and standard deviations.

MPCount and BDRNet are trained on the source for 80 epochs with $5 1 2 \times 5 1 2$ source crops, seed 3407, automatic mixed precision, and checkpoint selection from source validation. Each uses a physical batch size of one, and gradients are accumulated over six steps, giving an effective batch size of six. Both use AdamW with a learning rate of $1 0 ^ { - 3 }$ and a weight decay of $1 0 ^ { - 4 } ;$ later changes to the learning rate follow the schedule in each released implementation. Both begin from their standard ImageNet initialization. Here, a density scale is the fixed constant used by a released implementation to rescale density values during training. MPCount uses the official deterministic DGModel\_final, a density scale of 1000, and its density, classification map, and memory consistency losses. BDRNet uses the official architecture, a density scale of 100, density regression, and its auxiliary object region output. In the released BDRNet training code, the intersection over union term measures region overlap, but it uses a hard threshold and is converted to a scalar, so it supplies no gradient. We retain its effective differentiable objective: density mean squared error plus 0.01 times the sum of binary cross entropy and Dice loss for the auxiliary object region output. Dice loss also measures overlap between the predicted and reference object regions. For MPCount and BDRNet, the program first attempts inference on the complete resized image and uses fixed nonoverlapping 512 × 512 tiles only if a GPU memory error occurs. The selected checkpoint is unchanged across all three target domains. The architecture, fixed coefficients and thresholds, loss weights, point extraction settings, and input sizes are fixed implementation settings. None is selected or adjusted using target images, target labels, target statistics, or target evaluation results. In the main experiment, only RSOC Building is used to develop these settings and select the checkpoint. When another dataset is designated as the source, only the validation split of that source selects its checkpoint.

TABLE I  
MAIN COMPARISON WHEN RSOC BUILDING IS THE SOURCE DOMAIN AND DOTA LARGE VEHICLE, DOTA SMALL VEHICLE, AND DOTA SHIP ARE THREE UNSEEN TARGET DOMAINS. ALL TRAINING FOR THE COUNTING TASK AND CHECKPOINT SELECTION USE ONLY RSOC BUILDING. NO TARGET DATA ARE USED FOR ADAPTATION OR CHECKPOINT SELECTION. MT-MAE IS THE AVERAGE MAE OVER THE THREE TARGET DOMAINS AND EXCLUDES THE RSOC BUILDING SOURCE RESULT.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Year Adapt. Params (M) GFLOPs</td><td rowspan="2"></td><td colspan="2">RSOC Building</td><td colspan="2">DOTA LV</td><td colspan="2">DOTA SV</td><td colspan="2">DOTA Ship</td><td rowspan="2">MT-MAE</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="5">Standard / larger</td><td>ResNet50 FPN Counter [44], [45]</td><td>2017</td><td>X</td><td>27.15</td><td>66.86</td><td>10.023</td><td>14.794</td><td>292.845</td><td>668.910</td><td>605.817</td><td>1322.535</td><td>590.810</td><td>882.637</td><td>496.491</td></tr><tr><td>CSRNet [4]</td><td>2018</td><td>X</td><td>16.26</td><td>27.07</td><td>7.180</td><td>10.530</td><td>225.165</td><td>504.559</td><td>491.337</td><td>1118.768</td><td>494.861</td><td>687.234</td><td>403.788</td></tr><tr><td>CAN [6]</td><td>2019</td><td>X</td><td>18.18</td><td></td><td>9.120</td><td>13.380</td><td>76.182</td><td>96.163</td><td>415.150</td><td>1244.760</td><td>239.832</td><td>372.497</td><td>243.722</td></tr><tr><td>Bayesian Loss [7]</td><td>2019</td><td>X</td><td>21.50</td><td>26.99</td><td>28.950</td><td>32.930</td><td>583.585</td><td>826.571</td><td>872.199</td><td>1695.127</td><td>925.775</td><td>1316.316</td><td>793.853</td></tr><tr><td>P2PNet [9]</td><td>2021</td><td>X</td><td>19.20</td><td>23.29</td><td>7.460</td><td>10.340</td><td>124.993</td><td>158.005</td><td>411.526</td><td>1236.752</td><td>211.785</td><td>329.987</td><td>249.435</td></tr><tr><td rowspan="4">Compact</td><td>MCNN [3]</td><td>2016</td><td>X</td><td>0.13</td><td>1.38</td><td>12.130</td><td>17.350</td><td>479.545</td><td>613.845</td><td>630.747</td><td>1269.330</td><td>507.681</td><td>679.338</td><td>539.324</td></tr><tr><td>MobileNetV2 Counter [43]</td><td>2018</td><td>X</td><td>5.47</td><td>2.40</td><td>9.893</td><td>14.935</td><td>442.202</td><td>896.202</td><td>784.845</td><td>1485.697</td><td>1079.061</td><td>1705.149</td><td>768.703</td></tr><tr><td>DM-Count [8]</td><td>2020</td><td>X</td><td>0.83</td><td>18.80</td><td>9.427</td><td>13.688</td><td>202.395</td><td>299.733</td><td>473.152</td><td>1228.686</td><td>245.784</td><td>358.591</td><td>307.110</td></tr><tr><td>TransCrowd [10]</td><td>2022</td><td>X</td><td>0.97</td><td>0.71</td><td>8.580</td><td>12.510</td><td>134.181</td><td>180.581</td><td>415.060</td><td>1229.251</td><td>208.301</td><td>317.187</td><td>252.514</td></tr><tr><td rowspan="2">Generic DG</td><td>MLDG-Count [16]</td><td></td><td>X</td><td>24.81</td><td>71.79</td><td>11.121</td><td>15.113</td><td>38.262</td><td>59.958</td><td>557.179</td><td>1371.708</td><td>275.569</td><td>416.210</td><td>290.337</td></tr><tr><td>MPCount [31]</td><td>2024</td><td>X</td><td>33.23</td><td>143.79</td><td>12.818</td><td>17.732</td><td>45.092</td><td>68.044</td><td>490.854</td><td>1321.268</td><td>280.206</td><td>424.194</td><td>272.050</td></tr><tr><td></td><td>Remote sensing BDRNet [25]</td><td>2024</td><td>X</td><td>17.09</td><td>91.64</td><td>8.043</td><td>12.588</td><td>39.477</td><td>57.487</td><td>455.877</td><td>1298.813</td><td>264.399</td><td>401.384</td><td>253.251</td></tr><tr><td>Ours</td><td>COBICount</td><td></td><td>X</td><td>5.07</td><td>17.41</td><td>6.574</td><td>9.753</td><td>29.874</td><td>36.681</td><td>327.045</td><td>941.040</td><td>165.478</td><td>266.511</td><td>174.132</td></tr></table>

× means that no target adaptation is used. DG means domain generalization. Params (M) gives the number of parameters in millions, and GFLOPs gives billions of floating point operations. Within the Params (M) and GFLOPs columns, italic values are taken from the corresponding published papers and retain the settings reported there. Upright values are measured from the implemented models in the stated environment; their GFLOPs use a 512 × 512 input. “–” means that a value is unavailable or does not apply.

## D. Main Results

Table I compares methods trained under the same rule for access to target data. It is not a ranking of every published remote sensing counter, because many published results train and test a separate model on each category. Those numbers are not directly comparable unless the methods are retrained with one source and no target data. Methods that use target images, generated target labels, target statistics, or target adaptation are also excluded [28], [29], [36]. MLDG-Count and MPCount provide generic references under source-only counting [16], [17], [30], [31]. MPCount was originally designed for crowd counting across datasets, so its row reports retraining of the official deterministic architecture on RSOC Building rather than the result published for its original crowd datasets. BDR-Net provides a recent reference designed for remote sensing and is retrained under source-only counting [25]. The domain general crowd counting method in [30] remains outside the table because no corresponding result was produced under the present remote sensing protocol.

The source and target columns answer different questions. RSOC Building measures how well a method fits the category used for training, whereas the three DOTA columns measure the transfer of the same checkpoint selected from source validation to unavailable categories and scenes. A low source error therefore does not guarantee a low target error. P2PNet, for example, obtains a source MAE of 7.460 but an MT-MAE of 249.435. BDRNet shows the same gap, with a source MAE of 8.043 and an MT-MAE of 253.251. MPCount obtains 12.818 and 272.050, respectively.

COBICount obtains the lowest MT-MAE among the compared methods, at 174.132. The closest baseline is CAN at 243.722, corresponding to a 28.553% reduction in MT-MAE. P2PNet, TransCrowd, and BDRNet obtain 249.435, 252.514, and 253.251, respectively. The two added 2024 baselines, BDRNet and MPCount, obtain MT-MAE values of 253.251 and 272.050. Relative to them, COBICount reduces MT-MAE by 31.241% and 35.993%, respectively. COBICount also gives the lowest source MAE and the lowest MAE on each of the three unseen target domains.

The recent baselines provide two further observations. BDRNet is both smaller and more accurate than MPCount under this protocol: it uses 17.09 million parameters and 91.64 GFLOPs, compared with 33.23 million and 143.79 GFLOPs for MPCount, and it gives a lower MAE on the source and every target domain. This pattern is consistent with the value of a counting design for remote sensing, but it does not isolate architecture as the only cause because the methods retain different objectives and learning rate schedules. Both methods transfer comparatively well to DOTA Large Vehicle, with MAEs of 39.477 and 45.092, but their errors increase on DOTA Small Vehicle and Ship. In particular, their Small Vehicle RMSE values reach 1298.813 and 1321.268, confirming that dense small objects and a few images with

TABLE II  
EVALUATION WITH THREE CHOICES OF SOURCE DOMAIN. EACH ROW GIVES THE SINGLE DOMAIN USED FOR TRAINING AND CHECKPOINT SELECTION. EACH COLUMN GIVES THE DOMAIN USED FOR EVALUATION. ENTRIES OUTSIDE THE MATCHING SOURCE COLUMN MEASURE TRANSFER TO AN UNSEEN DOMAIN WITHOUT TARGET ADAPTATION.
<table><tr><td rowspan="2">Source domain</td><td rowspan="2">Adapt.</td><td colspan="2">RSOC Building</td><td colspan="2">| DIOR Airplane</td><td colspan="2">DOTA Ship</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>RSOC Building</td><td>×</td><td>6.574</td><td>9.753</td><td>15.206</td><td>21.957</td><td>165.478</td><td>266.511</td></tr><tr><td>DIOR Airplane</td><td>×</td><td>17.389</td><td>19.496</td><td>0.885</td><td>1.579</td><td>220.378317.497</td><td></td></tr><tr><td>DOTA Ship</td><td>X</td><td>11.278</td><td>16.373</td><td>49.170</td><td>50.339</td><td>81.980</td><td>141.044</td></tr></table>

× means that no target adaptation is used. DIOR Airplane uses only box centers as point annotations.

large errors remain difficult.

COBICount also remains imperfect. Its MAE/RMSE is 327.045/941.040 on DOTA Small Vehicle and 165.478/266.511 on DOTA Ship. After resizing, small vehicles occupy fewer output cells than source buildings, whereas parking lines, roads, shadows, water boundaries, and harbor structures remain visually strong. MLDG-Count, BDRNet, and MPCount obtain low Large Vehicle MAEs of 38.262, 39.477, and 45.092 but do not retain the same accuracy across the other two targets. These results indicate that source variation or a recent architecture for remote sensing alone does not determine whether a response lies on an object rather than on a repeated background pattern. The later ablation and diagnostic point analyses examine this distinction for the methods for which the required spatial diagnostic outputs were produced.

## E. Tests with Different Source Domains

The main comparison uses one source choice. Table II tests two additional choices to determine whether COBICount is tied to RSOC Building. Each row names the single annotated domain used for training and checkpoint selection. Each group of columns names the domain used for evaluation after that checkpoint is fixed. An entry in which the row and column name the same domain is a source reference. Every other entry evaluates an unseen domain. For DIOR Airplane, only box centers are used as points; box size and boundaries are not used.

The result changes when the training source changes. DIOR Airplane to RSOC Building gives 17.389/19.496 MAE/RMSE, showing that a model trained outside RSOC can still provide useful building counts. However, the same DIOR Airplane model gives 220.378/317.497 on DOTA Ship. In the reverse direction, the DOTA Ship model gives 49.170/50.339 on DIOR Airplane, compared with the DIOR Airplane source reference of 0.885/1.579. Transfer is also asymmetric between RSOC Building and DOTA Ship: the two directions give 165.478/266.511 and 11.278/16.373, respectively.

During training, source annotations show where counting responses should occur. From the source images around those annotations, the model also learns the usual width and shape of a response, the number of nearby objects, and the backgrounds that often appear around them. A new category can differ in each of these aspects. The table therefore supports two conclusions: COBICount is not restricted to RSOC Building as its source, but its accuracy still depends on how closely the source patterns match those in the unseen domain.

![](images/20f48053cb50de22572ef0950544881b17365440537416b40c65f884e812570e.jpg)  
Fig. 5. Examples from the ablation study when RSOC Building is the source and DOTA is unseen. The panels compare the input, reference points, the full model, and selected variants. They show how removing response generation, local acceptance, or structure suppression changes the final density map. Strong values on roads, parking structures, shadows, and repeated textures are responses without object support.

## V. ABLATION STUDY AND ANALYSIS

## A. Ablation of Model Parts

Table III tests which parts of COBICount contribute to the reported behavior. Every altered model is trained from source data, and its reported checkpoint is selected by RSOC Building validation MAE. The DOTA Proxy column evaluates this checkpoint under a controlled crop size. It is called a proxy because it samples fixed crops rather than complete DOTA images. The set contains $5 1 2 \times 5 1 2$ crops centered near a randomly selected DOTA reference point. The random seed and the allowed displacement are fixed, a crop can contain additional objects, and the number of crops drawn from each retained image is fixed. Images without an object of the selected category are excluded.

This sampling rule favors regions that contain targets and omits empty images. Its MAE and RMSE are therefore not comparable with the measures on complete DOTA images in Table I. The proxy values are used only for the ablation analysis. They do not enter the loss and do not select the checkpoint reported in Table III; that checkpoint is selected from RSOC Building validation data.

The variants follow the order of the counting path. “Density only” uses the three routed density branches and the effective valid mask, while bypassing the other CE cues, CA, BI, and Audit supervision. “w/o BI” sets the BI retention gate $\mathbf { V }$ to its neutral value. “w/o $\mathbf { B } \mathbf { G } ^ { \prime }$ removes the learned background response $\mathbf { B } _ { b g } .$ . “w/o Line/Broad/Grid” removes the three structure risk terms ${ \bf R } _ { l i n e }$ $\mathbf { R } _ { b r o a d } .$ , and ${ \mathbf { R } } _ { g r i d } .$ . “w/o Support” removes the pooled support cue $\mathbf { E } _ { h }$ from CE. “w/o Center” removes the center factor $\mathbf { A } _ { c }$ from CA while retaining its other factors. “w/o Audit” removes the four channel Audit output and its source supervision.

The variants reveal two different effects. First, “Density only” and “w/o $\mathbf { B } \mathbf { I } ^ { \prime \prime }$ remove different sets of parts. “Density only” removes the three bounded CE cues together with CA and BI, leaving only the routed density path and the effective valid mask. By contrast, “w/o BI” retains the broader CE map and CA but fixes the retention gate V at its neutral value. Counting mass that BI would normally suppress and rescale can therefore accumulate. This difference explains why “Density only” remains numerically stable, whereas “w/o $\mathbf { B } \mathbf { I } ^ { \prime \prime }$ raises RSOC Building MAE to 610.964 and DOTA Proxy MAE to 655.634. The result shows that BI controls both structure related responses and the scale of the summed density in the complete model; it does not show that each BI map identifies a named background category.

TABLE III  
ABLATION OF COBICOUNT ON RSOC BUILDING AND DOTA PROXY. EACH ROW REMOVES ONE MODEL PART OR ONE RELATED GROUP. THE REPORTEDCHECKPOINT IS SELECTED BY RSOC BUILDING VALIDATION MAE. DOTA PROXY IS USED ONLY TO ANALYZE THE SELECTED VARIANTS AND DOESNOT SELECT A REPORTED CHECKPOINT.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Bias Isolation</td><td rowspan="2">Candidate Evidence</td><td rowspan="2">Candidate Acceptance</td><td rowspan="2">Audit</td><td colspan="2">RSOC Building</td><td colspan="2">DOTA Proxy</td></tr><tr><td>BG</td><td>Line/Broad/Grid Support</td><td>Center</td><td>MAE</td><td>RMSE</td><td>MAE RMSE</td></tr><tr><td>Density only</td><td>一</td><td>一</td><td>一</td><td>一</td><td>1</td><td>9.334</td><td>14.342</td><td>25.841</td><td>54.013</td></tr><tr><td>w/o BI</td><td>1</td><td>一</td><td>√</td><td>V</td><td>√</td><td>610.964</td><td>625.417</td><td>655.634</td><td>665.124</td></tr><tr><td>w/o BG</td><td>一</td><td>√</td><td>√</td><td>V</td><td>V</td><td>261.307</td><td>268.222</td><td>283.996</td><td>292.246</td></tr><tr><td>w/o Line/Broad/Grid</td><td>√</td><td>一</td><td>√</td><td>V</td><td>V</td><td>447.158</td><td>457.968</td><td>484.697</td><td>492.678</td></tr><tr><td>w/o Support</td><td>√</td><td>V</td><td>一</td><td>√</td><td>√</td><td>10.657</td><td>15.300</td><td>27.835</td><td>54.651</td></tr><tr><td>w/o Center</td><td>√</td><td>√</td><td>V</td><td>一</td><td>√</td><td>7.043</td><td>10.961</td><td>18.865</td><td>54.687</td></tr><tr><td>w/o Audit</td><td>V</td><td>V</td><td>√</td><td>V</td><td>一</td><td>7.708</td><td>11.322</td><td>17.977</td><td>54.728</td></tr><tr><td>Full</td><td>V</td><td>√</td><td>√</td><td>√</td><td>V</td><td>6.574</td><td>9.753</td><td>16.937</td><td>53.063</td></tr></table>

BG is the learned background response. Line/Broad/Grid denotes the three structure risk terms in BI. Support is the pooled support cue $\mathbf { E } _ { h } .$ . Center is the center gate $\mathbf { A } _ { c } .$ Audit is an auxiliary output trained with source annotations; it does not gate the final density map. $\cdot \cdot \_ { \gamma }$ means that a part is disabled or does not apply.

Second, the BI terms are not interchangeable. Removing BG raises DOTA Proxy MAE to 283.996, while removing the line, broad, and grid terms raises it to 484.697. Figure 5 shows the corresponding increases on parking lines, road boundaries, shadows, and repeated structures. The table and figure therefore indicate that both the learned background response and the three structure risks are needed in the current formulation.

The CE support cue and the CA center gate have smaller but distinct effects. Removing $\mathbf { E } _ { h }$ raises DOTA Proxy MAE from 16.937 to 27.835, which is consistent with a loss of weak responses that need evidence from a wider local area. Removing $\mathbf { A } _ { c }$ raises RSOC Building MAE from 6.574 to 7.043 and DOTA Proxy MAE from 16.937 to 18.865. Removing Audit gives a more moderate change, as expected for an auxiliary output that does not gate the density map. The full model gives the lowest MAE on both sets.

## B. Prediction Bias

MAE and RMSE measure the size of an error, but not whether a model usually predicts too many or too few objects. To show this direction, we use Pred/GT, the ratio between the mean predicted count and the mean reference count:

$$
\mathrm { P r e d / G T } = \frac { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \widehat { N } _ { i } } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } N _ { i } ^ { * } } = \frac { \sum _ { i } \widehat { N } _ { i } } { \sum _ { i } N _ { i } ^ { * } } .\tag{80}
$$

TABLE IV  
PRED/GT FOR SELECTED METHODS WHEN RSOC BUILDING IS THE SOURCE AND THE THREE DOTA CATEGORIES ARE UNSEEN TARGETS.
<table><tr><td>Method</td><td>RSOC DOTA LV</td><td>DOTA SV</td><td>DOTA Ship</td></tr><tr><td>Bayesian Loss [7]</td><td>0.89</td><td>10.30</td><td>1.81 4.02</td></tr><tr><td>P2PNet [9]</td><td>0.96</td><td>2.96</td><td>0.44 0.88</td></tr><tr><td>CSRNet [4]</td><td>0.96</td><td>4.50 1.21</td><td>2.29</td></tr><tr><td>CAN [6]</td><td>0.90</td><td>2.06 0.38</td><td>0.70</td></tr><tr><td>DM-Count [8]</td><td>0.95</td><td>4.21 0.80</td><td>1.21</td></tr><tr><td>TransCrowd [10]</td><td>0.95</td><td>3.05 0.46</td><td>0.93</td></tr><tr><td>MobileNetV2 [43]</td><td>0.80</td><td>8.04 2.07</td><td>4.33</td></tr><tr><td>MLDG-Count [16]</td><td>0.89</td><td>0.45 0.05</td><td>0.09</td></tr><tr><td>COBICount</td><td>0.97</td><td>1.78 0.52</td><td>0.92</td></tr></table>

Values above one indicate a prediction that is too high on average. Values below one indicate a prediction that is too low.

This is a ratio of two dataset means, not the mean of the ratios for individual images. It remains defined when some images contain no target, provided that the complete evaluation set contains at least one target. A value above one means that the total prediction is too high on average. A value below one means that it is too low.

The errors have a clear direction. Bayesian Loss, CSRNet, DM-Count, and MobileNetV2 produce much too large a total on DOTA Large Vehicle or DOTA Ship. Together with the visual maps, this pattern is consistent with count being assigned to roads, roof edges, water boundaries, harbor structures, or other strong background patterns. CAN, TransCrowd, P2PNet, and MLDG-Count instead produce too small a total on DOTA Small Vehicle, which is consistent with weak vehicle responses being missed after the large change in response size.

COBICount reduces these extremes but does not remove them. Its Pred/GT is 1.78 on DOTA Large Vehicle, so the total remains too high. On DOTA Ship, 0.92 is close to one compared with the large values of Bayesian Loss, CSRNet, and MobileNetV2. On DOTA Small Vehicle, 0.52 shows that COBICount still misses count, although the shortage is less severe than for CAN, TransCrowd, P2PNet, and MLDG-Count. In combination with the ablation results, these trends are consistent with CE retaining weak responses and BI limiting large amounts of background response. Pred/GT is therefore useful beside MAE because two methods with a similar absolute error can err in opposite directions.

## C. Locations of Diagnostic Points

A correct total does not guarantee that counting responses lie on objects because missed objects and responses on background structures can partly cancel. We therefore use the diagnostic points defined in Section III-I. They are extracted after the numerical count and are used only to compute Recall@32, Precision@32, and Diag/GT and to support the visualization in Fig. 6. Their number is not the predicted count; for every density map method in this comparison, the count remains the sum of the complete density map.

The location comparison uses a representative diagnostic subset consisting of CAN, DM-Count, MLDG-Count, and COBICount. These methods cover a density counter that uses context, a counter based on distribution matching, a generic reference under source-only counting, and the proposed model. MPCount and BDRNet are included in the numerical comparison in Table I, but no validated diagnostic point results are reported for them. P2PNet is not included because its standard output is a set of predicted object points rather than a density map. TransCrowd is not included because the implementation used in this study returns a numerical count but does not expose a compatible spatial map. This analysis is therefore not an exhaustive ranking of every method in Table I; it inspects response locations for the stated subset.

The numerical count results in Table I follow the input and preprocessing settings stated in Section IV-B; each count is taken from the standard numerical output of the evaluated method. The location analysis in this subsection is a separate diagnostic run. Here, large images are processed with nonoverlapping 512 × 512 tiles, and a border tile is padded when needed. All extracted points are then returned to the coordinates of the evaluated image. This tiling rule is used to obtain the diagnostic points in Table V and does not replace the count results in Table I. Figure 6 shows selected crops, so its displayed “Pred” value refers only to the shown crop.

COBICount uses the fixed procedure defined in Section III-I because this procedure was defined with the model and operates on the network output grid, whose stride is 4. CAN, DM-Count, and MLDG-Count do not provide such a procedure, so one common rule is applied to their density maps at input image resolution. For these three methods, the threshold is the larger of $1 0 ^ { - 8 }$ and 0.28 times the largest map value. Nearby maxima within 6 pixels are suppressed, and at most 4096 points are retained. Neither procedure is changed for a target domain. Because the output grids and extraction rules differ, Table V is an analysis of the resulting response locations, not a controlled ranking under one identical point extraction rule.

For image i, let $\mathcal { G } _ { i }$ contain the reference points and let $\mathcal { Q } _ { i }$ contain the extracted diagnostic points. A reference point is covered when at least one diagnostic point lies within 32 pixels. A diagnostic point is supported when at least one reference point lies within the same radius. The reported measures pool the matched and total points over the full evaluation set:

$$
\begin{array} { r l r } & { } & { \mathrm { R e c a l l @ 3 2 } = \frac { \sum _ { i } \sum _ { { \bf g } \in { \mathcal G } _ { i } } { \bf 1 } [ \operatorname* { m i n } _ { { \bf q } \in { \mathcal Q } _ { i } } \| { \bf g } - { \bf q } \| _ { 2 } \leq 3 2 ] } { \sum _ { i } | { \mathcal G } _ { i } | } , } \\ & { } & { \mathrm { P r e c i s i o n @ 3 2 } = \frac { \sum _ { i } \sum _ { { \bf q } \in { \mathcal Q } _ { i } } { \bf 1 } [ \operatorname* { m i n } _ { { \bf g } \in { \mathcal G } _ { i } } \| { \bf q } - { \bf g } \| _ { 2 } \leq 3 2 ] } { \sum _ { i } | { \mathcal Q } _ { i } | } , } \\ & { } & { \mathrm { D i a g / G T } = \frac { \sum _ { i } | { \mathcal Q } _ { i } | } { \sum _ { i } | { \mathcal G } _ { i } | } . \qquad } \end{array}\tag{81}
$$

The minimum distance to an empty set is treated as infinity. Recall@32 is the fraction of reference objects covered by at least one diagnostic point. Precision@32 is the fraction of diagnostic points supported by at least one reference object. Diag/GT compares the total number of diagnostic points with the total number of reference points; Diag denotes the extracted diagnostic points, and GT denotes the reference points. The first two measures use independent nearest point tests, not a one to one assignment. Thus, two diagnostic points can be supported by the same object, and one diagnostic point can cover two nearby objects. If no diagnostic point is extracted, the denominator of Precision@32 is zero and the measure is undefined. We report this case as “–”, while Recall@32 and Diag/GT are both zero.

Table V confirms that count accuracy and response location describe different properties. On DOTA Large Vehicle, CAN extracts 3.408 diagnostic points per reference point, but its Recall@32 and Precision@32 are only 0.279 and 0.079. Thus, many diagnostic points are away from annotated vehicles, while most reference vehicles remain uncovered. On DOTA Small Vehicle, its Diag/GT falls to 0.714, and its Recall@32 and Precision@32 are 0.193 and 0.186. CAN therefore extracts fewer points relative to the number of objects and still misses most small vehicles.

DM-Count extracts even fewer diagnostic points on DOTA Small Vehicle, with a Diag/GT of 0.323. Its Recall@32 is 0.036 and its Precision@32 is 0.068, showing that the small set of extracted points still has weak agreement with the reference points. MLDG-Count shows a different error. Its Diag/GT reaches 1.623, but its Recall@32 and Precision@32 are only 0.060 and 0.071. These results show that neither a large nor a small number of diagnostic points alone indicates reliable response locations.

On the RSOC Building source domain, COBICount obtains a Recall@32 of 0.809 and a Precision@32 of 0.680. After transfer, it retains values of 0.742 and 0.542 on DOTA Large Vehicle and 0.614 and 0.394 on DOTA Ship. DOTA Small Vehicle remains more difficult. COBICount and CAN extract similar numbers of diagnostic points relative to the reference points, with Diag/GT values of 0.703 and 0.714, respectively. However, COBICount gives higher Recall@32 and Precision@32, at 0.238 and 0.301 compared with 0.193 and 0.186 for CAN. On DIOR Airplane, its Diag/GT is 1.879 but its Precision@32 is 0.214, indicating that many diagnostic points remain away from airplane annotations.

TABLE V  
LOCATION ANALYSIS FOR SELECTED COMBINATIONS OF METHODS AND DOMAINS USING A MATCHING RADIUS OF 32 PIXELS. ALL COMPARED OUTPUTS ARE DENSITY MAPS. RECALL@32 MEASURES THE COVERAGE OF REFERENCE OBJECTS, PRECISION@32 MEASURES THE FRACTION OF DIAGNOSTIC POINTS SUPPORTED BY REFERENCE OBJECTS, AND DIAG/GT COMPARES THE NUMBERS OF DIAGNOSTIC AND REFERENCE POINTS.
<table><tr><td>Method</td><td>Evaluation domain</td><td>Recall@32</td><td>Precision@32</td><td>Diag/GT</td></tr><tr><td>CAN [6]</td><td>DOTA LV</td><td>0.279</td><td>0.079</td><td>3.408</td></tr><tr><td>CAN [6]</td><td>DOTA SV</td><td>0.193</td><td>0.186</td><td>0.714</td></tr><tr><td>DM-Count [8]</td><td>DOTA SV</td><td>0.036</td><td>0.068</td><td>0.323</td></tr><tr><td>MLDG-Count [16]</td><td>DOTA SV</td><td>0.060</td><td>0.071</td><td>1.623</td></tr><tr><td>COBICount</td><td>RSOC Building</td><td>0.809</td><td>0.680</td><td>1.085</td></tr><tr><td>COBICount</td><td>DIOR Airplane</td><td>0.358</td><td>0.214</td><td>1.879</td></tr><tr><td>COBICount</td><td>DOTA LV</td><td>0.742</td><td>0.542</td><td>1.733</td></tr><tr><td>COBICount</td><td>DOTA SV</td><td>0.238</td><td>0.301</td><td>0.703</td></tr><tr><td>COBICount</td><td>DOTA Ship</td><td>0.614</td><td>0.394</td><td>1.273</td></tr></table>

![](images/9bdb018590497f98a0786121a59f20936d786a2e23a0e80854c8cf9bcc5184f0.jpg)  
Fig. 6. Response location examples on unseen DOTA and DIOR images. Each row contains an input crop, the reference points, and results from CAN, DM-Count, MLDG-Count, and COBICount. Green marks are reference points, and cyan marks are diagnostic points. The cyan marks are used only for visualization and location analysis; they are not native point predictions and do not produce the numerical count. As explained in Section V-C, COBICount and the baselines use fixed procedures suited to their respective output grids rather than one identical extraction rule. The displayed “Pred” value is the sum of the density map for the shown crop and is independent of the number of cyan marks.

Figure 6 provides examples of these location errors. Some density maps produce many diagnostic points on parking lines, roads, water boundaries, harbor structures, or empty regions, whereas other maps leave many reference objects uncovered. Count error, Pred/GT, and the diagnostic point measures therefore answer three separate questions: how large the numerical error is, whether the total is too high or too low, and whether the local responses occur near objects.

## D. Boundary Analysis on DOTA Small Vehicle

DOTA Small Vehicle marks the main boundary of the present method. After the input is resized and passed to the network output grid with stride 4, a small vehicle may occupy only a few grid cells. A source building usually produces a wider response. At the same time, parking lines, road boundaries, lane marks, shadows, and enclosed empty spaces can remain clear over many cells. The local response from a true vehicle can therefore be weaker than the response from a repeated background pattern.

This difference is an observed property of the evaluated images, not extra information supplied to the model. Target boxes are not used to set a response width, train the model, choose the checkpoint, or select an input size. Figure 7 groups the observed errors into responses on parking marks, roadside structures, and enclosed empty regions.

Parking marks  
![](images/7592f55e998133fdf28869cfb2c7b60ef04966ca46654a7a16693d91117782e9.jpg)

Roadside structures  
![](images/0dd96ea83270ee56a313cf1195fa5c59befc04f4296ed88a4cab4f4b757454ef.jpg)

![](images/a315eabd9e0b6a50cf0451cf4be06d3d7e84ec652ae2a3ade8524249be19b674.jpg)  
Fig. 7. Representative errors on DOTA Small Vehicle. Manual inspection compares the input image, reference points, final density map, and diagnostic points. Strong responses occur on parking marks, roadside structures, and enclosed empty regions. These descriptions are made by the evaluator; they are not semantic classes predicted by Audit.

Physical object size alone does not explain these errors. What matters to the network is how much visible evidence remains on its network output grid. A small vehicle can lose most of its local pattern after resizing, whereas a long parking line or road edge remains strong. This problem is less severe for DOTA Large Vehicle because a large vehicle occupies more cells and usually forms a clearer local response.

CE, CA, and BI reduce some of these errors, but none removes the boundary completely. CE can produce little evidence for a very weak vehicle. CA can also retain a compact response from an enclosed empty region, and BI may not fully separate a weak vehicle from a clear parking or roadside pattern. Future work should generate stronger responses for objects that occupy few cells and provide stronger suppression for repeated source background patterns that resemble such objects.

## VI. CONCLUSION

Many remote sensing counters are developed with annotated data from the category used at deployment, while methods for transfer often require target images or several annotated source domains. These requirements are difficult to meet when a new region, sensor, or object category must be processed without prior data collection. We therefore studied sourceonly counting. This setting reduces the data needed before deployment and avoids further training for every new target. The difficulty is that a source model may respond not only to real objects but also to roads, parking patterns, roof boundaries, water boundaries, and other repeated structures. COBICount addresses this difficulty by separating response generation, acceptance, and background suppression. CE first produces a broad map of possible responses. CA then retains compact responses that agree with the patterns learned around source point annotations. BI finally reduces responses associated with learned background structures. Their outputs form the final density map, whose sum gives the predicted count. All supervision is obtained from source images and source point annotations. No target image, label, automatically generated label, feature summary, or model selection signal is used before the model is fixed. A separate Audit output is used only to inspect response patterns; it neither changes the final density map nor predicts named target categories. When trained on RSOC Building and applied directly to DOTA Large Vehicle, Small Vehicle, and Ship, COBICount obtains the lowest MT-MAE among the 12 compared baselines, including the 2024 MPCount and BDRNet methods, at 174.132. Relative to the closest baseline, CAN at 243.722, this is a 28.553% reduction. The model contains 5.07 million parameters and requires 17.41 billion floating point operations for a $5 1 2 \times 5 1 2$ input. Tests that use DIOR Airplane or DOTA Ship as the source show that the model design can be trained from sources other than RSOC Building, although its accuracy still depends on the source data. The ablation results further show that CE, CA, and BI each contribute to the reported result. The experiments also show why count error alone provides incomplete evidence. MAE and RMSE measure the size of the numerical error, while Pred/GT shows whether the predicted total is generally too high or too low. Diagnostic point analysis examines whether local responses occur near reference points without changing the numerical count. On DOTA Large Vehicle, COBICount obtains a Recall@32 of 0.742 and a Precision@32 of 0.542. The lower values of 0.238 and 0.301 on DOTA Small Vehicle reveal the remaining difficulty in locating responses on dense small objects. These measures therefore separate numerical count accuracy from response location. DOTA Small Vehicle remains the main limitation of COBICount. After resizing, a small vehicle may occupy only a few output cells, while parking lines, road boundaries, shadows, and empty parking spaces can retain strong patterns. The model may therefore miss weak vehicle responses or retain responses from the background. Future work should strengthen responses for objects that occupy few cells, suppress repeated background structures without removing weak objects, and reduce computation for deployment on edge hardware. It should also examine how source data can be selected or combined without relying on target data.

## DATA AND CODE AVAILABILITY

The RSOC dataset is publicly available at https://opendatalab.org.cn/OpenDataLab/RSOC, and the DOTA dataset is publicly available at https://captain-whu.github.io/DOTA/dataset.html.

The code is available at https://github.com/yixuxi22/ COBICount.

## REFERENCES

[1] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “DOTA: A large-scale dataset for object detection in aerial images,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 3974–3983.

[2] D. Lam, R. Kuzma, K. McGee, S. Dooley, M. Laielli, M. Klaric, Y. Bulatov, and B. McCord, “xView: Objects in context in overhead imagery,” arXiv preprint arXiv:1802.07856, 2018.

[3] Y. Zhang, D. Zhou, S. Chen, S. Gao, and Y. Ma, “Single-image crowd counting via multi-column convolutional neural network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 589–597.

[4] Y. Li, X. Zhang, and D. Chen, “CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 1091–1100.

[5] G. Gao, Q. Liu, and Y. Wang, “Counting from sky: A large-scale data set for remote sensing object counting and a benchmark method,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 5, pp. 3642–3655, 2021.

[6] W. Liu, M. Salzmann, and P. Fua, “Context-aware crowd counting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 5099–5108.

[7] Z. Ma, X. Wei, X. Hong, and Y. Gong, “Bayesian loss for crowd count estimation with point supervision,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 6142–6151.

[8] B. Wang, H. Liu, D. Samaras, and M. H. Nguyen, “Distribution matching for crowd counting,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 1595–1607.

[9] Q. Song, C. Wang, Z. Jiang, Y. Wang, Y. Tai, C. Wang, J. Li, F. Huang, and Y. Wu, “Rethinking counting and localization in crowds: A purely point-based framework,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 3365–3374.

[10] D. Liang, X. Chen, W. Xu, Y. Zhou, and X. Bai, “TransCrowd: Weakly-supervised crowd counting with transformers,” Science China Information Sciences, vol. 65, no. 6, p. 160104, 2022.

[11] G. Gao, Q. Liu, Z. Hu, L. Li, Q. Wen, and Y. Wang, “PSGCNet: A pyramidal scale and global context guided network for dense object counting in remote-sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–12, 2022.

[12] X. Guo, M. Anisetti, M. Gao, and G. Jeon, “Object counting in remote sensing via triple attention and scale-aware network,” Remote Sensing, vol. 14, no. 24, p. 6363, 2022.

[13] Z. Shen, G. Li, R. Xia, H. Meng, and Z. Huang, “A lightweight object counting network based on density map knowledge distillation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 2, pp. 1492–1505, 2025.

[14] D. Tuia, C. Persello, and L. Bruzzone, “Domain adaptation for the classification of remote sensing data: An overview of recent advances,” IEEE Geoscience and Remote Sensing Magazine, vol. 4, no. 2, pp. 41– 57, 2016.

[15] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-adversarial training of neural networks,” Journal of Machine Learning Research, vol. 17, no. 59, pp. 1–35, 2016.

[16] D. Li, Y. Yang, Y.-Z. Song, and T. M. Hospedales, “Learning to generalize: Meta-learning for domain generalization,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[17] L. Mansilla, R. Echeveste, D. H. Milone, and E. Ferrante, “Domain generalization via gradient surgery,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 6630–6638.

[18] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 8748–8763.

[19] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.

[20] F. Liu, D. Chen, Z. Guan, X. Zhou, J. Zhu, Q. Ye, L. Fu, and J. Zhou, “RemoteCLIP: A vision language foundation model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 16, 2024.

[21] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, “RS5M and GeoRSCLIP: A large-scale vision-language dataset and a large vision-language model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–23, 2024.

[22] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 159, pp. 296–307, 2020.

[23] X. Sun, P. Wang, Z. Yan, F. Xu, R. Wang, W. Diao, J. Chen, J. Li, Y. Feng, T. Xu, M. Weinmann, S. Hinz, C. Wang, and K. Fu, “FAIR1M: A benchmark dataset for fine-grained object recognition in highresolution remote sensing imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 184, pp. 116–130, 2022.

[24] M.-R. Hsieh, Y.-L. Lin, and W. H. Hsu, “Drone-based object counting by spatially regularized regional proposal network,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 4145– 4153.

[25] H. Guo, J. Gao, and Y. Yuan, “Balanced density regression network for remote sensing object counting,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–13, 2024.

[26] Y. Wang, Z. Wen, and X. Huang, “MSCA-Net: Multiscale chunked attention network for high-resolution satellite stereo matching,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 27 745–27 763, 2025.

[27] M. Wieland, F. Fichtner, S. Martinis, S. Groth, C. Krullikowski, S. Plank, and M. Motagh, “S1S2-Water: A global dataset for semantic segmentation of water bodies from sentinel-1 and sentinel-2 satellite images,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 17, pp. 1084–1099, 2024.

[28] X. Ma, X. Zhang, X. Ding, M.-O. Pun, and S. Ma, “Decompositionbased unsupervised domain adaptation for remote sensing image semantic segmentation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–18, 2024.

[29] W. Liu, J. Liu, X. Su, H. Nie, and B. Luo, “Source-free domain adaptive object detection in remote sensing images,” arXiv preprint arXiv:2401.17916, 2024.

[30] Z. Du, J. Deng, and M. Shi, “Domain-general crowd counting in unseen scenarios,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 1, pp. 561–570, 2023.

[31] Z. Peng and S.-H. G. Chan, “Single domain generalization for crowd counting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 28 025–28 034.

[32] X. Chen, S. Huo, B. Jiang, H. Hu, and X. Chen, “Single domain generalization for few-shot counting via universal representation matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 4639–4649.

[33] Y. Cong, S. Khanna, C. Meng, P. Liu, E. Rozi, Y. He, M. Burke, D. B. Lobell, and S. Ermon, “SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 197–211.

[34] C. J. Reed, R. Gupta, S. Li, S. Brockman, C. Funk, B. Clipp, K. Keutzer, S. Candido, M. Uyttendaele, and T. Darrell, “Scale-MAE: A scale-aware masked autoencoder for multiscale geospatial representation learning,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4088–4099.

[35] M. Noman, M. Naseer, H. Cholakkal, R. M. Anwer, S. Khan, and F. S. Khan, “Rethinking transformers pre-training for multi-spectral satellite imagery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 811–27 819.

[36] Z. Gong, Z. Wei, D. Wang, X. Ma, H. Chen, Y. Jia, Y. Deng, Z. Ji, X. Zhu, N. Yokoya, J. Zhang, B. Du, and L. Zhang, “CrossEarth: Geospatial vision foundation model for domain generalizable remote sensing semantic segmentation,” arXiv preprint arXiv:2410.22629v1, 2024. [Online]. Available: https://arxiv.org/abs/2410.22629v1

[37] J. Pan, Y. Liu, Y. Fu, M. Ma, J. Li, D. P. Paudel, L. Van Gool, and X. Huang, “Locate anything on earth: Advancing open-vocabulary object detection for remote sensing community,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, pp. 6281–6289, 2025.

[38] D. Wang, J. Zhang, B. Du, M. Xu, L. Liu, D. Tao, and L. Zhang, “SAMRS: Scaling-up remote sensing segmentation dataset with segment anything model,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 8815–8827.

[39] K. Chen, C. Liu, H. Chen, H. Zhang, W. Li, Z. Zou, and Z. Shi, “RSPrompter: Learning to prompt for remote sensing instance segmentation based on visual foundation model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[40] L. P. Osco, Q. Wu, E. L. de Lemos, W. N. Gonc¸alves, A. P. M. Ramos, J. Li, and J. Marcato, Junior, “The segment anything model (SAM) for remote sensing applications: From zero to one shot,” International Journal of Applied Earth Observation and Geoinformation, vol. 124, p. 103540, 2023.

[41] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann, “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, no. 11, pp. 665–673, 2020.

[42] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 618–626.

[43] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 4510–4520.

[44] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[45] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2117–2125.