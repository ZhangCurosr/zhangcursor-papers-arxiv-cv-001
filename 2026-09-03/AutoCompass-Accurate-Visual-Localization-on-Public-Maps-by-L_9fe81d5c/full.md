# AutoCompass: Accurate Visual Localization on Public Maps by Learning from Weak Labels

Javier Tirado-Garín<sup>1⋆</sup>, Alan Savio Paul<sup>2</sup>, Shuai Chen<sup>2</sup>, Axel Barroso-Laguna<sup>2</sup>, Tommaso Cavallari<sup>2</sup>, Daniyar Turmukhambetov<sup>2</sup>, Victor Adrian Prisacariu<sup>2,3</sup>, and Eric Brachmann<sup>2</sup>

<sup>1</sup>I3A, Universidad de Zaragoza <sup>2</sup>Niantic Spatial <sup>3</sup>University of Oxford https://nianticspatial.github.io/autocompass/

Abstract. Neural map matchers estimate an image’s 3-DoF pose relative to a 2D map. These models are trained on large-scale datasets of geo-referenced images, whose position and heading labels often contain noise that afects the trained models. To address this, we present Auto-Compass, a supervision approach for training neural map matchers from inaccurate absolute pose labels. First, we show that heading labels are unnecessary: trained from raw GPS labels, models learn to predict accurate headings, automatically. Second, defining a tolerance region around raw GPS improves positional accuracy. Third, if available, our supervision uses relative poses between training images, obtained via SLAM or SfM, which provide a more accurate training signal. Across driving and egocentric benchmarks, AutoCompass consistently outperforms counterparts trained with the usual strong reliance on absolute pose labels.

Keywords: Visual Localization · Geo-Localization · Weak Supervision

## 1 Introduction

To go anywhere, first we must know where we are. To find ourselves, we consult maps, often aided by satellite positioning systems such as GPS. GPS tells us where we are, at least up to a few meters under favorable conditions. However, its performance declines when signals are obstructed or deliberately jammed. In such cases, we can practice our archaic skills of matching street names to maps.

Neural map matchers [36,83,84] ofer an alternative: they infer a local bird’seye-view (BEV) representation from an input image, and match it to a 2D map to estimate the 2D position and heading of the camera. Compared to GPS, they are unafected by multipath interference and signal jamming, and can be more accurate in urban environments, especially when estimates are fused sequentially. Neural map matchers are also scalable: once trained, they rely on 2D maps, which providers, such as OpenStreetMap [73], make freely available for much of the planet. This contrasts with 6-degree-of-freedom (DoF) visual localization [6, 13,

![](images/f8255dc59e2c2040f43f71371f8b4642c915d627ded0d189441f80baefb463b6.jpg)  
Fig. 1: Geo-localization datasets can be noisy. Pseudo ground-truth (GT) poses of a training sequence from the MGL dataset [83] are ofset by several meters: images captured on a sidewalk have poses inside a building. Methods trained with strong supervision, e.g. [83], learn and replicate these errors, while our proposed AutoCompass is robust to them. We instead treat the GT as weak labels, modeling the unknown true pose as lying anywhere within a neighborhood of the GT label. We also learn from relative poses, which can be obtained from SLAM or SfM and are thus more accurate.

81], which typically relies on memory-intensive maps built from densely captured images, incurring high storage and computational costs at city scale and beyond.

However, one crucial limitation of neural map matchers is their reliance on large-scale geo-referenced training data. Prior work has leveraged hundreds of thousands of images annotated with absolute global positions and headings [83]. At this scale, annotation must be automated, making it inherently susceptible to noise and inaccuracies. A common strategy is to reconstruct local image clusters using Structure-from-Motion (SfM) [75, 88, 96, 101], and to fuse them with raw GPS measurements, assuming that random errors cancel out. Such pipelines do not address systematic biases and can be unreliable when image clusters are small, poorly conditioned, or sparsely connected. Consequently, publicly available geo-referenced image datasets inevitably contain pose inaccuracies, which negatively impact models trained on them (see example in Fig. 1).

With AutoCompass, we reduce neural map matchers’ dependence on accurate geo-referenced camera pose labels for training. We first remove the need for heading labels: we find that the inductive bias of existing architectures is enough for accurate heading predictions to emerge. Next, to be robust to GT inaccuracies, we only assume that the true image position lies within a tolerance region of up to 20 m around the GPS coordinates. This makes our approach robust, e.g., to GPS errors of the magnitude commonly observed with receivers in egocentric devices [34, 57]. Finally, we find that relative camera poses, despite providing only local (and thus weaker) constraints, significantly improve the accuracy of the downstream learned absolute pose distribution. While geo-referencing at scale remains as a challenging problem, relative poses can be reliably estimated via SfM or Simultaneous Localization And Mapping (SLAM) [3,12,21,27,33,69,70].

Despite relying on weaker supervision than previous approaches, AutoCompass comes out on top in terms of accuracy, due to the aforementioned issues with geo-referenced pseudo ground-truth. AutoCompass is agnostic to the underlying architecture and can be applied to existing architectures without modification. We summarize our contributions as follows:

An absolute pose loss that adds significant error tolerance to geo-referenced position labels, and does not require any geo-referenced heading labels.

Using our absolute pose loss, we demonstrate that a neural map matcher can be trained from unoptimized GPS position labels, alone.

Various relative pose losses tailored to practical scenarios, e.g. non-metric poses or approximately geo-referenced ones. These losses significantly improve the accuracy of existing single-image supervised approaches

Our evaluation across multiple datasets shows that neural map matchers, trained with our proposed supervision, even when using only raw GPS labels, outperform baselines trained with strong supervision on geo-referenced image poses. Using relative pose supervision, AutoCompass sets a new state of the art in visual localization on 2D public maps.

## 2 Related work

## 2.1 Ground-to-ground visual localization

Traditional ground-to-ground visual relocalization operates in pre-mapped environments and relies on building and storing 3D representations from groundlevel data. Existing methods range from multi-stage pipelines, such as combining image retrieval [4, 5, 9, 10, 28, 41, 49–51, 99, 106, 116] with feature matching [8, 31, 32, 61, 64, 65, 68, 76, 81, 82, 85] or relative pose regression [6, 7, 29, 30], to more implicit approaches that rely on networks to memorize scenes, such as scene coordinate regression (SCR) [11, 13, 15–17, 22, 52, 94, 102, 103] or absolute pose regression (APR) [18, 26, 53–55, 67, 71, 89, 90, 109]. While most ground-toground methods focus on estimating 6-DoF camera poses, they face inherent trade-ofs between map scale, mapping and inference eficiency, and localization accuracy. For example, the most scalable structure-based solutions [81, 86, 87] require explicitly building and maintaining a large reference database via SfM, which imposes substantial storage requirements and increases matching complexity as the dataset grows. On the other hand, SCR [13, 19] or APR methods [25] can eficiently map individual scenes and scale by training on large numbers of small-to-medium-sized scenes to increase map coverage. However, none of the existing solutions can avoid the substantial costs of collecting dense ground data. Another premise of ground-to-ground methods is that accurate mapping data is essential. Prior studies [14,24] have shown that the quality of the pseudo groundtruth highly impacts localization results. Diferently, neural map matchers like

OrienterNet [83] use lightweight maps (taking about 200KB for a 128x128m area) that are free and publicly available, and do not require densely captured ground-level mapping images. In this work, our contributions reduce the strong reliance of such methods on often noisy geo-referenced ground truth.

## 2.2 Cross-view visual localization

Cross-view visual localization estimates 3-DoF camera poses (i.e., 2D position and orientation) by localizing images in 2D map representations derived from aerial/satellite imagery or planimetric maps such as OpenStreetMap (OSM). Since these map sources are typically georeferenced and available at large scale, cross-view methods enable wide-area localization without collecting and building dense image-based maps.

Early work on 2D map-based visual localization focused on controlled environments, where the map structure is relatively simple and reliable. In indoor scenes, e.g., robust pose estimation can be achieved from floor plans [23,42,45,46, 66]. More recently, C3PO [47] adapts DUSt3R point maps [104] for cross-view prediction between images and floor plans, further improving indoor localization performance. Beyond indoor settings, cross-view localization has also been explored in large-scale outdoor environments. Some methods target continentalscale localization [37, 63], achieving errors on the order of hundreds of meters. A prominent line of work uses aerial or satellite imagery as the reference map for matching [36,60,84,110,113]. While these methods can be highly robust and meter-level accurate, they often require substantial storage resources [83].

To reduce this dependence on dense imagery, recent methods have investigated localization using lightweight, widely available 2D maps. A seminal OSMbased approach, OrienterNet [83], performs neural matching by cross-correlating learned BEV features from the query image with learned map features. Following this direction, OSMLoc [62] leverages foundation models to inject strong semantic and geometric priors into its learning pipeline, while DifVL [39] reformulates visual localization as a GPS-denoising task using difusion models. Complementary approaches further improve accuracy by leveraging additional inputs at inference time, such as multiple images [20, 108] or LiDAR point clouds [117].

## 2.3 Weakly-supervised cross-view visual localization

A major challenge in cross-view localization is the need for accurate GT pose labels. To reduce this dependence, several works have explored weakly supervised learning. C-BEV [35] proposes a trainable retrieval system based on BEV features extracted from panoramic images, replacing traditional descriptor-based matching. It shows that camera pose estimation can emerge from 360°-BEV embeddings without explicit pose supervision. GeoDistill [98] also requires panoramas, using a teacher-student framework in which a teacher model with access to full panoramas supervises a student learning from perspective crops.

Other works relax the supervision requirements under diferent assumptions. Shi et al. [92] weakly supervise the position of the ground-level camera, while assuming a strong orientation prior at both training and inference time. Xia et al. [112] address datasets with heterogeneous ground-truth quality, adapting training to account for varying levels of informativeness in the GT labels.

(a) Neural-map matchers  
(b) Current methods  
![](images/22498bc0d837e3dae0381b41d519e406cce9cda4097ff2d78505d08f052afdcd.jpg)  
(c) AutoCompass

(d) AutoCompass supervision with relative poses  
![](images/cc3b993853e2d4ff92cb72f01a1264358e613a5a5be271d314fa6201bffdf936.jpg)  
Fig. 2: Robust visual localization via weak supervision. (a) We adopt neuralmap matching [83] to estimate a distribution over discrete camera poses. (b) Current methods supervise this distribution under the assumption of accurate 3-DoF pose labels (Opt.), which can break when GPS-based geo-referencing is noisy, as in the example shown. (c, d) We instead propose weak-supervision strategies that are robust to such inaccuracies. The simplest strategy, shown in (c), requires only noisy 2-DoF GPS labels, as it maximizes the probability assigned to a GPS-centered spatial chunk, thus only assuming that the true pose lies within this chunk. (d) We also learn from relative poses obtained from SfM, which are robust to ofsets in geo-referenced poses (see Fig. 3).

In contrast, AutoCompass assumes access only to perspective images at both training and inference time, together with noisy GPS measurements or relative poses for supervision. We show that this limited weak supervision is enough to outperform strongly supervised approaches.

## 3 Method

Task Given a gravity-aligned, calibrated query image I and a rasterized $H \times W$ local map tile from OpenStreetMap [73], our goal is to estimate the 3-DoF pose $\pmb { \xi } = ( x , y , \theta )$ corresponding to the image I in the map. The pose consists of a planar position $( x , \overset { \cdot } { y } ) \in \mathbb { R } ^ { 2 }$ (expressed in the geo-referenced map-tile coordinate frame) and a heading $\theta \in [ 0 , 3 6 0 ) ^ { \circ }$ . For simplicity, we assume a square tile and denote its size by $S \ ( i . e . , H = W = S )$ . In our experiments, S = 128 m.

Setting We adopt a neural map-matching approach [83] that discretizes the pose space and estimates a categorical probability distribution:

$$
P : = p ( \pmb { \xi } _ { u , v , k } \mid \mathbf { I } , \operatorname* { m a p } ) \ ,\tag{1}
$$

over a set of candidate poses $\{ \xi _ { u , v , k } \}$ , where $u , v \in \{ 0 , . . , S - 1 \}$ and $k \in$ $\{ 0 , \ldots , N { - } 1 \}$ correspond to headings $\theta _ { k } = 3 6 0 k / N ^ { \circ }$ . We use $P [ \pmb { \xi } ]$ to denote the estimated probability of the pose $\xi .$ . In practice, the network estimates an $S \times S \times N$ tensor of unnormalized scores via cross-correlation of learned $S \times S$ map features and $N$ rotated versions of bird’s-eye-view (BEV) features extracted from I. The scores are then normalized using softmax over all discrete poses $\xi _ { u , v , k } ,$ yielding $p ( \xi _ { u , v , k } \mid \mathbf { I } , \operatorname* { m a p } )$ as shown in Fig. 2. During inference, arg max $\xi _ { u , v , k } \lrcorner P [ \xi _ { u , v , k } ]$ is taken as the pose estimate. This setting is flexible and commonly adopted in related tasks such as satellite-based cross-view localization [35, 36].

Motivation Current approaches [39, 62, 83] assume perfectly geo-referenced ground-truth to supervise their 3-DoF pose estimates. However, geo-referencing is a challenging problem at scale, where manual verification is impractical. Open platforms such as Mapillary [2] help address this by jointly optimizing poses from multiple cameras that share visual observations. Although additional heuristics can be used to detect inaccuracies, such as agreement between GPS measurements and optimized poses [44,83], they do not guarantee the complete removal of incorrect estimates, and these errors can propagate to models trained on them (Fig. 1). To address this problem, we propose several supervision techniques capable of learning from potentially inaccurate data labels. Our proposals just require noisy 2-DoF GPS coordinates (Sec. 3.1) or relative poses (Sec. 3.2) obtained, $e . g .$ , from SfM. Our main strategies are depicted in Fig. 2.

## 3.1 Learning accurate 3-DoF poses from 2D labels

Automatic heading angle To supervise the probability volume predicted by neural map matchers, a common strategy [36,62,83] is to minimize the negative log-likelihood (NLL) at the ground-truth pose $\pmb { \xi } _ { ( u , v , k ) _ { \mathrm { G T } } }$

$$
\begin{array} { r } { \mathcal { L } _ { \xi } = - \log P \Big [ \xi _ { ( u , v , k ) _ { \mathrm { G T } } } \Big ] ~ . } \end{array}\tag{2}
$$

This supervision requires ground-truth labels for heading angles. However, the matching step between the learned map and BEV features provides a strong geometric inductive bias that we found can be exploited to predict the orientation. Intuitively, BEV features encode the scene semantics captured by the camera, as do the map features, so the similarity between the two feature sets is maximized at the correct orientation, provided that the geometry and semantics are estimated correctly. As a result, the heading angle can be supervised implicitly by maximizing the likelihood at the ground-truth position $( u _ { \mathrm { G T } } , v _ { \mathrm { G T } } )$ of the resulting probability distribution after marginalizing the heading probabilities:

$$
\mathcal { L } _ { \xi _ { X Y } } = - \log \sum _ { k = 0 } ^ { N - 1 } P \Big [ \xi _ { ( u _ { \mathrm { G T } } , v _ { \mathrm { G T } } , k ) } \Big ] \mathrm { ~ , ~ }\tag{3}
$$

![](images/df927728e7f04009ca8bd1b96099bac92882cf611bcf1e47d38804a557757185.jpg)  
Fig. 3: Robustness to ground-truth errors. AutoCompass’ weak supervision is insensitive to GT’s translation (middle) and rotation (right) ofsets shared by datapoints involved in our losses (Eqs. (7) and (11)) since we just use their relative information. In contrast, strong supervision [62, 83] is not robust and heavily penalizes these same errors (see loss curves), which forces networks to learn patterns that explain them.

and the model “automatically” learns useful patterns for predicting the heading.

A related observation was found by C-BEV [35], but under stronger constraints: 360° panoramic ground images and both positive and negative aerial tile samples contrasted against each query panorama during training. In contrast, our simple approach uses only single perspective images during training. Despite having a narrow field of view, this is suficient to predict heading with accuracy on-par with or better than methods supervised with heading labels.

Learning from GPS In favorable conditions, when not afected by adverse efects such as non-line-of-sight and multipath, GPS can provide meter-level accurate geo-referenced position estimates, with errors < 8 m in 95% of cases [1]. However, GPS errors are often biased [77, 107] and can become substantially larger under adverse conditions. Directly training on these measurements can thus encourage models to learn patterns that explain and replicate these errors.

Therefore, to improve over direct GPS supervision with Eq. (3), we “chunk” the probability distribution over position by marginalizing within a r ofset around the GPS coordinates, in addition to marginalizing over orientation. Concretely, we maximize the likelihood of the chunk that contains the GPS label:

$$
\mathcal { L } _ { \mathrm { G P S , \ c h u n k } } = - \log \sum _ { i = - r } ^ { r } \sum _ { j = - r } ^ { r } \sum _ { k = 0 } ^ { N - 1 } P \Bigl [ \pmb { \xi } _ { ( u _ { \mathrm { G P S } } + i , v _ { \mathrm { G P S } } + j , k ) } \Bigr ] ,\tag{4}
$$

where r is the chunk “radius”. In essence, we encourage the model to place probability mass anywhere within the r m neighborhood of the GPS measurement, again relying on the strong geometric inductive bias of the model. We experimentally show that this loss not only reduces the impact of GPS errors, but also improves over baseline models trained on optimized absolute 3-DoF poses.

As shown in Tab. 7, r is not a sensitive parameter, as similar performance is obtained for $r \in [ 5 , 2 0 ]$ m. The loss in Eq. (4) is thus robust, $e . g .$ , to GPS errors of the magnitude commonly observed with receivers in egocentric devices [34,57].

## 3.2 Learning accurate 3-DoF poses from relative poses

Relative poses, $e . g$ . obtained from SLAM [21,70] or SfM [75,88], are constrained by multi-view observations whose noise and outliers are well handled by robust optimization techniques [78, 100] and of-the-shelf systems [3, 21, 34, 75, 88]. We show that such relative poses can be used directly for training, without explicit geo-referencing, as long as a coarse 2D location $( e . g .$ ., from GPS) is available to sample an appropriate map tile covering a datapoint. To this end, and as shown in Fig. 2, we supervise the rotation and translation marginals of the relativepose distribution between pairs of datapoints, which allows us to weight them according to their discretization level. These marginals are computed from the independently predicted absolute-pose distributions of each datapoint, denoted by $P _ { 0 }$ and $P _ { 1 }$ for datapoints 0 and 1, respectively.

Relative rotation We first obtain the categorical distribution over the N discrete absolute headings by marginalizing over positions:

$$
P _ { i } ^ { \theta } [ k ] : = \sum _ { u = 0 } ^ { S - 1 } \sum _ { v = 0 } ^ { S - 1 } P _ { i } \left[ \pmb { \xi } _ { u , v , k } \right] , \qquad i \in \{ 0 , 1 \} , \ k \in \{ 0 , \ldots , N - 1 \} \ .\tag{5}
$$

Let $\Delta k \in \{ 0 , \ldots , N - 1 \}$ denote a discrete relative rotation, corresponding to $\varDelta \theta _ { \varDelta k } = 3 6 0 \varDelta k / N ^ { \circ }$ , then the distribution over relative rotations is the circular cross-correlation of the two absolute heading distributions:

$$
P _ { \Delta \Theta } [ \Delta k ] : = \sum _ { k = 0 } ^ { N - 1 } P _ { 0 } ^ { \Theta } [ k ] P _ { 1 } ^ { \Theta } \Big [ ( k + \Delta k ) \bmod N \Big ] ,\tag{6}
$$

$i . e .$ , we sum the joint probabilities of absolute headings with the same relative heading, $\varDelta k .$ under the assumption that $P _ { 0 } ^ { \Theta }$ and $P _ { 1 } ^ { \Theta }$ are independent.

For supervision, we minimize the NLL of the distribution at the ground-truth relative rotation $\varDelta k _ { \mathrm { G T } } : = \varDelta \theta _ { \mathrm { G T } } N / 3 6 0 ^ { \circ }$ :

$$
\mathcal { L } _ { \Delta \theta } ~ = ~ - \log P _ { \Delta \theta } [ \Delta k _ { \mathrm { G T } } ] ~ ,\tag{7}
$$

and linearly interpolate the log-probabilities to achieve sub-bin precision<sup>1</sup>.

Relative translation We obtain the categorical distribution over the absolute 2D position on each tile by marginalizing the predicted volume over headings:

$$
{ \cal P } _ { i } ^ { X Y } [ u , v ] : = \sum _ { k = 0 } ^ { N - 1 } { \cal P } _ { i } \left[ \xi _ { u , v , k } \right] , \qquad i \in \{ 0 , 1 \} , \ u , v \in \{ 0 , \ldots , S - 1 \} \ .\tag{8}
$$

To relate these local coordinates to a common (geo-referenced) frame, let ${ \bf o } _ { i } \in \mathbb { R } ^ { 2 }$ denote the known geo-referenced coordinates of the tile’s origin (i.e. the topleft corner of the map tile used for datapoint i). Then, each discrete absolute

translation can be written as $\mathbf { t } _ { i } ( u , v ) = \mathbf { o } _ { i } + \left[ u v \right] ^ { \top }$ , with $i \in \{ 0 , 1 \}$ , and the corresponding relative translation between two candidates is thus given by

$$
\mathbf { t } _ { 0 1 } : = \mathbf { t } _ { 1 } ( u _ { 1 } , v _ { 1 } ) - \mathbf { t } _ { 0 } ( u _ { 0 } , v _ { 0 } ) \ = \ \underbrace { ( \mathbf { o } _ { 1 } - \mathbf { o } _ { 0 } ) } _ { \mathrm { c o n s t a n t } } + \underbrace { \left[ u _ { 1 } - u _ { 0 } v _ { 1 } - v _ { 0 } \right] ^ { \top } } _ { ( \varDelta u , \varDelta v ) } \ .\tag{9}
$$

Since the tile origins $\mathbf { o } _ { 0 } , \mathbf { o } _ { 1 }$ are constants, the only random quantity to model is the discrete relative shift $( \varDelta u , \varDelta v )$ expressed in local tile coordinates. We obtain their probabilities by summing the joint probabilities of all absolute positions that induce that same shift:

$$
{ \cal P } _ { \varDelta X Y } [ \varDelta u , \varDelta v ] : = \sum _ { u = 0 } ^ { S - 1 } \sum _ { v = 0 } ^ { S - 1 } { \cal P } _ { 0 } ^ { X Y } [ u , v ] \ : { \cal P } _ { 1 } ^ { X Y } [ u + \varDelta u , v + \varDelta v ] ,\tag{10}
$$

where $P _ { 1 } ^ { X Y } [ \cdot , \cdot ]$ is treated as zero outside $\{ 0 , \ldots , S - 1 \} ^ { 2 }$ . Eq. (10) is a linear (2D) cross-correlation between two absolute translation distributions, producing a $( 2 S - 1 ) \times ( 2 S - 1 )$ categorical distribution over relative shifts.

For supervision, we minimize the NLL at the ground-truth relative shift

$$
{ \mathcal { L } } _ { \varDelta X Y } = - \log P _ { \varDelta X Y } [ \varDelta u _ { \mathrm { G T } } , \varDelta v _ { \mathrm { G T } } ] \ ,\tag{11}
$$

and bilinearly interpolate the log-probabilities to achieve sub-bin precision.

Relative distance As shown in Fig. 3, supervision with Eqs. (7) and (11) is invariant to ofsets in absolute-pose labels. However, a limitation of the relative translation loss in Eq. (11) is that it requires expressing the GT relative translation in the same (geo-referenced) coordinate system as the map tiles, so that it can be mapped to a GT relative shift $( \varDelta u _ { \mathrm { G T } } , \varDelta v _ { \mathrm { G T } } )$ in Eq. (9). To unlock the usage of relative translations expressed in an arbitrary coordinate system, we can instead supervise the probability distribution of their norms.

Starting from the discrete relative-shift distribution $P _ { \Delta X Y } ~ ( \mathrm { E q . ~ ( 1 0 ) } )$ , we can associate each shift $( \varDelta u , \varDelta v )$ with the norm of the corresponding translation, as the tile origins $\mathbf { o } _ { 0 } , \mathbf { o } _ { 1 }$ are geo-referenced and known:

$$
\mathrm { n o r m } ( \varDelta u , \varDelta v ) : = \| ( \mathbf { o } _ { 1 } - \mathbf { o } _ { 0 } ) + \left[ \varDelta u \varDelta v \right] ^ { \top } \| ,\tag{12}
$$

We then obtain the categorical distribution over (discrete) norms by marginalizing $P _ { \varDelta X Y }$ over all shifts that yield the same norm value:

$$
P _ { \| \Delta \mathbf { t } \| } [ d ] : = \sum _ { \substack { \Delta u , \Delta v : \mathrm { n o r m } ( \varDelta u , \varDelta v ) = d } } P _ { \varDelta X Y } [ \varDelta u , \varDelta v ] .\tag{13}
$$

$P _ { \| A \mathbf { t } \| }$ is non-smooth and not appropriate for interpolation at the ground-truth norm (as shown in Fig. 6). Thus, and similarly to the chunk-based marginalization of Sec. 3.1, we marginalize the raw distribution into contiguous bins and then maximize the likelihood of the bin that contains the ground-truth norm:

$$
\mathcal { L } _ { \| \Delta \mathbf { t } \| } : = - \log \sum _ { d \in \mathcal { C } ( r _ { \mathrm { G T } } ) } P _ { \| \Delta \mathbf { t } \| } [ d ] , \qquad \mathcal { C } ( r _ { \mathrm { G T } } ) : = \big \{ d \big | \big | d - d _ { \mathrm { G T } } \big | \leq r _ { \mathrm { \varDelta t } } \big \} ,\tag{14}
$$

where $r _ { \varDelta \mathbf { t } }$ represents the half-bin width. The resulting distribution is smoother and robust to errors in the relative translation norms. As shown in Tab. $7 , r _ { \mathrm { { \varDelta t } } }$ is not a sensitive parameter. In practice we use a conservative $r _ { \varDelta \mathbf { t } } = 5  { \mathrm { m } }$

## 3.3 Implementation

We build on OrienterNet [83]. For most experiments, we keep the same architecture and hyperparameters in order to isolate the efect of our contributions.

Training dataset We use the Mapillary Geo-Localization (MGL) dataset [83], containing 760k images from 12 cities across Europe and the US. At the time of writing, data from Amsterdam (72k images) is no longer available. We train AutoCompass and baselines on the remaining 11 cities. Images are captured by handheld devices or cameras mounted on cars or bikes, and come with OSM data and geo-referenced 6-DoF poses obtained by fusing SfM and GPS.

Relative-pose supervision We sample pairs of datapoints that were optimized within the same SfM cluster. We reject pairs whose camera centers are more than 100 m from each other to avoid large relative baselines potentially afected by drift. Otherwise, we do not restrict the type of relative motion. We sample OSM map tiles for each datapoint independently.

Training details We use the GPS coordinates to sample the map tile. We center the tile at the GPS location with a random translation and a random rotation to avoid overfitting. For comparisons to OrienterNet, we use the same U-Netbased architecture for the image and map branches, with ResNet-101 [43] and VGG-19 [95] backbones, respectively. We also experiment with a DINOv2 [74] image encoder inspired by [62]. We train with the Adam optimizer [56] and a batch size of 12. We typically train for 500k steps on two 40 GB NVIDIA A100 GPUs ( 3 days). More implementation details can be found in Sec. A.

## 4 Experiments

We evaluate our contributions on benchmarks with accurate, geo-referenced GT. For the main evaluations in this section, we consider two training strategies:

1. raw GPS: uses (Eq. (4)) with a chunk radius of $r { = } 5 \mathrm { m } ,$ , and   
2. w/ rel. poses: uses $0 . 1 \mathcal { L } _ { \varDelta \theta } + \mathcal { L } _ { \varDelta X Y }$ (Eq. (7), Eq. (11)) as loss.

We show that: (1) 2-DoF GPS labels, when combined with chunk marginalization (raw GPS), are suficient to outperform baselines trained with 3-DoF labels; and (2) relative-pose supervision (w/ rel. poses) yields a new state of the art. We evaluate additional weak-supervision strategies tailored to practical scenarios in Sec. C, which yield similar conclusions and $e . g$ . show that similar performance is obtained with $r \in [ 5 , 2 0 ] \mathrm { m }$ . Qualitative results are shown in Fig. 4 and Sec. D.

Baselines Our main baseline is OrienterNet [83]. We compare against its pretrained checkpoint, trained on a previous version of MGL [83] containing 12 cities. We also compare against the pretrained checkpoints of OSMLoc [62], which combines OrienterNet with a DINOv2 backbone [74] and additional losses to guide its depth predictions [115] and similarity between BEV and map features. Both baselines [62,83] are trained with strong supervision on geo-referenced labels, and we report their results after rerunning the evaluation with their released code and checkpoints. We additionally include two baselines: OrienterNet retrained on the 11 currently available cities of the MGL dataset, i.e. the same data used to train AutoCompass; and OrienterNet retrained with a DINOv2 image backbone, as in OSMLoc. Finally, for context, we also report results for stateof-the-art methods that match ground images to satellite maps [60, 111, 113].

<table><tr><td rowspan="2">Map</td><td rowspan="2">Training dataset</td><td rowspan="2">Method</td><td rowspan="2">Image encoder</td><td colspan="3">Lateral [m] R@1/3/5↑</td><td colspan="3">Longitudinal [m] R@1/3/5 ↑</td><td colspan="3">Orientation [°]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>R@1/3/5↑</td><td></td></tr><tr><td rowspan="3">Satellite</td><td rowspan="3">KITTI</td><td>SliceMatch [60] CCVPE [111]</td><td rowspan="3">N/A</td><td>24.0</td><td></td><td>72.9 60.5</td><td>7.2</td><td></td><td>33.1</td><td>31.7</td><td>31.7</td><td>31.7</td></tr><tr><td></td><td>23.4</td><td></td><td></td><td>11.8</td><td></td><td>42.1</td><td>3.1</td><td></td><td>14.6</td></tr><tr><td>Loc2 [113]</td><td>13.6</td><td></td><td>50.9</td><td>14.0</td><td></td><td>50.7</td><td>2.5</td><td></td><td>12.8</td></tr><tr><td rowspan="2">OSM</td><td rowspan="2">MGL (12 cities)</td><td>OrienterNet [83]</td><td>ResNet-101</td><td>42.5</td><td>74.8</td><td>83.2</td><td>21.7</td><td>49.8</td><td>61.0</td><td>20.3</td><td>50.9</td><td>65.1</td></tr><tr><td>OSMLoc [62]</td><td>DINOv2-B</td><td>49.9</td><td>82.5</td><td>87.0</td><td>25.7</td><td>56.4</td><td>66.5</td><td>22.4</td><td>56.2</td><td>72.7</td></tr><tr><td rowspan="3">OSM</td><td rowspan="3">MGL (11 cities)</td><td>OrienterNet [83] AutoCompass</td><td></td><td>37.7</td><td>67.3</td><td>77.4</td><td>17.6</td><td>42.6</td><td>53.7</td><td>15.7</td><td>42.1</td><td>57.1</td></tr><tr><td>→ raw GPS</td><td>ResNet-101</td><td>43.1</td><td>78.8</td><td>85.6</td><td>26.8</td><td>56.1</td><td>65.1</td><td>21.1</td><td>53.0</td><td>69.2</td></tr><tr><td>↔ w/ rel. poses</td><td></td><td>56.6</td><td>83.2</td><td>87.8</td><td>33.2</td><td>59.5</td><td>66.0</td><td>28.9</td><td>64.4</td><td>75.9</td></tr><tr><td rowspan="3">OSM</td><td rowspan="3">MGL (11 cities)</td><td>OrienterNet [83] AutoCompass</td><td></td><td>48.7</td><td>86.0</td><td>91.3</td><td>24.7</td><td>62.4</td><td>73.3</td><td>29.8</td><td>70.2</td><td>83.3</td></tr><tr><td>→ raw GPS</td><td>DINOv2-B</td><td>62.3</td><td>88.3</td><td>92.2</td><td>33.7</td><td>66.6</td><td>75.1</td><td>30.5</td><td>69.3</td><td>82.6</td></tr><tr><td>↔→ w/ rel. poses</td><td></td><td>70.8</td><td>91.1</td><td>94.2</td><td>34.1</td><td>70.8</td><td>77.6</td><td>37.9</td><td>78.5</td><td>88.5</td></tr></table>

Table 1: Results on KITTI [40]. We report position and orientation error recalls at diferent thresholds. Best results among methods using OSM are marked bold, and second best are underlined. The current version of the MGL dataset [83] is missing one of the 12 original cities. For a fair comparison, we retrain our baseline, OrienterNet [83], on the reduced training set. When trained on raw 2D GPS labels, AutoCompass outperforms OrienterNet trained on geo-referenced 3-DoF poses. When supervising with relative poses, AutoCompass achieves best results across all OSM methods, even those trained on more data. Using a DINOv2 backbone further improves the performance.

## 4.1 Driving scenarios

Dataset We use the “Test2” split defined by [91] on the KITTI dataset [40], which contains driving sequences in residential and road areas captured by cameras mounted on a car. It provides accurate RTK-corrected, geo-referenced ground-truth, and we use the rasterized OSM tiles provided by OrienterNet [83].

Setup We follow standard protocols [83, 113]: we compute lateral (perpendicular) and longitudinal (parallel) position errors relative to the driving direction, as well as orientation errors, and report recall at $1 / 3 / 5 \mathrm { m } / \mathrm { \Omega } ^ { \circ }$ . These protocols also restrict the search to a 40 40 m area whose center matches that of the tile and that is randomly sampled within 20m of the GT position. Following [113], we assume no orientation prior, while the evaluation in Sec. C follows [83], and restricts orientations to 10° from the GT orientation. See Sec. B for more details.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td rowspan="2">Training MGL version</td><td rowspan="2">Image Encoder</td><td colspan="3">XY [m]</td><td colspan="3">Orientation</td><td colspan="2">XY (seq)</td><td colspan="2">[m]</td><td colspan="2">Orient. (seq) [°]</td></tr><tr><td></td><td>R@1/3/5↑</td><td></td><td></td><td>R@1/3/5↑</td><td></td><td>R@1/3/</td><td></td><td>/5↑</td><td></td><td>R@1/3/5↑</td><td></td></tr><tr><td rowspan="9">Ia] L57]</td><td>GPS OrienterNet [83]</td><td>–</td><td>一 ResNet-101</td><td>6.0 2.7</td><td>37.2 12.6</td><td>65.2 20.1</td><td>- 5.0</td><td>- 14.6</td><td>- 22.1</td><td>9.3 36.5</td><td>49.3 77.2</td><td>74.7 80.2</td><td>16.0 59.5 48.7</td><td>81.5</td><td>77.2 84.5</td></tr><tr><td>OSMLoc-B [62]</td><td>12 cities</td><td>DINOv2-B</td><td>6.7</td><td>26.3</td><td>37.1</td><td>10.1</td><td>27.0</td><td>39.0</td><td>59.6</td><td>93.2</td><td>94.3</td><td>69.2</td><td>92.8</td><td>94.0</td></tr><tr><td>OrienterNet [83]</td><td></td><td></td><td>1.9</td><td>10.9</td><td>18.2</td><td>4.8</td><td>13.4</td><td>20.7</td><td>35.8</td><td>76.4</td><td>84.6</td><td>50.8</td><td>83.8</td><td>89.0</td></tr><tr><td>AutoCompass</td><td>11 cities</td><td>ResNet-101</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>→ raw GPS</td><td></td><td></td><td>3.7</td><td>16.5</td><td>26.3</td><td>7.3</td><td>20.0</td><td>28.7</td><td>47.9</td><td>85.6</td><td>86.4</td><td>61.8</td><td>89.0</td><td>91.7</td></tr><tr><td>↔ w/ rel. poses</td><td></td><td></td><td>8.1</td><td>25.2</td><td>33.0</td><td>10.5</td><td>26.4</td><td>35.5</td><td>72.6</td><td>92.1</td><td>92.6</td><td>77.8</td><td>92.8</td><td>94.0</td></tr><tr><td>OrienterNet [83]</td><td></td><td></td><td>9.1</td><td>32.5</td><td>43.6</td><td>11.4</td><td>30.4</td><td>43.3</td><td>57.5</td><td>92.9</td><td>93.7</td><td>70.8</td><td>94.8</td><td>95.0</td></tr><tr><td>AutoCompass → raw GPS</td><td>11 cities</td><td>DINOv2-B</td><td>10.9</td><td>40.8</td><td>52.3</td><td>14.5</td><td>37.7</td><td></td><td></td><td></td><td>95.6</td><td>82.1</td><td>95.0</td><td>95.0</td></tr><tr><td>↔ w/ rel. poses</td><td></td><td></td><td>20.9</td><td>49.2</td><td>56.3</td><td>19.5</td><td>47.0</td><td>51.8 59.5</td><td>70.5 95.1 86.2</td><td>295.1</td><td>95.3</td><td>87.8</td><td>96.0</td><td>96.0</td></tr><tr><td rowspan="10">[10] Da--Nht Oxoord</td><td>OrienterNet [83]</td><td></td><td>ResNet-101</td><td>10.4</td><td>37.2</td><td>52.2</td><td>15.6</td><td>38.6</td><td></td><td></td><td></td><td></td><td>70.5</td><td></td><td>99.9</td></tr><tr><td>OSMLoc-B [62]</td><td>12 cities</td><td>DINOv2-B</td><td>15.5</td><td>49.9</td><td>63.9</td><td>21.5</td><td></td><td>51.7</td><td>77.5 84.6</td><td>96.1</td><td>99.5 99.7</td><td>78.3</td><td>95.7 99.3</td><td>99.8</td></tr><tr><td>OrienterNet [83]</td><td></td><td></td><td>9.8</td><td>36.3</td><td>50.5</td><td>14.6</td><td>49.9 35.1</td><td>63.1 48.5</td><td>78.6</td><td>98.9 96.2</td><td>99.6</td><td>73.7</td><td>97.0</td><td>99.4</td></tr><tr><td>AutoCompass</td><td>11 cities</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>↔ raw GPS</td><td></td><td>ResNet-101</td><td>10.4</td><td>36.8</td><td>52.2</td><td>17.8</td><td>41.6</td><td>54.4</td><td>76.6</td><td>94.8</td><td>98.4</td><td></td><td>69.5 93.0</td><td>98.4</td></tr><tr><td>→ w/ rel. poses OrienterNet [83]</td><td></td><td></td><td>13.2 16.5</td><td>42.9 53.6</td><td>56.4 66.6</td><td>19.0 20.1</td><td>43.8 48.5</td><td>56.5</td><td>86.8 84.9</td><td>99.4 98.3</td><td>99.4 99.9</td><td>80.6 78.6</td><td>99.7</td><td>99.9</td></tr><tr><td>AutoCompass</td><td>11 cities</td><td></td><td></td><td></td><td></td><td></td><td></td><td>63.2</td><td></td><td></td><td></td><td></td><td>99.0</td><td>100.0</td></tr><tr><td>↔ raw GPS</td><td></td><td>DINOv2-B</td><td>16.8</td><td>55.1</td><td>68.6</td><td>22.9</td><td>52.4</td><td>66.9</td><td>83.2</td><td></td><td>99.0100.0</td><td>73.2 99.0</td><td></td><td>99.4</td></tr><tr><td>↔ w/ rel. poses</td><td></td><td></td><td>20.8 60.7</td><td></td><td>73.5</td><td>24.7</td><td>57.0</td><td>71.7</td><td>90.5</td><td></td><td>99.9100.0</td><td>83.2</td><td>99.2</td><td>99.4</td></tr></table>

Table 2: Results on egocentric sequences [57, 105] We show position and orientation recall at diferent thresholds. Best results in bold across OSM-based methods, second best underlined. Table center shows results based on single frame estimates, right side shows results after sequential fusion of 50 frames. AutoCompass supervised with relative poses and using a DINOv2 backbone performs best across both datasets.

Results As can be seen in Tab. 1, AutoCompass consistently outperforms strongly supervised baselines [62, 83], even when trained with just raw GPS measurements. Supervision with relative poses further improves accuracy, making AutoCompass the best-performing method, including those trained on the full MGL dataset and satellite-to-ground approaches trained on KITTI. Using a stronger DINOv2-based image encoder further improves the performance.

## 4.2 Egocentric sequences

Datasets We use the LaMAria [57] and Oxford Day-and-Night (ODN) [105] datasets, both captured with Aria Glasses [34]. LaMAria provides km-long sequences in Zurich with cm-accurate ground-truth poses obtained by recording survey-grade control points at regular intervals. While LaMAria includes natural motions representative of AR-device use, the motions required to record survey control points are artificial (mostly looking at the ground), which makes the benchmark harder in an unintended way. We manually remove these images to retain only those reflecting normal device usage (details in Sec. B).

ODN’s poses are estimated via multi-sensor fusion, including GPS and visualinertial measurements, and are optimized across multiple SLAM sessions with loop closures. The poses are locally highly accurate (cm-level [105]), but the georeferencing can have meter-level errors. Thus, we compute a robust per-sequence alignment between each method’s estimates and ground-truth (see Sec. B). We uniformly subsample both datasets to 5k evaluation images.

![](images/b0a37033fac92f5a952a6bd00f25c66e0ac722702efb98a2f1dd439592735f84.jpg)  
Fig. 4: Qualitative results. To accommodate inaccuracies in the training GT, a strongly supervised OrienterNet [83] learns smooth map features, and focuses on large regions useful for coarse localization, such as entire buildings (see feature norms). In contrast, AutoCompass learns much sharper features, focusing on keypoints visible in the images, such as building corners, more useful for fine-grained visual localization.

Setup We compute position and orientation error norms and report recall at $1 / 3 / 5 \mathrm { m } / \mathrm { \Omega } ^ { \circ }$ . We render 128 128 m tiles for each datapoint, use no orientation prior and restrict the position search to a 64 64 m area. In LaMAria, we center the tile and search area at the GPS coordinates. In ODN, GPS labels were released only recently, so we randomly sample the center within 32 m of the ground-truth.

Results Tab. 2 shows that AutoCompass generalizes well to urban scenes not seen during training and outperforms alternative approaches, even when they are trained on more data. Qualitative comparisons to OrienterNet are shown in Fig. 4. Using DINOv2 as the image backbone consistently improves over using a ResNet-101 backbone. Notably, on LaMAria, AutoCompass improves over GPS at strict error thresholds while achieving comparable performance at coarse error thresholds. In the following, we show that sequential fusion of single-view estimates yields a clear advantage over GPS in both accuracy and robustness.

![](images/abf5d31144ab13f6827f1d7c9822e7a60f63cbdfd4b60fe6bbac66babc356a7c.jpg)  
Fig. 5: Left: AutoCompass accuracy scales well with sequence length and improves faster than OrienterNet [83]. Right: Weak labels outperform OrienterNet and their performance correlates with label informativeness. Each point corresponds to (a) OrienterNet, (b) GPS labels, (c) GPS labels with non-metric relative poses, (d) metric relative poses, and (e) approximately geo-referenced relative poses. The losses for $\mathrm { ( b ) - ( e ) }$ , respectively, are L<sub>GPS,chunk</sub>, see Eq. (4); $\mathcal { L } _ { \mathrm { G P S , c h u n k } } + 0 . 5 \mathcal { L } _ { \Delta \theta }$ , see Eq. (7); $0 . 1 \mathcal { L } _ { \Delta \Theta } + \mathcal { L } _ { \| \Delta \mathbf { t } \| }$ , see Eq. (14); and $0 . 1 \mathcal { L } _ { \Delta \Theta } + \mathcal { L } _ { \Delta X Y }$ , see Eq. (11). All results use a ResNet-101 backbone, with chunk sizes $r = 2 0$ m in Eq. (4) and $r _ { \varDelta \mathbf { t } } = 5$ m in Eq. (14).

## 4.3 Sequential evaluation

Given relative pose estimates $\hat { \xi } _ { j , i }$ between views $j$ and $\textit { i } ( e . g .$ , from VIO/SLAM [21,34,70]), we can aggregate multi-view information for higher accuracy. Like [83], we compute the posterior $p ( \pmb { \xi } _ { 0 } | \mathbf { I } _ { 0 : t } , \operatorname* { m a p } _ { 0 : t } , \hat { \pmb { \xi } } )$ over the pose $\xi _ { 0 }$ at the first time step, given images ${ \mathbf { I } } _ { 0 : t } ,$ map tiles $\operatorname* { m a p } _ { 0 : t } ,$ and relative poses $\hat { \pmb { \xi } }$ up to t.

Setup We use the same splits as in the previous single-view benchmarks, sample contiguous datapoints in time, and vary the sequence length. We keep the same search radius for the poses as in their single-view evaluations. Recall metrics correspond to error diferences between the ground-truth and the aligned input trajectory with the transform derived from the smoothed estimate of $\xi _ { 0 }$ [83].

Results The results in Tab. 2 and Fig. 5 (Left) confirm that the more accurate single-view estimates achieved with our supervision are beneficial for sequential fusion. On the challenging LaMAria [57], fusing under 10 frames allows us to surpass the high recall@5m of GPS, which benefits less due to its biased measurements (as similarly found in [83] and exemplified in Fig. 7). A retrained OrienterNet on the same data requires over 20 frames to achieve this. This behavior is consistent across diferent egocentric [105] and driving [40] benchmarks.

## 4.4 Analyzing supervision types

As shown in Fig. 5 (Right), diferent types of weak labels can train accurate visual localization models. GPS provides noisy, potentially erroneous location estimates, to which we are robust by using a tolerance radius r (Eq. (4)) of up to 20 m. By comparison, relative poses provide a more accurate training signal, with performance improving as more information becomes available: non-metric relative poses already improve rotation accuracy by supervising the relative rotation, while position accuracy improves when using metric or coarsely geo-referenced relative translations, with the latter yielding the best overall performance.

## 5 Conclusion

We have presented AutoCompass, a novel approach for 3-DoF visual localization on 2D maps. AutoCompass tackles the problem of inaccurate geo-referenced training data via weak supervision, and requires only 2D GPS labels or relative poses, easier to obtain at scale than accurate geo-referenced 3-DoF poses. AutoCompass is robust to several types of geo-referencing errors, including GPS noise and absolute pose ofsets. Experimentally, AutoCompass consistently outperforms strongly supervised methods across driving and egocentric benchmarks.

## Acknowledgements

The authors sincerely thank Zirui Wang for generously providing assistance with the Oxford Day-and-Night benchmark. Javier Tirado-Garín is funded by the scholarship FPU21/04468.

## References

1. GPS Performance. https://www.gps.gov/gps-performance, accessed: 2025-12- 01

2. Mapillary. https://www.mapillary.com, accessed: 2025-11-15

3. OpenSfM. https://github.com/mapillary/OpenSfM, accessed: 2025-11-15

4. Arandjelović, R., Gronat, P., Torii, A., Pajdla, T., Sivic, J.: NetVLAD: CNN architecture for weakly supervised place recognition. In: CVPR (2016)

5. Arandjelović, R., Zisserman, A.: DisLocation: Scalable descriptor distinctiveness for location recognition. In: ACCV (2014)

6. Balntas, V., Li, S., Prisacariu, V.: RelocNet: Continuous Metric Learning Relocalisation using Neural Nets. In: ECCV (2018)

7. Barroso-Laguna, A., Cavallari, T., Prisacariu, V.A., Brachmann, E.: A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features. ICLR (2026)

8. Barroso-Laguna, A., Munukutla, S., Prisacariu, V.A., Brachmann, E.: Matching 2D images in 3D: Metric relative pose from metric correspondences. In: CVPR (2024)

9. Berton, G., Masone, C.: MegaLoc: One Retrieval to Place Them All. In: CVPR Workshops (2025)

10. Berton, G., Masone, C., Caputo, B.: Rethinking Visual Geo-Localization for Large-Scale Applications. In: CVPR (2022)

11. Bian, W., Barroso-Laguna, A., Cavallari, T., Prisacariu, V.A., Brachmann, E.: Scene Coordinate Reconstruction Priors. In: ICCV (2025)

12. Boche, S., Jung, J., Laina, S.B., Leutenegger, S.: OKVIS2-X: Open Keyframe-Based Visual-Inertial SLAM Configurable With Dense Depth or LiDAR, and GNSS. IEEE Transactions on Robotics (2025)

13. Brachmann, E., Cavallari, T., Prisacariu, V.A.: Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses. In: CVPR (2023)

14. Brachmann, E., Humenberger, M., Rother, C., Sattler, T.: On the Limits of Pseudo Ground Truth in Visual Camera Re-Localisation. In: ICCV (2021)

15. Brachmann, E., Krull, A., Nowozin, S., Shotton, J., Michel, F., Gumhold, S., Rother, C.: DSAC - Diferentiable RANSAC for Camera Localization. In: CVPR (2017)

16. Brachmann, E., Rother, C.: Learning Less is More - 6D Camera Localization via 3D Surface Regression. In: CVPR (2018)

17. Brachmann, E., Rother, C.: Visual Camera Re-Localization from RGB and RGB-D Images Using DSAC. IEEE TPAMI (2021)

18. Brahmbhatt, S., Gu, J., Kim, K., Hays, J., Kautz, J.: Geometry-Aware Learning of Maps for Camera Localization. In: CVPR (2018)

19. Bruns, L., Barroso-Laguna, A., Cavallari, T., Monszpart, A., Munukutla, S., Prisacariu, V.A., Brachmann, E.: ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training. In: CVPR (2025)

20. Camiletto, A.B., Bochicchio, A., Liniger, A., Dai, D., Gawel, A.: U-BEV: Heightaware Bird’s-Eye-View Segmentation and Neural Map-based Relocalization. In: IROS (2024)

21. Campos, C., Elvira, R., Rodríguez, J.J.G., M. Montiel, J.M., D. Tardós, J.: ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM. IEEE Transactions on Robotics (2021)

22. Cavallari, T., Golodetz, S., Lord, N.A., Valentin, J., Di Stefano, L., Torr, P.H.S.: On-the-fly adaptation of regression forests for online camera relocalisation. In: CVPR (July 2017)

23. Chen, C., Wang, R., Vogel, C., Pollefeys, M.: F3Loc: Fusion and filtering for floorplan localization. In: CVPR (2024)

24. Chen, S., Bhalgat, Y., Li, X., Bian, J.W., Li, K., Wang, Z., Prisacariu, V.A.: Neural Refinement for Absolute Pose Regression with Feature Synthesis. In: CVPR (2024)

25. Chen, S., Cavallari, T., Prisacariu, V.A., Brachmann, E.: Map-Relative Pose Regression for Visual Re-Localization. In: CVPR (2024)

26. Chen, S., Li, X., Wang, Z., Prisacariu, V.: DFNet: Enhance Absolute Pose Regression with Direct Feature Matching. In: ECCV (2022)

27. Davison, A.J., Reid, I.D., Molton, N.D., Stasse, O.: Monoslam: Real-time single camera slam. TPAMI (2007)

28. Deng, T., Chen, X., Li, Z., Shen, H., Wang, D., Civera, J., Wang, H.: UniPR-3D: Towards Universal Visual Place Recognition with Visual Geometry Grounded Transformer. arXiv (2025)

29. Deng, T., Wu, W., Wu, K., Wang, G., Zhu, S., Yuan, S., Chen, X., Shen, G., Liu, Z., Wang, H.: Reloc-VGGT: Visual Re-localization with Geometry Grounded Transformer. arXiv (2025)

30. Dong, S., Wang, S., Liu, S., Cai, L., Fan, Q., Kannala, J., Yang, Y.: Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. In: CVPR (2025)

31. Edstedt, J., Nordström, D., Zhang, Y., Bökman, G., Astermark, J., Larsson, V., Heyden, A., Kahl, F., Wadenbäck, M., Felsberg, M.: RoMa v2: Harder Better Faster Denser Feature Matching. arXiv (2025)

32. Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., Felsberg, M.: RoMa: Robust Dense Feature Matching. In: CVPR (2024)

33. Engel, J., Koltun, V., Cremers, D.: DSO: Direct sparse odometry. In: IEEE TPAMI (2017)

34. Engel, J., Somasundaram, K., Goesele, M., Sun, A., Gamino, A., Turner, A., Talattof, A., Yuan, A., Souti, B., Meredith, B., et al.: Project Aria: A New Tool for Egocentric Multi-Modal AI Research. arXiv (2023)

35. Fervers, F., Bullinger, S., Bodensteiner, C., Arens, M., Stiefelhagen, R.: C-BEV: Contrastive Bird’s Eye View Training for Cross-View Image Retrieval and 3-DoF Pose Estimation. arXiv (2023)

36. Fervers, F., Bullinger, S., Bodensteiner, C., Arens, M., Stiefelhagen, R.: Uncertainty-Aware Vision-Based Metric Cross-View Geolocalization. In: CVPR (2023)

37. Fervers, F., Bullinger, S., Bodensteiner, C., Arens, M., Stiefelhagen, R.: Statewide visual geolocalization in the wild. In: ECCV (2024)

38. Fontan, A., Fischer, T., Civera, J., Milford, M.: VSLAM-LAB: A Comprehensive Framework for Visual SLAM Methods and Datasets. In: IROS (2025)

39. Gao, L., Sun, H., Liu, L., Li, Y., Cai, Y.: DifVL: Difusion-Based Visual Localization on 2D Maps via BEV-Conditioned GPS Denoising. arXiv (2025)

40. Geiger, A., Lenz, P., Urtasun, R.: Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In: CVPR (2012)

41. Gordo, A., Almazán, J., Revaud, J., Larlus, D.: Deep image retrieval: Learning global representations for image search. In: ECCV (2016)

42. Grader, Y., Averbuch-Elor, H.: Supercharging Floorplan Localization with Semantic Rays. In: ICCV (2025)

43. He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. In: CVPR (2016)

44. Ho, C., Zou, J., Alama, O., Kumar, S.M.J., Chiang, B., Gupta, T., Wang, C., Keetha, N., Sycara, K., Scherer, S.: Map It Anywhere (MIA): Empowering Bird’s Eye View Mapping using Large-scale Public Data. In: NeurIPS (2024)

45. Howard-Jenkins, H., Prisacariu, V.A.: Lalaloc++: Global floor plan comprehension for layout localisation in unvisited environments. In: ECCV (2022)

46. Howard-Jenkins, H., Ruiz-Sarmiento, J.R., Prisacariu, V.A.: Lalaloc: Latent layout localisation in dynamic, unvisited environments. In: ICCV (2021)

47. Huang, K.W., Li, B., Hariharan, B., Snavely, N.: C3Po: Cross-View Cross-Modality Correspondence by Pointmap Prediction. In: NeurIPS Datasets and Benchmarks Track (2025)

48. Huang, T., Peng, L., Vidal, R., Liu, Y.H.: Scalable 3D Registration via Truncated Entry-wise Absolute Residuals. In: CVPR (2024)

49. Humenberger, M., Cabon, Y., Guerin, N., Morat, J., Revaud, J., Rerole, P., Pion, N., de Souza, C., Leroy, V., Csurka, G.: Robust Image Retrieval-based Visual Localization using Kapture (2020)

50. Izquierdo, S., Civera, J.: Close, But Not There: Boosting Geographic Distance Sensitivity in Visual Place Recognition. In: ECCV (2024)

51. Izquierdo, S., Civera, J.: Optimal transport aggregation for visual place recognition. In: CVPR (2024)

52. Jiang, X., Wang, F., Galliani, S., Vogel, C., Pollefeys, M.: R-Score: Revisiting Scene Coordinate Regression for Robust Large-Scale Visual Localization. In: CVPR (2025)

53. Kendall, A., Cipolla, R.: Modelling Uncertainty in Deep Learning for Camera Relocalization. In: ICRA (2016)

54. Kendall, A., Cipolla, R.: Geometric Loss Functions for Camera Pose Regression With Deep Learning. In: CVPR (2017)

55. Kendall, A., Grimes, M., Cipolla, R.: PoseNet: A Convolutional Network for Real-Time 6-DoF Camera Relocalization. In: ICCV (2015)

56. Kingma, D.P., Ba, J.: Adam: A Method for Stochastic Optimization. ICLR (2015)

57. Krishnan, A., Liu, S., Sarlin, P.E., Gentilhomme, O., Caruso, D., Monge, M., Newcombe, R., Engel, J., Pollefeys, M.: Benchmarking Egocentric Visual-Inertial SLAM at City Scale. In: ICCV (2025)

58. Lee, S.H., Civera, J.: Alignment Scores: Robust Metrics for Multiview Pose Accuracy Evaluation. In: ICCVW (2025)

59. Lee, S.H., Civera, J.: What’s wrong with the absolute trajectory error? In: EC-CVW (2025)

60. Lentsch, T., Xia, Z., Caesar, H., Kooij, J.F.: SliceMatch: Geometry-Guided Aggregation for Cross-View Pose Estimation. In: CVPR (2023)

61. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3D with MASt3R. In: ECCV (2024)

62. Liao, Y., Chen, X., Kang, S., Li, J., Dong, Z., Fan, H., Yang, B.: OSMLoc: Single image-based visual localization in openstreetmap with geometric and semantic guidances. arXiv (2024)

63. Lindenberger, P., Sarlin, P.E., Hosang, J., Balice, M., Pollefeys, M., Lynen, S., Trulls, E.: Scaling image geo-localization to continent level. NeurIPS (2025)

64. Liu, C., Jiao, J., Huang, H., Ma, Z., Kanoulas, D., Braud, T.: AIR-HLoc: Adaptive Retrieved Images Selection for Eficient Visual Localisation. In: ICRA (2025)

65. Lynen, S., Zeisl, B., Aiger, D., Bosse, M., Hesch, J., Pollefeys, M., Siegwart, R., Sattler, T.: Large-scale, real-time visual-inertial localization revisited. IJRR (2019)

66. Min, Z., Khosravan, N., Bessinger, Z., Narayana, M., Kang, S.B., Dunn, E., Boyadzhiev, I.: LASER: LAtent SpacE Rendering for 2D Visual Localization. In: CVPR (2022)

67. Moreau, A., Piasco, N., Tsishkou, D., Stanciulescu, B., de La Fortelle, A.: LENS: Localization enhanced by NeRF synthesis. In: CoRL (2021)

68. Morlana, J., Montiel, J.: Reuse Your Features: Unifying Retrieval and Feature-Metric Alignment. In: ICRA (2023)

69. Mur-Artal, R., Montiel, J.M.M., Tardos, J.D.: ORB-SLAM: A Versatile and Accurate Monocular SLAM System. IEEE Transactions on Robotics (2015)

70. Mur-Artal, R., Tardós, J.D.: ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras. IEEE Transactions on Robotics (2017)

71. Naseer, T., Burgard, W.: Deep Regression for Monocular Camera-Based 6-dof Global Localization in Outdoor Environments. In: IROS (2017)

72. Olson, E.: AprilTag: A Robust and Flexible Visual Fiducial System. In: ICRA (2011)

73. OpenStreetMap contributors: OpenStreetMap. https://www.openstreetmap. org (2017)

74. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning Robust Visual Features without Supervision. TMLR (2024)

75. Pan, L., Barath, D., Pollefeys, M., Schönberger, J.L.: Global Structure-from-Motion Revisited. In: ECCV (2024)

76. Panek, V., Kukelova, Z., Sattler, T.: Meshloc: Mesh-based visual localization. In: ECCV (2022)

77. Peretic, M., Gilabert, R., Carroll, J., Gutierrez, J., Moore, A., Christie, J., Dill, E.T.: Statistical Analysis of GNSS Multipath Errors in Urban Canyons. In: IEEE/ION Position, Location and Navigation Symposium (PLANS) (2025)

78. Raguram, R., Chum, O., Pollefeys, M., Matas, J., Frahm, J.M.: USAC: A Universal Framework for Random Sample Consensus. IEEE TPAMI (2013)

79. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision Transformers for Dense Prediction. In: CVPR (2021)

80. Salas, M., Latif, Y., Reid, I.D., Montiel, J.: Trajectory Alignment and Evaluation in SLAM: Horn’s Method vs Alignment on the Manifold. In: RSS Workshops (2015)

81. Sarlin, P.E., Cadena, C., Siegwart, R., Dymczyk, M.: From Coarse to Fine: Robust Hierarchical Localization at Large Scale. In: CVPR (2019)

82. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: SuperGlue: Learning feature matching with graph neural networks. In: CVPR (2020)

83. Sarlin, P.E., DeTone, D., Yang, T.Y., Avetisyan, A., Straub, J., Malisiewicz, T., Bulo, S.R., Newcombe, R., Kontschieder, P., Balntas, V.: OrienterNet: Visual Localization in 2D Public Maps with Neural Matching. In: CVPR (2023)

84. Sarlin, P.E., Trulls, E., Pollefeys, M., Hosang, J., Lynen, S.: SNAP: Self-supervised neural maps for visual positioning and semantic understanding. NeurIPS (2023)

85. Sarlin, P.E., Unagar, A., Larsson, M., Germain, H., Toft, C., Larsson, V., Pollefeys, M., Lepetit, V., Hammarstrand, L., Kahl, F., Sattler, T.: Back to the Feature: Learning Robust Camera Localization from Pixels to Pose. In: CVPR (2021)

86. Sattler, T., Leibe, B., Kobbelt, L.: Improving image-based localization by active correspondence search. In: ECCV (2012)

87. Sattler, T., Leibe, B., Kobbelt, L.: Eficient & Efective Prioritized Matching for Large-Scale Image-Based Localization. In: IEEE TPAMI (2016)

88. Schönberger, J.L., Frahm, J.M.: Structure-from-Motion Revisited. In: CVPR (2016)

89. Shavit, Y., Ferens, R., Keller, Y.: Learning Multi-Scene Absolute Pose Regression with Transformers. In: ICCV (2021)

90. Shavit, Y., Keller, Y.: Camera Pose Auto-Encoders for Improving Pose Regression. In: ECCV (2022)

91. Shi, Y., Li, H.: Beyond Cross-View Image Retrieval: Highly Accurate Vehicle Localization Using Satellite Image. In: CVPR (2022)

92. Shi, Y., Li, H., Perincherry, A., Vora, A.: Weakly-supervised camera localization by ground-to-satellite image registration. In: ECCV (2024)

93. Shi, Y., Wu, F., Perincherry, A., Vora, A., Li, H.: Boosting 3-dof ground-tosatellite camera localization accuracy via geometry-guided cross-view transformer. In: ICCV (2023)

94. Shotton, J., Glocker, B., Zach, C., Izadi, S., Criminisi, A., Fitzgibbon, A.: Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images. In: CVPR (2013)

95. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: ICLR (2015)

96. Snavely, N., Seitz, S.M., Szeliski, R.: Photo tourism: Exploring photo collections in 3d. In: ACM SIGGRAPH (2006)

97. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A Benchmark for the Evaluation of RGB-D SLAM Systems. In: IROS (2012)

98. Tong, S., Xia, Z., Alahi, A., He, X., Shi, Y.: Geodistill: Geometry-guided selfdistillation for weakly supervised cross-view localization. In: ICCV (2025)

99. Torii, A., Arandjelovic, R., Sivic, J., Okutomi, M., Pajdla, T.: 24/7 Place Recognition by View Synthesis. In: CVPR (2015)

100. Triggs, B., McLauchlan, P.F., Hartley, R.I., Fitzgibbon, A.W.: Bundle adjustment – a modern synthesis. In: Vision Algorithms: Theory and Practice (2000)

101. Ullman, S.: The interpretation of structure from motion. Proceedings of the Royal Society of London. Series B. Biological Sciences (1979)

102. Wang, F., Jiang, X., Galliani, S., Vogel, C., Pollefeys, M.: GLACE: Global Local Accelerated Coordinate Encoding. In: CVPR (2024)

103. Wang, S., Laskar, Z., Melekhov, I., Li, X., Zhao, Y., Tolias, G., Kannala, J.: HSC-Net++: Hierarchical Scene Coordinate Classification and Regression for Visual Localization with Transformer. IJCV (2024)

104. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: DUSt3R: Geometric 3D vision made easy. In: CVPR (2024)

105. Wang, Z., Bian, W., Li, X., Tao, Y., Wang, J., Fallon, M., Prisacariu, V.A.: Seeing in the Dark: Benchmarking Egocentric 3D Vision with the Oxford Day-and-Night Dataset. In: NeurIPS (2025)

106. Weyand, T., Kostrikov, I., Phiblin, J.: Planet - photo geolocation with convolutional neural networks. In: ECCV (2016)

107. Williams, S.D., Bock, Y., Fang, P., Jamason, P., Nikolaidis, R.M., Prawirodirdjo, L., Miller, M., Johnson, D.J.: Error Analysis of Continuous GPS Position Time Series. Journal of Geophysical Research: Solid Earth (2004)

108. Wu, H., Zhang, Z., Lin, S., Mu, X., Zhao, Q., Yang, M., Qin, T.: Maplocnet: Coarse-to-Fine Feature Registration for Visual Re-Localization in Navigation Maps. In: IROS (2024)

109. Wu, J., Ma, L., Hu, X.: Delving Deeper into Convolutional Neural Networks for Camera Relocalization. In: ICRA (2017)

110. Xia, Z., Alahi, A.: FG<sup>2</sup>: Fine-Grained Cross-View Localization by Fine-Grained Feature Matching. In: CVPR (2025)

111. Xia, Z., Booij, O., Kooij, J.F.P.: Convolutional Cross-View Pose Estimation. IEEE TPAMI (2024)

112. Xia, Z., Shi, Y., Li, H., FP Kooij, J.: Adapting fine-grained cross-view localization to areas without fine ground truth. In: ECCV (2024)

113. Xia, Z., Xu, C., Alahi, A.: Loc<sup>2</sup>: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching. In: ICLR (2026)

114. Yang, H., Shi, J., Carlone, L.: TEASER: Fast and Certifiable Point Cloud Registration. T-RO (2020)

115. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In: CVPR (2024)

116. Zamir, A.R., Shah, M.: Accurate image localization based on google maps street view. In: ECCV (2010)

117. Zhou, Z., Qi, Z., Cheng, L., Xiong, G.: SegLocNet: Multimodal Localization Network for Autonomous Driving via Bird’s-Eye-View Segmentation. arXiv (2025)

118. Zuñiga-Noël, D., Jaenal, A., Gomez-Ojeda, R., Gonzalez-Jimenez, J.: The UMA-VI Dataset: Visual–Inertial Odometry in Low-Textured and Dynamic Illumination Environments. IJRR (2020)

## Supplementary Material

Secs. A and B provide additional training and evaluation details, respectively, and Secs. C and D provide additional quantitative and qualitative results.

## A Additional training details

Tile sampling The Mapillary Geo-Localization (MGL) dataset [83], used for training, provides two types of geo-referenced position coordinates for each input image: (1) the raw GPS coordinates, and (2) refined coordinates obtained through SfM by fusing multi-view visual observations with the raw GPS measurements. During training, we use the raw GPS coordinates to center 128 128 m OSM tiles. Following the oficial OrienterNet checkpoint [83], we randomly perturb each tile center within 48 m and apply random flips and rotations in 90° increments to reduce overfitting, augmenting images and poses accordingly. This preprocessing is performed independently for each tile.

Relative poses For relative-pose supervision, we sample pairs of data points that were jointly optimized within the same SfM cluster. We identify these clusters using the merge\_cc attribute in the Mapillary metadata<sup>2</sup>. Since the two input tiles in a pair are augmented independently, we undo the corresponding flip and rotation augmentations for each predicted probability volume before computing the relative-pose probability distributions (Sec. 3.2). This ensures that the resulting joint probabilities are defined in a consistent, non-augmented coordinate frame. Examples of relative-pose probability distributions supervised during training are shown in Fig. 6.

Images Following [83], we apply color jitter augmentation to the input images and resize them to 512 512 when using a ResNet [43] image backbone. Most images in MGL are square perspective crops from panoramic images with a 90° field of view (FoV), resulting in an average focal length of approximately 256 pixels. When using a DINOv2 [74] backbone, we follow the same procedure but resize images to 518 518 so that their dimensions are divisible by the patch size used in DINOv2.

Architecture and hyperparameters Following [83], we use N = 64 discrete rotations during training and keep the same architecture design, including a 64.5 32 m BEV extent with a resolution of 0.5 m/pixel for both the BEV and the map tile. For our checkpoints, as well as our retrained OrienterNet baselines using a ViT backbone pretrained by DINOv2, we fine-tune this backbone with a learning rate of $1 0 ^ { - 7 }$ . We combine it with a DPT decoder [79], trained from scratch with a learning rate of $1 0 ^ { - 5 }$ . We use gradient clipping with a maximum norm of 1 and keep the same U-Net with VGG-19 map encoder architecture and learning rate, $1 0 ^ { - 4 }$ , as in our other checkpoints.

![](images/d8b70009860256d2a94fe24359379255c14b7cf61034a9ac621b90f8176925d7.jpg)  
Fig. 6: Probability distributions. We show a pair of training datapoints together with their predicted absolute translation $( P _ { i } ^ { X Y } [ u , v ]$ , Eq. (8)) and rotation $( P _ { i } ^ { \Theta } [ k ]$ Eq. (5)) distributions. The last row shows the relative pose distributions derived from the pair: the relative-distance distribution $P _ { \| \varDelta \mathbf { t } \| } [ d ]$ (Eq. (13)), its chunked version, the relative-shift distribution $P _ { \Delta X Y } [ \varDelta u , \varDelta v ]$ (Eq. (10)), and the relative-rotation distribution $P _ { \Delta \Theta } [ \varDelta k ] \ ( \mathrm { E q . \ ( 6 ) } )$ ). Note that the raw distance distribution is non-smooth, which makes interpolation dificult. We therefore use its chunked version in Eq. (14), which also adds robustness to errors in the ground-truth (GT) relative distances. For each distribution, we mark in red the values corresponding to the GT poses.

## B Additional evaluation details

General During evaluation, OrienterNet [83], OSMLoc [62], and AutoCompass use N = 256 discrete rotations. To match the image resolution and perspective distortion seen during training, we resize images so that their focal length is 256 pixels. We also crop LaMAria [57] and Oxford Day-and-Night (ODN) [105] images to 512 512 pixels, or to 518 518 pixels when using DINOv2.

KITTI KITTI images have an aspect ratio of  3.3:1 (W:H), which is out of distribution with respect to the 1:1 aspect ratio seen during training. For this reason, and for DINOv2-based models, after resizing images so that their focal length is 256 pixels, we center-pad them with zeros to the image size seen during training (518 518 pixels). ResNet-based models benefit less from this padding, so for them we only resize images so that the focal length is 256 pixels. We believe that the stronger sensitivity of DINOv2-based models to aspect-ratio changes is due to the learned absolute positional encodings in DINOv2 [74], which can lead to worse generalization across aspect ratios not seen during training.

![](images/abfbd17856c4f96af08a8ac130a3c67845c2e00e9e6df77cce9b4f30c5714060.jpg)  
Fig. 7: Evaluation on LaMAria [57]. We exclude clips recorded while circling around ground control points (see ground-truth poses indicated with red arrows ), whose locations correspond to that of the blurred AprilTags [72] appearing in the images. The LaMAria benchmark also provides GPS measurements (shown as black dots, •), which, as can be seen, are biased with consistent errors over time.

LaMAria We use the eleven LaMAria training sequences for which pseudodense GT is available and more than one control point is observed for georeferencing, i.e. with #CPs> 1. We use this pseudo-dense GT for evaluation. As mentioned in the main paper (Sec. 4.2), we manually filter these sequences to exclude clips corresponding to the capture of ground control points. To do so, we identify the first and last frames of each surveying clip and exclude all images in between. Fig. 7 shows an example of this filtering for one full sequence.

LaMAria’s pseudo-dense GT poses are expressed with respect to the LV95 projected coordinate system, which uses grid north, whereas the ENU topocentric coordinate system we use, following OrienterNet [83], uses geodetic north. To account for this diference, we ofset the heading angles by the meridian convergence at the center of our topocentric system, which is approximately 0.8°.

Oxford Day-and-Night We use the outdoor segments of the daytime sequences from four scenes<sup>3</sup>: bodleian-library, hb-allen-centre, keble-college, and observatory-quarter. As mentioned in the main paper (Sec. 4.2), before computing the error metrics, we align the ground-truth poses to the pose estimates of each method. This is motivated by the high local accuracy of the dataset, but the meter-level errors present in the geo-referenced poses (examples in Fig. 8). This procedure is akin to the standard evaluation protocol used in SLAM benchmarks [38, 80, 97, 118], which also require the alignment between ground-truth and predictions. The main diference is that, in our case, the alignment must be robust, since estimates are obtained from single-view observations and thus can contain outliers/wrong estimates that contaminate the alignment [58, 59].

Ground-truth Aligned ground-truth  
![](images/49bcc220e5447ffead3fab2418ecb64effd1161c0fd55fd40cb2c43db402a2fc.jpg)  
Fig. 8: Ground-truth alignment in Oxford Day & Night. We perform a persequence, per-model alignment between the ground truth and the model estimates. This alignment is computed robustly (via Eqs. (15) and (16)) to account for potential outliers in the model estimates. As shown in these examples, the resulting alignments yield more accurate reference poses with respect to which error metrics are computed.

Thus, we robustly compute, independently for each model and sequence of length $L ,$ a 3-DoF transformation $( \mathbf { R } \in \mathrm { S O } ( 2 ) , \mathbf { t } \in \mathbb { R } ^ { 2 } )$ to align the positions and a scalar ofset $\delta \in [ 0 , 2 \pi )$ to align the heading angles. The 3-DoF transformation is obtained by solving the following truncated least-squares problem:

$$
\operatorname* { m i n } _ { \mathbf { R } \in \mathrm { S O } ( 2 ) , \mathbf { t } \in \mathbb { R } ^ { 2 } } \sum _ { i = 1 } ^ { L } \operatorname* { m i n } \left( \left\| \mathbf { b } _ { i } - \mathbf { R a } _ { i } - \mathbf { t } \right\| ^ { 2 } , \bar { c } ^ { 2 } \right) \ ,\tag{15}
$$

where $\mathbf { a } _ { i } , \mathbf { b } _ { i } \in \mathbb { R } ^ { 2 }$ denote the ground-truth and estimated positions, respectively. We set the outlier threshold to $\bar { c } = 2$ m and solve this problem using TEASER++ [114]. Similarly, the ofset for the headings is obtained by solving the following truncated least-absolute-deviations problem:

$$
\operatorname* { m i n } _ { \delta \in \mathbb { R } } \sum _ { i = 1 } ^ { L } \operatorname* { m i n } \left( \left| \operatorname { w r a p } ( \beta _ { i } - ( \alpha _ { i } + \delta ) ) \right| , \bar { c } \right) .\tag{16}
$$

where $\alpha _ { i } , \beta _ { i }$ denote the ground-truth and estimated heading angles, respectively, and wrap( ) computes the angular error by mapping diferences to $( - \pi , \pi ]$ . We solve this problem using an implementation of the technique presented in TEAR [48], with the outlier threshold set to $\bar { c } = 3 ^ { \circ }$

## C Additional quantitative results

Supervision impact We expand the results over supervision types presented in Sec. 4.4 and in the right subfigure of Fig. 5 by providing separate results in KITTI (Tab. 4) and LaMAria and Oxford Day-and-Night (ODN) (Tab. 5). The notation and details used for our checkpoints are explained in Tab. 3. We also report sequential results, using a sequence length of 100 frames in KITTI and 50 frames in LaMAria and ODN. For reference, we also add metrics for the corresponding retrained version of OrienterNet with strong supervision [83].

<table><tr><td>Alias</td><td>Loss function</td><td>Equations</td><td>Details</td></tr><tr><td> $\mathrm { C h u n k } _ { r = a }$ </td><td> $\mathcal { L } _ { \mathrm { G P S , \ c h u n k } }$ </td><td> $\operatorname { E q . 4 }$ </td><td>chunk half-size r = ±a m</td></tr><tr><td> $\mathrm { C h u n k } _ { r = a } + \varDelta \theta$ </td><td> $\mathcal { L } _ { \mathrm { G P S , \ c h u n k } } + 0 . 5 \mathcal { L } _ { \Delta \theta }$ </td><td>Eqs. 4, 7</td><td>chunk half-size r = ±a m</td></tr><tr><td> $\Delta \Theta + \| \Delta \mathbf { t } \| _ { r _ { \Delta \mathbf { t } } = b }$ </td><td> $0 . 1 \mathcal { L } _ { \Delta \theta } + \mathcal { L } _ { | | \Delta \mathbf { t } | | }$ </td><td>Eqs. 7, 14</td><td>binning half-width  $r _ { \varDelta \mathbf { t } } = \pm b \mathbf { m }$ </td></tr><tr><td> $\varDelta \varTheta + \varDelta X Y$ </td><td> $0 . 1 \mathcal { L } _ { \Delta \Theta } + \mathcal { L } _ { \Delta X Y }$ </td><td>Eqs. 7, 11</td><td></td></tr><tr><td> ${ \mathrm { C h u n k } } _ { r = a } + \varDelta \theta + \varDelta X Y$ </td><td> $\mathcal { L } _ { \mathrm { G P S , ~ c h u n k } } + 0 . 5 \mathcal { L } _ { \Delta \theta } + \mathcal { L } _ { \Delta X Y }$ </td><td> $\mathrm { E q s . ~ 4 , 7 , 1 1 }$ </td><td>chunk half-size  $r = \pm a \mathrm { m }$ </td></tr></table>

Table 3: AutoCompass aliases used in the supplementary. These checkpoints are repesentative of a practical setup where only GPS labels are available $\left( \operatorname { C h u n k } _ { r = a } \right)$ and when diferent types of relative-pose labels are also available: nonmetric $( \mathrm { C h u n k } _ { r = a } ~ + ~ \Delta \theta )$ , metric $( \varDelta \theta + \| \varDelta \mathbf { t } \| _ { r _ { \varDelta \mathbf { t } } = b } )$ and coarsely geo-referenced $( \varDelta \theta + \varDelta X Y )$

<table><tr><td>Method</td><td colspan="3">Lateral [m]</td><td colspan="3">Longitudinal [m]</td><td colspan="3">Orientation [°]</td><td colspan="3">Lat. (seq) [m]</td><td colspan="3">Long. (seq) [m]</td><td colspan="3">Orient. (seq) [°]</td></tr><tr><td></td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td></tr><tr><td>OrienterNet [83]</td><td>37.7</td><td>67.3</td><td>77.4</td><td>17.6</td><td>42.6</td><td>53.7</td><td>15.7</td><td>42.1</td><td>57.1</td><td>75.2</td><td>95.3</td><td>98.5</td><td>47.9</td><td>92.9</td><td>95.1</td><td>70.8</td><td>93.8</td><td>98.7</td></tr><tr><td>AutoCompass</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>→ Chunkr=20</td><td>46.8</td><td>77.9</td><td>84.9</td><td>21.6</td><td>49.8</td><td>61.4</td><td>20.9</td><td>53.6</td><td>68.6</td><td>79.6</td><td>96.3</td><td>98.7</td><td>52.5</td><td>91.5</td><td>92.0</td><td>82.7</td><td>94.6</td><td>100.0</td></tr><tr><td>↔ Chunkr=20 + ∆θ</td><td>47.8</td><td>79.0</td><td>85.6</td><td>21.0</td><td>47.5</td><td>58.3</td><td>26.3</td><td>60.1</td><td>71.1</td><td>81.1</td><td>95.8</td><td>98.6</td><td>46.4</td><td>84.8</td><td>89.4</td><td>80.5</td><td>95.6</td><td>100.0</td></tr><tr><td>↔∆θ+∥|∆t||rst=5</td><td>55.3</td><td>82.9</td><td>87.6</td><td>29.3</td><td>56.0</td><td>63.4</td><td>25.5</td><td>62.9</td><td>75.8</td><td>84.6</td><td>97.2</td><td>98.7</td><td>60.6</td><td>91.5</td><td>92.0</td><td>84.3</td><td>96.9</td><td>98.7</td></tr><tr><td>→ ∆θ + ∆XY</td><td>56.6</td><td>83.2</td><td>87.8</td><td>33.2</td><td>59.5</td><td>66.0</td><td>28.9</td><td>64.4</td><td>75.9</td><td>86.4</td><td>96.5</td><td>98.4</td><td>76.8</td><td>92.9</td><td>96.0</td><td>90.7</td><td>96.9</td><td>99.6</td></tr></table>

Table 4: Results per supervision type on KITTI [40] without orientation priors. All methods use a ResNet-101 image backbone and are trained on the currently available sequences of MGL [83] (11 cities). The notation used and motivation for AutoCompass checkpoints is explained in Tab. 3.

Conclusions remain consistent with the average results: performance correlates with the informativeness of ground-truth labels, specially on KITTI and LaMAria (performance is similar across checkpoints on ODN, presumably due to slight canceling of errors when performing the alignment). The weak chunk marginalization loss is more appropriate than the standard strong supervision, and it is further outperformed when relative pose labels are available.

KITTI evaluation with $\mathbf { a } ~ \pm 1 0 ^ { \circ }$ orientation prior Tab. 6 reports results under the alternative standard protocol [83, 91], which restricts the pose search space to 20 m and 10° around the ground-truth pose. As in Sec. 4.1, we restrict the position search to a 40 40 m area centered on the OSM tile, whose center is randomly sampled within 20 m of the ground-truth position. The conclusions are very similar to those obtained without an orientation prior: AutoCompass’ weak supervision with either raw GPS labels or relative poses is also beneficial under this setting. Note that satellite-based baselines [92,93,110,113] are trained with this strong orientation prior, which can bias networks toward predicting orientations within this range.

<table><tr><td></td><td>Method</td><td colspan="3">XY [m]</td><td colspan="3">Orientation [°]</td><td colspan="3">XY (seq) [m]</td><td colspan="3">Orient. (seq) [°]</td></tr><tr><td></td><td></td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td><td colspan="3">R@1/3/5↑</td></tr><tr><td rowspan="6">I[a] L5]</td><td>OrienterNet [83] AutoCompass</td><td>1.9</td><td>10.9</td><td>18.2</td><td>4.8</td><td>13.4</td><td>20.7</td><td>35.8</td><td>76.4</td><td>84.6</td><td>50.8</td><td>83.8</td><td>89.0</td></tr><tr><td>↔ Chunkr=20</td><td>3.7</td><td>16.5</td><td>26.3</td><td>7.3</td><td>20.0</td><td>28.7</td><td>47.9</td><td>85.6</td><td>86.4</td><td>61.8</td><td>89.0</td><td>91.7</td></tr><tr><td></td><td>2.9</td><td>17.6</td><td>27.3</td><td>8.7</td><td></td><td>31.9</td><td>41.4</td><td></td><td>89.9</td><td>58.4</td><td></td><td>92.0</td></tr><tr><td> $ \mathrm { C h u n k } _ { r = 2 0 } + \varDelta \theta$ </td><td>5.3</td><td></td><td></td><td></td><td>22.5</td><td></td><td></td><td>85.1</td><td></td><td></td><td>87.0</td><td></td></tr><tr><td> $\mathrm { { } } \hookrightarrow \varDelta \theta + \| \varDelta \mathbf { t } \| _ { r _ { \varDelta \mathbf { t } } = 5 }$  ↔ ∆θ + ∆XY</td><td>8.1</td><td>20.5 25.2</td><td>28.6 33.0</td><td>9.5 10.5</td><td>23.4 26.4</td><td>31.9 35.5</td><td>60.2 72.6</td><td>86.8 92.1</td><td>90.0 92.6</td><td>72.6 77.8</td><td>90.5 92.8</td><td>93.5</td></tr><tr><td></td><td>9.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>94.0 99.4</td></tr><tr><td rowspan="6">[xlfr] 105] Da  iht</td><td>OrienterNet [83] AutoCompass</td><td></td><td>36.3</td><td>50.5</td><td>14.6</td><td>35.1</td><td>48.5</td><td>78.6</td><td>96.2</td><td>99.6</td><td>73.7</td><td>97.0</td><td></td></tr><tr><td></td><td>10.4</td><td>36.8</td><td>52.2</td><td>17.8</td><td></td><td>54.4</td><td></td><td>94.8</td><td>98.4</td><td>69.5</td><td>93.0</td><td>98.4</td></tr><tr><td>↔ Chunkr=20</td><td>9.3</td><td></td><td>51.8</td><td></td><td>41.6</td><td>55.7</td><td>76.6</td><td></td><td>99.3</td><td>65.7</td><td>93.6</td><td></td></tr><tr><td> $ \mathrm { C h u n k } _ { r = 2 0 } + \varDelta \theta$ </td><td>11.5</td><td>35.5</td><td></td><td>19.8</td><td>43.5</td><td></td><td>74.9 81.0</td><td>95.6</td><td></td><td></td><td></td><td>99.0</td></tr><tr><td>↔∆θ+||∆t||r^t=5</td><td></td><td>41.4</td><td>55.9</td><td>20.5</td><td>44.9</td><td>57.3</td><td></td><td>96.1</td><td>98.7</td><td>76.0</td><td>99.2</td><td>99.7</td></tr><tr><td>↔ ∆θ + ∆XY</td><td>13.2</td><td>42.9</td><td>56.4</td><td>19.0</td><td>43.8</td><td>56.5</td><td>86.8</td><td>99.4</td><td>99.4</td><td>80.6</td><td>99.7</td><td>99.9</td></tr></table>

Table 5: Results per supervision type on LaMAria [57] and Oxford Day & Night [105]. All methods use a ResNet-101 image backbone and are trained on the currently available sequences of MGL [83] (11 cities). The notation used and motivation for AutoCompass checkpoints is explained in Tab. 3.

<table><tr><td>Map</td><td>Training dataset</td><td>Method</td><td>Image encoder</td><td colspan="3">Lateral [m] R@1/3/5 ↑</td><td colspan="3">Longitudinal [m] R@1/3/5 ↑</td><td colspan="3">Orientation [°] R@1/3/5 ↑</td></tr><tr><td rowspan="4">Satellite</td><td rowspan="4">KITTI</td><td>Shi et al. [93]</td><td rowspan="4">N/A</td><td>57.7</td><td>86.8</td><td>91.2</td><td>14.2</td><td>34.6</td><td>45.0</td><td>99.0</td><td>100</td><td>100</td></tr><tr><td>Shi et al. [92]</td><td>64.7</td><td>86.2</td><td>=</td><td>11.8</td><td>34.8</td><td>一</td><td>100</td><td>100</td><td>100</td></tr><tr><td>FG2 [110]</td><td>37.9</td><td>1</td><td>85.7</td><td>22.0</td><td></td><td>60.8</td><td>23.0</td><td></td><td>77.8</td></tr><tr><td>Loc2 [113]</td><td>45.3</td><td>1</td><td>92.4</td><td>27.0</td><td>=</td><td>68.3</td><td>26.0</td><td>=</td><td>80.7</td></tr><tr><td rowspan="2">OSM</td><td rowspan="2">MGL (12 cities)</td><td>OrienterNet [83]</td><td>ResNet-101</td><td>50.4</td><td>85.2</td><td>92.7</td><td>24.4</td><td>56.1</td><td>68.0</td><td>29.3</td><td>68.2</td><td>84.5</td></tr><tr><td>OSMLoc [62]</td><td>DINOv2-B</td><td>60.1</td><td>93.0</td><td>96.6</td><td>28.7</td><td>61.9</td><td>72.6</td><td>30.9</td><td>71.9</td><td>89.2</td></tr><tr><td rowspan="2">OSM</td><td rowspan="2">MGL (11 cities)</td><td>OrienterNet [83]</td><td></td><td>48.3</td><td>82.0</td><td>91.4</td><td>20.8</td><td>50.3</td><td>62.6</td><td>25.5</td><td>62.0</td><td>79.8</td></tr><tr><td>AutoCompass → raw GPS</td><td>ResNet-101</td><td>51.5</td><td>88.8</td><td>94.1</td><td>29.6</td><td>61.4</td><td>71.2</td><td>29.7</td><td>69.0</td><td>86.2</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>↔ w/ / rel. poses OrienterNet [83]</td><td></td><td>64.7 53.5</td><td>91.6 92.8</td><td>95.1 97.4</td><td>36.3 26.2</td><td>65.1</td><td>71.3 77.5</td><td>37.8 36.9</td><td>80.6</td><td>93.3</td></tr><tr><td>AutoCompass</td><td></td><td></td><td></td><td></td><td></td><td>65.9</td><td></td><td></td><td>81.1</td><td>93.7</td></tr><tr><td rowspan="2">OSM</td><td rowspan="2">MGL (11 cities)</td><td>↔ raw GPS</td><td>DINOv2-B</td><td>70.9</td><td>95.0</td><td>97.1</td><td>31.2</td><td>65.3</td><td>75.4</td><td>35.3</td><td>81.1</td><td>94.2</td></tr><tr><td>↔ w/ rel. poses</td><td></td><td>76.2</td><td>95.6</td><td>98.2</td><td>35.9</td><td>74.0</td><td>81.1</td><td>43.5</td><td>86.6</td><td>95.8</td></tr></table>

Table 6: Results on KITTI [40] with $\mathbf { a } \pm 1 0 ^ { \circ }$ orientation prior. Following [83,91], we restrict the orientation search space $\mathrm { t o } \pm 1 0 ^ { \circ }$ around the ground-truth orientation. The results follow the same trends as in the evaluation without orientation priors reported in Tab. 1: AutoCompass improves over strongly-supervised baselines [62, 83] even when trained only with raw GPS labels, and supervision with relative poses yields the best performance among OSM-based methods.

Varying the chunk size r for GPS-only supervision Tab. 7 shows the performance obtained by varying the marginalization chunk size r used for GPS supervision with Eq. (4). Directly using GPS (r = 0), without marginalization over a local neighborhood, yields worse localization performance. Performance is similar for $r \in [ 5 , 2 0 ]$ m and thus consistently outperforms strong supervision. Performance decreases only for larger chunks (r = 30 m), arguably due to a weaker training signal.

Varying the norm binning size $r _ { \varDelta \mathbf { t } }$ Tab. 7 also shows that the binning size r<sub>∆t</sub> used for relative distance supervision (Eq. (14)) is not a sensitive parameter, as similar performance is obtained across diferent values of $r _ { \varDelta \mathbf { t } }$

<table><tr><td rowspan="2">Method</td><td colspan="3">XY [m]</td><td colspan="3">Orientation [°]</td></tr><tr><td colspan="3">R@1/3/5↑</td><td colspan="2">R@1/3/5↑</td></tr><tr><td>Varying the chunk-size (r) in Eq. (4)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>↔ Chunkr=0</td><td>5.4</td><td>24.3</td><td>37.1</td><td>13.0</td><td>32.4</td><td>43.3</td></tr><tr><td> $\hookrightarrow \mathrm { C h u n k } _ { r = 5 }$ </td><td>8.4</td><td>32.9</td><td>44.4</td><td>14.9</td><td>37.3</td><td>49.4</td></tr><tr><td> $\hookrightarrow \mathrm { C h u n k } _ { r = 1 0 }$ </td><td>8.1</td><td>33.0</td><td>45.7</td><td>15.6</td><td>38.7</td><td>51.0</td></tr><tr><td> $\hookrightarrow \mathrm { C h u n k } _ { r = 2 0 }$ </td><td>8.2</td><td>31.6</td><td>45.0</td><td>15.3</td><td>38.4</td><td>50.6</td></tr><tr><td> $\mathrm { ~ \hookrightarrow ~ C h u n k } _ { r = 3 0 }$ </td><td>6.2</td><td>25.7</td><td>37.6</td><td>14.1</td><td>35.1</td><td>46.2</td></tr><tr><td>Varying the binning-size (r∆t) in Eq. (14)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { \nabla } \hookrightarrow \Delta \theta + \| \Delta \mathbf { t } \| _ { r _ { \Delta \mathbf { t } } = 1 }$ </td><td>10.5</td><td>35.7</td><td>46.5</td><td>17.5</td><td>41.6</td><td>52.7</td></tr><tr><td> $\mathbf { \Sigma } \hookrightarrow \varDelta \theta + \| \varDelta \mathbf { t } \| _ { r _ { \varDelta \mathbf { t } } = 5 }$ </td><td>11.3</td><td>37.5</td><td>48.2</td><td>18.5</td><td>43.7</td><td>55.0</td></tr><tr><td> $\begin{array} { r } { \sum \Delta \theta + \| \Delta \mathbf { t } \| _ { r _ { \Delta \mathbf { t } } = 1 0 } } \end{array}$ </td><td>9.5</td><td>34.1</td><td>44.9</td><td>17.8</td><td>41.0</td><td>52.1</td></tr><tr><td>GPS + relative pose supervision</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>↔ Chunkr=20</td><td>8.2</td><td>31.6</td><td>45.0</td><td>15.3</td><td>38.4</td><td>50.6</td></tr><tr><td> $\hookrightarrow \mathrm { C h u n k } _ { r = 5 } + \varDelta \theta + \varDelta X Y$ </td><td>11.0</td><td>37.5</td><td>48.1</td><td>19.1</td><td>43.6</td><td>54.2</td></tr><tr><td> $\scriptstyle \hookrightarrow \mathrm { C h u n k } _ { r = 2 0 } + \Delta \theta + \Delta X Y$ </td><td>11.1</td><td>36.9</td><td>47.5</td><td>18.7</td><td>43.9</td><td>54.7</td></tr><tr><td> $\hookrightarrow \varDelta \theta + \varDelta X Y$ </td><td>14.4</td><td>40.7</td><td>50.8</td><td>19.4</td><td>44.9</td><td>56.0</td></tr></table>

Table 7: Additional experiments. We show average recalls across benchmarks for diferent setups: Varying the GPS neighborhood size r in Eq. (4), the norm binning size $r _ { \varDelta \mathbf { t } }$ in Eq. (14), and combining GPS and relative pose supervision. Increasing r up to 5 m improves performance, while r<sub>∆t</sub> has little impact. Finally, relative pose supervision remains more efective than GPS supervision alone.

GPS + relative pose supervision Finally, Tab. 7 also tests the combination of GPS supervision (Eq. (4)) with relative pose supervision (Eqs. (7) and (11)). Adding the latter improves performance over using GPS supervision alone. Interestingly, and contrary to what happens when using only GPS, increasing the size r of the supervised GPS neighborhood yields better performance, achieving a closer, but still worse, accuracy to that obtained when just using relative pose supervision. This suggests that relative pose labels, when available and on their own, are already suitable for training visual localization models.

## D Additional qualitative results

Comparison across methods Figs. 9 to 11 show qualitative results and comparisons with OrienterNet [83] and OSMLoc [62] across LaMAria [57], Oxford Day-and-Night [105] and KITTI [40]. For the visualizations we use our version of AutoCompass trained with relative pose supervision $( \varDelta \theta + \varDelta X Y$ on Tab. 3).

AutoCompass learns sharper map features, where even small alleys and narrow roads can be distinguished. The norm of these features is proportional to the magnitude of the cross-correlation scores, which indicates that AutoCompass learns to focus on distinctive keypoints visible in the input images, such as building corners and road intersections. AutoCompass also learns to ignore uninformative regions for fine-grained localization, such as large bodies of water. This contrasts with strongly supervised baselines [62, 83], which learn smoother feature maps, arguably due to inaccuracies in the ground-truth labels, and instead focus on larger areas that are mostly useful for coarse localization.

Comparison across types of weak supervision Fig. 12 compares results across our diferent proposed weak supervision types, confirming that the qualitative behavior described above is consistent across supervision types.

Images  
Input Map  
Neural Map  
Features Norm  
Probabilities  
![](images/c5e72d730f6da2764deb18598f86f110cae9154ef5a7fbfde02c0d132be4f91b.jpg)  
Fig. 9: Qualitative results on LaMAria [57]. On the map, we show the GT and predicted pose with a red ${ \mathbf { \Psi } } ,$ and black arrow, respectively. For each prediction, we also show its BEV frustum and its error, ∆ξ.

![](images/bfcdc7afdeeba69c047942db71de5f78b939b749f2547e1b7e3f5515b87ca1ea.jpg)  
Fig. 10: Qualitative results on Oxford Day & Night (ODN) [105]. On the map, we show the GT and predicted pose with a red , and black arrow, respectively. For each prediction, we also show its BEV frustum and error, ∆ξ. Since the geo-referenced poses in ODN can be ofset (Fig. 8), and in order to have the same GT with which to compare and compute $\varDelta \xi ,$ for just these examples we precompute an alignment using an independent AutoCompass model, trained with Chunk $\dot { \cdot } r = 2 0 + \Delta \theta + \Delta X Y$

Images  
Input Map  
Neural Map  
Norms  
![](images/689210b56bbbc371f7a23d99623b553200b0fb1abb9ff653a0e30de0ac9cef4b.jpg)  
Probs.  
Fig. 11: Qualitative results on KITTI [40]. On the map, we show the GT and predicted pose with a red , and black arrow, respectively. For each prediction, we also show its BEV frustum and its error, $\varDelta \xi .$

![](images/f7e5ffd158687d7c254ae5b6d8daedbc2fc840e46f533c2d4ba72c7d39da28c3.jpg)  
Fig. 12: Qualitative results across supervision types. We show example BEV and map features, as well as their norms, obtained with our diferent weak supervisions. The behavior is consistent across them: we obtain sharper map features than those of strongly supervised baselines [62,83], becoming even sharper when using relative poses. This suggests that weak labels, either relative poses or just assuming that the pose lies anywhere within a GPS neighborhood, are less noisy than absolute ones.