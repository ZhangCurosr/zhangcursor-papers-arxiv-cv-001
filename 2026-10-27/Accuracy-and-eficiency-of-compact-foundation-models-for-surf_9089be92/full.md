# Accuracy and eficiency of compact foundation models for surface-water mapping during floods

Parastoo Farajpoor<sup>a</sup>, Mohammadreza Narimani<sup>a,∗</sup>

<sup>a</sup>Department ofBiological and Agricultural Engineering, University ofCalifornia, Davis, Davis, CA, 95616, USA

## A R T I C L E I N F O

Keywords:   
surface-water mapping   
flood events   
geospatial foundation models   
Prithvi   
Sentinel-2   
computational eficiency   
geographic transfer   
calibration

## Abstract

Pretrained Earth observation models may reduce the efort needed to map surface water, but their value depends on the alternative against which they are compared. We evaluated Prithvi-EO-2.0-Tiny using the 446 hand-labeled Sentinel-2 chips from 11 flood events in Sen1Floods11. A fine-tuned model, a frozen-encoder model, and an identical randomly initialized architecture were compared with a similarly sized U-Net, a larger Prithvi model, and spectral-index thresholds. Experiments varied the number of training chips, excluded events from model fitting, and measured computational cost and prediction reliability. With 252 training chips, U-Net achieved a water intersection-over-union (IoU) of 0.856 on the oficial test set, compared with 0.833 for fine-tuned Prithvi-Tiny. Pretraining improved the tiny transformer over random initialization by a reported 0.046 IoU with 25 training chips and 0.023 with 252 chips, but did not reverse the U-Net ranking. Across 11 event-exclusion evaluations, mean IoU was 0.768 for fine-tuned Prithvi-Tiny and 0.777 for U-Net; the paired interval included zero. Fine-tuned Prithvi-Tiny processed a 512×512-pixel chip in 16.1 ms on a GPU and 258 ms on a laptop CPU, compared with 30.6 and 665 ms for U-Net. Errors were concentrated near water boundaries, and high predictive entropy identified many, but not all, erroneous pixels. These results support a specific role for compact pretrained models: lower measured computational demand with competitive accuracy, rather than consistent superiority. Event-dependent performance, shared hyperparameter selection, and the distinction between observed water and newly inundated land delimit the conclusions.

## 1. Introduction

Satellite maps made during floods help identify where water is present when field observations are incomplete or dificult to obtain. Their value is not limited to drawing an outline: they inform the interpretation of exposure, the prioritization of further observation, and the assessment of hydrological models. Global studies have documented substantial populations exposed to flooding and an increase in the population observed within large flood extents [1, 2]. For such applications, a useful mapping method must be accurate enough for its purpose, feasible to run, and explicit about what the imagery cannot reveal.

Optical images provide a relatively direct view of exposed surface water. Water commonly has low near-infrared and shortwave-infrared reflectance, which motivates the normalized diference water index (NDWI) and modified NDWI (MNDWI) [3, 4]. These indices are inexpensive and interpretable, but their thresholds are sensitive to scene composition and to surfaces that resemble water spectrally. Learned classifiers can use richer spectral and spatial information [5]. Neither approach removes the physical limitations of optical observations: clouds and vegetation can conceal inundation, and water-land mixtures complicate classification at boundaries [6]. Accordingly, a map of water visible in an image is not automatically a map of all flooding on the ground.

Deep-learning flood mapping has developed around encoder-decoder networks and increasingly diverse benchmark datasets [7]. Sen1Floods11 provides paired observations and reference masks from multiple flood events, including a geographically separate Bolivia evaluation set [8]. Optical flood-segmentation studies have demonstrated the usefulness of this approach, while cross-dataset assessments show that success on one benchmark does not settle transferability to another [9, 10]. A strong within-dataset score can coexist with failures in landscapes, water conditions, or labeling conventions that difer from the training examples.

This concern is not specific to floods. The systematic review of Sentinel-2 crop-yield estimation by Narimani et al. [11] also organizes evidence around heterogeneous targets, input representations, and validation settings. The broader lesson is that a model’s usefulness must be judged against the target and evaluation design actually being studied.

Geospatial foundation models ofer a diferent starting point for learning. Their encoders are pretrained on large image collections, then adapted to a smaller labeled task. Prithvi-EO-2.0 applies this approach to multispectral, multitemporal Earth observation imagery, and TerraTorch provides tools for downstream adaptation [12, 13]. The attraction is practical: a pretrained representation may make better use of a small reference set. However, this proposition contains two separate questions. Does pretraining improve the same network relative to random initialization, and does that pretrained network outperform a strong task-specific alternative? These questions require diferent controls.

A further distinction is whether the encoder itself is run and adapted or whether a classifier uses representations computed in advance. For example, Narimani et al. [14] used AlphaEarth embeddings for crop-system mapping. Here, the Prithvi network operates directly on the supplied multispectral images; frozen and fine-tuned encoders are therefore explicit experimental conditions rather than interchangeable descriptions of an embedding dataset.

Recent evidence argues against treating foundation-model superiority as a given. PANGAEA found that geospatial foundation models do not consistently outperform supervised baselines across tasks and data regimes [15]. Flood-specific work has examined sensor-dependent foundation-model performance and hybrid designs that combine pretrained features with convolutional detail [16, 17]. These studies motivate careful comparisons rather than a claim that another application of Prithvi is inherently novel. For a compact model, the relevant contribution is a controlled account of what is gained, what is lost, and under which evaluation conditions.

Geography is central to that account. Nearby samples often share landscape structure and acquisition conditions, so spatially dependent validation can make an environmental mapping method appear more transferable than it is [18, 19]. The distinction also matters in hazard applications outside flooding: a spatially validated analysis of wildfire structure loss reported a substantial change in apparent performance when evaluation moved beyond random partitions [20]. Event exclusion is useful for exposing such diferences, but its interpretation must include the origin of the hyperparameters as well as the images used for gradient updates.

This study evaluates the compact Prithvi-EO-2.0-Tiny encoder with a segmentation decoder against a similarly sized U-Net, spectral-index baselines, an identical randomly initialized transformer, and a larger Prithvi reference model. We ask whether pretraining improves the compact architecture, whether that improvement translates into an advantage over U-Net with fewer fine-tuning labels, and how accuracy and reliability change when an event is excluded from fitting. We also measure inference on a GPU and a laptop CPU. The central aim is to separate the benefit of pretraining from the choice of architecture and from the cost of running the resulting model.

## 2. Materials and methods

Figure 1 locates the study events, Figure 2 summarizes the workflow, and Figure 3 describes the imagery and reference labels. The two evaluation protocols distinguish unseen chips from events excluded from model fitting.

## 2.1. Flood-event imagery and reference labels

The study used the hand-labeled Sentinel-2 subset of Sen1Floods11 version 1.1 [8]. It contains 446 image chips, each 512×512 pixels, from 11 events observed between 2016 and 2019. Figure 1 shows the locations and the Bolivia evaluation region; Table 1 gives the acquisition dates and sample allocations. The source images contain 13 Sentinel-2 Level-1C bands, stored as integer top-of-atmosphere reflectance scaled by 10,000. They use a geographic WGS 84 grid with a pixel spacing of approximately $\mathrm { \bar { 8 . 9 8 } } \times 1 0 ^ { - 5 }$ degrees. This is about 10 m, but the east-west ground distance varies with latitude; no fixed square-kilometer throughput is inferred from chip counts.

Reference values are 1 for water, 0 for non-water, and -1 for unlabeled pixels. The water class includes permanent water and water observed during the flood event. It does not identify newly inundated land by itself. A co-registered permanent-water layer distributed with the benchmark and derived from the JRC Global Surface Water product was used only to stratify errors [21]. It was not a predictor. Water outside this permanent-water layer is called other reference water below, because absence from that layer is not independent proof of flood-induced change.

The inventory recorded aligned image-mask grids, five chips without valid labels, and 52 chips without reference water. Across the collection, 13.63% of pixels are unlabeled and 10.60% of labeled pixels are water. In 30 chips, part of the image lies outside the Sentinel-2 swath. The 60,454 labeled pixels with no signal in any band in these areas were excluded for every method, rather than treated as optically observable land or water. Figure 3 summarizes the resulting label distribution. Label provenance is important when comparing hazard benchmarks: the active-fire dataset of Mitra et al. [22], for example, uses spectral-rule-derived labels, whereas the present analysis uses the benchmark’s hand-labeled water masks. The two forms of supervision should not be treated as equivalent evidence.

## 2.2. Input representation and preprocessing

The six input bands were blue B2, green B3, red B4, narrow near-infrared B8A, and shortwave-infrared B11 and B12. B8A, rather than the wider B8 band, matches the intended Prithvi spectral correspondence. Digital numbers were clipped to [0, 10,000] and multiplied by $1 0 ^ { - 4 }$ . Images and masks remained on the distributed grid without further resampling. For networks, each band was standardized using the image mean and standard deviation of the relevant training pool; out-of-swath pixels were set to zero after standardization.

This preprocessing does not make the inputs identical to Prithvi’s pretraining data. Harmonized Landsat-Sentinel (HLS) provides surface reflectance on a 30-m grid [23], whereas these inputs are Level-1C top-of-atmosphere observations on the supplied finer grid. In Figure 3E, trainingpool band means difer from the HLS pretraining means by -0.019 to +0.041 reflectance, and their standard deviations are 1.3-3.1 times smaller. These descriptive diferences indicate an input-domain mismatch; they do not isolate its efect on accuracy. Normalization used the full training-image pool even in reduced-label experiments, so those experiments reduce labeled fine-tuning chips rather than the amount of unlabeled imagery available.

## 2.3. Evaluation design and label fractions

Protocol A preserved the benchmark allocation: 252 training chips, 89 validation chips, and 90 test chips from ten events, plus 15 Bolivia chips as a separate event-level evaluation set. The oficial test chips therefore represent events seen during training, even though the chips themselves were not used to fit the models. Bolivia was absent from both fitting and validation. We use seen-event test and Bolivia consistently to distinguish these settings.

![](images/c06fb76b358fff5405de8c4d3aaa43bb65b59e286b8b09a8d3ef21c95c191f42.jpg)  
Figure 1: Geographic coverage and the Bolivia holdout. (A) Locations, image dates, and hand-labeled chip counts for the 11 Sen1Floods11 events; symbol area scales with chip count. Bolivia is the Protocol A event holdout. (B) Bolivia chip locations and the metadata event footprint over Copernicus GLO-90 relief, with Natural Earth boundaries and rivers. (C) Two example chips, their reference-water boundaries, and Copernicus GLO-30 relief; unlabeled pixels are faded. Panel A uses an Equal Earth projection; the regional panels show WGS 84 geographic coordinates. The source cartography uses vertical exaggeration of 3 for regional shading and 12 for local shading, not for quantitative terrain analysis. Footprints describe the supplied event/chip coverage, not independently measured flood extents.

Protocol B excluded each target event from model fitting and evaluated all its available chips. The reported fold sizes in Table 1 show how this was implemented: the target event’s chips were removed from the fixed Protocol A training and validation pools. Bolivia remained outside those pools throughout. Thus, non-Bolivia target folds used nine other non-Bolivia events, not all ten other events in the complete dataset. The Bolivia evaluation reused the Protocol A seed-0 models. We retain the shorthand leave-one-event-out (LOEO)

for correspondence with the figures, but interpret it as eventexcluded fitting within fixed benchmark pools.

There is a further limitation to geographic independence. Learning rates were selected once using the full Protocol A validation pool and then reused in Protocol B. For non-Bolivia folds, this earlier selection could include validation observations from the subsequently excluded event. Protocol B therefore tests exclusion from gradient fitting and foldspecific checkpoint selection, but it is not fully nested, eventblind hyperparameter selection. Bolivia remains the cleaner event-level holdout with respect to this shared tuning step.

Table 1  
Event composition and the two evaluation allocations.
<table><tr><td>Event</td><td>S2 date</td><td>Chips</td><td>A: train / valid / test</td><td>B: train / valid</td><td>Water (%)</td><td>Unlabeled (%)</td><td>Permanent water (%)</td></tr><tr><td>Ghana</td><td>2018-09-19</td><td>53</td><td>31 / 11 / 11</td><td>221/78</td><td>6.3</td><td>33.8</td><td>1.4</td></tr><tr><td>India</td><td>2016-08-12</td><td>68</td><td>40 / 14 / 14</td><td>212 / 75</td><td>12.4</td><td>16</td><td>2.7</td></tr><tr><td>Cambodia</td><td>2018-08-04</td><td>30</td><td>18/6/6</td><td>234/83</td><td>24.1</td><td>9.3</td><td>7.6</td></tr><tr><td>Nigeria</td><td>2018-09-20</td><td>18</td><td>10 / 4 / 4</td><td>242 /85</td><td>23.5</td><td>14.3</td><td>1.9</td></tr><tr><td>Pakistan</td><td>2017-06-28</td><td>28</td><td>16/6/6</td><td>236/ 83</td><td>7.5</td><td>26.7</td><td>2.5</td></tr><tr><td>Paraguay</td><td>2018-10-31</td><td>67</td><td>39 / 14 / 14</td><td>213 / 75</td><td>8.4</td><td>5</td><td>2.9</td></tr><tr><td>Somalia</td><td>2018-05-05</td><td>26</td><td>15/5/6</td><td>237/ 84</td><td>6.5</td><td>21.1</td><td>0</td></tr><tr><td>Spain</td><td>2019-09-18</td><td>30</td><td>18/6/6</td><td>234/83</td><td>13.2</td><td>1.7</td><td>6.3</td></tr><tr><td>Sri Lanka</td><td>2017-05-28</td><td>42</td><td>24/9/9</td><td>228 /80</td><td>11.8</td><td>9.5</td><td>7.9</td></tr><tr><td>USA</td><td>2019-05-22</td><td>69</td><td>41 / 14 / 14</td><td>211 /75</td><td>4.6</td><td>2.4</td><td>1.5</td></tr><tr><td>Bolivia</td><td>2018-02-15</td><td>15</td><td>hold-out</td><td>252/89</td><td>15.9</td><td>27.1</td><td>1.5</td></tr><tr><td>Total</td><td>2016-2019</td><td>446</td><td>252 / 89 / 90</td><td>Varies</td><td>10.60</td><td>13.63</td><td></td></tr></table>

A: oficial benchmark pools, with 15 additional Bolivia holdout chips. B: counts after excluding the target event from the fixed A training/validation pools; Bolivia does not enter those pools. For non-Bolivia targets, nine other non-Bolivia events remain. Water and permanent-water percentages refer to labeled pixels; unlabeled percentages refer to all pixels. Event percentages are the supplied event summaries; the total row is the supplied pooled summary, not an average of event percentages. Permanent-water membership is from the distributed JRC-derived layer.

## AProcessing chain

![](images/a8e780119e5e78989eff779110779d202452bde034cdb1ba6166072ecca42b76.jpg)  
B Prithvi-EO-2.0 segmentation network (TerraTorch EncoderDecoderFactory)

![](images/391fb803d5616c5714c2d62d6e026ac79a444bdc334b7b6817c41dddefa6bb7c.jpg)  
Figure 2: Processing, architectures, and computational workflow. (A) Data preparation, evaluation allocations, methods, and diagnostics The leave-one-event-out shorthand refers to exclusion from the fixed benchmark training/validation pools; global learning rates were selected once under Protocol A and reused. (B) Single-date six-band Prithvi segmentation. Tiny features are taken from blocks 3, 6, 9, and 12 and decoded through a feature pyramid and UPerNet head. The decoder family is shared, but the non-encoder parameter count difers: approximately 2.15 M for Tiny and 5.73 M for the larger model. (C) Local preparation, Colab training, and local analysis. The 106 reported training runs comprise 12 selection runs and 94 evaluation runs.

D Example chips  
![](images/24474eadcf913f4ce4a86e6b91961531a625f57983101ca458541a5e981600a4.jpg)

![](images/588dd5430679ff22bdff85a75f75347420d7f53269161bca8c86bff31029a4b7.jpg)

![](images/06d4294e934aba346b738e0f830682d502fa2c6ea85421ffce0ec175446eff15.jpg)

![](images/dc77542ec26906967a74498bea0bdd4c298f5ba6b419069364371e4b16eb3066.jpg)  
Water Non-water Unlabelled (-1)Outside S2 swath (white in RGB)

![](images/3a146814d5f90f438584a3d00af1e84cd473804c5e7e9a977a26df85485733e1.jpg)

![](images/c0b99d443d495b712c6b95752ffde684b0efab09c785f9598f1a7cbe0d720b24.jpg)  
Training label fraction (nested, event-stratified)  
Figure 3: Dataset composition and input distributions. (A) Chip counts by event and Protocol A partition. (B) Valid-water fraction per chip, with event medians and a square-root horizontal scale; chips without valid labels are omitted. (C) Pixel composition by split; annotations give water as a fraction of labeled pixels. (D) Selected true-color images and hand labels, including the distinct treatment of unlabeled and out-of-swath areas. (E) Training-pool reflectance distributions compared with the reported HLS pretraining statistics; unequal distribution summaries are descriptive, not a formal domain-shift test. (F) Water fractions of the nested event-stratified training subsets. All training events remain represented, but both subset composition and optimization vary with seed.

This distinction is important when interpreting geographic transfer rather than merely describing nonoverlapping training files.

For label eficiency, nested subsets contained 25, 63, 126, and 252 training chips, approximately 10%, 25%, 50%, and 100% of the Protocol A pool. Whole chips were sampled with event stratification, retaining all ten training events. Seeds 0, 1, and 2 controlled subset sampling and network training. Validation and evaluation sets were fixed. Consequently, variation across seeds combines the efects of sample composition and optimization; it does not identify either component separately. The 89 validation labels and the fulllabel learning-rate selection were retained at every fraction. The curves therefore quantify fine-tuning-label sensitivity under shared model selection, not a complete annotationbudget comparison.

## 2.4. Spectral and neural comparison methods

NDWI and MNDWI were computed as in Equation (1), using B8A as the near-infrared input for consistency with the six-channel networks [3, 4].

$$
\mathrm { N D W I } = \frac { \rho _ { G } - \rho _ { N } } { \rho _ { G } + \rho _ { N } } , \qquad \mathrm { M N D W I } = \frac { \rho _ { G } - \rho _ { S } } { \rho _ { G } + \rho _ { S } }\tag{1}
$$

Here, $\rho _ { G }$ is green-band reflectance (B3), $\rho _ { N }$ is narrow near-infrared reflectance (B8A), and $\rho _ { S }$ is shortwave-infrared reflectance (B11). Each index was thresholded to label water. Three alternatives were evaluated: the fixed threshold zero; one threshold fitted on training labels on a grid with step 0.005 to maximize aggregated water IoU; and an unsupervised threshold estimated separately from each chip’s in-swath histogram using Otsu’s method [24]. Index fitting used the same labeled subsets as the corresponding neural experiments and no test labels.

The convolutional baseline was a six-channel U-Net trained from scratch [25]. It used four downsampling stages, base width 32, two $3 \times 3$ convolutions with batch normalization per stage, transposed-convolution upsampling, and skip connections. Its 7.76 million parameters make it a useful size-matched comparator for the complete Tiny segmentation network, rather than for the encoder alone.

Prithvi-EO-2.0-Tiny was loaded through TerraTorch with the public pretrained weights [12, 13]. Its encoder has 12 transformer blocks, embedding dimension 192, and approximately 5.63 million parameters. With one time step and 16 × 16-pixel patch embeddings, a 256×256 crop produces 256 spatial tokens. Optional time and location metadata were not supplied. Features from blocks 3, 6, 9, and 12 were reshaped into spatial maps, converted into a feature pyramid, and passed to a 128-channel UPerNet decoder and two-class head with dropout 0.1 [26]. This is actual network inference and adaptation, not classification of precomputed embeddings. The masked-autoencoder pretraining principle is described by He et al. [27].

Three Tiny variants shared the architecture. The frozen variant updated only the neck, decoder, and head; the finetuned variant updated the pretrained encoder and new layers;

and the random-initialization variant trained the architecture from scratch. The added Tiny components contain about 2.15 million parameters, giving 7.79 million in total. The larger Prithvi-EO-2.0-100M-TL reference contains 86.24 million encoder parameters and 91.97 million in the complete segmentation network. Although its decoder family and width match the Tiny configuration, its input adapters produce a larger non-encoder count, approximately 5.73 million. It was evaluated at 10% and 100% labels only, not under Protocol B. Table 2 summarizes these distinctions.

## 2.5. Training and inference

Each optimization step drew 16 random 256×256 crops. Crops with fewer than 5% valid labeled pixels were redrawn. Spatial augmentation consisted of flips and rotations by multiples of 90 degrees, applied consistently to images and masks. The two-class cross-entropy loss can be written in its equivalent binary form as Equation (2).

$$
\mathcal { L } ( \theta ) = - \frac { 1 } { N } \sum _ { i \in \Omega } \left[ y _ { i } \ln p _ { i } ( \theta ) + ( 1 - y _ { i } ) \ln \left( 1 - p _ { i } ( \theta ) \right) \right]\tag{2}
$$

In Equation (2), � denotes the trainable model parameters; Ω is the set of valid labeled pixels in the batch; � is its size; $y _ { i }$ is the reference class of pixel i, with 1 denoting water; and $p _ { i } ( \boldsymbol { \theta } )$ is the predicted water probability. Unlabeled and outof-swath pixels are excluded from Ω. Logarithms in this loss are natural logarithms. No class weights or Dice term were added. This choice provides a straightforward probabilistic objective, but does not itself guarantee calibrated predictions.

Optimization used AdamW [28], weight decay 0.05, 100 warm-up steps followed by cosine decay, gradient-norm clipping at 1.0, and mixed precision. The maximum budget was 1,500 steps. Validation was performed every 125 steps; the checkpoint with the greatest validation water IoU was selected. Early stopping required four evaluations without improvement after at least 500 completed steps. These settings define a common practical budget, not a proof that every architecture reached its best possible accuracy.

The base learning rate was selected from $1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$ using Protocol A, all training labels, and seed 0. U-Net and randomly initialized Tiny used $3 \times 1 0 ^ { - 4 }$ . Frozen and fine-tuned Tiny used $1 0 ^ { - 3 }$ for their new layers; the fine-tuned encoder used one-tenth of that rate. Prithvi-100M inherited the fine-tuned Tiny rate rather than receiving an independent search. The study reports 106 runs: 12 selection runs, 48 Protocol A Tiny-scale runs, 40 additional Protocol B runs, and six Prithvi-100M runs. The 94 evaluation runs exclude the selection runs; 30 stopped early.

Full-chip inference used nine overlapping 256-pixel windows with stride 128. Softmax water probabilities were averaged in overlap regions. The decision threshold was fixed at $\mathsf { p } > 0 . 5$ . This same stitching and threshold rule was used for the neural comparisons. The reported implementation retained model configurations, split identities, encoder fingerprints, training histories, metrics, and prediction files in a local Python and cloud-GPU training workflow (Figure 2C).

Neural architectures and selected learning rates.
<table><tr><td>Model</td><td>Encoder initialization</td><td>Updated components</td><td>Total (M)</td><td>Trainable (M)</td><td>Learning rate</td></tr><tr><td>U-Net</td><td>Random</td><td>All</td><td>7.76</td><td>7.76</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Tiny frozen</td><td>Pretrained</td><td>Neck + decoder +</td><td>7.79</td><td>2.15</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Tiny fine-tuned</td><td>Pretrained</td><td>head All</td><td>7.79</td><td>7.79</td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td></tr><tr><td>Tiny random</td><td>Random</td><td>All</td><td>7.79</td><td>7.79</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Prithvi-100M</td><td>Pretrained</td><td>All</td><td>91.97</td><td>91.97</td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td></tr></table>

For fine-tuned models, the two learning rates refer to new layers / encoder. Tiny encoder: approximately 5.63 M parameters; added components: 2.15 M. Prithvi-100M encoder: 86.24 M; added components: 5.73 M. The decoder family is UPerNet with width 128, but the input adapters and parameter counts are not identical across encoder sizes. Spectral indices have no neural parameters and use zero, training-fitted, or per-chip Otsu thresholds. Prithvi-100M inherits the Tiny learning rate and is evaluated only at 10% and 100% labels in Protocol A.

## 2.6. Accuracy, calibration, and statistical uncertainty

The primary statistic was water-class IoU obtained from a confusion matrix pooled over valid pixels in each evaluation set. Water F1 was reported alongside precision and recall; the definitions in Equation (3) make their relationship explicit [29].

$$
{ \mathrm { I o U } } = { \frac { \mathrm { T P } } { { \mathrm { T P } } + { \mathrm { F P } } + { \mathrm { F N } } } } , \qquad F _ { 1 } = { \frac { 2 { \mathrm { T P } } } { 2 { \mathrm { T P } } + { \mathrm { F P } } + { \mathrm { F N } } } }\tag{3}
$$

TP denotes pixels correctly classified as water, FP denotes non-water pixels classified as water, and FN denotes missed reference-water pixels. Precision is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F P } )$ , and recall is $\mathrm { T P / ( T P + F N ) }$ . True negatives contribute to balanced accuracy but not water IoU. A per-chip IoU is undefined when no valid reference pixels exist or when both reference and prediction contain no water; such chips do not contribute to a per-chip average. This convention difers from treating an empty-water chip as perfect. Aggregate and per-chip results answer diferent questions because their weighting difers.

The distinction between agreement and coverage also arises in the urban-canopy assessment of Narimani et al. [30], which evaluates candidate delineations separately from the amount of canopy represented. Here, it motivates retaining recall and boundary diagnostics beside IoU rather than allowing a single overlap score to stand for all aspects of a water map.

Water-probability calibration was summarized using 15 equal-width bins and the expected calibration error (ECE) in Equation (4), alongside reliability diagrams [31].

$$
\mathrm { E C E } = \sum _ { b = 1 } ^ { B } \frac { n _ { b } } { N } \left| \bar { p } _ { b } - \bar { y } _ { b } \right|\tag{4}
$$

In Equation (4), � is the number of bins, $n _ { b }$ is the number of valid pixels in bin �, � is the total number of evaluated valid pixels, $\bar { p } _ { b }$ is their mean predicted water probability, and $\hat { y } _ { b }$ is their observed water fraction. Empty bins contribute zero. Lower ECE indicates closer average agreement under this binning scheme; it is not a guarantee of accurate boundaries or of reliability within the minority water class. Stored probabilities have resolution 1/255, so very small calibration diferences should not be overinterpreted.

Predictive entropy was calculated using Equation (5).

$$
H ( p ) = - p \log _ { 2 } p - ( 1 - p ) \log _ { 2 } ( 1 - p )\tag{5}
$$

Here, $p$ is a pixel’s predicted water probability, $H ( p )$ is binary entropy in bits, and $\log _ { 2 }$ is the base-2 logarithm. The endpoint convention is $0 \log _ { 2 } 0 = 0$ . Entropy is zero at probabilities zero and one, and maximal at one-half. It was evaluated as an error-ranking score using the area under the receiver-operating-characteristic curve (AUROC), not interpreted as an error probability. Boundary analysis compared valid pixels within two pixels of a reference water/non-water boundary with interior pixels. The JRC stratum comparison measured recall for reference water marked permanent and for other reference water. These diagnostics used Protocol A, all labels, and seed 0 unless explicitly identified otherwise.

Confidence intervals used 2,000 cluster-bootstrap resamples, reflecting the dependence of pixels within chips [32]. For Protocol A, chips were sampled with replacement within each event and their confusion matrices were summed. For multiseed settings, the statistic was the mean of the seed-specific pooled IoUs on the same resampled chips. These intervals describe test-chip sampling conditional on the sampled events and trained models; the separate seed standard deviations describe training variability. Paired model diferences used identical resamples. An interval including zero does not demonstrate equivalence. Under Protocol B, event-specific IoUs were averaged without weighting by chip count, and intervals were obtained by resampling the 11 events. Eventlevel Wilcoxon tests and chip-property correlations were treated as exploratory. Shared training data between folds and the small number of events limit population-level inference.

## 2.7. Computational measurements

GPU measurements used a single NVIDIA Tesla T4 with 16 GB memory. Full-chip latency covered a batch of nine inference tiles in mixed precision, with 20 timed repetitions after five warm-up passes. CPU measurements used an Intel Core i9-12900H laptop, 14 threads, and float32 arithmetic, with ten timed repetitions after three warm-up passes. The three Tiny variants shared one CPU architecture timing. Fitted-index CPU measurements included index calculation and thresholding over the 105 test and Bolivia chips. Training time, peak GPU memory, parameter counts, and floating-point operations per 256-pixel tile were also recorded. These are device- and implementation-specific model timings, excluding an operational pipeline’s data acquisition, disk access, preprocessing, and export.

Table 3  
Seen-event test performance with all training labels (Protocol A).
<table><tr><td>Method</td><td>Water IoU mean  $\pm \thinspace \mathrm { S D }$ </td><td>IoU 95% CI</td><td>F1</td><td>Precision</td><td>Recall</td><td>ECE</td></tr><tr><td>U-Net</td><td> $0 . 8 5 6 \pm 0 . 0 0 1$ </td><td>0.796–0.897</td><td>0.922</td><td>0.936</td><td>0.909</td><td>0.014</td></tr><tr><td>Tiny frozen</td><td> $0 . 8 3 2 \pm 0 . 0 0 4$ </td><td>0.772–0.877</td><td>0.908</td><td>0.925</td><td>0.892</td><td>0.002</td></tr><tr><td>Tiny fine-tuned</td><td> $0 . 8 3 3 \pm 0 . 0 0 1$ </td><td>0.772–0.879</td><td>0.909</td><td>0.932</td><td>0.887</td><td>0.004</td></tr><tr><td>Tiny random</td><td> $0 . 8 0 9 \pm 0 . 0 0 3$ </td><td>0.748–0.859</td><td>0.895</td><td>0.926</td><td>0.865</td><td>0.003</td></tr><tr><td>Prithvi-100M</td><td> $0 . 8 4 1 \pm 0 . 0 0 2$ </td><td>0.783-0.886</td><td>0.914</td><td>0.936</td><td>0.892</td><td>0.005</td></tr><tr><td>MNDWI fitted</td><td> $0 . 7 8 7 \pm 0 . 0 0 0$ </td><td>0.711-0.846</td><td>0.881</td><td>0.869</td><td>0.893</td><td></td></tr><tr><td>NDWI fitted</td><td> $0 . 7 7 3 \pm 0 . 0 0 0$ </td><td>0.707–0.824</td><td>0.872</td><td>0.864</td><td>0.88</td><td></td></tr><tr><td>NDWI zero</td><td> $0 . 7 3 5$ </td><td>0.635–0.812</td><td>0.847</td><td>0.972</td><td>0.751</td><td></td></tr><tr><td>MNDWI zero</td><td>0.659</td><td>0.549–0.749</td><td>0.795</td><td>0.675</td><td>0.965</td><td></td></tr><tr><td>MNDWI Otsu</td><td>0.386</td><td>0.281–0.493</td><td>0.557</td><td>0.402</td><td>0.903</td><td></td></tr><tr><td>NDWI Otsu</td><td>0.308</td><td>0.220–0.398</td><td>0.471</td><td>0.32</td><td>0.896</td><td></td></tr></table>

90 test chips from ten events represented in the training pool. Neural metrics are means across three seeds; SD describes seed variation. Intervals are event-stratified chip-bootstrap intervals for the seed-mean pooled IoU, not precision or recall intervals. Fitted indices are deterministic with all labels. ECE uses 15 equal-width water-probability bins; An em dash indicates not applicable or not measured. The evaluation excludes unlabeled and out-of-swath pixels.

## 3. Results

## 3.1. Accuracy on seen events and the Bolivia holdout

U-Net achieved the greatest mean water IoU on the oficial seen-event test set: 0.856, with a 95% chip-bootstrap interval of 0.796-0.897 (Table 3). Fine-tuned Prithvi-100M reached 0.841, fine-tuned Tiny 0.833, frozen Tiny 0.832, and randomly initialized Tiny 0.809. The best index baseline was fitted MNDWI at 0.787. Precision varied less among the networks than recall: U-Net’s recall was 0.909, compared with 0.887 for fine-tuned Tiny. Thus, the U-Net advantage mainly reflected fewer missed water pixels rather than a large precision advantage.

Figure 4 places these aggregate results in their spatial context. Broad water bodies are often recovered by several methods, while disagreements occur along margins, narrow channels, and small features. The rows include a median example, relative model advantages, a Bolivia example, and a low-IoU case selected by predefined rules. They are not a random sample of average behavior. Among the 25 eligible seen-event test chips used for this illustration, fine-tuned Tiny exceeded U-Net on one chip, by 0.031 IoU. This is consistent with the aggregate ranking, but the restricted display subset should not replace the full test statistics.

The paired diference between fine-tuned Tiny and U-Net was -0.024 IoU, with a 95% interval of -0.037 to - 0.012 (Figure 5C). Fine-tuned Tiny exceeded fitted MNDWI by 0.046 (0.026-0.073). Against the identical randomly initialized architecture, the reported gain was 0.023 (0.014- 0.034). Paired diferences were computed before rounding; they can therefore difer by 0.001 from subtracting the displayed three-decimal means. The comparison supports the pretrained training strategy relative to random initialization under these settings, not superiority over the convolutional baseline.

Bolivia produced a diferent scale of uncertainty (Table 4). U-Net and Prithvi-100M achieved 0.795 and 0.794 IoU, respectively, followed by fitted NDWI at 0.780 and fine-tuned Tiny at 0.752. Frozen Tiny fell to 0.689. The fine-tuned Tiny-U-Net diference was -0.043, but its interval (-0.160 to 0.004) included zero. This does not establish equal performance; it indicates that the 15-chip holdout did not resolve that diference precisely. Pretrained fine-tuned Tiny exceeded its random counterpart by 0.046 (0.013-0.135). A single fitted spectral threshold was competitive in this event even though it was weaker on the seen-event test set.

## 3.2. Sensitivity to the number of fine-tuning chips

The benefit of pretraining within the Tiny architecture was largest with the smallest labeled subset (Figure 6; Table 5). With 25 training chips, fine-tuned Tiny achieved 0.814 IoU on the seen-event test set, compared with 0.768 for random initialization, a paired gain of 0.046 (0.029-0.066). The reported gain was 0.022-0.023 at larger fractions. Frozen Tiny was less efective with 25 chips, at 0.784, but approached the fine-tuned model with all labels.

U-Net nevertheless retained the highest seen-event mean at every label fraction. Its 25-chip IoU of 0.831 was numerically close to fine-tuned Tiny’s 252-chip value of 0.833. This is not an equivalence test, but it is an important practical comparison: the observed pretraining benefit did not establish greater fine-tuning-label eficiency than U-Net. Fitted MNDWI changed little, from 0.784 with 25 chips to 0.787 with 252, suggesting that this transparent spectral rule required relatively few labeled examples in the tested pool.

![](images/52d2e02c9ffcf623c2bc879e4cf7f9c9ab61b6d6c5e988e3e68eb70fe8ae4719.jpg)  
Figure 4: Qualitative segmentation comparisons under Protocol A, all labels, seed 0. Columns show true-color imagery, reference masks, MNDWI and neural error maps, and fine-tuned Tiny water probability. Error colors separate true positives, false positives, false negatives, true negatives, and invalid areas. Rows show a median eligible test example (A), the largest eligible relative Tiny and U-Net advantages (B,C), a median eligible Bolivia example (D), and a low-IoU example (E), with at most one chip per event. Selection used at least 5% reference water and 80% labeled pixels. These restricted and partly extreme examples do not represent a random sample of all chips. The annotated IoUs are single-chip, seed-0 values, not the three-seed pooled values in Tables 3 and 4.

The Bolivia curves were less stable and not monotonic. With 25 training chips, the seed standard deviation was 0.250 for U-Net, 0.382 for frozen Tiny, and 0.170 for fine-tuned Tiny. Fine-tuned Tiny reached 0.790 at 63 chips but 0.752 at 252. The seed-2, 25-chip runs had the lowest Bolivia IoU for every network; frozen Tiny reached only 0.044 in that run. Because the seed changed both chip selection and optimization, the result demonstrates sensitivity to the combined training realization, not proof that a particular subset alone caused failure. Prithvi-100M was measured only at the two endpoints; the connecting line in Figure 6 does not represent experiments at 25% or 50%.

## 3.3. Computational trade-ofs

The main advantage of the compact transformer was computational rather than a higher final overlap score (Figure 7; Table 6). Fine-tuned Tiny and U-Net had nearly identical total parameter counts, 7.79 and 7.76 million, but required 9.6 and 24.3 GFLOPs per tile. A full chip took 16.1 ms for fine-tuned Tiny and 30.6 ms for U-Net on the T4, a 1.90-fold latency ratio. On the CPU, 258 versus 665 ms gave a 2.58-fold ratio. The larger Prithvi model required 50.4 ms on the GPU and 1,386 ms on the CPU for a seen-event IoU increase from 0.833 to 0.841.

Bolivia event-holdout performance with all training labels (Protocol A).
<table><tr><td>Method</td><td>Water IoU mean ± SD</td><td>IoU 95% CI</td><td>F1</td><td>ECE</td></tr><tr><td>U-Net</td><td> $0 . 7 9 5 \pm 0 . 0 1 4$ </td><td>0.685–0.867</td><td>0.886</td><td>0.015</td></tr><tr><td>Tiny frozen</td><td> $0 . 6 8 9 \pm 0 . 0 2 8$ </td><td>0.470–0.796</td><td>0.815</td><td>0.036</td></tr><tr><td>Tiny fine-tuned</td><td> $0 . 7 5 2 \pm 0 . 0 3 1$ </td><td>0.552–0.844</td><td>0.859</td><td>0.025</td></tr><tr><td>Tiny random</td><td> $0 . 7 0 6 \pm 0 . 0 2 2$ </td><td>0.427–0.826</td><td>0.828</td><td>0.021</td></tr><tr><td>Prithvi-100M</td><td> $0 . 7 9 4 \pm 0 . 0 0 7$ </td><td>0.651–0.863</td><td>0.885</td><td>0.016</td></tr><tr><td>MNDWI fitted</td><td> $0 . 7 0 4 \pm 0 . 0 0 0$ </td><td>0.529–0.806</td><td>0.827</td><td></td></tr><tr><td>NDWI fitted</td><td> $0 . 7 8 0 \pm 0 . 0 0 0$ </td><td>0.643–0.857</td><td>0.876</td><td></td></tr><tr><td>NDWI zero</td><td>0.764</td><td>0.581–0.843</td><td>0.866</td><td></td></tr><tr><td>MNDWI zero</td><td>0.551</td><td>0.346-0.700</td><td>0.711</td><td></td></tr><tr><td>MNDWI Otsu</td><td>0.450</td><td>0.224–0.668</td><td>0.621</td><td></td></tr><tr><td>NDWI Otsu</td><td>0.439</td><td>0.210–0.662</td><td>0.611</td><td></td></tr></table>

15 chips from the Bolivia event, absent from gradient fitting, validation, and global learning-rate selection. Neural summaries use three seeds. Confidence intervals describe chip-sampling uncertainty within this event, not uncertainty over all possible future flood events. Model diferences with intervals spanning zero are not evidence of equivalence. All metrics use the same validity mask as Table 3.

Table 5  
Water IoU under reduced fine-tuning-label budgets (Protocol A).
<table><tr><td colspan="5">A. Seen-event test</td></tr><tr><td>Method</td><td>25 chips ( 10%)</td><td>63 chips (25%)</td><td> $1 2 6 \mathrm { c h i p s } ( 5 0 \% )$ </td><td>252 chips (100%)</td></tr><tr><td>U-Net</td><td> $0 . 8 3 1 \pm 0 . 0 1 4$ </td><td> $0 . 8 4 4 \pm 0 . 0 0 6$ </td><td> $0 . 8 5 7 \pm 0 . 0 0 5$ </td><td> $0 . 8 5 6 \pm 0 . 0 0 1$ </td></tr><tr><td>Tiny frozen</td><td> $0 . 7 8 4 \pm 0 . 0 5 2$ </td><td> $0 . 8 2 2 \pm 0 . 0 0 3$ </td><td> $0 . 8 2 2 \pm 0 . 0 1 0$ </td><td> $0 . 8 3 2 \pm 0 . 0 0 4$ </td></tr><tr><td>Tiny fine-tuned</td><td> $0 . 8 1 4 \pm 0 . 0 1 3$ </td><td> $0 . 8 3 4 \pm 0 . 0 0 3$ </td><td> $0 . 8 3 4 \pm 0 . 0 0 5$ </td><td> $0 . 8 3 3 \pm 0 . 0 0 1$ </td></tr><tr><td>Tiny random</td><td> $0 . 7 6 8 \pm 0 . 0 1 1$ </td><td> $0 . 8 1 1 \pm 0 . 0 0 8$ </td><td> $0 . 8 1 3 \pm 0 . 0 0 7$ </td><td> $0 . 8 0 9 \pm 0 . 0 0 3$ </td></tr><tr><td>Prithvi-100M</td><td> $0 . 8 2 0 \pm 0 . 0 0 5$ </td><td></td><td></td><td> $0 . 8 4 1 \pm 0 . 0 0 2$ </td></tr><tr><td>MNDWI fitted</td><td> $0 . 7 8 4 \pm 0 . 0 0 1$ </td><td> $0 . 7 8 4 \pm 0 . 0 0 3$ </td><td> $0 . 7 8 6 \pm 0 . 0 0 2$ </td><td> $0 . 7 8 7 \pm 0 . 0 0 0$ </td></tr><tr><td>NDWI fitted</td><td> $0 . 7 5 4 \pm 0 . 0 3 7$ </td><td> $0 . 7 6 3 \pm 0 . 0 1 4$ </td><td> $0 . 7 7 0 \pm 0 . 0 0 5$ </td><td> $0 . 7 7 3 \pm 0 . 0 0 0$ </td></tr></table>

<table><tr><td colspan="5">B. Bolivia holdout</td></tr><tr><td>Method</td><td>25 chips ( 10%)</td><td>63 chips (25%)</td><td>126 chips (50%)</td><td>252 chips (100%)</td></tr><tr><td>U-Net</td><td> $0 . 6 4 0 \pm 0 . 2 5 0$ </td><td> $0 . 7 5 7 \pm 0 . 0 7 3$ </td><td> $0 . 8 0 6 \pm 0 . 0 1 0$ </td><td> $0 . 7 9 5 \pm 0 . 0 1 4$ </td></tr><tr><td>Tiny frozen</td><td> $0 . 4 8 4 \pm 0 . 3 8 2$ </td><td> $0 . 6 5 5 \pm 0 . 1 5 2$ </td><td> $0 . 7 2 8 \pm 0 . 0 4 9$ </td><td> $0 . 6 8 9 \pm 0 . 0 2 8$ </td></tr><tr><td>Tiny fine-tuned</td><td> $0 . 6 8 0 \pm 0 . 1 7 0$ </td><td> $0 . 7 9 0 \pm 0 . 0 0 9$ </td><td> $0 . 7 5 6 \pm 0 . 0 3 0$ </td><td> $0 . 7 5 2 \pm 0 . 0 3 1$ </td></tr><tr><td>Tiny random</td><td> $0 . 6 7 3 \pm 0 . 0 7 3$ </td><td> $0 . 7 4 3 \pm 0 . 0 2 9$ </td><td> $0 . 7 3 9 \pm 0 . 0 4 0$ </td><td> $0 . 7 0 6 \pm 0 . 0 2 2$ </td></tr><tr><td>Prithvi-100M</td><td> $0 . 7 2 7 \pm 0 . 1 2 4$ </td><td></td><td></td><td> $0 . 7 9 4 \pm 0 . 0 0 7$ </td></tr><tr><td>MNDWI fitted</td><td> $0 . 7 0 5 \pm 0 . 0 2 4$ </td><td> $0 . 7 1 4 \pm 0 . 0 2 9$ </td><td> $0 . 7 1 4 \pm 0 . 0 1 7$ </td><td> $0 . 7 0 4 \pm 0 . 0 0 0$ </td></tr><tr><td>NDWI fitted</td><td> $0 . 7 7 2 \pm 0 . 0 1 8$ </td><td> $0 . 7 7 5 \pm 0 . 0 0 8$ </td><td> $0 . 7 7 8 \pm 0 . 0 0 3$ </td><td> $0 . 7 8 0 \pm 0 . 0 0 0$ </td></tr></table>

Each cell is mean ± SD across three seeds. Label fractions change training chips used for gradient updates or index fitting, while the 89-chip validation pool and full-data learning-rate selection remain fixed. Normalization uses all training images without their labels. An em dash indicates not applicable, not measured, or not run. Prithvi-100M was evaluated only at the endpoints. Paired-diference confidence intervals are shown in Figure 6 and discussed in the text: these SDs are not confidence intervals.

Frozen Tiny had the shortest median training run, 2.7 min; fine-tuned Tiny required 3.3 min, U-Net 5.6 min, and Prithvi-100M 8.7 min. Peak training memory was 1.8 GB for frozen Tiny, 2.0 GB for fine-tuned Tiny, 2.5 GB for U-Net, and 4.7 GB for Prithvi-100M. Spectral-index baselines remained much faster on the CPU, at 4.9-7.1 ms per chip. Model size alone therefore does not describe the practical trade-of. The tiny transformer was faster than a similarly sized convolutional network in this implementation, but it was neither the most accurate network nor the least expensive method overall.

## 3.4. Event-excluded evaluation

Across the 11 Protocol B evaluations, mean event IoU was 0.777 for U-Net, 0.768 for fine-tuned Tiny, 0.765 for frozen Tiny, and 0.725 for randomly initialized Tiny (Table 7; Figure 8A-C). The paired fine-tuned Tiny-U-Net diference was -0.010 (-0.034 to 0.015), with an exploratory Wilcoxon pvalue of 0.32. No event-average advantage of fine-tuned Tiny over U-Net was established. Fine-tuned Tiny did, however, exceed random initialization in all 11 target events, with an average gain of 0.043 (0.032-0.055).

![](images/07f6b35d8ade2f39139cbac56e2b05d25f30087f0448324f140e7c94b5d8bfe4.jpg)  
Figure 5: Accuracy and calibration with all training labels. (A) Pooled water IoU on 90 seen-event test chips (filled) and 15 Bolivia chips (open), with three-seed means, individual seeds, and chip-bootstrap 95% intervals. (B) Precision-recall positions and F1 contours. (C) Paired IoU diferences between fine-tuned Tiny and each comparator; filled markers identify intervals excluding zero. (D,E) Reliability diagrams for the seen-event test set and Bolivia, pooled across seeds; legends report mean seed ECE. Fifteen equal-width bins are used, and bins with fewer than 1,000 pixels are omitted from the curves. The displayed intervals concern IoU, not precision or recall. Index methods are omitted from probability-calibration panels.

Diferences among events were larger than the average diference between these networks. Fine-tuned Tiny exceeded U-Net in Paraguay (0.573 versus 0.522), Somalia (0.722 versus 0.653), and Sri Lanka (0.874 versus 0.864), but fell behind in Pakistan (0.624 versus 0.719) and India (0.731 versus 0.773). In Paraguay, both fitted indices, at 0.609 and 0.615, exceeded all networks. Conversely, fitted NDWI reached only 0.331 in Pakistan. Neither a pretrained encoder nor a simple spectral rule removed event-specific failure.

A paired comparison on the same 90 oficial test chips separated changes in fitting from changes in the evaluated images. Using seed 0, U-Net decreased from 0.856 to 0.847 when the respective event was excluded; fine-tuned Tiny decreased from 0.834 to 0.830. These values difer from the three-seed summaries by design. The small decreases should not be read as proof of broad geographic robustness: the global learning-rate selection was shared across protocols, and the full held-out event sets still showed substantial variation.

## 3.5. Reliability, water boundaries, and dificult cases

On the seen-event test set, mean ECE ranged from 0.002 for frozen Tiny to 0.014 for U-Net (Figure 5D). On Bolivia, it increased to 0.025 for fine-tuned Tiny and 0.036 for frozen Tiny, compared with 0.015 for U-Net. The reliability curves above the diagonal indicate that the Tiny models tended to assign water probabilities below the observed water frequency in those bins. Across Protocol B events, however, mean ECE was 0.009 for fine-tuned Tiny and 0.030 for U-Net. This change in ranking shows why calibration in one holdout is not a universal property of an architecture.

![](images/4ea4361fdf4b8830eb218150a24ba7ac01f1a97daaa926b23fc39f01f299f061.jpg)  
Figure 6: Sensitivity to the number of labeled fine-tuning chips. (A,B) Seen-event and Bolivia water IoU for nested training subsets, with three-seed means and individual seed points. The validation pool and global hyperparameter selection remain fixed. Prithvi-100M has measurements only at 25 and 252 chips; its connecting line does not imply intermediate experiments. (C,D) Paired diferences for pretrained versus random Tiny, frozen versus random Tiny, fine-tuned Tiny versus U-Net, and 100M versus fine-tuned Tiny. Bars are paired chip-bootstrap 95% intervals; filled markers indicate intervals excluding zero. Seed variation combines subset and optimization efects. Table 5 reports means and standard deviations, not confidence intervals for the upper curves

Pixels within two pixels of a water boundary accounted for 7.1% of valid test pixels but 68-76% of neural-model errors (Figure 8E). Fine-tuned Tiny’s boundary error rate was 22.0%, compared with 17.8% for U-Net, whereas their interior rates were similar, 0.65% and 0.63%. Extent and boundary behavior are therefore separate aspects of map quality, a distinction also made in the visible-boundary mapping work of Narimani et al. [33]. Here, the flood-specific evidence is the contrast between boundary and interior errors in Figure 8E.

JRC-permanent reference water was easier to recover than other reference water (Figure 8D). Neural recall was 0.985-0.993 for the permanent stratum and 0.800-0.854 for its complement. The diference cannot be interpreted as a measured penalty for newly flooded land without an independent pre-event reference. Per-chip IoU was positively associated with reference-water fraction (Spearman � 0.62- 0.73), while the bright-blue reflectance proxy and unlabeled fraction showed no clear association (absolute � at most 0.14; p at least 0.21). These exploratory associations do not establish turbidity, clouds, or vegetation as the causes of particular errors.

Entropy ranked erroneous pixels usefully, with AUROC 0.92-0.94 on seen-event test chips and 0.91-0.95 on Bolivia (Figure 8F). For fine-tuned Tiny, pixels at or above 0.5 bits represented 5.5% of the test pixels but contained 71% of its errors. Their error rate was 28%, compared with 0.7% elsewhere. Nevertheless, low entropy did not exclude failure. The Paraguay example in Figure 8G had high IoU (0.91), despite belonging to a dificult event, whereas the Pakistan canal example had IoU 0.34. These are examples from lowperforming events, not necessarily their worst individual chips.

Table 6 Measured computational cost of the evaluated implementations.
<table><tr><td>Method</td><td>Parameters total / trainable (M)</td><td>GFLOPs / 256-px tile</td><td>Training (min)</td><td>Training GPU memory (GB)</td><td>GPU (ms/chip)</td><td>CPU (ms/chip)</td></tr><tr><td>U-Net</td><td>7.8 / 7.8</td><td>24.3</td><td>5.6</td><td>2.5</td><td>30.6</td><td>665</td></tr><tr><td>Tiny frozen</td><td>7.8 / 2.2</td><td>9.6</td><td>2.7</td><td>1.8</td><td>15.9</td><td>258</td></tr><tr><td>Tiny fine-tuned</td><td>7.8 / 7.8</td><td>9.6</td><td>3.3</td><td>2.0</td><td>16.1</td><td>258</td></tr><tr><td>Tiny random</td><td>7.8 / 7.8</td><td>9.6</td><td>3.3</td><td>2.2</td><td>16.8</td><td>258</td></tr><tr><td>Prithvi-100M</td><td>92.0 / 92.0</td><td>53.0</td><td>8.7</td><td>4.7</td><td>50.4</td><td>1386</td></tr><tr><td>MNDWI fitted</td><td>0</td><td></td><td>一</td><td>一</td><td></td><td>4.9</td></tr><tr><td>NDWI fitted</td><td>0</td><td></td><td></td><td></td><td></td><td>5.7</td></tr><tr><td>MNDWI Otsu</td><td>0</td><td></td><td></td><td></td><td></td><td>6.7</td></tr><tr><td>NDWI Otsu</td><td>0</td><td></td><td></td><td></td><td></td><td>7.1</td></tr></table>

Training values: median at 100% labels, including early stopping. GPU: Tesla T4; one 512×512 chip is nine overlapping 256×256 tiles in one mixed-precision batch. CPU: Core i9-12900H, 14 threads, float32; the Tiny variants share one CPU timing. CPU index times include index calculation and thresholding. Timings exclude imagery download, file I/O, and geographic export. M denotes million parameters; an em dash indicates not applicable or not measured. Rounded parameter counts follow these measurements; exact architecture counts are in Table 2.

A GPU inference  
![](images/d195ce9beb6b2bfcf48b819263f6250cd706297fb0ea951acf766d45135d8df6.jpg)

B CPU inference  
![](images/faeb4ca60f5cf71d3e503425f216e3ce85cd1b76714fbea841ca239b708faff6.jpg)

C Training time  
![](images/291264b8723caf4beb05de418eb08f0a5f628f93ed801ff7899e9424b0b15b6f.jpg)

![](images/06059658d5590dcfd2276a01c588afb95624eca7d5c4b60295ab930ab893405b.jpg)

E Parameters  
![](images/1e9d44980b5138043a4cc5f3c0ec6c6880f8ced55f6ebff8b87c5e4648650a24.jpg)  
Figure 7: Accuracy and computational cost. (A,B) Three-seed pooled IoU against GPU and CPU latency per 512×512 chip. Filled markers denote seen-event test results and open markers Bolivia; marker area encodes parameter count. Tiny variants are ofset horizontally for legibility, and share one CPU timing. (C) Median training time with all labels and time per optimization step. (D) Peak GPU memory for training and full-chip inference. (E) Total/trainable parameter counts and per-tile floating-point operations. GPU timings use a Tesla T4, mixed precision, and nine overlapping tiles per chip; CPU timings use an i9-12900H with float32. Measurements exclude data acquisition and file I/O and do not estimate energy consumption.

Table 7  
Event-excluded fitting and matched seen/unseen comparisons (Protocol B).
<table><tr><td>Method</td><td>Mean event IoU (95% CI)</td><td>Worst event (IoU)</td><td>Events above U-Net</td><td>Unseen - seen IoU (95% CI)</td><td>Mean ECE</td></tr><tr><td>U-Net</td><td>0.777 (0.705–0.843)</td><td>Paraguay (0.522)</td><td></td><td>-0.009 (-0.020, -0.001)</td><td>0.030</td></tr><tr><td>Tiny frozen</td><td>0.765 (0.705–0.823)</td><td>Paraguay (0.564)</td><td>3/11</td><td>-0.011 (-0.018, -0.006)</td><td>0.010</td></tr><tr><td>Tiny fine-tuned</td><td>0.768 (0.703–0.828)</td><td>Paraguay (0.573)</td><td>3/11</td><td>-0.004 (-0.014, +0.006)</td><td>0.009</td></tr><tr><td>Tiny random</td><td>0.725 (0.657–0.792)</td><td>Paraguay (0.519)</td><td>1/11</td><td>-0.008 (-0.028, +0.010)</td><td>0.013</td></tr><tr><td>Prithvi-100M</td><td>not run</td><td>not run</td><td>not run</td><td>not run</td><td>not run</td></tr><tr><td>MNDWI fitted</td><td>0.730 (0.671–0.792)</td><td>Pakistan (0.602)</td><td>2/11</td><td>-0.002 (-0.008, +0.003)</td><td></td></tr><tr><td>NDWI fitted</td><td>0.695 (0.595–0.783)</td><td>Pakistan (0.331)</td><td>1/11</td><td>-0.014 (-0.034, +0.001)</td><td></td></tr></table>

Event means give equal weight to the 11 target events; CIs resample events. Each target uses seed 0 and the fixed-pool exclusions described in Section 2.3. Global learning-rate selection was reused from Protocol A; this is not fully nested event-blind tuning. Unseen - seen compares identical oficial test chips under A seed 0 and the corresponding B model, with a paired chip bootstrap. Prithvi-100M was not run under B. An em dash indicates not applicable or not measured.

## 4. Discussion

## 4.1. What pretraining did, and did not, provide

The results support pretraining within the compact transformer, but not a general advantage over U-Net. Fine-tuned Tiny exceeded the identical random-initialization architecture on the seen-event test set at every label fraction and in every Protocol B target event. The largest gain occurred with 25 training chips. Yet U-Net achieved higher seen-event mean IoU at every fraction and a slightly higher event-average value. This distinction is consistent with benchmarks showing that a large pretraining corpus does not automatically displace a strong supervised baseline [15, 16]. The useful finding is not that pretraining fails, but that its benefit depends on the comparator.

The random-initialization control strengthens this interpretation, with an important qualification. Architecture, decoder family, and maximum update budget were controlled, but pretrained and random models used diferent selected learning rates and encoder-rate policies. The result therefore estimates the benefit of the pretrained adaptation strategy under the tested tuning procedure, rather than an experiment changing only the initial weights. A fully factorial optimization study would be needed to separate these efects. Similarly, the larger model’s inherited learning rate and restricted evaluation do not establish the maximum attainable performance of Prithvi-100M.

Results from other spectral tasks should not be used to fill that evidential gap. The multi-trait grapevine study of

Farajpoor et al. [34] concerns relationships between leaf spectra and physiological or nutrient targets, rather than water segmentation. This emphasizes a target-specific interpretation of spectral representations: success with physiological traits does not imply the same architecture ranking for openwater boundaries.

Boundary behavior ofers a plausible explanation for the observed accuracy gap. At the supplied resolution, one 16- pixel transformer patch covers roughly 160 m in the northsouth direction, while the decoder must recover much finer edges. U-Net carries spatial detail through skip connections. Our similar interior error rates but diferent boundary rates are consistent with a local-detail limitation, although they do not prove that patch tokenization alone caused it. Hybrid approaches such as U-Prithvi explicitly combine pretrained representations with convolutional detail [17]. Such a model would be a logical next comparison, not an untested explanation to present as a result.

## 4.2. Label savings must be defined against the actual information budget

The smallest-subset experiments show that relatively few labeled training chips can support useful water segmentation in this benchmark. They do not show that the whole study could have been developed with only 25 labeled chips. The validation set remained labeled, learning rates were selected with full-label Protocol A training, and normalization used the full image pool. This is a fine-tuning-label experiment under shared development information. Future annotationbudget studies should vary validation and tuning resources as well as gradient-training labels.

The Bolivia variability also cautions against drawing a strong conclusion from a single subset. Some reduced-label means exceeded the full-label result, while their seed variability difered greatly. Because subset selection and optimization seeds were coupled, both can contribute. The broader smalldata remote-sensing literature emphasizes that experimental design, sample diversity, and transfer choices are as important as selecting a nominally data-eficient architecture [35]. Here, no method was reliably more transferable simply because its seen-event score changed little with fewer chips.

AHeld-out ∆ loU  
![](images/bb6dbf58e83754be29a8c6974fe3f76484b4ef07697531cd248146f48035b311.jpg)

B Held-out loU  
![](images/33fa3b7c9dbbeb4d15de9e48c27095714f321c6682ac8f1d7c3ab76d47cad4f6.jpg)

![](images/ad9b5867547784558452712e0cb1bbe2b23dbe36a9b3cad3775fdccbf282ca60.jpg)

D  
![](images/ceb00a0eca9c2f8552fa1725eedccfceec6d6da2bf247b370e618570ee3bd6fe.jpg)

![](images/ca39db351ee8bede728a1dd9ca372110554eef00d6a06b595caec6781300c147.jpg)

![](images/e2824b5b5e1c5020cde7eb3aebc291586e157b7970d4c56fc75c9dc875077ca9.jpg)

![](images/18761cec8cd84a68da7ce8a0c3c19468882d90d3f8c5318e7f686d4d1e47a3e8.jpg)  
Figure 8: Event-level performance and reliability. (A,B) Protocol B event IoUs and paired Tiny-U-Net diferences; the event mean gives equal weight to all 11 targets. These are event-excluded fits with shared Protocol A learning-rate selection, not fully nested tuning. (C) Changes on identical oficial test chips under seen-event versus event-excluded seed-0 fitting. (D) Recall for JRC-permanent and other reference water; the latter is not verified newly inundated land. (E) Boundary and interior error rates and the fraction of errors near boundaries. (F) Pixel error versus entropy, with error-detection AUROC. Panels D-F use Protocol A, all labels, seed 0. (G) Clear-image examples from Paraguay and Pakistan, two low-performing events; the Paraguay example itself has high IoU.

## 4.3. Geographic transfer is a property of the evaluation as well as the model

The observed event spread, including cases where indices outperformed every network, argues for reporting eventspecific results beside a pooled score. This agrees with cross-dataset flood-model assessments and with the wider caution against extrapolating environmental maps beyond their evaluated domain [10, 36]. An unweighted event mean answers a diferent question from a pixel-pooled IoU: it gives small events equal weight rather than allowing water-rich or larger collections to dominate.

Protocol B provides useful evidence about excluding an event from fitting, but its shared learning-rate selection prevents a claim of completely unseen event-level model development. Its non-Bolivia folds also omit Bolivia from the training pool, as dictated by the fixed benchmark manifests. These design details are not cosmetic. They distinguish the reported evaluation from fully nested leave-one-event-out validation. Replication should preserve them for comparability, while a future study should select hyperparameters inside each event-exclusion fold and repeat each fold with several seeds.

The target itself also constrains transfer claims. Work linking wildfire severity to snow persistence distinguishes predicting an environmental state from attributing a disturbancerelated response [37]. The parallel in this study is the difference between recognizing surface water and establishing that a flood newly created it. Event-level generalization of the former does not validate the latter.

## 4.4. A practical eficiency advantage, not an operational guarantee

Under the recorded implementation, fine-tuned Tiny reduced GPU latency by about 47% and CPU latency by about 61% relative to U-Net, while losing a reported 0.024 seen-event IoU. The decision between them is therefore application-specific. A resource-limited analyst may prefer the compact transformer when that accuracy diference is acceptable, whereas a user prioritizing the strongest seenevent overlap may prefer U-Net. Fitted spectral indices remain useful low-cost reference methods and, in some held-out events, competitive predictors.

These measurements do not include the cost of obtaining cloud-free observations, transferring imagery, reconstructing full scenes, or validating an operational map. They also do not measure electrical energy or carbon emissions. A shorter inference time should not be translated into an energyreduction percentage without power measurements. GPU and CPU ratios are specific to the tested hardware, precision, batching, and software. The study supports a computeaccuracy comparison for these implementations, not a general claim of operational readiness or reduced disaster losses.

## 4.5. Reliability maps support review but do not replace reference evidence

Entropy concentrated many errors into a relatively small subset of pixels. This gives it a practical use: prioritizing inspection of uncertain margins rather than treating every mapped pixel as equally trustworthy. However, AUROC measures ranking, not whether an entropy threshold provides a reliable probability of error. Low ECE can also coexist with locally important mistakes, particularly when non-water pixels dominate. Calibration under shift is known to depend on the target distribution [38], and the difering Bolivia and Protocol B rankings reinforce the need to inspect it locally.

The distinction between a mapped association and an explanation of a hazard is similarly important in multisource assessments of wildfire impacts and exposure [39]. In this study, lower recall outside the JRC-permanent-water stratum is an observed association. It cannot, by itself, identify water depth, turbidity, land cover, or flood origin as the mechanism. Those variables would require independent measurements.

The hand-labeled masks are a reference, not error-free physical truth. Boundary ambiguity, unobserved water beneath vegetation, and cloud obstruction afect both interpretation and evaluation [6]. The bootstrap intervals account for sampled chips or events, not annotation uncertainty, and the stored probabilities’ quantization limits very fine calibration comparisons. A clear-sky benchmark can establish agreement with visible reference water without demonstrating complete inundation mapping during cloudy flood peaks.

## 4.6. Limits and next tests

Several limits define the scope of the findings. The study contains only 11 events, unevenly represented by chips. Protocol A has three seeds, Protocol B one, and all models use a restricted update budget. Fine-tuned and random models do not share identical optimization policies. The 100M model lacks a separate learning-rate search and event-exclusion experiment. The supplied HLS-pretrained encoder receives Level-1C images at a diferent spatial scale; domain adaptation literature explains why normalization alone need not remove this mismatch [40]. A matched HLSinput experiment would be needed to estimate its efect.

Single-date input is another deliberate restriction. Satellite time-series methods can represent temporal development, as illustrated in the branched-broomrape study of Narimani et al. [41], but that is a diferent use of temporal information from the present experiment. For floods, the relevant next test would use pre-event and event-date observations to separate persistent water from change, rather than infer a temporal capability from a model name.

The most informative extensions are therefore targeted: nested event-level model selection, repeated event-exclusion seeds, an independently verified pre-event water reference, and optical-radar or multitemporal inputs evaluated under the same masks and metrics. These tests would address identified uncertainties without obscuring the current result with a more elaborate architecture. The present evidence is suficient to describe a useful computational trade-of, but not to declare a universally best flood-mapping model.

## 5. Conclusions

Compact Prithvi pretraining improved surface-water segmentation relative to random initialization of the same transformer, especially with fewer fine-tuning chips. It did not consistently outperform a similarly sized U-Net: U-Net led on the seen-event test set, while the event-average diference was unresolved by the reported interval. The principal advantage of fine-tuned Tiny was lower measured computational demand, with approximately half the GPU latency and two-fifths the CPU latency of U-Net under the tested implementation.

Event-specific errors, boundary behavior, and calibration are essential to interpreting that trade-of. Spectral indices remained valuable references and sometimes outperformed all networks on a dificult event. Entropy helped identify locations needing review, but did not establish correctness. Finally, these experiments map reference surface water during floods, not independently verified new inundation. Shared hyperparameter selection and fixed validation resources further limit claims about wholly unseen-event performance and total annotation savings. A defensible model choice should therefore combine accuracy, computation, and the similarity of the intended setting to the evaluated events, rather than rely on the presence or size of a pretrained encoder alone.

## Data availability statement

The study uses the public Sen1Floods11 version 1.1 dataset [8], the public NASA–IBM Prithvi-EO-2.0 model releases [12], and the JRC-derived permanent-water layer distributed with Sen1Floods11 [21]. Copernicus elevation data and Natural Earth geographic layers provide cartographic context in Figure 1. Derived numerical tables, split manifests, statistical outputs, an archived copy of the analysis code, and a sample of derived chip previews (true-colour excerpts, label visualisations, predicted masks, and georeferenced probability/error layers) are openly available on Zenodo at https://doi.org/10.5281/zenodo.23653544. The original Sen1Floods11 GeoTIFF collection remains available from its public Google Cloud bucket. Redistribution of source imagery, pretrained weights, and geographic layers follows the terms of the original providers.

## Code availability statement

Python routines for water-index baselines, segmentation metrics, event-stratified splits, and the U-Net / Prithvi model factory used in this study are available at https://github.com /MohammadrezaNarimaniUCDavis/Prithvi\_Tiny\_Floodw ater\_GeoAI. An archived replication snapshot is included in the Zenodo record https://doi.org/10.5281/zenodo.23653544.

## Author contributions

Parastoo Farajpoor: Conceptualization, Methodology, Validation, Visualization, Writing—original draft, Writing— review and editing.

Mohammadreza Narimani: Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Visualization, Writing—original draft, Writing—review and editing, Project administration.

## Funding

The authors declare that no external financial support was received for the research, authorship, or publication of this article.

## Conflict of interest

The authors declare that the research was conducted in the absence of commercial or financial relationships that could be construed as a potential conflict of interest.

## Ethics statement

The study used geospatial and environmental observations and did not involve human participants, identifiable personal information, or experimental work with animals.

## Generative AI statement

During the preparation of this work, the authors used ChatGPT to improve grammatical accuracy, refine sentence structure, and enhance visualizations. All AI-generated revisions were thoroughly reviewed and edited by the authors to ensure relevance and accuracy.

## Acknowledgments

The authors acknowledge the creators of Sen1Floods11, the NASA–IBM Prithvi team, and the developers of TerraTorch and the open-source geospatial software used in the study. Figure 1 contains modified Copernicus DEM data and Natural Earth geographic data. The study contains modified Copernicus Sentinel data. Terrain products were produced using Copernicus WorldDEM-30, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014– 2018, provided under Copernicus by the European Union and ESA.

## Abbreviations

AUROC, area under the receiver-operating-characteristic curve; ECE, expected calibration error; GFM, geospatial foundation model; HLS, Harmonized Landsat Sentinel-2; IoU, intersection-over-union; JRC, Joint Research Centre; LOEO, leave-one-event-out; MNDWI, modified normalized diference water index; NDWI, normalized diference water index; TOA, top-of-atmosphere; UPerNet, Unified Perceptual Parsing Network; ViT, vision transformer.

## References

[1] B. Tellman, J. A. Sullivan, C. Kuhn, A. J. Kettner, C. S. Doyle, G. R. Brakenridge, T. A. Erickson, and D. A. Slayback, “Satellite imaging reveals increased proportion of population exposed to floods,” Nature, vol. 596, no. 7870, pp. 80–86, 2021. [Online]. Available: https://doi.org/10.1038/s41586-021-03695-w

[2] J. Rentschler, M. Salhab, and B. A. Jafino, “Flood exposure and poverty in 188 countries,” Nature Communications, vol. 13, no. 1, 2022. [Online]. Available: https://doi.org/10.1038/s41467-022-30727-4

[3] S. K. McFeeters, “The use of the normalized diference water index (NDWI) in the delineation of open water features,” International Journal of Remote Sensing, vol. 17, no. 7, pp. 1425–1432, 1996. [Online]. Available: https://doi.org/10.1080/01431169608948714

[4] H. Xu, “Modification of normalised diference water index (NDWI) to enhance open water features in remotely sensed imagery,” International Journal of Remote Sensing, vol. 27, no. 14, pp. 3025–3033, 2006. [Online]. Available: https://doi.org/10.1080/01431160600589179

[5] F. Isikdogan, A. C. Bovik, and P. Passalacqua, “Surface water mapping by deep learning,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 10, no. 11, pp. 4909–4918, 2017. [Online]. Available: https://doi.org/10.1109/JSTARS.2017.2735443

[6] A. Shastry, E. Carter, B. Coltin, R. Sleeter, S. McMichael, and J. Eggleston, “Mapping floods from remote sensing data and quantifying the efects of surface obstruction by clouds and vegetation,” Remote Sensing of Environment, vol. 291, 2023. [Online]. Available: https://doi.org/10.1016/j.rse.2023.113556

[7] R. Bentivoglio, E. Isufi, S. N. Jonkman, and R. Taormina, “Deep learning methods for flood mapping: a review of existing applications and future research directions,” Hydrology and Earth System Sciences, vol. 26, no. 16, pp. 4345–4378, 2022. [Online]. Available: https://doi.org/10.5194/hess-26-4345-2022

[8] D. Bonafilia, B. Tellman, T. Anderson, and E. Issenberg, “Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2020, pp. 835–845. [Online]. Available: https://doi.org/10.1109/CVPRW50498.2020.00113

[9] E. Portalés-Julià, G. Mateo-García, C. Purcell, and L. Gómez-Chova, “Global flood extent segmentation in optical satellite images,” Scientific Reports, vol. 13, no. 1, 2023. [Online]. Available: https://doi.org/10.1038/s41598-023-47595-7

[10] E. Portalés-Julià, G. Mateo-García, and L. Gómez-Chova, “Understanding flood detection models across Sentinel-1 and Sentinel-2 modalities and benchmark datasets,” Remote Sensing of Environment, vol. 328, 2025. [Online]. Available: https: //doi.org/10.1016/j.rse.2025.114882

[11] M. Narimani, A. Pourreza, A. Moghimi, and P. Farajpoor, “Sentinel-2 for crop yield estimation: a systematic review,” Smart Agricultural Technology, vol. 14, 2026. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S2772375526006301

[12] D. Szwarcman, S. Roy, P. Fraccaro, T. E. Gislason, B. Blumenstiel, R. Ghosal, P. H. de Oliveira, J. L. de Sousa Almeida, R. Sedona, Y. Kang, S. Chakraborty, S. Wang, C. Gomes, A. Kumar, V. Gaur, M. Truong, D. Godwin, S. Khallaghi, H. Lee, C.-Y. Hsu, A. A. Asanjan, B. Mujeci, D. Shidham, R. O. Balogun, V. Kolluru, T. Keenan, P. Arevalo, W. Li, H. Alemohammad, P. Olofsson, T. Mayer, C. Hain, R. Kennedy, B. Zadrozny, D. Bell, G. Cavallaro, C. Watson, M. Maskey, R. Ramachandran, and J. B. Moreno, “Prithvi-EO-2.0: A versatile multitemporal foundation model for Earth observation applications,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–20, 2026. [Online]. Available: https://doi.org/10.1109/tgrs.2025.3642610

[13] C. Gomes, B. Blumenstiel, J. L. De Sousa Almeida, P. H. De Oliveira, P. Fraccaro, F. M. Escofet, D. Szwarcman, N. Simumba, R. Kienzler, and B. Zadrozny, “TerraTorch: The geospatial foundation models toolkit,” in IGARSS 2025 - 2025 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2025, pp. 6364–6368. [Online].

Available: https://doi.org/10.1109/igarss55030.2025.11243570

[14] M. Narimani, A. Pourreza, and P. Farajpoor, “Mapping tomato cropping systems in California using AlphaEarth geospatial embeddings and deep learning analysis,” arXiv preprint arXiv:2605.21804, 2026. [Online]. Available: https: //arxiv.org/abs/2605.21804

[15] V. Marsocci, Y. Jia, G. L. Bellier, D. Kerekes, L. Zeng, S. Hafner, S. Gerard, E. Brune, R. Yadav, A. Shibli, H. Fang, Y. Ban, M. Vergauwen, N. Audebert, and A. Nascetti, “PANGAEA: Assessing geospatial foundation models capabilities through a global and inclusive benchmark,” IEEE Geoscience and Remote Sensing Magazine, vol. 14, no. 1, pp. 245–285, 2026. [Online]. Available: https://doi.org/10.1109/mgrs.2025.3628194

[16] S. Kaushik, L. Maurya, B. Tellman, and Z. Zhang, “Assessing geo-foundational models for flood inundation mapping: Benchmarking models for Sentinel-1, Sentinel-2, and planetscope,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 19, pp. 5649–5665, 2026. [Online]. Available: https://doi.org/10.1109/jstars.2026.3656855

[17] V. Kostejn, Y. Essus, J. Abrahamson, and R. R. Vatsavai, “U-Prithvi: Integrating a foundation model and U-Net for enhanced flood inundation mapping,” in 13th International Conference on Geographic Information Science (GIScience 2025), ser. Leibniz International Proceedings in Informatics (LIPIcs), vol. 346. Schloss Dagstuhl – Leibniz-Zentrum für Informatik, 2025, pp. 18:1–18:17. [Online]. Available: https://doi.org/10.4230/LIPIcs.GIScience.2025.18

[18] D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schröder, W. Thuiller, D. I. Warton, B. A. Wintle, F. Hartig, and C. F. Dormann, “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017. [Online]. Available: https://doi.org/10.1111/ecog.02881

[19] P. Ploton, F. Mortier, M. Réjou-Méchain, N. Barbier, N. Picard, V. Rossi, C. Dormann, G. Cornu, G. Viennois, N. Bayol, A. Lyapustin, S. Gourlet-Fleury, and R. Pélissier, “Spatial validation reveals poor predictive performance of large-scale ecological mapping models,” Nature Communications, vol. 11, no. 1, 2020. [Online]. Available: https://doi.org/10.1038/s41467-020-18321-y

[20] P. Farajpoor and M. Narimani, “Explainable GeoAI for urban wildfire structure loss: vegetation and built environment in the 2025 palisades fire,” Frontiers in Sustainability, vol. 7, 2026. [Online]. Available: https://doi.org/10.3389/frsus.2026.1975724

[21] J.-F. Pekel, A. Cottam, N. Gorelick, and A. S. Belward, “Highresolution mapping of global surface water and its long-term changes,” Nature, vol. 540, no. 7633, pp. 418–422, 2016. [Online]. Available: https://doi.org/10.1038/nature20584

[22] S. Mitra, M. Narimani, and P. Farajpoor, “A Sentinel-2 benchmark dataset for deep-learning active-fire segmentation across 25 California wildfires, July 2020–August 2026,” Data in Brief, vol. 69, 2026. [Online]. Available: https://doi.org/10.1016/j.dib.2026.113357

[23] M. Claverie, J. Ju, J. G. Masek, J. L. Dungan, E. F. Vermote, J.-C. Roger, S. V. Skakun, and C. Justice, “The harmonized Landsat and Sentinel-2 surface reflectance data set,” Remote Sensing of Environment, vol. 219, pp. 145–161, 2018. [Online]. Available: https://doi.org/10.1016/j.rse.2018.09.002

[24] N. Otsu, “A threshold selection method from gray-level histograms,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66, 1979. [Online]. Available: https: //doi.org/10.1109/TSMC.1979.4310076

[25] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, ser. Lecture Notes in Computer Science. Springer International Publishing, 2015, pp. 234–241. [Online]. Available: https://doi.org/10.1007/978-3-319-24574-4\_28

[26] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified perceptual parsing for scene understanding,” in Computer Vision – ECCV 2018, ser. Lecture Notes in Computer Science. Springer

International Publishing, 2018, pp. 432–448. [Online]. Available: https://doi.org/10.1007/978-3-030-01228-1\_26

[27] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 15 979–15 988. [Online]. Available: https://doi.org/10.1109/CVPR52688.2022.01553

[28] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6Ri CqY7

[29] A. E. Maxwell, T. A. Warner, and L. A. Guillén, “Accuracy assessment in convolutional neural network-based deep learning remote sensing studies—part 2: Recommendations and best practices,” Remote Sensing, vol. 13, no. 13, p. 2591, 2021. [Online]. Available: https://doi.org/10.3390/rs13132591

[30] M. Narimani, S. Mitra, and P. Farajpoor, “From crown candidates to neighborhood screening: integrating optical GeoAI and spatial modeling for urban-canopy assessment in Davis, California,” arXiv preprint arXiv:2608.13856, 2026. [Online]. Available: https://arxiv.org/abs/2608.13856

[31] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proceedings of the 34th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 70. PMLR, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

[32] C. A. Field and A. H. Welsh, “Bootstrapping clustered data,” Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 69, no. 3, pp. 369–390, 2007. [Online]. Available: https://doi.org/10.1111/j.1467-9868.2007.00593.x

[33] M. Narimani, V. Anand, and P. Farajpoor, “Farmland extent and visible boundary mapping from 1 m NAIP imagery using residual U-Net and text-prompted SAM 3 refinement,” arXiv preprint arXiv:2607.21881, 2026. [Online]. Available: https://arxiv.org/abs/2607.21881

[34] P. Farajpoor, A. Pourreza, M. Narimani, A. El-Kereamy, and M. W. Fidelibus, “Multi-trait spectral modeling for estimating grapevine leaf traits and nutrients,” Plant Phenomics, vol. 7, 2025. [Online]. Available: https://doi.org/10.1016/j.plaphe.2025.100142

[35] A. Safonova, G. Ghazaryan, S. Stiller, M. Main-Knorn, C. Nendel, and M. Ryo, “Ten deep learning techniques to address small data problems with remote sensing,” International Journal of Applied Earth Observation and Geoinformation, vol. 125, 2023. [Online]. Available: https://doi.org/10.1016/j.jag.2023.103569

[36] H. Meyer and E. Pebesma, “Machine learning-based global maps of ecological variables and the challenge of assessing them,” Nature Communications, vol. 13, no. 1, 2022. [Online]. Available: https://doi.org/10.1038/s41467-022-29838-9

[37] P. Farajpoor and M. Narimani, “From wildfire severity to snow persistence: a multisource GeoAI study of the 2020 creek fire,” Frontiers in Forests and Global Change, 2026, in press. [Online]. Available: https://doi.org/10.3389/fgc.2026.2016172

[38] Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. V. Dillon, B. Lakshminarayanan, and J. Snoek, “Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift,” in Advances in Neural Information Processing Systems, vol. 32. Curran Associates, Inc., 2019. [Online]. Available: https://papers.nips.cc/paper\_files/paper/2019/hash/8558cb408c1d7 6621371888657d2eb1d-Abstract.html

[39] P. Farajpoor, M. Ardebili Pour, M. B. Ghiasi, and M. Narimani, “Multisource remote sensing and geospatial analysis of vineyard wildfire impacts and exposure: the 2019 kincade fire,” Frontiers in Sustainability, 2026, in press. [Online]. Available: https://doi.org/10.3389/frsus.2026.1997548

[40] D. Tuia, C. Persello, and L. Bruzzone, “Domain adaptation for the classification of remote sensing data: An overview of recent advances,” IEEE Geoscience and Remote Sensing Magazine, vol. 4, no. 2, pp. 41–57, 2016. [Online]. Available: https://doi.org/10.1109/MGRS.2016.2548504

[41] M. Narimani, A. Pourreza, A. Moghimi, P. Farajpoor, H. Jafarbiglu, and M. Mesgaran, “Branched broomrape detection in tomato farms using satellite imagery and time-series analysis,” in Proceedings ofSPIE, vol. 13475, 2025. [Online]. Available: https://doi.org/10.1117/12.3059998