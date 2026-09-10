# MethaneFuse: Learning from Multi-Sensor Satellite Observations for Methane Plume Detection

Yuyao Wang

Juliana Y. Leung

Di Niu

Department of Electrical and

Department of Civil and

Department of Electrical and

Computer Engineering

Environmental Engineering

Computer Engineering

University of Alberta

University of Alberta

University of Alberta

Edmonton, Canada

yuyao16@ualberta.ca

Edmonton, Canada

Edmonton, Canada

juliana2@ualberta.ca

dniu@ualberta.ca

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Abstract—Methane plume detection from satellite imagery is fundamentally constrained by incomplete observations: although public satellites provide complementary spatial, spectral, and atmospheric evidence, real plume cases rarely contain fully paired multi-sensor measurements because of revisit schedules, cloud coverage, acquisition quality, and the transient nature of emissions. Most learning-based detectors are developed around single-sensor inputs, especially Sentinel-2 (S2), but many reported plume cases do not have valid S2 observations. This creates a challenging partial-observation learning problem where different plume cases are captured by different subsets of heterogeneous sensors rather than by complete multi-sensor stacks.

To address this problem, we construct MethaneUnion, a temporal multi-sensor dataset built from Carbon Mapper plume reports and matched S2, Landsat 8/9 (L8/9), EMIT, and Sentinel-5P (S5P) observations. Built on MethaneUnion, we propose MethaneFuse, a learning framework for methane plume detection from heterogeneous satellite observations where fully paired multi-sensor measurements are almost never available.

MethaneUnion expands usable coverage from 3,211 valid S2-matched plume cases to 8,981 reported plume cases with multi-sensor observations. Experiments show that MethaneFuse consistently outperforms independently trained sensor predictors, heuristic fusion strategies, and generic Earth observation representation transfer. At the representative 480 m setting, MethaneFuse achieves 84.87 F1 and 93.62 AUROC, improving over the strongest baseline by 5.65 F1 and 8.30 AUROC points while reducing false positives by 8.19 points. Sensor-availability experiments further show that MethaneFuse improves detection when S2 is available and transfers plume knowledge to L8/9, EMIT, and S5P when S2 is unavailable. These results demonstrate that learning from incomplete heterogeneous sensor observations provides a practical and effective alternative to conventional single-sensor methane detection pipelines. The dataset and code are available at https://github.com/yuyao-wang/MethaneFuse.

Index Terms—Satellite methane plume detection, remote sensing, earth observation, multi-sensor fusion, missing-modality learning, vision transformers, parameter-efficient fine-tuning.

## I. INTRODUCTION

Methane is a short-lived but powerful greenhouse gas, making rapid emission reduction a high-impact path for limiting near-term warming [1]. Reliable methane monitoring is therefore essential for locating large or intermittent emission sources, verifying reported emissions, and prioritizing infrastructure repair [2]. However, detecting methane plumes from satellite observations is difficult because plume signals are transient, weak, and strongly affected by sensor resolution, spectral coverage, surface background, and acquisition conditions [3], [4]. Unlike land-cover classes or common objects in images, methane plumes are not persistent scene elements with stable shapes or clear visual boundaries.

Existing learning-based plume detectors commonly rely on Sentinel-2 (S2) because its fine-resolution multispectral shortwave infrared (SWIR) bands are useful for plume detection [5]–[7]. However, valid S2 observations are often unavailable for transient plume reports because of revisit timing, cloud cover, acquisition geometry, and quality filtering. In the Carbon Mapper plume reports collected from 2016–2025, more than 24,000 reported plume cases yield only 3,211 valid S2-matched cases after sensor matching and quality filtering, meaning that most reported plumes cannot be used by S2- only detectors [8], [9]. Other public satellites can provide additional methane-related evidence: Landsat 8/9 (L8/9) offers multispectral observations at approximately 30 m resolution, EMIT provides hyperspectral observations at approximately 60 m resolution, and Sentinel-5P (S5P) provides coarse CH products with kilometer-scale footprints [10]–[16].

The challenge is that this additional evidence is rarely available as complete multi-sensor measurements for the same plume case. Among the Carbon Mapper plume reports from 2016 to 2025, only four cases have observations from all four sensors. Therefore, the practical learning problem is to learn methane-relevant cues from plume cases observed by different satellite sources, rather than from plume cases where all sensors are available together. This motivates our central question: How can methane plume detectors learnfrom heterogeneous multi-sensor satellite observations when fully paired sensor measurements are almost unavailable?

In this paper, we first construct MethaneUnion, a temporal multi-sensor satellite dataset for methane plume detection. MethaneUnion reflects the real monitoring setting in which different plume cases are captured by different subsets of public satellites. Built on MethaneUnion, we propose MethaneFuse, a learning framework that learns methane plume evidence from available sensor subsets through sensor-native representation learning and transfers it to scale-controlled plume classification and segmentation with lightweight sensor-aware adaptation.

Our contributions are summarized as follows:

• We construct MethaneUnion, to the best of our knowledge, the first methane plume dataset specifically designed for temporal multi-sensor learning from naturally available satellite observations. Using Carbon Mapper plume reports as methane anchors, we match S2, L8/9, EMIT, and S5P observations near reported plume locations and times, organize each sensor stream into plumetime, recent-reference, and earlier-reference views, and derive plume-positive and local non-plume background samples for supervised evaluation. MethaneUnion expands usable coverage from 3,211 valid S2-matched plume cases to 8,981 reported plume cases with multisensor observations.

• We propose MethaneFuse for learning methane plume knowledge from heterogeneous multi-sensor plume observations. During Stage 1 representation learning, each available sensor is kept in its native spatial–spectral form and encoded independently, while unavailable sensors are masked out. A masked sensor-set fusion module learns to aggregate plume evidence across the available sensor views without requiring complete four-sensor measurements for each target location and time.

• We introduce a lightweight sensor-aware adaptation stage for downstream plume detection at different evaluation scales. The Stage 1 encoder is frozen, and only CLS-routed LoRA expert adapters and task heads are trained for query-level classification and plume segmentation. This adaptation stage transfers the learned multisensor representation to evaluation crops with 120–960 m ground regions, while requiring only a small number of trainable parameters and separating classification fusion from segmentation-capable sensor fusion.

We conduct extensive experiments on MethaneUnion and summarize three main findings. First, MethaneFuse outperforms independently trained sensor predictors, heuristic fusion strategies, and prior Earth observation pretrained models after supervised fine-tuning on the same multi-sensor data, including SatMAE [20], AnySat [23], and Panopticon [24]. In the main 480 m evaluation setting, MethaneFuse improves over the strongest baseline by 5.65 F1 points and 8.30 AUROC points while reducing false positives by 8.19 points. Second, MethaneFuse improves detection both when S2 is available and when it is missing. In S2-present groups, MethaneFuse improves F1 by 3.5 points for S2 alone and by 14.3 points for S2+S5P; in S2-absent groups, it improves all evaluated L8/9, EMIT, and S5P combinations, with gains up to 15.3 F1 points.

Third, MethaneFuse remains effective across evaluation crops from 120 m to 960 m on the ground, consistently outperforming the strongest heuristic fusion baseline at each scale. For plume segmentation, MethaneFuse-Seg further improves fused IoU+ using only sensors that can provide meaningful plumemask predictions.

## II. RELATED WORK

## A. Satellite Methane Detection and Benchmarks

Satellite methane detection uses heterogeneous instruments with different roles: coarse atmospheric products support broad CH<sub>4</sub> screening, multispectral SWIR imagery supports finerscale plume analysis, and hyperspectral observations provide denser spectral evidence for retrieval and mapping [3], [10], [12]–[15]. Operational systems often combine these sensors sequentially through a tip-and-cue pipeline, where coarse detections guide higher-resolution follow-up for localization, attribution, or quantification [10]. This demonstrates sensor complementarity, but methane plume detection has not been systematically studied as a multimodal learning problem over the heterogeneous satellite observations available for each target location and time.

Learning-based methane detection remains largely sensorspecific, with models typically built around one dominant observation source, including multispectral SWIR imagery, S2 transformers, TROPOMI screening, or hyperspectral plume benchmarks [6], [7], [10], [14], [16]–[18]. Recent datasets begin to collect methane plume examples across sensors [19], but they mainly unify sensor-specific datasets rather than formulating methane plume detection as temporal multi-sensor learning. They do not explicitly represent each plume case by the available heterogeneous satellite observations at the same target location and time. MethaneUnion targets this setting by organizing reported plume cases into multi-sensor observations with sensor-dependent labels.

## B. EO Foundation Models and Multi-Sensor Fusion

EO models learn transferable geospatial representations from large-scale satellite imagery using masked reconstruction, temporal or multispectral pretraining, cross-sensor alignment, and multimodal self-supervision [20], [21]. Recent multi-sensor and any-sensor models further support heterogeneous resolutions, spectral configurations, sensor metadata, and multimodal inputs within unified backbones [22]– [24]. These models provide strong generic representations for remote-sensing tasks, especially when target semantics are persistent across sensors and acquisition conditions.

Remote-sensing fusion and missing-modality learning have also been widely studied through early, intermediate, and late fusion, cross-modal attention, hyperspectral–multispectral fusion, SAR–optical fusion, spatiotemporal fusion, modality dropout, hetero-modal embeddings, and robust multimodal transformers [25]–[27]. These methods are highly relevant to incomplete sensor availability, but methane monitoring introduces a different target structure. Plume evidence is transient, diffuse, and observation-conditioned: useful cues may appear as SWIR absorption, hyperspectral gas-sensitive structure, weak multispectral anomaly, or coarse atmospheric enhancement depending on the sensor and scale. Therefore, methane evidence is not simply the common semantic intersection shared by all sensors. MethaneFuse addresses this gap through methane-supervised representation learning, sensornative tokenization, and masked sensor-set fusion, allowing the model to preserve and aggregate the union of available methane-relevant cues.

## C. Parameter-Efficient and Sensor-Aware Adaptation

Parameter-efficient fine-tuning has become a practical strategy for adapting large pretrained models with limited labeled data. LoRA freezes the backbone and learns low-rank residual updates [28], while recent geospatial studies show that LoRA, adapters, and related modules can adapt EO backbones for classification, segmentation, and change detection with lower training cost than full fine-tuning [29]. Mixture-ofexperts methods provide conditional specialization by routing inputs to different experts [30], and recent parameter-efficient adaptation studies have explored mixtures of LoRA experts and layer-wise expert allocation for task-, domain-, or inputconditioned specialization [31], [32]. Related remote-sensing work has explored expert-based models for heterogeneous sensors, missing modalities, and multimodal foundation-model pretraining [33], [34].

For methane monitoring, adaptation is not only a parameterefficiency issue. False positives and plume cues are strongly sensor-dependent because clouds, surface materials, spectral noise, retrieval artifacts, and spatial resolution affect each satellite differently. This motivates sensor-aware adaptation instead of applying a single shared fine-tuning module to all observations. In MethaneFuse, the shared methane representation is frozen during downstream adaptation, and lightweight sensor-aware LoRA expert adapters are inserted to suppress sensor-specific background artifacts while preserving crosssensor transfer for classification and segmentation.

## III. METHODOLOGY

We propose MethaneFuse, a two-stage framework for learning from event-centered multi-sensor observations. For each plume event or query anchor, only a naturally available subset of heterogeneous satellite sensors may be observed. To instantiate this setting, we construct MethaneUnion, an eventcentered partial-observation collection built from Carbon Mapper plume reports and naturally available S2, L8/9, EMIT, and S5P observations. MethaneUnion is not a fully paired multisensor dataset; instead, it preserves missing-sensor patterns caused by revisit timing, mission coverage, clouds, acquisition quality, and the transient lifetime of methane plumes.

MethaneFuse uses two training views derived from the same event-centered pool. Stage 1 learns methane representations from fixed-pixel sensor-native samples, while Stage 2 adapts them to query-level classification and segmentation.

Across both stages, the central inference problem remains unchanged: predicting methane occurrence from the naturally available subset of heterogeneous sensors for each queried event. Here, a query denotes a scale-controlled prediction unit derived from an event, specified by a timestamp, center location, and geographic footprint.

## A. MethaneUnion Dataset Construction

Let $ { \mathcal { S } } ~ = ~ \{ \mathrm { S 2 } , \mathrm { L 8 } / 9 , \mathrm { E M I T } , \mathrm { S 5 P } \}$ denote the supported sensor set, corresponding to S2, L8/9, EMIT, and S5P. For each Carbon Mapper plume report, we use its timestamp, geolocation, and plume mask as the event anchor [8]. Around each anchor, we collect naturally available observations from S2 Level-2A surface reflectance, L8/9 Collection 2 Level-2 surface reflectance, EMIT Level-2A hyperspectral surface reflectance, and S5P Level- $2 \ \mathrm { C H _ { 4 } }$ products [9], [11], [35], [36]. Each event is represented as a partial sensor set

$$
\mathcal { X } _ { i } = \left\{ ( \mathbf { X } _ { i } ^ { s } , m _ { i } ^ { s } ) : \mathbf { X } _ { i } ^ { s } \in \mathbb { R } ^ { T \times C _ { s } \times H _ { i } ^ { s } \times W _ { i } ^ { s } } , ~ m _ { i } ^ { s } \in \{ 0 , 1 \} \right\} _ { s \in \mathcal { S } _ { ( 1 ) } ^ { s } } ,
$$

where $T$ denotes the temporal context length, $C _ { s }$ is the sensorspecific spectral or product channel dimension, and $H _ { i } ^ { s } \times W _ { i } ^ { s }$ is the crop size determined by the corresponding training view. The binary variable $m _ { i } ^ { s }$ indicates whether sensor s is available for event i, and the observed sensor subset is

$$
{ \mathcal { M } } _ { i } = \{ s \in { \mathcal { S } } : m _ { i } ^ { s } = 1 \} .\tag{2}
$$

Only sensors in $\mathcal { M } _ { i }$ are used for tokenization, encoding, fusion, and inference.

For each available sensor, we retrieve a temporal context consisting of an event-day observation near the Carbon Mapper timestamp, a seasonal-history observation around three months before the event, and a long-term-background observation around one year before the event. In the implementation, we set $T = 3$ and stack these temporal observations along the channel dimension before tokenization. Thus, the raw input channel dimension for sensor s is $3 C _ { s } \mathrm { : }$ : S2 uses 12 surfacereflectance bands, L8/9 uses 7 surface-reflectance bands, EMIT is converted into a compact 16-band methane-relevant spectral representation, and S5P provides coarse Level-2 $\mathrm { C H _ { 4 } }$ context. The resulting channel-stacked inputs have 36 channels for S2, 21 channels for L8/9, and 48 channels for EMIT before sensor-native tokenization.

Specifically, each EMIT observation is projected through WorldView-3 spectral response functions using radiometric bandpass integration, radiometrically aligned by percentile matching, and reprojected to a local UTM coordinate system on the native 60 m EMIT grid [37]. This preprocessing reduces the original 285-band dimensionality, balances the number of input channels across sensors, and preserves shortwaveinfrared absorption structure near 2.3 µm that is relevant to methane detection.

We partition MethaneUnion before constructing Stage 1 crops, Stage 2 queries, or segmentation masks, so that all observations and derived samples associated with the same Carbon Mapper plume report remain in the same split. This avoids evaluating the model on different crops, query footprints, or sensor views derived from a plume report seen during training. Figure 2 summarizes the dataset construction and derived learning datasets; the chronological and geo-clustered evaluation protocols are specified in Section IV.

![](images/d80ed908a1dff7df1bdbc7062ab07030b10a329f85e4fcfd6f0cf25c14b4e126.jpg)  
Fig. 1. Overview of MethaneFuse. Stage 1 performs sensor-native representation learning on $\mathcal { D } _ { \mathrm { r e p } }$ , where available sensor observations are tokenized, encoded by a shared ViT backbone, and aggregated through masked sensor-set fusion. Stage 2 performs sensor-aware adaptation on $\mathcal { D } _ { \mathrm { q r y } } .$ , where the Stage 1 encoder is frozen and CLS-routed LoRA expert adapters provide lightweight input-conditioned adaptation. Classification uses learned masked fusion over available sensor CLS features, while segmentation uses available-sensor late fusion over aligned probability maps. In Stage 1, i indexes a training sample derived from an event, and in Stage 2, q indexes a scale-controlled query derived from an event.

## B. Sensor-Native Representation Learning

Stage 1 learns methane-aware representations from sensornative observations before imposing query-level geographic footprints. From the split event pool, we construct $\mathcal { D } _ { \mathrm { r e p } }$ by cropping a fixed number of pixels from each available sensor: 32×32 pixels for S2, L8/9, and EMIT, and 3×3 pixels for S5P. Because the sensors have different ground sample distances, these fixed-pixel crops correspond to different physical footprints. This design preserves the native spatial context exposed by each sensor instead of forcing all sensors into a shared geographic footprint. Each crop receives a binary methane label according to whether its geographic footprint overlaps a Carbon Mapper plume mask.

Each available observation $\mathbf { X } _ { i } ^ { s }$ is projected into a shared ViT token space while preserving its sensor-native spatial and spectral structure. Following a channel-to-patch design, we treat the temporal dimension as additional channel elements, apply a shared per-channel patchifier, inject sensor/channel metadata, and fuse channel tokens within each spatial patch:

$$
\mathbf { Z } _ { i } ^ { s } = \mathrm { A t t n } _ { \mathrm { c h } } \left( \operatorname { P a t c h i f y } _ { \psi } ( \mathbf { X } _ { i } ^ { s } ) + \mathbf { E } ^ { s } \right) \in \mathbb { R } ^ { N _ { i } ^ { s } \times D } .\tag{3}
$$

Here, $N _ { i } ^ { s }$ is the number of spatial patches and $D$ is the hidden dimension. The metadata embedding $\mathbf { E } ^ { s }$ encodes sensorspecific channel semantics, such as wavelength or spectralresponse information for optical and hyperspectral sensors, and product/channel identity for derived products such as S5P $\mathrm { C H } _ { 4 } .$ . A CLS token and sensor-specific positional embeddings are then added to form the ViT input sequence $\mathbf { U } _ { i , 0 } ^ { s } .$ Thus, heterogeneous sensors remain native before tokenization but share a common transformer interface after tokenization.

For each available sensor $s \in { \mathcal { M } } _ { i } ,$ , the shared ViT encoder produces a sensor-level CLS representation and patch-token features:

$$
\begin{array} { r } { [ \mathbf { c } _ { i } ^ { s } ; \mathbf { P } _ { i } ^ { s } ] = \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } _ { i , 0 } ^ { s } ) , \qquad s \in \mathcal { M } _ { i } . } \end{array}\tag{4}
$$

The encoder is shared across sensors, but each available sensor observation is encoded independently. Cross-sensor interaction is performed after encoding by aggregating the observed CLS representations.

To fuse the partially observed sensor set, MethaneFuse adds a learned sensor identity embedding $\mathbf { a } ^ { s }$ to each CLS representation and applies masked attention pooling (MAP) over the available sensors:

$$
\mathbf { z } _ { i } = \mathrm { M A P } \left( \{ \mathbf { c } _ { i } ^ { s } + \mathbf { a } ^ { s } \} _ { s \in \mathcal { M } _ { i } } \right) .\tag{5}
$$

The availability mask determines which sensor tokens enter fusion, allowing the same model to handle arbitrary observed subsets.

![](images/b0c4eafb03114386550bb9bb9fb983fc10da1c208521e160cbf0ce4d6551567a.jpg)  
Fig. 2. MethaneUnion dataset construction pipeline. Carbon Mapper plume reports are used as event anchors, and naturally available S2, L8/9, EMIT, and S5P observations are matched after sensor-specific preprocessing. Event-level splitting is applied before crop and query construction, producing two derived datasets: $\mathcal { D } _ { \mathrm { r e p } }$ for Stage 1 sensor-native representation learning and $\mathcal { D } _ { \mathrm { q r y } }$ for Stage 2 multi-footprint query-level classification and segmentation. The sensoravailability panel reports exclusive sensor-subset counts, highlighting the partial-observation structure of MethaneUnion.

The Stage 1 objective combines fused methane supervision with auxiliary sensor-level supervision:

$$
\mathcal { L } _ { \mathrm { r e p } } = \mathcal { L } _ { \mathrm { f u s e } } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } .\tag{6}
$$

Here, ${ \mathcal { L } } _ { \mathrm { f u s e } }$ is the cross-entropy loss from the fused prediction $h _ { \mathrm { f u s e } } ( \mathbf { z } _ { i } )$ , and $\mathcal { L } _ { \mathrm { a u x } }$ is the averaged sensor-level auxiliary cross-entropy over observed CLS representations. The auxiliary heads regularize sensor-level representation learning during Stage 1, while the fused head learns to predict from incomplete sensor sets.

## C. Sensor-Aware Adaptation for Query-Level Detection

Stage 2 adapts the Stage 1 representation to query-level methane classification and plume segmentation. From the same split event pool, we construct $\mathcal { D } _ { \mathrm { q r y } }$ , where each query is indexed by q and defined by

$$
q = ( i , t _ { q } , \ell _ { q } , r _ { q } ) ,\tag{7}
$$

where i is the source event index, $t _ { q }$ is the query timestamp, $\ell _ { q }$ is the query center, and $r _ { q }$ specifies the geographic footprint used to construct the query observation. Accordingly, Fig. 1 writes the query-level partial sensor set as $\{ ( \mathbf { X } _ { q } ^ { s } , m _ { q } ^ { s } ) \} _ { s \in \mathcal { S } } ,$ where $\mathbf { X } _ { q } ^ { s }$ is the sensor-s observation for query q and $m _ { q } ^ { s }$ indicates whether that observation is available; the observed query-level sensor subset is $\mathcal { M } _ { q } = \{ s \in \mathcal { S } : m _ { q } ^ { s } = 1 \}$ . For S2, L8/9, and EMIT, the sensor-specific crop size is determined by the corresponding ground sample distance, approximately $\lceil r _ { q } / \mathrm { G S D } ( s ) \rceil$ pixels per side. S5P is retained as a coarse contextual CH<sub>4</sub> view because its footprint is much larger than the plume-localization footprints considered in our evaluation.

Each query receives a binary classification label according to whether its geographic footprint overlaps a Carbon Mapper plume mask. Dense segmentation masks are generated by reprojecting the Carbon Mapper plume mask to the sensor grid for S2, L8/9, and EMIT; S5P is excluded from dense mask supervision because its coarse footprint does not support local plume delineation. In the experiments, we instantiate this query protocol with predefined evaluation footprints to assess detection performance under controlled spatial extents.

To adapt the learned representation, Stage 2 freezes the shared weights $\theta ^ { \star }$ and inserts trainable LoRA expert adapters into the query and value projections of each ViT block. Figure 3 summarizes the CLS-conditioned routing and expertresidual adaptation used in this stage.

For an input token sequence $\mathbf { X } ^ { ( \ell ) }$ at layer ℓ, the router first extracts the current CLS token

$$
\begin{array} { r } { \mathbf { g } ^ { ( \ell ) } = \mathrm { s o f t m a x } \left( \mathbf { W } _ { g } ^ { ( \ell ) } \mathbf { x } _ { \mathrm { c l s } } ^ { ( \ell ) } \right) , \qquad \mathbf { x } _ { \mathrm { c l s } } ^ { ( \ell ) } = \mathbf { X } _ { [ : , 0 , : ] } ^ { ( \ell ) } , } \end{array}\tag{8}
$$

where $\mathbf { g } ^ { ( \ell ) } \in \mathbb { R } ^ { K }$ gives the mixture weights over K LoRA experts. In our implementation, K is set to the number of supported sensors, but the router does not receive an explicit sensor ID. Instead, expert selection is conditioned on the current CLS representation, which reflects the sensor-native input, scene content, and intermediate transformer state.

For a frozen attention projection $\mathbf { W } \in \{ \mathbf { W } _ { Q } , \mathbf { W } _ { V } \}$ , the adapted projection is

$$
\begin{array} { c } { { \displaystyle { \bf W } { \bf x } \longrightarrow { \bf W } { \bf x } + \sum _ { k = 1 } ^ { K } g _ { k } ^ { ( \ell ) } \Delta _ { k } ^ { ( \ell ) } ( { \bf x } ) , } } \\ { { \Delta _ { k } ^ { ( \ell ) } ( { \bf x } ) = \displaystyle \frac { \alpha } { r } { \bf B } _ { k } ^ { ( \ell ) } { \bf A } _ { k } ^ { ( \ell ) } { \bf x } . } } \end{array}\tag{9}
$$

where r is the LoRA rank and α is the scaling factor. Here, ${ \bf A } _ { k } ^ { \left( \ell \right) }$ and $\mathbf { B } _ { k } ^ { ( \ell ) }$ are the down- and up-projection matrices of the k-th LoRA expert at layer ℓ. The base projection remains frozen, while the expert residuals provide lightweight input-conditioned adaptation. Although the number of experts matches the number of sensors, the experts are not hardassigned to sensors; they are softly selected by the CLS-token router.

![](images/853974eeff125f3347ac3201eb1aa405ce33560b148a5283185f120a3488e618.jpg)  
Fig. 3. CLS-conditioned LoRA expert adaptation used in Stage 2. The current CLS representation routes each input through a soft mixture of LoRA expert residuals, while the frozen attention projection remains unchanged.

After the adapted encoder, each available sensor produces an adapted CLS representation $\widehat { \mathbf { c } } _ { q } ^ { s }$ and adapted patch-token features $\widehat { \mathbf { P } } _ { q } ^ { s } .$ . For query-level classification, the adapted CLS representations are fused by masked attention pooling. At this fusion stage, a learned sensor identity embedding is added to each sensor-level CLS feature before attention aggregation:

$$
\widehat { \mathbf { z } } _ { q } = \mathrm { M A P } \left( \{ \widehat { \mathbf { c } } _ { q } ^ { s } + \mathbf { a } ^ { s } \} _ { s \in \mathcal { M } _ { q } } \right) .\tag{10}
$$

Thus, Stage 2 uses CLS-conditioned expert routing inside the frozen encoder and sensor-embedded masked fusion across available sensors.

For dense segmentation, a lightweight patch-token segmentation head is applied to sensors with meaningful plume-mask supervision, namely S2, L8/9, and EMIT. S5P is retained only as coarse atmospheric $\mathrm { C H _ { 4 } }$ context for classification because its footprint does not support local plume-boundary delineation. Segmentation fusion is performed as probabilitylevel late fusion: each available segmentation sensor first predicts an independent plume probability map, the maps are aligned to the Carbon Mapper plume-mask reference grid by geospatial reprojection when metadata is available or by resizing otherwise, and the aligned probabilities are averaged and thresholded at 0.5. This keeps classification fusion and segmentation fusion separate: classification uses learned masked fusion over adapted CLS representations, while segmentation uses late fusion over aligned per-sensor probability masks.

At inference time, MethaneFuse follows the same partialobservation protocol as training. Given a query anchor, only the naturally available sensors in $\mathcal { M } _ { q }$ are tokenized and encoded. For classification, the available sensor-level CLS representations are aggregated through learned masked sensorset fusion. For segmentation, MethaneFuse-Seg produces persensor probability maps for available S2, L8/9, and EMIT observations and applies the late-fusion procedure described above. The predefined query footprints are used as controlled evaluation settings, while the central inference problem remains predicting methane occurrence and plume support from naturally available heterogeneous observations.

## IV. EXPERIMENTS

## A. Experimental Setup and Baselines

Tasks and labels. For query-level classification, samples are labeled by overlap with the Carbon Mapper reference plume mask. Positive queries are those whose geographic footprint overlaps the reference plume mask. Negative queries are sampled from the same event-centered satellite acquisition but at least 5 km away from the reported plume location, and are retained only if their footprint has zero overlap with the reference plume mask at the evaluated scale. This provides local off-plume background samples from the same acquisition context while avoiding ambiguous plume-overlapping negatives. We use balanced positive and negative sampling, with approximately 50% positive samples.

Evaluation protocols and sample statistics. We use two event-level evaluation protocols. The primary protocol is a chronological split with May 16, 2025 as the cutoff date: Carbon Mapper plume reports on or before the cutoff are assigned to training, and later reports are assigned to testing. The split is applied before constructing sensor-specific crops, scalecontrolled queries, and segmentation masks, so all samples derived from the same plume report remain in the same partition. The query-level metadata contains approximately 625k generated samples under the default setting, with about 498k training samples and 126k testing samples, corresponding to an approximately 8/2 partition at the generated-sample level. As a complementary spatial-robustness protocol, we use a geoclustered split, where plume reports are grouped into spatial macro-regions and each region is assigned entirely to training or testing. The held-out test macro-regions are visualized in Fig. 7.

Metrics. Classification performance is measured by F1, accuracy, false positive rate (FPR), recall, and AUROC. Segmentation performance is measured by IoU+ following MethaneS2CM [17]. IoU+ uses standard IoU for non-empty reference plume masks, while assigning 1 to correctly predicted empty masks and 0 to false plume predictions on empty reference masks. We emphasize FPR and AUROC alongside F1 because methane plume classification is strongly affected by sensor-specific background artifacts and false alarms.

Baselines. We compare MethaneFuse with two controlled baseline groups on MethaneUnion. First, independent sensor predictors are trained separately for each sensor stream using the same sensor-native tokenization and ViT-style encoder design, and their available prediction scores are combined at inference time using majority voting, average score fusion, or logical OR. Second, we evaluate generic EO representation transfer by fine-tuning representative EO foundation models, including SatMAE, AnySat, and Panopticon, when their input interfaces can be adapted to the corresponding sensor products.

Implementation. Training was conducted on two NVIDIA A100 80GB GPUs. Stage 1 training required approximately 44 GPU-hours, and Stage 2 adaptation for the 480 m setting required approximately 8 GPU-hours. For Stage 2 adaptation, we used K = 4 experts, matching the number of supported sensors, with LoRA rank r = 8 and scaling factor α = 16 for all adapted query and value projections. MethaneFuse trains 137.54M parameters in Stage 1 and only 1.22M parameters in Stage 2. By comparison, the trainable parameter counts are 346.40M for the per-sensor ViT baseline, 342.52M for SatMAE-FT, 503.64M for AnySat-FT, and 395.92M for Panopticon-FT. These baselines use separate sensor-specific backbones, whereas MethaneFuse shares one backbone across sensors and only trains lightweight adapters during downstream adaptation.

## B. Main Query-Level Classification Results

Table I reports the main query-level classification results under naturally incomplete sensor availability at the 480 m footprint. Each query is evaluated using its available sensor subset, and multi-sensor predictions are produced either by heuristic score-level aggregation or by learned masked sensorset fusion. MethaneFuse is reported after Stage 2 adaptation, and the adaptation strategy is ablated in Table III.

The heuristic baselines show that score-level fusion can combine heterogeneous sensor predictions to some extent. Average score fusion gives the strongest heuristic result with 79.22 F1 and 85.32 AUROC, while Logical OR increases recall to 80.36 at the cost of a higher 26.96 FPR. Generic EO representation transfer is weaker in this methane-specific setting: Panopticon-FT reaches 77.65 F1 and 83.28 AU-ROC, while SatMAE-FT and AnySat-FT lag further behind. MethaneFuse achieves the strongest performance across all reported metrics, improving over average score fusion by +5.65 F1, +6.21 accuracy, +4.46 recall, and +8.30 AUROC while reducing FPR by 8.19 points. These results indicate that methane-supervised representation learning with learned sensor-set fusion is more effective than score-level fusion or generic EO transfer under partial sensor availability.

Figure 4 summarizes F1 across the four query footprints. MethaneFuse remains above the heuristic fusion baselines at 120 m, 360 m, 480 m, and 960 m, showing that the learned partial-observation representation is not tied to a single query scale.

TABLE I  
MAIN QUERY-LEVEL CLASSIFICATION UNDER PARTIAL SENSOR AVAILABILITY AT THE 480 M FOOTPRINT. VALUES ARE PERCENTAGES.
<table><tr><td>Method</td><td>Fusion</td><td>F1↑</td><td>Acc.↑</td><td>FPR↓</td><td>Recall↑</td><td>AUROC↑</td></tr><tr><td>Per-sensor ViT</td><td>Majority</td><td>78.21</td><td>77.11</td><td>23.14</td><td>77.33</td><td>77.81</td></tr><tr><td>Per-sensor ViT</td><td>Logical OR</td><td>78.72</td><td>76.93</td><td>26.96</td><td>80.36</td><td>84.89</td></tr><tr><td>Per-sensor ViT</td><td>Average</td><td>79.22</td><td>78.00</td><td>23.06</td><td>78.94</td><td>85.32</td></tr><tr><td>SatMAE-FT</td><td>Average</td><td>67.60</td><td>63.33</td><td>48.41</td><td>74.44</td><td>67.61</td></tr><tr><td>AnySat-FT</td><td>Average</td><td>58.90</td><td>58.96</td><td>37.53</td><td>55.82</td><td>62.48</td></tr><tr><td>Panopticon-FT</td><td>Average</td><td>77.65</td><td>75.61</td><td>29.13</td><td>79.80</td><td>83.28</td></tr><tr><td>MethaneFuse</td><td>Learned</td><td>84.87</td><td>84.21</td><td>14.87</td><td>83.40</td><td>93.62</td></tr></table>

![](images/6f0c0f0a020773f9f250a7d3de26664938f99add9c5fbcc068c5ef7bb4029aa9.jpg)  
Fig. 4. F1 sensitivity across predefined query footprints for MethaneFuse and baselines.

## C. Cross-Sensor Knowledge Transfer

We further test whether MethaneFuse transfers methanerelevant knowledge across sensor sets. For single-sensor regimes, the baseline is an independently trained per-sensor ViT. For multi-sensor regimes, the baseline averages the independently trained per-sensor ViT scores over the same available sensor group, while MethaneFuse uses learned masked sensor-set fusion and Stage 2 adaptation.

Figure 5 shows that MethaneFuse improves F1 across most availability regimes, rather than only in fully multi sensor settings. In the single-sensor rows, MethaneFuse improves all four streams, with the largest gain on EMIT, where F1 increases from 73.25 to 83.91. This suggests that the methane-supervised representation learned from irregular multi-sensor observations can also strengthen individual sensor streams. In the multi-sensor rows, MethaneFuse generally improves over average-score fusion, indicating that learned sensor-set fusion is more effective than treating independently trained sensor scores as interchangeable evidence. The strongest gains appear in several regimes involving weaker or coarser streams, showing that MethaneFuse can transfer useful methane-discriminative structure across heterogeneous observation subsets.

Figure 6 provides additional insight into false-positive control and ranking quality. MethaneFuse improves AUROC for all availability-conditioned groups, showing more consistent plume/background separation across sensor subsets. The FPR results are more regime-dependent: MethaneFuse substantially reduces false positives for several groups, especially EMIT and multiple S2-containing combinations, but does not reduce FPR uniformly for every availability regime. Together with the F1 results, these patterns indicate that MethaneFuse improves sensor-set transfer mainly by learning a more transferable methane representation and a more effective fusion rule, while still reflecting the noise, resolution, and product characteristics of each sensor group.

![](images/9c5a2782a8d0a9a94337372f3dbb73b7d8b57630dfb9399c73199fcdeecb9e71.jpg)  
Fig. 5. Sensor-set transfer at the 480 m footprint measured by F1. Rows are grouped into single-sensor, S2-present, and S2-absent regimes. L, E, and S denote L8/9, EMIT, and S5P.

![](images/ef796c7cde018810500ba114ba72de5630b7e3ccbb2c60562ff6701fe92c13dd.jpg)

![](images/efd807bf4c178fe2f7a53a420ec0ccb0630ffbd7462825b778af3d2a5cf775b4.jpg)  
Fig. 6. FPR and AUROC for the sensor-set transfer groups in Fig. 5. Bars compare the per-sensor ViT average-fusion baseline with MethaneFuse.

## D. Geographic Generalization under Geo-Cluster Splits

The primary task in this paper is event-level methane plume detection under naturally available satellite observations. Accordingly, the chronological split is the main evaluation protocol, because it reflects the operational setting of applying a detector to future plume events collected after the training period. In this setting, geographic context is part of the observable satellite evidence: methane-emitting facilities, surface backgrounds, acquisition conditions, and recurring source regions are spatially structured. A detector trained under a temporal split may therefore learn geographyassociated source-context and background cues in addition to plume-related spectral evidence.

We include the geo-cluster split as a complementary spatial robustness test. This protocol evaluates whether MethaneFuse can transfer to held-out macro-regions where facility context, surface materials, climate conditions, and sensor artifacts may differ from those seen during training. To construct this split, Haversine DBSCAN first groups nearby plume reports into tile-scale clusters and then merges them into 275 km macroregions. Each macro-region is assigned entirely to either training or testing, with held-out regions selected to contain approximately 20% of samples while preserving label balance and geographic diversity. This setting reduces geographic neighborhood overlap and provides a stricter domain-shift evaluation than the deployment-aligned chronological split.

![](images/05f3364795dfaf62509dce7dbb59f57734be745d6a36681313a6110e43611cfe.jpg)  
Fig. 7. Geo-cluster split for spatial robustness evaluation. Plume reports are grouped into tile-scale clusters and 275 km macro-regions, and each macroregion is assigned entirely to training or testing.

![](images/41d23beaf838f2be5d82b44f2e3f31f73c1f5454ce53aa5d8d96b576d8d515b7.jpg)

![](images/e83d834adfa4b694001f714594dc767a73938e34803e8357c08a8e979ece1f52.jpg)  
Fig. 8. Geographic split results under the spatially held-out protocol. The left panel shows F1 versus deployed inference-time parameters at 480 m, and the right panel shows MethaneFuse performance across query scales.

Figure 8 summarizes the geo-cluster split results. As expected, performance under spatially held-out macro-regions is lower than under the chronological split, reflecting the removal of region-specific source-context and background information. At the representative 480 m scale, MethaneFuse achieves the highest F1 while using substantially fewer inferencetime model parameters than both per-sensor ViT fusion and fine-tuned foundation-model baselines. Here, inference-time parameters count the deployed models used for prediction; for independent per-sensor fusion baselines, the parameters of the separately deployed sensor models are summed. The scalewise results further show stable performance across query scales, with the strongest results at intermediate scales of 360–480 m. These results suggest that MethaneFuse retains useful plume-discriminative structure under spatial domain shift, while the chronological split remains the deploymentaligned protocol for the main detection task.

TABLE II  
PLUME SEGMENTATION IOU+ (%) ACROSS QUERY SCALES. FUSION USES ALL AVAILABLE SEGMENTATION-CAPABLE SENSORS AMONG S2, L8/9, AND EMIT.
<table><tr><td>Scale</td><td>Model</td><td>S2</td><td>L8/9</td><td>EMIT</td><td>Fusion</td></tr><tr><td rowspan="3">120 m</td><td>U-Net</td><td>61.37</td><td>53.33</td><td>56.59</td><td>56.13</td></tr><tr><td>Per-sensor ViT</td><td>64.53</td><td>59.17</td><td>55.96</td><td>61.84</td></tr><tr><td>MethaneFuse-Seg</td><td>67.69</td><td>64.02</td><td>56.74</td><td>69.99</td></tr><tr><td rowspan="3">360 m</td><td>U-Net</td><td>52.70</td><td>46.58</td><td>56.35</td><td>56.24</td></tr><tr><td>Per-sensor ViT</td><td>61.28</td><td>56.37</td><td>56.61</td><td>57.69</td></tr><tr><td>MethaneFuse-Seg</td><td>68.11</td><td>64.04</td><td>57.25</td><td>60.56</td></tr><tr><td rowspan="3">480 m</td><td>U-Net</td><td>46.76</td><td>43.74</td><td>51.56</td><td>50.31</td></tr><tr><td>Per-sensor ViT</td><td>45.79</td><td>38.47</td><td>35.21</td><td>45.00</td></tr><tr><td>MethaneFuse-Seg</td><td>50.16</td><td>49.32</td><td>55.67</td><td>57.49</td></tr><tr><td rowspan="3">960 m</td><td>U-Net</td><td>41.33</td><td>41.06</td><td>50.72</td><td>34.43</td></tr><tr><td>Per-sensor ViT</td><td>26.48</td><td>35.02</td><td>49.49</td><td>37.56</td></tr><tr><td>MethaneFuse-Seg</td><td>45.53</td><td>43.16</td><td>53.10</td><td>45.78</td></tr></table>

## E. Plume Segmentation Results

Table II reports plume segmentation IoU+ (%) across query scales for S2, L8/9, EMIT, and available-sensor late fusion. The single-sensor columns evaluate predictions from the corresponding sensor when that sensor is available. The fusion column evaluates all samples with at least one segmentationcapable sensor among S2, L8/9, and EMIT. When only one segmentation-capable sensor is available, its predicted probability map is used directly; when multiple sensors are available, their probability maps are aligned to the Carbon Mapper plume-mask reference grid, averaged, and thresholded. The final prediction is compared with the Carbon Mapper plume mask on the same reference grid using IoU+. Because plume annotations are derived from satellite plume products with limited spatial precision, temporal mismatch, and sensordependent resolution, the segmentation labels should be interpreted as approximate plume support rather than exact plume boundaries.

MethaneFuse-Seg improves IoU+ over the U-Net baseline for most reported sensor-scale pairs, with the strongest gains on S2 and L8/9 at fine and medium query scales. The improvement on EMIT is more moderate but remains consistent across scales, reflecting the difficulty of plumeboundary localization under coarser and spectrally different observations. The late-fusion results show that probabilitylevel fusion can improve segmentation when multiple sensor views are available, especially at 120 m and 480 m. At 360 m and 960 m, fusion is less consistently better than the strongest individual sensor, suggesting that simple mean fusion remains affected by mask alignment uncertainty, resolution mismatch, and imperfect plume annotations.

## F. Ablation Study

We ablate the sensor-adaptive LoRA experts used in Stage 2. The Full FT baseline initializes from the Stage 1 encoder and performs full downstream fine-tuning of the encoder and heads. In contrast, MethaneFuse freezes the Stage 1 shared encoder and updates only the CLS-routed LoRA expert adapters, routing modules, and downstream heads. Table III reports the fusion-level results.

TABLE III  
FUSION-LEVEL ABLATION OF STAGE 2 ADAPTATION STRATEGIES. VALUES ARE PERCENTAGES.
<table><tr><td>Scale</td><td>Variant</td><td>F1↑</td><td>Acc.↑</td><td>FPR↓</td><td>Recall↑</td><td>AUROC↑</td></tr><tr><td rowspan="2">120 m</td><td>Full FT</td><td>80.23</td><td>79.32</td><td>21.10</td><td>79.70</td><td>88.37</td></tr><tr><td>MethaneFuse</td><td>80.96</td><td>80.15</td><td>19.87</td><td>80.17</td><td>87.44</td></tr><tr><td rowspan="2">360 m</td><td>Full FT</td><td>82.61</td><td>81.95</td><td>17.08</td><td>81.09</td><td>90.59</td></tr><tr><td>MethaneFuse</td><td>83.77</td><td>83.52</td><td>13.07</td><td>80.48</td><td>90.16</td></tr><tr><td rowspan="2">480 m</td><td>Full FT</td><td>83.50</td><td>82.63</td><td>17.51</td><td>82.75</td><td>90.89</td></tr><tr><td>MethaneFuse</td><td>84.87</td><td>84.21</td><td>14.87</td><td>83.40</td><td>93.62</td></tr><tr><td rowspan="2">960 m</td><td>Full FT</td><td>81.03</td><td>80.29</td><td>18.74</td><td>79.43</td><td>88.22</td></tr><tr><td>MethaneFuse</td><td>82.05</td><td>81.76</td><td>14.79</td><td>78.69</td><td>89.11</td></tr></table>

MethaneFuse improves F1, accuracy, and FPR over Full FT at all four query scales. The largest F1 gain occurs at 480 m (+1.37), where MethaneFuse also improves AUROC from 90.89 to 93.62. At 120 m and 360 m, Full FT has slightly higher AUROC, but MethaneFuse gives better F1 and lower FPR; at 960 m, MethaneFuse improves F1 by +1.02 and reduces FPR by 3.95 points. Together, these results support the proposed Stage 2 adaptation strategy as a controlled downstream update for false-positive suppression and scalecontrolled classification under heterogeneous partial sensor availability.

## V. DISCUSSION AND CONCLUSION

This paper studies methane plume detection from incomplete multi-sensor satellite observations, where different plume cases are captured by different subsets of public satellites rather than by complete multi-sensor measurements. We introduced MethaneUnion, a temporal multi-sensor dataset built from Carbon Mapper plume reports and matched S2, L8/9, EMIT, and S5P observations. Built on MethaneUnion, MethaneFuse learns methane plume cues from available sensor subsets through sensor-native representation learning and transfers them to plume classification and segmentation through lightweight sensor-aware adaptation. Across the main classification setting, sensor-availability tests, 120–960 m ground-region evaluations, geo-cluster split, and segmentation experiments, MethaneFuse consistently improves over independently trained sensor predictors, heuristic score fusion, and generic EO representation transfer. At the representative 480 m setting, MethaneFuse improves over the strongest baseline by 5.65 F1 points and 8.30 AUROC points while reducing false positives by 8.19 points. Sensor-set transfer results further show that MethaneFuse strengthens detection when S2 is available and transfers plume knowledge to L8/9, EMIT, and S5P when S2 is unavailable. These findings show that methane plume detection can move beyond S2-only pipelines by learning from incomplete but useful multi-sensor satellite observations.

Several limitations remain. MethaneUnion inherits spatial, temporal, and reporting biases from Carbon Mapper plume records and matched satellite acquisitions. Its plume masks provide approximate plume support rather than exact boundaries because of retrieval uncertainty, geolocation error, acquisition-time mismatch, wind-driven displacement, and sensor-dependent resolution. S5P is useful for coarse $\mathrm { C H _ { 4 } }$ classification context but cannot support plume-mask supervision at the evaluated localization scales. Future work will explore uncertainty-aware plume supervision, stronger temporal modeling, improved segmentation fusion, additional satellite products, and deployment-oriented calibration.

## ACKNOWLEDGMENT

This research received support from the Natural Sciences and Engineering Research Council of Canada (NSERC) Alliance Missions Grant (AMG 577118) and was enabled through collaboration with Enverus. The authors express their gratitude for the data provided by Enverus PRISM.

## REFERENCES

[1] United Nations Environment Programme and Climate and Clean Air Coalition, Global Methane Assessment: Benefits and Costs ofMitigating Methane Emissions. Nairobi, Kenya: UNEP, 2021. [Online]. Available: https://wedocs.unep.org/handle/20.500.11822/35913

[2] International Energy Agency, Global Methane Tracker 2025. Paris, France: IEA, 2025. [Online]. Available: https://www.iea.org/reports/ global-methane-tracker-2025

[3] D. J. Varon, D. Jervis, J. McKeever, I. Spence, D. Gains, and D. J. Jacob, “High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations,” Atmospheric Measurement Techniques, vol. 14, no. 4, pp. 2771–2785, 2021.

[4] J. Gorrono, D. J. Varon, I. Irakulis-Loitxate, and L. Guanter, “Un-˜ derstanding the potential of Sentinel-2 for monitoring methane point emissions,” Atmospheric Measurement Techniques, vol. 16, no. 1, pp. 89–107, 2023, doi: 10.5194/amt-16-89-2023.

[5] A. Vaughan, G. Mateo-Garc´ıa, L. Gomez-Chova, V. R´ u˚ziˇ cka, L. Guanter,ˇ and I. Irakulis-Loitxate, “CH4Net: A deep learning model for monitoring methane super-emitters with Sentinel-2 imagery,” Atmospheric Measurement Techniques, vol. 17, no. 9, pp. 2583–2593, 2024, doi: 10.5194/amt-17-2583-2024.

[6] B. Rouet-Leduc and C. Hulbert, “Automatic detection of methane emissions in multispectral satellite imagery using a vision transformer,” Nature Communications, vol. 15, Art. no. 3801, 2024, doi: 10.1038/s41467- 024-47754-y.

[7] A. Radman, M. Mahdianpari, D. J. Varon, and F. Mohammadimanesh, “S2MetNet: A novel dataset and deep learning benchmark for methane point source quantification using Sentinel-2 satellite imagery,” Remote Sensing of Environment, vol. 295, Art. no. 113708, 2023, doi: 10.1016/j.rse.2023.113708.

[8] Carbon Mapper, “Carbon Mapper data portal,” 2025. [Online]. Available: https://carbonmapper.org/data. Accessed: Jun. 2026.

[9] European Space Agency, Sentinel-2 User Handbook. Paris, France: ESA, 2015. [Online]. Available: https://sentinels.copernicus.eu/documents/ 247904/685211/Sentinel-2 User Handbook. Accessed: Jun. 2026.

[10] B. J. Schuit et al., “Automated detection and monitoring of methane super-emitters using satellite data,” Atmospheric Chemistry and Physics, vol. 23, no. 16, pp. 9071–9098, 2023, doi: 10.5194/acp-23-9071-2023.

[11] A. Apituley, M. Pedergnana, M. Sneep, J. P. Veefkind, D. Loyola, O. Hasekamp, A. Lorente Delgado, and T. Borsdorff, Sentinel-5 Precursor/TROPOMI Level 2 Product User Manual: Methane, document no. SRON-S5P-LEV2-MA-001, issue 2.4.0, 2022. [Online]. Available: https://sentinels.copernicus.eu/documents/247904/ 2474726/Sentinel-5P-Level-2-Product-User-Manual-Methane.pdf. Accessed: Jun. 4, 2026.

[12] J. Bian, J. Y. Leung, N. Volkmer, and J. Zheng, “An improved workflow in mass balance approach for estimating regional methane emission rate using satellite measurements,” Remote Sens. Earth Syst. Sci., vol. 8, pp. 1070–1083, 2025, doi: 10.1007/s41976-025-00237-0.

[13] J. Bian, J. Y. Leung, N. Volkmer, and J. Zheng, “A data analytics approach for unraveling the complexity of methane emissions: A Permian Basin study,” SPE Journal, vol. 30, no. 8, pp. 5104–5120, 2025, doi: 10.2118/228293-PA.

[14] V. Ru˚ziˇ cka, G. Mateo-Garcˇ ´ıa, L. Gomez-Chova, A. Vaughan, L. Guanter,´ and A. Markham, “Semantic segmentation of methane plumes with hyperspectral machine learning models,” Scientific Reports, vol. 13, Art. no. 19999, 2023, doi: 10.1038/s41598-023-44918-6.

[15] A. K. Thorpe et al., “Attribution of individual methane and carbon dioxide emission sources using EMIT observations from space,” Science Advances, vol. 9, no. 46, Art. no. eadh2391, 2023, doi: 10.1126/sciadv.adh2391.

[16] P. Joyce et al., “Using a deep neural network to detect methane point sources and quantify emissions from PRISMA hyperspectral satellite images,” Atmospheric Measurement Techniques, vol. 16, no. 11, pp. 2627–2640, 2023, doi: 10.5194/amt-16-2627-2023.

[17] H. Liu, J. Y. Leung, and D. Niu, “MethaneS2CM: A dataset for multispectral deep methane emission detection,” in Proc. ACM SIGKDD Conf. Knowledge Discovery and Data Mining, 2025, pp. 5640–5651.

[18] S. Zhao, Y. Zhang, S. Zhao, X. Wang, and D. J. Varon, “A data-efficient deep transfer learning framework for methane super-emitter detection in oil and gas fields using the Sentinel-2 satellite,” Atmospheric Chemistry and Physics, vol. 25, pp. 4035–4052, 2025.

[19] C. Aybar, J. Contreras, D. Montero, M. D. Mahecha, and L. Gomez-´ Chova, “MethaneSET: Unified multi-sensor datasets for satellite-based methane plume detection,” Hugging Face dataset, 2026. [Online]. Available: https://huggingface.co/datasets/tacofoundation/methaneset

[20] Y. Cong, S. Khanna, C. Meng, P. Liu, E. Rozi, Y. He, M. Burke, D. B. Lobell, and S. Ermon, “SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Proc. Adv. Neural Inf. Process. Syst., 2022.

[21] X. Guo et al., “SkySense: A multi-modal remote sensing foundation model towards universal interpretation for Earth observation imagery,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2024, pp. 27672–27683.

[22] B. Han, S. Zhang, X. Shi, and M. Reichstein, “Bridging remote sensors with multisensor geospatial foundation models,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2024, pp. 27852– 27862.

[23] G. Astruc, N. Gonthier, C. Mallet, and L. Landrieu, “AnySat: One Earth observation model for many resolutions, scales, and modalities,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2025, pp. 19530–19540.

[24] L. Waldmann, A. Shah, Y. Wang, N. Lehmann, A. J. Stewart, Z. Xiong, X. X. Zhu, S. Bauer, and J. Chuang, “Panopticon: Advancing any-sensor foundation models for Earth observation,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops, 2025, pp. 2229– 2239.

[25] M. Ma, J. Ren, L. Zhao, D. Testuggine, and X. Peng, “Are multimodal transformers robust to missing modality?” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2022, pp. 18177–18186.

[26] Y. Chen, M. Zhao, and L. Bruzzone, “A novel approach to incomplete multimodal learning for remote sensing data fusion,” IEEE Trans. Geosci. Remote Sens., vol. 62, Art. no. 5404914, pp. 1–14, 2024, doi: 10.1109/TGRS.2024.3387837.

[27] Y. Yang, J. Qu, L. Huang, and W. Dong, “DPMamba: Distillation prompt Mamba for multimodal remote sensing image classification with missing modalities,” in Proc. Int. Joint Conf. Artificial Intelligence, 2025, pp. 2224–2232, doi: 10.24963/ijcai.2025/248.

[28] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learning Representations, 2022.

[29] F. Marti Escofet, B. Blumenstiel, L. Scheibenreif, P. Fraccaro, and K. Schindler, “Fine-tune smarter, not harder: Parameter-efficient fine-tuning for geospatial foundation models,” in Machine Learning and Knowledge Discovery in Databases, Cham, Switzerland: Springer, 2026, pp. 516– 532, doi: 10.1007/978-3-032-06106-5 30.

[30] N. Shazeer et al., “Outrageously large neural networks: The sparselygated mixture-of-experts layer,” in Proc. Int. Conf. Learning Representations, 2017.

[31] X. Wu, S. Huang, and F. Wei, “Mixture of LoRA experts,” in Proc. Int. Conf. Learning Representations, 2024.

[32] C. Gao et al., “MoLA: MoE LoRA with layer-wise expert allocation,” in Findings Assoc. Comput. Linguistics: NAACL, 2025, pp. 5112–5127, doi: 10.18653/v1/2025.findings-naacl.284.

[33] H. Bi et al., “RingMoE: Mixture-of-modality-experts multi-modal foundation models for universal remote sensing image interpretation,” arXiv:2504.03166, 2025.

[34] Q. Gao et al., “Rethinking efficient mixture-of-experts for remote sensing modality-missing classification,” arXiv:2511.11460, 2025.

[35] U.S. Geological Survey, Landsat 8–9 Collection 2 Level-2 Science Product Guide. Sioux Falls, SD, USA: USGS, 2024. [Online]. Available: https://www.usgs.gov/landsat-missions/ landsat-collection-2-level-2-science-products. Accessed: Jun. 2026.

[36] R. O. Green, “EMIT L2A estimated surface reflectance and uncertainty and masks 60 m V001” [Data set]. NASA LP DAAC, 2022, doi: 10.5067/EMIT/EMITL2ARFL.001.

[37] E. H. Helmer and B. Ruefenacht, “A comparison of radiometric normalization methods when filling cloud gaps in Landsat imagery,” Canadian Journal of Remote Sensing, vol. 33, no. 4, pp. 325–340, 2007.