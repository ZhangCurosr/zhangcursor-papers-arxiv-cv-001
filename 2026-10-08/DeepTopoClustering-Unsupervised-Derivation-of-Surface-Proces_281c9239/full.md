# DeepTopoClustering: Unsupervised Derivation of Surface Process Taxonomy from 4D Point Clouds for Topographic Monitoring

Jiapan Wang<sup>a</sup>, Daan Hulskemper<sup>b</sup>, Mathilde Letard<sup>a</sup>, Roderik Lindenbergh<sup>b</sup>, Katharina Anders<sup>a</sup>

<sup>a</sup>Remote Sensing Applications, TUM School of Engineering and Design, Technical University of Munich, Munich, Germany <sup>b</sup>Department ofGeoscience & Remote Sensing, Delft University ofTechnology, Delft, The Netherlands

## Abstract

4D point clouds acquired by permanent laser scanning (PLS) enable accurate high-frequency monitoring of surface change in dynamic topographic environments. However, the resulting data volumes exceed the interpretive capacity of existent methods for identifying the underlying surface processes. Although these methods can detect and quantify where and when surface change occurs, they remain limited in organizing detected activities into meaningful process types. A sound methodology for identifying, comparing, and describing a variety of surface processes is therefore needed to support long-term topographic monitoring. To address this gap, we propose DeepTopoClustering (DTC), an unsupervised framework for deriving a hierarchical process taxonomy from object-based surface activities, socalled 4D objects-by-change (4D-OBCs). We first transform each 4D-OBC into a GeoMorphogram, a distributional sequence that represents the temporal evolution of topographic change within a spatially bounded surface activity. A convolutional autoencoder then learns latent embeddings from GeoMorphograms, which are then jointly optimized using a hierarchical deep clustering objective to organize surface activities into a hierarchy. We evaluate the learned hierarchy using expert annotations on two separate 4D datasets of sandy beach sites and their combination. Our results show that DTC with GeoMorphograms achieves the highest agreement with expert judgment at the taxonomy level comprising eight major process types (F = 0 78, match accuracy = 0 92), outperforming notably dimensionality reduction and conventional flat clustering, which are recent alternative methods applied to process similar time series data. The learned taxonomy separates major erosion- and deposition-dominated activities and further distinguishes finer subtypes with interpretable characteristics related to change magnitude, duration, compactness, and temporal evolution. These results demonstrate that DTC provides a scalable and interpretable route from 4D change detection to a data-driven, expert-supported surface process taxonomy, thereby advancing automated knowledge derivation and organization for understanding surface dynamics in topographic monitoring.

Keywords: Permanent laser scanning, Surface process understanding, Deep hierarchical clustering, Natural environments, Coastal monitoring, Change analysis

## 1. Introduction

Topographic landscapes are typically shaped by numerous surface processes that often operate under natural and anthropogenic forcing across a wide range of spatial and temporal scales (de Boer, 1992; Huggett, 2016). On sandy beaches, for example, storms trigger rapid erosion, waves and tides redistribute sediment, wind produces localized deposition and erosion patterns, and human activities modify the surface through intervention or recreational use (Kuschnerus et al., 2021, 2024a; Hulskemper et al., 2026b). Most of these processes are not directly observable. Instead, they have to be inferred from measured topographic change, that is, spatiotemporal elevation diferences derived from repeated surface observations. Many existent studies focus on detecting specific processes or landforms, such as rockfalls, landslides, and soil erosion (Williams et al., 2018; Kromer et al., 2017; Eltner et al., 2018). However, increasingly long-term monitoring offers a broader systematic perspective, in which the full range of observed surface dynamics is organized into interpretable surface process types. Such a perspective is necessary not only for describing individual processes but also for comparing recurring activities, identifying dominant process mechanisms, and ultimately understanding interactions among diferent surface processes. The key challenge is therefore to move beyond the measurement of topographic change and toward the derivation of a surface process taxonomy from spatiotemporal topographic data (Eitel et al., 2016; Woodcock et al., 2020).

Modern LiDAR techniques provide an efective basis for acquiring high-resolution 3D information of topographic surfaces across natural environments (Irwin et al., 2025). In particular, permanent laser scanning (PLS) systems can now generate dense 4D point clouds at hourly to seasonal timescales (Eitel et al., 2016; Lindenbergh et al., 2025). Such high-frequency 4D data create new opportunities for environmental monitoring of coastal areas, rivers, landslides, glaciers, and other dynamically evolving landscapes (Kromer et al., 2017; Williams et al., 2018; O’Dea et al., 2019; Woodcock et al., 2020). Recent studies have investigated the detection and quantification of topographic change from 4D point clouds. Methods such as robust co-registration (Yang and Schwieger, 2023; Yang and Holst, 2025), uncertainty-aware distance estimation (Lague et al., 2013; Winiwarter et al., 2021), and object-based spatiotemporal segmentation (Anders et al., 2020, 2021) have advanced the reliable detection of spatiotemporal surface activities from 4D point clouds. In particular, 4D objects-by-change (4D-OBCs) provide an objectbased approach for extracting and analyzing individual surface activities. These advances have enabled the detection of where and when surface activities occur and the extraction of large numbers of surface activities from point cloud time series. Nevertheless, they represent an unordered collection of potentially large varieties of surface process types.

For large-scale and long-term monitoring, as demonstrated for sandy beach environments (Vos et al., 2022, 2023), Anders et al. (2020) detect thousands of 4D-OBCs. To support geomorphic analysis and environmental management, a successive step requires organizing these surface activities into a meaningful structure that provides an overview of recurring surface process types. This step is similar to deriving a taxonomy, which provides an organized system for comparing complex instances, identifying common types, and describing variation within and between groups (Godfray, 2002). A surface process taxonomy would allow surface activities to be systematically compared across space and time, support the identification of dominant and rare process types, and provide a basis for posthoc analysis of process interactions and environmental drivers. However, deriving such a taxonomy for surface processes remains challenging because benchmark labeled datasets are not available, the number of surface process types is unknown a priori, and can vary between geographic settings. At the same time, surface activities often vary substantially in spatial extent, duration, geometry, magnitude, and internal dynamics (Wang and Anders, 2025).

Recent studies have addressed this interpretation gap by combining 4D-OBCs with descriptive features and unsupervised clustering. For example, handcrafted descriptors derived from segmented 4D-OBCs have been used with Self-Organizing Maps (SOMs) to group similar surface activities and support their interpretation in relation to potential driving forces, such as wind and tidal records (Hulskemper et al., 2026a,b). Related workflows based on handcrafted descriptors and static clustering have also provided useful summaries of coastal change patterns (Kuschnerus et al., 2021; Hulskemper et al., 2022). These studies demonstrate that 4D-OBCs are meaningful units for surface activity analysis and that clustering analysis can support the interpretation of surface process types. However, the representation of each surface activity still depends on manually designed features and requires domain knowledge, typically restricted to a single geographic setting. This limits scalability and flexibility, in particular when the relevant process properties are not known in advance. These current challenges motivate us to develop a reverse engineering approach that can summarize and extract domain knowledge, i.e., process properties, from the observed surface activities. A data-driven approach is therefore needed to learn process-relevant representations directly from detected 4D-OBCs and to organize them into a hierarchical structure without predefined labels or prior knowledge of the number of surface process types.

To address this, we introduce DeepTopoClustering (DTC), an unsupervised framework for deriving hierarchical surface process taxonomies from 4D point cloud data at the level of individual surface activities. The framework first represents each detected 4D-OBC as a so-called GeoMorphogram, which we develop as a distributional sequence describing the temporal evolution of topographic change within a local region (i.e., the spatial extent of a 4D-OBC). Based on these Geo-Morphograms, we adapt a deep clustering-based model, following Mautz et al. (2020), to jointly learn latent embeddings and a hierarchical clustering tree that separates activities into groups from coarse to fine levels of detail. We apply the proposed method to realworld 4D datasets acquired from two separate sandy beaches (Vos et al., 2022, 2023), where several thousand 4D-OBCs were automatically extracted (Anders et al., 2021; Hulskemper et al., 2026a). The resulting surface process taxonomy is evaluated using expert annotations that quantify the agreement between learned clusters and human interpretation. The main contributions of this research are the following:

• We present an end-to-end, object-centric pipeline that transforms 4D point clouds into an interpretable, hierarchical taxonomy of surface process types, addressing the gap between 4D topographic change detection and surface process understanding.

• We introduce GeoMorphograms as standardized representations for modeling individual 4D-OBCs as comparable distributional sequences, enabling discrimination of surface activities while accounting for irregular geometric properties, varying duration, and inherent noise within 4D-OBCs.

• We develop DeepTopoClustering as an unsupervised hierarchical deep clustering framework that derives surface activity clusters without requiring known labels or a predefined number of clusters.

• We derive the first data-driven process taxonomy of sandy beach surface processes from 4D point cloud data.

• We present an expert evaluation strategy for validating hierarchical clustering results and determining an optimal number of clusters in the absence of ground-truth labels.

## 2. Related work

The advanced technical capabilities of acquiring multi-temporal point clouds from LiDAR and photogrammetry have enabled near-continuous observation of surface dynamics. This has led to a shift from static 3D analysis to 4D point cloud time series, where topographic information is observed repeatedly over time at high frequency with respect to the observed phenomena (Eitel et al., 2016; Qin et al., 2016; Woodcock et al., 2020; Lindenbergh et al., 2025). Under such paradigm transition, recent research has developed along a connected direction: automatically acquiring 3D data nearcontinuously over time and aligning them in a common coordinate reference system, detecting and quantifying significant topographic change with associated uncertainty between epochs, and interpreting change information in terms of underlying surface processes (Lindenbergh et al., 2025). While the first two steps are extensively developed, the transition from change detection to process-level understanding remains limited.

In the following, we review existing work along this direction.

Change detection and quantification. Quantifying topographic change between 3D data acquired at two successive epochs is a central topic in topographic monitoring (Qin et al., 2016). A standard approach is Multiscale Model-to-Model Cloud Comparison (M3C2) (Lague et al., 2013), which estimates a spatially averaged distance directly between bitemporal point clouds along locally determined surface normals. By avoiding the transformation of the unstructured point cloud into a mesh or grid, M3C2 provides robust change estimates in rough topography and heterogeneous sampling under local planarity assumptions. Several variants have been proposed to improve accuracy and adapt M3C2 to specific surface conditions, e.g., correspondence-driven plane-based M3C2 (Zahs et al., 2022) and M3C2 with error propagation (Winiwarter et al., 2021).

Beyond bitemporal comparison, the full temporal dimension of 4D datasets has been incorporated into change analysis. Kalman filtering (Winiwarter et al., 2023) is used to account for spatiotemporal uncertainties, reduce noise, and handle missing data across multiple epochs. To jointly exploit spatial and temporal coherence, object-centric change analysis has been introduced with the 4D objects-by-change (4D-OBCs) method (Anders et al., 2020). 4D-OBCs delineate the spatial and temporal extent of surface activities by grouping spatially connected regions with similar temporal behavior. Each 4D-OBC is automatically extracted from 4D point clouds and represents a coherent surface activity that evolves over time, without predefined temporal windows (Anders et al., 2021). This allows surface activity to be analyzed at the occurrence level rather than over the full point cloud time series. Despite these advancements, most existing methods focus on accurately estimating where, when, and how much topographic change occurs. They provide a necessary foundation for 4D topographic monitoring, but do not directly support higher-level interpretation in terms of which process type the detected surface activities may represent.

From change analysis to characterization. Building on topographic change quantification in nearcontinuous time series, the nature of topographic changes can be characterized with respect to temporal behavior. A straightforward approach is to apply clustering algorithms to identify temporal patterns. For example, diferent clustering methods, such as k-means, agglomerative clustering, and DBSCAN, have been applied to derive meaningful patterns from point cloud time series for sandy beaches (Kuschnerus et al., 2021), past-glacial valleys (Wang et al., 2025b), vegetation (Puttonen et al., 2019), and landslides (Weidner et al., 2021). These studies show that unsupervised clustering can derive recurring change patterns, but they are commonly applied to point-wise or grid-wise full time series and therefore remain tied to fixed spatial resolutions rather than spatiotemporally variable surface activities.

Object-centric approaches address this limitation by using 4D Objects-by-Change (4D-OBCs) as natural units for describing surface activities, related to ero sion, deposition, and redistribution processes (Anders et al., 2020). Because 4D-OBCs vary in spatial extent, duration, and internal geometry, direct comparison between activities is dificult for machine learning methods that require standardized inputs. Recent research has addressed this by representing each 4D-OBC using handcrafted descriptors and resampled time series representations. Temporal descriptors, such as duration and timing of change magnitude, and geometric descriptors, such as location, area, and volume change, can be de rived to summarize each surface activity. Time series include seed time series (Hulskemper et al., 2022) or multivariate statistical time series (Hulskemper et al., 2026b), to enable clustering or classification. While these representations have shown value for interpreting geomorphic phenomena, they also introduce limitations: single time series are location-dependent and do not capture the full spatial variability of the 4D-OBC; data gaps and noise can reduce robustness; and resampling may cause information loss about the temporal shape of the activities. In addition, handcrafted descriptors depend on prior assumptions about which properties are relevant, while a generalized definition of process types is not available, even when considering domain expertise, which is limited to known processes. Handcrafted descriptors may overlook complex or unexpected patterns that are not known to analysts in advance. Overall, existing characterization workflows remain limited in their ability to compare diverse surface activities and to derive a systematic surface process taxonomy from long-term 4D topographic monitoring data.

## Deep clustering for unsupervised class derivation.

Recent advances in deep learning have shown strong potential for learning representations and discovering patterns from high-dimensional data, including spatiotemporal data, in an unsupervised manner (Bengio et al., 2013; Wang et al., 2022). In particular, deep clustering methods jointly learn latent embeddings and cluster assignments, enabling data-driven automatic clustering without requiring labeled data as input. Representative approaches include DeepCluster (Caron et al., 2018), which alternates between k-means clustering on extracted features and updating the network using pseudolabels, and Deep Embedded Clustering (DEC; Xie et al., 2016), which refines latent embeddings through iterative clustering with soft-assignments. Several variants of deep clustering methods have demonstrated efectiveness in image and 3D data analysis, especially when labeled data are limited or expensive to obtain (Zhou et al., 2024; de Gélis et al., 2023; Liu et al., 2024).

In remote sensing, similar concepts have been explored to address the lack of annotated datasets in realworld scenes. For example, DeepCropClustering (Wang et al., 2025a) combines contrastive learning and deep clustering using nearest-farthest neighbors to derive different crop types from satellite imagery. DC3DCD (de Gélis et al., 2023) applies the DeepCluster principle to 3D change detection, resulting in diferent change types by manually mapping pseudo-labels derived from bitemporal point clouds. In its first application to multi temporal 3D change analysis, DEC was applied to time series derived from 4D-OBCs to characterize surface activities within erosion and deposition as two separate categories (Wang and Anders, 2025). Existent work indicates that deep clustering can extract useful latent features from spatiotemporal data and support change analysis without manual labels. However, most existing approaches use flat clustering and require the number of clusters to be predefined. This is a major limitation for deriving surface process types, where their number is usually unknown a prior and where process categories may be more adequately represented as a hierarchy than as flat clusters. In a hierarchy, higher levels may share the same drivers and overall types; lower levels may be separated by fine-grained properties.

To address these limitations, recent methods have explored hierarchical or flexible clustering structures that do not require a fixed number of clusters in advance. Examples include DeepECT (Mautz et al., 2020), deep taxonomic networks (Wang et al., 2025c), and Deep-DPM (Ronen et al., 2022). Such approaches are particularly relevant for comprehensive topographic monitoring where the underlying category structure is unknown, ambiguous, or multi-level rather than being specifically defined by analysts. However, no existing framework integrates object-centric 4D change analysis, representation learning, and hierarchical deep clustering to derive generic interpretable surface process types. This motivates us to develop an unsupervised deep clustering approach to learn representations of surface activities and organize them into a structured taxonomy without prior assumptions on the number or properties of process types.

## 3. Data and methods

We develop a three-stage framework for deriving a data-driven surface process taxonomy from nearcontinuous 4D point clouds. The first stage is Process Representation: surface activities are extracted as 4D objects-by-change (4D-OBCs), which define discrete local surface activities evolving over time. Each 4D-OBC is then represented as a GeoMorphogram (Section 3.2), a distributional sequence that describes the temporal evolution of topographic change within a local region. The second stage is DeepTopoClustering: these GeoMorphograms are embedded into a compact feature space during a pretraining phase and are subsequently clustered using DTC (Section 3.3), a hierarchical deep clustering approach that organizes similar surface activities into a divisive tree structure and reveals multiple levels of process similarity. The third stage is Process Interpretation: the learned hierarchical linkage is evaluated on 4D test datasets (Section 3.1) using expert annotations to quantitatively assess and qualitatively interpret the derived surface process taxonomy (Section 3.4). The main steps of our framework are illustrated in Figure 1, and details on data and methods are outlined in the following subsections.

## 3.1. Data and preprocessing

For our study, we use two 4D point cloud datasets acquired with PLS systems installed at the sandy beach of Kijkduin (Vos et al., 2022) and Noordwijk (Vos et al., 2023) in the Netherlands (Figure 2). At each site, a Riegl VZ2000 laser scanner was mounted on a stable frame on a hotel roof overlooking the study areas. Kijkduin (52.07°N, 4.22°E) was monitored during the winter and spring of 2016 - 2017 for approximately six months, and Noordwijk (52.24°N, 4.42°E) was monitored between July 2019 and July 2022 for about three years. Hourly laser scans were collected over the sandy beach, including its dunes, backshore, berm, and intertidal zones, resulting in thousands to tens of thousands of point cloud epochs. Details on the monitoring setup and data characteristics are provided in Vos et al. (2022) and Vos et al. (2023), and a summary is presented in Table 1. The resulting near-continuous point cloud time series are subsequently processed to derive spatiotemporal elevation change and extract discrete activity objects as 4D-OBCs.

Table 1: Data summary of the two open source PLS datasets (i.e., Kijkduin and Noordwijk sandy beaches) used in this research.
<table><tr><td>Dataset</td><td>Kijkduin (NL)</td><td>Noordwijk (NL)</td></tr><tr><td>Location</td><td>52.07°N,4.22°E</td><td>52.24°N, 4.42°E</td></tr><tr><td>Extent</td><td>300 × 600 m</td><td>350 × 1000 m</td></tr><tr><td>Time interval</td><td>1 hour</td><td>1 hour</td></tr><tr><td>Time range</td><td>Nov. 2016 – May 2017</td><td>July 2019 – July 2022</td></tr><tr><td>Epochs</td><td>2,942</td><td>21,194</td></tr></table>

The PLS datasets are processed through several typical steps, including co-registration, change quantification, and surface activity extraction (Lindenbergh et al., 2025). We use the resulting 4D-OBCs extracted in previous work from the Kijkduin (Anders et al., 2021) and Noordwijk (Hulskemper et al., 2026b) 4D datasets. Each 4D-OBC represents an individual surface activity derived from the topographic change (here, M3C2 distances (Lague et al., 2013)) and serves as a fundamental unit for subsequent representation learning and clustering. In total, three datasets are considered: the Kijkduin dataset with 2,021 4D-OBCs, the Noordwijk dataset with 4,654 4D-OBCs, and a merged sandy beach activities dataset, referred to as BeachOBC, with 6,675 4D-OBCs (i.e., the sum of both). Each dataset is partitioned by time of occurrence: all 4D-OBCs are grouped by month using the middle of their time span, and the training and unseen subsets are randomly split at an 80-20 ratio within each month. This strategy ensures that the model is trained to capture full process types over the entire monitoring period and evaluated on pre viously unseen surface activities.

## 3.2. GeoMorphogram: a distributional sequence representation of a surface activity

To represent the full surface activity rather than selected summaries from a single time series, we introduce the GeoMorphogram, a distributional sequence representation that describes the temporal evolution of topographic change within each surface activity. A GeoMorphogram represents a 4D-OBC as an ordered sequence of topographic change distributions over time. At each epoch, topographic change values are extracted within the spatial extent of the 4D-OBC and aggregated into a histogram. This produces a 2-dimensional representation in which the horizontal axis denotes time and the vertical axis denotes topographic change magnitude. In contrast to time series summaries based on mean, median, or other object-level statistics, the Geo-Morphogram preserves the distributional state of a surface activity at each epoch, including both the magnitude and spread of change.

![](images/b555ef6b2684bbb3794c4cc67e87149e502eea7db901e902344f08ba5a5c25de.jpg)  
Figure 1: Overall research framework for deriving surface process taxonomy from 4D point clouds, including representing extracted 4D surface activities as GeoMorphograms, organizing them into learnable hierarchical clusters, and validating the learned taxonomy on 4D test datasets with expert annotations.

To construct a GeoMorphogram, let $O _ { i }$ denote a 4D-OBC, representing a surface activity occurring within spatial extent $\Omega _ { i } ~ \in ~ \mathbb { R } ^ { 2 }$ and temporal duration $T _ { i } ~ =$ $\{ t _ { 1 } , \ldots , t _ { D } \}$ . For each epoch $t _ { d } .$ , the topographic change within the 4D-OBC is represented by a set of topographic change values, measured relative to the surface state at the first epoch $t _ { 1 } .$ , at points $p \mathrm { : }$

$$
\{ \Delta z ( p , t _ { d } ) \mid p \in \Omega _ { i } \} .
$$

A histogram function $\mathcal { H } ( \cdot )$ is then applied to aggregate these values into a discrete distribution over B bins:

$$
h _ { i } ( t _ { d } ) = \mathcal { H } ( \{ \Delta z ( p , t _ { d } ) \mid p \in \Omega _ { i } \} ) \in \mathbb { R } ^ { B } ;
$$

The histogram is normalized to represent a probability

![](images/93fb8a1c102c645dba1ca63482dd8bf38a5d03ba9ef86d39220a8e20bdb8d71a.jpg)  
Figure 2: Permanent laser scanning point cloud of the study areas in Noordwijk and Kijkduin sandy beach. The terrestrial laser scans are colorized using high-resolution aerial imagery in 2021 (Noordwijk) and in 2016 (Kijkduin). The aerial imagery and administrative areas are provided by the Public Service On the Map (PDOK) in the Netherlands.

distribution:

$$
\sum _ { b = 1 } ^ { B } h _ { i } ^ { ( b ) } ( t _ { d } ) = 1 .
$$

This object-wise normalization reduces the influence of spatial variability, such as diferences in object size, because each epoch is represented by relative frequencies rather than absolute point counts. By encoding frequency distributions instead of raw topographic change magnitudes, the GeoMorphogram emphasizes change states and their temporal evolution rather than absolute topographic diferences. This facilitates consistent comparison among surface activities with diferent spatial scales and geometries.

The GeoMorphogram of object $O _ { i }$ is then defined as the ordered sequence of the topographic change distribution over time duration $D \colon$

$$
\mathbf { G } _ { i } = [ h _ { i } ( t _ { 1 } ) , \ldots , h _ { i } ( t _ { D } ) ] \in \mathbb { R } ^ { B \times D }\tag{1}
$$

where each element represents the occurrence frequency of a given change magnitude at a specific epoch. GeoMorphograms are constructed using fixed discretization parameters to ensure comparability across samples. In this study, the vertical axis is discretized using a bin resolution of 0.025 m over the range [−0 5 m 0 5 m], based on the 95% range ([−0 25 m 0 36 m]) of topographic change values in the datasets. The defined boundaries cover the major range of change values with a margin while keeping GeoMorphograms not too large, and the bin resolution is determined with respect to the minimum detectable change for the datasets (Kuschnerus et al., 2024b). Two additional bins are included at both ends to capture values outside this range, resulting in B = 42 bins. This configuration balances the need to retain meaningful variation in topographic change magnitude with consideration of measurement noise. The horizontal time axis is defined from the start of each activity and discretized at an hourly resolution up to a maximum duration of D = 1024 hours. This duration covers 97% (904 at 95%) of all 4D-OBCs while limiting excessive padding. 4D-OBCs shorter than this maximum duration are zeropadded at the end, whereas longer samples, which account for 3% of the dataset, are downsampled to 1024 epochs to preserve long-term deformation behavior. All GeoMorphograms are translated relative to the initial state of each 4D-OBC and represented as normalized probability distributions. These choices yield GeoMorphograms with consistent dimensions, enabling direct comparison among surface activities with diferent spatial extents and temporal durations.

(a)  
![](images/6008c74d9df65fa8e4a2697008576543d20015dccc614b1a330a90e8e50f5228.jpg)

![](images/68929d71418b64a3fa3df8b05441954ea429ba3cec1d598dc8d9927705ec63bd.jpg)  
Time(c)

(b)  
![](images/fd7720131313274ede710569ebc0d117da39b1b3e24ed19d0b4afcabd4aaba15.jpg)

![](images/128b5df6cffc4d62272fd73f2774a88637aaadedb44bbb24f2079ecb90b4875f.jpg)

![](images/66c7304b6a99c3731f8dd526900846f80c1a5963585ce2f28f5135f36a1faff7.jpg)

(d)  
![](images/3dea2261d5cdd4ee48ef7bf550d1fcf8cdd66dbfdd6c974ece76e964e7ae6ac0.jpg)  
Figure 3: Examples of 4D objects-by-change (4D-OBCs) and their corresponding GeoMorphograms. Subplots (a) and (c) show erosion- and deposition-dominated 4D-OBCs, respectively. For each object, the upper map shows the spatial extent with core points colored by their peak change magnitude and the seed point marked in red. The lower plot shows the change time series of all core points within the object, colored by their peak change magnitude; the red line indicates the seed time series and the gray dotted line marks the peak change epoch. Subplots (b) and (d) show the corresponding GeoMorphograms for the erosion 4D-OBC (a) and the deposition 4D-OBC (c), respectively. Color intensities show the distribution of change values at each epoch, and the activity duration is marked using a red dashed vertical line.

The resulting GeoMorphogram encodes both temporal dynamics and distributional variability within each 4D-OBC. For example, Figure 3 shows that an erosiondominated activity (a) produces temporal distributions (b) that are consistently skewed toward negative topographic changes, whereas a deposition-dominated activity (c) generates temporal distributions (d) concentrated in positive change bins. Although most time series broadly follow the trend of the seed time series, the GeoMorphograms also reveal wider or bimodal distributions across epochs. This reflects that sediment may be redistributed in a complex manner even within a single 4D-OBC.

## 3.3. DeepTopoClustering for hierarchical grouping

We use the Geomorphograms as a standardized representation of surface activities and map them into a latent space, where activities with similar temporal distributional change patterns are expected to have similar embeddings. To learn this representation in an unsupervised manner, the latent embeddings are first learned through self-reconstruction using an autoencoder (AE) (Bengio et al., 2013). Because the number of surface process types is also unknown a priori, the clustering framework should not require a predefined number of clusters and should allow the clustering structure to evolve during training. We therefore develop Deep-TopoClustering (DTC), a hierarchical deep clustering framework adapted from the Deep Embedded Clustering Tree (DeepECT) method (Mautz et al., 2020) for spatiotemporal topographic change analysis.

DTC consists of two main components: (1) a feature extraction module that learns latent embeddings from GeoMorphograms, and (2) a hierarchical clustering module that organizes these embeddings into a binary tree through iterative node splitting. In contrast to a flat clustering method, this structure allows broad differences among surface activities to be represented at shallow tree levels and finer variations to be represented at deeper levels. In our framework, DeepECT is adapted to operate on GeoMorphograms, to jointly optimize representation learning and hierarchical clustering, and to decode node centers into GeoMorphogram prototypes for interpretation. The learning process iteratively refines both the embedding space and the tree structure, enabling surface activities to be grouped at multiple levels of detail.

Feature extractor. We design an autoencoder-based feature extractor consisting of a four-layer convolutional neural network (CNN) as the encoder $f _ { \theta } ( \cdot )$ and a mirrored transposed CNN as the decoder $g _ { \theta } ( \cdot )$ This design treats the GeoMorphogram as a structured temporal-distributional image rather than a time series representation, allowing the model to capture local patterns in both temporal and spatial dimensions of topographic changes. The learnable parameters θ are randomly initialized and optimized during training.

Given a GeoMorphogram $\mathbf { G } _ { i } ~ \in ~ \mathbb { R } ^ { B \times D }$ , the encoder maps the input into a latent embedding with a fixed dimension of 16:

$$
z _ { i } = f _ { \theta } ( \mathbf { G } _ { i } ) .\tag{2}
$$

The decoder then reconstructs the input from the embedding:

$$
\tilde { \mathbf { G } } _ { i } = g _ { \theta } ( z _ { i } ) .\tag{3}
$$

Because GeoMorphograms represent normalized distribution sequences and may contain padded epochs beyond the valid duration of a surface activity, reconstruction is optimized using a masked Kullback–Leibler (KL) divergence loss. Let $\overline { { h } } _ { i } ^ { ( b ) } ( t _ { j } )$ and $\tilde { h } _ { i } ^ { ( b ) } ( t _ { j } )$ denote the true and the reconstructed probabilities for bin b at time $t _ { j } ,$ , respectively. The reconstruction loss is defined as

$$
\mathcal { L } _ { R E C } = \frac { 1 } { | S | } \sum _ { i = 1 } ^ { S } [ \frac { 1 } { m _ { i } } \sum _ { j = 1 } ^ { m _ { i } } \sum _ { b = 1 } ^ { B } h _ { i } ^ { ( b ) } ( t _ { j } ) \log ( \frac { h _ { i } ^ { ( b ) } ( t _ { j } ) } { \widetilde { h } _ { i } ^ { ( b ) } ( t _ { j } ) } ) ] ,\tag{4}
$$

where $m _ { i } \le D$ is the valid duration of object $i , D$ is the defined maximum duration across all objects, B is the number of topographic change bins, and S is the set of training samples. The masking ensures that only valid epochs contribute to the reconstruction objective. Minimizing the reconstruction loss $\mathcal { L } _ { R E C }$ encourages the latent space to preserve the main temporal-distributional patterns of surface activities while reducing the dimensionality of the input. This pre-training phase aims to establish a meaningful autoencoder and feature space for subsequent hierarchical clustering, as shown in stage 2 of Figure 1.

Hierarchical clustering. Given the latent embeddings, we implement the clustering module following the hierarchical clustering strategy of DeepECT. It constructs a binary tree, where each node is either a leaf node or a split node. The split node has two child nodes, which form a sibling pair and represent two subclusters of the samples assigned to their parent. Each node is associated with a vector $\mu _ { n }$ in the latent space, which serves as the node center. The tree is initialized with a single root node containing all samples and grows progressively by splitting heterogeneous leaf nodes during optimization.

While the tree is growing, the leaf node with the highest internal variability is selected to split. Internal variability is measured by the sum of squared distances between the node center and the embeddings assigned to that node. The selected node is divided into two child nodes by applying k-means++ with $k \ = \ 2$ to the assigned embeddings. We use k-means++ as it is considered more accurate and faster than standard k-means (Arthur and Vassilvitskii, 2007). This split operation introduces a new local separation in the latent space while preserving the previously learned hierarchical structure.

The clustering objective encourages samples assigned to each node to be compact along the direction that separates the node from its sibling. As demonstrated by Mautz et al. (2020), directly minimizing the Euclidean distance between samples and node centers can suppress structural information that is orthogonal to the separation direction. DeepECT, therefore, defines a sibling-specific projection direction. For two sibling nodes with centers $\mu _ { n }$ and $\mu _ { m } .$ , the normalized direction from node m to node n is defined as

$$
\rho _ { n } = \frac { \mu _ { n } - \mu _ { m } } { \vert \vert \mu _ { n } - \mu _ { m } \vert \vert _ { 2 } }\tag{5}
$$

where $\rho _ { n }$ defines the principal direction along which the sibling nodes are separated. The data compression (DC) loss then penalizes deviations of the assigned embeddings from their corresponding node centers along this direction:

$$
\mathcal { L } _ { D C } ~ = ~ \frac { 1 } { | N | \cdot | S | } \sum _ { n \in N } \sum _ { \mathbf { G } _ { i } \in S } \left| \rho _ { n } ^ { \top } ( \mu _ { n } - f _ { \theta } ( \mathbf { G } _ { i } ) ) \right|\tag{6}
$$

where N is the set of non-root nodes and S the set of samples. This loss compresses the embeddings within each node along the sibling-separation direction, thereby preserving variation that may be relevant for deeper splits. As a result, strong surface activity differences can be separated at shallow levels of the tree, whereas finer diferences can emerge at deeper levels.

Intuitively, during the optimization, the hierarchy is learned in a top-down manner through an alternating process of compression, splitting, and pruning. After a certain number of training epochs, the node with the most heterogeneous samples is selected as the split candidate, which creates two child nodes and builds a new local separation direction. Nodes that contain very few samples are pruned to avoid unstable or unnecessary over-split. Through this adaptive process, the tree grows progressively and organizes surface activities into a hierarchy that captures multiple levels of separation. For our dataset, we update the tree structure every 3 training epochs and prune nodes containing fewer than 10 samples.

Nodeprototyping. Since the node centers are learnable parameters and the latent space changes during training, we adopt the node center (NC) loss proposed in DeepECT (Mautz et al., 2020) to stabilize the clustering structure throughout the full optimization process. The NC loss updates the vector of each leaf node by aligning it with the mean embedding of data samples assigned to that node. Formally, the node center loss is defined as:

$$
\mathcal { L } _ { N C } = \frac { 1 } { | \mathcal { N } | } \sum _ { n \in N } \frac { 1 } { | S _ { n } | } \sum _ { \mathbf { G } _ { i } \in S _ { n } } \| \mu _ { n } - f _ { \theta } ( \mathbf { G } _ { i } ) \| _ { 2 }\tag{7}
$$

where N denotes the set of nodes, $S _ { n }$ is the set of samples assigned to node $n ,$ and $\mu _ { n }$ represents the cluster centroid. During optimization, the latent embeddings $f _ { \theta } ( \mathbf { G } _ { i } )$ are treated as constant when updating the node centers, such that the loss $\mathcal { L } _ { N C }$ only pulls the node center towards the mean of its assigned samples rather than pulling the samples toward the center. Furthermore, the split node centers are not directly updated, but are derived by the weighted average over child nodes.

Importantly, in our framework, each node center $\mu _ { n }$ serves as a latent prototype of an embedding group. By passing this prototype through the decoder $g _ { \theta } ,$ we obtain a reconstructed GeoMorphogram $\tilde { \mathbf { G } } _ { n } = g _ { \theta } ( \mu _ { n } )$ that represents the characteristic topographic change pattern of the surface activity cluster. These decoded prototypes can provide an interpretable summary of each cluster and allow direct visualization and comparison of the various surface activities. The learned hierarchy functions as both a grouping structure in the latent space and a set of representative topographic change patterns that support the interpretation of complex surface activities themselves.

Loss computation. The overall DTC objective combines the reconstruction loss (Eq. 4), the data compression loss (Eq. 6), and the node center loss (Eq. 6) in a weighted manner:

$$
\mathcal { L } = \lambda _ { R E C } \mathcal { L } _ { R E C } + \lambda _ { D C } \mathcal { L } _ { D C } + \lambda _ { N C } \mathcal { L } _ { N C }\tag{8}
$$

where $\lambda _ { R E C } , \lambda _ { D C } , \lambda _ { N C }$ are weighting coeficients that balance the scale of representation learning and hierarchical clustering.

Training is performed in two stages. First, the autoencoder is pretrained for 20 training epochs using only the reconstruction loss to learn a latent space that captures the temporal-distributional patterns of GeoMorphograms. Second, hierarchical clustering is introduced by jointly optimizing the combined loss in Eq. 8. During joint optimization, embeddings are extracted from all samples, samples are assigned to the nearest leaf nodes in the latent space using Euclidean distance, and the tree is updated based on assignments according to the splitting and pruning criteria. The whole autoencoder and node centers are then optimized with the combined loss. We stop training when the tree reaches a maximum of 50 leaf nodes or when the number of nodes remains stable after a split iteration. This process enables DTC to progressively derive a hierarchical structure of surface activity patterns without requiring the number of process types to be specified in advance.

All implementation parameters are fixed across the Kijkduin, Noordwijk, and combined BeachOBC experiments unless stated otherwise. We use a batch size of 64 and apply a dropout rate of 0.1 to reduce overfitting. Optimization is performed using the AdamW optimizer (Loshchilov and Hutter, 2019). A learning rate of $2 \times 1 0 ^ { - 3 }$ with weight decay $1 \times 1 0 ^ { - 4 }$ is used during pretraining to accelerate convergence, followed by a reduced learning rate of $8 \times 1 0 ^ { - 4 }$ with weight decay $5 \times 1 0 ^ { - 5 }$ during joint optimization to ensure stable refinement of both embeddings and cluster structure.

The loss weights are selected using Optuna (Akiba et al., 2019), which performs automated optimization based on a given evaluation objective. Here, the objective is to maximize the F1 score computed from expert annotations on the validation set and apply the same setting on the test set, as described in Section 3.4. Because our objective is to derive a general process taxonomy, here for sandy beach surface activities, we use the combined BeachOBC to decide the final hyperparameter setting, which is then used for the individual

Kijkduin and Noordwijk datasets. The resulting loss weights are $\lambda _ { R E C } = 1 . 0 , \lambda _ { D C } = 3 . 7 9 , \lambda _ { N C } = 7 . 6 3$ , which provide a balanced contribution of reconstruction, data compression, and node center objectives.

## 3.4. Evaluation

Pairwise expert annotation. To address the lack of labeled data and of a priori knowledge about the occurrence of surface activity types, we design a pairwise evaluation strategy that leverages expert knowledge while avoiding exhaustive object-level labeling. We select reference 4D-OBCs as a random subset and use handcrafted features to identify nearest neighbors as pairs for each 4D-OBC. The handcrafted features describe surface activities in terms of their spatial, temporal, and change-related properties, including size, shape, orientation, cross-shore location, total change volume, duration of occurrence, initial height, and initial timestamp, following Hulskemper et al. (2026b). Experts in coastal geomorphology and 4D point cloud analysis are then asked to annotate each sample pair with a binary label indicating whether two 4D-OBCs belong to the same or diferent types of surface activity through an online labeling interface (Figure 4). The final label of each pair is determined by weighted majority voting from different experts based on annotation confidences. After assembling annotations and filtering out low-confidence labels, we obtain 467 labeled pairs out of a total of 8,754 annotations by 7 experts. We randomly select 50% of these labeled pairs as a validation set to optimize the hyperparameters and use the other 50% as a test set to evaluate the model’s performance.

Multi-level comparison. Given that the proposed DTC framework produces a hierarchical clustering of topographic change objects, we evaluate the results against expert annotations across diferent levels of detail through the order of split nodes. Specifically, we consider pairs that are labeled as similar and grouped within the same node by DTC as true positives (TP), and pairs labeled diferently and assigned to separate nodes as true negatives (TN). Pairs labeled as diferent but grouped in the same node are false positives (FP), and pairs labeled as similar but separated by DTC are false negatives (FN). This evaluation reflects the nature of the hierarchical clustering: clearly distinct surface activities are expected to separate at shallow levels, whereas more subtle distinctions may emerge only at deeper levels. Conversely, pairs labeled as similar should remain in the same cluster until finer levels of the tree. This evaluation protocol does not aim to define a single “correct” cluster, but rather to quantify the consistency between data-driven clustering and expert interpretation. In doing so, it provides a practical and interpretable way to validate the uncovered structure and to indicate meaningful patterns that underlie observed surface activities. Furthermore, it enables identifying the level of granularity at which the learned hierarchy best aligns with expert-annotated distinctions of topographic surface activities.

Evaluation metrics. To quantify our model’s performance, we calculate precision, recall, F1 score, and match accuracy based on the binary agreement between clustering results and expert labels using the following formulas:

$$
P r e c i s i o n = \frac { T P } { T P + F P }
$$

$$
R e c a l l = { \frac { T P } { T P + F N } }
$$

$$
F _ { 1 } = 2 \times \frac { P r e c i s i o n \times R e c a l l } { P r e c i s i o n + R e c a l l }
$$

$$
M a t c h A c c u r a c y = \frac { T P + T N } { T P + F P + F N + T N } .
$$

## 3.5. Deriving a surface process taxonomy

To interpret the learned hierarchy, we visualize the samples and cluster-wise characteristics of each leaf node, inspired by the hierarchical morphotope classification of built form introduced by Fleischmann et al. (2026). To identify the main surface process types, we do not manually select a preferred level of detail. Instead, we follow the pairwise evaluation described in Section 3.4 and select the clustering level that shows the highest agreement with expert annotations. The number of nodes at this level is reported as an expert-supported number of clusters k. We therefore interpret the selected clusters as data-driven, expert-supported, and interpretable surface process types. This strategy allows the full taxonomy to be derived from the learned hierarchy, while the reported main types remain constrained by expert judgment.

To characterize clusters, we first compute selected object-wise handcrafted features, including average change, PCA relative orientation, cross-shore location, and time duration of activity, to showcase physical, geometry, domain, and temporal properties. Average change is calculated as the absolute total change volume $( m ^ { 3 } )$ divided by the footprint area $( m ^ { 2 } )$ , resulting in an area-normalized change magnitude in meters. PCA relative orientation describes the shape orientation of the

![](images/95cd7a7dfd9f5a4646f0c77dcd60d164272f798be553798cc16a49709b1a7fba.jpg)  
Figure 4: Labeling interface used to collect expert annotations. Each task presents a pair of 4D objects-by-change (4D-OBCs) through their spatial and temporal representation. Experts assess whether the two 4D-OBCs represent the same or diferent types of surface activity and provide a confidence score. The labeling interface figure is shown as guidance, as provided in the tutorial for expert annotators.

4D-OBC footprint. It is defined as the acute angle (0°– 90°) between the principal axis of the point distribution, estimated by PCA, and the waterline direction, where 0° indicates an orientation parallel to the waterline and 90° indicates an orientation perpendicular to it. Cross-shore location (m) is defined as the perpendicular distance between the median coordinates and the waterline boundary. Activity time duration (h) is defined as the number of active epochs of each 4D-OBC. These object-wise features provide descriptive context for the learned clusters and support the interpretation of the representative surface process types in Section 4.

## 3.6. Competing methods

To assess the respective contributions of the input representation and the clustering method, we compare DTC with several baselines. Principal component analysis (PCA) + k-means is used as a linear dimensional reduction baseline. Autoencoder (AE) + k-means and AE + agglomerative clustering evaluate whether AEbased embeddings improve clustering when the representation and clustering steps are optimized separately. DEC (Xie et al., 2016) is included as a flat deep clustering baseline, following Wang and Anders (2025), which jointly optimizes representation learning and clustering, but requires a predefined number of target clusters. For input representations, we compare our proposed Geo-Morphograms with the mean time series of 4D-OBCs, which are comparable to the inputs used in existing work (Wang and Anders, 2025; Hulskemper et al., 2022, 2026b). This setup allows us to evaluate the efects of input representation, latent embedding learning, and hierarchical clustering separately.

All experiments in this study are conducted on a Linux workstation (Ubuntu 22.04.5 LTS) equipped with an Intel® Xeon(R) w7-3455 CPU, 128 GB of RAM, and an NVIDIA RTX A4500 GPU (20 GB). This setup is used for both training and inference of the models.

Table 2: Quantitative comparison of diferent clustering methods and input representations on the BeachOBC, Kijkduin, and Noordwijk datasets. The selected number of clusters (k = 8) is determined from the DTC hierarchy using the highest validation F1 score. Best results for each dataset and metric are shown in bold. Precision (P), recall (R), F1, and match accuracy (Acc) are computed from test sets of expert pairwise annotations.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td colspan="4">BeachOBC</td><td colspan="4">Kijkduin</td><td colspan="4">Noordwijk</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>Acc</td><td>P</td><td>R</td><td>F1</td><td>Acc</td><td>P</td><td>R</td><td>F1</td><td>Acc</td></tr><tr><td>PCA + k-means</td><td>Time series</td><td>0.26</td><td>0.88</td><td>0.40</td><td>0.53</td><td>0.26</td><td>0.83</td><td>0.39</td><td>0.56</td><td>0.33</td><td>0.84</td><td>0.48</td><td>0.66</td></tr><tr><td>AE + k-means</td><td>Time series</td><td>0.57</td><td>0.77</td><td>0.66</td><td>0.86</td><td>0.59</td><td>0.68</td><td>0.63</td><td>0.86</td><td>0.61</td><td>0.79</td><td>0.69</td><td>0.87</td></tr><tr><td>AE + Agglomerative</td><td>Time series</td><td>0.55</td><td>0.71</td><td>0.62</td><td>0.84</td><td>0.50</td><td>0.69</td><td>0.58</td><td>0.83</td><td>0.63</td><td>0.67</td><td>0.59</td><td>0.84</td></tr><tr><td>DEC</td><td>Time series</td><td>0.53</td><td>0.67</td><td>0.59</td><td>0.83</td><td>0.76</td><td>0.69</td><td>0.72</td><td>0.91</td><td>0.72</td><td>0.73</td><td>0.72</td><td>0.90</td></tr><tr><td>DTC</td><td>Time series</td><td>0.54</td><td>0.68</td><td>0.60</td><td>0.84</td><td>0.51</td><td>0.64</td><td>0.57</td><td>0.84</td><td>0.76</td><td>0.74</td><td>0.75</td><td>0.91</td></tr><tr><td>PCA + k-means</td><td>GeoMorphograms</td><td>0.63</td><td>0.65</td><td>0.64</td><td>0.87</td><td>0.63</td><td>0.74</td><td>0.68</td><td>0.88</td><td>0.61</td><td>0.64</td><td>0.63</td><td>0.86</td></tr><tr><td>AE + k-means</td><td>GeoMorphograms</td><td>0.70</td><td>0.64</td><td>0.67</td><td>0.89</td><td>0.77</td><td>0.52</td><td>0.62</td><td>0.89</td><td>0.77</td><td>0.73</td><td>0.75</td><td>0.91</td></tr><tr><td>AE + Agglomerative</td><td>GeoMorphograms</td><td>0.76</td><td>0.65</td><td>0.70</td><td>0.90</td><td>0.49</td><td>0.67</td><td>0.57</td><td>0.82</td><td>0.80</td><td>0.69</td><td>0.75</td><td>0.91</td></tr><tr><td>DEC</td><td>GeoMorphograms</td><td>0.74</td><td>0.70</td><td>0.72</td><td>0.90</td><td>0.67</td><td>0.45</td><td>0.53</td><td>0.87</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.92</td></tr><tr><td>DTC</td><td>GeoMorphograms</td><td>0.76</td><td>0.82</td><td>0.78</td><td>0.92</td><td>0.92</td><td>0.83</td><td>0.87</td><td>0.95</td><td>0.84</td><td>0.82</td><td>0.83</td><td>0.94</td></tr></table>

![](images/0a8503ef29b5c683212d252b520a9de3dbe296ff4dda2cca21e8262531194669.jpg)  
Figure 5: Evaluation metrics computed on pairwise annotations across diferent numbers of clusters k on the test set. The dashed line marks the selected level, k = 8, where the highest overall F1 score is achieved by our approach.

## 4. Results

## 4.1. Quantitative performance

Figure 5 shows the evaluation metrics obtained for diferent numbers of clusters on the BeachOBC test set. The metric plot for the validation set is provided in the appendix (Figure 12). The highest validation and test F1 scores are obtained with DTC at k = 8, where precision and recall are best balanced: smaller values of k tend to over-aggregate surface activities that are diferent according to expert annotations, whereas larger values potentially over-split similar activities. We therefore select k = 8 as the expert-supported target number of clusters of major surface process types and for comparison with flat clustering baselines, which are evaluated using the same number of clusters.

Table 2 reports the quantitative results on the BeachOBC, Kijkduin, and Noordwijk datasets. Overall, DTC with GeoMorphograms as input achieves the highest F1 score, precision, and match accuracy across all three datasets and obtains the best or near-best recall performance. On the combined BeachOBC dataset, DTC achieves an F1 score of 0.78 and an accuracy of 0.92, outperforming all competing methods. Similar results are observed on the individual datasets, where DTC reaches an F1 score of 0.87 on Kijkduin and 0.83 on Noordwijk.

Across most methods, GeoMorphogram inputs yield higher F1 scores than mean time series inputs. For PCA + k-means, the F1 score increases from 0.40 to 0.64 on BeachOBC, from 0.39 to 0.68 on Kijkduin, and from 0.48 to 0.63 on Noordwijk. For DTC, the F1 score increases from 0.60 to 0.78 on BeachOBC, from 0.57 to 0.87 on Kijkduin, and from 0.75 to 0.83 on Noordwijk. These results indicate that GeoMorphograms provide a more discriminative representation of object-wise surface activity than mean time series. The comparison between clustering methods also shows that the best performance is obtained when GeoMorphograms are combined with the hierarchical DTC model.

The precision and recall values show diferent performance patterns across methods. PCA + k-means with time series input shows high recall on BeachOBC (0.88), Kijkduin (0.83), and Noordwijk (0.84) but low precision on BeachOBC (0.26), Kijkduin (0.26), and

![](images/1a9d4fbf8eb721ac4ba73b823dba4ec8742d8f126f42d200f618f6787bca0581.jpg)  
Figure 6: Learned hierarchical tree from the BeachOBC dataset. Each node image represents the decoded GeoMorphogram prototype from the corresponding cluster center, and each edge indicates the parent–child relation between nodes. Text labels indicate the dominant visual diferences between sibling branches and are used to support the interpretation of the splitting sequence. The derived expert-supported types (at k = 8) are highlighted by colorful boxes, and the branches are named based on the distinguishing properties along the tree.

Noordwijk (0.33), indicating that many similar pairs are grouped together, but many diferent pairs are also incorrectly assigned to the same cluster. In contrast, DTC with GeoMorphogram input achieves the secondhighest or equal-highest recall on BeachOBC (0.82), Kijkduin (0.83), and Noordwijk (0.82), while also achieving the highest precision on BeachOBC (0.76), Kijkduin (0.92), and Noordwijk (0.84), suggesting that the learned hierarchy can preserve broad similarities while separating heterogeneous surface activity types more efectively.

## 4.2. Learned hierarchy tree

The hierarchical clustering structure learned by DTC on the BeachOBC dataset is visualized as a binary decision tree in Figure 6. Each node represents a cluster of surface activities in the latent space and is illustrated by decoding its learned node center into a GeoMorphogram prototype. The decoded prototypes summarize the representative temporal distribution of topographic change within the corresponding node, while the edges indicate the parent–child relations formed through the splitting process. The clustering trees learned from the individual Kijkduin and Noordwijk datasets are provided in the Appendix (Figure 13 and Figure 14).

The branch labels in Figure 6 describe visually observable properties derived from the decoded prototypes. Change direction refers to the dominant sign of relative topographic change, and change magnitude refers to the distance of the dominant change values from zero. “Gradual” and “sudden” change describe whether the dominant change develops over several consecutive time steps or is concentrated within a short interval. “Long” and “short-term” duration refer to the active time span of the surface activity. “Compact” and “sparse” distribution describe whether the change values are concentrated near the average or spread over a broader portion of the GeoMorphogram. “Coherent” and “incoherent” evolution refer to whether the main distributional structure persists across consecutive time steps or turns directions. “Extreme” and “Noisy” refer to large-magnitude irregular changes or random fluctuations without a clear temporal pattern. These terms are therefore used to annotate the distinguishing properties at each split. Furthermore, each selected leaf node can be described by the sequence of distinguishing properties that separate it from its sibling branches. This provides a consistent naming principle for the cluster: a cluster name is based on the combination of distinguishing properties along its path, as shown in Figure 6 bottom.

At the first split, the hierarchy separates surface activities into two main branches corresponding to major types of surface activities. The decoded prototypes show that one branch is dominated by negative topographic change values, corresponding to erosiondominated activities, whereas the other is dominated by positive topographic change values, corresponding to deposition-dominated activities. Within the erosiondominated branch, for example, the second level separates clusters into relatively gradual and sudden surface activities. Deeper splits further distinguish activities with diferent durations, degrees of coherence, and compactness of distribution. Each erosion-dominated leaf node is therefore characterized by a specific combination of properties along its decision path from the root. Within the deposition-dominated branch, the second level separates gradual and sudden surface activities, where gradual deposition is represented by a more continuous increase and sudden deposition by a more abrupt signal of positive values over a short interval. Subsequent splits further distinguish deposition activities according to temporal coherence, compactness, and duration.

![](images/b65799ab6ea5e6695b467b414b6bf630d6f7cdb4c7fc826b91743cad0d7c0bae.jpg)  
Figure 7: Surface process taxonomy derived from the BeachOBC dataset, consisting of combined 4D-OBCs detected from the Kijkduin and Noordwijk datasets. The circular tree shows the selected hierarchical structure at k = 8, the outer panels show representative GeoMorphograms for each leaf node, and the surrounding feature rings summarize cluster-wise physical properties.

![](images/b1a58267d104c8442a204a98705aec54b5155e6478b8d89aa9f4492518c0ccdc.jpg)  
Figure 8: Temporal frequency of diferent surface process clusters at Kijkduin. Frequency is computed as the number of active 4D-OBCs per day for each selected cluster.

## 4.3. Surface process taxonomy

The derived surface process taxonomy is shown in Figure 7. Based on the quantitative evaluation in Section 3.4, the tree cut at cluster number of k = 8 achieves the highest agreement with expert annotations, with an F1 score of 0.78 on the BeachOBC dataset (Table 2). We therefore use this level of detail to define the main types of sandy beach surface processes. The selected branches are interpreted as data-driven and expert-supported surface process types. In Figure 7, the circular tree shows the hierarchical relations among all clusters, and the surrounding panels show representative GeoMorphograms for the corresponding leaf nodes.

The cluster-wise handcrafted feature summaries are presented as feature rings around the hierarchy in Figure 7. These features include average change, PCA relative orientation, cross-shore location, and time duration, as defined in Section 3.5. Each feature is categorized into five color intensities for direct visual comparison. The feature rings show that diferent clusters are associated with diferent combinations of physical, geometric, domain, and temporal features. Neighboring branches share similar feature profiles, whereas more distant branches show stronger diferences in one or more features. Together with the learned hierarchy and representative GeoMorphograms, this taxonomy provides the descriptive basis for interpreting the hierarchical clusters as surface process types.

## 4.4. Temporal occurrence of diferent process types

To showcase the applicability of the taxonomy for a deeper understanding of surface processes on sandy beaches, we examine the temporal occurrence patterns of the identified surface process types at the two study sites. Figures 8 and 9 show the daily frequency of each selected type over the monitoring period at Kijkduin and Noordwijk, respectively. The frequency is computed as the number of active 4D-OBCs per day for each type.

At Kijkduin, the frequency of the “long-term gradual erosion” type increases strongly from February to March (Figure 8). During the same period, the “coherent gradual deposition” type also shows increased activity. This peak coincides with a period of bar migration and erosion as observed by Vos et al. (2020). At Noordwijk, a similar increase is visible in February 2021 (Figure 9), with weaker increases in 2020 and 2022. These temporal occurrence plots show when diferent taxonomy types are active at each site and provide the basis for comparing the timing and frequency of surface activity patterns across monitoring periods.

## 5. Discussion

## 5.1. GeoMorphograms and hierarchical clustering

The quantitative results (Section 4.1) highlight the importance of combining GeoMorphogram-based representation and hierarchical clustering optimization in DTC. GeoMorphograms preserve variation in change magnitude, sign, temporal coherence, and temporal evolution within each 4D-OBC. These properties are also commonly considered during expert interpretation. In contrast, mean time series summarize a 4D-OBC into a limited set of values and may therefore suppress objectwise diferences relevant for distinguishing notably spatial heterogeneous process types. This explains why replacing mean time series with GeoMorphograms improves $F _ { 1 }$ scores across most methods.

The precision and recall values further clarify this effect. PCA + k-means with time series input achieves high recall but low precision, indicating that it tends to over-aggregate surface activities. This means that most pairs identified by experts as belonging together are successfully grouped into the same cluster, but the clusters also contain many pairs that experts consider diferent, resulting in a large number of false-positive matches.

![](images/84e2abfee07a643f0db2b149feb3f88b40615186ca118eb8aeeb7958333988d1.jpg)  
Figure 9: Temporal frequency of diferent surface process clusters at Noordwijk. Frequency is computed as the number of active 4D-OBCs per day for each selected cluster.

![](images/7470bbc3d4c0419c3d0d2ec04fed75c1f2bdbf230d9d1c10429c08379cc67c30.jpg)  
Figure 10: T-SNE visualization of the learned GeoMorphogram embeddings after pretraining (left) and clustering (right).

![](images/1fdb0ea0271251538b8c185256c62001289dddf704bf12d1cb6b5aae3cc56dcd.jpg)  
Figure 11: Performance comparison of integrating diferent handcrafted feature sets on the BeachOBC dataset. GM denotes the GeoMorphogram input (orange). Diferent conditioning settings (blue) are: (1) basic features, including area and total change volume; (2) geometric features, including area, principal axes of the point distribution estimated by PCA, PCA relative orientation, cross-shore location, initial mean elevation, and total change volume; (3) temporal features, including duration and cyclically encoded start time with sine/cosine transformation; and (4) all of the features of 1–3.

This behavior is consistent with a time series representation that captures broad temporal trends but ignores spatial diferences. In contrast, GeoMorphograms reduce this ambiguity by preserving distributionally spatial change within 4D-OBCs, leading to more balanced precision and recall, and higher F1 scores when combined with DTC.

The evolution of the embedding space provides further insight into the role of the joint optimization objective (Figure 10). After autoencoder pretraining, the latent embeddings exhibit a binary separation indicating erosion- and deposition-dominated surface activities. Finer distinctions between diverse clusters emerge only after the joint optimization with the hierarchical clustering objective. This demonstrates that reconstruction learning provides a meaningful initial latent space, whereas the hierarchical clustering refines the latent space towards deriving more subtle patterns beyond the primary types. These results therefore support the use of an end-to-end hierarchical deep clustering method over existing two-step workflows that separately perform feature extraction and clustering.

## 5.2. Efect of integrating additional handcrafted features

For sandy beach surface activities, object-wise handcrafted features may include area, cross-shore location, duration, magnitude, slope, and change volume (Hulskemper et al., 2026b). These features are intuitive and interpretable, making them attractive for domainspecific applications. Several of them are also reflected in the cluster characteristics of the derived taxonomy, although they are not explicitly involved during the training. To examine whether additional domain knowledge can further improve DTC performance, we incorporated multiple groups of handcrafted features into the latent embeddings through a lightweight conditioning module.

This module takes the normalized handcrafted features and uses a multi-layer perceptron (MLP) layer to project them into an 8-dimensional vector. This vector is then concatenated with the 16-dimensional Geo-Morphogram embedding to form the full embeddings of surface activities. We repeat the DTC training ten times on the BeachOBC dataset using four groups of handcrafted features: (1) basic features, including area, and total change volume; (2) geometric features, including area, principal axes of the point distribution estimated by PCA, PCA relative orientation, cross-shore location, initial mean elevation, and total change volume; (3) temporal features, including duration, and cyclically encoded start time with sine/cosine transformation; and (4) all of the features of 1–3.

The results in Figure 11 show that handcrafted feature conditioning provides only limited and inconsistent improvement over the GeoMorphograms alone. Conditioning with all handcrafted features yields the highest mean F1 score, precision, and accuracy, indicating that additional features can provide complementary information when combined. However, conditioning with only basic, geometric, or temporal features does not consistently improve performance and can reduce recall, accuracy, or F1 score. This suggests that the proposed GeoMorphograms already contain most of the relevant information required for clustering surface activities. Handcrafted features may provide secondary refinement, especially when multiple feature groups are combined, but simple concatenation does not essentially improve DTC’s performance.

## 5.3. Interpretation of the learned taxonomy

As shown in Figure 6, the learned hierarchy organizes all surface activities into a hierarchical structure. Coarse levels capture broad distinctions, such as erosion-dominated and deposition-dominated activities. Deeper levels separate more specific variations in duration, change magnitude, coherence, compactness, and temporal evolution. Building such a hierarchy is labelagnostic, although deriving major surface process types still requires expert support. The selected number of clusters k = 8 provides the highest agreement with expert annotations; however, other levels can still be useful for diferent analytical purposes.

The derived taxonomy (Figure 7) should therefore be interpreted as an expert-supported, data-driven organization of observed surface process types rather than as a definitive process classification. Surface processes on sandy beaches are often mixed, gradual, and contextdependent. A single 4D-OBC may contain both erosion and deposition signals, or may reflect multiple drivers acting over the same period. Therefore, process boundaries are not always discrete. The DTC provides the possibility to identify these clusters and indicates that the learned representation captures object-wise variations in surface processes that may not be explicitly recognized in existent analysis approaches.

This flexibility is important for topographic monitoring. A coastal manager may prefer coarse process types for a rapid overview, while a geomorphologist may inspect deeper branches to study subtle variations in sediment redistribution or recovery behavior. The decoded GeoMorphogram prototypes support this interpretation by showing the temporal distributional change properties of each cluster. The sequence of splits from root to leaf nodes further provides an interpretable chain of distinguishing properties, enabling systematic characterization of surface process types across multiple levels of detail.

## 5.4. Applicability in the coastal domain

The derived taxonomy provides a comprehensive and interpretable way to summarize large volumes of topographic 4D point cloud data. Instead of inspecting thousands of individual 4D-OBCs, users can analyze the occurrence of a limited set of surface process types. This reduces the complexity of the original point cloud time series while preserving information about the specific type and timing of surface processes.

The partially overlapping occurrence patterns of similar surface process types at Kijkduin (Figure 8) and Noordwijk (Figure 9), together with the relatively similar performance scores (Table 2), suggest that the derived taxonomy has potential transferability across sandy beach sites. This taxonomy can therefore be used at varying levels of detail to characterize diferent beaches in terms of their short-term responses through the identification of occupancy of the tree. This enables comparisons of seasonal and location-dependent variations in short-term responses of sandy beaches without any pre-defined labels.

Potentially, variations in the occupancy of taxonomy branches may also reflect diferences in environmental conditions between sites and time periods. In future statistical causal analysis, this might be exploited by characterizing variations in the degree of occupancy of the tree, and linking these to time-varying environmental variables through causal inference (Runge, 2023), so as to identify the causes of varying response of the beach system. The hierarchical structure aids in identifying at which level of detail diferences start to occur at each moment in time, whereas in flat clustering, these diferences might not become apparent.

Additionally, the clustering result could aid datadriven prediction of beach activities. The identified hierarchy greatly reduces the data complexity of the point cloud time series while preserving relevant change information, such that simpler one-dimensional time series based algorithms (e.g., regression, long short-term memory) can become applicable. If these are able to adequately predict future surface activities, the taxonomy, in particular, is well applicable for aiding timely response to storm erosion and response cycles through adequate beach management, as the taxonomy ofers a relatively simple interpretation of expected surface activities.

## 5.5. Limitations and future work

Despite the demonstrated advantages of the proposed DTC framework, several limitations should be acknowledged. First, GeoMorphograms abstract topographic change as temporal distributions within each 4D-OBC. This makes the representation compact and comparable, but it reduces explicit spatial information. Directions of sediment transport, internal spatial structure, and interactions between neighboring activities are not directly modeled. Future work could address this limitation by incorporating spatial GeoMorphograms, graphbased interactions, or spatial attention mechanisms.

Second, our experiments are currently limited to sandy beach environments. The transferability of the learned taxonomy to other geomorphic settings, such as riverbeds, glaciers, landslides, or urban scenes, remains to be tested. These environments may contain diferent spatial structures, temporal scales, and process interactions. More automatic generalization of 4D-OBC extraction might support broader application of our method.

Third, the evaluation relies on expert pairwise annotations. This is practical in the absence of benchmark labels, but it cannot remove subjectivity from process interpretation. The selected level of detail is therefore best understood as the level most consistent with the available expert judgments, rather than as an absolute ground truth. This also reflects the broader challenge that surface process categories are often ambiguous and context-dependent rather than strictly defined. Semantic annotation protocols and human-in-the-loop refinement could make the evaluation more robust and make the parameterization more automatic.

## 6. Conclusion

This study addresses the challenge of deriving meaningful surface process types from PLS-acquired 4D point cloud data without predefined labels or a fixed number of target clusters. To bridge the gap between object-based change detection and systematic character ization of complex surface activities, we introduce Ge oMorphograms and DeepTopoClustering (DTC), which integrate object-based 4D change analysis with representation learning and hierarchical deep clustering. Ex periments on two sandy beach 4D datasets, as well as the combined BeachOBC dataset, demonstrate that DTC can derive a data-driven and expert-supported surface process taxonomy. Quantitative evaluation based on expert annotations exhibits strong agreement between the learned clusters and expert judgments while identifying eight primary surface process types at the highest expert agreement level, achieving an F1 score of 0.78 and an accuracy of 0.92. From the interpreta tion, the learned hierarchy captures both major erosiondeposition distinctions and finer process variations in duration, magnitude, coherence, compactness, and temporal evolution that are dificult to define manually. Temporal occurrence analysis further shows that the derived process types exhibit distinct dynamics across sites and time periods. Overall, these results demon strate the potential of hierarchical deep clustering for transforming dense 4D point cloud data into interpretable surface process knowledge. The proposed DTC framework ofers a scalable and data-driven way to organize complex spatiotemporal surface activities into a hierarchical taxonomy while allowing the level of detai to be adapted to specific interpretation and application requirements. Future work should extend the framework to a wider range of geomorphic environments, incorporate richer spatial representations, and integrate semantic knowledge through advanced learning strate gies. More broadly, this research represents a step toward an automated system for deriving and organizing knowledge from surface activities observed in 4D poin clouds, thereby providing higher-level information fo understanding underlying environmental processes.

## Acknowledgments

This research was partly funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under project number 535733258 (Extract4D), and partly funded by the Bavarian State Ministry of Science and the Arts in the framework of the bidt Graduate Center for Postdocs (AI4ENV). Daan Hulskemper has been supported by the Nederlandse Organisatie voor Wetenschappelijk Onderzoek (grant no. 20014). The authors gratefully acknowledge Prabin Gyawali, Xiaoyu Huang, and Pim Maydhisudhiwongs for their valuable contributions to the expert annotation process, which was essential for the evaluation of the proposed framework.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this manuscript, the author(s) used Grammarly (TUM campus license) and DeepL (free version) to assist with grammar correction and language refinement, and Microsoft Copilot (TUM campus license; GPT-based) to improve phrasing and readability. The author(s) reviewed and edited the output as needed and take full responsibility for the content of the published article.

## Appendix

![](images/80e00efc4d110c9a41198be9a0d9981967757dd0f535a479d96a009c92abd2f1.jpg)  
Figure 12: Evaluation metrics computed on pairwise annotations across diferent numbers of clusters k on the validation set. The dashed line marks the selected level, k = 8, where the highest overall F1 score is achieved by our approach.

## Data and code availability

The permanent laser scanning data underlying this research is published at PANGAEA (https:

//doi.org/10.1594/PANGAEA.934058) and 4TU.ResearchData (https://doi.org/10.4121/ 1aac46fb-7900-4d4c-a099-d2ce354811d2).

The derived 4D-OBCs are available at 4TU.ResearchData (https://doi.org/10.4121/ 7cf573ec-ce73-4dcf-86ea-553f159c417c).

![](images/c9535253f8d2fb9cb56487a0a8c4c2cb6464a087011af099535d34700d383603.jpg)

Figure 13: Learned hierarchical tree from the Kijkduin dataset. Each node image represents the decoded GeoMorphogram prototype from the corresponding cluster center, and each edge indicates the parent–child relation between nodes.  
![](images/f3b83615f4a1f6ccfb01e7026fd7741feec7edb58e2093c40766e2d5a4d5e082.jpg)  
Figure 14: Learned hierarchical tree from the Noordwijk dataset. Each node image represents the decoded GeoMorphogram prototype from the corresponding cluster center, and each edge indicates the parent–child relation between nodes.

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M., 2019. Optuna: A Next-generation Hyperparameter Optimization Framework, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA. pp. 2623–2631. doi:10.1145/ 3292500.3330701.

Anders, K., Winiwarter, L., Lindenbergh, R., Williams, J.G., Vos, S.E., Höfle, B., 2020. 4D objects-by-change: Spa-

tiotemporal segmentation of geomorphic surface change from LiDAR time series. ISPRS Journal of Photogrammetry and Remote Sensing 159, 352–363. doi:10.1016/ j.isprsjprs.2019.11.025.

Anders, K., Winiwarter, L., Mara, H., Lindenbergh, R., Vos, S.E., Höfle, B., 2021. Fully automatic spatiotemporal segmentation of 3D LiDAR time series for the extraction of natural surface changes. ISPRS Journal of Photogrammetry and Remote Sensing 173, 297–308. doi:10.1016/j. isprsjprs.2021.01.015.

Arthur, D., Vassilvitskii, S., 2007. K-means++: The advantages of careful seeding, in: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, Society for Industrial and Applied Mathematics, USA. pp. 1027–1035.

Bengio, Y., Courville, A., Vincent, P., 2013. Representation Learning: A Review and New Perspectives. IEEE Trans. Pattern Anal. Mach. Intell. 35, 1798–1828. doi:10.1109/ TPAMI.2013.50.

Caron, M., Bojanowski, P., Joulin, A., Douze, M., 2018. Deep Clustering for Unsupervised Learning of Visual Features, in: Proceedings of the European Conference on Computer Vision (ECCV), pp. 132–149.

de Boer, D.H., 1992. Hierarchies and spatial scale in process geomorphology: A review. Geomorphology 4, 303–318. doi:10.1016/0169-555X(92)90026-K.

de Gélis, I., Lefèvre, S., Corpetti, T., 2023. DC3DCD: Unsupervised learning for multiclass 3D point cloud change detection. ISPRS Journal of Photogrammetry and Remote Sensing 206, 168–183. doi:10.1016/j.isprsjprs. 2023.10.022.

Eitel, J.U.H., Höfle, B., Vierling, L.A., Abellán, A., Asner, G.P., Deems, J.S., Glennie, C.L., Joerg, P.C., LeWinter, A.L., Magney, T.S., Mandlburger, G., Morton, D.C., Müller, J., Vierling, K.T., 2016. Beyond 3-D: The new spectrum of lidar applications for earth and ecological sciences. Remote Sensing of Environment 186, 372–392. doi:10.1016/j.rse.2016.08.018.

Eltner, A., Maas, H.G., Faust, D., 2018. Soil microtopography change detection at hillslopes in fragile Mediterranean landscapes. Geoderma 313, 217–232. doi:10.1016/j.geoderma.2017.10.034.

Fleischmann, M., Samardzhiev, K., Brázdová, A., Dancejová,ˇ D., Winkler, L., 2026. The hierarchical morphotope classification: A theory-driven framework for large-scale analysis of built form. Cities 174, 107047. doi:10.1016/j. cities.2026.107047.

Godfray, H.C.J., 2002. Challenges for taxonomy. Nature 417, 17–19. doi:10.1038/417017a.

Huggett, R., 2016. Fundamentals of Geomorphology. 4 ed., Routledge, London. doi:10.4324/9781315674179.

Hulskemper, D., Anders, K., Antolínez, J.a.Á., Kuschnerus, M., Höfle, B., Lindenbergh, R., 2022. Characterization of Morphological Surface Activities Derived from Near-Continuous Terrestrial LiDAR Time Series. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences XLVIII-2-W2-2022, 53–60. doi:10.5194/ isprs-archives-XLVIII-2-W2-2022-53-2022.

Hulskemper, D., Anders, K., Antolínez, J.A.A., Lindenbergh, R., 2026a. Uncovering Causation of Short-Term Sandy Beach Surface Dynamics Measured by Permanent Laser Scanning, in: Coelho, C., Hallin, C., Sancho, F., Silva, P.A. (Eds.), Coastal Dynamics 2025, Springer Nature Switzerland, Cham. pp. 338–344. doi:10.1007/ 978-3-032-15477-4\_51.

Hulskemper, D., Antolínez, J.A.Á., Lindenbergh, R., Anders, K., 2026b. Coastal process understanding through automated identification of recurring surface dynamics in permanent laser scanning data of a sandy beach. Earth Surface Dynamics 14, 329–359. doi:10.5194/ esurf-14-329-2026.

Irwin, L.A.K., Coops, N.C., Anders, K., Mandlburger, G., Winiwarter, L., 2025. Light detection and ranging of natural systems. Nat Rev Methods Primers 5, 76. doi:10. 1038/s43586-025-00446-3.

Kromer, R.A., Abellán, A., Hutchinson, D.J., Lato, M., Chanut, M.A., Dubois, L., Jaboyedof, M., 2017. Automated terrestrial laser scanning with near-real-time change detection – monitoring of the Séchilienne landslide. Earth Surface Dynamics 5, 293–310. doi:10.5194/ esurf-5-293-2017.

Kuschnerus, M., de Vries, S., Antolínez, J.A.Á., Vos, S., Lindenbergh, R., 2024a. Identifying topographic changes at the beach using multiple years of permanent laser scanning. Coastal Engineering 193, 104594. doi:10.1016/j. coastaleng.2024.104594.

Kuschnerus, M., Lindenbergh, R., Vos, S., 2021. Coastal change patterns from time series clustering of permanent laser scan data. Earth Surface Dynamics 9, 89–103. doi:10.5194/esurf-9-89-2021.

Kuschnerus, M., Lindenbergh, R., Vos, S., Hanssen, R., 2024b. Statistically assessing vertical change on a sandy beach from permanent laser scanning time series. ISPRS Open Journal of Photogrammetry and Remote Sensing 11, 100055. doi:10.1016/j.ophoto.2023.100055.

Lague, D., Brodu, N., Leroux, J., 2013. Accurate 3D comparison of complex topography with terrestrial laser scanner: Application to the Rangitikei canyon (N-Z). ISPRS Journal of Photogrammetry and Remote Sensing 82, 10– 26. doi:10.1016/j.isprsjprs.2013.04.009.

Lindenbergh, R., Anders, K., Campos, M., Czerwonka-Schröder, D., Höfle, B., Kuschnerus, M., Puttonen, E., Prinz, R., Rutzinger, M., Voordendag, A., Vos, S., 2025. Permanent terrestrial laser scanning for near-continuous environmental observations: Systems, methods, challenges and applications. ISPRS Open Journal of Photogrammetry and Remote Sensing 17, 100094. doi:10.1016/j. ophoto.2025.100094.

Liu, X., Han, X., Xia, H., Li, K., Zhao, H., Jia, J., Zhen, G., Su, L., Zhao, F., Cao, X., 2024. PointCluster: Deep Clustering of 3-D Point Clouds With Semantic Pseudo-Labeling. IEEE Transactions on Geoscience and Remote Sensing 62, 1–14. doi:10.1109/TGRS.2024.3393911.

Loshchilov, I., Hutter, F., 2019. Decoupled Weight Decay Regularization. doi:10.48550/arXiv.1711.05101, arXiv:1711.05101.

Mautz, D., Plant, C., Böhm, C., 2020. DeepECT: The Deep Embedded Cluster Tree. Data Sci. Eng. 5, 419–432. doi:10.1007/s41019-020-00134-0.

O’Dea, A., Brodie, K.L., Hartzell, P., 2019. Continuous Coastal Monitoring with an Automated Terrestrial Lidar Scanner. Journal of Marine Science and Engineering 7, 37. doi:10.3390/jmse7020037.

Puttonen, E., Lehtomäki, M., Litkey, P., Näsi, R., Feng, Z., Liang, X., Wittke, S., Pandžic, M., Hakala, T., Karjalainen,´ M., Pfeifer, N., 2019. A Clustering Framework for Monitoring Circadian Rhythm in Structural Dynamics in Plants From Terrestrial Laser Scanning Time Series. Front. Plant Sci. 10. doi:10.3389/fpls.2019.00486.

Qin, R., Tian, J., Reinartz, P., 2016. 3D change detection – Approaches and applications. ISPRS Journal of Photogrammetry and Remote Sensing 122, 41–56. doi:10.1016/j. isprsjprs.2016.09.013.

Ronen, M., Finder, S.E., Freifeld, O., 2022. DeepDPM: Deep Clustering With an Unknown Number of Clusters, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, New Orleans, LA, USA. pp. 9851–9860. doi:10.1109/cvpr52688.2022.00963.

Runge, J., 2023. Modern causal inference approaches to investigate biodiversity-ecosystem functioning relationships. Nat Commun 14, 1917. doi:10.1038/ s41467-023-37546-1.

Vos, S., Anders, K., Kuschnerus, M., Lindenbergh, R., Höfle, B., Aarninkhof, S., de Vries, S., 2022. A high-resolution 4D terrestrial laser scan dataset of the Kijkduin beach-dune system, The Netherlands. Sci Data 9, 191. doi:10.1038/ s41597-022-01291-9.

Vos, S., Kuschnerus, M., Lindenbergh, R., de Vries, S., 2023. 4D spatio-temporal laser scan dataset of the beach-dune system in Noordwijk, NL. doi:10.4121/ 1AAC46FB-7900-4D4C-A099-D2CE354811D2.

Vos, S., Spaans, L., Reniers, A., Holman, R., Mccall, R., de Vries, S., 2020. Cross-Shore Intertidal Bar Behavior along the Dutch Coast: Laser Measurements and Conceptual Model. Journal of Marine Science and Engineering 8, 864. doi:10.3390/jmse8110864.

Wang, H., Zhao, Y., Li, S., Liu, Z., Zhang, X., 2025a. Deep-CropClustering: A deep unsupervised clustering approach by adopting nearest and farthest neighbors for crop mapping. ISPRS Journal of Photogrammetry and Remote Sensing 224, 187–201. doi:10.1016/j.isprsjprs.2025. 04.007.

Wang, J., Anders, K., 2025. Unsupervised Deep Clustering on Spatiotemporal Objects Extracted from 4D Point Clouds for Automatic Identification of Topographic Processes in Natural Environments. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences X-G-2025, 929–936. doi:10.5194/ isprs-annals-X-G-2025-929-2025.

Wang, J., Eskandari, R., Lin, C., Mészáros, J., Obrecht, L., Rowe, E., Salzinger, J., Rutzinger, M., Mayr, A., 2025b. Investigation of Riverbank Peat Erosion in a small Alpine catchment by Multi-Temporal Point Cloud Change Analysis and Time Series Clustering. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences X-G-2025, 937–944. doi:10.5194/ isprs-annals-X-G-2025-937-2025.

Wang, S., Cao, J., Yu, P.S., 2022. Deep Learning for Spatio-Temporal Data Mining: A Survey. IEEE Transactions on Knowledge and Data Engineering 34, 3681–3700. doi:10. 1109/TKDE.2020.3025580.

Wang, Z., Haarer, E., Zhu, T., Dai, Z., MacLellan, C.J., 2025c. Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery. doi:10.48550/arXiv.2509.23602, arXiv:2509.23602.

Weidner, L., van Veen, M., Lato, M., Walton, G., 2021. An algorithm for measuring landslide deformation in terrestrial lidar point clouds using trees. Landslides 18, 3547– 3558. doi:10.1007/s10346-021-01723-4.

Williams, J.G., Rosser, N.J., Hardy, R.J., Brain, M.J., Afana, A.A., 2018. Optimising 4-D surface change detection: An approach for capturing rockfall magnitude-frequency. Earth Surface Dynamics 6, 101–119. doi:10.5194/ esurf-6-101-2018.

Winiwarter, L., Anders, K., Czerwonka-Schröder, D., Höfle, B., 2023. Full four-dimensional change analysis of topographic point cloud time series using Kalman filtering. Earth Surface Dynamics 11, 593–613. doi:10.5194/ esurf-11-593-2023.

Winiwarter, L., Anders, K., Höfle, B., 2021. M3C2-EP: Pushing the limits of 3D topographic point cloud change detection by error propagation. ISPRS Journal of Photogrammetry and Remote Sensing 178, 240–258. doi:10.1016/j. isprsjprs.2021.06.011.

Woodcock, C.E., Loveland, T.R., Herold, M., Bauer, M.E., 2020. Transitioning from change detection to monitoring

with remote sensing: A paradigm shift. Remote Sensing of Environment 238, 111558. doi:10.1016/j.rse.2019. 111558.

Xie, J., Girshick, R., Farhadi, A., 2016. Unsupervised Deep Embedding for Clustering Analysis, in: Proceedings of The 33rd International Conference on Machine Learning, PMLR. pp. 478–487.

Yang, Y., Holst, C., 2025. Piecewise-ICP: Eficient and robust registration for 4D point clouds in permanent laser scanning. ISPRS Journal of Photogrammetry and Remote Sensing 227, 481–500. doi:10.1016/j.isprsjprs.2025. 06.026.

Yang, Y., Schwieger, V., 2023. Supervoxel-based targetless registration and identification of stable areas for deformed point clouds. Journal of Applied Geodesy 17, 161–170. doi:10.1515/jag-2022-0031.

Zahs, V., Winiwarter, L., Anders, K., Williams, J.G., Rutzinger, M., Höfle, B., 2022. Correspondence-driven plane-based M3C2 for lower uncertainty in 3D topographic change quantification. ISPRS Journal of Photogrammetry and Remote Sensing 183, 541–559. doi:10.1016/j. isprsjprs.2021.11.018.

Zhou, S., Xu, H., Zheng, Z., Chen, J., Li, Z., Bu, J., Wu, J., Wang, X., Zhu, W., Ester, M., 2024. A Comprehensive Survey on Deep Clustering: Taxonomy, Challenges, and Future Directions. ACM Comput. Surv. doi:10.1145/ 3689036.