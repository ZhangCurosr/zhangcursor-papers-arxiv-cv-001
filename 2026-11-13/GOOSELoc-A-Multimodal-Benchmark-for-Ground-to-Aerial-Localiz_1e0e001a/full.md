# GOOSELoc: A Multimodal Benchmark for Ground-to-Aerial Localization in Unstructured Off-Road Terrain

Fabian Deuser<sup>1,3</sup> Peter Mortimer<sup>2</sup> Robin-Nico Kampa<sup>1</sup> Anna Boßend¨ orfer¨ <sup>1</sup> Mirko Mahlisch¨ <sup>2</sup> Martin Werner<sup>3,4</sup> Norbert Oswald<sup>1</sup>

<sup>1</sup>Institute of Distributed Intelligent Systems (ETTI 2), University of the Bundeswehr Munich, Neubiberg, Germany <sup>2</sup>Institute for Autonomous Driving (LRT 8.1), University of the Bundeswehr Munich, Neubiberg, Germany <sup>3</sup>Technical University of Munich, Germany; TUM School of Engineering and Design, Department of Aerospace and Geodesy <sup>4</sup>TUM School of Computation, Information and Technology; Munich Data Science Institute (MDSI)

{fabian.deuser, peter.mortimer, robin-nico.kampa, anna.boessendoerfer, mirko.maehlisch, norbert.oswald}@unibw.de, martin.werner@tum.de

## Abstract

Cross-view pose estimation estimates a ground camera’s geographic pose by matching onboard sensor data with geo-referenced overhead imagery. This is particularly relevant when GNSS is unreliable, for example under forest canopy or in urban canyons. Existing metric crossview benchmarks, such as VIGOR and cross-view KITTI, are dominated by urban and road scenes with strong human-made structure without off-road terrain. We introduce GOOSELoc, a multimodal benchmark for ground-toaerial localization in unstructured environments, including forests, fields, and rural tracks. GOOSELoc comprises 9,356 scenes recorded across all four seasons around Munich, Germany, pairing geo-referenced aerial orthophotos with surround-view RGB imagery and 3D LiDAR point clouds. Paved/unpaved labels enable analyzing the impact of terrain structure, while the seasonal recordings allow studying the appearance gap to mostly summer-captured orthophotos. To assess the transferability of existing methods, we adapt andfine-tunefour state-of-the-art cross-view localization baselines on GOOSELoc. Despite task-specific fine-tuning, the performance degrades substantially in offroad environments. We therefore propose a multimodal baseline combining surround-view imagery and LiDAR geometry, improving localization performance and providing a stronger reference for future work. Data and baselines are released at https://gooseloc.github.io.

## 1. Introduction

Accurate geo-localization is one of the cornerstones of modern navigation and underpins the safe transportation of goods and people. Systems deployed in cars, trucks, and UAVs rely predominantly on Global Navigation Satellite Systems (GNSS). However, GNSS signals degrade systematically in environments where direct sky visibility is obstructed: urban canyon effects limit accuracy in dense cities [2], and tree canopies together with uneven terrain cause comparable or worse degradation in forested and offroad areas [35, 44]. Robust backup systems are therefore required, particularly for the off-road domain, where these conditions are the norm rather than the exception.

Cross-view geo-localization [15] is a computer-visionbased alternative: it estimates a ground sensor’s pose by relating its observations to globally available, geo-referenced aerial imagery, yielding absolute positions. Existing benchmarks address two task formulations. Retrieval-oriented datasets such as CVUSA [36, 41] and CVACT [16] pair ground panoramas with aerial tiles and treat localization as nearest-neighbor matching against a tile database, where the recoverable position is bounded by the tile size. Metric pose estimation is targeted by VIGOR [45], which additionally supports retrieval, and by the cross-view extension of KITTI [7] introduced by Shi et al. [25], both targeting subtile localization at the meter level. These benchmarks share a limitation that reflects their origin: all target urban, roadcentric settings, with imagery captured from street-view panoramas or road-going vehicles along paved roads. Even datasets that include rural imagery, CVUSA for instance, capture it along highways and paved roads. Off-road perception itself is well studied [11, 19, 34], but almost exclusively for segmentation and navigation rather than groundto-aerial localization. This setting also imposes a fundamentally different appearance regime. Whereas urban and road scenes provide stable, human-made cross-view anchors such as buildings, lane markings and signage [4], unstructured terrain is dominated by vegetation whose appearance varies strongly with the season and shifts continuously

## GOOSE LOC

![](images/1893f78f79bf84ed3731d9c86c10c9fd233f54b5bfa11ae492b0e006262815ff.jpg)  
Figure 1. GOOSELoc Dataset Sample. The 4 roof-mounted cameras <sup>■ ■ ■ ■</sup> each have a 2448 × 1000 px resolution and a 94.71<sup>◦</sup> horizontal field of view after rectification, giving us a 360<sup>◦</sup> coverage of the surroundings. This can be fused with a sweep of the roof mounted 128-beam rotating LiDAR scanner <sup>■</sup>. The Bavarian orthophotos <sup>■</sup> are taken on a clear day in the summer and have a temporal mismatch to the ground-view images recorded during winter. The ground truth global location and heading ⋆ are obtained by an Oxford RT3000v3 INS system with differential RTK-GNSS corrections using the NTRIP protocol to receive accurate readings in forested areas.

between recordings, a well-documented source of degradation for place recognition and pose estimation [18, 23]. Orthophotos aggravate this mismatch: flown once or twice a year under clear-weather summer conditions, they freeze a single vegetation state, and even moderate ground-aerial temporal offsets are known to degrade cross-view performance [5].

We address this gap with GOOSELoc, a multimodal benchmark for ground-to-aerial localization in unstructured off-road terrain. GOOSELoc builds on the German Outdoor and Offroad Dataset (GOOSE) [19], originally introduced for outdoor and off-road semantic segmentation, and uses accurately localized multimodal recordings captured across all four seasons around Munich, Germany. We filter the recordings for positional accuracy and make the following contributions:

• We introduce GOOSELoc, a multimodal benchmark for metric ground-to-aerial localization in unstructured offroad terrain, comprising 9,356 scenes paired with georeferenced 20 cm ground sample distance (GSD) aerial orthophotos, see Figure 1.

• We provide synchronized surround-view RGB imagery and 3D LiDAR point clouds for each scene, together with paved/unpaved labels and four-season coverage, enabling analyses of terrain structure and seasonal appearance gaps.

• We adapt and fine-tune CCVPE [38], the work of Shi et al. [26], HC-Net [33] and FG<sup>2</sup> [37], as state-of-the-art cross-view localization methods, and show that they degrade substantially on GOOSELoc. We further propose an RGB-LiDAR fusion baseline that improves over the fine-tuned single-modality baseline.

## 2. Related Work

Cross-View Datasets Cross-view geo-localization was introduced by Lin et al. [15] as a feature-translation problem between ground-level and overhead imagery. The field then consolidated around a set of retrieval benchmarks: CVUSA [36, 41], the cross-area splits of Vo et al. [32], and CVACT [16] all match (near-)centered street-view panoramas against overhead tiles, while University-1652 [43] replaces the ground side with UAV queries. VIGOR [45] adds a fine-grained benchmark requiring meter-level regression within the retrieved patch, and Shi et al. [25] extend KITTI [7] to a harder 3-DoF setting in which only a forward-facing camera is available. All of these remain urban or automotive.

Cross-View Localization Methods Methods differ mainly in how precisely they localize. The retrieval line learns one global descriptor per view, so a ground query and its true overhead tile become nearest neighbors in a shared embedding. Early work paired this idea with task-specific modules for the panorama–aerial geometry gap: NetVLAD aggregation [1, 10], a polar transform with re-weighting of informative regions [24], and a ViT [6] attending twice to zoom into relevant patches [46]. Sample4Geo [4] drops these modules: a ConvNeXt encoder [17] trained with a symmetric InfoNCE loss [20] and hard negatives from geography and embedding space set the retrieval state of the art at the time and transferred better to unseen regions. Others add a bird’s-eye view (BEV) of the ground image as a second, geometrically aligned cue, either warping the panorama directly [39] or reusing a pyramid-occupancy transform pre-trained for semantic BEV prediction [28].

Retrieval returns a tile, not a metric pose. A second line therefore estimates the camera pose inside a known aerial patch. SliceMatch [12] scores a ground descriptor against pose-dependent aerial descriptors aggregated over horizontal viewing-direction slices, keeping the best 3-DoF candidate. CCVPE [38] predicts a dense, orientation-aware location heatmap coarse-to-fine with a translation-equivariant encoder, and HC-Net [33] casts localization as homography estimation through a differentiable spherical transform. $\mathrm { F G } ^ { 2 }$ [37] matches fine-grained features from a frozen DINOv2 backbone [21]: following camera-to-BEV perception [14, 22], it lifts ground features into a 3D point cloud, learns where along the height to pool them onto a BEV plane, and aligns the result to aerial points via Procrustes.

LiDAR-based and Cross-Modal Localization A parallel line localizes from 3D geometry rather than imagery. Building on learned global descriptors for point-cloud place recognition [31], later work crosses modalities: VXP [13] retrieves a camera query against a LiDAR map in a shared embedding, CrossLoc3D [8] matches ground against aerial point clouds, and L2RSI [27] localizes LiDAR submaps against overhead imagery. Geometry is attractive here because it is sensed actively and therefore independent of illumination, yet all of these operate in urban environments and pair one query modality with one reference, rather than combining LiDAR with imagery on the ground side.

Only recently has cross-view localization moved offroad. Yuan et al. [40] contribute a dataset covering unstructured terrain, and MoViX [3] performs cross-view video localization on TartanDrive 2.0 [30]. Both, however, observe the scene through a single forward-facing camera, so the surrounding context that carries much of the signal in crossview matching is never available, and neither combines this view with LiDAR. GOOSELoc targets exactly this gap.

## 3. The GOOSELoc Dataset

GOOSELoc turns the ground-level imagery of the GOOSE dataset [19] into a benchmark for cross-view visual localization in unstructured terrain, as can be seen in Figure 1. Each of the 9,356 ground scenes is paired with a co-registered aerial orthophoto and an absolute ground-truth pose, so that the task is to localize a vehicle from a single groundlevel capture against a top-down aerial map. Unlike existing geo-localization benchmarks, which are dominated by structured urban street scenes, GOOSELoc covers a neareven mix of paved roads and unstructured terrain—forest tracks, field paths, and rural roads—and spans a full calendar year, exposing methods to strong seasonal and appearance change. We compare it in Table 1 with other currently available datasets.

## 3.1. Ground-View Input

A GOOSELoc scene is one synchronized 360<sup>◦</sup> observation from the ground. It comprises four time-synchronized RGB views, front, right, back, and left, from the roofmounted surround cameras of the GOOSE recording vehicle MuCAR-3 [19] at 90<sup>◦</sup> azimuth spacing, each 2448 × 1000 px, together with the 128-beam roof-mounted rotating Velodyne LiDAR sweep captured at the same instant: a single point cloud available for every scene, with a median of 200,000 points (from 30,000 in open terrain to 380,000 under dense canopy) and most returns within ∼66 m horizontally. We use the surround cameras rather than the windshield camera that GOOSE annotates for semantic segmentation because their full 360<sup>◦</sup> coverage matches the topdown geometry of the aerial reference. Releasing both modalities per scene lets GOOSELoc support image-only localization as well as the image-LiDAR fusion architecture we propose.

## 3.2. Ground Truth and Aerial Reference

The ground-truth pose of each scene comes from the vehicle’s INS trajectory: we use the RTK-corrected NCom pose (pose lat/pose lon) of the Oxford RT3000v3 INS where available and fall back to its raw GNSS fix otherwise. The pose is projected from WGS84 to UTM zone 32N (EPSG:25832), Bavaria’s native grid. Details on the INS localization quality are in the appendix.

As the aerial reference we use the Bavarian DOP20 digital orthophotos at their native ground resolution of 20 cm/px. We cache source orthophotos as 1 km×1 km tiles (5000 × 5000 px) and, for each scene, cut a 1024 × 1024 px crop (204.8 m × 204.8 m) at native resolution, compositing up to four adjacent source tiles when the crop straddles a tile boundary. Crops are north-aligned, so the image axes coincide with UTM easting and northing. Crucially, the crop is not centered on the vehicle: the ground-truth vehicle pixel is placed uniformly at random within the central 512 × 512 region (within ±256 px=±51.2 m of the tile center, independent of heading), so that the full 204.8 m of context is retained and the vehicle is always at least 51.2 m from any edge, yet never at a fixed location a method could exploit, following VIGOR’s practice of decoupling the query position from the tile center [45]. The localization target is this pixel, which must be recovered by cross-view matching. Because the orthophotos are flown on dates independent of the ground drives, every ground-aerial pair carries a builtin temporal domain gap (vegetation state, snow cover, construction), which we treat as a core property of the benchmark rather than noise to be removed. All tiles were flown in June, July, or August and therefore show summer conditions only; 95% date from 2024 and 5% from August 2025.

GOOSELoc is geographically confined to a ∼45 km × 16 km band of Upper Bavaria south of Munich (latitude $4 7 . 8 1 ^ { \circ } - 4 8 . 2 4 ^ { \circ }$ , longitude $1 1 . 5 9 ^ { \circ } - 1 1 . 8 1 ^ { \circ } )$ captured between July 2022 and July 2023, as shown in Figure 2. Each scene additionally carries a manually verified paved/unpaved terrain label, used both to characterize the dataset and to balance the splits (Sec. 3.3).

<table><tr><td>Dataset</td><td>Size</td><td>Ground view</td><td>Modalities</td><td>Environment</td><td>Seasonal gap</td><td>Localization</td></tr><tr><td>CVUSA [36, 41]</td><td>44.4k</td><td>Panorama</td><td>RGB</td><td>Urban / rural</td><td>No</td><td>Retrieval</td></tr><tr><td>CVACT [16]</td><td>137.2k</td><td>Panorama</td><td>RGB</td><td>Urban</td><td>No</td><td>Retrieval</td></tr><tr><td>VIGOR [45]</td><td>105.2k</td><td>Panorama</td><td>RGB</td><td>Urban</td><td>No</td><td>Retrieval + Metric</td></tr><tr><td>Cross-View KITTI [25]</td><td>30.2k</td><td>Forward</td><td>RGB</td><td>Urban / rural</td><td>No</td><td>Metric</td></tr><tr><td>University-1652 [43]</td><td>146.0k</td><td>Social Media</td><td>RGB</td><td>Urban (buildings-only)</td><td>No</td><td>Retrieval</td></tr><tr><td>MoViX / TartanDrive 2.0 [3, 30]</td><td>4.6k</td><td>Forward (video)</td><td>RGB</td><td>Off-road</td><td>Yes</td><td>Metric</td></tr><tr><td>McPed [40]</td><td>29.9k</td><td>Forward</td><td>RGB +LiDAR</td><td>Field / urban</td><td>No</td><td>Metric</td></tr><tr><td>GOOSELoc (ours)</td><td>9.4k</td><td>Surround 360°</td><td>RGB +LiDAR</td><td>Off-road / rural / urban</td><td>Yes</td><td>Metric</td></tr></table>

Table 1. GOOSELoc in the context of existing cross-view benchmarks. A GOOSELoc scene comprises four surround images, a LiDAR sweep, and a geo-referenced orthophoto crop. GOOSELoc is the only benchmark pairing surround-view imagery and LiDAR against official orthophotos, and the only one combining paved-to-off-road coverage with a controlled seasonal ground-aerial gap. McPed [40] is the closest multimodal relative: RGB + LiDAR from a single forward-facing $7 9 ^ { \circ }$ camera on a tracked robot, recorded in clear weather without a seasonal protocol, with SLAM-derived ground truth and per-sequence rather than geographic splits. MoViX shares the off-road setting and a ground-aerial temporal gap but relies on a single forward-facing RGB camera.

![](images/2459f51e7e77a497dce0de4ecd9ee26b2d2db768712d20b1aa0334f51ac4ada3.jpg)  
Figure 2. Geographic layout of GOOSELoc across a ∼45 km × 16 km band of Upper Bavaria south of Munich, shown as (a) the southern band and (b) the northern hub. Scenes are colored by split and train/dev/test occupy geographically partitioned regions. Basemap © OpenStreetMap contributors, © CARTO.

## 3.3. Composition and Splits

Table 2 summarizes the three splits. The dataset is close to terrain-balanced overall (45.8% paved, 54.2% unpaved), and both train and test are individually near-balanced (46.5% and 48.3% paved), so the paved/unpaved axis can be studied in isolation. Split Partitioning: We split the data by place rather than by drive, in the spirit of VIGOR’s cross-area protocol [45], which measures generalization to regions unseen during training. Therefore, we overlay a regular grid of 1 km × 1 km cells on the recording area and assign each cell, with every scene inside it, entirely to one split, so that train, dev, and test cover disjoint patches of ground. A model therefore cannot score well simply by recognizing a location it has already seen in training. The only contact between splits is at cell borders, where a drive that crosses into a neighboring cell is cut, with each part kept in its respective set. In the appendix, we show that this border contact has no significant effect on localization performance. Which cell goes to which split is chosen by a search that balances the three splits in size (≈70/10/20), terrain class and season (Table 2). Finally, the test set keeps only pose-verified scenes, whose GNSS fix and INS pose agree to within 5 m, so that test ground-truth positions are reliable.

<table><tr><td></td><td colspan="4">Season</td><td></td><td></td></tr><tr><td>Split</td><td></td><td>Winter Spring</td><td>Summer</td><td>Autumn</td><td>Paved</td><td>Total</td></tr><tr><td>train</td><td>1,387</td><td>1,976</td><td>1,341</td><td>1,726</td><td>2,988</td><td>6,430</td></tr><tr><td>dev</td><td>154</td><td>304</td><td>103</td><td>503</td><td>400</td><td>1,064</td></tr><tr><td>test</td><td>332</td><td>608</td><td>532</td><td>390</td><td>900</td><td>1,862</td></tr><tr><td>Total</td><td>1,873</td><td>2,888</td><td>1,976</td><td>2,619</td><td>4,288</td><td>9,356</td></tr></table>

Table 2. Composition of GOOSELoc. Scene counts per split by season and terrain. Scenes that are not paved are unpaved (45.8% paved overall).

Heading settings: We evaluate under two heading settings that differ only in the heading prior given to the model. The north-up setting provides no orientation information, as the model receives only north-aligned overhead imagery. The IMU-prior setting limits the heading uncertainty to $\pm 1 0 ^ { \circ }$ by providing overhead imagery that is pre-rotated with a random uniform mismatch of up $\mathrm { t o } \pm 1 0 ^ { \circ }$ , reflecting the coarse heading available in deployment. North-up therefore serves as the harder stress test. Figure 3 illustrates

![](images/174fdcf5ec04bc5f453687476ea3a19101809eaba18cd09f42fc329fbac225c7.jpg)

![](images/16a8e5a45a2dbe51da61959cf8da3cf094d14fe7d097465600d499378e1345c7.jpg)  
Figure 3. The two heading settings on one scene $( \star = \mathrm { G T }$ position, red = true heading, white dashed = reference $\ " \mathrm { u p } ^ { \mathrm { , 5 } } \mathrm { . }$ , cyan = admissible range of ∆θ). Left – north-up: no prior, recover the full 360<sup>◦</sup>. Right – IMU prior: $\pm 1 0 ^ { \circ }$ coarse prior, recover only the residual $| \Delta \theta | \le 1 0 ^ { \circ }$ (offset sampled uniformly in $[ - 1 0 ^ { \circ } , + 1 0 ^ { \circ } ]$ at train and test).

both settings.

## 4. Methodology

To establish reference performance on our benchmark, we provide multiple baselines. We first adapt four existing methods to our setting: CCVPE [38], Shi et al. [26], HC-Net [33] and $\mathrm { F G } ^ { 2 } \ [ 3 7 ]$ These serve as a competitive reference and show that current methods struggle in off-road environments. Since CCVPE only supports north-aligned input and Shi et al. requires a bounded heading prior, we report each method only in the corresponding setting. As CCVPE, Shi et al., HC-Net and $\mathrm { F G ^ { 2 } }$ are purely image-based and cannot exploit LiDAR, we add a second, multimodal baseline that fuses image and LiDAR features. This demonstrates that our dataset enables multimodal localization and provides an initial indication of its benefits. Implementation details for reproducing all methods are provided in the appendix.

## 4.1. Our multimodal baseline

We build our multimodal baseline on two frozen foundation encoders: DINOv3 [29] for the image inputs and Utonia [42] for the LiDAR point clouds. The two tasks in our benchmark, position and heading prediction, are handled by two separate heads, which are illustrated in Figure 4. The overhead tile and the four camera views are encoded by a DINOv3 ViT-L/16 that is kept frozen throughout. For the point cloud we likewise keep Utonia frozen and train only a LoRA adapter [9]. This adaptation is necessary because without it the LiDAR branch does not contribute meaningfully, which we attribute to the large domain shift between Utonia’s pre-training data and off-road GOOSELoc. We feed the 1024×1024 overhead tile to DINOv3, which yields a 64 × 64 patch grid, and a linear layer projects its feature dimension from 1024 to a 512-d match space. The four camera views are encoded likewise and their [CLS] tokens are projected to the same 512-d space. The LiDAR branch produces two features: a mean-pooled global descriptor and $K = 8$ vehicle-frame sector descriptors, obtained by binning each point by its ego-frame azimuth atan $2 ( y , x )$ into 8 equal $4 5 . 0 ^ { \circ }$ wedges and mean-pooling the per-point Utonia features within each wedge.

Position Prediction We project the overhead patch tokens, the four camera tokens and a global-mean LiDAR token into a shared $d { = } 5 1 2$ space with a learnable linear projection. The camera tokens additionally receive a learned azimuth embedding so the model can distinguish front from back. Rather than fusing the two modalities into a single query before matching, we let each modality match the tile on its own and fuse the two resulting score maps. A singlequery attention pool over the four camera tokens yields a camera query $q _ { \mathrm { c a m } } ,$ and the projected LiDAR token is the LiDAR query $q _ { \mathrm { l i d } }$ . Both are ℓ<sub>2</sub>-normalized and correlated with every projected patch $p _ { i j }$ of the $6 4 \times 6 4$ grid, giving one score map per modality,

$$
s _ { i j } ^ { m } = \frac { \gamma _ { m } } { \tau } \hat { q } _ { m } ^ { \top } p _ { i j } , \qquad m \in \{ \mathrm { c a m } , \mathrm { l i d } \} ,\tag{1}
$$

with the shared learned temperature τ (initialized to 0.07) and a learned per-modality gain $\gamma _ { m }$ parameterized in log space and initialized to $\gamma _ { m } = 1$ , which restores the sharpness freedom that the normalization removes.

In our experiments, we observed that the contributions of the LiDAR and RGB ground views vary with the scene. We therefore add an evidence gate that estimates how much each score map $s ^ { m }$ should contribute. First we turn each map into a distribution $\pi ^ { m } = \mathrm { \ s o f t m a x } _ { i j } ( s ^ { m } )$ which is used only inside the gate. From each $\pi ^ { m }$ we take three scale-free statistics that say how peaked and how unambiguous the match is: the normalized entropy $\textstyle - \sum _ { i j } \pi _ { i j } ^ { m } \bar { \log { \pi _ { i j } ^ { m } } } / \log ( 6 4 ^ { 2 } )$ , the peak probability max<sub>ij</sub> π<sup>m</sup><sub>ij</sub> , and the gap between the largest and the secondlargest value. Stacking these three numbers for both modalities gives a six-dimensional gate input $\beta ,$ , which a two-layer MLP maps to the weights

$$
w = n _ { \mathrm { a c t } } \cdot \mathrm { s o f t m a x } \big ( \mathrm { M L P } ( \beta ) \big ) \in \mathbb { R } ^ { 2 } .\tag{2}
$$

The factor $n _ { \mathrm { a c t } }$ , the number of active modalities, makes the weights sum to $n _ { \mathrm { a c t } }$ , so the scale of the fused logits stays the same when one modality is masked out. The statistics are detached: the gate re-weights the match evidence, but no gradient reaches the maps through it. Based on these weights, the maps are fused as

$$
\begin{array} { r } { s _ { i j } = w _ { \mathrm { c a m } } s _ { i j } ^ { \mathrm { c a m } } + w _ { \mathrm { l i d } } s _ { i j } ^ { \mathrm { l i d } } , \qquad H = \mathrm { s o f t m a x } _ { i j } ( s ) . } \end{array}\tag{3}
$$

The final distribution map H is then used to decode the position by soft-argmax and a KL divergence loss is used on a soft Gaussian target as objective. We use this dense target for two reasons specific to our setting. First, the ground-truth location is continuous and often falls near a patch boundary, where the two adjacent patches are almost equally correct: an InfoNCE [20] target, which is often used in contrastive settings, labels exactly one patch positive and penalizes its neighbor as a hard negative, whereas the soft-Gaussian spreads mass over neighboring patches by distance and barely penalizes a near-boundary prediction. Second, off-road terrain is highly self-similar, so several patches can be genuinely plausible; a one-hot target forces over-confidence against this irreducible ambiguity, while the KL-to-Gaussian objective only asks the model to concentrate mass locally and tolerates a calibrated spread.

![](images/de48653e60fb1f5b69bbf62436ba473c35b0432c7beac66e5934b3686a40718a.jpg)  
Figure 4. Our architecture. Frozen DINOv3 and Utonia yield a shared token set. In the position branch each modality correlates its own query with the DOP20 patch features, and an evidence gate weights the two resulting score maps before they are summed. The orientation branch correlates the camera and LiDAR-wedge tokens against the tile and feeds the match maps, with the camera [CLS] tokens, to a 72-bin heading classifier.

Heading Prediction Heading is the rotation that aligns the vehicle with the north-aligned tile. This rotation is invisible in appearance alone and it shows up only in where each vehicle-fixed sensor matches the tile. We reuse the projected patch tokens $p _ { i j } ,$ here with a stop-gradient, and write $c _ { c }$ and $u _ { k }$ for the projected camera and LiDAR-wedge tokens. For each surround camera we correlate its token with every patch and soft-max over the 64 × 64 grid to form a cross-view match map,

$$
M _ { c } ^ { \mathrm { c a m } } [ i , j ] = \operatorname * { s o f t m a x } _ { i j } \frac { \langle c _ { c } , p _ { i j } \rangle } { \tau } , \qquad c = 1 , \dots , 4 .\tag{4}
$$

Each camera points along a fixed vehicle bearing, so its match map peaks where that bearing meets the tile: ahead for the front camera, behind for the rear, and at $\pm 9 0 ^ { \circ }$ for the two side cameras. The head reads the vehicle’s heading off the angle of this four-peak pattern, a generalized Hough vote over orientations that is learned rather than hand-designed. Because the four bearings are observed jointly, it also resolves the front/back $( \theta \ \mathrm { v s . } \ \theta { + } 1 8 0 ^ { \circ } )$ ambiguity that a single forward camera cannot.

We stack the four maps as a four-channel image and pass them through a small two-layer CNN, $\begin{array} { r l } { \phi _ { \mathrm { c a m } } } & { { } = } \end{array}$ $\mathrm { C N N } _ { \mathrm { c a m } } ( M _ { 1 : 4 } ^ { \mathrm { c a m } } ) \ \in \ \mathbb { R } ^ { 5 1 2 }$ The strided convolutions keep a coarse $4 \times 4$ grid, so the descriptor retains where each peak lies. A global pool would discard precisely the location that carries the rotation. We treat the $K { = } 8$ vehicle-frame LiDAR wedges the same way: each pooled wedge is correlated against the patches, $\begin{array} { r l } { M _ { k } ^ { \mathrm { l i d } } [ i , j ] } & { { } = } \end{array}$ softmax $_ { i j } \langle u _ { k } , p _ { i j } \rangle / \tau$ , and the K maps are reduced by a second two-layer CNN to $\phi _ { \mathrm { l i d } } \in \mathbb { R } ^ { 5 1 2 }$ . A raw wedge descriptor is invariant to the tile’s orientation; only its match location moves as the vehicle turns, which is what makes it informative here. Two weak appearance cues complete the input. We keep the raw camera [CLS] tokens $z _ { 1 : 4 } ,$ and add an overhead-conditioned cross-attention read-out in which each camera token attends over the raw overhead tokens o under a fixed 2D positional encoding $\rho ,$

$$
a _ { c } = \mathrm { A t t n } \big ( W _ { q } z _ { c } , ~ W _ { k } o + \pmb { \rho } , ~ W _ { v } o + \pmb { \rho } \big ) \in \mathbb { R } ^ { d } .\tag{5}
$$

The positional encoding makes this read-out carry the match location, so it acts as a soft, learned counterpart of the explicit maps. The heading feature concatenates these blocks,

$$
f = \big [ z _ { 1 : 4 } ; a _ { 1 : 4 } ; \phi _ { \mathrm { c a m } } ; \phi _ { \mathrm { l i d } } \big ] ,\tag{6}
$$

and we decode heading with a 72-bin (5<sup>◦</sup>) circular classifier. The classifier can place mass on both θ and $\theta + 1 8 0 ^ { \circ }$ and commit to the better-supported mode, which is what the front/back symmetry requires. At test time we take the argmax bin refined by a circular centroid over its two neighbors to obtain sub- $- 5 ^ { \circ }$ resolution.

<table><tr><td></td><td></td><td colspan="4">dev</td><td colspan="4">test</td></tr><tr><td></td><td></td><td colspan="2">↓Localization (m)</td><td colspan="2"> $\downarrow 0 \mathrm { { r i e n t a t i o n } \it { ( ^ { \circ } ) } }$ </td><td colspan="2">↓Localization (m)</td><td colspan="2"> $\downarrow 0 \mathrm { r i e n t a t i o n } ( ^ { \circ } )$ </td></tr><tr><td>Setting</td><td>Methods</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>North-up</td><td>Predict center</td><td>38.9</td><td>40.9</td><td> $9 0 . 0$ </td><td>90.0</td><td>39.7</td><td>41.5</td><td>90.0</td><td>90.0</td></tr><tr><td></td><td>CCVPE (surround)</td><td> $3 3 . 6 { \pm } 0 . 4 $ </td><td> $2 5 . 7 \pm 1 . 3$ </td><td> $3 7 . 2 \pm 8 . 5$ </td><td> $1 6 . 7 { \pm } 1 2 . 2$ </td><td> $4 5 . 4 \pm 2 . 9$ </td><td> $3 7 . 0 { \pm } 2 . 9 $ </td><td> $5 2 . 8 { \pm } 5 . 0 $ </td><td> $3 5 . 6 { \pm } 1 7 . 0$ </td></tr><tr><td></td><td>HC-Net (surround)</td><td> $3 1 . 3 { \pm } 0 . 2 $ </td><td> $2 7 . 3 { \pm } 0 . 6 $ </td><td> $8 5 . 7 \pm 1 . 5$ </td><td> $8 4 . 2 { \pm } 4 . 8 $ </td><td> $3 1 . 8 { \pm } 0 . 4 $ </td><td> $2 8 . 6 { \pm } 0 . 2 $ </td><td> $8 7 . 0 { \pm } 0 . 5 $ </td><td> $8 4 . 4 \pm 1 . 2$ </td></tr><tr><td></td><td> $\mathrm { F G ^ { 2 } \ ( f r o n t ) }$ </td><td> $3 2 . 8 { \pm } 0 . 9 $ </td><td> $2 9 . 3 { \pm } 0 . 2 $ </td><td> $8 9 . 9 2 0 . 4 $ </td><td> $9 1 . 3 { \pm } 0 . 4 $ </td><td> $3 7 . 7 \pm 1 . 2$ </td><td> $3 5 . 1 { \pm } 0 . 8 $ </td><td> $9 2 . 0 { \pm } 0 . 2 $ </td><td> $9 3 . 4 \pm 0 . 1 $ </td></tr><tr><td></td><td> $\mathrm { F G } ^ { 2 } ( \mathrm { s u r r o u n d } )$ </td><td> $3 0 . 4 \pm 0 . 2$ </td><td> $2 8 . 0 { \pm } 0 . 3 $ </td><td> $8 9 . 7 \pm 0 . 1 $ </td><td> $9 0 . 4 \pm 0 . 1$ </td><td> $3 4 . 4 { \pm } 0 . 4$ </td><td> $3 2 . 1 { \pm } 0 . 3 $ </td><td> $9 2 . 5 { \pm } 0 . 1 $ </td><td> $9 3 . 1 { \pm } 0 . 3 $ </td></tr><tr><td></td><td>Ours (RGB)</td><td> $2 5 . 9 2 0 . 1$ </td><td> $2 1 . 8 { \pm } 0 . 2 $ </td><td> $3 0 . 2 \pm 4 . 0$ </td><td> $1 1 . 5 { \pm } 0 . 9 $ </td><td> $2 9 . 8 { \pm } 0 . 1 $ </td><td> $2 5 . 5 { \pm } 0 . 1 $ </td><td> $4 9 . 0 { \pm } 5 . 3 $ </td><td> $2 7 . 9 { \pm } 3 . 3 $ </td></tr><tr><td></td><td>Ours (LiDAR-only)</td><td> $2 3 . 9 2 0 . 4$ </td><td> $1 9 . 8 { \pm } 0 . 4 $ </td><td> $5 8 . 3 { \pm } 4 . 0 $ </td><td> $1 6 . 5 { \pm 2 . 0 } $ </td><td> $2 8 . 7 { \pm } 0 . 3 $ </td><td> $2 4 . 9 { \pm } 0 . 8 $ </td><td> $6 4 . 3 { \pm } 3 . 6 $ </td><td> $3 8 . 5 { \pm } 5 . 4 $ </td></tr><tr><td></td><td>Ours (RGB+L)</td><td> ${ 2 3 . 1 \pm 0 . 3 }$ </td><td> ${ \bf 1 8 . 8 \pm 0 . 2 }$ </td><td> ${ \bf 2 6 . 2 \pm 1 . 0 }$ </td><td> $\mathbf { 1 0 . 2 \pm 0 . 4 }$ </td><td> $2 7 . 1 { \pm } 0 . 3 $ </td><td> $2 2 . 2 { \pm } 0 . 6 $ </td><td> ${ \bf 4 6 . 3 \pm 1 . 1 }$ </td><td> ${ \bf 2 4 . 8 { \pm 0 . 3 } }$ </td></tr><tr><td> $\mathrm { I M U \pm 1 0 ^ { \circ } }$ </td><td>Shi et al. (front)</td><td>32.4</td><td>22.7</td><td>3.3</td><td>2.5</td><td>35.4</td><td>26.6</td><td>4.1</td><td>3.2</td></tr><tr><td></td><td>HC-Net (surround)</td><td> $2 9 . 8 { \pm } 0 . 5 $ </td><td> $2 6 . 2 { \pm } 0 . 5 $ </td><td> $1 2 . 4 { \pm } 0 . 6 $ </td><td> $8 . 9 { \pm } 0 . 5$ </td><td> $3 1 . 9 { \pm } 3 . 3 $ </td><td> $2 7 . 2 { \pm } 0 . 3 $ </td><td> $1 5 . 0 { \pm } 1 . 5 $ </td><td> $1 1 . 7 { \pm } 0 . 8 $ </td></tr><tr><td></td><td> $\mathrm { F G ^ { 2 } \ ( f r o n t ) }$ </td><td> $2 3 . 1 { \pm } 0 . 4 $ </td><td> $1 7 . 0 { \pm } 0 . 4 $ </td><td> $5 . 2 \pm 0 . 2$ </td><td> $3 . 7 \pm 0 . 1$ </td><td> $2 9 . 2 { \pm } 0 . 3 $ </td><td> $2 4 . 6 { \pm } 0 . 6 $ </td><td> $6 . 6 { \pm } 0 . 1$ </td><td> $4 . 8 { \pm } 0 . 1$ </td></tr><tr><td></td><td> $\mathrm { F G } ^ { 2 } ( \mathrm { s u r r o u n d } )$ </td><td> $1 9 . 8 { \pm } 0 . 2 $ </td><td> $1 4 . 5 { \pm } 0 . 3 $ </td><td> $4 . 3 { \pm } 0 . 1$ </td><td> $3 . 6 { \pm } 0 . 1$ </td><td> $2 5 . 0 { \pm } 0 . 4 $ </td><td> $2 0 . 2 { \pm } 0 . 6 $ </td><td> $5 . 1 \pm 0 . 1$ </td><td> $4 . 4 { \pm } 0 . 2 $ </td></tr><tr><td></td><td> $\operatorname { O u r s } \left( \operatorname { R G B } \right)$ </td><td> $2 1 . 1 { \pm } 0 . 0 $ </td><td> $1 6 . 3 { \pm } 0 . 1 $ </td><td> $3 . 8 \pm 0 . 1$ </td><td> $3 . 0 { \pm } 0 . 1$ </td><td> $2 5 . 1 { \pm } 0 . 1 $ </td><td> $2 0 . 7 { \pm } 0 . 2 $ </td><td> $4 . 6 { \pm } 0 . 1$ </td><td> $3 . 9 { \pm } 0 . 1$ </td></tr><tr><td></td><td> $\mathrm { O u r s } \ : ( \mathrm { L i D A R – o n l y } )$ </td><td> $1 6 . 9 { \pm } 0 . 2 $ </td><td> $1 1 . 3 { \pm } 0 . 2 $ </td><td> $3 . 4 \pm 0 . 0$ </td><td> $2 . 6 { \pm } 0 . 1$ </td><td> $2 1 . 2 { \pm } 0 . 2 $ </td><td> $1 6 . 1 { \pm } 0 . 1 $ </td><td> $4 . 2 \pm 0 . 1$ </td><td> $3 . 5 { \pm } 0 . 1$ </td></tr><tr><td></td><td>Ours (RGB+L)</td><td> ${ \bf 1 6 . 2 \pm 0 . 2 }$ </td><td> $\mathbf { 1 0 . 5 \pm 0 . 3 }$ </td><td> $3 . 7 \pm 0 . 1$ </td><td> $2 . 8 { \pm } 0 . 0 $ </td><td> $\mathbf { 1 9 . 5 \pm 0 . 3 }$ </td><td> ${ \bf 1 4 . 1 { \pm } 0 . 1 }$ </td><td> $4 . 6 { \pm } 0 . 1$ </td><td> $3 . 8 \pm 0 . 1$ </td></tr></table>

Table 3. Baseline localization on GOOSELoc (dev and test). Mean and median error for position (m) and orientation (<sup>◦</sup>), averaged over three seeds. Predict center is a non-learned trivial baseline. We evaluate $\mathrm { F G ^ { 2 } }$ on the native front camera and on our surround-view adaptation, CCVPE and HC-Net on our surround pseudo-panorama, and Shi et al. on the front camera (its native configuration). CCVPE and HC-Net appear only in the north-aligned setting, Shi et al. only under the IMU prior (bounded-prior method). Best per column within each setting in bold. The IMU-prior setting predicts a bounded residual $| \Delta \theta | \leq 1 0 ^ { \circ }$

Training Objective Both heads are trained jointly with a single composite objective,

$$
\mathcal { L } = \underbrace { \mathrm { K L } \big ( g _ { \sigma } \| H \big ) } _ { \mathrm { p o s i t i o n } } - \underbrace { \sum _ { k = 1 } ^ { 7 2 } \tilde { g } _ { k } \log \hat { y } _ { k } } _ { \mathrm { h e a d i n g } } ,\tag{7}
$$

sits near $9 0 °$ mean orientation error in the north-up setting, i.e. chance for a $3 6 0 ^ { \circ }$ heading.

where the position term is the KL divergence from the soft-Gaussian target $g _ { \sigma } \ ( \sigma { = } 1 . 5$ patches) to the predicted heatmap H, and the heading term is a soft circular cross-entropy between the 72-bin prediction $\hat { y }$ and a wrapped-Gaussian soft label $\tilde { g } \ ( \sigma { = } 1 0 ^ { \circ } )$ centered on the ground-truth heading, which smooths the hard bin labels and respects the circular topology.

LiDAR and cameras play complementary roles. LiDAR is the stronger single modality for localization: 3D structure is a strong position cue against the overhead map. Without a heading prior, however, it cannot orient, since road and forest geometry are largely front/back symmetric; the cameras contribute the appearance information that breaks this $1 8 0 ^ { \circ }$ ambiguity. The fusion inherits both strengths and improves on each: it is more accurate than LiDAR-only in every localization column (paired per-scene difference −1.7 m on test under the IMU prior) while matching the camera-level heading. The remaining north-up orientation error is heavytailed: a few $\mathrm { n e a r - 1 8 0 ^ { \circ } }$ front/back flips, most frequent for LiDAR-only, pull the mean far above the median, so typical scenes are predicted far better than the mean suggests.

## 5. Experiments

We compare our approach with CCVPE, Shi et al., HC-Net, $\mathrm { F G ^ { 2 } }$ and a trivial center-prediction baseline in Table 3. In the north-up setting all three variants of our approach outperform every baseline in localization on both dev and test, and the camera-bearing variants (RGB, RGB+L) additionally lead in orientation. The picture is more nuanced under the IMU prior: the fusion again yields the best localization, while Shi et al. attains slightly better orientation, at the cost of the weakest mean localization in this setting. Our RGB variant is slightly weaker than $\mathrm { F G ^ { 2 } }$ (surround) on dev but generalizes to test on par with it. It is worth noting that $\mathrm { F G ^ { 2 } }$

The heading prior helps both tasks: it collapses orientation error to a few degrees and, by resolving the tile rotation, also lowers localization error. Absolute errors nonetheless remain large throughout, leaving clear headroom on GOOSELoc and supporting our framing of these models as reference baselines.

## 5.1. Ablation

We label each test scene as paved or unpaved and tag it by season, and report median test error per terrain (Tab. 4) and season (Tab. 5); further architectural ablations are in the ap-

paved (pos 41.0 m, head 103°)
<table><tr><td></td><td></td><td colspan="2">Paved</td><td colspan="2">Unpaved</td></tr><tr><td>Setting</td><td>Method</td><td>Loc (m)</td><td>Ori(°)</td><td>Loc (m)</td><td>Ori(°)</td></tr><tr><td>North-up</td><td>FG2 (surround)</td><td>27.6±0.4</td><td>89.9±0.2</td><td>36.1±0.4</td><td>98.0±0.9</td></tr><tr><td></td><td>Ours (RGB)</td><td>25.5±0.7</td><td>28.6±3.3</td><td>25.5±0.3</td><td>27.6±4.2</td></tr><tr><td></td><td>Ours (LiDAR-only)</td><td>23.2±0.6</td><td>35.6±7.3</td><td>26.5±0.9</td><td>40.8±3.9</td></tr><tr><td></td><td>Ours (RGB+L)</td><td>21.2±0.3</td><td>24.8±1.6</td><td>22.9±0.7</td><td>25.3±1.1</td></tr><tr><td>IMU ±10°</td><td>FG2 (surround)</td><td>13.8±0.1</td><td>4.0±0.2</td><td>27.0±0.7</td><td>4.7±0.2</td></tr><tr><td></td><td>Ours (RGB)</td><td>19.5±0.5</td><td>3.9±0.1</td><td>21.5±0.2</td><td>3.8±0.0</td></tr><tr><td></td><td>Ours (LiDAR-only)</td><td>13.4±0.1</td><td>3.5±0.2</td><td>19.1±0.5</td><td>3.5±0.1</td></tr><tr><td></td><td>Ours (RGB+L)</td><td>12.2±0.3</td><td>3.8±0.1</td><td>15.4±0.3</td><td>3.8±0.1</td></tr></table>

Table 4. Performance by terrain (test, median ± std over 3 seeds).

<table><tr><td colspan="2">Setting</td><td colspan="2">Winter</td><td colspan="2">Spring</td><td colspan="2">Summer</td><td colspan="2">Autumn</td></tr><tr><td colspan="2">/ Method</td><td>Loc(m)</td><td>Ori(°)</td><td>Loc(m)</td><td>Ori(°)</td><td>Loc(m)</td><td>Ori(°)</td><td>Loc(m)</td><td>Ori(°)</td></tr><tr><td>North-up</td><td>FG2</td><td>28.4±1.2</td><td>80.3±0.8</td><td>28.5±0.7</td><td>90.2±0.9</td><td>36.5±0.6</td><td>100.1±1.5</td><td>34.6±0.4</td><td>108.5±0.6</td></tr><tr><td></td><td>RGB</td><td>25.5±0.7</td><td>26.9±2.9</td><td>24.3±0.1</td><td>36.5±10.2</td><td>26.3±0.6</td><td>28.3±1.2</td><td>25.9±0.2</td><td>19.5±2.7</td></tr><tr><td></td><td>L-only</td><td>23.9±1.1</td><td>44.4±4.4</td><td>23.5±1.2</td><td>25.7±4.8</td><td>26.1±0.7</td><td>44.8±1.4</td><td>26.7±1.5</td><td>45.3±15.1</td></tr><tr><td></td><td>RGB+L</td><td>22.0±1.0</td><td>25.3±0.7</td><td>20.3±0.5</td><td>30.8±0.1</td><td>24.1±0.5</td><td>28.8±2.3</td><td>22.7±0.8</td><td>16.3±1.4</td></tr><tr><td>IMU</td><td>FG2</td><td>16.9±0.6</td><td>4.6±0.1</td><td>14.2±0.1</td><td>4.0±0.2</td><td>26.6±1.1</td><td>4.8±0.2</td><td>25.1±1.0</td><td>4.5±0.3</td></tr><tr><td></td><td>RGB</td><td>19.0±0.5</td><td>4.3±0.1</td><td>18.4±0.7</td><td>3.6±0.2</td><td>22.8±0.1</td><td>3.7±0.1</td><td>23.5±0.3</td><td>4.0±0.2</td></tr><tr><td></td><td>L-only</td><td>16.8±0.2</td><td>4.0±0.2</td><td>13.1±0.4</td><td>3.3±0.1</td><td>17.1±0.1</td><td>3.3±0.1</td><td>19.5±0.7</td><td>4.0±0.1</td></tr><tr><td></td><td>RGB+L</td><td>13.4±0.7</td><td>4.2±0.3</td><td>11.7±0.0</td><td>3.3±0.1</td><td>15.0±0.1</td><td>3.8±0.1</td><td>17.3±0.3</td><td>4.1±0.2</td></tr></table>

Table 5. Performance by season (test, median $\pm \ s t \mathbf { d }$ over 3 seeds).

pendix.

Terrain: $\mathrm { F G ^ { 2 } }$ is competitive only on paved roads with a prior, its native KITTI regime, and its error roughly doubles off-road for the IMU-prior setting, whereas our fusion degrades far less and is best in the off-road regime GOOSE-Loc targets. For orientation, the camera variants are essentially terrain-invariant, while LiDAR-only degrades offroad; the fusion keeps camera-level orientation on both terrains.

Season: Localization is hardest in the leaf-on seasons, where dense canopy occludes the overhead view. The fusion localizes best in every season and setting. Under the IMU prior, the LiDAR advantage over RGB grows from winter to summer, consistent with geometry being largely invariant to foliage. Orientation follows a different pattern: camera orientation is weakest in spring and strongest in autumn, LiDAR-only shows the mirror image, and the fusion is best in winter and autumn. $\mathrm { F G ^ { 2 } }$ remains near chance throughout.

Qualitative Results: Figure 5 shows typical successes and the two failure modes: $\mathbf { a } \sim 1 8 0 ^ { \circ }$ front/back flip, and a diffuse posterior whose soft-argmax drifts from the true position.

## 6. Conclusion

We introduced GOOSELoc, a multimodal benchmark for metric ground-to-aerial localization spanning the structured-to-unstructured spectrum from paved roads to off-road forest and field tracks. Fine-tuning four stateof-the-art image-only methods reveals that the accuracy achieved on on-road benchmarks decreases significantly off-road and worsens further without a heading prior. Our multimodal baseline narrows this gap and reveals a clear division of labor between the modalities. LiDAR geometry is the stronger position cue and provides the best localization even on its own. In contrast, camera appearance resolves absolute heading. Fusing the two modalities retains both capabilities.

paved (pos 6.9 m, head 8°)  
![](images/f9949b52ca5af00f5237103f64fa94fd6fa0cdfee86135018b9e53cfc498ceee.jpg)

offroad (pos 3.1 m, head 3°)  
![](images/0f19bfd6cf58b1d6af0424545bde39081aaabf53861b656d7c9cd031790b788e.jpg)

![](images/fc2bdb08d1507dfee5ad2a6028aa53757f48d8dea81a46046c0968ad0170b2cd.jpg)

offroad (pos 53.9 m, head 151°)  
![](images/882b830d90f61e4247e6f9033985ae505bedc0ec1a3098a8ba02eae4bbf96356.jpg)  
Figure 5. Qualitative predictions on GOOSELoc (test, northup, no prior). Gold ⋆ / arrow: ground-truth position / heading, red ◦/arrow: prediction; the viridis overlay is the predicted position posterior.

These results also reveal open research questions. Absolute errors remain at tens of meters and tens of degrees without a prior, far from deployment-grade, indicating that cross-view localization in unstructured terrain is a broader issue rather than a failing of any single method. GOOSELoc is also deliberately narrow: a single calibrated sensor rig in one region of Upper Bavaria, matched against summer-only orthophotos. Fixing sensor, geography, and aerial source isolates the terrain and seasonal axes we study, but limits external validity: our results characterize this regime, and geographic or sensor generalization must be established on data from other regions and platforms. Broadening this coverage and incorporating multi-frame reasoning are natural next steps. We release the data, splits, and baselines, and hope GOOSELoc serves as a reference point for cross-view localization beyond roads.

Acknowledgments. The authors gratefully acknowledge the Institute for Distributed Intelligent Systems for granting the computing time and providing access to the Monacum One GPU cluster at the University of the Bundeswehr Munich.

This work was supported by the Federal Office of Bundeswehr Equipment, Information Technology and In-Service Support (BAAINBw).

This work is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant number 583569863.

## References

[1] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. NetVLAD: CNN architecture for weakly supervised place recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5297–5307, 2016. 2

[2] Eli Brosh, Matan Friedmann, Ilan Kadar, Lev Yitzhak Lavy, Elad Levi, Shmuel Rippa, Yair Lempert, Bruno Fernandez-Ruiz, Roei Herzig, and Trevor Darrell. Accurate visual localization for automotive applications. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019. 1

[3] Zhiyun Deng, Dongmyeong Lee, Amanda Adkins, Jesse Quattrociocchi, Christian Ellis, and Joydeep Biswas. Spatiotemporal contrastive learning for cross-view video localization in unstructured off-road terrains. arXiv preprint arXiv:2506.05250, 2025. 3, 4

[4] Fabian Deuser, Konrad Habel, and Norbert Oswald. Sample4Geo: Hard negative sampling for cross-view geolocalisation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16847–16856, 2023. 1, 2

[5] Fabian Deuser, Wejdene Mansour, Hao Li, Konrad Habel, Martin Werner, and Norbert Oswald. Temporal resilience in geo-localization: adapting to the continuous evolution of urban and rural environments. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 479– 488, 2025. 2

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2

[7] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. The international journal of robotics research, 32(11):1231–1237, 2013. 1, 2

[8] Tianrui Guan, Aswath Muthuselvam, Montana Hoover, Xijun Wang, Jing Liang, Adarsh Jagan Sathyamoorthy, Damon Conover, and Dinesh Manocha. CrossLoc3D: Aerial-ground cross-source 3D place recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11335–11344, 2023. 3

[9] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. LoRA: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022. 5

[10] Sixing Hu, Mengdan Feng, Rang M. H. Nguyen, and Gim Hee Lee. CVM-Net: Cross-view matching network for image-based ground-to-aerial geo-localization. In Proceed ings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2

[11] Peng Jiang, Philip Osteen, Maggie Wigness, and Srikanth Saripalli. RELLIS-3D dataset: Data, benchmarks and analysis. In 2021 IEEE international conference on robotics and automation (ICRA), pages 1110–1116. IEEE, 2021. 1

[12] Ted Lentsch, Zimin Xia, Holger Caesar, and Julian FP Kooij. SliceMatch: Geometry-guided aggregation for cross-view pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17225–17234, 2023. 3

[13] Yun-Jin Li, Mariia Gladkova, Yan Xia, Rui Wang, and Daniel Cremers. VXP: Voxel-cross-pixel large-scale image-LiDAR place recognition. arXiv preprint arXiv:2403.14594, 2024. 3

[14] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. BEVFormer: learning bird’s-eye-view representation from LiDAR-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2020–2036, 2024. 3

[15] Tsung-Yi Lin, Serge Belongie, and James Hays. Cross-view image geolocalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 891–898, 2013. 1, 2

[16] Liu Liu and Hongdong Li. Lending orientation to neural networks for cross-view geo-localization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5624–5633, 2019. 1, 2, 4

[17] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF conference on com puter vision and pattern recognition, pages 11976–11986, 2022. 2

[18] Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman. 1 year, 1000 km: The Oxford RobotCar dataset. The International Journal ofRobotics Research, 36(1):3–15, 2017. 2

[19] Peter Mortimer, Raphael Hagmanns, Miguel Granero, Thorsten Luettel, Janko Petereit, and Hans-Joachim Wuensche. The GOOSE dataset for perception in unstructured environments. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 14838–14844. IEEE, 2024. 1, 2, 3

[20] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 2, 6

[21] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.

DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3

[22] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3D. In European conference on computer vision, pages 194–210. Springer, 2020. 3

[23] Torsten Sattler, Will Maddern, Carl Toft, Akihiko Torii, Lars Hammarstrand, Erik Stenborg, Daniel Safari, Masatoshi Okutomi, Marc Pollefeys, Josef Sivic, et al. Benchmarking 6DOF outdoor visual localization in changing conditions. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 8601–8610, 2018. 2

[24] Yujiao Shi, Liu Liu, Xin Yu, and Hongdong Li. Spatialaware feature aggregation for image based cross-view geolocalization. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019. 2

[25] Yujiao Shi, Xin Yu, Liu Liu, Dylan Campbell, Piotr Koniusz, and Hongdong Li. Accurate 3-DoF camera geo-localization via ground-to-satellite image matching. IEEE transactions on pattern analysis and machine intelligence, 45(3):2682– 2697, 2022. 1, 2, 4

[26] Yujiao Shi, Fei Wu, Akhil Perincherry, Ankit Vora, and Hongdong Li. Boosting 3-DoF ground-to-satellite camera localization accuracy via geometry-guided cross-view transformer. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 21459–21469. IEEE, 2023. 2, 5, 13

[27] Ziwei Shi, Xiaoran Zhang, Wenjing Xu, Yan Xia, Yu Zang, Siqi Shen, and Cheng Wang. L2RSI: Cross-view LiDARbased place recognition for large-scale urban scenes via remote sensing imagery. Advances in Neural Information Processing Systems, 38:160681–160704, 2026. 3

[28] Tavis Shore, Simon Hadfield, and Oscar Mendez. BEV-CV: Birds-eye-view transform for cross-view geo-localisation. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11048–11055. IEEE, 2024. 2

[29] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025. 5

[30] Matthew Sivaprakasam, Parv Maheshwari, Mateo Guaman Castro, Samuel Triest, Micah Nye, Steve Willits, Andrew Saba, Wenshan Wang, and Sebastian Scherer. TartanDrive 2.0: More modalities and better infrastructure to further selfsupervised learning research in off-road driving tasks. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12606–12606. IEEE, 2024. 3, 4

[31] Mikaela Angelina Uy and Gim Hee Lee. PointNetVLAD: Deep point cloud based retrieval for large-scale place recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4470–4479, 2018. 3

[32] Nam N Vo and James Hays. Localizing and orienting street views using overhead imagery. In European conference on computer vision, pages 494–509. Springer, 2016. 2

[33] Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, and Yu Zhang. Fine-grained cross-view geo-localization using a

correlation-aware homography estimator. Advances in Neu ral Information Processing Systems, 36:5301–5319, 2023. 2, 3, 5, 13

[34] Maggie Wigness, Sungmin Eum, John G. Rogers, David Han, and Heesung Kwon. A RUGD dataset for autonomous navigation and visual perception in unstructured outdoor environments. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5000–5007, 2019. 1

[35] Michael G Wing, Aaron Eklund, and Loren D Kellogg. Consumer-grade global positioning system (GPS) accuracy and reliability. Journal offorestry, 103(4):169–173, 2005. 1

[36] Scott Workman, Richard Souvenir, and Nathan Jacobs. Wide-area image geolocalization with aerial reference im agery. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 3961–3969, 2015. 1, 2, 4

[37] Zimin Xia and Alexandre Alahi. FG<sup>2</sup>: Fine-grained crossview localization by fine-grained feature matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6362–6372, 2025. 2, 3, 5, 13

[38] Zimin Xia, Olaf Booij, and Julian FP Kooij. Convolutional cross-view pose estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3813–3831, 2023. 2, 3, 5, 13

[39] Junyan Ye, Zhutao Lv, Weijia Li, Jinhua Yu, Haote Yang, Huaping Zhong, and Conghui He. Cross-view image geo localization with Panorama-BEV co-retrieval network. In European Conference on Computer Vision, pages 74–90. Springer, 2024. 2

[40] Xia Yuan, Kaiyang Wang, Riyu Qin, and Jiachen Xu. A multi-modality ground-to-air cross-view pose estimation dataset for field robots. Scientific Data, 12(1):754, 2025. 3, 4

[41] Menghua Zhai, Zachary Bessinger, Scott Workman, and Nathan Jacobs. Predicting ground-level scene layout from aerial imagery. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 867–875, 2017. 1, 2, 4

[42] Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, and Hengshuang Zhao. Utonia: Toward one encoder for all point clouds. arXiv preprint arXiv:2603.03283, 2026. 5

[43] Zhedong Zheng, Yunchao Wei, and Yi Yang. University-1652: A multi-view multi-source benchmark for dronebased geo-localization. In Proceedings of the 28th ACM international conference on Multimedia, pages 1395–1403, 2020. 2, 4

[44] Ni Zhu, Juliette Marais, David Betaille, and Marion Ber-´ bineau. GNSS position integrity in urban environments: A review of literature. IEEE Transactions on Intelligent Transportation Systems, 19(9):2762–2778, 2018. 1

[45] Sijie Zhu, Taojiannan Yang, and Chen Chen. VIGOR: Crossview image geo-localization beyond one-to-one retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3640–3649, 2021. 1, 2, 3, 4

[46] Sijie Zhu, Mubarak Shah, and Chen Chen. TransGeo: Transformer is all you need for cross-view image geo-localization.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1162–1171, 2022. 2

## A. GOOSELoc Pose Quality

The Oxford RT3000v3 INS system provides status codes that we can use to quantify the expected error in the ground truth poses used in the different splits of GOOSELoc.<sup>1</sup> These status codes are determined based on common indicators like the number of satellites visible for the GPS receiver INS system, the geometric quality of the satellites position in the sky and the availability of ground network based GNSS correction data via NTRIP.

For the status code RTK Integer one can expect cmlevel positioning and from status code RTK Float dmlevel positioning. We ensured that for the splits train and test we maintain a close balance based on these INS status codes (Tab. A1).

<table><tr><td rowspan="2">Split Scenes</td><td rowspan="2"></td><td colspan="3">GNSS position mode (%)</td></tr><tr><td>RTK Integer (cm)</td><td>RTK Float (dm)</td><td>Other</td></tr><tr><td>train</td><td>6430</td><td>74.4</td><td>10.4</td><td>15.2</td></tr><tr><td>dev</td><td>1064</td><td>89.6</td><td>5.9</td><td>4.5</td></tr><tr><td>test</td><td>1862</td><td>70.4</td><td>9.0</td><td>20.6</td></tr></table>

Table A1. Ground truth pose quality per split, decoded from the gps pos mode byte of the NCOM packets; rows sum to 100%. “Other” groups the status codes Differential, SPS and packets with no position solution at all. Pose quality was not a split constraint, but care was taken to maintain a good balance for the train and test split.

Simply expecting the scenes in category “Other” to have a much poorer ground-truth pose misrepresents the capability of the INS system. The modes RTK Integer and RTK Float can effectively only be achieved under open sky conditions, where multiple satellites have a direct line of sight to the dual antenna setup on the roof of MuCAR-3. This means that any scenes recorded in or along a forested area suffer from the canopy effect in their GNSS positioning. This is reflected in the scene statistics, with only 39.7% of the GOOSELoc scenes recorded within an OpenStreetMap tile marked as landuse=forest having an RTK Integer/Float status. In comparison, 85.6% of all scenes more than 25 m away from a forest have an RTK Integer/Float status (Fig. A1).

The Oxford RT3000v3 INS system uses a high-grade IMU system to maintain accurate localization even under harsh INS conditions. The routes with many scenes with lower GNSS heading solutions in GOOSELoc do not contain many turns and contain scenes with RTK accuracy in between, allowing the INS system to keep the accumulated errors from an IMU-only localization minimal.

![](images/cc134714a2db79115b63c671484cd53bf5e32e5d8146737729f6925efa2dc7af.jpg)  
Figure A1. Geographic layout of GOOSELoc in terms of tree cover and the GNSS heading solution. Basemap © Open-StreetMap contributors, © CARTO.

When manually inspecting the scenes with lower status codes in each split, we could not observe any mismatches. A human visual inspection without adequate tooling cannot recognize position mismatches at dm-scale. The fact that the best-performing models under ideal open sky conditions like in the category “Paved” currently have a median localization error larger than 10 m gives us confidence that the accuracy of the ground truth data with this type of INS system is suitable as training signal and as a test benchmark.

## B. $\mathbf { F } \mathbf { G } ^ { 2 }$ baseline

$\mathrm { F G ^ { 2 } }$ [37] ingests a single forward-facing ground image, so we apply it to each of the four surround cameras independently rather than fusing them into a single representation. For camera c with fixed azimuth offset $\delta _ { c } \in \mathsf { \Gamma }$ $\{ 0 ^ { \circ } , 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \}$ , we rotate the aerial tile by $\delta _ { c }$ so that camera c acts as a virtual forward view, and run the unmodified $\mathrm { F G ^ { 2 } }$ on the pair (image<sub>c</sub>, rotated tile). At training time we sample one camera per example, which acts as a 4× orientation augmentation. At inference we run all four cameras, map each predicted pose back into the common tile frame, and fuse them by a confidence-weighted average. Each camera’s confidence is the total cross-view matching probability of its weighted-Procrustes solution. The $\mathrm { F G ^ { 2 } }$ network itself is left unchanged; for a detailed description of the approach we refer to the original paper.

## C. CCVPE baseline

CCVPE [38] is a panoramic cross-view method: it consumes a single $3 6 0 ^ { \circ }$ ground panorama together with a north-up aerial patch covering the query’s neighborhood, and jointly predicts a dense location-probability map over the patch and the camera orientation, the latter obtained by matching the ground descriptor against rolled (rotated) copies of the aerial descriptor. Because CCVPE ingests one panoramic image, we fuse the four surround cameras at the input level rather than per camera as in $\mathrm { F G ^ { 2 } }$ : we place each camera in its $9 0 ^ { \circ }$ azimuth sector $\delta _ { c } \in \{ 0 ^ { \circ } , 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \}$ and concatenate them in order (front, right, back, left) into a single equirectangular-style panorama. The aerial branch receives the DOP20 tile resized to CCVPE’s $5 1 2 \times 5 1 2$ input and kept north-up. Since orientation is estimated internally, no heading prior is supplied: the network recovers the full 360<sup>◦</sup> vehicle yaw jointly with position in a single forward pass. We leave the CCVPE architecture and its training recipe unchanged. For a detailed description of their approach we refer to the original paper.

We report CCVPE only in the prior-free (north-up) setting. Unlike methods that consume a heading prior by rotating the aerial tile, CCVPE keeps the tile north-up and searches over panorama rotations, so a prior can only restrict that search. In a control experiment on the original run, a perfect heading prior left localization unchanged (31.2 vs. 31.1 m median on matched dev scenes): an IMUprior variant would reproduce the north-up localization numbers while trivializing heading to the prior itself. We report CCVPE from its best-position dev checkpoint.

## D. Shi et al. baseline

The method of Shi et al. [26] estimates rotation by neural optimization and translation by dense spatial correlation between a ground-view BEV projection and the aerial image. It assumes a bounded pose prior and normalizes rotation to the prior range, so it applies only to the IMU setting. We use the single front camera, its native KITTI configuration, with the real Basler intrinsics rescaled to the expected 1024 × 256 input. The aerial tile is downsampled to $5 1 2 \times 5 1 2 ( 0 . 4 0 \mathrm { m / p x ) }$ so that the correlation search covers the full tile, and the translation search range is set to 51.2 m accordingly. The ground-plane homography uses a camera height of 2.0 m, our best estimate for the roof-mounted rig. The authors pretrained checkpoints are not publicly downloadable, so we initialize the VGG-16 backbones from ImageNet and train the full model (7.2 M parameters) on our data. The network and losses are unchanged.

## E. HC-Net baseline

HC-Net [33] regresses a correlation-aware homography between a spherically projected ground panorama and the aerial patch, giving sub-pixel position and orientation. We feed it the same four-camera pseudo-panorama as CCVPE, projected to a $5 1 2 \times 5 1 2$ quasi-BEV with HC-Net’s own spherical transform, and the DOP20 tile resized to 512 × 512. We initialize from the released VIGOR checkpoint and fine-tune the full network (11.3 M parameters). One adaptation is required for our setting: HC-Net’s orientation loss assumes residuals within $\pm 4 5 ^ { \circ }$ , so we wrap the orientation residual $\mathrm { t o } \pm 1 8 0 ^ { \circ }$ for north-up training. In the IMU setting the prior enters by pre-rolling the pseudo-panorama toward north. HC-Net has no orientation search to restrict. Everything else follows the authors recipe.

## F. Implementation details

All learned baselines and all variants of our model are trained with three seeds and tables report mean and standard deviation over seeds. In the IMU setting, every method is evaluated with the same deterministic per-scene prior error of up $\mathrm { t o } \pm 1 0 ^ { \circ }$

For $\mathrm { F G ^ { 2 } }$ we fine-tune from the authors’ KITTI checkpoint for 30 epochs at effective batch 16 using Adam with a learning rate $1 0 ^ { - 5 }$ , keeping the original InfoNCE and virtual-correspondence losses. Ground images are resized to $3 6 4 \times 1 2 3 2$ and the aerial tile to $5 0 4 \times 5 0 4$ . We apply no image augmentation, the four-camera variant samples one surround camera per scene during training. $\mathrm { F G ^ { 2 } }$ updates 60.6 M parameters. For Shi et al. we train 15 epochs at batch 4 with Adam, learning rate $1 0 ^ { - 4 }$ dropping to $1 0 ^ { - 5 }$ after the first epoch, following the authors KITTI schedule. For HC-Net we train 30 epochs at batch 16 with AdamW under a one-cycle schedule with a peak learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $5 \times 1 0 ^ { - 4 }$ , gradient clipping 1.0, six GRU refinement iterations, following the authors VIGOR recipe. CCVPE trains for 15 epochs at batch 8 with Adam at $1 0 ^ { - 4 }$

Our model trains for 30 epochs at effective batch 16 with AdamW under a cosine schedule with one warm-up epoch, using a learning rate of $1 0 ^ { - 3 }$ for the projection layers and $5 \times 1 0 ^ { - 4 }$ for the heads, weight decay of $1 0 ^ { - 4 }$ and parameters without weight dimensions (norms, embeddings, the temperature, and the per-modality gains) are excluded from weight decay. The only augmentation is applied to the LiDAR point cloud: random dropout of up to 20% of the points, per-point Gaussian jitter $( \sigma = 0 . 0 5 \mathrm { m } )$ , global isotropic scaling of $\pm 3 \%$ , and a random $\pm 2 ^ { \circ }$ yaw rotation. The point cloud is subsampled to 8,000 points. The LoRA adapter on Utonia uses rank 8 and $\alpha = 1 6$ , adding 1.05 M trainable parameters. During training we apply persample modality dropout in the fusion: with probability 0.3 the cameras are dropped, with probability 0.15 the LiDAR is dropped, and with probability 0.55 both modalities are kept. In total the trainable surface is 3.6 M parameters for the image-only model, 6.2 M for the LiDAR-only variant, and 7.4 M with both branches of which the fusion gate accounts for roughly 300 parameters, an order of magnitude below $\mathrm { F G ^ { 2 } \ ' _ { s } }$ 60.6 M. Experiments ran on a DGX-2 with 16 V100 32 GB GPUs. Our model trains data-parallel across eight GPUs, each baseline on a single GPU.

## G. Additional ablations

## G.1. Robustness to the heading prior

The IMU setting rotates the aerial tile by the supplied heading prior before matching, so the model only has to resolve a bounded residual $| \Delta \theta | \le 1 0 ^ { \circ }$ . Table A2 shows how accurate that prior must actually be: at test time we corrupt it by a fixed offset and re-evaluate. Localization is robust for all three variants, staying within ∼1 m of the perfectprior value out to $\pm 2 0 ^ { \circ }$ and losing only ∼3 m at a gross $\pm 4 5 ^ { \circ }$ offset. Orientation is, as expected, more sensitive: the classifier holds a few degrees while the prior is good to $\pm 1 0 ^ { \circ }$ and rises to ${ \sim } 1 8 ^ { \circ }$ once the true residual leaves the trained range. Three points follow: first, the model needs only a coarse heading prior to realize most of the localization benefit. Second the fusion keeps its full margin over LiDAR-only at every prior quality and third all three variants degrade at the same rate, so prior tolerance does not depend on the sensor mix.

## H. Self-similarity of aerial tiles

For each test scene we measure how many locations in the aerial tile look like the true one. We take frozen, nonfinetuned DINOv3 ViT-L/16 patch tokens of the north-up $1 0 2 4 ^ { 2 }$ tile (64 × 64 patches of $3 . 2 \times 3 . 2 \mathrm { m } )$ , pool a $\mathbf { \varepsilon } _ { 1 3 \times 3 }$ de-

<table><tr><td rowspan="2">Prior |∆θ|</td><td colspan="2">Ours (RGB)</td><td colspan="2">Ours (RGB+L)</td><td colspan="2">Ours (LiDAR-only)</td></tr><tr><td>Loc</td><td>Ori</td><td>Loc</td><td>Ori</td><td>Loc</td><td>Ori</td></tr><tr><td> $0 ^ { \circ }$ </td><td>21.3</td><td>4.6</td><td>13.7</td><td>4.6</td><td>16.2</td><td>0.3</td></tr><tr><td> $5 ^ { \circ }$ </td><td>21.1</td><td>3.1</td><td>13.9</td><td>3.0</td><td>16.2</td><td>2.7</td></tr><tr><td> $1 0 ^ { \circ }$ </td><td>20.9</td><td>3.9</td><td>14.4</td><td>3.8</td><td>16.4</td><td>3.5</td></tr><tr><td> $2 0 ^ { \circ }$ </td><td>21.8</td><td>6.9</td><td>14.8</td><td>6.4</td><td>17.1</td><td>6.8</td></tr><tr><td> $4 5 ^ { \circ }$ </td><td>24.0</td><td>18.4</td><td>17.1</td><td>18.2</td><td>19.7</td><td>18.8</td></tr></table>

Table A2. Robustness to heading-prior error (test, median; Loc in m, Ori in $^ { \circ } \ ;$ seed-0 checkpoints). As the injected prior error $| \Delta \theta |$ grows, localization stays nearly flat up to $\pm 2 0 ^ { \circ }$ and loses only ∼3 m at $\pm 4 5 ^ { \circ }$ ; orientation stays at a few degrees up to $\pm 1 0 ^ { \circ }$ and degrades gracefully beyond $\left( 0 ^ { \circ } = \right.$ perfect prior). A coarse prior suffices, and the fusion advantage persists at every prior quality.

scriptor at the GT position, and compute its cosine similarity to every patch outside a 16 m exclusion window. A patch with similarity above a threshold τ counts as a look-alike. The probe shares no weights with any evaluated method except our own model’s frozen encoder.
<table><tr><td></td><td>Paved</td><td>Unpaved</td></tr><tr><td>Look-alike positions  $\left( \tau = 0 . 8 \right)$ </td><td> $7 9 ( 1 . 9 \% )$ </td><td>101 (2.5%)</td></tr><tr><td>Look-alike fraction  $\left( \tau { = } 0 . 7 \right)$ </td><td>5.6%</td><td>7.8%</td></tr></table>

Table A3. Perceptual aliasing by terrain (test, per-scene medians, unpaved $> \mathrm { p a v e d } )$ . Unpaved scenes contain 1.28× more look-alike positions at $\tau { = } 0 . 8$

<table><tr><td></td><td>Winter</td><td>Spring</td><td>Summer</td><td>Autumn</td></tr><tr><td>Look-alike positions (τ=0.8)</td><td>63</td><td>79</td><td>117</td><td>102</td></tr></table>

Table A4. Perceptual aliasing by season (median per-scene lookalike count). Aliasing peaks in the leaf-on seasons, matching where the appearance-based baselines degrade most.

Localization error rises with aliasing for every method. The correlation is strongest for the matching-based baselines $\mathrm { F G ^ { 2 } }$ and Shi et al.. Correlations are weaker in the north-up setting, where orientation ambiguity dominates the position error, and weakest for HC-Net, whose errors stem mainly from its ground-plane homography assumption. The seasonal profile mirrors the localization results: aliasing peaks in summer, exactly where the LiDAR advantage over RGB is largest.

## I. Aerial imagery statistics

All aerial tiles are official DOP20 digital orthophotos of the Bavarian survey administration (Bayerische Vermessungsverwaltung), published as open data under CC BY 4.0. Each scene uses a north-aligned 1024×1024 crop at 0.20 m ground sampling distance, covering 204.8 × 204.8 m centered near the scene position. The underlying survey flights all took place in the leaf-on season (June to August), so the overhead view is summer-only regardless of the season in which the ground data was recorded; Tab. A6 lists the capture dates per split. For a small number of train and dev scenes (33 and 26) the capture date could not be resolved from the provider metadata.

<table><tr><td>Method</td><td>Setting</td><td>ρ(all)</td><td>ρ(paved)</td><td>ρ(unpaved)</td></tr><tr><td>FG2 (surround)</td><td>IMU</td><td>+0.23</td><td>+0.18</td><td>+0.20</td></tr><tr><td>Shi et al. (front)</td><td>IMU</td><td>+0.21</td><td>+0.19</td><td>+0.20</td></tr><tr><td>Ours (RGB+L)</td><td>IMU</td><td>+0.18</td><td>+0.26</td><td>+0.05</td></tr><tr><td>CCVPE (surround)</td><td>NA</td><td>+0.16</td><td>+0.17</td><td>+0.10</td></tr><tr><td>Ours (LiDAR-only)</td><td>IMU</td><td>+0.15</td><td>+0.23</td><td>+0.01</td></tr><tr><td>FG2 (surround)</td><td>NA</td><td>+0.13</td><td>+0.08</td><td>+0.10</td></tr><tr><td>Ours (RGB)</td><td>IMU</td><td>+0.10</td><td>+0.15</td><td>+0.02</td></tr><tr><td>Ours (RGB+L)</td><td>NA</td><td>+0.08</td><td>+0.12</td><td>-0.00</td></tr><tr><td>HC-Net (surround)</td><td>IMU</td><td>+0.07</td><td>+0.05</td><td>+0.05</td></tr><tr><td>Ours (LiDAR-only)</td><td>NA</td><td>+0.05</td><td>+0.10</td><td>-0.05</td></tr><tr><td>Ours (RGB)</td><td>NA</td><td>+0.05</td><td>+0.12</td><td>-0.04</td></tr><tr><td>HC-Net (surround)</td><td>NA</td><td>+0.04</td><td>+0.01</td><td>+0.03</td></tr></table>

Table A5. Rank correlation between per-scene aliasing and localization error (Spearman $\rho ,$ look-alike count at τ=0.8, averaged over training seeds). Every method correlates positively. The matching-based baselines correlate most and remain positive within both terrain groups. Our fusion is least affected in the priorfree setting.

<table><tr><td>Capture date</td><td>train</td><td>dev</td><td>test</td></tr><tr><td>2024-06-18</td><td>550</td><td>72</td><td>3</td></tr><tr><td>2024-07-31</td><td>1590</td><td>177</td><td>326</td></tr><tr><td>2024-08-24</td><td>3920</td><td>749</td><td>1517</td></tr><tr><td>2025-08-19</td><td>27</td><td>3</td><td>16</td></tr><tr><td>2025-08-26</td><td>310</td><td>37</td><td>0</td></tr></table>

Table A6. DOP20 orthophoto capture dates per split. All survey flights fall in the leaf-on season. The two most recent dates stem from the 2025 flight campaign of the same areas.

## J. Buffered test split

Scenes were sampled along trajectories, so some test tiles share overhead imagery with training tiles: 34% of test scenes lie within one tile side (204.8 m) of a training scene. The buffered test split removes this overlap. A test scene is retained only if its aerial tile provably shares no pixel with the tile of any training scene, using a conservative per-axis bound on the tile-center distance that accounts for the perscene offset between vehicle position and tile center. Thus, borderline pairs count as overlapping. This leaves 840 of 1862 scenes with a median distance of 481 m to the nearest training scene (full test: 252 m).

<table><tr><td></td><td></td><td colspan="2">Full test (n=1862)</td><td colspan="2">Buffered (n=840)</td></tr><tr><td>Setting</td><td>Method</td><td>Loc (m)</td><td>Ori (°)</td><td>Loc (m)</td><td>Ori(°)</td></tr><tr><td rowspan="7">North-up</td><td>CCVPE (surround)</td><td>45.4±2.9</td><td>35.5±17.0</td><td>46.3±2.8</td><td>39.9±13.0</td></tr><tr><td>HC-Net (surround)</td><td>31.8±0.4</td><td> $8 4 . 4 { \pm } 1 . 2 $ </td><td>31.8±0.2</td><td>83.6±2.1</td></tr><tr><td>FG2 (surround)</td><td>34.4±0.4</td><td> $9 3 . 1 \pm 0 . 3 $ </td><td> $3 4 . 2 { \pm } 0 . 2 $ </td><td>92.8±0.2</td></tr><tr><td>Ours (RGB)</td><td> $2 9 . 8 { \pm } 0 . 1$ </td><td> $2 7 . 9 { \pm } 3 . 3 $ </td><td>28.8±0.1</td><td>35.1±10.6</td></tr><tr><td>Ours (LiDÁR-only)</td><td> $2 8 . 7 { \pm } 0 . 3 $ </td><td> $3 8 . 5 { \pm } 5 . 4 $ </td><td> $2 9 . 2 { \pm } 0 . 4 $ </td><td>49.3±6.4</td></tr><tr><td>Ours (RGB+L)</td><td>27.1±0.3</td><td>24.8±0.3</td><td>26.2±0.7</td><td>29.6±1.8</td></tr><tr><td>IMU ±10° Shi et al. (front)</td><td>35.5±0.5</td><td>3.1±0.2</td><td>34.1±0.4</td><td>3.5±0.1</td></tr><tr><td rowspan="5"></td><td>HC-Net (surround)</td><td>31.9±3.3</td><td>11.7±0.8</td><td>33.8±6.6</td><td>12.5±0.6</td></tr><tr><td>FG2 (surround)</td><td>25.0±0.4</td><td>4.4±0.2</td><td>25.2±0.5</td><td>4.9±0.3</td></tr><tr><td>Ours (RGB)</td><td>25.1±0.1</td><td> $3 . 9 { \pm } 0 . 1$ </td><td> $2 4 . 5 { \pm } 0 . 2 $ </td><td>4.2±0.2</td></tr><tr><td>Ours (LiDAR-only)</td><td>21.2±0.2</td><td> $3 . 5 { \pm } 0 . 1$ </td><td> $2 0 . 9 2 0 . 1 $ </td><td> $3 . 8 { \pm } 0 . 1$ </td></tr><tr><td>Ours (RGB+L)</td><td>19.5±0.3</td><td>3.8±0.1</td><td>18.7±0.2</td><td>4.0±0.2</td></tr></table>

Table A7. Full vs. buffered test (Loc = mean position error, Ori = median orientation error, ± std over 3 seeds). No method gains from the removed overlap: every ranking persists on the leakagefree subset, the paired fusion advantage grows (cf. the main text), and orientation degrades most for the camera-free variant.

<table><tr><td>Pair</td><td>Min</td><td>5th pct.</td><td>Median</td><td>Tile overlap</td></tr><tr><td>test → train</td><td>1.5m</td><td>30m</td><td>252 m</td><td>40%</td></tr><tr><td>dev → train</td><td>0.4m</td><td>20 m</td><td>179 m</td><td>58%</td></tr><tr><td>test → dev</td><td>0.1 m</td><td>10m</td><td>141 m</td><td>66%</td></tr></table>

Table A8. Inter-split nearest-neighbor distances between scene positions, and the share of scenes whose 204.8 m aerial crop overlaps a crop of the other split.