# CedarCypress3D: an annotated UAV-LiDAR dataset of individual trees in planted cedar and cypress forests

Katsuto Shimizu\*<sup>1</sup>, Fumiaki Kitahara<sup>2</sup>, Tomohiro Nishizono<sup>2</sup>, Hideki Saito<sup>3</sup>, Masayoshi Takahashi<sup>2</sup>, Shingo Obata<sup>4</sup>, Shunsuke Tei<sup>4</sup>, Naoyuki Furuya<sup>2</sup>, Tomoya Goto<sup>5</sup>, Eiji Kodani<sup>2</sup>, and Yusuke Yamada<sup>6</sup>

<sup>1</sup>Shikoku Research Center, Forestry and Forest Products Research Institute,

<sup>2</sup>Department of Forest Management, Forestry and Forest Products Research Institute, 1 Matsunosato, Tsukuba, Ibaraki 305-8687, Japan

<sup>3</sup>Forestry and Forest Products Research Institute, 1 Matsunosato, Tsukuba, Ibaraki 305-8687, Japan <sup>4</sup>Hokkaido Research Center, Forestry and Forest Products Research Institute, 7 Hitsujigaoka Toyohira, Sapporo, Hokkaido 062-8516, Japan <sup>5</sup>Green Kogyo, Nibancho 5-5, Chiyoda-ku, Tokyo 102-0084, Japan <sup>6</sup>Graduate School of Bioagricultural Sciences, Nagoya University, Furo-cho, Chikusa-ku, Nagoya, 464-8601, Japan

## ABSTRACT

Individual tree measurements derived from Light Detection and Ranging (LiDAR) mounted on Unmanned Aerial Vehicles (UAV) provide valuable information for forest inventory, ecosystem monitoring, and sustainable forest management. Recent advancements in machine learning have increased the demand for annotated datasets to develop and evaluate point cloud-based approaches, especially for individual tree segmentation. However, publicly available annotated UAV-LiDAR datasets in temperate forests are limited. In this article, we present CedarCypress3D, a manually annotated UAV-LiDAR dataset collected in Japanese cedar (Cryptomeria japonica) and Japanese cypress (Chamaecyparis obtusa) plantations in Japan. The dataset consists of UAV-LiDAR point clouds and field survey measurements from 34 circular plots across two sites with different topographic characteristics, along with terrestrial LiDAR point clouds available for a subset of 22 plots. A total of 1,627 trees were measured in the census field survey and manually annotated to match the corresponding trees in the UAV-LiDAR point clouds. For the subset of plots with terrestrial LiDAR data, semantic labels (i.e., stem and non-stem) were additionally assigned to tree points in the UAV-LiDAR data. CedarCypress3D provides high-quality annotated UAV-LiDAR data for developing and evaluating individual tree instance segmentation and semantic segmentation methods in temperate planted forests. The dataset can also support research on tree attribute prediction and multi-platform LiDAR analysis. The dataset is publicly available at https://doi.org/10.5281/zenodo.22168721.

Keywords Japanese cedar · Japanese cypress · UAV · instance segmentation

## 1 Introduction

Quantifying individual tree attributes is fundamental to sustainable forest management as it supports carbon accounting and silvicultural decision-making. Traditionally, tree attributes have been collected through field surveys. Although field measurements provide accurate information, they are labor-intensive and time-consuming, leading to limited spatial coverage. Recent advances in Light Detection and Ranging (LiDAR) remote sensing have improved the acquisition of three-dimensional forest structural data (Balestra et al., 2024; Feigl et al., 2025). LiDAR point clouds capture detailed information on tree crowns, stems, and spatial arrangement, enabling a better understanding of forest structure (Liang et al., 2022). These three-dimensional data enable the development of algorithms for individual tree

detection, segmentation, and tree attribute estimation for forest inventories and ecological studies (White et al., 2025;   
Takeshige et al., 2025; Erfanifard et al., 2025).

The emergence of Unmanned Aerial Vehicle (UAV) LiDAR has expanded the applications of LiDAR in forestry. Compared to conventional airborne LiDAR campaigns, UAV-LiDAR can be more easily operated over relatively small forest stands with dense point clouds often exceeding 100 points/m<sup>2</sup> (Liang et al., 2022). The dense point clouds enable the prediction of detailed individual tree attributes, such as diameter at breast height (DBH), crown base height, and crown structural metrics (Htoo et al., 2025; Eisenschink et al., 2025; Yang et al., 2025; Fu et al., 2025). Recent advances in deep learning have further improved individual tree analysis of LiDAR point clouds, especially for instance segmentation and tree attribute prediction (Straker et al., 2023; Lu et al., 2025; Luo et al., 2026). These approaches typically require large and accurately annotated point cloud datasets for model training and evaluation. However, publicly available annotated datasets suitable for individual tree analysis remain limited, particularly for specific forest types and regions (Ardohain et al., 2026).

In recent years, the number of publicly available annotated UAV-LiDAR point cloud datasets in forestry has been increasing. The FOR-instance dataset (Puliti et al., 2023) is one of the most widely used benchmarks (Straker et al., 2023; Xiang et al., 2024, 2025; Lu et al., 2025). Other publicly available datasets, such as ForestScan (Chavana-Bryant et al., 2026), Lin3D (Lu et al., 2025), and that by Weiser et al. (2022), contain UAV-LiDAR data, in some cases together with terrestrial laser scanning (TLS) and mobile laser scanning (MLS) data collected in tropical and temperate forests (Weiser et al., 2022; Bai et al., 2023; Lu et al., 2025; Chavana-Bryant et al., 2026). However, the geographic and forest-type coverage of existing datasets remains limited, with relatively few datasets available for temperate planted forests. In Japan, planted forests predominantly consist of Japanese cedar (Cryptomeria japonica) and Japanese cypress (Chamaecyparis obtusa) on steep terrain (Takahashi and Tanaka, 2026). Although these species typically have straight trunks, planted forests are often characterized by high stand densities, closed canopies, and abundant understory vegetation. These structural characteristics increase crown overlap and occlusion, which makes individual tree segmentation more challenging than in many existing datasets. Consequently, models trained on datasets from other forest types or geographic regions may not perform well in Japanese planted forests. This highlights the need for a high-quality, annotated UAV-LiDAR dataset that specifically represents temperate planted forests.

In this article, we present CedarCypress3D, an annotated UAV-LiDAR point cloud dataset from temperate planted forests in Japan. The dataset contains UAV-LiDAR point clouds collected in Japanese cedar and Japanese cypress forests, together with census field survey plot data. Terrestrial LiDAR point clouds are also provided for selected survey plots. Both the UAV-LiDAR and terrestrial LiDAR point clouds were manually annotated at the individual-tree level (instance segmentation), and for the subset of plots with corresponding terrestrial LiDAR data, points within individual trees in the UAV-LiDAR point clouds were additionally assigned semantic labels (stem and non-stem). By providing ready-to-use annotated point clouds and corresponding field measurements, CedarCypress3D facilitates the development, benchmarking, and evaluation of algorithms for individual tree analysis, such as instance segmentation, semantic segmentation, tree detection, and tree attribute prediction.

## 2 Materials and Methods

## 2.1 Study area

CedarCypress3D consists of UAV-LiDAR, terrestrial LiDAR, and field survey data acquired at two sites in Oita prefecture, western Japan (Figure 1). A total of 34 circular survey plots were established across the two sites. The first site is located in planted forests in the Saiki region (32.81<sup>◦</sup>N, 131.66<sup>◦</sup>E), and the second site is located in the Kokonoe region (33.24<sup>◦</sup>N, 131.20<sup>◦</sup>E). At both sites, the dominant planted conifer species are Japanese cedar (Cryptomeria japonica) and Japanese cypress (Chamaecyparis obtusa). Stand ages ranged from 25 to 85 years at the Saiki and from 12 to 63 years at the Kokonoe site (Forestry Agency, 2026). The Saiki site is characterized by steep terrain, with a mean topographic slope of $2 8 . 8 ^ { \circ } ( 1 2 . 0 { - } 3 7 . \dot { 6 } ^ { \circ } )$ , averaged across the survey plots and a mean elevation of approximately 270 m above sea level. In contrast, the Kokonoe site has relatively gentle terrain, with a mean topographic slope of 13.0<sup>◦</sup> (6.5–24.1<sup>◦</sup>) and a mean elevation of approximately 800 m. The mean annual temperature is $1 6 . 7 ~ ^ { \circ } \mathrm { C }$ at both sites.

![](images/597a27185bf3604e5b3ea31b68ecdace6990f8fa0a698aa4cafb77612d567fd6.jpg)  
Figure 1: Locations of the survey plots with a radius of 12 m. Latitude and longitude labels have been removed from the main maps. The prefectural boundary data from the National Land Numerical Information were used.

## 2.2 Field survey

A field survey was conducted at the Saiki site in January and April 2024. Circular plots with a radius of 12 m (0.04524 ha) were established. All planted trees within each plot were measured regardless of DBH, and other trees with DBH ≥ 10 cm were also measured. Of these survey plots, 16 were located in planted forests and included in the dataset. DBH was measured 1.2 m above the ground using a diameter tape. Tree and canopy base heights were measured using a Vertex 5 hypsometer (Haglöf, Sweden). In this study, canopy base height was defined as the vertical distance from the ground to the base of the lowest continuous live branch. Therefore, canopy base height was not recorded for dead trees. Because tree height measurement for one tree was omitted during the field survey, it was imputed using the DBH–height relationship of the measured trees within the same plot based on the Näslund function. Tree form classes were assigned to each tree in situ following the Terazaki’s tree form classification (Table 1) (Terazaki, 1951). Tree locations relative to the plot center were measured using a TruPulse 360 laser rangefinder (Laser Technology Inc, USA). The coordinates of each plot center were recorded using a Geode GNS2 Global Navigation Satellite System (GNSS) receiver (Juniper Systems, USA). Individual tree stem volumes were calculated from tree height, DBH, and species, based on stem volume equations (Japan Forestry Agency, 1970a,b; Hosoda et al., 2010) using the stemv package in R (Shimizu, 2024). The mean stand volume in Saiki was 952.4 m<sup>3</sup>/ha in the Japanese cedar plots and 547.4 m<sup>3</sup>/ha in the Japanese cypress plots. Aboveground biomass (AGB) was also calculated from stem volume using published biomass conversion equations (Greenhouse Gas Inventory Office of Japan and Ministry of the Environment, Japan, 2021; Shimizu et al., 2026).

A field survey was conducted at the Kokonoe site in April 2025. A total of 18 circular plots were established and measured. The same protocol used at the Saiki site was applied to collect individual tree attributes (DBH, tree height, canopy base height, tree form class, and tree species), tree locations, and plot center coordinates. Geode GNS2 and GNS3 GNSS receivers were used to determine the plot center coordinates at the Kokonoe site. The mean stand volume was 853.4 m<sup>3</sup>/ha in the Japanese cedar plots and 539.7 m<sup>3</sup>/ha in the Japanese cypress plots. Across the Saiki and Kokonoe sites, a total of 1,627 trees were measured in 34 plots. Although these plots were dominated by either Japanese cedar or Japanese cypress, a small number of other tree species were also observed. The dataset included 571 Japanese cedar trees, 1,051 Japanese cypress trees, and 5 trees of other species. Summary statistics of the forest structural attributes are presented in Table 2.

Table 1. Description of Terazaki’s tree form classification used in the field survey.
<table><tr><td>Tree form class</td><td>Group</td><td>Description</td></tr><tr><td>1</td><td>Dominant</td><td>Tree occupying sufficient growing space for crown development, with crown growth not suppressed by neighboring trees and showing no defects in stem/crown form.</td></tr><tr><td>2a</td><td>Dominant</td><td>Tree with an overgrown crown that is horizontally spreading or markedly flat in the upper part of the crown.</td></tr><tr><td>2b 2c</td><td>Dominant</td><td>Tree with a poorly developed crown and a smaller stem diameter.</td></tr><tr><td></td><td>Dominant</td><td>Tree located between neighboring trees, with insufficient space for crown development, resulting in a one-sided crown.</td></tr><tr><td>2d</td><td>Dominant</td><td>Tree with stem defects, such as a forked or swept stem.</td></tr><tr><td>2e 3</td><td>Dominant</td><td>Tree with a damaged stem or crown.</td></tr><tr><td></td><td>Suppressed</td><td>Tree with delayed growth, but whose crown is not yet totally suppressed and retains the potential for future height growth.</td></tr><tr><td>4</td><td>Suppressed</td><td>Tree whose crown is already suppressed, but remains alive.</td></tr><tr><td>5</td><td>Suppressed</td><td>Dead tree.</td></tr></table>

Table 2. Summary statistics of field survey measurements for main tree species at Saiki and Kokonoe sites. Values in parentheses indicate the minimum and maximum values. DBH, diameter at breast height; TH, tree height; BA, basal area. Main species indicates the dominant species in each plot; the stand statistics therefore refer to all trees within those plots, not only to individuals of the dominant species.
<table><tr><td>Site</td><td>Main species</td><td>n plots</td><td>Ave. trees /ha</td><td>Ave. DBH (cm)</td><td>Ave. TH (m)</td><td>Ave. stand volume (m³/ha)</td><td>Ave. BA (m²/ha)</td></tr><tr><td>Saiki</td><td>Japanese</td><td>11</td><td>940</td><td>35.1</td><td>24.0</td><td>952.4</td><td>86.5</td></tr><tr><td rowspan="6">Kokonoe</td><td>cedar</td><td></td><td>(420-1592)</td><td>(12.8-76.6)</td><td>(12-41.1)</td><td>(396-2012.9)</td><td>(47.4-156.2)</td></tr><tr><td>Japanese</td><td>5</td><td>1561</td><td>22.6</td><td>17.5</td><td>547.4</td><td>59.7</td></tr><tr><td>cypress</td><td></td><td>(752-2454)</td><td>(10-43)</td><td>(5.2-25.6)</td><td>(235.8-715.3)</td><td>(37.1-81.3)</td></tr><tr><td>Japanese</td><td>2</td><td>1304</td><td>28.9</td><td>20.9</td><td>853.4</td><td>86.1</td></tr><tr><td>cedar</td><td></td><td>(1105-1503)</td><td>(11-42.3)</td><td>(13.3-26.5)</td><td>(631.5-1075.4)</td><td>(77.3-95.0)</td></tr><tr><td>Japanese</td><td>16</td><td>951</td><td>29.4</td><td>17.3</td><td>539.7</td><td>63.6</td></tr><tr><td></td><td>cypress</td><td></td><td>(619-1614)</td><td>(9.2-74.2)</td><td>(6.8-25.9)</td><td>(293.3-896.4)</td><td>(45.5-88.2)</td></tr></table>

## 2.3 UAV-LiDAR measurements

Different UAV-LiDAR systems were used at Saiki and Kokonoe. At the Saiki site, a RIEGL VUX-1LR sensor mounted on a FAZER R G2 UAV was used to acquire the LiDAR point cloud data (Table 3). The flight campaign was conducted between April 12 and 18, 2022, at a flight altitude of approximately 80–130 m above ground level. The point cloud data had an average point density of 5,802 points/m<sup>2</sup> across all 16 plots at the site (Table 4). At the Kokonoe site, a DJI Zenmuse L2 sensor mounted on a Matrice 300 RTK UAV was used to acquire LiDAR point cloud data between November 18 and 20, 2024. The flight altitude was approximately 100 m above ground level, with an average point density of 676 points/m<sup>2</sup>. The UAV-LiDAR and field measurements were separated by two growing seasons at the Saiki site, whereas they were conducted within the same growing season at the Kokonoe site.

Both sets of UAV-LiDAR data were processed using the lidR package (Roussel et al., 2020). First, an isolated voxel filter was applied to remove outlier points. For the Saiki data, the progressive morphological filter algorithm was used to classify ground points because no ground classification was provided by the data vendor. In contrast, the existing ground classification provided by the data vendor was used for the Kokonoe data.

Table 3. Specifications of the LiDAR sensors used in the dataset. FWHM, Full Width at Half Maximum.
<table><tr><td>Specification</td><td>VUX-1LR</td><td>Zenmuse L2</td><td>UTM-30LX-EW</td></tr><tr><td>Platform</td><td>FAZER R G2</td><td>Matrice 300 RTK</td><td>OWL</td></tr><tr><td>Type</td><td>UAV</td><td>UAV</td><td>Terrestrial</td></tr><tr><td>Max. pulse repetition rate</td><td>1,500 kHz</td><td>240 kHz</td><td>43.2 kHz</td></tr><tr><td>Max. measuring range</td><td>1350 m</td><td>450 m</td><td>30 m</td></tr><tr><td rowspan="3">Ranging accuracy Beam divergence</td><td>(@60% reflectivity)</td><td>(@50% reflectivity)</td><td>(@90% reflectivity)</td></tr><tr><td>±15 mm</td><td>±20 mm (@ 150 m)</td><td>±50 mm (@30 m)</td></tr><tr><td>0.5 mrad</td><td>0.2 × 0.6 mrad</td><td></td></tr><tr><td>Angular resolution</td><td>(1/e2)</td><td>(FWHM)</td><td>4.4 mrad (0.25°)</td></tr></table>

Table 4. Summary of plot measurements in the CedarCypress3D dataset. Main species indicates the dominant species in each plot; the number of trees therefore refers to all trees within those plots, not only to individuals of the dominant species.
<table><tr><td></td><td>Main species</td><td>n plots</td><td>n terrestrial LiDAR</td><td>n trees</td><td>Sensor</td><td>Point density (pts/m2)</td><td>Average slope (°)</td></tr><tr><td>Site Saiki</td><td>Japanese cedar</td><td>11</td><td>7</td><td>468</td><td>RIEGL</td><td>5754</td><td>28.4</td></tr><tr><td rowspan="4">Kokonoe</td><td>Japanese cypress</td><td>5</td><td>0</td><td>353</td><td>VUX-1LR RIEGL</td><td>5909</td><td>29.7</td></tr><tr><td>Japanese cedar</td><td>2</td><td>1</td><td>118</td><td>VUX-1LR DJI</td><td>694</td><td>11.0</td></tr><tr><td>Japanese cypress</td><td>16</td><td>14</td><td>688</td><td>Zenmuse L2 DJI</td><td>674</td><td>13.3</td></tr><tr><td></td><td></td><td></td><td></td><td>Zenmuse L2</td><td></td><td></td></tr></table>

## 2.4 Terrestrial LiDAR measurements

For 22 of the 34 plots (7 in Saiki and 15 in Kokonoe), terrestrial LiDAR data was also collected using the Optical Woods Ledger (OWL) AME-OL104 (AdIn Research, Japan) during the field survey (Table 4). The OWL system is equipped with a UTM-30LX-EW laser sensor (Hokuyo, Japan) on a single-legged instrument. This system employs simultaneous localization and mapping (SLAM) and does not require remaining stationary during scanning (Tsubouchi, 2019). In each plot, nine scans were conducted at a distance of approximately 10 m. The multi-scan data acquired with OWL were processed using the OWL manager software (AdIn Research, Japan) to generate point clouds and tree location information. Ground points were classified using the progressive morphological filter algorithm in the lidR package (Roussel et al., 2020). Due to the laser sensor specification, canopy occlusion has been reported in previous studies, especially for taller trees (Kitahara et al., 2020; Nishizono et al., 2020; Shimizu et al., 2022). However, the point cloud data from this terrestrial LiDAR system were included to complement the UAV-LiDAR data.

## 2.5 Co-registration of field survey, UAV-LiDAR, and terrestrial LiDAR data

The relative tree locations recorded during the field survey were converted to geographic coordinates using the plot center coordinates from GNSS measurements. Because these tree locations had positional errors associated with GNSS measurements, they were spatially aligned with the UAV-LiDAR data prior to manual annotation using treetop locations provided by the data vendors. These treetop locations were used only for the co-registration process and were not used for subsequent analysis. To align the tree locations in the field surveys with the UAV-LiDAR-derived treetops, a hierarchical 2D rigid registration procedure was applied. First, rotation-invariant geometric features were computed from the sorted distances to the eight nearest neighbors for each tree, and candidate matches were established by selecting the five nearest neighbors in the feature space. Second, a coarse global alignment was performed using the Random Sample Consensus (RANSAC) with 8,000 iterations to estimate tentative rotation and translation parameters. Finally, the alignment was fine-tuned using a trimmed Iterative Closest Point (ICP) algorithm, which iteratively optimized the transformation parameters using only the closest 70% of the pairs to ensure robustness against outliers. After the final transformation parameters were determined, the UAV-LiDAR point clouds were clipped to a radius of 18 m (12 m plot radius + 6 m buffer) centered on each co-registered plot location (Figures 2 and 3).

![](images/25ecddc5838cf02be234af02609a8cbaa0bcc2a10192b2e258d7bd2ccbe13a61.jpg)  
Figure 2: Visualization of the clipped UAV-LiDAR point clouds at the Saiki site.

![](images/ffb70a4d6094b6dda6225719d1c7c75b4d1a11478270cd64e881c64799fdc87b.jpg)  
Figure 3: Visualization of the clipped UAV-LiDAR point clouds at the Kokonoe site.

For the OWL terrestrial LiDAR data, a similar co-registration procedure was applied to align the terrestrial LiDAR point clouds with the UAV-LiDAR data. The tree locations derived from terrestrial LiDAR were used as inputs for the same initial coarse registration procedure described above. The estimated transformation parameters were then applied to the original terrestrial LiDAR point clouds. Finally, a fine registration between the terrestrial LiDAR and UAV-LiDAR point clouds was performed using the ICP algorithm. For three plots at the Kokonoe site, manual coregistration was performed instead of ICP, because the ICP failed to align the two point clouds owing to insufficient overlap.

## 2.6 Manual annotation

The UAV-LiDAR point clouds were manually annotated into individual trees using the segment tool in CloudCompare v2.13.2 (Figure 4). Two annotators performed the initial annotation, which was subsequently reviewed and revised by a third annotator to determine the final labels. All trees within each plot were annotated. To facilitate the annotation process for the Kokonoe data, marker-controlled watershed segmentation was applied to digital canopy height models (DCHM) alongside tree locations recorded during the field survey. This procedure was not applied to the Saiki data. Noise points were also identified during annotation.

After annotation of the UAV-LiDAR point clouds, tree IDs from the field survey were assigned to the annotated trees based on the co-registered tree locations (Figure 5). Each annotated tree was linked to the nearest field-survey tree using the centroid of the annotated point cloud. Because field survey tree locations were co-registered to treetops derived from the UAV-LiDAR data, some trees located near the plot boundary may extend beyond the 12 m plot radius. This discrepancy reflects the positional uncertainty after co-registration and does not indicate that these trees are located outside the plot.

![](images/d9eb5bae05cbd24ededa1000112596ea581c6406567271cf16c6d1f0c595fccf.jpg)  
Figure 4: Visualization of individual tree annotations for plots in (A) Saiki (Plot 1) and (B) Kokonoe (Plot 3). Trees and understory vegetation outside the plot boundary are not shown.

For the OWL terrestrial LiDAR data, manually annotated UAV-LiDAR data were used to assign initial tree IDs. A k-nearest neighbor (kNN) approach was applied to transfer tree IDs from UAV-LiDAR point clouds to the terrestrial Li-DAR point clouds to reduce the time and effort required for manual annotation. For the Saiki data, nearest neighbors were identified using three-dimensional (XYZ) information, whereas only two-dimensional (XY) information was used for the Kokonoe data because the lower UAV-LiDAR point density reduced the reliability of three-dimensional allocation. The automatically assigned tree IDs were then manually reviewed and corrected by a single annotator. While both stem and crown (i.e., branches and leaves) points were corrected for the Saiki data, only minimal correction was applied to crown points for the Kokonoe data, because overlaps between adjacent trees made it difficult to determine which points belonged to the corresponding trees.

For the subset of plots with corresponding terrestrial LiDAR data (n = 22), semantic labels (stem and non-stem) were additionally assigned to points within individual trees in the UAV-LiDAR point clouds. To do this, a single annotator manually classified points as stem or non-stem (i.e., crown) within previously segmented individual trees using CloudCompare. A total of 244 trees from 7 plots at the Saiki site and 624 trees from 15 plots at the Kokonoe site were assigned semantic labels. However, trees at the Kokonoe site frequently lacked sufficient stem points because of lower point density of the UAV-LiDAR data.

A) Field survey  
![](images/6f6a00a6003287f7cee0de4404e6359e877dca1d0613bafc4474e85ae5a581a8.jpg)

B) UAV-LiDAR data  
![](images/d7c14ba73569868d9b6e7e7737f81af9a2fe8f222cf59e194ec26a17ff1eba9e.jpg)

C) Manual annotation  
![](images/dc40ee5a5b2b6e4fe2e752077de3e9fd50a4a186ca76f689dc2f14df4605e8ca.jpg)

D) Tree ID matching  
![](images/ba55f8b720cc0ec5454e7560fef4c18b65883bdfe732188dc8afa451dbd8b7ce.jpg)  
Figure 5: Visualization of tree ID allocation using tree locations recorded during the field survey and annotated UAV-LiDAR point clouds. Examples from Plot 10 at the Saiki site are presented. (A) Photograph of the field survey plot. (B) UAV-LiDAR point cloud colored by normalized height with dashed lines indicating the transect strip shown in (C). (C) Cross-sectional view of the manually annotated point cloud along the transect in (B). (D) Tree ID allocation based on nearest neighbor matching between field-survey tree locations (black) and UAV-LiDAR annotated tree centroids (red).

## 3 Dataset description

The UAV-LiDAR and terrestrial LiDAR point clouds are provided in LAS format. The terrestrial LiDAR data are available for a subset of the plots. The treeID attribute represents the annotated tree ID, which corresponds to the tree ID in the field survey data. Non-tree points and trees located outside the plot are assigned treeID = 0 (Figure 6). To anonymize the study sites, the point cloud coordinates were translated into a local coordinate system and the original geographic coordinates were removed.

The LAS point classification defined in this study is presented in Table 5. Points outside the plot boundary are assigned to the Outside plot class (Classification = 1). This class includes all points within the buffer around each plot and tree points belonging to trees rooted outside the plot, the crowns of which extended into the plot (Figure 6). Ground points (Classification = 2) were primarily identified using automatic ground classification, with manual corrections applied during annotation. Low vegetation (Classification = 3) includes vegetation within the plot that is neither ground nor tree. Manually annotated individual trees within the plot are assigned Classification = 5. Noise points are assigned Classification = 7.

In addition to the Classification attribute, the treeComponent attribute is stored in all UAV-LiDAR LAS data to distinguish stem from non-stem components of individual trees (Figure 6). The attribute is defined as unlabeled (= 0), stem (= 1), or non-stem (= 2) (Table 6). Stem and non-stem labels are available only for tree points (i.e., Classification = 5) in the UAV-LiDAR data from plots with corresponding terrestrial LiDAR measurements. For all other points, including tree points in plots without terrestrial LiDAR measurements, treeComponent is set to 0.

Field survey data are provided in CSV format. The treeID field corresponds to the treeID attribute in the UAV-LiDAR point cloud. Tree attributes measured during the field survey are provided for each tree, including plot ID, tree ID, species, tree form class, DBH, tree height, canopy base height, and stem volume. Plot-level attributes (i.e., survey date and stand age) are also provided.

Table 5. LiDAR point classification used in the dataset.

<table><tr><td>Classification</td><td>Description</td></tr><tr><td>1</td><td>Outside plot</td></tr><tr><td>2</td><td>Ground</td></tr><tr><td>3</td><td>Low vegetation within the plot</td></tr><tr><td>5</td><td>Trees within the plot</td></tr><tr><td>7</td><td>Noise</td></tr></table>

Table 6. Description of the treeComponent attribute in the UAV-LiDAR data. Stem and non-stem classification is available only for plots with corresponding terrestrial LiDAR data.
<table><tr><td>Value</td><td>Description</td></tr><tr><td>0</td><td>Unlabeled / no data (points with Classification ≠ 5, or tree points in plots without terrestrial LiDAR data)</td></tr><tr><td>1</td><td>Stem</td></tr><tr><td>2</td><td>Non-stem (branches and leaves)</td></tr></table>

![](images/35849e912d635bdabcb507be0c03a4e12ed4f4c1f9257ef6642dda18db3f0030.jpg)  
Figure 6: Example of point attributes for treeID, Classification, and treeComponent. An example cross-section from Plot 5 at the Saiki site is presented.

## 4 Dataset validation

## 4.1 Validation

To validate the manual annotation and tree correspondence between the LiDAR data and the field survey, tree heights derived from the LiDAR point clouds were compared with those from the field survey (Figure 7). Tree height from the LiDAR data was calculated as the height difference between the highest point of each annotated tree and the ground. For the UAV-LiDAR data, the root mean squared error (RMSE) of tree height was 1.24 m for Japanese cedar and 0.96 m for Japanese cypress after excluding dead trees. Furthermore, all field-surveyed trees were successfully matched to the annotated point clouds with a mean distance of 0.51 m between field survey tree locations and the centroid of the corresponding annotated point clouds. These results indicate good agreement between field survey data and annotated UAV-LiDAR data, supporting the accuracy of both manual annotation and tree matching. Although there were two growing seasons between the UAV-LiDAR acquisition and field survey measurements at the Saiki site, the resulting bias remained small. For the terrestrial LiDAR data, a larger bias (-2.94 m) and RMSE (3.85 m) were observed, which were expected because of canopy occlusion and the LiDAR sensor used in this study.

![](images/14ddc5e09d63edddb2da1c4ff5a7f9a1d8b7c44966b31c3abae7f99d6185803f.jpg)  
Figure 7: Comparison of tree heights measured in the field survey with those derived from the UAV-LiDAR and terrestrial LiDAR data (upper panels). Examples of annotated tree point clouds are shown in the lower panels.

## 4.2 Uncertainty

Despite careful manual annotation, some uncertainty remains as branches or leaves of adjacent trees often overlap, making it difficult to separate individual trees. This issue occurs more frequently in Japanese cypress forests. In addition, suppressed trees are inherently more difficult to annotate because of partial or complete occlusion by dominant trees. For semantic labeling (stem and non-stem), exact separation of stem points from branches and leaves was challenging; thus, this classification may contain uncertainties. These uncertainties in the manual annotation process should be considered when conducting analyses.

In plot 13 at the Saiki site, several trees within the plot boundary were identified in the UAV-LiDAR point clouds; however, corresponding records in the field survey data were lacking. These discrepancies were likely caused by omissions during the field survey. To address this issue, new tree IDs were assigned to these trees; consequently, some treeID in the point cloud data had no corresponding trees in the field survey data. For completeness, tree height and DBH for these trees were manually measured from the UAV-LiDAR point cloud and are provided in a separate file rather than within the field survey datasheet.

Several tree attributes in the field survey data involve inherent uncertainties. Although stem volume is provided for each tree, it was calculated using species specific stem volume equations. Consequently, stem volumes of individual trees may contain uncertainties. Previous studies have reported RMSEs of 2.5–13.5% between harvested and calculated stem volumes for Japanese cedar and Japanese cypress trees (Inoue and Kurokawa, 2001; Mitsuda et al., 2012). Additionally, stand age was obtained from forest management records provided by the Forestry Agency (Forestry Agency, 2026). These records may not accurately represent the actual stand conditions because of potential mismatches between historical management records and the current status of the stands.

The terrestrial LiDAR system used in this study is less suitable for measuring tree height, because of the characteristics of the laser sensor. This limitation is evident from the visual comparison of the point clouds (Figure 8) and the validation results presented above. Tree height measurements derived from these data should be interpreted with caution and are not recommended as a reference source. In contrast, terrestrial LiDAR data can generally provide accurate DBH measurements, and previous studies that used this terrestrial LiDAR sensor in Japanese planted forests have confirmed this, reporting RMSEs of 1.1–3.2 cm (Muroki and I, 2019; Matsumura et al., 2020; Suematsu et al., 2020; Shimizu et al., 2022). However, noise points were observed in some plots, and users should be aware of their presence and potential influence when analyzing point cloud data.

![](images/99fe50aea49ae4009683b4b28a9b60d0138729757cf891570b0d5b395892c36c.jpg)

![](images/6d7c2bb4a4182bc304bb9ae172c77b32f2d4967041b259c3186f68d3945caccf.jpg)

A) Saiki  
![](images/79d00b33adbccfddcbf3e8f2a5b2334266fb2e00aa4706f5ce5d502f24a309c9.jpg)

B) Kokonoe  
![](images/474ab89f52b36eae4336e9bdc98de528f8283dc70438605a03c6b1b4db3bf5f3.jpg)  
X (m)

![](images/38d7e4cbae2b16d9b19f4b43ab62dda92f1d0839aea8fc4be132884fb9294724.jpg)  
Figure 8: Example of normalized UAV-LiDAR and terrestrial LiDAR point clouds for (A) a Japanese cedar plot (Plot 1) in Saiki and (B) a Japanese cypress plot (Plot 1) in Kokonoe, shown in the left panels. The right panels show histograms of point clouds for each height bin after removing ground points.

## 5 Dataset usage

CedarCypress3D is designed to provide ready-to-use, manually annotated UAV-LiDAR data for individual tree analysis, such as the development and validation of point cloud-based instance segmentation methods. For a subset of UAV-LiDAR data (22 out of 34 plots), semantic labels (stem and non-stem) are also available for individual trees; thus, it is suitable for semantic segmentation. However, this dataset can also be used for a wide range of other pur poses. The field survey data provide tree-level reference measurements that could be used to develop and validate models for predicting tree attributes including DBH, tree height, and stem volume. In particular, the high point density of the UAV-LiDAR data at the Saiki site enables the direct estimation of stem diameter from the point clouds, offering opportunities to develop DBH estimation methods. Furthermore, the paired UAV-LiDAR, terrestrial LiDAR, and field survey data also facilitate comparative studies of different sensing platforms for individual tree analysis.

## Acknowledgments

This study was supported by the Japan Forest Foundation through a Grant-in-Aid for Forestry Promotion and JSPS KAKENHI Grant Number JP24K01804. The authors thank the Forest Research and Management Organization for

providing access to the UAV-LiDAR data. We are also grateful to Natsu Sannomiya for her assistance with the manual annotation of the LiDAR point cloud data. We thank Dr. Kazuo Hosoda and Dr. Gen Takao for their assistance with field data collection.

## Data availability

The dataset is available at https://doi.org/10.5281/zenodo.22168721. The exact plot locations are available from the corresponding author upon reasonable request.

## Declaration of interest

The authors report there are no competing interests to declare.

## Declaration of generative AI use

During the preparation of this work, the authors used ChatGPT and Claude to check grammatical errors and improve the language of the manuscript. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the article.

## Author contribution

Katsuto Shimizu: Writing – original draft, Writing – review & editing, Conceptualization, Data curation, Formal analysis, Methodology, Investigation, Resources, Validation. Fumiaki Kitahara: Writing – review & editing, Data curation, Formal analysis, Investigation. Tomohiro Nishizono: Methodology, Writing – review & editing, Data curation, Formal analysis, Investigation, Funding acquisition. Hideki Saito: Writing – review & editing, Investigation, Funding acquisition. Masayoshi Takahashi: Writing – review & editing, Investigation. Shingo Obata: Writing – review & editing, Investigation. Shunsuke Tei: Writing – review & editing, Investigation. Naoyuki Furuya: Writing – review & editing, Investigation, Data curation. Tomoya Goto: Writing – review & editing, Investigation. Eiji Kodani: Writing – review & editing, Investigation. Yusuke Yamada: Writing – review & editing, Investigation.

## References

Ardohain, C. M., Choi, D. H., Grong, K. A., Huang, Y., Lyon, N. S., Park, S., Shao, J., Thapa, B., Willsey, S. K., Wingren, C. P., Wang, J., Jo, I., and Fei, S. (2026). Deep Learning Applications in Remote Sensing for Forest Inventory Methods. Remote Sensing, 18(15):2490.

Bai, Y., Durand, J.-B., Vincent, G., and Forbes, F. (2023). Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination. Advances in Neural Information Processing Systems, 36:48293–48313.

Balestra, M., Marselis, S., Sankey, T. T., Cabo, C., Liang, X., Mokroš, M., Peng, X., Singh, A., Sterenczak, K., Vega,´ C., Vincent, G., and Hollaus, M. (2024). LiDAR Data Fusion to Improve Forest Attribute Estimates: A Review. Current Forestry Reports, 10:281–297.

Chavana-Bryant, C., Wilkes, P., Yang, W., Burt, A., Vines, P., Bennett, A. C., Pickavance, G. C., Cooper, D. L. M., Lewis, S. L., Phillips, O. L., Brede, B., Lau, A., Herold, M., McNicol, I. M., Mitchard, E. T. A., Coomes, D. A., Jackson, T. D., Makaga, L., Milamizokou Napo, H. O., Ngomanda, A., Ntie, S., Medjibe, V., Dimbonda, P., Soenens, L., Daelemans, V., Proux, L., Nilus, R., Labrière, N., Jeffery, K., Burslem, D. F. R. P., Clewley, D., Moffat, D., Qie, L., Bartholomeus, H., Vincent, G., Barbier, N., Derroire, G., Abernethy, K., Scipal, K., and Disney, M. (2026). ForestScan: A unique multiscale dataset of tropical forest structure across 3 continents including terrestrial, UAV and airborne LiDAR and in-situ forest census data. Earth System Science Data, 18(2):1243–1274.

Eisenschink, P. M., Obermeier, W. A., Zerres, V. H. D., Suerbaum, A. M., and Lehnert, L. W. (2025). Forest vari ables from LiDAR: Drone flight parameters impact the detection of tree stems and diameter estimates. Ecological Informatics, 88:103127.

Erfanifard, Y., Garbarino, M., and Sterenczak, K. (2025). Contribution of high-resolution remote sensing to spatial´ ecology of forest ecosystems at the single tree level: A systematic review. Remote Sensing Applications: Society and Environment, 40:101733.

Feigl, J., Frey, J., Seifert, T., and Koch, B. (2025). Close-Range Remote Sensing of Forest Structure for Biodiversity Assessments: A Systematic Literature Review. Current Forestry Reports, 11(1):18.

Forestry Agency (2026). National forest resource mesh. https://www.geospatial.jp/ckan/dataset/mesh\_tile.

Fu, Y., Li, W., Duan, W., Bai, J., Wang, L., and Niu, Z. (2025). Comparison of UAV-LiDAR-driven biomass estimation approaches in planted forests with different management. International Journal of Digital Earth, 18(2):2576910.

Greenhouse Gas Inventory Office of Japan and Ministry of the Environment, Japan (2021). National Greenhouse Gas Inventory Report of JAPAN, 2021. Technical Report CGER-I154-2021.

Hosoda, K., Mitsuda, Y., and Iehara, T. (2010). Differences between the present stem volume tables and the values of the volume equations, and their correction. Japanese Journal of Forest Planning, 44(2):23–39.

Htoo, K. K., Onishi, M., Rahman, M. F., Takeshige, R., Kitajima, K., and Onoda, Y. (2025). Development of crown-based allometric equations for estimating stem diameter and above-ground biomass using UAV-LiDAR in 23 species-rich natural forests of Japan. Journal ofForest Research, 30(6):491–501.

Inoue, A. and Kurokawa, Y. (2001). Theoretical Derivation of a Two-way Volume Equation in Coniferous Species. Journal ofthe Japanese Forestry Society, 83(2):130–134.

Japan Forestry Agency (1970a). Tree Volume Table for Eastern Japan. Japan Forestry Investigation Committie, Tokyo, Japan.

Japan Forestry Agency (1970b). Tree Volume Table for Western Japan. Japan Forestry Investigation Committie, Tokyo, Japan.

Kitahara, F., Nishizono, T., Hosoda, K., and Kodani, E. (2020). Comparison of forest measurement errors using two types of terrestrial laser scanning. Japanese Journal ofForest Planning, 54(1):63–66.

Liang, X., Kukko, A., Balenovic, I., Ninni, S., Junttila, S., Kankare, V., Holopainen, M., Martin, M., Surovy, P., Kaartinen, H., Luka, J., Honkavaara, E., Nasi, R., Jingbin, L., Hollaus, M., Tian, J., Yu, X., Jie, P., Shangshu, C., Virtanen, J. P., Wang, Y., and Hyyppa, J. (2022). Close-Range Remote Sensing of Forests: The State of the Art, Challenges, and Opportunities for Systems and Data Acquisitions. IEEE Geoscience and Remote Sensing Magazine, 10(3):32–71.

Lu, H., Li, B., Yang, G., Fan, G., Wang, H., Pang, Y., Wang, Z., Lian, Y., Xu, H., and Huang, H. (2025). Towards a point cloud understanding framework for forest scene semantic segmentation across forest types and sensor platforms. Remote Sensing ofEnvironment, 318:114591.

Luo, B., Yang, J., Shi, S., Gan, R., Wu, Z., Wang, S., Wang, A., Du, L., and Gong, W. (2026). InceptionFormer: A deep learning framework for UAV LiDAR point cloud completion to improve tree parameters estimation in dense forests. Remote Sensing ofEnvironment, 338:115348.

Matsumura, N., Arita, T., Hirose, Y., Numamoto, S., Shimada, H., and Nomura, H. (2020). Accuracy validation of various measurement instruments for acquisition of high precision forest resource information. Japanese Journal ofForest Planning, 54(1):55–61.

Mitsuda, Y., Inoue, A., Kitahara, F., Kadota, H., and Hirota, T. (2012). Examination of factors affecting the error of aboveground biomass estimation in overcrowded planted stands : A case study using sample felled trees from a 40-year-old Hinoki (Chamaecyparis obtusa) planted stand. Japanese Journal ofForest Planning, 46(1):15–24.

Muroki, N. and I, T. (2019). Estimating a stand volume using UAV-aerial images and terrestrial laser scanning. Japanese Journal ofForest Planning, 52(2):83–88.

Nishizono, T., Hosoda, K., Fukumoto, K., Yamada, Y., Takahashi, M., Saito, H., Kitahara, F., and Kodani, E. (2020). Effects of stand condition and history on measurement errors for tree size using terrestrial laser scanning in Chamaecyparis obtusa man-made forests. Japanese Journal ofForest Planning, 54(1):37–44.

Puliti, S., Pearse, G., Surový, P., Wallace, L., Hollaus, M., Wielgosz, M., and Astrup, R. (2023). FOR-instance: A UAV laser scanning benchmark dataset for semantic and instance segmentation of individual trees. arXiv:2309.01279.

Roussel, J.-R., Auty, D., Coops, N. C., Tompalski, P., Goodbody, T. R., Meador, A. S., Bourdon, J.-F., de Boissieu, F., and Achim, A. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. Remote Sensing ofEnvironment, 251:112061.

Shimizu, K. (2024). Stemv: An R package for calculating tree stem volume in Japan. Japanese Journal of Forest Planning, 58(2):55–60.

Shimizu, K., Nishizono, T., Kitahara, F., Fukumoto, K., and Saito, H. (2022). Integrating terrestrial laser scanning and unmanned aerial vehicle photogrammetry to estimate individual tree attributes in managed coniferous forests in Japan. International Journal ofApplied Earth Observation and Geoinformation, 106:102658.

Shimizu, K., Saito, H., Nishizono, T., Yamada, Y., and Kitahara, F. (2026). Mapping three decades of forest structural changes in Japan using Landsat time series and airborne LiDAR data. ISPRS Journal of Photogrammetry and Remote Sensing, 234:111–133.

Straker, A., Puliti, S., Breidenbach, J., Kleinn, C., Pearse, G., Astrup, R., and Magdon, P. (2023). Instance segmentation of individual tree crowns with YOLOv5: A comparison of approaches using the ForInstance benchmark LiDAR dataset. ISPRS Open Journal ofPhotogrammetry and Remote Sensing, 9:100045.

Suematsu, N., Ota, T., Shimizu, K., Fukumoto, K., Mizoue, N., Inoue, A., Kitazato, H., Kusano, H., Kai, H., and Omasa, Y. (2020). The influence of sampling grid resolution and understory on forest structure estimation from terrestrial laser scanning. Japanese Journal ofForest Planning, 54(1):45–54.

Takahashi, T. and Tanaka, S. (2026). Enhancing thinning efficiency in closed-canopy conifer plantations using dynamic threshold-derived live crown ratio and canopy cover from airborne discrete-return LiDAR data. Journal of Forest Research, 31(4):262–275.

Takeshige, R., Htoo, K. K., Onishi, M., Rahman, F. M., Hoshizaki, K., Ida, H., Ishihara, M. I., Itoh, A., Kaneko, T., Katayama, A., Kuramoto, S., Kurokawa, H., Maki, M., Masaka, K., Nakaji, T., Nakamura, M., Nishimura, N., Noguchi, M., Sakai, A., Takashima, A., Tashiro, N., Tokuchi, N., Yamagawa, H., and Onoda, Y. (2025). Highresolution digital canopy height models, terrain models, ortho-mosaic photos, and canopy tree crown shapes derived from UAV-borne LiDAR at 22 tree census plots across Japanese natural forests. Ecological Research, 40(4):657– 670.

Terazaki, W. (1951). Notes on the development about the application of the statistical techniques as a tool for the scientific researches in the circle of our Japanese forestry, especially referred to the forestry mensuration, silviculture and technological management. The Journal ofthe Japanese Forestry Society, 33(9):291–306.

Tsubouchi, T. (2019). Introduction to Simultaneous Localization and Mapping. Journal ofRobotics and Mechatronics, 31(3):367–374.

Weiser, H., Schäfer, J., Winiwarter, L., Krašovec, N., Fassnacht, F. E., and Höfle, B. (2022). Individual tree point clouds and tree measurements from multi-platform laser scanning in German forests. Earth System Science Data, 14(7):2989–3012.

White, J. C., Tompalski, P., Bater, C. W., Wulder, M. A., Fortin, M., Hennigar, C., Robere-McGugan, G., Sinclair, I., and White, R. (2025). Enhanced forest inventories in Canada: Implementation, status, and research needs. Canadian Journal ofForest Research, 55:1–37.

Xiang, B., Wielgosz, M., Kontogianni, T., Peters, T., Puliti, S., Astrup, R., and Schindler, K. (2024). Automated forest inventory: Analysis of high-density airborne LiDAR point clouds with 3D deep learning. Remote Sensing of Environment, 305:114078.

Xiang, B., Wielgosz, M., Puliti, S., Král, K., Kr˚ucek, M., Missarov, A., and Astrup, R. (2025). ForestFormer3D: Aˇ Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds. arXiv:2506.16991.

Yang, Z., Qi, Z., Chen, Y., Cheng, K., Yang, H., Chen, M., Xu, J., Zhang, Y., Ren, Y., Liu, W., Lin, D., Huang, G., Xiang, T., Xu, G., and Guo, Q. (2025). Revealing the spatial distribution of crown base height across China based on close-range Lidar data. Remote Sensing ofEnvironment, 331:115030.