# A 10K drone image dataset with component-level annotations for rooftop photovoltaic inspection

Penglei Zhang<sup>1,2</sup>, Wan Zhang<sup>1,2,∗</sup>, Ruizhe Deng<sup>1,2,3</sup>, Bo Chen<sup>4</sup>, Wenhao Duan<sup>4</sup>,

Jinhe Zhang<sup>4</sup>, Yanzhen Zhang<sup>5</sup>, Qi Chen<sup>4</sup>, Zhiling Guo<sup>3</sup>, Haoran Zhang<sup>5</sup>, Yuntian Chen<sup>1,2</sup>

<sup>1</sup>Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo, Zhejiang 315200, P.R. China <sup>2</sup>Zhejiang Key Laboratory of Industrial Intelligence and Digital Twin, Eastern Institute of Technology, Ningbo, Zhejiang 315200, P.R. China

<sup>3</sup>Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Kowloon, Hong Kong, China

<sup>4</sup>School of Geography and Information Engineering, China University of Geosciences (Wuhan), Wuhan 430074, China

<sup>5</sup>School of Urban Planning and Design, Peking University, Shenzhen, 518055, China

<sup>∗</sup>Corresponding author: Wan Zhang; email: wzhang@idt.eitech.edu.cn.

## Abstract

Component-level information supports rooftop photovoltaic (PV) inspection, but recognizing individual components across complex scenes remains challenging. We present PVIS-10K, a dataset of 10,065 drone images with component-level annotations covering 63,632 physical PV modules across 22 rooftop installations. The high-resolution red-green-blue (RGB) images were acquired by unmanned aerial vehicles (UAVs, also known as drones) in Ningbo, Shaoxing and Shenzhen, China. Multi-angle oblique observations capture installation-height and scale variation, dense arrangements, occlusion, shadows and reflections. Three-dimensional delineation, reprojection and image-by-image review produced 3,516,675 image-level polygon annotations, stored in Common Objects in Context (COCO) format. Station-disjoint partitions support evaluation on unseen stations within and across cities. Four supervised models and text-prompted zero-shot Segment Anything Model 3 (SAM 3) provide full-resolution benchmarks of instance matching and foreground coverage. PVIS-10K supports component localization, delineation and image-based counting for drone-based rooftop PV inspection research.

## 1 Background & Summary

Solar photovoltaic (PV) deployment continues to expand, with distributed rooftop installations forming an important part of this growth<sup>1</sup>. The growing installed asset base increases the need for inspection and maintenance to sustain system performance<sup>2</sup>. Component locations, boundaries and arrangement provide a spatial reference for these activities and support image-based counting.

Unmanned aerial vehicles (UAVs, also known as drones) provide high-resolution rooftop observations from diferent viewing angles<sup>3</sup>, supporting component mapping for inspection<sup>4</sup>. PV inventories establish facility locations <sup>5–8</sup>, while aerial annotations and detection and segmentation studies map array boundaries and PV coverage <sup>9–14</sup>. For component-level inspection, these spatial extents must be complemented by boundaries that distinguish neighbouring physical components.

Distinguishing components in rooftop imagery is complicated by scene geometry and viewing conditions (Fig. 1). Installation height, viewing angle and distance afect projected component sizes and shapes (Fig. 1a), while dense layouts and repetitive textures complicate component separation (Fig. 1b). Partial occlusion, shadows and reflections further obscure boundaries or alter appearance (Fig. 1c,d). Internal dividing lines can also resemble boundaries between components. Treating neighbouring components as one instance, or splitting one component, changes image-level counts even when the overall PV foreground remains similar.

![](images/906a26194c12e3cfa0c739c74045cf51846d896ec798e9e453fa3239481b0845.jpg)  
a Installation geometry and scale variation

![](images/70bff7bdbc2bf2d280f8bba0cee952dcabdd0810796ed34d37a77656d99d8bf9.jpg)  
b Dense layouts and orientation variation

![](images/6a90fc0ce675289d57b5f9782ad041b365fd7814c0d1f25f6656680e42b16973.jpg)  
c Partial occlusion and cast shadows

![](images/cd7e9bf396d61978d63a0f23bbe7245f2d49fa38313053e0f30ce5173017886d.jpg)  
d Illumination and reflectance variation  
Figure 1: Challenges in drone-based rooftop PV component recognition. Panels a–d show installation geometry and scale variation, dense layouts, partial occlusion and shadows, and illumination and reflectance variation, respectively. Insets b1 and b2 in b and the lower-left inset in c show enlarged regions. Yellow outlines indicate selected ground-truth instances, including projected occluded portions.

The open-access red-green-blue (RGB) datasets compared in Table 1 describe PV arrays or coverage regions. Bradbury, BDAPPV and Queens provide aerial array annotations<sup>9–11</sup>; PV01 and Emmen support semantic segmentation from orthographic imagery <sup>13,15</sup>. These resources identify PV extent, but their labels do not separate neighbouring physical components. Their orthographic or overhead representations also difer from multi-angle rooftop observations. Complementary data are needed to evaluate component separation across changing viewing angles, projected scales and occlusion conditions.

Table 1: Representative open-access datasets for rooftop PV mapping.
<table><tr><td>Dataset</td><td>Image count</td><td>Annotation count</td><td>GSD</td><td>Target level</td><td>Annotation type</td><td>Annotation workflow</td><td>View representation</td></tr><tr><td>9,16 Bradbury</td><td>601</td><td>19,433</td><td>30</td><td>Non-</td><td>Semantic component segmentation</td><td>Manual 2D annotation</td><td>Orthographic view</td></tr><tr><td>BDAPPV¹0</td><td>46,132</td><td></td><td>10; 20</td><td>Non-</td><td>Semantic component segmentation</td><td>Crowdsourced 2D annotation</td><td>Overhead views</td></tr><tr><td>Queens11</td><td>667</td><td>~14,000</td><td>15.2</td><td>Non-</td><td>Semantic component segmentation</td><td>Manual 2D annotation</td><td>Orthographic view</td></tr><tr><td>PV01¹3</td><td>645</td><td>5,252</td><td>10</td><td>Non- component segmentation</td><td>Semantic</td><td></td><td>Orthographic view</td></tr><tr><td>Emmen 15,17</td><td>5,327</td><td>4,389</td><td>7.5</td><td>Non-</td><td>Semantic component segmentation</td><td>Manual 2D annotation</td><td>Orthographic view</td></tr><tr><td>PVIS-10K (ours)</td><td>10,065</td><td>3,516,675</td><td></td><td>~0.5-1.1 Component</td><td>Instance segmentation</td><td>3D annotation and 2D reprojection</td><td>Multi-angle oblique views</td></tr></table>

Component and non-component denote individual components and arrays or coverage regions, respectively. GSD, ground sampling distance (cm/pixel); —, not reported or undetermined. BDAPPV provides 20,989 image masks rather than a reported instance total. PVIS-10K GSD is a nominal nadir-equivalent estimate.

Delineating and checking individual component boundaries is labour-intensive, particularly when the same components appear in multiple views. We developed PVIS-10K through threedimensional annotation, two-dimensional reprojection and image-by-image correction. It contains 10,065 multi-angle UAV images from 22 rooftop PV stations in Ningbo, Shaoxing and Shenzhen. Its 3,516,675 image-level polygon annotations represent 63,632 physical components and are stored in Common Objects in Context (COCO) format<sup>18</sup>. The dataset combines component-level instance boundaries with multi-angle oblique imagery of dense layouts, height and scale variation, occlusion, shadows and reflections. Station-disjoint partitions support evaluation on unseen stations within and across cities, using supervised and text-prompted zero-shot benchmarks of component matching and foreground coverage. These resources support component localization, delineation and image-based counting for drone-based rooftop PV inspection research.

## 2 Methods

Component outlines were delineated in reconstructed three-dimensional scenes and reprojected into selected images to reduce repeated boundary drawing across views. The projected annotations underwent image-by-image review and correction, followed by annotation export and station-level partitioning (Fig. 2).

UAV image acquisition. The study covered 22 PV stations in China: 18 in Ningbo and three in Shaoxing, Zhejiang Province, and one in Shenzhen, Guangdong Province. The stations varied in roof and support height, component orientation and layout, and surrounding building context (Table 2).

RGB images were acquired from January 2024 to January 2026 using a DJI Mavic 3E along oblique photogrammetric flight routes. Image dimensions were 5280 × 3956 pixels. Flight heights were 20 m relative to the roof in Ningbo and Shenzhen and 40 m in Shaoxing, with 80% forward and 70% side overlap. The estimated nadir-equivalent GSD at these heights is approximately 0.5–1.1 cm/pixel; local image scale varies with viewing angle and roof geometry. Station-specific acquisition months, reconstruction-input image counts and release counts are listed in Supplementary Table S7.

<table><tr><td>Split</td><td>City</td><td>Sites</td></tr><tr><td>Training</td><td>Ningbo</td><td>12</td></tr><tr><td>Within-city</td><td>Ningbo</td><td>6</td></tr><tr><td>Cross-city</td><td>Shaoxing</td><td>3</td></tr><tr><td>Cross-city</td><td>Shenzhen</td><td>1</td></tr></table>

![](images/8f827f881ab310df627da3ace7fa5764632c3c218b751308341f198a24430364.jpg)  
Figure 2: Dataset construction workflow. Left, image acquisition, reconstruction and three-dimensional component annotation. Centre, projection of component polygons into selected images. Right, manual review and correction, annotation export and station-level partitioning. Dashed lines indicate projection rays; solid arrows indicate workflow progression.

Table 2: Study areas, flight heights and partition assignments.
<table><tr><td>City</td><td></td><td></td><td>Stations Height Assignment</td></tr><tr><td>Ningbo</td><td>18</td><td>20 m</td><td>12 training stations; 6 independent test stations</td></tr><tr><td>Shaoxing</td><td>3</td><td>40 m</td><td>Cross-city testing</td></tr><tr><td>Shenzhen</td><td>1</td><td>20 m</td><td>Cross-city testing</td></tr></table>

Heights are relative to the roof.

3D reconstruction and annotation. Agisoft Metashape was used for aerial triangulation and bundle adjustment within a structure-from-motion photogrammetric workflow<sup>19</sup>. Onboard real-time kinematic (RTK) positioning constrained camera locations without ground control points. Bundle adjustment yielded camera interior and exterior orientation parameters and lens distortion coeficients. The triangulation results and images were imported into iTwin Capture Modeler Center 2024 Update 1 (version 24.1.7.4760, Bentley Systems). Three-dimensional scene reconstruction retained the imported camera parameters without further optimization.

Three-dimensional component annotation was performed by a professional team, with diferent personnel responsible for annotation and quality checking. In DasViewer V3.6.0.303Beta, each physical component was manually delineated along its outer boundary on the reconstructed three-dimensional model. Each resulting three-dimensional polygon received a distinct identifier. Adjacent components were labelled separately; internal cells and dividing lines did not define instance boundaries. Annotations underwent three tiers of quality control: automated checks, manual review and expert spot checks. Each three-dimensional polygon represented one physical component; its two-dimensional projections formed separate image-level observations.

UAV image selection and reprojection. To reduce redundant observations, image sparsification used the Reduce Overlap operation in Agisoft Metashape 2.1.3 after generating a three-dimensional scene surface model for coverage analysis. Surface coverage was set to $5 ,$ defining the target number of cameras observing a surface point from diferent angles. The retained images were used for undistortion, annotation reprojection and release.

Using the adjusted camera parameters exported from Metashape, a custom Python program undistorted the retained images with radial coeficients $\boldsymbol { k } _ { 1 } , \boldsymbol { k } _ { 2 } , \boldsymbol { k } _ { 3 }$ and tangential coeficients $p _ { 1 } , p _ { 2 }$ Undistortion preserved image dimensions and principal-point coordinates without cropping; the corrected-image projection model used zero distortion coeficients. Camera parameters were reformatted, with orientations expressed as $\varphi , \omega , \kappa$ angles. The four corners of each threedimensional component polygon were projected onto the image plane using the camera poses and corrected-image intrinsics. The projected vertices were connected in their original order to form two-dimensional outlines. The collinearity relationship was expressed in homogeneous coordinates as

$$
\lambda _ { i } \mathbf { p } _ { i } = \mathbf { K } \mathbf { R } ( \mathbf { P } _ { i } - \mathbf { C } ) , \qquad i = 1 , 2 , 3 , 4 .\tag{1}
$$

Here, $\mathbf { P } _ { i } = ( X _ { i } , Y _ { i } , Z _ { i } ) ^ { \mathsf { T } }$ and $\mathbf { C } = ( X _ { c } , Y _ { c } , Z _ { c } ) ^ { \mathsf { T } }$ denote the three-dimensional component corner and camera centre in the same world coordinate system. The image point is $\mathbf { p } _ { i } = ( u _ { i } , v _ { i } , 1 ) ^ { \mathsf { T } }$ Matrix R rotates world coordinates into camera coordinates, K contains corrected-image intrinsics, and $\lambda _ { i }$ is a projective scale factor.

The resulting two-dimensional polygons were geometrically intersected with the image extent and saved as image-associated shapefile (SHP) labels, retaining only their in-image portions. Subsequent annotations used pixel coordinates with an upper-left origin, x increasing rightward and y increasing downward.

Manual annotation correction. Projected SHP labels were converted into per-image annotations for the ISAT image segmentation annotation tool and overlaid on the undistorted images. One author inspected all images and corrected their annotations, and a second author reviewed the complete corrected annotation set. Corrections addressed displaced outlines, missing components, erroneous projections, duplicate labels, and incorrectly merged or split instances. Fully occluded components were removed, whereas partially occluded components retained their complete projected outlines, including occluded regions.

Corrected objects were exported as COCO instance segmentation annotations <sup>18</sup>, with one record per object and clipped fragments retained within the same instance. Polygon vertices were preserved in segmentation, while pycocotools rasterization yielded foreground-pixel area and axis-aligned bbox, stored as [x, y, width, height]. Conversion checked image–annotation associations, image dimensions, categories and coordinates, and assigned image and instance identifiers. Self-intersecting, degenerate or invalid polygons and empty rasterized masks were excluded and recorded. Instances with bounding-box width or height below five pixels were also excluded. Images confirmed to contain no targets remained in the COCO images list without instance records.

Station-level data partitioning. To keep spatially dependent observations together <sup>20</sup>, all images, annotations and derived crops from each station were assigned to the same partition. Twelve Ningbo stations were used for supervised training and six for within-city, cross-station testing; three Shaoxing stations and one Shenzhen station provided cross-city tests (Table 2).

Release preparation. Undistorted images were saved as JPEG without copying Global Positioning System (GPS), RTK or other source positioning metadata, preserving image dimensions and annotation coordinate alignment.

## 3 Data Records

Data access and organization. PVIS-10K (version 1.0.0) is publicly available from Zenodo<sup>21</sup> at https://doi.org/10.5281/zenodo.23886365 as independently downloadable station-level ZIP archives. Archives use train/<station> or test/<station> directories with three-digit station identifiers. Each images folder contains full-resolution, undistorted JPEG images. Each training and test station directory also contains COCO annotations in instances.json, a JavaScript Object Notation (JSON) file.

The comma-separated values (CSV) file station\_splits.csv records station identifiers (station\_id), cities (city) and training or test membership (split). The evaluation\_role field distinguishes training, within-city testing and cross-city testing. Relative paths to images and annotations for both partitions are recorded in images\_path and annotations\_path, respectively. README files describe dataset use, and LICENSE.md specifies the CC BY 4.0 licence (Table 3).

Table 3: Core files in the dataset.
<table><tr><td>Path or file</td><td>Format</td><td>Content</td></tr><tr><td>train/&lt;station&gt;/images/ test/&lt;station&gt;/images/</td><td>JPEG</td><td>Full-resolution undistorted RGB images</td></tr><tr><td>train/&lt;station&gt;/instances.json test/&lt;station&gt;/instances.json</td><td>COCO JSON</td><td>Training and test image records, component polygons and category</td></tr><tr><td>station_splits.csv</td><td>CSV</td><td>definitions Station, city, partition, role and relative paths</td></tr><tr><td>README.md</td><td>Markdown</td><td>Dataset description and usage</td></tr><tr><td>README_zh-CN.md</td><td></td><td>instructions</td></tr><tr><td>LICENSE.md</td><td>Text</td><td>CC BY 4.0 licence notice</td></tr></table>

Image and annotation records. Image filenames retain their acquisition basenames with the extension .jpg. Each station’s COCO file contains images, annotations and categories records. Image records provide filenames and pixel dimensions; annotations.image\_id links each annotation to its image, and category\_id identifies the single category, pv\_panel. Image and annotation identifiers are local to each station file, with the same structure used for training and test annotations.

The segmentation field stores polygon vertices as $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } , . . . ]$ in pixel coordinates, with an upper-left origin, x increasing rightward and y increasing downward. The bbox field stores the top-left coordinates, width and height as [x, y, width, height] in pixels; area counts foreground pixels after rasterization. Multiple fragments of one instance are stored as separate vertex sequences within the same annotation.

Annotations identify per-image component observations, not persistent physical identities across views. Target-free images have image records but no associated instances. Partially occluded components retain projected outlines that include occluded portions.

## 4 Data Overview

Table 4 summarizes the composition of PVIS-10K by city and data partition. The Ningbo training subset contains 5,266 images and 1,578,541 instance annotations, accounting for just over half of the images. Among the test subsets, Shaoxing contains fewer images but more annotations than Ningbo, whereas Shenzhen is the smallest in both measures.

Table 4: Dataset size and partition composition.
<table><tr><td>City</td><td>Split</td><td>Stations</td><td>Images</td><td>2D instances</td><td>Physical components</td><td>Instances per image</td></tr><tr><td rowspan="3">Ningbo</td><td>Training</td><td>12</td><td>5,266</td><td>1,578,541</td><td>28,731</td><td>299.8</td></tr><tr><td>Test</td><td>6</td><td>2,439</td><td>723,943</td><td>13,176</td><td>296.8</td></tr><tr><td>Test</td><td>3</td><td>1,906</td><td>1,096,688</td><td>18,091</td><td>575.4</td></tr><tr><td>Shaoxing Shenzhen</td><td>Test</td><td>1</td><td>454</td><td>117,503</td><td>3,634</td><td>258.8</td></tr><tr><td>Total</td><td></td><td>22</td><td>10,065</td><td>3,516,675</td><td>63,632</td><td>349.4</td></tr></table>

Instances count per-image observations; physical components count distinct annotated objects.

Figure 3 shows station-level image counts, annotation counts and mean instances per image. Station image counts range from 58 to 1,228, whereas mean instance counts range from 23.0 to 1,092.0 per image. Station 013 supplies the most images, but stations 034 and 006 each average more than 1,000 instances per image.

![](images/0dfa4189fa026f704b86a2971d5738ceccaf87d121a803aad52bad1c0890c6d3.jpg)

Figure 3: Station-level dataset composition. a, number of images; b, two-dimensional instance annotations, in thousands; $\mathbf { c } ,$ mean number of instances per image. Colours indicate the city and training or test subset.

## 5 Technical Validation

Technical validation combined data quality checks with supervised and zero-shot model evaluation across all ten test stations (Table 4).

Data quality checks. An automated audit covered all 10,065 JPEG images and 3,516,675 instance annotations across the 22 stations. Every image was fully decoded, and its dimensions were compared with the corresponding COCO record. All released images were traced to the retained image subset by filename. Checks covered file associations, identifier uniqueness within each station, category references, polygon validity and coordinate bounds. All instance polygons were rasterized with pycocotools to independently recompute their mask areas and bounding boxes.

No missing or unreadable images, invalid references, duplicate identifiers, invalid polygon geometries or out-of-bounds coordinates were detected. Recomputed mask areas and bounding boxes agreed with the stored values for every instance, and all bounding-box dimensions were at least five pixels. The COCO image lists for the complete dataset retained 712 images without associated instances. Station-level summaries included these target-free images when calculating instance density and described component pixel scale using bounding-box dimensions and the square root of mask area.

Model settings. Evaluation included four supervised instance segmentation models and the text-prompted zero-shot Segment Anything Model 3 (SAM 3) <sup>22</sup>. Both groups used the same station partitions, JPEG images and COCO reference annotations, with predictions evaluated at full-image resolution. The supervised baselines were Mask R-CNN <sup>23</sup>, Hybrid Task Cascade (HTC)<sup>24</sup>, $\mathrm { S O L O v 2 ^ { 2 5 } }$ and Mask2Former<sup>26</sup>, implemented in MMDetection<sup>27</sup> 3.2.0. HTC excluded the semantic segmentation branch; model-specific configurations are summarized in Table 5.

Supervised training used 512 × 512-pixel crops with a stride of 448 pixels. Crops were excluded if at least 60% of their channel elements were zero, and the training loader excluded crops without instances. Labels were clipped to the same windows, removing instances with bounding-box width or height below five pixels.

Preliminary station-disjoint validation with Mask R-CNN within the 12 training stations used COCO instance segmentation average precision (AP) to select a ten-epoch schedule, which was applied to all four supervised models. Each model was then reinitialized from the original pretrained weights and trained once on all 12 stations, using its epoch-ten checkpoint for testing. Test stations were excluded from training and model selection. Training used two NVIDIA A800 graphics processing units (GPUs), each with 80 GB memory. Software environments, crop-boundary handling and the shared optimization schedule are detailed in Supplementary Methods.

SAM 3 used Meta’s oficial implementation and oficial pretrained weights downloaded from ModelScope, without training or fine-tuning on PVIS-10K. The prompt “individual solar panel” was selected through preliminary comparisons on ten training images and fixed for all test images.

Full-image inference. All five models used $5 1 2 \times 5 1 2 .$ -pixel sliding windows with a stride of 448 pixels, retaining at most 100 candidates per crop. Predictions were mapped to full-image coordinates, followed by cross-window edge-instance stitching using the criteria in Supplementary Table S6. Bounding-box non-maximum suppression retained higher-confidence predictions and suppressed lower-confidence predictions with an intersection over union (IoU) above 0.50 to reduce duplicates. A subsequent mask-containment filter removed a smaller instance when at least 50% of its area was covered by a larger retained instance, without comparing confidences.

Table 5: Supervised instance segmentation training configurations.
<table><tr><td>Parameter</td><td>Mask R-CNN</td><td>HTC</td><td>SOLOv2</td><td>Mask2Former</td></tr><tr><td>Backbone</td><td>ResNet-5028</td><td>ResNet-50</td><td>ResNet-50</td><td>ResNet-50</td></tr><tr><td>Feature processing</td><td> $\mathrm { F P N ^ { 2 9 } }$ </td><td>FPN</td><td>FPN</td><td>Multi-scale deformable-attention</td></tr><tr><td>Backbone initialization</td><td>Torchvision pretrained</td><td>Torchvision pretrained</td><td>Torchvision pretrained</td><td>pixel decoder Torchvision</td></tr><tr><td>Optimizer</td><td>SGD</td><td>SGD</td><td>SGD</td><td>pretrained AdamW</td></tr><tr><td>Base learning rate</td><td>0.02</td><td>0.02</td><td>0.01</td><td>0.0001</td></tr><tr><td>Backbone learning-rate multiplier</td><td>1</td><td>1</td><td>1</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.05</td></tr><tr><td>Momentum  $/ \beta$ </td><td>0.9</td><td>0.9</td><td>0.9</td><td>(0.9, 0.999)</td></tr><tr><td>Maximum gradient  $L _ { 2 }$  norm</td><td></td><td></td><td>35</td><td>0.01</td></tr><tr><td>Batch size per GPU</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Global batch size</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Training epochs</td><td>10</td><td>10</td><td>10</td><td>10</td></tr></table>

FPN, feature pyramid network; SGD, stochastic gradient descent; AdamW, Adam with decoupled weight decay; –, no gradient clipping configured. Training used two GPUs without gradient accumulation, giving a global batch size of 16. Mask2Former normalization parameters and query\_embed, query\_feat and level\_embed parameters have no weight decay.

Evaluation metrics. Instance segmentation was evaluated using the COCO protocol <sup>18</sup>, reporting Mask AP over IoU thresholds from 0.50 to 0.95 in increments of 0.05. Mask AP50 and Mask AP75 denote AP at IoU thresholds of 0.50 and 0.75, respectively. Box AP and its corresponding threshold-specific metrics were reported as supplementary results. City-level AP was recalculated from all images, annotations and predictions within a city, rather than averaged across station AP values. Evaluation allowed up to 5,000 predictions per full image.

For pixel-level evaluation, predicted and ground-truth instance masks were separately united within each image. Foreground IoU and the $F _ { 1 }$ score were computed using their standard definitions<sup>30</sup>. City-level scores were calculated from confusion counts accumulated across all images in the city.

Supervised model evaluation. HTC achieved the highest Mask AP among the four supervised models in each tested city (Table 6). Mask AP was 72.72% in Ningbo, 57.98% in Shaoxing and 90.49% at the single Shenzhen test station. Performance on the cross-city Shenzhen set exceeded that on the within-city Ningbo set.

Station-level results revealed further variation within cities. HTC obtained 92.13% Mask AP at Ningbo station 024, compared with 45.72% at 013 and 45.17% at 030. In Shaoxing, its Mask AP was 74.47%, 78.08% and 38.25% at stations 004, 005 and 006, respectively. Supplementary Tables S1–S4 provide complete station-level supervised results.

Zero-shot SAM 3 evaluation. SAM 3 obtained Mask AP values of 4.75%, 33.37% and 2.46% in Ningbo, Shaoxing and Shenzhen, respectively, below all corresponding supervised baselines. Its relatively higher instance performance in Shaoxing contrasted with the city-level pattern of the supervised models. Complete station-level results are provided in Supplementary Table S5.

Table 6: City-level instance segmentation and foreground coverage results (%).
<table><tr><td>City</td><td>Model</td><td>Mask AP</td><td>AP50</td><td>AP75</td><td>IoU</td><td> $F _ { 1 }$ </td></tr><tr><td rowspan="5">Ningbo</td><td>Mask R-CNN</td><td>65.37</td><td>79.88</td><td>68.17</td><td>90.77</td><td>95.16</td></tr><tr><td>HTC</td><td>72.72</td><td>85.87</td><td>75.57</td><td>90.42</td><td>94.97</td></tr><tr><td>SOLOv2</td><td>68.93</td><td>83.59</td><td>73.60</td><td>91.87</td><td>95.76</td></tr><tr><td>Mask2Former</td><td>65.18</td><td>79.59</td><td>66.89</td><td>92.65</td><td>96.18</td></tr><tr><td>SAM 3</td><td>4.75</td><td>7.68</td><td>4.95</td><td>89.49</td><td>94.45</td></tr><tr><td rowspan="5">Shaoxing</td><td>Mask R-CNN</td><td>55.30</td><td>68.75</td><td>60.86</td><td>90.14</td><td>94.81</td></tr><tr><td>HTC</td><td>57.98</td><td>72.09</td><td>63.89</td><td>90.95</td><td>95.26</td></tr><tr><td>SOLOv2</td><td>54.49</td><td>69.91</td><td>60.41</td><td>89.65</td><td>94.54</td></tr><tr><td>Mask2Former</td><td>49.24</td><td>62.80</td><td>53.44</td><td>89.03</td><td>94.19</td></tr><tr><td>SAM 3</td><td>33.37</td><td>38.40</td><td>36.74</td><td>86.36</td><td>92.68</td></tr><tr><td rowspan="5">Shenzhen</td><td>Mask R-CNN</td><td>88.57</td><td>97.80</td><td>91.91</td><td>93.33</td><td>96.55</td></tr><tr><td>HTC</td><td>90.49</td><td>98.91</td><td>94.15</td><td>94.80</td><td>97.33</td></tr><tr><td>SOLOv2</td><td>88.54</td><td>97.90</td><td>94.03</td><td>94.58</td><td>97.21</td></tr><tr><td>Mask2Former</td><td>87.42</td><td>98.71</td><td>89.85</td><td>92.86</td><td>96.30</td></tr><tr><td>SAM 3</td><td>2.46</td><td>3.77</td><td>2.30</td><td>90.28</td><td>94.89</td></tr></table>

Bold values indicate the highest score for each metric within each city. AP50 and AP75 refer to mask-based evaluation. IoU and $F _ { 1 }$ refer to the foreground union of instance masks. City-level AP is recomputed from pooled predictions; pixel metrics use accumulated confusion counts.

Analysis of evaluation results. Instance matching and foreground coverage produced diferent model rankings (Table 6). In Ningbo, HTC achieved the highest Mask AP, whereas Mask2Former achieved the highest foreground IoU of 92.65%. Despite its low Mask AP, SAM 3 retained foreground IoU values of 89.49% in Ningbo and 90.28% in Shenzhen. Foreground unions discard instance identities, so duplicate, merged or split predictions can preserve coverage while reducing component-level matching accuracy.

Figure 4 compares HTC and SAM 3 predictions across four regions from three Ningbo test stations. SAM 3 misidentifies boundaries between densely arranged components $\left( \mathrm { F i g . \ A a } \right)$ Under structural occlusion, predictions following visible portions difer from the full projected annotations (Fig. 4c). Under inter-row occlusion, SAM 3 predicts instances spanning adjacent components (Fig. 4b). In the region of contrasting brightness, SAM 3 incorrectly merges multiple PV components into a single predicted instance (Fig. 4d).

## 6 Usage Notes

Extract station archives into the directory structure described in Section 3, preserving image filenames and image–annotation associations. Combining station-level COCO files requires remapping image and instance identifiers and updating their links. Released images are already undistorted and require no further distortion correction. Annotation coordinates are pixel coordinates, not geographic coordinates. Polygon annotations can be rasterized using pycocotools. Partially occluded components retain projected outlines that include occluded regions, rather than masks of visible surfaces alone.

Training crops can be generated with the repository’s tools using the window size, stride and filtering rules specified under Model settings in Technical Validation and in Supplementary Methods. Image and annotation crops must share window locations and edge handling. Each station and its derived crops must remain in the same partition. Any validation subset should be drawn from training stations, keeping stations disjoint between training and validation.

Baseline reproduction should follow the model settings reported in Technical Validation. Predictions from other methods should be mapped to complete test images and evaluated against the released test annotations using the same evaluation protocol.

![](images/907b378f9687a1f3e5e7fa3b14909070e06a0e2ce14de4940b6950744c0ad52b.jpg)  
Figure 4: Qualitative comparison of component-level instance segmentation. Columns show the original image, ground truth, HTC and SAM 3 predictions. Rows show a, densely packed components (station 022); b, inter-row occlusion (032); c, structural occlusion (028); and d, brightness variation (022). Crop and display scale are matched within each row. Green denotes ground-truth masks in the reference column and correct predictions in the model columns; yellow denotes split instances, where one PV component is divided into multiple predicted instances; purple denotes merged instances, where multiple PV components are combined into one predicted instance; and blue denotes missed instances, where a ground-truth component is not detected. Blue overlays mark missed ground-truth instances, not predicted masks. Colours illustrate the error types shown in these examples.

## Data Availability

PVIS-10K (version 1.0.0) is openly available from Zenodo <sup>21</sup> at https://doi.org/10.5281/ze nodo.23886365. The release includes undistorted JPEG images and COCO annotations for all 22 training and test stations, station-level partition definitions and supporting documentation. These materials are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0; https://creativecommons.org/licenses $/ \mathrm { b y } / 4 . 0 / )$ . Three-dimensional models, camera parameters and intermediate annotation files are not included.

## Code Availability

Data-processing scripts, model configurations and inference and evaluation code are publicly available at https://github.com/plzhang1998/PVDataset. The repository provides projectspecific scripts and configurations; the MMDetection and SAM 3 frameworks must be installed separately. Dependencies and execution instructions are documented in the repository. Original project code and modifications are licensed under Apache License 2.0; third-party components retain their respective licences. The trained checkpoints of the four supervised baselines are not distributed.

## References

1. de l’Epine, M., Dominguez, G., Kaizuka, I. & Jäger-Waldau, A. Snapshot of Global PV Markets 2026. Report IEA-PVPS T1-49:2026 (IEA PVPS, 2026). https://doi.org/10.69766 /WXLW1389

2. Jahn, U. et al. Guidelines for Operation and Maintenance of Photovoltaic Power Plants in Diferent Climates. Report IEA-PVPS T13-25:2022 (IEA PVPS, 2022). https://iea-pvps.or g/key-topics/guidelines-for-operation-and-maintenance-of-photovoltaic-power-plants-in-d ifferent-climates/

3. Colomina, I. & Molina, P. Unmanned aerial systems for photogrammetry and remote sensing: A review. ISPRS J. Photogramm. Remote Sens. 92, 79–97 (2014). https://doi.org/10.1016/j. isprsjprs.2014.02.013

4. Bommes, L. et al. Computer vision tool for detection, mapping, and fault classification of photovoltaics modules in aerial IR videos. Prog. Photovolt. Res. Appl. 29, 1236–1251 (2021). https://doi.org/10.1002/pip.3448

5. Dunnett, S., Sorichetta, A., Taylor, G. & Eigenbrod, F. Harmonised global datasets of wind and solar farm locations and power. Sci. Data 7, 130 (2020). https://doi.org/10.1038/s41597 -020-0469-8

6. Kruitwagen, L. et al. A global inventory of photovoltaic solar energy generating units. Nature 598, 604–610 (2021). https://doi.org/10.1038/s41586-021-03957-7

7. Yu, J., Wang, Z., Majumdar, A. & Rajagopal, R. DeepSolar: A Machine Learning Framework to Eficiently Construct a Solar Deployment Database in the United States. Joule 2, 2605–2617 (2018). https://doi.org/10.1016/j.joule.2018.11.021

8. Li, A. et al. Global photovoltaic solar panel dataset from 2019 to 2022. Sci. Data 12, 637 (2025). https://doi.org/10.1038/s41597-025-04985-y

9. Bradbury, K. et al. Distributed solar photovoltaic array location and extent dataset for remote sensing object identification. Sci. Data 3, 160106 (2016). https://doi.org/10.1038/sd ata.2016.106

10. Kasmi, G. et al. A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata. Sci. Data 10, 59 (2023). https://doi.org/10.1038/s41597-0 23-01951-4

11. Furedi, T., Kimsal, E., Cornejo, S., Liero, N. & Ranalli, J. Labeled photovoltaic installations for orthographic aerial imagery in Queens, New York. Sci. Data 13, 207 (2026). https: //doi.org/10.1038/s41597-025-06523-2

12. Malof, J. M., Bradbury, K., Collins, L. M. & Newell, R. G. Automatic detection of solar photovoltaic arrays in high resolution aerial imagery. Appl. Energy 183, 229–240 (2016). https://doi.org/10.1016/j.apenergy.2016.08.191

13. Jiang, H. et al. Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery. Earth Syst. Sci. Data 13, 5389–5401 (2021). https://doi.org/10.5194/es sd-13-5389-2021

14. García, G. et al. Generalized deep learning model for photovoltaic module segmentation from satellite and aerial imagery. Sol. Energy 274, 112539 (2024). https://doi.org/10.1016/j. solener.2024.112539

15. Luimstra, G. & Bunte, K. Solar Panel Segmentation on Aerial Images using Color and Elevation Information. In Proc. European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 195–200 (2025). https://doi.org/10.1 4428/esann/2025.ES2025-81

16. Bradbury, K. et al. Distributed Solar Photovoltaic Array Location and Extent Data Set for Remote Sensing Object Identification. figshare https://doi.org/10.6084/m9.figshare.3385780. v4 (2020).

17. Luimstra, G. Annotated High-Resolution Aerial Imagery of the Dutch Landscape for Solar Panel Detection and Segmentation. Zenodo https://doi.org/10.5281/zenodo.14860030 (2025).

18. Lin, T.-Y. et al. Microsoft COCO: Common Objects in Context. In Computer Vision – ECCV 2014, 740–755 (2014). https://doi.org/10.1007/978-3-319-10602-1\_48

19. Westoby, M. J., Brasington, J., Glasser, N. F., Hambrey, M. J. & Reynolds, J. M. ’Structurefrom-Motion’ photogrammetry: A low-cost, efective tool for geoscience applications. Geomorphology 179, 300–314 (2012). https://doi.org/10.1016/j.geomorph.2012.08.021

20. Roberts, D. R. et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40, 913–929 (2017). https://doi.org/10.1111/ecog.02881

21. Zhang, P. et al. PVIS-10K: A 10K drone image dataset with component-level annotations for rooftop photovoltaic inspection, version 1.0.0. Zenodo https://doi.org/10.5281/zenodo.2 3886365 (2026).

22. Carion, N. et al. SAM 3: Segment Anything with Concepts. In Proc. International Conference on Learning Representations (2026). https://openreview.net/forum?id=r35clVtGzw

23. He, K., Gkioxari, G., Dollár, P. & Girshick, R. Mask R-CNN. In Proc. IEEE International Conference on Computer Vision, 2980–2988 (2017). https://doi.org/10.1109/ICCV.2017.322

24. Chen, K. et al. Hybrid Task Cascade for Instance Segmentation. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4969–4978 (2019). https://doi.org/ 10.1109/CVPR.2019.00511

25. Wang, X., Zhang, R., Kong, T., Li, L. & Shen, C. SOLOv2: Dynamic and Fast Instance Segmentation. In Advances in Neural Information Processing Systems 33, 17721–17732 (2020). https://proceedings.neurips.cc/paper/2020/hash/cd3afef9b8b89558cd56638c36318 68a-Abstract.html

26. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A. & Girdhar, R. Masked-Attention Mask Transformer for Universal Image Segmentation. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1280–1289 (2022). https://doi.org/10.1109/CVPR52688.20 22.00135

27. Chen, K. et al. MMDetection: Open MMLab Detection Toolbox and Benchmark. Preprint at https://doi.org/10.48550/arXiv.1906.07155 (2019).

28. He, K., Zhang, X., Ren, S. & Sun, J. Deep Residual Learning for Image Recognition. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

29. Lin, T.-Y. et al. Feature Pyramid Networks for Object Detection. In Proc. IEEE Conference on Computer Vision and Pattern Recognition, 936–944 (2017). https://doi.org/10.1109/CV PR.2017.106

30. Reinke, A. et al. Understanding metric-related pitfalls in image analysis validation. Nat. Methods 21, 182–194 (2024). https://doi.org/10.1038/s41592-023-02150-0

## Competing Interests

The authors declare no competing interests.

## Funding

This work was supported by the 2024 First Batch Yongjiang Talent Programme for Young Innovative Talents (Grant No. 2024A-187-G), under the project “Research and Application of Hightransparency and High-brightness Object Target Detection and Visual Recognition Technology”, as well as by the National Natural Science Foundation of China (Grant No. 42371475).