# Exploring 2D backbone efects for indoor semantic occupancy prediction

Shizhang Fang<sup>a</sup>, Wanling Ye<sup>a</sup>, and Qi Zheng\*<sup>a</sup>

<sup>a</sup>College of Electronics and Information Engineering, Shenzhen University, Shenzhen, China

## ABSTRACT

Semantic occupancy prediction gives an embodied agent a voxel-level account of where space is free, occupied, and semantically meaningful. In RGB-D pipelines such as EmbodiedScan, the image encoder is often left as a default module, even though its features are the visual evidence later sampled into the 3D grid. We study this design choice directly.

A central finding is that changing the 2D backbone improves occupancy accuracy more than several carefully designed occupancy architectures or modules. We keep the main RGB-D projection, depth branch, and occupancy head fixed, and replace only the image backbone. The compared encoders are CLIP-ResNet, CLIP-ViT, BLIP2, and DINOv2. Under the controlled setting, the measured mIoU changes substantially: DINOv2 obtains 30.55%, BLIP2 obtains 29.49%, CLIP-ViT obtains 24.33%, and CLIP-ResNet obtains 17.41%. The stronger encoders also exceed the original EmbodiedScan ResNet-50 baseline without modifying the downstream 3D fusion pipeline. Class-level results give a more detailed picture: DINOv2 is stronger on many layout and structural categories, whereas BLIP2 remains close on several object-centered classes. CLIP-ViT improves clearly over CLIP-ResNet, showing that the way CLIP features are exposed as dense tokens matters for voxel lifting. These results indicate that the image backbone is not a secondary engineering detail in embodied semantic occupancy, but a major source of variation in the final 3D prediction.

Keywords: Semantic occupancy prediction, embodied scene understanding, RGB-D perception

## 1. INTRODUCTION

For an embodied agent, recognizing a chair in an image is not enough. The agent also needs to know where free space ends, which parts of the room are hidden, and how object labels are arranged in a shared 3D frame. Indoor RGB-D scene understanding benchmarks and recent indoor occupancy studies make this requirement increasingly concrete.<sup>1</sup> Semantic occupancy prediction addresses this need by assigning occupancy and semantic labels to voxels. The resulting grid is a practical intermediate representation for navigation, manipulation, and scene memory because it keeps geometry and semantics in the same coordinate system.

The field has moved quickly, but along several diferent axes. TPVFormer<sup>2</sup> and VoxFormer<sup>3</sup>mainly ask how the 3D representation can be made cheaper or more efective. SelfOcc<sup>4</sup> and OccCLIP<sup>5</sup> look instead at supervision cost and semantic generalization. Indoor RGB-D scenes add a separate set of issues: furniture occludes other furniture, categories are unevenly distributed, and many objects are only partially seen from an ego-centric trajectory. EmbodiedScan,<sup>6</sup> built from resources such as ScanNet, Matterport3D, and 3RScan, makes this indoor setting measurable.

In current semantic occupancy studies, most attention is placed on view transformation, voxel optimization, or multi-modal fusion architectures. Under this convention, the 2D image backbone is often treated as a pluggable and secondary engineering detail, and its choice usually follows a default setting rather than a controlled analysis.

This backbone-agnostic view is questionable in a visually dependent pipeline such as EmbodiedScan. Each voxel receives semantic evidence by projecting its center onto the image feature map and sampling the corresponding 2D descriptor. Therefore, the alignment quality and class separability of the image features constrain the prediction before 3D fusion takes place. If the initial 2D features fail to preserve fine-grained geometric boundaries, increasing the complexity of the later 3D decoder may not recover the lost spatial-semantic information.

This motivates a direct test of the backbone-agnostic view. If the 2D representation already limits the quality of voxel-level semantic evidence, then backbone choice should be treated as a core experimental variable rather than a default implementation detail.

This issue becomes more visible now that pretrained visual encoders are no longer variants of the same recipe. CLIP<sup>7</sup> brings image-language alignment; BLIP2<sup>8</sup> uses a frozen visual encoder inside a vision-language system; DINOv2<sup>9</sup> is trained self-supervised and is often used for dense visual correspondence. These objectives do not encourage the same spatial behavior. A representation that separates image-level concepts may not preserve object boundaries after projection, while a locally stable representation may be easier to reuse in voxel prediction.

Our study isolates this single variable. We do not introduce a new decoder or a new fusion block. Instead, we keep the RGB-D data pipeline, point branch, voxel projection, fusion neck, occupancy head, losses, optimizer, schedule, and evaluation protocol fixed, and change only the 2D encoder. With this setup, diferences in mIoU are mainly diferences in the visual representation that is lifted from image space into the 3D grid.

Concretely, we compare four pretrained backbones: CLIP-ResNet, CLIP-ViT, BLIP2, and DINOv2. The first two compare diferent visual architectures under CLIP-style language supervision, BLIP2 represents visionlanguage pretraining with a frozen visual encoder, and DINOv2 represents large-scale self-supervised visual pretraining. All variants are adapted to the same FPN-compatible interface and inserted into the same projectionbased RGB-D occupancy model.

The measured efect is larger than expected for a front-end replacement. DINOv2 reaches 30.55% mIoU, BLIP2 reaches 29.49%, CLIP-ViT reaches 24.33%, and CLIP-ResNet reaches 17.41%. This gap is larger than the improvement from several occupancy-specific module changes reported on the same benchmark. The CLIP-ViT result is also much higher than CLIP-ResNet, so visual architecture matters even inside a related pretraining family. Class-wise results split the story further: DINOv2 is more reliable on many structural and layout-related categories, while BLIP2 remains close on several object-centered categories.

The main contributions of this paper are:

• We present a controlled study of image backbone efects for EmbodiedScan semantic occupancy prediction, isolating the 2D encoder while fixing the RGB-D occupancy pipeline.

• We compare CLIP-ResNet, CLIP-ViT, BLIP2, and DINOv2 under the same projection-based 3D fusion framework, and show that DINOv2 gives the best overall transfer while BLIP2 remains close.

• We benchmark the backbone variants against the original EmbodiedScan result, our EmbodiedScan reproduction, DROcc(SwinU), and DROcc, showing that the 2D backbone alone can exceed several task-specific architectural baselines.

• We analyze both common categories and the 15 least frequent categories, finding that stronger backbones improve rare-object prediction but do not remove the long-tail weakness of indoor occupancy.

## 2. RELATED WORK

Semantic occupancy prediction. Semantic occupancy prediction inherits the goal of semantic scene completion: a partial observation must be converted into a voxel-level description of geometry and semantics. Recent work has improved this task by changing how views are transformed, how voxels are represented, how supervision is reduced,<sup>4</sup> or how semantic priors are introduced.<sup>5</sup> After August 2025, occupancy research has also moved toward causal 2D-to-3D lifting,<sup>10</sup> vertical-slice representations,<sup>11</sup> pseudo-label supervision,<sup>12</sup> temporal world modeling,<sup>13</sup> implicit occupancy supervision for vision-language-action models,<sup>14</sup> LiDAR-free native 3D supervision,<sup>15</sup> and generalized urban occupancy.<sup>16</sup> These directions are important, but they mainly alter the 3D representation or the decoder side of the pipeline. Our work asks a complementary question: before those 3D modules operate, how much does the sampled 2D visual representation already determine the final occupancy result?

Indoor embodied occupancy. Indoor embodied scenes are diferent from outdoor driving scenes. They are cluttered, object categories are long-tailed, and RGB-D observations are collected from ego-centric viewpoints. EmbodiedScan<sup>6</sup> provides the benchmark setting used in this paper, and its RGB-D baseline follows a projectionbased design: image features are sampled into a voxel grid, depth features provide geometric evidence, and both volumes are fused for occupancy prediction. $\mathrm { D R O c c ^ { 1 7 } }$ improves indoor RGB-D occupancy with a stronger fusion architecture. In contrast, we keep the EmbodiedScan-style pipeline fixed and study whether the image backbone itself changes the dense 3D prediction.

Image backbones for projected 3D features. Projection-based 3D perception does not use image features only for image-level classification. The feature map is sampled at projected voxel locations, moved into 3D, and fused with geometry. This makes the visual encoder a more consequential choice than it may appear in a modular implementation. Our comparison covers a convolutional CLIP variant based on ResNet,<sup>7,</sup> <sup>18</sup> a CLIP-ViT variant, the frozen visual encoder used in BLIP2,<sup>8</sup> and $\mathrm { D I N O v 2 . ^ { 9 } }$ The goal is not to introduce another occupancy decoder, but to measure how these pretrained visual representations behave under the same RGB-D lifting and fusion pipeline.

## 3. METHOD

This section fixes the experimental ground on which the backbone comparison is made. We use the same RGB-D occupancy pipeline for all runs and change only the image encoder. The point branch, projection operation, fusion neck, occupancy head, losses, optimizer, and evaluator are kept in place, so the comparison is about the visual features that enter the voxel grid rather than about a new 3D architecture.

## 3.1 Pipeline Overview

Figure 1 shows the pipeline used in every experiment. A sample contains several posed RGB-D views. The RGB branch sends each image through a replaceable 2D backbone and an FPN. For voxel lifting, the model projects 3D grid centers onto the 2D feature map, samples the descriptors at those projected locations, and writes the sampled descriptors back to the corresponding grid cells. The depth branch follows a separate route: depth maps are converted to scene points, voxelized, and encoded by a Minkowski ResNet-34. The resulting image and depth volumes are concatenated before the 3D neck and occupancy head.

## 3.2 Problem Formulation

Given a scene observed by V posed RGB-D views, each sample contains RGB images $\{ I _ { v } \} _ { v = 1 } ^ { V }$ , depth observations $\{ D _ { v } \} _ { v = 1 } ^ { V }$ , camera intrinsics $\{ K _ { v } \} _ { v = 1 } ^ { V }$ , and camera-to-scene transformations. The target is a semantic occupancy grid

$$
Y \in \{ 0 , 1 , \dots , C \} ^ { X \times Y \times Z } ,\tag{1}
$$

where each voxel is assigned either empty space or one of $C$ semantic object categories. We follow the EmbodiedScan occupancy setting;<sup>6</sup> the concrete grid size, scene range, and category count are reported with the experimental protocol in Section 4.1.

The central question of this work is how the 2D visual representation afects the final 3D occupancy prediction. Let ϕ denote the image backbone and let the remaining occupancy network be denoted by $f _ { \theta }$ . The prediction can be written abstractly as

$$
\hat { Y } = f _ { \theta } ( \phi ( I _ { 1 : V } ) , D _ { 1 : V } , K _ { 1 : V } , T _ { 1 : V } ) ,\tag{2}
$$

where $D _ { 1 : V }$ are depth observations and $T _ { 1 : V }$ are camera poses. We compare diferent choices of $\phi$ while keeping $f _ { \theta } ,$ , data processing, voxelization, training schedule, and losses unchanged. This isolates the role of the transferred image representation from other sources of architectural or optimization variation.

![](images/c7620c443593b2e11f7f0c1473c74f6b64388c05a878a97d912fe31741b5df47.jpg)

(a) Fixed RGB-D occupancy pipeline  
![](images/478d5bbc97fdb2788257c661ea77e4d62cd7c41cba774a887b83f1b2194d3f52.jpg)  
(b) Alternative image backbones  
Figure 1: Controlled pipeline used for the backbone comparison. (a) The RGB branch produces an FPNcompatible 2D feature hierarchy, and the depth branch produces a sparse-to-dense 3D feature volume. For both branches, feature resolutions decrease from left to right. Voxel centers or sparse voxel coordinates are projected to the selected image feature map, the corresponding descriptors are sampled, and the sampled descriptors are written back to the same 3D grid coordinates before RGB-D fusion. The projection step uses visibility and image-boundary masks but does not introduce an additional occlusion reasoning module. (b) The experimental intervention replaces only the image backbone with CLIP-ResNet, CLIP-ViT, BLIP2, or DINOv2. For transformer backbones, class tokens are discarded and patch tokens are reshaped into spatial maps before FPN adaptation. The projection rule, depth branch, fusion neck, occupancy head, losses, and training protocol are kept fixed.

## 3.3 Fixed RGB-D Occupancy Pipeline

We build on the multi-view RGB-D occupancy pipeline of EmbodiedScan.<sup>6</sup> Each training sample uses multiple posed RGB-D views. RGB images are normalized before feature extraction. Depth maps are converted into point clouds, transformed into a shared scene coordinate frame, range filtered, and sampled to a fixed number of points. The occupancy target is represented in the same metric scene frame as the voxel grid.

Image feature extraction. Each RGB view is passed through the same 2D image encoder and feature pyramid network (FPN).<sup>19</sup> The backbone output is converted to the channel layout expected by the FPN. For ResNet-style backbones, the native convolutional stages provide the feature hierarchy. For ViT-style backbones, we discard the class token, reshape the remaining patch tokens according to the image patch grid, apply linear or 1 × 1 projections to match the FPN channel dimensions, and resize the resulting spatial maps to the same feature scales used by the baseline FPN. We use the highest-resolution FPN level for lifting, because the sampling operation is tied to projected voxel centers and loses detail if the image map is too coarse. In the sparse stage, the same calibrated projection is applied to the coordinates of the sparse voxel set $V _ { 4 } ;$ sampled image descriptors are indexed back by their original voxel coordinates, which keeps $I _ { 4 }$ and $V _ { 4 }$ aligned in the shared scene grid.

Projection-based voxel lifting. For each voxel center $x _ { g }$ in the target 3D grid, the model first projects the grid point onto the 2D feature map of every selected camera view using the known camera parameters:

$$
u _ { v , g } = \pi ( K _ { v } T _ { v } x _ { g } ) ,\tag{3}
$$

where $\pi ( \cdot )$ is the perspective projection. A valid projected point is used to bilinearly sample the 2D feature map. The sampled descriptor is not a new 3D point; it is written back to the voxel center that produced the projection. Thus, in implementation terms, the operation is closer to “project a grid point, sample an image descriptor, and store it at the same grid point”. For multiple views, we average only the valid samples:

$$
F _ { g } ^ { i m g } = \frac { \sum _ { v = 1 } ^ { V } m _ { v , g } S ( F _ { v } , u _ { v , g } ) } { \operatorname* { m a x } ( \sum _ { v = 1 } ^ { V } m _ { v , g } , 1 ) } ,\tag{4}
$$

Here $F _ { v }$ is the feature map for view $v , S ( \cdot )$ denotes bilinear sampling, and $m _ { v , g }$ marks whether grid cell g has a valid projection in that view. The mask checks whether the projected point lies inside the image and has a valid camera projection. It does not perform explicit z-bufering or depth-order reasoning among voxels that fall on the same image pixel. Therefore, an occluded voxel may still receive an appearance descriptor from a visible surface along the same ray. We keep this lifting rule unchanged for all backbones, so the comparison remains controlled, but this limitation explains why the 2D encoder is not the only bottleneck for categories with weak observability. Because this step copies image descriptors into the volume before 3D fusion, errors in local alignment, visibility, or semantic separability can directly afect the final occupancy logits.

Depth branch and dense fusion. The point branch uses the aggregated depth points as a geometric input. Points are voxelized into a sparse tensor and processed by a Minkowski ResNet-34.<sup>20</sup> The final sparse feature level is densified into the target voxel grid, producing a depth feature volume $F ^ { d e p }$ . The image and depth volumes are concatenated channel-wise:

$$
F ^ { f u s e } = [ F ^ { i m g } ; F ^ { d e p } ] ,\tag{5}
$$

and processed by a 3D neck followed by an occupancy head. The resulting logits are

$$
Z = h _ { \theta } ( F ^ { f u s e } ) .\tag{6}
$$

This fusion design gives the model complementary cues: depth supplies metric geometry and visible surface structure, while image features supply appearance and semantic evidence for class assignment.

## 3.4 Training Objective

The occupancy head is supervised with voxel-wise semantic labels. Invisible voxels are assigned the ignore label 255 and are excluded from cross-entropy. Following the baseline implementation, the total loss combines semantic cross-entropy, semantic scaling loss, and geometric scaling loss:

$$
\mathcal { L } = \sum _ { s } \lambda _ { s } \left( \mathcal { L } _ { c e } ^ { s } + \mathcal { L } _ { s e m } ^ { s } + \mathcal { L } _ { g e o } ^ { s } \right) ,\tag{7}
$$

where s indexes the supervision scale and $\lambda _ { s } = 0 . 5 ^ { s }$ down-weights lower-resolution predictions. Cross-entropy optimizes the voxel-wise semantic decision, semantic scaling encourages class-level consistency, and geometric scaling strengthens the occupied-versus-empty structure. These losses match the fixed occupancy baseline and are not changed across backbone variants.

## 3.5 Controlled Backbone Intervention

For the backbone study, only $\phi$ is replaced. This restriction matters because occupancy scores are otherwise sensitive to voxel size, point sampling, depth quality, decoder capacity, and loss weights. Holding these choices fixed keeps the experiment focused on the image representation that is copied into the grid.

We evaluate four pretrained visual backbones.

• CLIP-ResNet. This variant uses a ResNet-style image encoder initialized from CLIP-pretrained weights.<sup>7,</sup> <sup>18</sup> CLIP pretraining aligns image representations with language supervision and often produces strong imagelevel semantic discrimination. In our setting, it tests whether such global semantic alignment transfers efectively to dense voxel prediction.

• CLIP-ViT. This variant uses a CLIP-pretrained Vision Transformer visual trunk.<sup>7</sup> Patch tokens from intermediate transformer layers are reshaped into spatial feature maps and projected to the FPN channel dimensions. This model tests whether CLIP-style language supervision becomes more suitable for voxel lifting when the visual encoder exposes dense token features rather than a convolutional hierarchy.

• BLIP2 visual backbone. This variant uses the frozen visual encoder from BLIP2.<sup>8</sup> Its features are adapted into FPN-compatible spatial maps and channel dimensions. BLIP2 provides vision-language representations shaped by image-text and query-based modeling, making it a useful comparison for object-centric semantic transfer.

• DINOv2 backbone. This variant uses a DINOv2 Vision Transformer.<sup>9</sup> Patch tokens are reshaped into 2D feature maps, projected to the channel dimensions expected by FPN, and resized to the required feature scales. DINOv2 is trained with self-supervised objectives that are known to preserve strong visual structure and dense correspondence cues, which are especially relevant to projection-based 3D lifting.

All four variants output feature levels compatible with the same downstream occupancy model. The adapters only make the feature tensor shapes consistent; they do not change the voxel lifting rule, 3D fusion module, or supervision. In Fig. 1, Backbone 3 denotes the CLIP-ViT setting, where dense patch tokens come from CLIP language-aligned pretraining. Backbones 4 and 5 denote BLIP2 and DINOv2, respectively; they use the same token-to-map and FPN adaptation interface, but difer in the pretrained visual encoder and learning objective. This design keeps the adapter interface fixed while testing whether language-aligned, vision-language, and selfsupervised visual representations behave diferently after projection into 3D. Therefore, the comparison measures how diferent pretraining paradigms and visual architectures behave after being inserted into the same geometric occupancy pipeline.

## 3.6 Evaluation Principle

All variants are evaluated with the same semantic occupancy protocol. We report mIoU and class-wise IoU because the average alone hides much of the behavior in indoor scenes. Large structural classes such as floor and wall are observed often, while small furniture and appliances appear sparsely. A useful backbone should therefore help the mean score without collapsing the less frequent categories.

## 4. EXPERIMENTS

This section evaluates how diferent image backbones afect EmbodiedScan semantic occupancy prediction. All backbone variants are tested under the RGB-D occupancy pipeline described in Section 3. The purpose of the experiments is therefore not to compare diferent decoders, losses, or input modalities, but to isolate the contribution of the 2D image representation that is lifted into the 3D voxel volume.

## 4.1 Dataset and Evaluation Protocol

Experiments are conducted on the EmbodiedScan semantic occupancy benchmark.<sup>6</sup> Each scene is represented by posed RGB-D observations and evaluated on a dense voxel grid. Following the original occupancy setting, the model predicts a 40 × 40 × 16 grid with 81 channels, corresponding to empty space and 80 semantic object categories. We report mean Intersection-over-Union (mIoU) over all semantic categories and further analyze class-wise IoU values.

The main comparison is controlled across four image backbones: CLIP-ResNet, CLIP-ViT, BLIP2, and DINOv2. The same data split, RGB-D inputs, 3D branch, fusion head, training objective, and evaluation metric are used for all variants. This protocol makes the measured diferences attributable mainly to the image backbone and its transferred feature representation.

Table 1: RGB-D semantic occupancy comparison on EmbodiedScan. Values are IoU in percentage. “Baseline” denotes EmbodiedScan. “Rep.” is our reproduction of the EmbodiedScan RGB-D baseline. “b.” denotes replacing only the image backbone in the fixed RGB-D pipeline. “refri.” denotes refrigerator.
<table><tr><td>Method</td><td>mIoU</td><td>empty</td><td>floor</td><td>wall</td><td>chair</td><td>cabinet</td><td>door</td><td>table</td><td>couch</td><td>shelf</td><td>window</td><td>bed</td><td>curtain</td><td>refri.</td><td>plant</td><td>stairs</td><td>toilet</td></tr><tr><td>Baseline6</td><td>19.97</td><td>71.21</td><td>64.92</td><td>55.00</td><td>52.04</td><td>27.35</td><td>33.97</td><td>47.93</td><td>46.26</td><td>31.87</td><td>27.98</td><td>46.58</td><td>46.56</td><td>24.05</td><td>39.01</td><td>24.40</td><td>67.79</td></tr><tr><td>Baseline (rep.)6</td><td>21.20</td><td>73.55</td><td>69.61</td><td>53.33</td><td>52.64</td><td>30.11</td><td>31.37</td><td>42.49</td><td>46.46</td><td>41.92</td><td>28.00</td><td>49.30</td><td>47.00</td><td>19.60</td><td>36.75</td><td>32.52</td><td>62.46</td></tr><tr><td>DROcc(SwinU)17</td><td>22.14</td><td>73.74</td><td>69.52</td><td>52.70</td><td>50.89</td><td>27.39</td><td>29.22</td><td>38.42</td><td>45.04</td><td>40.00</td><td>26.26</td><td>49.47</td><td>45.99</td><td>18.96</td><td>36.35</td><td>33.22</td><td>60.18</td></tr><tr><td>DROcc17</td><td>23.38</td><td>73.46</td><td>70.96</td><td>52.99</td><td>50.06</td><td>30.15</td><td>31.24</td><td>40.70</td><td>47.33</td><td>41.12</td><td>26.88</td><td>51.79</td><td>49.40</td><td>24.49</td><td>35.08</td><td>36.37</td><td>61.48</td></tr><tr><td>b.CLIP-ResNet</td><td>17.41</td><td>73.84</td><td>70.04</td><td>51.34</td><td>51.93</td><td>24.66</td><td>24.24</td><td>40.88</td><td>44.74</td><td>38.95</td><td>25.74</td><td>48.76</td><td>41.90</td><td>13.94</td><td>25.29</td><td>35.10</td><td>50.47</td></tr><tr><td>b.CLIP-ViT</td><td>24.33</td><td>74.67</td><td>70.90</td><td>54.76</td><td>55.14</td><td>31.07</td><td>33.69</td><td>43.24</td><td>51.76</td><td>42.52</td><td>29.97</td><td>55.53</td><td>51.22</td><td>31.16</td><td>40.24</td><td>38.66</td><td>64.54</td></tr><tr><td>b.BLIP2</td><td>29.49</td><td>74.22</td><td>70.33</td><td>55.64</td><td>57.99</td><td>34.51</td><td>40.33</td><td>50.12</td><td>58.70</td><td>43.75</td><td>31.81</td><td>58.87</td><td>54.29</td><td>38.20</td><td>41.10</td><td>43.13</td><td>66.43</td></tr><tr><td>b.DINOv2</td><td>30.55</td><td>75.03</td><td>71.26</td><td>57.10</td><td>57.98</td><td>34.95</td><td>41.09</td><td>47.75</td><td>58.59</td><td>43.72</td><td>34.40</td><td>62.83</td><td>55.70</td><td>39.80</td><td>40.46</td><td>37.00</td><td>67.52</td></tr></table>

## 4.2 Implementation Details

All variants use the same multi-view RGB-D setting. During training, ten RGB-D views are sampled from each scene. RGB images are resized to 480 × 480 and normalized before being processed by the image backbone and FPN. The point branch uses Minkowski ResNet-34.<sup>20</sup> The fusion neck receives a 256-channel image volume and a 512-channel depth volume after projection and sparse-to-dense conversion.

All models are trained for 24 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ and weight decay of $1 \times 1 0 ^ { - 2 }$ . Training uses four GPUs with one sample per GPU. The same optimizer, schedule, data pipeline, loss functions, and evaluator are used for all backbone variants.

## 4.3 Main Quantitative Results

Table 1 gives the main comparison. With all downstream 3D components fixed, DINOv2 reaches 30.55% mIoU and BLIP2 reaches 29.49%. CLIP-ViT is lower at 24.33%, but it still improves over CLIP-ResNet by 6.92 points. The spread from CLIP-ResNet to DINOv2 is 13.14 points, which is too large to treat the image encoder as a minor setting. Because the point branch, fusion module, losses, and schedule are unchanged, the result points to the transferred 2D representation itself.

Table 1 shows that the 2D backbone is a large source of variation. Our reproduced EmbodiedScan baseline reaches 21.20% mIoU, close to but slightly above the reported 19.97%. DROcc reaches 23.38%, and DROcc(SwinU) reaches 22.14%. In the same benchmark setting, replacing only the image backbone gives 24.33% with CLIP-ViT, 29.49% with BLIP2, and 30.55% with DINOv2. Thus, the improvement from a stronger image backbone is larger than the gain from several occupancy-specific module changes represented by these reference rows.

The comparison is not uniformly positive. CLIP-ResNet falls to 17.41%, below both the reported and reproduced EmbodiedScan baselines. This negative case is useful: it shows that pretrained semantic alignment by itself is not enough. The features must also remain spatially useful after projection, view aggregation, and voxel-level decoding. CLIP-ViT improves every common category over CLIP-ResNet, indicating that dense patch tokens are easier to reuse than the tested ResNet-style CLIP hierarchy. BLIP2 and DINOv2 then give a much larger jump, with DINOv2 producing the best overall mIoU. This does not change the focus of the paper into a decoder comparison; instead, it clarifies that backbone selection alone can produce large performance shifts.

## 4.4 Class-Wise Analysis

DINOv2 gives the best IoU on 10 of the 16 commonly reported categories: empty, floor, wall, cabinet, door, window, bed, curtain, refrigerator, and toilet. Many of these categories involve room layout, broad surfaces, or stable spatial boundaries. This pattern matches the expectation that DINOv2 features preserve local visual structure after voxel lifting.

BLIP2 has a diferent profile. It is best on chair, table, couch, shelf, plant, and stairs. These categories are more object-centered, and their appearance can be more important than large-scale layout. The result therefore does not reduce to a single ranking of backbones: DINOv2 is steadier for structural categories, while BLIP2 remains competitive for object-level semantics. CLIP-ViT lies between CLIP-ResNet and the two strongest variants, confirming that tokenized visual features help but do not fully close the gap to BLIP2 or DINOv2.

Table 2: IoU on the 15 least frequent classes used for long-tail analysis. Values are percentages. “exer.” denotes exercise equipment. These categories follow the rare-class grouping used in DROcc.<sup>17</sup>
<table><tr><td>Method</td><td>Average</td><td>roof</td><td>beam</td><td>frame</td><td>bicycle</td><td>exer.</td><td>microwave</td><td>printer</td><td>oven</td><td>mailbox</td><td>washbasin</td><td>partition</td><td>piano</td><td>countertop</td><td>drawer</td><td>carpet</td></tr><tr><td>Baseline (rep.)</td><td>7.12</td><td>0.00</td><td>0.00</td><td>0.00</td><td>20.98</td><td>0.65</td><td>16.94</td><td>40.00</td><td>13.08</td><td>0.16</td><td>0.62</td><td>2.54</td><td>5.23</td><td>0.33</td><td>1.25</td><td>4.95</td></tr><tr><td>DROcc</td><td>9.66</td><td>0.00</td><td>0.00</td><td>0.00</td><td>23.58</td><td>2.72</td><td>23.02</td><td>32.70</td><td>8.43</td><td>12.41</td><td>0.70</td><td>13.00</td><td>11.38</td><td>8.12</td><td>1.53</td><td>7.30</td></tr><tr><td>b.CLIP-ResNet</td><td>3.44</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>13.98</td><td>14.72</td><td>7.21</td><td>0.00</td><td>1.09</td><td>0.00</td><td>4.55</td><td>1.19</td><td>0.75</td><td>8.04</td></tr><tr><td>b.CLIP-ViT</td><td>9.86</td><td>0.00</td><td>0.00</td><td>0.00</td><td>28.30</td><td>15.41</td><td>22.19</td><td>32.05</td><td>8.90</td><td>0.00</td><td>1.35</td><td>0.27</td><td>28.03</td><td>9.20</td><td>2.15</td><td>0.00</td></tr><tr><td>b.BLIP2</td><td>15.82</td><td>0.00</td><td>0.00</td><td>0.74</td><td>52.16</td><td>32.67</td><td>30.06</td><td>43.14</td><td>12.45</td><td>1.83</td><td>4.55</td><td>3.18</td><td>50.39</td><td>4.72</td><td>1.29</td><td>0.13</td></tr><tr><td>b.DINOv2</td><td>17.28</td><td>0.00</td><td>0.00</td><td>0.00</td><td>49.00</td><td>19.03</td><td>39.46</td><td>41.73</td><td>20.35</td><td>5.82</td><td>2.25</td><td>22.96</td><td>42.76</td><td>7.91</td><td>2.08</td><td>5.91</td></tr></table>

Table 3: Resource comparison. Training time is reported in hours, memory in GiB, inference time in seconds per instance, and parameters in millions.

<table><tr><td>Method</td><td>Train time</td><td>Train mem.</td><td>Test time</td><td>Test mem.</td><td>Params</td></tr><tr><td>Baseline</td><td>38.03</td><td>19.86</td><td>2.65</td><td>7.15</td><td>751.28</td></tr><tr><td>DROcc</td><td>33.83</td><td>6.74</td><td>2.59</td><td>2.73</td><td>114.38</td></tr><tr><td>b.CLIP-ResNet</td><td>40.06</td><td>19.18</td><td>2.82</td><td>7.39</td><td>751.28</td></tr><tr><td>b.CLIP-ViT</td><td>35.98</td><td>19.73</td><td>2.70</td><td>7.83</td><td>787.98</td></tr><tr><td>b.BLIP2</td><td>49.90</td><td>21.71</td><td>3.75</td><td>13.52</td><td>1718.77</td></tr><tr><td>b.DINOv2</td><td>39.02</td><td>18.84</td><td>3.11</td><td>10.63</td><td>1034.62</td></tr></table>

## 4.5 Long-Tail Category Discussion

Table 2 shifts the analysis from common categories to the 15 least frequent classes. The first observation is that some categories remain nearly unsolved: roof and beam are zero for every method, and frame is almost always zero. These failures are not caused by a single backbone. They combine three factors: severe data imbalance, weak geometric observability from ego-centric RGB-D views, and limited feature resolution after projection. Roof and beam are structural labels that are often only partially visible or outside the dominant camera frustum, so stronger 2D descriptors cannot reliably supply evidence for them. Frame is usually thin and boundary-like, making it sensitive to voxel resolution and bilinear sampling. This suggests a category-specific transition point: for these labels, the 2D encoder is no longer the only bottleneck, and further gains likely require visibility-aware lifting, rare-class supervision, or higher-resolution geometric modeling. The second observation is that stronger backbones still help many rare objects. BLIP2 is strongest on bicycle, exercise equipment, printer, washbasin, and piano. DINOv2 is strongest on microwave, oven, and partition, and is close to BLIP2 on bicycle and printer. CLIP-ViT gives large gains over CLIP-ResNet on bicycle, exercise equipment, piano, countertop, and drawer. However, the table also shows that rare-class improvement is uneven: a method can be strong on object-like rare classes while remaining weak on structural rare labels. Backbone replacement therefore improves long-tail representation, but additional sampling, loss reweighting, or rare-class modeling is still needed.

## 4.6 Eficiency and Model Cost

Table 3 gives the resource side of the comparison. BLIP2 has the largest cost, with 1718.77M parameters, 49.90 training hours, and 13.52 GiB test memory. DINOv2 is lighter than BLIP2 but still substantially larger than the baseline in parameter count. CLIP-ViT has a modest parameter increase over the baseline and nearly the same runtime profile. DROcc is much smaller and cheaper than the baseline because it redesigns the fusion architecture, while our backbone variants keep the baseline 3D pipeline and only alter the image encoder. These numbers clarify the trade-of: DINOv2 and BLIP2 bring the strongest accuracy, but they also increase model size and, for BLIP2 especially, inference memory.

## 4.7 Qualitative Results

Figure 2 shows the same trend qualitatively. CLIP-ResNet produces fragmented regions and unstable foreground objects, matching its low mIoU. CLIP-ViT fills in more of the foreground and is less noisy, although it still misses some category distinctions. BLIP2 often recovers more object responses than the CLIP variants, but it can also leave scattered predictions near scene boundaries. DINOv2 gives cleaner room structure in several examples, especially on large surfaces and frequently observed objects.

![](images/255e7f27b5da017446feea505dab6aaedb95bb2c8124893c49b019aace0d5972.jpg)

![](images/7171ec15cf271a39860bb9c49343476facfcb342ddddd4d89d54ce8c84fe4c93.jpg)

![](images/2015689f865da74d1e654b00f874b65f5590e575a58ae018067c64385e5c174c.jpg)

![](images/e131204b49dab128310f45990b197179b8986c318e249207f5aa041b2459a211.jpg)

![](images/a21e95c29f089e70a19f02e8ddde2d2295a73d9e8a2f097b47dae501df18ed36.jpg)

![](images/eb70971be960a59002ea78fa3adf42604fb5b10992476ab6e678de8e58e7f85e.jpg)

![](images/bdc2e0c8b172671de734e1b12fe9d49d400d750c2705ef8b2a923562248b90d3.jpg)

![](images/2c00a190f146afc5d0c3fdc5d56bb8be6fec137b304488cbf0223b562f0c890c.jpg)

![](images/a77e4f6ab7c13d66b1935be4a1d1c5247a1c118af2956e2b242c3784a8de9eaf.jpg)

![](images/75f3c77f60dba10e6421c1d80b29e76d0a9c9c39383edeeb3851607085eb1f46.jpg)

![](images/3fac29d9b242aa8d6c178907102236e6795e95e299405a8f22b7115306932421.jpg)

![](images/8bb93c1239173c968f0f0449f99ac5a430cdf0c8bad084a2f32d3ca1677ae7d2.jpg)

![](images/73f171a9c117c8cab6fa17259bfd6404e40207a9a4ab1acf234a1ea444e6c344.jpg)

![](images/0084f8f200a62cc71add26f7cca3e4c824eb2e77be5fd87a2b49104927d90a88.jpg)

![](images/274e92de900ec1254a25d3b1e07b0243efe7044d9a9e9a0ec890b897a09c8730.jpg)

![](images/7fd432c9d1444959a4aaccc8636a3685fea561725d85e5c7eb962ac09a62b7b3.jpg)

![](images/c16265a0a8ce4265d9dcc4d7542c43aebbf3f776d453fc5cbaecf1c376bb4258.jpg)

![](images/fa8b61c373838297ac1c64c20d5e12ef935ba97d0605beaa8991e9e055c48b4b.jpg)

![](images/c70c4072f4aec74f8ad68f06d7581454d2e4af78e87eca1c1f9a2fb80a4dc414.jpg)

![](images/121fdd4c4ff66228c617be156b1f3619d5bef30dba98f02d78a0c3f28b0cca47.jpg)

![](images/3fb0a0498a634e359c22efa430375325e9f9cf458f04ff066e87ec7469681e65.jpg)

![](images/6b2c62d6cdacd90df28dde2064fca79fd569e0675583b6023957ee9b069eacb2.jpg)

![](images/a99709cce6aecc3c3e24e5e70f3e3df4a310fa66f8d5d2f7167453201e4a6bbb.jpg)

![](images/373e8898a07d720d30cd15de3acd9fb9e7b532c73214c2eebdc1afc057c48cc3.jpg)

![](images/f1ca25058e9731014b4f727cea493ee6b06244f531c235c7d124d49bff1fb892.jpg)

![](images/ece4c2075172eba5b2df2da7a099b8ba24293920a5c259ddc3db3fba50ffd0ce.jpg)

(a) Scene  
![](images/36fae7e03faa823588740787b9623cdb16cad3117a29aad79c482f62f2d96e29.jpg)

![](images/3ee68d5ebbbfcc7beb7ff656bee8568ed2fbb6fff5253a9ec8edb5ce15de9b20.jpg)

![](images/f7f625952ce29587441cc4e50a55330bbd679da20ec6e4374902bace2bf85e15.jpg)

![](images/f5246c6f173754d15805ce5cfb1b9ce47a272991ae4d3d506245a1988c5b9a2b.jpg)  
(b) GT

![](images/caaa6a20fc6b4350a73696ba8697121c564a32b315acd6e3d3f4f96afc33cd1d.jpg)

![](images/854939c310354465137176b2c07c526df0746db04110d1ef9f576f4c703f30e3.jpg)

![](images/1de38f2f09c1a12abe3cc5667b749454f58177d13a3ff0199f54d4b1130cdcb5.jpg)

![](images/8bdbd0f1fa97e5987f23e107195af7ff5d00aa3159318c98f0a79ce526494b4b.jpg)  
(c) ResNet-50  
(d) DROcc

![](images/9b2980808fb4955abb5913533d21093e2fbd4320a1d534cebc085a6e07308ed9.jpg)

(e) CLIP-Res.  
![](images/0e3cadb88eda491a509f8a40d5a18e95311e864183d3a9b5b880282d112da127.jpg)

![](images/29f20110b41bc56d5fe0f76b35fc78c8ab83def5b1d3fb1e9a53a6d29f687fcc.jpg)  
2:  
(f) CLIP-ViT

![](images/7336bc54e6ac8eb21b4b97880b21b3e1a4c1b5f2b5c9e66e8a60542ef81d4044.jpg)  
(g) BLIP2

![](images/c698c809aa63eefd305fd504a8f8cc818268d32c59ae5cb76eaba933da4346bc.jpg)

![](images/c171f27a81375cf3dfb6306741f84dcefacb15ee12d84e9654a0062e37a8ec94.jpg)  
(h) DINOv2

Figure 2: Qualitative comparison of semantic occupancy predictions. Columns show indoor scene input, ground truth, EmbodiedScan baseline, DROcc, CLIP-ResNet, CLIP-ViT, BLIP2, and DINOv2. CLIP-ResNet produces noisier and less complete object layouts, CLIP-ViT recovers more coherent foreground regions, and BLIP2 and DINOv2 recover richer semantic occupancy structures. DINOv2 is generally closer to the ground-truth room structure, whereas BLIP2 often preserves strong object-level responses.

These examples also explain why image-level semantic strength is not the whole story. After camera projection and view averaging, a useful feature must remain locally aligned and semantically separable. DINOv2 and BLIP2 meet this requirement better than the two CLIP variants, but not in the same way: DINOv2 looks more stable for layout and dense surfaces, whereas BLIP2 is still competitive for object-centered responses. CLIP-ViT confirms that dense token features help CLIP transfer, while its remaining gap shows that architecture and pretraining objective have to be considered together.

## 4.8 Discussion

The experiments lead to three conclusions. First, the image backbone should be treated as a first-order experimental variable in RGB-D occupancy prediction. A change in the 2D encoder alone can produce more than 13 mIoU points of diference under the same 3D pipeline. Second, DINOv2 is the strongest backbone in our controlled study, suggesting that self-supervised dense visual representations are well aligned with projectionbased voxel prediction. Third, vision-language representations remain useful but require careful interpretation: BLIP2 is close to DINOv2 overall and wins several object-centric categories, CLIP-ViT substantially improves over CLIP-ResNet, and the tested CLIP-ResNet variant is much weaker for dense semantic occupancy.

## 5. CONCLUSION

This paper studies the influence of image backbones on EmbodiedScan semantic occupancy prediction. Under a controlled RGB-D pipeline, we replace only the 2D image encoder and keep the depth branch, voxel projection, fusion neck, occupancy head, losses, and training protocol fixed. The results show that backbone choice has a large impact: DINOv2 achieves the best mIoU of 30.55%, BLIP2 reaches 29.49%, CLIP-ViT reaches 24.33%, and CLIP-ResNet reaches 17.41%. Compared with the original EmbodiedScan baseline, our reproduced baseline, DROcc(SwinU), and DROcc, the stronger backbone variants show that a front-end representation change can exceed several task-specific architectural improvements. The weaker CLIP-ResNet result also shows that pretraining alone does not guarantee better dense occupancy features.

The category-level analysis gives a more detailed conclusion. DINOv2 is strong on many layout and structural classes, while BLIP2 remains competitive on several object-centered classes. CLIP-ViT improves over CLIP-ResNet throughout the common-category table, demonstrating that dense token features matter within CLIPstyle pretraining. For the 15 least frequent categories, BLIP2 and DINOv2 recover several rare objects more efectively, but roof, beam, and frame remain nearly unsolved. This does not weaken the backbone finding; rather, it marks where backbone improvement has category-specific diminishing returns. When labels are extremely rare, weakly observed, or thinner than the efective voxel/image sampling resolution, the limiting factor shifts from visual representation quality to visibility modeling, supervision balance, and geometric resolution. These findings indicate that image backbone selection should be explicitly reported in embodied occupancy studies. Future work should combine stronger visual encoders with visibility-aware lifting, long-tail learning, improved RGB-D fusion, and decoders designed for sparse indoor objects.

## ACKNOWLEDGMENTS

We used OpenAI ChatGPT for language polishing, grammar refinement, LaTeX formatting verification, and manuscript self-checking against the SPIE submission guidelines. The tool was not used to generate experimental data, scientific claims, or research conclusions. All technical content, methodology, results, and interpretations were reviewed and verified by us.

Example prompts used with the tool included:

• “Polish the following paragraph for academic writing style while preserving the original technical meaning and terminology.”

• “Check whether this manuscript section complies with SPIE formatting and submission requirements.”

• “Review the following LaTeX code and identify possible compilation, formatting, or reference issues.”

• “Identify grammatical errors, inconsistent notation, or unclear expressions in the following manuscript text.”

We take full responsibility for the accuracy, originality, and integrity of the submitted work.

## REFERENCES

[1] Song, S., Lichtenberg, S. P., and Xiao, J., “Sun rgb-d: A rgb-d scene understanding benchmark suite,” in [Proceedings of the IEEE conference on computer vision and pattern recognition], 567–576 (2015).

[2] Huang, Y., Zheng, W., Zhang, Y., Zhou, J., and Lu, J., “Tri-perspective view for vision-based 3d semantic occupancy prediction,” in [Proceedings of the IEEE/CVF conference on computer vision and pattern recognition], 9223–9232 (2023).

[3] Li, Y., Yu, Z., Choy, C., Xiao, C., Alvarez, J. M., Fidler, S., Feng, C., and Anandkumar, A., “Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion,” in [Proceedings of the IEEE/CVF conference on computer vision and pattern recognition], 9087–9098 (2023).

[4] Huang, Y., Zheng, W., Zhang, B., Zhou, J., and Lu, J., “Selfocc: Self-supervised vision-based 3d occupancy prediction,” in [Proceedings of the IEEE/CVF conference on computer vision and pattern recognition], 19946– 19956 (2024).

[5] Zhang, Z., Gao, B., Ye, J., Jin, H., Jiang, L., and Yang, W., “Clip prior-guided 3d open-vocabulary occupancy prediction,” Pattern Recognition 162, 111347 (2025).

[6] Wang, T., Mao, X., Zhu, C., Xu, R., Lyu, R., Li, P., Chen, X., Zhang, W., Chen, K., Xue, T., et al., “Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition], 19757–19767 (2024).

[7] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., “Learning transferable visual models from natural language supervision,” in [International conference on machine learning], 8748–8763, PmLR (2021).

[8] Li, J., Li, D., Savarese, S., and Hoi, S., “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in [International conference on machine learning], 19730–19742, PMLR (2023).

[9] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193 (2023).

[10] Chen, D., Zheng, H., Zhou, Y., Li, X., Liao, W., He, T., Peng, P., and Shen, J., “Semantic causality-aware vision-based 3d occupancy prediction,” 24878–24888 (2025).

[11] Huang, H., Sun, H., Liu, N., Zhou, H., and Shen, J., “Slicesemocc: Vertical slice–based multimodal 3d semantic occupancy representation,” 119–132 (2025).

[12] Hayes, S., Sistu, G., and Eising, C., “Easyocc: 3d pseudo-label supervision for fully self-supervised semantic occupancy prediction models,” arXiv preprint arXiv:2509.26087 (2025).

[13] Jin, B., Gu, S., Hu, X., Zheng, Y., Guo, X., Zhang, Q., Long, X., and Yin, W., “Occtens: 3d occupancy world model via temporal next-scale prediction,” IEEE Robotics and Automation Letters (2026).

[14] Liu, R., Kong, L., Li, D., and Zhao, H., “Occvla: Vision-language-action model with implicit 3d occupancy supervision,” arXiv preprint arXiv:2509.05578 (2025).

[15] Boeder, S., Gigengack, F., Roesler, S., Caesar, H., and Risse, B., “Shelfocc: Native 3d supervision beyond lidar for vision-based occupancy estimation,” arXiv preprint arXiv:2511.15396 (2025).

[16] Cao, A.-Q. and Vu, T.-H., “Occany: Generalized unconstrained urban 3d occupancy,” (2026).

[17] Fang, S., Zheng, Q., and Wang, X., “A dual residual transformer architecture for indoor semantic occupancy prediction,” Pattern Recognition , 113667 (2026).

[18] He, K., Zhang, X., Ren, S., and Sun, J., “Deep residual learning for image recognition,” in [Proceedings of the IEEE conference on computer vision and pattern recognition], 770–778 (2016).

[19] Lin, T.-Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S., “Feature pyramid networks for object detection,” in [Proceedings of the IEEE conference on computer vision and pattern recognition], 2117–2125 (2017).

[20] Choy, C., Gwak, J., and Savarese, S., “4d spatio-temporal convnets: Minkowski convolutional neural networks,” in [Proceedings of the IEEE/CVF conference on computer vision and pattern recognition], 3075– 3084 (2019).