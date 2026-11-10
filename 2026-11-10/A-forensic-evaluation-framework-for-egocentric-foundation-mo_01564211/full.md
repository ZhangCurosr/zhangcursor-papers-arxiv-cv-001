# A forensic evaluation framework for egocentric foundation models: an EFM3D case study for crime scene reconstruction

Davidson Chisom Onwubiko<sup>a</sup>, Shahana Bano<sup>a</sup>, and Carlos Francisco Moreno-Garc´ıa<sup>a</sup>

<sup>a</sup>School of Computing, Engineering & Technology, Robert Gordon University, Aberdeen, United Kingdom

## ABSTRACT

Foundation models increasingly combine geometry, semantics, and spatial context for three-dimensional scene understanding, but their suitability for forensic crime scene reconstruction remains unclear. This study presents a structured forensic evaluation framework for egocentric foundation models and demonstrates it using the Egocentric Foundation Model for 3D (EFM3D). Four simulated indoor crime scenes were recorded with Meta Project Aria Gen 1 at Robert Gordon University’s East Lodge House forensic training facility. The framework evaluates room-scale reconstruction, semantic scene understanding, computational performance, and forensic suitability. Across the four scenes, EFM3D generated meshes averaging 12,668 ± 2,430 vertices and 24,838 ± 4,903 faces, with 39,529 recorded semantic predictions spanning 21–24 categories at a mean confidence of 0.40. Because reference meshes and expert-labelled 3D ground truth were unavailable, these statistics are treated as descriptive outputs rather than measures of geometric or detection accuracy. Final inference runs processed 1,388 ± 203 snippets per scene at 0.69 ± 0.01 samples/s and required 50.77 ± 6.12 minutes per recording. A seven-criterion capability profile indicated stronger support for room geometry, furniture localisation, scene context, and spatial relationships than for fine-scale forensic evidence. The results position EFM3D as a complementary tool for contextual documentation and scene review rather than a replacement for high-fidelity forensic capture. The framework provides a reproducible basis for evaluating future egocentric foundation models in specialised forensic settings.

Keywords: Foundation models, egocentric vision, machine vision, digital forensics, crime scene reconstruction, EFM3D, Project Aria, 3D scene understanding

## 1. INTRODUCTION

Foundation models are reshaping machine vision by bringing geometry, semantics, and spatial reasoning into increasingly unified representations. In contrast to conventional pipelines that treat reconstruction, localisation, and object recognition as separate stages, recent multimodal models can reason jointly about the structure and content of a scene. This shift is particularly relevant in applications where a useful representation must capture not only where objects are located, but also what they are and how they relate to the surrounding environment.

Digital forensic investigation is one such application. Crime scene reconstruction preserves spatial relationships after a physical scene has been released and supports later examination, communication, training, and courtroom presentation.<sup>1</sup> Photography, sketches, manual measurements, and total-station surveys remain important because they are established and defensible, yet they often leave investigators with fragmented records rather than a single explorable representation.<sup>1–4</sup> Three-dimensional capture has improved this process: photogrammetry, terrestrial laser scanning (TLS), and LiDAR can preserve geometry with high accuracy,<sup>3,</sup> <sup>5–7</sup> while Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) can produce photorealistic scene representations and flexible novel views.<sup>8–10</sup> Even so, these approaches remain primarily concerned with geometry and appearance. Semantic interpretation identifying objects and understanding their contextual role is commonly added later or performed manually<sup>11,</sup> <sup>12</sup> and remains an active area of methodological research in its own right, particularly with respect to loss-function design for segmentation accuracy.<sup>13</sup>

Wearable sensing ofers a diferent acquisition model. Meta Project Aria records synchronised visual, inertial, calibration, and other sensor streams from the wearer’s perspective.<sup>14</sup> Aria Machine Perception Services (MPS) convert these observations into trajectories, visual inertial odometry, and semi-dense geometry. EFM3D builds on this information through Egocentric Voxel Lifting (EVL), jointly predicting occupancy, object locations, and semantic context in three dimensions.<sup>15</sup> In principle, a natural investigator walkthrough can therefore support both scene reconstruction and contextual interpretation.

The unresolved question is whether capabilities learned from general indoor environments transfer to forensic scenes. EFM3D was developed and evaluated primarily on everyday activities, whereas crime scenes may contain disturbed layouts, restricted viewpoints, severe occlusion, reflective surfaces, and small evidence whose position and condition must be preserved accurately.<sup>1,</sup> <sup>15</sup> A model can therefore perform well at room-scale understanding while still failing on the details that matter most during evidential examination. Conventional machine-vision benchmarks do not fully capture this distinction. From a digital-forensics perspective, the resulting egocentric recordings, reconstructed geometry, semantic predictions, and spatial metadata form digital representations that can support subsequent scene examination, visualisation, and investigative review.

To address this gap, the present study introduces a structured forensic evaluation framework for egocentric foundation models and demonstrates it through an EFM3D case study using four simulated indoor crime scenes. The framework considers four complementary dimensions: room-scale reconstruction, semantic scene understanding, computational performance, and forensic suitability. Its purpose is not to replace established benchmark metrics, but to extend them with criteria that reflect practical forensic use.

The contribution of this work lies not in modifying EFM3D’s architecture, but in introducing and demonstrating a forensic-domain evaluation methodology that separates conventional model outputs from the requirements of forensic use. The main contributions are:

1. a structured forensic evaluation framework for assessing egocentric foundation models in crime scene reconstruction;

2. an EFM3D case study based on a four-room Project Aria dataset collected within a dedicated forensic training facility;

3. a domain-specific evaluation methodology that distinguishes descriptive model outputs from forensic accuracy and operational readiness; and

4. evidence-based recommendations for forensic datasets, evidence-aware semantics, hybrid reconstruction, and future operational validation.

The remainder of the paper reviews relevant reconstruction and egocentric-vision research, identifies the forensic evaluation gap, presents the framework and experimental methodology, reports the results, and discusses their implications and limitations.

## 2. RELATED WORK AND RESEARCH GAP

## 2.1 Digital Crime Scene Reconstruction

Digital reconstruction allows investigators to preserve spatial information after a physical scene has been released and supports measurement, review, and communication.<sup>1,</sup> <sup>2</sup> Conventional documentation remains indispensable, but its fragmented nature can make complex scenes dificult to revisit and interpret as a whole.<sup>2,</sup> <sup>3</sup> Photogrammetry, TLS, and LiDAR address this limitation by producing metrically useful 3D models,<sup>5–7</sup> while recent forensic platforms have begun integrating 3D reconstruction with analytical tools for scene and trajectory interpretation.<sup>16</sup> NeRF and 3DGS further improve visual realism and novel-view rendering,<sup>8–10,</sup> <sup>17</sup> with recent work extending 3DGS toward photorealistic immersive VR applications.<sup>18</sup>

These techniques difer in acquisition requirements and fidelity, yet most treat semantic understanding as a separate problem. A reconstruction may show where an object is located without identifying what it is or whether it has investigative relevance. Scene-labelling approaches can add semantics,<sup>12</sup> but forensic interpretation still depends heavily on expert review.<sup>11</sup> This creates a clear opportunity for systems that combine geometry with object-level reasoning.

## 2.2 Egocentric Wearable Sensing and Foundation Models

Egocentric devices capture a continuous first-person record as the wearer moves naturally through an environment. Project Aria combines RGB and monochrome scene cameras, inertial measurement units, eye cameras, microphones, and calibration data in a synchronised research platform.<sup>14</sup> MPS estimates trajectories and semi-dense geometry, providing the spatial basis for downstream perception.

EFM3D extends this pipeline toward learned 3D scene understanding. Its EVL component lifts image features into a shared voxel representation from which the model predicts occupancy, oriented 3D bounding boxes, and semantic categories.<sup>15</sup> The original EFM3D evaluation used Aria Digital Twin (ADT) and Aria Everyday Activities (AEA), which capture furnished apartments, ofices, and ordinary activities rather than staged forensic evidence.<sup>19,</sup> <sup>20</sup> This domain diference matters because models trained on everyday environments are naturally better represented for rooms, furniture, and common objects than for bloodstains, fingerprints, cartridge cases, fibres, or other evidence classes. Recent work on multimodal and geospatial forensic reconstruction further illustrates the need to preserve evidence-specific detail alongside scene-level geometry.<sup>4,</sup> <sup>21</sup>

## 2.3 Research Gap

Existing evaluations of egocentric foundation models emphasise reconstruction, detection, segmentation, and scene understanding within general indoor environments. Those benchmarks are valuable for measuring technica progress, but they do not ask whether outputs preserve evidential detail, provide meaningful forensic context, or remain practical within an investigative workflow. Forensic suitability therefore cannot be inferred from benchmark performance alone.

To our knowledge, structured evaluation of EFM3D under these forensic requirements remains limited. The present work addresses that gap by combining conventional machine-vision measurements with an explicit forensic suitability assessment. The resulting framework is intended as a reproducible evaluation methodology rather than a claim of universal forensic validation.

## 3. FORENSIC EVALUATION FRAMEWORK AND EXPERIMENTAL METHODOLOGY

## 3.1 Framework Motivation and Overview

A forensic reconstruction system must do more than recover visually plausible geometry. It should preserve the spatial arrangement of the scene, provide semantically useful context, operate within a practical processing workflow, and avoid overstating evidential detail that it cannot reliably represent. These requirements motivate a broader evaluation than geometry-only or detection-only benchmarks.

The proposed framework therefore organises assessment around four dimensions: room-scale reconstruction, semantic scene understanding, computational performance, and forensic suitability. Figure 1 shows how these dimensions connect the model outputs to an overall operational assessment. The framework is deliberately model-agnostic: EFM3D is used here as the case study, but the same dimensions can support future evaluations of other egocentric foundation models.

Table 1. Domain comparison between EFM3D benchmarks and the present dataset.
<table><tr><td>Characteristic</td><td> $\mathbf { A D T } ^ { 1 9 }$ </td><td> $\mathbf { A E A } ^ { 2 0 }$ </td><td>Forensic training facility</td></tr><tr><td>Environment</td><td>Furnished apartment and office</td><td></td><td>Everyday activities in real Four staged indoor crime-scene rooms</td></tr><tr><td>Sequences</td><td>200</td><td>homes 143</td><td>4</td></tr><tr><td>Volume</td><td>Sequence count reported</td><td>7.3 h; &gt;1M frames</td><td>563.7 s; 16,921 RGB frames</td></tr><tr><td>Object truth</td><td>ground 398 instances with 6DoF poses and None published segmentation</td><td></td><td>No manual ground truth; EFM3D outputs only</td></tr><tr><td>Semantic scope</td><td>Everyday household/office objects No released semantic labels</td><td></td><td>21-24 predicted everyday categories; no forensic classes</td></tr><tr><td>Scene intent</td><td>Natural domestic/office activity</td><td>Natural daily activity</td><td>Disturbed layouts and forensic walkthroughs</td></tr><tr><td>Evaluation focus</td><td>struction, pose</td><td>tric AI</td><td>Detection, segmentation, recon- Long-form multimodal egocen- Reconstruction, semantics, performance, and forensic suitability</td></tr></table>

![](images/6a5ccbe17fa0518a3c160ab3fbb5b5e287d358d66eca6503fa7fedd7e0f9ea6b.jpg)  
Figure 1. Proposed forensic evaluation framework for assessing egocentric foundation models. The framework evaluates an egocentric foundation model through four complementary dimensions: room-scale reconstruction, semantic scene understanding, computational performance, and forensic suitability, which together provide an overall operational assessment. In this study, the framework is demonstrated using EFM3D as the case study.

## 3.2 Objectives and Research Questions

The framework is demonstrated using the oficial EFM3D pipeline and four simulated indoor crime scenes. Table 1 contrasts the forensic training-facility data with the principal datasets originally used to evaluate EFM3D.

The four research questions correspond directly to the framework dimensions:

RQ1 What room-scale reconstruction characteristics does EFM3D exhibit within simulated forensic environments?

RQ2 What semantic object predictions does EFM3D produce in simulated forensic environments, and what limitations are evident for forensic scene understanding?

RQ3 What computational characteristics are observed when processing Project Aria recordings?

RQ4 To what extent are the generated outputs suitable for forensic crime scene documentation and interpretation?

## 3.3 Crime Scene Dataset and Acquisition

Data were collected at East Lodge House, Robert Gordon University’s forensic training facility. Four furnished rooms with diferent layouts and object distributions were treated as independent simulated crime scenes. The available dataset records room-level capture characteristics but does not contain a formal scene-by-scene inventory of staged evidence; consequently, no additional scene composition or evidence-count claims are made here. Each recording followed the same broad protocol: the wearer moved naturally around the room, covered the principal viewpoints, and avoided abrupt head movements that could disrupt visual–inertial tracking.

![](images/f8dbf580a31fc79e5b3be2ca141f5636326865aecdec82c3f1c4eaf7f46d07da.jpg)  
Figure 2. East Lodge House forensic training facility used for data collection.

Project Aria Gen 1 was operated with Recording Profile 28, which provides the synchronised streams required by MPS and EFM3D.<sup>14,</sup> <sup>15</sup> The four VRS recordings totalled 563.69 s, 16,921 RGB frames, and 8.642 GB. Their characteristics are reported in Table 2; the sensor profile is summarised in Table 3.

Table 2. Forensic training facility evaluation dataset.
<table><tr><td>Metric</td><td>Room 1</td><td>Room 2</td><td>Room 3</td><td>Room 4 Mean</td><td></td></tr><tr><td>Duration (s)</td><td>112.61</td><td>157.54</td><td>153.39</td><td>140.15</td><td>140.92</td></tr><tr><td>VRS size (GB)</td><td>1.314</td><td>1.962</td><td>1.762</td><td>1.604</td><td>1.661</td></tr><tr><td>RGB frames</td><td>3,380</td><td>4,729</td><td>4,605</td><td>4,207</td><td>4,230</td></tr><tr><td>RGB rate (fps)</td><td>30.02</td><td>30.02</td><td>30.02</td><td>30.02</td><td>30.02</td></tr></table>

Table 3. Project Aria Recording Profile 28.
<table><tr><td>Stream</td><td>Resolution</td><td>Rate</td><td>Format</td></tr><tr><td>RGB camera</td><td>1408 × 1408 px</td><td>30 fps</td><td>JPEG</td></tr><tr><td>SLAM cameras (×2)</td><td> $6 4 0 \times 4 8 0 ~ \mathrm { p x }$ </td><td>30 fps</td><td>JPEG</td></tr><tr><td>Eye cameras (×2)</td><td> $3 2 0 \times 2 4 0 \ { \overset { \cdot } { \operatorname { p x } } }$ </td><td>60 fps</td><td>JPEG</td></tr><tr><td>IMU 1 / IMU 2</td><td></td><td>1000  / 800 Hz</td><td></td></tr><tr><td>Magnetometer / barometer Microphones (×7)</td><td></td><td>10  / 50 Hz 48 kHz</td><td></td></tr></table>

## 3.4 Experimental Platform

The experiments used the oficial Meta EFM3D implementation without algorithmic modification. Processing was performed in Google Colaboratory on Ubuntu 24.04 LTS using Python 3.12, PyTorch 2.2.2, Open3D 0.19, CUDA 12.8, and an NVIDIA A100-SXM4-40GB GPU. Both oficial EFM3D checkpoints were included in the

released checkpoint package; however, the inference workflow reported in this study used model lite.pth. Project Aria recordings were pre-processed using Aria Machine Perception Services (MPS) before EFM3D inference.

Table 4. Experimental hardware and software configuration.
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Wearable sensor</td><td>Meta Project Aria Gen 1</td></tr><tr><td>Recording profile</td><td>Profile 28</td></tr><tr><td>Operating system</td><td>Ubuntu 24.04 LTS</td></tr><tr><td>Development environment</td><td>Google Colaboratory</td></tr><tr><td>Programming language</td><td>Python 3.12</td></tr><tr><td>GPU</td><td>NVIDIA A100-SXM4-40GB</td></tr><tr><td>CUDA</td><td>12.8</td></tr><tr><td>Deep learning framework</td><td>PyTorch 2.2.2</td></tr><tr><td>3D processing library</td><td>Open3D 0.19</td></tr><tr><td>Scene understanding</td><td>EFM3D (official Meta implementation)</td></tr><tr><td>Inference checkpoint</td><td>model_lite.pth</td></tr><tr><td>Perception pipeline</td><td>Aria Machine Perception Services (MPS)</td></tr></table>

## 3.5 Processing Pipeline

The oficial Meta workflow was retained without architectural modification (Fig. 3). MPS first generated camera trajectories, visual–inertial odometry, and semi-dense geometry. EFM3D then applied EVL to form a 3D feature representation and produced a reconstructed mesh, oriented bounding boxes, and semantic visualisations.<sup>14,</sup> <sup>15</sup> These outputs were subsequently assessed using the four dimensions of the proposed framework.

![](images/9a4cf28db28753263d59ccc20b576ac27d7e4df68dc27e1253e72728234ca517.jpg)  
Figure 3. Project Aria acquisition, MPS pre-processing, EFM3D inference, and forensic evaluation pipeline.

## 3.6 Evaluation and Analysis

Quantitative analysis used recording duration, frame count, mesh statistics, semantic predictions, confidence scores, processing logs, throughput, and inference time. Qualitative assessment was used where the dataset lacked manually annotated forensic ground truth. Because no reference meshes or expert-labelled 3D object annotations were available, mAP, IoU, F1-score, and absolute reconstruction error could not be calculated. The analysis therefore does not claim benchmark-level detection accuracy; instead, it characterises consistency across scenes and evaluates practical scene-documentation value.

For the forensic-suitability dimension, seven operational criteria were examined using a categorical capability profile. Each criterion was classified as supported, partially supported, limited, or not supported on the basis of the observed outputs. These categories are descriptive rather than numerical: they make the assessment criteria explicit and expose relative strengths and limitations within this case study without implying a validated forensicreadiness score. No validated weighting scheme or multi-rater instrument currently exists for this application. Future work should establish expert-derived criteria and inter-rater reliability before a formal forensic-readiness instrument is considered.

## 4. EVALUATION RESULTS

## 4.1 RQ1: Room-Scale Reconstruction Characteristics

The first framework dimension characterises room-scale reconstruction behaviour across scenes with diferent layouts and object distributions. Table 5 reports descriptive reconstruction statistics. EFM3D generated a recognisable room-scale model for every recording, with meshes averaging 12,668±2,430 vertices and 24,838±4,903 faces. Room 3 produced the largest mesh and Room 1 the smallest. These counts describe output density and consistency only: without a reference mesh they cannot establish geometric accuracy, completeness, or superiority. The efective acquisition rate remained approximately 30 fps across all recordings, indicating stable capture.

Table 5. Reconstruction characteristics across the four simulated crime scenes.
<table><tr><td>Metric</td><td>R1</td><td>R2</td><td>R3</td><td>R4</td><td>Mean ± SD</td></tr><tr><td>Duration (s)</td><td>112.61</td><td>157.54</td><td>153.39</td><td>140.15</td><td>140.92 ± 20.30</td></tr><tr><td>RGB frames</td><td>3,380</td><td>4,729</td><td>4,605</td><td>4,207</td><td>4,230 ± 608</td></tr><tr><td>RGB rate (fps)</td><td>30.02</td><td>30.02</td><td>30.02</td><td>30.02</td><td>30.02 ± 0.00</td></tr><tr><td>Mesh vertices</td><td>10,485</td><td>12,203</td><td>16,235</td><td>11,750 </td><td>0 12,668 ± 2,430</td></tr><tr><td>Mesh faces</td><td></td><td></td><td></td><td></td><td>20,590 23,811 31,904 23,045 24,838 ± 4,903</td></tr></table>

Figures 4 and 5 show the corresponding RGB/SLAM views, reconstructed surfaces, oriented bounding boxes, and semantic predictions. Qualitative inspection showed visually coherent representations of the main room boundaries and furniture across the four scenes. Local surface discontinuities and noisy regions were nevertheless visible where coverage or visual texture was limited. Emerging point-cloud completion techniques, which infer and fill missing geometric detail from partial observations, may ofer a route to addressing such reconstruction gaps in future work.<sup>22</sup> These observations are consistent with sensitivity to scene coverage, acquisition trajectory, and available visual texture, although these factors were not independently manipulated in the present study.

![](images/cc2530538c11cccc431305c4f27bc8fd9ef9805bcd0bbf1d3211d7585e4317a8.jpg)  
Figure 4. Representative EFM3D outputs for Rooms 1 (left) and 2 (right).

![](images/5f86f8aa3266a90140f7c8590953d44e58d75a5b581a22b273cc7c7f5a955f18.jpg)  
Figure 5. Representative EFM3D outputs for Rooms 3 (left) and 4 (right).

## 4.2 RQ2: Semantic Scene Understanding

The second framework dimension characterises the semantic object predictions produced by EFM3D and their limitations for forensic scene understanding. Across the four scenes, EFM3D generated 39,529 recorded semantic predictions spanning 21–24 semantic categories (Table 6). Chairs, tables, cabinets, sofas, lamps, and other large indoor objects dominated the output.

Table 6. Semantic prediction summary.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Total recorded predictions</td><td>39,529</td></tr><tr><td>Mean predictions per scene</td><td>9,882</td></tr><tr><td>Categories per scene</td><td>21-24</td></tr><tr><td>Mean confidence</td><td>0.400</td></tr><tr><td>Mean predictions per RGB frame</td><td>2.34</td></tr><tr><td>Mean simultaneous predictions</td><td>7.14</td></tr><tr><td>Detection coverage</td><td>32.80%</td></tr></table>

The semantic outputs also expose a clear domain limitation. Qualitative inspection indicated that small objects were less consistently represented, while forensic categories such as fingerprints, bloodstains, and trace material were absent from the model vocabulary. The mean confidence of 0.40 further indicates uncertainty in some everyday-object predictions. These observations suggest greater utility for broad contextual interpretation than for fine-grained evidential analysis. Detection coverage was approximately 32.8% across all scenes; this value denotes the proportion of RGB frames with at least one recorded semantic detection and is not an intrinsic accuracy measure.

## 4.3 RQ3: Computational Performance

The third framework dimension examines whether the workflow is practical for ofline forensic processing. Table 7 reports the final inference measurements. EFM3D processed between 1,105 and 1,554 snippets per scene at a stable throughput of 0.67–0.70 samples/s. Mean inference time was 50.77 ± 6.12 minutes per recording. Runtime increased with the amount of data processed, but throughput remained tightly clustered across the four scenes, indicating stable computational behaviour under the final experimental configuration.

Table 7. Computational performance of EFM3D across the four recordings.
<table><tr><td>Metric</td><td>R1</td><td>R2</td><td>R3</td><td>R4</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td>Snippets processed</td><td>1,105</td><td>1,554</td><td>1,513</td><td>1,380</td><td> $1 , 3 8 8 \pm 2 0 3$ </td></tr><tr><td>Throughput (samples/s)</td><td>0.70</td><td>0.69</td><td>0.69</td><td>0.67</td><td> $0 . 6 9 \pm 0 . 0 1$ </td></tr><tr><td>Inference time (s)</td><td>2580.41</td><td>3408.28</td><td>3257.84</td><td>2938.72</td><td> $3 0 4 6 . 3 1 \pm 3 6 7 . 1 5$ </td></tr><tr><td>Inference time (min)</td><td>43.01</td><td>56.80</td><td>54.30</td><td>48.98</td><td> $5 0 . 7 7 \pm 6 . 1 2$ </td></tr><tr><td>VRS size (GB)</td><td>1.314</td><td>1.962</td><td>1.762</td><td>1.604</td><td> $1 . 6 6 1 \pm 0 . 2 7 8$ </td></tr></table>

The observed runtime is not suitable for real-time deployment, but real-time operation was not the objective of the present study. For the short recordings evaluated here, ofline inference required approximately 43–57 minutes per scene. Whether this processing time is operationally acceptable in forensic practice requires practitionercentred evaluation. The measurements should therefore be interpreted as hardware- and implementation-specific rather than as a general benchmark of EFM3D speed.

## 4.4 RQ4: Forensic Suitability

The fourth and final framework dimension relates the observed technical behaviour to practical forensic requirements. Table 8 presents a seven-criterion categorical capability profile. Within this case study, room geometry, furniture localisation, and spatial relationships were classified as supported; scene context as partially supported; small-evidence representation as limited; and bloodstain and fingerprint recognition as not supported. These classifications constitute an exploratory single-assessor assessment rather than a validated multi-rater instrument.

Table 8. Structured forensic suitability assessment using the categorical capability profile defined in Section 3.6.
<table><tr><td>Criterion</td><td>Capability</td><td>Basis in observed outputs</td></tr><tr><td>Room geometry reconstruction</td><td>Supported</td><td>Recognisable room-scale meshes in all four scenes</td></tr><tr><td>Furniture localisation</td><td>Supported</td><td>Common furniture predictions present across scenes</td></tr><tr><td>Scene context understanding</td><td>Partially supported</td><td>Broad indoor semantic context represented, with limitations</td></tr><tr><td>Spatial relationships</td><td>Supported</td><td>Room-scale geometry and object locations jointly</td></tr><tr><td>Small-evidence representation</td><td>Limited</td><td>represented Qualitative inspection indicated weak/inconsistent</td></tr><tr><td>Bloodstain recognition</td><td>Not supported</td><td>representation Forensic-specific class absent from model vocabu- lary</td></tr><tr><td>Fingerprint recognition</td><td>Not supported</td><td>Forensic-specific class absent from model vocabu- lary</td></tr></table>

The profile exposes a scale-dependent boundary in the observed outputs: room layout, furniture, and broad spatial context were represented more efectively than trace-level evidence. The classifications should therefore be read as a transparent case-study synthesis rather than as a validated measure of forensic readiness.

## 4.5 Relationship to Existing Reconstruction Approaches

EFM3D addresses a diferent part of the reconstruction workflow from geometry-first techniques. Photogrammetry, TLS/LiDAR, NeRF, and 3DGS primarily preserve geometry or appearance, whereas EFM3D combines wearable egocentric acquisition with learned 3D occupancy and semantic object predictions.<sup>2,</sup> <sup>3,</sup> <sup>5–10,</sup> <sup>15</sup> The methods were not benchmarked on a common reference scene in this study; consequently, no ranking of their geometry, speed, or evidence-preservation performance is made. The present results instead motivate a hybrid evaluation question: whether semantic egocentric scene understanding can complement established high-fidelity capture while preserving clear boundaries between model inference and evidential observation. Comparative experiments using common reference scenes and ground-truth geometry are required to test that proposition.

## 4.6 Summary of Findings

Application of the proposed framework shows that EFM3D produced interpretable room-scale reconstructions, common-object semantic predictions, and stable ofline computational behaviour across the four simulated scenes. The framework also exposes an important domain gap: forensic-specific and fine-scale evidence was not adequately represented in the observed outputs. These findings support further investigation of egocentric foundation models as complementary tools for contextual documentation and scene review, while leaving evidential accuracy and operational readiness to be established through ground-truth and practitioner-centred validation.

## 5. DISCUSSION

## 5.1 Interpretation of the Findings

The case study demonstrates that a general-purpose egocentric foundation model can transfer useful room-level knowledge to a forensic setting without architectural modification. Across all four recordings, EFM3D produced representations of the dominant indoor structure and predictions corresponding to common furniture, combining room-scale geometry with semantic context. These observed capabilities motivate further evaluation for tasks such as scene familiarisation, collaborative review, training, and preparation of immersive scene interfaces.

The results also show why good scene understanding should not be confused with evidential completeness. The observed pattern is consistent with EFM3D’s general indoor-domain training and evaluation context: furniture and common indoor objects were more readily represented, whereas forensic-specific and trace-level evidence was absent or under-resolved. A coherent room-scale mesh can therefore coexist with the loss of precisely the details that make a reconstruction evidentially useful. The distinction is central to forensic deployment.

Wearable acquisition is nevertheless attractive. A natural walkthrough can preserve a continuous first-person record without occupying the investigator’s hands or requiring a scanner to be repositioned at multiple stations. In time-sensitive settings, this may provide a rapid contextual record before targeted high-resolution evidence capture begins. The present findings therefore support investigating EFM3D as a complementary rather than substitutive component of forensic scene-documentation workflows.

## 5.2 Implications for Machine Vision

The findings have implications beyond digital forensics. Foundation models are commonly judged using general benchmarks that emphasise reconstruction, detection, segmentation, or localisation. Those measures remain essential, but they provide only a partial view of suitability once a system enters a specialised or safety-critical domain. Application-specific requirements can expose limitations that remain invisible in benchmark scores.

The proposed framework illustrates one way to extend evaluation without discarding conventional metrics. By combining technical measurements with explicit operational criteria, it separates model capability from domain readiness. Similar reasoning may be relevant wherever foundation models are expected to support decisions in environments where missing small details, misinterpreting context, or overstating confidence carries practical consequences.

## 5.3 Implications for Digital Forensics

For forensic practice, the immediate value of egocentric foundation models lies in contextual documentation rather than evidential replacement. EFM3D can help an examiner understand room layout, navigate a digital scene, and identify major objects, but high-fidelity photography, photogrammetry, TLS, LiDAR, and specialist evidence imaging remain necessary for precise measurement and detailed examination.<sup>2,</sup> <sup>3,</sup> <sup>5–7</sup>

A hybrid workflow is therefore more realistic: wearable egocentric capture provides rapid scene-level context, while targeted high-resolution methods preserve evidentially important regions. Future systems could combine these complementary representations within a common spatial frame, allowing semantic scene understanding to guide investigators toward areas that require more detailed capture without presenting inferred semantics as evidence.

## 5.4 Threats to Validity

Several factors constrain the interpretation of the study. First, the evaluation used four simulated indoor crime scenes from a single training facility. The scenes provide controlled forensic context, but they do not represent outdoor, industrial, public, or highly dynamic environments. Second, the study evaluates a single egocentric foundation model using the oficial Meta implementation without task-specific fine-tuning on forensic datasets. Although EFM3D represents one of the most advanced publicly available egocentric foundation models, alternative models or domain-adapted versions may demonstrate diferent reconstruction characteristics and semantic capabilities.

Third, the dataset does not contain manually annotated 3D forensic ground truth. This prevented calculation of mAP, IoU, pose error, completeness, or metric reconstruction accuracy and required the study to combine objective output statistics with structured qualitative assessment. Fourth, the forensic suitability ratings were produced as an exploratory assessment rather than through multiple independent practitioners. Inter-rater reliability and alternative weighting strategies therefore remain to be validated. The recordings were also produced using a single acquisition procedure, so the efects of operator behaviour, viewing trajectory, walking speed, viewing distance, and scene coverage were not independently evaluated. Finally, the study addresses technical suitability rather than chain of custody, evidential provenance, explainability, repeatability, legal admissibility, or end-user efectiveness.

These limitations do not invalidate the case study, but they define its scope. The proposed framework should be regarded as a structured starting point for forensic evaluation rather than a validated standard.

## 5.5 Directions for Future Work

Future development should focus on four priorities. First, larger multi-site egocentric datasets are needed with expert labels for evidence type, 3D position, visibility, and uncertainty. Second, domain adaptation should introduce evidence-aware categories and contextual relationships rather than relying solely on everyday-object vocabularies. Third, comparative studies should apply the same framework to photogrammetry, LiDAR, NeRF, 3DGS, and emerging foundation models using common reference scenes, complemented by systematic comparison of object detection architectures suited to evidence-aware semantic categories.<sup>23</sup> Finally, practitioner-centred validation should assess inter-rater reliability, uncertainty communication, provenance, and XR-based review with investigators.<sup>11</sup>

## 6. CONCLUSION

This study presented a structured forensic evaluation framework for egocentric foundation models and demonstrated it through an EFM3D case study using four Project Aria recordings from simulated indoor crime scenes. The framework brings together room-scale reconstruction, semantic scene understanding, computational performance, and forensic suitability so that technical outputs can be interpreted in relation to operational forensic requirements.

The case study showed that EFM3D produced interpretable room-scale geometry and predictions for major furniture while adding scene-level semantic context. Inference behaviour was stable across the four recordings, with an average runtime of 50.77 ± 6.12 minutes under the final experimental configuration. At the same time, the evaluation exposed a clear evidential boundary: fine-scale evidence and forensic-specific semantics were not adequately represented in the observed outputs for detailed evidential examination. Within the conditions evaluated here, the results support considering EFM3D primarily for contextual documentation and scene review rather than as a replacement for validated high-precision capture methods.

More broadly, the findings reinforce the need to evaluate foundation models in the operational settings for which they are intended. As these models move beyond general benchmarks into specialised and safety-critica domains, application-specific evaluation will be essential for distinguishing technical capability from practica readiness. The proposed forensic evaluation framework provides a structured basis for assessing future egocentric foundation models within crime scene reconstruction and for designing comparative evaluations with ground-truth geometry, evidence-level annotations, and practitioner assessment.

## REFERENCES

1. M. S. Balbudhe, B. S. K. Shetty, N. A. Kumar, D. Singh, and A. I. Sherekar, “Crime scene reconstruction: a scoping review,” Egyptian Journal of Forensic Sciences 15, p. 79, 2025.

2. G. Sansoni, C. Cattaneo, M. Trebeschi, D. Gibelli, P. Poppa, D. Porta, M. Maldarella, and M. Picozzi, “Sceneof-crime analysis by a 3-dimensional optical digitizer: a useful perspective for forensic science,” American Journal of Forensic Medicine and Pathology 32(3), pp. 280–286, 2011.

3. U. Buck, S. Naether, B. R¨ass, C. Jackowski, and M. J. Thali, “Accident or homicide—virtual crime scene reconstruction using 3d methods,” Forensic Science International 225(1–3), pp. 75–84, 2013.

4. L. Luchowski, D. Pojda, A. A. Tomaka, K. Skabek, and P. Kowalski, “Multimodal imagery in forensic incident scene documentation,” Sensors 21(4), p. 1407, 2021.

5. F. M. Sheshtar, W. M. Alhatlani, M. Moulden, and J. H. Kim, “Comparative analysis of LiDAR and photogrammetry for 3D crime scene reconstruction,” Applied Sciences 15(3), p. 1085, 2025.

6. A. N. Sazaly, M. F. Mohd Arif, and A. F. Razali, “3D indoor crime scene reconstruction from micro UAV photogrammetry technique,” Engineering, Technology & Applied Science Research 13(6), pp. 12020–12025, 2023.

7. G. Galanakis, X. Zabulis, T. Evdaimon, S.-E. Fikenscher, S. Allertseder, T. Tsikrika, and S. Vrochidis, “A study of 3D digitisation modalities for crime scene investigation,” Forensic Sciences 1(2), pp. 56–85, 2021.

8. B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision, pp. 405–421, 2020.

9. B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3D gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics 42(4), pp. 139:1–139:14, 2023.

10. S. Cho and T. Woo, “Accuracy of three-dimensional gaussian splatting for virtual crime scene reconstruction,” Frontiers in Computer Science 8, p. 1755361, 2026.

11. M. A. Maneli and O. E. Isafiade, “3d forensic crime scene reconstruction involving immersive technology: A systematic literature review,” IEEE Access 10, pp. 88821–88857, 2022.

12. S. Zhi, T. Laidlow, S. Leutenegger, and A. J. Davison, “In-place scene labelling and understanding with implicit scene representation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15818–15827, 2021.

13. W. Swastika, “Comparative analysis of loss functions for semantic segmentation: An empirical study on Cityscapes dataset,” Journal of Image and Graphics 14(3), pp. 444–453, 2026.

14. J. Engel, K. Somasundaram, M. Goesele, A. Sun, A. Gamino, A. Turner, et al., “Project aria: A new too for egocentric multi-modal AI research,” 2023. arXiv preprint arXiv:2308.13561.

15. J. Straub, D. DeTone, T. Shen, N. Yang, C. Sweeney, and R. Newcombe, “EFM3D: A benchmark for measuring progress towards 3D egocentric foundation models,” 2024. arXiv preprint arXiv:2406.10224.

16. A. Ospina-Boh´orquez, E. Ruiz de O˜na, R. Yali, E. Patsiouras, K. Margariti, and D. Gonz´alez-Aguilera, “Comprehensive forensic tool for crime scene and trafic accident 3D reconstruction,” Algorithms 18(11), p. 707, 2025.

17. D. Rangelov, S. Waanders, K. Waanders, M. van Keulen, and R. Miltchev, “Impact of data capture methods on 3D reconstruction with gaussian splatting,” Journal of Imaging 11(2), p. 65, 2025.

18. M. M. Hossain and S. Kim, “Gaussian splatting for photorealistic immersive virtual tours,” Journal of Image and Graphics 14(3), pp. 387–402, 2026.

19. X. Pan, N. Charron, Y. Yang, S. Peters, T. Whelan, C. Kong, O. Parkhi, R. Newcombe, and Y. Ren, “Aria digital twin: A new benchmark dataset for egocentric 3d machine perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20133–20143, 2023.

20. Z. Lv, N. Charron, P. Moulon, A. Gamino, C. Peng, C. Sweeney, et al., “Aria everyday activities dataset,” 2024. arXiv preprint arXiv:2402.13349.

21. A. F. Razali, K. L. Edward, M. F. Mohd Arif, N. S. Jasni, N. Darwin, and Z. Majid, “Innovative forensic features and blood spatter 3D reconstruction from geospatial-driven datasets,” Engineering, Technology & Applied Science Research 15(3), pp. 23930–23937, 2025.

22. U. Haroon, A. AlMughrabi, R. Marques, and P. Radeva, “VolE-Complete: Enhancing food 3d reconstruction and volume estimation with symmetry-guided point cloud completion,” Journal of Image and Graphics 14(3), pp. 454–461, 2026.

23. T. Harminto, D. F. Najib, P. Wicaksono, and Z. N. Izdihar, “Performance comparison of YOLOv8, YOLOv5 and YOLOv11 in nature image segmentation,” Journal of Image and Graphics 13(5), pp. 515–520, 2025.