# Autonomous 3D Visual-Thermal Inspection for Utility-Scale Solar Farms

Lachlan Tychsen-Smith, Mark Cox, Brett Wood, David Haddon, Kenrick Anderson, Benjamin Duck, Jim O’Malley, Noel W Duffy, Peyman Moghadam

CSIRO

firstname.lastname@csiro.au

## Abstract

Utility-scale photovoltaic (PV) inspection is dominated by aerial surveys, which are effective for identifying module-level hot cells but provide limited visibility of critical components located under the array, including connectors, cabling, fasteners and structural elements. These assets are difficult to inspect consistently at scale, and their defects are often spatially small, partially occluded and difficult to localise from isolated images. We present an automated 3D visual-thermal inspection system (Tricorder) for under-panel PV asset inspection. The approach combines LiDAR-inertial Simultaneous Localization and Mapping (SLAM) with geometrically calibrated RGB and thermal imagery and machine-learning-based object detection and segmentation. Detected assets are associated with the reconstructed 3D environment, allowing visual and thermal observations from multiple viewpoints to be spatially fused and localised. We demonstrate the approach on utility-scale PV data for MC4 connector detection and connection-state recognition, thermal anomaly identification, and detection of structural components including bolts, nuts and torque tubes. The results demonstrate a pathway from image-based inspection toward spatially grounded autonomous inspection, where defects are detected and linked to specific assets and their 3D locations within the solar farm.

## Problem Statement

Aerial RGB and thermal inspection provides an efficient means of surveying large PV fields and identifying module-level anomalies such as hotspots. However, many critical electrical and structural failure points, including MC4 connectors, cabling, fasteners, and mounting components, are located beneath the modules and are difficult to observe from above. Inspecting these components, therefore, remains dependent on labour-intensive hand-held inspection. The challenge is not only detecting these assets, but determining what component is being observed, its condition, and where it is located in 3D space. Conventional 2D thermography lacks explicit scene geometry and can be sensitive to viewpoint, occlusion, and thermal reflections. Prior 3D thermography systems demonstrated that combining 3D geometry with visual and thermal observations can improve localisation, repeatability, and multi-view thermal interpretation [1–3]. We extend this principle toward automated, machine-learning-driven inspection of utility-scale PV infrastructure.

## Autonomous 3D Visual-Thermal Inspection System

We formulate under-panel PV inspection as a spatially grounded multimodal perception problem. The objective is not only to detect an asset or anomaly in an individual image, but to associate visual and thermal evidence with a specific physical component and its 3D location in the solar farm, enabling candidate faults to be localised.

The proposed system combines four stages. First, LiDAR-inertial SLAM [4] estimates the sensor trajectory and reconstructs the surrounding three-dimensional environment. Second, geometrically calibrated RGB and thermal observations are aligned to this common spatial reference. Third, machinelearning models [5, 6] detect and segment electrical and structural assets within the RGB imagery. Finally, these detections and their associated thermal measurements are projected into the reconstructed 3D map, allowing asset condition and defect observations from multiple viewpoints to be associated with specific physical components.

## Tricorder Platform and Data Collection

The sensing platform (called Tricorder) integrates a Li-DAR sensor, inertial measurement unit (IMU), RGB camera, thermal infrared camera, real-time kinematic (RTK) Global Navigation Satellite System (GNSS) receiver, and an NVIDIA Jetson Xavier for onboard data capture and machine learning processing. The platform can be operated in either handheld or vehicle-mounted configurations, enabling rapid automated inspection of largescale solar installations. LiDAR, IMU, and GNSS measurements are used to generate a georeferenced threedimensional representation of the solar farm. The RGB and thermal cameras are geometrically calibrated with respect to the LiDAR coordinate frame, allowing any asset detections and thermal measurements to be accurately localised within a global coordinate system.

![](images/52ce10dd1fb413f89e486f25d9d1cd982572fd5fd48840de3464d2b3cd57ff5c.jpg)  
Figure 1 Tricorder: a multi-modal sensing platform.

## Field Deployment and Experimental Evaluation

Field data were collected at CSIRO’s Photovoltaic Outdoor Research Facility in Newcastle, and three commercial utility-scale PV solar farms around Australia. These deployments capture substantial variation in PV structure, component configuration, illumination, weathering, corrosion and background clutter, providing representative data for developing and evaluating under-panel asset detection and defect-identification methods.

A subset of the collected RGB imagery was manually annotated with electrical and structural asset classes, including MC4 connectors and their connection state, bolts, nuts, bolt holes, torque marks, torque tubes and cabling. These annotations were used to train the DeNeT deep-learning framework for object detection and semantic segmentation. We demonstrate this capability through two representative use cases: structural inspection for missing fasteners and electrical inspection of MC4 connectors.

## Structural Inspection: Missing Fasteners

Structural inspection is performed by detecting bolt, nut, and hole classes within the imagery using our machine learning pipeline. The inclusion of a hole class provides additional evidence of potentially missing fasteners. Missing components can be identified by comparison with either expected structural configurations or similar structures elsewhere within the solar field. Additional ground truth permits detecting other structural and electrical components e.g., the torque tube, distribution wires, and bolt torque marks. Example outputs are shown in Figure 2.

For this work, we collected a dataset including 5K images from 3 Australian solar/industrial sites and annotated 57K bounding boxes. In some cases, fasteners were removed to simulate an improper installation. The Site A solar farm was freshly constructed and provided very clean samples with minimal rust/weathering. Comparatively, the Site B and Site C were significantly older with high levels of corrosion and damage.

To quantify the accuracy of the object detector, we identified a train/test data split from the Site A/Site B datasets and plotted precision-recall curves (Figure 3) for each independently. Our results indicate which classes are easiest to detect (e.g., Bolt heads are easier to identify than torque marks) and, by comparing Site A to Site B, the effect of environmental conditions on these metrics (reduced precision and recall).

![](images/76798a39d450e99d0615f04869a458234d0547154591e32853d891235155d757.jpg)  
(a) Bolts and Nuts (Site A)

![](images/aa24118634d0b9de1e2cf4f93f3db06fab8056290305ddad76d2cd1b456e2be2.jpg)  
(b) Bolts and Nuts (Site B)

![](images/c23518570d0f0f72f9399e9d043356c6dcb26c691baadbcacbf37e5481b03720.jpg)  
(c) Bolts, Nuts, Torque Tube and Wires (Site C)

Figure 2 Example autonomous detection of bolts, nuts, torque tube and wires.  
![](images/63dee19f17ae5a748883a24343bb6b1f2b6aed8d4a642352a28ca16af205fad2.jpg)  
(a) Site A

![](images/bf965a310ffec0f6580fdaa2c1ae239967b9f7eebd4d563339e404e05c4d8b14.jpg)  
(b) Site B  
Figure 3 Precision–recall curves for the DeNeT bolt detector.

## Electrical Inspection: Faulty MC4 connectors via thermal signature

MC4 connector inspection combines RGB detection, semantic segmentation, and thermal analysis. Connectors are first detected and segmented in RGB imagery using our machine learning pipeline, after which the corresponding pixels are transferred into the thermal image using the calibrated multimodal geometry. This allows connector temperature to be estimated from the segmented component region rather than from a coarse image window.

Thermal measurements are then compared with neighbouring connectors operating under similar conditions to identify anomalous heating that may indicate increased resistance or connector degradation. Connection state is assessed separately from RGB imagery, enabling connected and disconnected MC4 connectors to be distinguished. By combining component identity, thermal response and 3D location, the inspection output can associate a candidate fault with a specific connector within the solar farm. Examples of faulty MC4 connector detection and connected/disconnected connectorstate recognition are shown in Figure 4 and Figure 5.

## Conclusions & Future Work

This work demonstrates a spatially grounded approach to autonomous under-panel PV inspection, in which visual and thermal detections are associated with persistent 3D asset locations rather than treated as isolated image observations. The system combines LiDAR-inertial localisation, multimodal calibration and machine-learning-based asset recognition to identify structural and electrical components and localise candidate defects across utility-scale PV sites.

Future work will integrate the Tricorder perception stack with autonomous mobile ground robots capable of navigating beneath PV arrays and performing inspection missions without direct operator intervention. The longer-term vision is a fleet of coordinated robots that can repeatedly traverse large solar farms, build and update a shared 3D asset map, revisit previously identified components, and compare observations over time to detect emerging structural, electrical, and thermal defects. Such fleet-based inspection could complement aerial surveys with persistent under-panel coverage and enable scalable, autonomous condition monitoring across utility-scale solar farms.

![](images/cd2e5d15b2866e77fcc0224d7c92dbc73c44a5ff23fbb9b3b4353c618570aa61.jpg)  
(a) RGB

![](images/39cf017ff8f4e9ea76f7b378b45dcb80f48723284883eb547fc590418e0897b7.jpg)  
(b) Thermal

![](images/ef69777ddde0f5908b6e079119d9caee9e4924b64da7f73f82b2bf60c7bdd1bd.jpg)  
(c) 3D Pointcloud

Figure 4 Autonomous detection and segmentation of MC4 connectors. Note the difference in temperature between the failed connector (yellow) and the nominal connector (blue).  
![](images/84e040861983fa6a9ceda92c5a314cbe0ffe640de8cc0226724103cdba88d7ac.jpg)  
(a) RGB

![](images/f8adebbb6cb8c34140aa18768e292079d732d6b17569907c0558c8724a5e8429.jpg)  
(b) Thermal

![](images/fd1094b243778d6882cf7c79b01bb10881a4d57e8d9efd7aad5923b64d839876.jpg)  
(c) 3D Pointcloud  
Figure 5 Autonomous detection of connected and disconnected MC4 connectors.

## References

[1] Peyman Moghadam. 3D Imaging Method and System, May 29 2018. US Patent 9,986,176.

[2] Stephen Vidas and Peyman Moghadam. HeatWave: A handheld 3D thermography system for energy auditing. Energy and Buildings, 66:445–460, 2013.

[3] Peyman Moghadam and Stephen Vidas. HeatWave: the next generation of thermography devices. In Thermosense: Thermal Infrared Applications XXXVI, volume 9105, page 91050F. International Society for Optics and Photonics, 2014.

[4] Milad Ramezani, Kasra Khosoussi, Gavin Catt, Peyman Moghadam, Jason Williams, Paulo Borges, Fred Pauling, and Navinda Kottege. Wildcat: Online continuous-time 3d lidar-inertial slam, 2022.

[5] Lachlan Tychsen-Smith and Lars Petersson. Denet: Scalable real-time object detection with directed sparse sampling. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 428–436, 2017.

[6] Lachlan Tychsen-Smith and Lars Petersson. Improving object localization with fitness nms and bounded iou loss. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6877–6885, 2018.