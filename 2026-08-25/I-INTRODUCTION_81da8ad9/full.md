Bee Detection and Tracking at Hive Entrance using YOLO11 and ByteTrack

Thi Thu Thao Nguyen

IoT Engineering

Savonia University of Applied Sciences

Finland

Thaonguyenthithu0311@gmail.com

Johannes Reschke

Electrical and Information Engineering

Ostbayerische Technische

Hochschule Regensburg

Germany

johannes.reschke@oth-regensburg.de

Abstract—This work presents an automatic bee entrance monitoring system based on YOLO11 transfer learning and the ByteTrack tracking algorithm. The study investigates the influence of data augmentation, backbone freezing, and tracker parameter optimization on the detection and counting of small, fast-moving bees. The detector with progressive backbone unfreezing strategy achieved about 97.0% precision and 98.7% mAP50, while providing more stable convergence than full fine-tuning. Experiments also showed that light augmentation outperformed heavy augmentation. For tracking, ByteTrack parameters were optimized to improve trajectory continuity under low-confidence detections. On an independent 25 FPS side-view video, the optimized YOLO11-ByteTrack system correctly counted 43 of 47 incoming bees (91.5%) and 7 of 30 outgoing bees (23.3%). Error analysis showed that most counting errors were caused by missed detections due to rapid bee motion and motion blur, while tracking failures became less frequent after parameter optimization. Overall, the results indicate that moderate augmentation, progressive backbone unfreezing, and ByteTrack tuning improve the reliability of automatic bee entrance monitoring under realistic recording conditions.

## I. INTRODUCTION

Monitoring bee trafic at hive entrances has become an important task in precision apiculture. The number of bees entering and leaving a hive reflects colony activity and may indicate diseases, food availability, swarming behaviour, or pesticide exposure.

Traditional manual counting is labor-intensive and unsuitable for continuous monitoring. Recent advances in deep learning enable automatic bee detection and tracking using computer vision.

However, tracking bees near the hive entrance is still dificult. Bees occupy only a small part of the image, move rapidly, and are frequently afected by motion blur. In addition, several bees may overlap near the entrance, which can lead to fragmented trajectories and ID switches. This work investigates how modern object detection and tracking algorithms perform under these challenging conditions.

This work focuses on the following aspects:

• training a YOLO11 detector for bee detection,

• comparing diferent augmentation strategies,

• evaluating backbone freezing schemes,

• optimizing ByteTrack parameters for bee tracking,

• analyzing the main sources of counting errors.

## II. Related Work

Recent studies have shown that combining deep learning object detection with multi-object tracking is a practical solution for automatic bee monitoring at hive entrances. A closely related study, A Honey Bee In-and-Out Counting Method Based on Multiple Object Tracking Algorithm [1], proposed a complete pipeline using YOLOv8m together with ByteTrack, OC-SORT, and Deep OC-SORT. The authors also compared a virtual counting line with a counting box and reported the best performance with OC-SORT and the box-based strategy, achieving F1-scores of 91.49% for incoming bees and 89.08% for outgoing bees. The study provides a strong baseline for bee entrance counting. However, the experiments were conducted on videos collected from only two hives, and the training and testing data came from the same environment. This makes it dificult to judge how well the system would perform under diferent viewpoints, backgrounds, or lighting conditions. In addition, the impact of tracker parameter optimization was not investigated.

Another relevant work is A Method for Bee Activities Recognition from Videos Captured at the Beehive Entrance [2], which combined YOLOv5 with OC-SORT to recognize behaviors such as entering, leaving, and carrying pollen. The authors reported high detection precision and good tracking performance. The main limitation of this work is that it focuses on behavior recognition rather than counting accuracy. In addition, only one detector and one tracker were evaluated, and the efect of tracker parameter settings was also not investigated.

For long-term monitoring, Continuous Non-Invasive Monitoring of Hive Entrance Activity Reveals Honey Bee Colony Dynamics [3] used YOLOv8 together with BoT-SORT and evaluated the system on videos that were independent from the training data. This is an important strength because it provides a more realistic assessment of generalization. Nevertheless, the study concentrates on continuous colony monitoring and does not provide a detailed analysis of counting errors such as missed detections, ID switches, or temporary tracking loss near the hive entrance.

Recent work has also emphasized the importance of video quality. Physicsaware Vision Instrumentation for Stingless Bee Counting at Hive Entrance Using Hybrid Edge-Cloud Object Detection [4] showed that lower frame rates increase the displacement of bees between consecutive frames, which makes tracking more dificult and increases the probability of missed detections and ID switches. While the study ofers valuable insight into the relationship between hardware constraints and tracking performance, it focuses on stingless bees and does not evaluate newer detectors such as YOLO11.

Motivated by these limitations, the present study uses publicly available bee detection datasets together with an independent tracking video recorded from a diferent viewpoint. It evaluates YOLO11 detector combined with ByteTrack, investigates the influence of tracker parameter optimization, and provides a detailed analysis of detection failures, tracking failures, and counting errors under more challenging and realistic conditions.

## III. Dataset

## A. Detection Dataset

The dataset combines two publicly available sources: The Mendeley dataset [5] containing over 6,500 images and Dataset Ninja [6] containing over 3,000 images. In total, there are over 9,500 images, splitting into 7,635 training images and 1,909 validation images

The Mendeley dataset [5] provides a large collection of labeled images focusing on bee detection and movement direction on beehive landing boards, while the Dataset Ninja collection [6] contributes additional object detection images with diverse visual characteristics. They together enhance the quantities and variability in appearance, viewpoint, and environmental conditions.

![](images/84e9a0680f1b609e8bf6787c467f3baa22f67ffde9c57f1944028d8d5c73f1e3.jpg)

Fig1. Example image of bees at a hive entrance, adapted from Mendeley dataset, Sledevic [5]  
![](images/c81840b432afaa68eca9648b8722c3fd6ddac5c361c8106e5e08ac4dee21d53d.jpg)

Fig2. Example image of bees at a hive entrance, adapted from Mendeley dataset, Sledevic [5]  
![](images/3f0b9cebbb18723415bd48ac887d07f23455137ae09c78736f9d9b26e3ec0ecc.jpg)  
Fig.3. Example image of bees at a hive entrance, adapted from Dataset Ninja, AndrewL [6]

## B. Tracking Evaluation Video

The evaluation video was collected from Pexels [7]. It was recorded at 25 FPS and 30 second long, using a side-view camera under natural lighting conditions. Because this video was not included in the training data, it was used to evaluate the model’s ability to generalize to a diferent scene rather than to memorize the training environment.

![](images/d7bf4178ea935d437c52898b8f9d44fb2550d1156199731d98a10de35a7e4544.jpg)  
Fig.4. A frame extracted from the video by Ella Gronewold on Pexels [7]

IV. Methodology

A. Detection Model and training Strategy

• YOLOv8 and YOLO11

This study evaluates YOLOv8 and YOLO11 for bee detection. Both are onestage, anchor-free object detectors that provide high detection accuracy with real-time inference speed. Compared with YOLOv8, YOLO11 ofers improved feature representation and localization performance, making it more suitable for detecting small, fast-moving bees. [8]

## • Transfer Learning

Both models were initialized with COCO pre-trained weights and fine-tuned on the bee dataset. Transfer learning accelerates convergence, improves detection accuracy with limited data, and reduces the risk of overfitting.

## B. Tracking

After object detection, detected bees are tracked across consecutive frames using the ByteTrack multi-object tracking algorithm. Beside associating highconfidence detections, ByteTrack also considers low-confidence detection boxes. By matching both high- and low-confidence detections with existing trajectories, the algorithm recovers temporarily weak detections and produces more continuous object tracks. [9]

The tracking pipeline is illustrated as follows:

![](images/e4f0ab922946f5c9ebe1a20a443eeed03551fb223592e3e545bb4b3b6e778b37.jpg)  
Fig.5. Bytetrack pipeline

ByteTrack was selected because honey bees are small, fast-moving objects that often produce low-confidence detections due to motion blur and viewpoint changes. By retaining and associating these low-confidence detections instead of discarding them, ByteTrack reduces missed trajectories and improves counting accuracy while maintaining real-time performance.

## C. Counting Logic

The final stage of the proposed system estimates the number of bees entering and leaving the hive based on their tracked trajectories

![](images/89d2a97f0e7b8d2eaa32342d97f47c4699f317280582c63bb6904ced26ab1605.jpg)

## Fig.6. Counting logic graph

A rectangular counting box is defined at the hive entrance. For each tracked bee, the center point of its bounding box is computed in every frame. The current center position is then compared with its position in the previous frame to determine whether the bee has crossed the boundary of the counting box.

A bee is counted as IN or OUT when its center point crosses the bounding box boundary from the outside in, or from the inside out, respectively. Each counted tracking ID is added to the IN/OUT log for reference. If a bee enters and exits multiple times, each occurrence is recorded independently.

Compared with a single virtual counting line, the counting-box approach is more robust to the oscillatory flight behavior of bees near the hive entrance.

Algorithm 1: Boundary Crossing Logic for Object Counting   
Input: Current detection (track\_id, center\_x, center-y), trajectory   
history track\_history, line boundaries (X1, X2, Y1, Y2)   
Output: Updated counts count\_in, count\_out and tracking sets   
IN\_ids, OUT\_ids   
1 if track\_id ∈ track\_history then   
2 prev\_x,prev\_y ← track\_history[track\_id];   
3 prev\_inside ← (X1 < prev\_x < X2) ∧ (Y1 < prev−y < Y2);   
4 curr\_inside ← (X1 < center\_x < X2) ∧ (Y1 < center−y < Y2);   
5 if ¬prev\_inside ∧ curr\_inside then   
6 count\_in ← count\_in + 1;   
7 IN \_ids ← IN\_ids ∪ {track\_id};   
8 else   
9 if prev\_inside ∧ ¬curr\_inside then   
10 count\_out ← count\_out + 1;   
11 OUT\_ids ← OUT\_ids ∪ {track\_id};

## V. Experiments

## A. Experiment 1: Efect of Augmentation

Data augmentation is important to a model performance under limited-data conditions. However, when it comes to small objects like bees, my observation shows that heavy augmentation with RandAugment and random erasing can brutally diminish model performance in terms of mAP50-95, box loss and class loss, and recall value. Meanwhile, light augmentation consistently achieves higher localization accuracy, higher recall and lower losses, suggesting an optimization in model’s learning. [10]

![](images/cd9407c868f16ec454321358d350c50cc07a6ea8c2617734e5d8a29da5a857be.jpg)

Fig.7. Box\_losses & Class\_losses between light model and heavy augmented model  
![](images/b135b6cf66de6c975ae66df1c505aeca45ec871a382c319681253a3a2e896c84.jpg)

Fig.8. mAP50 & mAP50-95 between light model and heavy augmented model  
![](images/266c8e29eeba1dec8a4f3263c6b26d33763e6481131e44873b1304984177f66e.jpg)  
Fig.9. Recalls between light and heavy augmented models. Slight augmentation includes, model: yolov8s.pt, imgsz: 640, close\_mosaic: 15, hsv\_h: 0.01, hsv\_s: 0.25, hsv\_v: 0.25, degrees: 3.0, translate: 0.04, scale: 0.08, fliplr: 0.3,

mosaic: 0.3, mixup: 0.05; Heavy augmentation includes slight augmentation plus RandAugment

The reason lies in the fact that bees typically occupy less than 1–3% of the image area. Once random augmentation is applied, some shearing or erasing can significantly transform the bees, make them distorted and lose critical visual details. This becomes noises in the feature that negatively impacts detection accuracy.

## B. Experiment 2: Backbone Freezing

The tables below summarize the performance of three training strategies: fully unfreezing all layers (fine tune all), freezing the backbone for the first 30 epochs and then unfreezing (progressive backbone unfreezing), and freezing the backbone throughout the entire training process[11]. While all configurations achieve comparably high precision and recall values, there are diferences in validation losses, convergence behavior, and training stability.

TABLE I. Precision and Recall Comparison
<table><tr><td>Model</td><td>Precision</td><td>Recall</td></tr><tr><td>Unfreeze all</td><td>~0.969–0.973</td><td>~0.960-0.964</td></tr><tr><td>Freeze backbone 30</td><td>~0.971-0.974</td><td>~0.958-0.963</td></tr><tr><td>epochs Freeze backbone all epochs</td><td>~0.964-0.968</td><td>~0.958-0.961</td></tr></table>

TABLE II.mAP50 &mAP50-95 Comparison
<table><tr><td>Model</td><td>mAP50</td><td>mAP50-95</td></tr><tr><td>Unfreeze all</td><td>~0.984</td><td>~0.752-0.754</td></tr><tr><td>Freeze backbone 30 epochs</td><td>~0.987</td><td>~0.783-0.784</td></tr><tr><td>Freeze backbone all epochs</td><td>~0.986</td><td>~0.783-0.784</td></tr></table>

TABLE III. Validation losses Comparison
<table><tr><td>Model</td><td>val/box_loss</td><td>val/cls_loss</td><td>val/dfl_loss</td></tr><tr><td>Unfreeze all</td><td>~0.875</td><td>~0.345</td><td>~1.018</td></tr><tr><td>Freeze backbone 30 epochs</td><td>~0.862</td><td>~0.360</td><td>~0.955</td></tr><tr><td>Freeze backbone all epochs</td><td>~0.933</td><td>~0.380</td><td>~0.960</td></tr></table>

TABLE IV. Training duration

<table><tr><td>Model</td><td>Epochs</td></tr><tr><td>Unfreeze all</td><td>81</td></tr><tr><td>Freeze backbone 30 epochs</td><td>93 (highest)</td></tr><tr><td>Freeze backbone all epochs</td><td>75</td></tr></table>

When comparing the three backbone training strategies, mAP50, mAP50–95 and losses over epochs reveals a distinct performance. With mAP50–95 around 0.75–0.753, freezing the backbone for all epochs results in the lowest overall performance, showing minimal localization accuracy. Box loss, classification loss and dfl\_loss remain highest among three strategies, implying limited adaptability to the target dataset. This is due to features in backbone being fixed throughout training.

Full fine-tuning achieves high mAP50–95 at 0.784–0.785 along with over 96% precision and recall, demonstrating strong localization capability. However, its performance fluctuates more across epochs and is accompanied by higher validation losses as compared to progressive backbone unfreezing by over 6%, suggesting a greater risk of overfitting.

In contrast, progressive backbone freezing delivers the most stable and balanced performance. With high mAP50 (f 0.986–0.987), highest precision (f0.971–0.974) and around 96% recall rate, it represents accuracy and robustness of the model. Additionally, this strategy gains the lowest validation DFL loss (f0.955), indicating more accurate bounding box distribution and improved localization accuracy.

From a convergence aspect, the unfreezing all model reaches early convergence at epoch 82, whereas the progressive backbone unfreezing, which freezes backbone for the first 30 epochs, trains for the longest duration (93 epochs), indicating improved training stability and delayed overfitting. Given a small and less diverse dataset, full fine-tuning increases the risk of rapid memorization of training samples rather than learning generalizable representations.

Overall, under limited-data condition, progressive unfreezing, specifically freezing the backbone during early epochs provides the best stability and task-specific learning. This strategy efectively mitigates overfitting while improving localization accuracy, making it particularly suitable for small-scale object detection datasets.

![](images/e83bd288473b9a23c600a1667e3b9cdcbb0afe7fb20b582e8a1695695272e729.jpg)  
Fig.10. Training and Validation DFL Loss between Fine Tune all and Progressive Backbone Unfreezing.

## C. Experiment 3: Tracking Performance using YOLO8 vs YOLO11

This experiment compares the tracking performance of YOLOv8 and YOLO11 using the same tracking pipeline. YOLO11 preserved bee features more efectively, especially when bees moved quickly or were partially occluded. The C2PSA module improved attention to small localized targets, resulting in fewer missed detections.[12]

Tracking was also more stable with YOLO11. The use of C3k2 blocks produced more consistent features across consecutive frames, which reduced ID switches during tracking. As a result, YOLO11 achieved better tracking continuity and higher counting accuracy in continuous hive monitoring.

## D. Experiment 4: Tracker Parameter Optimization

TABLE V. Optimized ByteTrack Tracking Parameters and Their Rationale
<table><tr><td>Parameter</td><td>Selected Value</td><td>Rationale</td></tr><tr><td>track_high_thresh = 0.15</td><td>Lower than default (0.25)</td><td>The detection confidence of bees often fluctuates because of their small size and fast motion. Reducing the threshold allows more valid bee detections to participate in the first association stage,</td></tr><tr><td>track_low_thresh = 0.03</td><td>Lower than default (0.10)</td><td>Bees may disappear for only one or two frames because of partial occlusion at the hive entrance. A very low threshold enables</td></tr><tr><td>new_track_thresh = 0.15</td><td>Lower than default (0.25)</td><td>ByteTrack to recover these weak detections during the second association stage instead of terminating the track. Lowering this threshold allows newly appearing bees to be initialized more quickly when entering the camera view. Although this may introduce some</td></tr><tr><td>track_buffer = 50</td><td>Higher than default (30)</td><td>short-lived false tracks, it reduces missed bee entries, which is more critical for counting accuracy. Bees frequently overlap near the hive entrance. Increasing the buffer keeps lost tracks alive for a longer period, enabling successful re-association after temporary</td></tr><tr><td>match thresh = 0.85</td><td>Higher than default (0.80)</td><td>occlusion and reducing fragmented trajectories. Because many bees have similar appearances, association mainly depends on motion and spatial overlap. A higher matching threshold requires stronger spatial</td></tr><tr><td>fuse_score = True</td><td>Default</td><td>consistency before assigning an existing ID, reducing incorrect ID switches between nearby bees. Detection confidence is combined with motion similarity during association, improving matching robustness by favoring more reliable detections.</td></tr></table>

The tracker parameters[13] were intentionally adjusted to prioritize tracking continuity rather than conservative filtering. Honey bees are small, fast-moving, and frequently occluded at the hive entrance, causing temporary drops in detection confidence. Therefore, lower confidence thresholds and a longer track bufer were adopted to preserve object identities across short detection failures.

However, these more permissive settings increase the risk of incorrect associations. To compensate, a higher matching threshold was used, requiring stronger spatial consistency before assigning detections to existing tracks. This combination improves the continuity of bee trajectories while limiting unnecessary ID switches, ultimately leading to more reliable bee counting when individuals cross the virtual counting region.

## VI. Results

The counting performance of YOLOv8 and YOLOv11 under two conditions: with and without tracking parameter optimization are shown in the graph as compared to the ground truth.

In the testing video, there are actually 47 bees going in and 30 bees going out of the hive.

![](images/931efa9cdbb53b1b6fd0c40982160984cf046232807004838c6b1b86465596f8.jpg)

Fig.11. Bee Counting IN vs Ground Truth  
![](images/bbd91b33bc40e799f4fc9b2c4fccc732967730c475a5711c6e9eb91abf06a7f8.jpg)  
Fig.12. Bee Counting OUT vs Ground Truth

YOLOv11 consistently outperformed YOLOv8 in both configurations.

Without tracking:

IN: 38 vs. 30 (26.7% improvement)

OUT: 5 vs. 0

With the optimized tracker:

IN: 43 vs. 32 (34.4% improvement)

OUT: 7 vs. 4 (75% improvement)

The better accuracy of YOLOv11 suggests that it produced more reliable detections for small, fast-moving bees. Better detections provide higher-quality inputs to the tracker, resulting in more stable trajectories and improved counting performance.

Lowering the tracking confidence threshold increased the number of recovered detections, reducing missed bee trajectories. The overall counting accuracy improved. Also, there is a significant improvement in outgoing bees since low FPS video leads to big displacement when bee moving fast between frame, hence, the model could not regard the bee as the same ID

## VII. Error Analysis

## A. Detection Failures

The majority of detection failures occurred for outgoing bees, mainly because they move much faster than incoming bees and are captured from more challenging viewing angles. Several factors contributed to these missed detections:

Motion blur: Outgoing bees often leave the hive at high speed, causing motion blur in video frames. This reduces the visibility of important visual features and makes detection more dificult.

Viewpoint mismatch: The training dataset did not contain suficient variation in bee flight orientations. Bees flying at unusual or side angles may appear significantly diferent from the top-down views that dominate the dataset, which reduces the model’s generalization ability.

Lighting variation: The test video was recorded under relatively stable natural lighting. Illumination had only a minor influence on detection performance, although bees occasionally became darker when flying through shaded areas. Other than that, the recording conditions were relatively stable, so lighting was not considered a major source of error.

Bee shadows: Although shadows were present around the hive entrance, the detection model was generally able to focus on the characteristic body pattern and shape of the bees. As a result, shadows rarely caused false detections.

Overall, the results suggest that high-speed motion and insuficient viewpoint diversity in the training dataset were the primary causes of missed detections, especially for outgoing bees, whereas lighting changes and shadows had relatively little impact on detection performance.

![](images/20fe750114c6889f3e545a48414d060c06ac741eacb28b5a88f59f716b4cfddf.jpg)  
Fig.13. Example of a fast-moving bee exiting the hive, demonstrating motion blur and an oblique viewing angle.

## B. Tracking Failures

Tracking failures mainly occurred when bees moved quickly between consecutive frames.

Fast flight and low FPS: Because the video was recorded at 25 FPS, fast-moving bees often changed position considerably between frames. Large displacements make it harder for ByteTrack to associate detections with the correct existing track.[9]

ID switches: ID switches were observed occasionally, particularly when several bees were close to each other near the entrance.

Temporary disappearance: Bees were sometimes lost for one or two frames because of partial occlusion or reduced detection confidence, but this happened only rarely.

Rapid turning: Sudden changes in flight direction could also break trajectory continuity.

After parameter optimization, the number of fragmented tracks decreased, although a trade-of remained between preserving tracks for fast-moving bees and avoiding incorrect ID associations.

## VIII. Conclusion

This study presented a bee entrance monitoring system based on YOLO11 for detection and ByteTrack for tracking. The experiments showed that moderate data augmentation and progressive backbone unfreezing provided the best balance between accuracy and generalization. Tuning the ByteTrack parameters also improved trajectory continuity and reduced counting errors compared with the default configuration.

The error analysis indicated that the main limitation of the current system is missed detection of fast-moving outgoing bees. Once detections are lost, tracking becomes more dificult and counting accuracy decreases. In contrast, incoming bees are generally easier to detect and track because they move more slowly near the hive entrance.

Future work will focus on improving robustness to viewpoint changes and lowframe-rate videos, for example by adding more diverse training data and exploring tracking methods that make stronger use of temporal information.

## References

1. “A Honey Bee In-and-Out Counting Method Based on Multiple Object Tracking Algorithm.” Accessed: Jul. 25, 2026. [Online]. Available: https://www.mdpi.com/2075-4450/15/12/974

2. N. Le, T.-T.-H. Phan, and T.-L. Le, “A method for bee activities recognition from videos captured at the beehive entrance,” JMSTs Sect. Comput. Sci. Control Eng., no. CSCE8, pp. 3–13, Dec. 2024, doi: 10.54939/1859- 1043.j.mst.CSCE8.2024.3-13.

3. C. Ö. Tozkar, “Continuous Non-Invasive Monitoring of Hive Entrance Activity Reveals Honey Bee Colony Dynamics,” Biology, vol. 15, no. 9, p. 731, May 2026, doi: 10.3390/biology15090731.

4. M. A. Md Yunus et al., “Physics-aware vision instrumentation for stingless bee counting at hive entrance using hybrid edge-cloud object detection,” EPJ Web Conf., vol. 377, p. 02010, 2026, doi: 10.1051/epjconf/202637702010.

5. T. Sledevic, “Labeled dataset for bee detection and direction estimation on beehive landing boards,” vol. 6, Aug. 2024, doi: 10.17632/8gb9r2yhfc.6.

6. https://datasetninja.com/bee-image, “Bee Image Object Detection,” Dataset Ninja. Accessed: Jan. 16, 2026. [Online]. Available: https://datasetninja.com/bee-image

7. “Bees Flying Around Beehive Free Stock Video Footage, Royalty-Free 4K & HD Video Clip.” Accessed: Jul. 25, 2026. [Online]. Available: https://www.pexels.com/video/bees-flying-around-beehive-857037/

8. “YOLO11 vs YOLOv8 Comparison,” Ultralytics Docs. Accessed: Jul. 29, 2026. [Online]. Available: https://docs.ultralytics.com/compare/yolo11- vs-yolov8

9. Y. Zhang et al., “ByteTrack: Multi-object Tracking by Associating Every Detection Box,” in Computer Vision – ECCV 2022, vol. 13682, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds., in

Lecture Notes in Computer Science, vol. 13682. , Cham: Springer Nature Switzerland, 2022, pp. 1–21. doi: 10.1007/978-3-031-20047-2\_1.

10. “Value-Guided Adaptive Data Augmentation for Imbalanced Small Object Detection.” Accessed: Jul. 30, 2026. [Online]. Available: https://www.mdpi.com/2079-9292/13/10/1849

11. “Fine-Tune YOLO26 on a Custom Dataset,” Ultralytics Docs. Accessed: Jul. 30, 2026. [Online]. Available: https://docs.ultralytics.com/guides/finetuningguide

12. R. Khanam and M. Hussain, “YOLOv11: An Overview of the Key Architectural Enhancements,” Oct. 23, 2024, arXiv: arXiv:2410.17725. doi: 10.48550/arXiv.2410.17725.

13. “YOLO Multi-Object Tracking in Video,” Ultralytics Docs. Accessed: Jul. 30, 2026. [Online]. Available: https://docs.ultralytics.com/modes/track