# BMCTrack-d: Pig re-identification and tracking via back marks in challenging camera settings

David Brunner<sup>1,2,\*</sup>, Maciej Oczak<sup>3,4</sup>, Marie Bordes<sup>4</sup>, Jean-Loup Rault<sup>4</sup>, Stephan M. Winkler<sup>1</sup>, and Viktoria Dorfer<sup>1</sup>

<sup>1</sup>Bioinformatics Research Group, PLFDoc, University of Applied Sciences Upper Austria, Softwarepark 11, Hagenberg, 4232, Austria

<sup>2</sup>Computer Vision Lab, TU Wien, Favoritenstraße 9/11, Vienna, 1040, Austria

<sup>4</sup>Animal Welfare Science Unit, The University of Veterinary Medicine Vienna, Veterinärplatz 1, Vienna, 1210, Austria

<sup>\*</sup>corresponding author: david.brunner@fh-hagenberg.at

## Abstract

Automated pig monitoring is essential for assessing their health, behaviour, and welfare. To date, most pig monitoring solutions operate on the group-level, because individual-level monitoring requires reliable long-term identification and tracking of each animal. For domesticated pigs this remains challenging because pigs of the same breed often have highly uniform appearances. Moreover, research on pig monitoring is almost exclusively reported in top-down view camera settings, which considerably ease tracking, but are not always an option in practice. In this work, BMCTrack-d is presented, a novel tracking-by-detection approach that leverages unique back marks to enable robust pig re-identification and tracking in a challenging side-view camera setting, aflicted by rapidly moving pigs, severe occlusions and low resolution. The method first predicts the detected pigs’ identities using a neural network-based back mark classifier. To improve re-identification reliability over time, two dedicated post-processing stages are introduced: a temporal prediction consistency check, which validates the identity assignments against the recent prediction history, and deduplication, which resolves conflicting identity assignments in each time step. By explicitly prioritising accurate, appearance-based re-identification over continuous tracking, the proposed approach addresses a key limitation of existing trackers for individual-level monitoring scenarios. On a demanding test set BMCTrack-d outperforms two strong baselines, BoT-SORT-ReID and TrackTrack-ReID, by 9.11% and 1.03%, respectively, in higher-order tracking accuracy. These results demonstrate the efectiveness of back mark-based re-identification and tracking for robust individual-level pig monitoring in challenging settings.

Keywords: precision livestock farming; tracking; re-identification; pigs; back marks; detection

## 1 Introduction

Animal observation is an established way of deriving information on their reproductive state (Foote, 1975), health condition (Fernández-Carrión et al., 2017) and social relations (Clouard et al., 2024). Given that continuous observation imposes significant time investment on human observers and can be subjective, there is great interest in automated monitoring solutions, which typically take the form of sensors collecting data and machine learning (ML) algorithms extracting insights from these data. For the analysis of behaviour, a range of sensors can be used, like acceleration sensors (Mayrhuber et al., 2026) or cameras (Oczak et al., 2024). The latter show better scalability, because a single camera can cover multiple animals. However, while some sensors allow diferentiating animals simply by virtue of being attached to individuals, cameras necessitate advanced computer vision algorithms for this task. A number of studies have shown ML’s capacity for detecting animals (Liu et al., 2023) and analysing their behaviour (Liu et al., 2020). However, many of the existing studies perform behaviour recognition only on group level (Gan et al., 2021, 2022; Gao et al., 2023; Li et al., 2020, 2024; Liu et al., 2020; Zhang et al., 2019), e.g., counting the number of occurrences of certain behaviours in a pen. For specific interventions like treating a sick animal, or to identify social interactions between specific individuals (e.g., tail biting), it is necessary to monitor animals on the individual level instead.

To date, only few studies have reported results on individual-level monitoring in a group of animals (Odo et al., 2024, 2025; Psota et al., 2020). The main obstacle is the dificulty of tracking (Parmiggiani et al., 2023), i.e., continuously diferentiating the animals, that is required for attributing behaviours to individuals. Unlike humans, who often can be diferentiated by clothing alone, animals tend to have uniform appearance, which is especially true for domesticated species like pigs, that have been intensely bred for homogenous traits to enable the standardisation of production systems. To support humans in recognising individual pigs, measures like ear marks or back marks are commonly employed, e.g., to track medical treatment or reproductive state. However, to date, only few algorithms have made use of such marks for automatic livestock monitoring (Fruhner et al., 2022; Kashiha et al., 2013; Psota et al., 2020; Wutke et al., 2025).

Furthermore, research on pig monitoring is almost exclusively reported on top-down view camera recordings (e.g., Kashiha et al. (2013); Odo et al. (2025); Psota et al. (2020)). Such a setup significantly eases tracking, e.g., by limiting the possibility of occlusions, but might be prohibitive to downstream tasks such as behaviour detection, or unfeasible due to facility constraints (see Section 3.1.2). A side-view camera setting, in contrast, can pose significant challenges to tracking due to the possibility of severe occlusions and fluctuations in resolution whenever the pigs change their angle and distance to the camera, leading to lost identities. To enable true individual-level monitoring, a tracking algorithm must have strong re-identification (reID) capabilities to recover from such practical challenges.

Taking a step towards individual-level pig monitoring in non-idealised settings, this study proposes BMCTrack-d (back-mark-classification-based tracking with deduplication), a novel algorithm aimed at identifying and tracking pigs via their back marks, designed to prioritise strong reID over tracking in order to deal with challenging camera settings. The main contributions of this study are as follows:

• BMCTrack-d, a novel algorithm for pig reID and tracking via back marks

• A challenging test dataset, consisting of video recordings in side-view, containing high levels of occlusion, and motion blur, as well as low-resolution pigs

• A comprehensive set of experiments, including of a study of BMCTrack-d’s modules and a comparison to state-of-the-art trackers

## 2 Related Work

## 2.1 Tracking

The most widespread approach to tracking is to first detect all objects in every frame and subsequently try to match these detections across frames. This is referred to as tracking-by-detection and builds on the predictions of object detection algorithms (e.g., Khanam and Hussain (2024)). In the context of deep learning, a number of high-performing methods have emerged that follow the tracking-by-detection paradigm, among which Deep-SORT (Wojke et al., 2017), StrongSORT (Du et al., 2023), ByteTrack (Zhang et al., 2022), BoT-SORT (Aharon et al., 2022) and most recently TrackTrack (Shim et al., 2025) are prominent examples. Matching the detections across frames generally follows a twopronged principle: 1) association by motion features and optionally 2) association by appearance features. In simple words, the algorithms try to match objects in consecutive frames by their estimated movement and optionally via their appearance. Typically, the association strategies comprise handcrafted logic like

Hungarian matching, motion models and Kalman filters. Recently, Gao et al. (2025) presented MOTIP, which reformulates association as an end-to-end trainable task instead, reaching state-of-the-art results on multiple tracking benchmarks. However, contrary to previous methods which only require training data for the object detection task, end-to-end training requires a dedicated training set for the tracking task as well, making methods of this type unsuitable for scenarios with expensive data acquisition.

Interestingly, many of the studies on animal monitoring developed custom solutions instead of employing known methods. Liu et al. (2020) developed a simple short-term tracking method for tracking pigs in 1-second intervals, matching pigs across frames via features derived from the detected bounding boxes. Gan et al. (2021) developed an integrated architecture for the simultaneous detection and tracking of piglets. Parmiggiani et al. (2023) presented an eficient tracking algorithm that matches detections across frames via a graph structure. Research that used known trackers includes Lu et al. (2024), who employed a modified ByteTrack for pig tracking and Guo et al. (2023), who compared modified versions of several existing methods, concluding that FairMOT (Zhang et al., 2021) is best suited for the task of pig tracking, although based on outdated evaluation metrics. Tu and co-authors conducted a whole series of studies on pig tracking and behaviour detection, testing a variety of known methods: In Tu et al. (2024d) and Tu et al. (2024a) they used ByteTrack, in Tu et al. (2024b) BoT-SORT and in Tu et al. (2024c) OC-SORT (Maggiolino et al., 2023).

BMCTrack-d follows the tracking-by-detection paradigm but, contrary to most other trackers, drops motion-based association in favour of appearance-based association in the shape of strong reID.

## 2.2 Pig re-identification

There are at least three ways visual pig reID has been realized in literature, via 1) ear marks, 2) back marks, and 3) marker less reID via learned appearance features. Psota et al. (2020) proposed an algorithm for long-term tracking of a fixed number of pigs via custom alphanumeric ear marks. Fruhner et al. (2022) designed a special matrix pattern for ear marks allowing unique identification. More recently, Wutke et al. (2025) presented a multi-stage pipeline for commercial ear mark detection and identification. Given that ear marks are standard equipment in animal husbandry, using them for automated identification is a natural approach. However, many of the existing methods require custom ear marks. Furthermore, ear marks are limited in size by design, which makes them susceptible to occlusions and infeasible in case of far camera placement. Conversely, back marks can span the whole body of the pigs, making them more robust to occlusions and low resolution caused by the camera type or placement. While back marks are routinely used for manual pig identification, the only study to date to explicitly utilise back marks for automatic pig reID is Kashiha et al. (2013), who used algorithms from classic computer vision to identify pigs via specially designed

back mark patterns.

Some studies have investigated marker less pig reID (Odo et al., 2024; Wang et al., 2022). They use modified tracking architectures, equipped with a dedicated branch for learning to extract identifying appearance features. Some modern trackers have this capability built-in (BoT-SORT-ReID, TrackTrack-ReID). However, in both studies the pigs had distinguishing features in their appearance. The datasets of Odo et al. (2024) included pigs with unique coat patterns and coloured back marks. In the study of Wang et al. (2022) the pigs had numeric back marks. It stands to reason that in both cases the neural networks learned to extract these distinguishing marks, putting doubt on the feasibility of marker less pig reID. BMCTrack-d explicitly uses back marks for pig reID, outperforming methods that use learned appearance features (BoT-SORT-ReID, TrackTrack-ReID).

Moreover, all these previous studies evaluated their methods on data recorded in top-down view. The only exception is Wutke et al. (2025), who used an additional test set consisting of side-view recordings. However, this side-view camera was set-up at close-range at a feeder and only ever shows few individuals. Conversely, the test set in this study exclusively consists of full-pen side-view recordings, allowing for severe occlusions and very low-resolution pigs (∼40x30 pixels at the far end of the pen). This comprises a more realistic benchmark, given that in practice top-view cameras are not always an option, as discussed in Section 3.1.2 in more detail. Table 1 summarizes the setup of previous studies in contrast to ours.

## 3 Materials and Methods

## 3.1 Data and challenges

## 3.1.1 Experimental setup

The experimental setup was identical to the one described in Brunner et al. (2026b). It consisted of two pens (pen A, pen B), located at Medau, the pig research and teaching farm (“VetFarm”) of the University of Veterinary Medicine Vienna, Vienna, Austria. The pens were identical in build and were constructed for an ongoing multi-national observational study on social behaviour in pigs, focusing on helping behaviour (“Let me out”, doi:10.55776/I6488). They were 3 m x 4 m in size, with a slatted area of 0.6 m x 3 m at one end and equipped with a four-head-space feeder for weaners, as well as an automatic drinker. Each pen housed exactly ten pigs (Large White x Pietrain), selected from two diferent litters. Daily provision of hay and food ad libitum, as well as toys for enrichment were provided. To facilitate recognition, the pigs regularly received back marks using livestock spray paint which are described in more detail in Section 3.2.2. The study covered a total of seven groups of ten pigs, each group being observed for a 6-week period, from weaning (four weeks of age) to ten weeks of age. The pigs were recorded with two identical cameras (HIKVISION DS 2CD5046G0- AP, 1200x780@25, fisheye lens, Hikvision Co. Ltd., Hangzhou, Zhejiang) per pen, positioned in side view and top view, respectively. Due to the low ceiling, the top view camera covered only half of the pen. The camera streams of pen A are shown in Fig. 1.

## 3.1.2 Tracking challenges

The experimental setup described in Section 3.1.1 comprises an especially challenging scenario for tracking algorithms for multiple reasons, of which the most important are 1) the camera angle, 2) the camera resolution and 3) the fast movement of the animals.

The most frequently used camera setup in pig monitoring is top view, in which the camera is placed on the ceiling facing down. This is motivated by the fact that it allows for uninterrupted observation of all pigs, mostly eliminating situations in which they cover each other. However, there are situations in which a top view camera placement is not preferrable for practical or strategic reasons. The study described in this paper provides an example for both. For one, the low ceiling of the experimental facility simply prevents obtaining useful top-view recordings. Furthermore, behavioural studies benefit from camera angles that allow observing the animals’ legs, which might be implicated in certain behaviours (e.g., lying down or pawing behaviour) but are covered in top view. A side-view camera angle, however, comes at the cost of the aforementioned occlusions, caused by pigs covering their mates from the perspective of the camera. Occlusions can lead to lost tracks and pose an important challenge for tracking.

The moderate resolution of the cameras used in this study (1200 x 780 pixels) is another challenge for tracking algorithms. While high resolution cameras have become relatively inexpensive, their recordings take up much more disk space, which, in the case of continuous recordings, can be prohibitive. Also, because automated monitoring of animals is still a nascent field and previous studies were conducted primarily with human observers in mind, which are less encumbered by low resolution than algorithms, it should be expected that much available video material is in low resolution. In low resolution, physical characteristics that would allow diferentiating individual animals are less pronounced, posing a challenge for trackers that use appearance features in addition to movement.

Lastly, the study described in this paper is focused on young pigs, which are very active and prone to fast movement, which can pose a challenge for tracking algorithms for two reasons. First, large jumps in location make it hard to associate an individual in consecutive frames via its position. Second, tracking algorithms that rely on physical appearance are also afected by motion blur, which, similarly to low resolution, weakens idiosyncratic features that could be used for diferentiation. Figure 2 illustrates the described challenges.

## 3.2 Proposed tracking algorithm

## 3.2.1 Overview

The proposed tracking algorithm is termed BMCTrackd (back-mark-classification-based tracking with deduplication) and consists of four steps: 1) object detection,

Table 1: Summary of existing pig reID studies.
<table><tr><td>source</td><td>reID modality</td><td>test set camera view(s)</td><td>resolution</td></tr><tr><td>Kashiha et al. (2013)</td><td>back marks</td><td>top-down</td><td>720x576</td></tr><tr><td>Odo et al. (2025)</td><td>learned appearance features</td><td>top-down, top-down</td><td>1920x1080, n/a</td></tr><tr><td>Wang et al. (2022)</td><td>learned appearance features</td><td>top-down</td><td>1270x720</td></tr><tr><td>Psota et al. (2020)</td><td>ear marks (custom, alphanumeric)</td><td>top-down</td><td>2688x1520</td></tr><tr><td>Wutke et al. (2025)</td><td>ear marks (commercial)</td><td>top-down, side-view (close)</td><td>1920x1080, n/a</td></tr><tr><td>Fruhner et al. (2022)</td><td>ear marks (custom, pattern)</td><td>top-down</td><td>n/a</td></tr><tr><td>ours</td><td>back marks</td><td>side-view</td><td>1280x720</td></tr></table>

a)  
![](images/69606eae04b15414042bbb46a5798b08b9761e54967aba5c7929e2ae2347c370.jpg)

b)  
![](images/6b8c002ad5ae32523827d335fc338dfb6cca264291b83d6cbb0da7fd235fcedc.jpg)  
Figure 1: The camera setup. It consisted of two cameras per pen, a side-view camera (a) and a top-view camera (b). Adapted from Brunner et al. (2026b).

2) back mark classification, 3) temporal prediction consistency (TPC) check and 4) deduplication. For each frame, first, bounding boxes around all pigs in the scene are inferred. Then, each bounding box area is passed to an image classifier that predicts which individual is shown via its back mark. Next, the class predictions are refined by checking if they are consistent with the predictions in previous frames. Finally, by ensuring that each class is uniquely represented in the frame, remaining class collisions are resolved. Figure 3 shows a high-level illustration of the whole algorithm. Sections 3.2.2 – 3.2.5 provide details on the individual components.

## 3.2.2 Object detection and back mark classification

BMCTrack-d follows the tracking-by-detection paradigm, using YOLOv11 (Khanam and Hussain, 2024) as the object detector. While in this first detection step the algorithm closely follows existing trackers (e.g., ByteTrack (Zhang et al., 2022), BoT-SORT (Aharon et al., 2022)), the next step is where it diverges. Tracking algorithms typically assign a random ID to each detected object and then try to preserve this association between object and ID throughout the video clip (Aharon et al., 2022; Du et al., 2023; Maggiolino et al., 2023; Wojke et al., 2017; Zhang et al., 2022). Given that for the described behavioural study, it is important not only to keep the individuals apart over time but also to know their identity, a separate classification step is added. The detected bounding boxes are cropped from the frame and passed to an image classifier of type ResNet-50 (He et al., 2016), which predicts which of a list of known individuals the detected pig represents. Performed in every frame, this results in a track for each individual.

To support recognition, the pigs were equipped with back marks. Figure 4 shows the ten unique back marks used in this study. They were renewed every 3-4 days to ensure good readability. As the back marks were hand drawn, they were not perfectly consistent across groups. Insights into ways of improving the back mark design for future studies are described in (Brunner et al., 2026a).

## 3.2.3 Temporal prediction consistency

Recognising the individuals in any given frame requires the back marks to be visible. As described in Section 3.1.2, the data produced in this study include frames with occlusions, motion blur and moderate resolution, all of which pose obstacles for recognition. Due to the side-view camera angle, the visibility of the back marks is also dependent on the body pose and the orientation of the pigs in relation to the camera. Therefore, it is to be expected that the recognition in individual frames contains errors. To refine the class predictions for a given frame, they are compared to the associated predictions in previous frames. Thus, it is not necessary for each class prediction to be correct in a sequence of frames, only that the majority of predictions are correct. By ensuring the temporal prediction consistency (TPC) of a sequence of predictions, the tracking accuracy can be improved. Figure 5 illustrates this idea. Specifically, the class ID of the current frame is determined by the majority class prediction in the last n frames (the frame bufer) and the current prediction. If there is no majority, the current prediction is kept. The TPC check starts once n frames have accumulated. Section 3.2.6 gives a concrete example for this process.

![](images/5f8ba2431082413383af05eec74bee7a02c283c6b6053764c99c9d836be31d23.jpg)

Figure 2: The challenges posed by the experimental setup of the study. From left to right: low resolution, motion blur and occlusions.  
![](images/803977902dc61881971626c09cf8d4652ecf48a07aabd045cb7ba7b7e91e81bc.jpg)  
Figure 3: Overview of the workflow of BMCTrack-d. For a given input frame (a), object detection is performed to localize the pigs (b), subsequent classification predicts their identity (c) and a prediction refinement step consisting of a temporal prediction consistency check and deduplication is performed to improve the identification (d).

## 3.2.4 Deduplication

The classifications of individual bounding boxes in a frame are independent of each other, meaning that the assigned class IDs are not unique and multiple pigs can be assigned identical class IDs. Given that each individual can only appear once per frame, these collisions in recognition must be resolved. The object detection step is ignorant of individual class IDs but provides confidence scores for each predicted bounding box. As the number of pigs per pen is known to be exactly ten, in a first step, the bounding boxes can be ranked by confidence and clipped to ten. Unlike the object detector, the classifier produces a vector including a confidence score for each class ID. If collisions occur among the remaining bounding boxes, i.e., if two predictions assign the highest confidence to the same class, the one with the lower maximum confidence can be shifted to its second highest confidence class as illustrated in Fig. 6. Given that this could result in a new collision, the process is repeated until each class prediction is unique. This deduplication is performed after the TPC check and overrides the latter.

## 3.2.5 Matching variants

The TPC check described in Section 3.2.3 compares class predictions over time. In order to compare the class prediction for a specific individual in frame t with its predecessors in the last n frames, some sort of matching has to be performed to find the same individual in previous frames. One way of doing so is to measure the Intersection over Union (IoU) of all bounding boxes in two sequential frames and assume that those with the highest IoU represent the same individual. This typically works well for consecutive frames, as shown in Fig. 7a. However, depending on the video frame rate and the speed of the animals’ movement, it can fail for bigger temporal jumps, e.g., matching a bounding box in frame t with those in frame t-4, as illustrated in Fig. 7b. An alternative to matching bounding boxes is matching the keypoint skeletons predicted by a pose estimation model, as illustrated in Fig. 7c. The advantage of matching keypoint skeletons is that they not only encode the pigs’ locations but also their pose and orientation. If a pig in frame t occupies the same location that a diferent pig occupied in frame t-4, but they difer in their pose or orientation, a wrong match can be avoided. Analogously to IoU for matching bounding boxes, object keypoint similarity<sup>1</sup> (OKS) serves as metric for matching keypoint skeletons. Figure 7d depicts a simpler alternative to full skeleton matching, in which IoU-based matching is supplemented with a comparison of the current orientation of the pig in frame t to the mean orientation of the matched pig instances in the previous n frames. The orientation can be derived from only two keypoints, one at each end of the pig’s body. ViTPose (Xu et al., 2022) serves as the pose estimation model; for details on model training, data and the keypoint skeleton structure refer to Brunner et al. (2026b).

![](images/f6f15f6843635f6b005dd81b4c2f7975886cc1ef08608d4eae469fb177791503.jpg)  
Figure 4: An example for each back mark used in this study. From left to right: dot dot, dot line horizontal, i, line line horizontal, o, reverse t, s, v, vertical line, x. Adapted from Brunner et al. (2026a).

![](images/4d9f48f63f6edce2d5a6d36792342fe9d9a99237434b3a8ed3b4e1b5b6381165.jpg)  
Figure 5: An illustration of the temporal prediction consistency check. The current prediction is not in line with the previous n predictions, so it is updated to the majority prediction.

## 3.2.6 Algorithmic details

For a better understanding of how BMCTrack-d works, this section provides a concrete example, showcasing the interplay between the steps. Figure 8c illustrates the general workflow. The TPC check starts only after the number of frames specified by the frame bufer length were processed. Up until this point, for each frame only object detection, followed by back mark classification and deduplication are performed. Once the frame bufer is full, the TPC check is added between the back mark classification and deduplication. Figure 8a sketches the processing of the first six frames of a video clip showing five pigs with unique back marks, for frame bufer length n = 4. In frame t-4, after object detection and back mark classification, duplicate predictions for pigs a and b are resolved by deduplication, switching b’s class ID to 5. The same principle applies to frames t-3 to t-1, the changed predictions in bold. From frame t forward, the TPC check is added. The sequence of predictions in previous frames (blue dashed boxes) reads 1, 1, 2, 2, the prediction in frame t (blue solid box) is 2. The latter makes class 2 the majority prediction and it is adopted for frame t. However, because the prediction confidence (not shown in the figure) of class 2 for pig a is lower than for pig c, deduplication switches pig a’s class to 1, overriding the TPC check. In frame t+1 pig a is, again, misclassified as 2, which results in a majority in the sequence (red boxes) and the adoption of 2 by the TPC check, an error, once more corrected by deduplication. Figure 8b compares the raw class predictions to the final corrected predictions by BMCTrack-d and shows a trend in which the tracks of pigs a and b are gradually corrected to their true classes 1 and 5 at the cost of some errors in pig c’s track. Pigs d and e are detected correctly and also unafected by deduplication in this example.

## 3.3 Experiments

## 3.3.1 Datasets and model training

The training and validation datasets for the object detector consist of 567 frames and 30 frames, respectively, which were extracted from the video recordings collected in the study. They span multiple groups of pigs, in both pens (pen A, pen B) and both camera views (side view, top view). A summary is provided in Table 2. All 5715 pig instances were annotated with bounding boxes using either the Computer Vision Annotation Tool<sup>2</sup> (CVAT) or COCO Annotator (Brooks, 2019). The YOLOv11 object detector reached 99.42% mean average precision at IoU threshold 0.5 (mAP@0.5) on the validation set. For the training of the classifier a diferent, but overlapping set of the data was used, the areas of the annotated bounding boxes were extracted and all crops in which the back marks were not visible manually filtered. This version of the training and validation datasets, as well as the classifier training are explained in more detail in Brunner et al. (2026a). The classifier reached 91% accuracy on the validation set. Both the object detector’s and the classifier’s validation data were sampled from held out video data, recorded independently from the training data. This is to ensure generalisation across visual properties that difer over time, such as variations in the back marks, which cannot be drawn identically every time. The dataset and training details of the ViTPose pose estimation model are described in Brunner et al. (2026b).

![](images/28956d457f1d728f19e0cccc85aa8a9affff5630a76aa7f9d459bd205692f147.jpg)

![](images/cf50aa17ecc25e04d6c59ba75f17f92b8742d11ed0aef8ae9363bb69aca16317.jpg)

Figure 6: An illustration of the deduplication. The ResNet-50 classifier produces a confidence vector for each pig, indicating which class it most likely belongs to. Pigs b and d are both assigned to class 1. However, b has a lower maximum confidence for class 1 and can be reassigned to its second most likely class 4. For simplicity a scenario with a total of five pigs is assumed.  
![](images/9570b4401fd84fbb4855ec73ad8c494fb1c924cebda8c14688bf5d7f43396556.jpg)  
Figure 7: The matching variants. Overlap-based matching works well for consecutive frames (a) but might produce wrong results across larger gaps (b). Alternatively, keypoint skeletons (c) or the pigs’ orientation (d) can be used for matching instead. The additional information of pose and orientation helps to recover the correct match. Blue boxes, skeletons and arrows indicate a match.

The test dataset for evaluating BMCTrack-d consists of a total of 10 video clips, between 10 and 30 seconds in duration. The clips span multiple groups of pigs, show both pens and amount to 3500 frames in total. They exclusively consist of recordings from the more challenging side view angle and were annotated with bounding boxes and associated class IDs using CVAT. The clips were selected, such that they cover a range of scenarios which are relevant for tracking, as detailed in Table 3. This range of scenarios could not have been achieved by sampling from a single group, hence there was no dedicated, held-back test group. The loose categorisation into the three dificulty classes (low, medium, hard) in Table 3 indicates the expected dificulty of the test clips, dictated by properties such as speed of movement, average distance to camera and occlusions throughout the clips. The test clips are short because pigs tend to exhibit long stretches of stationary behaviour, interrupted by moments of interaction and bouts of frantic movement. It is precisely these latter situations that are most demanding on a tracker’s ability to retain tracks and hence provide the most information about a tracker’s performance. A visuali sation of the pigs’ movement trajectories per clip can be found in Appendix C, Fig. C.3.

![](images/df4594117f3f7731f7d5ab20babda95e4bf118e185aeaa019dc806026f5bff94.jpg)  
Figure 8: Details on BMCTrack-d’s algorithm. The processing of the first six frames of a video clip (a) is exemplified, as well as the resulting adjusted tracks (b) and the general workflow of the algorithm (c). For simplicity a scenario with a total of five pigs is assumed.

Table 2: The number of frames in the training and validation datasets for the YOLOv11 object detector.
<table><tr><td>dataset</td><td>total</td><td>side view</td><td>top view</td><td>pen A</td><td>pen B</td><td>instances total</td></tr><tr><td>training</td><td>567</td><td>454</td><td>113</td><td>396</td><td>171</td><td>5443</td></tr><tr><td>validation</td><td>30</td><td>24</td><td>6</td><td>15</td><td>15</td><td>272</td></tr></table>

All experiments were run on Ubuntu 20.04 and a NVIDIA GeForce RTX 3090 GPU (NVIDIA driver version 535.171.04., CUDA version: 12.2). The code was developed in Python 3.10 and the PyTorch framework.

## 3.3.2 Tracking metrics

The metric adopted for the evaluation of the tracking algorithms in this work is the higher order tracking accuracy (HOTA) (Luiten et al., 2021). HOTA can be decomposed into the sub-metrics localisation accuracy (LocA), association accuracy (AssA), and detection accuracy (DetA) for a more nuanced analysis of the tracking performance. A detailed description of the HOTA framework can be found in Appendix A.

## 3.3.3 Optimal configuration study

In the first set of experiments several variants of the proposed tracking algorithm are evaluated. To quantify the benefits of BMCTrack-d’s individual components, an ablation study is performed, discarding the TPCcheck (BMC-d), deduplication (BMCTrack) or both (BMC). For all variants the frame bufer length is set n = 4 and IoU used for matching. Additional experiments on the optimal frame bufer length, alternative matching variants and oriented bounding boxes can be found in Appendix B.

## 3.3.4 Algorithm evaluation

The second set of experiments pits BMCTrack-d against two established trackers and one novel tracker. Byte-Track (Zhang et al., 2022) is an eficient tracker, which achieves high tracking performance by introducing the idea of re-matching low-confidence detections in a second matching round. BoT-SORT (Aharon et al., 2022) refines the ByteTrack algorithm by a more accurate Kalman filter forecasting, as well as camera motion compensation. TrackTrack (Shim et al., 2025) introduces a track-centric matching strategy that assigns detections to existing tracks, rather than globally associate detections into tracks. Contrary to ByteTrack, which solely operates on location-based detection matching, BoT-SORT and TrackTrack ofer optional appearancebased ReID capabilities. To this end, an appearance feature vector is extracted from the bounding box of a detection with a deep neural network. A detection in a subsequent frame is matched, if the cosine similarity of the appearance feature vector with the new detection’s appearance features falls below a set threshold. The ap pearance feature vector is then updated by the features of the matched detection via the exponential moving average mechanism. This technique allows reinstating lost tracks, by using appearance clues to recognise that a recently disappeared and newly appearing individual are in fact the same. However, because of the continuous updating of the appearance vector this capability is typically limited to short time windows (e.g., 30 frames). The object detector is identical for all trackers. To facil itate a fair comparison, the strategy of clipping the set of predicted bounding boxes to ten per frame, employed in BMCTrack-d, is adopted for the other trackers as well. As these trackers are oblivious to the back mark classes and assign numeric IDs instead, for evaluation purposes it is assumed that their predictions in the first frame are perfect and a static mapping between the back mark classes and the assigned IDs was created. Finally, a runtime eficiency evaluation is performed, comparing the latency and throughput of the methods.

Table 3: Overview of video clips used for evaluation.
<table><tr><td>clip</td><td></td><td># frames duration (s)</td><td>difficulty</td><td>properties</td></tr><tr><td>penA_10s_5 250</td><td></td><td>10</td><td>medium</td><td>low movement; dispersed; moderate back mark visibility</td></tr><tr><td>penA_10s_6 250</td><td></td><td>10</td><td>low</td><td>moderate movement; dispersed; good back mark visibility</td></tr><tr><td>penA_10s_7 250</td><td></td><td>10</td><td>medium</td><td>moderate movement; dispersed; moderate back mark visibility</td></tr><tr><td>penA_10s_8 250</td><td></td><td>10</td><td>medium</td><td>moderate movement; increased distance to camera; moderate back mark visibility</td></tr><tr><td>penA_30s_2 750</td><td></td><td>30</td><td>high</td><td>very fast movement; dispersed; severely blurred and occluded back marks</td></tr><tr><td>penA_30s_5 750</td><td></td><td>30</td><td>high</td><td>very fast movement; dispersed; severely blurred and occluded back marks</td></tr><tr><td>penB_10s_2 250</td><td></td><td>10</td><td>low</td><td>moderate movement; close to camera; good back mark visibility</td></tr><tr><td>penB_10s_5 250</td><td></td><td>10</td><td>low</td><td>low movement; dispersed; good back mark vis- ibility</td></tr><tr><td>penB_10s_6 250</td><td></td><td>10</td><td>medium</td><td>low movement; increased distance to camera; moderate back mark visibility</td></tr><tr><td>penB_10s_7 250</td><td></td><td>10</td><td>medium</td><td>moderate movement; increased distance to camera; moderate back mark visibility</td></tr></table>

## 4 Results

## 4.1 Optimal configuration study

The results of the ablation study on algorithm variants are presented in Table 4. They show the baseline (BMCTrack-d) to outperform all variants (BMC, BMCd, BMCTrack). Section 5 discusses the results in more detail.

Table 4: The results of the ablation study. Best results in bold.
<table><tr><td>method</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>BMC</td><td>0.8864</td><td>0.4374</td><td>0.7388</td><td>0.5633</td></tr><tr><td>BMC-d</td><td>0.8864</td><td>0.5667</td><td>0.8591</td><td>0.6901</td></tr><tr><td>BMCTrack</td><td>0.8864</td><td>0.4056</td><td>0.7232</td><td>0.5261</td></tr><tr><td>BMCTrack-d</td><td>0.8864</td><td>0.7048</td><td>0.8591</td><td>0.7714</td></tr></table>

## 4.2 Algorithm evaluation

Table 5 presents a comparison of the test set performances between BMCTrack-d, ByteTrack, BoT-SORT and TrackTrack. It shows BMCTrack-d to outperform the other trackers, often by a significant margin. While ByteTrack remains non-competitive on all scores, BoT-SORT performs nearly identically to BMCTrack-d w.r.t. the localisation (LocA) and detection (DetA) but stays behind on the association score (AssA). TrackTrack further closes the gap, especially on enabling the ReID capabilities. Tables 6 and 7 show the performances on each of the test set clips separately for BMCTrack-d and TrackTrack, respectively. Both trackers’ performances show a similar trend on most clips but diverge significantly on penA\_30s\_2. The results are examined in detail in Section 5. Table 8 shows the results of the runtime eficiency evaluation.

Table 5: The results of the tracking algorithm evaluation. Best results in bold.
<table><tr><td>method</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>ByteTrack</td><td>0.8531</td><td>0.6069</td><td>0.7915</td><td>0.6680</td></tr><tr><td>BoT-SORT</td><td>0.8843</td><td>0.6525</td><td>0.8581</td><td>0.7173</td></tr><tr><td>BoT-SORT-ReID</td><td>0.8841</td><td>0.5847</td><td>0.8576</td><td>0.6803</td></tr><tr><td>TrackTrack</td><td>0.8877</td><td>0.6692</td><td>0.8583</td><td>0.7356</td></tr><tr><td>TrackTrack-ReID</td><td>0.8877</td><td>0.6970</td><td>0.8586</td><td>0.7611</td></tr><tr><td>BMCTrack-d</td><td>0.8864</td><td>0.7048</td><td>0.8591</td><td>0.7714</td></tr></table>

Table 6: Per-clip results of BMCTrack-d on the test set.
<table><tr><td>clip</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>penA_10s_5</td><td>0.7694</td><td>0.4013</td><td>0.7635</td><td>0.5535</td></tr><tr><td>penA_10s_6</td><td>0.9152</td><td>0.8855</td><td>0.8855</td><td>0.8855</td></tr><tr><td>penA_10s_7</td><td>0.8957</td><td>0.8503</td><td>0.8611</td><td>0.8557</td></tr><tr><td>penA_10s_8</td><td>0.9064</td><td>0.4917</td><td>0.8938</td><td>0.6624</td></tr><tr><td>penA_30s_2</td><td>0.8676</td><td>0.5162</td><td>0.8156</td><td>0.6486</td></tr><tr><td>penA_30s_5</td><td>0.8808</td><td>0.4686</td><td>0.8428</td><td>0.6282</td></tr><tr><td>penB_10s_2</td><td>0.9003</td><td>0.8832</td><td>0.8854</td><td>0.8843</td></tr><tr><td>penB_10s_5</td><td>0.9149</td><td>0.8969</td><td>0.8969</td><td>0.8969</td></tr><tr><td>penB_10s_6</td><td>0.9020</td><td>0.8699</td><td>0.8819</td><td>0.8759</td></tr><tr><td>penB_10s_7</td><td>0.9122</td><td>0.7841</td><td>0.8645</td><td>0.8233</td></tr><tr><td>mean</td><td>0.8864</td><td>0.7048</td><td>0.8591</td><td>0.7714</td></tr></table>

Table 7: Per-clip results of TrackTrack-ReID on the test set.
<table><tr><td>clip</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>penA_10s_5</td><td>0.7734</td><td>0.6416</td><td>0.7603</td><td>0.6981</td></tr><tr><td>penA_10s_6</td><td>0.9157</td><td>0.8325</td><td>0.8853</td><td>0.8584</td></tr><tr><td>penA_10s_7</td><td>0.8973</td><td>0.8356</td><td>0.8620</td><td>0.8487</td></tr><tr><td>penA_10s_8</td><td>0.9066</td><td>0.8944</td><td>0.8944</td><td>0.8944</td></tr><tr><td>penA_30s_2</td><td>0.8699</td><td>0.1688</td><td>0.8136</td><td>0.3704</td></tr><tr><td>penA_30s_5</td><td>0.8824</td><td>0.4412</td><td>0.8415</td><td>0.6093</td></tr><tr><td>penB 3_10s_2</td><td>0.9007</td><td>0.8631</td><td>0.8872</td><td>0.8750</td></tr><tr><td>penB 10s_5</td><td>0.9150</td><td>0.8981</td><td>0.8981</td><td>0.8981</td></tr><tr><td>penB _10s_6</td><td>0.9025</td><td>0.7282</td><td>0.8825</td><td>0.8015</td></tr><tr><td>penB 3_10s_7</td><td>0.9131</td><td>0.6661</td><td>0.8614</td><td>0.7574</td></tr><tr><td colspan="3">mean 0.8876 0.6970</td><td>0.8586</td><td>0.7611</td></tr></table>

Table 8: The results of the runtime eficiency evaluation. All experiments were run on a NVIDIA GeForce RTX 3090 GPU.
<table><tr><td>method</td><td>mean latency (ms)</td><td>fps</td></tr><tr><td>ByteTrack</td><td>16.79</td><td>59</td></tr><tr><td>BoT-SORT-ReID</td><td>31.11</td><td>32</td></tr><tr><td>TrackTrack-ReID</td><td>31.88</td><td>31</td></tr><tr><td>BMCTrack-d (of which TPC + dedup.)</td><td>143.48 (1.12)</td><td>7</td></tr></table>

## 5 Discussion

## 5.1 Strengths and weaknesses of BMCTrack-d

While some existing trackers can be described as tracking with ReID capabilities (Aharon et al., 2022), BMCTrack-d is most accurately described as ReID with tracking capabilities. For individual-level monitoring, it is of utmost importance to know the identities of all animals in the scene for as much of the time as possible. Given that the main goal of the algorithm is to assign a class ID per frame, it does not strictly enforce positionally consistent tracks. This means that if in frame t class 1 is (wrongly) detected in the opposite corner of the pen from where it was detected in frame t-1, this does not lead to the initialisation of a new track. While this “identification first” approach leads to less strictly enforced positional consistency, it also facilitates self-correcting, which is the biggest strength of the proposed algorithm. BMCTrack-d might make mistakes more frequently because it heavily relies on a fallible classifier, but, because of the continuous nature of the classification, has the ability to self-correct these mistakes later. In simple words, traditional trackers might be right for a long time and then wrong for a long time, while BMCTrack-d might make mistakes earlier and more frequently, but correct the mistakes along the way. Figure 9 illustrates this diference for pig dot horizontal line in test clip penA\_30s\_2. BMCTrack-d (left image) repeatedly misdetected the pig in phases of crowding (top right corner) and fast movement (top right to bottom left diagonal track) but continuously corrected these mistakes. BoT-SORT (right image) was more robust to fast movement, however on losing the track in the crowded phase (top right corner) it was unable to recover the pig’s identity. While BoT-SORT’s ability to re-activate a lost track is limited to short intervals (30 frames by default), BMCTrack-d’s ability to re-identify an individual, in principle, is temporally unlimited. BMCTrack-d’s self-correction ability is absolutely crucial in the face of severe occlusions and motion blur, which frequently lead to lost tracks, and allows BMCTrack-d to outperform traditional trackers in the challenging scenario of this study.

The use of a separate classifier model comes at the expense of runtime eficiency. However, it is not uncommon in animal monitoring to operate at low frame rates (e.g., Psota et al. (2020)). At a frame rate of e.g., 6 fps, BMCTrack-d runs at real-time speed on the hardware used in this study. It is also quite robust to low frame rates, as shown in Table 9, which compares the performance of BMCTrack-d to TrackTrack-ReID at 6 fps. BMCTrack-d’s performance remains more stable (-2.27% HOTA) than TrackTrack-ReID’s (-9.35% HOTA) compared to the original frame rate of 25 fps.

Table 9: Comparison of the test set performance of BMCtrack-d and TrackTrack-ReID at 6 fps.
<table><tr><td>method</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>TrackTrack-ReID</td><td>0.8869</td><td>0.6210</td><td>0.8260</td><td>0.6676</td></tr><tr><td>BMCTrack-d</td><td>0.8865</td><td>0.6687</td><td>0.8589</td><td>0.7487</td></tr></table>

## 5.2 BMCTrack-d modules

Extensive experimentation showed both the TPC check as well as deduplication to play important roles in BMCTrack-d’s performance. Deduplication could be shown to have clear benefits on its own (+12.93% AssA, BMC vs. BMC-d), which is not the case for the TPC check (-3.18% AssA, BMC vs. BMCTrack). Interestingly, the TPC check improved the performance only in combination with deduplication. This can be explained by viewing deduplication as a way of breaking out of a sequence of wrong predictions that the TPC check got stuck in. The TPC check works well for correcting singular wrong predictions but cannot correct sequences of wrong predictions (e.g., caused by a pig that remains partially occluded over an extended amount of time). These, however, are likely to cause class ID collisions at some point and will be corrected by deduplication, giving the TPC check a chance to reset. Figure 10 illustrates both ideas on test clip penA\_30s\_5. In combination, TPC check and deduplication lead to a significant improvement in association (+26.74% AssA, BMC vs. BMCTrack-d). As deduplication eliminates duplicate predictions, there was also an improvement in detection (+12.03% DetA).

## 5.3 Comparison to other trackers

BoT-SORT updates ByteTrack in 3 ways: 1) enhanced Kalman filter, 2) camera motion correction and 3) ReID capabilities. As the (optional) ReID capabilities were investigated separately (BoT-SORT-ReID) and the camera in this study’s setup was static, the improvement that BoT-SORT achieved over ByteTrack must be explained by the enhanced Kalman filter, which is plausible, because it alleviates the challenge that fast movement poses. Unfortunately, BMCTrack-d cannot benefit from Kalman filter forecasting, because the occasional jumps in location that BMCTrack-d allows are incompatible with the linear motion that Kalman filters assume. TrackTrack improves both these earlier tracking algorithms by assigning detections to existing trajectories from the track’s perspective rather than solving a global assignment problem. This strategy is aimed at improving tracking performance in crowded scenes with a high number of occlusions and is especially relevant for pig tracking. The evaluation shows this algorithmic improvement to have a noticeable effect on the tracking performance, especially in combination with appearance-based reID (+11.23% AssA, TrackTrack-ReID vs. BoT-SORT-ReID). Despite this innovation BMCTrack-d still outperforms TrackTrack on the tracking metrics (+1.03% HOTA), while simultaneously solving the reID task by not just tracing but recognising the individual pigs.

![](images/6e48be1fc28110f63bc41d25734e685a2f915d9dc19808894f95f91758123f16.jpg)  
Figure 9: Qualitative comparison between BMCTrack-d (left) and BoT-SORT (right) for tracking a single pig. The blue dots show the detected locations of dot line horizontal throughout the video. The black dots in the right image signalize the track ID changing, i.e., BoT-SORT losing the track.

The evaluation on the individual clips, shown in Tables 6 and 7, reveal a noticeable divergence in the performance of BMCTrack-d and TrackTrack-ReID on two of the test clips, penA\_30s\_2 (+27.82% HOTA, BMCTrack-d vs. TrackTrack-ReID) and penA\_10s\_8 (-23.2% HOTA, BMCTrack-d vs. TrackTrack-ReID). Deeper analysis shows that these cases align with the previously discussed strengths and weaknesses of BMCTrack-d. The clip penA\_30s\_2 shows a sequence of very rapid, erratic movement across the whole pen with high levels of motion blur, as well as low resolution pigs and severe occlusions whenever the pigs move to the far end of the pen. Fig. 11c shows that the performances of the trackers correlate with the occlusion severity (measured by the performance of the pig detector). To realise true individual-level monitoring, a tracker must be able to provide correct identity labels on the other side of such bouts of movement. Figure 11a shows that BMCTrack-d is capable of doing so. While TrackTrack-ReID is able to re-identify lost individuals after brief occlusions (Fig. 11b, indicated by the spikes in the curve), longer occlusions lead to permanently lost tracks. BMCTrack-d’s ability for reID is temporally unlimited, allowing it to fully obtain correct identity labels even after periods of severe occlusions.

Conversely, in test clip penA\_10s\_8 BMCTrack-d consistently confuses two individuals (Fig. 12a) Given that there are no notable occlusions (Fig. 12c), the error must have a diferent cause. Qualitative analysis shows that the error most likely stems from an anomaly in the appearance of the back marks at the beginning of the clip, in which specific circumstances caused back marks vertical line and v to resemble each other. Imprecision in the drawing of the back mark vertical line caused it to resemble back mark v. At the same time, v, at a specific view angle, resembled vertical line, causing BMCTrackd to flip the identity predictions, unable to correct this mistake for the rest of the video. This scenario shows that for the algorithmic use of back marks they must be carefully designed to minimise potential collisions (Brunner et al., 2026a). TrackTrack-ReID was unaffected by this and achieves perfect tracking throughout the whole clip. A quantitative analysis of the clips’ properties (resolution, occlusion, motion blur) can be found in Appendix C, Figures C.1 and C.2. On all other test clips the results of both methods are comparable. Figure 13 shows qualitative results of BMCTrack-d on test clips penA\_10s\_6 and penA\_10s\_7.

## 5.4 Scope

This study shows the potential of back marks for reID in challenging camera settings. Especially this robustness to non-idealised camera settings addresses an important gap in existing literature regarding practical application. However, while the non-idealised camera setting gives the results of this study high practical relevance, the experimental setup diverges from real-world farm settings in other important ways. At present, some of the experimental conditions in this study make BMCTrack-d more suitable for research settings and might limit its applicability to real-world farm settings. These limitations are discussed in detail in Section 5.5.

![](images/25244f835c4eb67c83c27be33603e24847d384b0c53ffbb5da6a03bc85cc3338.jpg)  
Figure 10: Illustration of the interplay between the temporal prediction consistency (TPC) check and the deduplication. The left image shows pig dot dot was tracked with high accuracy for a long stretch and single misclassifications (small red circles) were corrected promptly by the TPC check. Then, on fast movement, dot dot was misclassified as line line horizontal (red ellipse). The right image shows that this misclassification was corrected only after the pig slowed down (red ellipse). Presumably, the reduced motion blur and the resulting improved visibility of the back mark increased the classifiers confidence in the correct class, which was then reassigned by the deduplication step.

## 5.5 Limitations

## 5.5.1 Runtime eficiency

Farm settings might require running the tracking and all downstream tasks (such as behaviour recognition) directly on the camera stream, to allow real-time (online) intervention on detection of relevant behaviours. Research settings (such as observational studies) impose less strict requirements on algorithm eficiency, because the recording of the animals and the analysis of the data are two separate steps, of which the former might be finished before the latter is even started. The proposed tracking algorithm was predominantly developed for research settings. Therefore, runtime optimisation was not a priority and the algorithm in its current form might not be suited for online use. The back mark classification model architecture is a natural starting point for eficiency improvements, given that it is responsible for much of the processing time. Optimizing this architecture is left for future work.

## 5.5.2 Closed-set settings

BMCTrack-d was designed for closed-set applications, in which the set of possible identities is known beforehand, and these identities need to be continuously assigned to the correct individuals. This is a reasonable assumption for observational studies but might be prohibitive for many real-world farm settings. At present, BMCTrack-d is not able to dynamically handle the introduction of new back marks, that are not in the set of known identities. If a pen of size 10 is selected for individual-level monitoring and the classifier trained on 10 unique back marks, the post hoc addition of an 11th pig would require retraining the classifier. Conversely, the removal of pigs does not pose an issue. From the perspective of a tracker, situations in which not all known pigs are present arise regularly, namely whenever occlusions occur. These scenarios are covered by the test set, which includes clips with severe occlusions (e.g., penA\_30s\_2 ) and the evaluation shows that BMCTrack-d can handle such scenarios. The permanent removal of a pig is out of scope for the current study but could be handled based on its efects, which would be twofold: 1) the mean number of pig detections would drop (e.g., from 10 to 9) and 2) the afected back mark would appear less frequently in the predictions (only in case of mistakes). Thus, this situation could be handled automatically, by 1) detecting the removal via a drop in the mean number of detections, and 2) removing the afected back mark from the list of known back marks (i.e., ignoring its prediction). If the pig is reintroduced to the pen at a later point in time, the process can be reversed, i.e., detecting an increase in the mean number of detections and reactivating the back mark in the list. A practical evaluation of this is left for future work.

a)  
![](images/b34b94e7cb41452263ccfacf70c4a42645e069d51da0620e6d8a34ea196f1d03.jpg)

1  
![](images/ef06098b5b0572e01bd0269984e5d451b10e694162a2c4d91cc35226064f5437.jpg)

2  
b)  
Number of correct IDs over time (TrackTrack-RelD)  
![](images/43a6e62b945f8a090a0312758d987262b58b31040cda844b6560cc332065f828.jpg)

![](images/e3134d2bea5214cfb3aa7159051a6f08bd07b090c33a3c11fab8d8d4a9eeac9d.jpg)  
c)

3  
![](images/95d3f11479b886ab3487acf18900360231c02fc5f9eb40728e62db1881cc691e.jpg)

Number of detections over time (YOLOv11)  
![](images/e854e9a3fb9f0079b6a804c405ced03d391cdce19c6aa3e1fdf4068db81fcd9b.jpg)

4  
![](images/4dc510a214c8d637d4f51490f3e6fbd8c9f67927a91ba80dc71204a8bbcc1bc8.jpg)  
Figure 11: Per-frame comparison of BMCTrack-d (a) and TrackTrack-ReID (b) on test clip penA\_30s\_2. Lost identities correlate with phases of high occlusion (c) for both methods. TrackTrack-ReID is capable of re-identifying lost individuals in cases of brief occlusions (indicated by the spikes), but unable to recover them in case of lasting occlusions. BMCTrack-d is able to fully recover all identities whenever the occlusion severity decreases.

![](images/2849e9be4c36f0dcca04ef36cdbc6a530f6f43bd7431883cd5ce0591b3d1ed4a.jpg)

a)  
![](images/36926d6b7a5391a5594db89e2efbde443b0fb4f239b9312cef3c8578df5955e8.jpg)  
d)  
b)

![](images/2ef7ef7bc5ad33d6f08199996a2458746baaeca35bb93c926136ca964102abb1.jpg)  
c)

![](images/d75285b511b51d4052db79f5087363c97cc1c97cb549bdf9bea68dbbd4348e8c.jpg)

<table><tr><td>true class</td><td>pred. class</td></tr><tr><td>vertical_line</td><td>V</td></tr></table>

<table><tr><td>true class</td><td>pred. class</td></tr><tr><td>V</td><td>vertical_line</td></tr></table>

Figure 12: Per-frame comparison of BMCTrack-d (a) and TrackTrack-ReID (b) on test clip penA\_10s\_8. TrackTrack ReID perfectly tracks all individuals. BMCTrack-d confuses two individuals in absence of occlusions (c). Qualitative analysis shows that an imprecisely drawn back mark in combination with a specific view angle causes these back marks to resemble each other (d).

![](images/562772d47f079fff18fa8b894964c191925f384248f0504f1cde4370171ed8fd.jpg)  
Figure 13: Qualitative results of BMCTrack-d on two test clips, penA\_10s\_6 (left) and penA\_10s\_7 (right).

## 5.5.3 Back mark application

This study showed that back marks are a viable means for pig reID in challenging settings. However, the use of back marks comes with important practical challenges. In this study, the back marks were applied manually and refreshed 1-2 times a week. While this is a feasible option for research, it might be prohibitive for many real-world farm settings, which might require automatic application of the back marks. In some practical farm settings, e.g., pig units for genetic selection and performance evaluation, manual application of back marks might be feasible as well. Genetic evaluation is a highly resource-intensive operation compared to standard pork production, where pigs are weighed individually or ultrasound scans are performed routinely on live animals to measure backfat thickness and muscle depth. Camera based individual phenotyping might allow breeding organizations to select for complex traits recorded continuously, objectively and in high-frequency. These traits e.g., aggression in pigs, were previously impossible to be measured accurately. Higher expenses on such units, which might be related to individual back marking, might be justified as these units function as centralized hubs for accelerating genetic progress across the industry. Manual application of individual back marks on standard commercial farms might be less feasible. However, there are commercial products, which ofer automated solution e.g., Colortek (Fancom, Panningen, the Netherlands) for sow spray marking inside the feeding stations, which supports the farmers in identifying sows in heat or to provide information on their health status. Further development and adaptations of these systems might support practicality of using back marks for automated identification of pigs with BMCTrack-d. Development of BMCTrack-d in a pen with 10 pigs supports its practical use in conventional European finishing systems which commonly use small group sizes with 10 pigs e.g., in the Netherlands or Sweden. However, validation in larger group sizes is needed.

## 6 Conclusion

Reliable reID and tracking are imperative for individuallevel pig monitoring. This study proposed a novel tracking algorithm, which addresses two important challenges towards this goal, the dificulty of diferentiating individuals of uniform appearance and the dificulty of tracking in challenging situations caused by experimental conditions and pig behaviour. BMCTrack-d demonstrates the merit of back marks for diferentiating the pigs and shows high robustness when confronted with practical challenges like occlusions, motion blur and low-resolution recordings. Especially its ability to recover lost identities, which is crucial for downstream tasks like behaviour recognition, sets it apart from existing solutions. On a challenging side-view dataset, BMCTrack-d outperformed BoT-SORT-ReID and TrackTrack-ReID by 9.11% and 1.03% HOTA, respectively, while simultaneously solving the reID task by not just tracing but recognising the individual pigs. Subsequent studies should broaden the spectrum of possible applications of BMCTrack-d, by improving runtime eficiency via more specialised back mark classifiers, reviewing and optimising the back mark design for easier identification, and evaluating automated back mark application technologies.

## Acknowledgments

This research was funded in whole or in part by the Austrian Science Fund (FWF)

[https://doi.org/10.55776/DFH34]. For open access purposes, the author has applied a CC BY public copyright license to any author-accepted manuscript version arising from this submission. The data used in this study originates from the "Let me out" project, funded by the Austrian Science Fund (FWF), project I 6488-B [https://doi.org/10.55776/I6488]. Further, we would like to thank Janina Weißenborn and Stefan Kupfer from the University of Veterinary Medicine Vienna for help with the data annotation and technical support, respectively.

## Author Contributions

David Brunner: Conceptualization, Data curation, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft. Maciej Oczak: Conceptualization, Funding Acquisition, Methodology, Resources, Writing – review & editing. Marie Bordes: Data curation, Resources, Writing – review & editing. Jean-Loup Rault: Funding acquisition, Resources, Writing – review & editing. Stephan M. Winkler: Funding acquisition, Supervision, Writing – review & editing. Viktoria Dorfer: Funding acquisition, Project administration, Supervision, Writing – review & editing.

## Ethics statement

All methods and animal use were approved by the Animal Ethics Committee of the University of Veterinary Medicine, Vienna (reference number 2024-0.026.412), and carried out in accordance with Good Scientific Practice guidelines and national legislation.

## A Tracking metrics

The metric adopted for the evaluation of the tracking algorithms in this work is the higher order tracking accuracy (HOTA) (Luiten et al., 2021). It improves upon the most important traditional tracking metrics, multi-object tracking accuracy (MOTA) (Bernardin and Stiefelhagen, 2008) and identification F1 (IDF1) (Ristani et al., 2016), in many meaningful ways, which are criticised for putting too much emphasis on detection and association, respectively. To calculate HOTA, first, the detections of the tracker need to be matched to the ground truth bounding boxes. An important distinction in tracking evaluation is that between matches and associations. A match describes a situation in which the IoU between a detected bounding box and a ground truth bounding box in a frame exceeds a defined threshold. An association describes when the detected and the ground truth bounding box of a match also have the same (class) ID. The best pairing of detections and ground truths is searched, so that across the whole video the best final HOTA score is achieved. This only implicitly enforces connected trajectories – the probability that the best overall score is attained by matching the same ground truth to detections with different (class) IDs in subsequent frames is low, because for every match (in a given frame) the consequences for the whole video are checked. In addition to this association score, every match also has a localisation similarity, defined by the IoU between the detections. The Hungarian algorithm is used to maximise 1) the total number of matches, 2) the mean association score and 3) the mean localisation similarity. Once the best pairing is found, $H O T A _ { \alpha }$ is calculated as:

$$
\begin{array} { c } { { H O T A _ { \alpha } = \sqrt { \displaystyle \frac { \sum _ { c \in \{ T P \} } A ( c ) } { | T P | + | F P | + | F N | } } } } \\ { { A ( c ) = \displaystyle \frac { | T P A ( c ) | } { | T P A ( c ) | + | F P A ( c ) | + | F N A ( c ) | } } } \end{array}\tag{1}
$$

where c is a given match and $| T P |$ is the total number of matches in the optimised pairing. |F P| and |FN| are the total number of resulting false positives and false negatives, respectively. |T P A| is the number of true associations, i.e. the number of correct ID matches between ground truths and detections of the same ID (along the trajectory) that result from this match. |FPA| and |FNA| are the number of false positive associations and false negative associations, respectively. A great advantage of HOTA is that it can be decomposed into sub-metrics that allow evaluating a tracker’s performance on each of the capabilities involved in successful tracking separately, namely detection accuracy (DetA) and association accuracy (AssA):

$$
D e t A _ { \alpha } = \frac { | T P | } { | T P | + | F N | + | F P | }\tag{2}
$$

$$
A s s A _ { \alpha } = \frac { 1 } { | T P | } \sum _ { c \in \{ T P \} } A ( c )\tag{3}
$$

where

$$
H O T A _ { \alpha } = \sqrt { D e t A _ { \alpha } \cdot A s s A _ { \alpha } }\tag{4}
$$

The localisation accuracy (LocA) can be calculated separately, as:

$$
L o c { A _ { \alpha } } = \frac { 1 } { | T P _ { \alpha } | } \sum _ { c \in \{ T P _ { \alpha } \} } S ( c )\tag{5}
$$

(5) where S is the localisation similarity, measured as the IoU between detection and ground truth. $H O T A _ { \alpha }$ is calculated for every localisation similarity threshold αϵL, and the average gives the final score:

$$
\begin{array} { c } { { H O T A = \displaystyle \frac { 1 } { | L | } \sum _ { \alpha \in L } H O T A _ { \alpha } } } \\ { { L = \{ 0 . 0 5 , 0 . 1 , \ldots , 0 . 9 , 0 . 9 5 \} } } \end{array}\tag{6}
$$

## B Additional results

## B.1 Extended optimal configuration study

This section discusses additional experiments on the optimal configuration of BMCTrack-d. First, the optimal frame bufer length for the TPC check is assessed in the range [2, 6]. Next, alternative matching variants are evaluated, namely BMCTrack-d-OKS, which uses keypoint-skeleton-based matching, as well as BMCTrack-d-KP, which supplements IoU-based matching with orientation information derived from two keypoints. The results of the study on the optimal frame bufer length are summarised in Table B.1.1 and show number of last frames $n = 4$ to be the best setting. Table B.1.2 shows the results of the enhanced matching variants study.

Table B.1.1: The results of the frame bufer length study. Best results in bold.
<table><tr><td>frame buffer length</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>2</td><td>0.8864</td><td>0.6737</td><td>0.8591</td><td>0.7529</td></tr><tr><td>3</td><td>0.8864</td><td>0.6821</td><td>0.8591</td><td>0.7570</td></tr><tr><td>4</td><td>0.8864</td><td>0.7048</td><td>0.8591</td><td>0.7714</td></tr><tr><td>5</td><td>0.8864</td><td>0.6608</td><td>0.8591</td><td>0.7459</td></tr><tr><td>6</td><td>0.8864</td><td>0.6590</td><td>0.8591</td><td>0.7448</td></tr></table>

The frame bufer length represents a trade-of between detection matching and prediction correction. The longer the frame bufer, the more likely the matching fails. Matching detections between frame t and frame t-2 is more likely to produce correct matches than between frame t and frame t-5, because the potential ofset through movement is smaller. At the same time, the majority vote strategy of the TPC check benefits from longer sequences of class predictions, because, assuming that wrong predictions are in the minority, this minority becomes clearer in longer sequences. A frame bufer length of $n = 4$ seems to optimise this trade-of, as shown in Table B.1.1.

Table B.1.2: The results of the enhanced matching variants study.
<table><tr><td>method</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>BMCTrack-d</td><td>0.8864</td><td>0.7048</td><td>0.8591</td><td>0.7714</td></tr><tr><td>BMCTrack-d-OKS</td><td>0.8864</td><td>0.6608</td><td>0.8591</td><td>0.7461</td></tr><tr><td>BMCTrack-d-KP</td><td>0.8864</td><td>0.6879</td><td>0.8591</td><td>0.7628</td></tr></table>

Neither substituting IoU-based matching with OKSbased matching (BMCTrack-d-OKS), nor supplementing the IoU-based matching with keypoint-based orientation information (BMCTrack-d-KP) could improve performance. A likely explanation for this is that both require near-perfect pose estimation to work as intended. While the pose estimation model used in this work reaches high accuracy, it sufers from occasional pose inversions, in which the keypoint skeleton flips along a pig’s body for single frames, as depicted in Fig. B.1.1a. These flips completely disrupt the matching process and demonstrate the fragility of keypoint-based tracking. To study the theoretical benefits of these methods, ground truth keypoint skeletons are required, which were not available for the data in this study. This evaluation is left for future work.

## B.2 Oriented bounding boxes

Odo et al. (2025) have reported improvements in tracking and reID on using oriented instead of axis-aligned bounding boxes. To establish the benefit of oriented bounding boxes for BMCTrack-d an instance of the YOLOv11 object detector for oriented bounding box detection is trained. The oriented bounding box labels are created by passing the training, validation and test datasets (described in Section 3.3.1) to an instance of the Segment Anything Model (Kirillov et al., 2023) and fitting rectangles on the generated segmentations, which represents a slightly simplified version of the strategy proposed in Odo et al. (2024). Given that the generated oriented bounding boxes can contain imprecisions and errors, the test set was manually reviewed and frames with deficient bounding boxes were filtered out. YOLOv11-OBB, trained as described above, reached 99.27% mAP@0.5 (YOLOv11: 99.42%) on the validation set. For a fair comparison of the resulting BMCTrack-d-OBB to the baseline BMCTrack-d, the test set of the latter was reduced to the same set of frames. The more precisely fitting oriented bounding boxes also harbour benefits for classification, because they show less background and are less likely to include multiple pigs. For this reason, a separate instance of the image classifier was trained on oriented bounding boxes for this experiment, reaching 94.86% accuracy (axisaligned classifier: 91%) on the validation set. Table B.2.1 shows the results.

Table B.2.1: The results of the oriented bounding box study. Best results in bold.
<table><tr><td>method</td><td>LocA</td><td>AssA</td><td>DetA</td><td>HOTA</td></tr><tr><td>BMCTrack-d</td><td>0.8889</td><td>0.7366</td><td>0.8744</td><td>0.7963</td></tr><tr><td>BMCTrack-d-OBB</td><td>0.8583</td><td>0.6546</td><td>0.8151</td><td>0.7250</td></tr></table>

Contrary to existing work (e.g., Odo et al. (2025)), in this study, the use of oriented bounding boxes did not lead to improved tracking performance. Table B.2.1 shows the drop in performance to be most noticeable for association (AssA -8.20%). A possible explanation is that axis aligned bounding boxes promote the backwards matching in the TPC check, because the, on average, more expansive bounding box areas lead to greater overlaps and make matches more robust to movement, as illustrated in Fig. B.1.1b. For the more precise oriented bounding boxes, matching across frames frequently fails, especially if the pigs move rapidly.

## References

Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. BoT-SORT: Robust Associations Multi-Pedestrian Tracking, July 2022. URL http://arxiv.org/abs/2206. 14651. arXiv:2206.14651 [cs].

Keni Bernardin and Rainer Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. EURASIP Journal on Image and Video Processing, 2008:1–10, 2008. ISSN 1687-5176, 1687-5281. doi: 10.1155/2008/ 246309. URL http://jivp.eurasipjournals.com/ content/2008/1/246309.

Justin Brooks. COCO Annotator, 2019. URL https: //github.com/jsbroks/coco-annotator/.

David Brunner, Marie Bordes, Elisabeth Mayrhuber, Stephan M. Winkler, Viktoria Dorfer, and Maciej Oczak. Insights on back marking for the automated

a)  
![](images/c4608f927d7c31e31e70c797ea93f980a7d70a3d16770a145b8319eeb90aec97.jpg)

b)  
![](images/7712f6a886a6a09e706676ad5ed5258a507d9827ab86e58b970c42716b2ea389.jpg)  
Figure B.1.1: Illustration of the skeleton flip issue of the pose estimation model (a) and the reduced overlap of oriented bounding boxes in consecutive frames (b).

identification of animals, 2026a. URL https:// arxiv.org/abs/2603.25535. \_eprint: 2603.25535.

David Brunner, Marie Bordes, Elisabeth Mayrhuber, Stephan M. Winkler, Viktoria Dorfer, and Maciej Oczak. Skeleton integrity: A method for the eficient fine-tuning of pose estimation models for pigs. Biosystems Engineering, 264:104380, 2026b. ISSN 1537- 5110. doi: https://doi.org/10.1016/j.biosystemseng. 2025.104380. URL https://www.sciencedirect. com/science/article/pii/S1537511025003162.

Caroline Clouard, Auriane Foreau, Sébastien Goumon, Céline Tallet, Elodie Merlot, and Rémi Resmond. Evidence of stable preferential afiliative relationships in the domestic pig. Animal Behaviour, 213:95–105, 2024. ISSN 0003-3472. doi: https://doi.org/10.1016/j.anbehav.2024.04.009. URL https://www.sciencedirect.com/science/ article/pii/S0003347224001222.

Yunhao Du, Zhicheng Zhao, Yang Song, Yanyun Zhao, Fei Su, Tao Gong, and Hongying Meng. StrongSORT: Make DeepSORT Great Again. IEEE Transactions on Multimedia, 25:8725–8737, 2023. ISSN 1941-0077. doi: 10.1109/TMM.2023. 3240881. URL https://ieeexplore.ieee.org/ document/10032656/?arnumber=10032656. Conference Name: IEEE Transactions on Multimedia.

Eduardo Fernández-Carrión, Marta Martínez-Avilés, Benjamin Ivorra, Beatriz Martínez-López, Ángel Manuel Ramos, and José Manuel Sánchez-Vizcaíno. Motion-based video monitoring for early detection of livestock diseases: The case of African swine fever. PloS one, 12(9):e0183793, 2017.

RH Foote. Estrus detection and estrus detection aids. Journal of Dairy Science, 58(2):248–256, 1975.

Maik Fruhner, Heiko Tapken, and Henning Müller. Re-Identifikation markierter Schweine mit Computer Vision und Deep Learning. In 42. GIL-Jahrestagung, Künstliche Intelligenz in der Agrar-und Ernährungswirtschaft, pages 99–104. Gesellschaft für Informatik eV, 2022.

Haiming Gan, Mingqiang Ou, Endai Huang, Chengguo Xu, Shiqing Li, Jiping Li, Kai Liu, and Yueju Xue. Automated detection and analysis of social behaviors among preweaning piglets using key pointbased spatial and temporal features. Computers and Electronics in Agriculture, 188:106357, September 2021. ISSN 01681699. doi: 10.1016/j.compag.2021. 106357. URL https://linkinghub.elsevier.com/ retrieve/pii/S0168169921003744.

Haiming Gan, Chengguo Xu, Wenhao Hou, Jingfeng Guo, Kai Liu, and Yueju Xue. Spatiotemporal graph convolutional network for automated detection and analysis of social behaviours among pre-weaning piglets. Biosystems Engineering, 217:102–114, May 2022. ISSN 15375110. doi: 10.1016/j.biosystemseng. 2022.03.005. URL https://linkinghub.elsevier. com/retrieve/pii/S1537511022000575.

Ruopeng Gao, Ji Qi, and Limin Wang. Multiple object tracking as id prediction. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27883–27893. IEEE, 2025.

Yue Gao, Kai Yan, Baisheng Dai, Hongmin Sun, Yanling Yin, Runze Liu, and Weizheng Shen. Recognition of aggressive behavior of group-housed pigs based on CNN-GRU hybrid model with spatio-temporal attention mechanism. Computers and Electronics in Agriculture, 205:107606, 2023. ISSN 0168-1699. doi: https://doi.org/10.1016/j.compag.2022.107606. URL https://www.sciencedirect.com/science/ article/pii/S0168169922009140.

Qinghua Guo, Yue Sun, Clémence Orsini, J. Elizabeth Bolhuis, Jakob de Vlieg, Piter Bijma, and Peter H. N. de With. Enhanced camerabased individual pig detection and tracking for smart pig farms. Computers and Electronics in Agriculture, 211:108009, 2023. ISSN 0168-1699. doi: https://doi.org/10.1016/j.compag.2023.108009. URL https://www.sciencedirect.com/science/ article/pii/S0168169923003976.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition.

In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, Las Vegas, NV, USA, June 2016. IEEE. ISBN 978-1-4673- 8851-1. doi: 10.1109/CVPR.2016.90. URL http: //ieeexplore.ieee.org/document/7780459/.

Mohammadamin Kashiha, Claudia Bahr, Sanne Ott, Christel PH Moons, Theo A Niewold, Frank O Ödberg, and Daniel Berckmans. Automatic identification of marked pigs in a pen using image pattern recognition. Computers and electronics in agriculture, 93:111–120, 2013.

Rahima Khanam and Muhammad Hussain. Yolov11: An overview of the key architectural enhancements. arXiv preprint arXiv:2410.17725, 2024.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, and others. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.

Dan Li, Kaifeng Zhang, Zhenbo Li, and Yifei Chen. A Spatiotemporal Convolutional Network for Multi-Behavior Recognition of Pigs. Sensors, 20(8): 2381, April 2020. ISSN 1424-8220. doi: 10. 3390/s20082381. URL https://www.mdpi.com/ 1424-8220/20/8/2381.

Ran Li, Baisheng Dai, Yuhang Hu, Xin Dai, Junlong Fang, Yanling Yin, Honggui Liu, and Weizheng Shen. Multi-behavior detection of group-housed pigs based on YOLOX and SCTS-SlowFast. Computers and Electronics in Agriculture, 225:109286, 2024. ISSN 0168-1699. doi: https://doi.org/10.1016/j.compag. 2024.109286. URL https://www.sciencedirect. com/science/article/pii/S016816992400677X.

Dong Liu, Maciej Oczak, Kristina Maschat, Johannes Baumgartner, Bernadette Pletzer, Dongjian He, and Tomas Norton. A computer vision-based method for spatial-temporal action recognition of tail-biting behaviour in group-housed pigs. Biosystems Engineering, 195:27–41, July 2020. ISSN 15375110. doi: 10.1016/j.biosystemseng.2020.04. 007. URL https://linkinghub.elsevier.com/ retrieve/pii/S1537511020300982.

Dong Liu, Andrea Parmiggiani, Eric Psota, Robert Fitzgerald, and Tomas Norton. Where’s your head at? Detecting the orientation and position of pigs with rotated bounding boxes. Computers and Electronics in Agriculture, 212:108099, September 2023. ISSN 01681699. doi: 10.1016/j.compag.2023. 108099. URL https://linkinghub.elsevier.com/ retrieve/pii/S0168169923004878.

Jisheng Lu, Zhe Chen, Xuan Li, Yuhua Fu, Xiong Xiong, Xiaolei Liu, and Haiyan Wang. ORP-Byte: A multi-object tracking method of pigs that combines Oriented RepPoints and improved Byte. Computers and Electronics in Agriculture, 219:108782, 2024.

Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. HOTA: A Higher Order Metric for Evaluating Multi-object Tracking. International Journal of Computer Vision, 129(2):548–578, February 2021. ISSN 0920-5691, 1573-1405. doi: 10.1007/s11263-020-01375-2. URL https://link. springer.com/10.1007/s11263-020-01375-2.

Gerard Maggiolino, Adnan Ahmad, Jinkun Cao, and Kris Kitani. Deep OC-Sort: Multi-Pedestrian Tracking by Adaptive Re-Identification. In 2023 IEEE International Conference on Image Processing (ICIP), pages 3025–3029, 2023. doi: 10.1109/ICIP49359.2023. 10222576.

Elisabeth Mayrhuber, Kristina Maschat, David Brunner, Stephan M. Winkler, and Maciej Oczak. Improved and interpretable accelerometerbased farrowing prediction. Biosystems Engineering, 263:104381, 2026. ISSN 1537-5110. doi: https://doi.org/10.1016/j.biosystemseng.2025. 104381. URL https://www.sciencedirect.com/ science/article/pii/S1537511025003174.

Maciej Oczak, Jean-Loup Rault, Suzanne Truong, and Oceane Schmitt. Skeleton-based image feature extraction for automated behavioral analysis in human-animal relationship tests. Applied Animal Behaviour Science, 277:106347, August 2024. ISSN 01681591. doi: 10.1016/j.applanim.2024. 106347. URL https://linkinghub.elsevier.com/ retrieve/pii/S0168159124001953.

Anicetus Odo, Niall McLaughlin, and Ilias Kyriazakis. Automated Monitoring of Ear Biting in Pigs by Tracking Individuals and Events. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 7080–7088, Waikoloa, HI, USA, January 2024. IEEE. ISBN 979-8-3503-1892-0. doi: 10.1109/WACV57701.2024.00694. URL https: //ieeexplore.ieee.org/document/10484109/.

Anicetus Odo, Niall McLaughlin, and Ilias Kyriazakis. Re-identification for long-term tracking and management of health and welfare challenges in pigs. Biosystems Engineering, 251:89–100, 2025. ISSN 1537-5110. doi: https://doi.org/10.1016/j.biosystemseng.2025. 02.001. URL https://www.sciencedirect.com/ science/article/pii/S153751102500025X.

Andrea Parmiggiani, Dong Liu, Eric Psota, Robert Fitzgerald, and Tomas Norton. Don’t get lost in the crowd: Graph convolutional network for online animal tracking in dense groups. Computers and Electronics in Agriculture, 212:108038, September 2023. ISSN 01681699. doi: 10.1016/j.compag.2023. 108038. URL https://linkinghub.elsevier.com/ retrieve/pii/S016816992300426X.

Eric Psota, Ty Schmidt, Benny Mote, and Lance C. Pérez. Long-Term Tracking of Group-Housed

Livestock Using Keypoint Detection and MAP Estimation for Individual Animal Identification. Sensors, 20(13):3670, June 2020. ISSN 1424-8220. doi: 10.3390/s20133670. URL https://www.mdpi.com/ 1424-8220/20/13/3670.

Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pages 17–35. Springer, 2016.

Kyujin Shim, Kangwook Ko, Yujin Yang, and Changick Kim. Focusing on tracks for online multi-object tracking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11687–11696, 2025.

Shuqin Tu, Yifan Cai, Yun Liang, Hua Lei, Yufei Huang, Hongxing Liu, and Deqin Xiao. Tracking and monitoring of individual pig behavior based on YOLOv5- Byte. Computers and Electronics in Agriculture, 221: 108997, 2024a.

Shuqin Tu, Yuefei Cao, Yun Liang, Zhixiong Zeng, Haoxuan Ou, Jiaying Du, and Weidian Chen. Tracking and automatic behavioral analysis of grouphoused pigs based on YOLOX+ BoT-SORT-slim. Smart Agricultural Technology, 9:100566, 2024b.

Shuqin Tu, Jiaying Du, Yun Liang, Yuefei Cao, Weidian Chen, Deqin Xiao, and Qiong Huang. Tracking and behavior analysis of Group-Housed pigs based on a Multi-Object Tracking approach. Animals: an Open Access Journal from MDPI, 14(19):2828, 2024c.

Shuqin Tu, Haoxuan Ou, Liang Mao, Jiaying Du, Yuefei Cao, and Weidian Chen. Behavior Tracking and Analyses of Group-Housed Pigs Based on Improved ByteTrack. Animals, 14(22):3299, 2024d.

Meiqing Wang, Mona L. V. Larsen, Dong Liu, Jeanet F. M. Winters, Jean-Loup Rault, and Tomas Norton. Towards re-identification for longterm tracking of group housed pigs. Biosystems Engineering, 222:71–81, 2022. ISSN 1537-5110. doi: https://doi.org/10.1016/j.biosystemseng.2022. 07.017. URL https://www.sciencedirect.com/ science/article/pii/S1537511022001799.

Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In 2017 IEEE International Conference on Image Processing (ICIP), pages 3645– 3649, September 2017. doi: 10.1109/ICIP.2017. 8296962. URL https://ieeexplore.ieee.org/ document/8296962/?arnumber=8296962.

Martin Wutke, Damiano Debiasi, Shobhana Tomar, Jeanette Probst, Nicole Kemper, Kai Gevers, Marc-Alexander Lieboldt, and Imke Traulsen. Multistage pig identification using a sequential ear tag detection pipeline. Scientific reports, 15(1):20153, 2025.

Yufei Xu, Jing Zhang, Qiming ZHANG, and Dacheng Tao. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 38571–38584. Curran Associates, Inc., 2022. URL https://proceedings. neurips.cc/paper\_files/paper/2022/file/ fbb10d319d44f8c3b4720873e4177c65-Paper-Conference. pdf.

Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. Fairmot: On the fairness of detection and re-identification in multiple object tracking. International journal of computer vision, 129(11):3069–3087, 2021.

Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. ByteTrack: Multi-object Tracking by Associating Every Detection Box. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 1–21, Cham, 2022. Springer Nature Switzerland. ISBN 978-3-031-20047- 2.

Yuanqin Zhang, Jiahao Cai, Deqin Xiao, Zesen Li, and Benhai Xiong. Real-time sow behavior detection based on deep learning. Computers and Electronics in Agriculture, 163:104884, August 2019. ISSN 01681699. doi: 10.1016/j.compag.2019. 104884. URL https://linkinghub.elsevier.com/ retrieve/pii/S0168169919300080.

## C Additional visualisations

a)  
![](images/1ef5a52ee8f82d436f76388e8a71d213ce8fdd91c5141a792c7354ffa8bb67de.jpg)

a)  
![](images/f8262fb772c78a5427ae236dc832a8088d389ed00f08fac25577177c05ca3d53.jpg)

b)  
![](images/35894433c7d9720af3849abbbc150ddb230e58db90a6017845d7773ee0a4d439.jpg)

b)  
![](images/f36266e0524b58532f0c7f89f1d47b931fd775c0783cb8570b2a5f70b0026321.jpg)

c)  
![](images/4af3947bc3292fbd3acdd87e6c7b0c3bf33fd182e4ad1da4cc09254b55a4df0d.jpg)

c)  
![](images/90cf7ba8bf1de349346339d804182705f508a0339a0c08c782b9139ba0d88f8e.jpg)  
Figure C.1: Scene property study for test clip penA\_30s\_2. The resolution of the pigs’ bounding boxes (a) reflects the pigs’ movement towards and away from the camera. The distance to the camera also correlates with the level of occlusion as measured by the bounding box overlap (b), because the sharper angle at distance makes occlusions more likely. The variance of Laplacian (VoL) (c) is lowest both when the motion blur is highest, as well as when the resolution is lowest. These confounding efects are introduced by the side-view setting.  
Figure C.2: Scene property study for test clip penA\_10s\_8. The increasing mean resolution points at a movement tendency towards the camera (a). The divergence in the mean and max./min. values of the properties reflects the fact that in this clip only individual pigs move while the others are stationary. The moving pigs cause occlusions towards the end of the clip (b). The constant VoL reflects the slow movement of the pigs (c).

![](images/b8e4e2527a3fca23c3ffaf63e1a9965ba555d332ccac86722b5a67ce77297d45.jpg)  
Figure C.3: Visualisation of the pigs’ movement patterns in the test data clips.