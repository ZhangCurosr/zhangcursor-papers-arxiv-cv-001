# Egocentric Action Recognition with Objects and Vision-Language Affordances

Raul Boix-Aldeguer´ <sup>1</sup>, Manuel Benavent-Lledo<sup>2</sup>, David Ortiz-Perez<sup>1</sup> and Jose Garcia-Rodriguez<sup>1</sup>

<sup>1</sup> Department of Computer Technology, University of Alicante, Alicante, Spain <sup>2</sup> Visual Analysis and Perception Lab, Aalborg University, Denmark

Abstract— Action recognition, and more specifically egocentric action recognition, is a core task in computer vision, with applications such as monitoring the daily life activities of elderly people and task assistance. Recent methods rely on visual modalities such as RGB video or depth. However, in egocentric video, an action is largely defined by its objects, which these methods rarely model. We propose a method that combines a frozen SlowFast video expert with an attentionbased object expert by late fusion. Each object is represented with its affordances, that is, what can be done with it, generated zero-shot by a vision-language model. Since datasets such as MECCANO annotate only the objects involved in each action, evaluating with these boxes tells the model which objects are active. We therefore train on annotated boxes and evaluate the method with detections from the official active-object detector, and with annotated boxes as an upper bound. Without affordances, our fusion reaches 68.4% Top-1 with annotated boxes but 49.0% with detections. About half of this drop comes from non-active detections. To remove them, the same vision-language model generates an interaction affordance that decides whether a hand is manipulating each detected object. On MECCANO, objects raise the Top-1 of the video expert from 44.9% to 49.0%, and the two affordances raise it to 49.9%, above SlowFast and object-aware RGB baselines. Code will be released upon acceptance.

## I. INTRODUCTION

Egocentric action recognition identifies the actions performed by a person from videos captured from their own point of view. It has applications in robotics, augmented reality, and personalized assistance, such as monitoring the daily activities of elderly people or guiding a user through a task. The spread of smart glasses and head-mounted cameras has generated a growing interest in understanding human activities from a first-person perspective, and several egocentric datasets support its study [3], [10], [14], [23], [26].

Most action recognition methods learn holistic spatiotemporal features from the whole clip, with 3D convolutional networks [7], [31] or video transformers [2], [29]. Egocentric methods add further modalities such as depth, gaze, or audio [12], [22]. These models do not represent the objects explicitly, although in egocentric video the action is largely determined by the objects and what can be done with them. For example, screwing a screw by hand and with a screwdriver involve similar hand motions, and only the manipulated screwdriver tells them apart. The few recent egocentric methods that use objects add them as extra supervision during training [28] or combine detected regions with gaze [11], but they do not model what the objects afford.

![](images/5cd8feafaacc83a66e9ea9aefe83d5fecc34dc32fcf1a793634cdc021bdcf81e.jpg)  
Fig. 1. The interaction affordance removes an idle object. The VLM answers that no hand manipulates the booklet (red, dashed), and the filter removes it. The video-only SlowFast baseline and the fusion with all detected objects predict check booklet, whereas our method predicts the correct action, take bolt. Bars show the three most probable actions, averaged over five seeds for our method.

A second limitation is how object-based models are evaluated. They are usually tested with annotated boxes, but egocentric datasets such as MECCANO [22], [23] annotate only the active objects, so these boxes tell the model which objects take part in the action. However, a detector also reports objects that are present but not manipulated, so the accuracy drops. Works that replace annotations with handobject [27] or open-vocabulary detectors [11] rarely measure how much accuracy is lost with respect to annotated boxes, or why.

In contrast, we propose a method that combines a frozen SlowFast video expert with an attention-based object expert by late fusion. Each object is represented by its appearance, class name, position, and affordances, which describe what can be done with the object in that situation [8]. Our hypothesis is that a zero-shot vision-language model (VLM) provides object knowledge that the video expert does not learn from the training videos. The VLM generates the affordances of each object as a short text. It also decides whether a hand is manipulating each detected object, and this interaction affordance removes idle detections before they reach the object expert. Fig. 1 shows an example in which the filter removes an idle instruction booklet, and the method recovers the correct action. We evaluate the method on MECCANO using the official active-object detector and annotated boxes as an upper bound. With detections, it improves the video expert by 4.9 Top-1 and 8.4 Top-5 points and outperforms SlowFast and object-aware RGB baselines. Our contributions can be summarized as follows:

• A method for egocentric action recognition that fuses a frozen video expert with an attention-based object expert, including affordances generated zero-shot by a VLM.

• An interaction affordance, generated by the same VLM, that removes, at test time, detected objects not manipulated at that moment, so fewer non-active objects reach the object expert.

• An analysis of the drop from annotated boxes to detections, which shows that 48% of it comes from nonactive detections.

## II. RELATED WORK

Egocentric action recognition. Egocentric action recognition methods usually build on video backbones such as SlowFast [7] and add signals from wearable cameras. For instance, EPIC-Fusion [12] combines RGB, optical flow, and audio, and EgoVLP [15] pretrains video and language representations on egocentric clips. On MECCANO, the dataset authors report SlowFast baselines with RGB, depth, and gaze, and combining the three modalities improves over RGB alone. A late fusion of RGB and depth Swin3D models won first place in the MECCANO action recognition challenge [13]. More recently, G3Ego [11] builds graphs from gaze, hands, and objects. An open-vocabulary detector localizes these objects from the captions of a VLM, and gaze discards the objects that the camera wearer does not look at. In these methods, however, objects are not described by the affordances they offer. Like G3Ego, our method excludes irrelevant objects, but uses a VLM-based manipulation filter rather than gaze.

Object-based action recognition. Objects have been a central cue in egocentric video since early works. Fathi et al. [6] learn object models from egocentric activities with weak supervision, and Pirsiavash and Ramanan [20] observe that objects look different while they are being used and train separate detectors for active and passive objects. More recent models reason over the boxes of hands and objects. STIN [16] models how hand and object boxes interact over time and reports results with both annotated and detected boxes. Shiota et al. [28] use hand-object contact and object state as extra supervision during training, so inference needs only RGB video. Hand-object detectors [27] localize hands and the objects they touch, and MECCANO’s active-object detector localizes the objects involved in the action. However, a detector also reports objects that are present but not used, and annotated boxes hide this problem because they mark only the active objects. Our object expert also uses boxes, but it represents each object with its affordances and removes the detections that are not being manipulated.

Affordances in egocentric video. Affordances describe what an agent can do with an object [8]. Nagarajan et al. [18] learn interaction hotspots, maps of where an object is likely to be manipulated, by watching people use objects. Ego-Topo [19] links environmental zones to the activities they afford to anticipate future actions. VRB [1] learns from egocentric videos where and how people interact with a scene and transfers these affordances to robots. AFFttention! [17] introduces STAformer, which uses environment affordances and interaction hotspots to anticipate the next object interaction in egocentric video. These works localize or predict affordances to anticipate interactions or guide robots, whereas our method uses short text descriptions of each object for action recognition.

Language for action recognition. Language also provides knowledge about actions. ActionCLIP [33] treats action recognition as matching videos with the text of the class names, and LaViLa [34] adapts a large language model to narrate egocentric videos and trains video-language representations on these narrations. These methods describe whole clips or classes, whereas our method describes each detected object and what it affords.

Vision-language models for affordances. Visionlanguage models learn from large image-text corpora that often describe how objects are used, which makes them a source of affordances. AffordanceLLM [21] fine-tunes a VLM to ground affordance regions in images, and Spot-Light [5] uses VLM affordance predictions to select robot motion primitives. Tonini et al. [30] evaluate multimodal LLMs that describe human-object interactions in free-form text and compare visual prompts such as red circles and crops. Unlike these works, we use a VLM zero-shot, first, to describe each detected object and its affordances for action recognition, and second, to remove detections that a hand is not manipulating.

## III. METHOD

Given a video segment that contains one action, the method predicts its class among C action classes. The input is the RGB segment and the objects in a set of candidate frames of the segment, each one a box with an object class. During training, the boxes come from annotations of the objects involved in the action (active objects). At test time, an object detector trained on the same object classes gives the boxes, and it also finds objects that are present but not involved in the action (non-active objects). We do not train the method on detections, because the detector is trained on the same training videos and is almost perfect on them, so it does not make the errors it makes on new videos.

Fig. 2 shows the pipeline at test time. A frozen video expert computes class probabilities from the whole segment (Sec. III-A). In parallel, the object detector finds objects in the candidate frames, and a VLM generates two affordances for each detection (Sec. III-B): an object affordance describing what can be done with the object, and an interaction affordance that removes the objects that are not manipulated. The object expert classifies the remaining objects (Sec. III-A), and late fusion combines both experts (Sec. III-C). The interaction filter is used only at test time, because all annotated boxes are active objects.

![](images/fc009190349f24de3833192e50e0cf607bd1a9fed824121ffa0f9eadc120b62e.jpg)  
Fig. 2. Overview of the method at test time. A frozen SlowFast expert gives the video probabilities $p _ { v } .$ An object detector finds the objects in the candidate frames of the segment. For each detection, a VLM answers whether a hand is manipulating the object, and the interaction filter removes the detections answered with no (red, dashed). Then K frames are selected, and each remaining object becomes a token of crop, class name, position, and affordance features. The affordance is generated by a separate prompt to the same VLM. The object expert applies self-attention to the tokens, averages them, and an MLP gives the probabilities $p _ { o }$ . Finally, late fusion combines both experts with the weight α. During training, the annotated boxes replace the detector, and the filter is not used.

## A. Video and Object Experts

1) Video expert: The video expert is a video classifier trained on the target dataset and kept frozen. We use Slow-Fast [7], which samples frames over the whole segment and averages the class probabilities $p _ { v }$ of several spatial crops.

2) Object representation: The object expert observes $K = 4$ frames per segment. Among the candidate frames of the segment, it selects the K frames with the largest change in the set of object classes with respect to the previous candidate frame (maximize transitions). Frames where the set of objects changes tend to mark the steps of the action, such as taking a part or aligning two parts. If a segment has fewer than K candidate frames, all of them are used. During training, the boxes and classes come from the annotations. At test time, the selection uses the classes of the filtered detections (Sec. III-B).

Each object i of the N objects in the K selected frames is represented by four parts: visual appearance, class name, position, and affordance. The visual appearance $\mathbf { v } _ { i }$ is the [CLS] embedding of a ViT-B/16 [4], pretrained on ImageNet-21k and fine-tuned on ImageNet-1k, applied to the box crop. The class name, a text with the box corners, size, and center normalized by the frame size, and the affordance text generated by the VLM (Sec. III-B) are encoded with DistilBERT [25] ([CLS] token), giving the 768-dimensional embeddings $\mathbf { c } _ { i } , \mathbf { b } _ { i } .$ , and a<sub>i</sub>. Representing the affordances as text keeps the VLM frozen and lets the same text encoder process the class name, the position, and the affordances. The object token is

$$
\mathbf { x } _ { i } = [ \mathbf { v } _ { i } ; \mathbf { c } _ { i } ; \mathbf { b } _ { i } ; \boldsymbol { \phi } ( \mathbf { a } _ { i } ) ] \in \mathbb { R } ^ { 3 0 7 2 } ,\tag{1}
$$

where $\phi$ is a trained projection from 768 to 768 dimensions (linear layer, layer normalization, and ReLU) and all encoders are frozen.

3) Object expert: The object expert projects each token to $\mathbf { h } _ { i } \in \mathbb { R } ^ { 5 1 2 }$ (linear layer, layer normalization, and ReLU), applies self-attention among the objects, and averages the result:

$$
H ^ { \prime } = \operatorname { L N } \bigl ( H + \operatorname { M H A } ( H , H , H ) \bigr ) , \qquad \mathbf { o } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { h } _ { i } ^ { \prime } ,\tag{2}
$$

where $H = [ \mathbf { h } _ { 1 } , \dots , \mathbf { h } _ { N } ] \in \mathbb { R } ^ { N \times 5 1 2 }$ stacks the N object tokens, MHA is multi-head self-attention [32] with four heads, LN is layer normalization, $\mathbf { h } _ { i } ^ { \prime }$ is the i-th row of $H ^ { \prime } ,$ and o is a single representation of the whole segment. A segment without objects $( N ~ = ~ 0 )$ uses a learned token instead. An MLP with two hidden layers (512 and 256 units) maps o to the logits $\mathbf { z } _ { o } ,$ , and the expert is trained with crossentropy.

Only ϕ, the token projection, the self-attention layer, the token for segments without objects, and the MLP are trained, while the video expert, ViT, DistilBERT, the detector, and the VLM remain frozen.

## B. Vision-Language Affordances

1) Object affordances: A VLM generates the affordances of each object zero-shot as a short text. The VLM receives the class name, the full frame with the object in a red box, and the crop of the object. The red box tells the VLM which object to describe while preserving scene context, and the crop shows the object at higher resolution, as in recent work comparing visual prompts for describing interactions [30]. The prompt describes the task and restricts the answer to a comma-separated list built from the verb and object classes of the dataset (Fig. 3, center).

2) Interaction affordance: The annotated boxes used for training contain only active objects, but the detector also reports objects that are present but not active. To address this, the method asks the VLM whether a hand is manipulating each detected object at that moment. We call this the interaction affordance. If the answer is no, the object is removed. For each detection $d ,$ the VLM receives the frame with the object in a red box and its crop. It first describes where the hands are, and then answers whether a hand holds, touches, or manipulates the object (yes, no, or unsure) (Fig. 3, right). Describing the hands first is meant to make the VLM look at the hands and the object together, instead of answering from the object alone.

Let $\mathcal { D } _ { t }$ be the detections in frame $t , s ( d )$ the detector score, and $r ( d )$ the VLM’s answer. The filter keeps

$$
\mathcal { D } _ { t } ^ { \prime } = \{ d \in \mathcal { D } _ { t } : r ( d ) \neq n o \} \cup \{ \arg \operatorname* { m a x } _ { d \in \mathcal { D } _ { t } } s ( d ) \} ,\tag{3}
$$

for every frame with at least one detection, so each such frame keeps its most confident detection. Detections answered unsure are kept. The filter runs on detections before frame selection.

## C. Late Fusion

The final prediction combines both experts:

$$
\hat { y } = \arg \operatorname* { m a x } _ { k } \big [ \alpha \log p _ { v } ( k ) + ( 1 - \alpha ) \log p _ { o } ( k ) \big ] ,\tag{4}
$$

where k ranges over the $C$ action classes, $p _ { o } = \operatorname { s o f t m a x } ( \mathbf { z } _ { o } )$ and $\alpha \in [ 0 , 1 ]$ is chosen on the validation set with a step of 0.05. With $\alpha = 1$ , the method uses only the video expert, and with $\alpha = 0$ , only the object expert.

## IV. EXPERIMENTS

## A. Experimental Setup

Dataset. MECCANO [22], [23] contains 20 egocentric videos of people assembling a toy motorbike, annotated with 61 action classes, which combine 12 verbs and 20 object classes. The official split has 11, 2, and 7 videos for training, validation, and test, with 5,057, 977, and 2,823 action segments. Three exact duplicate rows are removed from the test annotations, which leaves 2,820 test segments. Only the objects involved in the action are annotated with boxes, and the active-object annotations include only frames with at least one active object. We use MECCANO because it annotates the active objects of every action segment with boxes and provides an official active-object detector for the same classes. Both are needed to train the object expert and to measure the drop from annotated boxes to detections.

Evaluation protocol. The method is evaluated in two settings. With annotated boxes as an oracle, the object expert uses the annotations for training, validation, and testing, providing an upper bound. With detections, the object expert is trained on annotated boxes (Sec. III), and the validation and test sets use the official active-object detector of MECCANO with a score threshold of 0.5. The interaction filter is applied to the validation and test detections, and α and early stopping use the validation set with detections.

Detector. The detector is a Faster R-CNN [24] and it runs on the frames listed in the active-object annotations (the candidate frames), without their boxes. On the test set, it reaches 31.6 AP at IoU 0.5, close to the 30.39 reported by the MECCANO authors [23]. A detection is active if it matches an annotated box with an IoU of at least 0.5, and non-active otherwise. Of the detections, 53.9% are nonactive, and 62.0% of the annotated boxes are found. The AP is low partly because only active objects are annotated, so correct boxes on non-active objects count as errors. On the training set, it reaches 94.3 AP, so the object expert is not trained on detections.

Model variants. The video expert is SlowFast alone. Video + objects is the late fusion of the video expert and the object expert, without affordances or interaction filter. Video $+ \ o b j e c t s \ + \ V L M$ is our method, which adds the affordances and the interaction filter. Unless stated otherwise, results use detections.

Metrics. The metrics are Top-1 and Top-5 accuracy and macro-F1. We include macro-F1 to show the performance on rare actions, as the classes are imbalanced (the most frequent action covers 27.3% of the test segments). Results are the mean of five seeds, with the standard deviation for Top-1. Differences between two models are reported with a 95% confidence interval from a paired bootstrap over the test segments. All results are on the test set unless stated otherwise, and \* marks differences whose 95% confidence interval excludes zero.

Implementation details. The video expert is SlowFast R50 (8 × 8) with the checkpoint released with MECCANO. It samples 32 frames over the segment and averages three spatial crops of $2 5 6 \times 2 5 6$ that cover the full frame width, as in the official test protocol. The object expert is trained with Adam (learning rate $2 \cdot 1 0 ^ { - 4 }$ , no weight decay, batch size 64, dropout 0.5) for at most 100 epochs, with early stopping on the validation Top-1 (patience of 15 epochs). The VLM is Gemini 3.1 Flash-Lite [9] with temperature 0. It runs offline in the batch mode of Vertex AI, and each detection needs two queries, one for its affordance and one for the interaction affordance. On the test set, this amounts to 53,748 queries. Each query has 2,300 to 2,400 input tokens, and each test segment needs about 19 queries.

## B. Comparison with Baselines

Table I compares the method with published results on the official test split. With detections, our method reaches 49.9% Top-1 and 82.5% Top-5. It improves over the SlowFast RGB baseline of MECCANO by 4.7 Top-1 and 8.8 Top-5 points, and over our video expert by 4.9 and 8.4 points. This video expert uses the SlowFast checkpoint released with MECCANO and stays within 0.5 points of its published Top-1 and Top-5. Our method also outperforms HOCL+OSL, which uses hand-object contact and object state as supervision during training, and G3Ego, which needs gaze at test time.

![](images/0c51ca2f7ecc11219c540ae89b3e7eb5493d1036a806e1b3727dbeedbf53f198.jpg)  
Fig. 3. Prompts of the two VLM queries for the gray bar of Fig. 2. Both queries receive the frame with the detection in a red box and its crop. The object affordance (center) returns a comma-separated list, which is encoded with DistilBERT. The interaction affordance (right) describes the hands and answers whether a hand manipulates the object; detections answered no are removed. Prompts used on MECCANO. The list of object classes is shortened.

TABLE I  
COMPARISON WITH PUBLISHED RESULTS ON MECCANO. Extra input: WHAT EACH METHOD USES AT INFERENCE BESIDES RGB VIDEO.
<table><tr><td>Method</td><td>Extra input</td><td>Top-1</td><td></td><td>Top-5 Macro-F1</td></tr><tr><td>SlowFast [22]</td><td>I</td><td>45.16</td><td>73.75</td><td>18.91†</td></tr><tr><td>HOCL+OSL [28]</td><td>一</td><td>44.81</td><td>77.01</td><td>16.70</td></tr><tr><td>UCF Swin3D-B [13]‡</td><td></td><td>50.80</td><td>82.36</td><td>一</td></tr><tr><td>SlowFast [22]</td><td>D+G</td><td>49.66</td><td>77.82</td><td></td></tr><tr><td>G3Ego [11]</td><td>G+H+VLM+OD</td><td>46.48</td><td>82.04</td><td>21.34</td></tr><tr><td>Video expert</td><td></td><td>44.93</td><td>74.18</td><td>19.43</td></tr><tr><td>+ objects</td><td>AOD</td><td>49.04±0.68</td><td>81.34</td><td>22.69</td></tr><tr><td>+ objects + VLM (ours)</td><td>AOD+VLM</td><td>49.87 ±0.47</td><td>82.54</td><td>24.06</td></tr><tr><td>Video + objects</td><td>GT boxes</td><td>68.38±1.42</td><td>95.34</td><td>42.57</td></tr></table>

D: depth; G: gaze; H: hands; OD: open-vocabulary detector; AOD: active-object detector; GT boxes: annotated boxes as an oracle. <sup>†</sup>Computed by [11]. <sup>‡</sup>Pretrained on Something-Something v2, trained with focal loss.

Its Top-1 is similar to that of SlowFast with RGB, depth, and gaze, and its Top-5 is 4.7 points higher. UCF Swin3D-B obtains a higher Top-1 (50.8%) and a similar Top-5 (82.4%) with RGB only, using a Swin3D-B backbone pretrained on Something-Something v2 and a focal loss. Among the methods without annotated boxes, ours has the highest Top-5 and macro-F1. Compared with the video expert, it improves macro-F1 by 24% and Top-1 by 11% (relative), suggesting that the objects help less frequent actions more than frequent ones. However, the comparison is not fully equivalent. The detector is trained on the boxes of the MECCANO training set and runs on the frames listed in the active-object annotations, which contain at least one active object. A deployed system would also process frames without active objects.

Using annotated boxes, Video + objects reaches 68.4%

![](images/cbfae17878ece12ca4d5f258966706d630c62cab1789a4b32f702cd7a7ff531f.jpg)  
Fig. 4. Decomposition of the Top-1 drop from annotated boxes to detections. Each orange bar is the loss of one step that replaces annotated boxes with detections. The blue bar is the gain of affordances and the interaction filter (Sec. IV-D). The dashed line marks the accuracy without non-active detections, the accuracy of a perfect interaction filter.

Top-1, but this upper bound is not comparable to the other rows because the annotated boxes reveal the active objects. Sec. IV-C analyzes the gap.

## C. From Annotated Boxes to Detections

Replacing the annotated boxes with detections lowers the Top-1 of Video + objects from 68.4% to 49.0%. To identify the causes of this drop, the annotated boxes of the test set are replaced by detections in three steps. First, the annotated boxes that the detector misses are removed. Second, the remaining annotated boxes are replaced by their matched detections, which adds errors in box position and class. Third, the non-active detections are added, which gives the detector output.

Fig. 4 shows that the non-active detections cause the largest loss, 9.2 points (48% of the drop). A likely reason is that the object expert is trained on annotated boxes, in which every object takes part in the action. However, at test time, it sees every detected object, so it treats non-active objects as part of the action. Missed objects cost 7.3 points (38%). The detector is weakest on small parts, such as the wrench, wheel axle, or rod, and strongest on large objects, such as the booklet and screwdriver.

TABLE II  
TOP-1 OF THE OBJECT EXPERT ALONE WITH PARTIAL OBJECT TOKENS(MEAN POOLING, NO AFFORDANCES).
<table><tr><td>Token parts</td><td>Annotated boxes</td><td>Detections</td></tr><tr><td>Full (crop + class + position)</td><td>63.28</td><td>36.39</td></tr><tr><td>Class only</td><td>63.71</td><td>33.83*</td></tr><tr><td>Crop only</td><td>52.81*</td><td>36.39</td></tr><tr><td>Position only</td><td>33.88*</td><td>26.11*</td></tr><tr><td>Crop + position (no class)</td><td>50.00*</td><td>35.27*</td></tr></table>

TABLE III

ABLATION OF THE AFFORDANCES AND THE INTERACTION FILTER (LATE FUSION, DETECTIONS).
<table><tr><td></td><td>Aff. Filter</td><td>Top-1</td><td></td><td>Top-5 Macro-F1</td><td>∆ Top-1</td></tr><tr><td>一</td><td>一</td><td> $4 9 . 0 4 \pm 0 . 6 8$ </td><td>81.34</td><td>22.69</td><td></td></tr><tr><td>√</td><td>一</td><td> $4 9 . 5 3 \pm 0 . 4 9$ </td><td>82.08</td><td>23.36</td><td> $+ 0 . 5 0 \ [ + 0 . 1 1 , + 0 . 9 3 ] ^ { * }$ </td></tr><tr><td>一</td><td>√</td><td> $4 9 . 5 4 \pm 0 . 9 5$ </td><td>81.74</td><td>23.64</td><td>+0.50  $[ + 0 . 0 4 , + 0 . 9 2 ] ^ { * }$ </td></tr><tr><td>√</td><td>√</td><td> ${ \bf 4 9 . 8 7 \pm 0 . 4 7 }$ </td><td>82.54</td><td>24.06</td><td>+0.84  $[ + 0 . 3 0 , + 1 . 3 4 ] ^ { \ast }$ </td></tr></table>

Without non-active detections, Video + objects would reach 58.3%. This is the accuracy of a perfect interaction filter, and it motivates the filter of Sec. III-B. This estimate is optimistic, because some non-active detections are duplicates or badly placed boxes rather than idle objects. On the test set, 1,155 pairs of detections share the same box (IoU of at least 0.5) with different classes, up to 8% of the non-active detections.

Table II shows why annotated boxes give an upper bound. With them, the class name alone is as accurate as the full token, and removing it costs 13.3 points. With detections, the class name alone loses 2.6 points, and the crop alone matches the full token. Annotated boxes therefore give the model the class of the active objects. Detections do not, and the appearance of the objects carries most of the information. The class name also loses the most from annotated boxes to detections (29.9 points, against 16.4 for the crop). With detections, the position alone falls below the most frequent action (26.1% against 27.3%), so the detected boxes carry little information without their appearance.

## D. Affordances and Interaction Filter

Table III adds the affordances and the interaction filter to Video + objects with detections. ∆ is the Top-1 difference to Video + objects, with its 95% confidence interval. Each component improves Top-1 by 0.5 points. Together, they improve Top-1 by 0.8, Top-5 by 1.2, and macro-F1 by 1.4 points, all with intervals that exclude zero. Our method outperforms Video + objects in four of the five seeds. Across seeds, our method ranges from 49.4% to 50.5% Top-1, and

![](images/d05d0a09a885048120249a518f97e276cac8fcbc9a1bfb278775e39ef797350c.jpg)  
Text only take the screwdriver, align the screwdriver to the screw, tighten the screw, loosen the screw, put the screwdriver Text + crop take screwdriver, screw screw, unscrew screw, tighten screw, loosen screw Text + frame take, screwdriver Text + frame + crop screw, tighten, unscrew, loosen  
Fig. 5. Affordances generated by the VLM for a detected screwdriver with the four inputs: the crop (left) and the frame with the detection in a red box (right). The text-only answer depends only on the class name and is the same for every screwdriver. With the crop, the answer describes the screwdriver in use, whereas the frame alone gives a short answer.

Video + objects from 48.4% to 50.2%. The gains are small but consistent on the test set. The two components help in different ways: the affordances raise Top-5 more (0.7 against 0.4 points), and the filter raises macro-F1 more (1.0 against 0.7). Our method also has the lowest standard deviation between seeds (0.47, against 0.68 for Video + objects). On the validation set, which has only two videos, the differences are within the variation between seeds (52.6% against 52.3% for Video + objects). On the full validation set, the filter removes only 7.4% of the non-active detections, which may explain why its gain does not appear there.

The prompt of the interaction affordance is chosen on 500 validation detections (Table IV). For each prompt, the table gives the share of detections answered no and the precision (P) and recall (R) of no for non-active detections. It also gives the share of non-active and active detections removed by the filter (Eq. 3). Asking the VLM to describe the hands before answering removes the most non-active detections, with the same share of active ones as the frame and crop alone. On these detections, the VLM never answers unsure, so the policy for this answer does not affect the results. On the test set, the interaction filter removes 12.1% of the non-active detections and 3.7% of the active ones, and the number of non-active detections per frame drops from 1.24 to 1.09. Keeping the most confident detection of each frame is important. Removing every detection answered no (Table IV, last row) would remove 20.5% of the non-active detections, but also 18.6% of the active ones. The VLM answers yes for 79.5% of the non-active detections, likely because many of them are close to the hands or even held (Fig. 6c). In addition, an active object is not always touched, such as the booklet in a check booklet action. As a result, the filter alone recovers 0.5 of the 9.2 points lost to non-active detections (Sec. IV-C).

We compare four variants that differ in the visual input: text only (the class name, one query per class), text + crop, text + frame (the full frame with the object in a red box), and text + frame + crop. The four affordance inputs improve Top-1 with detections by up to 0.5 points (Table V), and text + frame + crop, chosen on the validation set, gives the largest gain. Even text only, which gives the same affordances to every object of a class, improves Top-1 by 0.35 points with detections, so part of the gain comes from the knowledge of the VLM about each class rather than from the image. With annotated boxes, the affordances do not help consistently (between −0.7 and +0.6 points), because the annotated class already identifies the active objects (Table II). With detections, the VLM describes the object as it appears in the frame. In Fig. 5, the text-only answer is the same for every screwdriver, but with the crop, the answer describes the screwdriver in use.

TABLE IV  
PROMPTS OF THE INTERACTION AFFORDANCE.
<table><tr><td>VLM input</td><td>No</td><td>P/R</td><td colspan="2">Removed</td></tr><tr><td></td><td></td><td></td><td>Non-active Active</td><td></td></tr><tr><td>Validation (500 detections)</td><td></td><td></td><td></td><td></td></tr><tr><td>Frame</td><td></td><td>11.0 60.0 / 14.1</td><td>8.1</td><td>2.3</td></tr><tr><td>+ crop</td><td>11.0</td><td>58.2 / 13.7</td><td>7.3</td><td>2.6</td></tr><tr><td>+ hand description (used)</td><td>14.2</td><td>56.3 /  17.1</td><td>9.4</td><td>2.6</td></tr><tr><td>Test (26,874 detections)</td><td></td><td></td><td></td><td></td></tr><tr><td>+ hand description (used)</td><td></td><td>19.656.3 / 20.5</td><td>12.1</td><td>3.7</td></tr><tr><td>without keeping the top detection</td><td></td><td>19.6 56.3 / 20.5</td><td>20.5</td><td>18.6</td></tr></table>

TABLE V

TOP-1 DIFFERENCE OF EACH AFFORDANCE INPUT TO VIDEO + OBJECTS, WITHOUT INTERACTION FILTER.
<table><tr><td>VLM input</td><td>Annotated boxes</td><td>Detections</td></tr><tr><td>Text only</td><td>-0.06</td><td>+0.35*</td></tr><tr><td> $\mathrm { T e x t } + \mathrm { c r o p }$ </td><td>+0.55*</td><td>+0.41*</td></tr><tr><td> $\mathrm { T e x t } + \mathrm { f r a m e }$ </td><td>-0.72*</td><td>+0.02</td></tr><tr><td>Text + frame + crop (used)</td><td>-0.54</td><td>+0.50*</td></tr></table>

## E. Fusion and Frame Selection

This section justifies two design choices of Sec. III, the fusion strategy and the frame selection, both evaluated without affordances.

Table VI compares the fusion strategies. The learned fusions combine the video embedding and the pooled object representation before a shared MLP, and the object tokens are pooled with mean or attention (attn.). With annotated boxes, late fusion outperforms all learned fusions by about 10 points, and the learned fusions are below the object expert alone. The two experts are complementary. For example, with detections, the share of screw with hands and screw with screwdriver segments predicted as the other action drops from 6.6% with the video expert to 1.6% with Video + objects. Late fusion improves over the best single expert by 4.1 points in both settings: the video expert with detections and the object expert with annotated boxes. With detections, the object expert alone has a lower Top-1 than the video expert (37.8% against 44.9%), but a similar Top-5 (74.7% against 74.2%), so the objects narrow the candidate actions without deciding among them. With detections, only the object expert is correct in 12.4% of the test segments and only the video expert in 19.5%. With annotated boxes, only the object expert is correct in 30.6%, which explains why late fusion outperforms the learned fusions most in this setting.

TABLE VI  
FUSION STRATEGIES WITHOUT AFFORDANCES.
<table><tr><td rowspan="2">Model</td><td colspan="3">Detections</td><td rowspan="2">Annotated</td></tr><tr><td>Top-1</td><td>Top-51</td><td>Macro-F1</td></tr><tr><td>Video expert</td><td>44.93</td><td>74.18</td><td>19.43</td><td>44.93</td></tr><tr><td>Object expert, mean</td><td>36.39±0.38</td><td>72.00</td><td>11.20</td><td>63.28 ±0.33</td></tr><tr><td>Object expert, attn.</td><td> $3 7 . 7 5 \pm 0 . 5 5$ </td><td>74.69</td><td>12.84</td><td>64.27 ±0.70</td></tr><tr><td>Learned: sum, mean</td><td> $4 8 . 1 3 \pm 0 . 5 8$ </td><td>80.19</td><td>21.64</td><td>58.44 ±0.25</td></tr><tr><td>Learned: concat., mean</td><td> $4 7 . 2 9 \pm 0 . 3 6$ </td><td>77.03</td><td>21.02</td><td>55.26 ±0.32</td></tr><tr><td>Learned: sum, attn.</td><td> $4 8 . 3 1 \pm 0 . 3 1$ </td><td>79.33</td><td>22.16</td><td>57.97 ±0.33</td></tr><tr><td>Learned: weighted sum, attn.</td><td> $4 8 . 1 0 \pm 0 . 3 8$ </td><td>78.77</td><td>22.07</td><td>57.72 ±0.38</td></tr><tr><td>Learned: cross-attention</td><td> $4 6 . 8 6 \pm 0 . 7 0$ </td><td>74.95</td><td>21.06</td><td>54.91 ±0.42</td></tr><tr><td>Late, mean</td><td> $4 8 . 6 2 \pm 0 . 3 2$ </td><td>80.74</td><td>22.45</td><td>68.25 ±0.48</td></tr><tr><td>Late, attn. (used)</td><td>49.04±0.68</td><td>81.34</td><td>22.69</td><td>68.38±1.42</td></tr></table>

TABLE VII

FRAME SELECTION AND NUMBER OF FRAMES K (LATE FUSION, NO AFFORDANCES).
<table><tr><td>Frames</td><td>K</td><td></td><td>Detections Annotated</td></tr><tr><td>Maximize transitions (used)</td><td>4</td><td>49.04</td><td>68.38</td></tr><tr><td>Fixed offsets</td><td>4</td><td>-0.45</td><td>-4.55*</td></tr><tr><td>Uniform annotated</td><td>4</td><td>+0.72*</td><td>+2.05*</td></tr><tr><td>High variability</td><td>4</td><td>-0.06</td><td>-2.98*</td></tr><tr><td>Progressive discovery</td><td>4</td><td>+0.33</td><td>-1.01*</td></tr><tr><td>Maximize transitions</td><td>6</td><td>+0.22</td><td>+2.06*</td></tr><tr><td>Uniform annotated</td><td>6</td><td>+0.20</td><td>+2.92*</td></tr><tr><td>Uniform annotated</td><td>all</td><td>-0.01</td><td>+1.84*</td></tr></table>

The learned fusions do not exploit this complementarity. SlowFast was trained on the same training split and its head reaches 99.1% Top-1 on it, so a joint classifier fits the training labels from the video features alone and ignores the objects.

The Top-1 difference with detections is small (0.7 points over the best learned fusion) and not conclusive, but late fusion improves the Top-5 of this learned fusion by 2.0 points. The weight α chosen on the validation set increases from 0.40–0.60 with annotated boxes to 0.65–0.75 with detections, so the method relies more on the video expert when the objects come from the detector. Self-attention improves the object expert alone by 1.4 points over mean pooling with detections, against 1.0 with annotated boxes, possibly because attention lets each object weigh the others when some of them are non-active.

Table VII compares the frame selection strategies. The first row gives the Top-1 of maximize transitions with K = 4, and the other rows give the difference to it. Fixed offsets takes raw frames at fixed offsets from the segment start, and uniform annotated takes K annotated frames equally spaced in time. High variability and progressive discovery select annotated frames where the set of object classes changes or new classes appear. With annotated boxes, the strategy changes Top-1 by up to 7.5 points, but with detections, all strategies stay within 1.2 points. More than four frames do not help with detections, likely because each additional frame adds nonactive detections. Uniform annotated is the best strategy on the test set, but the validation set selects maximize transitions, and the method keeps this choice.

![](images/2b9e91b0a1930eb20245e727c4ca8c7dd472b873c855810be5e0ea5888ae2b03.jpg)  
Fig. 6. Qualitative results on the test set. Top: annotated boxes (blue). Middle: all detections (orange). Bottom: detections after the interaction filter (green: kept; red, dashed: removed), with the VLM’s answer in parentheses. Predictions are the majority over five seeds. (a) The filter removes the idle booklet, and our method corrects the prediction. (b) The VLM answers no for the bolt being taken, and the filter removes the only active object. (c) The VLM answers yes for every detection near the hands, and the non-active screwdriver remains.

## F. Qualitative Results

Fig. 6 shows three test segments with the annotated boxes, the detections, and the detections after the interaction filter. In Fig. 6a, the filter removes the booklet lying on the table, and our method corrects the prediction of Video + objects. Figs. 6b and 6c show the two failure modes of the filter. In Fig. 6b, the frame is blurred. The VLM answers no for the bolt that the person is taking, so the filter removes the only active object. In Fig. 6c, the VLM answers yes for all five detections, and both Video + objects and our method predict screw with screwdriver, the action of the nonactive screwdriver. The second failure is the most frequent. The filter removes only 12.1% of the non-active detections, compared with 3.7% of the active ones, so its recall is the

method’s main limitation.

## V. CONCLUSIONS

This paper presents a method for egocentric action recognition that fuses a frozen SlowFast video expert with an attention-based object expert. Each object token includes zero-shot affordances generated by a VLM, and the same VLM provides an interaction affordance that removes detections that no hand manipulates. On MECCANO with the official active-object detector, the method improves the video expert and outperforms SlowFast and object-aware RGB baselines, and the affordances and the interaction filter give small but consistent gains. The analysis of the drop from annotated boxes to detections shows that the main obstacle for object-based recognition with a real detector is nonactive detections, a problem that evaluation with annotated boxes hides. These results suggest that object-based methods should also be evaluated with detections.

Future work will use the temporal context of the segment, rather than a single frame, to decide whether an object is manipulated. It will also test hand-object detectors that localize only objects in contact with the hands, and evaluate our method on other egocentric datasets.

## REFERENCES

[1] S. Bahl, R. Mendonca, L. Chen, U. Jain, and D. Pathak. Affordances from human videos as a versatile representation for robotics. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13778–13790, 2023.

[2] G. Bertasius, H. Wang, and L. Torresani. Is space-time attention all you need for video understanding? In International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pages 813–824, 2021.

[3] D. Damen, H. Doughty, G. M. Farinella, A. Furnari, E. Kazakos, J. Ma, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray. Rescaling egocentric vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. International Journal of Computer Vision, 130:33–55, 2022.

[4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

[5] T. Engelbracht, R. Zurbrugg, M. Pollefeys, H. Blum, and Z. Bauer. ¨ SpotLight: Robotic scene understanding through interaction and affordance detection. In IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2025.

[6] A. Fathi, X. Ren, and J. M. Rehg. Learning to recognize objects in egocentric activities. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3281–3288, 2011.

[7] C. Feichtenhofer, H. Fan, J. Malik, and K. He. SlowFast networks for video recognition. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 6202–6211, 2019.

[8] J. J. Gibson. The Ecological Approach to Visual Perception. Houghton Mifflin, Boston, 1979.

[9] Google DeepMind. Gemini 3.1 Flash-Lite model card. https://deepmind.google/models/model-cards/ gemini-3-1-flash-lite/, 2026.

[10] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, 2022.

[11] M. Haralovic, A. Ramakrishnan, and E. Talavera Martinez. ´ G3Ego: Gaze-guided graphs for egocentric action understanding. arXiv:2608.20157 (CONTEXTUS Workshop, ECCV), 2026.

[12] E. Kazakos, A. Nagrani, A. Zisserman, and D. Damen. EPIC-Fusion: Audio-visual temporal binding for egocentric action recognition. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 5492–5501, 2019.

[13] J. Kini, S. Fleischer, I. Dave, and M. Shah. Egocentric RGB+Depth action recognition in industry-like settings. arXiv:2309.13962, 2023.

[14] Y. Li, M. Liu, and J. M. Rehg. In the eye of beholder: Joint learning of gaze and actions in first person video. In European Conference on Computer Vision (ECCV), pages 619–635, 2018.

[15] K. Q. Lin, J. Wang, M. Soldan, M. Wray, R. Yan, E. Z. Xu, D. Gao, R.-C. Tu, W. Zhao, W. Kong, C. Cai, H. Wang, D. Damen, B. Ghanem, W. Liu, and M. Z. Shou. Egocentric video-language pretraining. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 2022.

[16] J. Materzynska, T. Xiao, R. Herzig, H. Xu, X. Wang, and T. Darrell. Something-Else: Compositional action recognition with spatialtemporal interaction networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1049–1059, 2020.

[17] L. Mur-Labadia, R. Martinez-Cantin, J. J. Guerrero, G. M. Farinella, and A. Furnari. AFF-ttention! Affordances and attention models for short-term object interaction anticipation. In European Conference on Computer Vision (ECCV), pages 167–184, 2024.

[18] T. Nagarajan, C. Feichtenhofer, and K. Grauman. Grounded humanobject interaction hotspots from video. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 8688–8697, 2019.

[19] T. Nagarajan, Y. Li, C. Feichtenhofer, and K. Grauman. Ego-Topo: Environment affordances from egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 163–172, 2020.

[20] H. Pirsiavash and D. Ramanan. Detecting activities of daily living in first-person camera views. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2847–2854, 2012.

[21] S. Qian, W. Chen, M. Bai, X. Zhou, Z. Tu, and L. E. Li. AffordanceLLM: Grounding affordance from vision language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 7587–7597, 2024.

[22] F. Ragusa, A. Furnari, and G. M. Farinella. MECCANO: A multimodal egocentric dataset for humans behavior understanding in the industriallike domain. Computer Vision and Image Understanding, 235:103764, 2023.

[23] F. Ragusa, A. Furnari, S. Livatino, and G. M. Farinella. The MECCANO dataset: Understanding human-object interactions from egocentric videos in an industrial-like domain. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1569– 1578, 2021.

[24] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards realtime object detection with region proposal networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 28, 2015.

[25] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv:1910.01108 (EMC<sup>2</sup> Workshop, NeurIPS), 2019.

[26] F. Sener, D. Chatterjee, D. Shelepov, K. He, D. Singhania, R. Wang, and A. Yao. Assembly101: A large-scale multi-view video dataset for understanding procedural activities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21096– 21106, 2022.

[27] D. Shan, J. Geng, M. Shu, and D. F. Fouhey. Understanding human hands in contact at internet scale. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9869–9878, 2020.

[28] T. Shiota, M. Takagi, K. Kumagai, H. Seshimo, and Y. Aono. Egocentric action recognition by capturing hand-object contact and object state. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 6541–6551, 2024.

[29] Z. Tong, Y. Song, J. Wang, and L. Wang. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pretraining. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 2022.

[30] F. Tonini, A. Conti, L. Vaquero, C. Beyan, and E. Ricci. Towards unconstrained human-object interaction. In IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2026.

[31] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri. Learning spatiotemporal features with 3D convolutional networks. In IEEE International Conference on Computer Vision (ICCV), pages 4489– 4497, 2015.

[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[33] M. Wang, J. Xing, J. Mei, Y. Liu, and Y. Jiang. ActionCLIP: Adapting language-image pretrained models for video action recognition. IEEE Transactions on Neural Networks and Learning Systems, 36:625–637, 2025.

[34] Y. Zhao, I. Misra, P. Krahenb ¨ uhl, and R. Girdhar. Learning video ¨ representations from large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6586–6597, 2023.