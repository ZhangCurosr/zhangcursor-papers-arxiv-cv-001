# GenStream: Semantic Streaming Framework for Generative Reconstruction of Human-centric Media

Emanuele Artioli emanuele.artioli@aau.at Alpen-Adria-Universitaet Klagenfurt, Kaernten, Austria

Daniele Lorenzi daniele.lorenzi@aau.at Alpen-Adria-Universitaet Klagenfurt, Kaernten, Austria

Farzad Tashtarian farzad.tashtarian@aau.at Alpen-Adria-Universitaet Klagenfurt, Kaernten, Austria

Shivi Vats shivi.vats@aau.at Alpen-Adria-Universitaet Klagenfurt, Kaernten, Austria

Christian Timmerer christian.timmerer@aau.at Alpen-Adria-Universitaet Klagenfurt, Kaernten, Austria

## Abstract

Video streaming dominates global internet trafic, yet conventional pipelines remain ineficient for structured, human-centric content such as sports, performance, or interactive media. Standard codecs re-encode entire frames, foreground and background alike, treating all pixels uniformly and ignoring the semantic structure of the scene. This leads to significant bandwidth waste, particularly in scenarios where backgrounds are static and motion is constrained to a few salient actors. We introduce GenStream, a semantic streaming framework that replaces dense video frames with compact, structured metadata. Instead of transmitting pixels, GenStream encodes each scene as a combination of skeletal keypoints, camera viewpoint parameters, and a static 3D background model. These elements are transmitted to the client, where a generative model reconstructs photorealistic human figures and composites them into the 3D scene from the original viewpoint. This paradigm enables extreme compression, achieving over 99.9% bandwidth reduction compared to HEVC for the continuous data stream. We partially validate GenStream on Olympic figure skating footage and demonstrate potential for high perceptual fidelity under minimal data. While acknowledging the significant computational costs shifted to the client and challenges in generalization, GenStream opens new directions in volumetric avatar synthesis, canonical 3D actor fusion across views, and personalized viewing experiences, laying the groundwork for scalable, intelligent streaming in the post-codec era.

## CCS Concepts

• Computing methodologies → Object identification; Artificial intelligence; Concurrent algorithms; Image compression; • Information systems → Online analytical processing; Multimedia streaming.

## Keywords

HTTP adaptive streaming, Generative AI, End-to-end architecture, Quality of Experience

![](images/723d87320e6e8eef17c8ea940a4b39a3b263704695409b8d3ad19f6a7bfabd2f.jpg)

ACM Reference Format:   
Emanuele Artioli, Daniele Lorenzi, Shivi Vats, Farzad Tashtarian, and Christian Timmerer. 2025. GenStream: Semantic Streaming Framework for Generative Reconstruction of Human-centric Media. In Proceedings ofthe 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/ 3746027.3758153

## 1 Introduction

Video streaming plays a pivotal role in digital media consumption and accounts for the majority of global Internet data trafic [41]. Specifically, major live international events such as the Olympic Games attract millions of viewers worldwide, generating massive volumes ofdata as high-definition video streams are delivered simultaneously to heterogeneous devices [50]. With the growing appetite for immersive and interactive content, the limitations of existing video streaming paradigms are becoming increasingly evident.

HTTP Adaptive Streaming (HAS) has become the de facto standard for video delivery over the Internet, dynamically adjusting video quality based on real-time network conditions [5, 32]. In HAS, video content is partitioned into segments of fixed duration (e.g., 2 to 10 seconds [53]), each encoded independently using a video codec such as Advanced Video Coding (AVC) [51] or its successor, High Efficiency Video Coding (HEVC) [45]. To support adaptive switching, each segment is encoded into several quality representations, where each representation is defined by a specific bitrate and resolution pair, collectively referred to as the bitrate ladder. While modern codecs are efective at reducing redundancy through intra-frame (spatial) and inter-frame (temporal) compression [45, 51], they operate under strict constraints: each segment is independently encoded and, hence, can only reference frames within its temporal window, limiting inter-frame prediction to short-term dependencies [56]. This means that when long-term repetitive patterns occur, the codec repeatedly re-encodes similar content, failing to capitalize on the inherent structural redundancy [43]. A compelling example of such ineficiency can be observed in sporting events, where athletes move against a generally static or slowly evolving background, such as a stadium, lighting fixtures, and surrounding audience. Despite the structural consistency of such scenes, conventional codecs repeatedly re-encode the same portions of background every time the camera passes over them. This redundancy issue is further exacerbated in multi-camera productions, where identical scenes captured

This work is licensed under a Creative Commons Attribution 4.0 International License. MM ’25, Dublin, Ireland   
© 2025 Copyright held by the owner/author(s).   
ACM ISBN 979-8-4007-2035-2/2025/10   
https://doi.org/10.1145/3746027.3758153

from diferent viewpoints are encoded independently, squandering opportunities to exploit cross-view redundancy [54, 59]. To address these ineficiencies, we propose GenStream, a framework designed for live streaming that redefines how such events can be streamed and experienced. Instead of transmitting full video frames, GenStream captures skeleton-based keypoints (e.g., body landmarks) of the performers with their relative bounding box and sends this compact representation to the client. The server then tackles a critical challenge: estimating the original camera viewpoint for each frame, which is essential for replicating the broadcast viewpoint. This pose, along with the performer keypoints, is sent to the client. At the client side, a second core challenge arises: gen erating a photorealistic rendering of the performer from sparse input using a generative model such as a conditional Generative Adversatial Network (cGAN) [19] or difusion model [37]. The syn thesized figure is then composited into a pre-scanned 3D model of the venue, and rendered from the provided viewpoint to faithfully recreate the original scene. This paradigm shift ofers dramatically reduced bandwidth usage, and support for personalized viewing experiences, such as a stylistic customization of the performers or points of view that difer from what the real cameras are capturing. Such personalization is a critical factor in media delivery, as understanding and catering to individual user sensitivities has been shown to directly impact and improve user engagement [2].

According to our preliminary results, GenStream can reduce the required live streaming bitrate by over 99.9% compared to a baseline HEVC-encoded video, assuming the 3D representation of the venue and generative model weights are pre-distributed to clients. This impressive reduction in bandwidth enables a stall-free streaming experience at the massive scale required for worldwide sporting events, even under constrained or fluctuating network conditions. At the same time, GenStream preserves high visual fidelity through client-side synthesis, albeit at the expense of increased computational complexity.

## 2 Related Work

Our work is positioned at the intersection of traditional video streaming, semantic video analysis, and generative synthesis. We build upon decades of work in video compression while taking inspiration from emerging paradigms in volumetric and model-based media transmission.

## 2.1 Traditional Video Streaming and Codecs

Video streaming today is predominantly driven by standard protocols such as HTTP Adaptive Streaming (HAS) [44], which underlie systems like HLS [1] and MPEG-DASH [12]. These methods divide content into short segments and ofer multiple quality tiers to allow the client to adapt playback to current network conditions. Combined with client-side adaptive bitrate (ABR) logic, these systems ensure playback continuity and have been highly successful in scaling video delivery across heterogeneous networks and devices. Compression in these pipelines is handled by increasingly sophisti cated codecs, such as AV1 [11] and VVC (H.266) [7], which improve upon earlier standards in terms of compression eficiency, often achieving up to 50% savings. These advances are achieved through better motion prediction, transform coding, and entropy models.

However, such gains come at a significant computational cost, especially on the encoder side, with high-resolution or real-time use cases facing increased energy and latency burdens [7, 11]. Despite decades of progress, traditional codecs treat all pixels equally and operate at the block level, agnostic to scene semantics. As a result, even small improvements in compression now require disproportionate increases in complexity, yielding diminishing returns for practical deployment.

## 2.2 Emerging Architectures for Video Encoding

To address some of the ineficiencies of standard codecs, recent research has explored novel encoding paradigms that move beyond traditional block-based compression. One approach augments existing codecs with machine learning components to improve rate control, motion estimation, or mode decisions [23, 25]. These techniques often integrate seamlessly with legacy pipelines, but remain bound by the structure of the original codec. A more radical line of work proposes fully neural video compression systems, in which residuals and motion fields are learned end-to-end by deep networks [10, 25]. While promising in terms of visual quality, these models are often too computationally intensive for real-time use or deployment on resource-limited clients. Similar techniques have been developed for the 3D media landscape, i.e., volumetric video streaming, which aims to enable immersive, free-viewpoint experi ences. Research in this area has explored transmission schemes for point cloud videos, using neural networks to extract and reconstruct features to reduce bandwidth [60], and allow for photorealistic rendering from arbitrary viewpoints [15]. More recently, data-driven representations like Neural Radiance Fields (NeRF) [30] and 3D Gaussian Splatting (3DGS) [58] have opened new possibilities for compact and continuous scene representations. While typically used for static scene modeling, they provide new pathways for transmitting visual content as a structured function of space and viewpoint. Complementary to these eforts, other works focus on client-side enhancement techniques that improve the subjective quality of the received video. Methods such as frame interpolation [33, 52] and super-resolution [20, 35, 57] can increase perceived temporal or spatial fidelity without increasing the transmitted bitrate. However, these emerging techniques all share a common trait: they treat video as a dense signal composed of low-level features such as pixels, motion vectors, or residuals. They do not leverage higher-level understanding of scene composition, such as objects, or actions, and thus miss a key opportunity for more structured and eficient encoding.

## 2.3 Object-Centric Representations and Semantic Streaming

The core idea of transmitting semantic information instead of raw pixels has been explored extensively, particularly in the context of video conferencing and volumetric video. Early work in objectbased coding, such as MPEG-4 Part 2 [36], allowed for the separate encoding of video object planes (VOPs), but saw limited adoption due to the complexity of robust, real-time segmentation. More recently, the rise of deep learning has revitalized semantic approaches. For instance, ELVIS [3] uses semantic segmentation to identify and remove low-priority background regions before transmission, using generative inpainting on the client to reconstruct them. This, however, still relies on traditional codecs for both foreground and higher-priority background regions.

Semantic video streaming shifts the encoding focus from raw pixels to meaningful scene components [21]. This perspective recognizes that not all regions of a video are equally important: agents (such as humans, vehicles, or other active foreground elements) are typically of greater perceptual and narrative significance than background elements. By decoupling foreground and background and encoding them diferently, streaming systems can better match perceptual salience to bitrate allocation. Systems like SemConf [14] transmit facial landmarks and background masks to drive a generative model on the client side, achieving significant bandwidth savings for talking-head scenarios. Similarly, [49] proposed a one-shot free-view synthesis method for video conferencing, while [18] explored wireless semantic communications for the same application. These systems demonstrate the viability of generative reconstruction from sparse metadata. To enable semantic encoding, a variety of tools have emerged. Object detection and tracking algorithms allow for persistent identification of foreground elements across time [28, 55]. Instance segmentation networks can further delineate objects from their surroundings, enabling region-specific treatment during encoding [39]. These components form the foundation for object-centric video understanding. While these techniques have seen some integration in video conferencing systems, where avatar based transmission and background substitution are common, their use remains tightly scoped to face-to-face communication scenarios. In broader video streaming, particularly for entertainment, sports, or interactive media, semantic approaches are largely absent. The vast majority of streaming systems continue to transmit raw pixels rather than structured representations, leaving the potential of object-level encoding underutilized.

## 2.4 Generative Reconstruction

Generative models have recently demonstrated remarkable capa bilities in synthesizing photorealistic video content from abstract inputs such as pose skeletons, semantic maps, or depth [4, 8]. This progress has been fueled by advances in both image and video generation, ranging from GAN-based architectures to difusion models, and has enabled applications in human reanimation, style transfer, and controllable video generation [17, 19, 20, 37]. In the context of streaming, these models ofer an opportunity to decouple the representation of motion and appearance. Instead of transmitting dense pixel data, a system may instead send a sparse representation, such as 2D keypoints or depth maps, and rely on a generative model at the client to reconstruct the full image content. This approach opens the door to radically more eficient pipelines.

To overcome current limitations and develop a truly semantic streaming system, future eforts can draw from advances in camera motion estimation and 3D scene reconstruction. Accurate tracking of camera viewpoint [13, 31, 47] can enable dynamic viewpoint control on the client. Meanwhile, neural scene representations allow the static background to be precomputed and encoded eficiently as a view-consistent 3D asset. When combined with semantic foreground representations, these modalities enable a modular and compositional approach to streaming where agents, environments, and motion can each be transmitted and reconstructed independently.

## 3 Problem Description and Motivation

To evaluate the efectiveness of our approach in terms of video quality and bandwidth reduction, we first establish a baseline by measuring the performance of current state-of-the-art codecs under constrained bandwidth conditions. We apply this comparison to a representative 1080p video scene from an artistic ice skating competition, recorded at a frame rate of 30 fps and 50 s in duration. As our reference codec, we select the High-Eficiency Video Codec (HEVC) [45] due to its widespread adoption [6] and high compression eficiency [45]. The target video [34] is encoded at a resolution of 1920×1080 with a bitrate of 5.8 Mbps. Artistic ice skating competitions present a compelling opportunity for compression via semantic and structural decomposition. A typical broadcast sequence features: (i) a static or slowly evolving background (the ice rink, boards, and stadium interior), and (ii) one or two foreground objects of interest (an individual skater or a couple of skaters). Rather than encoding full video frames pixel-by-pixel, the scene can be decomposed into a persistent 3D background and a set of dynamic pose-driven elements. Therefore, assuming we possess the 3D point cloud of the stadium, to reconstruct the scene, we must first estimate the camera viewpoint � from the input video scene [16]:

$$
\mathbf { T } = { \left[ \begin{array} { l l } { \mathbf { R } } & { \mathbf { t } } \\ { \mathbf { 0 } ^ { \top } } & { 1 } \end{array} \right] } \in \mathbb { R } ^ { 4 \times 4 }
$$

Here, $\mathbf { R } \in \mathbb { R } ^ { 3 \times 3 }$ is the rotation matrix, and $\mathbf { t } \in \mathbb { R } ^ { 3 }$ is the camera center in world coordinates. The bottom row $[ \pmb { 0 } ^ { \top } 1 ]$ makes it a 4×4 homogeneous transformation matrix. This 3D scene needs to be rendered at the client side within the Unity [46] framework. Since Unity represents any rotation via a quaternion $\mathbf { q } \in \mathbb { R } ^ { 4 }$ , which represents a mathematically convenient alternative to the euler angle representation, we can convert R to q using quaternion algebra [48]. This way, we reduce the required values to represent T from 12 (3 for t and 9 for R) to 7 (3 for t and 4 for q). With the camera viewpoint T and the 3D point cloud of the stadium, the background can be easily reconstructed.

The foreground, on the other hand, includes dynamic objects, i.e., the skaters, that can be cropped out of the captured 2D video, delivered as separate streams to the clients, and overlaid in the camera view based on their original position. Although it requires less bandwidth, this approach presents several technical challenges. Since the bounding box used to extract each skater varies over time, the resulting cropped figures would have non-uniform spatial resolutions, complicating real-time rendering and synchronization on the client side. Padding cropped frames with black pixels to a fixed resolution is bandwidth-ineficient, similar to codec overhead from aligning with fixed-size coding blocks [22]. Transmitting each crop as an independent image sequence is even worse, as it forgoes inter-frame compression gains [40]. These ineficiencies led us to replace pixel-based representations with a minimal structured alternative. Rather than transmitting full pixel-based representations of the skaters, we transmit a compact set of skeletal keypoints

![](images/bb286784ce8718aee04758db3ecca713086651ff9422aa3406417228c6e2b545.jpg)  
Figure 1: Conceptual architecture of GenStream.

$S = \{ ( x ; y ) \mid x \leq W ; y \leq H \}$ , where � and � denote the width and height of the video frame, respectively. These keypoints eficiently encode the motion and structural pose of the skaters over time. The total number � of values necessary to reconstruct the scene is:

$$
V = ( 2 \times | \boldsymbol { S } | + | \mathcal { B } | + | \mathbf { T } | ) \times F \times L\tag{1}
$$

where B is the per-frame set ofvalues representing the bounding box of the skater, |T| is the minimum number of parameters to represent the camera viewpoint T, � is the frame rate of the video in frames per second, and � is the length of the video in seconds. It is worth noting that the generative model can be transmitted to the client before the stream takes place and, therefore, does not contribute to the streaming bandwidth.

Depending on the resolution of the input video, values in B and S can be eficiently represented using 11 bit (1920×1080) or 12 bit (3840×2160) integers. For the camera viewpoint �, we consider 32 bit floating-point values. The motion of each ice skater throughout the whole video, encoded as 17 keypoints [8], requires 64 kB (plus less than 4 kB for the bounding boxes), while camera viewpoints require additional 42 kB. The total data transmitted is, therefore, approximately 110 kB, corresponding to a 99.9% reduction in bandwidth compared to the original 134 MB HEVC-encoded video. It is important to note that this calculation pertains only to the continuous data stream and does not include the one-time, prestream distribution of the 3D venue model and generative model weights, which constitute a significant initial payload. Despite this impressive compression, perceptual coherence is preserved. The reconstructed stadium maintains spatial realism, while the skaters’ poses and trajectories are accurately rendered in appearance and timing. Although conventional quality metrics like VMAF may decline due to the lack of pixel-level fidelity, the semantic content and visual plausibility remain intact, especially for downstream use cases such as highlight replays, performance analysis, or live streaming in low-bandwidth environments. A formal perceptual study is required to quantify this trade-of, which remains a key area for future work.

## 4 System Design

GenStream proposes a radical departure from conventional video pipelines by eliminating pixel-level redundancy at the server and instead transmitting only a sparse, generative codebook of the scene. The system rethinks the encoding of human-centric motion and background information as two disjoint but complementary generative tasks: appearance reconstruction and spatiotemporal placement. Unlike conventional codecs or even object-aware compression pipelines, GenStream treats video as a generative instruction set rather than a sequence of images. At its core is a hybrid pipeline that combines object understanding, 3D scene modeling, and generative synthesis. Figure 1 illustrates the architecture of GenStream, broken down into its server-side and client-side pipelines.

## 4.1 Server-side GenStream

The server-side component transforms raw videos into a set of semantically meaningful layers: dynamic object keypoints, background images, and camera points of view. We describe four key stages below:

Salient Actor Extraction. The pipeline begins by identifying and segmenting salient actors from each frame. This is accomplished via Artificial Intelligence (AI) through object detection and instance segmentation models. To lower the chance of misclassifying a spectator for the actor, we segment only the largest detected human in each frame. This produces an alpha mask that is used to construct two complementary Red Green Blue Alpha (RGBA) layers per frame: (i) A foreground image containing only the actor, and (ii) a background image with the actor masked out.

The segmentation step acts as an attention filter that reduces each frame to its most perceptually critical content, discarding clutter, crowds, and occlusions.

Keypoint-based Pose Encoding. For each segmented actor, Gen-Stream extracts high-resolution keypoint trajectories across time. These form the structured motion encoding that guides generative synthesis on the client side. Our system uses state-of-the-art pose estimation models to track anatomical keypoints of the actor, and encodes them into a temporal stream. The keypoints serve as lightweight motion blueprints for actor synthesis downstream. Unlike traditional video codecs that preserve every pixel, GenStream transmits only these skeletal pose traces, implicitly entrusting the client to fill in visual details through generative inference.

Background Inpainting and 3D Reconstruction. The server assembles a clean, actor-free background model from the masked RGBA background layers. Regions previously occluded by actors are inpainted using context-aware algorithms to restore full-frame realism and aid the Structure-from-Motion (SfM) algorithm to infer camera viewpoint. These poses allow the client to re-render dynamic actors within a coherent 3D scene. This use of background-only frames to drive global camera understanding is a key innovation of Gen-Stream: it decouples the spatiotemporal understanding of scene layout from the actors themselves, enabling more robust reconstructions. It is important to note that this step becomes essential when the 3D model of the venue and camera viewpoints are not provided by the competition organizers apriori.

Encoding for Transmission. Finally, the server packages actor keypoints and camera viewpoints coordinates into a single metadata file. The structured nature of these coordinates allows for a highly optimized compression logic, resulting in massive bitrate gains when compared to the traditional block-based video encoding.

## 4.2 Client-side GenStream

The client-side of GenStream is responsible for converting the structured transmission into a visually plausible, temporally coherent video. Instead of decoding pixels, it interprets abstract control representations (keypoints + poses) and re-renders the video using a generative prior. Three key modules are involved:

Skeleton-Driven Actor Reconstruction. The received keypoint stream is decoded into a full-body actor reconstruction via a conditional generative model. This step synthesizes full-frame RGB actors at arbitrary resolutions and quality levels, depending on the client’s capabilities and time constraints.

Viewpoint-Aware Scene Recomposition. Using the 3D camera viewpoints and original spatial metadata, the client reinserts each synthesized actor into the 3D background, which can be a custom-made representation sent to the client prior to the streaming event, or extracted by the same algorithm that inferred the camera viewpoints. The composition is performed frame-by-frame, guided by the temporal evolution of the keypoints and viewpoint metadata.

Frame Synthesis and Playback. The final video is reconstructed by compositing synthesized actors into the 3D background. This reconstructed sequence is passed to the video player for smooth, low-latency playback. By relying on learned priors and sparse representations, GenStream is robust to bandwidth drops, jitter, or loss of intermediate frames. It behaves more like a generative animation system than a classic video streaming system.

## 4.3 Design Summary

In summary, GenStream replaces blocks of pixels with poses, shifting the core payload from image data to motion semantics, separates actor and background processing, enabling independent treatment and optimization of each layer, and uses client-side generative mod els to recreate known figures, dramatically reducing the required bandwidth. This design explores an alternative to the traditional pixel-based video stack, leveraging semantic understanding to create a blueprint for generative streaming systems. However, this approach introduces a fundamental trade-of: it drastically reduces network bandwidth at the cost of significantly increased computational complexity on the client device. Our proof-of-concept implementation relies on a powerful GPU for client-side rendering, a configuration that is not representative of typical consumer devices like smartphones or smart TVs. GenStream’s client-side burden is a major limitation of the current framework. Future work must explore strategies to mitigate this cost, such as generative model quantization, knowledge distillation to create lightweight client models, or ofloading synthesis to edge servers. Without such optimizations, the practical applicability of GenStream remains limited to powerful client devices.

## 5 Implementation

Content. To validate our approach, we applied the system to a dataset comprising more than 40 000 frames grouped in 48 scenes with duration ranging 9 s to 55 s extracted from four figure skating performances at 30 fps and 1080p resolution [34]. All videos belong to the 2018 Pyongchang Olympic Games, and thus share key characteristics, including: (i) the same ice rink background with predominantly stationary objects and spectators, (ii) one figure skater actively moving in the center of the ice rink, and (iii) a set of cameras, all following the skater from diferent points of the ice rink’s lower ring.

Setup. All processing is performed on an Ubuntu 20.04 LTS server equipped with an Intel Xeon Gold 5218 CPU (64 cores, 2.30 GHz) and an NVIDIA Quadro RTX 8000 GPU with 48 GB of memory. The full system is packaged into two Docker [29] containers: one for the server-side pipeline and another for Unity-based client rendering. The pipeline is implemented in Python using OpenCV, PyTorch, and external tools like COLMAP [42], with Unity handling the rendering pipeline on the client side. Several components of GenStream are fully implemented and tested independently, including actor segmentation, keypoint extraction, inpainting, and data export to 3D scenes. Others, such as camera viewpoint recovery, pose novel challenges and are currently under scrutiny.

Actor Segmentation and Frame Decomposition. Each frame captured by the camera is parsed using YOLOv11 [39], which detects all individuals present. Among them, we retain only the largest bounding box to discard bystanders or referees. This bounding box is passed to Segment Anything Model 2 (SAM 2) [38], which produces a fine segmentation mask, as shown in Figure 2 (Original Skater). SAM 2 is a high precision and reliability segmentation model. However, its complexity does not permit real time performance. Alternatively, the detection and segmentation can be performed in one pass by YOLOv11. This approach is much faster, and compatible with realtime live processing, but its output quality is not as robust. We use the mask to generate, for each frame: (i) actor.png: the actor extracted from the scene, and (ii) background.png: the frame with the actor region masked. The mask is used to add transparency, resulting in RGBA images that clearly diferentiate the portions of the frame that belong to the actor and the background, enabling actor-specific compression and background reconstruction.

![](images/9a6b846dc0c37d55c000138479e0f110ecdaf84ec5574d1ee91a151fd8c8af34.jpg)  
Figure 2: Visual comparison between original frame and various input maps and methods. All images are 1024×1024.

![](images/1db19a852fbc7b8043973acfe51f82aa3ac9a0cb9e31efba13296c9b4a560dc5.jpg)  
Figure 3: Qualitative comparison between original frame (left) and rendering of the 3D model with applied skater (right).

Keypoint extraction. After detection and segmentation, GenStream proceeds to a pose estimation stage. This step aims to convert each segmented actor into a structured representation of keypoints that compactly captures their motion and posture over time. Since the bounding box of the primary actor has already been computed in the segmentation phase, keypoint extraction is performed in parallel, enabling eficient processing. To extract keypoints from each actor, we compare three methods with decreasing spatial density: a Canny edge-based filter [27], a grid sampling filter, and a YOLOv11 Pose-based skeleton detector [26]. The Canny filter highlights contours by detecting high-gradient edges, while the grid filter overlays a regular grid on the edge map and selects representative points from high-detail regions, marking a square’s center when it contains suficient edge pixels. The YOLOv11 Pose model estimates 17 anatomical landmarks directly and connects them into a human skeleton using predefined edges. All three methods, shown in Figure 2, are applied to the segmented RGBA image of the actor, ensuring no background interference. Since all approaches yield comparable perceptual quality in downstream tasks, we select the YOLO-based method due to its compact representation and lower bandwidth requirements.

Background inpainting. To enable a clearer view synthesis, image inpainting is used to dynamically fill regions of the background images where actors have been segmented out. Tools such as OpenCV’s Telea [9] or Stable Difusion [37] are used depending on available computational resources. The output is a sequence of image files without foreground actors, to aid the camera viewpoint estimation phase. While inpainting may introduce some inconsistency across frames, leaving dynamic foreground actors in place typically leads to more severe errors in COLMAP’s reconstruction, as it assumes a static scene. Removing the actors and filling their regions, even imperfectly, helps reduce false feature matches and improve the accuracy of the background structure estimation.

![](images/ca2d7513772ebf62a4c6f14fad99b9d0da19d5f12eb233eb3f0357d8a29b32c5.jpg)  
Figure 4: COLMAP generated point cloud of one scene, complete of camera view intrinsics and extrinsics, figure skater in foreground, and ice rink in the background.

3D Ice Rink Modeling. We manually reconstructed an Olympic ice rink in Unity to serve as the scene’s geometric and visual reference. This includes accurately placed lines, lighting, and reflection properties to match typical broadcast footage (see Fig. 3).

Camera Viewpoint Recovery: Challenges and Approaches (Partially Implemented). Correctly recovering the camera parameters for each video remains one of the most critical and technically demanding components of GenStream. We explore two alternative approaches:

(i) COLMAP-based 3D Reconstruction (Partially Implemented) We use COLMAP to generate a sparse point cloud of the static scene and estimate the per-frame camera intrinsics and extrinsics. Figure 4 shows a 2D projection of a reconstruction result. While COLMAP succeeds in generating plausible camera trajectories, integration with Unity remains incomplete. Converting COLMAP’s coordinate system and focal length parameters into Unity’s camera model introduces geometric inconsistencies that still need to be resolved. Furthermore, to maintain the significant bandwidth reduction that GenStream proposes, COLMAP features extracted from diferent video scenes need to be fused together into a complete and realistic point cloud model, and sent to the client before the live even takes place. The origin of such video scenes may be the owner of the background stadium, or previous scenes that have been transmitted via regular video streaming.

(ii) Diferentiable Camera Viewpoint Optimization (Partially Implemented) Another explored approach involves directly optimizing camera parameters by rendering the 3D scene from a randomly initialized camera angle, capturing the resulting view, and comparing it to the inpainted background from the original video frame. The diference between the rendered and inpainted images serves as a loss function, which is minimized through iterative optimization to estimate the correct camera viewpoint. Eforts in this direction have raised many challenges, both in terms of implementation structure and resulting quality. Furthermore, the computation required for such an optimization is not compatible with live streaming. To address this last challenge, we implemented a more eficient solution: we perform the full optimization only for the first frame of each scene, then use a SIFT-based relative pose estimation [24] to propa gate camera viewpoints to subsequent frames by comparing each one to its immediate predecessor. This significantly reduces the overall computational burden while maintaining adequate tracking accuracy.

Data export (Partially Implemented. Finally, the extracted data, com prising actor bounding boxes and keypoints, and camera viewpoints (or optionally COLMAP’s complete scene information), is saved to a CSV file and then encoded for transmission.

With regards to keypoints transmission, each coordinate is encoded as a 11-bit unsigned integer, suficient to represent pixel positions within a 1080p resolution frame (up to 1920 pixels). Crucially, because the number and ordering of keypoints are fixed and known in advance, each segment of the bitstream corresponds to a specific coordinate of a specific object in a deterministic way. For example, the first 11 bits always encode the X-coordinate of the first keypoint, the next 11 bits the corresponding Y-coordinate, and so on. This layout removes the need for additional metadata, enabling both tight compression and extremely fast parsing on the client side. If an actor is not detected in a particular frame, a sentinel control value (2000, greater than 1920 and thus outside the possible coordinate range) is inserted in place of the missing coordinates. This allows the client to gracefully handle missed detections while maintaining alignment with the expected bitstream format. However, to develop such a tight compression scheme for COLMAP’s features, it is required to solve the related challenges mentioned previously. It is worth noting that the compressed data is delivered to the client at intervals, aligned with the target latency requirements of live video streaming applications.

Generative Model Training Setup. To generate realistic figure skater images from skeleton keypoints, we train a cGAN which receives in input paired image data, where each sample consists of two components: a reconstructed image of the person based on their keypoints, and the corresponding ground truth image. These two images, originally sized at either 128×128, 256×256, or 1024×1024, are concatenated to form paired input images of size 256×128, 512×256, and 2048×1024, respectively. The input images, split into 80 % for training and 20 % for testing, are augmented using random resizing, cropping, and mirroring to enhance robustness.

Client-Side Reconstruction. GenStream’s client-side rendering is implemented in Unity and currently uses the manually constructed 3D model of the Olympic ice rink shown in Figure 3 as its background rendering approach. To complete the synthetic view, the generated figure skater is placed at the correct spatial location along the camera’s optical axis, ensuring that their size and position match the original video. In our current implementation, camera parameters are inferred manually to approximate the original view. Using these values, we can overlay the generated figure skater and achieve a visually plausible frame composition on top of the 3D background. However, this manual solution is clearly not scalable, and useful only for qualitative presentations. Once this challenge is solved, the Unity renderer is already set up to accept per-frame camera values.

## 6 Conclusion

GenStream introduces a framework for video streaming, one that replaces dense pixel transmission with high-level semantic instructions that guide client-side generative reconstruction. By transmitting only sparse skeleton-based keypoints and camera viewpoint metadata, GenStream achieves a bandwidth reduction of over 99.9% compared to conventional HEVC encoding, while preserving essential perceptual cues for immersive viewing. This system provides a blueprint for a new generation of streaming architectures that prioritize semantic content over pixel fidelity. However, several open challenges remain before GenStream can reach its full potential.

As highlighted by our analysis, the most immediate challenge is the significant computational load shifted to the client. Making GenStream practical requires substantial research into lightweight, eficient generative models suitable for resource-constrained devices. This is a primary axis for future work.

A key future direction is the evolution from 2D sprite-like renderings to fully volumetric, viewpoint-consistent representations. Instead of compositing a 2D actor into a 3D scene, we envision modeling the actor as a moving 3D skeleton driving a volumetric avatar. This would allow the system to support free-viewpoint video synthesis, enabling clients to explore scenes from arbitrary perspectives, even those not originally captured by any camera. Realizing this vision introduces two critical challenges. First, it requires the fusion of multiple partial point clouds captured from diferent video sequences into a canonical 3D representation of the actor and background. Second, it demands lifting 2D skeletons to reliable 3D joint configurations in single-camera scenarios. These challenges intersect fields such as multi-scene registration, 3D human reconstruction, and sparse generative modeling. Furthermore, while our current implementation uses Unity for client-side rendering, future deployments might benefit from alternative engines or custom viewers that better align with emerging hardware and web standards. Importantly, GenStream remains renderer-agnostic; any tool capable of interpreting the structured metadata and synthesizing scenes from it could serve as the playback environment. Another important avenue lies in extending GenStream beyond the controlled setting of figure skating to more complex, multi-agent scenes such as football or gymnastics. These scenarios involve richer interactions, dynamic camera work, and frequent occlusions, each posing new demands on segmentation, multi-actor pose estimation, handling occlusions, and synthesizing interactions between generative avatars. Robustly scaling to such complexity is a non-trivial research problem. Finally, we note that background modeling using COLMAP or similar SfM tools requires careful fusion of features from multiple scenes to build a unified point cloud representation. These features can originate from the event organizer’s asset library or be extracted from previously streamed footage. Solving this fusion problem eficiently and compactly is crucial for pre-distributing 3D venue models to clients.

GenStream is a first step toward a future where semantic understanding and generative capabilities replace brute-force video transmission. By ofloading reconstruction to powerful client-side models and transmitting only what truly matters, it ofers a scalable, perceptually meaningful approach to media delivery for the post-codec era.

## 7 Acknowledgments

The financial support of the Austrian Federal Ministry for Digital and Economic Afairs, the National Foundation for Research, Technology and Development, and the Christian Doppler Research Association, is gratefully acknowledged. Christian Doppler Laboratory ATHENA: https://athena.itec.aau.at/

## References

[1] Apple. 2015. HTTP Live Streaming (HLS) Authoring Specification for Apple Devices. https://bit.ly/apple\_hls.

[2] Emanuele Artioli, Farzad Tashtarian, and Christian Timmerer. 2024. DIGIT WISE: Digital Twin-based Modeling of Adaptive Video Streaming Engagement. In Proceedings of the 15th ACM Multimedia Systems Conference (Bari, Italy) (MM-Sys ’24). Association for Computing Machinery, New York, NY, USA, 78–88. doi:10.1145/3625468.3647613

[3] Emanuele Artioli, Farzad Tashtarian, and Christian Timmerer. 2025. End-to-End Learning-based Video Streaming Enhancement Pipeline: A Generative AI Approach. In Proceedings of the 35th Workshop on Network and Operating System Support for Digital Audio and Video (Stellenbosch, South Africa) (NOSSDAV ’25). Association for Computing Machinery, New York, NY, USA, 50–56. doi:10.1145/ 3712678.3721881

[4] Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. 2020. BlazePose: On-device Real-time Body Pose tracking. arXiv:2006.10204 [cs.CV] https://arxiv.org/abs/2006.10204

[5] Abdelhak Bentaleb, Bayan Taani, Ali C. Begen, Christian Timmerer, and Roger Zimmermann. 2019. A Survey on Bitrate Adaptation Schemes for Streaming Media Over HTTP. IEEE Communications Surveys & Tutorials 21, 1 (2019), 562– 585. doi:10.1109/COMST.2018.2862938

[6] Bitmovin Inc. 2025. The Annual Bitmovin Video Developer Report. https: //bitmovin.com/video-developer-report/.

[7] Benjamin Bross, Ye-Kui Wang, Yan Ye, Shan Liu, Jianle Chen, Gary J. Sullivan, and Jens-Rainer Ohm. 2021. Overview of the Versatile Video Coding (VVC) Standard and its Applications. IEEE Transactions on Circuits and Systems for Video Technology 31, 10 (2021), 3736–3764. doi:10.1109/TCSVT.2021.3101953

[8] Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. 2021. OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Afinity Fields. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 1 (2021), 172– 186. doi:10.1109/TPAMI.2019.292925

[9] Preeti Chatterjee, Subhadeep Jana, and Souradeep Ghosh. 2021. Comparative study of opencv inpainting algorithms. Global Journal ofComputer Science and Technology 21, 2 (2021), 27–37.

[10] Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser-Nam Lim, and Abhinav Shrivastava. 2021. NeRV: neural representations for videos. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems (NIPS ’21). Curran Associates Inc., Red Hook, NY, USA, Article 1649, 12 pages.

[11] Yue Chen, Debargha Murherjee,Jingning Han, Adrian Grange, Yaowu Xu, Zoe Liu, Sarah Parker, Cheng Chen, Hui Su, Urvang Joshi, Ching-Han Chiang, Yunqing Wang, Paul Wilkins, Jim Bankoski, Luc Trudeau, Nathan Egge, Jean-Marc Valin, Thomas Davies, Steinar Midtskogen, Andrey Norkin, and Peter de Rivaz. 2018. An Overview of Core Coding Tools in the AV1 Video Codec. In 2018 Picture Coding Symposium (PCS). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 41–45. doi:10.1109/PCS.2018.8456249

[12] Dash Industry Forum. 2025. Client Implementation for the Playback of MPEG-DASH via Javascript. https://github.com/Dash-Industry-Forum/dash.js.

[13] Andrew J Davison, Ian D Reid, Nicholas D Molton, and Olivier Stasse. 2007. MonoSLAM: Real-time single camera SLAM. IEEE transactions on pattern analysis and machine intelligence 29, 6 (2007), 1052–1067.

[14] Xize Duan, Yili Jin, Lei Zhang, and Fangxin Wang. 2025. SemConf: A System for Multiparty Semantic Video Conferencing. In Proceedings ofthe 35th Workshop on Network and Operating System Support for Digital Audio and Video (Stellenbosch, South Africa) (NOSSDAV’25). Association for Computing Machinery, New York, NY, USA, 71–77. doi:10.1145/3712678.3721884

[15] Erfan Entezami and Hui Guan. 2024. AI-Driven Innovations in Volumetric Video Streaming: A Review. arXiv:2412.12208 [cs.CV] https://arxiv.org/abs/2412.12208

[16] Richard Hartley and Andrew Zisserman. 2004. Multiple View Geometry in Computer Vision (2 ed.). Cambridge University Press, Shaftesbury Road, Cambridge, CB2 8EA, United Kingdom.

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 574, 12 pages.

[18] Peiwen Jiang, Chao-Kai Wen, Shi Jin, and Geofrey Ye Li. 2022. Wireless semantic communications for video conferencing. IEEE Journal on Selected Areas in Communications 41, 1 (2022), 230–244.

[19] Sheela Raju Kurupathi, Veeru Dumpala, and Didier Stricker. 2023. Multi-stage Conditional GAN Architectures for Person-Image Generation. In Deep Learning Theory and Applications, Ana Fred, Carlo Sansone, and Kurosh Madani (Eds.). Springer Nature Switzerland, Cham, 24–48.

[20] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. 2017. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. arXiv:1609.04802 [cs.CV] https://arxiv.org/abs/1609.04802

[21] Jiakun Li, Yuan Zhang, Lingjun Pu, Tao Lin, and Jinyao Yan. 2025. Bimodal Semantic-Driven 3D Immersive Telepresence System. In Proceedings ofthe 35th Workshop on Network and Operating System Supportfor Digital Audio and Video (Stellenbosch, South Africa) (NOSSDAV ’25). Association for Computing Machinery, New York, NY, USA, 57–63. doi:10.1145/3712678.3721882

[22] Ming Li, Yilin Chang, Fuzheng Yang, and Shuai Wan. 2010. Rate-distortion criterion based picture padding for arbitrary resolution video coding using H. 264/MPEG-4 AVC. IEEE transactions on circuits and systems for video technology 20, 9 (2010), 1233–1241.

[23] Dong Liu, Yue Li, Jianping Lin, Houqiang Li, and Feng Wu. 2020. Deep Learning-Based Video Coding: A Review and a Case Study. Comput. Surveys 53, 1 (2020), 1–35. doi:10.1145/3368405

[24] G Lowe. 2004. Sift-the scale invariant feature transform. Int. J 2, 91-110 (2004), 2.

[25] Guo Lu, Wanli Ouyang, Dong Xu, Xiaoyun Zhang, Chunlei Cai, and Zhiyong Gao. 2019. DVC: An End-to-end Deep Video Compression Framework. arXiv:1812.00101 [eess.IV] https://arxiv.org/abs/1812.00101

[26] Debapriya Maji, Soyeb Nagori, Manu Mathew, and Deepak Poddar. 2022. YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 2636–2645. doi:10.1109/CVPRW56347.2022.00297

[27] William McIlhagga. 2011. The Canny Edge Detector Revisited. International Journal of Computer Vision 91 (2011), 251–261. doi:10.1007/s11263-010-0392-0

[28] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. 2022. TrackFormer: Multi-Object Tracking with Transformers . In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 8834–8844. doi:10.1109/CVPR52688. 2022.00864

[29] Dirk Merkel. 2014. Docker: Lightweight Linux Containers for Consistent Development and Deployment. Linux Journal, https://www.linuxjournal.com/ content/ docker-lightweight-linux-containers-consistent-development-and-deployment 239, 2 (2014), 2.

[30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

[31] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. 2015. ORB SLAM: A versatile and accurate monocular SLAM system. IEEE transactions on robotics 31, 5 (2015), 1147–1163.

[32] Minh Nguyen, Daniele Lorenzi, Farzad Tashtarian, Hermann Hellwagner, and Christian Timmerer. 2022. DoFP+: An HTTP/3-Based Adaptive Bitrate Approach Using Retransmission Techniques. IEEE Access 10 (2022), 109565–109579. doi:10. 1109/ACCESS.2022.3214827

[33] Simon Niklaus and Feng Liu. 2020. Softmax Splatting for Video Frame Interpolation . In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 5436–5445. doi:10.1109/CVPR42600.2020.00548

[34] Olympics. 2018. Yuzuru Hanyu (JPN) - Gold Medal | Men’s Figure Skating | Free Programme | PyeongChang 2018. https://www.youtube.com/watch?v= 23EfsN7vEOA.

[35] Hanhoon Park. 2024. Semantic Super-Resolution via Self-Distillation and Adversarial Learning. IEEE Access 12 (2024), 2361–2370. doi:10.1109/ACCESS.2023. 3349023

[36] Fernando C. Pereira and Touradj Ebrahimi. 2002. The MPEG-4 Book. Prentice Hall PTR, USA.

[37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Difusion Models for High-Resolution Image Synthesis. arXiv:2307.01952 [cs.CV] https://arxiv.org/abs/2307.01952

[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714 [cs.CV] https://arxiv.org/abs/ 2408.00714

[39] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. You Only Look Once: Unified, Real-Time Object Detection. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 779–788. doi:10.1109/CVPR.2016.91

[40] Khalid AM Salih, Ismail Amin Ali, and Ramadhan J Mstafa. 2024. Impact of Video Motion Content on HEVC Coding Eficiency. Computers 13, 8 (2024), 204.

[41] Sandvine Holdings UK Limited. 2022. 2022 Global Internet Phenomena Report. https://www.sandvine.com/global-internet-phenomena-report-2022.

[42] Johannes L. Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 4104–4113. doi:10.1109/CVPR.2016.445

[43] Robert Skupin, Christian Bartnik, Adam Wieckowski, Yago Sanchez, Benjamin Bross, Cornelius Hellge, and Thomas Schierl. 2021. Open GOP Resolution Switch ing in HTTP Adaptive Streaming with VVC. In 2021 Picture Coding Symposium (PCS). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 1–5. doi:10.1109/PCS50896.2021.9477501

[44] Thomas Stockhammer. 2011. Dynamic adaptive streaming over HTTP –: standards and design principles. In Proceedings ofthe Second Annual ACM Conference on Multimedia Systems (San Jose, CA, USA) (MMSys ’11). Association for Com puting Machinery, New York, NY, USA, 133–144. doi:10.1145/1943552.1943572

[45] Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. 2012. Overview of the High Eficiency Video Coding (HEVC) Standard. IEEE Transactions on Circuits and Systems for Video Technology 22, 12 (2012), 1649–1668. doi:10.1109/TCSVT.2012.2221191

[46] Unity Technologies. 2025. Unity Real-Time Development Platform | 3D, 2D, VR 0026 AR Engine. https://unity.com/.<sup>˘</sup>

[47] Sudheendra Vijayanarasimhan, Susanna Ricco, Cordelia Schmid, Rahul Sukthankar, and Katerina Fragkiadaki. 2017. SfM-Net: Learning of Structure and Motion from Video. ArXiv abs/1704.07804 (2017). https://api.semanticscholar. org/CorpusID:6192751

[48] John Voight. 2021. Quaternion algebras. Springer Nature, Europaplatz 3, D-69115 Heidelberg.

[49] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. 2021. One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 10034–10044. doi:10.1109/CVPR46437. 2021.00991

[50] Warner Bros. Discovery Sports Europe. 2024. Record Olympics Streaming and TV Audiences for Warner Bros. Discovery as MAX Boosts Viewership and En gagement Across Europe. https://media.wbdsports.com/post/record-olympicsstreaming-and-tv-audiences-for-warner-bros-disco.

[51] Thomas Wiegand, Gary J Sullivan, Gisle Bjontegaard, and Ajay Luthra. 2003. Overview ofthe H. 264/AVC Video Coding Standard. IEEE Transactions on Circuits and Systems for Video Technology 13, 7 (2003), 560–576. doi:10.1109/TCSVT.2003. 815165

[52] Xiangyu Xu, Li Siyao, Wenxiu Sun, Qian Yin, and Ming-Hsuan Yang. 2019. Quadratic video interpolation. Curran Associates Inc., Red Hook, NY, USA, 5436–5445.

[53] Tong Zhang, Fengyuan Ren, Wenxue Cheng, Xiaohui Luo, Ran Shu, and Xiaolan Liu. 2019. Towards influence of chunk size variation on video streaming in wireless networks. IEEE Transactions on Mobile Computing 19, 7 (2019), 1715– 1730.

[54] Xue Zhang, Laura Toni, Pascal Frossard, Yao Zhao, and Chunyu Lin. 2018. Adaptive streaming in interactive multiview video systems. IEEE Transactions on Circuits and Systems for Video Technology 29, 4 (2018), 1130–1144.

[55] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. 2022. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision. Springer, Springer Nature, Europaplatz 3, D-69115 Heidelberg, 1–21.

[56] Yongfei Zhang, Chao Zhang, Rui Fan, Siwei Ma, Zhibo Chen, and C.-C. Jay Kuo. 2020. Recent Advances on HEVC Inter-Frame Coding: From Optimization to Implementation and Beyond. IEEE Transactions on Circuits and Systems for Video Technology 30, 11 (Nov. 2020), 4321–4339. doi:10.1109/tcsvt.2019.2954474

[57] Zhengdong Zhang and Vivienne Sze. 2017. FAST: A Framework to Accelerate Super-Resolution Processing on Compressed Videos. arXiv:1603.08968 [cs.CV] https://arxiv.org/abs/1603.08968

[58] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. 2024. Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, IEEE Publications, 445 Hoes Lane, Piscataway, NJ 08855 USA, 21676–21685. doi:10.1109/CVPR52733.2024.02048

[59] Chen Zhu, Guo Lu, Bing He, Rong Xie, and Li Song. 2023. Implicitexplicit Integrated Representations for Multi-view Video Compression. arXiv:2311.17350 [cs.CV] https://arxiv.org/abs/2311.17350

[60] Yuanwei Zhu, Yakun Huang, Xiuquan Qiao, Zhijie Tan, Boyuan Bai, Huadong Ma, and Schahram Dustdar. 2022. A semantic-aware transmission with adaptive control scheme for volumetric video service. IEEE Transactions on Multimedia 25 (2022), 7160–7172.