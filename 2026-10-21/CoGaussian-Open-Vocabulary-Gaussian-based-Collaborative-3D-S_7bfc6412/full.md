Joint Training

Ground Truth

Zero-shot Concatenation

# CoGaussian: Open-Vocabulary Gaussian-based Collaborative 3D Scene Understanding

Om Shivam Verma<sup>1∗</sup>, Shaya Farahmand<sup>1∗</sup>, Lingjun Zhao<sup>1</sup>, Lu Gan<sup>1</sup>

Abstract— Collaborative perception enables agents to retrieve the scene semantics and geometry beyond the field of view of any individual agent. This broader scene understanding enhances transportation safety and improves efficiency in multi-robot systems. However, existing collaborative semantic occupancy methods rely on dense 3D voxel annotations and are limited to recognizing objects within a predefined set of semantic classes. To overcome these limitations, we present CoGaussian, an open-vocabulary Gaussian-based collaborative semantic occupancy prediction framework. CoGaussian represents a scene as a set of foundation model-aligned sparse 3D Gaussians, which serve as a compact representation for communicating and fusing geometric and semantic information across multiple agents. On the Semantic-OPV2V benchmark, our method outperforms existing open-vocabulary single-agent approaches, highlighting the effectiveness of open-vocabulary collaborative scene understanding.

## I. INTRODUCTION

Collaborative perception enables local agents to exchange information that may be occluded or beyond their fieldof-view, improving geometric and semantic understanding of their environment, and enabling safer and more reliable decision making in autonomous driving and robot navigation. Semantic occupancy prediction (SOP) is a representative and challenging task in 3D spatial understanding, which aims to predict voxel-wise occupancy status and semantic class. Recently, 3D Gaussian-based methods [1] have been shown to be effective for SOP. The GaussianFormer family [2]– [5] represents scenes as 3D Gaussians, supervised by dense occupancy labels. VOGS-CP [6] extends GaussianFormer to collaborative SOP which communicates Gaussian primitives between neighboring agents. However, these methods are limited to predicting semantics within a closed vocabulary and heavily rely on densely annotated 3D occupancy labels that are expensive to obtain.

Vision foundation models (VFMs), such as DINO [7] and CLIP [8] have demonstrated strong open-vocabulary capabilities in 2D perception. In addition, Talk2DINO [9] and DINO.txt [10] further align 2D VFM features with natural language. Building on the above VFMs, GaussTR [11] represents scenes as a set of sparse open-vocabulary 3D Gaussians that are predicted in a feed-forward manner and can be queried via natural language. The Gaussians are encoded with geometry and semantic information by splatting into 2D views and aligning rendered features with foundation modelderived depth [12] and semantics [7], [8]. However, GaussTR is designed for single-agent perception and cannot leverage information that is visible only to neighboring agents.

![](images/9374709a9837df32c287789f0d306edd4bea4fa9a7db8e263cbf8914ad88ed7c.jpg)

![](images/46e82e578ca2f9d9f645da5b44e1fc414218915d70c3c41d9fbf1ad9e6fca833.jpg)

![](images/3222b9650d35e1b43605e9d65f69316209b7fa117eef09a833a66ff331810a83.jpg)  
Radius-Gated Attention

![](images/8bb03ea50b00fd467988032d93406a0da01c0acc2f984d88c8251ced7f95ddad.jpg)  
Fig. 1. Qualitative comparison of three variants of CoGaussian with ground truth. The ego vehicle is marked as E. Another agent is in the front-right (not visible here), partially completing the front-right of the ego vehicle in collaborative predictions. Class legends are shown in Table I.

Based on these observations, we propose CoGaussian, the first open-vocabulary multi-robot collaborative perception framework. We represent scenes as sets of sparse VFMaligned 3D Gaussians, which also serve as the communication primitives among agents. We also explore and develop three different algorithms to aggregate and fuse information from neighboring agents. Our contributions can be summarized as follows:

• Open-Vocabulary Collaborative Perception: We propose CoGaussian, the first framework for openvocabulary collaborative perception (e.g. 3D semantic occupancy prediction) in outdoor environments.

• Multi-Robot 3D Gaussian Representation: We develop a foundation model-aligned 3D Gaussian representation for collaborative perception, along with fusion strategies to enhance information transfer among agents.

• Zero-Shot Benchmarking: We evaluate CoGaussian on the Semantic-OPV2V [13], [14] dataset in a zeroshot manner. Without any task-specific supervision, our approach achieves impressive performance for semantic occupancy prediction, highlighting the potential for open-vocabulary collaborative perception.

![](images/29fbbec40fd746ba6bf515fd1ba259a3f55fa4be27fed575a92d51c3d170ce59.jpg)  
Fig. 2. Overview. CoGaussian starts with randomly initialized embeddings. A transformer layer uses multiscale features of images to update the embeddings, followed by a Head Layer that predicts Gaussians. During Joint Training, all agents predict ego Gaussians which are communicated with each other and aggregated. This set of Gaussians is used for training. In the Fusion pipeline, the neighbors predict and send Gaussian embeddings to the ego agent. Radius-gated Attention uses these neighbor Gaussian embeddings to update ego embeddings. The Fusion Head predicts ego Gaussians, and pre-trained Head from Joint Training predicts neighbor Gaussians. These are aggregated to form the final prediction used for both training and inference.

## II. METHOD

We represent a scene as a sparse set of 3D Gaussians with associated DINOv2 [7] features. Predicted Gaussians are splatted back into the source camera view and supervised by DINOv2 [7] features and Metric3Dv2 [12] monocular depth. During inference, semantic class logits are computed as similarity between the predicted Gaussian features and Talk2DINO [9] text embeddings for each class.

## A. Preliminaries

Each 3D Gaussian $G$ is parametrized as:

$$
G = \{ \mu , \mathbf { s } , \mathbf { r } , \alpha , \mathbf { f } \} ,\tag{1}
$$

where $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ is the mean (center), $\textbf { s } \in \ \mathbb { R } ^ { 3 }$ is the peraxis scale, $\textbf { r } \in \mathbb { R } ^ { 4 }$ is the rotation quaternion of the axes, $\alpha \in [ 0 , 1 ]$ is the opacity, and $\mathbf { f } \in \mathbb { R } ^ { D }$ is the high-dimensional DINOv2 feature vector. The model inputs an image and initializes a set of Gaussian embeddings $\{ \mathbf { q _ { i } } \in \mathbb { R } ^ { E } \} _ { i = 1 } ^ { N } ,$ where $E$ denotes the dimension of the embedding. Each embedding vector is associated with a randomly sampled location $( u _ { i } , v _ { i } ) \in [ 0 , 1 ] ^ { 2 }$ on the image and its corresponding interpolated Metric3D depth value. The coordinates are concatenated with the depth value, and then projected into 3D space via camera intrinsics and LiDAR-camera extrinsics to obtain the Gaussian’s mean position $\pmb { \mu }$ in the LiDAR frame. Once the Gaussian embeddings are initialized, they are refined through a set of transformer [15] layers $\mathcal { F }$ that consists of a deformable attention module [16], a selfattention module, and an MLP layer, as shown in Fig. 2. The point $( u _ { i } , v _ { i } )$ serves as the reference point for deformable attention over the image features:

$$
\mathbf { q _ { i } ^ { \prime } } = \mathcal { F } ( \mathbf { q _ { i } } , ( u _ { i } , v _ { i } ) ) .\tag{2}
$$

Finally, the Gaussian parameters are predicted from embed dings $\mathbf { q _ { i } }$ using an MLP head:

$$
g _ { i } : = \{ \Delta \mu _ { i } , \mathbf { s } _ { i } , \mathbf { r } _ { i } , \alpha _ { i } , \mathbf { f } _ { i } \} = \mathrm { M L P } _ { \mathrm { G H } } ( \mathbf { q } _ { \mathrm { i } } ^ { \prime } ) .\tag{3}
$$

The means $\mu$ are iteratively refined after each layer using the predicted offsets $\Delta \mu ,$ , while the scale s and opacity α are scaled to appropriate ranges. The rotations r are fixed to be axis-aligned in the camera’s coordinate frame, and are not learned to simplify training. The model is trained by differentiably rendering the DINOv2 features and depth from the predicted Gaussians using alpha-blending, implemented with optimized CUDA kernels [17]. The training objective penalizes the difference between the rendered outputs and corresponding predicted depth and features [11].

## B. Joint Training

GaussTR [11] is trained per camera view, with Gaussians predicted from each camera splatted back into that view for supervision. While this approach is effective for datasets with limited camera overlap, such as in the nuScenes dataset [18], it is suboptimal for multi-agent perception, where independent supervision fails to exploit overlapping observations across agents. Therefore, the model is trained per scene as shown in Fig. 2. All Gaussians predicted from cameras in a scene are aggregated and splatted into every camera in which they are visible, not only the cameras that produced them. Thus, predictions consistent with a single view but inconsistent with other views are penalized. During inference, Gaussians are predicted from each camera, aggregated, and then used by agents for downstream tasks. CoGaussian can use weights from either per-image or perscene training. In Sec. III, we refer to the former as Zero-shot Concatenation, which combines per-image training with our proposed inference method, and the later as Joint Training, which uses per-scene training to enable collaboration.

## C. Radius-Gated Attention Fusion

Joint Training improves performance as reported in Table I. However, aggregating Gaussians from different cameras with overlapping fields of view causes the respective Gaussians to overlap. While this adds information for the ego, it also introduces noise. Additionally, the appearance of the same object often varies across different viewpoints. Thus, Gaussians for the same object may differ widely. Joint training helps mitigate such differences, but it does so implicitly during Gaussian prediction. Therefore, we make this explicit by introducing a fusion module. The fusion module aggregates useful information from Gaussians predicted by neighboring agents. We use single-head scaled dot product attention [15] between ego Gaussian embeddings and neighboring Gaussian embeddings. Afterwards, a Fusion Head with the same architecture as the Gaussian Head predicts Gaussians from the updated embeddings. The updated ego Gaussians and the neighboring Gaussians are aggregated to form the final set used for downstream tasks. The fusion pipeline is shown in Fig. 2.

Selecting neighbor Gaussians. Let $\mathcal { G } _ { e g o }$ be the set of ego Gaussians and $\mathcal { G } _ { n g h }$ the set of neighbor Gaussians, transformed into ego’s coordinate frame. For each ego Gaussian $g _ { i } ~ \in ~ \mathcal { G } _ { e g o }$ , we take its k nearest neighbors in $\mathcal { G } _ { n g h }$ by Euclidean distance between Gaussian means, denoted by $\textstyle { \mathcal { N } } ( g _ { i } )$ . Then, we create an attention mask M based on distance between $g _ { i }$ and $g _ { j } ~ \in ~ \mathcal { N } ( g _ { i } )$ : Both k and $\rho$ are hyperparameters. Taking the k nearest neighbors imposes an intuitive locality bias: an ego gaussian can only be updated by nearby neighbor Gaussians. The bias is more strongly enforced by allowing Gaussians only within a specific radius $\rho$ to contribute to the update.

Attention over Gaussian embeddings. The fusion uses Gaussian embeddings obtained from the transformer layers on both ego and neighboring agents along with the optimized mean. The neighbor Gaussian embeddings and their means are communicated to the ego agent, where the attention matrix is computed:

$$
\Delta \pmb { \mu } _ { i , j } = \pmb { \mu } _ { j } - \pmb { \mu } _ { i } , \quad \mathbf { z } _ { i , j } = \big [ \mathbf { q } _ { j } \parallel \Delta \pmb { \mu } _ { i , j } \parallel \eta ( \Delta \pmb { \mu } _ { i , j } ) \big ] ,\tag{4}
$$

$$
\mathbf { Q } _ { i } = \phi _ { q } ( \mathbf { q } _ { i } ) , \ \mathbf { K } _ { i , j } = \phi _ { k } ( \mathbf { z } _ { i , j } ) , \ \mathbf { V } _ { i , j } = \phi _ { v } ( \mathbf { z } _ { i , j } ) ,\tag{5}
$$

$$
A _ { i , j } = { \frac { \langle \mathbf { Q } _ { i } , \mathbf { K } _ { i , j } \rangle } { \sqrt { d } } } .\tag{6}
$$

Here, $\mathbf { q } _ { i }$ is the embedding corresponding to Gaussian $g _ { i }$ $\Delta \mu _ { i , j }$ is the relative position of $g _ { j }$ with respect to $g _ { i } , \phi _ { q } ,$ $\phi _ { k }$ , and $\phi _ { v }$ are two-layer MLPs and ∥ is concatenation. The query vector is built from ego Gaussian embedding alone. The key and value vectors combine the neighbor Gaussian embeddings with their position relative to ego Gaussian along with its positional encoding η( $\Delta \mu _ { i , j } )$ . d is the dimension of $\mathbf { Q } _ { i }$ and $\mathbf { K } _ { i , j }$

$$
w _ { i , j } = \frac { \mathrm { s o f t m a x } _ { j } ( A _ { i , j } ) M _ { i , j } } { \sum _ { j ^ { \prime } } \mathrm { s o f t m a x } _ { j ^ { \prime } } ( A _ { i , j ^ { \prime } } ) M _ { i , j ^ { \prime } } + \epsilon } ,\tag{7}
$$

$$
\mathbf { c } _ { i } = \sum _ { j \in \mathcal { N } ( g _ { i } ) } w _ { i , j } \mathbf { V } _ { i , j } , \quad \hat { \mathbf { q } } _ { i } = \mathbf { q } _ { i } + \lambda \psi ( \mathbf { c } _ { i } ) .\tag{8}
$$

The attention mask is applied while calculating the softmax, which allows only radius-gated neighboring Gaussians to have non-zero weights $w _ { i , j }$ . An MLP $\psi$ is used to obtain an update which is scaled by λ and added to the Gaussian embedding.

Training. We follow a two-stage training paradigm. Stage 1: We train the Transformer Layer and Gaussian Head as described in Sec. II-B. Stage 2: The Transformer Layer and Gaussian Head are frozen, and the model learns the Radius-Gated Attention Layer and the Fusion Head.

## III. EXPERIMENTS

## A. Dataset and Metrics

We evaluate on the Semantic-OPV2V benchmark [14], an extension of OPV2V dataset [13] with 3D SOP ground truth. OPV2V is a simulated dataset for vehicle-to-vehicle (V2V) collaboration with two to seven collaborating agents in a scene. Each vehicle is equipped with 4 cameras. The ground truth consists of 12 semantic labels over a $4 0 \times 4 0 \times 3 . 2 m$ area with a grid size of $1 0 0 \times 1 0 0 \times 8 .$ , giving a voxel size of 0.4m. Following previous work [6], [14], we report the perclass Intersection-over-Union (IoU), and its mean (mIoU), along with the overall occupancy IoU. Even though fullysupervised methods are included for reference, they are not comparable to our approach as they use dense 3D voxel supervision.

## B. Implementation Details

The model is trained per-view for 6 epochs for singleagent and zero-shot concatenation. Joint per-scene training is then run for 3 epochs. For Radius-Gated Attention Fusion, the transformer is frozen, and the attention module and fusion head are trained for 3 epochs. All training runs use a single NVIDIA A100 GPU and take 8-14 hours. The losses converge within 2 epochs for all runs. Metric3Dv2 [12] provides depth supervision, and DINOv2 [7] provides openvocabulary feature supervision. During inference, Gaussian features are compared with text embeddings aligned to DINOv2 using Talk2DINO [9], prompted with the 12 classes plus an extra class representing free space. We use AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ , weight decay of $5 \times 1 0 ^ { - 3 }$ and gradient clipping of 20. We evaluate models using 1200 Gaussian queries per camera in four configurations:

TABLE I  
3D SEMANTIC OCCUPANCY PREDICTION ON SEMANTIC-OPV2V DATASET
<table><tr><td rowspan="5">Method</td><td colspan="6">Fully-Supervised, Closed Vocabulary</td><td colspan="4">Self-Supervised, Open Vocabulary</td></tr><tr><td colspan="2">Single-Agent</td><td colspan="4">Collaborative Perception</td><td colspan="4">Single-Agent</td></tr><tr><td colspan="2">CoHFF GSFormer</td><td>CoHFF</td><td colspan="3">VOGS-CP</td><td>GaussTR</td><td colspan="3">CoGaussian (Ours)</td></tr><tr><td colspan="2"></td><td></td><td>Zero Shot</td><td>Naive Fusion</td><td>Learned Fusion</td><td></td><td>Concatenation</td><td>Trained</td><td>Attention</td></tr><tr><td>IoU (↑)</td><td>38.52</td><td>67.76</td><td>50.46</td><td>67.88</td><td>70.10</td><td>72.87</td><td>20.58</td><td>16.13</td><td>22.54</td><td>22.57</td></tr><tr><td>mIoU (1)</td><td>24.81</td><td>29.20</td><td>34.16</td><td>30.54</td><td>36.02</td><td>37.44</td><td>6.26</td><td>4.81</td><td>6.74</td><td>6.95</td></tr><tr><td>Building</td><td>21.04</td><td>3.84</td><td>25.72</td><td>3.91</td><td>7.18</td><td>9.61</td><td>2.03</td><td>1.25</td><td>1.65</td><td>1.93</td></tr><tr><td>Fence</td><td>20.05</td><td>14.10</td><td>27.83</td><td>17.29</td><td>25.66</td><td>29.20</td><td>3.19</td><td>2.61</td><td>3.80</td><td>3.39</td></tr><tr><td>Terrain</td><td>43.93</td><td>68.97</td><td>48.30</td><td>72.00</td><td>76.05</td><td>74.51</td><td>2.86</td><td>1.71</td><td>2.27</td><td>2.97</td></tr><tr><td>Pole</td><td>31.66</td><td>5.94</td><td>42.74</td><td>8.44</td><td>12.67</td><td>12.19</td><td>0.95</td><td>0.87</td><td>0.96</td><td>1.04</td></tr><tr><td>Road</td><td>55.83</td><td>79.37</td><td>61.77</td><td>81.35</td><td>78.60</td><td>83.05</td><td>27.68</td><td>19.71</td><td>32.41</td><td>32.70</td></tr><tr><td>Side walk</td><td>17.31</td><td>70.55</td><td>39.62</td><td>68.91</td><td>75.88</td><td>78.22</td><td>14.90</td><td>12.57</td><td>13.62</td><td>17.29</td></tr><tr><td>Vegetation</td><td>14.49</td><td>12.54</td><td>20.59</td><td>13.72</td><td>16.42</td><td>20.43</td><td>5.67</td><td>4.64</td><td>5.88</td><td>5.39</td></tr><tr><td>Vehicles</td><td>58.55</td><td>49.25</td><td>63.28</td><td>49.56</td><td>57.14</td><td>60.49</td><td>12.55</td><td>9.88</td><td>15.39</td><td>14.24</td></tr><tr><td>Wall</td><td>33.30</td><td>30.79</td><td>58.27</td><td>30.49</td><td>35.25</td><td>36.45</td><td>3.17</td><td>2.83</td><td>2.49</td><td>2.53</td></tr><tr><td>Guard rail</td><td>1.54</td><td>15.01</td><td>1.94</td><td>20.78</td><td>41.00</td><td>32.50</td><td>0.31</td><td>0.16</td><td>0.85</td><td>0.31</td></tr><tr><td>Traffic signs</td><td>0</td><td>0</td><td>16.33</td><td>0</td><td>6.35</td><td>8.26</td><td>0.45</td><td>0.44</td><td>0.52</td><td>0.52</td></tr><tr><td>Bridge</td><td>0</td><td>0</td><td>3.53</td><td>0</td><td>0</td><td>4.35</td><td>1.33</td><td>1.08</td><td>1.07</td><td>1.05</td></tr></table>

• Single Agent Perception: Each agent independently predicts occupancy.

• Zero-shot Concatenation: Gaussians predicted by all agents are aggregated in a common coordinate frame.

• Joint Training: Aggregated Gaussians from all cameras are used for fine-tuning.

• Radius-Gated Attention: The fusion pipeline runs for each agent, updating ego Gaussians using neighboring Gaussians.

## IV. RESULTS AND DISCUSSION

## A. Quantitative Results

Table I highlights quantitative results from previous works, as well as our experiments. Relative to the singleagent GaussTR baseline, zero-shot concatenation reduces performance by −4.45/−1.45 (IoU/mIoU), showing that naive Gaussian aggregation introduces spatially inconsistent structure. Joint Training improves over the baseline by $\mathbf { + 1 . 9 6 / + 0 . 4 8 }$ (IoU/mIoU), indicating that crossview rendering supervision reconciles these inconsistencies. Using Radius-Gated Attention gives a modest gain of +1.99/+0.69 (IoU/mIoU) over the baseline.

Class-wise behavior is mixed, with frequently observed classes like “road”, “sidewalk”, and “vehicles” benefiting the most from collaboration, as they have major overlap among agents and thus a stronger supervision signal. Prominent but less visible classes like ”vegetation” and ”terrain” struggle with collaboration and other small and uncommon classes have mixed results.

## B. Qualitative Results

Fig. 1 shows that the collaborative methods recover structure outside the ego view. Our method tends to produce volumetric predictions rather than one voxel thick surfaces in the ground truth. This reflects the nature of sparse 3D Gaussian representations, unlike methods like [2]–[4] that rely on 3D semantic voxel supervision and many surface-fitting small Gaussians. 2D image-based supervision also makes objects occupying large image regions more prominent in 3D space. The elongated prediction of Vehicle 1 and volumetric prediction of Vehicle 2 demonstrate this behavior.

Joint Training reduces these inconsistencies, producing smaller and more conservative Gaussians, evident in Vehicle 2 and smaller elliptical predictions throughout the scene. The fusion pipeline preserves this coherence while recovering more dominant regions of the scene, including the Road and some Vehicles. However, thin and infrequent structures remain difficult to reconstruct reliably under purely 2D self-supervision.

## V. CONCLUSION AND FUTURE WORK

We introduce CoGaussian which is, to the best of our knowledge, the first framework for open-vocabulary collaborative perception that communicates foundation modelaligned 3D Gaussians among agents to represent a scene. Evaluation shows that collaboration is effective when interaction is learned. Zero-shot collaboration hurts performance, whereas joint training and radius-gated attention improve over the single-agent baseline. CoGaussian also brings the flexibility of open-vocabulary to collaborative perception.

We plan to explore different fusion architectures and evaluate additional downstream tasks such as BEV Semantic Segmentation and 3D Object Detection. Efficient communication and inference optimization will also be crucial for enabling real-time implementation.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3D Gaussian¨ Splatting for Real-time Radiance Field Rendering,” ACM TOG, 2023.

[2] Y. Huang, W. Zheng, Y. Zhang, J. Zhou, and J. Lu, “GaussianFormer: Scene as Gaussians for Vision-based 3D Semantic Occupancy Prediction,” in ECCV, 2024.

[3] Y. Huang, A. Thammatadatrakoon, W. Zheng, Y. Zhang, D. Du, and J. Lu, “GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction,” in CVPR, 2025.

[4] L. Zhao, S. Wei, J. Hays, and L. Gan, “Gaussianformer3d: Multimodal gaussian-based semantic occupancy prediction with 3d deformable attention,” arXiv preprint arXiv:2505.10685, 2025.

[5] S. Zuo, W. Zheng, Y. Huang, J. Zhou, and J. Lu, “GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction,” in CVPR, 2025.

[6] C. Chen, H. Huang, and S. Bagchi, “Vision-only gaussian splatting for collaborative semantic occupancy prediction,” in AAAI, 2026.

[7] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, R. Howes, P.-Y. Huang, H. Xu, V. Sharma, S.-W. Li, W. Galuba, M. Rabbat, M. Assran, N. Ballas, G. Synnaeve, I. Misra, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” in TMLR, 2024.

[8] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in ICML, 2021.

[9] L. Barsellotti, L. Bianchi, N. Messina, F. Carrara, M. Cornia, L. Baraldi, F. Falchi, and R. Cucchiara, “Talking to dino: Bridging self-supervised vision backbones with language for open-vocabulary segmentation,” in ICCV, 2025.

[10] C. Jose, T. Moutakanni, D. Kang, F. Baldassarre, T. Darcet, H. Xu, D. Li, M. Szafraniec, M. Ramamonjisoa, M. Oquab, et al., “Dinov2 meets text: A unified framework for image-and pixel-level visionlanguage alignment,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 24 905– 24 916.

[11] H. Jiang, L. Liu, T. Cheng, X. Wang, T. Lin, Z. Su, W. Liu, and X. Wang, “Gausstr: Foundation model-aligned gaussian transformer for self-supervised 3d spatial understanding,” in CVPR, 2025.

[12] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[13] R. Xu, H. Xiang, X. Xia, X. Han, J. Li, and J. Ma, “Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-tovehicle communication,” in ICRA, 2022.

[14] R. Song, C. Liang, H. Cao, Z. Yan, W. Zimmer, M. Gross, A. Festag, and A. Knoll, “Collaborative semantic occupancy prediction with hybrid feature fusion in connected automated vehicles,” in CVPR, 2024.

[15] A. Vaswani, “Attention is all you need,” in NeurIPS, 2017.

[16] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable DETR: Deformable Transformers for End-to-End Object Detection,” in ICLR, 2021.

[17] V. Ye, R. Li, J. Kerr, M. Turkulainen, B. Yi, Z. Pan, O. Seiskari, J. Ye, J. Hu, M. Tancik, and A. Kanazawa, “gsplat: An open-source library for gaussian splatting,” Journal of Machine Learning Research, 2025.

[18] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in CVPR, 2020.