# Diffusion Trajectory Modeling for Semantic Correspondence

Yusung Choi

cyscyb@gmail.com

Department of Computer Engineering Pukyong National University Busan, Republic of Korea

## Abstract

Diffusion models generate images through an iterative diffusion process, and recent studies have demonstrated that the intermediate feature maps produced during this process contain rich visual representations, leading to their adoption across a variety of downstream tasks. However, most existing approaches are limited to either using a single feature map at a specific timestep or aggregating feature maps across multiple timesteps. We observe that intermediate representations in the diffusion process form meaningful trajectories along the time axis. In particular, the representation of each spatial patch evolves progressively throughout the generative process, encoding semantics that are difficult to capture from static snapshots alone. This observation motivates the need to treat diffusion representations as temporally structured trajectories rather than static snapshots. To this end, we propose Diffusion Trajectory Modeling (DTM), a framework that interprets the temporal evolution of each spatial patch as a trajectory and leverages it for semantic correspondence. By effectively modeling patch-wise trajectories generated across multiple timesteps, DTM captures correspondence cues that prior methods are not designed to capture. We further demonstrate empirically that spatially corresponding patches form similar trajectory patterns throughout the diffusion process, suggesting that the temporal axis of diffusion carries semantic information. Experiments on SPair-71k, SPair-U and AP-10K show that DTM achieves strong performance, presenting a new perspective for exploiting diffusion representations from a trajectory-centric viewpoint.

## Introduction

<sub>v</sub>The fundamental goal of computer vision is to extract task-relevant information from com-<sup>i</sup>plex visual inputs and organize it into generalizable representations. From this perspective, X<sub>many vision problems ultimately reduce to the challenge of learning robust and discrim-</sub> inative representations. Early computer vision research pursued local invariance and discriminability through handcrafted descriptors such as SIFT [16] and HOG [5], while subsequent CNN [11] and Vision Transformer-based [1, 6] methods enabled richer and more transferable visual representations through large-scale learning. More recently, generative models [12, 24, 27] — and diffusion models in particular — have emerged as a promising new source of visual representations, as they learn internal representations that capture semantic and structural information across multiple levels of abstraction. However, existing approaches to leveraging diffusion representations have largely been limited to selecting features at specific timesteps and layers [8, 27], or aggregating features across multiple stages [18, 29], leaving the temporally evolving nature of the diffusion process largely unexploited.

![](images/2ca07682ee5ecfe779e8d42a78c1dcea620ac885f7bacabe994a2703e17648cd.jpg)  
Figure 1: Visualization of diffusion trajectories across image pair $I _ { A }$ and $I _ { B }$ . Each color represents a corresponding patch pair, where circle (•) and star (⋆) denote features at timestep = 20 and timestep = 620, respectively.

In this work, we take the perspective that diffusion representations should be viewed not as static features at individual timesteps, but as trajectories along which spatial patches evolve over time. During the diffusion process, the representation of each spatial patch changes progressively along the time axis, forming not a mere collection of intermediate states, but a temporally ordered evolution of representations. Consequently, existing approaches [8, 18, 27, 29] that consume diffusion representations as snapshots at a single timestep are fundamentally limited in that they discard this temporal structure. To validate this perspective, we extract diffusion trajectories for multiple semantically corresponding patch pairs and visualize them via PCA (Figure 1), where each color represents one corresponding pair. Patches that are semantically corresponding form strikingly similar trajectories throughout the diffusion process, even across different images, whereas patches of different colors exhibit clearly divergent trajectories. This suggests that the temporal evolution of representations — beyond feature similarity at any single timestep — can itself reflect semantic consistency, motivating the use of trajectories as units of representation. To fully exploit this temporal structure, we propose Diffusion Trajectory Modeling (DTM), a framework that directly models patch-wise trajectories for semantic correspondence. DTM encodes the trajectory of each spatial patch as a sequential signal, incorporating temporal variation patterns into the representation that would be difficult to capture from static snapshots alone. Through this design, DTM effectively captures correspondence cues that prior methods [8, 18, 27, 29] are not explicitly designed to capture. We evaluate DTM on three semantic correspondence benchmarks: SPair-71k [20], SPair-U [19] and AP-10K [28]. Results show that trajectory information improves correspondence performance over existing representations.

• We introduce a trajectory-centric perspective that interprets diffusion representations as temporally structured trajectories and propose DTM, a framework that leverages this view for semantic correspondence.

• We provide empirical evidence that semantically corresponding spatial patches form similar trajectories throughout the diffusion process, suggesting that trajectories serve as structural units encoding semantic information.

• We demonstrate through experiments on SPair-71k [20], SPair-U [19] and AP-10K [28] that DTM improves correspondence performance over existing baselines.

## 2 Related Work

Diffusion Trajectory. Diffusion models can be understood as trajectory-based generative processes [12, 26] that progressively transition from a noisy state to a data state. Existing works have formalized the diffusion process from the perspectives of stochastic differential equations (SDEs [26]), probability flow ODEs [26], and reverse dynamics [25], interpreting the generative process as a continuous path in state space. More recent studies have analyzed the geometric structure and sampling dynamics of diffusion trajectories, or leveraged them to improve sampling efficiency [17, 31]. However, these works primarily focus on understanding and controlling the generative process, with analyses largely focused on sample-level or latent-level dynamics. In contrast, how spatial patches evolve across timesteps throughout the diffusion process — and how such patch-level trajectories can be exploited as structured representations for downstream vision tasks — remains largely unexplored. Our work is distinguished from prior work in that we approach diffusion trajectories from the perspective of patch-level representation dynamics rather than generative dynamics.

Representation for Semantic Correspondence. The quality of representations is a critical factor determining correspondence performance in semantic matching. Self-supervised representations such as DINO [2, 23] have served as strong baselines, and it has since been shown [27] that the internal representations of diffusion models can also serve as effective descriptors for semantic correspondence. DIFT [27] demonstrated that diffusion features alone can achieve strong correspondence performance, while SD+DINO [29] further improved zero-shot semantic correspondence by combining the complementary strengths of Stable Diffusion [24] and DINO [23]. Diffusion Hyperfeatures [18] consolidated multi-scale and multi-timestep feature maps into per-pixel descriptors via a learnable aggregation network. More recently, DiTF [8] showed that semantically discriminative features can be effectively extracted from Diffusion Transformers as well. Nevertheless, these works [8, 18, 27, 29] primarily focus on selecting features at specific timesteps or combining multiple representations, and attempts to directly incorporate the temporal structure inherent to the diffusion process into correspondence representations remain limited.

## 3 Diffusion Trajectory

We first define the notion of diffusion trajectory as used in this work, and provide an analysis suggesting that semantically corresponding spatial patches exhibit similar trajectory patterns throughout the diffusion process. Building on this observation, we describe DTM, our framework for leveraging patch-wise trajectories for semantic correspondence.

## 3.1 Diffusion Trajectory Definition

Diffusion models encode an input image into a latent space and process it iteratively across multiple timesteps, producing rich internal feature maps at each block. We adopt a pretrained Diffusion Transformer as our backbone and extract features following the DiTF framework [8]. Specifically, the noisy latent obtained at timestep t is passed through a sequence of DiT blocks, yielding an intermediate feature $z _ { t } ^ { k }$ at the k-th block. To address the massive activations problem in DiT features — a small number of feature dimensions with disproportionately large activation values that degrade correspondence performance based on cosine similarity — DiTF applies AdaLN-based channel-wise modulation to this pre-AdaLN feature:

$$
\hat { z } _ { t } ^ { k } = ( 1 + \gamma _ { k } ) \mathrm { L a y e r N o r m } ( z _ { t } ^ { k } ) + \beta _ { k }\tag{1}
$$

where $\gamma _ { k } , \beta _ { k }$ are channel-wise scale and shift parameters regressed by an MLP conditioned on the timestep t and text embedding c. A channel discard strategy is further applied to eliminate residual weak activations. Through this process, we obtain a semantically discriminative feature representation $\hat { z } _ { t } ^ { k }$ for each spatial patch at a given timestep t and block k.

The feature map $\hat { z } _ { t } ^ { k } \in \mathbf { R } ^ { C \times H \times W }$ obtained above represents the full spatial feature map of the input image at timestep t and block k, serving as the fundamental building block for constructing diffusion trajectories. Hereafter, we fix the block index k and write $\hat { \boldsymbol { z } } _ { t }$ for brevity.

Given an input image, we extract feature maps at N selected timesteps $\mathcal { T } = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } \}$ yielding $\hat { z } _ { t _ { i } } \in \dot { \mathbf { R } } ^ { C \times H \times \widecheck { W } }$ at each timestep $t _ { i } .$ To decompose this feature map into patchlevel representations, we index spatial positions over the $H \times W$ grid, where each position $p \in \{ 1 , \ldots , H \times W \}$ corresponds to a distinct spatial patch. The feature vector of the patch at position $p$ is extracted as:

$$
\hat { z } _ { t _ { i } } ^ { p } = \hat { z } _ { t _ { i } } [ p ] \in \mathbf { R } ^ { C }\tag{2}
$$

where $\hat { z } _ { t _ { i } } [ p ]$ denotes the channel vector at spatial position $p$ of the feature map $\hat { z } _ { t _ { i } }$ . Arranging the patch features across all timesteps $t _ { i } \in \mathcal { T }$ in temporal order, we define the diffusion trajectory of patch $p$ as:

$$
\tau ^ { p } = \left[ \widehat { z } _ { t _ { 1 } } ^ { p } , \widehat { z } _ { t _ { 2 } } ^ { p } , \ldots , \widehat { z } _ { t _ { N } } ^ { p } \right] \in \mathbf { R } ^ { N \times C }\tag{3}
$$

Rather than treating each timestep feature independently, we view $\tau ^ { p }$ as a temporally ordered sequence, regarding the evolution of patch representations across timesteps as a structured signal. In its raw form, $\tau ^ { p }$ is a stacked sequence of per-timestep features arranged in temporal order; nevertheless, we argue that this temporal ordering itself carries meaningful information. We first analyze in Section 3.2 whether this temporal ordering reflects semantic information. Building on this analysis, we show that explicitly modeling $\tau ^ { p }$ enables the capture of semantic dynamics that are difficult to recover from single-timestep features alone, and describe the specific modeling approach in Section 3.3.

## 3.2 Trajectory Similarity Analysis

To analyze whether the trajectory $\tau ^ { p }$ defined in Section 3.1 genuinely reflects semantic correspondence, we conduct a trajectory analysis on patch pairs extracted from images $I _ { a }$ and $I _ { b }$ (Figure 2).

Specifically, leveraging keypoint annotations from SPair-71k [20], we select M semantically corresponding patch pairs $\{ ( p _ { a } ^ { m } , p _ { b } ^ { m } ) \} _ { m = 1 } ^ { M }$ across two images and extract their respective trajectories $\tau ^ { p _ { a } ^ { m } } , \tau ^ { p _ { b } ^ { m } } \in \mathbf { R } ^ { N \times C }$ . The total 2M trajectories extracted from both images are then jointly projected onto a shared low-dimensional space via PCA. Applying PCA jointly ensures that trajectories from both images are compared within the same basis space, enabling a meaningful geometric comparison. In the visualization, the two trajectories belonging to the same corresponding pair $( p _ { a } ^ { m } , p _ { b } ^ { m } )$ are rendered in the same color, while different corresponding pairs are assigned distinct colors.

![](images/f2abd9be7d36e8ddbf06d9f5ca094fc6b89d96a21435dc05a65645681827278f.jpg)  
Figure 2: Diffusion trajectory visualization of semantically corresponding patch pairs.

Figure 2 presents the visualization results across multiple image pairs. Notably, trajectories of corresponding patch pairs, shown in matching colors, form remarkably similar patterns in the shared PCA space, despite being extracted from different images. In contrast, non-corresponding patch pairs exhibit clearly distinct and separable patterns. Furthermore, the PCA visualization suggests that temporal variation across the entire trajectory provides correspondence cues beyond individual timestep features. Semantically corresponding patches are similar not only at individual timesteps, but also in how their representations evolve throughout the diffusion process, indicating that temporal evolution itself may reflect semantic structure.

These observations support two important conclusions. First, diffusion trajectories can function as signals that reflect semantic correspondence across images. Second, this temporal structure is difficult to capture through existing approaches [8, 18, 27, 29] that select features at specific timesteps or aggregate them across timesteps, highlighting the need for an approach that models the entire trajectory as a sequential signal. Building on these findings, Section 3.3 describes our specific method for explicitly modeling $\tau ^ { p }$ and leveraging it for semantic correspondence.

## 3.3 Diffusion Trajectory Modeling

Given the trajectory $\tau ^ { p }$ extracted from the diffusion process, the most straightforward approach would be to directly use the raw concatenation of per-timestep features. However, this is problematic for two reasons: the dimensionality of the representation becomes excessively large as the trajectory spans multiple timesteps, and the naive concatenation structure fails to capture the sequential and structural characteristics inherent to the trajectory. To address this, we propose Diffusion Trajectory Modeling (DTM), a framework that treats the trajectory as a sequence and transforms it into a representation suitable for semantic

![](images/7074dd1f9d860ba165fca8b3c5e713096ccac897fd8303ce09502ed5251b2fba.jpg)  
Figure 3: Overview of the proposed Diffusion Trajectory Modeling (DTM) framework.

correspondence.

DTM treats each patch trajectory as a sequential signal and builds upon State Space Models (SSMs) [9, 10], which are well-suited for modeling long-range dependencies and sequential structure. In particular, we leverage Mamba [9], which exploits a selective state space mechanism to efficiently incorporate information across the entire sequence.

Specifically, the trajectory $\tau ^ { p } \in \mathbf { R } ^ { N \times C }$ of each spatial patch $p$ is processed independently by a unidirectional Mamba encoder $\mathcal { M } _ { 1 }$ , which operates along the timestep dimension with sequence length N and C-dimensional patch features as tokens. The outputs of $\mathcal { M } _ { 1 }$ across all spatial patches are then collectively reshaped into a spatiotemporal feature volume of size $N \times H \times W \times C$ . A shared spatial block consisting of residual convolutional layers is then applied to each timestep-wise feature map independently, incorporating local spatial context among neighboring patches while preserving the temporal dimension. The spatially enriched features are subsequently redistributed back into per-patch sequences and passed through a second unidirectional Mamba encoder $\mathcal { M } _ { 2 }$ , which further models the temporal structure with the benefit of spatial context. As a unidirectional sequential model, Mamba accumulates information across the entire input sequence into its final hidden state; we therefore take the last hidden state of $\mathcal { M } _ { 2 }$ as the trajectory-aware representation of the patch:

$$
h ^ { p } = \mathcal { M } _ { 2 } ( \mathrm { S p a t i a l B l o c k } ( \mathcal { M } _ { 1 } ( \tau ^ { p } ) ) ) \in \mathbf { R } ^ { d }\tag{4}
$$

This interleaving of temporal encoding and spatial context integration goes beyond simply applying a sequence model, jointly capturing the temporal evolution of each patch and its complementary spatial relationships with neighboring patches. Here, the last hidden state $h ^ { p }$ is a representation that implicitly encodes the temporal evolution of patch $p$ throughout the diffusion process, capturing correspondence cues that are difficult to recover from single-timestep features alone. Given the representations $h ^ { p _ { a } }$ and $h ^ { p _ { b } }$ extracted for patches in images $I _ { a }$ and $I _ { b } ,$ , correspondence is estimated based on cosine similarity:

$$
\mathrm { s i m } ( p _ { a } , p _ { b } ) = { \frac { h ^ { p _ { a } } \cdot h ^ { p _ { b } } } { \| h ^ { p _ { a } } \| \| h ^ { p _ { b } } \| } }\tag{5}
$$

The Mamba encoder is trained using keypoint annotations as supervision. A bidirectional InfoNCE loss [22] is employed to encourage representations of corresponding patch pairs to

be similar while pushing non-corresponding pairs apart:

$$
{ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { I n f o N C E } } ( h ^ { p _ { a } } \to h ^ { p _ { b } } ) + { \mathcal { L } } _ { \mathrm { I n f o N C E } } ( h ^ { p _ { b } } \to h ^ { p _ { a } } )\tag{6}
$$

The effectiveness of the proposed DTM is validated through the experiments.

## 4 Experiments

## 4.1 Experimental Details

Trajectories are extracted from the pre-trained FLUX.1-dev [13] and fed into DTM. For SPair-71k [20] and SPair-U [19], we evaluate both benchmarks using the same checkpoint trained on the training split of SPair-71k. For AP-10K [28], we train a separate checkpoint on its training split and evaluate under the Intra-Species, Cross-Species, and Cross-Family settings. For all experiments, we use the AdamW optimizer [15] with a cosine learning rate scheduler, decaying the learning rate from $1 \times 1 0 ^ { - 4 }$ to $1 \times 1 0 ^ { - 5 }$ over a total of 4,000 training steps. Input images are processed at a resolution of $9 6 0 \times 9 6 0$ , and all experiments are conducted on a single NVIDIA A100 SXM GPU.

For trajectory construction, we sample timesteps from the full $T = 1 0 0 0$ diffusion schedule at intervals of 20 within the range $t \in \{ 2 0 , 4 0 , 6 0 , \ldots , 6 2 0 \}$ , yielding a total of $N = 3 1$ timesteps per trajectory, and fix the block index at $k = 2 8$ following DiTF [8]. Timesteps beyond $T = 6 2 0$ are excluded as the corresponding feature maps contain excessive noise that disrupts the semantic structure of the trajectory.

## 4.2 Datasets

SPair-71k [20] is a large-scale semantic correspondence benchmark comprising 70,958 image pairs across 18 object categories. It evaluates correspondence between diverse object instances within the same category, providing a challenging setting with substantial variations in viewpoint, scale, and appearance. Its large scale and categorical diversity enable a thorough assessment of the generalization ability of correspondence methods.

SPair-U [19] is an extension of SPair-71k that introduces novel keypoint annotations not seen during training, designed to expose the generalization gap in supervised semantic correspondence methods. While SPair-71k evaluates performance on standard annotated keypoints, SPair-U assesses whether methods can generalize beyond sparsely annotated training keypoints, providing a more rigorous evaluation of correspondence generalization ability.

AP-10K [28] is a benchmark comprising animal images captured in the wild, originally proposed for animal pose estimation. For semantic correspondence evaluation, the benchmark is decomposed into Intra-Species (IS), which evaluates correspondence within the same species as those used for training; Cross-Species (CS), which evaluates correspondence to unseen species within the same family; and Cross-Family (CF), which evaluates correspondence to unseen families. This decomposition enables the generalization ability of correspondence methods to be assessed as the domain gap progressively widens.

## 4.3 Metrics

We evaluate using Percentage of Correct Keypoints (PCK), the standard metric for semantic correspondence. A predicted keypoint is considered correct if it falls within a radius of

α · max(h,w) from the ground-truth keypoint, where h and w refer to the height and width of the object bounding box $( \alpha _ { \mathrm { { b b o x } } } )$
<table><tr><td rowspan="3">Method</td><td colspan="3">SPair-71k</td><td colspan="3">SPair-U</td><td colspan="3">AP-10K (IS)</td><td colspan="3">AP-10K (CS)</td><td colspan="3">AP-10K (CF)</td></tr><tr><td colspan="3">α: bbox</td><td colspan="3">α: bbox</td><td colspan="3">α: bbox</td><td colspan="3">α: bbox</td><td colspan="3">α: bbox</td></tr><tr><td>0.05</td><td>0.1</td><td>0.15</td><td>0.05</td><td>0.1</td><td>0.15</td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.01</td><td>0.05</td><td>0.10</td><td>0.01</td><td>0.05</td><td>0.10</td></tr><tr><td>Zero-shot</td><td>49.5</td><td>61.1</td><td>67.2</td><td>37.1</td><td>53.4</td><td>63.0</td><td>7.6</td><td>49.7</td><td>62.2</td><td>6.8</td><td>48.2</td><td>61.7</td><td>5.7</td><td>34.2</td><td>48.9</td></tr><tr><td>GAP</td><td>52.0</td><td>64.1</td><td>70.3</td><td>38.0</td><td>54.9</td><td>65.3</td><td>8.2</td><td>51.5</td><td>64.2</td><td>7.7</td><td>50.6</td><td>63.0</td><td>6.5</td><td>36.4</td><td>50.7</td></tr><tr><td>Per-pixel descriptors</td><td>65.2</td><td>75.1</td><td>79.3</td><td>36.0</td><td>54.6</td><td>64.6</td><td>17.0</td><td>65.3</td><td>81.4</td><td>15.2</td><td>64.1</td><td>79.7</td><td>12.8</td><td>56.8</td><td>71.5</td></tr><tr><td>DTM (Ours)</td><td>70.0</td><td>80.5</td><td>84.4</td><td>39.2</td><td>58.7</td><td>68.3</td><td>20.5</td><td>71.0</td><td>86.2</td><td>18.9</td><td>68.2</td><td>84.8</td><td>15.3</td><td>62.1</td><td>77.9</td></tr></table>

Table 1: Per-image PCK comparison on SPair-71k, SPair-U, and AP-10K. All methods share the same FLUX backbone and differ only in how diffusion features are utilized. We compare a single-timestep feature map baseline (Zero-shot), a multi-timestep Global Average Pooling baseline (GAP), a reimplementation of Diffusion Hyperfeatures [18] that integrates multitimestep feature maps via a learnable weighted aggregation (per-pixel descriptors), and the proposed DTM. Per-pixel feature descriptors and DTM are trained under identical settings and differ only in their architecture. AP-10K results are reported under Intra-Species (IS), Cross-Species (CS), and Cross-Family (CF) settings, spanning increasing domain gaps. The highest PCK is highlighted in bold, and the second highest is underlined.

<table><tr><td>Method</td><td>aero</td><td>bike</td><td>bird</td><td>boat</td><td>bottle</td><td>bus</td><td>car</td><td>cat</td><td>chair</td><td>cow dog</td><td>horse</td><td>motor</td><td></td><td>person</td><td>plant</td><td>sheep</td><td>train</td><td>tv All</td></tr><tr><td>DIFT []</td><td>63.5</td><td>54.5</td><td>80.8</td><td>34.5</td><td>46.2</td><td>52.7</td><td>48.3</td><td>77.7 39.0</td><td>76.0</td><td>54.9</td><td>61.3</td><td>53.3</td><td>46.0</td><td>57.8</td><td>57.1</td><td>71.1</td><td>63.4</td><td>57.7</td></tr><tr><td>DINOv2 []</td><td>72.7</td><td>62.0</td><td>85.2</td><td>41.3</td><td>40.4</td><td>52.3 51.5</td><td>71.1</td><td>36.2</td><td>67.1</td><td>64.6</td><td>67.6</td><td>61.0</td><td>68.2</td><td>30.7</td><td>62.0</td><td>54.3</td><td>24.2</td><td>55.6</td></tr><tr><td>SD+DINO []</td><td>73.0</td><td>64.1</td><td>86.4</td><td>40.7</td><td>52.9</td><td>55.0 53.8</td><td>78.6</td><td>45.5</td><td>77.3</td><td>64.7</td><td>69.7</td><td>63.3</td><td>69.2</td><td>58.4</td><td>67.6</td><td>66.2</td><td>53.5</td><td>64.0</td></tr><tr><td>DiTF (FLUX) []</td><td>74.3</td><td>65.0</td><td>88.1</td><td>48.1</td><td>53.2</td><td>60.7 60.7</td><td>84.9</td><td>42.4</td><td>82.8</td><td>68.4</td><td>72.1</td><td>70.9</td><td>74.2</td><td>62.1</td><td>72.6</td><td>66.0</td><td>60.3</td><td>67.1</td></tr><tr><td>CATs []</td><td>52.0</td><td>34.7</td><td>72.2</td><td>34.3</td><td>49.9</td><td>57.5 43.6</td><td>66.5</td><td>24.4</td><td>63.2</td><td>56.5</td><td>52.0</td><td>42.6</td><td>41.7</td><td>43.0</td><td>33.6</td><td>72.6</td><td>58.0</td><td>49.9</td></tr><tr><td>CATs++ []</td><td>60.6</td><td>46.9</td><td>82.5</td><td>41.6</td><td>56.8</td><td>64.9 50.4</td><td>72.8</td><td>29.2</td><td>75.8</td><td>65.4</td><td>62.5</td><td>50.9</td><td>56.1</td><td>54.8</td><td>48.2</td><td>80.9</td><td>74.9</td><td>59.9</td></tr><tr><td>DHF []</td><td>74.0</td><td>61.0</td><td>87.2</td><td>40.7</td><td>47.8</td><td>70.0 74.4</td><td>80.9</td><td>38.5</td><td>76.1</td><td>60.9</td><td>66.8</td><td>66.6</td><td>70.3</td><td>58.0</td><td>54.3</td><td>87.4</td><td>60.3</td><td>64.9</td></tr><tr><td>SD+DINO (S) [四]</td><td>81.2</td><td>66.9</td><td>91.6</td><td>61.4</td><td>57.4</td><td>85.3 83.1</td><td>90.8</td><td>54.5</td><td>88.5</td><td>75.1</td><td>80.2</td><td>71.9</td><td>77.9</td><td>60.7</td><td>68.9</td><td>92.4</td><td>65.8</td><td>74.6</td></tr><tr><td>SD4Match [ 日</td><td>75.3</td><td>67.4</td><td>85.7</td><td>64.7</td><td>62.9</td><td>86.6 76.5</td><td>82.6</td><td>64.8</td><td>86.7</td><td>73.0</td><td>78.9</td><td>70.9</td><td>78.3</td><td>66.8</td><td>64.8</td><td>91.5</td><td>86.6</td><td>75.5</td></tr><tr><td>DTM (Ours)</td><td>85.0</td><td>72.3</td><td>92.2</td><td>71.3</td><td>63.2</td><td>88.6 78.9</td><td>94.9</td><td>66.7</td><td>90.2</td><td>77.1</td><td>81.6</td><td>80.7</td><td>84.9</td><td>70.8</td><td>73.0</td><td>94.9</td><td>89.7</td><td>80.9</td></tr></table>

Table 2: Evaluation on SPair-71k. Per-category PCK results at $\alpha _ { \mathrm { b b o x } } = 0 . 1$ on SPair-71k. Due to inconsistencies in evaluation protocols across prior works, we report per-keypoint PCK for zero-shot methods and per-image PCK for supervised methods separately [30].

Table 1 controls for a fixed backbone and training setup while varying only how diffusion features are utilized, whereas Table 2 provides a comparison with prior methods.

As shown in Table 1, the proposed DTM achieves the highest performance on SPair-71k and SPair-U, recording 80.5 PCK@0.1 and 58.7 PCK@0.1, respectively, outperforming the single-timestep baseline (Zero-shot), the simple aggregation baseline (GAP), and the per-pixel descriptor approach. Zero-shot does not leverage any temporal structure of the diffusion process, as it relies solely on a single-timestep feature. GAP aggregates features across multiple timesteps, which provides richer information than a single timestep; however, averaging across timesteps suppresses timestep-specific information, limiting its ability to capture the sequential structure of the trajectory. Per-pixel descriptors have the advantage of adaptively reflecting the importance of each feature map via a learnable weighted aggregation; however, it treats all feature maps independently and thus disregards the sequential dependencies between timesteps. In particular, DTM consistently outperforms per-pixel descriptor across all benchmarks under identical training settings, suggesting that explicitly modeling the sequential structure of trajectories enables richer representation learning than simple weighted aggregation.

This tendency becomes even more pronounced on SPair-U, which evaluates generalization to unseen keypoints not observed during training, where the performance ranking follows DTM > GAP > Per-pixel > Zero-shot. This suggests that Per-pixel descriptor, which learns a weighted aggregation optimized for training keypoints, may overfit to the seen keypoints and struggles to generalize to unseen ones. In contrast, DTM, which models the temporal structure of trajectories, appears to learn representations that are less dependent on specific keypoints, exhibiting stronger generalization ability. A similar trend is observed on AP-10K, where DTM maintains consistently superior performance over the baselines across the IS, CS, and CF settings as the domain gap widens, demonstrating that trajectory-based representations are robust to cross-domain generalization.

Table 2 presents per-category PCK results on SPair-71k. Among supervised methods, DTM achieves the highest overall mean PCK of 80.9, surpassing CATs [3] (49.9), CATs++ [4] (59.9), DHF [18] (64.9), SD+DINO [29] (S) (74.6), and SD4Match [14] (75.5). Examining per-category results, DTM achieves notably high performance on categories with consistent global structure and distinctive semantic layout, such as cat (94.9), bird (92.2), and train (94.9). These categories tend to form stable and consistent trajectory patterns throughout the diffusion process, allowing DTM to effectively capture correspondence cues. In contrast, relatively lower performance is observed on categories such as bottle (63.2) and chair (66.7); we speculate that large intra-class appearance and structural variations make it difficult to form consistent trajectory patterns across instances.

## 4.4 Ablation

Backbone Ablation. To investigate whether the effectiveness of leveraging diffusion trajectories is specific to a particular backbone or reflects a general property of diffusion models, we conduct experiments using Stable Diffusion 3.0 [7] (SD3.0), Stable Diffusion 3.5 [7] (SD3.5), and Stable Diffusion 1.5 [24] (SD1.5) as alternative backbones in addition to FLUX.1- dev. Among these, SD1.5 is based on a UNet architecture, while the others belong to the DiT family. All experiments are conducted under identical training settings and evaluated on SPair-71k. As shown in Table 3, DTM remains effective across all backbones despite these architectural differences. This suggests that the phenomenon whereby semantically corresponding patches form similar trajectories is not specific to a particular diffusion model, but rather reflects a structural property that generalizes across diffusion models. In other words, these results support the view that leveraging diffusion trajectories as representations constitutes a backbone-agnostic approach that does not rely on the characteristics of any specific model.

<table><tr><td>Backbone</td><td>PCK  $( \alpha _ { \mathrm { b b o x } } = 0 . 1 )$ </td></tr><tr><td>SD1.5</td><td>62.3</td></tr><tr><td>SD3.0</td><td>73.1</td></tr><tr><td>SD3.5</td><td>77.6</td></tr><tr><td>FLUX.1-dev</td><td>80.5</td></tr></table>

Table 3: Backbone ablation on SPair-71k $( \alpha _ { \mathrm { b b o x } } = 0 . 1 )$ .

Temporal order shuffle. In Section 3.2, we observed that semantically corresponding patches form similar trajectory patterns throughout the diffusion process. To examine whether these patterns depend on the semantic chronology of the diffusion process — that is, the temporally structured progression of noise — we conduct two temporal order perturbation experiments.

The first applies the same random permutation to all images (Same Shuffle), preserving the alignment of temporal axes across images while disrupting the original semantic chronology of the diffusion process. The second applies a different random permutation independently to each image (Different Shuffle), such that even the alignment of temporal axes across images is no longer guaranteed. For both shuffle variants, the permutation is applied consistently during both training and evaluation.

As shown in Table 4, the improvement from Different Shuffle to Same Shuffle (+2.1) reflects the contribution of temporal alignment, while that from Same Shuffle to the original order (+4.3) reflects the contribution of semantic chronology. Prior theoretical work [21] suggests that training a permutation-sensitive model with random permutations approximates a permutation-invariant function; indeed, Different Shuffle (74.1), which contains no order information at all, has the same capacity as the original DTM but converges to a level similar to that of per-pixel descriptor (75.1, Table 1), a simpler learned aggregation architecture. This suggests that a performance ceiling exists for order-agnostic approaches regardless of capacity, and that the improvement of the original-order DTM beyond this ceiling stems from exploiting order information in training, rather than from capacity.

Taken together, these results empirically demonstrate that multi-timestep coverage, temporal alignment, and semantic chronology each contribute independently to the performance of diffusion trajectories, with the contribution of chronology (+4.3) being more than twice as large as that of alignment (+2.1). This supports the view that the temporal ordering of diffusion trajectories serves as a meaningful signal reflecting semantic correspondence, beyond simply being a sequential structure.

<table><tr><td>Method</td><td>PCK  $( \alpha _ { \mathrm { { b b o x } } } = 0 . 1 )$ </td></tr><tr><td>DTM (Different Shuffle)</td><td>74.1</td></tr><tr><td>DTM (Same Shuffle)</td><td>76.2</td></tr><tr><td>DTM (Original Order)</td><td>80.5</td></tr></table>

Table 4: Temporal order shuffle ablation on SPair-71k $( \alpha _ { \mathrm { b b o x } } = 0 . 1 )$ .

## 4.5 Qualitative Comparison

Figure 4 presents qualitative correspondence results on SPair-71k [20], comparing DiTF [8], SD+DINO [29], and DTM. Green and red dots indicate correct and incorrect correspondences, respectively, under the PCK criterion with $\alpha _ { \mathrm { b b o x } } = 0 . 1$ . The methods differ in positional precision and in the number of correct matches. These differences are especially noticeable for small and distinctive parts, such as beaks and noses.

## 5 Limitations and Future Work

This work has several limitations. First, DTM exhibits limitations for objects with bilateral symmetry. For such objects, semantically corresponding parts and their mirror counterparts share highly similar visual features, leading to similar trajectory patterns that make it difficult to distinguish correct correspondences from their symmetric counterparts (Figure 5).

![](images/7bea7f3dea80a7b155e91e8e71fa1896e1fc38d780960a7d74a9698fde7e7e8f.jpg)  
Figure 4: Qualitative correspondence results on SPair-71k.

Incorporating additional representations or learning strategies to resolve such symmetric ambiguity remains an open direction for future work. Second, while we empirically observe through PCA visualization that diffusion trajectories suggest semantic information, a theoretical explanation of how such temporal patterns give rise to meaningful representations remains lacking. Establishing a theoretical foundation for trajectory-based representations is an important direction for future research. Third, the process of extracting features across multiple timesteps and encoding them with Mamba incurs additional computational overhead. Exploring efficient trajectory encoding techniques or model compression methods remains an open problem for future work.

![](images/2665068cd8a048aed47d9571876656a4183863d3f38d7ffb9a005963618a875a.jpg)  
Figure 5: Trajectory visualization for a cat image pair. Patches corresponding to symmetric parts (e.g., left/right eyes, left/right ears) exhibit similar trajectory patterns, illustrating the symmetric ambiguity challenge discussed in Section 5.

## 6 Conclusion

In this paper, we presented a novel perspective that treats diffusion representations as temporally structured trajectories, and proposed Diffusion Trajectory Modeling (DTM), a framework that leverages this view for semantic correspondence. Through our analysis, we showed that semantically corresponding spatial patches form similar trajectory patterns throughout the diffusion process, and demonstrated that the proposed method consistently outperforms single-timestep and simple aggregation-based representations. This work shows that the temporal structure inherent to the diffusion process can be directly exploited, presenting a new perspective for interpreting diffusion representations from a trajectory-centric viewpoint.

## References

[1] Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. Deep vit features as dense visual descriptors. arXiv preprint arXiv:2112.05814, 2(3):4, 2021.

[2] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[3] Seokju Cho, Sunghwan Hong, Sangryul Jeon, Yunsung Lee, Kwanghoon Sohn, and Seungryong Kim. Cats: Cost aggregation transformers for visual correspondence. Advances in Neural Information Processing Systems, 34:9011–9023, 2021.

[4] Seokju Cho, Sunghwan Hong, and Seungryong Kim. Cats++: Boosting cost aggregation with convolutions and transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):7174–7194, 2022.

[5] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), volume 1, pages 886–893. Ieee, 2005.

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

[8] Chaofan Gan, Yuanpeng Tu, Xi Chen, Tieyuan Chen, Yuxi Li, Mehrtash Harandi, and Weiyao Lin. Unleashing diffusion transformers for visual correspondence by modulating massive activations. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= s3MwCBuqav.

[9] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[10] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[13] Black Forest Labs. Flux. https://github.com/black-forest-labs/ flux, 2024.

[14] Xinghui Li, Jingyi Lu, Kai Han, and Victor Adrian Prisacariu. Sd4match: Learning to prompt stable diffusion model for semantic matching. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27558–27568, 2024.

[15] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[16] David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004.

[17] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpmsolver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.

[18] Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. Advances in Neural Information Processing Systems, 36:47500–47510, 2023.

[19] Octave Mariotti, Zhipeng Du, Yash Bhalgat, Oisin Mac Aodha, and Hakan Bilen. Jamais vu: Exposing the generalization gap in supervised semantic correspondence. arXiv preprint arXiv:2506.08220, 2025.

[20] Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. Spair-71k: A large-scale benchmark for semantic correspondence. arXiv preprint arXiv:1908.10543, 2019.

[21] Ryan L Murphy, Balasubramaniam Srinivasan, Vinayak Rao, and Bruno Ribeiro. Janossy pooling: Learning deep permutation-invariant functions for variable-size inputs. arXiv preprint arXiv:1811.01900, 2018.

[22] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[23] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684– 10695, 2022.

[25] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[26] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

[27] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. Advances in neural information processing systems, 36:1363–1389, 2023.

[28] Hang Yu, Yufei Xu, Jing Zhang, Wei Zhao, Ziyu Guan, and Dacheng Tao. Ap-10k: A benchmark for animal pose estimation in the wild. arXiv preprint arXiv:2108.12617, 2021.

[29] Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547, 2023.

[30] Junyi Zhang, Charles Herrmann, Junhwa Hur, Eric Chen, Varun Jampani, Deqing Sun, and Ming-Hsuan Yang. Telling left from right: Identifying geometry-aware semantic correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3076–3085, 2024.

[31] Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.