# Anatomical Prior-Guided Routing for 3D Lung Tumor Segmentation

Uihyeon Maeng<sup>1⋆</sup> , Jonghun Kim<sup>2⋆</sup> , and Hyunjin Park<sup>1,2†</sup>

<sup>1</sup> Department of Artificial Intelligence, Sungkyunkwan University, Suwon, Korea 2 Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon, Korea {uhmaeng,iproj2,hyunjinp}@skku.edu

Abstract. Accurate lung tumor segmentation in computed tomography (CT) is essential for treatment planning, response assessment, and quantitative imaging analysis. However, 3D lung tumor segmentation remains challenging because tumors occupy only a small portion of the whole volume and exhibit heterogeneous appearances across anatomical locations. In patch-based training and sliding-window inference for 3D medical image segmentation, location-dependent heterogeneity is further amplified because cropped image patches provide limited global anatomical context, making it dificult for a model to adapt its feature processing according to the anatomical region from which each patch is extracted. To address this limitation, we propose an Anatomical Prior-guided Routing Module (APRM) for lung tumor segmentation. For each image patch, we obtain anatomical prior information of regional occupancy and deterministically select an anatomical routing gate. The selected gate then performs sparse feature routing over a shared expert adapter pool, enabling anatomy-specific feature adaptation while sharing expert capacity across regions. We evaluate APRM on two datasets with various backbone networks and demonstrate consistent improvements over baseline models. Furthermore, feature visualizations show that APRM produces diferent feature responses depending on patch location, indicating that the proposed routing mechanism efectively adapts feature processing according to anatomical context.

Keywords: Computed Tomography · 3D Medical Image Segmentation · Anatomical Prior-guided Routing

## 1 Introduction

Lung cancer is one of the most common cancers worldwide [3] and is characterized by high incidence and mortality rates that demand early detection and precise management [33]. To diagnose and formulate treatment plans, computed tomography (CT) is widely used in clinical practice because it provides a non-invasive way to observe lung tumors and surrounding anatomical structures [9, 32]. Accurate lung tumor segmentation is essential for determining tumor extent, supporting treatment planning, evaluating response, and enabling quantitative imaging analysis [20,34,44]. However, tumor segmentation has traditionally relied on manual annotation by experienced clinicians. Manual annotation is inherently time-consuming, labor-intensive, and prone to high interobserver variability, which incurs substantial clinical costs. These limitations motivate deep learning-based automatic segmentation, which can provide eficient tumor annotation. Despite recent advances, lung tumor segmentation remains challenging [34,37]. Lung tumors often occupy only a small fraction of the entire CT volume. In addition, tumor boundaries can be obscured by surrounding anatomy such as the pleura, vessels, bronchi, and mediastinum, leading to location-dependent errors [20, 39].

![](images/389a23b9015dd9d0412479d97b56adeedc244aba128b4ab611f5d3ded7fe4298.jpg)  
(a) Subtype Distribution of Lung Tumors

![](images/5e475335d697baf7efe61e058e0bdb7b44bee924e373f9bd0809dce86f5c90d7.jpg)  
(b) Anatomical Distribution of Lung Tumors  
Fig. 1: Distribution of lung cancer subtypes and anatomical tumor locations within the NSCLC [1,6,45] dataset. Tumor location is defined as the anatomical region (e.g., right upper lobe) showing the maximum spatial overlap with the tumor. Outside indicates areas around the lungs that do not overlap with any anatomical masks.

Furthermore, lung tumors exhibit heterogeneity across histological subtypes and their imaging appearance can vary with both subtype and anatomical context [30, 50, 52]. As shown in Fig. 1, both lung cancer subtypes and tumor locations exhibit heterogeneous distributions. In particular, diferent subtypes appear with diferent frequencies and lung tumors are not uniformly distributed across anatomical regions. Prior lobe-level analysis has also shown that tumor occurrence and histologic composition difer across anatomical locations, suggesting that tumor subtype and anatomical location are related rather than independent factors [42]. Accordingly, Fig. 2 presents representative cases with diferent histological subtypes and lobar locations [1]. These examples show that tumorcentered patches can contain diverse local anatomical contexts, including lung parenchyma, pleural or chest-wall interfaces, and regions close to cardiomediastinal structures. Such subtype- and location-dependent heterogeneity may influence boundary conspicuity and apparent tumor shape, while introducing nearby non-tumor tissues that can be confused with the tumor. In addition, medical datasets are often limited in size because data collection and expert annotation are costly [21, 23, 25, 41], making it dificult for conventional segmentation methods to learn the full heterogeneity of lung tumors from image appearance alone. Therefore, incorporating anatomical priors is important: a model should adapt feature processing according to where a tumor patch is located, so that it can better interpret subtype- and location-dependent tumor appearance and distinguish tumors from surrounding anatomical structures [20, 39].

![](images/3750012517a436e25a87b3ff5f54ecb5ea25b722ef03d495240797953529ce75.jpg)  
Fig. 2: Illustration of lung tumor heterogeneity and anatomical context. Left: anatomical prior regions, including the five lung lobes and the central mediastinum. Right: diferent histological subtypes and lobar locations. The yellow bounding boxes show tumor regions and the red contours indicate tumor annotations.

Mixture-of-experts (MoE) methods perform dynamic routing according to input features, activating only a subset of experts for each input while increasing the overall model capacity [10, 51]. This conditional computation has been used to scale the number of parameters while keeping the number of activated parameters nearly fixed, leading to improved performance and eficiency [10]. Distribution-aware MoE (dMoE) interprets MoE from a control-theoretic perspective as a feedback control that flexibly adapts to heterogeneous medical image data distributions [43]. However, instead of relying only on feature-driven routing, prior knowledge can provide a meaningful control signal by changing the routing strategy according to the anatomical location.

We propose Anatomical Prior-guided Routing Module (APRM) for 3D lung tumor segmentation. We extract anatomical masks using TotalSegmentator [53] and route each image patch based on information from the corresponding anatomical mask. The selected anatomical gate then conditions a router that computes sparse routing weights over a shared expert pool. This design enables anatomical information to guide patch-wise feature adaptation in a stable and interpretable manner, while avoiding the need to allocate a separate expert set to each region. We validate APRM on two datasets with various backbone networks and observe consistent performance improvements when APRM is added to the baseline models. In addition, feature visualizations show that APRM produces diferent feature responses according to anatomical location, indicating that the proposed module efectively reflects anatomical context during segmentation.

The main contributions of this study are as follows:

1. We propose APRM, a plug-in anatomical prior-guided routing module that uses anatomical masks to guide patch-level feature adaptation through gateconditioned sparse routing.

2. We validate APRM on two datasets with various backbones, showing consistent performance improvements.

3. We demonstrate that diferent feature processing occurs according to the anatomical location, showing the efectiveness of the APRM through feature visualization.

## 2 Related Work

## 2.1 Medical Image Segmentation

Medical image segmentation aims to classify regions of interest, such as lesions, organs, and tumors, at the pixel or voxel level [19,22,24,26–29,49]. This task is important for diagnosis, treatment planning, surgical guidance, and quantitative analysis. U-Net [47] uses an encoder–decoder structure with skip connections, which is efective for voxel-wise prediction [5, 40, 47]. Many U-shaped variants have been proposed to improve this structure. ResUNet adds residual connections to convolution blocks for training deeper networks [7]. The nnU-Net instead uses a basic U-Net and automatically adapts preprocessing, training settings, and network configuration to each dataset [15]. Transformer-based methods have also been studied to capture wider context. UNETR uses a transformer encoder with a convolutional decoder [12], and SwinUNETR uses a hierarchical Swin Transformer [11]. Since full-resolution 3D volumes require large memory, these methods usually use patch-based training and sliding-window inference [15, 17]. Recent methods also improve U-Net with modern convolution blocks or largescale pre-training, such as MedNeXt [48] and STUNet [14]. Unlike these works, APRM does not introduce a new backbone. It is a plug-in routing module that can be added to existing segmentation models.

## 2.2 Anatomical Priors and Mixture-of-Experts

Anatomical priors provide information about body structures and have been used as spatial maps, shape constraints, auxiliary inputs, or post-processing steps in medical image segmentation [13, 31, 36, 54]. Automatic CT parsers such as TotalSegmentator can extract organ masks from CT images [53]. These masks can provide anatomical context for tasks such as lymph-node segmentation [8, 38]. MoE models use conditional computation. A router selects a small number of experts for each token or patch, so the model can increase its capacity without using all experts at once [10,46,51,55]. Recent medical imaging studies have used MoE for continual learning, multi-modal representation, SAM adaptation, and semi-supervised learning [4, 18, 35]. These methods usually use image features, modalities, or prompts for routing. However, feature-based routing methods may overlook anatomical location. In lung tumor segmentation, errors often depend on anatomical location because nearby structures such as the pleura, vessels, and mediastinum can obscure tumor boundaries. We use anatomical priors as the routing context. This allows the model to adapt patch-level features according to the location of each patch.

![](images/e523a920bcfd1bccdc948d88aa1717e86c48b726a8f51f7dbc5724828a2c3744.jpg)  
Fig. 3: Overview of proposed APRM. Anatomical masks are extracted using a pretrained TotalSegmentator [53]. An anatomical gate is assigned based on these anatomical masks. When the outside region is larger than $\tau _ { \mathrm { o u t } }$ , the patch is assigned to the outside gate. The selected gate computes top-K sparse weights using shared experts.

## 3 Method

We propose APRM, which adapts feature processing according to anatomical locations to reflect tumor heterogeneity. APRM is added to all scales of the backbone encoder and applies diferent feature routing strategies according to the anatomical location of each patch. Fig. 3 provides an overview of APRM. Let input image $I \in \mathbb { R } ^ { H \times W \times D }$ denote a CT volume, where H, W, and D are the spatial dimensions. Let $\omega \in \mathbb { R } ^ { h \times w \times d }$ denote a patch. The image is divided into patches. We define the patch set \Omega as

$$
\begin{array} { r } { \varOmega = \left\{ \omega \mid \omega \subset \left\{ 1 , \dots , H \right\} \times \left\{ 1 , \dots , W \right\} \times \left\{ 1 , \dots , D \right\} , \ \omega \in \mathbb { R } ^ { h \times w \times d } \right\} , } \end{array}\tag{1}
$$

where patch $\omega \in \varOmega$ denotes a patch region. The local image patch extracted from a patch region \omega is denoted by $I _ { \omega }$ . We use an anatomical label map A extracted using TotalSegmentator [53]. We compute the occupancy ratio of anatomical regions in $I _ { \omega }$ from A for routing gate selection. If the outside proportion exceeds a threshold $\tau _ { \mathrm { o u t } }$ , the patch is assigned to the outside gate. Otherwise, the anatomical gate is selected from among the five lung lobes and the central region. The selected gates independently route features to the shared experts. This design allows routing to vary according to anatomical context while enabling expert capacity to be shared across multiple anatomical regions.

## 3.1 Definition of Anatomical Prior

We predefine anatomical regions relevant to lung cancer and extract their masks using TotalSegmentator [53]. We use masks for the five lung lobes: right upper lobe (RUL), right middle lobe (RML), right lower lobe (RLL), left upper lobe (LUL), and left lower lobe (LLL). We also define a central region by merging masks of selected mediastinal structures, including the heart, aorta, trachea, esophagus, pulmonary vein, superior vena cava, and inferior vena cava. The anatomical regions, therefore, consist of the five lung lobes and the central region. Areas that do not belong to these six predefined regions are defined as outside, resulting in seven classes:

$$
\mathcal { C } = \left\{ c _ { \mathrm { o u t s i d e } } , c _ { \mathrm { r u l } } , c _ { \mathrm { r m l } } , c _ { \mathrm { r l l } } , c _ { \mathrm { l u l } } , c _ { \mathrm { l l l } } , c _ { \mathrm { c e n t r a l } } \right\} , \quad \mathcal { C } _ { \mathrm { a n a t } } = \mathcal { C } \setminus \left\{ c _ { \mathrm { o u t s i d e } } \right\} ,\tag{2}
$$

where $\mathcal { C }$ denotes the set of anatomical region classes and $\mathcal { C } _ { \mathrm { a n a t } }$ denotes the subset excluding the outside region.

## 3.2 Anatomical Prior-Guided Routing

Gate selection. We obtain the cropped anatomical label map $A _ { \omega }$ corresponding to the image patch $I _ { \omega }$ . To determine which anatomical gate should be used for the patch, we calculate the occupancy ratio of each predefined mask in the patch. The anatomical occupancy ratio is defined as

$$
q ( \omega ) = \frac { 1 } { | \omega | } \sum _ { u \in A _ { \omega } } \mathbb { 1 } \{ A _ { \omega } ( u ) = c \} , \quad c \in \mathcal { C } ,\tag{3}
$$

where $| \omega |$ denotes the total number of voxels in the patch, the numerator counts the voxels in the cropped anatomical label map $A _ { \omega }$ that belong to class $c ,$ and \protec \mathb m {1} is an indicator function. The outside o ccupancy q\_\mathr {outside}(\om ga ) is computed to classify the patch \omega into outside or anatomical regions. If this value is greater than or equal to the threshold $\tau _ { \mathrm { o u t } }$ , the patch is defined as outside. If the outside occupancy is smaller than the threshold, the patch is assigned to the class with the largest occupancy among $\mathcal { C } _ { \mathrm { a n a t } }$

$$
\begin{array} { r } { G _ { \omega } = \left\{ \begin{array} { l l } { G _ { \mathrm { o u t s i d e } } , } & { q _ { \mathrm { o u t s i d e } } ( \omega ) \geq \tau _ { \mathrm { o u t } } , } \\ { G _ { c ^ { * } } , \ c ^ { * } = \arg \operatorname* { m a x } _ { c \in \mathcal { C } _ { \mathrm { a n a t } } } q _ { c } ( \omega ) , } & { q _ { \mathrm { o u t s i d e } } ( \omega ) < \tau _ { \mathrm { o u t } } , } \end{array} \right. } \end{array}\tag{4}
$$

where $G _ { \omega }$ denotes the routing gate for patch $\omega .$ According to the anatomical location classes in Eq. 2, a routing gate is assigned to each anatomical region.

Feature routing with anatomical gates. APRM is applied at each encoder stage and the anatomical gate is selected according to the anatomical location of each patch. The selected gate routes the feature tokens to a shared expert pool, allowing the routing function to be anatomy-specific while keeping the expert capacity shared across all anatomical regions. For expert routing, the feature map $h _ { s }$ at stage s is flattened into a spatial token sequence $X _ { s }$

$$
X _ { s } = \mathrm { F l a t t e n } ( h _ { s } ) \in \mathbb R ^ { N _ { s } \times C _ { s } } , \quad h _ { s } \in \mathbb R ^ { C _ { s } \times H _ { s } \times W _ { s } \times D _ { s } } , \quad N _ { s } = H _ { s } W _ { s } D _ { s } ,\tag{5}
$$

where $C _ { s } , H _ { s } , W _ { s } ,$ and $D _ { s }$ denote the channel, height, width, and depth at stage $s ,$ respectively. After flattening, the feature map is represented by $N _ { s }$ tokens, each

with a dimension $C _ { s }$ . For each token $x _ { s } \in X _ { s }$ , the assigned gate $G _ { \omega }$ computes sparse expert weights over shared experts and the routed output is obtained as

$$
\tilde { x } _ { s } = x _ { s } + \sum _ { i = 1 } ^ { M } G _ { \omega } ( x _ { s } ) _ { i } \cdot E _ { i } ( x _ { s } ) ,\tag{6}
$$

where $M$ is the number of shared experts, $E _ { i } ( x _ { s } )$ denotes the output of the i-th expert network, and $G _ { \omega } ( x _ { s } )$ is the sparse routing weight assigned by gate $G _ { \omega }$ . To prevent the model from relying on only a small subset of experts, we employ noisy top- $K$ routing, which injects Gaussian noise into the routing logits before selecting the active experts. The noisy logits and sparse gating weights are defined as follows:

$$
G _ { \omega } ( x _ { s } ) = \mathrm { S o f t m a x } \left( \mathrm { K e e p T o p K } ( H ( x _ { s } ) , K ) \right) ,\tag{7}
$$

$$
H ( x _ { s } ) _ { i } = ( x _ { s } W _ { g } ) _ { i } + \mathcal { N } ( 0 , 1 ) \cdot \mathrm { S o f t p l u s } \left( ( x _ { s } W _ { \mathrm { n o i s e } } ) _ { i } \right) ,\tag{8}
$$

$$
\operatorname { K e e p T o p K } ( v , K ) _ { i } = { \left\{ v _ { i } \atop - \infty \right.}   \quad { \mathrm { i f ~ } } v _ { i } { \mathrm { ~ i s ~ a m o n g ~ t h e ~ t o p } } . K { \mathrm { ~ e l e m e n t s ~ o f ~ } } v ,\tag{9}
$$

where $W _ { g }$ and $W _ { \mathrm { n o i s e } }$ are the learnable parameters of the selected anatomical gate for routing logits and noise scale, respectively. The \protect \operatorname {Softplus} function constrains the noise scale to be positive. After all tokens are routed, the updated token sequence $\tilde { X } _ { s } = \{ \tilde { x } _ { s } \} _ { n = 1 } ^ { N _ { s } }$ is reshaped back to the feature-map resolution:

$$
\tilde { h } _ { s } = \mathrm { U n f l a t t e n } ( \tilde { \boldsymbol { X } } _ { s } ) \in \mathbb { R } ^ { C _ { s } \times H _ { s } \times \boldsymbol { W } _ { s } \times \boldsymbol { D } _ { s } } ,\tag{10}
$$

where $\tilde { h } _ { s }$ is the output feature map at stage s.

## 3.3 Objective Function

The segmentation objective is defined as the sum of Dice loss and cross-entropy (CE) loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { D i c e } } + \mathcal { L } _ { \mathrm { C E } } . } \end{array}\tag{11}
$$

In sparse routing, the router may repeatedly assign high probabilities to only a small subset of experts, which can reduce expert diversity and limit the $\operatorname { e f - }$ fective capacity of the shared expert pool. To mitigate this issue, we introduce a lightweight expert balancing loss [10]. This loss is computed from the dense softmax probabilities of the selected router before top-K masking, so that all experts receive a learning signal regardless of whether they are selected by the sparse routing operation.

$$
p ( x _ { s } ) _ { i } = \frac { \exp { ( H ( x _ { s } ) _ { i } ) } } { \sum _ { j = 1 } ^ { M } \exp { ( H ( x _ { s } ) _ { j } } ) } .\tag{12}
$$

Using these dense routing probabilities, the average usage probability of expert i at stage s is computed over all tokens in the stage as

$$
\bar { p } ( X _ { s } ) _ { i } = \frac { 1 } { N _ { s } } \sum _ { x _ { s } \in X _ { s } } p ( x _ { s } ) _ { i } .\tag{13}
$$

The expert balancing loss is then defined by penalizing concentrated average expert usage across the routed encoder stages:

$$
\mathcal { L } _ { \mathrm { b a l } } = \frac { 1 } { \vert S _ { r } \vert } \sum _ { s \in S _ { r } } M \sum _ { i = 1 } ^ { M } \left( \bar { p } ( X _ { s } ) _ { i } \right) ^ { 2 } ,\tag{14}
$$

where $S _ { r }$ denotes the set of encoder stages. By minimizing this term, the model is encouraged to distribute the average dense routing probability more evenly across the shared expert pool rather than concentrating on a few experts. The final training objective combines the segmentation loss with the balancing loss as

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda \mathcal { L } _ { \mathrm { b a l } } . } \end{array}\tag{15}
$$

The balancing weight \lambda is empirically set to $1 0 ^ { - 2 }$ in our experiments.

## 3.4 Sampling Process

During training, a cropped region (i.e., patch) \omega is sampled from the input volume I, and the corresponding image patch $I _ { \omega }$ and anatomical patch $A _ { \omega }$ are extracted together. The anatomical patch is used to determine the routing gate according to Eq. (4). The anatomical prior used for routing is spatially aligned with the feature tokens processed by APRM at each encoder stage. During sampling, we process the full volume using sliding-window inference with an overlap ratio of 0.5 along each spatial dimension. Predictions from overlapping windows are combined using Gaussian-weighted voxel-wise aggregation, following the standard nnU-Net inference procedure. For each patch, the same crop is applied to the anatomical label map and the patch-level anatomical occupancy is computed to select the corresponding anatomical gate. Predictions from overlapping windows are aggregated in the original image space and the final segmentation is obtained from the merged probability map.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate 3D lung tumor segmentation using two public CT datasets: NSCLC [1, 6, 45] and MSD [2]. Both NSCLC and MSD are non-small cell lung cancer datasets. NSCLC provides subtype annotations consisting of large cell carcinoma, squamous cell carcinoma (SCC), adenocarcinoma (Aden.), and not otherwise specified (NOS), whereas MSD does not provide subtype information.

Table 1: Quantitative lung tumor segmentation results on the NSCLC [1,6,45] dataset with various backbones. Results are reported for each tumor subtype, where SCC denotes squamous cell carcinoma, Aden. denotes adenocarcinoma, and NOS denotes not otherwise specified. Average indicates the mean performance across all subjects. DSC and IoU are reported as percentages and higher values indicate better performance. Bold indicates the best performance.
<table><tr><td rowspan="3">Subtype</td><td rowspan="2">Dataset</td><td colspan="10">NSCLC [1, 6, 45]</td></tr><tr><td colspan="2">Large cell</td><td colspan="2">SCC</td><td colspan="2">Aden.</td><td colspan="2">NOS</td><td colspan="2">Average</td></tr><tr><td>Module</td><td>DSC(%)</td><td>IoU(%)</td><td>DSC(%)</td><td>IoU(%)</td><td>DSC(%)</td><td>IoU(%)</td><td>DSC(%)</td><td>IoU(%)</td><td>DSC(%)</td><td>IoU(%)</td></tr><tr><td rowspan="3">nnU-Net [15]</td><td>Baseline</td><td>71.30</td><td>61.28</td><td>59.18</td><td>48.05</td><td>79.66</td><td>70.70</td><td>57.66</td><td>48.27</td><td>65.64</td><td>55.64</td></tr><tr><td>+ MoE</td><td>72.52</td><td>61.86</td><td>62.54</td><td>51.10</td><td>77.77</td><td>68.79</td><td>57.65</td><td>48.95</td><td>66.66</td><td>56.55</td></tr><tr><td>+ APRM</td><td>72.38</td><td>61.39</td><td>62.38</td><td>50.93</td><td>77.30</td><td>68.05</td><td>60.62</td><td>50.90</td><td>67.29</td><td>56.75</td></tr><tr><td rowspan="3">ResEnc [16]</td><td>Baseline</td><td>73.54</td><td>62.32</td><td>63.37</td><td>51.41</td><td>84.91</td><td>75.40</td><td>64.22</td><td>54.27</td><td>70.00</td><td>59.17</td></tr><tr><td>+ MoE</td><td>73.27</td><td>61.83</td><td>64.10</td><td>52.53</td><td>83.48</td><td>73.94</td><td>63.20</td><td>53.65</td><td>69.63</td><td>58.94</td></tr><tr><td>+ APRM</td><td>73.63</td><td>62.44</td><td>63.38</td><td>51.69</td><td>86.88</td><td>77.53</td><td>66.32</td><td>56.17</td><td>70.88</td><td>60.10</td></tr><tr><td rowspan="3">MedNeXt [48]</td><td>Baseline</td><td>66.80</td><td>56.06</td><td>54.63</td><td>43.20</td><td>70.84</td><td>59.61</td><td>57.22</td><td>47.62</td><td>64.08</td><td>53.13</td></tr><tr><td>+ MoE</td><td>72.76</td><td>61.91</td><td>63.04</td><td>51.36</td><td>78.70</td><td>69.60</td><td>60.54</td><td>51.03</td><td>67.76</td><td>57.30</td></tr><tr><td>+ APRM</td><td>72.52</td><td>61.43</td><td>65.53</td><td>53.54</td><td>85.37</td><td>75.20</td><td>58.38</td><td>49.37</td><td>68.81</td><td>58.16</td></tr><tr><td rowspan="3">STUNet [14]</td><td>Baseline</td><td>72.44</td><td>61.37</td><td>57.00</td><td>46.68</td><td>75.46</td><td>65.73</td><td>58.75</td><td>49.56</td><td>65.05</td><td>54.88</td></tr><tr><td>+ MoE</td><td>66.68</td><td>56.47</td><td>62.29</td><td>51.15</td><td>76.12</td><td>66.04</td><td>60.54</td><td>51.43</td><td>65.28</td><td>55.06</td></tr><tr><td>+ APRM</td><td>73.27</td><td>62.30</td><td>56.48</td><td>46.12</td><td>79.67</td><td>69.41</td><td>60.67</td><td>50.68</td><td>66.31</td><td>55.87</td></tr></table>

Following the nnU-Net preprocessing setup, both datasets were resampled to a target spacing of $1 \times 1 \times 3 \mathrm { m m ^ { 3 } }$ and have an image size of $5 1 2 \times 5 1 2 \times ( 8 2 \ – 1 7 6 )$ NSCLC is used for model training and evaluation. MSD is used as an external test dataset. From NSCLC, we use 319 subjects for training and 80 subjects for test. From MSD, we use 63 subjects for test.

Implementation details. All methods are implemented using the nnU-Net framework [15, 16]. The input shape of 224 × 192 × 48 was automatically determined during the nnU-Net preprocessing steps. We use the training objective in Eq. (15), with the balancing loss weight set to $\lambda = 1 0 ^ { - 2 }$ . Anatomical masks are extracted using TotalSegmentator [53]. We use seven anatomical routing gates and a shared expert pool per routed stage with $M = 8$ experts and Top-K parameter set to $K = 2$ . For the +MoE, we use the same expert architecture, number of experts M=8 , top-K sparsity K=2 , and sparse weighted aggregation as APRM, but use a single feature-driven router.

Evaluation metrics. We use Dice similarity coeficient (DSC) and intersection over union (IoU) to evaluate volumetric segmentation performance.

Comparison backbones. We compare four backbone architectures: the U-Net [47] architecture in nnU-Net [15], ResEnc [16], MedNeXt [48], and STUNet [14]. All backbones are trained and evaluated within the nnU-Net framework.

## 4.2 Results of Lung Tumor Segmentation

Results on NSCLC. We compare APRM with the baseline and MoE across several backbones, as shown in Table 1. Among the backbones, ResEnc achieves the best performance. Adding MoE improves the results for most tumor subtypes, while Adding APRM improves the performance of all backbones and achieves the best average scores for every backbone. These results indicate that anatomy-guided routing enables efective feature adaptation across heterogeneous tumor groups. Fig. 4 (top) shows qualitative results on NSCLC, where the tumors have diferent sizes and locations depending on the subtype. The baseline often fails to segment the tumor region and MoE captures part of the tumor but misses fine details. In contrast, APRM produces results that are closest to the ground truth. These results show that APRM can segment tumors well across diferent locations and sizes, demonstrating the efectiveness of the proposed module.

![](images/ab60d6896fca388d17e1391dac455d27dc3d836580efa33e2de9f1c15e57633e.jpg)  
Fig. 4: Qualitative lung tumor segmentation results. Left: comparison with diferent modules using ResEnc as the backbone. Right: comparison with diferent modules using MedNeXt as the backbone. The top two rows show results for diferent NSCLC tumor subtypes, while the bottom row shows results on the MSD dataset. Each result shows a zoomed-in view of the yellow bounding box in the image.

Results on MSD. We also evaluate each baseline and our module on MSD, which is used as an external test dataset. The quantitative results are shown in Table 2. Consistent with the results on NSCLC, ResEnc achieves the best performance among the baseline models, while APRM improves segmentation performance across all evaluated backbones. The bottom row examples in Fig. 4 show qualitative results on MSD using ResEnc and Med-NeXt for two cases: a large tumor on the left and a small tumor on the right. In the left case, the comparison methods miss a region that is separated from the main large tumor,

Table 2: Quantitative lung tumor segmentation results on the MSD dataset [2]. Bold indicates the best performance.
<table><tr><td></td><td>Dataset</td><td colspan="2">MSD [2]</td></tr><tr><td>Backbone</td><td>Module</td><td>DSC(%)</td><td>IoU(%)</td></tr><tr><td rowspan="3">nnU-Net [15]</td><td>Baseline</td><td>54.86</td><td>43.09</td></tr><tr><td>+ MoE</td><td>55.18</td><td>43.11</td></tr><tr><td>+ APRM</td><td>58.41</td><td>46.32</td></tr><tr><td rowspan="3">ResEnc [16]</td><td>Baseline</td><td>66.38</td><td>53.47</td></tr><tr><td>+ MoE</td><td>65.91</td><td>53.30</td></tr><tr><td>+ APRM</td><td>66.72</td><td>54.02</td></tr><tr><td rowspan="3">MedNeXt [48]</td><td>Baseline</td><td>49.36</td><td>37.69</td></tr><tr><td>+ MoE</td><td>62.49</td><td>49.96</td></tr><tr><td>+ APRM</td><td>62.91</td><td>50.33</td></tr><tr><td rowspan="3">STUNet [14]</td><td>Baseline</td><td>56.28</td><td>43.99</td></tr><tr><td>+ MoE</td><td>55.94</td><td>43.83</td></tr><tr><td>+ APRM</td><td>58.93</td><td>46.83</td></tr></table>

Table 3: Ablation study on the NSCLC and MSD. ResEnc is used as the backbone.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Dataset Setting</td><td colspan="2">NSCLC [1,6, 45]</td><td colspan="2">MSD [2]</td></tr><tr><td>DSC(%)</td><td>IoU(%)</td><td>DSC(%)</td><td>IoU(%)</td></tr><tr><td rowspan="2">Top-K sparsity</td><td>K = 1</td><td>70.54</td><td>59.93</td><td>65.06</td><td>52.36</td></tr><tr><td>K = 2</td><td>70.88</td><td>60.10</td><td>66.72</td><td>54.02</td></tr><tr><td rowspan="2">Number of experts</td><td>M = 4</td><td>70.80</td><td>60.21</td><td>65.74</td><td>53.29</td></tr><tr><td>M = 8</td><td>70.88</td><td>60.10</td><td>66.72</td><td>54.02</td></tr><tr><td rowspan="3">Outside threshold</td><td> $\tau _ { \mathrm { o u t } } = 0 . 8 5$ </td><td>69.83</td><td>59.06</td><td>66.56</td><td>53.91</td></tr><tr><td> $\tau _ { \mathrm { o u t } } = 0 . 9$ </td><td>70.88</td><td>60.10</td><td>66.72</td><td>54.02</td></tr><tr><td> $\tau _ { \mathrm { o u t } } = 0 . 9 5$ </td><td>70.32</td><td>59.61</td><td>66.15</td><td>53.39</td></tr></table>

whereas APRM correctly detects it as part of the tumor. In the right case, the tumor is very small, and the comparison methods fail to detect it, while APRM successfully captures the tumor region.

## 4.3 Ablation Studies

Top-K sparsity. We observe how the top-K setting afects the sparse routing strategy, as shown in Table 3 (top). Using the top two experts to aggregate features improves segmentation performance on both NSCLC and MSD compared with using only the top expert. Based on this result, we set $K = 2$ in our experiments.

Number of experts. We compare diferent values of the number of experts M in the shared expert pool, as shown in Table 3 (middle). Both $M = 4$ and $M = 8$ achieve similar performance, with M=8 providing a slight improvement. This result suggests that performance does not simply increase by adding more experts. Rather, the routing mechanism dynamically selects the features needed for each input from the shared expert pool. Based on these results, we use $M = 8$ in our experiments.

Outside threshold. We evaluate diferent values of the outside threshold $\tau _ { \mathrm { o u t } }$ and observe their efect on segmentation performance, as shown in Table 3 (bottom). Because most voxels within a patch are background on average and only a limited portion corresponds to lung-related anatomy, a relatively high \tau \_\mathrm {out} is needed to reliably determine the anatomical location of each patch. At the lower value $\tau _ { \mathrm { o u t } } = 0 . 8 5$ , performance decreases noticeably, indicating that many patches are assigned to the outside category and the model consequently relies predominantly on the outside gate. This behavior makes the performance close to that of ResEnc + MoE in Tables 1 and 2. In contrast, at the higher value $\tau _ { \mathrm { o u t } } = 0 . 9 5$ , even a small overlap with an anatomical mask can determine the patch location, which may make anatomical distinctions ambiguous and lead to degraded performance. We obtain the best performance with $\tau _ { \mathrm { o u t } } = 0 . 9$

![](images/b22618ab3896825528ef2c019fa2171e38dbb4bf5eec85428f8dd491c1c2d6e4.jpg)  
Fig. 5: Visualization of anatomical patch selection for anatomical gate assignment. Left: an upper (superior) slice closer to the head. Right: a lower (inferior) slice closer to the abdomen. In each patch image, the upper-left label indicates the selected anatomical region, while the upper-right inset shows the distribution of anatomical masks.

## 4.4 Computational Cost Analysis

We report computational cost using nnU-Net as the backbone. As shown in Table 4, although APRM adds additional anatomy-specific router parameters from 0.017M to 0.122M due to the seven anatomy-specific gates, the

Table 4: Computational cost on the NSCLC test set using nnU-Net. Inference time reports segmentation processing only.
<table><tr><td>Method</td><td></td><td>Total params Router params Infer. time (s)</td><td></td></tr><tr><td>nnU-Net</td><td>30.70M</td><td></td><td> $2 2 . 1 4 \pm 4 . 8 0$ </td></tr><tr><td>+ MoE</td><td>49.38M</td><td>0.017M</td><td> $4 8 . 5 7 \pm 1 5 . 0 1$ </td></tr><tr><td>+ APRM</td><td>49.48M</td><td>0.122M</td><td> $5 0 . 4 1 \pm 1 5 . 4 6$ </td></tr></table>

total parameter count increases only from 49.38M to 49.48M (∼0.2%) compared with MoE. Importantly, for each patch, APRM evaluates only the router associated with the selected anatomical gate rather than densely executing all seven routers. Accordingly, segmentation inference increases only modestly from 48.57s to 50.41s (∼3.8%).

## 4.5 Analysis of Anatomical Prior-guided Routing

Anatomical patch selection. Fig. 5 visualizes how patch locations are identified and how anatomical gates are selected during inference. In the upperleft patch of Fig. 5 (left), only a small portion overlaps with the right upper lobe; therefore, the patch is assigned to the outside region according to \tau \_\mathrm {out} . In the next patch, both the right upper lobe and left upper lobe are present, and the background ratio is below $\tau _ { \mathrm { o u t } }$ . Therefore, the right upper lobe is selected because it occupies the largest anatomical mask area. Although the exact tumor location is not explicitly given to the model, sliding-window inference allows each patch to be assigned to diferent anatomical regions and processed through the corresponding anatomical gates. This enables the model to select

Expert 7

Patch  
Expert 1  
Expert 2  
Expert 3  
Expert 4  
Expert 5  
Expert 6  
Expert 8  
Aggregation  
![](images/810b677acd8eefb64068932ea3fa5b13bc101c4770f86d47696010a6d4041a1c.jpg)  
Fig. 6: Visualization of expert features according to anatomical location. Each image shows the voxels for which the corresponding expert is selected as the top-1 expert and the red contour denotes the tumor region. The aggregation map represents the final feature map obtained after sparse routing.

segmentation-relevant features from multiple anatomical contexts and dynamically reflect heterogeneity caused by tumor location, size, and subtype. A similar trend is observed in Fig. 5 (right).

Routing feature visualization. We further visualize the features represented by the shared experts, as shown in Fig. 6. Because diferent experts are assigned at the voxel level, each expert produces a distinct top-1 voxel map. For the right upper lobe gate $( G _ { \mathrm { r u l } }$ , top row), tumor-related regions are mainly selected by Experts 2 and 3: Expert 2 focuses on the tumor interior, while Expert 3 responds to the tumor boundary. The aggregation feature map preserves these complementary tumor-related responses. For the right lower lobe gate $( G _ { \mathrm { r l l } }$ , second row), tumor-related regions are represented by Experts 1 and 7, showing a routing pattern that difers from that of the right upper lobe gate, and their aggregated feature map also clearly highlights the tumor region. Similarly, for the left upper lobe gate $( G _ { \mathrm { l u l } }$ , third row), Experts 4 and 5 are selected for tumorrelated regions. The central and outside gates also select diferent experts to represent tumor regions. These observations show that, although all anatomical gates share the same expert pool, diferent experts are assigned depending on the anatomical context and tumor appearance. Thus, the proposed model learns gate-specific routing strategies within the shared expert pool, enabling it to capture heterogeneity in tumor location and size and extract efective segmentation features.

## 5 Discussion

We proposed APRM to incorporate anatomical priors into patch-based 3D lung tumor segmentation. By selecting an anatomical gate for each patch and routing features through a shared expert pool, APRM enables location-aware feature adaptation without assigning separate experts to each anatomical region. The experimental results show that APRM consistently improves segmentation performance across multiple backbones on the NSCLC dataset and also generalizes to the external MSD dataset. Compared with a standard MoE module, APRM provides more stable gains, indicating that anatomical context is useful beyond simply increasing model capacity. The ablation and visualization results further show that sparse routing with shared experts learns diferent feature responses for diferent anatomical regions, helping the model capture tumor interiors, boundaries, and location-dependent heterogeneity.

Limitations. The proposed method depends on anatomical masks extracted by a pretrained segmentation model. Although this makes APRM practical because it does not require manual anatomical annotation, errors in the anatomical masks may lead to incorrect gate assignment, especially near lobe boundaries, mediastinal structures, or abnormal anatomy. The anatomical categories used in this work are defined at a relatively coarse level, including lung lobes, the central region, and an outside region. Finer anatomical subdivisions or vessel- and airway-aware priors may provide additional context for tumors with complex local interactions, but they may also increase routing ambiguity and computational cost. The current framework uses deterministic patch-level gate selection based on anatomical occupancy. This improves stability and interpretability, but it may be suboptimal for patches containing multiple relevant anatomical structures. Future work could explore soft anatomical routing, uncertainty-aware gate assignment, or joint optimization of anatomical priors and tumor segmentation.

## Acknowledgement

This study was supported by the SMC-SKKU Future Convergence Research Program Grant (SMO126115). It was also supported by the AI Graduate School Support Program (Sungkyunkwan University) (RS-2019-II190421), the National Research Foundation of Korea (RS-2024-00408040, RS-2026-25598771), the ICT Creative Consilience Program Grant (IITP-2026-RS-2020-II201821), the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant (RS-2026-25528384, Resource-Intensive AI Technologies Based on Sustainable GPU Integrated Platforms), the IITP grant (IITP-2026-RS-2026-25615334, AI Star Fellowship Support Program), and an IITP grant for the AI Computing Support Project for R&D (RS-2026-25505492, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale), all funded by the Ministry of Science and ICT (MSIT), Republic of Korea.

## References

1. Aerts, H.J., Velazquez, E.R., Leijenaar, R.T., Parmar, C., Grossmann, P., Carvalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., Rietveld, D., et al.: Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach. Nature communications 5(1), 4006 (2014)

2. Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B.A., Litjens, G., Menze, B., Ronneberger, O., Summers, R.M., et al.: The medical segmentation decathlon. Nature communications 13(1), 4128 (2022)

3. Bray, F., Laversanne, M., Sung, H., Ferlay, J., Siegel, R.L., Soerjomataram, I., Jemal, A.: Global cancer statistics 2022: Globocan estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: a cancer journal for clinicians 74(3), 229–263 (2024)

4. Chen, Q., Zhu, L., He, H., Zhang, X., Zeng, S., Ren, Q., Lu, Y.: Low-rank mixtureof-experts for continual medical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 382–392. Springer (2024)

5. Çiçek, Ö., Abdulkadir, A., Lienkamp, S.S., Brox, T., Ronneberger, O.: 3d u-net: learning dense volumetric segmentation from sparse annotation. In: International conference on medical image computing and computer-assisted intervention. pp. 424–432. Springer (2016)

6. Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Mafitt, D., Pringle, M., et al.: The cancer imaging archive (tcia): maintaining and operating a public information repository. Journal of digital imaging 26(6), 1045–1057 (2013)

7. Diakogiannis, F.I., Waldner, F., Caccetta, P., Wu, C.: Resunet-a: A deep learning framework for semantic segmentation of remotely sensed data. ISPRS Journal of Photogrammetry and Remote Sensing 162, 94–114 (2020)

8. Engelson, S., Ehrhardt, J., Kepp, T., Niemeijer, J., Handels, H.: Lnq challenge 2023: Learning mediastinal lymph node segmentation with a probabilistic lymph node atlas. arXiv preprint arXiv:2406.03984 (2024)

9. Ettinger, D.S., Wood, D.E., Aisner, D.L., Akerley, W., Bauman, J., Chirieac, L.R., D’Amico, T.A., DeCamp, M.M., Dilling, T.J., Dobelbower, M., et al.: Non–small cell lung cancer, version 5.2017, nccn clinical practice guidelines in oncology. Journal of the National Comprehensive Cancer Network 15(4), 504–535 (2017)

10. Fedus, W., Zoph, B., Shazeer, N.: Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal of Machine Learning Research 23(120), 1–39 (2022)

11. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H.R., Xu, D.: Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images. In: International MICCAI brainlesion workshop. pp. 272–284. Springer (2021)

12. Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R., Xu, D.: Unetr: Transformers for 3d medical image segmentation. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision. pp. 574–584 (2022)

13. Huang, H., Zheng, H., Lin, L., Cai, M., Hu, H., Zhang, Q., Chen, Q., Iwamoto, Y., Han, X., Chen, Y.W., et al.: Medical image segmentation with deep atlas prior. IEEE transactions on medical imaging 40(12), 3519–3530 (2021)

14. Huang, Z., Wang, H., Deng, Z., Ye, J., Su, Y., Sun, H., He, J., Gu, Y., Gu, L., Zhang, S., et al.: Stu-net: Scalable and transferable medical image segmen-

tation models empowered by large-scale supervised pre-training. arXiv preprint arXiv:2304.06716 (2023)

15. Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H.: nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18(2), 203–211 (2021)

16. Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., Jaeger, P.F.: nnu-net revisited: A call for rigorous validation in 3d medical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 488–498. Springer (2024)

17. Jeon, Y.S., Yang, H., Fu, H., Kway, Y., Feng, M.: No more sliding window: eficient 3d medical image segmentation with diferentiable top-k patch sampling. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 376–386. Springer (2025)

18. Jiang, Y., Shen, Y.: M4oe: A foundation model for medical multimodal image segmentation with mixture of experts. In: international conference on medical image computing and computer-assisted intervention. pp. 621–631. Springer (2024)

19. Jo, G., Kim, J., Park, B.Y., Park, H.: Deafusion: Detail-aware image fusion for whitematter hyperintensity segmentation. In: 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). pp. 3361–3366. IEEE (2024)

20. Kashyap, M., Wang, X., Panjwani, N., Hasan, M., Zhang, Q., Huang, C., Bush, K., Chin, A., Vitzthum, L.K., Dong, P., et al.: Automated deep learning–based detection and segmentation of lung tumors at ct imaging. Radiology 314(1), e233029 (2025)

21. Kim, J., Kim, M., Park, H.: Domain aware multi-task pretraining of 3d swin transformer for t1-weighted brain mri. In: Asian Conference on Computer Vision. pp. 121–141. Springer (2024)

22. Kim, J., Na, I., Chung, J., Song, H.N., Kim, K., Ju, S., Eun, M.Y., Seo, W.K., Park, H.: Enhancing intracranial vessel segmentation using difusion models without manual annotation for 3d time-of-flight magnetic resonance angiography. Computerized Medical Imaging and Graphics p. 102651 (2025)

23. Kim, J., Na, I., Ko, E.S., Park, H.: Tumor synthesis conditioned on radiomics. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 3635–3646. IEEE (2025)

24. Kim, J., Na, I., Kwon, J., Seo, W.K., Park, H.: Weakly-supervised segmentation using sparse single point annotations for lumen and wall of carotid arteries in 3d mri. Computer Methods and Programs in Biomedicine 269, 108881 (2025)

25. Kim, J., Park, H.: Visual prompt tuning for task-flexible medical image synthesis. Computer Methods and Programs in Biomedicine p. 109244 (2026)

26. Kwon, J., Kim, J., Kim, T., Seo, S.W., Cho, H.h., Park, H.: Blood pressure assisted cerebral microbleed segmentation via meta-matching. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 77–86. Springer (2025)

27. Kwon, J., Kim, J., Park, H.: Leveraging segmentation-guided spatial feature embedding for overall survival prediction in glioblastoma with multimodal magnetic resonance imaging. Computer Methods and Programs in Biomedicine 255, 108338 (2024)

28. Kwon, J., Seo, S.W., Park, H.: Anatomically-guided segmentation of cerebral microbleeds in t1-weighted and t2\*-weighted mri. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 24–33. Springer (2024)

29. Kwon, J., Seo, S.W., Park, H.: Enhancing cerebral microbleed segmentation with pretrained unetr++. In: 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). pp. 3372–3377. IEEE (2024)

30. Lambe, G., Durand, M., Buckley, A., Nicholson, S., McDermott, R.: Adenocarcinoma of the lung: from bac to the future. Insights into imaging 11(1), 69 (2020)

31. Larrazabal, A.J., Martínez, C., Glocker, B., Ferrante, E.: Post-dae: anatomically plausible segmentation via post-processing with denoising autoencoders. IEEE transactions on medical imaging 39(12), 3813–3820 (2020)

32. Latimer, K.M., Mott, T.F.: Lung cancer: diagnosis, treatment principles, and screening. American family physician 91(4), 250–256 (2015)

33. Li, C., Wang, H., Jiang, Y., Fu, W., Liu, X., Zhong, R., Cheng, B., Zhu, F., Xiang, Y., He, J., et al.: Advances in lung cancer screening and early detection. Cancer biology & medicine 19(5), 591–608 (2022)

34. Li, J., Liu, J., Wang, W., et al.: Systematic review and meta-analysis of deep learning applications in ct lung cancer segmentation. Biomed. Signal Process. Control 89, 105804 (2024)

35. Li, R., Wu, L., Gu, J., Xu, Q., Chen, W., Cai, X., Bu, J.: Moe-sam: Enhancing sam for medical image segmentation with mixture-of-experts. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 367–377. Springer (2025)

36. Liu, L., Wolterink, J.M., Brune, C., Veldhuis, R.N.: Anatomy-aided deep learning for medical image segmentation: a review. Physics in Medicine & Biology 66(11), 11TR01 (2021)

37. Liu, X., Song, L., Liu, S., Zhang, Y.: A review of deep-learning-based medical image segmentation methods. Sustainability 13(3), 1224 (2021)

38. Mathai, T.S., Liu, B., Summers, R.M.: Segmentation of mediastinal lymph nodes in ct with anatomical priors. International journal of computer assisted radiology and surgery 19(8), 1537–1544 (2024)

39. Mercieca, S., Belderbos, J.S., van Herk, M.: Challenges in the target volume definition of lung cancer radiotherapy. Translational lung cancer research 10(4), 1983 (2021)

40. Milletari, F., Navab, N., Ahmadi, S.A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 2016 fourth international conference on 3D vision (3DV). pp. 565–571. Ieee (2016)

41. Na, I., Kim, J., Ko, E.S., Park, H.: Radiomicsfill-mammo: Synthetic mammogram mass manipulation with radiomics features. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 723–733. Springer (2024)

42. Nilssen, Y., Brustugun, O.T., Fjellbirkeland, L., Helland, Å., Møller, B., Wahl, S.G.F., Solberg, S.: Distribution and characteristics of malignant tumours by lung lobe. BMC Pulmonary Medicine 24(1), 106 (2024)

43. Oh, Y., Jin, P., Park, S., Kim, S., yoon, S., Kim, J.S., Kim, K., Li, X., Li, Q.: Distribution-aware fairness learning in medical image segmentation from a controltheoretic perspective. In: Forty-second International Conference on Machine Learning (2025), https://openreview.net/forum?id=BUONdewsBa

44. Primakov, S.P., Ibrahim, A., van Timmeren, J.E., Wu, G., Keek, S.A., Beuque, M., Granzier, R.W., Lavrova, E., Scrivener, M., Sanduleanu, S., et al.: Automated detection and segmentation of non-small cell lung cancer computed tomography images. Nature communications 13(1), 3423 (2022)

45. Prior, F., Smith, K., Sharma, A., Kirby, J., Tarbox, L., Clark, K., Bennett, W., Nolan, T., Freymann, J.: The public cancer radiology imaging collections of the cancer imaging archive. Scientific data 4(1), 170124 (2017)

46. Riquelme, C., Puigcerver, J., Mustafa, B., Neumann, M., Jenatton, R., Susano Pinto, A., Keysers, D., Houlsby, N.: Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems 34, 8583–8595 (2021)

47. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

48. Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F., Maier-Hein, K.H.: Mednext: transformer-driven scaling of convnets for medical image segmentation. In: International conference on medical image computing and computer-assisted intervention. pp. 405–415. Springer (2023)

49. Rue, N., Jo, G., Na, I., Oh, S., Lee, H.Y., Park, H.: Label-free lung mri segmentation via misalignment-aware difusion translation. Journal of Imaging Informatics in Medicine pp. 1–17 (2026)

50. Sasaki, T., Kuno, H., Hiyama, T., Oda, S., Masuoka, S., Miyasaka, Y., Taki, T., Nagasaki, Y., Ohtani-Kim, S.J.y., Ishii, G., et al.: 2021 who classification of lung cancer: molecular biology research and radiologic-pathologic correlation. Radiographics 44(3), e230136 (2024)

51. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J.: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017)

52. Shi, L., Zhao, J., Wei, Z., Wu, H., Sheng, M.: Radiomics in distinguishing between lung adenocarcinoma and lung squamous cell carcinoma: a systematic review and meta-analysis. Frontiers in Oncology 14, 1381217 (2024)

53. Wasserthal, J., Breit, H.C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D.T., Cyriac, J., Yang, S., et al.: Totalsegmentator: robust segmentation of 104 anatomic structures in ct images. Radiology: Artificial Intelligence 5(5), e230024 (2023)

54. You, X., He, J., Yang, J., Gu, Y.: Learning with explicit shape priors for medical image segmentation. IEEE Transactions on Medical Imaging 44(2), 927–940 (2024)

55. Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A.M., Le, Q.V., Laudon, J., et al.: Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems 35, 7103–7114 (2022)