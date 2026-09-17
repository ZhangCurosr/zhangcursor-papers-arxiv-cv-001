# MCLC-NET: Multimodal Continual Learning for Leaf

Counting

Ruchi Bhatt, Pratibha Kumari, Shreya Bansal, Vedant Agnihotri, Dwarikanath Mahapatra, Mukesh Saini

Abstract—Leaf counting is an important task in plant phenotyping for monitoring plant growth and estimating crop yield. Most existing methods rely on RGB images, but their performance is often affected by occlusion, lighting variations, and other real-world challenges. Additional modalities, such as depth and thermal images, can provide useful complementary information. However, multimodal leaf counting remains underexplored. Also, many existing methods assume that all training data are available simultaneously, which is impractical in real agricultural settings, where data is collected over time from multiple sources. To address these challenges, we propose MCLC-NET, a multimodal continual learning framework for leaf counting. It learns tasks sequentially using a memory-based strategy with a memory buffer to retain important samples from previous tasks. We also introduce MMLC, a real-world multimodal leaf-counting dataset designed for a domainincremental scenario (DIS) in CL. It contains RGB, depth, and thermal images collected across different crop types under varying environmental conditions, arranged in three orderings: crop-wise, time-wise, and mixed. Experimental results, averaged over three random seeds, demonstrate that MCLC-NET consistently outperforms existing methods across all three task orderings, achieving the lowest AMSE of 0.675±0.027, 0.542±0.069, and 0.745±0.057, respectively. The MMLC dataset is available at MMLC Dataset.

Index Terms—multimodal, leaf counting, experience replay, continual learning

## I. INTRODUCTION

One of the key tasks in agricultural research is plant phenotyping [1], [2], which involves analyzing visible plant traits to monitor growth and development. Among these traits, leaf counting [3] is particularly important, as it provides valuable insights into the plant’s health [4], [5], growth stage [1], [6], and expected yield potential [7] [8]. Traditionally, researchers have developed imagebased methods to automate leaf counting that rely solely on unimodal images, typically RGB images. While RGB images are easily captured and widely used, they often face difficulties in real agricultural conditions. Problems such as occlusion, varying lighting, and complex backgrounds can affect their performance in real-world conditions [9]. As a result, models trained only on RGB data may not generalize well in practical scenarios. To improve this, researchers have begun exploring multimodal approaches that combine RGB data with other modalities, such as depth and thermal images.

Each modality provides different information. RGB captures color and texture, thermal images reflect temperature-related patterns, and depth images provide information about relative object distance. When used together, these modalities can complement each other and help the model better handle challenging conditions, such as poor lighting or occluded leaves. Despite its vast potential, only a few studies have explored leaf counting using multimodal data [9], making this an underexplored area. Also, existing multimodal datasets suitable for leaf counting, such as MSU-PID [10], are limited in number and are primarily collected in controlled environments. These settings may not reflect the variability and complexity of real field conditions. To address this limitation, we collect a new dataset from real farms comprising RGB, depth, and thermal images of different crops under varying conditions.

Most existing leaf-counting methods rely on one-time training and deployment of deep models and assume that all training data is available at once. In practice, this assumption does not hold, as in real deployments, data is not available all at once. Instead, it is collected sequentially over time across different sources, seasons, locations, and environmental conditions. This makes the retraining from scratch impractical due to computational and storage constraints. This creates a need for models that can continuously adapt to new data distributions while retaining previously learned knowledge. However, a key challenge in such settings is that models tend to forget previously learned information when trained on new data. This issue is known as ”catastrophic forgetting” [11]. Because of this, standard deep models are not well-suited to real-world agricultural applications where conditions are constantly changing. CL [12], [13] addresses this problem by allowing models to learn sequentially from new data while retaining past knowledge. This approach allows the model to adapt to changing environments, such as new crop types, growth stages, or imaging conditions, without losing performance on older data. We formulate leaf counting as a domainincremental scenario (DIS) problem [14], where the task remains the same but the data distribution changes across domains such as crop type, capture time, and environmental conditions. Unlike class-incremental learning [15], the goal is not to learn new categories but to adapt to these distribution shifts while preventing catastrophic forgetting.

Among CL strategies, namely regularization [12], architectural [16], and rehearsal-based (memorybased) [13], rehearsal-based methods are widely used due to their simplicity and effectiveness. According to the literature, generative replay avoids storing past data but requires complex sample generation [17]. In contrast, memory-based replay is simpler, often more effective, and widely used in CL [18], which motivates our choice in this work. Although rehearsal-based CL methods have shown promising performance, most existing approaches are primarily designed for classification problems and unimodal settings. Their direct application to multimodal leaf counting is challenging due to the regression nature of the task and the need to effectively utilize complementary information from multiple modalities. Moreover, existing memory-based replay strategies often rely on random sampling, which may result in the storage of redundant or less informative samples. In contrast, the proposed framework focuses on multimodal continual regression. It uses uncertaintyand diversity-aware sample selection rather than random sampling to retain representative samples from previous tasks while adapting to new domains.

In this paper, we propose a rehearsal-based framework, Multimodal Continual Learning for Leaf Counting (MCLC), that combines the strengths of multimodal data and CL to provide a robust, adaptive leafcounting model. MCLC integrates samples from multimodal datasets into a single representation and learns sequentially across tasks using a memory-based CL strategy. We store a few important samples from previous tasks and use them to represent past data while learning new tasks. Here, each task refers to a subset of data, defined by crop type, capture time, and environmental conditions. This DIS setup reflects real-world agricultural scenarios and highlights the need for CL in leaf counting. To support this, we introduce the Multimodal Leaf Counting (MMLC) dataset, collected from real farms. To the best of our knowledge, we are the first to explore multimodal continual learning for domainincremental leaf counting using RGB, depth, and thermal modalities.

The main contributions of this paper are highlighted below:

• We propose MCLC, a multimodal memory-based CL framework for leaf counting that combines uncertainty-aware and diversity-aware sample selection to retain informative samples from previous tasks while adapting to new tasks.

• We introduce MMLC, a real-world multimodal leaf counting dataset containing paired RGB, depth, and thermal images collected from different crops under varying environmental conditions for domainincremental CL.

The rest of the paper is structured as follows. Section II contains the related works. Section III consists of a detailed description of the proposed dataset. Section IV includes the proposed methodology. Section V consists of the experiments and results, followed by an ablation study in Section VI. The conclusion is given in section VII followed by limitations and future work in section VIII.

## II. RELATED WORK

Leaf counting is important for plant phenotyping [1] as it helps monitor plant growth [26] [27], development [28], and health [29]. Many deep learning-based methods have been proposed for automatic leaf-counting [22] [30]. In this section, we briefly review existing leaf-counting methods, the use of multimodal data for counting, and recent advances in CL, particularly in multimodal settings.

Leaf counting methods. Existing leaf counting paradigms fall into four main categories: direct regression [9], [19], [21], segmentation-based [31], [32], density estimation [33], [34], and object detection methods [22], [35]. While most works rely on RGB images, some have explored multimodal data (e.g., depth, thermal, NIR, fluorescence) to improve robustness [7], [9]. However, existing studies use static settings and do not address the challenges of learning from sequential data. In contrast, in this work, we explore leaf counting from a CL perspective using multimodal inputs, addressing both distribution shifts and scalability in real-world agricultural environments for the first time.

Multimodal Continual Learning. CL methods are broadly categorized into regularization-based [12], architectural-based [16], and rehearsal-based approaches [13], [36]. Regularization-based methods aim to preserve previously acquired knowledge by constraining updates to important model parameters, without requiring access to past data. Architecturalbased methods allocate distinct portions of the network to different tasks, using either fixed-capacity or dynamically expanding architectures to accommodate new information. Rehearsal-based approaches mitigate forgetting by storing and replaying representative samples from prior tasks. Within this category, experience replay (ER) [37] utilizes a memory buffer to retain raw data samples, while generative replay [38] relies on generative models to synthesize past data. Although generative methods reduce storage requirements, they introduce significant complexity. In contrast, ER has proven to be simpler, more effective, and widely adopted in recent studies [18], and is therefore explored in this work. A key challenge in ER is to select representative samples for the memory buffer. Earlier works, such as iCaRL [39], use a herding strategy to select samples closest to the class means, approximating the data distribution with limited memory. Similarly, prototype-based methods maintain representative feature distributions for replay and knowledge retention [40]. Also, Prototype-Guided Memory Replay [41] further improves replay efficiency by using class-level prototypes to select or generate informative samples. However, these approaches are primarily designed for classification tasks, where clear class boundaries exist, and class prototypes can be defined. In contrast, our work focuses on a regression setting for leaf counting, where outputs are continuous and defining class prototypes is not straightforward. Therefore, instead of relying on class-wise representatives, we propose a hybrid uncertainty-diversity sampling strategy, which selects samples based on prediction uncertainty and featurespace diversity. This enables the model to retain both informative and non-redundant samples, making it more suitable for multimodal regression-based CL.

TABLE I: Comparison of existing methods with our proposed approach. DIL refers to domain-incremental learning.
<table><tr><td>Method</td><td>Regression</td><td>CL</td><td>DIL</td><td>Modalities</td><td>Application</td><td>Field Data</td><td>Dataset / Remarks</td></tr><tr><td>Giuffrida et al. [19]</td><td>√</td><td>x</td><td>x</td><td>RGB</td><td>Leaf Count</td><td>√</td><td>CVPPP</td></tr><tr><td>Dobrescu et al. [7]</td><td>√</td><td>x</td><td>x</td><td>RGB, NIR, Fluorescence, Depth</td><td>Leaf Count</td><td>√</td><td>CVPPP + Custom</td></tr><tr><td>Giuffrida et al. [9]</td><td>√</td><td>x</td><td>x</td><td>RGB, NIR, Fluorescence</td><td>Leaf Count</td><td>√</td><td>MSU-PID</td></tr><tr><td>Xu et al. [20]</td><td>x</td><td>x</td><td>x</td><td>RGB</td><td>Segmentation</td><td>√</td><td>DeepLab-like</td></tr><tr><td>Itzhaky et al. [21]</td><td>x</td><td>x</td><td>x</td><td>RGB</td><td>Density Map</td><td>√</td><td>Leaf segmentation</td></tr><tr><td>Tu et al. [22]</td><td>x</td><td>x</td><td>x</td><td>RGB</td><td>Detection</td><td>√</td><td>YOLO-based</td></tr><tr><td>Sun et al. [23]</td><td>x</td><td>√</td><td>x</td><td>RGB, Text, Audio</td><td>Classification</td><td>x</td><td>Modality dropout</td></tr><tr><td>Srinivasan et al. [24]</td><td>x</td><td>√</td><td>x</td><td>Image, Text</td><td>Vision-Language applications</td><td>x</td><td>CLIP-based</td></tr><tr><td>Zhao et al. [25]</td><td>x</td><td>√</td><td>x</td><td>RGB</td><td>Disease Classification</td><td>√</td><td>AgriCL</td></tr><tr><td>MCLC (Ours)</td><td>√</td><td>√</td><td>√</td><td>RGB, Depth, Thermal</td><td>Leaf Count</td><td>√</td><td>MMLC (our dataset)</td></tr></table>

Some recent efforts have extended multimodal CL to other domains. Sun et al. update features and knowledge across tasks, even when modalities are missing [23]. Wang et al. focus on cross-modal retrieval with continual indexing [42], while a general framework has been developed for vision-language tasks [24]. Other studies have explored multimodal CL in robotics [43], activity monitoring [44], [45], and agriculture [46]. In agriculture [47], [48], CL has been applied to plant disease detection [25], [49], [50], plant recognition [51], and stress classification [52], but to the best of our knowledge, CL in leaf counting is still underexplored. Although CL is widely explored in classification, its use in multimodal regression remains underexplored [53]. This work presents the first CL approach for leaf counting, using multimodal data to support continual adaptation in dynamic agricultural environments.

From Table I, we observe that earlier methods mainly used RGB images and focused on static datasets for tasks such as leaf counting or segmentation. Although some included additional modalities such as NIR or fluorescence, none addressed CL, which provides temporal data across tasks. Some recent works have explored CL, but mostly for classification tasks using singlemodality data or non-visual inputs. These are often not designed for plant counting and rarely use real field data. To the best of our knowledge, no prior work has explored leaf counting in a CL setting using multimodal inputs, making our approach a novel contribution that addresses an existing gap in the literature. Our work uses RGB, depth, and thermal modalities for leaf-counting. This addresses both the need for multimodal fusion and continual adaptation in a real-world agricultural setting, enabling sequential learning over time.

## III. PROPOSED DATASET

We introduce a novel MMLC dataset for leaf-counting under a DIS in CL. The dataset contains 6.3K+ images across RGB, depth, and thermal modalities, captured in real agricultural fields at different times of day. The datasets include four vegetable crop types, including capsicum, zucchini, cucumber, and cauliflower, showing diversity in shape and appearance. The samples were collected within the first four weeks of plant growth, a crucial period for early phenotypic analysis. The image triplets, consisting of RGB, depth, and thermal images, were captured with a Google Pixel 5 smartphone camera, an Intel® RealSense™ D435 sensor, and a Fluke TiX580 infrared sensor, respectively. All plant images were acquired from a consistent top-down perspective at approximately 1 meter to ensure uniformity in data collection. To incorporate natural environmental variation, the plants were captured twice a day, once in the morning and once in the evening, enabling the dataset to reflect changes in illumination and temperature throughout the day.

![](images/6ec745aad0c4c8a4e2a8dd9344f95c6b9ca2fc33bf498156e1c5e8caa5499dea.jpg)  
Fig. 1: Overview of the proposed MMLC dataset. Top: It shows example images (RGB, depth, and thermal) from the 8 splits created from the MMLC dataset. Bottom: It presents the order of the splits in our 4 crops created for a CL setting. The three crop sequences CS1, CS2, and CS3 are formed by following an ordering by crop type, capture time, and mixed, respectively.

To enable CL benchmarking and evaluation on our MMLC dataset, we adopt a DIS in which tasks within a given sequence exhibit potential data distribution shifts, also termed ”domain shifts” [54]. In the MMLC dataset, the variation in data arises either due to changes in the crop species (capsicum, zucchini, cucumber, and cauliflower) or the lighting/thermal conditions based on the time of capture (morning or evening). We propose a partition of the MMLC dataset into eight splits (S1, . . . , S8), treated as eight tasks. Specifically, we group samples by crop type and capture time. Thus, we obtain a total of 8 tasks across 4 crop types and 2 capture times. A few samples from each of the splits are shown in Figure 1. Other splits, e.g., grouping samples only by crop type or capture time, can be used to simulate different DIS.

For an exhaustive CL benchmarking, we provide three possible orderings of the eight splits, termed CS1, CS2, and CS3, from MMCL as discussed below:

CS1 (Crop-wise): In this crop sequence, tasks are organized based on crop type. The first two tasks include images of capsicum for the morning and evening, respectively. Tasks 3 and 4 contain morning and evening images of a cucumber; tasks 5 and 6 contain morning and evening images of a zucchini; and tasks 7 and 8 contain morning and evening images of a cauliflower. This setup ensures the model is exposed to one crop at a time across different times of the day before moving to the next crop.

CS2 (Time-wise): Here, tasks are arranged based on the time of image capture. Tasks 1 to 4 consist of morning images of capsicum, cucumber, zucchini, and cauliflower, respectively. Tasks 5 to 8, then present the evening images of these same crops in the same order. This setup allows the model to first learn representations influenced by time of day, then generalize across crops. CS3 (Mixed): This sequence introduces a mix of crop types and capture times in a non-linear order. Task 1 contains morning images of capsicum; task 2 has evening images of cucumber; task 3 introduces a new crop, zucchini, in the morning; and task 4 has another new crop, cauliflower, in the evening. Tasks 5 to 8 revisit the previous crops at different times: morning images of cucumber, evening images of capsicum, morning images of cauliflower, and evening images of zucchini. This sequence mimics a more realistic and unpredictable data stream.

## IV. METHODOLOGY

We address the challenge of leaf counting in realworld agricultural environments, where non-stationary data distributions arise from changing environmental and plant conditions. To ensure robust and consistent performance in such dynamic scenarios, we propose an MCLC framework illustrated in Figure 2 that combines multimodal feature fusion with a CL strategy. In the following subsections, we formally define the problem, describe the proposed MCLC pipeline, and introduce the CL mechanism designed to mitigate catastrophic forgetting. The complete training procedure is summarized in Algorithm 1.

Algorithm 1 Multimodal Continual Learning   
1: Input: Tasks $\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { n } \}$ with data volume   
$\{ \mathcal { D } _ { 1 } , . . . , \mathcal { D } _ { n } \} ;$ buffer size $B$   
2: Output: Final model $\mathcal { M } _ { n }$ trained with replay   
3: Initialize memory buffer $B _ { 0 } \gets \emptyset$   
4: Initialize model $\mathcal { M } _ { 0 }$   
5: for $t = 1$ to n do   
6: Step 1: Train model on current task with replay   
and distillation   
7: for each mini-batch $( x , y ) \sim \mathcal { D } _ { t }$ do   
8: Compute prediction $\hat { y } \gets \mathcal { M } _ { t } ( x )$   
9: Compute task loss ${ \boldsymbol { \ell } } _ { t } \gets \ell ( \hat { y } , y )$   
10: Replay: Sample $( x _ { b } , f _ { b } , y _ { b } )$ from $B _ { t - 1 }$   
11: $\ell _ { \mathrm { r e p l a y } } \gets \ell ( \mathcal { M } _ { t } ( x _ { b } ) , y _ { b } ) + \ell ( \mathcal { R } ( f _ { b } ) , y _ { b } )$   
12: $\mathbf { i f } \ t > 1$ then   
13: Compute distillation loss:   
14: $\ell _ { \mathrm { d i s t i l } } \bar { \cdot }  \| \phi _ { t } ( x ) - \phi _ { t - 1 } ( x ) \| _ { 2 } ^ { 2 } + \| \mathcal { M } _ { t } ( x ) -$   
$\mathcal { M } _ { t - 1 } ( x ) \vert \vert _ { 2 } ^ { 2 }$   
15: end if   
16: Update model using:   
17: $\ell \gets \ell _ { t } + \lambda _ { 1 } \ell _ { \mathrm { r e p l a y } } + \lambda _ { 2 } \ell _ { \mathrm { d i s t i l l } }$   
18: end for   
19: Step 2: Update memory buffer   
20: for each $( x _ { t } ^ { i } , y _ { t } ^ { i } ) \in \mathcal { D } _ { t }$ do   
21: Compute uncertainty via MC Dropout:   
22: $\mathcal { U } ( x _ { t } ^ { i } ) \gets \mathrm { V a r } ( \{ \mathcal { M } _ { t } ^ { ( k ) } ( x _ { t } ^ { i } ) \} _ { k = 1 } ^ { T } )$   
23: Compute feature embedding $\hat { f } _ { t } ^ { i } \gets \phi ( x _ { t } ^ { i } )$   
24: Compute diversity:   
25: $\begin{array} { r } { \delta ( x _ { t } ^ { i } )  \operatorname* { m i n } _ { x ^ { \prime } \in \mathcal { B } _ { t - 1 } } \| f _ { t } ^ { i } - \phi ( x ^ { \prime } ) \| _ { 2 } } \end{array}$   
26: Compute score:   
27: $S ( x _ { t } ^ { i } ) \gets \alpha \mathcal { U } ( x _ { t } ^ { i } ) + \beta \delta ( x _ { t } ^ { i } )$   
28: end for   
29: Select top-B samples based on $S ( x )$ to form $B _ { t }$   
30: Store $( x , f , y )$ in $B _ { t }$   
31: Freeze current model: $\mathcal { M } _ { t - 1 }  \mathcal { M } _ { t }$   
32: end for   
33: Return: Final model $\mathcal { M } _ { n }$

## A. Problem Formulation

We consider a DIS in which a model is sequentially trained on different tasks (or episodes), each of which may involve domain shifts. Let there be n tasks denoted as $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \ldots , \mathcal { T } _ { n } \}$ , where each task $\mathcal { T } _ { t }$ corresponds to the $t ^ { t h }$ domain, i.e., a combination of a plant species and imaging conditions in our setting. Each task is associated with a data volume $\mathcal D _ { t } ~ = ~ \bar { \{ } (  x _ { t } ^ { i } , y _ { t } ^ { i } ) \} _ { i = 1 } ^ { N _ { t } }$ , where $N _ { t }$ is the total number of samples in $\mathcal { D } _ { t } , \ x _ { t } ^ { i }$ represents a multimodal input and $y _ { t } ^ { i }$ is the corresponding leaf count. Each input $\boldsymbol { x } _ { t } ^ { i }$ comprises three modalities: RGB, thermal, and depth. Specifically, we denote

$x _ { t } ^ { i } = \left\{ x _ { t } ^ { i , m } \ | \ m \in \left\{ \mathrm { R G B } , \ \mathrm { T h e r m a l } , \ \mathrm { D e p t h } \right\} \right\} \in \mathcal { X } _ { t } ,$ where $x _ { t } ^ { i , \mathrm { R G B } } \in \mathbb { R } ^ { H \times W \times 3 }$ is a 3-channel image, and $\boldsymbol { x } _ { t } ^ { i , \mathrm { T h e r m a l } ^ { \star } } , \ \boldsymbol { x } _ { t } ^ { i , \mathrm { D e p t h } } \in \mathbb { R } ^ { H \times W }$ are single-channel images. Here, H and $W$ denote the image height and width, respectively. Each ground truth label $y _ { t } ^ { i } ~ \in ~ \mathbb { R }$ . For simplicity, we will refer to the multimodal input as $\ v x _ { t } ^ { i }$ throughout the rest of the paper, with the understanding that it implicitly includes all three modalities: RGB, thermal, and depth, unless explicitly stated otherwise.

Ideally, the goal is to learn a model $\mathcal { M } _ { \theta }$ , parameterized by θ, that minimizes the cumulative loss across all tasks in the sequence $( \{ T _ { 1 } , T _ { 2 } , \dots , T _ { n } \} )$ for leaf counting (Eq. 1).

$$
\arg \operatorname* { m i n } _ { \theta } \sum _ { t = 1 } ^ { n } \sum _ { ( x _ { t } ^ { i } , y _ { t } ^ { i } ) \in \mathcal { D } _ { t } } \ell ( \mathcal { M } _ { \theta } ( x _ { t } ^ { i } ) , y _ { t } ^ { i } )\tag{1}
$$

where ℓ denotes the regression loss function, which we define as Mean Squared Error (MSE) in our setting.

This objective cannot be directly optimized, as the model has access only to the current data volume $\mathcal { D } _ { t }$ at the $t ^ { t h }$ training session. So, our objective is to model each task $\mathcal { T } _ { t }$ in sequence using a deep regression model $\mathcal { M } _ { t }$ with parameters $\theta _ { t } .$ , such that performance on previous tasks is preserved. Since previous task data is not retained, we store a subset of samples in a buffer $B _ { t }$ during training. This buffer enables each new model $\mathcal { M } _ { t + 1 }$ to retain knowledge of prior tasks $\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { t } \}$ . This buffer $B _ { t }$ is constructed by sampling from the current data volume $\mathcal { D } _ { t }$ based on a hybrid strategy combining uncertainty and diversity (see Sec. IV-C). Specifically, we compute uncertainty using multiple stochastic forward passes with dropout activated, which captures prediction variability. Further, instead of storing only raw samples, we store both input samples and their corresponding fused feature representations. The selected samples are then added to the buffer and used in future training to mitigate forgetting.

At each task $T _ { t } ,$ , we train the model $\mathcal { M } _ { t }$ on bufferaware data volume $\widetilde { \mathcal { D } } _ { t }$ , the union of the current data volume and buffered samples from previous tasks as given in (Eq. 2).

$$
\mathcal { \tilde { D } } _ { t } = \mathcal { D } _ { t } \cup \mathcal { B } _ { t - 1 }\tag{2}
$$

![](images/fc395e0ae246bbe76c3d61001401218e177b8cadc24d6b10ef8bdbc671c31db4.jpg)  
Fig. 2: Overview of Multimodal Continual Leaf Counting (MCLC) framework. It processes paired RGB, depth, and thermal images to predict leaf count. Features from each modality are extracted using a modality encoder block, adaptively weighted via modality attention, fused using cross-attention, and passed to an MLP regressor for prediction. To mitigate catastrophic forgetting, the model is trained on the current task $T _ { t }$ and on memory samples from a buffer $B _ { t - 1 }$ that stores input samples, feature representations, and labels. After each training session, samples from $T _ { t }$ are evaluated using a unified uncertainty-diversity score, where uncertainty is estimated via Monte Carlo dropout and diversity via feature embedding distance. The top-B samples are stored in the buffer for future training. Additionally, knowledge from previous tasks is preserved using distillation between $M _ { t - 1 }$ and $M _ { t }$

The learning objective is to minimize the MSE for each task T<sub>t</sub> (Eq. 3)

$$
\ell _ { t } = \frac { 1 } { | \widetilde { \mathcal { D } } _ { t } | } \sum _ { ( x , y ) \in \widetilde { \mathcal { D } } _ { t } } \left( \mathcal { M } _ { t } ( x ; \theta _ { t } ) - y \right) ^ { 2 }\tag{3}
$$

where $\ell _ { t }$ denotes the regression loss for task $T _ { t } ,$ , implemented using Smooth L1 loss for improved stability against outliers. While only the current data volume $\mathcal { D } _ { t }$ and buffered samples from $\boldsymbol { B } _ { t - 1 }$ are used for training at each step, we evaluate the final model $\mathcal { M } _ { n }$ on all task data volumes $\{ \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { n } \}$ to assess overall continual performance.

## B. Multimodal Leaf Counting

Our MCLC framework comprises four main components: a modality-encoder block (MEB), a modality attention block, a cross-attention fusion module, and a regression head. The architecture processes multimodal inputs to predict a single leaf count, as summarized in Algorithm 2.

Modality Encoder Block: Each modality, RGB, depth, and thermal, is processed by a dedicated modality encoder (ME) based on the ResNet-50 architecture. Let the input image for modality $m \in$ {RGB, Depth, Thermal} be denoted as $\boldsymbol { x } _ { t } ^ { i , m }$

We define the modality encoder $f _ { m } ( \cdot )$ and a projection head $g _ { m } ( \cdot )$ for each modality such that:

$$
z _ { t } ^ { i , m } = g _ { m } \left( f _ { m } \left( x _ { t } ^ { i , m } \right) \right) , \quad z _ { t } ^ { i , m } \in \mathbb { R } ^ { 5 1 2 }\tag{4}
$$

where: $f _ { m } ( \cdot )$ is the modality-specific encoder (ResNet-50-based), and $g _ { m } ( \cdot )$ is a projection head comprising a flattening layer and a fully connected layer that projects the 2048-dimensional feature to a 512-dimensional embedding, with ReLU activation and dropout regularization. $z _ { t } ^ { \bar { i } , m }$ is the resulting 512- dimensional embedding for the modality m of the sample i from the task $\mathcal { T } _ { t }$

Modality Attention: Before fusion, we introduce a modality attention mechanism to adaptively weight the contribution of each modality. Given embeddings $z _ { t } ^ { i , \mathrm { R G B } } , z _ { t } ^ { i , \mathrm { D e p t h } } , z _ { t } ^ { i , \mathrm { T h e r m a l } }$ , we compute attention weights as:

$$
\begin{array} { r } { \left[ w _ { \mathrm { R G B } } , w _ { \mathrm { D e p t h } } , w _ { \mathrm { T h e r m a l } } \right] = \mathrm { S o f t m a x } \Big ( h ( [ z _ { t } ^ { i , \mathrm { R G B } } , } \\ { z _ { t } ^ { i , \mathrm { D e p t h } } , z _ { t } ^ { i , \mathrm { T h e r m a l } } ] ) \Big ) } \end{array}\tag{5}
$$

There $h ( \cdot )$ is a two-layer MLP followed by Softmax to generate normalized attention weights for the RGB, depth, and thermal modalities. The re-weighted embeddings are:

$$
\tilde { z } _ { t } ^ { i , m } = w _ { m } \cdot z _ { t } ^ { i , m }\tag{6}
$$

This allows the model to dynamically focus on more informative modalities.

Cross-Attention Fusion: Different modality embeddings are fused using a multi-head cross-attention mechanism. The RGB embedding serves as the query (Q), while the depth and thermal embeddings act as keys (K) and values (V). We stack the depth and thermal embeddings to form the K and V matrices:

$$
K = V = \operatorname { S t a c k } \left( z _ { t } ^ { i , \mathrm { D e p t h } } , \ z _ { t } ^ { i , \mathrm { T h e r m a l } } \right) \in \mathbb { R } ^ { 2 \times 5 1 2 }\tag{7}
$$

The cross-attention module (CA) computes the fused representation as:

$$
\phi ( x _ { t } ^ { i } ) = z _ { t } ^ { i } = \mathbf { C A } \left( Q = z _ { t } ^ { i , \mathrm { R G B } } , \ K , \ V \right) \in \mathbb { R } ^ { 1 \times 5 1 2 }\tag{8}
$$

where CA denotes a multi-head cross-attention mechanism with 4 attention heads operating on 512- dimensional modality embeddings and $\phi ( \cdot )$ denotes the fused feature extractor for input $x _ { t } ^ { i } .$ The output $z _ { t } ^ { i }$ is a fused 512-dimensional feature vector containing integrated multimodal information.

Regression Head: The fused features $z _ { t } ^ { i }$ are passed through a multi-layer perceptron to predict the leaf count as a single scalar value $\hat { y } _ { t } ^ { i }$

## C. Continual Learning Strategy

To mitigate catastrophic forgetting, we devise a novel CL strategy in our MCLC framework. The key components are:

Experience Replay (ER): To mimic the past tasks, a global buffer is maintained with a fixed allowed size, B. The model is trained not only with the new data volume $\mathcal { D } _ { t }$ but also with the buffer $\boldsymbol { B } _ { t - 1 }$ which contains a few exemplars from already seen tasks $\{ \mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { t } \}$ . In each training batch, we randomly sample a subset of stored samples from $B _ { t - 1 }$ . The replay mechanism operates at two levels: (1) Feature-level replay, where stored fused features are directly used to train the regression head, and (2) Input-level replay, where stored raw samples are passed through the full model.

The combined replay loss is defined as:

$$
\ell _ { \mathrm { r e p l a y } } = \lambda _ { 1 } \cdot \ell ( \mathcal { R } ( f ) , y ) + \lambda _ { 2 } \cdot \ell ( \mathcal { M } _ { t } ( x ) , y )\tag{9}
$$

Algorithm 2 Multimodal Leaf Counting $\mathcal { M } _ { t }$   
1: Input: Multimodal sample   
${ x _ { t } ^ { i } } ^ { \star } = \{ x _ { t } ^ { i , \mathrm { R G B } } , x _ { t } ^ { i , \mathrm { D e p t h } } , x _ { t } ^ { i , \mathrm { T h e r m a l } } \} \in \widetilde { \mathcal { D } } _ { t }$   
2: Step 1: Modality Encoding   
3: for each modality $m \in \{ \mathbf { R } \mathbf { G } \mathbf { B } \colon$ , Depth, Thermal} do   
4: $z _ { t } ^ { i , m } \gets g _ { m } ( f _ { m } ( x _ { t } ^ { i , m } ) ) \in \mathbb { R } ^ { 5 1 2 }$   
5: end for   
6: Step 2: Modality Attention   
7: [w<sub>RGB</sub>, w<sub>Depth</sub>, w<sub>Thermal</sub>] ←   
Softma $\bar { \iota } \big ( h ( [ z _ { t \mathrm { ~ . ~ } } ^ { i , \mathrm { R G B } } , z _ { t } ^ { i , \mathrm { D e p t h } } , z _ { t } ^ { i , \mathrm { T h e r m a l } } ] ) \big )$   
8: $\tilde { z } _ { t } ^ { i , m } \gets \dot { w } _ { m } \cdot z _ { t } ^ { i , m }$   
9: Step 3: Cross-Attention Fusion   
10: $Q \doteq \tilde { z } _ { t } ^ { i , \mathrm { R G B } }$   
11: $\overset { \vartriangle } { \boldsymbol { K } } \gets \dot { \operatorname { S t a c k } } ( \tilde { z } _ { t } ^ { i , \mathrm { D e p t h } } , \tilde { z } _ { t } ^ { i , \mathrm { T h e r m a l } } )$   
12: $V  K$   
13: $z _ { t } ^ { i } \gets \mathbf { C A } ( Q , K , V )$   
14: Step 4: Regression   
15: $\hat { y } _ { t } ^ { i } \gets \mathrm { R e g H e a d } ( z _ { t } ^ { i } )$   
16: Output: Predicted leaf count $\hat { y } _ { t } ^ { i }$

where $f$ denotes stored fused features, R is the regression head, and $\lambda _ { 1 } , \lambda _ { 2 }$ are weighting factors. In our implementation, the weighting factors are set to $\lambda _ { 1 } = 0 . 3$ and $\lambda _ { 2 } ~ = ~ 0 . 7$ for feature-level and input-level replay, respectively.

Uncertainty-Diversity Sampling: When adding new samples to the buffer, we propose to prioritize those with high uncertainty and diversity. We measure uncertainty $( \mathcal { U } ( x _ { t } ^ { i } ) )$ using Monte Carlo dropout, which captures prediction variability under stochastic forward passes. Specifically, we perform T forward passes with dropout enabled and compute the variance of predictions:

$$
\mathcal { U } ( x _ { t } ^ { i } ) = \mathrm { V a r } \left( \{ \mathcal { M } _ { t } ^ { ( k ) } ( x _ { t } ^ { i } ) \} _ { k = 1 } ^ { T } \right)\tag{10}
$$

where $\mathcal { M } _ { { t } } ^ { ( k ) }$ denotes the model with dropout active during the $k ^ { \mathit { i h } }$ forward pass. In our implementation, uncertainty is estimated using 5 stochastic forward passes with dropout enabled during inference.

Further, instead of a strict two-stage filtering, we compute a unified score that jointly considers uncertainty and diversity. For each candidate sample $( x _ { t } ^ { i } , y _ { t } ^ { i } )$ , we compute a diversity score using Minimum Embedding Distance (MED) with respect to the existing buffer:

$$
\delta ( x _ { t } ^ { i } ) = \operatorname* { m i n } _ { x ^ { \prime } \in \mathcal { B } _ { t - 1 } } \big \| \phi ( x _ { t } ^ { i } ) - \phi ( x ^ { \prime } ) \big \| _ { 2 }\tag{11}
$$

The final selection score is computed as:

$$
S ( x _ { t } ^ { i } ) = \alpha \cdot \mathcal { U } ( x _ { t } ^ { i } ) + \beta \cdot \delta ( x _ { t } ^ { i } )\tag{12}
$$

where α and $\beta$ are weighting factors. The top B samples with the highest scores are selected to update the buffer:

$$
B _ { t } = \underset { x \in \widetilde { D } _ { t } } { \arg \operatorname* { m a x } } S ( x )\tag{13}
$$

The buffer maintains a fixed memory and is updated after each task by retaining samples with the highest combined uncertainty-diversity scores. This unified score ensures that selected samples are both uncertain (informative) and diverse (less redundant w.r.t. past memory), thereby improving the quality of the buffer in CL.

Knowledge Distillation: To further preserve knowledge from previous tasks, we incorporate a distillation mechanism. After each task, a frozen copy of the previous model $\mathcal { M } _ { t - 1 }$ is maintained. During training of $\mathcal { M } _ { t } ,$ we enforce consistency between the current and previous model in both feature and output space:

$$
\ell _ { \mathrm { d i s t i l } } = \| \phi _ { t } ( x ) - \phi _ { t - 1 } ( x ) \| _ { 2 } ^ { 2 } + \| \mathcal { M } _ { t } ( x ) - \mathcal { M } _ { t - 1 } ( x ) \| _ { 2 } ^ { 2 }\tag{14}
$$

This helps stabilize representations and reduce catastrophic forgetting.

The overall training objective combines current task loss, replay loss, and distillation loss:

$$
\ell = \ell _ { t } + \ell _ { \mathrm { r e p l a y } } + \ell _ { \mathrm { d i s t i l l } }\tag{15}
$$

## V. EXPERIMENTS AND RESULTS

In this section, we describe the experimental setup, the compared methods, the evaluation metrics, and the computational complexity. We then provide a quantitative and qualitative analysis of the proposed model’s performance relative to baseline methods and state-ofthe-art approaches.

## A. Experimental Setup

We evaluate our proposed method, MCLC, in a CL setting for multimodal leaf counting. Each task contains images from all three modalities, with the objective of predicting the leaf count.

Training and implementation details. All methods are trained sequentially across tasks using Smooth L1 and Mean Squared Error (MSE) losses for leaf counting. The buffer is used to retain the representative samples selected using the hybrid uncertainty-diversity criterion. To ensure fairness and a statistically reliable comparison, all experiments are repeated using three random seeds (42, 123, and 999). The reported results correspond to the mean and standard deviation across these three runs. The additional implementation details of the MCLC are given in Table II.

Datasets Used. We evaluate our approach and existing works using three datasets: MMLC, MSU-PID, and

TABLE II: Implementation details
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Backbone Network</td><td>ResNet-50</td></tr><tr><td>Embedding Dimension</td><td>512</td></tr><tr><td>Attention Heads</td><td>4</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning Rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Batch Size</td><td>8</td></tr><tr><td>Training Epochs</td><td>30</td></tr><tr><td>Dropout Rate</td><td>0.3</td></tr><tr><td>Buffer Size</td><td>{30,40,50,80}</td></tr><tr><td>MC Dropout Passes</td><td>5</td></tr><tr><td>Replay Weights</td><td>{0.3, 0.7}</td></tr><tr><td>Distillation Weight</td><td>0.05</td></tr><tr><td>Random Seed</td><td>{42, 123, 999}</td></tr><tr><td>GPU</td><td>NVIDIA H100</td></tr></table>

CVPPP. The MMLC dataset is our proposed benchmark for multimodal CL in leaf counting and has been described in detail in section III. It comprises RGB, depth, and thermal modalities. For experimental evaluation, we consider three dataset sequences, CS1, CS2, and CS3, to simulate different domain incremental CL settings. We consider the MSU-PID dataset [10], a publicly available multimodal leaf-counting dataset comprising RGB, thermal, and fluorescence images. Since it was not originally designed for CL evaluation, we adapted it to the CL setting by creating six sequential splits. MSU-PID contains approximately 1,000 samples from two crops (Arabidopsis and Beans), captured hourly over a period of 16 days. It serves as an additional benchmark for evaluating multimodal leaf-counting performance. Similar to MMLC, we adopt an 80:20 train:test division for each split. Additionally, we include the unimodal publicly available CVPPP dataset [55], a widely used benchmark for leaf counting based on RGB images of Arabidopsis and Tobacco plants. We used CVPPP to evaluate different leaf-counting paradigms. The detailed statistics and task distributions of the multimodal datasets are reported in Table III.

TABLE III: Train-test details of the splits curated from the MSU-PID [10] and the MMLC datasets for CL experiments.
<table><tr><td></td><td>Splits</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td><td>S5</td><td>S6</td><td>S7</td><td>S8</td></tr><tr><td>MMIC</td><td>Train Test</td><td>533 133</td><td>664 166</td><td>637 158</td><td>609</td><td>617</td><td>663</td><td>590</td><td>758</td></tr><tr><td>MSSU-UPPID</td><td>Train</td><td>87</td><td>88</td><td>88</td><td>151 88</td><td>153 88</td><td>165 85</td><td>146</td><td>187 一</td></tr><tr><td></td><td>Test</td><td>38</td><td>38</td><td>38</td><td>38</td><td>38</td><td>37</td><td></td><td>一</td></tr></table>

## B. Comparable Methods

We compare two types of baselines, rehearsal-based and regularization-based methods, and a recent state-ofthe-art approach. Rehearsal-based methods include ER [37], which stores past examples, and DER++ [56], which extends ER by enforcing consistency in predictions across time. Both are evaluated with memory sizes of {30, 40, 50, 80}. The GEM [13] and $\mathbf { A } \mathbf { - }$ GEM [36] control interference by projecting gradients using stored samples per task. We experiment with buffer sizes {4, 5, 6, 10} per task for GEM and A-GEM. Regularization-based methods include EWC [12], which constrains parameter updates to those important for previous tasks. These are evaluated with regularization coefficients $\lambda \in \{ 1 , 2 , 3 \}$ . We also include the recent state-of-the-art method AVQACL [57], which performs CL through adaptive quantization and contrastive learning and is evaluated with buffer sizes of {30, 40, 50, 80}. In addition, we report results for three non-continual baselines: a naive model trained sequentially without any forgetting mitigation, which is considered a lower bound; cumulative training, where a model is retrained on all data seen so far; and joint training on the union of all task data, both of which are considered an upper bound.

TABLE IV: Train-test performance matrix for $n = 6$
<table><tr><td>Train\Test</td><td> $\mathbf { T e _ { 1 } }$ </td><td> $\mathbf { T e _ { 2 } }$ </td><td> $\mathbf { T e _ { 3 } }$ </td><td> $\mathbf { T e _ { 4 } }$ </td><td> $\mathbf { T e } _ { 5 }$ </td><td> $\bf { T e _ { 6 } }$ </td></tr><tr><td> $\mathbf { T r _ { 1 } }$ </td><td> $R _ { 1 , 1 }$ </td><td> $R _ { 1 , 2 }$ </td><td> $R _ { 1 , 3 }$ </td><td> $R _ { 1 , 4 }$ </td><td> $R _ { 1 , 5 }$ </td><td> $R _ { 1 , 6 }$ </td></tr><tr><td> $\mathbf { T r } _ { 2 }$ </td><td> $R _ { 2 , 1 }$ </td><td> $R _ { 2 , 2 }$ </td><td> $R _ { 2 , 3 }$ </td><td> $R _ { 2 , 4 }$ </td><td> $R _ { 2 , 5 }$ </td><td> $R _ { 2 , 6 }$ </td></tr><tr><td> $\mathbf { T r _ { 3 } }$ </td><td> $R _ { 3 , 1 }$ </td><td> $R _ { 3 , 2 }$ </td><td> $R _ { 3 , 3 }$ </td><td> $R _ { 3 , 4 }$ </td><td> $R _ { 3 , 5 }$ </td><td> $R _ { 3 , 6 }$ </td></tr><tr><td> $\mathbf { T r _ { 4 } }$ </td><td> $R _ { 4 , 1 }$ </td><td> $R _ { 4 , 2 }$ </td><td> $R _ { 4 , 3 }$ </td><td> $R _ { 4 , 4 }$ </td><td> $R _ { 4 , 5 }$ </td><td> $R _ { 4 , 6 }$ </td></tr><tr><td> $\mathbf { T r } _ { \mathbf { 5 } }$ </td><td> $R _ { 5 , 1 }$ </td><td> $R _ { 5 , 2 }$ </td><td> $R _ { 5 , 3 }$ </td><td> $R _ { 5 , 4 }$ </td><td> $R _ { 5 , 5 }$ </td><td> $R _ { 5 , 6 }$ </td></tr><tr><td> $\mathbf { T r _ { 6 } }$ </td><td> $R _ { 6 , 1 }$ </td><td> $R _ { 6 , 2 }$ </td><td> $R _ { 6 , 3 }$ </td><td> $R _ { 6 , 4 }$ </td><td> $R _ { 6 , 5 }$ </td><td> $R _ { 6 , 6 }$ </td></tr></table>

## C. Evaluation metrics

After training on the $t ^ { \mathrm { { t h } } }$ task using training data $T r _ { t } ,$ we evaluate the model on all test data $T e _ { 1 } , T e _ { 2 } , \ldots , T e _ { n }$ . This results in a train-test matrix $R \in \mathbb { R } ^ { n \times n }$ , where $R _ { t , j }$ denotes the MSE on the test task j after training on tasks up to t. The sample train-test matrix is shown in the Table IV. This matrix captures the stability-plasticity trade-off [58] between the model’s ability to acquire new knowledge (plasticity) and its ability to retain prior knowledge (stability).

To evaluate performance, we used standard CL metrics commonly used in the literature. For overall performance, we compute the average mean squared error (AMSE) across all tasks after completing the final task T based on [13]:

$$
\mathrm { A M S E } = \frac { 1 } { T } \sum _ { j = 1 } ^ { T } R _ { T , j }\tag{16}
$$

Further, Backward Transfer (BWT) and Forward Transfer (FWT) [59] are used to evaluate forgetting and knowledge transfer. The lower the value, the better the model’s performance.

## D. Computational Complexity and Scalability

The proposed MCLC framework enables efficient sequential learning without retraining on all previously seen data. Instead, a bounded memory buffer is used to preserve representative samples from earlier tasks, thereby reducing storage and retraining requirements during continual adaptation. Using a memory buffer size of 80, the framework required an average training time of 13.06 minutes per task with a peak GPU memory usage of 3.75 GB. During inference, the model required only 10.68 ms per sample using a single forward pass. In addition, the modality attention and cross-attention modules introduced relatively low computational overhead compared to the backbone feature extractors. These characteristics show that MCLC can efficiently adapt to new tasks while maintaining low training, memory, and inference overhead, making it suitable for practical multimodal continual learning scenarios.

## E. Quantitative result analysis

We experimented with two multimodal datasets, namely MMLC and MSU-PID, and compared the MCLC model against baselines and state-of-the-art methods. Table V shows the performance of the methods on the best-performing hyperparameters, λ and buffer size on MCLC. Among all CL methods, the proposed MCLC consistently achieves the best performance across all three sequences. Specifically, MCLC obtains the lowest average AMSE of 0.675±0.027, 0.542±0.069, and $0 . 7 4 5 { \scriptstyle \pm 0 . 0 5 7 }$ for CS1, CS2, and CS3, respectively. It also achieves the lowest BWT values of 0.280±0.115, 0.145±0.116, and 0.055±0.038, along with the lowest FWT values of 0.661±0.104, 0.644±0.037, and 0.594±0.050 for the respective task sequences, indicating improved knowledge retention and forward transfer. Among the CL approaches compared, ER achieves the second-best performance in most cases, followed by A-GEM. In contrast, EWC, DER++, and AVQACL produce comparatively higher AMSE, BWT, and FWT values. Although rehearsal-based methods generally outperform regularization-based approaches such as EWC, DER++ performs poorly in our experiments. This is because it combines memory with logit distillation, which is designed for classification. When applied to regressionbased leaf counting, distilling continuous outputs propagates prediction errors across tasks, resulting in error accumulation and reduced stability.

TABLE V: Performance comparison of CL and non-CL methods across three MMLC dataset sequences (CS1, CS2, and CS3). For each method, the best-performing hyperparameters (Param.) are selected based on the lowest AMSE. Best and second-best CL performances are Bold and Underlined respectively.
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="4">CS1</td><td colspan="4">CS2</td><td colspan="4">CS3</td></tr><tr><td>Param.</td><td>BWT↓</td><td>FWT↓</td><td>AMSE↓</td><td>Parameters</td><td>BWT↓</td><td>FWT↓</td><td>AMSE↓</td><td>Parameters</td><td>BWT↓</td><td>FWT↓</td><td>AMSE↓</td></tr><tr><td>EWC</td><td></td><td>λ = 1</td><td>2.058±0.447</td><td>2.402±0.602</td><td>3.110±1.626</td><td>λ = 3</td><td>2.845±1.244</td><td>3.255±1.356</td><td>4.905±5.179</td><td>λ = 2</td><td>18.935±29.857</td><td>19.310±30.015</td><td>2.328±0.237</td></tr><tr><td>GEM</td><td></td><td>B=10x8</td><td>0.647±0.084</td><td>1.130±0.080</td><td>1.160±0.063</td><td>B=10x8</td><td>2.917±3.656</td><td>1.398±0.199</td><td>1.107±0.054</td><td>B=10x8</td><td>0.680±0.159</td><td>1.095±0.068</td><td>1.167±0.291</td></tr><tr><td>A-GEM</td><td></td><td>B=10x8</td><td>0.733±0.217</td><td>1.111±0.029</td><td>0.954±0.061</td><td>B=6x8</td><td>1.057±0.106</td><td>1.409±0.103</td><td>1.055±0.071</td><td>B=10x8</td><td>0.676±0.056</td><td>1.076±0.050</td><td>1.082±0.057</td></tr><tr><td>an DER++</td><td></td><td>B=40</td><td>1.576±1.703</td><td>21.301±1.866</td><td>24.462±2.159</td><td>B=80</td><td>1.108±0.463</td><td>22.119±0.163</td><td>25.157±0.929</td><td>B=50</td><td>1.758±0.786</td><td>21.613±2.484</td><td>26.132±2.456</td></tr><tr><td>ER</td><td></td><td>B=80</td><td>0.565±0.144</td><td>0.826±0.056</td><td>0.843±0.090</td><td>B=80</td><td>0.564±0.069</td><td>0.790±0.031</td><td>0.743±0.049</td><td>B=80</td><td>0.394±0.038</td><td>0.624±0.061</td><td>0.749±0.120</td></tr><tr><td>AVQACL</td><td></td><td>B=50</td><td>1.368±0.300</td><td>2.084±0.281</td><td>2.241±0.114</td><td>B=50</td><td>1.422±0.145</td><td>2.154±0.121</td><td>1.585±0.128</td><td>B=40</td><td>1.325±0.267</td><td>1.979±0.236</td><td>1.561±0.115</td></tr><tr><td>MCLC</td><td></td><td>B=80</td><td>0.280±0.115</td><td>0.661±0.104</td><td>0.675±0.027</td><td>B=80</td><td>0.145±0.116</td><td>0.644±0.037</td><td>0.542±0.069</td><td>B=80</td><td>0.055±0.038</td><td>0.594±0.050</td><td>0.745±0.057</td></tr><tr><td>Naive</td><td></td><td></td><td>1.570±0.510</td><td>1.854±0.474</td><td>1.518±0.303</td><td></td><td>1.966±0.713</td><td>2.428±0.543</td><td>2.674±1.873</td><td></td><td>1.722±0.397</td><td>2.085±0.419</td><td>2.222±0.361</td></tr><tr><td>No-CL</td><td>Cumulative</td><td></td><td></td><td>0.303±0.135</td><td>0.232±0.072</td><td></td><td>0.271±0.254</td><td>0.398±0.227</td><td>0.219±0.066</td><td></td><td>0.090±0.049</td><td>0.153±0.027</td><td>0.157±0.32</td></tr><tr><td>Joint</td><td></td><td>0.128±0.106</td><td></td><td></td><td>0.186±0.025</td><td></td><td></td><td></td><td>0.193±0.015</td><td></td><td></td><td></td><td>0.177±0.019</td></tr></table>

TABLE VI: Performance comparison of CL and non-CL methods on MSU-PID dataset. For each method, the best-performing hyperparameters are selected based on the lowest AMSE. Best and second-best CL performances are Bold and Underlined, respectively.
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="4">MSU-PID</td></tr><tr><td>Parameters</td><td>BWT ↓</td><td>FWT ↓</td><td>AMSE ↓</td></tr><tr><td rowspan="7">C</td><td>EWC [12]</td><td>λ=2</td><td> $1 . 0 6 2 \pm 0 . 5 7 5$ </td><td> $1 . 2 0 3 \pm 0 . 1 0 5$ </td><td> $0 . 8 8 2 \pm 0 . 0 7 4$ </td></tr><tr><td>GEM [13]</td><td> $\scriptstyle B = 1 0 \mathrm { x } 8$ </td><td> $1 . 1 3 4 \pm 0 . 3 5 6$ </td><td> $1 . 1 0 3 \pm 0 . 0 6 7$ </td><td> $0 . 8 6 9 \pm 0 . 0 5 2$ </td></tr><tr><td>A-GEM [36]</td><td>B=10x8</td><td> $\underline { { 1 . 0 7 3 } } \pm 0 . 5 7 4$ </td><td> $1 . 1 1 6 \pm 0 . 0 9 8$ </td><td> $0 . 8 1 3 \pm 0 . 0 8 4$ </td></tr><tr><td>DER++ [56]</td><td>B=80</td><td> $3 . 6 5 6 \pm 1 4 . 3 3 8$ </td><td> $3 6 . 9 0 9 \pm 6 . 8 2 6$ </td><td> $3 7 . 6 1 6 \pm 6 . 9 5 4$ </td></tr><tr><td>ER [37]</td><td>B=80</td><td> $1 . 1 4 4 \pm 0 . 5 9 0$ </td><td> $\underline { { 1 . 0 3 1 \pm 0 . 1 3 3 } }$ </td><td> $\underline { { 0 . 7 5 0 \pm 0 . 1 0 9 } }$ </td></tr><tr><td>AVQACL [57]</td><td>B=80</td><td> $\mathbf { 0 . 0 9 5 \ : \pm { \ : 0 . 2 9 2 } }$ </td><td> $1 . 1 8 3 \pm 0 . 1 8 6$ </td><td> $1 . 0 3 0 \pm 0 . 1 7 0$ </td></tr><tr><td>MCLC (Ours)</td><td>B=80</td><td> $2 . 2 4 4 \pm 0 . 1 4 8$ </td><td> ${ \bf 0 . 9 7 9 \pm 0 . 1 0 4 }$ </td><td> $\mathbf { 0 . 6 6 4 \ : \pm { \ : 0 . 0 7 1 } }$ </td></tr><tr><td rowspan="2">Non--CL</td><td>Naive</td><td>一</td><td> $\overline { { 0 . 9 4 8 \pm 0 . 2 5 5 } }$ </td><td> $\overline { { 1 . 3 3 4 \pm 0 . 1 4 5 } }$ </td><td> $\overline { { 1 . 0 9 7 \pm 0 . 3 2 0 } }$ </td></tr><tr><td>Cumulative</td><td>一</td><td> $0 . 9 9 8 \pm 0 . 3 5 4$ </td><td> $1 . 0 3 2 \pm 0 . 0 4 8$ </td><td> $0 . 7 5 3 \pm 0 . 1 1 8$ </td></tr><tr><td></td><td>Joint</td><td>一</td><td>一</td><td>一</td><td> $0 . 6 4 2 \pm 0 . 1 0 9$ </td></tr></table>

Table VI shows the performance of the methods on the best-performing hyperparameters, λ and buffer size on MSU-PID. Among all CL methods, the MCLC achieves the best overall performance, with the lowest average AMSE of 0.664±0.071 and the lowest FWT of 0.979±0.104, indicating better generalization to new tasks and improved forward knowledge transfer. Although AVQACL achieves the lowest BWT (0.095±0.292), MCLC provides a better balance between knowledge retention and adaptation, resulting in the best overall continual learning performance. ER achieves the second-best performance with an AMSE of 0.750±0.109 and an FWT of 1.031±0.133, demonstrating the effectiveness of rehearsal-based learning. In comparison, EWC, GEM, and A-GEM show higher errors, while DER++ performs significantly worse, as on the MMLC dataset. Among the non-CL methods, the Joint achieves an AMSE of 0.642±0.109, serving as an upper bound because it has access to all training data. Compared with all CL methods, the proposed MCLC achieves the lowest AMSE in the CL setting, demonstrating its robustness and effectiveness on the MSU-PID dataset.

Figure 3 shows the task-wise AMSE of different methods across the three task sequences of the MMLC dataset. Overall, ER, GEM, A-GEM, and MCLC show a gradual reduction in AMSE as more tasks are learned, indicating improved adaptation in the CL setting. Among the CL methods, MCLC consistently maintains one of the lowest AMSE values across almost all tasks and follows a trend close to the upper-bound methods, Joint Training and Cumulative. ER also demonstrates stable performance but generally exhibits slightly higher errors than MCLC. In contrast, AVQACL and Naive show larger fluctuations across tasks, indicating less stable learning under changing task distributions. Similarly, cumulative and joint training methods maintain consistently low error rates because they have access to all previously seen data during training. For visual clarity, EWC and DER++ are omitted because their larger AMSE values compress the remaining curves. Overall, these results show that MCLC achieves stable learning across different task sequences while effectively reducing catastrophic forgetting. A similar trend is observed in Figure 4 on the MSU-PID dataset. Although MCLC starts with a relatively higher AMSE after the first task, its error decreases rapidly and remains consistently low across subsequent tasks. ER also demonstrates stable performance with comparatively low AMSE, whereas AVQACL exhibits larger fluctuations, particularly during the early tasks. GEM and A-GEM perform competitively but generally maintain higher errors than MCLC and ER. As with the MMLC dataset, cumulative and joint training achieve consistently low errors, whereas DER++ is omitted for better visualization. Overall, MCLC demonstrates stable knowledge retention and achieves one of the lowest AMSE values throughout the CL process.

![](images/05613d7e6dcc048ab574dac2db8d36eb0216dbdd4550ed42fedf0b6ce3e39c8e.jpg)  
(a) CS1

![](images/6fcae12f42a50c2c5f3570fd9da2751b6e67b24261f13e7357b53992e3755f6c.jpg)  
(b) CS2

![](images/1d31600495151c813925c366acf850aa7807b298a02a065d40f40f94f5121123.jpg)  
(c) CS3

Fig. 3: AMSE after learning each task for different CL methods under three task sequences of the MMLC dataset.  
![](images/24c397be206908380a80ac64ddf8d2763dc022b4e6fc85a2277735abecc9ce74.jpg)  
Fig. 4: AMSE after learning each task on MSU-PID for different CL methods.

## F. Qualitative result analysis

To provide further insight into the behavior of the proposed MCLC framework, Figure 5 presents representative qualitative examples from the test set, including both successful and failure cases. Each example includes the corresponding RGB, thermal, and depth modalities, along with the ground-truth (GT) leaf count, predicted (Pred.) leaf count, and absolute error (AE). As shown in Figure 5(a), accurate predictions are obtained when the plant is clearly visible and complementary information is available across all three modalities. The RGB image provides clear visual information, while the thermal and depth images offer complementary cues that help the model estimate the leaf count correctly. As a result, the prediction error is negligible. Figure 5(b) shows examples where the prediction error is higher. These images contain small plants, uneven illumination, or low-contrast depth and thermal information, making the leaf structure less distinct. Such conditions can lead to an incorrect estimate of the leaf count. Despite these challenging cases, the proposed framework performs reliably on most test samples.

![](images/e5731e0ce49f6ba09fe31cbfb3527fd40abba3d31a0fb3a724cfcc73245ce720.jpg)  
Fig. 5: Qualitative analysis of the MCLC framework.

## VI. ABLATION STUDY

To comprehensively evaluate the contributions of the various components of our proposed MCLC framework, we conduct a detailed ablation study. This includes (i) comparison among standard leaf counting paradigms to identify the most suitable leaf counting paradigm, (ii) impact of continual multimodal fusion, (iii) performance across different dataset sequences, (iv) effect of memory size and regularization hyperparameters, and (v) performance on other datasets for generalizability.

## A. Choice of Leaf Counting Paradigm

We begin by comparing four standard leaf counting paradigms: segmentation, regression, density estimation, and object detection on the CVPPP [55] dataset in a non-CL setting. It is a benchmark dataset for leaf counting consisting solely of RGB images. As shown in Table VII, regression achieves the lowest Train MSE (0.34) and Test MSE (1.08), significantly outperforming others. Hence, regression is chosen as the core paradigm for predicting leaf count in our domain incremental CL setting.

TABLE VII: Performance evaluation of different existing leaf-counting methods on the CVPPP dataset.
<table><tr><td>Method</td><td>Train MSE ↓</td><td>Test MSE↓</td></tr><tr><td>Segmentation [31]</td><td>0.66</td><td>1.21</td></tr><tr><td>Regression [9]</td><td>0.34</td><td>1.08</td></tr><tr><td>Density estimation [33]</td><td>23.90</td><td>21.43</td></tr><tr><td>Object detection [22]</td><td>16.95</td><td>62.43</td></tr></table>

## B. Effect of Continual Multimodal Fusion

We evaluated different multimodal fusion methods within the proposed MCLC framework under the domain-incremental CL setting. For a fair comparison, all fusion methods were implemented using the same CL framework. Specifically, we compared our multimodal fusion strategy with [60], which performs statistical fusion using the mean and variance of modality-specific features; [61], a recent transformer-based multimodal learning framework that models cross-modal interactions through attention; and [62], a recent parameterefficient multimodal framework that employs dynamic sparse cross-modality fusion for feature integration. As shown in Table VIII, MCLC consistently outperforms, achieving the lowest BWT (0.432), FWT (0.855), and AMSE (0.861). These results demonstrate that MCLC’s continual multimodal fusion strategy is more effective at integrating complementary information from multiple modalities while preserving previously learned knowledge in the CL setting.

TABLE VIII: Comparison of multimodal fusion methods under the CL setting.
<table><tr><td>Method</td><td>BWT↓</td><td>FWT↓</td><td>AMSE↓</td></tr><tr><td>Havaei et al. [60]</td><td>0.86</td><td>2.146</td><td>1.671</td></tr><tr><td>Shah et al. [61]</td><td>1.830</td><td>2.171</td><td>1.669</td></tr><tr><td>Cai et al. [62]</td><td>1.449</td><td>1.632</td><td>1.256</td></tr><tr><td>MCLC (ours)</td><td>0.432</td><td>0.855</td><td>0.8610</td></tr></table>

## C. Impact of Performance across Task Sequences

To understand how task order influences performance, we evaluated existing methods on the MMLC dataset across three task sequences: CS1, CS2, and CS3. As shown in Table IX, the performance of most CL methods varies with task sequences, demonstrating that task ordering has a significant impact on CL.

In CS1, tasks are grouped by crop type, resulting in gradual domain transitions. This enables relatively stable learning for most methods. For example, ER achieves an AMSE of 0.843±0.090, while the MCLC further reduces the error to 0.675±0.027 using a buffer size of 80. Regularization-based EWC, however, exhibits considerably higher errors, with the best AMSE of $3 . 1 1 0 { \pm } 1 . 6 2 6 .$ In CS2, tasks are organized by image capture time, thereby introducing greater variation across crops. This sequence is more challenging for most methods, as reflected in the substantial performance degradation of EWC, whose AMSE increases to $9 . 0 0 2 { \pm } 1 2 . 7 1 5$ even with its bestperforming hyperparameter. ER remains relatively robust with an AMSE of 0.743±0.049, whereas the proposed MCLC achieves the lowest AMSE of 0.542±0.069, together with the lowest BWT (0.145±0.116) and FWT (0.644±0.037), demonstrating improved knowledge retention and forward transfer. In CS3, tasks are arranged in a non-linear mixed order that combines different crops and capture times. This sequence, therefore, introduces abrupt distributional changes. Although this sequence causes noticeable performance fluctuations for several methods, the proposed MCLC continues to achieve the best overall performance with an AMSE of 0.745±0.057, outperforming ER $( 0 . 7 4 9 { \pm } 0 . 1 2 0 )$ and AVQACL (1.561±0.115). In contrast, EWC exhibits extremely high variability, with BWT and FWT values of 18.935±29.857 and $1 9 . 3 1 0 { \scriptstyle \pm 3 0 . 0 1 5 }$ , indicating high sensitivity to task ordering.

From these observations, we conclude that task sequencing significantly influences CL performance. Also, regularization-based methods such as EWC are highly sensitive to task order and distributional shifts. Moreover, rehearsal-based methods provide greater stability and consistently lower errors across sequences. More importantly, the proposed MCLC consistently achieves the lowest AMSE across all three sequences while maintaining low BWT and FWT values, demonstrating strong robustness, effective knowledge retention, and improved generalization under diverse multimodal CL scenarios.

## D. Impact of Buffer and Hyperparameters

Buffer size and hyperparameters have a clear impact on performance across all sequences, as shown in Table IX. Regularization-based methods such as EWC rely on the regularization parameter λ to preserve previously learned knowledge. However, EWC exhibits inconsistent performance across different values of λ, with large variations in both the mean and standard deviation. In particular, increasing λ does not consistently improve performance. This indicates that simply strengthening the regularization term is insufficient to effectively prevent catastrophic forgetting under large distributional changes. For memory-based methods, the buffer memory plays an important role in improving performance. As the buffer memory increases from 4 × 8 to 10 × 8, both GEM and A-GEM generally achieve lower AMSE values. For example, GEM achieves its best performance with a buffer size of 10 × 8, obtaining AMSE values of 1.160±0.063 in CS1, 1.107±0.054 in CS2, and 1.167±0.291 in CS3. Similarly, A-GEM also benefits from larger buffer memory, obtaining its lowest AMSE values of 0.954±0.061, 1.055±0.071, and 1.082±0.057 across CS1, CS2, and CS3, respectively. These results suggest that increasing buffer memory helps reduce forgetting, although the performance improvement gradually saturates.

TABLE IX: Performance comparison of different methods under different hyperparameter (Param.) settings across three task sequences: CS1, CS2, and CS3 of the MMLC dataset.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Param.</td><td colspan="3">CS1</td><td colspan="3">CS2</td><td colspan="3">CS3</td></tr><tr><td>BWT↓</td><td>FWT ↓</td><td>AMSE↓</td><td>BWT ↓</td><td>FWT↓</td><td>AMSE↓</td><td>BWT ↓</td><td>FWT ↓</td><td>AMSE↓</td></tr><tr><td>EWC</td><td>λ = 1</td><td>2.058±0.447</td><td>2.402±0.602</td><td>3.110±1.626</td><td>4.456±4.365</td><td>4.787±4.459</td><td>9.002±12.715</td><td>2.087±0.599</td><td>2.643±0.928</td><td>2.596±0.734</td></tr><tr><td rowspan="3">[12]</td><td>λ = 2</td><td>5.592±6.786</td><td>10.036±13.845</td><td>11.829±17.133</td><td>3.675±2.681</td><td>3.995±2.733</td><td>6.897±8.877</td><td>18.935±29.857</td><td>19.310±30.015</td><td>2.328±0.237</td></tr><tr><td>λ = 3</td><td>2.358±1.026</td><td>2.699±0.810</td><td>4.055±3.835</td><td>2.845±1.244</td><td>3.255±1.356</td><td>4.905±5.179</td><td>177.998±305.667</td><td>178.405±305.771</td><td>2.492±0.424</td></tr><tr><td>B=4×8</td><td>0.794±0.190</td><td>1.466±0.364</td><td>1.305±0.007</td><td>0.691±0.376</td><td>1.567±0.049</td><td>1.295±0.135</td><td>0.662±0.281</td><td>1.186±0.030</td><td>1.200±0.101</td></tr><tr><td rowspan="5">GEM [13]</td><td>B=5×8</td><td>0.714±0.308</td><td>1.231±0.049</td><td>1.221±0.092</td><td>0.573±0.451</td><td>1.706±0.332</td><td>1.399±0.225</td><td>0.613±0.179</td><td>1.213±0.081</td><td>1.202±0.120</td></tr><tr><td>B=6×8</td><td>0.549±0.363</td><td>1.205±0.079</td><td>1.184±0.033</td><td>1.012±0.308</td><td>1.789±0.552</td><td>1.277±0.042</td><td>0.729±0.047</td><td>1.149±0.065</td><td>1.301±0.040</td></tr><tr><td>B=10×8</td><td>0.647±0.084</td><td>1.130±0.080</td><td>1.160±0.063</td><td>2.917±3.656</td><td>1.398±0.199</td><td>1.107±0.054</td><td>0.680±0.159</td><td>1.095±0.068</td><td>1.167±0.291</td></tr><tr><td>B=4×8</td><td>0.769±0.300</td><td>1.295±0.056</td><td>1.071±0.032</td><td>1.252±0.114</td><td>1.577±0.117</td><td>1.130±0.079</td><td>0.835±0.105</td><td>1.288±0.075</td><td>1.301±0.053</td></tr><tr><td>B=5×8</td><td>0.736±0.345</td><td>1.278±0.151</td><td>1.089±0.106</td><td>1.189±0.090</td><td>1.544±0.061</td><td>1.164±0.016</td><td>0.676±0.242</td><td>1.217±0.099</td><td>1.221±0.103</td></tr><tr><td rowspan="5">[36] DER++</td><td>B=6×8</td><td>0.692±0.386</td><td>1.263±0.091</td><td>1.052±0.062</td><td>1.057±0.106</td><td>1.409±0.103</td><td>1.055±0.071</td><td>0.774±0.078</td><td>1.216±0.091</td><td>1.225±0.103</td></tr><tr><td>B=10×8</td><td>0.733±0.217</td><td>1.111±0.029</td><td>0.954±0.061</td><td>0.987±0.035</td><td>1.315±0.046</td><td>1.077±0.020</td><td>0.676±0.056</td><td>1.076±0.050</td><td>1.082±0.057</td></tr><tr><td>B=30</td><td>2.721±1.485</td><td>23.220±2.344</td><td>27.249±2.383</td><td>2.245±1.257</td><td>23.960±1.510</td><td>26.949±1.264</td><td>2.576±1.823</td><td>22.832±1.135</td><td>27.671±0.908</td></tr><tr><td>B=40</td><td>1.576±1.703</td><td>21.301±1.866</td><td>24.462±2.159</td><td>1.959±0.980</td><td>23.159±0.843</td><td>26.453±1.327</td><td>2.695±0.874</td><td>21.602±2.216</td><td></td></tr><tr><td>B=50</td><td>1.975±1.292</td><td>22.271±1.694</td><td>26.995±1.633</td><td>1.514±1.280</td><td>21.918±1.572</td><td>25.591±1.990</td><td>1.758±0.786</td><td>21.613±2.484</td><td>26.686±1.731 26.132±2.456</td></tr><tr><td rowspan="5">[56] ER [37]</td><td>B=80</td><td>1.465±0.787</td><td>20.386±1.287</td><td>25.022±1.699</td><td>1.108±0.463</td><td>22.119±0.163</td><td>25.157±0.929</td><td>2.038±1.432</td><td>22.004±2.027</td><td>26.638±1.645</td></tr><tr><td>B=30</td><td>0.737±0.079</td><td>1.106±0.149</td><td>1.001±0.070</td><td>0.798±0.135</td><td>0.993±0.216</td><td>1.005±0.095</td><td>0.651±0.100</td><td>0.898±0.022</td><td></td></tr><tr><td>B=40</td><td>0.735±0.136</td><td>0.949±0.157</td><td>0.996±0.165</td><td>0.765±0.094</td><td>0.997±0.095</td><td>0.898±0.062</td><td>0.503±0.088</td><td></td><td>1.052±0.104</td></tr><tr><td>B=50</td><td>0.630±0.153</td><td>0.927±0.060</td><td>0.899±0.012</td><td>0.714±0.067</td><td>0.941±0.092</td><td></td><td>0.467±0.172</td><td>0.773±0.056</td><td>0.945±0.035</td></tr><tr><td>B=80</td><td>0.565±0.144</td><td>0.826±0.056</td><td>0.843±0.090</td><td>0.564±0.069</td><td>0.790±0.031</td><td>0.863±0.045</td><td></td><td>0.745±0.064</td><td>0.894±0.093</td></tr><tr><td rowspan="5">AVQACL [57]</td><td></td><td></td><td>2.516±0.495</td><td></td><td></td><td></td><td>0.743±0.049</td><td>0.394±0.038</td><td>0.624±0.061</td><td>0.749±0.120</td></tr><tr><td>B=30</td><td>1.788±0.536 1.401±0.172</td><td></td><td>2.726±1.421</td><td>1.641±0.267</td><td>2.354±0.261</td><td>1.889±0.145</td><td>1.198±0.201</td><td>1.893±0.136</td><td>1.617±0.094</td></tr><tr><td>B=40</td><td>1.368±0.300</td><td>2.114±0.152 2.084±0.281</td><td>2.565±0.631</td><td>1.692±0.246 1.422±0.145</td><td>2.445±0.217 2.154±0.121</td><td>1.843±0.171</td><td>1.325±0.267</td><td>1.979±0.236</td><td>1.561±0.115</td></tr><tr><td>B=50 B=80</td><td></td><td>2.681±0.962</td><td>2.241±0.114</td><td></td><td>2.405±0.364</td><td>1.585±0.128</td><td>1.272±0.305</td><td>1.963±0.300</td><td>1.748±0.286</td></tr><tr><td></td><td>2.018±0.847</td><td>0.772±0.176</td><td>2.629±0.635</td><td>1.655±0.381</td><td></td><td>1.780±0.083</td><td>1.375±0.304</td><td>2.050±0.311</td><td>1.933±0.510</td></tr><tr><td rowspan="4">MCLC (Ours)</td><td>B=30</td><td>0.334±0.090</td><td>0.654±0.157</td><td>1.376±0.477</td><td>0.192±0.108 0.145±0.163</td><td>0.769±0.018 0.756±0.111</td><td>0.736±0.176</td><td>0.097±0.020 0.258±0.219</td><td>0.686±0.080</td><td>1.063±0.235</td></tr><tr><td>B=40 B=50</td><td>0.218±0.071 0.186±0.083</td><td>0.557±0.063</td><td>0.972±0.272 0.795±0.201</td><td>0.174±0.038</td><td>0.709±0.052</td><td>0.742±0.213 0.655±0.088</td><td>0.297±0.201</td><td>0.714±0.132 0.721±0.143</td><td>1.084±0.145</td></tr><tr><td>B=80</td><td>0.280±0.115</td><td>0.661±0.104</td><td>0.675±0.027</td><td>0.145±0.116</td><td>0.644±0.037</td><td>0.542±0.069</td><td>0.055±0.038</td><td>0.594±0.050</td><td>1.066±0.308</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.745±0.057</td></tr></table>

ER shows the clearest effect of buffer size. As the buffer memory increases from 30 to 80, the AMSE consistently decreases across all task sequences. For instance, the AMSE improves from 1.001±0.070 to 0.843±0.090 in CS1, from 1.005±0.095 to 0.743±0.049 in CS2, and from 1.052±0.104 to 0.749±0.120 in CS3. Similarly, AVQACL benefits from larger buffer sizes and achieves its best performance with more memory, although its comparatively large standard deviations indicate greater sensitivity to initialization across different random seeds. Compared with all existing methods, the proposed MCLC consistently achieves the best performance across different buffer settings. Even with moderate buffer sizes of 30 and 40, MCLC performs competitively, and increasing the buffer size to 80 further improves performance, achieving the lowest AMSE values of 0.675±0.027, 0.542±0.069, and 0.745±0.057 across CS1, CS2, and CS3, respectively. Moreover, MCLC also achieves the lowest BWT and FWT values in most settings.

Overall, these results show that memory plays a crucial role in CL, and increasing buffer size generally improves performance. More importantly, the way memory is utilized is equally important. The proposed MCLC method demonstrates that selecting high-quality, representative samples yields better performance than simply increasing memory size.

## E. Evaluating Generalization Across Datasets

To further validate the effectiveness of our approach, we conducted experiments on an additional dataset, MSU-PID, using the same set of CL methods. As shown in Table X, the results follow trends similar to those observed on the MMLC dataset. Rehearsalbased methods clearly benefit from increased buffer sizes. For example, ER shows a consistent reduction in AMSE as the buffer size increases, improving from 0.968±0.009 (B=30) to 0.750±0.109 (B=80). Similarly, GEM and A-GEM also achieve lower AMSE values with larger buffers, reaching their best performance of 0.869±0.052 and 0.813±0.084, respectively, at $\scriptstyle B = 1 0 \times 8$ . However, regularization-based methods like EWC still perform relatively worse, with AMSE remaining between 0.882±0.074 and 0.896±0.135, highlighting their limitations in handling changing data distributions. DER++ performs significantly worse, with very high AMSE values, indicating instability in this domain incremental CL setting. AVQACL shows competitive performance at smaller buffer sizes but does not improve consistently as buffer size increases. Notably, our proposed MCLC outperforms all baselines, achieving the lowest AMSE of $0 . 6 6 4 { \scriptstyle \pm 0 . 0 7 1 }$ (B=80) along with the lowest FWT of $0 . 9 7 9 { \scriptstyle \pm 0 . 1 0 4 }$ . These results indicate that MCLC effectively preserves prior knowledge while adapting to new tasks, making it a robust and scalable solution across diverse datasets.

TABLE X: Performance comparison of different approaches under various hyperparameter settings on the MSU-PID dataset.
<table><tr><td>Approach</td><td>Parameter</td><td>BWT↓</td><td>FWT↓</td><td>AMSE ↓</td></tr><tr><td rowspan="3">EWC [12]</td><td>λ=1</td><td> $1 . 0 2 9 \pm 0 . 5 1 4$ </td><td> $1 . 2 4 6 \pm 0 . 1 1 8$ </td><td> $0 . 8 8 3 \pm 0 . 0 7 9$ </td></tr><tr><td>λ=2</td><td> $1 . 0 6 2 \pm 0 . 5 7 5$ </td><td> $1 . 2 0 3 \pm 0 . 1 0 5$ </td><td> $0 . 8 8 2 \pm 0 . 0 7 4$ </td></tr><tr><td>λ=3</td><td> $1 . 0 8 2 \pm 0 . 5 5 3$ </td><td> $1 . 2 4 5 \pm 0 . 0 4 5$ </td><td> $0 . 8 9 6 \pm 0 . 1 3 5$ </td></tr><tr><td rowspan="4">GEM [13]</td><td>B =4x8</td><td> $0 . 9 1 8 \pm 0 . 4 5 5$ </td><td> $1 . 4 0 5 \pm 0 . 1 4 1$ </td><td> $1 . 0 2 0 \pm 0 . 1 6 9$ </td></tr><tr><td>B =5x8</td><td> $1 . 1 9 0 \pm 0 . 6 5 2$ </td><td> $1 . 2 0 2 \pm 0 . 0 2 4$ </td><td> $1 . 0 5 8 \pm 0 . 1 5 4$ </td></tr><tr><td>B =6x8</td><td> $0 . 9 7 5 \pm 0 . 4 2 1$ </td><td> $1 . 2 1 8 \pm 0 . 0 9 4$ </td><td> $0 . 9 9 6 \pm 0 . 0 4 2$ </td></tr><tr><td>B =10x8</td><td> $1 . 1 3 4 \pm 0 . 3 5 6$ </td><td> $1 . 1 0 3 \pm 0 . 0 6 7$ </td><td> $0 . 8 6 9 \pm 0 . 0 5 2$ </td></tr><tr><td rowspan="4">A-GEM [36]</td><td>B =4x8</td><td> $0 . 9 8 4 \pm 0 . 4 3 7$ </td><td> $\overline { { 1 . 2 3 4 \pm 0 . 0 5 9 } }$ </td><td> $\overline { { 0 . 9 2 0 \pm 0 . 0 6 6 } }$ </td></tr><tr><td>B =5x8</td><td> $1 . 0 0 9 \pm 0 . 5 4 1$ </td><td> $1 . 1 2 1 \pm 0 . 1 0 3$ </td><td> $0 . 8 4 9 \pm 0 . 0 8 2$ </td></tr><tr><td>B=6x8</td><td> $1 . 1 3 5 \pm 0 . 6 6 5$ </td><td> $1 . 1 2 5 \pm 0 . 1 6 4$ </td><td> $0 . 8 5 9 \pm 0 . 0 1 6$ </td></tr><tr><td>B =10x8</td><td> $1 . 0 7 3 \pm 0 . 5 7 4$ </td><td> $1 . 1 1 6 \pm 0 . 0 9 8$ </td><td> $0 . 8 1 3 \pm 0 . 0 8 4$ </td></tr><tr><td rowspan="4">DER++ [56]</td><td>B=30</td><td> $- 5 . 5 1 4 \pm 1 9 . 0 2 4$ </td><td> $4 4 . 6 4 1 \pm 3 . 1 2 0$ </td><td> $4 9 . 1 4 0 \pm 4 . 5 6 2$ </td></tr><tr><td>B=40</td><td> $- 4 . 4 9 7 \pm 1 4 . 0 0 4$ </td><td> $3 7 . 0 2 2 \pm 3 . 5 5 5$ </td><td> $3 7 . 9 4 3 \pm 4 . 6 0 2$ </td></tr><tr><td>B=50</td><td> $- 3 . 8 8 8 \pm 1 7 . 0 1 7$ </td><td> $3 8 . 8 9 7 \pm 4 . 7 7 5$ </td><td> $4 1 . 9 5 3 \pm 5 . 5 9 0$ </td></tr><tr><td>B=80</td><td> $- 3 . 6 5 6 \pm 1 4 . 3 3 8$ </td><td> $3 6 . 9 0 9 \pm 6 . 8 2 6$ </td><td> $3 7 . 6 1 6 \pm 6 . 9 5 4$ </td></tr><tr><td rowspan="4">ER [37]</td><td>B=30</td><td> $0 . 9 8 6 \pm 0 . 3 4 7$ </td><td> $1 . 1 3 9 \pm 0 . 0 5 0$ </td><td> $0 . 9 6 8 \pm 0 . 0 0 9$ </td></tr><tr><td>B=40</td><td> $1 . 0 3 1 \pm 0 . 3 8 3$ </td><td> $1 . 0 5 0 \pm 0 . 0 2 2$ </td><td> $0 . 8 5 7 \pm 0 . 0 9 9$ </td></tr><tr><td>B=50</td><td>1.119 ± 0.564</td><td> $1 . 0 4 6 \pm 0 . 0 6 4$ </td><td> $0 . 8 6 2 \pm 0 . 0 3 2$ </td></tr><tr><td>B=80</td><td>1.144 ± 0.590</td><td> $1 . 0 3 1 \pm 0 . 1 3 3$ </td><td> $0 . 7 5 0 \pm 0 . 1 0 9$ </td></tr><tr><td rowspan="4">AVQACL [57]</td><td>B=30</td><td>0.104 ± 0.467</td><td> $\overline { { 1 . 3 8 5 \pm 0 . 3 3 3 } }$ </td><td> $\overline { { 1 . 1 8 0 \pm 0 . 1 3 6 } }$ </td></tr><tr><td>B=40</td><td>0.060 ± 0.630</td><td> $1 . 3 3 6 \pm 0 . 3 5 7$ </td><td> $1 . 2 7 7 \pm 0 . 3 8 0$ </td></tr><tr><td>B=50</td><td> $0 . 1 0 4 \pm 0 . 5 6 5$ </td><td> $1 . 3 4 5 \pm 0 . 3 1 1$ </td><td> $1 . 3 4 9 \pm 0 . 4 5 1$ </td></tr><tr><td>B=80</td><td> $0 . 0 9 5 \pm 0 . 2 9 2$ </td><td> $1 . 1 8 3 \pm 0 . 1 8 6$ </td><td> $1 . 0 3 0 \pm 0 . 1 7 0$ </td></tr><tr><td rowspan="4">MCLC (Ours)</td><td>B=30</td><td> $2 . 1 1 3 \pm 0 . 2 4 8$ </td><td> $1 . 0 9 9 \pm 0 . 0 8 1$ </td><td> $0 . 9 3 6 \pm 0 . 1 6 1$ </td></tr><tr><td>B=40</td><td> $2 . 1 5 9 \pm 0 . 1 8 8$ </td><td> $1 . 0 7 1 \pm 0 . 0 5 5$ </td><td> $0 . 8 6 4 \pm 0 . 0 0 6$ </td></tr><tr><td> $\scriptstyle B = 5 0$ </td><td> $2 . 1 5 4 \pm 0 . 2 0 1$ </td><td> $1 . 0 5 3 \pm 0 . 0 5 3$ </td><td> $0 . 7 9 8 \pm 0 . 0 2 8$ </td></tr><tr><td>B=80</td><td> ${ \bf 2 . 2 4 4 \pm 0 . 1 4 8 }$ </td><td> $\mathbf { 0 . 9 7 9 \pm 0 . 1 0 4 }$ </td><td> $\mathbf { 0 . 6 6 4 \pm 0 . 0 7 1 }$ </td></tr></table>

## VII. CONCLUSION

We address leaf counting in dynamic agricultural environments using a CL framework with multimodal data. We propose MCLC, which combines cross-modal fusion with a replay-based memory to learn tasks sequentially while reducing forgetting. Unlike traditional methods, MCLC can adapt to changing data distributions while retaining previously learned knowledge. The experimental results, averaged over three random seeds, show that MCLC consistently outperforms the baseline methods across all data sequences.For instance, in CS1, it achieves the lowest AMSE of $0 . 6 7 5 { \scriptstyle \pm 0 . 0 2 7 }$ while in CS2 and CS3 it further achieves AMSE values of $0 . 5 4 2 { \scriptstyle \pm 0 . 0 6 9 }$ and $0 . 7 4 5 { \scriptstyle \pm 0 . 0 5 7 }$ , respectively. It also maintains low BWT and FWT values, indicating effective knowledge retention and forward transfer across different task orders. Compared with EWC and memorybased methods such as ER, GEM and A-GEM, MCLC consistently achieves better performance, particularly with larger buffer sizes. Overall, MCLC provides a reliable and scalable solution for multimodal continual leaf counting. Its ability to handle distribution changes, reduce forgetting, and make effective use of memory makes it useful not only for plant phenotyping but also for other real-world CL applications.

## VIII. LIMITATIONS AND FUTURE WORK

Although the proposed MCLC framework demonstrates strong performance for multimodal continual leaf counting, it assumes the availability of aligned RGB, depth, and thermal modalities during both training and inference. In practical agricultural deployments, some modalities may be missing, noisy, or affected by sensor failures, environmental conditions, or unavailability. Such scenarios may reduce the effectiveness of multimodal fusion and continual adaptation. Also, MCLC currently relies on a fixed memory buffer, which may be challenging for large-scale, long-term CL settings.

Future work will focus on developing more robust multimodal CL strategies that handle missing modalities and noisy sensor inputs and on more memory-efficient buffer mechanisms for real-world agricultural applications.

## ACKNOWLEDGMENTS

The authors acknowledge ANNAM.AI, an AI-CoE of the Ministry of Education, Govt. of India, at the Indian Institute of Technology Ropar, for providing resources and support to execute this work. This work is also supported by a grant from DST, Govt. of India, for the Technology Innovation Hub at IIT Ropar in the framework of the National Mission on Interdisciplinary Cyber-Physical Systems.

## REFERENCES

[1] M. Buzzy, V. Thesma, M. Davoodi, and J. Mohammadpour Velni, “Real-time plant leaf counting using deep object detection networks,” Sensors, vol. 20, no. 23, p. 6896, 2020.

[2] Z. Li, R. Guo, M. Li, Y. Chen, and G. Li, “A review of computer vision technologies for plant phenotyping,” Computers and Electronics in Agriculture, vol. 176, p. 105672, 2020.

[3] A. Quamer, N. Asgari, and J. M. Pearce, “Two-stage deep learning model for non-destructive leaf counting in tomato crops,” Computers and Electronics in Agriculture, vol. 247, p. 111717, 2026.

[4] M. Minervini, A. Fischbach, H. Scharr, and S. A. Tsaftaris, “Finely-grained annotated datasets for image-based plant phenotyping,” Pattern recognition letters, vol. 81, pp. 80–89, 2016.

[5] J. Chen, J. Chen, D. Zhang, Y. Sun, and Y. A. Nanehkaran, “Using deep transfer learning for image-based plant disease identification,” Computers and Electronics in Agriculture, vol. 173, p. 105393, 2020.

[6] Y. Yue, J.-H. Li, L.-F. Fan, L.-L. Zhang, P.-F. Zhao, Q. Zhou, N. Wang, Z.-Y. Wang, L. Huang, and X.-H. Dong, “Prediction of maize growth stages based on deep learning,” Computers and Electronics in Agriculture, vol. 172, p. 105351, 2020.

[7] A. Dobrescu, M. Valerio Giuffrida, and S. A. Tsaftaris, “Leveraging multiple datasets for deep leaf counting,” in Proceedings of the IEEE International Conference on Computer Vision workshops, 2017, pp. 2072–2079.

[8] S. Sarkar, A. Dey, R. Pradhan, U. M. Sarkar, C. Chatterjee, A. Mondal, and P. Mitra, “Crop yield prediction using multimodal meta-transformer and temporal graph neural networks,” IEEE Transactions on AgriFood Electronics, vol. 2, no. 2, pp. 545– 553, 2024.

[9] M. V. Giuffrida, P. Doerner, and S. A. Tsaftaris, “Pheno-deep counter: A unified and versatile deep learning architecture for leaf counting,” The Plant Journal, vol. 96, no. 4, pp. 880–890, 2018.

[10] J. A. Cruz, X. Yin, X. Liu, S. M. Imran, D. D. Morris, D. M. Kramer, and J. Chen, “Multi-modality imagery database for plant phenotyping,” Machine Vision and Applications, vol. 27, no. 5, pp. 735–749, 2016.

[11] P. Kumari, D. Reisenbuchler, L. Luttner, N. S. Schaadt, F. Feuer-¨ hake, and D. Merhof, “Continual domain incremental learning for privacy-aware digital pathology,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 34–44.

[12] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[13] D. Lopez-Paz and M. Ranzato, “Gradient episodic memory for continual learning,” Advances in neural information processing systems, vol. 30, 2017.

[14] M. J. Mirza, M. Masana, H. Possegger, and H. Bischof, “An efficient domain-incremental learning approach to drive in all weather conditions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3001– 3011.

[15] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong, “Fewshot class-incremental learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 12 183–12 192.

[16] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks,” arXiv preprint arXiv:1708.01547, 2017.

[17] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning with deep generative replay,” Advances in neural information processing systems, vol. 30, 2017.

[18] S. A. Bidaki, A. Mohammadkhah, K. Rezaee, F. Hassani, S. Eskandari, M. Salahi, and M. M. Ghassemi, “Online continual learning: A systematic literature review of approaches, challenges, and benchmarks,” arXiv preprint arXiv:2501.04897, 2025.

[19] M. V. Giuffrida, M. Minervini, and S. A. Tsaftaris, “Learning to count leaves in rosette plants,” In CVPPP workshop, page 13. British Machine Vision Association, 2015.

[20] L. Xu, Y. Li, Y. Sun, L. Song, and S. Jin, “Leaf instance segmentation and counting based on deep object detection and segmentation networks,” in 2018 Joint 10th International Conference on Soft Computing and Intelligent Systems (SCIS) and 19th International Symposium on Advanced Intelligent Systems (ISIS). IEEE, 2018, pp. 180–185.

[21] Y. Itzhaky, G. Farjon, F. Khoroshevsky, A. Shpigler, and A. Bar-Hillel, “Leaf counting: Multiple scale regression and detection using deep cnns.” in BMVC, vol. 328. Newcastle, 2018.

[22] Y.-L. Tu, W.-Y. Lin, and Y.-C. Lin, “Automatic leaf counting using improved yolov3,” in 2020 International Symposium on

Computer, Consumer and Control (IS3C). IEEE, 2020, pp. 197– 200.

[23] F. Sun, H. Liu, C. Yang, and B. Fang, “Multimodal continual learning using online dictionary updating,” IEEE Transactions on Cognitive and Developmental Systems, vol. 13, no. 1, pp. 171–178, 2020.

[24] T. Srinivasan, T.-Y. Chang, L. Pinto Alva, G. Chochlakis, M. Rostami, and J. Thomason, “Climb: A continual learning benchmark for vision-and-language tasks,” Advances in Neural Information Processing Systems, vol. 35, pp. 29 440–29 453, 2022.

[25] Y. Zhao, C. Jiang, D. Wang, X. Liu, W. Song, and J. Hu, “Identification of plant disease based on multi-task continual learning,” Agronomy, vol. 13, no. 12, p. 2863, 2023.

[26] S. Bansal, M. Singh, S. Barda, N. Goel, and M. Saini, “Pardfknet: Unifying plant age estimation through rgb-depth fusion and knowledge distillation,” IEEE Transactions on AgriFood Electronics, vol. 2, no. 2, pp. 226–235, 2024.

[27] A. Bono, G. Vivaldi, C. Guaragnella, and T. D’Orazio, “A visual approach for estimating plant growth during the life cycle of a vineyard,” IEEE Transactions on AgriFood Electronics, vol. 4, no. 1, pp. 395–408, 2026.

[28] R. R. Donapati, R. Cheruku, and P. Kodali, “Real-time seed detection and germination analysis in precision agriculture: A fusion model with u-net and cnn on jetson nano,” IEEE Transactions on AgriFood Electronics, vol. 1, no. 2, pp. 145– 155, 2023.

[29] P. Garg, A. Mishra, R. Raja, A. Kumar, M. V. Joshi, and V. S. Palaparthy, “Multimodal data fusion by integrating iotenabled sensors and images for jamun crop disease detection with machine learning,” IEEE Transactions on AgriFood Electronics, vol. 3, no. 2, pp. 582–590, 2025.

[30] Y.-L. Tu, W.-Y. Lin, and Y.-C. Lin, “Toward automatic plant phenotyping: starting from leaf counting,” Multimedia Tools and Applications, vol. 81, no. 9, pp. 11 865–11 879, 2022.

[31] D. Kuznichov, A. Zvirin, Y. Honen, and R. Kimmel, “Data augmentation for leaf segmentation and counting tasks in rosette plants,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019.

[32] M. Deb, K. G. Dhal, A. Das, A. G. Hussien, L. Abualigah, and A. Garai, “A cnn-based model to count the leaves of rosette plants (lc-net),” Scientific Reports, vol. 14, no. 1, p. 1496, 2024.

[33] S. Aich and I. Stavness, “Improving object counting with heatmap regulation,” arXiv preprint arXiv:1803.05494, 2018.

[34] W. Xie, J. A. Noble, and A. Zisserman, “Microscopy cell counting and detection with fully convolutional regression networks,” Computer methods in biomechanics and biomedical engineering: Imaging & Visualization, vol. 6, no. 3, pp. 283–292, 2018.

[35] S. Lu, Z. Song, W. Chen, T. Qian, Y. Zhang, M. Chen, and G. Li, “Counting dense leaves under natural environments via an improved deep-learning-based object detection algorithm,” Agriculture, vol. 11, no. 10, p. 1003, 2021.

[36] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” arXiv preprint arXiv:1812.00420, 2018.

[37] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” Advances in neural information processing systems, vol. 32, 2019.

[38] Z. Wang, C. Subakan, E. Tzinis, P. Smaragdis, and L. Charlin, “Continual learning of new sound classes using generative replay,” in 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2019, pp. 308–312.

[39] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[40] M. Zhang, T. Wang, J. H. Lim, G. Kreiman, and J. Feng, “Variational prototype replays for continual learning,” arXiv preprint arXiv:1905.09447, 2019.

[41] S. Ho, M. Liu, L. Du, L. Gao, and Y. Xiang, “Prototype-guided memory replay for continual learning,” IEEE transactions on

neural networks and learning systems, vol. 35, no. 8, pp. 10 973– 10 983, 2023.

[42] K. Wang, L. Herranz, and J. van de Weijer, “Continual learning in cross-modal retrieval,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3628–3638.

[43] T. Lesort, V. Lomonaco, A. Stoian, D. Maltoni, D. Filliat, and N. D´ıaz-Rodr´ıguez, “Continual learning for robotics: Definition, framework, learning strategies, opportunities and challenges,” Information fusion, vol. 58, pp. 52–68, 2020.

[44] S. Gai, Z. Chen, and D. Wang, “Multi-modal meta continual learning,” in 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021, pp. 1–8.

[45] Y. Cai and M. Rostami, “Dynamic transformer architecture for continual learning of multimodal tasks,” arXiv preprint arXiv:2401.15275, 2024.

[46] J. Li, D. Chen, X. Qi, Z. Li, Y. Huang, D. Morris, and X. Tan, “Label-efficient learning in agriculture: A comprehensive review,” Computers and Electronics in Agriculture, vol. 215, p. 108412, 2023.

[47] G. M. Dimitri, A. Kocian, F. Scarselli, M. Gori, and S. Chessa, “Continual learning for agrifood: A systematic review,” IEEE Transactions on AgriFood Electronics, vol. 4, no. 1, pp. 117– 127, 2026.

[48] Y. Li and J. Yang, “Meta-learning baselines and database for fewshot classification in agriculture,” Computers and Electronics in Agriculture, vol. 182, p. 106055, 2021.

[49] M. Page-Fortin, “Class-incremental learning of plant and dis-´ ease detection: Growing branches with knowledge distillation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 593–603.

[50] D. Li, Z. Yin, Y. Zhao, J. Li, and H. Zhang, “Rehearsalbased class-incremental learning approaches for plant disease classification,” Computers and Electronics in Agriculture, vol. 224, p. 109211, 2024.

[51] H.-Y. Chien, K.-H. Liu, and C. Lin, “Plant species recognition based on continual learning strategy,” in IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium IEEE, 2024, pp. 10 363–10 367.

[52] A. S. Reza, F. U. Zaima, M. M. Annisa, and T. Sattar, “Optimizing cnn memory efficiency: a continual learning solution for plant stress classification,” Ph.D. dissertation, Brac University, 2024.

[53] Y. He and B. Sick, “Clear: An adaptive continual learning framework for regression tasks,” AI Perspectives, vol. 3, no. 1, p. 2, 2021.

[54] P. Kumari, J. Chauhan, A. Bozorgpour, B. Huang, R. Azad, and D. Merhof, “Continual learning in medical image analysis: A comprehensive review of recent advancements and future prospects,” Medical Image Analysis, p. 103730, 2025.

[55] J. Bell and H. M. Dee, “Aberystwyth leaf evaluation dataset,” URL: https://doi. org/10.5281/zenodo, vol. 168158, no. 17-36, p. 2, 2016.

[56] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: a strong, simple baseline,” Advances in neural information processing systems, vol. 33, pp. 15 920–15 930, 2020.

[57] K. Wu, X. Li, X. Li, C. Hu, and G. Wu, “Avqacl: A novel benchmark for audio-visual question answering continual learning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 3252–3261.

[58] M. Mermillod, A. Bugaiska, and P. Bonin, “The stabilityplasticity dilemma: Investigating the continuum from catastrophic forgetting to age-limited learning effects,” p. 504, 2013.

[59] N. D´ıaz-Rodr´ıguez, V. Lomonaco, D. Filliat, and D. Maltoni, “Don’t forget, there is more than forgetting: new metrics for continual learning,” arXiv preprint arXiv:1810.13166, 2018.

[60] M. Havaei, N. Guizard, N. Chapados, and Y. Bengio, “Hemis: Hetero-modal image segmentation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19. Springer, 2016, pp. 469–477.

[61] R. Shah, R. Mart´ın-Mart´ın, and Y. Zhu, “Mutex: Learning unified policies from multimodal task specifications,” arXiv preprint arXiv:2309.14320, 2023.

[62] J. Cai, J. Su, Q. Li, W. Yang, S. Wang, T. Zhao, S. He, and W. Liu, “Keep the balance: A parameter-efficient symmetrical framework for rgb+ x semantic segmentation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 10 587–10 598.