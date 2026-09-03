# ADAPTING A FOUNDATION MODEL FOR LUNAR SURFACE HEIGHT ESTIMATION

Patrick Bauer University of Technology of Troyes Hochschule Darmstadt patrick.bauer@utt.fr

Marius Schwinning European Space Agency marius.schwinning@esa.int

Andreas Weinmann Technische Hochschule Würzburg-Schweinfurt andreas.weinmann@thws.de

Melanie Siegel Hochschule Darmstadt melanie.siegel@h-da.de

Hichem Snoussi University of Technology of Troyes hichem.snoussi@utt.fr

## ABSTRACT

Digital elevation models (DEMs) can provide accurate height information, making it invaluable for analyzing the lunar surface. As the European Space Agency (ESA) prepares for future lunar missions that aim to land on the Moon, a precise method for height estimation will be essential for hazardous terrain that could endanger the landing approach. Traditional approaches to generate DEMs from imagery, such as shape from shading (SfS) and stereophotogrammetry (SPG) have been proven highly valuable for this task. However, due to advancements in machine learning, especially computer vision, the focus has shifted towards monocular depth estimation via deep learning. The lunar surface is covered by rocks and craters, and classic hazard detection methods rely solely on 2D image data. Our goal is to address this issue by developing a relative lunar surface height estimator that can provide additional information for hazard localization. In this letter, we present a methodology that builds on the well-known zero-shot relative depth estimation model Depth Anything V2 (DAV2). Other works have been using it as a state-of-the-art comparison for their proposed lunar DEM estimation method, but without adaptations to the target domain. Thus, it may underperform. Therefore, we propose a fine-tuning strategy with publicly available SPG-derived DEM data of the lunar surface. Our results demonstrate a significant improvement in performance compared to the zero-shot model, effectively transforming DAV2 into a reliable relative depth estimator of the lunar surface.

Keywords Digital elevation model · monocular depth estimation · lunar surface height estimation · Depth Anything V2

## 1 Introduction

After the Artemis II mission, in which a manned spacecraft left Earth on April 1, 2026, for a lunar flyby mission, that was planned organized, and conducted by NASA, interest in space science and industry increased rapidly. Upcoming lunar missions are of utmost importance and gain significant public attention. ESA contributes to past and future planned Artemis missions by developing and providing tools to the spacecraft. Further, ESA is developing its own access to the lunar surface with Argonaut [1]. However, landing on the Moon is a non-trivial endeavor, as the surface consists of a vast amount of craters and boulders of diverse sizes and shapes. Crater detection is a promising approach for identifying and localizing hazards on the lunar surface. Researchers are trying to detect craters using deep learning techniques to determine their locations and distribution [2, 3, 4]. These approaches typically involve lunar image data instead of digital elevation models (DEMs). By solely relying on image data, one is leaving out the third dimension and this can be fatal for detection quality, as some degraded craters are hard to detect [5]. The quality of crater detection could benefit by incorporating height information. Geologists and lunar researchers are already highly interested in accessing the most accurate and highest-resolution DEMs possible. In general, DEM generation can be approached in various ways. Since launching the Lunar Reconnaissance Orbiter (LRO) with two onboard cameras (NAC-L and NAC-R) [6] in 2009, NASA has provided a vast amount of image data with a resolution up to 0.5 m/px. The LRO also has a laser altimeter that can measure the elevation of the lunar surface with precise height measurements but coarse resolution only. Techniques such as Shape from Shading (SfS) [7] and stereophotogrammetry (SPG) [8], that typically need at least two images of the same scene with disparate emission angles, can be applied to lunar images to generate DEMs. SPG generates DEM products with a spatial resolution of up to 2 m/px but faces significant limitations since stereo observations covering the same scene are very limited. According to Wagner et al. [9], approximately 5% of the lunar surface is covered by stereo pairs. SfS relies on one or multiple images of the same area [10]. However, it lacks coverage of the same scene under varying illumination conditions and single-image SfS is underconstrained, and one has to make assumptions about the terrain’s albedo [11].

In recent years, deep learning-based approaches have been very prominent in various computer vision tasks, such as object detection and semantic segmentation. In the field of monocular depth estimation (MDE), various deep learning based models were introduced. These models can estimate depth information without requiring multiple images of the same scene. Models such as MiDAS [12], Marigold [13], Depth Anything [14] and its successor Depth Anything V2 (DAV2) [15] yield superior MDE results. They have the additional advantage of being foundation models. Foundation models are a class of models that have been extensively pre-trained on vast amounts of data and can be used for various downstream tasks. They are often based on a Transformer architecture [16] and, therefore, on attention modules. Recently, they have been applied more frequently in the area of remote sensing [17]. However, naively applying foundation models to remote sensing tasks does not yield good results due to a significant domain gap between natural scenery images in pre-training and remote sensing data in the downstream tasks, cf. Luo et al. [18]. Therefore, adapting and fine-tuning are required.

In the field of deep learning-based MDE for planetary surfaces such as the Moon and Mars, a broad range of approaches and techniques exist. For example, several works [10, 19, 20, 21, 22, 23] have been introduced and yielded very promising and highly accurate models trained for that task. For example, Chen et al. [22] developed ELunarDTMNet, which comprises a dual-branch encoder, using hierarchical Transformer blocks for images and convolutional blocks fo DTMs. This is followed by a fusion module and a residual-connected decoder module. The objective is to estimate the absolute height of the lunar surface, which is generally the aim of most proposed methods. La Grassa et al. [23] and Osadnik et al. [24] compared their approach with a state-of-the-art (SoTA) model, DAV2. However, due to the aforementioned domain gap, we argue that the comparison with a zero-shot model is limited.

Therefore we contribute a novel lunar surface estimation approach by fine-tuning the DAV2 using publicly available, high resolution SPG derived DEMs. Unlike existing approaches, which generally aim to reconstruct absolute heights in meters using coarse-resolution laser altimetry derived products as reference maps, our approach focuses on relative height estimation. This enables us to develop an auxiliary model that could support existing hazard detectors, which currently only consider 2D data.

## 2 Data and Methodology

## 2.1 Available Lunar DEM Data

NASA launched the LRO in 2009 [6]. Since then, it has been capturing high-resolution images of the lunar surface from varying orbits. Taking images of the same scenery on subsequent orbits from an oblique angle allows researchers to create SPG-derived DEMs of the lunar surface with a resolution of up to 2 m/px [8]. This approach relies on acquiring data on subsequent orbits. However, available stereo pairs only cover around 5% of the lunar surface [9]. DEM products based on the Lunar Reconnaissance Orbiter Laser Altimeter (LOLA) produce very accurate height maps, albeit at the cost of coarse resolution. For example, the SLDEM [25] has a resolution of 60 m/px. In order to be able to reconstruct the elevation profile of the lunar surface containing high-frequency details, we select the available SPG-derived DEM data products with a resolution of 2 m/px and their corresponding orthorectified images (ORIs) and download the data from https://data.lroc.im-ldi.com/lroc/rdr\_product\_select. To generate a train-test-validation split, we first crop all the selected ORIs and DEMs into tiles of size 259×259. To prevent data leakage, we ensure that the same physical region on the Moon is strictly assigned to one of three sets. Since some DEMs contain nodata values, we follow the strategy outlined in Tao et al. [20], where tiles with nodata values are removed. In total, we obtained 84,853 training images, 10,920 validation images, and 13,020 test images to align with an approximate 80—10—10 split.

## 2.2 Depth Anything V2

Yang et al. [14] introduced a MDE model called Depth Anything. Due to its architecture and the vast amount of training data, it is capable of zero-shot depth estimation on natural images. Subsequently, Yang et al. developed the successor, Depth Anything V2 (DAV2) [15], incorporating real pseudo-labeled images into training process. Specifically, they propose a three-step training pipeline. First, a reliable teacher is trained solely on synthetic imagery. This enables accurate pseudo depth to be produced on a vast amount of unlabelled real images, and a final student model is then trained on this data. There are four versions depending on the encoder: ViT-S, ViT-B, ViT-L and ViT-G and we refer to the ViT-L variant throughout. DAV2 consists of a DINOv2 [26] based encoder, by stacking 24 Transformer blocks. The decoder utilizes the DPT [27] decoder system, which is a convolutional neural network (CNN)-based model. It consists of three-stage Reassemble operations and the core in the decoder are RefineNet-based feature fusion blocks [28, 29]. The model first normalizes and resizes the image. The shorter side is resized to 518, and the other is resized to be a multiple of 14 that is nearest to the aspect-ratio-preserving value. For a square image, it is resized to $5 1 8 \times 5 1 8$ . For that case, the image is further split into 1,369 patches of size 14×14 with positional encodings added. After the processing through the Transformer modules, 4 tensors at different stages serve as the inputs to the decoder. After processing through the CNN-based DPT decoder modules, the output of the decoder block is the inverse depth map where the values represent relative inverse depth. During the pre-training phase of the models, a scale-and-shift invariant (SSI) loss was minimized based on the concept proposed by Ranftl et al. [12]. To develop a zero-shot capable MDE model, they combined various ground truth (GT) datasets. This resulted in the use of datasets with different scales and shifts inherent to the data. An SSI-based loss function overcomes this challenge. DAV2 follows that idea by applying an affine transformation to the prediction and the GT. Precisely, let d and <sup>ˆ</sup>d be the ground truth and the predicted depth map, respectively, both with height H and width W. Before minimizing the loss function, the normalization

$$
d ^ { * } = \frac { d - m ( d ) } { s ( d ) } , \quad \hat { d } ^ { * } = \frac { \hat { d } - m ( \hat { d } ) } { s ( \hat { d } ) } ,\tag{1}
$$

where $m ( d )$ denotes the median of d and s(d) is given by

$$
s ( d ) = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H W } | d _ { i } - m ( d ) | ,\tag{2}
$$

is applied to each map using its own median and scale. We adopt the same normalization for training our model.

## 2.3 Finetuning Strategy

Fine-tuning foundation models and applying it for downstream tasks is an increasing popular approach in machine learning. It is a non-trivial task as naively fully-unfreezing the model and update all parameters is inefficient [30]. As a consequence, researchers focused on developing parameter efficient fine-tuning strategies (PEFT) such as low-rank adaptation (LoRA) [30]. It is a widely used and popular method to update parameters using LoRA in Transformer modules due to their intrinsically low-rank structure of weight updates during fine-tuning. More precisely, low-rank matrices are introduced and added to the existing high-dimensional matrices. Let $W _ { p } \in \breve { \mathbb { R } } ^ { M \times N }$ denote a pre-trained weight matrix that is applied to hidden states $\boldsymbol { x } \in \mathbb { R } ^ { \bar { N } }$ , yielding $y = W _ { p } x$ . The idea is now to inject new parameters with matrices $A \in \mathbb { R } ^ { r \times \bar { N } } , B \in \mathbb { R } ^ { M \times r }$ with $r \ll M , N$ , yielding

$$
y = W _ { p } x + B A x .\tag{3}
$$

The background behind is that the update steps of neural networks even for high dimensional matrices essentially happens in low-rank subspaces [30]. Further, it has the advantage that it does not add any inference latency. Considering its advantages, we choose to update the Transformer-based encoder with LoRA by applying it to the query and value matrices with rank $r = 8 .$ . At the beginning of the training, we set $B = 0$ and A component-wise uniformly distributed, $A _ { i j } \sim \mathcal { U } ( - \frac { 1 } { \sqrt { N } } , \frac { 1 } { \sqrt { N } } )$

Instead of fully-unfreezing the CNN-based decoder network, we selectively unfreeze certain layers. Therefore, we keep the first layers of the decoder, namely the Reassemble layers, frozen. We unfreeze 3 of 4 RefineNet blocks and the output head to reduce an large imbalance between the number of learnable parameters in the encoder and the decoder. In total, we fine-tune $\sim 8$ million parameters out of ∼ 335 million.

We choose a hybrid loss function $L _ { \mathrm { t o t a l } }$ that consists of three partial losses

$$
L _ { \mathrm { t o t a l } } = \alpha L _ { \mathrm { B e r h u } } + \beta L _ { \mathrm { g m } } + \gamma L _ { \mathrm { n o r m } } ,\tag{4}
$$

with hyperparameters $\alpha , \beta , \gamma$ and $L _ { \mathrm { B e r h u } } , L _ { \mathrm { g m } } , L _ { \mathrm { n o r m } }$ as explained next. We base our approach on well-established loss functions for DEM estimation in previous studies. For example, Tao et al. [20] employed the Berhu [31] and a gradient matching loss term, and Chen et al. [21] incorporated a hybrid loss function containing a normal loss term. We apply Eq. 1 to both the prediction and the GT before calculating the loss function. The Berhu loss is defined as

$$
{ \cal L } _ { \mathrm { B e r h u } } ( d ^ { * } , \hat { d } ^ { * } ) = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H W } r _ { i } \mathbb { 1 } _ { \{ r _ { i } \leq \tau \} } + \frac { r _ { i } ^ { 2 } + \tau ^ { 2 } } { 2 \tau } \mathbb { 1 } _ { \{ r _ { i } > \tau \} } ,\tag{5}
$$

where $r _ { i } = | d _ { i } ^ { * } - \hat { d } _ { i } ^ { * } |$ | and $\begin{array} { r } { \tau = { \frac { 1 } { 5 } } \operatorname* { m a x } _ { j } r _ { j } } \end{array}$

The multiscale gradient matching loss $L _ { \mathrm { g m } }$ accounts for the deviation of the residual gradient of the prediction with respect to its corresponding GT. It incorporates downscale operations, that have been proven highly effective in [12].

$$
L _ { \mathrm { g m } } ( d ^ { * } , \hat { d } ^ { * } ) = \frac { 1 } { H W } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { H W } | \nabla _ { x } R _ { i } ^ { ( k ) } | + | \nabla _ { y } R _ { i } ^ { ( k ) } | ,\tag{6}
$$

where $\boldsymbol { R _ { i } } = \boldsymbol { d } _ { i } ^ { * } - \boldsymbol { \hat { d } } _ { i } ^ { * }$ and $R ^ { ( k ) }$ denotes the disparity of the corresponding maps at scale $k ,$ where we set $K = 4$ . The image resolution is halved at each scale level.

The third part is a normal surface loss $L _ { \mathrm { n o r m } }$ that measures the accuracy of the normal to the surface of the prediction and its corresponding GT.

$$
L _ { \mathrm { n o r m } } = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H W } \bigl ( 1 - \frac { \langle n _ { d _ { i } ^ { * } } , n _ { \hat { d } _ { i } ^ { * } } \rangle } { \| n _ { d _ { i } ^ { * } } \| \| n _ { \hat { d } _ { i } ^ { * } } \| } \bigr ) .\tag{7}
$$

The normal vector is given as $\begin{array} { r } { n = ( - \frac { \mathrm { d } z } { \mathrm { d } x } , - \frac { \mathrm { d } z } { \mathrm { d } y } , 1 ) ^ { T } } \end{array}$ . By applying the normalization to the prediction and the GT with Eq. (1), the infinitesimal changes in x and y direction are only a small fraction compared to the third component of the normal. As a consequence, this term would add only marginally to the overall loss. Therefore, we apply a scale j to the first two components of the normal and choose $j = 3 0$ . Henceforth, we refer to the fine-tuned DAV2 model as DEM-DAV2. We empirically choose $\alpha = 1 . 0 , \beta = 2 . 0$ and $\gamma = 0 . 5$

## 2.4 Implementation Details

To make the training procedure more robust and produce more artificial solar azimuth conditions, we apply basic data augmentation strategies. Specifically, we apply flipping in the vertical and horizontal directions, both with a probability of 0.5. In total, we train for 20 epochs with a per-GPU batch size of 8 and train parallel on 4 H100 GPUs with 96 GB RAM each. We apply weight decay of $1 \times 1 0 ^ { - 2 }$ and use the AdamW [32] optimizer. The learning rate for the encoder and decoder are chosen differently, both utilize a linear warmup for the first 10% of the iterations, and are then decayed to zero at full training with a cosine schedule. The initial learning rate for the LoRA weights is set to $1 \times 1 0 ^ { - 4 }$ and the initial learning rate for the decoder to $5 \times 1 0 ^ { - 5 }$ . We conduct experiments with the model that has the lowest validation loss (Eq. 4), which was reached at epoch 13.

## 3 Results

We show that DAV2 can be transformed into a relative depth estimator for the lunar surface. It was used as a state-ofthe-art (SoTA) comparison in the work of La Grassa et al. [23] and Osadnik et al. [24]. We demonstrate improved performance by fine-tuning it on lunar DEM data to create a promising lunar DEM estimator. We compare the results of DAV2 with those of our DEM-DAV2 on our test set.

The test set consists of 13,020 images of size 259×259. As we minimized an scale-and-shift invariant (SSI) loss, both the predictions of DAV2 and DEM-DAV2 are defined up to an affine transformation. To evaluate in absolute metrics, we follow Osadnik et al. [24] and align each prediction $\hat { d } \in \mathbb { R } ^ { 2 5 9 \times 2 5 9 }$ with the corresponding GT elevation map d $\in \mathbb { R } ^ { 2 5 9 \times 2 5 9 }$ (in meters) by

$$
\hat { d } ^ { + } = \sigma _ { \mathrm { G T } } \cdot \frac { \hat { d } - \mu _ { \hat { d } } } { \sigma _ { \hat { d } } } + \mu _ { \mathrm { G T } } ,\tag{8}
$$

with the corresponding mean values $\mu _ { \hat { d } } , \mu _ { \mathrm { G T } }$ and the standard deviations $\sigma _ { \mathrm { G T } }$ and $\sigma _ { \hat { d } } .$ . To evaluate the models’ performances quantitatively, we compute the mean absolute error (MAE) and the root mean squared error (RMSE) between d and $\hat { d } ^ { + }$

$$
\mathrm { M A E } = \frac { 1 } { 2 5 9 ^ { 2 } } \sum _ { i = 1 } ^ { 2 5 9 ^ { 2 } } | d _ { i } - \hat { d _ { i } } ^ { + } | ,\tag{9}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { 2 5 9 ^ { 2 } } \sum _ { i = 1 } ^ { 2 5 9 ^ { 2 } } \big ( d _ { i } - \hat { d _ { i } } ^ { + } \big ) ^ { 2 } } .\tag{10}
$$

![](images/e595d60010db2a7c3b8bec967dfbda0758463e7f218dc0a56ea03463bb694328.jpg)  
Figure 1: The results of the pre-trained Depth Anything V2 (DAV2) model are shown in the left column, compared to the outputs of the proposed DEM-DAV2 model in the right column. The ground truth (GT) is shown in the middle column. We note a significant improvement in the quality of the DEM estimation using the proposed approach.

## 3.1 Qualitative Evaluation

First, we evaluate the performance of DEM-DAV2 by comparing it visually on examples of our test dataset with the zero-shot performance of DAV2. Fig. 1 illustrates a strong performance improvement of DEM-DAV2 compared to DAV2. While DAV2 seems to recognize local high-frequency features, such as craters, it fails to capture the overall relief and structural elevation differences. Here, DEM-DAV2 has a significant advantage, because it captures high-frequency features and excels in determining the relief. Overall, we note that DEM-DAV2 can be used as a reliable relative lunar height estimator. To support that observations, we conduct a profile analysis in Fig. 2. We note that the orange profile, indicating the DEM-DAV2 profile, yields superior results compared to the green line, indicating the DAV2 profile. DAV2 has a solid profile in the second row, but it fails to reconstruct the high-frequency profile present in the first image.

## 3.2 Quantitative Evaluation

The quantitative evaluation on the test set is shown in Table 1. We perform inference on every image in the test set and calculate the metrics per tile. We then take an average over the full test set. We observe, that the DEM-DAV2 yields higher metrics on the test dataset compared to DAV2. This highlights the importance of fine-tuning foundation models for remote sensing tasks, given the domain gap. The improved evaluation metrics confirm the qualitative observations, showing that fine-tuning the pre-trained DAV2 zero-shot model for DEM estimation yields better results than using it without modification.

![](images/8bd82a18b6f46ed311ea8554d51214d37ee590f3a2ceca38918ad1e43cd94ca7.jpg)

![](images/739cfd963f5d91cda9b999cc06852821e0500e3858f37c03439ffce6efd64913.jpg)

![](images/ef8bd86c55d7f2c6a9be7cdd026e5d903f0acde9417f09decb48d1e5787ece7b.jpg)  
Figure 2: Height profile analysis of two distinct images of size 259×259. The images on the left illustrates the ground truth (GT) image, and the red line indicates the profile analysis path. On the right, the green line represents the DAV2 profile and the blue line the corresponding GT profile. In the first example, DAV2 struggles to estimate the height accurately, whereas it yields better results in the second example. In both cases, however, the DEM-DAV2 achieves superior results, as shown by the orange line.

Table 1: Quantitative comparison of DAV2 and DEM-DAV2 on the test dataset. The MAE and RMSE columns show the average values for the entire test dataset.
<table><tr><td>Method</td><td>avg. MAE (m)</td><td>avg. RMSE (m)</td></tr><tr><td>DAV2</td><td>11.64</td><td>14.13</td></tr><tr><td>DEM-DAV2</td><td>4.71</td><td>5.76</td></tr></table>

## 4 Conclusion and Future Work

In this letter, we introduced a fine-tuning strategy for DAV2 for lunar DEM height estimation. DAV2 was previously used without fine-tuning in the work of Osadnik et al. [24] and La Grassa et al. [23] as a SoTA comparison. However, applying foundation models to the task of remote sensing, poses significant limitations due to a domain gap. We addressed that by inserting new learnable parameters to the model encoder with LoRA and unfroze certain layers in the DPT decoder. For training and evaluation, we utilized the public available SPG-derived DEM products. Our results indicate that this enables us to transform DAV2 to a reliable relative height estimator of the lunar surface. Nonetheless, DAV2 still has some limitations. For example, it cannot be used for real-time DEM estimation due to a rather long inference time on a CPU. Additionally, due to the design of our training pipeline, we do not provide auxiliary global DEMs with absolute height information. This limits our approach, meaning it cannot reconstruct absolute heights. Also it is only applicable to the lunar surface. Adaptation to other celestial bodies, such as Mars, would need a new training, as the surface is fundamental different. Future work will incorporate DEM-DAV2 as an auxiliary model for crater detection.

## References

[1] European Space Agency. Argonaut: Europe’s lunar lander programme, 2025. URL: https://www.esa.int/Sc ience\_Exploration/Human\_and\_Robotic\_Exploration/Exploration/Argonaut\_Europe\_s\_lunar\_ lander\_programme.

[2] Atal Tewari, Vinay Verma, Pradeep Srivastava, Vikrant Jain, and Nitin Khanna. Automated Crater detection from Co-registered optical images, elevation maps and slope maps using deep learning. Planetary and Space Science, 218:105500, 2022. doi:10.1016/j.pss.2022.105500.

[3] Chen Zou, Jialong Lai, Yanshuang Liu, Feifei Cui, Yi Xu, and Le Qiao. Small lunar crater identification and age estimation in chang’e-5 landing area based on improved faster r-cnn. Icarus, 410:115909, 2024. URL: https://www.sciencedirect.com/science/article/pii/S0019103523004888, doi:10.1016/j.ic arus.2023.115909.

[4] Patrick Bauer, Marius Schwinning, Florian Renk, Andreas Weinmann, and Hichem Snoussi. Vision-Language Model for Accurate Crater Detection. IEEE Transactions on Aerospace and Electronic Systems, pages 1–16, 2026. doi:10.1109/TAES.2026.3725640.

[5] Atal Tewari, Kumar Prateek, Amrita Singh, and Nitin Khanna. Deep learning based systems for crater detection: A review. ArXiv, abs/2310.07727, 2023. URL: https://api.semanticscholar.org/CorpusID:263908778.

[6] M. S. Robinson, S. M. Brylow, M. Tschimmel, D. Humm, S. J. Lawrence, P. C. Thomas, B. W. Denevi, E. Bowman-Cisneros, J. Zerr, M. A. Ravine, M. A. Caplinger, F. T. Ghaemi, J. A. Schaffner, M. C. Malin, P. Mahanti, A. Bartels, J. Anderson, T. N. Tran, E. M. Eliason, A. S. McEwen, E. Turtle, B. L. Jolliff, and H. Hiesinger. Lunar Reconnaissance Orbiter Camera (LROC) Instrument Overview. Space Science Reviews, 150(1–4):81–124, January 2010. doi:10.1007/s11214-010-9634-2.

[7] Oleg Alexandrov and Ross A. Beyer. Multiview shape-from-shading for planetary images. Earth and Space Science, 5(10):652–666, October 2018. URL: http://dx.doi.org/10.1029/2018EA000390, doi:10.1029/ 2018ea000390.

[8] M.R. Henriksen, M.R. Manheim, K.N. Burns, P. Seymour, E.J. Speyerer, A. Deran, A.K. Boyd, E. Howington-Kraus, M.R. Rosiek, B.A. Archinal, and M.S. Robinson. Extracting accurate and precise topography from lroc narrow angle camera stereo observations. Icarus, 283:122–137, February 2017. URL: http://dx.doi.org/10. 1016/j.icarus.2016.05.012, doi:10.1016/j.icarus.2016.05.012.

[9] Robert V. Wagner, Megan R. Henriksen, Madeleine R. Manheim, Heather M. Meyer, Maria E. Banks, Erick Malaret, Erwan Mazarico, Emerson J. Speyerer, and Mark S. Robinson. Where is that crater? best practices for obtaining accurate coordinates from lroc nac data. The Planetary Science Journal, 5(7):157, jul 2024. doi:10.3847/PSJ/ad54c6.

[10] Yang Liu, Yexin Wang, Kaichang Di, Man Peng, Wenhui Wan, and Zhaoqin Liu. A generative adversarial network for pixel-scale lunar dem generation from high-resolution monocular imagery and low-resolution dem. Remote Sensing, 14(21):5420, October 2022. URL: http://dx.doi.org/10.3390/rs14215420, doi: 10.3390/rs14215420.

[11] Jonathan T. Barron and Jitendra Malik. High-frequency shape and albedo from shading using natural image statistics. In CVPR 2011, pages 2521–2528, 2011. doi:10.1109/CVPR.2011.5995392.

[12] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44:1623–1637, 2019. URL: https://api.semanticscholar.org/CorpusID: 195776274.

[13] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9492–9502, 2023. URL: https://api.semanticsc holar.org/CorpusID:265609019.

[14] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10371–10381, 2024. URL: https://api.semanticscholar.org/CorpusID: 267061016.

[15] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. ArXiv, abs/2406.09414, 2024. URL: https://api.semanticscholar.org/CorpusID: 270440448.

[16] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017. URL: https://api.semanticscholar.org/CorpusID:13756489.

[17] Danfeng Hong, Chenyu Li, Xuyang Li, Gustau Camps-Valls, and Jocelyn Chanussot. Foundation models in remote sensing: Evolving from unimodality to multimodality. IEEE Geoscience and Remote Sensing Magazine, 14(2):10–35, 2026. / .

[18] Muying Luo, Tao Zhang, Shiqing Wei, and Shunping Ji. Sam-rsis: Progressively adapting sam with box prompting to remote sensing image instance segmentation. IEEE Transactions on Geoscience and Remote Sensing, 62:1–14, 2024. doi:10.1109/TGRS.2024.3460085.

[19] Tianhao Chen, Yexin Wang, Jing Nan, Chenxu Zhao, Biao Wang, Bin Xie, Wai-Chung Liu, Kaichang Di, Bin Liu, and Shaohua Chen. A generative adversarial network for pixel-scale lunar dem generation from single high-resolution image and low-resolution dem based on terrain self-similarity constraint. Remote Sensing, 17(17), 2025. URL: https://www.mdpi.com/2072-4292/17/17/3097, doi:10.3390/rs17173097.

[20] Yu Tao, Siting Xiong, Susan J. Conway, Jan-Peter Muller, Anthony Guimpier, Peter Fawdon, Nicolas Thomas, and Gabriele Cremonese. Rapid single image-based dtm estimation from exomars tgo cassis images using generative adversarial u-nets. Remote Sensing, 13(15), 2021. URL: https://www.mdpi.com/2072-4292/13/15/2877, doi:10.3390/rs13152877.

[21] Hao Chen, Xuanyu Hu, Philipp Gläser, Haifeng Xiao, Zhen Ye, Hanyue Zhang, Xiaohua Tong, and Jürgen Oberst. Cnn-based large area pixel-resolution topography retrieval from single-view lroc nac images constrained with sldem. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 15:9398–9416, 2022. doi:10.1109/JSTARS.2022.3214926.

[22] Hao Chen, Philipp Gläser, Xuanyu Hu, Konrad Willner, Yongjie Zheng, Friedrich Damme, Lorenzo Bruzzone, and Jürgen Oberst. Elunardtmnet: Efficient reconstruction of high-resolution lunar dtm from single-view orbiter images. IEEE Transactions on Geoscience and Remote Sensing, 62:1–20, 2024. doi:10.1109/TGRS.2024.3501153.

[23] Riccardo La Grassa, Cristina Re, Adriano Tullo, Ignazio Gallo, and Gabriele Cremonese. Transformer-driven monocular high-resolution dtm generation on mars via multimodal integration of cassis imagery and mola altimetry. ISPRS Open Journal ofPhotogrammetry and Remote Sensing, 19:100118, 2026. URL: https://www.scienc edirect.com/science/article/pii/S2667393226000049, doi:10.1016/j.ophoto.2026.100118.

[24] Rafał Osadnik, Pablo Gómez, Eleni Bohacek, and Rickbir Bahia. Mcted: A machine-learning-ready dataset for digital elevation model generation from mars imagery, 2025. URL: https://arxiv.org/abs/2509.08027, arXiv:2509.08027.

[25] Michael K. Barker, Erwan M. Mazarico, Gregory A. Neumann, Maria T. Zuber, Jun’ichi Haruyama, and David E. Smith. A new lunar digital elevation model from the lunar orbiter laser altimeter and selene terrain camera. Icarus, 273:346–355, 2015. URL: https://api.semanticscholar.org/CorpusID:117043230.

[26] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. ArXiv, abs/2304.07193, 2023. URL: https: //api.semanticscholar.org/CorpusID:258170077.

[27] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12159–12168, 2021. URL: https://api.semant icscholar.org/CorpusID:232352612.

[28] Guosheng Lin, Anton Milan, Chunhua Shen, and Ian D. Reid. Refinenet: Multi-path refinement networks for high-resolution semantic segmentation. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5168–5177, 2016. URL: https://api.semanticscholar.org/CorpusID:5696978.

[29] Ke Xian, Chunhua Shen, ZHIGUO CAO, Hao Lu, Yang Xiao, Ruibo Li, and Zhenbo Luo. Monocular relative depth perception with web stereo data supervision. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 311–320, 2018. URL: https://api.semanticscholar.org/CorpusID:52860134.

[30] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models, 2021. URL: https://arxiv.org/abs/2106 .09685, doi:10.48550/ARXIV.2106.09685.

[31] Laurent Zwald and Sophie Lambert-Lacroix. The berhu penalty and the grouped effect. arXiv: Statistics Theory, 2012. URL: https://api.semanticscholar.org/CorpusID:88519298.

[32] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017.