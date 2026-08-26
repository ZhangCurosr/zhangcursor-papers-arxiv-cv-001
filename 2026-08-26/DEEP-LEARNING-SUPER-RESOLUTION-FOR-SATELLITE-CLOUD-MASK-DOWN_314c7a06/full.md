# DEEP LEARNING SUPER RESOLUTION FOR SATELLITE CLOUD MASK DOWNSCALING

Angelos Georgakis<sup>∗</sup>, Valentina Kanaki<sup>∗</sup>, Giorgos Giannopoulos<sup>∗</sup>, Stella Girtsou<sup>∗</sup>,

Ioannis Kontogiorgakis<sup>∗</sup>, Charalampos Kontoes<sup>∗</sup>, Kostas Philippopoulos<sup>†</sup>

<sup>∗</sup> BEYOND EO Centre, IAASARS, National Observatory of Athens, Athens, Greece

{ageorgakis, valekan, giannopoulos, sgirtsou, ikontog, kontoes}@noa.gr

<sup>†</sup> National and Kapodistrian University of Athens, Faculty of Physics, Section of Environmental Physics and Meteorology,

Athens, Greece

kphilip@phys.uoa.gr

Abstract—A vast amount of optical satellite data is being transmitted to Earth-based servers every day, and more than half of this data is affected by haze or clouds. Additionally, this data suffers from the fundamental trade-off between spatial and temporal resolution, which remains largely unresolved, making the acquisition of continuous high-resolution satellite observations of clouds an ongoing challenge. This work addresses this challenge by proposing two Deep Learning super-resolution methods for the accurate downscaling of SEVIRI cloud mask products, as well as a novel cross-sensor cloud mask dataset called SEVMOD-CM, created by spatially and temporally matching MODIS and SE-VIRI satellite observations. The two proposed models are a CNNbased (SpatialCNN) and a GAN-based (SpatialGAN) Neural Network. Trained on the SEVIRI spectral and cloud mask products, the proposed methods predict the corresponding MODIS Cloud masks, achieving a 4× spatial enhancement across sensor domains. Both approaches are evaluated experimentally, and compared against the standard bicubic interpolation upsampling technique. The experimental results demonstrate the value of the proposed models and dataset for the remote sensing community, highlighting the benefits of applying super-resolution techniques to geostationary-derived cloud mask products for applications such as atmospheric monitoring, weather forecasting, disaster risk reduction, solar energy forecasting, and climate research. Index Terms—Deep Learning, Super-Resolution, Downscaling, Cloud Mask, SEVIRI, MODIS.

## I. INTRODUCTION

Over the past decades remote sensing (RS) imagery has emerged as a central field of research, with crucial implications in applications such as Earth observation (EO) and environmental monitoring. Since 67% of the globe is covered<sup>˜</sup> by haze or clouds [1], optical RS retrievals are consequently greatly affected due to the scattering and attenuation of the electromagnetic signals leading to a deficiency of surface information [2]. Satellite retrievals of cloud formations that are of both high temporal and spatial resolution, are crucial for many application such as solar energy forecasting, weather prediction and disaster management, as many of such application utilize the cloud mask (CM) products derived from Geostationaty satellites such as MSG. However, due to technical limitations, such as sensor size, orbital altitude, and data transmission bandwidth, the persistent trade-off between spatial and temporal resolution (no sensor can capture information both at the highest possible spatial and temporal resolution across all wavebands) remains a significant challenge in the field of RS, that greatly affects all the CM dependent applications. Furthermore, cloud misclassifications reported in existing cloud mask products [3], [4], together with the limited availability of high-resolution imagery for cloud removal (CR) applications [5], further stretch the need for accurate highresolution CM products. Obtaining CM products of high temporal and spatial resolution has also been emphasized in studies on solar radiation nowcasting and forecasting [6], and on cloud-gap filling for applications such as grassland mowing event detection [7] among others.

Super-resolution (SR) is a well established set of algorithmic techniques in the field of computer vision and image processing that aim to improve the visual analysis of images. The term Super-Resolution is often used as an umbrella term to include many image upsampling techniques that have been widely categorized as, Interpolation-based [8], [9], Reconstruction-based [10], [11] and Learning-based [12] [13] algorithms. A widely adopted interpolation method is Bicubic interpolation [14], as it is a simple and explainable type of image upsampling that produces smooth HR images, however, it is known to suffer from the chessboard effect [15]. Reconstruction-based SR algorithms on the other hand, degrade rapidly when given large magnification factors or relatively few input images. In these cases, the result may be lacking important high-frequency details [16]. Therefore the emphasis is being given in the advancement of DL SR algorithms. Various such approaches have been proposed, which can be roughly categorized in the following two groups: Convolutional Neural Network (CNN)-based and Generative Adversarial Network (GAN)-based methods, each one with its own advantages and disadvantages [17]. The CNN-based methods originate from the SRCNN paper [18], where a 3-layer CNN was proposed to perform SR on all 3 color channels simultaneously, surpassing the then SotA methods. Soon after a GAN-based network for image SR was proposed [19], (SRGAN), obtaining better overall structural similarity. Since then, these models have been widely adjusted to adapt to different SR RS tasks [20], [21], but continue to provide the fundamental approaches on SR [22].

Despite the widespread application of SR in different RS tasks, the cross-sensor CM SR task is one that remains largely unexplored. In this paper we introduce two advanced DL models for cloud mask SR, as well as a novel DL-ready crosssensor CM dataset, called SEVMOD-CM, and demonstrate its applicability. The proposed models are drawn from the two major SR DL families: a CNN-based model (SpatialCNN) and a GAN-based model (SpatialGAN). In both architectures, we implement task specific extensions and enhancements (detailed in Section II-B) to enable effective learning that derive from the key characteristics of the CM products they are trained on. The models are implemented and evaluated against the standard bicubic interpolation upsampling method as a baseline. For the dataset creation, data from satellite sensors which differ in spatial, spectral, and temporal resolutions are combined. This data is derived from: (a) the MODIS sensor onboard the Aqua and Terra satellites, used as the High-Resolution (HR) ground truth for model training and (b) the SEVIRI instrument aboard the MSG satellite, serving as the Low-Resolution (LR) input to the SR models. These models are expected to significantly enhance the ability to detect and monitor cloud formations in near real time, thereby supporting faster responses to extreme weather events and mitigating their potential impacts. Moreover, improved CM spatial resolution can contribute to more accurate solar radiation monitoring and forecasting, facilitating its more effective integration into the power grid. To the best of the authors’ knowledge, this is the first effort to apply DL SR methods directly to the CM products of different satellite sensors.

The key contributions of this study are summarized as follows.

• we implement two advanced SR algorithms in order to improve the spatial analysis of the Cloud Mask product generated by the SEVIRI sensor aboard the MSG geostationary satellite

• we create SEVMOD-CM, a comprehensive cross sensor cloud mask oriented, AI-ready dataset, and demonstrate its usability

• we evaluate the proposed algorithms, and explore limitations and possible future research directions

## II. MATERIALS AND METHODS

Regarding the terminology used in this paper, as is usually the case in EO and RS research, the term “downscale” refers to the transition from low to high resolution, indicating the decrease of the actual spatial trace of a pixel on the map, while in computer vision studies the same process is described by the word ”upscale”, referring to the increase in image analysis with ”downscale” indicating the opposite process. The words ”upsampling” and ”downsampling” are also used instead of upscaling/downscaling. In this paper the term “upsample” is being prevalently used.

In this section, the design choices and implementation steps that were carried out are analyzed. Specifically the four basic steps below will be addressed:

• Finding and downloading the necessary data

• Data postprocessing and SEVMOD-CM creation

• Selection and implementation of SR DL models

## • SR result evaluation

These stages are presented in more detail below, while a clear roadmap of the dataset creation methodology is presented

## A. SEVMOD-CM DATASET

1) DATA ACQUISITION: For the development of the SEVMOD-CM dataset, paired low- and high-resolution satellite observations were collected from the SEVIRI instrument onboard Meteosat Second Generation (MSG) and the MODIS instruments onboard Aqua and Terra. Specifically, the Seviri 11 spectral channels and the CM product of 3km spatial resolution, which was separately downloaded but later processed in a combined netcdf file, as well as the Modis CM product of 1km spatial resolution, spanning 2020–2023 and covering Europe and North Africa. MODIS CM products (MOD35 L2 for Terra and MYD35 L2 for Aqua) were used as the high-resolution reference, and their acquisition times were used to define the SEVIRI candidate timestamps, within a time window of 15 minutes, to account for the much higher temporal sampling of MSG. MODIS data were retrieved from NASA Earthdata <sup>1</sup> and SEVIRI data from EUMETSAT <sup>2</sup>, resulting in a dataset of approximately 200 GB stored in a local server.

2) DATA PROCESSING: The downloaded MODIS and SEVIRI products were preprocessed to generate paired image patches, that are temporally synchronized and spatially aligned, and suitable for the training of DL SR algorithms. The MODIS CM product initially provides a large set of ancillary flags and test results encoded in 48-bit representation. By applying bitwise operators it was decoded and converted into a binary cloud mask, while the SEVIRI CM that initially contained categorical values (0: clear sky over water, 1: clear sky over land, 2: clouds, 3: raw or no data points, i.e. areas outside the Earth’s disk) indicating cloud conditions, was similarly binarized and appended as a twelfth SEVIRI channel. SEVIRI products, originally provided in a geostationary projection, were reprojected to EPSG:4326 to match the MODIS Coordinate Reference System (CRS) and allow for precise geolocation and spatial overlapping tests. Spatially overlapping regions were then identified and fixed 128 × 128 MODIS patches were extracted. Thus spatially corresponding SEVIRI patches were cropped, but not initially forced on a specific image size, to allow for covering the exactly same geographic region as MODIS. The cropped SEVIRI patches were then resampled to a canonical 32 × 32 grid, and paired with the MODIS ones. Invalid or empty patch pairs were discarded. The resulting paired patches were stored as GeoTIFF files with full metadata and split into training, validation, and test sets using an 80/10/10 ratio.

## B. Super Resolution Deep Learning Models

For the task of SR, we developed two distinct DL architectures to specifically address cross-sensor and crossresolution discrepancies between geostationary SEVIRI and polar-orbiting MODIS observations. The task was formulated as a computer vision supervised problem, where we mapped LR multi-spectral SEVIRI radiances and the associated cloud masks, to the spatially and temporally corresponding HR MODIS cloud masks. The first model is SpatialCNN, depicted in Fig. 2. It is a deep CNN model that extends traditional CNN SR approaches [18] by incorporating residual learning and progressive upsampling. These architectural configurations are based on thorough RS SR domain knowledge, and are proven to enhance image reconstruction fidelity [23], [24], [25], [26], [27], [28], [29]. On top of that, in order to adjust the model to the task of binary CM SR, the proposed architecture also supports a flexible number of input channels, allowing the network to ingest between one and twelve SEVIRI bands, or any arbitrary subset in between, thus enabling the systematic evaluation of spectral contribution to CM SR. Finally to adjust to the binary nature of the MODIS target cloud mask, BCE was the chosen loss function, while a sigmoid-bounded single-channel output constrains predictions to probabilistic CM values.

![](images/f48b4c303b4b49a19ad1b89623a4085245565d977c4da1d9d1a8248fef7d6930.jpg)  
Fig. 1. SEVMOD-CM Dataset creation methodology

The second model, SpatialGAN 3, is a GAN model based on SRGAN [19], but specifically adjusted to the binary CM SR task. While also modified to work with a flexible number of SEVIRI input channels, the key modification comes in the composite loss function of SpatialGAN. The SRGAN framework was originally proposed for photo-realistic image reconstruction, thus its loss formulation required reinterpretation for binary cloud mask SR. We extended it by adding a BCE term to explicitly enforce pixel-wise cloud classification consistency and faster convergence, while the initial perceptual and adversarial losses were kept but downgraded by adding a weight factor, to act as regularizers that encourage spatial coherence.

Model performance was evaluated on a held-out test set using Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (MSE), and Structural Similarity Index (SSIM), which were selected to quantify the reconstruction error and the spatial consistency of the unavoidable cross-sensor image misalign ment.

![](images/b7d518dc118cf4ef8e53483ba086eb9de22fe7de7bca33863ce1c404e510a911.jpg)

Fig. 2. SpatialCNN architecture  
![](images/6add6f42394f4c1fec9c7bb935375f7d4a493002f6f599d2008a835ad40e1f8b.jpg)  
Fig. 3. SpatialGAN architecture

## III. RESULTS AND DISCUSSION

Dataset. The present work resulted in the creation of the SEVMOD-CM Dataset, consisting of 4.8GB of data, with a total of 126216 .tif files, meaning 63108 complete MODIS-SEVIRI pairs. All the SEVIRI patches, contained 12 bands, 11 spectral and the binarized Cloud Mask product, while the MODIS, serving as the model target, has just the binary CM created during the preprocessing phase. Of these initial 63108 pairs, after filtering out the ones that contained no clouds at all 20828 final paired patches (33% retention rate) were kept

TABLE I  
QUANTITATIVE EVALUATION OF SUPER-RESOLUTION METHODS ON THE SEVMOD-CM TEST SET.
<table><tr><td>Metric</td><td>SpatialCNN</td><td>SpatialGAN</td><td>Bicubic Interpolation</td></tr><tr><td>PSNR (dB)</td><td>17.99</td><td>17.9</td><td>18.32</td></tr><tr><td>SSIM</td><td>0.534</td><td>0.7811</td><td>0.578</td></tr><tr><td>MSE</td><td>0.173</td><td>0.0480</td><td>0.12</td></tr></table>

to feed into the networks.

Evaluation Setting. The implementation of our models was done using the PyTorch framework. Initial experimentation on the training dataset lead us to the following hyperparameter configurations. Regarding SpatialCNN, the set of SEVIRI input channels that performed best is: 2,4,6,12. The network was trained for 29 epochs as early-stopping was implemented. LeakyReLU was chosen as the activation function of the hidden layers, while for the output layer the sigmoid function was used to constrain outputs to [0,1]. Adam was the selected optimizer algorithm, while a Batch size of 8 was used. During inference a threshold of 0.5 was used in order to retrieve the cloud masks in binary form, thus comparable to the SEVIRI provided ones.

Regarding the SpatialGAN model, we trained the network for 100 epochs using SEVIRI channels 2, 4, 6, and 12 as input as well, with a batch size of 16. Both the generator and discriminator were optimized using the Adam optimizer with learning rates of $1 \times 1 0 ^ { - 4 }$ . Finally to reduce discriminator overconfidence and stabilize the adversarial training process, the discriminator was updated less often than the generator with a 5:1 ratio, and a label smoothing of 0.9 was applied, helping prevent discriminator dominance. As for the baseline bicubic interpolation method, it was applied using PyTorch’s default kernel parameters (Keys’ cubic convolution kernel with a=-0.75)<sup>3</sup>.

Evaluation Results. The experimental findings in Table I demonstrate the comparative performance of the applied SR models on the held-out test set of the SEVMOD-CM dataset. As shown in Table I, SpatialGAN achieved by far the highest SSIM, as well as the lowest MSE values, while its PSNR was marginally lower compared to the other methods. On the other hand SpatialCNN performed similarly but slightly worse than bicubic inerpolation across all 3 evaluation metrics, surpassing only SpatialGAN in PSNR.

Overall, SpatialGAN succeeded in producing sharper and more perceptually accurate reconstructions than SpatialCNN, while when compared to the bicubic interpolation baseline approach it achieved slighlty lower PSNR but significantly higher SSIM and lower MSE. This indicates that while it might have missed some high frequency details, the overall structural similarity of the reconstructed cloud masks was greatly improved, highlighting the advantages of adversarial training for SR of satellite imagery. This is evident in figure 4, where the clear structural enhancement of the SEVIRI cloud shapes is visually demonstrated. While the MODIS targets complex and scattered cloud patterns result in lower numerical metrics, we see that SpatialGAN has successfully learned to reconstruct the overall cloud structure and distribution, demonstrating the visual fidelity and perceptual quality of the model’s outcomes.

Regarding the limitations of the method, the inherent spatial, temporal and spectral misalignment in cross-sensor imagery, remains a fundamental challenge that image upsampling methods cannot yet fully resolve. On the other hand, future research will investigate advanced DL architectures, including Vision Transformers and diffusion models to better address the crosssensor SR challenge, as well as applying them to different cloud related products.

![](images/249e1dcca7f8d75a2bc2ff1f8202ae584cc5482956a5f68c919358b39a4f3ba4.jpg)  
Fig. 4. Test results of SpatialGAN super-resolved images (middle), alongside the initial SEVIRI CM (left) and the target MODIS CM (right)

## IV. CONCLUSION

This paper presented a novel approach for cross-sensor satellite CM SR, using SEVIRI and MODIS satellite imagery and DL architectures. We developed two distinct advanced DL SR models, as well as a task specific dataset, and showcased their applicability. Specific model configurations exhibited much higher structural similarity and reconstruction accuracy compared to the baseline method, highlighting the effectiveness of GAN-based models in such tasks. These findings emphasize on the potential of SR methods utilizing geostationary observations as a valuable tool for near real-time cloud monitoring.

## REFERENCES

[1] M. D. King, S. Platnick, W. P. Menzel, S. A. Ackerman, and P. A. Hubanks, “Spatial and temporal distribution of clouds observed by modis onboard the terra and aqua satellites,” IEEE Transactions on Geoscience and Remote Sensing, vol. 51, no. 7, pp. 3826–3852, 2013.

[2] J. Gawlikowski, P. Ebel, M. Schmitt, and X. X. Zhu, “Explaining the effects of clouds on remote sensing scene classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 9976–9986, 2022.

[3] H. Letu, T. M. Nagao, T. Y. Nakajima, and Y. Matsumae, “Method for validating cloud mask obtained from satellite measurements using ground-based sky camera,” Appl. Opt., vol. 53, no. 31, pp. 7523–7533, Nov 2014. [Online]. Available: https://opg.optica.org/ao/abstract.cfm?URI=ao-53-31-7523

[4] P. Leinenkugel, C. Kuenzer, and S. Dech, “Comparison and enhancement of modis cloud mask products for southeast asia,” International Journal of Remote Sensing, vol. 34, no. 8, pp. 2730–2748, 2013. [Online]. Available: https: //doi.org/10.1080/01431161.2012.750037

[5] F. Xu, Y. Shi, P. Ebel, W. Yang, and X. X. Zhu, “Multimodal and multiresolution data fusion for high-resolution cloud removal: A novel baseline and benchmark,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[6] K. Papachristopoulou, I. Fountoulakis, A. F. Bais, B. E. Psiloglou, N. Papadimitriou, I.-P. Raptis, A. Kazantzidis, C. Kontoes, M. Hatzaki, and S. Kazadzis, “Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting,” Atmospheric Measurement Techniques, vol. 17, no. 7, pp. 1851–1877, 2024. [Online]. Available: https://amt.copernicus.org/articles/17/1851/2024/

[7] I. Tsardanidis, A. Koukos, V. Sitokonstantinou, T. Drivas, and C. Kontoes, “Cloud gap-filling with deep learning for improved grassland monitoring,” Computers and Electronics in Agriculture, vol. 230, p. 109732, 2025. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0168169924011232

[8] F. Zhou, W. Yang, and Q. Liao, “Interpolation-based image super-resolution using multisurface fitting,” IEEE Transactions on Image Processing, vol. 21, no. 7, pp. 3312–3318, 2012.

[9] Y. Zhang, Q. Fan, F. Bao, Y. Liu, and C. Zhang, “Single-image super-resolution based on rational fractal interpolation,” IEEE Transactions on Image Processing, vol. 27, no. 8, pp. 3782– 3797, 2018.

[10] Q. Yang, Y. Zhang, T. Zhao, and Y. Chen, “Single image super-resolution using self-optimizing mask via fractional-order gradient interpolation and reconstruction,” ISA Transactions, vol. 82, pp. 163–171, 2018, fractional Order Signals, Systems, and Controls: Theory and Application. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0019057817303427

[11] Y. Huang, J. Li, X. Gao, L. He, and W. Lu, “Single image superresolution via multiple mixture prior models,” IEEE Transactions on Image Processing, vol. 27, no. 12, pp. 5904–5917, 2018.

[12] T. Liu, K. de Haan, Y. Rivenson, Z. Wei, X. Zeng, Y. Zhang, and A. Ozcan, “Deep learning-based super-resolution in coherent imaging systems,” Scientific Reports, vol. 9, no. 1, p. 3926, 2019.

[13] W. Yang, X. Zhang, Y. Tian, W. Wang, J.-H. Xue, and Q. Liao, “Deep learning for single image super-resolution: A brief review,” IEEE Transactions on Multimedia, vol. 21, no. 12, pp. 3106–3121, 2019.

[14] R. Keys, “Cubic convolution interpolation for digital image processing,” IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153–1160, 1981.

[15] S. Dai, M. Han, W. Xu, Y. Wu, and Y. Gong, “Soft edge smoothness prior for alpha channel super resolution,” in 2007 IEEE Conference on Computer Vision and Pattern Recognition, 2007, pp. 1–8.

[16] S. Baker and T. Kanade, “Limits on super-resolution and how to break them,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 9, pp. 1167–1183, 2002.

[17] J. Guo, F. Lv, J. Shen, J. Liu, and M. Wang, “An improved generative adversarial network for remote sensing image super-resolution,” IET Image Processing, vol. 17, no. 6, pp. 1852–1863, 2023. [Online]. Available: https://ietresearch. onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.12760

[18] C. Dong, C. C. Loy, K. He, and X. Tang, “Image superresolution using deep convolutional networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 295–307, 2016.

[19] C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi, “Photo-realistic single image super-resolution using a generative adversarial network,” 2017. [Online]. Available: https://arxiv.org/abs/1609.04802

[20] Y. Gong, P. Liao, X. Zhang, L. Zhang, G. Chen, K. Zhu, X. Tan, and Z. Lv, “Enlighten-gan for super resolution reconstruction in mid-resolution remote sensing images,” Remote Sensing, vol. 13, no. 6, 2021. [Online]. Available: https://www.mdpi.com/2072-4292/13/6/1104

[21] S. Lei, Z. Shi, and Z. Zou, “Super-resolution for remote sensing images via local–global combined network,” IEEE Geoscience and Remote Sensing Letters, vol. 14, no. 8, pp. 1243–1247, 2017.

[22] X. Wang, J. Yi, J. Guo, Y. Song, J. Lyu, J. Xu, W. Yan, J. Zhao, Q. Cai, and H. Min, “A review of image super-resolution approaches based on deep learning and applications in remote sensing,” Remote Sensing, vol. 14, no. 21, 2022. [Online]. Available: https://www.mdpi.com/2072-4292/14/21/5423

[23] J. Liu, J. Tang, and G. Wu, “Residual feature distillation network for lightweight image super-resolution,” in Computer Vision – ECCV 2020 Workshops, A. Bartoli and A. Fusiello, Eds. Cham: Springer International Publishing, 2020, pp. 41–55.

[24] D. Liu, J. Li, and Q. Yuan, “A spectral grouping and attentiondriven residual dense network for hyperspectral image superresolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 9, pp. 7711–7725, 2021.

[25] L. Sun, Z. Liu, X. Sun, L. Liu, R. Lan, and X. Luo, “Lightweight image super-resolution via weighted multi-scale residual network,” IEEE/CAA Journal of Automatica Sinica, vol. 8, no. 7, pp. 1271–1280, 2021.

[26] K. Park, J. W. Soh, and N. I. Cho, “A dynamic residual self-attention network for lightweight single image superresolution,” IEEE Transactions on Multimedia, vol. 25, pp. 907– 918, 2023.

[27] W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop,´ D. Rueckert, and Z. Wang, “Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 1874–1883.

[28] Y. Wang, W. Liu, W. Sun, X. Meng, G. Yang, and K. Ren, “A progressive feature enhancement deep network for large-scale remote sensing image superresolution,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–13, 2023.

[29] Y. Wang, F. Perazzi, B. McWilliams, A. Sorkine-Hornung, O. Sorkine-Hornung, and C. Schroers, “A fully progressive approach to single-image super-resolution,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2018.