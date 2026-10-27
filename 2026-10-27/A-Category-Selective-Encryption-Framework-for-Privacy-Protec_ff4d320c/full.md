# A Category-Selective Encryption Framework for Privacy Protection of UAV Images

1<sup>st</sup> Ting Yan

2<sup>nd</sup> Minghui Shang

3<sup>rd</sup> Haolang Sun

Dartmouth College School of Information Science and Engineering School of Information Science and Engineering

Hanover, NH 03755, USA

University of Jinan

University of Jinan

ting.yan.gr@dartmouth.edu

Jinan 250022, China

Jinan 250022, China

2870062251@qq.com

lannistersun@126.com

4<sup>th</sup> Xu Wang

5<sup>th</sup> Yamin Han\*

6<sup>th</sup> Mingze He

School of Information Science and Engineering

Quancheng Laboratory Department of Computer Science and Technology

University of Jinan

Jinan 250100, China

Jinan 250022, China

Sichuan Police College

ise wangx@ujn.edu.cn

hr-hanym@qcl.edu.cn

Luzhou, 646000, China

mingzebox@163.com

Abstract—Imagery captured by unmanned aerial vehicle (UAV) systems increasingly raises privacy concerns, as UAV aerial surveillance routinely captures sensitive objects such as pedestrians and vehicles alongside ordinary scene content. Existing privacy-preserving approaches typically employ class-agnostic saliency detection to indiscriminately obfuscate all visually salient regions, leading to unnecessary information loss and hindering downstream scene analysis. This paper proposes a category-selective encryption framework for privacy-preserving UAV surveillance. The core advance over prior class-agnostic approaches is semantic-level selectivity: the system first recognizes what is present in the scene via an 80-class detector, then encrypts only user-specified sensitive categories while leaving non-sensitive objects and background intact for downstream analytics. The framework supports two complementary encryption schemes: Thumbnail-Preserving Encryption (TPE), which applies adaptive pixelation with a learnable blend parameter and recovers content through a jointly trained Enhanced U-Net, and XOR stream cipher encryption, which offers parameter-free, mathematically exact reversibility. The architectural design, composite loss formulation, and joint training methodology are presented in detail, establishing a foundation for practical privacy-preserving UAV surveillance that balances protection strength and scene utility.

Index Terms—Category-selective Encryption, Privacy Protection for UAV Images, Object Detection, TPE, XOR Stream Cipher

## I. INTRODUCTION

With the rapid development of the low-altitude economy, UAV aerial photography, as a high-efficiency data acquisition technology with high-resolution imaging capability, has been widely used in agricultural monitoring [1], urban planning [2], and environmental inspection [3]. However, such aerial images usually contain sensitive private information including human faces, pedestrians, and vehicle license plates. If these unprocessed images are leaked during air-to-ground wireless transmission or cloud storage, personal privacy will be exposed, which will lead to privacy breaches [4]. Moreover, with the continuous advancement of computer vision technologies, deep learning algorithms can accurately identify human contours and behavioral activities in aerial images [5].

To mitigate the privacy risks of aerial images, researchers have put forward a variety of image privacy protection schemes [6]. In recent years, a growing number of dedicated encryption algorithms have been developed specifically for UAV images. Shafique et al. [7] proposed an encryption scheme combining DNA encoding and bit-plane extraction to secure high-resolution UAV-captured images during transmission. Zhou et al. [8] designed an encryption and compression algorithm based on two-dimensional hyperchaos for aerial images collected by agricultural UAVs. For smart agriculture scenarios, Zhou et al. [9] constructed an image protection scheme that optimizes chaotic S-boxes via simulated annealing and affine transformation. Aiming at facial privacy protection in UAV remote sensing images, Wen et al. [10] built a hybrid encryption framework integrating chaotic systems and DNA cryptography. Nevertheless, nearly all of the above schemes adopt a single fixed encryption mode. They fail to achieve discriminative selective encryption for semantic targets such as pedestrians, vehicles and public background areas, so they cannot adapt to diverse UAV patrol scenarios. Other mainstream privacy protection schemes rely on universal visual saliency detection, which applies blurring and encryption uniformly to all salient regions in an image [11]. These schemes cause massive loss of valid scene information and severely interfere with subsequent data analysis.

To meet the requirements of personal information protection laws and low-altitude data security regulations, there is an urgent need to develop a UAV aerial image privacy protection scheme that supports category-selective processing while balancing privacy protection and scene usability. To this end, this paper proposes a category-selective encryption framework for UAV images. The framework integrates multi-category object detection to achieve semantic-level selective encryption and incorporates two complementary and switchable encryption strategies. It constructs a complete closed-loop processing pipeline of object detection, category-selective encryption, decryption and restoration, while maximizing the preservation of complete scene information under privacy security guarantees.

The main contributions of this paper are summarized as follows:

1) The proposed framework employs a YOLO11 multicategory object detector to identify all 80 classes of targets in the image and applies encryption only to userdefined sensitive categories.

2) The system integrates two pluggable and complementary encryption schemes, TPE and XOR stream cipher encryption. This design effectively accommodates the varying privacy protection requirements across different UAV aerial photography scenarios.

3) A collaborative optimization mechanism covering detection, encryption, and restoration is established with a multi-dimensional composite loss function to significantly improve the decryption quality of the TPE scheme.

The remainder of this paper is organized as follows. Section II reviews relevant studies. Section III elaborates on the specific implementation of the encryption framework we design. Section IV presents experimental results and analysis. Finally, Section V concludes this paper and outlines future research directions.

## II. RELATED WORKS

With the increasing popularity of UAV aerial photography applications, privacy security issues arising from aerial images have attracted widespread attention. This section introduces the TPE algorithm and XOR stream cipher encryption technology.

## A. TPE

To reduce the risk of image data leakage in cloud environments, traditional schemes encrypt images into visually meaningless noise. However, this approach renders ciphertext images completely unreadable and fails to support cloudside tasks such as image retrieval, management and analysis. To address this limitation, Wright et al. [12] proposed a novel encryption paradigm that balances privacy protection and image usability, namely TPE. A standard TPE scheme generally divides the original image into multiple sub-blocks and encrypts each sub-block independently while retaining the pixel average of each block to preserve the overall visual outline of the image. The trade-off between encryption security and image usability can be flexibly adjusted by modifying the block size. With the advancement of relevant research, existing TPE techniques are categorized into exact TPE [13], [14] and approximate TPE [15], [16] according to whether the ciphertext thumbnail is strictly consistent with the original thumbnail. Early studies mostly adopted traditional signal processing and cryptographic operations, maintaining thumbnail structures via block permutation, pixel replacement and mean value preservation to strike a balance between security and usability. In recent years, deep learning has achieved remarkable progress in image modeling and generation, motivating researchers to adopt learnable models to fit the complex mapping between privacy protection effects and visual usability of images. Following this research line, Chai et al. [17] designed a GAN-based TPE-GAN scheme that greatly improves the usability of ciphertext images, yet there remains room for further optimization in the visual reconstruction quality of encrypted and decrypted images.

## B. XOR Stream Cipher Fundamentals

XOR stream cipher implements encryption and decryption by performing bitwise XOR operations between the pseudorandom key stream and the binary sequences of image pixels. It relies on the inherent symmetric and reversible property of the bitwise XOR operation: for any pixel data d and key stream k, the equation

$$
d \oplus k \oplus k = d\tag{1}
$$

always holds. The original image can be losslessly restored using the identical seed key [18]. With low computational overhead, this algorithm is well suited for lightweight terminal devices. However, traditional XOR stream ciphers only support full-image or fixed-region encryption and lack the capability of semantic target screening. After encryption, the overall visual information of the image is completely lost, making it difficult to balance privacy security and image usability.

## III. PROPOSED SCHEME

This section proposes a category-selective encryption framework oriented to object detection, which forms a closedloop pipeline of object detection, category-selective encryption, decryption/image restoration as shown in Fig. 1. The framework consists of three core modules: a multi-class object detection module based on YOLO11, a TPE module, and an XOR stream cipher encryption module. An independent decryption and restoration module is matched to the above encryption branches, and a multi-stage joint training strategy is specially designed only for the TPE branch. The detailed implementation schemes of each module are elaborated as follows.

## A. Object Detection Module

This module adopts the lightweight multi-class detector YOLO11n to identify and categorize all objects within aerial scenes, which is fine-tuned on the VisDrone2019 dataset [19]. The detector outputs bounding boxes for all detected targets. Users can customize a set of sensitive semantic categories to be encrypted at runtime; only objects falling into the designated sensitive classes are processed with encryption. Bounding boxes screened by the category filter are normalized to a unified coordinate range for subsequent mask generation, enabling semantic-aware selective encryption.

![](images/83a66cf771b00d0c160b7062c3511bbbd3741c2db4ab675b48b22cd86d66cba5.jpg)  
Fig. 1. Overall Flowchart of the Proposed Scheme.

## B. Selective Encryption Schemes

The proposed framework integrates two interchangeable encryption schemes with fundamentally different mechanisms, namely TPE and XOR stream cipher encryption. The two schemes share a unified input interface. Only sensitive pixel regions covered by detection boxes are modified, while all nonsensitive background pixels remain identical to the original image.

1) TPE Encryption: TPE performs pixelation transformation on sensitive target regions, and the original visual content can be restored by a well-trained Enhanced U-Net decoder. The encryption generates mosaic blur via average pooling and nearest-neighbor upsampling. A learnable blending coefficient α is introduced to adjust the weight ratio between pixelated patches and original patches, which automatically balances privacy protection strength and image restoration quality during joint training. For each bounding box containing sensitive regions, the encryption pipeline consists of three steps:

a) Adaptive Block Size Calculation: The kernel size of each block is dynamically calculated according to the width and height of the target box:

$$
k = \operatorname* { m a x } \left( k _ { \operatorname* { m i n } } , \left\lfloor { \frac { \operatorname* { m i n } ( h , w ) } { d } } \right\rfloor \right)\tag{2}
$$

where the minimum block size $k _ { \operatorname* { m i n } { } } ~ = { } ~ 6$ and the scaling factor $d \ : = \ : 8$ . Larger blocks are used for larger targets to achieve stronger mosaic blur, while distant small targets use finer blocks, ensuring consistent visual encryption effects for objects of all scales.

b) Pixelation Processing: The image patch $\mathbf { P } _ { \mathrm { t a r g e t } }$ inside the bounding box is processed by average pooling with identical kernel size and stride k, then upsampled back to the original patch resolution $( h , w )$ via nearest-neighbor interpolation. Average pooling compresses each $k \times k$ pixel block into a single average color value, and the subsequent upsampling recovers the spatial size of the region to form standard mosaic texture.

c) Learnable Blending Fusion: The encrypted patch is obtained by linear weighted fusion of the pixelated patch and the original patch, where the weight is controlled by the learnable parameter α:

$$
\mathbf { P } _ { \mathrm { e n c } } = \alpha \cdot \mathbf { P } _ { \mathrm { p i x } } + ( 1 - \alpha ) \cdot \mathbf { P } _ { \mathrm { t a r g e t } }\tag{3}
$$

α is constrained within the range [0, 1] in the forward propagation. When $\alpha = 1$ , the region is fully pixelated with the strongest privacy protection; when $\alpha = 0 ,$ , the patch is identical to the original image without any encryption effect.

2) XOR Stream Cipher Encryption: The XOR stream cipher realizes encryption relying on the involution property of the bitwise XOR operation. Deterministic pseudo-random noise is generated from the user-specified secret key to obfuscate pixels in sensitive regions, which achieves mathematically lossless reversibility without any neural network for restoration. The user-defined key string is mapped to a fixedlength seed through SHA-256 hashing. The seed initializes the PCG-64 pseudo-random number generator to produce a three-channel uniform noise map with the same resolution as the input image. The whole noise map is generated at one time rather than box by box, which guarantees consistent encryption results even when target boxes overlap. A binary mask $\mathbf { \bar { M } } \in \{ 0 , 2 5 5 \} ^ { H \times W }$ is constructed, where pixels inside sensitive boxes are set to 255 and background pixels are set to 0. Bitwise XOR operation is only conducted on pixels marked by the mask:

$$
\mathbf { Y } [ i , \mathbf { M } = 2 5 5 ] = \mathbf { X } [ i , \mathbf { M } = 2 5 5 ] \oplus \mathbf { N } [ \mathbf { M } = 2 5 5 ]\tag{4}
$$

where X and Y denote the original uint8-format image and encrypted image respectively. Background pixels uncovered by the mask retain their original values.

## C. Decryption and Restoration Module

This paper designs supporting decryption pipelines for TPE and XOR encryption branches to restore original content from encrypted images.

1) TPE Decryption: The learnable mosaic blur of the TPE module lacks a straightforward mathematical inverse transformation. An enhanced U-Net decoder is thus utilized to recover texture information from masked sensitive regions. The network receives the encrypted image $\mathbf { I } _ { \mathrm { e n c } }$ as input. Its multi-scale skip connections suppress mosaic artifacts and restore target details without altering background pixels, yielding the restored output $\mathbf { I } _ { \mathrm { r e s t } }$ . Joint training is performed on the U-Net decoder and TPE module to maintain reliable restoration quality for targets of diverse scales and different encryption intensity α.

2) XOR Stream Cipher Decryption: Relying on the involution property of bitwise XOR, decryption follows the same workflow as encryption. The original secret key regenerates PCG-64 pseudo-random noise, and the encryption mask serves to localize sensitive pixels. Performing XOR once more over masked regions recovers lossless uint8 pixel values without neural network computation, which enables lightweight decryption.

3) Post-processing Coordinate Inversion: After image restoration of two decryption branches, metadata cached in LetterBox preprocessing (scaling ratio, padding offsets and original resolution) is loaded to implement geometric inverse transformation. Symmetric padding is removed and images are rescaled to original size, so as to eliminate image deformation caused by preprocessing.

## D. Joint Training Strategy

A modular joint training scheme is adopted for the detection, encryption and decryption modules. Gradient backpropagation optimization is only performed on the TPE encryption branch to balance detection accuracy, privacy protection performance and image restoration quality.

1) Parameter Freezing and Update Rules: The weights of the pre-trained YOLO11n detection network are fully frozen throughout joint training, so that the target localization and classification performance will not be disturbed by the encryption and restoration task. Only two types of parameters are updated during training: the learnable blending coefficient α of TPE and all weights of the Enhanced U-Net restoration network. This strategy drastically reduces the number of parameters to be optimized and cuts down GPU memory consumption during training. The binary mask M generated in the encryption stage is reused for loss calculation. Only sensitive pixels covered by the mask participate in loss computation and gradient backpropagation, while background pixels are excluded from optimization to guarantee consistency between the background area and the original image.

2) Loss Function Design: The overall joint loss is composed of a weighted sum of reconstruction loss and encryption regularization loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \alpha } \mathcal { L } _ { \alpha }\tag{5}
$$

where $\lambda _ { \alpha }$ denotes the regularization weight, which is set to 0.6 in the experiments.

a) Mask-Constrained Reconstruction Loss $\mathcal { L } _ { r e c } .$ : L1 loss is utilized to optimize the restoration quality of sensitive regions, and pixel-wise error is only calculated within the masked encryption area:

$$
\mathcal { L } _ { \mathrm { r e c } } = \left. \left( \mathbf { I } _ { \mathrm { r e s t } } - \mathbf { I } _ { \mathrm { g t } } \right) \odot \mathbf { M } \right. _ { 1 }\tag{6}
$$

$\mathbf { I } _ { \mathrm { g t } }$ represents the original clear ground-truth image, and $\mathbf { I } _ { \mathrm { r e s t } }$ stands for the restored image output by the network.

b) Encryption Strength Regularization Loss ${ \mathcal { L } } _ { \alpha } .$ : This loss constrains the blending coefficient to approach 1, preventing the model from excessively weakening mosaic encryption merely to minimize reconstruction loss and maintaining effective privacy shielding capability:

$$
\mathcal { L } _ { \alpha } = 1 - \alpha\tag{7}
$$

## IV. EXPERIMENT ANALYSIS

All experiments are carried out on the public UAV aerial dataset VisDrone [19]. The YOLO11n detection backbone loads weights fine-tuned on VisDrone and remains fixed throughout the joint training process. For the TPE module, the minimum block size $k _ { \operatorname* { m i n } { } } = 6$ and scaling factor $d = 8$ are adopted. The enhanced U-Net incorporates SE channel attention with a channel compression ratio r = 8. The AdamW optimizer is utilized for training with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 8. The regularization weight of the loss function is set to $\lambda _ { \alpha } = 0 . 6$ . The model is trained for a total of 120 epochs with an early stopping strategy enabled.

![](images/50a044dcb975985b40eae6eef711fe5e9c011d958388b3e6e2b7ae44804f20d7.jpg)  
Fig. 2. Visual comparison of two encryption strategies for UAV aerial images. Row (1) encrypts only pedestrians and Row (2) encrypts pedestrians together with vehicles. In each group, the top row corresponds to TPE encryption, and the bottom row corresponds to XOR encryption.

## A. Visual Effect Analysis

Two encryption algorithms are applied to two sensitive target filtering strategies in this section: encrypting pedestrians only, and encrypting both pedestrians and vehicles simultaneously. Visual results presented in Fig. 2 reveal that both algorithms exclusively encrypt sensitive regions while leaving background areas unchanged. For the TPE scheme, sensitive targets are fully occluded by adaptive mosaic blur. This mechanism makes private information such as human faces and vehicle appearances unrecognizable without introducing any distortion to the background. After decryption with the enhanced U-Net network, the textures and contours of encrypted targets can be satisfactorily reconstructed. In contrast, the XOR encryption branch covers sensitive regions with pseudorandom noise to achieve favorable visual obfuscation. Provided the correct secret key is available, strict lossless pixel-level image reconstruction can be accomplished. Moreover, both encryption schemes avoid blur, color shift and other background artifacts, and remain effective in challenging aerial scenes with sensitive targets visually close to background tones.

## B. Image Quality Analysis

To quantitatively evaluate the reconstruction performance of the two encryption schemes, we compute the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) between the original aerial images and decrypted images on the VisDrone test set to compare the image restoration performance of TPE and XOR encryption. Higher PSNR and SSIM values indicate less distortion in decrypted images and better visual consistency with the original images. As can

![](images/50f629c0944ede8835a9806d2d60a14becfa7eac3e620fd0996cb762089f972a.jpg)  
Original Image  
Encrypted Image Decrypted Image Original ImageZoomed View Zoomed View  
Encrypted Image Decrypted ImageZoomed View Zoomed View

Fig. 3. Visual Comparison of Visual Effects Between the Proposed scheme and TPE schemes

be seen in Table I, XOR encryption achieves an SSIM of 1 and infinite PSNR under both encrypted region settings, which demonstrates that XOR enables lossless pixel-level reconstruction without any image distortion after decryption. In contrast, TPE encryption introduces slight reconstruction degradation. The decryption quality remains favorable when only pedestrians are encrypted, while the restoration performance degrades when both pedestrians and vehicles are encrypted due to the expanded encrypted area.

TABLE I  
PSNR AND SSIM QUANTITATIVE METRICS OF TPE AND XOR UNDER DIFFERENT ENCRYPTED REGIONS
<table><tr><td colspan="2">Encrypted Region</td><td>SSIM</td><td>PSNR</td></tr><tr><td rowspan="2">TPE</td><td>person</td><td>0.9792</td><td>38.48dB</td></tr><tr><td>person &amp; car</td><td>0.9734</td><td>35.26dB</td></tr><tr><td rowspan="2">XOR</td><td>person</td><td>1</td><td>inf</td></tr><tr><td>person &amp; car</td><td>1</td><td>inf</td></tr></table>

## C. Comparative Analysis

Existing saliency-driven privacy schemes, traditional regionselective encryption and basic object anonymization share two inherent limitations. Category-customized encryption for designated semantic targets cannot be realized, and most alternatives only support irreversible visual masking without complete high-fidelity recovery pipelines.

Two classic full-image TPE algorithms, HF-TPE [15] and TPE-GAN [17], are taken as baselines for fair performance benchmarking of the proposed TPE and XOR encryption modules. As illustrated in Fig. 3, both baselines implement full-image encryption that triggers visual distortion on nonsensitive regions such as roads and buildings. Such distortion destroys inherent aerial scene features, impairs the performance of subsequent scene recognition tasks, and introduces substantial computational overhead. A favorable balance between privacy preservation and image utility is attained by the presented framework through encryption limited to predefined sensitive objects only. Reconstruction quality of all algorithms is quantified in Table II with SSIM and PSNR metrics, where all approaches are categorized into cryptographic and deep learning-based groups. Satisfactory reconstruction performance is obtained by HF-TPE among cryptographic schemes. Benefiting from invertible XOR operations, the propsed XOR scheme branch delivers perfect lossless pixel reconstruction with an SSIM value of 1 and infinite PSNR. For deep learningbased TPE-GAN, its GAN encoder-decoder architecture generates severe reconstruction artifacts and deteriorates the quality of decrypted outputs. By contrast, the proposed TPE scheme achieves outstanding reconstruction fidelity and effectively alleviates such severe distortion. Overall, redundant encryption over background regions is eliminated by both presented schemes. Furthermore, superior reconstruction performance against corresponding baselines within each technical category is achieved by the two proposed encryption branches.

TABLE II  
PSNR AND SSIM QUANTITATIVE METRICS OF DIFFERENT ENCRYPTION SCHEMES UNDER VARIOUS ENCRYPTED REGIONS
<table><tr><td></td><td>Encrypted Scheme</td><td>SSIM</td><td>PSNR</td></tr><tr><td>HF-TPE [15]</td><td>cryptography-based</td><td>0.9996</td><td>55.40dB</td></tr><tr><td>Proposed-XOR</td><td>cryptography-based</td><td>1</td><td>inf</td></tr><tr><td>TPE-GAN [17]</td><td>deep learning-based</td><td>0.8699</td><td>28.07dB</td></tr><tr><td>Proposed-TPE</td><td>deep learning-based</td><td>0.9734</td><td>35.26dB</td></tr></table>

## D. Detection Performance and Privacy Implications

Quantitative reliability tests on the VisDrone test set [19] show that the pre-trained model achieves mAP@0.5 = 27.6% and mAP@0.5:0.95 = 15.5%, which is standard for lightweight detectors in complex UAV scenes with dense small objects and varying altitudes. Although distant tiny targets have limited recognizable details due to low resolution, privacy leakage may still occur through cross-frame contextual reasoning and temporal correlation analysis. Future research will mitigate these risks through more robust detection pipelines or unified joint optimization of detection and encryption.

## E. Efficiency Analysis

Network parameter counts and average processing latencies of the proposed XOR and TPE modules are reported in Table III. The XOR pipeline is a handcrafted lightweight bitwise operation without neural network components, achieving only 0.5 ms latency for both encryption and decryption under the unified experimental environment. In the TPE branch, encryption and decryption show a clear latency gap: encryption uses only a manually defined hyperparameter α to control adaptive blurring intensity in sensitive areas, resulting in a low latency of approximately 1 ms. In contrast, TPE decryption employs a deep convolutional reconstruction network with 50.29 million trainable parameters to recover distorted texture details, incurring substantial matrix computation and memory access overhead and a latency of around 30 ms. Benefiting from target-selective encryption, both modules restrict computation to YOLO11n sensitive bounding boxes, thereby reducing the number of processed pixels.

TABLE III  
NETWORK PARAMETER COUNT AND AVERAGE PROCESSING LATENCY OF THE PROPOSED XOR AND TPE MODULES
<table><tr><td></td><td>Stage</td><td>Network Parameter Count</td><td>Average Latency</td></tr><tr><td rowspan="2">TPE</td><td>Encryption</td><td>— (hyperparameter α)</td><td>~1ms</td></tr><tr><td>Decryption</td><td>50.29M</td><td>~30ms</td></tr><tr><td rowspan="2">XOR</td><td>Encryption</td><td>N/A</td><td>~0.5ms</td></tr><tr><td>Decryption</td><td>N/A</td><td>~0.5ms</td></tr></table>

## V. CONCLUSION

This paper designs a dual-mode category-selective image encryption framework integrated with object detection. YOLO is utilized to locate sensitive targets, and TPE adaptive pixelation encryption and XOR stream cipher encryption are separately adopted to realize regional privacy protection. The TPE scheme leverages a learnable blending coefficient to balance the masking effect and image readability, and relies on an enhanced U-Net for texture restoration; the XOR scheme achieves lightweight lossless encryption and decryption. Experimental results demonstrate that the proposed scheme only encrypts targets of designated categories, which drastically reduces redundant computation. It simultaneously delivers visual privacy protection while preserving basic image content, making it suitable for real-time image transmission scenarios of UAVs. Future work will further optimize the network inference speed and encryption security strength, expand the applicable scope of the algorithm, and adopt more robust detection and joint detection-encryption optimization strategies to mitigate privacy leakage risks caused by undetected small, occluded or poorly illuminated sensitive objects.

## ACKNOWLEDGMENT

This work was supported in part by the Sichuan Science and Technology Program under Grant No. 2024NSFSC2045, and in part by the Research Project of Quancheng Laboratory, China under Grant No. QCL20250303.

[1] P. Daponte, L. De Vito, L. Glielmo, L. Iannelli, D. Liuzza, F. Picariello, and G. Silano, “A review on the use of drones for precision agriculture,” Journal of Physics: Conference Series, vol. 275, no. 1, p. 012022, 2019.

[2] H. Xu, L. Wang, W. Han, Y. Yang, J. Li, Y. Lu, and J. Li, “A survey on uav applications in smart city management: challenges, advances, and opportunities,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 8982–9010, 2023.

[3] D. Gallacher, “Drone applications for environmental management in urban spaces: a review,” International Journal of Sustainable Land Use and Urban Planning, vol. 3, no. 4, pp. 1–14, 2016.

[4] A. Abu-Khadrah, A. Al-Qerem, M. R. Hassan, A. M. Ali, and M. Jarrah, “Drone-assisted adaptive object detection and privacy-preserving surveillance in smart cities using whale-optimized deep reinforcement learning techniques,” Scientific Reports, vol. 15, no. 1, p. 9931, 2025.

[5] J.-X. Liu, G. Han, N. Sun, X.-F. Li, Z.-G. Gong, and Q.-S. Sun, “Generalized compressed sensing with qr-based vision matrix learning for face recognition under natural scenes,” Signal Processing: Image Communication, vol. 77, pp. 11–19, 2019.

[6] M. M. Eltoukhy, F. S. Alsubaei, Y. M. Elnabawy, and K. M. Hosny, “Multiple image encryption techniques: strategies, challenges, and potential future directions,” Alexandria Engineering Journal, vol. 125, pp. 367–387, 2025.

[7] A. Shafique, M. U. Rehman, K. H. Khan, S. S. Jamal, A. Mehmood, and S. A. Chaudhry, “Securing high-resolution images from unmanned aerial vehicles with dna encoding and bit-plane extraction method,” IEEE Access, vol. 11, pp. 44 559–44 577, 2023.

[8] L. Zhou, H. Xia, Q. Lin, X. Yang, X. Zhang, and M. Zhou, “Twodimensional hyperchaos-based encryption and compression algorithm for agricultural uav-captured planar images,” Scientific Reports, vol. 14, no. 1, p. 22423, 2024.

[9] L. Zhou, H. Chen, X. Zhou, Y. Yuan, W. Zhu, and M. Zhou, “A smart agriculture image protection scheme based on annealing algorithm and affine transformation optimized for chaos-generated s-box,” Nonlinear Dynamics, pp. 1–25, 2024.

[10] H. Wen, Z. Xie, Z. Wu, Y. Lin, and W. Feng, “Exploring the future application of uavs: face image privacy protection scheme based on chaos and dna cryptography,” Journal of King Saud University - Computer and Information Sciences, vol. 36, no. 1, p. 101871, 2024.

[11] R. Tu, W. Wen, and C. Hua, “An unequal image privacy protection method based on saliency detection,” Security and Communication Networks, vol. 2020, no. 1, p. 8842376, 2020.

[12] C. V. Wright, W. C. Feng, and F. Liu, “Thumbnail-preserving encryption for jpeg,” in Proceedings of the 3rd ACM Workshop on Information Hiding and Multimedia Security, 2015, pp. 141–146.

[13] Y. Peng, Z. Huang, Z. Li, and S. He, “Sa-tpe: An ideal thumbnail preserving encryption method based on selective area,” Expert Systems with Applications, vol. 278, p. 127275, 2025.

[14] Y. Zhang, C. Li, Y. Tong, Z. Li, S. He, and Y. Peng, “Thumbnail preserving encryption via triple random number pair-based sum preserving diffusion mechanism,” Expert Systems with Applications, vol. 305, p. 130896, 2026.

[15] Y. Zhang, R. Zhao, X. Xiao, R. Lan, Z. Liu, and X. Zhang, “HF-TPE: High-fidelity thumbnail-preserving encryption,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 3, pp. 947–961, 2022.

[16] D. An, X. Pu, D. Hao, R. Zhao, and Y. Zhang, “Reversible data hiding with flexible extraction for thumbnail-preserving encryption,” IEEE Transactions on Consumer Electronics, vol. 71, no. 2, pp. 2824– 2836, 2025.

[17] X. Chai, Y. Wang, X. Chen, Z. Gan, and Y. Zhang, “Tpe-gan: Thumbnail preserving encryption based on gan with key,” IEEE Signal Processing Letters, vol. 29, pp. 972–976, 2022.

[18] Z. A. M. A.-S. Mohammed and O. Albeerish, “Encryption and decryption of digital images using the xor algorithm,” Rawafed Journal of Scientific Studies and Research, vol. 2, no. 1, 2026.

[19] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, “Detection and tracking meet drones challenge,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7380–7399, 2021.