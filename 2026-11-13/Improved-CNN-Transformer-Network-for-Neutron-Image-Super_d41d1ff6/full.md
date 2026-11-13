# Improved CNN-Transformer Network for Neutron Image Super-

Resolution

Yuheng Chu<sup>1</sup>，Hongli Li<sup>1</sup>，Tian Zhang<sup>1\*</sup> (¹ Northeast Normal University, Changchun 130024, Jilin, China Corresponding author. E-mail: zhangt100@nenu.edu.cn)

Abstract: The spatial resolution of neutron images is of great importance for material defect diagnosis, structural characterization, and nondestructive testing. However, owing to limitations in neutron beam quality, scintillator conversion efficiency and light scattering, detector pixel size, and noise levels, conventional neutron images often suffer from speckle noise, blurred edges, and missing details, making them inadequate for high-precision analysis. To address these issues, an improved super-resolution model combining convolutional neural networks (CNNs) and Transformers is proposed. First, to overcome the difficulty of obtaining high-quality and clear training images for neutron imaging, a pretrained image restoration model is fine-tuned using real neutron images and then employed to enhance existing neutron images, thereby providing highquality reference images for subsequent super-resolution model training. Second, an improved CRAFT-based super-resolution network is developed by incorporating a hybrid prior-guided channel fusion module (HPCF), a mean–standard deviation dual-statistics squeeze-and-excitation block (MS-SEBlock), and a partial convolution feed-forward network (PCFN), with the aim of improving the reconstruction of important information, such as textures and fine details, in neutron images. Simulation results demonstrate that, compared with bicubic interpolation, SRCNN, SRGAN-VGG, and other methods, the proposed network achieves superior performance in both full-reference metrics, including PSNR and SSIM, and no-reference metrics, including AGM and SMD. The visual quality and structural fidelity of the reconstructed images are also significantly improved. This study provides a new approach to improving neutron image quality.

Keywords: Neutron imaging; Super-resolution; Deep learning

## 1. Introduction

Neutron imaging has significant applications in the field of nondestructive testing (NDT) owing to the high penetrating capability of neutrons and their sensitivity to light elements. In recent years, image super-resolution techniques have been increasingly applied to tasks such as medical diagnosis and industrial inspection <sup>[1–3]</sup>. Improving spatial resolution is therefore of great importance for extending the capabilities of neutron imaging in nondestructive testing. The spatial resolution of neutron images is not determined by a single factor; rather, it is jointly affected by key components, including neutron source characteristics, detector performance, and collimator design, as well as physical factors such as neutron scattering, particle cross-talk, statistical fluctuations in neutron fluence rate, and gamma-ray contamination <sup>[4–6]</sup>. Image blurring and noise degradation caused by these factors adversely affect subsequent nondestructive testing and elemental analysis.

Overcoming the resolution limitations at the hardware level requires upgrading large-scale neutron source facilities to increase neutron yield and improve beam quality, as well as developing more advanced scintillator screens and detectors. Although these improvements can significantly enhance imaging quality, they involve substantial costs and are difficult to implement on a wide scale <sup>[7–9]</sup>. Deep learning has been successfully applied to medical image enhancement, including low-dose CT denoising and MR image restoration <sup>[10–13]</sup>, motivating its application to neutron image super-resolution.

Traditional super-resolution methods mainly include interpolation-based, reconstruction-based, and learning-based approaches <sup>[14,15]</sup>. However, these methods have certain limitations in recovering details from complex images. In contrast, deep learning can learn image features in a data-driven manner and has demonstrated promising reconstruction capabilities in various fields, including medical imaging <sup>[16]</sup>, defect detection <sup>[17]</sup>, and industrial computed tomography (CT) <sup>[18]</sup>. It has also opened up new research directions for neutron image super-resolution. Owing to their high reconstruction accuracy and good scalability, deep learning-based methods have been widely applied to super-resolution reconstruction and can be readily integrated with auxiliary tasks such as detection and recognition. Furthermore, these methods can substantially enhance the information content of neutron images <sup>[19].</sup>

In recent years, deep learning has gradually been introduced into neutron image superresolution tasks. In 2023, the Bhabha Atomic Research Centre combined SRCNN with the POCS algorithm to achieve super-resolution reconstruction of simulated neutron images, providing an initial demonstration of the feasibility of convolutional neural networks in this field. However, the feature representation capability of the network was limited, and its ability to recover complex textures and high-frequency details remained insufficient <sup>[20]</sup>. Subsequently, Ma et al. incorporated a VGG-based perceptual loss into SRGAN and evaluated its reconstruction performance using real neutron images. Nevertheless, the adversarial generation process may introduce artificial textures and local artifacts, potentially affecting the fidelity of structural information <sup>[21]</sup>. In the same year, Ma et al. further proposed a hybrid attention network, RHAN, which enhances high-frequency information through multi-scale features and attention mechanisms. However, its adaptability to the complex degradation characteristics of real neutron images still requires improvement <sup>[22]</sup>. In 2025, Liu et al. proposed a neutron image super-resolution network based on parallel Transformers and achieved promising reconstruction results on both benchmark datasets and real images. However, its training data consisted primarily of grayscale natural images, resulting in a certain domain gap from real neutron images <sup>[23]</sup>. In 2026, Wu et al. applied Real-ESRGAN to neutron image reconstruction, improving visual quality and detail representation. Nevertheless, its training data were mainly derived from X-ray images, which differ from neutron images in terms of imaging physics and noise characteristics, potentially limiting the model's generalization capability <sup>[24]</sup>.

In summary, existing studies have made progress in neutron image super-resolution. However, several challenges remain, including insufficient adaptation to real neutron images, the risk of introducing artificial textures, and inadequate recovery of weak edges and high-frequency details. To address these issues, this study proposes an improved Cross-Refinement Attention Fusion Transformer (CRAFT) network tailored to the imaging characteristics of real neutron images. The proposed network aims to enhance local high-frequency detail extraction and global feature modeling, thereby improving detail recovery and structural representation in neutron images.

The main contributions of this study are summarized as follows:

1) An adaptive image restoration-based data preprocessing method is proposed for neutron images. A pretrained image restoration model is fine-tuned using neutron image data to improve the quality of the original images, thereby constructing corresponding reference image data for superresolution model training.

2) An improved CNN-Transformer hybrid super-resolution reconstruction network is developed based on the CRAFT architecture. To address the insufficient representation of local details and inadequate feature fusion in neutron images, the feature processing structures within the network are enhanced to improve local spatial feature modeling, information fusion between different feature branches, and channel-wise feature representation. The resulting network enables ×4 super-resolution reconstruction of neutron images.

3) Quantitative evaluation and visual comparison experiments are conducted on a selfconstructed neutron image dataset, providing preliminary validation of the proposed method's effectiveness in recovering image details and improving neutron image quality.

## 2. Methodology

## 2.1 Neutron Image Restoration and Reference Image Construction

Due to limitations in neutron source intensity, detector performance, and imaging conditions, raw neutron images often suffer from noise, blurring, low contrast, and missing fine details. To construct training data suitable for super-resolution reconstruction, a pretrained image restoration model is first employed to preprocess the raw neutron images, thereby improving image quality and providing reference images for subsequent dataset construction.

In this study, the Adaptive Sparse Transformer (AST-v2) image restoration network is adopted as the initial image restoration model. As illustrated in Fig. 1, the model employs a Transformerbased feature modeling mechanism to extract relevant image information through adaptive sparse self-attention and a feature refinement feed-forward network. Considering the differences in imaging characteristics between neutron images and natural images, the pretrained model is further fine-tuned using neutron image data to improve its adaptability to the noise distributions, edge structures, and texture characteristics of neutron images.

During model fine-tuning, the raw neutron images are first subjected to different types and levels of degradation, including noise interference, blurring, local brightness variations, and contrast variations, to generate the input data required for model training. The network is then initialized using the pretrained weights and fine-tuned specifically on neutron image data. Finally, the images generated by the restoration model are used as high-quality reference images in the super-resolution dataset.

It should be noted that, owing to the lack of physically measured high-resolution reference images strictly corresponding to the raw neutron images, the reference images used in this study are primarily generated by the image restoration model. Therefore, these images should be regarded as algorithmically enhanced high-quality reference data rather than ideal high-resolution images obtained through direct physical measurements.

![](images/219f8ca0e6ddeefb2a04e665cbcd616d31f90717f8a9d51e93b05ee56420d262.jpg)  
Figure 1: Main Network of AST-v2 Large Model

## 2.2 Dataset Construction

In this study, a neutron image dataset was constructed for super-resolution reconstruction. The images were collected from various representative neutron imaging objects, including aero-engine blades, nuclear fuel assemblies, space reactor heat pipes, special nuclear materials, rocks, batteries, and other mechanical structures. These images contain diverse material boundaries, internal structures, and fine details, making them suitable for evaluating the reconstruction capabilities of super-resolution models across different types of neutron images.

To increase the diversity of the training data, data augmentation was applied to the reference images. The augmentation operations mainly included horizontal flipping, vertical flipping, 90° rotation, grayscale enhancement, Gaussian blurring, brightness adjustment, and contrast adjustment. These operations increase the diversity of training samples and improve the model's adaptability to different image structures and imaging conditions.

Approximately 200 original neutron images were used to construct the dataset. Before patch extraction and data augmentation, the original images were divided into training, validation, and test sets at a ratio of 80%, 10%, and 10%, respectively. Images from the same specimen or acquisition sequence, where applicable, were assigned to the same subset to prevent information leakage. Consequently, no patches derived from the same original image were shared across different subsets.

The image restoration model was fine-tuned exclusively on the training subset. Neither validation nor test images were used for fine-tuning. The resulting model was then frozen and applied independently to each subset to generate algorithmically enhanced reference images.

Within each subset, the reference images were cropped into patches of 256 × 256 pixels, and the corresponding low-resolution patches of 64 × 64 pixels were generated by ×4 downsampling. Data augmentation was applied only to the training subset, while the validation and test subsets were processed without augmentation. Patch extraction and training-set augmentation yielded approximately 10,000 image patches in total. The validation subset was used for model selection, and the test subset was reserved for the final performance evaluation.

## 2.3 CRAFT-Based Super-Resolution Network

In this study, a neutron image super-resolution model is developed based on the CRAFT network. Through convolutional feature extraction, residual feature interaction, and multi-stage feature fusion, the network maps low-resolution images to high-resolution images. Its main components include shallow feature extraction, deep feature extraction, feature fusion, and image reconstruction, ultimately producing ×4 super-resolution results through upsampling. The overall architecture is shown in Fig. 2.

![](images/e7e483794b73a876c12669ccf73d0d19680a08c122fa15a901d46fdb50a8de06.jpg)  
Fig. 2. CRAFT Backbone Network

During neutron imaging, factors such as detector performance, imaging noise, and scattering often lead to low contrast, weak edge responses, and missing details. Consequently, the original network may be insufficient for recovering local details and textures. To address these challenges, this study leverages the specific imaging characteristics of neutron radiography to refine selected feature processing modules within the CRAFT network, thereby enhancing its representational capacity for local details and multi-level features.

Specifically, as shown in Fig. 3, a high-frequency enhancement residual block (HFERB) is employed within the CRFB to extract local high-frequency features, while a shifted rectangular window attention block (SRWAB) is used to model global contextual information. A partial convolution feed-forward network (PCFN) is introduced into the SRWAB to replace the original feed-forward network, enhancing local spatial feature transformation. Furthermore, a hybrid priorguided channel fusion module (HPCF) is designed to facilitate interactive feature fusion between the HFERB and SRWAB branches. A mean–standard deviation dual-statistics squeeze-andexcitation block (MS-SEBlock) is also incorporated to adaptively recalibrate feature channels within the CRFB, enhancing informative responses and suppressing redundant information.

Through these designs, the network captures both local details and global structural information, providing multi-level feature representations for neutron image super-resolution. Compared with the baseline network, the improved model places greater emphasis on weak edges, fine structures, and high-frequency textures in neutron images.

![](images/cc8069def6d96e35ede43553b47d6f37f55bab319d04aad6a8480726f2f23148.jpg)  
Fig. 3. Internal Architecture of the CRFB

## 2.4 Training Objective Function

To improve the pixel fidelity and edge detail of reconstructed images, a training objective combining the Charbonnier loss and gradient loss is adopted. The Charbonnier loss constrains the pixel-wise differences between the reconstructed and reference images, while the gradient loss constrains local gradient information to enhance the recovery of edges and fine structures. The overall loss function is defined as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { c h a r } } + \lambda \mathcal { L } _ { \mathrm { g r a d } }\tag{1}
$$

where $\mathcal { L } _ { \mathrm { c h a r } }$ denotes the Charbonnier loss, $\mathcal { L } _ { \mathrm { g r a d } }$ denotes the gradient loss, and λ is the weighting coefficient used to balance the two terms. In this study, λ is set to 0.05 to balance pixel

reconstruction accuracy and edge detail recovery.

## 3. Experimental Setup and Evaluation Metrics

## 3.1 Experimental Setup

The proposed super-resolution reconstruction network is implemented using the PyTorch deep learning framework and trained on an NVIDIA RTX 4090 GPU. The network parameters are optimized using the Adan optimizer, with an initial learning rate of $2 { \times } 1 0 ^ { - 4 }$ , a batch size of 32, and 500,000 training iterations. The learning rate is adjusted using the MultiStepLR strategy, with each adjustment reducing it to 50% of its previous value.

For image restoration preprocessing, a pretrained model is fine-tuned on neutron images. The AdamW optimizer is used during this stage, with the learning rate adjusted using a cosine annealing schedule. The model is trained for 100 epochs, with the learning rate gradually decreasing from1×10<sup>-5</sup> to 1×10<sup>-6</sup>.

Table 1. Fine-tuning training parameters of AST-v2
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Input</td><td>Degraded neutron images</td></tr><tr><td>Supervision target</td><td>Original neutron images</td></tr><tr><td>Output</td><td>Restored neutron images</td></tr><tr><td>Loss function</td><td>Charbonnier Loss</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 } – 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Learning rate schedule</td><td>Cosine annealing decay</td></tr><tr><td>Data augmentation</td><td>Random cropping</td></tr><tr><td>Training epochs</td><td>100 epoch</td></tr></table>

## 3.2 Evaluation Metrics

To quantitatively evaluate the super-resolution results, peak signal-to-noise ratio (PSNR), structural similarity index measure (SSIM), and learned perceptual image patch similarity (LPIPS) are adopted as the primary evaluation metrics. PSNR measures the pixel-wise error between reconstructed and reference images, SSIM evaluates their similarity in terms of luminance, contrast, and structure, while LPIPS quantifies perceptual differences based on distances in deep feature space. In general, higher PSNR and SSIM values and lower LPIPS values indicate greater similarity between the reconstructed and reference images.

In addition, average gradient magnitude (AGM) and standard deviation measure (SMD) are used to assess image details and sharpness. AGM reflects the average intensity of grayscale variations, while SMD describes the dispersion of the grayscale distribution. These metrics provide complementary assessments of image quality from different perspectives. However, their values may also be affected by noise and excessive sharpening; therefore, they should not be used alone as evidence of improvements in actual spatial resolution.

## 4. Experimental Results and Analysis

## 4.1 Quantitative Comparison

The original real neutron radiographic images of turbine blades, which contain noise and other degradations, were first denoised using a Swin Transformer. The denoised images were then processed using the proposed method and other competing methods for comparison. The visual

comparison results are shown below.

![](images/4a9032228bed4658cd44d1b2aac661d98277cab696f23e163a5453fca2f7dc93.jpg)  
LR  
Bicubic  
SRCNN  
SRGAN-VGG  
Ours  
Figure 4. Comparison of ×4 super-resolution reconstruction results for real neutron radiographic images: (a) Neutron radiography (NR) image of a turbine blade acquired at the NEUTRA neutron imaging facility at PSI<sup>[25]</sup>; (b) real neutron radiographic image after denoising using a Swin Transformer<sup>[4]</sup>.

A comparison of Bicubic, SRCNN, SRGAN-VGG, and Ours shows that the Bicubic results are generally smooth, with blurred horizontal stripe boundaries and limited detail recovery. Compared with Bicubic, SRCNN provides some improvement; however, over-smoothing remains evident, and the separation of the stripe structures is not substantially enhanced. SRGAN-VGG significantly sharpens the stripe boundaries and improves the perceived image sharpness, but it also carries the risk of over-sharpening and introducing artificial textures.

In contrast, Ours achieves a more balanced visual result. The horizontal stripe structures are clearer than those produced by Bicubic and SRCNN, without the excessive sharpening and noise amplification that may occur with SRGAN-VGG. Ours better preserves the grayscale transitions and structural continuity of the original neutron images, achieving a more favorable balance between image sharpness and fidelity. Therefore, Ours delivers superior overall visual quality. In terms of quantitative results, Ours also achieves improvements over the other three methods across all evaluated metrics.

Table 2 Evaluation of Image Quality of Neutron Imaging in Turbine Blades
<table><tr><td>Evaluation Metrics / Models</td><td>Bicubic</td><td>SRCNN</td><td>SRGAN-VGG</td><td>Ours</td></tr><tr><td>PSNR</td><td>21.96</td><td>32.06</td><td>31.35</td><td>32.99</td></tr><tr><td>SSIM</td><td>0.6568</td><td>0.9270</td><td>0.9252</td><td>0.9368</td></tr><tr><td>LPIPS</td><td>0.0352</td><td>0.0235</td><td>0.0611</td><td>0.0184</td></tr><tr><td>AGM</td><td>0.0545</td><td>0.0610</td><td>0.0580</td><td>0.0619</td></tr><tr><td>SMD</td><td>0.0057</td><td>0.0062</td><td>0.0064</td><td>0.0065</td></tr></table>

For real neutron camera images, the proposed method was further compared with several advanced super-resolution models, including the natural-image super-resolution models SAFMN and WaveMixSR, as well as SRNCT<sup>[23]</sup>, which is specifically designed for neutron image superresolution reconstruction. As shown in Fig. 5, enlarged local regions of the neutron camera images are used to examine the ability of different methods to recover edge contours, internal textures, and fine structural details. Compared with the other methods, the proposed method restores edge details and internal textures more clearly, providing better discernibility of fine structures and demonstrating stronger detail reconstruction capability.

![](images/b7b9c1fe96873048337187b5d057a365b921caf27ad96e508fe0fb9686fed133.jpg)  
Figure 5: Super-resolution visual effects of neutron cameras compared to other advanced natural super-resolution models

To further evaluate the reconstruction performance of different methods on real neutron camera images, five metrics—PSNR, SSIM, AGM, SMD, and LPIPS—were adopted for quantitative comparison. The results are presented in Table 3. The proposed method achieves a PSNR of 28.06 dB and an SSIM of 0.9336, both higher than those of the competing methods, demonstrating its advantages in pixel-level reconstruction accuracy and structural similarity. Meanwhile, the proposed method obtains the highest AGM (0.2816) and SMD (0.0372), indicating stronger gradient responses and improved detail clarity in the reconstructed images. In terms of perceptual quality, the proposed method achieves an LPIPS of 0.1573, lower than those of the other competing models, suggesting greater perceptual similarity in the feature space. Taken together, the qualitative and quantitative results demonstrate that the proposed network provides effective detail recovery and structural preservation on real neutron camera images, highlighting its potential for neutron image super-resolution tasks.

Table 3. Quantitative Evaluation Results of Different Super-Resolution Methods on Neutron Camera Images
<table><tr><td>Evaluation Metrics /Models</td><td>SAFMN</td><td>WaveMixSR</td><td>SRNCT</td><td>Ours</td></tr><tr><td>PSNR</td><td>18.02</td><td>18.17</td><td>21.31</td><td>28.06</td></tr><tr><td>SSIM</td><td>0.5879</td><td>0.6154</td><td>0.7701</td><td>0.9336</td></tr><tr><td>AGM</td><td>0.2653</td><td>0.2463</td><td>0.2465</td><td>0.2816</td></tr><tr><td>SMD</td><td>0.0345</td><td>0.0323</td><td>0.0324</td><td>0.0372</td></tr><tr><td>LPIPS</td><td>0.2636</td><td>0.2448</td><td>0.2669</td><td>0.1573</td></tr></table>

## 4.2 Discussion and Limitations

Although the proposed method demonstrates certain advantages in quantitative evaluation and visual comparisons, several limitations remain. First, the high-quality reference images used for training are primarily generated by an image restoration model, and their detail information may be influenced by the preprocessing model. Second, neutron imaging objects and imaging conditions are diverse, while the current dataset remains limited in size and sample distribution. Therefore, the generalization capability of the model across different imaging devices and experimental conditions requires further investigation.

Furthermore, metrics such as PSNR, SSIM, and LPIPS mainly evaluate the similarity between reconstructed and reference images and cannot fully reflect changes in the actual spatial resolution of neutron imaging systems. Therefore, future studies should incorporate more reliable physical reference data, edge response analysis, and modulation transfer function (MTF) measurements to further evaluate the actual resolving capability of the reconstructed images.

## 5. Conclusion

This study investigates deep learning-based super-resolution reconstruction for neutron images, addressing common issues such as noise, blurring, low contrast, and missing details. By using an image restoration model to construct high-quality reference data and introducing targeted feature extraction and fusion structures into the CRAFT architecture, a ×4 super-resolution reconstruction method for neutron images is developed.

Experimental results demonstrate that the proposed method achieves certain advantages in quantitative evaluation and visual comparisons, improving edge clarity and detail representation in neutron images. This study provides an initial approach to applying deep learning-based superresolution techniques to neutron imaging. Future work will focus on expanding the dataset, improving the physical resolution evaluation framework, and conducting more in-depth investigations into the internal feature interaction mechanisms and generalization capability of the network.

## References

[1] Wang Y L,Zhu J Y,Zhen J X. Utilization and Future Development of China Advanced Research Reactor [J]. Atomic Energy Science and Technology，2020, 54:213-217.

[2] Zhou S, Pan J, Yang J. Learning an adaptive sparse transformer for efficient image restoration[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025,72(12): 3674-3685.

[3] Wei G H,Han S B,Chen D F et al. Application of neutron radiography for nondestructive testing nuclear fuel elements[J]. Nuclear Techniques, 2012, 35 (11 ) :821- 826.

[4] Meng C B, Zhu W W, Zhang Z, et al. Multi-distortion suppression for neutron radiographic images based on generative adversarial network[J]. Nucl. Sci. Tech, 2024, 35(4): 176-188.

[5] Wang W, Zheng Y, Zhao C, et al. High-performance Defect Recognition and Segmentation Method for Dual-mode Industrial Radiographic Images[J]. IEEE Transactions on Nuclear Science, 2025.

[6] Xu P,YU W T,Bao Jie,et al. The Influencing Factor and Related Research Progress of Resolution in Fast Neutron Radiography [J]. Nuclear Physics Review,2026,43(1):202-212.

[7] W. Guo, S. Qiao, C. Zhao, and T. Zhang, “Defect detection for industrial neutron radiographic images based on modified YOLO network,” Nucl. Instrum. Methods Phys. Res. A, Accel. Spectrom. Detect.Assoc. Equip., vol. 1056, Nov. 2023, Art. no. 168694, doi: 10.1016/j.nima.2023.168694.

[8] Zhang, Z., Meng, CB., Jiang, XL. et al. Comprehensive quality assessment method for neutron radiographic images based on CNN and visual salience. NUCL SCI TECH 36, 118 (2025). https://doi.org/10.1007/s41365-025-01686-4

[9] Zhou Z, Zheng J, Ruan S, et al. A transparent hybrid metal halide glassy scintillation screen for high-resolution fast neutron radiography[J]. Nature Communications, 2025, 16(1): 6215.

[10] Chen H, Zhang Y, Kalra M K, et al. Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network[J]. IEEE Transactions on Medical Imaging, 2017, 36(12): 2524-2535. DOI: 10.1109/TMI.2017.2715284.

[11] Yang Q, Yan P, Zhang Y, et al. Low-Dose CT Image Denoising Using a Generative Adversarial Network With Wasserstein Distance and Perceptual Loss[J]. IEEE Transactions on Medical Imaging, 2018, 37(6): 1348-1357. DOI: 10.1109/TMI.2018.2827462.

[12] Wang D, Fan F, Wu Z, et al. CTformer: convolution-free Token2Token dilated vision transformer for low-dose CT denoising[J]. Physics in Medicine & Biology, 2023, 68(6): 065012. DOI: 10.1088/1361-6560/acc000.

[13] Sun, Y., Wang, L., Li, G. et al. A foundation model for enhancing magnetic resonance images and downstream segmentation, registration and diagnostic tasks. Nat. Biomed. Eng 9, 521–538 (2025). https://doi.org/10.1038/s41551-024-01283-7

[14] J.S. Brenizer, A review of significant advances in neutron imaging from conception to the present, Phys. Procedia 43 (2013) 10–20

[15] ZHANG L, WU X. An edge-guided image interpolation algorithm via directional filtering and data fusion[J]. IEEE Transactions on Image Processing, 2006, 15(8): 2226-2238.

[16] Li Y H,Wang Z,Zhang K B.et al. Survey on example learning-based single image super-resolution technique [J]. Computer Engineering and Applications, 2018, 54(15): 13-21.

[17] C. Ajmi et al., Using Deep Learning for Defect Classification on a Small Weld Xray Image Dataset,J. Nondestruct. Eval. 39 (2020) 68.

[18] E. Sogancioglu et al., Deep Learning for Chest X-ray Analysis: A Survey, Med. Image Anal. 72 (2021)102125

[19] T. Würfl, F.C. Ghesu, V. Christlein and A. Maier, Deep Learning Computed Tomography, in Medical Image Computing and Computer-Assisted Intervention - MICCAI 2016, S. Ourseline et al., Springer International Publishing (2016), p. 432– 440

[20] M.L. Yahiaoui, F. Kharfi and L. Boukerdja, Resolution enhancement of neutron radiography image using combined SRCNN-POCS method, Nucl. Instrum. Meth. A 1050 (2023) 168123.

[21] Ma Y, Li H, Yin W, et al. Enhancing the spatial resolution of neutron radiography with generative adversarial networks[J]. Journal of Instrumentation, 2025, 20(03): P03003.

[22] Ma Y, Li H, Yin W, et al. Enhancement of neutron radiography resolution through residual hybrid attention network[J]. Nondestructive Testing and Evaluation, 2025: 1-19.

[23] Liu X, Zhao C, Qiao S, et al. Neutron-image super-resolution based on convolutional neural networks and novel parallel transformer[J]. Nuclear Engineering and Technology, 2025, 57(5): 103357.

[24] Wu H, Wang Q, Song J, et al. Super-Resolution Reconstruction for Neutron Radiography Using Improved Real-ESRGAN[J]. IEEE Transactions on Nuclear

[25] Koerner S, Schillinger B, Vontobel P, et al. A neutron tomography facility at a low power research reactor[J]. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 2001, 471(1-2): 69-74

Science, 2026,73(2): 460-473.