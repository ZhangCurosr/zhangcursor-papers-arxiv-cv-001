# LAYERWISE TUNABLE LIFTING SCHEME FOR THE CONVOLUTIONAL NEURAL NETWORK

Abdumannon Yovkochov<sup>1</sup>, An Le<sup>1</sup>, Sungbal Seo<sup>2</sup>, You-Suk Bae<sup>2</sup>, Truong Nguyen<sup>1</sup>

<sup>1</sup>Electrical and Computer Engineering Department, University of California San Diego, La Jolla, CA 92093, USA {ayovkochov, d0le, tqn001}@ucsd.edu

<sup>2</sup>Department of Computer Engineering, Tech University of Korea, Siheung 15073, Korea {sungbal, ysbae}@tukorea.ac.kr

Abstract—This work introduces a family of tunable lifting schemes for biorthogonal wavelet filter banks. We propose three lifting strategies: low-pass tuning (LS-LayLatt-LP), high-pass tuning (LS-LayLatt-HP), and a sequential lifting scheme that jointly adapts low- and high-frequency branches (LS-LayLatt-Sequential). All proposed designs are formulated using a latticebased lifting structure [1], which guarantees invertibility and stability for arbitrary parameter values within the lifting functions. We evaluated the proposed methods by integrating them into a ResNet-18 [2] backbone for image classification on the Describable Textures Dataset (DTD) [3], as well as for anomaly detection on hazelnut images from the MVTec-AD [4] dataset and private KRC102S dataset. Experimental results demonstrate consistent performance improvements across all evaluated tasks.

Index Terms—Anomaly detection, Computer vision, Discrete wavelet transforms, Image processing, Wavelet transform

## I. INTRODUCTION

Convolutional Neural Networks (CNNs) such as ResNet [2], VGG [5] and DenseNet [6] have played a major role in computer vision tasks. Downsampling operations, such as max pooling, average pooling, and strided convolution, are considered among the most important components of these architectures [7]. However, these operations are deterministically lowpass, which leads to the loss of fine-grained information, introducing aliasing effects [7]. To address these aliasing effects, many architectures have been proposed that use frequencydomain [8], [9] or wavelet-based approaches [10] as alternatives. Nevertheless, these methods typically utilize only lowfrequency components or retain only low-pass information. As emphasized in Fig. 1, high-frequency components of images contain critical information for classification and detection. Consequently, Wavelet-Attention [11], OrthLatt-UwU [12], LS-BiorUwU [1], BiorLatt-UwU [13] models were proposed to incorporate high-frequency information into the network.

To address the aforementioned challenges and enhance flexibility, we propose a family of tunable biorthogonal wavelet lifting schemes for CNN downsampling operations. First, we investigate high-pass tuning to improve fine-detail preservation during the wavelet transform. In addition, low-pass tuning is studied to enhance coarse feature learning. Finally, we introduce a sequential lifting scheme that adaptively tunes both frequency branches while preserving perfect reconstruction. The proposed methods are integrated into a ResNet-18 backbone and evaluated on the DTD dataset [3] for image classification, as well as on the MVTec-AD (hazelnut) [4] and private KRC102S datasets for anomaly detection using CFLOW-AD [14]. These works have demonstrated that latticebased parametrization and lifting schemes show an improvement on classification tasks compared to baseline methods.

![](images/64430674c5f181e6cd4f495f7a55cc0bfcf2ef86dad1b45d9d6e16bf274281cf.jpg)  
Fig. 1. Wavelet (Haar) and frequency representations of samples from MVTecAD-hazelnut [4], DTD-cracked [3], and KRC102S (left to right). The figure highlights the role of high-frequency components in preserving discriminative visual information. In particular, for the DTD-cracked sample, retaining only the low-frequency component X<sub>ll</sub> removes most crack structures, which are primarily encoded in high-frequency subbands. This illustrates how suppressing high-frequency details can lead to a significant loss of information critical for accurate classification and anomaly detection.

## II. RELATED WORKS

Recent works have explored wavelet-based approaches for downsampling, pooling, and convolutional layers to improve performance on image classification and anomaly detection tasks. One of the earliest adopters of wavelet transforms for pooling layers is WaveCNet [10], which integrates discrete wavelet transforms (DWT) into CNN architectures using fixed orthogonal or biorthogonal wavelet families for DWT/IDWT. While effective, this fixed structure limits design flexibility. To address these limitations, subsequent works have proposed tunable and learnable wavelet constructions. For instance, OrthLatt-UwU [12] and BiorLatt-UwU [1] propose latticebased orthogonal and biorthogonal wavelet units, respectively, with trainable coefficients and guaranteed perfect reconstruction, demonstrating notable improvements in computer vision tasks. However, orthogonal and biorthogonal wavelet structure used in OrthLatt-UwU [12] and BiorLatt-UwU [13] restricts the filter design to equal filter lengths, limiting the ability to emphasize higher or lower frequency components. LS-BiorUwU [1] attempts to tackle this problem by building wavelets using a lifting scheme which allows unequal lengths for high-pass and low-pass filters. Although the lifting scheme is used to tune the high-pass filter in the LS-BiorUwU [1] method, the associated coefficients remain fixed by design throughout training.

In contrast, our method enables layer-wise tuning of lifting parameters, allowing each downsampling layer to learn its own wavelet parameters through cross-entropy–based optimization, while also supporting longer low-pass or high-pass filter lengths by design. This design allows different pooling layers to adapt their frequency responses according to their roles in the network hierarchy, leading to more effective feature extraction.

## III. PROPOSED METHODS

## A. General Formulation

Unlike standard orthogonal wavelets, in our work we use a biorthogonal wavelet, which relaxes orthogonality to biorthogonality, allowing filters to have unequal lengths. We use a lifting scheme to construct tunable biorthogonal wavelet filter banks. As illustrated in Fig. 2, the general form of lifting scheme consists of two lifting functions, $P _ { k } ( z )$ and $U _ { k } ( z )$ , each responsible for tuning the high-pass and lowpass filters of the wavelet, respectively. We can construct the comprehensive lifting scheme for K steps using the following equation:

$$
\prod _ { k = 0 } ^ { K - 1 } \left[ P _ { k } ( z ^ { 2 } ) \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { P _ { k } ( z ^ { 2 } ) } } & { { 1 } } \end{array} \right] \left[ \begin{array} { c c } { { 1 } } & { { U _ { k } ( z ^ { 2 } ) } } \\ { { 0 } } & { { 1 } } \end{array} \right]\tag{1}
$$

where we define $P _ { k } ( z ) ~ = ~ - a _ { k } + a _ { k } z ^ { - 2 k }$ and $U _ { k } ( z ) ~ =$ $- b _ { k } + b _ { k } z ^ { - 2 k }$ . Since both lifting matrices have unity on the diagonal, the determinant of each matrix is 1. Consequently, the determinant of the total lifting scheme is monomial, implying that the lifting structure is invertible. This ensures that, for any arbitrary choice of $a _ { k }$ and $b _ { k }$ in the lifting steps, the system satisfies the biorthogonality condition, and the synthesis filters can be exactly reconstructed by inverting the analysis operations in reverse order.

![](images/5277891edf2f4b595c75872edc3150fcff96f651534ebd968fb52570d5c92fb0.jpg)  
Fig. 2. Lifting Scheme Structure for the analysis part of the filter bank, with lifting functions $U _ { k } ( z ^ { 2 } )$ and $P _ { k } ( z ^ { 2 } )$

TABLE I  
FILTER COEFFICIENTS AFTER 1-STEP HIGH-PASS TUNING
<table><tr><td rowspan=1 colspan=1>Filter</td><td rowspan=1 colspan=1>Coefficients</td></tr><tr><td rowspan=1 colspan=1> $\overline { { h _ { 0 } } }$ </td><td rowspan=1 colspan=1>0.7071, 0.7071</td></tr><tr><td rowspan=1 colspan=1> $\overline { { h _ { 1 } } }$ </td><td rowspan=1 colspan=1>-0.1183, -0.1183, 0.7071, -0.7071, 0.1183, 0.1183</td></tr></table>

Additionally, as shown in Table I, performing the lifting scheme does not break the finite impulse response (FIR) and linear-phase properties of the wavelet filters. Let $\widehat { H } _ { 0 } ^ { k - 1 } ( z )$ and $\widehat { H } _ { 1 } ^ { k - 1 } ( z )$ denote the analysis filters at lifting step $k - 1$ and assume they are FIR and linear phase. Since the lifting scheme consists only of delays and finite-length filtering operations, finite support is preserved at every step. The newly constructed wavelet filter bank also preserves linear phase because each lifting operation modifies existing linear-phase filters through delayed linear combinations with real-valued coefficients, which introduce only constant phase shifts. The alignment delays in the lifting structure ensure a consistent phase center across all terms.

## B. High-Pass Tuning (LS-LayLatt-HP)

We first explored a tunable lifting scheme which increases the high-pass filter length by using the low-pass filter. This method helps the high-pass filter achieve greater adaptability to high-frequency detail features of an image, such as texture and edges. By using the lifting structure mentioned above, we can construct this method by setting the parameters for $U _ { k } = 0$ , and assigning distinct learnable coefficients $a _ { k }$ to the lifting function $P _ { k } ( z )$ for k steps. We can represent this using the following recursive equation:

$$
\begin{array} { l } { { [ \hat { H } _ { 0 } ^ { k } ( z ) ] = [ \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { R _ { k } ( z ^ { 2 } ) } } & { { 1 } } \end{array} ] [ 1 } } & { { 0 } } \\ { { [ \hat { H } _ { 1 } ^ { k } ( z ) ] = [ P _ { k } ( z ^ { 2 } ) ] [ 0 } } & { { z ^ { - 2 } ] [ \hat { H } _ { 1 } ^ { k - 1 } ( z ) ] } } \\ { { \mathrm { } } } & { { \hat { H } _ { 0 } ^ { k - 1 } ( z ) } } \\ { { \mathrm { } } } & { { [ - a _ { k } \hat { H } _ { 0 } ^ { k - 1 } ( z ) + z ^ { - 2 } \hat { H } _ { 1 } ^ { k - 1 } ( z ) + a _ { k } z ^ { - 4 k } \hat { H } _ { 0 } ^ { k - 1 } ( z ) ] } } \end{array}\tag{2}
$$

where $\widehat { H _ { 0 } }$ and $\widehat { H _ { 1 } }$ are the low-pass and high-pass filters of the wavelet filter bank, and k is in the range from 1 to N.

## C. Low-Pass Tuning (LS-LayLatt-LP)

We next investigate a complementary ”lifting-down” scheme with tunable parameters, aimed at enhancing low-frequency representation learning within CNN architectures. While highpass tuning emphasizes fine-scale and edge-related features, adaptive low-pass wavelet tuning is important for capturing coarse structures, global context, and long-range dependencies that are essential for robust feature hierarchies. Motivated by prior observations that tunable lifting parameters improve overall representation capacity, we extend the same principle to the low-pass branch of the wavelet filter bank. We construct the lifting scheme for low-pass tuning in a manner similar to high-pass tuning. First, we set the coefficients for $P _ { k } ( z ) = 0$ and assign a learnable parameter $b _ { k }$ to the lifting function $U _ { k } ( z )$ for k steps. Furthermore, we can express the entire lifting function using the following recursive form:

$$
\begin{array} { l } { { \left[ \widehat { H } _ { 0 } ^ { k } ( z ) \right] = \left[ \displaystyle { 1 } \quad U _ { k } ( z ^ { 2 } ) \right] \left[ z ^ { - 2 } \quad 0 \right] \left[ \widehat { H } _ { 0 } ^ { k - 1 } ( z ) \right] } } \\ { { \left[ \widehat { H } _ { 1 } ^ { k } ( z ) \right] = \left[ 0 \quad 1 \quad 1 \right] \left[ \displaystyle { 0 } \quad 1 \right] \left[ \widehat { H } _ { 1 } ^ { k - 1 } ( z ) \right] } } \\ { { = \left[ - b _ { k } \widehat { H } _ { 1 } ^ { k - 1 } ( z ) + z ^ { - 2 } \widehat { H } _ { 0 } ^ { k - 1 } ( z ) + b _ { k } z ^ { - 4 k } \widehat { H } _ { 1 } ^ { k - 1 } ( z ) \right] } } \\ { { \qquad \widehat { H } _ { 1 } ^ { k - 1 } ( z ) } } \end{array}\tag{3}
$$

## D. Sequential Tuning (LS-LayLatt-Sequential)

Further, we explored a sequential lifting scheme in which the lifting operations are applied in a staged manner. Specifically, the low-pass branch of the wavelet transform is first updated using a tunable lifting step. The resulting updated low-pass representation is then used to guide a subsequent lifting operation that updates the high-pass branch. This sequential dependency introduces stronger coupling between low-frequency and high-frequency components, enabling more expressive and adaptive wavelet decompositions. This design helps construct longer and more flexible filters without explicitly increasing the filter size or violating the perfect reconstruction constraints of the wavelet filter bank. We construct the new wavelet filter bank using a recursive form below:

$$
\begin{array} { r l } & { \left[ \widehat { H } _ { 0 } ^ { k } ( z ) \right] = \left[ \begin{array} { c c } { 1 } & { 0 } \\ { P _ { k } ( z ^ { 2 } ) } & { 1 } \end{array} \right] \left[ \begin{array} { c c } { 1 } & { 0 } \\ { 0 } & { z ^ { - 2 } } \end{array} \right] \left[ \begin{array} { c c } { 1 } & { U _ { k } ( z ^ { 2 } ) } \\ { 0 } & { 1 } \end{array} \right] } \\ & { \qquad \cdot \left[ \begin{array} { c c } { z ^ { - 2 } } & { 0 } \\ { 0 } & { 1 } \end{array} \right] \left[ \widehat { H } _ { 0 } ^ { k - 1 } ( z ) \right] } \\ & { \qquad = \left[ z ^ { - 2 } \widehat { H } _ { 0 } ^ { k - 1 } ( z ) + U _ { k } ( z ^ { 2 } ) \widehat { H } _ { 1 } ^ { k - 1 } ( z ) \right] } \\ & { \qquad = \frac { 2 } { z ^ { - 2 } } \widehat { H } _ { 1 } ^ { k - 1 } ( z ) + P _ { k } ( z ^ { 2 } ) \widehat { H } _ { 0 } ^ { k } ( z ) } \end{array}\tag{4}
$$

for k in the range from 1 to N, where $P _ { k } ( z ) = - a _ { k } + a _ { k } z ^ { - 2 k }$ and $U _ { k } ( z ) = - b _ { k } + b _ { k } z ^ { - 2 k }$ , and where $a _ { k }$ and $b _ { k }$ are tunable coefficients for the lifting scheme.

## E. 2D Implementation

To integrate the proposed lifting schemes into CNN architectures, we extend the one-dimensional (1D) lifting formulation to the two-dimensional (2D) domain using a separable wavelet transform. The learned 1D low-pass and high-pass filters are applied successively along the horizontal and vertical dimensions of the input feature map. This process produces four subbands, $\mathbf { X } _ { L L } , \mathbf { X } _ { L H } , \mathbf { X } _ { H L }$ , and $\mathbf { X } _ { H H }$ , corresponding to the low–low, low–high, high–low, and high–high frequency components, respectively. For a formulation suitable for deep learning, we construct low-pass L and high-pass H transform matrices from the learned filters $\widehat { H } _ { 0 } ^ { k }$ and $\widehat { H } _ { 1 } ^ { k }$ as

$$
{ \bf L } = { \bf D } { \bf C } _ { h _ { 0 } } , \quad { \bf H } = { \bf D } { \bf C } _ { h _ { 1 } }\tag{5}
$$

where D denotes the downsampling matrix, and $\mathbf { C } _ { h _ { 0 } }$ and $\mathbf { C } _ { h }$ are Toeplitz convolution matrices constructed from the

coefficients of the low-pass and high-pass filters, respectively. Using these matrices, the 2D discrete wavelet transform can be expressed as

$$
\begin{array} { r } { \mathbf { X } _ { L L } = \mathbf { L X L } ^ { T } , \quad \mathbf { X } _ { L H } = \mathbf { L X H } ^ { T } , } \\ { \mathbf { X } _ { H L } = \mathbf { H X L } ^ { T } , \quad \mathbf { X } _ { H H } = \mathbf { H X H } ^ { T } } \end{array}\tag{6}
$$

where X denotes the input feature map.

## F. CNN Implementation

The proposed methods are integrated into the ResNet family by replacing all conventional downsampling operations with the proposed tunable wavelet-based downsampling modules. Unlike fixed pooling layers in conventional CNNs, the lifting parameters in our method are trained jointly with the rest of the network using a cross-entropy loss function. This allows each downsampling layer to learn frequency responses that are optimized for the specific task for which it is used.

## IV. EXPERIMENTS AND RESULTS

We integrated the proposed methods into the ResNet-18 architecture and evaluated different lifting schemes on the DTD dataset [3]. Furthermore, the resulting pretrained models were incorporated into the CFLOW-AD [14] anomaly detection pipeline and applied to the MVTec-AD [4] hazelnut dataset as well as the KRC102S dataset. Across all datasets and experimental settings, the proposed models consistently outperformed the baseline methods.

## A. Image Classification: DTD

To evaluate the texture classification capabilities of the proposed tunable lifting schemes, we use the DTD dataset [3]. The dataset consists of 5,640 high-resolution images across 47 texture categories. Given that texture features mainly consist of high-frequency components, DTD serves as a good benchmark to evaluate our methods’ effectiveness.

All experiments were conducted using a ResNet-18 [2] backbone. We compare our proposed LS-LayLatt (High-Pass, Low-Pass Tuning and Sequential Tuning) against multiple baseline models: 1) ResNet-18 [2] with the standard pooling layers, 2) WaveCNet [10] model with Resnet18 backbone, 3) LS-BiorUwU [1] method which uses lifting scheme based method for high-pass filters, 4) the Orthogonal Lattice method (OrthLatt-UwU) [12], and 5) the Biorthogonal Lattice method with equal filter lengths (BiorLatt-UwU) [13].

1) LS-LayLatt Tuning Across different Strategies: Table II summarizes the classification performance of different proposed lifting scheme strategies on the DTD dataset [3].

We first focus on enhancing the high-frequency components of the wavelet filter bank, which stores a significant amount of information for texture classification. Compared to LS-BiorUwU-3Steps [1] (43.50%), our LS-LayLatt-HP introduces trainable lifting coefficients and learns a distinct wavelet per layer, reaching 45.32%, an 11.47 point gain over ResNet-18 and an improvement over OrthLatt-UwU-4Taps [12]. Although the BiorLatt-UwU-6Taps [13] method achieves slightly higher accuracy, it relies on a globally shared wavelet across all pooling layers and also requires the equal length by design.

TABLE II  
CLASSIFICATION ACCURACY (%) ON THE DTD DATASET OVER 5 RANDOM SEEDS FOR EACH MEASUREMENT.
<table><tr><td rowspan=1 colspan=2>DTD</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Accuracy (%)</td></tr><tr><td rowspan=1 colspan=1>ResNet-18 (Baseline) [2]</td><td rowspan=1 colspan=1>33.85 (±0.15)</td></tr><tr><td rowspan=1 colspan=1>WaveCNet [10]</td><td rowspan=1 colspan=1>26.70(±0.17)</td></tr><tr><td rowspan=1 colspan=1>LS-BiorUwU-3Steps [1]</td><td rowspan=1 colspan=1>43.50 (±0.21)</td></tr><tr><td rowspan=1 colspan=1>OrthLatt-UwU-4Taps [12]</td><td rowspan=1 colspan=1>44.51 (±0.48)</td></tr><tr><td rowspan=1 colspan=1>BiorLatt-UwU-6Taps [13]</td><td rowspan=1 colspan=1>45.63 (±0.37)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-HP-1 Step (ours)</td><td rowspan=1 colspan=1>42.59(±0.25)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-HP-2 Steps (ours)</td><td rowspan=1 colspan=1>44.21 (±0.13)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-HP-3 Steps (ours)</td><td rowspan=1 colspan=1>45.32 (±0.37)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-LP-1 Step (ours)</td><td rowspan=1 colspan=1>42.79 (±0.18)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-LP-2 Steps (ours)</td><td rowspan=1 colspan=1>43.75 (±0.21)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-LP-3 Steps (ours)</td><td rowspan=1 colspan=1>43.42 (±0.41)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-Sequential-1 Step (ours)</td><td rowspan=1 colspan=1>45.27 (±0.28)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-Sequential-2 Steps (ours)</td><td rowspan=1 colspan=1>46.12 (± 0.39)</td></tr></table>

Furthermore, low-pass tuning (LS-LayLatt-LP) expands the low-frequency receptive field and consistently outperforms ResNet-18 [2], but underperforms LS-LayLatt-HP, indicating that high-frequency cues remain the dominant signal for texture recognition. Sequential lifting (LS-LayLatt-Sequential) jointly extends both branches and achieves the best overall result in Table II, confirming that joint adaptation of both frequency branches produces the most expressive filters.

## B. Anomaly Detection: MVTecAD and KRC102S

For the anomaly detection task, we used the CFLOW-AD framework [14], a normalizing flow-based model, with modified or baseline Resnet18 [2] acting as a feature extractor. The feature extractor used for the CFLOW-AD framework [14] was pretrained on the DTD [3] dataset to leverage texturespecific features. The model was evaluated on the public MVTec-AD Hazelnut [4] class and the private KRC102S dataset.

We compare our proposed methods against five state-ofthe-art baselines on the MVTec-AD Hazelnut [4] dataset. The MVTec-AD Hazelnut dataset consists of images with various surface defects and contains 391 defect-free training samples and 110 test images (40 normal and 70 defective).

As demonstrated in Table III, our layer-specific tunable methods significantly outperform these baselines on MVTec-AD hazelnut. Specifically, the LS-LayLatt-Sequential (1-Step) method achieves the best image-level detection performance with an AUROC of 99.75%, while the LS-LayLatt-LP (2- Steps) method outperforms all other approaches in the segmentation task. Qualitative results on the MVTec-AD Hazelnut dataset are shown in Fig. 3, where anomaly heat maps and corresponding segmentation outputs are visualized. Although all compared methods are able to detect the anomalies, our proposed layer-specific tunable approaches produce highly localized segmentation maps.

We further evaluated our methods on the KRC102S dataset, collected by Tech University of Korea. The dataset consists of

TABLE III  
DETECTION (AUROC) AND SEGMENTATION (AUROC) PERFORMANCE ON THE MVTEC-AD-HAZELNUT DATASET OVER 5 RANDOM SEEDS FOR EACH MEASUREMENT.
<table><tr><td rowspan=1 colspan=3>CFLOW-AD Mvtec-AD hazelnut</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>DET-AUROC</td><td rowspan=1 colspan=1>SEG-AUROC</td></tr><tr><td rowspan=1 colspan=1>ResNet-18 [2] (Baseline)</td><td rowspan=1 colspan=1>92.43 (± 0.12)</td><td rowspan=1 colspan=1>96.45 (± 0.04)</td></tr><tr><td rowspan=1 colspan=1>WaveCNet [10]</td><td rowspan=1 colspan=1>94.78 (± 0.17)</td><td rowspan=1 colspan=1>98.33 (± 0.03)</td></tr><tr><td rowspan=1 colspan=1>LS-BiorUwU-3Steps [1]</td><td rowspan=1 colspan=1>95.47 (± 0.12)</td><td rowspan=1 colspan=1>97.56 (± 0.02)</td></tr><tr><td rowspan=1 colspan=1>OrthLatt-UwU-4Taps [12]</td><td rowspan=1 colspan=1>90.11 (± 0.21)</td><td rowspan=1 colspan=1>97.17 (± 0.05)</td></tr><tr><td rowspan=1 colspan=1>BiorLatt-UwU-6Taps [13]</td><td rowspan=1 colspan=1>88.62 (± 0.14)</td><td rowspan=1 colspan=1>97.09 (± 0.04)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-LP 2 Steps</td><td rowspan=1 colspan=1>99.35 (± 0.15)</td><td rowspan=1 colspan=1>98.54 (± 0.05)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-HP 2 Steps</td><td rowspan=1 colspan=1>98.37 (± 0.11)</td><td rowspan=1 colspan=1>98.31 (± 0.02)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-Sequential 1 Step</td><td rowspan=1 colspan=1>99.75 (± 0.18)</td><td rowspan=1 colspan=1>98.47 (± 0.07)</td></tr></table>

TABLE IV  
ANOMALY DETECTION ACCURACY FOR KRC102S OVER 3 RANDOM SEEDS FOR EACH MEASUREMENT.
<table><tr><td rowspan=1 colspan=2>KRC102S-TUKPCB</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>DET-AUROC</td></tr><tr><td rowspan=1 colspan=1>ResNet-18(Baseline)</td><td rowspan=1 colspan=1>89.29 (±0.43)</td></tr><tr><td rowspan=1 colspan=1>WaveCNet</td><td rowspan=1 colspan=1>85.74 (±0.52)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-LP 2 Steps</td><td rowspan=1 colspan=1>88.81 (±0.47)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-HP 2 Steps</td><td rowspan=1 colspan=1>89.43 (±0.37)</td></tr><tr><td rowspan=1 colspan=1>LS-LayLatt-Sequential 1 Step</td><td rowspan=1 colspan=1>92.21 (±0.53)</td></tr></table>

Printed Circuit Board (PCB) component images and contains 4,346 anomaly-free training images and 248 test images (60 normal and 188 with various minor defects). Similar to the evaluation on the MVTec-AD Hazelnut dataset [4], we used a ResNet-18 [2]-based encoder with different pooling methods as a feature extractor for the CFLOW-AD [14] model. Due to the absence of pixel-level ground-truth masks, evaluation was restricted to image-level detection AUROC. As shown in Table IV, the low-pass tuning method underperformed compared to the ResNet-18 baseline; however, the sequential lifting scheme achieved the best performance, outperforming both the baseline and the other tunable configurations.

![](images/84d4ced070d2048c6dd249f1bd1e9a1e20ca2610893a29345e9ac46d3713ea0f.jpg)  
Fig. 3. Segmentation and heatmaps generated by the CFLOW-AD method on the MVTec AD hazelnut dataset using a pretrained proposed and baseline methods with ResNet-18 backbone.

## C. Lifting Parameters Analysis

In this section, we examine the importance of parameter initialization for lifting-scheme coefficients. Since the lifting parameters are optimized using gradient-based methods with a cross-entropy loss, proper initialization is crucial for stable training and effective convergence. For all proposed methods, we initialize the wavelet filters using the Haar/Bior1.1 [15] wavelet. For LS-LayLatt-HP, we use a coefficient initialization strategy similar to LS-BiorUwU [1], where the lifting coefficients are initialized to approximate Bior1.3 and Bior1.5 wavelets for one and two lifting steps, respectively. This setup allows us to start from well-understood frequency response characteristics. For the lifting steps greater than two, additional coefficients are initialized with values close to zero, allowing the network to gradually learn higher-order refinements without introducing strong initial perturbations.

For LS-LayLatt-LP, the low-pass lifting coefficients are computed directly from the final filters obtained using LS-LayLatt-HP via the perfect-reconstruction relations between the analysis and synthesis filter banks. This approach provides a structured and effective initialization for texture-based classification tasks. In the sequential lifting scheme, the high-pass and low-pass branches are initialized using the coefficients learned from the LS-LayLatt-HP and LS-LayLatt-LP models, respectively. To avoid excessive filter growth, the number of lifting steps in the sequential scheme is limited to two.

## D. Computational Complexity and Experiment Setup

The computational complexity of the learnable lifting scheme introduces only 2K learnable scalars per downsampling site for K lifting steps and is therefore practically free in both parameters and floating-point operations. The observed overhead primarily from the 4× channel widening required to fuse the four wavelet subbands prior to the subsequent convolution, an architectural cost shared by all wavelet-pooling baselines [1], [12], [13]. Relative to the standard ResNet-18 backbone (11.7M parameters), the proposed LS-LayLatt variants reach 21.5M parameters, and these values are essentially independent of the number of lifting steps K since the per-site fusion convolution dominates the cost.

In total, we trained our LS-LayLatt models on the DTD dataset for 300 epochs, using a stage-wise transfer learning strategy [16] with three stages of 100 epochs each. At the beginning of each stage, the model was initialized from the checkpoint achieving the best validation performance in the previous stage. Training was performed with a batch size of 4 using stochastic gradient descent. The initial learning rate was set to 0.01, and a step learning rate decay was applied every 30 epochs. Data augmentation was performed using random resized cropping and horizontal flipping during training.

## V. CONCLUSION

In conclusion, the integration of tunable lifting schemes into CNN architectures consistently demonstrates improved performance in both image classification and anomaly detection tasks. By incorporating learnable lifting parameters within biorthogonal wavelet filter banks, the proposed methods improve the network’s ability to capture both high-frequency and low-frequency features in a more task-specific manner. Each lifting-based method exhibits measurable performance gains compared to conventional architectures, confirming the effectiveness of wavelet-domain tuning for feature extraction. Moreover, the flexibility of the proposed lifting schemes, enables better spectral control without violating perfect reconstruction constraints. These results show that tuning wavelet lifting parameters provides an effective approach for improving representation learning and achieving better performance across different computer vision tasks. Additionally, future work will focus on extending the proposed lifting strategy to be able to support different base wavelet filter banks with varying filter lengths.

## REFERENCES

[1] A. Le, H. Nguyen, S. Seo, Y.-S. Bae, and T. Nguyen, “Biorthogonal tunable wavelet unit with lifting scheme in convolutional neural network,” in 2025 33rd European Signal Processing Conference (EUSIPCO), 2025, pp. 1807–1811.

[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” 2015. [Online]. Available: https://arxiv.org/abs/1512.03385

[3] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi, “Describing textures in the wild,” in Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014.

[4] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9584–9592. [Online]. Available: https://ieeexplore.ieee.org/document/8954181

[5] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations, 2015.

[6] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2261–2269.

[7] R. Zhang, “Making convolutional networks shift-invariant again,” in Proceedings of the International Conference on Machine Learning (ICML), 2019.

[8] O. Rippel, J. Snoek, and R. P. Adams, “Spectral representations for convolutional neural networks,” in Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 2, ser. NIPS’15. Cambridge, MA, USA: MIT Press, 2015, p. 2449–2457.

[9] R. Riad, O. Teboul, D. Grangier, and N. Zeghidour, “Learning strides in convolutional neural networks,” ICLR, 2022.

[10] Q. Li, L. Shen, S. Guo, and Z. Lai, “Wavecnet: Wavelet integrated cnns to suppress aliasing effect for noise-robust image classification,” IEEE Transactions on Image Processing, vol. 30, p. 7074–7089, 2021. [Online]. Available: http://dx.doi.org/10.1109/TIP.2021.3101395

[11] Z. Xiangyu, “Wavelet-attention cnn for image classification,” 2022. [Online]. Available: https://arxiv.org/abs/2201.09271

[12] A. D. Le, S. Jin, Y.-S. Bae, and T. Q. Nguyen, “A lattice-structure-based trainable orthogonal wavelet unit for image classification,” IEEE Access, vol. 12, pp. 88 715–88 727, 2024.

[13] A. D. Le, S. Jin, S. Seo, Y.-S. Bae, and T. Q. Nguyen, “Biorthogonal lattice tunable wavelet units and their implementation in convolutional neural networks for computer vision problems,” IEEE Open Journal of Signal Processing, vol. 6, pp. 768–783, 2025.

[14] D. A. Gudovskiy, S. Ishizaka, and K. Kozuka, “CFLOW-AD: real-time unsupervised anomaly detection with localization via conditional normalizing flows,” CoRR, vol. abs/2107.12571, 2021. [Online]. Available: https://arxiv.org/abs/2107.12571

[15] G. Strang and T. Nguyen, Wavelets and Filter Banks. Wellesley– Cambridge Press, 1996.

[16] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Transactions on Knowledge and Data Engineering, vol. 22, no. 10, pp. 1345– 1359, 2010.