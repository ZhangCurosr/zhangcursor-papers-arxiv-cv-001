# FlowMoDL: Model-Based Deep Learning with Conjugate-Gradient Data Consistency for Highly Accelerated 4D Flow MRI Reconstruction

Tristan Gottwald<sup>1,3,⋆[0009</sup>−<sup>0004</sup>−<sup>3550</sup>−<sup>9779]</sup>, Michelle Bruch<sup>4[0009</sup>−<sup>0006</sup>−<sup>2485</sup>−<sup>1741]</sup>, Mubashir-Ul Hassan<sup>1,2,3[0009</sup>−<sup>0005</sup>−<sup>1404</sup>−<sup>9176]</sup> Fatma Alickovic<sup>5[0009</sup>−<sup>0001</sup>−<sup>6885</sup>−<sup>0269]</sup>, Milan Kloiber<sup>1[0009</sup>−<sup>0005</sup>−<sup>9204</sup>−<sup>6106]</sup>, Daniel Tenbrinck<sup>4[0000</sup>−<sup>0002</sup>−<sup>4788</sup>−<sup>9332]</sup>, Torsten Panholzer<sup>5[0000</sup>−<sup>0002</sup>−<sup>2808</sup>−<sup>6120]</sup>, Melanie Schaller<sup>1,3[0000</sup>−<sup>0002</sup>−<sup>5708</sup>−<sup>4394]</sup>, and Jana Hutter<sup>1,2,3[0000</sup>−<sup>0003</sup>−<sup>3476</sup>−<sup>3500]</sup>

<sup>1</sup> Institute of Information Processing (tnt), Leibniz University Hannover, Germany <sup>2</sup> CAIMED, Leibniz University Hannover, Hannover, Germany

3 L3S, Leibniz University Hannover, Hannover, Germany <sup>4</sup> Department of Data Science, Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany

5 Institute of Medical Biostatistics, Epidemiology and Informatics, University Medical Center of the Johannes Gutenberg University Mainz, Germany <sup>⋆</sup>Corresponding author: gottwald@tnt.uni-hannover.de

Abstract. We present FlowMoDL, an unrolled neural network for highly accelerated 4D flow MRI reconstruction that directly optimizes for both anatomical magnitude and phase-derived velocity accuracy. Building on the MoDL framework, FlowMoDL alternates a learned (3+1)D spatiotemporal denoiser with conjugate-gradient data-consistency updates based on the SENSE forward model. A novel dual-pathway conditioning scheme adapts the denoiser features and data-consistency weighting, enabling a single model to handle varying acceleration factors (10 to 50 ). To ensure physiological accuracy, the network is trained using a deep-supervision composite loss that explicitly penalizes velocity magnitude and angular errors, stabilized by a curriculum schedule. We evaluate FlowMoDL on the multi-center CMRx4DFlow dataset against classical and deep-learning baselines (CG-SENSE, MoDL, FlowVN, and FlowMRI-Net). A key advantage of FlowMoDL is its superior gradient step eficiency. When evaluated under an equivalent, limited budget of gradient steps, competing flow-specific networks degrade significantly. In contrast, FlowMoDL robustly converges and strictly outperforms all competitors across all acceleration factors in magnitude SSIM, nRMSE, relative velocity error, and angular error, successfully recovering sharp structural details and temporally coherent velocity fields.

Keywords: 4D Flow MRI Reconstruction · Model-based reconstruction using Deep Learned priors · Deep Learning · Aorta.

## 1 Introduction

Four-dimensional flow magnetic resonance imaging (4D flow MRI) captures timeresolved, volumetric anatomical and three-directional blood-flow information across the cardiac cycle [8]. However, its extensive data requirements necessitate aggressive k-space undersampling [9, 13]. Reconstructing these highly accelerated acquisitions is challenging because the velocity field is derived entirely from relative phase diferences between velocity encodings. Standard reconstruction methods prioritizing magnitude [1,5] are therefore insuficient, as even marginal phase errors induce severe, non-physiological velocity deviations. Furthermore, existing flow-specific deep learning architectures rely on computationally heavy recurrent mechanisms [6] or neglect explicitly optimizing for phase [15].

To address these limitations, we introduce FlowMoDL, a phase-aware unrolled neural network explicitly designed for highly accelerated 4D flow MRI. Built upon the MoDL framework [1], FlowMoDL couples a (3+1)D spatiotemporal regularizer with conjugate-gradient data-consistency updates. Our primary contributions are: 1) a hybrid architecture that preserves critical inter-encoding phase relationships; 2) a dual-pathway conditioning mechanism that adapts denoiser features and data-consistency weighting to handle acceleration rates from 10 to 50 with a single trained model; 3) a curriculum-guided composite loss optimizing directly for flow and 4) state-of-the-art performance and gradientstep eficiency, strictly outperforming classical and specialized baselines on the multi-center CMRx4DFlow dataset.

The code is available at https://github.com/tgottwald/FlowMoDL.

## 2 Method

## 2.1 Problem formulation

We formulate the highly accelerated 4D flow MRI reconstruction [6,15] as follows. Given the undersampled multi-coil k-space measurements $\mathbf { y } \in \mathbb { C } ^ { N _ { v } \times N _ { c } \times N _ { t } \times N _ { z } \times N _ { y } \times N _ { x } }$ , coil sensitivity maps $S \in \dot { \mathbb { C } } ^ { N _ { c } \times N _ { z } \times N _ { y } \times N _ { x } }$ , and an undersampling mask $M \in \{ 0 , 1 \} ^ { N _ { t } \times N _ { z } \times N _ { y } }$ , we aim to reconstruct the MRI volume sequence $\mathbf { x } \in \mathbb { C } ^ { N _ { v } \times N _ { t } \times N _ { z } \times N _ { y } \times N _ { x } }$ and derive the corresponding 4D flow velocity field. Here, $N _ { v }$ denotes the number of velocity encodings, $N _ { t }$ the number of cardiac phases, $N _ { c }$ the number of coils, and $N _ { z } , N _ { y }$ , and $N _ { x }$ the spatial dimensions. We consider $N _ { v } = 4$ , where the first encoding serves as the reference encoding and the phase diferences between the remaining three encodings and the reference yield the directional velocity components over the cardiac cycle [8]. Acquisition is modeled using the SENSE [12] forward operator A:

$$
A \mathbf { x } = M { \mathcal { F } } ( S \mathbf { x } ) ,\tag{1}
$$

where $\mathcal { F }$ corresponds to the spatial Fourier transform. The adjoint operator of $A$ is referred to as $A ^ { \mathsf { H } }$

In contrast to existing 4D flow approaches based on learned multidimensional filter-bank updates [15] or recurrent complex-valued updates that propagate information across cardiac phases and reconstruction iterations [6], we adopt the MoDL framework [1] to combine a spatiotemporal learned prior with conjugategradient data-consistency. The reconstruction problem is formulated as

![](images/58e19a10972a42d052625bae20626312a8ebca4df8a85a62f06b56be2018d816.jpg)  
Fig. 1: Detailed architecture of a single FlowMoDL cascade k (top) and the internal structure of its factorized $( 3 + 1 ) \mathrm { D }$ spatiotemporal blocks (bottom). The network maps the previous estimate $\hat { \mathbf { x } } ^ { ( k - 1 ) }$ to an intermediate denoised target $\mathbf { z } ^ { ( k ) }$ using a learned prior $\mathcal { D } _ { \theta _ { k } }$ equipped with a global residual connection. To improve the handling of varying acceleration rates, R is routed to both a FiLM generator within the denoiser and an MLP computing the data-consistency penalty weight $\lambda _ { k } ( R )$ . The cascade concludes by solving the linear data-consistency equations via conjugate-gradient iterations.

$$
\widehat { \mathbf { x } } = \arg \operatorname* { m i n } _ { \mathbf { x } } \frac { 1 } { 2 } \left\| A \mathbf { x } - \mathbf { y } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \left\| \mathbf { x } - \mathcal { D } _ { \theta } ( \mathbf { x } ) \right\| _ { 2 } ^ { 2 } ,\tag{2}
$$

where $\mathcal { D } _ { \theta }$ denotes a learned spatiotemporal denoiser and $\lambda > 0$ balances measurement consistency and the learned residual prior. FlowMoDL approximates this formulation by alternating learned denoising and conjugate-gradient dataconsistency updates across the unrolled cascades.

As velocity is derived from the relative phase between the directional and reference encodings, $v _ { i } \ \propto \ / ( \mathbf { x } _ { i } \mathbf { \overline { { x } } } _ { 0 } )$ , reconstruction must preserve inter-encoding phase. This motivates the proposed spatiotemporal architecture and the explicit penalization of velocity magnitude and direction errors in the training objective.

## 2.2 FlowMoDL

We propose FlowMoDL, a hybrid-unrolled neural network based on the Modelbased reconstruction using Deep Learned priors (MoDL) paradigm [1]. As shown in Figure 1, the reconstruction is unrolled into K cascades. The first cascade is initialized with the adjoint reconstruction $\hat { \mathbf { x } } ^ { ( 0 ) } = A ^ { \mathsf { H } } \mathbf { y }$

In each cascade $k \in \{ 1 , \ldots , K \}$ , the current estimate $\hat { \mathbf { x } } ^ { ( k - 1 ) }$ is mapped to a denoised intermediate $\mathbf { z } ^ { ( k ) }$ by the learned denoiser $\mathcal { D } _ { \theta _ { k } }$ . Unlike the original MoDL formulation, we condition the denoiser on the scalar acceleration factor R via feature-wise linear modulation (FiLM) [10]:

$$
\mathbf { z } ^ { ( k ) } = { \mathcal { D } } _ { \theta _ { k } } \bigl ( \hat { \mathbf { x } } ^ { ( k - 1 ) } , R \bigr )\tag{3}
$$

Holding $\mathbf { z } ^ { ( k ) }$ fixed, the subsequent data-consistency update obtains the cascade output by solving

$$
\left( A ^ { \mathsf { H } } A + \lambda _ { k } ( R ) I \right) \hat { \mathbf { x } } ^ { ( k ) } = A ^ { \mathsf { H } } \mathbf { y } + \lambda _ { k } ( R ) \mathbf { z } ^ { ( k ) } .\tag{4}
$$

This linear system is approximately solved using J iterations of the conjugate gradient (CG) method. To reduce the computational cost of each CG iteration, we transform the fully sampled readout axis into image space before the unrolled cascades, exploiting the undersampling mask being constant along this axis [4]. The resulting normal operator decomposes into independent 2D problems indexed by readout position, allowing each CG iteration to be performed using 2D FFTs while remaining equivalent to the full 3D data-consistency formulation.

Regularizer: To enable the network to compensate for higher acceleration factors compared to the original MoDL [1], each cascade utilizes an independent denoiser network, $\mathcal { D } _ { \theta _ { k } }$ . The denoiser processes the image by concatenating the real and imaginary parts of all velocity encodings along the channel axis, preserving cross-encoding phase relationships across layers. As detailed in the bottom panel of Figure 1, the backbone comprises N residual spatiotemporal blocks. Instead of separate hyperplane convolutions [15], each block employs a (3+1)D factorization [14]: a spatial 3D convolution per cardiac phase is sequentially followed by a ReLU activation and a temporal 1D convolution per voxel. This temporal convolution aims to capture temporal flow dynamics without requiring the recurrent hidden states used in FlowMRI-Net, which substantially slows down training [6]. As shown by the global skip connection around the denoiser in Figure 1, the zero-initialized final convolution is added to the denoiser input. Consequently, each denoiser initially returns its input, providing a stable initialization of the learned regularization pathway before the subsequent CG data-consistency update.

Data consistency: The update in Equation (4) is approximated using J conjugate-gradient iterations. Since we enforce $\lambda _ { k } ( R ) > 0$ , the system operator is Hermitian positive definite, ensuring that the corresponding linear system has a unique solution that is approximated by the unrolled CG iterations. Their diferentiable implementation allows gradients to propagate through the dataconsistency update.

Acceleration Conditioning: To handle varying acceleration factors with a single trained model, we introduce a dual-pathway conditioning scheme based on the log-scaled and normalized acceleration factor R. First, R is mapped by a FiLM generator to per-channel scales and shifts $( \gamma , \beta )$ , which are injected into the feature maps immediately following the denoiser stem and each subsequent residual block. Simultaneously, a separate Multi-Layer Perceptron (MLP) uses R to predict a per-sample ofset for each cascade’s regularization weight, $\lambda _ { k } ( R )$

This joint modulation allows both the denoiser features and the coupling between the reconstruction and denoised estimate to adapt to the acceleration factor. Both conditioning networks feature zero-initialized output layers, ensuring training begins entirely unconditioned.

Optimization Objective Approaches focused on flow reconstruction like [6,15] optimize their networks using only a weighted per-layer L1-loss. While the L1- loss implicitly optimizes phase, in practice it heavily biases the network toward magnitude-dominant reconstructions. FlowMoDL includes three additional loss terms specifically penalizing phase misalignment: a velocity error loss, a relative error loss, and an angular error loss. Given the chosen velocity encoding parameters $\mathbf { v } _ { \mathrm { e n c } } .$ the ground truth velocity vector field v and predicted velocity vector field $\hat { \mathbf { v } } ^ { ( k ) }$ are calculated from the complex-valued directional encodings (indices 1 to 3) and the reference encoding (index 0) as $\begin{array} { r } { \begin{array} { r c l } { \mathbf { v } } & { = } & { 1 / \pi \mathbf { v } _ { \mathrm { e n c } } \odot \angle \left( x _ { 1 : 3 } \overline { { \mathbf { x } } } _ { 0 } \right) } \end{array} } \end{array}$ Given the spatial segmentation mask $s \in \{ 0 , \mathrm { i } \} ^ { N _ { z } \times N _ { y } \times N _ { x } }$ delineating the ROI, which is broadcast across the temporal and velocity dimensions, the unnormalized per-cascade image and velocity L1 terms over the spatiotemporal domain $\varOmega$ are defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i m g } } ^ { ( k ) } = \displaystyle \sum _ { \Omega } s \odot \big \| \hat { \mathbf { x } } ^ { ( k ) } - { \mathbf { x } } \big \| _ { 1 } , \quad \mathcal { L } _ { \mathrm { v e l } } ^ { ( k ) } = \displaystyle \sum _ { \Omega } s \odot \big \| \hat { \mathbf { v } } ^ { ( k ) } - { \mathbf { v } } \big \| _ { 1 } , } \\ & { \qquad \quad \mathcal { L } _ { \mathrm { r e l } } ^ { ( k ) } = \sqrt { \frac { \displaystyle \sum _ { \Omega } s \odot \big ( \| { \mathbf { v } } \| _ { 2 } - \| \hat { \mathbf { v } } ^ { ( k ) } \| _ { 2 } \big ) ^ { 2 } } { \displaystyle \sum _ { \Omega } s \odot \| { \mathbf { v } } \| _ { 2 } ^ { 2 } + \varepsilon _ { \mathrm { r e l } } } } } \end{array}\tag{5}
$$

To penalize directional errors independently of magnitude, we introduce a voxelwise angular cosine-distance surrogate:

$$
\mathcal { L } _ { \mathrm { a n g } } ^ { ( k ) } = \sum _ { \Omega } s \odot \left( 1 - \cos \theta \right) , \quad \cos \theta = \mathrm { c l i p } \left( \frac { \hat { \mathbf { v } } ^ { ( k ) } \cdot \mathbf { v } } { \| \hat { \mathbf { v } } ^ { ( k ) } \| _ { 2 } \| \mathbf { v } \| _ { 2 } + \varepsilon _ { \mathrm { a n g } } } , - 1 , 1 \right) .\tag{6}
$$

The total loss is a deep-supervision composite combining these metrics:

$$
\mathcal { L } = \sum _ { k = 1 } ^ { K } w _ { k } \left[ \mathcal { L } _ { \mathrm { i m g } } ^ { ( k ) } + \omega _ { \mathrm { v e l } } \alpha _ { \mathrm { v } } ( e ) \mathcal { L } _ { \mathrm { v e l } } ^ { ( k ) } + \omega _ { \mathrm { r e l } } \alpha _ { \mathrm { v } } ( e ) \mathcal { L } _ { \mathrm { r e l } } ^ { ( k ) } + \omega _ { \mathrm { a n g } } \alpha _ { \theta } ( e ) \mathcal { L } _ { \mathrm { a n g } } ^ { ( k ) } \right] ,\tag{7}
$$

where k indexes the cascade output, e is the current epoch, and $w _ { k } = \exp \mathopen { } \mathclose \bgroup ( - \tau \mathopen { } \mathclose \bgroup ( K -$ $k ) )$ is an exponentially-decaying weight that concentrates gradients on later cascades as τ increases. $\mathrm { B y }$ including velocity and phase explicitly in the loss, the network optimizes directly for the targeted clinical metrics. To improve stability, the scalar multipliers $\alpha _ { \mathrm { v } } ( e )$ and $\alpha _ { \theta } ( e )$ apply a curriculum schedule [2]: the magnitude is optimized first, and the phase terms are ramped up subsequently. Training Procedure & Hyperparameters FlowMoDL is trained end-to-end using AdamW [7] with gradient clipping and a cosine-annealing learning rate schedule $( 1 0 ^ { - 3 } \ \mathrm { \dot { t o } \ 1 0 ^ { - 6 } ) }$ . SENSE operations and the CG solver compute in single precision, while the convolutional backbone uses mixed precision. Gradient checkpointing is used to fit training on a 48 GB GPU. The architecture unrolls

Table 1: Mean performance of reconstruction models on the unseen test set averaged across all acceleration factors $( R \in \{ 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \} )$ . The best results are highlighted in bold, and the second-best results are underlined.
<table><tr><td>Model</td><td>nRMSE (↓)</td><td>SSIM (↑)</td><td>RelErr (↓)</td><td>AngErr (↓)</td></tr><tr><td>CG-SENSE [11]</td><td>0.1592</td><td>0.6831</td><td>0.7087</td><td>45.6948</td></tr><tr><td>FlowVN [15]</td><td> $0 . 2 3 1 1 \pm 0 . 0 0 6 1$ </td><td> $0 . 4 6 8 7 \pm 0 . 0 2 0 0$ </td><td> $2 . 1 6 6 8 \pm 0 . 0 6 4 8$ </td><td> $7 9 . 8 4 8 1 \pm 0 . 4 2 3 8$ </td></tr><tr><td>FlowMRI-Net [6]</td><td> $0 . 2 1 2 6 \pm 0 . 0 5 4 0$ </td><td> $0 . 6 3 2 5 \pm 0 . 0 6 0 2$ </td><td> $0 . 8 6 6 0 \pm 0 . 0 7 4 1$ </td><td> $6 9 . 2 2 0 4 \pm 3 . 7 7 7 6$ </td></tr><tr><td>MoDL [1]</td><td> $\underline { { 0 . 1 1 4 7 } } \pm 0 . 0 0 3 5$ </td><td> $0 . 8 0 6 2 \pm 0 . 0 0 9 5$ </td><td> $0 . 7 4 1 2 \pm 0 . 0 1 2 6$ </td><td> $5 6 . 9 9 1 2 \pm 0 . 6 2 0 1$ </td></tr><tr><td>FlowMoDL (Ours)</td><td> $\mathbf { 0 . 0 4 6 9 } \pm 0 . 0 0 1 7$ </td><td>0.9409 ± 0.0026</td><td> $\mathbf { 0 . 2 6 6 4 } \pm 0 . 0 0 6 5$ </td><td> ${ \bf 2 4 . 5 6 3 9 } \pm 0 . 2 4 5 8$ </td></tr></table>

$K = 8$ untied cascades, each performing J = 10 CG iterations with an independent regularization weight (initialized to $\lambda _ { k } ( R ) = 0 . 0 5 )$ . The denoiser uses 64 channels, $N = 4$ residual blocks, a $3 \times 3 \times 3$ spatial kernel, and a length-5 temporal kernel. Loss weights are configured to $\omega _ { \mathrm { v e l } } = \omega _ { \mathrm { r e l } } = \omega _ { \mathrm { a n g } } = 0 . 1$ . The curriculum multipliers $\alpha _ { \mathrm { v } } ( e )$ and $\alpha _ { \theta } ( e )$ increase linearly over 8 epochs, starting at epochs 4 and 12, respectively. To prevent overfitting, training relies on random spatiotemporal crops of $1 6 \times 6 4 \times 6 4$ spatial voxels across 6 cardiac frames.

## 3 Experiments

## 3.1 Data

We utilize the multi-center, multi-vendor CMRx4DFlow challenge dataset [4], partitioning the 138 original training cases into custom training (96 cases), validation (21 cases), and test (21 cases) sets preserving the original center and scanner architecture distributions of the original cohort. During training, kt-Gaussian undersampling masks are generated randomly on the fly to match a randomly selected target acceleration factor. For validation and testing, we evaluate the models using undersampling masks pre-generated according to the same protocol at acceleration factors ranging from 10 to 50 .

## 3.2 Main Experiments

We evaluate FlowMoDL against classical CG-SENSE [11], the standard $M o D L$ architecture [1], FlowVN [15] and FlowMRI-Net [6]. To ensure a fair comparison, all learned models were trained for an equivalent number of gradient steps. This corresponds to 50 epochs using standard random spatiotemporal cropping, avoiding the highly correlated, sliding-window epoch definitions originally proposed for FlowVN and FlowMRI-Net. Performance is measured via magnitude structural similarity (SSIM) [16], magnitude nRMSE, velocity relative error (Rel-Err) in the aorta, and velocity angular error (AngErr) [15].

Table 1 and Figure 2 summarize the quantitative results on the unseen test set, averaged across five random seeds. FlowMoDL strictly outperforms all baselines, including MoDL, across all acceleration factors $( R \in \{ 1 0 , \ldots , 5 0 \} )$ . While MoDL provides a robust baseline for basic structural reconstruction, FlowMoDL’s explicit phase-aware objective and spatiotemporal formulation yield substantially lower angular and relative errors, while still outperforming MoDL on nRMSE and SSIM. Notably, FlowVN and FlowMRI-Net degrade significantly under this normalized gradient-step budget, underperforming classical CG-SENSE. Qualitative improvements are evident in Figure 3. At a high acceleration rate $( R = 5 0 )$ , both CG-SENSE and MoDL struggle with high-frequency spatial details and sufer from velocity artifacts. In contrast, FlowMoDL’s learned prior recovers sharp spatial structures and smooth velocity transitions. Beyond spatial fidelity, FlowMoDL preserves temporal coherence. Since blood flow v(t) is naturally continuous while residual undersampling noise is temporally uncorrelated, we quantify temporal coherence using the lag-one autocorrelation $\rho _ { 1 }$ and the fraction of temporal power above half the Nyquist frequency η [3]. At R = 50, CG-SENSE $\left( \rho _ { 1 } \approx 0 . 0 0 , \eta \approx 0 . 5 3 \right)$ and MoDL $( \rho _ { 1 } \approx - 0 . 0 5 , \eta \approx 0 . 5 5 )$ yield metrics indicative of temporally white noise artifacts. Conversely, FlowMoDL $\left( \rho _ { 1 } \approx 0 . 7 2 , \eta \approx 0 . 0 7 \right)$ closely matches the ground truth $\left( \rho _ { 1 } \approx 0 . 7 1 , \eta \approx 0 . 0 9 \right)$ , successfully reconstructing temporally coherent velocity fields with only marginal over-smoothing.

![](images/ce8df3c66bac45e45fc6bfd0e8c5b8f8b0a3b9c9ed7bc4084169e9ea2dbc59e7.jpg)  
<sup>(a)</sup> <sup>Angular</sup> <sup>Error</sup> <sup>[°]</sup> <sup>(</sup>↓<sup>)</sup>

![](images/a7ef3f39659837eadb0ea6f6e19ea3ee221c8de08e9b5e767d63915c5b169e87.jpg)  
<sup>(b)</sup> <sup>nRMSE</sup> <sup>(</sup>↓<sup>)</sup>

![](images/d737f7729a0869b7e4cc264b29d524b454ac2e04714e5b9a1792bbdc93eb9432.jpg)  
<sup>(c)</sup> <sup>SSIM</sup> <sup>(</sup>↑<sup>)</sup>

![](images/918c3a401229ded1bbf149ec85596e30de9ec56b4c2682f5126e5ad1e8bf643c.jpg)  
<sup>(d)</sup> <sup>Relative</sup> <sup>Error</sup> <sup>(</sup>↓<sup>)</sup>  
Fig. 2: Metrics across diferent R’s. For deep learning methods, solid lines denote the mean and shaded areas represent standard deviation (n = 5 seeds).

## 4 Conclusion

In this work, we presented FlowMoDL, an eficient model-based unrolled network tailored for highly accelerated 4D flow MRI reconstruction. By integrating a (3+1)D spatiotemporal learned prior with conjugate-gradient data-consistency and dynamic conditioning for the acceleration factor, FlowMoDL efectively recovers both anatomical structures and complex flow dynamics using a single trained model. Crucially, optimizing directly for phase alignment via a specialized curriculum-based composite loss, incorporating velocity, relative, and angular error terms, prevents the magnitude-bias common in generic MRI reconstruction tasks. Evaluated on the multi-center CMRx4DFlow dataset, FlowMoDL strictly outperforms classical CG-SENSE, standard MoDL, and specialized deep learning architectures (FlowVN and FlowMRI-Net) across acceleration factors from 10 to 50 . Furthermore, we demonstrated that FlowMoDL exhibits superior training eficiency compared to existing flow-specific networks under a normalized gradient-step budget, consistently yielding sharp magnitude reconstructions and physiologically accurate, temporally coherent velocity fields even at extreme undersampling rates.

![](images/dcb94ded96940d9bcd3cc7a2c50a652d4eecc63898ffaf0c585fee33c31bb813.jpg)  
Fig. 3: Representative qualitative results at R = 50 comparing CG-SENSE, MoDL, and the proposed FlowMoDL with the fully sampled reference. Flow-MoDL reconstructs the volume with the lowest magnitude and velocity errors.

Acknowledgments. This work was supported by DFG Heisenberg (502024488), ERC StG EARTHWORM (101165242), ERC Proof-of-concept grant SYNCWORM (101293293) and CAIMed - Lower Saxony Center for Artificial Intelligence and Causal Methods in Medicine (ZN4257).

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Aggarwal, H.K., Mani, M.P., Jacob, M.: Modl: Model-based deep learning architecture for inverse problems. IEEE transactions on medical imaging 38(2), 394–405 (2018)

2. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: Proceedings of the 26th annual international conference on machine learning. pp. 41–48 (2009)

3. Box, G.E., Jenkins, G.M., Reinsel, G.C., Ljung, G.M.: Time series analysis: forecasting and control. John Wiley & Sons (2015)

4. CMRx4DFlow2026 Team: CMRx4DFlow2026: Data. https://cmrxrecon.github.io/2026/data.html (2026), accessed: 27 July 2026

5. Heckel, R., Jacob, M., Chaudhari, A., Perlman, O., Shimron, E.: Deep learning for accelerated and robust mri reconstruction. Magnetic Resonance Materials in Physics, Biology and Medicine 37(3), 335–368 (2024)

6. Jacobs, L., Piccirelli, M., Vishnevskiy, V., Kozerke, S.: Flowmri-net: A generalizable self-supervised 4d flow mri reconstruction network (2025), https://arxiv.org/abs/2410.08856

7. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

8. Markl, M., Frydrychowicz, A., Kozerke, S., Hope, M., Wieben, O.: 4d flow mri. Journal of Magnetic Resonance Imaging 36(5), 1015–1036 (2012)

9. Munoz, C., Fotaki, A., Botnar, R.M., Prieto, C.: Latest advances in image acceleration: All dimensions are fair game. Journal of Magnetic Resonance Imaging 57(2), 387–402 (2023). https://doi.org/10.1002/jmri.28462

10. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI conference on artificial intelligence. vol. 32 (2018)

11. Pruessmann, K.P., Weiger, M., Börnert, P., Boesiger, P.: Advances in sensitivity encoding with arbitrary k-space trajectories. Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine 46(4), 638–651 (2001)

12. Pruessmann, K.P., Weiger, M., Scheidegger, M.B., Boesiger, P.: Sense: sensitivity encoding for fast mri. Magnetic Resonance in Medicine: An Oficial Journal of the International Society for Magnetic Resonance in Medicine 42(5), 952–962 (1999)

13. Stankovic, Z., Allen, B.D., Garcia, J., Jarvis, K.B., Markl, M.: 4d flow imaging with mri. Cardiovascular diagnosis and therapy 4(2), 173 (2014)

14. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., Paluri, M.: A closer look at spatiotemporal convolutions for action recognition. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition. pp. 6450–6459 (2018)

15. Vishnevskiy, V., Walheim, J., Kozerke, S.: Deep variational network for rapid 4d flow mri reconstruction. Nature Machine Intelligence 2(4), 228–235 (Apr 2020). https://doi.org/10.1038/s42256-020-0165-6, http://dx.doi.org/10.1038/s42256- 020-0165-6

16. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861