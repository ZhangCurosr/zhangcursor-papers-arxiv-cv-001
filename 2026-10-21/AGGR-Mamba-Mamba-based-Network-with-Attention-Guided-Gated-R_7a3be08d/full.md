# AGGR-Mamba: Mamba-based Network with Attention-Guided Gated Residual Refinement for Undersampled MRI Reconstruction

Yue Zhuo<sup>1</sup>, Wenbin Deng<sup>1</sup>, Lingjun Liu<sup>2</sup>, Xuhang Chen<sup>2</sup>, Zhonghua Xie<sup>2,\*</sup>

<sup>1</sup>School of Mathematics and Statistics, Huizhou University, Huizhou, China

<sup>2</sup>School of Computer Science and Engineering, Huizhou University, Huizhou, China

Abstract—Undersampled magnetic resonance imaging (MRI) reconstruction aims to recover high-quality images from partially acquired k-space data, thereby reducing scan time. However, missing frequency-domain measurements often lead to residual aliasing artifacts, blurred edges, and loss of local structural details. In this paper, we propose AGGR-Mamba, a lightweight output-stage residual refinement method built upon a Mamba backbone for undersampled brain MRI reconstruction. The Mamba backbone first produces an initial reconstruction, and the proposed refinement modules then correct the remaining local errors and high-frequency artifacts in a controlled residual manner.Specifically, an Attention Refine Head is introduced to recalibrate local anatomical structures, edge textures, and residual-error regions through channel and spatial attention. In addition, a GDFN-style gated residual refinement branch is designed to selectively preserve informative residual features while suppressing redundant or unstable responses. By combining attention-based recalibration with gated residual selection, AGGR-Mamba improves local structure recovery without substantially disturbing the backbone reconstruction. Experiments are conducted on the IXI brain MRI dataset with T1- and T2- weighted images under acceleration factors of R = 4 and R = 8. The results show that AGGR-Mamba achieves competitive PSNR and SSIM performance across different reconstruction settings, demonstrating the effectiveness of the proposed output-stage refinement strategy.

Index Terms—Magnetic resonance image reconstruction; undersampling; Mamba; IXI dataset

## I. INTRODUCTION

Magnetic resonance imaging (MRI) is an important medical imaging technique that provides high soft-tissue contrast, rich structural information, and no exposure to ionizing radiation. Owing to these advantages, it has been widely used in the analysis of brain disorders, clinical diagnosis, and medical research [1]. However, MRI data acquisition is typically timeconsuming. Long acquisition times can compromise clinical workflow efficiency, increase patient discomfort, and introduce motion-related artifacts caused by patient movement, which may adversely affect subsequent diagnosis and image analysis [2]. To improve MRI acquisition efficiency, a commonly used strategy is to undersample the data in k-space. By acquiring fewer k-space measurements, undersampling can reduce scan time, but the resulting lack of complete frequencydomain information makes reconstructed images susceptible to aliasing artifacts, loss of textural details, and blurred edges [3].

![](images/fac9992b0734f07c5f665238b43c858863ecbc36fcd1d3ef019522eccf1e28ab.jpg)  
Fig. 1. Illustration of the undersampled MRI reconstruction task: undersampled input (left) and AGGR-Mamba reconstruction (right) on IXI-T2 with R = 4. The two rows highlight different regions of the same slice; blue boxes and lines indicate the corresponding enlarged regions.

Figure 1 illustrates the undersampled MRI reconstruction task using an undersampled input and the corresponding AGGR-Mamba output, with two local regions enlarged for comparison.

Although conventional compressed sensing MRI methods are supported by a solid theoretical foundation [4], [5], they typically rely on manually designed regularization terms and require the tuning of multiple hyperparameters [6], while also suffering from relatively low computational efficiency [7]. Driven by advances in deep learning [8], [9], CNN-based approaches have been widely explored for learning the reconstruction mapping between undersampled and fully sampled MR images [10]. CNN-based methods are well suited for local feature extraction, enabling effective recovery of edges and textural details. However, their reliance on local convolutional operations limits their capacity to capture long-range dependencies and global anatomical structures. GAN-based approaches were later explored to enhance the perceptual quality of reconstructed images [11], [12], but the adversarial learning process may introduce training instability and produce anatomically implausible structures.

To better capture global contextual dependencies, Transformer-based approaches have been explored for MRI reconstruction [13]. These methods exploit self-attention to capture long-range dependencies, thereby partially compensating for the inherent limitation of CNNs in global context modeling. However, the self-attention mechanism in Transformers incurs high computational complexity, often resulting in substantial memory footprint and computational burden in high-resolution medical image reconstruction [14]. In recent years, state space models (SSMs), exemplified by Mamba [15], have introduced a new paradigm for long-sequence modeling. Compared with conventional Transformers, Mamba is capable of modeling long-range dependencies at significantly lower complexity and has been progressively integrated into medical image reconstruction tasks.

Despite the strengths of Mamba-type reconstruction frameworks in global structural modeling and computational efficiency [16], [17], the restoration of local details and edgestructure preservation at the output stage in undersampled MRI reconstruction can still be further improved. On the one hand, the reconstructed images may still retain local residual artifacts and high-frequency errors. On the other hand, fine anatomical edges, texture transition regions, and low-contrast structures are susceptible to blurring or insufficient detail during reconstruction. Moreover, a conventional output Refine Head typically applies a relatively uniform correction across the entire image and struggles to distinguish valid structural details from residual errors and spurious high-frequency noise, which may compromise the stability of the refinement process.

To address these limitations, this paper proposes AGGR-Mamba, an output-stage attention-guided and gated residual refinement method for undersampled MRI reconstruction. The proposed method uses a Mamba backbone to obtain an initial reconstruction and attaches a lightweight refinement branch at the output stage. This branch is designed to compensate for local reconstruction errors, edge blurring, and residual highfrequency artifacts that may remain after the backbone reconstruction. The refinement process combines attention-guided local detail recalibration with gated residual feature selection, allowing the network to enhance informative anatomical structures while suppressing unstable residual responses.

The major contributions of this study are as follows:

• A lightweight output-stage residual refinement design. We introduce a lightweight residual refinement branch after the Mamba backbone. Instead of modifying the main reconstruction process, the proposed design refines the backbone output by compensating for local errors and residual artifacts, leading to improved local structural representation in the reconstructed image.

• An attention-guided local detail recalibration mechanism. The proposed Attention Refine Head uses channel and spatial attention to adaptively recalibrate channel and spatial responses associated with local structures and residual-error regions. This design reduces uniform correction over the entire image and makes the refinement process more focused on structurally informative areas.

• A gated residual feature selection strategy. Since residual features may contain useful high-frequency details as well as noise and unstable responses, we introduce a GDFN-style gated branch to selectively filter candidate residual features. This mechanism enhances reconstruction-relevant details while suppressing ineffective high-frequency noise and residual artifacts.

## II. RELATED WORK

## A. CNN-based MRI Reconstruction

CNN-based MRI reconstruction methods generally learn a nonlinear mapping from undersampled images to their fully sampled counterparts. Representative architectures include U-Net, deep residual networks, and cascaded convolutional networks [18], [19]. These methods are effective at extracting local features and can recover edges, textures, and fine structural details. However, because convolutional operations mainly rely on local receptive fields, CNN-based models may struggle to restore subtle details and delineate structural boundaries clearly, especially in complex brain anatomy and highly accelerated undersampling scenarios.

## B. Physics-guided MRI Reconstruction

Physics-guided MRI reconstruction methods embed the imaging model of MRI into deep reconstruction networks [20]. These methods commonly rely on a data consistency module to keep the reconstructed result consistent with the sampled k-space data. By combining physical constraints with the representational capacity of deep networks, these methods tend to provide more stable performance in undersampled MRI reconstruction. Models such as E2E-VarNet [21] fall into this category. The Mamba backbone used in this study follows the same principle of physics-guided reconstruction, where the backbone reconstruction module and data consistency constraint jointly contribute to improved reconstruction quality.

## C. Transformer-based MRI Reconstruction

Transformer-based methods have also been explored for MRI reconstruction, mainly because self-attention can model long-range dependencies beyond the local receptive field of CNNs. For example, ReconFormer [22] adopts a recurrent Transformer design with multi-scale attention to improve reconstruction quality. Despite these advantages, the selfattention operation is often costly, and its use in highresolution image reconstruction can result in heavy GPU memory usage and increased computational burden [23].

## D. Mamba-based MRI Reconstruction

Mamba-based methods model long-range dependencies through state space models. Compared with Transformer-based models, their linear-complexity sequence modeling is more efficient for long inputs, which is beneficial for high-resolution medical image reconstruction. In MRI reconstruction, Mambabased designs are commonly combined with data consistency constraints so that the network can exploit both learned feature representations and the acquired k-space measurements [17].

Although these methods have shown strong potential in global context modeling and physics-guided reconstruction, most existing designs mainly focus on the main reconstruction stages. The output stage is less explicitly explored, especially for local error compensation, edge-detail preservation, and residual high-frequency artifact suppression. Motivated by this observation, AGGR-Mamba keeps the Mamba backbone as the main reconstruction component and introduces lightweight output-stage refinement modules to improve local structure recovery and refinement stability.

## III. METHODOLOGY

## A. Mamba Backbone

AGGR-Mamba adopts a Mamba-based reconstruction network as the backbone [17]. The backbone is organized as a six-stage unrolled reconstruction network. Each reconstruction stage consists of one VSSLayer followed by one Data Consistency (DC) module, where each VSSLayer contains two VSSBlocks. The VSSBlocks are used to model long-range dependencies and global contextual information, while the DC modules enforce consistency between the intermediate reconstruction and the acquired k-space measurements. Therefore, the backbone contains 12 VSSBlocks and six stage-wise DC modules in total. After the six stages, an additional final DC module is applied for data consistency enforcement. Based on the backbone output, AGGR-Mamba further applies the Attention Refine Head and the GDFN-style gated residual refinement branch to compensate for remaining local errors at the output stage.

For simplicity, each “VSSM Block” illustrated in Fig. 2 represents a VSSLayer containing two VSSBlocks.

Let x denote the complex-valued MRI image to be reconstructed, and let M, C, and n denote the sampling mask, coil sensitivity map, and noise term, respectively. The undersampled MRI acquisition model is written as

$$
\mathbf { y } = \mathbf { M } \odot \mathcal { F } ( \mathbf { C } \mathbf { x } ) + \mathbf { n } ,\tag{1}
$$

where y denotes the acquired undersampled k-space data, $\mathcal { F }$ denotes the two-dimensional Fourier transform, and $\odot$ denotes element-wise multiplication. The image-domain input reconstructed from the undersampled data is denoted by $\mathbf { x } _ { u s } .$ In the single-coil IXI setting used in this work, the coil sensitivity map is simplified as an identity map, i.e., C = 1.

In the network implementation, each complex MRI image is represented by two channels corresponding to the real and imaginary parts:

$$
\mathbf { x } = [ \mathbf { x } ^ { ( r ) } , \mathbf { x } ^ { ( i ) } ] \in \mathbb { R } ^ { 2 \times H \times W } .\tag{2}
$$

At the i-th reconstruction stage, $i = 1 , \ldots , 6 ,$ , the input is first processed by the corresponding VSSLayer:

$$
\begin{array} { r } { { \bf z } _ { i } = \mathcal { V } _ { i } ( { \bf x } _ { i - 1 } ) , } \end{array}\tag{3}
$$

and then passed through the DC block:

$$
\begin{array} { r } { { \bf x } _ { i } = \mathcal { D } _ { i } ( { \bf z } _ { i } , { \bf x } _ { u s } , { \bf M } , { \bf C } ) . } \end{array}\tag{4}
$$

Here, $\mathcal { V } _ { i } ( \cdot )$ denotes the i-th VSSLayer and $\mathcal { D } _ { i } ( \cdot )$ denotes the corresponding DC module. The VSSBlocks within each VSSLayer use multi-directional selective scanning to model spatial interactions among two-dimensional image patches.

For data consistency enforcement, the current prediction is transformed into k-space, and the sampled positions are constrained by the acquired k-space measurements:

$$
\mathbf { k } _ { p r e d } = \mathcal { F } ( \mathbf { C } \mathbf { x } _ { p r e d } ) , \quad \mathbf { k } _ { u s } = \mathcal { F } ( \mathbf { C } \mathbf { x } _ { u s } ) ,\tag{5}
$$

$$
{ \bf k } _ { d c } = ( 1 - { \bf M } ) \odot { \bf k } _ { p r e d } + { \bf M } \odot { \bf k } _ { u s } ,\tag{6}
$$

$$
\begin{array} { r } { { \bf x } _ { d c } = { \bf C } ^ { H } \mathcal { F } ^ { - 1 } ( { \bf k } _ { d c } ) . } \end{array}\tag{7}
$$

Here, $\mathbf { x } _ { p r e d }$ denotes the current complex-valued prediction, ${ \bf k } _ { p r e d }$ denotes its corresponding k-space representation, $\mathbf { k } _ { u s }$ denotes the k-space representation associated with the undersampled input, $\mathbf { C } ^ { H }$ denotes coil combination using the complex-conjugate sensitivity maps, and $\mathbf { x } _ { d c }$ denotes the result after data consistency enforcement. In the current IXI experiments, $\mathbf { C } = \mathbf { 1 }$ , so this expression reduces to the single-coil identity-map case implemented in the data loader.

After the unrolled VSSM-DC stages, the Mamba backbone outputs an initial reconstruction:

$$
{ \bf x } _ { m } = B _ { \theta _ { B } } ( { \bf x } _ { u s } , { \bf M } , { \bf C } ) ,\tag{8}
$$

where $B _ { \theta _ { B } }$ denotes the Mamba backbone and $\theta _ { B }$ denotes its parameters. The output $\mathbf { x } _ { m } \in \mathbb { R } ^ { 2 \times H \times W }$ contains the real and imaginary components of the reconstructed image. AGGR-Mamba then obtains the final reconstruction by applying the Attention Refine Head and the GDFN-style branch:

$$
\hat { \mathbf { x } } = G _ { \psi } \left( A _ { \phi } \left( B _ { \theta _ { B } } ( \mathbf { x } _ { u s } , \mathbf { M } , \mathbf { C } ) \right) \right) ,\tag{9}
$$

where $A _ { \phi }$ denotes the Attention Refine Head and $G _ { \psi }$ denotes the GDFN-style gated residual refinement branch.

## B. Attention Refine Head

The Mamba backbone can produce a relatively stable initial reconstruction, but residual errors may still remain around local edges, texture-transition regions, and low-contrast structures. To further strengthen local structural representation at the output stage, we attach an Attention Refine Head after the backbone output to perform lightweight feature recalibration on the two-channel reconstruction.

More concretely, the backbone output $\mathbf { x } _ { m }$ is first mapped into an intermediate feature space using a $3 \times 3$ convolution:

$$
\mathbf { F } _ { 0 } = P _ { i n } ( \mathbf { x } _ { m } ) ,\tag{10}
$$

where $P _ { i n }$ denotes the input projection layer. Several residual convolutional blocks are then used to extract local contextual features:

$$
{ \bf F } _ { r } = R _ { a } ( { \bf F } _ { 0 } ) ,\tag{11}
$$

where $R _ { a } ( \cdot )$ denotes the residual convolutional feature extraction operation.

![](images/16b496d405cbb57f2e111fc9e5a29abd0e189e72098fa50321a1e58b639714cf.jpg)  
Fig. 2. Schematic illustration of the proposed AGGR-Mamba method.

To make the output-stage refinement focus more on structurally informative regions, the Attention Refine Head introduces channel attention and spatial attention at its end. Channel attention generates channel-wise weights using global average pooling and max pooling, while spatial attention produces spatial weights from the average and maximum responses along the channel dimension. This process can be written as

$$
\mathbf { F } _ { a } = \mathrm { S A } \left( \mathrm { C A } \left( \mathbf { F } _ { r } \right) \right) ,\tag{ices,<sup>(12)</sup>}
$$

where $\mathrm { C A } ( \cdot )$ and SA(·) denote channel attention and spatial attention, respectively. Finally, the Attention Refine Head predicts a two-channel residual term, which is added to the backbone output with a learnable scaling factor $s _ { a } \colon$

$$
\mathbf { r } _ { a } = P _ { o u t } ( \mathbf { F } _ { a } ) ,\tag{date(13}
$$

$$
\mathbf { x } _ { a } = \mathbf { x } _ { m } + s _ { a } \mathbf { r } _ { a } .\tag{ghts<sup>(14)</sup>}
$$

re sav<sub>where</sub> $P _ { o u t }$ ording to validation PSNR. The fi<sub>denotes the output projection and</sub> $\mathbf { r } _ { a }$ testing<sub>is the</sub> predicted two-channel residual map. In the implementation, est model weights. The test set is not involved in model se-<sub>the final output convolution is initialized with zeros, so that</sub> ection. The model output is represented by two real/imaginary<sub>the module behaves approximately as an identity mapping at</sub> the beginning of optimization. This reduces the initial distur-SNR and SSIM are calculated. The SSIM(%) reported in<sub>bance introduced by output-stage refinement to the backbone</sub> he tables is de<sub>reconstruction.</sub>

## . Implementation DetailsC. GDFN-style Gated Residual Refinement Branch

The model used in the main experiments is AGGR-Mamba.The Attention Refine Head provides an initial recalibration Built on the Mamba backbone, AGGR-Mamba further incor-of the output features in both channel and spatial dimensions. orates an Attention Refine Head and a GDFN-style gatedOn this basis, a GDFN-style gated residual refinement branch esidual refinement branch at the output stage. In the finalis introduced to further screen the candidate residual inforptimization stage, the trained backbone is frozen and only themation. Through gated feature selection, the branch preserves Attention Refine Head and GDFN-style branch are updated.useful detail residuals while filtering redundant or unstable hese modules refine the backbone reconstruction throughcandidate residual responses, thereby reducing their interferocal feature recalibration and gateence with the final reconstruction.

More specifically, the input $\mathbf { x } _ { a }$ is first projected by a convolution layer to obtain intermediate features. A 1×1 convolution is then used to expand the channel dimension, followed by a depth-wise convolution for local context encoding:

$$
\mathbf { F } = \operatorname { D W C o n v } \left( P _ { e } \left( \operatorname { G E L U } ( P _ { g } ( \mathbf { x } _ { a } ) ) \right) \right) ,\tag{15}
$$

The<sup>where</sup> $P _ { g }$ omparison methods includ<sup>denotes</sup> <sup>the</sup> <sup>input</sup> <sup>projection,</sup> $P _ { e }$ UNet [10], E2E<sup>denotes</sup> <sup>the</sup> <sup>feature</sup> VarNet [21], SwinUNET [27], SwinMR [28], and Recon<sup>expansion</sup> <sup>operation,</sup> <sup>and</sup> <sup>DWConv</sup> <sup>denotes</sup> <sup>the</sup> <sup>depth-wise</sup> convolution. The expanded features are then split into two branches along the channel dimension:

$$
[ \mathbf { F } _ { 1 } , \mathbf { F } _ { 2 } ] = \mathrm { S p l i t } ( \mathbf { F } ) .\tag{16}
$$

methods, and ReconFormer represents recurrent TransformerThe gated features are obtained by element-wise multiplicabasetion:

$$
\mathbf { F } _ { g } = \operatorname { G E L U } ( \mathbf { F } _ { 1 } ) \odot \mathbf { F } _ { 2 } .\tag{these(17}
$$

After that, $\mathbf { F } _ { g }$ is mapped back to a two-channel residual through the feature projection and output projection layers:

$$
{ \bf r } _ { g } = P _ { \mathrm { o u t } } ^ { g } \left( P _ { \mathrm { p r o j } } ( { \bf F } _ { g } ) \right) .\tag{18}
$$

The scaled residual is then added to $\mathbf { x } _ { a }$ using the learnable residual coefficient α:

$$
\hat { \mathbf { x } } = \mathbf { x } _ { a } + \bar { \alpha } \mathbf { r } _ { g } , \quad \bar { \alpha } = \mathrm { c l i p } ( \alpha , 0 , 0 . 1 2 ) ,\tag{r the<sup>(19)</sup>}
$$

<sup>R</sup> <sup>=</sup> where $P _ { p r o j }$ = 8 undersampling settings. Overall, A<sub>denotes the gated-feature projection and</sub> $P _ { o u t } ^ { g }$ Mamba achieves competitive results across all four testing<sub>denotes the final two-channel output projection. The gating</sub> settings,<sub>operation</sub> $\mathrm { G E L U } ( \mathbf { F } _ { 1 } ) { \odot } \mathbf { F } _ { 2 }$ stable performance on the T2-<sub>filters</sub> <sub>candidate</sub> <sub>residual</sub> <sub>responses</sub> weighted reconstruction tasks. For IXI-T2 R = 4, AGGR<sub>before they are added to the reconstruction. Its contribution is</sub> Mamba reaches 46.99 dB in PSNR, hig<sub>assessed</sub> <sub>through</sub> <sub>the</sub> <sub>ablation</sub> <sub>experiments.</sub>

## D. Training Strategy and Objective Function

ghtly below ReconFormer’s 99.53. For IXI-T2 R = 8,The final AGGR-Mamba optimization adopts a frozen-AGGR-Mamba again reports the best PSNR, with 38.48 dBbackbone output-stage fine-tuning strategy. A trained Mamba versus 38.34 dB for ReconFormer. This result suggests that,backbone checkpoint and the existing Attention Refine Head for the T2 modality, output-stage attention recalibration andweights are first loaded. During final fine-tuning, the backbone gated residparameters $\theta _ { B }$ ompensation can efectively refine the initiaare fixed, while the Attention Refine Head reconstructparameters $\phi$ produced by the Mamba backbone.and GDFN-style branch parameters ψ remain trainable. The forward path is unchanged: the undersampled input $\mathbf { x } _ { u s } ,$ sampling mask M, and coil sensitivity map C are passed through the backbone, Attention Refine Head, and GDFN-style branch in sequence to produce xˆ.

During training, the two real/imaginary channels of the model output xˆ are first converted into a magnitude image:

$$
\mathcal { M } ( \hat { \mathbf { x } } ) = \sqrt { \left( \hat { \mathbf { x } } ^ { ( r ) } \right) ^ { 2 } + \left( \hat { \mathbf { x } } ^ { ( i ) } \right) ^ { 2 } } .\tag{20}
$$

The training loss is defined as the mean $L _ { 1 }$ loss computed on the magnitude images:

$$
\mathcal { L } = \frac { 1 } { \vert \boldsymbol { B } \vert H W } \sum _ { j = 1 } ^ { \vert \mathcal { B } \vert } \Vert \mathcal { M } ( \hat { \mathbf { x } } _ { j } ) - \mathbf { x } _ { g t , j } \Vert _ { 1 } ,\tag{21}
$$

where B represents a training batch, with |B| giving the corresponding batch size. $\mathbf { x } _ { g t , j }$ represents the fully sampled target magnitude image corresponding to the j-th sample. At the testing stage, the real/imaginary output channels are likewise converted into a magnitude image before calculating PSNR [24] and SSIM [25]. In the reported tables, SSIM(%) is defined as

$$
\mathrm { S S I M } ( \% ) = 1 0 0 \times \mathrm { S S I M } .\tag{22}
$$

The trainable parameter set in the final stage is

$$
\Omega = \{ \phi , \psi \} ,\tag{23}
$$

where $\phi$ includes the convolutional, channel-spatial attention, and residual-scaling parameters of the Attention Refine Head, and ψ includes the projection, depth-wise convolution, gating, and residual-scaling parameters of the GDFN-style branch. The frozen backbone satisfies

$$
\nabla _ { \theta _ { B } } \mathcal { L } = 0 , \qquad \theta _ { B } ^ { t + 1 } = \theta _ { B } ^ { t } .\tag{24}
$$

Accordingly, the optimization objective is

$$
\begin{array} { r } { \left( \phi ^ { * } , \psi ^ { * } \right) = \arg \operatorname* { m i n } _ { \phi , \psi } \mathcal { L } \left( G _ { \psi } \left( A _ { \phi } \left( B _ { \theta _ { B } } ( \mathbf { x } _ { u s } , \mathbf { M } , \mathbf { C } ) \right) \right) , \mathbf { x } _ { g t } \right) , } \\ { \theta _ { B } \ \mathrm { f i x e d } . \ } \end{array}\tag{25}
$$

The output-stage parameters are optimized using AdamW with a base learning rate of $1 \times 1 0 ^ { - 3 }$ and a minimum learning rate of $1 \times 1 0 ^ { - 6 }$ . The batch size is 4, the maximum number of iterations is 5000, and the warmup period is 100 iterations. Validation is performed every 500 iterations, and regular checkpoints are saved every 1000 iterations. For each IXI setting, the best checkpoint is selected according to validation PSNR, while the test set is used only for final performance reporting.

Algorithm 1 summarizes the final-stage optimization and evaluation procedure for each IXI setting.

## IV. EXPERIMENTAL SETUP AND RESULTS ANALYSIS

## A. Dataset

The effectiveness of the proposed AGGR-Mamba is evaluated on the IXI brain MRI dataset. The IXI dataset was collected as part of the IXI–Information eXtraction from Images project [26], and is publicly available at: https:// brain-development.org/ixi-dataset/. The dataset contains multiple MRI sequences from healthy subjects, including T1- weighted, T2-weighted, and PD-weighted images. This study focuses on single-modality undersampled brain MRI reconstruction and compares the proposed method with Recon-Former and other representative baselines. Experiments are conducted on both T1-weighted and T2-weighted modalities, with training and testing performed under the R = 4 and R = 8 undersampling settings.

Algorithm 1 AGGR-Mamba frozen-backbone output-stage   
fine-tuning   
Input: Training, validation, and test sets for one IXI setting; trained   
backbone and existing Attention Refine Head weights   
Output: Best-validation checkpoint and final test PSNR/SSIM   
1: Load the backbone and Attention Refine Head weights   
2: Fix $\theta _ { B } ;$ set trainable parameters $\boldsymbol { \Omega } = \{ \phi , \psi \}$   
3: Configure AdamW with base/minimum learning rates   
$1 0 ^ { - 3 } / 1 0 ^ { - 6 }$ and 100 warmup iterations   
4: $P _ { \mathrm { b e s t } }  - \infty$   
5: for $t = 1 , \dots$ , 5000 do   
6: Sample training batch B of size 4   
7: for each $( \mathbf { x } _ { u s , j } , \mathbf { M } _ { j } , \mathbf { C } _ { j } , \mathbf { x } _ { g t , j } )$ in B do   
8: $\mathbf { x } _ { m , j }  B _ { \theta _ { B } } ( \mathbf { x } _ { u s , j } , \mathbf { M } _ { j } , \mathbf { C } _ { j } )$   
9: $\mathbf { x } _ { a , j } \gets A _ { \phi } ( \mathbf { x } _ { m , j } )$   
10: $\hat { \mathbf { x } } _ { j }  G _ { \psi } \underline { { ( \dot { \mathbf { x } } _ { a , j } ) } }$   
11: $\hat { \bf m } _ { j }  \sqrt { ( \hat { \bf x } _ { j } ^ { ( r ) } ) ^ { 2 } + ( \hat { \bf x } _ { j } ^ { ( i ) } ) ^ { 2 } }$   
12: end for   
13: $\begin{array} { r } { \mathcal { L } \gets \frac { 1 } { | \mathcal { B } | H W } \sum _ { j = 1 } ^ { | \mathcal { B } | } \| \hat { \mathbf { m } } _ { j } - \mathbf { x } _ { g t , j } \| _ { 1 } } \end{array}$   
14: Update only ϕ, ψ using AdamW; keep $\theta _ { B }$ fixed   
15: if t mod 500 = 0 then   
16: Evaluate magnitude-domain validation PSNR $P _ { \mathrm { v a l } }$   
17: if $P _ { \mathrm { v a l } } > P _ { \mathrm { b e s t } } ^ { \overline { { } } }$ then   
18: Save the best checkpoint; $P _ { \mathrm { b e s t } }  P _ { \mathrm { v a l } }$   
19: end if   
20: end if   
21: if t mod 1000 = 0 then   
22: Save a regular checkpoint   
23: end if   
24: end for   
25: Reload the best-validation checkpoint   
26: Evaluate the test set using magnitude reconstructions   
27: Report test PSNR and 100 × SSIM

All experiments in this work are conducted on a server equipped with a single NVIDIA Tesla V100-PCIE-32GB GPU for training and testing. The model is implemented with PyTorch 2.0.1+cu118 and Python 3.8.20. The experiments use the IXI single-coil brain MRI data that have been preprocessed into H5 format. In the processed H5 data used in this work, the T1 setting contains 2950 training slices, 600 validation slices, and 1200 testing slices, while the T2 setting contains 2443 training slices, 512 validation slices, and 1040 testing slices. The four experimental settings, namely IXI-T1 R = 4, IXI-T1 R = 8, IXI-T2 R = 4, and IXI-T2 R = 8, are trained and tested independently.

During training, only the training set is used to update the output-stage parameters. The model is evaluated on the validation set every 500 iterations, and the best model weights are saved according to validation PSNR. The final testing metrics are computed on the test set using the corresponding best model weights. The test set is not involved in model selection. The model output is represented by two real/imaginary channels and is converted into a magnitude image before PSNR and SSIM are calculated. The SSIM(%) reported in the tables is defined as SSIM × 100.

## B. Implementation Details

The model used in the main experiments is AGGR-Mamba. Built on the Mamba backbone, AGGR-Mamba further incorporates an Attention Refine Head and a GDFN-style gated residual refinement branch at the output stage. In the final optimization stage, the trained backbone is frozen and only the Attention Refine Head and GDFN-style branch are updated. These modules refine the backbone reconstruction through local feature recalibration and gated residual compensation.

The comparison methods include UNet [10], E2E-VarNet [21], SwinUNET [27], SwinMR [28], and Recon-Former [22]. Among them, UNet serves as a representative CNN-based reconstruction baseline, E2E-VarNet represents physics-guided unrolled reconstruction methods, Swin-UNET and SwinMR are included as Transformer-based reconstruction methods, and ReconFormer represents recurrent Transformer-based MRI reconstruction. By covering CNNs, unrolled networks, Transformers, and recurrent Transformer models, these comparisons provide a more comprehensive evaluation of AGGR-Mamba on undersampled brain MRI reconstruction using the IXI dataset.

## C. Quantitative Analysis

Table I reports the quantitative comparison of different methods on the IXI-T1 and IXI-T2 modalities under the R = 4 and R = 8 undersampling settings. Overall, AGGR-Mamba achieves competitive results across all four testing settings, with particularly stable performance on the T2- weighted reconstruction tasks. For IXI-T2 R = 4, AGGR-Mamba reaches 46.99 dB in PSNR, higher than the 46.71 dB achieved by ReconFormer, and ranks first among all methods in terms of PSNR. Its SSIM(%) is 99.45, which is slightly below ReconFormer’s 99.53. For IXI-T2 R = 8, AGGR-Mamba again reports the best PSNR, with 38.48 dB versus 38.34 dB for ReconFormer. This result suggests that, for the T2 modality, output-stage attention recalibration and gated residual compensation can effectively refine the initial reconstruction produced by the Mamba backbone.

AGGR-Mamba also maintains a strong performance on the IXI-T1 tasks. Under the IXI-T1 R = 4 setting, it obtains a PSNR of 41.94 dB and an SSIM(%) of 99.11, which are close to ReconFormer’s 42.36 dB and 99.24. In the IXI-T1 R = 8 setting, E2E-VarNet achieves the best PSNR, while Recon-Former gives the best SSIM. AGGR-Mamba obtains a PSNR of 35.39 dB, outperforming ReconFormer at 33.87 dB and ranking second among all compared methods. These results suggest that AGGR-Mamba remains stable across different modalities and undersampling rates. Its advantage is more evident in PSNR improvement and residual error correction, although the structural similarity metric in some T1 settings still leaves room for further improvement.

From the perspective of model types, UNet and SwinUNET generally obtain lower quantitative scores, which suggests that pure CNN reconstruction and directly transferred Ushaped Transformer designs remain limited on the current IXI task. SwinMR improves over UNet and SwinUNET in most cases, but its performance is still behind ReconFormer and AGGR-Mamba. E2E-VarNet performs well on IXI-T1 R = 8, reflecting the benefit of physics-guided unrolled networks in enforcing data consistency under certain high-acceleration settings. By comparison, AGGR-Mamba first produces a stable reconstruction through the Mamba backbone and then improves PSNR with output-stage attention recalibration and gated residual compensation. Its SSIM remains second-best or close to the best result in several settings, suggesting that the proposed lightweight refinement design provides useful compensation for local reconstruction errors.

## D. Analysis of Experimental Results

Figure 3 shows the reconstructed images, zoomed-in regions, and error maps of different methods under the IXI-T2 R = 4 setting. As can be seen, the error maps of UNet, SwinUNET, and SwinMR still contain noticeable bright regions, indicating that residual errors remain around local textures and edges. E2E-VarNet and ReconFormer produce lower errors, and their reconstructed images are closer to the target image in terms of overall structure. By contrast, the error map of AGGR-Mamba is generally darker, and the brain tissue boundaries and texture patterns in the zoomed-in regions are also clearer. Its advantage is mainly reflected in reduced errormap intensity and more stable local structures. This suggests that the output-stage refinement modules further correct the initial reconstruction produced by the Mamba backbone, rather than improving the metrics by substantially changing the global image contrast.

## E. Ablation Study Analysis

To assess the effect of the output-stage refinement modules, we perform ablation experiments on IXI-T2 R = 4 and further compare the late-stage GDFN fine-tuning strategy with joint fine-tuning. All reported results are evaluated on the same fixed test set, with the best checkpoint selected based on validation PSNR.

Table II reports the module-wise ablation results under the IXI-T2 R = 4 setting. The baseline Mamba backbone obtains a PSNR of 46.4907 dB and an SSIM of 99.38%. Adding the Attention Refine Head raises the PSNR to 46.9393 dB, which shows that output-stage attention refinement helps recover local details in the initial reconstruction. When the GDFNstyle gated residual branch is further introduced, the PSNR improves to 46.9930 dB, suggesting that the gated residual selection mechanism provides additional compensation for the remaining reconstruction errors. With output-stage joint finetuning, the final AGGR-Mamba reaches 46.9945 dB in PSNR and 99.45% in SSIM. The improvement over late-stage GDFN fine-tuning is modest, but the results are more stable overall, indicating that jointly optimizing the Attention Refine Head and the GDFN-style branch benefits the final refinement stage.

![](images/fa8e3890a73d74117dd256bb04e037364ffcb1edcaa5b6c204267bbc9c9cdda2.jpg)  
Fig. 3. Reconstruction results and error maps of different methods on IXI-T2 R = 4. The zoomed-in regions and error maps are placed below each reconstructed image for visual comparison.

TABLE I  
QUANTITATIVE RESULTS OF DIFFERENT METHODS ON THE IXI DATASET. THE BEST PERFORMANCE IS MARKED IN BOLD, WHILE THE SECOND-BEST
<table><tr><td rowspan="2">Method</td><td colspan="2">IXI-T1 R = 4</td><td colspan="2">IXI-T1 R = 8</td><td colspan="2">IXI-T2 R = 4</td><td colspan="2">IXI-T2 R = 8</td></tr><tr><td>PSNR↑</td><td>SSIM(%)↑</td><td>PSNR↑</td><td>SSIM(%)↑</td><td>PSNR↑</td><td>SSIM(%)↑</td><td>PSNR↑</td><td>SSIM(%)↑</td></tr><tr><td>UNet</td><td>37.01</td><td>94.07</td><td>31.82</td><td>90.57</td><td>38.06</td><td>94.85</td><td>33.86</td><td>91.98</td></tr><tr><td>E2E-VarNet</td><td>41.94</td><td>98.56</td><td>35.72</td><td>97.21</td><td>42.96</td><td>98.69</td><td>37.23</td><td>97.34</td></tr><tr><td>SwinUNET</td><td>34.77</td><td>94.73</td><td>30.59</td><td>91.75</td><td>37.54</td><td>95.11</td><td>33.92</td><td>92.44</td></tr><tr><td>SwinMR</td><td>38.30</td><td>96.05</td><td>33.76</td><td>93.65</td><td>40.03</td><td>96.17</td><td>35.75</td><td>93.89</td></tr><tr><td>ReconFormer</td><td>42.36</td><td>99.24</td><td>33.87</td><td>97.51</td><td>46.71</td><td>99.53</td><td>38.34</td><td>98.45</td></tr><tr><td>AGGR-Mamba (Ours)</td><td>41.94</td><td>99.11</td><td>35.39</td><td>97.04</td><td>46.99</td><td>99.45</td><td>38.48</td><td>97.88</td></tr></table>

TABLE II

MODULE-WISE ABLATION STUDY ON IXI-T2 R = 4. ∆ INDICATES THE PSNR GAIN RELATIVE TO THE Mamba BACKBONE.
<table><tr><td>Variant</td><td>Attention Refine</td><td>GDFN</td><td>Joint Fine-tuning</td><td>PSNR(dB)↑</td><td>SSIM(%)↑</td><td>∆PSNR</td></tr><tr><td>Mamba Backbone</td><td>×</td><td>×</td><td>×</td><td>46.4907</td><td>99.38</td><td></td></tr><tr><td>Attention Refine</td><td>√</td><td>×</td><td>×</td><td>46.9393</td><td>99.44</td><td>+0.4486</td></tr><tr><td>Attention Refine + GDFN</td><td>√</td><td>√</td><td>×</td><td>46.9930</td><td>99.44</td><td>+0.5023</td></tr><tr><td>AGGR-Mamba</td><td>√</td><td>√</td><td>√</td><td>46.9945</td><td>99.45</td><td>+0.5038</td></tr></table>

TABLE III

COMPARISON BETWEEN LATE-STAGE GDFN FINE-TUNING AND JOINT FINE-TUNING ON THE FOUR IXI TEST SETTINGS. EACH CELL REPORTS PSNR(dB) / SSIM(%).
<table><tr><td>Setting</td><td>Late-stage GDFN Fine-tuning</td><td>Joint Fine-tuning</td><td>∆PSNR</td></tr><tr><td>IXI-T1 R = 4</td><td>41.7266 / 99.11</td><td>41.9360 / 99.11</td><td>+0.2094</td></tr><tr><td>IXI-T1 R = 8</td><td>35.3777  / 97.04</td><td>35.3879 / 97.04</td><td>+0.0102</td></tr><tr><td>IXI-T2 R = 4</td><td>46.9930 / 99.44</td><td>46.9945 / 99.45</td><td>+0.0015</td></tr><tr><td>IXI-T2 R = 8</td><td>38.4408 / 97.87</td><td>38.4789 / 97.88</td><td>+0.0381</td></tr></table>

![](images/07faedf5d3f2a736a200ec65bc7ed97b2b3912a6f86f354d6edbe8926a929bad.jpg)  
Fig. 4. PSNR comparison of different module combinations on IXI-T2 R = 4.

Table III compares the two output-stage training strategies. Joint fine-tuning yields higher PSNR than late-stage GDFN fine-tuning in all four evaluated settings, with gains ranging from 0.0015 to 0.2094 dB. These differences describe the reported checkpoints and do not establish statistical significance.

## V. DISCUSSION AND CONCLUSION

AGGR-Mamba refines the output of a trained Mamba reconstruction backbone through channel-spatial attention and GDFN-style gated residual filtering. Final-stage optimization updates only the two refinement modules while keeping the backbone fixed. On the evaluated IXI-T2 settings, the method achieves PSNR values of 46.99 dB at $R = 4$ and 38.48 dB at $R = 8$ . The IXI-T2 R = 4 ablation identifies Attention Refine as the main source of improvement, with smaller additional gains from GDFN and joint fine-tuning.

The findings are limited to the evaluated IXI single-coil brain MRI settings. Performance is more favorable on T2 images, and the method does not achieve the highest SSIM in every setting. The small differences between fine-tuning strategies require further validation before drawing conclusions about stability. Future work will investigate realistic multi-coil reconstruction and refinement designs incorporating frequency-domain constraints.

## REFERENCES

[1] K. P. Pruessmann, M. Weiger, M. B. Scheidegger, and P. Boesiger, “SENSE: Sensitivity encoding for fast MRI,” Magnetic Resonance in Medicine, vol. 42, no. 5, pp. 952–962, 1999, doi: https://doi.org/10.1002/ (SICI)1522-2594(199911)42:5⟨952::AID-MRM16⟩3.0.CO;2-S.

[2] M. A. Griswold, P. M. Jakob, R. M. Heidemann, M. Nittka, V. Jellus, J. Wang, B. Kiefer, and A. Haase, “Generalized autocalibrating partially parallel acquisitions (GRAPPA),” Magnetic Resonance in Medicine, vol. 47, no. 6, pp. 1202–1210, 2002, doi: 10.1002/mrm.10171.

[3] M. Lustig, D. Donoho, and J. M. Pauly, “Sparse MRI: The application of compressed sensing for rapid MR imaging,” Magnetic Resonance in Medicine, vol. 58, no. 6, pp. 1182–1195, 2007, doi: 10.1002/mrm.21391.

[4] E. J. Candes, J. Romberg, and T. Tao, “Robust uncertainty principles:\` Exact signal reconstruction from highly incomplete frequency information,” IEEE Transactions on Information Theory, vol. 52, no. 2, pp. 489–509, 2006, doi: 10.1109/TIT.2005.862083.

[5] D. L. Donoho, “Compressed sensing,” IEEE Transactions on Information Theory, vol. 52, no. 4, pp. 1289–1306, 2006, doi: 10.1109/TIT.2006.871582.

[6] M. Lustig, D. L. Donoho, J. M. Santos, and J. M. Pauly, “Compressed sensing MRI,” IEEE Signal Processing Magazine, vol. 25, no. 2, pp. 72–82, Mar. 2008, doi: 10.1109/MSP.2007.914728.

[7] D. Liang, J. Cheng, Z. Ke, and L. Ying, “Deep MRI reconstruction: Unrolled optimization algorithms meet neural networks,” arXiv preprint arXiv:1907.11711, 2020.

[8] S. Wang, Z. Su, L. Ying, X. Peng, S. Zhu, F. Liang, D. Feng, and D. Liang, “Accelerating magnetic resonance imaging via deep learning,” in Proc. IEEE International Symposium on Biomedical Imaging (ISBI), 2016, pp. 514–517, doi: 10.1109/ISBI.2016.7493320.

[9] B. Zhu, J. Z. Liu, B. R. Rosen, and M. S. Rosen, “Image reconstruction by domain-transform manifold learning,” Nature, vol. 555, pp. 487–492, 2018, doi: 10.1038/nature25988.

[10] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015, LNCS, vol. 9351, pp. 234–241, 2015, doi: https://doi.org/10.1007/ 978-3-319-24574-4 28.

[11] I. Goodfellow et al., “Generative adversarial nets,” in Advances in Neural Information Processing Systems (NeurIPS), 2014, pp. 2672–2680.

[12] G. Yang et al., “DAGAN: Deep de-aliasing generative adversarial networks for fast compressed sensing MRI reconstruction,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1310–1321, 2018, doi: 10.1109/TMI.2017.2785879.

[13] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998–6008.

[14] Z. Liu et al., “Swin Transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986.

[15] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in International Conference on Learning Representations (ICLR), 2024, arXiv:2312.00752, doi: 10.48550/arXiv.2312.00752.

[16] Y. Liu et al., “VMamba: Visual state space model,” in Advances in Neural Information Processing Systems (NeurIPS), 2024, arXiv:2401.10166.

[17] Y. Korkmaz and V. M. Patel, “MambaRecon: MRI reconstruction with structured state space models,” in Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025, doi: 10.48550/arXiv.2409.12401.

[18] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[19] J. Schlemper, J. Caballero, J. V. Hajnal, A. Price, and D. Rueckert, “A deep cascade of convolutional neural networks for dynamic MR image reconstruction,” IEEE Transactions on Medical Imaging, vol. 37, no. 2, pp. 491–503, 2018, doi: 10.1109/TMI.2017.2760978.

[20] K. Hammernik et al., “Learning a variational network for reconstruction of accelerated MRI data,” Magnetic Resonance in Medicine, vol. 79, no. 6, pp. 3055–3071, 2018, doi: 10.1002/mrm.26977.

[21] A. Sriram et al., “End-to-end variational networks for accelerated MRI reconstruction,” in Medical Image Computing and Computer-Assisted Intervention–MICCAI 2020, pp. 64–73, 2020, doi: https://doi.org/10. 1007/978-3-030-59713-9 7.

[22] P. Guo, Y. Mei, J. Zhou, S. Jiang, and V. M. Patel, “ReconFormer: Accelerated MRI reconstruction using recurrent Transformer,” IEEE Transactions on Medical Imaging, vol. 43, no. 1, pp. 582–593, 2024, doi: 10.1109/TMI.2023.3314747.

[23] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.- H. Yang, “Restormer: Efficient Transformer for high-resolution image restoration,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5718–5729, doi: 10.1109/CVPR52688.2022.00564.

[24] H. R. Sheikh, M. F. Sabir, and A. C. Bovik, “A statistical evaluation of recent full reference image quality assessment algorithms,” IEEE Transactions on Image Processing, vol. 15, no. 11, pp. 3440–3451, Nov. 2006, doi: 10.1109/TIP.2006.881959.

[25] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004, doi: 10.1109/TIP.2003.819861.

[26] IXI Dataset, “Information eXtraction from Images,” available online: https://brain-development.org/ixi-dataset/.

[27] H. Cao et al., “Swin-Unet: Unet-like pure Transformer for medical image segmentation,” arXiv preprint arXiv:2105.05537, 2021, doi: 10.48550/arXiv.2105.05537.

[28] J. Huang, Y. Fang, Y. Wu, H. Wu, Z. Gao, Y. Li, J. Del Ser, J. Xia, and G. Yang, “Swin Transformer for fast MRI,” Neurocomputing, vol. 493, pp. 281–304, 2022, doi: 10.1016/j.neucom.2022.04.051.