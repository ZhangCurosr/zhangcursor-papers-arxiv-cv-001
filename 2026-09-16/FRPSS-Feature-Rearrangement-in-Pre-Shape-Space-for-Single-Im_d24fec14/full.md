# FRPSS: Feature Rearrangement in Pre-Shape Space for Single-Image Generation

Yuexing Han<sup>1,2,∗</sup> Haoxuan Zhang<sup>1</sup> Bing Wang<sup>1,∗</sup>

<sup>1</sup>School of Computer Engineering and Science, Shanghai University, 99 Shangda Road, Shanghai 200444, People’s Republic of China <sup>2</sup>Key Laboratory of Silicate Cultural Relics Conservation (Shanghai University), Ministry of Education

E-mails: han\_yx@i.shu.edu.cn; 1649366224@qq.com; bingbignwang@shu.edu.cn <sup>∗</sup>Corresponding authors: Yuexing Han and Bing Wang

September 16, 2026

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

## Abstract

Generative models trained on a single image often struggle to balance global structural integrity and local diversity. Existing single-image generation methods commonly rely on random noise to drive the generation process and lack explicit global structural constraints, making the generated results prone to spatial structural misalignment when structural variations occur. To address the issue, Feature Rearrangement in Pre-Shape Space for Single-Image Generation (FRPSS) is proposed in this paper. The core of FRPSS is the Manifold Structural Rearrangement with Feature Augmentation on Geodesic Surface (MSR-FAGS) module. MSR-FAGS replaces the randomly initialized features of the low-scale generator with rearranged Pre-Shape features and uses the features to guide image generation at subsequent scales, thereby reducing the risk of structural misalignment. To support downstream tasks such as stylization, a Scale-adaptive Sliding-window Patch Extraction (SSPE) strategy is further designed, and a directional Contrastive Language-Image Pre-training supervision module with SSPE (CLIP-SSPE) is constructed. Qualitative and quantitative experiments demonstrate that FRPSS achieves the best Single Image Fréchet Inception Distance (SIFID) scores on all three datasets while maintaining competitive Learned Perceptual Image Patch Similarity (LPIPS). Further qualitative experiments verify the efectiveness of FRPSS across multiple downstream tasks with the CLIP-SSPE module.

Keywords: Single-image generation, Pre-Shape Space, Feature rearrangement, Stylized image generation, MSR-FAGS, CLIP-SSPE

## 1 Introduction

Modern image generation models, particularly Generative Adversarial Networks (GANs) [1, 2] and Denoising Difusion Probabilistic Models (DDPMs) [3, 4], have demonstrated remarkable performance in synthesizing visual content with both fidelity and diversity. The generative models have been widely applied in image super-resolution, semantic scene synthesis, cross-modal content generation, and other fields [5]. However, the strong generative capabilities of the models rely on massive amounts of training data, which greatly limits their applicability in specific domains. In fields such as material image analysis, historical artifact restoration, and private artistic creation, the acquisition and annotation of high-quality data are costly [6, 7], which has become an important factor restricting the widespread application of generative models. Against the background, Single-Image Generation (SIG) has gradually developed into an important generation method for extremely data-scarce scenarios [8]. SIG models can synthesize novel image variations that conform to the visual characteristics of the original image by capturing the internal statistics of a single input image. In addition, the single-image generation frameworks can support a series of downstream applications, such as text-guided style transfer, text-guided content generation, and paint-to-image translation. The downstream applications provide users with means for interactive and continuous semantic control over generated content and have certain practical value in customized graphic design, virtual reality, artistic creation, and other scenarios.

In recent years, research on SIG has gradually developed into two major categories. The first category is based on multi-scale GAN models [8, 9, 10], which employ a coarseto-fine hierarchical framework to model local image patch distributions. Although the multi-scale methods can capture the distribution patterns of image patches, the generation process relies on random noise for initialization at low scales. Small structural deviations produced at low scales are continuously amplified during subsequent upsampling, and the progressive accumulation of errors can lead to structural misalignment in the generated results. The second category focuses on recently developed single-image difusion models [11, 12]. The difusion-based methods train denoising difusion models on a single image. Although the difusion-based methods perform better than multi-scale models in image texture generation, the efective receptive field of the network usually needs to be restricted to prevent the difusion model from overfitting to the single image. The limited spatial perception makes it dificult for the model to capture and maintain the global image structure, resulting in structural misalignment in the final generated results.

In addition, stylization [13, 14] is also an important downstream task of SIG frameworks. However, introducing cross-modal semantic supervision based on Contrastive Language-Image Pre-training (CLIP) [15, 16] into existing SIG frameworks still presents certain challenges. Images generated by models that rely on noise initialization are often unstable. Directly applying CLIP-based semantic constraints to the SIG models can not only aggravate error accumulation but also easily introduce hallucinated content, thereby making efective stylization dificult to achieve [13].

To address the problems of existing single-image generation methods, FRPSS, a single-image generation method based on the Hierarchical Patch VAE-GAN framework [10], is proposed. The core component of FRPSS is the Manifold Structural Rearrangement with Feature Augmentation on Geodesic Surface (MSR-FAGS) module, which incorporates FAGS [17] to perform Geodesic surface interpolation in the Pre-Shape Space. The Pre-Shape Space theory [18] is introduced into the single-image generation task to model features and exploit the structural prior information contained in a single training sample. The MSR-FAGS module not only provides subsequent generators with diverse and structurally plausible features, but also alleviates error accumulation during the multi-scale generation process.

For the stylization task in single-image generation frameworks, a Contrastive Language-Image Pre-training supervision module with Scale-adaptive Sliding-window Patch Ex-

traction (CLIP-SSPE) is further designed. The CLIP-SSPE module adopts a scaleadaptive sliding-window patch extraction strategy and calculates the directional CLIP loss accordingly [16] to guide the image stylization process. The module guides the generation of stylized images corresponding to the text prompts provided by users.

![](images/908896d2ada0dc061fe85d86ee1fb0dbb1b637b1b93ac62acd2ab151587bd26d.jpg)  
Figure 1: Overall framework of FRPSS.

As shown in Fig. 1, the MSR-FAGS module generates diverse and structurally plausible features, reduces the risk of structural misalignment in the generated results, and improves the fidelity of the generated images. The PRE-MSR module ensures consistency in representation form between its output features and the features produced by MSR-FAGS. The CLIP-SSPE module employs directional CLIP supervision for semantic guidance to achieve image stylization. Extensive qualitative and quantitative experiments demonstrate that FRPSS is highly competitive in the single-image generation task and can support multiple downstream tasks.

The main contributions of the paper are summarized as follows:

• A single-image generation framework, FRPSS, is proposed. FRPSS can generate diverse images and support interactive control over the generation process.

• MSR-FAGS projects local features into the Pre-Shape Space and employs Geodesic surface interpolation to generate diverse and structurally plausible features.

• CLIP-SSPE constructs directional CLIP supervision during the multi-scale generation process and enables FRPSS to support downstream tasks.

## 2 Related work

## 2.1 Single-image generation

Single-image generation methods synthesize diverse and realistic samples without additional training data by learning the statistics of local image patches from a single input image [19]. Existing solutions in the field can be mainly divided into two categories: methods based on multi-scale Generative Adversarial Networks (GANs) [20] and methods based on difusion models [21, 13, 11, 12].

SinGAN [8] first introduced multi-scale GANs into the single-image generation task. In the architecture, cascaded generators progressively learn image structures and textures from coarse to fine scales. Subsequently, numerous variants based on the multi-scale GAN paradigm were proposed to further improve training eficiency, generation diversity, and other aspects of performance. For example, ConSinGAN [9] accelerates the training process through a parallel multi-stage optimization strategy, whereas HP-VAE-GAN [10] introduces a conditional Variational Autoencoder (VAE) branch to enhance the diversity of generated samples. To enhance the modeling capability for global structures, PetsGAN [22] introduces external priors from a pre-trained generative model to enhance high-leve semantic information in single-image generation. MOGAN [23] models regions of interest and backgrounds through a morphology-aware mechanism to maintain reasonable image structures. SAMDSinGAN [24] introduces a self-attention mechanism to capture longrange dependencies and global contextual information, thereby enhancing the modeling capability for overall image structures.

The second category is based on difusion models for single-image generation [3]. SinDDM [13] introduces a multi-scale cascaded architecture into single-image difusion generation, whereas SinDifusion [11] demonstrates that a difusion network can efectively learn the internal data distribution at a single scale by using a local receptive field, thereby achieving high texture fidelity. Based on the single-image difusion framework, the recent StructDif [12] introduces an adaptive receptive field mechanism to alleviate structural misalignment caused by a single receptive field. Recent studies have also begun to explore eficient training-free adaptation strategies [25]. In addition, recent studies have further extended single-image generation to more complex single-video generation tasks [26, 27].

Although both multi-scale GAN methods and difusion models have advanced the field of single-image generation, existing methods commonly adopt noise-based initialization strategies that inherently lack explicit structural priors. In multi-scale GAN architectures, the absence of reliable priors can lead to structural misalignment at the initial generation stage. The misalignment is progressively amplified when the generated results are upsampled to higher resolutions, leading to error accumulation. In difusion models, the lack of explicit structural priors similarly limits the ability to control global spatial structures. In addition, difusion models usually involve relatively high inference costs. To overcome the limitation caused by insuficient structural priors, FRPSS replaces random noise initialization with rearranged latent features in the Pre-Shape Space at the low-scale generation stage, thereby maintaining the stability of structural layouts in subsequent generated images.

## 2.2 Single-image stylized generation

Many existing stylization methods employ large-scale vision-language models [28] to guide image editing and generation. In early studies, StyleCLIP [15] employs a CLIPbased loss that minimizes the cosine distance between the embeddings of a generated image and a target text prompt to guide text-driven image manipulation. However, directly applying a global CLIP loss may lead to unstable optimization and degraded image quality. To address the limitation, StyleGAN-NADA [29] introduces a directional CLIP loss. The loss constrains the semantic change direction of an image to be consistent with the semantic change direction of text. Subsequent studies further extend directional CLIP guidance to text-guided image stylization and manipulation. For example, CLIP-Styler [16] proposes a patch-wise CLIP loss to guide local texture stylization. Similarly, DifusionCLIP [30] employs a directional CLIP loss for stylized generation in difusion models.

To enable interactive control over single-image generation models, many methods incorporate CLIP-based supervision into the generation process. For example, SinDDM [13] uses the gradients of a global CLIP loss to update the predicted clean image during difusion sampling, enabling text-guided image stylization. However, SinDDM simultaneously adjusts image structure and texture during iterative denoising. The coupling makes semantic supervision prone to disturbing the spatial layout of the image, which may lead to hallucinated content and structural distortion. To alleviate the problem, FRPSS applies a directional CLIP loss [16] to the generated result at each scale. Based on a stable image structure, FRPSS uses the directional CLIP loss to guide image generation and reduce interference with the original spatial layout during semantic injection, thereby reducing artifacts and hallucinated content in the generated results.

## 2.3 The Shape Space theory

The Shape Space theory was introduced by Kendall [18]. Under the theory, shape is defined as geometric information that remains invariant to translation, scaling, and rotation. Compared with the Shape Space, the Pre-Shape Space provides a simplified representation that removes only translation and scaling while retaining rotational information. The Pre-Shape Space also supports corresponding feature augmentation methods. Therefore, FRPSS adopts the Pre-Shape Space to model features. Given a feature vector $U = \{ u _ { 1 } , \cdot \cdot \cdot , u _ { m } \} \in \mathbb { R } ^ { m }$ , landmarks are constructed through a simple lifting operation by setting $v _ { i } = u _ { i }$ . The resulting landmark matrix is $A = [ ( u _ { 1 } , v _ { 1 } ) , \cdot \cdot \cdot , ( u _ { m } , v _ { m } ) ] \in \mathbb { R } ^ { 2 \times m }$ The centered form $A ^ { \prime }$ is then computed and normalized [18] to obtain the Pre-Shape τ :

$$
\begin{array} { l } { { A ^ { \prime } = [ ( u _ { 1 } - \bar { u } , v _ { 1 } - \bar { v } ) , \cdots , ( u _ { m } - \bar { u } , v _ { m } - \bar { v } ) ] , } } \\ { { \tau = \displaystyle \frac { A ^ { \prime } } { \| A ^ { \prime } \| } \in \mathbb { S } ^ { 2 m - 3 } , } } \end{array}\tag{1}
$$

where u¯ and v¯ denote the means of the landmarks along the two dimensions, respectively, and $\| \cdot \|$ denotes the Euclidean norm. All Pre-Shapes τ lie on the hypersphere $\mathbb { S } ^ { 2 m - 3 }$ corresponding to the Pre-Shape Space.

Following the interpolation method of Han et al. [31], FRPSS adopts Geodesic interpolation in the Pre-Shape Space to augment the Pre-Shape samples on the manifold. For two Pre-Shapes $\tau _ { a }$ and $\tau _ { b } .$ let $\theta = \operatorname { a r c c o s } ( \langle \tau _ { a } , \tau _ { b } \rangle )$ denote the Geodesic distance between them in the Pre-Shape Space. The interpolation point on the shortest Geodesic curve between the two Pre-Shapes is given by $\mathbb { G } _ { s l e r p } \ [ 3 1 , 3 2 ]$

$$
\mathbb { G } _ { s l e r p } ( \tau _ { a } , \tau _ { b } , t ) = \cos ( t \theta ) \tau _ { a } + \sin ( t \theta ) \frac { \tau _ { b } - \tau _ { a } \cos \theta } { \sin \theta } ,\tag{2}
$$

where $t \in [ 0 , 1 ]$ denotes the normalized interpolation position along the shortest Geodesic curve. When $t = 0$ , the interpolation result is $\tau _ { a }$ , and when $t = 1$ , the interpolation result is $\tau _ { b }$

To perform interpolation on a Geodesic surface constructed from a set of Pre-Shapes $\mathcal { T } = \{ \tau _ { 1 } , \dots , \tau _ { K } \}$ , FRPSS employs the FAGS algorithm [17] for iterative Geodesic interpolation. Given a set of weights $\omega = \{ \omega _ { 1 } , \dots , \omega _ { K } \}$ sampled from a Dirichlet distribution and satisfying $\textstyle \sum _ { i = 1 } ^ { K } \omega _ { i } = 1$ , FAGS synthesizes a Pre-Shape $\tau _ { s y n }$ through iterative Geodesic interpolation. The initialization is set as $\tilde { \tau } _ { 1 } = \tau _ { 1 }$ , and the subsequent iterations are defined as follows [17]:

$$
\tilde { \tau } _ { j } = \mathbb { G } _ { s l e r p } \left( \tilde { \tau } _ { j - 1 } , \tau _ { j } , \frac { \omega _ { j } } { \sum _ { k = 1 } ^ { j } \omega _ { k } } \right) , \quad \mathrm { f o r } \ j = 2 , \dots , K .\tag{3}
$$

The final output $\tau _ { s y n } = \tilde { \tau } _ { K }$ represents a first-order approximation to the weighted Fréchet mean of the set of Pre-Shapes under the weights $\omega .$

## 3 Methodology

Fig. 1 illustrates the two stages of the FRPSS framework: the Patch-VAE stage and the Patch-GAN stage. The Patch-VAE stage consists of an encoder $E ,$ the PRE-MSR module, and generators $G _ { 0 } , \cdots , G _ { M }$ The Patch-VAE stage is designed to reconstruct images at low scales. PRE-MSR ensures that the feature vectors extracted by E have the same representation form as the output features of MSR-FAGS. The Patch-GAN stage consists of the MSR-FAGS module, generators $G _ { M + 1 } , \cdots , G _ { i } , \cdots , G _ { N }$ , and the corresponding discriminators $D _ { M + 1 } , \cdot \cdot \cdot , D _ { i } , \cdot \cdot \cdot , D _ { N }$ . The Patch-GAN stage is designed to generate diverse images with high-frequency details. MSR-FAGS performs feature augmentation by taking the mean feature $Z$ produced by $E$ as input and generating the augmented feature $Z ^ { \ast }$ . In addition, FRPSS introduces cross-modal semantic supervision into the generation process. The CLIP-SSPE module calculates the directional CLIP loss to guide image generation according to user-provided text prompts.

## 3.1 Patch-VAE stage

Given a single input image x, FRPSS first resizes the image to the predefined largest scale $x _ { N }$ using bilinear interpolation. The resulting image $x _ { N }$ is then progressively downsampled to obtain a set of images at diferent scales, denoted as $\{ x _ { i } \} _ { i = 0 } ^ { N }$ . Here, $x _ { i }$ denotes the downsampled image at scale $i ,$ and $x _ { 0 }$ corresponds to the smallest scale. Each image $x _ { i }$ is associated with a generator $G _ { i }$ for image generation at the corresponding scale. The downsampling ratio between two adjacent scales is set to 0.75.

The encoder E first extracts the features of the input image $x _ { 0 }$ through a convolutional feature extraction module. Two convolutional branches then output the mean feature $\mu \in \mathbb { R } ^ { C \times H \times W }$ and the standard deviation feature $\sigma \in \mathbb { R } ^ { C \times H \times W }$ , respectively. Here, C denotes the channel dimension, while H and W denote the height and width of the feature space, respectively. For any spatial position $j \in \{ 1 , \ldots , H \times W \}$ in the feature space, the C-dimensional vectors $\mu _ { j }$ and $\sigma _ { j }$ jointly define the Gaussian distribution of the image patch corresponding to the position. Following the setting of HP-VAE-GAN [10], the patch-level Kullback–Leibler (KL) divergence can be independently computed at position $j$ . By summing the divergence values over all local positions in the feature map, the total KL loss for training the Patch-VAE stage is obtained as follows:

$$
\mathcal { L } _ { K L } ( x ) = \sum _ { j = 1 } ^ { H \times W } K L [ \mathcal { N } ( \mu _ { j } , \sigma _ { j } ^ { 2 } ) | | \mathcal { N } ( 0 , I ) ] .\tag{4}
$$

Then, standard reparameterization sampling is performed on $\mu$ and $\sigma$ output by E to obtain $Z ^ { \prime } \in \mathbb { R } ^ { C \times H \times W }$ :

$$
Z ^ { \prime } = \mu + \sigma \odot \epsilon ,\tag{5}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ , and $\odot$ denotes element-wise multiplication.

To make $Z ^ { \prime }$ have the same representation form as the output feature $Z ^ { * }$ of MSR-FAGS, the PRE-MSR module divides $Z ^ { \prime }$ along the width W into $K$ consecutive features of width $\alpha .$ . The features are denoted as $\mathcal { G } = ( g _ { 1 } , \cdot \cdot \cdot , g _ { k } , \cdot \cdot \cdot , g _ { K } )$ , where $K = W / \alpha$ and $g _ { k } \in \mathbb { R } ^ { C \times H \times \alpha }$ . Subsequently, according to $\mathrm { E q . } \ \mathrm { ( 1 ) }$ , each $g _ { k }$ is projected into the Pre-Shape Space to obtain $\xi _ { k } \in \mathbb { R } ^ { 2 C \times H \times \alpha }$ . The projected features are then concatenated in order to obtain $Z _ { p r o j } ^ { ' } = ( \bar { \xi } _ { 1 } , \cdot \cdot \cdot , \xi _ { k } , \cdot \cdot \cdot , \xi _ { K } ) \in \bar { \mathbb { R } } ^ { \bar { 2 } C \times H \times W }$ . The entire process is shown in Fig. 2(a). The detailed process of the PRE-MSR module is shown in Algorithm 1.

$Z _ { p r o j } ^ { ' }$ is input into the generator $G _ { 0 }$ to obtain the image $\bar { x } _ { 0 }$ , where $\hat { x } _ { 0 } = G _ { 0 } ( Z _ { p r o j } ^ { ' } )$ To progressively reconstruct the original image from low scales, details are gradually added at the subsequent intermediate scales $i \in \{ 1 , \cdots , M \}$ . The generators $G _ { 0 } , G _ { 1 } , \cdots , G _ { M }$ correspond to diferent generation scales. $G _ { 0 }$ reconstructs the low-scale image according to the latent feature output by the encoder E. The generators $G _ { 1 } , \cdots , G _ { M }$ predict the

Algorithm 1: PRE-MSR Module in the Patch-VAE training stage.   
Input : Latent Feature $\overline { { Z ^ { \prime } \in \mathbb { R } ^ { C \times H \times W } } }$ , Group size α   
Output: Projected Feature $Z _ { p r o j } ^ { ' } \in \mathbb { R } ^ { 2 C \times H \times W ^ { ' } }$   
// 1. Spatial Split   
1 Split $Z ^ { \prime }$ along width into spatial groups of size α;   
2 Let $\mathcal { G } = \{ g _ { 1 } , g _ { 2 } , \dots , g _ { K } \}$ be the sequence of groups, where $K = W / \alpha ;$   
3 Initialize projected list $\mathcal { P }  [ ] ;$   
// 2. Project to the Pre-Shape Space   
4 for $k \gets 1$ to K do   
$/ /$ Project $g _ { k } \in \mathbb { R } ^ { C \times H \times \alpha }$ to a manifold point   
5 $\xi _ { k } \gets \mathtt { P r o j }$ ect(g<sub>k</sub>);   
6 Append $\xi _ { k }$ to ${ \mathcal { P } } ;$   
// 3. Assembly & Feature Lifting   
7 $Z _ { f l a t } \gets \mathsf { C o n c a t } ( \mathcal { P } )$   
8 $Z _ { p r o j } ^ { ' } \gets$ Reshape $\left( Z _ { f l a t } \right)$   
9 return $Z _ { p r o j } ^ { ' } ;$   
(a) PRE-MSR   
(W) ξ2 SK   
Split Project   
(H)   
Concate   
Z   
g₁ g₂ gK Pre-Shape Space   
(b) MSR-FAGS   
(W)   
Split Project   
(H)   
T TK   
Z   
g1 g2 gK C1 C2 CK   
十★…★   
Reshuffle Concate   
τ2   
Split Project   
Pre-Shape Space   
Zreshuffled + + +   
g1 g2 gk

![](images/6be1219a64d0195a2b0197e25007f777a2520680e1217721b79634d9644fd625.jpg)  
Figure 2: Feature projection and rearrangement based on the Pre-Shape Space. (a) PRE-MSR module and (b) MSR-FAGS module.

residual of the current scale relative to the generated image at the previous scale. During training at scale $i ,$ the encoder E, the generator $G _ { 0 } ,$ and $G _ { i }$ are jointly optimized, while the parameters of the intermediate generators $G _ { 1 } , \ldots , G _ { i - 1 }$ remain frozen. Since the encoder $E$ is continuously updated during training at diferent scales, its output latent feature also changes accordingly. Therefore, $G _ { 0 }$ needs to be jointly optimized with E to maintain the mapping relationship between the latent representation and the reconstruction result at the lowest scale. The intermediate generators that have already been trained remain frozen to prevent subsequent scale training from changing their learned scale-specific residual mappings. Let $\bar { x } _ { i - 1 }$ denote the reconstruction result at the previous scale. The previous reconstruction $\bar { x } _ { i - 1 }$ is first upsampled using bilinear interpolation, denoted by ↑, to match the image resolution at scale i. The upsampled result is then input into the generator $G _ { i } , \bar { x } _ { i }$ can be expressed as the sum of the upsampled result and the residual predicted by the generator [10]:

$$
\begin{array} { r } { \bar { x } _ { i } = \uparrow \bar { x } _ { i - 1 } + G _ { i } \big ( \uparrow \bar { x } _ { i - 1 } \big ) . } \end{array}\tag{6}
$$

The reconstruction loss between the generated image $\bar { x } _ { i }$ and the real image $x _ { i }$ is used to calculate the diference between them:

$$
\mathcal { L } _ { R e c o n } ( \bar { x } _ { i } , x _ { i } ) = \| \bar { x } _ { i } - x _ { i } \| _ { 2 } ,\tag{7}
$$

where $\| \cdot \| _ { 2 }$ denotes the $L _ { 2 }$ norm.

Following the β-VAE framework proposed by Higgins et al. [33], the hyperparameter $\beta$ is introduced in the Patch-VAE stage to weight the $K L$ divergence loss. Finally, the total loss of the Patch-VAE stage is defined as follows [10]:

$$
\mathcal { L } _ { V A E } ( x _ { 0 } , \bar { x } _ { i } , x _ { i } ) = \lambda _ { r } \mathcal { L } _ { R e c o n } ( \bar { x } _ { i } , x _ { i } ) + \beta \mathcal { L } _ { K L } ( x _ { 0 } ) , i \in \{ 0 , \cdots , M \} .\tag{8}
$$

where $\lambda _ { r }$ is a hyperparameter used to balance the contributions of the losses.

## 3.2 Patch-GAN stage

When the scale $i \geq M + 1$ , the model enters the Patch-GAN stage to generate diverse images with high-frequency texture information. At the stage, the parameters of the encoder E and the generators $G _ { 0 } , \cdots , G _ { M }$ trained in the Patch-VAE stage are frozen.

The mean feature $\mu$ output by the encoder E serves as the input vector Z of the MSR-FAGS module, denoted as $Z ~ = ~ \mu$ Here, only the mean feature is used, and reparameterization sampling is no longer performed, because $\mu$ encodes the structural prior of the original image. The deterministic representation avoids the random noise perturbation introduced by the reparameterization sampling process and provides a stable reference for the subsequent projection into the Pre-Shape Space. The MSR-FAGS module rearranges the feature $Z$ in the Pre-Shape Space to generate a new feature $Z ^ { * }$ with structural variations. Subsequently, $Z ^ { * }$ is input into the generators $G _ { 0 } , \cdots , G _ { M }$ trained in the Patch-VAE stage to obtain the image $\hat { x } _ { M }$ at scale M. Based on the result, scale-by-scale adversarial training is performed at the subsequent scales $M + 1 , \cdots , N$

As shown in the Patch-GAN stage of Fig. 1, for scale i, the generated image $\bar { x } _ { i - 1 } ^ { a d v }$ at the previous scale is first upsampled to the current scale. Next, the upsampled result is added to the noise $z _ { i }$ at the current scale, and the sum is input into the generator $G _ { i }$ to predict the high-frequency texture residual. Finally, the residual output by $G _ { i }$ is added to the upsampled $\bar { x } _ { i - 1 } ^ { a d v }$ to obtain the generated image $\bar { x } _ { i } ^ { a d v }$ at the current scale $i .$ Therefore, the generation process of $\bar { x } _ { i } ^ { a d v }$ can be defined as follows:

$$
\bar { x } _ { i } ^ { a d v } = \uparrow \bar { x } _ { i - 1 } ^ { a d v } + G _ { i } ( \uparrow \bar { x } _ { i - 1 } ^ { a d v } + z _ { i } ) .\tag{9}
$$

The WGAN-GP loss [34] is adopted for adversarial training at the stage:

$$
\mathcal { L } _ { a d v } ( z _ { i } , x _ { i } ) = \operatorname* { m i n } _ { G _ { i } } \operatorname* { m a x } _ { D _ { i } } \left\{ \mathbb { E } [ D _ { i } ( x _ { i } ) ] - \mathbb { E } [ D _ { i } ( \bar { x } _ { i } ^ { a d v } ) ] - \lambda \mathbb { E } \left[ ( \| \nabla _ { \hat { x } } D _ { i } ( \hat { x } ) \| _ { 2 } - 1 ) ^ { 2 } \right] \right\} ,\tag{10}
$$

where $D _ { i }$ denotes the discriminator at the current scale i, ∇ denotes the gradient operator, and λ is the gradient penalty coeficient. Here, $x _ { i }$ denotes the real image at the current scale i, while xˆ denotes the uniformly interpolated result between the real sample $x _ { i }$ and the generated sample $\bar { x } _ { i } ^ { a d v }$ . The calculation of $\hat { x }$ can be expressed as follows [34]:

$$
\hat { x } = \epsilon x _ { i } + ( 1 - \epsilon ) \bar { x } _ { i } ^ { a d v } , \quad \epsilon \sim U ( 0 , 1 ) .\tag{11}
$$

In addition, the reconstruction loss used in the Patch-VAE stage is also adopted in the Patch-GAN stage. Therefore, the total adversarial training loss is as follows:

$$
\mathcal { L } _ { G A N } ( z _ { i } , \bar { x } _ { i } , x _ { i } ) = \lambda _ { r } \mathcal { L } _ { R e c o n } ( \bar { x } _ { i } , x _ { i } ) + \beta _ { a d v } \mathcal { L } _ { a d v } ( z _ { i } , x _ { i } ) , i \in \{ M + 1 , \cdots , N \} ,\tag{12}
$$

where $\lambda _ { r }$ and $\beta _ { a d v }$ are hyperparameters used to balance the contributions of the losses.   
The generation process of $\bar { x } _ { i }$ is consistent with that in the Patch-VAE stage.

## 3.3 Manifold Structural Rearrangement with FAGS

The MSR-FAGS module generates diverse and structurally plausible features. Algorithm 2 describes the complete execution process of the MSR-FAGS module.

To overcome the data limitation of a single sample, MSR-FAGS adopts the feature decomposition strategy of MultiOSG [35]. As shown in Fig. 2(b), the feature $Z$ is uniformly divided along the width W into K consecutive features in the same manner as the PRE-MSR module. The resulting features form an ordered set $\mathcal { G } ^ { \prime } = \{ g _ { 1 } ^ { \prime } , \ldots , g _ { k } ^ { \prime } , \ldots , g _ { K } ^ { \prime } \}$ where $g _ { k } ^ { \prime } \in \mathbb { R } ^ { C \times H \times \alpha }$ Here, $K = W / \alpha$ denotes the number of features in $\mathcal { G } ^ { \prime }$ , and α denotes the width of each feature. Subsequently, each feature $g _ { k } ^ { \prime }$ is projected into the Pre-Shape Space in order according to Eq. (1). The projection yields the feature set $\mathcal { T } = \{ \tau _ { 1 } , \dots , \tau _ { k } , . . . , \tau _ { K } \}$ , where $\tau _ { k } \in \mathbb { R } ^ { 2 C \times H \times \alpha }$

To introduce feature diversity, MSR-FAGS first splits the feature Z along the width W into W column features of size $C \times H$ . The order of the W features is then randomly shufled, and the features are concatenated again to obtain $Z _ { r e s h u f f l e d }$ . Subsequently, MSR-FAGS divides $Z _ { r e s h u f f l e d }$ along the width W into $K$ consecutive features to obtain the set $\mathcal { G } ^ { r } = \{ g _ { 1 } ^ { r } , . . . , g _ { k } ^ { r } , . . . , g _ { K } ^ { r } \}$ , where $g _ { k } ^ { r } \ \in \ \mathbb { R } ^ { C \times H \times \alpha }$ . Each feature in $\mathcal G ^ { r }$ is also projected into the Pre-Shape Space according to Eq. (1). The projection yields the feature set $\mathcal { T } ^ { r } = \{ \tau _ { 1 } ^ { r } , . . . , \tau _ { k } ^ { r } , . . . , \tau _ { K } ^ { r } \}$ , where $\tau _ { k } ^ { r } \in \mathbb { R } ^ { 2 C \times H \times \alpha }$

To expand the set $\mathcal { T } ^ { r }$ , iterative interpolation is performed on the Geodesic surface constructed by the Pre-Shapes in $\mathcal { T } ^ { r }$ according to Eq. (3). In each new feature generation step, MSR-FAGS first samples a set of weights ω from the Dirichlet distribution to determine the contribution proportion of each feature in $\mathcal { T } ^ { r }$ . Using the sampled weights $\omega ,$ iterative Geodesic interpolation is then performed on the features in $\mathcal { T } ^ { r }$ on the Geodesic surface. The interpolation generates a new feature $s _ { j }$ . After repeating the above process $Q$ times, the module obtains a new feature set $\mathcal { S } = \{ s _ { 1 } , \cdot \cdot \cdot , s _ { j } , \cdot \cdot \cdot , s _ { Q } \}$ and merges it with $\mathcal { T } ^ { r }$ to obtain the expanded feature set $\mathcal { C }$

Algorithm 2: MSR-FAGS module in the Patch-GAN stage.   
Input : Feature tensor $\overline { { Z \in \mathbb { R } ^ { C \times H \times W } } }$ , Group size α, Number of interpolated   
features $Q$   
Output: Enhanced Feature $Z ^ { * } \in \mathbb { R } ^ { 2 C \times H \times W }$   
// 1. Spatial Decomposition & Anchor Path Construction   
1 Split Z along width W into slices and group them by size α;   
2 Let $\mathcal { G } ^ { \prime } = \{ g _ { 1 } ^ { \prime } , \ldots , g _ { k } ^ { \prime } , \ldots , g _ { K } ^ { \prime } \}$ be the ordered groups, where $K = W / \alpha ;$   
3 for $k \gets 1$ to K do   
4 $\mathsf { \Gamma } \lfloor \tau _ { k } \gets \mathsf { P r o j e c t } ( g _ { k } ^ { \prime } )$ ; $/ /$ Map to the Pre-Shape Space   
5 Set $\mathcal { T }  \{ \tau _ { 1 } , . . . , \tau _ { K } \}$   
// 2. Candidate Path via Global Shuffling   
6 $Z _ { r e s h u f f l e d }  \texttt { G l }$ obalPermute $\left( Z , \dim = W \right) ; \quad / /$ Global random reshuffling   
7 Split $Z _ { r e s h u f f l e d }$ into $\mathcal { G } ^ { r } = \{ g _ { 1 } ^ { r } , . . . , g _ { k } ^ { r } , . . . , g _ { K } ^ { r } \}$   
8 for k ← 1 to K do   
9 $\lfloor \tau _ { k } ^ { r } \gets \mathtt { P r o j e c t } ( g _ { k } ^ { r } )$ ; $/ /$ Map to the Pre-Shape Space   
10 Set Candidate Base $\mathcal { T } ^ { r } \gets \{ \tau _ { 1 } ^ { r } , \dots , \tau _ { k } ^ { r } , \dots , \tau _ { K } ^ { r } \}$   
// 3. Manifold Expansion via FAGS   
11 Initialize $s  \emptyset ;$   
12 for $j  1$ to Q do   
13 Sample weights $\omega \sim \operatorname { D i r } ( \mathbf { 1 } _ { K } )$   
14 $s _ { j } \gets \mathtt { F A G S } ( T ^ { r } , \omega )$ $/ /$ Iterative Geodesic Interpolation   
15 Add $s _ { j }$ to $s ;$   
16 Set $\mathcal { C }  \mathsf { C o n c a t } ( \mathcal { T } ^ { r } , \mathcal { S } ) ;$   
// 4. Feature Reconstruction via Shortest Geodesic Matching   
17 Initialize reconstructed list ${ \mathcal { L } } \gets [ ] ;$   
18 for $k \gets 1$ to K do   
// Search for best feature match in $\mathcal { C }$   
19 $c _ { k } ^ { * } \gets$ arg min arccos $( \langle \tau _ { k } , c \rangle ) \colon$   
c∈C   
20 Append $c _ { k } ^ { * }$ to $\mathcal { L } ;$   
21 $Z ^ { * } \gets \mathtt { C o n c a t } ( { \mathcal { L } } )$ ; $/ /$ Reshape and Assembly (2C channels)   
22 return $Z ^ { * } ;$

MSR-FAGS then matches the features in $\tau$ with those in C using the shortest Geodesic distance. For each feature $\tau _ { k }$ corresponding to a position in $\tau ,$ MSR-FAGS searches C for the feature $c _ { k } ^ { * }$ with the shortest Geodesic distance to $\tau _ { k }$ . The matching allows the recombined feature to preserve the spatial coherence of the original image as much as possible while introducing variations. Here, MSR-FAGS adopts the shortest Geodesic distance in the Pre-Shape Space [31] as the matching criterion. The corresponding matching process is defined as follows:

$$
c _ { k } ^ { * } = \underset { c \in \mathcal { C } } { \arg \operatorname* { m i n } } \ \operatorname { a r c c o s } \left( \left. \tau _ { k } , c \right. \right) .\tag{13}
$$

MSR-FAGS concatenates the matched K features in order along the width $W$ to obtain $Z ^ { \ast } = ( c _ { 1 } ^ { \ast } , \dots , c _ { k } ^ { \ast } , \dots , c _ { K } ^ { \ast } ) \in \mathbb { R } ^ { 2 C \times H \times W }$ and inputs $Z ^ { * }$ into $G _ { 0 }$ . Finally, the generators are trained according to the process in Section 3.2.

## 3.4 Stylized generation

To further explore the applicability of FRPSS in practical scenarios, FRPSS is extended to downstream stylization tasks. In the stylization task, FRPSS uses a pre-trained CLIP model to provide cross-modal semantic priors. As shown in Fig. 1, the CLIP-SSPE module is incorporated into each generator $G _ { i }$ in both the Patch-VAE and Patch-GAN stages. FRPSS progressively accumulates the generation results at diferent scales to form the final generated image rather than relying on the last generator to independently generate the final image. As a result, semantic constraints applied only at the final scale cannot suficiently afect the visual features already formed at previous scales. CLIP-SSPE therefore progressively applies semantic supervision at each scale and introduces the target semantics scale by scale during the multi-scale generation process. As shown in Fig. 3, the CLIP-SSPE module contains a frozen CLIP text encoder $E ^ { T }$ a frozen image encoder $E ^ { I }$ , and the SSPE module. $E ^ { T }$ and $E ^ { I }$ encode the text and image, respectively, and project them into the joint embedding space. The SSPE module adaptively extracts image patches according to the generated images at diferent scales. Following the setting of CLIPStyler [16], CLIP-SSPE adopts a directional CLIP supervision mechanism. The module constructs the directional constraints $\mathcal { L } _ { g l o b a l }$ and $\mathcal { L } _ { p a t c h }$ at the global and patch-wise levels, respectively. The directional constraints make the visual change direction from the real image to the generated image consistent with the semantic change direction from the source text prompt to the target text prompt.

![](images/ae0da66d97b0aa177c3ec4f6b8e0a8a7d8443c5dc883a1ae814d6caea3f27284.jpg)  
Figure 3: Illustration of the CLIP stylization loss calculation based on the CLIP-SSPE module.  
The inputs of CLIP-SSPE include the real image $x _ { r e a l }$ , the generated image $x f a k e$ at

the current scale, and the source text prompt src and target text prompt trg provided by the user. Here, src indicates that the input image is an image, while $t r g$ indicates the target style to be changed. CLIP-SSPE inputs the two text prompts into the frozen CLIP text encoder $E ^ { T }$ to obtain the corresponding text features $E _ { s r c } ^ { T }$ and $E _ { t r g } ^ { T } .$ To align the image transformation direction with the text semantic transformation direction in the CLIP joint embedding space, the text direction vector $\Delta T$ is defined based on CLIPStyler [16] as follows:

$$
\begin{array} { r } { \Delta T = E _ { t r g } ^ { T } - E _ { s r c } ^ { T } . } \end{array}\tag{14}
$$

At any scale, the CLIP-SSPE module uses the complete generated image and the corresponding real image to calculate the global directional CLIP loss. The frozen image encoder $E ^ { I }$ extracts the features $E _ { f a k e } ^ { I }$ and $E _ { r e a l } ^ { I }$ from $x f a k e$ and x , respectively. Following CLIPStyler [16], the global image direction vector $\Delta I _ { g l o b a l }$ is defined as follows:

$$
\Delta I _ { g l o b a l } = E _ { f a k e } ^ { I } - E _ { r e a l } ^ { I } .\tag{15}
$$

The global directional loss $\mathcal { L } _ { g l o b a l }$ is used to maximize the cosine similarity between $\Delta I _ { g l o b a l }$ and $\Delta T$

$$
\mathcal { L } _ { g l o b a l } = 1 - \frac { \Delta I _ { g l o b a l } \cdot \Delta T } { \Vert \Delta I _ { g l o b a l } \Vert \cdot \Vert \Delta T \Vert } .\tag{16}
$$

For the calculation of the patch-wise directional CLIP constraint, SSPE extracts diferent numbers of image patches at diferent scales. Assume that $H \times W$ denotes the image resolution at the current scale. To perform local image patch extraction, SSPE adopts a sliding crop mechanism. The size r of the sliding crop window is adaptively adjusted according to the current image resolution and is restricted to the predefined range [64, 128]. The size r of the sliding crop window is calculated as follows:

$$
r = \operatorname* { m a x } \left( 6 4 , \operatorname* { m i n } \left( \left\lfloor { \frac { \operatorname* { m i n } ( H , W ) } { 2 } } \right\rfloor , 1 2 8 \right) \right) .\tag{17}
$$

To ensure a suficient overlap ratio between the cropped image patches for maintaining texture continuity, the sliding stride s is defined as half of the window size:

$$
s = { \frac { r } { 2 } } .\tag{18}
$$

Based on the above configuration, for an image with a resolution of $H \times W$ , the total number D of image patches extracted by the sliding window is:

$$
D = \left\lfloor { \frac { H } { s } } \right\rfloor \times \left\lfloor { \frac { W } { s } } \right\rfloor .\tag{19}
$$

It is worth noting that when the image size is smaller than the minimum window limit or when the image patch extraction condition cannot be satisfied, SSPE no longer performs local cropping. Instead, SSPE directly uses the complete generated image and real image to calculate the directional constraint. In the case, the patch-wise constraint degenerates into the global constraint.

During image patch cropping, let $p _ { f a k e } ^ { j }$ and $p _ { r e a l } ^ { j }$ denote the j-th corresponding image patches cropped from the generated image and the real image, respectively. The image direction vector $\Delta I _ { p a t c h } ^ { j }$ between the two patches is defined as follows [16]:

$$
\Delta I _ { p a t c h } ^ { j } = E ^ { I } ( p _ { f a k e } ^ { j } ) - E ^ { I } ( p _ { r e a l } ^ { j } ) .\tag{20}
$$

Subsequently, CLIP-SSPE calculates the cosine distance between $\Delta I _ { p a t c h } ^ { j }$ and the text direction vector $\Delta T$ . The arithmetic mean of the cosine losses corresponding to the D image patches cropped at the current scale is then taken to obtain the final patch-wise directional loss $\mathcal { L } _ { p a t c h }$ [16]:

$$
\mathcal { L } _ { p a t c h } = \frac { 1 } { D } \sum _ { j = 1 } ^ { D } \left( 1 - \frac { \Delta I _ { p a t c h } ^ { j } \cdot \Delta T } { \| \Delta I _ { p a t c h } ^ { j } \| \cdot \| \Delta T \| } \right) .\tag{21}
$$

Finally, the Total Variation (TV) regularization loss $\mathcal { L } _ { t v }$ [16] is also introduced into the CLIP-SSPE module. The overall CLIP stylization loss is:

$$
\begin{array} { r } { \mathcal { L } _ { c l i p } = \lambda _ { g l o b a l } \mathcal { L } _ { g l o b a l } + \lambda _ { p a t c h } \mathcal { L } _ { p a t c h } + \lambda _ { t v } \mathcal { L } _ { t v } . } \end{array}\tag{22}
$$

where $\lambda _ { g l o b a l } , \lambda _ { p a t c h }$ , and $\lambda _ { t v }$ are hyperparameters used to balance the contributions of the losses. In the experiments, $\lambda _ { g l o b a l } , \lambda _ { p a t c h }$ , and $\lambda _ { t v }$ are set to 3.0, 5.0, and 0.02, respectively.

The above text-guided method uses the text direction vector $\Delta T$ as the target semantic direction. The directional supervision mechanism of CLIP-SSPE can be further extended to image-guided style transfer tasks. As shown in Fig. 4, the image-guided method uses the target style image to provide the change direction information. Let $x _ { s r c }$

![](images/4219ee32d00fe2c44d4162e7563e225625076b7abc4518aa07979b15c0047ff7.jpg)  
Figure 4: Illustration of the loss calculation for image-guided style transfer.

denote the original image and $x _ { r e f }$ denote the target style image. FRPSS inputs the two images into the frozen CLIP image encoder $E ^ { I }$ , respectively, to obtain the corresponding image features $E _ { s r c } ^ { I }$ and $E _ { r e f } ^ { I }$ . Subsequently, the style change direction vector $\Delta I _ { s t y l e }$ is constructed according to the diference between the target image style feature vector and the original image feature vector:

$$
\Delta I _ { s t y l e } = E _ { r e f } ^ { I } - E _ { s r c } ^ { I } .\tag{23}
$$

During image-guided style transfer, $\Delta I _ { s t y l e }$ replaces the text direction vector $\Delta T$ in $\operatorname { E q }$ (14), and the global and patch-wise directional constraints are still used for optimization. In this way, the CLIP-SSPE module uses the visual information provided by the target style image $x _ { r e f }$ to guide the generated result toward the target style.

## 4 Experiments

## 4.1 Experimental datasets

FRPSS is evaluated on three general single-image generation datasets to examine the generation capability for images with diferent textures and semantic structures, including Places50 [8], MSID16 [8], and SIGD16 [36]. Places50 contains 50 scene images with complex semantic structures. MSID16 contains 16 images covering multiple types of visual content, including plants, animals, and natural landscapes. SIGD16 contains 16 images with significant topological variations and is used to evaluate the generation stability of the model. The dataset adopted by SinDDM [13] is used for the text-guided stylization experiments. The dataset provided by InstantStyle-Plus [37] is used for the image-guided stylization experiments.

## 4.2 Evaluation metrics

Single Image Fréchet Inception Distance (SIFID) [8] and Learned Perceptual Image Patch Similarity (LPIPS) [38] are used to quantitatively evaluate the generated results.

SIFID follows the Fréchet distance form of the standard FID [39], but the statistical object is changed from the feature distribution of an image set to the spatial feature distribution within a single image. Given a real image $x _ { r }$ and a generated image $\scriptstyle x _ { f } ,$ a shallow convolutional block of the pre-trained Inception-V3 [40] is used to extract the feature map $\psi ( { \boldsymbol { x } } ) ~ \in ~ \mathbb { R } ^ { C \times H \times W }$ The feature map is then unfolded along the spatial dimensions into $H \times W ~ C \mathrm { - } C$ imensional feature vectors to estimate the mean $\boldsymbol { \mu } \in \mathbb { R } ^ { C }$ and covariance matrix $\Sigma \in \mathbb { R } ^ { C \times C }$ of the local feature distribution of the image. The SIFID between the real image and a single generated image is defined as follows:

$$
\mathrm { S I F I D } ( x _ { r } , x _ { f } ) = \Vert \mu _ { r } - \mu _ { f } \Vert _ { 2 } ^ { 2 } + \mathrm { T r } \left( \Sigma _ { r } + \Sigma _ { f } - 2 \left( \Sigma _ { r } \Sigma _ { f } \right) ^ { 1 / 2 } \right) ,\tag{24}
$$

where $\mu _ { r } , \Sigma _ { r }$ and $\mu _ { f } , \Sigma _ { f }$ denote the feature means and covariance matrices corresponding to the real image and the generated image, respectively, and $\operatorname { T r } ( \cdot )$ denotes the trace of a matrix. For each original image, 20 samples are generated by the model, and the SIFID of each generated sample with respect to the original image is calculated. The mean SIFID is used as the final evaluation result. A lower SIFID value indicates that the distribution of the generated image is closer to that of the original image in terms of deep feature statistics.

To measure the diversity among the generated results, the pairwise LPIPS distances among the generated samples corresponding to the same original image are calculated. The LPIPS value is calculated based on the deep features extracted by the pre-trained AlexNet [41, 8] network ϕ(x). For any two generated images x and $y .$ their LPIPS distance is defined as follows [38]:

$$
d ( x , y ) = \sum _ { l = 1 } ^ { L } \frac { 1 } { H _ { l } W _ { l } } \sum _ { h = 1 } ^ { H _ { l } } \sum _ { w = 1 } ^ { W _ { l } } \left. w _ { l } \odot \left( \hat { \phi } _ { l } ( x ) _ { h , w } - \hat { \phi } _ { l } ( y ) _ { h , w } \right) \right. _ { 2 } ^ { 2 } ,\tag{25}
$$

where $\phi _ { l } ( \boldsymbol { x } ) \in \mathbb { R } ^ { C _ { l } \times H _ { l } \times W _ { l } }$ denotes the feature map output by the l-th layer of the pretrained AlexNet, $H _ { l }$ and $W _ { l }$ denote the height and width of the feature map at the layer, $( h , w )$ denotes the spatial position coordinates on the feature map, $\phi _ { l } ( x ) _ { h , w } \in \mathbb { R } ^ { C _ { l } }$ denotes the channel feature vector corresponding to the position, $\hat { \phi } _ { l } ( \cdot )$ denotes the feature obtained by normalizing $\phi _ { l } ( \cdot )$ along the channel dimension, and $w _ { l }$ denotes the linear weighting coeficient of the l-th layer. For the 20 samples generated from the same original image, the LPIPS distances of all 190 non-repeated image pairs are calculated, and their mean is taken as the diversity metric. A higher LPIPS value indicates a larger perceptual diference among the generated samples, while a lower value indicates a higher degree of homogeneity among the samples.

## 4.3 Qualitative and quantitative analysis of image generation

Fig. 5 and Fig. 6 visually compare the generation results of FRPSS with those of other methods on the Places50 and SIGD16 datasets.

Real  
SinGAN  
ConSinGAN  
PetsGAN  
SAMDSinGAN  
SinDiffusion  
Ours  
![](images/938f3ffe659d1b2dd6dd055aa493545a862294eb3ada2ff1ff627038a4386ef8.jpg)  
Figure 5: Images generated by diferent methods on the Places50 dataset.

Some results generated by traditional multi-scale generation models, such as Sin-GAN [8] and ConSinGAN [9], show obvious structural misalignment. Structural deviations produced at low-resolution stages may accumulate during the progressive generation process, further leading to local incoherence such as path distortion and mountain discontinuity. Similarly, although HP-VAE-GAN [10] achieves a relatively high LPIPS score, obvious global structural misalignment can still be observed in some generated results. GPNN [36] can synthesize relatively realistic local textures, but the method tends to copy the nearest image patches from the training image, thereby limiting the diversity of generated samples to some extent. In addition, difusion-based single-image generation methods such as SinDifusion [11] have a strong capability for detail synthesis, but color leakage between regions or local stitching incoherence can still be observed in some generated results.

Compared with the above methods, FRPSS achieves a better balance between global structural integrity and local visual fidelity. MSR-FAGS performs feature rearrangement and matching in the low-scale feature space to provide more possibilities for global layout variations, while preserving the coherence of the spatial structure as much as possible.

Real  
SinGAN  
HPVAEGAN  
GPNN  
PetsGAN  
Sindiffusion  
Ours  
![](images/4dbe041882a474cd423dfa433d32a71810c258aefbdb976f71dd4a95b9eebd11.jpg)  
Figure 6: Images generated by diferent methods on the SIGD16 dataset.

As shown in the figures, the images generated by FRPSS exhibit fewer spatial misalignment artifacts and maintain better consistency with the original image in terms of color distribution and contrast.

Table 1 reports the quantitative comparison results on the Places50, MSID16, and SIGD16 datasets. In terms of SIFID, FRPSS achieves the best results on all three datasets, indicating that the generated results of FRPSS are closer to the original images in terms of deep feature statistical distributions. In contrast, some comparison methods, such as SinDDM and SinDifusion, obtain relatively high LPIPS scores, but their SIFID values are also relatively high, indicating larger deviations between the generated results and the feature distributions of the original images. Methods such as ConSinGAN and SinGAN have relatively low SIFID values, but their LPIPS scores are also relatively low, indicating limited diversity. The results show that FRPSS can still maintain local texture statistical characteristics similar to those of the original images while introducing feature variations.

In terms of generation diversity, FRPSS achieves competitive LPIPS scores on all three datasets, indicating that the generated samples maintain good perceptual diversity while preserving the visual characteristics of the original images.

## 4.4 Image stylization transformation

The section presents the applications of FRPSS to multiple downstream image manipulation tasks, including text-guided image style transfer, text-guided image content generation, image-guided style transfer, and paint-to-image.

Text-guided style transfer. Here, src is set to “a photo”, and trg is set to “anime style”, “Monet style”, “Picasso style”, and “Van Gogh style”. CLIP-SSPE constructs directional constraints according to the visual change direction from the real image to the generated image and the semantic change direction from the source text to the target text, so that the generated results preserve the main content and spatial structure of the original image as much as possible while introducing the target style. Fig. 7, Fig. 8, Fig. 9, and Fig. 10 compare the generated results of FRPSS and SinDDM [13] under the four text prompts.

Table 1: Quantitative comparison on the three datasets.
<table><tr><td rowspan="2"></td><td colspan="2">Places50</td><td colspan="2">MSID16</td><td colspan="2">SIGD16</td></tr><tr><td>SIFID↓</td><td>LPIPS↑</td><td>SIFID↓</td><td>LPIPS↑</td><td>SIFID↓</td><td>LPIPS↑</td></tr><tr><td>SinGAN</td><td>0.09</td><td>0.266</td><td>0.28</td><td>0.279</td><td>0.27</td><td>0.344</td></tr><tr><td>HP-VAE-GAN</td><td>0.30</td><td>0.380</td><td>0.05</td><td>0.447</td><td>0.07</td><td>0.449</td></tr><tr><td>ConSinGAN</td><td>0.06</td><td>0.305</td><td>0.28</td><td>0.296</td><td>0.06</td><td>0.287</td></tr><tr><td>GPNN</td><td>0.07</td><td>0.238</td><td>0.57</td><td>0.289</td><td>0.11</td><td>0.317</td></tr><tr><td>SinDDM</td><td>0.68</td><td>0.345</td><td>0.36</td><td>0.364</td><td>0.85</td><td>0.349</td></tr><tr><td>PetsGAN</td><td>0.08</td><td>0.310</td><td>0.29</td><td>0.342</td><td>0.32</td><td>0.340</td></tr><tr><td>SinFusion</td><td>0.64</td><td>0.368</td><td>0.48</td><td>0.339</td><td>0.69</td><td>0.365</td></tr><tr><td>MultiOSG</td><td>0.39</td><td>0.310</td><td>0.20</td><td>0.382</td><td>0.13</td><td>0.362</td></tr><tr><td>SinDiffusion</td><td>0.06</td><td>0.387</td><td>0.49</td><td>0.489</td><td>0.43</td><td>0.445</td></tr><tr><td>StructDiff</td><td>0.04</td><td>0.311</td><td>0.40</td><td>0.368</td><td>0.14</td><td>0.330</td></tr><tr><td>Ours</td><td>0.02</td><td>0.321</td><td>0.03</td><td>0.386</td><td>0.03</td><td>0.356</td></tr></table>

Note: SIFID (↓) means lower is better; LPIPS (↑) means higher is better.  
Boldface indicates the best result; underlining indicates the second-best result.

![](images/cffc5df714298d9db4057b6354f173c5c6bcdacc4b0a8730f8c250c2e519131b.jpg)  
Figure 7: Text-guided stylization results using anime-style text prompts.

From the generated results, SinDDM has dificulty in completely preserving the original scene content during some text-guided style transfer processes. SinDDM is a difusion-based single-image generation method and performs semantic guidance during the iterative denoising sampling process. Therefore, some generated results may exhibit local content changes together with changes in visual style. For example, in Fig. 7 and Fig. 9, when trg is set to “Anime Style” and “Picasso Style”, faces or abstract elements that do not exist in the original image are introduced into the sky or mountain regions in some results. In Fig. 8 and Fig. 10, when trg is set to “Van Gogh Style” and “Monet Style”, local semantic content changes can also be observed. The above phenomena indicate that SinDDM may introduce content that does not exist in the original image during text-guided style transfer. In contrast, FRPSS can better preserve the main content of the original image, such as Joshua trees, grasslands, and rocks, while reducing the generation of additional content.

![](images/a1264afef30c1748b157853a09b338b02ab5c34b92225d07b54af1cd255fbb7a.jpg)

Figure 8: Text-guided stylization results using Monet-style text prompts.  
![](images/25a1dbb843732bf95c278c9f439cfc11831259b8c8575202d59c4d8191dc2b64.jpg)  
Figure 9: Text-guided stylization results using Picasso-style text prompts.

![](images/9c5de044001b2be63fdccf54f4f68c2ab9a6f7d742b64ae871a59ff139938cb3.jpg)  
Figure 10: Text-guided stylization results using Van Gogh-style text prompts.

Text-guided content generation. The text-guided image content generation task requires the model not only to change the visual appearance of the image, but also to generate content consistent with the target semantics according to the text prompt. The same input form as text-guided style transfer is adopted for the task. src is still set to $^ { 6 6 } \mathrm { a }$ photo”, while trg is set according to the desired semantic content, including “Matterhorn mountain”, “Sunset”, “A fire in the forest”, “Grand canyon”, and “Oasis”. Fig. 11 shows the generated results of FRPSS and the comparison method under diferent text prompts. When trg is set to “Oasis”, SinDDM fails to form an obvious oasis landscape. In addition,

![](images/b03bc383a0ea2bfdee2b65b0848280db66c67af9c372142ae83bf81c8be4ef3d.jpg)  
Figure 11: Text-guided image content generation results.

when trg is set to “A fire in the forest” and “Sunset”, some generated results of SinDDM mainly show changes in color or local texture and still difer from the semantic content described by the target text. In contrast, FRPSS can produce more obvious target semantic changes.

Image-guided style transfer. The performance of FRPSS on the image-guided style transfer task is further evaluated. Here, $x _ { s r c }$ is set to the content image shown on the left side of Fig. 12, and $x _ { r e f }$ is set to the five style reference images shown in the first row on the right side of Fig. 12, including Van Gogh style, ink landscape painting style, expressionist style, watercolor city style, and golden hall style. The second and third rows on the right side of Fig. 12 show the generated results of GPDM [42] and FRPSS on the task.

![](images/94164a374ec2aa4f07f9eead05bc03c0ba0048d8c3c9892db771deeb0c220682.jpg)  
Figure 12: Image-guided style transfer results.

As shown in Fig. 12, GPDM tends to produce over-smoothing during image-level feature injection, which makes the boundaries of some main structures in the original image blurred. In contrast, FRPSS can better transfer the texture features from the target style image to the original image while preserving the clarity of the main structures in the original image.

Paint-to-image. The goal of the paint-to-image task is to generate images with natural textures and visual details according to the structural layout provided by a paint. FRPSS follows the directional guidance mechanism used in image-guided style transfer. Here, $x _ { s r c }$ is set to the paint shown in the Paint column of Fig. 13, and $x _ { r e f }$ is set to the natural image shown in the Image column.

Image  
Paint  
SinGAN  
SAMDSinGAN  
Ours  
![](images/48635b49699c2660b33371eb6cbc4cee87558039d6541181487c78acf4dcd752.jpg)  
Figure 13: Paint-to-image results.

Baseline methods such as SinGAN and SAMDSinGAN usually inject paint information at specific low-scale stages of the generation process, but the implicit feature fusion tends to make the texture transfer in the generated results relatively rigid. In contrast, FRPSS can progressively add textures and visual details while preserving the main spatial layout of the paint, thereby generating more natural image results.

## 4.5 Outpainting

The goal of the outpainting task is to generate visual content beyond the boundaries of the original image. Following the implementation of MultiOSG [35], during inference, FRPSS first encodes the low-scale image to obtain the feature vector Z of the original image. Subsequently, the MSR-FAGS module generates the left extension feature $Z _ { l e f t } ^ { * }$ and the right extension feature $Z _ { r i g h t } ^ { * } ,$ respectively. The two extension features are then concatenated with the original feature Z along the width W. Since the generator has a fully convolutional structure, its parameters do not depend on a fixed spatial size of the input feature. Therefore, the concatenated extended feature can be directly input into the generator for subsequent generation. During the subsequent multi-scale generation process, bilinear interpolation is still applied to the current outpainting result at each scale according to the original scale ratio to progressively adjust its spatial resolution. As shown in Fig. 14, the generated outpainting regions maintain good continuity with the original image in terms of visual texture and overall scene layout.

![](images/c6f086a3c81c5415d70c6610e58e2c65dca256f43e7bee3422be27cb4f829278.jpg)  
Figure 14: Image outpainting. The yellow boxes indicate the original images.

## 4.6 Parameter influence analysis

The key hyperparameters of the MSR-FAGS module are quantitatively and qualitatively analyzed on the SIGD16 dataset, with a focus on the efects of the feature slice width α and the number of interpolated features $Q$ on generation quality and diversity. The quantitative results are shown in Table 2, and the corresponding qualitative results are shown in Fig. 15 and Fig. 16.

Table 2: Parameter influence analysis of the MSR-FAGS module on the SIGD16 dataset. SIFID (↓) and LPIPS (↑) results under diferent feature slice widths (α) and numbers of interpolated features (Q).
<table><tr><td>Settings</td><td>Configuration</td><td>SIFID ↓ LPIPS ↑</td></tr><tr><td rowspan="4">Varying α  $\left( Q = 2 \right)$ </td><td>α = 2</td><td>0.043 0.125</td></tr><tr><td> $\alpha = 4$ </td><td>0.034 0.244</td></tr><tr><td> $\alpha = 8$ </td><td>0.031 0.317</td></tr><tr><td> $\alpha = 1 6$ </td><td>0.033 0.353</td></tr><tr><td rowspan="4">Varying Q  $( \alpha = 1 6 )$ </td><td> $Q = 0$ </td><td>0.033 0.356</td></tr><tr><td> $Q = 2$ </td><td>0.032 0.353</td></tr><tr><td> $Q = 4$ </td><td>0.032 0.349</td></tr><tr><td> $Q = 8$ </td><td>0.031 0.347</td></tr></table>

Table 2 shows the efects of the feature slice width α on generation fidelity and diversity. When $\alpha = 2$ , the LPIPS value of the model decreases to 0.125, and the SIFID is 0.043. Combined with the qualitative results in Fig. 15, an excessively small α limits the spatial rearrangement range of the features and makes the model tend to reproduce the original image, thereby limiting the diversity of the generated results. As α increases to 16, LPIPS increases to 0.353, while SIFID remains at a relatively low level (0.033). The result indicates that appropriately increasing α can improve the diversity of the generated samples while keeping SIFID at a relatively low level.

With $\alpha = 1 6$ fixed, the efect of the number of interpolated features Q is further examined. When $Q = 0$ , the model achieves the highest LPIPS score (0.356), but the generated results tend to show local stitching artifacts due to the lack of intermediate features generated by interpolation. As the number of interpolated features $Q$ increases to 8, LPIPS slightly decreases to 0.347, while SIFID further decreases to 0.031. The red regions in Fig. 16 indicate that introducing an appropriate amount of feature interpolation helps the model synthesize more natural and coherent semantic features, thereby generating images with both structural plausibility and novelty.

![](images/6ea0ceb9f66fb404686a71605d6e58acd732fef9e5e60d21c8684977ed5ff5c6.jpg)  
Figure 15: Efect of the feature slice width α.

## 4.7 Stylization ablation experiment

The efects of the global directional loss $\mathcal { L } _ { g l o b a l }$ and the local directional loss $\mathcal { L } _ { p a t c h }$ on the text-guided stylization results are analyzed. Here, the original image is the sunset scene image shown on the left side of Fig. 17. src is set to “a photo”, while trg is set to “Rococo Style”, “Turner Style”, “Cubism Style”, “Ink Painting Style”, and “Pop Art Style”, respectively. As shown in Fig. 17, when only $\mathcal { L } _ { g l o b a l }$ is used, the generated results mainly show changes in the overall color tone, while the local style textures are relatively weak. When only $\mathcal { L } _ { p a t c h }$ is used, more obvious local textures can be introduced into the generated results, but the overall visual consistency is relatively insuficient. When the two losses are jointly used, both global style changes and local texture generation can be taken into account.

## 4.8 Limitations

Although FRPSS achieves good generation results on various single-image generation tasks, it still has certain limitations in some complex scenes. As shown in Fig. 18, for images such as Chinese landscape paintings and murals that contain large background regions and sparse semantic objects, the generated results may exhibit local content missing. For example, small semantic regions such as thatched cottages and human faces in the original image are not completely preserved in some generated results. The phenomenon may be related to the structural rearrangement of MSR-FAGS in the lowscale feature space. In low-scale representations, large background regions occupy a relatively high spatial proportion, while local objects such as thatched cottages and human faces are represented by only a small number of feature regions. During feature rearrangement and matching, the sparse semantic regions may be afected by background features with larger proportions, thereby weakening their structural information and causing local content missing. In addition, the current stylization mechanism of FRPSS directly introduces the target semantic direction into the training process of the multiscale generative model, resulting in a certain degree of coupling between the generative model and the specific target style. When the target style or guidance condition changes, the model usually needs to be retrained. Therefore, switching among multiple styles still incurs a certain training overhead.

“Cubism Style"  
“Rococo Style"  
![](images/ce7aa1224a7e81df11fd8f96d27c450250665245a642ad74e66e704928ff88a7.jpg)  
Figure 16: Efect of the number of interpolated features Q. The red boxes show bird images with obvious variations generated by FRPSS.

“Turner Style”  
“Ink Painting Style"  
"Pop Art Style"  
![](images/2f2b60ba4de2c93f5eaec90bbf12fa182898ca172c87bdd7c67bbf361c077470.jpg)  
Figure 17: Efects of the global and patch-wise CLIP losses on the stylization results.

![](images/72226813cf572386aa9f57db697d71c22aaa652228b63ee00d400f2d6881565b.jpg)  
Figure 18: Failure cases of FRPSS.

## 5 Conclusion

Existing single-image generation methods usually lack explicit global structural constraints and are prone to structural misalignment or incoherent results during generation. To address the problem, the paper proposes the FRPSS single-image generation framework. FRPSS uses the MSR-FAGS module to rearrange, augment, and match the latent features of a single training image in the Pre-Shape Space, providing latent features with structural variations for the subsequent multi-scale generation process and thereby reducing the risk of structural misalignment during generation. In addition, the CLIP-SSPE module achieves controllable semantic guidance through global and patch-wise directional CLIP constraints. Experimental results show that FRPSS achieves good generation fidelity on three single-image datasets and realizes an efective trade-of between fidelity and diversity. Qualitative experiments further verify its efectiveness on downstream tasks with CLIP-SSPE, including text-guided style transfer, text-guided content generation, image-guided style transfer, and paint-to-image.

FRPSS still has certain limitations. On the one hand, in complex scenes containing large background regions and sparse semantic objects, low-scale feature rearrangement may weaken the structural information of local semantic regions. On the other hand, the current stylization mechanism directly introduces the target semantic direction into the training process of the multi-scale model, resulting in a certain degree of coupling between the generative model and the target style. When the target style or guidance condition changes, the model usually needs to be retrained, thereby limiting the flexibility and application eficiency of the model in scenarios requiring rapid switching among multiple styles. In the future, region importance modeling or attention mechanisms can be further introduced into MSR-FAGS to enhance the preservation of sparse semantic regions. Meanwhile, conditional injection mechanisms that decouple semantic control from the training process of the generative model can be explored, allowing the model to perform style control according to diferent texts or reference images during inference and thereby reducing the repeated training overhead for diferent target styles.

## References

[1] Ian J Goodfellow et al. “Generative adversarial nets”. In: Advances in neural information processing systems 27 (2014).

[2] Tero Karras et al. “Analyzing and improving the image quality of stylegan”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020, pp. 8110–8119.

[3] Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising difusion probabilistic models”. In: Advances in neural information processing systems 33 (2020), pp. 6840– 6851.

[4] Robin Rombach et al. “High-resolution image synthesis with latent difusion models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 10684–10695.

[5] Erfan Azqadan, Hamid Jahed, and Arash Arami. “Predictive microstructure image generation using denoising difusion probabilistic models”. In: Acta Materialia 261 (2023), p. 119406.

[6] Teerath Kumar et al. “Image Data Augmentation Approaches: A Comprehensive Survey and Future Directions”. In: IEEE Access 12 (2024), pp. 187536–187571. doi: 10.1109/ACCESS.2024.3470122.

[7] Connor Shorten and Taghi M Khoshgoftaar. “A survey on image data augmentation for deep learning”. In: Journal of big data 6.1 (2019), pp. 1–48.

[8] Tamar Rott Shaham, Tali Dekel, and Tomer Michaeli. “SinGAN: Learning a Generative Model From a Single Natural Image”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). Oct. 2019.

[9] Tobias Hinz et al. “Improved Techniques for Training Single-Image GANs”. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). Feb. 2021, pp. 1300–1309.

[10] Shir Gur, Sagie Benaim, and Lior Wolf. “Hierarchical Patch VAE-GAN: Generating Diverse Videos from a Single Sample”. In: Advances in Neural Information Processing Systems. Ed. by H. Larochelle et al. Vol. 33. Curran Associates, Inc., 2020, pp. 16761–16772. url: https://proceedings.neurips.cc/paper\_files/ paper/2020/file/c2f32522a84d5e6357e6abac087f1b0b-Paper.pdf.

[11] Weilun Wang et al. “SinDifusion: Learning a Difusion Model From a Single Natural Image”. In: IEEE transactions on pattern analysis and machine intelligence 47.5 (2025), pp. 3412–3423.

[12] Yinxi He et al. “StructDif: A Structure-Preserving and Spatially Controllable Diffusion Model for Single-Image Generation”. In: arXiv preprint arXiv:2604.12575 (2026).

[13] Vladimir Kulikov et al. “Sinddm: A single image denoising difusion model”. In: International conference on machine learning. PMLR. 2023, pp. 17920–17930.

[14] Narek Tumanyan et al. “Splicing vit features for semantic appearance transfer”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 10748–10757.

[15] Or Patashnik et al. “StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery”. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society. 2021, pp. 2065–2074.

[16] Gihyun Kwon and Jong Chul Ye. “Clipstyler: Image style transfer with a single text condition”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 18062–18071.

[17] Martha Paskin et al. “A Kendall shape space approach to 3D shape estimation from 2D landmarks”. In: European conference on computer vision. Springer. 2022, pp. 363–379.

[18] David G Kendall. “Shape manifolds, procrustean metrics, and complex projective spaces”. In: Bulletin of the London mathematical society 16.2 (1984), pp. 81–121.

[19] Tom Tirer et al. “Deep internal learning: Deep learning from a single input”. In: IEEE Signal Processing Magazine 41.4 (2024), pp. 40–57.

[20] Zicheng Zhang, Congying Han, and Tiande Guo. “ExSinGAN: Learning an Explainable Generative Model from a Single Image”. In: 32nd British Machine Vision Conference 2021, BMVC 2021, Online, November 22-25, 2021. BMVA Press, 2021, 251. url: https://www.bmvc2021- virtualconference.com/assets/papers/ 0348.pdf.

[21] Yaniv Nikankin, Niv Haim, and Michal Irani. “SinFusion: Training Difusion Models on a Single Image or Video”. In: International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA. Ed. by Andreas Krause et al. Proceedings of Machine Learning Research. PMLR, 2023, pp. 26199–26214. url: https://proceedings.mlr.press/v202/nikankin23a.html.

[22] Zicheng Zhang et al. “Petsgan: Rethinking priors for single image generation”. In: Proceedings of the AAAI conference on artificial intelligence. Vol. 36. 3. 2022, pp. 3408–3416.

[23] Jinshu Chen et al. “Mogan: Morphologic-structure-aware generative learning from a single image”. In: IEEE Transactions on Systems, Man, and Cybernetics: Systems 54.4 (2023), pp. 2021–2033.

[24] Eyyup Yildiz, Mehmet Erkan Yuksel, and Selcuk Sevgen. “A single-image GAN model using self-attention mechanism and DenseNets”. In: Neurocomputing 596 (2024), p. 127873.

[25] Haojun Qiu, Kiriakos N. Kutulakos, and David B. Lindell. “Eficient and Training-Free Single-Image Difusion Models”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2026.

[26] Yingli Hou et al. “Single image and video generation using a receptive difusion model with convolutional spatiotemporal blocks”. In: Applied Soft Computing (2025), p. 113509.

[27] Ayushi Verma, Tapas Badal, and Abhay Bansal. “A novel framework for diverse video generation from a single video using frame-conditioned denoising difusion probabilistic model and ConvNeXt-V2”. In: Image and Vision Computing 154 (2025), p. 105422.

[28] Alec Radford et al. “Learning transferable visual models from natural language supervision”. In: International conference on machine learning. PmLR. 2021, pp. 8748– 8763.

[29] Rinon Gal et al. “Stylegan-nada: Clip-guided domain adaptation of image generators”. In: ACM Transactions on Graphics (TOG) 41.4 (2022), pp. 1–13.

[30] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. “Difusionclip: Text-guided difusion models for robust image manipulation”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 2426–2435.

[31] Yuexing Han et al. “Recognition of multiple configurations of objects with limited data”. In: Pattern Recognition 43.4 (2010), pp. 1467–1475.

[32] Yuexing Han, Hideki Koike, and Masanori Idesawa. “Recognizing objects with multiple configurations”. In: Pattern Analysis and Applications 17.1 (2014), pp. 195– 209.

[33] Irina Higgins et al. “beta-vae: Learning basic visual concepts with a constrained variational framework”. In: International conference on learning representations. 2017.

[34] Ishaan Gulrajani et al. “Improved training of wasserstein gans”. In: Advances in neural information processing systems 30 (2017).

[35] Yao Gou et al. “Multiple one-shot image generation via deep structure reshufle”. In: Neural Networks (2025), p. 107862.

[36] Niv Granot et al. “Drop the gan: In defense of patches nearest neighbors as single image generative models”. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, pp. 13460–13469.

[37] Haofan Wang et al. “InstantStyle-Plus: Style Transfer with Content-Preserving in Text-to-Image Generation”. In: CoRR abs/2407.00788 (2024). doi: 10 . 48550 / ARXIV.2407.00788. arXiv: 2407.00788. url: https://doi.org/10.48550/ arXiv.2407.00788.

[38] Richard Zhang et al. “The unreasonable efectiveness of deep features as a perceptual metric”. In: 2018 IEEE/CVF conference on computer vision and pattern recognition. IEEE. 2018, pp. 586–595.

[39] Martin Heusel et al. “Gans trained by a two time-scale update rule converge to a local nash equilibrium”. In: Advances in neural information processing systems 30 (2017).

[40] Christian Szegedy et al. “Rethinking the inception architecture for computer vision”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016, pp. 2818–2826.

[41] Alex Krizhevsky, Ilya Sutskever, and Geofrey E Hinton. “Imagenet classification with deep convolutional neural networks”. In: Advances in neural information processing systems 25 (2012).

[42] Ariel Elnekave and Yair Weiss. “Generating natural images with direct patch distributions matching”. In: Proceedings of the European Conference on Computer Vision (ECCV). 2022.