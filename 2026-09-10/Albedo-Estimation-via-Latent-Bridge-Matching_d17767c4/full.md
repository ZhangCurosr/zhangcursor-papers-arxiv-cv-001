# Albedo Estimation via Latent Bridge Matching

Carme Corbi, David Serrano-Lozano, Javier Vazquez-Corral, Maria Vanrell

Universitat Autonoma de Barcelona and Computer Vision Center\`

## Abstract

Recent advances in Intrinsic Image Decomposition (IID) have increasingly relied on generative models. However, progress remains limited by three key challenges: (a) insufficient physical consistency, (b) high computational cost at inference time, and (c) limited generalization capabilities. In this work, we show that latent bridge matching (LBM) effectively addresses these limitations for albedo estimation. We introduce a novel LBM-based architecture that enforces physical consistency through a pixel reconstruction loss, benefits from the inherent efficiency of LBM low-cost inference, and improves generalization across diverse datasets by incorporating a shading conditioning. In this extended version, we additionally show that conditioning the shading estimator itself on the predicted albedo further improves reconstruction fidelity, and we benchmark our best model against stateof-the-art IID methods across five real and synthetic datasets. https://github.com/CVC-Color/albedoLBM

## Introduction

Understanding the interaction between light and surfaces of the scene from a single image is a fundamental problem in computer vision. Although humans can unconsciously discount illumination changes and infer material properties, replicating this perceptual ability in computational systems remains a highly challenging task.

Intrinsic Image Decomposition (IID) aims to separate an image into its underlying physical components. Given a single RGB image I, IID seeks to recover two factorized images: reflectance or albedo, A, which represents the inherent color properties of surfaces; and shading S, which captures the effects of the interaction between light and scene geometry. The most basic formulation, as originally proposed by Barrow and Tenenbaum [1] is given by

$$
I ( x , y ) = A ( x , y ) \cdot S ( x , y )\tag{1}
$$

where (x,y) are pixel coordinates and · denotes pixel-wise product. It is based on a simplified Lambertian assumption that is frequently violated; as a consequence, most real-world images cannot be accurately explained using only these two components. Several works [2, 3] have proposed extending the basic model explicitly to account for non-Lambertian effects:

$$
I ( x , y ) = A ( x , y ) \cdot S ( x , y ) + R ( x , y )\tag{2}
$$

where $R ( x , y )$ captures residual effects including others global illumination phenomena and non-diffuse material components. In summary, IID is a severely ill-posed problem, as infinite pairs (A, S) or triplets (A, S,R) can explain the same observed image.

The estimation of intrinsic components has been addressed using a variety of techniques over the years. Recently, the primary focus has shifted toward generative frameworks. Despite their success, these approaches exhibit several notable drawbacks. First, they often lack sufficient physical consistency, likely due to the difficulty of effectively incorporating pixel-level physical constraints during training. Second, they tend to produce residual noise as a consequence of the generative process. Third, they incur a relatively high computational cost at inference time. Finally, all IID methods frequently demonstrate limited generalization capabilities, which may stem from their dependency on large-scale synthetic datasets required for training.

In this work, we hypothesize that LBM, a recent generative framework, can help address some of these limitations and provide a promising new approach for IID. Based on this hypothesis, and after presenting a brief overview of existing generative frameworks, we propose a pipeline that leverages the ability of LBM to be trained with a pixel-wise reconstruction loss while naturally incorporating constraints from the input image and additional cues. We then design a set of experiments to evaluate the proposed approach, leading to a novel method. We summarize the following contributions:

• Demonstrating that LBM constitutes an effective generative framework for IID, achieving improvements in both compu tational efficiency (up to 75% reduction in inference time) and accuracy (from 3 to 6 PSNR units depending on dataset) compared to other generative approaches.

• Exploiting the pixel-level nature of LBM to introduce a reconstruction loss that encourages physical consistency, promoting adherence of the estimated albedo to the underlying image formation model (reconstruction error is reduced to 50%)

• Investigating different conditioning mechanisms to enrich the generation process with additional contextual information, showing that shading outperforms surface normalbased alternatives.

• Training a complementary shading estimation model as an efficient conditioning module to guide albedo estimation, resulting in a physically grounded two-stage intrinsic decomposition pipeline.

## Related works

Classical approaches to IID predate the current generative wave and largely fall into three families. The first builds on Retinex theory [4], which assumes that reflectance is piece-wise constant while shading varies smoothly, attributing large image gradients to reflectance changes and small ones to illumination [1]; later variants extended this idea to color by exploiting shading-invariant chromaticity cues [5], amongst others. The second family casts decomposition as an energy-minimization problem, imposing hand-crafted priors on reflectance and shading (e.g., sparsity of reflectance, smoothness of shading, and local color constancy) and solving it through optimization [6, 7, 8]. A key advantage of these optimization-based methods is that they require no labeled training data, but they rely on assumptions that are frequently violated in real scenes and tend to generalize poorly to complex, non-Lambertian content—limitations that motivated the shift toward learning-based and, more recently, generative formulations. Therefore, the third family is the learning-based that comes in parallel with the advent of large-scale datasets and deep learning techniques that implicitly learn intrinsic regularities from example. They are based on an encoder mapping the input image into a compact yet expressive latent representation, which is then decoded into one or more output images, here we highlight latest ones, such as, PIE-Net [9], Zhu et-al [10] and Careaga etal [11]. The IID ill-posed nature makes that classical physical assumptions remain valuable through tailored loss functions or explicitly via regularization terms [12, 13]. Despite these advances, learning-based approaches remain sensitive to dataset bias, which limit their ability to robustly generalize to real-world scenes.

Generative approaches. Recently, generative models have taken over most low-level vision tasks, including IID. GAN-based methods cast the decomposition as a layer-separation problem, leveraging adversarial losses together with cycle-consistency and self-supervision constraints to recover albedo and shading without paired ground truth [14, 15]. More recently, diffusion models have become the dominant paradigm: by repurposing the strong priors of large-scale pretrained text-to-image generators, such as RGB↔X [16], IntrinsicDiffusion [17], Marigold [18], PRISM [19], and ReasonX [20] which achieve high-quality, generalizable decompositions, often framing IID as a conditional generation task. However, these approaches inherit the high inference cost and residual stochasticity of the generative sampling process.

IID datasets. A central obstacle in IID is the difficulty of obtaining ground-truth decompositions, which has shaped the available benchmarks. Existing datasets fall broadly into three categories, reflecting different trade-offs between realism, scale, and annotation quality. Real-world controlled datasets capture scenes under laboratory conditions with known illuminations; the MIT Intrinsic Images dataset [21] is the primary example, containing 220 images of 20 objects under 11 directional lights, with ground-truth shading obtained by spray-painting the objects gray. Its controlled setting makes decomposition tractable but limits generalization due to its small, object-centric scope. Synthetic datasets instead use physically based rendering to obtain dense, pixel-accurate annotations at scale: InteriorVerse [10] provides over 50K indoor renderings with reflectance, shading, and surface normal maps, Hypersim [22] offers more than 77K images across 461 cluttered indoor scenes with an explicit residual term for non-Lambertian effects (Eq. 2), and ARAP [23] contributes 149 photorealistic images rendered with LuxRender. All synthetic sources, however, share a sim-to-real gap that can hinder generalization to real images. Finally, real-world in-the-wild datasets trade dense supervision for scene diversity: Intrinsic Images in the Wild (IIW) [8] addresses this through nearly a million crowdsourced relative reflectance judgments over 5,230 indoor images, providing sparse supervision via the WHDR metric rather than dense ground truth. As a result, current methods do not agree on a common protocol: they are trained on different combinations of these datasets and evaluated with disparate, dataset-specific metrics, making fair comparison between approaches difficult – a point we return to in Section .

## Background

Diffusion models are generative models that learn to reverse a gradual noising process: starting from pure Gaussian noise, they iteratively denoise a sample by predicting, at each step, the noise added to a clean image [24]. Usually this iterative process is performed within a learned latent space, reducing dimensionality. While powerful, their many-step sampling makes inference slow. Flow Matching (FM) [25] reframes generation as learning a continuous velocity field $\nu _ { \theta }$ that transports samples from a simple prior to the data distribution. Given the linear interpolant $x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 }$ between a prior sample $x _ { 0 } \sim \mathcal { N } ( 0 , I )$ and a data sample $x _ { 1 }$ , the field is trained by regressing onto the constant target velocity:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } _ { t , x _ { 0 } , x _ { 1 } } \left[ \Vert \nu _ { \theta } ( x _ { t } , t ) - ( x _ { 1 } - x _ { 0 } ) \Vert ^ { 2 } \right] .\tag{3}
$$

Sampling then integrates $\nu _ { \theta }$ along straighter paths, so inference is not tied to a fixed noise schedule and requires fewer iterations [25].

For completeness, we note that the diffusion formulation above corresponds to a discrete-time Markov chain in which Gaussian noise is progressively added to a clean latent $z _ { 0 }$ over timesteps $t \in \{ 0 , 1 , \ldots , T \}$ , giving $z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \varepsilon ,$ , where $\bar { \alpha } _ { t }$ is the cumulative product of a predefined noise schedule. A network $\varepsilon _ { \boldsymbol { \theta } } ( z _ { t } , t )$ is trained to predict the injected noise, and generation proceeds by iteratively reversing this chain from $z _ { T } \sim \mathcal { N } ( 0 , I )$ down $\mathrm { t o } \ z _ { 0 } .$ . Flow Matching replaces this stochastic, many-step chain with the deterministic ODE trajectory of Eq. 3, which is what enables inference in far fewer steps.

However, a key limitation of both formulations for image-toimage tasks is that the trajectory always originates from a fixed Gaussian prior. The source image can only enter as a conditioning signal rather than as the starting point of the transport. This has two consequences. First, the output is never anchored to the observed pixels, making it hard to impose physical constraints on the generated result. Second, the stochastic starting point introduces residual noise and variability in the prediction.

Recently, LBM removes this constraint by building a stochastic interpolant directly between the source distribution π and the target distribution $\pi _ { 1 } .$ , learning a transport map between two arbitrary distributions instead of from noise [26]. In our case, π<sub>0</sub> corresponds to the distribution of natural images and $\pi _ { 1 }$ to the distribution of albedo maps. Given a paired sample $( x _ { 0 } , x _ { 1 } ) \sim \pi _ { 0 } \times \pi _ { 1 }$ where it corresponds to an image and its albedo, the bridge is de fined as

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } + \sigma \sqrt { t ( 1 - t ) } \varepsilon , \qquad \varepsilon \sim \mathcal { N } ( 0 , I ) ,\tag{4}
$$

and a network is trained to predict the corresponding drift, recovering $x _ { 1 }$ from $x _ { 0 }$ in as little as a single step. LBM applies this idea in the latent space of a pretrained autoencoder, achieving fast, even single-step, image-to-image translation across tasks [26, 27, 28]. This makes LBM particularly well suited to IID: because the trajectory begins at the source image itself, the prediction remains anchored to the observed pixels. As a result, we can decode the estimated target back to pixel space and impose an explicit reconstruction loss enforcing the image formation model, directly grounding the decomposition in physical constraints, which is precisely the property that diffusion and FM lack.

It is worth noting that the three formulations above form a hierarchy rather than three unrelated frameworks. Setting $\sigma = 0$ and replacing x<sub>0</sub> with a Gaussian sample reduces the bridge of Eq. 4 to the deterministic FM interpolant, and further replacing the deterministic transport with the discrete noising chain recovers standard diffusion. In our experiments (Section ) we exploit this relationship directly: the same UNet backbone and training pipeline are reused across all three variants, isolating the effect of the transport formulation itself, independent of architecture or capacity, on albedo estimation quality.

## Method

Our method addresses albedo estimation through a generative formulation built upon Latent Bridge Matching (LBM). We first describe how LBM is adapted to the albedo estimation task, then detail the latent bridge formulation that defines the transport process, the reconstruction loss that grounds the decomposition in the image formation model, and finally the shading-conditioned variant that further disentangles illumination from reflectance. In Figure 1 a global outline of the method is given.

Albedo estimation. We cast albedo estimation as an image-toimage translation between two domains: the observed RGB image and the corresponding ground-truth albedo, which is by definition independent of illumination effects such as shading, shadows, and specularities (Eq. 1). As illustrated in Fig. 1, the source image $x _ { 0 } \sim \pi _ { 0 }$ is the observed RGB input and the target $x _ { 1 } \sim \pi _ { 1 }$ is the ground-truth albedo. Both are mapped into a compressed latent space through the frozen encoder $\mathcal { E } ^ { \mathcal { C } }$ of a pretrained variational autoencoder (VAE), producing the latent representations $z _ { 0 } = \mathcal { E } ( x _ { 0 } )$ and $z _ { 1 } = \mathcal { E } ( x _ { 1 } )$ . The transport between $z _ { 0 }$ and $z _ { 1 }$ is modeled by a trainable drift network, and the predicted target latent $\hat { z } _ { 1 }$ is decoded back to pixel space through the frozen VAE decoder $\mathcal { D }$ to obtain the estimated albedo $\hat { x } _ { 1 } = \mathcal { D } ( \hat { z } _ { 1 } )$ . Crucially, because the trajectory starts from the input image rather than from Gaussian noise, the prediction remains anchored to the observed pixels throughout the process.

Latent Bridge Matching formulation. With the source and target latents z<sub>0</sub> and z<sub>1</sub> defined above, we instantiate the bridge of Eq. 4 directly in latent space, where $\sigma \ge 0$ controls the variance of the transport path and $t \sim \pi ( t )$ . In contrast to the continuous uniform schedule used in diffusion training, we sample t from a small set of equally spaced timesteps. This aligns the timesteps seen at training and inference and caps sampling at four steps, which is the primary source of LBM’s efficiency over conventional diffusion models.

The interpolated latent $z _ { t }$ is passed to a drift network $\nu _ { \theta } ( z _ { t } , t )$ that predicts the velocity pointing toward the clean target latent, from which the target is recovered in closed form,

$$
\hat { z } _ { 1 } = ( 1 - t ) \nu _ { \theta } ( z _ { t } , t ) + z _ { t } .\tag{5}
$$

The network is trained with the bridge-matching regression objective

$$
\mathcal { L } _ { \mathrm { L B M } } = \mathbb { E } _ { t , z _ { t } } \left[ \left| \left| \nu _ { \theta } ( z _ { t } , t ) - \nu ( z _ { t } , t ) \right| \right| ^ { 2 } \right] ,\tag{6}
$$

where $\nu ( z _ { t } , t )$ is the target drift induced by Eq. 5. We take $\mathcal { E } ^ { \mathcal { C } }$ and D to be the frozen VAE of Stable Diffusion XL and implement $\nu _ { \theta }$ as a UNet initialized from the same pretrained backbone.

Reconstruction Loss. The latent objective in Eq. 6 supervises the transport in latent space, but does not guarantee that the decoded albedo is consistent with the input image under the intrinsic image formation model. Because LBM operates on a pixelaligned trajectory, we can decode the predicted latent and impose supervision directly in pixel space. We therefore augment the objective with two pixel-level terms: a supervised term ${ \mathcal L } _ { \mathrm { p i x e l } }$ comparing the decoded prediction $\hat { x } _ { 1 }$ against the ground-truth albedo $x _ { 1 } .$ , and a reconstruction term ${ \mathcal { L } } _ { \mathrm { r e c o n } }$ that enforces physical consistency with the image formation model. The full training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L B M } } + \lambda \left( \alpha \mathcal { L } _ { \mathrm { p i x e l } } + \left( 1 - \alpha \right) \mathcal { L } _ { \mathrm { r e c o n } } \right) ,\tag{7}
$$

where $\lambda$ weights the pixel-space supervision against the latent objective and $\alpha \in [ 0 , 1 ]$ balances the supervised and reconstruction terms. For ${ \mathcal { L } } _ { \mathrm { p i x e l } }$ we consider the perceptual LPIPS loss. The reconstruction term is defined as the distance between the input image I and the image re-synthesized from the estimated albedo,

$$
\mathcal { L } _ { \mathrm { { r e c o n } } } = \mathcal { L } \big ( I , \hat { A } \cdot S \big ) \quad \mathrm { o r } \quad \mathcal { L } \big ( I , \hat { A } \cdot S + R \big ) ,\tag{8}
$$

where $\hat { A }$ is the estimated albedo $( \mathrm { i } . \mathrm { e } . \ \hat { x } _ { 1 } ) .$ , S the shading, and R a residual term capturing non-Lambertian effects. The choice between the two forms depends on the dataset: the Lambertian product model (Eq. 1) or its residual extension. This ability to attach a physically motivated, pixel-level supervision signal is precisely what makes LBM well suited to IID, where the image formation model operates directly at the pixel level.

Conditioning Albedo Estimation The LBM framework naturally admits a conditional extension. In addition to the paired samples $\left( { { x } _ { 0 } } , { { x } _ { 1 } } \right)$ , an auxiliary conditioning image $x _ { c }$ is introduced to guide the transport. As shown in Fig. 1, $x _ { c }$ is encoded with the frozen VAE into a latent $z _ { c }$ and concatenated with the interpolated latent $z _ { t }$ along the channel dimension, yielding the joint input $[ z _ { t } , z _ { c } ]$ to the drift network. This lets $\nu _ { \theta }$ predict the drift while explicitly accounting for the additional cue, improving controllability and reducing ambiguity in the estimated albedo.

To condition the albedo estimation process, we explore several alternatives based on complementary intrinsic components, namely surface normals—proven useful for related problems such as lighting ambient normalization [29]—and shading. As a first approach, we employ a foundation model for normal estimation based on Marigold [18] . Alternatively, we train two LBM-based models following the architecture shown in Figure 1: one for surface normal estimation and another for shading estimation. In Experiment 3, we evaluate these alternatives; our results indicate that shading-based conditioning provides the most effective guidance for albedo estimation among the alternatives considered. Providing the network with shading supplies an explicit signal indicating which intensity variations in the input are attributable to illumination, allowing the model to focus its capacity on recovering the residual reflectance. This improves the disentanglement between illumination and surface color and reduces the tendency to bake soft shadows and inter-reflections into the predicted albedo. To make this conditioning usable at inference, when ground-truth shading is unavailable, we pair it with a complementary shading estimation model that acts as an efficient conditioner, yielding a physically grounded two-stage decomposition pipeline. We note that this design introduces a two-stage dependency: the final albedo quality is inherently bounded by the accuracy of the shading estimator itself, since any error in the predicted shading is propagated as conditioning noise to the albedo model. We discuss this dependency, and a partial mitigation, further in Experiment 3.

![](images/e406df089bee09a6ac34d1716daa707c498f0cd11b12a40a03dfbbed5386f32f.jpg)  
Figure 1. Method Overview. Input image x , ground-truth albedo x , and optional conditioning x are mapped to the latent space through a frozen VAE encoder. The latent z and z are combined via a stochastic interpolant to produce z , which is concatenated with the conditioning z and fed to a Stable Diffusion XL UNet. The UNet predicts the velocity field $\nu _ { \theta } \left( z _ { t } , t \right) .$ , from which the estimated target latent zˆ is recovered and decoded by a frozen VAE decoder into the predicted albedo xˆ . Training combines the LBM objective $\mathcal { L } _ { L B M }$ on the velocity field with a pixel-space term $\mathcal { L } _ { p i x e l }$ and a reconstruction term $\mathcal { L } _ { r e c o n }$ enforcing the image formation model.

## Experiments and Results

In this section we propose a set of experiments to evaluate the initial hypothesis and the proposed methods.

## Experimental setup

Models are trained on InteriorVerse (≈ 44K images) and Hypersim (≈ 59K images) datasets for 50 epochs. InteriorVerse provides ground-truth albedo and surface normal maps, shading is computed using image formation model defined in equations 1 and 7 Hypersim provides ground-truth albedo, shading, surface normals, and an additional residual term capturing non-Lambertian effects.

The encoder E and decoder D are taken from the pre-trained variational autoencoder (VAE) of Stable Diffusion XL [30] and are kept frozen during training to preserve the learned latent representation. The parametrized drift function v is implemented as a U-Net, initialized from the pre-trained text-to-image Stable Diffusion XL model. The model is optimized using the AdamW optimizer with a learning rate $0 \mathrm { f } 4 \times 1 0 ^ { - 5 }$

To ensure compatibility with the pre-trained VAE, all input images and intrinsic properties are normalized to the range $[ - 1 , 1 ] .$ During training, we apply random cropping to 256 × 256 patches without padding. At inference time, all images are processed at their original resolution without any cropping or resizing, allowing the model to generalize to arbitrary input dimensions.

## Evaluation datasets and metrics

We evaluate on five benchmarks spanning real controlled, synthetic, and real in-the-wild scenarios: MIT Intrinsic Images (220 images), IIW (1,046 test images), ARAP (149 images), and the InteriorVerse (2,672 test images) and Hypersim (7,690 test images) held-out splits. Where prior work provides evaluation masks we follow the established protocol for each dataset; Hypersim, which has no mask annotations, is evaluated on the full image. Because no single dataset provides every ground-truth component alongside a consistent metric set, we report the following measures, matching the protocol used in prior IID literature for each benchmark.

MSE is the mean squared error between the predicted albedo $\hat { A }$ and the ground truth A over the N evaluated pixels, MSE = $\begin{array} { r } { \frac { 1 } { N } \sum _ { i } ( \hat { A } _ { i } - A _ { i } ) ^ { 2 } } \end{array}$ . LMSE (Local MSE) computes this same quantity within overlapping local windows $\Omega _ { k }$ before averaging, which reduces sensitivity to global scale ambiguities inherent to albedo estimation. PSNR is derived from MSE as PSNR = $1 0 \log _ { 1 0 } ( \mathrm { M A X } ^ { 2 } / \mathrm { M S E } )$ DSSIM is the structural dissimilarity, $\mathrm { D S S I M } = ( 1 - \mathrm { S S I M } ) / 2 .$ , where SSIM compares luminance, contrast, and structure between the two images. LPIPS measures perceptual similarity using weighted differences between deep features extracted from a pretrained network across layers l and spatial locations. WHDR (Weighted Human Disagreement Rate), used for IIW, is the weighted fraction of pairwise human judgments about relative reflectance that the predicted albedo contradicts, thresholded at $\delta ;$ we report it at the two thresholds standard in the literature, 10% and 20%. For all error-based metrics (↓) lower is better, while PSNR (↑) is better when higher.

## Experiment 1. Baseline

In this first experiment, we evaluated our initial hypothesis claiming LBM as a suitable framework to improve inference time efficiency and albedo estimation accuracy compared to previous generative formulations. In Table 1 we show the results for three different generative architectures trained for albedo estimation. These are SD-AID (Stable Diffusion for Albedo Intrinsic Decom-

![](images/cf2ba954b6c835235aac55afb8364a56c23d4dbb41056e6393a95874c27ef22e.jpg)

![](images/bba3d5e4a77ee42d827a84fb7966fb3ec6c60644113de97c65bd9a1fae50ebf3.jpg)

![](images/12c21addaa6840070939b6da64cedd0c7c95b780ae59c22e0dbd4f4e674ad3d4.jpg)

![](images/ae07163186276b2cb9ce3d335bf34540e69c5d0914b6439eb8418addb45fe28a.jpg)

![](images/fdfc88992d356ba2211a1648d41199693d65413d7e468bf1fed9279f1c69c91c.jpg)

![](images/595b9094ecf84d1baa8ee0a9cc515d485a82c171366e4ad7dfd5c130f156cea2.jpg)

![](images/b74289a7d58938bad0f75f7d412bd14e55cf363857cc8455109dc25f7f194d5e.jpg)

![](images/431b889eefa37fcda209014e04217defebe21c951fdcf7dd5b80f5fe9f81e63a.jpg)

![](images/e6123eb7f674af80ca64811e86cd55d7dc3763778aa108a6e340cf5736d4967d.jpg)

![](images/a39aaa8f50960543e45520d83393f7b712aa464f9c90b595e196d35fe2364553.jpg)

![](images/8681833465298c82e3d169c5e3fb1d6b4082b1a04dd46ab7ab5d2c34f148bc0c.jpg)

![](images/2995d3626aadb6b74ede45be57c3c72b115bba25c443d2263338d3d2595e3812.jpg)

![](images/8ea4611a1dfc7902628d4c18ee4b71ea3ac3c2c1aa52242b24160196fb617c50.jpg)

![](images/1fb2cdcc17e7cf33f070904a29af5da87bce8acfcb4f4b171a17cfef9321bd1e.jpg)  
Input

![](images/4a3dffede2c01a18eab12f77b43adc0da6c8d807e28bedd7157627107ee3fd35.jpg)

![](images/0a1aa9793031b5522a81075f0a748dd3bb7678d4846545b8f1621ee680787b05.jpg)

SD-AID  
![](images/184ee0b18cdcb645da18ebc48d74f9dd9dddc69b75aa97c171a3ed244352c902.jpg)  
FD-AID

![](images/63ed8d69bbec5188ad85f4dab82a3543c0c5b0ed2aadacc64f3f6a727bac0faf.jpg)

![](images/82c6e07388ae31b5414dfd12ae36dbc80a4b035d476cdfa7b5e7d6f8d2bf5df6.jpg)  
LBM-AID

![](images/042a2f26d5ae80bafcf4d7da46b7a341fafa3d2215a895642fe00023a9373662.jpg)  
GT Albedo

Figure 2. Qualitative comparison across MIT, ARAP, InteriorVerse and Hypersim datasets. From left to right: input image, estimated albedo for SD-AID, FD-AID and LBM-AID and ground-truth albedo.

position), likewise FM-AID and LBM-AID denote FM and LBM in the generative process. All three methods share a similar backbone architecture and are trained under identical conditions as described above. We observe that the LBM-AID model achieves the strongest performance for albedo estimation in most settings and the inference time is equally reduced by LBM and FM, compared to the inefficiency of the SD approach. Some albedo estimation results are qualitatively shown in Figure 2 for different datasets.

## Experiment 2. Setting LBM parameters

The second experiment is devoted to analysing the impact of the reconstruction loss term introduced in Equation 8. We examine the influence of the weighting coefficient α, which balances the contribution between the supervised pixel loss and the reconstruction-based consistency term. Table 2 makes us decide α = 0.9 as the best tradeoff for all datasets and will be used in the subsequent experiments.

## Experiment 3. Model Conditioning

As discussed earlier, an important advantage of LBM is the ease with which additional image-based information can be incorporated to condition the albedo estimation process. In this experiment, we explore two conditioning strategies based on the estimation of complementary intrinsic scene components, namely surface normals and image shading.

For surface normal estimation, we consider two alternatives: (a) using the foundation model proposed by Marigold et al. [18], and (b) training an LBM-based model for normal estimation following the same framework as in LBM-AID (we denote as LBM-NID for Normal Intrinsic Decomposition). In addition, we employ this same framework to train a shading estimation model (we denote it as LBM-SID for Shading Intrinsic Decomposition), which serves as a third conditioning option.

Results are summarized in Table 3, from which two main conclu sions can be drawn. First, conditioning does not always guarantee improved performance. In particular, surface normals appear to provide no benefit for albedo estimation, whereas shading proves to be a more effective conditioning term. Second, the best performance among the conditioning strategies tested is achieved when conditioning is based on shading. As shown in Table 4, the most effective shading conditioning arises from the best shading estimates, which are obtained when the LBM-SID model is trained without reconstruction loss and without normal-based conditioning. This is the opposite trend to what we observe for albedo estimation (Table 1), where the reconstruction loss consistently improves performance. A plausible explanation is that shading is a smooth, low-frequency signal, whereas the reconstruction term enforces pixel-level fidelity to the output image I, a criterion naturally aligned with recovering high-frequency reflectance detail. When applied to shading, this same pixel-wise constraint is comparatively under-determined: many locally smooth shading fields can satisfy I = A·S equally well once A is fixed, so the reconstruction signal contributes noisier, less informative gradients than the direct supervised loss.

This two-stage dependency is visible in our own results: the shading quality differences across LBM-SID variants (Table 4) propagate directly to the downstream albedo estimates (Table 3), where only the better shading estimators yield a net improvement over unconditioned albedo estimation. We partially mitigate this coupling in Section by conditioning the shading estimator itself on a first-pass albedo prediction, which improves the auxiliary shading estimate and, in turn, the final albedo (Table 5); however, this remains a two-pass approximation rather than a joint resolution of the circularity, since any error in the first-pass albedo can still degrade the retrained shading estimator and, consequently, the final prediction. Fully decoupling this dependency, for instance through joint end-to-end training of both models or an uncertainty-aware conditioning mechanism, remains a direction for future work.

Because shading and albedo are two views of the same decomposition, we further ask whether the dependency can be made bidirectional: instead of conditioning albedo estimation on a shading estimate alone, we also condition the shading estimator LBM-SID on the albedo predicted by LBM-AID. As Table 4 shows, this albedo-conditioned shading estimator improves over every previous LBM-SID variant on ARAP, and feeding its output back as the conditioning signal for albedo estimation (Table 5) yields the lowest reconstruction error we observe on all three datasets (MIT: 0.0104, ARAP: 0.0139, Hypersim: 0.0250). This confirms that albedo and shading are mutually informative signals under the LBM framework: each component provides a physically meaningful conditioning cue for estimating the other. We use this configuration – LBM-AID conditioned on the albedo-conditioned LBM-SID – as our best model in the comparison against state-ofthe-art methods in Section .

Finally, we revisit our initial hypothesis concerning adherence to the image formation model defined in Equation 1, which was previously evaluated for the baseline methods in Table 1. We now observe that our proposed approach improves physical consistency, both through the inclusion of the reconstruction loss and the incorporation of shading-based conditioning. These improvements are demonstrated in Table 5.

<table><tr><td></td><td colspan="3">MIT</td><td colspan="2">IIW</td><td colspan="3">Hypersim</td><td rowspan="2">Inf.Time (s/img)</td></tr><tr><td>Framework</td><td>PSNR↑</td><td>MSE↓</td><td>Recon. MSE↓</td><td>WHDR(10%)↓</td><td>WHDR(20%)↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>Recon. MSE↓</td></tr><tr><td>SD-AID (10)</td><td>13.96</td><td>0.05</td><td>0.0215</td><td>27.12</td><td>21.90</td><td>11.55</td><td>0.50</td><td>0.1346</td><td>0.77</td></tr><tr><td>SD-AID (50)</td><td>14.18</td><td>0.05</td><td>0.0204</td><td>28.21</td><td>23.66</td><td>12.11</td><td>0.49</td><td>0.1266</td><td>2.80</td></tr><tr><td>SD-AID-FM (10)</td><td>13.00</td><td>0.07</td><td>0.0256</td><td>30.37</td><td>25.12</td><td>11.08</td><td>0.58</td><td>0.1214</td><td>0.77</td></tr><tr><td>FM-AID</td><td>19.92</td><td>0.02</td><td>0.0129</td><td>32.54</td><td>27.83</td><td>15.40</td><td>0.32</td><td>0.0341</td><td>0.77</td></tr><tr><td>LBM-AID</td><td>21.51</td><td>0.01</td><td>0.0111</td><td>26.13</td><td>22.49</td><td>15.51</td><td>0.30</td><td>0.0362</td><td>0.77</td></tr></table>

Table 1. Performance of different generative models on MIT, IIW, and Hypersim datasets. Reconstruction error is computed between the input image and its estimated version <sup>ˆ</sup>I = A<sup>ˆ</sup> · S<sup>ˆ</sup>. SD-AID models were trained on Stable Diffusion 2.1 with a denoisingdiffusion objective (10 and 50 inference steps respectively); SD-AID-FM shares the same SD2.1 backbone but is trained with a flow-matching objective instead, isolating the effect of the training objective from that of the backbone. FM-AID and LBM-AID were trained using Stable Diffusion XL as velocity estimator.

<table><tr><td rowspan="2">α</td><td colspan="2">MIT</td><td colspan="2">ARAP</td><td colspan="2">Hypersim</td></tr><tr><td>DSSIM↓</td><td>MSE↓</td><td>DSSIM↓</td><td>MSE↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>1.0</td><td>0.0490</td><td>0.0100</td><td>0.1911</td><td>0.0371</td><td>15.51</td><td>0.30</td></tr><tr><td>0.9</td><td>0.0481</td><td>0.0097</td><td>0.1904</td><td>0.0670</td><td>16.09</td><td>0.30</td></tr><tr><td>0.8</td><td>0.0637</td><td>0.0239</td><td>0.1911</td><td>0.0708</td><td>16.18</td><td>0.30</td></tr><tr><td>0.7</td><td>0.0554</td><td>0.0140</td><td>0.1520</td><td>0.0284</td><td>14.07</td><td>0.32</td></tr><tr><td>0.5</td><td>0.0732</td><td>0.0291</td><td>0.1568</td><td>0.0324</td><td>14.39</td><td>0.32</td></tr><tr><td>0.0</td><td>0.0911</td><td>0.0411</td><td>0.1772</td><td>0.0368</td><td>9.72</td><td>0.51</td></tr></table>

Table 2. Parameter setting of the reconstruction loss weight (α, Equation 7) on MIT, ARAP, and Hypersim datasets. λ = 10 is fixed throughout.

## Comparison with state-of-the-art methods

We compare our best model – LBM-AID with the reconstruction loss, conditioned on the albedo-conditioned LBM-SID (Section ) – against recent state-of-the-art IID methods on all five benchmarks. Quantitative figures for competing methods are taken from their original publications; where prior work does not report a given dataset or metric, the corresponding cell is omitted from the tables below.

On MIT (Table 6), our method obtains the best LMSE among the compared methods, though at a higher DSSIM and MSE than PIE-Net [9] and FlowIID [31]. Qualitatively (Figure 3), our predictions closely follow the ground truth, correctly suppressing cast shadows and specular highlights, though some color misalignment remains on a small number of scenes.

On ARAP (Table 7), our method surpasses the deep-learning baselines but trails the diffusion-based IntrinsicDiffusion [17]. As Figure 4 shows, IntrinsicDiffusion introduces a consistent orange/yellowish color shift on wooden surfaces and faces that is not reflected in the aggregate metrics, whereas our predictions preserve fine surface texture well but tend to underestimate overall brightness.

Table 8 reports the remaining three benchmarks together, since they share the same set of compared methods. For Interior-Verse and Hypersim we additionally report recent reinforcementlearning-guided variants (Marigold-X, PRISM-X), which finetune existing generative decomposition models with perceptual feedback [20]. Our single-step model is competitive with, but does not match, these methods or DNF-Intrinsic [32], which achieves the strongest results on InteriorVerse; our method still clearly outperforms the earlier deep-learning baselines and Kocsis et al. [33] on PSNR. On IIW our method outperforms all deeplearning and most diffusion-based baselines, but again falls short of the RL-guided variants.

![](images/60cb7263c3b7fab22fe8e4a5885e92fad8d076871b115b0e8e1689656c532d88.jpg)  
Input RGB Input RGB  
Ours Ours  
GT Reflectance GT Reflectance  
Figure 3. Qualitative comparison of our method against the ground-truth reflectance on the MIT dataset. Cyan rectangles highlight shadow suppression (rows 1 and 4) and color misalignment (row 3).

The qualitative results on InteriorVerse (Figure 5) show that our method most reliably suppresses spurious reflections, such as the mirror in scene 1, that other methods leave baked into the albedo, at the cost of an occasional color shift on reflective or blue surfaces.

![](images/e6837a8ed8330e4ba4f495973bf3eebd3c330babfd37e091abbba9dda7d88742.jpg)  
Figure 4. Qualitative results on the ARAP dataset. Cyan rectangles highlight the orange/yellowish color shift introduced by IntrinsicDiffusion on the face and wooden surfaces, and the texture preservation ofour method on the floor.

<table><tr><td></td><td colspan="2">MIT</td><td colspan="2">IIW</td><td colspan="2">ARAP</td><td colspan="2">InteriorVerse</td></tr><tr><td>Conditioning</td><td>DSSIM↓</td><td>MSE↓</td><td>WHDR (10%)↓</td><td>WHDR (20%)↓</td><td>DSSIM↓</td><td>MSE↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>None</td><td>0.0481</td><td>0.0097</td><td>24.41</td><td>20.72</td><td>0.1904</td><td>0.0670</td><td>15.50</td><td>0.31</td></tr><tr><td>with LBM-NID</td><td>0.0561</td><td>0.0147</td><td>33.44</td><td>30.69</td><td>0.2058</td><td>0.0844</td><td>11.64</td><td>0.47</td></tr><tr><td>with Normal [18]</td><td>0.0554</td><td>0.0141</td><td>33.65</td><td>30.90</td><td>0.2033</td><td>0.0818</td><td>12.16</td><td>0.45</td></tr><tr><td>with LBM-SID</td><td>0.0472</td><td>0.0093</td><td>18.36</td><td>15.68</td><td>0.1536</td><td>0.0307</td><td>16.18</td><td>0.35</td></tr></table>

Table 3. Ablation study of different conditioning inputs across MIT, IIW, ARAP, and InteriorVerse datasets.

<table><tr><td rowspan="2">LBM Shading estimation</td><td colspan="3">ARAP</td></tr><tr><td>DSSIM↓</td><td>LMSE↓</td><td>MSE↓</td></tr><tr><td>LBM-SID without Rec.loss</td><td>0.1116</td><td>0.0102</td><td>0.0200</td></tr><tr><td>LBM-SID</td><td>0.1383</td><td>0.0164</td><td>0.0262</td></tr><tr><td>LBM-SID with Norm. cond.</td><td>0.1391</td><td>0.0157</td><td>0.0256</td></tr><tr><td>LBM-SID with Albedo cond.</td><td>0.1106</td><td>0.0090</td><td>0.0179</td></tr></table>

Table 4. Evaluation of shading prediction on the ARAP dataset. Extended: the last row conditions the shading estimator on the albedo predicted by LBM-AID (with reconstruction loss), rather than on ground-truth normals, and improves on every metric over the unconditioned model.

<table><tr><td rowspan="2">Method</td><td colspan="3">MSE Reconstruction↓</td></tr><tr><td>MIT</td><td>ARAP</td><td>Hypersim</td></tr><tr><td>SD-AID (10)</td><td>0.0215</td><td>0.0261</td><td>0.1346</td></tr><tr><td>SD-AID (50)</td><td>0.0204</td><td>0.0216</td><td>0.1266</td></tr><tr><td>FM-AID</td><td>0.0129</td><td>0.0194</td><td>0.0341</td></tr><tr><td>LBM-AID (wout Rec.Loss)</td><td>0.0111</td><td>0.0207</td><td>0.0362</td></tr><tr><td colspan="4">LBM-AID (with Rec.Loss &amp; Conditionings)</td></tr><tr><td>No conditioning</td><td>0.0111</td><td>0.0186</td><td>0.0305</td></tr><tr><td>LBM-NID</td><td>0.0120</td><td>0.0266</td><td>0.0326</td></tr><tr><td>Normal with [18]</td><td>0.0117</td><td>0.0230</td><td>0.0325</td></tr><tr><td>LBM-SID</td><td>0.0111</td><td>0.0147</td><td>0.0280</td></tr><tr><td>LBM-SID with Albedo cond.</td><td>0.0104</td><td>0.0139</td><td>0.0250</td></tr></table>

Table 5. Image reconstruction error for all the evaluated methods. Computed between input image and its estimated version <sup>ˆ</sup>I = A<sup>ˆ</sup> ·S from the estimated albedo. Rows 1 to 4 reproduce some values from Table 1. Rows 6 to 10 are for all versions of LBM-AID with reconstruction loss and different conditioning. Extended: the last row uses the albedo-conditioned shading estimator introduced in Table 4.

<table><tr><td>Method</td><td>DSSIM↓</td><td>LMSE↓</td><td>MSE↓</td></tr><tr><td>PIE-Net [9]</td><td>0.0340</td><td>0.0136</td><td>0.0028</td></tr><tr><td>FlowIID [31]</td><td>0.0435</td><td>0.0043</td><td>0.0040</td></tr><tr><td>Ours</td><td>0.0443</td><td>0.0030</td><td>0.0076</td></tr></table>

Table 6. Comparison with SOTA IID methods on MIT.

<table><tr><td>Method</td><td>DSSIM↓</td><td>LMSE↓</td><td>MSE↓</td></tr><tr><td>PIE-Net [9]</td><td>0.1598</td><td>0.0280</td><td>0.0428</td></tr><tr><td>Careaga et al. [11]</td><td>0.1555</td><td>0.0239</td><td>0.0366</td></tr><tr><td>Zhu et al. [10]</td><td>0.1363</td><td>0.0165</td><td>0.0276</td></tr><tr><td>IntrinsicDiffusion [17]</td><td>0.1399</td><td>0.0141</td><td>0.0235</td></tr><tr><td>Ours</td><td>0.1460</td><td>0.0204</td><td>0.0266</td></tr></table>

Table 7. Comparison with SOTA IID methods on ARAP.

<table><tr><td rowspan="2">Method</td><td>InteriorVerse</td><td></td><td>Hypersim</td><td></td><td>IIW (WHDR)</td></tr><tr><td>PSNR LPIPS</td><td></td><td>PSNR LPIPS</td><td>10%</td><td>20%</td></tr><tr><td>Zhu et al. [10]</td><td>13.6</td><td>0.24</td><td>11.7</td><td>0.54 34.7</td><td>24.1</td></tr><tr><td>PIE-Net [9]</td><td></td><td></td><td></td><td>33.3</td><td>23.5</td></tr><tr><td>Careaga et al. [11]</td><td>17.4</td><td>0.20</td><td>13.5</td><td>0.34 24.8</td><td>19.2</td></tr><tr><td>Kocsis et al. [33]</td><td>12.2</td><td>0.30</td><td>12.1</td><td>0.41 26.1</td><td>20.7</td></tr><tr><td>RGB↔X [16]</td><td>16.6</td><td>0.17</td><td>17.4</td><td>0.16 23.6</td><td>21.1</td></tr><tr><td>IntrinsicDiff. [17]</td><td></td><td></td><td></td><td>17.9</td><td>13.3</td></tr><tr><td>Marigold [18]</td><td>19.5</td><td>0.19</td><td>18.2</td><td>0.22 16.7</td><td>14.8</td></tr><tr><td>PRISM [19]</td><td>19.9</td><td>0.14</td><td>19.3</td><td>0.18 17.2</td><td>15.9</td></tr><tr><td>DNF-Intrinsic [32]</td><td>21.9</td><td>0.12</td><td></td><td></td><td></td></tr><tr><td>Marigold-X [20]</td><td>19.8</td><td>0.16</td><td>19.0</td><td>0.20 15.2</td><td>14.0</td></tr><tr><td>PRISM-X [20]</td><td>20.7</td><td>0.12</td><td>19.9</td><td>0.17 12.9</td><td>11.9</td></tr><tr><td>Ours</td><td>18.3</td><td>0.31</td><td>16.7</td><td>0.26</td><td>17.9 15.1</td></tr></table>

Table 8. Comparison with SOTA IID methods on the Interior-Verse, Hypersim, and IIW test sets, grouped since they share the same set of compared methods. IIW is evaluated with WHDR at the 10% and 20% thresholds. Dashes (–) denote results not reported in the corresponding original publication.

On Hypersim (Figure 6), our predictions remain consistent across scenes and preserve both surface color and structural detail, while competing methods either blur fine texture or introduce a global color cast; transparent regions remain the main failure case.

Finally, on IIW (Figure 7), our method and RGB↔X show the most faithful color rendition on opaque surfaces among the compared methods, while both still struggle with transparent and metallic materials. Because IIW annotations are sparse and purely relative, the WHDR metric does not penalize the color shifts that we and other diffusion-based methods occasionally exhibit, so long as both compared pixels fall inside the shifted region.

Overall, our single-step LBM formulation is competitive with far more expensive multi-step diffusion pipelines and, on some benchmarks, with methods that additionally fine-tune on realworld feedback via reinforcement learning, despite training exclusively on synthetic data with a small number of transport steps. The gap that remains relative to the RL-guided variants is consistent with our central finding: it is not primarily an architectural limitation, but a consequence of relying solely on synthetic supervision, which we discuss further in Section .

## Limitations and Future Work

Model capacity and efficiency. Our approach builds on the Stable Diffusion XL UNet backbone (≈2.5B parameters), which imposes substantial memory requirements and restricts training to 256 × 256 patches; prediction quality degrades beyond roughly 2K resolution at inference. Preliminary attempts to fine-tune with LoRA [34] to reduce this cost did not converge satisfactorily. Diffusion Transformer backbones, as used in Stable Diffusion 3, are a promising direction for improving scalability without this limitation.

![](images/fb6e377707f441bf3ed6347ece71ae55494fedaf7d2bc1d06bbd0cc05a6704e0.jpg)  
Figure 5. Qualitative comparison on the InteriorVerse dataset. Cyan rectangles highlight the hallucinated mirror reflection in Kocsis et al. (scene 1), the near-zero reflectance collapse on reflective and transparent surfaces in RGB↔X (scenes 1 and 5), the color shift introduced by our method (scenes 1 and 3), and artifacts introduced by Kocsis et al. on the chandelier and wall elements (scenes 3 and 4).

Dependence on synthetic training data. Both training datasets introduce a domain gap that constrains real-world generalization. In Hypersim, near-zero albedo values are assigned indiscriminately to dark opaque objects and non-diffuse materials, creating supervision ambiguity, and a small subset of corrupted, textureless scenes injects further noise during training. InteriorVerse exhibits a bias toward specific indoor color-material associations, which likely explains the color shifts observed in our qualitative results (Section ). More broadly, this reflects the scarcity of largescale, real-world datasets with dense intrinsic ground truth discussed in Section : an ideal dataset would combine dense annotations with broad scene diversity (indoor, outdoor, objects, people) under multiple illuminations and non-Lambertian effects such as interreflections and caustics.

Evaluation inconsistencies across the literature. As Tables 6– 8 illustrate, IID methods are evaluated with heterogeneous, dataset-specific metrics (WHDR for IIW; PSNR/LPIPS for InteriorVerse and Hypersim; DSSIM/LMSE/MSE for MIT and ARAP), and few methods report all five benchmarks. This fragmentation makes it difficult to identify a single best-performing method and motivates standardized, multi-dataset evaluation protocols for the field.

Future directions. Beyond addressing the limitations above, a particularly promising direction is reducing reliance on dense ground-truth annotations through alternative supervision signals; ReasonX [20] shows that reinforcement learning guided by perceptual feedback from multimodal large language models can substantially improve decomposition on real-world images, and being model-agnostic, it is a natural extension for our LBM-based pipeline. Extending the conditioning study in Section to other residual, non-Lambertian effects, such as specular reflections and interreflections, is a further direction suggested by our mutual albedo-shading conditioning results.

GT Reflectance

Input RGB

Zhu et al.

Kocsis et al.

RGB↔X

Ours

![](images/1e1d109e0c985af9cfa872d34b547e65a27c154e22b7fcb8bc5627e0ad82ffe2.jpg)  
Figure 6. Qualitative comparison on the Hypersim dataset. Cyan rectangles highlight the orange-brown color shift introduced by Kocsis et al. (scene 3), and the featureless estimates produced by RGB↔X in low-light regions (scenes 3 and 5).

## Conclusions

In this work, our results support the suitability of the LBM framework to deal with low-level image processing tasks compared to previous generative approaches. In particular, we focused on the IID problem that is highly ill-posed and requires solutions with physical coherence. We propose a novel model for albedo estimation that satisfies the image formation model while leveraging the flexibility of LBM to incorporate conditioning inputs, and we show that albedo and shading act as mutually informative conditioning signals for one another, whereas surface normals do not, despite their geometric expressiveness. Benchmarked against state-of-the-art IID methods on five datasets (Section ), our single-step model is competitive with substantially more expensive multi-step diffusion pipelines, though a gap remains relative to methods that additionally fine-tune on real-world feedback.

Finally, we want to note that the IID field appears to have reached a bottleneck, where progress has seemingly stalled. It is difficult to identify a clear state-of-the-art method, as no existing approach consistently outperforms others across all datasets, and evaluation practices remain fragmented across dataset-specific metrics (Section ). This limitation is likely due to insufficient generalization capabilities, which stem from the strong dependence on largescale synthetic datasets for training. Recently, Reason-X [20] has acknowledged and addressed this bottleneck by introducing a novel perspective based on reinforcement learning, aimed at improving the ability of trained models to generalize to real-world images. As a natural extension of our work, a key direction for future research will be to explore this approach to further enhance the performance and generalization of the proposed method.

## Acknowledgments

This work was funded by Grant PID2024-162555OB-I00; by MI-CIU/AEI/10.13039/501100011033, ERDF/EU and the FEDER; by the grant Catedra ENIA UAB-Cru\` ¨ılla (TSI-100929-2023-2) from the Ministry of Economic Affairs and Digital Transition of Spain. DSL also acknowledges the FPI grant from Spanish Ministry of Science and Innovation (PRE2022-101525). JVC also acknowledges the 2025 Leonardo Grant for Scientific Research and Cultural Creation from the BBVA Foundation. The BBVA Foundation accepts no responsibility for the opinions, statements and contents included in the project and/or the results thereof, which are entirely the responsibility of the authors.

![](images/5c8819ed99adbca21c6d5613ba85a1cc5e3e4420c3aadf20968e32b1a5d2b09c.jpg)  
Figure 7. Qualitative results on the IIW dataset. Green and red point pairs in the input image indicate human annotations of equal and different reflectance, respectively. Cyan rectangles highlight the orange/yellowish color shift introduced by IntrinsicDiffusion, the near-black predictions on window regions, and the reflectance collapse on the refrigerator’s metallic surface.

## References

[1] H. G. Barrow and J. M. Tenenbaum. Recovering Intrinsic Scene Characteristics from Images, pages 3–26. Academic Press, 1978.

[2] M. Serra, O. Penacchio, R. Benavente, M. Vanrell, and D. Samaras. The photometry of intrinsic images. In CVPR, 2014.

[3] J. Shi, Y. Dong, H. Su, and S. X. Yu. Learning nonlambertian object intrinsics across shapenet categories. In CVPR, 2017.

[4] E. H. Land and J. J. McCann. Lightness and retinex theory. JOSA, 61(1):1–11, 1971.

[5] B. V. Funt, M. S. Drew, and M. Brockington. Recovering shading from color images. In ECCV, 1992.

[6] M. F. Tappen, W. T. Freeman, and E. H. Adelson. Recovering intrinsic images from a single image. IEEE TPAMI, 2005.

[7] R. Grosse, M. K. Johnson, E. H. Adelson, and W. T. Freeman. Ground truth dataset and baseline evaluations for intrinsic image algorithms. In ICCV, 2009.

[8] S. Bell, K. Bala, and N. Snavely. Intrinsic images in the

wild. ACM ToG, 33(4):159:1–159:12, 2014.

[9] P. Das, S. Karaoglu, and T. Gevers. Pie-net: Photometric invariant edge guided network for intrinsic image decomposition. In CVPR, 2022.

[10] J. Zhu, F. Luan, Y. Huo, Z. Lin, Z. Zhong, D. Xi, R. Wang, H. Bao, J. Zheng, and R. Tang. Learning-based inverse rendering of complex indoor scenes with differentiable monte carlo raytracing. In SIGGRAPH Asia 2022, 2022.

[11] C. Careaga and Y. Aksoy. Intrinsic image decomposition via ordinal shading. ACM ToG, 43(1), 2023.

[12] Z. Li and N. Snavely. Cgintrinsics: Better intrinsic image decomposition through physically-based rendering. In ECCV, 2018.

[13] W-C. Ma, H. Chu, B. Zhou, R. Urtasun, and A. Torralba. Single image intrinsic decomposition without a single intrinsic image. In ECCV, 2018.

[14] L. Lettry, K. Vanhoey, and . Van Gool. Darn: A deep adversarial residual network for intrinsic image decomposition. In WACV, 2018.

[15] Y. Yang, H. A. Sial, R. Baldrich, and M. Vanrell. Relighting from a single image: Datasets and deep intrinsic-based architecture. IEEE Transactions on Multimedia, 2025.

[16] Z. Zeng, V. Deschaintre, I. Georgiev, Y. Hold-Geoffroy, Y. Hu, F. Luan, L-Q Yan, and M. Hasan. RGB ˇ ↔X: Image decomposition and synthesis using material- and lightingaware diffusion models. In ACM SIGGRAPH 2024, 2024.

[17] J. Luo, D. Ceylan, J. S. Yoon, N. Zhao, J. Philip, A. Fruhst¨ uck, and T. Wang. Intrinsicdiffusion: Joint intrin-¨ sic layers from latent diffusion models. In ACM SIGGRAPH 2024, 2024.

[18] B. Ke, K. Qu, T. Wang, N. Metzger, S. Huang, B. Li, and K. Schindler. Marigold: Affordable adaptation of diffusionbased image generators for image analysis. IEEE TPAMI, 2025.

[19] A. Dirik, T. Wang, D. Ceylan, S. Zafeiriou, and A. Fruhst¨ uck. Prism: A unified framework for photoreal-¨ istic reconstruction and intrinsic scene modeling. In ICPR, 2026.

[20] A. Dirik, T. Wang, D. Ceylan, S. Zafeiriou, and A. Fruhst¨ uck. Reasonx: Mllm-guided intrinsic image de-¨ composition. In CVPR, 2026.

[21] R. Grosse, M. K. Johnson, E. H. Adelson, and W. T. Freeman. Ground truth dataset and baseline evaluations for intrinsic image algorithms. In ICCV, 2009.

[22] M. Roberts, J. Ramapuram, A. Ranjan, A. Kumar, M. A. Bautista, N. Paczan, and J. M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021.

[23] N. Bonneel, B. Kovacs, S. Paris, and K. Bala. Intrinsic decompositions for image editing. Computer Graphics Forum, 36(2):1–12, 2017.

[24] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.

[25] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In ICLR, 2023.

[26] C. Chadebec, O. Tasar, S. Sreetharan, and B. Aubin. Lbm: Latent bridge matching for fast image-to-image translation. In ICCV, 2025.

[27] D. Serrano-Lozano, A. Bhattad, L. Herranz, J-F. Lalonde, and J. Vazquez-Corral. Synclight: Controllable and consistent multi-view relighting. arXiv preprint arXiv:2601.16981, 2026.

[28] D. Li, A. Yadav, C. Peng, R. Chellappa, and A. Bhattad. Syncfix: Fixing 3d reconstructions via multi-view synchronization. arXiv preprint arXiv:2604.11797, 2026.

[29] D. Serrano-Lozano, F. A. Molina-Bakhos, D. Xue, Y. Yang, M. Pilligua, R. Baldrich, M. Vanrell, and J. Vazquez-Corral. Promptnorm: Image geometry guides ambient light normalization. In CVPRW, 2025.

[30] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Podrasky, and R. Rombach. Sdxl: Improving¨ latent diffusion models for high-resolution image synthesis, 2023.

[31] M. Singla, S. Kumari, and S. Raman. Flowiid: Singlestep intrinsic image decomposition via latent flow matching. arXiv preprint arXiv:2601.12329, 2026.

[32] R. Zheng, Q. Zhang, C. Long, and W. S. Zheng. Dnfintrinsic: Deterministic noise-free diffusion for indoor inverse rendering. In ICCV, 2025.

[33] P. Kocsis, V. Sitzmann, and M. Nießner. Intrinsic image dif-

fusion for indoor single-view material estimation. In CVPR, 2024.

[34] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022.