# Geodesic-informed Generative Difusion Model For Topology-preserved Image Video Generation

Nian Wu <sup>1</sup> , Nivetha Jayakumar <sup>1</sup> , Jiarui Xing <sup>3</sup> , Miaomiao Zhang <sup>1,2</sup>

1 University of Virginia, Department of Electrical & Computer Engineering, Charlottesville, VA, USA

2 University of Virginia, Department of Computer Science, Charlottesville, VA, USA

3 Yale School of Medicine, New Haven, CT, USA

## Abstract

<sup>6</sup>Generative difusion models have emerged as a class of powerful techniques for various imaging applications, including but not limited to synthesis, reconstruction, and segmentation. Despite their success, current generative models pose two 2key limitations. First, they primarily rely on image intensity and texture information, with limited attention to underlying pobject geometry. As a result, they do not guarantee geometric or topological consistency during the generation process, <sup>e</sup>which is a crucial requirement for high-stakes domains such as computational anatomy, biology, and robotics, where preserving object structure is critical. Second, existing models fail to explicitly learn or represent shape changes in <sup>8</sup>the generative process. Such deformation dynamics remain occluded within network parameters; hence leaving the ]transformation process uninterpretable and physically uninformed. To address these challenges, we introduce IGG G<sub>(Image Generation informed by Geodesic dynamics), a novel framework that integrates topology-preserving geodesic</sub> L<sub>principles into the difusion-based generative process. In contrast to conventional methods that operate in image</sub> <sup>s</sup>intensity space, IGG learns and synthesizes diverse samples within geodesic deformation spaces, where geometric object [changes are learned as smooth and invertible smooth mappings from a given template/source image. Specifically, IGG employs a two-stage architecture: (i) a geodesic-informed image registration (GIR) module that directly encodes geodesic paths of image deformations into a compact latent space, and (ii) a latent geodesic difusion (LGD) mode 3that captures the distribution of these deformation representations, conditioned on a template image and/or text <sup>5</sup>prompts. We validate IGG on the datasets of plant growth and brain MRI scans. Experimental results demonstrate that 8IGG outperforms state-of-the-art image generation and editing models, producing realistic, high-quality images with <sup>0</sup>preserved topology and fewer artifacts. Furthermore, when used for data augmentation in downstream segmentation 9tasks, IGG substantially improves segmentation accuracy, particularly in low-data regimes. Our code is publicly available <sup>0</sup>at https://github.com/nellie689/IGG.

:Keywords

iGeodesics, Difeomorphisms, Generative difusion model, Neural Operator

<sup>r</sup>Article informations

https://doi.org/10.59275/j.melba.2026-eec1

Volume 2026, Received: 2025-11, Published 2026-08

Corresponding author: bsw3ac@virginia.edu

©2026 Nian Wu, Nivetha Jayakumar, Jiarui Xing, and Miaomiao Zhang. License: CC-BY 4.0

Check for updates

## 1. Introduction

Recent advances in deep generative modeling have significantly impacted various image analysis tasks, including but not limited to image synthesis (Xing et al., 2024; Xie et al., 2025; Liu et al., 2021; Reinhold et al., 2021), reconstruction (Jayakumar et al., 2023; Tezcan et al., 2018), translation (Deb et al., 2025; Graf et al., 2023; Ozbey<sup>¨</sup> et al., 2023), and more (Yang et al., 2019; Hossain and Zhang, 2025). Early research works, such as the variational autoencoder (VAE), assume a Gaussian prior over the latent feature space and learn its approximate posterior through variational inference (Kingma and Welling, 2013). Although efective, these models can produce overly smooth or blurry outputs due to their simplified assumptions (Hossain and Zhang, 2025). Another category of generative modeling is the generative adversarial networks (GANs), which learn to produce realistic data samples by training a generator to compete against a discriminator that distinguishes generated samples from real data (Goodfellow et al., 2020). However, such models may sufer from mode collapses, where the generator produces limited diversity of outputs by ignoring other modes of the true data distribution (Zhang et al., 2018; Bang and Shim, 2021; Li et al., 2021). In contrast, difusion models have recently emerged as a powerful class of deep generative frameworks that achieve state-of-the-art performance in modeling complex, multimodal data distributions and generating photorealistic images across diverse applications (Mukhopadhyay et al., 2023; Ho et al., 2020; Deb et al., 2025). These models employ an implicit generative scheme by learning to reverse a gradual noising process and iteratively transforming random noise into fine-grained and high-fidelity image samples (Ho et al., 2020). Due to these capabilities, difusion models have demonstrated remarkable success in synthesizing highly realistic medical images to address critical challenges in biomedical research, such as data/label scarcity Sharma et al. (2023); Jayakumar et al. (2024), missing or incomplete modalities Jayakumar et al. (2023); Deb et al. (2025), and the need for predictive imaging to support disease progression modeling and clinical decision-making Yoon et al. (2023); Dao et al. (2024); Zhang et al. (2024).

Despite the success of the aforementioned generative difusion models, current approaches face several limitations that can constrain their practical use in high-stakes domains requiring high accuracy and fidelity, such as medical imaging, robotics, and autonomous systems. In particular, most existing difusion models focus on image intensity and texture, with little explicit consideration of object geometry during learning and generation (Nguyen et al., 2024; Li et al., 2024; Hertz et al., 2022; Kawar et al., 2023). As a result, they may produce unrealistic or biologically implausible samples, which poses significant risks for downstream applications where accurate geometric preservation is critical (Azizi et al., 2023; Jayakumar et al., 2023; Starck et al., 2025). Recent advances have introduced simple geometric constraints such as shape localization (Patashnik et al., 2023; Gupta et al., 2024), boundary conditions (Maz´e and Ahmed, 2023), and 3D shape priors (Yu et al., 2025; Hu et al., 2024) in the modeling process. However, these approaches lack fine-grained structural preservation that is highly desirable in sensitive domains, including computational anatomy and medical imaging. Another important yet under-investigated limitation of current difusion models is their inability to provide interpretable metrics or quantitative measures of topological integrity for image objects. Commonly used evaluation metrics include Fr´echet Inception Distance (FID) for assessing distribution similarity between real and generated data (Heusel et al.,

2017), Inception Score (IS) for measuring diversity and quality (Salimans et al., 2016), and Structural Similarity Index (SSIM) for evaluating pixel-level fidelity (Wang et al., 2004). While these metrics efectively assess visual quality, they do not account for the preservation of object geometry and topology, raising questions about their validity in structure-sensitive applications.

In addition, existing difusion models do not explicitly capture or represent object shape changes during the generative process (Ho et al., 2020; Dao et al., 2024; Jayakumar et al., 2024; Deb et al., 2025; Yoon et al., 2023). Recent studies have demonstrated that incorporating a geodesicconstrained deformation process to model object shape changes within the network can promote physically interpretable deformation trajectories, which enables the registration framework to capture the underlying dynamics of shape evolution rather than performing purely data-driven alignment (Wu and Zhang, 2023, 2024). Moreover, this geodesic guidance provides an additional source of regularization in the deformation learning process; thereby improving the biological plausibility of the registration results by suppressing artifacts such as folding, tearing, and crossing in the predicted transformations.

This paper presents an extended work of Image Generation informed by Geodesic dynamics (IGG) (Wu et al., 2025), that for the first time generates images by deforming a given template along random geodesics in deformation spaces guided by text instructions. In contrast to current approaches (Brooks et al., 2023; Ho et al., 2022; Kim et al., 2022), the IGG framework ensures topological consistency by treating each generated image as a deformed variation of a template or reference image through learned difeomorphic transformations (also known as a one-to-one, smooth, and invertible mapping). A dynamic geodesic, producing natural transformations and providing interpretable metrics to quantify topological changes, are explicitly learned during the difusion process. More specifically, the IGG model comprises two key components: a latent representation learning framework of difeomorphic transformations informed by geodesic dynamics, and a latent geometric difusion model conditioned on user-defined text instructions. Building on top of IGG (Wu et al., 2025), our work further

(i) Expand the core method section of IGG by introducing details of the geodesic-informed image registration module, which provides a more principled representation of topology-preserving transformations.

(ii) Thoroughly investigate the quantitative metrics to assess both the fidelity and topological consistency of the generated samples.

(iii) Design downstream experiments in the context of image segmentation to demonstrate an additional clinical utility of the generated samples in real-world applications.

![](images/b0adc0c30c71a710d59480edda6b118e3048de0f576dc3abb3ace47a4ef32645.jpg)  
Figure 1: An illustrated example of a geodesic path in the space of brain deformations.

We demonstrate the efectiveness of IGG on a diverse set of real-world datasets, including Komatsuna plant growth data (Uchiyama et al., 2017) and brain MRIs (LaMontagne et al., 2019). Experimental results show that IGG significantly outperforms state-of-the-art text-guided generative models (Ho et al., 2022; Yang et al., 2024; Xing et al., 2025) in producing image samples with well-preserved topological structures. We further demonstrate the potential of IGG to enhance deep learning-based downstream tasks by training brain segmentation networks with synthesized image-label pairs. It is worth noting that, once trained, IGG generalizes well across image domains (e.g., variations in image contrast and scanner types) due to its flexibility in generating geometric deformations conditioned on any given template.

## 2. Background: Geodesics In Deformation Spaces

In this section, we briefly review the basic concept of geodesics in deformation spaces, which provides a principled way to model smooth and natural deformations between images while preserving structural integrity (Miller et al., 2002; Beg et al., 2005). With the underlying assumption that objects of a generic class (e.g., human brains, hearts, or lungs) are described as deformed variants of a given template, descriptors of that class arise naturally by deforming the template to other images along a geodesic - a shortest path with locally minimized energy of transformations (Avants et al., 2008; Joshi et al., 2004). In theory, every topological property of the deformed template can be preserved by enforcing the resulting transformations to be difeomorphisms, i.e., diferentiable, bijective mappings with diferentiable inverses (Beg et al., 2005; Arnold, 1966; Miller et al., 2002). Violations of such constraints introduce image artifacts, such as tearing, crossing, or passing through itself.

Let $\mathrm { D i f f } ^ { \infty } ( \Omega )$ denote the space of smooth difeomorphisms on a d-dimensional torus domain $\Omega ~ = ~ \mathbb { R } ^ { d } / \mathbb { Z } ^ { d }$ The tangent space of difeomorphisms is the space $V =$ ${ \mathfrak { X } } ^ { \infty } ( T \Omega )$ of smooth vector fields on Ω. Consider a timevarying velocity field, $\{ v _ { t } \} : [ 0 , 1 ] \to V$ , we can generate difeomorphisms $\{ \phi _ { t } \}$ between pairwise images by solving

$$
\frac { d \phi _ { t } } { d t } = v _ { t } ( \phi _ { t } ) , t \in [ 0 , 1 ] ,\tag{1}
$$

where $\phi _ { 0 }$ is the identity map, e, and $\phi _ { 1 }$ is the target transformation.

The geodesic minimizes the functional $\textstyle \int _ { 0 } ^ { 1 } ( { \mathcal { L } } v _ { t } , v _ { t } )$ dt (Beg et al., 2005), where $\mathcal { L } : V \to V ^ { * }$ is a symmetric, positivedefinite diferential operator that maps a tangent vector $v ( t ) \in V$ into its dual space as a momentum vector $m ( t ) \in V ^ { * }$ . We typically write $m ( t ) ~ = ~ \mathcal { L } v ( t )$ , or $v ( t ) = \mathcal { K } m ( t )$ , with K being an inverse operator of ${ \mathcal { L } } .$ In this paper, we adopt a commonly used Laplacian operator $\mathcal { L } = ( - \alpha \Delta + e ) ^ { 3 }$ , where $\alpha$ is a weighting parameter that controls the smoothness of transformation fields. The $( \cdot , \cdot )$ is a dual pairing, which is similar to an inner product between vectors. An example of geodesic path in the space of deformations is illustrated in Fig. 1.

According to a well-known geodesic shooting algorithm (Vialard et al., 2012), the minimum of the functional mentioned above is uniquely determined by solving a Euler-Poincar´e diferential (EPDif) equation (Vialard et al., 2012; Younes et al., 2009) with a given initial condition. That is, for $\forall v _ { 0 } \in V$ , a geodesic path $t \mapsto \phi _ { t }$ in the space of difeomorphisms can be computed by forward shooting the EPDif equation

$$
\frac { \partial v _ { t } } { \partial t } = - K \left[ ( D v _ { t } ) ^ { T } m _ { t } + D m _ { t } v _ { t } + m _ { t } \operatorname { d i v } v _ { t } \right] ,\tag{2}
$$

where D denotes the Jacobian matrix and div is a divergence operator.

Derive geodesics from images. Consider a source image S and a target image $T$ defined in the domain Ω $( S ( x ) , T ( x ) \ : \Omega  \mathbb { R } )$ The optimization of geodesic transformations can be formulated as minimizing an energy function over the initial velocity, subject to the EPDif equations, i.e.,

$$
E ( v _ { 0 } ) = ( \mathcal { L } v _ { 0 } , v _ { 0 } ) + \lambda \mathsf { D i s t } ( S ( \phi _ { 1 } ) , T ) \mathsf { s . t . } \mathsf { E q . } ( 1 ) \& ( 2 ) .
$$

Here, Dist $( \cdot , \cdot )$ is a distance function that measures the dissimilarity between images, and λ is a positive weighting parameter. In this paper, we will use the commonly used sum-of-squared intensity diferences (Beg et al., 2005; Wu et al., 2024; Zhang et al., 2017).

## 3. Method: IGG

This section introduces an extended work of, IGG, that for the first time generates images in deformation spaces guided by text instructions (Wu et al., 2025). A dynamic geodesic, producing natural transformations and providing interpretable metrics to quantify topological changes, will be explicitly learned during the generative difusion process. Our IGG model consists of two main components: (i) a geodesic-informed image registration (GIR) network that directly encodes geodesic paths of image deformations into a compact latent space; and (ii) a latent geodesic difusion (LGD) model that captures the latent distribution of these deformation representations, conditioned on a template image and/or user-defined text prompts. An overview of our model is illustrated in Fig. 2.

## 3.1 Representation Learning of Geodesic Deformations via GIR

Consider a given set of N template/source and target image pairs associated with text instructions $\{ s ^ { n } , f ^ { n } , \Lambda ^ { n } \} _ { n = 1 } ^ { N }$ We introduce a GIR network, based on an autoencoder architecture, to learn the latent representation of geodesic transformations, parameterized by velocity fields $\{ v _ { t } \}$ , between a template $s ^ { n }$ and a target $f ^ { n }$ . The encoder of $\mathsf { G l R } , \mathcal { E } _ { \theta _ { v } }$ , maps input data to the latent representation of a sequence of velocity fields, $\{ z _ { v _ { t } } \} _ { t = 0 } ^ { 1 }$ . Inspired by the recent work NeurEPDif (Wu and Zhang, 2023), which develops a neural operator to learn mappings governed by the EPDif equation, we incorporate the NeurEPDif module, $\mathcal { G } _ { \theta _ { r } }$ , into our IGG model to learn geodesics in latent deformation spaces. More specifically, starting with a latent initial velocity $z _ { v _ { 0 } }$ , the operator $\mathcal { G } _ { \boldsymbol { \theta } _ { 1 } }$ will iteratively propagate it along the geodesic path, $z _ { v _ { 0 } } \longmapsto \dots { } z _ { v _ { t } } \longmapsto \dots { } z _ { v _ { 1 } }$ , through a designed latent geodesic evolution module.

Latent Geodesic Evolution. Similar to (Wu and Zhang, 2023, 2024), we leverage neural operators as surrogate models to approximate the mapping function of the EPDif equation during network training. In particular, we employ a multilayer neural network with J hidden layers to simulate the geodesic mapping function $( \mathsf { E P D i f f } )$ from $z _ { v _ { t } }$ $\mathrm { t o ~ } z _ { v _ { t + 1 } }$ For notational simplicity, we let $u ^ { j }$ denote the input of the j-th hidden layer and $u ^ { j + 1 }$ its output, where $j \in \{ 1 , \cdots , J \}$ . At each hidden layer, we combine a local linear transformation $W ^ { j }$ with a global convolutional kernel $\mathcal { H } ^ { j }$ , enabling the extraction of both local and global representations. A nonlinear activation function, $\sigma ( \cdot )$ , is defined as a Gaussian Error Linear Unit (GeLU) (Hendrycks and Gimpel, 2016) with a smoothing operator K to ensure continuity and smoothness in the output signal, i.e., $\sigma ( \cdot ) : = K ( \mathsf { G e L U } ( \cdot ) )$ . Formally, we update $u ^ { j + 1 }  u ^ { j }$ by

$$
u ^ { j + 1 } : = \sigma ( W ^ { j } u ^ { j } + \mathcal { H } ^ { j } \ast u ^ { j } ) ,
$$

where ∗ represents a convolution operator. Note that, for computational eficiency, the global convolution is implemented as an element-wise multiplication in the Fourier domain, followed by an inverse transform back to the spatial domain.

Finally, the geodesic evolution of the latent velocity from $z _ { v _ { t } }$ to $z _ { v _ { t + 1 } }$ at each time point t can be defined as

$$
\boldsymbol z _ { \boldsymbol v _ { t + 1 } } : = \sigma ( { \boldsymbol W } ^ { J } , \mathcal { H } ^ { J } ) \circ \cdots \circ \sigma ( { \boldsymbol W } ^ { 2 } , \mathcal { H } ^ { 2 } ) \circ \sigma ( { \boldsymbol W } ^ { 1 } , \mathcal { H } ^ { 1 } , \boldsymbol { z } _ { \boldsymbol v _ { t } } ) ,
$$

where ◦ denotes the composition of network operations. The decoder, $\mathcal { D } _ { \theta _ { v } }$ , then projects these latent representations back to the full-dimensional image space to produce a geodesic flow of velocity fields, $v _ { 0 } \longmapsto \cdots v _ { t } \longmapsto \cdots v _ { 1 }$ The corresponding deformations, $\phi _ { 0 } \longmapsto \cdots \phi _ { t } \longmapsto \cdots \phi _ { 1 }$ are then generated according to Eq. 1. For simplicity of notation, we define $\Phi \triangleq \{ \phi _ { t } \} _ { t = 0 } ^ { 1 }$ in the following sections.

The network loss combines contributions from an unsupervised registration loss, and a geodesic loss guided by numerical solutions of the EPDif equation obtained via Euler integration. These solutions are represented as $\{ \hat { v } _ { t } ^ { n } \} , t \in ( 0 , 1 ]$ with a simultaneously learned initial velocity $\hat { v } _ { 0 } ^ { n }$ . To simplify the notation, we define $\hat { \pmb { v } } ^ { n } \triangleq \{ \hat { v } _ { t } ^ { n } \}$ before formulating the loss function as

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta _ { v } , \theta _ { r } ) = \sum _ { n = 1 } ^ { N } \lambda \| ( s ^ { n } ( \phi _ { 1 } ^ { n } ( \theta _ { v } ) ) - f ^ { n } \| _ { 2 } ^ { 2 } } \\ { \displaystyle + \frac { 1 } { 2 } ( \mathcal { L } v _ { 0 } ^ { n } ( \theta _ { v } ) , v _ { 0 } ^ { n } ( \theta _ { v } ) ) } \\ { \displaystyle + \eta \| \mathcal { D } _ { \theta _ { v } } ( \mathcal { G } _ { \theta _ { r } } ( z _ { v _ { 0 } } ) ) - \hat { v } ^ { n } \| _ { 2 } ^ { 2 } } \\ { \displaystyle + \mathsf { R e g } ( \theta _ { r } , \theta _ { v } ) , s . t . \mathsf { E q . } ( 1 ) , } \end{array}\tag{3}
$$

where $\lambda$ and $\eta$ are positive weighting parameters to balance the image matching term and the geodesic loss, and Reg(·) is a network regularization.

## 3.2 Latent Geodesic Difusion Model

With a pre-trained GIR network, we are now ready to introduce a latent geodesic difusion module to simulate the distribution of latent geodesic flows for image transformations, $\{ z _ { v _ { t } } ^ { n } \} _ { t = 0 } ^ { 1 }$ . To simplify the notation, we define $z ^ { n } : =$ $\{ z _ { v _ { t } } ^ { n } \} _ { t = 0 } ^ { 1 }$ for the n-th subject, where all time-dependent components $\{ z _ { v _ { t } } ^ { n } \}$ are concatenated to form a unified representation. In contrast to previous approaches that perform the difusion process in the image space (Brooks et al., 2023; Ho et al., 2022; Kim et al., 2022), our latent geometric difusion operates in the deformation space. Specifically, our difusion process directly samples geodesics of topologypreserving difeomorphic transformations, which are then applied to deform the input template to generate the final output.

![](images/884c7e0830d57e2d510ba23699c2abab48d3f022a590d2a328dafefbb9c1445d.jpg)  
Figure 2: An overview of our IGG framework with two key components: (i) GIR to learn the latent representations of geodesic transformations from paired template and target images; and (ii) LGD to generate a time-sequence of image transformations over time, conditioned on text instructions.

## 3.2.1 Forward difusion in geodesic deformation spaces.

Following the principles of denoising difusion probabilistic models (DDPM) (Ho et al., 2020) and video difusion models (Ho et al., 2022), our forward difusion is fixed to a Markov chain that progressively adds noise, $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$ to the initial latent geodesic representations $z _ { 0 } ^ { n } \sim p ( z _ { 0 } )$ with a number of $T$ steps. Mathematically, the forward difusion process can be formulated as

$$
q ( z _ { \tau } ^ { n } \mid z _ { \tau - 1 } ^ { n } ) : = \mathcal { N } \left( z _ { \tau } ^ { n } ; \sqrt { 1 - \beta _ { \tau } } z _ { \tau - 1 } ^ { n } , \beta _ { \tau } \mathbf { I } \right) ,
$$

where $\tau \in [ 1 , \cdots , T ]$ and $\beta _ { \tau }$ is a time-dependent variance schedule of scalar values $\in ( 0 , 0 . 9 9 9 ]$ used to parameterize the probabilistic transitions. Alternatively, the forward difusion process of sampling $z _ { \tau }$ at an arbitrary timestep $\tau$ can be formulated in closed form using

$$
\begin{array} { c } { { q ( z _ { \tau } ^ { n } | z _ { 0 } ^ { n } ) = \mathcal { N } ( z _ { \tau } ^ { n } ; \sqrt { \bar { \alpha } _ { \tau } } z _ { 0 } ^ { n } , ( 1 - \bar { \alpha } _ { \tau } ) \mathbf { I } ) , } } \\ { { z _ { \tau } ^ { n } = \sqrt { \bar { \alpha } _ { \tau } } z _ { 0 } ^ { n } + \sqrt { 1 - \bar { \alpha } _ { \tau } } \epsilon , } } \end{array}\tag{4}
$$

where $\alpha _ { \tau } ~ = ~ 1 - \beta _ { \tau }$ and $\begin{array} { r } { \bar { \alpha } _ { \tau } ~ = ~ \prod _ { s = 1 } ^ { \tau } \alpha _ { s } } \end{array}$ , where $s \in$ $[ 1 , \cdots , \tau ]$

## 3.2.2 Reverse difusion with geometric conditioning.

Similar to Ho et al. (2020), the reverse difusion process iteratively denoises the latent variable of the difusion model $z _ { \tau } ^ { n }$ over τ steps, given template-specific conditions. We introduce a novel geometric conditioning in the latent geodesic space, which enables the model to adapt more efectively to specific contexts; hence improving its generative performance based on observed template images and provided text instructions. More specifically, our geometric conditioning is achieved by concatenating two components: image embedding in the form of downsampled template image gradient, $\{ \nabla s ^ { n } \}$ and the geodesic deformation embedding, represented by the latent velocity fields, $\{ z _ { \tau } ^ { n } \}$ . Following a similar principle as seen in (Ho et al., 2020), we sample the denoised latent geodesic flow of velocity fields $z _ { 0 } ^ { n }$ starting from the noisy state $z _ { \tau } ^ { n }$ at the τ-th time-step and progressively remove noise to reconstruct the original latent representation. This process is mathematically expressed as

$$
\begin{array} { r l } & { \quad p _ { \theta } ( z _ { \tau - 1 } ^ { n } | z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau ) = } \\ & { \quad \mathcal { N } ( z _ { \tau - 1 } ^ { n } ; \mu _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau ) , \Sigma _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau ) ) , } \end{array}
$$

where $\mu _ { \theta }$ and $\Sigma _ { \theta }$ represent the parameterized mean and variance of the approximated Gaussian distribution at each time step of the reverse difusion process (Ho et al., 2022) . In our experiments, we employ a 3D UNet architecture (Ronneberger et al., 2015) $\mathbf { Z } _ { \theta }$ as the deep predictive model, which predicts the noise to be removed from $z _ { \tau } ^ { n }$ to reconstruct $z _ { \tau ^ { - } } ^ { n }$ using the equation:

$$
z _ { \tau - 1 } ^ { n } = \frac { 1 } { \sqrt { \alpha _ { \tau } } } ( z _ { \tau } ^ { n } - \frac { 1 - \alpha _ { \tau } } { \sqrt { 1 - \bar { \alpha } _ { \tau } } } \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau ) ) + \sigma _ { t } I ,\tag{5}
$$

where $I \sim \mathcal { N } ( 0 , { \bf I } )$ and $\begin{array} { r } { \sigma _ { t } = \tilde { \beta } _ { t } = \frac { 1 - \bar { \alpha } _ { t - 1 } } { 1 - \bar { \alpha } _ { t } } \cdot \beta _ { t } } \end{array}$

The text embedding, encoded using a pre-trained CLIP text encoder (Radford et al., 2021), $\{ \Lambda ^ { n } \}$ is integrated into this network via cross-attention. With an optimized denoising network $\mathbf { Z } _ { \theta }$ , the iterative sampling using Eq 5 yields a clean geodesic deformation embedding $z _ { 0 } ^ { n }$

The network loss of our proposed latent geodesic difusion module is formulated as

$$
\mathcal { L } _ { \boldsymbol { \theta } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } | | \epsilon _ { \tau } ^ { n } - \mathbf { Z } _ { \boldsymbol { \theta } } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau ) | | _ { 2 } ^ { 2 } + \mathbf { r e g } ( \boldsymbol { \theta } ) , \tau \in \mathsf { U } [ 1 , T ] ,\tag{6}
$$

where $\mathsf { r e g } ( \cdot )$ is a regularization of the network parameter $\theta ,$ U denotes a uniform distribution and $T$ is the total number of difusion timesteps.

Classifier-free guidance. To balance the trade-of between sample quality and diversity of velocity fields in the latent space after training, we optimize the combination of conditional and unconditional difusion models. This is achieved by leveraging image-conditioned guidance with scale $\delta _ { \mathrm { { I } } } ,$ , text-conditioned guidance with scale $\delta _ { \mathrm { T } }$ , and a null-condition ∅. Following similar principles from the classifier-free guidance approach (Ho and Salimans, 2022), we first jointly train a conditional and an unconditional IGG model. The resulting score estimates from these models are then combined to predict the noise $\hat { \mathbf { Z } }$ as

$$
\begin{array} { c } { \hat { \mathbf { Z } } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathbb { X } ^ { n } , \tau ) } \\ { = \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \mathcal { Q } , \emptyset , \tau ) } \\ { + \delta _ { \mathrm { I } } \cdot [ \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathcal { Q } , \tau ) - \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \mathcal { Q } , \mathcal { Q } , \tau ) ] } \\ { + \delta _ { \mathrm { T } } \cdot [ \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathbb { X } ^ { n } , \tau ) - \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathcal { Q } , \tau ) ] } \\ { = ( 1 - \delta _ { \mathrm { I } } ) \cdot \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \mathcal { Q } , \emptyset , \tau ) } \\ { + ( \delta _ { \mathrm { I } } - \delta _ { \mathrm { T } } ) \cdot \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathcal { Q } , \tau ) } \\ { + \delta _ { \mathrm { T } } \cdot \mathbf { Z } _ { \theta } ( z _ { \tau } ^ { n } , \nabla s ^ { n } , \mathbb { X } ^ { n } , \tau ) . } \end{array}
$$

## 3.3 Network Optimization

We first optimize the loss of representation learning of geodesic transformations (Eq. (3)), followed by the latent geodesic difusion model (Eq. (6)). The training process of IGG is summarized in $\mathsf { A l g . }$ 1. During testing, the procedure for sampling the geodesic of latent velocity fields, conditioned on a given template image and text instructions, is outlined in Alg. 2.

Algorithm 1 IGG Training   
Input: Data $\{ s ^ { n } , f ^ { n } , \Lambda ^ { n } \} _ { n = 1 } ^ { N _ { \mathrm { t r a i n } } }$   
1: Pretrain $\mathcal { E } _ { \theta _ { v } } , \mathcal { E } _ { \theta _ { r } } , \mathcal { D } _ { \theta _ { v } }$ via Eq. (3)   
2: repeat   
3: Initial latent geodesic representations, $z _ { 0 } ^ { n } \gets$   
$\mathcal { G } _ { \theta _ { r } } \Big ( \mathcal { E } _ { \theta _ { v } } ( s ^ { n } , f ^ { n } ) \Big )$   
4: Sample $\tau \sim \stackrel { \cdot } { \mathrm { U } } ( 1 , \mathrm { T } ) , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$   
5: Sample $z _ { \tau } ^ { n }$ at $\tau$ by $z _ { \tau } ^ { n }  q ( z _ { \tau } ^ { n } \mid z _ { 0 } ^ { n } )$ via Eq. (4),   
6: Optimize $\mathbf { Z } _ { \theta }$ by taking a gradient descent step on   
Eq. (6).   
7: until converged   
Algorithm 2 IGG Sampling   
Input: Data $\{ s ^ { n } , \Lambda ^ { n } \} _ { n = 1 } ^ { N _ { \mathrm { t e s t } } }$   
Output: Geodesic deformations $\Phi ^ { n }$   
1: Initialize $z _ { T } ^ { n } \sim \mathcal { N } ( 0 , \mathbf { I } )$   
2: for $\tau = \mathrm { T } , \dots , 1$ do   
3: $\hat { z } _ { \tau - 1 } ^ { n } \gets p _ { \boldsymbol { \theta } } \big ( z _ { \tau - 1 } ^ { n } \mid z _ { \tau } ^ { n } , \nabla s ^ { n } , \Lambda ^ { n } , \tau \big )$ via Eq. (5),   
4: end for   
5: $\{ \hat { v } ^ { n } \} \gets \mathcal { D } _ { \theta _ { v } } ( \hat { z } _ { 0 } ^ { n } ) ,$   
6: $\Phi ^ { n }  \{ \hat { v } ^ { n } \}$ via Eq. (1),   
7: return $\Phi ^ { n }$

## 4. Experimental Evaluation

We evaluate the proposed IGG framework using a diverse set of real-world image datasets that capture deformable shape changes over time. We first assess the quality of the learned latent representations of geodesics by comparing them to numerical solutions of the EPDif equation obtained via Euler integration.

Next, we evaluate the quality of the generated images deformed by the sampled geodesics guided by given text instructions. These results are compared with state-of-the-art generative models that synthesize image sequences/videos with publicly available training code or fine-tuning options, including the video difusion model (VDM) (Ho et al., 2022), CogVideoX (Yang et al., 2024), and DynamiCrafter (Xing et al., 2025). Note that all baselines are trained using sequences of deformed images generated along the geodesic transformations derived from the numerical solutions of EPDif equation. This approach ensures that the results are independent of the learned geodesics predicted by the first module of the IGG framework, providing a consistent and unbiased evaluation.

Finally, to further assess the quality of the generated samples and demonstrate their clinical utility in real-world applications, we design a downstream task in the context of brain MRI segmentation. More specifically, we use the synthesized image samples as augmentation data by pairing each transformed image with its corresponding segmentation label, both generated through the learned deformation fields applied to existing images and labels. We conduct experiments using a diverse set of widely used segmentation architectures, including U-Net (Ronneberger et al., 2015), TransUNet (Chen et al., 2021), and Attention U-Net (AttUNet) (Oktay et al., 2018), under limited training data conditions.

## 4.1 Dataset

Komatsuna plant. We include 300 frames of RGB-D label-maps representing five Komatsuna plants from the publicly available data repository (Uchiyama et al., 2017). The label maps cover diferent leaves that emerge from the bud and grow in size over time, where the plant growth was monitored between 228-236 hours. All data frames were resampled to to 128<sup>2</sup> and pre-aligned with afine transformations.

Longitudinal brain MRI. We include a total of 2618 T1-weighted longitudinal (time-series) brain MRIs sourced from the Open Access Series of Imaging Studies (OASIS-3) dataset (LaMontagne et al., 2019). This experiment aims to validate our method using longitudinal data that includes scans at varying time intervals for individuals spanning both healthy subjects and Alzheimer’s diseases (AD), aged 60-90. Given the scenario that many existing image generation/editing methods with text instructions focus on 2D natural images (Brooks et al., 2023; Meng et al., 2021), we specifically utilize 2D scans derived from this 3D brain data for comparison with state-of-the-art baselines. All MRIs undergo pre-processing, including resizing to 128<sup>2</sup>, with isotropic voxels of 1mm<sup>2</sup>, skull-stripping, intensity normalization, bias field correction, and pre-alignment using afine transformations.

Text instructions. Our text condition consists of a description of the template image, followed by details of the specific edits or progressions applied to it. For brain MRIs, this includes biological variables such as age, sex, and gender. In contrast, for plants, it primarily focuses on the growth timeline. All information is sourced directly from the original dataset available in the data repository.

## 4.2 Experimental Design and Implementation Details

Evaluate learned latent representations of geodesics via GIR. We evaluate GIR’s ability to predict geodesic dynamics by comparing the learned geodesics with numerical solutions to the EPDif equation using Euler integration. The assessment focuses on errors in velocity fields, transformations, and deformed images between the two approaches. Quantitative metrics, such as the mean absolute error of velocity fields, are computed at each time integration step.

Evaluate generated samples. We compare the quality of samples generated by IGG with three baseline models: VDM (Ho et al., 2022), CogVideoX (Yang et al., 2024), and DynamiCrafter (Xing et al., 2025)). In particular, we first assess individual synthesized images at each time step using standard image synthesis metrics, including the Fr´echet Inception Distance (FID) (Heusel et al., 2017) and Kernel Inception Distance (KID) (Binkowski et al., 2018) to´ measure the distributional similarity between generated and real images. Additionally, we use the Structural Similarity Index (SSIM) (Wang et al., 2004) to evaluate structural, contrast, and luminance similarities. We then evaluate the continuity and smoothness of generated video sequences by computing the Fr´echet Video Distance (FVD) (Unterthiner et al., 2018) and providing qualitative visual comparisons. Finally, we assess the perceptual quality of the generated samples using the Inception Score (IS) (Salimans et al., 2016).

Evaluate the preservation of object geometry and topology. To demonstrate the eficiency of our IGG model in preserving fine-grained geometric structure and topology, we first generate image sequences by deforming a template image using the sampled geodesic flow of deformations produced by IGG and compare these results with all baseline models. Secondly, we present the corresponding determinant of the Jacobian (DetJac) maps for the deformations. The DetJac values reveal important patterns of volume change: a value of 1 indicates no volume change, DetJac < 1 reflects volume shrinkage, and DetJac > 1 implies volume expansion. A DetJac value below zero indicates artifacts or singularities in the transformation field, highlighting a failure to maintain the topological integrity. Note that only our IGG model enables such a metric to quantify the topological changes in the generated samples.

Evaluate reliability of model predictions. We evaluate the reliability and confidence of IGG in learning the geodesic deformations of geometric shapes over time by computing pixel-wise mean and standard deviation for individual sampled image sequences along the geodesic path. We visualize confidence intervals to highlight regions with 95% certainty in growth patterns. These intervals are defined by the mean of 1000 generated samples with pixel-wise bounds that are two standard deviations above and below the mean.

Demonstrate utility of generated samples in downstream segmentation tasks. We evaluate the efectiveness of IGG generated images by using them as augmented data for segmentation tasks under conditions of limited number of training samples. We then compare the performance of segmentation models trained with and without our synthesized augmented data, considering dataset augmentations of 2× and 4× the size of the original dataset. The segmentation accuracy is assessed using the dice score (Dice, 1945), both averaged across all anatomical regions and separately for diferent structures. To highlight the flexibility of IGG in generating new data, we evaluate augmentation performance on OASIS-3 (LaMontagne et al., 2019) and OASIS-1 (Marcus et al., 2007) datasets independently, noting that our model is trained exclusively on OASIS-3. This setup allows us to examine the generalization capability of the synthesized data to an unseen dataset while quantifying its impact on segmentation performance.

Parameter Setting. We split all datasets into 80%, 10%, and 10% for training, validation, and testing. All experiments are conducted on NVIDIA A100 GPUs. For training the autoencoder for geodesic learning, we alternatively update the encoder-decoder and the latent geodesic neural operator, and finally jointly train both components for 2000 epochs per stage. We set the weight decay as $1 e ^ { - 4 }$ , batch size as 64, and a learning rate as $5 e ^ { - 4 }$ . For training the latent geodesic difusion model, we set the learning rate as $1 e ^ { - 4 }$ , the number of epochs as 6000, batch size as 36, and difusion steps as 500. All models are trained using the Adam optimizer (Kingma and Ba, 2014).

## 4.3 Experimental Results

The left panel of Fig. 3 visualizes the predicted transformations, velocity fields, and deformed images from IGG’s geodesic-informed registration network (GIR), compared with numerical solutions to the EPDif equation (Eq. (2)). These methods show a high degree of similarity. The right panel of Fig. 3 presents quantitative results of the absolute per-pixel error, computed between the velocity field obtained from the numerical solution of the EPDif equation and the velocity field predicted by the GIR module at each time step. These results collectively highlight GIR’s ability to efectively learn geodesic mapping functions comparable to real numerical solutions.

We also report the additional computational overhead introduced by the GIR, which primarily comes from the

![](images/b6621fe0b3e738c9d658425aeeb2e65253d73a9daee4587e51b288249950886a.jpg)  
Figure 3: Comparison of predicted geodesics by GIR (the first sub-module of IGG) vs. real numerical solutions from the EPDif equation. Left: Visualization of predicted deformed images, deformations, and velocities along time. Right: Mean absolute error of predicted velocities over time compared to numerical integration of EPDif.

Template

Target

5 <sub>.</sub> 0

0 <sub>.</sub> 1

Template

![](images/772f5dc580e261b2dcf6d0efb7386853a87a6ea15b257a8094ec133b8426d4b5.jpg)

![](images/45edd993e04ca525672ac5e2bb4433be0be0964d322b46b08bd547ede10f2caf.jpg)

![](images/4cd1b689e17069c3a6b3d77627d4c4948063af020c049d64c2091f78ef83b6f0.jpg)  
Target

![](images/ab1fa2b9792dd424a9f1e4c9497887c20488c15792deace92a3a69139cd9b691.jpg)

![](images/e74ace8da233ffcc50be1732a67a14acc2c9bfc9a746504ca25620e127bfeb80.jpg)  
Image of a 70 years old male who is cognitively normal. How will it be at the age of 77 years?  
1 <sub>.</sub> 0

Figure 4: A comparison of images generated by IGG (with corresponding DetJac) against all baseline models across diferent time frames. Given an input template image and text instructions, all models generate samples of target images. The ground truth ”target” along with input template images are provided on the left side of the panel for reference.

latent neural operator used to approximate the EPDif equation (Eq. (2)). In practice, in our experiments on 128<sup>2</sup> brain images, the training time per epoch is approximately 13.10s for GIR, compared to 4.81s for a standard registration model with numerical solutions to the EPDif Hinkle et al. (2018). The corresponding inference times are 0.01s vs. 0.004s, respectively.

Fig. 4 compares the generated images from our model IGG (along with the associated DetJac maps of the deformations) with the baseline generative models. The visualized images and DetJac values of IGG suggest that our model efectively preserves the topological structure of objects within the generated images with well-captured progression of geometric shape changes over time. In contrast, samples generated by the baselines fail to maintain the geometric integrity of various structures. For instance, in plant growth images, the leaves appear to merge or overlap unnaturally, disrupting their biological topology. Similarly, the baselines inaccurately predict parts of the ventricles, resulting in regions that do not correspond to the original brain structure. It is worth noting that the baseline methods perform difusion directly in the image intensity space, learning the distribution of raw pixel values. During sampling, they generate pixel intensities through iterative denoising, which can lead to blurring (or noisy) artifacts observed in difusion models operating in pixel space. In contrast, our method performs difusion in the deformation space, generating velocity or deformation fields that are subsequently applied to a template image to synthesize new image sequences. By modeling structural variation through geometric transformations rather than pixel-wise generation, our approach better preserves sharp and anatomically coherent structures.

Tab. 1 reports the evaluation metrics of sample quality and diversity for the generated images from our model IGG and the baselines. While all models achieve similar IS, IGG shows significantly lower FVD, FID, and KID scores, and higher SSIM scores. This indicates that while all models generate diverse images (leading to a good IS), the baselines lack realistic features or proper distribution alignment with real images. The observation of IGG achieving approximately 10 times better scores demonstrates its ability to generate more realistic images.

Fig. 5 visualizes examples of confidence maps of plant growth and brain progression images generated from IGG. It suggests that our model efectively captures the growth pattern of plants over time, primarily focusing on the boundaries of the leaves. The confidence maps of brain images highlight expanding patterns in ventricles, demonstrating the model’s ability to capture dynamic changes over time.

Fig. 6 presents the improvement in average dice scores for diferent segmentation models trained with varying dataset configurations. The baseline corresponds to training on the original limited dataset. It shows that with gradually increasing the amount of augmented data generated by IGG from 2× to 4× the size of the original training set, we can consistently obtain better segmentation performance with higher dice scores across all models. These results demonstrate that the generated image-label pairs are able to efectively augment real data, particularly under limitedlabel conditions. While the improvements on OASIS-1 (with maximal +5.1%) are smaller than those on OASIS-3 (with maximal +11.0%), they remain consistent, highlighting the generalizability of our model across datasets.

Fig. 7 further reports detailed dice scores across diferent anatomical structures when using IGG’s predictions as augmented data on both OASIS-3 and OASIS-1 datasets.

Fig. 8 visualizes representative segmentation maps predicted by a trained U-Net (Ronneberger et al., 2015) in diferent configurations of the training dataset. When trained on a limited dataset, and the model produces errors in certain anatomical structures, whereas training with augmented data yields predictions that more closely match the ground truth. These results further confirm the efectiveness of incorporating our generated samples, especially under limited-data conditions.

Table 1: A comparison of performance metrics across all methods for various datasets.
<table><tr><td>Dataset</td><td>Model</td><td>FVD ↓</td><td>FID↓</td><td>KID ↓</td><td>SSIM ↑</td><td>IS ↑</td></tr><tr><td rowspan="4">Plant</td><td>IGG</td><td>48.29</td><td>9.23</td><td> $\mathbf { 0 . 0 0 7 \pm 0 . 0 2 }$ </td><td>0.98 ± 0.02</td><td> $\pm 1 . 0 6 \pm 1 . 2 \%$ </td></tr><tr><td>VDM (Ho et al., 2022)</td><td>357.56</td><td>71.79</td><td> $0 . 2 3 \pm 0 . 0 3$ </td><td> $0 . 2 3 \pm 0 . 1 0$ </td><td> $1 . 0 5 \pm 1 . 0 \%$ </td></tr><tr><td>CogVideoX (Yang et al., 2024)</td><td>471.49</td><td>153.93</td><td> $0 . 6 3 \pm 0 . 0 7$ </td><td> $0 . 6 9 \pm 0 . 1 9$ </td><td> $1 . 0 5 \pm 0 . 6 \%$ </td></tr><tr><td>DynamiCrafter (Xing et al., 2025)</td><td>466.70</td><td>80.78</td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td> $0 . 8 9 \pm 0 . 0 4$ </td><td> $1 . 0 3 \pm 0 . 3 \%$ </td></tr><tr><td rowspan="4">Brain</td><td>IGG</td><td>26.39</td><td>6.23</td><td>0.006 ± 0.008</td><td>0.97 ± 0.03</td><td> $1 . 0 2 \pm 0 . 0 8 \%$ </td></tr><tr><td>VDM (Ho et al., 2022)</td><td>247.84</td><td>69.23</td><td> $0 . 4 6 \pm 0 . 0 3$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td> $1 . 0 2 \pm 0 . 0 7 \%$ </td></tr><tr><td>CogVideoX (Yang et al., 2024)</td><td>302.20</td><td>114.50</td><td> $0 . 4 6 \pm 0 . 0 4$ </td><td> $0 . 6 9 \pm 0 . 2 7$ </td><td> $\pm . 0 6 \pm 2 . 8 \%$ </td></tr><tr><td>DynamiCrafter (Xing et al., 2025)</td><td>288.62</td><td>142.40</td><td> $1 . 0 5 \pm 0 . 0 3$ </td><td> $0 . 8 8 \pm 0 . 0 3$ </td><td> $1 . 0 3 \pm 0 . 2 0 \%$ </td></tr></table>

Given is a plant at 164 hours. What would the plant look like at 172 hours?

![](images/881e9e5ce9bf1d0a75beb695c5b2fd4517070f6c4679e7a142bbef00ec218adf.jpg)  
Image of a 72 years old female who is cognitively normal. How will it be at t he age of 83 years?

![](images/5f494d74747ff0f46c109c6b1a3468ef42b7d3927f7c4e7741831d8a6559ab4c.jpg)  
Figure 5: Top: Input template images with text instructions. Bottom: Confidence maps illustrating the lower bounds (LB), upper bounds (UB), and confidence intervals (CI), which represent regions representing 95% of ideal growth patterns, based on 1000 samples generated by our IGG model across diferent time frames.

## 5. Conclusion & Discussion

This paper presents an extended work, IGG (Wu et al., 2025), a geodesic-informed generative difusion model for topology-preserved image video generation. In contrast to existing generative models that focus primarily on manipulating image intensity and texture, IGG explicitly learns the underlying latent distribution of geodesic-informed deformations from images, enabling the synthesis of diverse samples within geodesic deformation spaces. Our work is the first generative model to operate on natural and interpretable image transformations: it not only captures shape-changing processes during generation, but also provides quantitative metrics to assess topological consistency in the synthesized samples. The key contributions of IGG include (i) a geodesic-informed registration network that learns latent representations of time-dependent difeomorphic transformations, enabling accurate modeling of complex shape dynamics; and (ii) a latent geometric difusion model that captures sequential deformation features conditioned on text inputs and template images, allowing controlled and semantically meaningful generation. Our experimental results demonstrate that IGG significantly improves sample fidelity and diversity compared to state-of-the-art methods. Additionally, downstream segmentation experiments show substantial performance gains when using IGG-generated samples as data augmentation, particularly in low-data regimes.

![](images/de3d4a92eea201a952952e225b2974f691beabab2c95b191b3e9690bdf7b2d69.jpg)  
Figure 6: The improvement in average dice scores achieved by diferent segmentation models when trained with varying training dataset configurations. Baseline denotes training with the original limited dataset, while 2× and 4× correspond to augmentations of 2 times and 4 times the size of the original dataset, respectively.

![](images/5a0e236c68603ee2d9658d0bf22da6185a2b48349cd1bb4c8805845e0f4f29d2.jpg)  
Figure 7: A comparison of average dice scores across diferent anatomical structures under varying training dataset configurations. Left to right: base: training with the original limited dataset, and 2× and 4× correspond to augmented sizes of the original dataset. Top to bottom: results on the OASIS-3 vs. OASIS-1.

![](images/3554896090a54d33d668595531f4a932182afbd0fa39e0050ee562faf666e187.jpg)  
Figure 8: Examples of segmentation results on brain MRIs across diferent anatomical structures under varying training dataset configurations. Left to right: original MRI scans, ground truth labels, predicted labels from a mode trained on the original limited dataset, and predicted labels from models trained with 2× and 4× the size of the original dataset. Top to bottom: results on the OASIS-3 vs. OASIS-1.

Our proposed IGG model advances image generation by integrating geodesic principles, providing tools for topology assessment, and establishing a framework for synthesizing anatomically consistent image samples. We acknowledge that the fidelity of the synthesized data from our IGG model is inherently dependent on the accuracy of the underlying registration. When registration errors are large, the quality of the synthesized results may be correspondingly limited. This dependency reflects a critical assumption of our framework that anatomically meaningful correspondences can be reliably established. Our topological prior is most efective in applications where anatomical structures are expected to preserve difeomorphic consistency. However, in practical settings, imaging noise, pathology-induced appearance changes, or other non-difeomorphic structural changes may violate this assumption. In such cases, strict topology preservation may not fully capture the underlying anatomical variability. Our future work will be investigated to combine IGG’s strength in learning deformation fields with the modeling of image texture distributions, which will enable the development of more powerful and realistic generative models that capture both structural and appearance variations. Other future research will focus on several potential directions in extending IGG to handle 4D data (3D + time) to model dynamic longitudinal anatomical changes, such as cardiac motion or brain changes over time. These directions aim to further enhance the clinical and scientific utility of generative modeling in medical imaging.

## Acknowledgments

This work was supported by NSF CAREER Grant 2239977.

## Ethical Standards

The work follows appropriate ethical standards in conducting research and writing the manuscript, following all applicable laws and regulations regarding treatment of animals or human subjects.

## Conflicts of Interest

We declare we do not have conflicts of interest.

## Data availability

The Brain MRIs used in training and testing in this paper are from the Open Access Series of Imaging Studies (OASIS-3) dataset (LaMontagne et al., 2019), which are readily accessible and user-friendly. Readers interested in evaluating the accuracy of the method can access and utilize the publicly available datasets. The code for this study is also publicly available at https://github.com/nellie689/IGG.

## References

Vladimir Arnold. Sur la g´eom´etrie dif´erentielle des groupes de lie de dimension infinie et ses applications \`a l’hydrodynamique des fluides parfaits. In Annales de l’institut Fourier, volume 16, pages 319–361, 1966.

Brian B Avants, Charles L Epstein, Murray Grossman, and James C Gee. Symmetric difeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical image analysis, 12(1):26–41, 2008.

Shekoofeh Azizi, Simon Kornblith, Chitwan Saharia, Mohammad Norouzi, and David J Fleet. Synthetic data from difusion models improves imagenet classification. arXiv:2304.08466, 2023.

Duhyeon Bang and Hyunjung Shim. Mggan: Solving mode collapse using manifold-guided training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2347–2356, 2021.

Mirza Faisal Beg, Michael I Miller, Alain Trouv´e, and Laurent Younes. Computing large deformation metric mappings via geodesic flows of difeomorphisms. International journal of computer vision, 61(2):139–157, 2005.

Miko laj Binkowski, Danica J Sutherland, Michael Ar-´ bel, and Arthur Gretton. Demystifying mmd gans. arXiv:1801.01401, 2018.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392– 18402, 2023.

Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L Yuille, and Yuyin Zhou. Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306, 2021.

Duy-Phuong Dao, Hyung-Jeong Yang, and Jahae Kim. Conditional difusion model for longitudinal medical image generation. arXiv preprint arXiv:2411.05860, 2024.

Swakshar Deb, Nian Wu, Frederick H Epstein, and Miaomiao Zhang. Unsupervised cardiac video translation via motion feature guided difusion model. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 648–658. Springer, 2025.

Lee R Dice. Measures of the amount of ecologic association between species. Ecology, 26(3):297–302, 1945.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Robert Graf, Joachim Schmitt, Sarah Schlaeger, Hendrik Kristian M¨oller, Vasiliki Sideri-Lampretsa, Anjany

Sekuboyina, Sandro Manuel Krieg, Benedikt Wiestler, Bjoern Menze, Daniel Rueckert, et al. Denoising difusionbased mri to ct image translation enables automated spinal segmentation. European Radiology Experimental, 7(1):70, 2023.

Saumya Gupta, Dimitris Samaras, and Chao Chen. Topodifusionnet: A topology-aware difusion model. arXiv:2410.16646, 2024.

Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

J. Hinkle, D. Womble, and H.J.d Yoon. Difeomorphic autoencoders for LDDMM atlas building. In Medical Imaging with Deep Learning, 2018.

Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video difusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.

Tonmoy Hossain and Miaomiao Zhang. Mgaug: Multimodal geometric augmentation in latent spaces of image deformations. Medical Image Analysis, 102:103540, 2025.

Jiangbei Hu, Ben Fei, Baixin Xu, Fei Hou, Weidong Yang, Shengfa Wang, Na Lei, Chen Qian, and Ying He. Topology-aware latent difusion for 3d shape generation. arXiv:2401.17603, 2024.

Nivetha Jayakumar, Tonmoy Hossain, and Miaomiao Zhang. Sadir: shape-aware difusion models for 3d image reconstruction. In International workshop on shape in medical imaging, pages 287–300. Springer, 2023.

Nivetha Jayakumar, Srivardhan Reddy Gadila, Tonmoy Hossain, Yangfeng Ji, and Miaomiao Zhang. Tpie: Topologypreserved image editing with text instructions. arXiv preprint arXiv:2411.16714, 2024.

Sarang Joshi, Brad Davis, Matthieu Jomier, and Guido Gerig. Unbiased difeomorphic atlas construction for computational anatomy. NeuroImage, 23:S151–S160, 2004.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.

Boah Kim, Inhwa Han, and Jong Chul Ye. Difusemorph: Unsupervised deformable image registration using diffusion model. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXI, pages 347–364. Springer, 2022.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Pamela J LaMontagne, Tammie LS Benzinger, John C Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei G Vlassenko, et al. Oasis-3: longitudinal neuroimaging, clinical, and cognitive dataset for normal aging and alzheimer disease. medrxiv, pages 2019–12, 2019.

Dongxu Li, Junnan Li, and Steven Hoi. Blip-difusion: Pre-trained subject representation for controllable textto-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024.

Wei Li, Li Fan, Zhenyu Wang, Chao Ma, and Xiaohui Cui. Tackling mode collapse in multi-generator gans with orthogonal vectors. Pattern Recognition, 110:107646, 2021.

Xiaofeng Liu, Fangxu Xing, Maureen Stone, Jiachen Zhuo, Timothy Reese, Jerry L Prince, Georges El Fakhri, and Jonghye Woo. Generative self-training for crossdomain unsupervised tagged-to-cine mri synthesis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 138–148. Springer, 2021.

Daniel S Marcus, Tracy H Wang, Jamie Parker, John G Csernansky, John C Morris, and Randy L Buckner. Open access series of imaging studies (oasis): cross-sectional mri data in young, middle aged, nondemented, and demented older adults. Journal of cognitive neuroscience, 19(9):1498–1507, 2007.

Franc¸ois Maz´e and Faez Ahmed. Difusion models beat gans on topology optimization. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 9108–9116, 2023.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic diferential equations. arXiv:2108.01073, 2021.

Michael I Miller, Alain Trouv´e, and Laurent Younes. On the metrics and euler–lagrange equations of computational anatomy. Annual Review of Biomedical Engineering, 4 (1):375–405, 2002.

Soumik Mukhopadhyay, Matthew Gwilliam, Vatsal Agarwal, Namitha Padmanabhan, Archana Swaminathan, Srinidhi Hegde, Tianyi Zhou, and Abhinav Shrivastava. Difusion models beat gans on image classification, 2023.

Thao Nguyen, Yuheng Li, Utkarsh Ojha, and Yong Jae Lee. Visual instruction inversion: Image editing via image prompting. Advances in Neural Information Processing Systems, 36, 2024.

Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, et al. Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999, 2018.

Muzafer Ozbey, Onat Dalmaz, Salman UH Dar, Hasan A<sup>¨</sup> Bedel, S¸aban Ozturk, Alper G <sup>¨</sup> ung ¨ ¨or, and Tolga Cukur. Unsupervised medical image translation with adversarial difusion models. IEEE Transactions on Medical Imaging, 42(12):3524–3539, 2023.

Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23051–23061, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.

Jacob C Reinhold, Aaron Carass, and Jerry L Prince. A structural causal model for mr images of multiple sclerosis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 782–792. Springer, 2021.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Gulshan Sharma, Abhinav Dhall, and Ramanathan Subramanian. Medic: Mitigating eeg data scarcity via classconditioned difusion model. In Deep Generative Models for Health Workshop NeurIPS 2023, 2023.

Sophie Starck, Vasiliki Sideri-Lampretsa, Bernhard Kainz, Martin J Menten, Tamara T Mueller, and Daniel Rueckert. Dif-def: Difusion-generated deformation fields for conditional atlases. IEEE Transactions on Medical Imaging, 2025.

Kerem C Tezcan, Christian F Baumgartner, Roger Luechinger, Klaas P Pruessmann, and Ender Konukoglu. Mr image reconstruction using deep density priors. IEEE transactions on medical imaging, 38(7):1633–1642, 2018.

Hideaki Uchiyama, Shunsuke Sakurai, Masashi Mishima, Daisaku Arita, Takashi Okayasu, Atsushi Shimada, and Rin-ichiro Taniguchi. An easy-to-setup 3d phenotyping platform for komatsuna dataset. In Proceedings of the IEEE international conference on computer vision workshops, pages 2038–2045, 2017.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv:1812.01717, 2018.

Franc¸ois X Vialard, Laurent Risser, Daniel Rueckert, and Colin J Cotter. Difeomorphic 3d image registration via geodesic shooting using an eficient adjoint calculation. International Journal of Computer Vision, 97(2):229–241, 2012.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Nian Wu and Miaomiao Zhang. Neurepdif: Neural operators to predict geodesics in deformation spaces. In International Conference on Information Processing in Medical Imaging, pages 588–600. Springer, 2023.

Nian Wu and Miaomiao Zhang. Learning geodesics of geometric shape deformations from images. The Journal of Machine Learning for Biomedical Imaging, 2024.

Nian Wu, Jiarui Xing, and Miaomiao Zhang. Tlrn: Temporal latent residual networks for large deformation image registration. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 728–738. Springer, 2024.

Nian Wu, Nivetha Jayakumar, Jiarui Xing, and Miaomiao Zhang. Igg: Image generation informed by geodesic dynamics in deformation spaces. In International Conference on Information Processing in Medical Imaging, pages 232–246. Springer, 2025.

Jianhao Xie, Ziang Zhang, Zhenyu Weng, Yuesheng Zhu, and Guibo Luo. Meddif-ft: Data-eficient difusion model fine-tuning with structural guidance for controllable medical image synthesis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 306–316. Springer, 2025.

Jiarui Xing, Nivetha Jayakumar, Nian Wu, Yu Wang, Frederick H Epstein, and Miaomiao Zhang. Lamod: Latent motion difusion model for myocardial strain generation. In International Workshop on Shape in Medical Imaging, pages 164–177. Springer, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video difusion priors. In European Conference on Computer Vision, pages 399– 417. Springer, 2025.

Junlin Yang, Nicha C Dvornek, Fan Zhang, Julius Chapiro, MingDe Lin, and James S Duncan. Unsupervised domain adaptation via disentangled representations: Application to cross-modality liver segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 255–263. Springer, 2019.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xi aohan Zhang, Guanyu Feng, et al. Cogvideox: Textto-video difusion models with an expert transformer. arXiv:2408.06072, 2024.

Jee Seok Yoon, Chenghao Zhang, Heung-Il Suk, Jia Guo, and Xiaoxiao Li. Sadm: Sequence-aware difusion model for longitudinal medical image generation. In International Conference on Information Processing in Medical Imaging, pages 388–400. Springer, 2023.

Laurent Younes, Felipe Arrate, and Michael I Miller. Evolutions equations in computational anatomy. NeuroImage, 45(1):S40–S50, 2009.

Zhengming Yu, Zhiyang Dou, Xiaoxiao Long, Cheng Lin, Zekun Li, Yuan Liu, Norman Muller, Taku Komura, Marc¨ Habermann, Christian Theobalt, et al. Surf-d: Generating high-quality surfaces of arbitrary topologies using difusion models. In European Conference on Computer Vision, pages 419–438. Springer, 2025.

Kai Zhang, Geng Chen, Shijie Huang, Fangmei Zhu, Zhongxiang Ding, and Dinggang Shen. Developmentdriven difusion model for longitudinal prediction of fetal brain mri with unpaired data. IEEE Transactions on Medical Imaging, 2024.

Miaomiao Zhang, Ruizhi Liao, Adrian V Dalca, Esra A Turk, Jie Luo, P Ellen Grant, and Polina Golland. Frequency difeomorphisms for eficient image registration. In International Conference on Information Processing in Medical Imaging, pages 559–570. Springer, 2017.

Zhaoyu Zhang, Mengyan Li, and Jun Yu. On the convergence and mode collapse of gan. In SIGGRAPH Asia 2018 Technical Briefs, pages 1–4. 2018.