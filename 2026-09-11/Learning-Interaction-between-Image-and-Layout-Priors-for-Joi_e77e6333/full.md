# Learning Interaction between Image and Layout Priors for Joint Image-Layout Generation in Design Templates

Shirong Yang, Bo Yang, and Ying Cao

Abstract—In this paper, we address the problem of graphic design template creation, which generates a background image and a layout of foreground elements over the background to form a harmonious composition from an input text. Prior work on graphic design generation mostly adopts a sequential paradigm, where design elements are generated sequentially. We argue that such a sequential scheme falls short of faithfully capturing the dependency between the background and layout (and thus the joint image-layout distribution), which limits the quality of generated design templates. To overcome this limitation, we propose a model, InterIL, which jointly generates the two modalities—background image and layout—in a single generative process. The novel design of our joint model connects the backbones of pretrained image and layout diffusion models with a learnable communication module to explicitly model bidirectional image-layout interaction. During training, the image and layout backbones are frozen to maintain and leverage the vast pretrained single-modality prior knowledge, while only the communication module is updated, so that the model can focus on learning image-layout interaction and thereby better capture the joint image-layout distribution for improved composition harmony. Our model has no design-specific inductive bias, which allows it to better preserve the original characteristics of realistic designs. We further introduce a test-time guidance strategy to enable users to impose their specific preferences on generated results. Our experiments show that, compared with prior approaches, our model can generate significantly better results in terms of image, layout and image-layout harmonization, producing outputs closer to real samples. We also demonstrate the flexibility of our model in enforcing user preferences at inference without retraining.

Index Terms—Design template generation, diffusion models, layout generation.

## 1 INTRODUCTION

choose and start from pre-existing design templates, instead of designing from scratch, and iteratively refine the templates towards the final full designs. Design templates are of great value to designers in practice, which can provide inspirational ideas and help speed up the design process. However, creating good design templates is non-trivial, which often takes substantial manual effort and necessitates considerable design expertise.

In this paper, we aim to automate the creation of graphic design templates from input textual descriptions. Ideally, design templates should lie midway between high-level requirements and finalized designs, serving as a mid-level abstraction of complete designs: they are more concrete than high-level intentions with overall theme and structure to serve as good starting points, while being less complete than full designs with only a partial set of element attributes to leave room for creativity. In this work, we focus on a specific form of design templates, which is composed of a background image and a layout (spatial arrangement) of foreground elements over the background [1].

Despite remarkable progress in generative models for graphic design, previous methods mostly follow a sequential generation paradigm, where design elements are created one after another. Cascaded pipelines [2], [3] execute a series of sub-tasks (e.g., background generation, object generation and typography generation) in sequence. The causal nature of the autoregressive design generation models [4]– [6] imposes a dependency order between generated elements. A recent work on design template generation [1] takes a sequential two-stage approach, where a background is first generated, followed by generating a layout conditioned on the background. However, in design templates, the dependency between background and layout is bidirectional—background generation should take the layout into account (e.g., to allocate appropriate space for foreground element placement); layout generation should be aware of the background (e.g., to not occlude important background regions). As a result, the existing methods are limited in modeling desired background-layout interaction, which is crucial for harmonious compositions, and thus lead to degraded generation quality (e.g., visual conflict between background and layout).

To mitigate this issue, we propose InterIL, a latent diffusion model for design template generation, which enables joint generation of images and layouts in a single framework by learning image-layout interaction. Specifically, we first pretrain a latent diffusion model for unconditional layout generation as a layout prior, and leverage a pretrained text-to-image latent diffusion model [7] as an image prior. Each of the two priors alone can generate high-quality samples in its own domain (image or layout). We then construct the denoising network of InterIL by combining

![](images/1a62d8ad879000da7613d11eff6e00905447bcd924523c3b33c5ba4346f09558.jpg)  
A high-detail Gundam standing in fog, minimal background.

![](images/e09b07746be126024f0a4097bd3e90586050f182c54cafe3a17741ef3e522ce9.jpg)  
An autumn road framed on a rustic brick wall.

![](images/7bc3de9034621ca3b3d8d3adc03c978a5a7ea1095493985f5e1ef970a687e92c.jpg)  
A cat sitting on a windowsill with soft morning light.

![](images/c45873509178646ab3caf96a58bd8c67a95a6a0d8abb5762647ec26f53fa9fac.jpg)  
A vintage car parked on an empty road under soft daylight.

Fig. 1. Generated design templates by our model given text inputs. Each template specifies the overall theme and structure of the design, consisting of a background image and a layout of elements of different types: text (white), underlay (yellow), and button (red). The input text is shown below each generated template.

the pretrained backbones of the two priors and connecting them with a communication module dedicated to explicitly learning bidirectional image-layout interaction. When training InterIL, we keep the pretrained weights of the two backbones fixed, and only optimize the communication module. This allows us to focus learning on capturing image-layout interaction patterns, as the prior knowledge of generating realistic images and layouts is already available in the two backbones. With the learned communication module, InterIL can generate a holistic design template with a single denoising process, through which the image and layout backbones frequently communicate with each other to denoise noisy image and layout latents jointly. Due to the joint generation strategy and the explicit imagelayout interaction learning, our model is able to capture the joint image-layout distribution more faithfully than existing sequential methods. Moreover, unlike the previous work [1], our model refrains from using any design-related inductive bias, and therefore better fits the training data distribution, reproducing the unique characteristics involved in realistic designs. Our model also offers flexibility in deviating from realistic design characteristics and shifting generations towards user-preferred patterns (e.g., minimizing occlusion of salient background areas), through a proposed training-free guidance strategy.

To validate the effectiveness of our InterIL model, we introduce two evaluation metrics, TemplateFID and TemplateCLIP, which respectively assess the visual quality and text adherence of generated templates. We conduct experiments on the Web-design dataset [1], comparing InterIL with alternative approaches. Our model exhibits dramatically improved performance compared to the other methods, generating high-quality, visually harmonious design templates (see Figure 1). We also showcase how our proposed guidance technique can be applied at inference time to allow users to control generated templates through their expressed preferences, including reducing foregroundelement occlusion of salient background regions and improving text readability.

Our paper makes the following contributions:

• A new joint image-layout generation paradigm for graphic design template generation, which differs from the existing sequential generation scheme.

• A diffusion-based design template generation framework, which synthesizes images and layouts simultaneously in a single generation process by learning the interaction between the pretrained image and layout priors, which is in contrast to the sequential, multi-stage framework used in previous works.

• An inference-time guidance framework that can accommodate different user preferences on generated results without any model retraining.

## 2 RELATED WORK

## 2.1 Graphic Layout Generation

Graphic layout generation, which involves creating the spatial arrangement of elements on a canvas, has been extensively studied [8]–[11] Recent works have approached layout generation using various deep generative modeling frameworks, including Generative Adversarial Networks (GANs) [12]–[14], Variational Autoencoders (VAEs) [15], [16], Transformers [17]–[19], diffusion models [20]–[22], and flow-based models [23]. Some works also show the effectiveness of large language models (LLMs) in solving the layout generation task based on the layout-related knowledge that LLMs have acquired during pretraining [24]. Constrained layout generation has also been investigated, which imposes various user constraints on element attributes and relationships to control generated layouts [14], [19], [25]. Several recent methods investigate graphic design composition, in which a set of multimodal elements (images and texts) is used as conditioning information and composed into a cohesive design [4], [26]–[28]. Our work is related to a branch of methods on content-aware layout generation [13], which generates layouts conditioned on a background image [18], [29]–[32]. However, instead of tackling background-conditioned layout generation solely, we aim to generate both a layout and a background image to constitute a design template.

## 2.2 Graphic Design Generation

Recently, there has been a rising interest in developing models for graphic design generation. For example, CanvasVAE [33] trains a VAE to generate graphic designs represented as sets of canvas and element attributes. GOL [6] shows that learned element order can improve the performance of generative models that predict a sequence of element attribute tokens. Several methods have been proposed to automate graphic design generation from textual design intentions. One class of methods directly fine-tunes pretrained text-to-image diffusion models for text-to-design generation, producing highly aesthetic and coherent design images [34], [35]. Another class of methods tries to generate layered graphic designs, composed of multiple image and text layers, using a cascade of task-specific models [2], [3], or a multimodal large language model with an imagegeneration capability [5].

The design template generation problem that we focus on can be viewed as a specialization of design generation that considers a subset of element attributes, including background image and element layout. A recent work, Desigen [1], presents a solution to the problem. Desigen takes a two-stage approach that generates the background and layout sequentially using two separate models, and captures the dependency between them by simply conditioning one model on the final output of the other. This method suffers from two major shortcomings: first, the communication between the image and layout models is limited, which restricts its ability to capture interaction between image and layout; second, it explicitly introduces an inductive bias towards minimizing occlusion of salient background regions, which causes the distribution of generated samples to drift from the data distribution, washing out distinctive patterns of real samples from the model outputs. In contrast, our model generates a background image and a layout jointly in a single diffusion model, which focuses on learning image-layout interaction to achieve more coherent image-layout composition. In addition, our model imposes no design-specific inductive bias, thereby better maintaining the distinctive features of training samples in generated design templates.

## 3 METHOD

Given a text description ${ \mathcal P } ,$ our goal is to generate a design template $X ^ { D ^ { \smash { * } } }$ . Each design template consists of a background image $X ^ { I }$ and a layout $X ^ { L }$ , where the layout comprises a set of elements defined by their categories and bounding box coordinates. To this end, we aim to learn a joint distribution $p ( X ^ { I } , X ^ { L } | \mathcal { P } )$ of background images $X ^ { I }$ and layouts $X ^ { L }$ conditioned on ${ \mathcal P } ,$ from which we can sample $X ^ { \check { I } }$ and $X ^ { L }$ , and compose them into a coherent design template. The key idea of our method is to 1) train two expressive diffusion priors on image and layout domains (to ensure the high-quality generation of background images and layouts), and 2) learn the interaction between the two priors in a single joint diffusion model (to ensure a harmonious composition). This leads to a unified model, InterIL, for joint generation of background images and layouts from text inputs.

## 3.1 Domain-specific Priors

We aim to build a conditional image prior $p ( X ^ { I } | \mathcal { P } )$ that can generate images from text prompts, and a layout prior $p ( X ^ { L } )$ for generating high-quality layouts.

Image Prior. For $\overset { \cdot } { p } ( \check { X } ^ { I } | \overset { \cdot } { \mathcal { P } } )$ , we employ Stable Diffusion (SD) [7], a large-scale latent diffusion model (LDM) for textto-image generation. SD trains a variational autoencoder (VAE) with an image encoder ${ \mathcal { E } } _ { \mathrm { I } }$ and an image decoder $\mathcal { D } _ { \mathrm { I } }$ . The encoder ${ \mathcal { E } } _ { \mathrm { I } }$ maps an RGB image $X ^ { I } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ to a spatial latent representation $z _ { 0 } ^ { I } = \mathcal { \bar { E } } _ { \mathrm { I } } ( X ^ { I } ) \in \mathbb { R } ^ { h \times w \times d ^ { I } }$ (an $h \times w$ grid of embeddings, each of dimensionality $d ^ { I } )$ Then, a diffusion model is learned over latents instead of pixels. For image generation, a latent $\tilde { z } _ { 0 } ^ { I }$ is sampled from the diffusion model and then decoded by the decoder $\mathcal { D } _ { \mathrm { I } }$ into an image $\tilde { X } ^ { I } = \mathcal { D } _ { \mathrm { I } } ( \tilde { z } _ { 0 } ^ { I } )$ . To adapt the pretrained SD to the task of generating background images in graphic designs, we fine-tune it on a dataset of background images associated with text descriptions to obtain the adapted denoising network $\epsilon _ { \theta ^ { * } }$ , which we refer to as the image backbone in our joint model introduced later.

Layout Prior. For $p ( X ^ { L } )$ , we train another LDM from scratch on a layout dataset. Following prior work on layout generation [17], [19], [20], we define a layout element as a bounding box with 5 attributes, including category, left coordinate, top coordinate, width, and height. After discretizing the continuous attributes into bins using the kmeans algorithm, an element is formatted as a sequence of 5 attribute tokens, and a layout is the concatenation of the element sequences. We first train a VAE to project layouts into a latent space. The Transformer-based VAE encoder ${ \mathcal { E } } _ { \mathrm { L } }$ encodes a layout $X ^ { L }$ into a latent representation $z _ { 0 } ^ { L } = \mathcal { E } _ { \mathrm { L } } ( X ^ { L } ) \in \mathbb { R } ^ { N \times d ^ { L } }$ , where N is the number of elements in the layout and $d ^ { L }$ is the embedding dimensionality. We then train a diffusion transformer (DiT) [36] over the latent space. A layout can be generated by sampling $\tilde { z } _ { 0 } ^ { L }$ from the DiT, followed by feeding it into the Transformer-based VAE decoder $\mathcal { D } _ { \mathrm { L } } \colon \tilde { \dot { X } } ^ { L } = \tilde { \mathcal { D } _ { \mathrm { L } } } ( \tilde { z } _ { 0 } ^ { L } )$ . We refer to the trained Transformer-based denoising network $\epsilon _ { \phi ^ { * } }$ of the DiT as the layout backbone in our subsequent joint model.

## 3.2 Joint Model

Given the two pretrained priors that operate independently, we merge them into a single diffusion model to simultaneously generate background images and layouts. As illustrated in Figure 2, the core design of the joint model’s denoising network is to put the pretrained and frozen image and layout backbones together to jointly denoise noisy image and layout latents, while connecting the backbones using a learnable communication module to capture imagelayout interaction. Such a design has two primary benefits: first, by locking the weights of the backbones, we can retain the pretrained knowledge in them to effectively leverage their original capabilities to ensure the quality of generated images and layouts; second, as the prior knowledge of how to generate images and layouts independently is already available in the two pretrained backbones, our joint model, when trained on design template datasets, can focus on learning interaction between image and layout, thus improving composition harmony. Furthermore, the two backbones can communicate over a large number of denoising steps during the generation process of images and layouts. This is different from the sequential iterative refinement (e.g., in Desigen), where the image (layout) is generated after the generation of the layout (image). Consequently, our joint model can better model image-layout interaction than prior work.

![](images/d8c20db410e7bf1566a65b0a44467f00dc8894cd71d780e93e7f234e882e6cfa.jpg)  
Fig. 2. Overview of our joint model. Our model jointly denoises the noisy image and layout latents with an architecture that connects the pretrained, fixed backbones of image and layout diffusion models with a learnable communication module for explicitly learning bidirectional image-layout interaction.

To learn our model, we train a joint denoising network:

$$
\begin{array} { r } { [ \hat { \epsilon } _ { t } ^ { I } , \hat { \epsilon } _ { t } ^ { L } ] = \epsilon _ { \psi , \theta ^ { * } , \phi ^ { * } } ( z _ { t } ^ { I } , z _ { t } ^ { L } , t , \mathcal { P } ) , } \end{array}\tag{1}
$$

where $\hat { \epsilon } _ { t } ^ { I } , \hat { \epsilon } _ { t } ^ { L }$ are predicted noises. The training is performed by optimizing the following objective:

$$
\mathcal { L } = \mathbb { E } _ { z _ { 0 } ^ { I } , z _ { 0 } ^ { L } , t , \epsilon _ { t } ^ { I } , \epsilon _ { t } ^ { L } } \Big [ \lambda _ { I } \| \epsilon _ { t } ^ { I } - \hat { \epsilon } _ { t } ^ { I } \| _ { 2 } ^ { 2 } + \lambda _ { L } \| \epsilon _ { t } ^ { L } - \hat { \epsilon } _ { t } ^ { L } \| _ { 2 } ^ { 2 } \Big ] ,\tag{2}
$$

where $t \sim [ 1 , T ]$ (T is the number of diffusion timesteps), $\epsilon _ { t } ^ { I } \sim \mathcal { N } ( \mathbf { 0 } , \dot { \mathbf { I } } ) .$ , and $\epsilon _ { t } ^ { L } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Our implementation sets $\lambda _ { L } = 3$ and $\lambda _ { I } = 1$ . ψ contains the communication module’s trainable weights.

## 3.3 Communication Module

As illustrated in Figure 3, the communication module is designed to facilitate bidirectional information exchange between the image and layout backbones, while preserving their pretrained knowledge as much as possible. Let $h ^ { I } \in \mathbb { R } ^ { M \times d }$ be the flattened intermediate representation of the U-Net-based image backbone (from the second encoder block, the third encoder block or the middle block), and $h ^ { L } \in \mathbb { R } ^ { N \times d }$ be the output representation of the 19th DiT block of the layout backbone. The communication module augments $h ^ { I }$ and $h ^ { L }$ as:

$$
[ \bar { h } ^ { I } , \bar { h } ^ { L } ] = \mathrm { C O M M } ( h ^ { I } , h ^ { L } ) .\tag{3}
$$

The augmented representations $\bar { h } ^ { I } , \bar { h } ^ { L }$ are then fed into the subsequent blocks in the image and layout backbones, respectively. More specifically, to obtain $\bar { h } ^ { \check { I } }$ , the operations can be written as:

$$
\hat { h } ^ { I } = h ^ { I } + \eta \cdot \mathrm { C r o s s A t t n } ( h ^ { I } , h ^ { L } ) .\tag{4}
$$

The cross-attention can model image-layout interaction, propagating layout information from $h ^ { \check { L } }$ to $h ^ { I } . ~ \eta$ is set to 1 during training and can be varied during inference

to control when the communication is enabled over the denoising process. Similarly, $\bar { h } ^ { L }$ is obtained by:

$$
\hat { h } ^ { L } = h ^ { L } + \eta \cdot \mathrm { C r o s s A t t n } ( h ^ { L } , h ^ { I } ) .\tag{5}
$$

In this work, we opt to make the design of the communication module simple and effective (verified in Section 4.6), and leave the exploration of more sophisticated architectural designs for future work.

When training the model, we set $\eta = 1$ in Equation 4 and 5, enabling the communication module for all diffusion timesteps so that this module can be sufficiently updated. However, our early experiments show that using $\eta \ : = \ : 1$ throughout the entire denoising process at test time leads to unsatisfactory layout generation quality. To mitigate this issue, inspired by GLIGEN [37], we dynamically adjust the value of η during the sampling process to improve visual quality. The denoising process of diffusion models generates coarse-grained features $( \mathrm { e . g . , }$ global structures and colors) at early steps, and then generates perceptually important content and fine-grained details at later steps. Therefore, as illustrated in Figure 4, we opt to turn on the communication module $( \eta ~ = ~ 1 )$ during the first 30% of the denoising process, where the image backbone can use external knowledge from the layout backbone to generate a rough spatial arrangement of objects in the image that harmonizes with the layout, and vice versa for the layout backbone. In the remaining denoising steps, we disable the communication module $( \eta = 0 )$ , allowing the two backbones to rely upon their internal knowledge to generate high-quality images and layouts.

## 3.4 Preference-based Guidance

Our model uses no design-specific inductive bias, thereby enjoying the benefit of preserving the distinctive design patterns in realistic (training) design samples. However, in some cases, users may prefer the occurrence of specific design characteristics in generated templates (e.g., minimal occlusion of important background regions). To address this, we propose a guidance technique that is applied at inference to impose preference-based constraints on generated results.

Denoising Process  
![](images/d2f614e983fa8df798a96ea255dbfc1e7f08cf2e882bd8350ba0b6e82ed3f5ab.jpg)  
Fig. 3. The communication module enables bidirectional information exchange between the image backbone and layout backbone by augmenting the intermediate representation of each backbone with that of the other backbone through cross-attention.

![](images/f82263f49da38c3dee9fad44a9b4ac39b3780cb30e9213b49ca50f3c7d088f2a.jpg)  
Fig. 4. Scheduling the communication between the image and layout backbones during training and inference by setting the value of $\eta .$ During training, the communication is enabled by setting $\eta = 1$ at all diffusion time steps. During inference, the communication is only enabled for the first 30% of the denoising process where $\eta = 1$ , and is turned off for the remaining denoising steps where $\eta = 0$

Specifically, we guide the layout generation process by adjusting the noise prediction $\overline { { \hat { \epsilon } _ { t } ^ { L } } }$ of the layout backbone:

$$
\hat { \epsilon } _ { t } ^ { L } \gets \hat { \epsilon } _ { t } ^ { L } - s \sqrt { 1 - \bar { \alpha } _ { t } } \nabla _ { z _ { t } ^ { L } } \mathcal { L } _ { \mathrm { g u i d e } } \big ( \hat { \epsilon } _ { t } ^ { I } , \hat { \epsilon } _ { t } ^ { L } \big ) ,\tag{6}
$$

where s is the guidance scale, $\bar { \alpha } _ { t }$ is the multiplication of the forward-process noise variances from 1 to $t ,$ and $\begin{array} { r } { \mathcal { L } _ { \mathrm { g u i d e } } ~ = ~ \sum _ { k = 1 } ^ { K } \lambda _ { k } \mathcal { L } _ { k } } \end{array}$ is a weighted combination of differentiable objectives that represent different user preferences. $\big [ \hat { \epsilon } _ { t } ^ { I } , \hat { \epsilon } _ { t } ^ { L } \big ] ^ { ' } = \epsilon _ { \psi ^ { * } , \theta ^ { * } , \phi ^ { * } } \big ( z _ { t } ^ { \hat { I } } , z _ { t } ^ { L } , t , \mathcal { P } \big )$ , where $\psi ^ { * }$ denotes the trained weights obtained by minimizing Equation 2.

We instantiate this guidance technique with two concrete objectives as follows. The first objective represents preference for minimizing occlusion, discouraging foreground elements from occluding salient background regions. Given the predicted image noise $\hat { \epsilon } _ { t } ^ { I } .$ , we compute a clean image latent $\boldsymbol { \hat { z } } _ { 0 } ^ { I }$ analytically using the forward process marginal distribution $q \big ( z _ { t } ^ { I } | z _ { 0 } ^ { I } \big )$ , decode it into an image $\mathcal { D } _ { \mathrm { I } } ( \hat { z } _ { 0 } ^ { I } )$ , and compute a saliency map $S ~ = ~ f _ { \mathrm { s a l } } ( { \mathcal D } _ { \mathrm { I } } ( \hat { z } _ { 0 } ^ { I } ) )$ using an offthe-shelf saliency detector $f _ { \mathrm { s a l } }$ . We then compute a clean layout latent $\hat { z } _ { 0 } ^ { L }$ from the predicted layout noise $\hat { \epsilon } _ { t } ^ { L }$ with $q \big ( z _ { t } ^ { L } | z _ { 0 } ^ { L } \big )$ , and decode it into a layout $\dot { \mathcal { D } } _ { \mathrm { L } } ( \hat { z } _ { 0 } ^ { L } )$ . Let $\{ { \bf { b } } _ { i } \} _ { i = 1 } ^ { B }$ be the decoded element bounding boxes. The objective is defined as:

$$
\mathcal { L } _ { \mathrm { o c c } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 1 } { A _ { i } } \sum _ { p } M _ { \mathbf { b } _ { i } } ( p ) S ( p ) ,\tag{7}
$$

where $M _ { \mathbf { b } _ { i } }$ is a soft mask for $\mathbf { b } _ { i } , \ A _ { i }$ is the area of $\mathbf { b } _ { i } ,$ and $p$ indexes spatial positions. To compute $M _ { \mathbf { b } _ { i } }$ in a way that ensures that $\mathcal { L } _ { \mathrm { o c c } }$ is differentiable, we convert discrete bounding box parameters to continuous ones by computing each parameter as a weighted sum of quantization bin centers, weighted by predicted probabilities. Given a bounding box with continuous parameters $\mathbf { b } _ { i } \ = \ \left( x _ { i } , y _ { i } , w _ { i } , h _ { i } \right)$ , we derive the corner coordinates $( x _ { i } ^ { l } , y _ { i } ^ { t } )$ and $( x _ { i } ^ { r } , y _ { i } ^ { b } )$ , and then

compute $M _ { \mathbf { b } _ { i } }$ as:

$$
\begin{array} { r l } & { M _ { \mathbf { b } _ { i } } ( x , y ) = \sigma ( \lambda ( x - x _ { i } ^ { l } ) ) \times \sigma ( \lambda ( x _ { i } ^ { r } - x ) ) } \\ & { \qquad \times \sigma ( \lambda ( y - y _ { i } ^ { t } ) ) \times \sigma ( \lambda ( y _ { i } ^ { b } - y ) ) , } \end{array}\tag{8}
$$

where $\sigma ( \cdot )$ is the sigmoid function and $\lambda \ : = \ : 4 0$ . We refer to guidance using this objective as occlusion-aware guidance (OAG).

The second objective represents preference for improved text readability, which encourages text elements to lie on flat background regions. From the decoded image $\tilde { I } = \mathcal { D } _ { \mathrm { I } } ( \hat { z } _ { 0 } ^ { I } )$ we construct a map C that has higher values for positions in <sup>˜</sup>I where high-frequency patterns occur:

$$
C ( x , y ) = \sqrt { \left\| ( K _ { x } * \tilde { I } ) ( x , y ) \right\| _ { 2 } ^ { 2 } + \left\| ( K _ { y } * \tilde { I } ) ( x , y ) \right\| _ { 2 } ^ { 2 } + \varepsilon } ,\tag{9}
$$

where $K _ { x }$ and $K _ { y }$ are the horizontal and vertical Sobel kernels, respectively, ∗ denotes convolution, the $\ell _ { 2 }$ norm is computed over the RGB channels, and ε is a small constant for numerical stability.

From the decoded layout $\mathcal { D } _ { \mathrm { L } } \big ( \hat { z } _ { 0 } ^ { L } \big )$ , we select bounding boxes corresponding to text elements, denoted by $\{ { \bf b } _ { i } ^ { \mathrm { t e x t } } \} _ { i = 1 } ^ { B _ { T } }$ The objective is defined as:

$$
\mathcal { L } _ { \mathrm { r e a d } } = \frac { 1 } { B _ { T } } \sum _ { i = 1 } ^ { B _ { T } } \frac { 1 } { A _ { i } } \sum _ { p } M _ { \mathbf { b } _ { i } ^ { \mathrm { t e x t } } } ( p ) C ( p ) ,\tag{10}
$$

where $M _ { \mathbf { b } _ { i } ^ { \mathrm { t e x t } } }$ is a differentiable soft mask for ${ \bf b } _ { i } ^ { \mathrm { t e x t } } ,$ , which is computed in the same way as $M _ { \mathbf { b } _ { i } }$ above. We refer to guidance using this objective as readability-aware guidance (RAG).

## 4 EXPERIMENTS

## 4.1 Dataset and Implementation Details

We conduct our experiments on the Web-design dataset [1], which consists of 50K web banner designs collected from online shopping platforms. We use 41,270 samples (85%) for training, 2,427 (5%) for validation, and 4,856 (10%) for testing. For pretraining the layout prior, we train its VAE (embedding dimensionality 32) using β-VAE with $\beta ~ = ~ 5 \times 1 0 ^ { - 4 }$ . The layout backbone is trained for 1000 epochs with a batch size of 4096. For the image prior, we fine-tune the image backbone for 100 epochs with a learning rate of $1 \times 1 0 ^ { - 5 }$ on the background images from the Web-design. Both image and layout models use 1000 diffusion timesteps. During inference, we use the DDIM sampler for the layout backbone and the DDPM sampler for the image backbone, each using 50 sampling steps. The communication module is enabled at time steps greater than 700 to facilitate information exchange between the layout and image backbones. Both occlusion-aware guidance and readability-aware guidance are applied between time steps 200 and 700.

## 4.2 Compared Methods

We compare our method with a prior design template generation model, Desigen [1], which is already trained on the Web-design dataset. Desigen proposes an iterative strategy to refine the generated image and layout. We run this iterative refinement for 3 iterations in the comparison. We also consider a recent text-to-design generation model, OpenCOLE [3] as an additional baseline.

## 4.3 Evaluation Metrics

Domain-specific Metrics. Following the evaluation protocol of [1], we evaluate the quality of background image and layout separately using the following metrics. For background image evaluation, we use: Saliency Ratio [1] that measures the proportion of salient regions in an image; FID [38] that measures visual quality and is computed against the testing split; CLIP Score [39] that measures text-image alignment. For layout evaluation, we use: Alignment [12] that measures alignment between layout elements; Overlap [12] that measures the amount of overlap between layout elements; Occlusion [30] that measures the occlusion of background salient regions by layout elements. As additional layout metrics for comprehensive evaluation, we also include: LayoutFID [14] that measures overall layout quality and is computed against the testing split; Readability [18] that measures text readability.

Holistic Metrics. We further propose two metrics, Template-FID and TemplateCLIP, to evaluate holistic design templates by considering background image and layout jointly. TemplateFID measures how realistic generated templates are, while TemplateCLIP evaluates how well generated templates adhere to the text inputs.

To compute TemplateFID, we train a template autoencoder (TemplateAE) on a large-scale mixture dataset comprising GenPoster100K [32], CGL [40], Crello [33], and PKU [41], deliberately excluding Web-design to avoid evaluation bias. Figure 5 illustrates its architecture. Given a background image and a layout, the encoder maps them to a single template embedding. Specifically, an image encoder (the pretrained VAE encoder of Stable Diffusion v1.4) and a layout encoder first map the background image $X ^ { \bar { I } }$ and layout $X ^ { L }$ to their respective latent representations $z ^ { I } ~ \in ~ \hat { \mathbb { R } } ^ { 2 8 \times 2 8 \times 4 }$ and $z ^ { L } ~ \in ~ \dot { \mathbb { R } } ^ { N \times 3 2 }$ , where $\bar { N _ { \mathrm { ~ \scriptsize ~ = ~ } } } 7$ is the maximum number of elements in a layout and 32 is the embedding dimensionality. Note that the layout encoder here is not the VAE encoder of the layout prior in InterIL; we train a separate VAE on layouts in the mixture dataset and use its encoder $\mathcal { E } _ { \mathrm { L } } ^ { ' } .$ . Then, $z ^ { I }$ is flattened into an embedding of dimensionality 3136 and projected by a fully connected layer to match the dimensionality of $z ^ { L }$ . The projected $z ^ { I }$ is replicated N times to form $\bar { z } ^ { I } \in \mathbb { R } ^ { N \times 3 2 }$ $\stackrel { \cdot } { z } I$ and $z ^ { L }$ are concatenated along the sequence dimension and fed into another fully connected layer, producing a sequence of 2N embeddings of dimensionality 512. The embedding sequence and a learnable embedding (prepended to the sequence) are processed by a template encoder ${ \mathcal { E } } _ { \mathrm { T } }$ (a stack of Transformer encoder blocks), and the output of the final block for the learnable embedding is used as the template embedding $z ^ { T } \in \mathbb { R } ^ { 5 1 2 }$ . A template decoder $\mathcal { D } _ { \mathrm { T } }$ reconstructs the image latent representation and the layout from $z ^ { T }$ . In particular, the input to $\mathcal { D } _ { \mathrm { T } }$ is a sequence of N repeated $z ^ { T }$ with sinusoidal positional embeddings added. $\bar { \mathcal { D } } _ { \mathrm { T } }$ (a stack of Transformer encoder blocks) produces a sequence of output embeddings of dimensionality 512, which are passed through two prediction heads to reconstruct an image latent $\hat { z } ^ { \breve { I } }$ and a layout $\hat { y } ^ { L }$ (represented by class logits for each position in the layout sequence). The image head linearly projects each input embedding into dimensionality 3136, and then averages across the sequence dimension to produce a reconstructed image latent $\hat { z } ^ { I }$ Given the input embedding for each non-padding layout element, the layout head uses one linear layer to predict class logits for the category and four independent MLPs to predict class logits for each of the four bounding box coordinates. Thus, $\stackrel { \smile } { \hat { y } } ^ { L }$ comprises a sequence of non-padding layout elements, where each element occupies five positions and each position has a set of class logits. TemplateAE is trained by optimizing $\mathcal { L } = \lambda _ { \mathrm { i m g } } \mathcal { L } _ { \mathrm { i m g } } + \lambda _ { \mathrm { c l s } } ^ { - } \mathcal { L } _ { \mathrm { c l s } } + \hat { \lambda _ { \mathrm { b b o x } } } \mathcal { L } _ { \mathrm { b b o x } } .$ $\mathcal { L } _ { \mathrm { i m g } } = \dot { 1 } - \mathrm { c o s } ( \hat { z } ^ { I } , \breve { z } ^ { I } )$ , where cos(·, ·) is cosine similarity. $\mathcal { L } _ { \mathrm { c l s } }$ computes the mean cross-entropy over the category positions in $\hat { y } ^ { L }$ , and ${ \mathcal { L } } _ { \mathrm { b b o x } }$ computes the mean cross-entropy over the bounding box coordinate positions in $\hat { y } ^ { L }$ . During training, the image encoder and layout encoder are kept frozen, while the template encoder, template decoder, and MLP heads are updated.

![](images/e356eae8e3f1aab89e6b101d7e8eb6d83c4829a783783c18c8f87af11a201be9.jpg)  
Fig. 5. Architecture of the template autoencoder (TemplateAE).

We evaluate the quality of the learned template embeddings with a user study involving 104 participants, including 58 design experts and 46 non-experts. We first construct 30 triplets of the form $( x , \tilde { x } _ { \mathrm { r a n d } } , \tilde { x } _ { \mathrm { e m b } } )$ , where x is a reference template, $\tilde { x } _ { \mathrm { r a n d } }$ is a randomly selected template from the Web-design dataset, and $\tilde { x } _ { \mathrm { e m b } }$ is the most similar template to the reference retrieved from the Web-design dataset based on the learned template embeddings. The subjects were given a triplet and asked to select which of $\tilde { x } _ { \mathrm { r a n d } }$ and $\tilde { x } _ { \mathrm { e m b } }$ is more similar to the reference. The subjects selected the retrieved templates 83.4% of the time, suggesting that TemplateAE trained on the composed dataset captures important features of design templates and generalizes well to Web-design samples.

To compute TemplateFID, we calculate Frechet Inception´ Distance (FID) [38] between generated and real samples based on the template embeddings. To compute Template-CLIP, we initialize from the composed-dataset-pretrained TemplateAE encoder and further fine-tune it together with the pretrained SD text encoder using the contrastive learning objective of CLIP [39] on the Web-design train split.

## 4.4 Quantitative Results

Comparison to Desigen. Table 1 presents the quantitative comparison with Desigen. We report metrics computed on the test set of the Web-design dataset as a reference, shown as Real Data at the bottom of Table 1. For image generation, our method outperforms Desigen across all three metrics, with substantial improvements in FID and saliency ratio.

This indicates that our model can generate high-fidelity background images with sufficient empty space for foreground element placement. It should be noted that the CLIP score of real data is lower than those of the other methods, as real background images leave more empty space for foreground elements (lower saliency ratio), which weakens text–image alignment. For layout generation, our model achieves the best results on 4 out of 5 metrics, and is comparable to Desigen for the readability metric, demonstrating its superior layout generation capability. Thanks to its explicit learning of image-layout interaction through the communication module, our model is able to generate more harmonious design templates than Desigen, as evidenced by its noticeable improvement in TemplateFID. Furthermore, our model achieves the best TemplateCLIP, showing its strong text adherence.

As Desigen explicitly enforces occlusion avoidance constraints in its model design, its saliency ratio and occlusion scores decline over refinement iterations, meaning that the salient portion of the generated background image is gradually reduced, and layout elements and salient background regions are moved further apart from each other. While such iterative refinement boosts performance in some metrics (e.g., LayoutFID and readability), a side effect is exposed: the gaps between Desigen’s generated samples and real samples, in terms of several aspects (measured by metrics, e.g., alignment, overlap and occlusion), become increasingly larger. The growing gaps imply that the refined results lose some unique design characteristics in training examples, which is undesirable. For example, designers may occasionally make foreground elements partially occlude salient background objects, e.g., for creative or artistic purposes. Excessive occlusion reduction can lead to a lower occlusion score, but may generate unnatural outputs that don’t resemble what human designers create. Due to the existence of the aforementioned gaps, Desigen’s TemplateFID improves only slightly with the iterative refinement, still significantly lagging behind that of our model.

Comparison to OpenCOLE. OpenCOLE consists of three modules: 1) a design plan generation module (a pretrained large language model) to convert a brief design intention into a detailed design plan; 2) an image generation module (a fine-tuned SD model) that generates an image (i.e., a background image in our problem setting) conditioned on the concatenation of object and background captions from the design plan; 3) a typography generation module (a finetuned large multimodal model) that generates text attributes (including the bounding box parameters of texts) based on the design plan and the generated image. For design template generation, OpenCOLE can be seen as a sequential two-stage approach (image generation followed by text layout generation) similar to Desigen, with an additional component (the design plan generation module) to augment the input text prompt. For a fair comparison, we modify the image generation module to use the same SD model as our model, and change the typography generation module to output the layout of elements. The above two modules are fine-tuned on the Web-design dataset. Furthermore, we leverage GPT to turn short text inputs into long, detailed prompts before feeding them into our model. Table 2 reports the quantitative results of OpenCOLE and our model. It can be seen that our model outperforms OpenCOLE on 9 out of 10 metrics.

TABLE 1  
Quantitative comparison with Desigen on the Web-design dataset. “Desigen (n)” indicates applying iterative refinement n times. For each metric except ones related to CLIP and FID, values closer to that of real data (bottom row) indicate better performance. The best and second-best numerical results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">Image</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>Saliency Ratio</td><td>LayoutFID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>Desigen (0)</td><td>31.52</td><td>29.20</td><td>20.65%</td><td>0.56</td><td>0.35</td><td>14.41</td><td>13.47%</td><td>11.48%</td><td>118.96</td><td>3.19</td></tr><tr><td>Desigen (1)</td><td>30.84</td><td>28.87</td><td>19.29%</td><td>0.57</td><td>0.38</td><td>15.63</td><td>13.24%</td><td>11.56%</td><td>108.56</td><td>3.18</td></tr><tr><td>Desigen (2)</td><td>31.23</td><td>28.46</td><td>18.36%</td><td>0.51</td><td>0.37</td><td>15.52</td><td>12.70%</td><td>11.23%</td><td>114.10</td><td>3.19</td></tr><tr><td>Desigen (3)</td><td>30.79</td><td>29.04</td><td>17.67%</td><td>0.48</td><td>0.37</td><td>15.23</td><td>11.45%</td><td>10.97%</td><td>116.83</td><td>3.21</td></tr><tr><td>Ours</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.15</td><td>0.31</td><td>11.98</td><td>19.30%</td><td>11.41%</td><td>86.63</td><td>3.25</td></tr><tr><td>Real Data</td><td></td><td>27.50</td><td>14.17%</td><td>一</td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td></td><td>3.39</td></tr></table>

TABLE 2

Quantitative comparison with OpenCOLE on the Web-design dataset. The text prompts input to our model are augmented by GPT. For each metric except ones related to CLIP and FID, values closer to that of real data (bottom row) indicate better performance. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">Image</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>Saliency Ratio</td><td>LayoutFID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>OpenCOLE</td><td>23.79</td><td>30.49</td><td>23.24%</td><td>1.63</td><td>4.20</td><td>11.44</td><td>30.48%</td><td>11.74%</td><td>211.42</td><td>3.12</td></tr><tr><td>Ours (Prompt Aug.)</td><td>18.97</td><td>30.94</td><td>15.32%</td><td>0.16</td><td>0.32</td><td>12.06</td><td>19.74%</td><td>11.63%</td><td>85.35</td><td>3.28</td></tr><tr><td>Real Data</td><td></td><td>27.50</td><td>14.17%</td><td>1</td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td>1</td><td>3.39</td></tr></table>

TABLE 3

GPT-5 evaluation results across four aspects : (i) image quality, (ii) layout quality, (iii) image–layout harmony, (iv) text–design relevance. The best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>(i)</td><td>(ii)</td><td>(iii)</td><td>(iv)</td></tr><tr><td>Desigen</td><td>6.42</td><td>6.54</td><td>7.47</td><td>6.21</td></tr><tr><td>OpenCOLE Ours</td><td>6.81 6.94</td><td>6.02 7.76</td><td>5.84 8.07</td><td>6.38 7.34</td></tr><tr><td>Real Data</td><td>7.44</td><td>7.87</td><td>8.39</td><td>7.49</td></tr></table>

TABLE 4

Human preference results. Each number is the percentage of the time that a method is chosen as the best in terms of one aspect.
<table><tr><td>Aspect</td><td>Desigen</td><td>OpenCOLE</td><td>Ours</td></tr><tr><td>Image Quality</td><td>13%</td><td>19%</td><td>68%</td></tr><tr><td>Layout Quality</td><td>14%</td><td>8%</td><td>78%</td></tr><tr><td>Image-Layout Harmony</td><td>13%</td><td>4%</td><td>83%</td></tr><tr><td>Text-Design Relevance</td><td>13%</td><td>16%</td><td>71%</td></tr></table>

GPT-5 Evaluation. Inspired by recent work on graphic design generation [2], [3], we also leverage GPT-5 to evaluate design templates generated by Desigen, OpenCOLE, and our model on the Web-design test split. Specifically, GPT-5 assesses the generated results across four aspects: image quality, layout quality, image-layout harmony, and textdesign relevance. As summarized in Table 3, our model achieves the best results across all four aspects. See the supplementary material for prompt details.

Human Evaluation. We further conduct a human evaluation with 32 participants. For the evaluation, 40 input text prompts from the Web-design test split are randomly selected. The participants are shown an input text, along with three design templates that are generated from the text by Desigen, OpenCOLE, and our model, respectively, and are presented in randomized order. They are asked to select the best template for each of the four aspects: 1) image quality, 2) layout quality, 3) image–layout harmony, and 4) text–design relevance. The results are shown in Table 4. Our model is strongly preferred over Desigen and OpenCOLE on all aspects, especially image-layout harmony.

TABLE 5  
Average inference time per design template on the Web-design test set.
<table><tr><td rowspan=1 colspan=1>Method            Time</td></tr><tr><td rowspan=1 colspan=1>Desigen (0)          7.47Desigen (1)         14.85</td></tr><tr><td rowspan=1 colspan=1>Desigen (2)         22.42</td></tr><tr><td rowspan=1 colspan=1>Desigen (3)         29.89</td></tr><tr><td rowspan=1 colspan=1>GPT-40             10.86</td></tr><tr><td rowspan=1 colspan=1>OpenCOLE         25.80</td></tr><tr><td rowspan=1 colspan=1>Ours                5.01</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { O u r s } + \mathrm { O A G }$         8.69</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { O u r s } + \mathrm { R A G }$          8.47</td></tr><tr><td rowspan=1 colspan=1>Ours (Prompt Aug.)10.14</td></tr></table>

Computational Efficiency. Table 5 reports the inference time of different methods measured on a single NVIDIA A40 GPU. Due to our single-stage joint image-layout generation scheme, the base model is the most compute-efficient among all the methods. While the iterative refinement of Desigen can improve generation quality with more iterations, it incurs a substantial increase in computational cost—using 3 refinement iterations increases inference time from 7.47 s to 29.89 s. OpenCOLE is the most computationally expensive because its cascaded pipeline relies on multiple large models. Applying OAG or RAG to our model introduces only modest additional cost. The guided and prompt-augmented variants of our model remain considerably faster than Open-COLE and Desigen with iterative refinement.

![](images/ca847f2e944ed2758575fa9dce7e8147bb6ddfc5ddbf3ff93fe80659e9517b21.jpg)  
Fig. 6. Visual comparison of different methods. White, orange and red boxes denote text, underlay and button, respectively. Input texts are shown at the top of each column, and the last row displays the ground truth design templates.

## 4.5 Qualitative Results

Figure 6 shows the visual comparison of design templates generated by different methods. Our model can generate high-quality, diverse layouts and visually pleasing background images. More importantly, due to its strong ability to capture the interaction between layout and background image, our model can produce harmonious layout-image compositions. In contrast, the results of OpenCOLE and Desigen suffer from various issues. Specifically, in the results of OpenCOLE, foreground elements are significantly misaligned and salient background areas are often occluded by foreground elements. The layouts of Desigen sometimes exhibit undesirable overlap between elements (column 4). The background images generated by Desigen appear less visually appealing, mainly because its explicit occlusion avoidance bias could cause large empty regions (for foreground element placement) in generated background images (columns 3 and 5). Additional generated results by our method are shown in Figure 7.

## 4.6 Analysis on the Communication Module

Ablations. One key ingredient of our model is the communication module that enables the image and layout backbones to interact with each other during the generation process. To test the effect of this component, we consider a variant, where the communication module is disabled and thus the two backbones denoise independently. As shown in Table 6, using the communication module dramatically reduces the saliency ratio, occlusion, and readability, and contributes to a significant improvement in TemplateFID. This highlights the importance of the communication module for improving the composition harmony of the background image and layout. We further consider two variants that only allow unidirectional information flow from layout to image (by only augmenting the image representation $h ^ { I }$ using Equation 4) and from image to layout (by only augmenting the layout representation $\mathit { \Pi } _ { h ^ { L } }$ using Equation 5), respectively. Compared with our communication module that enables bidirectional information exchange, the unidirectional variants yield substantially worse performance on occlusion, readability, and TemplateFID. This suggests that bidirectional interaction between image and layout is critical for generating harmonious image-layout compositions. Figure 8 shows a visual comparison of results obtained using our communication module and the aforementioned variants.

![](images/c0fed8aecc14f1fd9ebb0e1d004e33d9b06696766d43bb103a0f80a182c1da41.jpg)  
Fig. 7. Additional results generated by our method. White, orange, and red boxes denote text, underlay, and button, respectively.

TABLE 6  
Ablation study on the communication module. For each metric except ones related to CLIP and FID, values closer to that of real data (bottom row) indicate better performance. The best results are in bold.
<table><tr><td rowspan="2">Communication</td><td colspan="3">Image</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>Saliency Ratio</td><td>LayoutFID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>None</td><td>19.32</td><td>30.37</td><td>21.37%</td><td>0.16</td><td>0.38</td><td>12.87</td><td>30.74%</td><td>12.23%</td><td>99.14</td><td>3.23</td></tr><tr><td>Layout → Image</td><td>19.74</td><td>29.84</td><td>15.40%</td><td>0.17</td><td>0.35</td><td>13.20</td><td>28.40%</td><td>14.30%</td><td>97.45</td><td>3.22</td></tr><tr><td>Image → Layout</td><td>19.39</td><td>30.43</td><td>20.90%</td><td>0.16</td><td>0.34</td><td>13.60</td><td>27.80%</td><td>14.00%</td><td>96.20</td><td>3.24</td></tr><tr><td>Bidirectional (Ours)</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.15</td><td>0.31</td><td>11.98</td><td>19.30%</td><td>11.41%</td><td>86.63</td><td>3.25</td></tr><tr><td>Real Data</td><td></td><td>27.50</td><td>14.17%</td><td></td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td></td><td>3.39</td></tr></table>

Visualization of Cross-Attention. To gain a more intuitive understanding of what the communication module learns, we visualize layout-to-image cross-attention during training and inference in Figure 9. During training, layout element tokens progressively focus on visually salient image regions, indicating that the module learns meaningful spatial relationships between layout elements and image content. At inference time, layout tokens already attend to salient regions at early denoising steps and become increasingly concentrated as denoising proceeds.

Communication Scheduling. In our implementation, we enable the communication module at the first 30% of the denoising process, which we find leads to overall good results. Let $\rho$ be the proportion of denoising steps in which the communication module is enabled; $\rho = 1 0 0 \%$ means the communication is enabled throughout all denoising steps, while $\rho = 0 \%$ disables the communication completely. We study the performance of our model at different $\rho$ values. The quantitative results are shown in Table 7. No communication $( \rho = 0 \% )$ yields the lowest image FID and highest image CLIP, but at the cost of degraded image-layout harmony: it yields the highest TemplateFID and its occlusion deviates significantly from that of real data, indicating that it fails to achieve layouts that harmonize well with the backgrounds. Excessive communication $( \rho ~ = ~ 1 0 0 \% )$ leads to the highest layout FID and image FID, suggesting that excessive cross-modal information exchange harms layout and image quality. $\rho \ : = \ : 3 0 \%$ yields the best image-layout harmonization (the best results in TemplateFID, occlusion, readability), while achieving high layout quality (the nearoptimal LayoutFID and the best alignment) and good image quality (competitive image FID and CLIP).

Generated  
Middle  
Late  
Ground Truth  
![](images/41e16db50a4e94041a797b48f74b1495d3736d6a6a6a5d63c5a29b06d33bb44a.jpg)  
Fig. 8. Qualitative comparison of our bidirectional communication module against its three ablated versions that allow no information exchange between image and layout (None), only allow layout-to-image information flow (Layout → Image), only allow image-to-layout information flow (Image → Layout). Each column uses the same input text and initial noise.  
Early  
Fig. 9. Layout-to-image cross-attention visualization in the communication module. Left: During training, layout element tokens (marked by golden stars) progressively attend to salient regions as the number of training steps increases from 0 to 1000. Right: At inference, they keep attending to salient image regions throughout all stages of the denoising process.

TABLE 7  
Quantitative performance of our model by enabling the communication module at the first $\rho$ of denoising steps. For all metrics except CLIP and FID-related ones, values closer to those calculated from real data indicate better performance. Best and second-best are bold and underlined, respectively.
<table><tr><td rowspan="2">ρ</td><td colspan="3">Image</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>Saliency Ratio</td><td>LayoutFID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>100%</td><td>22.84</td><td>27.97</td><td>14.31%</td><td>0.51</td><td>0.41</td><td>15.18</td><td>18.97%</td><td>13.38%</td><td>97.95</td><td>3.20</td></tr><tr><td>70%</td><td>22.39</td><td>28.44</td><td>14.53%</td><td>0.36</td><td>0.37</td><td>14.41</td><td>18.94%</td><td>12.79%</td><td>95.92</td><td>3.21</td></tr><tr><td>50%</td><td>21.66</td><td>28.42</td><td>14.34%</td><td>0.24</td><td>0.35</td><td>13.19</td><td>19.07%</td><td>12.37%</td><td>91.45</td><td>3.22</td></tr><tr><td>30%</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.15</td><td>0.31</td><td>11.98</td><td>19.30%</td><td>11.41%</td><td>86.63</td><td>3.25</td></tr><tr><td>10%</td><td>19.45</td><td>29.87</td><td>19.07%</td><td>0.14</td><td>0.32</td><td>11.41</td><td>26.76%</td><td>13.27%</td><td>89.42</td><td>3.22</td></tr><tr><td>0%</td><td>19.32</td><td>30.37</td><td>21.37%</td><td>0.16</td><td>0.38</td><td>12.87</td><td>30.74%</td><td>12.23%</td><td>99.14</td><td>3.23</td></tr><tr><td>Real Data</td><td></td><td>27.5</td><td>14.17%</td><td></td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td>一</td><td>3.39</td></tr></table>

![](images/ce1e11984018775c73a6f0a9eea031794a7256c6fa71d7fdc1e39253d73fcd38.jpg)  
Fig. 10. Qualitative results of enabling the communication module at the first ρ of the denoising steps. For each input text, all the results use identica initial noise and only $\rho$ is varied. Our implementation uses ρ = 30%.

TABLE 8  
Quantitative results of applying occlusion-aware guidance (OAG) and readability-aware guidance (RAG) to our model (Ours). For each metric except ones related to CLIP and FID, values closer to that of real data (bottom row) indicate better performance. The best and second-best numerical results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">Image</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>Saliency Ratio</td><td>LayoutFID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>Ours</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.15</td><td>0.31</td><td>11.98</td><td>19.30%</td><td>11.41%</td><td>86.63</td><td>3.25</td></tr><tr><td>Ours + OAG</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.21</td><td>0.32</td><td>13.26</td><td>12.66%</td><td>10.35%</td><td>92.81</td><td>3.23</td></tr><tr><td>Ours + RAG</td><td>19.57</td><td>29.79</td><td>14.56%</td><td>0.24</td><td>0.34</td><td>14.13</td><td>12.46%</td><td>7.49%</td><td>96.78</td><td>3.20</td></tr><tr><td>Real Data</td><td></td><td>27.50</td><td>14.17%</td><td></td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td></td><td>3.39</td></tr></table>

## 4.7 Preference-based Guidance

As shown in Table 8, applying occlusion-aware guidance (OAG) results in a significant reduction in occlusion from 19.30% to 12.66%, which means that occlusion of salient background regions is dramatically reduced. Using readability-aware guidance improves readability by a large margin (from 11.41% to 7.49%), suggesting that it effectively improves text readability. Note that enabling preferencebased guidance does compromise image quality, while leading to only minor degradation in other metrics (LayoutFID, alignment, overlap, TemplateFID, and TemplateCLIP). Figure 11 visually compares the results of our model against the variants obtained by applying OAG and RAG. It can be seen that OAG prevents foreground layout elements from occluding visually important objects in the background, while

RAG prefers to place foreground elements over uniform background regions.

## 5 CONCLUSION

In this paper, we present a generative model for graphic design templates based on a joint image-layout generation paradigm that contrasts with the sequential generation scheme commonly used in prior work. Our model is constructed within a latent diffusion framework and uses a specialized communication module to explicitly learn imagelayout interaction and jointly generate both an image and a layout in a single generative process. Our model is able to faithfully capture the joint distribution of images and layouts, thereby achieving high image-layout harmony. We also introduce a training-free guidance technique, which allows users to shift generated results away from the training data distribution and move them towards their preferred design patterns. Our experimental results demonstrate that our model is superior to existing methods, synthesizing visually appealing background images, high-quality layouts, and harmonious image-layout compositions.

![](images/70435a4254bb3fcf6c55d55546818c0db25c20e5327b1edc24a2b7d848f29ecb.jpg)  
Fig. 11. Qualitative results of preference-based guidance. Occlusion-aware guidance (OAG) prevents foreground elements from occluding important background regions; readability-aware guidance (RAG) improves text readability by putting foreground elements over flat background regions. White, orange, and red boxes denote text, underlay, and button, respectively.

## REFERENCES

[1] H. Weng, D. Huang, Y. Qiao, Z. Hu, C.-Y. Lin, T. Zhang, and C. L. P. Chen, “Desigen: A pipeline for controllable design template generation,” 2024.

[2] P. Jia, C. Li, Z. Liu, Y. Shen, X. Chen, Y. Yuan, Y. Zheng, D. Chen, J. Li, X. Xie et al., “Cole: A hierarchical generation framework for graphic design,” arXiv preprint arXiv:2311.16974, 2023.

[3] N. Inoue, K. Masui, W. Shimoda, and K. Yamaguchi, “OpenCOLE: Towards Reproducible Automatic Graphic Design Generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024.

[4] J. Lin, S. Sun, D. Huang, T. Liu, J. Li, and J. Bian, “From elements to design: A layered approach for automatic graphic design composition,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 8128–8137.

[5] Y. Qu, S. Fang, Y. Wang, X. Wang, Z. Chen, H. Xie, and Y. Zhang, “IGD: Instructional graphic design with multimodal layer generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 18 218–18 228.

[6] B. Yang and Y. Cao, “Order matters: Learning element ordering for graphic design generation,” ACM Trans. Graph., vol. 44, no. 4, Jul. 2025. [Online]. Available: https://doi.org/10.1145/3730858

[7] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[8] C. E. Jacobs, W. Li, E. Schrier, D. Bargeron, and D. Salesin, “Adaptive grid-based document layout,” ACM Transactions on Graphics, vol. 22, no. 3, pp. 838–847, 2003.

[9] P. O’Donovan, A. Agarwala, and A. Hertzmann, “Learning layouts for single-page graphic designs,” IEEE Transactions on Visualization and Computer Graphics, vol. 20, no. 8, pp. 1200–1213, 2014.

[10] X. Pang, Y. Cao, R. W. H. Lau, and A. B. Chan, “Directing user attention via visual flow on web designs,” ACM Transactions on Graphics, vol. 35, no. 6, pp. 240:1–240:11, 2016.

[11] X. Qiao, Y. Cao, and R. W. H. Lau, “Design order guided visual note layout optimization,” IEEE Transactions on Visualization and Computer Graphics, vol. 29, no. 9, pp. 3922–3936, 2023.

[12] J. Li, J. Yang, A. Hertzmann, J. Zhang, and T. Xu, “Layoutgan: Generating graphic layouts with wireframe discriminators,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. [Online]. Available: https://openreview.net/forum?id=HJxB5sRcFQ

[13] X. Zheng, X. Qiao, Y. Cao, and R. W. Lau, “Content-aware generative modeling of graphic design layouts,” ACM Transactions on Graphics (TOG), vol. 38, no. 4, pp. 1–15, 2019.

[14] K. Kikuchi, E. Simo-Serra, M. Otani, and K. Yamaguchi, “Constrained graphic layout generation via latent optimization,” in ACM International Conference on Multimedia, ser. MM ’21, 2021, pp. 88–96.

[15] A. A. Jyothi, T. Durand, J. He, L. Sigal, and G. Mori, “Layoutvae: Stochastic scene layout generation from a label set,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 9895–9904.

[16] D. M. Arroyo, J. Postels, and F. Tombari, “Variational transformer networks for layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 13 642–13 652.

[17] K. Gupta, J. Lazarow, A. Achille, L. S. Davis, V. Mahadevan, and A. Shrivastava, “Layouttransformer: Layout generation and completion with self-attention,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1004–1014.

[18] D. Horita, N. Inoue, K. Kikuchi, K. Yamaguchi, and K. Aizawa, “Retrieval-augmented layout transformer for content-aware layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 67–76.

[19] Z. Jiang, J. Guo, S. Sun, H. Deng, Z. Wu, V. Mijovic, Z. J. Yang, J.-G. Lou, and D. Zhang, “Layoutformer++: Conditional graphic layout generation via constraint serialization and decoding space

restriction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 403–18 412.

[20] N. Inoue, K. Kikuchi, E. Simo-Serra, M. Otani, and K. Yamaguchi, “Layoutdm: Discrete diffusion model for controllable layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 10 167–10 176.

[21] J. Zhang, J. Guo, S. Sun, J.-G. Lou, and D. Zhang, “Layoutdiffusion: Improving graphic layout generation by discrete diffusion probabilistic models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7226–7236.

[22] C.-Y. Cheng, F. Huang, G. Li, and Y. Li, “Play: parametrically conditioned layout generation using latent diffusion,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.

[23] J. J. A. Guerreiro, N. Inoue, K. Masui, M. Otani, and H. Nakayama, “Layoutflow: flow matching for layout generation,” in European Conference on Computer Vision. Springer, 2024, pp. 56–72.

[24] J. Lin, J. Guo, S. Sun, Z. Yang, J.-G. Lou, and D. Zhang, “Layoutprompter: Awaken the design ability of large language models,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[25] H.-Y. Lee, L. Jiang, I. Essa, P. B. Le, H. Gong, M.-H. Yang, and W. Yang, “Neural design network: Graphic layout generation with constraints,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer, 2020, pp. 491–506.

[26] Y. Cheng, Z. Zhang, M. Yang, N. Hui, C. Li, X. Wu, and J. Shao, “Graphic design with large multimodal model,” arXiv preprint arXiv:2404.14368, 2024.

[27] M. A. Shabani, Z. Wang, D. Liu, N. Zhao, J. Yang, and Y. Furukawa, “Visual layout composer: Image-vector dual diffusion model for design layout generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 9222– 9231.

[28] H. Zhang, D. Hong, M. Yang, Y. Chen, Z. Zhang, J. Shao, X. Wu, Z. Wu, and Y.-G. Jiang, “Creatidesign: A unified multi-conditional diffusion transformer for creative graphic design,” arXiv preprint arXiv:2505.19114, 2025.

[29] M. Zhou, C. Xu, Y. Ma, T. Ge, Y. Jiang, and W. Xu, “Composition-aware graphic layout gan for visual-textual presentation designs,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, 7 2022, pp. 4995–5001, aI and Arts. [Online]. Available: https://doi.org/10.24963/ijcai.2022/692

[30] Y. Cao, Y. Ma, M. Zhou, C. Liu, H. Xie, T. Ge, and Y. Jiang, “Geometry aligned variational transformer for image-conditioned layout generation,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 1561–1571.

[31] J. Seol, S. Kim, and J. Yoo, “Posterllama: Bridging design ability of langauge model to contents-aware layout generation,” ECCV, 2024.

[32] H. Wang, B. Zhao, J. Wang, H. Wang, H. Yang, W. Ji, H. Liu, and X. Xiao, “Sega: A stepwise evolution paradigm for contentaware layout generation with design prior,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 19 321–19 330.

[33] K. Yamaguchi, “Canvasvae: Learning to generate vector graphic documents,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5481–5489.

[34] Z. Wang, J. Bao, S. Gu, D. Chen, W. Zhou, and H. Li, “Designdiffusion: High-quality text-to-design image generation with diffusion models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 20 906–20 915.

[35] S. Chen, J. Lai, J. Gao, T. Ye, H. Chen, H. Shi, S. Shao, Y. Lin, S. Fei, Z. Xing et al., “Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework,” arXiv preprint arXiv:2506.10741, 2025.

[36] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[37] Y. Li, H. Liu, Q. Wu, F. Mu, J. Yang, J. Gao, C. Li, and Y. J. Lee, “Gligen: Open-set grounded text-to-image generation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22 511–22 521.

[38] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a

local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.

[39] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[40] M. Zhou, C. Xu, Y. Ma, T. Ge, Y. Jiang, and W. Xu, “Composition-aware graphic layout gan for visual-textual presentation designs,” in Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, L. D. Raedt, Ed. International Joint Conferences on Artificial Intelligence Organization, 7 2022, pp. 4995–5001, aI and Arts. [Online]. Available: https://doi.org/10.24963/ijcai.2022/692

[41] H. Hsu, X. He, Y. Peng, H. Kong, and Q. Zhang, “Posterlayout: A new benchmark and approach for content-aware visual-textual presentation layout,” in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6018–6026.

[42] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, “beta-vae: Learning basic visual concepts with a constrained variational framework,” in International conference on learning representations, 2017.

## APPENDIX A

## SUPPLEMENTARY MATERIAL

## A.1 Interpretability and Effectiveness of the Communication Module

## A.1.1 Training Stage Attention Map Visualization

To better understand how the layout and image modalities interact during training, we visualize cross-attention maps in both directions: from layout tokens to image patches and from image patches to layout tokens.

As shown in Figure S1, the attention from layout tokens to image features evolves significantly during training. In the early stage (training step 0), each layout token attends almost uniformly to the entire image, without clear spatial preference. As training progresses, however, these attentions become sharper and more semantically meaningful: after 1000 steps, layout tokens are able to consistently focus on salient objects and object boundaries (e.g., text regions or buttons), demonstrating that the communication module learns to capture the spatial layout of the background image and align layout elements with visual context.

Complementary to this observation, Figure S2 presents the reverse attention flow, from image patches to layout tokens. We find that patches corresponding to highly salient regions (e.g., human faces) have a much stronger influence on the layout: their attention maps exhibit consistently high responses across almost all bounding boxes. In contrast, patches from less salient background areas tend to affect only a subset of layout tokens and show much weaker activations overall.

## A.1.2 Inference Stage Attention Map Visualization

We further inspect the cross-attention weights from layout tokens to image patches during inference as the denoising process unfolds. As shown in Figure S3, even at early timesteps (t = 701), layout tokens already focus on salient regions of the image, such as objects and prominent anchors. As denoising progresses, these attentions sharpen and stabilize, indicating that the communication module dynamically refines its cross-modal grounding. Importantly, this behavior is consistent with the training stage observations: the communication module preserves the spatial alignment ability it learned during training and can readily attend to salient regions and boundaries in the image during inference. This consistency suggests that the learned crossmodal alignment is not only acquired in training, but also effectively retained and exploited during inference.

## A.2 GPT-Based Evaluation

## A.2.1 GPT-5 Evaluation

Inspired by recent work on graphic design evaluation [2], [3], we leverage GPT-5 to automatically assess the quality of generated design templates. Given a design template, GPT-5 is instructed to evaluate it across four complementary aspects, each scored on a scale of 1 to 10, where higher values indicate better performance. For each candidate model (Desigen, OpenCOLE, and ours), we generate 2,000 design templates using text inputs randomly sampled from the test set. The aggregated GPT-5 scores are reported in Table S1. To ensure consistency, we design a specialized prompt tailored for template evaluation, as detailed in PromptBox S1.

The four evaluation aspects capture different dimensions of design quality:

• Image Quality: Measures the visual fidelity, realism, and aesthetic appeal of the generated background image.

• Layout Quality: Evaluates the structural integrity and usability of the layout, including alignment, spacing, proportions, overlap avoidance, and readability.

• Image–Layout Harmony: Assesses how well the generated layout integrates with the background image, including visibility, contrast, and avoidance of distracting conflicts with salient regions.

• Text–Design Relevance: Evaluates whether the generated template is semantically relevant to the input text prompt in both visual content and overall design intent.

As shown in Table S1, our model consistently outperforms existing baselines across all four evaluation aspects. In Image Quality, it achieves 6.94, higher than Desigen (6.42) and OpenCOLE (6.81), and close to real data (7.44). In Layout Quality, our method reaches 7.76, substantially improving over Desigen (6.54) and OpenCOLE (6.02), and approaching real data (7.87). The advantage is also clear in Image–Layout Harmony, where our score of 8.07 exceeds Desigen (7.47) and OpenCOLE (5.84), indicating more coherent multimodal compositions. Finally, in Text–Design Relevance, our score of 7.34 is notably better than Desigen (6.21) and OpenCOLE (6.38), and close to real data (7.49), suggesting stronger semantic alignment between the generated template and the input prompt. Overall, these GPT-5 results provide additional evidence that our method produces more coherent and relevant design templates than both Desigen and OpenCOLE.

TABLE S1  
GPT-5 evaluation results across four aspects of design quality: (i) Image Quality, (ii) Layout Quality, (iii) Image–Layout Harmony, (iv) Text–Design Relevance. The best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>(i)</td><td>(ii)</td><td>(iii)</td><td>(iv)</td></tr><tr><td rowspan="2">Desigen OpenCOLE</td><td>6.42</td><td>6.54</td><td>7.47</td><td>6.21</td></tr><tr><td>6.81</td><td>6.02</td><td>5.84 8.07</td><td>6.38 7.34</td></tr><tr><td>Ours Real Data</td><td>6.94 7.44</td><td>7.76 7.87</td><td>8.39</td><td>7.49</td></tr></table>

## A.3 Blind Human Preference Study on Generated Templates

To complement the automatic and metric-based evaluations, we conduct a blind forced-choice study comparing templates generated by InterIL, Desigen, and OpenCOLE. The study involves 32 participants and 40 text prompts. For each prompt, outputs from the three methods are presented with their method identities hidden and their display order randomized. Participants are asked to select their preferred result separately according to four aspects: image quality, layout quality, image–layout harmony, and text–design relevance.

Across all responses, InterIL is selected as the preferred result in 68% of comparisons for image quality, 78% for layout quality, 83% for image–layout harmony, and 71% for text–design relevance. These human judgments are consistent with the GPT-5 results in Table S1 and with the TemplateFID ranking, providing direct perceptual evidence for the quality and coherence of the generated templates.

![](images/070065f0e9f93952336f38b002bde05c9603ddd42e01e31abc939cbb40b2edf8.jpg)  
Fig. S1. Visualization of layout-to-image cross-attention across training. The leftmost column shows the ground-truth layout, where the bounding box marked with a golden star indicates the selected layout token. The three images to the right illustrate how this token attends to different regions of the background image at training steps 0, 200, and 1000. As training progresses, the attention becomes increasingly focused on salient objects, indicating that the communication module gradually learns meaningful spatial correspondences between layout elements and image content.

## A.4 Preference-Specific Guidance

## A.4.1 Occlusion-Aware Guidance (OAG)

To illustrate how the proposed inference-time guidance framework can accommodate different user preferences, we first analyze occlusion-aware guidance (OAG) as one concrete example. To qualitatively assess its effect during generation, we visualize the denoising trajectory of a sample under guidance scale $s \ : = \ : 3$ . As shown in Figure S4, the layout elements initially overlap with salient regions of the image (e.g., foreground objects). As denoising proceeds, however, these elements are gradually pushed away from the salient content and repositioned into less intrusive areas. This stepwise visualization demonstrates how one guidance objective can steer the layout toward a user-preferred lowocclusion solution at test time.

## A.4.2 Ablation on OAG Scale

We conduct an ablation study on the OAG guidance scale s to better understand this particular guidance instantiation. Since OAG directly influences only the placement of layout elements, we evaluate both layout-level and template-level metrics (Table S2).

As s increases, the model becomes more conservative in placing layout elements over salient regions. Occlusion decreases consistently and reaches its minimum at larger scales $( \mathrm { e . g . } , s = 1 0 )$ , indicating that stronger OAG more aggressively enforces avoidance of important background areas. Readability also improves with moderate scales (around $s = 3 \mathrm { ~ t o ~ } s = 5 )$ , but excessive values lead to diminishing returns. At the same time, stronger guidance introduces side effects: LayoutFID and alignment gradually deteriorate, and template-level realism is harmed. Specifically, TemplateFID rises steadily $( \mathrm { e } . \mathrm { g } . , \ 8 6 . 6 3 \  \ 1 1 4 . 3 \bar { 4 } )$ , while TemplateCLIP drops slightly $( 3 . 2 5  3 . 1 6 )$ , suggesting that overly strong preference steering distorts the natural distribution of layouts and weakens overall design quality.

A moderate guidance scale achieves the best trade-off. In particular, s = 2 and s = 3 strike a balance between reducing occlusion and maintaining competitive LayoutFID, alignment, and template-level realism. Qualitative results in Figure S5 confirm this trend: at $s = 3 ,$ , layout elements are already well separated from salient objects in the background, resulting in cleaner and more harmonious compositions without sacrificing fidelity. Therefore, we adopt $s = 3$ as the default OAG setting in our main experiments. Larger scales $( \mathbf { e . g . } , s = 7 \ \mathrm { o r } \ s = 9 )$ can be useful when strict occlusion avoidance is preferred, but they may come at the cost of reduced layout fidelity and visual realism. This trade-off is central to the overall paper narrative: the unguided model best matches the joint data distribution, whereas guidance is intended to provide additional flexibility for users with more specific preferences.

## A.4.3 Readability-Aware Guidance (RAG)

RAG is the second guidance example considered in the paper. It uses the same inference-time steering mechanism

Low  
High  
![](images/136810f8c785f4df19057a99da9c64e2e09eb46279f034c36856baa3a3710273.jpg)  
Fig. S2. Cross-attention visualization from image patches to layout elements at different denoising timesteps. Each red dashed box marks a selected image patch, and the color overlay on each layout element (red = high attention, blue = low attention) indicates how strongly that patch influences the layout token. We observe that patches located in visually salient regions (e.g., faces, objects) exert much stronger influence on nearly all layout boxes, whereas patches from non-salient background areas contribute weakly and only to a subset of elements.

TABLE S2  
Ablation on different OAG scales (s) using 50 DDIM steps. This table analyzes one concrete guidance instantiation rather than the core unguided model itself. For each metric, the best result is marked in bold, and the second best is underlined. The last row shows statistics computed from real data as reference.
<table><tr><td rowspan="2">Setting</td><td colspan="5">Layout</td><td colspan="2">Template</td></tr><tr><td>L-FID↓</td><td>Align</td><td>Overlap</td><td>Occlusion</td><td>Readability</td><td>TemplateFID↓</td><td>TemplateCLIP↑</td></tr><tr><td>s=0</td><td>0.148</td><td>0.31</td><td>11.98</td><td>19.30%</td><td>11.41%</td><td>86.63</td><td>3.25</td></tr><tr><td>s=1</td><td>0.1547</td><td>0.42</td><td>11.90</td><td>15.40%</td><td>10.50%</td><td>87.05</td><td>3.25</td></tr><tr><td>s=2</td><td>0.1874</td><td>0.38</td><td>11.38</td><td>13.40%</td><td>10.27%</td><td>89.41</td><td>3.24</td></tr><tr><td>s=3</td><td>0.2113</td><td>0.32</td><td>13.26</td><td>12.66%</td><td>10.35%</td><td>92.81</td><td>3.23</td></tr><tr><td>s=4</td><td>0.2274</td><td>0.36</td><td>11.80</td><td>12.21%</td><td>10.25%</td><td>94.35</td><td>3.23</td></tr><tr><td>s=5</td><td>0.2899</td><td>0.34</td><td>13.47</td><td>11.22%</td><td>10.26%</td><td>104.83</td><td>3.22</td></tr><tr><td>s=6</td><td>0.2946</td><td>0.41</td><td>13.06</td><td>11.22%</td><td>10.19%</td><td>104.32</td><td>3.21</td></tr><tr><td>s=7</td><td>0.3324</td><td>0.34</td><td>13.79</td><td>10.35%</td><td>9.97%</td><td>109.44</td><td>3.21</td></tr><tr><td>s=8</td><td>0.3658</td><td>0.48</td><td>13.58</td><td>10.07%</td><td>10.01%</td><td>112.37</td><td>3.19</td></tr><tr><td>s=9</td><td>0.3689</td><td>0.46</td><td>13.85</td><td>10.25%</td><td>10.22%</td><td>112.48</td><td>3.18</td></tr><tr><td>s=10</td><td>0.4426</td><td>0.47</td><td>13.73</td><td>9.71%</td><td>10.05%</td><td>114.34</td><td>3.16</td></tr><tr><td>Real Data</td><td></td><td>0.31</td><td>9.34</td><td>21.14%</td><td>8.53%</td><td></td><td>3.39</td></tr></table>

![](images/fb3e35c8271bb42990efddd7c106b778e6cb564035cb45ddea2d376d71b85dc4.jpg)  
Fig. S3. Cross-attention visualization from layout tokens to image patches across early, middle, and late denoising stages. For several representative layout tokens, we highlight the image regions they attend to. Even at early steps, the tokens already focus on salient objects, and this focus becomes sharper as denoising progresses, indicating that the communication module learns meaningful cross-modal correspondences throughout the generation process.

## PromptBox S1: System Prompt for GPT-5 Evaluation

You are an autonomous AI Assistant who evaluates graphic design templates with objectivity, precision, and consistency. Your goals are: Provide unbiased and actionable critiques of design templates based on established visual design principles. Assess image quality, layout quality, image-layout harmony, and text-design relevance. Maintain strict scoring standards and concise reasoning.

All coordinates follow the convention that the upper-left corner is the origin, the x-axis increases rightward, and the y-axis increases downward.

Please abide by the following rules: Score as objectively as possible. A flawless template may score 10 points, a mediocre one around 7, a template with clear issues around 4, and a very poor template 1–2 points. Keep reasoning brief. Grading criteria:

• Image Quality (1–10): Evaluate the visual fidelity, realism, and aesthetic appeal of the background image. High scores reflect sharp details, natural textures, clean rendering, and an overall pleasing composition. Low scores indicate blurriness, artifacts, noise, distortions, or visually incoherent backgrounds.

as OAG, but replaces the saliency-based objective with a clutter-based readability objective defined on text regions. This variant is useful when users specifically prefer text to be placed on smoother, lower-texture background regions. Consistent with the quantitative results in the main paper, • Layout Quality (1–10): Evaluate the intrinsic structural soundness and usability of the bounding-box layout. High-scoring layouts exhibit precise alignment, consistent spacing, balanced proportions, minimal overlap, low unintended occlusion, and clear readability. Low scores reflect misalignment, irregular spacing, excessive overlap, poorly sized regions, or other structural flaws that hinder clarity and usability.

• Image–Layout Harmony (1–10): Evaluate how effectively the layout interacts with the background image. High scores indicate strong contrast, low visual interference, minimal conflict with salient regions, and placements that enhance visibility and focus. Low scores reflect distracting overlaps, poor visibility, or layout elements placed on cluttered or visually dominant areas that reduce clarity.

• Text–Design Relevance (1–10): Evaluate whether the generated design is semantically relevant to the input text prompt. High scores indicate that the visual content, layout decisions, and overall style match the intended theme, product, or message expressed in the prompt. Low scores indicate weak semantic correspondence, mismatched imagery, or a design that fails to reflect the intended content or communicative purpose.

RAG improves readability-oriented preference metrics, but it may also reduce holistic realism and fidelity to the joint data distribution. As with OAG, RAG should therefore be interpreted as an optional preference-specific control rather than a replacement for the base model.

![](images/c953c321107c3412b21208838f353e135255e9f9f720bd684d74998cf3d74940.jpg)  
(a) Step 800

![](images/b13a1418aeb2bab4def2191c6b83f5814374af297bdcc1645a8ccd4e42a9f13f.jpg)  
(b) Step 500

![](images/82ad9e6ef9840d2d245189b06971ec5a46b58d89c20a6aa8112366fcb1711e79.jpg)  
(c) Step 200

![](images/fbe4f408b6a84cd747b95f1bc0271d9114410acfb8742510b084948b0f7ee295.jpg)  
(d) Step 0

Fig. S4. Denoising trajectory under occlusion-aware guidance (occ scale = 3), shown as the first guidance example in our framework. The layout is gradually pushed away from salient image regions as denoising progresses, demonstrating how OAG steers the model toward low-occlusion configurations.  
![](images/91b3fd480e3cf143dabe613d1c5fc04fe3963ce94a3e56e6c9bea5ffa97e5d46.jpg)  
Fig. S5. Visualization of two samples under different OAG scales. From left to right: guidance scale = 0, 1, and 3. The comparison shows how stronger OAG encourages layout elements to avoid salient regions more consistently, resulting in cleaner and more readable templates.

## A.5 User Study on Template Embedding Similarity

To further validate the perceptual quality of our learned template embedding space, we conducted a user study involving 104 participants, including 58 design experts and 46 non-experts. We include non-experts because design templates are also consumed and adapted by general users in practical applications. Each participant was shown triplets of the form $( x , \tilde { x } _ { \mathrm { r a n d } } , \tilde { x } _ { \mathrm { e m b } } )$ , where x is a reference template, $\tilde { x } _ { \mathrm { r a n d } }$ is a randomly selected template from the Web-design dataset, and $\tilde { x } _ { \mathrm { e m b } }$ is the most similar template to x retrieved by our learned embeddings. Although the embedding model is trained on a composed dataset that excludes Web-design, participants were asked to choose, based on their intuition, which of $\tilde { x } _ { \mathrm { r a n d } }$ or $\tilde { x } _ { \mathrm { e m b } }$ is more similar to x, considering both layout and image content.

Figure S6 shows three representative cases, with the Preference Rate on the right indicating how often $\tilde { x } _ { \mathrm { e m b } }$ was chosen over $\tilde { x } _ { \mathrm { r a n d } }$ . In Case 1, $\tilde { x } _ { \mathrm { e m b } }$ was preferred 90% of the time, in Case 2 the rate was 77.5%, and in Case 3 it was 72.5%. Across all 30 triplets, $\tilde { x } _ { \mathrm { e m b } }$ was chosen in 83.4% of comparisons, demonstrating that our learned embeddings align well with human perception. These results also suggest that the proposed TemplateFID metric is consistent with human judgments of template similarity.

## A.6 Dataset

We conduct our experiments on the Web-design dataset [1], which consists of around 50K web banner designs collected from real-world online shopping platforms. Each sample includes a high-resolution background image, structured layout annotations (bounding boxes and element types), and a product description that conveys information such as commercial purpose, theme, or target audience. We use 41,270 samples (85%) for training, 2,427 samples (5%) for validation, and 4,856 samples (10%) for testing.

![](images/f1b34a3846a0bacefaf3029040e044d6fd1735cdafd6ac8cee7b24fb33a1703d.jpg)  
Fig. S6. User study examples. Each row shows one triplet: reference x (left), embedding-based retrieval $\tilde { x } _ { \mathrm { e m b } }$ (middle), and random retrieval $\tilde { x } _ { \ r a n d }$ (right). The Preference Rate on the right shows how often $\tilde { x } _ { \mathsf { e m b } }$ was chosen by participants. Blue bounding boxes denote text regions, while red bounding boxes denote button elements.

## A.7 Implementation Details of the InterIL Model

Layout Prior. Each layout is represented as a sequence of up to 7 elements, where each element is defined by five attributes: category, left coordinate, top coordinate, width, and height. We discretize each attribute into 64 values by replacing raw coordinates with their nearest k-means cluster index. Layouts with fewer than 7 elements are padded to a fixed length. Notably, only about 0.97% of layouts in the dataset contain more than 7 elements.

Image Prior. For a fair comparison between our method, OpenCOLE, and Desigen, we use the same SD-v1.4 backbone and set the output resolution to 512 × 512, consistent with Desigen. For the teaser in the supplementary material, we adopt SD-v2.0 with an output resolution of $7 6 8 \times 7 6 8$

We use a β-VAE to model the layout, with a latent size of 32. The VAE encodes layout tokens into compact latent representations. For layout denoising, we adopt a transformerbased architecture using DiT as the backbone. Specifically, our Layout LDM consists of 28 transformer blocks, each with 8 attention heads and a query/key/value dimension of 512. This configuration enables the model to effectively capture spatial dependencies and hierarchical structures across layout elements during the denoising process.

Communication Module. In the joint modeling stage, we introduce a communication module that facilitates feature exchange between the layout and image backbones. Specifically, we extract multi-scale image features from the second and third downsampling blocks as well as the middle block of the image U-Net, and use the output features of the 19th transformer block of the layout diffusion model as the layout representation. Cross-attention is performed in both directions: the layout representation attends to the image features to incorporate visual context, while the image features attend to the layout representation to integrate structural information. The image-informed layout representation is fused with the original 19th-block output through a residual connection and then fed into the 20th transformer block; analogously, the attended image representations are fused back into the corresponding U-Net blocks. During this stage, both the pre-trained layout diffusion model and image backbone are kept frozen, and only the communication module is updated.

Inference Details. During inference, we use a DDIM sampler for the layout backbone and a DDPM sampler for the image backbone, each with 50 sampling steps. The communication module is enabled at timesteps greater than 700 (corresponding to $\rho = 3 0 \%$ of the denoising trajectory) to facilitate cross-modal information exchange. In addition, both occlusion-aware guidance (OAG) and readabilityaware guidance (RAG) are applied between steps 200 and 700.

![](images/4a36eafb6bce230153492ca400fcc5bfef4f633772974a422cfc8f20554efa1f.jpg)  
Fig. S7. Architecture of the template autoencoder (TemplateAE). The frozen image encoder $\mathcal { E } _ { \vert }$ and the independent TemplateAE layout encoder $\mathcal { E } _ { \mathrm { L } } ^ { ' }$ map the image and layout to their latent spaces. The independent layout encoder $\mathcal { E } _ { \mathrm { L } } ^ { ' }$ is trained on the mixture dataset and is not shared with InterIL. The concatenated embeddings are processed by the template encoder $\mathcal { E } ^ { T }$ to produce a unified design latent $z ^ { T }$ , which is then decoded by the template decoder $\bar { \mathcal { D } } ^ { T }$ and two MLPs to reconstruct the image latent and layout logits.

## A.8 Training Details

Our training pipeline consists of two main stages: (1) Building Domain-specific Priors, which includes pretraining the layout VAE, training the Layout LDM, and fine-tuning Stable Diffusion; and (2) Training the Joint Model, which focuses on learning the cross-domain communication module. All experiments are conducted on an NVIDIA A40 GPU with 48 GB of memory.

A.8.0.1 Stage 1: Building Domain-specific Priors.:

• Layout VAE Training. We employ a β-VAE [42] to encode layouts into latent representations. The $\beta$ coefficient is set to $5 \times 1 0 ^ { - 4 }$ to find a sweet spot between reconstruction accuracy and latent disentanglement.

• Layout LDM Training. The layout diffusion model is trained to model the distribution of latent layouts. We use a batch size of 4096 and train for 1000 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$

• Stable Diffusion Fine-tuning. To ensure fair comparison with Desigen [1], the previous state-of-the-art method, we fine-tune the Stable Diffusion model on the corresponding dataset. The model is trained for 100 epochs with a batch size of 8 and a learning rate of $1 \times 1 0 ^ { - 5 }$

## A.8.0.2 Stage 2: Training the Joint Model.:

• Cross-Domain Communication Module. The feature exchange module is trained independently to facilitate information transfer between the layout and image domains. The model is trained for 20 epochs with a batch size of 20 and a learning rate of $1 \times 1 0 ^ { \dot { - } 4 }$

A.8.0.3 Optimization.: All training stages adopt the AdamW optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and a weight decay of $\dot { 1 } \times 1 0 ^ { - 2 }$ . The learning rate schedule follows a cosine decay strategy.

## A.9 Implementation Details of Preference-Specific Guidance

We now detail the two guidance objectives used in the paper. Both are plugged into the same inference-time guidance framework and differ only in the choice of differentiable objective computed from the decoded image-layout predictions.

Occlusion-Aware Guidance (OAG). Given the predicted image noise $\hat { \epsilon } _ { t } ^ { I } ,$ , we compute a clean image latent $\hat { z } _ { 0 } ^ { I }$ analytically using the forward-process marginal distribution $q \big ( z _ { t } ^ { I } | z _ { 0 } ^ { I } \big )$ , decode it into an image $\mathcal { D } _ { \mathrm { I } } ( \hat { z } _ { 0 } ^ { \check { I } } )$ , and compute a saliency map $\begin{array} { r } { S \ = \ f _ { \mathrm { s a l } } ( \mathcal { D } _ { \mathrm { I } } ( \tilde { z } _ { 0 } ^ { I } ) ) } \end{array}$ using an off-the-shelf saliency detector $f _ { \mathrm { s a l } }$ . We then compute a clean layout latent $\hat { z } _ { 0 } ^ { L }$ from the predicted layout noise $\hat { \epsilon } _ { t } ^ { L }$ with $q \big ( z _ { t } ^ { \mathbf { \breve { L } } } | z _ { 0 } ^ { L } \big )$ , and decode it into a layout $\mathcal { D } _ { \mathrm { L } } ^ { \mathrm { ~ \scriptsize ~ ( ~ \hat { z } _ { 0 } ^ { L } ) ~ } }$ . Let $\{ \dot { \mathbf { b } _ { i } } \} _ { i = 1 } ^ { B }$ be the decoded element bounding boxes. The OAG objective is written as:

$$
\mathcal { L } _ { \mathrm { o c c } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 1 } { A _ { i } } \sum _ { p } M _ { \mathbf { b } _ { i } } ( p ) S ( p ) ,\tag{S1}
$$

where $M _ { \mathbf { b } _ { i } }$ is a soft mask for $\mathbf { b } _ { i } , A _ { i }$ is the area of $\mathbf { b } _ { i } ,$ and $p$ indexes spatial positions.

Since the decoded layout elements have discrete bounding box parameters, we convert them into continuous ones to make the guidance objectives differentiable. We estimate each continuous parameter as a weighted sum of quantization-bin centers, where the weights are the predicted probabilities over bins. We denote by $\begin{array} { r l } { \mathbf { b } _ { i } } & { { } = } \end{array}$ $\mathbf { \hat { \Phi } } ( x _ { i } , y _ { i } , w _ { i } , \hat { h } _ { i } ) \in \mathbb { R } ^ { 4 }$ the estimated continuous bounding box parameters of the i-th element, where $( x _ { i } , y _ { i } )$ denotes the top-left location, and $w _ { i }$ and $h _ { i }$ are, respectively, the width and height. From $\mathbf { b } _ { i } ,$ we compute the top-left coordinates $( x _ { i } ^ { l } , y _ { i } ^ { t } )$ and the bottom-right coordinates $( x _ { i } ^ { r } , y _ { i } ^ { b } )$ of the bounding box. The soft mask $M _ { \mathbf { b } _ { i } }$ is computed in a differentiable manner, with the value at position $( x , y ) { \mathrm { ; } }$

$$
\begin{array} { r l } & { M _ { \mathbf { b } _ { i } } ( x , y ) = \sigma ( \lambda ( x - x _ { i } ^ { l } ) ) \times \sigma ( \lambda ( x _ { i } ^ { r } - x ) ) } \\ & { \qquad \times \sigma ( \lambda ( y - y _ { i } ^ { t } ) ) \times \sigma ( \lambda ( y _ { i } ^ { b } - y ) ) , } \end{array}\tag{S2}
$$

where $\sigma ( \cdot )$ is the sigmoid function. $\lambda$ is set to 40. This differentiable box representation is used in both OAG and ${ \mathrm { R A G } } .$

Readability-Aware Guidance (RAG). For ${ \mathrm { R A G } } ,$ we use the same decoded image $\mathcal { D } _ { \mathrm { I } } ( \hat { z } _ { 0 } ^ { I } )$ and decoded layout $\mathcal { D } _ { \mathrm { L } } \big ( \hat { z } _ { 0 } ^ { L } \big )$

but replace the saliency map with a differentiable clutter map $\bar { C }$ computed from spatial gradient operators on the decoded image, so that regions with stronger edges, textures, or other high-frequency patterns receive larger values. From the decoded layout, we select the subset of boxes corresponding to text elements, denoted by $\{ { \bf b } _ { i } ^ { \mathrm { t e x t } } \} _ { i = 1 } ^ { B _ { T } }$ . The RAG objective is written as:

$$
\mathcal { L } _ { \mathrm { r e a d } } = \frac { 1 } { B _ { T } } \sum _ { i = 1 } ^ { B _ { T } } \frac { 1 } { A _ { i } } \sum _ { p } M _ { \mathbf { b } _ { i } ^ { \mathrm { t e x t } } } ( p ) C ( p ) ,\tag{S3}
$$

where $M _ { \mathbf { b } _ { i } ^ { \mathrm { t e x t } } }$ is the soft mask for the i-th decoded text box. In this way, OAG and RAG share the same inferencetime mechanism and differ only in their preference-specific objective functions.

## A.10 TemplateAE Architecture Details

To enable holistic evaluation of design templates, we train a template autoencoder (TemplateAE) to extract joint image-layout embeddings. For TemplateFID, TemplateAE is trained on a large-scale mixture dataset comprising Gen-Poster100K [32], CGL [40], Crello [33], and PKU [41], deliberately excluding Web-design to avoid evaluation bias.

Before training TemplateAE, we separately train another Layout VAE on the same mixture dataset and use its encoder, denoted by $\mathcal { E } _ { \mathrm { L } } ^ { ' } ,$ , as the frozen layout encoder of TemplateAE. This Layout VAE is independent of the one used by InterIL and shares neither parameters nor training data with it.

As shown in Figure S7, given a background image $X ^ { I }$ and layout $X ^ { L }$ , they are first mapped to latent space through the frozen image encoder ${ \mathcal { E } } _ { \mathrm { I } }$ and the independent layout encoder $\mathcal { E } _ { \mathrm { L } } ^ { ' }$ to obtain $z ^ { I } = \mathcal { E } _ { \mathrm { I } } ( X ^ { I } )$ and $z ^ { L } = \dot { \mathcal { E } } _ { \mathrm { I } } ^ { ' } ( X ^ { L } )$ These embeddings are concatenated and processed through a template encoder ${ \mathcal { E } } ^ { T }$ to produce a unified design latent $z ^ { T }$ . Subsequently, $z ^ { T }$ is combined with cosine position embeddings and fed into a template decoder $\mathcal { D } ^ { T }$ , followed by two separate MLPs that reconstruct the respective domain outputs: image latent $\hat { z } ^ { I }$ and layout logits $\dot { \hat { y } } ^ { L } \in \mathbb { R } ^ { N \times d ^ { L } \times V }$ During training, we freeze ${ \mathcal { E } } _ { \mathrm { I } }$ and $\mathcal { E } _ { \mathrm { L } } ^ { ' }$ and only optimize $\mathcal { E } ^ { T } .$ $\mathcal { D } ^ { T }$ , and the MLPs. The training objective on this mixture dataset combines reconstruction losses for both modalities:

$$
\mathcal { L } _ { \mathrm { r e c } } = \gamma \big ( 1 - \frac { z ^ { I } \cdot \hat { z } ^ { I } } { \| z ^ { I } \| \| \hat { z } ^ { I } \| } \big ) + \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { V } - X _ { i , j } ^ { L } \log ( \hat { y } _ { i , j } ^ { L } )\tag{S4}
$$

where the first term measures image latent reconstruction quality and the second term evaluates layout prediction accuracy. γ is the balance factor that brings the two terms to comparable scales; in our experiments, we set $\gamma = 5 . 0 .$ For TemplateCLIP, we initialize from this mixture-datasetpretrained TemplateAE encoder and further fine-tune it on the Web-design train split together with the pretrained SD text encoder using a CLIP-style contrastive objective.

## A.11 Further Inference Time Analysis

All inference times reported in Table S3 are measured on a single NVIDIA A40 GPU for fair and consistent comparison. Varying the communication ratio $\rho$ has only a minor effect on inference time, which decreases from 5.24 s at $\rho = 1 0 0 \%$ to 4.91 s at $\rho = 0 \%$

Table S3 summarizes the average time required to generate one design template across different models on the Web-design test set. Our joint diffusion framework achieves substantially faster generation than both Desigen [1] and OpenCOLE [2], while maintaining superior design quality.

Specifically, our base model requires only 5.01 s per template on average, achieving a $1 . 5 \times - 6 \times$ speed-up over Desigen variants, whose multi-stage iterative refinement introduces considerable latency at each iteration. Even with guidance variants such as OAG and $\mathbf { R A G } ,$ or with GPT-augmented prompts, our variants remain competitive (8.69 s, 8.47 s, and 10.14 s, respectively), and still significantly outperform OpenCOLE (25.8 s). This efficiency primarily stems from our single-pass joint denoising process, which eliminates redundant layout–image alternation and avoids the sequential generation bottlenecks present in prior frameworks.

In contrast, OpenCOLE adopts a more complex threestage pipeline: a design plan generation module expands the input text; an image generation module synthesizes a background image; and a typography module predicts layout elements. While this decomposition improves controllability, it substantially increases inference latency because these components must be executed sequentially. Consequently, OpenCOLE requires 25.8 s per template, approximately 5× slower than our unified framework.

Overall, these results demonstrate that InterIL not only captures image–layout interactions effectively, but also scales efficiently for real-time or large-scale design generation scenarios.

TABLE S3  
Average inference time per design template on the Web-design test set. Our joint framework is substantially faster than Desigen and OpenCOLE, while the guided variants add only modest overhead.
<table><tr><td>Method</td><td>Time (s / template)</td></tr><tr><td>Desigen (0) Desigen (1)</td><td>7.47 14.85</td></tr><tr><td>Desigen (2)</td><td>22.42</td></tr><tr><td>Desigen (3)</td><td>29.89</td></tr><tr><td>OpenCOLE</td><td>25.80</td></tr><tr><td>Ours</td><td>5.01</td></tr><tr><td>Ours + OAG</td><td>8.69</td></tr><tr><td>Ours + RAG</td><td>8.47</td></tr><tr><td></td><td></td></tr><tr><td>Ours (Prompt Aug.)</td><td>10.14</td></tr></table>