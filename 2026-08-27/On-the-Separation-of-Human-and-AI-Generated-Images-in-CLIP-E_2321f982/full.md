# On the Separation of Human and AI-Generated Images in CLIP Embedding Space

Andrea Asperti<sup>1</sup>

<sup>1</sup>Department of Informatics - Science and Engineering (DISI), University of Bologna, Via Mura Anteo Zamboni 7, Bologna, 40126, Italy.

Contributing authors: andrea.asperti@unibo.it;

## Abstract

We identify a previously unreported phenomenon in CLIP representations: human and AI-generated paintings spontaneously separate along the dominant principal directions of their joint embedding distribution, without any supervised objective designed to distinguish the two classes. Rather than exploiting this phenomenon for detection, our objective is to interpret it: we seek to identify the visual information underlying the separation and to trace it back from the embedding space to the image domain.

We pursue this objective through a progressive investigation combining interpretable image representations with gradient-based inversion, used systematically as an experimental probe of the relationships identified in feature space. Robustness experiments and increasingly expressive statistical descriptors progressively rule out several intuitive explanations based on global image properties and simple local statistics, and point instead to distributed multiscale image structure. Multiscale scattering provides the most informative interpretable representation considered, but ofers only a partial account of the phenomenon.

Direct inversion provides a complementary and striking observation: substantial displacements along the dominant CLIP directions can be induced by image perturbations that remain nearly imperceptible to human observers, showing that the directions involved in the separation are highly sensitive to image variations with very low perceptual salience for humans. Taken together, these results reveal a significant diference between the visual evidence reflected in CLIP representations and that readily accessible to human perception, raising broader questions about the relationship between artificial and human vision and, ultimately, between artificial and human aesthetic judgment.

Keywords: CLIP, Vision-Language Models, Embedding Geometry, Interpretable Image Representations, Scattering Transform, Representation Inversion, Artificial Vision, AI-Generated Art

## 1 Introduction

Several recent studies on AI-generated art [1, 2] have shown that CLIP representations are highly efective at discriminating between human and AI-generated images, often outperforming non-expert human observers. Remarkably, this phenomenon is common to diferent image generators, diferent collections of human paintings, and diferent versions of CLIP.

The most intriguing aspect of this phenomenon is that the separation does not need to be induced by supervised learning, but appears to emerge naturally from the geometry of the embedded data. More precisely, when the CLIP embeddings of a mixed collection of human and AI-generated paintings are projected onto a low-dimensional space through Principal Component Analysis (PCA), the two classes exhibit a substantial degree of separation along the dominant principal directions. This separation emerges without any optimization objective specifically designed for discrimination: the distinction between human and AI-generated paintings is already present in the empirical distribution of the embeddings.

Figure 1 illustrates this phenomenon for images from the AI-Pastiche dataset [3] (red points) and paintings from the National Gallery of Art, Washington [4] (blue points). The CLIP model employed throughout these experiments is ViT-L/14@336px.

![](images/74e5bd24c9995cfdeec5a470f42d36a0a4ce92452e4b072a47146b4fe000192b.jpg)  
Fig. 1: Two-dimensional PCA projection of CLIP embeddings for AI-generated paintings (red) and human paintings (blue).

The reasons underlying this phenomenon remain largely unexplained. Rather than developing a more accurate detector, our objective is to understand why this unsupervised separation emerges and to identify the visual information responsible for it. In particular, we seek to characterize

• interpretable image properties,

• invariant statistical structures,

## • families of image transformations

that account for the dominant principal directions of the embedded dataset.

Our investigation follows a progressive strategy. We first establish that the observed separation is remarkably robust to a variety of transformations, including grayscale conversion, aggressive downsampling, cropping, and changes of image resolution. We then investigate increasingly expressive families of interpretable image representations. Elementary image statistics and local gradient descriptors explain only a limited fraction of the observed variability, allowing a number of intuitive hypotheses to be discarded. This progressive analysis ultimately leads to multiscale scattering representations, which provide the most informative statistical description of the phenomenon.

Empirically, the dominant principal directions are best explained by multiscale scattering features computed over image patches at diferent spatial resolutions, suggesting that the relevant information is encoded in distributed mesoscopic image statistics spanning multiple scales and locations rather than in isolated visual attributes.

Direct optimization experiments reveal an even more intriguing aspect of the phenomenon. By backpropagating through CLIP, it is possible to induce substantial displacements along the dominant principal directions through image perturbations that remain almost imperceptible to human observers (Section 3.2). This observation provides an independent validation of the proposed statistical interpretation while revealing that CLIP relies on visual evidence largely inaccessible to human perception.

Taken together, these findings provide a partial explanation for the intrinsic geometry of CLIP embedding space. More broadly, they provide quantitative evidence of a systematic mismatch between artificial and human visual perception. Although both systems operate on the same paintings, they appear to rely on fundamentally diferent visual evidence. Our results suggest that CLIP exploits stable mesoscopic statistical regularities that remain largely inaccessible to visual inspection, whereas human observers predominantly rely on semantic, stylistic, and compositional properties. This discrepancy is particularly significant in the artistic domain, where visual perception constitutes the primary medium of evaluation. If humans and artificial vision systems rely on diferent visual evidence, the assumption that they share common aesthetic criteria becomes dificult to sustain. In this sense, the distinction between human and AI-generated paintings provides a unique experimental setting for investigating the diferences between biological and artificial vision.

The main contributions of this work are the following:

• We identify and investigate a previously unexplained phenomenon: the spontaneous separation of human and AI-generated paintings along the dominant principal directions of CLIP embeddings.

• We develop a progressive analytical framework based on increasingly expressive families of interpretable image representations, systematically testing and ruling out several plausible explanations for the observed separation.

• We show that multiscale scattering statistics capture a significant, but still limited, fraction of the observed embedding geometry. This provides a partial explanation of the phenomenon and points to the relevance of distributed mesoscopic image structures, while leaving a substantial part of the separation unexplained.

• Through gradient-based inversion, we show that substantial displacements along the dominant embedding directions can be induced by image perturbations that remain almost imperceptible to human observers. This provides direct evidence of a mismatch between the visual information exploited by CLIP and that accessible to human perception, with potential implications for computational aesthetics and the understanding of artificial vision.

The remainder of the paper follows the progressive investigative strategy outlined above. Section 2 reviews the relevant background on CLIP representations, AI-generated image detection, interpretable image descriptors, and representation inversion. Section 3 introduces the experimental setting, including the datasets, CLIP models, and the inversion framework employed throughout the study. Section 4 presents the general methodology. The core of the paper is organized as a progressive search for an explanation of the observed separation. In Section 5, the robustness of the separation to a wide range of image transformations is investigated. Sections 6-8 progressively investigate increasingly expressive image representations, from elementary statistics and local gradient descriptors to multiscale scattering features. Section 9 validates the results of the previous Sections through gradient-based inversion experiments, revealing image perturbations that substantially alter the CLIP representation while remaining nearly imperceptible to human observers. Finally, Section 10 discusses the implications of these findings for the understanding of artificial vision and computational aesthetics, while Section 11 concludes the paper and outlines directions for future research.

## 2 Background and Related Work

The present work combines ideas originating from several research areas, including interpretable image representations, representation inversion, and CLIP-based image analysis. This section briefly reviews the concepts most directly related to our approach.

## 2.1 Understanding CLIP Representations

Contrastive Language–Image Pretraining (CLIP) [5] has emerged as one of the most influential vision-language models, providing a shared embedding space in which images and text are represented through contrastive learning. Owing to its remarkable transferability, CLIP has become a standard visual representation for a wide range of downstream tasks, including image retrieval, recognition, segmentation, and multimodal reasoning. Beyond its practical success, its embedding space has attracted increasing attention as an object of study in its own right.

Several studies have shown that CLIP embeddings exhibit a rich semantic organization, in which high-level visual concepts and semantic attributes are encoded in structured regions of the embedding space and can be efectively aligned with natural language descriptions [5–7]. Other works have investigated the internal organization of CLIP by analyzing the contribution of individual transformer layers, attention heads, and image patches to the final representation, revealing that semantic localization and concept-specific processing emerge naturally within the model [8]. More recently, representation decomposition techniques have been proposed to express CLIP representations in terms of human-interpretable concepts, providing insight into the semantic structure of individual neurons and embedding components [6, 9].

While these studies have substantially improved our understanding of the semantic organization of CLIP representations, they provide limited insight into why particular image collections naturally separate within the embedding space.

## 2.2 AI-Generated Image Detection

The rapid difusion of text-to-image models has stimulated extensive research on distinguishing AI-generated images from human-created ones. Early approaches relied on handcrafted forensic features or generator-specific artifacts [10, 11], whereas more recent methods increasingly exploit large pretrained vision models, particularly CLIP, as generic feature extractors [12, 13]. CLIP-based detectors have demonstrated remarkable robustness and generalization across generators, often requiring only lightweight classifiers trained on frozen embeddings [12].

Nevertheless, to the best of our knowledge, existing approaches formulate the problem exclusively as supervised classification. The objective is to optimize a decision boundary that discriminates between real and synthetic images while maximizing classification accuracy and generalization to unseen generators. Recent work has also investigated robustness, cross-generator generalization, and interpretation of frozen CLIP detectors, but still within the supervised detection paradigm [12, 14].

Rather than training a detector, we investigate the geometry of the embedded dataset itself, seeking to explain why the observed separation emerges before any supervised learning is performed.

## 2.3 Interpretable Image Representations

Long before the advent of deep learning, computer vision relied extensively on handcrafted image representations designed to capture well-defined geometric and statistical properties of natural images. These descriptors have been successfully employed in object recognition, texture analysis, image retrieval, and scene understanding, and remain valuable because of their mathematical interpretability and their ability to characterize specific visual phenomena. A comprehensive review of local image descriptors is provided by [15], while [16] provides an interesting survey of statistical approaches to image representation and analysis.

Among the most influential local descriptors are the Scale-Invariant Feature Transform (SIFT) [17] and Histograms of Oriented Gradients (HOG) [18], both of which characterize local image structure through gradient-based statistics while exhibiting robustness to moderate geometric and photometric variations. These representations have become standard tools for describing shape, edges, and local texture, providing features that are both computationally eficient and readily interpretable.

A complementary line of research models images through multiscale statistical representations derived from wavelet decompositions and natural image statistics. The seminal work of Portilla and Simoncelli [19] demonstrated that a large class of natural textures can be characterized through joint statistics of multiscale wavelet coeficients, while subsequent studies emphasized the broader role of natural image statistics in visual representation and perception [20]. Building upon these ideas, the Scattering Transform [21–23] provides a mathematically grounded representation obtained by cascading wavelet convolutions, modulus nonlinearities, and local averaging operations. Scattering coeficients are stable to small deformations while preserving rich multiscale information about image structure, and have demonstrated excellent performance in texture discrimination and image classification.

In the present work, HOG and Scattering are therefore employed not as alternatives to deep representations, but as analytical tools for interpreting the geometry of CLIP embeddings.

## 2.4 Representation Inversion and Feature Visualization

Understanding the visual information encoded by deep neural networks has motivated a broad family of visualization, attribution, and inversion techniques. Originally introduced to interpret learned representations and understand the hierarchical organization of deep features, inversion methods have recently experienced renewed interest owing to their central role in generative models, latent-space editing, and representation analysis.

Activation maximization methods synthesize inputs that strongly excite individual neurons or output classes, providing insight into the semantic concepts learned by deep representations [24, 25]. Complementary feature inversion techniques reconstruct images from intermediate representations, enabling the analysis of the information preserved at diferent stages of a neural network [26, 27]. More generally, optimizationbased inversion has become an important tool for probing latent representations, investigating model behavior, and understanding the geometry of learned feature spaces.

Closely related are adversarial optimization methods, which compute small image perturbations capable of producing large changes in a network’s internal representation or output [28, 29]. While these techniques are typically employed to study the robustness and vulnerabilities of deep models, they also provide a powerful means of exploring the geometry of learned representations.

In the present work, inversion is employed as an analytical probe of the empirical geometry of the CLIP embedding space, rather than as a visualization or adversarial technique.

## 2.5 Position of This Work

Existing studies have investigated the semantic organization of CLIP representations, developed increasingly robust methods for AI-generated image detection, proposed interpretable handcrafted image representations, and employed inversion techniques to analyze learned models. However, these lines of research have largely evolved independently. To the best of our knowledge, no previous work has investigated why human and AI-generated paintings naturally separate in the empirical distribution of CLIP embeddings without supervised learning, nor related this phenomenon to interpretable image statistics through a combination of robustness analysis, progressively more expressive image descriptors, regression, and inversion. The present work addresses this gap.

## 3 Experimental Setting

The phenomenon illustrated in Figure 1 is consistently observed across diferent datasets, image generators, and CLIP variants. To make an alternative example, in Figure 2 we show the 2D PCA projection of one thousand human and ai generated images borrowed from the AI-WikiArt [30] dataset.

![](images/5bf7cc386ccadb35c0552fde9f09855ffb1a9409e1c61f184531cae264055d92.jpg)  
Fig. 2: Two-dimensional PCA projection of CLIP embeddings for a random subset of AI-generated paintings (yellow) and human paintings (orange) borrowed from the AI-WikiArt dataset.

For the purposes of this article, we mostly worked with AI-WikiArt [30], AIPastiche [3], and paintings from the National Gallery of Art Dataset (NGAD) [4].

The AI-WikiArt dataset is a large-scale collection developed to evaluate visionlanguage models on tasks such as authorship attribution and AI-generated art detection [30]. It contains both authentic and synthetic artworks. The real subset includes 39,530 WikiArt paintings from 128 identified artists, covering 10 genres and 27 artistic styles, with metadata for each image. The synthetic subset was created with three text-to-image models: Stable Difusion, Flux, and F-Lite.

The AI-Pastiche dataset is a smaller dataset of generated paintings; however, it covers a much larger variety of generators, including very recent ones like Imagen4, FireflyImage4, Playgroundv2.5, Midjourney 7, Flux.2 and gpt-image-1.5. In its current version, AI-pastiche contains 1474 AI-generated paintings inspired by well-known artistic styles of the Western tradition, from the Renaissance onward. The dataset was constructed using 73 manually designed textual prompts, combined with 19 diferent image generators spanning multiple generations of difusion-based and generative models. The dataset also includes extensive metadata documenting the generation process, including prompts, models, and generation settings. Additionally, it provides human based annotations, relative to perceived authenticity, prompt adherence, and defects.

As an additional reference set of human artworks, we used the National Gallery of Art Dataset (NGA) [4], obtained from the publicly available collection of the National Gallery of Art, Washington. The dataset contains European paintings together with associated metadata, including title, attribution, artistic period, and links to highresolution images distributed through the IIIF protocol.

For our experiments, we mostly used AI-WikiArt for training, reserving AIPastiche and NGA for testing.

As for CLIP, the results presented in this article have mostly been obtained using the CLIP model ViT-L/14@336px, which rescales input images to a standard resolution of 336 × 336 pixels and produces embedding vectors of dimension 768. Additional experiments with diferent CLIP versions didn’t show any significant diferences.

## 3.1 2D PCA Projection

As already observed in the introduction, the striking phenomenon is not the possibil ity to train a supervised classifier to discriminate between Human and AI-generated artworks, but the fact that this distinction manifests itself in a completely natural way when considering the principal components of the combined dataset, as shown in Figure 1. In fact, more than explaining the separability between the two classes, we are even more interested in elucidating the variation factors that govern the movement along the two main components PC1 and PC2 of the 2D projection.

In explicit terms, the goal of our research can be further refined to:

• explain what PC1 and PC2 correspond to in image space;

• identify image modifications capable of reducing the observed separation.

In the rest of the article, we shall frequently project images in the 2D space defined by the original PC1 and PC2, to understand how diferent image transformations are reflected along these projections. An example will be given in the next section, where we invert PC1 through CLIP in real space and reproject the transformed image back into the 2D space, to observe its displacement.

Finally, it is worth noting that, before PCA projections, the two sets of pictures (AI and Human) are relatively close in CLIP’s embedding space, as conceptually exemplified in Figure 3

The previous observation is important for understanding that the components we are interested in are not general components of CLIP’s embedding space, but specific components of our manifold of data.

![](images/4fdd4e8cc2f0dc3429dc2c68014d4eb6809bb840008ff59cc52a514358e347f3.jpg)  
Fig. 3: Conceptualization of the relative position of the manifolds of human and AIgenerated paintings in CLIP’s embedding space.

## 3.2 Direct CLIP Inversion and Perceptual Dissociation

To better understand the geometric structure underlying the observed separation, we performed direct optimization experiments inside CLIP space. Starting from an AIgenerated image $x _ { 0 }$ , we searched for a perturbation capable of moving the image along selected principal directions of the CLIP embedding space while preserving visual similarity from the point of view of a human observer.

More precisely, let

$$
f ( x ) \in \mathbb { R } ^ { 7 6 8 }
$$

denote the normalized CLIP embedding of an image x, and let

$$
v _ { 1 } , v _ { 2 }
$$

be the first principal directions obtained from PCA over the full embedding dataset. The corresponding PCA coordinates are computed as

$$
p _ { i } ( x ) = \langle f ( x ) - \mu , v _ { i } \rangle ,
$$

where $\mu$ denotes the empirical mean embedding.

The optimization problem consists in finding a perturbed image

$$
x = x _ { 0 } + \delta
$$

that maximizes or minimizes one of the principal coordinates while remaining visually close to the original image.

<table><tr><td>image id</td><td>x init</td><td>y init</td><td>x final</td><td>y final</td><td>distance</td><td>max perturb.</td></tr><tr><td>354</td><td>-0.36</td><td>-0.20</td><td>-0.15</td><td>-0.18</td><td>0.21</td><td>0.049</td></tr><tr><td>336</td><td>0.03</td><td>-0.35</td><td>0.43</td><td>-0.08</td><td>0.47</td><td>0.044</td></tr><tr><td>758</td><td>-0.45</td><td>-0.10</td><td>-0.14</td><td>-0.13</td><td>0.30</td><td>0.046</td></tr><tr><td>729</td><td>-0.11</td><td>-0.41</td><td>0.28</td><td>-0.23</td><td>0.43</td><td>0.049</td></tr></table>

Table 1: Movements in the 2D PCA space induced by the gradient ascent along the main component. We provide initial and final coordinates, and euclidean distance. The perturbation is measured as max absolute diference on images, normalized in the [0,1] interval.

The perturbation is parameterized as

$$
\delta = \varepsilon \operatorname { t a n h } ( u ) ,
$$

where u is the optimization variable and ε controls the maximum perturbation amplitude. This parameterization ensures bounded perturbations while maintaining smooth gradients during optimization.

The optimization objective takes the form

$$
\begin{array} { r } { \mathcal { L } ( x ) = \pm p _ { i } ( x ) + \lambda \mathrm { T V } ( \delta ) , } \end{array}
$$

where TV denotes a Total Variation regularization term and λ controls the strength of the smoothness constraint.

The Total Variation penalty is introduced to discourage high-frequency perturbations and isolated pixel-level noise. Rather than allowing arbitrary unconstrained modifications, the regularization biases the optimization toward spatially coherent transformations distributed across the image.

Optimization is performed directly through backpropagation on the input image using the CLIP encoder as a diferentiable mapping. The procedure is conceptually related to the generation of adversarial perturbations in deep neural networks [28]. However, the objective here is not to induce arbitrary classification errors, but rather to navigate the intrinsic geometric organization already present in the embedding space. Specifically, we are moving along the main component of the data manifold.

Remarkably, substantial movements inside CLIP space can be obtained while the modified image remains visually almost indistinguishable from the original one, as exemplified in Figure 4.

In particular, perturbation amplitudes remain numerically small (below 0.05 on the unitarian scale, see Table 1) and visually coherent, while the corresponding displacement along the principal CLIP directions can become very significant, as shown in Figure 5.

Because gradient-based inversion through CLIP is computationally expensive, we limited the experiments to a representative subset of images. Nevertheless, all optimization experiments produced qualitatively similar results.

This phenomenon produces a genuine perceptual dissociation:

![](images/82a35d7ed823bd4fcb2832202a46af7352519cbf2ddc64da5c52bc4be58e7862.jpg)

![](images/5e7e77c1d3d20270b5695e66b6092b27f9d614520151d41d0cfb5eddf42e3790.jpg)

![](images/57f4644a3452329246fd506df52e6acfc0a72178375d507a143af992b1fe998e.jpg)

![](images/02fe7f8dca3b1160b9f4d3a5c3edfa2c8a4441388b4d113f74896c5d4ee4e5c5.jpg)

![](images/dfb48c6271a79eb6da12599da691934f394bb31bead1812bb5a7a1a10766cdcf.jpg)

original  
![](images/2dcf0eb71117b021e848072dd26ebaafa160129b763d638f79f019075f218c0e.jpg)  
modified  
diference  
Fig. 4: In the left column the original image, in the middle column the modified version, and in the right column the diference between them.

• from the human point of view, the optimized image remains essentially unchanged;

• from the point of view of CLIP, the image undergoes a substantial transformation.

The importance of this result is not merely the existence of adversarial sensitivity, which is already well known in deep neural networks [28], but rather the fact that these perturbations operate along stable geometric directions that emerge spontaneously at the dataset level.

![](images/aa746ad4b738f48e4a2a2709c5d2b29b138f2ad092cc4af706b2a1dd76ce2ca4.jpg)  
Fig. 5: Movements in the 2D latent space induced by gradient ascent through CLIP along the main PCA component. Opposite movements for reproductions of human paintings can be easily generated too.

This suggests that the CLIP embedding space organizes visual similarity around structures that are only partially aligned with human perceptual salience. At the same time, an alternative interpretation remains possible: generative models themselves may implicitly introduce subtle statistical regularities that become highly coherent within the representation space while remaining nearly invisible to human observers.

The remainder of this work is devoted to investigating the nature of these hidden structures and identifying which statistical or perceptual properties may explain the observed separation.

## 4 Methodological framework

Our investigation is structured into the following stages:

• Robustness analysis, aimed at evaluating the stability of the AI-human separation under a wide range of image transformations.

• Basic statistical descriptors, investigating whether elementary image statistics related to color, luminance, and texture are suficient to explain the dominant principal directions.

• Advanced image descriptors, extending the analysis to progressively richer local and multiscale representations, with particular emphasis on Histograms of Oriented Gradients (HOG) and Scattering Transform features, while also investigating the role of their spatial organization.

• Regression Model Inversion, validating the proposed statistical explanations by identifying image perturbations that induce controlled displacements within the PCA representation.

The four stages should not be regarded as independent experiments but as successive steps of the same investigation. Each stage is motivated by the limitations of the previous one and progressively narrows the range of plausible explanations for the observed separation. The following subsections describe each stage in greater detail.

## 4.1 Robustness Analysis

Before attempting to characterize the visual properties responsible for the separation between human and AI-generated paintings, we performed a set of preliminary experiments aimed ensuring that the phenomenon is genuine and not an artifact of a particular dataset, preprocessing pipeline, or model configuration.

To this end, we performed a series of robustness experiments designed to assess the stability of the AI-human discrimination direction under a variety of transformations and experimental conditions.

The analysis considered file formats and preprocessing procedures, grayscale conversion and color manipulations, image resolution, and cropping strategies. Tests were performed on diferent AI generators, various human art collections, and multiple CLIP variants. Together, these experiments provide a systematic assessment of the stability of the observed separation and establish the empirical foundations for the subsequent analyzes.

## 4.2 Basic Statistical Descriptors

As a first step, we investigated whether the separation between human and AIgenerated paintings could be explained by simple global image statistics. Such descriptors are attractive because they are easily interpretable and can often reveal systematic diferences in color distribution, luminance, texture, or frequency content.

To this end, we considered several families of global descriptors, including pixel and luminance histograms, color statistics, radial frequency power spectra, and simple texture measures. We also examined correlations between these quantities and the principal directions of variation in the CLIP embedding space. The objective was to determine whether the AI-human separation could be attributed to a small number of low-level statistical properties before moving to more structured and spatially localized image representations.

## 4.3 Advanced Image descriptors

The limited explanatory power of simple image statistics, in combination with the robustness of the AI-Human separation to major image perturbations like cropping, blurriness, or grayscale conversion, suggests that the signal could be better captured by more sophisticated statistics over larger image regions.

Motivated by the patch-based architecture of Vision Transformers, we investigated several families of patch-based representations.

Histogram of Oriented Gradients (HOG) [18] describes a patch through histograms of local gradient orientations and provides a compact representation of shape and texture.

Scattering coeficients [21] provide a parametric family of multi-scale and multiorientation representations based on cascades of wavelet convolutions followed by modulus and averaging operations. Given a family of oriented wavelets, the parameter J controls the maximum spatial scale represented in the decomposition, while a diferent parameter L determines the number of sampled orientations. First-order coeficients capture the distribution of image energy across scales and orientations, whereas second-order coeficients additionally encode interactions between diferent scales and orientations. Unless otherwise specified, we employed second-order scattering coeficients and explored diferent values of J and L to assess the role of scale and angular resolution in the resulting representations.

In addition to HOG and Scattering, we also conducted experiments with Radial Frequency Power descriptors, which summarize the distribution of spectral energy across spatial frequencies by averaging the Fourier power spectrum over concentric frequency bands and Orientation Transport descriptors, which characterize the spatial organization and coherence of local orientations within the patch. However, these techniques didn’t provide additional insight into the problem, and they will not be discussed.

For HOG and Scattering, we devoted particular attention to determining whether the discriminative information arises primarily from the local image statistics themselves or from their spatial arrangement within the artwork. To investigate this question, we systematically evaluate several feature aggregation strategies for both descriptor families. These include patch-based feature extraction, which preserves local spatial information, as well as global aggregation methods based on both linear and nonlinear averaging (MLP, attention, single layer Transformers).

To further evaluate the role of spatial organization, we also performed patchshufling experiments in which patch descriptors were randomly permuted before being processed by the predictive model. Since shufling preserves the distribution of local descriptors while destroying their spatial arrangement, the resulting performance provides a direct measure of the contribution of spatial structure to the discrimination task.

Together, these experiments allow us to distinguish between information encoded in the local statistics themselves and information arising from their organization across the image.

## 4.4 Regression Model Inversion

While the regression models provide a quantitative measure of how accurately the selected image descriptors predict the principal component coordinates, prediction accuracy alone is not suficient to assess the quality or interpretability of a given descriptor. Empirically, many feature representations achieved excellent regression per formance, yet failed to produce meaningful inversion results. In other words, a low regression error does not necessarily imply that the learned mapping captures image characteristics that can be manipulated to produce a coherent displacement in the latent space. For this reason, regression performance should always be interpreted in conjunction with the corresponding inversion experiments, even though the two analyzes are presented separately in the following sections.

The main objective of the inversion process is to identify image modifications that induce a controlled displacement of an artwork within the two-dimensional PCA space. Starting from a given image, we seek perturbations that maximize or minimize the predicted PCA coordinates while preserving the perceptual appearance of the image. The inversion stage, therefore, constitutes the ultimate validation of a descriptor: a successful inversion demonstrates that the learned relationship between the descriptors and the PCA embedding is not only predictive but also actionable, enabling controlled navigation of the latent space through physically meaningful image modifications.

Throughout the optimization, perturbations are constrained to remain visually negligible, allowing the inversion process to reveal subtle statistical cues without altering the semantic content of the artwork.

Beyond visualization, inversion serves two complementary purposes. First, it discriminates between descriptors that merely correlate with the PCA coordinates and those that genuinely encode the visual information responsible for the observed separation. Only the latter produce stable and coherent image modifications capable of moving an image toward the region occupied by AI-generated or human-created artworks. Second, inversion provides an interpretable visualization of the image attributes that most strongly influence the learned representation, ofering insights that cannot be obtained from regression accuracy alone.

## 5 Robustness Experiments

## 5.1 File format and preprocessing invariance

To assess whether the observed separation could be attributed to diferences in file formats or preprocessing pipelines, we conducted a controlled re-encoding experiment. Each image was loaded, converted to a standard RGB representation, saved to disk in PNG format, and reloaded before being passed through the CLIP preprocessing pipeline. We then compared the resulting embeddings with those obtained from the original images using cosine similarity.

Across both human and AI-generated images, the cosine similarity between original and reprocessed embeddings was consistently equal to 1 (up to numerical precision). This indicates that the CLIP representation is efectively invariant to this re-encoding procedure, and that file format, compression artifacts, or metadata diferences introduced at this stage do not contribute to the observed separability.

This experiment allows us to rule out a broad class of potential confounders, including diferences in file format, compression schemes, metadata, and color encoding pipelines. Specifically, it suggests that the separation between human and AI-generated images is not driven by superficial artifacts introduced during image storage or loading.

## 5.2 Color to Grayscale degradation

Overall, as also testified by the statistics in Section 6.2, color appears to play a more limited role in CLIP’s visual representation than one might initially expect. To investigate this hypothesis, we compared the embeddings of the original images with those of their grayscale counterparts.

The average cosine similarity between the embedding of an image and that of its grayscale version is approximately 0.90±0.04 for AI-Pastiche and 0.89±0.04 for NGA. Thus, grayscale conversion produces only a relatively small displacement in CLIP’s embedding space.

The distance becomes even smaller when projected into the 2D space. Figure 6 shows the projection of grayscale images onto the original PCA space defined by PC1 and PC2.

![](images/4429776e7b3b8363ef018d0c0443fe57f84d6ec7597e0061d8def1f7ff0e1771.jpg)  
Fig. 6: Projection of grayscale AI-Pastiche (dark gray) and NGA (light gray) images onto the PCA space computed from the original CLIP embeddings.

The two classes become slightly more entangled after grayscale conversion, suggesting that color contributes to the separation captured by PC1 and PC2. However, the efect remains limited, and most of the original structure is preserved. This indicates that the dominant diferences between the two datasets are largely independent of color information.

## 5.3 Resolution Robustness

We next investigated the robustness of CLIP’s representation to resolution degradation. The standard input resolution for the version of CLIP used in our experiments is 336 × 336 pixels. To evaluate the efect of image resolution, we first downsampled each image to a lower resolution and then resized it back to the standard input size before computing its embedding. We also performed analogous experiments using progressive Gaussian blurring and obtained qualitatively similar results.

Table 2 reports the cosine similarity between the original embedding and the embedding obtained after the downsampling-upsampling process.

The embedding is only moderately afected by downsampling to 168 × 168 or 112×112 pixels. More severe degradation produces a larger displacement in the embedding space, although the images remain substantially correlated with their original representations.

The efect of resolution degradation can be visualized more clearly through the PCA projections shown in Figures 7, 8, and 9.

<table><tr><td></td><td> $1 6 8 \times 1 6 8$ </td><td> $1 1 2 \times 1 1 2$ </td><td> $4 8 \times 4 8$ </td><td> $2 4 \times 2 4$ </td></tr><tr><td>AI-Pastiche</td><td> $\overline { { 0 . 9 3 \pm 0 . 0 2 } }$ </td><td> $\overline { { 0 . 8 6 \pm 0 . 0 3 } }$ </td><td> $\overline { { 0 . 7 2 \pm 0 . 0 6 } }$ </td><td> $\overline { { 0 . 6 3 \pm 0 . 0 7 } }$ </td></tr><tr><td>NGA</td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $0 . 8 7 \pm 0 . 0 3$ </td><td> $0 . 6 7 \pm 0 . 0 6$ </td><td> $0 . 5 6 \pm 0 . 0 8$ </td></tr></table>

Table 2: Cosine similarity between the original CLIP embeddings and the embeddings obtained after resolution degrada tion.

![](images/c11297a3099dc444f5137ab118c35f227369d9b353d96ef6e5653bc65b9c8c27.jpg)  
Fig. 7: Projection of images degraded to $1 1 2 \times 1 1 2$ pixels onto the original PCA space. Although the embeddings are displaced, the separation between AI-generated and human paintings remains largely preserved.

Overall, the separation between AI-generated and human paintings proves remarkably robust to substantial reductions in image resolution. This result suggests that the dominant CLIP components identified in our analysis are not driven by highfrequency details or fine-scale texture. Instead, they appear to reflect visual properties that remain stable across a broad range of spatial scales.

## 5.4 Crop Robustness

To distinguish between local and global sources of information, we investigated the efect of random image crops.

Table 3 reports the performance of a logistic regression classifier trained to distinguish AI-generated from human paintings using CLIP embeddings computed from random crops of diferent sizes. Crop size is expressed as a percentage of the shortest image dimension. Consequently, a crop of 10% corresponds to less than 1% of the total image area.

![](images/2e26c4013fc7a90e112759421d56917148b4b00d2e516ba18acef4f7da779b46.jpg)  
Fig. 8: Projection of images degraded to 48 × 48 pixels. The two distributions become more entangled, although a substantial fraction of AI-generated images remains separated from the human paintings.

![](images/379cd6383d6c22a4bb30224f7b372f905335cb401f7a4ffeb3a2d65d13b4c8c1.jpg)  
Fig. 9: Projection of images degraded to 24 × 24 pixels. At this resolution the two datasets become substantially entangled along the original PCA directions.

The results reveal a sharp transition as a function of crop size. Very small crops (10%) contain essentially no discriminative information, whereas crops covering approximately 40% of the shortest image dimension recover almost the full classification performance obtained using complete images.

Interestingly, crops of size 40% remain strongly correlated with the embeddings of the corresponding full images. The average cosine similarity between crop and fullimage embeddings is approximately 0.82 ± 0.07 for AI-Pastiche and $0 . 7 7 \pm 0 . 0 8$ for NGA.

<table><tr><td>Crop size</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>10%</td><td>0.516</td><td>0.516</td><td>0.515</td><td>0.515</td></tr><tr><td>20%</td><td>0.763</td><td>0.767</td><td>0.763</td><td>0.762</td></tr><tr><td>40%</td><td>0.943</td><td>0.946</td><td>0.945</td><td>0.945</td></tr></table>

Table 3: AI vs. human classification performance using CLIP embeddings computed from random image crops. Classification is performed directly in the embedding space, before PCA projection.

Figure 10 shows the projection of cropped images onto the original PCA space.  
![](images/87256212883147e385e5f1f8b8676ff86e6175be56e286f8844e5c8c0dc6ed9f.jpg)  
Fig. 10: Projection of random image crops onto the PCA space defined by the original CLIP embeddings. Although crops introduce additional variability, the separation between AI-generated and human paintings remains largely preserved for suficiently large crop sizes.

These results allow us to rule out several simple explanations for the observed separation:

• Fine-scale texture cues confined to small image regions.

• Border or framing efects.

• Purely global semantic or compositional structure, since crops covering only a fraction of the image remain highly discriminative.

Overall, the experiment suggests that the dominant discriminative signal is neither purely local nor purely global. Instead, it appears to emerge at an intermediate, mesoscopic scale, requiring suficiently large image regions while remaining largely independent of the exact global image layout.

## 5.5 Generator Robustness

The AI-Pastiche dataset covers a large collection of generative models belonging to diferent generations; hence, the separation problem does not seem to be reducible to a particular model or class. In Figure 11 we show the disposition of the AI-generated images for the diferent models in the first release of AI-Pastiche.

![](images/dccd591f1074513e41b6a12b5b90f5792c4eb76a56669deceda687b3916c3266.jpg)  
Fig. 11: Distribution of AI-artworks for the diferent generative models of AI-Pastiche.

The distinction also persists in the most recent generators. In Figure 11 we plot the disposition of a very recent extension to the AI-Pastiche dataset, comprising Imagen4, FireflyImage4, Playgroundv2.5, Midjourney 7, Flux.2 and gpt-image-1.5.

Taken together, the robustness experiments indicate that the observed separation is remarkably stable across datasets, generators, preprocessing strategies, and image transformations. Consequently, the phenomenon is unlikely to be explained by simple image artifacts or isolated visual cues, motivating the search for increasingly expressive statistical descriptors in the following sections.

## 6 Global Statistical Descriptors

Our investigation begins with the simplest explanatory hypothesis: that the separation between human and AI-generated paintings can be accounted for by elementary global image statistics. These descriptors are attractive because they are directly interpretable and have long been used to characterize color, luminance, texture, and frequency content. Their analysis also provides an important baseline, allowing us to determine whether the observed geometry of the CLIP embedding distribution can be explained by low-level statistical properties before introducing progressively richer image representations.

![](images/1e360a07fbf33e7b931ea3ccb1b6797b4f04ad10a2277a97af2bb409a53d1415.jpg)  
Fig. 12: Distribution of AI-craftworks for generative models of the latest generation.

## 6.1 Basic statistics

Table 4 reports the basic statistics for the two datasets along the three RGB components. Specifically, we give the min, max, mean, and (average) intra image standard deviation.

<table><tr><td>AI-Pastiche</td><td>min</td><td>max</td><td>mean</td><td>std</td></tr><tr><td>red</td><td> $5 . 7 \pm 9 . 3$ </td><td> $2 4 9 . 5 \pm 9 . 2$ </td><td> $1 1 6 . 7 \pm 6 3 . 3$ </td><td> $6 3 . 4 \pm 1 3 . 2$ </td></tr><tr><td>green</td><td> $6 . 6 \pm 9 . 9$ </td><td> $2 4 2 . 1 \pm 1 2 . 9$ </td><td> $1 0 5 . 7 \pm 8 8 . 2$ </td><td> $5 7 . 4 \pm 1 4 . 3$ </td></tr><tr><td>blue</td><td> $4 . 0 \pm 8 . 1$ </td><td> $2 2 9 . 5 \pm 2 1 . 7$ </td><td> $1 1 6 . 7 \pm 6 3 . 4$ </td><td> $5 4 . 4 \pm 1 4 . 3$ </td></tr><tr><td>NGA</td><td> $\operatorname { m i n }$ </td><td>max</td><td>mean</td><td> $\overline { { \mathrm { s t d } } }$ </td></tr><tr><td>red</td><td> $2 0 . 2 \pm 2 6 . 5$ </td><td> $2 3 9 . 8 \pm 1 8 . 3$ </td><td> $1 1 8 . 6 \pm 4 3 . 1$ </td><td> $4 8 . 0 \pm 1 5 . 6$ </td></tr><tr><td>green</td><td> $2 0 . 9 \pm 2 2 . 7$ </td><td> $2 2 4 . 0 \pm 2 6 . 7$ </td><td> $1 0 4 . 1 \pm 4 2 . 8$ </td><td> $4 2 . 9 \pm 1 4 . 6$ </td></tr><tr><td>blue</td><td> $1 3 . 2 \pm 2 0 . 6$ </td><td> $2 0 0 . 7 \pm 3 7 . 0$ </td><td> $8 2 . 2 \pm 4 1 . 7$ </td><td> $3 9 . 7 \pm 1 6 . 8$ </td></tr></table>

Table 4: Basic statistics about AI-Pastiche and NGA data.

Several diferences are statistically apparent, particularly in the blue channel and in the intra-image variability. However, these diferences are relatively modest compared with the magnitude of the separation observed in CLIP space and do not provide a convincing explanation for the dominant principal directions.

The following table reports the Shannon entropy of the empirical distribution of pixel intensities. For each image, all pixel values from the three color channels are pooled together and regarded as samples from a discrete distribution over the 256 possible intensity levels.

Table 5: Entropy of the empirical distribution of pixel intensities for AI-pastiche and NGA.

In Figure 13 we visualize the histogram of the distribution of pixel intensities for the two datasets.

![](images/0795fce706342f66393c5d7ecde3c7ad4ada2615dfdf415c9de56b6584e2d871.jpg)  
(a)

![](images/774ac7be2d7ccf4bb139b8ec677feaaaa593a176f635e212cf239b31e80fbf2f.jpg)  
(b)  
Fig. 13: Histograms of distribution of pixel intensities for AI-pastiche and NGA.

To assess whether the observed diferences are primarily driven by global intensity statistics, we also evaluated several normalization procedures. None of them produced a substantial modification of the embedding geometry.

## 6.2 Color Palette

The color palettes of the two datasets are broadly similar, with a predominance of brown and gray tones (Figure 14).

![](images/506fb9e46cde1740a57fddf7713dcb8dc25c11d45b1e7dfade1de192a8b5bd56.jpg)  
Fig. 14: Empirical color palette distribution for the AI-Pastiche and NGA datasets.

<table><tr><td></td><td>red</td><td>orange</td><td>yellow</td><td>lime</td><td>green</td><td>cyan</td><td>blue</td><td>violet</td><td>magenta</td><td>a pink</td><td>brown</td><td>gray</td></tr><tr><td>AI</td><td>.002</td><td>.016</td><td>.002</td><td>.001</td><td>.000</td><td>.001</td><td>.002</td><td>.000</td><td>.000</td><td>.126</td><td>.424</td><td>.423</td></tr><tr><td>NGA</td><td>.001</td><td>.012</td><td>.002</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.113</td><td>.472</td><td>.400</td></tr></table>

Table 6: Relative frequency of dominant color categories in the two datasets.

Overall, the color distributions of the two datasets are remarkably similar, exhibiting only minor quantitative diferences. This observation is consistent with the robustness experiments of Section 5, where the separation remained largely unchanged after grayscale conversion. Taken together, these results suggest that global color composition is unlikely to constitute the primary statistical cue underlying the observed geometry of the CLIP embedding space.

## 6.3 Local structure: luminance and texture

The first descriptors exhibiting a non-negligible correlation with the principal components are those related to luminance variability and local contrast.

<table><tr><td>Metric</td><td>AI-Pastiche</td><td>Human</td><td>AI-Pastiche</td><td>Human</td><td>AI-Pastiche</td><td>Human</td></tr><tr><td></td><td colspan="2">336 × 336</td><td colspan="2">168 × 168</td><td colspan="2">84 × 84</td></tr><tr><td>Chroma</td><td> $1 9 . 7 \pm 9 . 8$ </td><td> $1 9 . 1 \pm 8 . 6$ </td><td> $1 9 . 4 \pm 9 . 6$ </td><td> $1 9 . 0 \pm 8 . 6$ </td><td> $1 9 . 4 3 \pm 9 . 5 8$ </td><td> $1 8 . 8 7 \pm 8 . 5 8$ </td></tr><tr><td>Saturation HSL</td><td> $0 . 3 6 \pm 0 . 1 5$ </td><td> $0 . 3 2 \pm 0 . 1 3$ </td><td> $0 . 3 4 \pm 0 . 1 4$ </td><td> $0 . 3 2 \pm 0 . 1 3$ </td><td> $0 . 3 4 \pm 0 . 1 4$ </td><td> $0 . 3 1 \pm 0 . 1 3$ </td></tr><tr><td>Luminance mean</td><td> $4 4 . 7 \pm 1 4 . 7$ </td><td> $4 4 . 2 \pm 1 6 . 8$ </td><td> $4 4 . 7 \pm 1 4 . 7$ </td><td> $4 4 . 2 \pm 1 6 . 8$ </td><td> $4 4 . 7 \pm 1 4 . 7$ </td><td> $4 4 . 2 \pm 1 6 . 8$ </td></tr><tr><td>Luminance std</td><td> $2 2 . 6 \pm 4 . 3$ </td><td> $1 7 . 2 \pm 5 . 6$ </td><td> $2 2 . 0 \pm 4 . 3$ </td><td> $1 6 . 8 \pm 5 . 6$ </td><td> $2 1 . 3 \pm 4 . 2$ </td><td> $1 6 . 3 \pm 5 . 6$ </td></tr><tr><td>local L contrast</td><td> $6 . 8 \pm 2 . 2$ </td><td> $4 . 6 \pm 1 . 8$ </td><td> $9 . 8 \pm 2 . 7$ </td><td> $6 . 9 \pm 2 . 4$ </td><td> $9 . 8 4 \pm 2 . 7$ </td><td> $6 . 8 \pm 2 . 4$ </td></tr><tr><td>edge L contrast</td><td> $1 0 . 8 \pm { 3 . 8 }$ </td><td> $7 . 1 \pm 2 . 9$ </td><td> $1 5 . 7 \pm 4 . 9$ </td><td> $1 0 . 7 \pm 4 . 0$ </td><td> $1 5 . 7 \pm 4 . 9$ </td><td> $1 0 . 7 \pm 4 . 0$ </td></tr></table>

Table 7: Basic luminance and color statistics at diferent scales.

Table 15a gives the correlation between the previous statistics with respect to the PC1 and PC2. A major problem is that all statistics are strongly correlated between each other (between 0.86 and 1), as reported in Figure 15b. So, there is little to be gained from considering a linear combination of them.

These observations suggest that simple global statistics and elementary measures of local contrast capture only a limited aspect of the phenomenon. This motivates the transition to descriptors capable of representing richer spatial organization, beginning with local gradient statistics in the following section.

## 7 Histogram of Oriented Gradients

Histogram of Oriented Gradients (HOG) descriptors [18] characterize local image structure by accumulating gradient orientations into histograms computed over small spatial cells. In our implementation, each $2 4 \times 2 4$ patch was subdivided into a $2 \times 2$ grid of cells, each represented by a histogram of 9 orientation bins. Thus, every patch was encoded by ${ \mathrm { ~ a ~ 2 ~ } } \times { \mathrm { ~ 2 ~ } } \times { \mathrm { ~ 9 ~ } } = { \mathrm { ~ 3 6 ~ } }$ dimensional descriptor. Since a $3 3 6 \times 3 3 6$ image contains a regular grid of $2 3 \times 2 3$ overlapping patches (stride 14), the complete HOG representation forms a tensor of size $2 3 \times 2 3 \times 2 \times 2 \times 9$ . Diferent aggregation strategies were subsequently investigated, ranging from global spatial averaging to preserving the complete descriptor field, as described in the next subsection.

<table><tr><td>metric</td><td> $R ^ { 2 } P C 1$ </td><td> $R ^ { 2 } P C 2$ </td></tr><tr><td>saturation</td><td>0.13</td><td>-0.15</td></tr><tr><td>chroma</td><td>0.12</td><td>0.02</td></tr><tr><td>entropy</td><td>0.31</td><td>-0.05</td></tr><tr><td>luminance mean</td><td>0.12</td><td>0.23</td></tr><tr><td>luminance std</td><td>0.35</td><td>-0.26</td></tr><tr><td>local contrast</td><td>0.33</td><td>-0.23</td></tr><tr><td>edge contrast</td><td>0.36</td><td>-020</td></tr></table>

![](images/26f8e6685a5fa41f3d712716282094320af2a46838b01d0dd4f053b3404ae645.jpg)  
(a) $R ^ { 2 }$ Correlation between local structure (b) Self correlation between local strucstatistics versus PC1 and PC2. ture statistics

In Table 8 we report the $R ^ { 2 }$ similarity of diferent aggregations to PC1 and PC2.
<table><tr><td>feature</td><td>size</td><td>PC1  $R ^ { 2 }$ </td><td>PC2  $R ^ { 2 }$ </td></tr><tr><td>HOG field</td><td> $2 3 \times 2 3 \times 2 \times 2 \times 9$ </td><td>0.30</td><td>0.32</td></tr><tr><td>field of HOG av.</td><td> $2 3 \times 2 3 \times 2 \times 2$ </td><td>0.24</td><td>0.25</td></tr><tr><td>field of HOG av.</td><td> $2 3 \times 2 3$ </td><td>0.10</td><td>0.11</td></tr><tr><td>HOG spatial av.</td><td>9</td><td>0.03</td><td>0.02</td></tr><tr><td>HOG spatial av.</td><td> $2 \times 2 \times 9$ </td><td>0.06</td><td>0.05</td></tr></table>

Table 8: $R ^ { 2 }$ similarity of diferent aggregations of 24 × 24 patch based hog features to PC1 and PC2.

The full field provides the best results, while the signal is nearly erased by spatial averaging.

In Figure 16 we show the field of mean HOG values (a), max HOG values (b) and prevalent orientations(c). The first two maps are relatively uniform (the range of values is very limited), while the third one is quite chaotic.

A few observations can be drawn from the analysis. First, the discriminative signal does not exhibit sparse localization, indicating that it is not concentrated in a few isolated regions of the image. Likewise, there are no evident ”artifact zones” that could be associated with localized generation defects. Instead, the observed signal has a relatively low amplitude but remains spatially coherent across the image. Overall, the resulting maps are smooth and broadly distributed, supporting the hypothesis that the distinction between AI-generated and human-created artworks is encoded in subtle, spatially organized statistical patterns rather than in isolated visual artifacts.

![](images/722c25498267578c2beeee48689031c958866b604803837c022db01c5ded79ef.jpg)  
(a) HOG mean

![](images/ae242f2379dc8930064831339cbb208a18b610cb45494ce91a2dccfb2d695b76.jpg)  
(b) HOG max

![](images/a6ddc7b4c81a2f8bb04f01b17fdf2e1d2173a677ee1481b3622c73f70ecc6ded.jpg)  
(c) HOG orientations  
Fig. 16: Visualization of HOG statistics.

## 7.1 Inter-patch investigations

In the previous experiment, each patch provided an independent linear contribution to the result. It is natural to wonder if a better result can be obtained by comparing patches together, deriving non-linear dependencies.

As a preliminary test in this direction, we investigated inter-patch features via non-linear methods, specifically a simple two layer MLP. As input features, we used the full HOG field. Due to the number of features, strong regularization was required. We obtained the best results with $\alpha = 2 0$ (the default is alpha=0.0001). All tests were obtained through Kfold cross validation with 5 splits. Results are provided in Table 9. In addition to MPL, we also tested Ridge over quadratic features (projected into 100 PCA components) but did not obtain significant results.

<table><tr><td>model</td><td>features</td><td>PC1 R²</td><td>PC2 R2</td></tr><tr><td>MLP</td><td> $2 3 \times 2 3 \times 2 \times 2 \times 9$ </td><td>0.39</td><td>0.41</td></tr></table>

Table 9: $R ^ { 2 }$ similarity over a field of HOG features processed by a shallow MLP.

The previous result seems to confirm that spatial arrangement matters. At the same time. MLP only modestly improves over Ridge, so the signal is not dominated by strong nonlinear interactions.

Overall, the results suggest that the distinction between AI-generated and humancreated artworks is primarily encoded in weak but spatially distributed gradient statistics. Although the discriminative signal is subtle, it exhibits a consistent spatial organization rather than being confined to isolated image regions. Furthermore, the experimental results indicate that most of the relevant information can be efectively captured through linear aggregation of local descriptors, while more sophisticated nonlinear aggregation strategies provide only marginal improvements. This suggests that the underlying discriminative patterns are largely additive in nature, with nonlinear processing ofering only limited refinement beyond the information already contained in the aggregated local statistics.

Next, we try to understand at what spatial scale the spatial arrangement becomes meaningful. We do it by shufling HOG cells within local neighborhoods at progressive scales.

<table><tr><td>block size</td><td>1</td><td>2 × 2</td><td>4× 4</td><td>8× 8</td><td>23</td></tr><tr><td>PC1</td><td>0.39</td><td>0.36</td><td>0.30</td><td>0.27</td><td>0.04</td></tr><tr><td>PC2</td><td>0.41</td><td>0.37</td><td>0.29</td><td>0.22</td><td>0.16</td></tr></table>

Table 10: HOG field shufling, processed through MLP. The block size is the area being shufled.

Table 10 shows that preserving the spatial organization of local HOG descriptors is essential for accurately predicting the PCA coordinates. While shufling small regions only moderately afects performance, progressively larger shufling blocks lead to a substantial reduction in the correlation with the principal components. These results indicate that preserving the spatial organization of local image statistics is essential for explaining the dominant embedding directions.

## 7.2 Local Correlation descriptors

The previous experiments suggested that the predictive signal captured by HOG descriptors is spatially distributed and cannot be reduced to isolated local features. This naturally led to the hypothesis that the relevant information might be encoded not in the marginal statistics of local descriptors, but rather in their spatial relation ships and correlations. To this aim, we constructed several families of descriptors based on correlations or similarities between neighboring patches.

The experiments were performed on diferent underlying representations, including local luminance averages, local luminance standard deviations, raw image patches, and local HOG descriptors. In all cases, neighboring spatial cells were compared at fixed ofsets (horizontal, vertical, diagonal, and distance-two neighbors), and the resulting similarity distributions were summarized through low-dimensional statistics such as mean, standard deviation, and quantiles.

For HOG-based descriptors, neighboring cells were represented by their full 36- dimensional orientation histograms. For luminance and raw-patch representations, analogous correlation statistics were computed directly on pixel intensities or local patch vectors.

Despite the variety of descriptors considered, all these approaches exhibited limited predictive power. The resulting regression scores remained substantially below those obtained using the full spatial HOG representation. Together with the shufling experiments, these results suggest that the discriminative information depends on spatial organization, but that this organization cannot be adequately described by simple pairwise relationships between neighboring regions. This motivates the search for representations capable of capturing richer multiscale interactions, which is the subject of the next section.

<table><tr><td>Covariance descriptor</td><td>Correlated representation</td><td>PC1  $R ^ { 2 }$ </td><td>PC2 R²</td></tr><tr><td>Coarse HOG covariance</td><td>block-level 36-D HOG descriptors</td><td>0.07</td><td>0.04</td></tr><tr><td>Local HOG neighbor corr.</td><td>neighboring 36-D HOG cells</td><td>0.18</td><td>0.33</td></tr><tr><td>Luminance mean neighbor corr.</td><td>neighboring scalar luminance cells</td><td>0.04</td><td>0.02</td></tr><tr><td>Luminance-std neighbor corr.</td><td>neighboring scalar contrast cells</td><td>0.02</td><td>0.05</td></tr><tr><td>Raw patch neighbor corr.</td><td>neighboring raw patch vectors</td><td>0.01</td><td>0.01</td></tr></table>

Table 11: Regression performance obtained using pairwise spatial descriptors computed from diferent local image representations. Neighboring regions were compared through correlation or covariance statistics, while the coarse HOG descriptor was obtained by partitioning the image into a 4 × 4 grid and computing covariance statistics between the corresponding block-level HOG descriptors.

## 8 Scattering

The scattering transform is parameterized by two quantities. The parameter J specifies the number of dyadic scales (equivalently, the maximum spatial scale represented), while L denotes the number of wavelet orientations sampled at each scale. Throughout this work, we employed Kymatio scattering descriptors with maximum order two. Consequently, the representation is not limited to first-order wavelet responses, but also includes the zeroth-order low-pass coeficient and second-order coeficients describing interactions between first-order wavelet energies across diferent scales and orientations. The dimensionality of the resulting descriptor is

$$
N ( J , L ) = 1 + J L + \frac { J ( J - 1 ) } { 2 } L ^ { 2 } ,
$$

where the three terms correspond, respectively, to the zeroth-, first-, and secondorder coeficients.

We performed two complementary classes of experiments:

• Global scattering, aimed at assessing how much information is contained in conventional scattering descriptors and how the predictive performance depends on the maximum wavelet scale J.

• Patch scattering, investigating whether the same descriptors become more informative when they are organized as a set of regional descriptors rather than being globally pooled into a single representation.

## 8.1 Global Scattering

We first evaluated the predictive power of conventional global scattering descriptors. Scattering coeficients were computed over the entire image, varying both the maximum wavelet scale J and the angular resolution L. The resulting feature vectors were used to predict the first two CLIP principal components through Ridge regression, with the regularization parameter selected by cross-validation.

Table 12 summarizes the results.
<table><tr><td> $J$ </td><td>Max scale  $2 ^ { J }$ </td><td> $\mathrm { L }$ </td><td>Features</td><td>PC1  $R ^ { 2 }$ </td><td>PC2  $R ^ { 2 }$ </td><td>α</td></tr><tr><td>3</td><td>8</td><td>4</td><td>61</td><td>0.33</td><td>0.40</td><td>1</td></tr><tr><td>3</td><td>8</td><td>8</td><td>217</td><td>0.33</td><td>0.44</td><td>5</td></tr><tr><td>4</td><td>16</td><td>4</td><td>113</td><td>0.32</td><td>0.44</td><td>1</td></tr><tr><td>4</td><td>16</td><td>8</td><td>417</td><td>0.37</td><td>0.47</td><td>5</td></tr><tr><td>5</td><td>32</td><td>4</td><td>181</td><td>0.35</td><td>0.46</td><td>1</td></tr><tr><td>5</td><td>32</td><td>8</td><td>681</td><td>0.40</td><td>0.48</td><td>10</td></tr><tr><td>6</td><td>64</td><td>4</td><td>265</td><td>0.37</td><td>0.47</td><td>1</td></tr><tr><td>6</td><td>64</td><td>8</td><td>1009</td><td>0.41</td><td>0.49</td><td>20</td></tr></table>

Table 12: Ridge regression from order-two scattering descriptors at diferent maximum scales and diferent orientations.

Increasing the maximum scale consistently enlarges the descriptor dimensionality, as explicitly reported in Table12. The predictive performance increases accordingly, but with a relatively modest improvement, passing from $R ^ { 2 } \approx 0 . 3 3$ to $R ^ { 2 }$ ≈ 0.41 for PC1, and from $R ^ { 2 }$ ≈ 0.44 to $R ^ { 2 }$ ≈ 0.49 for PC2. These results indicate that scattering coeficients capture a non-negligible fraction of the information encoded by the CLIP components. However, increasing the receptive field of the scattering transform alone provides diminishing returns. In particular, descriptors including wavelet interactions up to a spatial scale of 32 pixels $( J = 5 )$ already capture almost all the predictive information obtainable from globally pooled scattering features, while doubling the maximum scale to 64 pixels produces virtually no further improvement.

This saturation suggests that the main limitation is not the intrinsic scale of the scattering representation, but rather the use of a single global descriptor. This observation motivates the second series of experiments, where scattering coeficients are extracted independently from multiple image regions before being combined by a learnable model.

Similarly to the case of HOG, before addressing patches, we conducted a few preliminary experiments with MLPs.

A small multilayer perceptron substantially improved predictive performance, reaching approximately R 2 =0.60 for both PC1 and PC2. This result demonstrates that scattering coeficients contain considerable information about the PCA coordinates. However, as discussed in Section 9, predictive performance alone should not be interpreted as equivalent to explanatory power. In particular, the configurations achieving the highest regression scores did not produce the largest or most coherent displacements during inversion. The best inversion results were instead obtained for J=4 and L=4.

<table><tr><td>J</td><td>L</td><td>features</td><td>PC1  $R ^ { 2 }$ </td><td> $\mathrm { P C 2 } ~ R ^ { 2 }$ </td></tr><tr><td>3</td><td>4</td><td>61</td><td>0.54</td><td>0.57</td></tr><tr><td>3</td><td>8</td><td>217</td><td>0.55</td><td>0.59</td></tr><tr><td>4</td><td>4</td><td>113</td><td>0.56</td><td>0.60</td></tr><tr><td>4</td><td>8</td><td>417</td><td>0.58</td><td>0.60</td></tr><tr><td>5</td><td>4</td><td>181</td><td>0.58</td><td>0.61</td></tr><tr><td>5</td><td>8</td><td>681</td><td>0.60</td><td>0.61</td></tr><tr><td>6</td><td>4</td><td>265</td><td>0.60</td><td>0.60</td></tr><tr><td>6</td><td>8</td><td>1009</td><td>0.60</td><td>0.60</td></tr></table>

Table 13: Non linear (MLP) predictors from scattering descriptors. Results measured with Kfolds over 5 splits. Dropout 0.3, 200 epochs.

## 8.2 Patch Scattering

The previous experiments established that scattering descriptors contain substantial information about the dominant PCA coordinates. While linear models achieved moderate predictive performance, a nonlinear MLP achieved $R ^ { 2 }$ values of approximately 0.60 for both principal components. Since the descriptors were computed using order-two scattering transforms, these results already incorporate information about multiscale correlations among oriented structures, rather than merely local edge statistics.

The objective of the patch-based analysis was, therefore, not simply to improve regression accuracy, but to determine whether explicitly preserving the spatial organization of scattering responses yields a representation more closely associated with the image variations underlying the PCA directions. We experimented with several patch sizes and scattering configurations. For 84 × 84 patches, we considered $J = 5$ with both $L = 4$ (181 coeficients) and L = 8 (681 coeficients). For 56 × 56 patches, we used J = 4, L = 4 (113 coeficients); for 28 × 28 patches J = 3, L = 4 (61 coeficients); and for $1 1 2 \times 1 1 2$ patches J = 5, L = 4 (181 coeficients). These configurations make it possible to investigate independently the influence of patch size, maximum represented scale, and angular resolution.

For each image, RGB values were converted to grayscale and partitioned into overlapping square patches. A second-order scattering transform was computed independently on each patch, producing one feature vector per patch. Depending on the values of J and $L ,$ each patch was represented by 61, 113, 181, or 681 coeficients. The collection of patch descriptors forms a spatial feature map whose dimensions depend on the patch size $( \mathrm { e . g . , a 4 \times 4 \times 6 8 1 }$ tensor for 84 × 84 patches). To avoid an explosion in the number of parameters, we mostly worked with L=4. A few experiments with L=8 yielded only marginal improvements.

Patch descriptors were processed by diferent lightweight models to assess various aggregation techniques. Given the relatively strong performance of the linear model on global scattering statistics, we used its prediction as a baseline and trained the patchbased models to estimate a residual correction. We tested a single-layer Transformer and a multi-head attention layer, both with four heads. In addition, we tested both models in two diferent configurations, one with an additional classification token (CL), and one without CL but with a self-learned positional encoding. We applied independent standardization over input Scattering coeficients, consistently with the preprocessing adopted by the Ridge baseline.

We also tested diferent models, comprising a few convolutional neural networks, without obtaining performance improvements.

<table><tr><td rowspan="3"></td><td rowspan="3"></td><td rowspan="3"></td><td colspan="4"> $R ^ { 2 }$  PC1</td><td colspan="4"> $R ^ { 2 }$  PC2</td></tr><tr><td colspan="2">transformer</td><td colspan="2">attention</td><td colspan="2">transformer</td><td colspan="2">attention</td></tr><tr><td></td><td></td><td></td><td></td><td>CL</td><td></td><td></td><td></td></tr><tr><td>patch dim  $2 8 \times 2 8$ </td><td>3</td><td>features  $\overline { { 6 1 \times 1 2 \times 1 2 } }$ </td><td>CL 0.50</td><td>pos 0.56</td><td>CL 0.43</td><td>pos 0.50</td><td>0.50</td><td>pos 0.54</td><td>CL 0.47</td><td>pos 0.49</td></tr><tr><td> $5 6 \times 5 6$ </td><td>4</td><td> $1 1 3 \times 6 \times 6$ </td><td>0.51</td><td>0.57</td><td>0.47</td><td>0.52</td><td>0.53</td><td>0.55</td><td>0.52</td><td>0.53</td></tr><tr><td> $8 4 \times 8 4$ </td><td>4</td><td> $1 1 3 \times 4 \times 4$ </td><td>0.51</td><td>0.57</td><td>0.50</td><td>0.52</td><td>0.51</td><td>0.54</td><td>0.48</td><td>0.52</td></tr><tr><td>112 × 112</td><td>5</td><td> $1 8 1 \times 3 \times 3$ </td><td>0.52</td><td>0.58</td><td>0.52</td><td>0.58</td><td>0.51</td><td>0.53</td><td>0.52</td><td>0.52</td></tr></table>

Table 14: PC1 and PC2 regression values for a lightweight model acting residually over a linear baseline. We tested a single layer Transform and a pure multi-head attention layer, both with four head. The two models have been tested in two configurations: with a classification token, and with positional encoding with averaging.

The resulting predictive performance generally lies between that of the linear and MLP models. More importantly, despite explicitly preserving the spatial organization of the scattering coeficients, the patch-based models did not produce a corresponding improvement in the inversion experiments. Thus, increasing the sophistication of the aggregation mechanism improves neither the interpretability nor the controllability of the learned relationship. This result further emphasizes that regression accuracy alone is insuficient to identify the image statistics genuinely associated with movement along the principal CLIP directions.

## 9 Inversion

To investigate the visual meaning of the CLIP principal components, we optimized an input image to maximize or minimize the score predicted by our diferentiable scattering surrogate. Since the scattering descriptors and the Transformer regressors are fully diferentiable, gradients can be propagated back to the input image.

Optimization starts from the original image x<sub>0</sub>. Instead of optimizing the image directly, we introduce an unconstrained latent variable u and define the image perturbation as

$$
\delta = \varepsilon \operatorname { t a n h } ( u ) ,
$$

which guarantees that every pixel displacement remains bounded by ±ε. Furthermore, updates are restricted to visually meaningful regions by multiplying the perturbation by a fixed spatial mask M extracted from the original image. The mask is

obtained from a smoothed edge map and concentrates the optimization budget around textured regions and object boundaries, while largely preserving homogeneous areas. The optimized image is therefore

$$
x = \mathrm { c l i p } ( x _ { 0 } + M \odot \delta , 0 , 1 ) .
$$

The optimization objective is

$$
\begin{array} { r } { \mathcal { L } ( x ) = \mathrm { d i r e c t i o n } \cdot s ( x ) + \lambda _ { \mathrm { T V } } \mathrm { T V } ( \delta ) + \lambda _ { \mathrm { H F } } \mathrm { H F } ( \delta ) , } \end{array}
$$

where direction $\in \ \{ - 1 , + 1 \}$ determines the desired direction of displacement along the selected principal component, the total variation term encourages spatial smoothness, and the high-frequency penalty suppresses perturbations dominated by fine-scale Fourier components. Images were optimized for 150 iterations using the Adam optimizer with an initial learning rate of 1e-02.

In Table 15 we report the movements obtained in the PCA space after re-projecting the image modified through the inversion technique.

Although the induced displacements are substantially smaller than those obtained by directly inverting CLIP, their consistent magnitude indicates that scattering captures part of the visual information associated with the dominant CLIP directions.

<table><tr><td rowspan=2 colspan=4>PC1 displacementfeatures               model                   std</td></tr><tr><td rowspan=1 colspan=1>model</td><td rowspan=1 colspan=1>mean</td></tr><tr><td rowspan=1 colspan=1>CLIP embeddinggs</td><td rowspan=1 colspan=1>CLIP</td><td rowspan=1 colspan=1>0.079</td><td rowspan=1 colspan=1>0.056</td></tr><tr><td rowspan=1 colspan=1>hog field</td><td rowspan=1 colspan=1>ridgemlp</td><td rowspan=1 colspan=1>0.0100.09</td><td rowspan=1 colspan=1>0.090.07</td></tr><tr><td rowspan=1 colspan=1>global scattering J3 L4</td><td rowspan=1 colspan=1>ridgemlp</td><td rowspan=1 colspan=1>0.0160.020</td><td rowspan=1 colspan=1>0.0170.021</td></tr><tr><td rowspan=1 colspan=1>global scattering J3 L8</td><td rowspan=1 colspan=1>ridge</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>0.016</td></tr><tr><td rowspan=1 colspan=1>global scattering J4 L4</td><td rowspan=1 colspan=1>ridgemlp</td><td rowspan=1 colspan=1>0.0180.019</td><td rowspan=1 colspan=1>0.0200.023</td></tr><tr><td rowspan=1 colspan=1>global scattering J4 L8</td><td rowspan=1 colspan=1>ridge</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.018</td></tr><tr><td rowspan=1 colspan=1>global scattering J5 L4</td><td rowspan=1 colspan=1>ridgemlp</td><td rowspan=1 colspan=1>0.0160.016</td><td rowspan=1 colspan=1>0.0170.022</td></tr><tr><td rowspan=1 colspan=1>global scattering J5 18</td><td rowspan=1 colspan=1>ridge</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>0.016</td></tr><tr><td rowspan=1 colspan=1>global scattering J6 L4</td><td rowspan=1 colspan=1>ridgemlp</td><td rowspan=1 colspan=1>0.0140.015</td><td rowspan=1 colspan=1>0.0150.023</td></tr><tr><td rowspan=1 colspan=1>global scattering J6 L8</td><td rowspan=1 colspan=1>mlp</td><td rowspan=1 colspan=1>0.014</td><td rowspan=1 colspan=1>0.015</td></tr><tr><td rowspan=1 colspan=1>patch-28 scattering</td><td rowspan=1 colspan=1>Res. Attention</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.019</td></tr><tr><td rowspan=1 colspan=1>patch-56 scattering</td><td rowspan=1 colspan=1>Res. Attention</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.020</td></tr><tr><td rowspan=1 colspan=1>patch-84 scattering</td><td rowspan=1 colspan=1>Res. Attention</td><td rowspan=1 colspan=1>0.018</td><td rowspan=1 colspan=1>0.019</td></tr><tr><td rowspan=1 colspan=1>patch-128 scattering</td><td rowspan=1 colspan=1>Res. Attention</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.019</td></tr><tr><td rowspan=2 colspan=1>global scattering J3-5</td><td rowspan=2 colspan=1>Ridge Ensemblemlp Ensemble</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>0.017</td></tr><tr><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>0.019</td></tr></table>

Table 15: Summary of movements in the 2D PCA space induced by inversion of diferent regression techniques over diferent extracted features. The mean is computed over all images in AI-Pastiche.

The magnitude of the displacement obtained through scattering inversion is approximately one quarter of that obtained by direct CLIP inversion. While these quantities should not be interpreted as a direct measure of explained variance, the comparison provides a useful indication of the extent to which scattering captures the image variations to which CLIP is sensitive.

Compared with Ridge regression, the MLP provides a modest improvement in the mean displacement for $J = 3$ and $J = 4$ . This advantage disappears for larger values of J: at $J = 5$ and J = 6, the mean displacement decreases while its variance remains approximately unchanged. This suggests that the nonlinear model primarily amplifies the inversion signal already captured by the simpler representation. At larger scattering scales, the additional information does not translate into a more coherent displacement, and the variability across images becomes comparable to, or larger than, the mean efect. In this regime, the inversion is therefore increasingly dominated by image-dependent variability rather than by a stable direction associated with the target principal component.

In Table15, we only report the values for the Attention layers with self-learned positional encoding, but the results are similar for the other architectures. Explicitly preserving the spatial organization of patch-level scattering coeficients also provides no systematic improvement over the global scattering baseline. Thus, the information missing from the scattering representation does not appear to be recoverable simply by retaining the spatial arrangement of local scattering responses.

In Figure 17a, we show the efective displacement of one hundred random samples of AI-Pastiche towards the Human region.

![](images/8ba8faad6664399d91d0fc17e164192e66f248a0382c008e221fe667bcafa094.jpg)  
(a) AI to Human

![](images/99e2d6c461dac4c0b4e4748d506932a7b76dcb90e2ad15fa92789f0601cb7cb7.jpg)  
(b) Human to AI  
Fig. 17: Movements in 2D space induced through scattering inversion. Movements are governed by the x-axis (PC1).

Similarly, in Figure 17b we show the efective displacement of one hundred random samples of the paintings of the National Gallery towards the AI region.

An important aspect of these results is the large variance of the induced displacement, also visible in the irregular trajectories of Figure 17. While scattering inversion produces a systematic displacement on average, its efect varies considerably across individual images. This variability provides further evidence that scattering captures only part of the statistical information governing the CLIP principal directions.

In order to reduce the variance, we computed the inversion score using an ensemble of the four ridge regressors for $\mathrm { J } { = } 3$ to J=6. Denoting by $s _ { j }$ the prediction of the model associated with scattering features at J=j, the optimization objective is based on the ensemble score

$$
s ( x ) = \frac 1 4 \left( s _ { 3 } ( x ) + s _ { 4 } ( x ) + s _ { 5 } ( x ) + s _ { 6 } ( x ) \right) .
$$

This is the model in the last row of Table 15. The motivation for the ensemble is that each model captures similar, but not identical, aspects of the CLIP components. During optimization, the four predictors therefore act as implicit regularizers for one another, discouraging image modifications that improve only a single surrogate while leaving the others unchanged. Working with the ensemble also reduces the introduction of artifacts so typical of inversion techniques.

## 10 Discussion

## 10.1 Interpreting the Scattering Representation

Although the scattering transform provides a rich and stable representation of image textures, interpreting its coeficients is not straightforward. In particular, each coeficient is associated with a specific combination of scattering order, spatial scale, and orientation, making it dificult to directly relate individual coeficients to meaningful image characteristics. Moreover, the relatively large number of coeficients produced by the transform further complicates their visual inspection and comparison across diferent image datasets. Displaying the distribution of every coeficient separately would require dozens of plots, making it challenging to identify the dominant trends and diferences.

To facilitate the interpretation of the scattering representation, we aggregate coefficients that share the same physical meaning. The objective is not to reduce the descriptive power of the scattering transform, but rather to obtain a compact representation that preserves its principal scale-dependent information while enabling an efective visual comparison between datasets.

For a scattering configuration adopted in this work, with $J = 3$ scales and $L = 4$ orientations, each image is represented by a vector of 61 coeficients. These naturally decompose as

$$
6 1 = 1 + 1 2 + 4 8 ,
$$

corresponding to one zeroth-order coeficient, twelve first-order coeficients, and fortyeight second-order coeficients.

The first-order coeficients are organized into three spatial scales, with four orientations associated with each scale. Since the objective of this analysis is to compare the response at diferent scales rather than along individual directions, the four orientation responses are averaged, producing one coeficient for each scale. Consequently, the twelve first-order coeficients are reduced to three aggregated features.

The second-order coeficients describe interactions between pairs of scales. For $J = 3$ , only three scale combinations are possible,

$$
( j _ { 1 } , j _ { 2 } ) \in \{ ( 0 , 1 ) , ( 0 , 2 ) , ( 1 , 2 ) \} ,
$$

each containing all $4 \times 4$ orientation combinations. Averaging over these orientation pairs yields one coeficient for each scale pair, reducing the forty-eight second-order coeficients to three aggregated features.

The resulting representation, therefore, consists of only seven features:

• $S _ { 0 }$ , the zeroth-order coeficient, representing the low-frequency average intensity;

$S _ { 1 } ( 0 ) , \ S _ { 1 } ( 1 )$ , and $S _ { 1 } ( 2 )$ , the first-order coeficients averaged over orientations, characterizing the image response at increasing spatial scales;

$S _ { 2 } ( 0 , 1 ) , \ S _ { 2 } ( 0 , 2 )$ , and $S _ { 2 } ( 1 , 2 )$ , the second-order coeficients averaged over orientation pairs, describing the interactions between structures occurring at diferent scales.

This aggregation reduces the dimensionality from 61 to $7$ features while preserving the information associated with the diferent spatial scales. The resulting representation is considerably easier to interpret visually, allowing the distributions of the aggregated coeficients to be compared directly using histograms or other statistical visualizations.

Figure 18 describes the distribution of scattering coeficients at $\mathrm { J } = 3$ in the case of AI-Pastiche and paintings of the National Gallery of $\mathrm { A r t }$

![](images/8b2c0d523017cb6168f28692b7b40698cf40013e8db272f88851d66283f2f7c5.jpg)

![](images/203ff09d5ea8b23cd3a6ac991076f6aaacf465082518d279465c2e8556877ebc.jpg)

![](images/e5861f783113d779f0a3ca86a57480c14596779e270bf9cba2d45e8cfbe8010f.jpg)

![](images/930434e25610ff70253144c1dce47b381c3726ade073dc4b0cc32a3a55143f19.jpg)

![](images/18bba8f54d294adbb277eb95dd6ee0aeb5c2df9f1718ab7dbf96e52cc9120de3.jpg)

![](images/1338262a9a4b08f9f4d29c6e36592cefc2d3111288902c120b6e7963ff72d1ef.jpg)

![](images/e129eb948c6c6a1f5abf7da39f0f01195f61965dcb125a5e19754c90dce23db6.jpg)  
Fig. 18: Distribution of scattering coeficients along aggregated groups at $\mathrm { J } { = } 3$ for works of AI-Pastiche (AI-generated) and NGA (human).

We can repeat a similar computation at $\mathrm { J } { = } 4$ , resulting in this case in 11 resulting features, whose distribution is visualized in Figure 19.

Despite the diferent scattering configurations, both figures exhibit the same overall behavior, indicating that the observed diferences between AI-generated and human images are robust with respect to the choice of the maximum scattering scale.

The zeroth-order coeficient, $S _ { 0 }$ , shows a substantial overlap between the two datasets for both values of $^ { J , }$ that is consistent with our previous discoveries.

![](images/bdeaf58e0c30e5810e523382c490b8ee0e6fcffea3faf013ea893e9987ec4934.jpg)  
Fig. 19: Distribution of scattering coeficients along aggregated groups at $\mathrm { J } { = } 4$ for works of AI-Pastiche (AI-generated) and NGA (human).

A markedly diferent behavior is observed for the first-order coeficients. For every spatial scale, the distributions corresponding to the AI-generated images are consistently shifted towards larger values with respect to the NGA dataset. This indicates that the wavelet responses of the synthetic images are, on average, stronger across all analyzed scales. Moreover, the magnitude of the shift remains remarkably consistent as the scale increases, suggesting that this is a global characteristic of the generated images rather than an efect confined to a particular frequency range.

The most pronounced diferences are observed for the second-order coeficients. For all combinations of scales, the distributions associated with the AI images exhibit larger values and a longer right-hand tail than those corresponding to the human images. Since second-order scattering coeficients quantify the interactions between image structures occurring at diferent spatial scales, these results suggest that AIgenerated images possess stronger multiscale dependencies and more structured crossscale correlations.

The analysis performed with $J = 4$ further reinforces this interpretation. Besides reproducing the same behavior observed for the common features already present with $J = 3 ,$ , the additional coarse-scale coeficients, namely $S _ { 1 } ( 3 ) , S _ { 2 } ( 0 , 3 ) , S _ { 2 } ( 1 , 3 )$ , and $S _ { 2 } ( 2 , 3 )$ , display the same systematic shift towards larger values for the AI dataset. Therefore, the observed discrepancy is not limited to fine or medium spatial scales, but also persists at coarser resolutions. This indicates that the diferences between AI-generated and human images extend across the entire hierarchy of spatial scales captured by the scattering transform.

Considering scattering over spatial patches, it is possible to observe that the spatial coeficient of variation $\sigma / \mu$ is approximately the same for the two sets of images. So the larger scattering coeficients are not due to a regular, repeated texture, but to an amplified scattering response: since the local scattering coeficients are globally larger, their absolute variance also increases, and the increase in variability is approximately proportional to the increase in the mean.

## 10.2 A Speculative Interpretation

The previous analyses establish several robust empirical observations but do not by themselves explain the mechanisms responsible for them. In the following, we discuss

two speculative hypotheses suggested by our experimental findings. These hypotheses are intended as possible interpretations rather than definitive explanations, and should therefore be regarded as directions for future investigation.

## 10.2.1 Why do AI-generated paintings exhibit systematically larger scattering responses?

A possible explanation for the systematically larger scattering responses observed in generated images is that they arise as a consequence of the constraints imposed during training. Generative models are not optimized solely to reproduce the empirical pixel distribution of real images. Instead, their training objectives typically favor perceptual sharpness, structural coherence, realistic texture, and consistency between local details and global image content.

These constraints may introduce a statistical bias toward image structures that are particularly salient to multiscale wavelet filters. In particular, the penalization of blurred or weakly defined structures may reinforce edges, contours, and local contrast variations, thereby increasing the first-order scattering coeficients. At the same time, objectives that encourage fine-scale details to remain consistent with larger semantic and geometric structures may strengthen the coupling between spatial scales, leading to larger second-order scattering coeficients. These operations can produce images that are visually plausible but whose frequency statistics difer systematically from those of real images. Such spectral discrepancies have been repeatedly observed in generated imagery, including both excesses and structured inconsistencies in particular frequency band [31].

Under this interpretation, the higher scattering response of generated images is not simply a consequence of increased signal energy. Rather, it reflects a systematic amplification of perceptually relevant and multiscale-coherent structures induced by the training objective. The fact that the patch-level coeficient of variation remains comparable, or is slightly lower, for generated images further suggests that this efect is approximately multiplicative: both the mean response and its absolute spatial variability increase, while the relative variability remains largely unchanged.

## 10.2.2 Why is CLIP particularly sensitive to mesoscale statistics?

A second question raised by our experiments concerns why CLIP appears to be particularly sensitive to mesoscale image statistics. One possible explanation lies in the contrastive objective used during pretraining. Positive image–text pairs are typically subjected to image transformations while preserving their semantic correspondence, so that the learned representation is encouraged to remain stable under changes that alter local image appearance without modifying the underlying content. Interestingly, the AI–human separation observed in our experiments is remarkably robust to several transformations of this kind, including cropping, resizing, grayscale conversion, and moderate degradation.

Under this perspective, the discriminative signal is unlikely to depend on isolated pixels or highly localized image structures. Instead, it should reside in statistical regularities that survive such transformations while remaining suficiently informative to characterize the image. Mesoscale scattering statistics possess precisely these properties: they aggregate information across multiple spatial scales, are stable to local deformations, and retain structured information about interactions between image features at diferent scales.

We therefore hypothesize that the contrastive learning objective may favor visual cues with invariance properties similar to those captured by scattering representations. This does not imply that CLIP explicitly computes scattering-like features, but rather that distributed multiscale statistics may constitute a particularly efective form of visual evidence for a representation that must remain stable under substantial variation of the input. The robustness results reported in this work are consistent with this interpretation, although establishing a direct causal link between CLIP’s training procedure and its sensitivity to mesoscale statistics would require dedicated experiments.

## 10.3 Limitations

The objective of the present work is not to provide an exhaustive characterization of every factor contributing to the separation between human and AI-generated paintings, but rather to investigate whether this phenomenon can be explained in terms of interpretable statistical image representations. Accordingly, our analysis follows a progressive strategy, moving from simple image statistics to increasingly expressive descriptors, with the aim of identifying the level of statistical complexity required to account for the observed geometry of the CLIP embedding space. Several important questions nevertheless remain open.

First, our analysis is restricted to CLIP representations. Although CLIP is one of the most influential vision foundation models, it remains to be established whether the statistical mechanisms identified in this work are specific to contrastive vision– language models or constitute a more general property of modern artificial vision systems. Extending the same methodology to self-supervised vision transformers and other foundation models represents a natural direction for future research.

Second, our experiments focus exclusively on artistic images. This choice is motivated by the central role of human perception in artistic evaluation and by the particularly clear separation observed in this domain. Whether similar statistical mechanisms govern the distinction between human and AI-generated natural photographs remains an open question.

Finally, although the robustness experiments rule out several simple explanations based on color, image quality, or preprocessing artifacts, they do not isolate every possible source of statistical regularity introduced by image-generation pipelines. For example, hidden generator-specific signatures or watermarking mechanisms cannot be completely excluded. We consider such explanations unlikely to account for the phenomenon in its entirety, given the distributed multiscale statistical diferences revealed by the scattering analysis and the consistency of the observed behavior across different experimental settings. Nevertheless, explicitly disentangling these efects would further strengthen the interpretation proposed in this work.

These limitations do not alter the main empirical conclusions of the study: interpretable mesoscale statistics capture a meaningful component of the intrinsic geometry of the CLIP embedding space, but provide only a partial account of the observed separation. Rather than closing the problem, the present results identify the level at which part of the relevant visual information becomes accessible and define the scope for further investigation.

## 11 Conclusion

We investigated the spontaneous separation between human and AI-generated paintings in the CLIP embedding space. Through a progressive analysis of increasingly expressive interpretable image representations, we showed that simple image statistics and local gradient descriptors account for only a limited part of the observed geometry, whereas multiscale scattering descriptors provide a substantially more informative statistical representation. Scattering, nevertheless, provides only a partial account of the phenomenon: it captures a meaningful component of the visual information associated with the dominant principal directions, but a large part of their statistical structure remains unexplained.

A natural question is therefore what kind of interpretable representation could capture the statistical structure that remains unexplained. Scattering provides a multiscale description based on localized wavelet interactions and is conceptually close to the hierarchical processing characteristic of convolutional representations. CLIP, however, relies on a Vision Transformer, whose attention-based architecture can encode longrange and content-dependent interactions that are not naturally described within this framework. Developing interpretable statistical descriptors that more closely reflect attention-based processing may therefore provide access to aspects of the CLIP representation that are not captured by scattering, and constitutes an important direction for future investigation.

An equally important direction concerns the generality of the observed phenomenon. The present study focuses on CLIP because of its widespread adoption and well-established embedding space. Whether similar statistical mechanisms emerge in other vision foundation models, including self-supervised and purely visual architectures, remains an important open question. Comparative analyses across diferent representation learning paradigms may help distinguish properties that are specific to CLIP from more general characteristics of artificial vision.

More broadly, our results call into question an assumption that is often made implicitly throughout the computer vision community: namely, that modern vision foundation models extract visual information that is broadly aligned with human perception. The systematic discrepancies observed in this work suggest that this assumption deserves closer experimental scrutiny rather than being taken for granted.

This observation also has potential consequences for the interpretation of featurebased evaluation metrics. Measures such as FID and related perceptual distances rely on representations extracted by deep vision models and are therefore meaningful only insofar as these representations reflect perceptually relevant image diferences. If artificial and human vision rely on systematically diferent statistical evidence, such metrics may reflect biases that do not necessarily coincide with human visual judgment.

Finally, we believe that these questions are particularly relevant in the artistic domain, where visual perception constitutes the primary medium of evaluation. If humans and artificial systems rely on diferent visual evidence, then agreement in semantic understanding does not necessarily imply agreement in aesthetic judgment. Rather than assuming that artificial vision and human perception converge toward a common notion of visual quality, future research should seek to understand where they coincide, where they diverge, and what these diferences reveal about both natural and artificial intelligence.

## Declarations

• Conflict of interest. The author declares no conflict of interest.

• Data and code availability. Code, embeddings and precomputed features are available at the gitbub repository Interpreting Human AI separability.

## References

[1] Liao, P., Li, X., Liu, X., Keutzer, K.: The artbench dataset: Benchmarking generative models with artworks. CoRR abs/2206.11404 (2022) https://doi.org/ 10.48550/ARXIV.2206.11404 2206.11404

[2] Asperti, A., Dess\`ı, L., Wu, N.: Art through clip’s eyes. J. Comput. Cult. Herit. (2026) https://doi.org/10.1145/3812546 . Just Accepted

[3] Asperti, A., George, F., Marras, T., Stricescu, R.C., Zanotti, F.: A critical assessment of modern generative models’ ability to replicate artistic styles. Big Data and Cognitive Computing 9(9), 1–26 (2025) https://doi.org/10.3390/bdcc9090231

[4] Art, N.G.: National Gallery of Art Open Data Program. Accessed: 2024-01-29 (2024). https://www.nga.gov/open-access-images/open-data.html

[5] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR, (2021). Accessed: 2025-02-13. http://proceedings.mlr.press/v139/radford21a.html

[6] Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F.P., Lakkaraju, H.: Interpreting CLIP with sparse linear concept embeddings (splice). In: Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J.M., Zhang, C. (eds.) Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 (2024). http://papers.nips.cc/paper files/paper/2024/hash/996bef37d8a638f37bdfcac2789e835d Abstract-Conference.html

[7] Goh, G., †, N.C., †, C.V., Carter, S., Petrov, M., Schubert, L., Radford, A., Olah, C.: Multimodal neurons in artificial neural networks. Distill (2021) https: //doi.org/10.23915/distill.00030 . https://distill.pub/2021/multimodal-neurons

[8] Gandelsman, Y., Efros, A.A., Steinhardt, J.: Interpreting clip’s image representation via text-based decomposition. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, (2024). https://openreview.net/forum?id=5Ca9sSzuDp

[9] Oikarinen, T.P., Weng, T.: Clip-dissect: Automatic description of neuron representations in deep vision networks. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, (2023). https://openreview.net/forum?id=iPWiwWHc1V

[10] Wang, S., Wang, O., Zhang, R., Owens, A., Efros, A.A.: Cnn-generated images are surprisingly easy to spot... for now. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pp. 8692–8701. Computer Vision Foundation / IEEE, (2020). https://doi.org/10.1109/CVPR42600.2020.00872 https://openaccess.thecvf.com/content CVPR 2020/html/Wang CNN-Generated Images Are Surprisingly Easy to Spot... for Now CVPR 2020 paper.html

[11] Gragnaniello, D., Cozzolino, D., Marra, F., Poggi, G., Verdoliva, L.: Are GAN generated images easy to detect? A critical analysis of the state-of-the-art. In: 2021 IEEE International Conference on Multimedia and Expo, ICME 2021, Shenzhen, China, July 5-9, 2021, pp. 1–6. IEEE, (2021). https://doi.org/10.1109/ ICME51207.2021.9428429 . https://doi.org/10.1109/ICME51207.2021.9428429

[12] Cozzolino, D., Poggi, G., Corvi, R., Nießner, M., Verdoliva, L.: Raising the bar of ai-generated image detection with CLIP. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024 - Workshops, Seattle, WA, USA, June 17-18, 2024, pp. 4356–4366. IEEE, (2024). https://doi.org/10.1109/ CVPRW63382.2024.00439 . https://doi.org/10.1109/CVPRW63382.2024.00439

[13] Moskowitz, A.G., Gaona, T., Peterson, J.: Detecting ai-generated images via CLIP. CoRR abs/2404.08788 (2024) https://doi.org/10.48550/ARXIV.2404. 08788 2404.08788

[14] Gaintseva, T., Kushnareva, L., Magai, G., Piontkovskaya, I., Nikolenko, S.I., Benning, M., Barannikov, S., Slabaugh, G.G.: Improving interpretability and robustness for the detection of ai-generated images. CoRR abs/2406.15035 (2024) https://doi.org/10.48550/ARXIV.2406.15035 2406.15035

[15] Tuytelaars, T., Mikolajczyk, K.: Local invariant feature detectors: A survey. Found. Trends Comput. Graph. Vis. 3(3), 177–280 (2007) https://doi.org/10. 1561/0600000017

[16] Vigneron, V., Sylvie, L.: Statistics for image processing. Pattern Recognition Letters 31, 2191 (2010) https://doi.org/10.1016/j.patrec.2010.07.003

[17] Lowe, D.G.: Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vis. 60(2), 91–110 (2004) https://doi.org/10.1023/B:VISI.0000029664. 99615.94

[18] Dalal, N., Triggs, B.: Histograms of oriented gradients for human detection. In: 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2005), 20-26 June 2005, San Diego, CA, USA, pp. 886– 893. IEEE Computer Society, (2005). https://doi.org/10.1109/CVPR.2005.177 . https://doi.org/10.1109/CVPR.2005.177

[19] Portilla, J., Simoncelli, E.P.: A parametric texture model based on joint statistics of complex wavelet coeficients. Int. J. Comput. Vis. 40(1), 49–70 (2000) https: //doi.org/10.1023/A:1026553619983

[20] Simoncelli, E.P., Olshausen, B.A.: Natural image statistics and neural representation. Annual review of neuroscience 24(1), 1193–1216 (2001)

[21] Mallat, S.: Group invariant scattering. CoRR abs/1101.2286 (2011) 1101.2286

[22] Bruna, J., Mallat, S.: Invariant scattering convolution networks. IEEE Trans. Pattern Anal. Mach. Intell. 35(8), 1872–1886 (2013) https://doi.org/10.1109/ TPAMI.2012.230

[23] Sifre, L., Mallat, S.: Rotation, scaling and deformation invariant scattering for texture discrimination. In: 2013 IEEE Conference on Computer Vision and Pattern Recognition, Portland, OR, USA, June 23-28, 2013, pp. 1233– 1240. IEEE Computer Society, (2013). https://doi.org/10.1109/CVPR.2013.163 . https://doi.org/10.1109/CVPR.2013.163

[24] Erhan, D., Bengio, Y., Courville, A., Vincent, P.: Visualizing higher-layer features of a deep network. Technical Report 1341, University of Montreal, Canada (June 2009)

[25] Yosinski, J., Clune, J., Nguyen, A.M., Fuchs, T.J., Lipson, H.: Understanding neural networks through deep visualization. CoRR abs/1506.06579 (2015) 1506.06579

[26] Mahendran, A., Vedaldi, A.: Understanding deep image representations by inverting them. In: IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pp. 5188–5196. IEEE Computer Society, (2015). https://doi.org/10.1109/CVPR.2015.7299155 . https://doi.org/10.1109/CVPR.2015.7299155

[27] Dosovitskiy, A., Brox, T.: Inverting visual representations with convolutional

networks. In: 2016 IEEE Conference on Computer Vision and Pattern Recogni tion, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 4829–4837. IEEE Computer Society, (2016). https://doi.org/10.1109/CVPR.2016.522

[28] Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I.J., Fergus, R.: Intriguing properties of neural networks. In: Bengio, Y., LeCun, Y. (eds.) 2nd International Conference on Learning Representations, ICLR 2014, Banf, AB, Canada, April 14-16, 2014, Conference Track Proceedings (2014). http://arxiv.org/abs/1312.6199

[29] Goodfellow, I.J., Shlens, J., Szegedy, C.: Explaining and harnessing adversarial examples. In: Bengio, Y., LeCun, Y. (eds.) 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings (2015). http://arxiv.org/abs/1412.6572

[30] Fu, T., Conde, J., Mart´ınez, G., Reviriego, P., Merino-G´omez, E., Moral, F.: Artificial intelligence and misinformation in art: Can vision language models judge the hand or the machine behind the canvas? CoRR abs/2508.01408, 1–18 (2025) 2508.01408

[31] Dzanic, T., Shah, K., Witherden, F.D.: Fourier spectrum discrepancies in deep network generated images. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, Virtual (2020). https://proceedings.neurips.cc/paper/2020/hash/1f8d87e1161af68b81bace188a1ec624- Abstract.html