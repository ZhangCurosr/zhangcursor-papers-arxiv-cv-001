# Beauty is in the ELBO of the Beholder: A Variational Account of Processing Fluency in Face Perception

Francisco M. L´opez

lopez@fias.uni-frankfurt.de

Frankfurt Institute for Advanced Studies, Germany

School of Computer Science and Engineering, UNSW Sydney, Australia

Jochen Triesch

triesch@fias.uni-frankfurt.de

Frankfurt Institute for Advanced Studies, Germany

Goethe University Frankfurt, Germany

## Abstract

Facial attractiveness has been linked to statistical regularities such as symmetry and averageness, suggesting that beauty may depend on the ease with which a face is perceived. We empirically test this hypothesis by training variational autoencoders on four face datasets without attractiveness supervision and evaluating their representations on the 597 faces from the Chicago Face Database. Across models, human attractiveness ratings closely aligns with the direction defined by the VAE evidence lower bound (ELBO) in ratedistortion space. Independently learned latent spaces contain an attractiveness direction that transfers strongly across random initializations and training data. We also find that attractive faces are more prototypical in both shape and latent space. Our results connect classic accounts of aesthetics with learned generative models and provide empirical support for a variational interpretation of the processing fluency theory of aesthetic pleasure.

Keywords: Variational autoencoders; Generative models; Rate-distortion; Attractiveness

## 1. Introduction

What makes a face beautiful? Classic accounts in face perception link attractiveness to statistical regularities such as symmetry (Rhodes et al., 1999; Perrett et al., 1999; Bertamini et al., 2019), averageness (Langlois and Roggman, 1990; Rhodes et al., 1999; Trujillo et al., 2014; Lee et al., 2025; Burton et al., 2005), sexual dimorphism (Hoss et al., 2005; Lee et al., 2025; Kleisner et al., 2024), and overall prototypicality within a population (Ryali et al., 2020). These findings suggest that beauty may depend on the ease with which a face is perceived by the observer. The processing fluency theory establishes that stimuli that can be represented or retrieved more easily are typically evaluated more positively (Reber et al., 1998, 2004). While this fluency heuristic has been extensively studied in behaviora experiments (Reber et al., 1998, 2004; Winkielman et al., 2006; Trujillo et al., 2014; Ryal et al., 2020), it remains unclear how processing fluency emerges in representation learning and how it can be instantiated in computational models of face perception.

Previous computational accounts have proposed explanations of aesthetic value in terms of the eficiency or predictability of perceptual representations. Eficient coding accounts have suggested that preferences may arise from the adaptation of the perceptual system to the sensory statistics (Renoult et al., 2016; L´opez et al., 2024, 2026). Similarly, predictive coding approaches emphasize uncertainty reduction as a source of aesthetic preference (Frascaroli et al., 2023; Van de Cruys et al., 2023). Most closely related to our proposal, recent works have formalized aesthetic value under probabilistic scalar models (Brielmann and Dayan, 2022; Brielmann et al., 2023; Nath et al., 2026), with immediate reward determined by the fluency of processing the stimulus and an additional component capturing expected learning progress (Schmidhuber, 2008). These works define the core of computational neuroaesthetics research but rely on overly abstract models that lack the detailed face representation qualities of modern generative models, e.g., Karras et al. (2019); Higgins et al. (2017) (see Phillips and White (2026) for a recent review).

Addressing this mismatch between theories of neuroaesthetics and performant generative models of faces, we inquire how processing fluency can be modeled in generative models. Variational autoencoders (VAEs) (Kingma and Welling, 2013; Kingma et al., 2017; Higgins et al., 2017) learn probabilistic representations that balance two competing components in their compressive bottlenecks: preserving enough information about an input to reconstruct it accurately and maintaining a regularized latent representation that remains close to a prior distribution. We postulate that this unsupervised representation learning objective, known as the evidence lower bound (ELBO) (Kingma and Welling, 2013), provides a computationally explicit notion of processing fluency. The ease with which a stimulus is represented can be quantified by this trade-of between the reconstruction distortion and the regularization rate.

We investigate this hypothesis using convolutional VAEs trained independently on four face image datasets without attractiveness supervision. We then evaluate all learned representations on the 597 faces from the Chicago Face Database (CFD; Ma et al. 2015), for which human attractiveness ratings are available. We find that human attractiveness gradients closely match the ELBO in the rate-distortion space across training datasets and demographics. Independently learned latent spaces contain strongly aligned attractiveness directions. Attractive faces are also more prototypical in both shape and latent space and have their facial geometry represented more faithfully. Together, these findings suggest that facial attractiveness can be explained in terms of a variational interpretation of processing fluency. Thus, our work provides a bridge between classical accounts of face perception, processing fluency theories of aesthetic pleasure, and modern generative models.

## 2. A variational interpretation of processing fluency

The processing fluency theory posits that stimuli that are easier to encode, retrieve, or interpret tend to elicit more positive evaluations from human subjects (Reber et al., 1998, 2004). In this view, preferences are not determined by the objective physical properties of a stimulus in isolation, but rather by how that stimulus “fits” into the observer’s perceptual system. The familiarity (Duke et al., 2014), prototypicality (Winkielman et al., 2006; Ryali et al., 2020), and predictability (Ryali et al., 2020) of an experience can all increase its fluency and therefore its preference. Consequently, processing fluency emerges from the organization of neural representations around the sensory experiences of each individual subject. Processing fluency has not yet been modeled explicitly in modern face representation models (Phillips and White, 2026).

Probabilistic generative models provide a natural framework for modeling processing fluency. Perception can be viewed as inference under an internal model of how sensory observations are generated (von Helmholtz, 1867; Friston, 2010). Given an observation x, the system infers latent causes z that can explain that observation. VAEs make use of this principle explicitly by learning an approximate posterior $q _ { \phi } ( z \mid x )$ together with a generative model $p _ { \theta } ( x \mid z )$ under a prior $p ( z )$ . Because the exact marginal likelihood is generally intractable, VAEs minimize the negative evidence lower bound (ELBO) (Kingma and Welling, 2013) given by:

![](images/2011d1f26febfdbe27c0adf7d61dd01f64139925a6a3c22825d631ce57a4be6f.jpg)  
(b) Reconstructions from VAE trained on FairFace (Karkkainen and Joo, 2021)  
Figure 1: Examples of original face images and their reconstructions from an independently trained VAE, all from the Latino-Male category of the Chicago Face Database (CFD), ordered by increasing human attractiveness rating. Numbers above the original faces indicate within-group standardized human attractiveness ratings. Numbers above the reconstructions indicate within-group standardized ELBO ratings. Examples for all other models and demographic groups are provided in Appendix C.

$$
\begin{array} { r } { - \mathrm { E L B O } ( x ) = \underbrace { \mathbb { E } _ { q _ { \phi } ( z \mid x ) } \left[ - \log p _ { \theta } ( x \mid z ) \right] } _ { \mathrm { d i s t o r t i o n } D ( x ) } + \beta \underbrace { \mathrm { K L } \left[ q _ { \phi } ( z \mid x ) \parallel p ( z ) \right] } _ { \mathrm { r a t e } R ( x ) } , } \end{array}\tag{1}
$$

where $\beta$ acts as a rate-distortion trade-of parameter (Higgins et al., 2017), with $\beta = 1$ defining the standard ELBO. The first term measures the fidelity of the latent representation in terms of how well it explains the sensory input. For a Gaussian decoder with fixed variance, this distortion is proportional to a squared reconstruction error (Kingma et al., 2017). The second term is the regularization rate of the latent space. It measures how far the posterior deviates from the prior, quantified as a Kullback-Leibler (KL) divergence.

This decomposition suggests a concrete variational interpretation of processing fluency. A stimulus is represented with ease when it can be explained with high fidelity (low distortion) and at a low representational cost (rate). The ELBO combines these two desiderata into a single quantity and thus provides a scalar measure for processing fluency. Under this interpretation, we predict that aesthetic pleasure stems from a high ELBO (low distortion and low rate) induced by the learned generative representation.

## 3. Methods

To test the proposed variational interpretation of processing fluency, we train convolutional VAEs on face datasets without attractiveness supervision and evaluate whether human

attractiveness judgments align with the resulting rate-distortion geometry and latent representations. The experimental design is summarized below, with an extended description of the methodology provided in Appendix A.

## 3.1. Datasets

We trained unsupervised face representations from four datasets containing thousands of images: FairFace (Karkkainen and Joo, 2021), FFHQ (Karras et al., 2019), CelebA (Liu et al., 2015), and UTKFace (Zhang et al., 2017). These datasets difer substantially in demographic composition, image quality, and pose variability, and therefore allow us to test whether the learned representational geometry generalizes across training distributions.

Human attractiveness judgments were evaluated on the Chicago Face Database (CFD) (Ma et al., 2015). We used the 597 neutral-expression face images with available attractiveness ratings (A). CFD contains systematic diferences in both images and ratings across demographics. All analyses were conditioned on eight separate demographic groups: four ethnicities (Asian, Black, Latino, White) and two genders (Female, Male). Therefore, all values were standardized to within-group zero-mean and unit variance z-scores.

## 3.2. Preprocessing

All training and evaluation images were processed by the same pipeline. Facial landmarks were detected using MediaPipe Face Landmarker (Lugaresi et al., 2019), and each image was transformed with a translation, rotation, and uniform scaling, such that the midpoint between the eyes and the center of the mouth lay at fixed coordinates within the $2 2 4 \times 2 2 4$ output image. The relative morphology of the face was fully preserved. Pixels outside the face region detected by MediaPipe were replaced by a gray color (see Fig. 1). Images for which the required facial landmarks could not be reliably detected, e.g. because of occlusions or large rotations, were discarded.

## 3.3. Variational face representations

We trained convolutional VAEs independently on each of the four training datasets. The encoder consisted of five convolutional layers that compressed the $3 \times 2 2 4 \times 2 2 4$ RGB image to a $5 1 2 \times 7 \times 7$ representation, followed by two separate linear projections for the posterior mean $\mu _ { \phi }$ and log variance log $\sigma _ { \phi } ^ { 2 } .$ , combined via the reparametrization trick (Kingma and Welling, 2013) into a linear latent representation space with a standard Gaussian prior and a baseline dimension size $d _ { z } = 1 2 8$ . For each training dataset, we trained three independent models from diferent random initializations, giving 12 baseline representations in total. Baseline models used $\beta = 1$ and a fixed Gaussian decoder scale $\sigma = 1$ , and were trained for 50 epochs with a learning rate of $2 \times 1 0 ^ { - 4 }$ , and KL warm-up over the first 10 epochs.

After training, all model parameters were frozen and the rates, distortions, and latent representations of the unseen 597 CFD faces were recorded for each model. The distortion for each face and model was computed from a Monte Carlo estimate with 16 samples.

![](images/ab05c5362fb5083bdb3a6cf3a5a0551647edb11c3ab2bda576b3bfb8dd4f3a4e.jpg)  
(a)

![](images/82a298d43ac466ace116f095222e6a3bdbc91047a4453782fce6be20a4960d9b.jpg)  
(b)

![](images/b59b6d5b7758c62b46b3437db873db80154a11612d5137d0281f4516f9736797.jpg)  
(c)  
Figure 2: Alignment between human attractiveness judgments and processing fluency in VAEs. (a) Relationship between human ratings and VAE ELBO for representations learned from CelebA. Each point denotes one CFD identity and the green line shows the linear regression with its 95% bootstrapped confidence interval. (b) Distribution of human attractiveness in the rate-distortion space. Each point denotes one CFD identity colored by its human attractiveness rating. The background shows a descriptive attractiveness landscape, with red indicating lower and green indicating higher attractiveness. (c) Alignment between the empirical direction of increasing human attractiveness (gray) and the direction of increasing ELBO (green) for VAEs trained on diferent datasets.

## 4. Results

## 4.1. Attractiveness aligns with VAE’s ELBO in rate-distortion space

We first inquired whether human attractiveness judgments are systematically organized in the rate-distortion space of the learned variational face representations. Across all four training distributions, attractiveness was positively correlated with the ELBO (Fig. 2) with Pearson correlations of $r = 0 . 3 1 2 \pm . 0 0 6$ for CelebA, $0 . 2 4 9 \pm . 0 0 6$ for FairFace, $0 . 2 8 5 \pm . 0 0 6$ for FFHQ, and $0 . 2 7 3 \pm . 0 0 9$ for UTKFace (all $p < 0 . 0 0 1 )$ . The association was significant for every individual baseline model $( r = 0 . 2 4 4$ to 0.317, $p < 0 . 0 0 1 )$ , despite the models having been trained independently and without aesthetic supervision. The two terms of the ELBO showed complementary associations with attractiveness. Lower distortion was consistently associated with higher attractiveness, with correlations $r = - 0 . 2 5 7$ to −0.300 $( p < 0 . 0 0 1 )$ . Lower rate was also associated with higher attractiveness in every baseline model: $r = - 0 . 1 1 7 \ \mathrm { t o } \ - 0 . 2 5 6 \ ( p < 0 . 0 1 )$ . Thus, attractive faces were both reconstructed more accurately and represented at lower rate. Across all 8 ethnicity×gender groups and 12 baseline models, the within-group correlation between attractiveness and ELBO was positive in all 96/96 combinations. The distortion correlations were likewise positive in $9 6 / 9 6$ combinations, while the rate associations were positive in 94/96. The within-group correlations between attractiveness and ELBO, averaged across all models, ranged from $r = 0 . 1 7 1$ (Latino-Female) to 0.431 (Latino-Male).

We next tested the geometric prediction that the direction of increasing attractiveness in rate-distortion space should align with the direction defined by the variational objective itself. Let $\bar { D } ( x )$ and $\bar { R } ( x )$ denote the within-group standardized rate-distortion coordinates of a face x. For each model, we estimated the empirical attractiveness direction by optimizing the parameters $a _ { D } , a _ { R }$ , and ϵ to fit the relation

$$
\tilde { A } ( x ) = a _ { D } D ( x ) + a _ { R } R ( x ) + \epsilon ,\tag{2}
$$

resulting in a direction vector ${ \bf { a } } = ( a _ { D } , a _ { R } )$ . Likewise, the ELBO direction in these coordinates is given by $\mathbf { e } \propto ( - s _ { D } , - s _ { R } )$ , where $e _ { D }$ and $e _ { R }$ are the residual scales of distortion and rate before standardization (see Appendix A.5). We quantified alignment directly by the cosine similarity

$$
\cos \theta = { \frac { \mathbf { a } ^ { \intercal } \mathbf { e } } { \left\| \mathbf { a } \right\| \left\| \mathbf { e } \right\| } } ,\tag{3}
$$

where a cosine similarity of 1 indicates perfect agreement between the directions of increasing attractiveness and increasing ELBO. Our results showed a consistent agreement across all trained models. The per-dataset cosine similarities were $0 . 9 7 7 \pm . 0 2 3$ for CelebA, $0 . 9 2 9 \pm . 0 2 1$ for FairFace, $0 . 9 8 8 \pm . 0 0 8$ for FFHQ, and $0 . 9 6 9 \pm . 0 2 7$ for UTKFace. Across all 12 individual models, cos θ ranged from 0.905 to 0.997, corresponding to angular deviations of only $4 . 4 ^ { \circ }$ to $2 5 . 2 ^ { \circ }$ . Therefore, the empirical attractiveness gradient consistently pointed toward the region of simultaneously lower distortion and lower rate selected by the variational objective.

Finally, we examined whether the result depended on the baseline parameter choice. Using FairFace as a controlled training distribution, we varied latent dimensionality $( d _ { z } \in$ {64, 128, 256, 512}), regularization bottleneck $( \beta \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \} )$ , and Gaussian decoder scale $( \sigma \in \{ 0 . 5 , 0 . 7 0 7 , 1 , 1 . 4 1 4 \} )$ ), with three independent seeds per configuration. Across all these diferent representation models, the mean correlation between attractiveness ELBO remained positive (0.199 to .254, $p < 0 . 0 0 1 ,$ , while the mean cosine similarity between the empirical attractiveness gradient and the ELBO direction remained high (0.896 to 0.998). The correspondence between human attractiveness and variational geometry was therefore not specific to any particular dataset or configuration but rather a property of the variational representation learning.

## 4.2. Attractiveness geometry is conserved across diferent latent spaces

The previous analysis established a common relationship between attractiveness and the rate-distortion space of the VAEs. We next asked whether these independently trained VAEs also organize their face codes along a shared latent attractiveness direction. We compared the mean posterior representations $\mu _ { \phi } ( x ) \in \mathbb { R } ^ { 1 2 8 }$ relative to their demographic group centroids by aligning the 12 latent spaces via generalized orthogonal Procrustes. For each model $m .$ , we estimated a linear direction $\mathbf { A } _ { m }$ associated with increasing attractiveness. We then measured the cosine similarity between these directions across every pair of independently trained models. The resulting directions were strongly conserved (Fig. 3). Across all 66 model pairs, the mean pairwise cosine similarity was $0 . 8 3 3 \pm 0 . 0 3 0$ , and every pair was positively aligned, with similarities ranging from 0.764 to 0.914. Directions learned from diferent random seeds of the same training distribution were especially similar $( 0 . 8 7 0 \pm 0 . 0 3 2 )$ , and the efect remained strong when comparing models trained on diferent face datasets $( 0 . 8 2 5 \pm 0 . 0 3 4 )$ . For comparison, we permutated the attractiveness labels within demographic groups while preserving the correspondence of faces across representations and found that the mean pairwise direction cosine was $0 . 6 3 2 \pm 0 . 0 3 8$ , reflecting the shared structure of the aligned face manifolds but well below the attractiveness alignment.

Prototypicality and latent direction  
![](images/331fd819ac4f795cb3df768fd0149a1de98a5851ee1bd622e6bb75fe7f940714.jpg)

![](images/6c33baaa0f5590192358004a0e669eb72df1c02149647b30aa735251757ad11d.jpg)  
Attractiveness direction

![](images/741a7070006a04130bd20f6ebaad2451e5861dbbb0571ddd044932d1fbfffa9b.jpg)

(a) Visualization of the attractiveness-aligned latent geometries.  
![](images/4e866e715e0a08819e96cb743a67027362db740fe311720cadf69198e5ded1c4.jpg)

![](images/3942d30f6bed0c5c748f05639a2d3abea2f9378327a61d5908f29afc49ae26d5.jpg)  
(b) Correspondence in pairs of models

![](images/4c82caa6d56cc68bccbdd0f23d3a2b326c182b70f8a485d16134da76f93ba40d.jpg)  
(c) Prediction sources  
Figure 3: Attractiveness geometry is conserved across independently learned latent spaces beyond prototypicality. (a) Two-dimensional projection of the latent spaces onto the attractiveness direction (horizontal) and the first principal component of the remaining orthogonal directions. (b) Pairwise cosine similarity (left) and cross-model transfer (right) between pairs of attractiveness directions in the 12 aligned VAE latent spaces. (c) Held-out predictions from diferent attractiveness sources.

Next, to further investigate the correspondence between attractiveness directions among pairs of models, we performed a cross-model transfer analysis with five repeats of a five-fold cross-validation. The learned direction from one model (source) was transported into another (target) latent space and used to predict attractiveness scores on the held-out faces. At no point were the attractiveness ratings or latent representations of held-out identities used to estimate the alignment or direction. When the source and target were the same model, i.e. no cross-model transfer, the baseline correlation with attractiveness was $r = 0 . 5 3 4 \pm . 0 1 2$ across the 12 models. Replacing the source representation by an independently initialized model trained on the same dataset produced almost no degradation $( r = 0 . 5 2 9 \pm . 0 1 1 )$ . Most importantly, transfer remained nearly as strong when source and target models were trained on diferent face distributions with mean held-out correlation $r = 0 . 5 1 6 \pm . 0 1 7$ . In other words, the human attractiveness dimension identified in one learned representation can be transported to another representation with little loss of predictive information. This suggests that human attractiveness judgments correspond to a reproducible direction embedded in the higher-dimensional geometry learned by generative models of faces.

## 4.3. Attractive faces have prototypical shapes and latent codes

We next inquired whether attractiveness could be explained by a simpler account based on the prototypicality of the faces. Classic theories of face perception have emphasized that faces closer to a population prototype are often judged as more attractive. We first quantified classical shape prototypicality from the 468 facial landmarks detected by MediaPipe. Each face was compared with a leave-one-out prototype computed from the remaining members of the same demographic group. Shape prototypicality was defined as the negative Procrustes distance to this within-group prototype, such that larger values indicate more prototypical facial shape. We found that more prototypical faces were rated as more attractive $( r = 0 . 2 6 8 , p < 0 . 0 0 1 )$ . This replicated the classical averageness efect (Winkielman et al., 2006; Ryali et al., 2020) within the CFD faces.

We then asked whether the same relationship emerges in the latent representations learned by the VAEs. For each model $m ,$ latent prototypicality was defined as the negative Euclidean distance between the mean posterior representation $\mu _ { \phi _ { m } } ( x )$ and the leave-one-out centroid of the corresponding demographic group. Attractiveness was positively associated with latent prototypicality in every model, with Pearson correlations of $r = 0 . 3 1 5 \pm . 0 0 2$ for CelebA, $0 . 2 9 4 \pm . 0 1 3$ for FairFace, $0 . 2 9 4 \pm . 0 0 4$ for FFHQ, and $0 . 2 8 1 \pm . 0 1 2$ for UTKFace. Thus, the tendency for attractive faces to lie closer to the within-group prototypes is reproduced not only in explicit landmark geometry but also in independently learned generative representations. The two notions of prototypicality were related but not redundant. Across models, the correlation between landmark-space and latent-space prototypicality ranged from $r = 0 . 4 5 5$ to 0.544. Hence, the learned representations preserve a substantial component of classical facial averageness, while also reorganizing the faces according to additional image structure not captured by the landmark prototype alone (e.g., skin tone or eye color).

Finally, we found that the projections onto the latent attractiveness directions identified in Sec. 4.2 were substantially more predictive of attractiveness than either prototype measure (see Fig. 3). Across training datasets, the latent direction alone predicted held-out attractiveness with correlations of $r = 0 . 5 2 6$ to 0.541. The full model containing shape prototypicality, latent prototypicality, and the latent attractiveness direction reached only moderately higher correlations $( r = 0 . 5 4 6$ to 0.570). In sum, most of the predictable attractiveness structure captured by these representations lies along the learned latent direction rather than being reducible to either notion of prototypicality.

## 5. Discussion

This work investigated whether human judgments of facial attractiveness are reflected in the structure of independently learned variational representations of faces. Across four training datasets, our results converge on a set of common findings. First, attractiveness increases along the direction of higher processing fluency: attractive faces occupy regions characterized by lower reconstruction distortion and lower regularization cost. Second, the relationship extends beyond the scalar ELBO objective. After aligning independently learned latent spaces, attractiveness defines a direction that transfers across models and training datasets with little loss of predictive information. Third, attractive faces are more prototypical in both landmark and latent space, but this prototype efect accounts for only part of the latent attractiveness geometry. Together, these findings suggest that classical findings of facial attractiveness can be understood as manifestations of a representational organization resulting from unsupervised learning of a generic generative model.

The two components of the ELBO, rate and distortion, could hint to two distinct sources for processing fluency. The distortion is a measure of the fidelity with which a stimulus is represented whereas the rate quantifies the complexity of that representation. We found that both components correlate with attractiveness, although it is their trade-of in the ELBO that is best aligned with attractiveness ratings. Our variational account can thus integrate two complementary interpretations of processing fluency, identified by Brielmann and Dayan (2022) as the a priori and a posteriori signals. In this view, a low rate indicates that the representation remains close to the model’s expectations and a low distortion reflects that the stimulus was correctly processed. However, this correspondence remains preliminary and will require empirical validation to disentangle the two sources of processing fluency.

While our results support the hypothesis that faces are represented with ease when they can be reconstructed accurately while deviating minimally from the latent prior, this interpretation should not be taken to imply that human observers explicitly optimize an ELBO. Instead, the ELBO provides a computationally explicit measure of processing fluency in a learned generative representation. The most surprising result is that a direction determined entirely by the model’s unsupervised objective is closely aligned with human preference. In Appendix B we report that objective facial properties (such as the distance between the eyes) predicted both human attractiveness and VAE ELBO with remarkable similarity $( r = 0 . 5 0 9 $ and 0.499, respectively), whereas social and afective judgments were much more predictive of human attractiveness $( r = 0 . 7 0 4 )$ than of VAE ELBO $( r = 0 . 2 6 2 )$ Therefore, human aesthetic judgments can only partially be captured by an unsupervised image model trained passively on face images.

Several limitations constrain the conclusions of the present study. The VAEs considered here are deliberately simple generative models and should not be interpreted as mechanistic models of the human visual system. The CFD dataset contains frontal neutral-expression images, which makes it well suited to isolate facial structure but limits its generalization to naturalistic viewing conditions, pose variability, and dynamic face expressions. Our demographic conditioning reduces obvious within-group confounding efects but relies on coarse dataset categories and does not capture the full diversity of individual experience or cultural preference. The attractiveness ratings are also population averages and do not capture the variability of individual diferences. Finally, it must be noted that all reported relationships are correlational and not causal. Our results can only establish a correspondence between human preference and learned representational geometry. Nevertheless, this correspondence provides an empirical computational validation of the variational interpretation of processing fluency in face perception, linking human preference to the structure of generative representations.

## References

Marco Bertamini, Giulia Rampone, Alexis DJ Makin, and Andrew Jessop. Symmetry preference in shapes, faces, flowers and landscapes. PeerJ, 7:e7078, 2019.

Aenne A Brielmann and Peter Dayan. A computational model of aesthetic value. Psychological review, 129(6):1319, 2022.

Aenne A Brielmann, Max Berentelg, and Peter Dayan. Modelling individual aesthetic judgements over time. Philosophical Transactions of the Royal Society B: Biological Sciences, 379(1895):20220414, 2023.

A Mike Burton, Rob Jenkins, Peter JB Hancock, and David White. Robust representations for face recognition: The power of averages. Cognitive psychology, 51(3):256–284, 2005.

Devin Duke, Chris M Fiacconi, and Stefan K¨ohler. Parallel efects of processing fluency and positive afect on familiarity-based recognition decisions for faces. Frontiers in Psychology, 5:328, 2014.

Jacopo Frascaroli, Helmut Leder, Elvira Brattico, and Sander Van de Cruys. Aesthetics and predictive processing: Grounds and prospects of a fruitful encounter. Philosophical Transactions of the Royal Society B: Biological Sciences, 379(1895):20220410, 2023.

Karl Friston. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2):127–138, 2010.

Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations, 2017.

Rebecca A Hoss, Jennifer L Ramsey, Angela M Grifin, and Judith H Langlois. The role of facial attractiveness and facial masculinity/femininity in sex classification of faces. Perception, 34(12):1459–1474, 2005.

Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1548–1558, 2021.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 4396–4405. IEEE, 2019.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Diederik P Kingma et al. Variational inference & deep learning. A New Synthesis. University of Amsterdam, 2017.

Karel Kleisner, Petr Tureˇcek, S Adil Saribay, Ondˇrej Pavloviˇc, Juan David Leong´omez, S Craig Roberts, Jan Havl´ıˇcek, Jaroslava Varella Valentova, Silviu Apostol, Robert Mbe Akoko, et al. Distinctiveness and femininity, rather than symmetry and masculinity, afect facial attractiveness across the world. Evolution and Human Behavior, 45(1):82–90, 2024.

Judith H Langlois and Lori A Roggman. Attractive faces are only average. Psychological science, 1(2):115–121, 1990.

Pengting Lee, Jingheng Li, Yasaman Rafiee, Benedict C Jones, and Victor KM Shiramizu. Further evidence that averageness and femininity, rather than symmetry and masculinity, predict facial attractiveness judgments. Scientific Reports, 15(1):5498, 2025.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision, pages 3730–3738, 2015.

Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, Chuo-Ling Chang, Ming Yong, Juhyun Lee, et al. Mediapipe: A framework for perceiving and processing reality. In Third workshop on computer vision for AR/VR at IEEE computer vision and pattern recognition (CVPR), volume 2019, 2019.

Francisco M L´opez, Bertram E Shi, and Jochen Triesch. Prioritizing compression explains human perceptual preferences. In Intrinsically-Motivated and Open-Ended Learning Workshop@ NeurIPS2024. NeurIPS, 2024.

Francisco M L´opez, Bertram E Shi, and Jochen Triesch. Eficient coding in active perception: A developmental perspective on autonomous control. Advances in Child Development and Behavior, 70:117–156, 2026.

Debbie S Ma, Joshua Correll, and Bernd Wittenbrink. The chicago face database: A free stimulus set of faces and norming data. Behavior research methods, 47:1122–1135, 2015.

Surabhi S Nath, Franziska Br¨andle, Eric Schulz, Peter Dayan, and Aenne Brielmann. Relating objective complexity, subjective complexity, and beauty in binary pixel patterns. Psychology of Aesthetics, Creativity, and the Arts, 20(1):238, 2026.

David I Perrett, D Michael Burt, Ian S Penton-Voak, Kieran J Lee, Duncan A Rowland, and Rachel Edwards. Symmetry and human facial attractiveness. Evolution and human behavior, 20(5):295–307, 1999.

P Jonathon Phillips and David White. The state of modelling face processing in humans with deep learning. British Journal of Psychology, 117(2):656–676, 2026.

Rolf Reber, Piotr Winkielman, and Norbert Schwarz. Efects of perceptual fluency on afective judgments. Psychological science, 9(1):45–48, 1998.

Rolf Reber, Norbert Schwarz, and Piotr Winkielman. Processing fluency and aesthetic pleasure: Is beauty in the perceiver’s processing experience? Personality and social psychology review, 8(4):364–382, 2004.

Julien P Renoult, Jeanne Bovet, and Michel Raymond. Beauty is in the eficient coding of the beholder. Royal Society Open Science, 3(3):160027, 2016.

Gillian Rhodes, Alex Sumich, and Graham Byatt. Are average facial configurations attractive only because of their symmetry? Psychological science, 10(1):52–58, 1999.

Chaitanya K Ryali, Stanny Gofin, Piotr Winkielman, and Angela J Yu. From likely to likable: The role of statistical typicality in human social assessment of faces. Proceedings of the National Academy of Sciences, 117(47):29371–29380, 2020.

J¨urgen Schmidhuber. Driven by compression progress: A simple principle explains essential aspects of subjective beauty, novelty, surprise, interestingness, attention, curiosity, creativity, art, science, music, jokes. In Workshop on anticipatory behavior in adaptive learning systems, pages 48–76. Springer, 2008.

Logan T Trujillo, Jessica M Jankowitsch, and Judith H Langlois. Beauty is in the ease of the beholding: A neurophysiological test of the averageness theory of facial attractiveness. Cognitive, Afective, & Behavioral Neuroscience, 14:1061–1076, 2014.

Sander Van de Cruys, Jacopo Frascaroli, and Karl Friston. Order and change in art: towards an active inference account of aesthetic experience. Philosophical Transactions of the Royal Society B: Biological Sciences, 379(1895):20220411, 2023.

Hermann von Helmholtz. Handbuch der physiologischen Optik, volume 9. L. Voss, 1867.

Piotr Winkielman, Jamin Halberstadt, Tedra Fazendeiro, and Steve Catty. Prototypes are attractive because they are easy on the mind. Psychological science, 17(9):799–806, 2006.

Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/regression by conditional adversarial autoencoder. In 2017 IEEE conference on computer vision and pattern recognition (CVPR), pages 4352–4360. IEEE, 2017.

## Appendix A. Extended methods

## A.1. Dataset details

## A.1.1. Training data

We trained face representations from four datasets with diferent demographic distributions and image statistics. Importantly, all available labels and metadata were ignored during training. The models learned face representations fully unsupervised.

CelebA (Liu et al., 2015) contains 202,599 face images (163,456 retained after preprocessing) of celebrity identities collected from the internet. This was the largest dataset used in our experiments, and its interest is that it provides substantial variation in pose, appearance, and background. However, CelebA was not constructed to be demographically balanced, and must thus be seen as a large in-the-wild distribution of faces with implicit biases.

FairFace (Karkkainen and Joo, 2021) contains 108,501 face images (52,801 retained after preprocessing), collected primarily from Flickr and subsequently annotated for race, gender, and age. It was explicitly constructed to reduce the strong racial imbalance present in CelebA and other popular face datasets. FairFace distinguishes seven race categories (Black, East Asian, Indian, Latino, Middle Eastern, Southeast Asian, and White), two gender categories, and nine age ranges. These annotations were not used during training.

FFHQ (Karras et al., 2019) contains 70,000 high-quality 1024 × 1024 face images (57,741 retained after preprocessing) collected from Flickr. The dataset was designed for generative face modeling and deliberately includes broad variation in age, ethnicity, and background. FFHQ provides high-quality images beyond the ranges of the other datasets, although this property is mostly lost since we resized the images to 224 × 224 pixels.

UTKFace (Zhang et al., 2017) contains more than 20,000 internet face images (18,492 after preprocessing) covering a large range of identities. It is particularly interesting because it covers ages from 0 to 116 years with five ethnicity categories (Asian, Black, Indian, White, and Other). In line with the other training datasets, the annotations were ignored during training.

## A.1.2. Evaluation data

Chicago Face Database (CFD) (Ma et al., 2015) contains standardized face images from 597 individuals. The participants self-identified as Asian, Black, Latino, or White and as Female or Male, yielding eight ethnicity×gender groups. The dataset contains multiple expressions but we used exclusively the neutral-expression images. All 597 images were retained after preprocessing. CFD provides both objective measurements of facial morphology and subjective ratings, including attractiveness, collected from an average of 25 participants per image on a scale between 1 and 7.

No CFD images or ratings were used during the training of the VAEs or for hyperparameter selection. Facial statistics and attractiveness ratings were found to difer between demographics, particularly with Female faces yielding significantly higher scores. To standardize our analysis, every scalar quantity y<sub>i</sub> (e.g. the attractiveness rating) was transformed as

![](images/8a416ca17e1acd5bb65c27c5fecd16e14782e150123fb52a92c18ed80b8c4b81.jpg)  
(a)

![](images/8cbf275e533e41ee81e511c1afe0b7c0b73e68f94f5bd5d1ea6f64d9bf40d1e5.jpg)  
(b)

![](images/9fc129cf05a6e97213e62496c8aedfd24ce5f8dce04bee31d79010a7192dc9b2.jpg)  
(c)

![](images/d83255b5c7165e487f950a70d2e51e39da92e11fef1c26fe51a02f7cef16297b.jpg)  
(d)  
Figure 4: Example of face preprocessing pipeline with an image from the FFHQ dataset. (a) Original image. (b) Facial geometry detected by MediaPipe Face Landmarker. (c) Image after translation, rotation, and uniform scaling. (d) Final 224 × 224 input after masking the non-face region.

$$
y _ { i } ^ { ( z ) } = \frac { y _ { i } - \mu _ { g ( i ) } } { \sigma _ { g ( i ) } } ,\tag{4}
$$

where $\mu _ { g ( i ) }$ and $\sigma _ { g ( i ) }$ are the mean and standard deviation among CFD faces belonging to the same demographic group g(i) as face i.

## A.2. Face preprocessing

All five image datasets were preprocessed with the same pipeline before training or evaluation. First, we detected facial geometry using MediaPipe Face Landmarker (Lugaresi et al., 2019), which provides a dense set of 468 facial landmarks. Images for which the eyes, mouth, or complete face outline could not be localized were rejected. Additionally, images were the face was not facing forward were also rejected. For each accepted image, we applied a transformation containing translation, rotation, and uniform scaling, such that the midpoint between the two eyes was located at (0.50W, 0.35H) and the mouth center at (0.50W, 0.68H) in the final coordinate frame. No warping or anisotropic scaling was used. This procedure standardized global face positions size while preserving individual facial morphology, e.g. the relative separation of the eyes with respect to the distance between the eyes and the mouth was maintained. After the alignment, the facial region was detected with the MediaPipe face-oval contour, and all pixels outside the resulting facial mask were replaced by the constant RGB value (128, 128, 128). The final images were stored as 224 × 224 RGB arrays. Fig. 4 illustrates the complete procedure.

## A.3. VAE architecture

All experiments used the same convolutional VAE architecture (see Table 1). The model received a 224 × 224 RGB image $x \in [ 0 , 1 ] ^ { 3 \times 2 2 4 \times 2 2 4 }$ and encoded it through five convolutional layers. The final $5 1 2 \times 7 \times 7$ tensor was flattened to a 25,088-dimensional vector and projected independently onto the mean $\mu _ { \phi } ( x )$ and log-variance log $\sigma _ { \phi } ^ { 2 } ( x )$ of a diagonal Gaussian approximate posterior. The baseline latent dimensionality was $d _ { z } = 1 2 8$ and the prior was a standard Gaussian $p ( z ) = \mathcal { N } ( 0 , I )$ . The decoder mirrored the encoder. The latent samples were obtained with the reparameterization trick (Kingma and Welling, 2013):

$$
\begin{array} { r } { z = \mu _ { \phi } ( x ) + \sigma _ { \phi } ( x ) \odot \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{5}
$$

Table 1: Architecture of the convolutional VAE used in our experiments. All convolutions and transposed convolutions used kernel size 4, stride 2, and padding 1. The baseline latent dimensionality was $d _ { z } = 1 2 8$
<table><tr><td>Stage</td><td>Layer</td><td>Output shape</td><td>Activation</td></tr><tr><td>Input</td><td></td><td> $3 \times 2 2 4 \times 2 2 4$ </td><td></td></tr><tr><td>Encoder 1</td><td> $\mathrm { C o n v 2 d 3 }  3 2$ </td><td> $3 2 \times 1 1 2 \times 1 1 2$ </td><td>ReLU</td></tr><tr><td>Encoder 2</td><td> $\mathrm { C o n v 2 d ~ 3 2 }  6 4$ </td><td> $6 4 \times 5 6 \times 5 6$ </td><td>ReLU</td></tr><tr><td>Encoder 3</td><td> $\mathrm { C o n v 2 d 6 4 \to 1 2 8 }$ </td><td> $1 2 8 \times 2 8 \times 2 8$ </td><td>ReLU</td></tr><tr><td>Encoder 4</td><td> $\mathrm { C o n v 2 d ~ 1 2 8  2 5 6 }$ </td><td> $2 5 6 \times 1 4 \times 1 4$ </td><td>ReLU</td></tr><tr><td>Encoder 5</td><td> $\mathrm { C o n v 2 d ~ 2 5 6 \to 5 1 2 }$ </td><td> $5 1 2 \times 7 \times 7$ </td><td>ReLU</td></tr><tr><td>Flatten</td><td></td><td>25,088</td><td></td></tr><tr><td>Posterior mean</td><td>Linear 25,088 → 128</td><td>128</td><td></td></tr><tr><td>Posterior log-var.</td><td>Linear 25,088 → 128</td><td>128</td><td></td></tr><tr><td>Latent sample</td><td>Reparameterization</td><td>128</td><td></td></tr><tr><td>Projection</td><td>Linear  $1 2 8 \to 2 5 { , } 0 8 8$ </td><td> $5 1 2 \times 7 \times 7$ </td><td></td></tr><tr><td>Decoder 1</td><td>ConvTranspose2d 512 → 256</td><td> $2 5 6 \times 1 4 \times 1 4$ </td><td>ReLU</td></tr><tr><td>Decoder 2</td><td>ConvTranspose2d  $2 5 6 \to 1 2 8$ </td><td> $1 2 8 \times 2 8 \times 2 8$ </td><td>ReLU</td></tr><tr><td>Decoder 3</td><td>ConvTranspose2d  $1 2 8 \to 6 4$ </td><td> $6 4 \times 5 6 \times 5 6$ </td><td>ReLU</td></tr><tr><td>Decoder 4</td><td>ConvTranspose2d  $6 4  3 2$ </td><td> $3 2 \times 1 1 2 \times 1 1 2$ </td><td>ReLU</td></tr><tr><td>Decoder 5</td><td>ConvTranspose2d  $3 2  3$ </td><td> $3 \times 2 2 4 \times 2 2 4$ </td><td>Sigmoid</td></tr></table>

Unless otherwise specified, all experiments used $d _ { z } = 1 2 8 , \beta = 1$ , and $\sigma _ { x } = 1$ . For each of the four datasets, three models were trained from independent random initializations, yielding the 12 baseline representations analyzed throughout the main experiments. Additional experiments with the FairFace dataset retained the same architecture while varying latent dimensionality, regularization trade-of $\beta ,$ or decoder scale $\sigma ,$ as described in Sec. 4.1.

## A.4. Rate-distortion evaluation

For each trained model and each CFD face $x _ { i } ,$ the encoder produced a posterior distribution $q _ { \phi } ( z \mid x _ { i } ) = \mathcal { N } ( \mu _ { i } , \mathrm { d i a g } ( \sigma _ { i } ^ { 2 } ) )$ . The rate was computed analytically as

$$
R _ { i } = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { z } } \left( \mu _ { i j } ^ { 2 } + \sigma _ { i j } ^ { 2 } - \log { \sigma _ { i j } ^ { 2 } } - 1 \right) .\tag{6}
$$

In parallel, the distortion was approximated by the Monte Carlo estimate estimated as

$$
D _ { i } = \mathbb { E } _ { q _ { \phi } ( z | x _ { i } ) } \left[ \frac { 1 } { 2 \sigma _ { x } ^ { 2 } } \lVert x _ { i } - f _ { \theta } ( z ) \rVert _ { 2 } ^ { 2 } \right] ,\tag{7}
$$

using 16 independent posterior samples.

## A.5. ELBO in rate-distortion space

Our first experiment (Sec. 4.1) tested whether attractiveness is organized along the ratedistortion trade-of induced by Eq. (1). For every model, we computed the distortion $D _ { i }$ and the rate $R _ { i }$ for all CFD faces. To test the prediction that attractiveness follows the direction selected by the variational objective, we standardized the within-group residual distortion and rate coordinates,

$$
\bar { D } _ { i } = \frac { D _ { i } - \mathbb { E } [ D _ { i } \mid g _ { i } ] } { s _ { D } } , \qquad \bar { R } _ { i } = \frac { R _ { i } - \mathbb { E } [ R _ { i } \mid g _ { i } ] } { s _ { R } } ,\tag{8}
$$

where $s _ { D }$ and $s _ { R }$ denote the residual standard deviations, so that the direction of the ELBO in standardized rate-distortion coordinates is given by

$$
\mathbf { e } \propto ( - s _ { D } , - s _ { R } ) .\tag{9}
$$

We also estimated the empirical attractiveness gradient with the linear model

$$
\tilde { A } = a _ { D } \bar { D } + a _ { R } \bar { R } + \epsilon ,\tag{10}
$$

from which we define the attractiveness direction

$$
\mathbf { a } _ { A } = ( a _ { D } , a _ { R } ) .\tag{11}
$$

Finally, the agreement between the empirical attractiveness direction and the ELBO was measured as the cosine similarity between both vectors.

## A.6. Alignment and transfer of latent attractiveness geometry

Our second analysis asked whether attractiveness is associated with a common direction in the latent representations learned by independent VAEs. For model m, we represented every CFD face by its posterior mean

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { ( m ) } = \mu _ { \phi _ { m } } ( x _ { i } ) \in \mathbb { R } ^ { 1 2 8 } . } \end{array}\tag{12}
$$

Within each ethnicity×gender group we subtracted the group-specific latent centroid. First, we aligned the 12 representations using generalized orthogonal Procrustes analysis. Specifically, for centered representation matrices $Z _ { m }$ , we estimated orthogonal transformations $Q _ { m }$ with respect to a common template M (also optimized) by minimizing a Procrustes objective

$$
\mathcal { L } = \sum _ { m } \left\| Z _ { m } Q _ { m } - M \right\| ^ { 2 } , \qquad \mathrm { s u b j e c t } \mathrm { t o } Q _ { m } ^ { \top } Q _ { m } = I .\tag{13}
$$

After alignment, we estimated the linear attractiveness direction and measured the similarity between every pair of model directions via their cosine similarity. To compare with a “default alignment baseline”, we constructed a within-group permutation where the attractiveness ratings were shufled relative to the identities. We repeated the procedure with 1000 permutations.

## A.7. Prototypicality and geometric representability

Our third analysis tested whether the attractiveness efects above could be explained from the likeness of each face to its within-group prototype. We considered prototypicality independently in landmark and latent space. First, each face was represented by the 468 MediaPipe facial landmarks (see Fig. 4). The landmarks were aligned by with a two-dimensional orthogonal Procrustes transformation. We subsequently computed the leave-one-out withingroup prototype $\bar { S } _ { g , - i }$ of identify i. Prototypicality was defined as the negative Procrustes distance to the prototype. Additionally, for each trained VAE, latent prototypicality was analogously defined from posterior means as the Euclidian distance in latent space to the leave-one-out centroid of the corresponding group. We also measured the relationship between shape and latent prototypicality as their linear correlation to determine how closely the learned notion of latent prototypicality corresponded to classical facial shape averageness.

We also quantified how faithfully each VAE preserved facial geometry. For each CFD face, we decoded the posterior mean, $\hat { x } _ { i } = f _ { \theta } ( \mu _ { \phi } ( x _ { i } ) )$ , and extracted its facial landmarks using MediaPipe on the reconstruction ${ \hat { x } } _ { i }$ . We then computed the Procrustes distance between the original landmarks and the reconstructed landmarks. Higher distances indicate indicate less faithful preservation of facial structure.

Finally, we tested whether the latent attractiveness direction carries predictive information beyond prototypicality. We compared linear models based on shape prototypicality, latent prototypicality, latent attractiveness-direction, and their combinations. All prototypes, demographic centroids, normalization parameters, and attractiveness directions were estimated from training identities only. We computed the correlation between held-out predictions and actual attractiveness ratings.

## A.8. Statistics

Unless otherwise stated, values reported across models are means and standard deviations across the three independent seeds for each training distribution. All within-group demographic standardization used only the ethnicity×gender labels supplied by CFD. For descriptive analyses (correlations, alignment), statistics were computed from the complete evaluation set. For the predictive cross-validation analyses all the values were estimated exclusively from training identities and subsequently applied without any changes to them held-out identities, on which the results and statistics were computed.

## Appendix B. Supplementary results

## B.1. Robustness across hyperparameters

Using the FairFace dataset, we tested whether the correlation between attractiveness and VAE ELBO and the alignment of their directions in rate-distortion space was conserved across diferent sets of hyperparameters. We varied one of the default model parameters $( d _ { z } = 1 2 8 , \ : \beta = 1 , \ : \sigma = 1 )$ at a time while holding all other components fixed across the ranges $d _ { z } \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ $\beta \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ , and $\sigma \in \{ 0 . 5 , 0 . 7 0 7 , 1 , 1 . 4 1 4 \}$ , all with three random seeds. The results are shown in Fig. 5. In all individual models, the correlation between human ratings and ELBO was significant, yielding Pearson correlations $r = 0 . 1 9 0$ to $r = 0 . 2 6 2 .$ , all $p < 0 . 0 0 1$ . The lowest average correlation was observed for $\beta = 4$ with $r = 0 . 1 9 9 \pm 0 . 0 1 5$ , possibly due to an over-regularization of the latent space where the face reconstructions are too degraded, although the correspondence between attractiveness and processing fluency was retained. The lowest alignment between the directions of the attractiveness and ELBO vectors in rate-distortion space was observed for $\sigma = \sqrt { 2 }$ whereas the highest value was $0 . 9 9 8 \pm 0 . 0 0 3$ for $\sigma = 1 / \sqrt { 2 }$ , where the two vectors were virtually collinear.

![](images/4d2e9dfcf062abf747b7f721a3fe4c9257a0d9c507fd66b4e76bc837c7d5ee89.jpg)  
Figure 5: Robustness of attractiveness-ELBO alignment to diferent hyperparameters. (A) Pearson correlations attractiveness ratings and VAE ELBO. (B) Cosine similarity between attractiveness and ELBO directions in rate-distortion space.

## B.2. Robustness across demographic groups

To validate whether the relationship between attractiveness and processing fluency was consistent in all demographics (ethnicity×gender), we independently computed the Pearson correlations between the human ratings and the ELBO, negative distortion, and negative rate within group for each of the 12 baseline models. In total, 96 comparisons per metric were computed. We found that the correlation between ELBO and attractiveness was positive in all 96 comparisons, with mean correlation $r = 0 . 2 8 6 \pm 0 . 0 9 7$ and individual values ranging between $r = 0 . 1 1 5$ and 0.457. Likewise, the negative distortion was positively correlated with attractiveness across all 96 comparisons $\left( r = 0 . 1 1 3 \mathrm { ~ t o ~ } 0 . 4 3 1 \right)$ . On the other hand, the negative rate was positively correlated in 94 of the 96 comparisons, with the lowest efect being virtually zero: $r = - 0 . 0 2 2$ . Interestingly, our results show considerable diferences among diferent demographics. The highest within-group correlations between attractiveness and ELBO were achieved for the Latino-Male $( r = 0 . 4 3 1 )$ and White-Female $( r = 0 . 4 0 8 )$ groups, whereas the lowest correlations were achieved for the Latino-Female $( r = 0 . 1 7 1 )$ and White-Male $( r = 0 . 1 7 8 )$ groups. We speculate that these diferences are more reflective of training and evaluation data imbalances than of objective properties of these particular demographics. Nevertheless, further experiments are required to validate this hypothesis.

![](images/294f78d6f2b1a289d21b5eed0309714efc83923a23915e94509cea3bbefe5260.jpg)  
Figure 6: Robustness of correlations across demographic groups. Each column represents a seed within a training dataset (see Fig. 3).

## B.3. Preserved facial geometry

We tested whether attractive facial geometry is represented more faithfully by the generative model, that is whether the VAEs preserve the shape of attractive faces more than unattractive faces in their reconstructions. For each reconstructed face, we extracted the same facial landmarks and measured the Procrustes discrepancy between the original and reconstructed shape. We found that landmark preservation was moderately associated with attractiveness in all 12 models, with correlations of $r = 0 . 1 3 5 \pm . 0 0 5$ for CelebA, $0 . 1 3 8 \pm . 0 1 5$ for FairFace, $0 . 1 5 2 \pm . 0 0 7$ for FFHQ, and $0 . 1 5 8 \pm . 0 0 8$ for UTKFace. Attractive faces therefore exhibited greater latent prototypicality and more faithful reconstruction of the facial geometry, although the latter association was moderate.

## B.4. Objective and subjective facial properties

Finally, we investigated how diferent facial properties correspond to human attractiveness judgments and the VAE’s processing fluency. The face properties were split into two separate categories. On the one hand, we considered objective geometric measurements derived from facial landmarks, such as the distance between the eyes (interocular width) or the aforementioned shape prototypicality. On the other hand, we evaluated other social of afective ratings included in the CFD: trustworthiness, happiness, femininity, disgust,

![](images/79904729b0dbaa23ca40a47ea6d68805f209e0a9f2dbcc06c8ec0705b655a336.jpg)

![](images/1a909fff5e349d65274be1ff73f483687aa4271decb66d06c29d2f66f1467c08.jpg)  
Figure 7: Correlations of human attractiveness judgments (gray) and VAE ELBO scores (green) against objective and subjective face properties.

etc. These judgments were reported by the same participants that judged attractiveness.   
Consequently, the two subjective judgments were expected to be tightly associated.

The objective facial properties showed substantial agreement between human and VAE ratings. The correlations between the two correlation profiles shown in Fig. 7 was $r = 0 . 7 6 0$ with some particular properties being closely matched. For example, the upper-face ratio correlated positively with both scores $( r = 0 . 2 9 7$ for human and $r \ : = \ : 0 . 3 0 6$ for ELBO). However, other geometric features deviated considerably, showing that attractiveness cannot be reduced to objective properties alone. As for the subjective judgments, the disagreement between data and model was larger. Faces were considered more attractive when they were more trustworthy, feminine, and happy, and less attractive when they were more threatening and sad. Interestingly enough, the VAE is able to show similar trends, albeit in a moderate efect. For example, trustworthiness had a Pearson correlation of $r = 0 . 5 5 9$ with attractiveness but only $r = 0 . 2 0 9$ with ELBO.

To further compare objective and subjective properties, we performed a ridge-regressions for each separate and for their combination and evaluated them with out-of-fold predictions. Objective properties predicted attractiveness and ELBO with comparable accuracy $( r =$ 0.509 and 0.499, respectively). However, subjective properties where considerably better for human attractiveness $( r = 0 . 7 0 4 )$ than for ELBO $( r = 0 . 2 6 2 )$ . The combination of all properties increased accuracies to $r = 0 . 7 6 5$ for human ratings and 0.523 for ELBO.

In sum, these results support our variational account of processing fluency insofar as the VAEs can capture the statistical regularities of the faces, whereas social and afective judgments remain unavailable to models trained with unsupervised learning. However, the moderate similarities between human ratings and VAE ELBO for subjective judgments points to a possible extension of this work beyond attractiveness and into other evaluations.

## Appendix C. Examples of face reconstructions

![](images/4e39e048dfc30ff012d94a134d6a7304f26383ebaa50d742e1bd3259fc2941e4.jpg)  
Figure 8: Reconstructions of Asian Female (top) and Male (bottom) faces. Rows: (1) original CFD images, (2-5) Reconstructions from VAE trained on (2) FairFace, (3) FFHQ, (4) CelebA, (5) UTKFace. See Fig. 1 for details.

![](images/23f2159f79682c21bd2a80fdc013b9a49d89990d11d32d4e432fc676f88ab2e0.jpg)  
Figure 9: Reconstructions of Black Female (top) and Male (bottom) faces. Rows: (1) original CFD images, (2-5) Reconstructions from VAE trained on (2) FairFace, (3) FFHQ, (4) CelebA, (5) UTKFace. See Fig. 1 for details.

![](images/1164edef91aa1f08c8748a75045063fb453fb551f6e8192d5f5ecd26107544c8.jpg)  
Figure 10: Reconstructions of Latino Female (top) and Male (bottom) faces. Rows: (1) original CFD images, (2-5) Reconstructions from VAE trained on (2) FairFace, (3) FFHQ, (4) CelebA, (5) UTKFace. See Fig. 1 for details.

![](images/43935ca12946b7d8c41b8cd1a8a8c2036ae2e2dbb7d7703219eabe21b40bd0fc.jpg)  
Figure 11: Reconstructions of White Female (top) and Male (bottom) faces. Rows: (1) original CFD images, (2-5) Reconstructions from VAE trained on (2) FairFace, (3) FFHQ, (4) CelebA, (5) UTKFace. See Fig. 1 for details.