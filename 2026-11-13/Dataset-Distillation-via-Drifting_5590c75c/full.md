# Dataset Distillation via Drifting

George Cazenavette

Angelina Quan

Giannis Daras

Antonio Torralba

Vincent Sitzmann

Massachusetts Institute of Technology

drifting-distillation.github.io

![](images/f3477f7190895506aad62b861b138d98e9fed7b9e00a3f16f74cb78a1fed6c63.jpg)  
Figure 1: A selection of 6 classes from ImageNet-1k [12] distilled to 10 images per class via drifting with DINOv3 [53] features. Images seen here were initialized with random noise. Our method yields synthetic data that are high in both quality and diversity by virtue of the attractive and repulsive forces of our drifting loss.

## Abstract

The task of dataset distillation concerns the learning of a small set of synthetic samples that can serve as an effective replacement for a large real dataset when used for downstream tasks. While many solutions have been proposed since the problem’s inception, most of them primarily seek to optimize the distilled data with respect to a classifier. In this work, we propose a new distillation method that learns a set of samples such that they directly approximate the per-class data distribution. Specifically, we “drift” our distilled data in the latent space of a feature extractor such that they move both toward the real data and away from each other without using a classification objective. We find that our drifted data cover the real distribution significantly better than existing methods while still serving as powerful training samples for classification. Furthermore, we showcase our method’s use as a visualization tool for better interpreting models and extend it beyond images to other modalities. Our work advances the frontier of dataset distillation, offering a novel synthesis method and proposing distribution-based metrics as a complementary evaluation. More broadly, our findings help address the growing need to understand both data and models by uncovering more about how they interact.

## 1 Introduction

Dataset distillation takes the classical problem of coreset selection [25] (choosing a representative subset of a large dataset) and extends it by allowing for the learning of a set of optimal samples rather than restricting to a selection of real data points. Since the task’s proposal in the titular paper [65], dataset distillation solutions have focused on the optimization of synthetic samples for the purpose of training image classification models. Be it via gradient- [74, 76], trajectory- [4, 10, 24], or featurematching [35, 62, 75] or directly maximizing a classifier’s confidence [8, 9, 36, 49, 56, 71, 72, 78], distillation methods have traditionally indexed heavily toward the classification objective, often paying less attention to other potentially desirable attributes of a tiny synthetic dataset. Data distilled with respect to a decision boundary inherit the classifier’s biases, possibly limiting the data’s usability for other tasks or generalization to other models.

In this work, we aim to distill a dataset with the explicit goal of best approximating the real data distribution. Inspired by recent work in generative modeling [13], we learn our synthetic data via “drifting” such that each synthetic sample is attracted toward each real sample while simultaneously being repelled from every other synthetic sample. The competing positive and negative forces encourage the distilled data to spread across the real distribution in a way that promotes both coverage and diversity. As a handful of images alone cannot adequately approximate the real distribution, we instead drift many augmentations of each synthetic image at a time such that the distilled images, together with their augmentations, best cover the target distribution (Figure 2).

We find that this drifting method yields qualitatively more diverse and higher-quality images than existing methods (Figures 1 and 4). Despite being distilled without a classification objective, our images still serve as superior data for training such classifiers from scratch: the traditional dataset distillation benchmark (Sec. 4.1). Using MoCo-v3 [7] for the drifting feature space, our method reaches over 40% and 52% test accuracy when training a ResNet-18 [26] from scratch with fixed soft labels after distilling ImageNet-1k [12] down to just 10 and 50 images per class, respectively.

We further evaluate the distilled data on a set of feature-space metrics (coverage [41] along with nearest-neighbor and linear probe classification) where our method continues to match or outperform existing methods (Sec. 4.2). Our method also offers some model interpretability insights (Sec. 4.3), generalizes to other modalities (Sec. 4.4), and even serves as an avenue for art (Sec. 4.5).

Code and distilled datasets are available on our project page: drifting-distillation.github.io.

## 2 Related Work

Dataset Distillation. The concept of coreset selection [25], or finding a representative subset of a large dataset, has been thoroughly researched over many decades [2, 20, 37, 44, 46, 48, 57, 66, 69]. The idea of training a model to competitive performance using only a fraction of the data is attractive for many reasons, the most promising being the potential efficiency gains. More recently, dataset distillation [65], the learning of a small, representative dataset, has gained favor, as the direct optimization of a set of data points allows for more expressive and content-rich examples than being constrained to a selection of real samples.

For several years, gradient- and trajectory-matching methods [3, 4, 10, 24, 65, 74, 76] yielded the best performance on classification-based dataset distillation tasks. However, these expensive bi-level optimization methods fail to scale to realistic models (ResNet [26], ViT [16], etc.) and datasets (e.g., ImageNet-1k [12]). In contrast, the relatively cheap, often task-agnostic, distribution-matching methods [62, 75] scale computationally to large models and data but perform poorly at that scale.

Large-scale distillation began with the Squeeze, Recover, and Relabel [72] paradigm, which adopts the DeepInversion [70] objective: a DeepDream-style [39] model inversion that adds BatchNorm-statistic matching to the cross-entropy minimization with respect to a pre-trained classifier. Many follow-up works have since refined this method [9, 36, 49, 50, 71, 73]. These methods were the first to perform competitively on realistic models and datasets, but the distilled data are effective mainly for knowledge distillation [27] rather than from-scratch training. Some methods address the homogeneity issues that come with the classification objective via diversity-promoting regularizers [17, 38, 40, 51].

A related line of work instead selects (and then optionally further optimizes) real image patches by scoring them with a pre-trained classifier [8, 33, 56, 59, 64]. These methods are either restricted to, or heavily biased toward, the empirical data distribution, severely limiting their application to other areas, such as privacy [15] or interpretability [3].

Another emerging area of dataset distillation focuses on leveraging pre-trained (most often diffusionbased) generative models to construct the synthetic data [23, 47, 55, 61, 77]. While previous works made use of generative priors [6] as a form of regularization, these methods leverage diffusion model to effectively draw information-dense samples from the real data manifold.

This work leverages newly proposed ideas from generative modeling to more effectively cover the data distribution. Notably, our method strictly focuses on approximating the real distribution and thus does not require a classification objective. Rather than requiring an extra loss term, diversity in our method arises naturally from the repulsive term of the drifting objective. Our method can also be initialized with random noise and does not make use of a pre-trained generative model.

Drifting. As a recently proposed approach to generative modeling [13], drifting upends the typical paradigm of diffusion [28, 29, 54] and flow-matching [34] models by moving the generated samples toward the target distribution at training time, removing the need for a costly denoising process at inference by natively enabling sampling in a single forward pass. In the simplest terms, each generated sample is simultaneously pulled toward every real sample and pushed away from every other generated sample (weighted by a softmax kernel) such that the distribution of generated samples eventually matches that of the real samples.

We apply drifting to dataset distillation. Rather than back-propagating the gradients of the drifting loss into the weights of a generative model, we directly optimize a discrete set of images such that these images, together with their augmentations, cover the target distribution. Aside from this change in parameterization, our only modification is to jointly drift many augmentations of each image at once, treating each augmentation as a unique image when computing negative distances. The rest of the drifting paradigm remains unchanged.

## 3 Method

In this section, we first formally motivate our distribution matching objective (Sec. 3.1) before defining our drifting loss (Sec. 3.2) and pyramid parameterization (Sec. 3.3).

## 3.1 Distribution Matching

The goal of the original drifting formulation [13] (or most generative modeling frameworks) is to learn a process that maps from one distribution $( p _ { \epsilon } )$ to another $\left( { p _ { \mathrm { d a t a } } } \right)$ . Typically, $p _ { \epsilon }$ is an infinitely sampleable, continuous noise distribution, and either a single- (drifting [13], GAN [22]) or multi-step (diffusion [28, 54], flow matching [34]) model f is learned to map from $p _ { \epsilon }$ to $p _ { \mathrm { d a t a } }$ such that p<sub>data</sub> $\approx f _ { \# } p _ { \epsilon } .$ , where $" f _ { \# } , "$ denotes the pushforward induced by the model.

Rather than a generative model $f ,$ our goal is to directly learn a small set of synthetic samples $\mathbf { x } \in S$ such that the distribution of these samples $( p _ { \mathrm { s y n } } )$ matches that of the real data:

$$
p _ { \mathrm { s y n } } \triangleq \frac { 1 } { | \boldsymbol { S } | } \sum _ { \mathbf { x } \in \boldsymbol { S } } \delta _ { \mathbf { x } } \approx p _ { \mathrm { d a t a } }\tag{1}
$$

where $\delta$ is the Dirac delta distribution. As described, this is not feasible: a handful of synthetic samples alone cannot adequately approximate an entire distribution of real data. Rather than attempting to optimize our synthetic data such that they alone cover the real distribution, we sample from a distribution of augmentations $A \sim P _ { A }$ and optimize such that our samples, together with their augmentations, properly approximate the real distribution:

$$
\mathbb { E } _ { A \sim P _ { A } } [ A _ { \# } p _ { \mathrm { s y n } } ] \approx p _ { \mathrm { d a t a } }\tag{2}
$$

In practice, $P _ { A }$ consists of random resized crops, horizontal flips, and Gaussian noise.

## 3.2 Drifting for Dataset Distillation

To match distributions, we adopt the drifting field of [13]. Rather than the “evolution ≈ denoising” analogy, we view it as a “force” field wherein the synthetic samples are pulled and pushed in various

![](images/6231585900b859d78e8ea6354745131163352c64a7f5519e75f9a3432ab58d2a.jpg)  
Figure 2: Augmentations of Distilled Images Cover the Real Distribution. At each distillation step, we take multiple augmentations of each synthetic sample (left). The synthetic data, together with their augmentations, approximately cover the distribution of real data (right). In this plot, the stars represent the distilled images, the triangles 10 augmentations of each distilled image, and the circles real data. Drifting multiple augmentations of each synthetic sample at each step leads to high intra-image diversity, allowing some distilled images (such as the orange star) to cover many parts of the real distribution at the same time.

directions. That is, for a given sample $\mathbf { x } \in S$ and drifting field V, V(x) gives the total “force” acting on x. This leads to the following loss function where x is “pushed” toward its drifted target x + V(x):

$$
\mathcal { L } = \sum _ { \mathbf { x } _ { i } \in \cal S } \left\| \mathbf { x } _ { i } - \mathrm { s t o p g r a d } \left( \mathbf { x } _ { i } + \mathbf { V } ( \mathbf { x } _ { i } ) \right) \right\| _ { 2 } ^ { 2 } .\tag{3}
$$

The stopgrad operation provides a frozen target for the distilled samples. Rather than the weights of a generative model, this drifting loss is back-propagated directly to our synthetic data. Our drifting field V consists of two components: positive (attractive) and negative (repulsive). For a given synthetic sample $\mathbf { x } _ { i } \in S$ , we have:

$$
\begin{array} { r l } & { { \bf V } ^ { + } ( { \bf x } _ { i } ) : = \frac { 1 } { Z _ { \mathrm { d a t a } } } \mathbb { E } _ { { \bf y } \sim p _ { \mathrm { d a t a } } } \left[ \kappa ( { \bf x } _ { i } , { \bf y } ) ( { \bf y } - { \bf x } _ { i } ) \right] } \\ & { { \bf V } ^ { - } ( { \bf x } _ { i } ) : = \frac { 1 } { Z _ { \mathrm { s y n } } } \frac { 1 } { | S | } \displaystyle \sum _ { j , j \ne i } \kappa ( { \bf x } _ { i } , { \bf x } _ { j } ) ( { \bf x } _ { j } - { \bf x } _ { i } ) } \end{array}\tag{4}
$$

where $Z _ { \mathrm { d a t a } }$ and $Z _ { \mathrm { s y n } }$ are the normalization factors

$$
Z _ { \mathrm { d a t a } } : = \mathbb { E } _ { y \sim p _ { \mathrm { d a t a } } } \left[ \kappa ( \mathbf { x } _ { i } , \mathbf { y } ) \right] , \qquad Z _ { \mathrm { s y n } } : = \frac { 1 } { | \mathcal { S } | } \sum _ { j , j \neq i } \kappa ( \mathbf { x } _ { i } , \mathbf { x } _ { j } )\tag{5}
$$

and κ is the kernel

$$
\kappa ( \mathbf x , \mathbf y ) = \exp \left( - \frac { 1 } { \tau } \| \mathbf x - \mathbf y \| _ { 2 } \right)\tag{6}
$$

with temperature τ. The combined drifting field can be written as

$$
\mathbf { V } ( \mathbf { x } ) : = \mathbf { V } ^ { + } ( \mathbf { x } ) - \mathbf { V } ^ { - } ( \mathbf { x } )\tag{7}
$$

![](images/363670c22cf50e44bc3d921a11167028cc02ede7e380f44c3eb9905dfbfd6cc2.jpg)  
Figure 3: We represent distilled data as a Laplacian pyramid of cascading resolutions. To reconstruct a synthetic sample, each level (top) is bicubically interpolated to the full resolution (middle) before being added together (bottom). This acts as a hyperparameter-free implicit regularizer on the distilled data, preventing overfitting to adversarial patterns often seen in model inversion tasks.

Intuitively, these positive and negative fields can be thought of as the attractive (or repulsive) force between a sample $\mathbf { x } _ { i }$ and all other real (or synthetic) samples weighted by the distances between them via a softmax operation. In short, each synthetic sample is optimized to both be close to real samples and far from other synthetic samples. A given synthetic sample $\mathbf { x } _ { i }$ will reach equilibrium when the total attractive and repulsive forces induced by the other real and synthetic samples cancel each other out, leading to $\mathbf { V } ( \mathbf { x } _ { i } ) = \mathbf { 0 }$

Rather than in pixel space, we compute the drifting field in the latent space of a pre-trained feature extractor, $\phi .$ The choice of ϕ plays a crucial role, as it defines the space in which we want to match the real distribution; a stronger feature extractor may lead to better semantic alignment in image space. Combined with the augmentations $A \sim P _ { A }$ described in Sec. 3.1, we have our full loss function<sup>1</sup>:

$$
\mathcal { L } = \mathbb { E } _ { A \sim P _ { A } } \left[ \sum _ { \mathbf { x } \in S } \| \phi \left( A ( \mathbf { x } ) \right) - \operatorname { s t o p g r a d } \left( \phi \left( A ( \mathbf { x } ) \right) + \mathbf { V } ( \phi \left( A ( \mathbf { x } ) \right) ) \right) \| _ { 2 } ^ { 2 } \right]\tag{8}
$$

In practice, multiple augmentations of the same synthetic sample are also considered negative samples with respect to each other, leading to increased intra-sample diversity as well as the inherent intra-class variety afforded by the drifting paradigm. We visualize our distilled distribution in Figure 2 and include pseudocode of the full process in Algorithms 1 and 2.

## 3.3 Implicit Regularization via Laplacian Pyramids

Inspired by recent work in dataset distillation [3] and model inversion [21], we choose to parameterize our distilled data as a Laplacian pyramid of different resolutions. Rather than naively optimizing the pixels of an image, we instead represent each sample as a cascade of images with progressively lower resolutions: $\rho = \bar { \{ 1 \times 1 , 2 \times 2 , 4 \times 4 , \ldots , R \times R \} }$ where $R \times R$ is the full resolution. Together, these layers form a Laplacian pyramid $P = \{ \mathbf { P } _ { r } | r \in \mathcal { \bar { \rho } } \}$

At each optimization step, the synthetic sample x is first built in the same manner an image is reconstructed from its Laplacian pyramid; each layer of the pyramid is bicubically upsampled to the canonical resolution before all layers are added together:

$$
\mathbf { x } = { \mathrm { s i g m o i d } } \left( \sum _ { r \in \rho } { \mathrm { r e s i z e } } _ { R } \left( \mathbf { P } _ { r } \right) \right)\tag{9}
$$

The sigmoid operation acts as a soft, differentiable clamp to keep the range of the synthetic samples within that of the real data. As with prior work [3], we initialize our pyramid at low resolution and progressively add more layers throughout the distillation process. We also apply a fixed $3 \times 3$ decorrelation transform to the color channels such that the distilled images are biased toward natural colors [42].

Notably, this pyramid technique generalizes effortlessly to both 1D (audio) and 3D (video) data, as seen in Section 4.4. Please see Figure 3 for a visualization of one of our distilled image pyramids.

<table><tr><td rowspan="2">Method</td><td colspan="2">Hard-Label</td><td colspan="2">Soft-Label</td><td colspan="2">Knowledge Distillation</td></tr><tr><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Noise init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SRe²L [72]</td><td> $8 . 1 \pm 0 . 2$ </td><td> $1 0 . 7 \pm 0 . 3$ </td><td> $1 7 . 5 \pm 0 . 2$ </td><td> $2 2 . 9 \pm 0 . 3$ </td><td> $4 0 . 6 \pm 0 . 3$ </td><td> $5 0 . 1 \pm 0 . 1$ </td></tr><tr><td>CDA [71]</td><td> $9 . 5 \pm 0 . 1$ </td><td> $1 6 . 1 \pm 0 . 3$ </td><td> $1 9 . 2 \pm 0 . 2$ </td><td> $2 8 . 8 \pm 0 . 1$ </td><td> $4 1 . 1 \pm 0 . 2$ </td><td> $5 0 . 8 \pm 0 . 1$ </td></tr><tr><td>G-VBSM† [49]</td><td> $1 1 . 7 \pm 0 . 3$ </td><td> $2 4 . 9 \pm 0 . 9$ </td><td> $2 9 . 8 \pm 0 . 2$ </td><td> $4 4 . 9 \pm 0 . 3$ </td><td> $4 3 . 8 \pm 0 . 3$ </td><td> $5 3 . 0 \pm 0 . 3$ </td></tr><tr><td>WMDD [35]</td><td> $1 0 . 2 \pm 0 . 3$ </td><td> $1 1 . 6 \pm 0 . 6$ </td><td> $2 9 . 4 \pm 0 . 5$ </td><td> $4 0 . 6 \pm 0 . 1$ </td><td> $4 3 . 7 \pm 0 . 1$ </td><td> $5 3 . 2 \pm 0 . 2$ </td></tr><tr><td>SC-DD [78]</td><td> $6 . 1 \pm 0 . 2$ </td><td> $1 0 . 2 \pm 0 . 1$ </td><td> $1 9 . 1 \pm 0 . 2$ </td><td> $3 1 . 0 \pm 0 . 3$ </td><td> $4 0 . 8 \pm 0 . 2$ </td><td> $5 1 . 2 \pm 0 . 2$ </td></tr><tr><td>Ours (ResNet-18)</td><td> $1 7 . 6 \pm 0 . 2$ </td><td> $1 8 . 4 \pm 0 . 5$ </td><td> $3 0 . 5 \pm 0 . 8$ </td><td> $4 3 . 1 \pm 0 . 3$ </td><td> $\underline { { 4 4 . 8 } } \pm 0 . 7$ </td><td> $5 5 . 0 \pm 0 . 3$ </td></tr><tr><td>Ours (MoCo-v3)</td><td> $2 7 . 0 \pm 0 . 2$ </td><td> $3 3 . 5 \pm 1 . 1$ </td><td> ${ \bf 4 0 . 3 \pm 0 . 1 }$ </td><td> ${ \pm 2 . 8 \pm 0 . 0 }$ </td><td> ${ \pm } 8 . 5 \pm 0 . 1$ </td><td> ${ \bar { \bf s } } 7 . 2 \pm 0 . 3$ </td></tr><tr><td>Ours (DINOv3)</td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 0 . 2 \pm 0 . 2$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> $5 0 . 7 \pm 0 . 1$ </td><td> $4 5 . 0 \pm 0 . 2$ </td><td> $5 6 . 0 \pm 0 . 2$ </td></tr><tr><td>Real init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RDED [56]</td><td> $2 2 . 1 \pm 0 . 1$ </td><td> $3 3 . 8 \pm 1 . 0$ </td><td> $3 3 . 1 \pm 0 . 2$ </td><td> $4 4 . 0 \pm 0 . 3$ </td><td> $4 6 . 7 \pm 0 . 3$ </td><td> $5 6 . 1 \pm 0 . 1$ </td></tr><tr><td>CUDD [36]</td><td> $1 5 . 2 \pm 0 . 2$ </td><td> $1 9 . 8 \pm 0 . 4$ </td><td> $2 8 . 3 \pm 0 . 1$ </td><td> $4 1 . 1 \pm 0 . 3$ </td><td> $4 6 . 6 \pm 0 . 1$ </td><td> $5 5 . 4 \pm 0 . 1$ </td></tr><tr><td>FADRM [8]</td><td> $1 8 . 7 \pm 0 . 1$ </td><td> $2 5 . 7 \pm 0 . 5$ </td><td> $2 9 . 7 \pm 0 . 1$ </td><td> $4 0 . 0 \pm 0 . 3$ </td><td> $4 8 . 5 \pm 0 . 3$ </td><td> $5 6 . 4 \pm 0 . 1$ </td></tr><tr><td> $\mathrm { F A D R M + ^ { \dag } \left[ 8 \right] }$ </td><td> $2 0 . 5 \pm 0 . 5$ </td><td> $2 6 . 2 \pm 0 . 6$ </td><td> $3 2 . 5 \pm 0 . 2$ </td><td> $4 1 . 8 \pm 0 . 3$ </td><td> ${ \bf 4 9 . 2 \pm 0 . 5 }$ </td><td> $5 6 . 6 \pm 0 . 1$ </td></tr><tr><td> $\mathrm { C V - D D ^ { \dag } \ [ 9 ] }$ </td><td> $1 6 . 7 \pm 0 . 0$ </td><td> $2 1 . 9 \pm 0 . 3$ </td><td> $2 8 . 7 \pm 0 . 3$ </td><td> $3 8 . 3 \pm 0 . 1$ </td><td>45.6 ± 0.1</td><td>54.2 ± 0.1</td></tr><tr><td>Ours (ResNet-18)</td><td> $2 0 . 4 \pm 0 . 4$ </td><td>22.0 ± 0.4</td><td> $3 3 . 9 \pm 0 . 2$ </td><td> $4 5 . 4 \pm 0 . 5$ </td><td>47.8 ± 0.2</td><td>56.4 ± 0.3</td></tr><tr><td>Ours (MoCo-v3)</td><td> $\pm 8 . 9 \pm 0 . 2$ </td><td>36.0 ± 0.7</td><td> ${ \bf 4 1 . 7 \pm 0 . 0 }$ </td><td>53.4 ± 0.1</td><td>49.6 ± 0.2</td><td>57.7 ± 0.2</td></tr><tr><td>Ours (DINOv3)</td><td>26.0 ± 0.2</td><td>41.1 ± 0.4</td><td> $4 0 . 5 \pm 0 . 0$ </td><td>54.3 ± 0.4</td><td>49.5 ± 0.3</td><td>58.1 ± 0.1</td></tr></table>

Table 1: From-Scratch Training. This table shows ImageNet-1k [12] classification performance when training a ResNet-18 [26] from scratch. We include results using one-hot hard labels (left), one-time soft labels (middle), and full knowledge distillation (right). Methods marked with a dagger<sup>†</sup> distill using an ensemble of encoders. Across all three training paradigms, our distilled data lead to the best downstream performance.

## 4 Experiments

As the focus of this work is large-scale dataset distillation, our experimental results center on ImageNet-1k [12]. All distillation experiments are conducted one class at a time, allowing for easy parallelization across many GPUs. We primarily use three backbones for our image distillation experiments: ResNet-18 [26], DINOv3 [53], and MoCo-v3 [7] (ViT-B for both transformers). The specific features to which we apply the drifting loss are detailed in Appendix C. We compare against both noise-initialized methods (SRe<sup>2</sup>L [72], CDA [71], G-VBSM [49], WMDD [35], and SC-DD [78]) and real-initialized methods (RDED [56], CUDD [36], FADRM and FADRM+ [8], and CV-DD [9]). Since much of the improvement among real-initialized methods seems to come from how the initial images are chosen [8, 36, 56], we report the two groups separately and run our own method under both initializations. Noise initialization remains of particular interest, as initializing with real images limits usefulness for some downstream tasks, such as those reliant on privacy [15]. We compare against diffusion-based methods [23, 47, 55, 61, 77] in Appendix Table B.2.

## 4.1 Dataset Distillation via Drifting Yields Superior Training Samples.

While many prior large-scale dataset distillation works bake the classification task into the data by directly minimizing a classifier’s loss [8, 9, 36, 49, 56, 71, 72, 78], our drifting method is solely designed to match the per-class distribution of the real data. In fact, our method does not require a classification objective at all; drifting can occur in the space of any feature extractor, regardless of training objective. Nevertheless, we still find that our synthetic images serve as excellent data for training classifiers.

In Table 1, we train ResNet-18 [26] from scratch using the distilled data from each method in three settings: hard-label, soft-label, and knowledge distillation. Hard-label uses one-hot labels, while soft-label first computes a single fixed target vector for each training sample using a teacher model. Knowledge distillation (the evaluation method used in most prior large-scale distillation works) computes a new target vector at every step by passing the current augmented version of the training image through the teacher model; we include it for comparison with prior work, but note that it is effectively a teacher-student transfer task rather than from-scratch training (see Appendix C). To avoid confounding evaluation factors, metrics for all methods were recomputed using the exact same training recipes (Appendix D). We report the mean and standard deviation over 3 fixed seeds.

![](images/dc33432c21c159dfe468a0515799cf0dae2903eb2c96d84ecc7a199478b3fe7b.jpg)  
Figure 4: Comparing Noise-Initialized Distillation Methods. Even when using ResNet-18 as the distillation backbone, our method yields higher-quality and more diverse samples. When using a stronger backbone (DINOv3 [53]), the visual quality and diversity of our method increase further.

For the hard- and soft-label settings, our method consistently outperforms both the noise- and realinitialized baselines. In the knowledge distillation setting, we still match or exceed the baselines despite the smaller gap (due to the task’s tendency to saturate). The performance difference is most dramatic in the hard-label setting, especially with noise initialization. At just 10 images per class, our best noise-initialized variant (MoCo-v3 [7]) reaches 27.0% test accuracy while the best noiseinitialized baseline (G-VBSM [49]) achieves only 11.7%. With real initialization, our MoCo-v3 data reach 28.9% versus 22.1% for the best baseline (RDED [56]), and at 50 images per class the gaps remain large (33.5% vs. 24.9% with noise initialization and 41.1% vs. 33.8% with real initialization).

Notably, the best training data come from the self-supervised backbones (MoCo-v3 and DINOv3), which were never trained with labels, while the data distilled with ResNet-18 (a supervised classifier with the same architecture as the student) is the weakest of our three in every setting. Even so, our ResNet-18 data still outperforms the individual baselines in 49 of 60 pairwise comparisons (28 of 30 with noise initialization), and outperforms every noise-initialized baseline in four of the six settings. This suggests that the efficacy of our data stems from matching the real distribution in a rich feature space rather than from alignment with the student architecture or the classification task. We also note that G-VBSM, FADRM+, and CV-DD (marked † in Tables 1 and 2) distill against an ensemble of backbones, whereas each of our variants uses a single encoder.

<table><tr><td></td><td colspan="4">Coverage</td><td colspan="4">k-NN</td><td colspan="4">Linear</td></tr><tr><td></td><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td></tr><tr><td>Method</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Noise init</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SRe2L [72]</td><td>7.2</td><td>11.6</td><td>4.9</td><td>8.5</td><td>50.1</td><td>54.3</td><td>46.0</td><td>49.1</td><td>51.2</td><td>55.6</td><td>41.1</td><td>45.4</td></tr><tr><td>CDA [71]</td><td>6.5</td><td>11.1</td><td>4.6</td><td>8.7</td><td>44.5</td><td>51.7</td><td>44.8</td><td>49.5</td><td>51.0</td><td>57.7</td><td>43.2</td><td>50.9</td></tr><tr><td>G-VBSM† [49]</td><td>5.7</td><td>8.6</td><td>5.3</td><td>8.4</td><td>41.3</td><td>45.9</td><td>39.7</td><td>42.5</td><td>62.3</td><td>70.5</td><td>47.1</td><td>61.8</td></tr><tr><td>WMDD [35]</td><td>9.5</td><td>17.5</td><td>9.4</td><td>19.0</td><td>49.6</td><td>56.0</td><td>45.8</td><td>50.7</td><td>65.2</td><td>66.4</td><td>45.7</td><td>58.7</td></tr><tr><td>SC-DD [78]</td><td>9.7</td><td>16.1</td><td>9.3</td><td>16.8</td><td>48.5</td><td>53.3</td><td>49.1</td><td>51.1</td><td>52.7</td><td>58.0</td><td>48.7</td><td>52.8</td></tr><tr><td>Ours (ResNet-18)</td><td>19.7</td><td>38.4</td><td>41.3</td><td>66.3</td><td>54.3</td><td>61.2</td><td>55.3</td><td>58.7</td><td>69.6</td><td>71.2</td><td>59.8</td><td>66.5</td></tr><tr><td>Ours (MoCo-v3)</td><td>61.2</td><td>86.9</td><td>(91.9)</td><td>(97.9)</td><td>64.3</td><td>68.4</td><td>(60.2)</td><td>(62.5)</td><td>74.6</td><td>74.6</td><td>(66.6)</td><td>(69.9)</td></tr><tr><td>Ours (DINOv3)</td><td>77.0</td><td>93.2</td><td>83.4</td><td>96.3</td><td>69.9</td><td>72.5</td><td>61.1</td><td>63.2</td><td>77.5</td><td>78.6</td><td>66.3</td><td>70.3</td></tr><tr><td>Real init</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RDED [56]</td><td>23.3</td><td>34.6</td><td>31.8</td><td>49.2</td><td>70.2</td><td>71.6</td><td>61.0</td><td>62.1</td><td>74.4</td><td>77.2</td><td>63.5</td><td>66.6</td></tr><tr><td>CUDD [36]</td><td>11.2</td><td>15.1</td><td>10.5</td><td>17.1</td><td>56.3</td><td>58.0</td><td>52.1</td><td>53.9</td><td>63.2</td><td>63.5</td><td>53.9</td><td>58.9</td></tr><tr><td>FADRM [8]</td><td>28.5</td><td>43.0</td><td>30.3</td><td>52.3</td><td>62.8</td><td>66.9</td><td>57.8</td><td>61.3</td><td>67.6</td><td>71.5</td><td>61.1</td><td>64.9</td></tr><tr><td>FADRM+† [8]</td><td>32.1</td><td>48.0</td><td>36.5</td><td>59.7</td><td>64.3</td><td>68.1</td><td>58.5</td><td>61.8</td><td>69.2</td><td>73.1</td><td>61.8</td><td>65.4</td></tr><tr><td>CV-DD{ [9]</td><td>10.0</td><td>16.6</td><td>8.4</td><td>15.8</td><td>55.7</td><td>61.1</td><td>51.8</td><td>54.9</td><td>62.7</td><td>66.6</td><td>52.7</td><td>57.4</td></tr><tr><td>Ours (ResNet-18)</td><td>30.0</td><td>57.5</td><td>54.9</td><td>82.0</td><td>62.1</td><td>68.3</td><td>58.0</td><td>61.3</td><td>72.8</td><td>74.5</td><td>63.1</td><td>67.8</td></tr><tr><td>Ours (MoCo-v3)</td><td>60.6</td><td>88.5</td><td>(91.3)</td><td>(98.0)</td><td>65.0</td><td>69.3</td><td>(60.2)</td><td>(62.2)</td><td>75.0</td><td>75.8</td><td>(66.4)</td><td>(70.0)</td></tr><tr><td>Ours (DINOv3)</td><td>79.9</td><td>95.5</td><td>85.1</td><td>97.5</td><td>69.9</td><td>72.8</td><td>60.3</td><td>63.0</td><td>77.0</td><td>78.5</td><td>65.7</td><td>70.2</td></tr></table>

Table 2: Feature-Space Metrics. Coverage (left), k-NN accuracy (middle), and linear probe accuracy (right) of distilled ImageNet-1k data in the SigLIP2 [60] and MoCo-v3 [7] latent spaces (see Appendix D). Parenthesized entries are self-evaluations and are not ranked. Methods marked with a dagger<sup>†</sup> distill using an ensemble of encoders. Our method outperforms all baselines in every feature-space setting except for RDED [56] on IPC 10 k-NN in both latent spaces. The performance gap is especially large on the coverage metric.

We include results for additional distillation backbones in Appendix Table B.1 and report from-scratch performance on the ImageNette and ImageWoof [19] datasets in Appendix Tables B.4 and B.5.

## 4.2 Drifted Synthetic Datasets Cover the Real Distribution.

While the traditional goal of dataset distillation methods has been the training of classifiers, our primary motivation is to learn a small set of synthetic images that best approximates the real distribution. To measure this directly, we evaluate the distilled data in the latent space of pre-trained encoders using three metrics: coverage [41], 1-nearest-neighbor (k-NN with $k = 1 )$ classification, and linear probe accuracy (details in Appendix D). Coverage measures the fraction of real validation points $\mathbf { y } _ { j } \in \mathcal { T } _ { \mathrm { v a l } }$ whose neighborhoods, defined by their k-th nearest-neighbor distances $\mathrm { N N D } _ { k } ( { \bf y } _ { j } )$ with respect to other real points, contain at least one synthetic point:

$$
\mathrm { C o v e r a g e } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { 1 } \left[ \exists i \mathrm { ~ s . t . ~ } d \big ( \phi ( \mathbf { x } _ { i } ) , \phi ( \mathbf { y } _ { j } ) \big ) \leq \mathrm { N N D } _ { k } ( \mathbf { y } _ { j } ) \right]\tag{10}
$$

where d is the cosine distance. We use $k = 5$ and report the average coverage over all classes. We also add 10 random crops of each image to the pool to widen the synthetic distribution.

We evaluate in the latent spaces of SigLIP2 [60] (language-supervised) and MoCo-v3 [7] (selfsupervised). Cells in which our MoCo-v3 variant is scored in its own distillation space are shown in parentheses and excluded from the rankings. Such self-evaluation seems to strongly inflate coverage (91.9% versus 83.4% for our DINOv3 variant at 10 images per class with noise initialization), but not k-NN or linear accuracy, where our MoCo-v3 variant is roughly on par with our DINOv3 variant.

In Table 2, our DINOv3 variant achieves the highest coverage in every setting (excluding self-evaluation); at 10 images per class, it covers 77.0% (SigLIP2) and 83.4% (MoCo-v3) of the real distribution, compared to at most 9.7% for any noise-initialized and 36.5% for any real-initialized baseline. Our ResNet-18 variant covers more of the distribution than every noise-initialized baseline in all settings, and more than every real-initialized baseline in all but one (SigLIP2 at 10 images per class, where FADRM+ reaches 32.1% versus our 30.0%). Our DINOv3 variant is also the best on k-NN and linear probe accuracy in every setting except two k-NN cells at 10 images per class, where RDED is ahead by less than one point. Notably, the gaps in coverage are far larger than those in k-NN or linear accuracy: classification metrics only require the distilled data to separate classes, whereas coverage rewards spanning the full extent of each class distribution, which is precisely what drifting optimizes for. Qualitatively, we see our distilled images are more diverse than those of the baseline methods, likely aiding in the distribution coverage (Figure 4).

![](images/10ebd0243a5f9ca6c295f762aa68c31ebbca46d5130c714c61f753e508cce0fc.jpg)  
Figure 5: Distilled images elucidate model representations. Each of the models shown here induces a different style of distilled images to best cover its own distribution. CLIP [45] is a notable outlier, possibly due to its strong text-entanglement and the contrastive loss not used in more recent vision-text models, such as SigLIP2 [60].

Much of this distributional coverage comes from the distillation-time augmentations (Table A.1), which allow our method to explicitly optimize for intra-image diversity. We visualize this in Figure 2; while the distilled images themselves (stars) are relatively evenly spaced throughout the latent space, their augmentations (triangles) help cover the rest of the real distribution.

## 4.3 Distilled Data Elucidate Model Biases.

One recent work (Linear Gradient Matching [3]) focused specifically on distillation for pre-trained feature extractors. While this method highlighted the interactions between different models and their distilled data, it could not scale past 1 image per class due to massive resource requirements as a result of needing to jointly distill all classes together. We pick up where this work left off and distill 10 images per class with seven different backbones (ResNet-18 [26], ViT [16], CLIP [45], SigLIP2 [60], EVA-02 [18], MoCo-v3 [7], and DINOv3 [53]). In Figure 5, we see the monarch butterfly class distilled to 10 images per class for each of these models. As noted for 1 image per class in the linear gradient matching settings [3], each model yields images with its own particular bias despite using the same underlying distillation method. Most notably, we see that the CLIP images are quite murky and contain less-well-defined backgrounds than those of the other models, possibly due to the model’s text entanglement. This is reflected quantitatively as well; when each set is evaluated in the other models feature spaces (Appendix Table B.6), CLIP-distilled data cover the least of the real distribution (on average) while DINOv3-distilled data cover the most.

![](images/84d957a7dced21cf0d18ef324accabc2ed89e1ed0ef2426420acedfa9e07f0a2.jpg)

<table><tr><td></td><td colspan="2">Audio</td><td colspan="2">Video</td></tr><tr><td>Method</td><td>Linear</td><td>k-NN</td><td>Linear</td><td>k-NN</td></tr><tr><td>Herding</td><td>45.6</td><td>39.9</td><td>66.6</td><td>60.4</td></tr><tr><td>Ours</td><td>51.4</td><td>52.6</td><td>70.3</td><td>65.8</td></tr></table>

Figure 6: Four frames of a basketball dribbling video from HMDB51 [32] distilled via drifting with VideoMAE-v2 [63] into a 3D Laplacian pyramid.  
![](images/059f21683a36287d10e8096f0abf7283786b71db21eeba5e2078436051bb265d.jpg)

Table 3: Our drifting method generalizes to both audio and video, outperforming the coreset baseline.  
![](images/466e202efad5e489ff2fb79866223f01904f46c19694c9fe1a92ec3d02a4b0a5.jpg)  
Figure 7: Distilling Canvases. We can distill a collection of images into a single large canvas via drifting by using a large number of random crops at each step. The above 2048 × 2048 pixel images were synthesized by drifting SigLIP2 [60] features to match the distribution of Claude Monet’s (left) and Qi Baishi’s (right) paintings.

## 4.4 Drifting Distillation Generalizes to Other Modalities.

We find that our drifting method and pyramid parameterization generalize to both audio and video data. For audio, we distill the FMA [11] small dataset of 8 music genres using the MusicFM-FMA [68] feature extraction model. For video, we distill the HMDB51 [32] human action recognition dataset of 51 action categories using the VideoMAE-v2 [63] feature extraction model. We distill 5 samples per class and evaluate with a linear probe and cosine 1-NN classification on mean-pooled features. For both modalities, we compare to and outperform the herding [66] coreset selection baseline (Table 3). Figure 6 shows frames from one of our distilled “dribbling” videos.

## 4.5 An Image is Worth a Thousand Paintings.

While dataset distillation’s artistic value has been briefly explored in the past [5], we find that our drifting method excels at distilling the essence of an art style into a large canvas [52]. In Figure 7, we see the paintings of Claude Monet and Qi Baishi distilled into single images using SigLIP2 [60]. Remarkably, each distilled image clearly reflects its painter’s individual style and contains common elements found throughout their work (e.g., Monet’s water lilies and Qi’s squirrels).

## 5 Conclusion

In this work, we introduced a new method of Dataset Distillation via Drifting, designed to explicitly learn a synthetic set of images that best covers the target distribution. In addition to best approximating the real data distribution when compared to the baseline methods, our distilled images also serve as superior training data for classification models from scratch.

Furthermore, our drifting method offers insights into model interpretability, generalizes to audio and video, and can be used to synthesize artwork. We hope our work inspires further dataset distillation research, especially in the direction of distribution coverage.

Acknowledgments. This work was supported by the National Science Foundation under Grant No. 2211259, by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) under 140D0423C0075, by the Amazon Science Hub, by the MIT-Google Program for Computing Innovation, by AMD via the MIT AI Hardware Program, by an ONR MURI grant 037564-00001, and by a 2025 MIT Office of Research Computing and Data Seed Grant. Giannis Daras is supported by a Jane Street Research Collaboration award, a TPU Research Cloud grant, and Lambda’s Research Grant Program. We thank Mingyang Deng for sharing an early version of his drifting code with us and other members of the MIT community for providing valuable feedback on this work.

## References

[1] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, M. Lezcano, Y. Liang, J. Liang, Y. Lu, C. K. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, S. Zhang, M. Suo, P. Tillet, X. Zhao, E. Wang, K. Zhou, R. Zou, X. Wang, A. Mathews, W. Wen, G. Chanan, P. Wu, and S. Chintala. PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In ASPLOS, 2024. 25

[2] Z. Borsos, M. Mutny, and A. Krause. Coresets via bilevel optimization for continual learning and streaming. NeurIPS, 2020. 2

[3] G. Cazenavette, A. Torralba, and V. Sitzmann. Dataset distillation for pre-trained self-supervised vision models. NeurIPS, 2025. 2, 3, 5, 9

[4] G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, and J.-Y. Zhu. Dataset distillation by matching training trajectories. In CVPR, 2022. 2

[5] G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, and J.-Y. Zhu. Wearable ImageNet: Synthesizing tileable textures via dataset distillation. In CVPR, 2022. 10

[6] G. Cazenavette, T. Wang, A. Torralba, A. A. Efros, and J.-Y. Zhu. Generalizing dataset distillation via deep generative prior. In CVPR, 2023. 3

[7] X. Chen, S. Xie, and K. He. An empirical study of training self-supervised vision transformers. In ICCV, 2021. 2, 6, 7, 8, 9, 18, 20, 25

[8] J. Cui, X. Bi, Y. Luo, X. Zhao, J. Liu, and Z. Shen. FADRM: Fast and accurate data residual matching for dataset distillation. In NeurIPS, 2025. 2, 6, 8, 19, 20

[9] J. Cui, Z. Li, X. Ma, X. Bi, Y. Luo, and Z. Shen. Dataset distillation via committee voting. arXiv, 2025. 2, 6, 8

[10] J. Cui, R. Wang, S. Si, and C.-J. Hsieh. Scaling up dataset distillation to ImageNet-1K with constant memory. In ICML, 2023. 2

[11] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson. FMA: A dataset for music analysis. In ISMIR, 2017. 10, 25

[12] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009. 1, 2, 6, 21, 25

[13] M. Deng, H. Li, T. Li, Y. Du, and K. He. Generative modeling via drifting. In NeurIPS, 2026. 2, 3, 5, 21, 24

[14] P. Dey, A. Sahdev, S. Bhati, K. R. Mopuri, and R. V. Babu. Rethinking dataset distillation: Hard truths about soft labels. In CVPR, 2026. 21

[15] T. Dong, B. Zhao, and L. Liu. Privacy for free: How does dataset condensation help privacy? In ICML, 2022. 3, 6

[16] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 2, 9, 20, 25

[17] J. Du, X. Zhang, J. Hu, W. Huang, and J. T. Zhou. Diversity-driven synthesis: Enhancing dataset distillation through directed weight adjustment. NeurIPS, 2024. 2

[18] Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao. EVA-02: A visual representation for neon genesis. Image and Vision Computing, 2024. 9, 20, 25

[19] Fastai. Fastai/imagenette: A smaller subset of 10 easily classified classes from imagenet, and a little more french. 8, 19, 20, 21, 25

[20] D. Feldman and M. Langberg. A unified framework for approximating and clustering data. In STOC, 2011. 2

[21] S. Fort and J. Whitaker. Direct ascent synthesis: Revealing hidden generative capabilities in discriminative models. arXiv, 2025. 5

[22] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial networks. Communications ofthe ACM, 63(11), 2020. 3

[23] J. Gu, S. Vahidian, V. Kungurtsev, H. Wang, W. Jiang, Y. You, and Y. Chen. Efficient dataset distillation via minimax diffusion. In CVPR, 2024. 3, 6, 18, 19, 21

[24] Z. Guo, K. Wang, G. Cazenavette, H. Li, K. Zhang, and Y. You. Towards lossless dataset distillation via difficulty-aligned trajectory matching. In ICLR, 2024. 2

[25] S. Har-Peled and S. Mazumdar. On coresets for k-means and k-median clustering. In STOC, 2004. 2

[26] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016. 2, 6, 9, 21, 25

[27] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. In NeurIPS Workshop, 2015. 2

[28] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3

[29] E. Hoogeboom, J. Heek, and T. Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. 3

[30] Hugging Face. Hugging Face: The AI community building the future. https://huggingface.co, 2016. 25

[31] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 21

[32] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A large video database for human motion recognition. In ICCV, 2011. 10, 25

[33] M. Li, H. Gou, Y. Ma, R. Wang, K. Qin, and T. He. Fixed anchors are not enough: Dynamic retrieval and persistent homology for dataset distillation. In CVPR, 2026. 2

[34] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In ICLR, 2023. 3

[35] H. Liu, Y. Li, T. Xing, P. Wang, V. Dalal, L. Li, J. He, and H. Wang. Dataset distillation via the Wasserstein metric. In ICCV, 2025. 2, 6, 7, 8, 19, 20

[36] Z. Ma, A. Cao, F. Yang, Y. Gong, and X. Wei. Curriculum dataset distillation. TIP, 2025. 2, 6, 8

[37] B. Mirzasoleiman, K. Cao, and J. Leskovec. Coresets for robust training of deep neural networks against noisy labels. NeurIPS, 2020. 2

[38] S. Mohanty, A. Reddy, and K. R. Mopuri. DiRe: Diversity-promoting regularization for dataset condensa tion. In WACV, 2026. 2

[39] A. Mordvintsev, C. Olah, and M. Tyka. DeepDream – a code example for visualizing neural networks. Google Research Blog, jul 2015. 2

[40] B. B. Moser, S. Sarode, F. Raue, S. Frolov, K. Adamkiewicz, A. Shanbhag, J. Folz, T. C. Nauen, and A. Dengel. PRISM: Diversifying dataset distillation by decoupling architectural priors. TMLR, 2026. 2

[41] M. F. Naeem, S. J. Oh, Y. Uh, Y. Choi, and J. Yoo. Reliable fidelity and diversity metrics for generative models. In ICML, 2020. 2, 8, 23

[42] C. Olah, A. Mordvintsev, and L. Schubert. Feature visualization. Distill, 2017. https://distill.pub/2017/feature-visualization. 5

[43] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS. 2019. 25

[44] M. Paul, S. Ganguli, and G. K. Dziugaite. Deep learning on a data diet: Finding important examples early in training. NeurIPS, 2021. 2

[45] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 9, 20, 25

[46] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. iCaRL: Incremental classifier and representation learning. In CVPR, 2017. 2

[47] A. Roy, W.-Y. A. Lee, R. Chakraborty, and V. S. Lokhande. ManifoldGD: Training-free hierarchical manifold guidance for diffusion-based dataset distillation. In CVPR, 2026. 3, 6, 18, 19

[48] O. Sener and S. Savarese. Active learning for convolutional neural networks: A core-set approach. In ICLR, 2018. 2

[49] S. Shao, Z. Yin, M. Zhou, X. Zhang, and Z. Shen. Generalized large-scale data condensation via various backbone and statistical matching. In CVPR, 2024. 2, 6, 7, 8, 19, 20

[50] S. Shao, Z. Zhou, H. Chen, and Z. Shen. Elucidating the design space of dataset condensation. NeurIPS, 2024. 2, 22

[51] Z. Shen, A. Sherif, Z. Yin, and S. Shao. DELT: A simple diversity-driven EarlyLate training for dataset distillation. In CVPR, 2025. 2

[52] A. Shul, E. Horwitz, and Y. Hoshen. Distilling datasets into less than one image. TMLR, 2025. 10

[53] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski. DINOv3. arXiv, 2025. 1, 6, 7, 9, 18, 19, 20, 25

[54] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 3

[55] D. Su, J. Hou, W. Gao, Y. Tian, and B. Tang. D<sup>4</sup>M: Dataset distillation via disentangled diffusion model. In CVPR, 2024. 3, 6, 18, 19

[56] P. Sun, B. Shi, D. Yu, and T. Lin. On the diversity and realism of distilled dataset: An efficient dataset distillation paradigm. In CVPR, 2024. 2, 6, 7, 8, 19, 20, 21

[57] M. Toneva, A. Sordoni, R. T. d. Combes, A. Trischler, Y. Bengio, and G. J. Gordon. An empirical study of example forgetting during deep neural network learning. In ICLR, 2019. 2

[58] TorchVision maintainers and contributors. TorchVision: PyTorch’s computer vision library. https: //github.com/pytorch/vision, 2016. 25

[59] M.-T. Tran, T. Le, X.-M. Le, T.-T. Do, and D. Phung. Enhancing dataset distillation via non-critical region refinement. In CVPR, 2025. 2

[60] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv, 2025. 8, 9, 10, 20, 25

[61] H. Wang, Z. Zhao, J. Wu, Y. Shang, G. Liu, and Y. Yan. CaO<sub>2</sub>: Rectifying inconsistencies in diffusion-based dataset distillation. In ICCV, 2025. 3, 6, 18, 19

[62] K. Wang, B. Zhao, X. Peng, Z. Zhu, S. Yang, S. Wang, G. Huang, H. Bilen, X. Wang, and Y. You. CAFE: Learning to condense dataset by aligning features. CVPR, 2022. 2

[63] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao. VideoMAE V2: Scaling video masked autoencoders with dual masking. In CVPR, 2023. 10, 25

[64] S. Wang, Y. Yang, G. Chen, P. Li, K. Li, Y. Zhou, Z. Chen, and L. Zhang. Grounding and enhancing informativeness and utility in dataset distillation. In ICLR, 2026. 2

[65] T. Wang, J.-Y. Zhu, A. Torralba, and A. A. Efros. Dataset distillation. arXiv, 2018. 2

[66] M. Welling. Herding dynamical weights to learn. In ICML, 2009. 2, 10

[67] R. Wightman. PyTorch image models. https://huggingface.co/timm, 2019. 25

[68] M. Won, Y.-N. Hung, and D. Le. A foundation model for music informatics. In ICASSP, 2024. 10, 25

[69] S. Yang, Z. Cao, S. Guo, R. Zhang, P. Luo, S. Zhang, and L. Nie. Mind the boundary: Coreset selection via reconstructing the decision boundary. In ICML, 2024. 2

[70] H. Yin, P. Molchanov, J. M. Alvarez, Z. Li, A. Mallya, D. Hoiem, N. K. Jha, and J. Kautz. Dreaming to distill: Data-free knowledge transfer via DeepInversion. In CVPR, 2020. 2

[71] Z. Yin and Z. Shen. Dataset distillation via curriculum data synthesis in large data era. TMLR, 2024. 2, 6, 7, 8, 19, 20

[72] Z. Yin, E. Xing, and Z. Shen. Squeeze, recover and relabel: Dataset condensation at ImageNet scale from a new perspective. NeurIPS, 2023. 2, 6, 7, 8, 19, 20

[73] X. Zhang, J. Du, P. Liu, and J. T. Zhou. Breaking class barriers: Efficient dataset distillation via inter-class feature compensator. In ICLR, 2025. 2

[74] B. Zhao and H. Bilen. Dataset condensation with differentiable Siamese augmentation. In ICML, 2021. 2

[75] B. Zhao and H. Bilen. Dataset condensation with distribution matching. WACV, 2023. 2

[76] B. Zhao, K. R. Mopuri, and H. Bilen. Dataset condensation with gradient matching. In ICLR, 2020. 2

[77] L. Zhao, Y. Wu, X. Jiang, J. Gu, Y. Wang, X. Xu, P. Zhao, and X. Lin. Taming diffusion for dataset distillation with high representativeness. In ICML, 2025. 3, 6, 18, 19

[78] M. Zhou, Z. Yin, S. Shao, and Z. Shen. Self-supervised dataset distillation: A good compression is all you need. arXiv, 2024. 2, 6, 7, 8

[79] C. Zhu, S. Suri, C. Jose, M. Oquab, M. Szafraniec, W. Wen, Y. Xiong, P. Labatut, P. Bojanowski, R. Krishnamoorthi, and V. Chandra. Efficient universal perception encoder. arXiv, 2026. 20, 25

## Appendix

## A Hyperparameter Sweeps

Each study below changes a single setting of our default configuration (DINOv3 backbone, noise initialization, ImageNet-1k, IPC 10) and reports the same from-scratch and feature-space metrics as Tables 1 and 2; defaults are marked †. Each table is paired with a figure showing all 10 distilled images of a single class.

Augmentations per step. Drifting augmented views of each image is critical. Without augmentation, every metric collapses (e.g., 0.2% hard-label accuracy and 1.0% coverage). Additionally, every metric improves monotonically as more augmented views are drifted per step, with 10 views performing best throughout (Table A.1, Figure A.1).

![](images/db92cc725bd8495ee1f0f7068f097036a0ef79537ee3c95867e00a5df7af6912.jpg)  
Figure A.1: Effects of augmentations. Here, we visualize the effect the number of augmentations per step has on the distilled data. As we increase the number of augmentations, the images appear to (qualitatively) become richer in semantic information. Images shown here were initialized with noise and distilled with DINOv3 to match ImageNet-1k’s “orange” class. For quantitative results, please see Table A.1.

<table><tr><td rowspan="2"></td><td colspan="3">ResNet-18 from scratch</td><td colspan="2">Coverage</td><td colspan="2">k-NN</td><td colspan="2">Linear</td></tr><tr><td>Hard</td><td>Soft</td><td>KD</td><td> ${ \mathrm { S i g L I P } } 2$ </td><td> $\mathbf { M o C o - v } 3$ </td><td>SigLIP2</td><td>MoCo-v3</td><td> ${ \mathrm { S i g L I P } } 2$ </td><td>MoCo-v3</td></tr><tr><td>None</td><td> $0 . 2 \pm 0 . 0$ </td><td> $0 . 2 \pm 0 . 0$ </td><td> $1 . 9 \pm 0 . 1$ </td><td>1.0</td><td>0.6</td><td>2.7</td><td>5.8</td><td>6.5</td><td>4.9</td></tr><tr><td>1</td><td> $7 . 5 \pm 0 . 3$ </td><td> $2 7 . 4 \pm 0 . 3$ </td><td> $3 9 . 3 \pm 0 . 3$ </td><td>35.8</td><td>28.6</td><td>56.0</td><td>50.2</td><td>69.5</td><td>59.2</td></tr><tr><td>2</td><td> $1 5 . 5 \pm 0 . 4$ </td><td> $3 4 . 4 \pm 0 . 2$ </td><td> $4 2 . 4 \pm 0 . 6$ </td><td>65.5</td><td>66.0</td><td>66.5</td><td>57.4</td><td>76.6</td><td>64.9</td></tr><tr><td>5</td><td> $\underline { { 1 9 . 1 } } \pm 0 . 4$ </td><td> $3 5 . 4 \pm 0 . 3$ </td><td> $\underline { { 4 4 . 2 } } \pm 0 . 4$ </td><td>75.1</td><td>79.9</td><td>69.4</td><td>60.3</td><td>77.3</td><td>66.1</td></tr><tr><td>10†</td><td> $2 1 . 3 \pm 0 . 5$ </td><td>35.9 ± 0.1 45.0 ± 0.2</td><td></td><td>77.0</td><td>83.4</td><td>69.9</td><td>61.1</td><td>77.5</td><td>66.3</td></tr></table>

Table A.1: Augmented copies per step. The number of augmented views of each distilled image used at every optimization step. None is a single deterministic resized view. Performance monotonically increases with the number of augmentations, with diminishing returns as we approach 10.

Backbone size. A larger backbone does not necessarily lead to better data. DINOv3 ViT-B is best or tied on every metric, while ViT-L matches it only in the knowledge distillation setting and ViT-S covers notably less of the distribution (Table A.2, Figure A.2). However, the ViT-L variant could possibly perform better with more hyperparameter tuning.

![](images/e5102850fcc64af0810be913a9b1730c10b805c93bec5232c4ed9ab95b6a57ad.jpg)  
Figure A.2: Effect of backbone size. All 10 distilled images of the ImageNet-1k “lorikeet” class using DINOv3 ViT-S, ViT-B (default), and ViT-L. For quantitative results, see Table A.2.

<table><tr><td rowspan="3"></td><td colspan="3">ResNet-18 from scratch</td><td colspan="2">Coverage</td><td colspan="2">k-NN</td><td colspan="2">Linear</td></tr><tr><td>Hard</td><td>Soft</td><td>KD</td><td></td><td>SigLIP2 MoCo-v3</td><td>SigLIP2</td><td>MoCo-v3</td><td></td><td>SigLIP2 MoCo-v3</td></tr><tr><td>Small</td><td> $1 8 . 5 \pm 0 . 5$ </td><td> $3 1 . 9 \pm 0 . 5$ </td><td> $4 1 . 9 \pm 0 . 5$ </td><td>61.8</td><td>77.4</td><td>67.6</td><td>60.9</td><td>77.2</td><td>65.3</td></tr><tr><td>Base†</td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> ${ \bf 4 5 . 0 \pm 0 . 2 }$ </td><td>77.0</td><td>83.4</td><td>69.9</td><td>61.1</td><td>77.5</td><td>66.3</td></tr><tr><td>Large</td><td> $1 8 . 3 \pm 0 . 3$ </td><td> ${ 3 4 . 0 \pm 0 . 4 }$ </td><td> ${ \pm } 4 . 6 \pm 0 . 4$ </td><td>75.6</td><td>78.1</td><td>69.8</td><td>60.2</td><td>76.9</td><td>65.6</td></tr></table>

Table A.2: Backbone model size. Base size performs the best with our settings, but the larger model might improve with some tuning.

Kernel temperature. A small temperature $( \tau \ : = \ : 0 . 0 5 )$ slightly improves k-NN accuracy but substantially lowers soft-label accuracy and coverage, while a large temperature $( \tau = 5 )$ matches the default in the soft-label and knowledge distillation settings but lowers hard-label accuracy and coverage. Our default $( \tau = 0 . 5 )$ gives the best balance (Table A.3, Figure A.3).

![](images/fed7ca8b5b6697d0340cdbabfe0ec267704d736f9bf4fbdfaf496164d2335b69.jpg)  
Figure A.3: Effect of kernel temperature. All 10 distilled images of the ImageNet-1k “terrapin” class for three drifting kernel temperatures τ (default $\tau = 0 . 5 )$ . For quantitative results, see Table A.3.

<table><tr><td rowspan="2"></td><td colspan="3">ResNet-18 from scratch</td><td colspan="2">Coverage</td><td colspan="2">k-NN</td><td colspan="2">Linear</td></tr><tr><td>Hard</td><td>Soft</td><td>KD</td><td></td><td>SigLIP2 MoCo-v3</td><td>SigLIP2</td><td>2 MoCo-v3</td><td></td><td>SigLIP2 MoCo-v3</td></tr><tr><td> $\tau = 0 . 0 5$ </td><td> $\underline { { 1 7 . 9 } } \pm 0 . 2$ </td><td> $2 6 . 4 \pm 0 . 1$ </td><td> $3 9 . 6 \pm 0 . 2$ </td><td>71.6</td><td>66.6</td><td>72.7</td><td>62.7</td><td>76.9</td><td>64.4</td></tr><tr><td> $\tau = 0 . 5 ^ { \dagger }$ </td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> ${ \bf 4 5 . 0 \pm 0 . 2 }$ </td><td>77.0</td><td>83.4</td><td>69.9</td><td>61.1</td><td>77.5</td><td>66.3</td></tr><tr><td> $\tau = 5$ </td><td> $1 7 . 7 \pm 0 . 1$ </td><td> ${ \bf 3 5 . 9 \pm 0 . 4 }$ </td><td> ${ \bf 4 5 . 1 \pm 0 . 5 }$ </td><td>64.7</td><td>70.7</td><td>66.2</td><td>57.8</td><td>76.1</td><td>65.0</td></tr></table>

Table A.3: Drifting distance kernel temperature. The method remains decently robust to a higher temperature, but a lower one significantly hurts performance.

Drifted features. Drifting the CLS token alone already yields usable data, but adding the mean patch token and, even more so, every patch token steadily improves from-scratch accuracy (11.8% → 15.0% → 21.3% hard-label) and coverage, while k-NN and linear accuracy remain nearly unchanged (Table A.4, Figure A.4).

![](images/15ad935dd2e1d7d6bec30a678b6c82953ca4d3332a4c7ab974d450864e1f3b0c.jpg)

Figure A.4: Effect of the drifted features. All 10 distilled images of the ImageNet-1k “golden retriever” class when the drifting loss is applied to the CLS token only, the CLS token plus the mean patch token, or the CLS token plus every patch token (default). For quantitative results, see Table A.4.
<table><tr><td rowspan="2"></td><td colspan="3">ResNet-18 from scratch</td><td colspan="2">Coverage</td><td colspan="2">k-NN</td><td colspan="2">Linear</td></tr><tr><td>Hard</td><td>Soft</td><td>KD</td><td>SigLIP2 MoCo-v3</td><td></td><td>SigLIP2</td><td>MoCo-v3</td><td></td><td>SigLIP2 MoCo-v3</td></tr><tr><td>CLS</td><td> $1 1 . 8 \pm 0 . 2$ </td><td> $2 6 . 4 \pm 0 . 0$ </td><td> $4 1 . 3 \pm 0 . 3$ </td><td>70.2</td><td>72.5</td><td>69.4</td><td>61.8</td><td>76.7</td><td>66.0</td></tr><tr><td>+mean</td><td> $1 5 . 0 \pm 0 . 3$ </td><td> $3 1 . 0 \pm 0 . 3$ </td><td> $4 3 . 6 \pm 0 . 6$ </td><td>74.2</td><td>80.0</td><td>69.3</td><td>61.3</td><td>77.4</td><td>66.6</td></tr><tr><td>+grid†</td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> ${ \bf 4 5 . 0 \pm 0 . 2 }$ </td><td>77.0</td><td>83.4</td><td>69.9</td><td>61.1</td><td>77.5</td><td>66.3</td></tr></table>

Table A.4: Drifted features. The DINOv3 tokens to which the drifting loss is applied: the CLS token only, the CLS token plus the mean patch token (+mean), or the CLS token plus every patch token (+grid). Drifting the CLS token with the full grid performs best.

Parameterization. Both components of our parameterization matter. Removing color decorrelation lowers hard-label accuracy from 21.3% to 16.5% and SigLIP2 coverage from 77.0% to 62.0%, and replacing the Laplacian pyramid with raw pixels collapses from-scratch accuracy (3.3% hard-label) and coverage (41.9%). Notably, k-NN and linear accuracy barely change in either case, again showing that these metrics are much less sensitive to the quality of the distilled data than from-scratch training or coverage (Table A.5, Figure A.5).

![](images/8987bf357a095a87560bfa2a700ceaa081e693bb8cf2bf68601307c7b6062f9b.jpg)

Figure A.5: Effect of the parameterization. All 10 distilled images of the ImageNet-1k “flamingo” class with our full parameterization, without color decorrelation (−color), and with raw pixels instead of a Laplacian pyramid (−pyramid). For quantitative results, see Table A.5.
<table><tr><td rowspan="2"></td><td colspan="3">ResNet-18 from scratch</td><td colspan="2">Coverage</td><td colspan="2">k-NN</td><td colspan="2">Linear</td></tr><tr><td>Hard</td><td>Soft</td><td>KD</td><td> ${ \mathrm { S i g L I P } } 2$ </td><td>MoCo-v3</td><td>SigLIP2</td><td>MoCo-v3</td><td>SigLIP2 MoCo-v3</td><td></td></tr><tr><td>Full†</td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> ${ \bf 4 5 . 0 \pm 0 . 2 }$ </td><td>77.0</td><td>83.4</td><td>69.9</td><td>61.1</td><td>77.5</td><td>66.3</td></tr><tr><td>-color</td><td> $\underline { { 1 6 . 5 } } \pm 0 . 5$ </td><td> $3 1 . 4 \pm 0 . 8$ </td><td> $3 8 . 1 \pm 0 . 8$ </td><td>62.0</td><td>76.5</td><td>69.7</td><td>60.6</td><td>77.4</td><td>65.7</td></tr><tr><td>—pyramid</td><td> $3 . 3 \pm 0 . 3$ </td><td> $1 1 . 0 \pm 1 . 0$ </td><td> $2 4 . 7 \pm 1 . 0$ </td><td>41.9</td><td>46.4</td><td>69.2</td><td>59.7</td><td>77.2</td><td>62.1</td></tr></table>

Table A.5: Parameterization. Our full parameterization, without color decorrelation (−color), and with raw pixels in place of the Laplacian pyramid (−pyramid). The full parameterization performs best.

<table><tr><td rowspan="2">Method</td><td colspan="2">Hard-Label</td><td colspan="2">Soft-Label</td><td colspan="2">Knowledge Distillation</td></tr><tr><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Noise init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td> $1 7 . 6 \pm 0 . 2$ </td><td> $1 8 . 4 \pm 0 . 5$ </td><td> $3 0 . 5 \pm 0 . 8$ </td><td> $4 3 . 1 \pm 0 . 3$ </td><td> $4 4 . 8 \pm 0 . 7$ </td><td> $5 5 . 0 \pm 0 . 3$ </td></tr><tr><td>ViT</td><td> $2 0 . 7 \pm 0 . 4$ </td><td> $2 8 . 4 \pm 0 . 6$ </td><td> $3 4 . 0 \pm 0 . 6$ </td><td> $4 8 . 7 \pm 0 . 1$ </td><td> $4 4 . 2 \pm 0 . 2$ </td><td> $5 4 . 9 \pm 0 . 2$ </td></tr><tr><td>CLIP</td><td> $6 . 1 \pm 0 . 2$ </td><td> $8 . 4 \pm 0 . 3$ </td><td> $1 8 . 5 \pm 0 . 2$ </td><td> $2 9 . 2 \pm 0 . 5$ </td><td> $3 2 . 9 \pm 0 . 3$ </td><td> $4 4 . 3 \pm 0 . 4$ </td></tr><tr><td>SigLIP2</td><td> $1 4 . 9 \pm 0 . 2$ </td><td> $2 1 . 0 \pm 0 . 0$ </td><td> $3 0 . 7 \pm 0 . 5$ </td><td> $4 4 . 5 \pm 0 . 7$ </td><td> $4 0 . 8 \pm 0 . 3$ </td><td> $5 2 . 2 \pm 0 . 6$ </td></tr><tr><td>EVA-02</td><td> $1 6 . 8 \pm 0 . 3$ </td><td> $2 4 . 7 \pm 0 . 2$ </td><td> $3 1 . 5 \pm 0 . 3$ </td><td> $4 5 . 9 \pm 0 . 5$ </td><td> $4 2 . 8 \pm 0 . 9$ </td><td> $5 3 . 8 \pm 0 . 1$ </td></tr><tr><td>MoCo-v3</td><td> $2 7 . 0 \pm 0 . 2$ </td><td> $3 3 . 5 \pm 1 . 1$ </td><td> ${ \bf 4 0 . 3 \pm 0 . 1 }$ </td><td> ${ \pm 2 . 8 \pm 0 . 0 }$ </td><td> ${ \bf 4 8 . 5 \pm 0 . 1 }$ </td><td> ${ \pm } 7 . 2 \pm 0 . 3$ </td></tr><tr><td>DINOv3</td><td> $\underline { { 2 1 . 3 } } \pm 0 . 5$ </td><td> $3 0 . 2 \pm 0 . 2$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> $5 0 . 7 \pm 0 . 1$ </td><td> $4 5 . 0 \pm 0 . 2$ </td><td> $5 6 . 0 \pm 0 . 2$ </td></tr><tr><td>Real init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td> $2 0 . 4 \pm 0 . 4$ </td><td> $2 2 . 0 \pm 0 . 4$ </td><td> $3 3 . 9 \pm 0 . 2$ </td><td> $4 5 . 4 \pm 0 . 5$ </td><td> $4 7 . 8 \pm 0 . 2$ </td><td> $5 6 . 4 \pm 0 . 3$ </td></tr><tr><td>ViT</td><td> $2 5 . 6 \pm 0 . 2$ </td><td> $3 7 . 6 \pm 0 . 4$ </td><td> $3 9 . 8 \pm 0 . 5$ </td><td> $5 2 . 9 \pm 0 . 2$ </td><td> $4 8 . 8 \pm 0 . 3$ </td><td> $5 7 . 6 \pm 0 . 1$ </td></tr><tr><td>CLIP</td><td> $2 0 . 9 \pm 0 . 2$ </td><td> $3 6 . 5 \pm 0 . 5$ </td><td> $3 5 . 5 \pm 0 . 2$ </td><td> $4 9 . 1 \pm 0 . 2$ </td><td> $4 7 . 3 \pm 0 . 2$ </td><td> $5 5 . 9 \pm 0 . 2$ </td></tr><tr><td>SigLIP2</td><td> $2 1 . 9 \pm 0 . 2$ </td><td> $3 2 . 3 \pm 0 . 3$ </td><td> $3 7 . 1 \pm 0 . 3$ </td><td> $5 0 . 0 \pm 0 . 2$ </td><td> $4 7 . 6 \pm 0 . 1$ </td><td> $5 6 . 4 \pm 0 . 2$ </td></tr><tr><td>EVA-02</td><td> $2 4 . 1 \pm 0 . 3$ </td><td> $\underline { { 3 7 . 7 } } \pm 0 . 2$ </td><td> $3 8 . 1 \pm 0 . 1$ </td><td> $5 1 . 8 \pm 0 . 4$ </td><td> $4 8 . 4 \pm 0 . 1$ </td><td> $5 7 . 1 \pm 0 . 3$ </td></tr><tr><td>MoCo-v3</td><td> $\pm 8 . 9 \pm 0 . 2$ </td><td> $3 6 . 0 \pm 0 . 7$ </td><td> ${ \bf 4 1 . 7 \pm 0 . 0 }$ </td><td> $5 3 . 4 \pm 0 . 1$ </td><td> ${ \bf 4 9 . 6 \pm 0 . 2 }$ </td><td> $5 7 . 7 \pm 0 . 2$ </td></tr><tr><td>DINOv3</td><td> $2 6 . 0 \pm 0 . 2$ </td><td> ${ \bf 4 1 . 1 \pm 0 . 4 }$ </td><td> $4 0 . 5 \pm 0 . 0$ </td><td> ${ \pm } 4 . 3 \pm 0 . 4$ </td><td> ${ \bf 4 9 . 5 \pm 0 . 3 }$ </td><td> ${ \pm } 8 . 1 \pm 0 . 1$ </td></tr></table>

Table B.1: From-scratch training across distillation backbones. Most backbones yield distilled data that can effectively train a model from scratch, with MoCo-v3 [7] and DINOv3 [53] eking out a slight edge over the others.

<table><tr><td rowspan="2"></td><td colspan="2">Hard-Label</td><td colspan="2">Soft-Label</td><td colspan="2">Knowledge Distillation</td></tr><tr><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Minimax [23]</td><td> $2 2 . 2 \pm 0 . 2$ </td><td> $3 6 . 1 \pm 0 . 3$ </td><td> $3 5 . 9 \pm 0 . 2$ </td><td> $4 8 . 8 \pm 0 . 2$ </td><td> $4 9 . 7 \pm 0 . 1$ </td><td> $5 8 . 1 \pm 0 . 1$ </td></tr><tr><td>D4M [55]</td><td> $1 7 . 0 \pm 0 . 1$ </td><td> $3 0 . 5 \pm 0 . 2$ </td><td> $3 5 . 8 \pm 0 . 2$ </td><td> $5 0 . 3 \pm 0 . 1$ </td><td> $4 9 . 6 \pm 0 . 3$ </td><td> $5 8 . 5 \pm 0 . 1$ </td></tr><tr><td>D³HR [77]</td><td> $2 3 . 7 \pm 0 . 2$ </td><td> $3 8 . 0 \pm 0 . 4$ </td><td> $3 5 . 7 \pm 0 . 1$ </td><td> $5 0 . 2 \pm 0 . 2$ </td><td> $4 8 . 1 \pm 0 . 3$ </td><td> $5 7 . 7 \pm 0 . 1$ </td></tr><tr><td>CaO2 [61]</td><td> $2 8 . 8 \pm 0 . 2$ </td><td> $4 4 . 3 \pm 0 . 1$ </td><td> $4 0 . 7 \pm 0 . 2$ </td><td> $5 4 . 1 \pm 0 . 1$ </td><td> ${ \pm } 1 . 4 \pm 0 . 3$ </td><td> $5 9 . 5 \pm 0 . 1$ </td></tr><tr><td>ManifoldGD [47]</td><td> $2 7 . 6 \pm 0 . 3$ </td><td> ${ \pm 3 . 8 \pm 0 . 4 }$ </td><td> $4 1 . 2 \pm 0 . 2$ </td><td> ${ \pm } 4 . 6 \pm 0 . 0$ </td><td> ${ \bf 5 1 . 5 \pm 0 . 1 }$ </td><td> ${ \pm \bf 9 . 8 \pm 0 . 1 }$ </td></tr><tr><td>Ours, noise init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td> $1 7 . 6 \pm 0 . 2$ </td><td> $1 8 . 4 \pm 0 . 5$ </td><td> $3 0 . 5 \pm 0 . 8$ </td><td> $4 3 . 1 \pm 0 . 3$ </td><td> $4 4 . 8 \pm 0 . 7$ </td><td> $5 5 . 0 \pm 0 . 3$ </td></tr><tr><td>MoCo-v3</td><td> $2 7 . 0 \pm 0 . 2$ </td><td> $3 3 . 5 \pm 1 . 1$ </td><td> $4 0 . 3 \pm 0 . 1$ </td><td> $5 2 . 8 \pm 0 . 0$ </td><td> $4 8 . 5 \pm 0 . 1$ </td><td> $5 7 . 2 \pm 0 . 3$ </td></tr><tr><td>DINOv3</td><td> $2 1 . 3 \pm 0 . 5$ </td><td> $3 0 . 2 \pm 0 . 2$ </td><td> $3 5 . 9 \pm 0 . 1$ </td><td> $5 0 . 7 \pm 0 . 1$ </td><td> $4 5 . 0 \pm 0 . 2$ </td><td> $5 6 . 0 \pm 0 . 2$ </td></tr><tr><td>Ours, real init</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td> $2 0 . 4 \pm 0 . 4$ </td><td> $2 2 . 0 \pm 0 . 4$ </td><td> $3 3 . 9 \pm 0 . 2$ </td><td> $4 5 . 4 \pm 0 . 5$ </td><td> $4 7 . 8 \pm 0 . 2$ </td><td> $5 6 . 4 \pm 0 . 3$ </td></tr><tr><td>MoCo-v3</td><td> $\pm 8 . 9 \pm 0 . 2$ </td><td> $3 6 . 0 \pm 0 . 7$ </td><td> ${ \bf 4 1 . 7 \pm 0 . 0 }$ </td><td> $5 3 . 4 \pm 0 . 1$ </td><td> $4 9 . 6 \pm 0 . 2$ </td><td> $5 7 . 7 \pm 0 . 2$ </td></tr><tr><td>DINOv3</td><td> $2 6 . 0 \pm 0 . 2$ </td><td> $4 1 . 1 \pm 0 . 4$ </td><td> $4 0 . 5 \pm 0 . 0$ </td><td> ${ \pm } 4 . 3 \pm 0 . 4$ </td><td> $4 9 . 5 \pm 0 . 3$ </td><td> $5 8 . 1 \pm 0 . 1$ </td></tr></table>

Table B.2: Comparison with Diffusion-Based Methods. Our method matches the performance of the best diffusion-based methods in the soft-label setting while falling slightly behind on hard-label and knowledge distillation. Despite not leveraging a pre-trained generative model, our distilled data remain competitive.

## B More Datasets and Backbones

Table B.1 extends Table 1 to additional distillation backbones. Table B.2 compares against diffusionbased methods, which we exclude from the main text because they rely on a generative model pre-trained on the target data; Table B.3 makes the same comparison in feature space. Tables B.4 and B.5 report from-scratch results on ImageNette and ImageWoof using their official splits. Finally, Table B.6 scores each of our distilled sets in every other backbone’s latent space.

<table><tr><td rowspan="3"></td><td colspan="4">Coverage</td><td colspan="4">k-NN</td><td colspan="4">Linear</td></tr><tr><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td><td colspan="2">SigLIP2</td><td colspan="2">MoCo-v3</td></tr><tr><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Diffusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Minimax [23]</td><td>60.6</td><td>82.3</td><td>67.6</td><td>88.2</td><td>67.2</td><td>70.3</td><td>57.9</td><td>61.4</td><td>71.9</td><td>74.4</td><td>61.2</td><td>64.4</td></tr><tr><td>D4M [55]</td><td>37.1</td><td>60.3</td><td>47.1</td><td>76.3</td><td>57.8</td><td>62.3</td><td>48.1</td><td>53.5</td><td>65.6</td><td>67.7</td><td>55.2</td><td>62.2</td></tr><tr><td>D3HR [77]</td><td>55.0</td><td>78.9</td><td>65.0</td><td>88.5</td><td>67.9</td><td>71.0</td><td>59.4</td><td>62.7</td><td>72.7</td><td>75.1</td><td>62.4</td><td>66.2</td></tr><tr><td>CaO2 [61]</td><td>72.9</td><td>92.3</td><td>71.2</td><td>94.1</td><td>67.3</td><td>70.2</td><td>58.5</td><td>61.1</td><td>73.1</td><td>75.9</td><td>62.8</td><td>66.9</td></tr><tr><td>ManifoldGD [47]</td><td>77.4</td><td>94.7</td><td>75.1</td><td>94.9</td><td>68.5</td><td>72.1</td><td>59.4</td><td>62.9</td><td>74.5</td><td>77.2</td><td>63.9</td><td>68.1</td></tr><tr><td>Ours, noise init</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td>19.7</td><td>38.4</td><td>41.3</td><td>66.3</td><td>54.3</td><td>61.2</td><td>55.3</td><td>58.7</td><td>69.6</td><td>71.2</td><td>59.8</td><td>66.5</td></tr><tr><td>MoCo-v3</td><td>61.2</td><td>86.9</td><td>(91.9)</td><td>(97.9)</td><td>64.3</td><td>68.4</td><td>(60.2)</td><td>(62.5)</td><td>74.6</td><td>74.6</td><td>(66.6)</td><td>(69.9)</td></tr><tr><td>DINOv3</td><td>77.0</td><td>93.2</td><td>83.4</td><td>96.3</td><td>69.9</td><td>72.5</td><td>61.1</td><td>63.2</td><td>77.5</td><td>78.6</td><td>66.3</td><td>70.3</td></tr><tr><td>Ours, real init</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-18</td><td>30.0</td><td>57.5</td><td>54.9</td><td>82.0</td><td>62.1</td><td>68.3</td><td>58.0</td><td>61.3</td><td>72.8</td><td>74.5</td><td>63.1</td><td>67.8</td></tr><tr><td>MoCo-v3</td><td>60.6</td><td>88.5</td><td>(91.3)</td><td>(98.0)</td><td>65.0</td><td>69.3</td><td>(60.2)</td><td>(62.2)</td><td>75.0</td><td>75.8</td><td>(66.4)</td><td>(70.0)</td></tr><tr><td>DINOv3</td><td>79.9</td><td>95.5</td><td>85.1</td><td>97.5</td><td>69.9</td><td>72.8</td><td>60.3</td><td>63.0</td><td>77.0</td><td>78.5</td><td>65.7</td><td>70.2</td></tr></table>

Table B.3: Feature-Space Comparison with Diffusion-Based Methods. Unlike in the from-scratch regime, our method (DINOv3 [53] variant) consistently outperforms the diffusion-based methods across all feature-space metrics. This is quite remarkable given that our method does not use a generative model to constrain the distilled data to the real image manifold.

<table><tr><td></td><td colspan="2">Hard-Label</td><td colspan="2">Knowledge Distillation</td></tr><tr><td>Method</td><td>IPC 10</td><td>IPC 50</td><td>IPC 10</td><td>IPC 50</td></tr><tr><td>Noise init</td><td></td><td></td><td></td><td></td></tr><tr><td>SRe²L [72]</td><td> $1 0 . 6 \pm 0 . 3$ </td><td> $1 5 . 9 \pm 0 . 7$ </td><td> $7 7 . 6 \pm 1 . 0$ </td><td> $9 1 . 8 \pm 0 . 3$ </td></tr><tr><td>CDA [71]</td><td> $1 7 . 3 \pm 0 . 8$ </td><td> $3 2 . 7 \pm 0 . 3$ </td><td> $7 5 . 7 \pm 2 . 2$ </td><td> $9 1 . 0 \pm 0 . 3$ </td></tr><tr><td>G-VBSM [49]</td><td> $2 3 . 4 \pm 1 . 2$ </td><td> $2 1 . 5 \pm 0 . 5$ </td><td> ${ \pm 2 . 8 \pm 0 . 4 }$ </td><td> ${ \bf 9 3 . 6 \pm 0 . 3 }$ </td></tr><tr><td>WMDD [35]</td><td> $4 2 . 4 \pm 1 . 2$ </td><td> $5 5 . 7 \pm 0 . 1$ </td><td> $8 1 . 2 \pm 0 . 7$ </td><td> $9 2 . 4 \pm 0 . 1$ </td></tr><tr><td>Ours (ResNet-18)</td><td> ${ \pm } 6 . 2 \pm 0 . 4$ </td><td> $\underline { { 7 0 . 7 } } \pm 0 . 7$ </td><td> ${ \bf 8 2 . 7 \pm 0 . 6 }$ </td><td> $\overline { { 9 2 . 4 } } \pm 0 . 3$ </td></tr><tr><td>Ours (DINOv3)</td><td> ${ \pm 6 . 4 \pm 0 . 8 }$ </td><td> ${ \bf 7 1 . 9 \pm 0 . 3 }$ </td><td> $7 9 . 9 \pm 0 . 3$ </td><td> $9 2 . 2 \pm 0 . 3$ </td></tr><tr><td>Real init</td><td></td><td></td><td></td><td></td></tr><tr><td>RDED [56]</td><td> $4 2 . 8 \pm 0 . 8$ </td><td> $6 4 . 6 \pm 0 . 7$ </td><td> $7 8 . 7 \pm 0 . 2$ </td><td> ${ \bf 9 2 . 8 \pm 0 . 2 }$ </td></tr><tr><td>FADRM [8]</td><td> $5 0 . 4 \pm 1 . 2$ </td><td> $7 2 . 6 \pm 0 . 5$ </td><td> ${ \bf 8 3 . 9 \pm 0 . 6 }$ </td><td> ${ \bf 9 3 . 2 \pm 0 . 3 }$ </td></tr><tr><td>Ours (ResNet-18)</td><td> ${ \bf 6 0 . 9 \pm 0 . 6 }$ </td><td> $7 4 . 7 \pm 1 . 1$ </td><td> ${ \pm } \ : 0 . 5 \pm 0 . 9$ </td><td> ${ \bf 9 2 . 9 \pm 0 . 4 }$ </td></tr><tr><td>Ours (DINOv3)</td><td> ${ \bf 6 1 . 3 \pm 0 . 6 }$ </td><td> $7 5 . 9 \pm 0 . 3$ </td><td> ${ \pm 1 . 0 }$ </td><td> ${ \bf 9 3 . 1 \pm 0 . 4 }$ </td></tr><tr><td>SRe2L [72]</td><td> $1 5 . 3 \pm 0 . 3$ </td><td> $2 5 . 6 \pm 0 . 5$ </td><td> $5 6 . 7 \pm 0 . 6$ </td><td> $8 0 . 6 \pm 0 . 3$ </td></tr><tr><td>CDA [71]</td><td> $1 5 . 8 \pm 0 . 1$ </td><td> $2 4 . 2 \pm 0 . 6$ </td><td> $5 7 . 1 \pm 0 . 6$ </td><td> $8 0 . 8 \pm 0 . 2$ </td></tr><tr><td>G-VBSM [49]</td><td> $1 8 . 7 \pm 0 . 8$ </td><td> $2 6 . 8 \pm 0 . 5$ </td><td> $6 2 . 3 \pm 0 . 2$ </td><td> ${ \bf 8 4 . 1 \pm 0 . 3 }$ </td></tr><tr><td>WMDD [35]</td><td> $1 6 . 7 \pm 0 . 2$ </td><td> $2 0 . 7 \pm 0 . 9$ </td><td> $3 3 . 3 \pm 0 . 9$ </td><td> $6 5 . 9 \pm 1 . 4$ </td></tr><tr><td>Ours (ResNet-18)</td><td> ${ 3 4 . 2 \pm 1 . 0 }$ </td><td> $5 3 . 2 \pm 0 . 2$ </td><td> ${ \bf 6 5 . 6 \pm 0 . 7 }$ </td><td> $8 1 . 7 \pm 0 . 6$ </td></tr><tr><td>Ours (DINOv3)</td><td> $3 7 . 7 \pm 0 . 4$ </td><td> ${ \pm 7 . 8 \pm 0 . 8 }$ </td><td> ${ \bf 6 6 . 5 \pm 0 . 9 }$ </td><td> ${ \pm } 3 . 3 \pm 0 . 6$ </td></tr><tr><td>Real init</td><td></td><td></td><td></td><td></td></tr><tr><td>RDED [56]</td><td> $1 4 . 6 \pm 0 . 9$ </td><td> $2 7 . 2 \pm 1 . 2$ </td><td> $6 0 . 4 \pm 0 . 7$ </td><td> ${ \pm } \ : 0 . 7 \pm 0 . 5$ </td></tr><tr><td>FADRM [8]</td><td> $2 5 . 0 \pm 0 . 8$ </td><td> $4 1 . 6 \pm 0 . 6$ </td><td> $6 1 . 2 \pm 2 . 2$ </td><td> $8 1 . 9 \pm 0 . 6$ </td></tr><tr><td>Ours (ResNet-18)</td><td> ${ \bf 4 0 . 1 \pm 0 . 8 }$ </td><td> ${ \underline { { 5 6 . 6 } } } \pm 1 . 5$ </td><td> ${ \bf 6 6 . 3 \pm 1 . 2 }$ </td><td> $8 2 . 1 \pm 0 . 4$ </td></tr><tr><td>Ours (DINOv3)</td><td> ${ \bf 4 0 . 3 \pm 0 . 7 }$ </td><td> ${ \pm \ : 9 . 5 \pm 1 . 0 }$ </td><td> ${ \bf 6 8 . 4 \pm 1 . 4 }$ </td><td> ${ \bf 8 4 . 3 \pm 0 . 3 }$ </td></tr></table>

Table B.4: ImageNette From-Scratch Training. On the “easy” 10-class ImageNette [19] dataset, we outperform the baselines in the hard-label setting and match or beat them in the knowledge distillation regime.

Table B.5: ImageWoof From-Scratch Training. On the “hard” 10-class ImageWoof [19] dataset, we also outperform the baselines in the hard-label setting and match or beat them in the knowledge distillation regime.

<table><tr><td rowspan=1 colspan=8>Feature spaceDistilled set    ViT   CLIP  SigLIP2 EVA-02 MoCo-v3  EuPE  DINOv3</td></tr><tr><td rowspan=1 colspan=1>ViT [16]</td><td rowspan=1 colspan=1>(94.6)</td><td rowspan=1 colspan=1>59.9</td><td rowspan=1 colspan=1>50.6</td><td rowspan=1 colspan=1>57.2</td><td rowspan=1 colspan=1>69.9</td><td rowspan=1 colspan=1>41.7</td><td rowspan=1 colspan=1>39.6</td></tr><tr><td rowspan=1 colspan=1>CLIP [45]</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>(77.2)</td><td rowspan=1 colspan=1>50.4</td><td rowspan=1 colspan=1>59.1</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>41.3</td><td rowspan=1 colspan=1>28.6</td></tr><tr><td rowspan=2 colspan=1>SigLIP2 [60]EVA-02 [18]</td><td rowspan=1 colspan=1>73.5</td><td rowspan=1 colspan=1>84.7</td><td rowspan=1 colspan=1>(78.0)</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>56.9</td><td rowspan=1 colspan=1>71.5</td><td rowspan=1 colspan=1>56.1</td></tr><tr><td rowspan=1 colspan=1>76.2</td><td rowspan=1 colspan=1>86.2</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>(91.8)</td><td rowspan=1 colspan=1>58.7</td><td rowspan=1 colspan=1>76.5</td><td rowspan=1 colspan=1>64.6</td></tr><tr><td rowspan=1 colspan=1>MoCo-v3 [7]</td><td rowspan=1 colspan=1>89.5</td><td rowspan=1 colspan=1>72.2</td><td rowspan=1 colspan=1>61.2</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>(91.9)</td><td rowspan=1 colspan=1>52.3</td><td rowspan=1 colspan=1>48.2</td></tr><tr><td rowspan=1 colspan=1>EuPE [79]</td><td rowspan=1 colspan=1>82.1</td><td rowspan=1 colspan=1>85.5</td><td rowspan=1 colspan=1>75.8</td><td rowspan=1 colspan=1>90.4</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>(81.8)</td><td rowspan=1 colspan=1>67.3</td></tr><tr><td rowspan=1 colspan=1>DINOv3 [53]</td><td rowspan=1 colspan=1>82.8</td><td rowspan=1 colspan=1>86.1</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>89.6</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>84.1</td><td rowspan=1 colspan=1>(95.1)</td></tr></table>

Table B.6: Inter-Model Coverage. Coverage of our noise-initialized ImageNet-1k distilled sets (IPC 10), distilled with the backbone in each row and evaluated in the feature space of the backbone in each column. Some encoders yield data that generalize very well (e.g., DINOv3) while data yielded by other encoders seem very overfit to the source model (e.g., CLIP). Same-model performance is indicated by (parentheses).

## C More Method Details

Note on from-scratch evaluation: As pointed out in recent work [14], the most popular large-scale dataset distillation evaluation method effectively changes the task to knowledge distillation rather than training from scratch. While knowledge distillation is an interesting task deserving of its own attention, we wish to focus on from-scratch training, the original goal of dataset distillation. For our soft-label evaluations, rather than generating new soft labels at every step, we use a teacher model to generate exactly one soft label per image before training by averaging its temperature-softened predictions over 10 random crops. We re-evaluate prior works under this training paradigm as well.

## Distillation

For our real-initialized variants, we initialize the distilled images with those of RDED [56] and distill for 5000 iterations. Our noise-initialized variants start from random noise and are distilled for 10000 iterations. We use all real data as positive samples at every step (this is feasible even on small GPUs with ∼1.3k real embeddings per class).

To optimize our distilled data, we use Adam [31] with a learning rate of 0.001.

Images are distilled at $2 5 6 \times 2 5 6$ resolution. Image pyramids are initialized with the first 4 layers (up to $8 \times 8 )$ as standard Gaussian noise. A new layer is added every 200 iterations (as standard Gaussian noise) until the full resolution is reached.

Videos are distilled at $2 5 6 \times 2 5 6$ spatial resolution and 16 frames. Video pyramids are initialized with the first 4 layers and 1 frame as standard Gaussian noise. A new layer with twice the spatial and temporal resolution is added every 200 iterations (as standard Gaussian noise) until the full resolution is reached.

Audio pyramids are initialized as just the first layer. A new layer is added every 100 steps until the final resolution of 524,288 samples is reached.

Drifting field implementation. For clarity, Section 3.2 presents the drifting field in its simplest form. In practice, we follow the full implementation of [13]: the kernel is additionally normalized over the synthetic-sample axis (their Algorithm 2), and we apply the feature and drift normalizations of their Appendix A.6. Concretely, each feature $\phi _ { j }$ is rescaled by a stop-gradient scale $S _ { j }$ such that the average pairwise distance is approximately $\sqrt { C _ { j } }$ (where $C _ { j }$ is the feature dimension), the kernel temperature is correspondingly scaled to $\tau _ { \sqrt { \bigcirc _ { j } } }$ , and the resulting field $\mathbf { V } _ { j }$ is divided by its root-mean-square per-dimension magnitude. A single normalization scale is shared across all spatial locations of a given feature map, and the total loss is the sum of the per-feature losses.

Features. The feature extractor ϕ yields several features $\phi _ { j }$ per sample. For ViT backbones, we use the CLS token together with all patch tokens. For ResNet-18, we use the final global-average-pooled vector together with the per-location feature vectors of the final two convolutional layers. For both audio and video methods, we drift the mean token at each layer.

## D Evaluation Protocols

All from-scratch evaluations train a ResNet-18 [26] student and report top-1 accuracy of the finalepoch model on the ImageNet-1k validation set (single 224 × 224 center crop after resizing to 256). All numbers are the mean and standard deviation over 3 training seeds. Every method, including ours, is re-evaluated with identical recipes.

## D.1 Hard-Label Training

On ImageNet-1k [12], hard-label training uses the soft-label recipe below with $\alpha = 0 ( \mathrm { i . e . }$ ., only the label-smoothed cross-entropy on the one-hot labels), so the two settings differ only in the targets.

On ImageNette and ImageWoof [19], we use the evaluation protocol from Minimax Diffusion [23]:

• net\_type = resnet

$$
\cdot \ \mathrm { d e p t h } \ = \ 1 8
$$

```python
• norm_type = instance
• optimizer = SGD
• lr = 0.01
• momentum = 0.9
• weight_decay = 5e-4
• epochs (E)
– IPC = 1 → E = 3000
– IPC = 10 → E = 2000
– IPC = 50 → E = 1500
• lr_schedule = MultiStepLR
– milestones = [2·E/3, 5·E/6]
– gamma = 0.2
• batch_size = 64
```

## D.2 Soft-Label Training

We implement our own recipe for soft-label training and apply it equally to our method and all baselines. Before training, a pre-trained ResNet-18 teacher (torchvision weights) computes exactly one soft label per distilled image: the average of its temperature-scaled softmax outputs (T = 10) over 10 random-resized crops (scale [0.3, 1]) with horizontal flips. The student is trained with

$$
\mathcal { L } = \alpha T ^ { 2 } \mathrm { C E } ( \mathrm { s o f t m a x } ( z / T ) , y \mathrm { s o f t } ) + ( 1 - \alpha ) \mathrm { C E } _ { \mathrm { l s } = 0 . 1 } ( z , y _ { \mathrm { h a r d } } )
$$

where z are the student logits and α = 0.9. Other hyperparameters are as follows:

• student = resnet18 (batch norm)   
• optimizer = AdamW   
• lr = 5e-4   
• lr\_schedule = 20-epoch linear warmup, then cosine to 1e-5   
• weight\_decay = 0.5 → 0.1 (cosine)   
• epochs = 1000   
• batch\_size = 50   
• ema\_decay = 0.9995   
• augmentation   
– RandomResizedCrop(min\_scale=0.3)   
– RandomHorizontalFlip()   
– TrivialAugmentWide()   
– MixUp(alpha=0.2, p=0.5)

## D.3 Knowledge Distillation

Rather than from-scratch training, most existing works on large-scale dataset distillation (i.e., ImageNet-1k for ResNet-18) have evaluated using a knowledge distillation protocol. Rather than using a (hard or soft) fixed label for each distilled image, the knowledge distillation protocol uses a teacher model to recompute a new label at every training step using a newly augmented version of the image. For our knowledge distillation evaluations, we use the recipe defined in EDC [50].

The method uses a set of M teacher models $f _ { m }$ to compute a soft target for each augmented sample x as the average of each model’s raw output logits:

$$
y _ { \mathrm { s o f t } } = \frac { 1 } { M } \sum _ { m } f _ { m } ( x )
$$

The total knowledge distillation loss is then

$$
\mathcal { L } _ { \mathrm { K D } } = \mathrm { M S E } ( z , y _ { \mathrm { s o f t } } ) + \lambda _ { \mathrm { C E } } \mathrm { C E } ( z , y _ { \mathrm { h a r d } } )
$$

where z are the output logits of the student model, MSE is the mean-squared-error loss, CE is the cross-entropy loss, $y _ { \mathrm { h a r d } }$ is the class-specified one-hot label, and λ<sub>CE</sub> is the cross-entropy weight.

Other hyperparameters are as follows:

• student = resnet18   
• teachers   
– ImageNet-1k → 4-model EDC ensemble   
resnet18   
mobilenet\_v2   
\* <sup>shufflenet\_v2\_x0\_5</sup>   
\* <sup>alexnet</sup>   
– ImageNette/ImageWoof → resnet18   
• ce\_weight = 0.025   
• optimizer = AdamW   
• lr = 1e-3   
• weight\_decay = 0.01   
• lr\_schedule = cosine   
• lr\_schedule\_deceleration = 2   
• ema\_decay = 0.99   
• augmentation   
– RandomResizedCrop(min\_scale=0.5)   
– RandomHorizontalFlip()   
– RandAugment(n=2, m=9) (ImageNette and ImageWoof only)   
– CutMix(alpha=1.0)   
• batch\_size   
– ImageNet-1k → 100   
– ImageNette/ImageWoof → 50   
• epochs (E)   
– ImageNet-1k → E = 300   
– ImageNette/ImageWoof → E = 1000

## D.4 Coverage

For a set of distilled points, we compute the coverage [41] by measuring the fraction of real test set points that have at least one distilled point within the ball defined by their k-th nearest real neighbor in the latent space of some encoder. We use cosine distance and k = 5 by default. We compute the coverage per class and report the average. The pool of distilled points includes each distilled image’s center crop together with 10 random-resized crops (scale [0.08, 1]) per image.

## D.5 k-Nearest Neighbors

We compute the k-Nearest Neighbor (k-NN) performance of a distilled dataset in the typical fashion; for each real test point, we compute the k nearest distilled points (in the space of some encoder) and predict the most common label among those k neighbors. By default, we use cosine distance and k = 1. As with coverage, the pool of distilled points includes each distilled image’s center crop together with 10 random-resized crops (scale [0.08, 1]) per image.

## D.6 Linear Classification

We train a linear classifier on frozen, unnormalized features of an augmented bank: each distilled image’s center crop together with 10 random-resized crops (scale [0.7, 1]), sampled uniformly. We use the following parameters:

• optimizer = AdamW   
• lr = 1e-3   
• lr\_schedule = cosine annealing to 0 over all steps (stepped   
per step)   
• weight\_decay = 0   
• batch\_size = 256 (sampled uniformly with replacement)   
• steps: IPC = 10 → 4000, IPC = 50 → 19600

## E Algorithms

```latex
Algorithm 1 Distillation loss for one class (one optimization step)
Require: Pyramids of the synthetic samples $\{ \mathbf { x } _ { s } \} _ { s = 1 } ^ { | \mathcal { S } | }$ , real images of the class $\{ \mathbf { y } _ { k } \} _ { k = 1 } ^ { M }$ , features
$\{ \phi _ { j } \}$ , augmentation distribution $P _ { A }$ , augmentations per step n
1: Decode each $\mathbf { x } _ { s }$ from its pyramid ▷ Eq. 9
2: Sample $A _ { 1 } , \dots , A _ { n } \sim \bar { P _ { A } }$
3: for each feature $\phi _ { j }$ do
4: $\tilde { \mathbf { x } } _ { a , s }  \phi _ { j } ( A _ { a } ( \mathbf { \bar { x } } _ { s } ) )$ for all $a , s$ $> N = n | { \cal S } |$ embeddings
5: $\tilde { \mathbf { y } } _ { k }  \phi _ { j } ( \mathbf { y } _ { k } )$ ▷ un-augmented; precomputed
6: V, S ← DRIFTFIELD $\big ( \mathrm { s t o p g r a d } ( \tilde { \mathbf { x } } ) , \tilde { \mathbf { y } } \big )$ ▷ Algorithm 2
7: $\begin{array} { r } { \mathcal { L } _ { j }  \mathrm { M S E } ( \tilde { \mathbf { x } } / S ,  } \end{array}$ stopgrad(x˜/S + V)
8: end for
9: return $\begin{array} { r } { \mathcal { L } = \sum _ { j } \mathcal { L } _ { j } } \end{array}$ ▷ back-propagated to the pyramids
```

Algorithm 2 DRIFTFIELD: the drifting field for one feature $\phi _ { j } ,$ following Alg. 2 and App. A.6 of [13].   
All steps are applied independently at each spatial location of $\phi _ { j } ;$ only S and $\lambda _ { \tau }$ are computed jointly   
over all locations.   
Require: Synthetic embeddings $\{ \tilde { \mathbf { x } } _ { i } \} _ { i = 1 } ^ { N }$ (detached), real embeddings $\{ \tilde { \mathbf { y } } _ { k } \} _ { k = 1 } ^ { M }$ , feature dimension   
C, temperatures $\tau$   
Ensure: Normalized drifting field $\mathbf { V } ( \tilde { \mathbf { x } } _ { i } )$ for $i = 1 , \ldots , N$ , feature scale S   
1: $d _ { i k } ^ { + }  \| \tilde { \mathbf { x } } _ { i } - \tilde { \mathbf { y } } _ { k } \| _ { 2 } , \quad d _ { i l } ^ { - }  \| \tilde { \mathbf { x } } _ { i } - \tilde { \mathbf { x } } _ { l } \| _ { 2 }$   
2: $S \gets \mathsf { s t o p g r a d } \big ( \operatorname* { m e a n } _ { \mathbf { \mu } \sim \mathbf { \mu } } ( \{ d _ { i k } ^ { + } \} \cup \{ d _ { i l } ^ { - } \} ) \big ) / \sqrt { C }$ ▷ feature normalization   
3: Divide all $d , { \tilde { \mathbf { x } } } ,$ and y˜ by S   
4: $d _ { i i } ^ { - }  \infty$ ▷ mask self-pairs   
5: $\mathbf { V } ( \tilde { \mathbf { x } } _ { i } ) \gets \mathbf { 0 }$   
6: for $\tau \in \mathcal { T }$ do   
7: $\ell _ { i , \cdot } \gets - \big [ d _ { i , 1 : M } ^ { + } \big | \big | d _ { i , 1 : N } ^ { - } \big ] / ( \tau \sqrt { C } )$ ▷ logits over all targets   
8: $A  \sqrt { }$ softmax<sub>targets</sub>(ℓ) ⊙ softmax<sub>i</sub>(ℓ) ▷ normalize over both axes   
9: Split $\overset { \cdot } { A } = \left[ A ^ { + } \parallel \overset { . } { A ^ { - } } \right]$ into positive and negative columns   
10: $\begin{array} { r } { W _ { i k } ^ { + }  A _ { i k } ^ { + } \sum _ { l } A _ { i l } ^ { - } , ~ W _ { i l } ^ { - }  A _ { i l } ^ { - } \sum _ { k } A _ { i k } ^ { + } } \end{array}$   
11: $\begin{array} { r } { \mathbf { V } _ { \tau } ( \tilde { \mathbf { x } } _ { i } )  \sum _ { k } W _ { i k } ^ { + } \tilde { \mathbf { y } } _ { k } - \sum _ { l } W _ { i l } ^ { - } \tilde { \mathbf { x } } _ { l } } \end{array}$   
12: $\lambda _ { \tau }  \sqrt { \mathrm { m e a n } _ { i } \| \mathbf { V } _ { \tau } ( \tilde { \mathbf { x } } _ { i } ) \| _ { 2 } ^ { 2 } / C }$ ▷ drift normalization   
13: $\mathbf { V } ( \tilde { \mathbf { x } } _ { i } ) \dot { {  } } \mathbf { V } ( \tilde { \mathbf { x } } _ { i } ) \dot { {  } } \mathbf { V } _ { \tau } ( \tilde { \mathbf { x } } _ { i } ) / \lambda _ { \tau }$   
14: end for   
15: return $\{ \mathbf { V } ( \tilde { \mathbf { x } } _ { i } ) \} _ { i = 1 } ^ { N } , S$

## F Limitations, Compute, Broader Impacts, and Licenses

## Limitations

Our largest limitation is the time needed to distill the data. For 10 images per class, each class takes ∼0.25 GPU hours (more or less depending on GPU model), corresponding to roughly 250 GPU hours for ImageNet-1k before accounting for hyperparameter sweeps, multiple backbones, and larger IPC settings. Although distilling each class separately makes the process easy to parallelize, the total compute cost is significant.

## Compute Resources

We ran our experiments on a shared cluster using whatever GPUs we could get. Used GPUs include RTX 3090s, RTX 4090s, L40Ss, V100s, A100s, H100s, and H200s. All image experiments could comfortably fit on a 24 GB GPU. Audio and video experiments required an A100, H100, or H200.

## Broader Impacts

Our method does not come with any readily apparent negative effects. Advances in dataset distillation may, in fact, lead to more efficient training, thereby shrinking the AI industry’s carbon footprint.

## Licenses

Datasets. For ImageNet [12] we build on the Torchvision [58] implementation as part of the PyTorch [1, 43] ecosystem. ImageNet has a unique non-commercial license, to which we have agreed. The remaining datasets are used as follows:

• ImageNette and ImageWoof [19]: the official imagenette2 and imagewoof2 releases (Apache-2.0); both are subsets of ImageNet and are also subject to the ImageNet terms above.

• FMA [11]: the fma\_small subset. Its metadata is released under CC BY 4.0, and each audio track is distributed under the Creative Commons license chosen by its artist; the dataset is intended for research use.

• HMDB51 [32]: released under CC BY 4.0.

• Paintings: the Claude Monet and Qi Baishi paintings in Section 4.5 were collected from WikiArt, the Art Institute of Chicago, and the Metropolitan Museum of Art. Images from the Art Institute of Chicago and the Met are public-domain works released under CC0; WikiArt presents public-domain works and, under a fair-use rationale, copyrighted works. Monet’s paintings are in the public domain. We use all paintings solely for non-commercial research.

Models. All backbones are used with their publicly released pre-trained weights and are kept frozen.

• ResNet-18 [26] and ViT-B/16 [16]: Torchvision [58] IMAGENET1K\_V1 weights (BSD-3- Clause). The ResNet-18, MobileNetV2, ShuffleNetV2, and AlexNet teachers used for soft labels and knowledge distillation also come from Torchvision (BSD-3-Clause).

• CLIP [45]: openai/clip-vit-base-patch16 on Hugging Face [30], released by OpenAI under the MIT license.

• SigLIP2 [60]: google/siglip2-base-patch16-224 (Apache-2.0).

• EVA-02 [18]: eva02\_base\_patch14\_224.mim\_in22k from PyTorch Image Models [67] (MIT).

• MoCo-v3 [7]: ViT-B weights loaded from nyu-visionx/moco-v3-vit-b; the official MoCo-v3 release is licensed under CC BY-NC 4.0.

• EuPE [79]: facebook/EUPE-ViT-B (FAIR Noncommercial Research License).

• DINOv3 [53]: facebook/dinov3-vitb16-pretrain-lvd1689m, along with the ViT-S/16 and ViT-L/16 variants for the model-size ablation (DINOv3 License).

• VideoMAE-v2 [63]: OpenGVLab/VideoMAEv2-Base (CC BY-NC 4.0).

• MusicFM [68]: the FMA checkpoint from minzwon/MusicFM (MIT).