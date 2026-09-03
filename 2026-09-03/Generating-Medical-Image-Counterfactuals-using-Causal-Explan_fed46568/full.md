# Generating Medical Image Counterfactuals using Causal Explanations

David A. Kelly<sup>∗,1</sup> , Tom Yaacov<sup>∗,1</sup> , Nathan Blake<sup>1</sup> , Sander Beckers<sup>2</sup> , and Hana Chockler<sup>1</sup>

## Abstract

Deep learning models have achieved impressive performance in medical image diagnosis, yet their deployment in clinical settings remains constrained by limited explainability. Counterfactual images provide one means of auditing model behavior by showing how an image would need to changefor a classifier to produce a different prediction. Existing approaches typically generate such explanations using auxiliary models, including generative adversarial networks and diffusion models. While often capable of producing visually realistic images, these methods explain one black-box model using another, making it difficult to separate the classifier’s decision-making process from the inductive biases of the generator.

We propose a novel counterfactual-generation framework that requires no generative model. Instead, counterfactuals are constructed directly from causal evidence ex tracted from the classifier. The resulting approach is deterministic, requires no additional model training, and enables controllable edits within user-specified regions of interest. Experiments on real-world medical imaging datasets demonstrate that the proposed method successfully changes classifier predictions while remaining closer to the original image than generative baselines, providing a more direct and transparent view of the classifier’s decision boundary.

## 1. Introduction

Deep image classifiers achieve strong performance across many medical-imaging tasks, yet their use remains limited by poor generalization, dataset shift, and a lack of transparent explanations [6]. These concerns are not unique to healthcare: they reflect a broader challenge in computer vision, where high-capacity models may produce accurate predictions while relying on features that are difficult for domain experts to inspect. In medical imaging, the cost of such opacity is especially acute. A clinician or model auditor may need to know not only where a classifier attends, but also what visual evidence would have changed its decision.

Counterfactual explanations address this question by asking how an input image would need to change for a classifier to assign it to a different class [28]. In medical imaging, this often takes the form of transforming an image classified as healthy into one classified as diseased. Such ‘what-if’ images can help users probe the decision boundary of a model and identify its modes of failure. This differs from many post-hoc explainable AI (XAI) methods such as SHAP [20], LIME [25], or ReX [10], which identify those features responsible for a given classification. These attribution methods ask which features support the current prediction; counterfactual generation asks what would need to change for the prediction to change. This is not the same as simulating plausible pathological counterfactuals, but rather a model-specific audit. A small non-pathological alteration may induce a class to change to ’disease’: such a discrepancy is precisely what the method seeks to expose.

Most existing approaches to image counterfactual generation rely on generative models, including GANs, conditional GANs, and diffusion models [19, 21, 22, 26, 34]. These methods synthesize plausible samples from a target class and have been widely adopted in medical imaging because they can model complex anatomical and pathological variation [26]. However, they introduce a fundamenta challenge for explainability: the explanation of one blackbox model is produced by another. As a result, it can be difficult to determine whether a counterfactual reflects the classifier’s decision boundary, the generative model’s image prior, or artifacts of the optimization procedure. Generative approaches are also typically data-intensive and sensitive to training choices, which is problematic in medical imaging where large, well-annotated datasets are often scarce [11]. For model auditing, the most informative counterfactual is not necessarily the most realistic target-class image, but a close image that the classifier itself accepts as belonging to the target class.

In this paper, we introduce a non-generative approach to image counterfactual generation. Instead of learning a separate image-generation process, we directly exploi the classifier’s own causal explanations (Section 2.1) to construct targeted counterfactual perturbations. We precompute causal explanations for positively classified images, retrieve explanations from images close to the target instance, and partially transfer the associated causal pixel values to generate candidate counterfactuals. Consequently, the search space is constrained entirely by evidence that the classifier has already identified as causally relevant, rather than by a learned image prior.

![](images/d171c94fb37a4926c456598048e3643a51e0784ce8f091f963f22786b8c37dac.jpg)

To our knowledge, this is the first counterfactualgeneration framework that directly reuses causal explanation. Our method offers three advantages for visual model auditing. First, it requires no additional trained models, isolating the behavior of the classifier under inspection and eliminating generator-induced confounding effects. Second, the procedure is deterministic, avoiding the stochasticity common to training and sampling from generative models. Third, perturbations are local and controllable: users can restrict the search to a region of interest and directly test whether modifying a specific anatomical structure is sufficient to alter the classifier’s prediction (Figure 1).

Conceptually, our work reframes counterfactual generation as a causal-retrieval problem rather than a generativemodeling problem. Instead of learning how to synthesize target-class images, we identify and transfer causal evidence already used by the classifier when making positive predictions. This produces counterfactuals that are inherently model-specific: every modification can be traced to a previously observed causal explanation. As a result, the generated images provide a more direct view of the classifier’s decision boundary and avoid the ambiguity introduced by auxiliary generative models.

Our principle contributions are:

• The first counterfactual framework built directly from causal explanations.

• Counterfactuals are constrained by classifier-derived causal evidence rather than image realism priors.

• Every perturbation is traceable to an identified causal contribution.

• Counterfactual generation becomes a model-auditing procedure rather than an image-synthesis procedure.

The complete code for the evaluation and the implementation are available in [3].

## 2. Background

This section introduces the necessary background that is developed in greater detail in [16]. The general idea is that we can interpret a classifier as a causal model [15, 23], where the (causally) independent pixels X<sup>⃗</sup> jointly cause the output classification O via the causal mechanism implemented by the classifier N. Doing so allows us to invoke well-defined notions of causal explanations that are grounded in the theory of actual causality [4, 15]. We briefly review a simplified version of the relevant concepts and refer the reader to [16] for a comprehensive treatment.

Figure 1. A clinician may be surprised to find Figure 1a classed as ‘no tumor’. In fact, there is a tumor present – Figure 1b – that the model has failed to detect. Our approach generates an approximately minimal counterfactual, where ‘minimal’ stands for a smallest change to the smallest number of pixels.

## 2.1. Causal Explanations

Let the set of variables $\vec { X }$ correspond to the set of pixels of an image, so that any setting ${ \vec { X } } = { \vec { x } }$ corresponds to a specific image. Let O be the output variable of a classifier ${ \mathcal { N } } .$ , so that $\mathcal { N } ( \vec { x } ) = o$ means the image ${ \vec { X } } = { \vec { x } }$ is classified as belonging to the class $O = o$

In order to derive sufficient explanations of the classification, we evaluate the behavior of the classifier on images that are the result of placing a mask onto the original image. For each $X _ { i } ~ \in ~ \vec { X }$ , we let the Boolean variable $V _ { i }$ denote whether the corresponding pixel is masked $- \ V _ { i } = 1 - \mathrm { o r }$ takes on its original pixel value $- V _ { i } = 0$ . A mask is implemented using a predefined masking value, or baseline value.

Given a masking image $\vec { V } = \vec { v }$ and an original image ${ \vec { X } } = { \vec { x } } .$ , placing the mask onto the original image results in the Hadamard product of ${ \vec { X } } = { \vec { x } }$ and ${ \vec { V } } = { \vec { v } } ,$ denoted as $\vec { v } \odot \vec { x }$ . If the masked image does not alter the classification, then the unmasked part of the original image suffices for the classification. For it to also explain the classification, the unmasked part should not contain any redundant pixels, and thus we also demand that it is minimal. Concretely, we define sufficient explanations as follows.

Definition 1 (Sufficient Explanation (MSPS) [10]). Given an input image ${ \vec { X } } = { \vec { x } }$ and a classifier N , we say that a subset $\vec { X } _ { e x p } = \vec { x } _ { e x p }$ of ${ \vec { X } } = { \vec { x } }$ is a sufficient explanation of the classification $\mathcal { N } ( \vec { x } )$ if the following conditions hold:

EXIM1. $\mathcal { N } ( \vec { v } _ { - e x p } \odot \vec { x } ) = \mathcal { N } ( \vec { x } )$

EXIM2. $\vec { X } _ { e x p }$ is minimal; there is no strict subset $\vec { X } _ { e x p } ^ { \prime }$ of $\vec { X } _ { e x p }$ that satisfies EXIM1.

Here $\vec { V } = \vec { v } _ { - e x p }$ is the mask over all pixels not in $\vec { X } _ { e x p }$

In short, sufficient explanations are simply Minimal Sufficient Pixel $S e t s - \mathbf { M S P S }$

[16] also introduce the stronger notion of complete explanations for image classifiers. Informally, a complete explanation is an explanation that is both sufficient and necessary, and minimally so.

Definition 2 (Complete Explanation). Given an input image $\vec { X } \ = \ \vec { x }$ and a classifier ${ \mathcal { N } } .$ , we say that a subset $\vec { X } _ { e x p } = \vec { x } _ { e x p }$ of ${ \vec { X } } = { \vec { x } }$ is a complete explanation of the classification $\mathcal { N } ( \vec { x } )$ if it satisfies EXIM1 and the following conditions hold:

EXN1. $\mathcal { N } ( \vec { v } _ { e x p } \odot \vec { x } ) \ne \mathcal { N } ( \vec { x } )$

EXN2. $\vec { X } _ { e x p }$ is minimal; there is no strict subset of $\vec { X } _ { e x p }$ that satisfies both EXIM1 and EXN1.

## 2.2. ReX

We use the tool ReX [10] to generate the causal explanations just described. ReX is an XAI tool based in the theory of actual causality [4, 15]. We use ReX because, unlike other XAI tools, can provide us with causal explanations directly. We provide here a brief overview of the algorithm and direct the interested reader to [10] for full details.

ReX starts by dividing an image into 4 parts. We call each part a superpixel. ReX creates mutants of the original image by covering all combinations of superpixels with a baseline value (typically 0). These combinations are tested against the model and sorted between those which satisfy the required classification and those which do not. Causal responsibility is distributed over the different superpixels, where responsibility is a quantitative measure of causality and, broadly speaking, measures the amount of causal influence on the classification [8, 10]. Passing superpixel combinations are further refined into smaller superpixel combinations and tested using the same procedure.

In this way, mutant generation is iteratively guided by the model, and responsibility tends to concentrate on those pixels which occur most frequently in class preserving combinations. This is repeated many times, starting from a different random partition of the image. This procedure provides a ranking of pixels based on their responsibility. The pixels are ranked in the order of their responsibility for the classification and a causal explanation, or a minimal sufficient pixel set (MSPS) (Def. 1), is extracted from this ranking using a greedy algorithm. The number of calls ReX makes to the model to calculate initial causal explanations is $O ( 2 ^ { s } n N )$ , where s is the size of the partition in each step (by default $s = 4 )$ , n is the number of pixels in the original image $x ,$ and N is the number of initial partitions [10]. ReX is also capable of discovering multiple explanations [9] but these do not occur in our datasets.

ReX also computes complete explanations (Def. 2), minimal sets of pixels which are both sufficient and necessary for the original classification [16]. Recall, that this means the pixels, by themselves, are sufficient against the baseline, and, when removed from the original image, they change the classification. These are, in general, larger than sufficient explanations and contain more information. We evaluate both sufficient and complete explanations.

## 3. Causal Explanations and Counterfactual Generation

Although both sufficient and complete explanations can be applied to positively and negatively classified images alike ([17]), they are less insightful for negative classifications (which indicate the absence of a tumor). The reason is that these explanations rely on masking parts of the image to identify the location of a tumor, and thus they are less appropriate for an image where no tumor was identified to being with (see Section 5). Therefore negatively classified images require invoking a different type of well-known causal explanation, namely a counterfactual explanation. In the context of images, this means constructing counterfactual images.

Our approach to counterfactual image generation builds on the idea of viewing a classifier as a causal model, that allowed us to invoke well-defined notions of causal explanations in Section 2. A counterfactual is a claim about what would have happened if some occurrence were different, all else being equal. Concretely, it offers answers to queries of the form: “Given that we observed ${ \vec { X } } = { \vec { x } }$ that led to $O = o$ , what would O have been had some ${ \vec { X } } ^ { \prime } \subseteq { \vec { X } }$ been set to ${ \vec { x } } ^ { \prime \prime \prime }$ . Equivalently, using our previous notation, we are asking whether $\mathcal { N } ( \vec { x } ^ { \prime \prime } ) = \mathcal { \bar { N } } ( \vec { x } )$ , where ${ \vec { x } } ^ { \prime \prime }$ agrees with ${ \vec { x } } ^ { \prime }$ on ${ \vec { X ^ { \prime } } }$ and agrees with ⃗x elsewhere.

In principle, a counterfactual explanation of an image simply consists of any image that changes the classification, but the quality of a counterfactual explanation is determined by how similar, or close, the new image is to the original one [21]. Our method uses sufficient and complete explanations to systematically generate counterfactual explanations that are very close to the original image. Figure 2 shows a schematic of the workflow.

## 3.1. Problem Statement

Given an image that is originally classified as healthy, we ask the question “What minimal causal evidence does the classifier itself consider sufficient to change its prediction to diseased?”. While clinical plausibility is important when counterfactuals are intended to model real-world disease processes, it is not a prerequisite for model auditing. Auditing seeks to understand the classifier’s decision rather than the underlying pathology. By focusing on classifier-derived causal evidence rather than clinical realism, our approach provides a more direct probe of the model’s decision boundary.

![](images/1931e5d04a7bbc8638930b57f97dea9b1a8e9a8d9357bf55e63f9bec108d1d07.jpg)  
Figure 2. Our workflow: $\textcircled{1}$ an image classified as healthy; ➁ search for nearby images (mediated by some distance function); ➂ their causal explanations are pre-computed and saved in a database; ➃ extract causal values from the explanations; ➄-➅ main loop: insert causal values and query the model; finally ➆ an approximately minimal counterfactual generated by the classifier itself.

Algorithm 1 CausalCounterfactual $( \mathcal { N } , x , \mathcal { P } , \mathcal { P } _ { \exp } , d , M , t )$   
INPUT: a classifier ${ \mathcal { N } } ,$ a negative image x, a set P of   
positive images, a set $\mathcal { P } _ { \mathrm { e x p } }$ of causal explanations, one for   
each member of $\mathcal { P } _ { \cdot }$ , a distance function d, a set of masks   
M, a distance threshold value t;   
OUTPUT: a transformed image $x ^ { \prime }$ or ∅   
1: p<sup>′</sup> ← arg min<sub>p∈P</sub> d(x, p)   
2: E<sup>′</sup> ← get explanation of $\overline { { p ^ { \prime } } }$ from $\mathcal { P } _ { \mathrm { e x p } }$   
3: $\dot { V _ { p } ^ { \prime } }$ ← get actual values $E _ { p } ^ { \prime } , p ^ { \prime } )$   
4: $\dot { C _ { p } ^ { \prime } } \gets$ calculate centroid $( \dot { E } _ { p } ^ { \prime } )$   
5: for m ∈ M do   
6: x<sup>′</sup> ← apply m to x on location $C _ { p } ^ { \prime }$   
7: while $d ( x , x ^ { \prime } ) < t$ do   
8: $x ^ { \prime } \gets$ update values under mask(x<sup>′</sup>, $V _ { p } ^ { \prime } )$   
9: if $\mathcal { N } ( x ^ { \prime } ) = 1$ then   
10: $x ^ { \prime } \gets$ smooth $( x ^ { \prime } )$   
11: return $x ^ { \prime }$   
12: end if   
13: end while   
14: end for   
15: return $\varnothing$

## 3.2. Our Solution

Existing counterfactual methods explain a classifier using a second learned representation (a GAN latent space, diffusion latent space, etc.). Our method explains a classifier using evidence extracted from the classifier itself. Algorithm 1 outlines our approach. The algorithm takes as input the reference set of positive images, $\mathcal { P } _ { \cdot }$ , and its corresponding set of explanations, $\mathcal { P } _ { \exp } . ~ \mathcal { P }$ can be obtained from the dataset used to train the classifier. $\mathcal { P } _ { \mathrm { e x p } }$ is precomputed by ReX using the same model, ${ \mathcal { N } } ,$ as in the main algorithm. We start by finding, in the given set of positive images $\mathcal { P }$ , the image $p ^ { \prime }$ that is closest to the target image x (Line 1). This is mediated by a given distance function d.

We find the matching explanation for this positive image (Line 2) in $\mathcal { P } _ { e x p }$ . The explanation is given in the form of a binary mask, $E _ { p } ^ { \prime } ,$ defined over a matrix of the same size and shape as the image $p ^ { \prime }$ , and indicates which pixels are part of an explanation. We apply the mask over the positive image $p ^ { \prime }$ to obtain the actual values of the pixels selected by $E _ { p } ^ { \prime } .$ We take these actual values, remove duplicates, and sort them by their values, giving us an ordered array of values, $V _ { p } ^ { \prime } .$ . These values are what will be used to generate candidate perturbations. We need to choose a location in x as a mutation target (Line 4). For our experiments, we use the centroid location of the abnormality in $p ^ { \prime }$ , but equally, a clinician could select a location and pass this on as an additional parameter to our algorithm.

The main loop of Algorithm 1 starts on Line 5. We start by applying a user-defined binary mutation mask m over the chosen location in x, giving us $x ^ { \prime } .$ . The mutant mask can be any initial size or shape (we use a simple rectangle in our experiments), and limits which pixels can be perturbed. We move each value v under m one step closer to its closest value in $V _ { p } ^ { \prime } .$ The effect of this is to bring pixels in $x ^ { \prime }$ to pixel values known to be part of an explanation for $p ^ { \prime } .$

If this minimal shift does not change the outcome, we increase the step taken in $V _ { p } ^ { \prime } ,$ resulting in a new image with values further away from the original x, but with the same total number of altered pixels. The loop continues as long as $x ^ { \prime }$ remains within the distance threshold t from x. If we cannot change the classification of x using the mask m and values $V _ { p } ^ { \prime } ,$ we use a bigger mask $m ^ { \prime } .$ . Line 10 is an optional smoothing calculation: the successful mask, m, might introduce hard lines or unnatural shapes, depending on initial choices; ‘smooth’ attempts to reduce these artifacts. In our experiments, we start from rectangular masks and use a segmentation-based approach to smooth the outlines.

The algorithm always terminates, as both $V _ { p } ^ { \prime }$ and M are finite. In the event that no combination of mask size and value shift changes the classification, we return the empty set. The overall computational complexity is $O ( | M | \times | V | )$ Our algorithm includes a sort in Line 3, with the standard complexity $O ( | V | \log | V | )$ , but this is dominated by $O ( | M | \times | V | )$ in all but degenerate cases. The set of causal explanations, $\mathcal { P } _ { \exp } ,$ is precomputed once for a trained classifier using ReX during pre-processing, hence there is no need to compute them for each counterfactual generation. The computational complexity of running ReX is given in Section 2.2.

## 4. Evaluation

This section evaluates our approach for generating counterfactual explanations in two case studies. The evaluation concentrates on the two main criteria for counterfactual images: 1) Correctness: the classifier should predict that the counterfactual image belongs to a different class than the original. 2) Similarity: the counterfactual should be as similar as possible to the original image. For the distance function d in our algorithm implementation, we used the L2 norm. To compute the centroid of the explanation mask, we took the median location of the explanation. Our smoothing operator uses the SLIC segmentation algorithm [1]. The complete code for the evaluation and the implementation of our approach, including details on the data preprocessing, are available in [3].

## 4.1. Case Studies

The first case study was a dataset of brain MRI data obtained from The Cancer Imaging Archive [7]. This curated dataset of 110 pre-operative patients with low grade gliomas (LGG) was gathered from 5 US institutions. All the images are axial slices of the brain and include fluid-attenuatedinversion recovery (FLAIR) sequence, while 101 also have pre-contrast sequence and 104 have post-contrast sequence images. Each patient had between 20 and 88 slices taken, with a total of 3, 929 images. Each image contains 3 channels, one for each of the three sequences. Following [18], we used their pre-trained CNN classifier based on the ResNet50 architecture and did not perform any additional classifier training. Consistent with their experimental setup, the dataset was split at the patient level, with 80% of patients used for training and 20% for testing.

The second case study was a dataset of skin lesion images from The International Skin Imaging Collaboration (ISIC) archive [14]. The dataset includes a representative set of both malignant and benign skin lesion images. The dataset contains 1250 images, partitioned into train and test before release. For the classifier, we took a trained attention-based CNN based on the VGG architecture from [31], and did not perform any additional classifier training.

In our evaluation, we considered all images that the classifier predicted as negative, including both true and false negatives. This choice reflected the intended clinical use, where a clinician seeks an explanation of the model, where the ground-truth is not yet established. This yielded 560 tested images for the brain MRI dataset and 277 for the ISIC skin lesion dataset.

## 4.2. Evaluation Metrics

A counterfactual image explains a classification only if it remains sufficiently similar, or close, to the original image. A sensible distance metric should take into account two dimensions: the set of pixels on which they disagree, and/or the disagreement between the pixel values. The first dimension gives us local explanations, allowing the identification of a specific, small, part of the input that is responsible for the change in classification. The second dimension gives us insight into how substantial the change was. The trade-off between both dimensions depends on the distance metric used and can be domain-dependent. Therefore, we evaluated the quality of the counterfactual image under a distance constraint, using several distance metrics. More specifically, we used the distance-constrained success criterion, where for a given threshold t original image x and generate image $x ^ { \prime } { . }$

$$
\operatorname { s u c c e s s } ( x , x ^ { \prime } , t ) = \mathbb { I } [ \mathcal { N } ( x ) \neq \mathcal { N } ( x ^ { \prime } ) \land d ( x , x ^ { \prime } ) < t ] .
$$

Simply put, under this success criterion, a counterfactual is successful if it changes the original prediction of the classifier, and its distance from the original image is lower than t.

We evaluated distances using the L1 and L2 norms, Structural Similarity Index Measure (SSIM) [29], and the Learned Perceptual Image Patch Similarity (LPIPS) metric [32]. The latter better aligns with human perceptions of image similarity. In our implementation of the algorithm, we used the L2 norm as the distance function to find the closest positive image in the reference dataset $\mathcal { P } .$ This choice was deliberate, since our method aims to be fully transparent, and incorporating LPIPS, which is blackbox, would undermine this goal. However, we stress that our approach is general and can accommodate any distance metric, interpretable or not. We selected L2 over other interpretable metrics because it yielded slightly better performance. We still include all distance metrics as a part of the evaluation to align with prior related work [21, 34].

In addition, to assess whether the different approaches to counterfactual generation map the population to a similar region of the input space, we computed all pairwise interimage distances before and after transformation and compared the resulting distance distributions (Figure 3b).

## 4.3. Comparison

We evaluated our approach using two variants: one using sufficient explanations (SE), and one using complete explanations (CE). To evaluate our contribution, we compared our approach with two state-of-the-art methods for generating counterfactual explanations for medical images: 1) GANterfactual, a generative adversarial learning (GAN) approach [21], and 2) MoPaDi, a diffusion model (DM) approach [24, 34]. Both GANterfactual and MoPaDi were trained using the default parameters provided in their publicly available implementations. We also evaluated a naive approach that creates a counterfactual image by taking the explanation of the closest positive image in the training data and inserting it onto the original image. We show the results of this approach for both sufficient and complete explanations, which we denote naive sufficient explanations (NSE) and naive complete explanations (NCE), respectively. All methods were given access to the same training data. The generative approaches (GAN and DM) used the full training set during model training, whereas the causal approaches (SE, CE, NSE, and NCE) used the positively classified training samples to construct the precomputed reference database of causal explanations used for counterfactual generation.

## 4.4. Results

Figure 3a presents the distance-constrained success rate of the methods for both datasets. To determine a meaningful upper bound on acceptable distances, we selected the cutoff based on the variability in its respective dataset. Specifically, we computed the distribution of pairwise distances between images in the dataset and selected the 99<sup>th</sup> percentile as the maximal distance threshold. Distances beyond this point correspond to image pairs that are already extremely dissimilar within the dataset itself, making it unlikely that a counterfactual exceeding this bound would remain usefully similar to the original image from which it was generated.

On the brain MRI dataset, the CE approach outperformed the other approaches at lower distance thresholds, achieving an 85% success rate with distance thresholds of 0.25 in LPIPS and 0.2 in SSIM. Additionally, the diffusion model (DM) method differed from the causal approaches, producing successful counterfactuals but which were significantly farther from their original images. The GAN approach generated correct counterfactuals for most of the dataset, but they differed substantially from the original images. In the case of the brain MRI use case with L1 and L2, none of the counterfactuals were at a distance less than the maximal threshold, and therefore, its distance-constrained success rate remains 0 in the examined range.

For the skin lesion dataset, the CE approach again outperformed other approaches at lower distance thresholds, achieving a 67% success rate at distance thresholds of 0.48 in LPIPS and 0.21 in SSIM. At higher distance thresholds, NCE, GAN, and DM achieve higher success rates. However, this improvement is driven by transformations that substantially alter the original image (examples of such transformations are shown in Figure 4). That is, it is always possible to perturb an image sufficiently to change the classification. However, larger changes reduce the practical usefulness of the explanation. This accounts for the poor performance of these methods at lower distance thresholds.

Figure 3b shows the distribution of the pairwise interimage distances for the original images (before transformation) and after transformation of each method. To ensure a fair comparison, we restrict the dataset to the intersection of images for which all methods successfully generated a valid counterfactual, so that all statistics are computed over an identical evaluation set. All causal approaches preserved the distribution of the original dataset in both use cases, whereas the GAN approach generated counterfactuals with pairwise distances that were significantly lower than those of the original dataset. This indicates that the GAN model converged to a narrow transformation pattern, producing counterfactuals that are highly similar to each other but highly dissimilar from the original data distribution. This is evidence that the GAN model has learned a simple shortcut, transforming all images to very similar counterfactuals (visible as the sharp peaks in Figure 3b). This may be exacerbated by shortcut learning in the original classifier, which is common in medical models which are often trained in datasparse regimes. The DM approach showed a similar trend in the skin lesion dataset. However, for the brain MRI dataset, the DM generated counterfactuals with pairwise distances that were higher than those of the original dataset.

## 4.5. Limitations

An important caveat of our approach is its dependence on the diversity of the reference set and the corresponding explanations. The method will likely benefit from a reference dataset with a broad range of pathological appearances and imaging conditions. If certain patterns are underrepresented, the space of available perturbations may become limited. Nevertheless, this dependency is less restrictive than those imposed by generative methods. Generative models must learn a full image distribution and, therefore, typically require large, diverse datasets to produce viable results. In contrast, our approach reuses localized causal evidence directly from existing images and does not require training an additional model. As a result, the data requirements are substantially smaller.

![](images/cdfa0eb347341481e863b2edde3125b0ee41b729b20809da51ba98babc39a207.jpg)  
(a) The distance-constrained success rate of the examined methods for the brain MRI (left column) and ISIC skin lesion datasets (right column). The top row presents the results for the LPIPS metric for the two case studies, followed by the complement of SSIM (1-SSIM), L1, and L2.

## 5. Related Work

Counterfactual explanations have become an important approach for providing actionable explanations and satisfying emerging expectations for AI transparency [13, 27, 28]. In computer vision, most counterfactual methods generate an alternative image that would be assigned a different label by a classifier. A dominant line of work uses generative models to perform image-to-image translation. Early approaches employed CycleGANs to change class labels without requiring paired training examples [33], a property that is particularly attractive in medical imaging where paired pathological and healthy images are rarely available. Building on this idea, GANterfactual [21] used GAN-based image translation to produce medical counterfactual explanations, while [26] introduced a conditional GAN framework designed to preserve rare but clinically important image details.

![](images/3b0e24ac2e36c8d58a00cc39c47cbbdb97b9fc19f150e97468c5db3736414a4b.jpg)  
(b) Distribution of the pairwise inter-image distances for the original images (before transformation) and after transformation of each method for the brain MRI (left column) and ISIC skin lesion datasets (right column). The top row presents the results for the LPIPS metric for the two case studies, followed by the complement of SSIM (1-SSIM), L1, and L2.

More recent work has largely shifted towards diffusionbased generators. [34] and [2] exploit diffusion models to generate medically plausible counterfactuals, while [19] and [22] learn dedicated generative mechanisms for controllable counterfactual image synthesis in medical settings. Beyond image generation, [30] employ a deep structural causal model to manipulate demographic and clinical factors in chest radiographs, enabling counterfactual analyses of attributes such as race and sex. Despite their methodological differences, these approaches share a common characteristic: they require training an additional generative or structural model whose learned representation mediates the explanation process. This reliance on a second model introduces a potential tension for explainability. The resulting counterfactual may reflect not only the classifier under inspection but also the inductive biases of the generator itself. Consequently, it can be difficult to disentangle whether a generated image reflects the classifier’s actual decision boundary, the generative model’s image prior, or optimization artifacts [11]. Furthermore, most existing methods prioritize visual realism and clinical plausibility. While desirable for data generation or simulation, realism is not necessarily synonymous with explanation: a highly realistic image may obscure the specific evidence that the classifier

![](images/14e2c0c19de07899598ecc42e99792b7fdf1b81f65c5cfd9bcc0ae5eedaded46.jpg)  
Figure 4. Our approach compared with other techniques on the two datasets. Column 1 shows samples classified as healthy. Using our method, we generate the images shown in column 2. If we apply ‘naive patching’ we get the images in column 3. These are all classified as positive, but a large part of the original image has been replaced. In the skin lesion samples, the entire lesion is replaced. Finally, column 4 shows the output from the DM. Although all these images are classified as positive, some images show a completely different sample, and in others, the changes are not local.

## uses when making a prediction [5, 26].

from which counterfactuals are constructed.

The closest work to ours is [12], who generate counterfactual visual explanations by copying image regions from a “distractor” image into a query image. Their method demonstrates that small localized edits can alter classifier predictions, but the transferred content is selected through an optimization procedure and is not tied to features that are known to be causally responsible for the classifier’s decision. Consequently, it remains unclear why a particular perturbation succeeds. Our approach differs from both generative and retrieval-based counterfactual methods in a fundamental way. Rather than learning a latent representation of the data or searching for effective pixel substitutions, we construct counterfactuals directly from causal evidence extracted from the classifier itself. To our knowledge, this is the first image-counterfactual framework in which causal explanations are not merely used to interpret counterfactuals after generation, but serve as the primary representation

## 6. Conclusion

We explore the nature of counterfactual images from the perspective of the classifier. By calculating the information change required to move ‘healthy’ to ‘diseased’ images using only the classifier itself, we are able to provide ‘whatif’ images to clinicians tasked with understanding a model’s modes of failure. Our method induces a change in classification using far smaller changes compared to state-of-theart generative models, resulting in counterfactuals closer to the original image. Future work will assess whether clinicians do indeed find our counterfactual images useful for auditing models.

## References

[1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, Sabine Susstrunk, et al. Slic superpixels.¨ Technical report, Technical report EPFL, 2010. 5

[2] Matan Atad, David Schinz, Hendrik Moeller, Robert Graf, Benedikt Wiestler, Daniel Rueckert, Nassir Navab, Jan S. Kirschke, and Matthias Keicher. Counterfactual explanations for medical image classification and regression using diffusion autoencoder. Machine Learning for Biomedical Imaging, 2024. 7

[3] Anonymous Authors. Counterfactual explanations for medical imaging classification using causality — supplementary material. https://figshare.com/s/ f82b2a8afe4b366ee07b, 2026. 2, 5

[4] Sander Beckers. Causal explanations and XAI. In Proceedings of the First Conference on Causal Learning and Reasoning, pages 90–109. PMLR, 2022. 2, 3

[5] Dipkamal Bhusal, Michael Clifford, Sara Rampazzi, and Nidhi Rastogi. Face: Faithful automatic concept extraction. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 8

[6] N. Blake, D.A. Kelly, S.C. Pena, A. Chanchal, and H. Chock-˜ ler. Mrxai: Black-box explainability for image classifiers in a medical setting. Ceur Workshop Proceedings, 4059, 2025. 1

[7] Mateusz Buda, Ashirbani Saha, and Maciej A Mazurowski. Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. Computers in biology and medicine, 109:218– 225, 2019. 5

[8] Hana Chockler and Joseph Y. Halpern. Responsibility and blame: A structural-model approach. J. Artif. Intell. Res., 22:93–115, 2004. 3

[9] Hana Chockler, David Kelly, and Daniel Kroening. Multiple different black box explanations for image classifiers. In ECAI 2025 - 28th European Conference on Artificial Intelligence - Proceedings, pages 699 – 706, 2025. Publisher Copyright: © 2025 The Authors. 3

[10] Hana Chockler, David A Kelly, Daniel Kroening, and Youcheng Sun. Causal explanations for image classifiers. Journal ofArtificial Intelligence Research, 2026. 1, 2, 3

[11] Joseph Paul Cohen, Rupert Brooks, Sovann En, Evan Zucker, Anuj Pareek, Matthew P Lungren, and Akshay Chaudhari. Gifsplanation via latent shift: a simple autoencoder approach to counterfactual generation for chest x-rays. In Medical Imaging with Deep Learning, pages 74–104. PMLR, 2021. 1, 7

[12] Yash Goyal, Ziyan Wu, Jan Ernst, Dhruv Batra, Devi Parikh, and Stefan Lee. Counterfactual visual explanations. In Proceedings of the 36th International Conference on Machine Learning, pages 2376–2384. PMLR, 2019. 8

[13] Riccardo Guidotti, Anna Monreale, Salvatore Ruggieri, Franco Turini, Fosca Giannotti, and Dino Pedreschi. A survey of methods for explaining black box models. ACM Computing Surveys, 51(5):93, 2018. 7

[14] David Gutman, Noel CF Codella, Emre Celebi, Brian Helba, Michael Marchetti, Nabin Mishra, and Allan Halpern. Skin

lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (isbi) 2016, hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1605.01397, 2016. 5

[15] Joseph Y. Halpern. Actual Causality. The MIT Press, 2016. 2, 3

[16] David A Kelly and Hana Chockler. Sufficient, necessary and complete causal explanations in image classification. arXiv preprint arXiv:2507.23497, 2025. 2, 3

[17] David A Kelly, Hana Chockler, and Nathan Blake. Explain ing negative classifications of ai models in tumor diagnosis. In The 41st Conference on Uncertainty in Artificial Intelli gence, 2025. 3

[18] Benedicte Legastelois, Amy Rafferty, Paul Brennan, Hana Chockler, Ajitha Rajan, and Vaishak Belle. Challenges in explaining brain tumor detection. In Proceedings ofthe First International Symposium on Trustworthy Autonomous Systems, pages 1–8, 2023. 5

[19] Shiyu Liu, Fan Wang, Zehua Ren, Chunfeng Lian, and Jianhua Ma. Controllable counterfactual generation for interpretable medical image classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 143–152. Springer, 2024. 1, 7

[20] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. Advances in neural informa tion processing systems, 30, 2017. 1

[21] Silvan Mertes, Tobias Huber, Katharina Weitz, Alexander Heimerl, and Elisabeth Andre. Ganterfac-´ tual—counterfactual explanations for medical non-experts using generative adversarial learning. Frontiers in artificial intelligence, 5:825565, 2022. 1, 3, 6, 7

[22] Kwanseok Oh, Jee Seok Yoon, and Heung-Il Suk. Learnexplain-reinforce: counterfactual reasoning and its guidance to reinforce an alzheimer’s disease diagnosis model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4843–4857, 2022. 1, 7

[23] Judea Pearl. Causality: Models, Reasoning and Inference. Cambridge University Press, 2nd edition, 2009. 2

[24] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10619– 10629, 2022. 6

[25] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ” why should i trust you?” explaining the predictions of any classifier. In Proceedings ofthe 22ndACM SIGKDD interna tional conference on knowledge discovery and data mining, pages 1135–1144, 2016. 1

[26] Sumedha Singla, Motahhare Eslami, Brian Pollack, Stephen Wallace, and Kayhan Batmanghelich. Explaining the black box smoothly—a counterfactual approach. Medical Image Analysis, 84:102721, 2023. 1, 7, 8

[27] Sahaj Verma, John Dickerson, and Keegan Hines. Counterfactual explanations and algorithmic recourses for machine learning: A review. arXiv preprint arXiv:2010.10596, 2020. 7

[28] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the gdpr. Harv. JL & Tech., 31:841, 2017. 1, 7

[29] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 5

[30] Tian Xia, Melanie Roschewitz, Fabio De Sousa Ribeiro,´ Charles Jones, and Ben Glocker. Mitigating attribute amplification in counterfactual image generation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 546–556. Springer, 2024. 7

[31] Yiqi Yan, Jeremy Kawahara, and Ghassan Hamarneh. Melanoma recognition via visual attention. In International Conference on Information Processing in Medical Imaging, pages 793–804. Springer, 2019. 5

[32] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5

[33] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings ofthe IEEE international conference on computer vision, pages 2223– 2232, 2017. 7

[34] Laura Zigutyte, Tim Lenz, Tianyu Han, Katherine J. He-<sup>ˇ</sup> witt, Nic G. Reitsam, Sebastian Foersch, Zunamys I. Carrero, Michaela Unger, Alexander T. Pearson, Daniel Truhn, and Jakob Nikolas Kather. Counterfactual diffusion models for mechanistic explainability of artificial intelligence models in pathology. bioRxiv, 2024. 1, 6, 7