# Beyond Benchmarks: Using VLMs to Reveal Systematic Classification Failures Under Real World Conditions

Dieuwertje Alblas<sup>a\*</sup>, Alma M. Liezenga<sup>a\*</sup>, Jan Erik van Woerden<sup>a</sup>, Fedor Taggenbrock<sup>a</sup>, Dalia Aljawaheri<sup>a</sup>, and Klamer Schutte<sup>a</sup>

<sup>a</sup>Intelligent Imaging, Defense, Safety & Security, TNO, The Hague, The Netherlands

## ABSTRACT

Verification and validation (V&V) of classification models is crucial to enable a wide range of sensor processing applications. Currently, the V&V process relies on time-consuming manual inspection of erroneous samples to find meaningful patterns. This work explores the use of Vision Language Models (VLMs) to speed up this laborious process. VLMs are trained to embed images into a semantically meaningful vector representation, from which human-interpretable systematic errors can be distilled. Deploying such VLM-based methods in a defence context introduces two major challenges: (1) the defence domain is underrepresented in the training data of VLMs, and (2) surroundings and context are less diverse than for other domains. This study provides an initial assessment of the suitability of VLM-based methods for V&V of defence applications. We propose a VLM-based error slice detection (ESD) method that independently groups and labels systematic errors made by a classification model. We demonstrate that this method is able to identify operationally-relevant artificially added perturbations in a non-military dataset. In a military context, our method clusters and describes images based on their surroundings, but also exhibits overlap between cluster descriptions. We further investigate the diference in embedding variation between our military and non-military dataset, which remains a topic of interest. Although the results do not yet warrant fully automated V&V through VLM-based ESD, they show that VLMs could be used to accelerate V&V processes in the future.

Keywords: Verification & Validation, Vision Language Models, Image Classification, Error Slice Discovery, EDF STORE

## 1. INTRODUCTION

Modern military vehicles are equipped with an increasing number of sensors to enhance situational awareness, such as cameras for target detection and recognition. This growth leads to high-volume sensor data streams, necessitating automatic analysis of sensor measurements.<sup>1</sup> Artificial intelligence (AI), particularly discriminative AI, has shown promising results in, e.g. small object detection,<sup>2</sup> fine-grained military vehicle classification<sup>3</sup> and classification of vessels.<sup>4</sup> To facilitate safe and reliable deployment in operational scenarios, verification and validation (V&V) of AI models is crucial but poses a significant challenge.<sup>5,</sup> <sup>6</sup> In this work, we focus on V&V of discriminative AI methods operating on visual data. These vision models may exhibit degraded performance under conditions such as adverse weather, occlusions, camouflage, or smoke, which frequently occur in operational environments.<sup>7,</sup> <sup>8</sup> Being aware of these shortcomings prior to deployment and mitigating their impact is important for the safe and responsible deployment of AI models in operational contexts.

AI models are black-box by design and learn non-linear mappings between input and output from a large set of training examples. The black-box nature of these models results in unclear decision boundaries. The essentially unlimited space of possible input data specific to vision further complicates comprehensive testing of AI models. Moreover, model strongly depends on the training data; typically AI models do not generalize wel to out-of-distribution samples.<sup>9</sup> When using third party models, there is potentially limited knowledge about the data that was used to train the AI model, which makes identifying model vulnerabilities dificult. Together, these aspects make systematic identification of failure modes a complex process, limiting the confidence in a model’s behaviour under previously unseen but operationally relevant conditions.

Traditional V&V methods such as failure modes and efect analysis (FMEA)<sup>10</sup> are commonly applied to rule-based systems, where system behaviour is explicitly defined. However, for AI models with complex and implicit decision boundaries, some parts of FMEA should be reconsidered as a full sweep of potential inputs is not feasible. Consequently, AI models are primarily evaluated based on their observed behaviour on a finite set of test samples. For V&V purposes, it is crucial to understand why a model fails in specific test cases. For example, which environmental factors, visual attributes or conditions lead to incorrect predictions. This knowledge can then be used to mitigate vulnerabilities through, e.g. design choices or model fine-tuning.

One commonly used approach to gain insight into model vulnerabilities is explainable AI. For vision models, saliency-based methods such as Grad-CAM<sup>11</sup> are a well-established technique. These methods generate heatmaps that indicate which image regions are most strongly associated with a model’s prediction. While saliency maps can provide qualitative insights into individual model decisions, extracting actionable knowledge from them is labor-intensive. In practice, an operator should manually inspect large numbers of images and interpret the highlighted regions to categorize failure cases on image semantics. This manual step limits the scalability for systematic V&V of AI models.

Until recently, semantic reasoning over image data was restricted to human interpretation. The emergence of contrastive vision-language models (VLMs) fundamentally changes this restriction. Contrastive VLMs embed images and text into a shared latent space, which is structured by semantic concepts.<sup>12</sup> As a result, images containing similar semantic attributes are mapped close together in this latent space, and distance measurements can be interpreted as a notion of semantic similarity.<sup>13</sup> This opens up new avenues for automatic interpretation of image semantics. Generative VLMs, on the other hand, take image and/or text input and generate textual (or visual) responses based on them,<sup>14</sup> ofering even more opportunities for automated semantic interpretation.

Recent works have leveraged VLMs for automatic error slice discovery (ESD), i.e. the automatic discovery of systematic errors of AI models. We distinguish two types of ESD methods. First, slice-then-tag methods that leverage the semantic structure of the embedding space provided by a contrastive VLM to detect error clusters, after which cluster descriptions are automatically derived using a generative VLM.<sup>15–17</sup> Second, tagthen-slice methods,<sup>18,</sup> <sup>19</sup> where this order is reversed: a generative VLM is used to label images based on a set of predetermined attributes and tags. In this work, we focus on the first type of methods.

Applying VLMs for automatic ESD in defence-related scenarios introduces two key challenges. First, the defence domain is underrepresented in VLM training data, which may result in inaccurate or unreliable image embeddings. Second, these ESD methods are often developed using academic datasets captured under ideal conditions and containing a wide variation of environments. In contrast, (operational) military data is subject to significantly lower data quality and covers a much smaller range of environments. In this work, we assess how VLM-based error slice discovery transfers from a well-represented domain to the military domain. To this end, we leverage a dataset with images of diferent dog breeds and multiple datasets with images of military vehicles. Moreover, we assess if VLM-based ESD can detect artificially introduced systematic perturbations in the data.

## 2. MATERIALS & METHODS

In this work, we evaluate the value of VLMs in automatic error slice discovery for the purpose of V&V of AI models in a defense context. VLMs have the capability to map images to a latent space, structured by semantics. We introduce a VLM-based automatic ESD method, inspired by the Domino method introduced in Eyuboglu et al.<sup>15</sup>

## 2.1 VLM preliminaries

The contrastive VLM that is central in our ESD method operates in three distinct spaces: (1) image space I, (2) embedding space $\mathcal { E } \subset \mathbf { R } ^ { d }$ , and (3) semantic space S. The VLM learns mappings from both image space I and semantic inputs S, e.g. text, to a shared d-dimensional embedding space E. We consider a classification dataset $\mathcal { X } \subset \mathcal { T }$ , consisting of n images, each assigned to exactly one of k classes. We denote the subset of images from X that belong to class $c \in { 1 , 2 , 3 , . . . , k \mathrm { \ a s \ } \{ x _ { c } \} }$

![](images/df354e4df51e20995308c55865f69b7206bf11920ec5d17b6c233f83d3c6befa.jpg)  
Figure 1. Overview of the VLM-based ESD method, consisting of five steps. 1 the AI model subject to V&V is applied on a set of test images, resulting in correctly (green) and wrongly (red) predicted samples. 2 the test images are mapped to the joint text and image embedding space using the VLM. 3 to shift the embeddings’ focus from the main object to its surroundings, the respective mean class embedding is subtracted from the sample embeddings. 4 clusters are retrieved from these residual embeddings. 5 images from a cluster are jointly fed to a generative VLM, which is prompted to describe the similarities between the images, in this case furniture, lying down and indoors.

The spaces I, S, and E difer substantially in terms of geometric properties and interpretability. While elements from I and $s ,$ i.e. images and text, are human-interpretable, the calculation and interpretation of (Euclidean) distances between elements in these spaces is ambiguous. In contrast, although individual points in the joint embedding space E are not human-interpretable, the geometry of this space is structured such that small distances reflect semantic similarity.<sup>13</sup>

Images typically comprise multiple objects and environmental elements, which are jointly encoded by a VLM into a single latent representation. For ESD, all these components are equally important. However, VLM embeddings tend to be dominated by the largest or most salient object in the image.<sup>20</sup> Recent work has shown that VLM latent spaces exhibit local linearity, with embeddings approximating linear combinations of semantic concepts.<sup>13,</sup> <sup>21</sup> This property enables the manipulation of embeddings to disentangle objects from contextual elements.

## 2.2 VLM-based Error Slice Discovery

Our VLM-based ESD method consists of five steps, as shown in Figure 1. First, the AI model subject to V&V is inferenced on a set of test images, resulting in a number of correctly and wrongly predicted samples. Second, the same test images are embedded using the VLM $g ( x ; \theta ) : \mathcal { T } \to \mathcal { E }$ , resulting in embeddings $e _ { x } \in \mathcal { E }$ Third, we subtract the class-level average embedding $e _ { c }$ from the image embeddings $( e _ { x } )$ belonging to that corresponding class, with the idea of shifting the representation away from the main object and toward its surrounding environment. We calculate the mean class embeddings as $e _ { c } = \frac { 1 } { | x _ { c } | } \sum _ { x \in x _ { c } } e _ { x }$ . Fourth, we perform

dimensionality reduction on these residual embeddings and clustering in this lower dimensional space. Fifth, we yield a human-interpretable description of each cluster by leveraging a generative VLM. For each cluster, we create a collage containing multiple images that is fed to the VLM, which is prompted to describe the cluster using five words ranked from most important to least important<sup>∗</sup>. The full prompt is given in Appendix A.

These five steps result in automatically acquired, human-interpretable descriptions of semantically related test samples. Note that these clusters could contain both correctly and wrongly predicted samples, which gives insight into the model performance in a certain type of environment.

A Dog breed classification dataset

## ① Dog breeds

![](images/3a156201e9e4f4486683822c00d6216621ace596990bad5310c9505b3065720d.jpg)

![](images/dd33484669c661207943b17df48cccf24c1107592aa806d76d544249ba5c91d2.jpg)

![](images/a83baade6604d02b0c1b07217d319beb43d7b62f68d73ff0a9d6b1dc9ae69861.jpg)

![](images/cdf1490972681ca9aa50634d59a6cf7b49d1ff67982cf9f6a5c277a3f5287ca0.jpg)  
A Public military vehicle classification dataset

## ② Military vehicles

![](images/6f68262842f26ea527503f4f4a321cf8d30d0ff9ac5b272558b20f38cb1a4818.jpg)  
CSTORE military vehicle classification dataset

![](images/2acaf24df5a7c3a481307777f32558ddc97635f65e6363dd4944d3c74b4215b6.jpg)  
Kongsberg, Norway

![](images/2ec9c53b3156ddfcf09a04a5528ac8d6bedef48d5f6d90fd2fef44df2fe4de33.jpg)  
Fontevraud, France

![](images/2262f81faeac561bc479e025828d4582cee797fca57ad390f34a5392553bbad7.jpg)  
Fontevraud, France  
B Ukraine military vehicle classification dataset

![](images/cd5ddde4fb60b56038a491f811439194deaea39de0b7b621d30f285bb8997ed3.jpg)  
Figure 2. Overview of the datasets from two diferent domains covered in this work: 1 dog breeds and 2 military vehicles, consisting of one and three sources, respectively <sup>‡</sup>. Datasets 1A and 2A are obtained from public resources, while datasets 2B and 2C are obtained from bounding boxes in video frames and represent more challenging conditions.

## 2.3 Data

In this work, we use four classification datasets from two domains: (1) dog breeds, and (2) military vehicles. Figure 2 shows an overview of these datasets, consisting of one dog breed dataset and three datasets containing military vehicles. We include these two domains as we expect a discrepancy in their representation in the training data of the VLM. Images of dogs in diferent surroundings are widely available on the internet, while images of military vehicles are less represented. Moreover, the surroundings of the images of dog breeds varies more than for the military vehicles. These two aspects could influence the quality of the VLM embeddings of these images, resulting in performance discrepancies of the ESD method between the two domains. We first describe the four datasets in more detail.

1A: Dog breed classification dataset<sup>22</sup> consists of 8.040 web-based images of 93 diferent dog breeds. This dataset resembles the typical academic conditions, where image quality is high and the main objects are mostly centered. Nevertheless, this dataset contains a large variety in environments and small inter-class diferences, making it useful for understanding strengths and weaknesses of the ESD method.

2A: Public military vehicles classification dataset<sup>3,</sup> <sup>23</sup> is an in-house dataset compiled from public sources and consists of 18 vehicle classes representing diverse military platforms: tanks (T-62, T-72, T-90,

Leopard 2, M1A2), armored personnel carriers (BTR-80, BMP-1, Fuchs, Boxer, Patria), reconnaissance vehicles (BRDM-2, Fennek), self-propelled artillery (2S1, 2S3, M109, MSTA, Panzerhaubitze 2000), and support vehicles (military truck). Each class contains approximately 50 images. As most samples from this dataset were acquired from public sources, most images have a high resolution and contain centered objects. More information on the dataset can be found in van Woerden et al. (2025).<sup>3</sup>

2B: Ukraine military vehicles classification dataset is an in-house dataset acquired from frames from three publicly available videos from Ukraine<sup>§</sup>. The 289 samples in this dataset are cropped bounding boxes containing vehicles present in the frames, which were manually labelled using function classes, e.g., armored personal carriers, military trucks and self-propelled artillery. As the samples are cropped from video frames, this dataset contains low resolution and challenging operational conditions.

2C: STORE military object classification dataset<sup>24</sup> is a military dataset that was acquired from videos recorded during two campaigns, (1) at the military site in Fontevraud-l’Abbaye, France and (2) at a testing facility near Kongsberg, Norway; as part of the EDF project STORE. The objects present during these campaigns include military and civilian vehicles. The dataset was recorded using a mix of drones, stationary, and moving ground vehicles, hence includes videos from both an air-to-ground and ground-to-ground perspective. The dataset was recorded with the goal of representing an operational, military context and varying conditions: one portion was recorded in Nordic, snowy conditions and the other portion in a milder Mid-European climate with some rainfall. Objects could be occluded by trees or rainfall. The selected dataset contains 1,813 cropped bounding boxes obtained from 80 videos.

Besides two diferent domains with varying coverage in the VLM training data, the four datasets include diferent levels of challenging operational conditions. While dataset 2A consists of military vehicles, most images are high-quality and contain centered objects, making it more aligned with typical academic settings. In contrast, dataset 2B contains images from an operational environment and 2C contains images from a campaign meant to simulate operational conditions, including adverse weather conditions, camouflage and small objects. The images in 2C are also not publicly available, meaning that the VLMs could not have been trained on them, whereas it is possible that the images in 1A, 2A and 2B were part of the training data of the VLMs that we used. Additionally, datasets 2B and 2C were obtained by cropping frames in videos, resulting in lower quality and partially occluded objects. Moreover, as these datasets were acquired from video frames, they have a smaller variation in surroundings compared to dataset 2A. We hypothesize that more variation in the environment will benefit the class-level average embeddings $e _ { c }$ and, in turn, the quality of the derived clusters. Therefore, this work includes experiments on a single military vehicle dataset, as well as a combination of the three military vehicle datasets.

## 2.3.1 Introducing perturbations

We assess how well our VLM-based ESD method captures and clusters systematic challenges in the data. We add artificial perturbations to the image data to simulate operational challenges.<sup>3</sup> Motion blur, random snow, slice blackout, bars blackout and random rain are applied at various severities: s ∈ {20%, 40%, 60%, 80%}, as shown in Figure 3.

## 2.4 Evaluation

We evaluate the derived clusters both qualitatively and quantitatively. For quantitative evaluation, we use the silhouette score<sup>25</sup> of the derived clusters in the embedding space. This score measures how similar each sample is to its own cluster compared to the other clusters and ranges between -1 and 1, where higher is better. We qualitatively evaluate the cluster descriptions obtained from the generative VLM and the semantic content from the images in the cluster. Here we focused on overlap and distinctiveness of the cluster descriptions. Moreover, we manually assessed correctness for the top descriptors in dataset 1A.

## 2.5 Implementation details

This section provides the implementation details of the models, algorithms and datasets used in our experiments.

![](images/e819526b70482e2d717e9733b4f8c15ff479852997fabf744a349f5b124cc46c.jpg)  
Figure 3. Examples of the diferent perturbations and occlusions that are used to corrupt our data, for increasing severities s ∈ {20, 40, 60, 80} %.

## 2.5.1 Classification models

Since we are studying ESD methods with the purpose of V&V, we introduce several classification models as models under evaluation. Note that since our focus is on the identification of systematic errors, training a perfect classification model is outside the scope of this work. We used diferent models for dog breed and military vehicle classification. For dog breed classification, we used YOLO26-cls, version L.<sup>26</sup> While limited resources were spent on fine-tuning this model for dog-breed classification (50 epochs), it achieved an accuracy of 88% on the validation set and 90% on the held-out test set. For military vehicle classification, we leveraged CLIP PE-Core-L-14-336<sup>27,</sup> <sup>28</sup> as a zero-shot classifier based on initial experiments showing its strong baseline performance for the military domain. Evaluation was conducted only on the Ukraine military vehicle classification dataset (2B Figure 2) and resulted in varying performance depending on the video quality. Videos 1 and 3 resulted in an accuracy of 81% and 88%, respectively, while the classification accuracy in video 2 was only 45%.

## 2.5.2 VLMs

We made use of two diferent VLMs for steps 2 , 3 and 5 of our method (Figure 1). For obtaining the embeddings of the images in the joint latent space, we used the contrastive CLIP PE-Core-L-14-336<sup>27,</sup> <sup>28</sup> model. This model was also used to obtain the class-level average embeddings $e _ { c }$ for the available classes in the dataset. For obtaining the cluster descriptions, we used the generative GPT-5 mini model,<sup>29</sup> as this provided the best trade-of between high performance and computational eficiency. Given a collage of 10 images for each cluster, we prompted this model to give a description of the top-5 commonalities between the images, ranked from most important to least important. The full prompt is given in Appendix A.

## 2.5.3 Data splits

The dog breed classification dataset (1A, Figure 2) was provided with a training, validation and test split consisting of 6391, 762, and 887 images, respectively. As this dataset contained an abundance of samples, we fine-tuned the classification model on the training split, and used the validation split for obtaining the class-level average embeddings $( e _ { c } )$ . The test split was used to validate the full ESD pipeline, as described in Figure 1. In contrast, the Ukraine vehicle classification dataset (2B) was much smaller and hence all images (289) were used both for obtaining class-level average embeddings and for validation the full ESD pipeline. Since a zero-shot classifier was used, no training set was necessary. For the STORE dataset (2C), the average class embeddings were calculated based on 382 samples, held out from the test set with 1431 samples. Finally, 260 samples from the public military dataset (2A) were used for calculating the average class embeddings and 477 diferent samples were used for testing the ESD pipeline.

## 2.5.4 Clustering

Prior to clustering of the residual embeddings in the joint latent space E, we relied on non-linear dimensionality reduction to 10 dimensions using the UMAP algorithm.<sup>30</sup> We applied this algorithm with Euclidean distance, a minimum distance of 0.0 and 15 neighbours. Subsequently, we used the HDBSCAN algorithm to identify clusters in this lower dimensional embedding space.<sup>31</sup> For this algorithm we used a minimal cluster size of 5 and a minimum sample size of 25. These algorithms and parameter settings were selected based on them yielding the highest silhouette score on our dog breed classification dataset compared to other parameters when doing a full parameter sweep and methods, namely T-SNE, UMAP, Sparse autoencoder, PCA and Kernel PCA and K-means, HDBSCAN, DBSCAN, Agglomerative and Spectral clustering.

## 3. EXPERIMENTS & RESULTS

In this section, we provide experimental results of our ESD method on both dog breed classification and military vehicle classification. We start with results on the dog breed data, a domain which is well represented in VLM training datasets. Here, we assess the cluster quality and the ability of the ESD method to capture artificially introduced perturbations (Figure 3). Subsequently, we move towards the domain of military vehicle classification, which has a lower representation in the VLM training data compared to the dogs. We assess the efect of this in terms of the cluster quality, but also include an analysis at the embedding level. We also include a comparative analysis between the two domains at the embedding level.

## 3.1 Dog breed classification

For our dog breed classification dataset, we assess if our ESD method can derive semantically meaningful clusters from a domain that is well represented in the VLM training data and contains largely varying environments. Moreover, we assess if our method can identify artificially introduced systematic perturbations in the data.

![](images/5a2c7b49a79dc17eac8288c60898e039d0623e43bfcafa37dd6c6511674d8138.jpg)

![](images/19046fd3156c8cb082b4ce93ab8c626070d9fa9111a95443c4aa5e4b05cf6f7d.jpg)

![](images/bae6b6904ad86db59bfa14a8af5ce55a77e8a18d2c0221e1f30b5ed63a96e924.jpg)  
UMAP 1  
Figure 4. UMAP visualization of the residual embeddings of the dog breed dataset. Some representative examples, including cluster descriptors, are shown for four clusters, namely 2 (red), 5 (pink), 7 (yellow) and 8 (blue). Unassigned samples are indicated by cluster ‘-1’. Quantitative results for all nine clusters are shown in Table 1.

## 3.1.1 Cluster quality

Our method yielded nine clusters on the dog breed dataset, achieving a silhouete score of 0.67. Moreover, the majority of image samples were not assigned to any cluster (469 images, 53%). Nevertheless, the derived nine clusters, shown in Figure 4, display semantic coherence. For example, cluster 2 consists of puppies, cluster 5 contains dogs lying on furniture, and cluster 7 contains dogs competing in a dog show. Table 1 shows the VLM-derived cluster descriptions, which are largely in agreement with the depictions provided by the qualitative overview in Figure 4. To support this quantitatively, we manually assessed the presence of some of the attributes in the description and reported the accuracy. In the best case, 100% of the images agreed with the attribute while this was 79% for the worst performing cluster.

Moreover, we assessed the performance of the classification model on each of the clusters, which is included in Table 1. This reveals that the classification performance on cluster 6 is far below the reported average performance of 90%. Based on the description generated for this cluster, the model struggles with samples where the subject is centered in a cluttered background and subject to low resolution, blur and varied lighting conditions. In contrast, the model performs better on classifying dogs in the snow (cluster 1), or in the grass (cluster 3).

Table 1. The clusters formed by the slice-then-tag method on the dog breed dataset, paired with the correctness of the first/most important word(s) (in bold), assessed through manual inspection, and the model accuracy for that cluster.
<table><tr><td>Cluster no.</td><td>Cluster description</td><td>Images assigned</td><td>Cluster correctness</td><td>Model accuracy</td></tr><tr><td>-1</td><td>side-facing, outdoor, low-angle, occluded, cluttered-background</td><td>469</td><td></td><td>91%</td></tr><tr><td>0</td><td>multiple, occlusion, outdoor, side-view, lighting</td><td>37</td><td>100%</td><td>86%</td></tr><tr><td>1</td><td>snow, overcast, centered, occluded, low-contrast</td><td>35</td><td>94%</td><td>94%</td></tr><tr><td>2</td><td>puppies, centered, varied-backgrounds, similar-lighting, occlusion</td><td>77</td><td></td><td>87%</td></tr><tr><td>3</td><td>grass, outdoor, centered, lighting, pose</td><td>53</td><td>81%, 96%</td><td>96%</td></tr><tr><td>4</td><td>indoor, centered, close-up, flash, top-down</td><td>65</td><td>86%</td><td>91%</td></tr><tr><td>5</td><td>indoor, lying, closeup, furniture, dim</td><td>52</td><td>90%, 87%</td><td>87%</td></tr><tr><td>6</td><td>centered subject, similar poses, cluttered backgrounds, low resolution/blur, varied lighting</td><td>14</td><td>93%</td><td>71%</td></tr><tr><td>7</td><td>side-profile, standing, handler-present, leash-visible, cluttered-background</td><td>42</td><td>86%</td><td>93%</td></tr><tr><td>8</td><td>profile, fur, standing, outdoor, occlusion</td><td>43</td><td>79%</td><td>93%</td></tr></table>

## 3.1.2 Systematic perturbations

In this experiment, we simulate systematic challenges in the data by randomly applying five occlusions and perturbations shown in Figure 3 to the images prior to acquiring VLM embeddings. For each sample in our test set, we applied one of the five transforms at a severity s ∈ {20%, 40%, 60%, 80%} randomly sampled from a uniform distribution. Based on these perturbations, the performance of our classifier dropped from 90% on the clean test set to 38% on the perturbed set. Subsequently, we let our ESD method find clusters of related samples. This results in three clusters, achieving a silhouette score of 0.87. This indicates an improved coherence and separation of the clusters compared to the previous experiment. Moreover, all samples were assigned to one of the three clusters.

Figure 5 shows how the occlusions at diferent severities are distributed across the clusters. We observe that images transformed using random snow and random rain were placed together in cluster 1. Similarly, images that were perturbed using motion blur and slide blackout were placed in cluster 3, while images transformed using bars blackout formed cluster 2 on their own. Only a small number of images that underwent a bars blackout transform ended up in cluster 3. Note that the severity of the transform does not afect the assigned cluster, indicating that also subtle alterations of the data can be picked up by the ESD method.

The VLM-derived cluster descriptions are also included in Figure 5. These seem to overlap with the factual description of the transformation applied to the images, e.g. ‘noise-overlay’ and ‘partial occlusion’ describing the random rain and random snow perturbations and ‘vertical occlusion’ describing bars blackout. Cluster 3 contains both the motion blur and slide blackout transforms and is described as ‘cropped’, potentially referring to the partial coverage of the image by the blackout area, and ‘blur’, aligning with the motion blur transform. Interestingly, the final descriptor ‘indoor’ refers to an environmental factor rather than one of the perturbations, though, upon manual inspection, that environmental factor does not seem to be consistently present in the images assigned to that cluster.

We conclude that our ESD method is able to detect and categorize consistent challenges in the image data. Notably, many of the identified phenomena correspond to operational conditions rather than physical objects, suggesting that the VLM embedding space captures higher-level scene characteristics beyond object-level semantics.

![](images/e85f0eed38ab5c6ac19155e3e48819f37a648fb32b7b3c899899b251e14d07a7.jpg)  
Figure 5. The number of images per cluster formed by the slice-then-tag method on the perturbed dog breed classification dataset, plotted against the added augmentations and their severity.

## 3.2 Military vehicle classification

Based on these initial experiments, we next investigate how well our ESD method works in the military domain.   
We investigate if this method can also form semantically coherent clusters of military vehicle imagery.

## 3.2.1 Cluster quality

Applying the ESD method to the Ukraine vehicle classification dataset (2B) resulted in a relatively high silhouette score of 0.83 with 46 unassigned samples, indicating that the residual embeddings formed tight and well-separated clusters. Figure 6 shows that the derived clusters are more disjoint than those of the dog breed dataset in Figure 4. However, the cluster descriptions in Table 2 are less distinct and show considerable overlap, making them less informative than those obtained for the dog breed dataset. Closer inspection revealed that the derived clusters largely overlap with the three distinct videos that make up dataset 2B: cluster 0 corresponded nearly entirely with video 2, while videos 1 and 3 were distributed across multiple clusters (bottom right, Figure 6). This also explains recurring cluster descriptions, such as snow-related terms for video 1. Moreover, the description for cluster 0, containing low-quality footage from video 2, does not mention this obvious aspect in the description. In addition, it seems that the vehicle class is not fully separated from the derived clusters. For example, Figure 6 shows images from video 1 in clusters 3 and 4, each containing diferent vehicles as the only obvious diferentiating characteristic. In conclusion, the derived clusters and descriptions have limited additional value in this setting, as the same information could have been derived from existing metadata.

The previous experiment demonstrated that the clustering was dominated by video-specific characteristics, resulting in limited new insights. To evaluate the utility of our ESD method in a broader military context, we combined datasets 2A, 2B and 2C prior to embedding and clustering. This experiment included 2,197 images of military vehicles. The method yielded 22 diferent clusters of varying size (Figure 7). Analysis of the dataset distribution across clusters revealed that most clusters contained samples from only a single dataset, indicating that clustering was largely driven by dataset-specific characteristics. Cluster 14 was the only notable exception and was substantially larger than the other clusters, containing all samples from dataset 2A as well as samples from 2B and 2C. Figure 7 also presents the top 3 of the cluster descriptions. Similar to the labels provided in Table 2, these descriptions contain recurring elements (e.g. ‘low resolution’, ‘small’, ‘low contrast’, ‘occluded’), limiting their value for diferentiating between clusters. This lack of distinctiveness is further reflected by the cluster visualizations shown in Figure 8, revealing that the diferences between clusters are often subtle.

![](images/c1c7e6a3d9f730330bc274bf6ce6a12211531edb916627dc579a877325939b84.jpg)

![](images/ad6a5be2f90f0a4e7f20df0030213801d22dee76723416290d6872f7eef4c690.jpg)

![](images/272f16f4e5241b71e71f2676f3dec77dbc56340aa6963e711ea44547e23b70db.jpg)

![](images/64ee2452fe608508ccc7613cc3d50dbcf581263cae9c1f5ffb88cbf476dc9dc2.jpg)

![](images/6567f246504d92a2faa57e7e8eafcd7dc10c396dcb3905892dd444be37ac49cf.jpg)

![](images/2202cac81012ef832a5699928fc87578f1d0f6a7b32637f4e007f402dbf7b975.jpg)

![](images/fc59b3fce806a11e9149dc2432bd1a022ede4302898c87da186c347b94487cd4.jpg)  
Figure 6. A visualization of the clusters detected on the Ukraine military vehicle classification dataset (2B) with the first word of the corresponding cluster description. The bottom right plot shows the same UMAP plot, but colored per video i.o. per cluster.

Table 2. The clusters formed by the slice-then-tag method on dataset 2B, paired with the number of images assigned to that cluster and the model accuracy for that cluster.
<table><tr><td>Cluster no.</td><td>Cluster description</td><td>Images assigned</td><td>Model accuracy</td></tr><tr><td>-1</td><td>snowy-road, overhead-angle, low-contrast. large-dark-silhouettes, partial-occlusion</td><td>46</td><td>89%</td></tr><tr><td>0</td><td>silhouette, color, angle, background, lighting</td><td>57</td><td>44%</td></tr><tr><td>1</td><td>angle, lighting, color, background, occlusion</td><td>34</td><td>82%</td></tr><tr><td>2</td><td>road, forest, frontal, overcast, occluded</td><td>50</td><td>84%</td></tr><tr><td>3</td><td>viewpoint, background, lighting, occlusion, snow</td><td>26</td><td>100%</td></tr><tr><td>4</td><td>snow-covered background, armored vehicles,side view, low-contrast lighting, partial occlusions</td><td>61</td><td>79%</td></tr><tr><td>5</td><td>viewpoint, snow, low-contrast, motion-blur, occlusion</td><td>15</td><td>100%</td></tr></table>

![](images/b9d3701591ea30e31360ba574dafb0e0cc2208473557bf8cb9358b5ece210220.jpg)

Figure 7. Overview of the clusters derived from a combination of dataset 2A, 2B and 2C. Each row gives the distribution of the cluster across the three datasets, as well as the top-3 automatically generated descriptions.  
![](images/18b2dadeaf1c775f8e3acdade0bf4b44162f044fa7fb83fd5fbd75bf97c24861.jpg)  
Figure 8. Visualization of the clusters derived from the combination of datasets 2A, 2B and 2C.

## 3.3 Dataset analysis

We observed a performance discrepancy for our ESD method between the dog breed classification and the military vehicle classification datasets. We hypothesize this is related to a lack of variation in the environment of the main object and evaluate if this can be derived by comparing the variation in the (residual) image embeddings for both datasets.

We measure the variation of both the raw image embeddings and the residual embeddings using the distribution of pairwise cosine distances and the cumulative explained variance from PCA analysis. Figure 9 shows the results of this comparison for dataset 1A, 2B and a combination of 2A, 2B and 2C. The distribution of pairwise cosine distances of the image embeddings of dataset 1A roughly follows a normal distribution. In contrast, dataset 2B clearly shows three local maxima, representing the three videos this dataset consists of. The distribution of cosine distances for the combination of the three military vehicle datasets is more similar to the dog breed dataset, but has one additional local maximum. This local maximum disappears when the pairwise distances between residual embeddings are considered. Nevertheless, the local maxima in the distribution o dataset 2B persists. Moreover, the cosine distances between samples increase for all datasets after the mean class embedding is subtracted. This implies a larger variance in all residual embeddings compared to the original embeddings. Still, a clear diference in variance between the dog breed dataset and the combination of the three military vehicle datasets is not observed from these plots.

The PCA analysis in the bottom row of Figure 9 tells a similar story as the cosine distances in the top row. Here, we see that more than 90% of the variation in dataset 2B is explained in the 30 first principle components. Meanwhile, this amount of variance is explained in the first 99 and 140 components for all military vehicle datasets and the dog breeds dataset, respectively. This implies a much more limited variation in the embeddings of dataset 2B and a higher variation for the datasets 2A and 2C. The residual embeddings require more principal components to explain 90% of the variation, which implies that the residual embeddings show more variation than the original embeddings.

The increased cosine similarities and variation after subtracting the class mean embeddings suggest that a dominant component is removed and individual variations are amplified, potentially at the expense of datasetlevel semantics. As the cosine similarity is based on directions, subtracting the mean class-wise direction may distort the directions in the resulting residual embeddings. We investigate this efect by recomputing the pairwise distances after adding the dataset-level average embedding (˜e) to the residual embeddings. Figure 10 shows that this consistently reduces the pairwise distance compared to the original embeddings, as shown in the top left of Figure 9. This could imply that adding the dataset-level average embedding removes class-level information and preserves dataset-level information, while this information may be lost in embeddings where only the class-level average is subtracted.

## 3.4 Residual embedding

Subtracting the mean class embedding is an important step in the proposed ESD method. In this section, we assess the efect of this operation separately. We qualitatively compare the image embeddings before and after subtracting the mean class embedding for diferent dataset compositions. In addition, we compute the silhouette score using the class descriptions as cluster assignments to quantitatively assess the class-related structure in the embeddings.

Figure 11 shows that the dog breeds form a dominant component in the image embeddings of dataset 1A, as they are strongly organised according to their class descriptions. This observation is reflected by the relatively high silhouette score of 0.64. After subtracting the mean class embeddings, the classes become substantially more mixed, resulting in a silhouette score of -0.13. The UMAP embeddings were based on the image embeddings $( e _ { x } )$ and separately determined for each dataset cohort in Figure 11. We used the same UMAP projection for the image and residual embeddings, enabling a direct visual comparison of the resulting structures. These results suggest that subtracting the mean class embeddings removes a dominant class-related feature from the embeddings, allowing other sources of variation in the image data to become more prominent.

In contrast, for dataset 2B, embeddings of images that belong to the same class are not consistently grouped together. This indicates that the object class constitutes a less dominant component of the embedding representation. Consequently, subtracting the mean class embeddings has a smaller efect on the overall embedding structure, which is reflected by the nearly unchanged silhouette score before and after subtraction. The combination of datasets 2A, 2B and 2C shows an intermediate efect. Here, the image embeddings are organized by class, resulting in a high silhouette score of 0.75. Subtracting the mean class embedding reveals a diferent structure, where embeddings of classes are more mixed. Likewise, the silhouette score decreases to 0.22, indicating that this efect is less pronounced than is the case for dataset 1A.

Cosine distance  
![](images/c7a64317d5455efae430264ac0010b41db6990435e30c36def67b662bc635526.jpg)

![](images/8eb6e57c63fa4230874d6cd456f69c42f82dc5500f4fe7fa46f67022e096bd9b.jpg)

![](images/dc7d5fd539b3604e236166d4ac50f0362fccfa67848b3892df6dc1f68b384a67.jpg)

![](images/dbc338de5df6d2c32d373e7d606f1dbea3d0c3e5d18161f7d3d63161de4fb12e.jpg)  
Figure 9. Comparison between the dog breed dataset and the military vehicle datasets. Top row: distribution of pairwise cosine similarities for the image embeddings (left) and the residual embeddings (right). Bottom row: cumulative explained variance from the PCA analysis for the image embeddings (left) and the residual embeddings (right).

Alternative residual embeddings $( e _ { x } - e _ { c } + \tilde { e } )$  
![](images/3bb2f4bc16b31de3c695f5a5db32804ff3e7e940a09a144f994e2e8a93f892da.jpg)  
Figure 10. Distribution of cosine similarities for the three data cohorts for the residual embeddings corrected with the dataset level average embeddings.

![](images/550e46e0211239d3a9cce6e2cca3a7e6f0ad8d3fa5d5d34f3158066e718e2379.jpg)

Residual embeddings $( e _ { x } - e _ { c } )$  
![](images/16f149e7fbb6b8a8e38bdf1671995bd9fe9be2350d5dcf9bdfc7a394b08884fd.jpg)

![](images/05d1549450fe02cf2570cfc1f7703045dd785a6cf7a99e2cd41128acdfb8268e.jpg)

![](images/ef12559088ca6bd388007aba900c9bfd3c5248ca1cf0be290b197bec952c1814.jpg)

![](images/291c0d1e6b009462e471aba45c58372d12d984849fca3405b4211ca0ff605208.jpg)  
UMAP1

![](images/e3f604a4d7f2451ee81ae27e56993d9dc46724d371e14badda6f7180a92cc951.jpg)  
UMAP1  
Figure 11. UMAP visualizations of the original image embeddings (left column) and the residual embeddings (right column) for the dog breed dataset (top row) Ukraine military vehicle classification datset (middle row) and the combination of all three military vehicle classification datasets (bottom row). Diferent classes are indicated by marker colors, embeddings from images $\left( e _ { x } \right)$ are indicated as ○, while mean class embeddings (e ) are indicated as é. The top right of each plo shows the silhouette scores calculated from the class descriptions.

## 4. DISCUSSION

In this study, we have presented several experiments using a VLM-based ESD method to automatically discover systematic errors in an AI model. We have focused on the ability of this method to detect systematic errors in dog breed and military vehicle classification datasets. Based on the results, we have further investigated the diferences in variation of embeddings and residual embeddings for aforementioned datasets.

Our method showed promising results on the dog breed classification dataset. In particular, it successfully identified the systematic vulnerabilities that were deliberately introduced through image perturbations. Furthermore, subtracting the mean class embedding resulted in meaningful clusters corresponding to diferent environments. Figure 11 confirms that the class forms a prominent part of the embeddings for dogs, and that subtracting the mean class embedding reveals novel structures in the embedding space. These findings suggest that the approach can detect and categorize consistent challenges and higher-level scene characteristics in the image data. However, a notable limitation is that the majority of samples were not assigned to any cluster, reducing the overall coverage of the analysis. This may be a consequence of the clustering algorithm used, and alternative clustering approaches could potentially improve the fraction of data captured in meaningful slices.

In contrast to the dog breed dataset, our method yielded fewer novel insights into systematic vulnerabilities in the military vehicle datasets. Discovered cluster descriptions overlapped and clusters corresponded to individua videos, though this was expected given the goal of our method. Videos naturally exhibit the same environmental conditions, hence deriving clusters that match the diferent videos is in line with the results on the dog breed dataset. We expect that this method is most valuable for datasets that are derived from separate images, which contain more varying environmental conditions. However, we also suspect that subtracting video average embeddings rather than class average embeddings may be a promising route towards better results for video data.

A prerequisite for the residual embeddings leveraged by our ESD method is concept purity of the VLM in the domain it was used on. This entails that image embeddings are composed of a linear combination of the main object and its surroundings, i.e. $e _ { x } \approx e _ { c } + e _ { \mathrm { e n v i r o n m e n t } } .$ . Moreover, in order to obtain accurate mean class embeddings, the object should be in diferent environments, to average out the environmental component in the image embeddings. We observed two important aspects that difer between the dogs and military vehicle images that may hinder the efectiveness of residual embeddings for the military dataset. First, military vehicles are underrepresented in the training data compared to diferent dog breeds, as these are more widely available on the internet. This afects the quality of the learned class representation. Second, the surroundings of the military vehicles are monotonous compared to the dog breeds dataset. The embeddings of military vehicles may contain parts of the environment as a result of these spurious correlations.<sup>32</sup> Figure 11 supports this claim; for the diferent dog breeds, the image embeddings are mostly focused on the specific class, possibly because during training these dog breeds were seen in diferent environments. In contrast, the embeddings of military vehicles are not as focused on the specific class. Therefore, subtracting the mean class embedding for military vehicles results in a less pronounced structural change compared to the dog breeds. This efect may be mitigated by adding the dataset-level average embedding to the residual embeddings, as supported by the results in Figure 10. A reduction in the pairwise distances suggests the class average embedding contributes substantially to the tota variation. Correcting the residual embeddings with the dataset average reduces the variation in the dataset. Future work should investigate if this results in more meaningful clusters, particularly for data-scarce domains.

## 4.1 Limitations

Several limitations of this study should be taken into consideration. First of all, the availability of public military vehicle classification datasets is overall limited. This study included 3 such sets, of which one was recorded in an actual operational environment (2B) and one in an environment meant to simulate operational conditions (2C). Both had shortcomings: 2B only included images from 3 videos and 2C was also recorded across only two locations. This limited environmental variability for both datasets could cause the clustering to stir towards videos/environments and thus limit the ability of our method to form clusters beyond those characteristics, as discussed in the previous paragraph.

Another shortcoming of this study concerns the method of evaluation, especially quantitative evaluation. The V&V process inherently relies strongly on qualitative assessment and though this study attempted to reduce this reliance, it in itself required qualitative evaluation. The silhouette score was used as an indicator of the cohesion o the clusters formed but it is not a representative metric of the usefulness of those clusters. Therefore, qualitative assessment had to be included, though it is subject to subjectivity and bias of the authors.

Lastly, subsection 3.3 and subsection 3.4 hypothesize on the applicability of VLMs to the military domain in general and the usefulness of the residual embedding. Though these sections ofer first insights into these domains, they do not ofer a full ablation study of the diferent aspects impacting the performance of our method, namely the VLM selected, the knowledgeability of the VLM in the given domain, the calculation of the average class embedding and the residual embedding.

## 4.2 Future work

We identify several courses for future research. First of all, future work should address concept purity of VLMs in the military domain. The lack of concept purity and the existence of spurious correlations seemed to have limited the efectiveness of the use of residual embeddings in this domain. We identified two sources for this efect: underrepresentation in the training data, and limited variation of surroundings. Our experimental results are not conclusive for the individual contributions of both aspects and further investigation is required to understand the impact of and connection between these aspects. Moreover, the use of residual embeddings could be extended to an iterative method, where cluster average embeddings are iteratively subtracted to progressively reveal more subtle factors of variation and potential error modes.

Additionally, the current work attempted to derive cluster labels based on feeding a sample of images with a prompt to a generative VLM instead of using the embeddings that actually informed the formation of those clusters. This approach required using a diferent VLM for the cluster descriptions than for deriving the clusters, potentially afecting the distillation of the underlying semantic concepts found in the embedding space. Other methods for deriving the cluster descriptions should be considered in future work. For example, dictionary learning could be explored.<sup>33</sup> This would allow us to make the transfer from embedding space to semantic space instead, potentially leading to results that allow for more diferentiating cluster labels.

This study set out to accelerate the V&V process of black-box third party AI systems. This acceleration can be promoted not only through full automation but also through supporting, prioritizing and steering a human analyst with VLM-based methods. Besides the technical improvements of the method and underlying principles, future work should therefore also focus on developing AI-assisted workflows that combine the scalability of VLMs with human expertise and judgment.

## 5. CONCLUSION

We conclude that current VLM-based ESD methods are not yet suitable to fully automate the V&V process of AI models. However, they do show potential to accelerate the V&V process by helping human analysts identify and characterize potential failure modes, especially in third party models. Additionally, the semantically structured embedding space provided by VLMs ofers opportunities to characterize training and test data, identify underrepresented scenarios and reveal coverage gaps. However, the efectiveness of these approaches depends on both the target domain and the level of semantic interpretation required. Rather than fully automating V&V, our findings suggest that the most promising role of VLMs is to support human analysts in discovering, prioritizing, and investigating systematic model errors.

## ACKNOWLEDGMENTS

The authors used ChatGPT (OpenAI) for language and grammar refinement of the manuscript text and the generation of code for conducting experiments; all scientific content and interpretations are the sole responsibility of the authors.

This work received funding from the European Defence Fund through the project STORE (Shared daTabase for Optronics image Recognition and Evaluation), grant agreement №101121405. We would like to particularly express our gratitude to all consortium partners involved in data acquisition and processing.

[1] European Commision, “AI in Defence,” (2025).

[2] van Leeuwen, M. C., Fokkinga, E. P., Huizinga, W., Baan, J., and Heslinga, F. G., “Toward versatile small object detection with temporal-yolov8,” Sensors 24(22), 7387 (2024).

[3] van Woerden, J. E., Burghouts, G., Nijskens, L., Liezenga, A. M., van Rooij, S., Ruis, F., and Kuijf, H. J., “Occlusion robustness of clip for military vehicle classification,” in [Artificial Intelligence for Security and Defence Applications III], 13679, 412–422, SPIE (2025).

[4] den Hollander, R. J., van Rooij, S. B., van den Broek, S. P., and Dijk, J., “Vessel classification for naval operations,” in [Artificial Intelligence and Machine Learning in Defense Applications III], 11870, 115–132, SPIE (2021).

[5] Paardekooper, J.-P. and Borth, M., “Toward a methodology for the verification and validation of ai-based systems,” SAE International Journal of Connected and Automated Vehicles 8(12-08-01-0006), 71–82 (2024).

[6] Fokkinga, E. P., te Hofst´e, M. E., den Hollander, R. J., van der Meer, R., Benders, F. P., ter Haar, F. B., Marquis, V. E., van Berkel, M., Voogd, J. M., Eker, T. A., et al., “The validation of simulation for testing deep-learning-based object recognition,” in [Artificial Intelligence for Security and Defence Applications II], 13206, 253–275, SPIE (2024).

[7] Mirza, M. J., Buerkle, C., Jarquin, J., Opitz, M., Oboril, F., Scholl, K.-U., and Bischof, H., “Robustness of object detectors in degrading weather conditions,” in [2021 IEEE International Intelligent Transportation Systems Conference (ITSC)], 2719–2724, IEEE (2021).

[8] Nijskens, L., den Hollander, R. J., Melo, J. G., Benders, F. P., and Schutte, K., “Predicting dnn classification performance on degraded imagery,” in [Artificial Intelligence for Security and Defence Applications], 12742, 226–242, SPIE (2023).

[9] Liu, J., Shen, Z., He, Y., Zhang, X., Xu, R., Yu, H., and Cui, P., “Towards out-of-distribution generalization: A survey,” arXiv preprint arXiv:2108.13624 (2021).

[10] Bluvband, Z. and Grabov, P., “Failure analysis of fmea,” in [2009 Annual Reliability and Maintainability Symposium], 344–347, IEEE (2009).

[11] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D., “Grad-cam: visual explanations from deep networks via gradient-based localization,” International journal of computer vision 128(2), 336–359 (2020).

[12] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., “Learning transferable visual models from natural language supervision,” in [International conference on machine learning], 8748–8763, PmLR (2021).

[13] Papadimitriou, I., Su, H., Fel, T., Kakade, S., and Gil, S., “Interpreting the linear structure of visionlanguage model embedding spaces,” arXiv preprint arXiv:2504.11695 (2025).

[14] Raja, R. and Vats, A., “Advancing vision-language models with generative ai,” in [International Conference of Global Innovations and Solutions], 1–18, Springer (2025).

[15] Eyuboglu, S., Varma, M., Saab, K. K., Delbrouck, J.-B., Lee-Messer, C., Dunnmon, J., Zou, J., and Re, C., “Domino: Discovering systematic errors with cross-modal embeddings,” in [International Conference on Learning Representations], (2022).

[16] Jain, S., Lawrence, H., Moitra, A., and Madry, A., “Distilling model failures as directions in latent space,” arXiv preprint arXiv:2206.14754 (2022).

[17] Yenamandra, S., Ramesh, P., Prabhu, V., and Hofman, J., “Facts: First amplify correlations and then slice to discover bias,” in [Proceedings of the IEEE/CVF International Conference on Computer Vision], 4794–4804 (2023).

[18] Chen, M., Zhao, C., and Xu, Q., “Hibug2: Eficient and interpretable error slice discovery for comprehensive model debugging,” arXiv preprint arXiv:2501.16751 (2025).

[19] Chen, M., Li, Y., and Xu, Q., “Hibug: On human-interpretable model debug,” Advances in Neural Information Processing Systems 36, 4753–4766 (2023).

[20] Ruthardt, J., Gaur, M., Ramanan, D., Tapaswi, M., and Asano, Y. M., “Steerable visual representations,” arXiv preprint arXiv:2604.02327 (2026).

[21] Tewel, Y., Shalev, Y., Schwartz, I., and Wolf, L., “Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic,” in [Proceedings of the IEEE/CVF conference on computer vision and pattern recognition], 17918–17928 (2022).

[22] Kabilan, “Dog breed classification.” https://www.kaggle.com/datasets/kabilan03/ dogbreedclassification (2022). Kaggle dataset. Accessed: 2026-05-28.

[23] van Woerden, J. E., Burghouts, G. J., van Rooij, S. B., Ruis, F., Dijk, J., and Kuijf, H. J., “Visual prompt tuning and ensemble undersampling for one-shot vehicle classification,” in [Artificial Intelligence for Security and Defence Applications II], 13206, 152–160, SPIE (2024).

[24] Langlois, S., Achour, W., Bensberg, S., Camarlinghi, N., Ehrhart, M., Klier, N., Lemeur, A., Legoupil, S., A. Liezenga, R. P., Pruuden, T., and Saasen, K.-D., “Challenges in creating an experimental database for artificial intelligence: the edf store project,” in [12th international symposium on optronics in defence & security], (2026).

[25] Rousseeuw, P. J., “Silhouettes: a graphical aid to the interpretation and validation of cluster analysis,” Journal of computational and applied mathematics 20, 53–65 (1987).

[26] Sapkota, R., Cheppally, R. H., Sharda, A., and Karkee, M., “Yolo26: key architectural enhancements and performance benchmarking for real-time object detection,” arXiv preprint arXiv:2509.25164 (2025).

[27] Cho, J. H., Madotto, A., Mavroudi, E., Afouras, T., Nagarajan, T., Maaz, M., Song, Y., Ma, T., Hu, S., Jain, S., et al., “Perceptionlm: Open-access data and models for detailed visual understanding,” Advances in Neural Information Processing Systems 38, 23475–23537 (2026).

[28] Bolya, D., Huang, P.-Y., Sun, P., Cho, J. H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Bangalath, H., et al., “Perception encoder: The best visual embeddings are not at the output of the network,” Advances in Neural Information Processing Systems 38, 60884–60937 (2026).

[29] Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al., “Openai gpt-5 system card,” arXiv preprint arXiv:2601.03267 (2025).

[30] McInnes, L., Healy, J., and Melville, J., “Umap: Uniform manifold approximation and projection for dimension reduction,” arXiv preprint arXiv:1802.03426 (2018).

[31] Campello, R. J., Moulavi, D., Zimek, A., and Sander, J., “Hierarchical density estimates for data clustering, visualization, and outlier detection,” ACM Transactions on Knowledge Discovery from Data (TKDD) 10(1), 1–51 (2015).

[32] Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P., “Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization,” arXiv preprint arXiv:1911.08731 (2019).

[33] Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., et al., “Towards monosemanticity: Decomposing language models with dictionary learning,” Transformer Circuits Thread 2(5), 6 (2023).

## APPENDIX A. PROMPT FOR CLUSTER DESCRIPTION GENERATION

The full prompt fed into GPT-5 mini, paired with 10 sampled images: “Please describe the similarities between these images, focussing on the factors that might make a classification model misclassify them, for example the environment (e.g. forest, snow, urban, road), position of the main object, lighting conditions, backgrounds, angle, or any image perturbations, augmentations or object occlusions. Please do not focus on the [dog breed/class]. Please respond with maximum 5 words, separated by commas, starting with the most important similarity.”