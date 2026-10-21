# Copyright Leakage Detection in AI Generated Videos

Shaif Chowdhury, Moshiur Rahman Tonmoy

Department of Electrical Engineering and Computer Science, Texas A&M University–Kingsville

Kingsville, TX 78363, USA

shaif.chowdhury@tamuk.edu

moshiur rahman.tonmoy@students.tamuk.edu

Abstract—Video generation models can create realistic-looking video footage from images and text prompts. Some of the recent advances have raised concerns about the reproduction of copyrighted visual content. However, copyright leakage detection in generated videos remains largely unexplored.

In this paper, we introduce a new benchmark data set for copyright leakage detection, comprising 50 copyrighted concepts, each with 5 images. For each concept, we have 5 videos created using the image+prompt video generation model and the promptonly video generation model. Following that, we develop an AI toolset based on a contrastive retrieval framework that formulates copyright leakage detection as a video-to-image retrieval problem by matching aggregated video frame embeddings against a database of copyrighted image embeddings. Our method uses a single similarity threshold and retrieves the most similar copyrighted reference as supporting evidence.

We conducted experiments showing strong retrieval and detection performance, achieving 99.6% detection accuracy with only a single false negative. These results show the effectiveness of the retrieval-based method for scalable and explainable copyright leakage detection in AI-generated videos. Data can be accessed publicly from https://zenodo.org/records/21386117 and the source code for this project is publicly available at https://github.com/Shaif95/video-cp-Leakage.

Keywords—video generation, copyright leakage, video understanding, contrastive learning, resnet

## I. INTRODUCTION

Recent advances in generative AI have made it relatively easy to create realistic digital content across modalities such as images, video, graphics, websites, art, and news media [1]. Within Computer Vision, AI-generated content has become one of the most important areas of research. Generative adversarial models initially gained popularity for their ability to generate photorealistic images. Following that, diffusion models [1] and autoregressive transformers have emerged as the most successful models for image generation. Based on the capability of diffusion models, several video generation models have been developed. These include text-to-video [2], image-to-video, video-to-video [3], video-editing, video generation [4] conditioned on image+text and many other video generation applications. With these impressive video generation capabilities, there are several legal and ethical challenges that we have to address. Recent research has shown that image and video generation models can recreate copyrighted content from the internet or other sources [5], [6]. In this paper, we focus on conditioning-based copyright propagation, where a copyrighted reference image is supplied as input to an image-to-video (I2V) pipeline; training-based memorization, where a model reproduces content memorized from its training data, is outside the scope of this work. We introduce a dataset for copyright detection in AI-generated video. Along with that, we propose a contrastive retrievalbased method for copyright detection. The goal is to provide a baseline and open dataset to promote more research in this area, leading to fair use of video generation models.

![](images/07502b6ef1eb8341344ad832bf7faf855f6573c9b3d13ea7fa20476b8aa20eeb.jpg)  
Fig. 1. Example copyrighted concepts and their five reference images used to build the reference database, showing the pose, lighting, and style variation the encoder must be invariant to.

## A. Background and Motivation

Stability AI is a company that has emerged as the leading model provider for image and video generation using AI. Currently, Stability AI is facing several lawsuits in the U.S. and the UK for scraping images to train its generative AI model, Stable Diffusion. One of the landmark lawsuit by Getty images alleged that Stability AI has used 12 million of its photographs, captions, and metadata without permission [5]. As evidence, they have shown that earlier versions of Stable Diffusion sometimes replicated Getty watermarks. Another lawsuit by Disney, NBCUniversal, and Warner Bros. against Midjourney focuses on the generation of copyrighted material by AI models. The studios allege that the AI platform massinfringed their intellectual property by training models on copyrighted works to let users generate unauthorized, highquality images and videos of iconic characters [5], [7].

In this paper, we have created a dataset for copyright leakage detection in AI-generated video, with 50 copyrighted concepts, 5 reference images each, and 5 videos per concept from both image-conditioned and prompt-only generators. We also propose a contrastive retrieval method that detects leakage by matching video frame embeddings against a database of copyrighted image embeddings. This way, instead of just giving a yes/no answer, our framework can also point out which exact copyrighted image is being leaked. Both are released as an open baseline to support fairer video generation research.

## B. Limitations of Existing Copyright Detection Methods

Existing research on copyright leakage detection focuses on image generation [8]. While image generation has a significant risk of copyright infringement [8], [9], video generation can often generate copyrighted characters as well as personal identities. This makes it necessary to focus on the detection of copyrighted content for video-based models. Existing research also lacks open data sets necessary for continued research [5], [7], [8], [10]. We aim to establish an introductory benchmark and tool chain that can encourage more work in this area.

## C. Contributions

The main contributions of this work are summarized as follows:

• We introduce a new benchmark data set for conditioning-based copyright leakage detection in AIgenerated videos, consisting of 50 copyrighted concepts, reference images, and paired image-conditioned and prompt-only generated videos.

• We provide a simple, reproducible retrieval baseline that formulates copyright leakage detection as a videoto-image retrieval problem by matching aggregated video frame embeddings against a database of copyrighted reference image embeddings. Rather than proposing a new architecture or loss, we deliberately build on standard components (a ResNet-18 encoder trained with SupCon and nearest-neighbor search) so that the benchmark is paired with a transparent and easily extensible pipeline.

We also establish an evaluation protocol for copyright leakage detection using both retrieval and classification metrics.

• Extensive experiments demonstrate the effectiveness of the proposed framework, achieving high retrieval performance and a detection accuracy of 99.6% on the proposed benchmark. Together, this data set and the AI tool set can encourage more research in this area and lead to mitigation methods.

## II. RELATED WORK

## A. Video Generation Models

Diffusion models have emerged as the dominant paradigm for state-of-the-art high-fidelity video generation. Building on image diffusion, Video Diffusion Models [11] and Make-A-Video [2] extended denoising generation to the temporal domain, and more recent latent video diffusion approaches such as Stable Video Diffusion [1] scaled this to high-resolution, image-conditioned generation. In parallel, powerful text-toimage (T2I) diffusion models such as SDXL [12] have become widely adopted as keyframe generators in video production pipelines, whereas dedicated video generation frameworks, including AnimateDiff [13], ModelScopeT2V [4] (and its community fine-tuned derivative Zeroscope), and CogVideoX [14], support both text-to-video (T2V) and image-to-video (I2V) synthesis. A defining property of these models is that they are conditioned on external images or text and trained on large web-scraped corpora, through which copyrighted visual content can propagate into their outputs. Concretely, copyrighted visual content may enter a generated video through two distinct mechanisms: training-based memorization, in which representations learned from a web-scale corpus reappear in outputs, and conditioning-based propagation, in which the visual identity of a reference image or prompt is preserved through the generation pipeline. The latter mechanism, prevalent in image-to-video and key-frame-driven production workflows, remains relatively underexplored and is the focus of this work.

## B. Copyright, Memorization, and Replication in Visual Generative Models

The question of whether visual generative models reproduce copyrighted content has moved from a legal concern to an active research problem [7], [15]. A growing body of work shows that large-scale diffusion models do not merely generalize but can memorize and reproduce individual training examples. Notably, Carlini et al. [10] demonstrated that nearverbatim copies of training images can be extracted from diffusion models, while Somepalli et al. [5] showed that a nontrivial fraction of generated images are partial or full replications of specific training samples, detectable by retrieving the nearest training image for each generation. Follow-up work analyzed the causes of this behavior and proposed mitigations at generation time [16], [17]. A complementary, mitigationoriented line removes protected visual concepts or styles from the model itself through concept erasure [18] or concept ablation [19]. However, existing approaches almost exclusively address the training-based replication channel, operating on static images, matching generated images against training images in a single modality, and typically assuming access to the model or its training data. In contrast, the conditioningbased propagation channel, where a copyrighted reference is deliberately supplied as input to an I2V pipeline, and the video setting more broadly remain relatively underexplored. Yet this setting is a growing and legally salient risk: modern production workflows routinely animate reference frames into full sequences, and the resulting videos may reproduce protected characters, costumes, or cinematic identities without authorization. Crucially, whether such propagation is legally infringing is a downstream determination; the technical task we address is to flag and attribute candidate reproductions to a reference database, providing an interpretable audit signal rather than a legal verdict

## C. Contrastive Representation Learning

Contrastive representation learning has emerged as a dominant paradigm for self-supervised feature extraction, mapping data into a latent embedding space where positive pairs are pulled together while negative pairs are pushed apart, yielding representations invariant to nuisance transformations such as cropping, color jitter, and blur [20]–[23]. Supervised contrastive learning (SupCon) [24], [25] generalizes this objective to leverage label information, pulling together all samples sharing a class while separating differing classes, which produces tighter, more cluster-structured embeddings. Beyond hand-crafted augmentations, self-distillation methods such as DINO [26] learn similarly discriminative features without negatives, and large vision-language models such as CLIP [27] provide strong general-purpose embeddings from image–text contrast. Overall, the invariance and instance-discriminative structure induced by these contrastive objectives are precisely the properties needed to recognize the same copyrighted identity across pose, lighting, and stylistic variations introduced by generative pipelines, making contrastive representations a natural foundation for a similarity-based leakage detector.

## D. Cross-Modal and Video Retrieval

Detecting copyrighted content through similarity search is closely related to content-based copy detection, which seeks to identify near-duplicate or transformed versions of a reference item. In the image domain, the DISC21 benchmark [9] spurred descriptors robust to strong edits, most notably SSCD [6], a self-supervised copy-detection embedding that treats copy detection as instance-level retrieval. In the video domain, segment- and video-level copy detection has been studied on large benchmarks such as VCSL [28], and crossmodal retrieval methods built on contrastive vision–language embeddings [3], [27] match content across text, image, and video. However, in generated videos, frames may deviate from any real reference through synthesis artifacts, motion, and stylistic drift. Thus, matching it against static, real copyrighted images is an inherently asymmetric and cross-domain problem.

![](images/c754b7165a2b93a633ff3c5ba874f76287308b5db093c5213a950d2fa60c0209.jpg)  
Fig. 2. Construction of the benchmark data set. Reference images from 50 copyrighted concepts are animated into copyright videos via imageconditioned generation (SDXL img2img followed by Stable Video Diffusion), while 50 original fictional concepts are rendered directly from text prompts using Zeroscope to form the non-copyright videos. The two branches together form the final benchmark.

Moreover, conditioning-based propagation preserves identity (character, costume, palette) rather than pixels, requiring identity or visual concept-level rather than instance-level matching. We address both by aggregating per-frame embeddings into a single video-level representation and retrieving against a concept-structured reference database.

To the best of our knowledge, there is no open-source dataset available for research on copyright leakage and potentially illegal video generation. No existing framework performs retrospective, model-agnostic detection and tracing of copyrighted visual content in generated video via video-to-image retrieval. This gap between the growing risk of copyright leakage and the lack of dedicated tools directly motivates the benchmark and framework proposed in this paper.

## III. METHOD

## A. Problem Formulation

Let $\mathcal { C } = \{ c _ { 1 } , \ldots , c _ { K } \}$ denote the set of K copyrighted concepts, each associated with a set of images $R _ { k } = \{ I _ { k } ^ { 1 } , \ldots , I _ { k } ^ { 5 } \}$ Here, $c _ { i }$ is a concept and $I _ { i }$ is an image. Given a generated video $V ~ = ~ ( F _ { 1 } , \ldots , F _ { T } )$ , the goal is to check whether it reproduces any of the reference images. Our method tries to learn an encoder $f _ { \theta } : \mathbb { R } ^ { H \times W \times 3 }  \mathbb { R } ^ { d }$ that maps an image to an embedding $z _ { i } = f _ { \theta } ( I _ { i } )$ . We build an image embedding database $\mathcal { D } = \{ ( z _ { k } ^ { j } , c _ { k } ) \}$ built from $R _ { K }$ . For a generated video $V ,$ we compute a single video embedding v and compare it against our image database D. Copyright leakage is then cast as a nearest-neighbor decision: V is flagged as leaking concept $c _ { k }$ if the similarity between v and its closest entry in D exceeds a threshold τ.

![](images/5afd8ea479ea9d35d9fe20ecf408a30d4bc7e0e12ff659303b6e574b3fcc0d9f.jpg)  
(a) Retrieval-based copyright detection  
Fig. 3. Overview of the proposed retrieval-based copyright detection framework. Reference images and generated video frames are mapped into a shared embedding space by a contrastive encoder (ResNet-18 trained with SupCon). The resulting video embedding is compared against the reference database, and a copyright leak is declared when its similarity to the nearest reference exceeds a threshold τ.

## B. Benchmark Dataset

The benchmark is developed using K=50 copyrighted concepts, each with 5 images collected from archive.org, resulting in a total dataset of 250 images. For each concept, we generate 5 videos using the video generation model with a prompt and an image as input. These 250 videos become positive samples for copyright leakage videos. We also create 250 videos based on 50 different concepts using only the text prompt as input to the video generation model. These become the negative samples for copyright leakage.

1) Copyright Video Generation: For each reference image $I _ { k } ^ { j } ,$ we synthesize a keyframe using an image-conditioned diffusion model $G _ { \mathrm { i m g } } , \hat { I } = G _ { \mathrm { i m g } } ( I _ { k } ^ { j } , p ; s )$ , where $p$ is a fixed cinematic prompt and s is the denoising strength controlling how closely <sup>ˆ</sup>I preserves the reference identity. The keyframe is then used as input to a video diffusion model generating a video, $V = G _ { \mathrm { s v d } } ( \hat { I } )$ . This two-stage process yields videos that are conditioned on copyrighted visual content.

2) Non-Copyright Video Generation: For the negative set, videos are produced by a text-to-video model $G _ { \mathrm { t } 2 \mathrm { v } }$ using a concept name and a generic prompt, $V = G _ { \mathrm { t 2 v } } ( p _ { k } )$ , with no image conditioning. Because $G _ { \Omega \mathrm { v } }$ never observes a reference image, these videos share the same generation pipeline structure as the copyright set but contain no copyrighted visual signal.

## C. Contrastive Image Representation Learning

We have used contrastive learning to generate embeddings from the images and videos. Given an input anchor image, contrastive learning generates an augmented view and learns representations by maximizing agreement between the anchor and positive sample, while increasing the distance between the anchor and negative sample in the feature space. Contrastive learning has an encoder that can be used to generate representations from images and videos. The encoder backbone is a ResNet-18 followed by a projection head $g _ { \phi } ,$ producing $z ~ = ~ g _ { \phi } ( f _ { \theta } ( I ) ) / \| g _ { \phi } ( f _ { \theta } ( I ) ) \|$ . We chose ResNet-18 deliberately: with only 250 reference images, a compact backbone reduces the risk of overfitting during contrastive training and keeps frame-level screening computationally inexpensive. Larger foundation backbones such as CLIP [27] or DINOv2 [26] are natural drop-in replacements that we leave for future work. For each image, we create two augmented views and train with the supervised contrastive loss: $\begin{array} { r } { \mathcal { L } _ { \mathrm { S u p C o n } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { | P ( i ) | } \sum _ { p \in P ( i ) } \mathrm { \bar { l o g } } \frac { \exp ( z _ { i } \cdot z _ { p } / \tau ) } { \sum _ { a \in A ( i ) } \exp ( z _ { i } \cdot z _ { a } / \tau ) } } \end{array}$

where P(i) is the set of augmented views of anchor image i, A(i) is all other views in the batch, and τ is a temperature parameter. This objective pulls embeddings of the same concept together and pushes different concepts apart.

## D. Reference Database Construction

After training, we use the encoder to generate features from images $z = f _ { \theta } ( I ) / \| f _ { \theta } ( I ) \|$ , forming the database $\mathcal { D } =$ $\{ ( z _ { k } ^ { j } , c _ { k } ) \} _ { k = 1 , j = 1 } ^ { \overline { { K } } , 5 } .$ . D is indexed for efficient nearest-neighbor search and serves as the fixed target when searching for copyright leakage among videos.

## E. Video Representation

We extract frames from every video $\{ F _ { t } \} _ { t = 1 } ^ { T }$ , and every frame is passed through the same encoder $f _ { \theta }$ to obtain frame

## Example Generated Videos from the Benchmark Dataset

Videos Generated with Copyright Images

![](images/4e523ce6c8cd515cfa8e66b925859184b07592e647ed671c4275c9e79e9fd86e.jpg)  
Fig. 4. Example frames from the benchmark’s generated videos. Top two rows: image-conditioned videos built from copyrighted reference images. Bottom two rows: prompt-only videos generated without any copyrighted reference. For both generation settings, the character-name-conditioned prompt used was: “Generate a cinematic video closely recreating the character. Preserve facial structure, costume, lighting, and mood. Subtle natural motion only. A [concept] character.”

Videos Generated Without Copyright Images

embeddings $\{ u _ { t } \}$ . We compute the average of these features to create a single video-level embedding.

$$
v = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } u _ { t } \Big / \Big \lVert \frac { 1 } { T } \sum _ { t = 1 } ^ { T } u _ { t } \Big \rVert ,\tag{1}
$$

This feature summarizes the video’s overall visual identity while remaining directly comparable to reference embeddings in D. Retrieval is therefore performed in a shared image embedding space (video-to-image retrieval) rather than through full spatio-temporal video modeling. Frame averaging suits the conditioning-based setting, where the copyrighted identity (character, costume, palette) is carried by appearance and persists across frames. However, it discards temporal order, so infringement expressed only through motion or sequence (e.g., choreography or a specific animated sequence) is beyond the scope of this work.

## F. Copyright Leakage Detection

Given a video embedding v, we compute its similarity to every entry in the image feature database D as $s ( v , z ) = \bar { v ^ { \top } }$ z and retrieve the nearest reference $z ^ { \star } = \arg \operatorname* { m a x } _ { z \in \mathcal { D } } s ( v , z )$ . Using that image, we also fetch the nearest concept label $c ^ { \star }$ . The video is declared a copyright leak of $c ^ { \star }$ if $s ( v , z ^ { \star } ) \geq \tau ,$ , and clean otherwise. So that τ becomes the parameter that directly controls the sensitivity of detection. Because the decision is based on nearest neighbor search, the framework also returns the specific copyrighted concept, providing retrieved reference evidence for each decision rather than an opaque label. This output is a screening signal for human or legal review, not a determination of infringement or fair use, which depends on legal factors such as transformative use that a similarity score cannot capture.

Similarity Score Distribution  
![](images/dd36644d9d3b0596c0a519d31ac2aed10c73833f4e24683d3af847a6bb4d2134.jpg)  
Fig. 5. Distribution of similarity scores between generated videos and their nearest reference embedding, split by copyright and non-copyright videos, with the learned threshold $\bar { \tau } \doteq 0 . 7 \bar { 7 }$ (red dashed line) separating the two populations.

Fig. 2 summarizes how the benchmark’s copyright and noncopyright videos are generated from their respective concept sets, and Fig. 3 illustrates the full detection pipeline, from contrastive encoding to the threshold-based copyright decision. The use of same encoder allows threshold rule to generalize across both video generation pipelines shown in Fig. 2.

TABLE I  
ABLATION OF THE DECISION STRATEGY: THE PROPOSED RETRIEVAL-BASED THRESHOLDING VERSUS STANDARD CLASSIFIERS USING IDENTICAL RESNET-18 + SUPCON EMBEDDINGS. ALL METHODS USE THE SAME ENCODER, REFERENCE DATABASE, AND TRAIN/TEST SPLIT, SO THIS IS NOT A COMPARISON WITH STATE-OF-THE-ART COPY DETECTION METHODS. BEST RESULTS ARE SHOWN IN BOLD.
<table><tr><td>Method</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td><td>ROC-AUC</td></tr><tr><td>Proposed: Retrieval + Threshold</td><td>0.9960</td><td>1.0000</td><td>0.9928</td><td>0.9964</td><td>0.9998</td></tr><tr><td>Linear SVM</td><td>0.8924</td><td>0.9308</td><td>0.8705</td><td>0.8996</td><td>0.9777</td></tr><tr><td>Random Forest</td><td>0.8765</td><td>0.8649</td><td>0.9209</td><td>0.8920</td><td>0.9442</td></tr><tr><td>Logistic Regression</td><td>0.8645</td><td>0.8723</td><td>0.8849</td><td>0.8786</td><td>0.9392</td></tr><tr><td>k-NN (k = 5)</td><td>0.8327</td><td>0.9802</td><td>0.7122</td><td>0.8250</td><td>0.9472</td></tr></table>

![](images/b5bd0d1d2a6d488e62a68e687ea7871b7874790863d638f88f00906371709475.jpg)  
Fig. 6. UMAP projection of the learned reference image embeddings, colored by concept. Images of the same copyrighted character form tight, wellseparated clusters.

## IV. RESULTS

## A. Dataset

The benchmark comprises 50 copyrighted concepts and 50 original non-copyright concepts, each with 5 generated videos, giving 500 videos in total across the image-conditioned and prompt-only pipelines. Fig. 1 shows example concepts with their five reference images. Videos are decoded frame-byframe, resized to $2 2 4 \times 2 2 4$ , and normalized before they are passed to the ResNet-18 encoder trained with contrastive learning. Each video is represented by the average of features and compared against the reference database via cosine similarity. 40% of the videos were used to create the decision threshold, and the rest were used for testing and analysis. We report binary classification accuracy, precision, recall, and confusion metrics computed on the held-out test set using the threshold learned on the training split.

## B. Copyright Detection Results

Using a threshold of $\tau = 0 . 7 7$ selected on the training split, our method achieves 99.6% accuracy on the test set, with

![](images/030d7009880ed56a87768610811c1f25af8992c81e4a526db7184ab66216ebbf.jpg)  
Fig. 7. t-SNE projection of generated video embeddings. Copyright videos (triangles) and non-copyright videos (squares) occupy largely distinct regions of the space.

138 of 139 copyright videos correctly flagged and all 112 non-copyright videos correctly cleared. This corresponds to a precision of 100% and a recall of 99.3%, with only a single false negative and zero false positives. Fig. 5 shows that the similarity scores of copyright and non-copyright videos form two well-separated distributions, with the learned threshold falling cleanly in the gap between them.

## C. Retrieval Results

Beyond the binary decision, our framework returns the specific reference image responsible for the highest similarity score, allowing each detected copyright leak to be traced back to a source image. In practice, the top-ranked retrieval consistently corresponds to the correct ground-truth concept for copyright videos, confirming that the embedding space is discriminative enough for localization and not just detection. Fig. 6 visualizes the learned reference image embeddings with UMAP, showing that images of the same copyrighted concept are grouped together. Fig. 7 shows the corresponding t-SNE projection of generated video embeddings, where copyright and non-copyright are clustered in different regions with few overlaps.

## D. Qualitative Results

Fig. 4 shows some example frames from the generated videos. The image-conditioned videos (top two rows) visibly preserve the identity, costume, and color palette of their copyrighted reference, while the prompt-only videos (bottom two rows) depict generic characters with no identifiable source. This contrast shows the difference in similarity scores between copyright and non-copyright videos.

## E. Comparison with Standard Classifiers

To compare against standard classifiers, we used the same ResNet-18 + SupCon encoder, reference database, and train/test split as the proposed method. Each video was represented by a cosine similarity vector computed between its embedding and all reference image embeddings (250 dimensions for the copyright-only database and 440 dimensions after adding non-copyright references). Logistic Regression, Linear SVM, Random Forest, and feature-space k-NN were trained using these similarity vectors, while the proposed retrieval method and image-database k-NN directly queried the reference database at inference time. Thus, all methods used identical features and differed only in their decision strategy; this experiment isolates the effect of the decision rule rather than comparing against state-of-the-art video copy detection descriptors such as SSCD [6] or frozen foundationmodel embeddings such as CLIP [27] and DINOv2 [26], which we leave for future work. Table I shows that the proposed retrieval-based thresholding approach consistently outperformed all competing methods.

## F. Ablation Study, Limitations and Future Work

In this paper, we use two different strategies to analyze video embeddings: feature averaging across all frames and taking the maximum per-frame similarity to the database. The late fusion of feature averaging allows a smoother, more stable similarity score, while max-frame similarity allows it to be more interpretable.

Beyond the detection framework and experiments, the value of this benchmark lies in its paired construction: because every copyrighted concept has a matched non-copyright counterpart generated with the same pipeline, it provides a controlled comparison for examining image conditioning. However, the benchmark should be viewed as a controlled pilot study: with 50 concepts, 250 reference images, and 500 videos, its scale is small, and results may not transfer to in-the-wild usergenerated videos, to subtler forms of copying such as style or likeness imitation, or to videos from unknown generators. The 99.6% accuracy may also be influenced by the controlled data set construction, and frame averaging does not capture infringement expressed only through motion. In addition, we have not yet evaluated robustness to common evasion edits such as cropping, color jittering, additive noise, or inpainting, which a deployed copyright detector must withstand. Our method also relies on a single embedding model, and evaluating alternative embeddings is an important direction for future work. A further deployment consideration is the privacy of user-provided reference images and audited videos. Extending copyright leakage detectors to a federated setting [29]– [32] is therefore a promising direction for privacy-preserving auditing across content platforms. In addition, future work could investigate lightweight and compression techniques [31], [33]–[35] to reduce the computational cost of frame-level copyright screening frameworks, thereby enabling on-device decision-making and advancing edge intelligence in AI-based copyright-auditing systems.

We release the data set, trained models, source code, reference embeddings, and evaluation protocol openly, in the hope of encouraging further research into detection and, ultimately, mitigation strategies that allow video generation models to be developed and deployed more fairly.

## V. CONCLUSION

In this paper, we address a largely unexplored problem: detecting conditioning-based copyright leakage in AI-generated video, where a copyrighted reference image is propagated through an image-to-video pipeline. As video generation models become stronger, they are increasingly capable of creating realistic video conditioned on copyrighted images and prompts. The models are able to reproduce protected characters, artwork, real human identities, etc. This creates legal and ethical concerns for model developers, platforms, and rights holders alike. To address this gap, we introduced a new benchmark data set of 50 copyrighted concepts, each paired with reference images and videos. We also create a set of non-copyright videos generated through an identical pipeline. This paired construction helps examine the role of image conditioning in copyright leakage and enables precise, reproducible evaluation.

Beyond copyright, this paper connects to the deepfake video problem, where generated videos are misused for impersonation and misinformation. Our retrieval approach can be extended for identity verification and content provenance too. We release our complete tool set, including the data set, models, embeddings, source code, and evaluation protocol, publicly. Future work includes extending detection to trainingbased memorization, where text-to-video models reproduce copyrighted content without image conditioning, exploring mitigation such as concept erasure, and testing robustness against adversarial evasion. We hope this encourages more research towards safer video generation. The data set will be publicly released upon publication; the data set link is omitted from the current submission to preserve anonymity.

## REFERENCES

[1] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.

[2] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni, et al., “Make-a-video: Text-to-video generation without text-video data,” arXiv preprint arXiv:2209.14792, 2022.

[3] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, and T. Li, “Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293–304, 2022.

[4] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.

[5] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Diffusion art or digital forgery? investigating data replication in diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6048–6058, 2023.

[6] E. Pizzi, S. D. Roy, S. N. Ravindra, P. Goyal, and M. Douze, “A selfsupervised descriptor for image copy detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14532–14542, 2022.

[7] A. F. Cooper and J. Grimmelmann, “The files are in the computer: on copyright, memorization, and generative ai,” Chi.-Kent L. Rev., vol. 100, p. 141, 2025.

[8] S. Roy, A. Mishra, A. S. Sharma, S. Karande, and M. Mandal, “Guardians of generation: Dynamic inference-time copyright shielding with adaptive guidance for ai image generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2551– 2561, 2025.

[9] M. Douze, G. Tolias, E. Pizzi, Z. Papakipos, L. Chanussot, F. Radenovic, T. Jenicek, M. Maximov, L. Leal-Taixe, I. Elezi, ´ et al., “The 2021 image similarity dataset and challenge,” arXiv preprint arXiv:2106.09672, 2021.

[10] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in 32nd USENIX security symposium (USENIX Security 23), pp. 5253–5270, 2023.

[11] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in neural information processing systems, vol. 35, pp. 8633–8646, 2022.

[12] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller,¨ J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” in International Conference on Learning Representations, vol. 2024, pp. 1862–1874, 2024.

[13] Y. Guo, C. Yang, A. Rao, Z. Liang, Y. Wang, Y. Qiao, M. Agrawala, D. Lin, and B. Dai, “Animatediff: Animate your personalized textto-image diffusion models without specific tuning,” arXiv preprint arXiv:2307.04725, 2023.

[14] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” in International Conference on Learning Representations, vol. 2025, pp. 83048–83077, 2025.

[15] Z. Jia, C. Huang, Y. Zhu, H. Fei, Y. Deng, Z. Yuan, J. Zhang, J. Zhang, and J. Zhou, “From imitation to innovation: The emergence of ai’s unique artistic styles and the challenge of copyright protection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18980–18989, 2025.

[16] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Understanding and mitigating copying in diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 47783–47803, 2023.

[17] Y. Wen, Y. Liu, C. Chen, and L. Lyu, “Detecting, explaining, and mitigating memorization in diffusion models,” in The Twelfth International Conference on Learning Representations, 2024.

[18] R. Gandikota, J. Materzynska, J. Fiotto-Kaufman, and D. Bau, “Erasing concepts from diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 2426–2436, 2023.

[19] N. Kumari, B. Zhang, S.-Y. Wang, E. Shechtman, R. Zhang, and J.-Y. Zhu, “Ablating concepts in text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, pp. 22691–22702, 2023.

[20] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning, pp. 1597–1607, PmLR, 2020.

[21] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for unsupervised visual representation learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.

[22] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[23] H. Xu, G. Ghosh, P.-Y. Huang, D. Okhonko, A. Aghajanyan, F. Metze, L. Zettlemoyer, and C. Feichtenhofer, “Videoclip: Contrastive pretraining for zero-shot video-text understanding,” in Proceedings of the

2021 conference on empirical methods in natural language processing, pp. 6787–6800, 2021.

[24] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” Advances in neural information processing systems, vol. 33, pp. 18661–18673, 2020.

[25] S. Chowdhury, M. Rahman, and G. Hamerly, “Supervised contrastive frame aggregation for video representation learning,” in International Symposium on Visual Computing, pp. 98–110, Springer, 2025.

[26] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[27] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning, pp. 8748–8763, PmLR, 2021.

[28] S. He, X. Yang, C. Jiang, G. Liang, W. Zhang, T. Pan, Q. Wang, F. Xu, C. Li, J. Liu, et al., “A large-scale comprehensive dataset and copy-overlap aware evaluation protocol for segment-level video copy detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21086–21095, 2022.

[29] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics, pp. 1273–1282, Pmlr, 2017.

[30] M. R. Tonmoy, M. M. Hossain, M. Safran, S. Alfarhood, D. Che, et al., “Leveraging federated learning for efficient privacy-enhancing violent activity recognition from videos,” Computers, Materials & Continua, vol. 85, no. 3, pp. 5747–5763, 2025.

[31] S. Chowdhury, A. Carney, G. Hamerly, and G. Speegle, “Acfed: communication-efficient & class-balancing federated learning with adaptive consensus dropout & model quantization,” in 2024 IEEE International Conference on Big Data (BigData), pp. 7707–7716, IEEE, 2024.

[32] C. Zhang, Y. Xie, H. Bai, B. Yu, W. Li, and Y. Gao, “A survey on federated learning,” Knowledge-based systems, vol. 216, p. 106775, 2021.

[33] M. R. Tonmoy, M. A. Adnan, S. M. R. Al Masud, M. Safran, S. Alfarhood, J. Shin, and M. Mridha, “Attention mechanism-based ultralightweight deep learning method for automated multi-fruit disease recognition system,” Agronomy Journal, vol. 117, no. 2, p. e70035, 2025.

[34] B. Rokh, A. Azarpeyvand, and A. Khanteymoori, “A comprehensive survey on model quantization for deep neural networks in image classification,” ACM Transactions on Intelligent Systems and Technology, vol. 14, no. 6, pp. 1–50, 2023.

[35] M. R. Tonmoy, A. F. Rakib, R. Rahman, M. A. Adnan, M. F. Mridha, J. Huang, and J. Shin, “A lightweight visual font style recognition with quantized convolutional autoencoder,” IEEE Open Journal of the Computer Society, vol. 5, pp. 120–130, 2024.