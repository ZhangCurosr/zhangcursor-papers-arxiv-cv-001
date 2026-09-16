# FROD: Feature Matching Residual Denoising Oracle Bone Decipher

Yanbin Hou<sup>1</sup>, Biao Xiong<sup>1</sup>, Guojun Xu<sup>1</sup>, Jianwen Xiang<sup>1</sup>, Cheng Tan<sup>1</sup>, Yanchao Yang<sup>1</sup>, and Junwei Zhou<sup>1(B)</sup>

School of Artificial Intelligence, Wuhan University of Technology, Wuhan 430070, China

houyanbin@whut.edu.cn, b.xiong@whut.edu.cn, guojunxu@whut.edu.cn,jwxiang@whut.edu.cn, cheng\_tan@whut.edu.cn, yangyc@whut.edu.cn,junweizhou@msn.com

Abstract. Oracle bone script (OBS), one of the earliest Chinese writing systems, plays an important role in the study of Chinese etymology. Traditional decipherment relies heavily on domain experts who analyze characters through semantic context and structural evolution. To assist this labor-intensive process, we formulate OBS decipherment assistance as a cross-era image translation task and propose FROD (Feature Matching Residual Denoising Oracle Bone Decipher). Although many OBS characters difer substantially from their modern counterparts, they often preserve local topological invariants at the radical level. During training, FROD leverages fast feature matching to provide gated segmentation supervision: paired samples with suficient matches are processed patch-wise to align fine-grained radicals, whereas low-similarity pairs are trained holistically to avoid mismatched artifacts. In addition, a Residual Denoising Difusion Model (RDDM) jointly estimates noise and residual signals, thereby reducing the positional drift and stroke disorder commonly observed in standard difusion models. Finally, a multi-stage font stylization refinement network refines the generated images by eliminating edge noise and stabilizing stroke structures. On our augmented character-disjoint dataset, FROD achieves higher Top-1 recognition accuracy than the evaluated baselines, with a 3.8% absolute gain over OBSD.

Keywords: oracle bone script, image translation, difusion model, feature matching

## 1 Introduction

Oracle bone script is one of the earliest forms of pictographic writing and was used during the Shang Dynasty. The study of oracle bone script (OBS) is fundamental to research on Chinese etymology and ancient history. However, among the approximately 4,500 unique characters discovered so far, only about 1,500 have been reliably deciphered. The decipherment process remains laborintensive, requiring substantial domain knowledge and associative reasoning to map ancient symbols to their modern counterparts.

Although end-to-end decipherment ultimately depends on contextual and linguistic evidence, generating visually recognizable modern Chinese character candidates from OBS rubbings can greatly accelerate the research workflow. Recent studies have therefore formulated this problem as an image-to-image translation task. In particular, OBSD [1] employs a difusion model with Local Structural Sampling (LSS) to align fine-grained patches between OBS and modern characters. However, OBS characters often exhibit substantial structural variation. Forcing highly abstract or weakly aligned pairs into naive patch-wise mappings can lead to severe positional drift, mismatched artifacts, and stroke disorder. In such cases, blind segmentation becomes unreliable because explicit structural correspondences are weak.

To address these challenges, we propose FROD, an image translation framework for OBS decipherment assistance. Compared with OBSD, our approach ofers three key improvements. First, rather than blindly segmenting all training pairs, we use the fast feature-matching model LightGlue [2] to provide gated segmentation supervision. During training, paired OBS and modern characters with suficient topological overlap, such as shared radicals, are processed patchwise to align local features, whereas highly variable pairs are trained holistically to preserve global structure. Second, we adopt the Residual Denoising Difusion Model (RDDM) [3] in place of a standard difusion model. By modeling target residuals and generated noise as separate components, RDDM alleviates the positional drift and structural bias that often arise in conventional noise-prediction frameworks. Third, we apply a multi-stage font stylization refinement network [4] to refine the synthesized outputs, suppress edge noise, and produce standardized modern glyphs.

Our main contributions are as follows:

We introduce a feature-matching-gated training strategy for OBS image translation. By assigning paired training samples to patch-wise or holistic supervision according to structural similarity, our method avoids detrimental misalignments while capturing local radical correspondences when appropriate.

– We adapt a Residual Denoising Difusion Model to the OBS domain, which explicitly models residual signals to better capture stroke positional patterns and preserve structural fidelity compared to standard difusion baselines.

– We integrate a font stylization refinement module to refine the predicted glyphs. Extensive experiments show that the full pipeline achieves the best OCR accuracy among the evaluated baselines on our augmented characterdisjoint dataset and improves Top-1 OCR accuracy over OBSD.

By producing reliable modern character candidates, FROD helps bridge computer vision and archaeology and provides a practical assistive tool for OBS decipherment.

## 2 Related Work

OBS research has advanced substantially with the release of several digitized datasets. Representative collections include Oracle-20K [5], OBC306 [6], and HWOBC [7]. However, these early datasets focus primarily on isolated character recognition and do not provide structural evolutionary mappings between OBS and modern Chinese characters. More recent resources, such as EVOBC [8], HUST-OBC [9], and OBC-V [10], address this limitation by providing moderncharacter correspondences and expanded class coverage, thereby laying the foundation for translation-based approaches.

For image-to-image translation, conditional Generative Adversarial Networks (GANs) such as Pix2Pix [11], CycleGAN [12], and DRIT++ [13], as well as difusion models such as Palette [14] and BBDM [15], have shown strong performance. However, they often struggle to align highly deformed cross-era topologies without explicit structural guidance.

Within OBS studies, most prior work has focused on character recognition using computer vision or natural language processing techniques, whereas AIassisted interpretation of undeciphered characters remains limited. Zhang et al. [16] proposed a case-based reasoning method that retrieves structurally similar cases from adjacent writing systems to assist expert decipherment. Chang et al. [17] introduced a cascaded GAN framework that models intermediate stages of Chinese-character evolution from OBS to modern forms. Guan et al. [1] introduced a difusion-based approach for detailed OBS-modern character alignment, thereby providing clearer evolutionary links.

Nevertheless, existing methods often struggle to preserve fine-grained structural diferences between OBS and modern character images, which leads to positional drift, missing strokes, and blurred outputs. In addition, few pipelines include explicit font standardization after generation, which further limits the legibility of translated characters for downstream recognition.

## 3 Feature Matching and Segmentation

Due to the highly variable structure of OBS characters, establishing valid correlations with modern Chinese character images is essential. During paired training, blindly applying patch-wise supervision to highly abstract OBS-modern pairs often produces disordered or noisy strokes because blind segmentation forces alignments where explicit structural correspondences do not exist.

To address this issue, we introduce a training-time gating mechanism driven by fast feature matching. The number of high-confidence matched keypoints serves as a quantitative criterion for determining whether a paired training sample should receive patch-wise local supervision or holistic supervision.

## 3.1 Feature Matching and Gating Mechanism

We adopt LightGlue (LG) [2] to extract and match keypoints due to its eficiency on low-complexity images like skeletonized OBS strokes. LG utilizes selfand cross-attention layers enhanced by Rotary Positional Embeddings to capture spatial context. Based on these contextualized features, a lightweight head predicts point correspondences. Let $\mathbf { x } _ { i } ^ { I }$ denote the descriptor of the i-th keypoint from the modern character image and $\mathbf { x } _ { j } ^ { S }$ denote the descriptor of the j-th keypoint from the paired OBS image, where M and N are the corresponding numbers of detected keypoints. The similarity score matrix $\mathbf { S } \in \mathbb { R } ^ { M \times N }$ is computed as

![](images/ed8a22476d6b7a863748f9fe09a819d11f3be1d4c8a0008342ac019dd5d7e869.jpg)  
Fig. 1. Training-time gating mechanism for OBS (left) and modern characters (right). Top: dense keypoint matches $( n _ { \mathrm { m a t c h } } > \gamma )$ trigger patch-wise local segmentation for radical alignment. Bottom: sparse matches $( n _ { \mathrm { m a t c h } } \leq \gamma )$ lead to holistic processing to preserve global character topology, where $\gamma = 1 5$

$$
S _ { i j } = ( \mathbf { W } _ { m a t c h } \mathbf { x } _ { i } ^ { I } ) ^ { \top } ( \mathbf { W } _ { m a t c h } \mathbf { x } _ { j } ^ { S } ) .\tag{1}
$$

With point-wise matchability confidences $\beta _ { i } ^ { I } , \beta _ { j } ^ { S } \in [ 0 , 1 ]$ , the soft assignment matrix O is formulated as

$$
O _ { i j } = \beta _ { i } ^ { I } \cdot \beta _ { j } ^ { S } \cdot \mathrm { S o f t m a x } _ { k \in I } ( S _ { k j } ) _ { i } \cdot \mathrm { S o f t m a x } _ { k \in S } ( S _ { i k } ) _ { j } .\tag{2}
$$

A match is considered valid if it is mutually maximal and its score $O _ { i j }$ strictly exceeds a confidence threshold τ. The number of valid matches is not treated as a semantic equivalence measure; instead, it serves as a practical proxy for the reliability of local structural correspondences between paired glyph images.

Gating Threshold (γ): During training, we count the total number of valid matches for each paired OBS-modern image sample. We set $\gamma = 1 5$ as a conservative empirical threshold to select pairs with suficiently dense local correspondences for patch-wise supervision. If the match count exceeds this threshold, the pair exhibits suficient structural correlation and is routed to the patch-wise segmentation module. Otherwise, it bypasses segmentation and is trained holistically to preserve global topology. As shown in Fig. 1, the upper examples exceed the threshold γ, triggering segmentation, while the lower ones do not.

## 3.2 Image Segmentation Method

For paired training samples that satisfy the gating condition, we adopt an LSS strategy. Specifically, the resized 100×100 input image $I \in \mathbb { R } ^ { 1 0 0 \times 1 0 0 \times C }$ is divided into $D = 8$ overlapping local regions of size $p \times p$ (with $p = 6 4 )$ using a sliding window. To obtain exactly eight local regions while maintaining suficient coverage of the character strokes, we use a grid sampling strategy with appropriate strides. In implementation, each local region is represented on a fixed $1 0 0 \times 1 0 0$ canvas before being fed into the difusion network, while holistic samples also use the original $1 0 0 \times 1 0 0$ image. Thus, patch-wise and holistic training share the same network input resolution; $p$ only defines the local support region used for segmentation and blending.

![](images/227ff52405c085c1fc43a25162db0c644444b7604c20e3ce91ba7441334de7f2.jpg)  
Fig. 2. Steps of the residual denoising difusion model.

These local patch representations, denoted as $I _ { 0 } ^ { ( d ) }$ for the modern character and $\tilde { I } ^ { ( d ) }$ for the corresponding OBS character, are processed independently during training. A Gaussian blending mask $\mathbf { P } _ { d }$ is generated for each patch position, which is later utilized during the reverse generation process to seamlessly blend overlapping regions and prevent visible seam artifacts.

## 4 Residual Denoising Difusion Method

Traditional conditional difusion models typically estimate either the target image or the injected noise. Although methods such as I<sup>2</sup>SB [18] model bridges between paired image domains, they do not explicitly decouple deterministic domain residuals from stochastic noise. For OBS decipherment, where the structural gap between paired glyphs can be large, we adapt the RDDM framework of Liu et al. [3] to OBS image translation. By explicitly treating the target residual and generated noise as independent components, this dual-estimation framework preserves the OBS structural condition more faithfully. The overall process is illustrated in Fig. 2.

## 4.1 Residual Denoising Formulation

Following RDDM [3], we redefine the difusion process to transition from the target modern character image $I _ { 0 }$ to a state heavily dependent on the conditional OBS image <sup>˜</sup>I. We define the domain residual as $I _ { \mathrm { r e s } } = \tilde { I } - I _ { 0 }$

Forward Difusion Process: The forward process gradually adds noise and shifts the mean toward <sup>˜</sup>I. The marginal distribution at step t is formulated

Algorithm 1 Patch-wise Reverse Sampling with Gaussian Blending   
Input: OBS image ${ \tilde { I } } ;$ conditional networks $\epsilon _ { \theta } ( \cdot , \cdot , t )$ and $I _ { \mathrm { r e s } , \theta } ( \cdot , \cdot , t ) ;$ ; Gaussian blending   
masks $\mathbf { P } _ { d }$   
Output: Preliminary translated modern character $I _ { 0 }$   
1: Sample initial noise $\underline { { \mathbf { z } } } _ { T } \sim \mathcal { N } ( 0 , \mathbf { I } )$   
2: Initialize $I _ { T }  \tilde { I } + \bar { \beta } _ { T } \mathbf { z } _ { T }$   
3: Set stability constant $c \gets 1 0 ^ { - 8 }$   
4: for $t = T$ downto 1 do   
5: Initialize accumulators $\Sigma _ { t }  \mathbf { 0 } , \varPhi _ { t }  \mathbf { 0 }$ , and $\mathbf M \gets \mathbf 0$   
6: for $d = 1$ to 8 do   
7: Extract local region d from $I _ { t }$ and embed it into a fixed-size canvas $I _ { t } ^ { ( d ) }$   
8: Extract local region d from I<sup>˜</sup> and embed it into a fixed-size canvas $\tilde { I } ^ { ( d ) }$   
9: Predict noise $\hat { \epsilon } _ { t } ^ { ( \vec { d } ) } \gets \epsilon _ { \theta } \big ( I _ { t } ^ { ( d ) } , \tilde { I } ^ { ( d ) } , t \big )$   
10: Predict residual $\hat { I } _ { \mathrm { r e s } , t } ^ { ( d ) } \gets I _ { \mathrm { r e s } , \theta } ( I _ { t } ^ { ( d ) } , \tilde { I } ^ { ( d ) } , t )$   
11: Crop the valid local support from $\hat { \epsilon } _ { t } ^ { ( d ) }$ and $\hat { I } _ { \mathrm { r e s } , t } ^ { ( d ) }$ , and project it back to local   
region d   
12: $\Sigma _ { t }  \Sigma _ { t } + \mathrm { P r o j } _ { d } ( \hat { \epsilon } _ { t } ^ { ( d ) } )$ ⊙ $\mathbf { P } _ { d }$   
13: $\Phi _ { t } \gets \Phi _ { t } + \mathrm { P r o j } _ { d } ( \hat { I } _ { \mathrm { r e s } , t } ^ { ( d ) } ) \odot \mathbf { P } _ { d }$   
14: $\mathbf { M } \gets \mathbf { M } + \mathbf { P } _ { d }$   
15: end for   
16: $\bar { \boldsymbol { \epsilon } } _ { t } \gets \boldsymbol { \Sigma } _ { t } / ( \mathbf { M } + \boldsymbol { c } )$   
17: $\bar { I } _ { \mathrm { r e s } , t } \gets \varPhi _ { t } / ( \mathbf { M } + c )$   
18: Sample step noise $\mathbf z _ { t } \sim \mathcal { N } ( 0 , \mathbf I ) { \mathrm { ~ i f ~ } } t > 1$ , else 0   
19: res\_term $ ( \bar { \alpha } _ { t } - \bar { \alpha } _ { t - 1 } ) \cdot \bar { I } _ { \mathrm { r e s } , t }$   
20: noise\_term $ ( \bar { \beta } _ { t } - \sqrt { \bar { \beta } _ { t - 1 } ^ { 2 } - \sigma _ { t } ^ { 2 } } ) \cdot \bar { \epsilon } _ { t }$   
21: $I _ { t - 1 }  I _ { t } - \mathrm { r e s } _ { - }$ term − noise\_term ${ } + \sigma _ { t } \cdot \mathbf { z } _ { t }$   
22: end for   
23: return $I _ { 0 }$

consistently using schedule parameters $\bar { \alpha } _ { t }$ and $\bar { \beta } _ { t }$ :

$$
q ( I _ { t } \mid I _ { 0 } , I _ { \mathrm { r e s } } ) = \mathcal { N } ( I _ { t } ; I _ { 0 } + \bar { \alpha } _ { t } I _ { \mathrm { r e s } } , \bar { \beta } _ { t } ^ { 2 } \mathbf { I } ) ,\tag{3}
$$

where I denotes the identity matrix. The parameter $\bar { \alpha } _ { t } \in [ 0 , 1 ]$ monotonically increases to 1 at $t = T$ , and $\bar { \beta } _ { t }$ represents the noise scale. Using the reparameterization trick, we can express $I _ { t }$ as

$$
I _ { t } = I _ { 0 } + \bar { \alpha } _ { t } I _ { \mathrm { r e s } } + \bar { \beta } _ { t } \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) .\tag{4}
$$

Notice that when $t = T , \bar { \alpha } _ { T } \approx 1$ , leading to the boundary condition $I _ { T } \approx \tilde { I } + \bar { \beta } _ { T } \epsilon$ This ensures that the reverse process initiates from a noisy version of the OBS image rather than pure Gaussian noise.

Reverse Generation Process: To reverse the process from $I _ { T }$ to $I _ { 0 }$ , the model employs two coupled networks to predict the residual and noise simultaneously: $I _ { \mathrm { r e s } , \theta } ( I _ { t } , \tilde { I } , t )$ and $\epsilon _ { \theta } ( I _ { t } , \tilde { I } , t )$ . The target image is estimated at each step as

$$
I _ { 0 , \theta } = I _ { t } - \bar { \alpha } _ { t } I _ { \mathrm { r e s } , \theta } - \bar { \beta } _ { t } \epsilon _ { \theta } .\tag{5}
$$

The reverse transition step $p _ { \theta } ( I _ { t - 1 } \mid I _ { t } , \tilde { I } )$ is parameterized as $\mathcal { N } ( I _ { t - 1 } ; \mu _ { \theta } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ Deriving from the posterior distribution $q ( I _ { t - 1 } \mid I _ { t } , I _ { 0 } , I _ { \mathrm { r e s } } )$ and substituting our estimates, the mean $\mu _ { \theta }$ is computed. The final sampling step becomes

$$
I _ { t - 1 } = I _ { t } - ( \bar { \alpha } _ { t } - \bar { \alpha } _ { t - 1 } ) I _ { \mathrm { r e s } , \theta } - \left( \bar { \beta } _ { t } - \sqrt { \bar { \beta } _ { t - 1 } ^ { 2 } - \sigma _ { t } ^ { 2 } } \right) \epsilon _ { \theta } + \sigma _ { t } \mathbf { z } ,\tag{6}
$$

where $\mathbf { z } \sim \mathcal { N } ( 0 , \mathbf { I } )$ , and $\sigma _ { t } ^ { 2 }$ is the reverse step variance controlled by a stochasticity parameter $\eta \in [ 0 , 1 ]$ . Following the RDDM sampling schedule, $\sigma _ { t }$ is chosen such that $\bar { \beta } _ { t - 1 } ^ { 2 } - \sigma _ { t } ^ { 2 } \ge 0$ , which keeps the square-root term well-defined.

The network is optimized using a weighted combination of the residual loss and noise loss:

$$
L ( \theta ) = L _ { \mathrm { r e s } } ( \theta ) + \lambda L _ { \epsilon } ( \theta ) ,\tag{7}
$$

$$
L _ { \mathrm { r e s } } ( \theta ) = \mathbb { E } _ { t , I _ { 0 } , \tilde { I } , \epsilon } \left[ \left\| I _ { \mathrm { r e s } } - I _ { \mathrm { r e s } , \theta } ( I _ { t } , \tilde { I } , t ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{8}
$$

$$
L _ { \epsilon } ( \theta ) = \mathbb { E } _ { t , I _ { 0 } , \tilde { I } , \epsilon } \left[ \Big \| \epsilon - \epsilon _ { \theta } ( I _ { t } , \tilde { I } , t ) \Big \| _ { 2 } ^ { 2 } \right] .\tag{9}
$$

## 4.2 Patch-wise Training Loss Adaptation

As discussed in Sect. 3.2, the paired OBS and target images are segmented into D local patches $\tilde { I } ^ { ( d ) }$ and $I _ { 0 } ^ { ( d ) }$ under the LSS strategy. The training objective averages prediction errors across patches and time steps. Accordingly, the expected loss functions are computed in a patch-wise manner:

$$
\hat { L } ( \theta ) = \hat { L } _ { \mathrm { r e s } } ( \theta ) + \lambda \hat { L } _ { \epsilon } ( \theta ) ,\tag{10}
$$

$$
\begin{array} { r } { \hat { L } _ { \mathrm { r e s } } ( \theta ) = \mathbb { E } _ { t , d , I _ { 0 } , \tilde { I } , \epsilon } \left[ \left\| I _ { \mathrm { r e s } } ^ { ( d ) } - I _ { \mathrm { r e s } , \theta } \big ( I _ { t } ^ { ( d ) } , \tilde { I } ^ { ( d ) } , t \big ) \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \hat { L } _ { \epsilon } ( \theta ) = \mathbb { E } _ { t , d , I _ { 0 } , \tilde { I } , \epsilon } \left[ \Big | \Big | \epsilon ^ { ( d ) } - \epsilon _ { \theta } \big ( I _ { t } ^ { ( d ) } , \tilde { I } ^ { ( d ) } , t \big ) \Big | \Big | _ { 2 } ^ { 2 } \right] . } \end{array}\tag{12}
$$

The goal is to minimize the diference between predicted and ground truth patch residuals $I _ { \mathrm { r e s } } ^ { ( d ) }$ and noise $\epsilon ^ { ( d ) }$ . Averaging the prediction errors across patches helps the model learn the localized distribution of structural correspondences, preventing overfitting to individual patches. Here, each patch residual is defined as $I _ { \mathrm { r e s } } ^ { ( d ) } = \tilde { I } ^ { ( d ) } - I _ { 0 } ^ { ( d ) }$ , consistent with the global residual definition.

For pairs that do not meet the gating threshold, no spatial decomposition is applied and $D$ is set to 1. This allows the same loss formulation to cover both patch-wise and holistic training samples.

## 4.3 Inference Process

At inference time, since paired modern targets are unavailable, the LightGluebased gate is not used. FROD adopts a fixed patch-wise reverse sampling strategy for every OBS input and reconstructs the final output using Gaussian blending. Specifically, the input OBS image <sup>˜</sup>I is decomposed into eight overlapping local regions, and each local region is translated at the same 100 × 100 network resolution as holistic processing. This inference-time patch decomposition does not impose additional paired local correspondence supervision; it serves as an input decomposition and blending strategy for recovering local strokes while preserving global consistency. Algorithm 1 summarizes the patch-wise reverse sampling procedure. During training, gated samples use the patch-wise losses in Sect. 3.2, whereas low-match samples use the same loss with D = 1 as holistic supervision.

![](images/92433b680a15c543057fa1ee9bd3490a22a8d9013950dde54fe28ad79f09d42b.jpg)  
Fig. 3. Steps of font stylization refinement.

## 5 Font Stylization Refinement

Although RDDM substantially improves the recovery of modern character topology from oracle bone script, the preliminary outputs often retain raw brush textures, structural fragmentation, and artifacts inherited from the ancient domain. To bridge the large domain gap between these coarse outputs and standard printed modern fonts, which is crucial for reliable downstream OCR rather than mere visual refinement, we introduce a multi-stage font stylization refinement network. This stage preserves the core character structure while regularizing stroke widths and removing raster artifacts.

Inspired by few-shot font generation methods, we adopt the MSD-Font architecture proposed in [4]. The font stylization refinement module maps coarse translated characters to clean, stylized modern fonts and thereby unifies visual representations for fair evaluation.

Although the modern font corpus covers the evaluated modern character categories, it contains only standard printed glyphs and no OBS images or OBS-modern paired samples. The font stylization refinement module is therefore trained independently as a modern-font structural and stylistic prior, rather than as OBS-to-modern paired supervision. During inference, the style image only provides font-level appearance guidance and is not the paired target glyph of the input OBS sample. This module serves as a post-generation standardization stage that regularizes the coarse RDDM output into a clean modern glyph form.

The overall MSD-Font architecture is shown in Fig. 3. We use the preliminary translation from the RDDM stage as the source image $I _ { s }$ . The source image is encoded by a Vector Quantized VAE (VQ-VAE) encoder into the latent feature z˜<sup>s</sup>. In parallel, a style image $I _ { g }$ is processed by the style encoder to produce the style condition $y _ { i }$ . The style image provides font-level appearance guidance and is not used as a paired target glyph for the test OBS input. During the forward difusion process, Gaussian noise is added to $\tilde { z } _ { 0 } ^ { s }$ to obtain the intermediate latent state $\tilde { z } _ { t _ { 2 } }$ . During reverse difusion, MSD-Font employs a style-conditioned prediction network $\tilde { z } ^ { ( r , i ) } ( \tilde { z } _ { t } , t , y _ { i } )$ to progressively transform the source latent toward a clean stylized result, and the VQ-VAE decoder finally outputs the generated modern glyph $I _ { r }$

The reverse process consists of three distinct stages: glyph construction, font transformation, and refinement. Starting from the noisy latent state associated with the source image, the model first reconstructs the coarse content structure of $I _ { s }$ during the glyph construction stage to anchor the basic topology. This stage simplifies to a single-step forward transition at an intermediate timestep $t _ { 2 } { \mathrm { : } }$

$$
\tilde { z } _ { t _ { 2 } } = \sqrt { \bar { \alpha } _ { t _ { 2 } } } \tilde { z } _ { 0 } ^ { s } + \sqrt { 1 - \bar { \alpha } _ { t _ { 2 } } } \epsilon _ { 0 } ,\tag{13}
$$

where $\bar { \alpha } _ { t _ { 2 } }$ is the predefined cumulative difusion coeficient at time $t _ { 2 } .$ , and $\epsilon _ { \mathrm { 0 } } \sim$ $\mathcal { N } ( 0 , \bf { I } )$

After obtaining the first-stage latent noise map $\tilde { z } _ { t _ { 2 } }$ , the font transformation stage progressively transforms it into the intermediate stylized latent $\tilde { z } _ { t _ { 1 } }$ using a style-conditioned network $\tilde { z } ^ { ( r , 1 ) } \big ( \tilde { z } _ { t } , t , y _ { 1 } \big )$ . Finally, the font refinement stage uses a secondary conditional network $\tilde { z } ^ { ( r , 2 ) } ( \tilde { z } _ { t } , t , y _ { 2 } )$ to refine $\tilde { z } _ { t _ { 1 } }$ , repairing local stroke intersections and yielding the final latent representation for the generated character. The VQ-VAE decoder then produces the final clear, stylized modern Chinese character image $I _ { r }$

## 6 Experimental Results and Analysis

## 6.1 Experimental Settings

During training, the difusion model is optimized with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ . We maintain an Exponential Moving Average (EMA) of model parameters with a decay rate of 0.995 to improve optimization stability. Training is conducted for 350 epochs with a batch size of 16. All network inputs are resized to 100×100 pixels for both patch-wise and holistic processing. For gated samples, we use eight overlapping local regions with p = 64 as the segmentation support, and the segmentation threshold is set to $\gamma = 1 5$

## 6.2 Datasets and Evaluation Metrics

We use OBC-V [10] as the base dataset and augment it with supplementary OBS images from EVOBC [8] and HUST-OBC [9]. This augmentation is performed by adding real OBS samples from external datasets rather than synthesizing new glyph images. After duplicate removal, the augmented dataset contains 74,219 OBS images mapped to 1,590 interpreted modern Chinese character classes. To prevent class-level leakage and ensure rigorous evaluation, we implement a character-disjoint train-test split while maintaining an approximately 9:1 split ratio. Specifically, all OBS samples belonging to the same modern Chinese character class are assigned exclusively to either the training set or the test set, and no modern character class appears in both sets. We further remove duplicate OBS instances before splitting and ensure that instances extracted from the same OBS source image are not shared across splits.

We evaluate the framework using both image quality metrics and recognition accuracy. Image Quality Metrics: We report standard metrics, including FID, RMSE, SSIM, and LPIPS. The generated characters are compared directly with the corresponding standard printed modern Chinese fonts used as ground-truth references. Recognition Accuracy (Top-k): To evaluate isolated-glyph automatic decipherment objectively, we deploy an OCR model. Specifically, we train a ResNet-50 classifier on a standard corpus of printed modern Chinese characters covering all 1,590 evaluated classes. The OCR classifier is trained on standard printed modern Chinese fonts and is used solely as an automatic recognizer for generated modern glyphs. The generated translation outputs are fed into this classifier to compute Top-1, Top-5, Top-10, Top-100, and Top-500 accuracies.

For the OBSD [1] baseline, we use the oficial implementation and retain its LSS-based initial decipherment and zero-shot refinement stages, so OBSD is evaluated as its complete published pipeline.

## 6.3 Comparison and Analysis of Results

Table 1 reports the quantitative image translation results. FROD achieves the best performance across all image-quality metrics. GAN-based methods, including Pix2Pix, CycleGAN, and DRIT++, perform poorly because they have limited capacity to model the severe structural deformations between OBS and modern characters. Difusion-based methods generalize better, and FROD further improves over the strongest baseline, OBSD, by learning more robust structural priors through RDDM.

Table 2 summarizes the OCR-based automatic decipherment results. FROD improves Top-1 accuracy over OBSD by 3.8 absolute percentage points (42.8% vs. 39.0%). This result indicates that the characters generated by FROD are more likely to be recognized as the correct modern character, suggesting that the proposed font stylization refinement and residual denoising mechanisms improve recognizability and complement the image-quality evaluation.

Fig. 4 presents a qualitative comparison. Although OBSD captures the overall structure, it frequently sufers from stroke omissions and misplacements because of its blind LSS strategy. The comparison between “FROD-w/o Font” and the full FROD output further shows that the font stylization refinement stage suppresses raster artifacts and regularizes stroke appearance. In contrast, FROD, trained with gated segmentation supervision and RDDM, reconstructs intricate stroke topologies with higher fidelity to modern character standards.

Table 1. Comparative evaluation of image translation quality.
<table><tr><td>Method</td><td>FID↓</td><td>RMSE↓</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Pix2Pix</td><td>246.08</td><td>0.5302</td><td>0.287</td><td>0.6037</td></tr><tr><td>DRIT++</td><td>201.77</td><td>0.4877</td><td>0.313</td><td>0.5312</td></tr><tr><td>CycleGAN</td><td>212.46</td><td>0.4618</td><td>0.321</td><td>0.5133</td></tr><tr><td>BBDM</td><td>122.52</td><td>0.4181</td><td>0.458</td><td>0.3978</td></tr><tr><td>OBSD</td><td>45.18</td><td>0.3037</td><td>0.622</td><td>0.2226</td></tr><tr><td>FROD</td><td>35.92</td><td>0.2788</td><td>0.689</td><td>0.1798</td></tr></table>

Table 2. OCR-based automatic decipherment evaluation of generated images on the character-disjoint augmented dataset.
<table><tr><td>Method</td><td>Top-1</td><td>Top-5</td><td>Top-10</td><td>Top-100</td><td>Top-500</td></tr><tr><td>Pix2Pix</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>1.9%</td><td>6.3%</td></tr><tr><td>DRIT++</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>2.5%</td><td>8.2%</td></tr><tr><td>CycleGAN</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>13.8%</td><td>20.1%</td></tr><tr><td>BBDM</td><td>18.2%</td><td>22.0%</td><td>23.9%</td><td>33.3%</td><td>37.1%</td></tr><tr><td>OBSD</td><td>39.0%</td><td>42.1%</td><td>45.2%</td><td>57.9%</td><td>61.6%</td></tr><tr><td>FROD</td><td>42.8%</td><td>44.0%</td><td>47.2%</td><td>59.1%</td><td>62.9%</td></tr></table>

Fig. 5 also illustrates representative failure cases. The primary limitations arise from incomplete modeling of complex character structures and insuficient recovery of critical local details. In these examples, the preliminary translation preserves only coarse topology, and the stylization stage may amplify structural ambiguities rather than resolve them. These failures explain part of the remaining gap in both image-quality metrics and OCR accuracy.

## 6.4 Ablation Study Results and Analysis

To isolate and validate the contribution of each component, we conduct ablation studies in Table 3. We consider the following variants:

FROD-AlwaysSeg: Removes the LightGlue threshold and blindly segments all training pairs.

FROD-StandardDif: Replaces RDDM with a standard noise-predicting conditional difusion model.

FROD-w/o Font: Removes the multi-stage font stylization refinement step.

The results show that “AlwaysSeg” performs substantially worse than the gated strategy, which supports our hypothesis that blindly segmenting highly abstract OBS pairs damages structural integrity. Replacing RDDM with a standard difusion model (“StandardDif”) increases positional drift and degrades FID. Finally, removing the font stylization refinement stage (“w/o Font”) leaves noticeable edge noise and raster artifacts, further confirming that each module is important for producing high-quality translation results.

<table><tr><td rowspan=1 colspan=1>Oracle BoneCharacters</td><td rowspan=1 colspan=1>ND</td><td rowspan=1 colspan=1>保</td><td rowspan=1 colspan=1>食</td><td rowspan=1 colspan=1>个</td><td rowspan=1 colspan=1>路</td></tr><tr><td rowspan=1 colspan=1>Ground Truth</td><td rowspan=1 colspan=1>白</td><td rowspan=1 colspan=1>保</td><td rowspan=1 colspan=1>魚</td><td rowspan=1 colspan=1>竹</td><td rowspan=1 colspan=1>陟</td></tr><tr><td rowspan=1 colspan=1>Pix2Pix</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>DRIT++</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>m一</td><td rowspan=1 colspan=1>7.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>CycleGAN</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>川</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>&lt;</td></tr><tr><td rowspan=1 colspan=1>BBDM</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>毛</td><td rowspan=1 colspan=1>非</td><td rowspan=1 colspan=1>陟</td></tr><tr><td rowspan=1 colspan=1>OBSD</td><td rowspan=1 colspan=1>脊</td><td rowspan=1 colspan=1>保</td><td rowspan=1 colspan=1>魚</td><td rowspan=1 colspan=1>竹</td><td rowspan=1 colspan=1>陟</td></tr><tr><td rowspan=1 colspan=1>FROD-w/o Font</td><td rowspan=1 colspan=1>沓</td><td rowspan=1 colspan=1>保</td><td rowspan=1 colspan=1>任</td><td rowspan=1 colspan=1>竹</td><td rowspan=1 colspan=1>陟</td></tr><tr><td rowspan=1 colspan=1>FROD</td><td rowspan=1 colspan=1>沓</td><td rowspan=1 colspan=1>保</td><td rowspan=1 colspan=1>魚</td><td rowspan=1 colspan=1>竹</td><td rowspan=1 colspan=1>陟</td></tr></table>

Fig. 4. Comparative analysis of oracle bone script image generation quality across diferent methods.  
![](images/2469b5f8157580b5af538236ed197a9ee16c0a5544828b8fb8ad2532cf9fdcad.jpg)  
Fig. 5. Decipherment failure cases.

Table 3. Ablation study of FROD components.
<table><tr><td>Method</td><td>FID↓</td><td>RMSE↓</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>FROD-AlwaysSeg</td><td>44.51</td><td>0.3121</td><td>0.601</td><td>0.2305</td></tr><tr><td>FROD-StandardDiff</td><td>49.33</td><td>0.3418</td><td>0.582</td><td>0.2511</td></tr><tr><td>FROD-w/o Font</td><td>56.99</td><td>0.3596</td><td>0.559</td><td>0.2833</td></tr><tr><td>FROD (Full)</td><td>35.92</td><td>0.2788</td><td>0.689</td><td>0.1798</td></tr></table>

## 6.5 Limitations

Although the character-disjoint split evaluates generalization to unseen modern character classes while preventing shared source instances across training and testing, the reported OCR accuracy should still be interpreted as an automatic proxy for isolated-glyph decipherment rather than definitive philological decipherment. End-to-end decipherment ultimately requires contextual, philological, and archaeological evidence beyond isolated glyph images. In addition, the current evaluation reports aggregate performance over the full character-disjoint augmented test set; more fine-grained analysis by stroke complexity, radical composition, and structural deformation will be valuable in future dataset releases.

## 7 Conclusion

In this paper, we introduced FROD, an image translation framework for assisting the decipherment of oracle bone script by generating recognizable modern Chinese character candidates. To address the severe structural variation between ancient and modern scripts, FROD uses a LightGlue-based feature matching mechanism to provide gated segmentation supervision, thereby improving local radical alignment without forcing mismatched patch correspondences. At inference time, FROD adopts a fixed patch-wise reverse sampling strategy reconstructed with Gaussian blending. We further adapt RDDM to model residual signals explicitly, which mitigates the positional drift and stroke disorder prevalent in standard difusion methods. Combined with a multi-stage font stylization refinement network, FROD produces clean and standardized modern character images. Extensive experiments show that the proposed method improves image generation quality and downstream OCR accuracy, yielding a 3.8% absolute gain in Top-1 accuracy over OBSD on our character-disjoint augmented dataset.

In future work, we plan to explore vector-based font generation for synthesizing scalable outlines and to investigate the integration of multi-task perceptual recognition losses into difusion training in order to further improve Top-1 accuracy and domain applicability.

## References

1. Guan, H., Yang, H., Wang, X., Han, S., Liu, Y., Jin, L., et al.: Deciphering Oracle Bone Language with Difusion Models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15554–15567 (2024)

2. Lindenberger, P., Sarlin, P.-E., Pollefeys, M.: LightGlue: Local feature matching at light speed. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17627–17638 (2023)

3. Liu, J., Wang, Q., Fan, H., Wang, Y., Tang, Y., Qu, L.: Residual denoising difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2773–2783 (2024)

4. Fu, B., Yu, F., Liu, A., Wang, Z., Wen, J., He, J., et al.: Generate like experts: Multi-stage font generation by incorporating font transfer process into difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6892–6901 (2024)

5. Guo, J., Wang, C., Roman-Rangel, E., Chao, H., Rui, Y.: Building hierarchical representations for oracle character and sketch recognition. IEEE Transactions on Image Processing 25(1), 104–118 (2016)

6. Huang, S., Wang, H., Liu, Y., Shi, X., Jin, L.: OBC306: A large-scale oracle bone character recognition dataset. In: Proceedings of the International Conference on Document Analysis and Recognition, pp. 681–688 (2019)

7. Li, B., Dai, Q., Gao, F., Zhu, W., Li, Q., Liu, Y.: HWOBC-A handwriting oracle bone character recognition database. Journal of Physics: Conference Series 1651(1), 012050 (2020)

8. Guan, H., Wan, J., Liu, Y., Wang, P., Zhang, K., Kuang, Z., et al.: An open dataset for the evolution of oracle bone characters: EVOBC. arXiv preprint arXiv:2401.12467 (2024)

9. Wang, P., Zhang, K., Wang, X., Han, S., Liu, Y., Wan, J., et al.: An open dataset for oracle bone character recognition and decipherment. Scientific Data 11, 976 (2024)

10. Zhou, J., Tu, Q., Xu, G.: Oracle character recognition using universal inverted bottleneck and inverse image frequency. International Journal on Document Analysis and Recognition 28(4), 609–622 (2025)

11. Isola, P., Zhu, J.-Y., Zhou, T., Efros, A.A.: Image-to-image translation with conditional adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1125–1134 (2017)

12. Zhu, J.-Y., Park, T., Isola, P., Efros, A.A.: Unpaired image-to-image translation using cycle-consistent adversarial networks. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2223–2232 (2017)

13. Lee, H.-Y., Tseng, H.-Y., Mao, Q., Huang, J.-B., Lu, Y.-D., Singh, M.K., et al.: DRIT++: Diverse image-to-image translation via disentangled representations. International Journal of Computer Vision 128(10–11), 2402–2417 (2020)

14. Saharia, C., Chan, W., Chang, H., Lee, C., Ho, J., Salimans, T., et al.: Palette: Image-to-image difusion models. In: ACM SIGGRAPH 2022 Conference Proceedings, pp. 1–10 (2022)

15. Li, B., Xue, K., Liu, B., Lai, Y.-K.: BBDM: Image-to-image translation with Brownian bridge difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1952–1961 (2023)

16. Zhang, G., Liu, D., Smyth, B., Dong, R.: Deciphering ancient Chinese oracle bone inscriptions using case-based reasoning. In: Sánchez-Ruiz, A.A., Floyd, M.W. (eds.) ICCBR 2021. LNCS, vol. 12877, pp. 309–324. Springer, Cham (2021)

17. Chang, X., Chao, F., Shang, C., Shen, Q.: Sundial-GAN: A cascade generative adversarial networks framework for deciphering oracle bone inscriptions. In: Proceedings of the 30th ACM International Conference on Multimedia, pp. 1195–1203 (2022)

18. Liu, G.-H., Vahdat, A., Huang, D.-A., Theodorou, E.A., Nie, W., Anandkumar, A.: I<sup>2</sup>SB: Image-to-image Schrödinger bridge. In: Proceedings of the 40th International Conference on Machine Learning, PMLR 202, pp. 22042–22062 (2023)