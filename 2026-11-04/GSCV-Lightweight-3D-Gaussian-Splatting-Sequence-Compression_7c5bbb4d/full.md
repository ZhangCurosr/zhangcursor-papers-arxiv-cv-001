# GSCV++: Lightweight 3D Gaussian Splatting Sequence Compression via Standard Video Codecs

Qi Yang, Member, IEEE, Yingzhan Xu, Kai Zhang, Senior Member, IEEE, Li Zhang, Senior Member, IEEE

Abstract—This paper presents GSCV++, a lightweight pipeline for 3D Gaussian Splatting (GS) sequences Compression with standard Video codecs. To meet the requirements of low-complexity and hardware compatibility from the industry, GSCV++ jointly improves preprocessing, 3D-to-2D mapping, and practical video compression configuration of GS compression. First, for preprocessing, energy-adaptive inter-frame spherical harmonic (SH) alternative current (AC) dimensionality reduction (EA-SHDR) is proposed to produces a compact and stable temporal SH AC feature space. It can reduce the data volume of following mapping and practical compression, as well as less complexity while eliminating redundant information. Second, for 3D-to-2D mapping, prior-based Inter-MiniPLAS (P-MiniPLAS) is designed to generate smooth 2D feature maps through prior-based weighting factor inference and small scale remapping. Compared with vanilla Parallel Linear Assignment Sorting (PLAS) and Morton scan, P-MiniPLAS ensures the generated 2D maps are more suitable for the video codec while maintaining a low level of computational complexity. Third, for practica compression, content-aware generalized compression (CA-GC) is developed to use a non-uniform bit-depth split for coordinate map generation, and a coding unit size selection is introduced for efficient compression complexity control, which fully exploits the performance of video codecs without involving normative changes. Experiments demonstrate that GSCV++ reports over 20% in intraand 60% in inter-prediction rate-distortion gains compared with state-of-the-art (SOTA) methods. GSCV++ also achieves faster 2D map generation, encoding, and decoding than other SOTA methods using the same type of video codecs. Ablation studies verify the effectiveness and robustness of the proposed modules and provide insights for subsequent research.

Index Terms—Gaussian splatting, compression, video codec

## 1 INTRODUCTION

3 <sup>D</sup> <sup>Gaussian</sup> <sup>splatting</sup> <sup>(GS)</sup> <sup>[1]</sup> <sup>achieves</sup> <sup>state-of-the-art</sup>(SOTA) performance in 3D reconstruction, as well as an ing quality. Different from the neural radiance field (NeRF) [2] that embeds scene information into neural networks, GS represents 3D scenes using numerous primitives with multi-channel attributes, including geometric coordinates, color spherical harmonic (SH) direct current (DC), color SH alternating current (AC), opacity, scale, and rotation. The high-dimensional explicit representation imposes substantial memory and storage demands, leading to considerable attention to 3D GS compression in both academia and industry.

The Moving Picture Experts Group (MPEG) has considered two approaches to compress 3D GS: A-3DGS and I-3DGS [3]. A-3DGS uses multiview images/videos as input and generates a compact representation through an optimized reconstruction process, as in HAC [4], HAC++ [5], and CompGS [6]. I-3DGS takes generated 3D GS data as input, generates bitstreams, and reconstructs the symmetrical GS data, as in GGSC [7], LGSCV [8], and HGSC [9]. A-3DGS and I-3DGS differ in their input requirements, optimization scope, and computational resources. Generally, A-3DGS methods can report better rate-distortion (RD) results than I-3DGS [10], while also requiring higher encoding times and heavier computational resources. Recently, a joint industry proposal [11] motivates lightweight GS compression through the reuse of standard video and point cloud codecs, such as HEVC [12] and GPCC [13], to reduce deployment and development costs. Therefore, I-3DGS is regarded as a higher priority, i.e., fast track, in the current standardization stage, such as V-PCC AMD1 [14].

![](images/bb1d3d8275efa2785d5d495b56aa40e28379404183f04e796c5058665979cf9b.jpg)  
Fig. 1. Comparison of video-based GS compression pipelines. GSCodec Studio, GSCV, and GSCV++ are contrasted across preprocessing, 3D-to-2D mapping, and compression.

In this paper, we follow the I-3DGS track and focus on reusing standard video codecs to compress preconstructed

GS sequences [11], which involve three main steps, as shown in Fig. 1: preprocessing, 3D-to-2D mapping, and the compression module. First, given a 3D GS sample, a preprocessing step is applied to reduce the data volume by removing redundant information, such as pruning [15], [16] and feature dimensionality reduction [17], [18]. Second, 3D GS to 2D mapping is a key step for efficient GS coding via video codecs. Because standard video codecs are optimized for natural image content, their performance improves as the generated 2D maps better match the distribution of natural videos. Finally, the compression module implements the actual compression process, converting the mapped 2D maps into standard bitstreams.

The current MPEG video-based GS compression anchor, i.e., GSCodec Studio [19], uses Parallel Linear Assignment Sorting (PLAS) [20] to convert GS into smooth 2D maps, followed by FFMPEG libx265 to perform compression. It uses a bit depth (BD) of 16 for geometric coordinates and 8 BD for other attributes to generate quantized images/videos. Based on preliminary results, GSCodec Studio performs stably and effectively on the MPEG tracked dataset. However, there are three limitations of GSCodec Studio: 1) SH AC attributes occupy the majority of the data volume, and GSCodec Studio directly converts them into 2D maps and feeds them into the video codec, which can lead to a lower compression ratio and higher coding complexity. 2) The vanilla PLAS algorithm is a heuristic algorithm that progressively reorders the GS maps from an extremely large receptive field, which is relatively time-consuming (see section 3.1 for details) and difficult to converge to optimal results. For example, PLAS requires around 30 seconds on GS samples containing approximately half a million primitives. 3) For sequence compression, GSCodec Studio reuses the I-frame PLAS permutation for later frames and, therefore, assumes consistent primitive indices. If no tracked information is provided by the GS sequences, or if some preprocessing operators (e.g., pruning) destroy the tracked information, the P-frame 2D maps resemble noise-like images and incur poor inter-frame compression results, limiting the applicability of this strategy.

The current study has developed methods to address the problems mentioned above. First, for preprocessing, some research has proven that SH AC illustrates high interchannel redundancy [21] for a single GS frame. However, how to reduce inter-frame SH AC redundancy while keeping stable rendering quality still requires further study. Second, for 3D-to-2D mapping, a more efficient and lightweight 2D map generation method is proposed by LGSCV [8], in which the two-stage Morton initialization and MiniPLAS are utilized. However, it also only considers the all-intra case. Third, for sequence compression, GSCV [22] proposes an inter-PLAS algorithm to build the track information between adjacent frames, which is more robust to different GS sequences. However, inter-PLAS significantly increases the computational complexity; a more efficient correspondence reconstruction is desired.

To develop a more effective and lightweight GS sequence compression framework, we propose GSCV++ to improve the RD and processing-time trade-off without supplied inter-frame tracked information. First, we design an energy-adaptive inter-frame SH AC dimensionality reduction (EA-SHDR) module. Based on principal component analysis (PCA), EA-SHDR selects an I-frame dimension using explained variance and reuses the basis and mean for the following frames, which can extend earlier all-intra dimensionality reduction [8], [17] to a common and stable feature space within a GoP in an extremely low-complexity implementation.

Second, a prior-based Inter-MiniPLAS refinement (P-MiniPLAS) is proposed to realize fast and smooth 2D map generation for inter-mode. P-MiniPLAS is a variation of vanilla PLAS, which consists of two main enhancements: First, vanilla PLAS assigns equal weighting factors to all the GS attributes during the reordering process, meaning that all selected features use the same scaling factor in the reordering objective. However, GS attributes have different importance in terms of rendering quality, as well as different sensitivities to lossy compression [23]. Generally, for certain GS attributes, the smoother the 2D map is, the fewer bits are needed after compression. Therefore, inspired by the rate control study [24], we interpret the reordering process of PLAS as a form of bitrate pre-allocation across different GS attributes. In light of the lack of a universal rate control approach [25] or loss function [26] for GS attributes, we use the initial attribute data volume and the quantization parameter (QP) as the priors to infer weighting factors for the reordering operation. Second, the 2D map reordering process is different for I- and P-frames. For I-frames, P-MiniPLAS inherits the strategy of the two-stage Morton scan and MiniPLAS [8] to generate the block-wise 2D maps (see section 3.2 for details), while the prior-based weighting factors are introduced to enhance the MiniPLAS performance. For P-frames, we initialize block-wise 2D maps via the two-stage Morton scan first, followed by Inter-MiniPLAS to refine them. Specifically, Inter-MiniPLAS uses the I-frame 2D maps as an anchor and predefines a maximum block size (MBS). A finite set of block sizes, selected based on MBS and conforming to the coding unit (CU) sizes supported by the standard video codecs, is used to refine P-frame block-wise maps. Using P-MiniPLAS, a small scale of reordering can generate close I- and P-frame GS maps, leading to highly efficient inter-prediction.

For the compression module, we propose a contentaware generalized compression (CA-GC), in which two techniques are introduced. CA-GC is based on the compression framework designed by GSCV [22], in which 20 BD and 10 BD are used to quantize geometry coordinates and other attributes (see section 3.3). GA-GC replaces the uniform 10:10 coordinate split with a non-uniform split, such as 16:4 and 12:8. The most significant bit (MSB) maps use video coding, while the least significant bit (LSB) coding choice can depend on the compression mode and split. Second, we couple the encoder’s maximum CU selection with the MBS selection in P-MiniPLAS, achieving stable and robust encoding complexity control of real compression. We also optimize the implementation of CA-GC to be compatible with multiple prevalent video codecs and platforms, including HM18.0 [27], VTM23.11 [28], FFMPEG libx265, and FFMPEG libx264 [29].

Based on the proposed GSCV++, we conducted a serial experiment and ablation study on both the MPEG and Audio Video Coding Standard Workgroup of China (AVS)

GS sequences. The results show that GSCV++ can report significant gains in both intra- and inter-prediction modes compared to the SOTA method; e.g., the gains over GSCV are approximately 39.6% and 68.7% on MPEG data in all-intra and inter modes, respectively. Besides, GSCV++ requires less time to generate GS 2D maps, decreasing from 24.78 to 4.58 seconds for I-frames and from 74.60 to 14.67 seconds for P-frames. Last but not least, GSCV++ reports faster encoding and decoding speeds than GSCV and GSCodec Studio when using the same type of downstream codecs, indicating that the 2D videos generated by GSCV++ are smoother. The contributions are summarized as follows:

We propose a comprehensive 3D GS sequence compression method, GSCV++, which can achieve SOTA compression performance, as well as lightweight 2D map generation and fast coding speed.

GSCV++ applies optimizations across all stages of the GS sequence compression: EA-SHDR for preprocessing, P-MiniPLAS for 3D GS to 2D mapping, and CA-GC for the compression module. Stable and efficient compression of GS sequences is achieved even when tracked information is unavailable.

We perform extensive experiments to demonstrate the effectiveness, robustness, and generalization of GSCV++. It consistently delivers strong compression performance across varying datasets and compression configurations.

The remainder of this paper is organized as follows. Section 2 presents the related work on 3D GS compression. Section 3 introduces the background used in this paper, including the PLAS algorithm, two-stage Morton scan, Mini-PLAS, and the GSCV compression pipeline. Section 4 details the design of GSCV++. Section 5 presents the experimental results and examines the evaluated design choices. Section 6 concludes the article.

## 2 RELATED WORK

In this section, we first introduce GS compression in terms of both A-3DGS and I-3DGS tracks in academia, then summarize the explorations of the MPEG GSC working groups.

## 2.1 A-3DGS: Optimization-Based GS Compression

In the academic community, A-3DGS methods are commonly referred to as optimization-based or generative compression methods. The generation process of 3D GS offers considerable scope for more compact data representations, since the mapping from 3D GS representations to rendered images is not one-to-one: distinct 3D GS representations may yield similar rendered content and quality. Therefore, the input of A-3DGS methods is multiview images/videos, and a more compact GS representation is generated via an end-to-end training process. The compact representation can use the vanilla GS format or other explicit, implicit, or hybrid representations. Scaffold-GS [30] is an early example of this approach, in which a group of anchors was initialized with latent features. The explicit GS primitives can be reconstructed from the anchor information via MLPs. Inspired by Scaffold-GS, HAC [4] and HAC++ [5] were proposed by introducing an effective context model based on multiresolution hash coding, which can achieve approximately 60x to 100x reduced storage. CompGS and CompGS++ [6] also constructed compact latent features based on anchors, and both intra- and inter-prediction were implemented via anchor and residual features. In contrast, LightGaussian [31] approached the problem from the perspective of controlling the number of primitives and proposed a rendering-based pruning method. The experiment shows that removing 50% to 60% of primitives can achieve similar rendering quality after a limited number of iterations of fine-tuning. EAGLES [32] and HybridGS [21] used low-dimensional latent features to reconstruct high-dimensional explicit GS attributes, provided some GS features are highly redundant. Besides, HybridGS realizes fast encoding and decoding by combining standard point cloud encoders with compact GS data. Codebooks and vector quantization constitute another commonly used approach, such as [33], [34]. More recently, A-3DGS methods have been extended to the 4D domain to generate compact dynamic GS representations, such as 4DGC [35], ADC-GS [36], and Light4GS [37].

Optimization-based and dynamic GS methods can jointly redesign the representation and reconstruction processes, which can offer compression opportunities beyond coding an existing GS sequence. Generally, they have a higher requirements for encoding time, computational resources, as well as customized design. Our focus is narrower and inspired by industrial requirements: reuse standard video codecs for already constructed GS sequences without requiring scene-specific reconstruction optimization during compression.

## 2.2 I-3DGS: Optimization-Free GS Compression

To develop more practical GS compression methods, I-3DGS methods have attracted more attention in recent times. Different from A-3DGS, which involves a time-consuming optimization process, I-3DGS aims to compress GS while retaining the same representation type: both the input and output are the generated GS data. Inspired by point cloud compression, GGSC [7] designed a two-branch compression pipeline: using GPCC to compress the GS coordinates and graph signal processing [38] related techniques to compress other attributes. HGSC [9] proposed a hierarchical strategy for progressive primitive compression. FCGS [39] is the first method that realizes learning-based I-3DGS compression via both intra- and inter-primitive context models.

To facilitate the deployment of the 3D GS compression method, MPEG prioritizes the lightweight I-3DGS method based on the requirement proposal from the industry [11]: using standard video or point cloud codecs to achieve effective GS compression. GSCodec Studio [19] was selected as the video-based GS compression anchor, while GPCC v1 [40] was regarded as the point cloud-based anchor. Specifically, GSCodec Studio uses the PLAS algorithm [41] to convert 3D GS data into smooth 2D maps, followed by an FFMPEG libx265 codec to compress these maps. The proposal [42] merges multiple PLAS maps from a single GS frame into a single video and compresses it using HM18.0 [27]. LGSCV [8] combines MiniPLAS with PCA [17] and HM18.0 for efficient all-intra GS coding; it does not introduce inter-frame correspondence modeling. GSCV [22] is the pioneering work that studies GS sequence compression without tracked information: its Inter-PLAS matches laterframe attribute maps to an anchor when reliable primitive correspondence is unavailable. However, Inter-PLAS has very high complexity, and the performance of GSCV can still be significantly improved due to the lack of preprocessing optimization.

Although current video-based GS compression methods have demonstrated impressive compression performance, there is still room for more effective 2D map generation for both intra- and inter-prediction, as well as a more appropriate video codec configuration. This motivates the design of GSCV++ in this paper.

## 3 PRELIMINARY

In this section, we review three relevant components underlying GSCV++. First, we introduce the implementation of PLAS, which is the basis of the GS 3D-to-2D map generation. Second, the two-stage Morton scan and MiniPLAS are summarized, which form the basis of the proposed P-MiniPLAS. Third, the compression module designed in GSCV is used as the basic framework of GSCV++, leading to a more detailed description in this section.

## 3.1 Brief Introduction of PLAS

PLAS [20] is a heuristic algorithm that reorders GS primitives into smooth 2D attribute maps. The feature matrix $G \in \mathbb { R } ^ { n \times 5 9 }$ contains n primitives with 3 coordinate, 3 SH DC, 45 SH AC, 1 opacity, 3 scale, and 4 rotation channels. Minimal pruning based on scale and opacity fits the primitives to a square grid, followed by random initialization of $J _ { 0 } ~ = ~ I ~ \in ~ \mathbb { R } ^ { M \times M \times m }$ , where $m ~ \leq ~ 5 9$ channels guide smoothness evaluation. A common permutation preserves alignment across all attribute channels without additional signaling.

PLAS uses progressively smaller Gaussian-blur radii $r _ { i } ~ = ~ \lfloor r _ { 0 } \tau ^ { i - 1 } \rfloor$ , where $r _ { 0 } ~ = ~ M / 2 - 1 , ~ \tau ~ \in ~ ( 0 , 1 )$ is the decay factor, and the minimum radius is 1. At level $i ,$ the blur-kernel width is $K _ { i } \ = \ \operatorname* { m i n } ( M - 1 , 2 r _ { i } + 1 )$ and the permutation-block width is $B _ { i } = \operatorname* { m a x } \{ B _ { m i n } , 2 \lfloor \operatorname* { m i n } ( K _ { i } +$ ${ \bar { 1 } } , M ) / 2 | \}$ , with $B _ { m i n } = 1 6$ in vanilla PLAS. The target map is the Gaussian-blurred input, $T _ { i } = b l u r ( I _ { i - 1 } , K _ { i } )$

The input and target maps are partitioned into corresponding $\mathsf { \bar { B } } _ { i } \times \boldsymbol { B } _ { i }$ blocks for batched processing. Within each complete block, whose even width allows grouping into sets of four pixels, PLAS evaluates all 4! = 24 permutations of each randomly selected group and retains the assignment with the smallest L2 loss to the corresponding target pixels. The resulting map $I _ { i }$ is passed to the next level. Nonincreasing $K _ { i }$ and $B _ { i }$ yield coarse-to-fine refinement toward smooth attribute maps.

## 3.2 Two-Stage Morton Scan and MiniPLAS

## 3.2.1 Two-Stage Morton Scan

Given spatial coordinates $( x , y , z )$ , a 3D Morton code interleaves the binary representations of the coordinates, which can permute the spatially adjacent primitives into close indices [43]. Attributes of 3D GS primitives exhibit local spatial similarity, indicating that after 3D Morton sorting, primitives with adjacent indices tend to share similar attribute values. Then, the scan order of the 2D Morton code is used to generate 2D maps. The generated 2D maps exhibit noticeable block structures, with smooth content within each block, making them well-suited for video codecs.

## 3.2.2 MiniPLAS

MiniPLAS improves vanilla PLAS by defining a group of new block sizes for pixel permutation: the block sizes of permutation are defined as 4n $\times ~ 4 n , ~ n ~ = ~ 1 , 2 , 4 , \ldots$ . The selected block sizes are the CU or sub-CU sizes in video codecs, such as $8 \times 8 .$ MiniPLAS is also structured in a progressive manner. Based on the ablation study in LGSCV [8], for single-frame compression in all-intra mode, using a block size of $8 \times 8$ for two iterations is good enough, which is also adopted in the proposed GSCV++ I-frame 2D map generation.

## 3.3 Compression Module of GSCV

GSCV [22] combines 8 frames into one GoP, with the first frame designated as the I-frame and the remaining 7 frames as non-anchor frames (called P-frames here for map generation, irrespective of their codec picture types). 20 BD is used to quantize the coordinates and 10 BD is applied to quantize other attributes. For each GS frame, GSCV generates 1, 15, and 1 frame images for the color SH DC, SH AC, and scale. For opacity and rotation, GSCV uses the mid-value of BD to pad two channels to form 1 and 2 images, respectively. Most video codecs cannot compress 20 BD images directly. GSCV splits the coordinate map into 2 images, namely the high part and the low part of the data. Each part takes 10 BD. For each GS frame, 22 images are generated. GSCV extracts the corresponding images from each frame within a GoP to form 22 videos and saves them in YUV444 format. Lossless compression is used for coordinates, while other attributes employ the lossy mode. Note that GSCV directly considers the video data as YUV rather than RGB, omitting the operation that converts data from RGB to the YUV color space for YUV file generation. More details can be found in [22].

## 4 METHOD

Fig. 2 shows the framework of GSCV++, in which the main contributions are in three modules: EA-SHDR, P-MiniPLAS, and CA-GC. EA-SHDR belongs to the preprocessing step illustrated in Fig. 1, where we propose a lightweight method based on PCA to achieve color SH AC dimensionality reduction for inter-frame mode. P-MiniPLAS corresponds to the 3D-to-2D mapping procedure. Two technologies are included: 1) a prior-based PLAS weighting factor inference algorithm designed to assign different smoothness levels for different GS attributes, leading to bitrate pre-allocation; 2) a weighted version of MiniPLAS used for I-frame 2D map generation and an inter-MiniPLAS proposed for P-frame 2D map generation. CA-GC is an optimized compression module, in which we design a non-uniform BD split and a differentiated compression scheme for the GS geometric coordinate. The block sizes used in P-MiniPLAS are specified in the video codec CU configuration, enabling the development of a complexity-controlled compression module with a fast encoding speed. We will detail the proposed GSCV++ in the following part of this section.

![](images/a6c86b8c17ff35f940e38c27c308a8021582671fdf0b1b6ef5678eb2da8c9745.jpg)  
Fig. 2. Framework of GSCV++.

## 4.1 Energy-Adaptive Inter-Frame SH AC Dimensionality Reduction (EA-SHDR)

Previous studies [8], [17] have shown that the GS SH AC component is highly redundant; introducing dimensionality reduction before compression can yield significant RD gains. To facilitate GS sequence compression and improve the inter-prediction of SH AC, GSCV++ adopts different strategies for I- and P-frames. For an I-frame with N GS primitives, let $G _ { A C } ^ { I } \in \mathbb { R } ^ { N \times 4 5 }$ contain its 45 SH AC feature channels. Unlike LGSCV, which fixes the latent dimension, GSCV++ selects it using the cumulative explained variance ratio (EVR) of the I-frame features. The reduction process consists of three steps:

i) Centering and basis estimation. First, compute the perchannel mean µ<sup>I</sup> = mean $( G _ { A C } ^ { I } ) \ \in \ \mathbb { R } ^ { 1 \times 4 5 }$ . Subtract this mean from every primitive and apply SVD:

$$
\begin{array} { r l } { G _ { A C } ^ { I } = G _ { A C } ^ { I } - \mathbf { 1 } _ { N } \pmb { \mu } ^ { I } , \quad \overline { { G _ { A C } ^ { I } } } } & { { } = \mathbf { U } ^ { I } \pmb { \Sigma } ^ { I } ( \mathbf { V } ^ { I } ) ^ { \mathsf { T } } . } \end{array}\tag{1a}
$$

Here, ${ \bf 1 } _ { N }$ is an N-dimensional all-ones column vector, and $\mathbf { U } ^ { I }$ and $\mathbf { V } ^ { I }$ contain the left and right singular vectors. The diagonal entries $\boldsymbol { \sigma } _ { i } ^ { I }$ of $\Sigma ^ { I }$ are the singular values in descending order. The corresponding columns $\mathbf { v } _ { i } \in \mathbb { R } ^ { 4 5 }$ of $\mathbf { V } ^ { I }$ are the principal directions.

ii) Energy-adaptive dimension selection. The variance along principal direction $\mathbf { v } _ { i }$ is $\lambda _ { i } ~ = ~ ( \sigma _ { i } ^ { I } ) ^ { 2 } / ( N - 1 )$ . Given a target retained-variance fraction $A \in ( 0 , 1 ] ,$ select the fewest leading components whose cumulative explained variance reaches A:

$$
k ^ { * } = \operatorname* { m i n } \left\{ d \in \{ 1 , \dots , 4 5 \} \ : \left| \begin{array} { c } { { \sum _ { i = 1 } ^ { d } \lambda _ { i } } } \\ { { \sum _ { i = 1 } ^ { 4 5 } \lambda _ { i } } } \end{array} \geq { \cal A } \right. \right\} .\tag{1b}
$$

Thus, $k ^ { * }$ adapts to the SH AC variance distribution of the

I-frame.

iii) Three-channel packing and projection. Round the retained dimension up to $k = 3 \lceil k ^ { * } / 3 \rceil$ so that the latent channels form three-channel images without padding. Using the first k principal directions, define $\mathbf { V } _ { k } ^ { I } { \bf \bar { \Psi } } = \left[ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { k } \right] \ \in$ $\mathbb { R } ^ { 4 5 \times k }$ and project the centered features:

$$
z _ { A C } ^ { I } = \overline { { G _ { A C } ^ { I } } } \mathbf { V } _ { k } ^ { I } \in \mathbb { R } ^ { N \times k } ,\tag{1c}
$$

where $z _ { A C } ^ { I }$ is the reduced I-frame SH AC feature matrix.

Applying PCA independently to P-frames can produce different principal directions, hindering attribute matching in P-MiniPLAS and the subsequent inter-prediction process. We therefore reuse both the I-frame mean $\mu ^ { I }$ and basis $\mathbf { V } _ { k } ^ { I }$ to express the P-frame SH AC features in the same coordinate system:

$$
z _ { A C } ^ { P } = ( G _ { A C } ^ { P } - \mathbf { 1 } _ { N } \mu ^ { I } ) \mathbf { V } _ { k } ^ { I } \in \mathbb { R } ^ { N \times k } .\tag{2}
$$

The decoder reconstructs SH AC features as $\widehat { G } _ { A C } ^ { t } =$ $\widehat { z } _ { A C } ^ { t } ( \mathbf { V } _ { k } ^ { I } ) ^ { \mathsf { T } } + \mathbf { 1 } _ { N } \mu ^ { I }$ for both I- and P-frames. This method facilitates GS inter-frame compression and reduces preprocessing time by reusing the I-frame PCA results. The decoder uses the same basis and mean for all frames in the GoP.

## 4.2 Prior-Based Inter-MiniPLAS Refinement (P-MiniPLAS)

3D GS to 2D mapping is a key step because the characteristics of the generated 2D video will directly influence the effectiveness of the downstream standard video codecs. The proposed P-MiniPLAS module is an enhancement of PLAS [41] that adds attribute-aware weighting factors and interframe refinement.

## 4.2.1 Prior-based PLAS Weighting Factor Inference

For the I-frame GS samples, we first use the two-stage Morton scan [8] to generate block-wise 2D images and scale them into the same value range, e.g., [0,255]. Unlike LGSCV, which directly performs MiniPLAS on these 2D images, a weighting factor is calculated for each GS attribute to rescale its image values. These weighting factors act as featurescaling coefficients that can guide the subsequent permutation process, prioritizing the more important attributes instead of uniformly handling all attributes in terms of texture smoothness. There are two types of prior knowledge used to infer these weighting factors: initial attribute data volume and target bitrate.

First, assuming the same degree of lossy compression, e.g., the same QP, the larger initial data volume generally corresponds to a larger proportion of the final bitstream. If the permutation process of PLAS can generate smoother images for the attributes that originally contain larger amounts of data, a substantial reduction in the overall bitrate can be achieved. Second, for 3D GS, different attributes vary in their importance for rendering quality, as well as in their sensitivity to lossy compression. There is a consensus that geometric coordinates play the most critical role, and scale and opacity are more sensitive to lossy compression than color SH. However, we still lack a generalized model for quantifying the relative importance of different GS attributes. Nevertheless, valuable insights can be drawn from the QP design in MPEG GSC research: attributes of higher importance are typically assigned smaller QP values to achieve a more favorable RD trade-off.

Therefore, the weighting factors can be formulated as follows:

$$
W _ { a } = f ^ { 1 } ( D ^ { a } ) \cdot f ^ { 2 } ( Q P ^ { a } ) ,\tag{3}
$$

where a represents the specific attributes, such as SH $\mathrm { A C }$ and opacity, $D ^ { a }$ and $Q \bar { P ^ { a } }$ are the initial data volume and QP of the corresponding attribute. $f ^ { 1 } ( \cdot )$ and $f ^ { 2 } ( \cdot )$ are the mapping functions to derive the quantified parameters. For $f ^ { 1 } ( \cdot )$ , we define

$$
f ^ { 1 } ( D ^ { a } ) = B D ^ { a } \cdot C H ^ { a } ,\tag{4}
$$

where $B D ^ { a }$ and $C H ^ { a }$ are the BD and channel numbers of attribute a.

The calculation of $f ^ { 2 } ( \cdot )$ is inspired by Q-domain ratedistortion analysis from video coding [24]. The rate $R$ generally decreases as the QP increases. However, research shows that the relation between R and $1 / Q _ { s t e p }$ can be more accurately approximated by a linear model rather than the relation between R and $1 / { \dot { Q } } P .$ , therefore, we set

$$
f ^ { 2 } ( Q P ^ { a } ) = \frac { 1 } { Q _ { s t e p } ^ { a } } = 2 ^ { \frac { 4 - Q P ^ { a } } { 6 } } ,\tag{5}
$$

$$
R ^ { a } \approx K ^ { a } f ^ { 2 } ( Q P ^ { a } ) ^ { \beta ^ { a } } = K ^ { a } 2 ^ { \beta ^ { a } ( 4 - Q P ^ { a } ) / 6 } ,\tag{6}
$$

where $R ^ { a }$ is the estimated number of coded bits for attribute $a . ~ f ^ { 2 } ( Q P ^ { a } )$ shows a positive correlation with $R ^ { a }$ but is not strictly proportional, as $R ^ { a }$ is additionally influenced by content-dependent hyperparameters $K ^ { a }$ and $\beta ^ { a }$ . However, for our work, this correlation provides a sufficient, computationally inexpensive basis for the weighting factor; therefore, we simply assume $f ^ { 2 } ( Q P ^ { a } ) \propto R ^ { a }$ . Precisely computing $R ^ { a }$ incurs a significant computational overhead, as PLAS is a multi-iteration optimization process.

As a bitrate-allocation prior, $W _ { a }$ scales the normalized feature values before evaluating squared Euclidean distances, its contribution to the permutation cost is proportional to $W _ { a } ^ { 2 }$ . After we obtain $W _ { a }$ for each GS attribute, we use MiniPLAS to refine I-frame block-wise images to obtain smooth 2D maps. Given different QPs, we will generate different 2D maps before compression. If the encoding configuration is not available before 2D map generation, using only the initial data volume also yields a clear gain over equal weighting factors, as shown in Section 5.3.3. It indicates that the proposed method is flexible and scalable.

## 4.2.2 Inter-MiniPLAS for P-frame Refinement

Theoretically, given I-frame maps as anchors, directly using the Inter-PLAS [22] can generate P-frame maps similar to the I-frame maps if a sufficient number of iterations are conducted. Using four selected positions $p _ { 1 } , \ldots , p _ { 4 }$ within a block as an example, let $\widetilde { \mathbf { g } } ^ { I }$ and $\widetilde { \mathbf { g } } ^ { P }$ denote the normalized, weighted feature vectors. The selected assignment is $\begin{array} { r } { \boldsymbol { \pi } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { \pi } \in S _ { 4 } } \sum _ { j = 1 } ^ { 4 } \| \widetilde { \mathbf { g } } ^ { P } ( p _ { \pi ( j ) } ) - \widetilde { \mathbf { g } } ^ { I } ( p _ { j } ) \| _ { 2 } ^ { 2 } , } \end{array}$ where $S _ { 4 }$ contains all 24 permutations. Considering a 2D map derived from a GS with more than one million primitives, carrying out a comprehensive exhaustive search of all permutations is extremely costly, which can explain why the Inter-PLAS algorithm has a substantially higher computational cost.

Therefore, for P-frame GS samples, we propose an Inter-MiniPLAS algorithm to generate P-frame 2D maps, as Fig. 3 illustrates. After using a two-stage Morton scan to initialize P-frame 2D maps, we propose progressive local permutations to remap them with I-frame MiniPLAS maps as anchors. The progressive block sizes are small, such as 32x32 and 4x4. The reason is that the two-stage Morton scan already uses the spatial distribution to initialize block-wise 2D maps [8], it is unnecessary to permute the pixels from a very large receptive field. We consequently restrict the permutation search to a small local region, which can accelerate convergence without disrupting the local block correlation. Besides, the downstream video codecs use square pixel blocks of certain sizes as coding units (CU). In the absence of explicit motion information, favorable compression performance can be achieved as long as the pixel blocks at corresponding positions are sufficiently similar.

Standard video codecs use a coding tree unit (CTU) as the largest basic processing unit, such as $6 4 \times 6 4$ in HEVC and 128×128 in VVC. We define the maximum permutation block size (MBS) as $B _ { M B S } \times B _ { M B S } , B _ { M B S } \in \{ \mathrm { C T U S i z e } \}$ Considering that the PLAS algorithm is a progressive process, the block size used in the i-th iteration for the P-frame is selected as

$$
B _ { i } = \frac { B _ { M B S } } { 2 ^ { i } } , i = 0 , 1 , . . . ,\tag{7}
$$

where $B _ { i }$ decreases to 4 as the minimum block size. For example, when $B _ { M B S } = 3 2$ , there are four block-size levels with $\mathit { \bar { B } } _ { i } = 3 2 , 1 6 , 8 , 4$ . Each level can contain multiple permutation sweeps.

For an $M \times M$ map, the permutation-enumeration cost of one sweep at block size $B _ { i }$ is $\begin{array} { r } { O ( \lfloor \frac { M } { B _ { i } } \rfloor ^ { 2 } \times ( \frac { B _ { i } ^ { 2 } } { 4 } \times 2 4 ) ) } \end{array}$ for a single feature channel, where ⌊·⌋ is the floor function. $\begin{array} { r } { { \lfloor \frac { M } { B _ { i } } \rfloor ^ { 2 } } } \end{array}$ is the number of blocks in the 2D map, and $\frac { B _ { i } ^ { 2 } } { 4 } \times 2 4$ is the number of permutations within each block, because PLAS uses 4 pixels as one permutation group (4! = 24 permutations for each unit). Therefore, assuming one complete sweep per block-size level, the overall computation of P-frame Inter-MiniPLAS for one channel of data is $\begin{array} { r } { O ( \Sigma _ { i = 0 } ^ { l } \lfloor \frac { M } { B _ { i } } \rfloor ^ { 2 } \times ( \frac { B _ { i } ^ { 2 } } { 4 } \times 2 4 ) ) , l \ = \ \log _ { 2 } B _ { M B S } - 2 , } \end{array}$ When $B _ { M B S } \ll \mathsf { \bar { M } } ,$ , the computational complexity can be simplified and estimated by $\mathbf { \dot { \Pi } } O ( M ^ { 2 } \log B _ { M \dot { B } S } )$ . If level i requires $s _ { i }$ sweeps over C feature channels, the permutation cost becomes $\begin{array} { r } { O ( C M ^ { 2 } \sum _ { i = 0 } ^ { l } s _ { i } ) } \end{array}$ . The measured runtimes are reported separately in Table 2.

![](images/74c14d03e47f3e4c38172a919c19ceb3ce145f51e2e5fddcc5b2a6527df57a7f.jpg)  
Fig. 3. Illustration of Inter-MiniPLAS. The I-frame map is held fixed while the initialized P-frame map is refined through local permutations of four primitives and progressively smaller blocks. The same permutation is applied to all attribute channels.

The block size setting of Inter-MiniPLAS not only reduces the computation time and complexity for the 2D map generation, but it can also be used as prior information for downstream video codec configuration. We will detail this part in the next section.

## 4.3 Content-Aware Generalized Compression (CA-GC)

We use the compression framework proposed in GSCV as the basis of CA-GC, and introduce two new techniques for efficient GS map compression. The first technique focuses on geometric coordinates, because coordinate data have a relatively high BD and are encoded in a lossless manner, which causes them to occupy a substantial portion of the total bitrate. In particular, in low-bitrate regimes, the bitrate contribution of the coordinate can exceed half of the overall bitrate, as discussed in section 5.2.3. Therefore, targeted optimization for coordinate compression can yield significant benefits. The second technique builds the connection between the MBS of Inter-MiniPLAS and the maximum CU size of the codec, which can significantly reduce the coding time without sacrificing obvious coding gains.

## 4.3.1 Differentiated Encoding Strategy for Geometric Coordinates

Although GSCV splits the 20-bit coordinate map into two 10 BD maps, which are compatible with standard video codecs, this operation fails to deliver effective compression results. The reason is that the LSB part is highly irregular, as shown in Fig. 4 upper right, leading to limited performance when using video codecs. However, for 10 BD LSB, we can still find some contours that satisfy the block-wise style, indicating that 10 BD LSB images exhibit some low-frequency information.

We use a non-uniform coordinate split to assign more bits to the MSB maps and leave fewer bits in the LSB maps. The bottom panels of Fig. 4 illustrate the less structured appearance of the 8-bit and 4-bit LSB maps. Experimental results show that standard video codecs can report better lossless compression performance for non-uniform coordinate maps than for uniform splits. Besides, for noise-like LSB maps, video codecs might not be necessary, and we can consider some more lightweight compression algorithms, such as LZMA [44]. Sometimes, we do not even need to conduct real compression for LSB due to very limited compression room. Section 5.3.6 compares LZMA with lossless HM18.0 and gives mode- and split-dependent recommendations.

![](images/bd96d9ac990e326cd3c0b5c22df98d70d7c15933860df7fe5bf77e0120233454.jpg)  
Fig. 4. Snapshot of the geometric coordinate and its LSB part.

## 4.3.2 Encoding Complexity Control via CU Size

In standard video codecs, given a predefined maximum CU size, the RD optimization (RDO) algorithm is conducted to determine the best CU partition and compression modes, which is an extremely time-consuming process due to the need to assess all strategies. The local refinement of Inter-MiniPLAS motivates using MBS as a prior for limiting the maximum CU size. Given $B _ { M B S }$ , we select the closest supported maximum CU size for compression. Note that using larger $B _ { M B S }$ and CU size cannot always provide better RD results than the smaller ones: excessively large B<sub>MBS</sub> can severely disrupt the smooth geometric coordinate maps generated by the two-stage Morton scan, as well as incur poorer overall bitrate distribution. In the experiments, we evaluate this design choice and quantify the reduction in encoding complexity.

## 5 EXPERIMENTS

In this section, we report the performance of the proposed GSCV++ on both datasets provided or derived by MPEG and AVS. Besides, we conduct a comprehensive ablation study to demonstrate the effectiveness of different modules.

## 5.1 Experimental Settings

## 5.1.1 Dataset Selection and Construction

For the MPEG datasets, we select three GS sequences, including “bartender”, “breakfast”, and “cinema”, with tracked and semi-tracked versions from the MPEG GSC working groups. Each sequence has 32 frames; the main experiments use the first eight frames as one GoP to test RD performance (Larger GoP sizes are reported in Section

![](images/12f4bb8d371a1a7a88eb1e453b329e1d22b42a7906e6caaebdadda9b888ee476.jpg)  
Fig. 5. Snapshots of the datasets.

5.3.8). “Tracked” means all the GS frames are refined from a shared canonical GS model, which was trained using the merged point cloud with ground truth images from all viewpoints and time instants. Densification is disabled during the refinement process, ensuring that all the frames have the same number of primitives. “semi-tracked” means densification is allowed during the finetuning process, which leads to a different number of primitives for different frames. For AVS datasets, we select two dynamic scenes that exhibit clear differences from those provided by MPEG, i.e., “dance dunhuang pair”(dance) from PKU-DyMVHumans [45] and “basketball” from AVS-VRU [46]. We follow the instructions of the MPEG document [47] to generate GS sequences. Snapshots of the datasets are shown in Fig. 5.

The target condition is unavailable inter-frame primitive tracked information. To construct this condition from tracked sequences, we shuffle each frame independently before compression; the label tracked below describes the source data, not the preserved input indices. For the semitracked sequences, LightGaussian pruning [31] produces the equal primitive counts required by the evaluated fixedresolution map pipeline.

## 5.1.2 Comparison Method

We compare the proposed GSCV++ against two anchors adopted in the MPEG GSC working group, GSCodec Studio [19] and GPCC v1 [40], as well as the SOTA video-based GS sequence compression method GSCV, which also addresses the problem of unavailable inter-frame correspondence. GPCC v1 does not support GS inter-prediction, so we use all-intra mode for all the testing conditions. GSCodec Studio [19] pipeline does not support the no-tracked cases, which is also one of the motivations for this paper. We therefore exclude GSCodec Studio and GPCC v1 from RD gains and do not use their results to rank compression performance. Following [48], quality is evaluated at views

![](images/46e155303081b983ff4eead32a4dae0adc2707665210d443e6e58462b7cb2195.jpg)  
Fig. 6. Intra-prediction performance comparison on the MPEG and AVS dataset.

9 and 11 for “bartender” and “cinema”, and views 6 and 8 for “breakfast”. The AVS sequences use views 1, 11, 21, 31, 41, and 51 for “dance”, and views 1, 11, 21, and 31 for “basketball”.

## 5.1.3 Model Parameters

For the EA-SHDR module, we use an energy threshold $A \ : = \ : 0 . 9 9$ to implement color SH AC dimensionality reduction. To avoid padding the latent feature z<sub>AC</sub> when generating three-channel images, we use $k = 3 \lceil k ^ { * } / 3 \rceil$ . For example, $\bar { k } ^ { * } ~ = ~ 2 0$ in Eq. (1) gives $k ~ = ~ 2 1$ . For the P-MiniPLAS module, when calculating the PLAS weight $W _ { a }$ for coordinate lossless compression, we set $Q P = Q { \bar { P } } _ { D C }$ to calculate $f ^ { 2 } ( Q P ^ { a } )$ , then scale this value up by a factor of 1.25. The reason is that color SH DC usually uses a small QP. We expect a relatively larger weighting factor for the coordinate, considering that lossless compression generally leads to a larger bitstream. Other coordinate-weight settings are discussed in Section 5.3.3. We set the block size of I-frame MiniPLAS to [8,8] as suggested by [8], while the MBS for the P-frame Inter-MiniPLAS is set to 64. For the CA-GC module, as with GSCV, 20 BD and 10 BD are used to quantize the coordinate and other attributes. We use 16:04 to split the coordinate map. HM18.0 [27] is used as a representation of standard video codecs in the experiments; additional selections of other standard video codecs are reported in section 5.3.7. When using HM18.0, we set the maximum CU size to 64.

## 5.2 Experimental Results

In this section, we report the performance of GSCV++ on both intra- and inter-prediction modes. For intra-prediction, we illustrate the RD curve of the first frame of each sequence as a representative example; for inter-prediction, we use the averaged frame-wise RD curve results within one GoP.

## 5.2.1 Intra-Prediction Results

Fig. 6 illustrates the RD curves of the proposed GSCV++, GSCV, GSCodec Studio, and GPCC v1. We see that: 1)

GSCV++ reports significant gains compared to all the other methods; 2) Considering that GSCV++, GSCV, and GSCodec Studio are all video-based methods, the performance of GSCV++ is the best, followed by GSCV and GSCodec Studio; 3) GPCC v1 reports better performance in the lowbitstream range on the MPEG dataset because GPCC v1 is more effective at coordinate compression, and the coordinates occupy most of the bitstream in the low-bitstream region; 4) However, GPCC v1 reports different performance on the MPEG and AVS datasets: GPCC v1 performs better on the MPEG dataset but worse on the AVS dataset. After checking the dataset, we find that the primitive density of the AVS dataset is sparser than that of the MPEG sequences. Considering that GPCC v1 is initially designed for point clouds and is more powerful for compressing dense point clouds, sparse GS samples present a relatively challenging data distribution for GPCC v1; 5) Table 1 left part quantifies the averaged RD gains of GSCV++ on both the MPEG dataset and the AVS dataset. Specifically, GSCV++ exhibits gains of around 25% to 40% compared with GSCV.

TABLE 1  
RD gain of GSCV++ over GSCV
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Anchor: GSCV</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Intra</td><td rowspan=1 colspan=2>Inter</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>MPEG</td><td rowspan=1 colspan=1>AVS</td><td rowspan=1 colspan=1>MPEG</td><td rowspan=1 colspan=1>AVS</td></tr><tr><td rowspan=1 colspan=1>GSCV++</td><td rowspan=1 colspan=1>-39.61%</td><td rowspan=1 colspan=1>-24.50%</td><td rowspan=1 colspan=1>-68.70%</td><td rowspan=1 colspan=1>-109.93%</td></tr></table>

## 5.2.2 Inter-Prediction Results

Fig. 7 illustrates the RD curves of the proposed GSCV++, GSCV, GSCodec Studio, and GPCC v1 under the interprediction condition. We see that: 1) compared with the allintra mode, GSCV++ demonstrates more pronounced gains under the inter-frame prediction mode; 2) different from the intra-prediction mode, where GSCodec Studio reports performance similar to GPCC v1 in most sequences; however, GSCodec Studio illustrates extremely poor performance under the inter-prediction mode. The reason is that GSCodec

![](images/210b227f90b5d7a4ee2c4fd5f7688f11b16424c84dd3f1800102ec2bc5ef53dd.jpg)  
Fig. 7. Inter-prediction performance comparison on the MPEG and AVS datasets.

Studio uses the I-frame PLAS sorting index to generate Pframe 2D maps, which assumes that I- and P-frames are associated with stable tracking information. Without tracked information, the P-frame 2D maps of GSCodec Studio show significant content variations. It also reveals the importance of the proposed P-MiniPLAS module, which can construct robust tracked information for arbitrary GS sequences; 3) GSCV reports stable performance due to the introduction of inter-PLAS. However, GSCV++ includes more effective 2D map generation methods, resulting in even higher RD gains than those of the intra-prediction mode, as illustrated in Table 1 right part: the gains of GSCV++ compared to GSCV increased from around 25% - 40% to around 65% - 110%.

## 5.2.3 Rate Allocation

The rate allocation of “bartender” is shown in Fig. 8. We use the three rate points and detail the bits per primitive (BPP) for all the components. We observe that for the highbitrate point, e.g., R05, the color SH AC occupies the largest part of the bitstream, while for the low-bitrate point, the coordinates account for nearly half of the total bitstream, e.g., over 30 BPP for coordinates. This reveals why videobased GS compression methods are worse than GPCC v1 for the low-bitrate region: GPCC v1 is more powerful for coordinate compression, enabling it to achieve around 10-15 coordinate BPP.

## 5.2.4 Time Complexity

In video-based GS compression frameworks, the computational cost mainly consists of two parts: 1) T1: the time spent converting vanilla GS into 2D video; and 2) T2: the subsequent video encoding and decoding time. First, Table 2 illustrates the T1 time of GSCV++ and GSCV on “bartender”. Specifically, for GSCV++, there are three steps to obtain the final 2D GS videos: EA-SHDR, two-stage Morton scan, and P-MiniPLAS. For GSCV, there are two steps: spatial-stable initialization (SSI) and anchor-based PLAS refinement (A-PLASre). We see that for both I- and P-frames, GSCV++ can exhibit a faster 2D map generation speed, which can benefit the practical application of 3D GS.

![](images/928bfe48e12a180156fff10ed5daaa515841edc649b008c133934117177171fc.jpg)  
Fig. 8. Rate allocation of semi-tracked “bartender”.

TABLE 2  
Computation time (s) of preprocessing and 2D map generation for GSCV++ and GSCV
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GSCV</td><td rowspan=1 colspan=1>++</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>GSCV</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>EA-SHDR</td><td rowspan=1 colspan=1>Two-stageMorton Scan</td><td rowspan=1 colspan=1>P-MiniPLAS(MBS = 64)</td><td rowspan=1 colspan=1>All</td><td rowspan=1 colspan=1>SSI</td><td rowspan=1 colspan=1>Inter-PLAS</td><td rowspan=1 colspan=1>All</td></tr><tr><td rowspan=1 colspan=1>I-frame</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>3.96</td><td rowspan=1 colspan=1>4.575</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>24.71</td><td rowspan=1 colspan=1>24.78</td></tr><tr><td rowspan=1 colspan=1>P-frame</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.075</td><td rowspan=1 colspan=1>14.54</td><td rowspan=1 colspan=1>14.665</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>74.51</td><td rowspan=1 colspan=1>74.6</td></tr></table>

Second, the T2 time is related to the downstream codecs used in practical compression. Therefore, we list the encoding time (En) and decoding time (De) of GSCV++, GSCV, GSCodec Studio (GSCS), and GPCC v1 in Table 3. For GSCV++ and GSCV, we test two different video codecs, HM18.0 and FFMPEG libx265. We see that: 1) GSCV++ libx265 reports the fastest encoding and decoding times; 2) using the same video codecs, GSCV++ requires less encoding and decoding time than GSCV and GSCodec Studio. It indicates that the generated GS maps are more userfriendly for standard video codecs by introducing smoother textures; 3) using GSCV++ libx265 R05 as an example, the average encoding and decoding times per image are around 2.39/14 = 0.171s and 0.50/14 = 0.035s. This means that currently, GSCV++ can achieve almost real-time decoding when employing multi-threaded parallel decoding (around 30 FPS), considering that the decoding process of different

TABLE 3  
The averaged frame-wise encoding and decoding times of different methods on the tracked “bartender”.
<table><tr><td>Codec</td><td colspan="2">GSCV++ HM18.0</td><td colspan="2">GSCV++ libx265</td><td colspan="2">GSCV HM18.0</td><td colspan="2">GSCV libx265</td><td colspan="2">GSCS libx265</td><td colspan="2">GPCC v1</td></tr><tr><td>Rate Point</td><td>En</td><td>De</td><td>En</td><td>De</td><td>En</td><td>De</td><td>En</td><td>De</td><td>En</td><td>De</td><td>En</td><td>De</td></tr><tr><td>R05</td><td>397.8</td><td>1.4</td><td>2.4</td><td>0.5</td><td>1183.5</td><td>3.9</td><td>4.6</td><td>1.2</td><td>8.9</td><td>1.4</td><td>304.1</td><td>86.0</td></tr><tr><td>R04</td><td>332.1</td><td>1.2</td><td>2.4</td><td>0.4</td><td>1012.5</td><td>2.9</td><td>4.1</td><td>0.9</td><td>8.5</td><td>1.4</td><td>298.0</td><td>88.4</td></tr><tr><td>R03</td><td>246.6</td><td>1.0</td><td>2.2</td><td>0.4</td><td>540.3</td><td>2.1</td><td>2.8</td><td>0.7</td><td>7.3</td><td>1.3</td><td>291.7</td><td>86.7</td></tr><tr><td>R02</td><td>213.1</td><td>0.9</td><td>2.2</td><td>0.4</td><td>353.4</td><td>1.6</td><td>2.3</td><td>0.6</td><td>6.4</td><td>1.3</td><td>291.9</td><td>86.0</td></tr><tr><td>R01</td><td>187.3</td><td>0.8</td><td>2.1</td><td>0.3</td><td>301.5</td><td>1.3</td><td>1.9</td><td>0.5</td><td>5.2</td><td>1.1</td><td>293.2</td><td>85.4</td></tr></table>

GS attributes is independent. Given that GS cannot currently be generated in real time, real-time encoding is not a strict industrial requirement. Even so, reducing encoding latency remains beneficial for facilitating broader GS deployment. One promising method is to control the number of primitives before compression, such as introducing pruning during GS generation [21] or adding it as part of the preprocessing step.

## 5.3 Ablation Study

In this section, we report the ablation study of GSCV++. Specifically, we use three MPEG semi-tracked sequences as test sequences. If the ablation experiments exhibit consistent performance across the three sequences, we report the average results over all three. Otherwise, when the performance is inconsistent, we select the sequences with divergent outcomes for further analysis and discussion.

## 5.3.1 Effectiveness ofEA-SHDR and P-MiniPLAS

To quantify the contribution of the two modules, we evaluate four combinations of EA-SHDR and P-MiniPLAS, Table 4 reports the arithmetic mean of the three per-sequence rate changes, using the setting with both modules disabled as the common reference. Enabling EA-SHDR alone reports a 21.92% RD gain, while enabling P-frame refinement alone illustrates 32.43% gain. Combining both modules yields the largest average gain, 41.79%, and also provides the largest reduction relative to the common baseline on each sequence. Using the EA-SHDR-only setting as the reference, adding P-frame refinement provides a further average gain of 24.65%. Conversely, adding EA-SHDR to the refinementonly setting yields a gain of 10.67%. These results support the complementary roles of the two modules in the main configuration.

TABLE 4  
Average RD changes of module combinations. The last row is the main GSCV++ configuration.
<table><tr><td>EA-SHDR</td><td>P-MiniPLAS</td><td>Avg. Gain (%)</td></tr><tr><td>w/o</td><td>w/o</td><td>0.00</td></tr><tr><td>W</td><td>w/o</td><td>-21.92</td></tr><tr><td>w/o</td><td>W</td><td>-32.43</td></tr><tr><td>W</td><td>W</td><td>-41.79</td></tr></table>

To better illustrate the influence of different modules on different bitrate ranges, we additionally present EA-SHDR and P-MiniPLAS in RD curves using three alternatives: 1) disable both modules (w/o PLAS and PCA); 2) disable EA-SHDR (w/o PCA); and 3) use separate PCA bases for Iand P-frames (SePCA). We show the results of “bartender” and “cinema” in Fig. 9. We see that both EA-SHDR and P-MiniPLAS contribute to the overall performance. Introducing PCA-related operations will incur a lower-quality upper bound; however, for “bartender”, the performance of SePCA and w/o PCA is close, while the performance without PCA is superior to SePCA for “cinema”. This indicates that the principal component space of “cinema” color SH AC exhibits significant variation across different frames. Using I-frame truncated SVD results for P-frames is important to ensure stable prediction gain.

![](images/770689c8e0b1784abc7397a1d80707625d19bd299aee372be35e1b513a711001.jpg)

![](images/b56b640d5752cb5e2d5df0d15f301ff6d3507d2f23f344afe8bb8bcdd704e1f8.jpg)  
Fig. 9. Ablation Study of EA-SHDR and P-MiniPLAS on the MPEG semitracked dataset.

We further compare the complete Inter-PLAS [22] with our P-MiniPLAS on the same sequences, retaining EA-SHDR, shared I/P PCA bases, coordinate coding, HM settings, and the five operating points. The Inter-PLAS alternative applies vanilla PLAS to the I-frame and globally refines each P-frame using its own I-frame ordering and quantized anchor, with unit attribute weights, radius decay 0.95, and minimum sorting block size 16. The results are shown in Table 5. The proposed P-MiniPLAS reports an obvious gain of 24.49% on average, with gains of 24.11%, 28.87%, and 20.50% on “bartender”, “breakfast”, and “cinema”, respectively. Mean 2D map generation time decreases from 25.61 to 2.10 s for the I-frame and from 63.04 to 16.43 s per Pframe, corresponding to a 3.84× P-frame sorting speedup. These results show consistent RD improvements and lower sorting cost across the three sequences.

TABLE 5  
Average RD change and computation time of complete I/P map generation between Inter-PLAS and P-MiniPLAS.
<table><tr><td>Method</td><td>∆R (%)</td><td>I sorting (s)</td><td>P sorting (s/frame)</td></tr><tr><td>Inter-PLAS</td><td>0.00</td><td>25.61</td><td>63.04</td></tr><tr><td>P-MiniPLAS</td><td>-24.49</td><td>2.10</td><td>16.43</td></tr></table>

## 5.3.2 Energy Ratio for EA-SHDR

In Section 5.1.3, we set the energy ratio as A = 0.99. In this part, we show the results of A = 0.999, 0.950, 0.900, 0.980 in Fig. 10. We see that the “bartender” and “breakfast” report different RD curves corresponding to a monotonic variation of energy, while “cinema” is close to “breakfast”. For “bartender”, it shows perfect hierarchical performance with the increase of energy, while “breakfast” reports the best RD results when A = 0.900 and an obvious performance gap with A = 0.850. The reason is that the texture of “breakfast” is more complex than that of “bartender”, and the energy distribution is different: the latent feature channels $k ^ { * }$ for “bartender” and “breakfast” are [27, 21, 15, 12, 9] and [24, 18, 12, 9, 6], respectively. The energy of “breakfast” is more concentrated. The best RD curve corresponds to finding the best solution of $J = { \cal D } + \lambda R ,$ and “bartender” has different global optima from “breakfast”. Further work is needed to explore an RD analysis for the GS 2D map corresponding to the color SH AC energy.

![](images/8afac4fb4cf5e9d25a5c9c426da56f737a8f1a81814d1bb5fdb372e7d8964727.jpg)

![](images/89bfa08abd2e8f9a30186e10b125c425ed073ae97f1072c5bbe3cacfe2e9e575.jpg)  
Fig. 10. Ablation Study of EA-SHDR energy ratio on the MPEG semitracked dataset.

## 5.3.3 Influence ofPLAS Weighting Factor

In Section 4.2.1, we introduce a prior-based PLAS weighting factor to realize content-aware PLAS image smoothness. To highlight the effectiveness of this method, we report the performance of GSCV++ corresponding to: 1) equal weighting factors for all the GS attributes; 2) attribute channelbased weighting factors, i.e., coordinate: color SH DC: color SH AC: opacity: scale: rotation $= 3 { : } 3 { : } 4 5 { : } 1 { : } 3 { : } 4 ;$ and 3) when calculating the weight of coordinate, we fix $Q P = 0$ for $f ^ { 2 } ( Q P ^ { a } )$ (noted as proposed (coordinate $\mathrm { Q P } = 0 ) )$ , which means that regardless of the QP range of other attributes, the weighting factors of coordinate are fixed. The results are shown in Fig. 11 (a). We observe that the performance of the proposed and proposed (coordinate $\bar { \mathrm { Q P } _ { } ^ { - } } = 0 )$ are the best, followed by channel-based and equal weight. Using equal weight as a baseline, the RD gains of the channel-based, proposed, and proposed (coordinate $\mathrm { Q P } = 0 )$ methods are 6.2%, 7.5%, and 8.3%, respectively. Using coordinate QP as 0 is slightly better than setting coordinate $Q P \ : = \ : Q P _ { D C } .$ It is worth noting that the meaning of QP=0 differs across different encoders: nearly lossless for HM/VTM, while real lossless for FFMPEG libx264. Considering that the proposed method requires some prior knowledge, such as the QP of the downstream video codecs, we suggest using channelbased weights when QP prior knowledge is not available.

## 5.3.4 Influence ofCU Size

In Section 4.2.2, we introduce the setting of the P-frame MBS, which can be used as a prior for CU size selection in the downstream video codec. In Section 5.1.3, we set MBS to 64 and adopt a progressive block size of [64x64, 32x32, 16x16, 8x8, 4x4] for Inter-MiniPLAS. In this Section, we report the performance of using maximum CU size and partition depth (PD) as (64, 4), (32, 3), and (16,2) for real coding, ensuring that the minimum CU size is 8x8. The results are shown in Fig. 11 (b). We see that the performance of different CU and PD settings yields similar RD performance: using CU16 PD2 as a benchmark, the performance gains of CU32 PD3 and CU64 PD4 are 0.33% and 0.41%, respectively.

![](images/4db7b57d29f04d9a6266966ababb91584f8ce8cbe9b82fd22492f4ad4e689640.jpg)  
(a) PLAS Weighting Factors

![](images/f44ebc20f5b09ddc692e734ef9165666159c31461c0e74057748cefdecde342d.jpg)  
(b) CU Size  
Fig. 11. (a): Ablation study of PLAS weight; (b): Ablation study of CU size.

One advantage of using a smaller CU size and PD is low encoding complexity. According to the RDO algorithm, the encoder traverses all possible partitioning schemes and computes the reconstruction error and bitrate to determine the optimal choice. Smaller CU involves fewer partitioning schemes, allowing the optimal coding decision to be reached more efficiently. Using the “cinema” interprediction mode with the highest bitrate as an example, the averaged HM18.0 encoding times of each 2D GS map of CU16 PD2, CU32 PD3, and CU64 PD4 are 16.39, 18.85, and 22.45. For low-delay applications, P-MiniPLAS can provide an important prior for the trade-off between RD and encoding time.

## 5.3.5 Influence ofMBS

We illustrate the performance of GSCV++ with different MBS in this part. First, we test the performance of different MBS, including MBS = 128, 64, 32, 16, and 8. Given the MBS, we configure the encoder by selecting the maximum CU size that is closest to the MBS. For example, MBS = 128 with CU size = 64, MBS = 32 with CU size = 32, and MBS = 8 with CU size = 16. The results are shown in Fig. 12 (a). We observe that: 1) the overall performance increases from $\mathrm { M B S } = 8$ to 64, but decreases when $\mathrm { M B S } = 1 2 8 .$ . This result indicates that larger MBS does not necessarily lead to better performance. Considering that P-MiniPLAS is a progressive process, an excessively large MBS can lead to local optima and prevent subsequent smaller block size values from escaping these local optima. 2) a larger MBS requires more computation time. Using “bartender” as an example, it has around half a million primitives, and the P-MiniPLAS times are 17.27s, 13.7s, 6.24s, 5.76s, 1.40s from MBS = 128 to 8. With a reasonable MBS value range, the compression ratio generally tends to increase proportionally with the P-MiniPLAS computation time.

Section 4.2.2 motivates limiting the maximum CU size using MBS as an engineering prior. To evaluate this design choice, we illustrate additional experiments with MBS = 8 and three different maximum CU sizes that are larger than 8 in Fig. 12 (b). The three curves are observed to nearly overlap, which supports this observation on the tested sequences.

![](images/9093a93243ab61ab0aef2f2213908f7375f61d13f706df66b139e65c9487a0b8.jpg)

![](images/6fb2a160f0cbaaf72681db2eda8b46a245df9a9053a8cd65e265a32428f26292.jpg)  
(a) P-MiniPLAS with different MBS(b) Small MBS with larger CU size. and CU combination.  
Fig. 12. (a): Ablation study of P-MiniPLAS MBS on the MPEG semi tracked dataset; (b): Ablation study of CU size on small MBS setting.

TABLE 6  
Coordinate compression of MPEG semi-tracked sequences.
<table><tr><td rowspan="2">Sequence</td><td rowspan="2" colspan="2">Setting</td><td colspan="3">MSB:HM18.0 LSB:HM18.0</td><td colspan="3">MSB:HM18.0 LSB:LZMA</td></tr><tr><td>MSB</td><td>LSB</td><td>ALL</td><td>MSB</td><td>LSB</td><td>ALL</td></tr><tr><td rowspan="6">bartender</td><td rowspan="3">All-intra</td><td>10:10</td><td>1.99</td><td>28.06</td><td>30.05</td><td>1.99</td><td>30.88</td><td>32.86</td></tr><tr><td>12:08</td><td>4.93</td><td>25.04</td><td>29.96</td><td>4.93</td><td>24.00</td><td>28.92</td></tr><tr><td>16:04</td><td>15.64</td><td>13.62</td><td>29.25</td><td>15.64</td><td>12.46</td><td>28.1</td></tr><tr><td rowspan="3">Inter</td><td>10:10</td><td>2.86</td><td>27.56</td><td>30.42</td><td>2.86</td><td>28.36</td><td>31.22</td></tr><tr><td>12:08</td><td>5.72</td><td>24.73</td><td>30.45</td><td>5.72</td><td>25.43</td><td>31.15</td></tr><tr><td>16:04</td><td>16.41</td><td>13.62</td><td>30.03</td><td>16.41</td><td>14.32</td><td>30.73</td></tr><tr><td rowspan="6">cinema</td><td rowspan="3">All-intra</td><td>10:10</td><td>2.27</td><td>28.05</td><td>30.32</td><td>2.27</td><td>30.97</td><td>33.24</td></tr><tr><td>12:08</td><td>5.36</td><td>24.87</td><td>30.23</td><td>5.36</td><td>24.00</td><td>29.36</td></tr><tr><td>16:04</td><td>16.07</td><td>13.62</td><td>29.69</td><td>16.07</td><td>12.46</td><td>28.53</td></tr><tr><td rowspan="3">Inter</td><td>10:10</td><td>3.14</td><td>26.93</td><td>30.08</td><td>3.14</td><td>28.14</td><td>31.29</td></tr><tr><td>12:08</td><td>6.02</td><td>23.92</td><td>29.93</td><td>6.02</td><td>25.03</td><td>31.05</td></tr><tr><td>16:04</td><td>15.88</td><td>13.62</td><td>29.5</td><td>15.88</td><td>14.16</td><td>30.05</td></tr><tr><td rowspan="6">breakfast</td><td rowspan="3">All-intra</td><td>10:10</td><td>5.44</td><td>30.77</td><td>36.22</td><td>5.44</td><td>31.85</td><td>37.29</td></tr><tr><td>12:08</td><td>10.22</td><td>25.52</td><td>35.74</td><td>10.22</td><td>24.00</td><td>34.22</td></tr><tr><td>16:04</td><td>21.99</td><td>13.62</td><td>35.61</td><td>21.99</td><td>12.48</td><td>34.47</td></tr><tr><td rowspan="3">Inter</td><td>10:10</td><td>4.82</td><td>27.91</td><td>32.72</td><td>4.82</td><td>28.28</td><td>33.10</td></tr><tr><td>12:08</td><td>7.93</td><td>24.44</td><td>32.38</td><td>7.93</td><td>25.18</td><td>33.11</td></tr><tr><td>16:04</td><td>18.9</td><td>13.62</td><td>32.52</td><td>18.9</td><td>14.29</td><td>33.20</td></tr></table>

## 5.3.6 Influence ofDifferentiated Encoding on Coordinate

Section 4.3.1 introduces non-uniform coordinate splitting. Table 6 compares lossless HM18.0 and LZMA for LSB coding on three semi-tracked sequences, using 10:10, 12:08, and 16:04 splits in all-intra and inter modes. The MSB uses HM18.0. LZMA receives direct LSB data for I-frames and I-frame-referenced residuals for subsequent frames. The highlighted entries indicate the smallest total coordinate size among the tested configurations for each sequence and mode.

The results support a mode- and split-dependent LSB choice. For all-intra coding, LZMA gives a smaller LSB size than HM18.0 at 12:08 and 16:04 on all three sequences; the 10:10 split instead favors HM18.0. For inter coding, HM18.0 gives a smaller LSB size for all nine tested sequence/split combinations. We therefore recommend LZMA for the evaluated all-intra configurations and HM18.0 for the evaluated inter configurations.

## 5.3.7 Performance ofDifferent Standard Codecs

In Section 5.1.3, we use HM18.0 as the representation of standard video codecs for GSCV++. In Fig. 13, we report the performance of using VTM23.11 [28], FFMPEG libx265, and FFMPEG libx264 [29] as practical codecs. Specifically, the libx265 and libx264 configurations evaluated here support maximum input BDs of 12 and 10, respectively. Therefore, we use 12:08 for coordinate compression for FFMPEG libx265, while 10:10 for FFMPEG libx264. We see that: 1) the performance of HM18.0 and VTM23.11 is close, followed by FFMPEG libx265 and libx264. Using FFMPEG libx264 as the benchmark, the RD gains of libx265, VTM23.11, and HM18.0 are 12.01%, 23.27%, and 23.91%; 2) VTM23.11 does not report better performance than HM18.0, which conflicts with the conclusion on natural videos. The reason is that compared with natural images, GS 2D maps illustrate a more pronounced heavy-tail behavior, and these maps do not provide any explicit motion like natural videos. Some advanced video coding modules involved in VVC do not illustrate significant gains over HEVC.

![](images/93ebb20da5f6d51d2b8c15d04a93c6e30c9de518b35d035d13d5c611b5fa2a7b.jpg)  
Fig. 13. Performance of GSCV++ with different video codecs.

![](images/ef4044a95073b87c64fc2af235f41587a99a604ec14be8177f6678bc1fb870d9.jpg)  
(a) Residual histogram of rendered-view images.

![](images/c61391c98eb0d1034aa5cc12107727a68860935b36fc5d98b8e18ccac3b6d16d.jpg)  
(b) Residual histogram of P-MiniPLAS scale map.  
Fig. 14. Residual Analysis.

We use the “cinema” rendered view 9th images from frames 0 and 1 as the representations of natural images. The residual histogram for the inter-prediction is demonstrated in Fig. 14(a), and the same residual analysis of P-MiniPLAS scale maps is used for comparison and illustrated in Fig. 14(b). Based on the histogram, we also provide fitted Gaussian and Laplace curves. The larger Laplace scale parameter b of the natural image is 0.012, while that of the scale map is 0.021. This reflects the conclusion that P-MiniPLAS images demonstrate an obvious heavy-tail behavior. Additionally, the entropy of the natural and scale residual images is 4.26 and 6.44, respectively. This evidence reveals that the P-MiniPLAS image presents a more challenging condition for current standard video codecs.

## 5.3.8 Performance ofDifferent GoP Size

To evaluate the effect of GoP size beyond the eight-frame setting, we compress the same 32 consecutive frames of the MPEG semi-tracked “cinema” sequence using GoP sizes of 8, 16, and 32. These settings partition the sequence into four, two, and one independently coded groups, respectively, each containing one I-frame followed by P-frames that reference that group’s I-frame. We retain the main parameters and five operating points. Each new I-frame refreshes its 2D mapping and EA-SHDR basis.

TABLE 7  
GoP-size comparison on the 32-frame “cinema” sequence. RD changes use all five operating points with GoP8 as the reference. Temporal diagnostics are measured at R03.
<table><tr><td>Metric</td><td>GoP8</td><td>GoP16</td><td>GoP32</td></tr><tr><td>RD gain (%)</td><td>0.00</td><td>+3.91</td><td>+28.04</td></tr><tr><td>PSNR, frames 0–15 (dB)</td><td>34.70</td><td>34.35</td><td>34.37</td></tr><tr><td>PSNR, frames 16–31 (dB)</td><td>35.04</td><td>34.87</td><td>32.66</td></tr><tr><td>PCA dim., frames 16–31</td><td>24</td><td>24</td><td>21</td></tr><tr><td>Proj. error, frames 16–31 (%)</td><td>0.25</td><td>0.25</td><td>1.02</td></tr></table>

Fig. 15(a) and Table 7 show a small RD degradation for GoP16 and a larger degradation for GoP32, with loss of 3.91% and 28.04% relative to GoP8. At R03, GoP32 increases size from 3.557 to 4.362 MB/frame while reducing mean PSNR from 34.87 to 33.51 dB relative to GoP8. The quality gap is concentrated in the second half of the sequence. In Fig. 15(b), GoP16 and GoP32 have nearly identical average PSNR over frames 0–15 (34.35 and 34.37 dB), but GoP32 is 2.21 dB lower over frames 16–31. Refreshing the I-frame at frame016 in GoP8/16 also updates the PCA latent dimension from 21 to 24, whereas GoP32 keeps the 21-dimensional frame000 representation. We also report the projection error of EA-SHDR in Table $7 , \mathrm { i . e . , } \| X _ { c } - \widehat { X _ { c } } \| _ { F } ^ { 2 } / \| X _ { c } \| _ { F } ^ { 2 }$ , where $X _ { c }$ is centered using the group’s I-frame mean. Consequently, the second-half normalized projection error is 1.02% for GoP32 and approximately 0.25% for GoP8/16. The 0.99 energy threshold selects the dimension only at the I-frame and does not guarantee the same retained energy for subsequent Pframes. This difference includes both basis refresh and dimension adaptation. These observations are consistent with poorer matching to the distant I-frame anchor and reduced compressibility that local P-MiniPLAS refinement cannot fully recover. Therefore, periodic I-frame refresh maintains more stable RD performance and supports the default GoP8 setting.

![](images/0ed17099364011beddf02c370585df0abd4f74fd0b68cc2a4e18e0098b5a33a1.jpg)  
(a) RD performance on the same 32 frames.

![](images/c2f52cf31e55e79cf5f7aae40e70e3454526abcabd10f33bc02719f21db3a7af.jpg)  
(b) Frame-wise PSNR at R03.  
Fig. 15. (a): Performance of different GS GoP sizes on the 32- frame semi-tracked “cinema” sequence; (b): frame-wise PSNR at R03, squares mark I-frames, the dashed line separates frames 0–15 and 16– 31.

## 6 CONCLUSION

This paper presents a comprehensive and effective 3D GS sequence compression framework via the video codec GSCV++. GSCV++ optimizes the end-to-end video-based 3D GS compression pipeline with EA-SHDR, P-MiniPLAS, and CA-GC. It reports SOTA performance in both intraand inter-prediction modes, as well as faster GS 2D map generation and coding speeds. Based on the experimental results, we demonstrate the robustness, compatibility, and alignment with the requirements of standardization for lightweight 3D GS compression.

Limitations: The GS geometry coordinate LSB map compression is less compressible for standard video codecs. Additionally, using energy solely to control the latent dimension of color SH AC cannot achieve an optimized RD balance for some GS sequences. Moreover, PLAS images report different characteristics compared with natural images; even the slightest roughness will limit the performance of the encoder. Some learning-based methods, such as Sandwich Network [49], might further improve the performance of standardized video codecs.

## REFERENCES

[1] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, “3d gaus- ¨ sian splatting for real-time radiance field rendering.” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021.

[3] WG7, “[gsc][jee6.4] draft use cases and requirements for gaussian splat coding,” ISO/IEC JTC1/SC29 WG7 Document m71640, 2025.

[4] Y. Chen, Q. Wu, W. Lin, M. Harandi, and J. Cai, “Hac: Hashgrid assisted context for 3d gaussian splatting compression,” in European Conference on Computer Vision. Springer, 2024, pp. 422– 438.

[5] ——, “Hac++: Towards 100x compression of 3d gaussian splatting,” arXiv preprint arXiv:2501.12255, 2025.

[6] X. Liu, X. Wu, P. Zhang, S. Wang, Z. Li, and S. Kwong, “Compgs: Efficient 3d scene representation via compressed gaussian splatting,” in Proc. ACM Int. Conf. Multimedia, 2024, pp. 2936–2944.

[7] Q. Yang, K. Yang, Y. Xing, Y. Xu, and Z. Li, “A benchmark for Gaussian splatting compression and quality assessment study,” in Proc. ACM Int. Conf. Multimedia in Asia. Association for Computing Machinery, 2024.

[8] Q. Yang, G. Van Der Auwera, and Z. Li, “Lightweight 3d gaussian splatting compression via video codec,” in Data Compression Conference, 2026, pp. 362–371.

[9] H. Huang, W. Huang, Q. Yang, Y. Xu, and Z. Li, “A hierarchical compression technique for 3d gaussian splatting compression,” in IEEE Int. Conf. Acoustics, Speech and Signal Processing, 2025, pp. 1–5.

[10] Y. Xing, W. Gordon, Q. Yang, K. Yang, J. Wang, and Y. Xu, “3dgsvbench: A comprehensive video quality evaluation benchmark for 3dgs compression,” in Int. Conf. Visual Communications and Image Processing, 2025, pp. 1–5.

[11] WG4, “[gsc][jee6.4-related] on the use case and requirements for lightweight gsc,” ISO/IEC JTC1/SC29 WG4 Document m72430, 2025.

[12] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (HEVC) standard,” IEEE Trans. Circuits and Systems for Video Technology, vol. 22, no. 12, pp. 1649– 1668, 2012.

[13] S. Schwarz, M. Preda, V. Baroncini, M. Budagavi, P. Cesar, P. A. Chou, R. A. Cohen, M. Krivokuca, S. Lasserre, Z. Li, J. Llach,´ K. Mammou, R. Mekuria, O. Nakagami, E. Siahaan, A. Tabatabai, A. M. Tourapis, and V. Zakharchenko, “Emerging mpeg standards for point cloud compression,” IEEE Journal on Emerging and Selected Topics in Circuits and Systems, vol. 9, no. 1, pp. 133–148, 2019.

[14] WG7, “V-pcc gs performance evaluation and anchor results,” ISO/IEC JTC1/SC29 WG7 Document N1496, 2026.

[15] J.-Y. Jeong, K.-J. Oh, and G. Lee, “[gsc][jee6.7-related] importancescore based gaussian pruning,” ISO/IEC JTC1/SC29 WG4 Document m74048, 2025.

[16] Y. Xu, K. Yang, S. Xia, Y. Hu, and X. Ma, “[gsc][jee6.7-related] a fast view-independent pruning method for gsc,” ISO/IEC JTC1/SC29 WG4 Document m74344, 2025.

[17] Z. Zhu, Y. Liao, and L. Yu, “[gsc][jee6.7] pca-based reduction of sh-ac coefficients for efficient gaussian splat coding,” ISO/IEC JTC1/SC29 WG4 Document m74243, 2025.

[18] J.-Y. Jeong, K.-J. Oh, J.-Y. Lee, G. Lee, A. Dziembowski, and B. Szydelko, “[gsc][jee6.7-related] principal component analysis for reducing sh ac coefficients,” ISO/IEC JTC1/SC29 WG4 Document m74049, 2025.

[19] S. Li, C. Wu, H. Li, X. Gao, Y. Liao, and L. Yu, “Gscodec studio: A modular framework for gaussian splat compression,” IEEE Trans. Circuits and Systems for Video Technology, vol. 36, no. 4, pp. 5483– 5496, 2026.

[20] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d scene representation via self-organizing Gaussian grids,” in European Conference on Computer Vision. Springer, 2024, pp. 18–34.

[21] Q. Yang, L. Yang, G. Van der Auwera, and Z. Li, “Hybridgs: High-efficiency gaussian splatting data compression using dualchannel sparse representation and point cloud encoder,” in The International Conference on Machine Learning, 2025.

[22] Q. Yang, S. Xia, L. Yang, G. Van Der Auwera, and Z. Li, “Efficient gaussian splatting sequence compression with standard video codecs,” arXiv preprint arXiv:2610.07795, 2026.

[23] H. Huang, Q. Yang, M. Liu, W. Huang, Y. Xu, and Z. Li, “Anchordriven compact gaussian splatting for dynamic scene reconstruc-

tion,” IEEE Trans. Visualization and Computer Graphics, vol. 32, no. 7, pp. 7188–7201, 2026.

[24] S. Ma, W. Gao, and Y. Lu, “Rate-distortion analysis for h.264/avc video coding and its application to rate control,” IEEE Trans. Circuits and Systems for Video Technology, vol. 15, no. 12, pp. 1533– 1544, 2005.

[25] Y. Wang, M. Liu, Q. Yang, H. Huang, L. Yang, and Y. Xu, “On the efficient adaptive streaming of 3d gaussian splatting over dynamic networks,” IEEE Trans. Circuits and Systems for Video Technology, vol. 36, no. 4, pp. 4594–4608, 2026.

[26] Q. Yang, Y. Zhang, S. Chen, Y. Xu, J. Sun, and Z. Ma, “Mped: Quantifying point cloud distortion based on multiscale potential energy discrepancy,” IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 6037–6054, 2023.

[27] I.-T. H.265, “Hm reference software for HEVC,” https://vcgit.hhi. fraunhofer.de/jvet/HM/-/tree/HM-18.0?ref type=tags.

[28] I.-T. H.266, “Vtm reference software for VVC,” https://vcgit.hhi. fraunhofer.de/jvet/VVCSoftware VTM.

[29] “Ffmpeg,” https://ffmpeg.org/, accessed: 2026-01-01.

[30] T. Lu, M. Yu, L. Xu, Y. Xiangli, L. Wang, D. Lin, and B. Dai, “Scaffold-gs: Structured 3d gaussians for view-adaptive rendering,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2024, pp. 20 654–20 664.

[31] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, Z. Wang et al., “Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,” Advances in Neural Information Processing Systems, vol. 37, pp. 140 138–140 158, 2024.

[32] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated 3d gaussians with lightweight encodings,” in European Conference on Computer Vision. Springer, 2025, pp. 54–71.

[33] K. Navaneet, K. Pourahmadi Meibodi, S. Abbasi Koohpayegani, and H. Pirsiavash, “Compgs: Smaller and faster gaussian splatting with vector quantization,” in European Conference on Computer Vision. Springer, 2025, pp. 330–349.

[34] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d gaussian splatting for accelerated novel view synthesis,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2024, pp. 10 349–10 358.

[35] Q. Hu, Z. Zheng, H. Zhong, S. Fu, L. Song, X. Zhang, G. Zhai, and Y. Wang, “4dgc: Rate-aware 4d gaussian compression for efficient streamable free-viewpoint video,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2025, pp. 875–885.

[36] H. Huang, Q. Yang, M. Liu, Y. Xu, and Z. Li, “Adc-gs: Anchordriven deformable and compressed gaussian splatting for dynamic scene reconstruction,” arXiv preprint arXiv:2505.08196, 2025.

[37] M. Liu, Q. Yang, H. Huang, W. Huang, Z. Yuan, Z. Li, and Y. Xu, “Light4gs: Lightweight compact 4d gaussian splatting generation via context model,” arXiv preprint arXiv:2503.13948, 2025.

[38] Q. Yang, Z. Ma, Y. Xu, Z. Li, and J. Sun, “Inferring point cloud quality via graph similarity,” IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 44, no. 6, pp. 3015–3029, 2022.

[39] Y. Chen, Q. Wu, M. Li, W. Lin, M. Harandi, and J. Cai, “Fast feedforward 3d gaussian splatting compression,” in The International Conference on Learning Representations, 2025.

[40] WG7, “Revised text of iso/iec fdis 23090-9 geometry-based point cloud compression,” ISO/IEC JTC1/SC29 WG7 Document N0348, 2023.

[41] F. HHI. (2024) Parallel linear assignment sorting (PLAS). [Online]. Available: https://github.com/fraunhoferhhi/PLAS

[42] J. Do, J.-B. Jeong, H. Lee, and G. Bang, “[gsc][jee6.7-related] a lightweight 3d gaussian splats coding framework for 1f-vid track with a single video decoder instance,” ISO/IEC JTC 1/SC 29/WG 4 m73294, 2025.

[43] G. M. Morton, A computer oriented geodetic data base and a new technique in file sequencing. International Business Machines Company, 1966.

[44] J. Ziv and A. Lempel, “A universal algorithm for sequential data compression,” IEEE Trans. Information Theory, vol. 23, no. 3, pp. 337–343, 2003.

[45] X. Zheng, L. Liao, X. Li, J. Jiao, R. Wang, F. Gao, S. Wang, and R. Wang, “Pku-dymvhumans: A multi-view video benchmark for high-fidelity dynamic human modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 530–22 540.

[46] AVS. (2024) Basketball. [Online]. Available: https://huggingface. co/datasets/BestWJH/VRU Basketball

[47] J. Y. Jeong, R. Koo, K.-J. Oh, H.-C. Shin, and G. Lee, “[gsc][jee6.1- related] training method for generating temporally consistent perframe i-3dgs dataset,” ISO/IEC JTC 1/SC 29/WG 4 m71763, 2025.

[48] WG4, “Description of jee 6.1 on data preparation,” ISO/IEC JTC1/SC29 WG4 Document N0676, 2025.

[49] F. Yu, Q. Yang, W. Huang, Y. Guan, Y. Xu, and Z. Li, “Codeccompatible learned compression for 3d gaussian splatting,” arXiv preprint arXiv:2610.26920, 2026.

![](images/efe04c4cfcdb94862be6710aac1d6a567173660bd0929928e099262d94c02be6.jpg)

Qi Yang received the B.S. degree in communication engineering from Xidian University, Xi’an, China, in 2017, and Ph.D degree in information and communication engineering at Shanghai Jiao Tong University, Shanghai, China, 2022. He worked as a researcher in Tencent MediaLab from 2022 to 2024, and worked as a research associate at the University of Missouri–Kansas City from 2024 to 2026. Now, he is a video coding researcher at ByteDance Inc., San Diego, US. He has published more than 35 conference and journal articles, including TPAMI, TIP, TVCG, TMM, TCSVT, ICML, CVPR, ICCV, IJCAI, ACM MM, etc. He is also an active member in standard organizations, including MPEG, AOMedia, and AVS. His research interests include video coding, 3D/4D GS compact generation and compression, 3D point cloud and mesh quality assessment. He received the Best Paper Award of ACM TOMM in 2026, and Best Paper Candidate of VCIP in 2024.

![](images/f45fccdd0e4bd84a4900921459ef9e2a1248594b1a9f2ed49d8e162fba7546c0.jpg)  
core experiments.

Yingzhan Xu received the B.S. and M.S. degrees in Communication Engineering from Shanghai Jiao Tong University, Shanghai, China, in 2018 and 2021, respectively. He is currently a video coding engineer in Bytedance Inc., Beijing. He was actively participating in the research of 3D graphics compression standards. His current research interests include point cloud compression and point cloud processing. During the development of G-PCC and E-GPCC, he cochaired the specification text editing and several

![](images/9defd2d0d3323d58ad972798c91edd568f62b992e30bdb1e14ed747728008869.jpg)

Kai Zhang (M’17, SM’ 20) received the B.S. degree in computer science from Nankai University, Tianjin, China, in 2004. In 2011, he received the Ph.D. degree in computer science from the Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China. From 2011 to 2012, he worked as a researcher in Tencent Inc. Beijing, China. From 2012 to 2016, he worked as a team manager in Mediatek Inc. Beijing, China, leading a research team to propose novel technologies to emerging video coding

standards. From 2016 to 2018, he worked in Qualcomm Inc. San Diego, CA, still focusing on video coding standardization. Now, he is leading the standardization team in Bytedance Inc. San Diego, CA. Dr. Zhang research interests include video/image compression, coding, processing and communication, especially video coding standardization. From 2006, he has contributed more than 500 proposals to JVT, VCEG, JCT-VC, JCT-3V, JVET and AVS, covering many important aspects of major standards such as H.264/AVC, HEVC, 3D-HEVC, VVC and AVS-1,2,3. He has 800+ granted or pending U.S. patents applications. Most of these patents are essential to popular video coding standards. During the development of VVC, Dr. Zhang co-chaired several core experiments and branch of groups. Currently, Dr. Zhang serves as a coordinator of the reference software known as ECM in JVET, to explore video coding technologies beyond VVC. Dr. Zhang has co-authored 100+ papers and reviewed 80+ papers on top-tier journals/conferences. He is the AE of IEEE T-CSVT and IET-IP and a TPC member of DCC 2024 2027. He was an organizer of the Grand Challenge on Neural Network-based Video Coding in ISCAS 2022 2026.

![](images/f7771c6d858c5de8822fd7e2707c7cbaa2a9486b54c7dde21aad0a33df8a13ca.jpg)

Li Zhang (M’11–SM’20) received the Ph.D. degree in Computer Science from the Institute of Computing Technology, Chinese Academy of Sciences, in 2009. She currently leads the Multimedia Lab at ByteDance, San Diego, CA, USA. Prior to this, she held research positions at Qualcomm Inc. and Peking University.

Her research interests include image and video coding and AI-driven multimedia processing, understanding, and generation, covering both 2D and 3D representations. She has made significant contributions to international standards, including VVC (H.266), AVS, MPEG G-PCC, IEEE 1857, JPEG AI, and 3D extensions of HEVC (H.265). She has co-chaired multiple ad hoc groups and core experiments in standardization, served as an Editor for AVS, and was the Main Editor of the Software Test Model for 3DV. Her contributions have been recognized with multiple Certificates of Appreciation from the IEEE Standards Association and AVS. Her work has been widely adopted in international standards and industry deployments.

Dr. Zhang has authored over 220 technical publications and made extensive contributions to standardization, including more than 600 adopted proposals and over 1,000 granted U.S. patents. Her work has received several honors, including the 2026 IEEE Transactions on Circuits and Systems for Video Technology (T-CSVT) Best Paper Award, Best Paper Award at ISCAS 2022 (Visual Signal Processing and Communications track), Top 15 Best Paper at ICME 2025, and Top 10 Best Paper at PCS 2021. She has also led teams to first place in major international challenges, including CVPR CLIC, CVPR NTIRE, and ECCV AIM.

She served as an Associate Editor for IEEE Transactions on Circuits and Systems for Video Technology from 2022 to 2025 and has been an Associate Editor for IEEE Transactions on Image Processing since 2026. She has taken on key roles in major conferences, including Area Chair for AAAI 2027, ICME 2025, ICME 2026, Area Chair for the Multimedia Signal Processing Track at ICASSP 2026-2027, and Senior Program Committee Member for AAAI 2026. She has also contributed to the organization of leading conferences such as ACM Multimedia (ACMMM), ICIP, MMSP, and PCS, and actively organizes special sessions and grand challenges. She was a panelist at DCC 2025 and an industry keynote speaker at ICIP 2025.