# Eficient Gaussian Splating Sequence Compression with Standard Video Codecs

Qi Yang<sup>1</sup>\*, Shuting Xia<sup>2</sup>\*, Le Yang<sup>3</sup>, Geert Van Der Auwera<sup>4</sup>, Zhu Li<sup>1</sup>

<sup>1</sup>: School of Science and Engineering, University of Missouri - Kansas City

<sup>2</sup>: Cooperative Medianet Innovation Center, Shanghai Jiaotong University

<sup>3</sup>: Electrical and Computer Engineering, University of Canterbury

<sup>4</sup>: Qualcomm

qiyang@umkc.edu, xiashuting@sjtu.edu.cn, le.yang@canterbury.ac.nz, geertv@qti.qualcomm.com, lizhu@umkc.edu

## Abstract

This paper presents a novel efective Gaussian Splatting (GS) sequence Compression method that utilizes the Video codec (GSCV). Existing video-based GS sequence compression relies on the Parallel Linear Assignment Sorting (PLAS) and tracked primitive information to convert GS into smooth 2D videos. However, tracked information is not available for most practical applications, and without it, using the vanilla PLAS can generate images exhibiting weak inter-frame correlation, due to its stochastic nature. GSCV incorporates a simple yet eficient Inter-PLAS method to produce close images between the I- and P-frames of GS, enhancing the inter-frame performance of video codec greatly. GSCV also real izes a new pipeline based on the state-of-the-art video codecs with high bit-depth GS images, achieving higher compressibility while simultaneously providing a higher quality upper bound. Experimental results show that the proposed GSCV exhibits obviously improved performance over MPEG video and point cloud-based anchors in GS sequence compression. The code is available at https://github.com/Qi-Yangsjtu/GSCV.

## CCS Concepts

• Computing methodologies → Computer graphics.

## Keywords

Gaussian Splatting, Compression, Video-based Compression

## ACM Reference Format:

Qi Yang<sup>1</sup>\*, Shuting Xia<sup>2</sup>\*, Le Yang<sup>3</sup>, Geert Van Der Auwera<sup>4</sup>, Zhu Li<sup>1</sup>. 2018. Eficient Gaussian Splatting Sequence Compression with Standard Video Codecs. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 18 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

3D Gaussian splatting (GS) [25] has greatly advanced novel view synthesis thanks to its fast generation speed and impressive quality. However, the explicit GS primitive format induces a large data volume, which attracts considerable attention for efective GS data compression [3].

To facilitate GS compression standardization, the Moving Picture Experts Group (MPEG) has established a GS compression Ad Hoc group and delineated two lines of investigation: the A-3DGS and I-3DGS tracks [47]. A-3DGS uses 2D videos/images as the input. With any agreed-upon GS-related representation as the intermediate, A-3DGS involves training GS models as part of the encoding process (i.e., optimization-based compression). I-3DGS aims to compress the trained GS data (i.e., optimization-free compression), resulting in a symmetric output with the same format as the input, such as GGSC [51], HGSC [23], and FCGS [9]. At present, most GS compression eforts focus on A-3DGS. They include HAC [10] and CompGS [30]. Using an efective context model and rate-distortion (RD) loss function, they can achieve a compression ratio of more than 60 to 100x without noticeable distortion [50]. However, these optimization-based algorithms typically involve a complex coding process and rigid rate control [52] based on GPU, which restricts their application to certain use cases. The lightweight requirement from the industry [45] promotes more attention devoted to I-3DGS. One promising approach to develop lightweight solutions is to compress GS data using canonical codecs such as the video or point cloud (PC) codec [36, 38], where the compression is mainly performed on CPUs. A more detailed related work is presented in Appendix A.

Video-based GS compression depends on the use of Parallel Linear Assignment Sorting (PLAS) [32] to convert GS data to a group of smooth 2D maps. PLAS is a progressive process inspired by image sorting [5]. It first randomly places the primitive attributes into 2D grids, then gradually generates a smoother image via Gaussian blurring from coarse to fine as a target to sort the pixels until the 2D grids converge to a smooth distribution. For intra-frame compression strategies, PLAS significantly improves video codec performance, reporting compression ratios of 6 to 20x on HEVC [13]. Based on the preliminary MPEG experiments, for tracked GS sequence compression (see Appendix 5 for an explanation of “tracked”), the video-based anchor (GSCodec Studio [29]) is superior to the PC-based anchor (GPCC v1 [46]) owing to its eficient inter-frame compression [44]. However, few works focused on GS inter-frame compression based on video codecs, and the current

This work is supported in part by an award from NSF 2148382, and a gift grant from Qualcomm. Qi Yang and Shuting Xia contribute equally to this work. Corresponding author: Qi Yang.

anchor has obvious limitations, which motivate the study of this paper.

To the best of our knowledge, GSCodec Studio pioneered the open-source solution that supports GS inter-frame compression based on video codecs. In its inter-prediction mode, PLAS is used for the first frame (i.e., the I-frame), and then the I-frame PLAS index is utilized for sorting the following frames (i.e., the P-frames) within the same group of pictures (GoP) to generate videos. This operation is built upon two assumptions: 1) the arrangement of the I- and P-frame primitives demonstrates an explicit spatial correspondence, whereby primitives with identical indices constitute spatial nearest neighbors sharing close attributes; and 2) directly using the I-frame PLAS index can generate smooth and close P-frame images.

Nevertheless, the above assumptions can be violated in practical applications, resulting in suboptimal results as shown in Fig. 1. For the first assumption, GS data is characterized by permutation invariance as PC [28]. Even though the initial GS data might share some correspondence between adjacent frames (e.g., they may be generated from the same canonical reference after fine-tuning), a considerable number of preprocessing operations before compression, such as sampling and editing [8, 19], can scramble the primitive order. Even a simple primitive shufle operation, which does not change GS data and rendering results, can incur obvious performance degradation (see Appendix H). For the second assumption, GS data is not injective due to �-blending [39], which means spatially neighboring primitives in the I- and P-frames may inherently possess substantially diferent attributes, especially in the large motion region. This observation implies that index-based primitive matching alone is insuficient to ensure that the P-frame images are smooth and close to the corresponding I-frame images. Besides, the current video-based anchor uses 16 bit depth (BD) for coordinates and 8 BD for quantising other attributes, which is not enough for complex GS scenes. Results from [54] showed that the optimal coordinate quantization BD lies within the range of 14 to 18, depending on particular datasets. Considering that GS is very sensitive to coordinate distortion, it would be better to use a larger BD during quantization and realize quality-bitstream balance via compression. Otherwise, the quality upper bound will be limited even under lossless compression.

To improve the GS sequence compression performance and robustness, we propose GSCV, a new pipeline for GS sequence Compression based on Video codec. First, an efective and simple Inter-PLAS method, which consists of two steps, is proposed to reduce the PLAS image diference between adjacent GS frames. Inter-PLAS adopts a spatial-stable initialization (SSI) module designed to address the GS permutation-invariance issue, leading to stable I-frame PLAS image generation without afecting the compression eficiency. It also enables initializing the P-frame with low complexity using the I-frame context. Additionally, Inter-PLAS has an anchor-based PLAS refinement, which is proposed to optimize the P-frame image for better inter-frame prediction. Second, we establish a new PLAS image compression framework compatible with state-of-the-art (SOTA) video codecs, including HEVC and VVC [7]. For a higher quality upper bound, 10 BD images are used for video sequence generation after Inter-PLAS. The coordinate maps, which are quantized by 20 BD, are split into two 10 BD images. Channel padding is adopted for attributes (e.g., opacity and rotation) that are not integer multiples of three channels to simplify compression configuration, as well as making it compatible with diferent image or video codecs. GSCV reports superiority on both MPEG tracked and semi-tracked datasets over benchmark methods.

![](images/dc3e5eda8282619c02cbfe492a7442190d5231b9f76a7f86f29e5684dd8d4930.jpg)  
Figure 1: Examples of current inter-frame GS coding based on the video codecs.

Our contributions can be summarized as follows:

• A new GS sequence compression pipeline based on video codecs, GSCV, is proposed to take advantage of advanced video codecs for eficient inter-frame GS sequence compression;

• We propose a simple yet efective Inter-PLAS scheme to generate close 2D maps for adjacent GS frames to improve inter-frame prediction performance. We use high BD images to form video sequences, which can provide wider bitrate and quality ranges;

• The experiment results show that the proposed GSCV is superior to MPEG video- and PC-based anchors. Ablation study demonstrates the robustness and generalization capability of GSCV.

## 2 Brief Review of PLAS

PLAS is a heuristic algorithm to map GS attributes into 2D maps. Ideally, the generated 2D maps should be as smooth as possible so that they can be efectively compressed using video codecs. Given GS data $G \in R ^ { n \times 5 9 }$ , where � represents the number of primitives in GS, and 59 is the number of feature channels, including 3, 3, 45, 1, 3, and 4 channels for 3D coordinates, color direct current (DC), color spherical harmonic (SH), opacity, scaling, and rotation. A simple pruning based on the linear processing of scaling and opacity, is first applied so that after discarding a minimal set of GS, the remaining primitives can be accommodated in a square 2D grid. Next, a random sorting is used to map the primitive attributes onto 2D grids. � $( m < = 5 9 )$ images are obtained as input for the downstream operations, where � corresponds to the number of attributes selected as references for smooth evaluation. For the pixel values that are from diferent images but have the same position, they originate from the same primitive, indicating that no extra signaling is needed for attribute alignment after sorting.

PLAS takes a progressive approach: 1) a set of Gaussian blur window radius sizes � is selected based on the resolution of the image $I \in R ^ { H \times W \times m } . r = \{ r _ { i } , \tau \}$ with $r _ { 1 } = m a x ( H , W ) / 2 - 1$ , and $r _ { i } = r _ { i - 1 } \times \tau . \ \tau \in ( 0 , 1 )$ is a decay factor, and the minimum value of $r _ { i }$ is set to be 1; 2) a set of block sizes $B _ { i }$ is calculated in accordance with the radius $r _ { i } .$ . Specifically, $B _ { i } = m a x ( B _ { m i n } , f l o o r ( r _ { i } \times 2 + 1 ) + 1 )$ $B _ { m i n }$ is the minimum block size which is set to 16 in the vanilla $\mathrm { P L A S }$ . The definition of $B _ { i }$ ensures that the pixels in the blocks can be divided into groups, each of which has 4 elements; 3) for iteration �, a target smooth image $T _ { i }$ is generated from $I _ { i - 1 } . \ T _ { i }$ is in fact the Gaussian blurred version of $I _ { i - 1 }$ with the kernel size $\sigma _ { i } = f l o o r ( r _ { i } \times 2 + 1 ) , \mathrm { i . e . , } T _ { i } = b l u r ( I _ { i - 1 } , \sigma _ { i } )$ . The iteration starts with $I _ { 0 } = I ,$ the original image; 4) both $T _ { i }$ and $I _ { i - 1 }$ are divided into sub-images with size equal to the block size $B _ { i }$ for batch operation. PLAS permutes the pixels of the image from the previous iteration $I _ { i - }$ within each block, making them as close as possible to those of the corresponding block in $T _ { i }$ . Note that the permutation is carried out for a group of 4 pixels selected randomly from $I _ { i - 1 } ,$ leading to 4! = 24 possible permutations. The one with the smallest L2 loss is chosen to produce $I _ { i } ,$ the image to be passed on to the next iteration; 5) given that $\sigma _ { i }$ and block size $B _ { i }$ are monotonically decreasing with respect to the iteration index �, a coarse-to-fine permutation is applied for the original image � and results in the final PLAS image with smooth texture.

## 3 Inter-PLAS image Generation

Due to the use of random initialization and permutation, the vanilla PLAS generates diferent images for the same or close GS content, as shown in Fig. 2. These images are suitable for the intra-compression mode of the video codec but are challenging for inter-prediction. This motivates us to design the new efective algorithm, Inter-PLAS. It has improved inter-prediction using two steps, namely SSI and anchor-based PLAS refinement. The diagram of Inter-PLAS is given in Fig. 3.

![](images/bf4aad4813cc77aac0ac876d09bc3a5f9186a51603dcc9585ef041284c8315b7.jpg)  
Figure 2: First row: PLAS results of “bartender” frames 1 and 2. Second row: frame 1/2 color DC and scale PLAS image.

## 3.1 Spatial-stable initialization (SSI)

PLAS applies random sorting as the initialization step, which can produce diferent results even for the same GS data. Even though we may fix the random seed, the permutation-invariant characteristic of GS data can still lead to diferent results, indicating that current PLAS is non-deterministic and unstable. For GS sequence coding, a stable, simple, and efective initialization method is required for generating highly reproducible results. To simplify the problem, we use the I- and P-frames from video coding as the reference and target GS frames.

A reasonable assumption for the initialization process is that the spatially proximate primitives should have close attributes. This indicates that for each primitive from the P-frame, we can match it to the nearest neighbor from the I-frame and make them share the same image position. However, this one-to-one matching requires using Earth Mover’s Distance (EMD) whose complexity is $O ( n ^ { 3 } )$ [27]. For GS data that generally has hundreds of thousands to tens of millions of primitives, it is extremely expensive to evaluate the EMD [53]. Considering that we only need a stable initialization rather than requiring this initialization to generate similar images for the I- and P-frames, we address the spatial matching problem with spatial sorting based on Morton code [33].

![](images/8f3a9ba252dd6df355d4cac4baead01bd164e48696cc31b5022be7a61a71c8b3.jpg)  
Figure 3: Framework of Inter-PLAS.

Morton code interleaves the binary representation of the 3D coordinates of the primitive. A smaller Z value reflects that the primitive is closer to the “initial position," which is set to be the top-left corner or the original position. For the I- and P-frames, we first sort the primitives based on their Morton code. The primitives with the same sorting index from the I- and P-frames are roughly considered proximal to each other in the EMD sense. Morton code sorting is lightweight, e.g., it only requires around 0.1 seconds for GS with half a million primitives. Empirical results show that, compared with random initialization, using the Morton code-based sorting as initialization and then performing the vanilla PLAS can generate more similar images for the I and P-frames. Close I-frame all-intra compression performance can also be obtained (see Section 5.3.1), indicating that PLAS is robust to the initialization method in terms of all-intra compression.

![](images/3d24307ce1734f5e69d93053df56f579260914a06205a69a84cab81a72bac5b6.jpg)

![](images/eac6fb07dc0b7f7d2cfdba7269a8553999b85f5385da3464de1724cd09c546ae.jpg)  
Figure 4: PLAS loss and block size variation curve of “bartender” frame 1 (I-frame) and 2 (P-frame).

For the I-frame, we produce a 2D map $I _ { i n i }$ in a row-by-row manner (see Section 5.3.1 for other manners), with the row indices coming from the Morton code sorting results. Then, a vanilla PLAS is applied to obtain the attribute image $I _ { P L A S }$ . The I-frame PLAS sorting index $I d x _ { P L A S }$ is saved as the context for the P-frame ini tialization. For the P-frame, after 3D Morton code sorting to obtain $P _ { i n i } , I d x _ { P L A S }$ is used to re-arrange the index of the $P _ { i n i }$ again and generate the 2D maps $P _ { i n i } ^ { ' }$ as the initialized version. $P _ { i n i } ^ { ' } ,$ a smoother version of $P _ { i n i } ,$ , can be fed into the vanilla PLAS to obtain $P _ { P L A S } ,$ which saves around 15% GPU time. The benefit of this design is illustrated using the MPEG tracked dataset “bartender”: performing PLAS on $I _ { i n i }$ takes 36s, while 30s for $P _ { i n i } ^ { ' } .$ . Based on the PLAS distance and block size variation curves shown in Fig. 4, we found that PLAS exhibits obviously faster convergence speed, especially for the large block size stage. For the I-frame, PLAS uses around 4,000 iterations to converge to the final results, while the P-frame only requires 3,000 iterations. This implies that a smoother initialization can reduce the time complexity of PLAS. Considering that the P-frame requires fewer iterations, especially for larger block sizes, we can use a smaller � to accelerate block size convergence while ensuring satisfactory results.

## 3.2 Anchor-based PLAS refinement

Although $P _ { P L A S }$ generated using SSI and the vanilla PLAS is smooth as a whole, and it is closer to ����� than applying the vanilla PLAS on the I- and P-frames separately, evident discrepancies may continue to be present in some content or attributes. An example is shown in the first and second rows of Fig. 5. Using a video codec such as HEVC in the inter-prediction mode only has around 8% ∼ 14% bitrate savings compared to the all-intra mode. This motivates us to propose an anchor-based PLAS refinement to adjust $P _ { i n i } ^ { ' }$ rather than directly using a vanilla PLAS. Specifically, we disable the generation of the blurred target $T _ { i }$ in PLAS and use $I _ { P L A S }$ instead as the anchor to progressively smooth $P _ { i n i } ^ { ' } ,$ , resulting in the proposed anchor-based PLAS refinement. Fig. 5 third row shows the results of applying the modified PLAS with the proposed anchor-based refinement to $P _ { i n i ^ { \cdot } } ^ { ' } A$ prediction gain (PG) is used to quantify the benefits of inter-frame compression:

$$
\mathrm { P G ( I , P ) } = 1 0 \mathrm { l o g } _ { 1 0 } ( \mathrm { E } _ { \mathrm { I } } / \mathrm { E } _ { \mathrm { r e s } } ) , \mathrm { E } _ { \mathrm { I } } = \mathrm { v a r ( I ) , ~ } \mathrm { E } _ { \mathrm { r e s } } = \mathrm { v a r ( I - P ) , }\tag{1}
$$

where var(·) represents the variation operator. For the vanilla GS, only the color DC map has 6.61 dB PG, while scaling and rotation have negative PG, which means: 1) the pixel-to-pixel similarity between the I- and P-frames is low; and 2) the inter-frame spatial correlation ofcolor DC is more pronounced. An obviously increased PG is observed for the anchor-based PLAS refinement, which means I- and P-frames are now closer to each other, suggesting better interprediction potential.

The vanilla PLAS and proposed anchor-based PLAS focus on diferent perspectives, namely intra-frame smoothness v.s. interframe similarity. Based on Fig. 5 (g), high-frequency noise, such as “salt&pepper” noise, is observed, while Fig. 5 (d) is spatially smoother. For video codecs in inter-prediction mode, both intraframe content smoothness and inter-frame similarity will influence P-frame compression eficiency. With the developed Inter-PLAS, higher inter-frame similarity is achieved: over 20% bitrate saving can be obtained for P-frame coding.

One potential shortcoming of the anchor-based PLAS refinement comes from directly using a smooth image as the target, which is inconsistent with PLAS having a coarse-to-fine process, leading to slow convergence. Fortunately, inspired by the discussion about

![](images/734778c4866c3c7434eecd148134eef2bf96b5fc21a7cacffd8380ebc2c30519.jpg)  
(g) PG=14.2 dB  
(h) PG=8.50 dB  
(i) PG=4.23 dB  
Figure 5: Anchor-based PLAS refinement. First row: PLAS results for the 1st frame of “bartender”; second row: PLAS results for $P _ { i n i } ^ { ' }$ third row: the proposed anchor-based PLAS refinement for $P _ { i n i } ^ { ' } .$

Fig. 4, we find a very easy method to realize complexity control without significantly sacrificing compression eficiency, which is reported in Appendix I.

## 4 Gaussian splatting sequence coding using video codec

Based on the images generated by Inter-PLAS, the proposed effective GS sequence compression scheme GSCV is established, as shown in Fig. 6. It can be seen that first, we combine 8 frames into 1 GoP, with the first frame designated as the I-frame and the remaining 7 frames as the P-frames. SSI + Vanilla PLAS and Inter-PLAS are used for the I- and P-frames, respectively, to project primitives onto 2D grids. After that, 20 BD is used to quantize the coordinates and 10 BD is applied to quantize other attributes. For each GS frame, we generate 1, 15, and 1 frame images for the color DC, SH, and scaling. For opacity and rotation, we use the mid-value of BD to pad two channels to form 1 and 2 images, respectively. Most video codecs cannot compress 20 BD images directly. To improve the compatibility of GSCV, we split the coordinates map into 2 images, namely the high part (HP) and low part (LP) of data. Each part takes 10 BD.

For each GS frame, 22 images are generated. We extract the corresponding images from each frame within a GoP to group 22 videos and save these videos in YUV444 format. Lossless compression is used for coordinates, while other attributes employ the lossy mode. Note that we directly consider the video data as YUV rather than RGB, omitting the operation that converts data from RGB to the

![](images/41766512d38ca4a34cacf9829f1ab27cb79205dc23b79287db5035f4ec1a714b.jpg)  
Figure 6: Diagram of the proposed GSCV.

YUV color space for YUV file generation. The reasons are twofold: 1) except for the color DC and SH, there is no inter-relationship analogous to the other attributes; applying an RGB-to-YUV transformation is thus redundant and ofers no benefit for compression; 2) even with a lossless compression, the RGB-to-YUV transformation introduces inevitable calculation errors that result in noticeable quality degradation (especially for the coordinate HP image) at high bitrate due to the lossy forward and inverse processes. The influence of introducing RGB-to-YUV is reported in Appendix G.

## 5 Experiments

In this section, the performance of the proposed GSCV on an authoritative dataset from MPEG is reported. We compare it against two anchors adopted in the MPEG GS compression working group, namely GSCodec Studio and GPCC v1 (see Appendix C for an introduction). The ablation study is presented at the end of this section to demonstrate the efectiveness of diferent modules.

## 5.1 Dataset and GSCV Implementation

Three GS sequences with tracked and semi-tracked versions from the MPEG GS compression working group are used to test the proposed method: “bartender”, “cinema”, and “breakfast” (see Appendix D for details). Each sequence has 32 frames and we use the first 8 frames as 1 GoP in the following experiment. Here, “tracked” means that 32 frames GS are refined from a shared canonical GS model, which was trained using the merged point cloud with ground truth images from all viewpoints and time instants. Densification is disabled during the refinement process, ensuring that all the frames have the same number of primitives. “Semi-tracked” means densification is allowed during the refinement, which results in a diferent number of primitives for diferent frames. More details about the tracked and semi-tracked sequences generation can be found in [24]. To simulate the practically interesting situation that some operations might afect the permutation of primitives, we shufle randomly the tracked data of each frame before testing. For video sequence generation, we use LightGaussian to prune the semi-tracked dataset to ensure frames within 1 GoP have the same number of primitives by removing the minimum element. No shufle is introduced for semi-tracked data. The implementation of the proposed Inter-PLAS is based on the vanilla PLAS [21] realized in PyTorch. All the attributes are used in PLAS image generation, i.e., � = 59 (see Appendix K for other attribute settings). Two popular video codecs, HEVC reference software (HM) v18.0 [17] and VVC reference software (VTM) v23.11 [18], are used as the codec anchor of GSCV. The detailed codec configuration is presented in Appendix B. As suggested by [43], views 9 and 11 for “bartender” and “cinema”, while views 6 and 8 are used for “breakfast” to compute the quality metrics. The reported results are obtained on a single NVIDIA L40s GPU and AMD EPYC 7H12 CPU.

## 5.2 Overall evaluation

Fig. 7 illustrates the rate (MB) vs. distortion (RD) curves of the proposed GSCV on the MPEG tracked and semi-tracked dataset. The results of bits per primitive (BPP) and the Bjøntegaard delta bitrate (BD-Rate) [6] performance are shown in Appendix E, and the RD curves per component are shown in Appendix O. The quantization parameters (QPs) of five bitrates are presented in Appendix B. GSCV shows obvious superiority over GSCodec Studio and GPCC v1. For example, although both use video codecs, GSCV reports around 8 dB gain on “bartender” compared with GSCodec Studio when the bitstream is 10MB, thanks to the proposed Inter-PLAS and new compression framework. The inter-frame compression performance ofGSCodec Studio highly relies on the initial primitive permutation between diferent frames as discussed in Appendix H. With VVC, close performance compared with HEVC is obtained. This is not consistent with the case of ordinary videos where VVC can achieve 50% gain over HEVC. Recent V-PCC Amd1 work also found this problem [1]. The reason is that the current video codecs are optimized for natural images that are completely diferent from the PLAS images. Based on the residual analysis in Appendix N, the PLAS image shows a more pronounced heavy-tail behavior than natural images. One potential enhancement to solve this problem is to use a “Sandwich network” [16] to convert PLAS images to a data distribution more suitable for traditional video codec compression (see Appendix R). The same phenomenon also occurs when using FFMPEG libx265 and libx264, as reported in Appendix P. This also demonstrates that the superiority of the proposed GSCV does not mainly come from using a better video codec than GSCodec Studio. Instead, it originates from the more efective and robust pipeline pivoting a better GS video generation.

Second, GSCV provides better performance than GPCC v1 with the middle to high bitstreams, while poorer results with the low bitstream. The underlying reasons are twofold: 1) the setting of the color DC QP is too large, leading to inappropriate bitrate distribution. We tried other QP and YUV modes in Appendix G, and 2) GPCC v1 is more efective in encoding the coordinate which is always lossless compressed. A greater proportion of geometry at lower bitrates can lead to improved GPCC performance. A detailed bitrate allocation and coding time are reported in Appendix F and P with sample snapshots given in Appendix T. PLAS coordinate LP images are close to noise maps, resulting in limited performance when using video codecs. Third, GSCV reports consistent performance on both tracked and semi-tracked sequences, indicating the robustness of the proposed Inter-PLAS. Semi-tracked sequences are more challenging than the tracked version, indicating densification tends to amplify the inter-frame diferences. The comparable RD performance of GSCodec Studio on semi-tracked data and randomly shufled tracked data suggests that random shufling can, to some extent, capture the instability in primitive correspondence introduced by practical preprocessing operations such as densification and pruning. More experiment results on datasets with larger motion, and comparison with A-3DGS are illustrated in Appendix S and Q.

![](images/aff7c10c2ca7088fc570d47b3ac1af94c0000dde0518275bcbbd62df59cb04bc.jpg)

![](images/c5d06d0b4f2004405d6923f08e197846fe2ca78e70118173d14544963d21c758.jpg)

![](images/45ba3731ad522b0759c4bf45d78f62ce933cf0212937a345113103cb65b5c830.jpg)

![](images/24904b381f3b07dc22b89cb4d59725c8fb94b4b5545ea1d05d343d955638d8f6.jpg)  
(a)

![](images/6351251d75acddf77211f66ea0a063857f2692189c6f9133d475c8ea48c33618.jpg)  
(b)

![](images/5664f79d92bef20b9ff1939fb2dcb219cc745ea2fac527936537e26bd13af023.jpg)  
(c)  
Figure 8: Ablation study on (a)-(b): PLAS initialization and (c): efectiveness of Inter-PLAS.

## 5.3 Ablation Study

5.3.1 Influence ofPLAS initialization on image generation and compression. The first step of Inter-PLAS provides a good initialization method to solve the problem introduced by GS permutationinvariant characteristics, generating a stable I-frame image as the P-frame’s context. The initialization should not damage the intraframe compression efectiveness of the I-frame, considering that the I-frame occupies more bitstream than the P-frame. We compare the results of GSCV all-intra compression mode on PLAS image with diferent initialization methods followed by the vanilla PLAS with tracked sequences: 1) random initialization; 2) Morton1, using 3D Morton scan first and then generating 2D maps row by row; 3) Morton2, using 3D Morton scan first and then mapping it to 2D maps based on 2D Morton scan order; 4) using a PLAS index from another frame as context. The visualization of four initialization methods and RD are shown in Figs. 8 (a) and (b). We can see that the RD curves of diferent initialization methods are almost the same, indicating that the PLAS is robust to initialization in all-intra compression. However, using another frame as context requires the shortest PLAS time, indicating that a smoother initialization can accelerate PLAS convergence to stable states.

5.3.2 Efectiveness ofInter-PLAS. Although GSCodec Studio also supports inter-prediction for GS sequences as GSCV, it directly uses the PLAS index from the I-frame to sort P-frames, which cannot ensure that the P-frame image aligns well with the I-frame. To highlight this problem, we use GSCV based on three diferent methods to generate images with tracked sequence: 1) using the vanilla PLAS on each frame separately; 2) using SSI to initialize the P-frames followed by the vanilla PLAS to refine images; and 3) the proposed Inter-PLAS that uses the I-frame to refine P-frame images. The results are shown in Fig. 8 (c). As we discussed for Fig. 2, directly using the vanilla PLAS will generate diferent images for close content due to the random process, resulting in the worst inter-prediction results. Using a good context to initialize the Pframe with PLAS can generate better P-frame images. However, vanilla PLAS only focuses on intra-frame smoothness, while there is no supervision on inter-frame similarity, leading to limited gain for inter-prediction. The proposed Inter-PLAS solves the image alignment problem by a simple and efective strategy with high PG, showing an obvious gain on inter-prediction cases.

## 6 Conclusions

In this paper, we propose GSCV, an efective GS sequence compression pipeline based on video codecs. We design an Inter-PLAS method to generate high-correlation GS images, which can facili tate the inter-frame compression for video codecs greatly. We also design a new GS image compression pipeline based on SOTA video codecs. By using high BD images and being compatible with prevalent video codecs, GSCV reports better performance on both tracked and semi-tracked sequences than video- and PC-based anchors used in the current MPEG standardization study.

Limitations: PLAS images are very diferent from natural images, leading to restricted compression ratios of the proposed GSCV, which is now based on generic video codecs. Dividing the GS data into multiple video sequences and compressing them separately cannot fully utilize the primitive intra-channel correlations, especially for the color SH (see Appendix J). We will explore how to better group color SH images in future work. For tracked GS data after permutation, Inter-PLAS cannot completely recover the primitive correspondence due to its heuristic mechanism inherited from PLAS.

## References

[1] Patrice Alface and Lukasz Kondrad. 2026. [V-PCC][Amd1] Support for VVC encoding. ISO/IEC JTC 1/SC 29/WG 7 m76603 (2026).

[2] AVS. 2024. Basketball. https://huggingface.co/datasets/BestWJH/VRU\_Basketball

[3] Milena T Bagdasarian, Paul Knoll, Y Li, Florian Barthel, Anna Hilsmann, Peter Eisert, and Wieland Morgenstern. 2025. 3dgs. zip: A survey on 3d gaussian splatting compression methods. In Computer Graphics Forum, Vol. 44. Wiley Online Library, e70078.

[4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. 2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 5470–5479.

[5] Kai Uwe Barthel, Nico Hezel, Klaus Jung, and Konstantin Schall. 2023. Improved evaluation and generation of grid layouts using distance preservation quality and linear assignment sorting. In Computer Graphics Forum, Vol. 42. Wiley Online Library, 261–276.

[6] Gisle Bjøntegaard. 2001. Calculation of Average PSNR Diferences Between RD-Curves. VCEG Contribution VCEG-M33. ITU-T Video Coding Experts Group, Austin, TX, USA.

[7] Benjamin Bross, Ye-Kui Wang, Yan Ye, Shan Liu, Jianle Chen, Gary J Sullivan, and Jens-Rainer Ohm. 2021. Overview of the versatile video coding (VVC) standard and its applications. IEEE Trans. Circuits and Systems for Video Technology 31, 10 (2021), 3736–3764.

[8] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin. 2024. Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 21476–21485.

[9] Yihang Chen, Qianyi Wu, Mengyao Li, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. 2025. Fast Feedforward 3D Gaussian Splatting Compression. In The International Conference on Learning Representations.

[10] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. 2024. Hac: Hash-grid assisted context for 3d gaussian splatting compression. In European Conference on Computer Vision. Springer, 422–438.

[11] Yihang Chen, Qianyi Wu, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. 2025. Hac++: Towards 100x compression of 3d gaussian splatting. arXiv preprint arXiv:2501.12255 (2025).

[12] Ricardo L. De Queiroz and Philip A. Chou. 2016. Compression of 3D Point Clouds Using a Region-Adaptive Hierarchical Transform. IEEE Trans. Image Processing 25, 8 (2016), 3947–3956. doi:10.1109/TIP.2016.2575005

[13] Jihoon Do, Jong-Beom Jeong, Hahyun Lee, and Gun Bang. 2025. [GSC][JEE6.7- related] A Lightweight 3D Gaussian Splats Coding Framework for 1F-vid Track with a Single Video Decoder Instance. ISO/IEC JTC 1/SC 29/WG 4 m73294 (2025).

[14] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. 2024. Lightgaussian: Unbounded 3d gaussian compression with 15x reduc tion and 200+ fps. Advances in Neural Information Processing Systems 37 (2024), 140138–140158.

[15] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. 2025. Eagles: Eficient accelerated 3d gaussians with lightweight encodings. In European Conference on Computer Vision. Springer, 54–71.

[16] Onur G Guleryuz, Philip A Chou, Berivan Isik, Hugues Hoppe, Danhang Tang, Ruofei Du, Jonathan Taylor, Philip Davidson, and Sean Fanello. 2024. Sandwiched Compression: Repurposing Standard Codecs with Neural Network Wrappers. arXiv preprint arXiv:2402.05887 (2024).

[17] ITU-T H.265. [n. d.]. HM reference software for HEVC. https://vcgit.hhi. fraunhofer.de/jvet/HM/-/tree/HM-18.0?ref\_type=tags.

[18] ITU-T H.266. [n. d.]. VTM reference software for VVC. https://vcgit.hhi. fraunhofer.de/jvet/VVCSoftware\_VTM.

[19] Alex Hanson, Allen Tu, Vasu Singla, Mayuka Jayawardhana, Matthias Zwicker, and Tom Goldstein. 2025. Pup 3d-gs: Principled uncertainty pruning for 3d gaussian splatting. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition

Conference. 5949–5958.

[20] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel Brostow. 2018. Deep blending for free-viewpoint image-based ren dering. ACM Trans. Graphics 37, 6 (2018), 1–15.

[21] Fraunhofer HHI. 2024. Parallel Linear Assignment Sorting (PLAS). https://github. com/fraunhoferhhi/PLAS

[22] Qiang Hu, Zihan Zheng, Houqiang Zhong, Sihua Fu, Li Song, Xiaoyun Zhang, Guangtao Zhai, and Yanfeng Wang. 2025. 4DGC: Rate-Aware 4D Gaussian Compression for Eficient Streamable Free-Viewpoint Video. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 875–885.

[23] He Huang, Wenjie Huang, Qi Yang, Yiling Xu, and Zhu Li. 2025. A hierarchical compression technique for 3d gaussian splatting compression. In IEEE Int. Conf. Acoustics, Speech and Signal Processing. 1–5.

[24] Jun YoungJeong, Reagan Koo, Kwan-Jung Oh, Hong-Chang Shin, and Gwangsoon Lee. 2025. [GSC][JEE6.1-related] Training Method for Generating Temporally Consistent Per-frame I-3DGS Dataset. ISO/IEC JTC 1/SC 29/WG 4 m71763 (2025).

[25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

[26] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Trans. Graphics 36, 4 (2017), 1–13.

[27] Harold W Kuhn. 1955. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly 2, 1-2 (1955), 83–97.

[28] Jiaxin Li, Ben M Chen, and Gim Hee Lee. 2018. So-net: Self-organizing network for point cloud analysis. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 9397–9406.

[29] Sicheng Li, Chengzhen Wu, Hao Li, Xiang Gao, Yiyi Liao, and Lu Yu. 2025. GSCodec Studio: A Modular Framework for Gaussian Splat Compression. arXiv preprint arXiv:2506.01822 (2025).

[30] Xiangrui Liu, Xinju Wu, Pingping Zhang, Shiqi Wang, Zhu Li, and Sam Kwong. 2024. Compgs: Eficient 3d scene representation via compressed gaussian splat ting. In Proc. ACM Int. Conf. Multimedia. 2936–2944.

[31] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scafold-gs: Structured 3d gaussians for view-adaptive rendering. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 20654–20664.

[32] Wieland Morgenstern, Florian Barthel, Anna Hilsmann, and Peter Eisert. 2024. Compact 3d scene representation via self-organizing Gaussian grids. In European Conference on Computer Vision. Springer, 18–34.

[33] Guy M Morton. 1966. A computer oriented geodetic data base and a new technique in file sequencing. International Business Machines Company.

[34] KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, and Hamed Pirsiavash. 2025. Compgs: Smaller and faster gaussian splatting with vector quantization. In European Conference on Computer Vision. Springer, 330– 349.

[35] Simon Niedermayr, Josef Stumpfegger, and Rüdiger Westermann. 2024. Compressed 3d gaussian splatting for accelerated novel view synthesis. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 10349–10358.

[36] Sebastian Schwarz, Marius Preda, Vittorio Baroncini, Madhukar Budagavi, Pablo Cesar, Philip A. Chou, Robert A. Cohen, Maja Krivokuća, Sébastien Lasserre, Zhu Li, Joan Llach, Khaled Mammou, Rufael Mekuria, Ohji Nakagami, Ernestasia Siahaan, Ali Tabatabai, Alexis M. Tourapis, and Vladyslav Zakharchenko. 2019. Emerging MPEG Standards for Point Cloud Compression. IEEE Journal on Emerging and Selected Topics in Circuits and Systems 9, 1 (2019), 133–148. doi:10.1109/JETCAS.2018.2885981

[37] Yiting Shao, Zhaobin Zhang, Zhu Li, Kui Fan, and Ge Li. 2017. Attribute compression of 3D point clouds using Laplacian sparsity optimized graph transform. In IEEE Visual Communications and Image Processing. 1–4. doi:10.1109/VCIP.2017. 8305131

[38] Gary J Sullivan, Jens-Rainer Ohm, Woo-Jin Han, and Thomas Wiegand. 2012. Overview of the high eficiency video coding (HEVC) standard. IEEE Trans. Circuits and Systems for Video Technology 22, 12 (2012), 1649–1668.

[39] Ayush Tewari, Justus Thies, Ben Mildenhall, Pratul Srinivasan, Edgar Tretschk, Wang Yifan, Christoph Lassner, Vincent Sitzmann, Ricardo Martin-Brualla, Stephen Lombardi, et al. 2022. Advances in neural rendering. In Computer Graphics Forum, Vol. 41. Wiley Online Library, 703–735.

[40] Boyuan Tian, Qizhe Gao, Siran Xianyu, Xiaotong Cui, and Minjia Zhang. 2025. FlexGaussian: Flexible and Cost-Efective Training-Free Compression for 3D Gaussian Splatting. arXiv preprint arXiv:2507.06671 (2025)

[41] Gregory K. Wallace. 1991. The JPEG still picture compression standard. Commun. ACM 34, 4 (1991), 30–44. doi:10.1145/103085.103089

[42] Chenjunjie Wang, Shashank N Sridhara, Eduardo Pavez, Antonio Ortega, and Cheng Chang. 2025. Adaptive Voxelization for Transform coding of 3D Gaussian splatting data. arXiv preprint arXiv:2506.00271 (2025).

[43] WG4. 2025. Description of JEE 6.1 on data preparation. ISO/IEC JTC1/SC29 WG4 Document N0676 (2025).

[44] WG4. 2025. [GSC][JEE2] A Potential Video-based Anchor for Gaussian Splats Coding. ISO/IEC JTC1/SC29 WG4 Document m72063 (2025).

[45] WG4. 2025. [GSC][JEE6.4-related] On the use case and requirements for lightweight GSC. ISO/IEC JTC1/SC29 WG4 Document m72430 (2025).

[46] WG7. 2023. Revised Text of ISO/IEC FDIS 23090-9 Geometry-based Point Cloud Compression. ISO/IEC JTC1/SC29 WG7 Document N0348 (2023).

[47] WG7. 2025. [GSC][JEE6.4] Draft use cases and requirements for Gaussian splat coding. ISO/IEC JTC1/SC29 WG7 Document m71640 (2025)

[48] Adam Wieckowski, Jens Brandenburg, Tobias Hinz, Christian Bartnik, Valeri George, Gabriel Hege, Christian Helmrich, Anastasia Henkel, Christian Lehmann, Christian Stofers, Ivan Zupancic, Benjamin Bross, and Detlev Marpe. 2021. VVenC: An Open And Optimized VVC Encoder Implementation. In Proc. IEEE Inter. Conf. Multimedia Expo Workshops (2021). 1–2. doi:10.1109/ICMEW53276. 2021.9455944

[49] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024. 4d gaussian splatting for real-time dynamic scene rendering. In Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition. 20310–20320.

[50] Yuke Xing, William Gordon, Qi Yang, Kaifa Yang, Jiarui Wang, and Yiling Xu. 2025. 3DGS-VBench: A Comprehensive Video Quality Evaluation Benchmark for 3DGS Compression. arXiv preprint arXiv:2508.07038 (2025).

[51] Qi Yang, Kaifa Yang, Yuke Xing, Yiling Xu, and Zhu Li. 2024. A Benchmark for Gaussian Splatting Compression and Quality Assessment Study. In Proc. ACM Int. Conf. Multimedia in Asia. Association for Computing Machinery, Article 12, 8 pages. doi:10.1145/3696409.3700172

[52] Qi Yang, Le Yang, Geert Van der Auwera, and Zhu Li. 2025. HybridGS: High-Eficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder. In The International Conference on Machine Learning.

[53] Qi Yang, Yujie Zhang, Siheng Chen, Yiling Xu, Jun Sun, and Zhan Ma. 2023. MPED: Quantifying Point Cloud Distortion Based on Multiscale Potential Energy Discrepancy. IEEE Trans. Pattern Analysis and Machine Intelligence 45, 5 (2023), 6037–6054. doi:10.1109/TPAMI.2022.3213831

[54] Alexandra Zaghetto, Danillo Graziosi, and Ali Tabatabai. 2024. 3DGS Geometry Quantization. ISO/IEC JTC 1/SC 29/WG 7 m69034 (2024).

[55] Xiaoyun Zheng, Liwei Liao, Xufeng Li, Jianbo Jiao, Rongjie Wang, Feng Gao, Shiqi Wang, and Ronggang Wang. 2024. Pku-dymvhumans: A multi-view video benchmark for high-fidelity dynamic human modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22530–22540.

# Appendix of Eficient Gaussian Splatting Sequence Compression with Standard Video Codecs

This is the Appendix of "Eficient Gaussian Splatting Sequence Compression with Standard Video Codecs".

## A Related Work

A-3DGS: A-3DGS optimizes the GS generation process by adding extra constraints during training to obtain a more compact data representation. Scafold-GS [31] proposed an anchor-based latent representation for primitive description, which reduces the spatial redundancy and also serves as the baseline of HAC [10] and HAC++ [11]. CompGS [30] also used an anchor-based approach with an entropy model to realize eficient adjacent primitive feature prediction. Another prevalent compact representation is using a codebook and vector quantization to map the GS attributes to a limited feature space, such as [34, 35]. Considering that the GS densification process can generate many redundant primitives, LightGaussian [14] and EAGLES [15] developed efective GS pruning methods by calculating an importance score for each primitive. Empirical results show that removing 50% to 60% of primitives followed by finetuning can still ofer a quality comparable to that of the vanilla GS. In summary, A-3DGS methods can realize an impressive compression ratio without evident quality degradation, indicating there is a huge room to optimize the GS generation process, specifically over the densification and high-dimensional spherical harmonic (SH) coeficients. These advances also inspired the study of the I-3DGS approaches.

I-3DGS: Diferent from A-3DGS, I-3DGS methods focus on how to compress trained GS data without resorting to training optimization that relies on the GPUs. GS data format is similar to that of PC. An adaptive voxelization was designed in [42] for GS. It takes into consideration that the primitive distribution is irregular to facilitate the usage of the MPEG GPCC codec on GS compression. Also inspired by the PC compression [37], GGSC [51] used a graph signal processing (GSP)-based approach to compress GS attributes through discarding high-frequency components. HGSC [23] proposed an optimization-free GS pruning method based on LightGaussian, followed by a hierarchical strategy to divide prim itives into diferent layers for progressive coding. FCGS [9] used grids to establish an efective context model for learning-based GS compression. GSCodec Studio [29] is the first modular framework that supports GS reconstruction, compression, and rendering, in which video- and PC-based GS compression are unified under this pipeline. Except for GSCodec Studio, other methods reviewed in this section all focus on GS intra-frame compression, which motivated us to further explore how to realize efective and stable GS inter-frame compression in this paper.

## B Bitrate Setting Details

In Table 1, we show the QP values of five bitstreams for GSCV. For HEVC lossy compression, “encoder\_randomaccess\_main\_rext.cfg” is used for inter-frame compression; for lossless compression, two flags are set: “TransquantBypassEnabledFlag = 1” and “CUTransquantBypassFlagForce = 1”. For VVC lossy compression, “encoder\_ randomaccess\_vtm.cfg” is used; for lossless compression, “lossless444.cfg” and “lossless.cfg” are applied. We report using HM18.0 “encoder\_low\_delay” configuration with diferent coding sequencing in Appendix L.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>QP</td></tr><tr><td rowspan=1 colspan=1>Rate</td><td rowspan=1 colspan=1>Coordinate</td><td rowspan=1 colspan=1>Color DC</td><td rowspan=1 colspan=1>Color SH (degree 1/2/3)</td><td rowspan=1 colspan=1>opacity</td><td rowspan=1 colspan=1>scale</td><td rowspan=1 colspan=1>rotation</td></tr><tr><td rowspan=1 colspan=1>R05</td><td rowspan=5 colspan=1>lossless</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0/0/0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>R04</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7/12/17</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>R03</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>17/22/27</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>R02</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>22/27/32</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>7</td></tr><tr><td rowspan=1 colspan=1>R01</td><td rowspan=1 colspan=1>47</td><td rowspan=1 colspan=1>32/37/42</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>17</td></tr></table>

Table 1: QP of five bitrates of GSCV

## C Baseline Anchors

GSCodec Studio: GSCodec Studio [29] uses FFMPEG x265 codec, which supports GoP-wise compression with inter-prediction. The default GoP size is 16, and 8 BD is used to quantize the PLAS image. In our experiment, we change the GoP size to 8 for fair comparison.

GPCC v1: GPCC v1 [46] was originally designed for PC data compression. MPEG extended its interface that supports GS data loading and used it as the PC-based anchor for GS compression. The compression of coordinates and other attributes inherits the compression method of PC: Octree for coordinates and RAHT [12] for attributes compression. It uses 18 bits to quantize coordinates and 12 bits for other attributes. Current GPCC v1 does not support GS inter-frame compression, therefore, we use all-intra mode in Section 5.

## D Dataset Information

The number of primitives for MPEG tracked and semi-tracked sequences is shown in Table 3. For tracked sequences, the primitive number is the same for diferent frames. For semi-tracked sequences, the red number is the minimum primitives within this GoP, which will be used as the target number after pruning. Considering that densification is allowed during refinement, semi-tracked sequences have more primitives than the tracked version.

## E BPP-based RD Curve

The results of the rate (BPP) vs. distortion are summarized in Fig. 9. The primitive numbers of the tracked and semi-tracked “bartender”, “breakfast”, and “cinema” are 567,724/569,246, 528,154/530,509, and 423,249/425,953, respectively. The Bjøntegaard delta bitrate (BD-Rate) [6] performance is shown in Tab. 2.

## F Bitrate Allocation

To illustrate the bitrate distribution for diferent components on GSCV, we use tracked “bartender” rate 1, 3, and 5 with HEVC as an example. The BPP is shown in Fig. 10. We can see that for high BPP, color SH occupies more bitstreams, while geometry-related features contribute more to low BPP. Considering that GS quality is more sensitive to geometry-related features, saving bitstream from color-related features can provide a better quality-bitstream tradeof. The coding time, which is shown in Fig. 11, illustrates that the decoding is faster than encoding, and this coding time is thus only for complexity illustration purpose. Currently, we establish GSCV based on the HM or VTM software, which is mainly used for standardization studies. The coding speed is relatively slow owing to the lack of engineering optimizations (e.g., using multithreaded processing). Faster coding speed can be achieved by choosing “encoder\_lowdelay\_main\_rext.cfg” file or using optimized codec platforms (e.g., FFMPEG or VVdeC [48])

![](images/d2ae9bcefbb767a949deb610089c23f2df91db30f24b33901723510740d1a4ac.jpg)

![](images/392d57d145a3e1e2b6847f4db84dfb0183e447afb3982a98aa35da683e9589bc.jpg)

![](images/f9539d9bcef7d3eba941c88e9f0e386de75a31f7724b8f5552950ba4a2ecf1b0.jpg)

Figure 9: PSNR performance comparison on MPEG tracked and semi-tracked dataset.
<table><tr><td rowspan="2">Sequence</td><td rowspan="2">Tracking mode</td><td colspan="2">Proposed (HEVC)</td><td colspan="2">Proposed (VVC)</td></tr><tr><td>Ref. GSCodec</td><td>Ref. GPCC</td><td>Ref. GSCodec</td><td>Ref. GPCC</td></tr><tr><td rowspan="2">Bartender</td><td>Tracked</td><td>-52.18%</td><td>-14.81%</td><td>-51.64%</td><td>-13.59%</td></tr><tr><td>Semi-tracked</td><td>-47.33%</td><td>-10.18%</td><td>-45.86%</td><td>-6.70%</td></tr><tr><td rowspan="2">Breakfast</td><td>Tracked</td><td>-50.01%</td><td>-13.81%</td><td>-50.29%</td><td>-13.43%</td></tr><tr><td>Semi-tracked</td><td>-46.98%</td><td>-11.06%</td><td>-45.29%</td><td>-7.60%</td></tr><tr><td rowspan="2">Cinema</td><td>Tracked</td><td>-51.64%</td><td>-12.75%</td><td>-51.81%</td><td>-12.60%</td></tr><tr><td>Semi-tracked</td><td>-46.34%</td><td>-7.85%</td><td>-44.79%</td><td>-4.92%</td></tr></table>

Table 2: BD-Rate results of the proposed method under HEVC and VVC configurations. Negative values indicate bitrate savings relative to the corresponding reference method.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Primitive</td><td rowspan=1 colspan=1>Number</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Sequence</td><td rowspan=1 colspan=2>Bar</td><td rowspan=1 colspan=1>tender</td><td rowspan=1 colspan=2>Bre</td><td rowspan=1 colspan=1>akfast</td><td rowspan=1 colspan=1>Ci</td><td rowspan=1 colspan=1>nema</td></tr><tr><td rowspan=1 colspan=1>Frame Index</td><td rowspan=1 colspan=2>Tracked</td><td rowspan=1 colspan=1>Semi-tracked</td><td rowspan=1 colspan=2>Tracked</td><td rowspan=1 colspan=1>Semi-tracked</td><td rowspan=1 colspan=1>Tracked</td><td rowspan=1 colspan=1>Semi-tracked</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>570255</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>531263</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>426968</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>570104</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>530997</td><td></td><td rowspan=1 colspan=1>426752</td></tr><tr><td rowspan=1 colspan=1>3</td><td></td><td></td><td rowspan=1 colspan=1>569842</td><td></td><td></td><td rowspan=1 colspan=1>530998</td><td></td><td rowspan=1 colspan=1>426497</td></tr><tr><td rowspan=1 colspan=1>4</td><td></td><td></td><td rowspan=1 colspan=1>569733</td><td></td><td></td><td rowspan=1 colspan=1>530857</td><td></td><td rowspan=1 colspan=1>426209</td></tr><tr><td rowspan=1 colspan=1>5</td><td></td><td></td><td rowspan=1 colspan=1>569556</td><td></td><td></td><td rowspan=1 colspan=1>530686</td><td></td><td rowspan=1 colspan=1>426145</td></tr><tr><td rowspan=1 colspan=1>6</td><td></td><td></td><td rowspan=1 colspan=1>569285</td><td></td><td></td><td rowspan=1 colspan=1>530582</td><td></td><td rowspan=1 colspan=1>425953</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1>569299</td><td></td><td></td><td rowspan=1 colspan=1>530509</td><td></td><td rowspan=1 colspan=1>426035</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>569246</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>530524</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>426033</td></tr></table>

Table 3: Number of primitives for MPEG tracked and semitracked sequences.

## G Bitrate Redistribution

In Fig. 7, we find that the proposed GSCV exhibits faster PSNR decay since the third bitrate point. The reason may be that we use a large QP for color DC, while color DC serves as a cornerstone for visual quality. Therefore, we try the other QP setting (noted as C2, while the QP reported in Table 1 is referred to as C1): for R01 to R03, we use QP = 7 for color DC as suggested by [13]. Besides, color SH contributes to the majority of the overall bitstream as shown in Fig. 10. We use YUV420 to reduce the bits of chrominance from color SH, and use C1 and C2 to generate new RD curves. The results are shown in Fig. 12. We find that: 1) C2 yields better results than C1 at low to medium bitrates, indicating that more bitrates should be devoted to the color DC component; 2) using YUV420 for color SH demonstrates even better performance at low to medium bitrates, while the quality ceiling has been substantially decreased even though QP=0. It suggests that YUV444 should be used if high quality is desired.

![](images/afba0a62e840bb7efedbf1290191d341d602d1207b5aa32dd2af97a6f977776a.jpg)  
Figure 10: Bitrate allocation of GSCV.

## H Influence of Data shufling

In Section 1, we highlight that GSCodec Studio directly uses the Iframe PLAS index for the P-frame image, exploiting the assumption that the arrangement of the I- and P-frame primitives demonstrates explicit spatial correspondence. The MPEG tracked GS data provides this spatial correspondence naturally due to its generation process [24]; therefore, GSCodec Studio reports impressive compression results on the vanilla MPEG tracked dataset. We test the GSCodec Studio with and without data shufling on the MPEG tracked dataset. The results are shown in Fig. 13. We see that the GSCodec Studio reports obviously poorer performance when data shufling is applied. Specifically, increasing the bitrate by four times is required for achieving the same PSNR compared with the noshufle case. In order to ensure I- and P-frame alignment, GSCodec Studio uses primitive padding rather than pruning for the square 2D grid requirement of PLAS, considering that even a minor pruning diference of the I- and P-frames will result in incorrect overall index alignment and the invalidation of index inheritance. The results of GSCodec Studio with pruning are also illustrated in Fig. 13. The obtained observation is close to the shufle case. Based on the finding from LightGaussian [14] that GS might have many redundant primitives due to the incomplete densification mechanism, the downstream codec should be compatible with primitive pruning for more flexible rate control, which also highlights the significance and value of the proposed GSCV. Note that if tracked information is available for GSCV (i.e., disable Inter-PLAS), GSCV can still report better performance than GSCodec Studio because GSCV has better compression eficiency with the same PLAS image input as shown in Appendix J.

![](images/13cf049844debbbc88d615792530eca6c030532555c1ed317f20143f75dafad1.jpg)  
Figure 11: Coding time of GSCV.

![](images/cc2e7ffafa6a6664d86a0acbd94c99265adf04d1e51d75d93c60c8c58896f378.jpg)  
Figure 12: Additional RD curves of GSCV on tracked “bartender”.

## I Efectiveness of Anchor-based PLAS refinement

As the prerequisite for realizing eficient inter-frame PLAS image prediction, anchor-based PLAS refinement is important for aligning I- and P-frames. However, diferent from the vanilla PLAS, which adopts a progressive coarse-to-fine blurred image generation strategy to generate smooth intra-frame images, using an I-frame PLAS image as a target to refine the P-frame is more challenging and results in extended computation time requirements. For example, the vanilla PLAS on “bartender”, which has around half a million primitives, requires around 30s GPU time, while anchor-based PLAS refinement requires around 70s GPU time. Inspired by the PLAS loss and block size curves shown in Fig. 4, we find that adapting the block size decay factor � can result in an improved balance between

P-frame image compression efectiveness and generation speed. The default � is 0.95 in the vanilla PLAS. We try $\tau = 0 . 8 , 0 . 7 , 0 . 6$ and report the RD curve and PLAS time in Fig. 14. When reducing � from 0.95 to 0.6, we can save around half the GPU time with a minor performance decrease. Therefore, we suggest choosing a proper � for P-frame image generation if computation complexity is emphasized in certain use cases. Besides, we also explore the influence of using a dynamic � according to the GS sequence length in Appendix M.

## J All-Intra Performance

The proposed GSCV is designed specifically for the GS sequence, while remaining compatible with single-frame compression. We use all-intra mode for GSCV to test the first frame of MPEG “bartender”, and compare it with GSCodec Studio (all-intra mode), GPCC v1, HGSC [23], and FCGS [9]. We find that: 1) The FCGS reports obviously better performance than other methods. The reason is that FCGS is a learning-based method with complex context models. It uses a mask to divide the primitives into diferent groups and compress them in either the explicit or latent domain. Besides, the FCGS requires GPUs to realize data compression, which is expensive as reported in [40]. 2) HGSC also requires GPUs to do the primitive pruning based on rendering results as one step of data compression. 3) GSCV, GSCodec Studio, and GPCC v1 are based on CPU, which is lightweight. Although GSCV and GSCodec require PLAS that is based on GPUs, this process is generally not regarded as part of the compression, particularly taking into account that PLAS images may be produced in the course of GS generation or directly thereafter [32]. 4) GSCV reports better all-intra performance than GSCodec Studio, indicating that GSCV can still ofer better inter-prediction results if using the same PLAS videos as input.

## K Influence of PLAS feature channel

In Section 5, we use all the 59 GS features for Inter-PLAS image generation. The time of Inter-PLAS and compression ratio are highly related to the feature channels used in PLAS. To highlight this problem, three cases are proposed to generate PLAS images: 1) using all the features; 2) using coordinates and color DC, i.e., six channels; and 3) using all the attributes except color SH, i.e., fourteen channels. The results are shown in Fig. 15.

We see that employing more features leads to improved compression eficiency. However, more features result in more GPU time for Inter-PLAS. Using “bartender” as an example, applying Inter-PLAS to the I- and P-frames in the three cases considered takes 30.5s/73.4s, 7.9s/22.6s, 12.6s/32.0s. Therefore, besides using � to control the Inter-PLAS complexity as reported in Appendix I, controlling the feature channels used in PLAS also deserves more investigation.

![](images/8561e6cbf61e0afbcda49646a4b41cb172f3973f29623d0c9f25984f1bbdddbb.jpg)

![](images/def4e6dce9fb541452cc2c4edad28c3ad3e8121bfb2b0405e752c91a8fc8165e.jpg)

![](images/ad63becc31fcd51e96bf4fa5731ed89cd34af780dddba428ef99c479f1211393.jpg)

Figure 13: Influence of data shufle on GSCodec Studio on tracked dataset.  
![](images/d8bbfe0ce2d3f6cc077e1ca09f4073ade0f5039bdfe56caf3ff0058d8f6603f4.jpg)  
(a) RD curve  
t]

![](images/404b54ce646951200d0c1a554452637a9dfca68b77073198f5425a20f9230350.jpg)  
(b) I-frame

![](images/3d15a8246172145c0f40af00de291f8129d3bfdab9693e1dc998e2734ac02908.jpg)  
(c) � = 0.95, PG = 8.5 dB

![](images/3fdc017bf8f3de08dc5dbf53bf45fce4246169561b4f7f8dc06417515d9984d9.jpg)  
(d) � = 0.8, PG = 8.11 dB

![](images/dc0ceafd6a17463c8a8e41f040e3ba021e70644a7229b4fc8e5953721343b375.jpg)  
(e) � = 0.7, PG = 7.93 dB

![](images/67963fc2ce3624062960b3b0efb5441a2637dc59371525b23316f20d8ad4a5f9.jpg)  
(f) � = 0.6, PG = 7.57 dB  
Figure 14: Ablation study of anchor-based PLAS refinement.

## L Influence of Compression Sequencing

In Section 5, we choose the default “encoder\_randomaccess" configuration to compress a GoP of PLAS images, in which the first frame is regarded as the I-frame, all the other frames are B-frames (i.e., bidirectionally predicted-coded frame) with compression sequencing as 0-4-2-1-3-6-5-7. We explore the influence of coding sequencing on final results in this section. We choose HEVC “encoder\_low\_delay” configuration and set frames 1 to 7 as P-frames (predictive-coded frame), the coding sequencing is 0-1-2-3-4-5-6- 7. The results are shown in Fig. 17. We see that “RandomAccess” coding mode is slightly better than “low\_delay” mode with 3.63% RD gains, indicating that using B-frames in compression can improve overall performance. However, introducing P-frames coding, which does not depend on frames in the future moment, generally requires less bufer time, therefore is more friendly for real-time applications.

## M The Influence of Dynamic Decay Factor

We have discussed the influence of � in Appendix I, where we find larger � represents using more iterations for P-frame PLAS image generation, which generally produces a closer image compared with I-frame images, as well as longer computation time. Theoretically, � should be increased with the increase of the length of IPPP sequences. The reason is that with the increase of the sequence length, the diference between the P-frames and I-frames increases, which means the Inter-PLAS requires more iterations to refine P-frames. We perform a new ablation study here. For P-frames of “bartender”, we set: 1) D1: initial $\tau _ { 0 }$ as 0.6, and $\tau _ { t } = m i n ( \tau _ { 0 } \times 1 . 1 ^ { t - 1 } , 0 . 9 5 ) , t > = 1 _ { \cdot }$ in which �<sub>�</sub> gradually increases to 0.95; and 2) D1: initial $\tau _ { 0 }$ as 0.95, and $\tau _ { t } = m a x ( \tau _ { 0 } \times 0 . 9 ^ { t - 1 } , 0 . 6 ) , t > = 1$ , in which �<sub>�</sub> gradually decreases to 0.6. The result is shown in Fig. 18.

![](images/4f354090d2aeaec7affb49d19138d0c5aadfce224f6f59f490f7ee8abc0e9a5f.jpg)

![](images/2c91159a7f14099788a48a09a3988ee1020cfdfec71e9d26a418d4dd28db3084.jpg)

![](images/d1443d6591bd170672537ebec7e148439ffb88b35ca2975a3483de8eae894268.jpg)  
Figure 15: Influence of feature channels on Inter-PLAS

![](images/b658f89b63fe465e516c4e528ee5b6ccbea1eee618d10f7a5c286cd41d709c3e.jpg)  
Figure 16: RD curves of single-frame results.

We can see that using a fixed � of 0.95 reports the best performance, while the case with an increased � introduces around 2.12% gain compared with the case that uses a decreased �. This corroborates our conclusion that for the GS sequences, the P-frames distant from the I-frames exhibit greater diferences and therefore should be assigned a larger �. Considering D1 and D2 require a close overall inter-PLAS image generation time, D1 is a better choice given a relatively long GS sequence.

## N Residual Analysis

Based on the results in Fig. 7, we find the results of HM18.0 and VTM23.11 close to each other, which is in conflict with the results on natural images. Therefore, we illustrate the residual map and the corresponding statistics histogram between frames 0 and 1 of “bartender” color DC and scaling as examples. For a fair comparison, we use the rendered “bartender” with view 9 as the representation of nature images.

![](images/031e7042e2b0a66c4ed570797b91688beee7b12b4cd582d1a94b1291bd838fd4.jpg)

Figure 17: RD curves of diferent coding sequencing.  
![](images/8db498c646329c9777c3095008f706f3baf2decc303eb06531c61c46b49694ee.jpg)  
Figure 18: RD curves of dynamic �.

Based on the histogram, we further provide fitted Gaussian and Laplace distribution curves. The results are shown in Fig. 19. We

Rendered Nature Image — Residual Map (clipped ±0.3) PLAs Color DC Image — Residual Map (clipped ±0.5) PLAS Scale Image — Residual Map (clipped ±0.2

![](images/7c132909076440a501ee95e723d544c1af738eca98cf523d03f0222ce19de517.jpg)  
(a) Nature, Std=0.019

![](images/cb3edeaf45d6ab811f0c459d3a94e69f917c1120c47bb567e73661b34ab12051.jpg)  
(b) DC, Std=0.05

![](images/de13f6771d948c9e55cd20e31369a5047cad44b21e25ae580b8a919e5d16069b.jpg)  
(c) Scale, Std=0.03

![](images/e9c3c66f6158e2a0cfe418f52280d1e556dac47a86eb16cceb5f581c743a9517.jpg)  
(d) Entropy=3.91

![](images/05053f644eb1726fc52dd7085237181880a2c73529ccabbc3d91e82d415f4594.jpg)  
(e) Entropy=6.17

![](images/3c215b3a735f5361ace3e927878872f3f128965131adc916b0cc26bd470befb3.jpg)  
(f) Entropy=7.16  
Figure 19: Illustration of residual map and statistic histogram.

see that the natural image reports the smallest standard deviation (Std), and the corresponding histogram reveals smaller entropy. The larger Laplace scale parameter � also reflects that PLAS images demonstrate a more pronounced heavy-tail behavior, indicating more challenging compression conditions. Besides, natural images illustrate obvious motive texture, while PLAS images resemble residual signals that lack semantic content. Consequently, many advanced techniques used in video coding may not function efectively on such images, which can explain the reason that VTM23.11 reports close performance with HM18.0.

<table><tr><td rowspan=1 colspan=2>Component</td><td rowspan=1 colspan=5>QP</td></tr><tr><td rowspan=1 colspan=2>Color DC</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>47</td></tr><tr><td rowspan=3 colspan=1>Color SH</td><td rowspan=1 colspan=1>degree1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>degree2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>37</td></tr><tr><td rowspan=1 colspan=1>degree3</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>42</td></tr><tr><td rowspan=1 colspan=2>Opacity</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=2>Scaling</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>22</td></tr><tr><td rowspan=1 colspan=2>Rotation</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>22</td></tr></table>

Table 4: QP of diferent components

## O RD curve for diferent components

We illustrate the RD curves of diferent components in this section. First, we plot the RD curve for color DC, color SH, Opacity, Scaling, and Rotation of method in Fig. 9 as shown in Fig. 20. For certain regions of some features, a few vertically varying points appear because two adjacent bitrate points use the same QP value, while other attributes adopt diferent QPs.

To better visualize the influence of diferent components on quality, we perform new experiments: give a certain component, we fix the QP of all the other components as 0, and set five diferent QP for this certain component. The QP selection is shown in Table 4, and the corresponding RD curves are shown in Fig. 21.

We see that rendered quality is more sensitive to opacity, scaling, and rotation, considering smaller QPs of these components report close PSNR with larger QPs of color DC and SH. Therefore, we choose relatively smaller QP for opacity, scaling, and rotation in our main experiment results, i.e., Fig. 7.

## P Influence of video codec types

In Section 5, we use HM18.0 and VTM23.11 as representations to test the proposed GSCV. Considering both HM18.0 and VTM23.11 are reference software for academic study, we further use FFM-PEG libx265 and libx264 as codecs to test the performance of the proposed GSCV. The RD curves are shown in Fig. 22.

We see that: 1) the performance of FFMPEG libx265 and libx264 is close. Considering that the same phenomenon also occurs in HM18.0 and VTM23.11, in conjunction with our residual analysis in Section N, we think the main reason is that the characteristics of PLAS image are diferent from natural images, resulting in some advanced compression modules being inefective for PLAS image compression. 2) HM18.0 reports obviously better performance than FFMPEG libx265, while they both follow the same HEVC (H.265) standard: HM serves as the oficial reference model for algorithm verification, while libx265 is an optimized engineering realization designed for practical and eficient video compression. This inspires us to give the following conclusion: although some compression modules are improved from AVC to VVC, not all the improvements are useful for PLAS images. There are two analyses corresponding to these results:

![](images/b5d1f951dbee0fcbc17c9546d68e9328585550a9880cdae5bc8e05b9d5dfad14.jpg)

![](images/a8ddc937499e3c085c638177c4f2bb22e9164b89dbeff081d352e8f2c84bbb37.jpg)

![](images/84c7a2a7240b985673308a5818e19a798606d452239533af5fccbf31b61dafd6.jpg)

![](images/14ead9902f49d2de107554589c11abb11a8cd3f7726e24facfb0bee037d99893.jpg)

![](images/a645beaad19e60f93f1caddc7e9eab1f65a398b2ebb6c003690dfadf19a9e911.jpg)  
Figure 20: RD curves of diferent components.

![](images/665f01196e5aaf0f587662299057d6a2a5f6c94c748237caa3ca0f04a797b09d.jpg)

![](images/9af5cea8e67e97f37c5afff3e03952d9e0212cc705b396e12922cb4db084e580.jpg)

![](images/a2a547fc938390ec747608275e6ac69c198cacd3c4dfc2de591a9aec38ac2790.jpg)

Figure 21: RD curves of diferent components with smooth QP variation.
<table><tr><td rowspan=1 colspan=1>Codec</td><td rowspan=1 colspan=2>GSCV-HM18.0</td><td rowspan=1 colspan=2>GSCV-VTM23.11</td><td rowspan=1 colspan=2>GSCV-libx265</td><td rowspan=1 colspan=2>GSCV-libx264</td><td rowspan=1 colspan=2>GSCodec Studio</td><td rowspan=1 colspan=2>GPCC v1</td></tr><tr><td rowspan=1 colspan=1>Rate Point</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td><td rowspan=1 colspan=1>enc</td><td rowspan=1 colspan=1>dec</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1183.5</td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>11288.9</td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>4.6</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>8.9</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>304.1</td><td rowspan=1 colspan=1>86.0</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1012.5</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>11234.2</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>0.9</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>1.4</td><td rowspan=1 colspan=1>298.0</td><td rowspan=1 colspan=1>88.4</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>540.3</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>6385.6</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>291.7</td><td rowspan=1 colspan=1>86.7</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>353.4</td><td rowspan=1 colspan=1>1.6</td><td rowspan=1 colspan=1>3482.8</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>291.9</td><td rowspan=1 colspan=1>86.0</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>301.5</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>2391.8</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>293.2</td><td rowspan=1 colspan=1>85.4</td></tr></table>

Table 5: Encoding and decoding time of diferent codecs on one GoP (8 frames)

• First, the mode of intra prediction increases from 9 (AVC), 35 (HEVC), to 69 (VVC). However, considering the PLAS image does not have semantic texture due to the pixel-level permutation, the increase of intra prediction mode might not improve the compression ratio significantly.

![](images/ef24b366faecd065bacd83a33d7b3dd5f17d1acfcbb395a6b79f92504ce58b1d.jpg)  
Figure 22: RD curves of using FFMPEG codecs

• FFMPEG libx265 uses early skip/ termination for coding unit (CU) partition. Considering PLAS is a progressive method that generates local smoothness images, and the smoothness range and degree are not even for the whole PLAS maps, trying more CU partitions would help the codecs choose the best RD cost, which might be the main reason that libx265 reports obviously worse performance than HM18.0

Although FFMPEG codecs report inferior performance to HM and VTM, they demonstrate obviously higher coding and decoding speed, as shown in Table 5. We see that 1) video-based codecs report faster decoding speed than point cloud based codecs; 2) with the same codec type as libx265, the proposed GSCV reports faster encoding and decoding speed than GSCodec Studio, indicating the advantage of the proposed PLAS image generation method, as well as the framework ofthe whole pipeline. Therefore, for low delay use cases, we suggest using FFMPEG codecs on the GSCV framework.

## Q Comparison with A-3DGS based method

Diferent from the I-3DGS track, the A-3DGS track uses the multiview images as input and generates compact 3D GS data via longtime optimization on GPUs. Generally, A-3DGS presents a higher compression ratio than I-3DGS due to the end-to-end training with RD loss. In this section, we compare two I-3DGS methods, i.e., GSCV-HEVC and GSCodec Studio, with four A-3DGS methods: HAC [10] and CompGS [30] for the all-intra method, and 4DGS [49] and 4DGC [22] for the inter-based method. For a fair comparison, we use the multiview images as ground truth to test the quality score of GSCV-HEVC and GSCodec Studio, the results are shown in Fig. 23.

We see that 4DGC reports the best compression ratio, followed by HAC, CompGS, 4DGS, and two I-3DGS methods. However, we find A-3DGS methods have two weaknesses based on the RD curve: 1) regarding the quality of the input 3D GS samples as an upper bound, A-3DGS methods generally demonstrate lower quality and smaller bitrate range; 2) some A-3DGS methods fails to consistently produce a monotonic curve in which the quality increases with the bitrate, due to each rate point needs to be trained from scratch and the optimization process involves a certain degree of randomness.

## R Influence of using Sandwiched Network

Considering the characteristics of PLAS images are obviously dif ferent from natural images, which can limit the performance of canonical video codecs, we try a “Sandwich network” to improve the video codec eficiency [16]. Specifically, the Sandwich network consists of three parts: the preprocessing network (PreN), the JPEG proxy, and the postprocessing network (PostN). An end-to-end training is conducted based on an established dataset. The RD loss is used to train the PreN and PostN, in which the bitrate is approximated by the JPEG proxy. The trained PreN can convert the target image into a bottleneck image whose data distribution is more friendly for canonical image/video codecs. The practical compression and decompression are performed on the bottleneck image, and the PostN can recover the bottleneck image into the original version.

![](images/bde430fb678c8c76f5cfa68c503e1fa89edac939dbf8299e11c523e221544901.jpg)

![](images/4cb1761475cc3d44037f2db30671fe39bac7a5ecb034c9b1218b68ae5ac280b1.jpg)  
Figure 23: RD curves of A-3DGS methods.

Three prevalent datasets, i.e., Mip-NeRF360 [4], deep blending [20], and tanks&temples [26] were applied to generate 3D GS samples and corresponding PLAS images, and build the training dataset. Considering that PLAS images for diferent components exhibit diferent characteristics, and each type of GS attributes needs a separate Sandwich network, in this part, we only used color DC PLAS images as a representation. For PreN and PostN, we used a lightweight UNet consisting of a 4-block encoder and a 5-block decoder. Each encoder block contains two 3×3 Conv-ReLU layers followed by simple pixel-subsampling that halves the spatial resolution, with increasing channel sizes (32→64→128→256). The decoder mirrors the encoder structure, using two 3×3 Conv-ReLU layers per block, bilinear upsampling by a factor of two, and skip-connections that concatenate encoder features at matching scales. The decoder uses filter sizes 512→256→128→64→32. A final 3×3 convolution layer projects the result to the desired number of output channels.

For testing, we use “cinema” as a representation; besides color DC, we keep other attributes lossless. Considering that the Sandwich network using JPEG proxy to train the PreN and PostN, we adopt JPEG [41] as the codec in the proposed GSCV to better reflect the efectiveness of the Sandwich network. The RD curves obtained were given in Fig. 25. We observed that: 1) the use of FFMPEG libx265 reports better performance than JPEG in the low bitrate region, while JPEG is superior in the high bitrate range; 2) using the Sandwich network followed by the JPEG ofers the best performance for the whole bitrate range, indicating the efectiveness of the Sandwich network. Future work is to design Sandwich networks for diferent GS attributes.

![](images/fc2af7190eb6a85317ce754ac6d1e2b2d7a92d1381161b7bf82220fa689f0d01.jpg)

![](images/14ca03b47aa932ae75e97d967f5af6354fc6ac8fe9aeab4ba3a9cc8b53a0ca86.jpg)  
Figure 24: RD curves on “dance” and “basketball”.

![](images/2ca64688f9d60e56388dd58dbdeae3c5ab7d68b068b1eff0b0116fcf2b780498.jpg)

Figure 25: RD curves of using the Sandwich network.  
![](images/ff9e1b05dbd179f6d0d440208981f33f79534ceab3c7885dba5a526a1253244e.jpg)  
dance

![](images/f5b67752b89e383f5df84d1aa325d5107fe2b48ed0b433c5c0422ca4ba017a25.jpg)  
basketball  
Figure 26: Snapshot of “dance” and “basketball”

## S Results on more sequences

In this section, we construct two additional GS sequences for testing our method. Specifically, we use “dance\_dunhuang\_pair” (dance) from PKU-DyMVHumans [55] and “basketball” from AVS-VRU [2], as shown in Fig. 26.

These two sequences only provide multiview videos; therefore, we use the first 8 frames from the video to generate colmap file, as well as 3D GS samples. We follow the instructions of the MPEG document [24] to merge the per-frame sparse point clouds into a unified one, followed by a uniform downsampling as the input for 3D GS generation. Then, we train a canonical 3D GS model with ground truth images from all viewpoints and time points. Finally, for each time point, we use the canonical 3D GS model as input to perform attribute fine-tuning with ground truth images.

![](images/9213867ac3371afceaf4f3faa0b18260bf764f84d6cfb62c7762d8bf92173843.jpg)

![](images/387ab45afb28b5d5328437520c35f24d02f97d05ab8cca7ec8c2a9d92a22d6dd.jpg)

(a) Ground Truth  
![](images/e28305cd339a713eceb6317f9e8b0ce0bbbcee95957a140c5c39abfeae100ead.jpg)  
(c) R3, 6.4 MB, 35.4 PSNR

(b) R5, 17.6 MB, 44.9 PSNR  
![](images/c1a0777fb44a3612554937514acf861f455edbcca638c2ee5b195024e5b8a863.jpg)  
(d) R1, 4.0 MB, 28.7 PSNR  
Figure 27: Snapshot of MPEG “bartender”.

We test the proposed GSCV-HEVC, GSCodec Studio, and GPCC v1 on the generated sequences, the RD curve are shown in Fig. 24. We see that the proposed GSCV demonstrates obvious better performance than GSCodec Studio and GPCC v1, which reveals the superiority of the Inter-PLAS, as well as the compression pipeline.

## T Sample Visualization

We give the ground truth, rates 5, 3, and 1 of three MPEG tracked GS sequences in Figs. 27 to 29.

![](images/590c5d3ae189b8a162ed869e60a0e12e6240c776d17be38abcca9b25329489e6.jpg)  
(a) Ground Truth

![](images/6cd7ca7ca6a4569e41729153fb9f580aa32c13c36b42dc34cd555ad5bb1f4740.jpg)  
(b) R5, 13.9 MB, 40.4 PSNR

![](images/368d6acce8653d0cfd28ae483a6fd8a736e610fd001c91096e00e6df29679227.jpg)  
(c) R3, 5.5 MB, 31.2 PSNR

![](images/54004c5577daaa7df2b9a1d2f84df076ff0337823147212156154a0178d540e2.jpg)  
(d) R1, 3.9 MB, 25.7 PSNR

Figure 28: Snapshot of MPEG “breakfast”.  
![](images/d5bc7f7814267763a53370306eff75117f53d720765ad065fd88c726d2fc4a5e.jpg)  
(a) Ground Truth

![](images/6ea7f3e3c2b50887e0f1ae47c08016487ba9a59008f1ed6f0cf3022ebcc87816.jpg)  
(b) R5, 11.9 MB, 41.3 PSNR

![](images/9464ffec3ea9580e6d0461ff35883b5836f239e23cac7a3ffe2f8fceab8065a6.jpg)  
(c) R3, 4.3 MB, 33.3 PSNR

![](images/17a9cacc923d89bbc5bc20c0d1e8c4683f9e707fcb3da738c1d01b806e50c9a9.jpg)  
(d) R1, 2.8 MB, 26.6 PSNR  
Figure 29: Snapshot of MPEG “cinema”.