# Attention-Guided Sequence Feature Fusion and Mixed-Space Metric Learning for Cross-View Image Sequence Geo-Localization

Nan Wu, Chunfang Yang, Jiangshan Li, Xiangyang Luo

Abstract—Cross-view image sequence geo-localization aims to determine the geographic location of a sequence of ground images by matching it against geo-tagged satellite imagery. Although sequential observations provide complementary scene information, substantial overlap between adjacent images introduces feature redundancy, while conventional fusion strategies may weaken spatial relationships essential for cross-view matching. Therefore, we propose a cross-view image sequence geo-localization method via Attention-guided Sequence fusion and Metric Learning (ASML). Specifically, a Sequential Content Fusion (SCF) module models cross-image feature correlations to suppress redundant content and adaptively preserve geographically informative nonoverlapping regions. Next, a Spatial Feature Aggregation (SFA) module is used to capture and aggregate the spatial and semantic information of local features, so as to maintain the spatial layout of the image sequence. Finally, a Geo-InfoNCE loss function is proposed, which comprehensively considers the feature distance and geographic distance for mixed-space metric learning, thus improving the ability of the model to distinguish hard negative samples. A new cross-view image sequence dataset CrossRoad-66K is established to provide more complex scenarios. Extensive experiments and visualizations on the SeqGeo and CrossRoad-66K datasets demonstrate that compared to state-ofthe-art (SOTA) methods, the proposed method ASML achieves an improvement of 3.37% and 2.10% at R@1, respectively.

Index Terms—Image Sequence, Geo-localization, Cross-view, Image Retrieval.

## I. INTRODUCTION

nique that retrieves the most similar image from a database of geo-tagged aerial images (usually satellite images) for a ground image sequence, thus estimating the geographic location where the ground image was captured [1]–[5]. Previous studies focused on matching a single ground image with aerial images in a reference dataset [6]–[8]. In recent years, with the rise of short-form video social media platforms, numerous ground image sequences have been produced quickly. These ground image sequences contain richer semantic information than a single image. Therefore, it is significantly meaningful to geo-localize the ground image sequence in many applications, such as autonomous driving [9], augmented reality [10], drone navigation [11].

![](images/c4f6c1e8851a3c2e5ef7ffe37f6ef88454bd16af2db829c00b306212c956a160.jpg)  
(a) Average ratio of overlapping area under different sequence lengths

![](images/8c253b59122413cdcfe0aa8fef8d03bd578bede003a52bca0cc6d37de5f695dd.jpg)  
(b) Visualization of the overlapping areas in ground images  
Fig. 1. Quantitative and qualitative analysis of overlapping areas in crossview image sequences. (a) The average ratio of overlapping areas among ground image sequences with different sequence lengths on the SeqGeo and CrossRoad-66K datasets. (b) Visualization of overlapping regions between adjacent ground images. This figure provides a dataset-level analysis of sequence redundancy rather than a performance comparison between different methods.

Cross-view image sequence geo-localization is an extension of cross-view image geo-localization. Existing cross-view image geo-localization methods [12]–[17] aim to minimize the distance between the features of the matched ground image and satellite image while maximizing the distance between the features of the unmatched ones. Cross-view image sequence geo-localization methods [18]–[21] essentially follow the same principles. The difference lies on extracting the feature representation of the image sequence by fusing the local features of multiple ground images in the sequence, which is significant for subsequent feature matching.

The early cross-view image sequence methods extract the feature of each ground image independently, and then concatenate these features along the sequence dimension [20]. Recently, researchers [21] propose to select the important components from the features of ground images by multihead attention mechanism. However, these methods do not consider the influence of overlapping areas in the image sequence, which may introduce redundant information and disrupt the spatial layout of objects in different images. Fig. 1 analyzes the inherent redundancy characteristics of image sequences in different datasets, which motivates the design of the proposed overlap-aware feature selection strategy. In other words, if the redundant content from overlapping areas are not effectively suppressed, it would weaken the discriminative ability of the feature representation and decrease the geo-localization accuracy.

Therefore, this work proposes a cross-view image sequence geo-localization method via attention-guided sequence feature fusion and metric learning (ASML). In the proposed method, the Sequential Content Fusion (SCF) module is designed to fuse geographic-related features from sequential ground images while eliminating redundant content through the attention-guided feature selection. The Geo-InfoNCE loss function is presented to perform mixed-space metric learning, which simultaneously combines feature space and geographic distance to guide model optimization. Extensive experiments on the SeqGeo and CrossRoad-66K datasets demonstrate that ASML significantly improves geo-localization accuracy and reduces computational costs compared to existing methods.

The main contributions can be summarized as follows.

• We propose a cross-view image sequence geo-localization method ASML, with a task-specific Sequential Content Fusion (SCF) module. Unlike conventional attentionbased feature fusion strategies, SCF transforms crossimage correlation into an overlap-aware selection mechanism, enabling redundancy suppression and spatially ordered feature reconstruction.

• A Geo-InfoNCE loss function is designed to integrate feature space and geographic space. It enables more accurate quantification of the prediction error by a mixedspace alignment mechanism and guides the optimization of the model.

• We establish a new cross-view image sequence dataset CrossRoad-66K to provide a complementary benchmark with urban-dominant scenarios. Compared with existing datasets, CrossRoad-66K contains more complex scenarios, enabling a more comprehensive evaluation of crossview image sequence geo-localization methods.

• Extensive experiments demonstrate that, compared to existing methods, the proposed ASML achieves improvements of 3.37% and 2.10% at R@1 on the SeqGeo and CrossRoad-66K datasets, respectively, while effectively controlling computational costs.

The remainder of this manuscript is organized as follows. Section II reviews the related work on cross-view image geolocalization and image sequence geo-localization. Section III discusses the key challenge of cross-view image sequence geolocalization that this work aims to address. Section IV provides a detailed introduction to the proposed method ASML and its core modules. Section V presents extensive experiments and analyses of ASML, and compares its performance with existing methods. Finally, Section VI summarizes the main contributions of this work.

## II. RELATED WORK

In this section, we briefly review existing works on cross-view image geo-localization and image sequence geolocalization.

## A. Cross-View Image Geo-Localization

The key points of cross-view image geo-localization lie in the feature aggregation approach and the loss function.

Due to the drastic difference in spatial layouts between ground and satellite views, researchers developed a series of feature aggregation methods to enhance the representational capability of ground image and satellite image features. For example, Hu et al. [22] proposed CVM-Net, which added a NetVLAD [23] layer on the VGG16 network to embed ground images and satellite images in a unified feature space. Shi et al. [24] added two consecutive fully connected layers after the backbone network as the Spatial Aware Feature Aggregation Module (SAFA), which encodes the scenery information in ground and satellite images to improve the robustness of feature representation. To reduce the view difference between ground images and satellite images, Ye et al. [25] transformed street-view panoramas into BEV (Bird’s Eye View) image and combine panorama-to-satellite and BEV-to-satellite retrieval branches to improve geo-localization accuracy. Ouyang et al. [15] proposed CLGT, which integrates causal learning and BEV-based geometric topology fusion to improve the robustness and discriminability of cross-view image features.

The loss function also plays a significant role in enhancing the model’s generalization ability and robustness. In the early years, Hu et al. [22] proposed a weighted soft-margin triplet loss, which adds a scaling factor to amplify the loss based on the soft-margin triplet loss. To further improve the model’s ability to distinguish hard negative samples, Cai et al. [26] and Guo et al. [27] introduced a hard sample mining strategy that differentiates the simple and hard samples within a batch and re-weights them.

Existing methods have achieved impressive performance, some of them have been extended to cross-view image sequence geo-localization. However, because these methods independently extract features from each ground image in an image sequence, it is difficult for them to capture the spatial layout of the objects in the realistic scenario, leading to poor geo-localization performance.

## B. Image Sequence Geo-Localization

Image sequence geo-localization can be divided into sameview and cross-view image sequence geo-localization according to the view of the reference image. In the early years, same-view image sequence geo-localization is the mainstream methods. For example, Gonzalo et al. [28] segmented street view videos into query frames, then matched the handcrafted features of query frames and reference ground images to estimate the trajectory of the query video. Regmi et al. [29] designed a temporal feature extraction network to extract the features of query frames and reference ground images and proposed a deep learning approach to correct the offsets of estimated GPS positions.

![](images/e4ac0bce993bb520fdeb0cb4ab6887af0f05d088118e6d6c1875e3abb10250cb.jpg)  
(a)

![](images/80c9aaf2d89cb069485f76ebe528f933af20150b40e10bffac18a5fde113cce4.jpg)  
(b)

![](images/401fd5e73cbea19ab2f8c54805a41f54dd4f9987fb03d6f61f0e8b332c9fc1c6.jpg)  
(e)

![](images/575f7a51e25b3f7c62393f4f645cdc7871aa63576a5175f03f448f4f839a1b0a.jpg)  
(c)

![](images/71dd8fbf558624d3e80fd0a0a5a864559e75c0e1130575cba647973586dd6100.jpg)  
(f)

![](images/40d40eaf52d93a80486bb7322630db61e92779ddc1940a950e59c3bcb0a6e382.jpg)  
(d)

![](images/fc0f4de7fca910871e9b9d1d51db14da9a8ef8f869a12ceb46133dd52060163b.jpg)  
(g)  
Fig. 2. A pair of samples in SeqGeo dataset. (a) shows the satellite image. (b)-(d) show the ground images captured along a straight road, whose locations are marked by green points in (a). (e)-(g) show the ground images captured around a corner, whose locations are marked by red points in (a).

In recent years, with the improvement of satellite image resolution, cross-view image sequence geo-localization has attracted increasing attention. Vyas et al. [18] matched videoframe features with aerial image patches to localize the geographic region covered by a video. Shi et al. [30] proposed CVLNet, which projects sequential ground-view features into the overhead view and aggregates them into a global sequence representation while estimating relative displacement for cross-view alignment. Pillai et al. [19] further adopted a coarse-to-fine strategy, first retrieving a large aerial region using the global video feature and then matching individual frames with local aerial patches for trajectory prediction.

For sequence-level representation learning, Zhang et al. [20] randomly masked ground-image features and concatenated the retained features to suppress redundant content. Yuan et al. [21] employed multi-head attention between adjacent ground images to aggregate temporal features. More recently, Wu et al. [31] introduced the Set-CVGL task and the FlexGeo framework to adaptively fuse features from unordered groundview image sets for cross-view geo-localization.

However, existing methods mainly focus on sequence-level feature aggregation or coarse image-level selection, while the redundant overlapping content between adjacent ground images is not explicitly modeled.

## III. PROBLEM STATEMENT

Given a ground image sequence $S ^ { \mathrm { g r d } } = \{ I _ { i } ^ { \mathrm { g r d } } \} _ { i = 1 } ^ { K }$ , where K is the sequence length, cross-view image sequence geolocalization aims to retrieve the satellite image covering the geographic area captured by the sequence from a reference gallery $\{ I _ { j } ^ { \mathrm { s a t } } \} _ { j = 1 } ^ { R }$ . In this task, multiple ground images are first encoded and compressed into a sequence-level descriptor:

$$
d ^ { \mathrm { g r d } } = \mathcal { A } \left( f ( I _ { 1 } ^ { \mathrm { g r d } } ) , f ( I _ { 2 } ^ { \mathrm { g r d } } ) , \cdots , f ( I _ { K } ^ { \mathrm { g r d } } ) \right) ,\tag{1}
$$

where $f ( \cdot )$ denotes the image feature extractor and $\boldsymbol { \mathcal { A } } ( \cdot )$ denotes the sequence-level feature fusion function. The geo-

TABLE I  
COMPARATIVE ANALYSIS OF GROUND IMAGE SEQUENCE FEATURE WITH SATELLITE IMAGE FEATURE USING STRUCTURAL SIMILARITY AND KL DIVERGENCE. ↓ INDICATES LOWER VALUES WITH GREATER SIMILARITY AND ↑ INDICATES HIGHER VALUES WITH GREATER SIMILARITY.
<table><tr><td rowspan="2">Max Masks</td><td colspan="2">Structural Similarity</td><td colspan="2">KL Divergence</td></tr><tr><td>L2 Distance↓</td><td>Cosine Similarity↑</td><td> $\overline { { \mathbf { G } 2 \mathbf { S } ( 1 0 ^ { - 5 } ) } } \downarrow$ </td><td> $\overline { { \mathbf { S } 2 \mathbf { G } ( 1 0 ^ { - 5 } ) \downarrow } }$ </td></tr><tr><td>5</td><td>0.1033</td><td>0.0465</td><td>1.7548</td><td>1.7551</td></tr><tr><td>3</td><td>0.1019</td><td>0.0480</td><td>1.6820</td><td>1.6826</td></tr><tr><td>1</td><td>0.1021</td><td>0.0478</td><td>1.6935</td><td>1.6945</td></tr><tr><td>0</td><td>0.1024</td><td>0.0475</td><td>1.7083</td><td>1.7088</td></tr></table>

localization result is then obtained by matching $d ^ { \mathrm { g r d } }$ with the descriptor of each reference satellite image $d _ { j } ^ { \mathrm { s a t } }$

$$
b = \underset { j \in 1 , \cdots , R } { \arg \operatorname* { m i n } } D \left( d ^ { \mathrm { g r d } } , d _ { j } ^ { \mathrm { s a t } } \right) ,\tag{2}
$$

where $D ( \cdot , \cdot )$ denotes the feature distance, and b represents the index of the matched satellite image.

As shown in Fig. 1b, adjacent ground images usually contain large overlapping regions, where scenery objects such as buildings, roadsides, trees, and traffic facilities repeatedly appear. To evaluate the influence of such redundancy, we randomly select 100 sample pairs from SeqGeo and compare the similarity between the fused ground image sequence feature and the corresponding satellite feature under different random masking settings. The similarity is measured by L2 distance, cosine similarity, G2S KL divergence, and S2G KL divergence. In Table I, Max Masks denotes the maximum number of ground images randomly removed from an input sequence in SeqGeo [20]. Larger values of Max Masks indicate that more images may be discarded before feature concatenation, thus reducing part of the redundant content.

As shown in Table I, these results suggest that both excessive redundancy and overly aggressive random masking may weaken cross-view feature similarity. Therefore, coarse imagelevel masking is insufficient, demanding local overlap-aware feature selection.

Spatial layout preservation is another key issue. For straightroad sequences, adjacent ground images often maintain horizontal continuity, so direct feature concatenation may partially preserve object layout. However, for corner-road sequences, abrupt trajectory changes break this continuity, and direct concatenation or random masking may distort the spatial relationship among geographic objects, especially when the ground observations correspond to a turning road structure in the satellite view.

![](images/fde8e87d2208d0cc976ff229af5b37a9a50feb198d2acc525ca6b2f9356defbc.jpg)  
Fig. 3. Overview architecture of ASML. Specifically, the Sequential Content Fusion Module (SCF) integrates the geographical semantics among the ground image sequence, and reduces the number of tokens in the subsequent processing. Spatial Feature Aggregation Module (SFA) mainly contains a Transformer Encoder, which is used to aggregate feature descriptors. The Geo-InfoNCE Loss is used to learn discriminative features in feature space and geographic space.

Therefore, the main challenge is not merely aggregating multiple ground image features, but constructing an overlapaware sequence representation that suppresses redundant overlapping contents while preserving the spatial organization of non-overlapping geographic-related features for cross-view matching.

## IV. METHODOLOGY

Inspired by the analysis in Section III, the most intuitive idea is to eliminate redundant information while maintaining the spatial layout information of the objects within the ground images. Therefore, we propose a cross-view image sequence geo-localization method via attention-guided sequence feature fusion and metric learning (ASML). In this section, we introduce the architecture and each component of ASML in detail.

## A. Method Overview: ASML

As shown in Fig. 3, the proposed method ASML mainly consists of the following components: (1) a ConvNeXt backbone [32] for feature extraction, (2) a Sequential Content Fusion (SCF) module for global feature integration, (3) a Spatial Feature Aggregation (SFA) module for feature aggregation, and (4) a Geo-InfoNCE loss function for mixed-space metric learning.

These components constitute two pipelines: the ground image pipeline and the satellite image pipeline. The ground image pipeline takes the ground image sequence $S ^ { \mathrm { g r d } }$ in an image sequence as input. First, we feed them into the ConvNeXt backbone to extract local semantic features. The extracted feature maps $F _ { i } ^ { \mathrm { g r d } }$ are then inputted into the SCF module to obtain the global ground feature map $F _ { \mathrm { g l o b a l } } ^ { \mathrm { g r d } }$ . Next, the global feature map is aggregated into a ground image descriptor $d ^ { \mathrm { g r d } }$ through the SFA module. The satellite image pipeline takes the satellite image $I ^ { \mathrm { s a t } }$ as input. Similarly, the ConvNeXt backbone extracts the local feature map $F ^ { \mathrm { s a t } }$ from the satellite image, which is then aggregated into the satellite image descriptor $d ^ { \mathrm { s a t } }$ using the SFA module. Finally, the Geo-InfoNCE loss function is used to measure the difference between the ground image feature descriptor $d ^ { \mathrm { g r d } }$ and the satellite image feature descriptor $d ^ { \mathrm { s a t } }$ , allowing for optimization of model parameters.

Notably, ASML constructs the image sequence descriptor through global feature reconstruction and descriptor aggregation, while the Transformer-based SFA module further aggregates the cross-view feature map into a compact descriptor for matching. Therefore, SFA serves as a descriptor aggregation component rather than a generic sequence fusion module.

## B. Sequential Content Fusion Module (SCF)

Unlike direct concatenation or generic Transformer-based fusion, the proposed Sequential Content Fusion (SCF) module performs task-specific feature selection before aggregation, as shown in Fig. 4. It calculates the local correlations between adjacent ground images to identify the overlapping regions, then preserves the non-overlapping regions that provide richer geographic information. The selected features are further reconstructed into a spatially ordered global feature map.

![](images/fbedf9e3282618dc31d7dd2e62eb5d9d78ec7e6d4bcab739fdfc04b13b01ef5d.jpg)  
Fig. 4. The operational workflow of the Sequential Content Fusion Module (SCF). First, a similarity scores map $S _ { i }$ is calculated between the adjacent local features. Next, an attention scores map $\bar { A _ { i + 1 } }$ is generated from similarity scores map, which is then translated to an attention mask $M _ { i + 1 }$ . Finally, the attention mask is utilized to select the features in non-overlapping areas and eliminate redundant features.

Different from conventional cross-attention modules that aim to enhance feature interaction, SCF uses correlation scores as an indicator of spatial redundancy rather than feature importance. Specifically, SCF implements the sequence-level feature fusion through three steps: local similarity calculation, adaptive feature selection, and global feature map reconstruction.

1) Step 1: Local Similarity Calculation: The inputs of the SCF module are $K$ sequential local feature maps $F _ { i } ^ { \mathrm { g r d } } \in$ $\mathbb { R } ^ { C _ { 1 } \times H \times W }$ , where $C _ { 1 } , H$ , and W represent the channel dimension, height, and width of the local ground feature map, respectively. Through reshaping operations along the spatial dimension, the three-dimensional feature maps are flattened to the two-dimensional version $\tilde { F } _ { \dot { \boldsymbol { i } } } ^ { \mathrm { g r d } } \in \mathbb { R } ^ { C _ { 1 } \times ( H \times W ) }$ <sup>)</sup>. Then, the local similarity matrix $S _ { i } \in \mathbb { R } ^ { ( \mathring { H } \times W ) \times ( H \times W ) }$ is calculated between the adjacent feature maps.

$$
S _ { i } = ( \tilde { F } _ { i } ^ { \mathrm { g r d } } ) ^ { \mathrm { T } } \times \tilde { F } _ { i + 1 } ^ { \mathrm { g r d } } ,\tag{3}
$$

where $S _ { i } ( m , n ) , ~ m , n ~ \in ~ [ 1 , H \times W ]$ represents the dot product between the m-th local feature of $\hat { F } _ { i } ^ { \mathrm { g r d } }$ and the $n \mathrm { - }$ th local feature of $\tilde { F } _ { i + 1 } ^ { \mathrm { g r d } }$ . Low similarity scores indicate weak attention correlation between the paired areas, which geometrically corresponds to the non-overlapping areas between adjacent ground images. Therefore, the objective of SCF is opposite to conventional attention weighting strategies: instead of emphasizing highly correlated regions, it suppresses redundant correlated regions and retains complementary geographic information.

2) Step 2: Adaptive Feature Selection: To identify potentially non-overlapping areas, we first generate an attention mask $M _ { i + 1 } \in \{ 0 , 1 \} ^ { H \times W }$ through:

$$
M _ { i + 1 } ( k , l ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ A _ { i + 1 } ( k , l ) < \tau _ { i } ( k ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{4}
$$

$$
A _ { i + 1 } ( k , l ) = \operatorname* { m a x } _ { 1 \leq j \leq H \times W } S _ { i } \big ( j , ( k - 1 ) W + l \big ) ,\tag{5}
$$

$$
\tau _ { i } ( k ) = \mathrm { Q u a n t i l e } \left( A _ { i + 1 } ( k , : ) , \beta _ { i } \right) ,\tag{6}
$$

where $A _ { i + 1 }$ is the attention scores map of $F _ { i + 1 } ^ { \mathrm { g r d } }$ , and $( k -$ $1 ) W + l$ denotes the flattened spatial index corresponding to position $( k , l ) . \ \tau _ { i } ( k )$ is the row-wise quantile threshold of the attention scores, and $\beta _ { i } ~ \in ~ [ \beta _ { \operatorname* { m i n } } , \beta _ { \operatorname* { m a x } } ]$ denotes the adaptive feature selection ratio for the i-th pair of adjacent ground images. The selection ratio $\beta _ { i }$ is adaptively determined from the attention-score distribution. Accordingly, the lowestscoring features in each row are selected to suppress redundant content.

The generated attention mask $M _ { i + 1 }$ is a binary matrix, whose one-valued elements indicate regions with relatively low cross-image correlation, which are potentially non-overlapping areas in the feature map $F _ { i + 1 } ^ { \mathrm { g r d } }$ . Then, the attention mask $M _ { i + 1 } \in \{ 0 , 1 \} ^ { H \times W }$ is broadcast to $M _ { i + 1 } ^ { \prime } \in$ $\{ 0 , 1 \} ^ { C _ { 1 } \times H \times W }$ by replicating it $C _ { 1 }$ times along the channel dimension, which is used to extract the selected local features:

$$
\hat { F } _ { i + 1 } ^ { \mathrm { g r d } } = F _ { i + 1 } ^ { \mathrm { g r d } } \odot M _ { i + 1 } ^ { \prime } ,\tag{7}
$$

where $\odot$ denotes point-wise multiplication, and $\hat { F } _ { i + 1 } ^ { \mathrm { g r d } }$ is the masked feature map.

3) Step 3: Global Feature Map Reconstruction: First, we reshape $\hat { F } _ { i + 1 } ^ { \mathrm { g r d } }$ to select its geographic-related features and adjust its spatial dimension. Since the adaptive selection ratio $\beta _ { i }$ may vary across adjacent image pairs, a fixed output width

(a)

![](images/16aadeed6993d97b06084e47053d174dfedc516bd82c03c409833465c5086d80.jpg)

![](images/d67ba344074b69dc36bd247818738cedf3873297bdcd66e1daf3b32df342ec0a.jpg)  
Fig. 5. The main processing of the spatial feature aggregation module (SFA). (a) shows the workflow of SFA module. (b) shows the detailed structure of the Transformer Encoder in (a).

$W _ { \mathrm { m a x } } = \lceil \beta _ { \mathrm { m a x } } W \rceil$ is adopted to ensure consistent feature dimensions:

$$
F _ { i + 1 } ^ { s } = \mathrm { P a d d i n g } \left( \mathrm { R e s h a p e } \left( \hat { F } _ { i + 1 } ^ { \mathrm { g r d } } \right) , W _ { \mathrm { m a x } } \right) ,\tag{8}
$$

where the Reshape operation compacts the selected features in each row according to their original spatial order, and the Padding operation fills the remaining positions with zeros to maintain a fixed output width. Specifically, approximately $\lceil \beta _ { i } W \rceil$ features are retained per row, while the output feature map has a fixed dimension $F _ { i + 1 } ^ { s } ~ \in ~ \mathbb { R } ^ { C _ { 1 } \times H \times \bar { W } _ { \operatorname* { m a x } } }$ . This process preserves the relative spatial order of the selected features and constructs a compact representation for subsequent descriptor aggregation.

By repeating the above operations on subsequent images in the sequence, K − 1 feature maps $F _ { i + 1 } ^ { s } , \ i \in \ \{ 1 , \dots , K -$ 1}, are obtained. Finally, a global feature map $F _ { \mathrm { g l o b a l } } ^ { \mathrm { g r d } } \ \in$ $\bar { \mathbb { R } ^ { C _ { 1 } \times H \times \left[ W + ( K - 1 ) W _ { \operatorname* { m a x } } \right] } }$ is reconstructed as follows:

$$
\begin{array} { r } { F _ { \mathrm { g l o b a l } } ^ { \mathrm { g r d } } = F _ { 1 } ^ { \mathrm { g r d } } \Vert F _ { 2 } ^ { s } \Vert \cdot \cdot \cdot \Vert F _ { K } ^ { s } , } \end{array}\tag{9}
$$

where ∥ denotes width-wise concatenation.

In summary, the SCF module adaptively fuses sequential local feature into a global feature map through attentionguided spatial feature selection, thus preserving spatial layout and eliminating redundant content. Unlike generic feature fusion, SCF is specifically designed for the redundancy and spatial layout challenges in cross-view image sequence geolocalization. As no additional learnable parameters are introduced, it can also control computational cost.

## C. Spatial Feature Aggregation Module (SFA)

Feature aggregation modules [22]–[24] are commonly used to enhance spatial and semantic consistency between matched ground-satellite features. Following recent works [19], [33], [34], we adopt DeiT [35] as the Spatial Feature Aggregation Module (SFA), since its class and distillation tokens can aggregate local features and capture positional-semantic correlations to produce discriminative global descriptors.

The specific processing of the SFA module is illustrated in Fig. 5. The input feature map is first decomposed into N unitscale (1×1) patches. These patches are then fed into a linear projection layer to generate the corresponding patch tokens $t _ { i } \in \mathbb { R } ^ { C _ { 2 } } , i \in [ 1 , N ]$ , where $C _ { 2 }$ is the channel dimension of each token. Then, the distillation token $t _ { \mathrm { d i s } }$ and the class token $t _ { \mathrm { c l s } }$ are inserted into patch tokens, where $t _ { \mathrm { d i s } }$ serves as an auxiliary learnable aggregation token to provide complementary discriminative information and $t _ { \mathrm { c l s } }$ is used to collect global contextual information from all patch tokens. These tokens are augmented with a learnable positional encoding $E _ { \mathrm { p o s } } \in \mathbb { R } ^ { C _ { 2 } \times ( \mathbf { \check { N } } + 2 ) }$

$$
X _ { 0 } = [ t _ { \mathrm { d i s } } , t _ { \mathrm { c l s } } , t _ { 1 } , t _ { 2 } , \cdot \cdot \cdot , t _ { N } ] + E _ { \mathrm { p o s } } ,\tag{10}
$$

where $X _ { 0 }$ is the input of the Transformer Encoder. The positional encoding $E _ { \mathrm { p o s } }$ corresponds to the positions in the reconstructed global feature map, rather than the original coordinates of the unmasked feature map. Therefore, it can model the relative and global spatial order of the compact sequence-level representation. Finally, the global feature descriptor $d _ { \mathrm { g l o b a l } }$ is generated as follows:

$$
X _ { l } = T ( X _ { 0 } ) = [ d _ { \mathrm { d i s } } , d _ { \mathrm { c l s } } , d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { N } ] ,\tag{11}
$$

$$
d _ { \mathrm { g l o b a l } } = \mathrm { L P } ( d _ { \mathrm { d i s } } ) + \mathrm { L P } ( d _ { \mathrm { c l s } } ) ,\tag{12}
$$

where $T$ represents the l-layer Transformer Encoder, $X _ { l }$ is the output of the l-layer Transformer Encoder, LP represents linear projection, $d _ { i } ~ \in ~ \mathbb { R } ^ { C _ { 2 } } , i ~ \in ~ [ 1 , N ]$ are the output of patch tokens. Notably, when the input is the global ground feature map $F _ { \mathrm { g l o b a l } } ^ { \mathrm { g r d } } , d _ { \mathrm { g l o b a l } }$ corresponds to the ground descriptor $d ^ { \mathrm { g r d } }$ . When the input is the satellite feature map $F ^ { \mathrm { s a t } }$ $d _ { \mathrm { g l o b a l } }$ corresponds to the satellite descriptor $d ^ { \mathrm { s a t } }$ . Relying on the powerful multi-head attention mechanism of DeiT, the distillation token $d _ { \mathrm { d i s } }$ and the class token $d _ { \mathrm { c l s } }$ can effectively integrate the global contexts of each token, thus keeping the spatial layout information.

In summary, the SFA module is adopted as a descriptor aggregation component to encode the reconstructed global feature map. It can capture the spatial layout of scenery objects in ground images and aggregate semantically discriminative global feature descriptors, thus significantly enhancing crossview image geo-localization robustness.

## D. Geo-InfoNCE Loss Function

Cross-view image geo-localization aims to pull matched ground-satellite pairs closer in the feature space while pushing unmatched pairs apart. The InfoNCE loss [36] has been widely adopted in contrastive representation learning and cross-view geo-localization tasks [37], [38]. It can be formulated as:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { \exp { ( ( 1 - D _ { i , i } ) / \tau ) } } { \sum _ { j = 1 } ^ { B } \exp { ( ( 1 - D _ { i , j } ) / \tau ) } } ,\tag{13}
$$

$$
D _ { i , j } = 1 / 2 \left\| \mathbf { q } _ { i } - \mathbf { k } _ { j } \right\| _ { 2 } = 1 - \mathbf { q } _ { i } \cdot \mathbf { k } _ { j } ,\tag{14}
$$

(43.4811,-72.3928)  
![](images/e5ae34d056c7b086efe82435c0d4b6ada0164c4bc806df87593458a19d41e5fd.jpg)

(43.4811,-72.3927)  
![](images/d3556e59b856967a5f82568d58d551d5e5f4c97cb71449d78ec1c3139c9dc071.jpg)

![](images/082167c91491fcf511e72110891a7aea8ba95254a32a761ecc36ebcaa5086ea8.jpg)

(43.4849,-72.4001)  
![](images/6145f1b4153c217847e396c1d8d8a729aadf094e8ab218fe49c4276d844c44ae.jpg)

![](images/022b91a3047f162e72993ab2c9568ac8dddfee95fd85e6fc568e7335dd6fef02.jpg)

(43.4849,-72.4001)  
(a)  
(b)  
![](images/aa95991c7de3deb6f7d9a6524450981b7c506d1152d64efc5c7fb7fd55f5c30b.jpg)  
(c)  
Fig. 6. Visualization of some hard negative samples in the SeqGeo dataset, which are easily confused. (a) shows one image within the ground image sequence. (b) shows the matched satellite image. (c) shows the unmatched satellite image. The left ground image and the right satellite image are a pair of difficult samples. Each image is labeled with the GPS coordinates of the capture location.

where $\| \mathbf { q } _ { i } - \mathbf { k } _ { j } \| _ { 2 }$ denotes the L2 distance between the i-th query image’s feature vector $\mathbf { q } _ { i }$ and the j-th reference image’s feature vector $\mathbf { k } _ { j } , ~ i , j ~ \in ~ [ 1 , B ]$ ], B is the batch size, and τ is a learnable temperature parameter. Although InfoNCE effectively optimizes feature-space similarity, it does not explicitly consider the geographic distance between samples. Therefore, visually similar but geographically distant satellite images may be treated as hard negative samples and interfere with model optimization, as shown in Fig. 6. Existing works have introduced geographic constraints into the loss function [39]–[41], but they are usually formulated as an additional matching constraint and may still provide insufficient penalties for visually similar but geographically distant hard negatives.

To address this issue, we propose a Geo-InfoNCE loss function $\mathcal { L } _ { \mathrm { G e o } }$ , which integrates feature distance and geographic distance into a unified mixed-space metric:

$$
\mathcal { L } _ { \mathrm { G e o } } = - \log \frac { \exp \left( ( 1 - \overline { { D } } _ { i , i } ) / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( ( 1 - \overline { { D } } _ { i , j } ) / \tau \right) } ,\tag{15}
$$

$$
\overline { { D } } _ { i , j } = D _ { i , j } - \lambda _ { g } \cdot \mathrm { N o r m } ( G _ { i , j } ) ,\tag{16}
$$

$$
G _ { i , j } = { \sqrt { \left( \mathrm { l o n } _ { i } - \mathrm { l o n } _ { j } \right) ^ { 2 } + \left( \mathrm { l a t } _ { i } - \mathrm { l a t } _ { j } \right) ^ { 2 } } } ,\tag{17}
$$

where $\overline { { D } } _ { i , j }$ denotes the mixed distance between the i-th query feature and the j-th reference feature. $D _ { i , j }$ represents the feature distance between the corresponding feature representations. $G _ { i , j }$ denotes the GPS distance between the center points of the satellite images corresponding to the i-th and j-th samples within a batch. $\lambda _ { g }$ is a scaling coefficient that controls the contribution of geographic distance in the mixedspace metric. Norm(·) denotes a batch-wise normalization operation applied to the geographic distance matrix. Therefore, Geo-InfoNCE serves as a task-oriented mixed-space metric for contrastive optimization, where $D _ { i , j }$ measures the discrepancy between the learned visual representations, while $G _ { i , j }$ provides complementary geographic information to suppress visually similar but geographically inconsistent negative pairs.

It should be noted that D, G, and D are all $B \times B$ matrices, whose elements are scalar terms rather than high-dimensional vectors. Empirically, the feature distance D and the geographic distance G in city-scale datasets exhibit comparable numerical magnitudes and remain within the same order of magnitude. Therefore, this lightweight batch-wise normalization is sufficient to stabilize their numerical scales without introducing additional learnable parameters or a more complex mapping function. The scaling coefficient $\lambda _ { g }$ is further introduced to adjust the relative contribution of geographic distance, and its effectiveness is verified by an ablation study. Since the samples are captured from the same city, the distance deviation caused by the Earth’s curvature has negligible influence on the loss optimization. Therefore, we calculate the GPS distance using the L2 distance through latitude and longitude.

Generally, Geo-InfoNCE loss $\mathcal { L } _ { \mathrm { G e o } }$ integrates feature space and geographic space through a mixed-space alignment mechanism, as demonstrated in Fig. 7. It can prevent the model from the interference caused by hard negative samples, thus better guiding the optimization of the model parameters.

## V. EXPERIMENT

## A. Cross-View Image Sequence Datasets

The experiments were performed on SeqGeo and CrossRoad-66K datasets, which are introduced as follows.

1) SeqGeo Dataset: SeqGeo dataset [20] is a large-scale cross-view image sequence dataset collected in Vermont, USA, comprising 38,863 ground image sequences and 38,863 satellite images. This dataset was split into a training set with 31,091 image pairs and a testing set with 7,772 image pairs. The ground images in this dataset were collected on approximately 500 km of urban and suburban roads, with around 70% from suburban areas and 30% from urban areas. Each ground image sequence contains 7 images, with 8 meters between the capture coordinates of adjacent images, and the FoV (field-of-view) of each ground image is approximately 120<sup>◦</sup>.

2) CrossRoad-66K Dataset: We establish a new crossview image sequence dataset CrossRoad-66K to evaluate the performance of the proposed method in challenging scenarios. It is an urban-dominant dataset that contains more complex scenarios such as road structures, denser buildings, and diverse variations. The specific establishment process of the CrossRoad-66K dataset is as follows.

Ground Image Sampling: Real driving video data was collected on approximately 400 km of roads in Ziyang City, Sichuan Province, China, combined with the corresponding GPS coordinates. Then these driving videos were segmented into frames as ground images, with a size of 2560 × 1180 and a FoV of approximately $1 4 0 ^ { \circ }$ . The total number of ground images is 66,384, and the distance between the capture coordinates of adjacent images is 8 meters. Similar to the SeqGeo dataset, every 6 successive ground images are grouped into a ground image sequence. Due to the consecutive sampling strategy of the satellite images in the CrossRoad-66K dataset, there are overlapping areas between adjacent satellite images, which results in some ground images being covered by two or three satellite images. Therefore, the total number of ground images is not six times of the satellite images.

![](images/525408ba4cb5a9bd11f74fb8251b7a30fd54be0a5bbf469737a4e7777f9e248e.jpg)  
Fig. 7. The mixed distance metric optimization strategy for the samples within a batch. The samples matrix represents the relationship between the samples within a batch, where the elements on the diagonal are positive samples, while the others are negative samples. M refers to the feature distance matrix D minus the GPS distance matrix G.

TABLE II  
COMPARISON BETWEEN THE PROPOSED CROSSROAD-66K DATASET AND EXISTING CROSS-VIEW IMAGE DATASETS.
<table><tr><td>Datasets</td><td>CVACT [42]</td><td>CVUSA [43]</td><td>VIGOR [44]</td><td>SeqGeo [20]</td><td>CrossRoad-66K</td></tr><tr><td>Satellite Images</td><td>44,416</td><td>44,416</td><td>90,618</td><td>38,863</td><td>21,320</td></tr><tr><td>Ground Images</td><td>44,416</td><td>44,416</td><td>105,214</td><td>118,549</td><td>66,384</td></tr><tr><td>Ground Images per Satellite Image</td><td>1</td><td>1</td><td>5</td><td>7</td><td>6</td></tr><tr><td>Seamless Sampling</td><td>X</td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Sequential Ground Images</td><td>X</td><td>×</td><td>×</td><td>√</td><td>√</td></tr><tr><td>Full Panorama</td><td>√</td><td>√</td><td>√</td><td>X</td><td>×</td></tr><tr><td>Ground Image Center Orientation</td><td>North</td><td>North</td><td>North</td><td>Random</td><td>Random</td></tr><tr><td>Coverage</td><td>Urban, Suburban</td><td>Urban, Suburban</td><td>Suburban</td><td>Urban, Suburban</td><td>Urban, Suburban</td></tr></table>

HYPERPARAMETER TUNING FOR THE QUANTILE THRESHOLD β.  
TABLE III
<table><tr><td rowspan="3"> $\beta$ </td><td colspan="3">SeqGeo</td><td colspan="3">CrossRoad-66K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@1%</td><td>R@1</td><td>R@5</td><td>R@1%</td></tr><tr><td>8.3%</td><td>15.89%</td><td>34.61%</td><td>74.15%</td><td>4.57%</td><td>13.02%</td><td>38.30%</td></tr><tr><td>16.7%</td><td>21.53%</td><td>41.46%</td><td>79.04%</td><td>5.25%</td><td>15.75%</td><td>42.82%</td></tr><tr><td>25%</td><td>21.34%</td><td>41.69%</td><td>78.44%</td><td>4.99%</td><td>14.81%</td><td>41.76%</td></tr><tr><td>33.3%</td><td>19.97%</td><td>38.45%</td><td>76.37%</td><td>4.61%</td><td>14.24%</td><td>41.38%</td></tr><tr><td>Adap.</td><td>23.53%</td><td>44.30%</td><td>80.77%</td><td>6.20%</td><td>16.83%</td><td>43.94%</td></tr></table>

Satellite Image Sampling: Since each ground image sequence contains 6 images, the average GPS coordinate of all ground images in a ground image sequence was calculated as the central coordinate. The satellite image at the central coordinate of each ground image sequence was then downloaded from Bing Maps. In order to cover all the capture coordinates of the images in each ground image sequence, the satellite images are 300 × 300 pixels in size and have a spatial resolution of $0 . 2 9 8 ~ \mathrm { m / p i x e l }$

The CrossRoad-66K dataset was split into a training set containing 17,056 image pairs and a testing set containing 4,264 image pairs. The training and testing sets were collected from different regions. The differences between the CrossRoad-66K dataset and existing cross-view image datasets are shown in Table II.

## B. Implementation Details and Evaluation Metrics

Implementation Details: ASML was implemented in Py-Torch and deployed on two NVIDIA RTX 3090 GPUs (24GB). ConvNeXtV2-F pre-trained on ImageNet-1k was utilized as the backbone. DeiT-S pre-trained on ImageNet-21k was utilized as the SFA module, which was the same as TransGeo [33]. The number of layers l in DeiT-S was set to 12 while the patch size was set to 1. Each experiment was trained for 200 epochs with a batch size of 64 using the AdamW optimizer with an initial learning rate of 0.0001, and the weight decay coefficient was set to 0.03. The ground image size and the satellite image size were set to 192 × 384 and $3 2 0 \times 3 2 0$ respectively. The scaling coefficient $\lambda _ { g }$ of Geo-InfoNCE Loss is set to 0.75. To ensure the fairness of experiments, all the experimental results used for comparison are obtained from their papers or evaluated using their public code.

The feature selection ratio $\beta _ { i }$ in SCF controls the proportion of low-correlation features retained from each ground image. A small $\beta _ { i }$ may discard informative geographic features, whereas a large $\beta _ { i }$ may introduce redundant content. To accommodate varying overlap characteristics across image sequences, we adaptively determine $\beta _ { i }$ from the attentionscore distribution, with $\beta _ { \mathrm { m i n } } = 8 . 3 \%$ and $\beta _ { \mathrm { m a x } } = 3 3 . 3 \%$ . We further compare the adaptive strategy (Adap.) with different fixed selection ratios, as reported in Table III. The adaptive strategy consistently outperforms all evaluated fixed ratios across the retrieval metrics on both SeqGeo and CrossRoad-66K, demonstrating the effectiveness of dynamically adjusting the feature selection budget according to cross-image correlations.

TABLE IV  
COMPARISON WITH EXISTING METHODS. BOLDFACED TEXTS DENOTE THE BEST RESULTS AND UNDERLINED TEXTS DENOTE THE SUBOPTIMAL RESULTS.
<table><tr><td rowspan="2">Methods</td><td colspan="4">SeqGeo</td><td colspan="4">CrossRoad-66K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1%</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1%</td></tr><tr><td>SeqGeo [20]</td><td>2.07%</td><td>8.12%</td><td>13.16%</td><td>40.10%</td><td>1.86%</td><td>7.69%</td><td>13.56%</td><td>35.65%</td></tr><tr><td>GeoDTR+ [45]</td><td>2.52%</td><td>7.29%</td><td>11.69%</td><td>36.85%</td><td>1.29%</td><td>3.89%</td><td>6.32%</td><td>17.93%</td></tr><tr><td>GeoDTR [46]</td><td>3.20%</td><td>9.03%</td><td>14.08%</td><td>39.98%</td><td>3.15%</td><td>9.14%</td><td>14.18%</td><td>27.33%</td></tr><tr><td>GAReT [19]</td><td>3.34%</td><td>11.19%</td><td>17.18%</td><td>44.39%</td><td>4.10%</td><td>14.24%</td><td>21.46%</td><td>40.48%</td></tr><tr><td>L2LTR [47]</td><td>3.78%</td><td>12.00%</td><td>18.21%</td><td>47.59%</td><td>2.53%</td><td>9.71%</td><td>15.74%</td><td>23.15%</td></tr><tr><td>TransGeo [33]</td><td>9.38%</td><td>22.43%</td><td>29.81%</td><td>57.41%</td><td>2.37%</td><td>9.85%</td><td>15.62%</td><td>34.47%</td></tr><tr><td>Sample4Geo [37]</td><td>20.16%</td><td>39.63%</td><td>49.38%</td><td>75.24%</td><td>2.35%</td><td>7.34%</td><td>11.37%</td><td>23.57%</td></tr><tr><td>FlexGeo [31]</td><td>3.51%</td><td>14.22%</td><td>20.77%</td><td>48.53%</td><td>3.82%</td><td>14.17%</td><td>21.15%</td><td>41.41%</td></tr><tr><td>ASML (Ours)</td><td>23.53%</td><td>44.30%</td><td>53.23%</td><td>80.77%</td><td>6.20%</td><td>16.83%</td><td>24.40%</td><td>43.94%</td></tr></table>

![](images/c3c6a10486503dcd16a80dd7cb720f94d10ad45971b21c6738afece35f91314e.jpg)  
Fig. 8. Visualization comparison of attention maps between the baseline method and ASML. (a) shows the satellite images, (b) presents the attention maps generated by the baseline method [33], and (c) shows the attention maps produced by the proposed ASML. Compared with the baseline, ASML generates more concentrated responses on geographically discriminative regions while suppressing redundant regions, demonstrating its effectiveness in selecting informative features for cross-view matching.

Evaluation Metrics. We adopt the recall rate at top-k (R@k) and the recall rate at 1% (R@1%) to measure the retrieval accuracy. Given a query ground image sequence, R@k and R@1% represent the probability of the correct satellite image ranking within the top-k and top-1% retrieval results, respectively. In the experiments, we evaluate the performance of all methods using the following four evaluation metrics: R@1, R@5, R@10, and R@1%.

## C. Comparing ASML with State-of-the-art Methods

1) Quantitative Performance Comparisons: The performance of the proposed ASML and several advanced methods was tested on the SeqGeo and CrossRoad-66K datasets. For single-image methods [33], [37], [45]–[47] , we adapted them to the sequence setting by concatenating each ground image feature. For sequence-based methods, we followed their original sequence processing strategy. Table IV shows the R@1, R@5, R@10, and R@1% of these methods on the two datasets. On the SeqGeo dataset, ASML outperforms the suboptimal method Sample4Geo [37] with 3.37%, 4.67%, 3.85%, and 5.53% at R@1-R@1%, respectively. On the more challenging CrossRoad-66K dataset, ASML outperforms the suboptimal methods with 2.10%, 2.59%, 2.94%, and 2.53% at R@1-R@1%. It demonstrates that explicit geographic feature selection and redundant information removal are important for image sequence geo-localization. It can also be observed that there is a significant gap in the recall rates across the two datasets, indicating that the complex scenario of the CrossRoad-66K dataset is a real challenge for geo-localization.

2) Attention Map Analysis: As shown in Fig. 8, the baseline method tends to assign attention responses to broad visual regions, including redundant areas that are less beneficial for cross-view localization. In contrast, the proposed ASML produces more focused attention distributions on geographically informative structures, such as distinctive building layouts and road configurations. This difference originates from the Sequential Content Fusion (SCF) module, which explicitly models the correlation between adjacent ground images and suppresses redundant overlapping information before feature aggregation.

![](images/72910a47c931ed1670061e6bf1c54a76ca46a4b021400d10dff3219d01943d77.jpg)  
(a) Ground Images

![](images/9535495fb6a5b77a0d10bbc1a30a74e6bf2ed0a6a2306b8506e141e80f00ff8a.jpg)  
Truth  
(b) Ground-

![](images/1d319be76b4f93ae80edc2c709339e5ff2a45df6e90a50d4002018d607d3c39c.jpg)  
(c) Top-3 Results from ASML

![](images/68c70c981457cdd3f0fab1accd7f02016d82b097f37923717db4264fd40acaa7.jpg)  
(d) Top-3 Results from Sample4Geo [37]

![](images/73c2dce8d5ebdac0e8079d2068594f00969831caac7eedba1b171c620effbe0c.jpg)  
(e) Top-3 Results from GAReT [19]  
Fig. 9. Qualitative comparison of ASML and other methods. (a) shows the first images of the query ground image sequences. (b) shows the ground-truth results. (c) shows the Top-3 satellite images returned from ASML. (d) shows the Top-3 satellite images returned from Sample4Geo. (e) shows the Top-3 satellite images returned from GAReT. The first and second rows show the geo-localization results of dense-building areas, the third and fourth rows show the results of sparse-building areas, and the last row shows the results of overpass-intersection areas. The satellite images marked with green boxes denote the ground-truth results, which provide a common reference for all compared methods. The images marked with red boxes represent incorrect retrieval results. The first and fourth rows are samples from the CrossRoad-66K dataset, while the rest are samples from the SeqGeo.

3) Geo-Localization Results Visualization: The geolocalization results obtained from ASML and other methods are visualized in Fig. 9. It shows several geo-localization results in three typical scenarios: dense-building areas, sparsebuilding areas, and overpass-intersection areas.

Geo-Localization Results in Dense-Building Areas: The first and second rows of Fig. 9 show the geo-localization results in dense-building areas. For the given ground images, ASML can match the target satellite image accurately, while Sample4Geo fails to do so. This is because the images captured in dense-building areas contain more scenery objects, whose spatial layout is relatively complex. The proposed ASML not only eliminates redundant content, but also keeps spatial layout information. In contrast, when Sample4Geo is used for image sequence geo-localization, the redundant content among the ground images cannot be removed. Therefore, it will cause repeated appearance of certain scenery features and the loss of spatial layout information, thus adversely affecting geo-localization performance.

Geo-Localization Results in Sparse-Building Areas: The third and fourth rows of Fig. 9 show the geo-localization results in sparse-building areas. For these given images, ASML also achieves better geo-localization performance than Sample4Geo. This is because the images captured in sparsebuilding areas contain fewer scenery objects, whose spatial layout is relatively simple, resulting in more hard negative samples in sparse-building areas. Therefore, the Geo-InfoNCE loss function can guide ASML to better distinguish these hard negative samples during training. Consequently, ASML exhibits great geo-localization performance even in sparsebuilding areas.

Geo-Localization Results in Overpass-Intersection Areas: The last row of Fig. 9 shows the geo-localization result in overpass-intersection area. For the given ground image, neither ASML nor Sample4Geo is able to geo-localize the correct satellite image. Occlusions at overpasses and intersections can cause certain scene elements to be visible in ground images but absent from satellite imagery. It would introduce interference to the geo-localization task.

## D. Ablation Studies

In order to evaluate the effectiveness of each component in ASML, we performed extensive ablation studies. Moreover, considering the different numbers of ground images within a sequence in real-world scenarios, the recall rates of ASML on diverse ground image sequence lengths K are shown in Table VII.

1) Ablation Study on Sequential Content Fusion Module (SCF): In order to evaluate the effectiveness of our SCF module, we removed the SCF module and directly concatenated the local feature maps of the images in the sequence. As shown in Table V, the version integrated with the SCF module achieves gains of 19.58% on the SeqGeo dataset and 3.96% on the CrossRoad-66K dataset at R@1, respectively. The difference in performance gain between the two datasets may be related to their different scene distributions and visual characteristics. Compared with SeqGeo, CrossRoad-66K exhibits more complex scenes and more diverse ground-satellite appearance variations, which may affect the relative gain of the SCF module. This is consistent with the geo-localization results in Fig. 9.

TABLE V  
ABLATION STUDY OF EACH COMPONENT ON SEQGEO AND CROSSROAD-66K DATASETS.
<table><tr><td colspan="3">Ablations</td><td colspan="4">SeqGeo</td><td colspan="5">CrossRoad-66K</td></tr><tr><td>Fusion</td><td>Aggregation</td><td>Position Encoding</td><td>Loss Function</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1%</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1%</td></tr><tr><td>Concat</td><td>SFA</td><td>Compact</td><td>Geo-InfoNCE</td><td>3.95%</td><td>12.46%</td><td>19.29%</td><td>50.86%</td><td>2.24%</td><td>7.27%</td><td>11.94%</td><td>29.35%</td></tr><tr><td>Random Mask</td><td>SFA</td><td>Sparse</td><td>Geo-InfoNCE</td><td>20.61%</td><td>40.33%</td><td>49.21%</td><td>76.52%</td><td>5.01%</td><td>14.28%</td><td>21.12%</td><td>39.42%</td></tr><tr><td>SCF</td><td>GAP</td><td>Compact</td><td>Geo-InfoNCE</td><td>3.55%</td><td>12.18%</td><td>18.54%</td><td>50.08%</td><td>3.11%</td><td>10.90%</td><td>16.97%</td><td>38.89%</td></tr><tr><td>SCF</td><td>ViT-S</td><td>Compact</td><td>Geo-InfoNCE</td><td>12.24%</td><td>29.08%</td><td>38.79%</td><td>72.30%</td><td>3.18%</td><td>11.13%</td><td>16.81%</td><td>37.52%</td></tr><tr><td>SCF</td><td>SFA</td><td>Compact</td><td>Soft-Margin Triplet</td><td>13.58%</td><td>29.80%</td><td>38.17%</td><td>70.37%</td><td>1.69%</td><td>5.56%</td><td>9.79%</td><td>25.40%</td></tr><tr><td>SCF</td><td>SFA</td><td>Compact</td><td>InfoNCE</td><td>17.70%</td><td>35.76%</td><td>44.32%</td><td>73.39%</td><td>2.99%</td><td>10.29%</td><td>16.43%</td><td>34.68%</td></tr><tr><td>SCF</td><td>SFA</td><td>Sparse</td><td>Geo-InfoNCE</td><td>20.89%</td><td>41.30%</td><td>50.56%</td><td>78.49%</td><td>3.48%</td><td>10.09%</td><td>14.54%</td><td>31.89%</td></tr><tr><td>SCF</td><td>SFA</td><td>Compact</td><td>Geo-InfoNCE</td><td>23.53%</td><td>44.30%</td><td>53.23%</td><td>80.77%</td><td>6.20%</td><td>16.83%</td><td>24.40%</td><td>43.94%</td></tr></table>

TABLE VI

QUANTITATIVE COMPARISON OF DIFFERENT FEATURE-SELECTION STRATEGIES ON SEQGEO AND CROSSROAD-66K.
<table><tr><td>Dataset</td><td>Selection Strategy</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1%</td></tr><tr><td rowspan="3">SeqGeo</td><td>Random Mask</td><td>20.61%</td><td>40.33%</td><td>49.21%</td><td>76.52%</td></tr><tr><td>High-Similarity SCF</td><td>17.07%</td><td>35.85%</td><td>44.87%</td><td>72.77%</td></tr><tr><td>SCF</td><td>23.53%</td><td>44.30%</td><td>53.23%</td><td>80.77%</td></tr><tr><td rowspan="3">CrossRoad-66K</td><td>Random Mask</td><td>5.01%</td><td>14.28%</td><td>21.12%</td><td>39.42%</td></tr><tr><td>High-Similarity SCF</td><td>4.55%</td><td>14.99%</td><td>22.75%</td><td>41.41%</td></tr><tr><td>SCF</td><td>6.20%</td><td>16.83%</td><td>24.40%</td><td>43.94%</td></tr></table>

Fig. 10 visualizes representative feature-selection masks generated by SCF on the SeqGeo dataset. As illustrated in the figure, SCF tends to retain regions with relatively low cross-image correlation, which frequently correspond to complementary content that is less repeatedly observed in adjacent ground images. We note that such low-similarity regions are treated as a heuristic indicator of potentially complementary information rather than an exact estimate of geometric nonoverlap.

To further verify whether the proposed feature-selection strategy contributes to geo-localization performance, we conduct a controlled comparison among SCF, random mask, and high-similarity SCF. All three variants retain the same number of feature patches and use identical subsequent feature reconstruction, aggregation, training, and evaluation settings, with only the feature-selection strategy being changed.

As shown in Table VI, compared with the version equipped with random mask, SCF improves R@1 by 2.92% and 1.19% on SeqGeo and CrossRoad-66K, respectively, indicating that the performance gain cannot be attributed merely to discarding a subset of sequence features. Moreover, compared with highsimilarity SCF, SCF improves R@1 by 6.46% and 1.65%, respectively. It demonstrates that suppressing highly correlated content and retaining lower-similarity, potentially complementary features, is more beneficial for sequence geo-localization. Therefore, the qualitative visualization and quantitative ablation jointly demonstrate the task-level effectiveness of the proposed feature selection strategy.

2) Ablation Study on Spatial Feature Aggregation Module (SFA): In order to evaluate the effectiveness of the SFA module, we first performed experiments by replacing it with a global average pooling layer (GAP) followed by a fully connected classifier. Then, we conducted experiments by replacing it with a ViT-S that does not contain the distillation token. As shown in Table V, compared to the version integrated with GAP, ASML achieves gains of 19.98% on the SeqGeo dataset and 3.09% on the CrossRoad-66K dataset at R@1. Compared to the version integrated with ViT-S, ASML achieves gains of 11.29% on the SeqGeo dataset and 3.02% on CrossRoad-66K dataset at R@1. These substantial improvements are mainly derived from that the SFA module enables adaptive cross-region information interaction among the global ground feature map, thus obtaining more discriminative feature descriptors. We also implemented experiments on the version equipped with sparse position encoding, i.e., the global feature map is encoded based on the order of the initial local feature maps, preserving their original positional relationships. It can be observed that the version with sparse position encoding performs worse than the compact position encoding. This is because the sparse position encoding introduces noisy or inconsistent location information into the attention aggregation process and weakens the discriminative ability of the final descriptor.

![](images/4f0adfb26b955fc37d4816e4de8aa703b90f856a44b6aee2db307fcdb6ee460b.jpg)  
Fig. 10. Visualization of the attention masks on SeqGeo dataset. (a) shows the ground images. (b) shows the corresponding attention masks.

TABLE VII  
ABLATION STUDY ON DIFFERENT NUMBERS K OF GROUND IMAGES IN A SEQUENCE.
<table><tr><td rowspan="2">K</td><td colspan="3">SeqGeo</td><td colspan="3">CrossRoad-66K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@1%</td><td>R@1</td><td>R@5</td><td>R@1%</td></tr><tr><td>2</td><td>12.45%</td><td>30.42%</td><td>69.45%</td><td>3.41%</td><td>10.73%</td><td>33.12%</td></tr><tr><td>3</td><td>15.86%</td><td>34.52%</td><td>74.07%</td><td>5.26%</td><td>14.61%</td><td>38.47%</td></tr><tr><td>4</td><td>19.11%</td><td>37.55%</td><td>76.29%</td><td>6.20%</td><td>16.83%</td><td>43.94%</td></tr><tr><td>5</td><td>23.53%</td><td>44.30%</td><td>80.77%</td><td>4.21%</td><td>12.48%</td><td>38.94%</td></tr><tr><td>6</td><td>20.49%</td><td>39.99%</td><td>77.62%</td><td>3.67%</td><td>11.23%</td><td>35.95%</td></tr></table>

3) Ablation Study on Geo-InfoNCE Loss: In order to evaluate the effectiveness of Geo-InfoNCE loss, we compared it with the soft-margin triplet loss [22] and the original InfoNCE loss, which are widely used in cross-view image geolocalization tasks. Quantitative comparisons are shown in Table V. Compared to the version trained with soft-margin triplet loss, ASML achieves gains of 9.95% on the SeqGeo dataset and 4.51% on the CrossRoad-66K dataset at R@1. Compared to the version trained with the original InfoNCE loss, ASML achieves gains of 5.83% on the SeqGeo dataset and 3.21% on the CrossRoad-66K dataset at R@1, respectively. The results indicate that the Geo-InfoNCE loss is superior to the other two losses, thus demonstrating the feasibility of the mixed-space alignment mechanism.

TABLE VIII  
COMPARISON WITH EXISTING METHODS ON PARAMETERS, COMPUTATIONAL COST, AND R@1 (ON CROSSROAD-66K DATASET).
<table><tr><td>Methods</td><td>Architecture</td><td>Parameters</td><td>GFLOPs</td><td>R@1</td></tr><tr><td>L2LTR [47]</td><td>CNN+Transformer</td><td>195.9M</td><td>44.06</td><td>2.53%</td></tr><tr><td>TransGeo [33]</td><td>Transformer</td><td>44.9M</td><td>11.34</td><td>2.37%</td></tr><tr><td>Sample4Geo [37]</td><td>Transformer</td><td>87.6M</td><td>20.05</td><td>2.35%</td></tr><tr><td>SeqGeo [20]</td><td>CNN+Transformer</td><td>47.02M</td><td>10.30</td><td>1.86%</td></tr><tr><td>GeoDTR [46]</td><td>CNN+Transformer</td><td>48.5M</td><td>38.89</td><td>3.15%</td></tr><tr><td>FlexGeo [31]</td><td>Transformer</td><td>87.6M</td><td>140.14</td><td>3.82%</td></tr><tr><td>ASML(Ours)</td><td>CNN+Transformer</td><td>54.2M</td><td>10.22</td><td>6.20%</td></tr></table>

TABLE IX

SENSITIVITY ANALYSIS OF VARYING SCALING COEFFICIENT $\lambda _ { g } .$
<table><tr><td rowspan="2"> $\lambda _ { g }$ </td><td colspan="3">SeqGeo</td><td colspan="3">CVZY</td></tr><tr><td>R@1</td><td>R@5</td><td>R@1%</td><td>R@1</td><td>R@5</td><td>R@1%</td></tr><tr><td>0.25</td><td>18.11%</td><td>36.39%</td><td>73.85%</td><td>4.50%</td><td>13.61%</td><td>40.86</td></tr><tr><td>0.5</td><td>20.00%</td><td>39.60%</td><td>77.90%</td><td>5.47%</td><td>15.81%</td><td>44.65%</td></tr><tr><td>0.75</td><td>23.53%</td><td>44.30%</td><td>80.77%</td><td>6.20%</td><td>16.83%</td><td>43.94%</td></tr><tr><td>1</td><td>23.32%</td><td>44.04%</td><td>80.48%</td><td>6.11%</td><td>16.67%</td><td>43.92%</td></tr><tr><td>1.25</td><td>20.17%</td><td>39.47%</td><td>78.39%</td><td>5.75%</td><td>17.14%</td><td>44.27%</td></tr></table>

We further investigate the influence of the geographic term in Geo-InfoNCE. Since the feature distance term $D _ { i , j }$ and the geographic term $G _ { i , j }$ originate from different metric spaces, the coefficient $\lambda _ { g }$ explicitly controls the contribution of geographic term to the mixed-space metric. A small $\lambda _ { g }$ weakens the geographic constraint and makes the objective closer to conventional feature-based contrastive learning, while an excessively large $\lambda _ { g }$ may overemphasize geographic relationships at the expense of visual discrimination. To quantitatively evaluate this trade-off, we conducted a sensitivity analysis on $\lambda _ { g } ,$ as shown in Table IX. These results indicate that the contribution of geographic information needs to be appropriately balanced with visual feature discrimination rather than simply strengthened. We select $\lambda _ { g } { = } 0 . 7 5$ because it yields the highest R@1 on both datasets and achieves a favorable overall balance across the evaluated retrieval metrics.

4) Ablation Study on the Sequence Length K: We performed experiments to test the impact of sequence length K on geo-localization accuracy. Table VII demonstrates the quantitative results on the SeqGeo and CrossRoad-66K datasets. It can be observed that ASML achieves the best geo-localization performance at $K = 5$ on the SeqGeo dataset and K = 4 on the CrossRoad-66K dataset. Fig. 11 reveals that the last two images capture many objects (e.g., vehicles, and buildings), which are not captured in the corresponding satellite images.

![](images/b8e4033d0958c5dda7a93bcd8b1d0e6041194c537003300f60634bfa1785edde.jpg)  
Fig. 11. Visualization of the last image in the ground image sequence and the satellite image. The red box in the left ground image represents irrelevant content between the ground image and the satellite image, while the points in the right satellite image represent the capture positions of each image in the ground image sequence.

TABLE X  
COMPARISON OF CORNER-ROAD AND STRAIGHT-ROAD SCENARIOS UNDERVARYING TRAJECTORY-CHANGE THRESHOLDSτ<sub>θ</sub> .
<table><tr><td>Dataset</td><td>Scenario</td><td>Thresholdτθ/Numbers</td><td>R@1</td><td>R@5</td><td>R@1%</td></tr><tr><td rowspan="6">SeqGeo</td><td rowspan="4">Corner</td><td>15° (264)</td><td>27.27%</td><td>46.21%</td><td>84.09%</td></tr><tr><td>20° (106)</td><td>17.93%</td><td>35.85%</td><td>84.91%</td></tr><tr><td>25°(64)</td><td>14.06%</td><td>31.25%</td><td>89.06%</td></tr><tr><td>30°(41)</td><td>17.07%</td><td>34.15%</td><td>90.24%</td></tr><tr><td rowspan="4">Straight</td><td>15° (7,508)</td><td>23.22%</td><td>43.33%</td><td>80.16%</td></tr><tr><td>20°(7,666)</td><td>23.42%</td><td>43.53%</td><td>80.22%</td></tr><tr><td>25° (7,708)</td><td>23.43%</td><td>43.65%</td><td>80.20%</td></tr><tr><td>30° (7,731)</td><td>23.39%</td><td>43.47%</td><td>80.24%</td></tr><tr><td rowspan="7">CrossRoad-66K</td><td rowspan="4">Corner</td><td>15° (479)</td><td>9.40%</td><td>25.47%</td><td>57.83%</td></tr><tr><td>20°(342)</td><td>7.90%</td><td>23.39%</td><td>56.43%</td></tr><tr><td>25°(232)</td><td>6.90%</td><td>23.28%</td><td>56.03%</td></tr><tr><td>30° (155)</td><td>6.45%</td><td>22.58%</td><td>56.13%</td></tr><tr><td rowspan="4">Straight</td><td>15° (3,785)</td><td>5.79%</td><td>15.67%</td><td>41.48%</td></tr><tr><td>20°(3,922)</td><td>6.03%</td><td>16.19%</td><td>42.17%</td></tr><tr><td>25°(4,032)</td><td>6.15%</td><td>16.39%</td><td>42.58%</td></tr><tr><td>30° (4,109)</td><td>6.18%</td><td>16.55%</td><td>42.83%</td></tr></table>

Therefore, when K is higher than the thresholds, it will introduce some redundant scenery features that do not exist in satellite images. In contrast, when K is lower than the thresholds, the obtained ground image features cannot fully capture all the geographic-related features. In either case, it will reduce the correlation between ground image features and satellite image features, leading to a gradual decrease in geolocalization accuracy.

## E. Discussion

1) Computational Efficiency: In order to evaluate the computational efficiency of the proposed ASML, we compare the parameters, architecture, GFLOPs, and R@1 accuracy of ASML and several representative methods. As shown in Table VIII, the parameters of ASML are lower than L2LTR [47], Sample4Geo [37], and FlexGeo [31], and comparable to TransGeo [33], SeqGeo [20], and GeoDTR [46]. However, because the proposed ASML uses smaller input image sizes, its computational costs are significantly lower than those of other methods, while the R@1 accuracy is also higher. This indicates that the proposed ASML has a higher computational efficiency and achieves higher geo-localization accuracy with lower computational costs.

2) Influence of Scenario: As we mentioned in Sec. III, we investigate the influence of trajectory patterns on sequence geo-localization using four trajectory-change thresholds, $i . e . ,$ $\tau _ { \theta } \in \{ 1 5 ^ { \circ } , 2 0 ^ { \circ } , 2 5 ^ { \circ } , 3 0 ^ { \circ } \}$ . For each ground image sequence, we calculate its trajectory change $\Delta \theta$ and classify it as a corner-road sequence when $\Delta \theta \ge \tau _ { \theta } ;$ otherwise, it is classified as a straight-road sequence. As shown in Table X, the number of corner-road samples decreases from 264 to 41 on SeqGeo and from 479 to 155 on CrossRoad-66K as the threshold increases. We report R@1, R@5, and R@1% for each scenario using the same complete satellite gallery.

![](images/0701a44b63cc5720e43b8a3372c0020f3b6ee4cb070ef038c7854484d1f699b8.jpg)  
(a) Non-overlap Precision (NoP)

![](images/3a5ebf642954aca86aaf2266df37b1e2f6b33eba4dd08d9b48871fe511e889e0.jpg)  
(b) Overlap FSR (O-FSR)

![](images/2f36d67e0365a8c93bd99eaaa4614445f892b8dfbb57ae31f2f5d77f9bf5a821.jpg)  
(c) ROI ∆FSR

![](images/7e36769af06b0beaa5602089cef8207bcf5d10a5422c7a34c07a12cba88d3fd6.jpg)  
(d) Aligned Mask IoU  
Fig. 12. Quantitative analysis of SCF mask sensitivity to controlled perturbations on the SeqGeo and CrossRoad-66K datasets. (a) Non-overlap Precision (NoP) measures the proportion of selected features located in geometrically non-overlapping regions. (b) Overlap False Selection Rate (O-FSR) measures the proportion of overlapping features incorrectly selected. (c) ROI ∆FSR measures the increase in false selection within the perturbed overlapping region relative to the clean input. (d) Aligned Mask IoU evaluates the spatial consistency between the clean and perturbed masks. Higher NoP and Mask IoU, and lower O-FSR and ROI ∆FSR indicate better selection performance and robustness.

The results reveal distinct performance patterns across the two datasets. On CrossRoad-66K, corner-road sequences consistently outperform straight-road sequences across all three retrieval metrics under the evaluated thresholds, although the R@1 difference gradually narrows as the threshold increases. On SeqGeo, corner-road sequences achieve higher recall across all metrics at $1 5 ^ { \circ }$ . However, at stricter thresholds, straight-road sequences achieve higher R@1 and R@5, whereas corner-road sequences consistently maintain higher R@1%. For example, at $3 0 ^ { \circ }$ , corner-road sequences achieve an R@1% of 90.24%, compared with 80.24% for straight-road sequences.

These phenomena suggest that the influence of trajectory variation depends on both dataset SCENARIO. Turning trajectories can provide complementary observations that improve the coverage of correct locations within a broader retrieval shortlist, but this advantage does not necessarily translate into more accurate predictions. These results highlight the different contributions of trajectory patterns to candidate retrieval and fine-grained localization.

3) Perturbation Robustness Analysis: To investigate whether SCF confuses nuisance-induced feature dissimilarity with geometric novelty, we evaluate its robustness to five controlled perturbations: feature noise, illumination changes, synthetic occlusion, moving proxy (simulating local appearance changes through region replacement), and viewpoint changes (2-D perspective transformations). Each perturbation is applied only to the second ground image of each geometrically controlled image pair, while the first ground image remains unchanged. For each dataset, we sampled 150 held-out sequences and generated two geometrically controlled image pairs from one source ground image per sequence, using horizontal displacement ratios of

![](images/a6eb28c8078eeccb2cea0e9c6dd54732c7c9e7e06b4a47c9df9cdab9e3e0944a.jpg)  
(d) Ground-Truth  
(e) Removing SFA  
(f) Replacing Geo-InfoNCE with InfoNCE  
Fig. 13. Visualization of module-specific failure cases. (a) shows the first images of the query ground image sequences. (b) shows the Top-3 results from ASML. (c) shows the Top-3 results from the version without SCF. (d) shows the ground-truth. (e) shows the Top-3 results from the version without SFA (f) shows the Top-3 results from the version without Geo-InfoNCE. The green boxes indicate the ground-truth satellite images, while the red boxes denote incorrect retrieval results.

15% and 25%. This yielded 300 image pairs per dataset, with non-overlapping regions determined directly from the crop coordinates.

Then, we adopt four metrics to quantify the selection behavior: (1) Non-Overlap Precision (NoP), measuring the proportion of selected features located in genuinely nonoverlapping regions (higher is better); (2) Overlap False Selection Rate (O-FSR), measuring the proportion of overlapping features incorrectly selected (lower is better); (3) ROI ∆FSR, measuring the change in the selection rate within the perturbed overlapping region relative to the clean input (lower is better); and (4) Aligned Mask IoU, measuring the consistency between the clean and perturbed selection masks after spatial alignment (higher is better).

As shown in Fig. 12, SCF exhibits a consistent preference for geometrically non-overlapping regions across both datasets. Under clean conditions, SCF achieves NoP values of 80.42% and 92.86% on SeqGeo and CrossRoad-66K, respectively, while maintaining low O-FSR values of 2.03% and 0.77%. Different perturbations affect the selection behavior to varying degrees. Feature noise has a relatively limited impact on mask consistency, with aligned mask IoU values of 0.82 and 0.84 on the two datasets. In contrast, illumination and viewpoint changes lead to more noticeable variations in the selected regions, although SCF still maintains a preference for non-overlapping content.

The moving proxy introduces localized selection errors, increasing ROI ∆FSR by 17.50% and 22.42% on SeqGeo and CrossRoad-66K, respectively, while the overall O-FSR remains relatively low. This indicates that localized feature discrepancies can affect the selection mask without substantially changing the global selection statistics. Among all the perturbations, occlusion produces the most pronounced degradation. Compared with the clean condition, NoP decreases from 80.42% to 51.33% on SeqGeo and from 92.86% to 68.69% on CrossRoad-66K, accompanied by increases of 50.33% and 51.83% in ROI ∆FSR, respectively. These results reveal a consistent selection pattern across the two datasets: SCF preferentially retains geometrically complementary features, while appearance-induced dissimilarity, particularly from occlusion, can also increase the selection of overlapping regions. This behavior reflects the inherent trade-off of similarity-based feature selection between suppressing redundant content and preserving complementary information under varying visual conditions. In summary, low cross-image similarity can indicate geometrically complementary content, but it may also arise from appearance variations or feature instability.

4) Failure Case Analysis: To further investigate the behavior of each component, we conduct a module-specific failure analysis, as illustrated in Fig. 13. Specifically, SCF and SFA are removed individually, while Geo-InfoNCE is replaced with InfoNCE and the other settings remain unchanged. For each query, we report the Top-3 retrieved satellite images obtained by the complete ASML and the corresponding ablated variants.

The failure cases demonstrate the complementary roles of SCF, SFA, and Geo-InfoNCE. In scenes dominated by repetitive roads, intersections, buildings, or vegetation, removing SCF increases confusion among visually similar candidates because redundant sequence observations are no longer explicitly suppressed. Removing SFA makes the model more dependent on local appearance similarity and less capable of distinguishing candidates with similar visual elements but different global spatial layouts. Replacing Geo-InfoNCE with InfoNCE, visually plausible hard negatives become more difficult to separate, showing the benefit of geographic supervision in constraining ambiguous feature-space matches.

These observations also demonstrate the limitations of each component. SCF may interpret low correlation caused by illumination changes, occlusions, moving objects, viewpoint variations, or feature instability as complementary information. SFA provides less benefit when spatial layouts are highly repetitive or informative features have already been discarded, while Geo-InfoNCE may be less effective for geographically close samples or noisy geographic annotations. In summary, the failure case analysis is consistent with the quantitative ablation results. SCF mainly reduces redundancy across sequence observations, SFA preserves discriminative spatial organization, and Geo-InfoNCE alleviates ambiguity through geographic supervision.

## VI. CONCLUSION

In this work, we propose a novel cross-view image sequence geo-localization method ASML, which addresses the difficulties of capturing spatial layout information and eliminating redundant content interference. Unlike existing methods that overlook spatial coherence and simply concatenate sequence features, ASML utilizes the spatial attention correlations between the adjacent images in the image sequence to eliminate redundant content by the sequential content fusion module. Moreover, a Geo-InfoNCE loss function is designed to better guide the optimization of the model by simultaneously constraining feature similarity and geographic distance. We also establish a new dataset CrossRoad-66K to evaluate the performance of cross-view image sequence methods in complex scenarios. Extensive experiments demonstrate that ASML outperforms existing methods on both SeqGeo and CrossRoad-66K datasets. Ablation studies and controlled geometric evaluations further confirm the contributions of the proposed components and the preference of SCF for geometrically complementary content. Future work will investigate the integration of motion and visibility cues into adaptive feature selection to improve robustness in complex real-world environments.

## REFERENCES

[1] J. Lin, Z. Zheng, Z. Zhong, Z. Luo, S. Li, Y. Yang, and N. Sebe, “Joint representation learning and keypoint detection for cross-view geo-localization,” IEEE Transactions on Image Processing, vol. 31, pp. 3780–3792, 2022.

[2] J. Wen, H. Yu, and Z. Zheng, “Fanet: Fovea attention network for robust aerial geo-localization across diverse weather conditions,” IEEE Transactions on Image Processing, vol. 35, pp. 5552–5566, 2026.

[3] C. Chen, Q. Chen, M. Ye, and X. Zhang, “Hisymgeo: Hierarchical context symbiosis for cross-view object-level image geo-localization,” IEEE Transactions on Image Processing, vol. 35, pp. 6401–6415, 2026.

[4] D. Wilson, X. Zhang, W. Sultani, and S. Wshah, “Image and object geo-localization,” International Journal of Computer Vision, vol. 132, no. 4, pp. 1350–1392, 2024.

[5] J. Cui, X. Luo, W. Shi, X. Ma, N. Wu, and H. Li, “Tracing locations from pixels: A survey of worldwide image geo-localization,” Information Processing & Management, vol. 64, no. 2, Part C, p. 105226, 2027.

[6] L. Wu, L. Bo, S. Ouyang, and Y. Zhu, “Mrgeo: Robust cross-view geo-localization of corrupted images via spatial and channel feature enhancement,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, pp. 10691–10699, 2026.

[7] N. Wu, C. Yang, B. Qi, M. Zhu, J. Li, and X. Luo, “Ccigeo: Crossview and cross-day-night image geo-localization using daytime image supervision,” IEEE Transactions on Multimedia, vol. 27, pp. 6475–6488, 2025.

[8] J. Li, C. Yang, B. Qi, M. Zhu, J. Chen, and V. C. M. Leung, “Cig2s: A cross-view image geo-localization model based on g2s transform suitable for center-misaligned scenarios,” IEEE Transactions on Computational Social Systems, vol. 12, no. 5, pp. 2342–2357, 2025.

[9] J. Lin, Z. Luo, D. Lin, S. Li, and Z. Zhong, “A self-adaptive feature extraction method for aerial-view geo-localization,” IEEE Transactions on Image Processing, vol. 34, pp. 126–139, 2025.

[10] W. Shi, X. Li, K. Li, J. Fang, Q. Zhou, Q. Geng, and Z. Zhou, “Geobayes: Probabilistic image geo-localization inference via sequential bayesian updating,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, pp. 8997–9005, 2026.

[11] S. Hu, Z. Shi, T. Jin, and Y. Liu, “Query-driven feature learning for cross-view geo-localization,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[12] R. Rodrigues and M. Tani, “Global assists local: Effective aerial representations for field of view constrained image geo-localization,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 3871–3879, 2022.

[13] W. Hu, Y. Zhang, Y. Liang, Y. Yin, A. Georgescu, A. Tran, H. Kruppa, S.-K. Ng, and R. Zimmermann, “Beyond geo-localization: fine-grained orientation of street-view images by cross-view matching with satellite imagery,” in Proceedings of the 30th ACM International Conference on Multimedia, pp. 6155–6164, 2022.

[14] C. Bi, B. Sun, J. Wang, Y. Yuan, and G. Liu, “Bemn: Balanced bias enhanced multi-branch network for cross-view geo-localization,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 36, no. 5, pp. 5968–5982, 2026.

[15] S. Ouyang and Y. Zhu, “Cvgl: Causal learning and geometric topology,” in Advances in Neural Information Processing Systems, vol. 38, pp. 78247–78266, 2025.

[16] J. Li, J. Li, L. Chen, N. Wu, and H. Chen, “From scene to object: Crossview object geo-localization via cross-task knowledge transfer learning,” Knowledge-Based Systems, vol. 351, p. 116682, 2026.

[17] N. Wu, X. Luo, C. Yang, J. Li, L. Yu, and J. Cui, “Smartgeo: Stable mixture-of-experts feature aggregation for cross-view image geolocalization,” in Proceedings of the 34th ACM International Conference on Multimedia, p. 11623–11631, 2026.

[18] S. Vyas, C. Chen, and M. Shah, “Gama: Cross-view video geolocalization,” in European Conference on Computer Vision, pp. 440–456, Springer, 2022.

[19] M. S. Pillai, M. N. Rizve, and M. Shah, “Garet: Cross-view video geolocalization with adapters and auto-regressive transformers,” in European Conference on Computer Vision, pp. 466–483, 2024.

[20] X. Zhang, W. Sultani, and S. Wshah, “Cross-view image sequence geolocalization,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2914–2923, 2023.

[21] D. Yuan, F. Maire, and F. Dayoub, “Temporal attention for crossview sequential image localization,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 7429–7436, 2024.

[22] S. Hu, M. Feng, R. M. Nguyen, and G. H. Lee, “Cvm-net: Cross-view matching network for image-based ground-to-aerial geo-localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7258–7267, 2018.

[23] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic, “Netvlad: Cnn architecture for weakly supervised place recognition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5297–5307, 2016.

[24] Y. Shi, L. Liu, X. Yu, and H. Li, “Spatial-aware feature aggregation for image based cross-view geo-localization,” Advances in Neural Information Processing Systems, vol. 32, pp. 10090–10100, 2019.

[25] J. Ye, Z. Lv, W. Li, J. Yu, H. Yang, H. Zhong, and C. He, “Crossview image geo-localization with panorama-bev co-retrieval network,” in European Conference on Computer Vision, pp. 74–90, 2024.

[26] S. Cai, Y. Guo, S. Khan, J. Hu, and G. Wen, “Ground-to-aerial image geo-localization with a hard exemplar reweighting triplet loss,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8391–8400, 2019.

[27] Y. Guo, M. Choi, K. Li, F. Boussaid, and M. Bennamoun, “Soft exemplar highlighting for cross-view image-based geo-localization,” IEEE Transactions on Image Processing, vol. 31, pp. 2094–2105, 2022.

[28] G. Vaca-Castano, A. R. Zamir, and M. Shah, “City scale geo-spatial trajectory estimation of a moving camera,” in Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1186–1193, IEEE, 2012.

[29] K. Regmi and M. Shah, “Video geo-localization employing geo-temporal feature learning and gps trajectory smoothing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12126– 12135, 2021.

[30] Y. Shi, X. Yu, S. Wang, and H. Li, “Cvlnet: Cross-view semantic correspondence learning for video-based camera localization,” in Proceedings of the Asian Conference on Computer Vision, pp. 123–141, 2022.

[31] Q. Wu, P. Xia, L. Yu, Y. Liu, M. Xiong, L. Zhong, J. Chen, M. Yang, Y. Zhang, and Y. Wan, “Set-cvgl: A new perspective on cross-view geolocalization with unordered ground-view image sets,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 233, pp. 328–345, 2026.

[32] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A convnet for the 2020s,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11976–11986, 2022.

[33] S. Zhu, M. Shah, and C. Chen, “Transgeo: Transformer is all you need for cross-view image geo-localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1162– 1171, 2022.

[34] T. Wang, J. Li, and C. Sun, “Dehi: A decoupled hierarchical architecture for unaligned ground-to-aerial geo-localization,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 3, pp. 1927– 1940, 2023.

[35] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International Conference on Machine Learning, pp. 10347–10357, 2021.

[36] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, “Supervised contrastive learning,” Advances in Neural Information Processing Systems, vol. 33, pp. 18661–18673, 2020.

[37] F. Deuser, K. Habel, and N. Oswald, “Sample4geo: Hard negative sampling for cross-view geo-localisation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16847–16856, 2023.

[38] T. Lentsch, Z. Xia, H. Caesar, and J. F. Kooij, “Slicematch: Geometryguided aggregation for cross-view pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17225–17234, 2023.

[39] S. Xu, C. Zhang, L. Fan, G. Meng, S. Xiang, and J. Ye, “Addressclip: Empowering vision-language models for city-wide image address localization,” in European Conference on Computer Vision, pp. 76–92, 2024.

[40] P. Jia, Y. Liu, X. Li, X. Zhao, Y. Wang, Y. Du, X. Han, X. Wei, S. Wang, and D. Yin, “G3: an effective and adaptive framework for worldwide geolocalization using large multi-modality models,” Advances in Neural Information Processing Systems, vol. 37, pp. 53198–53221, 2024.

[41] P. Jia, S. Park, S. Gao, X. Zhao, and S. Li, “Georanker: Distance-aware ranking for worldwide image geolocalization,” in Advances in Neural Information Processing Systems, vol. 38, pp. 17673–17699, 2025.

[42] L. Liu and H. Li, “Lending orientation to neural networks for crossview geo-localization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5624–5633, 2019.

[43] S. Workman, R. Souvenir, and N. Jacobs, “Wide-area image geolocalization with aerial reference imagery,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3961–3969, 2015.

[44] S. Zhu, T. Yang, and C. Chen, “Vigor: Cross-view image geo-localization beyond one-to-one retrieval,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3640–3649, 2021.

[45] X. Zhang, X. Li, W. Sultani, C. Chen, and S. Wshah, “Geodtr+: Toward generic cross-view geolocalization via geometric disentanglement,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[46] X. Zhang, X. Li, W. Sultani, Y. Zhou, and S. Wshah, “Cross-view geolocalization via learning disentangled geometric layout correspondence,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 37, pp. 3480–3488, 2023.

[47] H. Yang, X. Lu, and Y. Zhu, “Cross-view geo-localization with layer-tolayer transformer,” Advances in Neural Information Processing Systems, vol. 34, pp. 29009–29020, 2021.