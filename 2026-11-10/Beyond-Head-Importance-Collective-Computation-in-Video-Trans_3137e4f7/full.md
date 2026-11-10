# Beyond Head Importance: Collective Computation in Video Transformers

Yi Yang<sup>1</sup> Siyuan Liu<sup>1</sup> Bingbing Nie<sup>1,2\*</sup>

<sup>1</sup>School of Vehicle and Mobility, Tsinghua University, China <sup>2</sup>State Key Laboratory of Intelligent Green Vehicle and Mobility, Tsinghua University, China

![](images/95c08cdaaa160e958d71ccb75d5b477079cf343c907f397cad1593d8549fb926.jpg)

## Abstract

Attention heads in video Transformers vary greatly infunctional importance. Understanding their contributions requires examining not only which heads are important, but also how they participate in collective computation within the network. We study this collective computation from two perspectives: the downstream use of head outputs and the composition of interactions across layers. Ablation analyses offour video Transformers show that highimportance heads identified on a single task also support other video understanding tasks and help preserve the representational structure that distinguishes different videos. Representational responses to head ablation correlate weakly with head importance after the attention module, but more strongly after the MLP in the same block. Adapting this downstream processing through MLP-only fine-tuning further improves the use of lower-importance head outputs. Meanwhile, through composition across layers, sets ofheads with shorter spatiotemporal attention distances support stronger perturbation responses at distant tokens than sets with longer distances. Their joint ablation causes even greater performance loss than ablating equally sized sets of high-importance heads, revealing collective contributions not fully captured by individual importance

rankings. These findings highlight collective computation as a key perspectivefor understanding attention headfunction: functional contributions depend on both how head outputs are used downstream and how local interactions compose across layers.

## 1. Introduction

Understanding how deep neural networks form and use information through internal computation is an important goal of model interpretability research [14, 21, 49]. Video understanding requires models to integrate information across space and time to recognize visual content and capture the dynamics of actions, motion, and events [7, 61, 65]. Video Transformers use multi-head self-attention to enable spatiotemporal interactions and progressively build visual representations across layers [4, 19, 57, 64]. Representation probing and attention analysis can reveal the semantic and dynamic information encoded by these models, but understanding how representations are formed and maintained also requires examining their internal computations [17, 29, 30, 35]. In particular, how attention heads contribute to collective computation in video Transformers remains unclear.

The functional importance of attention heads in Transformers is highly uneven: ablating many low-importance heads has little effect on task performance, whereas ablating just a few high-importance heads can substantially degrade performance [35, 45, 50, 58]. Ablation analysis can identify the key heads on which a model relies, but offers limited insight into the computations underlying their functional contributions [27, 36]. Individual head importance may also fail to fully capture the collective contribution of multiple heads working together [10, 48].

We therefore ask: How do attention heads participate in collective computation in video Transformers? We approach this question from two perspectives. First, head outputs are processed by MLPs and subsequent network layers, making their contributions dependent on how downstream computation uses these signals. Second, a head’s attention range describes only a single interaction, whereas local interactions can combine across layers to produce broader spatiotemporal effects. We first use cross-task ablation analysis to identify heads that support multiple tasks, then examine how downstream processing and cross-layer composition shape their functional contributions.

We conduct attention head ablation analysis in four video Transformers: V-JEPA [6] and V-JEPA 2 [5], which are trained through feature prediction, and VideoMAE [56] and VideoMAE V2 [60], which are trained through masked reconstruction. We then focus on V-JEPA to investigate collective computation through downstream processing and cross-layer interactions. Our main contributions are as follows:

• We characterize the cross-task importance and representational roles of attention heads. High-importance head sets identified on a single task also support other video understanding tasks and help preserve representational differences across videos.

• We show how downstream MLPs shape the use of head outputs. Ablation-induced representational changes correlate more strongly with head importance after MLP processing than immediately after the attention update. Fine-tuning MLPs alone partially restores performance when only lower-importance heads remain, suggesting improved use of their outputs.

• We reveal collective contributions of local heads beyond individual importance rankings. Across layers, local head sets support stronger perturbation responses at distant tokens than global head sets, and their joint ablation causes greater performance loss than ablating equally sized high-importance head sets. Fixed spatiotemporal kernels partially restore performance after ablation, indicating that stable routing supports part of their function.

## 2. Related work

Internal representations in vision and video Transformers. Prior work has used comparisons of representations across layers and attention analysis to reveal how information is organized within vision Transformers. Representations in image ViTs evolve with depth differently from those in convolutional networks [53]. Attention patterns and dense features in self-supervised vision Transformers reflect object regions, spatial structure, and semantic information, and are shaped by the pretraining objective [3, 11, 51, 59]. In video Transformers, Kowal et al. used concept discovery to reveal a progression from representations of spatiotemporal positions in early layers to object and event concepts in deeper layers [35]. Layer-wise probes have also been used to assess how readily motion and physical properties can be decoded at different depths [29, 47]. These studies characterize representational content and its distribution across layers, providing a foundation for analyzing the underlying internal computations.

Specialization, importance, and redundancy of attention heads. Prior work has identified functional specialization among Transformer attention heads, with some heads exhibiting syntactic, coreferential, or positional attention patterns [13, 34, 58]. Importance analysis and pruning methods have also been used to identify key components in vision Transformers [38, 50, 62]. In video models, Kowal et al. combined concept discovery with joint masking to assess head importance and perform pruning [35]; Joseph et al. used targeted head interventions to study physical and temporal reasoning abilities [29]. Functional relationships among heads have also received attention: individual head importance does not necessarily align with actual pruning outcomes [10], the functional roles of heads depend on other heads [48], and cooperation, interference, and redundancy have been observed among heads in vision Transformers [41, 43]. We focus on the gap between individual head importance and the collective functional contributions of head sets in video Transformers, testing whether the former can adequately predict the performance degradation caused by jointly ablating head sets.

Information flow and cross-layer interactions in attention. Prior work has shown that understanding the computational roles of Transformer attention heads requires considering factors beyond attention weights, including residual connections, normalization, and feed-forward networks [9, 28, 31]. Kobayashi et al. examined how residual connections and normalization affect representational updates, while Geva et al. revealed the roles of feed-forward networks in processing representations and forming predictions [23, 24, 32]. Related studies have also used attention rollout, cross-layer attribution, and computational circuit analysis to examine computational dependencies among tokens and network components across layers [1, 12, 18, 46]. In vision models, connections between local attention and convolution, along with spatiotemporal factorization and local window designs for video attention, further illustrate the roles of different interaction ranges and organizational patterns [7, 15, 40]. We combine head ablation, MLP interventions, and cross-layer perturbation analysis to study collective computation in video Transformers, examining how downstream processing shapes head contributions and how local interactions compose across layers to produce broader effects.

## 3. Head importance

In this section, we rank attention heads in video Transformers by importance using random joint ablation, providing a common basis for subsequent analyses. We examine whether important heads identified on a single task also affect other tasks, and how head sets with different levels of importance influence the structure of video representations. We further examine how well head importance can be predicted from computational properties.

## 3.1. Importance estimation

Functional redundancy among attention heads means that ablating a single head often produces only a small change in performance, making importance estimates sensitive to evaluation variability [45, 58]. We therefore estimate the relationship between each head and task performance loss through repeated random joint ablation of different head sets.

We use the ViT-L encoders of V-JEPA, V-JEPA 2, Video-MAE, and VideoMAE V2, each comprising 24 Transformer blocks with 16 attention heads per block, for a total of 384 heads. During evaluation, we freeze the pretrained encoder and re-extract video representations under each ablation condition, then train a linear classifier to assess the linear decodability of task-relevant information after intervention [2, 26].

We define four video understanding tasks covering general action and object recognition as well as fine-grained judgments of dynamics. Action evaluates action category recognition using a ten-class subset of Something-Something V2 (SSv2) [25]. Object uses nine instrumentrelated activity classes from UCF101 to assess object- or scene-related semantic information [54]. Direction and Outcome are binary classification tasks based on SSv2, assessing the direction of object motion caused by an action (left or right) and whether the action is successfully completed, respectively. We estimate head importance separately on Action and Object to assess the consistency of the resulting rankings. All subsequent cross-task ablations and other analyses use the importance ranking derived from ${ \mathrm { A c } } -$ tion. Full dataset details and evaluation protocols are provided in Appendix A.1.

In each trial, we randomly select 25% of all attention heads for joint ablation, repeating the procedure for $N =$ 3000 trials. Each selected head’s output is replaced with its mean over training samples and token positions to suppress sample- and position-specific information [63]. Let $S _ { n }$ denote the set of heads ablated in the n-th trial, and let $D ( S _ { n } )$ denote the drop in task accuracy relative to the unablated model (Intact). The importance $I ( h )$ of head h is defined as the average performance loss across all ablation trials that include that head:

$$
I ( h ) = \frac { 1 } { N _ { h } } \sum _ { n = 1 } ^ { N } \mathbf { 1 } [ h \in S _ { n } ] D ( S _ { n } ) ,\tag{1}
$$

where $N _ { h }$ is the number of trials in which head $h$ is included in the ablated set. We rank heads in descending order of $I ( h )$ for subsequent cumulative ablation and functional analysis.

## 3.2. Cross-task consistency

Using the importance ranking derived from Action, we examine how head ablation affects other video understanding tasks. As shown in Figure 1, when low-importance heads are ablated first in V-JEPA, performance on all four tasks remains largely unchanged even after approximately half of the heads have been ablated. In contrast, ablating high-importance heads first causes accuracy to drop rapidly across all tasks. These results indicate that tasks relying on different types of video information nevertheless share a dependence on some key attention heads.

![](images/d6ab8fef317de8b46903237984e3d77b028d5adbed2a76f93818d0e8dc3b1c6a.jpg)

![](images/5f6cfe8bf50a1a884835988f985d428c08fdc09c206590b952c1f4c4278b8b9f.jpg)  
Figure 1. Task performance under different attention head ablation orders. Solid lines indicate cumulative ablation from lowest to highest importance (Bottom-first), while dashed lines indicate ablation from highest to lowest importance (Top-first).

This pattern holds across all four video Transformers. Table 1 shows that ablating the top 25% of heads by importance also substantially reduces accuracy on the other tasks, whereas ablating the bottom 25% generally has a smaller effect and even improves performance on some tasks. Comparing the importance rankings estimated independently on Action and Object, we find positive Spearman correlations in all four models, with 34–57 heads shared between their top 25% sets, exceeding the overlap expected by chance in every model. These results indicate that multiple video understanding tasks rely on a shared subset of high-importance heads. Cumulative ablation results for the remaining models are provided in Figure 8 in the appendix.

Table 1. Cross-task ablation results for head sets of varying importance. Values indicate accuracy changes relative to Intact (in percentage points, pp) after ablating the top or bottom 25% of heads. The last two columns report the Spearman correlation between the Action and Object importance rankings and the overlap between their top 25% head sets (expected overlap by chance: 24).
<table><tr><td>Model</td><td>Ablated heads</td><td>Action</td><td>Object</td><td>Direction</td><td>Outcome</td><td> $\rho _ { \mathbf { O b j } , \mathbf { A c t } }$ </td><td>Top 25%∩</td></tr><tr><td rowspan="2">V-JEPA</td><td>Top 25%</td><td>-34.3</td><td>-30.6</td><td>-31.1</td><td>-19.3</td><td>0.61</td><td>57</td></tr><tr><td>Bottom 25%</td><td>+3.9</td><td>+0.8</td><td>+1.8</td><td>+2.5</td><td></td><td></td></tr><tr><td rowspan="2">VideoMAE</td><td>Top 25% Bottom 25%</td><td>-25.7</td><td>-51.6</td><td>-10.0</td><td>-11.8</td><td>0.43</td><td>46</td></tr><tr><td></td><td>+6.1</td><td>-5.2</td><td>+2.5</td><td>+2.1</td><td></td><td></td></tr><tr><td rowspan="2">V-JEPA2</td><td>Top 25% Bottom 25%</td><td>-27.9 +12.9</td><td>-26.6 0.0</td><td>-20.7 +3.2</td><td>-16.1</td><td>0.43</td><td>51</td></tr><tr><td></td><td></td><td></td><td></td><td>+2.5</td><td></td><td></td></tr><tr><td rowspan="2">VideoMAEv2</td><td>Top 25% Bottom 25%</td><td>-23.9</td><td>-11.1</td><td>-8.9</td><td>-6.1</td><td>0.17</td><td>34</td></tr><tr><td></td><td>+8.6</td><td>0.0</td><td>+8.9</td><td>+3.2</td><td></td><td></td></tr></table>

(a)  
![](images/eaf35aad09d8b9977347bb8643b22460588f5f81c05ec6d7d846a9c88c805776.jpg)

![](images/aa4b96d0f821c4029e479843e433903ab84ee6c110ac56a7090fc2804805fe8c.jpg)

(b)  
![](images/c859cd7943d930ed1c9a955ed1e1d629e474ca8cf51e79e2db73fb110e55b2fb.jpg)  
(c)

![](images/974e4f9d5afd6c133d7e09fcba2c60afc8bf55c1371a1ed2fe979fa128e68c32.jpg)

![](images/4adcb2176be40d5952014ab01e7a935ea21bd816d5687c02cf195dce30a3d5cc.jpg)

(d)  
![](images/8459e1786f36de783904582e02d8aaa7ebab7d5ffe15e979c425c01bac2d4390.jpg)  
Figure 2. Representational structure under head sets with different importance levels. (a) Cross-layer CKA matrices between intervened and Intact representations. (b) Shared UMAP projections before and after ablation, using an Intact-fitted mapping. (c) Layer-wise mean pairwise cosine similarity across videos. (d) Spearman correlation of inter-video distance rankings and 10-NN Jaccard similarity, both relative to Intact.

## 3.3. Representational structure

To examine whether the cross-task roles of high-importance heads are associated with a shared internal representational structure, we retain only the top or bottom 50% of attention heads by importance in separate conditions. We compare the resulting representations with those of Intact and assess how well the original structure of representational differences across videos is preserved.

High-importance head sets better preserve the original representations. As shown in Figure 2(a), when only high-importance heads are retained, representational similarity measured by centered kernel alignment (CKA) [33] between the intervened model and Intact remains high across most layers, particularly in deeper layers. Retaining only low-importance heads substantially reduces representational similarity in deeper layers. The shared UMAP [44] projections show a similar trend (Figure 2(b)): retaining high-importance heads yields a distribution of video samples closer to that of Intact, whereas retaining lowimportance heads leads to larger sample displacements and a more concentrated distribution.

Figure 2(c) shows that retaining only low-importance heads substantially increases the mean pairwise cosine similarity between video representations in deeper layers, indicating that representations of different videos become more aligned in direction. Retaining high-importance heads better preserves representational differences across videos. Figure 2(d) further shows that, when high-importance heads are retained, both inter-video distance rankings and nearestneighbor structure remain closer to those of Intact. These results indicate that high-importance head sets help maintain relative distances and local neighborhood relationships among video representations.

The other three video Transformers also show a relative advantage of high-importance head sets in preserving representational structure (Figure 9 in the appendix). Together with the cross-task ablation results, these findings suggest that preserving representational distinctions across videos may be one reason why high-importance heads play important roles in multiple tasks.

## 3.4. Importance prediction

Building on the importance estimates above, we examine how the computational properties of attention heads relate to and predict their functional importance. We focus on V-JEPA, with results for the other models provided in $\mathsf { A p - }$ pendix C.3.

We extract ten attention head properties covering four aspects: spatiotemporal attention range, attention weight structure, outputs and representational responses, and network position. The complete list and definitions are provided in Appendix B. Among these properties, mean spatial attention distance $d _ { \mathrm { s } }$ and mean temporal attention distance $d _ { \mathrm { t } }$ quantify the spatial and temporal extent of attention, respectively. $\Delta _ { \mathrm { a t t h } }$ and $\Delta _ { \mathrm { m l p } }$ measure the magnitudes of representational changes induced by the same head ablation, measured after the attention module and the MLP in the same block, respectively.

We use these properties to predict head importance $I ( h )$ and evaluate generalization to unseen heads through repeated cross-validation. Predictive performance is measured by the Spearman rank correlation between predicted and actual importance rankings, and the contribution of each property is assessed through permutation importance [20]. Details of property selection, model training, and evaluation protocols are provided in Appendix A.2.

As shown in Figure 3(a), predictive performance improves as more properties are included and begins to saturate at around seven properties. Figure 3(b) shows that the post-MLP ablation-induced representational change $\Delta _ { \mathrm { m l p } }$ and the mean spatial and temporal attention distances $d _ { \mathrm { s } }$ and $d _ { \mathrm { t } }$ provide the main predictive cues: heads with larger $\Delta _ { \mathrm { m l p } }$ and shorter attention distances tend to be more important. Results for the other video Transformers likewise identify spatial attention range as a predictive cue, while the contributions of other properties and overall predictive performance vary across models (Figure 10 in the appendix).

Overall, prediction correlations remain modest, indicating that the computational properties of individual heads explain only part of the variation in head importance. The two types of predictive cues identified above warrant further investigation: post-MLP ablation-induced representational changes reflect how downstream computation processes head outputs, while spatiotemporal attention distances raise the question of how attention interactions with different ranges combine across layers. The next two sections examine these two directions in turn.

(a)  
![](images/9740768ceb3d306f664b0e7eef560f4d8545393511538430bb2018ce1ba5567a.jpg)

(b)  
![](images/d0c37267c3b001e1197f4f00fc64aae51eb466ddb97d3566c64c472fdfa50b49.jpg)  
Figure 3. Head importance prediction and property contributions. (a) Spearman correlation between predicted and actual importance rankings for unseen heads, using different numbers of properties. Results are evaluated over 10 repetitions of six-fold cross-validation; intervals indicate variability across data splits. (b) Decrease in prediction correlation after randomly permuting each property individually when using seven properties.

## 4. Downstream computation

Attention head outputs are processed through residual connections, MLPs, and subsequent blocks, so their functional importance may depend on how the downstream network uses these signals. The post-MLP ablation-induced representational change $\Delta _ { \mathrm { m l p } }$ is one of the main predictive cues for head importance (Figure 3(b)). In this section, we compare ablation-induced representational changes before and after the MLP and combine MLP interventions with selfsupervised adaptation experiments to examine how downstream computation shapes the functional contributions of attention heads.

As shown in Figure 4(a), we compare the representational changes induced by the same head ablation after the attention residual addition and after the MLP residual addition in the same block, denoted by $\Delta _ { \mathrm { a t t n } }$ and $\Delta _ { \mathrm { m l p } } .$ , respectively. Figure 4(b) shows that the Spearman correlation between $\Delta _ { \mathrm { a t t n } }$ and head importance rank is only −0.05, compared with −0.21 for $\Delta _ { \mathrm { m l p } } ,$ , with a few high-importance heads exhibiting exceptionally large post-MLP representational changes. These results indicate that representational changes after MLP processing better reflect the functional characteristics of high-importance heads than the immediate changes following the attention update.

We set the output of a Transformer block’s MLP branch to zero, so that representations pass through the MLP sublayer solely via the residual connection, and re-estimate the importance ranking of heads within that block. Figure 4(c) shows substantial changes in the within-block ranks of some heads across the six selected blocks, indicating that the computational conditions following the attention update influence the relative functional importance of attention heads.

(a)  
![](images/9328ee4253cc8a318cef534127c34574d7ee95cb1950290e93c02625ef692d48.jpg)

(b)  
![](images/7e75db214d3b98cc6dc2e72a9e5f786ca3ea010a206daf2f3c66506dcacea232.jpg)

![](images/ae76c3b2b7390ed1e6562cea61771b0f37437e24a4c8feae56b649b3bb8426d9.jpg)

![](images/6489f7d5cfd961082e16e690d4137c432e8674966c2fc61d53683933f29c7f43.jpg)

![](images/2e6538710cfe1fd788c3cddec9160a2615210256b6cbcc92de5cab3ab45d9845.jpg)

![](images/b7400427087e98edb6b2fe6aa6b8630a7d750335775a9455f8ba8d772ed323cc.jpg)  
Figure 4. Downstream MLP processing and the functional importance of attention heads. (a) Measurement points for ablation-induced representational changes $\Delta _ { \mathrm { a t t } 1 }$ and $\Delta _ { \mathrm { m l p } }$ . (b) Relationships between ablation-induced representational changes and head importance ranks. Each point represents a head. Spearman correlations $\rho$ and p-values are annotated. (c) Absolute changes in within-block head importance ranks after ablating the MLP in the same block, shown for six blocks. Box plots show the distributions, and white diamonds indicate the means. (d) Layer-wise accuracy on Action and Object under different head retention and MLP update conditions.

With only the bottom half of attention heads by importance retained, we fine-tune only the MLP parameters using V-JEPA’s original feature prediction pretraining objective, keeping all other encoder parameters frozen (see Appendix A.3 for training details). Self-supervised adaptation partially restores performance on Action and Object using features from each layer (Figure 4(d)). This indicates that, after high-importance heads are ablated, downstream MLPs can still adapt through self-supervision to improve the usability of task-relevant information in the remaining representations.

## 5. Local attention across layers

Shorter spatial and temporal attention distances are associated with greater head importance (Figure 3(b)), raising the question of how local interactions support video understanding through composition across layers. In this section, we examine the collective computation of local attention heads through long-range perturbation responses, joint ablation, and fixed attention routing.

## 5.1. Long-range interactions

The effective spatial extent of network computation can emerge through the composition of local operations across layers, as previously studied in convolutional neural networks [42]. We first examine how local attention heads support interactions between distant tokens across layers. We rank all attention heads by mean spatial attention distance $d _ { \mathrm { s } }$ and retain either the half with the shortest distances or the half with the longest distances, denoted as Local half and Global half, respectively. We apply a small perturbation to the central token of the input sequence and measure the representational responses of other tokens at different network depths to assess the range and strength of interactions across positions supported by the two head sets. Response magnitudes are computed using central finite differences; the detailed protocol is provided in Appendix A.4.

Despite their shorter single-step attention range, local head sets can support stronger interactions between distan tokens through computation across multiple layers. Figure 5(a) shows the response distributions in the final block under three conditions: responses under Local half extend from the vicinity of the perturbation source to distant spatial regions and other time slices, whereas Global half exhibits weaker responses overall. To quantitatively compare the long-range responses supported by the two head sets, we compute the ratio of Local half to Global half response magnitudes at the 52 tokens along the outer boundary of the 14 × 14 grid in the time slice containing the perturbation source (see Appendix A.4 for calculation details). Figure 5(b) shows the per-token ratios and their mean. In early layers, Global half produces stronger long-range perturbation responses than Local half, consistent with its broader single-step spatial attention range. As network depth increases, local interactions combine across layers to propagate the perturbation’s effects to more distant positions, allowing the long-range responses under Local half to gradually surpass those under Global half. These results show that short-range interactions can produce widespread and stronger long-range effects through composition across layers.

![](images/face72a9429c2da0803621718323a259def3822e4f60add6a92cc954eeefc327.jpg)

![](images/7843ef88a4394ce36f9935d011584f44b96b1b6124679317a70b0c5e563ccb5b.jpg)  
Figure 5. Perturbation responses under different head sets. (a) Final-layer response distributions under three conditions. Red crosses mark the perturbation source, whose position is left blank. (b) Layer-wise ratios of Local half to Global half response magnitudes at the 52 boundary tokens in the temporal slice containing the perturbation source. Scatter points show per-token ratios, and the line shows their mean.

## 5.2. Collective contributions

To examine whether the cross-layer role of local heads is reflected in downstream task performance, we compare performance degradation under cumulative ablation of different head sets. As shown in Figure 6(a), ablating spatially or temporally local heads first causes substantial performance degradation on both main tasks. As the ablated fraction increases, accuracy drops particularly sharply when spatially local heads are ablated first, with the curve even falling below that obtained by ablating heads in descending order of importance. These results indicate that individual head importance rankings cannot fully predict the performance degradation caused by jointly ablating head sets and may underestimate the overall contribution of spatially local head sets to task performance. Ablating global heads first still causes greater performance loss than ablating lowimportance heads first, indicating that global heads also make functional contributions.

Figure 6(b) shows limited correspondence between head importance quartiles and spatial or temporal attention distance quartiles, indicating that the substantial performance loss from ablating local head sets cannot be readily explained by their overlap with high-importance heads. Spatial and temporal attention distances are more closely aligned, with heads concentrated mainly in the same or adjacent quartiles across the two dimensions. This also provides insight into the responses across time slices in Figure 5(a): although Local half is selected by spatial attention distance and also tends to have a short temporal attention range, perturbation effects can still extend to other time slices through local interactions across multiple layers. Some tasks in the other three models are also relatively sensitive to the ablation of local head sets (Figure 13 in the appendix). These results further indicate that individual head importance does not fully align with collective functional contributions, and that the role of local heads must be understood in terms of their composition across layers.

## 5.3. Fixed attention routing

Jointly ablating local head sets causes substantial performance loss, but do their functions depend on inputdependent attention routing [39, 55]? Figure 6(c) shows two examples, one spatially local head and one temporally local head: the query token’s mean attention is concentrated in its spatial neighborhood or at nearby temporal positions, exhibiting local aggregation patterns resembling convolutional kernels. This suggests that relatively stable spatiotemporal routing may support some of the functions of local heads.

To test this hypothesis, we restore attention computation in the ablated heads using fixed three-dimensional spatiotemporal kernels. We first compute each head’s mean attention matrix, then average the weights over query-key position pairs with the same relative spatiotemporal displacement to construct fixed kernels, followed by row-wise normalization. After replacement, the attention weights no longer vary with the video input, but they still aggregate the current video’s value vectors through weighted summation, preserving input-dependent content information.

Recovery with fixed 3-D kernels  
![](images/deaaded0e064d6cdb87579b9575fc561722dc68fd61884e4aefb3c93ea7ebb94.jpg)

(a)  
![](images/11975bd1d778b967f9c43666567b518a5be04a4362d5fdf97ccca3f9e56f0c59.jpg)

![](images/24c9f1764c4cb0294a0b98d5d52b2f594c760e83fab5e5ecbf2b934284729f03.jpg)

(c)  
![](images/c408856bd1aac372b9a4eafe27f18709996eb9e61b7f266974435b5755aa3cc8.jpg)

(b)  
![](images/5a8bdc5c879458e434708f755101c1613fc8bcd8f6dd25586ba00579d9d58807.jpg)

![](images/13298ded5810a2b8f38d72a98870092cee9a97133162b814616d865e9eeb766f.jpg)  
Q1: top 25%; Q2: 25–50%; Q3: 50–75%; Q4: 75–100% (by importance/locality rank)

(d)  
![](images/18fc427a8d3fca9fce54a6887f2e081ac50ed12fd4688a207d2a3412c977d8b8.jpg)  
Figure 6. Performance loss from ablating head sets and recovery with fixed routing. (a) Cumulative ablation results with heads ranked by importance, spatial attention distance, or temporal attention distance. (b) Cross-tabulations of quartile groups for each pair of metrics. Q1 through Q4 correspond to importance from highest to lowest and attention distance from shortest to longest. Numbers indicate head counts in the corresponding groups. (c) Two example attention distributions, one for a spatially local head and one for a temporally loca head. Crosses mark query positions; heatmaps and bar plots show the query token’s mean spatial and temporal attention distributions, respectively. (d) Action accuracy loss (pp) under mean-replacement ablation and replacement with fixed spatiotemporal kernels for three head sets: the top 25% by importance, the most spatially local 25%, and the most temporally local 25%. Arrows indicate the fraction of the ablation-induced loss recovered.

Figure 6(d) shows that ablating the most spatially local 25% of attention heads causes a greater loss in task accuracy than ablating the top 25% by importance (43.6 vs. 34.3 percentage points). Restoring computation with fixed spatiotemporal kernels recovers 56% of the loss for the former set, substantially more than the 27% recovered for the latter. For the most temporally local 25% of heads, fixed kernels recover 75% of the ablation-induced loss. These results indicate that a substantial portion of the functions of local head sets can be supported by stable routing for local information aggregation, without relying entirely on inputdependent attention weights.

## 6. Conclusion, limitations and future work

This work investigates collective computation in video Transformers by examining how head outputs are used downstream and how interactions compose across layers. Across four models, high-importance heads support multiple tasks and preserve representational distinctions across videos. Further analyses of V-JEPA show that MLP processing shapes head importance, while MLPonly fine-tuning partially restores performance when lowerimportance heads remain. Local interactions compose across layers to support long-range perturbation responses and collective contributions beyond individual importance rankings. Partial recovery with fixed spatiotemporal kernels further suggests that stable routing supports some functions of local heads. Our study has two limitations: evaluations cover limited task subsets and use linear probes, while ablation and perturbation responses reveal functional dependencies without fully identifying the computational pathways. Future work could trace how attention heads and MLPs jointly transform local spatiotemporal information into object, motion, and event representations, and examine how these pathways support temporal reasoning across broader video tasks.

## Acknowledgments

This study is supported by the National Natural Science Foundation of China (52572413).

## References

[1] Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In ACL, pages 4190–4197, 2020. 2, 4

[2] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2018. 3

[3] Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. On the effectiveness of ViT features as local semantic descriptors. In Computer Vision – ECCV 2022 Workshops, pages 39–55, Cham, 2023. Springer. 2

[4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. ViViT: A video´ vision transformer. In ICCV, pages 6836–6846, 2021. 1

[5] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 2

[6] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, 2024. 2

[7] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, pages 813–824, 2021. 1, 3, 4

[8] Leo Breiman. Random forests. Machine Learning, 45(1): 5–32, 2001. 1

[9] Gino Brunner, Yang Liu, Damian Pascual, Oliver Richter,´ Massimiliano Ciaramita, and Roger Wattenhofer. On identifiability in transformers. In ICLR, 2020. 2

[10] Aakriti Budhraja, Madhura Pande, Preksha Nema, Pratyush Kumar, and Mitesh M. Khapra. On the weak link between importance and prunability of attention heads. In EMNLP, pages 3230–3235, Online, 2020. Association for Computational Linguistics. 2

[11] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9650–9660, 2021. 2

[12] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In CVPR, pages 782–791, 2021. 2, 4

[13] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. What does BERT look at? an analysis of BERT’s attention. In Proceedings ofthe 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 276–286, Florence, Italy, 2019. Association for Computational Linguistics. 2

[14] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria Garriga-Alonso. Towards\` automated circuit discovery for mechanistic interpretability. In NeurIPS, pages 16318–16352. Curran Associates, Inc., 2023. 1

[15] Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self-attention and convolutional layers. In ICLR, 2020. 3, 4

[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,

Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 4

[17] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9:160–175, 2021. 1

[18] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Das-Sarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformercircuits.pub/2021/framework/index.html. 2

[19] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In ICCV, pages 6824–6835, 2021. 1

[20] Aaron Fisher, Cynthia Rudin, and Francesca Dominici. All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20 (177):1–81, 2019. 5, 1

[21] Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaud hary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard. Causal abstraction: A theoretical foundation for mechanistic interpretability. Journal of Machine Learn ing Research, 26(83):1–64, 2025. 1

[22] Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006. 1

[23] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In EMNLP, pages 5484–5495, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 2

[24] Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In EMNLP, pages 30–45, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. 2

[25] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The “something something” video database for learning and evaluating visual common sense. In ICCV, pages 5842–5850, 2017. 3

[26] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In EMNLP, 2019. 3

[27] Alon Jacovi and Yoav Goldberg. Towards faithfully inter pretable NLP systems: How should we define and evaluate

faithfulness? In ACL, pages 4198–4205, Online, 2020. Association for Computational Linguistics. 2

[28] Sarthak Jain and Byron C. Wallace. Attention is not explanation. In NAACL-HLT, pages 3543–3556, 2019. 2

[29] Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel, Shahab Bakhtiari, Blake Aaron Richards, and Michael Rabbat. Interpreting physics in video world models. In ICML, pages 54687–54724, 2026. 1, 2

[30] Youngjun Jun, Seil Kang, Woojung Han, and Seong Jae Hwang. I’m a map! interpretable motion-attentive maps: Spatio-temporally localizing concepts in video diffusion transformers. In CVPR, pages 11525–11535, 2026. 1

[31] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight: Analyzing transformers with vector norms. In EMNLP, pages 7057–7075, 2020. 2, 4

[32] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Incorporating Residual and Normalization Layers into Analysis of Masked Language Models. In EMNLP, pages 4547–4568, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 2

[33] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In ICML, pages 3519–3529. PMLR, 2019. 4

[34] Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. Revealing the dark secrets of BERT. In EMNLP-IJCNLP, pages 4365–4374, 2019. 2, 4

[35] Matthew Kowal, Achal Dave, Rares Ambrus, Adrien Gaidon, Konstantinos G. Derpanis, and Pavel Tokmakov. Understanding video transformers via universal concept discovery. In CVPR, pages 10946–10956, 2024. 1, 2

[36] Maximilian Li and Lucas Janson. Optimal ablation for interpretability. In NeurIPS, pages 109233–109282. Curran Associates, Inc., 2024. 2

[37] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. MViTv2: Improved multiscale vision transformers for classification and detection. In CVPR, pages 4804–4814, 2022. 4

[38] Yiran Li, Junpeng Wang, Xin Dai, Liang Wang, Chin-Chia Michael Yeh, Yan Zheng, Wei Zhang, and Kwan-Liu Ma. How does attention work in vision transformers? a visual analytics attempt. IEEE TVCG, 29(6):2888–2900, 2023. 2

[39] Hanxiao Liu, Zi-Hang Dai, David R. So, and Quoc V. Le. Pay attention to mlps. In NeurIPS, 2021. 7

[40] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In CVPR, pages 3202–3211, 2022. 3

[41] Zhenyu Lu, Yuheng Jia, Wei You, and Hao Chen. Discovering attention head interactions in vision transformers. In CVPR, 2026. 2

[42] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. In NeurIPS, pages 4898–4906, 2016. 6

[43] Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. The hydra effect: Emergent selfrepair in language model computations, 2023. 2

[44] Leland McInnes, John Healy, and James Melville. UMAP: Uniform manifold approximation and projection for dimension reduction, 2020. 4

[45] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In NeurIPS, 2019. 2, 3

[46] Ali Modarressi, Mohsen Fayyaz, Yadollah Yaghoobzadeh, and Mohammad Taher Pilehvar. GlobEnc: Quantifying global token attribution by incorporating the whole encoder layer in transformers. In NAACL-HLT, pages 258–271, 2022. 2

[47] Sharon S. Musa, Fereshteh Forghani, Harrish Thasarathan, Sonia Joseph, Matthew Kowal, and Konstantinos G. Derpanis. What, where, and how: Probing spatiotemporal representations in video foundation models. arXiv preprint arXiv:2609.01551, 2026. 2

[48] Andrew Nam, Henry Conklin, Yukang Yang, Tom Griffiths, Jonathan D Cohen, and Sarah-Jane Leslie. Causal head gat ing: A framework for interpreting roles of attention heads in transformers. In NeurIPS, pages 115609–115631. Curran Associates, Inc., 2025. 2

[49] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3), 2020. 1

[50] Raquel Panadero Palenzuela, Dominik Schorkhuber, and¨ Margrit Gelautz. Importance-guided interpretability and pruning for video transformers in driver action recognition. In WACV, pages 5295–5304, 2025. 2

[51] Namuk Park, Wonjae Kim, Byeongho Heo, Taekyung Kim, and Sangdoo Yun. What do self-supervised vision transform ers learn? In ICLR, 2023. 2

[52] Hanchuan Peng, Fuhui Long, and Chris H. Q. Ding. Fea ture selection based on mutual information: Criteria of maxdependency, max-relevance, and min-redundancy. IEEE TPAMI, 27(8):1226–1238, 2005. 1

[53] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? In NeurIPS, 2021. 2, 4

[54] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 3

[55] Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, and Che Zheng. Synthesizer: Rethinking self-attention for transformer models. In ICML, pages 10183–10192. PMLR, 2021. 7

[56] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022. 2

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS. Curran Associates, Inc., 2017. 1

[58] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In ACL, pages 5797–5808, 2019. 2, 3

[59] Matthew Walmer, Saksham Suri, Kamal Gupta, and Abhinav Shrivastava. Teaching matters: Investigating the role of supervision in vision transformers. In CVPR, pages 7486– 7496, 2023. 2

[60] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. VideoMAE V2: Scaling video masked autoencoders with dual masking. In CVPR, pages 14549–14560, 2023. 2

[61] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. MeMViT: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In CVPR, pages 13587–13597, 2022. 1, 4

[62] Lu Yu and Wei Xiang. X-Pruner: explainable pruning for vision transformers. In CVPR, pages 24355–24363, 2023. 2

[63] Fred Zhang and Neel Nanda. Towards best practices of activation patching in language models: Metrics and methods, 2024. 3

[64] Yanyi Zhang, Xinyu Li, Chunhui Liu, Bing Shuai, Yi Zhu, Biagio Brattoli, Hao Chen, Ivan Marsic, and Joseph Tighe. VidTr: Video transformer without convolutions. In ICCV, pages 13577–13587, 2021. 1

[65] Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba. Temporal relational reasoning in videos. In ECCV, pages 803–818, 2018. 1

# Beyond Head Importance: Collective Computation in Video Transformers

Supplementary Material

## A. Experimental protocols

## A.1. Evaluation tasks and ablation protocol

We define four tasks: Action, Object, Direction, and Outcome. Action uses ten Something-Something V2 classes covering object motion, deformation, and manipulation (e.g., throwing, folding, and pouring), while Object uses nine instrument-related UCF101 activity classes. For both tasks, we sample 100 videos per class. Direction and Outcome are binary tasks constructed by merging original Something-Something V2 classes. Direction groups four classes involving pushing or pulling objects to the left or right according to motion direction. Outcome groups five types of actions (picking up, pouring out, attaching, tearing, and opening) and their corresponding pretend or failed-action classes into two categories: successful and unsuccessful execution. Both binary tasks contain 500 videos per class. All tasks are class-balanced, with videos split into training and test sets at an 18:7 ratio.

For each video, we average-pool the encoder’s output token representations to obtain a video-level feature. Our linear probe combines feature standardization with a linear classifier trained using SGD. We refit the probe under each intervention condition and evaluate accuracy on a fixed test set. Attention head ablation uses mean replacement: for each head, we compute its mean output over all training videos and token positions in the Intact model, then replace the ablated head’s output with this mean to suppress its sample- and position-specific information. For both Action and Object, we perform 3000 random joint ablation trials on each of the four models. In each trial, we randomly ablate 25% of the attention heads and record the accuracy drop relative to Intact.

## A.2. Predicting and evaluating head importance

For each model, we predict attention heads’ percentile importance ranks on Action using repeated six-fold cross-validation to evaluate generalization to unseen heads. Within each training fold, we select properties using maximum relevance minimum redundancy (mRMR) [52] and fit an ExtraTrees regressor [8, 22] with 200 extremely randomized trees. Selection and fitting use only trainingfold data; evaluation uses Spearman correlation between predicted and ground-truth importance ranks on the test fold.

Figure 3(a) evaluates prediction using one to ten properties over 10 cross-validation repetitions [20]. Mean Spearman correlation rises from 0.112 with one property to 0.282 with seven, then plateaus at 0.284 and 0.282 with eight and ten, respectively. We therefore use seven properties for permutation importance analysis.

Figure 3(b) assesses property contributions over 50 six-fold cross-validation repetitions, selecting seven properties per training fold. Each selected property is permuted ten times within the test fold; its contribution is the mean decrease in predictive Spearman correlation. Unselected properties receive zero importance, and results are aggregated across all ten candidates. Mean predictive correlation is 0.279, with a variation interval of [0.249, 0.307] across data splits.

## A.3. Self-supervised MLP fine-tuning

The self-supervised fine-tuning of the Bottom half retained model in Figure 4(d) uses the full SSv2 training set without action or object labels, retaining V-JEPA’s masked feature prediction objective, predictor, and exponential moving average (EMA) target encoder. We update only the MLP parameters in the context encoder and predictor, freezing all other parameters, including those of the attention modules, patch embedding, positional encodings, and normalization layers. Inputs are 16-frame clips cropped to 224 × 224, with masking and data augmentation following the original pretraining pipeline. We train for one epoch (2639 iterations) with a batch size of 64, warming up the learning rate to $1 . 5 \times 1 0 ^ { - 4 }$ over 0.1 epoch and then decaying it to $1 0 ^ { - 6 }$

## A.4. Perturbation responses at distant tokens

For Figure 5, we use a subset of 174 videos from Something-Something V2, with one video per class. We rank all 384 attention heads by their mean spatial attention distance $d _ { \mathrm { s } }$ and retain either the half with the shortest distances (Local half) or the half with the longest distances (Global half), comparing both with the unablated model (Intact).

For each sampled view $u ,$ let $\mathbf { z } _ { u }$ denote the input token sequence. We apply a small perturbation at the central position s, with magnitude $\epsilon _ { u } = \lambda \| \mathbf { z } _ { u } ( s ) \| _ { 2 }$ , where $\lambda = 0 . 0 2$ , so that the perturbation norm is 2% of the source token’s representation norm. We apply positive and negative perturbations along each of four unit directions and use central finite differences to measure the response at each output position. Let $F _ { \ell , u , k } ^ { ( c , \pm ) } ( i )$ denote the output representation at position i of ViT block ℓ under condition c after applying a positive or negative perturbation along direction $k ,$ where $c \in$ {Intact, Local, Global}. Averaging over the four perturbation directions and all sampled views gives the response magnitude:

$$
g _ { \ell } ^ { ( c ) } ( i ) = \frac { 1 } { 4 | \mathcal { U } | } \sum _ { u \in \mathcal { U } } \sum _ { k = 1 } ^ { 4 } \frac { \left\| F _ { \ell , u , k } ^ { ( c , + ) } ( i ) - F _ { \ell , u , k } ^ { ( c , - ) } ( i ) \right\| _ { 2 } } { 2 \epsilon _ { u } } ,\tag{2}
$$

Here, U denotes the set of all sampled views. This metric measures the mean representational response magnitude at each output position per unit input perturbation. To quantitatively compare responses at spatially distant positions, we focus on the time slice containing the perturbation source, $t = 3 .$ , and define B as the set of 52 distinct tokens on the outer boundary of its $1 4 \times 1 4$ spatial grid. We first compute the ratio of the mean response magnitude for Local half to that for Global half at each position, then average these ratios over all boundary positions:

$$
r _ { \ell } ( i ) = \frac { g _ { \ell } ^ { ( \mathrm { L o c a l } ) } ( i ) } { g _ { \ell } ^ { ( \mathrm { G l o b a l } ) } ( i ) } ,\tag{3}
$$

$$
R _ { \ell } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } r _ { \ell } ( i ) , \qquad | \mathcal { B } | = 5 2 .\tag{4}
$$

## B. Head property definitions

Drawing on prior work, we define ten attention head properties across four categories: spatiotemporal attention range, attention weight structure, head outputs and representational responses, and network position. These properties are summarized in Table 2. In what follows, h denotes an attention head, x an input video, and i and j the query and key positions, respectively. $\bar { A } _ { i j } ^ { ( h , x ) }$ denotes the softmax-normalized attention weight. $Q _ { x }$ is the set of query positions for video x, and $K _ { i }$ is the set of key positions accessible to query position i.

## B.1. Spatiotemporal attention range

We first quantify the spatiotemporal attention range of each head using attention-weighted mean distances. Let $\mathbf { p } _ { i }$ denote the pixel coordinates of the center of the image patch corresponding to token $i ,$ and let $t _ { i }$ denote its tubelet index. The mean spatial and temporal attention distances are defined, respectively, as

$$
d _ { \mathrm { s } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \sum _ { j \in K _ { i } } A _ { i j } ^ { ( h , x ) } \| { \bf p } _ { j } - { \bf p } _ { i } \| _ { 2 } ,\tag{5}
$$

$$
d _ { \mathrm { t } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \sum _ { j \in K _ { i } } A _ { i j } ^ { ( h , x ) } | t _ { j } - t _ { i } | .\tag{6}
$$

We also examine the distribution of attention weights across temporal positions and within local neighborhoods in the same temporal slice. The cross-temporal attention ratio $r _ { \mathrm { c r o s s } }$ measures the fraction of attention weight assigned to temporal slices other than the query’s own and is defined as

$$
r _ { \mathrm { c r o s s } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \sum _ { j \in K _ { i } } A _ { i j } ^ { ( h , x ) } \mathbb { 1 } [ t _ { j } \neq t _ { i } ] .\tag{7}
$$

The same-slice local attention ratio $r _ { \mathrm { l o c a l } }$ measures the fraction of attention weight assigned to the local spatial neighborhood within the query’s temporal slice and is defined as

$$
r _ { \mathrm { l o c a l } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \sum _ { j \in K _ { i } } A _ { i j } ^ { ( h , x ) } \mathbb { 1 } [ t _ { j } = t _ { i } ] \mathbb { 1 } [ \| \mathbf { q } _ { j } - \mathbf { q } _ { i } \| _ { 2 } \leq 1 ] ,\tag{8}
$$

Here, ${ \bf q } _ { i }$ denotes the integer patch-grid coordinates of token i. The local neighborhood is defined by a Euclidean radius of one grid unit, including the current position and its four immediate horizontal and vertical neighbors, but excluding diagonal neighbors.

## B.2. Attention weight structure

We use the convolutional approximation error $e _ { \mathrm { c o n v } }$ to quantify how closely a head’s mean attention weight distribution matches a spatiotemporal kernel shared across positions. First, we compute each head’s mean attention matrix over the training set, $\mathbf { \bar { \chi } } _ { \mathbf { \bar { A } } } ^ { \mathbf { ( } h \mathbf { ) } } ~ = ~ \mathbb { E } _ { x } [ A ^ { ( h , x ) } ]$ For each relative spatiotemporal offset $\delta = ( \Delta t , \Delta y , \Delta x )$ , we define the shared kernel $K _ { h } ( \delta )$ by averaging the attention weights over all valid query–key position pairs with that offset. We then reconstruct the attention matrix $\bar { A } _ { \mathrm { c o n v } } ^ { ( h ) }$ using zero padding and row-wise normalization, and define

$$
e _ { \mathrm { c o n v } } ^ { ( h ) } = \frac { \| A _ { \mathrm { c o n v } } ^ { ( h ) } - \bar { A } ^ { ( h ) } \| _ { F } } { \| \bar { A } ^ { ( h ) } \| _ { F } } .\tag{9}
$$

A smaller error indicates that the head’s mean attention distribution more closely resembles a 3D convolutional routing pattern shared across positions.

Query pattern consistency $c _ { \mathrm { q u e r y } }$ measures the consistency of a head’s relative attention distributions across query positions. For each query position i, we aggregate and normalize its attention weights by relative spatiotemporal offset δ to obtain a distribution $P _ { i }$ . We then compute the mean distribution across all query positions within each video, $\begin{array} { r } { \bar { P } = | Q _ { x } | ^ { - 1 } \sum _ { i \in Q _ { x } } P _ { i } } \end{array}$ . The metric is defined as

$$
c _ { \mathrm { q u e r y } } ^ { ( h ) } = 1 - \mathbb { E } _ { x } \frac { 1 } { \vert Q _ { x } \vert } \sum _ { i \in Q _ { x } } \mathrm { J S D } _ { 2 } ( P _ { i } , \bar { P } ) ,\tag{10}
$$

Here, $\mathrm { J S D _ { 2 } }$ denotes the Jensen–Shannon divergence computed using base-2 logarithms. Thus, $c _ { \mathrm { q u e r y } } \in [ 0 , 1 ]$ , with larger values indicating greater consistency of relative attention patterns across query positions.

## B.3. Outputs and representational responses

We quantify the effects of attention heads at different computational stages through the representational changes induced by head ablation. Let $R _ { \ell , i } ^ { \mathrm { a t t n } } ( x )$ and $R _ { \ell , i } ^ { \mathrm { m l p } } ( x )$ denote the token representations at position i for video x after the attention and MLP modules of Transformer block $\ell ,$ respectively. A tilde denotes the corresponding representation after ablating head h. The two ablationinduced representational responses are defined as

$$
\Delta _ { \mathrm { a t t n } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \left. R _ { \ell , i } ^ { \mathrm { a t t n } } ( x ) - \widetilde { R } _ { \ell , h , i } ^ { \mathrm { a t t n } } ( x ) \right. _ { 2 } ,\tag{11}
$$

$$
\Delta _ { \mathrm { m l p } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { | Q _ { x } | } \sum _ { i \in Q _ { x } } \left. R _ { \ell , i } ^ { \mathrm { m l p } } ( x ) - \widetilde { R } _ { \ell , h , i } ^ { \mathrm { m l p } } ( x ) \right. _ { 2 } .\tag{12}
$$

Both metrics use the same mean-replacement head ablation, measuring representational changes after the attention module and after the MLP module in the same block, respectively.

We also use the projected output norm $n _ { \mathrm { h e a d } }$ to measure the magnitude of each attention head’s output contribution. Let $O _ { h , i } ( x )$ denote the output of head h at position i before head concatenation, and let $W _ { O } ^ { ( h ) }$ denote the corresponding weight block in the multi-head attention output projection matrix. The head’s projected output at position i is $z _ { h , i } ^ { ( x ) } ~ = ~ O _ { h , i } ( x ) W _ { O } ^ { ( h ) }$ , with the shared output bias excluded from individual head contributions. We define

$$
n _ { \mathrm { h e a d } } ^ { ( h ) } = \mathbb { E } _ { x } \frac { 1 } { \left. Q _ { x } \right. } \sum _ { i \in Q _ { x } } \left. z _ { h , i } ^ { ( x ) } \right. _ { 2 } .\tag{13}
$$

## B.4. Network position

We use normalized depth to represent an attention head’s position within the network. For a model with $L = 2 4$ Transformer blocks, the normalized depth of a head in block ℓ is defined as

$$
b ^ { ( h ) } = \frac { \ell } { L } ,\tag{14}
$$

where $\ell \in \{ 1 , \ldots , L \}$

In the importance prediction experiments, all nine properties other than normalized depth b are converted to percentile ranks across all 384 attention heads within each model; b retains its orig inal definition.

## C. Cross-model results

## C.1. Head importance

Figure 7 shows the distribution of attention head importance on Action and Object across the four video Transformers. All models exhibit pronounced variation both within and across layers: heads in the same block differ in importance, while the mean importance within each block also varies with network depth. Highimportance heads tend to occur more frequently in the middle or middle-to-late layers, although their distribution varies across models and tasks. For example, in VideoMAE, mean head importance peaks in the middle layers on Action but in shallower layers on Object. These results indicate that attention head importance exhibits structure across layers, alongside variation within layers that cannot be captured by network depth alone.

Figure 8 provides additional cross-task cumulative ablation results for VideoMAE, V-JEPA 2, and VideoMAE V2. All curves use the importance ranking derived from Action, cumulatively ablating heads from either highest to lowest or lowest to highest importance. Ablating high-importance heads first causes accuracy to decline rapidly on multiple tasks. In contrast, ablating lowimportance heads first generally leads to a slower decline and can even improve performance on some tasks. This trend is consistent with the V-JEPA results in the main text, indicating that highimportance heads identified on a single task also play important roles in other video understanding tasks.

## C.2. Representational structure

Figure 9 provides additional representational analyses for Video-MAE, V-JEPA 2, and VideoMAE V2. Using the importance ranking derived from Action, we retain only the top or bottom 50% of attention heads and compare the resulting models with the intact model. The CKA results show that retaining high-importance heads better preserves representational similarity to Intact across multiple intermediate layers and at the final output layer. This indicates that the differences between high- and low-importance head sets are reflected not only in the final output, but also in the evolution of representations across layers.

The shared UMAP projections further illustrate how the two interventions affect the distribution of video representations. In VideoMAE, retaining only low-importance heads concentrates many video samples in a small region of the projection space, whereas retaining high-importance heads better preserves the original sample distribution. In contrast, V-JEPA 2 and VideoMAE V2 exhibit sample clustering under both retention conditions, with less pronounced differences in clustering than in VideoMAE. This suggests that clustering in two-dimensional projections alone is insufficient to explain the differences in task performance after ablating different head sets. Other representational properties and downstream computation must also be considered.

## C.3. Importance prediction

Figure 10 provides additional head importance prediction results for VideoMAE, V-JEPA 2, and VideoMAE V2. For all three models, predictive performance generally improves as more properties are included and largely plateaus by seven properties, yielding mean Spearman rank correlations of 0.356, 0.355, and 0.418, respectively. Including additional properties produces only small fluctuations, with no consistent improvement.

Permutation importance analysis shows that the mean spatial attention distance $d _ { \mathrm { s } }$ makes the largest contribution to prediction in all three models, although the contributions of the remaining properties differ. For VideoMAE, $d _ { \mathrm { s } }$ and the post-MLP ablationinduced representational response $\Delta _ { \mathrm { m l p } }$ contribute most, followed by the same-slice local attention ratio $r _ { \mathrm { l o c a l } }$ and query pattern consistency $c _ { \mathrm { q u e r y } }$ . For V-JEPA 2, $d _ { \mathrm { s } }$ is the dominant predictor, followed by $\Delta _ { \mathrm { m l p } }$ , with smaller contributions from $c _ { \mathrm { q u e r y } }$ and $r _ { \mathrm { l o c a l } }$ . For VideoMAE V2, $d _ { \mathrm { s } }$ contributes substantially more than any other property. These results identify spatial attention range as a shared predictive cue across the three models, with model-dependent contributions from post-MLP representational responses and attention weight structure.

## C.4. Attention range and collective contributions

Figure 11 shows the spatiotemporal attention ranges of heads and their relationships with importance across the four models. In each model, the mean attention distance within each block varies with network depth, with shorter distances typically occurring in the middle layers. Attention ranges also vary substantially among heads within the same block. Across all four models, mean spatial attention distance $d _ { \mathrm { s } }$ is negatively correlated with importance on Action, indicating that heads with shorter spatial attention ranges tend to have higher importance. In contrast, the relationship between mean temporal attention distance $d _ { \mathrm { t } }$ and importance is weaker and varies across models. These results complement the property-based importance prediction analysis in the main text and further illustrate the connection between spatial locality and head importance.

Figure 12 uses four attention heads from V-JEPA to illustrate the attention patterns associated with different spatial and tempo ral attention ranges. Temporally local heads concentrate attention mainly in the temporal diagonal blocks of the full attention matrix, while spatially local heads focus on the spatial neighborhood of the query position within the time slices they attend to. Spa tial and temporal attention ranges can combine in different ways, yielding four patterns: temporally global/spatially global, temporally local/spatially global, temporally global/spatially local, and temporally local/spatially local. For spatially local heads, visualizations at three different query positions further show that the region of concentrated attention shifts with the query position, reflecting local aggregation around each query.

Figure 13 compares cumulative ablation results for the other three models, with heads ranked by importance, spatial attention distance, or temporal attention distance. At 25% head ablation, ablating spatially local heads first causes greater Action performance degradation than ablating high-importance heads first in V-JEPA 2 and VideoMAE V2, and comparable degradation in VideoMAE. On Object, ablating spatially local head sets causes comparable or greater degradation in all three models. In contrast, ablating spatially or temporally global heads first generally slows perfor mance decline. Together with the main-text V-JEPA experiments, these results show that jointly ablating spatially local head sets substantially degrades performance across models and tasks, and that individual head importance rankings cannot fully predict their collective functional contributions.

## Supplementary figures and tables

Table 2. Ten computational properties of attention heads. Symbols, descriptions, and related work for each property. Detailed definitions are provided in Appendix B.
<table><tr><td>Category</td><td>Symbol</td><td>Description</td><td>References</td></tr><tr><td rowspan="4">Spatiotemporal attention range</td><td> $d _ { \mathrm { s } }$ </td><td>Attention-weighted mean spatial distance</td><td>[16]</td></tr><tr><td> $d _ { \mathrm { t } }$ </td><td>Attention-weighted mean temporal distance</td><td>[7]</td></tr><tr><td> $r _ { \mathrm { c r o s s } }$ </td><td>Fraction of attention weight assigned to other temporal slices</td><td>[61]</td></tr><tr><td> $r _ { \mathrm { l o c a l } }$ </td><td>Fraction of attention weight within the same-slice neighborhood</td><td>[37]</td></tr><tr><td rowspan="2">Attention weight structure</td><td> $e _ { \mathrm { c o n v } }$ </td><td>Deviation of the mean attention matrix from a shared kernel</td><td>[15]</td></tr><tr><td> $c _ { \mathrm { q u e r y } }$ </td><td>Consistency of relative attention patterns across query positions</td><td>[34]</td></tr><tr><td rowspan="3">Outputs and representational responses</td><td> $\Delta _ { \mathrm { a t t } }$ </td><td>Post-attention representational change induced by head ablation</td><td>[12]</td></tr><tr><td> $\Delta _ { \mathrm { m l p } }$ </td><td>Post-MLP representational change induced by head ablation</td><td>[1]</td></tr><tr><td> $n _ { \mathrm { h e a d } }$ </td><td>Norm of the head&#x27;s projected output</td><td>[31]</td></tr><tr><td>Network position</td><td> $^ { b }$ </td><td>Normalized depth of the block containing the head</td><td>[53]</td></tr></table>

![](images/e8099e8c3762a08751d20b9c6ff430041793ee40c13127701464186c847e631a.jpg)  
Figure 7. Head importance distributions on Action and Object across four models. Importance is measured as the mean accuracy drop across random joint ablation trials containing the head. In each heatmap, heads within each block are sorted by decreasing importance. The curves below show the mean importance within each block.

![](images/87c2d8d49757a3dd4883773d629d4bd17bfb33db2f59dffaf57ec21edfb44cb8.jpg)  
Figure 8. Cross-task cumulative ablation results across models. We perform cumulative ablation using the importance ranking derived from Action, ablating heads from highest to lowest importance (Top-first) or lowest to highest importance (Bottom-first).

![](images/86574e5d84e28ea720cbb9f6da0b55a5ff2270158b10f9ad76fcc2311946892b.jpg)  
Figure 9. Video representations with the top or bottom half of heads retained. The first two columns show cross-layer CKA matrices comparing Intact with models retaining only the top or bottom 50% of heads by importance. The last two columns show shared UMAP projections of final-layer video representations, with lines connecting the positions of the same video before and after intervention.

![](images/891d6f0a4ff1762e152b709a0590393a0761a5e6ec39ff17139d932f310c5b39.jpg)  
Figure 10. Head importance prediction and property contributions across models. The top row shows Spearman correlations between predicted and actual importance ranks for unseen heads as a function of the number of properties. Lines indicate means, and shaded regions show the 2.5th–97.5th percentile intervals across data splits. The bottom row shows permutation importance using seven properties, measured as the decrease in predictive correlation caused by randomly permuting a single property.

![](images/83e73b4a08d61b8cec4431d9cdb7aacab64d951ae394a6e04b0053c8faccd99f.jpg)  
Figure 11. Spatiotemporal attention range and head importance across models. The first two rows show the layerwise distributions of mean spatial attention distance $d _ { \mathrm { s } }$ and mean temporal attention distance $d _ { \mathrm { t } } ,$ , respectively. Points represent individual heads, and lines indicate the mean within each block. The third row shows the joint distribution of the two distances. Point colors indicate head importance on Action, measured as the mean accuracy drop across random joint ablation trials containing the head.

![](images/f94b9ce86d711514230bf6cc7887857bc55670589c72be68b9ccccb837291de4.jpg)  
Figure 12. Attention patterns across different spatiotemporal attention ranges. The four columns show V-JEPA heads that are tem porally global/spatially global, temporally local/spatially global, temporally global/spatially local, and temporally local/spatially local, respectively. The first row shows the full attention matrices, with queries and keys grouped by temporal slice in temporal order and spatia tokens within each slice flattened into a contiguous sequence. Grid lines mark boundaries between temporal slices. The remaining three rows show the attention distributions across temporal slices for three query positions, with green crosses marking the query positions.

![](images/f6c46ac8ccb87c66a8fe09eb10acf5810dda0ce80bcace14f0472398b70a027d.jpg)  
Figure 13. Cumulative ablation of head sets across models and tasks. Heads are ranked by importance on Action or by mean spatial or temporal attention distance. The first and third columns ablate high-importance or local heads first, while the second and fourth columns ablate low-importance or global heads first.