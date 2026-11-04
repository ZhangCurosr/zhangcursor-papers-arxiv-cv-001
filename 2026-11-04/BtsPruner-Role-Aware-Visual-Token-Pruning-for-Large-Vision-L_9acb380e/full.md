# BtsPruner: Role-Aware Visual Token Pruning for Large Vision-Language Models

Zihao Qi<sup>1,2∗</sup>, Zhibo Zhang<sup>1,2∗</sup>, Miaoxin Zhang<sup>2∗</sup>, Jin Cheng Cai<sup>2∗</sup>, Zengqiang Yan<sup>1,2†</sup>

<sup>1</sup>Nanjing University

<sup>2</sup>Huazhong University of Science and Technology qizihao@smail.nju.edu.cn, z\_yan@hust.edu.cn

## Abstract

Large Vision-Language Models (LVLMs) often process hundreds or thousands of visual tokens, resulting in substantial computational and memory overhead during inference. Existing training-free pruning methods typically estimate token importance using language-model attention, visual attention, or feature similarity. However, these methods treat visual tokens with uniform criteria and overlook that tokens may serve diferent roles during visual encoding. In this work, we investigate the evolution of visual token roles in LVLMs and observe that tokens progressively transition from local detail carriers to globally referenced representatives as encoding proceeds. Based on this observation, we propose BtsPruner, a trainingfree role-aware visual token pruning framework that preserves complementary token roles under a fixed budget. Specifically, BtsPruner first identifies the token-role transition point based on the evolution of attention concentration. It then discovers globally referenced representatives from later encoding stages and recursively retrieves local detail tokens from earlier stages to recover complementary fine-grained evidence. Extensive experiments across LLaVA-1.5, LLaVA-NeXT, Qwen-VL, InternVL, and CogVLM demonstrate that BtsPruner achieves superior performance retention under aggressive token reduction. On LLaVA-1.5-7B, BtsPruner preserves 91.6% of the dense model’s performance using only 32 of 576 visual tokens (5.6%). In the high-resolution LLaVA-NeXT-7B setting, it achieves up to a 5.95× CUDA-latency speedup.

Code — https://github.com/Halenbest/BtsPruner

## Introduction

Large vision-language models (LVLMs) extend large language models (LLMs) to multimodal understanding and reasoning (Bai et al. 2023b; Jiang et al. 2023; Touvron et al. 2023; Bai et al. 2023a; Cao et al. 2025; Chen et al. 2024c). A typical LVLM encodes an image into visual tokens, projects them into the language-model embedding space, and jointly processes them with textual tokens. However, visual tokens often dominate the input sequence, substantially increasing computation, key-value cache usage, and inference latency, especially in high-resolution scenarios.

Training-free visual token pruning provides a practical solution without updating model parameters. FastV (Chen et al. 2024b) and subsequent methods (Xing et al. 2024; Ye et al. 2024; Zhang et al. 2024) use text-to-visual attention in the language model, but such attention may shift or disperse across layers (Xu et al. 2025). Recent methods therefore perform pruning near the vision encoder using visual attention, feature similarity, or their combination (Shang et al. 2024; Yang et al. 2024; Alvar et al. 2025; Zhang et al. 2025). Nevertheless, most existing methods estimate token importance with a unified criterion and overlook that visual tokens may develop diferent roles during encoding, leading to the loss of either globally representative information or fine-grained local details under aggressive pruning as illustrated in Fig. 1.

![](images/0a25791b5ba4b35ca3d1410f782e86d98c55b3948ed92e940e663137e2d952ad.jpg)  
Figure 1: Qualitative comparison of visual token pruning strategies. Existing methods select tokens according to a unified criterion, while BtsPruner preserves complementary token roles under the same token budget.

To investigate this issue, we analyze the evolution of visual token roles during LVLM encoding. We observe that visual tokens gradually transition from local detail carriers to globally referenced representatives. In early encoding stages, tokens mainly interact with nearby or visually related regions and preserve localized visual details. As encoding progresses, a subset of tokens becomes increasingly shared across visual queries and provides broader contextual representations. This role evolution suggests that efective pruning should preserve complementary token roles rather than select tokens solely according to a single importance score.

Based on this observation, we propose BtsPruner, a training-free role-aware visual token pruning framework. BtsPruner first identifies the token role transition according to attention concentration changes. It then discovers globally referenced representatives through attention-guided selection and diversity filtering, and recursively retrieves local detail tokens to preserve complementary fine-grained evidence. By maintaining both token roles under a fixed budget, BtsPruner reduces redundant visual tokens while preserving visual coverage. We evaluate BtsPruner on LLaVA-1.5 (Liu et al. 2024a), LLaVA-NeXT (Liu et al. 2024b), Qwen-VL (Bai et al. 2023a), InternVL (Chen et al. 2024c), and CogVLM (Wang et al. 2025). Extensive experiments across multimodal benchmarks demonstrate that BtsPruner achieves superior performance retention under aggressive token compression. On LLaVA-1.5-7B, BtsPruner preserves 91.6% of the dense model’s performance using only 32 of 576 visual tokens (5.6%). In high-resolution LLaVA-NeXT-7B settings, it achieves up to a 5.95× CUDA-latency speedup. The main contributions are summarized as follows:

• We reveal the role evolution phenomenon of visual tokens in LVLM encoders, where tokens transition from local detail carriers to globally referenced representatives during encoding.

• We propose BtsPruner, a role-aware visual token pruning framework that discovers and preserves complementary token roles under a fixed token budget.

• Extensive experiments across multiple LVLM architectures and benchmarks demonstrate strong performance retention, particularly under extreme token compression.

## Related Work

## Large Vision-Language Models

Large vision-language models (LVLMs) extend large language models to multimodal understanding and generation tasks (Bai et al. 2023b; Ouyang et al. 2022; Achiam et al. 2023; Liu et al. 2024c; Team et al. 2023; Wang et al. 2025). Most LVLMs adopt a modular architecture consisting of a vision encoder, a multimodal projector, and a large language model. The vision encoder converts images into visual token sequences, while the projector aligns visual representations with the language-model embedding space. Common projector designs include linear projection (Liu et al. 2024c), MLP-based mapping (Liu et al. 2024a), and query-based modules (Bai et al. 2023a; Li et al. 2023a). Although increasing image resolution improves multimodal perception ability (Liu et al. 2024b), it also introduces substantially longer visual-token sequences, increasing computational and memory costs. Therefore, eficient processing of visual tokens has become an important problem for LVLM deployment.

## Visual Token Pruning in LVLMs

Visual token pruning aims to reduce redundant visual representations while preserving task-relevant information during LVLM inference. Existing approaches can be broadly categorized into training-based and training-free methods. Training-based methods introduce additional token compression or resampling modules (Cai et al. 2024; Chen et al. 2024a; Li et al. 2024), but require extra training costs and modifications to existing models. In contrast, training-free methods directly estimate token importance and can be applied to pretrained LVLMs.

Early training-free methods mainly perform pruning within the language model. FastV (Chen et al. 2024b) introduces text-to-visual attention-based pruning, followed by several approaches that adopt similar strategies (Xing et al. 2024; Ye et al. 2024; Zhang et al. 2024). However, languagemodel-side attention may sufer from attention shift and dispersion across layers (Xu et al. 2025). Recent studies therefore move pruning closer to the vision encoder and exploit visual attention, feature similarity, or their combination. DivPrune (Alvar et al. 2025) selects diverse tokens based on feature similarity, while PruMerge, VisionZip, and VisPruner (Shang et al. 2024; Yang et al. 2024; Zhang et al. 2025) combine attention-based importance estimation with similaritybased redundancy reduction.

Despite their efectiveness, existing training-free methods generally assume that visual tokens can be ranked using a unified importance criterion. Such strategies overlook that visual tokens may gradually develop diferent roles during encoding, where some tokens preserve local details while others become globally referenced representatives. This motivates role-aware visual token pruning that preserves complementary token roles rather than relying on a single importance estimation criterion.

## Visual Token Role Evolution in LVLMs

Existing visual token pruning methods usually estimate token importance using a unified criterion, assuming that visual tokens are interchangeable during encoding. However, visual tokens may gradually develop diferent functional roles as the representation evolves. In this section, we investigate the evolution of visual token roles during LVLM encoding and identify two complementary roles: local detail tokens, which preserve fine-grained visual evidence, and global representative tokens, which provide broader contextual coverage. We first introduce the self-attention mechanism and the statistics used for analysis, then analyze the role transition of visual tokens and discuss its implications for role-aware token pruning.

## Preliminaries

Most vision encoders used in LVLMs, such as CLIP (Radford et al. 2021), are based on the Transformer architecture. Given an input token sequence

$$
X = [ x _ { \mathrm { c l s } } ; x _ { v } ^ { 1 } , \ldots , x _ { v } ^ { n } ] \in \mathbb { R } ^ { ( n + 1 ) \times d } ,\tag{1}
$$

where $x _ { \mathrm { c l s } }$ denotes the [CLS] token, $x _ { v } ^ { i }$ denotes the i-th visual token, n is the number of visual tokens, and d is the hidden dimension, the query, key, and value representations are computed as

$$
Q = X W _ { Q } , \qquad K = X W _ { K } , \qquad V = X W _ { V } ,\tag{2}
$$

![](images/9ba8dbb8c12e00d08592f7680f7c4d761b3f79f1df9fad4158632698f0b53880.jpg)  
Figure 2: Visual token role evolution during LVLM encoding. Left: Attention patterns show a transition from localized interactions to shared representations. Right: Representative local detail and global representative tokens across diferent encoding stages.

where $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { d \times d }$ are learnable projection matrices. The scaled dot-product attention and its output are calculated as

$$
A = \mathrm { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) , \qquad O = A V .\tag{3}
$$

We refer to the first row of A as the [CLS] attention. By removing the [CLS] row and column, we obtain the visualtoken attention matrix $A ^ { \prime } \in \mathbb { R } ^ { n \times n }$ . The i-th row, $A _ { i , : } ^ { \prime }$ , represents the attention distribution of the i-th visual token over other visual tokens. To characterize whether a token gradually becomes a shared representation, we define the mean incoming attention of token j as

$$
\bar { a } _ { j } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } A _ { i j } ^ { \prime } ,\tag{4}
$$

where $\bar { a } _ { j }$ measures the degree to which token j is referenced by other visual-token queries. A higher value indicates that the token receives attention from more visual tokens and may serve as a shared representation. For multi-head selfattention, we average the attention matrices across all heads before analysis.

## Token Role Transition Analysis

To investigate whether visual tokens develop diferent roles during encoding, we extract attention scores from the vision encoder of LLaVA-1.5-7B (Liu et al. 2024a) using a representative example from VQAv2 (Goyal et al. 2017). Figure 2 presents the evolution of visual-token attention across diferent encoder layers. As illustrated in Fig. 2, in early encoding stages (e.g., Layer 9), the attention matrix exhibits localized diagonal structures, indicating that visual tokens mainly interact with nearby or visually related regions. We refer to these tokens as local detail tokens since they preserve spatially specific visual patterns. As encoding progresses, attention gradually becomes shared across queries and concentrates on a smaller set of commonly referenced positions in deeper layers (e.g., Layer 17). The attention statistics further support this transition. The [CLS] cumulative-attention curve changes from a uniform distribution to a concentrated distribution, and [CLS] attention exhibits similar peaks with mean incoming visual-token attention. Visualization in Fig. 2 further shows that token 284 becomes a global representative token, while tokens 175 and 109 preserve local details. These observations indicate that visual tokens evolve from local detail carriers to globally referenced representatives during encoding.

## Implications for Role-Aware Token Pruning

Attention weights do not provide a complete causal explanation of representation formation, as token representations are also afected by value projections, residual connections, and feed-forward networks. Nevertheless, self-attention reflects token interactions within Transformer blocks. The observed role evolution suggests that visual token pruning should preserve complementary roles rather than rank all tokens using a single criterion. Selecting only global representative tokens may lose fine-grained details, while selecting only local detail tokens may introduce redundancy and insuficient global context. Therefore, efective pruning should preserve both token roles under a limited budget. This motivates BtsPruner to discover global representative tokens and retrieve complementary local detail tokens for role-aware visual token compression.

## Method

In this section, we introduce BtsPruner, a training-free roleaware visual token pruning framework. As illustrated in Fig. 3, BtsPruner first identifies the transition of token roles during encoding. It then discovers global representative tokens through attention-based selection and diversity filtering, and recursively retrieves local detail tokens to preserve complementary visual evidence. Finally, tokens with diferent roles are combined for inference.

![](images/00c6c46314fb82dd4b1ca58b3f3c10c9e231af4cdc44fe3d65dc1f08da16cff0.jpg)  
Figure 3: Overview of BtsPruner. BtsPruner first identifies the token role transition during visual encoding, then discovers global representative tokens and recursively recovers local detail tokens through a shared diversity selection module. The selected tokens with complementary roles are combined for eficient LVLM inference.

## Role Transition Identification

Visual tokens gradually evolve from local detail carriers to global representatives during encoding. Therefore, BtsPruner first identifies the transition point between these roles through a token role transition identification module (TRIM). As shown in Fig. 2, the [CLS] cumulative attention is relatively uniform in earlier layers but becomes concentrated on several tokens in later layers. We use this change in attention concentration to identify the transition point. At layer i, the [CLS] attention is denoted as ${ \mathrm { C L S } } _ { i } \in { \dot { \mathbb { R } } } ^ { n }$ . We calculate its range as:

$$
{ \mathrm { r a n g e } } _ { i } = { \mathrm { m a x } } ( { \mathrm { C L S } } _ { i } ) - { \mathrm { m i n } } ( { \mathrm { C L S } } _ { i } )\tag{5}
$$

The change between adjacent layers is defined as:

$$
\Delta _ { i } = \mathrm { r a n g e } _ { i } - \mathrm { r a n g e } _ { i - 1 }\tag{6}
$$

Then, the transition point is determined by:

$$
\hat { i } = \{ i | \Delta _ { i } \geq ( \mathrm { C L S } _ { m a x } - \mathrm { C L S } _ { m i n } ) / U \} ,\tag{7}
$$

where $\mathrm { C L S } _ { m a x }$ and CL $ { \mathrm { S } } _ { m i n }$ denote the maximum and minimum values of [CLS] attention across all encoder layers, respectively. U controls the threshold and is set to 2 by default. The layers before $\widehat { i }$ are used for local detail recovery, while later layers are used for global representative discovery. For vision encoders without a [CLS] token, the mean incoming attention of visual tokens is used instead.

## Global Representative Discovery

As encoding progresses, some visual tokens become commonly referenced by multiple visual queries and provide global contextual representations. BtsPruner discovers these global representatives using attention-guided selection and diversity filtering. Given pruning ratio ${ \check { R } } ,$ we first select candidate tokens according to the attention of the second-to-last encoding layer. Given $Z _ { v }$ as the visual token features output by the vision encoder, we select candidate tokens as:

$$
Z _ { \mathrm { g l o b a l } } ^ { c a n d } = \mathrm { T o p K } ( Z _ { v } , \mathrm { C L S _ { - 2 } } ) ,\tag{8}
$$

where $\mathrm { C L S _ { - 2 } }$ denotes the [CLS] attention from the secondto-last layer, and $K = n ( 1 - R )$ is the number of selected candidate tokens. Following previous studies (Yang et al. 2024; Zhang et al. 2025), we use the second-to-last layer as the default representative discovery layer. The candidate tokens are further filtered by the diversity selection module:

$$
P = \mathrm { D i v e r s e } ( Z _ { \mathrm { g l o b a l } } ^ { c a n d } , N _ { \mathrm { g l o b a l } } , T ) ,\tag{9}
$$

where $P$ denotes the selected global representative token set, and $T$ denotes the number of candidate tokens transferred in each iteration of the diversity selection process. $N _ { \mathrm { g l o b a l } }$ is defined as $n ( 1 - R ) r$ where $n ( 1 - R )$ represents the total number of retained tokens and r determines the proportion allocated to global representatives. $N _ { \mathrm { g l o b a l } }$ is also the target size $D _ { t }$ of the diversity selection module. The selected indices form the global representative set $I _ { g }$

<table><tr><td>Method</td><td>VQAv2 VizWiz SQAIMG</td><td>GQA</td><td></td><td></td><td>VQAText</td><td>POPE</td><td>MME</td><td>MMB</td><td>MMBCN</td><td>MMVet</td><td>Avg.</td><td>Rel. (%)</td></tr><tr><td colspan="9">Upper Bound (576 tokens)</td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-1.5-7B</td><td>78.5</td><td>62.0</td><td>50.0</td><td>66.8</td><td>58.2</td><td>85.9</td><td>1510.7</td><td>64.3</td><td>58.3</td><td>31.1</td><td>|63.1</td><td>100.0%</td></tr><tr><td colspan="10">Retain 128 Tokens (↓ 77.8%)</td><td></td><td></td><td></td></tr><tr><td>FastV (ECCV24)</td><td>73.2</td><td>55.8</td><td>51.5</td><td>68.2</td><td>56.9</td><td>68.8</td><td>1442.3</td><td>63.2</td><td>56.7</td><td>28.0</td><td>59.4</td><td>94.7%</td></tr><tr><td>SparseVLM (ICML25)</td><td>76.2</td><td>58.4</td><td>50.0</td><td>65.2</td><td>56.8</td><td>85.2</td><td>1380.4</td><td>63.7</td><td>58.1</td><td>28.3</td><td>61.1</td><td>96.7%</td></tr><tr><td>DivPrune (CVPR25)</td><td>75.7</td><td>59.3</td><td>53.3</td><td>68.0</td><td>55.5</td><td>84.3</td><td>1407.7</td><td>61.1</td><td>54.4</td><td>28.7</td><td>61.1</td><td>96.8%</td></tr><tr><td>PruMerge+ (2024.05)</td><td>74.9</td><td>57.9</td><td>52.9</td><td>65.2</td><td>54.8</td><td>84.2</td><td>1402.5</td><td>60.2</td><td>55.6</td><td>30.0</td><td>60.6</td><td>96.3%</td></tr><tr><td>VisionZip (CVPR25)</td><td>75.3</td><td>57.5</td><td>52.6</td><td>68.4</td><td>56.3</td><td>83.0</td><td>1434.4</td><td>62.3</td><td>56.2</td><td>29.8</td><td>61.3</td><td>97.4%</td></tr><tr><td>VisPruner (ICCV25) BtsPruner (Ours)</td><td>75.7</td><td>58.2</td><td>52.6 52.7</td><td>69.0</td><td>57.2</td><td>84.3</td><td>1430.7</td><td>60.9</td><td>55.4</td><td>30.6</td><td>61.5</td><td>97.8%</td></tr><tr><td></td><td>76.1</td><td>58.5</td><td></td><td>69.0</td><td>57.7</td><td>84.8</td><td>1439.2</td><td>61.4</td><td>56.3</td><td>31.6</td><td>62.0</td><td>98.7%</td></tr><tr><td colspan="10">Retain 64 Tokens (↓ 88.9%)</td><td rowspan="2"></td></tr><tr><td>FastV (ECCV24)</td><td>66.3</td><td>51.5</td><td>51.6</td><td>68.0</td><td>55.0</td><td>64.1</td><td>1235.1</td><td>60.2</td><td>52.7</td><td>28.2</td><td>55.9</td></tr><tr><td>SparseVLM (ICML25) DivPrune (CVPR25)</td><td>70.2</td><td>53.8</td><td>50.1</td><td>66.3</td><td>53.4</td><td>78.4</td><td>1252.3</td><td>59.4</td><td>52.5</td><td>24.1</td><td>57.1</td><td>90.2%</td></tr><tr><td>PruMerge+ (2024.05)</td><td>73.9</td><td>57.4</td><td>53.8</td><td>67.9</td><td>54.9</td><td>81.0</td><td>1357.6</td><td>59.2</td><td>51.6</td><td>27.7</td><td>59.5</td><td>94.4%</td></tr><tr><td>VisionZip (CVPR25)</td><td>71.8</td><td>55.5</td><td>53.4</td><td>65.0</td><td>54.0</td><td>80.1</td><td>1348.5</td><td>58.2</td><td>53.0</td><td>26.7</td><td>58.5</td><td>92.8%</td></tr><tr><td></td><td>72.1</td><td>55.0</td><td>53.1</td><td>68.8</td><td>55.2</td><td>77.0</td><td>1353.1</td><td>59.4</td><td>54.6</td><td>29.7</td><td>59.3</td><td>94.5%</td></tr><tr><td>VisPruner (ICCV25) BtsPruner (Ours)</td><td>72.7</td><td>56.0</td><td>53.4 53.4</td><td>68.5</td><td>55.8</td><td>80.1</td><td>1365.9</td><td>58.6</td><td>54.3</td><td>30.1</td><td>59.8</td><td>95.3%</td></tr><tr><td></td><td>73.4</td><td>56.3</td><td></td><td>69.6</td><td>56.3</td><td>80.2</td><td>1364.4</td><td>60.1</td><td>54.6</td><td>31.4</td><td>60.4</td><td>96.4%</td></tr><tr><td colspan="10">Retain 32 Tokens (↓ 94.4%)</td></tr><tr><td>FastV (ECCV24)</td><td>56.9</td><td>46.7</td><td>49.2</td><td>66.2</td><td>51.9</td><td>57.3</td><td>995.5</td><td>51.2</td><td>43.6</td><td>20.6</td><td>49.3</td><td>78.8%</td></tr><tr><td>SparseVLM (ICML25)</td><td>64.5</td><td>50.2</td><td>48.1</td><td>66.9</td><td>49.9</td><td>70.7</td><td>1130.9</td><td>56.8</td><td>46.3</td><td>20.7</td><td>53.1</td><td>83.7%</td></tr><tr><td>DivPrune (CVPR25)</td><td>70.1</td><td>53.9</td><td>52.9</td><td>67.3</td><td>52.4</td><td>78.5</td><td>1301.4</td><td>56.9</td><td>49.2</td><td>26.2</td><td>57.2</td><td>90.7%</td></tr><tr><td>PruMerge+ (2024.05)</td><td>66.6</td><td>51.7</td><td>52.8</td><td>65.6</td><td>52.2</td><td>74.6</td><td>1203.1</td><td>55.6</td><td>48.5</td><td>25.5</td><td>55.3</td><td>88.0%</td></tr><tr><td>VisionZip (CVPR25)</td><td>66.8</td><td>51.6</td><td>52.4</td><td>68.7</td><td>52.7</td><td>68.3</td><td>1229.3</td><td>56.2</td><td>50.4</td><td>27.3</td><td>55.6</td><td>88.9%</td></tr><tr><td>VisPruner (ICCV25)</td><td>67.6</td><td>52.0</td><td>52.5</td><td>69.1</td><td>53.5</td><td>72.9</td><td>1258.6</td><td>56.8</td><td>48.5</td><td>25.1</td><td>56.1</td><td>89.1%</td></tr><tr><td>BtsPruner (Ours)</td><td>69.0</td><td>53.6</td><td>53.1</td><td>69.5</td><td>54.5</td><td>74.3</td><td>1292.2</td><td>58.0</td><td>51.6</td><td>27.1</td><td>57.5</td><td>91.6%</td></tr></table>

Table 1: Comparison of visual token pruning methods on LLaVA-1.5-7B across three pruning rates. Avg. denotes the normalized average score across benchmarks, and Rel. denotes the relative performance retention ratio compared with the dense model.

## Diversity Selection Module

The diversity selection module is shared by global representative discovery and local detail recovery. Given candidate features, we first initialize a selected set S and remaining set R. The cosine similarity between selected and remaining features is calculated, and candidates with the lowest maximum similarity to S are iteratively transferred into S. This process continues until the selected token number reaches the predefined target size $D _ { t }$

## Local Detail Recovery

Although global representatives provide broad contextual coverage, they may omit fine-grained visual details. Therefore, BtsPruner further retrieves complementary local detail tokens. Starting from laye $\widehat { i } - 1$ , the selected global representatives $I _ { g }$ are treated as parent tokens. For each parent token, we select the top-4b candidates according to its attention distribution:

$$
{ \cal Z } _ { \mathrm { l o c a l } } ^ { c a n d } = \mathrm { T o p } _ { 4 b } ( { \cal Z } _ { v } , A _ { p } ) ,\tag{10}
$$

where $A _ { p }$ denotes the attention row of its parent token. The candidate tokens are then filtered using the diversity selection module:

$$
C = { \mathrm { D i v e r s e } } ( Z _ { \mathrm { l o c a l } } ^ { c a n d } , b , T ) ,\tag{11}
$$

where b denotes the number of local detail tokens assigned to each parent token, and C represents the local detail tokens selected from the candidates of the current parent token. The selected local detail tokens are added to $I _ { l }$ and become parent tokens for the next recovery step. This recursive process continues until the target token budget $N _ { k e e p }$ is reached or the earliest layer is reached.

## Inference with Role-Aware Token Pruning

The final selected indices are obtained by combining global and local tokens:

$$
I = \mathrm { s o r t } ( \mathrm { u n i q u e } ( I _ { g } \cup I _ { l } ) ) , \qquad H _ { v } ^ { \prime } = Z _ { v } [ I ] ,\tag{12}
$$

where unique(·) removes duplicated token indices from the two selected sets, and sort(·) arranges the remaining indices according to their original order in the visual token sequence. The selected visual tokens are first mapped into the languagemodel embedding space through the multimodal projector $W$ , and then concatenated with the text prompt $H _ { q }$ before being fed into the language model $f _ { \phi }$

$$
H _ { v } = W ( H _ { v } ^ { \prime } ) , \qquad X _ { a } = f _ { \phi } ( H _ { v } \oplus H _ { q } ) ,\tag{13}
$$

where W denotes the multimodal projector that aligns visual token representations with the language-model embedding space, and $f _ { \phi }$ denotes the language model parameterized by ϕ. By preserving complementary visual token roles, BtsPruner achieves efective token compression while maintaining visual evidence for multimodal reasoning.

## Experiments

## Experimental Setup

Datasets. We evaluate BtsPruner on ten widely used multimodal benchmarks, including VQAv2 (Goyal et al. 2017), GQA (Hudson and Manning 2019), VizWiz (Gurari et al. 2018), ScienceQA-IMG (Lu et al. 2022), TextVQA (Singh et al. 2019), POPE (Li et al. 2023b), MME (Fu et al. 2026), MMBench (Liu et al. 2025), MMBench-CN (Liu et al. 2025), and MMVet (Yu et al. 2023). All evaluations follow the oficial settings and metrics.

Model Architectures. We evaluate BtsPruner on LLaVA-1.5 (Liu et al. 2024a), LLaVA-NeXT (Liu et al. 2024b) for high-resolution scenarios, and other LVLM architectures including Qwen-VL (Bai et al. 2023a), InternVL (Chen et al. 2024c), and CogVLM (Wang et al. 2025). All models follow their original evaluation settings.

Comparison Methods. We compare with representative training-free pruning methods, including language-modelside attention methods (FastV (Chen et al. 2024b), Sparse-VLM (Zhang et al. 2024)), similarity-based methods near the vision encoder (DivPrune (Alvar et al. 2025)), and attentionsimilarity hybrid methods (PruMerge (Shang et al. 2024), VisionZip (Yang et al. 2024), VisPruner (Zhang et al. 2025)).

Implementation Details. All experiments are conducted on NVIDIA GeForce RTX 3090 GPUs. The ratio between global representatives and local detail tokens, r, is set to 0.5 by default. We further investigate the influence of r through ablation experiments.

## Main Results

We first evaluate BtsPruner on LLaVA-1.5-7B. Table 1 compares diferent pruning methods under three token budgets. BtsPruner achieves the highest Avg. and Rel. across all evaluated budgets. When retaining 128 tokens, BtsPruner preserves 98.7% of the dense model performance. With only 64 retained tokens, it still maintains 96.4% performance retention. Under extreme pruning with only 32 retained tokens (5.6% of the original visual tokens), BtsPruner preserves 91.6% of the dense model performance. Compared with the strongest baseline VisPruner, BtsPruner improves relative performance retention by 0.9, 1.1, and 2.5 percentage points under 128, 64, and 32 tokens, respectively. Such results confirm that preserving complementary visual token roles is particularly efective under limited token budgets.

## BtsPruner in High-Resolution Settings

High-resolution LVLMs generate substantially longer visualtoken sequences, making token pruning particularly important. We apply BtsPruner to LLaVA-NeXT-7B, which processes 2880 visual tokens. As stated in Table 2, BtsPruner achieves the highest performance retention across all pruning ratios. When retaining 640 tokens, BtsPruner preserves 99.1% of the dense model performance. Even with only 160 retained tokens, it maintains 91.3% performance retention, demonstrating that preserving complementary token roles remains efective under high-resolution token compression.

<table><tr><td>Method</td><td colspan="5">|VQAv2 GQA VQAText POPE MME |Avg. Rel. (%)</td></tr><tr><td colspan="5">Upper Bound (2880 tokens)</td><td></td></tr><tr><td>LLaVA-1.6-7B</td><td>81.3 62.4</td><td>60.2</td><td>86.7</td><td>1511.8|</td><td>|73.2 100.0%</td></tr><tr><td colspan="6">Retain 640 Tokens (↓ 77.8%)</td></tr><tr><td>FastV SparseVLM DivPrune VisionZip</td><td>78.6 60.0 77.4 59.8 77.1 59.8 77.0 58.6</td><td>59.0 57.0 54.5 56.1</td><td>72.9 81.0 82.4 82.5</td><td>1467.6| 1457.5 1444.2 1388.8</td><td>68.8 94.4% 69.6 95.1% 69.2 94.3% 68.7 93.8%</td></tr><tr><td colspan="6">VisPruner 78.8 61.3 60.1 85.7 1484.2| Ours 79.3 61.6 60.7 86.6 1488.5</td></tr><tr><td>FastV SparseVLM DivPrune 57.9 53.5</td><td>Retain 320 Tokens (↓ 73.5 56.0 72.1 54.4 75.3</td><td>56.4 52.8</td><td>88.9%) 69.0 73.3 80.7</td><td>1360.4 1334.6| 1393.3</td><td>|64.6 88.7% 63.9 87.3% 67.4 91.9%</td></tr><tr><td colspan="6">VisionZip VisPruner 75.8 59.0 58.2 81.2 1405.8 Ours 76.7 59.5 59.7 82.3 1451.5</td></tr><tr><td></td><td></td><td>Retain 160 Tokens (↓ 94.4%)</td><td></td><td></td><td></td></tr><tr><td>FastV SparseVLM</td><td>63.1 50.7 66.1 51.4</td><td>52.9 47.8</td><td>62.0 64.7</td><td>1123.3| 1215.3|</td><td>57.0 78.5% 58.1 79.6%</td></tr><tr><td>DivPrune</td><td>73.2 56.9</td><td>52.5</td><td>78.6</td><td>1355.2</td><td>65.8 89.7%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionZip</td><td>71.1</td><td>53.8</td><td></td><td></td><td>62.9</td></tr><tr><td></td><td>54.5</td><td></td><td>78.1</td><td>1144.5</td><td>86.0%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>71.6</td><td>56.0</td><td></td><td></td><td></td></tr><tr><td>VisPruner</td><td></td><td></td><td>75.3</td><td>1285.3</td><td>64.6</td></tr><tr><td></td><td>56.0</td><td></td><td></td><td></td><td>88.5%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>57.7</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>1358.9|</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>91.3%</td></tr><tr><td></td><td></td><td></td><td>77.5</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>73.4</td><td>56.7</td><td></td><td></td><td>66.6</td></tr><tr><td></td></table>

Table 2: Performance comparison of visual token pruning methods on LLaVA-1.6-7B, i.e., LLaVA-NeXT-7B, across three pruning rates. Avg. denotes the normalized average score across benchmarks, and Rel. denotes the relative performance retention ratio compared with the dense model.

## BtsPruner across Diverse LVLM Architectures

To evaluate the general applicability of BtsPruner, we further apply it to LVLMs with diferent projector designs, including Qwen-VL (Bai et al. 2023a), InternVL (Chen et al. 2024c), and CogVLM (Wang et al. 2025) as summarized in Table 3. These models adopt diferent visual-language alignment strategies, providing a comprehensive evaluation beyond the LLaVA series.

For Qwen-VL, which employs a query-based projector, we introduce a lightweight adaptation. Specifically, after pruning visual tokens with BtsPruner, we select query tokens according to their attention responses over the retained visual tokens. For InternVL and CogVLM, which use MLP-based projectors, BtsPruner is directly applied after the vision encoder. Across various architectures, BtsPruner achieves the highest Avg. and Rel. among comparison methods, demonstrating that the proposed role-aware token preservation strategy generalizes across diferent LVLM architectures.

## Ablation Study

We conduct ablation experiments to evaluate the contribution of each component in BtsPruner and analyze the influence of the global-representative token ratio r. Table 4 presents the component ablation results on TextVQA using LLaVA-1.5-7B, where r = 0.5 and 64 visual tokens are retained. Starting from random selection, introducing global representative discovery improves the Avg. from 48.5 to 54.7, demonstrating the importance of preserving globally referenced tokens. Adding local detail recovery further improves the performance to 55.3, indicating that local details provide complementary visual evidence. TRIM and Diverse bring additional improvements, and their combination achieves the best performance of 56.3. These results verify the efectiveness of each component in BtsPruner.

<table><tr><td rowspan=1 colspan=2>Method  VQAv2GQA VizWizSQAVQAText</td><td rowspan=1 colspan=1>Avg.Rel. (%)</td></tr><tr><td rowspan=1 colspan=3>Qwen-VL-7B (128/256, ↓ 50%)</td></tr><tr><td rowspan=3 colspan=1>Qwen-VLFastVDivPruneVisPrunerOurs</td><td rowspan=1 colspan=1>78.8  59.3  35.2  67.1   63.8</td><td rowspan=2 colspan=1>60.8 100.0%58.1 96.0%</td></tr><tr><td rowspan=1 colspan=1>73.3  55.2  35.2  67.3   59.7</td></tr><tr><td rowspan=1 colspan=1>77.8  57.8  33.0  67.7   56.677.1  57.6  30.9  67.7   55.478.1  58.4  33.2  68.4   61.2</td><td rowspan=1 colspan=1>58.6 95.9%57.7 94.1%59.9 98.0%</td></tr><tr><td rowspan=1 colspan=3>InternVL-13B (144/576, ↓ 75%)</td></tr><tr><td rowspan=3 colspan=1>InternVLFastVDivPruneVisPrunerOurs</td><td rowspan=1 colspan=1>79.3  62.9  52.6  66.3   57.0</td><td rowspan=1 colspan=1>63.6 100.0%</td></tr><tr><td rowspan=1 colspan=1>74.3  58.4  51.4  66.6   55.6</td><td rowspan=1 colspan=1>61.3 96.5%</td></tr><tr><td rowspan=1 colspan=1>76.4  60.6  54.5  67.2   54.476.2  60.4  53.2  67.2   54.577.1  60.7  53.6  67.4   55.4</td><td rowspan=1 colspan=1>62.6 98.6%62.3 98.0%62.8 98.9%</td></tr><tr><td rowspan=1 colspan=3>CogVLM-7B (123/1225, ↓90%)</td></tr><tr><td rowspan=2 colspan=1>CogVLMFastV</td><td rowspan=1 colspan=1>80.0  59.4  38.9  70.5   69.2</td><td rowspan=3 colspan=1>63.8 100.0%44.1 68.6%59.3 92.7%</td></tr><tr><td rowspan=1 colspan=1>59.7  42.2  25.4  66.6   26.4</td></tr><tr><td rowspan=2 colspan=1>DivPruneVisPrunerOurs</td><td rowspan=2 colspan=1>76.4  54.8  35.4  70.1   59.772.0  52.2  32.2  68.9   52.975.8  52.5  35.8  70.4   63.2</td></tr><tr><td rowspan=1 colspan=1>55.6 86.8%59.5 93.1%</td></tr></table>

Table 3: Performance across LVLM architectures. Avg. denotes the normalized average score across benchmarks, and Rel. denotes the relative performance retention ratio compared with the dense model.

<table><tr><td rowspan=1 colspan=1>Global Rep. |</td><td rowspan=1 colspan=1>Local Detail</td><td rowspan=1 colspan=1>| TRIM |</td><td rowspan=1 colspan=1>Diverse |</td><td rowspan=1 colspan=1>Avg.</td></tr><tr><td rowspan=1 colspan=1>Random</td><td rowspan=1 colspan=1>Random</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1>48.5</td></tr><tr><td rowspan=2 colspan=1>Our</td><td rowspan=1 colspan=1>Random</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>54.7</td></tr><tr><td rowspan=1 colspan=1>Our</td><td rowspan=1 colspan=1>X√×√</td><td rowspan=1 colspan=1>XX√√</td><td rowspan=1 colspan=1>55.355.656.156.3</td></tr></table>

Table 4: Ablation study of BtsPruner components on TextVQA. Global Rep. and Local Detail denote global representative discovery and local detail recovery, respectively.

We further investigate the influence of the role allocation ratio r. Following the same setting, we evaluate r ∈ {0.25, 0.5, 0.75} under diferent token budgets. As stated in Table 5, BtsPruner remains stable across diferent values of r and consistently outperforms VisPruner. This indicates that preserving both global representatives and local details is more efective than relying on a single token selection strategy.

## Eficiency Analysis

To evaluate the computational eficiency of BtsPruner, we compare it with FastV, DivPrune, and VisPruner on LLaVA-

<table><tr><td># Token</td><td>r</td><td>VisPruner</td><td>BtsPruner</td></tr><tr><td rowspan="3">128</td><td>0.25</td><td>56.7</td><td>57.7</td></tr><tr><td>0.50</td><td>57.2</td><td>57.7</td></tr><tr><td>0.75</td><td>56.8</td><td>57.3</td></tr><tr><td rowspan="3">64</td><td>0.25</td><td>55.7</td><td>56.3</td></tr><tr><td>0.50</td><td>55.8</td><td>56.3</td></tr><tr><td>0.75</td><td>56.0</td><td>56.2</td></tr><tr><td rowspan="3">32</td><td>0.25</td><td>53.2</td><td>53.5</td></tr><tr><td>0.50</td><td>53.5</td><td>54.5</td></tr><tr><td>0.75</td><td>53.3</td><td>54.0</td></tr></table>

Table 5: Influence of the global-representative token ratio r on TextVQA performance.

<table><tr><td>Method</td><td># Token</td><td>FLOPs ↓(T)</td><td>Cache ↓(MB)</td><td>GPU Mem. ↓(GB)</td><td>CUDA Lat. ↓(ms)</td></tr><tr><td>LLaVA-NeXT</td><td>2880</td><td>40.8</td><td>1471</td><td>17.8</td><td>821</td></tr><tr><td>FastV</td><td>640</td><td>13.2</td><td>370</td><td>17.6</td><td>332</td></tr><tr><td>DivPrune</td><td>640</td><td>11.4</td><td>351</td><td>15.2</td><td>282</td></tr><tr><td>VisPruner</td><td>640</td><td>11.1</td><td>351</td><td>15.5</td><td>264</td></tr><tr><td>Ours</td><td>640</td><td>11.1</td><td>351</td><td>15.5</td><td>270</td></tr><tr><td>FastV</td><td>160</td><td>7.4</td><td>125</td><td>17.6</td><td>207</td></tr><tr><td>DivPrune</td><td>160</td><td>4.9</td><td>111</td><td>14.7</td><td>157</td></tr><tr><td>VisPruner</td><td>160</td><td>4.8</td><td>111</td><td>15.4</td><td>145</td></tr><tr><td>Ours</td><td>160</td><td>4.8</td><td>111</td><td>15.4</td><td>138</td></tr></table>

Table 6: Eficiency comparison on LLaVA-NeXT-7B using the POPE benchmark. Latency is measured as CUDA inference latency on a single RTX 3090 GPU.

NeXT-7B in terms of FLOPs, cache usage, GPU memory, and CUDA latency. The experiments are conducted on a single NVIDIA GeForce RTX 3090 GPU with the POPE benchmark. As stated in Table 6, when retaining only 160 visual tokens, BtsPruner reduces CUDA latency from 821 ms to 138 ms, achieving a 5.95× speedup. Compared with existing pruning methods, BtsPruner introduces negligible additional overhead while maintaining better role preservation and performance retention under aggressive token reduction.

## Conclusion

We propose BtsPruner, a training-free role-aware visual token pruning framework for LVLMs. By analyzing the evolution of visual token roles during encoding, we observe that visual tokens gradually transition from local detail carriers to globally referenced representatives. Based on this observation, BtsPruner identifies the role transition point, discovers global representative tokens, and recursively retrieves complementary local detail tokens under a fixed token budget. Extensive experiments across multiple LVLM architectures, multimodal benchmarks, and pruning ratios demonstrate that BtsPruner efectively preserves complementary visual token roles under aggressive compression. Without training, BtsPruner retains over 91% of the dense model performance while keeping only 5.6% of the visual tokens, and achieves a 5.95× CUDA-latency speedup in high-resolution settings.

## References

Achiam, J.; et al. 2023. GPT-4 Technical Report. arXiv:2303.08774.

Alvar, S. R.; Singh, G.; Akbari, M.; and Zhang, Y. 2025. DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models. In CVPR, 9392–9401.

Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin, J.; Zhou, C.; and Zhou, J. 2023a. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv:2308.12966.

Bai, J.; et al. 2023b. Qwen Technical Report. arXiv:2309.16609.

Cai, M.; Yang, J.; Gao, J.; and Lee, Y. J. 2024. Matryoshka Multimodal Models. arXiv:2405.17430.

Cao, J.; Zhang, Y.; Huang, T.; Lu, M.; Zhang, Q.; An, R.; Ma, N.; and Zhang, S. 2025. MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders. In CVPR, 19846–19856.

Chen, J.; Ye, L.; He, J.; Wang, Z.-Y.; Khashabi, D.; and Yuille, A. 2024a. LLaVOLTA: Eficient Multi-Modal Models via Stage-Wise Visual Context Compression. arXiv:2406.20092.

Chen, L.; Zhao, H.; Liu, T.; Bai, S.; Lin, J.; Zhou, C.; and Chang, B. 2024b. An Image Is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. arXiv:2403.06764.

Chen, Z.; et al. 2024c. InternVL: Scaling Up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In CVPR, 24185–24198.

Fu, C.; et al. 2026. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. In NeurIPS, volume 38.

Goyal, Y.; Khot, T.; Summers-Stay, D.; Batra, D.; and Parikh, D. 2017. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In CVPR, 6904–6913.

Gurari, D.; Li, Q.; Stangl, A. J.; Guo, A.; Lin, C.; Grauman, K.; Luo, J.; and Bigham, J. P. 2018. VizWiz Grand Challenge: Answering Visual Questions from Blind People. In CVPR, 3608–3617.

Hudson, D. A.; and Manning, C. D. 2019. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In CVPR, 6700–6709.

Jiang, A. Q.; et al. 2023. Mistral 7B. arXiv:2310.06825.

Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models. In ICML, 19730– 19742.

Li, W.; Yuan, Y.; Liu, J.; Tang, D.; Wang, S.; Zhu, J.; and Zhang, L. 2024. TokenPacker: Eficient Visual Projector for Multimodal LLM. arXiv:2407.02392.

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.- R. 2023b. Evaluating Object Hallucination in Large Vision-Language Models. arXiv:2305.10355.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024a. Improved Baselines with Visual Instruction Tuning. In CVPR, 26296– 26306.

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024b. LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge. https://llava-vl.github.io/blog/2024-01- 30-llava-next/. Accessed: 2026-01-30.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2024c. Visual Instruction Tuning. In NeurIPS, volume 36.

Liu, Y.; et al. 2025. MMBench: Is Your Multi-Modal Model an All-Around Player? In ECCV, 216–233.

Lu, P.; Mishra, S.; Xia, T.; Qiu, L.; Chang, K.-W.; Zhu, S.-C.; Tafjord, O.; Clark, P.; and Kalyan, A. 2022. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In NeurIPS, volume 35, 2507– 2521.

Ouyang, L.; et al. 2022. Training Language Models to Follow Instructions with Human Feedback. In NeurIPS, volume 35, 27730–27744.

Radford, A.; et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In ICML, 8748– 8763.

Shang, Y.; Cai, M.; Xu, B.; Lee, Y. J.; and Yan, Y. 2024. LLaVA-PruMerge: Adaptive Token Reduction for Eficient Large Multimodal Models. arXiv:2403.15388.

Singh, A.; Natarajan, V.; Shah, M.; Jiang, Y.; Chen, X.; Batra, D.; Parikh, D.; and Rohrbach, M. 2019. Towards VQA Models That Can Read. In CVPR, 8317–8326.

Team, G.; et al. 2023. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805.

Touvron, H.; et al. 2023. LLaMA: Open and Eficient Foundation Language Models. arXiv:2302.13971.

Wang, W.; et al. 2025. CogVLM: Visual Expert for Pretrained Language Models. In NeurIPS, volume 37, 121475– 121499.

Xing, L.; et al. 2024. PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction. arXiv:2410.17247.

Xu, R.; Wang, Y.; Luo, Y.; and Du, B. 2025. Rethinking Visual Token Reduction in LVLMs Under Cross-Modal Misalignment. arXiv:2506.22283.

Yang, S.; Chen, Y.; Tian, Z.; Wang, C.; Li, J.; Yu, B.; and Jia, J. 2024. VisionZip: Longer Is Better but Not Necessary in Vision Language Models. arXiv:2412.04467.

Ye, W.; Wu, Q.; Lin, W.; and Zhou, Y. 2024. Fit and Prune: Fast and Training-Free Visual Token Pruning for Multi-Modal Large Language Models. arXiv:2409.10197.

Yu, W.; Yang, Z.; Li, L.; Wang, J.; Lin, K.; Liu, Z.; Wang, X.; and Wang, L. 2023. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. arXiv:2308.02490.

Zhang, Q.; Cheng, A.; Lu, M.; Zhang, R.; Zhuo, Z.; Cao, J.; Guo, S.; She, Q.; and Zhang, S. 2025. Beyond Text-Visual Attention: Exploiting Visual Cues for Efective Token Pruning in VLMs. In ICCV, 20857–20867.

Zhang, Y.; et al. 2024. SparseVLM: Visual Token Sparsification for Eficient Vision-Language Model Inference. arXiv:2410.04417.

# Supplementary Material for

# BtsPruner: Role-Aware Visual Token Pruning for Large Vision-Language Models

This appendix is organized as follows.

• In Section A, additional visualization results are presented.

## Visualization

## Visualization comparison of visual token pruning strategies

To further validate the efectiveness of BtsPruner for visual token pruning, we present more comprehensive visualizations of the baseline networks (Liu et al. 2024a) under diferent pruning methods. As shown in Fig. 4, existing pruning strategies (Chen et al. 2024b; Alvar et al. 2025; Zhang et al. 2025) evaluate all visual tokens with a unified criterion, making them insensitive to the distinct roles played by diferent tokens. Consequently, given the same token budget, they may either miss globally representative tokens or discard tokens that encode rich local fine-grained details. In contrast, BtsPruner adopts a role-aware selection mechanism that preserves complementary visual information in a more comprehensive manner.

## Supplementary Visualizations of Attention Patterns

In this section, we present attention visualizations during the encoding process across diferent image samples. As shown in Fig. 5, the results exhibit a broadly consistent attention pattern, progressing from localized interactions to shared representations.

![](images/6b65cb6c9a680dae55dd7d28bd8778c19123b031bda85a6d3b8f3338ba9042c6.jpg)  
Figure 4: Qualitative comparison of visual token pruning strategies.

![](images/0abdbc4326de680785ed0952d3dfc2bf1578c4fed11907c738b2fc48dc943b27.jpg)  
Figure 5: Visualization of attention evolution during the encoding process across diferent image samples.