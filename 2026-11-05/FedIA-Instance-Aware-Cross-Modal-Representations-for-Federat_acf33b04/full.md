# FedIA: Instance-Aware Cross-Modal Representations for Federated Prompt Learning

Xuanjing Zhang<sup>a</sup>, Qi Guo<sup>b</sup>, Yanan Qiao<sup>a,∗</sup>, Yong Han<sup>a</sup> and Junge Bo<sup>a</sup>

<sup>a</sup>School ofComputer Science and Technology, Xi’an Jiaotong University, Xi’an, 710049, Shaanxi, China

<sup>b</sup>Air Trafic Control and Navigation School, Air Force Engineering University, Xi’an, 710051, Shaanxi, China

## A R T I C L E I N F O

Keywords: Federated Learning Prompt Learning Vision-Language Models

## A BS T RA C T

Federated Prompt Learning (FPL) enables parameter-eficient adaptation of pre-trained Vision-Language Models across decentralized clients without exchanging raw data. To cope with crossclient heterogeneity, existing FPL methods typically pair a globally shared prompt with a clientspecific local prompt. However, this design personalizes only at the client level: the same local prompt is applied to all samples of a client, and the global and local prompts are combined in an inputagnostic, fixed manner. Consequently, it overlooks the substantial visual variations among samples within the same client and cannot adjust the balance between shared and personalized knowledge to each input. To address these limitations, we propose Federated Prompt Learning with Instance-Aware Cross-Modal Representations (FedIA), an FPL framework that learns instance-aware cross-modal representations. Specifically, FedIA first uses global text prompts as client-invariant semantic anchors to select image patches most relevant to each candidate class, suppressing distracting regions. The selected visual evidence then conditions the generation of instance-aware local text representations. Finally, an adaptive global–local fusion module calibrates the contributions of local and global text representations for each image–class pair. In this way, FedIA shifts personalization in FPL from the client level to the instance level while preserving the semantic knowledge shared across clients. Extensive experiments under domain shift, pathological label skew, and Dirichlet-based label skew show that FedIA outperforms FPL baselines, with maximum accuracy gains of 1.44%–4.37% over the latest baseline across the three settings. Ablation studies and analyses further verify the complementary contributions of instance-aware representation generation and adaptive global–local fusion.

## 1. Introduction

Federated Prompt Learning (FPL) [1, 2] has emerged as an eficient paradigm for adapting large-scale pre-trained Vision-Language Models (VLMs), such as CLIP [3] and ALIGN [4], to decentralized data. Rather than fine-tuning the entire model, FPL keeps the pre-trained backbone frozen and lets clients collaboratively optimize only a small set of learnable prompt tokens [1, 5] under the Federated Learning (FL) framework [6]. By exchanging only these lightweight prompts, FPL leverages the transferable knowledge of VLMs while keeping private data local and substantially reducing computation and communication costs.

Despite these advantages, FPL faces a central challenge arising from data heterogeneity, as client data are typically non-independent and identically distributed (Non-IID) [7, 8, 9, 10]. Diferent clients possess data from distinct domains, classes, or statistical distributions, making it dificult for a shared prompt to adequately capture both common semantic knowledge and client-specific characteristics.

To address client heterogeneity, existing FPL methods commonly maintain a globally shared prompt to capture knowledge common across clients and a personalized local prompt to model client-specific characteristics [11, 12, 13]. The global prompt promotes knowledge transfer and generalization across heterogeneous clients, whereas the local prompt enables adaptation to distinctive data distributions and visual patterns. Although this global–local design improves personalization, it remains limited in two important aspects, as illustrated in Figure 1.

Existing FPL methods often rely on client-level local prompts, applying the same prompt parameters to all samples within a client [12, 14, 15, 16]. While this design can capture coarse client-level biases, it cannot adapt to the instance-level visual variations that naturally arise among samples from the same client. For example, even samples from one client difer in background context, visual style, or class-relevant regions. Client-level prompts ignore these diferences because they are optimized independently from the visual modality and do not dynamically adjust their semantic focus according to each input image. This modality isolation [17] weakens fine-grained cross-modal alignment and causes the local prompt to encode only averaged client preferences rather than discriminative instancespecific cues. Moreover, making local representations sensitive to individual images introduces a related coordination challenge: the contribution of local visual evidence may vary across samples, requiring the interaction with shared semantic knowledge to adapt accordingly. However, many existing global–local designs often combine global and local prompts in an input-agnostic manner, such as direct concatenation or fixed fusion [11, 18, 19]. These strategies assume that the relative contributions of global and local knowledge remain constant across all samples, which fails to account for the varying semantic requirements of diferent images. Some images may benefit more from global knowledge that captures class semantics shared across clients, whereas others may require stronger local adaptation to represent domainspecific appearances or client-level characteristics. Consequently, a fixed fusion strategy can assign excessive weight to local prompts, weakening the consistency of shared semantic information, or rely too heavily on global prompts, limiting the expression of personalized visual cues. These observations suggest that efective FPL methods should generate fine-grained text representations enriched with visual information and dynamically coordinate them with globally shared semantic knowledge [11, 20].

![](images/15435a542d72d6d6242199ee13cbc955f7cef40967d72b8ce53c1943768447f9.jpg)  
Figure 1: Motivation and overview of FedIA. Challenge (1): Existing FPL methods typically use client-level prompts, which lack visual context and fail to capture instance-level variations within each client. Challenge (2): They also rely on inputagnostic fixed global–local fusion, making it dificult to adaptively balance personalization and generalization.

In this work, we propose Federated Prompt Learning with Instance-Aware Cross-Modal Representations, termed FedIA, which learns cross-modal local prompt representations and adaptively coordinates them with globally shared semantic knowledge for each image–class pair. FedIA is designed to address the limited instance-level adaptability of existing FPL methods and the fixed coordination between global and local representations under heterogeneous client distributions. To address the first challenge, FedIA introduces Instance-aware Local Representation Generation (ILRG), which dynamically generates local text representations conditioned on image semantics instead of assigning a fixed prompt to all samples from the same client. To strengthen fine-grained cross-modal alignment, FedIA further employs Global Prompt-Guided Patch Selection, which uses global text prompts as semantic anchors to prioritize class-relevant visual patches while reducing the influence of patches with low relevance to the candidate classes. The selected visual evidence is then injected into the instanceaware local text representation generation process, enabling FedIA to construct instance-aware cross-modal representations that capture discriminative visual semantics while remaining aligned with textual class knowledge. To address the second challenge, FedIA designs Adaptive Global– Local Representation Fusion (AGLRF), which adaptively calibrates the contributions of instance-aware local text representations and global text representations according to the current image and class semantics. Through these components, FedIA jointly improves instance-level personalization and global–local text representation coordination in heterogeneous federated prompt learning.

Our main contributions are summarized as follows:

• We expose an underexplored instance-level bottleneck in personalized federated prompt learning: a shared local prompt is used to represent all samples within each client, while global and local knowledge are integrated in an input-agnostic manner. We build on this observation to shift the focus of personalization from modeling coarse client-level distributions to capturing fine-grained, sample-dependent visual semantics.

• We propose FedIA, a unified FPL framework that generates instance-aware local text representations guided by selected class-relevant visual patches and adaptively calibrates personalization with globally shared knowledge for each input.

• Extensive experiments on three federated settings covering domain and label-distribution heterogeneity show that FedIA generally outperforms representative FPL methods. Further analyses confirm the complementary benefits of instance-aware text representation and adaptive global–local fusion.

## 2. Related Work

## 2.1. Federated Learning

Federated Learning (FL) has emerged as a promising paradigm to enable collaborative model training across multiple participants without sharing their raw data, thereby effectively preserving data privacy. A foundational algorithm FedAvg [6] aggregates the locally trained model parameters from all participants to update a shared global model. While FedAvg is simple and efective under independent and identically distributed (IID) data settings, its performance deteriorates significantly when faced with Non-IID data distributions where participants often hold data from diferent domains, classes, or statistical distributions [10]. To address this challenge, numerous methods have been proposed to mitigate the adverse efects of heterogeneity [9, 10]. Fed-Prox [10] makes local updates close to the global model by adding a proximal term to the local objective. Although these methods have made some progress, they still distribute a unified global model which is dificult to apply to all participants.

To address the limitations of a single global model under heterogeneous client data distributions, personalized federated learning (PFL) [7, 21] has been proposed. PFL aims to learn a personalized model tailored to each participant’s local data distribution while benefiting from knowledge shared across participants. Existing approaches achieve this objective through personalized optimization, model partitioning, or adaptive aggregation. FedPer [22] separates the model into shared base layers trained through federated averaging and personalized layers retained locally.

Despite their efectiveness in addressing data heterogeneity, conventional FL and PFL methods often require substantial model training and parameter exchange. Leveraging pre-trained models ofers an alternative by transferring existing knowledge to downstream tasks, reducing the need to learn representations from scratch. However, updating and communicating all parameters of large pre-trained models remains costly in federated settings. This motivates federated parameter-eficient fine-tuning, which enables privacypreserving adaptation of pre-trained models to downstream tasks without sharing raw data, while updating and exchanging only a small set of trainable parameters and keeping the backbone frozen.

## 2.2. Prompt Learning in VLMs

Large-scale vision-language models (VLMs) [3, 4, 23], such as CLIP [3], learn transferable cross-modal representations from massive collections of image-text pairs and exhibit strong zero-shot generalization. By aligning visual and textual representations in a shared embedding space, these models enable recognition through similarity matching between image embeddings and textual class descriptions. This alignment provides a foundation for adapting pre-trained VLMs to a wide range of downstream recognition tasks. To achieve eficient adaptation, prompt learning has emerged as an efective parameter-eficient fine-tuning paradigm [24, 25, 26, 27, 28, 29, 30, 31]. It optimizes a small set of prompt parameters while keeping the pre-trained backbone frozen, enabling task-specific adaptation with a limited number of trainable parameters. CoOp [5] replaces hand-crafted text templates with learnable context vectors, allowing task-specific contextual information to be learned directly from downstream data. To better preserve the generalization ability of the pre-trained model, KgCoOp [30] minimizes the discrepancy between text embeddings generated by learned and hand-crafted prompts. This constraint preserves pre-trained textual knowledge and improves generalization to unseen classes. Beyond text-side adaptation, MaPLe [32] extends prompt learning to both the textual and visual branches through cross-modal coupling, promoting coordinated adaptation across modalities at multiple layers.

These methods demonstrate the efectiveness of prompt learning for adapting pre-trained VLMs while preserving their transferable knowledge. However, they are primarily designed for centralized training and do not explicitly account for data distributed across heterogeneous clients. Extending prompt learning to such settings requires accommodating diferences in client data while enabling collaborative knowledge sharing. This motivates federated prompt learning, which combines eficient VLMs adaptation with collaborative learning under decentralized data distributions.

## 2.3. Federated Prompt Learning

To adapt vision-language models (VLMs) to decentralized scenarios, federated prompt learning (FPL) has attracted increasing attention [33, 34, 35, 36, 37, 38]. By optimizing a small set of prompt parameters while keeping the pre-trained VLMs backbone frozen, FPL substantially reduces the number of trainable parameters and the communication overhead associated with federated adaptation. However, heterogeneity in local domains, class distributions, and visual characteristics poses challenges to collaborative learning. An important objective is therefore to preserve transferable semantic knowledge across clients while accommodating their specific adaptation needs.

PromptFL [1] introduces prompt tuning into federated learning, allowing clients to upload locally optimized prompts for server-side aggregation. Through this process, clients collaboratively acquire transferable task knowledge without sharing raw data. Building on this paradigm, subsequent methods explore conditional prompt generation and personalized prompt learning to improve generalization and local adaptation. FedTPG [39] collaboratively learns a shared prompt generator conditioned on task-related textual information. This design enables task-aware prompt generation and improves generalization to unseen classes and datasets. To explicitly balance shared knowledge with client-specific adaptation, several methods incorporate personalized components into federated prompt learning. FedPGP [11] personalizes a globally shared prompt through client-specific low-rank adaptation and introduces a prompt-wise contrastive loss to balance generalization and personalization. FedOTP [14] combines global and local prompts to capture shared knowledge and client-specific category characteristics, respectively. It further employs unbalanced optimal transport to align their textual representations with local visual features. FedPHA [12] combines fixed-length global prompts with variable-length local prompts and uses SVDbased projection to mitigate conflicts between global and local knowledge. Together, these methods highlight the importance of coordinating shared and personalized knowledge under client heterogeneity.

Unlike these methods, FedIA does not treat the local prompt as a client-level representation. Instead, it emphasizes instance-level adaptation by generating local text representations conditioned on selected visual patches from each image. These image-conditioned representations are then adaptively fused with global text representations for each image–class pair. This design allows the contribution of local visual evidence to vary across both images and candidate classes, integrating instance-specific visual information with shared semantic knowledge.

## 3. Preliminaries

## 3.1. Prompt Learning

Large-scale pre-trained vision-language models, such as CLIP [3], demonstrate strong cross-modal representation and zero-shot transferability through contrastive learning on massive image-text pairs. The CLIP architecture comprises an image encoder �(⋅) and a text encoder �(⋅). In conventional zero-shot classification, the model relies on manually crafted discrete prompt templates (e.g., "a photo of a [CLASS]"). For the ℎ-th class $c _ { h } ,$ the class name is embedded into the template and processed by the text encoder to generate the text embedding. Given an input image $x ,$ CLIP computes the cosine similarity between the visual features $f ( x )$ and the embeddings of all candidate classes, ultimately selecting the class with the highest similarity score as the predicted result.

To enhance the adaptability of CLIP to downstream tasks, Parameter-Eficient Fine-Tuning methods optimize only a small subset of parameters while keeping the pretrained backbone frozen. Among these methods, prompt learning is particularly suitable for federated adaptation due to its compact trainable parameterization and low communication overhead. Specifically, prompt learning replaces hand-crafted templates with learnable continuous context vectors, denoted as $P = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { L } \} \in \mathbb { R } ^ { L \times d }$ , where � is the prompt length and � is the embedding dimension. The input to the text encoder for the ℎ-th class, denoted by $P _ { h } = \{ p _ { 1 } , . . . , p _ { L } , c _ { h } \}$ , is formed by concatenating learnable context vectors with the class name embedding $c _ { h } .$ . Given an image sample � with label ℎ, the prediction probabilities are as follows:

$$
P ( h | x ) = \frac { e x p ( s i m ( f ( x ) , g ( P _ { h } ) ) / \tau ) } { \sum _ { c = 1 } ^ { N _ { c } } e x p ( s i m ( f ( x ) , g ( P _ { c } ) ) / \tau ) } ,\tag{1}
$$

where $N _ { c }$ is the number of classes, ���(⋅, ⋅) is cosine similarity and � is the temperature parameter. The learnable prompts are optimized using the cross-entropy loss:

$$
\mathcal { L } _ { C E } = - \sum _ { h = 1 } ^ { N _ { c } } y _ { h } \log P ( h | x ) ,\tag{2}
$$

where $y _ { h }$ is the ℎ-th component of the one-hot label vector.

## 3.2. Federated Prompt Learning

We consider a federated learning system consisting of � clients and a central server. Each client � is equipped with a pre-trained CLIP model and a local dataset $\mathcal { D } _ { i } ~ =$ $\{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { | D _ { i } | }$ , where $( x _ { j } , y _ { j } )$ denotes the �-th image–label pair and $| D _ { i } |$ is the number of local samples. To accommodate client heterogeneity, our framework adopts a dualprompt mechanism comprising a personalized local prompt that captures client-specific characteristics and a shared global prompt that captures common semantic knowledge across clients. Let $c _ { t }$ be the selected clients at communication round �. For each �, the selected clients initialize their global prompts with the global prompt from the previous round. Subsequently, each client � performs local training for � epochs to optimize the local prompt and the global prompt by minimizing the cross-entropy loss. For each epoch $e ,$ the updates for the prompts are formulated as:

$$
\begin{array} { r } { \boldsymbol { P } _ { g , i } ^ { t , e + 1 } = \boldsymbol { P } _ { g , i } ^ { t , e } - \eta \nabla _ { \boldsymbol { P } _ { g , i } } \mathcal { L } ( \boldsymbol { P } _ { g , i } ^ { t , e } , \boldsymbol { P } _ { l , i } ^ { t , e } ; \mathcal { D } _ { i } ) , } \end{array}\tag{3}
$$

$$
P _ { l , i } ^ { t , e + 1 } = P _ { l , i } ^ { t , e } - \eta \nabla _ { P _ { l , i } } \mathcal { L } ( P _ { g , i } ^ { t , e } , P _ { l , i } ^ { t , e } ; \mathcal { D } _ { i } ) ,\tag{4}
$$

where $P _ { g , i }$ denotes the global prompt and $P _ { l , \iota }$ denotes the local prompt of client �.

After local training, the global prompt $P _ { g , i } ^ { t , E }$ is uploaded to the server, while the local prompt $P _ { l , i } ^ { t , E }$ is retained on the client to preserve personalized knowledge. The server then aggregates the global prompts received from all participating clients as follows:

$$
P _ { g } ^ { t } = \sum _ { i \in C _ { t } } \frac { | D _ { i } | } { \sum _ { j \in C _ { t } } | D _ { j } | } P _ { g , i } ^ { t , E } .\tag{5}
$$

The server distributes the aggregated global prompt to the clients for the next communication round.

## 4. Methodology

In this section, we propose Federated Prompt Learning with Instance-Aware Cross-Modal Representations (FedIA), a novel FPL framework. We leverage Instance-aware Local Text Representation Generation to capture personalized knowledge and Adaptive Global–Local Representation Fusion to calibrate it with global consensus. The overview of FedIA is shown in Figure 2.

## 4.1. Instance-Aware Local Representation Generation

In FL, data heterogeneity across clients remains a critical challenge. Existing FPL methods lack the fine-grained adaptability needed to capture instance-level variations and underexploit visual modality information. To address this, we propose Instance-aware Local Representation Generation (ILRG), which generates instance-aware local text representations from visual semantics. Because image-level features encompass visual cues with varying degrees of relevance to the target category, we further introduce Global Prompt-Guided Patch Selection to identify and retain patches that provide the most category-relevant information.

During local optimization for client �, for a given �- th input image $x _ { j } .$ let $z _ { j } ~ = ~ \{ z _ { j , p } \} _ { p = 1 } ^ { m } ~ \in ~ \mathbb { R } ^ { m \times d }$ denote the extracted visual patch features, obtained following [40], where � is the number of patches and � is the feature dimension. And $v _ { j } = f ( x _ { j } ) \in \mathbb R ^ { d }$ denotes the holistic visual features of the entire image. The global text representations are computed as $t _ { g , i } = g ( P _ { g , i } )$ . We apply $L _ { 2 }$ normalization to both the patch features and the global text representations to ensure scale invariance. Subsequently, the semantic relevance between each patch and the global text representation of each category is mathematically formulated as:

$$
s _ { j , p , h } = c o s ( z _ { j , p } , t _ { g , i , h } ) = \frac { z _ { j , p } ^ { \top } t _ { g , i , h } } { | | z _ { j , p } | | _ { 2 } | | t _ { g , i , h } | | _ { 2 } } ,\tag{6}
$$

where $s _ { j , p , h }$ quantifies the semantic similarity between the �-th patch feature of the �-th image and the global text embeddings corresponding to the ℎ-th class. CLIP is pretrained with a contrastive objective that aligns semantically corresponding visual and textual representations in a shared embedding space. Therefore, after visual projection and $L _ { 2 }$ normalization, the cosine similarity between a patch feature and a class-conditioned global text representation can be used as a semantic compatibility score. Specifically, $s _ { j , p , h }$ measures the degree to which the �-th patch of image $x _ { j }$ is aligned with the semantic concept of class ℎ. The global prompt provides a client-invariant class prototype, while the patch features provide instance-level visual evidence. Thus, global-prompt-guided patch selection enables the model to identify visual regions that are more semantically relevant to the target category.

![](images/89899c928cfe258bad08db6a9a5ed43337dfecb6160c58d938286a91e87ffb51.jpg)  
Figure 2: Framework of FedIA. FedIA first performs global-prompt-guided patch selection to retain class-relevant visual patches for each input image. The selected patches are then used by an instance-aware text representation generator to construct imageconditioned local representations. A lightweight gating network further adaptively fuses the instance text features with globa knowledge, producing calibrated text representations balance personalized information and shared semantics.

Then, the image-level visual features and the global text representations are utilized to compute the global logits, which are passed through a softmax function to derive the class weights. These weights explicitly reflect the semantic distribution of the input image across the global prompt branch. The mathematical formulation is presented as follows:

$$
\sigma _ { j , h } = \gamma s i m ( v _ { j } , t _ { g , i , h } ) ,\tag{7}
$$

$$
\omega _ { j , h } = \frac { e x p ( \sigma _ { j , h } ) } { \sum _ { c = 1 } ^ { N _ { c } } e x p ( \sigma _ { j , c } ) } ,\tag{8}
$$

where $\gamma$ is the logit scale of CLIP, ���(⋅, ⋅) represents cosine similarity, and $\omega _ { j , h }$ denotes the prediction weight of the �-th image for the ℎ-th class.

Following this, the patch-class similarity scores are weighted and aggregated using these weights, yielding the final importance score for each individual patch, which is

mathematically defined as:

$$
r _ { j , p } = \sum _ { h = 1 } ^ { N _ { c } } \omega _ { j , h } s _ { j , p , h } ,\tag{9}
$$

where $r _ { j , p }$ represents the final importance score of the �-th patch feature within the �-th image, collectively forming the image-specific patch importance $r _ { j } ~ \in ~ \mathbb { R } ^ { m }$ . Consequently, the top-� visual patches with the highest scores are retained:

$$
\begin{array} { r } { \pmb { { \cal K } } _ { j } = T o p \pmb { K } ( r _ { j } , k ) , } \end{array}\tag{10}
$$

where � is the number of patches to be retained and the selected set of patches is $z _ { i } ^ { * } = \{ z _ { j , p } | p \in K _ { j } \} \in \mathbb { R } ^ { k \times d }$

After selecting semantically relevant patches, we construct instance-specific local text representations that incorporate fine-grained visual information from each input image to better adapt to local data distributions. For each client $i ,$ we introduce an Instance-aware Local Text Representation Generator, implemented as a cross-attention [41] module with parameters $I _ { i } ,$ to dynamically generate instance-aware representations by incorporating visual information from the selected class-relevant image patches. To generate instanceaware text representations, the Instance-aware Local Text Representation Generator uses text features to query relevant visual information from the selected image patches. Specifically, we use the local text features $t _ { l , i } = g ( P _ { l , i } )$ as queries (Q) and the selected visual patch features $z _ { j } ^ { \ast }$ as keys (K) and values (V). Formally, the operation is defined as:

$$
t _ { j } = \phi ( t _ { l , i } , z _ { j } ^ { * } ) = M L P \left( s o f t m a x \left( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } \right) V \right)\tag{11}
$$

$$
Q = t _ { l , i } W _ { Q } , K = z _ { j } ^ { * } W _ { K } , V = z _ { j } ^ { * } W _ { V } ,\tag{12}
$$

where $W _ { Q } \in \mathbb { R } ^ { d \times d _ { k } } , W _ { K } \in \mathbb { R } ^ { d \times d _ { k } } , W _ { V } \in \mathbb { R } ^ { d \times d _ { v } }$ are learnable linear projection matrices, $d _ { k }$ is the scaling factor representing the dimension of the keys. The Instance-aware Local Text Representation Generator is updated locally on each client, learning to dynamically align visual semantics with text representations under client-specific heterogeneous data distributions.

## 4.2. Adaptive Global–Local Representation Fusion

Instance-aware local text representations capture classrelevant visual cues and adapt to individual samples, emphasizing client-specific patterns learned from local data. Global text representations derived from the shared prompt provide complementary class semantics shared across clients, while ofering less information about image-specific visual details. The relative importance of these two sources of information varies across images and categories. To coordinate their complementary strengths, we introduce Adaptive Global–Local Representation Fusion (AGLRF), which dynamically adjusts their contributions for each image–class pair.

Specifically, for the �-th image, $t _ { j }$ denotes the instancelevel text features. We concatenate the instance-aware local text representation, the global text representation and the image feature, and feed them into a lightweight gating network:

$$
\alpha _ { j , h } = \sigma ( M L P ( [ t _ { j , h } ; t _ { g , i , h } ; v _ { j } ] ) ) ,\tag{13}
$$

where $\alpha _ { j , h }$ denotes the gating weight for input image � and class $h , \dot { t } _ { j , h }$ is the instance-aware text representation for class ℎ generated by image $j , t _ { g , i , h }$ is the global text representation of class $h , \sigma ( \cdot )$ denotes the sigmoid activation function, and the gating network is an MLP consisting of two linear layers, with parameters represented by $G _ { i } .$ . The final text features are obtained by adaptively interpolating between the instanceaware local representation and the global representation:

$$
\bar { t } _ { j , h } = ( 1 - \alpha _ { j , h } ) t _ { j , h } + \alpha _ { j , h } t _ { g , i , h } .\tag{14}
$$

Rather than using a fixed fusion ratio, the gate takes the instance-aware text representation, the global text representation, and the image representation as input, and estimates which text representation is more reliable for each image–class pair. The instance-aware text representation captures client- and image-specific characteristics, providing personalized evidence from the local data distribution. In contrast, the global representation serves as a stable semantic anchor shared across clients, capturing common semantic knowledge. The image representation further provides complementary context about the input. By jointly considering these three sources, the gate determines the relative contributions of local and global knowledge for each sample. This enables the model to adaptively emphasize distinctive instance-level evidence or rely on global semantics as needed, achieving a better balance between instance-level adaptation and global semantic consistency.

Algorithm 1 Federated Prompt Learning with Instance-  
Aware Cross-Modal Representations (FedIA)   
Input: Communication rounds �; Local epochs �; Learn  
ing rate $\eta ;$ Number of selected patches $k ;$ Client datasets   
$\{ \bar { D } _ { i } \} _ { i = 1 } ^ { N } ;$ Selected client sets $\{ \mathcal { C } _ { t } \} _ { t = 1 } ^ { R } ;$ ; Initial server pa  
rameters $P _ { g } ^ { 0 } , G _ { g } ^ { 0 } ;$ Initial local parameters $\{ P _ { l , i } ^ { 0 } , I _ { i } ^ { 0 } \} _ { i = 1 } ^ { N }$   
Output: Final server parameters $( P _ { g } ^ { R } , G _ { g } ^ { R } )$ and local pa  
rameters $\{ P _ { l , i } ^ { R } , I _ { i } ^ { R } \} _ { \underline { { i } } = 1 } ^ { N }$   
1: // Federated Rounds   
2: for $t = 1 , 2 , \dots , R$ do   
3: // Local Training   
4: for each client $i \in { \mathcal { C } } _ { t }$ in parallel do   
5: $( P _ { g , i } ^ { t } , G _ { g , i } ^ { t } , P _ { l , i } ^ { t } , I _ { i } ^ { t } )$   
6: $\mathbf { \gets \mathrm { L o c a l U p d a t e } } ( P _ { g } ^ { t - 1 } , G _ { g } ^ { t - 1 } , P _ { l , i } ^ { t - 1 } , I _ { i } ^ { t - 1 } , \mathcal { D } _ { i } )$   
7: end for   
8: // Server Aggregation   
9: $\begin{array} { r } { P _ { g } ^ { t }  \sum _ { i \in C _ { t } } \frac { | D _ { i } | } { \sum _ { j \in C _ { t } } | D _ { j } | } P _ { g , i } ^ { t } } \end{array}$   
10: $\begin{array} { r } { G _ { g } ^ { t }  \sum _ { i \in C _ { t } } \frac { | D _ { i } | } { \sum _ { j \in C _ { t } } | D _ { j } | } G _ { g , i } ^ { t } } \end{array}$   
11: end for   
12: return $( P _ { g } ^ { R } , G _ { g } ^ { R } )$ and $\{ P _ { l , i } ^ { R } , I _ { i } ^ { R } \} _ { i = 1 } ^ { N }$   
13:   
14: LocalUpdate $( P _ { g } , G _ { g } , P _ { l , i } , I _ { i } , D _ { i } ) \colon$   
15: $P _ { g , i }  P _ { g } , \quad G _ { g , i } ^ { - }  G _ { g }$   
16: for $e = 1 , 2 , \ldots , E$ do   
17: for each mini-batch $B \subset D _ { i }$ do   
18: Extract patch features $z _ { j }$ and image features $v _ { j }$   
using the frozen image encoder; obtain $t _ { g , i , h }$ and   
$t _ { l , i , h }$ using the text encoder.   
19: Compute $s _ { j , p , h } , \omega _ { j , h } , r _ { j , p } , \mathcal { K } _ { j } ,$ and $z _ { j } ^ { * }$ by Eqs. (6)–   
(10).   
20: Compute instance-aware text representations $t _ { j , h }$   
by Eqs. (11)–(12).   
21: Compute $\alpha _ { j , h } , \bar { t } _ { j , h } ;$ class probabilities, and $\mathcal { L } _ { C E }$ by   
Eqs. (13)–(15).   
22: $\theta _ { i }  \theta _ { i } - \eta \nabla _ { \theta _ { i } } \mathcal { L } _ { C E } , \quad \theta _ { i } = \{ P _ { g , i } , P _ { l , i } , I _ { i } , G _ { g , i } \}$   
23: end for   
24: end for   
25: return $( P _ { g , i } , G _ { g , i } , P _ { l , i } , I _ { i } ) .$

## 4.3. Local Training and Server Aggregation

During local training, each client optimizes the prompt learner, instance-aware local text representation generator, and lightweight fusion module while keeping the CLIP image and text encoders frozen. Given an input image, the model obtains the image features and patch-level visual tokens from the frozen CLIP visual encoder. The selected local patches are used to generate instance-aware text representations, which are then adaptively fused with the global text representations to obtain the final text representations �<sup>̄</sup>. The loss function is formulated as:

$$
\mathcal { L } _ { C E } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Bigg ( - \sum _ { h } y _ { h } \log P _ { b } ( h | x _ { b } ) \Bigg ) ,\tag{15}
$$

$$
\mathrm { w h e r e } P _ { b } ( h | x _ { b } ) = \frac { e x p ( s i m ( f ( x _ { b } ) , \bar { t } _ { b , h } ) / \tau ) } { \sum _ { c = 1 } ^ { N _ { c } } e x p ( s i m ( f ( x _ { b } ) , \bar { t } _ { b , c } ) / \tau ) } .
$$

After local training, the server aggregates global parameters, including global prompt $P _ { g }$ and fusion network �:

$$
P _ { g } ^ { t } = \sum _ { i \in C _ { t } } \frac { \lvert D _ { i } \rvert } { \sum _ { j \in C _ { t } } \lvert D _ { j } \rvert } P _ { g , i } ^ { t , E } ,\tag{16}
$$

$$
G _ { g } ^ { t } = \sum _ { i \in C _ { t } } \frac { | D _ { i } | } { \sum _ { j \in C _ { t } } | D _ { j } | } G _ { i } ^ { t , E } .\tag{17}
$$

To distill global consensus, the central server aggregates the localized knowledge uploaded from all participating clients, thereby extracting a unified semantic representation. Upon completing this aggregation, the server broadcasts the global prompt and the updated gating network weights back to each client for the next round of training.

## 5. Experiments

## 5.1. Experimental Setup

## 5.1.1. Datasets.

We evaluate our method FedIA on multiple public benchmark datasets. We design three distinct Non-IID settings:

• Domain Shift: We utilize three multi-domain datasets, PACS (4 domains) [42], OficeHome (4 domains) [43] and Ofice31 (3 domains) [44], wherein each client’s data is exclusively sampled from a single domain.

• Pathological Label Skew: We group five visual classification datasets (Food101 [45], Caltech101 [46], Flowers102 [47], OxfordPets [48] and DTD [49]) as a single-domain CLIP benchmark, randomly assigning distinct, non-overlapping classes to each client.

• Dirichlet Imbalance: We partition CIFAR-10 [50] and CIFAR-100 [50] among clients using a Dirichlet distribution with Dirichlet coeficients $\beta \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ to simulate varying degrees of label-distribution heterogeneity.

## 5.1.2. Baselines.

We compare our FedIA with the following methods:

• Zero-Shot CLIP (ZS-CLIP) [3]: CLIP relies on manually crafted text prompt templates to evaluate and generate the model’s initial performance without the need for downstream fine-tuning.

• PromptFL [1]: A prompt-based federated learning method that uses a federated averaging mechanism to learn a unified text prompt across clients.

• FedMaPLe [32]: It directly extends the MaPLe framework into the FL setting to enable distributed multimodal prompt tuning.

• FedOTP [14]: It enhances personalization by employing Unbalanced Optimal Transport to accurately align both global and local prompts with the most relevant local visual features.

• FedPGP [11]: It utilizes low-rank adaptation and a prompt-wise contrastive loss to balance generalization and client-specific personalization in heterogeneous data.

• FedTPG [39]: A unified prompt generation network conditioned on task-related textual information, which generates context-aware prompts for adapting visionlanguage models across decentralized clients.

• FedPHA [12]: It combines fixed-length global prompts with variable-length local prompts, utilizing an SVDbased projection mechanism to resolve conflicts between global and local knowledge.

• pFedMMA [51]: A personalized federated fine-tuning method that inserts lightweight multimodal adapters into the upper layers of a frozen vision-language model. Each adapter has modality-specific down/up projections and a shared cross-modal projection.

## 5.1.3. Implementation details.

We use CLIP with a ViT-B/16 [52] backbone in all experiments. The prompt length is 16, and prompts are randomly initialized from a normal distribution. We set the local epoch to � = 1 and the communication round to � = 50, except for CIFAR-10 and CIFAR-100 where $R \ : = \ : 2 5 .$ The learning rate is 0.001, and the training batch size is 32. The number of clients varies across the three federated settings. For PACS, OficeHome, and Ofice31, we assign two clients to each domain by partitioning its data into two subsets, resulting in 8, 8, and 6 clients, respectively. This configuration preserves domain heterogeneity across clients while allowing multiple clients to share the same domain. For the CLIP benchmark datasets, including Food101, Caltech101, Flowers102, OxfordPets, and DTD, we use � = 10 clients with disjoint class subsets to simulate pathological label skew. For CIFAR-10 and CIFAR-100, we set � = 100, with client data partitioned according to a Dirichlet distribution to model heterogeneous label distributions. All experiments are conducted on an NVIDIA L20 GPU, and the reported results are averaged over three runs with diferent random seeds.

## 5.2. Performance Analysis

We first evaluate FedIA under three heterogeneous federated settings: domain shift, pathological label skew, and

Table 3  
Table 2  
Comparison of diferent methods on PACS under � = 49. Results are reported as mean ± standard deviation of accuracy (%) over three random seeds. Avg denotes the average accuracy across all domains, and the best average results are shown in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="5">PACS</td></tr><tr><td>A</td><td>C</td><td>P</td><td>S</td><td>Avg</td></tr><tr><td>ZS-CLIP</td><td> $9 7 . 2 6 \pm 0 . 0 0$ </td><td> $9 9 . 1 1 \pm 0 . 0 0$ </td><td> $9 9 . 9 5 \pm 0 . 0 0$ </td><td> $8 7 . 7 7 \pm 0 . 0 0$ </td><td> $9 6 . 0 2 \pm 0 . 0 0$ </td></tr><tr><td>PromptFL</td><td> $9 8 . 6 3 \pm 0 . 1 4$ </td><td> $9 9 . 2 4 \pm 0 . 0 5$ </td><td> $9 9 . 9 2 \pm 0 . 0 3$ </td><td> $9 4 . 4 7 \pm 0 . 4 0$ </td><td> $9 8 . 0 6 \pm 0 . 1 1 $ </td></tr><tr><td>FedMaPLe</td><td> $9 8 . 4 7 \pm 0 . 2 1 $ </td><td> $9 9 . 1 1 \pm 0 . 1 9$ </td><td> $9 9 . 9 0 \pm 0 . 0 4 $ </td><td> $9 5 . 4 3 \pm 0 . 0 6$ </td><td> $9 8 . 2 3 \pm 0 . 1 2 $ </td></tr><tr><td>FedOTP</td><td> $9 8 . 4 3 \pm 0 . 0 8 $ </td><td> $9 9 . 1 8 \pm 0 . 0 9$ </td><td> $9 9 . 7 3 \pm 0 . 0 7$ </td><td> $9 6 . 4 2 \pm 0 . 1 3$ </td><td> $9 8 . 4 4 \pm 0 . 0 3$ </td></tr><tr><td>FedPGP</td><td> $9 8 . 9 7 \pm 0 . 0 8 $ </td><td> $9 9 . 5 9 \pm 0 . 0 8$ </td><td> $9 9 . 9 5 \pm 0 . 0 1 $ </td><td> $9 6 . 6 5 \pm 0 . 1 0$ </td><td> $9 8 . 7 9 \pm 0 . 0 3$ </td></tr><tr><td>FedTPG</td><td> $9 8 . 4 7 \pm 0 . 0 7$ </td><td> $9 9 . 1 7 \pm 0 . 0 9$ </td><td> $9 9 . 8 9 \pm 0 . 0 7$ </td><td> $9 4 . 4 5 \pm 0 . 1 1 $ </td><td> $9 8 . 0 0 \pm 0 . 0 3 $ </td></tr><tr><td>FedPHA</td><td> $9 9 . 1 9 \pm 0 . 0 2$ </td><td> $9 9 . 6 5 \pm 0 . 0 3$ </td><td> $9 9 . 8 9 \pm 0 . 0 2$ </td><td> $9 7 . 1 4 \pm 0 . 1 1$ </td><td> $9 8 . 9 7 \pm 0 . 0 2 $ </td></tr><tr><td>pFedMMA</td><td> $9 7 . 7 0 \pm 0 . 7 5$ </td><td> $9 9 . 3 3 \pm 0 . 0 7$ </td><td> $9 9 . 8 8 \pm 0 . 0 8$ </td><td> $9 9 . 0 7 \pm 0 . 1 3$ </td><td> $9 9 . 0 0 \pm 0 . 1 7 $ </td></tr><tr><td>FedIA</td><td> $9 9 . 2 5 \pm 0 . 0 2$ </td><td> $9 9 . 7 4 \pm 0 . 0 5$ </td><td> $9 9 . 8 5 \pm 0 . 0 4$ </td><td> $9 7 . 7 3 \pm 0 . 0 8$ </td><td> ${ \bf 9 9 . 1 4 \pm 0 . 0 2 }$ </td></tr></table>

Comparison of diferent methods on OficeHome under $k = 4 9 .$ Results are reported as mean ± standard deviation of accuracy (%) over three random seeds. Avg denotes the average accuracy across all domains, and the best average results are shown in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="5">OfficeHome</td></tr><tr><td>A</td><td>C</td><td>P</td><td>R</td><td>Avg</td></tr><tr><td>ZS-CLIP</td><td> $8 4 . 4 5 \pm 0 . 0 0$ </td><td> $6 6 . 1 4 \pm 0 . 0 0$ </td><td> $8 9 . 5 0 \pm 0 . 0 0$ </td><td> $8 9 . 5 5 \pm 0 . 0 0$ </td><td> $8 2 . 4 1 \pm 0 . 0 0$ </td></tr><tr><td>PromptFL</td><td> $8 7 . 1 3 \pm 0 . 7 5$ </td><td> $7 5 . 2 8 \pm 0 . 3 3$ </td><td> $9 4 . 7 4 \pm 0 . 4 4$ </td><td> $9 2 . 4 8 \pm 0 . 2 6$ </td><td> $8 7 . 4 1 \pm 0 . 1 8$ </td></tr><tr><td> $\mathsf { F e d M a P L e }$ </td><td> $8 5 . 6 7 \pm 0 . 7 1$ </td><td> $7 7 . 3 8 \pm 0 . 1 8$ </td><td> $9 4 . 5 8 \pm 0 . 3 0 $ </td><td> $9 2 . 6 2 \pm 0 . 2 8 $ </td><td> $8 7 . 5 6 \pm 0 . 2 9$ </td></tr><tr><td> $\mathsf { F e d O T P }$ </td><td> $8 0 . 5 0 \pm 1 . 0 8$ </td><td> $7 6 . 9 2 \pm 0 . 2 1$ </td><td> $9 2 . 0 9 \pm 0 . 1 7$ </td><td> $8 8 . 0 2 \pm 0 . 0 8$ </td><td> $8 4 . 3 8 \pm 0 . 1 7$ </td></tr><tr><td>FedPGP</td><td> $8 8 . 8 9 \pm 0 . 6 5$ </td><td> $7 8 . 3 2 \pm 0 . 6 4$ </td><td> $9 5 . 4 8 \pm 0 . 3 7$ </td><td> $9 3 . 9 2 \pm 0 . 3 7$ </td><td> $8 9 . 1 5 \pm 0 . 2 2 $ </td></tr><tr><td>FedTPG</td><td> $8 5 . 6 9 \pm 0 . 4 8$ </td><td> $7 3 . 4 7 \pm 0 . 3 3$ </td><td> $9 3 . 9 5 \pm 0 . 1 5$ </td><td> $9 2 . 6 6 \pm 0 . 2 7$ </td><td> $8 6 . 4 4 \pm 0 . 1 7$ </td></tr><tr><td>FedPHA</td><td> $8 7 . 7 8 \pm 0 . 3 4$ </td><td> $7 9 . 6 1 \pm 0 . 3 5$ </td><td> $9 5 . 8 3 \pm 0 . 1 9$ </td><td> $9 3 . 6 6 \pm 0 . 1 4$ </td><td> $8 9 . 2 2 \pm 0 . 1 1 $ </td></tr><tr><td>pFedMMA</td><td> $8 9 . 3 6 \pm 0 . 4 4$ </td><td> $7 7 . 7 7 \pm 0 . 0 5$ </td><td> $9 3 . 5 8 \pm 0 . 5 6 $ </td><td> $9 2 . 4 6 \pm 0 . 3 1$ </td><td> $8 8 . 2 9 \pm 0 . 2 8$ </td></tr><tr><td>FedIA</td><td> $8 8 . 6 7 \pm 0 . 9 8 $ </td><td> $8 1 . 4 6 \pm 0 . 1 4$ </td><td> $9 5 . 7 4 \pm 0 . 3 3$ </td><td> $9 3 . 1 8 \pm 0 . 1 5$ </td><td> ${ \bf 8 9 . 7 6 \pm 0 . 2 7 }$ </td></tr></table>

Dirichlet imbalance, covering variations in both visual domains and client label distributions. We set $k = 4 9$ as the default number of selected patches in FedIA. For the ViT-B/16 backbone, this retains 25% of the 196 visual patches. This selection ratio allows the model to focus on patches with high category relevance while retaining diverse visual cues for instance-aware text representation generation.

Results under domain shift. To evaluate the efectiveness of FedIA under domain shift, we conduct experiments on PACS, OficeHome, and Ofice31, partitioning the data within each domain between two clients using a Dirichlet distribution with coeficient $\beta = 0 . 5$ to simulate client heterogeneity. We analyze the average performance of clients within each domain and the overall average performance of all clients. Table 1 reports an average accuracy of 99.14% for FedIA on PACS, which is higher than those of all compared methods. For OficeHome, the results in Table 2 show that FedIA reaches an average accuracy of 89.76%. It achieves a particularly notable result on the Clipart domain, reaching 81.46% and exceeding FedPHA by 1.85 percentage points. On Ofice31, FedIA obtains the highest average accuracy of 95.08%, as reported in Table 3. It also achieves the best performance on all three domains. Overall, FedIA achieves the highest average accuracy on PACS, OficeHome, and Ofice31, reaching 99.14%, 89.76%, and 95.08%, respectively. These results are consistent with our design motivation: instance-aware text representations enable adaptation to sample-specific visual cues, while adaptive global– local fusion balances personalized information with semantic knowledge shared across clients.

Comparison of diferent methods on Ofice31 under � = 49. Results are reported as mean ± standard deviation of accuracy (%) over three random seeds. Avg denotes the average accuracy across all domains, and the best average results are shown in bold.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Office31</td></tr><tr><td>A</td><td>D</td><td>W</td><td>Avg</td></tr><tr><td>ZS-CLIP</td><td> $8 1 . 1 5 \pm 0 . 0 0$ </td><td> $7 1 . 6 3 \pm 0 . 0 0$ </td><td> $7 3 . 1 5 \pm 0 . 0 0$ </td><td> $7 5 . 3 1 \pm 0 . 0 0$ </td></tr><tr><td>PromptFL</td><td> $8 8 . 4 3 \pm 0 . 0 8$ </td><td> $8 6 . 7 0 \pm 2 . 0 1$ </td><td> $9 1 . 1 8 \pm 1 . 0 5$ </td><td> $8 8 . 7 7 \pm 0 . 7 3$ </td></tr><tr><td>FedMaPLe</td><td> $8 8 . 6 0 \pm 0 . 2 4$ </td><td> $8 8 . 5 9 \pm 1 . 6 1$ </td><td> $9 1 . 0 6 \pm 0 . 7 1 $ </td><td> $8 9 . 4 1 \pm 0 . 4 3 $ </td></tr><tr><td>FedOTP</td><td> $8 6 . 3 3 \pm 0 . 4 7$ </td><td> $8 9 . 3 6 \pm 1 . 8 7$ </td><td> $9 3 . 8 7 \pm 0 . 7 1 $ </td><td> $8 9 . 8 5 \pm 0 . 5 2 $ </td></tr><tr><td>FedPGP</td><td> $9 0 . 1 7 \pm 0 . 4 3$ </td><td> $9 1 . 0 9 \pm 1 . 9 7$ </td><td> $9 5 . 4 0 \pm 0 . 3 3$ </td><td> $9 2 . 2 2 \pm 0 . 6 7$ </td></tr><tr><td>FedTPG</td><td> $8 6 . 6 5 \pm 0 . 3 3$ </td><td> $8 4 . 1 9 \pm 0 . 9 7$ </td><td> $8 7 . 8 2 \pm 1 . 7 5$ </td><td> $8 6 . 2 2 \pm 0 . 6 5$ </td></tr><tr><td>FedPHA</td><td> $9 0 . 3 7 \pm 0 . 4 6 $ </td><td> $9 5 . 7 3 \pm 0 . 8 4$ </td><td> $9 7 . 5 6 \pm 0 . 8 8$ </td><td> $9 4 . 5 5 \pm 0 . 0 2$ </td></tr><tr><td>pFedMMA</td><td> $8 9 . 7 8 \pm 0 . 5 4$ </td><td> $9 4 . 6 5 \pm 2 . 0 2$ </td><td> $9 5 . 1 9 \pm 0 . 8 1$ </td><td> $9 3 . 2 1 \pm 0 . 6 7$ </td></tr><tr><td>FedIA</td><td> $9 0 . 9 2 \pm 0 . 2 8$ </td><td> $9 6 . 1 0 \pm 0 . 8 4$ </td><td> $9 8 . 2 0 \pm 0 . 6 6$ </td><td> ${ \bf 9 5 . 0 8 \pm 0 . 4 8 }$ </td></tr></table>

![](images/744641c8e90af1e403070c69f7758c8c6b623e3c2e496ebc65369d6aeded72c4.jpg)  
(a) PACS

![](images/9e8f9436c6fb882f0396a418010e311c8dc185f2665737ce552614adaa20856f.jpg)  
(b) OficeHome

![](images/d77c23a38473b46714b115ba796dca1f04334430281bfe8d6e094fb12d078686.jpg)  
(c) Ofice31  
Figure 3: Test accuracy over communication rounds on PACS, OficeHome, and Ofice31 with $k = 4 9 .$

Table 4  
Comparison of diferent algorithms across pathological label skew datasets under $k = 4 9$ . Results are reported as mean ± standard deviation of accuracy (%) over three random seeds.
<table><tr><td>Methods</td><td>Food101</td><td>Caltech101</td><td>Flowers102</td><td>OxfordPets</td><td>DTD</td></tr><tr><td>ZS-CLIP</td><td> $8 5 . 2 6 \pm 0 . 0 0$ </td><td> $9 1 . 9 3 \pm 0 . 0 0$ </td><td> $6 7 . 8 8 \pm 0 . 0 0$ </td><td> $8 9 . 0 5 \pm 0 . 0 0$ </td><td> $4 3 . 3 3 \pm 0 . 0 0$ </td></tr><tr><td>PromptFL</td><td> $8 6 . 7 6 \pm 0 . 1 9$ </td><td> $9 3 . 4 2 \pm 0 . 1 5$ </td><td> $8 7 . 5 5 \pm 2 . 0 7$ </td><td> $9 3 . 4 3 \pm 0 . 2 7$ </td><td> $5 6 . 0 8 \pm 0 . 5 7$ </td></tr><tr><td>FedMaPLe</td><td> $8 6 . 6 6 \pm 0 . 2 2$ </td><td> $9 3 . 8 1 \pm 0 . 1 5$ </td><td> $8 7 . 6 0 \pm 1 . 6 1 $ </td><td> $9 3 . 4 8 \pm 0 . 6 1$ </td><td> $5 5 . 4 4 \pm 0 . 7 8$ </td></tr><tr><td>FedOTP</td><td> $9 2 . 9 7 \pm 0 . 1 8$ </td><td> $9 8 . 1 3 \pm 0 . 1 8$ </td><td> $9 8 . 5 5 \pm 0 . 1 4$ </td><td> $9 8 . 7 0 \pm 0 . 0 9$ </td><td> $9 0 . 5 3 \pm 0 . 4 0$ </td></tr><tr><td>FedPGP</td><td> $8 7 . 8 8 \pm 0 . 7 8$ </td><td> $9 4 . 9 6 \pm 1 . 1 0$ </td><td> $9 6 . 9 7 \pm 1 . 0 0$ </td><td> $9 4 . 6 8 \pm 0 . 5 7$ </td><td> $6 2 . 2 4 \pm 2 . 3 4$ </td></tr><tr><td>FedTPG</td><td> $8 6 . 5 3 \pm 0 . 4 5$ </td><td> $9 3 . 8 6 \pm 0 . 4 6$ </td><td> $8 0 . 0 4 \pm 4 . 4 2$ </td><td> $9 3 . 0 7 \pm 0 . 1 1 $ </td><td> $5 5 . 1 7 \pm 0 . 7 5$ </td></tr><tr><td>FedPHA</td><td> $9 6 . 3 7 \pm 0 . 0 5$ </td><td> $9 8 . 8 4 \pm 0 . 0 5$ </td><td> ${ \bf 9 9 . 3 1 \pm 0 . 0 6 }$ </td><td> $9 9 . 0 7 \pm 0 . 1 9$ </td><td> ${ \bf 9 1 . 9 4 \pm 0 . 6 1 }$ </td></tr><tr><td>pFedMMA</td><td> $9 6 . 0 3 \pm 0 . 1 8$ </td><td> $9 8 . 5 7 \pm 0 . 1 3 $ </td><td> $9 5 . 7 4 \pm 1 . 5 0$ </td><td> $9 8 . 6 9 \pm 0 . 1 5$ </td><td> $8 7 . 3 2 \pm 1 . 2 8$ </td></tr><tr><td>FedIA</td><td> ${ \bf 9 6 . 5 8 \pm 0 . 0 4 }$ </td><td> $\mathbf { 9 9 . 1 3 \pm 0 . 1 0 }$ </td><td> $9 9 . 2 5 \pm 0 . 0 3$ </td><td> ${ \bf 9 9 . 2 8 \pm 0 . 0 8 }$ </td><td> $9 1 . 6 9 \pm 0 . 2 7$ </td></tr></table>

Results under pathological label skew. Table 4 reports the results on five single-domain CLIP benchmarks, including Food101, Caltech101, Flowers102, OxfordPets, and DTD, under the 16-shot setting. Each dataset is partitioned among 10 clients with non-overlapping class subsets, resulting in substantial label heterogeneity and limited class coverage at each client. FedIA achieves the best performance on Food101, Caltech101, and OxfordPets, reaching 96.58%, 99.13%, and 99.28%, respectively. The small standard deviations across three random seeds also suggest stable performance under this setting. Overall, these results indicate that client-level or globally shared prompts alone have limited ability to address severe class heterogeneity. By constructing instance-aware representations, FedIA captures both client-specific semantics and visual variations among individual samples, allowing personalization to account for the characteristics of each input. Meanwhile, its adaptive global–local fusion retains shared semantic knowledge while incorporating instance-specific local information, thereby improving personalization without sacrificing cross-client generalization.

Results under Dirichlet imbalance. We evaluate FedIA on CIFAR-10 and CIFAR-100 using Dirichlet partitions with $\beta \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ to examine its performance under varying degrees of label heterogeneity. Smaller values of � generally produce more skewed client-level class distributions. Table 5 shows that FedIA achieves the highest average accuracy among the compared methods on both datasets for all three values of �. On CIFAR-10, FedIA obtains accuracies of 97.40%, 95.73%, and 94.21% for $\beta ~ = ~ 0 . 1 , ~ 0 . 3$ , and 0.5, respectively. The corresponding accuracies on CIFAR-100 are 88.28%, 81.10%, and 77.57%. The improvements are more pronounced on CIFAR-100, where FedIA exceeds the strongest baseline in each setting by 0.74, 1.24, and 1.44 percentage points, respectively. In particular, at $\beta = 0 . 5 ,$ FedIA improves the accuracy from 76.13% achieved by pFed-MMA to 77.57%. With more categories and visually similar classes, CIFAR-100 places greater demands on the model’s ability to capture discriminative visual details. These results support the efectiveness of combining instance-aware local text representations with adaptive global–local fusion under diferent levels of label heterogeneity.

Table 5  
Performance comparison on CIFAR-10 and CIFAR-100 under Dirichlet partitioning with $\beta \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ and $k = 4 9$ . Results are reported as mean ± standard deviation of accuracy (%) over three random seeds.
<table><tr><td rowspan="2">Methods</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td></tr><tr><td> $\beta = 0 . 1$ </td><td> $\beta = 0 . 3$ </td><td> $\beta = 0 . 5$ </td><td> $\beta = 0 . 1$ </td><td> $\beta = 0 . 3$ </td><td> $\beta = 0 . 5$ </td></tr><tr><td>ZS-CLIP</td><td> $8 7 . 8 4 \pm 0 . 3 8$ </td><td> $8 8 . 0 5 \pm 0 . 1 1$ </td><td> $8 7 . 8 4 \pm 0 . 1 5$ </td><td> $6 4 . 9 4 \pm 0 . 1 3$ </td><td> $6 4 . 8 9 \pm 0 . 0 2$ </td><td> $6 4 . 9 0 \pm 0 . 0 4$ </td></tr><tr><td>PromptFL</td><td> $8 9 . 6 8 \pm 0 . 6 7$ </td><td> $9 1 . 7 0 \pm 0 . 1 8$ </td><td> $9 1 . 8 0 \pm 0 . 3 8 $ </td><td> $7 0 . 9 6 \pm 0 . 1 5$ </td><td> $7 2 . 4 5 \pm 0 . 1 4$ </td><td> $7 2 . 8 9 \pm 0 . 1 5$ </td></tr><tr><td>FedMaPLe</td><td> $9 2 . 5 7 \pm 0 . 4 4$ </td><td> $9 3 . 4 4 \pm 0 . 6 0$ </td><td> $9 3 . 5 8 \pm 0 . 9 7$ </td><td> $7 4 . 6 1 \pm 0 . 4 3$ </td><td> $7 5 . 6 3 \pm 0 . 2 6$ </td><td> $7 5 . 9 2 \pm 0 . 3 6$ </td></tr><tr><td> $\mathsf { F e d O T P }$ </td><td> $9 6 . 9 5 \pm 0 . 3 1$ </td><td> $9 5 . 0 7 \pm 0 . 3 0$ </td><td> $9 3 . 5 5 \pm 0 . 3 8$ </td><td> $8 6 . 4 7 \pm 0 . 1 8$ </td><td> $7 9 . 0 5 \pm 0 . 2 8$ </td><td> $7 5 . 2 1 \pm 0 . 3 3$ </td></tr><tr><td>FedPGP</td><td> $9 4 . 6 7 \pm 0 . 1 4$ </td><td> $9 4 . 2 0 \pm 0 . 3 7$ </td><td> $9 3 . 3 5 \pm 0 . 7 3$ </td><td> $8 6 . 4 4 \pm 0 . 0 5$ </td><td> $7 9 . 0 9 \pm 0 . 5 1$ </td><td> $7 5 . 8 1 \pm 0 . 4 3$ </td></tr><tr><td>FedTPG</td><td> $8 9 . 2 0 \pm 0 . 2 5$ </td><td> $9 1 . 1 4 \pm 0 . 4 4$ </td><td> $9 1 . 2 9 \pm 0 . 5 8$ </td><td> $6 9 . 4 0 \pm 0 . 3 9$ </td><td> $7 0 . 1 0 \pm 0 . 4 3$ </td><td> $7 0 . 0 8 \pm 0 . 5 4$ </td></tr><tr><td>FedPHA</td><td> $9 7 . 3 5 \pm 0 . 2 9$ </td><td> $9 5 . 5 8 \pm 0 . 2 7$ </td><td> $9 4 . 1 5 \pm 0 . 3 3 $ </td><td> $8 7 . 3 5 \pm 0 . 1 0$ </td><td> $7 9 . 8 6 \pm 0 . 3 7$ </td><td> $7 6 . 1 0 \pm 0 . 4 1$ </td></tr><tr><td>pFedMMA</td><td> $9 7 . 1 6 \pm 0 . 2 7$ </td><td> $9 5 . 4 3 \pm 0 . 1 0$ </td><td> $9 3 . 8 2 \pm 0 . 2 6$ </td><td> $8 7 . 5 4 \pm 0 . 2 2$ </td><td> $7 9 . 7 5 \pm 0 . 5 4$ </td><td> $7 6 . 1 3 \pm 0 . 2 3$ </td></tr><tr><td>FedIA</td><td> ${ \bf 9 7 . 4 0 \pm 0 . 1 9 }$ </td><td> ${ \bf9 5 . 7 3 \pm 0 . 1 9 }$ </td><td> ${ \bf 9 4 . 2 1 \pm 0 . 2 7 }$ </td><td> ${ \bf 8 8 . 2 8 \pm 0 . 3 4 }$ </td><td> ${ \bf 8 1 . 1 0 \pm 0 . 4 6 }$ </td><td> ${ \bf 7 7 . 5 7 \pm 0 . 6 1 }$ </td></tr></table>

![](images/7cf4132e1ff7492ce242a1af1227fec5b9e1f12ab8e8caff4a5faaaf5789e106.jpg)  
(a) � = 0.1

![](images/7b4e218e946e108c492ba6008516097d25ebd706c98d2267a9324ef45c4fa907.jpg)  
(b) � = 0.3

![](images/45ad963e89ee869fb66618e76f60f81d9e03b1e1496070d14ca3819c1cc260cc.jpg)  
(c) $\beta = 0 . 5$  
Figure 4: Test accuracy over communication rounds on CIFAR-10 under Dirichlet partitioning with $\beta \in \left\{ 0 . 1 , 0 . 3 , 0 . 5 \right\}$ and $k = 4 9 .$

Overall, the results across the three heterogeneous federated settings demonstrate the efectiveness of FedIA in generating instance-aware local text representations and adaptively fusing them with shared global representations.

## 5.3. Ablation Study

We conduct ablation studies to assess the individual contributions and combined efectiveness of Instance-aware Local Representation Generation (ILRG) and Adaptive Global– Local Representation Fusion (AGLRF).

![](images/6ac5052489bb4d998172822ea2a8e8a5f4aa28b1f146ab02c249e883fd6cd362.jpg)  
(a) � = 0.1

![](images/dd4ed8e099c873d5c7870caae670e7e219899caa27081643fe9a88703aa5152c.jpg)  
(b) � = 0.3

![](images/5a7b71949ac52e04a5fe5fc9bbb097c230946f9351a3b7264365d46227da6589.jpg)  
Figure 5: Test accuracy over communication rounds on CIFAR-100 under Dirichlet partitioning with $\beta \in \left\{ 0 . 1 , 0 . 3 , 0 . 5 \right\}$ and $k = 4 9 .$  
(c) � = 0.5

Ablation study of FedIA on Ofice31 under $k = 2 5$ and $k = 4 9$ . Δ denotes the accuracy improvement over the baseline in percentage points.
<table><tr><td>k</td><td>ILRG</td><td>AGLRF</td><td>A</td><td>Δ</td><td>D</td><td>Δ</td><td>W</td><td>Δ</td><td>Avg</td><td>Δ</td></tr><tr><td rowspan="4">25</td><td></td><td></td><td>88.43</td><td></td><td>86.70</td><td></td><td>91.18</td><td></td><td>88.77</td><td></td></tr><tr><td>√</td><td></td><td>90.70</td><td>↑2.27</td><td>95.82</td><td>↑9.12</td><td>97.67</td><td>↑6.49</td><td>94.73</td><td>↑5.96</td></tr><tr><td></td><td>√</td><td>90.40</td><td>↑1.97</td><td>95.36</td><td>↑8.66</td><td>97.65</td><td>↑6.47</td><td>94.47</td><td>↑5.70</td></tr><tr><td>√</td><td>√</td><td>90.94</td><td>↑2.51</td><td>95.95</td><td>↑9.25</td><td>98.08</td><td>↑ 6.90</td><td>94.99</td><td>↑6.22</td></tr><tr><td rowspan="4">49</td><td></td><td></td><td>88.43</td><td>一</td><td>86.70</td><td>一</td><td>91.18</td><td>一</td><td>88.77</td><td>一</td></tr><tr><td>√</td><td></td><td>90.60</td><td>↑2.17</td><td>95.47</td><td>↑8.77</td><td>98.01</td><td>↑6.83</td><td>94.69</td><td>↑5.92</td></tr><tr><td></td><td>√</td><td>90.40</td><td>↑1.97</td><td>95.36</td><td>↑8.66</td><td>97.65</td><td>↑6.47</td><td>94.47</td><td>↑5.70</td></tr><tr><td>√</td><td>√</td><td>90.92</td><td>↑2.49</td><td>96.10</td><td>↑9.40</td><td>98.20</td><td>↑7.02</td><td>95.08</td><td>↑6.31</td></tr></table>

Ablation study of FedIA on OficeHome, CIFAR-10, and CIFAR-100 under $k = 2 5$ and $k = 4 9$ with $\beta = 0 . 5 $ . The first value in each pair is the average accuracy (%), and the second is the improvement over the baseline in percentage points.
<table><tr><td rowspan="2">ILRG</td><td rowspan="2">AGLRF</td><td colspan="6"> $k = 2 5$ </td><td colspan="6"> $k = 4 9$ </td></tr><tr><td colspan="2">OfficeHome</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">OfficeHome</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td></tr><tr><td rowspan="3">√</td><td></td><td>87.41</td><td></td><td>91.80</td><td></td><td>72.89</td><td></td><td>87.41</td><td></td><td>91.80</td><td></td><td>72.89</td><td></td></tr><tr><td></td><td>89.45</td><td>↑2.04</td><td>93.93</td><td>↑2.13</td><td>76.83</td><td>↑3.94</td><td>89.56</td><td>↑2.15</td><td>93.99</td><td>↑2.19</td><td>77.03</td><td>↑4.14</td></tr><tr><td>√</td><td>89.57</td><td>↑2.16</td><td>94.19</td><td>↑2.39</td><td>76.17</td><td>↑3.28</td><td>89.57</td><td>↑2.16</td><td>94.19</td><td>↑2.39</td><td>76.17</td><td>↑3.28</td></tr><tr><td>√</td><td>√</td><td>89.73</td><td>↑2.32</td><td>94.22</td><td>↑2.42</td><td>77.53</td><td>↑4.64</td><td>89.76</td><td>↑2.35</td><td>94.21</td><td>↑2.41</td><td>77.57</td><td>↑4.68</td></tr></table>

## 5.4. Convergence Analysis

Table 6 presents the results on Ofice31. Both ILRG and AGLRF independently improve the average accuracy over the baseline, demonstrating the benefit of each component. Their combination further increases performance, achieving the highest average accuracies of 94.99% and 95.08% under $k = 2 5$ and $k = 4 9$ , respectively. The consistent improvements across both patch selection settings support the complementary roles of the ILRG and AGLRF. Table 7 further evaluates their contributions on OficeHome, CIFAR-10 and CIFAR-100. Using ILRG alone consistently improves the baseline under both $k ~ = ~ 2 5$ and $k \ = \ 4 9 .$ . Specifically, with $k = 4 9$ , ILRG increases accuracy by $2 . 1 5 , 2 . 1 9$ , and 4.14 percentage points on the three datasets, respectively. These gains support the efectiveness of generating instanceaware local text representations from selected visual patches, allowing local representations to incorporate fine-grained visual information specific to each input. Using AGLRF alone also yields consistent gains, with an improvement of 3.28 percentage points over the baseline on CIFAR-100 under $k = 4 9$ . This result supports adaptively adjusting the contributions of personalized local information and globally shared semantics. When both modules are enabled, FedIA achieves the highest accuracy among the evaluated configurations on all three datasets under both values of �, further improving upon either module alone.

Together, these results demonstrate the complementary benefits of ILRG and AGLRF: ILRG enriches local text representations with instance-specific visual evidence, while AGLRF adaptively integrates these representations with shared semantic knowledge.

We examine the convergence behavior of FedIA by tracking test accuracy over communication rounds on the domain-shift benchmarks and the CIFAR datasets. Figure 3 presents the results on PACS, OficeHome, and Ofice31. FedIA achieves substantial accuracy improvements during the early rounds and subsequently maintains competitive performance. On PACS, it converges to an accuracy comparable to the strongest baselines, while on OficeHome and Ofice31, it maintains a clearer advantage throughout much of the training process. Figures 4 and 5 further report the results under Dirichlet partitions with $\beta \in \ \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ On CIFAR-10, FedIA improves rapidly during the initial rounds and reaches a high accuracy early in training across all three settings. Its subsequent improvements are more gradual, with limited fluctuations as training progresses. Several strong baselines exhibit similar convergence trends, but FedIA consistently finishes with the highest accuracy. On CIFAR-100, FedIA also shows rapid initial improvements and maintains a clear final advantage across the three Dirichlet settings. Overall, these curves show that FedIA learns effectively within the evaluated communication budgets across diferent heterogeneous settings. Together with the ablation results, they support the efectiveness of instance-aware local representation generation and adaptive global–local fusion.

## 5.5. Sensitivity Analysis

Sensitivity to �. As shown in Figure 6, FedIA maintains relatively stable performance across diferent values of �, indicating limited sensitivity to the number of selected patches. On CIFAR-100 with $\beta = 0 . 5$ , accuracy varies only slightly from � = 25 to $k = 6 4$ , reaching its highest value of 77.57% at $k = 4 9$ , before decreasing to 77.42% at $k = 9 8$ OficeHome and Ofice31 exhibit a similar preference for a moderate selection size, achieving their best accuracies of 89.76% and 95.08%, respectively, at $k = 4 9$ . These results suggest that a moderate number of semantically relevant patches provides suficient visual evidence for instanceaware text representation generation. A small � excludes useful category-relevant cues, whereas a large � includes additional patches with lower semantic relevance, reducing the concentration of informative visual content. Overall, the small accuracy variations across the evaluated settings support the robustness of FedIA to this hyperparameter and the choice of � = 49 as the default value.

![](images/acbba481dbb089b84041af00c5f7dda427d8a3a4a8f2f482bada32ebd30377ef.jpg)  
(a) OficeHome

![](images/dd703adf135c5699ec5f1eb7ffb3f300b1e96a5d62f5bdc586ba396b61951dbe.jpg)  
(b) Ofice31

![](images/474ba9334c39da00ebbab1769ad6986ace93b70c7a0d44ba653d95bcc1e78c9c.jpg)  
(c) CIFAR-100

Figure 6: Test accuracy of FedIA on OficeHome, Ofice31, and CIFAR-100 under diferent values of �.  
![](images/3d154c0f36c06c37d5261a47e95fe77d5ba4b1cc2cba64138e2af44e502401af.jpg)  
(a) Food101

![](images/009e5dd4e9395a8d442f79ef83008a67e64b4ccd8aa5133e94a8b6bb4297fe43.jpg)  
(b) Caltech101

![](images/ce021df057385186fe8f33ae341aee39cfbe1fdab1f30a5f9c8ab0175eb830d5.jpg)  
(c) OxfordPets  
Figure 7: Test accuracy on Food101, Caltech101, and OxfordPets under diferent numbers of shots.

Sensitivity to Shot Number. Figure 7 reports the performance under diferent shot numbers on Food101, Caltech101 and OxfordPets. FedIA achieves gains in low-shot scenarios, suggesting that instance-aware text representation generation can make efective use of limited supervision by incorporating sample-specific visual information. As the number of shots increases, its accuracy further improves, showing that the method also benefits from additional labeled examples. FedIA remains competitive across the evaluated shot settings and achieves the highest average accuracy at 16 shots on these three datasets. These results support its efectiveness under varying levels of data availability, from limited-label settings to those with more supervision.

## 6. Conclusion

In this work, we propose FedIA, an FPL framework that learns instance-aware cross-modal representations for adapting vision-language models under heterogeneous federated settings. FedIA addresses the limitations of clientlevel prompts and input-agnostic global–local interactions by selecting class-relevant visual patches, generating imageconditioned local text representations, and adaptively fusing them with globally shared text representations. Through this design, FedIA captures fine-grained instance-level visual semantics while preserving transferable global knowledge. Extensive experiments under diverse Non-IID settings demonstrate that FedIA generally outperforms representative FPL baselines. Furthermore, ablation studies and sensitivity analyses verify the efectiveness, contribution, and robustness of its key components. Overall, the results show that FedIA provides an efective solution for improving personalized multimodal adaptation in heterogeneous federated environments.

## References

[1] T. Guo, S. Guo, J. Wang, X. Tang, W. Xu, Promptfl: Let federated participants cooperatively learn prompts instead of models - federated

learning in age of foundation model, IEEE Trans. Mob. Comput. 23 (5) (2024) 5179–5194. doi:10.1109/TMC.2023.3302410. URL https://doi.org/10.1109/TMC.2023.3302410

[2] H. Zhao, W. Du, F. Li, P. Li, G. Liu, Fedprompt: Communicationeficient and privacy-preserving prompt tuning in federated learning, in: IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023, IEEE, 2023, pp. 1–5. doi:10.1109/ICASSP49357.2023.10095356. URL https://doi.org/10.1109/ICASSP49357.2023.10095356

[3] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever, Learning transferable visual models from natural language supervision, in: M. Meila, T. Zhang (Eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, Vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 8748–8763. URL http://proceedings.mlr.press/v139/radford21a.html

[4] C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li, T. Duerig, Scaling up visual and vision-language representation learning with noisy text supervision, in: M. Meila, T. Zhang (Eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, Vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 4904–4916.

[5] K. Zhou, J. Yang, C. C. Loy, Z. Liu, Learning to prompt for visionlanguage models, Int. J. Comput. Vis. 130 (9) (2022) 2337–2348. doi:10.1007/S11263-022-01653-1. URL https://doi.org/10.1007/s11263-022-01653-1

[6] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas, Communication-eficient learning of deep networks from decentralized data, in: A. Singh, X. J. Zhu (Eds.), Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AIS-TATS 2017, 20-22 April 2017, Fort Lauderdale, FL, USA, Vol. 54 of Proceedings of Machine Learning Research, PMLR, 2017, pp. 1273– 1282.

[7] Q. Guo, Y. Qi, S. Qi, D. Wu, Q. Li, Fedmcsa: Personalized federated learning via model components self-attention, Neurocomputing 560 (2023) 126831. doi:10.1016/J.NEUCOM.2023.126831. URL https://doi.org/10.1016/j.neucom.2023.126831

[8] W. Huang, M. Ye, Z. Shi, H. Li, B. Du, Rethinking federated learning with domain shift: A prototype view, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, IEEE, 2023, pp. 16312–16322. doi: 10.1109/CVPR52729.2023.01565. URL https://doi.org/10.1109/CVPR52729.2023.01565

[9] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, A. T. Suresh, SCAFFOLD: stochastic controlled averaging for federated learning, in: Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, Vol. 119 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 5132–5143.

[10] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, Federated optimization in heterogeneous networks, in: I. S. Dhillon, D. S. Papailiopoulos, V. Sze (Eds.), Proceedings of the Third Conference on Machine Learning and Systems, MLSys 2020, Austin, TX, USA, March 2-4, 2020, mlsys.org, 2020. URL https://proceedings.mlsys.org/paper\_files/paper/2020/hash/ 1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

[11] T. Cui, H. Li, J. Wang, Y. Shi, Harmonizing generalization and personalization in federated prompt learning, in: R. Salakhutdinov, Z. Kolter, K. A. Heller, A. Weller, N. Oliver, J. Scarlett, F. Berkenkamp (Eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, Vol. 235 of Proceedings of Machine Learning Research, PMLR / OpenReview.net, 2024, pp. 9646–9661.

URL https://proceedings.mlr.press/v235/cui24c.html

[12] C. Fang, W. Huang, G. Wan, Y. Yang, M. Ye, Fedpha: Federated prompt learning for heterogeneous client adaptation, in: A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, J. Zhu (Eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Vol. 267 of Proceedings of Machine Learning Research, PMLR / OpenReview.net, 2025.

[13] T. Guo, S. Guo, J. Wang, pfedprompt: Learning personalized prompt for vision-language models in federated learning, in: Y. Ding, J. Tang, J. F. Sequeda, L. Aroyo, C. Castillo, G. Houben (Eds.), Proceedings of the ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May 2023, ACM, 2023, pp. 1364–1374. doi: 10.1145/3543507.3583518. URL https://doi.org/10.1145/3543507.3583518

[14] H. Li, W. Huang, J. Wang, Y. Shi, Global and local prompts cooperation via optimal transport for federated learning, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, IEEE, 2024, pp. 12151– 12161. doi:10.1109/CVPR52733.2024.01155. URL https://doi.org/10.1109/CVPR52733.2024.01155

[15] Q. Li, Y. Zhou, J. Zhou, X. Yang, X. Gu, Diverse and public features cooperation via gradient rectification for federated prompt learning, in: C. Gurrin, K. Schoefmann, M. Zhang, L. Rossetto, S. Rudinac, D. Dang-Nguyen, W. Cheng, P. Chen, J. Benois-Pineau (Eds.), Proceedings of the 33rd ACM International Conference on Multimedia, MM 2025, Dublin, Ireland, October 27-31, 2025, ACM, 2025, pp. 4077–4086. doi:10.1145/3746027.3755308. URL https://doi.org/10.1145/3746027.3755308

[16] J. Luo, C. Chen, S. Wu, Mixture of experts made personalized: Federated prompt learning for vision-language models, in: The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, OpenReview.net, 2025. URL https://openreview.net/forum?id=xiDJaTim3P

[17] H. Zheng, S. Yang, Z. He, J. Yang, Z. Huang, Hierarchical crossmodal prompt learning for vision-language models, in: IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, IEEE, 2025, pp. 1891–1901. doi: 10.1109/ICCV51701.2025.00184. URL https://doi.org/10.1109/ICCV51701.2025.00184

[18] Y. Chen, X. Lin, J. Wu, X. Cai, Q. Zhang, X. Fan, J. Xu, X. Su, Multi-modal style transfer-based prompt tuning for eficient federated domain generalization, in: S. Koenig, C. Jenkins, M. E. Taylor (Eds.), Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, AAAI Press, 2026, pp. 20427–20435. doi:10.1609/AAAI.V40I25.39177. URL https://doi.org/10.1609/aaai.v40i25.39177

[19] Y. Yang, W. Huang, G. Wan, B. Yang, M. Ye, Federated disentangled tuning with textual prior decoupling and visual dynamic adaptation, in: A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, J. Zhu (Eds.), Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, Vol. 267 of Proceedings of Machine Learning Research, PMLR / OpenReview.net, 2025. URL https://proceedings.mlr.press/v267/yang25k.html

[20] W. Deng, C. Thrampoulidis, X. Li, Unlocking the potential of prompttuning in bridging generalized and personalized federated learning, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, IEEE, 2024, pp. 6087–6097. doi:10.1109/CVPR52733.2024.00582. URL https://doi.org/10.1109/CVPR52733.2024.00582

[21] A. Z. Tan, H. Yu, L. Cui, Q. Yang, Towards personalized federated learning, IEEE Trans. Neural Networks Learn. Syst. 34 (12) (2023) 9587–9603. doi:10.1109/TNNLS.2022.3160699. URL https://doi.org/10.1109/TNNLS.2022.3160699

URL https://openreview.net/forum?id=NW31gAylIm

[22] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, S. Choudhary, Federated learning with personalization layers, CoRR abs/1912.00818 (2019). arXiv:1912.00818, doi:10.48550/ARXIV.1912.00818. URL https://doi.org/10.48550/arXiv.1912.00818

[23] L. Yao, R. Huang, L. Hou, G. Lu, M. Niu, H. Xu, X. Liang, Z. Li, X. Jiang, C. Xu, FILIP: fine-grained interactive language-image pretraining, in: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, OpenReview.net, 2022.

[24] B. Chen, L. Zuo, M. Jing, K. He, Y. Wang, Dropout prompt learning: Towards robust and adaptive vision-language models, in: S. Koenig, C. Jenkins, M. E. Taylor (Eds.), Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, AAAI Press, 2026, pp. 19987–19995. doi:10.1609/AAAI. V40I24.39083.

[25] M. Jia, L. Tang, B. Chen, C. Cardie, S. J. Belongie, B. Hariharan, S. Lim, Visual prompt tuning, in: S. Avidan, G. J. Brostow, M. Cissé, G. M. Farinella, T. Hassner (Eds.), Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXIII, Vol. 13693 of Lecture Notes in Computer Science, Springer, 2022, pp. 709–727. doi:10.1007/ 978-3-031-19827-4\_41.

[26] M. U. Khattak, S. T. Wasim, M. Naseer, S. Khan, M. Yang, F. S. Khan, Self-regulating prompts: Foundational model adaptation without forgetting, in: IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, IEEE, 2023, pp. 15144–15154. doi:10.1109/ICCV51070.2023.01394. URL https://doi.org/10.1109/ICCV51070.2023.01394

[27] X. Li, Y. Min, H. Chen, Z. Du, F. Li, J. Li, Generalizing visionlanguage models with dedicated prompt guidance, in: S. Koenig, C. Jenkins, M. E. Taylor (Eds.), Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, AAAI Press, 2026, pp. 23239–23247. doi:10.1609/AAAI. V40I28.39492.

[28] F. Song, Y. Li, J. Li, R. Wang, C. Zheng, F. Xu, H. Xiong, Ample: Supporting vision-language models via adaptive-debiased ensemble multi-prompt learning, Int. J. Comput. Vis. 134 (2) (2026) 67. doi: 10.1007/S11263-025-02591-4.

[29] Y. Wang, L. Cheng, C. Fang, D. Zhang, M. Duan, M. Wang, Revisiting the power of prompt for visual tuning, in: R. Salakhutdinov, Z. Kolter, K. A. Heller, A. Weller, N. Oliver, J. Scarlett, F. Berkenkamp (Eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, Vol. 235 of Proceedings of Machine Learning Research, PMLR / OpenReview.net, 2024, pp. 50233–50247.

URL https://proceedings.mlr.press/v235/wang24i.html

[30] H. Yao, R. Zhang, C. Xu, Visual-language prompt tuning with knowledge-guided context optimization, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, IEEE, 2023, pp. 6757–6767. doi:10.1109/CVPR52729.2023.00653.

[31] W. Zhang, F. Lu, S. Yan, Hierarchical cross-modal prompt learning with llm-guided alignment for vision-language models, Knowl. Based Syst. 350 (2026) 116568. doi:10.1016/J.KNOSYS.2026.116568. URL https://doi.org/10.1016/j.knosys.2026.116568

[32] M. U. Khattak, H. A. Rasheed, M. Maaz, S. H. Khan, F. S. Khan, Maple: Multi-modal prompt learning, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver,

BC, Canada, June 17-24, 2023, IEEE, 2023, pp. 19113–19122. doi: 10.1109/CVPR52729.2023.01832. URL https://doi.org/10.1109/CVPR52729.2023.01832

[33] C. Feng, B. Li, X. Xu, Y. Liu, H. Fu, W. Zuo, Learning federated visual prompt in null space for MRI reconstruction, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, IEEE, 2023, pp. 8064–8073. doi:10.1109/CVPR52729.2023.00779. URL https://doi.org/10.1109/CVPR52729.2023.00779

[34] S. Su, M. Yang, B. Li, X. Xue, Federated adaptive prompt tuning for multi-domain collaborative learning, in: M. J. Wooldridge, J. G. Dy, S. Natarajan (Eds.), Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2024, February 20-27, 2024, Vancouver, Canada, AAAI Press, 2024, pp. 15117–15125. doi:10.1609/AAAI.V38I13.29434. URL https://doi.org/10.1609/aaai.v38i13.29434

[35] Z. Wang, W. Huang, T. Chen, Z. Shi, G. Wan, Y. Qiao, B. Yang, J. Wang, B. Li, M. Ye, An empirical study of federated prompt learning for vision language model, in: Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI 2025, Montreal, Canada, August 16-22, 2025, ijcai.org, 2025, pp. 10705– 10713. doi:10.24963/IJCAI.2025/1188. URL https://doi.org/10.24963/ijcai.2025/1188

[36] K. Zeng, H. Wen, T. Shen, R. Li, Ft-promptfl: A feature transmissionbased framework for communication-eficient prompt federated learning, in: IEEE INFOCOM 2026 - IEEE Conference on Computer Communications, Tokyo, Japan, May 18-21, 2026, IEEE, 2026, pp. 1–10. doi:10.1109/INFOCOM59046.2026.11571515. URL https://doi.org/10.1109/INFOCOM59046.2026.11571515

[37] C. Zhang, J. Zhang, Y. Xie, W. Mao, X. Wang, B. Zhou, B. Yu, Fedhap: Federated hierarchical adaptive prompt learning for crossdomain generalization, Appl. Soft Comput. 197 (2026) 115193. doi: 10.1016/J.ASOC.2026.115193. URL https://doi.org/10.1016/j.asoc.2026.115193

[38] R. Zhu, L. Bai, Y. Guo, Y. Ruan, T. Yu, Z. Lu, Controllable federated prompt learning at test time, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2026, Denver, CO, USA, June 3-7, 2026, Computer Vision Foundation, 2026, pp. 39455–39465. URL https://openaccess.thecvf.com/content/CVPR2026/html/Zhu\_ Controllable\_Federated\_Prompt\_Learning\_at\_Test\_Time\_CVPR\_2026\_ paper.html

[39] C. Qiu, X. Li, C. K. Mummadi, M. R. Ganesh, Z. Li, L. Peng, W. Lin, Federated text-driven prompt generation for vision-language models, in: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, OpenReview.net, 2024.

[40] X. Dong, J. Bao, Y. Zheng, T. Zhang, D. Chen, H. Yang, M. Zeng, W. Zhang, L. Yuan, D. Chen, F. Wen, N. Yu, Maskclip: Masked self-distillation advances contrastive language-image pretraining, in: IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, IEEE, 2023, pp. 10995–11005. doi:10.1109/CVPR52729.2023.01058. URL https://doi.org/10.1109/CVPR52729.2023.01058

[41] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, Attention is all you need, in: I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, 2017, pp. 5998–6008. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[42] D. Li, Y. Yang, Y. Song, T. M. Hospedales, Deeper, broader and artier domain generalization, in: IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017,

URL https://doi.org/10.1007/978-3-319-10599-4\_29

IEEE Computer Society, 2017, pp. 5543–5551. doi:10.1109/ICCV. 2017.591.

URL https://doi.org/10.1109/ICCV.2017.591

[43] H. Venkateswara, J. Eusebio, S. Chakraborty, S. Panchanathan, Deep hashing network for unsupervised domain adaptation, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, IEEE Computer Society, 2017, pp. 5385–5394. doi:10.1109/CVPR.2017.572. URL https://doi.org/10.1109/CVPR.2017.572

[44] K. Saenko, B. Kulis, M. Fritz, T. Darrell, Adapting visual category models to new domains, in: K. Daniilidis, P. Maragos, N. Paragios (Eds.), Computer Vision - ECCV 2010, 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV, Vol. 6314 of Lecture Notes in Computer Science, Springer, 2010, pp. 213–226. doi:10.1007/978-3-642-15561-1\ \_16.

[45] L. Bossard, M. Guillaumin, L. V. Gool, Food-101 - mining discriminative components with random forests, in: D. J. Fleet, T. Pajdla, B. Schiele, T. Tuytelaars (Eds.), Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI, Vol. 8694 of Lecture Notes in Computer Science, Springer, 2014, pp. 446–461. doi:10.1007/978-3-319-10599-4\ \_29.

[46] L. Fei-Fei, R. Fergus, P. Perona, Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories, in: IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2004, Washington, DC, USA, June 27 - July 2, 2004, IEEE Computer Society, 2004, p. 178. doi:10.1109/CVPR.2004.383. URL https://doi.org/10.1109/CVPR.2004.383

[47] M. Nilsback, A. Zisserman, Automated flower classification over a large number of classes, in: Sixth Indian Conference on Computer Vision, Graphics & Image Processing, ICVGIP 2008, Bhubaneswar, India, 16-19 December 2008, IEEE Computer Society, 2008, pp. 722– 729. doi:10.1109/ICVGIP.2008.47. URL https://doi.org/10.1109/ICVGIP.2008.47

[48] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and dogs, in: 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI, USA, June 16-21, 2012, IEEE Computer Society, 2012, pp. 3498–3505. doi:10.1109/CVPR.2012.6248092. URL https://doi.org/10.1109/CVPR.2012.6248092

[49] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, Describing textures in the wild, in: 2014 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2014, Columbus, OH, USA, June 23-28, 2014, IEEE Computer Society, 2014, pp. 3606–3613. doi:10.1109/CVPR.2014.461. URL https://doi.org/10.1109/CVPR.2014.46

[50] A. Krizhevsky, G. Hinton, et al., Learning multiple layers of features from tiny images (2009).

[51] S. Ghiasvand, M. Alizadeh, R. Pedarsani, pfedmma: Personalized federated fine-tuning with multi-modal adapter for vision-language models, in: C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, A. Faust (Eds.), The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026, proceedings.iclr.cc, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 7b6e1905ebe2caa41870fcdcb9898243-Abstract-Conference.html

[52] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale, in: 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, OpenReview.net, 2021.