# 3D-SceneQ: Empowering 3D LLM With Query-Guided Adaptive Pruning and Multi-Modal Feature Enhancement

Yongqi Shan, Lu Zhang, Member, IEEE, Jiazuo Yu, Yunzhi Zhuge, and Huchuan Lu, Fellow, IEEE

Abstract—Current 3D scene understanding pipelines typically concatenate vast numbers of 3D object tokens with text tokens and feed the resulting sequence to a Large Language Model (LLM). However, existing 3D-LLMs for scene understanding encounter two major limitations: (i) excessive inclusion of taskirrelevant object data, introducing noise that reduces reasoning accuracy and increases hallucinations; and (ii) reliance solely on point cloud data, which inherently lacks the rich semantic information available in complementary 2D modalities, such as color, material properties, texture, and high-level contextual relationships. To address these challenges, we introduce 3D-SceneQ, a 3D LLM that unifies adaptive token pruning with multimodal semantic enrichment, markedly advancing scene understanding, reasoning, and grounding. Specifically, we propose a Query-Guided Adaptive Pruning (QGAP) module that adaptively selects task-relevant objects based on the intended meaning of user instructions, leveraging a learnable latent query to retain critical information while effectively suppressing irrelevant noise. In addition, we introduce a Multi-modal Object-level Feature Enhancement (MOFE) module, which integrates 2D feature embeddings from pre-trained models into 3D representation to enrich object-level semantic information. Evaluations show that 3D-SceneQ outperforms current state-of-the-art methods across multiple benchmarks (e.g., +5.7 CIDEr on Scan2Cap and +17.0 EM@1 on SQA3D), demonstrating its remarkable capabilities in various 3D language reasoning tasks.

Index Terms—Multi-modal Object-level Feature Enhancement, Query-Guided Adaptive Pruning, 3D Scene Understanding

## I. INTRODUCTION

U <sup>NDERSTANDING</sup> <sup>3D</sup> <sup>visual</sup> <sup>information</sup> <sup>is</sup> <sup>fundamental</sup>to numerous emerging technological domains, including autonomous navigation [1], augmented and virtual reality [2], robotics [3], and intelligent manufacturing systems [4]. Effective interpretation of spatial structures and object relationships within 3D environments significantly enhances the capability of artificial intelligence (AI) systems to perceive, understand, and interact with the physical world. In recent years, considerable progress has been made in Large Language Models (LLMs), extending their applications from traditional natural language processing to advanced multimodal tasks, including complex 3D scene understanding [5], [6]. Building upon these advancements, recent LLMs have demonstrated strong potential in 3D-related tasks. By integrating and analyzing rich spatial point cloud data, LLMs have demonstrated strong potential in a range of 3D-related tasks such as precise scene interpretation [7], robust object perception [8], and advanced spatial reasoning [9]. Such capabilities substantially enhance the contextual awareness and environmental comprehension of AI systems, ensuring more reliable and efficient performance in real-world applications. Therefore, continued exploration and development in this interdisciplinary field promise to foster deeper and more meaningful interactions among humans, machines, and complex physical environments.

![](images/dd783f0c66ef4510ec4d14e65a52a79c61f716c2aaea1ac9149650c6e5a91f64.jpg)  
Fig. 1: (a) Conventional methods concatenate 3D and text tokens. (b) 3D-SceneQ uses a latent query to select taskrelevant objects, then fuses 2D semantics for a compact, informative scene representation. represents the concatenation operation, denotes multimodal feature fusion, and denotes the projection of 2D features into 3D space.

Although these methods have achieved significant advancements, comprehensive 3D scene understanding remains hindered by two major challenges: (i) Numerous Object Token Noise: current 3D-LLMs [9], [10] typically employ 3D encoders (e.g. PointNet [11] and PointNet++ [12]) to convert 3D scenes into a large number of object tokens, among which only a small subset is task-relevant, introducing noise and increasing the likelihood of hallucinations [13]. (ii) Missing

2D Semantic Cues: existing models [8], [12] for 3D scene understanding primarily rely on point cloud data, which inherently lacks rich semantic cues such as color, material, and high-level contextual relationships, all of which are typically provided by 2D visual features. However, most contemporary SOTA 3D LLMs, including Leo [3] and Point-LLM [14] (see Fig. 1(a)), ignore these two critical challenges and simply feed the LLM a concatenation of text and 3D object tokens, which limits the model’s ability to reason about spatial semantics and to attain a nuanced understanding of complex 3D scenes.

To address these challenges, we introduce 3D-SceneQ, a novel 3D LLM designed to enhance scene perception, grounding, reasoning, and planning. As illustrated in Fig. 1(b), 3D-SceneQ comprises two lightweight components: Query-Guided Adaptive Pruning (QGAP) and Multi-modal Objectlevel Feature Enhancement (MOFE). First, QGAP tackles the challenge of task-relevant object selection by dynamically adjusting the number of objects retained based on the specific requirements of different questions. To achieve this adaptability, QGAP introduces a learnable latent query that adjusts the filtering ratio according to the question’s informational needs. By tailoring object selection to the task, QGAP reduces noise, mitigates hallucinations, and enhances the model’s effectiveness. Second, MOFE addresses the semantic limitations of point cloud data by incorporating 2D semantic features derived from multi-view scene images processed by pre-trained models such as CLIP [15]. Exploiting the camera intrinsics and extrinsics, MOFE projects multi-view 2D features onto the 3D space, pools the resulting point-wise embeddings into objectlevel embeddings, and integrates them with 3D tokens through cross-attention to produce a unified multi-modal representation. Through object-level feature fusion, MOFE effectively integrates multi-modal information to enhance the semantic representation of task-relevant objects in 3D LLMs, enabling a richer and more comprehensive understanding of the scene. Unlike traditional multi-view stereo (MVS) methods [16], [17] that aim to reconstruct dense 3D geometry, or pointlevel contrastive approaches such as OpenScene [18], MOFE focuses on injecting language-aligned semantic features into object-level 3D representations, enabling deeper multi-modal fusion without relying on geometric reconstruction.

Beyond their individual strengths, the synergistic integration of QGAP and MOFE gives rise to a virtuous cycle: QGAP filters out redundant clutter, sharpening the model’s focus on task-relevant objects, while MOFE enriches these objects with multi-modal semantics, which in turn reinforces the relevance signals that guide QGAP’s pruning. This synergy enables 3D-SceneQ to achieve a compact yet semantically rich understanding of complex 3D scenes, leading to consistently strong performance across diverse tasks. We evaluate 3D-SceneQ on a diverse set of 3D benchmarks, including 3D captioning [19], 3D question answering [20], and situated question answering [21]. Comprehensive experimental results show that 3D-SceneQ achieves state-of-the-art (SOTA) performance across all these tasks, achieving 78.1 CIDEr on Scan2Cap, 104.7 CIDEr on ScanQA, and 67.0 EM@1 on SQA3D, underscoring its strong generalization and reasoning capacity in complex 3D scenes. The main contributions of this work can be summarized as follows:

• We introduce an innovative 3D-SceneQ to enhance 3D scene understanding and reasoning by reducing taskirrelevant information from 3D encoder and integrating 2D semantic features. Comprehensive experiments demonstrate that 3D-SceneQ achieves SOTA performance across diverse 3D understanding tasks.

• We propose a Query-Guided Adaptive Pruning (QGAP) module, which dynamically selects task-relevant objects based on a specific user’s question, mitigating hallucinations and improving reasoning accuracy in complex 3D environments by eliminating task-irrelevant tokens.

• We develop a Multi-modal Object-level Feature Enhancement (MOFE) module, which integrates semantically rich 2D features with spatially detailed 3D data to enrich 3D object-level semantic information, enabling deeper scene understanding and robust reasoning.

## II. RELATED WORKS

## A. Point Cloud Understanding

Understanding 3D scenes is essential for intelligent systems to interact with the physical world. Unlike 2D images, 3D point clouds capture rich spatial and geometric information, making them crucial for tasks such as object recognition, manipulation, and reasoning in applications like robotics and autonomous driving. Early advances in 3D deep learning [22] were led by architectures such as PointGT [23] and DT-Net [24], which laid the foundation for 3D representation learning by effectively processing raw point clouds for classification and segmentation tasks. Meanwhile, inspired by visionlanguage models like CLIP [15], there has been growing interest in aligning 3D point clouds with language. Methods such as PointCLIP [25] and CLIP2Point [26] project point clouds into 2D depth images for zero-shot classification using 2D CLIP models. Later works like ULIP [27] and OpenShape [28] employ contrastive learning to align 3D encoders with image and text through point-image-text triplets, improving the transferability of 3D representations. Unfortunately, these methods are inherently constrained to object-level tasks and struggle to generalize to scene-level understanding.

Recently, there has been growing interest in combining 3D point clouds with natural language to enable tasks [10], [19], [29]–[31]. 3D Question Answering [29] which focuses on answering questions grounded in 3D scenes, 3D Visual Grounding [30], [32] which evaluates object localization from text and 3D captioning [19] further tests a model’s ability to describe 3D scenes in natural language. However, most existing methods are designed for specific tasks and struggle to generalize beyond the distribution of their training data. In contrast, recent 3D Multi-modal Large Language Models (3D-LLMs) unify diverse 3D-language tasks within a single architecture, leveraging LLMs’ reasoning abilities to handle QA, grounding, captioning, and more complex tasks like 3D dialogue and task decomposition.

## B. Multi-Modal Large Language Models

Multi-modal Large Language Models (MLLMs) are designed to understand and reason across diverse modalities beyond plain text, including images [15], [33]–[35], video [36], [37], and audio [38]. These models aim to bridge the gap between language and perception by learning aligned representations across different input modalities. A landmark in this direction is CLIP [15], which jointly trains image and text encoders using contrastive learning on large-scale web data, enabling strong zero-shot transfer to downstream tasks. Building upon this, models such as BLIP [33] and BLIP-2 [34] adopt a two-stage framework that connects frozen vision encoders with large language models via a learnable query transformer, followed by instruction tuning. LLaVA [39] further enhances alignment by integrating CLIP visual features with open-source LLMs like Vicuna [40], enabling multiturn visual conversation and reasoning. More recent works, including Flamingo [35], Shikra [41], and MiniGPT-4 [42], introduce gated cross-attention mechanisms and lightweight adapters to achieve more efficient and flexible vision-language integration. These models have shown strong performance across a variety of tasks such as visual question answering, image captioning, and multimodal instruction following, demonstrating the growing potential of MLLMs as generalpurpose reasoning engines across modalities.

![](images/99a2bcab08e6249f476f36470fca64bf932751d4ce018a2e0247eda1424aa690.jpg)  
Fig. 2: Overview of the 3D-SceneQ framework, which integrates multi-view 2D images, 3D point clouds, and textual instructions to generate informed responses. QGAP refers to the Query-Guided Adaptive Pruning module, and MOFE denotes the Multi modal Object-level Feature Enhancement module.

## C. 3D Multi-Modal Large Language Models

The recent success of multi-modal large language models (LLMs) has spurred growing interest in extending their capabilities to the 3D domain [3], [7], [8], [10], [14], [43]–[48]. Among these efforts, 3D-LLM [10] introduces 3D point cloud features into pre-trained 2D vision-language models (VLMs), enabling spatially grounded understanding across diverse 3Dlanguage tasks. SegPoint [7] adopts a unified architecture that directly feeds point cloud data into LLMs for a wide range of segmentation tasks. Point-LLM [14] employs a two-stage training approach—first aligning 3D features with the language space, followed by instruction tuning—which facilitates stronger integration between geometric understanding and linguistic reasoning. In contrast, 3D-MiT [8] proposes a more streamlined framework by eliminating the explicit alignment stage, directly fusing visual features and textual prompts to construct multimodal 3D inputs with reduced training complexity. OneLLM [44] and Point-Bind LLM [45] further explore the alignment of 3D point clouds with other modalities— including images, audio, and text—within a shared embedding space, thereby enhancing cross-modal reasoning and enabling seamless interaction between geometric and semantic cues. GPT4Point [46] extends this direction with a generative multimodal model that supports 3D understanding and controllable output generation. LEO [3] represents a different line of work, focusing on embodied agents that integrate 3D point clouds and egocentric RGB images for manipulation tasks. However, in LEO, visual inputs are primarily used as low-level control signals, without integrating the high-level semantics of images into a unified 3D representation.

Despite these advancements, a fundamental challenge remains: point clouds provide rich geometric information but are inherently sparse in semantic content, and often contain significant structural redundancy. Most existing architectures do not explicitly model these characteristics, limiting their ability to fully leverage the expressive potential of 3D representations for downstream reasoning and decision-making tasks.

## III. METHODOLOGY

## A. 3D-SceneQ

3D-SceneQ adopts a two-stage learning scheme to integrate multi-modal information for downstream tasks: (i) 3D Vision-Language Alignment, aligning 3D object features with linguistic representations to form a shared feature space; and (ii) Instruction Tuning, fine-tuning the model with task-specific instructions to improve 3D understanding and reasoning. In the first stage, we employ the object-centric 3D point cloud encoder, PointNet++ [12], to interpret the scene. This encoder extracts geometric features for each 3D object and represents them as point cloud embeddings. Spatial Transformer [49] is then applied after feature extraction to process the point cloud embeddings of all objects, transforming them into object-centric 3D object tokens. Subsequently, these 3D object tokens are projected through a linear layer to align their dimensionality with the requirements of the LLM, producing the final 3D object representation $f _ { 3 D } ~ \in ~ \mathbb { R } ^ { N \times d }$ , where N is the number of objects in the scene, and d is the feature dimensionality compatible with the LLM. The input in this stage includes system messages, user’s questions, and 3D object tokens f<sub>3D</sub>, forming a multi-modal sequence. This sequence is then processed by the pre-trained LLM to generate the final response. Text-based inputs, such as system messages and user’s questions, are tokenized using SentencePiece [50] with a 32k vocabulary.

The second stage applies instruction tuning for 3D task, it builds upon the first stage but introduces additional mechanisms to enhance task-specific performance. As shown in Fig. 2, the 3D features are refined through Query-Guided Adaptive Pruning (QGAP) and enriched using Multi-modal Object-level Feature Enhancement (MOFE). Details of QGAP and MOFE are provided in Section III-B and Section III-C, respectively. These two processes result in enhanced 3D object tokens $f _ { E _ { 3 D } } ~ \in ~ \mathbb { R } ^ { K \times d }$ , where K represents the number of objects remaining in the scene after pruning by QGAP. The input multi-modal sequence in this stage includes system messages, questions and enhanced 3D object tokens $f _ { E _ { 3 D } }$ . For a batch B of token sequences, 3D-SceneQ is optimized using the following objective function L:

$$
\mathcal { L } ( \theta , \mathcal { B } ) = - \sum _ { b = 1 } ^ { | \mathcal { B } | } \sum _ { t = 1 } ^ { T } \log p _ { \theta } \left( s _ { \mathrm { r e s } } ^ { ( b , t ) } \mid s _ { \mathrm { r e s } } ^ { ( b , < t ) } , s _ { \mathrm { p r e f i x } } ^ { ( b ) } \right) ,\tag{1}
$$

where $s _ { \mathrm { r e s } } ^ { ( b , t ) }$ denotes the target token at time step $t , \ s _ { \mathrm { r e s } } ^ { ( b , < t ) }$ represents all preceding tokens in the response, and $s _ { \mathrm { p r e f i x } } ^ { ( b ) }$ comprises the system message, 2D and 3D features, and the input question. T denotes the length of the target response sequence. The probability distribution over the next token is parameterized by $p _ { \theta }$ , where θ denotes the model parameters.

![](images/276082677113da5f0a39f51276f6d9c87e0f16d47592c26abc598ab992894ce5.jpg)  
Fig. 3: Overview of the Query-Guided Adaptive Pruning (QGAP) module, which retains task-relevant object tokens based on a global latent query.

## B. Query-Guided Adaptive Pruning

To adaptively filter object tokens based on relevance to a specific task, we propose Query-Guided Adaptive Pruning (QGAP). It dynamically retains the most relevant object tokens by comparing the user’s instruction with predicted 3D object category labels. As presented in Fig. 3, given a scene’s point cloud $p$ and a question $q ,$ we encode q using a pre-trained, frozen BERT [51] to obtain the question embedding q<sub>emb</sub>:

$$
q _ { \mathrm { e m b } } = \mathrm { B E R T } ( q ) .\tag{2}
$$

Next, the scene’s point cloud $p$ is processed by a 3D encoder PointNet++ to extract object tokens $o _ { i }$ and their corresponding predicted categories label<sub>i</sub>. Each predicted category is then transformed into an embedding $o _ { \mathrm { e m b } } ^ { ( i ) }$ using BERT [51]:

$$
o _ { \mathrm { e m b } } ^ { ( i ) } = \mathbf { B E R T } ( \mathrm { l a b e l } _ { i } ) , \quad i = 1 , \ldots , n ,\tag{3}
$$

where n denotes the total number of object tokens in the scene. We further compute the cosine similarity between $q _ { \mathrm { e m b } }$ and each $o _ { \mathrm { e m b } } ^ { ( i ) }$ to assess the relevance of each object token:

$$
\sin ( q _ { \mathrm { e m b } } , o _ { \mathrm { e m b } } ^ { ( i ) } ) = \frac { q _ { \mathrm { e m b } } \cdot o _ { \mathrm { e m b } } ^ { ( i ) } } { \| q _ { \mathrm { e m b } } \| \| o _ { \mathrm { e m b } } ^ { ( i ) } \| } ,\tag{4}
$$

these similarity scores are used to rank the objects by their relevance to $q _ { \mathrm { e m b } }$ and are organized into a sorted list $S \colon$

$$
S = \{ \sin ( q _ { \mathrm { e m b } } , o _ { \mathrm { e m b } } ^ { ( i ) } ) \} .\tag{5}
$$

To further adaptively prune the scene’s object tokens based on the similarity scores stored in S, we introduce a globally learnable latent query LQ. The latent query LQ captures task-relevant information by interacting with the question embedding $q _ { \mathrm { e m b } }$ through a cross-attention mechanism. To further refine the extracted features, the cross-attention output is processed through two layers of self-attention. Subsequently, the refined features are aggregated along the sequence dimension via a pooling operation. The aggregated representation is passed through a linear transformation followed by a sigmoid activation function to compute the initial filtering ratio $\gamma \colon$

![](images/574b95876cc64867e93d35b0643878209656199891b0fc06d3588ee888fcd198.jpg)  
Fig. 4: Overview of the Multi-modal Object-level Feature Enhancement (MOFE) module, which combines features across modalities. represents mapping multi-view 2D features to each point in 3D space, with aggregation and averaging across views. denotes pooling point-level features into object-level features using object masks.

$$
\gamma = \sigma ( \mathrm { L i n e a r } ( \mathrm { P o o l } ( \mathrm { S e l f A t t n } ^ { 2 } ( \mathrm { C r o s s A t t n } ( L Q , q _ { \mathrm { e m b } } ) ) ) ) ) ,\tag{6}
$$

where σ denotes the sigmoid activation function. To ensure a reasonable proportion of objects is retained, the final number of object tokens k is determined as follows:

$$
k = { \mathrm { r o u n d } } \left( \left( { \frac { \gamma } { 2 } } + 0 . 5 \right) \times n \right) .\tag{7}
$$

Finally, we select the top-k object tokens with the highest relevance to the question based on the similarity list S. These tokens are then retained for downstream processing.

## C. Multi-modal Object-level Feature Enhancement

Multi-modal Object-level Feature Enhancement (MOFE) consists of two main steps: Point-level Feature Extraction and Cross-modal Object-level Fusion. As shown in Fig. 4, we extract dense, per-pixel embeddings from multi-view images and project them onto the 3D surface to integrate 2D image features with 3D point cloud data. These projected features are then fused into a unified multi-modal representation.

Point-level feature extraction. For an RGB image with resolution $H \times W$ , we use the pre-trained segmentation model OpenSeg [52] to extract per-pixel embeddings $f _ { i } \in \mathbb { R } ^ { H \times W \times C }$ where C is the embedding dimension and i denotes the image index in a sequence of views. Each 3D point $p \in \mathbb { R } ^ { 3 }$ in the point cloud $\bar { P } \in \mathbb { R } ^ { M \times 3 }$ , with M points, is projected onto its corresponding pixel $u \ = \ ( x , y )$ in the image using the camera’s intrinsic $I _ { i }$ and extrinsic $E _ { i }$ matrices in the image for each view. This process can be expressed as:

$$
u = I _ { i } \cdot E _ { i } \cdot p .\tag{8}
$$

To aggregate features from multiple views for each projected point $p ,$ we average the per-pixel embedding from each view at the corresponding pixel location $u ,$ as expressed by:

$$
f _ { 2 D } ( u ) = \frac { 1 } { V } \sum _ { i = 1 } ^ { V } f _ { i } ( u ) ,\tag{9}
$$

where V is the number of views. This results in a set of pointlevel 2D features $f _ { 2 D } \in \mathbb { R } ^ { M \times C }$ , where each row corresponds to the aggregated feature of a point in the point cloud.

Cross-modal object-level fusion. We first employ object masks $M _ { o }$ to identify the sets of points belonging to each object in the point cloud. By pooling the 2D features $f _ { 2 D }$ associated with all points within each object region, we derive an object-level feature vector $f _ { o } .$ To align this object-level representation with the input requirements of the LLM, we apply a linear projection parameterized by $W _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d \times C } ,$ which maps $f _ { o }$ into a d-dimensional space. Formally, this transformation can be expressed as:

$$
f _ { o } ^ { \prime } = W _ { \mathrm { p r o j } } \cdot f _ { o } ,\tag{10}
$$

where $f _ { o } ^ { \prime }$ is the final object-level feature vector compatible with the LLM input dimension. We perform cross-modal fusion using cross-attention, where 3D object-level features, extracted by a pre-trained 3D encoder, serve as queries $Q$ and values V, transformed 2D object-level features provide keys K. This process is defined as:

$$
f _ { f u s e } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d _ { k } } } \right) V ,\tag{11}
$$

where $Q = W _ { q } \cdot f _ { 3 D } , K = W _ { k } \cdot f _ { o } ^ { \prime } , V = W _ { v } \cdot f _ { 3 D } . W _ { q } , W _ { k }$ and $W _ { v } \in \mathbb { R } ^ { d _ { k } \times d }$ are learnable projection matrices. $\bar { f _ { 3 D } } \in \mathbb { R } ^ { N \times d }$ represents 3D object-level features extracted by the pre-trained 3D encoder. The parameter $d _ { k }$ scales the attention scores. To enhance robustness, dropout and a residual connection are applied, resulting in the fused feature $f _ { E _ { 3 D } } ,$ , which is then fed into the LLM for multi-modal 3D scene understanding:

$$
f _ { E _ { 3 D } } = \mathrm { D r o p o u t } ( f _ { f u s e } ) + f _ { 3 D } .\tag{12}
$$

Role Prompt:   
You are an AI assistant situated in a 3D environment. You   
perceive all scene objects, including yourself, and respond to   
user instructions based on visual input.   
3D Objects Prompt:   
Objects (including you) in the scene:   
This section is followed by 3D scene   
features extracted by a pre-trained   
encoder, capturing object properties and   
spatial relations.   
Task Prompt:   
User: {instruction} ASSISTANT:   
The placeholder {instruction} is replaced   
with the user’s natural language query   
during inference.  
TABLE I: Prompt structure used in 3D-SceneQ, combining role description, object context, and user instruction in a unified input format.

## IV. EXPIREMENTS

## A. Experimental Setups

Benchmarks. Following [3], we utilise a diverse set of 3D vision-language datasets tailored for alignment and instruction tuning. In the 3D vision-language alignment phase, we train the model to bridge the gap between 3D scene representations and natural language using three types of captioning data. These include (i) object-level captions from Cap3D [53], which provide fine-grained textual descriptions of objects from Objaverse [54]; (ii) object-in-context referring expressions from ScanScribe [55] and ReferIt3D [31], further augmented with LLM-generated annotations to enhance contextual diversity; and (iii) scene-level summaries from 3RScan [56], capturing global information about object layouts, spatial relations, and scene semantics. In the instruction tuning phase, we refine the model’s ability to understand, reason, and respond to natural language instructions in 3D environments. We utilize tasks based on ScanNet [57] and 3RScan, covering open-ended 3D captioning and question answering, where the model is expected to describe scenes or answer queries about object properties, locations, and relationships.

Evaluation Metrics. We use a set of standard evaluation metrics to assess model performance across various tasks. For language generation tasks, we report CIDEr, BLEU, METEOR, and ROUGE, which measure relevance, fluency, and overlap with reference texts from different perspectives. In open-ended generation tasks, where multiple correct answers may exist for a given input, we additionally use sentence similarity [58] to better capture semantic alignment. For questionanswering tasks that require factual precision, we report top-1 exact match accuracy (EM@1), which reflects the proportion of predictions that exactly match the ground truth.

Implementation Details. We adopt Vicuna-7B [40] as the language backbone, and we directly utilize the pre-trained model from the first stage of LEO [3] for initialization. The 3D encoder, the 2D semantic segmentation model used in

<table><tr><td>Component</td><td>Number of Parameters</td></tr><tr><td>QGAP</td><td>7.1M</td></tr><tr><td>MOFE</td><td>70.3M</td></tr><tr><td>Trainable Parameters (Total)</td><td>122.6M</td></tr><tr><td>3D-SceneQ (Overall)</td><td>7.0B</td></tr></table>

TABLE II: Parameter overview for 3D-SceneQ and its key components.

MOFE, the language tokenizer and language backbone, as well as the BERT encoders used for query and object category representation in QGAP, are kept frozen during training. We adopt LoRA [59] with a low-rank setting $( r ~ = ~ 1 6 )$ to enable parameter-efficient fine-tuning of the Vicuna-7B language backbone. The remaining trainable parameters, including the proposed QGAP, MOFE, the object-to-text feature mapping layer, and the spatial transformer built atop PointNet++, are trained from scratch. Training is conducted for 10 epochs using the AdamW optimizer with a learning rate of $3 \times 1 0 ^ { - 5 }$ . All experiments are performed on two NVIDIA A800 GPUs.

We propose a prompt structure to facilitate 3D-SceneQ in comprehending tasks by integrating the model’s role description, object context within the scene, and user instructions, as detailed in Table I. In terms of model size, 3D-SceneQ comprises approximately 7 billion parameters, yet only 122.6 million parameters are tunable during training, as summarized in Table II. Notably, the QGAP and MOFE modules are exceptionally lightweight, contributing merely 7.1 million and 70.3 million parameters, respectively. As a result, these modules have a negligible impact on the overall computational and memory footprint of the model.

## B. Comparison with State-of-the-Arts

We evaluate our method across three prominent 3D scene understanding tasks: 3D scene captioning with Scan2Cap [19], 3D question answering (QA) with ScanQA [20], and 3D scene-based reasoning with SQA3D [21]. As summarized in Table III, 3D-SceneQ consistently achieves state-of-theart performance on all these benchmarks, surpassing strong task-specific baselines as well as fine-tuned specialized models across all evaluation metrics considered. Specifically, on Scan2Cap it lifts METEOR to 30.5 and ROUGE to 65.3, exceeding the previous best by 2.6 and 7.2 points, respectively; on ScanQA it pushes CIDEr to 104.7 and BLEU-4 to 13.6, reflecting more relevant and fluent answers; and on the SQA3D test set it attains an EM@1 of 67.0%, a striking 17 point leap over the strongest prior method, demonstrating exceptional proficiency in generating precise and contextually informative outputs. In contrast to conventional methods that depend on carefully engineered task-specific architectures, 3D-SceneQ adopts a unified LLM-based framework augmented with lightweight components—QGAP and MOFE. This unified architecture allows the model to flexibly and effectively adapt to diverse 3D reasoning tasks, enabling robust openended response generation while maintaining highly competitive quantitative performance achieved by specialized methods.

TABLE III: Comparison of 3D-SceneQ with state-of-the-art models on Scan2Cap, ScanQA, and SQA3D benchmarks across various evaluation metrics. “EM@1" denotes top-1 exact match accuracy; Scan2Cap n-gram metrics use $\mathrm { I o U } \geq 0 . 5 . ^ { \dagger }$ is GPT-3 answers based on generated captions. “Sim” denotes sentence similarity.
<table><tr><td rowspan="2">Method</td><td colspan="5">Scan2Cap (val)</td><td colspan="4">ScanQA (val)</td><td>SQA3D (test)</td></tr><tr><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>Sim</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td></tr><tr><td>Task-specific models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scan2Cap [19]</td><td>35.2</td><td>22.4</td><td>21.4</td><td>43.5</td><td></td><td></td><td></td><td></td><td></td><td>41.0†</td></tr><tr><td>Vote2Cap-DETR [60]</td><td>61.8</td><td>34.5</td><td>26.2</td><td>54.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScanRefer+MCAN [61]</td><td></td><td></td><td></td><td></td><td></td><td>55.4</td><td>7.9</td><td>11.5</td><td>30.0</td><td></td></tr><tr><td>ScanQA [20]</td><td></td><td></td><td></td><td></td><td></td><td>64.9</td><td>10.1</td><td>13.1</td><td>33.3</td><td>47.2</td></tr><tr><td>Task-specific fine-tuned</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3DMIT [8]</td><td></td><td></td><td></td><td></td><td></td><td>48.0</td><td>5.2</td><td>10.7</td><td>26.2</td><td></td></tr><tr><td>Chat-3D [9]</td><td></td><td></td><td></td><td></td><td></td><td>53.2</td><td>6.4</td><td>11.9</td><td>28.5</td><td></td></tr><tr><td>3D-VisTA [55]</td><td>66.9</td><td>34.0</td><td>27.1</td><td>54.3</td><td>53.8</td><td>69.6</td><td>10.4</td><td>13.9</td><td>35.7</td><td>48.5</td></tr><tr><td>3D-LLM (FlanT5) [10]</td><td></td><td></td><td></td><td></td><td></td><td>69.4</td><td>12.0</td><td>14.5</td><td>35.7</td><td></td></tr><tr><td>NaviLLM [62]</td><td></td><td></td><td></td><td></td><td></td><td>75.9</td><td>12.5</td><td>15.4</td><td>38.4</td><td></td></tr><tr><td>LEO [3]</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td><td>55.3</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td></tr><tr><td>3D-SceneQ</td><td>78.1</td><td>43.6</td><td>30.5</td><td>65.3</td><td>64.2</td><td>104.7</td><td>13.6</td><td>20.4</td><td>49.1</td><td>67.0</td></tr></table>

TABLE IV: Performance of 3D-SceneQ with different pruning and enhancement strategies. RandFilter refers to random pruning without adaptive mechanisms.
<table><tr><td rowspan="2">Method</td><td colspan="5">Scan2Cap (val)</td><td colspan="4">ScanQA (val)</td><td>SQA3D (test)</td></tr><tr><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>Sim</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td></tr><tr><td>Baseline</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td><td>55.3</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td></tr><tr><td>Baseline + RandFilter</td><td>52.3</td><td>38.0</td><td>27.8</td><td>62.3</td><td>55.2</td><td>90.1</td><td>13.4</td><td>18.1</td><td>43.7</td><td>68.9</td></tr><tr><td>Baseline + QGAP</td><td>78.4</td><td>44.1</td><td>30.4</td><td>65.6</td><td>64.2</td><td>101.6</td><td>14.1</td><td>19.9</td><td>48.7</td><td>70.1</td></tr><tr><td>Baseline + MOFE</td><td>77.8</td><td>44.4</td><td>30.5</td><td>65.1</td><td>64.1</td><td>102.0</td><td>14.6</td><td>20.0</td><td>48.8</td><td>66.7</td></tr><tr><td>3D-SceneQ</td><td>78.1</td><td>43.6</td><td>30.5</td><td>65.3</td><td>64.2</td><td>104.7</td><td>15.2</td><td>20.4</td><td>49.1</td><td>67.0</td></tr></table>

## C. Ablation Studies

1) Effectiveness of Core Components: We conducted ablation studies in Table IV to disentangle the individual and combined effects of 3D-SceneQ’s two core modules: Query-Guided Adaptive Pruning (QGAP) and the Multi-modal Object-Feature Enhancer (MOFE). Using the LEO [3] architecture with embodied agent related components removed as the Baseline model, we draw four nuanced observations: First, random pruning helps reduce redundancy but risks discarding crucial cues. The RandFilter variant stochastically discards 40% of object tokens whenever a scene contains more than 20 objects. This naïve compression alone propels SQA3D EM@1 from 50.0 to 68.9 by reducing extraneous information evidence that minimizing redundant information helps the model reason more effectively. Yet the indiscriminate pruning also removes salient objects and attributes, collapsing Scan2Cap CIDEr from 72.4 to 52.3 and diminishing ScanQA performance as well. The result exposes a core tension: token economy is vital, but pruning must be relevance-aware. Second, QGAP enhances both descriptive quality and reasoning accuracy. QGAP learns a filtering ratio over object tokens conditioned on the user’s question, excising only those entities that contribute negligibly to the downstream objective. This targeted sparsification raises Scan2Cap CIDEr to 78.4 and SQA3D EM@1 to 70.1, outperforming both the Baseline and RandFilter. The gains affirm that relevance-aware pruning preserves key geometric and semantic cues while still reaping the efficiency benefits of a lighter context. Third, MOFE boosts holistic performance, especially on ScanQA. By fusing geometric, visual, and linguistic signals into a refined object embedding, MOFE elevates Scan2Cap CIDEr to 77.8 and boosts SQA3D EM@1 to 66.7. The improvement on ScanQA is particularly pronounced, suggesting that MOFE’s crossmodal representations equip the model to resolve subtle spatial references and attribute dependencies that commonly confound 3D question answering. Finally, synergistic coupling of QGAP and MOFE yields the most balanced, robust gains. Our full model, 3D-SceneQ, combines QGAP’s task-relevant pruning strategy with MOFE’s semantic enrichment, producing consistent improvements across all benchmarks (e.g., +5.7 CIDEr over Baseline on Scan2Cap, +20.0 EM@1 on SQA3D). Guided by the user query, QGAP selects task-relevant object tokens. MOFE then enriches these tokens with 2D features and feeds back sharper relevance cues to QGAP. This virtuous loop enables 3D-SceneQ to form a compact yet information-dense representation of complex 3D scenes, yielding balanced gains across diverse tasks.

2) Impact of Filtering Strategies and Ratios in QGAP: To evaluate the effectiveness of QGAP’s learnable latent query mechanism, we conduct controlled experiments replacing the adaptive pruning module with two fixed-ratio strategies: LowSimFilt, which discards tokens with the lowest BERT-based cosine similarity to the query, and RandFilt, which randomly removes tokens. We vary the pruning ratio from 10% to 50% in 10% increments and report results in

TABLE V: Performance of 3D-SceneQ with different token pruning ratios. “LowSimFilt" refers to pruning a fixed proportion of object tokens with the lowest similarity to the query.
<table><tr><td>Method</td><td colspan="5">Scan2Cap (val)</td><td colspan="4">ScanQA (val)</td><td>SQA3D (test)</td></tr><tr><td></td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>Sim</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td></tr><tr><td>Baseline</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td><td>55.3</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td></tr><tr><td>Baseline +10% LowSimFilt</td><td>73.7</td><td>43.7</td><td>30.0</td><td>65.2</td><td>63.1</td><td>93.3</td><td>13.2</td><td>18.3</td><td>45.6</td><td>67.2</td></tr><tr><td>Baseline +20% LowSimFilt</td><td>74.5</td><td>43.5</td><td>30.2</td><td>65.3</td><td>63.8</td><td>97.6</td><td>13.2</td><td>18.8</td><td>46.7</td><td>69.0</td></tr><tr><td>Baseline +30% LowSimFilt</td><td>69.8</td><td>42.7</td><td>29.7</td><td>64.8</td><td>62.0</td><td>90.5</td><td>12.4</td><td>18.0</td><td>44.5</td><td>69.0</td></tr><tr><td>Baseline +40% LowSimFilt</td><td>67.2</td><td>42.4</td><td>29.6</td><td>64.3</td><td>61.9</td><td>89.4</td><td>11.5</td><td>16.5</td><td>42.4</td><td>61.3</td></tr><tr><td>Baseline +50% LowSimFilt</td><td>66.0</td><td>41.8</td><td>29.4</td><td>64.0</td><td>61.5</td><td>81.5</td><td>10.5</td><td>16.4</td><td>41.4</td><td>59.5</td></tr><tr><td>Baseline + QGAP</td><td>78.4</td><td>44.1</td><td>30.4</td><td>65.6</td><td>64.2</td><td>101.6</td><td>14.1</td><td>19.9</td><td>48.7</td><td>70.1</td></tr></table>

TABLE VI: Performance of 3D-SceneQ with different token pruning ratios. “RandFilt" denotes randomly removing a fixed percentage of object tokens without adaptive selection.
<table><tr><td rowspan="2">Method</td><td colspan="5">Scan2Cap (val)</td><td colspan="4">ScanQA (val)</td><td>SQA3D (test)</td></tr><tr><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>Sim</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td></tr><tr><td>Baseline</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td><td>55.3</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td></tr><tr><td>Baseline +10% RandFilt</td><td>62.7</td><td>40.2</td><td>28.7</td><td>63.2</td><td>57.5</td><td>98.2</td><td>15.4</td><td>19.2</td><td>46.3</td><td>62.5</td></tr><tr><td>Baseline +20% RandFilt</td><td>55.4</td><td>38.6</td><td>28.1</td><td>62.8</td><td>56.1</td><td>93.2</td><td>14.3</td><td>18.3</td><td>44.8</td><td>68.4</td></tr><tr><td>Baseline +30% RandFilt</td><td>53.7</td><td>38.7</td><td>27.8</td><td>62.7</td><td>54.3</td><td>89.4</td><td>13.7</td><td>17.5</td><td>43.4</td><td>66.6</td></tr><tr><td>Baseline +40% RandFilt</td><td>52.3</td><td>38.0</td><td>27.8</td><td>62.3</td><td>55.2</td><td>90.1</td><td>13.4</td><td>18.1</td><td>43.7</td><td>68.9</td></tr><tr><td>Baseline +50% RandFilt</td><td>48.0</td><td>32.6</td><td>25.5</td><td>59.7</td><td>47.2</td><td>85.5</td><td>11.9</td><td>15.4</td><td>37.9</td><td>67.4</td></tr><tr><td>Baseline + QGAP</td><td>78.4</td><td>44.1</td><td>30.4</td><td>65.6</td><td>64.2</td><td>101.6</td><td>14.1</td><td>19.9</td><td>48.7</td><td>70.1</td></tr></table>

TABLE VII: Performance of 3D-SceneQ under simulated camera calibration noise. Mild noise perturbs focal length and SE(3) parameters with Gaussian noise sampled from $\mathcal { N } ( 0 , ( 0 . 0 5 x ) ^ { 2 } )$ , and is applied with 10% probability. Severe noise increases the perturbation scale to $\mathcal { N } ( 0 , ( 0 . 1 0 x ) ^ { 2 } )$ with 15% probability, where x denotes the original parameter value.
<table><tr><td rowspan="2">Method</td><td colspan="5">Scan2Cap (val)</td><td colspan="4">ScanQA (val)</td><td>SQA3D (test)</td></tr><tr><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>Sim</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM@1</td></tr><tr><td>Baseline</td><td>72.4</td><td>38.2</td><td>27.9</td><td>58.1</td><td>55.3</td><td>101.4</td><td>13.2</td><td>20.0</td><td>49.2</td><td>50.0</td></tr><tr><td>Mild Noise</td><td>78.0</td><td>43.4</td><td>30.4</td><td>65.4</td><td>63.9</td><td>100.4</td><td>13.8</td><td>19.5</td><td>47.6</td><td>68.7</td></tr><tr><td>Severe Noise</td><td>77.4</td><td>43.6</td><td>30.5</td><td>64.2</td><td>64.1</td><td>98.2</td><td>14.2</td><td>18.9</td><td>46.2</td><td>68.1</td></tr><tr><td>Baseline + MOFE</td><td>77.7</td><td>44.4</td><td>30.5</td><td>65.1</td><td>64.1</td><td>102.0</td><td>14.6</td><td>20.0</td><td>48.8</td><td>66.7</td></tr></table>

TABLE VIII: Performance of 3D-SceneQ under different training setups. w/o Align trains the model without the alignment stage; ScanNet-only and 3RScan-only are variants trained exclusively on scenes from their respective datasets. Exact-Match (EM) accuracy is reported for question-answering tasks, while METEOR is used for all other tasks. Underlined values denote zero-shot results on unseen scenes.
<table><tr><td rowspan="2">Method</td><td colspan="3">ScanNet</td><td colspan="3">3RScan</td></tr><tr><td>Scan2Cap</td><td>ScanQA</td><td>SQA3D</td><td>3RQA</td><td>3RDialog</td><td>3RPlan</td></tr><tr><td>w/o Align</td><td>28.3</td><td>20.4</td><td>68.9</td><td>46.4</td><td>29.8</td><td>21.3</td></tr><tr><td> $\mathtt { S c a n N e t - o n l y }$ </td><td>30.4</td><td>22.2</td><td>50.1</td><td>37.3</td><td>6.3</td><td>1.9</td></tr><tr><td>3RScan-only</td><td>15.8</td><td>14.0</td><td>25.9</td><td>50.2</td><td>29.9</td><td>21.4</td></tr><tr><td>3D-SceneQ</td><td>30.5</td><td>24.3</td><td>67.0</td><td>48.8</td><td>29.9</td><td>22.7</td></tr></table>

Table V and Table VI. As shown in both tables, all fixedratio pruning strategies result in consistent and monotonic degradation in performance as the pruning ratio increases. This suggests that static heuristics are inherently brittle, as they lack the adaptivity needed to tailor pruning decisions to different query demands: over-pruning risks eliminating crucial context, while under-pruning retains distracting or irrelevant tokens. In contrast, QGAP adaptively predicts a query-specific pruning ratio via a learnable latent query mechanism. This allows the model to dynamically retain just the right amount of visual context for each question, leading to superior and stable performance across all benchmarks. These results empirically validate the conceptual motivation behind QGAP: different questions require vastly different levels of contextual detail, general scene descriptions need broader visual coverage, while targeted questions require concise yet relevant context, thereby reducing the influence of redundant information. The learnable latent query enables the model to align token retention with the intent and scope of the query, thereby achieving better performance and robustness than any fixed-rule filtering strategies.

3) Robustness of MOFE to Sensor Calibration and Registration Errors: MOFE relies on accurate 2D-to-3D feature projection, which depends on intrinsic and extrinsic camera parameters. In real-world scenarios, small calibration or registration errors are inevitable. To assess robustness, we simulate two levels of parameter noise during training: Mild noise adds Gaussian perturbations (10% probability) to focal length and SE(3) parameters, sampled from $\mathcal { N } ( 0 , ( 0 . 0 5 x ) ^ { 2 } )$ , where x is the original parameter value. Severe noise increases the probability to 15% and the noise scale to $\mathcal { N } ( 0 , ( 0 . 1 0 x ) ^ { 2 } )$ .

![](images/dbb55886189259fb7afcb2833b5d60a44f40e00ba68f7be0ea1c403875d78b32.jpg)  
Fig. 5: Qualitative comparison between the baseline and our 3D-SceneQ model, with original scenes and their filtered counterparts shown before and after QGAP. (a) and (b) highlight the improved accuracy of 3D-SceneQ’s color perception, while (c) and (d) showcase its richer, more detailed answers. Red boxes indicate regions where objects have been pruned by QGAP, distinguishing them from the original scene.

As shown in Table VII, model performance remains strong under both conditions. For example, Scan2Cap CIDEr scores reach 78.0 (mild) and 77.4 (severe), close to the clean setup (77.7). On SQA3D, EM@1 slightly improves (68.7 and 68.1 vs. 66.7). This robustness is attributed to the design of MOFE. Instead of relying on pixel-level precision, MOFE aggregates 2D features across full object masks, making it less sensitive to small spatial misalignments. Minor projection errors caused by pose noise typically affect only a few peripheral pixels, which are averaged out during object-level pooling. Furthermore, the public datasets used for evaluation, such as ScanNet and 3RScan, offer highly accurate camera poses, with real-world deviations far smaller than our simulated noise. These results demonstrate that 3D-SceneQ is robust to moderate calibration errors, and MOFE remains effective under realistic conditions.

4) Influence of Training Setups: We analyze the impact of different training setups on 3D-SceneQ performance in Table VIII, including variants trained without alignment (w/o Align) and models trained exclusively on Scan-Net (ScanNet-only) or 3RScan (3RScan-only). The ScanNet-only variant slightly underperforms compared to the full 3D-SceneQ model even on ScanNet tasks and notably struggles with cross-domain tasks (3RQA, 3RDialog, 3RPlan). Similarly, 3RScan-only achieves strong performance on 3RScan tasks but shows limited zero-shot generalization capability on ScanNet tasks. These results underscore the benefits of joint training, which offers broader coverage across diverse scenes and tasks and thereby leads to superior overall performance and enhanced cross-domain generalization in 3D-SceneQ.

![](images/825d0ac49b7e60dc3d7308c8bfabe184e1e15268c0df217a244c0edb42f24c74.jpg)  
Fig. 6: Adaptive object filtering with Query-Guided Adaptive Pruning (QGAP): (a)-(c) visualize progressively higher pruning ratios, from low to high, revealing how QGAP selectively removes irrelevant objects in response to the user’s query. Red boxes indicate regions where objects have been pruned by QGAP, distinguishing them from the original scene.

## D. Visual Analysis

In Fig. 5, we evaluate the performance of 3D-SceneQ against the baseline method across two aspects of 3D scene understanding. First, 3D-SceneQ demonstrates a significant advantage in color perception. By fusing rich appearance cues extracted from a large-scale BERT model with the extensive spatial context captured by its 3D backbone, our method can discern subtle chromatic variations that remain unrecognized by the baseline. For example, in Fig. 5(a), the baseline misclassifies a brown chair as white, whereas 3D-SceneQ correctly identifies its actual color. Similarly, in Fig. 5(b), 3D-SceneQ accurately identifies the green color of the plants, highlighting its improved understanding of color information compared to the baseline. Second, 3D-SceneQ outperforms the baseline in contextual reasoning. As shown in Fig.5(c), 3D-SceneQ suggests putting books and magazines on shelves or on the tabletop, choices that match everyday usage patterns and ergonomic reach. The baseline, by contrast, offers no specific placement at all. This difference is even more pronounced in Fig.5(d), where 3D-SceneQ grounds its recommendation in the room’s geometry, selecting the commode because that spot is easy to reach and naturally anchors the room’s visual focus. Although the baseline suggests the same location, it does so in a formulaic way, without any spatial justification, highlighting its lack of scene-aware reasoning.

To evaluate QGAP’s scene-pruning strategy, we conduct a series of experiments and visualize three representative scenarios in Fig. 6, spanning pruning levels from minimal to aggressive. In the first example (Fig.6(a)), the generic instruction “Tidy up and arrange the room.” does not reference any specific object; consequently, QGAP eliminates only 11% of the object tokens, retaining almost all objects to avoid discarding potentially relevant information. When the query targets a particular object or category, QGAP applies a more aggressive filter to remove irrelevant elements. For instance, in the second example (Fig.6(b)), the query “refrigerator” prompts QGAP to drop unrelated items such as windows and walls, resulting in a filtering ratio of 18%. In the third example (Fig.6(c)), the query “furnishings” leads to the removal of objects like plants and pillows, raising the ratio to 22%. Overall, these results show that QGAP adaptively modulates its pruning strategy to match the semantic intent of each query, enabling context-aware scene refinement that balances preserving crucial content against excising irrelevant tokens.

![](images/abfbab33a66b56b5fc096a3533bf89e378f8a643c7892534e4c6a00aa0a15cf3.jpg)  
Fig. 7: Qualitative evidence of 3D-SceneQ’s hallucination mitigation: (a)–(c) reduce hallucinations of nonexistent or irrelevant objects, (d)–(f) lessen attribute hallucinations, and (g)–(h) correctly distinguish objects with similar visual features.

To evaluate 3D-SceneQ’s effectiveness in mitigating hallucinations, we carried out a series of visual analyses (Fig. 6). The results indicate that our method markedly reduces hallucinations commonly afflicting 3D LLMs, from which we derive three main observations: First, 3D-SceneQ reduces hallucinations of irrelevant or nonexistent objects. Unlike the baseline, which invents items absent from the scene, such as a “dirty tower” (Fig. 7(a)), a “cutting board” (Fig. 7(b)), or a “cooking pot” (Fig. 7(c)), 3D-SceneQ always remains faithful to the scene content. This robustness stems from the collaborative mechanisms of QGAP and MOFE. When the queried object is absent, QGAP adaptively reduces the pruning ratio, as the similarity between all object-category embeddings in the scene and the question embedding is low, thereby preventing forced matches. Meanwhile, MOFE enriches object understanding with multi-view image features, offering stronger semantic cues. Together, they help 3D-SceneQ recognize invalid queries and respond cautiously instead of hallucinating incorrect answers. Second, 3D-SceneQ substantially mitigates perceptual errors in recognizing object attributes. Hallucinations are not limited to whole objects, they also distort geometry and material. In Fig. 7(d)–(f) the Baseline warps a square window into a triangle, morphs a round coffee table into a rectangle, and turns wooden cabinets metallic.

In contrast, 3D-SceneQ consistently preserves accurate object properties, maintaining robust attribute recognition. Third, 3D-SceneQ excels at distinguishing objects with similar visual features. Objects with similar silhouettes or textures often confound vision-language models. The Baseline mislabels a whiteboard as a bulletin board (Fig. 7(g)), and a stool as a trash can (Fig. 7(h)). 3D-SceneQ consistently delivers reliable responses, demonstrating an exceptional ability to distinguish genuine scene elements from hallucinations. Across a wide spectrum of challenging scenes, 3D-SceneQ consistently delivers precise visual perception and contextual reasoning, establishing a new benchmark for 3D scene understanding through its robust and reliable performance.

To evaluate generalization beyond the training domain, we perform zero-shot experiments on the Matterport3D dataset [63], a large-scale 3D benchmark containing scenes that 3D-SceneQ has not encountered during training. As it lacks language annotations for captioning or QA, we manually construct a set of prompts to test the model’s ability to understand novel scenes. Fig. 8(a) and (b) show its ability to identify salient objects, such as correctly detecting the bed at the center, which the baseline misses. In Fig. 8(c) and (d), it shows clear awareness of material and shape, capturing fine-grained surface and geometry information. Fig. 8(e) and (f) highlight its robustness in grounding objects in cluttered scenes, successfully locating items like a toilet or coffee table. Finally, Fig. 8(g) and (h) illustrate accurate color perception, with 3D-SceneQ correctly identifying the blue flooring while the baseline erroneously predicts brown. These results indicate that 3D-SceneQ generalizes well to previously unseen environments, even without fine-tuning, confirming its robustness and broader applicability across domains.

## V. LIMITATIONS AND FUTURE WORK

3D-SceneQ is optimized for indoor scenes and therefore exhibits limited generalization to outdoor settings. Moreover, the absence of explicit temporal modeling reduces robustness in dynamic scenes with moving agents, and its reasoning depth remains constrained without post-training signals. Future work includes adopting more general-purpose encoders and curating balanced indoor–outdoor datasets to narrow the performance gap and strengthen cross-domain generalization. In parallel, the incorporation of post-training signals (e.g., chain-of-thought supervision and reinforcement learning) and the integration of explicit temporal modeling will be explored to enhance reasoning depth and reliability in dynamic scenes.

![](images/c3ed8e9a5fc43c36e70dc5989ff51189bc11d6e9ad4e4d442505a03a8358d3f8.jpg)  
Fig. 8: Qualitative comparisons on out-of-distribution scenes from Matterport3D. (a) and (b) demonstrate 3D-SceneQ’s capability to identify and describe the most salient objects within the scene. (c) and (d) highlight the model’s ability to perceive material and shape attributes. (e) and (f) illustrate the accuracy of object grounding in visually complex environments. (g) and (h) showcase 3D-SceneQ’s sensitivity to object color.

## VI. CONCLUSION

In this paper, we present 3D-SceneQ, a novel 3D LLM framework designed to advance scene understanding, reasoning, and planning within complex 3D environments. To tackle the challenges of redundant object interference and hallucination issues in 3D LLMs, we introduce QGAP, an adaptive pruning mechanism that selectively eliminates irrelevant object tokens. Additionally, to overcome the semantic limitations inherent in point cloud data, we develop MOFE, a multi-modal feature integration module to overcome the semantic limitations of point cloud data by integrating 2D features to enhance the model’s comprehension of 3D scenes. Extensive experimental evaluations on benchmark tasks, including 3D captioning, question answering, and task planning, consistently demonstrate the superior performance of 3D-SceneQ compared to SOTA approaches. These results highlight the framework’s promising generalization capabilities and enhanced reasoning abilities in challenging 3D environments. Overall, 3D-SceneQ represents a significant advancement in various 3D understanding tasks, offering a strong foundation for future research and applications in intelligent interaction with complex 3D environments.

## REFERENCES

[1] E. Arnold, O. Y. Al-Jarrah, M. Dianati, S. Fallah, D. Oxtoby, and A. Mouzakitis, “A survey on 3d object detection methods for autonomous driving applications,” IEEE Transactions on Intelligent Transportation Systems, vol. 20, no. 10, pp. 3782–3795, 2019.

[2] J. Xiong, E.-L. Hsiang, Z. He, T. Zhan, and S.-T. Wu, “Augmented reality and virtual reality displays: emerging technologies and future perspectives,” Light: Science & Applications, vol. 10, no. 1, p. 216, 2021.

[3] J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S.-C. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” in Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 20 413–20 451.

[4] A. I. Saimon, E. Yangue, X. Yue, Z. Kong, and C. Liu, “Advancing additive manufacturing through deep learning: A comprehensive review of current progress and future challenges,” IISE Transactions, pp. 1–24, 2025.

[5] Y. Hong, Y. Du, C. Lin, J. Tenenbaum, and C. Gan, “3d concept grounding on neural fields,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 7769–7782, 2022.

[6] H. Xiong, Y. Zhuge, J. Zhu, L. Zhang, and H. Lu, “3ur-llm: An endto-end multimodal large language model for 3d scene understanding,” IEEE Trans. Multimedia, vol. 27, pp. 2899–2911, 2025.

[7] S. He, H. Ding, X. Jiang, and B. Wen, “Segpoint: Segment any point cloud via large language model,” in Eur. Conf. Comput. Vis. Springer, 2024, pp. 349–367.

[8] Z. Li, C. Zhang, X. Wang, R. Ren, Y. Xu, R. Ma, X. Liu, and R. Wei, “3dmit: 3d multi-modal instruction tuning for scene understanding,” in Int. Conf. Multimedia and Expo. IEEE, 2024, pp. 1–5.

[9] Z. Wang, H. Huang, Y. Zhao, Z. Zhang, and Z. Zhao, “Chat-3d: Dataefficiently tuning large language model for universal dialogue of 3d scenes,” arXiv preprint arXiv:2308.08769, 2023.

[10] Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3dllm: Injecting the 3d world into large language models,” Adv. Neural Inform. Process. Syst., vol. 36, pp. 20 482–20 494, 2023.

[11] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 652–660.

[12] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in Adv. Neural Inform. Process. Syst., 2017, pp. 5099–5108.

[13] Y. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su, B. Wilie, H. Lovenia, Z. Ji, T. Yu, W. Chung et al., “A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity,” in Proceedings of the 13th international joint conference on natural language processing and the 3rd conference of the asia-pacific chapter of the association for computational linguistics (volume 1: Long papers), 2023, pp. 675–718.

[14] R. Xu, X. Wang, T. Wang, Y. Chen, J. Pang, and D. Lin, “Pointllm: Empowering large language models to understand point clouds,” in Eur. Conf. Comput. Vis. Springer, 2024, pp. 131–147.

[15] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in Int. Conf. Mach. Learn. PMLR, 2021, pp. 8748–8763.

[16] F. Wang, S. Galliani, C. Vogel, P. Speciale, and M. Pollefeys, “Patchmatchnet: Learned multi-view patchmatch stereo,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 14 194–14 203.

[17] Y. Cabon, L. Stoffl, L. Antsfeld, G. Csurka, B. Chidlovskii, J. Revaud, and V. Leroy, “Must3r: Multi-view network for stereo 3d reconstruction,” in IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 1050–1060.

[18] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser et al., “Openscene: 3d scene understanding with open vocabularies,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 815–824.

[19] Z. Chen, A. Gholami, M. Nießner, and A. X. Chang, “Scan2cap: Context-aware dense captioning in rgb-d scans,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 3193–3203.

[20] D. Azuma, T. Miyanishi, S. Kurita, and M. Kawanabe, “Scanqa: 3d question answering for spatial scene understanding,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 19 129–19 139.

[21] X. Ma, S. Yong, Z. Zheng, Q. Li, Y. Liang, S.-C. Zhu, and S. Huang, “Sqa3d: Situated question answering in 3d scenes,” in Int. Conf. Learn. Represent., 2023.

[22] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, “Point-bert: Pretraining 3d point cloud transformers with masked point modeling,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 19 313–19 322.

[23] H. Zhang, C. Wang, L. Yu, S. Tian, X. Ning, and J. Rodrigues, “Pointgt: A method for point-cloud classification and segmentation based on local geometric transformation,” IEEE Trans. Multimedia, vol. 26, pp. 8052– 8062, 2024.

[24] X.-F. Han, Y.-F. Jin, H.-X. Cheng, and G.-Q. Xiao, “Dual transformer for point cloud analysis,” IEEE Trans. Multimedia, vol. 25, pp. 5638–5648, 2022.

[25] R. Zhang, Z. Guo, W. Zhang, K. Li, X. Miao, B. Cui, Y. Qiao, P. Gao, and H. Li, “Pointclip: Point cloud understanding by clip,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 8552–8562.

[26] T. Huang, B. Dong, Y. Yang, X. Huang, R. W. Lau, W. Ouyang, and W. Zuo, “Clip2point: Transfer clip to point cloud classification with image-depth pre-training,” in Int. Conf. Comput. Vis., 2023, pp. 22 157– 22 167.

[27] L. Xue, M. Gao, C. Xing, R. Martín-Martín, J. Wu, C. Xiong, R. Xu, J. C. Niebles, and S. Savarese, “Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 1179–1189.

[28] M. Liu, R. Shi, K. Kuang, Y. Zhu, X. Li, S. Han, H. Cai, F. Porikli, and H. Su, “Openshape: Scaling up 3d shape representation towards openworld understanding,” Adv. Neural Inform. Process. Syst., vol. 36, pp. 44 860–44 879, 2023.

[29] S. Ye, D. Chen, S. Han, and J. Liao, “3d question answering,” IEEE Transactions on Visualization and Computer Graphics, vol. 30, no. 3, pp. 1772–1786, 2022.

[30] D. Z. Chen, A. X. Chang, and M. Nießner, “Scanrefer: 3d object localization in rgb-d scans using natural language,” in Eur. Conf. Comput. Vis. Springer, 2020, pp. 202–221.

[31] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas, “Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes,” in Eur. Conf. Comput. Vis. Springer, 2020, pp. 422–440.

[32] Y. Chen, S. Yang, H. Huang, T. Wang, R. Lyu, R. Xu, D. Lin, and J. Pang, “Grounded 3d-llm with referent tokens,” arXiv preprint arXiv:2405.10370, 2024.

[33] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in Int. Conf. Mach. Learn. PMLR, 2022, pp. 12 888–12 900.

[34] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in Int. Conf. Mach. Learn. PMLR, 2023, pp. 19 730–19 742.

[35] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 23 716–23 736, 2022.

[36] S. Gong, Y. Zhuge, L. Zhang, Z. Yang, P. Zhang, and H. Lu, “The devil is in temporal token: High quality video reasoning segmentation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2025, pp. 29 183–29 192.

[37] H. Xiong, Z. Yang, J. Yu, Y. Zhuge, L. Zhang, J. Zhu, and H. Lu, “Streaming video understanding and multi-round interaction with memory-enhanced knowledge,” in Int. Conf. Learn. Represent., 2025.

[38] P. K. Rubenstein, C. Asawaroengchai, D. D. Nguyen, A. Bapna, Z. Borsos, F. d. C. Quitry, P. Chen, D. E. Badawy, W. Han, E. Kharitonov et al., “Audiopalm: A large language model that can speak and listen,” arXiv preprint arXiv:2306.12925, 2023.

[39] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Adv. Neural Inform. Process. Syst., vol. 36, pp. 34 892–34 916, 2023.

[40] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality,” https://lmsys.org/blog/2023-03-30-vicuna/, Mar. 2023, accessed: 2026- 03-18.

[41] K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” arXiv preprint arXiv:2306.15195, 2023.

[42] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” in Int. Conf. Learn. Represent., 2024.

[43] L. Shi, S. Kan, Y. Jin, L. Zhang, and Y. Cen, “Multi-modal selfperception enhanced large language model for 3d region-of-interest captioning with limited data,” IEEE Trans. Multimedia, vol. 27, pp. 2935–2948, 2025.

[44] J. Han, K. Gong, Y. Zhang, J. Wang, K. Zhang, D. Lin, Y. Qiao, P. Gao, and X. Yue, “Onellm: One framework to align all modalities with language,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 26 584–26 595.

[45] Z. Guo, R. Zhang, X. Zhu, Y. Tang, X. Ma, J. Han, K. Chen, P. Gao, X. Li, H. Li et al., “Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following,” arXiv preprint arXiv:2309.00615, 2023.

[46] Z. Qi, Y. Fang, Z. Sun, X. Wu, T. Wu, J. Wang, D. Lin, and H. Zhao, “Gpt4point: A unified framework for point-language understanding and generation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 26 417–26 427.

[47] F. Yin, X. Chen, C. Zhang, B. Jiang, Z. Zhao, W. Liu, G. Yu, and T. Chen, “Shapegpt: 3d shape generation with a unified multi-modal language model,” IEEE Trans. Multimedia, vol. 27, pp. 4107–4120, 2025.

[48] Y. Wu, Q. Gao, R. Zhang, H. Li, and J. Zhang, “Language-assisted 3d scene understanding,” IEEE Trans. Multimedia, vol. 27, pp. 3869–3879, 2025.

[49] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Language conditioned spatial relation reasoning for 3d object grounding,” Adv. Neural Inform. Process. Syst., vol. 35, pp. 20 522–20 535, 2022.

[50] T. Kudo and J. Richardson, “Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing,” in Proceedings of the 2018 conference on empirical methods in natural language processing: System demonstrations, 2018, pp. 66–71.

[51] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[52] G. Ghiasi, X. Gu, Y. Cui, and T.-Y. Lin, “Scaling open-vocabulary image segmentation with image-level labels,” in Eur. Conf. Comput. Vis. Springer, 2022, pp. 540–557.

[53] T. Luo, C. Rockwell, H. Lee, and J. Johnson, “Scalable 3d captioning with pretrained models,” Adv. Neural Inform. Process. Syst., vol. 36, 2024.

[54] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 13 142–13 153.

[55] Z. Zhu, X. Ma, Y. Chen, Z. Deng, S. Huang, and Q. Li, “3d-vista: Pre-trained transformer for 3d vision and text alignment,” in Int. Conf. Comput. Vis., 2023, pp. 2911–2921.

[56] J. Wald, A. Avetisyan, N. Navab, F. Tombari, and M. Nießner, “Rio: 3d object instance re-localization in changing indoor environments,” in Int. Conf. Comput. Vis., 2019, pp. 7658–7667.

[57] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “Scannet: Richly-annotated 3d reconstructions of indoor scenes,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 5828– 5839.

[58] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[59] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in Int. Conf. Learn. Represent., 2022.

[60] S. Chen, H. Zhu, X. Chen, Y. Lei, G. Yu, and T. Chen, “End-to-end 3d dense captioning with vote2cap-detr,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 11 124–11 133.

[61] Z. Yu, J. Yu, Y. Cui, D. Tao, and Q. Tian, “Deep modular co-attention networks for visual question answering,” in IEEE Conf. Comput. Vis Pattern Recog., 2019, pp. 6281–6290.

[62] D. Zheng, S. Huang, L. Zhao, Y. Zhong, and L. Wang, “Towards learning a generalist model for embodied navigation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 13 624–13 634.

[63] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Niebner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3d: Learning from rgb-d data in indoor environments,” in 2017 International Conference on 3D Vision (3DV). IEEE Computer Society, 2017, pp. 667–676.