# LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Eficient Multimodal Fine-Tuning

Muyao Yuan<sup>∗</sup>   
MOEKLINNS, and School of   
Computer Science and Technology   
Xi’an Jiaotong University   
Xi’an, China   
yuanmuyao@stu.xjtu.edu.cn   
Weizhan Zhang<sup>†</sup>   
MOEKLINNS, and School of   
Computer Science and Technology   
Xi’an Jiaotong University   
Xi’an, China   
zhangwzh@xjtu.edu.cn   
Yuan Gao   
China Telecom   
Xi’an, China   
gaoy97@chinatelecom.cn   
Muyan Jiao<sup>∗</sup>   
MOEKLINNS, and School of   
Computer Science and Technology   
Xi’an Jiaotong University   
Xi’an, China   
jmy0406@stu.xjtu.edu.cn   
Yuanhong Zhang   
MOEKLINNS, and School of   
Computer Science and Technology   
Xi’an Jiaotong University   
Xi’an, China   
yuanhongzhang@stu.xjtu.edu.cn

Jiangyong Ying E-surfing Vision Technology Co., Ltd China Telecom Hangzhou, China yingjiangyong@chinatelecom.cn

Lan Ma   
China Telecom   
Xi’an, China   
malan@chinatelecom.cn   
Haipeng Du   
MOEKLINNS, and School of   
Computer Science and Technology   
Xi’an Jiaotong University   
Xi’an, China   
duhaipeng@xjtu.edu.cn

## Abstract

While Multimodal Large Language Models (MLLMs) exhibit strong generalization, visual instruction tuning for downstream tasks inevitably causes catastrophic forgetting, impairing overall generalization. While existing methods regulate weight updates to reduce forgetting, they overlook the fundamental cross-modal alignment in MLLMs. Based on prior work and our observations, we argue that cross-modal alignment is implicitly captured in the informationcompression trajectory. To preserve the alignment flow embedded in the trajectory, we propose LLaVAFlow, an information-theoretic distillation framework. First, we compress the mutual information between the extracted relations and MLLM embeddings, encouraging a learnable module to produce a refined alignment flow that benefits downstream tasks. Second, we maximize the mutual information between the extracted alignment flows of the pretrained and fine-tuned MLLMs, enabling the transfer of compact alignment information. Extensive experiments show that LLaVAFlow is an efective plug-and-play framework that preserves alignment flow and enhances both downstream performance and generalization.

CCS Concepts • Computing methodologies → Artificial intelligence.

## Keywords

Multimodal Large Language Models, Visual Instruction Tuning, Multimodal Alignment, Information Theory, Distillation

ACM Reference Format: Muyao Yuan, Muyan Jiao, Jiangyong Ying, Weizhan Zhang, Yuanhong Zhang, Lan Ma, Yuan Gao, and Haipeng Du. 2026. LLaVAFlow: Preserving Latent Alignment Flow for Parameter-Eficient Multimodal Fine-Tuning. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308.3835680

## 1 Introduction

Multimodal large language models (MLLMs) exhibit strong generalization, enabling efective transfer across modalities and tasks [5, 17, 22]. To enhance MLLM performance on specific downstream tasks, visual instruction tuning [23] is widely adopted. However, visual instruction tuning introduces significant generalization risks, as it disrupts the original feature space of the pretrained MLLM. Extensive studies [7, 10, 18] have shown that fine-tuning can lead to catastrophic forgetting, and this problem becomes even more severe in MLLMs because of their complex architectures, larger modality gaps, and stronger task interference, making it particularly challenging to balance general knowledge retention with improvements on downstream tasks [36, 39, 50].

To mitigate generalization issues, existing approaches [19, 46, 56] constrain weight updates through regularization and pruning to limit harmful information and reduce conflicts with the model’s prior general knowledge. However, these methods operate solely on weight-level analysis and largely ignore multimodal information. A central characteristic of MLLMs is their cross-modal alignment, which has often been overlooked during downstream transfer [13, 54]. As a result, existing fine-tuning methods inadvertently disrupt the alignment between visual and textual features, a key cause of catastrophic forgetting, in which the model fails to ground images to their correct semantics, degrading its comprehension and reasoning capabilities.

<table><tr><td></td><td></td><td rowspan=1 colspan=1>aire</td><td rowspan=1 colspan=1>ni</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>aire</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1>è</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>ni</td><td rowspan=1 colspan=1>edad</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ni</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1>→</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>loc</td><td rowspan=1 colspan=1>ppi</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>loc</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>loc</td><td rowspan=1 colspan=1>loc</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>loc</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>1   Deviation</td><td rowspan=1 colspan=1>from</td><td rowspan=1 colspan=1>visual</td><td rowspan=1 colspan=2>features</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>cat</td><td rowspan=1 colspan=1>blue</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>cat</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>LoRASculpt No</td><td></td><td rowspan=1 colspan=1>cat</td><td rowspan=1 colspan=1>blue</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>cat</td><td></td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>Ours       Yes</td><td></td><td rowspan=1 colspan=1>cat</td><td rowspan=1 colspan=1>blue</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>cat</td><td></td></tr></table>

Figure 1: Motivation. We map visual tokens to semantic words along the forward pass using Logit Lens [32]. Existing methods [19] disrupt the alignment within the feature flow, leading to incorrect predictions. By preserving the alignment flow, we efectively mitigate forgetting.

Since cross-modal alignment directly afects downstream performance, understanding how it forms is essential for preserving it. Existing studies [31, 53] show that in MLLMs, visual and textual embeddings align and fuse mainly within the LLM backbone. Specifically, as illustrated in Figure 1, when the image embedding of a cat patch is processed, we capture its semantic evolution by applying the LM head to the hidden states (known as Logit Lens [32]). The representation transitions from an initially non-semantic state into the explicit text token: cat. However, after fine-tuning, this trajectory is disrupted and the model can no longer arrive at the correct semantic interpretation. This observation suggests that mit igating catastrophic forgetting of cross-modal alignment amounts to preserving the underlying semantic trajectory. We term this trajectory the alignment flow, through which visual features are progressively grounded into textual semantics.

To preserve the alignment flow during downstream fine-tuning, a natural approach is to distill it from the pretrained model into the fine-tuned one. However, this poses two key challenges. First, the alignment flow is inherently latent [31], and the internal representation of the LLM backbone contains significant noise irrelevant to both modality alignment and downstream tasks. Second, existing distillation methods [2, 9, 37] typically rely on rigid mimicry of fi nal output distributions or individual features, which is inadequate for preserving modality-dependent feature flows and impedes the model’s ability to acquire new task-specific knowledge.

This motivates the need for a method that 1) extracts alignment information from thefeature trajectory and 2) enables its efective transfer between models, mitigating catastrophic forgetting while facilitating the acquisition of new knowledge.

Building on this insight, we propose LLaVAFlow, an informationtheoretic distillation framework. To tackle the first challenge, we introduce a novel Alignment Flow Module (AFM) that models the alignment flow via inter-embedding relations, enabling eficient extraction of modality alignment information. By compressing the mutual information between the extracted relations and the original MLLM embeddings, the AFM produces a refined alignment flow beneficial for downstream tasks. To address the second challenge, we maximize the mutual information between the AFM-extracted alignment flows of the pretrained and fine-tuned MLLMs. Unlike rigid mimicry, this objective leverage mutual information to preserve structural knowledge and transfer compact alignment information. Notably, these two objectives jointly realize the information bottleneck principle, preserving alignment-relevant information while discarding irrelevant input.

Extensive experiments on VQA and Captioning tasks demonstrate that LLaVAFlow is an efective plug-and-play framework, achieving high downstream performance while mitigating generalization loss in MLLMs through alignment flow preservation. Our contributions are threefold:

❶ A Novel Perspective on Alignment Flow. We introduce alignment flow as a new lens for preserving pretrained knowledge, encoding key cross-modal information within feature propagation.

❷ Information-Theoretic Distillation Framework. We propose LLaVAFlow, which extracts and transfers alignment information between pretrained and fine-tuned MLLMs, mitigating catastrophic forgetting while efectively acquiring new knowledge.

❸ Comprehensive Experimental Validation. Extensive experiments on VQA and captioning demonstrate that LLaVAFlow efectively preserves alignment flow, improves downstream performance, and mitigates generalization loss.

## 2 Related Works

## 2.1 Catastrophic Forgetting

Deep learning models often sufer from catastrophic forgetting, where acquiring new knowledge causes the loss of previously learned information [16]. Various continual learning strategies have been proposed to tackle this problem, usually classified as rehearsalbased [3, 26], regularization-based [16, 25], or architecture-based methods [35, 42]. For large foundation models, the challenge has shifted to preserving the general knowledge encoded in the pretrained model after task-specific fine-tuning. Many methods have been proposed to mitigate catastrophic forgetting in LLM [34, 44, 46]. However, they show limited efectiveness on MLLMs due to the complex architecture and non-uniform weight distributions resulting from their cross-modal nature [56]. For catastrophic forgetting in MLLMs, the EMT [50] evaluates LLaVA and InstructBLIP using classification task performance but does not examine broader multimodal tasks. Model Tailor [56] ofers a post-training adjust ment method that preserves most pre-trained parameters while selectively updating a small fraction of weights. LoRASculpt [19] mitigates conflicts between incremental and original weights and prunes harmful updates during training, efectively maintaining performance on previous tasks while accommodating new ones. However, these methods primarily focus on weight updates and overlook the loss of modality alignment during fine-tuning. To address this, we propose LLaVAFlow, an orthogonal approach that preserves essential alignment information in feature propagation, efectively mitigating catastrophic forgetting in MLLMs.

## 2.2 Knowledge Distillation

Knowledge distillation (KD) aims to transfer knowledge from a large, well-trained teacher model to a smaller, lightweight student model [11, 48]. Early distillation methods for VLMs are primarily designed prior to the foundation model era [8, 41, 43]. Although these works provide valuable insights into multimodal knowledge transfer, they are inadequate for capturing the complex cross-modal alignment inherent in modern MLLMs [9]. Recent eforts have begun to explore distillation tailored specifically for MLLMs. VLM-KD [52] leverages LLaVA-Next to generate textual prompts and employs contrastive learning to enhance long-tail recognition in vision models. LLaVA-MoD [37] introduces a progressive distillation framework that equips the small-scale MLLM (s-MLLM) with a sparse Mixture-of-Experts (MoE) architecture and transfers knowledge from the large-scale MLLM (l-MLLM) through KL-based mimic learning followed by preference optimization. LLaVA-KD [2] proposes a multimodal distillation framework that transfers crossmodal representations and visual token relations via MDist and RDist losses, improving both alignment quality and multimodal understanding in s-MLLMs through a three-stage training scheme. Align-KD [9] introduces a cross-modal distillation method that explicitly transfers shallow-layer alignment knowledge and vision-totext projection capabilities from teacher VLMs, enhancing student performance without increasing model size. Despite these advances, existing methods predominantly focus on transferring knowledge from l-MLLMs to s-MLLMs while overlooking the alignment flow information, which is crucial for preserving general knowledge during task-specific fine-tuning and mitigating catastrophic forgetting. In contrast, our method addresses this gap by explicitly modeling and preserving alignment flow throughout the distillation process.

## 3 Preliminaries

## 3.1 Matrix-Based Rényi’s �-entropy

Rooted in information theory, Rényi’s �-entropy generalizes Shannon entropy and provides a flexible measure for quantifying both single-variable information and interactions among variables. It is defined for a continuous random variable � with probability density function $p ( x )$ over a finite set $\chi$ as:

$$
H _ { \alpha } ( X ) = { \frac { 1 } { 1 - \alpha } } \log \int _ { X } { p ^ { \alpha } ( x ) d x } ,\tag{1}
$$

where $H _ { \alpha } ( X )$ recovers to Shannon entropy as $\alpha  1$ . However, its computation depends on the probability density function $p ( x )$ which is generally inaccessible in MLLM learning. Matrix-Based Rényi’s �-entropy overcomes this limitation by directly estimating entropy from data through the eigenspectrum of a Gram matrix in a reproducing kernel Hilbert space (RKHS).

For a single variable, let $\kappa : \mathcal { X } \times \mathcal { X }  \mathcal { X }$ R be a positive, infinitely divisible kernel [1]. Given sampled data $\{ x _ { i } \} _ { i = 1 } ^ { n } \subset X _ { : }$ , where each �<sub>�</sub> is a real-valued scalar or vector, the Gram matrix � is constructed with entries $K _ { i j } = \kappa ( x _ { i } , x _ { j } )$ . The matrix-based Rényi’s �-entropy is defined on a positive semi-definite matrix A with unit trace as:

$$
S _ { \alpha } ( \mathbf { A } ) = \frac { 1 } { 1 - \alpha } \log \left( \mathrm { t r } ( \mathbf { A } ^ { \alpha } ) \right) = \frac { 1 } { 1 - \alpha } \log \left( \sum _ { i = 1 } ^ { n } \lambda _ { i } ^ { \alpha } ( \mathbf { A } ) \right) ,\tag{2}
$$

where $\lambda _ { i } ( \mathbf { A } )$ denotes the �-th eigenvalue of $\mathbf { A } .$ . In practice, A is constructed via a normalized kernel matrix:

$$
{ \bf A } _ { i j } = \frac { 1 } { n } \frac { K _ { i j } } { \sqrt { K _ { i i } K _ { j j } } } ,\tag{3}
$$

which preserves positive semi-definiteness and satisfies $\operatorname { t r } ( \mathbf { A } ) = 1$

For multiple variables, let $\kappa _ { 1 } : \mathcal { X } _ { 1 } \times \mathcal { X } _ { 1 } \to \mathbb { R } , \ldots , \kappa _ { L } : \mathcal { X } _ { L } \times \mathcal { X } _ { L } \to$ R be positive, infinitely divisible kernels, and let $\{ x _ { i } ^ { 1 } , . . . , x _ { i } ^ { L } \} _ { i = 1 } ^ { n } \subset$ $X _ { 1 } \times \cdots \times X _ { L }$ denote a set of � samples. The matrix-based Rényi �-order joint entropy of these � variables is defined as

$$
S _ { \alpha } ( \mathbf { A } _ { 1 } , \ldots , \mathbf { A } _ { L } ) = S _ { \alpha } \left( \frac { \mathbf { A } _ { 1 } \circ \dots \circ \mathbf { A } _ { L } } { \operatorname { t r } ( \mathbf { A } _ { 1 } \circ \dots \circ \mathbf { A } _ { L } ) } \right) ,\tag{4}
$$

where $\mathbf { A } _ { 1 } , \dotsc , \mathbf { A } _ { L }$ are the normalized kernel matrices for each variable, and ◦ denotes the Hadamard product.

## 3.2 Matrix-Based Mutual Information

Since mutual information can be expressed in terms of entropy, the mutual information between ${ \bf A } _ { 1 }$ and $\mathbf { A } _ { 2 } ,$ , denoted as $I _ { \alpha } ( \mathbf { A } _ { 1 } ; \mathbf { A } _ { 2 } )$ , is given by

$$
I _ { \alpha } ( { \bf A } _ { 1 } ; { \bf A } _ { 2 } ) = S _ { \alpha } ( { \bf A } _ { 1 } ) + S _ { \alpha } ( { \bf A } _ { 2 } ) - S _ { \alpha } ( { \bf A } _ { 1 } , { \bf A } _ { 2 } ) .\tag{5}
$$

This matrix-based approach avoids high-dimensional probability density estimation, providing a more accurate and computationally eficient formulation for mutual information [47].

## 4 Method

## 4.1 Overview of LLaVAFlow

The overview of LLaVAFlow is presented in Figure 2. LLaVAFlow adopts a distillation paradigm, where a frozen pretrained MLLM serves as the teacher, and a parameter-eficiently fine-tuned model trained from it acts as the student. The framework comprises an Alignment Flow Module (AFM) and an Information Bottleneck Distillation (IBD) that jointly regulate the fine-tuning of the MLLM.

The AFM captures multimodal alignment during feature propagation. Specifically, based on data processing inequality (DPI), it takes the embeddings before the first layer and after the final layer of the LLM backbone as inputs, and extracts an alignment flow that characterizes the evolution of multimodal representations across the backbone. The AFM is attached to both the teacher and student models with shared parameters.

Based on this alignment flow, we formulate distillation as an information bottleneck objective composed of two coupled terms. The Flow Refinement Loss serves as the compression term on the AFM of the teacher model by minimizing the mutual information between the alignment flow and the LLM backbone embeddings, thereby suppressing noise in the extracted flow. The Alignment Preservation Loss acts as the distillation term by maximizing the mutual information between the alignment representations of the pretrained and fine tuned MLLMs, which transfers alignment knowledge from the pretrained model and prevents the degradation of flow alignment during adaptation.

![](images/901a51a63bf0dc4e3be9010c45b8777d8444d55671b05f1098320e915d3de05b.jpg)  
Figure 2: Overview of LLaVAFlow. LLaVAFlow is a distillation framework that efectively transfers alignment flow from a pretrained MLLM to its fine-tuned counterpart from an information-theoretic perspective. The framework comprises an Alignment Flow Module (AFM) and an Information Bottleneck Distillation (IBD) objective that jointly regulate MLLM fine tuning: the Flow Refinement Loss drives the AFM to capture refined alignment flow, while the Alignment Preservation Loss efectively transfers it to the student. The T-MLLM (teacher) and S-MLLM (student) share the same AFM parameters.

## 4.2 Alignment Flow Module (AFM)

The Alignment Flow Module (AFM) is designed to capture the evolution of multimodal alignment within a model. Intuitively, valuable alignment information can be extracted from the trajectory of multimodal embeddings. Let $\{ h _ { l } \} _ { l = 0 } ^ { L }$ represent the multimodal embeddings at each layer of an �-layer LLM backbone (ℎ<sub>0</sub> denotes the embeddings before the first layer). In principle, the AFM can be formulated as a function of all layer embeddings to obtain the alignment flow $F _ { a } \colon$

$$
F _ { a } = f _ { A F M } ( h _ { 0 } , h _ { 1 } , . . . , h _ { L } ) .\tag{6}
$$

However, performing computations across all hidden embeddings is impractical. Since each layer in the LLM backbone processes inputs exclusively from its immediate predecessor, the model efectively forms a Markov chain across its layers [40]. Let � represent the output of the multimodal task. According to the data processing inequality (DPI) [6, 15, 40], the mutual information between the representation at each layer and the final output must satisfy:

$$
I ( h _ { 0 } ; Y ) \ge I ( h _ { 1 } ; Y ) \ge \cdots \ge I ( h _ { L } ; Y ) .\tag{7}
$$

This implies that as representations propagate through the layers, the quantity of task-relevant information is non-increasing. Empirically, after training, the final-layer embedding $h _ { L }$ retains nearly

all task-relevant information from the input ℎ<sub>0</sub>. Consequently, the AFM can efectively capture the core multimodal alignment using only the input and output embeddings:

$$
F _ { a } = f _ { A F M } ( h _ { 0 } , h _ { L } ) ,\tag{8}
$$

which significantly reduces computational cost while preserving essential alignment information.

Specifically, the AFM learns a relation between $h _ { 0 }$ and $h _ { L }$ to characterize the alignment flow. We model this relation using a similarity term and further introduce an adaptive attention mechanism as a residual compensation to refine it. Given $h _ { 0 } , h _ { L } \in \mathbb { R } ^ { B \times N \times D }$ where �, �, and � denote the batch size, the number of multimodal tokens, and the embedding dimension, respectively, the AFM computes a relation map $F _ { a } ~ \mathbf { \bar { \mu } } \in \mathbb { R } ^ { B \times N \times N }$ , where each entry reflects the relevance between an input token and an output token. Let $\tilde { h } _ { L } \ = \ \mathrm { L a y e r N o r m } ( h _ { L } )$ and $\tilde { h } _ { 0 } \ = \ \mathrm { L a y e r N o r m } ( h _ { 0 } )$ . The base similarity is computed by the dot product of the normalized embeddings $\tilde { h } _ { L } \tilde { h } _ { 0 } ^ { \top }$ . To refine this relation, we introduce a residual component implemented with a scaled dot-product attention mechanism: $Q \ = \ W _ { Q } \tilde { h } _ { L } , \ K \ = \ W _ { K } \tilde { h } _ { 0 }$ , where $W _ { Q } , W _ { K } \ \in \ \mathbb { R } ^ { D \times D }$ are learnable projection matrices. The final alignment flow is computed as

$$
F _ { a } = f _ { A F M } ( h _ { 0 } , h _ { L } ) = \tilde { h } _ { L } \tilde { h } _ { 0 } ^ { \top } + Q K ^ { \top } / \sqrt { D } .\tag{9}
$$

In this way, the AFM can adaptively adjust how the relation between ℎ and $h _ { L }$ is modeled, providing a robust and efective mechanism to capture alignment information in the flow.

## 4.3 Information Bottleneck Distillation (IBD)

The goal of Information Bottleneck Distillation (IBD) is to transfer the refined alignment flow from the teacher to the student. Specifically, IBD addresses two key questions: how to capture critical alignment information from the teacher’s feature propagation and how to efectively preserve it in the student during fine-tuning. Accordingly, we propose the Flow Refinement Loss and the Alignment Preservation Loss under the information bottleneck principle.

Flow Refinement Loss. While the AFM described in Sec. 4.2 extracts alignment flow from the embeddings before the first layer and after the final layer, these embeddings contain substantial redundancy beyond the alignment information useful for the downstream task. To address this issue, we adopt the compression principle of the information bottleneck to refine the alignment flow.

Given the teacher MLLM’s input embeddings $h _ { 0 } ^ { T }$ and output embeddings $h _ { L } ^ { T }$ , the AFM produces the corresponding alignment flow $F _ { a } ^ { T }$ . Our objective is to regulate this flow by minimizing the mutual information $I _ { \alpha } ( h _ { 0 } ^ { T } , h _ { L } ^ { \breve { T } } ; F _ { a } ^ { T } )$ . Here, $I _ { \alpha } ( \cdot ; \cdot )$ denotes the �-Rényi mutual information, which quantifies the amount of information shared between the backbone embeddings and the extracted alignment flow. This objective encourages $F _ { a } ^ { T }$ to discard redundant information from the embeddings while retaining the essential structure of multimodal alignment. In other words, this can be formulated as a loss to supervise the learning of AFM:

$$
\mathcal { L } _ { F R } = I _ { \alpha } ( h _ { 0 } ^ { T } , h _ { L } ^ { T } ; F _ { a } ^ { T } )\tag{10}
$$

To make Eq. 10 computationally tractable in MLLM training, we leverage the matrix-based mutual information in Eq. 5 to reformulate the objective.

$$
\begin{array} { r l } & { \mathcal { L } _ { F R } = I _ { \alpha } ( h _ { 0 } ^ { T } , h _ { L } ^ { T } ; F _ { a } ^ { T } ) } \\ & { \qquad = S _ { \alpha } ( { \bf G } _ { 0 } ^ { T } , { \bf G } _ { L } ^ { T } ) + S _ { \alpha } ( { \bf G } _ { F } ^ { T } ) - S _ { \alpha } ( { \bf G } _ { 0 } ^ { T } , { \bf G } _ { L } ^ { T } , { \bf G } _ { F } ^ { T } ) , } \end{array}\tag{11}
$$

where ${ \bf G } _ { 0 } ^ { T } , { \bf G } _ { L } ^ { T }$ , and ${ \bf G } _ { F } ^ { T } \in \mathbb { R } ^ { B \times B }$ denote the normalized Gram matrices defined in $\operatorname { E q . 3 , }$ constructed from the embeddings $h _ { 0 } ^ { T } , h _ { L } ^ { T }$ and the alignment flow $F _ { a } ^ { T }$ , respectively. Here, � denotes the batch size. In our implementation, all Gram matrices are computed using a degree-1 polynomial kernel [49]. As the teacher model remains frozen during training, the term $S _ { \alpha } ( \mathbf { G } _ { 0 } ^ { T } , \mathbf { G } _ { L } ^ { T } )$ , which depends only on the teacher embeddings, can be omitted. So $\mathcal { L } _ { F R }$ can be reformulated using Eq. 2 and Eq. 4:

$$
\begin{array} { r l } & { \mathcal { L } _ { F R } = S _ { \alpha } ( { \mathbf { G } } _ { F } ^ { T } ) - S _ { \alpha } ( { \mathbf { G } } _ { 0 } ^ { T } , { \mathbf { G } } _ { L } ^ { T } , { \mathbf { G } } _ { F } ^ { T } ) } \\ & { \quad \quad = \frac { 1 } { 1 - \alpha } \left( \log \displaystyle \sum _ { i = 1 } ^ { B } \lambda _ { i } ^ { \alpha } ( { \mathbf { G } } _ { F } ^ { T } ) - \log \displaystyle \sum _ { i = 1 } ^ { B } \lambda _ { i } ^ { \alpha } ( { \mathbf { G } } _ { 0 L F } ^ { T } ) \right) , } \end{array}\tag{12}
$$

where $\mathbf { G } _ { 0 L F } ^ { T } = ( \mathbf { G } _ { 0 } ^ { T } \circ \mathbf { G } _ { L } ^ { T } \circ \mathbf { G } _ { F } ^ { T } ) / \mathrm { t r } ( \mathbf { G } _ { 0 } ^ { T } \circ \mathbf { G } _ { L } ^ { T } \circ \mathbf { G } _ { F } ^ { T } )$ , and ◦ denotes the Hadamard product. To avoid the computational cost of eigenvalue calculation, we follow prior work [47] by setting $\alpha \ = \ 2 ,$ which allows a Frobenius-norm-based approximation of Rényi’s �-entropy [29, 51]:

$$
| | \mathbf { G } | | _ { F } ^ { 2 } = \mathrm { t r } ( \mathbf { G G } ^ { \top } ) = \sum _ { i = 1 } ^ { B } \lambda _ { i } ^ { 2 } ( \mathbf { G } ) ,\tag{13}
$$

yielding the final computable loss:

$$
\mathcal { L } _ { F R } = - \log | | \mathbf { G } _ { F } ^ { T } | | _ { F } ^ { 2 } + \log | | \mathbf { G } _ { 0 L F } ^ { T } | | _ { F } ^ { 2 } .\tag{14}
$$

In this way, the Flow Refinement Loss enforces a compact representation of the alignment flow, reducing noise while preserving its core multimodal structure.

Alignment Preservation Loss. Under the supervision of the Flow Refinement Loss, the AFM acts as an efective lens for extracting key alignment information from the flow. To preserve this information during fine-tuning, we incorporate the AFM into both teacher and student models with shared parameters, aiming to align the student’s flow $F _ { a } ^ { S }$ with the teacher’s $F _ { a } ^ { T }$ and thereby guide the alignment process in the student model.

Specifically, we achieve this transfer by maximizing the mutual information between the alignment flows of the pretrained and fine-tuned models. The Alignment Preservation Loss is defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { A P } = - I _ { \alpha } ( F _ { a } ^ { T } ; F _ { a } ^ { S } ) } \\ & { \qquad = - S _ { \alpha } ( \mathbf { G } _ { F } ^ { T } ) - S _ { \alpha } ( \mathbf { G } _ { F } ^ { S } ) + S _ { \alpha } ( \mathbf { G } _ { F } ^ { T } , \mathbf { G } _ { F } ^ { S } ) , } \end{array}\tag{15}
$$

where $\mathbf { G } _ { F } ^ { T } , \mathbf { G } _ { F } ^ { S } \in \mathbb { R } ^ { B \times B }$ denote the normalized Gram matrices of $F _ { a } ^ { T }$ and $F _ { a } ^ { S }$ , respectively. Analogous to 14, the computable form of the loss is:

$$
\mathcal { L } _ { A P } = \log \| { \bf G } _ { F } ^ { T } \| _ { F } ^ { 2 } + \log \| { \bf G } _ { F } ^ { S } \| _ { F } ^ { 2 } - \log \| { \bf G } _ { F } ^ { T S } \| _ { F } ^ { 2 } ,\tag{16}
$$

where ${ \bf G } _ { F } ^ { T S } = ( { \bf G } _ { F } ^ { T } \circ { \bf G } _ { F } ^ { S } ) / \mathrm { t r } ( { \bf G } _ { F } ^ { T } \circ { \bf G } _ { F } ^ { S } )$ . The Alignment Preservation Loss is based on mutual information and difers from traditional distillation. $\operatorname { E q . }$ 16 provides a more relaxed imitation from the student to the teacher, transferring only the most relevant information and mitigating the interference of the distillation loss on the task loss.

Overall Loss and Information-Theoretic View. To jointly optimize task performance and alignment preservation, our overall training objective combines the downstream task loss with proposed distillation losses. Specifically, we incorporate the task loss (i.e., SFT loss), denoted as $\mathcal { L } _ { t a s k }$ , together with the Flow Refinement Loss and the Alignment Preservation Loss:

$$
\mathcal { L } = \mathcal { L } _ { t a s k } + \lambda _ { F R } \mathcal { L } _ { F R } + \lambda _ { A P } \mathcal { L } _ { A P } ,\tag{17}
$$

where $\lambda _ { F R }$ and $\lambda _ { A P }$ balance the contributions of the two distillation losses. Focusing on the distillation component, the distillation loss can be written as $\mathcal { L } _ { d } = \lambda _ { F R } \mathcal { L } _ { F R } + \lambda _ { A P } \mathcal { L } _ { A P } $ . In terms of mutual information, following Eq. 10 and Eq. 15, this corresponds to

$$
\operatorname* { m a x } I _ { \alpha } ( F _ { a } ^ { T } ; F _ { a } ^ { S } ) - \frac { \lambda _ { F R } } { \lambda _ { A P } } I _ { \alpha } ( h _ { 0 } ^ { T } , h _ { L } ^ { T } ; F _ { a } ^ { T } ) ,\tag{18}
$$

which enforces a compact representation that selectively preserves the most relevant alignment flow, and facilitates robust alignment transfer from teacher to student. This is consistent with the information bottleneck principle [33], retaining target-relevant information while discarding irrelevant input.

## 5 Experiments

## 5.1 Experimental Setup

Datasets. To evaluate the efectiveness of LLaVAFlow, we follow prior work [19, 22, 23] and fine-tune the model on Visual Question Answering (VQA) and Captioning tasks using the ScienceQA [27] and COCO-Caption [21] datasets. To assess general knowledge retention in the MLLM, we evaluate its performance on four upstream datasets after fine-tuning on the downstream tasks, including OKVQA [28], OCRVQA [30], GQA [14], and TextVQA [38]. We report Accuracy for VQA tasks and CIDEr for Captioning.

Table 1: Comparison with state-of-the-art methods on the visual question answering task ScienceQA and the image captioning task COCO-Caption. Source indicates the average performance across upstream datasets, Target reports the performance on the fine-tuned dataset, and Avg is the mean of Source and Target.
<table><tr><td rowspan="2">Methods</td><td colspan="6">ScienceQA</td><td colspan="8">COCO-Caption</td></tr><tr><td>ⅡI ∥| OKVQA</td><td>OCRVQA GQA</td><td></td><td>TextVQA</td><td>Source</td><td>Target</td><td>Avg</td><td>OKVQA OCRVQA</td><td></td><td>GQA</td><td>TextVQA</td><td>Source</td><td>Target</td><td>Avg</td></tr><tr><td>Zero-shot</td><td>58.00</td><td>66.20</td><td>61.94</td><td>58.21</td><td>61.09</td><td>70.22</td><td>65.65</td><td>58.00</td><td>66.20</td><td>61.94</td><td>58.21</td><td>61.09</td><td>40.34</td><td>50.71</td></tr><tr><td>LoRA Rank = 32</td><td colspan="8"></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td>56.99</td><td>65.25</td><td>61.52</td><td>56.64</td><td>60.10</td><td>87.57</td><td>73.84</td><td>48.60</td><td>66.00</td><td>56.73</td><td>43.91</td><td>53.81</td><td>128.46</td><td>91.14</td></tr><tr><td>LoRA+Ours</td><td>56.94</td><td>65.65</td><td>61.19</td><td>57.07</td><td>60.21</td><td>89.46</td><td>74.84</td><td>52.18</td><td>67.25</td><td>59.33</td><td>51.20</td><td>57.49</td><td>130.04</td><td>93.76</td></tr><tr><td>DoRA</td><td>55.10</td><td>61.05</td><td>59.68</td><td>54.18</td><td>57.50</td><td>88.94</td><td>73.22</td><td>49.69</td><td>66.50</td><td>57.62</td><td>45.81</td><td>54.90</td><td>128.26</td><td>91.58</td></tr><tr><td>DoRA+Ours</td><td>56.86</td><td>65.85</td><td>61.05</td><td>56.72</td><td>60.12</td><td>89.13</td><td>74.62</td><td>51.59</td><td>67.25</td><td>59.63</td><td>51.46</td><td>57.48</td><td>129.97</td><td>93.73</td></tr><tr><td>LLaVAMod</td><td>57.51</td><td>65.10</td><td>61.71</td><td>56.96</td><td>60.32</td><td>87.55</td><td>73.94</td><td>55.34</td><td>67.10</td><td>61.39</td><td>55.26</td><td>59.77</td><td>125.99</td><td>92.88</td></tr><tr><td>LLaVAMod+Ours</td><td>57.35</td><td>66.10</td><td>61.46</td><td>57.36</td><td>60.57</td><td>88.56</td><td>74.57</td><td>56.31</td><td>67.00</td><td>61.32</td><td>57.14</td><td>60.44</td><td>128.13</td><td>94.29</td></tr><tr><td>LLaVAKD</td><td>57.81</td><td>66.15</td><td>62.01</td><td>57.53</td><td>60.88</td><td>88.38</td><td>74.63</td><td>55.49</td><td>67.05</td><td>61.40</td><td>55.58</td><td>59.88</td><td>126.87</td><td>93.38</td></tr><tr><td>LLaVAKD+Ours</td><td>57.80</td><td>66.25</td><td>61.82</td><td>57.98</td><td>60.96</td><td>88.73</td><td>74.85</td><td>55.91</td><td>67.15</td><td>61.08</td><td>56.71</td><td>60.21</td><td>129.02</td><td>94.62</td></tr><tr><td>DARE</td><td>56.32</td><td>64.50</td><td>61.46</td><td>55.75</td><td>59.51</td><td>87.46</td><td>73.48</td><td>46.26</td><td>65.85</td><td>54.61</td><td>39.75</td><td>51.62</td><td>128.32</td><td>89.97</td></tr><tr><td>DARE+Ours</td><td>56.21</td><td>66.15</td><td>61.13</td><td>56.95</td><td>60.11</td><td>89.37</td><td>74.74</td><td>51.49</td><td>66.70</td><td>59.06</td><td>50.50</td><td>56.94</td><td>129.23</td><td>93.08</td></tr><tr><td>Model Tailor</td><td>57.10</td><td>65.50</td><td>61.58</td><td>57.10</td><td>60.32</td><td>87.24</td><td>73.78</td><td>49.37</td><td>66.05</td><td>57.77</td><td>44.85</td><td>54.51</td><td>128.58</td><td>91.54</td></tr><tr><td>Model Tailor+Ours</td><td>56.93</td><td>66.10</td><td>61.43</td><td>57.11</td><td>60.39</td><td>89.20</td><td>74.80</td><td>52.41</td><td>67.20</td><td>60.17</td><td>52.08</td><td>57.96</td><td>129.32</td><td>93.64</td></tr><tr><td>LoRASculpt</td><td>52.78</td><td>50.60</td><td>56.65</td><td>50.29</td><td>52.58</td><td>85.26</td><td>68.92</td><td>56.86</td><td>66.30</td><td>61.85</td><td>57.59</td><td>60.65</td><td>128.81</td><td>94.73</td></tr><tr><td>LoRASculpt+Ours</td><td>58.00</td><td>64.95</td><td>61.70</td><td>57.54</td><td>60.55</td><td>85.69</td><td>73.12</td><td>57.05</td><td>67.10</td><td>61.68</td><td>57.77</td><td>60.90</td><td>129.40</td><td>95.15</td></tr><tr><td colspan="2">LoRA Rank = 64</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td>52.14</td><td>61.05</td><td>58.90</td><td>50.60</td><td>55.67</td><td>88.49</td><td>72.08</td><td>44.57</td><td>62.90</td><td>51.58</td><td>38.37</td><td>49.36</td><td>128.13</td><td>88.74</td></tr><tr><td>LoRA+Ours</td><td>54.11</td><td>64.15</td><td>59.60</td><td>54.65</td><td>58.13</td><td>89.46</td><td>73.79</td><td>44.10</td><td>66.10</td><td>53.87</td><td>42.86</td><td>51.73</td><td>129.33</td><td>90.53</td></tr><tr><td>DoRA</td><td>51.65</td><td>62.25</td><td>52.19</td><td>50.20</td><td>54.07</td><td>88.40</td><td>71.24</td><td>45.70</td><td>65.70</td><td>54.82</td><td>40.32</td><td>51.64</td><td>127.86</td><td>89.75</td></tr><tr><td>DoRA+Ours</td><td>52.11</td><td>63.85</td><td>57.85</td><td>51.67</td><td>56.37</td><td>88.73</td><td>72.55</td><td>44.30</td><td>66.80</td><td>54.12</td><td>42.97</td><td>52.05</td><td>129.20</td><td>90.62</td></tr><tr><td>LLaVAMod</td><td>55.79</td><td>64.05</td><td>61.23</td><td>54.75</td><td>58.96</td><td>88.45</td><td>73.70</td><td>53.07</td><td>66.95</td><td>60.39</td><td>52.25</td><td>58.17</td><td>126.75</td><td>92.46</td></tr><tr><td>LLaVAMod+Ours</td><td>56.28</td><td>64.35</td><td>60.50</td><td>56.65</td><td>59.44</td><td>88.75</td><td>74.10</td><td>52.68</td><td>67.15</td><td>60.13</td><td>53.21</td><td>58.29</td><td>128.39</td><td>93.34</td></tr><tr><td>LLaVAKD</td><td>56.83 57.38</td><td>66.05</td><td>61.19</td><td>56.74</td><td>60.20</td><td>88.63</td><td>74.42</td><td>55.03</td><td>66.60</td><td>61.54</td><td>55.28</td><td>59.61</td><td>125.97</td><td>92.79</td></tr><tr><td>LLaVAKD+Ours</td><td>50.75</td><td>66.60</td><td>61.19</td><td>56.97</td><td>60.54</td><td>88.80</td><td>74.67</td><td>54.79</td><td>67.40</td><td>60.88</td><td>55.39</td><td>59.61</td><td>127.98</td><td>93.80</td></tr><tr><td>DARE</td><td>52.83</td><td>58.80</td><td>58.16 59.91</td><td>48.87</td><td>54.14</td><td>88.35</td><td>71.25</td><td>42.17</td><td>61.55</td><td>47.42 52.31</td><td>31.34</td><td>45.62</td><td>127.50</td><td>86.56</td></tr><tr><td>DARE+Ours</td><td>55.00</td><td>63.65 63.55</td><td>59.83</td><td>54.02 54.15</td><td>57.60 58.13</td><td>89.32 88.45</td><td>73.46 73.29</td><td>42.98 45.65</td><td>66.00 63.85</td><td>53.57</td><td>36.42 39.66</td><td>49.43 50.68</td><td>128.76 128.42</td><td>89.09 89.55</td></tr><tr><td>Model Tailor Model Tailor+Ours</td><td>54.86 53.83</td><td>65.85</td><td>60.23</td><td>55.80</td><td>59.18</td><td>89.46</td><td>74.32</td><td>45.19</td><td>66.30</td><td>55.05</td><td>44.19 52.68</td></table>

Implementation Details All compared baselines are adapted using LoRA and implemented within the LLaVA-1.5 framework [22]. Following prior work [19, 22], we apply LoRA with ranks of 32 and 64 to all layers of the LLM using a learning rate of 2e-4, and perform full fine-tuning in the connector with a learning rate of 2e-5. Training is conducted for one epoch with a batch size of 16. All experiments are carried out on a single NVIDIA A100 80GB GPU. The key hyperparameters are set to $( \lambda _ { F R } , \lambda _ { A P } ) = ( 1 . 5 , 0 . 5 )$ for VQA and (1.5, 2.0) for Captioning.

Baselines. We evaluate LLaVAFlow in a plug-and-play fashion, integrating it with existing state-of-the-art methods. This demonstrates its ability to preserve alignment flow, which consistently enhances both downstream performance and model generalization. The evaluation covers a diverse set of baselines, including parameter-eficient fine-tuning methods (LoRA [12], DoRA [24]), MLLM distillation approaches (LLaVAMod [37], LLaVAKD [2]), post-training methods (DARE [46], Model Tailor [56]), and regularization strategies (LoRASculpt [19]). Details of the compared baselines are provided in Appendix A.

## 5.2 Main Results

Comparison to State-of-the-Arts. We conduct extensive experiments by equipping LLaVAFlow with various state-of-the-art baselines across diferent rank settings on two downstream tasks. As shown in Table 1, LLaVAFlow brings notable gains on both Source and Target datasets, indicating that preserving alignment flow helps retain pretrained knowledge and facilitate downstream adaptation simultaneously. Importantly, these improvements hold across diverse methods, spanning adapter tuning (LoRA, DoRA), knowledge distillation (LLaVA-KD, LLaVA-Mod), and parameter regularization (DARE, Model Tailor, LoRASculpt), confirming the broad compatibility of our framework. This also suggests that alignment flow preservation ofers a complementary benefit that is largely orthogonal to existing strategies, providing a new and efective angle for mitigating forgetting in MLLMs.

## 5.3 Interpretation of LLaVAFlow

Preservation ofAlignment. To investigate modality alignment, we analyze the alignment trends across layers by measuring the average cosine similarity and normalized Euclidean distance between features at the original vision and text embedding positions across diferent models. The experiments are conducted on ScienceQA [27] (target dataset) and OCRVQA [30] (source dataset), and all fine-tuned models are fine-tuned on ScienceQA [27].

![](images/7d6e0c1d7db838fc4bbda6622d6077af5b7bf6f584fdceae42bab65d7e477816.jpg)  
(a) Cosine similarity on target

![](images/5fefb454aab399a1b4083f477209a46113b6cc6ac393ca17da0f4de48d8180ec.jpg)  
(b) Cosine similarity on source

![](images/24a461a86d745d47b02e64f4ed6b73083dcaad52b7e8a94d1f13f65203a700b7.jpg)  
(c) Euclidean distance on target

![](images/c8c8aee8837a3dab8f6e968b3423ddce4aaefd7d710887c485b6577dca89dd49.jpg)  
(d) Euclidean distance on source

Figure 3: Alignment trends across layers. (a) and (b) show the average cosine similarity between features at the original vision and text embedding positions within each layer on the target and source datasets, respectively. (c) and (d) show the corresponding normalized Euclidean distance. All fine-tuned models are fine-tuned on ScienceQA [27]. Our model preserves the pre-trained modality alignment in the shallow layers (1–25), indicating retention of general cross-modal alignment, while exhibiting divergence in the deeper layers as they adapt to task-specific objectives.  
![](images/3d4e7f4aa403188dbc6a75ceeb2827199dfb7ceaf0640763da31df74b10bdbdd.jpg)

Question：How the piece of furniture that is made of wood is called? Answer the question using a single word or phrase.
<table><tr><td>LLaVA</td><td>Coffee table</td></tr><tr><td>LoRASculpt</td><td>Couch</td></tr><tr><td>Ours</td><td>Coffee table</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=6>LLaVA    LoRASculpt    Ours</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Warsza</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>★</td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>edeut</td><td rowspan=1 colspan=1>Warsza</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>edeut</td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>10</td><td rowspan=2 colspan=1>umerate</td><td rowspan=2 colspan=1>rugu</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>umerate</td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ł</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=2 colspan=1>15</td><td rowspan=2 colspan=2>Export</td><td rowspan=2 colspan=1>totalité</td><td rowspan=2 colspan=2></td></tr><tr><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Devi</td><td rowspan=1 colspan=1>ation fro</td><td rowspan=1 colspan=1>m visual</td><td rowspan=1 colspan=1>features</td><td></td></tr><tr><td rowspan=3 colspan=1>2025</td><td rowspan=1 colspan=1>wood</td><td rowspan=1 colspan=1>wooden</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>wood</td><td></td></tr><tr><td rowspan=2 colspan=1>table</td><td rowspan=2 colspan=1>chair</td><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>table</td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ł</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=2>coffee       chair</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>coffee</td><td></td></tr></table>

![](images/64d2a5ad96db3c7ba4d41ba458e3b9fe91c730a900315435e72dcecb3c749c5c.jpg)  
Question ： What device is sitting next to the mouse pad? Answer the question using a single word or phrase

<table><tr><td>LLaVA</td><td>Keyboard</td></tr><tr><td>LoRASculpt</td><td>Computer</td></tr><tr><td>Ours</td><td>Keyboard</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=3>LLaVA     LoRASculpt</td><td rowspan=1 colspan=1>Ours</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>arab</td><td rowspan=1 colspan=1>arab</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>arab</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>obox</td><td rowspan=1 colspan=1>arab</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>obox</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>chev</td><td rowspan=1 colspan=1>arab</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>bunch</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>iali</td><td rowspan=1 colspan=1>ulaire</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>iali</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>↓     Devi</td><td rowspan=1 colspan=1>ation fro</td><td rowspan=1 colspan=1>m visualf</td><td rowspan=1 colspan=1>eatures</td></tr><tr><td rowspan=2 colspan=1>2025</td><td rowspan=1 colspan=1>mouse</td><td rowspan=1 colspan=1>controls</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mouse</td></tr><tr><td rowspan=1 colspan=1>mouse</td><td rowspan=1 colspan=1>controls</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mouse</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>↓</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=2>mouse</td><td rowspan=1 colspan=1>peri</td><td rowspan=1 colspan=1></td></tr></table>

Figure 4: Comparison of semantic trajectories across models, obtained via Logit Lens [32]. Existing methods, such as LoRASculpt, fail to preserve alignment within the feature flow, leading to incorrect predictions. In contrast, our method retains most of the trajectory of the pre-trained LLaVA, efectively mitigating catastrophic forgetting. The trajectory is derived from the image patch indicated by the red box. Full trajectories are provided in Appendix B.1.

As shown in Figure 3, in the shallow layers (1–25), the cosine similarity and normalized Euclidean distance of our model closely resemble those of the pre-trained LLaVA model. Since these layers primarily capture general knowledge [45, 55], maintaining the orig inal cross-modal alignment is crucial for preserving the representational capacity acquired during pre-training [9, 20]. The consistent modality gap across these layers confirms that our method successfully retains this alignment. In the deeper layers (beyond layer 25), however, the modality gap of our model begins to diverge. This is reasonable, as the deeper layers are more responsible for encoding task-specific knowledge [45, 55], where the alignment naturally shifts to accommodate downstream objectives. This behavior reflects the strength of our information distillation strategy, which focuses on relational knowledge transfer rather than hard mimicry, enabling the model to preserve pre-trained alignment where it matters most while allowing suficient flexibility for task adaptation. This phenomenon also suggests that, although the alignment flow is computed using only the embeddings at the input and output of the network, it efectively regulates the entire forward feature propagation process.

Semantics Flow ofVisual Tokens. To demonstrate alignment flow preservation, we visualize the semantic trajectories of diferent methods by applying the LM head to intermediate hidden states, following the Logit Lens paradigm.

As shown in Figure 4, in Example 1, LoRASculpt loses critical vision-language alignment information along the trajectory and fails to map the image patch of cat to the correct textual token cat, unlike the pre-trained LLaVA model. In contrast, our method preserves most of the trajectory of LLaVA and produces the correct answer, indicating efective alignment flow preservation and mitigation of catastrophic forgetting.

## 5.4 Ablation Studies

Ablation ofInformation Bottleneck Distillation. We conduct an ablation study to evaluate the contribution of each component in our proposed LLaVAFlow: Flow Refinement Loss $\mathcal { L } _ { F R }$ and Alignment Preservation Loss $\mathcal { L } _ { A P } .$ . Experiments are performed on the ScienceQA dataset using LoRA with ranks of 32 and 64. To ablate $\mathcal { L } _ { F R : }$ , we replace the learnable relation module with a fixed dotproduct operation and remove the Flow Refinement Loss. To ablate $\mathcal { L } _ { A P } ,$ , we substitute the information-based loss with an MSE loss. As shown in Table 2, both components contribute positively to the overall performance. Notably, under rank 64, the LoRA baseline sufers from a significant drop in source performance (55.67), indicating catastrophic forgetting. In this setting, $\mathcal { L } _ { A P }$ alone recovers the source performance from 55.67 to 58.71, demonstrating its efectiveness in preserving the pre-trained alignment flow. Meanwhile, $\mathcal { L } _ { F R }$ contributes more to target task adaptation, boosting target accuracy from 88.49 to 88.52, suggesting that the informationbottleneck-based loss efectively filters out noisy signals that would otherwise harm downstream performance. When both losses are combined, the model achieves the highest target accuracy (89.46) while substantially mitigating catastrophic forgetting on the source datasets (58.13), confirming that the two components are complementary.

Table 2: Ablation study of Flow Refinement Loss $\mathcal { L } _ { F R }$ and Alignment Preservation Loss $\mathcal { L } _ { A P }$ . All experiments are conducted with LoRA on the ScienceQA dataset.
<table><tr><td rowspan="2">LFR</td><td rowspan="2">1  $\mathcal { L } _ { A P }$ </td><td colspan="3">LoRA Rank = 32</td><td rowspan="2"></td><td colspan="3">LoRA Rank = 64</td></tr><tr><td>Target</td><td>Source</td><td> $\operatorname { A v g }$ </td><td>1 Target</td><td>Source</td><td> $\operatorname { A v g }$ </td></tr><tr><td>Zero-shot</td><td></td><td>1 70.22</td><td>61.09</td><td></td><td>65.65 1</td><td>70.22</td><td>61.09</td><td>65.65</td></tr><tr><td>x</td><td>x</td><td>87.57</td><td>60.10</td><td></td><td>73.84</td><td>88.49</td><td>55.67</td><td>72.08</td></tr><tr><td>x</td><td>V</td><td>89.08</td><td>60.04</td><td></td><td>74.56</td><td>87.86</td><td>58.71</td><td>73.28</td></tr><tr><td>V</td><td>x</td><td>89.37</td><td>60.05</td><td>74.71</td><td></td><td>88.52</td><td>57.11</td><td>72.81</td></tr><tr><td>√</td><td>√</td><td>89.46</td><td>60.21</td><td>74.84</td><td></td><td>89.46</td><td>58.13</td><td>73.79</td></tr></table>

Table 3: Comparison of diferent alignment flow modeling strategies. All experiments are conducted with LoRA on the ScienceQA dataset. Attn-� denotes stacking �−1 standard attention layers followed by AFM as the final scoring layer.
<table><tr><td rowspan="2">Method</td><td colspan="3">LoRA Rank = 32</td><td colspan="3">LoRA Rank = 64</td></tr><tr><td>Target</td><td>Source</td><td>Avg</td><td>Target</td><td>Source</td><td>Avg</td></tr><tr><td>Zero-shot</td><td>70.22</td><td>61.09</td><td>65.65</td><td>70.22</td><td>61.09</td><td>65.65</td></tr><tr><td>Scale</td><td>88.92</td><td>59.97</td><td>74.45</td><td>87.97</td><td>57.18</td><td>72.58</td></tr><tr><td>Linear</td><td>88.73</td><td>59.34</td><td>74.03</td><td>87.55</td><td>34.99</td><td>61.27</td></tr><tr><td>Attn-2</td><td>89.39</td><td>59.70</td><td>74.54</td><td>87.34</td><td>52.02</td><td>69.68</td></tr><tr><td>Attn-3</td><td>88.23</td><td>60.12</td><td>74.18</td><td>86.68</td><td>38.61</td><td>62.64</td></tr><tr><td>Ours</td><td>89.46</td><td>60.21</td><td>74.84</td><td>89.46</td><td>58.13</td><td>73.79</td></tr></table>

Ablation ofAlignment Flow Module Design. We conduct a series of experiments on the ScienceQA dataset using LoRA with a rank of 32 to evaluate the design choices of our Alignment Flow Module (AFM). We compare the proposed AFM against three alternative architectures for modeling alignment flow between $h _ { 0 }$ and $h _ { L }$ , i.e., the embeddings before the first layer and after the final layer of the LLM backbone: (1) Scale Module, which applies learnable channel-wise scaling factors to compute relation scores between $h _ { 0 }$ and $h _ { L }$ via a scaled dot product, augmented with a learnable bias term; (2) Linear Module, which constructs pairwise feature concatenations across all position pairs of $h _ { 0 }$ and $h _ { L }$ and maps them to relation scores through a linear layer, combined with residual dotproduct scores; (3) Attention Module (� layers), which stacks �−1 standard attention layers with residual connections to iteratively refine $h _ { L }$ by attending to $h _ { 0 } ,$ followed by our AFM as the final scoring layer. We evaluate $L = 2$ and � = 3. As shown in Table 3, our single-layer AFM achieves the best overall performance under both rank settings. The diferences among methods are relatively small at rank 32 but become substantially larger at rank 64, where the multilayer Attention modules sufer severe source degradation, with source accuracy dropping to as low as 38.61. We attribute this to the over-parameterized projection modules absorbing the teacher’s alignment patterns within their own parameters [4], rather than forcing the student backbone to internalize this knowledge. This weakens the implicit regularization provided by the distillation objective, allowing the LoRA parameters to be dominated by the downstream task loss and resulting in catastrophic forgetting. The problem is amplified at rank 64 due to the increased degrees of freedom. In contrast, our lightweight single-layer AFM serves as an efective conduit for learning alignment knowledge and transferring it to the student, thereby achieving a better balance between task adaptation and alignment retention.

![](images/f878dce77041bdde9e92063308649e6626306b389fd8d480997a5d36f8edec28.jpg)

![](images/4a9ae57cc5340ffe0fbdc52ce024f701b2836fa9c86708dde8798dd03d9559c7.jpg)

![](images/1601e8e18beacf196707c1b8755649370ec1054aa69313996a47704a37a0f98d.jpg)

![](images/977c3dfb1bd757199581453c7fdc853c4ddfdcc8d2d8ec271fd80480b7c2dc17.jpg)  
(a) LoRA, VQA (b) LLaVAKD, VQA (c) LoRA, Cap. (d) LLaVAKD, Cap.  
Figure 5: Hyperparameter analysis of $\lambda _ { F R }$ and $\lambda _ { A P } .$ . All results are obtained with LLaVAFlow applied.

Analysis of Hyperparameters. We conduct a sensitivity analysis of the hyperparameters $\lambda _ { F R }$ and $\lambda _ { A P } ,$ , which regulate the trade-of between the Flow Refinement loss $\mathcal { L } _ { F R }$ and the Alignment Preservation loss $\mathcal { L } _ { A P }$ across diverse settings. Specifically, we evaluate their efects on two tasks, VQA and Captioning, and integrate our method with two representative paradigms: LoRA as a parametereficient fine-tuning approach, and LLaVA-KD as a distillation-based method. As shown in Figure 5, each subfigure presents a heatmap of the average performance over both source and target datasets under diferent hyperparameter configurations. Based on these empirical observations, we select $\lambda _ { F R } = 1 . 5 , \lambda _ { A P } = 0 . 5$ for VQA, and $\lambda _ { F R } = 1 . 5 , \lambda _ { A P } = 2 . 0$ for Captioning.

## 6 Conclusion

In this paper, we present LLaVAFlow, an information-theoretic distillation framework designed to mitigate catastrophic forgetting in MLLMs during visual instruction tuning. We introduce the concept of alignment flow, which captures cross-modal alignment information implicitly encoded in the feature propagation trajectory of the LLM backbone. To extract this alignment flow, we design the Alignment Flow Module (AFM) as a lightweight relational module, and employ an information bottleneck objective that compresses the mutual information between the AFM output and the raw MLLM embeddings, thereby encouraging the extraction of alignment-relevant signals while discarding redundant noise. To transfer the extracted alignment flow, we maximize the mutual information between the alignment flows of the pretrained and fine-tuned models, preserving structural cross-modal knowledge throughout adaptation. Extensive experiments on VQA and Captioning tasks demonstrate that LLaVAFlow functions as a plugand-play framework, achieving strong downstream performance while efectively retaining the general knowledge of the MLLM.

## Acknowledgments

This work was supported in part by the New Generation Artificial Intelligence-National Science and Technology Major Project No. 2025ZD0123003, NSFC under Grant 62576268 and 62302384, and the Key Research and Development Project in Shaanxi Province No. 2024PT-ZCK-89.

## References

[1] Rajendra Bhatia. 2006. Infinitely divisible matrices. The American Mathematical Monthly 113, 3 (2006), 221–235.

[2] Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong Liu, and Xiang Bai. 2025. Llava-kd: A framework of distilling multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 239–249.

[3] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajan than, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. 2019. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486 (2019).

[4] Yudong Chen, Sen Wang, Jiajun Liu, Xuwei Xu, Frank de Hoog, and Zi Huang. 2022. Improved feature distillation via projector ensemble. Advances in Neural Information Processing Systems 35 (2022), 12084–12095.

[5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 24185–24198.

[6] Thomas M Cover. 1999. Elements of information theory. John Wiley & Sons

[7] Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Wei Shen, Limao Xiong, Yuhao Zhou, Xiao Wang, Zhiheng Xi, Xiaoran Fan, et al. 2024. LoRAMoE: Alleviating world knowledge forgetting in large language models via MoE-style plugin. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1932–1945.

[8] Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lijuan Wang, Yezhou Yang, and Zicheng Liu. 2021. Compressing visual-linguistic model via knowledge distillation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 1428–1438.

[9] Qianhan Feng, Wenshuo Li, Tong Lin, and Xinghao Chen. 2025. Align-kd: Distill ing cross-modal alignment knowledge for mobile vision-language large model enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4178–4188.

[10] Jiayi Han, Liang Du, Hongwei Du, Xiangguo Zhou, Yiwen Wu, Yuanfang Zhang, Weibo Zheng, and Donghong Han. 2025. Slim: Let llm learn more and forget less with soft lora and identity mixture. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 4792–4804.

[11] Geofrey Hinton, Oriol Vinyals, and Jef Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015).

[12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. Iclr 1, 2 (2022), 3.

[13] Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Weiming Zhang, and Nenghai Yu. 2025. Deciphering Cross-Modal Alignment in Large Vision-Language Models via Modality Integration Rate. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 218–227

[14] Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6700–6709.

[15] Kenji Kawaguchi, Zhun Deng, Xu Ji, and Jiaoyang Huang. 2023. How does information bottleneck help deep learning?. In International conference on machine learning. PMLR, 16049–16096.

[16] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe national academy ofsciences 114, 13 (2017), 3521– 3526.

[17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. 2024. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024).

[18] Hongyu Li, Liang Ding, Meng Fang, and Dacheng Tao. 2024. Revisiting catastrophic forgetting in large language model tuning. In Findings ofthe association for computational linguistics: EMNLP 2024. 4297–4308.

[19] Jian Liang, Wenke Huang, Guancheng Wan, Qu Yang, and Mang Ye. 2025. Lorasculpt: Sculpting lora for harmonizing general and specialized knowledge in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26170–26180.

[20] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. 2022. Mind the gap: understanding the modality gap in multi-modal contrastive

representation learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems. 17612–17625.

[21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision. Springer, 740–755.

[22] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26296–26306.

[23] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.

[24] Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. Dora: Weight-decomposed low-rank adaptation. In Forty-first International Conference on Machine Learning.

[25] Xialei Liu, Marc Masana, Luis Herranz, Joost Van de Weijer, Antonio M Lopez, and Andrew D Bagdanov. 2018. Rotate your networks: Better weight consolidation and less catastrophic forgetting. arXiv preprint arXiv:1802.02950 (2018).

[26] David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems 30 (2017).

[27] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems 35 (2022), 2507–2521.

[28] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings ofthe IEEE/cvfconference on computer vision and pattern recognition. 3195–3204.

[29] Roy Miles, Adrian Lopez Rodriguez, and Krystian Mikolajczyk. 2021. Information theoretic representation distillation. arXiv preprint arXiv:2112.00459 (2021).

[30] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR). IEEE, 947–952.

[31] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. 2024. Towards interpreting visual information processing in vision-language models. arXiv preprint arXiv:2410.07149 (2024).

[32] Nostalgebraist. 2020. Interpreting GPT: The Logit Lens. https: //www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpretinggpt-the-logit-lens. Accessed: 2026-03-24.

[33] Changdae Oh, Jiatong Li, Shawn Im, and Sharon Li. 2025. Visual instruction bottleneck tuning. arXiv preprint arXiv:2505.13946 (2025).

[34] Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. 2023. Task-specific skill localization in fine-tuned language models. In International Conference on Machine Learning. PMLR, 27011–27033.

[35] Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. 2023. Progressive prompts: Continual learning for language models. arXiv preprint arXiv:2301.12314 (2023).

[36] Ying Shen, Zhiyang Xu, Qifan Wang, Yu Cheng, Wenpeng Yin, and Lifu Huang. 2024. Multimodal instruction tuning with conditional mixture of lora. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 637–648.

[37] Fangxun Shu, Yue Liao, Lei Zhang, Le Zhuo, Chenning Xu, Guanghao Zhang, Haonan Shi, Weilong Dai, ZhongTao, Zhelun Yu, Wanggui He, Siming Fu, Haoyuan Li, Si Liu, Hongsheng Li, and Hao Jiang. 2025. LLaVA-MoD: Making LLaVA tiny via MoE-knowledge distillation. In International Conference on Learning Representations, Vol. 2025. 9386–9404.

[38] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 8317–8326.

[39] Yi-Lin Sung, Jaehong Yoon, and Mohit Bansal. 2023. Ecoflap: Eficient coarse-to-fine layer-wise pruning for vision-language models. arXiv preprint arXiv:2310.02998 (2023).

[40] Naftali Tishby and Noga Zaslavsky. 2015. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw). Ieee, 1–5.

[41] Tiannan Wang, Wangchunshu Zhou, Yan Zeng, and Xinsong Zhang. 2023. Eficientvlm: Fast and accurate vision-language models via knowledge distillation and modal-adaptive pruning. In Findings ofthe Association for Computational Linguistics: ACL 2023. 13899–13913.

[42] Zhicheng Wang, Yufang Liu, Tao Ji, Xiaoling Wang, Yuanbin Wu, Congcong Jiang, Ye Chao, Zhencong Han, Ling Wang, Xu Shao, et al. 2023. Rehearsal-free continual language learning via eficient parameter isolation. In Proceedings of the 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 10933–10946.

[43] Caixia Yan, Muyan Jiao, Nuohan Xue, Weizhan Zhang, Jiahao Wang, Xiaojun Chang, and Feng Tian. 2026. VLDUS: Vision-language distillated unseen synthe sizer for zero-shot object detection. Neural Networks (2026), 108899.

[44] Yibo Yang, Xiaojie Li, Zhongzhu Zhou, Shuaiwen L Song, Jianlong Wu, Liqiang Nie, and Bernard Ghanem. 2024. Corda: Context-oriented decomposition adap tation of large language models for task-aware parameter-eficient fine-tuning. Advances in Neural Information Processing Systems 37 (2024), 71768–71791.

[45] Jason Yosinski, Jef Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? Advances in neural information processing systems 27 (2014).

[46] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are super mario: Absorbing abilities from homologous models as a free lunch. In Forty-first International Conference on Machine Learning

[47] Shujian Yu, Luis Gonzalo Sanchez Giraldo, Robert Jenssen, and Jose C Principe. 2019. Multivariate Extension of Matrix-Based Rényi’s �-Order Entropy Functional. IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 11 (2019), 2960–2966.

[48] Muyao Yuan, Weizhan Zhang, Caixia Yan, Tieliang Gong, Yuanhong Zhang, and Jiangyong Ying. 2024. Adaptive token selection for eficient detection transformer with dual teacher supervision. Knowledge-Based Systems 300 (2024), 112036.

[49] Muyao Yuan, Yuanhong Zhang, Weizhan Zhang, Lan Ma, Yuan Gao, Jiangyong Ying, and Yudeng Xin. 2025. InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer. arXiv preprint arXiv:2511.15967 (2025).

[50] Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. 2024. Investigating the catastrophic forgetting in multimodal large language

model fine-tuning. In Conference on Parsimony and Learning. PMLR, 202–227.

[51] Yuanhong Zhang, Muyao Yuan, Weizhan Zhang, Tieliang Gong, Wen Wen, Jiangyong Ying, and Weijie Shi. 2025. InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective. arXiv preprint arXiv:2505.21920 (2025).

[52] Zaiwei Zhang, Gregory P Meyer, Zhichao Lu, Ashish Shrivastava, Avinash Ravichandran, and Eric M Wolf. 2024. Vlm-kd: Knowledge distillation from vlm for long-tail visual recognition. arXiv preprint arXiv:2408.16930 (2024).

[53] Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. 2025. Cross-modal information flow in multimodal large language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 19781–19791.

[54] Fei Zhao, Taotian Pang, Chunhui Li, Zhen Wu, Junjie Guo, Shangyu Xing, and Xinyu Dai. 2024. Aligngpt: Multi-modal large language models with adaptive alignment capability. arXiv preprint arXiv:2405.14129 (2024).

[55] Zheng Zhao, Yftah Ziser, and Shay B Cohen. 2024. Layer by layer: Uncovering where multi-task learning happens in instruction-tuned large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing. 15195–15214.

[56] Didi Zhu, Zhongyi Sun, Zexi Li, Tao Shen, Ke Yan, Shouhong Ding, Chao Wu, and Kun Kuang. 2024. Model tailor: mitigating catastrophic forgetting in multi-modal large language models. In Proceedings of the 41st International Conference on Machine Learning. 62581–62598.