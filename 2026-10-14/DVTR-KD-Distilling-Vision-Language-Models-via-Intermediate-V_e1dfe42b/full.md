# DVTR-KD: Distilling Vision-Language Models via Intermediate Visual Token Reconstruction

Haolan Wei Jiahui Wang Junli Zha Xinkai Lu Yunxiao Zou Chao Zhou Xiang Song

SF Technology Co., Ltd

Shenzhen, China

weihaolan6@gmail.com, {wangjiahui, zhajunli}@sf-express.com

## Abstract

Multimodal large language models (MLLMs) have achieved strong performance on diverse vision-language tasks, but their computational demands limit deployment under constrained budgets, particularly on edge devices. Many works focus on Knowledge Distillation (KD) and try to transfer knowledge from powerful teacher models to students. In addition to reducing model size, visual token compression provides a complementary route to efficiency, but introduces mismatched visual token counts that complicate direct token-wise knowledge transfer. To mitigate this, we propose DVTR-KD, a knowledge distillation framework based on Dense Visual Token Reconstruction, which reconstructs the teacher’s dense visual tokens from the student’s compressed tokens. We reconstruct visual states after the final LLM layer and at the input to an intermediate layer identified as important for visual-information readout. At the intermediate layer, reconstruction is supervised through the teacher’s response predictions rather than by direct tokenwise KL or MSE matching. An auxiliary attention-readout loss complements this response-level supervision by aligning how the student and the original teacher integrate contextual information when predicting responses. Evaluations on diverse benchmarks demonstrate that DVTR-KD improves the performance oftoken-compressed students while retaining the inference efficiency afforded by visual token compression.

## 1. Introduction

Large language models (LLMs) [2, 6, 13, 15, 38] have achieved strong performance in language understanding, generation, and reasoning. Building upon these advances, recent studies have sought to extend LLMs beyond the text modality by integrating visual perception capabilities, giving rise to Multimodal Large Language Models (MLLMs). By connecting vision encoders with powerful LLMs [3, 19,

20, 23, 24], MLLMs are able to perceive visual content and perform language-based reasoning over multimodal inputs. This paradigm has achieved impressive performance across a wide range of vision-language tasks, including visual question answering [1, 14, 17, 29], image understanding [27], optical character recognition [28, 30, 31, 33], and visual reasoning [39], establishing MLLMs as a promising foundation for general-purpose multimodal intelligence.

However, unlike earlier MLLMs that process images at fixed resolutions [23–25, 40], advanced models increasingly support native-resolution visual inputs to preserve fine-grained visual information [3, 20]. While such designs improve visual perception, especially for high-resolution images in perception tasks, they often produce substantially longer visual token sequences and consequently introduce considerable computational cost. This issue becomes particularly pronounced in the LLM decoder, where the computational cost grows quadratically with the sequence length. To improve the inference efficiency, some researchers focus on reducing the number of visual tokens while preserving the capabilities of MLLMs [9, 35]. These methods reduce visual redundancy, enabling MLLMs to process high-resolution visual inputs with significantly fewer tokens.

Knowledge distillation (KD) provides an effective way for transferring knowledge from a knowledgeable teacher model to a student model, typically by encouraging the student to imitate the teacher’s output distributions or representations. In the context of MLLMs, several studies have explored knowledge transfer between teacher and student MLLMs. LLaVA-KD [4], for example, presents a comprehensive MLLM distillation framework that combines conventional response distillation with visual representation and relation distillation. However, relatively little attention has been paid to distillation between a teacher and a tokencompressed student, where the teacher and student produce visual sequences of substantially different lengths. EM-KD [11] addresses this mismatch using the Hungarian matching algorithm to establish one-to-one correspondences between teacher and student visual tokens. Despite its effectiveness, such one-to-one matching inherently utilizes only a subset of the teacher’s dense visual tokens. Given $N _ { T }$ teacher tokens and $N _ { S } < N _ { T }$ student tokens, one-to-one matching uses at most $N _ { S }$ teacher tokens, leaving at least a fraction $1 - N _ { S } / N _ { T }$ without direct token-level supervision. Consequently, an increasingly large fraction of the teacher’s visual representations remains unmatched as the compression ratio increases, potentially limiting the fine-grained visual knowledge transferred to highly compressed students.

To address this limitation, we propose DVTR-KD, which takes different direction: rather than selecting a subset of teacher tokens to match the compressed student sequence, we reconstruct dense visual representations from the student’s compressed visual tokens and align them with the complete teacher visual sequence. By performing token-wise distillation on the reconstructed representations, DVTR-KD allows supervision from all teacher visual tokens, encouraging the compressed student representations to preserve sufficient information to recover the teacher’s fine-grained visual knowledge.

In addition, while Align-KD [10] explores cross-modal distillation through first-layer attention and visual-feature alignment, the importance of visual information may vary across different depths of the LLM. To investigate this, we conduct a layer-wise visual ablation study on MLLMs built upon Qwen3 [38] and MoonViT [19], trained under different compression ratios and quantities of training samples. Specifically, we zero out the value vectors of visual tokens in the self-attention module of one LLM layer at a time and measure the resulting increase in response NLL. As illustrated in Figure 1, removing visual information at the first layer has a relatively limited impact, whereas perturbations at certain intermediate layers lead to substantially larger degradation. We refer to the layer exhibiting the largest ∆NLL as the Principal Visual Readout Layer (PVRL). We observe that for a given LLM backbone, the PVRL remains same across different training data scales and visual token compression ratios. Meanwhile, stronger models consistently exhibit smaller performance degradation under the same intervention, suggesting different degrees of sensitivity to visual information at this critical layer.

These observations motivate us to perform visual knowledge distillation at the model’s PVRL, where visual information has a particularly strong influence on response predictions. Specifically, DVTR-KD employs a reconstructor to recover the teacher’s dense visual representations from the student’s compressed visual states. The reconstructed representations are then processed by the remaining teacher layers, providing teacher-guided supervision that encourages the student to preserve visual information essential for subsequent language generation. We also align the student’s native attention outputs with those of the original teacher at the PVRL. Together with response distillation and finallayer visual distillation, these objectives supervise both the reconstructed visual states and the student’s own use of multimodal context.

![](images/04185aa86b82bfe92c70ae943b913555c66d26eb9086c174531571a3268c9961.jpg)  
Figure 1. Layer-wise sensitivity to visual-value ablation after SFT and before distillation. We zero the value vectors of all visual tokens in one transformer layer at a time and report the change in response NLL (∆NLL), averaged over the same diagnostic subset. In the legends, 1× denotes uncompressed models, while 4× and 16× denote models with nominal fourfold and sixteenfold reductions in visual-token count. The sample counts are summed over all training datasets before distillation.

Extensive experiments demonstrate that DVTR-KD consistently improves the performance of compressed MLLMs across different model scales and compression ratios.

In conclusion, our contributions are as follows:

• We propose DVTR-KD, which reconstructs dense visual states from compressed student tokens to enable distillation between models with different visual sequence lengths.

• We use layer-wise visual-value interventions to select an intermediate reconstruction layer. We supervise reconstruction at the selected layer using response predictions from the remaining teacher layers.

• We evaluate DVTR-KD on various benchmarks. The experimental results demonstrate that across the evaluated settings, DVTR-KD achieves state-of-the-art (SOTA) performance, without adding modules or computation at inference time.

## 2. Related work

## 2.1. Multimodal Large Language Models

Following the remarkable success of large language models (LLMs) in natural language processing [32, 36], substantial efforts have been devoted to extending their capabilities to visual understanding. Early MLLMs typically connect a pretrained visual encoder with a pretrained LLM through lightweight cross-modal modules. BLIP-2 [23] introduces a Querying Transformer (Q-Former) to bridge frozen image encoders and LLMs, while LLaVA [24] adopts a simpler architecture that projects visual features into the LLM embed ding space and further performs visual instruction tuning. Building upon the LLaVA architecture, subsequent models substantially improve visual resolution and task coverage. LLaVA-NeXT [26] introduces higher-resolution visual inputs to enhance fine-grained perception and OCR capabilities, while LLaVA-OneVision [22] further unifies singleimage, multi-image, and video understanding within a single model. More recent MLLMs support flexible-resolution visual inputs: Qwen3-VL [3] dynamically represents images of different resolutions with different numbers of visual tokens, while Kimi-VL [19] employs native-resolution visual encoding to preserve fine-grained image information. These advances have greatly enhanced the perceptual and reasoning capabilities of MLLMs, but processing highresolution visual inputs also introduces increasingly long visual sequences and substantial computational overhead.

## 2.2. Distillation for MLLMs

Knowledge distillation has been extensively studied for transferring the capabilities of large language models to students. MiniLLM [16] introduces an on-policy method and performs distillation on student-generated sequences instead of ground-truth responses, while DistiLLM [21] develops a skewed-KL objective together with an adaptive offpolicy strategy for more efficient LLM distillation.

Recent studies further extend knowledge distillation to multimodal language models. LLaVA-KD [4] provides a general MLLM distillation framework that combines response distillation with visual representation. Align-KD [10] focuses on cross-modal knowledge transfer by distilling first-layer vision–language matching. More recently, Switch-KD [34] instead feeds student visual representations into the teacher’s language pathway, enabling multimodal supervision within a shared text-probability space. DVTR-KD instead reconstructs the teacher’s intermediate visual states from the student’s visual representations and optimizes this reconstruction using response-level supervision.

Most existing methods assume compatible visual representations between teacher and student and do not explicitly address the unequal visual sequence lengths introduced by token compression. EM-KD [11] directly studies this problem and employs Hungarian matching to establish oneto-one correspondences between teacher and student visual tokens, followed by visual semantic and vision–language affinity distillation. However, when the student is highly compressed, only the matched subset of teacher visual tokens directly participates in token-level supervision. In contrast, DVTR-KD reconstructs a dense visual representation from compressed student tokens and aligns it with the complete teacher sequence, allowing all teacher visual positions to contribute to direct visual supervision.

## 3. Method

## 3.1. Composition of Distilled MLLM

Visual Encoder. Both teacher and student use the same pretrained Vision Transformer (ViT) [8] as their visual backbone. Given an image $X _ { v } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ , each model pads it to $H ^ { \prime } \times W ^ { \prime }$ , with both dimensions chosen to accommodate the patch size $S _ { p }$ and its visual-token compression configuration. The padded image is divided into $N _ { p } = H ^ { \prime } W ^ { \prime } / S _ { p } ^ { 2 }$ non-overlapping patches, from which the ViT extracts spatial visual features.

Visual Projector and Compressor. The visual projector maps visual features into the LLM embedding space. The student reduces the visual-token count through spatial compression within the ViT and/or in the projector. We denote the overall spatial downsampling factor by $\rho ,$ so that $\rho ^ { 2 }$ is the nominal visual-token compression factor. Let $T$ and S denote the teacher and student models, respectively, and let $N _ { v , T }$ and $N _ { v , S }$ denote their visual-token counts at the LLM input, with $N _ { v , S } < N _ { v , T }$

Large Language Model (LLM). The LLM processes visual and prompt embeddings $H = [ H _ { v } , H _ { t } ]$ . We denote the output logits by $Z = [ Z _ { v } , Z _ { t } , Z _ { r } ]$ and the corresponding vocabulary probability distributions by $\boldsymbol { Y } ~ = ~ [ Y _ { v } , Y _ { t } , Y _ { r } ]$ where v, t, and r indicate visual, prompt, and response positions, respectively. For each position i, the probability distribution is computed as

$$
{ \bf y } _ { i } = \frac { \exp ( { \bf z } _ { i } / \tau ) } { \left\| \exp ( { \bf z } _ { i } / \tau ) \right\| _ { 1 } } ,\tag{1}
$$

where $\tau$ is the temperature, K is the vocabulary size, and $\mathbf { z } _ { i } , \mathbf { y } _ { i } \in \mathbb { R } ^ { K }$

## 3.2. Training Scheme

Before PVRL analysis and distillation, we prepare teacher and student MLLMs through pretraining and SFT.

Stage 1: Pretraining. We freeze the pretrained visual encoder and LLM, and train the visual projector together with any learnable student compressor on image-caption pairs. This stage aligns visual representations with the LLM embedding space. For powerful teachers we further introduce an optional stage following LLaVA-OneVision [22], unfreezing the teacher LLM and jointly training it with the projector on detailed image descriptions.

![](images/fa208ff02c2a1febe76c463abfeb4dbc7d0386638675c24f979ee2f825bf1d8b.jpg)  
Figure 2. Overview of DVTR-KD. (a) The intermediate reconstructor $R _ { \mathrm { m i d } }$ reconstructs student visual states in the teacher’s dense layout at PVRL, substituting teacher’s visual states; this reconstruction is supervised through response predictions produced by the teacher from the PVRL onward. Native response distillation and an independent final-layer reconstructor $R _ { \mathrm { l a s t } }$ provide additional supervision. (b) Readout alignment uses attention outputs from the original teacher and the student. (c) An illustration to the reconstructor.

Stage 2: SFT. We fine-tune both teacher and student on multimodal instruction-following data, updating the LLM, projector, and any learnable compressor while keeping the visual encoder frozen. The resulting SFT models are used for PVRL analysis, and each distillation run starts from the corresponding SFT checkpoint.

Supervision objective. All preparation stages minimize the autoregressive cross-entropy over response tokens. For model $M \in \{ T , S \}$ , let $Y _ { r , M }$ denote its predicted response distributions and $Y _ { \mathrm { g t } }$ the corresponding ground-truth distributions, where each $\mathbf { y } _ { \mathrm { g t } , j }$ is a one-hot probability vector.

$$
\mathcal { C } ( Y _ { r , M } , Y _ { \mathrm { g t } } ) = - \frac { 1 } { N _ { r } } \sum _ { j = 1 } ^ { N _ { r } } \left. \mathbf { y } _ { \mathrm { g t } , j } , \log \mathbf { y } _ { r , M , j } \right. .\tag{2}
$$

Here, $N _ { r }$ is the number of response tokens, C represents cross entropy, $\langle \cdot , \cdot \rangle$ denotes the inner product, and the logarithm is applied element-wise. Each predicted distribution

is conditioned on the image, prompt, and preceding groundtruth response tokens.

## 3.3. Principal Visual Readout Layer

We identify the principal visual readout layer (PVRL) by measuring how suppressing visual information at each LLM layer affects response prediction. For a given LLM M with L transformer layers, we intervene on one layer at a time. At layer ℓ, we set the value vectors at visual-token positions to zero, leaving the queries, keys, text-token values, and attention normalization unchanged. This removes the direct contribution of visual values to that layer’s attention output while preserving its residual connection. The remaining LLM computations proceed normally. We denote the resulting model by $M _ { \rho } ^ { V = \bar { 0 } }$   
ℓ

We evaluate the original and intervened models on the same fixed diagnostic subset sampled from the SFT dataset. Let d denote the diagnostic set and $\mathrm { N L L } _ { i } ( M )$ denote the mean negative log-likelihood of the response tokens in example i under model M. The sensitivity score of layer ℓ is

$$
\Delta \mathrm { N L L } _ { \ell } = \frac { 1 } { | d | } \sum _ { i \in d } \left[ \mathrm { N L L } _ { i } ( M _ { \ell } ^ { V = 0 } ) - \mathrm { N L L } _ { i } ( M ) \right] .\tag{3}
$$

Thus, response tokens are averaged within each example. A larger positive $\Delta \mathrm { N L L } _ { \ell }$ means that zeroing visual-token values at layer ℓ causes a greater reduction in the likelihood assigned to ground-truth responses, indicating greater sensitivity to the removal of direct visual-value contributions at that layer. We select the layer with the largest sensitivity score as the PVRL:

$$
\ell ^ { * } = \arg \operatorname* { m a x } _ { \ell \in \{ 1 , \dots , L \} } \Delta \mathrm { N L L } _ { \ell } .\tag{4}
$$

Note that this procedure is applied independently to each model under analysis, with its parameters held fixed.

## 3.4. DVTR-KD

Reconstructor. To bridge the token-count mismatch, we upsample the student’s coarse visual grid to the teacher grid and learn a residual correction. Because teacher and student use different padding, we map teacher-token centers to the student grid using image coordinates. The operator B bilinearly interpolates neighboring student states at these locations, downweighting cells that overlap padding. Given student visual states $H _ { v , S }$ , we compute

$$
\begin{array} { c } { { U = B ( H _ { v , S } ) , } } \\ { { \hat { H } _ { v , R } = R ( H _ { v , S } ) = U + \varphi _ { \theta } ( U + \mathrm { p o s } ) , } } \end{array}\tag{5}
$$

where pos contains 2D positional embeddings. $\hat { H } _ { v , R } \in$ $\mathbb { R } ^ { N _ { v , T } \times D }$ contains $N _ { v , T }$ reconstructed visual states, where D is the LLM hidden dimension. The refinement module $\varphi _ { \theta }$ consists of one bidirectional transformer layer followed by a zero-initialized output projection, so reconstruction initially equals the upsampled states $U$ . We use independent reconstructors $R _ { \mathrm { l a s t } }$ and $R _ { \mathrm { m i d } }$ for final-layer and intermediate-layer supervision, respectively. Implementation details are provided in supplementary materials.

Visual Semantic Alignment. Aligning vocabulary distributions at visual-token positions has proved effective in MLLM distillation, as demonstrated by the visual component of LLaVA-KD [4] and EM-KD [11]. However, the token-wise alignment cannot be directly applied when teacher and student have unequal token counts. EM-KD addresses this mismatch through one-to-one token matching, but directly supervises at most $N _ { v , S }$ teacher positions, leaving the remaining $( N _ { v , T } \mathrm { ~ - ~ } N _ { v , S } )$ tokens without a matched target. We instead reconstruct a dense student representation and apply visual semantic supervision across the teacher grid.

Let $H _ { v , S }$ denote the student visual states after the final LLM layer. We reconstruct $\hat { H } _ { v , S } ~ = ~ R _ { \mathrm { l a s t } } ( H _ { v , S } )$ on the teacher grid, then apply the student LLM head and the softmax in Eq. (1) to obtain $\hat { Y } _ { v , S } ~ = ~ \{ \hat { \mathbf { y } } _ { v , S , j } \} _ { j = 1 } ^ { N _ { v , T } }$ The reconstructed student distributions $\hat { Y } _ { v , S }$ are aligned with the teacher distributions $Y _ { v , T }$ at corresponding grid positions. We denote D as the average of KL divergences scaled by τ denoted in Eq 1 over corresponding token positions in two collections of distributions. The visual semantic alignment loss is

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { v i s } } = \mathcal { D } ( Y _ { v , T } \parallel \hat { Y } _ { v , S } ) } } \\ & { = \frac { 1 } { N _ { v , T } } \sum _ { j = 1 } ^ { N _ { v , T } } \tau ^ { 2 } \mathrm { K L } ( { \bf y } _ { v , T , j } \| \hat { \bf y } _ { v , S , j } ) } \end{array}\tag{6}
$$

Reconstruction at PVRL. Align-KD [10] demonstrates the benefits of distilling cross-modal alignment at the first LLM layer, highlighting the value of supervision beyond final-layer outputs. We further consider where intermediate distillation can be more effective. Our layer-wise diagnostic motivates choosing the PVRL, where suppressing visual values causes the largest increase in response NLL. We therefore reconstruct dense visual states at the input to this layer. Under strong visual-token compression, however, recovering every component of the teacher’s hidden states is challenging and may be unnecessary for answering the question. We instead supervise reconstruction through the teacher’s response predictions rather than using tokenwise MSE or KL loss, encouraging the student to preserve visual information that supports answering.

Let $H _ { M } ^ { \ell ^ { * } }$ denote the states entering the selected layer. We reconstruct the student visual states as $\hat { H } _ { v , R } = R _ { \mathrm { m i d } } ( H _ { v , S } ^ { \ell ^ { \ast } } )$ and substitute them for all visual positions in $H _ { v , T } ^ { \ell ^ { * } }$ , forming $\tilde { H } _ { T } ^ { \ell ^ { * } } = [ \hat { H } _ { v , R } , H _ { t , T } ^ { \ell ^ { * } } , H _ { r , T } ^ { \ell ^ { * } } ]$ . The original teacher prompt states $H _ { t , T } ^ { \ell ^ { * } }$ and response states $H _ { r , T } ^ { \ell ^ { * } }$ are retained. The resulting sequence is processed by the teacher’s subsequent layers:

$$
Y _ { R } = \phi _ { T } ^ { \ell ^ { * } : L } ( \widetilde { H } _ { T } ^ { \ell ^ { * } } ) ,\tag{7}
$$

where $\phi _ { T } ^ { \ell ^ { * } : L }$ contains the remaining teacher layers and LLM head, followed by the normalization in Eq. (1). Let $Y _ { r , R }$ denote the response distributions in $Y _ { R }$ . Using the CE objective defined in Eq. (2), we combine ground-truth supervision with response distillation from the original teacher:

$$
\mathcal { L } _ { \mathrm { D V T R } } = \lambda \mathcal { C } ( Y _ { r , R } , Y _ { \mathrm { g t } } ) + ( 1 - \lambda ) \mathcal { D } ( Y _ { r , T } \| Y _ { r , R } ) .\tag{8}
$$

This intermediate reconstruction is optimized for its contribution to answering, without directly regressing each reconstructed visual state to its teacher counterpart.

Attention-Readout Alignment. As an auxiliary objective, we introduce attention-readout alignment to guide the student in learning what information the teacher retrieves at the PVRL. Specifically, we align the native student’s attention outputs with those of the teacher at response positions, encouraging the student to extract similar information. Let $a _ { r , M } ^ { \ell ^ { * } } \in \mathbb { R } ^ { N _ { r } \times D }$ denote the attention output after output projection at layer $\ell ^ { * }$ for the $N _ { r }$ supervised response positions. The readout loss is defined as

$$
\mathcal { L } _ { \mathrm { r e a d o u t } } = \mathrm { M S E } ( a _ { r , S } ^ { \ell ^ { * } } , a _ { r , T } ^ { \ell ^ { * } } )\tag{9}
$$

Native Response Supervision. We combine ground-truth supervision with response distillation from the original teacher for the native student:

$$
\mathcal { L } _ { \mathrm { r e s p } } = \lambda \mathcal { C } ( Y _ { r , S } , Y _ { \mathrm { g t } } ) + ( 1 - \lambda ) \mathcal { D } ( Y _ { r , T } \| Y _ { r , S } ) .\tag{10}
$$

## 3.5. Complete Training Objective

The complete training objective combines native response supervision with intermediate reconstruction, attentionreadout alignment, and final-layer visual semantic alignment:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r e s p } } + \lambda _ { \mathrm { D } } \mathcal { L } _ { \mathrm { D V T R } } + \lambda _ { \mathrm { r } } \mathcal { L } _ { \mathrm { r e a d o u t } } + \lambda _ { \mathrm { v } } \mathcal { L } _ { \mathrm { v i s } } . } \end{array}\tag{11}
$$

## 4. Experiments

## 4.1. Experimental Settings

Implementation Details. We adopt two variants of Qwen3 [38], Qwen3-0.6B and Qwen3-4B, as the language backbones to construct two teacher-student configurations. Within each configuration, the teacher and student share the same LLM architecture, while differing in their visual token compression strategies. We primarily conduct experiments with Qwen3-0.6B and further extend our evaluation to Qwen3-4B to validate the generality of our method across different model scales. Both teacher and student models are initialized from the same officially released text-only LLM checkpoints, without prior multimodal fine-tuning. For visual feature extraction, we employ a pretrained MoonViT-SO-400M [19] as the frozen visual encoder, which supports native-resolution image processing and produces finegrained visual representations.

For the uncompressed teacher models, we employ a twolayer MLP projector to map visual representations into the LLM embedding space without reducing the number of visual tokens. For compressed models, we investigate two visual-token compression ratios, $\rho ^ { 2 } ~ \in ~ \{ 4 , 1 6 \}$ For the 4× compressed models, we adopt a pixel-unshuffle projector [7], which concatenates each $2 \times 2$ group of spatially adjacent visual tokens along the channel dimension, followed by projection into the LLM embedding space. For the 16× compressed models, inspired by LLaVA-UHD v4 [9], we additionally insert a compressor after the 6-th ViT layer. This intra-ViT module employs local window attention to aggregate neighboring visual information, followed by $2 \times 2$ pixel-unshuffle and feature fusion to reduce the token count by a factor of four. Combined with the final pixelunshuffle projector, this achieves an overall 16× compression ratio. To accommodate different compression ratios, we adopt ratio-dependent image padding strategies. Specifically, the image height and width are padded to multiples of

14 pixels for the teacher, 28 pixels for students with $\rho ^ { 2 } = 4 .$ and 56 pixels for students with $\rho ^ { 2 } = 1 6$

The experiments are conducted on 8× NVIDIA A800 GPUs. The weights of loss functions are $\lambda = 0 . 5 , \lambda _ { D } = 2$ $\lambda _ { r } ~ = ~ 1$ and $\lambda _ { v } ~ = ~ 0 . 5$ . Based on the PVRL analysis in Figure 1, we use $\ell ^ { * } = 1 2$ and $\ell ^ { * } = 2 0$ for Qwen3-0.6B and Qwen3-4B, respectively. More implementation details are shown in supplementary materials.

Training Datasets. As shown in Table 1, we compare distillation from same-data teachers, trained on the same data as the students, and data-rich teachers, trained with additional data. Before distillation, we prepare the models through the following stages. The ViT remains frozen throughout.

In Stage 1 (pretraining), all models use 558K imagecaption pairs [24] to align visual representations with the LLM embedding space. We train the projector and any learnable student compressor while freezing the LLM. Only data-rich teachers undergo Stage 1.5, where we jointly train the LLM and projector on detailed image descriptions from LLaVA-ReCap-558K [26] to improve visual understanding.

In Stage 2 (visual instruction tuning), students and same-data teachers use the LLaVA-NeXT 779K instructiontuning dataset [26]. Data-rich teachers use approximately 3.3M instruction-tuning samples following the LLaVA-OneVision single-image recipe [22]. This stage updates the LLM, projector, and any learnable student compressor.

For knowledge distillation, all methods use the LLaVA-NeXT 779K instruction-tuning dataset and initialize the student from its corresponding Stage-2 SFT checkpoint.

Baseline Methods. To demonstrate the effectiveness of DVTR-KD, we make fair comparisons with several representative methods: MiniLLM [16], LLaVA-KD [4] and EM-KD [11]. Note that LLaVA-KD cannot be directly applied to MLLMs with unbalanced visual tokens. We reimplement it with a reconstructor to align the number of visual tokens between teacher and student MLLMs, which is one of the key contributions in our work.

Benchmarks. We evaluate the effectiveness of our method on 11 benchmarks. They are categorized into three groups based on different areas. (1) General Question Answering: GQA [17], MME [12], RealWorldQA (RWQA) [37]; (2) Comprehensive Knowledge Reasoning: ScienceQA-IMG (SQA) [29], AI2D [18], MMMU [39], MMStar [5]; (3) OCR and Chart Parsing: ChartQA [30], DocVQA [31], TextVQA [33], and OCRBench [28]. To compare the performance, we report the mean score across all 11 benchmarks, dividing MME scores by 28 before averaging.

Table 1. Training data for model preparation and knowledge distillation. “–” indicates a skipped stage; “Frozen” indicates that the teacher is used without parameter updates during KD.
<table><tr><td>Model</td><td>Stage 1</td><td>Stage 1.5</td><td>SFT</td><td>KD</td></tr><tr><td>Student</td><td>LLaVA-PreTrain-558K</td><td>一</td><td>LLaVA-NeXT-779K</td><td>LLaVA-NeXT-779K</td></tr><tr><td>Same-data teacher</td><td>LLaVA-PreTrain-558K</td><td></td><td>LLaVA-NeXT-779K</td><td>Frozen</td></tr><tr><td>Data-rich teacher</td><td>LLaVA-PreTrain-558K</td><td>LLaVA-ReCap-558K</td><td>LLaVA-OneVision-SI-3.3M</td><td>Frozen</td></tr></table>

Table 2. Distillation results across visual-token compression ratios, teacher data regimes and teacher scales. All methods are reimplemented for fair comparison and start from the corresponding SFT model. $\rho ^ { 2 }$ denotes the nominal visual-token compression factor. LLaVA-KD<sup>∗</sup> means we use our reconstruction method to align different visual-token counts after last block. MME scores are divided by 28 when calculating the average scores. Avg is the mean over all 11 benchmarks.
<table><tr><td></td><td></td><td colspan="3">General</td><td colspan="3">Knowledge</td><td colspan="5">OCR</td><td></td></tr><tr><td>Method</td><td> $\rho ^ { 2 }$ </td><td>GQA</td><td>MME</td><td>RWQA</td><td>SQA AI2D</td><td></td><td>MMMU</td><td>MMStar</td><td>ChartQA</td><td>DocVQA TextVQA</td><td></td><td>OCRBench Avg11</td><td></td></tr><tr><td></td><td colspan="10">Qwen3-0.6B teacher trained on 4.4M samples → Qwen3-0.6B compressed students</td><td></td><td></td><td></td></tr><tr><td>Teacher</td><td>1×</td><td>56.6</td><td>1862.9</td><td>61.7</td><td>88.2</td><td>73.6</td><td>37.0</td><td>50.2</td><td>78.8</td><td>87.5</td><td>77.3</td><td>74.3</td><td>68.3</td></tr><tr><td>SFT</td><td>4×</td><td>57.8</td><td>1450.2</td><td>53.6</td><td>59.8</td><td>59.9</td><td>32.1</td><td>38.3</td><td>64.6</td><td>67.7</td><td>63.1</td><td>49.8</td><td>54.4</td></tr><tr><td>MiniLLM</td><td>4×</td><td>58.3</td><td>1616.3</td><td>56.1</td><td>61.3</td><td>60.8</td><td>32.1</td><td>38.6</td><td>64.6</td><td>66.9</td><td>65.1</td><td>52.2</td><td>55.8</td></tr><tr><td>LLaVA-KD*</td><td>4×</td><td>59.7</td><td>1636.3</td><td>58.0</td><td>62.6</td><td>64.9</td><td>32.3</td><td>40.5</td><td>70.6</td><td>74.4</td><td>68.0</td><td>56.3</td><td>58.7</td></tr><tr><td>EM-KD</td><td>4×</td><td>58.3</td><td>1461.6</td><td>55.0</td><td>62.4</td><td>63.1</td><td>30.1</td><td>39.9</td><td>69.0</td><td>73.7</td><td>66.3</td><td>53.8</td><td>56.7</td></tr><tr><td>Ours</td><td>4×</td><td>59.7</td><td>1648.3</td><td>59.2</td><td>65.4</td><td>67.4</td><td>30.6</td><td>40.8</td><td>73.0</td><td>77.7</td><td>70.0</td><td>56.7</td><td>59.9</td></tr><tr><td>SFT</td><td>16×</td><td>56.0</td><td>1408.3</td><td>53.7</td><td>61.6</td><td>60.7</td><td>32.9</td><td>38.1</td><td>52.2</td><td>45.4</td><td>51.0</td><td>32.1</td><td>48.5</td></tr><tr><td>MiniLLM</td><td>16×</td><td>56.6</td><td>1528.7</td><td>56.7</td><td>62.0</td><td>61.0</td><td>35.6</td><td>37.7</td><td>53.4</td><td>46.1</td><td>52.9</td><td>35.0</td><td>50.1</td></tr><tr><td>LLaVA-KD*</td><td>16×</td><td>57.9</td><td>1347.2</td><td>54.6</td><td>62.4</td><td>65.3</td><td>33.0</td><td>37.5</td><td>58.6</td><td>51.5</td><td>53.9</td><td>34.0</td><td>50.6</td></tr><tr><td>EM-KD</td><td>16×</td><td>56.7</td><td>1379.8</td><td>52.4</td><td>60.4</td><td>62.8</td><td>32.4</td><td>38.6</td><td>59.4</td><td>51.1</td><td>53.4</td><td>33.1</td><td>50.0</td></tr><tr><td>Ours</td><td>16×</td><td>58.0</td><td>1549.1</td><td>56.5</td><td>64.9</td><td>65.3</td><td>33.8</td><td>39.7</td><td>62.6</td><td>55.2</td><td>56.8</td><td>35.1</td><td>53.0</td></tr><tr><td>Qwen3-0.6B teacher trained on 1.3M samples → Qwen3-0.6B compressed students</td><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td>Teacher</td><td>1x</td><td>58.9</td><td>1567.2</td><td>56.6</td><td>62.3</td><td>64.9</td><td>32.8</td><td>40.7</td><td>71.6</td><td>74.8</td><td>67.9</td><td>59.7</td><td>58.7</td></tr><tr><td>SFT</td><td>4×</td><td>57.8</td><td>1450.2</td><td>53.6</td><td>59.8</td><td>59.9</td><td>32.1</td><td>38.3</td><td>64.6</td><td>67.7</td><td>63.1</td><td>49.8</td><td>54.4</td></tr><tr><td>MiniLLM</td><td>4×</td><td>58.2</td><td>1458.2</td><td>53.6</td><td>61.1</td><td>60.3</td><td>31.8</td><td>37.4</td><td>64.0</td><td>67.6</td><td>64.8</td><td>52.1</td><td>54.8</td></tr><tr><td>LLaVA-KD*</td><td>4×</td><td>59.0</td><td>1553.6</td><td>53.9</td><td>61.1</td><td>63.5</td><td>30.7</td><td>39.5</td><td>69.0</td><td>71.8</td><td>65.8</td><td>53.4</td><td>56.6</td></tr><tr><td>EM-KD</td><td>4×</td><td>58.4</td><td>1451.8</td><td>53.9</td><td>59.6</td><td>62.1</td><td>30.1</td><td>39.3</td><td>68.2</td><td>71.4</td><td>64.4</td><td>54.5</td><td>55.8</td></tr><tr><td>Ours</td><td>4×</td><td>59.3</td><td>1523.7</td><td>56.3</td><td>61.6</td><td>64.1</td><td>30.6</td><td>39.6</td><td>70.8</td><td>73.5</td><td>66.9</td><td>55.3</td><td>57.5</td></tr><tr><td>Qwen3-4B teacher trained on 4.4M samples → Qwen3-4B compressed students</td><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td>Teacher</td><td>1×</td><td>61.4</td><td>2163.3</td><td>67.3</td><td>96.8</td><td>82.2</td><td>48.8</td><td>63.9</td><td>84.2</td><td>93.0</td><td>81.4</td><td>78.7</td><td>75.9</td></tr><tr><td>SFT</td><td>4×</td><td>61.3</td><td>1896.4</td><td>61.4</td><td>78.1</td><td>76.0</td><td>44.0</td><td>49.3</td><td>76.4</td><td>80.9</td><td>71.2</td><td>63.1</td><td>66.3</td></tr><tr><td>MiniLLM</td><td>4×</td><td>61.5</td><td>1961.3</td><td>61.4</td><td>78.9</td><td>76.8</td><td>43.2</td><td>47.9</td><td>72.9</td><td>81.2</td><td>74.4</td><td>63.2</td><td>66.5</td></tr><tr><td>LLaVA-KD*</td><td>4×</td><td>62.5</td><td>1880.2</td><td>64.1</td><td>78.3</td><td>76.8</td><td>42.4</td><td>48.9</td><td>78.3</td><td>84.5</td><td>75.4</td><td>68.1</td><td>67.9</td></tr><tr><td>EM-KD</td><td>4×</td><td>61.8</td><td>1954.0</td><td>63.0</td><td>79.3</td><td>76.5</td><td>41.7</td><td>47.6</td><td>77.1</td><td>83.0</td><td>73.8</td><td>65.4</td><td>67.2</td></tr><tr><td>Ours</td><td>4×</td><td>62.8</td><td>2008.6</td><td>63.7</td><td>80.9</td><td>77.8</td><td>44.0</td><td>49.9</td><td>80.9</td><td>87.4</td><td>77.5</td><td>67.9</td><td>69.5</td></tr></table>

## 4.2. Main Results

As shown in Table 2, DVTR-KD consistently outperforms competing distillation methods across different experimental settings, achieving the highest average scores.

With a same-data teacher, DVTR-KD reduces the average-score gap between the 4× SFT student and its uncompressed teacher from 4.3 to 1.2 points. Using the datarich teacher further improves the student from 57.5 to 59.9.

Specifically, comparing the same-data teacher and compressed students without distillation, we observe that aggressive visual token compression has a relatively limited impact on General and Knowledge tasks, but substantially degrades OCR performance. Specifically, the average non-OCR scores are 53.2, 50.5, and 50.5 for the uncompressed same-data teacher and students with compression ratios $\rho ^ { 2 } = 4 ~ \mathrm { a n d } ~ \rho ^ { 2 } = 1 6$ , respectively. In contrast, the corresponding mean OCR scores drop sharply from 68.5 to 61.3 and 45.2, highlighting the vulnerability of fine-grained visual perception to aggressive token compression. DVTR-KD effectively alleviates this degradation, achieving consistent improvements across OCR benchmarks. Compared with the strongest competing baseline, DVTR-KD improves $A v g _ { 1 1 }$ by 0.9–1.6 points under $\rho ^ { 2 } = 4$ and 2.4 points under $\rho ^ { 2 } = 1 6$ , demonstrating its effectiveness in transferring fine-grained visual knowledge to compressed MLLMs.

Table 3. Ablation study on contributions from key components in our method.
<table><tr><td> $\mathcal { L } _ { \mathrm { r e s p } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { v i s } }$ </td><td>LDVTR</td><td> $\mathcal { L } _ { \mathrm { r e a d o u t } }$ </td><td> $A v g _ { 1 1 }$ </td></tr><tr><td>√</td><td>一</td><td>一</td><td>一</td><td>51.0</td></tr><tr><td>√</td><td>√</td><td>一</td><td>一</td><td>51.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>一</td><td>52.6</td></tr><tr><td> $\checkmark$ </td><td>√</td><td> $\checkmark$ </td><td>√</td><td>53.0</td></tr></table>

## 4.3. Ablation Study

We conduct ablation studies to evaluate the effectiveness of each component in DVTR-KD. Unless otherwise specified, experiments are conducted on 0.6B students with $\rho ^ { 2 } = 1 6$ and a data-rich teacher.

Key Components. We progressively introduce ${ \mathcal { L } } _ { \mathrm { v i s } } ,$ $\mathcal { L } _ { \mathrm { D V T R } }$ , and $\mathcal { L } _ { \mathrm { r e a d o u t } }$ to the response distillation baseline ${ \mathcal { L } } _ { \mathrm { r e s p } } .$ . As shown in Table 3, each component contributes to performance improvements, increasing $A v g _ { 1 1 }$ from 51.0 to 51.2, 52.6, and 53.0, respectively. The full objective achieves the best performance, demonstrating the effectiveness of combining these components.

Supervision for Reconstruction at PVRL. To optimize $\mathcal { L } _ { \mathrm { D V T R } }$ , we replace the teacher’s visual states at the PVRL with student-reconstructed representations and apply KL divergence and cross-entropy for response-level supervision. We compare this approach with direct feature alignment using MSE loss. As shown in Table 4, compared with response-level supervision, MSE slightly increases Avg<sub>11</sub> by 0.3 points under $\rho ^ { 2 } = 4$ but significantly decreases it by 1.1 points under $\rho ^ { 2 } = 1 6$ . These results suggest that strict feature matching becomes less effective under aggressive compression, while response-guided supervision is more robust, performing well on both compression ratios.

Reconstruction at Different Layers. Align-KD [10] introduces first-layer cross-modal attention distillation, demonstrating the potential of supervision beyond finallayer outputs. Although it cannot be directly applied to mismatched visual sequences, we investigate reconstruction at the first layer (ℓ = 1) as well as two neighboring layers $( \ell = 1 1 , 1 3 )$ of the PVRL $( \ell ^ { * } \ = \ 1 2 )$ . As shown in Table 5, reconstruction at the PVRL achieves the highest overall score, supporting the effectiveness of our layer selection strategy.

Table 4. Ablation study on supervision type for reconstruction at PVRL.
<table><tr><td> $\rho ^ { 2 }$ </td><td>type</td><td> $A v g _ { 1 1 }$ </td></tr><tr><td>4×</td><td>MSE</td><td>60.2</td></tr><tr><td>4×</td><td>response</td><td>59.9</td></tr><tr><td>16×</td><td>MSE</td><td>51.9</td></tr><tr><td>16×</td><td>response</td><td>53.0</td></tr></table>

Table 5. Reconstruction at different layers. Avg , Avg , and Avg<sub>O</sub> denote the mean scores on the General, Knowledge, and OCR benchmarks, respectively.
<table><tr><td>l</td><td> $A v g _ { G }$ </td><td> $A v g _ { K }$ </td><td> $A v g o$ </td><td> $A v g _ { 1 1 }$ </td></tr><tr><td>1</td><td>55.5</td><td>49.7</td><td>51.6</td><td>52.0</td></tr><tr><td>11</td><td>56.6</td><td>50.8</td><td>51.6</td><td>52.7</td></tr><tr><td>13</td><td>55.9</td><td>50.5</td><td>52.3</td><td>52.6</td></tr><tr><td>12 (l*)</td><td>56.6</td><td>51.0</td><td>52.4</td><td>53.0</td></tr></table>

## 5. Conclusion

This paper proposes DVTR-KD, a novel knowledge distillation framework for transferring visual knowledge from a teacher MLLM to a token-compressed student. To address the mismatch in visual sequence lengths, DVTR-KD reconstructs dense visual representations from compressed student tokens, enabling token-wise alignment with the complete teacher visual sequence. We further introduce intermediate-layer distillation at PVRL, identified as the layer most sensitive to visual information removal. Specifically, we replace the teacher’s visual states at the input to PVRL with student-reconstructed representations and supervise them through the remaining teacher layers using response-level objectives. Comprehensive experiments demonstrate the effectiveness of DVTR-KD.

## References

[1] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings ofthe IEEE international conference on computer vision, pages 2425– 2433, 2015. 1

[2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,

Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 3

[4] Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong Liu, and Xiang Bai. LLaVA-KD: A Framework of Distilling Multimodal Large Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 239–249, 2025. 1, 3, 5, 6, S2

[5] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are We on the Right Way for Evaluating Large Vision-Language Models? In Advances in Neural Information Processing Systems, pages 27056– 27087. Curran Associates, Inc., 2024. 6

[6] DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026. 1

[7] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. In Advances in Neural Information Processing Systems, pages 42566–42592, 2024. 6

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 3

[9] Kechen Fang, Yihua Qin, Chongyi Wang, Wenshuo Ma, Tianyu Yu, and Yuan Yao. LLaVA-UHD v4: What Makes Efficient Visual Encoding in MLLMs? arXiv:2605.08985, 2026. 1, 6

[10] Qianhan Feng, Wenshuo Li, Tong Lin, and Xinghao Chen. Align-KD: Distilling Cross-Modal Alignment Knowledge for Mobile Vision-Language Large Model Enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4178–4188, 2025. 2, 3, 5, 8

[11] Ze Feng, Sen Yang, Boqiang Duan, Wankou Yang, and Jingdong Wang. EM-KD: Distilling Efficient Multimodal Large Language Model with Unbalanced Vision Tokens. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 21111–21119. Association for the Advancement of Artificial Intelligence (AAAI), 2026. 1, 3, 5, 6, S3

[12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. In Advances in Neu-

ral Information Processing Systems. Curran Associates, Inc., 2025. 6

[13] GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengx iao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering, 2026. 1

[14] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 1

[15] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 1

[16] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge Distillation of Large Language Mod els. In International Conference on Learning Representa tions, pages 32694–32717, 2024. 3, 6

[17] Drew A. Hudson and Christopher D. Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6700–6709, 2019. 1, 6

[18] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A Diagram Is Worth a Dozen Images. In Computer Vision – ECCV 2016, pages 235–251. Springer, 2016. 6

[19] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-VL Technical Report. arXiv:2504.07491, 2025. 1, 2, 3, 6

[20] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026. 1

[21] Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards streamlined distillation for large language models. In Proceedings of the 41st International Conference on Machine Learning, pages 24872– 24895. PMLR, 2024. 3

[22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. Transactions on Machine Learning Research, 2025. 3, 4, 6

[23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PmLR, 2023. 1, 3

[24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In Advances in Neural Informa tion Processing Systems, pages 34892–34916. Curran Associates, Inc., 2023. 1, 3, 6

[25] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26296–26306, 2024. 1

[26] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3, 6, S2

[27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 1

[28] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models. arXiv:2305.07895, 2024. 1, 6

[29] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In Advances in Neural Information Processing Systems, pages 2507–2521. Curran Associates, Inc., 2022. 1, 6

[30] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279. Association for Computational Linguistics, 2022. 1, 6

[31] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A Dataset for VQA on Document Images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2200–2209, 2021. 1, 6

[32] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 3

[33] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8317–8326, 2019. 1, 6

[34] Haoyi Sun, Xiaoxiao Wang, Ning Mao, Qian Wang, Lifu Mu, Wen Zheng, Tao Wei, and Wei Chen. Switch-KD: Visual-Switch Knowledge Distillation for Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 9316–9326, 2026. 3

[35] Shichu Sun, Yichen Zhang, Haolin Song, Zonghao Guo, Chi Chen, Yidan Zhang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. LLaVA-UHD v3: Progressive Visual Compression for Efficient Native-Resolution Encoding in MLLMs, 2025. 1

[36] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste´

Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aure-\` lien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 3

[37] xAI. Grok-1.5 vision preview. https://x.ai/news/ grok-1.5v, 2024. 6

[38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 Technical Report. arXiv:2505.09388, 2025. 1, 2, 6

[39] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2024. 1, 6

[40] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. In The Twelfth International Conference on Learning Represen tations, pages 18378–18394, 2024. 1

# DVTR-KD: Distilling Vision-Language Models via Intermediate Visual Token Reconstruction

Supplementary Material

## A. Geometry-aware reconstruction

We detail the reconstructor. Following the main text, $H _ { v , S }$ denotes the student visual states and $\hat { H } _ { v , R }$ their dense reconstruction; the layer index is omitted for brevity.

Image Geometry. Teacher and student use different token grids and add different amounts of padding to the bottom and right of the image. For example, in our experiments, the patch size of ViT’s processor is 14 × 14, hence the teacher’s image height and width are padded to multiples of 14 pixels, whereas a student with $\bar { \rho } ^ { 2 } = 1 6$ requires multiples of 56 pixels. A 524 × 524 image is therefore padded to 532 × 532 for the teacher and 560 × 560 for the student, yielding $3 8 \times 3 8 = 1 { , } 4 4 4$ and $1 0 \times 1 0 = 1 0 0$ visual tokens, respectively. The actual token-count ratio is thus $N _ { v , T } / N _ { v , S } = 1 4 . 4 4$ , rather than exactly 16, because padding rounds up each dimension separately. To align their grids, we locate each teacher token center in image coordinates and express it in student-grid coordinates. This uses each model’s token-cell size rather than scaling the two full padded grids to match.

Bilinear Upsampling. At each mapped location, we interpolate up to four neighboring student visual states. For each neighboring student token, we calculate the fraction of its area covered by the image itself, excluding padding. We multiply its bilinear weight by this fraction, then normalize the weights. This gives one upsampled state per teacher token and produces $U ~ = ~ B ( H _ { v , S } )$ In implementation, we bilinearly sample the features multiplied by these fractions and the fractions themselves, then divide the two results. The denominator is clamped below at $1 0 ^ { - 6 }$ . Both sampling operations use zero padding and align corners=False.

Algorithm 1 explains the same computation as a pertoken loop; the implementation is vectorized. The argument image denotes the image itself, excluding padding. The helper cell size divides each model’s padded image dimensions by its token-grid dimensions. Token indices start at zero, and arithmetic on two-dimensional coordinates is component-wise. The loops omit points outside the student grid, equivalent to zero-padded sampling before renormalization.

Residual Reconstruction. We add continuous twodimensional sine/cosine positional embeddings pos, computed from teacher-cell centers normalized by the teacher’s padded image dimensions, and refine the upsampled states with a single bidirectional Transformer layer:

```prolog
Algorithm 1 Geometry-aware upsampling (Python-like)
1 def coverage(cell, image):
2 # cell: the area represented by one
token.
3 # image: the original image , no
padding.
4 # Return how much of the cell is
original image:
5 # 1 = all image; values between 0 and 1
6 # mean part of the cell is padding.
7 return area(cell ∩ image) / area(cell)
8
9 def upsample(S, T, image):
10 # S: student grid with visual features.
11 # T: teacher grid; only geometry is
used.
12 # Each grid stores its own padded
sizes.
13 # image: the original image, no
padding.
14 # Return U: same length as teacher
tokens.
15 for t in T:
16 # Teacher token center in image
pixels.
17 p = (index(t) + 0.5) <sub>*</sub> cell_size(T)
18
19 # Express this point in the
student grid.
20 # -0.5 makes integers token
centers.
21 q = p / cell_size(S) - 0.5
22
23 # Four grid corners surrounding q;
24 # skip corners outside the grid.
25 value, norm = 0, 0
26 for s in neighbors(S, q):
27 # Multiply the horizontal and
vertical interpolation weights.
28 w = product(1 - abs(q -
index(s)))
29 # Reduce weights for partly
padded cells.
30 c = coverage(cell(s), image)
31 w <sub>*</sub>= c
32 value += w <sub>*</sub> S[s]
33 norm += w
34
35 # Normalize the interpolation
weights.
36 U[t] = value / max(norm, 1e-6)
37 return U
```

$$
\begin{array} { r l } & { X _ { 0 } = U + \mathrm { p o s } , } \\ & { X _ { 1 } = X _ { 0 } + \mathrm { M H S A } ( \mathrm { R M S N o r m } ( X _ { 0 } ) ) , } \\ & { X _ { 2 } = X _ { 1 } + \mathrm { S w i G L U } ( \mathrm { R M S N o r m } ( X _ { 1 } ) ) , } \\ & { \hat { H } _ { v , R } = U + W _ { \mathrm { o u t } } \mathrm { R M S N o r m } ( X _ { 2 } ) . } \end{array}\tag{A.1}
$$

The normalization modules have separate parameters. The hidden width, head count, and feed-forward width follow the student LLM. Only $W _ { \mathrm { o u t } }$ is initialized to zero, making the initial reconstruction equal to U while retaining a direct gradient path to the student. $R _ { \mathrm { m i d } }$ and $R _ { \mathrm { l a s t } }$ use this architecture with independent parameters. The former receives intermediate visual states, while the latter applies the same operations to student visual states after the final RMSNorm.

## B. Experiment Details

PVRL Diagnosis. We use 1,024 unique images drawn from twelve traceable LLaVA-NeXT [26] sources, with 85- 86 images per source. Each sample is under a maximum text length of 2,048. All models share the manifest and retain their native preprocessing.

At each block independently, the diagnostic zeros the value projections of all visual tokens, while preserving queries, keys, the attention denominator, and residual states. We compute the NLL difference between the original and intervened models on the same response tokens, averaging first over tokens within each sample and then equally over samples.

Training Hyperparameters. The training hyperparameters are summarized in Table 6. During Stage-1, we freeze the ViT and LLM, optimizing the projector with a learning rate of $1 0 ^ { - 3 }$ and the additional compressor for 16× students with $1 0 ^ { - 4 }$ . Stage-1.5, Stage-2 and knowledge distillation (KD) share a learning rate of $2 \times 1 0 ^ { - 5 }$ . We use $\tau = 2$ in computing KL divergence and $\tau = 1$ for cross entropy. All stages are trained for one epoch using AdamW with cosine learning rate decay and a 3% warmup ratio.

Evaluation Protocol. Table 7 specifies the evaluation splits, example counts, and metrics for benchmarks. All eleven benchmarks use at most 512 generated tokens. The 43 multi-image questions in MMMU are converted to labeled image grids consistently across models.

## C. Effect of Relation-Based Distillation

Beyond token-wise visual semantic distillation, previous works also explore relation-based supervision to transfer structural knowledge between visual tokens. Specifically, LLaVA-KD [4] introduces Relation Distillation (RDist) to preserve the teacher’s ability to model spatial and semantic dependencies among visual tokens. It constructs visualtoken self-correlation matrices $R _ { v } ^ { T }$ and $R _ { v } ^ { S }$ from the finallayer outputs of the teacher and student, respectively. The relation loss is defined as $\begin{array} { r } { \begin{array} { l } { \mathcal { L } _ { \mathrm { r e l } } } \end{array} = \begin{array} { r } { { 1 } \ - \ \mathrm { \bar { C } o s } ( R _ { v } ^ { \dot { S } } , R _ { v } ^ { T } ) } \end{array} } \end{array}$ encouraging the student to reproduce the teacher’s pairwise visual-token relationships. Similarly, EM-KD [11] introduces Vision-Language Affinity Distillation (VLAD), which aligns the correlations between visual and textual tokens to transfer cross-modal knowledge.

Table 6. Training hyperparameters across different stages.
<table><tr><td>Hyperparameter</td><td>Stage-1 Stage-1.5</td><td>Stage-2</td><td>KD</td></tr><tr><td>Train ViT Backbone</td><td>一</td><td>一</td><td></td></tr><tr><td>Train Projector</td><td>√</td><td>√</td><td>√ √</td></tr><tr><td>Train Compressor</td><td>√</td><td>一</td><td>√ √</td></tr><tr><td>Train LLM</td><td>一</td><td>√ √</td><td>√</td></tr><tr><td>Train Reconstructor</td><td></td><td>一 一</td><td>√</td></tr><tr><td>Projector LR</td><td> $1 0 ^ { - 3 }$ </td><td> $2 \times 1 0 ^ { - 5 }$   $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Compressor LR</td><td> $1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Global Batch Size</td><td>32</td><td>128 128</td><td>128</td></tr><tr><td>Epochs</td><td></td><td>1</td><td></td></tr><tr><td>Optimizer</td><td></td><td>AdamW</td><td></td></tr><tr><td>Weight Decay</td><td></td><td>0</td><td></td></tr><tr><td>LR Scheduler</td><td></td><td>Cosine decay</td><td></td></tr><tr><td>Warmup Ratio</td><td></td><td>0.03</td><td></td></tr><tr><td>DeepSpeed</td><td></td><td>ZeRO-1</td><td></td></tr></table>

Table 7. Evaluation splits and metrics. Counts denote evaluated questions.
<table><tr><td>Benchmark Split</td><td></td><td>Count</td><td>Metric</td></tr><tr><td>GQA</td><td>Test-dev balanced</td><td>12,578</td><td>Exact match</td></tr><tr><td>MME</td><td>Test</td><td>2,374</td><td> ${ \mathrm { A c c . + A c c . } } ^ { + }$ </td></tr><tr><td>RWQA</td><td>Test</td><td>765</td><td>Accuracy</td></tr><tr><td>SQA</td><td>Test (image subset)</td><td></td><td>2,017 Accuracy</td></tr><tr><td>AI2D</td><td>Test</td><td></td><td>3,088 Accuracy</td></tr><tr><td>MMMU</td><td>Validation</td><td></td><td>900 Accuracy</td></tr><tr><td>MMStar</td><td>Validation</td><td></td><td>1,500 Accuracy</td></tr><tr><td>ChartQA</td><td>Test</td><td></td><td>2,500 Relaxed accuracy</td></tr><tr><td>DocVQA</td><td>Validation</td><td>5,349 ANLS</td><td></td></tr><tr><td>TextVQA</td><td>Validation</td><td></td><td>5,000 VQA accuracy</td></tr><tr><td>OCRBench test</td><td></td><td></td><td>1,000 Accuracy</td></tr></table>

Table 8. Effect of relation distillation in our adapted LLaVA-KD baseline under $\rho ^ { 2 } = 1 6$
<table><tr><td> $\mathcal { L } _ { \mathrm { r e s p } }$ </td><td> ${ \mathcal { L } } _ { \mathrm { v i s } }$ </td><td> $\mathcal { L } _ { \mathrm { r e l } }$ </td><td> $A v g _ { 1 1 }$ </td></tr><tr><td>√</td><td>一</td><td>一</td><td>51.0</td></tr><tr><td>√</td><td>√</td><td>一</td><td>51.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>50.6</td></tr></table>

![](images/a95fbdaf5d128d231792717fb4fe0f4ed7ecc53eedf3552bd94db7aab185d073.jpg)  
Figure 3. Layer-wise sensitivity to visual-value ablation after distillation. We compare DVTR-KD students with the data-rich teacher and the corresponding SFT students.

To investigate the effectiveness of relation distillation under aggressive token compression, we reproduce LLaVA-KD in our setting by reconstructing dense visual tokens from the student’s final-layer outputs, enabling both visual semantic and relation distillation. As shown in Table 8, adding ${ \mathcal { L } } _ { \mathrm { v i s } }$ to response supervision improves $\mathrm { A v g } _ { 1 1 }$ from 51.0 to 51.2. However, further introducing $\mathcal { L } _ { \mathrm { r e l } }$ reduces the score to 50.6 under $\rho ^ { 2 } = 1 6$ , even below the response-only baseline.

One explanation is that aggressive compression makes it difficult to preserve fine-grained relationships among visual tokens. Although reconstruction restores spatial correspondence with the teacher, it does not necessarily recover the original relational structure. Enforcing such relationships may therefore introduce additional constraints that interfere with response prediction. Based on these observations, we retain visual semantic distillation but omit relation-based losses from our proposed DVTR-KD objective.

## D. PVRL Diagnosis After Distillation

We further examine the layer-wise ∆NLL of students distilled with DVTR-KD, using the same images and visualvalue interventions. As shown in Figure 3, distillation substantially reduces the increase in response NLL at the PVRL for both compression ratios, bringing it closer to that of the