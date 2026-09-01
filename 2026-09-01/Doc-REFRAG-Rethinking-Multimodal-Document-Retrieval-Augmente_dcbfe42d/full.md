# Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation

Ruofan Hu\*, Shengyang Xu\*, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Ke Lei, Tao Jin, Zhou Zhao<sup>†</sup> Zhejiang University ruofanhu@zju.edu.cn

## Abstract

Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering. However, existing multimodal RAG models are primarily designed for single-image or closed-document settings and exhibit limited accuracy in realistic multi-image scenarios. Moreover, processing numerous retrieved images incurs substantial computational overhead from irrelevant visual tokens. To address these challenges, we introduce DocLongRAG, a large-scale dataset of 343K question– answer pairs, each associated with an average of 37.4 retrieved images to reflect authentic RAG workflows. Building on this dataset, we propose Doc-REFRAG, a question-guided framework that compresses visual tokens into coarse chunks and selectively expands questionrelevant ones via a lightweight RL-based selector. Experiments on six benchmarks show that Doc-REFRAG outperforms eleven strong baselines, achieving state-of-the-art accuracy with significantly lower inference latency. Our resources are available at https://github. com/Collab-Gen/Doc-REFRAG.

## 1 Introduction

Multimodal large language models (MLLMs) have rapidly developed, achieving remarkable performance in various visual understanding tasks (Lu et al., 2024; Bai et al., 2025; Chen et al., 2024b; Yao et al., 2024; Li et al., 2024a). Despite these advances, current MLLMs remain limited by static parametric knowledge. To address this limitation, RAG enhances MLLMs with retrieved evidence from external corpora, thereby expanding factual coverage and mitigating hallucination (Arslan et al., 2024). Crucially, a significant portion of real-world knowledge resides in visually rich documents, which often contain textual passages, charts, and figures. However, existing MLLMs trained on individual images struggle with multi-image reasoning (Ma et al., 2024; Dong et al., 2025a; Tito et al., 2023). Moreover, substantial visual tokens from retrieved images impose significant computational overhead (Ye et al., 2023; Liu et al., 2024a).

![](images/1b48808661b0bc982d8ba899b54e4d9743d1e1625d3a6090a4110b7d3e6c4aa3.jpg)  
Figure 1: Doc-REFRAG achieves the best accuracy– efficiency trade-off among MLLMs.

To handle multi-image inputs, recent works construct long-document training corpora (Van Landeghem et al., 2023; Hu et al., 2024a; Tanaka et al., 2025; Duan et al., 2025) and apply curriculum learning to extend capacity from single to multiimage layouts (Hu et al., 2024b; Duan et al., 2025). However, these works generally assume a closeddomain setting where all images belong to a single, coherent document. RAG, however, retrieves images across diverse documents that are heterogeneous, redundant, and contradictory. As shown in (Dong et al., 2025b; Cuconasu et al., 2024), this distribution shift severely degrades MLLM generalization to the fragmented and noisy images encountered during RAG inference.

Orthogonal to the reasoning challenge, existing compression methods fall into two types. Trainingfree approaches typically rely on either token importance (Chen et al., 2024a; He et al., 2024) or redundancy estimation (Tan et al., 2025; Wang et al., 2025) to prune visual tokens. In contrast, trainingaware methods introduce a linear projector to refine the limited coarse tokens with fine-grained visual details (Li et al., 2025; Hu et al., 2024b). Both categories prioritize visually complex patches, assuming such regions are more informative for downstream tasks. This assumption fails in RAG scenarios, where visually complex patches are often question-irrelevant. Consequently, processing multiple retrieved images still incurs numerous redundant visual tokens and substantial prefill latency (Li et al., 2024b).

To address these problems, we introduce Doc-LongRAG, a new multimodal document dataset. Compared to existing work (Hu et al., 2024a; Tanaka et al., 2025; Duan et al., 2025; Luo et al., 2025), this dataset has the following features. (1) High-Quality Retrieval Contexts: semantically similar images are explicitly interleaved with hard negative distractors to simulate real-world RAG inputs. (2) Long-Range Reasoning: each question is associated with an average of 37.4 images, enabling training on substantially longer sequences. (3) Broad Format Coverage: 343,474 question– answer pairs grounded in 3.3M document images spanning PDFs, slides, and posters, supporting robust generalization across document formats.

Building on DocLongRAG, we propose Doc-REFRAG, a framework for efficient multimodal RAG decoding. Unlike pruning-based methods, Doc-REFRAG adopts a “compress-and-expand” strategy: visual tokens are first compressed into non-overlapping chunks. A lightweight questionconditioned selector then re-expands only the most relevant chunks into full-resolution tokens, recovering fine-grained semantics precisely where needed without the context loss of static pruning.

Integrating mixed sequences of chunked and original tokens requires a specialized training strategy. Doc-REFRAG adopts a three-stage framework: single-image reconstruction for chunk semantic fidelity, multi-image continual pretraining for mixed-sequence processing, and RL-based selector training using accuracy as reward. As shown in Fig. 1, this cuts response time while improving accuracy over prior work.

The contributions are summarized as follows:

• We introduce DocLongRAG, a large-scale multimodal dataset reflecting the noisy, long-context nature of real-world retrieval.

• We propose Doc-REFRAG, which rethinks visual token compression as dynamic, questionguided selective expansion, via a three-stage curriculum with an RL-based selector.

• Experiments on six benchmarks show that our method outperforms eleven existing approaches.

## 2 Related Work

## 2.1 Document Understanding Models in RAG

RAG enhances MLLMs by retrieving external knowledge. In multimodal document RAG, existing research focuses on retriever design (Yu et al., 2024; Yan et al., 2025; Zhang et al., 2025a; Hu et al., 2025; Yang et al., 2025; Hu et al., 2026) while employing general-purpose MLLMs as generators. Early VDU MLLMs (Appalaraju et al., 2024; Wang et al., 2023) and subsequent multiimage variants (Feng et al., 2024; Duan et al., 2025; Liu et al., 2025) primarily target coherent, singlesource documents. In contrast, real-world RAG involves retrieving fragmented and contradictory images from diverse sources, creating a distribution shift that current models struggle to handle. While some works reason over massive document piles (Chen et al., 2025), they overlook the efficiency required for interactive generation.

## 2.2 Visual Token Compression

Multi-image inputs cause a token explosion that degrades inference efficiency, motivating visual token compression. These methods fall into trainingfree and training-aware approaches. Training-free methods typically adopt importance-based strategies (Chen et al., 2024a; Zhang et al., 2024c), pruning tokens by attention scores, and redundancybased approaches (Jiang et al., 2025; Wen et al., 2025a), merging similar tokens by feature similarity. Training-aware methods adopt lightweight component replacements (Dai et al., 2023; Dong et al., 2024; Li et al., 2024c). Recent examples include MQT-LLaVA (Hu et al., 2024c), which uses a dynamic Q-Former, and TokenPacker (Li et al., 2025) and DocOwl2 (Hu et al., 2024b), which use coarse-to-fine condensation. However, these methods prioritize visually salient regions (Zhang et al., 2025b), misaligning with question relevance in RAG. A structured comparison across six design dimensions is in App. A.

## 3 DocLongRAG

Conventional VDU datasets focus on individual or coherent images, offering limited context. Conversely, RAG requires reasoning across long, noisy,

![](images/0aaec6dac70e6bf570c1870a2c71f8c86c5331efc6a4fad458f0e26320ab91bd.jpg)  
Figure 2: Process of creating DocLongRAG.

and conflicting retrieved inputs. DocLongRAG simulates the scale and noise of practical retrieval, engineered for training stability.

## 3.1 Dataset Collection and Filtering

To support RAG training, we curate a diverse VDU corpus integrating Doc-750K (Duan et al., 2025) and OpenDocVQA (Tanaka et al., 2025), which aggregate English data from 15 open-source datasets (see App. B) and scientific documents. To address instances conflicting with RAG requirements or ethical guidelines, we filter four sample categories:

• Context-independent questions: Factoid questions answerable from parametric knowledge alone, without any retrieved document image (e.g., “Who won the Turing Award in 2018?”).

• Online search questions: Questions requiring external web resources beyond the document context (e.g., “What is the author’s Google Scholar citation count?”), violating the assumption that all information resides in the documents.

• Unanswerable questions: Questions lacking factual grounding in the documents, excluded as they do not support document-based reasoning.

• Ethically problematic content: Questions with personally identifiable information, offensive language, harmful instructions, or sensitive imagery, violating ethical standards.

This filtering reduces the pool by 18.2%, yielding a curated corpus C enforcing retrieval dependence. The corpus retains diverse formats (PDFs, web screenshots, slides) and answer types (extractive, abstractive), providing a challenging training foundation. This filtering applies only to the training corpus, while the evaluation benchmarks retain unanswerable and context-independent questions.

## 3.2 Negative Document Extension

We construct DocLongRAG to train VDU models on multiple document images under realistic retrieval noise. Inspired by (Gao et al., 2025), we interleave relevant images with hard-negative distractors from C to form extended input sequences.

<table><tr><td></td><td>Doc-750k</td><td>OpenDocVQA</td><td>DocLongRAG</td></tr><tr><td>Context-Dependent</td><td>X</td><td>√</td><td>√</td></tr><tr><td>Online-Independent</td><td>X</td><td>√</td><td>V</td></tr><tr><td>Answerable</td><td>√</td><td>X</td><td>√</td></tr><tr><td>Ethically Reviewed</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Retrieval-Noisy</td><td>X</td><td>X</td><td></td></tr><tr><td>#QA Pairs</td><td>758,000</td><td>43,474</td><td>343,474</td></tr><tr><td>#Images (Pages)</td><td>3,100,000</td><td>206,267</td><td>3,306,267</td></tr><tr><td>#Avg. Images per Question</td><td>4.1</td><td>2.1</td><td>37.4</td></tr></table>

Table 1: Comparison of related datasets.

Specifically, each instance in C comprises a question q, relevant document images $\begin{array} { r l } { D } & { { } = } \end{array}$ $\{ d _ { 1 } , \hdots , d _ { N } \}$ , and an answer a. As shown in Fig. 2, for each image $d _ { j } \in D _ { \operatorname { \mathrm { : } } }$ we retrieve the top-k most similar images from $C \setminus \mathcal { D }$ using the document-aware retriever (Günther et al., 2025). However, due to the semantic richness of document content, some images ∈ C \ D may still contain answer-supporting evidence. To avoid mislabeling such ambiguous candidates as negatives, we manually evaluate 500 randomly sampled queries and find that on average the top 2.47 retrieved images remain relevant (App. C). We therefore exclude the top-3 results and define the next m images (ranked 4 onward) as hard negatives for $d _ { j }$ , where m is uniformly sampled from [20, 30]. We construct the final sequence by interleaving each relevant image $d _ { j }$ with its hard negatives as $l = \big [ \{ d _ { j } , n _ { j } ^ { 1 } , \dots , n _ { j } ^ { m } \} _ { j = 1 } ^ { N } \big ]$ This yields a long, redundant, and sometimes conflicting multimodal context, closely mirroring real-world RAG. Applying this to all instances in C yields the DocLongRAG corpus. We emphasize that such constructed noise is used only for training, while evaluation relies on the noise produced by a real retriever over real benchmark documents (Sec. 5).

Through ablation studies on the placement position of relevant images (i.e., the meta-chunk) within the hard-negative sequence (App. D), we show that placing relevant document images before hard negatives consistently improves model performance during training. This strategic ordering helps establish longer dependencies, enhancing the effectiveness of DocLongRAG.

## 3.3 Comparison with Related Datasets

Tab. 1 presents a statistical comparison of DocLongRAG with existing VDU datasets. DocLongRAG differs from prior work along three key dimensions, establishing a more challenging and realistic setting for training. First, it aggregates instances from multiple source datasets, covering diverse document image formats and question types. Second, while Doc-750K contains context-independent questions and OpenDocVQA comprises unanswerable ones, DocLongRAG requires all answers to be grounded in the provided document images, enforcing a retrieval-dependent setting. Third, whereas Open-DocVQA grounds questions in individual images and Doc-750K in coherent document images, Doc-LongRAG associates each question with an average of 37.4 images, intentionally curated to include redundancy, noise, and conflicting evidence.

## 4 Doc-REFRAG

## 4.1 Model Architecture

The overall architecture of Doc-REFRAG is illustrated in Fig. 3, consisting of four key components: (1) a document-tailored vision encoder built upon DocOwl2, (2) a two-layer MLP projector for chunk embedding, (3) a lightweight RL-based selector for dynamic token restoration, and (4) a decoder-only language model. A consolidated overview of all components, training stages, and inference-time operations is provided in App. E.

Given a question consisting of q tokens, denoted as $\mathbf { t } = [ t _ { 1 } , \ldots , t _ { q } ]$ , each retrieved document image is encoded by the vision encoder, which maps the input image to a compact sequence of $m = 3 2 4$ visual tokens. Concatenating these visual tokens from all n retrieved images yields the initial visual token sequence in the input $\mathbf { v } = [ v _ { 1 } , \dots , v _ { n \times m } ]$

In RAG, only a small subset of regions is semantically relevant to the question, rendering most visual tokens redundant. Existing compression methods, however, are typically question-agnostic: they prioritize visually dense regions (e.g., text-heavy blocks) and discard sparser ones (e.g., blank areas), yet visual density correlates poorly with semantic relevance.

To address this mismatch, we propose a questionguided compression strategy that prioritizes semantic relevance over visual appearance. Specifically, Doc-REFRAG first partitions the visual token sequence of each retrieved image into $L \ = \ m / k$ non-overlapping chunks, which serve as units for coarse-level semantic modeling. For the i-th image, we define its chunk group $\mathcal { G } _ { ( i ) } = \{ C _ { i } ^ { 1 } , \ldots , C _ { i } ^ { L } \}$ where each chunk $C _ { i } ^ { j }$ consists of k tokens from $v _ { i }$ This chunking preserves the 2D arrangement of the page. We chunk layout-aware tokens rather than raw patches: each token from the encoder summarizes a fixed region of the page and has aggregated global context through self-attention. Since consecutive tokens in raster order are spatially adjacent, chunking them keeps each group localized on the page rather than scattered across arbitrary positions. Expansion then restores the original tokens of exactly the same region, reinstating fine-grained spatial detail where the selector fires. In this way, two-dimensional reasoning is inherited from the encoder, while our compression reuses rather than rebuilds this spatial structure.

A two-layer MLP projector $\phi$ then projects each chunk into a single d-dimensional embedding $\mathbf { c } _ { i } ^ { j } = \phi ( C _ { i } ^ { j } ) \in \mathbb { R } ^ { d } .$ , where $d$ denotes the hidden dimension of the decoder’s input embedding space, preserving semantic content while reducing length. The chunk embeddings from all retrieved images are concatenated with question tokens to form the input sequence:

$$
\mathbf { z } = [ t _ { 1 } , \dots , t _ { q } , \mathbf { c } _ { 1 } ^ { 1 } , \dots , \mathbf { c } _ { n } ^ { L } ] \in \mathbb { R } ^ { ( q + n \times L ) \times d } ,
$$

which is fed into a decoder-only foundation model to generate the answer $y \sim p _ { \boldsymbol { \theta } } ( y \mid \mathbf { z } )$

To retain fine-grained details that coarse embeddings may obscure, a lightweight RL-based selector dynamically identifies critical chunks and expands them back to their original k constituent tokens during decoding. Let $\delta _ { i } ^ { j } \in \{ 0 , 1 \}$ denote whether the j-th chunk of the i-th image should be expanded. The final input sequence becomes $\mathbf { z } ^ { \prime } = [ t _ { 1 } , \ldots , t _ { q } , { \tilde { \mathbf { c } } } _ { 1 } ^ { 1 } , \ldots , { \tilde { \mathbf { c } } } _ { n } ^ { L } ]$ , where $\tilde { \mathbf { c } } _ { i } ^ { j } = C _ { i } ^ { j } \in$ $\mathbb { R } ^ { k \times d } \mathrm { i f } \delta _ { i } ^ { j } = 1$ and $\tilde { \mathbf { c } } _ { i } ^ { j } = \mathbf { c } _ { i } ^ { j } \in \mathbb { R } ^ { d }$ otherwise. This mechanism balances computational efficiency and semantic fidelity based on the question.

## 4.2 Model Training

To align the encoder and decoder with ${ \mathbf z } ^ { \prime } ,$ we adopt a three-stage protocol: Single-Image Reconstruction, Multi-image Continual Pretraining (CPT), and RL-Based Selector Training.

Single-Image Reconstruction. We freeze the decoder and train the encoder and MLP projector on DocStruct4M (Hu et al., 2024a) with two objectives: (1) aligning chunk embeddings with the decoder’s input embedding space, and (2) preserving original visual semantics. We formulate two complementary tasks: text parsing, where the model recognizes all textual content in a document image, and text localization, where it extracts text from a spatially specified region. The encoder is then frozen for subsequent stages.

![](images/6dfb37fc3ca5441f5c51255767dc7a2d7ebb23e0de81aee5a58daebf010a8f46.jpg)  
Figure 3: Architecture of Doc-REFRAG. Retrieved document images are encoded patch-level into chunk embeddings. A query-aware RL-based selector identifies informative chunks, which are expanded to fine-grained tokens before being fed with the query into a decoder-only language model.

Multi-image Continual Pretraining. In the CPT stage, we employ curriculum learning to train the decoder on increasingly complex datasets, with two objectives: (1) scaling from single- to multi-image settings, and (2) enabling effective processing of hybrid sequences that interleave chunk and visual token embeddings.

Chunk expansion is supervised by pseudo-labels $\delta _ { i } ^ { j }$ derived from ColQwen (Faysse et al., 2024) question-to-patch similarity scores. Since our encoder and ColQwen share the same normalized image coordinate convention, a deterministic geometric mapping from each coarse chunk to the fine patches it subsumes is established (App. $\mathrm { F } ;$ robustness in App. G). At inference time, $\delta _ { i } ^ { j }$ is predicted by the RL-based selector.

Building upon this pseudo-labeling strategy, we begin with OpenDocVQA (Tanaka et al., 2025), a single-image VDU dataset: for each question, we expand the top 20% of chunks ranked by ColQwen similarity scores, enabling the decoder to fuse finegrained visual details with coarse chunk summaries within a single image.

Next, we train on Doc-750K, a multi-image VDU dataset comprising documents split across multiple images, applying the same 20% expansion per image. The resulting longer hybrid sequences challenge the decoder to perform cross-image reasoning under extended context.

Finally, we train on DocLongRAG, which simulates realistic RAG conditions. Unlike Doc-750K, DocLongRAG comprises images from heterogeneous sources exhibiting redundancy, semantic overlap, and contradiction. We reduce expansion to 10% and additionally activate randomly selected irrelevant chunks, introducing a dual challenge: noisy heterogeneous context combined with sparse and imperfect token expansion, thereby better aligning the model with practical RAG deployment.

RL-Based Selector Training. The RL-based selector is a lightweight two-layer transformer that computes selection logits over all chunk embeddings. Given the question t and chunk sequence $\{ \mathbf { c } _ { i } ^ { j } \}$ , the policy $\pi _ { \theta }$ samples binary decisions $\delta _ { i } ^ { j } \in$ {0, 1} to determine whether each chunk should be expanded to its original k tokens. The policy is constrained to select exactly $T = \lfloor p \cdot n \cdot L \rfloor$ chunks for expansion, where $p \in ( 0 , 1 ]$ ] controls the budget. We train a dedicated selector for each value of k and demonstrate the influence in App. H. In our experiments, $p$ is uniformly sampled from [0.1, 0.3] during training and fixed at 0.2 during inference, which is identified as optimal in App. I.

We train the selector on DocLongRAG using a GRPO-style policy optimization framework with clipped PPO updates. The reward signal is the answer accuracy of the frozen decoder when processing the input sequence $\{ \delta _ { i } ^ { j } \}$ constructed from the selector’s choices. As shown in Sec. 5.2, this yields lower inference latency than ColQwen with competitive accuracy. Full training details, including trajectory sampling and advantage normalization, are provided in App. J.

## 5 Experiments

## 5.1 Experimental Setup

Dataset & Benchmarks. For training, we perform single-image reconstruction with Doc-Struct4M, followed by multi-image CPT on Open-

<table><tr><td>Model</td><td>Size</td><td>TokenV</td><td>ChartQA</td><td>SlideVQA</td><td>InfoVQA</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>#Acc.</td><td>#TTFT</td><td>#TTIT</td></tr><tr><td>General MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL3.5</td><td>8B</td><td>~2,304</td><td>56.8</td><td>53.1</td><td>48.1</td><td>45.8</td><td>42.1</td><td>51.5</td><td>49.5</td><td>2.78</td><td>0.92</td></tr><tr><td>LLaVA-OneVision</td><td>8B</td><td>~1,920</td><td>52.1</td><td>50.9</td><td>46.3</td><td>47.2</td><td>42.7</td><td>49.0</td><td>48.0</td><td>4.03</td><td>0.05</td></tr><tr><td>MiniCPM-V-2.6</td><td>8B</td><td>~576</td><td>54.8</td><td>52.3</td><td>48.5</td><td>46.4</td><td>41.1</td><td>48.8</td><td>48.6</td><td>5.80</td><td>0.12</td></tr><tr><td>Qwen3-VL</td><td>8B</td><td>~512</td><td>58.2</td><td>59.2</td><td>55.7</td><td>52.5</td><td>43.5</td><td>52.4</td><td>51.3</td><td>5.91</td><td>0.05</td></tr><tr><td>MQT-LLaVA†</td><td>7B</td><td>256</td><td>35.7</td><td>39.3</td><td>33.4</td><td>40.5</td><td>37.4</td><td>34.2</td><td>36.7</td><td>3.43</td><td>0.05</td></tr><tr><td>EPIC†</td><td>7B</td><td>256</td><td>39.7</td><td>41.1</td><td>42.7</td><td>32.4</td><td>33.3</td><td>38.2</td><td>37.9</td><td>3.42</td><td>0.06</td></tr><tr><td>LLaVA-Mini†</td><td>7B</td><td>144</td><td>35.4</td><td>38.5</td><td>44.6</td><td>37.5</td><td>39.1</td><td>40.2</td><td>39.2</td><td>2.67</td><td>0.06</td></tr><tr><td colspan="10">Document-aware MLLMs</td><td></td></tr><tr><td>Docopilot</td><td>8B</td><td>~2304</td><td>58.7</td><td>56.6</td><td>49.4</td><td>48.6</td><td>45.7</td><td>46.5</td><td>50.9</td><td>1.81</td><td>2.25</td></tr><tr><td>VDocGenerator</td><td>3B</td><td>~1920</td><td>53.3</td><td>42.8</td><td>43.2</td><td>37.9</td><td>34.6</td><td>36.8</td><td>41.4</td><td>9.20</td><td>6.40</td></tr><tr><td>DocOwl2†</td><td>7B</td><td>~324</td><td>57.5</td><td>53.2</td><td>45.2</td><td>44.7</td><td>47.5</td><td>45.2</td><td>48.9</td><td>3.94</td><td>0.07</td></tr><tr><td>TokenPacker†</td><td>7B</td><td>144</td><td>46.4</td><td>39.9</td><td>41.3</td><td>33.5</td><td>36.7</td><td>35.9</td><td>37.5</td><td>4.43</td><td>0.05</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Doc-REFRAG</td><td>7B</td><td>~150</td><td>61.2</td><td>60.8</td><td>58.7</td><td>53.2</td><td>57.3</td><td>54.9</td><td>57.5</td><td>2.57</td><td>0.07</td></tr></table>

Table 2: Comparison with general MLLMs and document-aware MLLMs on six benchmarks under RAG settings. The ‘<sup>†</sup>’ indicates models that use visual token compression. ‘Token<sup>V</sup>’ denotes the average number of visual tokens per image. ‘Bold’ indicates SOTA performance within the group and ‘Underline’ indicates second-best performance within the group. Accuracy (% ↑), TTFT, and TTIT are reported in seconds (↓). # denotes averaged metrics.

DocVQA, Doc-750K, and DocLongRAG. The selector is also trained on DocLongRAG. All testbenchmark images are excluded to prevent data leakage. Benchmarks include: ChartQA (Masry et al., 2022), SlideVQA (Tanaka et al., 2023), InfoVQA (Mathew et al., 2022), DUDE (Van Landeghem et al., 2023), ViDoRe-v2 (Macé et al., 2025), and MMDocIR (Dong et al., 2025a).

Baselines. We compare Doc-REFRAG with MLLMs that support multi-image input, which excludes methods designed for single-image settings. First, we include general MLLMs: Qwen3- VL (Bai et al., 2025), MiniCPM-V-2.6 (Yao et al., 2024), LLaVA-OneVision (Li et al., 2024a), and InternVL3.5 (Chen et al., 2024b). Second, we evaluate visual token compression variants: MQA-LLaVA (Hu et al., 2024c), EPIC (Wen et al., 2025b), and LLaVA-Mini (Zhang et al., 2025b). Finally, we include models designed for document understanding: VDocGenerator (Tanaka et al., 2025), Docopilot (Duan et al., 2025), TokenPacker (Li et al., 2025), and DocOwl2 (Hu et al., 2024b).

Implementation details. Doc-REFRAG is built upon DocOwl2, adopting its encoder and decoder as the backbone. In single-image reconstruction, we fine-tune the linear vision-to-text alignment module in the encoder, along with a two-layer MLP projector (hidden dimension 4096, matching the decoder’s output dimension), using a peak learning rate of $2 \times 1 0 ^ { - 4 }$ . In multi-image CPT, we employ LoRA (rank = 64) (Hu et al., 2022) to finetune the decoder using AdamW $( 1 \times 1 0 ^ { - 5 }$ learning rate, 4% linear warmup, cosine decay) with a batch size of 64. Both stages use cross-entropy loss over predicted tokens. The RL-based selector is trained with GRPO (clip ratio = 0.2). All stages train for 2 epochs. Experiments are conducted on 8 × NVIDIA RTX 3090 GPUs. Per-stage datasets and trainable parameters are detailed in App. K.

## 5.2 Main Result

We evaluate Doc-REFRAG against eleven state-ofthe-art MLLMs across six VDU benchmarks. To assess model robustness under realistic RAG settings, we retrieve the top-20 images for each question from its source benchmark with the state-of-theart retriever Jina-Embeddings-V4 (Günther et al., 2025), and provide them as input to all MLLMs. These top-20 candidates cover 93.4% of the gold images on average across the six benchmarks (Recall@20), indicating that retrieval recall is not the bottleneck in our evaluation. We present quantitative results in this section and defer qualitative examples to App. L.

Quality analysis. As shown in Tab. 2, Doc-REFRAG with k = 3 outperforms both general MLLMs and specialized long-document MLLMs. This gain stems from two design aspects. First, these MLLMs are typically trained on short, coherent documents. This diverges from the noisy retrieval contexts in evaluation, which include redundant, conflicting, or irrelevant content that impairs reasoning. In contrast, Doc-REFRAG trains on DocLongRAG, which is explicitly designed to mimic such noisy retrieval contexts. Second, most MLLMs degrade with input length due to attention saturation, while Doc-REFRAG’s question-guided compression dynamically reduces tokens while preserving task-critical information.

Efficiency Analysis. We measure TTFT and TTIT latency under a batch size of 1 with 20 retrieved images per question (accuracy–latency trade-offs with more retrieved images are analyzed in App. M). Within Doc-REFRAG, TTFT drops from 3.24 s (k=2, 194 tokens) to 1.95 s (k=6, 108 tokens), with k=3 chosen as the default at 2.57 s for the best accuracy–efficiency trade-off (Tab. 2). Compared to Docopilot, the most accurate baseline, Doc-REFRAG incurs a slight TTFT increase but reduces TTIT by 2.18 s, accelerating end-toend generation for long outputs. When compared to token-compressed MLLMs under a comparable visual token budget, Doc-REFRAG matches their TTFT while improving accuracy by over 18%.

Selector Analysis. Doc-REFRAG uses an RL selector to decide which compressed chunks to expand at decoding. We compare three strategies: (1) Random, uniformly sampling chunks; (2) ColQwen, selecting chunks with the highest question-chunk similarity, mirroring training; (3) Perplexity-based, a heuristic ranking chunks by decoder perplexity given the question and chunk embeddings, with lower values indicating higher relevance.

As shown in Tab. 3, our RL-based selector outperforms random (38.8%) and perplexity-based (43.7%) strategies, achieving 57.5% average accuracy. In comparison, ColQwen, a dedicated dense retriever for question-document relevance, achieves higher accuracy on InfoVQA, possibly because InfoVQA is part of its training data (Faysse et al., 2024). However, ColQwen computes similarity scores over patches and aggregates them to infer chunk relevance, incurring a latency of 9.3 s. By contrast, our selector identifies relevant chunks in a single forward pass at 3.7 s.

## 5.3 Analysis

Can the chunk embedding handle dense text? Chunk embeddings risk losing textual details critical for RAG. We design a synthetic benchmark of A4 document images (595 × 842 pixels), synthesizing English passages via PyMuPDF with font sizes 10–20 and 500–6,000 characters per image.

We evaluate all models by their ability to recover original text from images using Average Normalized Levenshtein Similarity (ANLS). As shown in Fig. 4, Doc-REFRAG (150 tokens/image) achieves ANLS above 95% for documents up to 4,000 characters, matching DeepSeek-OCR (256 tokens) (Wei et al., 2025) and DocOwl2 (324 tokens) despite its compact representation. Even at 6,000 characters, performance degrades only marginally, remaining within a 5% absolute gap from the baselines. This demonstrates that chunk embeddings preserve textual semantics even under high lexical density.

Can our training tasks be beneficial? Tab. 4 confirms that our training tasks are essential. Removing single-image reconstruction drops accuracy from 55.1% to 19.7%, showing that chunk embeddings lose semantic fidelity without reconstruction supervision. Replacing curriculum learning with randomly shuffled multi-image sequences reduces accuracy to 24.2%, showing that mixedtoken handling requires progressive exposure during training. These results validate that our training strategy enables effective compression while preserving textual alignment.

Can our dataset improve the performance? Conventional VDU datasets often lack the noise and redundancy characteristic of multi-image retrieval. Excluding DocLongRAG from training reduces Doc-REFRAG’s average accuracy from 55.1% to 48.9%, underscoring its critical role in robust RAG adaptation. To assess generalizability, we fine-tune InternVL3.5, MiniCPM-V-2.6, and DocOwl2 on DocLongRAG, yielding accuracy improvements of 3.9%, 4.3%, and 3.9%, respectively. This consistent cross-model gain validates DocLongRAG as the first long-context, challenging dataset for multimodal document RAG. A 2×2 factorial decomposition of architecture and data contributions is provided in App. N. To validate that gains stem from the construction pipeline rather than mere data volume, we compare against a naive random baseline in App. O.

Can Doc-REFRAG help improve the reranker performance? Document reranking is commonly formulated as binary classification, where MLLMs sequentially assess the relevance of each candidate image to a given question via the logit of the “Yes” token. However, processing each image incurs high inference latency, prohibitive across many candidates.

![](images/21428c667a6affc8470c11a46066204bee56e51e0ec41035feef9775b9f890b0.jpg)  
Figure 4: Parsing performance on A4 document image.

<table><tr><td>Strategy</td><td>ChartQA</td><td>SlideVQA</td><td>InfoVQA</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>Average</td><td>Latency</td></tr><tr><td>Random</td><td>39.4</td><td>42.5</td><td>48.3</td><td>33.2</td><td>38.6</td><td>30.8</td><td>38.8</td><td></td></tr><tr><td>Perplexity</td><td>46.5</td><td>48.3</td><td>47.2</td><td>39.6</td><td>40.4</td><td>40.3</td><td>43.7</td><td>20.7</td></tr><tr><td>ColQwen</td><td>56.6</td><td>54.0</td><td>59.5</td><td>49.8</td><td>53.5</td><td>52.3</td><td>53.6</td><td>9.3</td></tr><tr><td>RL (Ours)</td><td>61.2</td><td>60.8</td><td>58.7</td><td>53.2</td><td>57.3</td><td>54.9</td><td>57.5</td><td>3.7</td></tr></table>

Table 3: Performance of different selection strategies for Doc-REFRAG. Accuracy (% ↑) and Latency in seconds (↓) are measured on 8×RTX 3090 GPUs.

<table><tr><td>Variant</td><td>Vidore</td><td>MMDocIR</td><td>DUDE</td><td>Average</td></tr><tr><td>Doc-REFRAG (Full)</td><td>53.2</td><td>57.3</td><td>54.9</td><td>55.1</td></tr><tr><td>w/o Single-image Recon.</td><td>20.2</td><td>16.5</td><td>22.5</td><td>19.7</td></tr><tr><td>w/o Curriculum Learning</td><td>24.5</td><td>22.8</td><td>25.4</td><td>24.2</td></tr><tr><td>w/o Both</td><td>7.2</td><td>8.5</td><td>8.4</td><td>8.0</td></tr></table>

Table 4: Ablation study of training tasks. Results are reported as accuracy (% ↑) on three key benchmarks.

<table><tr><td>Model</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>Average</td></tr><tr><td>InternVL3.5 (orig.)</td><td>45.8</td><td>42.1</td><td>51.5</td><td>46.5</td></tr><tr><td>InternVL3.5 (ft.)</td><td>47.1</td><td>50.4</td><td>53.8</td><td>50.4</td></tr><tr><td>MiniCPM-V-2.6 (orig.)</td><td>46.4</td><td>41.1</td><td>48.8</td><td>45.4</td></tr><tr><td>MiniCPM-V-2.6 (ft.)</td><td>51.0</td><td>45.3</td><td>52.8</td><td>49.7</td></tr><tr><td>DocOwl2 (orig.)</td><td>44.7</td><td>47.5</td><td>45.2</td><td>45.8</td></tr><tr><td>DocOwl2 (ft.)</td><td>48.4</td><td>51.6</td><td>49.1</td><td>49.7</td></tr><tr><td>Doc-REFRAG (w/o DocLongRAG)</td><td>48.5</td><td>43.7</td><td>54.6</td><td>48.9</td></tr><tr><td>Doc-REFRAG (Full)</td><td>53.2</td><td>57.3</td><td>54.9</td><td>55.1</td></tr></table>

Table 5: Ablation study of DocLongRAG. General and document-aware MLLMs are compared in original (orig.) and fine-tuned (ft.) versions on DocLongRAG. “Doc-REFRAG (w/o DocLongRAG)” denotes our model trained without the DocLongRAG corpus. Results are reported as accuracy (% ↑).

<table><tr><td>Model</td><td>MMDocIR</td><td>Vidore</td><td>DUDE</td><td>Latency</td></tr><tr><td>Jina-m0 (Günther et al., 2025)</td><td>81.1</td><td>84.6</td><td>86.2</td><td>13.6</td></tr><tr><td>MM-R5 (Xu et al., 2025)</td><td>82.3</td><td>81.9</td><td>81.3</td><td>49.5</td></tr><tr><td>MonoQwen2 (Chaffin and Lac, 2024)</td><td>81.4</td><td>82.1</td><td>82.4</td><td>12.5</td></tr><tr><td>Doc-REFRAG</td><td>84.2</td><td>85.8</td><td>87.7</td><td>7.3</td></tr></table>

Table 6: Performance comparison across rerankers. Results are reported as Recall@10 (↑). Latency is measured in seconds (↓).

Doc-REFRAG reduces per-document cost through efficient visual encoding. By globally compressing visual tokens and dynamically restoring question-relevant details, it enables faster relevance scoring without sacrificing accuracy. As shown in Tab. 6, Doc-REFRAG achieves higher reranking recall than dedicated rerankers on MMDocIR, yet incurs much lower per-document latency.

<table><tr><td>Pipeline Strategy</td><td>Selection Method</td><td>Latency</td><td>Acc.</td></tr><tr><td>Direct Retrieval</td><td>Top-5 (No Reranker)</td><td>4.2</td><td>46.6</td></tr><tr><td>Conventional Reranking</td><td>MonoQwen2 (Top-20→5)</td><td>15.4</td><td>53.2</td></tr><tr><td>Doc-REFRAG</td><td>Selective Encoding (Top-20)</td><td>6.1</td><td>57.5</td></tr></table>

Table 7: Comparison of different RAG pipelines. Results are reported as accuracy (% ↑) averaged across six benchmarks, and total end-to-end time in seconds (↓), including retrieval, (re-)ranking, and generation.

Can Doc-REFRAG surpass the reranker pipeline? We compare Doc-REFRAG against two standard retrieval-then-rerank baselines: (1) Direct Retrieval, which feeds the top-5 candidates from the retriever Jina-Embeddings-V4 directly into Qwen3-VL-8B, and (2) Reranking Pipeline, which uses MonoQwen2 (Chaffin and Lac, 2024) to rerank top-20 candidates to 5 before answering.

As shown in Tab. 7, the Direct Retrieval baseline suffers from evidence loss due to the performance ceiling of dual-tower retrievers, where limited topk candidates fail to encapsulate sufficient information. The Reranking Pipeline mitigates this loss, improving accuracy to 53.2%, but incurs 15.4s latency from the LLM-based reranking stage (∼0.63s/img). In contrast, Doc-REFRAG processes all top-20 candidates via question-guided compression, achieving 57.5% accuracy in only 6.1s, resolving the trade-off between retrieval recall and inference efficiency in multimodal RAG.

## 6 Conclusion

We introduce Doc-REFRAG, an efficient decoding framework for multimodal document RAG. It employs question-guided compression and a three-stage training paradigm to compress visual tokens into chunks, selectively expanded via an RLbased selector. Experiments across six benchmarks show Doc-REFRAG outperforms eleven baselines, achieving state-of-the-art accuracy with lower latency. We further validate our training methodology, the role of DocLongRAG, and its transferability to document reranking. Our approach enables efficient large-context inference in RAG, especially in resource-constrained deployments.

## Limitations

While Doc-REFRAG advances efficient multimodal document RAG, three limitations remain. First, the chunk size k is fixed as a training hyperparameter and cannot adapt dynamically at in ference time. Although k = 3 achieves the best overall accuracy–efficiency trade-off (App. H), performance degrades by approximately 5% ANLS on documents exceeding 4,000 characters, indicat ing that adaptive granularity could further improve coverage in high-density layouts. Adapting k at inference time, however, is not free. Each selector is trained at a fixed k, so supporting multiple chunk sizes requires an additional routing step that dispatches each input to the matching selector, whose latency works against the efficiency goal of the method. We leave a single selector that generalizes across chunk sizes to future work. Second, the RL based selector is trained with answer accuracy as the sole reward signal, providing no fine-grained attribution of which individual chunks contributed to the correct answer. As shown in App. L, 64% of remaining errors concentrate on dense or structurally complex text regions, suggesting that richer reward shaping or chunk-level supervision could improve selector precision in these cases. Third, while our evaluation uses a real retriever over real documents, the training noise in DocLongRAG is constructed, and no single construction pipeline covers every deployment scenario. DocLongRAG is also limited to English documents. Although the construction pipeline is language-agnostic, dense CJK text packs more information per patch and may favor a smaller chunk size. Finally, due to the high computational cost of training large mul timodal models, we report results from a single training run, which is standard practice in the mul timodal document understanding community. All baseline comparisons follow the same single-run protocol.

## Ethics Statement

DocLongRAG is constructed from publicly available datasets and document corpora. During dataset construction, we apply a four-stage filtering pipeline that removes context-independent questions, online-search-dependent queries, unanswerable instances, and ethically problematic content including personally identifiable information, offensive language, and sensitive imagery (Sec. 3). The synthetic document images used in our densetext evaluation are generated programmatically and contain no real user data.

Doc-REFRAG is designed as a document understanding tool and does not store or transmit user documents beyond the scope of inference. Although grounding responses in retrieved documents reduces hallucination risk relative to purely parametric models, the system can still produce errors in documents with dense, overlapping, or contradictory evidence. Deployment in high-stakes decision-making contexts requires human oversight and should not rely solely on model outputs.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant No. U24A20326. We thank the anonymous reviewers and action editors for their constructive feedback.

## References

Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, and R Manmatha. 2024. Docformerv2: Local features for document understanding. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 709–718.

Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. 2024. A survey on rag with llms. Procedia computer science, 246:3781–3790.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. Longalign: A recipe for long context alignment of large language models. arXiv preprint arXiv:2401.18058.

Antoine Chaffin and Aurélien Lac. 2024. Monoqwen: Visual document reranking.

Jun Chen, Dannong Xu, Junjie Fei, Chun-Mei Feng, and Mohamed Elhoseiny. 2025. Document haystacks: Vision-language reasoning over piles of 1000+ documents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24817– 24826.

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2023. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, and 1 others. 2024b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The power of noise: Redefining retrieval for rag systems. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 719–729.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267.

Kuicai Dong, Yujing Chang, Xin Deik Goh, Dexun Li, Ruiming Tang, and Yong Liu. 2025a. Mmdocir: Benchmarking multi-modal retrieval for long documents. arXiv preprint arXiv:2501.08828.

Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, and Yong Liu. 2025b. Benchmarking retrieval-augmented multimodal generation for document question answering. arXiv preprint arXiv:2505.16470.

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, and 1 others. 2024. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions from 336 pixels to 4k hd. Advances in Neural Information Processing Systems, 37:42566–42592.

Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, and 1 others. 2025. Docopilot: Improving multimodal models for documentlevel understanding. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 4026–4037.

Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024. Colpali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449.

Hao Feng, Qi Liu, Hao Liu, Jingqun Tang, Wengang Zhou, Houqiang Li, and Can Huang. 2024. Docpedia: Unleashing the power of large multimodal model in the frequency domain for versatile document understanding. Science China Information Sciences, 67(12):220106.

Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, and Songlin Hu. 2025. Nextlong: Toward effective long-context training without long documents. arXiv preprint arXiv:2501.12766.

Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang, and 1 others. 2025. jina-embeddings-v4: Universal embeddings for multimodal multilingual retrieval. arXiv preprint arXiv:2506.18902.

Yefei He, Feng Chen, Jing Liu, Wenqi Shao, Hong Zhou, Kaipeng Zhang, and Bohan Zhuang. 2024. Zipvl: Efficient large vision-language models with dynamic token sparsification and kv cache compression. arXiv preprint arXiv:2410.08584.

Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, and 1 others. 2024a. mplug-docowl 1.5: Unified structure learning for ocr-free document understanding. arXiv preprint arXiv:2403.12895.

Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2024b. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. arXiv preprint arXiv:2409.03420.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Ruofan Hu, Yan Xia, Minjie Hong, Jieming Zhu, Bo Chen, Xiaoda Yang, Minghui Fang, and Tao Jin. 2025. Vela: scalable embeddings with voice large language models for multimodal retrieval. arXiv preprint arXiv:2506.14445.

Ruofan Hu, Menghui Zhu, Jieming Zhu, Bo Chen, Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Li Tang, Tao Jin, and 1 others. 2026. Docretriever: A plug-and-play framework for multimodal document retrieval with comprehensive benchmark. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 1780–1791.

Wenbo Hu, Zi-Yi Dou, Liunian Li, Amita Kamath, Nanyun Peng, and Kai-Wei Chang. 2024c. Matryoshka query transformer for large vision-language models. Advances in Neural Information Processing Systems, 37:50168–50188.

Kai Huang, Hao Zou, Ye Xi, BoChen Wang, Zhen Xie, and Liang Yu. 2024. IVTP: Instruction-Guided Visual Token Pruningfor Large Vision-Language Models, page 214–230. Springer Nature Switzerland.

Yutao Jiang, Qiong Wu, Wenhao Lin, Wei Yu, and Yiyi Zhou. 2025. What kind of visual tokens do we need? training-free visual token pruning for multi-modal large language models from the perspective of graph.

In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 4075–4083.

Samir Khaki, Junxian Guo, Jiaming Tang, Shang Yang, Yukang Chen, Konstantinos N. Plataniotis, Yao Lu, Song Han, and Zhijian Liu. 2025. Sparsevila: Decoupling visual sparsity for efficient vlm inference. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), page 23784–23794. IEEE.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. 2024. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong, Qing Li, and Lei Chen. 2024b. A survey on large language model acceleration based on kv cache management. arXiv preprint arXiv:2412.19442.

Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. 2025. Tokenpacker: Efficient visual projector for multimodal llm. International Journal of Computer Vision, pages 1–19.

Yanwei Li, Chengyao Wang, and Jiaya Jia. 2024c. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer.

Ronghui Liu, Hao Ren, Haojie Ren, Wu Rui, Wei Cui, Xiaojun Liang, Chunhua Yang, and Weihua Gui. 2025. Knowledge enhanced industrial questionanswering using large language models. Engineering.

Yuliang Liu, Biao Yang, Qiang Liu, Zhang Li, Zhiyin Ma, Shuo Zhang, and Xiang Bai. 2024a. Textmonkey: An ocr-free large multimodal model for understanding document. arXiv preprint arXiv:2403.04473.

Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and 1 others. 2024b. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. Advances in Neural Information Processing Systems, 37:8698–8733.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, and 1 others. 2024. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525.

Qi Luo, Xiaonan Li, Tingshuo Fan, Xinchi Chen, and Xipeng Qiu. 2025. Towards global retrieval augmented generation: A benchmark for corpus-level reasoning. arXiv preprint arXiv:2510.26205.

Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems, 37:95963–96010.

Quentin Macé, António Loison, and Manuel Faysse. 2025. Vidore benchmark v2: Raising the bar for visual retrieval. arXiv preprint arXiv:2505.17166.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings ofthe IEEE/CVF winter conference on applications of computer vision, pages 2200–2209.

Xudong Tan, Peng Ye, Chongjun Tu, Jianjian Cao, Yaoxin Yang, Lin Zhang, Dongzhan Zhou, and Tao Chen. 2025. Tokencarve: Information-preserving visual token compression in multimodal large language models. arXiv preprint arXiv:2503.10501.

Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2025. Vdocrag: Retrieval-augmented generation over visually-rich documents. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 24827–24837.

Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13636–13645.

Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021. Visualmrc: Machine reading comprehension on document images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 13878– 13888.

Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multipage docvqa. Pattern Recognition, 144:109834.

Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski,

Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, and 1 others. 2023. Document understanding dataset and evaluation (dude). In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19528–19540.

Dongsheng Wang, Natraj Raman, Mathieu Sibue, Zhiqiang Ma, Petr Babkin, Simerjot Kaur, Yulong Pei, Armineh Nourbakhsh, and Xiaomo Liu. 2023. Docllm: A layout-aware generative language model for multimodal document understanding. arXiv preprint arXiv:2401.00908.

Haicheng Wang, Zhemeng Yu, Gabriele Spadaro, Chen Ju, Victor Quétu, Shuai Xiao, and Enzo Tartaglione. 2025. Folder: Accelerating multi-modal large language models with enhanced performance. arXiv preprint arXiv:2501.02430.

Haoran Wei, Yaofeng Sun, and Yukun Li. 2025. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234.

Zichen Wen, Yifeng Gao, Weijia Li, Conghui He, and Linfeng Zhang. 2025a. Token pruning in multimodal large language models: Are we solving the right problem? arXiv preprint arXiv:2502.11501.

Zichen Wen, Shaobo Wang, Yufa Zhou, Junyuan Zhang, Qintong Zhang, Yifeng Gao, Zhaorun Chen, Bin Wang, Weijia Li, Conghui He, and 1 others. 2025b. Efficient multi-modal large language models via progressive consistency distillation. arXiv preprint arXiv:2510.00515.

Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, and Hengxing Cai. 2025. Mm-r5: Multimodal reasoning-enhanced reranker via reinforcement learning for document retrieval. arXiv preprint arXiv:2506.12364.

Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James Kwok, and Xuming Hu. 2025. Docpruner: A storageefficient framework for multi-vector visual document retrieval via adaptive patch-level embedding pruning. arXiv preprint arXiv:2509.23883.

Jianxin Yang. 2023. Longqlora: Efficient and effective method to extend context length of large language models. arXiv preprint arXiv:2311.04879.

Xiaoda Yang, Xize Cheng, Minghui Fang, Hongshun Qiu, Yuhang Ma, JunYu Lu, Jiaqi Duan, Sihang Cai, Zehan Wang, Ruofan Hu, and 1 others. 2025. Multimodal conditional retrieval with high controllability. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 3577–3585.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, and 1 others. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, and 1 others. 2023. mplugdocowl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499.

Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and 1 others. 2024. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594.

Chenghao Zhang, Guanting Dong, Xinyu Yang, and Zhicheng Dou. 2025a. Towards mixed-modal retrieval for universal retrieval-augmented generation. arXiv preprint arXiv:2510.17354.

Jiajie Zhang, Yushi Bai, Xin Lv, Wanjun Gu, Danqing Liu, Minhao Zou, Shulin Cao, Lei Hou, Yuxiao Dong, Ling Feng, and 1 others. 2024a. Longcite: Enabling llms to generate fine-grained citations in long-context qa. arXiv preprint arXiv:2409.02897.

Jiajie Zhang, Zhongni Hou, Xin Lv, Shulin Cao, Zhenyu Hou, Yilin Niu, Lei Hou, Yuxiao Dong, Ling Feng, and Juanzi Li. 2024b. Longreward: Improving long-context large language models with ai feedback. arXiv preprint arXiv:2410.21252.

Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, and Qin Jin. 2023. Mpmqa: multimodal question answering on product manuals. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 13958–13966.

Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. 2025b. Llava-mini: Efficient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895.

Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and 1 others. 2024c. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417.

## A Differentiated Contributions Over Related Compression Methods

We provide a structured comparison of Doc-REFRAG against the most closely related visual token compression methods across six dimensions in Tab. 8. The comparison covers methods that appear as baselines in our main evaluation (Tab. 2), DocOwl2 (our base model), and three additional query-conditioned compression methods (Sparse-VILA, IVTP, LLaMA-VID) that are most architecturally similar to ours.

We elaborate on the dimensions where Doc-REFRAG introduces substantive novelty.

Query-conditioned compression. Methods such as SparseVILA, IVTP, and LLaMA-VID condition token selection or compression on the input query. However, their query-conditioning mechanisms are heuristic: SparseVILA scores visual tokens against the query using attention at decode time (training-free); IVTP aggregates text instructions into a CLS token to guide pruning via attention rollout; LLaMA-VID generates a querydependent context token alongside fixed content tokens. None of these methods learn a selection policy through end-to-end optimization against a downstream task reward. Doc-REFRAG trains its selector via GRPO, directly optimizing chunk selection for answer accuracy—making the selection policy a first-class learned component rather than a post-hoc heuristic.

Dynamic budget with RL training. Prior methods either fix the compression ratio at design time (IVTP, EPIC, LLaVA-Mini) or apply queryindependent dynamic sparsity (SparseVILA). Doc-REFRAG adopts a dynamic budget $p \in [ 0 . 1 , 0 . 3 ]$ sampled during training, enabling the model to learn budget-robust selection policies via GRPO with clipped PPO updates, using the frozen decoder’s answer accuracy as the reward signal—a training paradigm not previously applied to visual token compression.

RAG noise robustness. All existing compression methods are developed and evaluated on coherent single-source documents or video frames. Doc-REFRAG is trained on DocLongRAG, which explicitly simulates realistic retrieval noise: retrieved images are heterogeneous, may be redundant or contradictory, and include hard negatives interleaved with relevant content. This distributional alignment between training and deployment is a key factor behind Doc-REFRAG’s performance advantage in RAG settings (Tab. 2).

Learned restoration vs. heuristic retrieval vs. condensation. SparseVILA retains the full KV cache at prefill and selectively retrieves tokens per decoding step—a form of training-free heuristic restoration, not a learned policy. TokenPacker and DocOwl2 implement coarse-to-fine condensation: progressively merging tokens into compact representations (irreversible and question-agnostic). Doc-REFRAG instead implements learned restoration: it first compresses all tokens into chunk embeddings via an MLP projector, then an RLtrained selector dynamically identifies and restores original fine-grained tokens for question-relevant chunks. The restoration decision is learned endto-end against task reward, not computed by a heuristic. This makes Doc-REFRAG’s approach fundamentally distinct from both the condensation paradigm (TokenPacker, DocOwl2) and the training-free retrieval paradigm (SparseVILA).

## B Source Datasets in DocLongRAG Corpus

DocLongRAG aggregates English data from the following 15 open-source datasets: (Van Landeghem et al., 2023; Masry et al., 2022; Mathew et al., 2021; Ye et al., 2023; Mathew et al., 2022; Tanaka et al., 2021, 2023; Zhang et al., 2023; Laurençon et al., 2024; Liu et al., 2024b; Chen et al., 2023; Yang, 2023; Zhang et al., 2024a; Bai et al., 2024; Zhang et al., 2024b).

## C Evaluation of Hard Negative Relevance

Limitations about the annotation. Here, we present an example from the benchmarks to illustrate their limitations regarding the relevance annotation, which emphasizes the importance of negative samples exclusion.

When the query is [From this report, which subgroup among Hispanics has gained most confidence from 2008 to 2015?], the original benchmark MMDocIR designates Fig. 5a as the exclusive retrieval target, as it provides a bar chart comparing financial optimism across various Latino subgroups from 2008 to 2015. The 08-15 column highlights the percentage-point increase in optimism for each group, with the “Some college or more” subgroup exhibiting the largest change (+20 percentage points), reflecting the greatest gain in confidence during this period.

<table><tr><td>Method</td><td>Query-conditioned</td><td>Dynamic Budget</td><td>RL-trained Selector</td><td>RAG Noise Robust.</td><td>Learned Restoration</td><td>Multi-image RAG</td></tr><tr><td>MQT-LLaVA (Hu et al., 2024c)</td><td>x</td><td>√</td><td>X</td><td>x</td><td>X</td><td>x</td></tr><tr><td>EPIC (Wen et al., 2025b)</td><td>x</td><td>x</td><td>X</td><td>x</td><td>X</td><td>x</td></tr><tr><td>LLaVA-Mini (Zhang et al., 2025b)</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>TokenPacker (Li et al., 2025)</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td><td>X</td></tr><tr><td>DocOwl2 (Hu et al., 2024b)</td><td>x</td><td>X</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>SparseVILA (Khaki et al., 2025)</td><td>√</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>IVTP (Huang et al., 2024)</td><td></td><td>x</td><td>x</td><td>X</td><td>x</td><td>x</td></tr><tr><td>LLaMA-VID (Li et al., 2024c)</td><td></td><td>V</td><td>x</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Doc-REFRAG (Ours)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 8: Comparison of Doc-REFRAG with related visual token compression methods across six design dimensions. Query-conditioned: selection conditioned on the input question. Learned Restoration: selective restoration of compressed tokens via a learned policy (as opposed to training-free heuristic retrieval).

However, Fig. 5b also constitutes a valid retrieval target. By explicitly referencing 2008 and noting that this year was “7 years ago,” the document confirms that the data summarizes trends from 2008 to 2015. Furthermore, it specifies that economic optimism among Latinos with some college education increased by +20 percentage points—roughly twice the growth observed among those with only a high school diploma (+9) or less education (+11). This reinforces that the “Some college or more” subgroup experienced the most substantial rise in confidence. Thus, the pair from the raw benchmark overlooks an equally relevant candidate, which hinders the fair evaluation for retrieval systems.

Human validation about the hard negative relevance. To rigorously validate the semantic relevance of non-annotated images and inform the design of hard negative sampling in DocLongRAG, we conduct a controlled human evaluation study.

We randomly select 500 questions from the curated corpus C, each associated with a set of document images retrieved by a document-aware retriever. For each query, we examine the top-20 retrieved images that do not belong to the groundtruth relevant set D. Specifically, we analyze images at retrieval ranks 1 through 20 to quantify how semantic relevance decays with decreasing rank.

Two expert annotators participate in the evaluation. Given a query q, its ground-truth answer a, and a candidate image d, each annotator rates the degree to which d contains content that supports answering q. Relevance is assessed on a 5-point Likert scale defined as follows:

• 5: The image contains explicit and sufficient evidence to answer the question.

• 4: The image provides partial or indirect evidence that strongly supports the answer.

• 3: The image is topically related but lacks direct evidential value.

• 2: The image shares superficial similarity (e.g., same document type or layout) but is not helpful for answering.

• 1: The image is clearly irrelevant.

Annotators work independently, and interannotator agreement is measured using Krippendorff’s alpha, yielding a value of 0.78, which indicates substantial agreement.

The average relevance scores exhibit a clear decay trend across retrieval ranks: images at rank 1, 2, and 3 receive average scores of 4.62, 4.31, and 3.94, respectively, indicating that they frequently contain meaningful supporting evidence. At rank 4 and onward, the average score remains lower than 3.27, suggesting limited relevance. This pattern confirms that images beyond rank 3 are substantially less likely to provide evidential support for the correct answer. Even so, the relevance at rank 4 and beyond is not zero, so a small fraction of hard negatives may still contain answer evidence. This does not corrupt the training objective: the annotated relevant image is always retained in the sequence by construction, so the supervision remains welldefined, and such cases act as additional retrieval noise rather than false supervision.

## D Placing Meta-Chunk at Different Positions

We explore three different strategies for combining meta-chunk and hard negatives, which are represented by the following descriptions:

1. Head: Placing the meta-chunk at the beginning of the retrieved hard negatives.

2. Tail: Placing the meta-chunk at the end of the retrieved hard negatives.

![](images/34bf088e10eebb0decda877ab0f530d2034311a8a8edca39726d70038f25bf48.jpg)  
Figure 5: Example for Benchmark.

## 3. Random: Randomly inserting the metachunk within the retrieved hard negatives.

Tab. 9 shows that placing the meta-chunk at the beginning (Head) yields better performance. Placing relevant images before hard negatives establishes longer dependencies between the questionrelevant content and the decoder, which enhances overall reasoning effectiveness.

Training-time ordering and inference-time robustness. We investigate whether the headposition advantage (Tab. 9) introduces a positional bias at inference time, causing the decoder to exploit token position rather than semantic content.

We emphasize that the meta-chunk ordering strategy studied here is a training-time design choice for CPT, while at inference time the RLbased selector determines which chunks are expanded based purely on content relevance to the question—not position. Specifically, the selector operates on chunk embeddings from all retrieved images jointly and assigns expansion decisions $\delta _ { i } ^ { j } \in \{ 0 , 1 \}$ via a learned policy π<sub>θ</sub> trained with GRPO rewards on answer accuracy. The input to the selector is a set of chunk-level semantic embeddings; the reward signal is answer accuracy, which is invariant to the physical ordering of images in the context. As a consequence, the training objective provides no gradient signal that encourages position-dependent selection behavior.

This design ensures that the head-position benefit observed in Tab. 9 stems from improved longrange dependency modeling during training—not from a spurious positional shortcut exploited at inference. The meta-chunk always appears at the head of the training sequence, but at inference time the selector re-evaluates every chunk independently and can assign high expansion probability to any image regardless of its position in the retrieved list.

## E System Components: Architecture, Training, and Inference

To clarify the roles of Doc-REFRAG’s components across three conceptually distinct phases, we provide a consolidated overview in Tab. 10. Each component is associated with a specific phase (Architecture, Training, or Inference) and a brief description of its role, with exact hyperparameters for cross-referencing with Secs. 3–5.

<table><tr><td>Model</td><td>ChartQA</td><td>SlideVQA</td><td>InfoVQA</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>#Acc.</td></tr><tr><td>Head</td><td>61.2</td><td>60.8</td><td>58.7</td><td>53.2</td><td>57.3</td><td>54.9</td><td>57.5</td></tr><tr><td>Tail</td><td>50.1</td><td>52.6</td><td>50.7</td><td>40.8</td><td>40.7</td><td>44.9</td><td>46.6</td></tr><tr><td>Random</td><td>58.5</td><td>54.1</td><td>53.5</td><td>45.4</td><td>42.8</td><td>53.0</td><td>51.2</td></tr></table>

Table 9: Performance Comparison of Different Insertion Strategies.
<table><tr><td>Component</td><td>Phase</td><td>Description</td><td>Details</td></tr><tr><td>Vision Encoder</td><td>Architecture</td><td>Encodes each image into 324 visual tokens</td><td>Inherited from DocOwl2; frozen after Stage 1</td></tr><tr><td>MLP Projector (φ)</td><td>Architecture</td><td>Maps k visual tokens to one chunk embedding c3  $\in \mathbb { R } ^ { d }$ </td><td>2-layer MLP, hidden dim 4096; trained in Stage 1</td></tr><tr><td>Decoder (LLM)</td><td>Architecture</td><td>Autoregressively generates the answer from the reconstructed token sequence</td><td>7B decoder with LoRA fine-tuned in Stage 2</td></tr><tr><td>RL Selector (πθ)</td><td>Architecture</td><td>Predicts binary expansion decisions δ for each chunk</td><td>2-layer Transformer, hidden dim 256, 8 heads; &lt; 1M params</td></tr><tr><td>Stage 1: Single-Image Reconstruction</td><td>Training</td><td>Aligns MLP projector with decoder embedding space</td><td>DocStruct4M; decoder frozen; lr 2 × 10−4</td></tr><tr><td>Stage 2: Multi-image CPT</td><td>Training</td><td>Scales to multi-image hybrid sequences with pseudo-label expansion</td><td>OpenDocVQA + Doc-750K + DocLongRAG; LoRA rank 64; lr 1 × 10−5</td></tr><tr><td>Stage 3: RL Selector Training</td><td>Training</td><td>Trains selector to maximize answer accuracy via GRPO</td><td>DocLongRAG; GRPO + clipped PPO (clip 0.2); reward = decoder accuracy</td></tr><tr><td>Chunk Encoding</td><td>Inference</td><td>Compresses each image to L = m/k chunk embeddings</td><td>m = 324 tokens/image, k = 3, L = 108 chunks/image</td></tr><tr><td>Selector Forward Pass</td><td>Inference</td><td>Single forward pass to predict which chunks to expand; budget p = 0.2</td><td>Expands top T = [0.2 · n · L] chunks; latency 3.7 s vs. 9.3 s (ColQwen)</td></tr><tr><td>Chunk Expansion</td><td>Inference</td><td>Selected chunks restored to original k visual tokens before decoding</td><td>Enables fine-grained detail recovery for question-relevant regions</td></tr></table>

Table 10: System overview of Doc-REFRAG, organized by architectural components, training stages, and inferencetime operations.

## F Geometric Mapping Between Coarse and Fine Patch Grids

To utilize question-to-patch similarity scores from ColQwen as pseudo-labels in CPT, we establish a deterministic geometric mapping between the coarse spatial grid used by our encoder and the fine-grained patch grid employed by ColQwen.

Both models discretize the input images into spatial patches using normalized image coordinates, where each pixel location is represented as $( x , y ) \in [ 0 , 1 ] ^ { 2 }$ . Let $G _ { c }$ denote the coarse grid used by CPT, with patch size $\begin{array} { r } { s _ { c } = \frac { 1 } { H _ { c } } \times \frac { 1 } { W _ { c } } } \end{array}$ , where $H _ { c }$ and $W _ { c }$ are the number of horizontal and vertical divisions. Similarly, let $G _ { f }$ denote the fine grid used by ColQwen, with patch size $\begin{array} { r } { s _ { f } = \frac { 1 } { H _ { f } } \times \frac { 1 } { W _ { f } } } \end{array}$ where $H _ { f } > H _ { c }$ and $W _ { f } > W _ { c }$

For any coarse patch $p _ { c } \in G _ { c } ,$ , its spatial extent is defined by the bounding box:

$$
p _ { c } = \left[ i \cdot s _ { c } , ( i + 1 ) \cdot s _ { c } \right] \times \left[ j \cdot s _ { c } , ( j + 1 ) \cdot s _ { c } \right] ,
$$

where (i, j) indexes the patch in the coarse grid.

Each such coarse patch $p _ { c }$ covers a subset of fine patches in $G _ { f }$ . Specifically, the set of fine patches contained within $p _ { c }$ is given by:

$$
{ \mathcal { F } } ( p _ { c } ) = \{ p _ { f } \in G _ { f } \mid { \mathsf { b b o x } } ( p _ { f } ) \subseteq { \mathsf { b b o x } } ( p _ { c } ) \} .
$$

Due to the alignment of coordinate systems and uniform sampling, this containment relationship is fully deterministic. We compute $\mathcal { F } ( p _ { c } )$ by iterating over all fine patches and checking whether their bounding boxes lie entirely within $p _ { c }$ . In practice, this can be implemented via integer arithmetic based on the ratio $\begin{array} { r } { r = { \frac { H _ { f } } { H _ { c } } } = { \frac { W _ { f } } { W _ { c } } } } \end{array}$ , which must be an integer for perfect alignment.

This mapping allows us to aggregate the similarity scores from the fine patches in $\mathcal { F } ( p _ { c } )$ to generate a pseudo-label $\delta _ { i } ^ { j }$ for the importance of coarse chunk i in question $j$ . The aggregation is performed as:

$$
\delta _ { i } ^ { j } = \frac { 1 } { \vert \mathcal { F } ( p _ { c } ) \vert } \sum _ { p _ { f } \in \mathcal { F } ( p _ { c } ) } \sin ( q _ { j } , p _ { f } ) ,
$$

where sim $( q , p _ { f } )$ denotes the question-to-patch similarity score provided by ColQwen.

This deterministic and spatially aligned mapping ensures that the pseudo-labels are geometrically consistent and preserve local structure, enabling effective supervision of chunk selection during training.

## G Robustness Analysis of Pseudo-Labels

Since ground-truth chunk-level relevance labels are unavailable in our setting, we conduct a human evaluation study to assess the validity of the binary pseudo-labels $\delta _ { i } ^ { j } \in \{ 0 , 1 \}$ generated in App. F.

We randomly sample 200 question–document pairs from DocLongRAG. For each document, we compute $\delta _ { i } ^ { j }$ from the aggregated ColQwen similarity scores. Two expert annotators, familiar with document understanding tasks, independently label each chunk as relevant or irrelevant based on whether it contains content that directly or indirectly supports answering the given question. The inter-annotator agreement (Cohen’s κ) is 0.79, indicating substantial agreement.

We compare the pseudo-labels against the human-annotated labels. The pseudo-labels achieve an accuracy of 86.4%, with a precision of 84.1%, recall of 80.3%, and F1 score of 0.82. These results demonstrate that the pseudo-labels align well with human judgments of chunk relevance, providing strong empirical justification for their use as supervision signals in training.

## H Analysis of Chunk Size k

In this section, we provide a detailed comparison of the performance and efficiency of Doc-REFRAG under different chunk sizes $k \in \{ 2 , 3 , 6 \}$ . As discussed in the main paper, the hyperparameter $k$ determines the granularity of visual token compression, where the original visual tokens of each image are partitioned into non-overlapping chunks of size $k .$

Tab. 11 presents the quantitative results across six document-level benchmarks. Our experiments reveal that $k = 3$ achieves the optimal balance between semantic fidelity and computational efficiency, yielding the highest average accuracy of 57.5%. Specifically, compared to $k = 2$ , the $k = 3$ configuration improves average accuracy by 2.0% while maintaining comparable inference latency. This suggests that a slightly coarser compression $( k \ : = \ : 3 )$ effectively filters out redundant visual noise without sacrificing critical details, whereas $k = 2$ may retain excessive irrelevant tokens that hinder the selector’s decision-making.

On the other hand, increasing the chunk size to $k = 6$ significantly reduces the number of visual tokens (from ∼194 to ∼108 per image) and lowers the Time-To-First-Token (TTFT) latency from 2.57s to 1.95s. However, this comes at the cost of a noticeable performance drop, with average accuracy decreasing by 5.2% compared to $k = 3$ . The degradation is particularly evident in data-intensive tasks like ChartQA and InfoVQA, indicating that overly aggressive compression $( k = 6 )$ compromises the preservation of fine-grained textual and structural details essential for complex reasoning.

Based on these findings, we adopt $k = 3$ as the default setting for Doc-REFRAG in our main evaluations, as it delivers state-of-the-art accuracy while ensuring efficient inference suitable for practical RAG applications.

## I Optimal Chunk Expansion Budget

The budget parameter $p \in ( 0 , 1 ]$ controls the number of chunks selected for expansion, defined as $T = \lfloor p \cdot n \cdot L \rfloor$ , where $n \cdot L$ is the total number of chunks. During training, p is uniformly sampled from the interval [0.1, 0.3] to encourage exploration and prevent overfitting to a fixed budget. To determine the optimal value of $p$ during inference, we conduct a hyperparameter search on the six benchmarks.

Specifically, we primarily fix the $p$ to values in the list {0.1, 0.15, 0.2, 0.25, 0.3} and measure the answer accuracy of Doc-REFRAG when processing the expanded input sequence generated by the selector. As shown in Tab. 12, $p = 0 . 2$ achieves the highest accuracy (57.5%) while maintaining low inference latency. Values below 0.2 provide insufficient context, compromising accuracy, whereas values above 0.2 incur higher computational costs without yielding significant improvements. This demonstrates that $p \ = \ 0 . 2$ achieves an optimal balance between performance and efficiency.

Comparison with ColQwen across the full budget range. Tab. 12 also enables a direct accuracy– latency comparison against ColQwen, which serves as an upper-bound selector oracle in our framework (Sec. 5.2). ColQwen achieves 53.6% average accuracy at a latency of 9.3 s. Our RL-based selector at $p = 0 . 2$ surpasses ColQwen with 57.5% accuracy while requiring only 3.7 s of selection latency—a 2.5× reduction. At $p = 0 . 2 5$ the selector reaches 56.4% in 2.62 s, and at $p = 0 . 3$ it reaches 53.7% in 2.84 s, all remaining well below ColQwen’s latency budget. Conversely, even at $p = 0 . 1$ (2.19 s, 51.6%), the selector is substantially faster than ColQwen while sacrificing only 2.0 percentage points of accuracy. Across the entire search range $p \in \{ 0 . 1 , \ldots , 0 . 3 \}$ , the RL selector consistently occupies a more favorable accuracy– latency operating point than ColQwen, confirming that replacing the dense retriever with a lightweight learned selector does not require trading accuracy for speed.

## J Training Procedure for the Selector

We train the chunk selection policy using the GRPO framework. The full training procedure is outlined below.

Policy Network Architecture. The selector is a lightweight two-layer transformer encoder with hidden dimension 256 and 8 attention heads. Given a question embedding t and a sequence of chunk embeddings $\mathbf { c } _ { i } ^ { j }$ , the policy network computes logits over each chunk:

<table><tr><td>Model</td><td>Size</td><td>TokenV</td><td>ChartQA</td><td>SlideVQA</td><td>InfoVQA</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>#Acc.</td><td>#TTFT</td><td>#TTIT</td></tr><tr><td>Doc-REFRAGk=2</td><td>7B</td><td>~194</td><td>58.6</td><td>56.2</td><td>56.4</td><td>51.9</td><td>55.1</td><td>53.7</td><td>55.5</td><td>3.24</td><td>0.07</td></tr><tr><td>Doc-REFRAGk=3</td><td>7B</td><td>~150</td><td>61.2</td><td>60.8</td><td>58.7</td><td>53.2</td><td>57.3</td><td>54.9</td><td>57.5</td><td>2.57</td><td>0.07</td></tr><tr><td> $\mathrm { D o c - R E F R A G } _ { k = 6 }$ </td><td>7B</td><td>~108</td><td>53.2</td><td>54.8</td><td>53.9</td><td>48.5</td><td>52.4</td><td>52.2</td><td>52.3</td><td>1.95</td><td>0.07</td></tr></table>

Table 11: Comparison of Doc-REFRAG performance with different chunk sizes k. $k = 3$ achieves the best accuracy–efficiency trade-off. TTFT denotes Time-To-First-Token.

<table><tr><td>p</td><td>Accuracy (%)</td><td>TTFT (s)</td><td>TTIT (s)</td></tr><tr><td>0.1</td><td>51.6</td><td>2.19</td><td>0.07</td></tr><tr><td>0.15</td><td>54.2</td><td>2.34</td><td>0.07</td></tr><tr><td>0.2</td><td>57.5</td><td>2.57</td><td>0.07</td></tr><tr><td>0.25</td><td>56.4</td><td>2.62</td><td>0.07</td></tr><tr><td>0.3</td><td>53.7</td><td>2.84</td><td>0.07</td></tr></table>

Table 12: Answer accuracy under different chunk expansion budgets $p .$

$$
\delta _ { i } ^ { j } = \mathbf { W } _ { o } \cdot \mathbf { M L P } \left( \operatorname { T r a n s f o r m e r } ( \mathbf { t } , \mathbf { c } _ { i } ^ { j } ) \right) ,
$$

where ${ \bf W } _ { o }$ is a learnable output weight matrix. The policy $\pi _ { \theta }$ samples binary decisions $\delta _ { i } ^ { j } \in \{ 0 , 1 \}$ independently for each chunk.

Group-Wise Relative Advantage Estimation. For each question-document pair, we sample G = 8 chunk selection policies $\hat { \{ \delta ^ { ( 1 ) } , \delta ^ { ( 2 ) } , \dots , \delta ^ { ( G ) } \} }$ from $\pi _ { \theta }$ . For each policy, we reconstruct the input sequence and compute the reward $r ^ { ( g ) }$ using the fixed decoder from the CPT stage. The relative advantage is defined as:

$$
A _ { i } = r ^ { ( i ) } - \frac { 1 } { G } \sum _ { g = 1 } ^ { G } r ^ { ( g ) } ,
$$

which measures the performance of the i-th policy relative to the group mean.

Policy Update with Clip and KL Regularization. The policy is updated using the following objective:

$$
\begin{array} { r l } & { \mathcal { L } = \mathbb { E } _ { \delta \sim \pi _ { \theta } } \Big [ \operatorname* { m i n } \Big ( \rho ( \delta ) A ( \delta ) , } \\ & { \qquad \mathrm { c l i p } ( \rho ( \delta ) , 1 - \epsilon , 1 + \epsilon ) A ( \delta ) \Big ) \Big ] } \\ & { \qquad - \beta \cdot D _ { \mathrm { K L } } \big ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } \big ) , } \end{array}\tag{1}
$$

where $\rho ( \delta ) = \pi _ { \theta } ( \delta ) / \pi _ { \mathrm { r e f } } ( \delta )$ is the importance sampling ratio, $\epsilon = 0 . 2 , \beta = 0 . 1$ , and $\pi _ { \mathrm { r e f } }$ is the reference model initialized with SFT weights.

## K Training Cost Analysis

The four components are not trained separately at full cost: only Stage 2 carries a non-trivial cost, while Stages 1 and 3 are lightweight. We provide a stage-by-stage breakdown of training configuration across the three stages in Tab. 13. All experiments are conducted on 8× NVIDIA RTX 3090 GPUs (24 GB VRAM each). The base model is DocOwl2 (7B parameters).

<table><tr><td>Stage</td><td>Dataset</td><td>Trainable Parameters</td><td>Epochs</td></tr><tr><td>Stage 1: Single-Image Recon.</td><td>DocStruct4M</td><td>Encoder align. + MLP projector</td><td>2</td></tr><tr><td>Stage 2: Multi-image CPT</td><td>OpenDocVQA + Doc-750K + DocLongRAG</td><td>LoRA (rank 64) on 7B decoder</td><td>2</td></tr><tr><td>Stage 3: RL Selector</td><td>DocLongRAG</td><td>Selector (&lt; 1M params)</td><td>2</td></tr></table>

Table 13: Training cost summary for Doc-REFRAG, broken down by stage, dataset, and trainable parameters.

Stage 1 is computationally lightweight: the decoder is fully frozen and only the linear alignment module in the vision encoder and the two-layer MLP projector are updated. The small number of trainable parameters and single-image inputs (DocStruct4M (Hu et al., 2024a)) make this stage converge quickly.

Stage 2 constitutes the dominant cost. Training on the combined corpus of OpenDocVQA (Tanaka et al., 2025), Doc-750K, and DocLongRAG— totaling over 1.1M question–image pairs—requires processing substantially longer multi-image hybrid sequences. Despite using LoRA (rank=64) to reduce the number of trainable parameters in the 7B decoder, the dataset scale and extended context lengths make this stage the bottleneck of the overall pipeline.

Stage 3 is again lightweight because the selector itself is a tiny two-layer transformer (<1M parameters); the 7B decoder is frozen and used only for reward evaluation on DocLongRAG with GRPO group size $G = 8 .$ The modest parameter count and single-dataset scope ensure this stage adds minimal overhead relative to Stage 2.

Overall, the use of LoRA in Stage 2 and a lightweight selector in Stage 3 keeps the total training footprint manageable on commodity RTX 3090 hardware, making Doc-REFRAG accessible without specialized data-center resources.

## L Qualitative Analysis

To better understand the failure modes of Doc-REFRAG, we conduct a qualitative analysis on 100 incorrectly answered questions, categorizing the source of errors based on the type of information required for correct inference. As shown in Fig. 6, the majority of errors (64%) stem from textual content that is dense or semantically complex, where critical details are embedded across multiple paragraphs and require fine-grained reasoning.

In contrast, 22% of failures involve figures, charts, or diagrams—visual elements that may not be fully reconstructed due to selective chunking, leading to incomplete interpretation. Another 14% arise from structured data such as tables or lists, which often contain precise numerical or relational information that can be lost during compression if key rows or columns are pruned.

These results suggest that Doc-REFRAG handles unstructured textual content effectively, with the majority of failures (64%) concentrated in the most challenging regime: dense or semantically complex text requiring fine-grained multiparagraph reasoning. Visual and structured content together account for the remaining 36% of errors, reflecting the inherent trade-off between compression efficiency and fidelity when selective chunking omits visually or numerically critical regions.

![](images/6e366deaaa994c7e5db8d4f1b028c14a683a8dd6b677d49634f1b4822eb29e09.jpg)  
Figure 6: Qualitative analysis of the modalities that predominantly contribute to incorrect answers in Doc-REFRAG.

## M Empirical Efficiency Validation

To quantitatively assess the efficiency gains of Doc-REFRAG, we define the acceleration ratio as:

$$
{ \mathrm { A c c e l e r a t i o n R a t i o } } = { \frac { \mathrm { L a t e n c y } _ { \mathrm { b a s e l i n e } } } { \mathrm { L a t e n c y } _ { \mathrm { o u r s } } } } ,
$$

where latency is measured in terms of time-to-firsttoken (TTFT) and time-to-inter-token (TTIT) under a batch size of 1, with N retrieved images conditioned on each question.

As shown in Fig. 7, the TTFT acceleration ratio increases monotonically with N, peaking at 2.46× when $N = 5 0$ . This reflects the effectiveness of our question-guided chunk selection in reducing visual tokens, thereby accelerating initial attention computation. In contrast, TTIT acceleration grows gradually (max 1.12×), indicating that post-initial-token processing still involves selected chunks. Moreover, Doc-REFRAG achieves an average accuracy of 50.2% with 50 retrieved images, substantially outperforming DocOwl2, which attains only 37.2%.

![](images/5b7085369501219a99ec7610c60dc1dc482d22fb77b49ed72ea8cdc437a0ea5d.jpg)  
Figure 7: Acceleration ratio of Doc-REFRAG versus DocOwl2 in terms of TTFT (blue) and TTIT (green), as a function of the number of retrieved images. The red dashed line indicates the baseline (DocOwl2 Latency = 1×).

## N Disentangling Architectural and Data Contributions

We next analyze the independent and joint contributions of architectural design and training data to Doc-REFRAG’s overall performance. To this end, we construct a $2 \times 2$ factorial analysis using results already reported in Tab. 5, varying both architecture (DocOwl2 vs. Doc-REFRAG) and training data (without vs. with DocLongRAG).

Tab. 14 reveals three complementary sources of improvement.

<table><tr><td></td><td>w/o DocLongRAG</td><td>w/ DocLongRAG</td></tr><tr><td>DocOwl2 (base architecture)</td><td>45.8</td><td>49.7</td></tr><tr><td>Doc-REFRAG (our architecture)</td><td>48.9</td><td>55.1</td></tr><tr><td>Data gain  $\left( \Delta _ { \mathrm { d a t a } } \right)$ </td><td> $4 9 . 7 - 4 5 . 8 = + 3 . 9 \%$ </td><td></td></tr><tr><td>Architect  $\mathrm { u r e } \ g a i n \left( \Delta _ { \mathrm { a r c h } } \right)$ </td><td> $4 8 . 9 - 4 5 . 8 = + 3 . 1 \%$ </td><td></td></tr><tr><td>Synergy  $( \Delta _ { \mathrm { s y n } } )$ </td><td> $5 5 . 1 - 4 9 . 7 - 3 . 1 = + 2 . 3 \%$ </td><td></td></tr><tr><td>Total gain</td><td> $5 5 . 1 - 4 5 . 8 = + 9 . 3 \%$ </td><td></td></tr></table>

Table 14: 2×2 factorial decomposition of average accuracy (%) across DUDE, Vidore, and MMDocIR. Base: DocOwl2 (orig.). $\Delta _ { \mathrm { { d a t a } } } \mathrm { { : } }$ gain from DocLongRAG alone. $\Delta _ { \mathrm { a r c h } } \mathrm { : }$ gain from Doc-REFRAG architecture alone. $\Delta _ { \mathrm { { s y n } } } \colon$ additional gain from their interaction.

Data contribution (+3.9%). Fine-tuning the original DocOwl2 on DocLongRAG alone raises average accuracy from 45.8% to 49.7%. This is consistent with the finding in Sec. 5.3 that DocLongRAG provides a broadly beneficial training signal for models across different architectures.

Architecture contribution (+3.1%). Even without DocLongRAG, replacing DocOwl2’s encoding strategy with Doc-REFRAG’s chunk compression and RL-based selector raises accuracy from 45.8% to 48.9%. This demonstrates that the architectural design itself provides meaningful gains that are independent of the new training data.

Synergistic interaction (+2.3%). The full system (Doc-REFRAG trained on DocLongRAG) achieves 55.1%, exceeding the sum of the two individual effects by 2.3 percentage points. This superadditive interaction arises because Doc-REFRAG’s question-guided compression makes the model better able to exploit the noisy-retrieval training signal present in DocLongRAG: the RL selector learns to identify relevant chunks from heterogeneous retrieved images, a skill that DocLongRAG specifically exercises.

In summary, both contributions are independently meaningful and mutually reinforcing. The architectural gains cannot be explained by data alone: even when DocOwl2 is fine-tuned on DocLongRAG (49.7%), it remains 5.4 percentage points below Doc-REFRAG Full (55.1%), demonstrating the independent value of our architectural design. Conversely, the data gains cannot substitute for the architecture: Doc-REFRAG without Doc-LongRAG (48.9%) still lags behind the full system (55.1%), confirming that the dataset and the architecture are complementary rather than redundant.

## O Validating the DocLongRAG Construction Pipeline

We investigate whether the performance improvements attributed to DocLongRAG stem from the carefully engineered construction pipeline (hard negative mining and quality filtering) or simply from exposing the model to a larger number of images during training. To isolate these factors, we design a naive random baseline that uses the identical model architecture but replaces the curated DocLongRAG training data with randomly assembled training instances.

Specifically, for this baseline—denoted Doc-REFRAG (naive random)—we draw questions at random from Doc-750K and OpenDocVQA, and pair each question with approximately 37 randomly sampled images (matching the average input length in DocLongRAG), without performing any hard negative mining or quality filtering. All other training hyperparameters and architectural choices are held constant.

<table><tr><td>Model</td><td>DUDE</td><td>Vidore</td><td>MMDocIR</td><td>Average</td></tr><tr><td>DocOwl2 (orig.)</td><td>44.7</td><td>47.5</td><td>45.2</td><td>45.8</td></tr><tr><td>Doc-REFRAG (naive random)</td><td>34.2</td><td>36.1</td><td>35.8</td><td>35.4</td></tr><tr><td>Doc-REFRAG (Full)</td><td>53.2</td><td>57.3</td><td>54.9</td><td>55.1</td></tr></table>

Table 15: Comparison of Doc-REFRAG trained on randomly assembled data (naive random) versus the full DocLongRAG pipeline. The naive random baseline uses the same architecture but pairs each question with ∼37 randomly selected images, without hard negative mining or quality filtering. Results are reported as average accuracy (% ↑) on DUDE, Vidore, and MMDocIR. See also Tab. 5.

Tab. 15 yields three clear findings. First, Doc-REFRAG (naive random) achieves only 35.4% average accuracy—10.4 percentage points below DocOwl2 (orig.) at 45.8%—demonstrating that simply exposing the model to more images is not only unhelpful but actively harmful: random image pairings introduce retrieval noise that disrupts the model’s reasoning without providing a useful training signal. Second, the full DocLongRAG pipeline raises average accuracy from the DocOwl2 baseline of 45.8% to 55.1%, a gain of 9.3 percentage points, confirming that the hard negative mining and quality filtering steps are the true source of improvement. Third, the gap between Doc-REFRAG (Full) and Doc-REFRAG (naive random) is 19.7 percentage points (55.1% − 35.4%), clearly isolating the contribution of the pipeline design from the contribution of raw image quantity. Together, these results validate that the DocLongRAG construction pipeline itself, not mere data volume, is responsible for the observed performance gains.