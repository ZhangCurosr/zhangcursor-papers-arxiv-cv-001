# Autoregressive Retriever: Improving Query Understanding from Item Feedback for Universal Multimodal Retrieval

Jianfei Zhao<sup>1,2</sup>, Yifan Wang<sup>4</sup>, Feng Zhang<sup>1</sup>, Xin Sun<sup>1</sup>, Chong Feng<sup>1,5</sup>, Zhixing Tan<sup>3</sup>, Yang Luo<sup>4</sup>, Boyuan Pan<sup>4</sup>, Xu Kai<sup>4</sup>, Yao Hu<sup>4</sup>

<sup>1</sup>School of Computer Science and Technology, Beijing Institute of Technology, <sup>2</sup>Zhongguancun Academy, <sup>3</sup>Zhongguancun Laboratory <sup>4</sup>Xiaohongshu   
<sup>5</sup>Southeast Academy of Information Technology, Beijing Institute of Technology, zhqingan@bit.edu.cn

## Abstract

Universal multimodal retrieval typically encodes a query once and ranks independently indexed items by embedding similarity. This design supports efficient search, but leaves the query representation unchanged even when retrieved items could help clarify the information need. We introduce the AutoRegressive Retriever (ARR), a multimodal retrieval model that learns both to select informative items and to use their content to refine subsequent retrieval. ARR alternates between retrieving an item and updating the query embedding, then uses the final embedding to rank the collection. Supervised fine-tuning teaches the encoder to use feedback through stepwise contrastive supervision. Reinforcement learning treats feedback items as actions and optimizes their selection using the final reciprocal rank of a relevant item. A query-side adapter enables this optimization against a fixed item index. ARR demonstrates strong retrieval performance on both in-domain and zero-shot benchmarks, outperforming the compared baselines on average. Further analyses show that feedback improves retrieval at inference time and that training with feedback also improves the initial query embedding, before any item is observed.

## 1 Introduction

Retrieval systems increasingly need to search across text, images, and their combinations, supporting a wide range of applications, including retrieval-augmented generation (Sun et al., 2026a; Zhang et al., 2026c), multimedia platforms (Zhao et al., 2026), and e-commerce search (Zhang et al., 2026b; Sun et al., 2026a). Universal multimodal retrieval (UMR) addresses this need with a single model that maps heterogeneous queries and items into a shared embedding space (Wei et al., 2024; Liu et al., 2025; Lin et al., 2025). In the standard embedding-based approach, queries and items are encoded independently and matched by similarity. This separation enables item embeddings to be computed in advance and reused across queries, making large collections searchable through a precomputed index.

![](images/4d2f5fbf3435c44d1c5b5670f4c4a29629ff091217130921a3aa4d56727ee738.jpg)  
Figure 1: Autoregressive retrieval with ARR. Retrieved items provide context for successive query embeddings; the final embedding ranks the collection.

However, this separation also limits a query’s ability to leverage the collection being searched. A query may omit important details, express its intent only implicitly, or contain visual and textual cues of unequal importance. An independently computed query embedding must resolve such ambiguities before observing any candidate items. Although contrastive training (He et al., 2020; Chen et al., 2020) learns to align queries with relevant items across the training data, at inference time the query representation remains fixed and cannot adapt to the candidates encountered during retrieval. We refer to this lack of candidate-conditioned adaptation as the query–item gap.

Recent works improve query interpretation through stronger multimodal large language models (MLLMs) (Jiang et al., 2024; Li et al., 2026a), richer representations (Sun et al., 2026b; Wu et al., 2026), and reasoning before embedding (Cui et al., 2026; Zhang et al., 2026a; Hao et al., 2026). These approaches extract more information from the input, but do not by themselves determine how a retriever should respond to newly encountered items. Retrieval feedback offers a complementary direction. Prior methods either directly refine dense query representations using retrieved documents (Yu et al., 2021), or rewrite queries and reencode them (Tu et al., 2026; Bigdeli et al., 2026). Yet, the retrieved items are typically processed by an auxiliary refinement mechanism rather than by the retriever itself. We instead ask: can the retriever learn to use the items it actually encounters to decide how to search better?

We introduce the AutoRegressive Retriever (ARR), a multimodal retrieval model that alternates between retrieving an item and updating the query embedding (Figure 1). We refer to this iterative process as autoregressive retrieval. At each intermediate step, ARR selects an item by embedding similarity, appends its multimodal content to the query history, and encodes the expanded context for the next search. After a fixed number of steps, the final embedding ranks the collection. The process is autoregressive over retrieved items: each selection depends on the original query and earlier feedback. Item embeddings remain independently indexable, preserving the standard embedding-based search interface.

Learning this process requires distinguishing an item’s relevance as an answer from its usefulness as feedback. An item that is not the target may still reveal information that helps retrieve the target later; conversely, a plausible match may distract the encoder. The model receives item content without a relevance label and must learn which evidence to use. We train ARR in two stages to address these challenges. Supervised fine-tuning (SFT) first teaches the encoder to interpret feedback using stepwise contrastive supervision on offline trajectories. Reinforcement learning (RL) then learns feedback selection from trajectories sampled by the current policy, using final retrieval quality as the reward. Its actions are retrieved items rather than generated text tokens. A query-side adapter learns this policy against a fixed item index.

Experiments on M-BEIR and seven zero-shot benchmarks show benefits from both feedbackconditioned inference and feedback-based training. ARR consistently achieves the highest average performance among all compared methods across both benchmarks. Training with feedback also improves first-step retrieval, indicating that its benefits extend to inference without feedback.

Our contributions are threefold:

• We introduce ARR, a multimodal retriever that performs autoregressive retrieval by conditioning query embeddings on a sequence of retrieved items while retaining independently indexed item embeddings.

• We develop a two-stage training procedure that combines supervised feedback interpretation with item-selection policy optimization, linking intermediate retrieval actions to the final ranking outcome.

• We evaluate ARR across in-domain and zeroshot tasks, and use ablation studies to distinguish the benefits of feedback at inference time from improvements learned during training.

## 2 Related Work

Universal multimodal retrieval. UMR brings diverse query–item modality configurations into a shared retrieval model (Wei et al., 2024). Contrastive models such as CLIP (Radford et al., 2021) provide a foundation for cross-modal alignment, while MLLM-based encoders extend this approach to more flexible input combinations (Jiang et al., 2024; Li et al., 2026a). MM-Embed (Lin et al., 2025) uses modality-aware hard-negative mining to improve multimodal retrieval. LamRA (Liu et al., 2025) combines a retrieval model with a separate reranking component trained with pointwise and listwise objectives. BToks (Sun et al., 2026b) and LaME (Wu et al., 2026) investigate bottleneck representations and latent computation for stronger embeddings. These directions improve representation learning or candidate ranking. ARR instead focuses on how retrieved content can inform subsequent query representations and how feedback items should be selected.

Retrieval feedback. Using retrieved results to refine a query is an established idea in information retrieval. ANCE-PRF (Yu et al., 2021) learns a dense query encoder that consumes the original query and top-ranked documents while keeping the document index fixed. More recent approaches use language models to turn retrieval evidence into query reformulations: GPRF (Tu et al., 2026) learns retrieval-oriented rewrites, and ADORE (Bigdeli et al., 2026) iteratively expands queries using assessments of retrieved content. ARR shares the motivation of adapting a query to the collection. Its focus is on learning a sequence of multimodal feedback actions within the embedding retriever: the encoder both consumes retrieved items and defines the policy that selects the next item, with selection trained for its effect on the final ranking.

Reasoning and reinforcement learning for retrieval. Reasoning-enhanced retrieval enriches inputs before constructing embeddings. TTE (Cui et al., 2026) uses generated context to improve multimodal representations, while TWN (Zhang et al., 2026a) and TRACE (Hao et al., 2026) study when to invoke explicit reasoning. Retrieval-oriented query rewriting provides another way to expose information that is implicit in the input (Zhu et al., 2024). RL has also been used to align generation with retrieval outcomes: GRAPE (Zhang et al., 2025) optimizes query rewriting using ranking rewards, and ELVA (Liu et al., 2026) uses ranking and margin rewards to train representations produced after textual synthesis. ARR differs in its action space and feedback loop. It samples item identities directly from embedding similarities, observes the selected items, and optimizes these selections for downstream retrieval quality. Thus, its autoregressive sequence consists of retrieved items rather than a generated reasoning or rewriting trace.

## 3 Preliminaries

Task formulation. Let D denote an item corpus and q a query. Both q and an item $d \in \mathcal { D }$ may contain text, images, or a combination of the two. A query may also specify an instruction describing the desired relevance relation. UMR uses a shared retrieval model across these input configurations to rank items according to their relevance to $q .$ . We write a training example as $( q , d ^ { + } )$ , where $d ^ { + }$ is an annotated relevant item.

MLLM-based encoder. An MLLM-based encoder maps queries and items into a shared embedding space. Text and images are converted into token representations and assembled using an input template. The MLLM processes the resulting sequence, and an embedding readout extracts a fixeddimensional vector from its hidden states. After $\ell _ { 2 }$ normalization, the embeddings and their similarity are

$$
\begin{array} { r } { \mathbf { z } _ { q } = f _ { \theta } ( q ) , \qquad \mathbf { v } _ { d } = f _ { \theta } ( d ) , } \\ { s _ { \theta } ( q , d ) = \mathbf { z } _ { q } ^ { \top } \mathbf { v } _ { d } , \qquad } \end{array}\tag{1}
$$

where $f _ { \theta }$ denotes the encoder and $\mathbf { z } _ { q } , \mathbf { v } _ { d } \in \mathbb { R } ^ { p }$ Their inner product measures cosine similarity in the shared embedding space.

Retrieval objective. The retriever is trained with a query-to-item contrastive objective. For each query $q ,$ let $B _ { q }$ contain its annotated positive $d ^ { + }$ and negative items. We use the InfoNCE loss (He et al., 2020),

$$
\ell _ { \mathrm { N C E } } ( \mathbf { z } , d ^ { + } , \mathcal { B } _ { q } ; \{ \mathbf { v } _ { d } \} ) = - \log \frac { \exp ( \mathbf { z } ^ { \top } \mathbf { v } _ { d ^ { + } } / \tau ) } { \sum _ { d \in \mathcal { B } _ { q } } \exp ( \mathbf { z } ^ { \top } \mathbf { v } _ { d } / \tau ) } ,\tag{2}
$$

where $\tau > 0$ is the temperature. Minimizing this loss increases the positive item’s similarity relative to the negatives in $B _ { q }$

## 4 Method

ARR performs autoregressive retrieval by conditioning query representations on retrieved content while preserving independently encoded item embeddings. We first define this retrieval process (Section 4.1), then describe SFT for interpreting feedback (Section 4.2) and RL for selecting feedback that improves the final ranking (Section 4.3).

## 4.1 Autoregressive Retrieval

ARR uses N query states with N − 1 feedback selections. It first encodes the original query, then repeatedly retrieves an item and appends its content and available metadata to the query context. The encoder processes this context to produce the next query embedding. The $N ^ { \mathrm { t h } }$ embedding determines the final retrieval results. No relevance label is included in the feedback.

Query Template   
SYSTEM: <instruction>   
USER: <query>   
ASSISTANT: <pad>   
USER: <item>   
ASSISTANT: <pad>   
. . .

At step t, the model observes the original query and all previously selected feedback items. Let $a _ { t } ~ \in ~ \mathcal { D }$ denote the item selected at step t. We define the state and query embedding as

$$
\begin{array} { r } { h _ { t } = ( q , a _ { 1 } , \dots , a _ { t - 1 } ) , \quad \mathbf { z } _ { t } = f _ { \theta } \big ( T ( h _ { t } ) \big ) , } \\ { t = 1 , \dots , N , } \end{array}\tag{3}
$$

where $\tau$ serializes the history using the MLLM’s multi-turn conversation template, illustrated on the right. The <pad> token serves as the representation token.

A history mask prevents repeated selection of the same item. Let $\mathcal { C } ( q ) \subseteq \mathcal { D }$ denote the feedback candidate pool for query q, with at least $N - 1$ items, and let $H _ { t } = \{ a _ { 1 } , \ldots , a _ { t - 1 } \}$ be the set of previously selected items. For $t < N$ , embedding similarities define a policy over the remaining candidates:

$$
\begin{array} { r l } & { \pi _ { \boldsymbol { \theta } } ( d \mid h _ { t } ; \mathcal { C } ( q ) ) = } \\ & { \left\{ \begin{array} { l l } { \displaystyle \exp ( \mathbf { z } _ { t } ^ { \top } \mathbf { v } _ { d } / \tau ) } \\ { \displaystyle \sum _ { d ^ { \prime } \in \mathcal { C } ( q ) \backslash H _ { t } } \exp ( \mathbf { z } _ { t } ^ { \top } \mathbf { v } _ { d ^ { \prime } } / \tau ) } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{4}
$$

The probability of a sampled feedback trajectory ${ \boldsymbol { \xi } } = \left( a _ { 1 } , \ldots , a _ { N - 1 } \right)$ factorizes as

$$
P _ { \theta } ( \xi \mid q ; { \mathcal C } ( q ) ) = \prod _ { t = 1 } ^ { N - 1 } \pi _ { \theta } ( a _ { t } \mid h _ { t } ; { \mathcal C } ( q ) ) .\tag{5}
$$

This factorization makes retrieval autoregressive over item identities: earlier selections affect both the next query embedding and subsequent selection probabilities. After incorporating $N - 1$ feedback items, ARR uses the final query representation $\mathbf { z } _ { N }$ to retrieve the top-k items:

$$
\mathrm { R e t r i e v e } _ { k } ( q , \xi ) = \mathrm { T o p } { \ - } { \bf k } _ { d \in { \mathcal { D } } } { \bf z } _ { N } ^ { \top } { \bf v } _ { d } .
$$

The history mask is applied only to feedback selection; previously selected items remain eligible for retrieval at each step.

## 4.2 Supervised Fine-Tuning

SFT teaches the encoder to use retrieved content while preserving the original query intent $( { \mathrm { F i g } } -$ ure 2a). To avoid repeatedly searching the corpus during this stage, we construct an offline feedback pool using the pretrained encoder with parameters $\theta _ { 0 }$

$$
\mathcal { C } _ { \mathrm { S F T } } ( { q } ) = \mathrm { T o p } { - } K _ { d \in \mathcal { D } } f _ { \theta _ { 0 } } ( { q } ) ^ { \top } f _ { \theta _ { 0 } } ( d ) .\tag{6}
$$

Here, $K \geq N - 1$ . During training, we randomly sample an ordered sequence of $N - 1$ items without replacement from this pool and construct the N query states in Eq. 3. We neither inject a missing positive into the feedback pool nor filter feedback by relevance labels. The contrastive set $B _ { q }$ , which contains the annotated positive for supervision, is separate from this feedback pool.

We supervise every query embedding with the relevant item $d ^ { + }$ . Let $\operatorname { s g } ( \cdot )$ denote stop-gradient.

We use the same contrastive set $B _ { q }$ across the states of a trajectory and define

$$
\begin{array} { r } { \ell _ { t } = \ell _ { \mathrm { N C E } } \big ( \mathbf { z } _ { t } , d ^ { + } , \mathcal { B } _ { q } ; \{ \widetilde { \mathbf { v } } _ { d } ^ { ( t ) } \} \big ) , } \\ { \widetilde { \mathbf { v } } _ { d } ^ { ( t ) } = \left\{ \mathbf { v } _ { d } , \qquad t = 1 , \right. } \\ { \left. \qquad \mathbf { v } _ { d } ^ { ( \mathbf { v } _ { d } ) } , \quad t > 1 . \right. } \end{array}\tag{7}
$$

The first-step loss backpropagates through both query and item embeddings. Subsequent losses backpropagate through the query branch only, so feedback-conditioned states do not directly pull standalone item embeddings in different directions.

Retrieved content can clarify a query or distract the encoder. We therefore penalize increases in contrastive loss between consecutive states. For $N > 1$ , the degradation penalty is

$$
\mathcal { L } _ { \mathrm { d e g } } = \frac { 1 } { N - 1 } \sum _ { t = 2 } ^ { N } \left[ \log ( \ell _ { t } + \delta ) \right. \biggr . \biggr .\tag{8}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ and $\delta > 0$ is a numerical stability constant. The log difference measures a relative increase in loss. Taking the positive part at each step prevents an improvement elsewhere from canceling a harmful update. The stop-gradient prevents the reference loss from being directly increased to reduce the penalty. This encourages, but does not guarantee, improved rankings after feedback.

The overall objective is

$$
\mathcal { L } _ { \mathrm { S F T } } = \mathbb { E } _ { ( q , d ^ { + } ) , \xi \sim p _ { \mathrm { o f f } } ( \cdot \vert q ) } \left[ \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \ell _ { t } + \lambda _ { \mathrm { d e g } } \mathcal { L } _ { \mathrm { d e g } } \right] ,\tag{9}
$$

where $p _ { \mathrm { o f f } }$ is the offline trajectory distribution and $\lambda _ { \mathrm { d e g } } \geq 0$ weights the penalty. For $N = 1$ , we set ${ \mathcal { L } } _ { \mathrm { d e g } } = 0$ . SFT supplies relevance supervision at every state without requiring expert labels for which feedback item to select.

## 4.3 Reinforcement Learning for Retrieval

SFT learns to interpret feedback, but does not explicitly optimize which items to select. Its trajectories come from an offline sampling procedure, and its contrastive losses supervise each state rather than assigning credit to actions for their downstream effects. We address these limitations by sampling trajectories from the current retrieval policy and optimizing feedback selection with a terminal ranking reward. Figure 2b illustrates this stage, which adapts group relative policy optimization (GRPO) (Shao et al., 2024) to item-selection actions.

![](images/f0b60a967abfd8aee0cfa61e08f42d3f693f18de4ac085987e090df339000f79.jpg)  
Figure 2: Overview of ARR training. (a) SFT uses offline feedback trajectories, stepwise contrastive supervision, and a degradation penalty. Only the initial-step loss backpropagates through standalone item embeddings. (b) RL freezes the SFT backbone and item index while training a query-side LoRA adapter. Terminal reciprocalrank rewards supervise feedback selection through GRPO, complemented by contrastive supervision of the fina embedding.

Let θ denote the parameters after SFT. We freeze this model and attach an additional query-side lowrank adaptation (LoRA) (Hu et al., 2022) module with trainable parameters $\phi$ . The query and item representations are

$$
\begin{array} { r } { \mathbf { z } _ { t } ^ { \phi } = f _ { \theta , \phi } \big ( \mathcal { T } ( h _ { t } ) \big ) , \qquad \overline { { \mathbf { v } } } _ { d } = f _ { \theta } ( d ) . } \end{array}\tag{10}
$$

The adapter is active only when encoding query contexts, including their feedback items. Standalone item embeddings are computed without it and remain fixed throughout RL. Consequently, optimization is confined to query interpretation and feedback selection within the SFT embedding space, without requiring the corpus index to be rebuilt after each update.

Terminal retrieval reward. The state is the complete history $h _ { t }$ , the action is a feedback item, and the transition appends that item to the history. This ffldefines a finite-horizon Markov decision process with deterministic transitions. Let $\mathcal { C } _ { \mathrm { R L } } ( q )$ denote the item pool for query $q ,$ and let $\{ \overline { { \mathbf { v } } } _ { d } \} _ { q }$ denote the precomputed static item embeddings. For a trajectory $\xi ,$ , the terminal reward is the reciprocal rank of $d ^ { + }$ under the final query embedding:

$$
r _ { \phi } ( q , \xi ) = 1 / \operatorname { r a n k } ( d ^ { + } ; \mathbf { z } _ { N } ^ { \phi } , \{ \overline { { \mathbf { v } } } _ { d } \} _ { q } ) .\tag{11}
$$

Rank is one-based and determined by descending similarity over the entire training pool, including any previously selected items.

GRPO-style optimization. For each query, we sample $M \geq 2$ trajectories $\{ \xi ^ { ( m ) } \} _ { m = 1 } ^ { M }$ from a rollout policy $\pi _ { \phi _ { \mathrm { o l d } } } .$ , using Eq. 4 over $\mathcal { C } _ { \mathrm { R L } } ( q )$ with fixed item vectors. We compute rewards $r _ { m } =$ $r _ { \phi _ { \mathrm { o l d } } } ( q , \xi ^ { ( m ) } )$ for each trajecory.

For an observed state–action pair, the probability ratio is

$$
\rho _ { m , t } ( \phi ) = \frac { \pi _ { \phi } ( a _ { t } ^ { ( m ) } \mid h _ { t } ^ { ( m ) } ; \mathcal { C } _ { \mathrm { R L } } ( q ) ) } { \pi _ { \phi _ { \mathrm { o l d } } } ( a _ { t } ^ { ( m ) } \mid h _ { t } ^ { ( m ) } ; \mathcal { C } _ { \mathrm { R L } } ( q ) ) } .\tag{12}
$$

The objective for one query is

$$
\begin{array} { c } { \displaystyle { J _ { \mathrm { G R P O } } ( \phi ; q ) = \frac { 1 } { M ( N - 1 ) } \sum _ { m = 1 } ^ { M } \sum _ { t = 1 } ^ { N - 1 } \operatorname* { m i n } \Bigl [ \rho _ { m , t } ( \phi ) A _ { m } , } } \\ { \displaystyle { \mathrm { c l i p } \bigl ( \rho _ { m , t } ( \phi ) , 1 - \epsilon , 1 + \epsilon \bigr ) A _ { m } \Bigr ] . } } \end{array}\tag{13}
$$

Here, $\epsilon > 0$ is the clipping parameter and $A _ { m } =$ $( r _ { m } - \bar { r } ) / \sigma _ { r }$ is the group-relative advantage, where r¯ and $\sigma _ { r }$ are the mean and standard deviation of the M rewards. Each action receives its trajectory’s terminal advantage, encouraging selections associated with better final rankings within the group.

Final embedding supervision. The policy objective acts on the first N − 1 states. Although the final embedding determines the reward, the detached ranking reward provides no gradient through that embedding. Shared parameters can still change its representation indirectly. We therefore add direct contrastive supervision at the final state, using the fixed item embeddings in $\mathcal { C } _ { \mathrm { R L } } ( q )$

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { r e t } } ( \phi ; q ) = \displaystyle \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbb { I } [ r _ { m } < R ] } \\ & { \quad \quad \cdot \ell _ { \mathrm { N C E } } \Big ( \mathbf { z } _ { N } ^ { \phi , ( m ) } , d ^ { + } , \mathcal { C } _ { \mathrm { R L } } ( q ) ; \{ \overline { { \mathbf { v } } } _ { d } \} _ { q } \Big ) . } \end{array}\tag{14}
$$

The combined objective is

$$
\begin{array} { r } { \begin{array} { r l } & { \mathcal { L } _ { \mathrm { R L } } ( \phi ) = \mathbb { E } _ { q , \{ \xi ^ { ( m ) } \} } \left[ - J _ { \mathrm { G R P O } } ( \phi ; q ) + \lambda \mathcal { L } _ { \mathrm { r e t } } ( \phi ; q ) \right] } \\ & { \qquad \sim \pi _ { \phi _ { \mathrm { O l d } } } } \\ & { \qquad \lambda = \lambda _ { \mathrm { r e t } } \mathbb { I } [ r < R ] . } \end{array} } \end{array}\tag{15}
$$

Here, I[·] is the indicator function, R is a reward threshold, and $\lambda _ { \mathrm { { r e t } } } \geq 0$ weights the auxiliary loss. The gate is applied to each trajectory before averaging: only trajectories with rollout reward below R receive the auxiliary loss. The gate remains fixed during the policy update. This concentrates direct supervision on trajectories with poorer final rankings. We do not include a reference-policy KL penalty in the RL objective. Instead of constraining the policy to remain close to a separate reference model, we directly preserve retrieval quality through contrastive supervision on the final representation. This contrastive objective provides taskspecific supervision for the final retrieval embedding, while the RL objective optimizes the model’s feedback-selection behavior.

## 5 Experiments

## 5.1 Experimental Setup

We present only the key experimental settings in this section, while Appendix A provides the full details.

Benchmarks. We train and evaluate ARR on M-BEIR (Wei et al., 2024), which contains 16 dataset–task combinations spanning eight query– item modality configurations.

Implementation. We initialize ARR from the 2B and 8B Qwen3-VL-Embedding models (Li et al., 2026a). The visual encoder remains frozen throughout training. SFT fine-tunes the remaining backbone parameters, whereas RL freezes the SFT backbone and trains only a query-side LoRA adapter. SFT uses offline top-10 feedback pools. For RL, we sample 2K queries from each M-BEIR dataset– task combination, giving 32K queries in total. To make training tractable, we use a query-specific pool $\mathcal { C } _ { \mathrm { R L } } ( q ) = \mathcal { C } _ { 1 0 0 } ( q ) \cup \{ d ^ { + } \}$ , where $\mathcal { C } _ { 1 0 0 } ( q )$ contains the SFT model’s top-100 candidates. Feedback sampling, reward computation, and final-step contrastive supervision all use this pool. Unless otherwise stated, we use $N = 4$ for training and inference and select feedback greedily at inference time. ARR-SFT and ARR-RL denote the models after the two training stages.

Baselines. We compare against MM-Embed (Lin et al., 2025) and LamRA-Ret (Liu et al., 2025), as well as TRACE (Hao et al., 2026), which incorporates adaptive reasoning, and ELVA (Liu et al., 2026), which uses ranking-driven RL. These comparisons cover contrastive embedding learning, reasoning-enhanced retrieval, and retrievaloriented policy optimization.

## 5.2 In-Domain Evaluation

Table 1 reports results on the M-BEIR test set. ARR-RL achieves the highest average score among the compared methods at both scales: 56.7 for 2B and 61.0 for 8B. RL improves the reported averages over ARR-SFT by 0.7 and 1.2 points, respectively. At 8B, ARR-RL exceeds the strongest baseline average, TRACE’s 58.8, by 2.2 points.

The gains vary across tasks. For the 8B model, RL increases EDIS (ES) text-to-multimodal recall from 66.8 to 73.2 and OVEN (ON) multimodal-totext recall from 58.7 to 63.5. These tasks require connecting the query to an item with a different information structure, making them promising settings for feedback-conditioned adaptation. Overall, RL improves the SFT model on most tasks, demonstrating the effectiveness of policy optimization for ARR. Some task configurations exhibit slight performance degradation, including VisualNews (VN) at 2B and CIRR (CR) at both scales. We hypothesize that these drops may stem from the sensitivity of RL training to the data: since RL is performed intensively on a relatively small subset of training data, variations in data quality and distribution across tasks may introduce task-specific biases. Nevertheless, the gains on most tasks indicate that RL provides an overall improvement over the SFT model.

## 5.3 Analysis of the Iterative Retrieval Process

Table 2 separates the effects of training with feedback from using feedback at inference time. Subscripts indicate the number of training states; columns report retrieval from each inference state. All RL-stage variants start from $\mathrm { A R R - S F T } _ { N = 4 }$ The N = 1 RL-stage control receives additional contrastive training on the RL data but uses neither feedback nor GRPO during that training.

Table 1: M-BEIR results in the local-pool setting (%). Superscripts t and i denote text and image inputs, respectively. ARR rows report the final state $( N = 4 )$
<table><tr><td rowspan="2">Method</td><td colspan="3"> $q ^ { \mathrm { t } }  d ^ { \mathrm { i } }$ </td><td colspan="2"> $q ^ { \mathrm { t } }  d ^ { \mathrm { t } }$ </td><td colspan="2"> $q ^ { \mathrm { t } } \to \dot { d } ^ { \mathrm { i , t } }$ </td><td colspan="2"> $q ^ { \mathrm { i } }  d ^ { \mathrm { t } }$ </td><td> $q ^ { \mathrm { i } }  \dot { d } ^ { \mathrm { i } }$ </td><td> $q ^ { \mathrm { i , t } } \to d ^ { \mathrm { t } }$ </td><td colspan="2"> $q ^ { \mathrm { i , t } } \to \dot { d } ^ { \mathrm { i } }$ </td><td colspan="2"></td><td colspan="2"> $q ^ { \mathrm { i , t } } \to d ^ { \mathrm { i , t } }$ </td></tr><tr><td>VN R@5</td><td>CO R@5</td><td>F200 R@10</td><td>WQ R@5</td><td>ES R@5</td><td>WQ R@5</td><td>VN R@5</td><td>CO R@5</td><td>F200 R@10</td><td>NS R@5</td><td>ON R@5</td><td>InS R@5</td><td>FIQ R@10</td><td>CR R@5</td><td>ON R@5</td><td>InS R@5</td><td>Avg.</td></tr><tr><td>2B models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LamRA-Ret (Liu et al., 2025)</td><td>30.8</td><td>78.8</td><td>23.1</td><td>82.5</td><td>54.3</td><td>77.8</td><td>31.2</td><td>88.5</td><td>27.1</td><td>28.7</td><td>51.1</td><td>44.2</td><td>28.9</td><td>47.7</td><td>72.3</td><td>60.8</td><td>51.6</td></tr><tr><td>ELVA (Liu et al., 2026)</td><td>35.6</td><td>80.3</td><td>25.0</td><td>88.0</td><td>56.1</td><td>80.5</td><td>33.4</td><td>90.2</td><td>25.9</td><td>29.3</td><td>52.0</td><td>47.4</td><td>30.9</td><td>50.0</td><td>72.8</td><td>61.3</td><td>53.8</td></tr><tr><td>ARR-SFT</td><td>32.6</td><td>84.4</td><td>27.2</td><td>93.3</td><td>60.4</td><td>84.1</td><td>32.1</td><td>94.3</td><td>27.6</td><td>32.2</td><td>55.2</td><td>42.9</td><td>33.1</td><td>59.7</td><td>74.2</td><td>62.8</td><td>56.0</td></tr><tr><td>ARR-RL</td><td>32.0</td><td>84.0</td><td>28.2</td><td>94.0</td><td>63.9</td><td>85.1</td><td>31.8</td><td>94.2</td><td>28.7</td><td>34.1</td><td>57.7</td><td>42.4</td><td>33.3</td><td>59.3</td><td>75.3</td><td>63.9</td><td>56.7</td></tr><tr><td>8B/7B models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MM-Embed (Lin et al., 2025)</td><td>41.0</td><td>71.3</td><td>17.1</td><td>95.9</td><td>68.8</td><td>85.0</td><td>41.3</td><td>90.1</td><td>18.4</td><td>32.4</td><td>42.1</td><td>42.3</td><td>25.7</td><td>50.0</td><td>64.1</td><td>57.7</td><td>52.7</td></tr><tr><td>LamRA-Ret (Liu et al., 2025)</td><td>41.6</td><td>81.5</td><td>28.7</td><td>86.0</td><td>62.6</td><td>81.2</td><td>39.6</td><td>90.6</td><td>30.4</td><td>32.1</td><td>54.1</td><td>52.1</td><td>33.2</td><td>53.1</td><td>76.2</td><td>63.3</td><td>56.6</td></tr><tr><td>TRACE (Hao et al., 2026)</td><td>42.1</td><td>82.3</td><td>30.5</td><td>87.8</td><td>64.1</td><td>82.5</td><td>41.2</td><td>91.3</td><td>33.2</td><td>33.6</td><td>57.4</td><td>55.8</td><td>36.4</td><td>57.3</td><td>78.5</td><td>67.1</td><td>58.8</td></tr><tr><td>ELVA (Liu et al., 2026)</td><td>43.5</td><td>83.0</td><td>29.2</td><td>91.0</td><td>63.5</td><td>83.1</td><td>41.7</td><td>92.2</td><td>32.1</td><td>32.8</td><td>56.0</td><td>55.5</td><td>34.6</td><td>55.4</td><td>77.5</td><td>67.1</td><td>58.7</td></tr><tr><td>ARR-SFT</td><td>40.6</td><td>85.8</td><td>27.8</td><td>95.9</td><td>66.8</td><td>86.0</td><td>40.1</td><td>94.7</td><td>28.1</td><td>33.6</td><td>58.7</td><td>56.1</td><td>35.0</td><td>64.4</td><td>75.3</td><td>67.8</td><td>59.8</td></tr><tr><td>ARR-RL</td><td>41.3</td><td>85.6</td><td>30.0</td><td>95.7</td><td>73.2</td><td>86.2</td><td>41.0</td><td>94.7</td><td>30.2</td><td>34.0</td><td>63.5</td><td>55.8</td><td>35.1</td><td>64.0</td><td>78.1</td><td>68.2</td><td>61.0</td></tr></table>

Feedback use must be learned. $\mathbf { A R R - S F T } _ { N = 1 }$ scores 55.23 at the initial state but falls to 46.90 when extended to four inference states, showing that simply appending retrieved content can harm a model not trained to use it. In contrast, ARR-${ \mathrm { S F T } } _ { N = 4 }$ improves from 55.63 to 56.01. Most of this gain occurs after the first feedback item, at Iter-2 (55.95), with smaller changes thereafter.

Table 2: Effect of training and inference steps for ARR-2B on M-BEIR (Avg.%).
<table><tr><td>Method</td><td>Iter-1</td><td>Iter-2</td><td>Iter-3</td><td>Iter-4</td></tr><tr><td> $\mathbf { A R R - S F T } _ { N = 1 }$ </td><td>55.23</td><td>49.85</td><td>48.37</td><td>46.90</td></tr><tr><td> $\mathrm { A R R - S F T } _ { N = 4 }$ </td><td>55.63</td><td>55.95</td><td>56.01</td><td>56.01</td></tr><tr><td> $\mathrm { A R R - R L } _ { N = 1 }$ </td><td>55.44</td><td>55.57</td><td>55.54</td><td>55.47</td></tr><tr><td> $\mathrm { A R R - R L } _ { N = 2 }$ </td><td>56.40</td><td>56.59</td><td>56.58</td><td>56.64</td></tr><tr><td> $\mathrm { A R R - R L } _ { N = 3 }$ </td><td>56.40</td><td>56.57</td><td>56.62</td><td>56.63</td></tr><tr><td> $\mathrm { A R R - R L } _ { N = 4 }$ </td><td>56.36</td><td>56.68</td><td>56.66</td><td>56.74</td></tr></table>

Feedback training also improves initial retrieval. Before observing any feedback, $\mathrm { A R R - S F T } _ { N = 4 }$ outperforms $\mathbf { A R R - S F T } _ { N = 1 }$ by 0.40 points. Thus, training on feedback-conditioned states benefits the shared encoder even when inference stops after the first step. One possible explanation is that supervising several contexts encourages features that also help represent the original query; these results establish the performance effect, rather than a particular feature-learning mechanism.

Policy training provides further gains. Policy training consistently improves upon the SFT initialization. $\mathbf { A R R - R L } _ { N = 4 }$ achieves scores of 56.36 at Iter-1 and $5 6 . 7 4$ at Iter-4, compared with 55.63 and 56.01, respectively, for its SFT initialization. Notably, the improvement at Iter-1 indicates that the benefits of RL are not limited to exploiting feedback from previous retrieval steps. We attribute this gain to the richer supervision provided by the RL objective. While the SFT objective treats each item as either positive or negative, the RL objective assigns credit to a selected item according to its contribution to the eventual retrieval outcome. It therefore provides a softer and more informative learning signal, allowing the model to favor useful intermediate items even when they are not annotated positives and to better distinguish hard negatives based on their utility for subsequent retrieval. The performance gap between $\mathbf { A R R - R L } _ { N = 1 }$ and $\mathrm { A R R - R L } _ { N = 2 }$ further supports this interpretation. $\mathrm { A R R - R L } _ { N = 1 }$ , which is optimized only with a contrastive objective, achieves 55.44 at Iter-1, whereas $\mathrm { A R R - R L } _ { N = 2 }$ , trained with the RL objective, reaches 56.40 at the same step. These results suggest that policy-level credit assignment provides complementary supervision beyond conventional contrastive learning.

Additional steps yield diminishing returns. Training with two, three, or four states gives similar initial scores. Four-state training produces the best final score, but its advantage over two-state training at Iter-4 is only 0.10 points. Together with the marginal improvements beyond Iter-2, these results suggest a favorable trade-off between test-time scaling and retrieval performance in ARR, with Iter-2 offering a good balance between computational cost and performance gains.

Table 3: Zero-shot retrieval results for 8B models (%). Iter-1 uses the initial query embedding without feedback; unqualified ARR rows use the final state $( N = 4 )$ . An asterisk marks benchmarks with images sourced from COCO or FashionIQ.
<table><tr><td rowspan="2">Method</td><td colspan="3"> $q ^ { \mathrm { t } }  d ^ { \mathrm { i } }$ </td><td colspan="3"> $q ^ { \mathrm { i } }  { d ^ { \mathrm { t } } }$ </td><td colspan="2"> $q ^ { \mathrm { i , t } }  d ^ { \mathrm { i } }$ </td><td colspan="2"> $q ^ { \mathrm { d i a l o g } }  d ^ { \mathrm { i } }$ </td><td rowspan="2"> $q ^ { \mathrm { i } \oplus \mathrm { t } } \to d ^ { \mathrm { i } }$ </td><td rowspan="2"> $\mathbf { A v g } .$ </td></tr><tr><td>Share4V R@1</td><td>Urban* R@1</td><td>Flickr R@1</td><td>Share4V R@1</td><td>Urban* R@1</td><td>Flickr R@1</td><td>CIRCO* MAP@5</td><td>GeneCIS* R@1</td><td>VisD* R@1</td><td>MT-FIQ* R@5</td></tr><tr><td>LamRA-Ret (Liu et al., 2025)</td><td>93.3</td><td>95.1</td><td>82.8</td><td>88.1</td><td>94.3</td><td>92.7</td><td>33.2</td><td>18.9</td><td>62.8</td><td></td><td>60.9</td><td>72.21</td></tr><tr><td>TRACE (Hao et al., 2026)</td><td>94.9</td><td>94.8</td><td>84.5</td><td>89.1</td><td>94.1</td><td>94.5</td><td>34.8</td><td>20.5</td><td>65.4</td><td></td><td>63.2</td><td>73.58</td></tr><tr><td>ELVA (Liu et al., 2026)</td><td>96.6</td><td>96.1</td><td>84.4</td><td>92.0</td><td>95.5</td><td>95.2</td><td>34.5</td><td>20.2</td><td>65.3</td><td></td><td>61.2</td><td>74.10</td></tr><tr><td>ARR-SFT (Iter-1)</td><td>98.8</td><td>98.9</td><td>87.3</td><td>99.0</td><td>99.0</td><td>97.7</td><td>39.3</td><td>20.9</td><td>75.5</td><td></td><td>63.1</td><td>77.95</td></tr><tr><td>ARR-SFT</td><td>98.8</td><td>99.1</td><td>87.6</td><td>99.0</td><td>99.0</td><td>97.8</td><td>41.6</td><td>21.0</td><td>76.3</td><td></td><td>63.8</td><td>78.39</td></tr><tr><td>ARR-RL (Iter-1)</td><td>98.4</td><td>99.0</td><td>87.5</td><td>99.0</td><td>99.0</td><td>98.0</td><td>39.9</td><td>21.0</td><td>76.6</td><td></td><td>65.9</td><td>78.42</td></tr><tr><td>ARR-RL</td><td>98.4</td><td>99.0</td><td>88.0</td><td>99.0</td><td>99.0</td><td>98.3</td><td>43.4</td><td>21.1</td><td>77.9</td><td></td><td>66.0</td><td>79.01</td></tr></table>

## 5.4 Zero-Shot Evaluation

Table 3 evaluates the 8B models on seven benchmarks without task-specific fine-tuning. ARR-RL obtains the highest reported average, 79.01, compared with 74.10 for the strongest baseline average. Its scores of 77.9 on Visual Dialog and 66.0 on Multi-round FashionIQ also indicate transfer to dialogue and interactive retrieval.

The models from both training stages benefit from feedback at inference time: ARR-SFT improves from 77.95 at Iter-1 to 78.39 at the final step, and ARR-RL improves from 78.42 to 79.01. The gains are most visible on CIRCO and Visual Dialog. RL also raises the average at both the initial and final states relative to SFT, although some individual scores decrease slightly. These results extend the two observed benefits of feedback—better inference with retrieved context and better initial representations after training—beyond the M-BEIR evaluation tasks.

## 5.5 Ablation Study

Table 4 examines ARR’s training objectives using the 2B model.

SFT design choices. Retrieval feedback is intended to help the model better understand the query. Accordingly, contrastive supervision at feedback-conditioned steps should provide a learning signal to the query representation, and we therefore stop gradients through the standalone item embeddings. When item gradients are enabled, the final score drops from 56.01 to 55.71, suggesting that the model may exploit undesirable shortcuts to minimize the training loss rather than learn meaningful item representations. Removing the degradation penalty leaves initial performance nearly unchanged (55.69 vs. 55.63), but eliminates the gain from feedback: the final score is 55.68 instead of 56.01. The penalty therefore improves the usefulness of later states rather than simply increasing first-step accuracy.

Table 4: Training ablations for ARR-2B on M-BEIR (Avg.%). “w/o reward gate” applies final-step supervision to all trajectories.
<table><tr><td>Setting</td><td>Iter-1</td><td>Iter-2</td><td>Iter-3</td><td>Iter-4</td></tr><tr><td>ARR-SFT</td><td>55.63</td><td>55.95</td><td>56.01</td><td>56.01</td></tr><tr><td>w/o sg(vd)</td><td>55.31</td><td>55.62</td><td>55.72</td><td>55.71</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { d e g } }$ </td><td>55.69</td><td>55.61</td><td>55.66</td><td>55.68</td></tr><tr><td>ARR-RL</td><td>56.36</td><td>56.68</td><td>56.66</td><td>56.74</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e t } }$ </td><td>56.20</td><td>56.40</td><td>56.39</td><td>56.43</td></tr><tr><td>w/o reward gate</td><td>56.39</td><td>56.55</td><td>56.52</td><td>56.39</td></tr><tr><td>w/KL</td><td>56.20</td><td>56.46</td><td>56.50</td><td>56.49</td></tr><tr><td>w/o JGRPO</td><td>55.74</td><td>55.65</td><td>55.47</td><td>55.29</td></tr></table>

Policy and final embedding supervision. Removing the final embedding contrastive loss lowers the final score from 56.74 to 56.43, which remains above the SFT score. Conversely, removing GRPO yields 55.29 at the final step, below the scores of both the full RL model and its SFT initialization. Because the final embedding is conditioned on the retrieval trajectory, supervising only the final embedding can disrupt the model’s retrieval policy, which in turn degrades the quality of the final representation. This result further suggests that the gains of ARR-RL stem from learning an effective retrieval policy, rather than from exploiting superficial item-specific patterns. The two objectives are therefore complementary in this setting: policy training improves feedback selection, and direct terminal supervision improves the representation

used for ranking.

Reward gating and KL regularization. Applying the auxiliary contrastive loss to every trajectory, instead of gating it by reward, gives a final score of 56.39. This supports concentrating the auxiliary loss on trajectories with lower rewards. The variant with a KL penalty scores 56.49, compared with 56.74 for the default model. Thus, the default objective performs better in this comparison while avoiding a separate reference-model evaluation. Appendix B further examines the weights $\lambda _ { \mathrm { d e g } }$ and $\lambda _ { \mathrm { { r e t } } }$

## 6 Conclusion

We presented ARR, a multimodal retriever that learns to select and use retrieved items as feedback through an autoregressive retrieval process. ARR preserves independently indexed item embeddings while conditioning the query on a sequence of retrieved observations. SFT teaches feedback interpretation, and RL optimizes item selection for the final ranking outcome. Experiments show improvements in average in-domain and zero-shot retrieval performance. Further analyses highlight two complementary benefits of item feedback: using feedback during inference improves subsequent retrieval, while learning from feedback during training improves initial retrieval even before any feedback is observed.

## References

Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. 2023. Zero-shot composed image retrieval with textual inversion. In ICCV, pages 15338–15347.

Amin Bigdeli, Negar Arabzadeh, Radin Hamidi Rad, Sajad Ebrahimi, Charles LA Clarke, and Ebrahim Bagheri. 2026. Adore: Iterative query expansion with retrieval-grounded relevance feedback. arXiv preprint arXiv:2606.13905.

Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. 2022. WebQA: Multihop and multimodal QA. In CVPR, pages 16495–16504.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024. ShareGPT4V: Improving large multi-modal models with better captions. In ECCV, pages 370– 387. Springer.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for con-

trastive learning of visual representations. In ICML, pages 1597–1607. PmLR.

Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. 2023. Can pre-trained vision and language models answer visual information-seeking questions? In EMNLP, pages 14948–14968.

Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Mishra, Taipeng Tian, Qi Guo, et al. 2026. Think then embed: Generative context improves multimodal embedding. In ICLR, volume 2026, pages 2690–2709.

Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José MF Moura, Devi Parikh, and Dhruv Batra. 2017. Visual dialog. In CVPR, pages 326–335.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. 2023. DreamSim: Learning new dimensions of human visual similarity using synthetic data. In NeurIPS.

Xintong Han, Zuxuan Wu, Phoenix X Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S Davis. 2017. Automatic spatially-aware fashion concept discovery. In ICCV, pages 1463–1471.

Xiangzhao Hao, Shijie Wang, Tianyu Yang, Tianyue Wang, Haiyun Guo, and Jinqiao Wang. 2026. Trace: Task-adaptive reasoning and representation learning for universal multimodal retrieval. arXiv preprint arXiv:2603.02929.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9726–9735. IEEE.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In ICLR.

Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. 2023. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In ICCV, pages 12065–12075.

Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580.

Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. 2026a. Qwen3-vlembedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720.

Xiaojie Li, Chu Li, Shi-Zhe Chen, and Xi Chen. 2026b. U-marvel: Unveiling key factors for universal multimodal retrieval via embedding learning with mllms. In ICLR, volume 2026, pages 121298–121319.

Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2025. Mm-embed: Universal multimodal retrieval with multimodal llms. In ICLR, volume 2025, pages 44215–44234.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft COCO: Common objects in context. In ECCV, pages 740– 755. Springer.

Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente Ordonez. 2021a. Visual News: Benchmark and challenges in news image captioning. In EMNLP.

Siqi Liu, Weixi Feng, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. 2023. EDIS: Entity-driven image search over multimodal web content. In EMNLP, pages 4877–4894.

Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. 2025. Lamra: Large multimodal model as your advanced retrieval assistant. In CVPR, pages 4015– 4025. IEEE.

Yuhan Liu, Pei Fu, Hang Li, Yukun Qi, Chao Jiang, Jingwen Fu, Zhen Liu, Bin Qin, Zhenbo Luo, Jian Luan, et al. 2026. Elva: Exploring rankingdriven universal multimodal retrieval. arXiv preprint arXiv:2606.20280.

Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. 2021b. Image retrieval on real-life images with pre-trained vision-and-language models. In ICCV, pages 2125–2134.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k Entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In ICCV, pages 2641–2649.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML, pages 8748– 8763. PmLR.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Dike Sun, Zheng Zou, Jingtong Zang, Qi Sun, Huaipeng Zhaoand Tao Luo, and Xiaoyi Zeng. 2026a. Queryagent-r1: Bridging query generation and product retrieval for e-commerce query recommendation. arXiv preprint arXiv:2606.05671.

Siyu Sun, Jing Ren, Zhaohe Liao, Dongxiao Mao, Xiangyuan Ren, Yiyi Zhang, Haohua Zhao, Weixiong Lin, Jiang Shaohua, Liqing Zhang, et al. 2026b. Bottleneck tokens for unified multimodal retrieval. arXiv preprint arXiv:2604.11095.

Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, Fen Lin, Qin Liu, and Qingyao Ai. 2026. Generalized pseudo-relevance feedback. In Proceedings of the ACM Web Conference 2026, pages 1876–1886.

Sagar Vaze, Nicolas Carion, and Ishan Misra. 2023. GeneCIS: A benchmark for general conditional image similarity. In CVPR, pages 6862–6872.

Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. 2024. Uniir: Training and benchmarking universal multimodal information retrievers. In ECCV, pages 387–404. Springer.

Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. 2021. Fashion IQ: A new dataset towards retrieving images by natural language feedback. In CVPR, pages 11307–11317.

Peixi Wu, Biao Yang, Feipeng Ma, Bosong Chai, Bo Lin, Wei Yuan, Fan Yang, Tingting Gao, Hebei Li, and Xiaoyan Sun. 2026. Lame: Learning to think in latent space for multimodal embedding via information bottleneck. arXiv preprint arXiv:2606.13061.

HongChien Yu, Chenyan Xiong, and Jamie Callan. 2021. Improving query representations for dense retrieval with pseudo relevance feedback. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management.

Yifei Yuan and Wai Lam. 2021. Conversational fashion image retrieval via multiturn natural language feedback. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 839–848.

Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. 2024. Long-CLIP: Unlocking the long-text capability of CLIP. In ECCV, pages 310– 325. Springer.

Longxiang Zhang, Weilong Dai, Guanghao Zhang, Hao Jiang, and Pipei Huang. 2026a. Think when needed: Adaptive reasoning-driven multimodal embeddings with a dual-lora architecture. arXiv preprint arXiv:2605.14448.

Xuxin Zhang, Ben Chen, Yue Lv, Siyuan Wang, Yupeng Li, Yufei Ma, Zihan Liang, Tong Zhao, Ying Yang, Huangyu Dai, et al. 2026b. Oneretrieval: Unifying multi-branch e-commerce retrieval with an editable generative model. arXiv preprint arXiv:2606.13533.

Yihua Zhang, Mingfu Liang, Jiyan Yang, Rong Jin, Wen-Yen Chen, Yiping Han, Huayu Li, Buyun Zhang, Liang Luo, Luke Simon, et al. 2026c. Reasonrec: A reasoning-augmented multimodal agent for unified

recommendation. In Findings of ACL, pages 7960– 7980.

Zhaohua Zhang, Jianhuan Zhuo, Muxi Chen, Chenchen Zhao, Wenyu Jiang, Tianwen Jiang, Mingyang Chen, Qiuyong Xiao, Jihong Zhang, Zhixun Su, et al. 2025. Grape: Let grpo supervise query rewriting by ranking for retrieval. arXiv preprint arXiv:2509.23370.

Jinghan Zhao, Wenwei Jin, Anqi Li, Jintao Tong, Luya Mo, Jiawei Li, Yao Hu, and Bin Li. 2026. Uninote: A unified embedding model for multimodal representation and ranking. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 8555–8564.

Hongyi Zhu, Jia-Hong Huang, Stevan Rudinac, and Evangelos Kanoulas. 2024. Enhancing interactive image retrieval with query rewriting using large language models and vision language models. In Proceedings of the 2024 international conference on multimedia retrieval, pages 978–987.

## A Detailed Experimental Settings

## A.1 Benchmarks

In-domain evaluation. We use the M-BEIR local-pool setting (Wei et al., 2024), in which candidates from the task-specific collection are ranked for each query. M-BEIR comprises 16 dataset– task combinations from 10 datasets, covering eight query–item modality configurations and four domains: news, fashion, Wikipedia, and miscellaneous content. Table 5 lists rounded benchmark split sizes and candidate-pool sizes. We report Recall@10 for Fashion200K and FashionIQ and Recall@5 for all other tasks.

Zero-shot evaluation. Table 6 lists query counts, candidate-pool sizes, and metrics for the seven benchmarks in Table 3. ShareGPT4V and Urban-1K evaluate retrieval between images and long captions, and Flickr30K evaluates image–caption retrieval. CIRCO and GeneCIS condition retrieval on an image and text. Visual Dialog evaluates dialogue-to-image retrieval, while Multi-round FashionIQ uses multi-turn image–text interactions. We evaluate these tasks without task-specific finetuning. Several benchmarks reuse images from COCO or FashionIQ, so this protocol does not imply that all visual content is unseen.

## A.2 Implementation and Training

Model and feedback pools. We use the 2B and 8B Qwen3-VL-Embedding backbones (Li et al., 2026a). The visual encoder remains frozen in both stages. SFT fine-tunes the remaining backbone parameters; RL freezes the SFT backbone and trains an additional query-side LoRA adapter. The adapter is active when encoding the query and its feedback context, but inactive when encoding standalone corpus items.

For SFT, the pretrained encoder retrieves a top-10 feedback pool for each query. We sample three items without replacement to construct four query states. Missing positives are not added to this pool, and relevance labels are not used to filter feedback. The annotated positive is included separately in the contrastive supervision set. We train the SFT model for two epochs. In the first epoch, we disable the degradation penalty by setting $\lambda _ { \mathrm { d e g } } = 0$ in Eq. 9, allowing the model to first acquire basic multi-round retrieval capability. In the second epoch, we enable the degradation penalty to encourage the model to leverage retrieved items to refine the query representation.

For RL, we sample 2K queries from each of the 16 M-BEIR dataset–task combinations, giving 32K queries. We form $\mathcal { C } _ { \mathrm { R L } } ( q ) = \mathcal { C } _ { 1 0 0 } ( q ) \cup \{ d ^ { + } \}$ from the SFT model’s top-100 results and the annotated positive, removing duplicates. The resulting pool is shared by feedback sampling, reward computation, and final-step contrastive supervision. The history mask applies only to feedback selection. The reward gate in (14) is applied to each trajectory before averaging its auxiliary loss. Unless otherwise stated, N = 4 at both training and inference, and inference selects the highest-scoring unvisited feedback item at each step.

For queries with multiple positive items, during SFT we randomly select one positive as the ground-truth target and mask the remaining positives. During RL, the final-embedding supervision averages the losses over all positive items, while the reward function applies max pooling over their rewards.

Tables 7 and 8 list settings shared by ARR-2B and ARR-8B. SFT batch sizes count query–positive pairs. RL rollout batch sizes count queries, whereas optimization batch sizes count trajectories.

SFT takes 24 hours on 8 NVIDIA H20 GPUs for the 2B model and 48 hours on 16 NVIDIA H20 GPUs for the 8B model. RL uses 8 NVIDIA H20 GPUs for both models, taking 8 hours for 2B and 24 hours for 8B.

## A.3 Baselines

MM-Embed. MM-Embed (Lin et al., 2025) adapts an MLLM for universal multimodal retrieval through contrastive learning and modality-aware hard-negative mining, with text retrieval data providing additional supervision.

LamRA-Ret. LamRA (Liu et al., 2025) provides separate retrieval and reranking components. Its retrieval component, LamRA-Ret, learns multimodal embeddings through language-only pretraining and multimodal instruction tuning. We compare with the retrieval component in our main tables.

TRACE. TRACE (Hao et al., 2026) integrates reasoning and representation learning in an MLLM, adapting its reasoning behavior to the retrieval task. It represents an approach that improves query interpretation before embedding.

ELVA. ELVA (Liu et al., 2026) constructs embeddings after generating a textual synthesis of the input and uses ranking and margin rewards for RL. Its policy training targets retrieval-oriented representations, providing a comparison with ARR’s policy over feedback items.

Table 5: M-BEIR tasks and rounded benchmark statistics (Wei et al., 2024; Li et al., 2026b). Here, t denotes text and i denotes image; i+t denotes a joint image–text input. Split sizes count queries; “Pool” counts candidates in the local evaluation collection. These are benchmark statistics, not counts after ARR-specific preprocessing.
<table><tr><td>Dataset</td><td></td><td>Query → item Train / dev / test queries</td><td>Pool</td></tr><tr><td>VisualNews (VN) (Liu et al., 2021a)</td><td>t →i</td><td>99K / 20K / 20K</td><td>542K</td></tr><tr><td>MSCOCO (CO) (Lin et al., 2014)</td><td>t → i</td><td>100K / 24.8K / 24.8K</td><td>5K</td></tr><tr><td>Fashion200K (F200) (Han et al., 2017)</td><td>t → i</td><td>15K / 1.7K / 1.7K</td><td>201K</td></tr><tr><td>WebQA (WQ) (Chang et al., 2022)</td><td>t → t</td><td>16K / 1.7K / 2.4K</td><td>544K</td></tr><tr><td>EDIS (ES) (Liu et al., 2023)</td><td>t → i+t</td><td>26K / 3.2K / 3.2K</td><td>1M</td></tr><tr><td>WebQA (WQ)</td><td>t → i+t</td><td>17K / 1.7K / 2.5K</td><td>403K</td></tr><tr><td>VisualNews (VN)</td><td>i → t</td><td>100K / 20K / 20K</td><td>537K</td></tr><tr><td>MSCOCO (CO)</td><td>i → t</td><td>113K / 5K / 5K</td><td>25K</td></tr><tr><td>Fashion200K (F200)</td><td>i → t</td><td>15K / 4.8K / 4.8K</td><td>61K</td></tr><tr><td>NIGHTS (NS) (Fu et al., 2023)</td><td>i →i</td><td>16K / 2K / 2K</td><td>40K</td></tr><tr><td>OVEN (ON) (Hu et al., 2023)</td><td>i+t → t</td><td>150K / 50K / 50K</td><td>676K</td></tr><tr><td>InfoSeek (InS) (Chen et al., 2023)</td><td>i+t → t</td><td>141K/11K /11K</td><td>611K</td></tr><tr><td>FashionIQ (FIQ) (Wu et al., 2021)</td><td> $\mathrm { i + t } \to \mathrm { i }$ </td><td>16K / 2K / 6K</td><td>74K</td></tr><tr><td>CIRR (CR) (Liu et al., 2021b)</td><td> $\mathrm { i + t } \to \mathrm { i }$ </td><td>26K / 2K / 4K</td><td>21K</td></tr><tr><td>OVEN (ON)</td><td> $\mathrm { i + t } \to \mathrm { i + t }$ </td><td>157K / 14.7K / 14.7K</td><td>335K</td></tr><tr><td>InfoSeek (InS)</td><td>i+t → i+t</td><td>143K / 17.6K / 17.6K</td><td>481K</td></tr></table>

Table 6: Zero-shot evaluation tasks and metrics. Counts are listed separately when retrieval directions differ. Candidate counts refer to the pool searched by each query; GeneCIS uses query-specific candidate sets.
<table><tr><td>Benchmark</td><td>Query → item</td><td>Queries / candidates</td></tr><tr><td>ShareGPT4V (Share4V) (Chen et al., 2024)</td><td>t → i; i → t</td><td>1K / 1K (both directions)</td></tr><tr><td>Urban-1K (Urban) (Zhang et al., 2024)</td><td> $\mathbf { t }  \mathbf { i } ; \mathbf { i }  \mathbf { t }$ </td><td>1K / 1K (both directions)</td></tr><tr><td>Flickr30K (Flickr) (Plummer et al., 2015)</td><td> $\mathbf { t }  \mathbf { i } ; \mathbf { i }  \mathbf { t }$ </td><td>t → i: 5K / 1K; i → t: 1K / 5K</td></tr><tr><td>CIRCO (Baldrati et al., 2023)</td><td>i+t →i</td><td>800 / 120K</td></tr><tr><td>GeneCIS (Vaze et al., 2023)</td><td>i+t → i</td><td>8K / 10-15</td></tr><tr><td>Visual Dialog (VisD) (Das et al., 2017)</td><td>dialogue → i</td><td>2K/ 2K</td></tr><tr><td>Multi-round FashionIQ (MT-FIQ) (Yuan and Lam, 2021) interactive i/t → i</td><td></td><td>2.4K / 6.2K</td></tr></table>

Table 7: SFT training settings shared by ARR-2B and ARR-8B.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimizer Weight decay Learning rate Learning-rate schedule Warmup steps</td><td>AdamW 0.01  $5 \times 1 0 ^ { - 5 }$  Cosine 150</td></tr><tr><td>Training epochs Effective global batch size (pairs) Gradient clipping (maximum norm)</td><td>2 1,024 1.0</td></tr><tr><td>Contrastive temperature (τ in Eq. 2) Degradation penalty weight  $\lambda _ { \mathrm { d e g } }$  Hard negatives</td><td>0.02 0.1 0</td></tr><tr><td>Maximum text tokens Maximum image pixels Minimum image pixels Training precision</td><td>1,024  $1 , 2 8 0 \times 3 2 \times 3 2$   $6 4 \times 3 2 \times 3 2$  BF16</td></tr></table>

Table 8: RL training settings shared by ARR-2B and ARR-8B. All other settings follow those used for SFT.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Learning rate</td><td>5 × 10−6</td></tr><tr><td>Learning-rate schedule</td><td>Cosine</td></tr><tr><td>Warmup steps</td><td>50</td></tr><tr><td>Training epochs Effective global batch size (trajectories)</td><td>2 256</td></tr><tr><td>LoRA rank</td><td>128</td></tr><tr><td>LoRA scaling α LoRA dropout</td><td>256 0.0</td></tr><tr><td>Global rollout batch size (queries)</td><td>1,024</td></tr><tr><td>Trajectories per query M</td><td>8</td></tr><tr><td>Policy sampling temperature (τ in Eq. 4)</td><td></td></tr><tr><td>GRPO clipping €</td><td>0.1</td></tr><tr><td></td><td>0.2</td></tr><tr><td>Optimization passes per rollout batch</td><td>1</td></tr><tr><td>Final-step contrastive temperature (τ in Eq. 14)</td><td>0.05</td></tr><tr><td>Final-step contrastive weight  $\lambda _ { \mathrm { { r e t } } }$ </td><td>0.1</td></tr><tr><td>Reward threshold R</td><td>0.2</td></tr></table>

## B Hyperparameter Analysis

Table 9: Sensitivity to auxiliary loss weights for ARR-2B on M-BEIR (average recall, %). Bold marks the best result in each column within each training stage.
<table><tr><td>Setting</td><td>Iter-1</td><td>Iter-2</td><td>Iter-3</td><td>Iter-4</td></tr><tr><td> $A R R – S F T$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { d e g } } = 0$ </td><td>55.69 55.63</td><td>55.61 55.95</td><td>55.66 56.01</td><td>55.68 56.01</td></tr><tr><td> $\lambda _ { \mathrm { d e g } } = 0 . 1 ( \mathrm { d e f a u l t } )$   $\lambda _ { \mathrm { d e g } } = 1 . 0$ </td><td>55.60</td><td>55.76</td><td>55.86</td><td>55.89</td></tr><tr><td> $A R R – R L$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { { r e t } } } = 0$ </td><td>56.20</td><td>56.40</td><td>56.39</td><td></td></tr><tr><td></td><td>56.15</td><td></td><td></td><td>56.43</td></tr><tr><td> $\lambda _ { \mathrm { r e t } } = 0 . 0 1$ </td><td></td><td>56.45</td><td>56.46</td><td>56.47</td></tr><tr><td> $\lambda _ { \mathrm { r e t } } = 0 . 1$  (default)</td><td>56.36</td><td>56.68</td><td>56.66</td><td>56.74</td></tr><tr><td> $\lambda _ { \mathrm { r e t } } = 1 . 0$ </td><td>55.59</td><td>55.92</td><td>56.01</td><td>56.02</td></tr></table>

Table 9 examines the degradation-penalty weight $\lambda _ { \mathrm { d e g } }$ in (9) and the final-step supervision weight $\lambda _ { \mathrm { { r e t } } }$ in (15), using ARR-2B. We use 0.1 for both weights in the default configuration.

Degradation penalty. Without the penalty $( \lambda _ { \mathrm { d e g } } ~ = ~ 0 )$ , the score remains nearly constant across inference states: 55.69 at Iter-1 and 55.68 at Iter-4. With $\lambda _ { \mathrm { d e g } } = 0 . 1$ , it rises from 55.63 to 56.01. Increasing the weight to 1.0 also produces improvements across states, but yields a lower final score of 55.89. Thus, greater emphasis on preventing relative degradation does not necessarily maximize absolute retrieval accuracy. Among the tested values, 0.1 gives the best final score while maintaining comparable initial performance.

Final-step contrastive supervision. Without the auxiliary loss $( \lambda _ { \mathrm { { r e t } } } = 0 )$ , policy training reaches 56.43 at Iter-4, above the SFT score of 56.01. Setting $\lambda _ { \mathrm { { r e t } } } = 0 .$ 1 further raises this score to 56.74. A weight of 1.0 yields 56.02, suggesting that excessive emphasis on the auxiliary objective can reduce the gains from policy training. The intermediate value 0.1 performs best among the tested settings.