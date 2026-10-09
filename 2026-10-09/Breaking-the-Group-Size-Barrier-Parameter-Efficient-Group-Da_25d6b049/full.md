# Breaking the Group Size Barrier: Parameter-Efficient Group Dance Generation with Chain-of-Dancers

Jing Xu Cunjian Chen<sup>∗</sup> Qiuhong Ke Department of Data Science and Artificial Intelligence, Monash University, Australia

## Abstract

Group dance generation aims to synthesize coordinated multi-dancer choreography from music, with broad applications in animation and interactive content creation. This task requires modeling dense inter-person dependencies to ensure spatial coordination, while naturally preserving individual dancer identities. Existing approaches model all dancers jointly with end-to-end transformers, which tie the architecture to a fixed group size and entangle per-dancer identities across frames. We propose ChainDance, a scalable framework that reformulates group dance generation as a Chain-of-Dancers: a sequential decomposition over per-dancer conditional distributions, allowing a single model to scale across variable group sizes without retraining and naturally preserving per-dancer identity. Built on a frozen single-dancer diffusion backbone, ChainDance introduces two lightweight modules: a Role-Aware Text Encoder (RATE) for per-dancer semantic conditioning, and a Group-Aware Motion Encoder (GAME) that aggregates previously generated dancers via a distance-weighted graph convolutional network, and incorporates a training-free noise optimization procedure at inference time to enforce global spatial coherence. Experiments on AIOZ-GDance demonstrate that ChainDance achieves state-of-the-art motion quality and group coordination while structurally preserving per-dancer identity, with 3-4× fewer parameters and requiring 3-6× less training time compared to prior approaches.

## 1 Introduction

Group dance generation aims to synthesize coordinated multi-person motion from music while preserving individual identities and interactions [28, 39]. Despite recent progress, existing approaches [16, 38, 3] are fundamentally limited by how they formulate the problem. Most methods model all dancers jointly within a fixed-size representation, coupling group size to model architecture. This leads to two inherent constraints (Figure 1): (1) group-size rigidity, requiring retraining to handle different numbers of dancers, and (2) identity entanglement, where individual dancer characteristics are obscured by shared representations.

These limitations are not merely architectural inefficiencies. They stem from a deeper assumption that group choreography must be generated as a single joint process. In contrast, we ask: can group choreography be constructed as a composition of individual motion processes while still capturing inter-person coordination? This question is grounded in real choreographic practice, where professional choreographers first match music-aligned motions to performers, then design spatial formations.

In this work, we answer this question affirmatively by introducing a new formulation of group dance generation as a sequential conditional decomposition over dancers, which we term a Chain-of-Dancers. Rather than treating dancers as simultaneously generated tokens, we model each dancer as conditioned on previously generated ones, enabling flexible scaling to arbitrary group sizes while preserving consistent identity structure. This perspective shifts the problem from joint generation to structured composition, decoupling group size from model capacity.

![](images/1b454c8177c6f86ece2c583e94f692bf60770c3bcd3c787d2000317e6a99f0f0.jpg)  
(a) Intensive attention computation when retraining for different group sizes.

![](images/3ae8c6b0c34e012295a61481c3b2184754666bc4477acbac952a1c82c2ed8096.jpg)  
(b) Identity entanglement problem.  
Figure 1: Key challenges in group dance generation. (a) Group-size rigidity: adapting to a new group size N requires retraining from scratch, but self-attention over the concatenated sequence scales as $O ( ( N L ) ^ { 2 } )$ , making per-size retraining prohibitively expensive. As a result, existing models are locked to a fixed group size. (b) Identity entanglement: joint representations obscure perdancer identity, causing frequent swaps across frames. We visualize three consecutive frames from CoDancers [38]: between Frame 1 and Frame 2, the pink and blue dancers abruptly swap positions; in Frame 3, the blue dancer suddenly relocates to the leftmost position. When rendered as video, these inconsistencies appear as visible flickering and identity-switching artifacts.

This reformulation enables a single pretrained single-dancer model [9, 6, 25] to generalize to multidancer scenarios without retraining, while still capturing global coordination through explicit interdancer conditioning. Crucially, it allows identity to be preserved by construction, rather than enforced through architectural constraints.

Building on this formulation, we introduce a lightweight instantiation that injects role-specific semantics and inter-dancer interactions into a frozen backbone. We further incorporate a trainingfree inference procedure [24] to enforce global spatial coherence. Together, these design choices operationalize the proposed decomposition while maintaining parameter efficiency.

Experiments demonstrate that this reformulation achieves strong motion quality and coordination across varying group sizes, while significantly reducing training cost. More importantly, our results suggest that structured decomposition is a viable alternative to joint modeling for multi-agent motion generation, opening a new direction for scalable and controllable choreography synthesis. Our contributions can be summarized as follows:

• Reformulating group dance generation. We propose a Chain-of-Dancers formulation that models group choreography as a sequential conditional composition of individual dancers, shifting from joint generation to structured decomposition and addressing group-size rigidity and identity entanglement.

• Scalable and identity-preserving generation. Our approach decouples group size from model architecture, enabling a single pretrained model to generalize to variable numbers of dancers without retraining, while preserving each dancer’s identity by construction.

• Efficient coordination with minimal overhead. We instantiate the formulation with a Role-Aware Text Encoder, a Group-Aware Motion Encoder, and a training-free inference strategy for global coherence, achieving strong results with significantly reduced training cost.

## 2 Related Work

## 2.1 Music-Driven Single Dance Generation

Music-to-dance generation targets rhythmic and stylistically consistent motion synthesis [12, 30, 31]. Transitioning from rule-based heuristics [15, 13, 27] to deep learning on large datasets [18, 22, 20, 29], various architectures have emerged: RNNs and autoregressive models capture transitions but struggle with error accumulation [1, 36], while Transformers and GANs/VAEs improve long-range coherence and realism [32, 5, 18, 11, 8]. Currently, diffusion models define the state-of-the-art via audio-aligned denoising [33, 41]. However, most methods focus on music-only conditioning, limiting controllability, semantic expressiveness, and global coherence.

## 2.2 Group Dance Generation

Group dance generation requires spatial coordination, identity preservation, and semantic diversity across multiple performers [39, 33]. Many works adopt a joint modeling strategy that concatenates all dancer motions and processes them with transformer backbones, e.g., GDanceR [17], GCD [16], TCDiff [3], and CoDancers [38]. While capturing global synchronization, this design fixes the dancer dimension and requires retraining for each group size, blurs per-dancer identity through uniform tokenization, and typically offers only music-only conditioning, widening the semantic gap between music and motion [34, 18]. These limitations motivate frameworks that scale across group sizes without retraining, support joint music–text control, and explicitly preserve per-dancer identity. A complementary line of work models pairwise leader-follower interactions [29] but is restricted to fixed two-person generation. In contrast, we generalize this paradigm to variable group sizes via a shared bidirectional pair-wise model with graph-based group conditioning.

## 2.3 Multimodal Conditioning with Text and Music

Controlling human motion with semantic and rhythmic cues has been widely studied [34, 2]. Text-tomotion methods map language to coherent movements [40, 42], while music-conditioned models learn beat- and phase-aligned trajectories [44, 19, 22]. Recent works fuse text and audio via crossmodal encoders to couple semantic intent with temporal structure [37, 7], improving controllability over unimodal conditioning. However, simultaneous text-and-music control for single dance is still limited. Most models use music alone or coarse style tags rather than free-form language [33, 10]. The gap widens in group dance, where role-aware conditioning is rare due to scarce multi-person annotations [43]. Moreover, many multimodal pipelines emphasize global synchronization or average style, causing identity permutation and weak role differentiation in group dance settings. We address this gap by enabling coordinated group generation under both text and music while explicitly maintaining dancer-specific identity and role consistency.

## 3 Method

Given an input music clip M and an optional dance instruction T (style, dancer count, interaction type), our goal is to synthesize a spatiotemporal group dance $\{ \mathbf { X } _ { 1 : L } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , where $\mathbf { X } _ { l } ^ { ( i ) }$ denotes the motion of dancer i at time l.

Rather than modeling all dancers jointly, we introduce a Chain-of-Dancers formulation, where group choreography is constructed as a sequence of conditional single-dancer generations. This shifts group generation from joint modeling to structured composition, enabling scalable and identity-consistent synthesis.

This formulation introduces three key challenges: (a) how to inject inter-dancer dependencies into a single-dancer backbone, (b) how to support variable group sizes without retraining, and (c) how to enforce global coordination beyond local interactions.

We address these challenges through ChainDance (Figure 2), a unified framework that combines conditional generation (§3.1), pairwise decomposition (§3.2), and inference-time optimization (§3.3).

## 3.1 Multimodal Single-Dancer Generation and Inter-Dancer Interaction Modeling

To realize the proposed decomposition, each dancer must be generated conditionally on both global instructions and previously generated dancers. We achieve this by augmenting a pretrained singledancer diffusion model with two types of conditioning: (a) semantic conditioning from text (roles, interactions, style), and (b) relational conditioning from previously generated dancers.

Rather than modifying the backbone architecture, we inject these signals through lightweight conditioning pathways, allowing the model to remain fixed while adapting to group scenarios. This design ensures that each dancer retains individual identity, and coordination emerges through explicit conditioning, not shared latent entanglement.

Textual Instruction Generation. As shown in Figure 3, we first merge structured multimodal metadata (music captions such as tempo and genre, motion captions such as style and interaction type) into a global choreography prompt. We then use an LLM (ChatGPT-4o) to generate a textual dance instruction T that aligns with the music M. This global instruction is decomposed into individualized instructions based on the pairing strategy. Each dancer pair is assigned an interaction keyword (e.g., approach, keep distance, face to, lead follow, swap, passby) to encourage diverse yet coherent dancer-aware behaviors. Existing group dance datasets provide high-quality kinematic trajectories but lack fine-grained textual annotations of inter-dancer interactions. We bridge this gap by automatically deriving interaction keywords from motion samples via geometrically-grounded rule-based detection (Appendix C), serving as approximate yet effective weak supervision (Table 3, w/o RATE).

![](images/ab7b179eae7333d838309f5a5db1a7762caafce9705d866af35f9e29727ae2ac.jpg)  
Figure 2: Overall framework of ChainDance. The model builds on a frozen single dance diffusion backbone and introduces lightweight multimodal encoders for text, music, and other dancers. Through sequential chain generation and inference-time global optimization, ChainDance produces coherent and coordinated group choreographies.

Lightweight Multimodal Encoders. While the decomposition reduces complexity, coordination still requires modeling interactions between dancers. We capture these interactions through two complementary mechanisms: (a) textual interaction cues, which specify roles and interaction types (e.g., approach, follow, keep distance), and (b) motion-based relational context, derived from previously generated dancers. These signals provide structured guidance for coordination, and importantly, interaction modeling is explicit and compositional, rather than implicitly entangled in a joint representation.

To realize these two mechanisms, two parameter-efficient encoders are inserted into the cross-attention layers of the frozen single-dancer diffusion backbone B. Pretrained single-dancer models condition only on a global instruction, whereas group choreography requires each dancer to carry role-specific information (e.g., who leads, who interacts). To bridge this gap, the Role-Aware Text Encoder (RATE) encodes dancer-aware instructions using a CLIP text encoder [26], projects them through a low-rank LoRA module and a lightweight Transformer, and provides them as $( K _ { t e x t } , V _ { t e x t } )$ in a cross-attention computation:

$$
\mathrm { A t t n } \big ( \mathbf { H } ^ { ( i ) } \big ) = \mathrm { s o f t m a x } \left( \frac { \mathbf { H } ^ { ( i ) } K _ { t e x t } ^ { \top } } { \sqrt { d } } \right) V _ { t e x t } ,\tag{1}
$$

![](images/4b586374ceb48c00d316d5b65fd005401c0d8316832138c6a2d2404ce5a9fa7c.jpg)  
Figure 3: Prompt generation and role-specific conditioning. The system converts structured multimodal captions into individualized text prompts for each dancer. Based on a global choreography description, dancer pairs are randomly sampled and assigned interaction keywords (e.g., approach, keep distance, face to, lead follow, swap, or passby), providing diverse yet coherent textual conditions for ChainDance.

![](images/fc26282641de4248807b3c58ea9216aa70714bc8dd24f19d742cf86f17d3f1bd.jpg)  
Figure 4: Chain-of-Dancers generation design. At inference, ChainDance generates dancers sequentially and refines global consistency via diffusion-based optimization. Each dancer is conditioned on the previously generated one, and this process repeats until the whole group is complete. After that, optimize the shared initial noise via gradient descent until the global penalty L converges below the target threshold. All pairs share the weights of the text and motion encoders.

where $\mathbf { H } ^ { ( i ) }$ denotes the hidden state of dancer i. In this way, style and role information are injected into text embeddings, which are then incorporated into the encoder output (Figure 2) to enable fine-grained semantic control.

The Group-Aware Motion Encoder (GAME) conditions on previously generated dancers by constructing a distance-weighted graph G over dancers, where edge weight $w _ { i j }$ encodes spatial proximity or interaction strength. A GCN [14, 23, 4] aggregates neighbor features, refined by a Transformer with LoRA updates, to produce motion-aware $( K _ { g r p } , V _ { g r p } )$ fused with $\mathbf { H } ^ { ( i ) }$ through cross-attention (see Appendix B.1 and B.2). This captures local pairwise dependencies. The Role-Aware Text Encoder provides semantic cues via $( K _ { t e x t } , V _ { t e x t } )$ from textual prompts, while the Group-Aware Motion Encoder introduces spatial context through $( K _ { g r p } , V _ { g r p } )$ from previously generated dancer $X _ { j }$

All these (K, V) pairs are fed into the multimodal encoders together with the music condition. The encoder outputs are fused (through concatenation and projection) into key-value pairs $( K _ { f u s e } , V _ { f u s e } )$ that are applied to the cross-attention layers in the frozen generation model. Through this mechanism, the hidden state of dancer i integrates textual semantics, music, and prior motions for coherent and coordinated choreography. Together, these encoders transform the single-dancer diffusion model into a multimodal generator capable of synchronized and identity-consistent group choreography.

## 3.2 Pair-wise Decomposition for Scalable Generation

A key consequence of the Chain-of-Dancers formulation is that modeling full joint dependencies is unnecessary. Instead, we approximate group choreography through pairwise conditional generation. Concretely, we decompose the generation process into leader–follower pairs, where each dancer is conditioned on another dancer. A single shared model is trained on such pairs and reused across all positions in the chain. At inference time, group choreography is constructed sequentially: each subsequent dancer is conditioned on previously generated ones, and the same model is reused for all steps. This enables: (a) scalability to varying group sizes and (b) no retraining when adding new dancers.

Pair-wise training. Given a group of N dancers, we decompose the joint distribution into pairwise conditionals via the chain rule (see Appendix A):

$$
P ( X _ { 1 } , \dots , X _ { N } ) = P ( X _ { 1 } ) \prod _ { i = 2 } ^ { N } P ( X _ { i } \mid X _ { 1 : i - 1 } ) .\tag{2}
$$

We approximate each conditional $P ( X _ { i } \mid X _ { 1 : i - 1 } )$ with a shared pair-wise model: each training sample is a pair $( X _ { j } , X _ { i } )$ , where $X _ { j }$ is the conditioning leader and $X _ { i }$ is the follower to be denoised. The same RATE and GAME modules are shared across all pairs, so adding a new dancer at inference adds no learnable parameters.

To avoid directional bias, we adopt bidirectional pairing, allowing the model to learn symmetric interactions rather than fixed roles: both $( i , j )$ and $( j , i )$ samples are included, allowing the model to learn mutual conditioning rather than a fixed leader–follower direction. Two pair sampling strategies are explored:

i. Chain pairing: pairs follow consecutive dataset indices, $\mathrm { e . g . , } ( D _ { 1 } , D _ { 2 } ) , ( D _ { 2 } , D _ { 3 } )$ . Indices reflect data order, not spatial position.

ii. Random pairing: two dancers are uniformly sampled from the same group.

Both yield comparable results in our experiments, and we use chain pairing as the default. The training loss combines a reconstruction term with four auxiliaries:

$$
{ \mathcal { L } } _ { \mathrm { l o c a l } } = { \mathcal { L } } _ { \mathrm { s i m p l e } } + \lambda _ { \mathrm { p o s } } { \mathcal { L } } _ { \mathrm { p o s } } + \lambda _ { \mathrm { v e l } } { \mathcal { L } } _ { \mathrm { v e l } } + \lambda _ { \mathrm { c o n t a c t } } { \mathcal { L } } _ { \mathrm { c o n t a c t } } + \lambda _ { \mathrm { a l i g n } } { \mathcal { L } } _ { \mathrm { a l i g n } } .\tag{3}
$$

$\mathcal { L } _ { \mathrm { s i m p l e } }$ is the reconstruction loss; $\mathcal { L } _ { \mathrm { p o s } } , \mathcal { L } _ { \mathrm { v e l } } , \mathcal { L } _ { \mathrm { c o n t a c t } }$ ensure pose accuracy, smooth transitions, and physical foot–ground contact; $\mathcal { L } _ { \mathrm { a l i g n } }$ aligns text–music–motion embeddings via cosine similarity. Only RATE and GAME are trainable; the diffusion backbone B remains frozen.

Inference: from pair to chain. At test time, the same model generates groups of variable size N through sequential conditioning (Figure 4). Starting from $X _ { 1 }$ produced by the frozen single-dancer backbone, we iteratively generate dancer $X _ { i }$ by feeding the most recent leader $X _ { j }$ into $\mathrm { { G A M E } , }$ which constructs a distance-weighted graph over previously generated dancers to compute spatial weights, then aggregates this context onto the leader feature to produce $( K _ { g r p } , V _ { g r p } )$ ; RATE simultaneously injects the role-specific text prompt for dancer $i .$ The pair-wise model naturally generalizes to chained generation, since each step reuses the same shared encoders and the group-aware motion graph grows incrementally as new dancers are added (see Appendix B.1). Algorithm details see Appendix B.2.

## 3.3 Global Consistency via Inference-Time Optimization

Sequential generation captures local dependencies but does not guarantee global coherence, such as formation structure or collision avoidance among non-adjacent dancers. Instead of increasing model complexity, we enforce global consistency through a training-free inference-time optimization. Specifically, we refine the shared latent variables to minimize a global objective encoding spatial formation constraints, inter-dancer distance relationships, and collision avoidance. This optimization complements the sequential generation process, enabling global coordination without retraining or architectural changes.

Following the principle of PINO [24], we keep all network parameters frozen and instead optimize the shared diffusion noise via gradient descent, backpropagating a global penalty through the diffusion sampling process. Crucially, because all dancers are generated from a shared latent variable, this optimization provides a unified control signal that simultaneously adjusts all dancers, enabling grouplevel coordination that cannot be achieved by per-dancer generation alone (see Table 8). The test-time objective decomposes into global controls and overlap penalties,

$$
L = L _ { \mathrm { g l o b a l } } \ + \ L _ { \mathrm { o v e r l a p } } ,\tag{4}
$$

where $L _ { \mathrm { g l o b a l } }$ encodes formation and orientation constraints, and $L _ { \mathrm { o v e r l a p } }$ discourages inter-person penetration. The global part is a weighted sum of differentiable controls:

$$
L _ { \mathrm { g l o b a l } } = \sum _ { k \in \mathcal { S } } \lambda _ { k } L _ { k } ,\tag{5}
$$

where $s$ is the subset of enabled controls $( \mathrm { e . g . }$ , root position, movement region, orientation, relative distance), chosen to reflect group choreography constraints. The overlap term is implemented as a hinge on pairwise root distances:

$$
L _ { \mathrm { o v e r l a p } } = \frac { 1 } { T } \sum _ { t } \sum _ { i < j } \operatorname* { m a x } \Bigl ( 0 , \ : \delta - \left\| p _ { \mathrm { r o o t } } ^ { ( i ) } ( t ) - p _ { \mathrm { r o o t } } ^ { ( j ) } ( t ) \right\| _ { 2 } \Bigr ) .\tag{6}
$$

At inference, we update only the shared latent to reduce Eqn. (4), which adds a small latency overhead while improving interaction quality.

## 4 Experimental Results

We evaluate ChainDance in standard single- and group-dance settings. Unless stated otherwise, we freeze the backbone and train only RATE and GAME. We report quality, physical plausibility, group coherence, and efficiency, along with ablation studies.

## 4.1 Experimental Setup

Baseline Methods. We compare ChainDance with other group dance methods that generate all dancers jointly as high-capacity baselines: GCD [16], which integrates group-based contrastive learning with diffusion; CoDancers [38], which employs music-aligned choreographic units; TCDiff [3], which incorporates trajectory control for temporal modeling; and ST-GDance [35], which factorizes temporal and spatial modeling to preserve layouts while avoiding parameter blow-up at scale. All methods share the same music inputs, beat extraction, and evaluation protocols. Each baseline is retrained per group size, while our model is trained once and generalizes to varying group sizes.

Datasets. We pretrain our single-dancer backbones on two standard single dance datasets: AIST++ [22] and FineDance [20]. AIST++ contains 1,408 paired 3D music–dance sequences across 10 genres, performed by 60 professional dancers. FineDance builds upon AIST++ with more diverse styles and longer clips, offering broader coverage of musical and motion patterns. All evaluations are conducted on AIOZ-GDance [17], the only publicly available large-scale group dance benchmark comprising 16.7 hours of music-aligned 3D motion data from over 4,000 dancers, spanning 7 dance styles and 16 music genres. We follow the official train/test split protocol from [17].

Evaluation Metrics. Following standard practice [3, 35], we evaluate both single- and group-level performance, along with model efficiency:

• Single dance metrics. Frechet Inception Distance (FID, ↓) assesses distribution fidelity, Generation Diversity (Div, ↑) quantifies sample diversity, and Physical Foot Contact (PFC, ↓) measures foot sliding artifacts.

• Group dance metrics. Group Motion Realism (GMR, ↓) measures feature similarity via Frechet Inception Distance; Group Motion Correlation (GMC, ↑) evaluates coherence through crosscorrelation between generated dancers; Trajectory Intersection Frequency (TIF, ↓) counts the frequency of inter-person body-part collisions.

• Efficiency. We report FLOPs, parameter count, training time per epoch, and inference time.

Training Strategy. We freeze the pretrained backbone and train the two lightweight LoRA encoders. We adopt a teacher forcing strategy during training, where each follower conditions on the groundtruth motion of previously generated dancers, perturbed with noise to reduce the train-inference gap. At inference, the model operates auto-regressively, conditioning on its own predictions. We optimize using the Adam optimizer with cosine learning rate decay.

Implementation Details. We train our model on a dual RTX 4090 GPU setup. ChainDance uses the Lodge [21] backbone pretrained on FineDance and employs a chain pairing strategy with bidirectional pairs during training for the main results. Following previous works [3, 34], the loss function includes weighted terms for position $( \lambda _ { \mathrm { p o s } } = 0 . 6 4 6 )$ , velocity $( \lambda _ { \mathrm { v e l } } = 2 . 9 6 4 )$ , and contact $( \lambda _ { \mathrm { c o n t a c t } } = 1 0 . 9 4 2 )$ to model motion dynamics, along with a cross-modal alignment loss $( \lambda _ { \mathrm { a l i g n } } = 1 . 0 )$ to ensure consistency between the generated motion and the input text/music. For all group sizes, the sequence length is set to 120 frames. Our model was trained for 1000 epochs using the Adam optimizer with an initial learning rate of 0.001 and a cosine learning rate scheduler. At inference, we adopt a conservative KV cache that reuses final-projection key–value states across dancers and maintains a causal temporal cache to avoid recomputing past frames. Code will be released upon acceptance.

Table 1: Single and Group dance performance on AIOZ-GDance. Best results are bold; secondbest are underlined. \* denotes models without the global noise optimization penalty.
<table><tr><td rowspan="2">Method</td><td colspan="3">1 Dancer</td><td colspan="3">2 Dancers</td><td colspan="3">3 Dancers</td><td colspan="3">4 Dancers</td></tr><tr><td>FID↓</td><td>Div↑</td><td>PFC↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>GCD [16]</td><td>39.24</td><td>9.64</td><td>2.53</td><td>34.39</td><td>80.32</td><td>0.17</td><td>30.22</td><td>80.22</td><td>0.19</td><td>36.28</td><td>81.82</td><td>0.13</td></tr><tr><td>CoDancers [38]</td><td>23.98</td><td>9.48</td><td>3.53</td><td>24.55</td><td>72.52</td><td>0.08</td><td>26.34</td><td>74.22</td><td>0.08</td><td>26.44</td><td>75.34</td><td>0.10</td></tr><tr><td>TCDiff [3]</td><td>37.31</td><td>14.01</td><td>0.51</td><td>15.77</td><td>81.92</td><td>0.12</td><td>15.36</td><td>82.77</td><td>0.11</td><td>13.44</td><td>81.70</td><td>0.15</td></tr><tr><td>ST-GDance [35]</td><td>28.87</td><td>12.82</td><td>0.97</td><td>19.42</td><td>80.52</td><td>0.12</td><td>14.76</td><td>81.24</td><td>0.11</td><td>14.02</td><td>81.42</td><td>0.10</td></tr><tr><td>ChainDance*</td><td>23.12</td><td>13.85</td><td>0.50</td><td>14.10</td><td>81.52</td><td>0.11</td><td>12.44</td><td>82.32</td><td>0.11</td><td>13.28</td><td>82.13</td><td>0.12</td></tr><tr><td>ChainDance</td><td>23.12</td><td>13.85</td><td>0.50</td><td>13.67</td><td>81.28</td><td>0.12</td><td>11.63</td><td>83.74</td><td>0.10</td><td>12.59</td><td>82.16</td><td>0.13</td></tr></table>

## 4.2 Quantitative Comparisons Against Baseline Models

Single Dance (1 Dancer). As shown in Table 1, our model achieves the best overall performance in single-dancer generation, reaching the lowest FID (23.12) and PFC (0.50) and the second-highest diversity (13.85). This indicates superior motion quality and stronger music alignment compared to ST-GDance [35] and TCDiff [3]. These results also confirm that the proposed multimodal encoders effectively enhance motion quality and controllability under text–music supervision, without compromising the fidelity of the pretrained single-dancer backbone.

Group Dance (2–4 Dancers). For multi-dancer choreography, ChainDance achieves the best GMR↓, indicating the highest motion realism through improved feature distribution matching. It also achieves the best GMC↑ at three and four dancers, demonstrating well-balanced group dance quality. Specifically, for three dancers, our model reduces GMR to 11.63 and achieves the highest GMC (83.74), indicating superior coordination and spatial harmony. Even as the number of dancers increases to four, ChainDance maintains more coherent and rhythmic stability, outperforming all baselines with lower GMR and higher GMC values. Moreover, the variant without the global optimization penalty still outperforms other approaches on most metrics, demonstrating that the core chain-of-dancers design and pairwise training strategy are sufficient to ensure strong inter-dancer coordination. These results validate the model’s ability to generalize to variable group sizes, supporting efficient and coherent choreography generation from single to multi-person scenarios. To mitigate data confounds, Appendix D.5 reports ChainDance pretrained solely on AIOZ-GDance, which remains competitive (Table 9). Visualizations for larger groups are available on the supplementary webpage.

Computational Efficiency. In Table 2, our model remains compact and trainable, evidenced by the second lowest FLOPs (16.92G), the fewest parameters (15.27M), and the fastest training time per epoch (0:14). Without the optional test-time refinement, inference reduces from 0:15 to 0:06, comparable to the fastest baselines while preserving generation quality. This overhead is adjustable (fewer optimization steps trade a small amount of accuracy for speed; see appendix) and requires no retraining. In summary, ChainDance has a small model size, fast training, and produces high-quality group dances with controllable inference.

Table 2: Efficiency comparison (3 dancers group). Best results are bold; second-best are underlined. \* denotes models without the global noise optimization penalty.
<table><tr><td>Method</td><td>FLOPs (G)↓</td><td>Params (M)↓</td><td>Train/epoch (min:sec)↓</td><td>Inf Time (min:sec)↓</td></tr><tr><td>GCD [16]</td><td>27.69</td><td>62.16</td><td>1:04</td><td>0:05</td></tr><tr><td>Codancers [38]</td><td>58.95</td><td>59.32</td><td>0:42</td><td>0:02</td></tr><tr><td>TCDiff [3]</td><td>22.04</td><td>62.48</td><td>1:29</td><td>0:08</td></tr><tr><td>ST-GDance [35]</td><td>6.75</td><td>50.21</td><td>0:37</td><td>0:03</td></tr><tr><td>ChainDance*</td><td>16.92</td><td>15.27</td><td>0:14</td><td>0:06</td></tr><tr><td>ChainDance</td><td>16.92</td><td>15.27</td><td>0:14</td><td>0:15</td></tr></table>

Controllability and User Study. We report quantitative controllability evaluation across 10 music genres in Appendix D.8 and a 30-participant user study in Appendix D.4.

## 4.3 Qualitative Results

![](images/119b02afc347ce80b5d1220291f83b1f679e91b2cd45cc4a763e9c347ff5d52c.jpg)  
Figure 5: Qualitative results of generated 3-dancers group sequence compared with baselines.

Figure 5 compares dance sequences generated by our method, GCD, and ST-GDance under the same clip length and frame stride. ST-GDance exhibits frequent apparent “position swaps” between dancers, which do not reflect true choreographic exchanges but arise from the identity-shift issue, where global motion remains plausible while per-dancer temporal continuity is lost. Meanwhile, GCD shows interpenetration artifacts (red boxes), where the rightmost dancer’s foot penetrates the ground plane, indicating weak physical plausibility. In contrast, ChainDance maintains stable identities and consistent spatial relations, producing collision-free trajectories and realistic ground contact throughout the sequence. This directly supports the identity preservation claim: while joint modeling architecturally permits dancer-feature swaps, ChainDance’s sequential decomposition structurally precludes them, since each dancer is an independent conditional sample with no shared feature dimensions. More qualitative videos are on the supplementary webpage.

## 4.4 Ablation Study

Ablation Settings. To better understand the contribution of each component in our framework, we conduct ablation studies on the group dance generation task using the Lodge backbone pretrained on FineDance. We progressively remove or disable key modules to isolate their impact.

w/o GAME: The follower dancers are generated independently without conditioning on the motion of previously generated dancers. This removes cross-person interaction modeling.

w/o RATE: The system operates without semantic dance-style guidance, relying solely on music and motion context.

w/o Global Penalty: We remove the global motion-level consistency penalty at inference, which is designed to prevent inter-person collisions and encourage coherent global formation.

The sequence length is set to 120 frames. All models are trained and evaluated following the same protocol, and group-level metrics are computed on group dance sequences from AIOZ-GDance.

Results. As summarized in Table 3, removing the group-aware motion encoder (w/o GAME) leads to the most pronounced degradation (GMR 17.46, GMC 79.14), highlighting the importance of modeling motion dependencies among dancers. Disabling the role-aware text prompt (w/o RATE) substantially degrades both GMR (from 11.63 to 14.32) and GMC (from 83.74 to 81.98), suggesting textual guidance improves both stylistic synchronization and spatial coordination; beyond raw metrics, RATE also provides a human-interpretable interface for per-dancer

Table 3: Ablation study for group dance generation (3 dancers). Results are obtained on the Lodge backbone pretrained on FineDance. Bold indicates the best result.
<table><tr><td>Method</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>w/o GAME</td><td>17.46</td><td>79.14</td><td>0.19</td></tr><tr><td>w/o RATE</td><td>14.32</td><td>81.98</td><td>0.11</td></tr><tr><td>w/o Global Penalty</td><td>12.44</td><td>82.32</td><td>0.11</td></tr><tr><td>ChainDance</td><td>11.63</td><td>83.74</td><td>0.10</td></tr></table>

control (Appendix D.8). Removing the global penalty (w/o Global Penalty) leads to moderate degradation across all metrics, confirming its role in inter-person coherence. The full model achieves the best GMR (11.63), GMC (83.74), and TIF (0.10), demonstrating complementary component effects.

## 4.5 Different Backbones and Pretraining Datasets

Table 4: Comparison across backbones and pretraining datasets. Best results are bold, and second-best are underlined.
<table><tr><td rowspan="3">Method</td><td colspan="6">w/o Global Optimization</td><td rowspan="2"></td><td colspan="6">with Global Optimization</td></tr><tr><td colspan="3">Single-dance</td><td colspan="3">Group-dance</td><td colspan="3">Single-dance</td><td colspan="3">Group-dance</td></tr><tr><td>FID↓</td><td>Div↑</td><td>PFC↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>FID↓</td><td>Div↑</td><td></td><td>PFC↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>EDGE (AIST++)</td><td>24.74</td><td>12.98</td><td>1.04</td><td>15.23</td><td></td><td>80.21</td><td>0.13</td><td>24.74</td><td>12.98</td><td>1.04</td><td>12.32</td><td>82.49</td><td>0.11</td></tr><tr><td>EDGE (FineDance)</td><td>26.12</td><td>13.34</td><td>0.62</td><td>14.93</td><td>80.73</td><td>0.12</td><td>26.12</td><td>13.34</td><td></td><td>0.62</td><td>12.25</td><td>82.65</td><td>0.10</td></tr><tr><td>Lodge (AIST++)</td><td>27.31</td><td>14.41</td><td>1.01</td><td>14.63</td><td>80.99</td><td>0.12</td><td>27.31</td><td></td><td>14.41</td><td>1.01</td><td>11.69</td><td>83.30</td><td>0.11</td></tr><tr><td>Lodge (FineDance)</td><td>23.12</td><td>13.85</td><td>0.50</td><td>12.44</td><td>82.32</td><td>0.11</td><td>23.12</td><td></td><td>13.85</td><td>0.50</td><td>11.63</td><td>83.74</td><td>0.10</td></tr></table>

Table 4 summarizes results across different backbones and pretraining datasets. The global noise optimization defines penalty rules over inter-dancer interactions and therefore does not affect singledance generation. Overall, adding this global penalty improves group dance (3 dancers) quality in most settings, particularly for GMR and GMC. For example, with the Lodge backbone on FineDance, GMR decreases from 12.44 to 11.63 and GMC increases from 82.32 to 83.74, showing better spatial coordination and rhythm alignment. The EDGE backbone is more efficient but produces slightly lower quality than Lodge. Models trained on FineDance outperform those trained on AIST++, likely due to FineDance’s broader style coverage and richer motion variety. These results show that ChainDance successfully adapts single-dancer models for stable multi-person generation across different architectures and datasets.

## 5 Conclusion

We presented ChainDance, a parameter-efficient framework that scales group dance generation by reformulating the task as a Chain-of-Dancers. By adapting a frozen backbone with two lightweight modules, a Role-Aware Text Encoder and a Group-Aware Motion Encoder, our model synthesizes variable group sizes through pair-wise conditioning and inference-time noise optimization. On AIOZ-GDance, ChainDance achieves state-of-the-art motion quality while structurally preserving each dancer’s identity, using 3-4× fewer parameters and 3-6× less training time. More broadly, our results suggest that structured decomposition is a viable alternative to joint modeling for multi-agent motion generation.

## Acknowledgments and Disclosure of Funding

Funding. This research was supported by the Australian Government through the Australian Research Council Discovery Early Career Researcher Award (DE250100030) and Discovery Project (DP260100218). Qiuhong Ke is the recipient of the Discovery Early Career Researcher Award (DE250100030).

Competing interests. The authors declare no competing interests.

## References

[1] Omid Alemi, Jules Françoise, and Philippe Pasquier. GrooveNet: Real-time music-driven dance movement generation using artificial neural networks. In Proceedings of the 4th International Workshop on Machine Learningfor Creativity (ML4Creativity), ACM SIGKDD, Halifax, Nova Scotia, Canada, 2017.

[2] Lele Chen, Guofeng Cui, Celong Liu, Zhong Li, Ziyi Kou, Yi Xu, and Chenliang Xu. Talkinghead generation with rhythmic head motion. In European conference on computer vision, pages 35–51. Springer, 2020.

[3] Yuqin Dai, Wanlu Zhu, Ronghui Li, Zeping Ren, Xiangzheng Zhou, Xiu Li, Jun Li, and Jian Yang. Harmonious group choreography with trajectory-controllable diffusion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2025.

[4] Sihao Ding, Fuli Feng, Xiangnan He, Yong Liao, Jun Shi, and Yongdong Zhang. Causal incremental graph convolution for recommender system retraining. IEEE Transactions on Neural Networks and Learning Systems, 35(4):4718–4728, 2022.

[5] Joao P Ferreira, Thiago M Coutinho, Thiago L Gomes, José F Neto, Rafael Azevedo, Renato Martins, and Erickson R Nascimento. Learning to dance: A graph convolutional adversarial network to generate realistic dance motions from audio. Computers & Graphics, 94:11–21, 2021.

[6] Anindita Ghosh, Bing Zhou, Rishabh Dabral, Jian Wang, Vladislav Golyanik, Christian Theobalt, Philipp Slusallek, and Chuan Guo. Duetgen: Music driven two-person dance generation via hierarchical masked modeling. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.

[7] Kehong Gong, Dongze Lian, Heng Chang, Chuan Guo, Zihang Jiang, Xinxin Zuo, Michael Bi Mi, and Xinchao Wang. TM2D: Bimodality driven 3d dance generation via music-text integration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9942–9952, 2023.

[8] Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. AvatarCLIP: Zero-shot text-driven generation and animation of 3d avatars. ACM Transactions on Graphics (TOG), 2022.

[9] Dongjin Huang, Yue Zhang, Zhenyan Li, and Jinhua Liu. TG-Dance: Transgan-based intelligent dance generation with music. In International Conference on Multimedia Modeling, pages 243–254. Springer, 2023.

[10] Ruozi Huang, Huang Hu, Wei Wu, Kei Sawada, Mi Zhang, and Daxin Jiang. Dance revolution: Long-term dance generation with music via curriculum learning. In International conference on learning representations, 2020.

[11] Yin-Fu Huang and Wei-De Liu. Choreography cgan: generating dances with music beats using conditional generative adversarial networks. Neural Computing and Applications, 33(16): 9817–9833, 2021.

[12] Manish Joshi and Sangeeta Chakrabarty. An extensive review of computational dance automation techniques and applications. Proceedings of the Royal Society A, 477(2251):20210071, 2021.

[13] Tae-hoon Kim, Sang Il Park, and Sung Yong Shin. Rhythmic-motion synthesis based on motion-beat analysis. ACM Transactions on Graphics (TOG), 22(3):392–401, 2003.

[14] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

[15] Lucas Kovar, Michael Gleicher, and Frédéric Pighin. Motion graphs. In Proceedings of SIGGRAPH, 2002.

[16] Nhat Le, Tuong Do, Khoa Do, Hien Nguyen, Erman Tjiputra, Quang D Tran, and Anh Nguyen. Controllable group choreography using contrastive diffusion. ACM Transactions on Graphics (TOG), 42(6):1–14, 2023.

[17] Nhat Le, Thang Pham, Tuong Do, Erman Tjiputra, Quang D Tran, and Anh Nguyen. Musicdriven group choreography. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8673–8682, 2023.

[18] Hsin-Ying Lee, Xiaodong Yang, Ming-Yu Liu, Ting-Chun Wang, Yu-Ding Lu, Ming-Hsuan Yang, and Jan Kautz. Dancing to music. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[19] Buyu Li, Yongchi Zhao, Shi Zhelun, and Lu Sheng. Danceformer: Music conditioned 3d dance generation with parametric motion transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1272–1279, 2022.

[20] Ronghui Li, Junfan Zhao, Yachao Zhang, Mingyang Su, Zeping Ren, Han Zhang, Yansong Tang, and Xiu Li. Finedance: A fine-grained choreography dataset for 3d full body dance generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10234–10243, 2023.

[21] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1524–1534, 2024.

[22] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. AI choreographer: Music conditioned 3d dance generation with aist++. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 13401–13412, 2021.

[23] Weizhi Nie, Rihao Chang, Minjie Ren, Yuting Su, and Anan Liu. I-GCN: Incremental graph convolution network for conversation emotion detection. IEEE Transactions on Multimedia, 24: 4471–4481, 2021.

[24] Sakuya Ota, Qing Yu, Kent Fujiwara, Satoshi Ikehata, and Ikuro Sato. Pino: Person-interaction noise optimization for long-duration and customizable motion generation of arbitrary-sized groups. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10676–10685, 2025.

[25] Qiaosong Qi, Le Zhuo, Aixi Zhang, Yue Liao, Fei Fang, Si Liu, and Shuicheng Yan. Diffdance: Cascaded human motion diffusion model for dance generation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1374–1382, 2023.

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[27] Alla Safonova and Jessica K Hodgins. Construction and optimal search of interpolated motion graphs. In Proceedings ofACM SIGGRAPH, pages 106–es, 2007.

[28] Judith L Schwartz. The passacaille in Lully’s “armide”: Phrase structure in the choreography and the music. Early Music, 26(2):301–320, 1998.

[29] Li Siyao, Tianpei Gu, Zhitao Yang, Zhengyu Lin, Ziwei Liu, Henghui Ding, Lei Yang, and Chen Change Loy. Duolando: Follower gpt with off-policy reinforcement learning for dance accompaniment. In The Twelfth International Conference on Learning Representations.

[30] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando: 3d dance generation by actor-critic gpt with choreographic memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11050–11059, 2022.

[31] Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, and Ziwei Liu. Bailando++: 3d dance gpt with choreographic memory. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(12):14192–14207, 2023. doi: 10.1109/TPAMI. 2023.3319435.

[32] Guofei Sun, Yongkang Wong, Zhiyong Cheng, Mohan S Kankanhalli, Weidong Geng, and Xiangdong Li. Deepdance: music-to-dance motion choreography with adversarial learning. IEEE Transactions on Multimedia, 23:497–509, 2020.

[33] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 448–458, 2023.

[34] Qing Wang, Xiaohang Yang, Yilan Dong, Naveen Raj Govindaraj, Gregory Slabaugh, and Shanxin Yuan. Dancechat: Large language model-guided music-to-dance generation. arXiv preprint arXiv:2506.10574, 2025.

[35] Jing Xu, Weiqiang Wang, Cunjian Chen, Jun Liu, and Qiuhong Ke. St-gdance: Long-term and collision-free group choreography from music. In Proceedings of the British Machine Vision Conference (BMVC). BMVA Press, 2025.

[36] Nelson Yalta, Shinji Watanabe, Kazuhiro Nakadai, and Tetsuya Ogata. Weakly-supervised deep recurrent neural networks for basic dance step generation. In 2019 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2019.

[37] Han Yang, Kun Su, Yutong Zhang, Jiaben Chen, Kaizhi Qian, Gaowen Liu, and Chuang Gan. Unimumo: Unified text, music, and motion generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25615–25623, 2025.

[38] Kaixing Yang, Xulong Tang, Ran Diao, Hongyan Liu, Jun He, and Zhaoxin Fan. Codancers: Music-driven coherent group dance generation with choreographic unit. In Proceedings of the 2024 International Conference on Multimedia Retrieval, pages 675–683, 2024.

[39] Siyue Yao, Mingjie Sun, Bingliang Li, Fengyu Yang, Junle Wang, and Ruimao Zhang. Dance with you: The diversity controllable dancer generation via diffusion models. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8504–8514, 2023.

[40] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8466–8476, 2024.

[41] Wenjie Yin, Hang Yin, Kim Baraka, Danica Kragic, and Mårten Björkman. Dance style transfer with cross-modal transformer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5058–5067, 2023.

[42] Ling-An Zeng, Guohong Huang, Gaojie Wu, and Wei-Shi Zheng. Light-t2m: A lightweight and fast model for text-to-motion generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9797–9805, 2025.

[43] Zixiang Zhou, Yu Wan, and Baoyuan Wang. AvatarGPT: All-in-one framework for motion understanding planning generation and beyond. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1357–1366, 2024.

[44] Wenlin Zhuang, Congyi Wang, Jinxiang Chai, Yangang Wang, Ming Shao, and Siyu Xia. Music2dance: Dancenet for music-driven dance generation. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 18(2):1–21, 2022.

## A Prior Assumptions: Joint Distribution Perspective on Group Dance Generation

From a probabilistic modeling perspective, generative models fundamentally learn conditional probability distributions over data. For the group dance generation task considered in this work, let $\overset { \cdot } { \chi } _ { 1 : N } = \{ \overset { \cdot } { X } _ { 1 } , \overset { \cdot } { X } _ { 2 } , \dots , \overset { \cdot } { X } _ { N } \}$ denote a set of dance motion sequences performed by N dancers, where each $X _ { i }$ represents the full temporal motion trajectory of the i-th dancer. The learning objective of group dance generation can thus be formulated as modeling the joint distribution $P ( \mathcal { X } _ { 1 : N } )$ ).

According to the chain rule of probability, the joint distribution can be exactly factorized as:

$$
P ( \mathcal X _ { 1 : N } ) = P ( X _ { 1 } ) \prod _ { i = 2 } ^ { N } P ( X _ { i } \mid \mathcal X _ { 1 : i - 1 } ) ,\tag{7}
$$

which is a mathematically equivalent decomposition and does not introduce any approximation.

Our approach is directly motivated by this factorization. We first leverage a well-trained and stable single-dancer diffusion model to capture the marginal distribution $P ( X _ { 1 } )$ , providing strong temporal dynamics and motion priors. For each subsequent dancer $X _ { i } ,$ we model the conditional distribution $P ( X _ { i } \mid \mathcal { X } _ { 1 : i - 1 } )$ via conditional generation, where the conditioning signals consist of the motions of previously generated dancers together with multimodal control inputs.

This formulation offers two key advantages. First, temporal continuity, rhythm, and motion realism— already well captured by the single-dance generator—do not need to be relearned in the group setting. Second, inter-dancer interactions and spatial constraints are explicitly introduced through conditional dependencies during generation, avoiding the need to directly model the full high-dimensional joint distribution in a single step.

Therefore, from the perspective of joint probability modeling, the proposed sequential conditional generation strategy preserves the expressiveness of the target group dance distribution while providing a structured and scalable approach that improves training stability and generation robustness.

## B Implementation Details

This section provides the detailed formulations of graph construction, all loss terms and encoder details related to ChainDance. Each loss term corresponds to either the local reconstruction stage or the global refinement stage described in Section 3 of the main paper.

## B.1 Group-Aware Motion Graph

The Group-Aware Motion Encoder (Section 3) is implemented as a graph convolutional network operating on a distance-weighted dynamic graph that grows incrementally as new dancers are generated.

Distance-weighted edges. For the i-th dancer (node) being generated, we define raw interaction weights $w _ { i j }$ with all previously generated dancers j based on their Euclidean distance:

$$
w _ { i j } = \frac { 1 } { | p _ { i } - p _ { j } | + \epsilon } ,\tag{8}
$$

where $p _ { i }$ and $p _ { j }$ denote the global root positions of dancer i and j, respectively, and $\epsilon = 1 0 ^ { - 6 }$ is a small constant added to avoid division by zero. This formulation encodes a natural inductive bias: closer dancers exert a stronger influence on the generation ofthe current dancer, mirroring how real choreography prioritizes nearby partners over distant ones.

Graph construction. Following the chain-of-dancers formulation, the graph is built incrementally rather than over a fully connected static topology. When a new dancer is added to the chain, edges are dynamically constructed only between the new node and the previously generated dancers, while existing edges are preserved. This design has been shown effective in recent graph-based generation studies, including I-GCN [23] and IGC [4], and naturally fits our sequential generation scheme. During diffusion sampling, the follower’s root position is taken from the intermediate prediction $\hat { X } _ { i } ^ { ( t ) }$ at each timestep t; the graph is recomputed dynamically as denoising progresses.

Top-k edge sparsification. To focus on local coordination and improve computational efficiency, we sparsify the graph by retaining only the top 60% nearest edges $( \mathrm { i . e . }$ , highest $w _ { i j }$ values) for each newly added dancer. This avoids over-smoothing from far-away dancers whose contributions to local coordination are minimal, while preserving sufficient connectivity to model meaningful inter-dancer relationships. The sparsified weights are then row-normalized via softmax to obtain the final adjacency matrix used in the GCN layer:

$$
\tilde { w } _ { i j } = \frac { \exp ( w _ { i j } ) } { \sum _ { k \in \mathcal { N } _ { i } } \exp ( w _ { i k } ) } ,\tag{9}
$$

where ${ \mathcal { N } } _ { i }$ denotes the set of retained neighbors of dancer i after top-k sparsification.

## B.2 Algorithm Details

Due to space limitations in the main paper, we provide a detailed description of the Group-Aware Motion Encoder in Algorithm 1. This section explicitly specifies the input and output representations, intermediate feature dimensionalities, and the sequence of modules used to generate inter-dancer motion-aware key/value pairs. The goal is to clarify how motion interactions are encoded and injected into the frozen single-dancer diffusion backbone, complementing the high-level description in the main text.

Algorithm 1 Group-Aware Motion Encoder   
Input: Immediate leader motion $\mathbf { X } _ { j }$ (where $j = i - 1 )$ , all previously generated motions $\{ \mathbf { X } _ { 1 } , \dotsc , \mathbf { X } _ { i - 1 } \}$   
Output: Inter-dancer key/value $( \mathbf { K } _ { g r p } , \mathbf { V } _ { g r p } )$   
$\bar { \mathcal { G } } { ' } - \mathrm { B u i l d G r a p h } ( \{ \mathbf { X } _ { 1 } ^ { \cdot } , \dots , \mathbf { X } _ { i - 1 } \} ) \quad \bar { / } /$ Distance-weighted graph over all previously generated dancers   
$\mathbf Z _ { j } \gets f _ { \mathrm { m o t i o n } } ( \mathbf X _ { j } )$ // Leader Motion Encoding, $\mathbf { X } _ { j } \in \overset { \sim } { \mathbb { R } } ^ { B \times T \times D _ { m } } , \mathbf { Z } _ { j } \in \mathbb { R } ^ { B \times T \times d _ { z } }$   
$\tilde { \mathbf Z } _ { j } \gets \mathrm { G C N } ( \mathbf Z _ { j } , \mathcal { G } )$ // Aggregate group-wide spatial context onto leader feature, temporal length L   
preserved   
$\mathbf { H } _ { j } \gets$ Transformer $\mathbf { \sigma } _ { \mathrm { o R A } } ( \tilde { \mathbf { Z } } _ { j } )$ // LoRA-Adapted Transformer Encoding, which applies low-rank updates to   
linear projections   
$\mathbf { K } _ { g r p } \dot {  } \mathbf { H } _ { j } \mathbf { W } _ { K }$ // Key / Value Projection, $\mathbf { K } _ { g r p } , \mathbf { V } _ { g r p } \in \mathbb { R } ^ { B \times H \times L \times d _ { h } }$   
$\mathbf { V } _ { g r p }  \mathbf { H } _ { j } \mathbf { W } _ { V }$   
Inject $( \mathbf { K } _ { g r p } , \mathbf { V } _ { g r p } )$ into frozen backbone // Cross-Attention Injection

## B.3 Training Loss

The detailed definitions of the loss terms used in Equation 3 in Section 3.2 are given as follows.

Pose Reconstruction Loss. To ensure accurate recovery of dancer poses, we use an $\ell _ { 1 }$ loss between predicted $\hat { \mathbf { X } } _ { l } ^ { ( n ) }$ and ground-truth $\mathbf { X } _ { l } ^ { ( n ) }$ joint representations:

$$
\mathcal { L } _ { \mathrm { p o s e } } = \frac { 1 } { N L } \sum _ { n = 1 } ^ { N } \sum _ { l = 1 } ^ { L } \left\| \hat { \mathbf { X } } _ { l } ^ { ( n ) } - \mathbf { X } _ { l } ^ { ( n ) } \right\| _ { 1 } .\tag{10}
$$

This term drives the model to reproduce spatially precise skeleton structures at each frame.

Velocity Regularization Loss. To suppress temporal jitter and enforce smooth motion transitions, we apply a velocity consistency constraint:

$$
\mathcal { L } _ { \mathrm { v e l } } = \frac { 1 } { N L } \sum _ { n , l } \left\| ( \hat { \mathbf { X } } _ { l } ^ { ( n ) } - \hat { \mathbf { X } } _ { l - 1 } ^ { ( n ) } ) - ( \mathbf { X } _ { l } ^ { ( n ) } - \mathbf { X } _ { l - 1 } ^ { ( n ) } ) \right\| _ { 1 } .\tag{11}
$$

It encourages the predicted motion derivatives to follow the ground-truth temporal dynamics.

Foot-Ground Contact Loss. To reduce foot sliding and floating artifacts, we constrain the contact joints (e.g., ankles and toes) relative to the ground plane:

$$
\mathcal { L } _ { \mathrm { c o n t a c t } } = \frac { 1 } { L } \sum _ { l } \sum _ { k \in \Omega _ { \mathrm { f o o t } } } \left( \left| \hat { z } _ { l , k } \right| + \alpha \left| \hat { v } _ { l , k } \right| \right) ,\tag{12}
$$

where $\hat { z } _ { l , k }$ and $\hat { v } _ { l , k }$ denote the height and velocity of foot joint k, and α balances positional and velocity constraints.

Multimodal Alignment Loss. We follow the text-pivoted alignment formulation proposed in DanceChat [34], which encourages consistency between modalities by aligning both text–music and text–motion embeddings via cosine similarity. At each timestep i, the alignment loss is defined as:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Big [ ( 1 - \mathrm { s i m } ( E _ { t _ { i } } , E _ { m _ { i } } ) ) + ( 1 - \mathrm { s i m } ( E _ { t _ { i } } , E _ { x _ { i } } ) ) \Big ] ,\tag{13}
$$

where $\begin{array} { r } { \sin ( \mathbf { a } , \mathbf { b } ) = \frac { \left. \mathbf { a } , \mathbf { b } \right. } { \left\| \mathbf { a } \right\| \left\| \mathbf { b } \right\| } } \end{array}$ denotes cosine similarity and $E _ { t _ { i } } , E _ { m _ { i } } , E _ { x _ { i } }$ represent the encoded embeddings of text, music, and motion, respectively, at timestep i. While we adopt the same loss structure as DanceChat, we re-implement the encoders $E _ { m }$ and $E _ { x }$ using our own music and motion backbones to ensure compatibility with the encoder-based architecture.

## B.4 Inference Global Penalties

We used GPU parallel acceleration in different sample batch dimensions to minimize latency, where multiple noise samples are optimized in parallel within each iteration of gradient descent. In Section 3.3, we introduce a global penalty at inference time. The loss terms of $L _ { \mathrm { g l o b a l } }$ in Equation 5 are listed as follows:

Root Position Penalty. To ensure that an individual reaches a specific location at a particular time, we define the root position penalty $L _ { \mathrm { r o o t } }$ as follows:

$$
L _ { \mathrm { r o o t } } = \sum _ { l \in L } \operatorname* { m a x } \left( 0 , \Vert p _ { \mathrm { r o o t } } ^ { p } ( l ) - p _ { \mathrm { t a r g e t } } ( l ) \Vert ^ { 2 } - \delta \right) ,\tag{14}
$$

where $p _ { \mathrm { r o o t } } ^ { p } ( l )$ represents the root position of individual $p$ at frame $l ,$ and $p _ { \mathrm { t a r g e t } } ( l )$ is the target position at the same frame. The hyperparameter δ acts as a threshold distance, below which no penalty is applied. This formulation relaxes the loss function by penalizing deviations from the target position only when the distance exceeds δ. Such a design avoids overly rigid constraints on close positions and encourages more natural and flexible motion synthesis, consistent with prior work PINO [24].

Movement Region Penalty. To restrict movement within a defined area, we introduce the movement region penalty $L _ { \mathrm { r e g i o n } }$ for a chosen individua $p \mathrm { : }$

$$
L _ { \mathrm { r e g i o n } } = \frac { 1 } { L J } \sum _ { n } \sum _ { j } \phi \left( p _ { j } ^ { p } ( n ) \right) ,\tag{15}
$$

where L is the total number of frames, J is the total number of joints, and $\phi ( \cdot )$ penalizes positions outside the desired region. Here, $p _ { j } ^ { p } ( n )$ denotes the position of joint j of individual $p$ at frame $n .$ . The penalty function $\phi ( \cdot )$ can be flexibly defined according to task requirements. For example, restricting movement within a rectangular cuboid region with lower bounds ${ \boldsymbol { l } } = ( l _ { x } , l _ { y } , l _ { z } )$ and upper bounds $\boldsymbol { u } = ( u _ { x } , u _ { y } , u _ { z } )$ yields:

$$
\phi ( p ) = \sum _ { k \in \{ x , y , z \} } \operatorname* { m a x } ( 0 , l _ { k } - p _ { k } ) + \operatorname* { m a x } ( 0 , p _ { k } - u _ { k } ) ,\tag{16}
$$

where $p _ { k }$ is the k-th coordinate of $p .$

<table><tr><td>Relation (REL)</td><td>Decision rule in window W</td><td>Artifacts (for prompt)</td></tr><tr><td>approach</td><td>Monotonic decrease in distance:  $\Delta d / W < - 0 . 1$  and  $d _ { i j } ( w _ { 0 } + W - 1 ) \in [ d _ { \mathrm { n e a r } } , d _ { \mathrm { m i d } } ] .$ </td><td>WHEN, target distance  $d ^ { \star } \approx d _ { \mathrm { n e a r } }$ </td></tr><tr><td>keep_distance</td><td>Low variance around mid  $\operatorname { V a r } _ { w \in \mathcal { W } } ( d _ { i j } ( w ) ) ~ < ~ 0 . 0 5$  and  $\mathrm { m e a n } ( d _ { i j } ) \in$   $[ d _ { \mathrm { m i d } } - 0 . 2 , d _ { \mathrm { m i d } } + 0 . 2 ] .$ </td><td>distance: WHEN, target distance  $d ^ { \star }$  ≈  $d _ { \mathrm { m i d } }$ </td></tr><tr><td>face_to</td><td>Mutual facing with proximity: cos  $\left( \mathbf { f } _ { i } , \mathbf { p } _ { j } - \mathbf { p } _ { i } \right) >$  0.7 and cos  $\left( \mathbf { f } _ { j } , \mathbf { p } _ { i } - \mathbf { p } _ { j } \right) > 0 . 7$  and  $d _ { i j } < d _ { \mathrm { f a r } }$ </td><td>WHEN</td></tr><tr><td>lead_follow</td><td>Cross-correlation of motion energies peaks at a WHEN, lag positive lag: arg  $\operatorname* { m a x } _ { \Delta w \in [ 2 , 8 ] }$  xcorr  $( \hat { E } _ { i } , E _ { j } ) =$   $\bar { \Delta } w > 0 .$  (Here i leads, j follows.)</td><td> $\Delta w ;$  leader/follower IDs</td></tr><tr><td>swap</td><td>Start/end positions swap across the window: WHEN  $\| \mathbf { p } _ { i } ( w _ { 0 } ) - \mathbf { \hat { p } } _ { j } ( w _ { 0 } + W - \hat { 1 } ) \| _ { 2 } < \epsilon \mathrm { a n d } \| \mathbf { p } _ { j } ( w _ { 0 } ) -$   $\dot { \mathbf p } _ { i } ( \dot { w } _ { 0 } + W - 1 ) \lVert _ { 2 } < \epsilon ,$  while trajectories do not collide.</td><td></td></tr><tr><td>passby</td><td>Trajectories intersect near mid/near distance with opposite normal velocities:  $\exists w \in \mathcal { W } \ \mathrm { s . t . }$   $\hat { d _ { i j } } ( w ) \in [ d _ { \mathrm { n e a r } } , d _ { \mathrm { m i d } } ] , \langle \dot { \bf p } _ { i } ^ { \perp } , \dot { \bf p } _ { j } ^ { \perp } \rangle < 0 ,$  and dis- tance increases rapidly after intersection.</td><td>1 WHEN</td></tr></table>

Table 5: Window-based relation detection rules for prompt mining. WHEN records the time span(s) in W where the rule is satisfied; artifacts $( \mathrm { e . g . , } d ^ { \star }$ , contacted parts, lag ∆t) are injected into prompt templates.

Orientation Penalty. To control facing directions at specific frames, we define the orientation penalty $L _ { \mathrm { o r i e n t } }$ with threshold δ as:

$$
L _ { \mathrm { o r i e n t } } = \sum _ { l \in L } \operatorname* { m a x } \left( 0 , 1 - d ^ { p } ( l ) \cdot d _ { \mathrm { t a r g e t } } ( l ) - \delta \right) ,\tag{17}
$$

where $d ^ { p } ( l )$ is the normalized facing direction of individual $p$ at frame l, and $d _ { \mathrm { t a r g e t } } ( l )$ is the desired direction. The threshold δ determines the tolerance margin within which no penalty is applied. By penalizing deviations only when direction similarity falls below $1 - \delta .$ , this formulation promotes natural orientation control and avoids abrupt directional changes, similar to the soft-direction constraints adopted in PINO [24].

Relative Position Penalty. To ensure that the root positions of two individuals remain within a desired distance range, we define the relative position penalty $L _ { \mathrm { r e l a t i v e } }$ as:

$$
\begin{array} { r l r } {  { L _ { \mathrm { r e l a t i v e } } = \sum _ { l \in L } \Big [ \operatorname* { m a x } \big ( 0 , d _ { \mathrm { m i n } } - \| p _ { \mathrm { r o o t } } ^ { 1 } ( l ) - p _ { \mathrm { r o o t } } ^ { 2 } ( l ) \| \big ) } } \\ & { } & { + \operatorname* { m a x } \big ( 0 , \| p _ { \mathrm { r o o t } } ^ { 1 } ( l ) - p _ { \mathrm { r o o t } } ^ { 2 } ( l ) \| - d _ { \mathrm { m a x } } \big ) \Big ] , } \end{array}\tag{18}
$$

where $p _ { \mathrm { r o o t } } ^ { 1 } ( l )$ and $p _ { \mathrm { r o o t } } ^ { 2 } ( l )$ denote the root positions of two individuals at frame $l ,$ and $d _ { \mathrm { m i n } }$ and $d _ { \mathrm { m a x } }$ represent the lower and upper bounds of the allowable interpersonal distance. This constraint maintains natural interaction spacing and prevents unrealistic crowding or separation, an important requirement also emphasized by PINO [24].

To accelerate the optimization of global penalty over initial noise at the inference stage, in the main experiment, we did not use the penalty of the entire long-time sequence as the optimization target, but instead selected the 70 frames with the largest penalty values in the entire sequence as the optimization target. Meanwhile, we used GPU parallel acceleration in different sample batch dimensions to minimize latency in the inference stage.

## C From Rules to Interaction Types

As described in Section 3.1 and illustrated in Figure 3, the mined pairwise relations from each window W are verbalized into dancer-specific textual instructions that serve as the conditioning input for

Table 6: Unified comparison across 1–5 dancers on the AIOZ-GDance dataset. For all baselines, separate models are trained and evaluated for each group size. In contrast, ChainDance uses a single unified model for all group sizes (1–5 dancers). Best results are bold; second-best are underlined. Results with \* denote models without the global noise optimization penalty.
<table><tr><td rowspan="2">Method</td><td colspan="3">1 dancer</td><td colspan="3">2 dancers</td><td colspan="3">3 dancers</td><td colspan="3">4 dancers</td><td colspan="3">5 dancers</td></tr><tr><td>FID↓</td><td>Div↑</td><td>PFC↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>GCD [16]</td><td>39.24</td><td>9.64</td><td>2.53</td><td>34.39</td><td>80.32</td><td>0.17</td><td>30.22</td><td>80.22</td><td>0.19</td><td>36.28</td><td>81.82</td><td>0.13</td><td>38.43</td><td>81.44</td><td>0.17</td></tr><tr><td>CoDancers [38]</td><td>23.98</td><td>9.48</td><td>3.53</td><td>24.55</td><td>72.52</td><td>0.08</td><td>26.34</td><td>74.22</td><td>0.08</td><td>26.44</td><td>75.34</td><td>0.10</td><td>27.27</td><td>74.34</td><td>0.11</td></tr><tr><td>TCDiff [3]</td><td>37.31</td><td>14.01</td><td>0.51</td><td>15.77</td><td>81.92</td><td>0.12</td><td>15.36</td><td>82.77</td><td>0.11</td><td>13.44</td><td>81.70</td><td>0.15</td><td>14.62</td><td>81.40</td><td>0.11</td></tr><tr><td>ST-GDance [35]</td><td>28.87</td><td>12.82</td><td>0.97</td><td>19.42</td><td>80.52</td><td>0.12</td><td>14.76</td><td>81.24</td><td>0.11</td><td>14.02</td><td>81.42</td><td>0.10</td><td>23.22</td><td>80.76</td><td>0.12</td></tr><tr><td>ChainDance*</td><td>23.12</td><td>13.85</td><td>0.50</td><td>14.10</td><td>81.52</td><td>0.11</td><td>12.01</td><td>82.68</td><td>0.11</td><td>13.28</td><td>82.13</td><td>0.12</td><td>15.34</td><td>80.93</td><td>0.13</td></tr><tr><td>ChainDance</td><td>23.12</td><td>13.85</td><td>0.50</td><td>13.67</td><td>81.28</td><td>0.12</td><td>11.63</td><td>83.74</td><td>0.10</td><td>12.59</td><td>82.16</td><td>0.13</td><td>14.73</td><td>81.56</td><td>0.12</td></tr></table>

ChainDance. We use a sliding temporal window W to define these relations: for each window, we look at the locations of two dancers within this short time segment and decide whether an interaction occurs and, if so, what type it is. This design uses windows instead of the whole sequence because interactions do not happen at every frame and are usually completed within a short time range. We adopt minimal, compositional templates that consume the artifacts produced by the decision rules (e.g., estimated distance $d ^ { \star }$ , lag $\Delta t .$ and time span WHEN), while strictly following the prompt format constraints in Section 3.1. The detailed computation of these rules is summarized in Table 5.

Templates. Detected relations are mapped to short phrases as follows:

• approach: “Dancer-i approaches Dancer-j to about $d ^ { \star } .$ 99

• keep\_distance: “i and j keep a steady distance ≈ $d ^ { \star } . ^ { \star }$

• face\_to: “i and j face each other.”

• lead\_follow: “i leads j (lag ∆w).”

• swap: “i and j swap positions at.”

• passby: “i and j pass by.”

These phrases are then injected into per-dancer prompts (e.g., “The dancer i . . . , facing toward Dancer j.”) on top of the global description, yielding individualized conditioning as specified in Section 3.1.

De-duplication and Priority. If multiple relations fire in the same window, we keep the one with the highest confidence (e.g., larger margin to threshold) and merge contiguous WHEN intervals if gaps are < 0.5 s. The final per-dancer prompt is constructed by (i) preserving the global motif and (ii) appending at most one relation phrase per active pair within W.

## D More Quantitative Results

## D.1 Comparison of 5 Dancers

Table 6 summarizes the results across the 1–5 dancer settings. We report 5-dancer results here rather than in the main paper because the number of 5-dancer samples in the AIOZ-GDance test set is limited, making the comparison less statistically robust. At N=5, ChainDance achieves the best GMC (81.56) and the second-best GMR (14.73 vs. TCDiff’s 14.62), with the small GMR gap reflecting a fundamental trade-off of our design: while joint-modeling baselines can be specifically optimized for each fixed group size, ChainDance’s unified pair-wise training transfers across all sizes from a single model. Across the full 1–5 dancer range, ChainDance remains consistently competitive with one unified model, an advantage no baseline can match.

## D.2 Comparison on DD100 dataset

Although AIOZ-GDance is the only publicly available large-scale group dance benchmark, several duet datasets exist. We therefore evaluate ChainDance’s generalization on DD100 [29], a duet dance dataset with 117 minutes of professional motion-capture performances across 10 genres. Following our main setup, we adopt the Lodge [21] backbone pretrained on FineDance [20] and post-train the group-aware modules on DD100. Since DD100 does not provide text annotations, we disable the Role-Aware Text Encoder for this experiment and condition only on music, while retaining the Group-Aware Motion Encoder and chain-of-dancers formulation. We follow the official train/test split and compare with Duolando [29] and GCD [16] under the same evaluation protocol. Results are reported in Table 7.

Table 7: Two-dancer generation results on the DD100 dataset. Best results are bold. “→” indicates that closer to ground-truth (GT) is better.
<table><tr><td>Method</td><td>FID↓</td><td>Div→</td><td>PFC↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>GT</td><td>一</td><td>15.67</td><td>0.36</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Duolando [29]</td><td>12.42</td><td>14.35</td><td>16.22</td><td>14.52</td><td>81.32</td><td>0.15</td></tr><tr><td>GCD [16]</td><td>9.73</td><td>14.62</td><td>5.12</td><td>14.31</td><td>81.21</td><td>0.13</td></tr><tr><td>ChainDance (w/o RATE)</td><td>2.97</td><td>15.52</td><td>4.41</td><td>12.21</td><td>82.42</td><td>0.12</td></tr></table>

As shown in Table 7, ChainDance achieves the best performance across all metrics on DD100, demonstrating that our framework generalizes effectively beyond AIOZ-GDance. In particular, ChainDance reduces FID from 9.73 (GCD) to 2.97, achieves the closest Div to GT (15.52 vs. GT’s 15.67), and outperforms both Duolando and GCD on group-level metrics. These results confirm that even with text conditioning disabled, the chain-of-dancers formulation, combined with the Group-Aware Motion Encoder, generalizes effectively to the duet setting.

## D.3 Comparison with PINO

A potential concern is whether the gains of ChainDance over end-to-end baselines simply come from the test-time noise optimization adopted from PINO [24]. To address this, we directly compare ChainDance with PINO using the same single-dancer backbone for group sizes 2–4, isolating the contribution of our proposed encoders and chain-of-dancers formulation.

As shown in Table 8, ChainDance consistently outperforms PINO across all group sizes. Notably, the performance gap widens as the group size increases (e.g., GMR improves by 0.43 for 2 dancers but by 1.24 for 4 dancers), indicating that relying solely on global physical constraints becomes increasingly insufficient as inter-dancer interactions grow more complex. While PINO refines spatial consistency at inference time, it lacks the capacity to model fine-grained inter-dancer relationships during generation. Our encoders address this gap by injecting relational information directly into the latent space, providing the structural foundation that PINO’s optimization alone cannot establish. This isolation experiment confirms that ChainDance’s contribution goes beyond inference-time refinement: the proposed encoders and chain-of-dancers formulation are the key drivers of the observed performance gains.

## D.4 User Study

Beyond automatic metrics, we conduct a user study to evaluate the perceptual quality of generated group dances. Following the protocol of our main baseline TCDiff [3], we invite 30 participants (ages 18–55, aligning with the target user group of our application scenario) to participate in a blind ranking test. Outputs from all methods are anonymized and shuffled before presentation. For each comparison, participants are shown four videos (one per method) and asked to rank them from best to worst along four perceptual dimensions:

• Realism: How natural and human-like are the generated motions?

• Music Correlation: How well do the dance movements align with the music’s rhythm and style?

• Formation Aesthetics: How visually pleasing and well-coordinated are the group formations?

Table 8: Comparison with PINO across different group sizes. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">2</td><td colspan="3">3</td><td colspan="3">4</td></tr><tr><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>PINO [24]</td><td>14.10</td><td>80.83</td><td>0.12</td><td>12.69</td><td>82.92</td><td>0.11</td><td>13.83</td><td>80.93</td><td>0.14</td></tr><tr><td>ChainDance</td><td>13.67</td><td>81.28</td><td>0.12</td><td>11.63</td><td>83.74</td><td>0.10</td><td>12.59</td><td>82.16</td><td>0.13</td></tr></table>

![](images/a7332c5cf94c1c2407005192bd9880a50ce9f1cca592addc617ca5b5e2ae5ed1.jpg)  
Figure 6: User study results across four perceptual dimensions. ChainDance is consistently preferred over baselines on Realism, Music Correlation, Formation Aesthetics, and Dancer Harmony. The largest improvement is observed on Dancer Harmony (0.80 vs. 0.50 of the best baseline), which jointly captures inter-dancer coordination and per-dancer identity stability—both structurally improved by our sequential decomposition.

• Dancer Harmony: How harmoniously do the dancers interact and complement each other, and how consistently does each dancer maintain a stable identity throughout the sequence (i.e., absence of identity swaps or sudden position jumps)?

Rankings are converted into scores from 4 (best) to 1 (worst), and the averaged scores are further normalized to [0, 1] for fair comparison. Results are reported in Figure 6.

As shown in Figure 6, ChainDance is preferred across all four dimensions, achieving the highest normalized scores on Realism (0.70), Music Correlation (0.75), Formation Aesthetics (0.70), and Dancer Harmony (0.80). The largest gap appears on Dancer Harmony, where ChainDance scores 0.80 compared to 0.50 for the strongest baseline on this dimension (ST-GDance). This dimension captures both inter-dancer coordination and per-dancer identity stability over time—two aspects that joint-modeling baselines struggle with simultaneously, as joint tokenization tends to produce identity swaps and flickering artifacts (Figure 1b). ChainDance’s sequential decomposition structurally avoids identity entanglement, while RATE and GAME ensure coherent inter-dancer interaction. This provides direct perceptual evidence supporting the identity preservation claim, complementing the objective metrics in Table 1 and Table 8, jointly demonstrating that ChainDance produces group choreography that is both quantitatively superior and perceptually preferred.

## D.5 Other Ablation Settings

To comprehensively evaluate ChainDance, we compare results under different configurations: singledancer generators pretrained on AIOZ-GDance, models pretrained jointly on FineDance and AIST++, and variants trained without motion captions in text descriptions.

Table 9 summarizes several additional ablation settings. When we pretrain the single-dancer generator on the AIOZ-GDance subset and then finetune with multi-dancer clips, the performance is slightly lower than the main results in the paper. This is understandable because AIOZ-GDance is primarily a multi-person dataset, and its single-dance samples are limited, which makes the single-dancer pretraining less effective.

Table 9: Comparison of 3 dancers performance under different ablation settings on the AIOZ-GDance dataset. Best results are bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">Group Dance</td></tr><tr><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>ChainDance</td><td>11.63</td><td>83.74</td><td>0.10</td></tr><tr><td>Ours-Lodge(AIOZ-GDance)</td><td>12.08</td><td>82.32</td><td>0.11</td></tr><tr><td>Ours-Lodge (FineDance&amp;AIST++)</td><td>11.97</td><td>82.41</td><td>0.12</td></tr><tr><td>ChainDance (w/o Motion Caption)</td><td>11.89</td><td>82.52</td><td>0.11</td></tr><tr><td>ChainDance (w/o bidirectional pairing)</td><td>12.93</td><td>79.42</td><td>0.13</td></tr><tr><td>ChainDance (with random pairing strategy)</td><td>11.58</td><td>83.80</td><td>0.10</td></tr></table>

Using a backbone pretrained jointly on FineDance and AIST++ achieves similar performance to the AIOZ-GDance-based model, but remains below the version pretrained only on FineDance in the main paper. A possible explanation is that the mixed-domain training increases the diversity of styles and camera motions, which benefits generality but introduces additional variance that is not fully aligned with AIOZ-GDance.

Removing motion captions from the text descriptions leads to a small drop in performance, but the results remain stable and still compare favorably with end-to-end multi-dancer baselines that do not use such information. This suggests that motion captions are helpful, but they are not the dominant factor. The improvements mainly come from the full text inputs (music, motion caption, beat structure) together with the ChainDance formulation, which simultaneously provide the model with strong guidance and allow efficient multi-dancer generation. Training pairs without bidirectional leader relationship between dancers may cause the model to learn only some fixed local interaction patterns, thus greatly reducing the effectiveness of group dance generation. Comparing the two pairing strategies, both random pairing and chain pairing yield similar group-level performance, with only marginal differences across metrics. For clarity and reproducibility, we use chain pairing as the default setting in the main experiments.

## D.6 Error Accumulation Analysis Across Group Sizes

A natural concern with sequential chain generation is whether errors accumulate as group size grows. Our inference-time global optimization is designed to mitigate such error propagation. Table 10 compares generation quality with and without the global penalty across group sizes 2–5.

Table 10: Comparison of results with / without the global penalty across group sizes. Format: with penalty / without penalty.
<table><tr><td>N</td><td>GMR↓</td><td>GMC↑</td><td></td><td>TIF↓</td></tr><tr><td>2</td><td>13.67 14.52</td><td>81.28</td><td>80.92</td><td>0.12 0.13</td></tr><tr><td>3</td><td>11.63 12.44</td><td>83.74</td><td>82.12</td><td>0.12 0.13</td></tr><tr><td>4</td><td>12.59 14.30</td><td>82.16</td><td>81.47</td><td>0.13 0.13</td></tr><tr><td>5</td><td>14.73 15.52</td><td>81.56</td><td>80.64</td><td>0.12 0.13</td></tr></table>

Several observations are noteworthy. First, metrics do not degrade monotonically with group size—for instance, TIF for 5 dancers (0.12) is lower than for 4 dancers (0.13), and GMC for 3 dancers (83.74) outperforms 2 dancers (81.28), suggesting the chain formulation benefits from richer conditioning context as more dancers are added. Second, the global penalty consistently improves performance across all group sizes, confirming that it effectively mitigates error propagation inherent in sequential generation rather than simply applying post-hoc corrections.

## D.7 Analysis on Different Optimization Steps and More Number of Dancers

Table 11 compares generation quality and inference efficiency under different penalty threshold settings. The ChainDance row shows the configuration that achieves the best performance (consistent with the main experiments in the paper). We then examine the trade-offs between generation quality and efficiency across different multiplication factors.

Table 11 shows that fewer optimization steps reduce computation but also weaken the global refinement, leading to higher GMR and lower GMC. As the multiplier increases, the global penalty constraints take effect more strongly, and both GMR and GMC improve, but at the cost of longer inference time. The ChainDance configuration achieves the best overall balance, giving the lowest GMR and the highest GMC while keeping the inference cost within a reasonable range. This confirms that the chosen penalty provides the most stable refinement behavior for multi-dancer generation. We omit TIF metric in this comparison because the variations across settings are minimal.

Table 12 reports the trade-off between inference time and group-level dance quality as the number of dancers increases. As the group size increases from 2 to 6, inference time increases gradually. Importantly, this growth is accompanied by relatively stable group dance quality, indicating that scalability does not come at the cost of coordination or realism. In particular, GMR and GMC remain competitive across different group sizes, with the best overall balance observed at three dancers, while

TIF shows only a mild increase for larger groups, indicating that collision artifacts do not escalate dramatically with scale.

These results suggest that although sequential generation incurs additional inference cost for larger groups, the proposed method maintains consistent coordination and motion realism as the number of dancers increases. This highlights a practical trade-off between scalability and efficiency: higher group sizes naturally require longer generation time, but the quality degradation remains limited and gradual, demonstrating the robustness of the chain-based formulation in multi-dancer scenarios. While not real-time, it is effective for high-quality offline content of arbitrary-sized group motion, whose training data is scarce. Accelerating optimization is future work.

## D.8 Quantitative Evidence for Music and Text Controllability

To rigorously validate the responsiveness of ChainDance to its multimodal conditioning signals, we provide quantitative measurements using Beat Alignment Score (BAS) for music alignment and R-precision for text alignment. Since no prior baseline reports such metrics for group dance generation, we report ground-truth (GT) values as upper bounds.

## D.8.1 Matched vs. Mismatched Inputs

We measure model responsiveness by comparing performance under matched inputs (correctly paired music/text and motion) versus mismatched inputs (randomly swapped from other categories). A large gap between the two indicates that the model genuinely conditions on the input rather than ignoring it.

Table 13: Quantitative results for music and text alignment. Matched uses correctly aligned inputs; Mismatched uses randomly swapped inputs from other categories. A large gap indicates strong responsiveness to the conditioning signal.
<table><tr><td></td><td>Matched</td><td>Mismatched</td><td>Gap↑</td></tr><tr><td>GT (BAS)</td><td>0.2463</td><td>1</td><td></td></tr><tr><td>Ours (BAS)</td><td>0.2247</td><td>0.0624</td><td>0.1623</td></tr><tr><td>GT (R-precision)</td><td>0.5106</td><td>1</td><td></td></tr><tr><td>Ours (R-precision)</td><td>0.4152</td><td>0.1093</td><td>0.3059</td></tr></table>

As shown in Table 13, ChainDance achieves a substantial gap between matched and mismatched conditions (+0.1623 for music, +0.3059 for text), providing quantitative evidence that the model is genuinely responsive to both music and text inputs. The matched-condition scores remain close to GT (0.2247 vs. 0.2463 for BAS; 0.4152 vs. 0.5106 for R-precision), indicating that ChainDance generates motions well-aligned with the conditioning signals.

Table 11: In-depth analysis of the trade-off between inference time and dance quality on the AIOZ-GDance dataset (3 dancers). Best results are bold; second-best are underlined.
<table><tr><td rowspan=2 colspan=1>Optimization Steps</td><td rowspan=1 colspan=1>Group Dance</td><td rowspan=1 colspan=1>Efficiency</td></tr><tr><td rowspan=1 colspan=1>GMR↓ GMC↑</td><td rowspan=1 colspan=1>Inf Time(min:sec)↓</td></tr><tr><td rowspan=1 colspan=1>0×</td><td rowspan=2 colspan=1>12.01  82.6811.83  82.04</td><td rowspan=3 colspan=1>0:060:080:10</td></tr><tr><td rowspan=1 colspan=1>0.1×</td></tr><tr><td rowspan=1 colspan=1>0.25×</td><td rowspan=1 colspan=1>11.78  81.72</td></tr><tr><td rowspan=2 colspan=1>0.5×0.75×</td><td rowspan=1 colspan=1>11.52  82.35</td><td rowspan=1 colspan=1>0:11</td></tr><tr><td rowspan=1 colspan=1>11.72  82.59</td><td rowspan=1 colspan=1>0:13</td></tr><tr><td rowspan=1 colspan=1>ChainDance (80 steps)</td><td rowspan=1 colspan=1>11.63  83.74</td><td rowspan=1 colspan=1>0:15</td></tr></table>

Table 12: In-depth analysis of the trade-off between inference time and dance quality on the AIOZ-GDance dataset for more dancers. Best results are bold; second-best are underlined.
<table><tr><td rowspan="2">Number of Dancers</td><td colspan="3">Group Dance</td><td>Efficiency</td></tr><tr><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td><td>Inf Time (min:sec)↓</td></tr><tr><td>2</td><td>13.67</td><td>81.28</td><td>0.12</td><td>0:07</td></tr><tr><td>3</td><td>11.63</td><td>83.74</td><td>0.10</td><td>0:15</td></tr><tr><td>4</td><td>12.59</td><td>82.16</td><td>0.13</td><td>0:19</td></tr><tr><td>5</td><td>14.73</td><td>81.56</td><td>0.12</td><td>0:22</td></tr><tr><td>6</td><td>14.85</td><td>80.87</td><td>0.13</td><td>0:29</td></tr></table>

## D.8.2 Genre-wise Statistical Analysis

To further demonstrate controllability and generation quality across diverse music genres, we conduct a genre-wise analysis over all 10 music genres in the AIOZ-GDance test set, reporting both alignment metrics (BAS, R-Precision) and group dance quality metrics (GMR, GMC, TIF).

Table 14: Quantitative results across different music genres. ChainDance maintains stable alignment scores (BAS, R-precision) and consistent group dance quality (GMR, GMC, TIF) across all 10 genres, demonstrating robust dual-condition control.
<table><tr><td rowspan="2">Music genre</td><td colspan="2">BAS</td><td colspan="2">R-precision</td><td colspan="3">Group Dance Quality</td></tr><tr><td>GT</td><td>Ours</td><td>GT</td><td>Ours</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>Disco</td><td>0.2871</td><td>0.2661</td><td>0.5234</td><td>0.4194</td><td>13.52</td><td>83.03</td><td>0.12</td></tr><tr><td>Electronic</td><td>0.2712</td><td>0.2562</td><td>0.5076</td><td>0.4212</td><td>13.21</td><td>82.83</td><td>0.13</td></tr><tr><td>Folk</td><td>0.2154</td><td>0.1961</td><td>0.5112</td><td>0.4287</td><td>13.84</td><td>81.97</td><td>0.13</td></tr><tr><td>Funk</td><td>0.2176</td><td>0.1923</td><td>0.5154</td><td>0.4431</td><td>12.87</td><td>82.42</td><td>0.12</td></tr><tr><td>Indian</td><td>0.2234</td><td>0.2001</td><td>0.5223</td><td>0.4076</td><td>13.48</td><td>83.13</td><td>0.12</td></tr><tr><td>Latin</td><td>0.2715</td><td>0.2504</td><td>0.4989</td><td>0.3821</td><td>13.24</td><td>81.32</td><td>0.13</td></tr><tr><td>Pop</td><td>0.2203</td><td>0.1983</td><td>0.5187</td><td>0.4102</td><td>12.76</td><td>80.98</td><td>0.12</td></tr><tr><td>R&amp;B</td><td>0.2728</td><td>0.2553</td><td>0.4923</td><td>0.3954</td><td>13.49</td><td>82.84</td><td>0.13</td></tr><tr><td>Rap</td><td>0.2156</td><td>0.1929</td><td>0.5067</td><td>0.3998</td><td>11.98</td><td>83.13</td><td>0.12</td></tr><tr><td>Reggae</td><td>0.2681</td><td>0.2393</td><td>0.5095</td><td>0.4445</td><td>12.57</td><td>82.76</td><td>0.12</td></tr></table>

Table 14 shows that BAS scores remain consistently close to GT across all genres (gap typically below 0.03), with no notable degradation in R-precision. Group dance quality metrics (GMR, GMC, TIF) remain stable across all genres, with no genre showing significant performance drops. These results confirm that ChainDance maintains robust dual-condition control and group coordination across a broad range ofmusical styles, rather than being biased toward a particular genre. Together with Table 13, these analyses provide rigorous quantitative evidence supporting the controllability claim made in the main paper.

## D.9 Order Sensitivity Analysis

A natural concern with sequential generation is whether ChainDance is sensitive to the choice of generation order or leader selection. Although Section 3.2 of the main paper shows that chain pairing and random pairing yield comparable results, that experiment evaluates each strategy only once. To rigorously assess the model’s sensitivity to generation order, we conduct two additional analyses on the AIOZ-GDance dataset.

We note that the AIOZ-GDance dataset lacks leader annotations, and its dancer indexing does not correspond to any physical position. ChainDance therefore does not depend on a predefined order or fixed leader choice—instead, the model is encouraged to focus on who participates, their motion characteristics, and their relative spatial relations, rather than the absolute feature dimension assigned to each dancer.

Table 15: Evaluation results for 3-dancer generation under all possible generation orders. The model is evaluated on 100 randomly sampled sequences. Standard deviations across orders are negligible compared to the gap between methods, indicating order-insensitivity.
<table><tr><td>Order</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>123</td><td>11.42</td><td>83.53</td><td>0.11</td></tr><tr><td>132</td><td>11.54</td><td>82.67</td><td>0.10</td></tr><tr><td>213</td><td>11.14</td><td>83.76</td><td>0.10</td></tr><tr><td>231</td><td>11.61</td><td>83.21</td><td>0.11</td></tr><tr><td>312</td><td>11.63</td><td>84.03</td><td>0.11</td></tr><tr><td>321</td><td>11.73</td><td>84.52</td><td>0.11</td></tr><tr><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $1 1 . 5 1 \pm 0 . 2 2$ </td><td> $8 3 . 6 2 \pm 0 . 6 7$ </td><td> $0 . 1 0 7 \pm 0 . 0 0 5$ </td></tr></table>

## D.9.1 All Possible Orders for 3-Dancer Generation

We randomly sample 100 sequences with 3 dancers and evaluate ChainDance under all $3 ! = 6$ possible generation orders. Results are reported in Table 15.

As shown in Table 15, varying the generation order causes negligible performance variance, with standard deviations of only 0.22 (GMR), 0.67 (GMC), and 0.005 (TIF). These variations are far smaller than the gaps between ChainDance and prior baselines (e.g., GMR gap of 3.13 against TCDiff in Table 1), indicating that the model is insensitive to the choice of generation order.

This robustness arises from two design choices: (i) the dancer indexing in AIOZ-GDance is not aligned with physical positions, so the model cannot exploit fixed-order priors during training; and (ii) the GCN-based Group-Aware Motion Encoder captures coordination through relative positions and connectivity, rather than absolute feature dimensions assigned to each dancer.

## D.9.2 Random Orders for 5-Dancer Generation

To verify that order-insensitivity also holds for larger groups, we further evaluate ChainDance on 5-dancer generation under 4 different random orders.

Table 16: Results for 5-dancer generation under 4 different random orders. ChainDance maintains stable performance across orders, confirming that order-insensitivity extends to larger group sizes.
<table><tr><td>Run</td><td>GMR↓</td><td>GMC↑</td><td>TIF↓</td></tr><tr><td>1</td><td>14.87</td><td>81.21</td><td>0.13</td></tr><tr><td>2</td><td>14.63</td><td>81.63</td><td>0.13</td></tr><tr><td>3</td><td>14.32</td><td>81.83</td><td>0.13</td></tr><tr><td>4</td><td>14.21</td><td>82.02</td><td>0.12</td></tr><tr><td> $\mathbf { M e a n } \pm \mathbf { S t d }$ </td><td> $1 4 . 5 1 \pm 0 . 3 1$ </td><td> $8 1 . 6 7 \pm 0 . 3 5$ </td><td> $0 . 1 2 8 \pm 0 . 0 0 5$ </td></tr></table>

Table 16 confirms that the model remains order-insensitive at larger group sizes, with standard deviations on the same order of magnitude as the 3-dancer case. The observed variations are again much smaller than the performance gaps between competing methods, demonstrating that ChainDance produces consistent group choreography regardless ofthe chosen generation order, even as group size increases.

## E Limitations and Future Work

While ChainDance addresses group-size rigidity and structurally preserves dancer identity by construction, several limitations remain that point to future research directions.

Inference Latency. While our approach avoids the prohibitively expensive $\mathcal { O } ( ( N L ) ^ { 2 } )$ training cost associated with joint modeling, its inference latency is relatively high. This is a common bottleneck for diffusion-based models due to the iterative denoising process, which currently limits strictly real-time interactive deployment. However, this latency (measured in seconds rather than minutes) is acceptable for our primary target applications, such as offline animation and video production. Accelerating inference toward real-time interactivity is a natural next step, e.g., via diffusion distillation or consistency-model variants.

Scaling to Massive Crowds. While ChainDance addresses flexible group-size modeling, generating highly coordinated motions for extremely large crowds (e.g., N > 10) remains a distinct research problem. Although our dynamic growing graph and test-time penalty optimization effectively mitigate spatial inconsistencies for typical group sizes, maintaining uniform global coherence at massive scale becomes increasingly difficult. Our current capacity nonetheless covers the majority of real-world applications, such as generating pop-group (e.g., boy or girl band) choreographies for music videos and gaming, where group sizes rarely exceed ten. To scale seamlessly to massive crowds, future work will explore parallel-friendly mechanisms or hierarchical generation strategies.

## F More Qualitative Results

To visually demonstrate the effect of individual components in ChainDance, we present the qualitative results of the ablation study in Figure 7a. Without GAME, each dancer moves almost independently, and the group looks uncoordinated. Without text conditioning, the motions of different dancers become too similar and lack variation. Without the global penalty, errors accumulate over time, and dancers drift to unreasonable positions with overly large distances. These examples show that all three parts are important for coordinated, diverse, and reasonable group choreography.

Then, we provide additional frame-by-frame visualizations of the three types of interaction control in Figure 7b, and show ChainDance results under different group sizes in Figure 9. Figure 8 further visualizes the process of progressively adding dancers to the same music, showing that ChainDance integrates new dancers smoothly and remains effective for large groups of 6–8 dancers. All examples follow the same visualization settings as in the main paper, where we use the simple SMPL built-in mesh to convert PKL sequences into FBX files.

For the supplementary video demos included in the index.html file, we further adopt the refined YBot model to render the motions with higher visual quality. The videos are rendered in Blender using scripts adapted from the open-source TCDiff [3] code, to which we are grateful.

![](images/10f8b22a0a41dde0305b2a63c1a4d96ec58bd9743d045e0fe98bf1e9d57db631.jpg)  
(a) Ablation study

![](images/b884ed30eced6a68badf4a3365e2dd1b5af674ce2f6d17d833d7e6588201c6b7.jpg)  
(b) Interaction control via text  
Figure 7: Qualitative visualizations. (a) Ablation study showing the effect of each component. (b) Interaction control under different text prompts.

![](images/76eaf869324d699e5c7fd21a98760dee35a5490b84151d48fbdc4199b0d50935.jpg)  
Figure 8: ChainDance results on the same music under varying group sizes (from 6 to 10 dancers). The generated choreography remains coordinated and diverse as the group size scales up.

![](images/bdb6d188f542651a4a0f5dbb9dc1de35f98c7c185836bb652d4420687d68d360.jpg)  
<sub>ergetically</sub> <sub>hops</sub> <sub>in</sub> <sub>place,</sub> <sub>swaying</sub> <sub>their</sub> <sub>shoulders</sub> <sub>rhyth</sub>m<sup>ically</sup> <sup>while</sup> <sup>each</sup> <sup>alternates</sup> <sup>raising</sup> <sup>knees</sup> <sup>high</sup> <sup>and</sup> <sup>swingin</sup>  
<sub>harply</sub> <sub>bounces</sub> <sub>in</sub> <sub>place</sub> <sub>in</sub> <sub>a</sub> <sub>Lock</sub> <sub>style,</sub> <sub>snapping</sub> <sub>their</sub> <sub>shoulders</sub> <sub>rhyth</sub>m<sup>ically</sup> <sup>while</sup> <sup>each</sup> <sup>alternates</sup> <sup>popping</sup> <sup>el</sup> <sub>d,</sub> <sub>holdi</sub>n<sup>g</sup> <sup>crisp</sup> <sup>poses</sup> <sup>to</sup> <sup>the</sup>

![](images/b1d44d1b3b811c8b373d30728e4941e09221b930914f71efefc2ba405fd01e5a.jpg)  
<sub>ualitative</sub> <sub>results</sub> <sub>of</sub> <sub>Cha</sub><sup>inDance</sup> <sup>under</sup> <sup>different</sup>