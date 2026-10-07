# 2D SPATIAL REASONING WITH ADAPTIVE NEURAL CELLULAR AUTOMATA

Martin Spitznagel<sup>1</sup>, Janis Keuper<sup>1,2</sup>

<sup>1</sup> Institute for Machine Learning and Analytics (IMLA), Offenburg University, Germany

<sup>2</sup> University of Mannheim, Germany

firstname.lastname@hs-offenburg.de

Accepted at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)

## ABSTRACT

Many modern learning approaches are still struggling with spatial reasoning tasks, i.e. they lack the ability to utilize geometric information of perceived entities and their spatial relation to each other to solve problems. We introduce a novel Adaptive Neural Cellular Automata (aNCA) architecture which replaces the static and spatially invariant perception of standard NCAs by learnable, spatially variant and data-adaptive perceptive fields. We show that this conceptual change enables NCAs to iteratively reason over 2D spatial relations on grid-like data structures (e.g. images). Empirical results on public benchmarks show state of the art comprehensible results with high gen eralization abilities for solving image based puzzles like Sudoku or finding the shortest path in a maze.

The implementation and all experimental setups are openly accessible as an extension to the NCAtorch framework at https://github.com/mspitzna/NCAtorch.

## 1 Introduction

Spatial reasoning is the ability to mentally visualize, manipulate, and functionally relate 2D and 3D objects, as well as to understand the relationships between parts, positions, and movements of one or more objects in physical space. It is a key cognitive skill used for navigation, problem-solving, engineering, and designing [1, 2]. Despite its high practical impact, spatial reasoning skills are generally still underdeveloped in most current learning models [3, 1].

In contrast to the recently emerging work on “World Models” [4], which try to learn to model complex relations in a top-down fashion, we are following a bottom-up approach which tries to isolate spatial reasoning problems in controlled environments. The focus of our proposed method lies on the also open sub-problem of spatial reasoning on 2D grid structures with a fixed x, y-coordinate system, for which we introduce a novel architecture based on Neural Cellular Automata [5]. Following the DISE [2] schematic of spatial reasoning taxonomies, our proposed aNCA architecture is specifically targeting 2D Extrinsic-Static problems, that assess the relationships between multiple objects in a fixed scene (like finding the shortest path in 2D mazes [6]) and Extrinsic-Dynamic reasoning tasks, that involve reasoning about the changing spatial relationships between multiple objects (like solving Sudokus [7]).

The key contributions of this paper can be summarized as follows:

• We are the first to apply the concept of learnable CA to spatial reasoning tasks, introducing Adaptive Neural Cellular Automata (aNCA). For this purpose we move from the space- and time-invariant update rules in traditional NCA to a more powerful and flexible input-dependent, space- and time-variant regime, and show that this significantly boosts spatial reasoning performance

• The inherent recursive nature of aNCA enables us to learn very small, yet competitive and highly generalizable models that are able to infer diverse solutions to ambiguous problems. Visualizing the learned perceptive fields allows us to make the reasoning of spatial relations interpretable.

• aNCA results surpass current state-of-the-art 2D spatial reasoning methods on various public benchmarks, including Sudoku, image-based Sudoku and maze shortest-path finding.

## 1.1 Related Work

The advancement of spatial reasoning in neural networks has evolved from basic object detection to complex relational logic and artificial dynamic mental simulation. Despite the linguistic fluency of modern multimodal large language models (MLLMs), current research identifies a persistent “semantic-to-geometric gap” where models fail to ground spatial prepositions in coherent geometric world models [3, 1].

Cognitive Frameworks and Taxonomies. To diagnose these systematic failures, recent literature has adopted cognitively grounded taxonomies. The DISE framework [2] categorizes spatial tasks into a $2 \times 2$ matrix based on Intrinsic vs. Extrinsic properties and Static vs. Dynamic environments. While models are proficient in static settings, they exhibit a high failure rate in tasks requiring dynamic mental simulation, primarily due to deficits in “spatial working memory” and an inability to apply basic geometric axioms.

Architectural Inductive Biases. Traditional architectures often lack the structure for relational reasoning. To address this, Relation Networks (RNs) were introduced to explicitly compute pairwise relations between all objects in a set [8]. For more complex scenes, Scene Graph Neural Networks (SGNNs) represent environments as structured graphs where nodes denote objects and edges represent their relationships, utilizing message-passing to propagate context [9, 10]. More recently, mechanisms for Vision Transformers (ViTs), such as Context-Aware Gating (CAG) in the Spatial Decay Transformer (SDT), inject data-dependent spatial decay into self-attention to selectively focus on semantically relevant regions while maintaining a locality bias [11, 12].

General Spatial Benchmarking. No generally accepted closed-form definition reduces spatial reasoning ability to a single numerical score. Evaluation therefore relies on proxy problems that isolate particular spatial capabilities; designing such benchmarks is itself an active field of research, and their results provide task-specific rather than universal measures [2]. Benchmarking spatial reasoning abilities has transitioned from early synthetic 2D/3D datasets like CLEVR [13] and GQA [14] to complex 3D and multi-hop scenarios. Benchmarks such as StepGame [15] and SpartQA [16] test multi-step logical inference in text and vision. For 3D awareness, 3DSRBench [17] and Ego3D-Bench evaluate robustness across varying camera viewpoints, while SITE [18] and OmniSpatial [19] provide comprehensive evaluations across diverse spatial factors including figural and environmental scales. SpatiaLQA emphasizes “spatial logical reasoning” within open-vocabulary indoor environments [20].

Spatial Reasoning on 2D (Grid) Data. While most of the previous benchmarks focus on spatial reasoning in 3D scenes (or 3D→2D projections of these), there is a growing body of recent works that focus on pure 2D spatial reasoning: e.g. [7] introduced spatial reasoning with SRN Diffusion Models which operate on images. Another line of work proposed to solve 2D reasoning with Hierarchical Reasoning Models (HRM) [6], an approach which has recently been extended by ItrSA++ [21] and TRM [22].

Benchmarks for 2D Spatial Reasoning. 2D tasks not only provide simpler spatial reasoning problems which allows the study of novel approaches more easily, but also have several practically relevant benchmark applications like solving Sudoku on images of handwritten digits [7], finding the shortest path in a maze [6] or finding spatially local solutions with global constraints [7].

We compare aNCA with state-of-the-art SRM [7], HRM [6], and TRM [22] on the tasks introduced in section 3.

## 2 Neural Cellular Automata

Originating from the seminal work of John von Neumann [23], Cellular Automata (CA) are discrete computational systems studied since the 1940s. Typically, a CA is defined on a regular, finite grid G composed of discrete cells. At each time step t, every cell i maintains a discrete state $s _ { i , t } .$ . The system’s evolution is governed by the synchronous application of a rule set R across all cells, where the subsequent state $s _ { i , t + 1 }$ is derived from the local neighborhood $\bar { \mathcal { N } } ( s _ { i , t } )$ such that $s _ { i , t + 1 } : = \mathcal { R } [ \mathcal { N } ( s _ { i , t } ) ]$ [24]. This process is illustrated in Figure 1.

CAs gained significant prominence in the 1970s via Conway’s “Game of $L i f e ^ { , \mathrm { , } \mathrm { , } }$ [25], a 2D CA utilizing binary states $s _ { ( x , y ) , t } \in \{ 0 , 1 \}$ and 3×3 neighborhood rules. CAs have proven effective in modeling diverse phenomena in biology [26, 27, 28], chemistry [29], and physics [30, 31]. Furthermore, their status as a universal model of computation was confirmed by the proof of Turing Completeness [32] for specific configurations. These developments

![](images/d70a35d056438a2e5c977484853abc8f07377175b9047f60a513adfc41554294.jpg)  
Figure 1: Sketch of a basic 1D Cellular Automaton with three discrete states $s _ { i } \in \{ 0 , 1 , 2 \}$ and seven update rules R, which operate on the sum over a 3-cell neighborhood N. a) shows the inference of complex patterns from an initial state by the rules visualized in b) over time.

underpinned Stephen Wolfram’s “A New Kind Of Science” [33], which posited that complex scientific problems might be better characterized by simple recursive programs like CAs than with differential equations. Traditional CA models, such as those discussed in [33], [25], and [23], utilize manually defined rule sets. While these rules can be enumerated combinatorially for a given state space $s$ and neighborhood size |N|, the search space expands exponentially; for instance, a binary 2D CA with a $3 \times 3$ neighborhood already permits $2 ^ { 2 ^ { 3 \times 3 } } = 2 ^ { 5 1 2 }$ possible rules. To address this complexity, [5] proposed learning rules from data, introducing Neural Cellular Automata (NCA). In this framework, handcrafted rules are replaced by an artificial neural network $f _ { \phi }$ with learnable parameters ϕ. The update step is expressed as:

$$
s _ { i , t + 1 } : = s _ { i , t } + f _ { \phi } [ \mathcal { N } ( s _ { i , t } ) ]\tag{1}
$$

Since $f _ { \phi }$ is applied identically to every position i (spatial invariance), [5] utilized “nested” Convolutional Neural Networks (CNNs) [35] for efficient implementation. Here, convolutional filters function as learnable local rules over kernel-defined neighborhoods. This architecture is depicted in Figure 2. While [36] established the theoretical equivalence between CA and such CNNs, most NCA models depart from tradition by employing continuous vector-valued states $s _ { i , t } \in \mathbb { R } ^ { n }$ rather than discrete ones.

Initial work by [5] demonstrated robust self-organization and classification using single-layer convolutional NCAs. This has since expanded into more expressive architectures: [37] applied larger CNNs for segmentation, [38] introduced attention-based variants and [39] real-time dynamic texture synthesis using NCAs. Furthermore, NCAs have been integrated with GANs [40], VARs [41], and diffusion models [42] for generative tasks, while [43] extended the paradigm to graph-structured data. Due to their inherent spatial structure, NCAs intuitively appear to be well suited for (2D) spatial reasoning tasks. However, to the best of our knowledge, this has never been attempted before.

![](images/e9ff048bf6ad3efcf8c2182c1479b15fd2b09adc36c008a2608e58d17b38182a.jpg)  
Figure 2: Sketch of a basic CNN implementation [34] of a 2D NCA with a 3D state space. The perception module processes the cell state using one of several configurable perception options. The update module processes the perceived features to compute state updates. Stochastic updates are applied before updating the visible and hidden channels to form the new cell state for the next iteration. The sample pool is used after one batch of images is processed to retain and reuse evolved states for subsequent training iterations.

## 2.1 NCA Perception Modules

Building on eq. (1), we follow the notation from [34] and factor the learnable update rule $f _ { \phi }$ into two distinct sub-modules: a perception module $\mathcal { P } _ { \phi }$ and an update module $\mathcal { U } _ { \phi }$ , and re-formulate the NCA iteration as:

$$
s _ { i , t + 1 } = s _ { i , t } + \mathcal { U } _ { \phi } \big [ \mathcal { P } _ { \phi } ( s _ { i , t } ) \big ] .\tag{2}
$$

The perception module $\mathcal { P } _ { \phi }$ determines how each cell reads information from its neighborhood and therefore controls the effective receptive field and communication structure of the automaton, while the update module $\mathcal { U } _ { \phi }$ transforms the perceived features into the state increment applied to the cell. In our implementation, the cell state is first concatenated channel-wise with the task condition and, when enabled, fixed positional embeddings. The resulting tensor is processed by one or more perception branches operating in parallel; their outputs are concatenated and passed to a shared $1 \times 1$ convolutional update network that predicts the state increment. Because we keep $\mathcal { U } _ { \phi } ,$ the cell-state structure, and the training protocol fixed across experiments, any difference in task performance can be attributed to the inductive bias of $\mathcal { P } _ { \phi } .$ . Figure 2 illustrates how these components interact within a single NCA step. In the remainder of this section we describe three perception variants: standard k × k convolution $( \mathbf { e . g . ~ 3 \times 3 ~ o r ~ 5 \times 5 } )$ , dilated convolution and a Sudoku-specific constraint-aligned variant that combines row, column, and $3 \times 3 – \mathbf { b o x }$ convolutions.

Standard $k \times k$ convolution. Our baseline perception is a single learnable $k \times k$ convolution layer with 128 kernels and leaky-ReLU activation and circular padding, following the original formulation of [5]. Each cell’s receptive field is therefore limited to its immediate neighbors per step, and longer-range dependencies must be built up recurrently.

Dilated Convolutions. We propose dilated convolutions [44] as an NCA perception module for spatial reasoning tasks requiring long-range communication. A dilated $3 \times 3$ kernel with rate r samples a $( 2 r + 1 ) \times ( 2 \bar { r } + 1 )$ grid region at no extra parameter cost, directly enlarging the per-step receptive field. In the NCA setting, fewer iterations are needed to propagate information across the grid, which is a particularly effective inductive bias for tasks where constraints span the entire grid.

Manually Constraint-Aligned Convolutions. For tasks whose constraint structure is known a priori, we propose to integrate that structure manually into the perception module itself rather than asking the NCA to discover it over many iterations. For Sudoku in particular, every valid solution must satisfy three constraint families, one per row, column, and $3 \times 3$ box, each of which spans an entire "unit" of the $9 \times 9 ~ \mathrm { g r i d }$ . We therefore combine three parallel perception branches whose kernels match the shapes of these units: $\mathbf { a \ 1 \times 9 }$ row convolution, a $9 \times 1$ column convolution, and a structured $3 \times 3$ box perception that aggregates the other cells in each $3 \times 3$ box.

## 2.2 Training NCA

Following [34], we train the NCA by unrolling the update rule for a randomly sampled number of steps $\dot { T } \sim \mathcal { U } [ T _ { \mathrm { m i n } } , T _ { \mathrm { m a x } } ]$ and comparing the resulting state to the target output under a task-specific objective. For conditional tasks such as Sudoku and maze solving, each cell is initialized from the puzzle input and conditioned on the same input throughout the rollout, so that the observable evidence remains available at every step. Gradients are backpropagated through the full unrolled trajectory. To further encourage temporal stability, we adopt the sample pool strategy of [5]: a fraction of each mini-batch is initialized from previously evolved states drawn from a fixed-size buffer rather than from fresh seeds, and the newly evolved states are written back into the pool after each training step. This persistent-state regime exposes

![](images/3350b6ef721742fa7d04ec6e1cdaa9fb677785e299ad75105265b0383f60975a.jpg)  
Figure 3: The four 2D spatial reasoning tasks evaluated in this work. Each column shows the input (top) and target (bottom).

the update rule to intermediate configurations far from the initial seed and discourages it from collapsing or diverging on longer rollouts, which is essential for tasks that require many iterations to propagate information across the grid. Full hyperparameter details for each task are listed in sections A.3.1, A.5.1 and A.6.1 in the Appendix.

## 3 Spatial Reasoning with NCA

To the best of our knowledge, we are the first to propose the use of NCAs for (2D) spatial reasoning tasks. We cast each spatial reasoning task as a recurrent computation on a 2D grid G, solved by the NCA defined in eq. (1). All tasks share the same architecture, optimizer, and sample-pool training regime: we unroll for $T \sim \mathcal { U } [ T _ { \mathrm { m i n } } , \bar { T } _ { \mathrm { m a x } } ]$ steps, optimize with AdamW under gradient clipping, and concatenate the condition channel-wise into the cell state at every step. Because high-difficulty Sudoku puzzles admit many valid completions, path-finding admits shortest-path ties, and the color-balance task has no per-pixel supervision, we use constraint-based objectives that reward any valid output. The four tasks, illustrated in fig. 3, use established benchmarks from prior work.

Sudoku. We use the one-million-puzzle dataset of [7], generating puzzles on the fly by masking cells at a configurable difficulty; a held-out set of solutions is never seen during training. Each cell carries $D = 9$ visible digit channels and the condition $c _ { i } \in \{ 0 , 1 \}$ marks clue cells. The objective exploits the puzzle’s constraint structure: letting $p _ { i , d }$ be the softmax probability of digit d at cell i, we penalize the squared deviation of per-digit unit sums from one across all rows, columns, and $3 \times 3$ boxes, rewarding any structurally valid completion independently of which specific digits the NCA commits to. Training hyperparameters, loss details, and computational cost are provided in section A.3.

MNIST Sudoku. We reuse the same solutions and split, rendering each cell as a handwritten glyph drawn from MNIST to give a $9 s \times 9 s$ image where each $s \times s$ tile is blank or carries a digit. The condition is a binary pixel mask over given cells. End-to-end NCA training exposed a trade-off: rollouts either produced crisp MNIST glyphs without satisfying Sudoku constraints, or satisfied constraints while letting glyphs drift. We therefore train a separate Sudoku codec (fig. 4): a frozen encoder $E _ { \psi }$ maps a pixel Sudoku to a $9 \times 9 \times 9$ digit-logit grid and a frozen decoder $D _ { \omega }$ renders a board back to pixels via a conditional VAE with a style latent. The NCA operates entirely in board latent space under the same constraint objective as the array variant, with per-digit probabilities supplied by a frozen MNIST-CNN applied to the decoded tiles rather than read from raw digit channels. Codec architecture, training details, and an isolation evaluation of encoder and decoder accuracy are provided in section A.4.

Maze solving. We use the 30 × 30 maze dataset of [6] (1K train, 1K test), with shortest-path targets. The condition encodes walls, start, and goal. The target is a binary shortest-path mask. Since walls are trivially predictable from the condition, the loss is restricted to non-wall cells and combines a Dice term (handling path/non-path imbalance), a BCE term (encouraging binary commitment), and a normalized length penalty $\lvert \mathrm { l e n } _ { \mathrm { p r e d } } - \mathrm { l e n } _ { \mathrm { t r u e } } \rvert$ . Training hyperparameters and loss details are provided in section A.5.

Global color balance. We adapt the global-constraint task of [7] to black-and-white pixels. Starting from a randomly biased binary image, the NCA must fill an unconditioned grid to realize a prescribed black/white ratio (e.g. 50/50), with no per-pixel supervision. The objective combines a count term pushing the global color mean toward the target ratio and a saturation term that zeroes only at fully committed pixel values. Training hyperparameters and loss details are provided in section A.6.

![](images/d4194b24d451388707bb8bff22284a92ea169c63940c65c2dce9c8c495c6f49d.jpg)  
Figure 4: Latent pixel Sudoku pipeline. The encoder $E _ { \psi }$ maps the initial handwritten Sudoku image into a $9 \times 9 \times 9$ grid of digit logits, the NCA iterates in this latent board space under the Sudoku constraint loss until a valid completion emerges, and the decoder $D _ { \omega }$ renders the final board back to pixels. Freezing $E _ { \psi }$ and $D _ { \omega }$ decouples perception from reasoning and allows different NCA variants to be evaluated within the same image-space pipeline.

## 4 Adaptive Neural Cellular Automata (aNCA)

We propose Adaptive Neural Cellular Automata (aNCA), which extend the reasoning abilities from learned space- and time-invariant update rules of traditional NCA to a more powerful and flexible input-dependent, space- and time-variant regime. We achieve this significant change in NCA capabilities through a multi-head perception module based on DCNv2 deformable convolutions [45, 46], which instantiates the NCA framework of eq. (2) as perception module $\mathcal { P } _ { \phi }$ and allows the effective neighborhood $\mathcal { N }$ to adapt across grid positions and NCA steps. Figure 5 illustrates the architecture: fig. 5a contrasts the fixed sampling patterns of the perception variants above against the adaptive sampling of the proposed deformable perception, fig. 5b shows the full single-step aNCA computation graph. The deformable perception module is formalized below.

![](images/cde6bbf11845e71c1783d2204bae03da2eae256d9cc36e153ceba4a1186d34d9.jpg)  
Figure 5: (a) Comparison of perception sampling patterns: standard $3 \times 3 ,$ dilated, constraint-aligned, and deformable convolution. (b) One aNCA update step: the perception module $\mathcal { P } _ { \phi }$ predicts offsets $\Delta { \bf p } _ { n } ( { \bf p } )$ and modulations $m _ { n }$ from $s _ { t } ,$ which guide the deformable convolution to aggregate features from adaptive grid locations. The update module $\mathcal { U } _ { \phi }$ then produces the next state $s _ { t + 1 }$ .

Adaptive Perception via Deformable Convolutions. Unlike dilated convolutions, which fix the neighborhood geometry at initialization, deformable convolutions [45] augment each kernel tap with a learned, position-dependent spatial offset. Rather than committing to a fixed sampling pattern, the model learns which positions each cell should attend to, depending on the current input. This allows it to discover long-range constraint structures from data without explicit apriori modeling of these structures. Concretely, a standard $k \times$ k convolution at grid position p reads

$$
\mathcal { P } ( s , \mathbf { p } ) = \sum _ { n = 1 } ^ { k ^ { 2 } } w _ { n } \cdot s \left( \mathbf { p } + \mathbf { p } _ { n } \right) ,\tag{3}
$$

where $\mathbf { p } _ { n }$ enumerates the $k ^ { 2 }$ fixed kernel positions and $w _ { n }$ are learnable weights. We use the DCNv2 variant [46], which augments each kernel position with both a spatial offset $\Delta \mathbf { p } _ { n } ( \mathbf { p } ) \in \mathbb { R } ^ { 2 }$ and a learned modulation scalar $m _ { n } ( \mathbf { p } ) \in [ 0 , 2 ]$

each predicted by a separate auxiliary convolution applied to $s _ { i , t } .$ . Its difference from the offset-only DCNv1 variant is detailed in section A.2.1. Fractional positions are resolved by bilinear interpolation:

$$
\mathcal { P } _ { \phi } ^ { \mathrm { d e f } } ( s , \mathbf { p } ) = \sum _ { n = 1 } ^ { k ^ { 2 } } w _ { n } \cdot m _ { n } ( \mathbf { p } ) \cdot s ( \mathbf { p } + \mathbf { p } _ { n } + \Delta \mathbf { p } _ { n } ( \mathbf { p } ) ) .\tag{4}
$$

The modulation scalar allows each kernel position to be suppressed or amplified independently, giving the model additional flexibility beyond the spatial offset alone. We stack $H = 3$ deformable branches in parallel as ${ \mathcal { P } } _ { \phi } ,$ , producing $3 \times ( 2 + 1 ) \times 9 = 8 1$ offset and modulation values per cell per step. Different heads can thereby specialize to complementary constraint directions. Because the learned offsets are explicit and inspectable, we can directly audit which spatial relations the NCA has discovered. In Section 5.1 we show that the three heads learn sampling patterns that align with Sudoku’s row, column an box constraints, without encoding this explicitly in the architecture.

Our approach significantly differs from the "deformable convolution" implemented in in the NCATorch framework of [34] - there a single deformable convolution based on [45] has been (unsuccessfully) used for generative tasks and is not identical with our formulation (which is based on [46]. In Contrast to [34], our per-tap modulation lets a cell suppress or amplify individual sampling positions rather than only displace them. This allows different heads to specialize to complementary constraint directions, which is critical for spatial reasoning.

## 5 Empirical Evaluation

## 5.1 Solving Sudoku on Array Representations

We evaluate on the held-out test split of the SRM Sudoku dataset [7], grouping puzzles by difficulty into Easy (1–27 removed cells), Medium (28–54), and Hard (55–81) buckets. Two metrics are reported per bucket: the Average Solve Rate (ASR), the fraction of test puzzles whose final NCA state is a valid Sudoku completion of the given clues, and the Average Cell Precision (ACP), the fraction of non-clue cells whose predicted digit is consistent with a valid completion. Because high-difficulty puzzles often admit several valid completions, ASR is the primary metric.

Table 1 compares the perception variants at matched cell-state size and network capacity, reporting mean and standard deviation over four training runs each. All variants solve Easy and Medium puzzles near-perfectly, so the evaluation performance is decided on the Hard buckets, where aNCA leads on both metrics while remaining comparable in parameter count. The aNCA performance gain can not be explained by receptive-field size or capacity: i.e. enlarging the fixed kernel beyond $5 \times 5$ does not increase the performance, $7 \times 7$ and $9 \times 9$ variants even perform worse on Hard despite using more parameters then aNCAs. The $9 \times 9$ variant is also markedly unstable across seeds (±37.15 on Hard), whereas the smaller fixed kernels are reproducible.

Table 1: Sudoku on the array representation, reported as mean ± standard deviation over four training runs per perception. Best Hard-bucket ASR/ACP in bold.
<table><tr><td rowspan="2">Perception</td><td colspan="2">Easy [1, 27]</td><td colspan="2">Medium [28, 54]</td><td colspan="2"> $\mathrm { H a r d } [ 5 5 , 8 1 ]$ </td><td rowspan="2">Param.</td></tr><tr><td>ASR</td><td>ACP</td><td>ASR</td><td>ACP</td><td>ASR</td><td>ACP</td></tr><tr><td>NCA w. 3x3 Conv.</td><td> $1 0 0 \pm 0$ </td><td>100 ± 0</td><td> $8 6 . 3 5 \pm 1 . 0 0$ </td><td> $9 8 . 8 5 \pm 0 . 1 3$ </td><td> $7 8 . 2 3 \pm 1 . 8 4$ </td><td> $9 8 . 6 3 \pm 0 . 1 0$ </td><td>50k</td></tr><tr><td>NCA w. 5x5 Conv.</td><td> $1 0 0 \pm 0$ </td><td>100 ± 0</td><td> $8 7 . 4 7 \pm 1 . 6 5$ </td><td> $9 8 . 9 3 \pm 0 . 1 2$ </td><td> $7 5 . 0 3 \pm 8 . 3 8$ </td><td> $9 6 . 2 7 \pm 4 . 0 5$ </td><td>105k</td></tr><tr><td>NCA w. 7x7 Conv.</td><td>100 ± 0</td><td>100 ± 0</td><td> $8 2 . 7 0 \pm 1 . 6 2$ </td><td> $9 8 . 4 3 \pm 0 . 2 1$ </td><td> $7 6 . 9 5 \pm 2 . 0 7$ </td><td> $9 8 . 4 3 \pm 0 . 1 5$ </td><td>189k</td></tr><tr><td>NCA w. 9x9 Conv.</td><td>100 ± 0</td><td>100 ± 0</td><td>71.58 ± 18.82</td><td> $9 5 . 8 8 \pm 4 . 4 1$ </td><td> $3 9 . 8 0 \pm 3 7 . 1 5$ </td><td> $8 4 . 9 5 \pm 1 9 . 7 1$ </td><td>300k</td></tr><tr><td>NCA w. R/C/B Conv.</td><td>100 ± 0</td><td>100 ± 0</td><td>87.38 ± 2.64</td><td> $9 8 . 9 0 \pm 0 . 2 2$ </td><td> $7 8 . 1 8 \pm 2 . 8 1$ </td><td> $9 8 . 5 8 \pm 0 . 1 9$ </td><td>145k</td></tr><tr><td>NCA w. Dilated Conv.</td><td>100 ± 0</td><td>100 ± 0</td><td>86.98 ± 1.64</td><td> $9 8 . 8 8 \pm 0 . 1 7$ </td><td> $7 7 . 2 3 \pm 1 0 . 4 3$ </td><td> $9 8 . 4 8 \pm 0 . 6 7$ </td><td>150k</td></tr><tr><td>NCA w. DCNv1</td><td> $9 9 . 9 \pm 0 . 1 4$ </td><td>100 ± 0</td><td> $7 5 . 7 8 \pm 6 . 9 3$ </td><td> $9 7 . 7 5 \pm 0 . 7$ </td><td> $6 4 . 5 5 \pm 1 0 . 4 3$ </td><td> $9 7 . 5 \pm 0 . 5 8$ </td><td>159k</td></tr><tr><td>aNCA</td><td> $\overline { { 1 0 0 \pm 0 } }$ </td><td>100 ± 0</td><td> $\overline { { { \bf 8 9 . 5 0 \pm 4 . 6 6 } } }$ </td><td> $\mathbf { \overline { { 9 9 . 0 8 \pm 0 . 4 0 } } }$ </td><td> $\overline { { 8 7 . 7 8 \pm 6 . 7 4 } }$ </td><td> $\mathbf { 9 9 . 1 8 \pm 0 . 4 6 }$ </td><td>165k</td></tr></table>

Visualizing Learned Spatial Relations. A central appeal of the adaptive perception module is that the predicted per-cell offsets are themselves interpretable: for every Sudoku cell position at every timestep, they allow to reconstruct to which other cells the model learned to attend. We exploit this to ask whether the adaptive perception learns Sudoku’s row, column, and 3 × 3-box constraint graph from data, despite this is never explicitly modeled. For each test puzzle we run the trained NCA for the standard rollout horizon and record every head’s tap offsets at the final step. Fixing one source cell on the $9 \times 9$ board, we then aggregate where the nine taps of each head end up sampling from that cell across held-out testset puzzles. Figure 6 shows the result as one heatmap per head, for two interior source cells ((3, 3) at the intersection of the top-left $3 \times 3$ box with the center box, and (4, 4) at the center of the board).

Dashed rectangles overlay the source cell’s row, column, and $3 \times 3$ box, so the reader can directly compare the learned hot cells against Sudoku constraints. Across both positions and all heads, the heat concentrates inside those three rectangles. The heads also specialize, with different deformable convolutions preferring the row, column, or box direction, so that together they cover the full constraint graph of the source. Quantitative alignment scores are reported in table 7.

![](images/f6425426454964e5326f6201d75a442d11077719b8fa6277b06e389c10ffdca8.jpg)

![](images/0a30bd8805579cbbf89aad95e47aa12e6cdae4e64c6ee7222ec7bba22d219ebb.jpg)  
(a) Source cell (3, 3).

![](images/84429ece5748906e5bd4116ff7e32acecefab20f79f31f6d2bc4e1db749a129c.jpg)

![](images/f0fdc7b2fc953859879412ae8621df4b277a5fa458ddfb1bedad653710bd4af1.jpg)

![](images/1c12e4cd1531845f52c2896b82cf1a1a64aa557fb7a51b85d0839ab49bd6bc24.jpg)

![](images/f29e056f72eb5a523b27c325824034e34bafbd58ea4957afe386b208270f719d.jpg)  
(b) Source cell (4, 4).  
Figure 6: Learned sampling reach of each deformable perception head from two interior source cells (black-bordered squares at grid positions (3, 3) and (4, 4)). Each panel is one head: the heatmap shows the fraction of the head’s total sampling weight placed on every cell of the $9 \times 9$ board, aggregated over held-out test puzzles at the final NCA step. Dashed overlays mark the source cell’s row (green), column (orange), and 3 × 3 box (purple). The learned heat concentrates inside these overlays and the three heads specialize to complementary constraint directions.

## 5.2 Solving Sudoku on Images

We extend the perception comparison to the image-level Sudoku task, in which each cell is rendered as an MNISTstyle digit glyph [7]. The NCA operates in the latent board space defined by the frozen encoder–decoder pipeline of fig. 4. Held-out splits, difficulty buckets, and metrics are reused, with ASR evaluated with a frozen MNIST classification CNN on the decoded pixel image. We further compare against the two image-space Sudoku solvers released with the dataset [7]: a 118M-parameter diffusion model and SRM at the same capacity. Unlike these end-to-end baselines, our pipeline decouples digit recognition from constraint reasoning via the frozen codec. This modularity potentially allows the same reasoner to operate on any object encoding for which a codec maps inputs to the board representation. Codec isolation accuracy is reported in section A.4.3. Table 2 reports the results. aNCA surpass both reference models on every difficulty bucket while using fewer parameters. Among

Table 2: MNIST Sudoku. Results in ASR (Average Solve Rate), reported as mean ± standard deviation over four training runs. Best Hard-bucket ASR in bold. Per-bucket ACP is reported in table 9.
<table><tr><td>Method</td><td>Easy [1,27]</td><td>Medium [28,54]</td><td>Hard [55,81]</td><td>Param.</td></tr><tr><td>Diffusion Model SRM</td><td>99.4 99.8</td><td>53.6 75.4</td><td>0.8 51.6</td><td>118M 118M</td></tr><tr><td>NCA 3x3 Conv.</td><td> $9 9 . 4 0 \pm 0 . 1 4$ </td><td> $8 0 . 9 0 \pm 0 . 9 0$ </td><td> $7 1 . 9 8 \pm 1 . 6 1$ </td><td>Codec + NCA</td></tr><tr><td>NCA 5x5 Conv.</td><td> $9 9 . 1 3 \pm 0 . 5 0 $ </td><td> $7 9 . 9 7 \pm 1 . 7 0$ </td><td> $6 9 . 0 0 \pm 6 . 6 3$ </td><td>1.1M + 50k 1.1M + 105k</td></tr><tr><td>NCA 7x7 Conv.</td><td> $9 8 . 7 3 \pm 0 . 1 0$ </td><td> $7 6 . 5 5 \pm 1 . 2 1$ </td><td> $7 2 . 3 0 \pm 2 . 6 0$ </td><td>1.1M + 189k</td></tr><tr><td>NCA 9x9 Conv.</td><td> $9 9 . 3 0 \pm 0 . 2 3 $ </td><td>66.73 ± 16.99</td><td> $3 5 . 8 3 \pm 3 3 . 5 6$ </td><td>1.1M + 300k</td></tr><tr><td>NCA R/C/B Conv.</td><td> $9 9 . 2 0 \pm 0 . 0 8$ </td><td> $8 0 . 3 8 \pm 1 . 4 1$ </td><td> $7 1 . 2 5 \pm 2 . 6 1$ </td><td>1.1M + 145k</td></tr><tr><td>NCA Dilated Conv.</td><td> $9 9 . 2 5 \pm 0 . 1 7$ </td><td> $8 0 . 7 8 \pm 0 . 9 5$ </td><td> $7 0 . 9 5 \pm 8 . 3 9$ </td><td>1.1M + 150k</td></tr><tr><td>NCA DCNv1</td><td> $9 8 . 4 3 \pm 0 . 5 4$ </td><td> $7 0 . 1 5 \pm 6 . 7 9$ </td><td> $6 0 . 1 3 \pm 9 . 7 5$ </td><td>1.1M + 159k</td></tr><tr><td>aNCA</td><td> $\overline { { 9 9 . 0 0 \pm 0 . 3 7 } }$ </td><td>81.90 ± 5.50</td><td> $\overline { { 8 2 . 1 0 \pm 4 . 8 9 } }$ </td><td> $\overline { { 1 . 1 \mathbf { M } + 1 6 5 \mathbf { k } } }$ </td></tr></table>

the NCA variants themselves, the ordering matches the array task, with aNCA ahead on Medium and Hard. The parameter column lists the shared codec and the NCA separately. The reported NCAs are the exact checkpoints evaluated in table 1 on the array task.

![](images/c0cfecfcc5f85e99d2f88ed665a6c0e879ec1d4ca711be5b321e806c18351786.jpg)

![](images/ee0af5c303283d50a1e6bd28c2124a16c970ec70977327e36dd2b6cc972dcaac.jpg)  
(a)

![](images/9d1b850c4be22dc9c53e92c7272a6237b412e1eb3a9c39ec8c1f94f93fc3a504.jpg)  
(b)  
Figure 7: Output diversity of the Sudoku codec. (a) For an underconstrained clue pattern with at least $1 0 ^ { 6 }$ valid completions, repeated stochastic NCA rollouts produce multiple distinct valid solutions. (b) For a puzzle with a unique solution, resampling the decoder style latent preserves the solved board while varying the digit style.

Output Diversity. Highly underconstrained Sudoku puzzles admit many valid completions, so matching the dataset target is neither necessary nor desirable. Sampling the same NCA 1000 times on a puzzle with at least $1 0 ^ { 6 }$ valid completions yields valid completions in 997/1000 rollouts and 303 distinct boards (fig. 7a). For a puzzle with a unique solution, resampling produces identical boards but varied digit stroke styles (fig. 7b).

## 5.3 Solving Maze Puzzles

![](images/52a0545eee416f203f07bb662995e320ed05daed909bc5d824952b675a6f1ecc.jpg)

![](images/25dc3db934a4875e38c11186ed8c54353cb98b8b8729ee82bdaef6f126ed12c2.jpg)  
Figure 8: (a) Maze solving: the model receives walls (black), start (green), and goal (red). Blue cells indicate correct path predictions matching the target; yellow cells are predicted path pixels that deviate from the target. (b) Generalization to unseen $3 0 \times 3 0$ Prim-style maze structures.

We evaluate on the $3 0 \times 3 0$ mazes described in section 3 and report two graph-level metrics: Valid Path, the fraction of predictions that connect start and goal without crossing walls, and Valid Optimal, the fraction whose path length matches the shortest-path target.

Table 3 shows that purely local perception is insufficient: the 3 × 3 NCA almost never forms a complete path. Enlarging the communication radius closes much of this gap, with dilated perception reaching 64.30% optimality. Adaptive NCAs perform best, producing valid paths for 98.56% of mazes and optimal paths for 95.2%, surpassing HRM, ItrSA++ and TRM-Att while using fewer parameters. The R/C/B variant also transfers well despite being designed for Sudoku, but trails deformable perception, suggesting that content-adaptive routing is more useful than a fixed constraint geometry on mazes. Same-resolution transfer to a procedurally distinct maze distribution is reported in section A.5.3.

## 5.4 Learning to Adhere Global Constraints

The color-balance task tests whether the NCA can satisfy a constraint defined over the entire grid, without local target labels or a conditioning image. We evaluate the output by its absolute color-count error and exact-ratio accuracy, reporting mean and standard deviation across independent training runs in table 4. Table 4 shows that aNCA achieves the highest mean exact-ratio accuracy among the compared models while using substantially fewer parameters than the diffusion and SRM references. The $5 \times 5 \ : \mathrm { N C A }$ remains competitive and achieves the lowest mean count error. These metrics capture different aspects of performance: exact accuracy measures how often the global constraint is satisfied, whereas mean count error measures the average deviation from the 50/50 balance.

Table 3: Maze solving on $3 0 \times 3 0$ grids, reported as mean ± standard deviation over five training runs. Valid Path checks connectivity and wall avoidance. Valid Optimal additionally requires shortest-path length.
<table><tr><td>Method</td><td>Source</td><td>Valid Path</td><td>Valid Optimal |Param.</td><td></td></tr><tr><td>HRM</td><td>[6]</td><td>★</td><td>74.5%</td><td>27M</td></tr><tr><td>ItrSA++</td><td>[21]</td><td>1*</td><td>78.6%</td><td>3M</td></tr><tr><td>TRM-Att</td><td>[22]</td><td>★</td><td>85.3%</td><td>7M</td></tr><tr><td>NCA w. 3x3 Conv.</td><td>Ours</td><td> $2 . 5 4 \pm 0 . 8 4$ </td><td> $0 . 0 8 \pm 0 . 0 8$ </td><td>50k</td></tr><tr><td>NCA w. 5x5 Conv.</td><td>Ours</td><td> $2 4 . 7 8 \pm 1 . 6 8$ </td><td> $1 0 . 7 8 \pm 1 . 2 1$ </td><td>105k</td></tr><tr><td>NCA w. Dilated Conv.</td><td>Ours</td><td> $7 4 . 2 6 \pm 1 . 0 7$ </td><td> $6 4 . 3 0 \pm 1 . 3 2$ </td><td>150k</td></tr><tr><td>NCA w. R/C/B Conv.</td><td>Ours</td><td> $9 3 . 7 8 \pm 1 . 9 9$ </td><td> $8 1 . 2 6 \pm 2 . 0 1$ </td><td>280k</td></tr><tr><td>NCA w. DCNv1</td><td>Ours</td><td> $8 6 . 7 3 \pm 4 . 5 8$ </td><td> $7 7 . 4 8 \pm 6 . 4 2$ </td><td>236k</td></tr><tr><td>aNCA</td><td>Ours</td><td> $\overline { { 9 8 . 5 6 \pm 0 . 7 2 } }$ </td><td> $\overline { { 9 4 . 1 8 \pm 1 . 1 4 } }$ </td><td>256k</td></tr></table>

indicates metrics not reported in prior work.

Table 4: Global color balance. NCA results are mean ± standard deviation across independent training runs. Imbalance measures absolute color-count error. Accuracy measures exact satisfaction of the 50/50 ratio.
<table><tr><td>Perception</td><td>Source Imbalance ↓ Accuracy ↑ |Param.</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Diffusion SRM</td><td>[7]</td><td>1.27</td><td>25.0%</td><td>19.7M</td></tr><tr><td>[7]</td><td>0.53</td><td>51.8%</td><td>19.7M</td></tr><tr><td>NCA w. 3x3 Conv.</td><td>Ours</td><td> $1 . 1 3 { \pm } 0 . 1 7$ </td><td>26.0%±5%</td><td>48k</td></tr><tr><td>NCA w. 5x5 Conv.</td><td>Ours</td><td> ${ \bf 0 . 0 7 \pm 0 . 0 6 }$ </td><td>95.3%±2%</td><td>100k</td></tr><tr><td>NCA w. DCNv1</td><td>Ours</td><td> $1 . 6 1 { \pm } 0 . 5 9$ </td><td> $1 8 . 7 \% \pm 1 1 \%$ </td><td>147k</td></tr><tr><td>aNCA</td><td>Ours</td><td> $\overline { { 0 . 1 \pm 0 . 0 8 } }$ </td><td> $9 6 . 8 \% \pm 3 \%$ </td><td>152k</td></tr></table>

These results show that adaptive perception also helps for a spatially uniform objective. In table 4, aNCA with modulated DCNv2 clearly improves exact accuracy over offset-only DCNv1, while gains over the $5 \times 5$ baseline are smaller and within run-to-run variability (also see fig. 10 for qualitative results).

## 6 Conclusion

We introduced Adaptive Neural Cellular Automata (aNCA), which implement a conceptually novel perception approach to enable NCAs to learn reasoning capabilities. On spatially structured tasks, Sudoku, image-based Sudoku, and maze solving, aNCA sets new state-of-the-art results while using a fraction of the parameters of prior methods. The learned perceptions are directly inspectable, revealing that the model learns task-relevant spatial relations (e.g. Sudoku rows, columns, and boxes) without explicit encoding in the architecture.

Limitations. While aNCA achieves strong results on spatially structured tasks. However, its adaptive perception comes at higher per-step computational cost than fixed-convolution baselines. Detailed resource measurements are provided in section A.3.3. The current architecture targets regular 2D grids. Adapting its learned perception mechanism to irregular graphs, 3D scenes, and variable-resolution inputs is a promising direction for future work.

## AI use statement

In this work, we used generative AI tools as a coding assistant and for language editing, to revise and shorten passages of the manuscript that we had drafted ourselves. Generative AI tools were not used to generate research ideas, to design the proposed architecture, to produce or interpret experimental results, or to retrieve or summarize related work. All code was reviewed and tested by the authors, and every reported number was produced by running that code on the checkpoints described in section A.2. We have reviewed all AI-assisted text and take full responsibility for the final content of this work, including all text, claims and artifacts.

## Ethics statement

This work studies architectures for 2D spatial reasoning on synthetic and publicly available benchmarks. It involves no human subjects, no personally identifiable information, and no user data. All datasets used are public and were employed within their intended scope: the Sudoku and MNIST-Sudoku data of [7] and the maze dataset of [6]. We release no new dataset. The proposed models are small and were trained on a single consumer GPU, so the computational footprint of this work is modest. We see no direct pathway from puzzle solving to harmful application, although improved spatial reasoning is a general capability whose downstream uses we cannot fully anticipate. The authors declare no conflicts of interest.

## Reproducibility statement

All datasets and benchmarks used in this work are publicly available and are cited in section 3. Model and training hyperparameters for each task are given in sections A.3.1, A.5.1 and A.6.1, the exact training objectives in sections A.3.2, A.5.2 and A.6.2, and the encoder-decoder codec used for MNIST Sudoku in sections A.4.1 and A.4.2, together with an isolation evaluation of its accuracy in section A.4.3. Computational cost and hardware are reported in section A.3.3 and section A.2. All reported results are means and standard deviations over multiple independent training runs rather than single runs, with per-seed behavior discussed in section A.3.4. The full source code, including the configuration files reproducing every reported run, will be released upon publication as a plugin for the NCATorch framework of [34].

## References

[1] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 9161–9175, 2023.

[2] Xinmiao Huang, Qisong He, Zhenglin Huang, Boxuan Wang, Zhuoyun Li, Guangliang Cheng, Yi Dong, and Xiaowei Huang. Spatial-dise: A unified benchmark for evaluating spatial reasoning in vision-language models. arXiv preprint arXiv:2510.13394, 2025.

[3] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

[4] Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, Jie Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, et al. Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys, 58(3):1–38, 2025.

[5] Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. Growing neural cellular automata. Distill, 2020. https://distill.pub/2020/growing-ca.

[6] Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025.

[7] Christopher Wewer, Bartlomiej Pogodzinski, Bernt Schiele, and Jan Eric Lenssen. Spatial reasoning with denoising models. In International Conference on Machine Learning (ICML), 2025.

[8] Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. Advances in neural information processing systems, 30, 2017.

[9] Lizong Zhang, Haojun Yin, Bei Hui, Sijuan Liu, and Wei Zhang. Knowledge-based scene graph generation with visual contextual dependency. Mathematics, 10(14):2525, 2022.

[10] Quang PM Pham, Khoi TN Nguyen, Lan C Ngo, Truong Do, Dezhen Song, and Truong-Son Hy. Tesgnn: Temporal equivariant scene graph neural networks for efficient and robust multi-view 3d scene understanding. arXiv preprint arXiv:2411.10509, 2024.

[11] Yuxin Mao, Zhen Qin, Jinxing Zhou, Bin Fan, Jing Zhang, Yiran Zhong, and Yuchao Dai. Learning spatial decay for vision transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7945–7953, 2026.

[12] Qihang Fan, Huaibo Huang, Mingrui Chen, Hongmin Liu, and Ran He. Rmt: Retentive networks meet vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5641–5651, 2024.

[13] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

[14] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

[15] Zhengxiang Shi, Qiang Zhang, and Aldo Lipani. Stepgame: A new benchmark for robust multi-hop spatial reasoning in texts. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 11321–11329, 2022.

[16] Roshanak Mirzaee, Hossein Rajaby Faghihi, Qiang Ning, and Parisa Kordjamshidi. Spartqa: A textual question answering benchmark for spatial reasoning. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4582–4598, 2021.

[17] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025.

[18] Wenqi Wang, Reuben Tan, Pengyue Zhu, Jianwei Yang, Zhengyuan Yang, Lijuan Wang, Andrey Kolobov, Jianfeng Gao, and Boqing Gong. Site: towards spatial intelligence thorough evaluation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9058–9069, 2025.

[19] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025.

[20] Yuechen Xie, Xiaoyan Zhang, Yicheng Shan, Hao Zhu, Rui Tang, Rong Wei, Mingli Song, Yuanyu Wan, and Jie Song. Spatialqa: A benchmark for evaluating spatial logical reasoning in vision-language models. arXiv preprint arXiv:2602.20901, 2026.

[21] Kenji Kubo, Shunsuke Kamiya, Masanori Koyama, Kohei Hayashi, Yusuke Iwasawa, and Yutaka Matsuo. Cvoting: Confidence-based test-time voting without explicit energy functions. In The Fourteenth International Conference on Learning Representations, 2026.

[22] Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025.

[23] John Von Neumann, Arthur Walter Burks, et al. Theory of self-reproducing automata, 1966.

[24] Joel L Schiff. Cellular automata: a discrete view of the world. John Wiley & Sons, 2011.

[25] Martin Gardner. Mathematical games - the fantastic combinations of john conway’s new solitaire game "life". Scientific American, 223(4):4, 1970.

[26] Stephen Coombes. The geometry and pigmentation of seashells. Nottingham: Department of Mathematical Sciences, University of Nottingham, 2009.

[27] Y Bouligand. Fibroblasts, morphogenesis and cellular automata. In Disordered Systems and Biological Organization, pages 367–379. Springer, 1986.

[28] Haralampos Hatzikirou, David Basanta, Matthias Simon, Karl Schaller, and Andreas Deutsch. ‘go or grow’: the key to the emergence of invasion in tumour progression? Mathematical medicine and biology: ajournal ofthe IMA, 29(1):49–65, 2012.

[29] Martin Gerhardt and Heike Schuster. A cellular automaton describing the formation of spatially ordered structures in chemical systems. Physica D: Nonlinear Phenomena, 36(3):209–221, 1989.

[30] Stephen Wolfram. Statistical mechanics of cellular automata. Reviews of modern physics, 55(3):601, 1983.

[31] Magdalena Załuska-Kotur, Hristina Popova, and Vesselin Tonchev. Step bunches, nanowires and other vicinal “creatures”—ehrlich–schwoebel effect by cellular automata. Crystals, 11(9):1135, 2021.

[32] Matthew Cook et al. Universality in elementary cellular automata. Complex systems, 15(1):1–40, 2004.

[33] Stephen Wolfram and M Gad-el Hak. A new kind of science. Appl. Mech. Rev., 56(2):B18–B19, 2003.

[34] Martin Spitznagel and Janis Keuper. A new kind of network? review and reference implementation of neural cellular automata, 2026.

[35] Yann LeCun and Yoshua Bengio. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 1998.

[36] William Gilpin. Cellular automata as convolutional neural networks. Physical Review E, 100(3):032402, 2019.

[37] Mark Sandler, Andrey Zhmoginov, Liangcheng Luo, Alexander Mordvintsev, Ettore Randazzo, et al. Image segmentation via cellular automata. arXiv preprint arXiv:2008.04965, 2020.

[38] Mattie Tesfaldet, Derek Nowrouzezahrai, and Chris Pal. Attention-based neural cellular automata. Advances in Neural Information Processing Systems, 35:8174–8186, 2022.

[39] Ehsan Pajouheshgar, Yitao Xu, Tong Zhang, and Sabine Süsstrunk. Dynca: Real-time dynamic texture synthesis using neural cellular automata. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20742–20751, 2023.

[40] Maximilian Otte, Quentin Delfosse, Johannes Czech, and Kristian Kersting. Generative adversarial neural cellular automata. arXiv preprint arXiv:2108.04328, 2021.

[41] Rasmus Berg Palm, Miguel González-Duque, Shyam Sudhakaran, and Sebastian Risi. Variational neural cellular automata. arXiv preprint arXiv:2201.12360, 2022.

[42] John Kalkhof, Arlene Kühn, Yannik Frisch, and Anirban Mukhopadhyay. Frequency-time diffusion with neural cellular automata. arXiv preprint arXiv:2401.06291, 2024.

[43] Daniele Grattarola, Lorenzo Livi, and Cesare Alippi. Learning graph cellular automata. Advances in Neural Information Processing Systems, 34:20983–20994, 2021.

[44] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions, 2016.

[45] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks, 2017.

[46] Xizhou Zhu, Han Hu, Stephen Lin, and Jifeng Dai. Deformable convnets v2: More deformable, better results. CoRR, abs/1811.11168, 2018.

[47] R. C. Prim. Shortest connection networks and some generalizations. The Bell System Technical Journal, 36(6):1389–1401, 1957.

## A Technical appendices and supplementary material

## Contents

A.1 Video Material 12   
A.2 Implementation Details 13   
A.2.1 Difference between DCNv1 and DCNv2 . 13   
A.3 Sudoku 14   
A.3.1 Model and Training Hyperparameters 14   
A.3.2 Training Loss 14   
A.3.3 Computational Cost 15   
A.3.4 Additional Results: Sudoku 15   
A.4 MNIST Sudoku 17   
A.4.1 Codec Hyperparameters 17   
A.4.2 Codec Training Loss 18   
A.4.3 Codec Accuracy 18   
A.4.4 Additional Results: MNIST Sudoku 19   
A.5 Maze 20   
A.5.1 Model and Training Hyperparameters 20   
A.5.2 Training Loss 20   
A.5.3 Additional Results: Mazes 21   
A.6 Global Color Balance 22   
A.6.1 Model and Training Hyperparameters 22   
A.6.2 Training Loss 22   
A.6.3 Additional Results: Global Color Balance 23

## A.1 Video Material

https://youtu.be/WF0Bm6Ie7WA

• 0:00 Sudoku: aNCA Solving (Batch)

• 0:16 Sudoku: aNCA Solving (Hard)

• 0:47 Sudoku: aNCA Output Diversity

• 1:10 Sudoku MNIST: aNCA Solving (Batch)

• 1:47 Sudoku MNIST: aNCA Solving (Hard)

• 2:10 Sudoku MNIST: aNCA Output Diversity (Solutions)

• 2:25 Sudoku MNIST: aNCA Output Diversity (Digit Style)

• 2:42 Sudoku MNIST: aNCA learned spatial relations

• 4:54 Maze 30x30: NCA w. 3x3 Conv.

• 5:07 Maze 30x30: aNCA

• 5:20 Maze DFS 20x20 (OOD): aNCA

• 5:37 Maze Prim 40x40 (OOD): aNCA

• 5:45 Color Balance Task Perception Comparison

## A.2 Implementation Details

All models are implemented in PyTorch. Deformable convolutions use torchvision.ops.deform\_conv2d, the standard torchvision DCNv2 implementation. All experiments were run on a single NVIDIA RTX 4090. To provide reproducibility, the code will be released upon publication, as a plugin for the NCATorch framework of [34].

## A.2.1 Difference between DCNv1 and DCNv2

Both variants predict $2 k ^ { 2 }$ offsets from the current cell state, yielding two displacement coordinates for each of the $k ^ { 2 }$ kernel taps. DCNv1 then samples only at the displaced positions:

$$
\mathcal { P } _ { \phi } ^ { \mathrm { D C N v 1 } } ( s , \mathbf { p } ) = \sum _ { n = 1 } ^ { k ^ { 2 } } w _ { n } \cdot s ( \mathbf { p } + \mathbf { p } _ { n } + \Delta \mathbf { p } _ { n } ( \mathbf { p } ) ) .\tag{5}
$$

DCNv2 additionally predicts one content-dependent modulation value per tap, requiring a further $k ^ { 2 }$ outputs, and weights every sampled feature before aggregation:

$$
\mathscr { P } _ { \phi } ^ { \mathrm { D C N v 2 } } ( s , \mathbf { p } ) = \sum _ { n = 1 } ^ { k ^ { 2 } } w _ { n } \cdot m _ { n } ( \mathbf { p } ) \cdot s ( \mathbf { p } + \mathbf { p } _ { n } + \Delta \mathbf { p } _ { n } ( \mathbf { p } ) ) .\tag{6}
$$

Thus, DCNv1 adapts only where each tap samples, whereas DCNv2 also adapts how strongly each sampled location contributes. In our implementation, $m _ { n } ( \mathbf { p } ) = 2 \sigma ( a _ { n } ( \mathbf { p } ) ) \in [ 0 , 2 ]$ , where $a _ { n }$ is produced by a separate modulation convolution. The offset and modulation predictors are zero-initialized, so the module initially behaves like a regular convolution: offsets are zero and modulation values are one.

## A.3 Sudoku

## A.3.1 Model and Training Hyperparameters

Table 5: Hyperparameters shared across all NCA variants for the Sudoku task. Perception-specific settings (kernel size, dilation, number of deformable heads) vary per variant and are described in section 2.
<table><tr><td>Category</td><td>Hyperparameter Value 24</td></tr><tr><td rowspan="2">NCA state</td><td>Cell channels</td></tr><tr><td>Positional embeddings yes</td></tr><tr><td>Update network</td><td>Architecture MLP (1 ×1 Conv), 1 hidden layer, 128 channels AdamW</td></tr><tr><td rowspan="6">Optimization</td><td>Optimizer Learning rate</td></tr><tr><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule WSD (warmup-stable-decay) 10% / 80% / 10%</td></tr><tr><td>Warmup / stable / decay</td></tr><tr><td>Final LR</td></tr><tr><td>Gradient clip norm</td></tr><tr><td rowspan="3">Training</td><td>Total steps</td></tr><tr><td>Batch size 128</td></tr><tr><td>Rollout steps T U[18, 24]</td></tr><tr><td rowspan="2">Evaluation</td><td>Inference rollout steps</td></tr><tr><td>1500 Digit prediction argmax over 9 channels (no post-processing)</td></tr><tr><td rowspan="2">Sample pool</td><td>Pool size</td></tr><tr><td>512 Pool fraction 0.5</td></tr></table>

## A.3.2 Training Loss

Let $p _ { i , d } = \mathrm { s o f t m a x } ( \hat { s } _ { i } ) _ { d }$ be the predicted probability of digit $d \in \{ 1 , \ldots , 9 \}$ at cell i. A valid Sudoku completion requires each digit to appear exactly once in every row, column, and 3 × 3 box. Denoting the set of all 27 such units as U, we penalize squared deviations from the per-unit digit count of one:

$$
\mathcal { L } _ { \mathrm { c o n s t r a i n t } } = \frac { 1 } { | \mathcal { U } | } \sum _ { U \in \mathcal { U } } \sum _ { d = 1 } ^ { 9 } \left( \sum _ { i \in U } p _ { i , d } - 1 \right) ^ { 2 } .\tag{7}
$$

An additional cross-entropy term enforces consistency with the given clue cells C whose correct digit is $y _ { i } { \mathrm { : } }$

$$
\mathcal { L } _ { \mathrm { g i v e n } } = - \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } \log p _ { i , y _ { i } } .\tag{8}
$$

The total loss is $\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c o n s t r a i n t } } + \mathcal { L } _ { \mathrm { g i v e n } } . } \end{array}$

## A.3.3 Computational Cost

Table 6: Resource usage of the Sudoku NCA variants. Training VRAM is the peak CUDA memory allocated during a representative training step (batch size 128, rollout length 24). Inference VRAM and runtime are measured at batch size 1 with a rollout of 1500 steps on an NVIDIA RTX 4090. Inference time is averaged per puzzle over repeated runs after warm-up.
<table><tr><td>Variant</td><td>Params</td><td>Train VRAM (MB)</td><td>Infer VRAM (MB)</td><td>Infer ms / puzzle</td></tr><tr><td>NCA w. 3x3 Conv.</td><td>50k</td><td>303.1</td><td>0.5</td><td>159.64</td></tr><tr><td>NCA w. 5x5 Conv.</td><td>105k</td><td>311.9</td><td>0.9</td><td>166.38</td></tr><tr><td>NCA w. R/C/B Conv.</td><td>145k</td><td>610.3</td><td>1.0</td><td>251.04</td></tr><tr><td>NCA w. Dilated Perc.</td><td>150k</td><td>480.8</td><td>1.2</td><td>238.28</td></tr><tr><td>aNCA</td><td>165k</td><td>1139.6</td><td>17.7</td><td>832.75</td></tr></table>

## A.3.4 Additional Results: Sudoku

This section supplements the main array-Sudoku evaluation with further analysis of aNCA’s adaptive perception and reproducibility. We visualize the sampling-distance distributions of the deformable heads, quantify their alignment with Sudoku’s row, column, and box constraints, and report the variation across independent training seeds.

![](images/631e8305730980b4325b711933ebf5020fcedab49bc45d1a58d5a39dad1c6e58.jpg)  
(a) A

![](images/c01a40d5fe9e4839ea22f475734b15bf0d08eed1e2f90b3b62e90a49a17e3a4c.jpg)

(b) B  
![](images/e7bde3fb8c0b19563a3a1332dff361616153cf42e6a8b6556db9f8261783bb31.jpg)  
(c) C  
Figure 9: Learned sampling distance distributions of the three deformable perception heads (A, B, C) at the final NCA step, aggregated over held-out test puzzles. Each plot shows how far each kernel tap reaches from the source cell, revealing that the heads develop complementary routing strategies: some taps remain local while others extend to capture row, column, or box constraints.

Quantitative alignment scores. Table 7 reports, for each deformable head, the mean fraction of total sampling weight landing inside the source cell’s row, column, 3 × 3 box, and their union (Sudoku-relevant), averaged over all 81 source cells and 10 000 held-out test puzzles at the final rollout step. The random baseline is the expected fraction under uniform sampling. All three heads concentrate 81–87% of their weight on Sudoku-relevant cells, 3.1–3.4× above the 25.9% random baseline.

Table 7: Per-head constraint alignment at the final NCA step, averaged over all 81 source cells and 10 000 test puzzles. Union% is the fraction of weight on the source cell’s row ∪ column ∪ box. Random baseline is the expected Union% under uniform sampling (25.9%).
<table><tr><td>Head</td><td>Row%</td><td>Col%</td><td>Box%</td><td>Union%</td><td>Enrichment</td></tr><tr><td>Head 1</td><td>20.7</td><td>32.6</td><td>47.5</td><td>81.0</td><td>3.13×</td></tr><tr><td>Head 2</td><td>61.1</td><td>59.6</td><td>69.6</td><td>86.9</td><td>3.35×</td></tr><tr><td>Head 3</td><td>53.0</td><td>45.8</td><td>54.3</td><td>83.5</td><td>3.22×</td></tr><tr><td>Random baseline</td><td>一</td><td>一</td><td>一</td><td>25.9</td><td>1.00×</td></tr></table>

## A.4 MNIST Sudoku

The NCA variants evaluated on MNIST Sudoku are the exact checkpoints trained for the array task (section A.3); no additional NCA training is performed. The pixel-space pipeline wraps these frozen NCA checkpoints with a separately trained encoder–decoder codec, described below.

## A.4.1 Codec Hyperparameters

Table 8: Architecture and training hyperparameters for the Sudoku codec. Encoder and decoder are trained independently with AdamW.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="2">Encoder  $E _ { \psi }$ </td><td>Base channels</td><td>32</td></tr><tr><td>Hidden dim</td><td>128</td></tr><tr><td rowspan="3">Decoder  $D _ { \omega }$ </td><td>Base channels</td><td>64</td></tr><tr><td>Hidden dim</td><td>128</td></tr><tr><td>Style latent dim</td><td>8</td></tr><tr><td rowspan="3">Optimization</td><td>Optimizer</td><td>AdamW  $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Learning rate LR schedule</td><td>Warmup-cosine (1000 steps)</td></tr><tr><td>Total steps</td><td>100 000</td></tr><tr><td>Training</td><td>Batch size</td><td>32</td></tr></table>

## A.4.2 Codec Training Loss

The decoder $D _ { \omega }$ is trained in isolation with a composite loss. Let $\hat { x } = D _ { \omega } ( b , z )$ be the rendered image for board b with style latent $z \sim q _ { \omega } ( z \mid b ) , x ^ { * }$ the target pixel image, and $M _ { \mathrm { p i x } }$ a pixel-space binary mask over occupied (non-blank) cells. The total loss is

$$
\mathcal { L } _ { \mathrm { d e c } } = w _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } + w _ { \mathrm { p i x } } \mathcal { L } _ { \mathrm { p i x } } + w _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } + w _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + w _ { \mathrm { T V } } \mathcal { L } _ { \mathrm { T V } } + w _ { \mathrm { i n k } } \mathcal { L } _ { \mathrm { i n k } } + w _ { \mathrm { b i n } } \mathcal { L } _ { \mathrm { b i n k } }\tag{9}
$$

with weights (w<sub>KL</sub>, w<sub>pix</sub>, w<sub>feat</sub>, w<sub>cls</sub>, w<sub>TV</sub>, w<sub>ink</sub>, w<sub>bin</sub>) = (0.001, 1.0, 0.5, 1.0, 0.01, 0.0025, 0.01).

KL divergence. $\mathcal { L } _ { \mathrm { K L } } = D _ { \mathrm { K L } } ( q _ { \omega } ( z \mid b ) \parallel \mathcal { N } ( 0 , I ) )$ , the standard VAE term regularising the style posterior.

Pixel L1. $\begin{array} { r } { \mathcal { L } _ { \mathrm { p i x } } = \sum _ { i } \left| \hat { x } _ { i } - x _ { i } ^ { * } \right| \cdot M _ { \mathrm { p i x } , i } / \sum _ { i } M _ { \mathrm { p i x } , i } . } \end{array}$ , a masked per-pixel reconstruction loss.

Feature L1. Let $\phi ( \cdot )$ denote the pooled intermediate features of the frozen digit classifier $E _ { \psi }$ , and let ${ \hat { c } } , c ^ { * }$ be the rendered and target cell crops. $\mathcal { L } _ { \mathrm { f e a t } } = \Vert \phi ( \hat { c } ) - \phi ( c ^ { * } ) \Vert _ { 1 }$ is a perceptual term encouraging feature-level similarity to the target digit appearance.

Classifier CE. $\mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } ( E _ { \psi } ( \hat { c } ) , d ^ { * } )$ , cross-entropy under the frozen classifier, directly rewarding legible digit rendering.

Total variation. Let ink $= ( 1 - \hat { x } ) \cdot M _ { \mathrm { p i x } }$ . Then $\mathcal { L } _ { \mathrm { T V } } = \mathrm { m e a n } ( | \Delta _ { h } \mathrm { i n k } | + | \Delta _ { w } \mathrm { i n k } | )$ encourages smooth, connected strokes.

Ink sparsity. $\begin{array} { r } { \mathcal { L } _ { \mathrm { i n k } } = \sum _ { i } ( 1 - \hat { x } _ { i } ) M _ { \mathrm { p i x } , i } / \sum _ { i } M _ { \mathrm { p i x } , i } } \end{array}$ penalises dark pixels outside digit regions.

Binary commitment. $\begin{array} { r } { \mathcal { L } _ { \mathrm { b i n } } = \sum _ { i } \hat { x } _ { i } \big ( 1 - \hat { x } _ { i } \big ) M _ { \mathrm { p i x } , i } \big / \sum _ { i } M _ { \mathrm { p i x } , } } \end{array}$ <sub>i</sub> penalises intermediate grey values, pushing pixels toward binary black/white.

The encoder $E _ { \psi }$ is trained separately with a masked cross-entropy loss on the given (clue) cells C:

$$
\mathcal { L } _ { \mathrm { e n c } } = - \frac { 1 } { \vert \mathcal { C } \vert } \sum _ { i \in \mathcal { C } } \log p _ { i , y _ { i } } ,\tag{10}
$$

where $p _ { i , d }$ is the predicted probability of digit d at cell i and $y _ { i }$ is the ground-truth digit.

## A.4.3 Codec Accuracy

We evaluate the encoder $E _ { \psi }$ and decoder $D _ { \omega }$ in isolation on 10 000 test puzzles across the full difficulty range to quantify how much of the pipeline error originates from the codec versus the NCA.

Encoder. Running $E _ { \psi }$ on the input pixel Sudoku and taking the argmax over digit logits yields 99.38% per-cell accuracy on the given (non-blank) cells. The perfect-board rate (the fraction of boards where every given cell is correctly recognized) is 74.58%, meaning roughly one in four boards contains at least one misrecognised clue cell. Despite these imperfect inputs, aNCA maintains strong downstream solving performance (table 9) without additional training, highlighting its ability to operate within noisy encoder conditioning.

Decoder round-trip. We pass the ground-truth one-hot board through $D _ { \omega } ,$ decode the resulting pixel image with $E _ { \psi } ,$ and compare the re-encoded digits to the ground truth. Per-cell accuracy is 100% and the perfect round-trip rate is 100%: the decoder renders digits reliably enough that the encoder always recovers the correct symbol. Decoder errors therefore contribute zero to the pipeline failure rate. ASR is evaluated against ground-truth clue digits, so encoder misrecognitions affect NCA conditioning but not the validity metric.

## A.4.4 Additional Results: MNIST Sudoku

This section supplements the main MNIST Sudoku evaluation with complete per-bucket results. In addition to ASR, we report ACP for every NCA perception variant on the Easy, Medium, and Hard splits, using the same frozen codec and array-trained NCA checkpoints as in the main evaluation.

Table 9: Sudoku Images
<table><tr><td rowspan="2">Method</td><td rowspan="2">Source</td><td colspan="2">Easy [1, 27]</td><td colspan="2">Medium [28, 54]</td><td colspan="2">Hard [55, 81]</td><td rowspan="2">Param.</td></tr><tr><td>ASR</td><td>ACP</td><td>ASR</td><td>ACP</td><td>ASR</td><td>ACP</td></tr><tr><td colspan="8">Related Work</td></tr><tr><td>Diffusion Model</td><td>[7]</td><td>99.4%</td><td>*</td><td>53.6%</td><td>*</td><td>0.8%</td><td></td><td>118M</td></tr><tr><td>SRM</td><td>[7]</td><td>99.8%</td><td></td><td>75.4%</td><td>*</td><td>51.6%</td><td>*</td><td>118M</td></tr><tr><td colspan="8">NCAs</td></tr><tr><td>NCA w. 3x3 Conv.</td><td>Ours</td><td> $9 9 . 4 0 \pm 0 . 1 4 9 9 . 9 0 \pm 0 . 0 0$ </td><td></td><td> $8 0 . 9 0 \pm 0 . 9 0$ </td><td> $9 8 . 3 3 \pm 0 . 1 0 | $ </td><td> $7 1 . 9 8 \pm 1 . 6 1$ </td><td> $9 8 . 2 0 \pm 0 . 0 8$ </td><td> $1 . 1 \mathrm { M } + 5 0 \mathrm { k }$ </td></tr><tr><td>NCA w. 5x5 Conv.</td><td>Ours</td><td> $9 9 . 1 3 \pm 0 . 5 0 9 9 . 8 7 \pm 0 . 0 6$ </td><td></td><td> $7 9 . 9 7 \pm 1 . 7 0$ </td><td> $9 8 . 1 7 \pm 0 . 1 5$ </td><td> $6 9 . 0 0 \pm 6 . 6 3$ </td><td> $9 5 . 7 7 \pm 4 . 1 3$ </td><td> $1 . 1 \mathrm { M } + 1 0 5 \mathrm { k }$ </td></tr><tr><td>NCA w. 7x7 Conv.</td><td>Ours</td><td> $9 8 . 7 3 \pm 0 . 1 0 9 9 . 8 3 \pm 0 . 0 5$ </td><td></td><td> $7 6 . 5 5 \pm 1 . 2 1$ </td><td> $9 7 . 7 3 \pm 0 . 1 7$ </td><td> $7 2 . 3 0 \pm 2 . 6 0$ </td><td> $9 8 . 1 5 \pm 0 . 1 9$ </td><td> $1 . 1 \mathrm { M } + 1 8 9 \mathrm { k }$ </td></tr><tr><td>NCA w. 9x9 Conv.</td><td>Ours</td><td> $9 9 . 3 0 \pm 0 . 2 3 9 9 . 9 3 \pm 0 . 0 5$ </td><td></td><td> $6 6 . 7 3 \pm 1 6 . 9 9$ </td><td> $9 5 . 1 8 \pm 4 . 4 2$ </td><td> $3 5 . 8 3 \pm 3 3 . 5 6$ </td><td> $8 4 . 2 5 \pm 2 0 . 1 5$ </td><td> $1 . 1 \mathrm { M } + 3 0 0 \mathrm { k }$ </td></tr><tr><td>NCA w. R/C/B Conv.</td><td>Ours</td><td> $9 9 . 2 0 \pm 0 . 0 8 9 9 . 9 0 \pm 0 . 0 0$ </td><td></td><td> $8 0 . 3 8 \pm 1 . 4 1$ </td><td> $9 8 . 1 3 \pm 0 . 1 7$ </td><td> $7 1 . 2 5 \pm 2 . 6 1$ </td><td> $9 8 . 1 3 \pm 0 . 1 7$ </td><td> $1 . 1 \mathrm { M } + 1 4 5 \mathrm { k }$ </td></tr><tr><td>NCA w. Dilated Perc.</td><td>Ours</td><td> $9 9 . 2 5 \pm 0 . 1 7 9 9 . 8 8 \pm 0 . 0 5$ </td><td></td><td> $8 0 . 7 8 \pm 0 . 9 5$ </td><td> $9 8 . 1 3 \pm 0 . 0 5$ </td><td> $7 0 . 9 5 \pm 8 . 3 9$ </td><td> $9 8 . 0 5 \pm 0 . 5 7$ </td><td> $1 . 1 \mathrm { M } + 1 5 0 \mathrm { k }$ </td></tr><tr><td>aNCA</td><td>Ours</td><td> $9 9 . 0 0 \pm 0 . 3 7 $ </td><td>99.88 ± 0.05</td><td> ${ \bf 8 1 . 9 0 \pm 5 . 5 0 }$ </td><td>98.40 ± 0.50</td><td> ${ \bf 8 2 . 1 0 \pm 4 . 8 9 }$ </td><td>98.85 ± 0.40</td><td> $1 . 1 \mathrm { M } + 1 6 5 \mathrm { k }$ </td></tr></table>

⋆ indicates metrics not reported in prior work.

## A.5 Maze

## A.5.1 Model and Training Hyperparameters

Table 10: Hyperparameters shared across all NCA variants for the maze task. Perception-specific settings vary per variant and are described in section 2.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="3">NCA state</td><td>Cell channels</td><td>24</td></tr><tr><td>Positional embeddings</td><td>no 0.5</td></tr><tr><td>Fire rate</td><td></td></tr><tr><td>Update network</td><td>Architecture</td><td>MLP (1 ×1 Conv), 1 hidden layer, 128 channels</td></tr><tr><td rowspan="5">Optimization</td><td>Optimizer</td><td> $\mathrm { A d a m W }$   $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Learning rate</td><td></td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR schedule</td><td>WSD (warmup-stable-decay)</td></tr><tr><td>Warmup / stable / decay Gradient clip norm</td><td>10% / 80% / 10% 1.0</td></tr><tr><td rowspan="3">Training</td><td>Total steps</td><td>100 000</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Rollout steps T</td><td>U[18, 24]</td></tr><tr><td rowspan="2">Evaluation</td><td></td><td>100</td></tr><tr><td>Inference rollout steps Path prediction</td><td>threshold at 0.5 (no post-processing)</td></tr><tr><td rowspan="2">Sample pool</td><td></td><td>512</td></tr><tr><td>Pool size Pool fraction</td><td>0.5</td></tr></table>

## A.5.2 Training Loss

Let $\hat { y } _ { i } \in [ 0 , 1 ]$ be the predicted path probability at cell $i , y _ { i } \in \{ 0 , 1 \}$ the ground-truth binary path mask, and F the set of non-wall cells. Walls are excluded since they are trivially predictable from the condition input. The loss combines three terms. A BCE term encourages binary commitment to path vs. non-path:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { | \mathcal { F } | } \sum _ { i \in \mathcal { F } } \bigl [ y _ { i } \log \hat { y } _ { i } + ( 1 - y _ { i } ) \log ( 1 - \hat { y } _ { i } ) \bigr ] .\tag{11}
$$

A Dice term directly optimizes path-prediction overlap, handling the imbalance between path and non-path cells:

$$
\mathcal { L } _ { \mathrm { { D i c e } } } = 1 - \frac { 2 \displaystyle \sum _ { i \in \mathcal { F } } \hat { y } _ { i } y _ { i } + \epsilon } { \displaystyle \sum _ { i \in \mathcal { F } } \hat { y } _ { i } + \sum _ { i \in \mathcal { F } } y _ { i } + \epsilon } .\tag{12}
$$

A length penalty penalizes paths that are longer or shorter than the shortest-path target, normalized by the number of free cells:

$$
\mathcal { L } _ { \mathrm { l e n } } = \left| \frac { \sum _ { i \in \mathcal { F } } \hat { y } _ { i } } { \left| \mathcal { F } \right| } - \frac { \sum _ { i \in \mathcal { F } } y _ { i } } { \left| \mathcal { F } \right| } \right| .\tag{13}
$$

The total loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } } + w _ { \mathrm { l e n } } \mathcal { L } _ { \mathrm { l e n } }$ with $w _ { \mathrm { l e n } } = 0 . 0 5$

## A.5.3 Additional Results: Mazes

This section supplements the main maze evaluation with qualitative rollout trajectories and a quantitative same-resolution distribution-shift experiment. We evaluate the trained 30 × 30 aNCA checkpoint, without finetuning, on 1000 unseen Prim-generated mazes and compare its Valid Path and Valid Optimal scores with those on the original Sapient test distribution.

Generalization beyond the training maze distribution. To evaluate whether the trained aNCA learns a reusable path-finding procedure, we run the same checkpoint, without finetuning, on 1000 procedurally generated $3 0 \times 3 0$ mazes using randomized Prim-style maze generation [47]. Although these mazes differ from the training distribution in wall density and corridor geometry, the NCA propagates a connected path from start to goal and adapts its rollout to the new layout. It achieves 100% Valid Path and 99.4% Valid Optimal, showing robust transfer to a new maze distribution at the same grid size. Qualitative examples and the full comparison are shown in fig. 8b.

Table 11: Generalization of the deformable aNCA maze solver to a different maze distribution at the training resolution. The model is trained on $3 0 \times 3 0$ Sapient mazes and evaluated without finetuning on 1000 mazes per test distribution. Valid Path checks connectivity and wall avoidance. Valid Optimal additionally requires shortest-path length.
<table><tr><td>Test Maze Size</td><td>Generator</td><td>Valid Path</td><td>Valid Optimal</td></tr><tr><td> $3 0 \times 3 0$ </td><td>Sapient</td><td>98.56%</td><td>94.18%</td></tr><tr><td> $2 0 \times 2 0$ </td><td>Prim</td><td>100%</td><td>100%</td></tr><tr><td> $3 0 \times 3 0$ </td><td>Prim</td><td>100%</td><td>99.4%</td></tr></table>

## A.6 Global Color Balance

## A.6.1 Model and Training Hyperparameters

Table 12: Hyperparameters for the global color balance task.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>NCA state</td><td>Cell channels Positional embeddings Fire rate</td><td>24 no 0.5</td></tr><tr><td>Update network</td><td>Architecture</td><td>MLP (1 ×1 Conv), 1 hidden layer, 128 channels</td></tr><tr><td>Optimization</td><td>Optimizer Learning rate LR schedule Warmup / stable / decay Gradient clip norm</td><td>AdamW  $3 \times 1 0 ^ { - 4 }$  WSD (warmup-stable-decay) 10% / 80% / 10% 1.0</td></tr><tr><td>Training</td><td>Total steps Batch size Rollout steps  $T$ </td><td>25000 16 U[12, 24]</td></tr><tr><td>Evaluation</td><td>Inference rollout steps Test samples per run Independent training runs</td><td>200 1000 3</td></tr><tr><td>Sample pool</td><td>Pool size Pool fraction</td><td>512 0.5</td></tr><tr><td>Loss weights</td><td> $w _ { \mathrm { c o u n t } }$   $w _ { \mathrm { s a t } }$ </td><td>1.0 0.01</td></tr></table>

## A.6.2 Training Loss

Let $x _ { i } \in [ - 1 , 1 ]$ be the predicted pixel value of cell i (where −1 = black, +1 = white), N the total number of pixels, and $\mu ^ { * } = 2 r - 1$ the target mean corresponding to a prescribed white fraction r. The loss combines two terms. A count term pushes the global pixel mean toward the target ratio:

$$
\mathcal { L } _ { \mathrm { c o u n t } } = \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } x _ { i } - \mu ^ { * } \right) ^ { 2 } .\tag{14}
$$

A saturation term pushes each pixel to commit to either −1 or +1, zeroing only at fully binary values:

$$
\mathcal { L } _ { \mathrm { s a t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( 1 - x _ { i } ^ { 2 } \right) .\tag{15}
$$

The total loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { c o u n t } } + \mathcal { L } _ { \mathrm { s a t } }$

## A.6.3 Additional Results: Global Color Balance

Seed  
Step 10  
Step 20  
Prediction  
![](images/eef5cdf14ede5638b51e841bf7706cad0ae3a29ec0b9c2568145c88101f574c5.jpg)  
Figure 10: Global color balance: the model must satisfy a prescribed black/white ratio without per-pixel supervision.