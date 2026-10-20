# Context-Aware Topological Distillation Graph Network for Asymmetric EEG–Eye Movement Emotion Recognition

Peijun Shi<sup>∗</sup>, Xiujin Zhu<sup>∗</sup>, Zhenguo Lao, Chee-Onn Chow, Senior Member, IEEE, Wei Ru Wong, Zhen Liu, and Xiaobing Jiang

Abstract—EEG–eye movement emotion recognition can support mental-health monitoring and adaptive interfaces, but most fusion models treat the two signals as parallel features, classify each window alone and degrade when eye data are lost. We propose the Context-Aware Topological Distillation Graph Network (CTD-GNet), in which EEG features form the nodes of an electrode graph, and eye movements and preceding windows modulate its edges, nodes and readout, the edges through electrode-specific prompts. Bidirectional distillation of the graph, node relations and predictions couples this multimodal teacher with a single-window EEG student, improving the teacher and aligning the two graphs 35-fold. In leave-one-subject-out tests on SEED and SEED-IV against 17 baselines, CTD-GNet has the highest mean accuracy, 84.91% and 76.17%, significantly above 9 and 6 of 13 eye-based baselines after Holm correction, the smallest cross-subject spread among multimodal models and the highest worst-subject accuracy. It is the most accurate model at every rate of missing eye data on SEED-IV, keeps 66.1% on SEED without eye data, more than any EEG-only model, and stays 9.3 and 11.9 points above all single-window fusion models under eye-movement noise of one standard deviation. On film clips unseen in training, it leads on SEED with 77.1%.

Index Terms—Electroencephalography, emotion recognition, eye movements, graph neural networks, knowledge distillation, missing modalities, multimodal fusion.

## 1 INTRODUCTION

toring, adaptive interfaces and driver monitoring. Electroencephalography (EEG) measures cortical activity, and eye tracking measures fixations, saccades, blinks and pupil size, which reflect visual attention and arousal [2]. On SEED [3] and SEED-IV [4], combining the two signals outperforms either alone [4]–[6]. Three problems keep this gain from reaching practice.

First, fusion models treat the two signals alike. Concatenation, bimodal autoencoders [7], canonical correlation [6], tensor and low-rank fusion [8], [9] and cross-modal attention [10] encode each signal as a vector and merge the vectors. Yet eye movements indicate where attention is directed, and attention changes which cortical regions exchange information [11]: eye movements describe how regions interact, whereas EEG describes what each region does. Merging the two as parallel vectors discards this difference.

Second, emotions induced by a film clip unfold over tens of seconds [12], but most models classify each window

![](images/c6e311280fd6e32c2459687a5c57151f4d804216e1b443000c8b716d8018efbf.jpg)  
Fig. 1. Two ways to combine EEG and eye movements. (a) Symmetric fusion encodes both signals as feature vectors and merges them by concatenation or cross-attention; missing eye data change the fused feature, and each window is classified independently. (b) CTD-GNet: EEG gives the nodes of an electrode graph, and eye movements (green) and the preceding windows (purple) modulate its edges, nodes and readout. Bidirectional topology distillation couples the graph with that of an EEG-only student, which serves devices without an eye tracker.

alone, smooth the features over the trial [3], [13] or feed the window sequence to a recurrent classifier [14]. In all three cases, the preceding windows do not change how the regions of the current window are related.

Third, eye trackers lose data during blinks, head movements and calibration drift. Fusion models use both signals in every prediction, and many collapse without the eye data: in our experiments, cross-attention fusion falls to 24.0% on SEED-IV when the eye data are removed, below chance.

Missing-modality training [15] and reconstruction [16] let a fused model tolerate absent inputs, but the eye data still enter the fused representation directly, and multi-window models need a buffer of past windows on the device.

The three problems share one cause: eye movements and temporal context are treated as additional features rather than as signals that shape cortical interaction. We therefore build an electrode graph from EEG and let the eye movements and the context modulate its edges, nodes and readout. This design addresses the three problems together. The two signals take different roles; the preceding windows change the graph of the current window; and because the EEG evidence enters every graph computation, missing or corrupted eye data weaken the prediction instead of corrupting it. The effect of eye movements and context on connectivity is concentrated in the adjacency matrix, which a model that sees only EEG can learn to approximate.

We propose the Context-Aware Topological Distillation Graph Network (CTD-GNet), shown in Fig. 1. Its teacher computes the edges by an attention over the 62 electrodes, with a recurrent context of the preceding windows in the queries and the eye movements in the keys; a gated node pathway and a gated multimodal unit [17] add the eye movements to the nodes and the readout. Because eye movements and past windows are available during training but not always at deployment, we treat them as privileged information [18], [19] and train an EEG-only student that sees only the current window. The two networks are trained jointly by bidirectional distillation of the graph, the node relations and the predictions. The main contributions of this work are as follows.

We propose an asymmetric fusion paradigm in which EEG forms the nodes of an electrode graph and eye movements and temporal context modulate its edges, nodes and readout, so that the EEG evidence enters every computation; the resulting model has the highest worst-subject accuracy of all models, 74.02% on SEED.

We introduce bidirectional distillation between the multimodal teacher and a single-window EEG student at the levels of the graph, the node relations and the predictions, which aligns their graphs 35-fold, so that both networks can be read on one connectivity, and also improves the teacher.

We design and evaluate the framework for unreliable sensors, with missing and noisy eye data and film clips unseen in training; with 80% of the eye data missing on SEED-IV, the teacher is 4.1 points more accurate than the strongest baseline.

The code will be released at https://github.com/ Peijun1122/CTD-GNet. The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents CTD-GNet. Section 4 describes the experimental setup and results. Section 5 discusses the findings, limitations and future work, and Section 6 concludes the paper.

## 2 RELATED WORK

## 2.1 EEG-based emotion recognition

Differential entropy (DE) features [20] are standard on SEED and SEED-IV, and Zheng and Lu [3] identified the $\beta$ and γ bands and the lateral temporal electrodes as the most discriminative. Convolutional models [21] learn from raw signals or feature maps, and hemispheric models [22], [23] use the difference between the hemispheres. Cross-subject models align the features of different subjects, for example by adversarial training [22] or with class prototypes [24]. Graph models treat electrodes as nodes: DGCNN [25] learns one adjacency matrix, RGNN [26] initializes it from electrode distances and hemispheric pairs, and instance-adaptive and self-organized graphs [27], [28] compute it from the EEG of each sample, as graph attention [29] computes edge weights from node features. In all of them, the graph depends on EEG only.

## 2.2 Multimodal fusion and temporal context

EEG and eye movements have been fused at feature and decision level [5], by a bimodal deep autoencoder (BDAE) [7], a bimodal LSTM [14], and deep canonical correlation analysis (DCCA) [30] with its attention variant DCCA-AM [6]. Recent models let one modality guide the encoding of the other [31], align the two modalities across subjects [32]– [34] or adapt the fused features to the unlabeled data of a new subject [35]. General multimodal models use tensor products [8], low-rank factors [9], invariant and specific subspaces [36], gated units [17] or cross-modal attention [10]. Most of these models fuse or align feature vectors of the two signals, so the eye movements do not shape the relations between electrodes, and a corrupted eye input alters the fused representation directly. Temporal context enters through smoothing of the features [3], [13] or recurrent [37] and temporal convolutional [38] networks over the window sequence; it changes the features or the output, but not the graph.

## 2.3 Distillation, privileged information and missing modalities

Distillation transfers outputs [39], features [40], sample relations [41], [42] or the local structure of graph embeddings [43], and deep mutual learning [44] distills between peers in both directions; across modalities, a visual teacher has been distilled into an EEG student for continuous emotion recognition [45]. Learning using privileged information [18] and generalized distillation [19] exploit data available only in training. Missing-modality methods drop modalities during training [15], reconstruct them [16], [46], model their shared and specific features [47] or learn a prompt for each pattern of missing inputs [48], so that one fused model tolerates absent inputs; when the eye data are present, they still enter the fused features directly. Prompts [49] and FiLM [50] condition networks with learned or input-dependent vectors, an idea that CTD-GNet applies to the edges of a graph. None of these methods transfers the graph through which a privileged modality acts, or couples the teacher to that graph in the opposite direction.

## 3 METHOD

Fig. 2 shows the architecture. The teacher uses EEG, eye movements and context, the student uses the current EEG

![](images/cfea7632986e782a42dbe36393c65fdd030674a221c760524290fde89c864e61.jpg)  
Fig. 2. Architecture of CTD-GNet. Teacher (blue): a GRU context bank encodes the K preceding windows into $\mathbf { c } _ { t } ;$ a gated node pathway adds the eye movements to the EEG node embeddings; the generator adds the context prompt $\mathbf { I } ^ { c }$ to the queries, the eye prompt Π<sup>e</sup> to the keys and log $\dot { { \mathbf A } ^ { \mathrm { p h y } } }$ to the scores, and a row-wise softmax gives A ; a GCN propagates over $\mathbf { A } _ { T } ,$ and a gated multimodal unit fuses the graph readout with the eye movements and $\mathbf { c } _ { t } .$ Student (orange): the same operations with prompts from the current EEG window. Red dashed arrows: distillation losses, with gradients to both networks. Right: deployment and training objective.

window, and both use the same physical prior. Section 3.1 defines the task, Section 3.2 describes the teacher, Section 3.3 the student, and Section 3.4 the distillation and deployment.

## 3.1 Problem formulation

Each trial is divided into non-overlapping windows indexed by t. Window t provides the DE features ${ \bf X } _ { q } ^ { t } \in \mathbb { R } ^ { N \times d _ { g } }$ of $\dot { N } \ : = \ : 6 2$ electrodes in $d _ { g } \ = \ 5$ frequency bands, the eyemovement features $\mathbf { x } _ { e } ^ { t } \in \mathbb { R } ^ { d _ { e } }$ , and the label $y ^ { t } \in \{ 1 , \ldots , \dot { C } \}$ of the trial. The history $\mathbf { \mathcal { H } } _ { t } = \{ ( \mathbf { X } _ { q } ^ { t ^ { \prime } } , \mathbf { x } _ { e } ^ { t ^ { \prime } } ) \} _ { t ^ { \prime } = t - K } ^ { t - 1 }$ contains the $K$ preceding windows of the same trial. The teacher uses the eye cue $\begin{array} { r } { \bar { \mathbf { x } } _ { e } ^ { t } = \frac { 1 } { K + 1 } \sum _ { t ^ { \prime } = t - K } ^ { t } \mathbf { x } _ { e } ^ { t ^ { \prime } } . } \end{array}$ , which averages tracker noise. The teacher $f _ { T }$ maps $( \mathbf { X } _ { q } ^ { t } , \mathbf { x } _ { e } ^ { t } , \mathcal { H } _ { t } )$ to logits $\mathbf { P } _ { T } \in \mathbb { R } ^ { C }$ , and the student $f _ { S }$ maps $\mathbf { X } _ { g } ^ { t }$ to $\mathbf { P } _ { S } \in \mathbb { R } ^ { C }$ ; the eye movements and the history are privileged information [18] available only to the teacher. Vectors are row vectors, $\mathbf { M } _ { i }$ is the i-th row of M, softma $\mathrm { X } _ { \mathrm { r o w } }$ is the row-wise softmax, and sg(·) is the stop-gradient operator.

## 3.2 Context-aware teacher

The teacher consists of a temporal context bank, a dualprompt graph generator and a graph backbone with a gated readout. The temporal context bank (TCB) summarizes the preceding windows with a gated recurrent unit (GRU) [37]. Each history window $t ^ { \prime }$ is mapped to a token

$$
\begin{array} { r l } & { { \mathbf { r } ^ { t ^ { \prime } } } = [ { \mathbf { x } _ { e } ^ { t ^ { \prime } } } , \ \phi _ { g } ( { \mathbf { X } _ { g } ^ { t ^ { \prime } } } ) ] \in \mathbb { R } ^ { d _ { e } + d _ { s } } , } \\ & { \quad \phi _ { g } ( { \mathbf { X } } ) = \mathrm { v e c } ( { \mathbf { X } } ) \mathbf { W } _ { g } + \mathbf { b } _ { g } , } \end{array}\tag{1}
$$

where $\phi _ { g }$ linearly projects the 310-dimensional DE vector to $d _ { s }$ dimensions. The GRU reads the sequence $\mathbf { r } ^ { t - K } , \ldots , \mathbf { r } ^ { t - 1 }$ in temporal order,

$$
\begin{array} { r } { \pmb { \xi } _ { k } = \mathrm { G R U } \big ( \mathbf { r } ^ { t - K - 1 + k } , \pmb { \xi } _ { k - 1 } \big ) , \qquad k = 1 , \ldots , K , } \\ { \mathbf { c } _ { t } = \mathrm { L N } ( \pmb { \xi } _ { K } ) \in \mathbb { R } ^ { d _ { c } } , \qquad } \end{array}\tag{2}
$$

with ${ \bf \xi } _ { { \bf \xi } _ { 0 } } = { \bf \xi } { \bf 0 }$ and layer normalization LN. The GRU gates decide how much of the accumulated state each window keeps, so $\mathbf { c } _ { t }$ follows the slow course of the induced emotion. The validation protocol selects between the GRU and a temporal convolutional network [38]. Windows at the start of a trial are padded with the first window, and histories do not cross trials.

The dual-prompt graph generator computes one adjacency matrix per window from node embeddings, two prompts and a physical prior. Each electrode i is a node with the EEG embedding

$$
\mathbf { e } _ { i } = \mathbf { x } _ { g , i } ^ { t } \mathbf { W } _ { \mathrm { i n } } + \mathbf { p } _ { i } \in \mathbb { R } ^ { d _ { h } } , \qquad i = 1 , \dots , N ,\tag{3}
$$

where $\mathbf { x } _ { g , i } ^ { t }$ is the i-th row of $\mathbf { X } _ { q } ^ { t }$ and $\mathbf { p } _ { i }$ is a learnable electrode embedding; $\mathbf { E } \in \mathbb { R } ^ { N \times d _ { h } ^ { \ast } }$ stacks the embeddings. The eye movements reach the nodes through a gate that each electrode sets for itself:

$$
\begin{array} { r } { \mathbf { v } _ { t } = \mathrm { L N } ( \bar { \mathbf { x } } _ { e } ^ { t } \mathbf { W } _ { n } ) , \qquad } \\ { \mathbf { g } _ { i } = \sigma \big ( \mathrm { L N } ( \mathbf { e } _ { i } ) \mathbf { W } _ { g e } + \mathbf { v } _ { t } \mathbf { W } _ { g v } + \mathbf { b } _ { n } \big ) , } \\ { \tilde { \mathbf { e } } _ { i } = \mathbf { e } _ { i } + \mathbf { g } _ { i } \odot \mathbf { u } _ { i } ^ { n } \odot \mathbf { v } _ { t } , \qquad } \end{array}\tag{4}
$$

where σ is the logistic function and $\mathbf { u } _ { i } ^ { n } \in \mathbb { R } ^ { d _ { h } }$ is an electrode embedding. Each electrode admits or blocks the eye input according to its own EEG state, and the EEG term $\mathbf { e } _ { i }$ is always kept; E<sup>˜</sup> stacks the gated embeddings.

The edge weights come from an attention over the electrodes. The context modulates the queries, which determine what each electrode attends to, and the eye movements modulate the keys, which determine which electrodes act as sources:

$$
\boldsymbol { \pi } _ { i } ^ { c } = \mathbf { u } _ { i } ^ { c } \odot ( \mathbf { c } _ { t } \mathbf { W } _ { c } ) , \qquad \boldsymbol { \pi } _ { j } ^ { e } = \mathbf { u } _ { j } ^ { e } \odot ( \bar { \mathbf { x } } _ { e } ^ { t } \mathbf { W } _ { e } ) ,\tag{5}
$$

$$
\mathbf { q } _ { i } = \tilde { \mathbf { e } } _ { i } \mathbf { W } _ { q } + \boldsymbol { \pi } _ { i } ^ { c } , \qquad \mathbf { k } _ { j } = \tilde { \mathbf { e } } _ { j } \mathbf { W } _ { k } + \boldsymbol { \pi } _ { j } ^ { e } ,\tag{6}
$$

$$
S _ { i j } = \mathbf { q } _ { i } \mathbf { k } _ { j } ^ { \top } / \sqrt { d _ { k } } ,\tag{7}
$$

where $\mathbf { W } _ { c } \in \mathbb { R } ^ { d _ { c } \times d _ { k } }$ and $\mathbf { W } _ { e } \in \mathbb { R } ^ { d _ { e } \times d _ { k } }$ are projections and ${ \bf u } _ { i } ^ { c } , { \bf u } _ { i } ^ { e } \in \mathbb { R } ^ { d _ { k } }$ are electrode prompt embeddings. The product with u gives each electrode its own version of a global cue, as FiLM [50] does for feature channels, and the matrices Π<sup>c</sup> and Π<sup>e</sup> stack the prompts.

The prompts are electrode-specific by design. A simpler design adds one global embedding of each cue to all queries or keys; Proposition 1, which motivates our choice, shows that such a cue cannot change the graph electrode by electrode, and Fig. 3(a) illustrates the decomposition used in the proof.

Proposition 1 (Prompt cancellation). Consider an attention matrix ${ \bf A } \ =$ softmax<sub>row</sub>(S) with scores $S _ { i j } \ =$ $( \mathbf { a } _ { i } + \pmb { \gamma } ) ( \mathbf { b } _ { j } + \pmb { \eta } ) ^ { \top } / \sqrt { d _ { k } }$ , where ${ \bf a } _ { i }$ and $\mathbf { b } _ { j }$ depend on the electrode features and $\gamma , \eta \in \mathbb { R } ^ { d _ { k } }$ are prompts shared by all electrodes. Then (i) A does not depend on ${ \boldsymbol { \eta } } ,$ and (ii) A depends on $\gamma$ only through $\gamma \mathbf { b } _ { i } ^ { \top } / \sqrt { d _ { k } }$ , which adds the same vector to every row of S.

Proof: Expanding the product gives $\begin{array} { r l } { \sqrt { d _ { k } } S _ { i j } } & { { } = } \end{array}$ $\mathbf { a } _ { i } \mathbf { b } _ { j } ^ { \top } + \gamma \bar { \mathbf { b } } _ { j } ^ { \top } + \bar { ( \mathbf { a } _ { i } + \gamma ) \pmb { \eta } } ^ { \top }$ . The last term does not depend on j. The softmax of a row does not change when a constant is added to all its entries, so this term cancels, which proves (i). The remaining prompt term $\gamma \mathbf { b } _ { j } ^ { \intercal }$ does not depend on $i ,$ which proves (ii). □

A global eye embedding added to the keys thus has no effect on $\mathbf { A } ,$ and a global context embedding added to the queries shifts all rows by the same vector, so that every electrode re-weights its neighbors in the same way. Proposition 1 concerns additive cues, the usual way to condition attention with prompts [49]; electrode-specific prompts keep this additive form and let a cue act on each electrode separately. With Eq. (5), the logit difference between two neighbors j and $j ^ { \prime }$ of electrode i contains ${ \bf q } _ { i } \left( \left( { \bf u } _ { j } ^ { e } - { \bf u } _ { j ^ { \prime } } ^ { e } \right) \odot \bar { { \bf x } } _ { e } ^ { t } { \bf W } _ { e } \right) ^ { \mathrm { ~ \tiny ~ ! ~ } } / \sqrt { d _ { k } }$ , which depends on $\bar { \mathbf { x } } _ { e } ^ { t }$ whenever ${ \bf u } _ { j } ^ { e } \ \ne \ { \bf u } _ { j ^ { \prime } } ^ { e } ,$ as Fig. 3(b) illustrates; the term $\pi _ { i } ^ { c } \mathbf { k } _ { j } ^ { \top }$ likewise depends on the context and on both electrodes. Proposition 1 is a design motivation rather than a predictor of accuracy: because the eye movements reach the model through several redundant routes, removing one of them has a limited effect in the ablations of Section 4.3.

Finally, a physical prior constrains the graph. Coupling between electrodes decreases with distance, and emotion processing involves homologous regions of the two hemispheres [23], [51]. The prior

$$
A _ { i j } ^ { \mathrm { p h y } } = \left\{ \begin{array} { l l } { 1 , } & { i = j \mathrm { o r } ( i , j ) \in \mathcal { P } , } \\ { \exp \bigl ( - \| \rho _ { i } - \rho _ { j } \| _ { 2 } ^ { 2 } / ( 2 \sigma _ { \rho } ^ { 2 } ) \bigr ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{8}
$$

uses the 3-D electrode positions $\rho _ { i }$ of the extended 10–20 system, the set $\mathcal { P }$ of $2 7$ symmetric pairs, such as FP1–FP2, and the median pairwise distance $\sigma _ { \rho } .$ . Unlike RGNN [26], which uses a similar prior only for initialization, CTD-GNet applies it to every generated graph:

$$
\begin{array} { r } { { \bf Z } _ { T } = { \bf S } + \log { \bf A } ^ { \mathrm { p h y } } , \qquad { \bf A } _ { T } = \mathrm { s o f t m a x } _ { \mathrm { r o w } } ( { \bf Z } _ { T } ) . } \end{array}\tag{9}
$$

Since $A _ { T , i j } = \exp ( S _ { i j } ) A _ { i j } ^ { \mathrm { p h y } } / \sum _ { k } \exp ( S _ { i k } ) A _ { i k } ^ { \mathrm { p h y } }$ , the prior acts as a soft mask, as Fig. 3(c) shows: distant pairs are penalized in proportion to their squared distance, symmetric pairs are not, and each row of ${ \bf A } _ { T }$ sums to one. We call $\mathbf { Z } _ { T }$ the topology logits.

A backbone of L residual graph convolution layers [52] then propagates the node embeddings over the graph:

$$
\mathbf { H } ^ { ( l ) } = \mathrm { L N } \Big ( \mathbf { H } ^ { ( l - 1 ) } + \mathrm { D r o p } \big ( \mathrm { G E L U } ( \mathbf { A } \mathbf { H } ^ { ( l - 1 ) } \mathbf { W } ^ { ( l ) } ) \big ) \Big ) ,\tag{10}
$$

where all layers use the adjacency matrix of the current window. The teacher uses $\mathbf { H } ^ { ( 0 ) } = \check { \mathbf { E } } , \mathbf { A } = \mathbf { A } _ { T }$ and $L = L _ { T }$ giving H<sub>T</sub>; mean and max pooling over the nodes give the graph representation $\mathbf { z } _ { g }$

For the readout, a gated multimodal unit (GMU) [17] combines the graph with the eye encoding $\mathbf z _ { e } = \mathrm { M L P } ( \bar { \mathbf x } _ { e } ^ { t } )$ and the context $\mathbf { z } _ { c } = \mathbf { c } _ { t }$

$$
\begin{array} { r l r } & { } & { { \bf h } = \displaystyle \sum _ { m \in \{ g , e , c \} } \alpha _ { m } \odot \mathrm { t a n h } ( { \bf z } _ { m } { \bf W } _ { m } ) , } \\ & { } & { \ [ \alpha _ { g } ; \alpha _ { e } ; \alpha _ { c } ] = \mathrm { s o f t m a x } _ { m } \big ( [ { \bf z } _ { g } , { \bf z } _ { e } , { \bf z } _ { c } ] { \bf W } _ { \alpha } \big ) , } \end{array}\tag{11}
$$

where the softmax runs over the three inputs for each of the $d _ { r }$ dimensions, and a two-layer perceptron maps h to ${ \bf P } _ { T }$ Auxiliary linear heads on $\mathbf { z } _ { g }$ and $\mathbf { z } _ { e . }$ , trained with the loss $\mathcal { L } _ { \mathrm { a u x } }$ of Eq. (19), keep both pathways discriminative [53].

The eye movements thus act at three levels: they reweight the edges, with each row of ${ \bf A } _ { T }$ constrained by the physical prior; they add a gated term to each node; and they compete with the graph representation in the readout. In all three, the EEG evidence stays in place and the eye movements modulate ${ \mathrm { i t } } ,$ in analogy to the way attention modulates cortical interaction [11]. The adjacency matrix ${ \bf A } _ { T }$ collects their effect on connectivity and serves as the interface between teacher and student.

## 3.3 Self-prompted student

The student receives $\mathbf { X } _ { q } ^ { t }$ only. Its generator has the same form as the teacher’s, with prompts computed from EEG; the superscript S marks student quantities:

$$
\mathbf { e } _ { i } ^ { S } = \mathbf { x } _ { g , i } ^ { t } \mathbf { W } _ { \mathrm { i n } } ^ { S } + \mathbf { p } _ { i } ^ { S } , \qquad \mathbf { s } ^ { t } = \phi _ { s } ( \mathbf { E } ^ { S } ) ,\tag{12}
$$

$$
\boldsymbol { \pi } _ { i } ^ { c , S } = \mathbf { u } _ { i } ^ { c , S } \odot ( \mathbf { s } ^ { t } \mathbf { W } _ { c } ^ { S } ) , \qquad \boldsymbol { \pi } _ { j } ^ { e , S } = \mathbf { u } _ { j } ^ { e , S } \odot ( \mathbf { s } ^ { t } \mathbf { W } _ { e } ^ { S } ) ,\tag{13}
$$

$$
\mathbf { Q } ^ { S } = \mathbf { E } ^ { S } \mathbf { W } _ { q } ^ { S } + \mathbf { H } ^ { c , S } , \qquad \mathbf { K } ^ { S } = \mathbf { E } ^ { S } \mathbf { W } _ { k } ^ { S } + \mathbf { H } ^ { e , S } ,
$$

$$
\mathbf { S } ^ { S } = \mathbf { Q } ^ { S } \mathbf { K } ^ { S \top } / \sqrt { d _ { k } ^ { S } } , \qquad \mathbf { Z } _ { S } = \mathbf { S } ^ { S } + \log \mathbf { A } ^ { \mathrm { p h y } } ,
$$

$$
\mathbf { A } _ { S } = \operatorname { s o f t m a x } _ { \operatorname { r o w } } ( \mathbf { Z } _ { S } ) ,\tag{14}
$$

![](images/45c9e69b13e6c8f5ff893e34620abc429101ec34acea1b33b84849e0c4097d68.jpg)  
Fig. 3. Dual-prompt graph generator. (a) Proposition 1: with shared prompts γ and η, the term $( \mathbf { a } _ { i } + \boldsymbol { \gamma } ) \boldsymbol { \eta } ^ { \top }$ is constant within each row (red frame) and is removed by the row-wise softmax. (b) Electrode-specific prompts make $\mathbf { q } _ { i } \pi _ { j } ^ { e \top }$ vary with i and j, so the eye movements change the graph. (c) The physical prior acts as a soft mask: softmax $\mathrm { { r o w } } \left( \mathbf { S } \right) \odot \mathbf { A } ^ { \mathrm { { p h y } } }$ normalized per row equals softmax (S + log A<sup>phy</sup>).

where $\phi _ { s } ( \mathbf { E } ^ { S } ) = \mathrm { v e c } ( \mathbf { E } ^ { S } ) \mathbf { W } _ { s } + \mathbf { b } _ { s }$ . The student backbone follows Eq. (10) with ${ \bf { I } } ^ { ( 0 ) } = { \bf { E } } ^ { S } , { \bf { A } } = { \bf { A } } _ { S }$ and $L _ { S } < L _ { T }$ layers of width $d _ { h } ^ { S } \ < \ d _ { h } ^ { T }$ , and a mean–max readout gives $\mathbf { P } _ { S }$ . Because both generators have the same form, $\mathbf { A } _ { S }$ can express connectivity of the same kind as ${ \bf A } _ { T }$ , and distillation teaches the student to approximate the teacher graph from EEG alone. The student needs neither an eye tracker nor a buffer of past windows.

## 3.4 Bidirectional topology distillation and deployment

As Fig. 4(a) illustrates, three losses align the two networks at the graph, at the node representations and at the output, and their gradients reach both networks. At the graph level, with the softened topology $\mathbf { A } ^ { \tau } = \mathrm { s o f t m a x } _ { \mathrm { r o w } } ( \mathbf { Z } / \tau )$

$$
\mathcal { L } _ { \mathrm { t o p o } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { K L } \big ( \mathbf { A } _ { T , i } ^ { \tau } \big | \big | \mathbf { A } _ { S , i } ^ { \tau } \big ) + \frac { 1 } { N } \big | \big | \mathbf { A } _ { T } - \mathbf { A } _ { S } \big | \big | _ { F } ^ { 2 } .\tag{15}
$$

The KL term matches each electrode’s distribution over all neighbors, including weak edges; the Frobenius term matches the edges that dominate message passing. Proposition 2 relates this term to the output of a graph layer; it is a standard Lipschitz bound, stated for completeness rather than as a new result.

Proposition 2 (Topology gap bounds the message gap). For node features $\mathbf { \bar { H } } \in \mathbb { R } ^ { \breve { N } \times d } ,$ a weight matrix $\mathbf { \bar { W } } \in \mathbb { R } ^ { d \times d ^ { \prime } }$ and an element-wise activation δ with Lipschitz constant $L _ { \delta } ,$

$$
\begin{array} { r l } & { \left\| { \delta { ( { \bf A } _ { T } { \bf H } { \bf W } ) } - \delta { ( { \bf A } _ { S } { \bf H } { \bf W } ) } } \right\| _ { F } } \\ & { \qquad \leq L _ { \delta } \| { \bf H } \| _ { 2 } \| { \bf W } \| _ { 2 } \| { \bf A } _ { T } - { \bf A } _ { S } \| _ { F } . } \end{array}\tag{16}
$$

Proof: Applied to each entry, the Lipschitz property $\mathrm { g i v e s } \ \lVert \delta ( \mathbf { \dot { M } } ) - \dot { \delta } ( \mathbf { \dot { M } } ^ { \prime } ) \rVert _ { F } \leq L _ { \delta } \lVert \mathbf { M } - \mathbf { \dot { M } } ^ { \prime } \rVert _ { F } . \dot { \operatorname { S e t t i n g } } \mathbf { \dot { M } } - \mathbf { \dot { M } } ^ { \prime } =$ $( \mathbf { A } _ { T } - \mathbf { A } _ { S } ) \mathbf { H } \mathbf { W }$ and applying $\lVert \mathbf { X } \mathbf { Y } \rVert _ { F } \leq \lVert \mathbf { X } \rVert _ { F } \lVert \mathbf { Y } \rVert _ { 2 }$ and $\mathbf { \| \mathbf { H } \mathbf { W } \| _ { 2 } } \leq \| \mathbf { H } \| _ { 2 } \| \mathbf { W } \| _ { 2 }$ gives Eq. (16). □

The bound holds for any node features and weights, in particular for those of the student. As Fig. 4(b) illustrates, when $\mathcal { L } _ { \mathrm { t o p o } }  0 _ { \mathrm { \ell } }$ , propagating the student’s features over $\mathbf { A } _ { S }$ approaches propagating them over the teacher graph; layer normalization bounds $\| \mathbf { H } \| _ { 2 }$

At the node level, with H<sup>¯</sup> the row-normalized $\mathbf { H } ,$

$$
\begin{array} { r } { { \bf R } = \mathrm { s o f t m a x } _ { \mathrm { r o w } } \left( { \bar { \bf H } \bar { \bf H } ^ { \top } } / \tau _ { r } \right) , } \\ { { \mathcal L } _ { \mathrm { r e l } } = \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } \mathrm { K L } \big ( { \bf R } _ { T , i } \big | \big | { \bf R } _ { S , i } \big ) . } \end{array}\tag{17}
$$

The matrix R is $N \times N$ in both networks, so no projection between widths is needed; unlike relational distillation across samples [41], [42], it relates the electrodes of one window, and $\tau _ { r } < 1$ sharpens the cosine similarities.

At the output, temperature-scaled logit distillation [39] gives

$$
\mathcal { L } _ { \mathrm { l o g i t } } = \tau ^ { 2 } \mathrm { K L } \big ( \mathrm { s o f t m a x } ( \mathbf { P } _ { T } / \tau ) \mathrm { \| s o f t m a x } ( \mathbf { P } _ { S } / \tau ) \big ) .\tag{18}
$$

Teacher and student are trained together with

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { c e } } ( { \bf P } _ { T } , y ) + \mathcal { L } _ { \mathrm { c e } } ( { \bf P } _ { S } , y ) + \lambda _ { a } \mathcal { L } _ { \mathrm { a u x } } } \\ & { ~ + \omega ( e ) \big ( \lambda _ { 1 } \mathcal { L } _ { \mathrm { t o p o } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { r e l } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { l o g i t } } \big ) , } \end{array}\tag{19}
$$

where $\mathcal { L } _ { \mathrm { c e } }$ is the cross-entropy, $\mathcal { L } _ { \mathrm { a u x } }$ the sum of the crossentropies of the two auxiliary heads with $\lambda _ { a } ~ = ~ 0 . 5 ,$ e the epoch and $\omega ( e ) = \mathrm { m i n } ( 1 , e / E _ { w } )$ a ramp that delays distillation while the graphs form, as Fig. 4(c) shows. The distillation losses update both networks: the student learns the graph, relations and predictions of the teacher, and the teacher receives the gradients of the same losses. Deep mutual learning [44] couples peers at the output only; here the coupling also acts on the graph through which the privileged information enters, so that the two networks can be read on one connectivity. The validation protocol selects between this bidirectional form and a one-directional variant with sg(·) on all teacher terms. Training is a single stage.

![](images/2eada9279574b3c252d2573ec524907d33b2a787ec6f92077f9ceae75ccafb56.jpg)  
Fig. 4. Bidirectional multi-level topology distillation. (a) $\mathcal { L } _ { \mathrm { t o p o } }$ compares rows of the softened topologies, $\mathcal { L } _ { \mathrm { r e l } }$ rows of the $N \times N$ electrode relation matrices, which need no projection between networks of different width, and $\mathcal { L } _ { \mathrm { l o g i t } }$ the class distributions. (b) Proposition 2: the topology difference ∆A bounds the message difference. (c) Training: ω(e) ramps to 1 over $E _ { w }$ epochs, and the distillation gradients reach both networks.

At deployment, systems with an eye tracker use the teacher. The tracker’s validity check marks each window; windows without valid data, including those in the TCB history, enter with their eye features set to the training mean. Systems without an eye tracker can run the teacher in the same way, with all eye features at the training mean, or the student, which needs a quarter of the computation and no buffer of past windows. For comparison, Section 4.4 also evaluates a dual-mode variant that uses the student for the windows without valid eye data.

## 4 EXPERIMENTS

## 4.1 Experimental setup

SEED [3] and SEED-IV [4] provide synchronized 62-channel EEG (ESI NeuroScan) and eye movements (SMI glasses) recorded during emotional film clips. SEED has 15 subjects with three sessions of the same 15 clips, labeled positive, neutral or negative; eye movements exist for 12 subjects (1– 5 and 8–14), on which all methods are evaluated. SEED-IV has 15 subjects with three sessions of 24 different clips each, labeled happy, sad, fear or neutral. The official eyemovement features cover pupil diameter, dispersion, fixations, saccades, blinks and events (d<sub>e</sub> = 33 on SEED, 31 on SEED-IV).

DE features [20] are computed from the 200-Hz EEG on 4-s non-overlapping windows in the δ, θ, α, β and γ bands (2526 windows per subject on SEED, 2505 on SEED-IV). No smoothing across windows is applied, so on the EEG side the context bank is the only source of temporal information. The official eye-movement features, computed on the same windows and smoothed by the providers, are used identically by all methods. Because EEG and eye-movement features shift between subjects and sessions, all methods use a session-wise calibration: each feature is standardized with the mean and standard deviation of its recording session, computed over all windows of the session without labels. The calibration is transductive, since it uses the unlabeled windows of the test session, and it is applied identically to every method.

Published cross-subject accuracies on these datasets, often above 85% for EEG with eye movements [32], [35], are obtained under protocols that differ in feature smoothing, session splits, domain adaptation to the test subject and model selection. Under a strict protocol with selection on validation subjects, the benchmarks LibEER [54] and LibE-MER [55] report far lower accuracies, for example 56.89% for the best EEG–eye model on SEED in LibEMER. We therefore re-implement all baselines under one protocol instead of quoting published numbers.

TABLE 1  
Leave-one-subject-out results on SEED and SEED-IV (mean ± standard deviation over folds, %). Worst: accuracy on the most difficult test subject. Hist.: the K = 5 preceding windows. Best in bold, second best underlined. Markers: two-sided Wilcoxon signed-rank test on per-fold accuracies with Holm correction, at the 0.05 level, EEG-only methods against the student and all other methods against the teacher; <sup>▲</sup>/<sup>▽</sup>: significantly higher/lower than CTD-GNet.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">Input</td><td colspan="3">SEED (3 classes)</td><td colspan="3">SEED-IV (4 classes)</td></tr><tr><td>Accuracy</td><td>Macro-F1</td><td>Worst</td><td>Accuracy</td><td>Macro-F1</td><td>Worst</td></tr><tr><td rowspan="4">EEG only</td><td>SVM</td><td>EEG</td><td> $6 4 . 0 2 \pm 6 . 3 6$ </td><td> $6 3 . 7 6 \pm 6 . 3 7$ </td><td>54.51</td><td> $4 3 . 8 5 \pm 4 . 6 8 ^ { \pm }$ </td><td> $4 3 . 4 6 \pm 4 . 9 0$ </td><td>36.61</td></tr><tr><td>EEGNet [21]</td><td>EEG</td><td> $5 1 . 0 1 \pm 7 . 8 6 ^ { \nabla }$ </td><td> $4 7 . 2 6 \pm 9 . 3 9$ </td><td>35.00</td><td> $3 5 . 5 7 \pm 5 . 3 3 ^ { \nabla }$ </td><td> $3 1 . 9 9 \pm 6 . 1 6$ </td><td>26.60</td></tr><tr><td>DGCNN [25]</td><td>EEG</td><td> $6 3 . 6 8 \pm 6 . 8 3$ </td><td> $6 3 . 3 4 \pm 7 . 0 2$ </td><td>53.68</td><td> $4 2 . 7 9 \pm 5 . 1 8$ </td><td> $4 2 . 4 7 \pm 5 . 3 4$ </td><td>34.07</td></tr><tr><td>RGNN [26]</td><td>EEG</td><td> $5 9 . 0 9 \pm 5 . 4 2 ^ { \nabla }$ </td><td> $5 8 . 1 6 \pm 5 . 7 0$ </td><td>50.09</td><td> $3 6 . 8 6 \pm 3 . 2 6 ^ { \nabla }$ </td><td> $3 5 . 8 5 \pm 3 . 9 5$ </td><td>30.42</td></tr><tr><td rowspan="2">Eye only</td><td>MLP</td><td>Eye</td><td> $7 5 . 0 8 \pm 9 . 6 7 ^ { \circ }$ </td><td> $7 5 . 1 0 \pm 9 . 6 8$ </td><td>57.28</td><td> $7 1 . 5 1 \pm 1 0 . 9 7 ^ { \circ }$ </td><td> $7 1 . 1 3 \pm 1 1 . 0 4$ </td><td>44.70</td></tr><tr><td>Random forest</td><td>Eye</td><td> $7 4 . 6 8 \pm 1 1 . 2 7 ^ { \circ }$ </td><td> $7 4 . 6 0 \pm 1 1 . 1 7$ </td><td>57.84</td><td> $7 0 . 7 1 \pm 1 3 . 6 4$ </td><td> $7 0 . 2 3 \pm 1 3 . 7 3$ </td><td>38.92</td></tr><tr><td rowspan="9">Fusion</td><td>Feature concatenation</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 2 . 5 6 \pm 6 . 8 7 ^ { \nabla }$ </td><td> $8 2 . 4 3 \pm 6 . 9 0$ </td><td>66.67</td><td> $7 3 . 4 1 \pm 9 . 4 4 ^ { \nabla }$ </td><td> $7 3 . 3 0 \pm 9 . 3 1$ </td><td>50.84</td></tr><tr><td>BDAE [4], [7]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 2 . 2 9 \pm 7 . 4 2$ </td><td> $8 2 . 2 1 \pm 7 . 4 1$ </td><td>63.79</td><td> $7 3 . 6 9 \pm 9 . 4 3$ </td><td> $7 3 . 5 6 \pm 9 . 2 8$ </td><td>51.99</td></tr><tr><td>DCCA [6]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $7 6 . 5 4 \pm 6 . 5 5 ^ { \circ }$ </td><td> $7 6 . 2 9 \pm 6 . 5 7$ </td><td>62.66</td><td> $6 8 . 3 6 \pm 1 1 . 7 6 ^ { \circ }$ </td><td> $6 7 . 4 5 \pm 1 1 . 7 9$ </td><td>36.89</td></tr><tr><td>DCCA-AM [6]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 0 . 8 7 \pm 6 . 0 1 ^ { \nabla }$ </td><td> $8 0 . 6 9 \pm 6 . 1 2$ </td><td>68.26</td><td> $7 3 . 1 0 \pm 8 . 9 1$ </td><td> $7 2 . 9 7 \pm 8 . 8 4$ </td><td>50.05</td></tr><tr><td>TFN [8]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 0 . 9 7 \pm 8 . 9 0 ^ { \nabla }$ </td><td> $8 0 . 8 3 \pm 9 . 0 2$ </td><td>56.91</td><td> $7 4 . 7 4 \pm 1 0 . 0 0$ </td><td> $7 4 . 5 6 \pm 9 . 8 6$ </td><td>49.45</td></tr><tr><td>LMF [9]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 2 . 1 0 \pm 7 . 4 3$ </td><td> $8 1 . 9 7 \pm 7 . 5 2$ </td><td>63.42</td><td> $7 4 . 4 4 \pm 9 . 8 9$ </td><td> $7 4 . 0 9 \pm 1 0 . 0 7$ </td><td>49.22</td></tr><tr><td>MISA [36]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $7 7 . 7 8 \pm 8 . 6 9 ^ { \nabla }$ </td><td> $7 7 . 6 1 \pm 8 . 8 9$ </td><td>63.13</td><td> $7 1 . 6 2 \pm 8 . 8 2 ^ { \nabla }$ </td><td> $7 1 . 4 1 \pm 8 . 8 4$ </td><td>51.94</td></tr><tr><td>Cross-attention [10]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $7 5 . 5 3 \pm 9 . 4 2 ^ { \nabla }$ </td><td> $7 5 . 3 5 \pm 9 . 5 2$ </td><td>57.21</td><td> $6 9 . 6 9 \pm 9 . 7 1 ^ { \bigtriangledown }$ </td><td> $6 9 . 4 0 \pm 9 . 4 6$ </td><td>42.79</td></tr><tr><td>Bimodal-LSTM [14]</td><td> $\mathrm { E E G } + \mathrm { E y e } + \mathrm { H i s t . }$ </td><td> $8 3 . 7 5 \pm 6 . 6 0$ </td><td> $\underline { { 8 3 . 5 6 } } \pm 6 . 7 9$ </td><td>67.12</td><td> $\underline { { 7 6 . 0 5 } } \pm 9 . 3 1$ </td><td> $\underline { { 7 5 . 8 6 } } \pm 9 . 0 8$ </td><td>52.12</td></tr><tr><td>Missing-</td><td>Cross-attention + ModDrop [15]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $7 8 . 1 0 \pm 8 . 9 5 ^ { \circ }$ </td><td> $7 7 . 9 4 \pm 9 . 0 8$ </td><td>55.66</td><td> $7 0 . 2 2 \pm 9 . 9 9 ^ { \nabla }$ </td><td> $6 9 . 9 8 \pm 9 . 8 4$ </td><td>43.01</td></tr><tr><td rowspan="2">modality</td><td>MMIN [16]</td><td> $\mathrm { E E G } + \mathrm { E y e }$ </td><td> $8 2 . 6 1 \pm 7 . 8 3$ </td><td> $8 2 . 4 9 \pm 7 . 9 2$ </td><td>64.16</td><td> $7 5 . 2 9 \pm 9 . 2 0$ </td><td> $7 5 . 1 3 \pm 9 . 1 9$ </td><td>51.46</td></tr><tr><td>CTD-GNet teacher</td><td> $\mathrm { E E G } + \mathrm { E y e } + \mathrm { H i s t . }$ </td><td> $\mathbf { 8 4 . 9 1 \pm 5 . 6 0 }$ </td><td> $8 4 . 6 7 \pm 5 . 7 5$ </td><td>74.02</td><td> ${ \bf 7 6 . 1 7 \pm 7 . 6 8 }$ </td><td> ${ \bf 7 6 . 0 1 \pm 7 . 5 2 }$ </td><td>54.81</td></tr><tr><td>Ours</td><td>CTD-GNet student</td><td>EEG</td><td> $6 4 . 1 0 \pm 6 . 3 0$ </td><td> $6 3 . 6 3 \pm 6 . 3 0$ </td><td>53.95</td><td> $4 2 . 0 4 \pm 5 . 3 1$ </td><td> $4 1 . 8 8 \pm 5 . 4 3$ </td><td>34.76</td></tr></table>

All experiments use leave-one-subject-out (LOSO) crossvalidation with 12 folds on SEED and 15 on SEED-IV. In each fold, two training subjects form the validation set for early stopping and all selection; the labels of the test subject are never used. For CTD-GNet, $K , \tau$ and the loss weights are selected by the mean validation macro-F1 over the folds in a base configuration, which uses a temporal convolutional network (TCN) as context encoder and onedirectional distillation; the context encoder (GRU or TCN) and the direction of distillation are then selected in the same way. The selected final configuration uses a GRU and bidirectional distillation. Each fold is run with three seeds; we report the mean and standard deviation over folds and test differences with the two-sided Wilcoxon signed-rank test on per-fold accuracies, with Holm correction over the baselines of each comparison [56] at the 0.05 level.

and the random forest [58] of 500 trees its maximum depth from {unlimited, 10, 20}, and each distillation baseline its loss weight from {0.1, 0.5, 1, 2}. The neural baselines follow the architectures of their original papers and share the optimizer, schedule, early stopping and validation subjects with CTD-GNet. The search of CTD-GNet covers $K , \tau$ and three loss weights, over which its validation accuracy varies by less than half a point, as Section 4.3 shows.

All 17 baselines are re-implemented with the same folds, features, calibration and validation subjects. The EEG-only baselines are an SVM with an RBF kernel on the 310- dimensional DE vector, EEGNet [21] on raw 4-s segments, and DGCNN [25] and RGNN [26] without domain adaptation; the eye-only baselines are an MLP and a random forest. The fusion baselines are feature concatenation, BDAE [4], [7], DCCA [6], [30], DCCA-AM [6], TFN [8], LMF [9], MISA [36], cross-modal attention [10] over 62 electrode tokens and one eye token, and Bimodal-LSTM [14] with the same $K = 5$ preceding windows as the teacher. The missing-modality baselines are cross-attention with Mod-Drop [15] at a drop rate of 0.5 and MMIN [16]. For EEG-only inference, the CTD-GNet student is also trained with vanilla KD [39], FitNets [40], RKD [41], SP [42] and LSP [43]. On the validation subjects, the SVM [57] selects $C \in \{ 0 . 1 , 1 , 1 0 \}$

The teacher has three graph layers of width $d _ { h } = 6 4$ with $d _ { k } = 3 2 , d _ { s } = 3 2$ and $d _ { c } = d _ { r } = 6 4 ;$ ; the student has two layers of width 32 with $d _ { k } \ : = \ : 1 6 ,$ and both use a dropout rate of 0.3. They are trained jointly with AdamW [59] at a learning rate of $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 2 }$ and batch size 128, with a cosine decay to $\mathrm { \bar { 1 } 0 ^ { - 5 } }$ over at most 100 epochs, mixup [60] with $\alpha = 0 . 4$ on windows and histories, and an exponential moving average of the weights with decay 0.995 [61]; each network keeps its epoch with the best validation macro-F1, with a patience of 20 epochs. The distillation ramp lasts $E _ { w } = 1 0$ epochs and $\tau _ { r } = 0 . 1$ . The search covers $\lambda _ { i } \in \{ 0 . 5 , 1 , 2 \} , \tau \in \{ 2 , 4 , 8 \}$ and $K \in \{ 3 , 5 , 7 , 9 \}$ ; it selects $K = 5 \mathrm { ~ o n ~ }$ both datasets, $\tau = 4$ and $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } ) = ( 2 , 2 , 2 )$ on SEED, and $\tau = 2$ and $( 0 . 5 , 2 , 0 . 5 )$ on SEED-IV.

## 4.2 Comparison with existing methods

Table 1 reports the results with complete eye data. The CTD-GNet teacher has the highest mean accuracy on both datasets, 84.91% on SEED and 76.17% on SEED-IV, and the highest macro-F1. With Holm correction for multiple comparisons, it is significantly more accurate than 9 of the 13 eye-only, fusion and missing-modality baselines on SEED and 6 of them on SEED-IV, while its margin over Bimodal-LSTM and MMIN is not significant. Its advantage is largest where the baselines are weakest: on its most difficult test subject it reaches 74.02% on SEED and 54.81% on SEED-IV,

5.8 and 2.7 points above the worst-subject accuracy of any baseline, and it has the smallest standard deviation across subjects among multimodal models, 5.60 and 7.68 points. Fig. 5 shows where this consistency comes from. On SEED, the teacher has the highest lower quartile and the highest worst-subject accuracy of all models; on SEED-IV, its two most difficult test subjects reach 54.8% and 66.6%, against 52.1% and 58.7% for Bimodal-LSTM and 51.5% and 58.5% for MMIN. Table 2 gives the F1 score of each class: the teacher is the best model for all three emotions on SEED and for neutral and happy on SEED-IV.

The strongest baseline, Bimodal-LSTM, also reads the preceding windows, which points to the value of temporal context; CTD-GNet uses this context to shape the electrode graph and is more consistent, with a worst-subject accuracy 6.9 and 2.7 points higher. The teacher exceeds the eye-only MLP by 9.8 and 4.7 points, the largest multimodal gain of all models, so the graph extracts EEG information that complements the eye movements. The student, which sees only the current EEG window, reaches 64.10% on SEED and 42.04% on SEED-IV, on par with DGCNN, and needs neither an eye tracker nor a window buffer.

Table 3 compares predictions without eye movements; each distillation baseline is listed with the loss weight selected on the validation subjects. Topology distillation gives the most accurate distilled student on SEED-IV; its main benefits, the graph alignment and, in the bidirectional form, the gain of the teacher, are analyzed in Sections 4.3 and 4.5. The student outperforms ModDrop and, unlike MMIN, needs no reconstruction network. Without its eye input, Bimodal-LSTM, the strongest baseline with complete data, falls below every student, to 61.04% on SEED and 37.22% on SEED-IV, whereas the CTD-GNet teacher, whose graph and context are computed from EEG, keeps 66.10% and 43.77%, the highest accuracy in Table 3, although its macro-F1 on SEED-IV is below that of all students and MMIN. A device without an eye tracker can therefore run the teacher with mean eye features or, at a quarter of the computation and without a buffer of past windows, the student.

Table 4 compares the computational cost. The student needs 24% of the teacher’s FLOPs and processes a window in 0.21 ms on one CPU core, three times faster than the teacher. The teacher has 59% of the parameters of Bimodal-LSTM, 39% of those of MMIN and 39% of the FLOPs of cross-attention, and all models run in under 1 ms per window, far within the 4-s update interval.

A recognizer in daily use meets new people and new stimuli at the same time. Because every subject of SEED and SEED-IV watches the same film clips [62], and the pupil responds to the luminance of a clip as well as to arousal [2], we also evaluate a stimulus-disjoint protocol that tests both kinds of generalization together. In each fold, one clip per emotion, on SEED-IV one per emotion and session, is removed from all training and validation subjects, and the test subject is evaluated on these clips only; the heldout clips rotate over the folds. Table 5 reports the accuracy on these unseen clips. On SEED, the CTD-GNet teacher is the most accurate model, 77.1%, and its lead widens under this protocol: over Bimodal-LSTM from 1.2 points in Table 1 to 4.5 points, and over MMIN from 2.3 to 5.9 points. With Holm correction, it is significantly more accurate than the eye-only MLP and DGCNN, $p = 0 . 0 1 7$ for both, whereas its lead over the fusion models is not significant, $p \ \geq \ 0 . 1 9 .$ On SEED-IV, it matches the best model, Bimodal-LSTM, with 62.1% against 62.3%, and is 3.2 points above MMIN. All fusion models stay far above EEG alone, by about 20 points on SEED-IV, so the eye movements carry emotion information that transfers to new stimuli.

## 4.3 Ablation study

The upper part of Table 6 varies the final configuration. Replacing the GRU by a TCN, making the distillation one-directional or removing it lowers the teacher on both datasets, by 1.0 to 1.2 points on SEED, with uncorrected p between 0.027 and 0.044, and by 0.3 to 0.5 points on SEED-IV. One-directional distillation matches no distillation, because the stop-gradient isolates the teacher. Coupling only the predictions in both directions gives 84.57% and 76.25%, close to the full objective, whereas coupling only the graphs leaves the teacher at the level without distillation: the gain of the teacher is consistent with mutual learning through the predictions [44], and the topology coupling provides the 35- fold graph alignment of Section 4.5, which is its direct target. The eye movements can reach the teacher through either route. With the eye movements only in the graph, through the edge prompts, the node pathway and the context bank, or only in the readout, the teacher stays within 0.5 points of the full model on both datasets. Modulating the EEG graph thus carries the information of the eye movements without any eye features in the readout, which is the premise of CTD-GNet.

The lower part removes components from the base configuration, and every removal lowers the accuracy on both datasets. The largest effects come from the components that make the graph change with every window: without the temporal context the teacher loses 1.1 and 1.9 points, a static adjacency shared by all windows, as in DGCNN, costs 1.7 and 1.4 points, and removing both prompts costs 0.5 and 2.0 points, so the graph should follow both the current window and its history. Mixup and weight averaging add 0.1 to 0.8 points. Fig. 6 shows that on SEED the accuracy rises during the first minutes of a trial, for the teacher as for the student, as the induced emotion builds up.

The hyperparameters have small effects, as Fig. 7 shows: over histories of three to nine windows, over τ and over all 27 combinations of the loss weights, the validation accuracy varies by less than half a point, so CTD-GNet needs no careful tuning.

## 4.4 Robustness to missing and noisy eye data

At test time, a fraction $r \in \{ 0 , 2 0 , \ldots , 1 0 0 \}$ % of the windows lose their eye features, which are replaced by the training mean, also in the TCB history; in a second test, Gaussian noise with $\sigma _ { n } \in \{ 0 , 0 . 2 5 , 0 . 5 , 1 , 2 \}$ in standardized units is added to all eye features. Each setting is repeated with five random draws. Fig. 8 plots the accuracy, and Table 7 lists the values of the main models.

With missing eye data, the teacher is the most accurate model at every missing rate on SEED-IV, and at 80% it exceeds Bimodal-LSTM by 4.1 points and MMIN by 7.1 points, as Fig. 8(a)–(b) shows. On SEED, it is the most accurate model with complete eye data, at 20% and when all eye data are missing. Models that read only the current window degrade almost linearly; without eye data, crossattention falls to 36.2% on SEED and 24.0% on SEED-IV. The teacher degrades gracefully, likely because its context and eye cue still hold the valid eye data of preceding windows and its graph always carries the EEG evidence; without any eye data it keeps 66.1% on SEED, more than any EEGonly model, and 43.8% on SEED-IV. Whenever eye data are missing, the teacher alone is more accurate than the dual-mode variant, so the full system needs no switching mechanism.

(a) SEED, 12 test subjects  
![](images/119f6dcdd1e9fb985922b6948581dca47c70ec46a1b3745695f9dc773750493b.jpg)

![](images/ff2e80dfab02663e1edaee2088cd5817ca578d9b266a96d4a58f9b9b0c68179b.jpg)  
Fig. 5. Accuracy on each test subject for the 14 models that use eye movements. Boxes span the quartiles over the test subjects, whiskers reach the worst and the best subject, dots mark the individual subjects and diamonds the mean.

TABLE 2  
F1 score of each class (%), mean over the test subjects. Macro: mean over the classes, as in Table 1. The best value in each column is in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">SEED</td><td colspan="5">SEED-IV</td></tr><tr><td>Negative</td><td>Neutral</td><td>Positive</td><td>Macro</td><td>Neutral</td><td>Sad</td><td>Fear</td><td>Happy</td><td>Macro</td></tr><tr><td>SVM</td><td>54.8</td><td>57.3</td><td>79.3</td><td>63.8</td><td>45.9</td><td>43.7</td><td>38.7</td><td>45.6</td><td>43.5</td></tr><tr><td>EEGNet [21]</td><td>38.1</td><td>43.1</td><td>60.6</td><td>47.3</td><td>34.2</td><td>29.1</td><td>26.5</td><td>38.2</td><td>32.0</td></tr><tr><td>DGCNN [25]</td><td>53.4</td><td>57.5</td><td>79.1</td><td>63.3</td><td>43.7</td><td>42.0</td><td>38.5</td><td>45.6</td><td>42.5</td></tr><tr><td>RGNN [26]</td><td>44.5</td><td>51.6</td><td>78.3</td><td>58.2</td><td>40.3</td><td>33.0</td><td>29.5</td><td>40.5</td><td>35.9</td></tr><tr><td>MLP (eye only)</td><td>71.2</td><td>80.6</td><td>73.5</td><td>75.1</td><td>76.0</td><td>63.2</td><td>78.3</td><td>67.0</td><td>71.1</td></tr><tr><td>Random forest (eye only)</td><td>71.8</td><td>78.8</td><td>73.2</td><td>74.6</td><td>75.0</td><td>60.8</td><td>78.0</td><td>67.0</td><td>70.2</td></tr><tr><td>Feature concatenation</td><td>78.7</td><td>82.5</td><td>86.0</td><td>82.4</td><td>77.0</td><td>65.2</td><td>78.4</td><td>72.6</td><td>73.3</td></tr><tr><td>BDAE [4], [7]</td><td>79.2</td><td>82.5</td><td>84.9</td><td>82.2</td><td>77.9</td><td>65.1</td><td>77.9</td><td>73.3</td><td>73.6</td></tr><tr><td>DCCA [6]</td><td>71.8</td><td>76.7</td><td>80.3</td><td>76.3</td><td>74.5</td><td>57.3</td><td>76.4</td><td>61.6</td><td>67.5</td></tr><tr><td>DCCA-AM [6]</td><td>77.1</td><td>80.2</td><td>84.7</td><td>80.7</td><td>79.1</td><td>63.2</td><td>76.9</td><td>72.7</td><td>73.0</td></tr><tr><td>TFN [8]</td><td>77.9</td><td>81.9</td><td>82.7</td><td>80.8</td><td>78.9</td><td>66.9</td><td>78.9</td><td>73.6</td><td>74.6</td></tr><tr><td>LMF [9]</td><td>79.1</td><td>82.5</td><td>84.4</td><td>82.0</td><td>78.3</td><td>66.0</td><td>78.8</td><td>73.2</td><td>74.1</td></tr><tr><td>MISA [36]</td><td>74.6</td><td>81.2</td><td>77.0</td><td>77.6</td><td>77.4</td><td>62.3</td><td>76.1</td><td>69.9</td><td>71.4</td></tr><tr><td>Cross-attention [10]</td><td>72.8</td><td>78.6</td><td>74.7</td><td>75.3</td><td>76.6</td><td>60.2</td><td>75.1</td><td>65.7</td><td>69.4</td></tr><tr><td>Bimodal-LSTM [14]</td><td>80.3</td><td>83.4</td><td>86.9</td><td>83.6</td><td>80.8</td><td>68.5</td><td>79.3</td><td>74.8</td><td>75.9</td></tr><tr><td>Cross-attention + ModDrop [15]</td><td>74.8</td><td>79.8</td><td>79.3</td><td>77.9</td><td>75.8</td><td>61.3</td><td>75.2</td><td>67.6</td><td>70.0</td></tr><tr><td>MMIN [16]</td><td>79.7</td><td>83.3</td><td>84.5</td><td>82.5</td><td>79.5</td><td>67.4</td><td>79.3</td><td>74.3</td><td>75.1</td></tr><tr><td>CTD-GNet teacher</td><td>81.2</td><td>84.0</td><td>88.8</td><td>84.7</td><td>82.3</td><td>67.4</td><td>79.2</td><td>75.3</td><td>76.0</td></tr><tr><td>CTD-GNet student</td><td>53.8</td><td>57.0</td><td>80.1</td><td>63.6</td><td>43.0</td><td>40.7</td><td>38.2</td><td>45.6</td><td>41.9</td></tr></table>

With noisy eye data, the teacher is the most accurate model up to $\sigma _ { n } ~ = ~ 0 . 5$ on SEED and at $\sigma _ { n } ~ = ~ 0 . 2 5$ on SEED-IV, and second only to Bimodal-LSTM at the higher noise levels. At $\sigma _ { n } ~ = ~ 1$ , it keeps 80.5% on SEED and 66.5% on SEED-IV, 9.3 and 11.9 points above every singlewindow fusion and missing-modality model in Table 7. Two design choices likely contribute: averaging the eye cue over windows attenuates independent noise, and in the edge and node routes corrupted eye features can only reweight the edges and gate the node terms of an EEG graph, with the row-wise softmax and the log-prior bounding their influence on the edges.

## 4.5 Analysis of the learned graphs

Bidirectional distillation brings the teacher and student graphs close together, as Table 8 shows. Compared with a student trained without distillation, the mean row-wise KL divergence between ${ \bf A } _ { T }$ and ${ \bf A } _ { S }$ falls from 7.90 to 0.23 on

## TABLE 3

Teacher Distilled student (a) SEED  
Prediction without eye movements (accuracy and macro-F1, %). Upper block: student trained with different distillation losses in the base configuration. Middle block: multimodal models tested with all eye movements removed. Lower block: CTD-GNet in the final configuration.
<table><tr><td rowspan="2">Method</td><td colspan="2">SEED</td><td colspan="2">SEED-IV</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>Student without distillation</td><td>63.85</td><td>63.22</td><td>41.93</td><td>41.58</td></tr><tr><td>Vanilla KD [39]</td><td>64.46</td><td>63.84</td><td>41.85</td><td>41.54</td></tr><tr><td>FitNets [40]]</td><td>64.00</td><td>63.55</td><td>41.99</td><td>41.77</td></tr><tr><td>RKD [41]</td><td>63.92</td><td>63.25</td><td>41.55</td><td>41.17</td></tr><tr><td>SP [42]</td><td>63.61</td><td>63.04</td><td>42.08</td><td>41.78</td></tr><tr><td>LSP [43]</td><td>63.76</td><td>63.29</td><td>42.40</td><td>42.23</td></tr><tr><td>Topology distillation (ours)</td><td>63.96</td><td>63.20</td><td>42.49</td><td>42.19</td></tr><tr><td>Cross-attention + ModDrop [15]</td><td>62.00</td><td>61.02</td><td>38.31</td><td>36.41</td></tr><tr><td>MMIN [16]</td><td>62.76</td><td>62.26</td><td>42.61</td><td>42.19</td></tr><tr><td>Bimodal-LSTM [14]</td><td>61.04</td><td>57.47</td><td>37.22</td><td>28.15</td></tr><tr><td>Student without distillation</td><td>63.66</td><td>63.23</td><td>41.98</td><td>41.74</td></tr><tr><td>Student, bidirectional distillation</td><td>64.10</td><td>63.63</td><td>42.04</td><td>41.88</td></tr><tr><td>Teacher, eye movements removed</td><td>66.10</td><td>65.33</td><td>43.77</td><td>40.59</td></tr></table>

TABLE 4

Computational cost per window on SEED-IV (K = 5). FLOPs count a multiply-accumulate as two operations, with recurrent layers counted analytically; latency (ms) is measured at batch size one on one core of an Intel Xeon Gold 6430 CPU and on an NVIDIA RTX 4090 GPU. Hist.: K past windows.
<table><tr><td>Model</td><td>Input</td><td>Params</td><td>FLOPs</td><td>CPU</td><td>GPU</td></tr><tr><td>DGCNN [25]</td><td>EEG</td><td>131.5k</td><td>0.79M</td><td>0.09</td><td>0.20</td></tr><tr><td>RGNN [26]</td><td>EEG</td><td>2.6k</td><td>0.12M</td><td>0.08</td><td>0.20</td></tr><tr><td>DCCA [6]</td><td>EEG+Eye</td><td>120.7k</td><td>0.24M</td><td>0.08</td><td>0.15</td></tr><tr><td>TFN [8]</td><td>EEG+Eye</td><td>204.6k</td><td>0.41M</td><td>0.09</td><td>0.16</td></tr><tr><td>LMF [9]</td><td>EEG+Eye</td><td>65.3k</td><td>0.13M</td><td>0.10</td><td>0.21</td></tr><tr><td>MISA [36]</td><td>EEG+Eye</td><td>125.5k</td><td>0.44M</td><td>0.16</td><td>0.29</td></tr><tr><td>Cross-attention [10]</td><td>EEG+Eye</td><td>174.1k</td><td>12.43M</td><td>0.88</td><td>0.92</td></tr><tr><td>MMIN [16]</td><td>EEG+Eye</td><td>395.5k</td><td>0.52M</td><td>0.17</td><td>0.32</td></tr><tr><td>Bimodal-LSTM [14]</td><td>EEG+Eye+Hist.</td><td>262.7k</td><td>3.01M</td><td>0.36</td><td>0.21</td></tr><tr><td>CTD-GNet teacher</td><td>EEG+Eye+Hist.</td><td>155.0k</td><td>4.82M</td><td>0.63</td><td>0.89</td></tr><tr><td>CTD-GNet student</td><td>EEG</td><td>74.3k</td><td>1.15M</td><td>0.21</td><td>0.37</td></tr></table>

## TABLE 5

Generalization to new subjects and new stimuli: accuracy (%) on film clips that no training or validation subject has seen. In each fold, one clip per emotion, on SEED-IV one per emotion and session, is held out from all training and validation subjects.
<table><tr><td>Method</td><td>SEED</td><td>SEED-IV</td></tr><tr><td>DGCNN [25]</td><td>63.4</td><td>39.8</td></tr><tr><td>MLP (eye only)</td><td>63.2</td><td>57.6</td></tr><tr><td>Feature concatenation</td><td>70.2</td><td>61.7</td></tr><tr><td>Bimodal-LSTM [14]</td><td>72.6</td><td>62.3</td></tr><tr><td>MMIN [16]</td><td>71.2</td><td>58.9</td></tr><tr><td>CTD-GNet student</td><td>64.2</td><td>38.9</td></tr><tr><td>CTD-GNet teacher</td><td>77.1</td><td>62.1</td></tr></table>

SEED-IV and from 7.55 to 0.21 on SEED, and the Jaccard overlap of the 10% strongest edges rises from 0.093 to 0.226 and from 0.209 to 0.414. From a single EEG window, the distilled student shares 2.0 to 2.4 times as many of the teacher’s strongest edges as the student without distillation, so both networks can be read on one connectivity.

On SEED-IV, the teacher graph of each emotion is organized around bilateral hubs at the lateral temporal elec-

## TABLE 6

Ablation of the teacher (accuracy and macro-F1, %). Upper block: the final configuration and variants of it. Lower block: components removed from the base configuration, which uses a TCN context encoder and one-directional distillation. $\stackrel { . } { : } p < 0 . 0 5$ against the first row of the block,

two-sided Wilcoxon signed-rank test on per-fold accuracies, uncorrected; with Holm correction within each block, no difference reaches the 0.05 level.
<table><tr><td rowspan="2">Variant</td><td colspan="2">SEED</td><td colspan="2">SEED-IV</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>Final (GRU, bidirectional)</td><td>84.91</td><td>84.67</td><td>76.17</td><td>76.01</td></tr><tr><td>TCN instead of GRU</td><td>83.92†</td><td>83.68</td><td>75.65</td><td>75.43</td></tr><tr><td>One-directional distillation</td><td>83.71†</td><td>83.52</td><td>75.90</td><td>75.74</td></tr><tr><td>No distillation</td><td>83.71†</td><td>83.53</td><td>75.82</td><td>75.65</td></tr><tr><td>Bidirectional, logit loss only</td><td>84.57</td><td>84.35</td><td>76.25</td><td>76.08</td></tr><tr><td>Bidirectional, topology loss only</td><td>83.74</td><td>83.55</td><td>75.98</td><td>75.79</td></tr><tr><td>Eye movements only in the graph</td><td>84.81</td><td>84.60</td><td>75.97</td><td>75.82</td></tr><tr><td>Eye movements only in the readout</td><td>84.45</td><td>84.27</td><td>76.40</td><td>76.19</td></tr><tr><td>Base (TCN, one-directional)</td><td>82.19</td><td>82.00</td><td>75.71</td><td>75.53</td></tr><tr><td>w/o temporal context</td><td>81.09</td><td>80.98</td><td>73.78†</td><td>73.50</td></tr><tr><td>w/o both prompts</td><td>81.74</td><td>81.60</td><td>73.75</td><td>73.52</td></tr><tr><td>Static learnable adjacency</td><td>80.45†</td><td>80.28</td><td>74.32</td><td>74.09</td></tr><tr><td>w/o mixup</td><td>82.12</td><td>82.03</td><td>74.90</td><td>74.60</td></tr><tr><td>w/o weight averaging</td><td>81.87</td><td>81.68</td><td>74.98</td><td>74.82</td></tr></table>

![](images/9c2dad15bbd3563ceccf24b5e61854e42770240562d3656db3dcd44dee74c8eb.jpg)

(b) SEED-IV  
![](images/6bcc8a6cfdd5b9cd04fd64fd578b377d51985c18d85cd1f00db80e83631a8573.jpg)  
Time from trial onset (s)  
Time from trial onset (s)  
Fig. 6. Test accuracy versus the position of the window within the trial, in 20-s bins plotted at the bin centre, over all folds and seeds. Bins with fewer than 1000 test windows are omitted.

trodes, as Fig. 9 shows, and for sadness and fear a centroparietal midline hub collects edges from across the scalp. The distilled student shares these hubs, whereas the student without distillation concentrates its edges on one left temporal electrode for every emotion. On SEED, Fig. 10 shows that the teacher and the distilled student share bilateral hubs at the lateral temporal electrodes, the region that Zheng and Lu [3] found most discriminative.

In the eye-movement attribution ∆A of Figs. 9 and 10, the edges that lose weight link the temporal hubs to frontal and central electrodes. Table 9 quantifies this reorganization by the attention that the electrodes of each scalp region receive, $\begin{array} { r } { \frac { 1 0 0 } { N } \sum _ { i } \sum _ { j \in \mathcal { R } } \Delta A _ { i j } } \end{array}$ for a region $\mathcal { R } ;$ because every row of ${ \bf A } _ { T }$ sums to one, the changes sum to zero over the regions. On both datasets and for every emotion, the eye movements take attention away from the temporal electrodes, by 1.30 to 2.43 points on SEED and 0.24 to 0.55 points on SEED-IV, and give it to the frontal electrodes, by 0.49 to 0.95 and 0.27 to 0.58 points; the occipital electrodes gain in six of the seven emotion classes. This shift toward frontal electrodes, near the eyes and the frontal eye fields of the dorsal attention network [11], and toward occipital electrodes over the visual cortex is consistent with the involvement of these regions in eye movements and visual processing.

![](images/f957cd70a75f19d9627cc6c9a55ae70e78b005536ca9492ec273aa1b62af7815.jpg)

![](images/2855c755e498a8d7624caa3401c10f5cf6e5ff37556f7f2df06b28b17bf99180.jpg)  
Fig. 7. Validation accuracy over the hyperparameter grid, relative to the selected configuration, which lies at zero. Each panel varies one hyperparameter and keeps the others at their selected values. The temperature and the loss weights are searched in the base configuration, whose stop-gradient leaves the teacher unaffected by them, so only the student is shown in these panels.  
Fig. 8. Accuracy under missing (a, b) and noisy (c, d) eye-movement data, mean over the test subjects. Blue band: teacher ± one standard deviation over the test subjects. Grey band: range of the other seven models that use eye movements, namely the eye-only MLP, feature concatenation, DCCA-AM, TFN, LMF, MISA and ModDrop. The dual mode uses the student for windows without eye data; under noise, it differs from the teacher only on the few windows whose eye data are invalid in the recordings.

To relate the graph to the frequency bands, we define the band-wise topology sensitivity of electrode i as

$$
I _ { b , i } = \mathbb { E } _ { t } \Big [ \sum _ { j = 1 } ^ { N } \big | A _ { T , i j } - A _ { T , i j } ^ { ( \setminus b ) } \big | \Big ] ,\tag{20}
$$

where $\mathbf { A } _ { T } ^ { ( \backslash b ) }$ is computed with band b set to its calibrated mean. Fig. 11 maps $I _ { b , i }$ over the scalp for the $\alpha , ~ \beta$ and $\gamma$ bands. On SEED-IV, for every emotion, the mean sensitivity over the electrodes is higher in the $\beta$ and $\gamma$ bands, 0.22 to 0.23 and 0.19 to 0.23, than in the α band, 0.18 to 0.19, and in each of these bands the most sensitive electrode is the midline electrode CPz or FPz or the left temporal electrode T7. On SEED, the sensitivity grows from the α band, 0.13 to 0.14, to the $\beta$ band, 0.18 to 0.19, and the $\gamma$ band, 0.24 to 0.27, and the parietal electrode P1 is the most sensitive in the $\gamma$ band for every emotion. This emphasis is consistent with Zheng and Lu [3], who found the $\bar { \boldsymbol { \beta } }$ and $\gamma$ bands most discriminative.

## 5 DISCUSSION

## 5.1 Eye movements as modulators of cortical connectivity

CTD-GNet tests the hypothesis that the emotion information in eye movements and temporal context can be expressed as a modulation of how cortical regions interact [11]. The results are consistent with it. The components that make the graph depend on the window and its context have the largest ablation effects, and the teacher keeps its accuracy when the eye movements only modulate the EEG graph and never enter the readout, so modulating the graph is a sufficient route for the eye information. The eye movements shift the attention of the graph from the temporal electrodes toward frontal electrodes for every emotion and toward occipital electrodes in six of seven classes, which may reflect the role of these regions in eye movements and vision. The model that keeps EEG as the substrate is the most consistent across subjects, has the highest mean accuracy, degrades gracefully without eye data and is the most accurate on film clips unseen in training on SEED. Eye movements are the stronger single modality on both datasets, yet this model gains the most over eye-only recognition: the role of a signal in a model need not follow its standalone accuracy.

TABLE 7  
Accuracy (%) under missing and noisy eye-movement data, mean over the test subjects and five random draws. The results without corruption are those of Table 1. The best value in each column is in bold.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="5">Missing rate r</td><td colspan="4">Noise level  $\sigma _ { n }$ </td></tr><tr><td>20%</td><td>40%</td><td>60%</td><td>80%</td><td>100%</td><td>0.25</td><td>0.5</td><td>1</td><td>2</td></tr><tr><td rowspan="10">SEED</td><td>MLP (eye only)</td><td>66.7</td><td>58.5</td><td>50.0</td><td>41.7</td><td>33.4</td><td>73.0</td><td>67.8</td><td>57.7</td><td>47.2</td></tr><tr><td>Feature concatenation</td><td>78.1</td><td>73.5</td><td>69.0</td><td>64.4</td><td>60.0</td><td>81.6</td><td>78.9</td><td>71.2</td><td>58.5</td></tr><tr><td>DCCA-AM [6]</td><td>76.4</td><td>71.8</td><td>67.3</td><td>62.8</td><td>58.3</td><td>79.6</td><td>76.5</td><td>69.4</td><td>59.9</td></tr><tr><td>TFN [8]</td><td>75.9</td><td>70.7</td><td>65.5</td><td>60.4</td><td>55.3</td><td>79.6</td><td>75.8</td><td>67.3</td><td>56.6</td></tr><tr><td>LMF [9]</td><td>76.9</td><td>71.6</td><td>66.3</td><td>61.1</td><td>55.9</td><td>80.1</td><td>76.0</td><td>67.2</td><td>56.5</td></tr><tr><td>MISA [36]</td><td>71.1</td><td>64.5</td><td>57.7</td><td>51.1</td><td>44.5</td><td>74.9</td><td>69.3</td><td>59.5</td><td>49.8</td></tr><tr><td>Cross-attention [10]</td><td>67.6</td><td>59.8</td><td>51.9</td><td>44.0</td><td>36.2</td><td>73.0</td><td>67.3</td><td>56.9</td><td>46.8</td></tr><tr><td>Bimodal-LSTM [14]</td><td>83.6</td><td>82.8</td><td>80.3</td><td>74.2</td><td>61.0</td><td>83.4</td><td>82.6</td><td>80.8</td><td>76.5</td></tr><tr><td>Cross-attention + ModDrop [15]</td><td>74.9</td><td>71.6</td><td>68.4</td><td>65.2</td><td>62.0</td><td>75.7</td><td>70.5</td><td>59.7</td><td>48.1</td></tr><tr><td>MMIN [16]</td><td>78.7</td><td>74.6</td><td>70.7</td><td>66.7</td><td>62.8</td><td>80.8</td><td>76.8</td><td>67.0</td><td>54.4</td></tr><tr><td>CTD-GNet student</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td><td>64.1</td></tr><tr><td>CTD-GNet dual mode</td><td>80.3</td><td>75.2</td><td>70.5</td><td>66.4</td><td>64.1</td><td>84.6</td><td>83.7</td><td>80.5</td><td>71.7</td></tr><tr><td>CTD-GNet teacher</td><td>84.2</td><td>82.4</td><td>79.3</td><td>74.1</td><td>66.1</td><td>84.6</td><td>83.7</td><td>80.5</td><td>71.7</td></tr><tr><td>MLP (eye only)</td><td>62.3</td><td>53.1</td><td>44.0</td><td>34.8</td><td>25.7</td><td>68.6</td><td>61.5</td><td>49.1</td><td>37.7</td></tr><tr><td rowspan="9"></td><td>Feature concatenation</td><td>65.7</td><td>57.9</td><td>50.2</td><td>42.5</td><td>34.8</td><td>71.3</td><td>65.8</td><td>54.6</td><td>41.9</td></tr><tr><td>DCCA-AM [6]</td><td>65.0</td><td>57.0</td><td>48.9</td><td>40.8</td><td>32.8</td><td>70.2</td><td>63.6</td><td>52.1</td><td>40.9</td></tr><tr><td>TFN [8]</td><td>66.7</td><td>58.7</td><td>50.7</td><td>42.7</td><td>34.7</td><td>71.7</td><td>65.1</td><td>52.8</td><td>40.6</td></tr><tr><td>LMF [9]</td><td>66.4</td><td>58.4</td><td>50.3</td><td>42.3</td><td>34.3</td><td>71.2</td><td>64.1</td><td>51.6</td><td>39.8</td></tr><tr><td>MISA [36]</td><td>62.4</td><td>53.2</td><td>44.0</td><td>34.7</td><td>25.5</td><td>67.3</td><td>59.2</td><td>47.2</td><td>36.9</td></tr><tr><td>Cross-attention [10]</td><td>60.5</td><td>51.4</td><td>42.3</td><td>33.1</td><td>24.0</td><td>66.3</td><td>59.4</td><td>48.3</td><td>37.9</td></tr><tr><td>Bimodal-LSTM [14]</td><td>74.3</td><td>70.4</td><td>63.2</td><td>52.1</td><td>37.2</td><td>75.1</td><td>73.0</td><td>67.8</td><td>58.8</td></tr><tr><td>Cross-attention + ModDrop [15]</td><td>63.8</td><td>57.4</td><td>51.1</td><td>44.7</td><td>38.3</td><td>67.1</td><td>60.3</td><td>49.0</td><td>38.3</td></tr><tr><td>MMIN [16]</td><td>68.7</td><td>62.2</td><td>55.7</td><td>49.1</td><td>42.6</td><td>72.0</td><td>64.7</td><td>52.0</td><td>39.5</td></tr><tr><td>CTD-GNet student</td><td>42.0</td><td>42.0</td><td>42.0</td><td>42.0</td><td>42.0</td><td>42.0</td><td></td><td>42.0 42.0</td><td>42.0</td></tr><tr><td>CTD-GNet dual mode</td><td>68.2</td><td>59.9</td><td>51.8</td><td>45.4</td><td>42.0</td><td>75.3</td><td></td><td>73.0 66.5</td><td>54.5</td></tr><tr><td>CTD-GNet teacher</td><td>74.5</td><td>70.9</td><td>65.1</td><td>56.2</td><td>43.8</td><td>75.3</td><td>72.9</td><td>66.5</td><td>54.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE 8

Agreement between the teacher graph ${ \bf A } _ { T }$ and the student graph $\mathbf { A } _ { S }$ on the test windows: mean row-wise KL divergence K $\mathsf { L } ( \mathbf { A } _ { T , i } \| \mathbf { A } _ { S , i } )$ and Jaccard overlap of the 10% strongest edges, mean ± standard deviation over the 36 (SEED) and 45 (SEED-IV) runs of all folds and seeds.
<table><tr><td rowspan="2">Distillation</td><td colspan="2">SEED</td><td colspan="2">SEED-IV</td></tr><tr><td>KL↓</td><td>Jaccard ↑</td><td>KL↓</td><td>Jaccard ↑</td></tr><tr><td>None</td><td> $7 . 5 5 \pm 1 . 4 2$ </td><td> $0 . 2 0 9 \pm 0 . 0 4 7$ </td><td> $7 . 9 0 \pm 1 . 4 7$ </td><td> $0 . 0 9 3 \pm 0 . 0 3 0$ </td></tr><tr><td>Bidirectional</td><td> ${ \bf 0 . 2 1 \pm 0 . 0 9 }$ </td><td> ${ \bf 0 . 4 1 4 \pm 0 . 1 3 1 }$ </td><td> ${ \bf 0 . 2 3 \pm 0 . 0 6 }$ </td><td> $\mathbf { 0 . 2 2 6 \pm 0 . 0 6 3 }$ </td></tr></table>

## 5.2 What bidirectional distillation provides

Learning using privileged information usually passes outputs or features from teacher to student [19]. CTD-GNet couples the two networks in both directions, at the predictions, the node relations and the adjacency matrix, and the coupling serves two purposes. It improves the teacher: onedirectional distillation leaves the teacher at the level without distillation, whereas the bidirectional form raises it by 1.2 points on SEED, and coupling the predictions alone gives most of this gain, in line with deep mutual learning [44], where networks trained together improve each other. The coupling of the graphs reduces the teacher–student graph divergence 35-fold, so that the EEG-only student, which runs without an eye tracker, approximates the graph of the multimodal teacher and both networks can be read on one graph. Coupling with an EEG-only student can thus improve the multimodal model itself, not only the deployable

TABLE 9  
Change of the attention received by each scalp region through the eye movements, $\begin{array} { r } { \frac { 1 0 0 } { N } \sum _ { i } \sum _ { j \in \mathcal { R } } \Delta A _ { i j } } \end{array}$ (points), for the teacher graph averaged over the test windows of each emotion. Frontal: FP, AF and F electrodes (14); central: FC and C (14); temporal: FT7, FT8, T7, T8, TP7 and TP8 (6); parietal: CP and P (16); occipital: PO, O and CB (12). Each row sums to zero.
<table><tr><td>Dataset</td><td>Emotion</td><td>Frontal</td><td>Central</td><td>Temporal</td><td>Parietal</td><td>Occipital</td></tr><tr><td rowspan="3">SEED</td><td>Negative</td><td>0.70</td><td>0.33</td><td>-2.05</td><td>0.48</td><td>0.54</td></tr><tr><td>Neutral</td><td>0.95</td><td>0.14</td><td>-2.43</td><td>0.54</td><td>0.80</td></tr><tr><td>Positive</td><td>0.49</td><td>0.14</td><td>-1.30</td><td>0.16</td><td>0.51</td></tr><tr><td rowspan="4">SEED-IV</td><td>Fear</td><td>0.38</td><td>-0.01</td><td>-0.55</td><td>-0.31</td><td>0.50</td></tr><tr><td>Happy</td><td>0.34</td><td>0.16</td><td>-0.24</td><td>-0.52</td><td>0.25</td></tr><tr><td>Neutral</td><td>0.58</td><td>0.00</td><td>-0.27</td><td>-0.27</td><td>-0.04</td></tr><tr><td>Sad</td><td>0.27</td><td>-0.16</td><td>-0.31</td><td>-0.14</td><td>0.34</td></tr></table>

one.

## 5.3 Implications

Emotion-aware systems for mental health, education, driving and virtual reality must work for new users and with imperfect sensors, yet the fusion baselines fall below EEGonly accuracy when the eye tracker fails. Consistency across subjects and graceful degradation, the two strengths of CTD-GNet, address exactly these conditions. The principle extends to any auxiliary signal that describes how the parts of a primary signal interact, such as facial video or peripheral physiology for EEG: Proposition 1 holds for every attention graph with additive cues, and bidirectional topology distillation aligns the graph of the multimodal model with that of a model of the primary signal alone.

![](images/e1a6594ff6902784231d7b877c10687885533ff4d7a60bf0027eae80e0a55462.jpg)  
Fig. 9. Learned graphs per emotion on SEED-IV. Rows: teacher topology $\mathbf { A } _ { T } ,$ , distilled student topology $\mathbf { A } _ { S } ,$ , student topology without distillation, and eye-movement attribution $\Delta \mathbf { A } = \mathbb { E } _ { t } [ \mathbf { A } _ { T } - \mathbf { A } _ { T } ^ { ( \mathbf { x } _ { e } = \mathbf { 0 } ) } ]$ , where the eye features of all windows are set to their calibrated mean of zero; red edges gain and blue edges lose weight through the eye movements. Each panel shows the 5% strongest edges averaged over the test windows of the class.

![](images/2006fa5045a99967d87e7f3bdd70b935497149839f7dc1d6c127686fdc188187.jpg)  
Fig. 10. Learned graphs per emotion on SEED, drawn as in Fig. 9.

## 5.4 Limitations and future work

The evaluation covers two film-clip datasets from the same laboratory, with the officially smoothed eye-movement features and a session-wise calibration that uses the unlabeled windows of the test session, which an online system would replace by a calibration recording or running statistics. On film clips unseen in training, all models that use eye movements lose 10 to 14 points, against 1.6 to 3.2 points for EEG alone, so part of the eye-movement accuracy is specific to the clips. The margins over Bimodal-LSTM and MMIN are not significant, the student gains little accuracy from distillation, and the frontal gain in Table 9 may partly reflect electro-oculographic (EOG) activity, for which we ran no artifact-removal control. Future work will extend CTD-GNet to further datasets with more stimuli per emotion, to recordings outside the laboratory, to raw gaze features and to EOG-corrected EEG, and will give the student a longer EEG context.

## 6 CONCLUSION

We presented CTD-GNet, in which EEG forms the nodes of an electrode graph and eye movements and temporal context modulate its edges, nodes and readout. Coupling the multimodal teacher with a single-window EEG student in both directions improves the teacher and aligns the two graphs 35-fold, so that the EEG-only student approximates the teacher graph. Against 17 baselines on SEED and SEED-IV, CTD-GNet achieves the highest mean accuracy, 84.91% and 76.17%, with the smallest spread across subjects among multimodal models and the highest worst-subject accuracy, 74.02% and 54.81%. It is the most accurate model at every rate of missing eye data on SEED-IV, keeps 66.1% on SEED without eye data, more than any EEG-only model, and is the most accurate on film clips unseen in training on SEED. Assigning each signal its physiological role thus yields multimodal recognizers that stay consistent across users and robust to sensor failure.

## REFERENCES

[1] S. M. Alarcao and M. J. Fonseca, “Emotions recognition using EEG ˜ signals: A survey,” IEEE Trans. Affect. Comput., vol. 10, no. 3, pp. 374–393, 2019.

[2] M. M. Bradley, L. Miccoli, M. A. Escrig, and P. J. Lang, “The pupil as a measure of emotional arousal and autonomic activation,” Psychophysiology, vol. 45, no. 4, pp. 602–607, 2008.

[3] W.-L. Zheng and B.-L. Lu, “Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks,” IEEE Trans. Auton. Ment. Develop., vol. 7, no. 3, pp. 162– 175, 2015.

[4] W.-L. Zheng, W. Liu, Y. Lu, B.-L. Lu, and A. Cichocki, “EmotionMeter: A multimodal framework for recognizing human emotions,” IEEE Trans. Cybern., vol. 49, no. 3, pp. 1110–1122, 2019.

[5] Y. Lu, W.-L. Zheng, B. Li, and B.-L. Lu, “Combining eye movements and EEG to enhance emotion recognition,” in Proc. 24th Int. Joint Conf. Artif. Intell., 2015, pp. 1170–1176.

[6] W. Liu, J.-L. Qiu, W.-L. Zheng, and B.-L. Lu, “Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition,” IEEE Trans. Cogn. Develop. Syst., vol. 14, no. 2, pp. 715–729, 2022.

[7] W. Liu, W.-L. Zheng, and B.-L. Lu, “Emotion recognition using multimodal deep learning,” in Proc. Int. Conf. Neural Inf. Process., 2016, pp. 521–529.

![](images/6283e32edf777270bb2ac07a054156d18a0caf597cd9e0fc0a630c05dec4d1d3.jpg)  
Fig. 11. Band-wise topology sensitivity ${ { I } _ { b , i } }$ of the teacher, defined in Eq. (20), for the $\alpha ,$ β and γ bands and each emotion, averaged over the test windows of all folds and seeds. The values at the 62 electrodes (dots) are interpolated with a thin-plate spline; nose up, left hemisphere on the left, one colour scale for both datasets.

[8] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L.-P. Morency, “Tensor fusion network for multimodal sentiment analysis,” in Proc. Conf. Empirical Methods Natural Lang. Process., 2017, pp. 1103– 1114.

[9] Z. Liu, Y. Shen, V. B. Lakshminarasimhan, P. P. Liang, A. B. Zadeh, and L.-P. Morency, “Efficient low-rank multimodal fusion with modality-specific factors,” in Proc. 56th Annu. Meeting Assoc. Comput. Linguistics, 2018, pp. 2247–2256.

[10] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics, 2019, pp. 6558–6569.

[11] M. Corbetta and G. L. Shulman, “Control of goal-directed and stimulus-driven attention in the brain,” Nat. Rev. Neurosci., vol. 3, no. 3, pp. 201–215, 2002.

[12] P. Kuppens and P. Verduyn, “Emotion dynamics,” Curr. Opin. Psychol., vol. 17, pp. 22–26, 2017.

[13] L.-C. Shi and B.-L. Lu, “Off-line and on-line vigilance estimation based on linear dynamical system and manifold learning,” in Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc., 2010, pp. 6587–6590.

[14] H. Tang, W. Liu, W.-L. Zheng, and B.-L. Lu, “Multimodal emotion recognition using deep neural networks,” in Proc. Int. Conf. Neural Inf. Process., 2017, pp. 811–819.

[15] N. Neverova, C. Wolf, G. Taylor, and F. Nebout, “ModDrop: Adaptive multi-modal gesture recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 38, no. 8, pp. 1692–1706, 2016.

[16] J. Zhao, R. Li, and Q. Jin, “Missing modality imagination network for emotion recognition with uncertain missing modalities,” in Proc. 59th Annu. Meeting Assoc. Comput. Linguistics and 11th Int. Joint Conf. Natural Lang. Process., 2021, pp. 2608–2618.

[17] J. Arevalo, T. Solorio, M. Montes-y Gomez, and F. A. Gonz´ alez,´ “Gated multimodal units for information fusion,” in Proc. Int. Conf. Learn. Represent. Workshop, 2017.

[18] V. Vapnik and A. Vashist, “A new learning paradigm: Learning using privileged information,” Neural Netw., vol. 22, no. 5–6, pp. 544–557, 2009.

[19] D. Lopez-Paz, L. Bottou, B. Scholkopf, and V. Vapnik, “Unifying¨ distillation and privileged information,” in Proc. Int. Conf. Learn. Represent., 2016.

[20] R.-N. Duan, J.-Y. Zhu, and B.-L. Lu, “Differential entropy feature for EEG-based emotion classification,” in Proc. 6th Int. IEEE/EMBS Conf. Neural Eng., 2013, pp. 81–84.

[21] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces,” J. Neural Eng., vol. 15, no. 5, p. 056013, 2018.

[22] Y. Li, W. Zheng, Y. Zong, Z. Cui, T. Zhang, and X. Zhou, “A bihemisphere domain adversarial neural network model for EEG emotion recognition,” IEEE Trans. Affect. Comput., vol. 12, no. 2, pp. 494–504, 2021.

[23] Y. Li, L. Wang, W. Zheng, Y. Zong, L. Qi, Z. Cui, T. Zhang, and T. Song, “A novel bi-hemispheric discrepancy model for EEG emotion recognition,” IEEE Trans. Cogn. Develop. Syst., vol. 13, no. 2, pp. 354–367, 2021.

[24] R. Zhou, Z. Zhang, H. Fu, L. Zhang, L. Li, G. Huang, F. Li, X. Yang, Y. Dong, Y.-T. Zhang, and Z. Liang, “PR-PL: A novel prototypical representation based pairwise learning framework for emotion recognition using EEG signals,” IEEE Trans. Affect. Comput., vol. 15, no. 2, pp. 657–670, 2024.

[25] T. Song, W. Zheng, P. Song, and Z. Cui, “EEG emotion recognition using dynamical graph convolutional neural networks,” IEEE Trans. Affect. Comput., vol. 11, no. 3, pp. 532–541, 2020.

[26] P. Zhong, D. Wang, and C. Miao, “EEG-based emotion recognition using regularized graph neural networks,” IEEE Trans. Affect. Comput., vol. 13, no. 3, pp. 1290–1301, 2022.

[27] T. Song, S. Liu, W. Zheng, Y. Zong, and Z. Cui, “Instance-adaptive graph for EEG emotion recognition,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 3, 2020, pp. 2701–2708.

[28] J. Li, S. Li, J. Pan, and F. Wang, “Cross-subject EEG emotion recognition with self-organized graph neural network,” Front. Neurosci., vol. 15, p. 611653, 2021.

[29] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li ´ o, and\` Y. Bengio, “Graph attention networks,” in Proc. Int. Conf. Learn. Represent., 2018.

[30] G. Andrew, R. Arora, J. Bilmes, and K. Livescu, “Deep canonical correlation analysis,” in Proc. 30th Int. Conf. Mach. Learn., 2013, pp. 1247–1255.

[31] B. Fu, W. Chu, C. Gu, and Y. Liu, “Cross-modal guiding neural

network for multimodal emotion recognition from EEG and eye movement signals,” IEEE J. Biomed. Health Inform., vol. 28, no. 10, pp. 5865–5876, 2024.

[32] M. Zhu, Q. Wu, Z. Bai, Y. Song, and Q. Gao, “EEG-eye movement based subject dependence, cross-subject, and cross-session emotion recognition with multidimensional homogeneous encoding space alignment,” Expert Syst. Appl., vol. 251, p. 124001, 2024.

[33] Q. Zhu, T. Zhu, L. Fei, C. Zheng, W. Shao, D. Zhang, and D. Zhang, “Multi-modal cross-subject emotion feature alignment and recognition with EEG and eye movements,” IEEE Trans. Affect. Comput., vol. 16, no. 3, pp. 2102–2115, 2025.

[34] C. Chen, Z. Li, K. I. Kou, J. Du, C. Li, H. Wang, and C.-M. Vong, “Comprehensive multisource learning network for cross-subject multimodal emotion recognition,” IEEE Trans. Emerg. Topics Comput. Intell., vol. 9, no. 1, pp. 365–380, 2025.

[35] M. Jimenez-Guarneros and G. Fuentes-Pineda, “CFDA-CSF: A´ multi-modal domain adaptation method for cross-subject emotion recognition,” IEEE Trans. Affect. Comput., vol. 15, no. 3, pp. 1502– 1513, 2024.

[36] D. Hazarika, R. Zimmermann, and S. Poria, “MISA: Modalityinvariant and -specific representations for multimodal sentiment analysis,” in Proc. 28th ACM Int. Conf. Multimedia, 2020, pp. 1122– 1131.

[37] K. Cho, B. van Merrienboer, C¸ . Gulc¸ehre, D. Bahdanau,¨ F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. Conf. Empirical Methods Natural Lang. Process., 2014, pp. 1724–1734.

[38] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv:1803.01271, 2018.

[39] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.

[40] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio, “FitNets: Hints for thin deep nets,” in Proc. Int. Conf. Learn. Represent., 2015.

[41] W. Park, D. Kim, Y. Lu, and M. Cho, “Relational knowledge distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 3967–3976.

[42] F. Tung and G. Mori, “Similarity-preserving knowledge distillation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2019, pp. 1365– 1374.

[43] Y. Yang, J. Qiu, M. Song, D. Tao, and X. Wang, “Distilling knowledge from graph convolutional networks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 7074–7083.

[44] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu, “Deep mutual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 4320–4328.

[45] S. Zhang, C. Tang, and C. Guan, “Visual-to-EEG cross-modal knowledge distillation for continuous emotion recognition,” Pattern Recognit., vol. 130, p. 108833, 2022.

[46] M. Ma, J. Ren, L. Zhao, S. Tulyakov, C. Wu, and X. Peng, “SMIL: Multimodal learning with severely missing modality,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 3, 2021, pp. 2302–2310.

[47] H. Wang, Y. Chen, C. Ma, J. Avery, L. Hull, and G. Carneiro, “Multi-modal learning with missing modality via shared-specific feature modelling,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 15 878–15 887.

[48] Y.-L. Lee, Y.-H. Tsai, W.-C. Chiu, and C.-Y. Lee, “Multimodal prompting with missing modalities for visual recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 14 943–14 952.

[49] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, “Visual prompt tuning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 709–727.

[50] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., 2018, pp. 3942–3951.

[51] R. J. Davidson, “Anterior cerebral asymmetry and the nature of emotion,” Brain Cogn., vol. 20, no. 1, pp. 125–151, 1992.

[52] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Represent., 2017.

[53] W. Wang, D. Tran, and M. Feiszli, “What makes training multimodal classification networks hard?” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 12 695–12 705.

[54] H. Liu, S. Yang, Y. Zhang, M. Wang, F. Gong, C. Xie, G. Liu, Z. Liu, Y.-J. Liu, B.-L. Lu, and D. Zhang, “LibEER: A comprehensive benchmark and algorithm library for EEG-based emotion recognition,” IEEE Trans. Affect. Comput., vol. 16, no. 4, pp. 3596– 3613, 2025.

[55] Z. Liu, Y. Chen, C. Xie, Y. Xie, and H. Liu, “LibEMER: A novel benchmark and algorithms library for EEG-based multimodal emotion recognition,” arXiv:2509.19330, 2025.

[56] S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.

[57] C. Cortes and V. Vapnik, “Support-vector networks,” Mach. Learn., vol. 20, no. 3, pp. 273–297, 1995.

[58] L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5– 32, 2001.

[59] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent., 2019.

[60] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond empirical risk minimization,” in Proc. Int. Conf. Learn. Represent., 2018.

[61] A. Tarvainen and H. Valpola, “Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,” in Proc. Adv. Neural Inf. Process. Syst., 2017, pp. 1195–1204.

[62] J. A. Kilgallen, B. A. Pearlmutter, and J. M. Siskind, “The repeatedstimulus confound in electroencephalography,” arXiv:2508.00531, 2025.