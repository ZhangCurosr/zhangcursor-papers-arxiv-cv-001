# Mask IPL: Noise-Free Intrinsic Position Learning via Computation Graph Clipping for Event-Based Spike-Driven Tracking

Yimeng Shan, Malu Zhang

Abstract—Spiking neural networks (SNNs) match the event-driven nature of event cameras and naturally extract spatiotemporal features These properties have given rise to a series of recent studies on event-based tracking with SNNs. Because Intrinsic Position Learning (IPL) acquires stronger position information without introducing any parameters, it has become one of the mainstream approaches to acquiring position information in event-based spike-driven tracking. However, the mechanism behind its effectiveness still lacks a systematic theoretical analysis. Moreover, our analysis shows that IPL introduces additional noise in both forward and backward propagation. The former increases the inference error, and the latter prevents the parameters from converging to a better solution. This paper presents a systematic analysis of IPL and reveals that its effectiveness stems from the synergy between IPL and multi-stage convolution. The zero blocks of the joint tensor are equivalent to zero padding for convolution, and the resulting boundary effect spreads layer by layer through the multi-stage convolution. Every parameter update is therefore driven by a gradient that perceives the relative displacement between the template and search frames. A positional encoding added after the convolutional stage cannot provide this information. We further propose a simple Computation Graph Clipping method. It applies a validity mask determined by the layout to the operations of every layer, so that the invalid regions become equivalent to zero padding in both forward and backward propagation. This eliminates the above noise without introducing any parameters and makes the actual gradient coincide with the ideal gradient. We name the improved method Mask IPL. Without increasing the number of parameters or the computational cost, Mask IPL improves the AUC of the Tiny-scale tracker on FE108, FELT, and VisEvent and consistently improves the Base-scale tracker as well.

Index Terms—Brain-inspired Computing, Spiking Neural Networks, Single Object Tracking, Neuromorphic Vision

## 1 INTRODUCTION

Event cameras capture visual information asynchronously and output sparse event streams, properties that match the spike-driven computation of spiking neural networks (SNNs). Combining SNNs with event cameras for low-power, high-temporal-resolution vision applications is therefore an important research direction [1]. Among these applications, event-based single object tracking (SOT), a fundamental vision task, has drawn wide attention [2, 3] because it requires modeling the temporal information in the event stream and extracting the target’s spatial features.

Early SNN-based trackers for event cameras usually contain floating-point multiplications and therefore cannot fully exploit the advantages of SNNs in event-driven computation and low power consumption [4, 5]. To the best of our knowledge, SDTrack [6], proposed by Shan et al., is the first event-based SOT framework built entirely from SNNs. Its results show that a network built entirely from SNNs reaches or even surpasses the tracking accuracy of conventional artificial neural networks (ANNs) while considerably reducing the inference energy. The success of SDTrack has made event-based tracking a common benchmark for evaluating SNN vision models and related techniques. Since then, many studies have explored network architectures [7], model compression [8], optimizers [9], plugand-play attention modules [10], and loss functions [11], and have validated their effectiveness on event-based tracking.

Many of these methods acquire position information with Intrinsic Position Learning (IPL) rather than with the explicit positional encoding of the conventional SOT pipeline [12, 13, 14]. Existing results show that IPL generally outperforms explicit positional encoding in tracking performance. However, the mechanism behind the effectiveness of IPL still lacks a systematic analysis. We further find that IPL introduces additional feature noise in forward propagation and additional gradient noise in backward propagation. The former increases the inference error, and the latter disturbs parameter optimization and prevents the model from converging to a better solution.

To address these issues, this paper presents a in-depth analysis of IPL. We find that the effectiveness of IPL stems from its synergy with multi-stage convolution. The zero blocks of the joint tensor are equivalent to zero padding for convolution, and the resulting boundary effect spreads layer by layer through the multi-stage convolution. Every parameter update is therefore driven by a gradient that perceives the relative displacement between the template and search frames. Such cross-frame relative position information is absent from a positional encoding added after the convolutional stage. This finding explains why IPL outperforms explicit positional encoding and offers a new perspective on IPL-based trackers. On this basis, we propose a simple Computation Graph Clipping method. It applies a validity mask determined by the layout to the operations of every layer, so that the invalid regions become equivalent to zero padding in both forward and backward propagation. This eliminates the noise without introducing any parameters and makes the actual gradient coincide with the ideal gradient. We name the improved method Mask IPL. Without introducing any additional parameters or computational cost, Mask IPL improves the AUC of the Tiny-scale tracker on FE108, FELT, and VisEvent and also yields consistent improvements on the Base-scale tracker. These results indicate that efficiently modeling the relative position between the template and search frames is likely one of the key factors for improving event-based SOT. They also point to a new direction for the design of IPL-based trackers.

## 2 RELATED WORK

Early attempts to introduce SNNs into event-based trackers were built on Siamese architectures. Zhang et al. proposed STNet [4], which is based on a dynamic threshold strategy for spiking neurons, and SNNTrack [5], which equips the neurons with a dynamic decay factor. Neither tracker is built entirely from SNNs, and both retain a large number of floating-point multiplications, which prevents their deployment on neuromorphic chips for edge inference.

After SDTrack [6] was proposed, spike-driven trackers built entirely from SNNs became a major line of research. Zhou et al. provided a new optimization scheme named AdaS [9], which helps the training of the tracker converge to the global optimum. Wang et al. proposed a new tracker based on bipolar self-attention [7], which improves the tracking performance considerably. TP-Spikformer [8] obtains a lightweight event-based spike-driven tracker through dynamic computational sparsification and effectively reduces the inference cost. SpikeFET [11] proposes a spatiotemporal regularization (STR) strategy that repairs the similarity degradation of asymmetric features across time steps.

Overall, most recent studies on event-based spike-driven tracking acquire position information with IPL or adopt similar strategies inspired by it. However, why IPL is effective has not been analyzed, and the noise that it introduces in forward and backward propagation has gone unnoticed. Both issues limit the further adoption of IPL and the performance it can deliver. This work provides a mechanistic analysis of IPL and removes this noise through Computation Graph Clipping, which further improves IPL-based trackers without introducing any parameters.

## 3 METHOD

## 3.1 Preliminary

## 3.1.1 Problem Definition

A standard single object tracker takes one template frame and one search frame as input. Both are obtained by aggregating the event stream over a time interval with an event aggregation method. We denote the template frame by $\mathbf { Z } \in \mathbb { R } ^ { C \times H _ { Z } \times W _ { Z } }$ and the search frame by $\mathbf { X } \in \mathbb { R } ^ { C \times H _ { X } \times W _ { X } }$ Given $\mathbf { Z } ,$ which is cropped around the target, the tracker locates the target in X and estimates its size. That is, it predicts the bounding box $\boldsymbol { \hat { P } } = \left( \boldsymbol { \hat { c } } , \boldsymbol { \hat { s } } \right)$ of the target in the coordinate frame of $\mathbf { \bar { X } } ,$ , where $\hat { c } \in \mathbb { R } ^ { 2 }$ is the center and $\hat { s } \in \mathbb R ^ { 2 }$ contains the width and height. The ground-truth box is denoted by $P ^ { * } = ( c ^ { * } , s ^ { * } )$ . The parameters of the tracker are optimized by minimizing the standard tracking loss

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { i o u } } \mathcal { L } _ { \mathrm { i o u } } + \lambda _ { \mathcal { L } 1 } \mathcal { L } _ { 1 } ,\tag{1}
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ is the weighted focal loss applied to the center heatmap, $\mathcal { L } _ { \mathrm { i o u } }$ and $\mathcal { L } _ { 1 }$ are the generalized IoU loss and the $\ell _ { 1 }$ loss applied to the predicted box, respectively, and $\lambda _ { \mathrm { i o u } }$ and $\lambda _ { \mathcal { L } 1 }$ are fixed weights.

## 3.1.2 Spiking Neuron

The nervous system of the brain is the source of inspiration for neural networks. Its diverse neural dynamics have led researchers to design various types of spiking neurons for SNNs. Our method is general and does not rely on any specific neuron type. We therefore describe spiking neurons with a unified set of dynamical equations:

$$
\mathbf { U } [ t ] = \mathbf { H } [ t - 1 ] + \frac { 1 } { \tau } ( \mathbf { I } [ t ] - ( \mathbf { H } [ t - 1 ] - \mathbf { U } _ { r e s t } ) ) ,
$$

$$
\mathbf { S } [ t ] = f ( \mathbf { U } [ t ] - \mathbf { U } _ { t h r } ) ,\tag{2}
$$

$$
\mathbf { H } [ t ] = \mathbf { U } [ t ] ( 1 - \mathbf { S } [ t ] ) .\tag{3}
$$

(4)

When a neuron receives the input I[t] at time $t ,$ its membrane potential is updated to U[t]. Eq. (3) then determines the spike value at time $t ,$ where $\mathbf { S } [ t ] = 0$ indicates that no spike is emitted and $\mathbf { U } _ { t h r }$ is the threshold. H[t] denotes the membrane potential after the spike generation of Eq. (3). τ is the decay factor of the membrane potential, which simulates the decay of the stimulus when the neuron receives no stimulation for an extended period. $\mathbf { U } _ { r e s t }$ denotes the resting potential toward which the membrane potential decays in Eq. (2). After a spike is emitted, Eq. (4) resets the membrane potential to zero.

Mainstream spiking neuron models can be represented by Eqs. (2)–(4). For instance, when $f ( \cdot )$ is the Heaviside step function, the model represents an IF neuron [15] if τ equals 1 and an LIF neuron [16] if τ exceeds 1. When $\mathbf { U } _ { r e s t } \ =$ $\mathbf { U } _ { t h r } = 0$ and $\textstyle f ( x ) = { \frac { 1 } { D } }$ · Clip(round(x), 0, D), the model becomes the I-LIF neuron [17], where $x = \mathbf { U } [ t ]$ and D is the number of virtual time steps. Clip(x, min, max) constrains x to [min, max], and round(·) rounds to the nearest integer. During inference, this model can be converted to binary (0/1) spikes with the spike-ahead principle. In this case, $\dot { D }$ is folded into the actual number of iterative time steps $T ,$ giving $T \times D$ time steps in total.

These neurons therefore achieve spike-driven inference, replacing multiply-accumulate (MAC) operations with accumulate (AC) operations and thereby reducing the computational cost substantially. In the following, $\mathcal { S } \tilde { \mathcal { N } } ( \cdot )$ denotes a layer of spiking neurons governed by Eqs. (2)–(4), which maps its input current to the spike output S[t] position by position. Every convolution and linear projection in the tracker receives the spikes emitted by such a layer. Since all operations other than the neuronal dynamics in Eqs. (2)–(4) are identical across time steps, the time index t is omitted in the following sections.

## 3.1.3 Hierarchical Tracker

Siamese trackers extract features from X and Z with a shared network and model their relation in a separate module. OSTrack [13] and SimTrack [14] show that a single Vision

Transformer can perform feature extraction and relation modeling jointly, so that a convolutional front end is not necessary. Hierarchical architectures such as Swin Transformer [18] and MetaFormer [19] nevertheless retain a multistage convolutional front end, and their advantages have motivated hierarchical trackers. SDTrack is the representative MetaFormer-style tracker for event-based spike-driven track ing. Its backbone consists of two stages: a convolutional stage $\mathcal { F } _ { \theta } ^ { \mathrm { c o n v } }$ with M layers and a transformer stage $\mathcal { F } _ { \theta } ^ { \mathrm { a t t n } }$ built from spiking self-attention (SSA) blocks. The backbone is followed by a tracking head $g ( \cdot )$ . Layer m of the convolutional stage has kernel size $k _ { m }$ and stride $s _ { m }$ . We write $\begin{array} { r } { j _ { m } = \prod _ { l < m } s _ { l } } \end{array}$ for the cumulative stride at its output, so that an input of size $H \times W$ yields a feature map of size $( H / j _ { m } ) \times ( \hat { W } / j _ { m } )$ at layer m. The total stride of the convolutional stage is $j _ { M }$

## 3.1.4 Conventional Design and IPL

In the conventional pipeline, X and Z pass through the convolutional stage separately, with shared or separate weights, are tokenized, receive a positional encoding, and are concatenated before entering the transformer stage. IPL replaces this procedure with a single joint input. The two frames are placed on the diagonal of one tensor, referred to as the joint tensor $\mathbf { U } \in \mathbb { R } ^ { C \times \mathbf { \breve { H } } _ { U } \times W _ { U } }$ with $H _ { U } = H _ { X } + H _ { Z }$ and $\dot { W _ { U } } = W _ { X } + W _ { Z } ,$ , which is constructed as

$$
\mathbf { U } = \mathrm { I P L } ( \mathbf { X } , \mathbf { Z } ) ,\tag{5}
$$

$$
\mathrm { I P L } ( \mathbf { X } , \mathbf { Z } ) = \left[ \begin{array} { c c } { \mathbf { X } } & { \boldsymbol { \mathrm { O _ { 1 } } } } \\ { \boldsymbol { \mathrm { O _ { 2 } } } } & { \mathbf { Z } } \end{array} \right] ,\tag{6}
$$

where ${ \mathrm { O } _ { 1 } } ~ \in ~ \mathbb { R } ^ { C \times H _ { X } \times W _ { Z } }$ and $\mathrm { O _ { 2 } } ~ \in ~ \mathbb { R } ^ { C \times H _ { Z } \times W _ { X } }$ are zero blocks. Let $\Omega _ { X }$ and $\Omega _ { Z }$ denote the sets of positions of U occupied by the search frame and the template frame, respectively. We refer to $\Omega ^ { + } = \Omega _ { X } \cup \Omega _ { Z }$ as the valid regions, which hold observations. Conversely, the positions of the two zero blocks form the invalid regions $\Omega ^ { 0 }$ , which hold no observations and exist only to build U.

The joint tensor is processed by the convolutional stage as a single input, which lets the network handle the concatenated frames in the same way as a single-stream input. At the output of the convolutional stage, the two regions are restored from their known locations, tokenized separately, and concatenated before entering the transformer stage. No positional encoding is added at any point, and IPL introduces no parameters. Ideally, the invalid regions would remain inactive throughout the network, so that they add neither information nor computation. Since a zero input produces no spikes, this holds at the input of the network. However, Sec. 3.3 shows that it does not hold in the subsequent layers.

Extensive experiments have shown that IPL outperforms explicit positional encoding [6], but the underlying reason has not been revealed, which limits further research on IPLbased trackers. The next section provides this analysis.

## 3.2 Mechanistic Analysis of IPL

The convolutional stage $\mathcal { F } _ { \theta } ^ { \mathrm { c o n v } } ( \cdot )$ of SDTrack extracts features from the joint tensor U obtained by IPL. The template and search frames are then restored to their original spatial structure, tokenized separately, and concatenated. The transformer stage $\mathcal { F } _ { \theta } ^ { \mathrm { a t t n } } ( \cdot )$ , which contains spiking self-attention, models the relation between the template and search features. The tracking head $g ( \cdot )$ then outputs the predicted position and size of the target. This process can be formalized as the differentiable computation graph

$$
\hat { P } = g ( \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } ) ) ,\tag{7}
$$

where $\mathcal { F } _ { \theta }$ denotes the backbone, i.e., the convolutional stage ${ \mathcal { F } } _ { \theta } ^ { \mathrm { c o n v } }$ followed by the restoration and tokenization step and the transformer stage $\mathcal { F } _ { \theta } ^ { \mathrm { a t t n } }$ , and θ collects all learnable parameters. The graph is optimized with the standard SOT loss in Eq. (1). In the pair-matching training paradigm of SOT, the template frame is cropped around the target center, so that its center coordinate $P _ { Z }$ corresponds to the position of the target in the template frame. The search frame is cropped from a larger region in which the target may appear, and the ground-truth box $\boldsymbol { P ^ { * } } = \left( c ^ { * } , s ^ { * } \right)$ is defined in the coordinate frame of the search frame. Expressing $P _ { Z }$ in the same coordinate frame, the center of the ground-truth box can be decomposed as

$$
c ^ { * } = P _ { Z } + \Delta P ,\tag{8}
$$

where $\Delta P$ is the spatial displacement from the center of the template frame to the true position of the target in the search frame. Since the template frame is always cropped around the target, $P _ { Z }$ is a fixed constant, namely the geometric center of the template frame. The variation of $c ^ { * }$ is therefore determined entirely by the relative displacement $\Delta P .$ . This implies that every term in $\operatorname { E q . } \left( 1 \right)$ is directly governed by $\Delta P ,$ Consequently, the loss L in Eq. (1) is essentially a function that is highly sensitive to the relative position between the template and search frames.

With a gradient-based optimizer, the update direction of θ is determined by $\nabla _ { \boldsymbol { \theta } } \mathcal { L } .$ . By the chain rule, the gradient expands as

$$
\frac { \partial \mathcal { L } } { \partial \boldsymbol { \theta } } = \frac { \partial \mathcal { L } } { \partial \hat { P } } \cdot \frac { \partial \hat { P } } { \partial \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } ) } \cdot \frac { \partial \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } ) } { \partial \boldsymbol { \theta } } .\tag{9}
$$

The first factor encodes the deviation between the predicted and the true position, which, according to Eq. (8), is determined by $\Delta \bar { P }$

We now examine the structure of the last factor. For a position whose convolutional receptive field overlaps a padded region, the response depends on where that position lies relative to the padding. This boundary effect is the source of the absolute position information encoded by convolutional networks [20, 21]. Owing to successive convolutions, the fraction of a feature map that carries such information grows with the depth of the network.

When U is processed, the receptive field of a convolutional kernel inevitably covers pixels of the template frame, pixels of the search frame, and the zero-padded regions at their boundaries at the same time. Through the convolution operations, this spatial layout causes the relative position information between the two frames to spread through the network layer by layer. Let $\mathcal { N } _ { k _ { m } }$ denote the set of kernel offsets of layer m, ${ \bf { W } } _ { m } [ { \boldsymbol { o } } ]$ the kernel weight at offset $o \in \mathcal { N } _ { k _ { m } } ,$ and $\mathcal { R } _ { m } ( p )$ the receptive field on U of output position p of layer m. The convolutional stage follows the recursion

$$
\mathbf { Y } _ { m } ( p ) = S \mathcal { N } \Big ( \sum _ { o \in \mathcal { N } _ { k _ { m } } } \mathbf { W } _ { m } [ o ] \mathbf { Y } _ { m - 1 } ( p + o ) \Big ) ,\tag{10}
$$

for $m = 1 , \ldots , M ,$ with $\mathbf { Y } _ { 0 } = \mathbf { U }$ and batch normalization omitted for brevity. (1) In the first layer, when the kernel moves to within $\dot { \lfloor k _ { 1 } / 2 \rfloor }$ pixels of the boundary between a region and a zero block, $\mathcal { R } _ { 1 } ( p )$ covers valid pixels of that region together with pixels of $\mathrm { O _ { 1 } \ a n d / o r { O _ { 2 } } } .$ . Near the corner at which the two regions meet, it covers pixels of both regions. The output features at these boundary positions implicitly encode the relative spatial layout of the two regions. (2) In the second layer, most kernel positions do not touch both regions at the same time, but their receptive fields cover the boundary features of the first layer. Hence, whenever some $p + o$ is a boundary position of the first layer, ${ \bf Y } _ { 2 } ( p )$ indirectly acquires position information. (3) After M layers, the receptive-field radius in the coordinates of U is

$$
r _ { m } = \sum _ { l = 1 } ^ { m } \Big \lfloor \frac { k _ { l } - 1 } { 2 } \Big \rfloor j _ { l - 1 } , \qquad j _ { 0 } = 1 ,\tag{11}
$$

which reduces to the sum of the kernel half-widths when all strides equal one; each downsampling layer multiplies the growth rate of the receptive field in all subsequent layers by its stride. By translation equivariance, the feature at position p of layer m can depend on the position of $p$ itself only if $\mathcal { R } _ { m } ( p )$ contains pixels of a zero block or of the outer padding of the convolution. Otherwise, it depends only on the pixel values within the receptive field. When $r _ { M }$ is comparable to the side lengths of the regions, as is the case for the multi-stage front end of SDTrack, this condition is satisfied throughout the template region and the search region. Even positions inside a region therefore perceive the other region indirectly. Consequently, every position of $\mathbf { Y } _ { M }$ implicitly encodes the relative spatial position of the two regions, which we write as

$$
{ \displaystyle { \bf Y } _ { M } = { \bf Y } _ { M } ( { \bf X } , { \bf Z } , \xi ) , }\tag{12}
$$

where ξ denotes the layout of X and Z in U.

It follows that $\partial \mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { U } ) / \partial \boldsymbol { \theta }$ cannot be decomposed into independent terms that depend only on X or only on $\mathbf { Z } ;$ it couples the spatial layouts of the two frames. The first factor of $\operatorname { E q . }$ . (9) encodes the prediction error determined by $\Delta P _ { \cdot }$ the last factor couples the layouts of the two frames through layer-wise propagation, and the chain rule multiplies the two. Therefore, every parameter update is driven by a gradient signal that perceives the relative displacement and the spatial layout at the same time.

## 3.3 Noise in the IPL Computation Graph

In $\mathrm { { I P L } , \mathrm { { O _ { 1 } } } }$ and $\mathrm { O _ { 2 } }$ are invalid regions introduced to construct the joint tensor U, yet the tracker processes them in the same way as the valid regions. This section examines how this computation introduces noise in convolution, batch normalization, the neuron layer, and attention, and how the noise affects inference and optimization.

Convolution. Eq. (10) treats all positions of the joint tensor equally. Let ${ \bf A } _ { m } ( { q } )$ denote the convolutional sum inside the parentheses of Eq. (10), i.e., the output of the convolution of layer m at position q before normalization.

Let $\Omega _ { m } ^ { 0 }$ and $\Omega _ { m } ^ { + }$ denote the counterparts of $\Omega ^ { 0 }$ and $\Omega ^ { + }$ at the resolution of layer m.

In the first layer, the input U is exactly zero on the invalid regions. Hence, for a position $q \in \Omega _ { 1 } ^ { + }$ at the edge of a valid region, the terms of Eq. (10) with $q + o \in \Omega ^ { 0 }$ vanish, and the convolution behaves at this edge exactly as it does at the outer zero padding. This is the boundary effect on which Sec. 3.2 relies. For a position $q \in \Omega _ { 1 } ^ { 0 }$ in an invalid region, $\mathbf { A } _ { 1 } ( q )$ is either zero or depends only on the valid content inside $\mathcal { R } _ { 1 } ( q )$ . Nevertheless, the shift $\beta _ { 1 }$ of batch normalization and the subsequent neuron layer cause these positions, which carry no valid information, to emit spikes. The next layer cannot distinguish these spikes from those produced by events. From the second layer on, the inputs on the invalid regions are no longer zero, which has two consequences. First, ${ \bf A } _ { m } ( \ v q )$ is in general nonzero for $q \in \Omega _ { m } ^ { 0 }$ Second, the positions at the edge of a valid region no longer receive a fixed zero boundary but rather a response to the neighboring content reflected through the invalid region. The reference against which position is measured is therefore no longer constant.

The invalid regions also take part in backpropagation. Let $\delta _ { m } ( q ) = \partial \mathcal { L } / \partial \mathbf { A } _ { m } ^ { - } ( q )$ denote the gradient backpropagated to position q. The weight gradient is

$$
\frac { \partial \mathcal { L } } { \partial \mathbf { W } _ { m } [ o ] } = \sum _ { q } \delta _ { m } ( q ) \mathbf { Y } _ { m - 1 } ( q + o ) ,\tag{13}
$$

where $o \in \mathcal { N } _ { k _ { m } }$ is the kernel offset in Eq. (10) and ${ \bf Y } _ { m - 1 } ( q +$ $o )$ is the input to the convolution of layer m that contributes to the output at q. Eq. (13) sums over all output positions q and all input positions $q + o ,$ including terms in which q or $q + o$ lies in an invalid region. If the invalid regions did not take part in the computation, the sum would retain only the terms with $q \in \Omega _ { m } ^ { + }$ and $q + o \in \Omega _ { m - 1 } ^ { + }$ . To express this constraint, let $\mathcal { M } _ { m }$ be the binary validity mask of layer $m ,$ which equals 1 on $\Omega _ { m } ^ { + }$ and 0 on $\Omega _ { m } ^ { 0 }$ . The ideal gradient is then $\begin{array} { r l } { \phantom { } } & { { } \sum _ { q } \mathbf { \hat { M } } _ { m } ( q ) \mathbf { \mathcal { M } } _ { m - 1 } ( q + o ) \delta _ { m } ^ { + } ( q ) \mathbf { \hat { Y } } _ { m - 1 } ( q + o ) } \end{array}$ , where $\delta _ { m } ^ { + }$ is the backpropagated gradient when the invalid regions do not take part in the computation. Their difference is

$$
\sum _ { q } \Big [ \delta _ { m } ( q ) - \mathcal { M } _ { m } ( q ) \mathcal { M } _ { m - 1 } ( q + o ) \delta _ { m } ^ { + } ( q ) \Big ] \mathbf { Y } _ { m - 1 } ( q + o ) .\tag{14}
$$

For the terms in which q or $q + o$ lies in an invalid region, the bracket equals $\delta _ { m } ( q )$ , so the whole term is spurious. Among these terms, those with q in a valid region and $q + o$ in an invalid region correspond to the edge positions that receive input from an invalid region in the forward pass. Those with q in an invalid region have nonzero $\delta _ { m } ( q )$ because the leaked activations of the invalid regions feed the valid regions of later layers. For the terms in which both q and $q + o$ lie in valid regions, the bracket equals $\delta _ { m } ( q ) - \delta _ { m } ^ { + } ( q )$ , i.e., the change in the backpropagated gradient itself caused by the contaminated forward pass.

Batch normalization. The statistics of batch normalization are estimated over all positions. Let $\mu _ { m }$ and $\mu _ { m } ^ { + }$ denote the means of $\mathbf { A } _ { m }$ over $\Omega _ { m } ^ { + } \cup \Omega _ { m } ^ { 0 }$ and over $\Omega _ { m } ^ { + }$ , respectively, and let $\mu _ { m } ^ { 0 }$ denote its mean over $\Omega _ { m } ^ { 0 }$ . Then

$$
\mu _ { m } - \mu _ { m } ^ { + } = \frac { \left| \Omega _ { m } ^ { 0 } \right| } { \left| \Omega _ { m } ^ { + } \right| + \left| \Omega _ { m } ^ { 0 } \right| } ( \mu _ { m } ^ { 0 } - \mu _ { m } ^ { + } ) ,\tag{15}
$$

and an analogous discrepancy holds for the variance. This effect is global: every valid position, including those whose receptive field contains only valid features, is normalized with statistics that depend on the content of the invalid regions. Since the proportion of invalid positions is fixed by the layout and their content is a deterministic function of the input, the perturbation is systematic and persists at inference through the running statistics. In backpropagation, the gradient of batch normalization contains terms averaged over all positions of the layer, so the gradient $\delta _ { m } ( q )$ at a valid position is mixed with contributions from invalid positions, which is one source of the difference $\delta _ { m } ( q ) - \delta _ { m } ^ { + } ( q )$ in Eq. (14). The invalid positions also obtain nonzero gradients in this way and enter the shared weights through Eq. (13).

Spiking neuron. The perturbed normalized values form the input current of the next neuron layer. According to Eqs. (2)–(4), a small shift of the membrane potential near the threshold changes the emitted spike and, during training, changes the interval in which the surrogate derivative is nonzero. The invalid regions therefore induce continuous perturbations before spiking and discrete changes of the spike pattern after spiking.

Attention. The valid regions of the output of the convolutional stage are restored and converted into $N _ { X } + N _ { Z }$ tokens, where $\begin{array} { l c l } { N _ { X } } & { = } & { ( H _ { X } / j _ { M } ) ( W _ { X } / j _ { M } ) } \end{array}$ and $\begin{array} { r l } { N _ { Z } } & { { } = } \end{array}$ $( H _ { Z } / j _ { M } ) ( W _ { Z } / j _ { M } )$ . All of these tokens are valid, but their features already carry the perturbations introduced by convolution, normalization, and the neuron layers described above. Let ${ \bf q } _ { a } , { \bf k } _ { b } ,$ , and $\mathbf { v } _ { b }$ denote the query, key, and value vectors of tokens a and b obtained by projecting the token features, and let ${ \bf y } _ { a }$ denote the output token. The spiking self-attention

$$
{ \bf Y } ^ { \mathrm { a t t n } } = ( Q K ^ { \mathsf { T } } ) V , \qquad { \bf y } _ { a } = \sum _ { b } ( { \bf q } _ { a } \cdot { \bf k } _ { b } ) { \bf v } _ { b } ,\tag{16}
$$

expresses each output token as a sum of the values of all tokens weighted by similarity. Consequently, the perturbation carried by any single token becomes a global perturbation after SSA, and the relation modeling between the template and search features deviates accordingly. In backpropagation, the gradient of the value is $\begin{array} { r } { \partial \mathcal { L } / \partial \mathbf { v } _ { b } = \sum _ { a } ( \mathbf { q } _ { a } \cdot \mathbf { \hat { k } } _ { b } ) \mathbf { \hat { \phi } } \partial \bar { \mathcal { L } } / \partial \mathbf { y } _ { a } , } \end{array}$ and the gradients of the key and query have the same structure. That is, the gradient of each token is a sum of the output gradients of all tokens weighted by the perturbed similarities. The perturbation therefore affects not only the output of attention but also, in the same way, the gradient propagated back to the convolutional stage.

Effect on inference and optimization. At inference, the features of the valid positions depend on the content of the invalid regions through Eqs. (10), (15), and (16), which increases the prediction error. During training, the gradient used for the update is

$$
g _ { n } = g _ { n } ^ { + } + \zeta _ { n } \implies \langle g _ { n } , g _ { n } ^ { + } \rangle = \| g _ { n } ^ { + } \| _ { 2 } ^ { 2 } + \langle \zeta _ { n } , g _ { n } ^ { + } \rangle ,\tag{17}
$$

where $g _ { n } ^ { + }$ is the ideal gradient at iteration $n ,$ i.e., the gradient when the invalid regions do not take part in the computation. The term $\zeta _ { n }$ collects the differences of Eq. (14) over all layers together with the backward perturbations introduced by nor malization and attention. The right-hand side follows from taking the inner product of both sides of the first equality with ${ \bar { g } } _ { n } ^ { + }$ . The perturbation is not zero-mean, because its sign and magnitude are determined by the layout and the input.

It therefore does not cancel by averaging over iterations, and any gradient statistics that the optimizer accumulates from the contaminated $g _ { n }$ inherit it. The second term on the righthand side of Eq. (17) measures how the perturbation changes the alignment between the actual gradient and the intended descent direction. Its sign is not fixed a priori. Whenever it is negative, the alignment is reduced. ${ \bar { \operatorname { I f } } } ,$ in addition, its magnitude exceeds $\begin{array} { r } { \| g _ { n } ^ { + } \| _ { 2 } ^ { 2 } . } \end{array}$ , i.e., $\langle \zeta _ { n } , g _ { n } ^ { + } \rangle \ : < \ : - \| g _ { n } ^ { + } \| _ { 2 } ^ { 2 }$ , then $\langle g _ { n } , g _ { n } ^ { + } \rangle < 0$ and the update direction is even reversed at that iteration. Since the perturbation does not average out, the optimization path deviates systematically from the intended descent direction, and the parameters converge to an inferior solution. This shows that the original IPL, while providing relative position information, also introduces additional noise that affects both inference and optimization.

## 3.4 Mask IPL: Computation Graph Clipping

The noise originates from the participation of the invalid regions in the computation. As long as the invalid regions are equivalent to the outer zero padding, i.e., their outputs remain zero and they contribute nothing to the gradient, the noise no longer arises. Mask IPL therefore applies the validity mask $\mathcal { M } _ { m }$ to the operations of every layer and clips the computation graph to the subgraph formed by the valid regions. According to the layout in Eq. $( 6 ) , \mathcal { M } _ { m }$ equals 1 on the $( H _ { X } / j _ { m } ) \times ( \breve { W } _ { X } / j _ { m } )$ search region at the top left and on the $( H _ { Z } / j _ { m } ) \times ( W _ { Z } / j _ { m } )$ template region at the bottom right of the $( H _ { U } / j _ { m } ) \times ( W _ { U } / j _ { m } )$ feature map of layer m, and 0 elsewhere. It is determined solely by the layout and the cumulative stride and contains no learnable parameters.

Specifically, when the tracker adopts Mask IPL, the m-th Conv-BN-SN block computes

$$
\widetilde { \mathbf { Y } } _ { m } = S \mathcal { N } \big ( \mathrm { B N } _ { \mathcal { M } } ( \mathcal { M } _ { m } \odot \mathbf { A } _ { m } ) \big ) ,\tag{18}
$$

where $\odot$ denotes element-wise multiplication broadcast over channels, and $\mathrm { B N } _ { \mathcal { M } }$ normalizes with the statistics of the valid regions only and outputs zero on $\Omega _ { m } ^ { 0 }$ . Its mean is

$$
\widetilde { \mu } _ { m } = \frac { 1 } { | \Omega _ { m } ^ { + } | } \sum _ { q \in \Omega _ { m } ^ { + } } \mathbf { A } _ { m } ( q ) ,\tag{19}
$$

its variance is defined analogously, and the running statistics are accumulated from both. Since $\mathrm { B N } _ { \mathcal { M } }$ outputs zero on $\Omega _ { m } ^ { 0 }$ and a neuron that receives zero input emits no spike, the output of Eq. (18) is zero on the invalid regions. For a block with a residual connection, the quantity added to the output of Eq. (18) is the input $\widetilde { \mathbf { Y } } _ { m - 1 }$ of that block. The sum remains zero on the invalid regions as long as this input is zero there. By induction from $\tilde { \mathbf { Y } } _ { 0 } = \mathbf { U } , ( 1 - \mathcal { M } _ { m } ) \odot$ $\widetilde { \mathbf { Y } } _ { m } = 0$ therefore holds for every layer. The input of every convolution is thus exactly zero on the invalid regions, as it is on the outer zero padding. The boundary mechanism of Sec. 3.2 is therefore preserved, whereas the perturbations produced by the invalid regions in Sec. 3.3 no longer arise. In backpropagation, since $\begin{array} { r }  \bar { \partial ( \mathcal { M } _ { m } \odot \mathbf { A } _ { m } ) / \partial \mathbf { A } _ { m } = \bar { \mathcal { M } } _ { m } , } \end{array}$ the positions in the invalid regions receive no gradient, and since the features on the invalid regions are zero, the weight

<table><tr><td rowspan="2">Methods*</td><td rowspan="2">Param. (M)</td><td rowspan="2">Spiking Neuron</td><td rowspan="2">Timesteps  $( T \times { \bar { D ) } }$ </td><td colspan="2">FE108</td><td colspan="2">FELT</td><td colspan="2">VisEvent</td></tr><tr><td>AUC(%)</td><td>PR(%)</td><td>AUC(%)</td><td>PR(%)</td><td>AUC(%)</td><td>PR(%)</td></tr><tr><td>STARK [22]</td><td>28.23</td><td></td><td>1 × 1</td><td>57.4</td><td>89.2</td><td>39.3</td><td>50.8</td><td>34.1</td><td>46.8</td></tr><tr><td>SimTrack [14]</td><td>88.64</td><td></td><td>1 × 1</td><td>56.7</td><td>88.3</td><td>36.8</td><td>47.0</td><td>34.6</td><td>47.6</td></tr><tr><td>OSTrack256 [13]</td><td>92.52</td><td></td><td>1 × 1</td><td>54.6</td><td>87.1</td><td>35.9</td><td>45.5</td><td>32.7</td><td>46.4</td></tr><tr><td>ARTrack256 [23]</td><td>202.56</td><td></td><td>1 × 1</td><td>56.6</td><td>88.5</td><td>39.5</td><td>49.4</td><td>33.0</td><td>43.8</td></tr><tr><td>SeqTrack-B256 [24]</td><td>90.60</td><td></td><td>1 × 1</td><td>53.5</td><td>85.5</td><td>33.0</td><td>42.0</td><td>28.6</td><td>43.3</td></tr><tr><td>HiT-B [25]</td><td>42.22</td><td></td><td>1× 1</td><td>55.9</td><td>88.5</td><td>38.5</td><td>48.9</td><td>34.6</td><td>47.6</td></tr><tr><td>GRM [26]</td><td>99.83</td><td></td><td>1× 1</td><td>56.8</td><td>89.3</td><td>37.2</td><td>47.4</td><td>33.4</td><td>47.7</td></tr><tr><td>HIPTrack [27]</td><td>120.41</td><td></td><td>1 × 1</td><td>50.8</td><td>81.0</td><td>38.2</td><td>48.9</td><td>32.1</td><td>45.2</td></tr><tr><td>ODTrack [28]</td><td>92.83</td><td></td><td>1 × 1</td><td>43.2</td><td>69.7</td><td>29.7</td><td>35.9</td><td>24.7</td><td>34.7</td></tr><tr><td>SiamRPN [29]</td><td></td><td></td><td>1 × 1</td><td></td><td></td><td></td><td></td><td>24.7</td><td>38.4</td></tr><tr><td>ATOM [30]</td><td>一</td><td></td><td>1 × 1</td><td></td><td></td><td>22.3</td><td>28.4</td><td>28.6</td><td>47.4</td></tr><tr><td>DiMP [31]</td><td>一</td><td></td><td>1 × 1</td><td></td><td></td><td>37.8</td><td>48.5</td><td>31.5</td><td>44.2</td></tr><tr><td>PrDiMP [32]</td><td></td><td></td><td>1 × 1</td><td></td><td></td><td>34.9</td><td>44.5</td><td>32.2</td><td>46.9</td></tr><tr><td>MixFormer [33]</td><td>37.55</td><td></td><td>1× 1</td><td></td><td>一</td><td>38.9</td><td>50.4</td><td>一</td><td>一</td></tr><tr><td>STNet [4]</td><td>20.55</td><td>LIF</td><td>3 × 1</td><td></td><td></td><td></td><td>一</td><td>35.0</td><td>50.3</td></tr><tr><td>SNNTrack [5]</td><td>31.40</td><td>BA-LIF</td><td>5× 1</td><td>一</td><td>一</td><td>1</td><td>1</td><td>35.4</td><td>50.4</td></tr><tr><td>SpikeET [11]</td><td>22.36</td><td>I-LIF</td><td>1× 4</td><td>63.9</td><td>93.7</td><td></td><td>一</td><td>39.9</td><td>54.8</td></tr><tr><td>BSA [7]</td><td>19.61</td><td>I-LIF</td><td>1×4</td><td>59.2</td><td>91.4</td><td>40.9</td><td>51.8</td><td>36.8</td><td>52.3</td></tr><tr><td>MLPixer [34]</td><td>22.99</td><td>LIF</td><td>4× 1</td><td>57.9</td><td>90.1</td><td></td><td></td><td>34.5</td><td>48.9</td></tr><tr><td>AdaS [9] TP-Spikformer-0.65 [8]</td><td>19.61</td><td>I-LIF</td><td>1× 4</td><td>60.2</td><td>92.5</td><td></td><td></td><td>36.3</td><td>50.5</td></tr><tr><td></td><td></td><td>I-LIF</td><td>1×4</td><td>59.0</td><td>91.2</td><td>39.1</td><td>50.4</td><td>35.3</td><td>49.7</td></tr><tr><td rowspan="3">SDTrack-Tiny [6]</td><td>19.61</td><td>LIF</td><td>4× 1</td><td>56.7</td><td>89.1</td><td>35.8</td><td>44.0</td><td>35.4</td><td>48.7</td></tr><tr><td></td><td>I-LIF</td><td>2× 2</td><td>55.3</td><td>88.1</td><td>35.7</td><td>45.3</td><td>35.4</td><td>49.5</td></tr><tr><td></td><td>I-LIF</td><td>1×4</td><td>59.0</td><td>91.3</td><td>39.3</td><td>51.2</td><td>35.6</td><td>49.2</td></tr><tr><td>SDTrack-Base [6]</td><td>107.26</td><td>I-LIF</td><td>1 × 4</td><td>59.9</td><td>91.5</td><td>40.0</td><td>51.4</td><td>37.4</td><td>51.5</td></tr><tr><td>SDTrack-Tiny† SDTrack-Base</td><td>22.28 114.40</td><td>I-LIF</td><td>1× 4</td><td>62.8</td><td>92.8</td><td>40.5</td><td>51.4</td><td>39.8</td><td>55.0</td></tr><tr><td></td><td></td><td>I-LIF</td><td>1×4</td><td>64.0</td><td>94.0</td><td>40.8</td><td>51.8</td><td>40.5</td><td>55.2</td></tr><tr><td>SDTrack-Tiny with Mask IPL</td><td>22.28</td><td>I-LIF</td><td>1×4</td><td>63.4</td><td>93.6</td><td>40.7</td><td>51.7</td><td>40.2</td><td>55.2</td></tr><tr><td>SDTrack-Base with Mask IPL</td><td>114.40</td><td>I-LIF</td><td>1×4</td><td>64.4</td><td>94.7</td><td>41.1</td><td>52.2</td><td>40.7</td><td>55.4</td></tr></table>

TABLE 1: Comparison with representative ANN-based and SNN-based trackers on three event-based SOT benchmarks. Param. is the number of parameters, and $T \times D$ is the number of actual and virtual time steps. ‡ marks SDTrack retrained with the training strategy and data augmentation in Sec. 4.1, as the original SDTrack uses no data augmentation; these rows serve as the baseline of Mask IPL. The top three results are shown in red, blue, and green, respectively.

gradient takes the form

$$
\frac { \partial \mathcal { L } } { \partial \mathbf { W } _ { m } [ o ] } = \sum _ { q } \mathcal { M } _ { m } ( q ) \mathcal { M } _ { m - 1 } ( q + o ) \delta _ { m } ( q ) \widetilde { \mathbf { Y } } _ { m - 1 } ( q + o ) ,\tag{20}
$$

and the input gradient likewise propagates only along connections with both ends in the valid regions. Eq. (20) has the same form as the ideal gradient in Sec. 3.3, and the normalization statistics likewise come from the valid regions only. Hence, the actual gradient is the ideal gradient, and $\zeta _ { n }$ in Eq. (17) vanishes. The update direction coincides exactly with the intended descent direction, and the deviation of the optimization path described in Sec. 3.3 is eliminated. Eq. (18) applies to all convolutions, linear projections, and batch normalizations of both the convolutional and the transformer stages. Under this rule, the restored tokens no longer carry the perturbations described in Sec. 3.3, and the projections and MLPs of the transformer stage introduce no new perturbations. The attention in Eq. (16) therefore requires no additional treatment. Mask IPL thereby preserves the boundary mechanism through which the convolutional stage learns the relative position between the template and search frames, introduces no parameters, and makes the actual gradient coincide with the ideal gradient.

## 4 EXPERIMENTS

## 4.1 Implementation Details

The networks used in this study are identical to those of SDTrack, except for two modifications. First, a depthwise separable convolution block is inserted before each attention module. Second, a type coding is introduced, which assigns a separate learnable encoding to the template frame and to the search frame. We first pre-train the backbone on ImageNet-1K with the same pre-training strategy as SDTrack and then finetune it on the pair-matching task with 60,000 sample pairs per epoch. Training uses the AdaNSD optimizer [9] with cosine learning rate decay. On FE108 [3] and VisEvent [2], the tracker is trained for 50 epochs with the Voxel event representation and standard data augmentation; on FELT [35], it is trained for 300 epochs with the GTP event representation and without any data augmentation. All experiments run on four H100 GPUs with CUDA 13.0 and PyTorch 2.14.

## 4.2 Main Results

We compare the proposed method with representative ANNbased and SNN-based trackers on FE108, FELT, and VisEvent in Tab. 1. For a fair comparison, we reproduce SDTrack with the same training strategy and data augmentation as Mask IPL (Sec. 4.1). The original SDTrack, in contrast, is trained without data augmentation. The reproduced models are marked with ‡ and serve as the baselines of Mask IPL. In our experiments, data augmentation combined with the Voxel event representation performs considerably better on FE108 and VisEvent. On FELT, however, it performs clearly worse. These results will be provided in a later version of this paper, possibly with a detailed analysis.

Overall, we find that inserting a convolution block before each attention module noticeably improves tracking performance when data augmentation is applied. Following SpikeFET [11], we adopt this design (Sec. 4.1). Consequently, both the reproduced SDTrack and the trackers with Mask IPL contain more parameters than the original SDTrack. The increase is 2.67M at the Tiny scale and 7.14M at the Base scale. The type coding described in Sec. 4.1 also contributes to this increase. However, it adds only a few thousand parameters, which is negligible. Mask IPL itself introduces no parameters. Each tracker with Mask IPL therefore has the same number of parameters as its reproduced baseline.

Replacing IPL with Mask IPL improves the reproduced SDTrack-Tiny on all three benchmarks. On FE108, FELT, and VisEvent, the AUC increases by 0.6, 0.2, and 0.4 percentage points, respectively. The PR increases by 0.8, 0.3, and 0.2 per centage points, respectively. This improvement also extends to the Base scale on FE108, FELT, and VisEvent.

More notably, Mask IPL can reduce the amount of computation on any hardware platform. This requires only rewriting or modifying the convolution operators. Mask IPL clips the computation graph to the subgraph formed by the valid regions (Sec. 3.4). The modified operators can therefore skip the invalid regions. In contrast, the corresponding reduction for IPL is available only on neuromorphic chips. Even there, it is limited by the spread of the noise analyzed in Sec. 3.3. Because of this spread, the invalid regions emit spikes after the first layer and still trigger computation. The modified operators will be released in a later version.

## 5 CONCLUSION

This paper analyzes Intrinsic Position Learning (IPL), a parameter-free way of acquiring position information in event-based spike-driven tracking, and explains its effective ness. The zero blocks of the joint tensor are equivalent to zero padding for convolution, and the resulting boundary effect spreads layer by layer through the multi-stage convolution. Every parameter update is therefore driven by a gradient that perceives the relative displacement between the template and search frames. A positional encoding added after the convolutional stage cannot provide this information.

The same analysis shows that the invalid regions of the joint tensor take part in convolution, batch normalization, the neuron layers, and attention. They thereby introduce feature noise in forward propagation and gradient noise in backward propagation. The former increases the inference error, and the latter causes the optimization path to deviate from the ideal descent direction. We accordingly propose Computation Graph Clipping. It applies a validity mask determined by the layout to the operations of every layer, so that the invalid regions become equivalent to zero padding in both forward and backward propagation. The boundary mechanism is preserved, the noise is eliminated, and the actual gradient coincides with the ideal gradient, all without any additional parameters.

The resulting Mask IPL improves both the Tiny-scale and the Base-scale trackers on FE108, FELT, and VisEvent. These results indicate that efficiently modeling the relative position between the template and search frames is one of the key factors for event-based SOT. The analysis also offers a basis for the further design of IPL-based trackers.

## 6 LIMITATIONS

This manuscript is released in its current form so that Mask IPL can be made available as early as possible as a more robust alternative for acquiring position information in event-based spike-driven tracking. Owing to the limited preparation time, some of the content is not yet presented with full rigor, and we welcome comments from readers. Most of the material will be reorganized in the formal version, whose structure and presentation may differ considerably.

## REFERENCES

[1] G. Gallego, T. Delbrück, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis et al., “Event-based vision: A survey,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 1, pp. 154–180, 2020.

[2] X. Wang, J. Li, L. Zhu, Z. Zhang, Z. Chen, X. Li, Y. Wang, Y. Tian, and F. Wu, “Visevent: Reliable object tracking via collaboration of frame and event flows,” IEEE Transactions on Cybernetics, 2023.

[3] J. Zhang, X. Yang, Y. Fu, X. Wei, B. Yin, and B. Dong, “Object tracking by jointly exploiting frame and event domain,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13 043–13 052.

[4] J. Zhang, B. Dong, H. Zhang, J. Ding, F. Heide, B. Yin, and X. Yang, “Spiking transformers for event-based single object tracking,” in Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2022, pp. 8801–8810.

[5] J. Zhang, M. Zhang, Y. Wang, Q. Liu, B. Yin, H. Li, and X. Yang, “Spiking neural networks with adaptive membrane time constant for event-based tracking,” IEEE Transactions on Image Processing, 2025.

[6] Y. Shan, Z. Ren, H. Wu, W. Wei, R.-J. Zhu, S. Wang, D. Zhang, Y. Xiao, J. Zhang, K. Shi et al., “Sdtrack: A baseline for event-based tracking via spiking neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 36 245– 36 254.

[7] S. Wang, M. Zhang, J. Wang, D. Zhang, Y. Shan, J. E. Zhang, Y. Xiao, H. Cao, H. Zhang, Z. Ma et al., “Bipolar

self-attention for spiking transformers,” Advances in Neural Information Processing Systems, vol. 38, pp. 101 586– 101 611, 2026.

[8] W. Wei, X. Zhou, M. Zhang, A. Belatreche, Q. Sun, Y. Shan, D. Zhang, Z. Zhou, Z. Ma, Y. Yang et al., “Tpspikformer: Token pruned spiking transformer,” arXiv preprint arXiv:2603.00527, 2026.

[9] Z. Zhou, H. Cao, A. Belatreche, W. Wei, Y. Shan, Y. Liang, Y. Yang, S. Wang, Y. Ye, M. Zhang et al., “Adas: Adaptive gradient descent for spiking transformers,” in Forty-third International Conference on Machine Learning.

[10] Y. Shan and H. Qu, “Spiking neural networks with asynchronous spatiotemporal attention for neuromorphic vision.”

[11] J. Yang, L. Fan, J. Zhang, X. Lian, H. Shen, and D. Hu, “Fully spiking neural networks for unified frame-event object tracking,” Advances in Neural Information Processing Systems, vol. 38, pp. 121 132–121 163, 2026.

[12] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[13] B. Ye, H. Chang, B. Ma, S. Shan, and X. Chen, “Joint feature learning and relation modeling for tracking: A one-stream framework,” in European Conference on Computer Vision. Springer, 2022, pp. 341–357.

[14] B. Chen, P. Li, L. Bai, L. Qiao, Q. Shen, B. Li, W. Gan, W. Wu, and W. Ouyang, “Backbone is all your need: A simplified architecture for visual object tracking,” in European Conference on Computer Vision. Springer, 2022, pp. 375–392.

[15] N. Brunel and M. C. Van Rossum, “Lapicque’s 1907 paper: from frogs to integrate-and-fire,” Biological cybernetics, vol. 97, no. 5, pp. 337–339, 2007.

[16] W. Maass, “Networks of spiking neurons: the third generation of neural network models,” Neural networks, vol. 10, no. 9, pp. 1659–1671, 1997.

[17] M. Yao, X. Qiu, T. Hu, J. Hu, Y. Chou, K. Tian, J. Liao, L. Leng, B. Xu, and G. Li, “Scaling spike-driven transformer with efficient spike firing approximation training,” arXiv preprint arXiv:2411.16061, 2024.

[18] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in 2021 IEEE/CVF international conference on computer vision (ICCV). Ieee, 2021, pp. 9992–10 002.

[19] W. Yu, C. Si, P. Zhou, M. Luo, Y. Zhou, J. Feng, S. Yan, and X. Wang, “Metaformer baselines for vision,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

[20] M. A. Islam, S. Jia, and N. D. Bruce, “How much position information do convolutional neural networks encode?” arXiv preprint arXiv:2001.08248, 2020.

[21] O. S. Kayhan and J. C. v. Gemert, “On translation invariance in cnns: Convolutional layers can exploit absolute spatial location,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 274–14 285.

[22] B. Yan, H. Peng, J. Fu, D. Wang, and H. Lu, “Learning spatio-temporal transformer for visual tracking,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10 448–10 457.

[23] X. Wei, Y. Bai, Y. Zheng, D. Shi, and Y. Gong, “Autoregressive visual tracking,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 9697–9706.

[24] X. Chen, H. Peng, D. Wang, H. Lu, and H. Hu, “Seqtrack: Sequence to sequence learning for visual object tracking,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 572–14 581.

[25] B. Kang, X. Chen, D. Wang, H. Peng, and H. Lu, “Exploring lightweight hierarchical vision transformers for efficient visual tracking,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9612–9621.

[26] S. Gao, C. Zhou, and J. Zhang, “Generalized relation modeling for transformer tracking,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 686–18 695.

[27] W. Cai, Q. Liu, and Y. Wang, “Hiptrack: Visual tracking with historical prompts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19 258–19 267.

[28] Y. Zheng, B. Zhong, Q. Liang, Z. Mo, S. Zhang, and X. Li, “Odtrack: Online dense temporal token learning for visual tracking,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7588– 7596.

[29] B. Li, J. Yan, W. Wu, Z. Zhu, and X. Hu, “High performance visual tracking with siamese region proposal network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 8971– 8980.

[30] M. Danelljan, G. Bhat, F. S. Khan, and M. Felsberg, “Atom: Accurate tracking by overlap maximization,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4660–4669.

[31] G. Bhat, M. Danelljan, L. V. Gool, and R. Timofte, “Learning discriminative model prediction for tracking,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 6182–6191.

[32] M. Danelljan, L. V. Gool, and R. Timofte, “Probabilistic regression for visual tracking,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 7183–7192.

[33] Y. Cui, C. Jiang, L. Wang, and G. Wu, “Mixformer: End-to-end tracking with iterative mixed attention,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 13 608–13 618.

[34] J. E. Zhang, X. Zhou, S. Wang, W. Wei, H. Liu, Q. Sun, M. Zhang, Y. Yang, and H. Li, “Unveiling the spatialtemporal effective receptive fields of spiking neural networks,” Advances in Neural Information Processing Systems, vol. 38, pp. 35 904–35 927, 2026.

[35] X. Wang, J. Huang, S. Wang, C. Tang, B. Jiang, Y. Tian, J. Tang, and B. Luo, “Long-term frame-event visual tracking: Benchmark dataset and baseline,” arXiv preprint arXiv:2403.05839, 2024.