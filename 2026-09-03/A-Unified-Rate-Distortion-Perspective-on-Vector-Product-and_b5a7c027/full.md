# A Unified Rate-Distortion Perspective on Vector, Product, and Scalar Quantization

Xianghong Fang<sup>1</sup> Wenlong Mou<sup>1</sup> Yuan Yuan<sup>2</sup>

Dehan Kong<sup>1</sup> Tim G. J. Rudner<sup>1,3</sup>

<sup>1</sup>University of Toronto <sup>2</sup>Boston College <sup>3</sup>Vijil

<sup></sup> Website <sup>§</sup> Code & Models

## Abstract

Discrete visual tokenization, predominantly driven by vector, scalar, and product quantization, lacks a unified conceptual framework for understanding quantization tradeofs. In this paper, we propose a unified rate–distortion perspective on modern discrete visual tokenization. By viewing quantization as lossy compression, we characterize the nominal fixed-length coding rate through token count and codebook size, and quantization error as the distortion. Within this framework, we resolve three central questions. First, we theoretically and empirically show that minimizing distortion, rather than maximizing codebook utilization, is the primary intrinsic objective for reconstruction fidelity, with a direct connection to the STE-induced gradient discrepancy. Second, we establish two critical fairness conditions for intrinsic quantization comparison: controlling latent feature statistics and enforcing identical coding rates. Third, under these conditions, we recover the VQ–PQ–SQ distortion hierarchy in modern visual tokenization and show empirically that modern VQ methods achieve the lowest distortion. This work provides a foundational rate–distortion reframing of modern discrete visual tokenization, resolves ambiguities in quantizer evaluation, and provides a controlled framework for isolating intrinsic quantization efectiveness under fixed-rate constraints.

## 1 Introduction

Discrete visual tokenization has achieved remarkable success in recent years, driven by advances in vector quantization [VQ; ERO21; Sun+24; Tia+24], product quantization [PQ; JDS11; Guo+24; Li+25], and scalar quantization [SQ; Men+24; Yu+24a; ZXK25; Han+25]. A major line of VQ research has focused on improving codebook utilization through improved codebook updates, optimization, or parameterizations [Lee+22; ZV23; Zhu+24a; Zhu+24b; Shi+25; Cha+26; Lu+26]. Despite this rapid progress, the conceptual relationship between utilization, distortion, and quan tization efectiveness remains comparatively less understood. In particular, high or even complete codebook utilization does not necessarily imply faithful representation of the input. This motivates a more fundamental framework for understanding the objectives and tradeofs underlying diferent quantization algorithms.

In this paper, we propose a unified rate-distortion perspective on quantization to systematically address these considerations. Rate-distortion theory provides the classical framework for characterizing lossy compression, and has also motivated learned image compression methods that jointly optimize coding rate and reconstruction distortion [BLS17; LHB25; ZW24; Jia+26]. We use rate-distortion theory as an analytical framework for comparing discrete visual quantizers by defining distortion as quantization error and rate as the nominal fixed-length coding budget determined by token count and codebook size. Under this perspective, we resolve three central open questions in discrete visual tokenization, which are summarized in Figure 1.

![](images/02252d2cee9c1e1dadd6e563d5860f4d0cd8b490216092075949191b2064c8e5.jpg)  
Figure 1: Overview of our unified rate-distortion perspective on quantization. (a) Primary objective. Distortion is a more fundamental criterion than codebook utilization, with direct links to utilization, STE-induced gradient discrepancy, and reconstruction fidelity. (b) Fair comparison. Intrinsic comparison requires matched latent feature distributions and the same nominal fixed-length rate. (c) Intrinsic efectiveness. VQ, PQ, and SQ satisfy $\mathsf { S Q } \subseteq \mathsf { P Q } \subseteq \mathsf { V Q }$ , implying $\mathcal { E } _ { \mathrm { V Q } } ^ { \ast } \leq \mathcal { E } _ { \mathrm { P Q } } ^ { \ast } \leq \mathcal { E } _ { \mathrm { S Q } } ^ { \ast }$ under matched latent distributions and fixed rate.

First, we investigate the primary optimization objective for quantization algorithms in modern visual tokenization. From a rate-distortion perspective, minimizing distortion rather than directly maximizing codebook utilization is the key to achieving training stability and high-fidelity reconstruction. We provide both theoretical and empirical evidence supporting distortion minimization as the more fundamental optimization principle. Specifically, we prove that minimizing distortion necessarily implies full codebook utilization under mild conditions, whereas the converse does not hold. We further establish a direct connection between distortion and the gradient discrepancy induced by the straight-through estimator(STE) [BLC13]. Empirically, we demonstrate substantially stronger correlations between distortion and reconstruction fidelity (measured by rFID) than those observed with codebook utilization. These results motivate a distortion-centric perspective on VQ optimization beyond codebook utiliza tion, as summarized in Figure 1(a), complementing prior analyses of STE bias [Huh+23; Fif+25] and recent eforts to address codebook collapse [Zhu+24b; Shi+25; Cha+26; Lu+26].

Second, we establish two fundamental conditions for fairly comparing the intrinsic efectiveness of quantization algorithms, as illustrated in Figure 1(b). First, the latent feature distribution must be controlled across compared algorithms. In particular, under global rescaling, the minimum achievable squared quantization distortion changes proportionally with the latent variance, so diferences in feature statistics can confound comparisons of quantizer efectiveness. Second, all algorithms must operate at the same nominal fixed-length coding rate, R = T log K, which is jointly determined by the number of visual tokens T and the cardinality of the discrete code space K. Only under matched latent distributions and coding rates can the intrinsic efectiveness of diferent quantization algorithms be meaningfully compared.

Third, we examine the intrinsic efectiveness of quantization algorithms under rigorously controlled conditions. As illustrated in Figure 1(c), SQ, PQ, and VQ form a nested hierarchy in which VQ generalizes PQ and PQ generalizes SQ, implying greater modeling flexibility and no higher optimal distortion for VQ. Related work on neural image compression has similarly shown that vector or lattice quantization can achieve lower distortion than scalar quantization when when exploiting cross-dimensional dependencies [LHB25; ZW24]. However, early visual-tokenizer studies reported severe codebook collapse for VQ, particularly with large codebooks [Dha+20; Tak+22; Yu+22; Lee+22; ZV23], leading to inferior empirical performance relative to SQ and PQ in prior works [Men+24; Yu+24a; ZXK25; Guo+24]. To resolve this apparent conflict, we examine whether the classical distortion hierarchy among VQ, PQ, and SQ can be realized in modern visual tokenization under controlled fixed-rate conditions. Beyond this nested ordering, we show that VQ can exploit low-dimensional source structure to achieve strictly better distortion scaling than PQ and SQ. Our empirical study further shows that advanced VQ algorithms [Fan+26a; Fan+26b] consistently achieve lower distortion and better reconstruction under the two fairness conditions above.

Our key contributions are as follows:

1. We establish distortion minimization as a fundamental intrinsic optimization principle for modern visual quantization. We show that distortion minimization implies full codebook utilization under mild conditions, whereas the converse does not hold. We further connect quantization distortion to the gradient discrepancy induced by the straight-through estimator..

2. We identify two fundamental conditions for fair intrinsic comparison of quantization algorithms. We show that comparisons must control both the scale of latent features and the coding rate, since diferences in either factor can obscure the intrinsic efectiveness of the quantizer itself.

3. We characterize the VQ–PQ–SQ distortion hierarchy in modern visual tokenization and show that VQ can exploit low-dimensional source structure to achieve strictly better distortion scaling than PQ and SQ. Empirically, modern VQ methods achieve the lowest distortion and best reconstruction after decoder adaptation under controlled comparisons.

## 2 Background

## 2.1 Discrete Visual Tokenization

Discrete visual tokenizers map images into sequences of discrete symbols that can subsequently be modeled by generative models [OVK17; ERO21; Sun+24; Tia+24]. A typical discrete tokenizer consists of an encoder $\mathcal { E } _ { \theta _ { : } }$ , a quantization module $\mathcal { Q } _ { \phi }$ , and a decoder $\mathcal { D } _ { \varphi }$ . Given an image $\pmb { x } \in \mathbb { R } ^ { H \times W \times 3 }$ the encoder produces continuous latent features

$$
\begin{array} { r } { z _ { e } = \mathscr { E } _ { \theta } ( \pmb { x } ) \in \mathbb { R } ^ { h \times w \times d } , \qquad h = H / f , \ w = W / f , } \end{array}
$$

where $f$ denotes the spatial downsampling factor and d the latent dimension.

At each spatial position (i, j), the quantizer maps the continuous feature $z _ { e } ^ { i j } \in \mathbb { R } ^ { d }$ to a discrete representation,

$$
r ^ { i j } = \mathcal { I } _ { \phi } ( z _ { e } ^ { i j } ) , \qquad z _ { q } ^ { i j } = \mathcal { Q } _ { \phi } ( z _ { e } ^ { i j } ) ,
$$

where $r ^ { i j }$ denotes the discrete symbol and $z _ { q } ^ { i j }$ the quantized representation. The resulting discrete symbols can be used as visual tokens for generative modeling, while the quantized latent is decoded as $\hat { \pmb x } = \mathcal { D } _ { \varphi } ( z _ { q } )$ . Diferent quantization algorithms impose diferent structures on the mapping from $z _ { e } \mathrm { t o } z _ { q }$ . In this work, we focus on three widely used families: vector quantization (VQ), product quantization (PQ), and scalar quantization (SQ), which we introduce below.

## 2.2 Vector Quantization

Vector quantization (VQ) discretizes an entire d-dimensional feature vector jointly [OVK17]. Given a learnable codebook $\phi = \{ e _ { k } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { d }$ , containing K code vectors, VQ assigns each continuous feature $z _ { e } ^ { i j }$ to its nearest codebook entry:

$$
r ^ { i j } = \underset { k \in \{ 1 , \ldots , K \} } { \mathrm { a r g m i n } } \Vert z _ { e } ^ { i j } - e _ { k } \Vert _ { 2 } ^ { 2 } , \qquad z _ { q } ^ { i j } = e _ { r ^ { i j } } .
$$

Thus, each spatial feature is represented by one of the K d-dimensional code vectors.

A longstanding optimization challenge in VQ is codebook collapse, where only a subset of the avail able code vectors is frequently selected [Dha+20; Tak+22; Yu+22; Lee+22; ZV23]. The problem can become particularly pronounced for large codebooks [ZV23; Men+24]. A substantial body of work has therefore developed improved codebook updates, optimization procedures, and parameterizations to increase codebook usage [ROV19; Wil+20; Zha+23; Zhu+24a; Zhu+24b; Shi+25; Cha+26; Lu+26]. Other work has studied dificulties caused by the non-diferentiability of quantization, including the straight-through estimator and alternative gradient transformations [Huh+23; Fif+25].

## 2.3 Product Quantization

Product quantization (PQ) factorizes a d-dimensional feature vector into M lower-dimensional subvectors and quantizes each subvector independently [JDS11; Guo+24; Li+25]. Specifically,

$$
z _ { e } ^ { i j } = \bigoplus _ { m = 1 } ^ { M } z _ { m } ^ { i j } , \qquad z _ { m } ^ { i j } \in \mathbb { R } ^ { d _ { m } } , \qquad \sum _ { m = 1 } ^ { M } d _ { m } = d ,
$$

where L denotes channel-wise concatenation. Each subvector is quantized using an independent subcodebook $\phi _ { m } = \{ e _ { m , k } \} _ { k = 1 } ^ { n _ { m } } \subset \mathbb { R } ^ { d _ { m } }$ such that

$$
r _ { m } ^ { i j } = \operatorname * { a r g m i n } _ { k \in \{ 1 , \ldots , n _ { m } \} } \| z _ { m } ^ { i j } - e _ { m , k } \| _ { 2 } ^ { 2 } , \qquad \hat { z } _ { m } ^ { i j } = e _ { m , r _ { m } ^ { i j } } .
$$

The full quantized feature is then $z _ { q } ^ { i j } = \oplus _ { m = 1 } ^ { M } \hat { z } _ { m } ^ { i j }$ . The M subcodebooks jointly induce $K =$ $\textstyle \prod _ { m = 1 } ^ { M } n _ { m }$ possible composite codewords. Hence, PQ can represent a combinatorial discrete space using $\textstyle \sum _ { m = 1 } ^ { M } n _ { m }$ explicitly stored subcodebook entries. For later analysis, we use K to denote this composite code-space cardinality, rather than the number of explicitly stored vectors.

## 2.4 Scalar Quantization

Scalar quantization (SQ) independently discretizes scalar components of a latent representation and can be viewed as the maximally factorized case of PQ with $M = d .$ Writing

$$
z _ { e } ^ { i j } = \bigoplus _ { m = 1 } ^ { d } z _ { m } ^ { i j } , \qquad z _ { m } ^ { i j } \in \mathbb { R } ,
$$

each scalar component is quantized independently using a scalar codebook $\phi _ { m } = \{ e _ { m , k } \} _ { k = 1 } ^ { n _ { m } } \subset \mathbb { R }$ The corresponding discrete index and quantized value are

$$
r _ { m } ^ { i j } = \operatorname * { a r g m i n } _ { k \in \{ 1 , . . . , n _ { m } \} } | z _ { m } ^ { i j } - e _ { m , k } | ^ { 2 } , \qquad \hat { z } _ { m } ^ { i j } = e _ { m , r _ { m } ^ { i j } } .
$$

The resulting quantized vector is $z _ { q } ^ { i j } = \textstyle \bigoplus _ { m = 1 } ^ { d } \hat { z } _ { m } ^ { i j }$ , with composite code-space cardinality $\begin{array} { r } { K = \prod _ { m = 1 } ^ { d } n _ { m } } \end{array}$ Modern visual tokenizers instantiate this general scalar factorization in diferent ways. FSQ uses a small number of finite scalar levels per latent dimension [Men+24]. LFQ uses binary values for each quantized coordinate [Yu+24a], while BSQ combines binary quantization with hyperspherical normalization [ZXK25].

## 3 Toward a Controlled Comparison ofQuantization Algorithms

Comparing quantization algorithms across independently trained visual tokenizers does not isolate the contribution of the quantizer itself. End-to-end tokenizer performance is jointly afected by the quantization module and the surrounding training system, including encoder–decoder capacity and architecture, discriminator design, training data, optimization schedule, and computational budget. These factors can vary substantially across existing tokenizer systems, while training each alternative quantizer from scratch under an otherwise identical setup is often computationally expensive. Consequently, empirical comparisons frequently rely on separately developed tokenizer systems whose experimental conditions are not fully matched.

As illustrated in Table 1, representative tokenizers difer substantially along these dimensions [ERO21; Lee+22; Zhu+24a; Sun+24; Tia+24; Li+25; Ma+25]. As a result, diferences in reconstruction performance may reflect not only the quantization algorithm, but also changes in model capacity, training objectives, data, or optimization resources. Such comparisons remain informative for evaluating complete tokenizer systems, but are insuficient for isolating the intrinsic efectiveness of VQ, PQ, and SQ.

Table 1: Representative implementation diferences among discrete visual tokenizers. Diferences in model design, training setup, and computational budget can confound direct comparisons of quantization algorithms. ‘-’ denotes unreported information, and “Para” denotes the number of encoder–decoder parameters.
<table><tr><td>Tokenizers</td><td>Para</td><td>Architectures</td><td>Discriminator</td><td>Training Datasets</td><td>Training Epochs</td><td>Training GPU Hours</td></tr><tr><td>VQGAN [ERO21]</td><td>68M</td><td>CNN U-Net</td><td>PatchGAN</td><td>ImageNet-1k</td><td></td><td></td></tr><tr><td>RQVAE [Lee+22]</td><td>95M</td><td>CNN U-Net</td><td>PatchGAN</td><td>ImageNet-1k</td><td>50</td><td></td></tr><tr><td>VQGAN-LC [Zhu+24a]</td><td>68M</td><td>CNN U-Net</td><td>PatchGAN</td><td>ImageNet-1k</td><td>20</td><td>32 × V100 - Hours</td></tr><tr><td>Llama GEN [Sun+24]</td><td>68M</td><td>CNN U-Net</td><td>PatchGAN</td><td>ImageNet-1k</td><td>40</td><td>2 × A100 200 Hours</td></tr><tr><td>VAR [Tia+24]</td><td>104M</td><td>CNN U-Net</td><td>StyleGAN</td><td>OpenImages</td><td>16</td><td>16 × A100 60 Hours</td></tr><tr><td>ImageFolder [Li+25]</td><td></td><td>Transformer SEED</td><td>StyleGAN</td><td>ImageNet-1k</td><td>200</td><td>32 × A100 40 Hours</td></tr><tr><td>UniTok [Ma+25]</td><td></td><td>Transformer SEED</td><td>StyleGAN</td><td>OpenImages</td><td></td><td>256 × A100 50 Hours</td></tr></table>

These confounders motivate three fundamental questions about intrinsic quantization efectiveness: (Q1) What should a quantization algorithm fundamentally optimize? (Q2) What conditions are required for a fair intrinsic comparison of diferent quantization algorithms? (Q3) Under these controlled conditions, how do VQ, PQ, and SQ compare in achievable distortion? In the next section, we develop a unified rate-distortion perspective to answer these three questions.

## 4 A Rate-distortion Perspective

We now develop a unified rate–distortion framework to address the three questions posed in Section 3. We first formalize the rate and distortion of discrete quantization, and then study its fundamental optimization objective, the conditions required for fair intrinsic comparison, and the achievable distortion of VQ, PQ, and SQ under controlled conditions.

## 4.1 Fixed-Length Rate and Quantization Distortion

We formulate discrete quantization as a fixed-rate lossy compression problem. Unlike entropy-coded neural compression, where the rate is determined by the expected bitstream length under a learned entropy model [BLS17; Jia+26], our goal is to compare the intrinsic efectiveness of diferent quantization structures under a controlled representation budget. We therefore characterize the representational budget using the nominal fixed-length coding rate and measure the information lost through quantization using squared-error distortion.

Nominal fixed-length rate. Consider a tokenizer that produces T discrete tokens, each taking values in a discrete space of cardinality K. Its nominal fixed-length coding rate is

$$
R = T \log _ { 2 } K \quad { \mathrm { b i t s } } .\tag{4.1}
$$

Equivalently, each token carries a nominal coding budget of log K bits. This quantity measures the representational budget available to the quantizer under fixed-length coding and should be distinguished from the expected entropy-coded rate, which additionally depends on the probability distribution of the discrete symbols.

For VQ, K is the number of available code vectors. For PQ and SQ, K denotes the cardinality of the composite discrete code space rather than the number of explicitly stored code vectors. Specifically, if the representation is factorized into M independently quantized components with sub-codebook sizes $\lbrace n _ { m } \rbrace _ { m = 1 } ^ { M }$ , then $\begin{array} { r } { K = \prod _ { m = 1 } ^ { M } n _ { m } } \end{array}$ . Thus, VQ, PQ, and SQ can be compared under the same nominal coding rate even though they impose diferent structural constraints on the discrete representation.

Quantization distortion. Let $X \sim P$ denote a latent feature vector in $\mathbb { R } ^ { d } ;$ , and let $\mathcal { Q } _ { \phi }$ denote a quantizer that maps X to a quantized representation $\mathcal { Q } _ { \phi } ( X )$ . We measure quantization distortion using the expected squared Euclidean error

$$
\small \mathcal { E } ( P , \mathcal { Q } _ { \phi } ) = \mathbb { E } _ { X \sim P } \left[ \left. X - \mathcal { Q } _ { \phi } ( X ) \right. _ { 2 } ^ { 2 } \right] .\tag{4.2}
$$

In empirical evaluations, this expectation is approximated by averaging the squared quantization error over the observed latent features. This distortion directly measures the information lost by the quantization operation and serves as the central quantity in our subsequent analysis.

Remark 1 (Rate allocation). Under the nominal fixed-length rate $R = T \log _ { 2 } K ,$ , doubling the token count is equivalent to squaring the code-space cardinality in terms of representation rate:

$$
2 T \log _ { 2 } K = T \log _ { 2 } K ^ { 2 } .\tag{4.3}
$$

This equivalence concerns the nominal coding budget only; equal rates do not in general imply identical distortion, since diferent allocations of the same rate across token count and code-space cardinality may induce diferent representation structures. In our controlled comparisons, we therefore match both the token count and the composite code-space cardinality whenever possible, providing a stronger control than matching R alone.

## 4.2 Distortion Minimization: The Primary Optimization Objective

A substantial line of work on VQ has focused on mitigating codebook collapse and improving codebook utilization [Zhu+24a; Dha+20; Wil+20; ZV23; Zha+23; Ram+21]. In this work, we contend that minimizing distortion constitutes a more fundamental optimization objective. Intuitively, lower distortion promotes more stable and faithful representations in lossy compression systems [TL99; TRZ17; TL01].

We first show that distortion controls the local gradient discrepancy induced by the straight-through estimator (STE). As derived in Appendix E, under standard local smoothness, the squared gradient discrepancy G satisfies

$$
\mathcal { G } = \mathcal { O } \big ( \| z _ { e } - z _ { q } \| _ { 2 } ^ { 2 } \big ) .\tag{4.4}
$$

Since $\| z _ { e } - z _ { q } \| _ { 2 } ^ { 2 }$ is exactly the squared-error distortion, this establishes a local connection between quantization distortion and STE gradient fidelity: as the distortion approaches zero, the gradient discrepancy also vanishes. Thus, reducing distortion makes the gradient evaluated at the quantized representation locally closer to that evaluated at the continuous representation, providing a mechanism through which lower distortion can contribute to more stable optimization.

We further formalize the relationship between distortion minimization and codebook utilization. Let X be a random variable supported on $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ . A deterministic quantizer with codebook size $K \in \mathbb { N } _ { + }$ is a mapping $f : \mathcal { X } \to \{ 1 , \dots , K \}$ . Define

$$
\mathcal { E } ( f ) \triangleq \operatorname* { m i n } _ { \substack { \hat { x } : \{ 1 , \ldots , K \}  \mathbb { R } ^ { d } } } \mathbb { E } \big [ \| X - \hat { x } ( f ( X ) ) \| _ { 2 } ^ { 2 } \big ] , \quad \mathrm { U } ( f ) \triangleq \frac { \big | \{ k : \mathbb { P } ( f ( X ) = k ) > 0 \} \big | } { K } .
$$

Assumption 1 (Non-degenerate splittability). For any measurable set $A \subseteq { \mathcal { X } }$ with $\mathbb { P } ( X \in A ) > 0$ and nonzero conditional variance, there exists a partition of A into two measurable subsets $A = A _ { 1 } \dot { \cup } A _ { 2 }$ each with positive probability, such that

$$
\mathbb { E } [ X \mid X \in A _ { 1 } ] \neq \mathbb { E } [ X \mid X \in A _ { 2 } ] .
$$

This mild condition is satisfied by standard continuous distributions with densities. The following proposition further excludes the trivial setting in which the support of the distribution can already be represented exactly using at most K codewords, since in that case zero distortion may be achieved without utilizing the entire codebook. For example, for an empirical distribution supported on n distinct samples, $K < n$ is suficient to rule out such a zero-distortion solution.

Proposition 2. (a) Let

$$
f ^ { \star } \in \operatorname * { a r g m i n } _ { f : \mathcal { X } \to \{ 1 , . . . , K \} } \mathcal { E } ( f ) .\tag{4.5}
$$

Under Assumption 1, if no zero-distortion quantizer with at most K codewords exists, then every global distortion minimizer satisfies $\mathrm { U } ( f ^ { \star } ) = 1$

(b) There exist quantizers $g : \mathcal { X }  \{ 1 , \dotsc , K \}$ with $\mathrm { U } ( g ) = 1$ that are not distortion minimizers:

$$
\mathcal { E } ( g ) > \operatorname* { m i n } _ { f : \mathcal { X } \to \{ 1 , . . . , K \} } \mathcal { E } ( f ) .
$$

Proposition 2 establishes that, under these mild conditions, every global distortion minimizer fully utilizes the codebook, whereas full utilization does not imply distortion optimality. The proof is provided in Appendix A.

## 4.3 Conditions for Fair Intrinsic Comparison

A fair intrinsic comparison of quantization algorithms requires controlling both the latent source distribution and coding rate. First, all quantizers must operate on the same latent distribution, because quantization distortion depends not only on the quantization algorithm but also on the statistical properties of the source distribution. Even under the simplest change in source statistics, namely a global rescaling $X ^ { \prime } = a X$ , the optimal K-point squared-error distortion changes as

$$
{ { \mathcal { E } } ^ { \star } } ( a X , K ) = a ^ { 2 } { { \mathcal { E } } ^ { \star } } ( X , K ) .\tag{4.6}
$$

Thus, diferences in latent feature statistics can directly confound distortion comparisons. This result is formally established in Proposition 5 in Appendix F.

Condition 1 (Matched latent distribution). All compared quantizers must be evaluated on the same latent distribution P.

Second, distortion must be compared under the same coding budget. For $T$ discrete tokens with code-space cardinality K, the nominal fixed-length coding rate is $R = T \log _ { 2 } K$ . Since both T and K afect representational capacity and reconstruction behavior [Yu+24a; Zhu+24a; Yu+24b; Bac+25], comparisons at diferent rates do not isolate intrinsic quantization efectiveness.

Condition 2 (Matched nominal rate). The nominal fixed-length coding rate must be identical across all compared algorithms.

In our controlled experiments, we adopt the stronger criterion $T _ { A } = T _ { B }$ and $K _ { A } = K _ { B }$ , rather than matching only $T \log _ { 2 } K$ , to avoid confounding diferent allocations of the same nominal rate.

## 4.4 Rate–Distortion Analysis of VQ, PQ, and SQ

In this section, we study the rate–distortion behavior of idealized VQ, PQ, and SQ, and compare their distortion under equal coding rates.

Given a probability distribution $\mathbb { P }$ over $\mathbb { R } ^ { d }$ and a codebook size $K \in \mathbb { N } _ { + }$ , we define the optimal distortions for VQ, PQ, and SQ as follows:

$$
\begin{array} { r } {  { \mathcal E } _ { \mathrm { V Q } } ^ { * } ( \mathbb { P } , K ) : = \operatorname* { i n f } _ { \phi } \Big \{  { \mathbb E } \left[ \| X -  { \mathcal Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \right] : | \phi | \le K \Big \} , } \end{array}
$$

$$
\mathcal { E } _ { \mathrm { P Q } } ^ { * } \big ( \mathbb { P } , K , M \big ) : = \operatorname* { i n f } _ { \phi } \Big \{ \mathbb { E } \big [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \big ] : \phi = \bigoplus _ { m = 1 } ^ { M } \phi _ { m } , \phi _ { m } \subseteq \mathbb { R } ^ { d _ { m } } , | \phi _ { m } | = n _ { m } , \prod _ { m = 1 } ^ { M } n _ { m } \leq K \Big \} ,
$$

$$
\mathcal { E } _ { \mathrm { S Q } } ^ { * } ( \mathbb { P } , K ) : = \operatorname* { i n f } _ { \phi } \Big \{ \mathbb { E } \big [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \big ] : \phi = \bigoplus _ { m = 1 } ^ { d } \phi _ { m } , \phi _ { m } \subseteq \mathbb { R } , | \phi _ { m } | = n _ { m } , \prod _ { m = 1 } ^ { d } n _ { m } \leq K \Big \} .
$$

Here $| \phi |$ denotes the size of the codebook $\phi ,$ and $\oplus$ denotes the Cartesian product of sets.

Clearly, SQ is a special case of PQ with $M = d ,$ and PQ is a special case of ${ \mathrm { V Q } } ,$ a classical structural relationship in quantization theory. Thus, the optimal distortions satisfy:

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { V Q } } ^ { \ast } ( \mathbb { P } , K ) \leq \mathcal { E } _ { \mathrm { P Q } } ^ { \ast } \big ( \mathbb { P } , K , M \big ) \leq \mathcal { E } _ { \mathrm { S Q } } ^ { \ast } \big ( \mathbb { P } , K \big ) , \ \mathrm { f o r ~ a n y ~ } 2 \leq M \leq d . } \end{array}
$$

Let $\mathcal { P }$ be the set of all probability distributions over $[ - 1 , 1 ] ^ { d }$ . The following results provide quantitative characterizations of the optimal distortions for VQ, PQ, and SQ.

Proposition 3. For any $K \geq 2 ^ { d } ,$ , we have

$$
\frac { d } { 8 K ^ { 2 / d } } \leq \operatorname* { s u p } _ { \mathbb { P } \in \mathcal { P } } \mathcal { E } _ { V Q } ^ { * } ( \mathbb { P } , K ) \leq \operatorname* { s u p } _ { \mathbb { P } \in \mathcal { P } } \mathcal { E } _ { S Q } ^ { * } ( \mathbb { P } , K ) \leq \frac { 4 d } { K ^ { 2 / d } }
$$

See Section B for the proof. Proposition 3 shows that for worst-case distributions, the optimal distortions of VQ, PQ, and SQ are the same up to universal constant factors. All three methods achieve a distortion that scales as $\Theta ( d / K ^ { 2 / d } )$ ).

On the other hand, when the data distribution has intrinsic low-dimensional structures, VQ can achieve strictly better distortion scaling than PQ and SQ. We illustrate this phenomenon in the following proposition.

Proposition 4. If the support of $\mathbb { P } \in \mathcal { P }$ is contained in a d -dimensional subspace of $\mathbb { R } ^ { d }$ with $d _ { e f f } < d ,$ then for any $K \geq 2 ^ { d _ { e f f } }$ , we have

$$
\mathcal { E } _ { V Q } ^ { * } ( \mathbb { P } , K ) \leq \frac { 4 d d _ { e f f } } { K ^ { 2 / d _ { e f f } } } .
$$

On the other hand, there exists a 1-dimensional linear subspace $L \subseteq \mathbb { R } ^ { d }$ and a distribution P supported on $L \cap [ - 1 , 1 ] ^ { d }$ such thatfor any $K \geq 2 ^ { d }$ , we have

$$
\mathcal { E } _ { P Q } ^ { * } ( \mathbb { P } , K , M ) \geq \frac { M } { 4 } K ^ { - 2 / M } , f o r a n y 2 \leq M \leq d , a n d \mathcal { E } _ { S Q } ^ { * } ( \mathbb { P } , K ) \geq \frac { d } { 4 } K ^ { - 2 / d } .
$$

See Section C for the proof. Proposition 4 shows that when the data distribution has intrinsic dimension $d _ { \mathrm { e f f } } < d , \mathrm { V Q }$ can achieve distortion scaling as $\mathcal { O } ( K ^ { - 2 / d _ { \mathrm { e f f } } } )$ , significantly smaller than the worst-case rate $\Theta ( d / K ^ { 2 / d } )$ when $d _ { \mathrm { e f f } } \ll d .$ . In contrast, for a simple 1-dimensional distribution, SQ still sufers $\Omega ( d / \dot { K } ^ { 2 / d } )$ while VQ can achieve $\mathcal { O } ( d K ^ { - 2 } )$ . PQ interpolates between these extremes, with performance depending on the number of blocks M. This separation goes beyond the nested family ordering: VQ can exploit intrinsic low-dimensional structure, whereas fixed coordinate-wise factorization in PQ and SQ may prevent such exploitation. The VQ upper-bound argument also extends to supports with covering-number growth $N \bar { ( } \varepsilon ) \lesssim C \varepsilon ^ { - d _ { \mathrm { e f f } } }$ , with constants depending on covering regularity; the linear-subspace case above is the cleanest special case. Appendix D discusses the relationship of these results to classical quantization theory and clarifies the novelty boundary.

## 5 Empirical Evaluation

We empirically compare VQ, PQ, and SQ under matched latent distributions and nominal coding rates, with primary experiments conducted on ImageNet-1K [Den+09]. We further examine rate-distortion behavior across coding budgets and validate the conclusions across datasets and in a controlled pixel-space setting.

## 5.1 Experimental Setup

Controlled latent-space comparison. For our primary experiments, we use a pretrained VAR tokenizer [Tia+24] within the VQ-Transplant framework [Fan+26b]. During quantization-module substitution, the pretrained encoder and decoder are frozen while the original quantizer is replaced by the method under evaluation. During decoder adaptation, the encoder and transplanted quantizer are frozen while the decoder adapts to the quantized representation. Since all methods share the same pretrained encoder and representation pipeline, they operate on the same latent source distribution, satisfying the matched-distribution condition in Section 4.3.

Table 2: Quantization and reconstruction performance of VQ, PQ, and SQ on ImageNet-1K under the controlled latent-space setting. <sup>†</sup>: Results cited from VQ-Transplant [Fan+26b]. Within each quantization type and phase (Substitution/Adaptation), optimal values are underlined; overall best results per metric are bold.
<table><tr><td>Approaches</td><td>Types</td><td>Phase</td><td>Tokens</td><td>K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td><td>r-IS(↑)</td></tr><tr><td>Vanilla VQ†</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.422</td><td>0.2%</td><td>22.04</td><td>53.1</td><td>0.243</td><td>10.89</td><td>103.8</td></tr><tr><td>EMA VQ†</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.217</td><td>65.5%</td><td>24.94</td><td>65.9</td><td>0.127</td><td>1.78</td><td>185.8</td></tr><tr><td>Online VQ†</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.280</td><td>13.5%</td><td>24.42</td><td>63.2</td><td>0.147</td><td>2.28</td><td>174.2</td></tr><tr><td>Wasserstein VQ†</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.201</td><td>99.6%</td><td>25.22</td><td>66.9</td><td>0.121</td><td>1.76</td><td>186.0</td></tr><tr><td>MMD VQ†</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.201</td><td>99.9%</td><td>25.24</td><td>66.8</td><td>0.121</td><td>1.69</td><td>187.3</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.233</td><td>59.2%</td><td>24.80</td><td>65.3</td><td>0.130</td><td>1.84</td><td>183.1</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.209</td><td>100%</td><td>25.08</td><td>66.4</td><td>0.123</td><td>1.68</td><td>187.2</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.211</td><td>100%</td><td>25.09</td><td>66.3</td><td>0.124</td><td>1.79</td><td>185.7</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.217</td><td>100%</td><td>24.79</td><td>65.8</td><td>0.128</td><td>1.78</td><td>185.7</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.212</td><td>100%</td><td>25.00</td><td>66.1</td><td>0.123</td><td>1.61</td><td>189.5</td></tr><tr><td>FSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.272</td><td>100%</td><td>24.46</td><td>62.5</td><td>0.140</td><td>2.35</td><td>174.8</td></tr><tr><td>LFQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.279</td><td>29.8%</td><td>24.06</td><td>62.6</td><td>0.146</td><td>2.15</td><td>176.2</td></tr><tr><td>BSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.231</td><td>100%</td><td>24.55</td><td>65.2</td><td>0.132</td><td>1.96</td><td>182.1</td></tr><tr><td>Vanilla VQ†</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.422</td><td>0.2%</td><td>21.19</td><td>50.7</td><td>0.209</td><td>5.05</td><td>118.9</td></tr><tr><td>EMA VQ†</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.217</td><td>65.5%</td><td>24.36</td><td>64.1</td><td>0.111</td><td>0.99</td><td>194.3</td></tr><tr><td>Online VQ†</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.280</td><td>13.5%</td><td>23.84</td><td>61.6</td><td>0.130</td><td>1.38</td><td>182.9</td></tr><tr><td>Wasserstein VQ†</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.201</td><td>99.6%</td><td>24.68</td><td>65.4</td><td>0.106</td><td>0.92</td><td>195.5</td></tr><tr><td>MMD VQ†</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.201</td><td>99.9%</td><td>24.65</td><td>65.0</td><td>0.106</td><td>0.86</td><td>197.1</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.233</td><td>59.2%</td><td>24.28</td><td>64.0</td><td>0.114</td><td>1.07</td><td>191.7</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.209</td><td>100%</td><td>24.55</td><td>64.9</td><td>0.107</td><td>0.93</td><td>195.4</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.211</td><td>100%</td><td>24.53</td><td>64.7</td><td>0.108</td><td>0.95</td><td>195.3</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.217</td><td>100%</td><td>24.44</td><td>64.6</td><td>0.110</td><td>0.99</td><td>193.5</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.212</td><td>100%</td><td>24.43</td><td>64.5</td><td>0.109</td><td>0.95</td><td>196.1</td></tr><tr><td>FSQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.272</td><td>100%</td><td>23.87</td><td>61.0</td><td>0.119</td><td>1.23</td><td>186.6</td></tr><tr><td>LFQ BSQ</td><td>SQ SQ</td><td>Adaptation Adaptation</td><td>512 512</td><td>65536 65536</td><td>0.279 0.231</td><td>29.8% 100%</td><td>23.42 24.06</td><td>60.7 63.6</td><td>0.130 0.117</td><td>1.30 1.07</td><td>183.2 190.8</td></tr></table>

Quantization algorithms. We compare VQ, PQ, and SQ. For VQ, we evaluate Vanilla VQ [OVK17], EMA VQ [ROV19], Online VQ [ZV23], Wasserstein VQ [Fan+26a], and MMD VQ [Fan+26b]. For PQ, we use the corresponding Vanilla VP2, EMA VP2, Online VP2, Wasserstein VP2, and MMD VP2 variants. For SQ, we evaluate FSQ [Men+24], LFQ [Yu+24a], and BSQ [ZXK25]. In the primary latent-space comparison, all methods use T = 512 tokens and composite code-space cardinality $K = 6 5 { , } 5 3 6$ , matching both components of the nominal rate $R = T \log _ { 2 } K$ . Full implementation and training details for the latent- and pixel-space experiments are provided in Appendix H.

Evaluation metrics. We report quantization distortion E and codebook utilization U. Reconstruction quality is evaluated using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), LPIPS [Zha+18], reconstruction Fréchet Inception Distance (r-FID) [Heu+17], and reconstruction Inception Score (r-IS) [Sal+16].

## 5.2 Experimental Results

Controlled quantization comparison in latent space. Table 2 reports the primary ImageNet-1K comparison under matched latent feature statistics, token count, and code-space cardinality. A clear distortion hierarchy emerges: the best VQ method achieves lower distortion than the best PQ method, which in turn outperforms the best SQ method. Specifically, MMD VQ attains $\mathcal { E } = 0 . 2 0 1$ , compared with 0.209 for PQ and 0.231 for SQ, consistent with the theoretical ordering $\mathcal { E } _ { \mathrm { V Q } } ^ { \star } \le \mathcal { E } _ { \mathrm { P Q } } ^ { \star } \le \mathcal { E } _ { \mathrm { S Q } } ^ { \star }$ in Section 4.4. Lower distortion also yields better reconstruction after decoder adaptation: MMD VQ achieves the best r-FID of 0.86, compared with 0.93 for PQ and 1.07 for SQ. Meanwhile, several PQ and SQ methods attain 100% codebook utilization despite higher distortion, showing that high utilization alone does not imply stronger quantization efectiveness.

Rate-distortion curves. Figure 2 compares the bestperforming method within each quantization family across diferent coding rates. Consistent with our family-level theoretical comparison, at each nominal per-token rate $\log _ { 2 } K ,$ we report the minimum distortion achieved among the evaluated variants within each quantization family. Increasing the coding rate monotonically reduces distortion for all three quantization families. More importantly, the ordering remains consistent across the evaluated range: VQ achieves the lowest distortion at every operating point, followed by PQ and SQ. This persistent separation across representation

![](images/3fc9b6501924810bd8fa365f44af3661c7b490da7da73f6451d32bbcde9a11d4.jpg)  
Figure 2: Rate–distortion curves on ImageNet-1K, reporting the best result within each quantization family at each rate.

budgets is consistent with the theoretical hierarchy in Section 4.4 and shows that VQ’s distortion advantage is not specific to a particular code-space cardinality.

Distortion versus codebook utilization. We further examine whether distortion or codebook utilization is more strongly associated with reconstruction fidelity. Using Table 2, we compute Spearman’s rank correlations with r-FID. Distortion shows a near-perfect positive correlation with r-FID $( \rho = 0 . 9 9 6 , p < 1 0 ^ { - 8 } )$ , whereas codebook utilization exhibits a substantially weaker negative correlation $( \rho = - 0 . 5 4 0 , p = 0 . 0 5 7 )$ . Thus, distortion is considerably more informative of reconstruc tion fidelity than utilization, supporting the distortion-minimization principle in Section 4.2: high utilization alone does not ensure faithful preservation of the source representation.

Additional rate-distortion analyses. We further analyze the efects of code-space cardinality K, token count T, and matched-rate allocations. Increasing either K or T reduces distortion, while equalrate configurations yield similar but not identical performance. See Appendix I for full results.

Generalization across datasets and representation spaces. We further verify that our findings generalize beyond ImageNet-1K and the pretrained VAR latent space. Under the same controlled latent-space protocol, VQ retains its advantage over PQ and SQ on FFHQ and CelebA-HQ, with distortion remaining more strongly associated with reconstruction fidelity than codebook utilization (Appendix J). A controlled pixel-space comparison on CelebA-HQ yields the same qualitative conclusions, with VQ achieving the lowest distortion and best reconstruction fidelity (Appendix K). These results demonstrate robustness across datasets and representation spaces.

## 6 Conclusion

In this work, we presented a unified rate-distortion perspective for understanding and comparing vector, product, and scalar quantization. We characterize quantization error as distortion and the nominal fixed-length coding budget through token count and discrete code-space cardinality. This perspective yields three main conclusions. First, distortion minimization is a more fundamental optimization principle than codebook utilization: under mild conditions, it implies full utilization and controls the local STE-induced gradient discrepancy. Second, fair intrinsic comparison requires matched latent distributions and nominal coding rates. Third, beyond the classical ordering $\mathcal { E } _ { \mathrm { V Q } } ^ { \star } \leq \mathcal { E } _ { \mathrm { P Q } } ^ { \star } \leq \mathcal { E } _ { \mathrm { S Q } } ^ { \star } ,$ VQ can exploit low-dimensional source structure to achieve substantially better distortion scaling than fixed coordinate-factorized PQ and SQ. Our controlled experiments support these conclusions: distortion is more strongly associated with reconstruction fidelity than codebook utilization, and modern VQ methods achieve lower distortion and better reconstruction performance than PQ and SQ under matched representation budgets. Overall, our results provide a principled framework for evaluating intrinsic quantization efectiveness and for guiding the analysis and design of future discrete representation learning systems.

## Acknowledgements

Dehan Kong was partially supported by the Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant RGPIN-2022-04646.

## References

[Bac+25] R. Bachmann, J. Allardice, D. Mizrahi, E. Fini, O. F. Kar, E. Amirloo, A. El-Nouby, A. Zamir, and A. Dehghan. “Flextok: resampling images into 1d token sequences of flexible length”. In: ICML. 2025 (cit. on p. 7).

[BLS17] J. Ballé, V. Laparra, and E. P. Simoncelli. “End-to-end optimized image compression”. In: ICLR. 2017 (cit. on pp. 1, 5).

[BLC13] Y. Bengio, N. Léonard, and A. C. Courville. “Estimating or propagating gradients through stochastic neurons for conditional computation”. ArXiv (2013) (cit. on pp. 2, 18).

[BW82] J. A. Bucklew and G. L. Wise. “Multidimensional asymptotic quantization theory with r-th power distortion measures”. IEEE Transactions on Information Theory (1982) (cit. on p. 18).

[Cha+26] Y. Chang, J. Qin, L. Qiao, X. Wang, Z. Zhu, L. Ma, and X. Wang. “Scalable training for vector-quantized networks with 100% codebook utilization”. In: ICLR. 2026 (cit. on pp. 1–3).

[Den+09] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. “Imagenet: a large-scale hierarchical image database”. In: CVPR. 2009 (cit. on pp. 8, 21).

[Dha+20] P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, and I. Sutskever. “Jukebox: a generative model for music”. ArXiv (2020) (cit. on pp. 2, 3, 6).

[ERO21] P. Esser, R. Rombach, and B. Ommer. “Taming transformers for high-resolution image synthesis”. In: CVPR. 2021 (cit. on pp. 1, 3–5).

[Fan+26a] X. Fang, L. Guo, H. Chen, Y. Zhang, X. Xia, D. Song, Y. xin Liu, H. Wang, H. Yang, Q. Sun, and Y. Yuan. “Distributional matching for vector quantization: a unified theoretical and empirical framework”. ArXiv (2026) (cit. on pp. 2, 9, 21, 22).

[Fan+26b] X. Fang, Y. Yuan, D. Kong, and T. G. J. Rudner. “VQ-transplant: eficient plug-and-play vq-module integration for pre-trained visual tokenizers”. In: ICLR. 2026 (cit. on pp. 2, 8, 9, 21, 22).

[Fif+25] C. Fifty, R. G. Junkins, D. Duan, A. Iger, J. W. Liu, E. Amid, S. Thrun, and C. R’e. “Restructuring vector quantization with the rotation trick”. In: ICLR. 2025 (cit. on pp. 2, 3).

[Ge+13] T. Ge, K. He, Q. Ke, and J. Sun. “Optimized product quantization for approximate nearest neighbor search”. In: CVPR. 2013 (cit. on p. 18).

[GG92] A. Gersho and R. M. Gray. Vector Quantization and Signal Compression. 1992 (cit. on p. 18).

[GN98] R. M. Gray and D. L. Neuhof. “Quantization”. IEEE Transactions on Information Theory (1998) (cit. on p. 18).

[Guo+24] H. Guo, F. Xie, D. Yang, H. Lu, X. Wu, and H. M. Meng. “Addressing index collapse of large-codebook speech tokenizer with dual-decoding product-quantized variational auto-encoder”. Arxiv (2024) (cit. on pp. 1, 2, 4).

[Han+25] J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu. “Infinity: scaling bitwise autoregressive modeling for high-resolution image synthesis”. In: CVPR. 2025 (cit. on p. 1).

[Heu+17] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. “Gans trained by a two time-scale update rule converge to a local nash equilibrium”. In: NeurIPS. 2017 (cit. on p. 9).

[Huh+23] M. Huh, B. Cheung, P. Agrawal, and P. Isola. “Straightening out the straight-through estimator: overcoming optimization challenges in vector quantized networks”. In: ICML. 2023 (cit. on pp. 2, 3).

[JDS11] H. Jégou, M. Douze, and C. Schmid. “Product quantization for nearest neighbor search”. TPAMI (2011) (cit. on pp. 1, 4).

[Jia+26] S. Jiang, W. Long, M. Han, Z. Chen, C. Zhu, and S. Gu. “Diferentiable vector quantization for rate-distortion optimization of generative image compression”. In: CVPR. 2026 (cit. on pp. 1, 5).

[Kar+17] T. Karras, T. Aila, S. Laine, and J. Lehtinen. “Progressive growing of gans for improved quality, stability, and variation”. ArXiv (2017) (cit. on pp. 21, 24, 25).

[KLA19] T. Karras, S. Laine, and T. Aila. “A style-based generator architecture for generative adversarial networks”. In: CVPR. 2019 (cit. on pp. 21, 24).

[Lee+22] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han. “Autoregressive image generation using residual quantization”. In: CVPR. 2022 (cit. on pp. 1–5).

[LHB25] E. Lei, H. Hassani, and S. S. Bidokhti. “Approaching rate-distortion limits in neural compression with lattice transform coding”. In: ICLR. 2025 (cit. on pp. 1, 2).

[Li+25] X. Li, K. Qiu, H. Chen, J. Kuen, J. Gu, B. Raj, and Z. Lin. “Imagefolder: autoregressive image generation with folded tokens”. In: ICLR. 2025 (cit. on pp. 1, 4, 5).

[LH19] I. Loshchilov and F. Hutter. “Decoupled weight decay regularization”. In: ICLR. 2019 (cit. on p. 21).

[Lu+26] H. Lu, O. C. Koyun, Y. Guo, Z. Zhu, A. Alili, and M. N. Gürcan. “Beyond stationarity: rethinking codebook collapse in vector quantization”. ArXiv (2026) (cit. on pp. 1–3).

[Ma+25] C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi. “Unitok: a unified tokenizer for visual generation and understanding”. ArXiv (2025) (cit. on pp. 4, 5).

[Men+24] F. Mentzer, D. C. Minnen, E. Agustsson, and M. Tschannen. “Finite scalar quantization: vq-vae made simple”. In: ICLR. 2024 (cit. on pp. 1–4, 9, 22).

[NF13] M. Norouzi and D. J. Fleet. “Cartesian k-means”. In: CVPR. 2013 (cit. on p. 18).

[OVK17] A. van den Oord, O. Vinyals, and K. Kavukcuoglu. “Neural discrete representation learning”. In: NeurIPS. 2017 (cit. on pp. 3, 9, 21, 22).

[Ram+25] V. Ramanujan, K. Tirumala, A. Aghajanyan, L. S. Zettlemoyer, and A. Farhadi. “When worse is better: navigating the compression-generation tradeof in visual tokenization”. In: NeurIPS. 2025 (cit. on p. 20).

[Ram+21] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever. “Zero-shot text-to-image generation”. In: ICML. 2021 (cit. on p. 6).

[ROV19] A. Razavi, A. van den Oord, and O. Vinyals. “Generating diverse high-fidelity images with vq-vae-2”. In: NeurIPS. 2019 (cit. on pp. 3, 9, 21, 22).

[RFB15] O. Ronneberger, P. Fischer, and T. Brox. “U-net: convolutional networks for biomedical image segmentation”. In: MICCAI. 2015 (cit. on p. 21).

[Sal+16] T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. “Improved techniques for training gans”. In: NeurIPS. 2016 (cit. on p. 9).

[Shi+25] F. Shi, Z. Luo, Y. Ge, Y. Yang, Y. Shan, and L. Wang. “Taming scalable visual tokenizer for autoregressive image generation”. In: ICCV. 2025 (cit. on pp. 1–3).

[Sun+24] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. “Autoregressive model beats difusion: llama for scalable image generation”. ArXiv (2024) (cit. on pp. 1, 3–5, 21).

[Tak+22] Y. Takida, T. Shibuya, W.-H. Liao, C.-H. Lai, J. Ohmura, T. Uesaka, N. Murata, S. Takahashi, T. Kumakura, and Y. Mitsufuji. “Sq-vae: variational bayes on discrete representation with self-annealed stochastic quantization”. In: ICML. 2022 (cit. on pp. 2, 3).

[Tia+24] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. “Visual autoregressive modeling: scalable image generation via next-scale prediction”. In: NeurIPS. 2024 (cit. on pp. 1, 3–5, 8, 21).

[TRZ17] M. S. Tomar, M. Rungger, and M. Zamani. “Invariance feedback entropy of uncertain control systems”. IEEE Transactions on Automatic Control (2017) (cit. on p. 6).

[TL99] H. Touchette and S. Lloyd. “Information-theoretic limits of control”. Physical review letters (1999) (cit. on p. 6).

[TL01] H. Touchette and S. Lloyd. “Information-theoretic approach to the study of control systems”. Physica A-statistical Mechanics and Its Applications (2001) (cit. on p. 6).

[Wil+20] W. Williams, S. Ringer, T. Ash, J. Hughes, D. Macleod, and J. Dougherty. “Hierarchical quantized autoencoders”. In: NeurIPS. 2020 (cit. on pp. 3, 6).

[YW25] J. Yao and X. Wang. “Reconstruction vs. generation: taming optimization dilemma in latent difusion models”. In: CVPR. 2025 (cit. on p. 20).

[Yu+22] J. Yu, X. Li, J. Y. Koh, H. Zhang, R. Pang, J. Qin, A. Ku, Y. Xu, J. Baldridge, and Y. Wu. “Vector-quantized image modeling with improved vqgan”. In: ICLR. 2022 (cit. on pp. 2, 3).

[Yu+24a] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, B. Gong, M.-H. Yang, I. Essa, D. A. Ross, and L. Jiang. “Language model beats difusion - tokenizer is key to visual generation”. In: ICLR. 2024 (cit. on pp. 1, 2, 4, 7, 9, 22).

[Yu+24b] Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L.-C. Chen. “An image is worth 32 tokens for reconstruction and generation”. In: NeurIPS. 2024 (cit. on p. 7).

[Zad82] P. L. Zador. “Asymptotic quantization error of continuous signals and the quantization dimension”. IEEE Transactions on Information Theory (1982) (cit. on p. 18).

[Zha+23] J. Zhang, F. Zhan, C. Theobalt, and S. Lu. “Regularized vector quantization for tokenized image synthesis”. In: CVPR. 2023 (cit. on pp. 3, 6).

[Zha+18] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. “The unreasonable efectiveness of deep features as a perceptual metric”. In: CVPR. 2018 (cit. on p. 9).

[ZW24] X. Zhang and X. Wu. “Learning optimal lattice vector quantizers for end-to-end neural image compression”. In: NeurIPS. 2024 (cit. on pp. 1, 2).

[ZXK25] Y. Zhao, Y. Xiong, and P. Krähenbühl. “Image and video tokenization with binary spherical quantization”. In: ICLR. 2025 (cit. on pp. 1, 2, 4, 9, 22).

[ZV23] C. Zheng and A. Vedaldi. “Online clustered codebook”. In: ICCV. 2023 (cit. on pp. 1–3, 6, 9, 21, 22).

[Zhu+24a] L. Zhu, F. Wei, Y. Lu, and D. Chen. “Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99%”. In: NeurIPS. 2024 (cit. on pp. 1, 3–7).

[Zhu+24b] Y. Zhu, B. Li, Y. Xin, and L. Xu. “Addressing representation collapse in vector quantized models with one linear layer”. ArXiv (2024) (cit. on pp. 1–3).

## A Proof of Proposition 2

We prove the relationship between distortion minimization and codebook utilization. Let X be a random variable supported on $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ . A deterministic quantizer with codebook size K is a measurable mapping $f : \mathcal { X } \to \{ 1 , \dots , K \}$ . Given a reconstruction mapping $\hat { x } : \{ 1 , \dotsc , K \}  \mathbb { R } ^ { d }$ define the squared-error distortion as

$$
\mathcal { E } ( f , \hat { x } ) : = \mathbb { E } \big [ \| X - \hat { x } ( f ( X ) ) \| _ { 2 } ^ { 2 } \big ] .
$$

For a fixed quantizer $f ,$ the optimal reconstruction is given by the conditional mean ${ \hat { x } } _ { f } ( k ) =$ $\mathbb { E } [ X \mid f ( X ) = k ]$ , for every codeword k with $\mathbb { P } ( f ( X ) = k ) > 0$ . Substituting the optimal reconstruction yields

$$
\mathcal { E } ( f ) : = \operatorname* { m i n } _ { \hat { x } } \mathcal { E } ( f , \hat { x } ) = \mathbb { E } \left[ \| X - \mathbb { E } [ X \mid f ( X ) ] \| _ { 2 } ^ { 2 } \right] .
$$

Equivalently, ${ \mathcal { E } } ( f ) = \mathbb { E } [ \operatorname { V a r } ( X \mid f ( X ) ) ]$ , where for a vector-valued random variable,

$$
\operatorname { V a r } ( X \mid f ( X ) ) : = \mathbb { E } \left[ \| X - \mathbb { E } [ X \mid f ( X ) ] \| _ { 2 } ^ { 2 } \mid f ( X ) \right] .
$$

Proof. Part (a). We prove by contradiction.

Suppose that $f ^ { \star }$ minimizes distortion but does not use all codewords. Then there exists an unused codeword $k _ { \mathrm { n e w } }$ such that

$$
\mathbb { P } ( f ^ { \star } ( X ) = k _ { \mathrm { n e w } } ) = 0 .
$$

Since no zero-distortion quantizer exists, we must have $\mathcal { E } ( f ^ { \star } ) > 0$ . Therefore, there exists at least one quantization cell

$$
A ^ { \star } = \{ x \in \mathcal { X } : f ^ { \star } ( x ) = k ^ { \star } \}
$$

with positive probability and nonzero conditional variance.

By Assumption 1, there exists a measurable partition

$$
A ^ { \star } = A _ { 1 } \dot { \cup } A _ { 2 }
$$

such that both $A _ { 1 }$ and $A _ { 2 }$ have positive probability and

$$
\mu _ { i } = \mathbb { E } [ X \mid X \in A _ { i } ] , \qquad \mu _ { 1 } \neq \mu _ { 2 } .
$$

Construct a refined quantizer $f ^ { \prime }$ by splitting $A ^ { \star }$ :

$$
f ^ { \prime } ( x ) = \left\{ \begin{array} { l l } { k ^ { \star } , } & { x \in A _ { 1 } , } \\ { k _ { \mathrm { n e w } } , } & { x \in A _ { 2 } , } \\ { f ^ { \star } ( x ) , } & { x \notin A ^ { \star } . } \end{array} \right.
$$

For all cells other than $A ^ { \star }$ , the distortion contribution remains unchanged. Within $A ^ { \star }$ , the law of total variance gives

$$
\operatorname { V a r } ( X \mid X \in A ^ { \star } ) = \sum _ { i = 1 } ^ { 2 } \mathbb { P } ( A _ { i } \mid A ^ { \star } ) \operatorname { V a r } ( X \mid X \in A _ { i } ) + \sum _ { i = 1 } ^ { 2 } \mathbb { P } ( A _ { i } \mid A ^ { \star } ) \| \mu _ { i } - \mu \| _ { 2 } ^ { 2 } ,
$$

where $\mu = \mathbb { E } [ X \mid X \in A ^ { \star } ]$

Since $\mu _ { 1 } \neq \mu _ { 2 }$ , the second term is strictly positive. Hence

$$
\sum _ { i = 1 } ^ { 2 } \mathbb { P } ( A _ { i } \mid A ^ { \star } ) \operatorname { V a r } ( X \mid X \in A _ { i } ) < \operatorname { V a r } ( X \mid X \in A ^ { \star } ) .
$$

Therefore, $\mathcal { E } ( f ^ { \prime } ) < \mathcal { E } ( f ^ { \star } )$ , contradicting the optimality of $f ^ { \star }$ . Hence every distortion-minimizing quantizer must use all K codewords: $\mathrm { U } ( f ^ { \star } ) = 1$

Part (b). We construct a full-utilization quantizer that is not distortion-optimal. Let $K = 2$ and let $X \sim \mathrm { U n i f } [ 0 , 1 ]$ . Consider the quantizer

$$
f _ { 0 } ( x ) = { \left\{ \begin{array} { l l } { 1 , } & { x \in [ 0 , 1 / 2 ] , } \\ { 2 , } & { x \in ( 1 / 2 , 1 ] . } \end{array} \right. }
$$

Its optimal reconstructions are

$$
\hat { x } _ { f _ { 0 } } ( 1 ) = { \frac { 1 } { 4 } } , \qquad \hat { x } _ { f _ { 0 } } ( 2 ) = { \frac { 3 } { 4 } } .
$$

Hence

$$
{ \mathcal E } ( f _ { 0 } ) = \int _ { 0 } ^ { 1 / 2 } \left( x - \frac { 1 } { 4 } \right) ^ { 2 } d x + \int _ { 1 / 2 } ^ { 1 } \left( x - \frac { 3 } { 4 } \right) ^ { 2 } d x = \frac { 1 } { 4 8 } .
$$

Now define another quantizer

$$
g ( x ) = { \left\{ \begin{array} { l l } { 1 , } & { x \in [ 0 , 9 / 1 0 ] , } \\ { 2 , } & { x \in ( 9 / 1 0 , 1 ] . } \end{array} \right. }
$$

Both codewords are used, so $\mathrm { U } ( g ) = 1$

The optimal reconstructions are

$$
\hat { x } _ { g } ( 1 ) = { \frac { 9 } { 2 0 } } , \qquad \hat { x } _ { g } ( 2 ) = { \frac { 1 9 } { 2 0 } } .
$$

Therefore,

$$
{ \mathcal { E } } ( g ) = \int _ { 0 } ^ { 9 / 1 0 } \left( x - { \frac { 9 } { 2 0 } } \right) ^ { 2 } d x + \int _ { 9 / 1 0 } ^ { 1 } \left( x - { \frac { 1 9 } { 2 0 } } \right) ^ { 2 } d x = { \frac { 7 3 } { 1 2 0 0 } } .
$$

Since

$$
{ \frac { 7 3 } { 1 2 0 0 } } > { \frac { 1 } { 4 8 } } ,
$$

we obtain

$$
\operatorname* { m i n } _ { f : \mathcal { X } \to \{ 1 , 2 \} } \mathcal { E } ( f ) \leq \mathcal { E } ( f _ { 0 } ) < \mathcal { E } ( g ) .
$$

Thus $g$ uses all codewords but is not distortion-optimal.

## B Proof of Proposition 3

We first prove the upper bound by constructing an $\mathsf { S Q }$ scheme. Let $n = | K ^ { 1 / d } |$ . Since $K \geq 2 ^ { d }$ , we have $K ^ { 1 \bar { / } d } \geq 2$ and hence $n \geq K ^ { 1 \lceil d } / 2$ . For each coordinate $m \in [ d ]$ , partition $[ - 1 , 1 ]$ into n intervals of equal length $2 / n$ and choose their midpoints as scalar reconstruction levels,

$$
\phi _ { m } = \left\{ - 1 + \frac { 2 i - 1 } { n } : i = 1 , \ldots , n \right\} .
$$

Let $\phi = \textstyle \bigoplus _ { m = 1 } ^ { d } \phi _ { m }$ . Then $| \phi | = n ^ { d } \leq K$ . For any $x \in [ - 1 , 1 ] ^ { d } .$ , nearest-neighbor scalar quantization satisfies $| x _ { m } - Q _ { \phi _ { m } } ( x _ { m } ) | \leq 1 / n$ for every coordinate. Therefore,

$$
\begin{array} { r l } & { \mathbb { E } \bigl [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \bigr ] \leq \underset { x \in [ - 1 , 1 ] ^ { d } } { \operatorname* { s u p } } \| x - \mathcal { Q } _ { \phi } ( x ) \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq \frac { d } { n ^ { 2 } } \leq \frac { 4 d } { K ^ { 2 / d } } . } \end{array}
$$

Since this construction is valid for every $\mathbb { P } \in \mathcal { P }$

$$
\operatorname* { s u p } _ { \mathbb { P } \in \mathcal { P } } \mathcal { E } _ { \mathrm { S Q } } ^ { \ast } ( \mathbb { P } , K ) \leq \frac { 4 d } { K ^ { 2 / d } } .
$$

We next prove the lower bound using the uniform distribution $\mathbb { P } = \mathrm { U n i f } ( [ - 1 , 1 ] ^ { d } )$ . Let $\phi = \{ e _ { 1 } , \ldots , e _ { K } \}$ be an arbitrary VQ codebook, and let $A _ { 1 } , \ldots , A _ { K }$ be its nearest-neighbor cells intersected with $[ - 1 , 1 ] ^ { d }$ Write $v _ { k } = | A _ { k } |$ and let $V _ { d }$ denote the volume of the Euclidean unit ball in $\mathbb { R } ^ { d }$ . For any measurable set $A \subseteq \mathbb { R } ^ { d }$ of volume v and any $e \in \mathbb { R } ^ { d }$ , the centered Euclidean ball of volume v minimizes the second moment about $e ,$ so

$$
\int _ { A } \| x - e \| _ { 2 } ^ { 2 } d x \geq \frac { d } { d + 2 } V _ { d } ^ { - 2 / d } v ^ { 1 + 2 / d } .
$$

Consequently,

$$
\begin{array} { r l r } & { } & { \mathbb { E } \big [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \big ] = \displaystyle \frac { 1 } { 2 ^ { d } } \sum _ { k = 1 } ^ { K } \int _ { A _ { k } } \| x - e _ { k } \| _ { 2 } ^ { 2 } d x \quad \quad } \\ & { } & { \geq \displaystyle \frac { 1 } { 2 ^ { d } } \frac { d } { d + 2 } { V _ { d } ^ { - 2 / d } } \sum _ { k = 1 } ^ { K } v _ { k } ^ { 1 + 2 / d } . } \end{array}
$$

Because $\textstyle \sum _ { k = 1 } ^ { K } v _ { k } = 2 ^ { d }$ , convexity gives

$$
\sum _ { k = 1 } ^ { K } v _ { k } ^ { 1 + 2 / d } \geq K \left( { \frac { 2 ^ { d } } { K } } \right) ^ { 1 + 2 / d } = 2 ^ { d + 2 } K ^ { - 2 / d } .
$$

Thus

$$
\mathbb { E } \big [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \big ] \geq \frac { 4 d } { d + 2 } V _ { d } ^ { - 2 / d } K ^ { - 2 / d } .
$$

Finally, $V _ { d } ^ { 2 / d } \leq 3 2 / ( d + 2 )$ for every integer $d \geq 1$ . For $d = 1 , 2$ this follows directly from $V _ { 1 } = 2$ and $V _ { 2 } = \pi ;$ for $d \geq 3 ,$ , it follows from the standard bound $V _ { d } \leq ( 2 \pi e / d ) ^ { d / 2 }$ and $2 \pi e / d \leq 3 2 / ( d + 2 )$ . Hence

$$
\mathbb { E } \bigl [ \| X - \mathcal { Q } _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \bigr ] \geq \frac { d } { 8 } K ^ { - 2 / d } .
$$

Since the codebook $\phi$ was arbitrary,

$$
\operatorname* { s u p } _ { \mathbb { P } \in \mathcal { P } } \mathcal { E } _ { \mathrm { V Q } } ^ { \ast } ( \mathbb { P } , K ) \geq \frac { d } { 8 K ^ { 2 / d } } .
$$

Combining this with $\mathcal { E } _ { \mathrm { V Q } } ^ { \ast } ( \mathbb { P } , K ) \leq \mathcal { E } _ { \mathrm { S Q } } ^ { \ast } ( \mathbb { P } , K )$ completes the proof.

## C Proof of Proposition 4

We first prove the VQ upper bound. By assumption, there exists a $d _ { \mathrm { e f f } } .$ -dimensional linear subspace $L \subseteq \mathbb { R } ^ { d }$ such that P is supported on $L \cap [ - 1 , 1 ] ^ { d }$ . Let $U \in \mathbb { R } ^ { d \times d _ { \mathrm { e f f } } }$ have orthonormal columns spanning L. For $X \sim \mathbb { P }$ , write $X = U Z$ with $Z = U ^ { \top } X$ , and let $\widetilde { \mathbb { P } }$ denote the distribution of $Z .$ Since $U$ is an isometry on $\mathbb { R } ^ { d _ { \mathrm { e f f } } }$

$$
\| Z \| _ { 2 } = \| U Z \| _ { 2 } = \| X \| _ { 2 } \leq { \sqrt { d } } ,
$$

and hence $\| Z \| _ { \infty } \leq { \sqrt { d } } .$ . Therefore, $\widetilde { \mathbb { P } }$ is supported on $[ - \sqrt { d } , \sqrt { d } ] ^ { d _ { \mathrm { e f f } } }$

Let $n = \lfloor K ^ { 1 / d _ { \mathrm { e f f } } } \rfloor$ . Because $K \ge 2 ^ { d _ { \mathrm { e f f } } }$ , we have $n \ge K ^ { 1 / d _ { \mathrm { e f f } } } / 2$ . Quantize each coordinate of $Z$ independently using n equally spaced intervals on $[ - \sqrt { d } , \sqrt { d } ]$ and their midpoints as reconstruction levels. The resulting Cartesian-product codebook contains $\dot { n } ^ { d _ { \mathrm { e f f } } } \leq K$ points, and every coordinate incurs squared error at most $d / \bar { n ^ { 2 } }$ . Consequently,

$$
\mathbb { E } _ { Z \sim \widetilde { \mathbb { P } } } \big [ \| Z - \mathcal { Q } _ { \widetilde { \phi } } ( Z ) \| _ { 2 } ^ { 2 } \big ] \le \frac { d d _ { \mathrm { e f f } } } { n ^ { 2 } } \le \frac { 4 d d _ { \mathrm { e f f } } } { K ^ { 2 / d _ { \mathrm { e f f } } } } .
$$

Mapping every reconstruction point e˜ back to $U \tilde { e } \in L$ preserves Euclidean distances. Thus the same distortion is achievable by a VQ codebook in $\mathbb { R } ^ { d }$ , proving

$$
\mathcal { E } _ { \mathrm { V Q } } ^ { * } ( \mathbb { P } , K ) \leq \frac { 4 d d _ { \mathrm { e f f } } } { K ^ { 2 / d _ { \mathrm { e f f } } } } .
$$

We next prove the PQ and SQ lower bounds. Consider

$$
L = \{ \alpha \mathbf { 1 } : \alpha \in \mathbb { R } \} , \qquad \mathbf { 1 } = ( 1 , \ldots , 1 ) ^ { \top } \in \mathbb { R } ^ { d } ,
$$

and let $X = A \mathbf { 1 }$ , where $A \sim \mathrm { U n i f } [ - 1 , 1 ]$ . Then X is supported on $L \cap [ - 1 , 1 ] ^ { d }$

We use the following elementary one-dimensional fact. If $A \sim \mathrm { U n i f } [ - 1 , 1 ]$ , then every quantizer with at most n reconstruction values satisfies

$$
\mathbb { E } \big [ ( A - \widehat { A } ) ^ { 2 } \big ] \geq \frac { 1 } { 3 n ^ { 2 } } .\tag{C.1}
$$

Indeed, for a nearest-neighbor scalar quantizer, let the lengths of its nonempty Voronoi cells be $\ell _ { 1 } , \ldots , \ell _ { r }$ , where $r \leq n$ and $\textstyle \sum _ { j = 1 } ^ { r } \ell _ { j } = 2$ . Replacing the reconstruction value in each cell by its midpoint can only decrease squared error, so

$$
\mathbb { E } \big [ ( A - \widehat { A } ) ^ { 2 } \big ] \geq \frac { 1 } { 2 } \sum _ { j = 1 } ^ { r } \frac { \ell _ { j } ^ { 3 } } { 1 2 } = \frac { 1 } { 2 4 } \sum _ { j = 1 } ^ { r } \ell _ { j } ^ { 3 } \geq \frac { 1 } { 2 4 } \frac { ( \sum _ { j = 1 } ^ { r } \ell _ { j } ) ^ { 3 } } { r ^ { 2 } } \geq \frac { 1 } { 3 n ^ { 2 } } ,
$$

where the penultimate inequality follows from convexity.

Now consider an arbitrary coordinate-block PQ quantizer with M blocks of dimensions $d _ { 1 } , \ldots , d _ { M }$ where $d _ { m } \geq 1$ and $\textstyle \sum _ { m = 1 } ^ { M } d _ { m } = d$ . Let the m-th sub-codebook contain $n _ { m }$ points, with $\textstyle \prod _ { m = 1 } ^ { M } n _ { m } \leq K$ The input to block m is $A \mathbf { 1 } _ { d _ { m } }$ . For any reconstruction point in $\mathbb { R } ^ { d _ { m } }$ , orthogonally projecting it onto the line span $\left( \mathbf { 1 } _ { d _ { m } } \right)$ cannot increase its distance to $A \mathbf { 1 } _ { d _ { m } }$ . Hence an $n _ { m }$ -point vector quantizer for this block cannot have smaller distortion than the optimal $n _ { m }$ -level scalar quantizer for $A ,$ scaled by $\| \mathbf { 1 } _ { d _ { m } } \| _ { 2 } ^ { 2 } = d _ { m }$ . Using (C.1),

$$
 { \mathbb E } \left[ \| A \mathbf { 1 } _ { d _ { m } } - \boldsymbol { \mathcal Q } _ { \phi _ { m } } ( A \mathbf { 1 } _ { d _ { m } } ) \| _ { 2 } ^ { 2 } \right] \geq \frac { d _ { m } } { 3 n _ { m } ^ { 2 } } \geq \frac { d _ { m } } { 4 n _ { m } ^ { 2 } } .
$$

Summing over the mutually orthogonal coordinate blocks gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \| X - Q _ { \phi } ( X ) \| _ { 2 } ^ { 2 } \right] \geq \frac { 1 } { 4 } \sum _ { m = 1 } ^ { M } \frac { d _ { m } } { n _ { m } ^ { 2 } } } \\ { \displaystyle \geq \frac { M } { 4 } \left( \prod _ { m = 1 } ^ { M } \frac { d _ { m } } { n _ { m } ^ { 2 } } \right) ^ { 1 / M } } \\ { \displaystyle \geq \frac { M } { 4 } \left( \prod _ { m = 1 } ^ { M } n _ { m } \right) ^ { - 2 / M } } \\ { \displaystyle \geq \frac { M } { 4 } K ^ { - 2 / M } . } \end{array}
$$

Here the second line is AM–GM, and the third uses $d _ { m } \geq 1$ . Because the PQ codebooks and rate allocation $( n _ { 1 } , \ldots , n _ { M } )$ were arbitrary, taking the infimum proves

$$
\mathcal { E } _ { \mathrm { P Q } } ^ { * } ( \mathbb { P } , K , M ) \geq \frac { M } { 4 } K ^ { - 2 / M } , \qquad 2 \leq M \leq d .
$$

Finally, SQ is the case $M = d$ and $d _ { m } = 1$ for every block, yielding

$$
\mathcal { E } _ { \mathrm { S Q } } ^ { * } ( \mathbb { P } , K ) \geq \frac { d } { 4 } K ^ { - 2 / d } .
$$

This completes the proof.

## D Relationship to Classical Quantization Theory

The $K ^ { - 2 / d }$ distortion scaling in Proposition 3 has the same high-rate exponent as Zador’s foundational asymptotic quantization theorem [Zad82] and subsequent multidimensional extensions [BW82], as surveyed by Gray and Neuhof [GN98]. These classical results characterize optimal quantization distortion for regular d-dimensional sources in the high-rate regime. Proposition 3, by contrast, is a finite-K worst-case result over distributions supported on a bounded domain. Similarly, under the quantizer families defined in Section 4.4, the nested ordering $\mathcal { E } _ { \mathrm { V Q } } ^ { \ast } \leq \mathcal { E } _ { \mathrm { P Q } } ^ { \ast } \leq \mathcal { E } _ { \mathrm { S Q } } ^ { \ast }$ follows directly from the structural inclusion ${ \mathrm { S Q } } \subseteq { \mathrm { P Q } } \subseteq { \mathrm { V Q } } { \mathrm { : } }$ SQ is the fully factorized special case of PQ, while a PQ codebook is a Cartesian-product-structured VQ codebook. This relationship is standard in classical treatments of vector and product quantization [GG92; GN98]. Proposition 3 therefore serves as a self-contained finite-rate reference point within our framework rather than as a novel asymptotic quantization result.

The main theoretical distinction of our analysis lies in Proposition 4, which provides the explicit, non-asymptotic lower bound $\textstyle \mathcal { E } _ { \mathrm { P O } } ^ { * } ( P , K , M ) \ge \frac { M } { 4 } K ^ { - 2 / M }$ for the fixed coordinate-block PQ family on an axis-misaligned one-dimensional source. This yields a quantitative VQ–PQ separation whose scaling depends explicitly on the number of blocks M. Prior work has established that PQ distortion can depend strongly on the choice of subspace decomposition and can be improved by optimizing the orientation or decomposition of the feature space in approximate nearest-neighbor search [Ge+13; NF13]. To our knowledge, however, this literature does not provide the explicit finite-K, block-countdependent lower bound established here.

Our framework also extends beyond classical distortion analysis in two directions. Appendix E connects quantization distortion to the STE-induced gradient gap, relating quantization error to the optimization dynamics of learned discrete representations. In addition, our empirical study compares VQ, PQ, and SQ under matched source representations, token counts, and composite codespace cardinalities, providing a controlled visual-tokenization setting for examining the theoretical relationships above.

## E Distortion Minimization and STE Gradient Discrepancy

In this section, we derive a local relationship between quantization distortion and the gradient discrepancy induced by the Straight-Through Estimator (STE) [BLC13]. The analysis shows that, under standard local smoothness, the gradient discrepancy vanishes quadratically with the quantization distance, providing theoretical support for distortion minimization as a means of improving the local fidelity of the STE gradient.

Problem Setup: Let L denote the loss as a function of the latent representation. In the quantization process, the encoder outputs a continuous latent vector $z _ { e } ,$ which is discretized into a quantized latent representation $z _ { q }$ . During backpropagation, the non-diferentiable quantization operation is bypassed using the STE, which propagates the gradient evaluated at the quantized representation:

$$
\mathrm { S T E } \mathrm { A p p r o x i m a t i o n : } \quad \frac { \partial \mathcal { L } } { \partial z _ { e } } \gets \frac { \partial \mathcal { L } } { \partial z _ { q } } .\tag{E.1}
$$

We define the gradient gap $\mathcal { G }$ as the squared discrepancy between the gradients evaluated at the continuous and quantized latent representations:

$$
\mathcal { G } \triangleq \left. \frac { \partial \mathcal { L } } { \partial z _ { e } } - \frac { \partial \mathcal { L } } { \partial z _ { q } } \right. _ { 2 } ^ { 2 } .\tag{E.2}
$$

Local Taylor Analysis: Assume that $\mathcal { L }$ is twice diferentiable in a neighborhood of $z _ { q }$ . Let

$$
\mathbf { H } ( z _ { q } ) = \left. \frac { \partial ^ { 2 } \mathcal { L } } { \partial x ^ { 2 } } \right| _ { x = z _ { q } }
$$

denote the Hessian at $z _ { q } . \mathrm { A }$ first-order expansion of the gradient around $z _ { q }$ gives

$$
\frac { \partial \mathcal { L } } { \partial z _ { e } } = \frac { \partial \mathcal { L } } { \partial z _ { q } } + \mathbf { H } ( z _ { q } ) ( z _ { e } - z _ { q } ) + o ( \| z _ { e } - z _ { q } \| _ { 2 } ) .\tag{E.3}
$$

Therefore,

$$
\frac { \partial \mathcal { L } } { \partial z _ { e } } - \frac { \partial \mathcal { L } } { \partial z _ { q } } = \mathbf { H } ( z _ { q } ) ( z _ { e } - z _ { q } ) + o ( \| z _ { e } - z _ { q } \| _ { 2 } ) ,\tag{E.4}
$$

and hence

$$
\mathcal { G } = \| \mathbf { H } ( z _ { q } ) ( z _ { e } - z _ { q } ) + o ( \| z _ { e } - z _ { q } \| _ { 2 } ) \| _ { 2 } ^ { 2 } = \mathcal { O } ( \| z _ { e } - z _ { q } \| _ { 2 } ^ { 2 } ) .\tag{E.5}
$$

Moreover, the leading-order term is

$$
\mathcal { G } = \| \mathbf { H } ( z _ { q } ) ( z _ { e } - z _ { q } ) \| _ { 2 } ^ { 2 } + o ( \| z _ { e } - z _ { q } \| _ { 2 } ^ { 2 } ) .\tag{E.6}
$$

Since $\| z _ { e } - z _ { q } \| _ { 2 } ^ { 2 }$ is exactly the squared quantization distortion, Equation E.5 establishes a local connection between distortion and the STE-induced gradient discrepancy: as the quantization distortion approaches zero, the gradient gap also vanishes at least quadratically in the quantization distance. Thus, reducing distortion makes the gradient evaluated at the quantized representation locally closer to that evaluated at the continuous representation. This provides a theoretical explanation for why distortion reduction can improve the fidelity of gradient propagation through the STE.

## F Efect of Latent Scale on Quantization Distortion

We provide both a theoretical analysis and an empirical verification of the efect of latent feature scale on optimal quantization distortion. The result motivates the matched-distribution condition introduced in Section 4.3.

Theoretical analysis. Let the optimal K-point vector-quantization distortion under squared Euclidean distance be

$$
\mathcal { E } ^ { \star } ( X , K ) = \operatorname* { i n f } _ { \mathcal { C } : | \mathcal { C } | = K } \mathbb { E } \left[ \operatorname* { m i n } _ { c \in \mathcal { C } } \| X - c \| _ { 2 } ^ { 2 } \right] .\tag{F.1}
$$

Proposition 5 (Scale dependence of optimal distortion). For any scalar $a \in \mathbb { R } ,$

$$
{ { \mathcal { E } } ^ { \star } } ( a X , K ) = a ^ { 2 } { { \mathcal { E } } ^ { \star } } ( X , K ) .\tag{F.2}
$$

Proof. For $a \neq 0 _ { \mathrm { { i } } }$ , the mapping

$$
\mathcal { C } ^ { \prime } \mapsto a \mathcal { C } ^ { \prime } = \{ a c ^ { \prime } : c ^ { \prime } \in \mathcal { C } ^ { \prime } \}
$$

is a bijection over the set of all K-point codebooks. Hence,

$$
\mathcal { E } ^ { \star } ( a X , K ) = \operatorname* { i n f } _ { \substack { c \colon | c | = K } } \mathbb { E } \left[ \operatorname* { m i n } _ { c \in \mathcal { C } } \| a X - c \| _ { 2 } ^ { 2 } \right]\tag{F.3}
$$

$$
= \operatorname* { i n f } _ { \mathcal { C } ^ { \prime } : | \mathcal { C } ^ { \prime } | = K } \mathbb { E } \left[ \operatorname* { m i n } _ { c ^ { \prime } \in \mathcal { C } ^ { \prime } } \| a X - a c ^ { \prime } \| _ { 2 } ^ { 2 } \right]\tag{F.4}
$$

$$
= a ^ { 2 } \operatorname* { i n f } _ { \mathcal { C } ^ { \prime } : | \mathcal { C } ^ { \prime } | = K } \mathbb { E } \left[ \operatorname* { m i n } _ { c ^ { \prime } \in \mathcal { C } ^ { \prime } } \| X - c ^ { \prime } \| _ { 2 } ^ { 2 } \right]\tag{F.5}
$$

$$
= a ^ { 2 } { \mathcal { E } } ^ { \star } ( X , K ) .\tag{F.6}
$$

For $a = 0 ,$ both sides are zero, completing the proof.

Proposition 5 shows that the optimal squared quantization distortion is homogeneous of degree two with respect to global scaling of the source distribution. Importantly, this result does not rely on a Gaussian assumption or a high-rate approximation.

A useful special case follows by writing a location-scale family as

$$
X = \mu + \sigma Z ,\tag{F.7}
$$

where $Z$ has a fixed reference distribution. Since translation can be absorbed by an equal translation of the codebook,

$$
{ \mathcal { E } } ^ { \star } ( \mu + \sigma Z , K ) = \sigma ^ { 2 } { \mathcal { E } } ^ { \star } ( Z , K ) .\tag{F.8}
$$

Thus, when the shape of the latent distribution is fixed and only its scale varies, the optimal distortion grows linearly with the source variance $\sigma ^ { 2 }$

## G Limitations

Our results should be interpreted within several important scope boundaries. First, we study the intrinsic efectiveness of quantization under a nominal fixed-length representation rate, $R = T \log _ { 2 } K$ rather than the expected bitstream length produced by entropy coding. Consequently, our analysis does not directly characterize entropy-constrained quantization, variable-length coding, or adaptive bit allocation, where the probability distribution of discrete symbols also contributes to the efective coding rate. Extending the framework to these settings would provide a closer connection to classical rate-distortion optimization in learned compression.

Second, our theoretical analysis adopts squared Euclidean quantization distortion as the intrinsic information-loss measure. This choice enables a clean characterization of source scaling and of the structural relationships among VQ, PQ, and SQ, but it does not directly cover perceptual, semantic, or task-dependent distortion measures. Moreover, the theoretical distortion hierarchy concerns the optimal achievable distortion within each quantizer family; it does not imply that every practical VQ optimization algorithm will outperform every PQ or SQ method under finite-data and finiteoptimization regimes.

Finally, our conclusions concern intrinsic quantization and reconstruction efectiveness, rather than downstream generative modeling performance. Improved reconstruction fidelity does not necessarily imply improved generation quality [Ram+25; YW25], since downstream performance also depends on properties of the resulting discrete representation and on the capacity and optimization of the generative model. Understanding how intrinsic quantization distortion interacts with representation modelability and downstream generation therefore remains an important direction for future work.

## H Experimental Details

Data Preprocessing. For all three datasets, ImageNet-1k [Den+09], FFHQ [KLA19], and CelebA-HQ [Kar+17], we follow Llama Gen [Sun+24] by applying iterative box downsampling to resize all images to a 256×256 resolution.

Encoder-Decoder Architecture. For experiments conducted in the latent space, we adopt the VQ Transplant framework [Fan+26b] and initialize all quantization methods with the same pretrained VAR tokenizer [Tia+24]. This setup fixes the encoder-decoder architecture and latent representation across methods, enabling a controlled comparison of diferent quantization algorithms under the same latent space. The encoder is a U-Net [RFB15] that downsamples input images by a factor of $^ { 1 6 , }$ producing latent features $z _ { e } = \mathcal { E } _ { \theta } ( \pmb { x } ) \in \mathbb { R } ^ { 1 6 \times 1 6 \times 3 2 }$ with a spatial resolution of $1 6 \times 1 6$ . In addition to these latent-space experiments, we also consider a pixel-space setting constructed directly from image pixels without relying on a pretrained encoder-decoder representation.

Training Details. Following [Fan+26b], all latent-space models were trained on two NVIDIA H100 GPUs using the AdamW optimizer [LH19] with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 5$ . During VQ module substitution, we used an initial learning rate of $1 0 ^ { - 4 }$ with linear decay to $1 0 ^ { - 5 }$ , while the learning rate was fixed at $1 0 ^ { - 5 }$ during decoder adaptation. For ImageNet-1k, VQ module substitution and decoder adaptation were performed for 2 and 5 epochs, respectively. For both FFHQ and CelebA-HQ, VQ module substitution and decoder adaptation were each performed for 30 epochs. The batch size was fixed at 32 per GPU for all latent-space experiments. We additionally conducted pixel-space experiments on CelebA-HQ using a single-stage training procedure for 30 epochs. We used the same AdamW optimizer and batch size as in the latent-space experiments, with an initial learning rate of $1 0 ^ { - 4 }$ linearly decayed to $1 0 ^ { - 5 }$

Loss Weights. For the latent-space experiments, we follow the loss formulation and notation of VQ-Transplant [Fan+26b]. Here, $\lambda _ { P }$ and $\lambda _ { G }$ denote the perceptual and adversarial loss weights, respectively, while γ controls the weight of the distribution-matching objective. We set $\lambda _ { P } = 1$ and $\lambda _ { G } = 0 . 4$ . For configurations using the Wasserstein distance (Wasserstein VQ and Wasserstein VP2), we set $\gamma = 0 . 2$ , while for configurations using the MMD distance (MMD VQ and MMD VP2), we set $\gamma = 0 . 5$ . For the pixel-space experiments, the training objective consists of the reconstruction, perceptual, and quantization losses, with the perceptual loss weight fixed to $\lambda _ { P } = 1$ . For Wassersteinand MMD-based quantizers, the corresponding distribution-matching weight γ follows the same setting as in the latent-space experiments.

Latent-Space VQ Implementation. We implement five VQ variants within the VQ-Transplant framework: Vanilla VQ [OVK17], EMA VQ [ROV19], Online VQ [ZV23], Wasserstein VQ [Fan+26a], and MMD VQ [Fan+26b]. We first use the pretrained

![](images/87bf937b12de57238c6296ab7c005921b81593abf7270694449a77e6c38967e2.jpg)  
Figure 3: Illustration of the implementation details.

VAR encoder to extract latent features $z _ { e } = \mathcal { E } _ { \theta } ( \pmb { x } ) \in \mathbb { R } ^ { 1 6 \times 1 6 \times 3 2 }$ . A three-layer convolutional projector then processes $z _ { e }$ while preserving the 32-dimensional feature space. Each 32-dimensional feature vector is partitioned into two 16-dimensional vectors, each of which is treated as an independent quantization token and processed by a separate VQ module. Consequently, the $1 6 \times 1 6$ latent grid yields a total of $T = 5 1 2$ quantization tokens. The two quantized 16-dimensional vectors at each spatial location are subsequently concatenated to recover a 32-dimensional representation, as illustrated in Figure 3. A second three-layer convolutional projector maps the concatenated quantized features to the decoder input. The code-space cardinality of each token is fixed to $K = 6 5 { , } 5 3 6$ , yielding a nominal representation rate of $R = T \log _ { 2 } K$ with $T = 5 1 2$

Latent-Space PQ Implementation. We implement five PQ variants within the VQ-Transplant frame work: Vanilla VP2 [OVK17], EMA VP2 [ROV19], Online VP2 [ZV23], Wasserstein VP2 [Fan+26a], and MMD VP2 [Fan+26b]. The overall architecture and tokenization scheme are identical to those used for VQ, with the only change being the internal quantization structure. Specifically, each 16- dimensional quantization token is further partitioned into $M = 2$ 8-dimensional sub-vectors. The two sub-vectors are independently quantized using separate codebooks of size 256, producing a composite code-space cardinality of $2 5 6 ^ { 2 } = 6 5 { , } 5 3 6$ for each token. Therefore, PQ retains the same number of quantization tokens, $T = 5 1 2 ,$ and the same efective code-space cardinality, $K = 6 5 { , } 5 3 6$ as VQ, ensuring a matched nominal representation rate.

Latent-Space SQ Implementation. We implement three SQ variants within the VQ-Transplant framework: FSQ [Men+24], LFQ [Yu+24a], and BSQ [ZXK25]. The surrounding architecture is kept consistent with the VQ and PQ implementations, while the quantization module is replaced by the corresponding scalar quantizer. For LFQ, each 16-dimensional quantization token is discretized using binary scalar levels, resulting in a code-space cardinality of $2 ^ { 1 6 } = 6 5 { , } 5 3 6$ . BSQ uses the same 16-dimensional token configuration, with normalization applied before binary spherical quantization, likewise yielding $2 ^ { 1 6 } = 6 5 { , } 5 3 6$ possible codes. For FSQ, a three-layer convolutional projector reduces the 32-dimensional feature representation to 16 dimensions, which are organized into two 8-dimensional quantization tokens. Each scalar dimension is discretized into four fixed levels, giving a code-space cardinality of $4 ^ { 8 } = 6 5 { , } 5 3 6$ per token. A symmetric three-layer convolutional projector then maps the quantized representation back to 32 dimensions. Thus, all VQ, PQ, and SQ configurations use $T = 5 1 2$ quantization tokens with an efective code-space cardinality of $K = 6 5 { , } 5 3 6$ per token, ensuring matched nominal representation rate across methods.

Pixel-Space Implementation. In addition to the latent-space experiments, we conduct pixel-space experiments on CelebA-HQ, where quantization is performed without a pretrained encoder-decoder. Given an input image $\pmb { x } \in \mathbb { R } ^ { 2 5 6 \times 2 5 6 \times 3 }$ , we first apply PixelUnshufle with a factor of 4, which deterministically rearranges each 4 × 4 spatial block into the channel dimension and produces a $6 4 \times 6 4 \times 4 8$ representation without discarding information. This representation is then processed by a three-layer convolutional projector that maps the 48 channels to the quantization feature dimension. The projected features are quantized using the corresponding quantization module. Since the resulting feature grid has a spatial resolution of $6 4 \times 6 4 ,$ the pixel-space setting contains $T = 4 0 9 6$ quantization tokens. The code-space cardinality of each token is fixed to $K = 6 5 { , } 5 3 6 ,$ yielding a nominal representation rate of $R = T \log _ { 2 } K$ . After quantization, a symmetric threelayer convolutional projector maps the quantized features back to 48 channels, and PixelShufle with a factor of 4 is applied to reconstruct the image at the original $2 5 6 \times 2 5 6$ resolution. All compared methods in the pixel-space setting share the same PixelUnshufle/PixelShufle operations, convolutional projectors, token count, and code-space cardinality, such that the quantization module remains the primary varying component across methods.

Remark. Our comparisons are controlled separately within the latent-space and pixel-space settings. In the latent-space experiments, all methods operate on the same pretrained VAR encoder output and use $T = 5 1 2$ quantization tokens with a code-space cardinality of $K = 6 5 { , } 5 3 6$ per token. In the pixel-space experiments, all methods share the same PixelUnshufle operation and convolutional projector, and use $T = 4 0 9 6$ quantization tokens with the same code-space cardinality $K = 6 5 { , } 5 3 6$ Therefore, the nominal representation rate $R = T \log _ { 2 } K$ is matched across quantization methods within each setting, satisfying Condition 2. With respect to Condition 1, all compared methods within each setting receive the same source representation before quantizer-specific processing, providing a controlled basis for comparing intrinsic quantization efectiveness.

Table 3: Quantization and reconstruction performance under varying per-token code-space cardinality K with the token count fixed at $T = 5 1 2 .$
<table><tr><td>Methods</td><td>Phase</td><td>Tokens</td><td>Codebook Size K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td><td>r-IS(↑)</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>1024</td><td>0.318</td><td>99.6%</td><td>23.75</td><td>60.8</td><td>0.162</td><td>2.68</td><td>167.6</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>2048</td><td>0.296</td><td>99.4%</td><td>24.12</td><td>62.4</td><td>0.159</td><td>2.59</td><td>168.8</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>4096</td><td>0.273</td><td>99.4%</td><td>24.41</td><td>63.5</td><td>0.141</td><td>1.96</td><td>178.0</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>8192</td><td>0.252</td><td>99.5%</td><td>24.67</td><td>64.6</td><td>0.135</td><td>1.85</td><td>181.0</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>16384</td><td>0.234</td><td>99.8%</td><td>24.89</td><td>65.4</td><td>0.130</td><td>1.84</td><td>183.7</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>32768</td><td>0.215</td><td>99.7%</td><td>25.06</td><td>66.3</td><td>0.126</td><td>1.79</td><td>184.8</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.201</td><td>99.9%</td><td>25.24</td><td>66.8</td><td>0.121</td><td>1.69</td><td>187.3</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>1024</td><td>0.318</td><td>99.6%</td><td>23.06</td><td>58.8</td><td>0.148</td><td>1.90</td><td>169.8</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>2048</td><td>0.296</td><td>99.4%</td><td>23.58</td><td>61.2</td><td>0.137</td><td>1.63</td><td>176.6</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>4096</td><td>0.273</td><td>99.4%</td><td>23.89</td><td>61.8</td><td>0.128</td><td>1.28</td><td>185.1</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>8192</td><td>0.252</td><td>99.5%</td><td>24.11</td><td>62.9</td><td>0.121</td><td>1.18</td><td>187.9</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>16384</td><td>0.234</td><td>99.8%</td><td>24.31</td><td>63.7</td><td>0.115</td><td>1.05</td><td>191.2</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>32768</td><td>0.215</td><td>99.9%</td><td>24.53</td><td>64.7</td><td>0.110</td><td>0.97</td><td>194.1</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.201</td><td>99.9%</td><td>24.65</td><td>65.0</td><td>0.106</td><td>0.86</td><td>197.1</td></tr></table>

Table 4: Quantization and reconstruction performance under varying token count T with the per-token code-space cardinality fixed at $K = 1 6 { , } 3 8 4 .$
<table><tr><td>Methods</td><td>Phase</td><td>Tokens</td><td>Codebook Size K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td><td>r-IS(↑)</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>256</td><td>16384</td><td>0.369</td><td>99.6%</td><td>22.97</td><td>57.2</td><td>0.194</td><td>4.91</td><td>141.4</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>16384</td><td>0.234</td><td>99.8%</td><td>24.89</td><td>65.4</td><td>0.130</td><td>1.84</td><td>183.7</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>1024</td><td>16384</td><td>0.098</td><td>100%</td><td>26.40</td><td>71.0</td><td>0.100</td><td>2.01</td><td>191.7</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>2048</td><td>16384</td><td>0.035</td><td>100%</td><td>27.16</td><td>73.1</td><td>0.089</td><td>2.36</td><td>192.2</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>256</td><td>16384</td><td>0.369</td><td>99.6%</td><td>22.41</td><td>55.9</td><td>0.171</td><td>3.06</td><td>148.9</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>16384</td><td>0.234</td><td>99.8%</td><td>24.31</td><td>63.7</td><td>0.115</td><td>1.05</td><td>191.2</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>1024</td><td>16384</td><td>0.098</td><td>100%</td><td>26.03</td><td>69.6</td><td>0.079</td><td>0.54</td><td>210.1</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>2048</td><td>16384</td><td>0.035</td><td>100%</td><td>27.31</td><td>73.2</td><td>0.060</td><td>0.42</td><td>217.0</td></tr></table>

Table 5: Comparison of diferent token-count and code-space allocations under matched nominal rate.
<table><tr><td>Methods</td><td>Phase</td><td>Tokens</td><td>Codebook Size K</td><td>Rate R</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td><td>r-IS(↑)</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>16384</td><td>7,168 bits</td><td>0.234</td><td>99.8%</td><td>24.89</td><td>65.4</td><td>0.130</td><td>1.84</td><td>183.7</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>16384</td><td>7,168 bits</td><td>0.234</td><td>99.8%</td><td>24.31</td><td>63.7</td><td>0.115</td><td>1.05</td><td>191.2</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>1024</td><td>128</td><td>7,168 bits</td><td>0.243</td><td>100%</td><td>24.72</td><td>64.9</td><td>0.132</td><td>1.80</td><td>184.5</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>1024</td><td>128</td><td>7,168 bits</td><td>0.243</td><td>100%</td><td>24.01</td><td>63.0</td><td>0.118</td><td>1.10</td><td>190.4</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>8,192 bits</td><td>0.201</td><td>99.9%</td><td>25.24</td><td>66.8</td><td>0.121</td><td>1.69</td><td>187.3</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>8,192 bits</td><td>0.201</td><td>99.9%</td><td>24.65</td><td>65.0</td><td>0.106</td><td>0.86</td><td>197.1</td></tr><tr><td>MMD VQ</td><td>Substitution</td><td>1024</td><td>256</td><td>8,192 bits</td><td>0.212</td><td>100%</td><td>25.11</td><td>66.4</td><td>0.124</td><td>1.67</td><td>187.6</td></tr><tr><td>MMD VQ</td><td>Adaptation</td><td>1024</td><td>256</td><td>8,192 bits</td><td>0.212</td><td>100%</td><td>24.51</td><td>64.6</td><td>0.108</td><td>0.96</td><td>195.3</td></tr></table>

## I Rate–Distortion Analysis of Code-Space Cardinality and Token Count

We first examine the efect of code-space cardinality while fixing the token count at $T \ = \ 5 1 2$ Specifically, we vary the per-token cardinality from $K = 1 , 0 2 4 { \mathrm { ~ t o ~ } } K = 6 5 , 5 3 6$ by successive factors of two. Under the nominal fixed-length rate $R = T \log _ { 2 } K$ , increasing K increases the representation budget. As shown in Table $^ { 3 , }$ the quantization distortion $\mathcal { E }$ decreases monotonically from 0.318 to 0.201, while r-FID after decoder adaptation improves from 1.90 to 0.86. Thus, increasing $K$ at fixed $T$ reduces quantization distortion and improves reconstruction quality.

We next vary the token count from $T = 2 5 6$ to $T = 2 { , } 0 4 8$ while fixing $K = 1 6 { , } 3 8 4$ . As shown in Table $^ { 4 , }$ increasing $T$ reduces distortion from 0.369 to 0.035, while adapted r-FID improves from 3.06 to 0.42. These results show that increasing token count can likewise substantially reduce information loss. The diferent improvement patterns obtained by varying $T$ and $K$ further indicate that the two quantities may afect representation structure diferently despite both contributing to the nominal rate.

Finally, we compare diferent allocations of the same nominal representation rate. Under $R =$ $T \log _ { 2 } K$ , doubling the token count and squaring the per-token code-space cardinality produce the same nominal rate:

$$
2 T \log _ { 2 } K = T \log _ { 2 } K ^ { 2 } ,\tag{I.1}
$$

This identity concerns only the representation budget and does not imply identical distortion. Table 5 compares matched-rate configurations, including $( T , K ) = ( 5 1 2 , 1 6 , 3 8 4 )$ versus (1,024, 128) and (512, 65,536) versus (1,024, 256). These pairs achieve similar, though not identical, distortion and reconstruction performance, consistent with the rate accounting in Section 4.1 and with the fact that diferent rate allocations can induce diferent representation structures.

Table 6: Quantization and reconstruction performance of VQ, PQ, and SQ on FFHQ under matched token count (T = 512) and code-space cardinality (K = 65,536). Within each quantization type and phase (Substitution/Adaptation), optimal values are underlined; overall best results per metric are bold.
<table><tr><td>Approaches</td><td>Types</td><td>Phase</td><td>Tokens</td><td>K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td></tr><tr><td>Vanilla VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.286</td><td>0.2%</td><td>24.20</td><td>66.7</td><td>0.159</td><td>7.16</td></tr><tr><td>EMA VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.146</td><td>52.9%</td><td>27.66</td><td>77.4</td><td>0.075</td><td>2.13</td></tr><tr><td>Online VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.186</td><td>12.4%</td><td>26.98</td><td>75.2</td><td>0.088</td><td>2.41</td></tr><tr><td>Wasserstein VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.132</td><td>99.5%</td><td>28.05</td><td>78.3</td><td>0.070</td><td>2.13</td></tr><tr><td>MMD VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.131</td><td>99.3%</td><td>28.06</td><td>78.3</td><td>0.069</td><td>2.14</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.159</td><td>40.3%</td><td>27.33</td><td>76.7</td><td>0.078</td><td>2.38</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.136</td><td>100%</td><td>27.94</td><td>77.9</td><td>0.071</td><td>2.14</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.137</td><td>100%</td><td>27.89</td><td>77.9</td><td>0.072</td><td>2.43</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.146</td><td>99.9%</td><td>27.64</td><td>77.6</td><td>0.079</td><td>2.95</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.139</td><td>100%</td><td>27.83</td><td>77.9</td><td>0.072</td><td>2.47</td></tr><tr><td>FSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.170</td><td>100%</td><td>27.27</td><td>75.4</td><td>0.081</td><td>3.18</td></tr><tr><td>LFQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.224</td><td>7.2%</td><td>25.62</td><td>71.6</td><td>0.110</td><td>3.70</td></tr><tr><td>BSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.155</td><td>100%</td><td>27.04</td><td>76.7</td><td>0.080</td><td>2.09</td></tr><tr><td>Vanilla VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.286</td><td>0.2%</td><td>23.63</td><td>63.8</td><td>0.142</td><td>2.31</td></tr><tr><td>EMA VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.146</td><td>52.9%</td><td>27.35</td><td>75.9</td><td>0.071</td><td>1.25</td></tr><tr><td>Online VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.186</td><td>12.4%</td><td>26.57</td><td>73.3</td><td>0.085</td><td>1.77</td></tr><tr><td>Wasserstein VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.132</td><td>99.5%</td><td>27.69</td><td>76.5</td><td>0.067</td><td>0.89</td></tr><tr><td>MMD VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.131</td><td>99.3%</td><td>27.75</td><td>76.9</td><td>0.067</td><td>0.85</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.159</td><td>40.3%</td><td>27.11</td><td>75.2</td><td>0.074</td><td>1.33</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.136</td><td>100%</td><td>27.57</td><td>76.1</td><td>0.070</td><td>1.05</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.137</td><td>100%</td><td>27.54</td><td>75.8</td><td>0.069</td><td>1.10</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.146</td><td>99.9%</td><td>27.28</td><td>75.5</td><td>0.072</td><td>1.11</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.139</td><td>100%</td><td>27.58</td><td>76.4</td><td>0.068</td><td>1.16</td></tr><tr><td>FSQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.170</td><td>100%</td><td>26.83</td><td>73.5</td><td>0.076</td><td>1.61</td></tr><tr><td>LFQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.224</td><td>7.2%</td><td>25.23</td><td>69.9</td><td>0.102</td><td>1.74</td></tr><tr><td>BSQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.155</td><td>100%</td><td>26.73</td><td>75.1</td><td>0.078</td><td>1.54</td></tr></table>

## J Generalization Across Datasets

To examine whether the main empirical findings generalize across data distributions, we further evaluate VQ, PQ, and SQ on FFHQ [KLA19] and CelebA-HQ [Kar+17]. We follow the same controlled latent-space protocol described in Appendix H, with T = 512 and K = 65,536 for all methods. Tables 6 and 7 report the results.

Generalization of Quantization Comparisons. The results are consistent with the main experimental findings reported in Table 2. On FFHQ, the best VQ, PQ, and SQ methods achieve distortions of 0.131, 0.136, and 0.155, respectively; after decoder adaptation, their best r-FID values are 0.85, 1.05, and 1.54. On CelebA-HQ, the corresponding best distortions are 0.111, 0.115, and 0.133, while the best r-FID values are 1.73, 1.96, and 2.24. These results show that the advantage of modern VQ methods under controlled feature statistics and matched nominal rate persists across datasets.

Generalization of Distortion–Reconstruction Correlation. We further compute Spearman’s rank correlations between r-FID and quantization distortion E or codebook utilization U. On FFHQ, distortion is strongly correlated with r-FID $( \rho = 0 . 9 7 9 , p = 5 . 4 9 \times 1 0 ^ { - 9 } )$ , whereas utilization shows a much weaker correlation $( \rho = - 0 . 4 9 2 , p = 0 . 0 8 8 )$ . CelebA-HQ exhibits the same pattern, with $\rho = 0 . 9 4 4 ~ ( p = 1 . 2 9 \times 1 0 ^ { - 6 } )$ for distortion and $\rho = - 0 . 5 5 9 \ ( p = 0 . 0 4 7 )$ for utilization. Together with the ImageNet-1K results in the main text, these findings consistently show that distortion is

Table 7: Quantization and reconstruction performance of VQ, PQ, and SQ on CelebA-HQ under matched token count $( T = 5 1 2 )$ and code-space cardinality $( K = 6 5 , 5 3 6 )$ . Within each quantization type and phase (Substitution/Adaptation), optimal values are underlined; overall best results per metric are bold.
<table><tr><td>Approaches</td><td>Types</td><td>Phase</td><td>Tokens</td><td>K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td></tr><tr><td>Vanilla VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.269</td><td>0.1%</td><td>23.39</td><td>64.9</td><td>0.176</td><td>16.2</td></tr><tr><td>EMA VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.131</td><td>31.9%</td><td>27.87</td><td>77.4</td><td>0.071</td><td>3.17</td></tr><tr><td>Online VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.180</td><td>11.6%</td><td>26.82</td><td>74.4</td><td>0.093</td><td>4.44</td></tr><tr><td>Wasserstein VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.111</td><td>99.3%</td><td>28.32</td><td>78.7</td><td>0.064</td><td>2.63</td></tr><tr><td>MMD VQ</td><td>VQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.111</td><td>99.1%</td><td>28.33</td><td>78.5</td><td>0.064</td><td>2.66</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.168</td><td>11.6%</td><td>26.67</td><td>75.1</td><td>0.086</td><td>3.78</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.115</td><td>100%</td><td>28.25</td><td>78.5</td><td>0.065</td><td>3.03</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.119</td><td>99.9%</td><td>28.14</td><td>78.3</td><td>0.066</td><td>3.01</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.123</td><td>99.9%</td><td>27.82</td><td>77.7</td><td>0.068</td><td>2.81</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.120</td><td>100%</td><td>27.99</td><td>78.1</td><td>0.069</td><td>2.92</td></tr><tr><td>FSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.147</td><td>99.9%</td><td>27.58</td><td>75.9</td><td>0.075</td><td>4.64</td></tr><tr><td>LFQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.211</td><td>2.9%</td><td>25.40</td><td>71.1</td><td>0.115</td><td>6.56</td></tr><tr><td>BSQ</td><td>SQ</td><td>Substitution</td><td>512</td><td>65536</td><td>0.133</td><td>100%</td><td>27.28</td><td>77.0</td><td>0.074</td><td>2.58</td></tr><tr><td>Vanilla VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.269</td><td>0.1%</td><td>22.95</td><td>62.4</td><td>0.159</td><td>5.13</td></tr><tr><td>EMA VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.131</td><td>31.9%</td><td>27.53</td><td>75.6</td><td>0.069</td><td>2.34</td></tr><tr><td>Online VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.180</td><td>11.6%</td><td>26.30</td><td>72.0</td><td>0.090</td><td>3.46</td></tr><tr><td>Wasserstein VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.111</td><td>99.3%</td><td>27.98</td><td>76.6</td><td>0.064</td><td>1.73</td></tr><tr><td>MMD VQ</td><td>VQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.111</td><td>99.1%</td><td>27.98</td><td>76.7</td><td>0.063</td><td>1.76</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.168</td><td>11.6%</td><td>26.37</td><td>72.8</td><td>0.083</td><td>2.36</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.115</td><td>100%</td><td>27.78</td><td>76.0</td><td>0.068</td><td>1.96</td></tr><tr><td>Online VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.119</td><td>99.9%</td><td>27.67</td><td>75.3</td><td>0.068</td><td>2.11</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.123</td><td>99.9%</td><td>27.67</td><td>75.9</td><td>0.068</td><td>2.02</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.120</td><td>100%</td><td>27.68</td><td>75.8</td><td>0.066</td><td>2.07</td></tr><tr><td>FSQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.147</td><td>99.9%</td><td>27.16</td><td>73.6</td><td>0.072</td><td>2.24</td></tr><tr><td>LFQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.211</td><td>2.9%</td><td>24.88</td><td>68.1</td><td>0.111</td><td>3.35</td></tr><tr><td>BSQ</td><td>SQ</td><td>Adaptation</td><td>512</td><td>65536</td><td>0.133</td><td>100%</td><td>26.93</td><td>74.9</td><td>0.074</td><td>2.47</td></tr></table>

more strongly associated with reconstruction fidelity than codebook utilization, supporting distortion minimization as a more reliable criterion for intrinsic quantization efectiveness.

## K Controlled Quantization Comparison in Pixel Space

To further examine whether our conclusions depend on the particular latent representation induced by a pretrained encoder, we conduct an additional controlled comparison directly from image pixels on CelebA-HQ [Kar+17]. This experiment removes the dependence on a pretrained latent feature distribution, thereby reducing a potential source of confounding in quantizer comparison. As described in Appendix H, all methods share the same PixelUnshufle operation and convolutional projectors, and are evaluated with the same token count $T = 4 0 9 6$ and code-space cardinality $K = 6 5 { , } 5 3 6$ Thus, the compared methods difer primarily in their quantization modules while operating under the same representation pipeline and nominal rate. Table 8 reports the results.

Quantization and Reconstruction Performance. The pixel-space results are consistent with the latent-space findings. MMD VQ achieves the lowest quantization distortion, $\mathcal { E } = 0 . 0 0 2 1$ , compared with 0.0023 for the best PQ variant and 0.0032 for the best SQ variant. It also achieves the best r-FID of 3.64, compared with 3.77 for the best PQ method and 4.54 for the best SQ method. More broadly, modern VQ variants achieve strong reconstruction performance despite PQ and SQ methods often attaining higher codebook utilization. These results indicate that the relative advantage of VQ is not specific to the pretrained latent space used in the main experiments.

Distortion–Reconstruction Correlation. We further compute Spearman’s rank correlations between r-FID and quantization distortion E or codebook utilization U. Distortion exhibits a strong positive correlation with r-FID $( \rho = 0 . 9 4 7 , p = 2 . 9 1 \times 1 0 ^ { - 6 } )$ , whereas codebook utilization shows

Table 8: Quantization and reconstruction performance of VQ, PQ, and SQ methods on CelebA-HQ under the controlled pixel-space setting with matched token count $( T = 4 0 9 6 )$ and code-space cardinality $( K = 6 5 , 5 \bar { 3 } 6 )$ . Within each quantization type, the best results are underlined, while the overall best result for each metric is shown in bold. Results for Online VQ are omitted because the experiment could not be completed due to out-of-memory errors.
<table><tr><td>Approaches</td><td>Types</td><td>Tokens</td><td>K</td><td>ε(↓)</td><td>U (↑)</td><td>PSNR(↑)</td><td>SSIM(↑)</td><td>LPIPS (↓)</td><td>r-FID(↓)</td></tr><tr><td>Vanilla VQ</td><td>VQ</td><td>4096</td><td>65536</td><td>0.0055</td><td>0.2%</td><td>30.99</td><td>86.1</td><td>0.085</td><td>8.45</td></tr><tr><td>EMA VQ</td><td>VQ</td><td>4096</td><td>65536</td><td>0.0033</td><td>4.2%</td><td>33.33</td><td>89.4</td><td>0.057</td><td>6.51</td></tr><tr><td>Online VQ</td><td>VQ</td><td>4096</td><td>65536</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wasserstein VQ</td><td>VQ</td><td>4096</td><td>65536</td><td>0.0022</td><td>96.0%</td><td>35.32</td><td>92.9</td><td>0.026</td><td>3.75</td></tr><tr><td>MMD VQ</td><td>VQ</td><td>4096</td><td>65536</td><td>0.0021</td><td>98.5%</td><td>35.37</td><td>92.9</td><td>0.023</td><td>3.64</td></tr><tr><td>Vanilla VP2</td><td>PQ</td><td>4096</td><td>65536</td><td>0.0024</td><td>100%</td><td>34.71</td><td>93.0</td><td>0.025</td><td>3.93</td></tr><tr><td>EMA VP2</td><td>PQ</td><td>4096</td><td>65536</td><td>0.0027</td><td>100%</td><td>34.12</td><td>90.7</td><td>0.057</td><td>6.37</td></tr><tr><td>Online VP2</td><td>PQ</td><td>4096</td><td>65536</td><td>0.0023</td><td>100%</td><td>34.96</td><td>92.7</td><td>0.027</td><td>4.00</td></tr><tr><td>Wasserstein VP2</td><td>PQ</td><td>4096</td><td>65536</td><td>0.0024</td><td>99.2%</td><td>34.88</td><td>92.5</td><td>0.027</td><td>4.21</td></tr><tr><td>MMD VP2</td><td>PQ</td><td>4096</td><td>65536</td><td>0.0023</td><td>100%</td><td>35.15</td><td>92.9</td><td>0.025</td><td>3.77</td></tr><tr><td>FSQ</td><td>SQ</td><td>4096</td><td>65536</td><td>0.0032</td><td>100%</td><td>33.40</td><td>91.8</td><td>0.031</td><td>4.54</td></tr><tr><td>LFQ</td><td>SQ</td><td>4096</td><td>65536</td><td>0.0050</td><td>10.1%</td><td>31.27</td><td>87.2</td><td>0.071</td><td>7.99</td></tr><tr><td>BSQ</td><td>SQ</td><td>4096</td><td>65536</td><td>0.0041</td><td>99.8%</td><td>32.15</td><td>90.1</td><td>0.039</td><td>4.75</td></tr></table>

a substantially weaker and statistically insignificant negative correlation $( \rho = - 0 . 4 1 3 , p = 0 . 1 8 2 )$ This closely mirrors the latent-space results and provides additional evidence that distortion is more strongly associated with reconstruction fidelity than codebook utilization. Overall, the pixel-space experiments support the same conclusions as the latent-space analysis while reducing dependence on a particular pretrained feature space.