# Bound-Conditioned Latent Inference for Progressive Image Compression

Jaeseok Jang<sup>†</sup> Korea University jsjang@mcl.korea.ac.kr

Kwang Pyo Choi Samsung Electronics kp5.choi@samsung.com

Seungmin Jeon<sup>†</sup> Korea University seungminjeon@mcl.korea.ac.kr

Chang-Su Kim<sup>∗</sup> Korea University changsukim@korea.ac.kr

## Abstract

Progressive image compression reconstructs images from prefixes of a single bitstream and therefore involves sequential latent inference under partial observations. Existing trit-plane codecs mainly treat decoded trits as discrete symbols for probability prediction, without fully exploiting their role as interval constraints on the underlying continuous latents. We propose bound-conditioned latent inference (BLI), a framework that reformulates progressive trit-plane coding as sequential latent inference under shrinking interval constraints. At each coding step, the trits decoded so far define lower and upper bounds for each latent element. BLI uses the center and width of these bounds, together with hyperprior information shared by the encoder and decoder, to jointly refine the Gaussian parameters $( \mu , \sigma )$ for entropy modeling and the latent estimate yˆ for reconstruction. As additional trits are decoded, the feasible intervals shrink monotonically, yielding tighter constraints for the next inference step and enabling flexible in-plane refinement schedules. Experiments on Kodak, CLIC, and JPEG-AI show that BLI achieves state-ofthe-art rate-distortion performance among progressive image codecs, reducing BD-rate by 17.78% over DPICT and by 7.96% over CTC on Kodak, while replacing CTC’s plane-specific predictors with a shared model that uses about 5.3× fewer parameters and achieves lower latency than CTC.

## 1 Introduction

Progressive image compression differs from standard fixed-rate compression in that it allows the decoder to reconstruct an image from prefixes of a single bitstream. At an intermediate decoding step, only part of the latent representation has been transmitted, while the unresolved latent values must be inferred from partial observations. Thus, progressive compression is not merely a rate-distortion (R-D) optimization problem, but also a sequential latent inference problem under partial observations. This property is essential for adaptive streaming, bandwidth-limited transmission, and storage systems that require multiple quality levels from a single bitstream.

Traditional and learned codecs have achieved strong R-D performance, from hand-crafted transform codecs such as JPEG and JPEG2000 (Wallace, 1992; Skodras et al., 2001) to neural codecs with learned transforms, hyperpriors, and context models (Agustsson et al., 2017; Ballé et al., 2017, 2018; Minnen et al., 2018; He et al., 2021). However, most high-performance learned codecs are optimized for a single target rate. Multirate codecs (Cui et al., 2021; Yang et al., 2021; Xu et al., 2025) improve rate flexibility, but generally require separately encoded bitstreams for different rates, in contrast to progressive codecs, every bitstream prefix of which remains decodable.

![](images/439f5e6ebb849a8d893097f79168d2198d2e50be7eb04892e753f35c660b5058.jpg)  
Figure 1: Rate-allocation maps of the proposed bound-conditioned latent inference (BLI) codec. Each column corresponds to the same trit-plane level, i.e., the same degree of latent quantization, while each row uses a different number of in-plane refinement steps. The top row encodes an entire trit-plane in a single step, whereas the middle and bottom rows divide each trit-plane into two and four substeps, respectively. Colors indicate spatial rate allocation, with warmer colors corresponding to higher coding rates and cooler colors to lower rates. More frequent bound-conditioned updates iteratively refine the latent distribution within the same quantization level, yielding better rate efficiency without changing the final trit-plane precision.

Learned progressive codecs provide decodable bitstream prefixes by decomposing latent representations into successively refinable components. PLONQ (Lu et al., 2021) uses nested quantization grids for coarse-to-fine refinement, while DPICT (Lee et al., 2022) discretizes each latent element into trits that are transmitted sequentially. CTC (Jeon et al., 2023) further improves trit-plane coding by introducing context models that refine ternary probabilities and partial reconstructions. However, existing trit-plane codecs still primarily treat decoded trits as discrete symbols for probability prediction and reconstruction update. In particular, CTC improves discrete trit prediction and partial reconstruction, but it does not explicitly perform in-plane continuous latent inference conditioned on the feasible intervals induced by previously decoded trits. The key observation behind our approach is that decoded trits are not merely discrete symbols: in trit-plane coding, they also restrict each underlying continuous latent element to a feasible interval, which can serve as an explicit constraint for latent inference.

We present bound-conditioned latent inference (BLI), a framework that explicitly exploits these feasible intervals by reformulating progressive trit-plane coding as sequential latent inference under shrinking interval constraints. In BLI, each partially decoded latent element is modeled as a continuous variable constrained by lower and upper bounds determined by the trits decoded so far. At each coding step, BLI takes the center and width of the current feasible interval, together with hyperprior information shared by the encoder and decoder, and jointly refines the Gaussian parameters (µ, σ) for entropy modeling and the latent estimate yˆ for reconstruction. As additional trits are decoded, the feasible intervals shrink monotonically, yielding tighter constraints for the next inference step. As illustrated in Figure 1, this bound-conditioned formulation naturally supports flexible in-plane transmission schedules: each trit-plane can be encoded at once or divided into finer substeps, allowing the model to update the latent distribution and latent estimate more frequently. Conceptually, BLI can be interpreted as constraint-consistent latent refinement, related to projection-based feasibility methods such as POCS (Bauschke and Borwein, 1996) and to denoising or inpainting under interval constraints. Unlike exact metric projection, however, BLI uses a learned amortized estimator trained directly for entropy modeling and reconstruction quality.

Our main contributions are summarized as follows:

• We reformulate progressive trit-plane coding as sequential latent inference under shrinking interval constraints, where decoded trits define feasible ranges for the underlying continuous latent elements.

• We propose BLI, a learned constrained inference framework that jointly refines entropy-model parameters and latent estimates from bound and hyperprior information, with flexible in-plane autoregressive updates within each trit-plane.

• Experiments show that BLI achieves state-of-the-art R-D performance among progressive image codecs, reducing BD-rate by 17.78% over DPICT and by 7.96% over CTC on the Kodak dataset, while replacing CTC’s plane-specific prediction networks with a shared model that uses about 5.3× fewer parameters and achieves lower latency than CTC.

## 2 Related Work

Learned image compression: Traditional codecs (Wallace, 1992; Bellard, 2014; Bross et al., 2021) follow the transform coding paradigm (Goyal, 2001), while learned codecs replace hand-crafted transforms and entropy models with neural networks. Early learned codecs introduced differentiable quantization and end-to-end R-D optimization (Agustsson et al., 2017; Ballé et al., 2017), and hyperprior-based models improved entropy estimation by transmitting side information about latent distributions (Ballé et al., 2018). Later methods further developed expressive priors, including Gaussian mixture and asymmetric Gaussian models (Cheng et al., 2020; Cui et al., 2021).

Entropy models improve probability estimation by exploiting dependencies among latent elements. Autoregressive context models (Minnen et al., 2018; Mentzer et al., 2018; Lee et al., 2019) improve compression efficiency through sequential prediction, but incur high coding latency due to limited parallelism. Partially parallel models (Minnen and Singh, 2020; He et al., 2021) reduce this cost by exploiting latent correlations more efficiently.

Recent learned codecs incorporate attention mechanisms and transformers into analysis-synthesis networks and entropy models. Transformer-based codecs (Zhu et al., 2022; Zou et al., 2022; Liu et al., 2023; Li et al., 2024) improve global context modeling and yield compact latents, while transformerbased entropy models (Qian et al., 2022; Kim et al., 2022) capture long-range dependencies for more accurate probability estimation. These advances provide strong fixed-rate baselines, but most of them are not designed to produce a single progressively decodable bitstream.

Scalable image compression: Beyond fixed-rate learned codecs, scalable compression aims to support multiple bitrates using a single trained model or a single embedded bitstream. Variable-rate methods adjust the operating rate by modulating internal representations, such as gain parameters, channel width, or spatial transforms (Cui et al., 2021; Yang et al., 2021; Song et al., 2021), or by using adjustable quantizers (Chen and Ma, 2020; Tong et al., 2023; Xu et al., 2025). Although these methods provide flexible rate control from a single model, each target bitrate generally requires a separately encoded bitstream. Therefore, they differ from progressive codecs, in which every prefix of a single bitstream remains decodable at multiple quality levels.

Progressive image compression uses embedded bitstreams, where reconstruction quality improves as additional bits are appended. Classical codecs such as progressive JPEG (Wallace, 1992), JPEG2000 (Skodras et al., 2001), and SPIHT (Said and Pearlman, 1996) achieve this through spectral selection, embedded bitstreams, or significance-driven coefficient ordering. In learned compression, PLONQ (Lu et al., 2021) introduced nested quantization for coarse-to-fine refinement, while DPICT (Lee et al., 2022) realized fine-grained scalability by slicing latents into trit-planes. CTC (Jeon et al., 2023) further improved ternary probability modeling and partial reconstructions. However, DPICT and CTC still primarily model decoded trits as discrete symbols. As a result, the feasible intervals induced by decoded trits are not explicitly exploited for in-plane continuous latent inference, which is the key limitation addressed by BLI.

## 3 Proposed Algorithm

We present bound-conditioned latent inference (BLI) for progressive trit-plane coding. BLI treats each partially decoded latent element as a continuous variable constrained by lower and upper bounds determined by the trits decoded so far. Within these bounds, BLI jointly refines the Gaussian parameters $( \mu , \sigma )$ for entropy modeling and the latent estimate yˆ for reconstruction as the bounds shrink over coding steps. Unlike CTC (Jeon et al., 2023), which improves ternary probability modeling and partial reconstruction through discrete context prediction, BLI performs bound-conditioned Gaussian inference at every refinement step. This enables iterative in-plane refinement of both the latent distribution and latent estimate, improving entropy modeling accuracy and reconstruction quality.

![](images/73c60f9d5596ab358b6d24efd26ebc85b709b03d2af847af5c02b5a1aac227f8.jpg)  
Figure 2: Overview of the BLI process. At coding step $k ,$ the trits decoded so far define bound tensors $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } )$ , which represent the current feasible ranges of latent elements. Given these bounds and the hyperprior parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ , BLI estimates the Gaussian parameters $\left( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } \right)$ for entropy coding and the latent tensor $\hat { \mathbf { Y } } ^ { k - 1 }$ for reconstruction under the current bounds. After a subset of trits is decoded using $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ , the bounds are updated to $( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ , yielding tighter feasible ranges that, in turn, are used in the next coding step. This step-wise process iteratively refines the latent distribution and reconstruction.

## 3.1 Bound-Conditioned Latent Inference

Suppose that an image X is transformed into a latent tensor $\mathbf { Y } \in \mathbb { R } ^ { C \times H \times W }$ , as illustrated in Figure 2. Each element $y _ { i }$ in $\mathbf { Y }$ is represented by trits and progressively transmitted. BLI proceeds in coding steps indexed by a global counter $k = 1 , 2 , \cdots$ , which increases monotonically across trit-planes. At step $k ,$ , we define the following state variables.

$( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ : Gaussian mean and standard deviation parameters used for entropy coding at step $k .$

• ${ \hat { \mathbf { Y } } } ^ { k } ;$ : Reconstructed latent estimate consistent with the bounds after step $k .$

$( \mathbf { L } ^ { k } , \mathbf { U } ^ { k } )$ : Lower and upper bounds of the latent elements after step k.

$( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ : Centers and widths of the feasible intervals after step $k ,$ defined as

$$
\mathbf { C } ^ { k } = \frac { \mathbf { L } ^ { k } + \mathbf { U } ^ { k } } { 2 } , \quad \mathbf { D } ^ { k } = \mathbf { U } ^ { k } - \mathbf { L } ^ { k } .\tag{1}
$$

Note that a hyperprior entropy model initializes $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ , while $( \mathbf { L } ^ { 0 } , \mathbf { U } ^ { 0 } )$ , and equivalently $( \mathbf { C } ^ { 0 } , \mathbf { D } ^ { 0 } )$ , define the initial feasible support of the latent elements.

At step $k ,$ , the current feasible interval of $y _ { i }$ is given by $\mathcal { T } _ { i } ^ { k - 1 } = [ \mathbf { L } _ { i } ^ { k - 1 } , \mathbf { U } _ { i } ^ { k - 1 } ]$ . This interval is equally divided into three sub-intervals $\varmathcal { T } _ { i , t } ^ { k - 1 }$ for $t = 0 , 1 , 2$ , corresponding to the next undetermined trit at location $i .$ BLI takes the current bound tensors $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } )$ and the hyperprior parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ as inputs, and estimates the Gaussian parameters $( \mathbf { M } ^ { k } , \dot { \Sigma } ^ { k } )$ for entropy coding at step k. If $y _ { i }$ lies in the $t _ { i }$ -th sub-interval, the corresponding trit $t _ { i }$ is encoded by an arithmetic coder using

$$
- \log _ { 2 } P ( y _ { i } \in \mathcal { T } _ { i , t _ { i } } ^ { k - 1 } | \mathbf { M } _ { i } ^ { k } , \Sigma _ { i } ^ { k } )\tag{2}
$$

bits. Accordingly, BLI is trained to estimate $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ by minimizing the rate loss

$$
\begin{array} { r } { \mathcal { L } _ { R } ^ { k } = { - \sum _ { i } \log _ { 2 } P ( y _ { i } \in \mathcal { T } _ { i , t _ { i } } ^ { k - 1 } \mid \mathbf { M } _ { i } ^ { k } , \pmb { \Sigma } _ { i } ^ { k } ) } . } \end{array}\tag{3}
$$

In addition to the Gaussian parameters, BLI estimates a latent reconstruction within the current bounds by predicting a residual correction to the interval center:

$$
\begin{array} { r } { \hat { \mathbf { Y } } ^ { k - 1 } = \mathbf { C } ^ { k - 1 } + \Delta \hat { \mathbf { Y } } ^ { k - 1 } . } \end{array}\tag{4}
$$

Here, the indices reflect the timing of the coding step; at step $k ,$ the Gaussian parameters $\left( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } \right)$ are used to encode or decode the next subset of trits, whereas $\hat { \mathbf Y } ^ { k - 1 }$ is the latent estimate consistent with the bounds $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } )$ available prior to coding that subset. After the subset is coded and the bounds are updated, the corresponding ${ \hat { \mathbf { Y } } } ^ { k }$ is estimated and used for reconstruction. The Gaussian parameters and latent estimates are produced from shared attention features but through separate prediction heads. The distortion loss is then defined as

![](images/8fbf347dce5e5a686c210c0bdbb93cfaa0a27b90bddc94fc22bcb985de6663b8.jpg)  
Figure 3: Illustration of the BLI refinement process using a 1D sequence of latent elements. Given the current bounds $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } )$ , BLI estimates $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ for entropy coding and $\hat { \mathbf { Y } } ^ { k - 1 }$ for reconstruction under those bounds. After selected trits are decoded, the bounds are updated to $( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ , yielding tighter feasible intervals for the next BLI step. Red dots denote original latent values, blue dots denote latent estimates, and blue densities denote estimated Gaussian distributions

$$
\mathcal { L } _ { D } ^ { k } = \| \mathbf { X } - \hat { \mathbf { X } } ^ { k } \| _ { 2 } ^ { 2 } = \| \mathbf { X } - g _ { s } ( \hat { \mathbf { Y } } ^ { k } ) \| _ { 2 } ^ { 2 } .\tag{5}
$$

The subset of trits is coded using $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ , and the decoded trits update the bounds from $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } ) \mathrm { t o } \left( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } \right)$ .

Figure 3 illustrates the refinement process using a 1D sequence of latent elements. At step k, the decoded trits define interval constraints $\mathcal { T } _ { i } ^ { k - 1 } = [ \mathbf { L } _ { i } ^ { k - 1 } , \mathbf { \bar { U } } _ { i } ^ { k - 1 } ]$ for each latent element, and BLI performs learned constrained inference within these intervals. As additional trits are decoded, the intervals shrink monotonically, yielding tighter constraints for the next inference step. This process is conceptually related to denoising or inpainting under range constraints and to projection-based feasibility methods such as POCS (Bauschke and Borwein, 1996), since each step refines the estimates while enforcing consistency with known constraints. Unlike exact metric projection, however, BLI uses a learned amortized estimator trained for entropy modeling and reconstruction quality.

## 3.2 BLI Codec Architecture

Unlike previous trit-plane codecs (Lee et al., 2022; Jeon et al., 2023) that reuse the initial Gaussian parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { \hat { 0 } } )$ throughout progressive decoding, BLI performs step-wise autoregressive refinement. Each trit-plane is divided into multiple coding steps, and both the Gaussian parameters $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ and the latent estimate $\hat { \mathbf { Y } } ^ { k - 1 }$ are inferred within the current feasible bounds at step k.

Before entropy coding at step k, the BLI module operates within the feasible interval ${ \mathcal { T } } ^ { k - 1 } =$ $[ \mathbf { L } ^ { k - 1 } , \mathbf { U } ^ { k - 1 } ]$ determined by the trits decoded so far. As shown in Figure 4, the BLI module receives the interval center $\mathbf { C } ^ { k - 1 }$ , the half-width $\mathbf { D } ^ { k - 1 } / 2$ , and the hyperprior parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ as inputs. A stack of Swin attention blocks (Liu et al., 2021) extracts bound-conditioned features by modeling spatial dependencies among partially decoded latent elements. These features are then processed by CNN blocks and plane-wise MLP heads to estimate $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ and $\hat { \mathbf Y } ^ { k - 1 }$ . The plane-wise MLP heads allow the module to adapt its prediction to each trit-plane level.

For the estimation of $\mathbf { M } ^ { k }$ and $\hat { \mathbf { Y } } ^ { k - 1 }$ , the CNN features are additionally concatenated with $\mathbf { C } ^ { k - 1 }$ and $\mathbf { D } ^ { k - 1 } / 2$ , since the interval center provides a natural anchor and the half-width specifies the allowable residual range. The outputs for $\mathbf { M } ^ { k }$ and $\hat { \mathbf { Y } } ^ { k - 1 }$ are processed through a tanh activation and rescaled by $\mathbf { D } ^ { k - 1 } / 2$ around $\mathbf { C } ^ { k - 1 }$ , ensuring consistency with the feasible interval $\mathcal { T } ^ { k - 1 }$ . As illustrated in Figures 2 and 3, the estimates at step k are used to code the next subset of trits, after which the bounds are updated to $( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ . As the decoded trits progressively tighten the feasible bounds, this step-wise refinement improves the accuracy of entropy parameter estimation and reconstruction.

![](images/ab40697cf5ff6fceed78742748662b0896640ef7b8eb363cc215b1e96edb356f.jpg)  
Figure 4: Architecture of the BLI module. BLI predicts the Gaussian parameters $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ and the latent estimate $\hat { \mathbf Y } ^ { k - 1 }$ from the current bounds and hyperprior parameters.

## 3.3 Implementation

We embed the proposed BLI framework into FTIC (Li et al., 2024), a state-of-the-art learned image codec. The original entropy model in FTIC performs channel-wise autoregressive prediction for standard fixed-rate decoding. This design is not directly compatible with trit-plane coding, where each latent element may be only partially decoded at intermediate steps. We therefore replace the channel-wise autoregressive entropy model with a non-autoregressive hyperprior variant that directly estimates the initial Gaussian parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ from the hyper latent tensor Z.

Entropy coding is performed using the ANS algorithm (Duda, 2013). We train the codecs on the Flickr2W dataset (Liu et al., 2020) using $2 5 6 \times 2 5 6$ patches, a batch size of 4, and the AdamW optimizer (Loshchilov and Hutter, 2019). Training is conducted on an NVIDIA RTX A6000 GPU.

The BLI codec is optimized in two stages. First, we train the modified FTIC backbone at the highest bitrate point. It is initialized from pretrained FTIC weights and optimized for 1.0M iterations with a learning rate of $1 0 ^ { - 4 }$ using the corresponding Lagrangian multiplier. This trained backbone is used as the common basis for BLI and for the DPICT (Lee et al., 2022) and CTC (Jeon et al., 2023) baselines, with the additional training scheme of Jeon et al. (2023) applied to CTC.

Second, we optimize the BLI codec using the following combined loss:

$$
\mathcal { L } = \sum _ { l \in \mathcal { K } _ { \mathrm { e n d } } } \alpha _ { l } \Big ( \sum _ { k = 1 } ^ { l } \mathcal { L } _ { R } ^ { k } + \lambda \mathcal { L } _ { D } ^ { l } \Big ) ,\tag{6}
$$

where $\kappa _ { \mathrm { e n d } }$ denotes the set of endpoint step indices of the trit-planes. $\mathcal { L } _ { R } ^ { k }$ and $\mathcal { L } _ { D } ^ { l }$ are defined in (3) and (5), respectively. The rate term is accumulated up to each endpoint step $l \in \mathcal { K } _ { \mathrm { e n d } }$ , so that all coding steps up to the corresponding trit-plane endpoint contribute to the loss. The weight α balances the contributions of endpoint losses at different trit-plane levels, because the corresponding quantization intervals have different scales; its detailed setting is provided in Appendix B.2. We set $\bar { \lambda } = 0 . 0 4 8 3$ and optimize BLI for 0.5M iterations with a learning rate of $5 \times 1 0 ^ { - 5 }$

For transmission scheduling, we use random trit assignments during training and the R-D prioritized transmission order of Lee et al. (2022) during evaluation. During training, each trit-plane is divided into four steps, $i . e . , K _ { \mathrm { e n d } } = \{ 4 , 8 , 1 2 , 1 6 , 2 0 \bar  \}$ , with trits randomly assigned across steps to expose BLI to diverse partial-observation patterns. During evaluation, we use $\mathcal { K } _ { \mathrm { e n d } } = \{ 4 , 8 , 1 4 , 3 0 , 5 4 \}$ since the trit-planes closer to the LST typically contain more high-entropy trits and therefore benefit more from finer subdivision. Further implementation details are provided in Appendix B.

## 4 Experimental Results

We evaluate the proposed BLI codec on the Kodak Lossless dataset (Kodak, 1999), the CLIC 2021 validation set (CLIC, 2021), and the JPEG-AI test set (JPEG-AI, 2020). Kodak contains 24 images at

![](images/ee5e6215cb116fa368048e5e1369bba9d56f721edd853d810f495d7a627b5298.jpg)

![](images/fecfbee52a9f9c9d7e7b1ac815c3663c8f148cb7ff567014b3f3de326b2ccc23.jpg)  
Figure 5: R-D curves on Kodak against traditional, fixed-rate learned, and progressive learned codecs. The left plot reports PSNR, and the right plot reports MS-SSIM in dB.

Table 1: PSNR BD-rates (%) with respect to FTIC-DPICT.
<table><tr><td colspan="2"></td><td>Kodak</td><td>CLIC</td><td>JPEG-AI</td></tr><tr><td rowspan="3">Traditional</td><td>JPEG2000 (Skodras et al., 2001)</td><td>47.54</td><td>58.41</td><td>55.73</td></tr><tr><td>BPG444 (Bellard, 2014)</td><td>-7.92</td><td>0.21</td><td>0.74</td></tr><tr><td>VTM 12.0 (Bross et al., 2021)</td><td>-2.14</td><td>-2.17</td><td>-0.85</td></tr><tr><td rowspan="3">Fixed-rate</td><td>STF (Zou et al., 2022)</td><td>-14.85</td><td>-12.87</td><td>-11.69</td></tr><tr><td>LIC-TCM (Liu et al., 2023)</td><td>-18.69</td><td>-19.37</td><td>-18.18</td></tr><tr><td>FTIC (Li et al., 2024)</td><td>-26.37</td><td>-23.33</td><td>-20.49</td></tr><tr><td rowspan="3">Progressive</td><td>PLONQ (Lu et al., 2021)</td><td>14.83</td><td></td><td>21.29</td></tr><tr><td>FTIC-CTC (Jeon et al., 2023)</td><td>-9.82</td><td>-11.52</td><td>-11.53</td></tr><tr><td>BLI</td><td>-17.78</td><td>-17.12</td><td>-17.52</td></tr></table>

Table 2: Comparison of parameter counts and latencies. The channel-wise autoregressive entropy model of FTIC is removed for progressive methods, and the DPICT postprocessing network is not used.
<table><tr><td rowspan=1 colspan=4># Params. (M) Latency (s)</td></tr><tr><td rowspan=1 colspan=1>Fixed-rate</td><td rowspan=1 colspan=1>LIC-TCM (Liu et al., 2023)FTIC (Li et al., 2024)</td><td rowspan=1 colspan=1>76.5771.01</td><td rowspan=1 colspan=1>6.53215.73</td></tr><tr><td rowspan=1 colspan=1>Progressive</td><td rowspan=1 colspan=1>FTIC-DPICT (Lee et al., 2022)FTIC-CTC (Jeon et al., 2023)BLI</td><td rowspan=1 colspan=1>40.531,032.65196.12</td><td rowspan=1 colspan=1>5.856.624.77</td></tr></table>

resolutions of $5 1 2 \times 7 6 8$ or 768 × 512, while CLIC and JPEG-AI contain 41 and 16 higher-resolution images, respectively. Compression performance is measured in bits per pixel (bpp), PSNR, and MS-SSIM, and summarized using BD-rate (Pateux and Jung, 2007).

## 4.1 Comparative Assessment

R-D curve comparison: We compare BLI with representative image compression methods: JPEG2000 (Skodras et al., 2001), BPG444 (Bellard, 2014), and VTM 12.0 (Bross et al., 2021) as traditional codecs; STF (Zou et al., 2022), LIC-TCM (Liu et al., 2023), and FTIC (Li et al., 2024) as fixed-rate learned codecs; and PLONQ (Lu et al., 2021), DPICT (Lee et al., 2022), and CTC (Jeon et al., 2023) as progressive codecs. Since earlier progressive methods were built on CNN-based baselines (Minnen et al., 2018; Cheng et al., 2020), we re-implement DPICT and CTC on the stronger FTIC backbone for fair comparison, denoted as FTIC-DPICT and FTIC-CTC. Figure 5 shows the PSNR and MS-SSIM R-D curves on the Kodak dataset, with MS-SSIM converted to dB. BLI consistently outperforms existing progressive codecs over a wide bitrate range, with particularly strong gains at low-to-medium bitrates, where limited decoded information makes bound-conditioned inference especially important for reducing reconstruction distortions. Additional R-D curves and BD-rate results on the CLIC and JPEG-AI datasets are provided in Appendix A.

Compared with fixed-rate learned codecs, BLI remains competitive despite the additional constraint of progressive decoding. In particular, BLI approaches the performance of strong fixed-rate methods and surpasses STF over most of the MS-SSIM range. This suggests that bound-conditioned refinement helps preserve structural and perceptual details. Note that STF, LIC-TCM, and FTIC are individually optimized for each target rate, whereas BLI generates all R-D points from a single progressive codec.

Among progressive approaches, BLI achieves the best R-D performance over most of the bitrate range. Its consistent advantage over FTIC-DPICT reflects the benefit of bound-conditioned inference. The gains over FTIC-CTC arise from finer step-wise autoregressive modeling within each trit-plane, which enables more effective use of spatial context and bound information for entropy modeling and latent reconstruction.

![](images/eedd9df0a372d4b9f6ccb24947ac93a944654713a6aee1556d5117ea9af04c4c.jpg)  
Figure 6: Qualitative comparison against representative traditional and progressive codecs.

![](images/bcd4b945051193d1e727f78e5c0aeeba3e7f77054d551805f9d4534988bd38e7.jpg)  
Figure 7: Qualitative comparison with existing trit-plane codecs at trit-plane endpoints. Within each column, the same number of trits is transmitted for all three codecs. The numbers below each result indicate bpp and PSNR.

BD-rate analysis: Table 1 reports PSNR BD-rates relative to FTIC-DPICT. BLI reduces bitrate by 17.78%, 17.12%, and 17.52% on Kodak, CLIC, and JPEG-AI, respectively, showing consistent gains within the progressive setting. These improvements stem from flexible in-plane autoregression, which exploits spatial context and bound information at multiple coding steps. In contrast, FTIC-CTC does not perform step-wise bound-conditioned refinement within each trit-plane, limiting its coding efficiency. Overall, BLI provides a clear R-D advantage for progressive image compression.

Model complexity: Table 2 compares the parameter counts and latency of BLI with fixed-rate and progressive baselines. BLI has about 5.3× fewer parameters than FTIC-CTC, which uses a separate prediction network for each trit-plane and therefore incurs substantial parameter overhead. Although BLI requires more parameters than FTIC-DPICT, it replaces plane-specific predictors with a shared BLI module and achieves the lowest latency among the compared codecs. This latency reduction stems from efficient Gaussian interval integration, which lowers non-neural computation compared with the loop-based discrete probability composition used in FTIC-DPICT and FTIC-CTC.

Qualitative results: Figure 6 compares reconstructed images at nearly identical bitrates. Traditional codecs such as BPG444 and VTM 12.0 introduce noticeable blurring around petal boundaries and feather textures. FTIC-DPICT also suffers from blurred details, while FTIC-CTC produces sharper structures but still fails to recover fine textures faithfully. In contrast, BLI better preserves high-frequency details and produces fewer noticeable artifacts overall. This improvement comes from BLI’s bound-conditioned autoregressive refinement, which uses spatial context and interval constraints to recover fine structures from partially decoded latents.

Figure 7 compares BLI with existing trit-plane codecs at the endpoints of trit-planes, where the same number of trits is transmitted within each column. BLI consistently achieves the best R-D performance, followed by FTIC-CTC and FTIC-DPICT. This shows that, even at the same trit-plane endpoint, BLI encodes the transmitted trits with fewer bits while producing more accurate latent estimates. The results further confirm that bound-conditioned inference improves both entropy modeling and reconstruction at intermediate progressive decoding stages.

![](images/314e839db2b264d96406a340b39974a1ab9518271234e69a2d248775d7945602.jpg)  
(a) BLI

![](images/18a5c4d6bb71722f56b16bc39e5989002bdda904a9a5c22eae51ed8e6efc5917.jpg)  
(b) Method I  
Figure 8: Step-wise updates to predicted Gaussian distributions for two latent elements during evaluation. We compare BLI with method I, the ablation variant without the bound-width input and interval-rescaled output constraint. Colors transition from blue to red as decoding proceeds.

Table 3: Comparison of ablation variants and perplane refinement schedules on the Kodak dataset. BD-rates are measured relative to FTIC-DPICT.
<table><tr><td></td><td># Params. (M)</td><td>Latency (s)</td><td>BD-rate (%)</td></tr><tr><td>Method I</td><td>155.28</td><td>4.74</td><td>-15.64</td></tr><tr><td>Method II</td><td>139.41</td><td>4.58</td><td>-15.01</td></tr><tr><td>Method III</td><td>119.32</td><td>4.00</td><td>-12.48</td></tr><tr><td>1 step/plane</td><td>196.12</td><td>3.82</td><td>-10.40</td></tr><tr><td>2 steps/plane</td><td>196.12</td><td>3.92</td><td>-12.90</td></tr><tr><td>4 steps/plane</td><td>196.12</td><td>4.15</td><td>-14.79</td></tr><tr><td>BLI</td><td>196.12</td><td>4.77</td><td>-17.78</td></tr></table>

![](images/2d133f49eea7b95a9df289da4a3046dbdcb8e29a3848b165eef1007a7879f413.jpg)  
Figure 9: Effect of the number of refinement steps per trit-plane on the Kodak dataset.

## 4.2 Analysis

Model design ablations: We evaluate three variants to examine the effects of the bound-related model design, the reconstruction prediction branch, and module capacity. Table 3 summarizes the results. In method I, we remove $\mathbf { D } ^ { k - 1 } / 2$ from the BLI input and disable the interval-rescaled output constraint for $\mathbf { M } ^ { k }$ and $\hat { \mathbf Y } ^ { k - 1 }$ . In method II, we remove the reconstruction prediction branch and instead use the expectation computed from the predicted Gaussian distribution for reconstruction. In method III, we reduce the hidden dimension of the attention module and the number of CNN layers.

As shown in Table 3, method I results in a BD-rate of −15.64%, indicating that the bound width and interval-consistent prediction are important for effective progressive inference. Method II yields −15.01%, showing that directly estimating the latent reconstruction is more effective than relying on the Gaussian expectation. Method III gives −12.48%, confirming the effect of module capacity.

Step-wise distribution refinement: Figure 8 illustrates how the predicted Gaussian distributions for selected latent elements evolve during evaluation. Each curve corresponds to the predicted distribution at a particular step, with colors transitioning from blue (early steps) to red (later steps). The vertical dashed line indicates the ground-truth latent value. Compared with method I, BLI shifts the predicted distributions more consistently toward the ground-truth latent and shows clearer variance reduction over successive steps. This confirms that the bound-width input and interval-rescaled output constraint help refine uncertainty during progressive decoding.

Refinement steps per trit-plane: Table 3 and Figure 9 analyze the effect of the number of refinement steps within each trit-plane on the Kodak dataset. We compare schedules using one, two, or four steps per trit-plane with the full BLI schedule, $i . e . ,$ , the evaluation schedule with $\bar { \mathcal { K } _ { \mathrm { e n d } } } = \{ 4 , 8 , 1 4 , 3 0 , 5 \bar { 4 } \}$ As the number of steps increases, R-D performance improves consistently: the BD-rate decreases from −10.40% for the 1-step setting to −12.90% and −14.79% for the 2-step and 4-step settings, respectively, and further reaches −17.78% for the full setting. Notably, even the 1-step configuration slightly outperforms FTIC-CTC (−10.40% vs. −9.82%), despite using far fewer parameters and lower latency. These results show that BLI achieves strong R-D performance through more frequent in-plane autoregressive refinement, which allows bound information and spatial context to be used more effectively throughout progressive decoding.

Table 4: Cross-backbone PSNR BD-rates (%) with respect to the corresponding DPICT baseline.  
Table 5: Controlled comparison with discrete prediction. PSNR BD-rates (%) are relative to FTIC-DPICT.
<table><tr><td>Backbone</td><td>CTC</td><td>BLI</td></tr><tr><td>Cheng et al. (2020)</td><td>-14.84</td><td>-19.87</td></tr><tr><td>Liu et al. (2023)</td><td>-6.86</td><td>-22.30</td></tr></table>

<table><tr><td>Method</td><td>Params. (M)</td><td>1 step/plane</td><td>Full</td></tr><tr><td>IPAR + discrete</td><td>207.80</td><td>-9.30</td><td>-14.06</td></tr><tr><td>BLI</td><td>196.12</td><td>-10.40</td><td>-17.78</td></tr></table>

Cross-backbone generalization: We further evaluate BLI on two backbones of learned image codecs beyond FTIC: the CNN-based backbone of Cheng et al. (2020), used in the original DPICT and CTC papers, and Liu et al. (2023). Table 4 reports BD-rates with respect to the corresponding DPICT baseline. On the backbone of Cheng et al. (2020), BLI achieves −19.87%, compared with the −14.84% reported for CTC in its original paper. On the backbone of Liu et al. (2023), we train CTC and BLI under the same training configurations. BLI improves the BD-rate from −6.86% to −22.30%. These results indicate that the benefit of BLI is preserved across different $g _ { a }$ and $g _ { s }$ backbones.

Discrete vs. continuous distributions: To distinguish the effect of BLI’s continuous distribution modeling from that of in-plane autoregressive refinement, we replace Gaussian interval integration with direct ternary-logit prediction, similar to CTC, while retaining the same context module, reconstruction branch and refinement schedule. Table 5 compares this variant, denoted “IPAR + discrete”, with BLI at matched refinement depths. BLI achieves lower BD-rates with one step per plane (−10.40% vs. −9.30%) and with the full schedule (−17.78% vs. −14.06%), despite using fewer parameters. Thus, increasing the number of refinement steps benefits both methods, while direct discrete prediction does not reproduce the gains of continuous interval-based distribution modeling.

## 4.3 Limitations and Future Work

Although BLI supports flexible transmission schedules within each trit-plane, our current evaluation adopts the R-D-prioritized transmission order of Lee et al. (2022). Designing transmission orders better matched to bound-conditioned inference remains an important direction for future work. A performance gap also remains between BLI and recent fixed-rate codecs, as expected for progressive codecs that reconstruct from bitstream prefixes. Closing this gap is another important direction for future work.

## 5 Conclusions

We introduced bound-conditioned latent inference (BLI), a progressive image compression framework that reformulates trit-plane coding as sequential latent inference under shrinking interval constraints. By jointly refining the Gaussian parameters for entropy modeling and the latent estimates for reconstruction within progressively updated bounds, BLI enables flexible in-plane autoregressive refinement and more effective use of the information shared by the encoder and decoder at each coding step. Experiments show that BLI achieves state-of-the-art R-D performance among progressive codecs and preserves high-frequency details more faithfully, while replacing CTC’s plane-specific predictors with a shared BLI module that uses fewer parameters and achieves lower latency. Overall, BLI provides an effective framework for progressive learned image compression.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Research Foundation of Korea (NRF) grants funded by the Korea government(MSIT) (No. RS-2024-00397293 and No. RS-2026-25616634).

The authors declare no competing interests.

## References

Eirikur Agustsson, Fabian Mentzer, Michael Tschannen, Lukas Cavigelli, Radu Timofte, Luca Benini, and Luc Van Gool. Soft-to-hard vector quantization for end-to-end learning compressible representations. In NeurIPS, 2017.

Nicola Asuni and Andrea Giachetti. TESTIMAGES: a large-scale archive for testing visual devices and basic image processing algorithms. In STAG, 2014.

Johannes Ballé, Valero Laparra, and Eero P. Simoncelli. End-to-end optimized image compression. In ICLR, 2017.

Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, and Nick Johnston. Variational image compression with a scale hyperprior. In ICLR, 2018.

Heinz H. Bauschke and Jonathan M. Borwein. On projection algorithms for solving convex feasibility problems. SIAM Rev., 38, 1996.

F. Bellard. BPG image format. Online: https://bellard.org/bpg, 2014.

Benjamin Bross, Ye-Kui Wang, Yan Ye, Shan Liu, Jianle Chen, Gary J. Sullivan, and Jens-Rainer Ohm. Overview of the versatile video coding (VVC) standard and its applications. IEEE TCSVT, 31, 2021.

Tong Chen and Zhan Ma. Variable bitrate image compression with quality scaling factors. In ICASSP, 2020.

Zhengxue Cheng, Heming Sun, Masaru Takeuchi, and Jiro Katto. Learned image compression with discretized Gaussian mixture likelihoods and attention modules. In CVPR, 2020.

Francois Chollet. Xception: Deep learning with depthwise separable convolutions. In CVPR, 2017.

CLIC. Workshop and challenge on learned image compression. In CVPRW, 2021.

Ze Cui, Jing Wang, Shanyin Gao, Tiansheng Guo, Yihui Feng, and Bo Bai. Asymmetric gained deep image compression with continuous rate adaptation. In CVPR, 2021.

Jarek Duda. Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding. arXiv preprint arXiv:1311.2540, 2013.

Vivek K. Goyal. Theoretical foundations of transform coding. IEEE Signal Process. Mag., 18, 2001.

Dailan He, Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin. Checkerboard context model for efficient learned image compression. In CVPR, 2021.

Seungmin Jeon, Kwang Pyo Choi, Youngo Park, and Chang-Su Kim. Context-based trit-plane coding for progressive image compression. In CVPR, 2023.

JPEG-AI. JPEG-AI test images. Online: https://jpegai.github.io/test\_images, 2020.

Jun-Hyuk Kim, Byeongho Heo, and Jong-Seok Lee. Joint global and local hierarchical priors for learned image compression. In CVPR, 2022.

Kodak. Kodak lossless true color image suite. Online: https://r0k.us/graphics/kodak, 1999.

Jae-Han Lee, Seungmin Jeon, Kwang Pyo Choi, Youngo Park, and Chang-Su Kim. DPICT: Deep progressive image compression using trit-planes. In CVPR, 2022.

Jooyoung Lee, Seunghyun Cho, and Seung-Kwon Beack. Context-adaptive entropy model for end-to-end optimized image compression. In ICLR, 2019.

Han Li, Shaohui Li, Wenrui Dai, Chenglin Li, Junni Zou, and Hongkai Xiong. Frequency-aware transformer for learned image compression. In ICLR, 2024.

Jiaheng Liu, Guo Lu, Zhihao Hu, and Dong Xu. A unified end-to-end framework for efficient deep image compression. arXiv preprint arXiv:2002.03370, 2020.

Jinming Liu, Heming Sun, and Jiro Katto. Learned image compression with mixed transformer-CNN architectures. In CVPR, 2023.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Yadong Lu, Yinhao Zhu, Yang Yang, Amir Said, and Taco S. Cohen. Progressive neural image compression with nested quantization and latent ordering. In ICIP, 2021.

Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte, and Luc Van Gool. Conditional probability models for deep image compression. In CVPR, 2018.

David Minnen and Saurabh Singh. Channel-wise autoregressive entropy models for learned image compression. In ICIP, 2020.

David Minnen, Johannes Ballé, and George Toderici. Joint autoregressive and hierarchical priors for learned image compression. In NeurIPS, 2018.

S. Pateux and J. Jung. An Excel add-in for computing Bjontegaard metric and its evolution, 2007.

Yichen Qian, Xiuyu Sun, Ming Lin, Zhiyu Tan, and Rong Jin. Entroformer: A transformer-based entropy model for learned image compression. In ICLR, 2022.

Amir Said and William A. Pearlman. A new, fast and efficient image codec based on set partitioning in hierarchical trees. IEEE TCSVT, 6, 1996.

Athanassios Skodras, Charilaos Christopoulos, and Touradj Ebrahimi. The JPEG 2000 still image compression standard. IEEE Signal Process. Mag., 18, 2001.

Myungseo Song, Jinyoung Choi, and Bohyung Han. Variable-rate deep image compression through spatially-adaptive feature transform. In ICCV, 2021.

Kedeng Tong, Yaojun Wu, Yue Li, Kai Zhang, Li Zhang, and Xin Jin. QVRF: A quantization-erroraware variable rate framework for learned image compression. In ICIP, 2023.

Gregory K. Wallace. The JPEG still picture compression standard. IEEE Trans. Consum. Electron., 38, 1992.

Hao Xu, Xiaolin Wu, and Xi Zhang. Multirate neural image compression with adaptive lattice vector quantization. In CVPR, 2025.

Fei Yang, Luis Herranz, Yongmei Cheng, and Mikhail G. Mozerov. Slimmable compressive autoencoders for practical neural image compression. In CVPR, 2021.

Yinhao Zhu, Yang Yang, and Taco Cohen. Transformer-based transform coding. In ICLR, 2022.

Renjie Zou, Chunfeng Song, and Zhaoxiang Zhang. The devil is in the details: Window-based attention for image compression. In CVPR, 2022.

## A Additional Experimental Results

More R-D curves: For completeness, we provide additional comparative results on the Kodak, CLIC 2021, and JPEG-AI datasets. Figure 10 presents the full PSNR and MS-SSIM R-D curves for all three datasets. Table 6 summarizes the BD-rates for both PSNR and MS-SSIM with respect to FTIC-DPICT.

![](images/58a1bfbcde581e55349a074f3e4921e940963266d90d42d40409c4688a82dbbb.jpg)

![](images/24bc8585d63f85c18de8d790e5d71fb947537a9c163879f7f70caa2eba7d797b.jpg)

![](images/1ac6e2546d7959a128a28b1bb5016908944726bb277bd579006af09fb78fd278.jpg)

![](images/7e920bf39a1a10d0f9192778241d25dfeb73d49c74e5ec97bff8cd4230dab79b.jpg)

![](images/b4f87b68078c3053c4da803b788aea94b40f635f5da322e23e49120374430544.jpg)

![](images/513c54a7ebcea1b52ce24dd936eabeb0c847ffc80a787665f501352adff6f73a.jpg)  
Figure 10: R-D curve comparison of the proposed BLI codec with existing progressive and fixed-rate image codecs. From top to bottom, the rows correspond to Kodak, CLIC 2021, and JPEG-AI. Best viewed digitally with zooming.

## B Implementation Details

## B.1 Architecture Configurations

We embed the proposed BLI framework into FTIC (Li et al., 2024), where the hyper latent Z has 128 channels and the primary latent Y has 320 channels. To estimate the initial Gaussian parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ , we employ two identical predictor networks. Each predictor consists of a residual CNN block with three $\bar { 3 } \times \bar { 3 }$ convolution layers, followed by two FAT blocks (Li et al., 2024), and a final $3 \times 3$ convolution layer, yielding a 320-channel output tensor that provides the initial distribution estimates. All latent-related tensors, including Y, $\mathbf { M } ^ { k } , \Sigma ^ { k } , \mathbf { L } ^ { k } , \mathbf { \Sigma } ^ { \mathbf { ' } } \mathbf { U } ^ { k } , \mathbf { C } ^ { k }$ , and $\mathbf { D } ^ { k }$ , have shape $C \times H \times W$ , where $C = 3 2 0$ and $( H , W )$ are $1 / 1 6$ of the input image resolution. In the BLI model architecture, a total of 10 Swin attention (Liu et al., 2021) layers are used, each with a hidden dimension of 960, 16 attention heads, and a window size of 8. Each CNN branch uses four depthwise convolution (Chollet, 2017) layers to improve efficiency while preserving spatial modeling capacity.

Table 6: BD-rates (%) with respect to FTIC-DPICT.
<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">PSNR</td><td colspan="3">MS-SSIM</td></tr><tr><td>Kodak</td><td>CLIC</td><td>JPEG-AI</td><td>Kodak</td><td>CLIC</td><td>JPEG-AI</td></tr><tr><td rowspan="2">Traditional</td><td>JPEG2000 (Skodras et al., 2001) BPG444 (Bellard, 2014)</td><td>47.54</td><td>58.41</td><td>55.73</td><td>183.09</td><td>232.17</td><td>205.86</td></tr><tr><td>VTM 12.0 (Bross et al., 2021)</td><td>-7.92 -2.14</td><td>0.21 -2.17</td><td>0.74 -0.85</td><td>64.25 69.49</td><td>74.67 72.06</td><td>89.49 94.68</td></tr><tr><td rowspan="4">Fixed-rate</td><td>STF (Zou et al., 2022)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-14.85</td><td>-12.87</td><td>-11.69</td><td>-12.64</td><td>-7.73</td><td>-5.57</td></tr><tr><td>LIC-TCM (Liu et al., 2023) FTIC (Li et al., 2024)</td><td>-18.69</td><td>-19.37</td><td>-18.18</td><td>-13.84</td><td>-7.74</td><td>-6.53</td></tr><tr><td></td><td>-26.37</td><td>-23.33</td><td>-20.49</td><td>-25.27</td><td>-20.44</td><td>-16.58</td></tr><tr><td rowspan="3">Progressive</td><td>PLONQ (Lu et al., 2021) FTIC-CTC (Jeon et al., 2023)</td><td>14.83</td><td></td><td>21.29</td><td></td><td></td><td></td></tr><tr><td></td><td>-9.82</td><td>-11.52</td><td>-11.53</td><td>-8.43</td><td>-10.58</td><td>-9.20</td></tr><tr><td>BLI</td><td>-17.78</td><td>-17.12</td><td>-17.52</td><td>-18.56</td><td>-18.14</td><td>-17.61</td></tr></table>

## B.2 Training Loss

In (6), the endpoint indices are given by $\mathcal { K } _ { \mathrm { e n d } } = \{ 4 , 8 , 1 2 , 1 6 , 2 0 \}$ , where $l = 4$ and $l = 2 0$ represent the MST and the LST, respectively. The weight $\alpha _ { l }$ is adopted to account for the different quantization intervals across plane levels. Specifically, because the quantization interval increases by a factor of 3 from the LST to the MST, we set $\alpha _ { l }$ proportional to the inverse squared quantization interval. As a result, the weights decrease by a factor of $1 / 9$ from the LST to the MST, corresponding to $\{ ( 1 / 9 ) ^ { 0 } , ( 1 / 9 ) ^ { 1 } , \ldots , ( 1 / 9 ) ^ { 4 } \}$ from the LST to the MST.

## B.3 Transmission Schedule

In BLI, the trits transmitted within each trit-plane are divided into multiple coding steps. The step endpoints are represented by $\kappa _ { \mathrm { e n d } }$ , which gives the cumulative coding-step indices at the endpoints of successive trit-planes. During training, each trit-plane is uniformly divided into four steps, yielding $\mathcal { K } _ { \mathrm { e n d } } = \{ 4 , 8 , 1 2 , 1 6 , 2 0 \}$ . To expose BLI to diverse partial-observation patterns, we randomly assign trits within each trit-plane to the four steps during the initial training stage. We then fine-tune the codec using the same R-D-prioritized transmission order (Lee et al., 2022) as in evaluation.

During evaluation, we use a nonuniform full BLI schedule, $\mathcal { K } _ { \mathrm { e n d } } = \{ 4 , 8 , 1 4 , 3 0 , 5 4 \}$ , corresponding to step allocations of (4, 4, 6, 16, 24) from the MST to the LST. This schedule allocates more refinement steps to trit-planes closer to the LST, since these planes typically contain more highentropy trits and benefit more from finer in-plane subdivision. Within each trit-plane, the assignment of trits to coding steps follows the R-D-prioritized order. Specifically, for each latent element, we estimate the expected bit increase $\Delta R$ and the distortion change $\Delta D$ using the predicted distribution parameters available at the beginning of the plane. Trits are then ranked in descending order of $- \Delta D / \Delta R .$ , so that those expected to yield larger distortion reduction per bit are transmitted earlier.

## B.4 Trit Partitioning Within a Trit-Plane

Using the initial scale parameters $\pmb { \Sigma } ^ { 0 }$ predicted from the hyper latent $\hat { \mathbf { Z } } ,$ we determine the number of trits required to represent each latent element. For every Σ<sup>0</sup>, we compute the width of the symmetric Gaussian interval that contains probability mass $1 - \mathrm { \dot { 1 } 0 ^ { - \tilde { 9 } } }$ . This interval length is given by $2 \kappa \Sigma _ { i } ^ { 0 } .$ where $\kappa = - \Phi ^ { - 1 } ( 5 \times 1 0 ^ { - 1 0 } )$ ). The required number of trits is then obtained by taking the ceiling of the base-3 logarithm of this length:

$$
n _ { i } = \lceil \log _ { 3 } ( 2 \kappa \Sigma _ { i } ^ { 0 } ) \rceil .\tag{7}
$$

Thus, $n _ { i }$ represents the number of ternary digits needed to encode the i-th latent element. In each trit-plane, before processing the first step, we partition all elements satisfying $n _ { i } \geq l$ in the current plane l into multiple subsets, which correspond to the coding steps.

## B.5 Algorithmic Workflow

Since the encoder and decoder share the same transmission schedule described in Section B.3, they can identify the trits processed at each step. Once the trits for a given step are decoded, the corresponding latent bounds are updated by shrinking to the selected sub-intervals. Full pseudo-code for the encoding and decoding procedures is provided in Algorithms 1 and 2. During training, the forward pass follows the same procedure as the decoding process in Algorithm 2, except that no entropy decoding is performed. Instead, the lower and upper bounds at each step are directly obtained from the ground-truth latent Y.

Algorithm 1: Encoding in BLI   
Data: Initial parameters $\overline { { { \bf M } ^ { 0 } , { \bf \Sigma } { \bf { \Sigma } } ^ { 0 } } }$ , trit counts $n _ { i }$   
Result: Encoded bitstream   
/\* Initialization \*/   
Compute initial bounds:   
${ \bf C } _ { i } ^ { 0 } \gets { \bf M } _ { i } ^ { 0 } , \qquad { \bf D } _ { i } ^ { 0 } \gets 3 ^ { n _ { i } } .$   
Set step index $k \gets 1 ;$   
while not all trits encoded do   
Estimate $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } , \hat { \mathbf { Y } } ^ { k - 1 } )$ using $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } , \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ // BLI   
Identify trits to be encoded for step k;   
Encode the trits for step k using $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ // Entropy encoding   
Update bounds $( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ ; // Bound update   
$k \gets k + 1 ;$

Algorithm 2: Decoding in BLI   
Data: Initial parameters $\mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } .$ , trit counts $n _ { i } ,$ encoded bitstream   
Result: Reconstructed latent   
/\* Initialization \*/   
Compute initial bounds:   
${ \bf C } _ { i } ^ { 0 } \gets { \bf M } _ { i } ^ { 0 } , \qquad { \bf D } _ { i } ^ { 0 } \gets 3 ^ { n _ { i } } .$   
Set step index $k \gets 1 ;$   
while target quality or bitrate not reached do   
Estimate $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } , \hat { \mathbf { Y } } ^ { k - 1 } )$ using $( \mathbf { C } ^ { k - 1 } , \mathbf { D } ^ { k - 1 } , \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ // BLI   
Identify trits to be decoded for step k;   
Decode the trits for step k from the bitstream using $( { \bf { M } } ^ { k } , { \pmb { \Sigma } } ^ { k } ) ; / /$ Entropy decoding   
Update bounds $( \mathbf { C } ^ { k } , \mathbf { D } ^ { k } )$ using decoded trits ; // Bound update   
$k \gets k + 1 ;$

## B.6 Baseline Implementations

For the FTIC (Li et al., 2024) backbone, which underlies FTIC-DPICT (Lee et al., 2022), FTIC-CTC (Jeon et al., 2023), and BLI, we replaced the original entropy model with a simple structure described

JPEG2000: We employ the open-source implementation provided in Skodras et al. (2001). The   
commands used for encoding and decoding are provided below. We convert the RGB images into   
raw format before processing.   
{ b u i l d p a t h } / o p j \_ c o m p r e s s − i { i n p u t f i l e }   
−o { b i n } − r {15:150}   
−F { width } ,{ h e i g h t } , 3 , 8 , u@1x1 : 1 x1 : 1 x1   
{ b u i l d p a t h } / o p j \_ d e c o m p r e s s − i { b i n } −o   
{ o u t p u t f i l e }   
BPG444: We use the software provided in Bellard (2014) and run the commands listed below.   
{ b u i l d p a t h } / b p g e n c { i n p u t f i l e } −o { b i n }   
−q {26:52}   
− f 444 −e x265   
{ b u i l d p a t h } / b p g d e c −o { o u t p u t f i l e } { b i n }   
VTM 12.0: We run the VTM 12.0 reference software package available at https://vcgit.hhi.   
fraunhofer.de/jvet/VVCSoftware\_VTM/-/tree/VTM-12.0 using the following commands.   
{ b u i l d p a t h } / EncoderApp − i { i n p u t f i l e } −c   
{ c fg p a t h } / e n c o d e r \_ i n t r a \_ v t m . c fg   
−o / d e v / n u l l −b { b i n } − w d t { w i d t h } − h g t   
{ h e i g h t } − f r 1 − f 1   
−q 24 , 26 , 30 , 31:43   
−−InputChromaFormat =444   
−− I n p u t B i t D e p t h =8   
−−ConformanceWindowMode=1   
−− I n p u t C o l o u r S p a c e C o n v e r t =RGBtoGBR   
−− S N R I n t e r n a l C o l o u r S p a c e =1   
−− O u t p u t I n t e r n a l C o l o u r S p a c e =0   
{ b u i l d p a t h } / DecoderApp −b { b i n } −o   
{ o u t p u t f i l e } −d 8   
−− O u t p u t C o l o u r S p a c e C o n v e r t =GBRtoRGB   
For the other methods in comparison, we use their official implementations or the reported results   
provided in their papers.

in Section B.1 and initialized the network using the pretrained FTIC weights released in its official GitHub repository for training.

We implement the traditional codecs JPEG2000 (Skodras et al., 2001), BPG444 (Bellard, 2014), and VTM 12.0 (Bross et al., 2021) as follows.

## C Extended Analyses and Results

## C.1 Computational Complexity

To further analyze the computational cost of the progressive codecs, we report the floating-point operations and decompose the end-to-end latency into neural-network inference and non-neural processing. The FLOP counts are measured for a 768 × 512 input, corresponding to the Kodak resolution. Latency is measured on the Kodak dataset using an AMD Ryzen 9 3900X CPU and an NVIDIA RTX 4090 GPU. The non-neural processing time includes probability processing and entropy coding outside the neural-network forward passes.

BLI requires more neural computation than FTIC-DPICT because it repeatedly refines the entropy distribution and latent estimate through in-plane autoregressive inference. Its FLOP count is compa rable to that of FTIC-CTC, while its neural inference time is substantially higher than those of both

Table 7: Comparison of computational complexity on the Kodak dataset.
<table><tr><td>Method</td><td>FLOPs (T)</td><td>Neural inference (s)</td><td>Non-neural processing (s)</td><td>Total latency (s)</td></tr><tr><td>FTIC-DPICT</td><td>41.42</td><td>0.08</td><td>5.77</td><td>5.85</td></tr><tr><td>FTIC-CTC</td><td>66.69</td><td>0.18</td><td>6.44</td><td>6.62</td></tr><tr><td>BLI</td><td>65.96</td><td>3.72</td><td>1.05</td><td>4.77</td></tr></table>

Table 8: Effect of the within-plane transmission ordering on the Kodak dataset. BD-rates (%) are measured relative to FTIC-DPICT.
<table><tr><td>Trans. ordering</td><td>Full BD-rate (%)</td><td>Endpoint BD-rate (%)</td></tr><tr><td>Random</td><td>-10.01</td><td>-19.63</td></tr><tr><td>Raster scan</td><td>-7.62</td><td>-18.52</td></tr><tr><td>σ ascending</td><td>-2.47</td><td>-18.02</td></tr><tr><td>R-D prioritized</td><td>-17.78</td><td>-19.59</td></tr></table>

FTIC-DPICT and FTIC-CTC. Nevertheless, BLI achieves the lowest end-to-end latency because its non-neural processing overhead is significantly smaller.

This difference mainly arises from the computation of trit probabilities. FTIC-DPICT and FTIC-CTC construct progressive discrete probabilities using loop-based probability composition, whereas BLI directly obtains the ternary probabilities by integrating the predicted Gaussian distribution over the corresponding sub-intervals. Consequently, the additional neural computation introduced by the BLI refinement steps is offset by substantially lower non-neural processing overhead, resulting in the lowest overall latency among the three progressive codecs.

## C.2 Sensitivity to Transmission Ordering

We further analyze the effect of the within-plane transmission ordering on the progressive R-D performance of BLI. Using the same trained BLI model, we compare the R-D-prioritized ordering used in our main experiments with three alternative strategies: random ordering, raster scan, and ascending order of the predicted scale parameter. Table 8 reports two types of BD-rate. The full BD-rate is computed using all progressive reconstruction points, whereas the endpoint BD-rate is computed using only the fully decoded trit-plane endpoints.

The transmission ordering has a substantial effect on the full progressive R-D performance, while its effect on the trit-plane endpoints is relatively small. The results indicate that transmission ordering mainly affects how efficiently information is delivered at intermediate prefixes within each trit-plane, while the reconstruction quality at fully decoded trit-plane endpoints is less sensitive to the ordering. The R-D-prioritized strategy therefore improves progressive efficiency primarily by transmitting high-impact trits earlier within each plane.

## C.3 Entropy Modeling vs. Latent Reconstruction

To assess the relative contributions of entropy modeling and latent reconstruction, we perform testtime ablations using the same trained BLI model. We selectively disable each refinement pathway while retaining the other. Table 9 summarizes the results on the Kodak dataset.

When the step-wise refined entropy parameters $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ are retained but the predicted latent estimate $\hat { \mathbf { Y } } ^ { k - 1 }$ is replaced by the Gaussian expectation, the BD-rate changes from −17.78% to −16.38%. In contrast, disabling entropy refinement and using the initial hyperprior parameters $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ for entropy coding, while retaining the predicted latent reconstruction, reduces the BD-rate gain to $- 1 0 . \hat { 2 } \hat { 4 } \%$ . These results indicate that progressively refining the entropy parameters provides a larger contribution to the overall R-D improvement. The explicit latent-reconstruction prediction provides a smaller but consistent additional gain, showing that the two pathways play complementary roles in BLI.

Table 9: Test-time ablation of entropy modeling and latent reconstruction on the Kodak dataset. BD-rates (%) are measured relative to FTIC-DPICT.
<table><tr><td>Test-time variant</td><td>Entropy params. for coding</td><td>Latent recon.</td><td>BD-rate (%)</td></tr><tr><td>Recon. with Gaussian mean</td><td> $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ </td><td>Gaussian mean</td><td>-16.38</td></tr><tr><td>No entropy refinement</td><td> $( \mathbf { M } ^ { 0 } , \pmb { \Sigma } ^ { 0 } )$ </td><td> $\hat { \mathbf Y } ^ { k - 1 }$ </td><td>-10.24</td></tr><tr><td>Full BLI</td><td> $( \mathbf { M } ^ { k } , \pmb { \Sigma } ^ { k } )$ </td><td> $\hat { \mathbf Y } ^ { k - 1 }$ </td><td>-17.78</td></tr></table>

Table 10: Comparison of perceptual quality on the Kodak dataset at low bitrates. Lower LPIPS and DISTS values indicate better perceptual quality.
<table><tr><td>bpp</td><td>Metric</td><td>BPG444</td><td>VTM 12.0</td><td>FTIC-DPICT</td><td>FTIC-CTC</td><td>BLI</td></tr><tr><td>0.10</td><td>LPIPS DISTS</td><td>0.3848 0.2038</td><td>0.3494 0.1994</td><td>0.4008 0.2308</td><td>0.3465 0.2089</td><td>0.3203 0.1910</td></tr><tr><td>0.15</td><td>LPIPS</td><td>0.3207</td><td>0.3064</td><td>0.3569</td><td>0.3123</td><td>0.2868</td></tr><tr><td></td><td>DISTS</td><td>0.1771</td><td>0.1800</td><td>0.2049</td><td>0.1884</td><td>0.1736</td></tr><tr><td>0.20</td><td>LPIPS</td><td>0.2760</td><td>0.2663</td><td>0.3045</td><td>0.2661</td><td>0.2366</td></tr><tr><td></td><td>DISTS</td><td>0.1586</td><td>0.1606</td><td>0.1780</td><td>0.1630</td><td>0.1510</td></tr></table>

## C.4 Perceptual Evaluation

We further evaluate perceptual quality at low bitrates on the Kodak dataset using LPIPS and DISTS. For each method, the metric values at common target bitrates of 0.10, 0.15, and 0.20 bpp are obtained by linear interpolation between adjacent measured operating points. Table 10 reports the dataset-level averages over all Kodak images, where lower values indicate better perceptual quality.

BLI achieves the lowest LPIPS and DISTS values at all three target bitrates, consistently outperforming both conventional codecs and the progressive learned baselines. These results indicate that the R-D improvements of BLI are also accompanied by improved perceptual quality in the low-bitrate regime, rather than being limited to distortion metrics such as PSNR and MS-SSIM.

## C.5 Reconstructed Images

Figures 11∼19 present examples of images progressively reconstructed by the PSNR-trained BLI model on the Kodak (Kodak, 1999), CLIC (CLIC, 2021), and JPEG-AI (JPEG-AI, 2020) datasets.

## C.6 Tests on Challenging Color Patterns

To evaluate the robustness of BLI on challenging color patterns, we apply our framework to two images from the Tecnick dataset (Asuni and Giachetti, 2014) that includes smooth low-frequency gradients and sharp chromatic edges. The resulting reconstructions are shown in Figure 20.

![](images/58772c4e054175f274f6c20dd2a4456310e65f8670028c87d9b969530d290442.jpg)  
Figure 11: Progressively reconstructed images of ‘kodim04.png’ and ‘kodim18.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/322c8837ff35325a5895d4eeea49423d8241a3606e8e0fa4c75c0452153ee244.jpg)

0.062 / 19.68 / 7.45  
![](images/45162e14c87b53545573b725aedfa89933b569fa0422b1f6f70b179c226a7156.jpg)

0.164 / 24.44 / 12.17  
![](images/a20ceaf7fc55b1daf8f031dc32bef6f9902fb8c4ce7172864ade69c7e0ea797a.jpg)  
0.533 / 30.27 / 17.35

![](images/95378c9b36273084d97aabaa5ab1907995d0a2b7ffd3c918d355350ed3e83c13.jpg)

![](images/2a6ccc3116d1e1088f072f661c399139944600d9752ce40cd20c0a230d38ccb3.jpg)  
1.394 / 36.09 / 22.63  
2.126 / 35.67 / 22.10  
Figure 12: Progressively reconstructed images of ‘kodim08.png’ and ‘kodim13.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/79745ff968e78271f844991c9b4a16167a50c4dfd33cc8853129e624b1032def.jpg)

![](images/7e29259fc0753d4e64d65f5d3a37d972339d36a5c2531d86eda8d1da6592f771.jpg)  
Figure 13: Progressively reconstructed images of ‘kodim21.png’ and ‘kodim22.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/abcc51197e369ed51b1153872e1e0944801e91b205a34f63cb089bb8536e9a94.jpg)

![](images/e20849c189648912723e96f13533d853f5aa48aac75b78acd121eb9185c8397b.jpg)  
Figure 14: Progressively reconstructed images of ‘andrew-ruiz-376.png’ and ‘clem-onojeghuo-33741.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/eb5ae5dd74de79472f48fa8134f2764f39f39e10bf2d1285cb287cbaf15d2c67.jpg)  
0.045 / 23.07 / 6.78

![](images/3f57a76407c26e3f770ea4dbe4b05cb271d41e2e68a1704c0dfbb29d7ae14436.jpg)  
0.050 / 24.35 / 8.28

![](images/e17f2228285096fc0e0b851a4642e27a939771be4b75b4eda1986f526f683039.jpg)  
0.098 / 26.64 / 10.53

![](images/ce2f980601da8c6f1e75a2228be52dbdc0c346b6ee160bb613926f19d90aa498.jpg)  
0.087 / 28.74 / 12.10

![](images/259e9deeb9196447ad1544c0b07239aa5f51cf825a80f721ca55d0fc8c38fb40.jpg)  
0.345 / 31.48 / 15.69

![](images/2d205d5335bbb1fcc6b3294de118bf76e801a2eac73a48d33b668cb8280827b1.jpg)  
0.229 / 33.61 / 16.83

![](images/d5846cc31b5465fce511e429c4594a0018b8c75929815b94e2f4a36764c7486f.jpg)  
0.974 / 36.28 / 20.26

![](images/f30b3dc21f971028a295688a4081df632a5cef109af91fe5c74ac9eb012da9e9.jpg)  
0.586 / 38.09 / 21.52  
Figure 15: Progressively reconstructed images of ‘daniel-robert-405.png’ and ‘mullermarc-184.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

0.997 / 36.71 / 20.68  
![](images/c8e756617ed2dcf9e5e9289b370ee4de2c1b2859b1f7928c3c4f615716392b37.jpg)

![](images/3119f4b56224f08873560ff6fcf5c95b87f2c83c73d0f24f27650087bae3c2fe.jpg)  
Figure 16: Progressively reconstructed images of ‘thong-vo-428.png’ and ‘vita-vilcina-3055.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

0.559 / 38.21 / 21.28  
![](images/f06ee5b82a21b6825231e431e413e631f2fd11119a7c321a200ee3b58f88d301.jpg)

![](images/244396aeab682c98dcc6f755d99dbd70144f3eb28dc6c1a4253bf5fa3b49840d.jpg)  
0.881 / 36.46 / 17.01  
Figure 17: Progressively reconstructed images of ‘00002\_TE\_2144x1424.png’ and ‘00003\_TE\_1944x1296.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/e79693955a932ea8a586e9cc3e0230edbc1fb5fc6fc71d1bae72fda6e92d747a.jpg)  
0.065 / 17.97 / 6.44

![](images/63ba7195349fe08f7a1e1d6b0cab6585f0c1040d6e355a5cdcbfc4c8faf7ef0d.jpg)

![](images/87623a619f31be565b6d5fb3d9fe431a021bb0f964b12ab9cfaa8fdfbadf8515.jpg)

0.262 / 22.17 / 11.2  
![](images/8e78236006a7267df11473dce981f19dcf889b3e16539f7e5bfa166d5e4fa573.jpg)

0.851 / 28.25 / 17.04  
0.336 / 40.63 / 23.37  
![](images/f5496340e526d11574ae1e7ce7446ffcb727afceca5d9b71da3419023865ac8e.jpg)  
1.856 / 33.11 / 21.83  
Figure 18: Progressively reconstructed images of ‘00005\_TE\_1336x872.png’ and ‘00009\_TE\_1976x1312.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/a6043f224fbbbecf9eef63880facfd605ace057ca0a86105a1363549b9d0bf4d.jpg)

![](images/7293bb9139738fc21afe5a576b003533b9b9554b80672d01850e95e472c71301.jpg)  
Figure 19: Progressively reconstructed images of ‘00010\_TE\_2000x1128.png’ and ‘00011\_TE\_1744x1160.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.

![](images/129e3d6ced60423a96efde5c33cfdd45c61f5b335a04f3327afe98636f20bd22.jpg)

![](images/bfc96acd4ec1662d13009d3ce3fb14f108c87faf3a9d5f5455fc93306aaffe64.jpg)  
Figure 20: Progressively reconstructed images of ‘img\_1280x720\_3x8bit\_RGB\_color\_SMPTE\_RP\_219\_2002.png and ‘img\_1280x720\_3x8bit\_RGB\_color\_rainbow\_gradient.png’ obtained by the PSNR-trained model. The bitrate (bpp), PSNR (dB), and MS-SSIM (dB) are reported below each image.