# Accelerated Decoding of Centroid Positional Encoding for Instance Segmentation

Carmelo Scribano<sup>1</sup>, Filippo Muzzini<sup>1</sup>, Nedyalko Prisadnikov<sup>2</sup> Mohammad Mahdi<sup>2</sup>, Yuqian Fu<sup>2</sup>, Giorgia Franchini<sup>1</sup>, Danda Pani Paudel<sup>2</sup>, Marko Bertogna<sup>1</sup>, and Luc Van Gool<sup>2</sup>

<sup>1</sup> University of Modena and Reggio Emilia, Italy {name}.{surname}@unimore.it

INSAIT, Sofia University “St. Kliment Ohridski”, Bulgaria {name}.{surname}@insait.ai

Abstract. Beyond model inference, the decoding stage, which converts raw network outputs into task-level representations, constitutes a significant portion of the execution cost. Despite its practical impact, prediction decoding has received comparatively little attention and is often implemented using generic CPU routines or ineficient GPU kernels, limiting the benefits of advances in model eficiency. In this work, we investigate the decoding overhead associated with a recent sinusoidal centroid encoding for Instance Segmentation, in which each pixel regresses a positional embedding of its instance centroid. This approach allows flexible segmentation without predefined proposals, but extracting instance masks from dense embeddings incurs a high computational cost. We present an optimized CUDA-based implementation of the decoding algorithm tailored to this encoding, explicitly addressing challenges related to parallelization, synchronization, and memory access on modern GPUs. Our solution significantly reduces decoding overhead and improves End-to-End inference latency, outperforming both CPU-based approaches and naive GPU implementations. The results demonstrate that eficient decoding is essential to fully exploit the advantages of advanced output representations and highlight the importance of jointly designing encoding schemes and their decoding algorithms for real-time computer vision systems.

Keywords: Edge AI · Eficient Post-Processing · CUDA Acceleration

## 1 Introduction

The computational cost of modern computer vision systems has been steadily increasing, along with the performance achieved on a broad set of tasks. While significant efort has been devoted to designing more eficient model architectures and advanced model compression strategies, the computational cost of prediction decoding (the transformation from raw network outputs to task-level representations) has received significantly less interest. In practice, the decoding stage can incur significant overhead and, in some cases, dominate end-to-end inference time. This issue is particularly relevant in edge deployments, where computational resources are limited and real-time processing is often required. For example, smart-city applications frequently execute vision models directly on edge devices [13], making low-latency inference essential for timely decision-making. Common instances of decoding operations include non-maximum suppression in object detection [3], clustering-based grouping in instance segmentation [4], and parts grouping in human pose estimation [14]. These decoding steps are often implemented using generic CPU routines or suboptimal GPU kernels, leading to memory-bound operations, excessive synchronization, and poor utilization of modern accelerators. As a result, improvements in model eficiency or backbone speed do not necessarily translate into proportional gains in overall system performance.

This study explores the computational eficiency of sinusoidal centroid encodings for instance segmentation [9]. Although this representation enables the modeling of an arbitrary number of instances without complex proposal logic, recovering the final masks remains a bottleneck due to the multi-stage nature of the decoding algorithm. We present a high-performance CUDA implementation specifically designed to overcome these parallelization and memory access challenges. Our optimized decoder enables the model to achieve its full potential in real-time scenarios, efectively bridging the gap between theoretical representation and practical deployment. Our findings highlight that the practical value of novel output representations is closely linked to the eficiency of their associated decoding algorithms.

## 2 Background and Related Work

## 2.1 Instance Segmentation

Formally, the goal of the instance segmentation (IS) task is to assign each pixel in an image to a unique object instance. This task is strictly related to Semantic Segmentation (SS), which instead assigns each pixel to a semantic class. The two tasks can be combined as Panoptic Segmentation, which assigns pixels to instance masks with a semantic label. This paper focuses on the class-agnostic IS task.

Instance Segmentation (IS) is challenging because the number of object instances is not known in advance. As a result, the model must simultaneously localize objects and assign pixels to the correct instance, producing a variable number of output masks. Top-Down methods (e.g., Mask R-CNN [2]) simplify this by detecting bounding boxes and then segmenting each box independently. While efective, this sequential processing of proposals scales poorly with object density. Bottom-Up methods instead predict dense pixel-level embeddings that are aggregated post-inference. This family of approaches avoids the proposal bottleneck, ofering a theoretically superior path toward parallelized, real-time decoding.

![](images/1e53c963a0708e7ce8bdbc847b9699b0116169d4bc009e5ada5e8bd44407a795.jpg)  
(a) Source

![](images/94b784f2ce9a08a2a183662019013cd551e54ed58aa1e862003f612e6a62622e.jpg)

![](images/f6c52056e7c2c7231835e9aa3f2ea8f537204440bba95e4affbbaa71d9000dd1.jpg)  
(b) Votes Histogram (c) LocalMax Indi- (d) Instance Masks cator

![](images/2803514acc283b6b167ef4329bf4aa9ddfd3584e788f92119f3b56388d3abfd1.jpg)  
Fig. 1: Overview of the decoding process for the proposed encoding.

## 2.2 Background

The instance segmentation approach discussed in this paper is introduced in [9], where, in conjunction with a strong DINOv2 backbone [8] and a carefully designed loss function [9], it achieves state-of-the-art results on the COCO dataset [6]. DINOv2 has already been successfully applied to segmentation [10] and object detection [7, 11], making it a strong backbone model for a variety of tasks. This work extends the paradigm adopted in Painter [16] and is conceptually related to DCME [17]. Painter encodes instances via dense RGB embeddings regressed per pixel, while DCME predicts per-pixel displacement vectors to instance centroids; sinusoidal positional encodings provide a more robust alternative to both approaches. In [12], a basic CUDA-based decoder is presented to leverage GPU acceleration. However, this approach sufers from atomic contention and poor memory reuse, leading to sub-optimal performance.

## 2.3 CUDA and GPU programming

Modern GPUs leverage the SIMD (Single Instruction Multiple Data) paradigm to provide massive parallel processing capabilities. Under the GPGPU framework, tasks are executed by transferring data to the device, launching a specialized kernel, and copying results back to the host. Using the CUDA programming model, developers can explicitly manage this parallelism by configuring threads into a grid of three-dimensional blocks. This hierarchy allows for fine-grained control over resources: threads within the same block utilize high-speed, on-chip shared memory and barriers for synchronization. By leveraging shared memory as a low-latency cache for frequently accessed data, the overhead of global memory access is mitigated, significantly enhancing the overall performance of the implementation.

## 3 Methodology

We adopt the encoding formulation proposed in [9]. Let $M = \{ M _ { k } \} _ { k = 1 } ^ { N }$ be the set of ground-truth instance masks. For each instance k, let $( x _ { c } ^ { k } , y _ { c } ^ { k } ) \in \mathbb { R } ^ { 2 }$ denote the image-space coordinates of its centroid, normalized in $[ - 1 , 1 ]$ , and let $\varOmega _ { k } \subset$ $\{ 1 , \dots , H _ { O } \} \times \{ 1 , \dots , W _ { O } \}$ denote the set of normalized pixel locations belonging to mask $M _ { k }$ . We define the instance-segmentation target $T \in \mathbb { R } ^ { H _ { O } \times W _ { O } \times 4 L }$ such that each spatial location encodes the sinusoidal positional encoding of the centroid of the instance it belongs to, or a void encoding for background pixels. Formally, for each location $( i , j )$

$$
T ( i , j ) = \left\{ \begin{array} { l l } { \mathsf { c o n c a t } ( \gamma ( x _ { c } ^ { k } ) , \gamma ( y _ { c } ^ { k } ) ) } & { \mathrm { i f ~ } ( i , j ) \in \varOmega _ { k } } \\ { \mathbf { 0 } \in \mathbb { R } ^ { 4 L } } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

where the positional encoding $\gamma ( \boldsymbol { p } ) : \mathbb { R } \to \mathbb { R } ^ { 2 L }$ , derived from [15], is defined as:

$$
\gamma ( p ) = ( s i n ( 2 ^ { l } \pi p ) , c o s ( 2 ^ { l } \pi p ) ) _ { l = 0 } ^ { L - 1 }\tag{2}
$$

L defines the number of used harmonics; $L = 4$ is used.

## 3.1 Decoding Algorithm

The algorithm to recover the predicted instance masks $\hat { M }$ from a noisy estimate $\hat { T }$ requires several steps. Let the network prediction at location $( i , j )$ be:

$$
\hat { T } ( i , j ) = \mathsf { c o n c a t } \left( \hat { x } ( i , j ) , \hat { y } ( i , j ) \right) \quad \hat { x } ( i , j ) , \hat { y } ( i , j ) \in \mathbb { R } ^ { 2 L }\tag{3}
$$

First, a coarse voting histogram $H \ \in \ \mathbb { R } ^ { ( B _ { w } + 1 ) \times ( B _ { h } + 1 ) }$ is computed, with $\boldsymbol { B } = [ 1 , . . . , B _ { w } + 1 ] \times [ 1 , . . . , B _ { h } + 1 ]$ denoting the set of bins locations (each representing a candidate instance centroid). Each output location $( i , j )$ casts soft votes for all histogram bins $( a , b ) \in B$ by comparing its predicted encodings to the positional encodings of the bin coordinates. The separable distance between output location $( i , j )$ and histogram bin $( a , b )$ in the encoded space is defined as $d _ { i j } ( a , b ) = d _ { x } ( i , j , a ) + d _ { y } ( i , j , b )$ , with:

$$
\begin{array} { r } { d _ { x } ( i , j , a ) = | | \hat { x } ( i , j ) - \gamma ( a ) | | } \\ { d _ { y } ( i , j , b ) = | | \hat { y } ( i , j ) - \gamma ( b ) | | } \end{array}\tag{4}
$$

The corresponding vote weight is given by the negative exponential of the distance:

$$
w _ { i j } ( a , b ) = e ^ { - d _ { i j } ( a , b ) }\tag{5}
$$

The coarse voting histogram H is then obtained by aggregating votes over the set of participating output locations $\boldsymbol { S } = [ 1 , . . . , H _ { O } ] \times [ 1 , . . . , W _ { O } ]$ ]:

$$
H ( a , b ) = \sum _ { ( i , j ) \in S } w _ { i j } ( a , b )\tag{6}
$$

Local-maxima search is used to retrieve a set of $\hat { K }$ candidate mask centroids $\mathcal { C } = \{ ( a _ { k } , b _ { k } ) \} _ { k = 1 } ^ { \hat { K } } ;$ , while simultaneously suppressing weak centroid candidates caused by prediction noise. Each element of C represents a hypothesized instance

centroid in the discretized output coordinate space. Each output location $( i , j )$ contributes to the m-th mask, $M _ { m }$ , associated with the closest candidate centroid bin in the encoding space:

$$
m = \arg \operatorname* { m i n } _ { k } d _ { i j } ( a _ { k } , b _ { k } )\tag{7}
$$

$$
\hat { M } _ { m } ( i , j ) = \left\{ { \begin{array} { l l } { 1 } & { \mathrm { i f } \quad d _ { i j } ( a _ { m } , b _ { m } ) < \tau } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} } \right.\tag{8}
$$

## 3.2 Decoding Implementation

In this section, we describe our implementation of the decoding phase. We exploit the CUDA ecosystem to ofload the computation on the GPU. First, we precompute the palettes for the encoded bin coordinates in x and y directions, denoted as $P _ { w }$ and $P _ { h }$ respectively. Where:

$$
P _ { n } = \left\{ \gamma \left( - 1 + { \frac { 2 i } { n + 1 } } \right) \bigg | i = 1 , 2 , \ldots , n \right\}\tag{9}
$$

denotes the vector of precomputed $\gamma ( \cdot )$ evaluated at evenly spaced candidate centroid locations corresponding to histogram bins. Unlike the original implementation, we do not use an explicit void encoding for background pixels; pixels not associated with any valid instance are implicitly treated as background, and the centroid voting histogram therefore consists of $B _ { w } \times B _ { h }$ bins. $P _ { h }$ and $P _ { w }$ are stored as model parameters preloaded in memory and can be accessed without recomputing or memory copying.

Histogram accumulation. First, we compute H from Equation 6. The naive implementation [12] assigns one $( ( i , j ) , ( a , b ) )$ tuple to each thread. The thread explicitly computes $w _ { i j } ( a , b )$ for the assigned pixel-bin pair and adds the result to the global histogram bin $H ( a , b )$ . Since many threads vote for the same histogram bin concurrently, this accumulation requires atomicAdd operations on global memory. As a result, updates to popular bins become serialized, creating severe contention, limiting memory throughput, and reducing parallel eficiency. To overcome this limitation, we approach the problem by relying on a decomposition of H and on the use of the optimized $c u \bar { B } L A S ^ { \mathrm { ~ 3 ~ } }$ library. The sum over $s$ in Equation (6) admits the following decomposition by factorizing the exponential of the sum in $w _ { i , j } ( a , b )$ as product of exponentials $w _ { i , j } ( a , b ) = E _ { x } ( i , j , a ) E _ { y } ( i , j , b )$ where:

$$
\begin{array} { r } { E _ { x } ( i , j , a ) = e ^ { - | | \hat { x } ( i , j ) - P _ { w } ( a ) | | } } \\ { E _ { y } ( i , j , b ) = e ^ { - | | \hat { y } ( i , j ) - P _ { h } ( b ) | | } } \end{array}\tag{10}
$$

Therefore, Equation (6) decomposes as:

$$
H ( a , b ) = E _ { x } ^ { \top } E _ { y }\tag{11}
$$

Delegating the computation of $E _ { x }$ and $E _ { y }$ to specialized kernels, this formulation eliminates the need for an accumulator and atomic operations. The two kernels are launched with $H _ { O } * W _ { O } * a$ threads $( H _ { O } * W _ { O } * b$ for $E _ { y }$ matrix), so each thread computes a single element of the corresponding matrix. Once $E _ { x }$ and $E _ { y }$ are computed, H is obtained using the optimized cuCBLAS function cublasSgemmStrided to implementing Equation 11.

Local Maxima Search. The computed H is a noisy estimate of candidate mask centroids (Figure 1b). The Local Maxima Search stage produces a sparse binary indicator map H marking which bins are selected as candidate centroids (Figure 1c). Conceptually, the kernel assigns one CUDA thread to each histogram cell $( x , y )$ . Each thread loads the center value $H ( x , y )$ and tests whether it is a strict local maximum within its 8-connected neighborhood. If this condition is satisfied and the value exceeds a predefined threshold, a binary peak indicator is written to $\overline { { H } } ( x , y )$

Mask Aggregation. The final stage assigns each output pixel to an instance mask according to Equation 8. The kernel is launched with one thread per output pixel $( i , j )$ . Given the binary centroid map H and the afinity matrices $E _ { x }$ and $E _ { y } ,$ each thread scans all active centroid bins $( a , b )$ such that ${ \overline { { H } } } ( a , b ) = 1$ . For each candidate, it computes the combined afinity $E _ { x } ( i , j , a ) * E _ { y } ( i , j , b )$ and selects the centroid with the highest value among the candidates that satisfy the afinity threshold $e ^ { - \tau }$ . Before scanning the centroids, each thread loads the afinity values associated with its pixel into shared memory. This avoids repeatedly reading the same $E _ { x }$ and $E _ { y }$ entries from global memory while evaluating multiple candidate centroids. The output memory is preallocated for all $B _ { w } \times B _ { h }$ possible masks, one per histogram bin. Once the best centroid $( a _ { m } , b _ { m } )$ is selected, the thread sets $\hat { M } _ { m } ( i , j ) = 1$ in the corresponding output mask. The kernel also maintains a counter vector C, where $C _ { m }$ stores the number of pixels assigned to mask m. Since multiple threads may assign pixels to the same mask, updating this counter requires an atomicAdd. In practice, this atomic operation is limited to one update per assigned pixel, while the main centroid search is performed locally by each thread.

Compared with a naive implementation [12] that launches one thread per tuple $( i , j , m )$ , this design reduces memory trafic by evaluating all candidate masks for a pixel within a single thread and reusing the pixel-specific afinity values from shared memory. This avoids repeatedly loading the same afinity vectors for each candidate mask and improves the eficiency of the aggregation stage.

## 4 Results and Discussion

In this experimental section, we discuss the performance of our CUDA-accelerated implementation of the decoding algorithm. We are interested in both system and task metrics. In particular, we measure the inference time of the decoder (Inference Time), the End-to-End inference time of the model (End-to-End Inference Time), the memory footprint, and the Panoptic Quality (PQ) [5]. First, we identify a latency/performance tradeof varying the number of histogram bins, while also validating the task performances against the PyTorch reference implementation [9]. Then we assess the latency and memory footprint at diferent input resolutions. Finally, we break down the End-to-End inference performance, including model inference time, to showcase the significant impact of the decoding stage.

## 4.1 Experimental setup

![](images/c9fb1b18eb4111294743682606d2ce66ca0b40553c4360f5c9de8d2e260f72de.jpg)  
Fig. 2: Dummy ONNX model used to profile TensorRT execution.

Latency and memory consumption are profiled on an NVIDIA Jetson Orin Nano, using TensorRT for inference acceleration. The decoder is implemented as a TensorRT plugin via the IPluginV3 interface. To evaluate the decoder in isolation from the rest of the model, we designed a minimal ONNX graph (Figure 2) which is compiled into a TensorRT engine. We leverage the trtexec utility to measure execution time, omitting data transfer latencies<sup>4</sup>.

The memory footprint is assessed by implementing a custom CUDA allocator within the TensorRT environment to track peak memory usage during inference. Task performance (Table 1) is quantified using the Panoptic Quality (PQ) metric, following the protocol established in [9]. To isolate the impact of our optimizations, we evaluate PQ across diferent decoder implementations while keeping the rest of the model architecture constant.

## 4.2 Impact of the number of bins

Decoder eficiency and task accuracy are primarily influenced by the histogram bin resolution. We assess this trade-of by comparing our CUDA implementation against the original PyTorch reference [9]. Results in Table 1 demonstrate that our decoder is consistently faster and exhibits superior timing stability, which is essential for predictable real-time performance in safety-critical applications. This stability stems from our use of static memory pre-allocation during the initialization phase. Our analysis reveals that PQ gains follow a trend of diminishing returns, plateauing at a resolution of $3 2 \times 3 2$ bins. Given that higher resolutions increase latency without significant quality improvements, we adopt the $3 2 \times 3 2$ configuration for all subsequent evaluations.

Table 1: Comparison of execution time and memory footprint of diferent decoder implementations.
<table><tr><td rowspan="2">Bins</td><td colspan="2">PQ</td><td colspan="3">Infer time (ms) Infer time st.dev Memory (MB)</td><td colspan="2"></td></tr><tr><td>Torch</td><td>TRT</td><td>Torch</td><td>TRT</td><td>Torch</td><td>TRT</td><td>Torch TRT</td></tr><tr><td> $( 8 \times 8 )$ </td><td>26.68</td><td>22.57</td><td>56.44</td><td>2.13</td><td>0.16</td><td>0.001</td><td>64.55 47.58</td></tr><tr><td> $( 1 6 \times 1 6 )$ </td><td>43.72</td><td>34.91</td><td>74.91</td><td>6.13</td><td>0.20 0.003</td><td>77.54</td><td>150.67</td></tr><tr><td> $\mathbf { ( 3 2 \times 3 2 ) }$ </td><td>51.32</td><td>50.78</td><td>196.18</td><td>30.66</td><td>0.54 0.01</td><td>121.92</td><td>547.18</td></tr><tr><td> $( 6 4 \times 6 4 )$ </td><td>52.26</td><td>52.08</td><td>493.76</td><td>101.37</td><td>27.01</td><td>0.017 218.25</td><td>2101.47</td></tr><tr><td> $( 8 0 \times 8 0 )$ </td><td>52.34</td><td>52.14</td><td>687.99</td><td>144.00</td><td>42.18</td><td>0.026 272.25</td><td>3259.25</td></tr></table>

## 4.3 Impact of the feature resolution

We further evaluate the scalability of our decoder with respect to the encoding space resolution $\hat { T } \in \mathbb { R } ^ { H _ { O } \times W _ { O } \times 4 \check { L } }$ . Since the spatial dimensions $H _ { O } , W _ { O }$ scale with the model input resolution, we measure inference time for diferent featuremap sizes using a fixed $3 2 \times 3 2$ bin configuration (Figure 3b). The proposed implementation consistently outperforms the $\mathrm { P y }$ Torch reference and exhibits better scaling behavior as the feature-map resolution increases. These results indicate that the optimized kernels efectively limit the additional computational cost associated with higher-resolution inputs.

## 4.4 End-to-End Latency Breakdown

To evaluate system-level performance, we integrate the proposed decoder into the model from [9]. The network uses a DINOv2 [8] (ViT-L [1]) backbone and four transposed convolutions for feature upsampling. After export to ONNX, the model is optimized with TensorRT and benchmarked at diferent precision levels. This configuration enables a realistic assessment of the decoder’s contribution to end-to-end inference performance. We report the End-to-End latency at diferent combinations of input resolution (280 × 280, 336 × 336, 448 × 448 and $6 1 6 \times 6 1 6 )$ and numerical precision (FP32, FP16, and INT8).

The end-to-end performance comparison is reported in Table 2. To isolate the impact of the decoder, both pipelines use the same TensorRT-optimized backbone and intermediate model components; the only diference is the decoding stage, which is implemented either using the original $\mathrm { P y }$ Torch decoder or the proposed CUDA implementation. Under these conditions, the proposed decoder consistently reduces end-to-end latency, achieving speedups of up to 55% in INT8 mode at a resolution of 280 × 280. The comparison in Figure 3a highlights that decoding is a non-trivial bottleneck and can dominate the End-to-End execution time when relying on a suboptimal implementation.

Table 2: Inference time and PQ performance as a function of bin resolution for baseline and optimized decoders.
<table><tr><td>Input Res.</td><td>Precision</td><td>End-to-end Infer - Torch (ms)</td><td>End-to-end Infer - Our (ms)</td><td>speedup %</td></tr><tr><td rowspan="3">280 × 280</td><td>FP32</td><td>225.61</td><td>154.56</td><td>31.49</td></tr><tr><td>FP16</td><td>138.40</td><td>67.35</td><td>51.34</td></tr><tr><td>INT8</td><td>129.12</td><td>58.07</td><td>55.03</td></tr><tr><td rowspan="3">336× 336</td><td>FP32</td><td>319.17</td><td>227.09</td><td>28.85</td></tr><tr><td>FP16</td><td>183.27</td><td>91.19</td><td>50.24</td></tr><tr><td>INT8</td><td>171.15</td><td>79.07</td><td>53.80</td></tr><tr><td rowspan="3">448 × 448</td><td>FP32</td><td>595.24</td><td>482.33</td><td>18.97</td></tr><tr><td>FP16</td><td>284.85</td><td>171.94</td><td>39.64</td></tr><tr><td>INT8</td><td>260.71</td><td>147.80</td><td>43.31</td></tr><tr><td rowspan="3">616 × 616</td><td>FP32</td><td>1332.53</td><td>1167.01</td><td>12.42</td></tr><tr><td>FP16</td><td>507.73</td><td>342.21</td><td>32.60</td></tr><tr><td>INT8</td><td>465.52</td><td>300.00</td><td>35.56</td></tr></table>

## 4.5 Kernel Execution Profiling

We further break down the performance profile of our implementation by analyzing individual kernel latencies (Table 3). The Histogram accumulation (HistAcc) kernel is the most significant contributor to total latency, followed by Mask Aggregation (Mask Agg), while cublasSgemmStrided and Local Maxima Search (LMS) have a marginal impact. The implementation demonstrates high timing stability, which is vital for safety-critical applications. The minor execution variance in the Mask Aggregation and cublasSgemmStrided kernels stems from atomic-induced memory contention. Our design choices in the Optimized implementation have significantly curtailed raw GPU compute time with respect to the Naive implementation, shifting the performance bottleneck toward memory operations, which now account for 33% of the total decoder inference time.

A comparison with the naive implementation [12] highlights that the decomposition of H and the use of cuBLAS are the primary contributors to the improvement in decoder execution time. Additionally, the optimization of the Mask Aggregation kernel provides a substantial benefit, achieving an approximately 6- fold speedup. The naive and optimized kernels are mutually incompatible due to difering data encoding schemes in their respective preceding pipeline stages. In summary, our optimizations drastically reduce the decoder execution time, which in turn lowers the End-to-End latency of the entire model, as the decoder represents a significant portion of the overall pipeline.

![](images/c4f5c3315a6e166194f6d2792d7db7acce1112616cb36fdb804282c96c1d490b.jpg)  
(a) End-to-End inference latency comparison across varying resolutions at INT8 precision.

![](images/def58bf826ebe26c052227ca876a91b86dd4801292ed6dfa9f22d1f9b0695f2f.jpg)  
(b) Impact of encoding resolution on latency: PyTorch baseline vs. optimized CUDA implementation (fixed 32×32 bins, L = 4).

Fig. 3: Latency analysis across implementations and resolutions.  
Table 3: Execution time of each kernel
<table><tr><td colspan="2">Operation</td><td>Kernel</td><td colspan="3">Exec (ms) Exec std (ms) time %</td></tr><tr><td rowspan="5">Optim.</td><td rowspan="3">HistAcc</td><td>compute ex kernel</td><td>6.47</td><td>0.001</td><td>21.09</td></tr><tr><td>compute ey kernel</td><td>6.47</td><td>0.001</td><td>21.09</td></tr><tr><td>cublasSgemmStrided</td><td>0.75</td><td>0.002</td><td>2.44</td></tr><tr><td>LMS</td><td>total</td><td>13.68</td><td></td><td>44.62</td></tr><tr><td></td><td>local_maxima</td><td>0.010</td><td>0.000</td><td>0.03</td></tr><tr><td rowspan="4">Naive</td><td>MaskAgg</td><td>mask_aggregation</td><td>6.88</td><td>0.007</td><td>22.44</td></tr><tr><td>HistAcc</td><td>fused_vote_kernel</td><td>77.99</td><td>0.003</td><td>73.41</td></tr><tr><td>LMS</td><td>local_maxima_nai</td><td>0.008</td><td>0.000</td><td>0.01</td></tr><tr><td>MaskAgg</td><td>mask_aggregation_nai</td><td>21.26</td><td>0.003</td><td>20.02</td></tr></table>

## 5 Conclusion

In this work, we introduced a high-performance GPU-accelerated decoder for instance segmentation that achieves substantial speedups over existing implementations. Our results demonstrate that optimizing traditionally CPU-bound auxiliary stages is as critical to reducing End-to-End latency as refining the core model architecture. By outperforming baseline GPU implementations, our design validates the necessity of hardware-aware decoding. Future research will focus on optimizing memory management—which currently accounts for 33% of the remaining latency—by transitioning from framework-level handling to custom, low-level memory orchestration.

Acknowledgments. This research was partially funded by the dAIedge project (HORIZON-CL4-2022-HUMAN-02-02, Grant Agreement Number: 101120726) and the Ministry of Education and Science of Bulgaria (support for INSAIT, part of the Bulgarian National Roadmap for Research Infrastructure).

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Dosovitskiy, A.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

2. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: Proceedings of the IEEE international conference on computer vision. pp. 2961–2969 (2017)

3. Hosang, J., Benenson, R., Schiele, B.: Learning non-maximum suppression. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (July 2017)

4. Jiang, L., Zhao, H., Shi, S., Liu, S., Fu, C.W., Jia, J.: Pointgroup: Dual-set point grouping for 3d instance segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2020)

5. Kirillov, A., He, K., Girshick, R., Rother, C., Dollár, P.: Panoptic segmentation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9404–9413 (2019)

6. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

7. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: European conference on computer vision. pp. 38–55. Springer (2024)

8. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

9. Prisadnikov, N., Van Gansbeke, W., Paudel, D.P., Van Gool, L.: A simple and generalist approach for panoptic segmentation. arXiv preprint arXiv:2408.16504 (2024)

10. Ren, T., Chen, Y., Jiang, Q., Zeng, Z., Xiong, Y., Liu, W., Ma, Z., Shen, J., Gao, Y., Jiang, X., et al.: Dino-x: A unified vision model for open-world object detection and understanding. arXiv preprint arXiv:2411.14347 (2024)

11. Ren, T., Jiang, Q., Liu, S., Zeng, Z., Liu, W., Gao, H., Huang, H., Ma, Z., Jiang, X., Chen, Y., et al.: Grounding dino 1.5: Advance the" edge" of open-set object detection. arXiv preprint arXiv:2405.10300 (2024)

12. Scribano, C., Mahdi, M., Muzzini, F., Prisadnikov, N., Fu, Y., Verucchi, M., Sanudo Olmedo, I., Paudel, D.P., Van Gool, L.: Edge Deployment of Multi-Task Vision Models for Smart City Infrastructures. In: EEAI 2025 - European Conference on EDGE AI Technologies and Applications. Naples, Italy (Oct 2025), to appear

13. Scribano, C., Sanudo Olmedo, I., Verucchi, M., Bertogna, M., et al.: On-the-edge inference enabled vision system for smart cities. In: SMART 2025, The Fourteenth International Conference on Smart Cities, Systems, Devices and Technologies. pp. 24–48 (2025)

14. Tang, W., Wu, Y.: Does learning specific features for related parts help human pose estimation? In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

16. Wang, X., Wang, W., Cao, Y., Shen, C., Huang, T.: Images speak in images: A generalist painter for in-context visual learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6830–6839 (2023)

17. Watanabe, T., Wolf, D.: Distance to center of mass encoding for instance segmentation. In: 2018 21st International conference on intelligent transportation systems (ITSC). pp. 3825–3831. IEEE (2018)