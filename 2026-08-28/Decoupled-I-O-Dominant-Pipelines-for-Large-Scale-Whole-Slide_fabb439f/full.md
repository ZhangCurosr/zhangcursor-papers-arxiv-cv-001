# Decoupled I/O-Dominant Pipelines for Large-Scale Whole-Slide Image Embedding Extraction

Mayanka Chandrashekar Oak Ridge National Laboratory

Tirthankar Ghosal

Oak Ridge National Laboratory

Xi Zhang Oak Ridge National Laboratory

Ethan Seefried Oak Ridge National Laboratory

John Gounley Oak Ridge National Laboratory

Heidi Hanson Oak Ridge National Laboratory

Abstract—Whole-slide images (WSIs) are central to computational pathology but are prohibitively large, making patch-based processing the practical unit for foundation model inference. At scale, however, generating and handling massive numbers of patches on quickly introduces significant I/O and orchestration overhead, often dominating end-to-end performance. We present a decoupled, I/O-aware pipeline for large-scale WSI embedding extraction that decomposes the workflow into three stages: (1) patch generation and staging, (2) embarrassingly parallel embedding inference, and (3) sharded vector database ingestion. This design isolates data movement from compute, enabling efficient patch delivery, scalable multi-node inference with minimal communication. The resulting system produces a distributed vector database where embeddings are persistently coupled with rich metadata (e.g., patient, slide, and patch attributes), enabling efficient filtering, retrieval, and downstream reuse. This representation database is compact and reusable for tasks such as retrieval, classification, and few-shot learning, particularly benefiting low-resource environments. We show that decoupling I/O, computation, and ingestion enables high-throughput WSI embedding extraction at scale. By characterizing the scaling envelope, we demonstrate that storage dominates beyond moderate concurrency, reframing WSI embedding extraction as a datacentric systems problem rather than a purely compute-bound workload.

## I. INTRODUCTION

Whole-slide images (WSIs) are a cornerstone of computational pathology, enabling high-resolution analysis of tissue at gigapixel scale. The emergence of foundation models has further increased the demand for large-scale embedding extraction from WSIs, supporting downstream tasks such as retrieval, classification, and cohort-level analysis. However, the sheer size of WSIs—often exceeding tens of gigapixels per image—makes direct processing infeasible, necessitating patchbased decomposition as the fundamental unit of computation.

While patching enables scalable model inference, it also fundamentally reshapes the systems problem. At moderate scale, patches can be generated and processed on demand with acceptable overhead. At large scale, the need to generate, move, and process millions of patches introduces significant data orchestration challenges. In particular, patch generation from large WSIs, coordination of data movement across storage and compute nodes, and efficient handling of intermediate representations can dominate runtime, shifting the bottleneck away from accelerator throughput toward I/O and pipeline design.

Existing WSI pipelines are typically designed as tightly coupled workflows in which patch generation, model inference, and output handling are interleaved. While such designs simplify implementation, they often lead to inefficient resource utilization, increased contention on shared storage systems, and limited scalability. Furthermore, these pipelines frequently treat embedding outputs as transient artifacts, missing the opportunity to store reusable embeddings that can be used for downstream applications and reduce the cost of future largescale computation.

We define an I/O-dominated workload as an end-to-end runtime governed by data movement and storage system behavior rather than compute throughput or communication overhead. By data-centric, we refer to workloads where performance is determined primarily by data orchestration—generation, movement, and persistence—rather than computation or communication. Large-scale WSI embedding extraction is an I/Odominated, data-centric workload in which performance is governed by data movement and storage system behavior rather than compute or communication.

In this work, we show the utility of treating large-scale WSI embedding extraction as a data-centric systems problem, rather than a purely compute-driven workload. To this end, we present a decoupled pipeline architecture that separates the workflow into three stages: (1) patch generation and I/O-aware data staging, (2) massively parallel embedding inference, and (3) sharded vector database ingestion. This decomposition enables each stage to be optimized independently according to its dominant constraints—storage throughput, compute parallelism, and data management—while minimizing cross-stage interference.

A central aspect of our design is the treatment of the vector database as a reusable resource to be shared with the scientific and medical communities. Rather than producing embeddings as temporary files used for a specific project, our pipeline directly ingests them into a distributed, sharded vector database, with rich metadata (e.g., patient, slide, and patchlevel attributes) persistently associated with each embedding. This creates a shared resource that supports efficient indexing, filtering, and retrieval, converting the output of largescale computation into a reusable, queryable, and energyefficient resource that accelerates downstream analyses. A stored embedding resource is also particularly valuable in lowresource environments, where extracting embeddings from a large number of WSIs is impractical.

We implemented our pipeline on Oak Ridge Leadership Computing - Frontier Exascale System [1] and evaluated it on a large-scale WSI dataset. Our results show that decoupling patch orchestration, inference, and ingestion enables high throughput, scalability, and resource efficiency. More broadly, our work demonstrates that aligning pipeline structure with data movement characteristics is necessary for scaling dataintensive AI workloads.

This paper makes the following contributions:

• We identify and characterize WSI embedding extraction as an I/O- and orchestration-dominated workload at large scale.

• We design and implement a three-stage decoupled pipeline that separates patch generation, parallel inference, and sharded vector database ingestion.

• We introduce the construction of metadata-aware vector databases as a reusable resource for large-scale WSI processing pipelines.

• We demonstrate improved scalable execution and characterize resource bottlenecks on supercomputing infrastructure.

## II. BACKGROUND, MOTIVATION AND RELATED WORK

Recent advances in computational pathology have been driven by large-scale representation learning for WSIs. Early approaches relied on weak supervision and multiple-instance learning to aggregate patch-level features into slide-level predictions [14], [15].

Subsequent work introduced self-supervised learning and vision transformer-based approaches that improved representation quality and generalizability [16]. Hierarchical architectures such as HIPT further model multi-scale structure, capturing both cellular and tissue-level context [2]. More recently, whole-slide foundation models have emerged as the dominant paradigm. Models such as Virchow and H-Optimus-0 leverage large-scale pretraining on histopathology datasets to achieve strong performance across tasks [3], [4]. Multimodal extensions (e.g., TITAN, mSTAR) integrate WSIs with clinical reports and molecular data, enabling retrieval, pathology report generation, and cross-modal reasoning [17], [18]. Despite these advances, prior work largely focuses on improving model accuracy and do not address computational bottlenecks that slow large scale deployment.

On the systems side, WSI pipelines are typically implemented as tightly coupled workflows in which patch generation, model inference, and output handling are interleaved. While such designs simplify implementation, they limit scalability due to shared I/O contention and prevent independent optimization of pipeline stages. As noted in our pipeline formulation, WSI processing naturally decomposes into patch generation, embedding inference, and data management stages with distinct resource characteristics .

In parallel, the HPC community has extensively studied data movement bottlenecks in large-scale workflows. Prior work highlights that I/O and data orchestration increasingly dominate runtime and proposes techniques such as burst buffers, hierarchical storage, and workflow decoupling [9], [10], [8]. However, these approaches are designed for workloads that involve large contiguous datasets and structured communication patterns. In contrast, WSI pipelines generate massive numbers of small, independent data objects and exhibit embarrassingly parallel computation with minimal communication, creating a mismatch with existing HPC assumptions.

Recent HPC efforts have begun to explore WSI-scale workloads. Systems such as GEMS demonstrate distributed training for high-resolution pathology images [6], and subsequent work investigates scalable pipelines for gigapixel WSI processing on leadership-class systems [7]. However, these approaches primarily focus on training or preprocessing and do not address embedding extraction pipelines or their I/O-dominant behavior.

On the data systems side, vector databases such as FAISS and Milvus enable scalable indexing and retrieval over large embedding collections [11], [12]. Emerging WSI ecosystems such as LazySlide emphasize interoperability and integration with downstream multimodal workflows [13]. However, these systems remain decoupled from upstream embedding pipelines, and embeddings are typically treated as transient outputs rather than reusable resources.

What is missing is a system that treats WSI embedding extraction itself as an I/O-dominant pipeline, rather than assuming patch access is cheap. WSI models focus on representation learning, HPC systems address data movement under different assumptions, and vector databases operate independently of data generation pipelines. No prior work jointly addresses I/O-dominant embedding extraction, pipeline decoupling, and embedding reuse in large-scale WSI processing.

## III. WSI PIPELINE AND PARALLEL EXECUTION MODEL

Our pipeline (Figure 1) is organized as three decoupled stages: patch generation, embedding inference, and vectordatabase ingestion. The key systems design choice is that each stage is parallelized according to its dominant bottleneck rather than forcing a single end-to-end execution strategy. In the current implementation, Stage 1 uses MPI-based spatial decomposition over patch coordinates, Stage 2 uses SPMD (Single Program Multiple Data) rank-parallel inference with one data shard per task/GPU, and Stage 3 uses shard-parallel database construction. This decoupling is central to making the workflow HPC-ready because it isolates I/O-heavy expansion, accelerator-heavy inference, and write-heavy persistence into independently scalable components.

<table><tr><td>Category</td><td>Model / System</td><td>Learning Type | Data Scale</td><td></td><td>Key Strength</td><td>Pipeline Coupling</td><td>I/O Aware- ness</td><td>Scalability</td><td>Embedding Reuse</td></tr><tr><td colspan="9">WSI Representation Models Used in This Work</td></tr><tr><td>Hierarchical</td><td>HIPT [2]</td><td>Hierarchical SSL Large-scale</td><td>10,678 WSIs 1.5M-3.1M+</td><td>Multi-scale tissue modeling Strong general-</td><td>N/A N/A</td><td>X X</td><td>N/A N/A</td><td>Partial Partial</td></tr><tr><td>Foundation Model Foundation Model</td><td>Virchow 1 Virchow2 [3] H-Optimus / H- Optimus-0 [4]</td><td>SSL Pathology FM</td><td>H&amp;E slides Large-scale histopathology</td><td>purpose features Pathology- specialized representation</td><td>N/A</td><td>X</td><td>N/A</td><td>Partial</td></tr><tr><td colspan="9">corpus learning WSI and HPC Pipelines</td></tr><tr><td>Traditional Pipelines</td><td>OpenSlide-based workflows [5]</td><td>Patch extraction + inference Distributed</td><td>Variable Large-</td><td>Simple and widely used Demonstrates HPC</td><td>Tight</td><td>X Partial</td><td>Limited High</td><td>X</td></tr><tr><td colspan="9">WSI training scale WSI viability for WSI- HPC pipeline [7] preprocessing workloads scale data HPC and Data Systems</td></tr><tr><td>HPC Workflows</td><td>Pegasus [8] Burst buffers, I/O</td><td>Workflow DAGs Storage</td><td>Large-scale science HPC</td><td>Decoupled execution Data movement opti-</td><td>Decoupled Decoupled</td><td>5 √</td><td>High High</td><td>N/A N/A</td></tr><tr><td>Vector Databases</td><td>studies [9], [10] FAISS / MiÍvus [11], [12] LazySlide [13]</td><td>optimization ANN indexing Interoperable WSI software</td><td>workloads Billion-scale vectors Large-scale WSI analysis</td><td>mization Fast retrieval Integration with downstream</td><td>Separate Partial</td><td>5 Partial</td><td>High Moderate</td><td>√ √</td></tr><tr><td colspan="9">multimodal workflows This Work</td></tr><tr><td>Decoupled WSI Pipeline</td><td>| Ours</td><td>Multi-stage pipeline</td><td>100M+ patches</td><td>I/O-dominant scaling with reusable em- beddings</td><td>Fully de- coupled</td><td>√</td><td>High (embar- rassingly parallel)</td><td>√ (persistent + metadata)</td></tr></table>

TABLE I: Comparison of representative WSI models, WSI/HPC pipelines, and data systems. Modern pathology models improve representation quality but generally assume efficient data access. Our work targets the orthogonal systems problem: scalable, I/O-aware embedding extraction with persistent reuse.

## Stage 1: Patch Generation (MPI Spatial Decomposition)

Patch generation is implemented as an MPI program over a single WSI at a specified pyramid level. All ranks independently construct the same global grid of valid patch coordinates. Work is distributed using deterministic cyclic partitioning, where rank r processes indices $r , r + R , r + 2 R , \ldots ,$ with R denoting the total number of MPI ranks.

For each assigned coordinate $( x _ { L } , y _ { L } )$ at level L, the rank maps to level-0 coordinates using the slide downsampling factor, reads the corresponding patch via read\_region, computes a white-background fraction, and writes the patch to a rank-local directory. Metadata including spatial coordinates, centroid, and patch path are stored locally.

No communication occurs in the steady-state extraction loop. Synchronization is limited to a single barrier, after which rank 0 gathers metadata from all ranks and writes a global index.csv. This design avoids communication in the critical path, but introduces significant filesystem pressure due to large-scale small-file writes.

This stage exhibits spatial data parallelism over the WSI coordinate space. The cyclic partitioning ensures balanced work distribution even under irregular tissue layouts, while avoiding synchronization during extraction. However, performance is dominated by small-file I/O and filesystem contention rather than compute.

Stage 2: Embedding Inference (SPMD Data Parallelism) Embedding extraction is executed in Single Program, Multiple Data (SPMD) fashion across distributed tasks, typically with one task per GPU. Each task independently loads the same patch index and constructs an identical dataset abstraction. Work is partitioned using strided indexing, where task r processes dataset entries $r , r + R , r + 2 R , \ldots .$

Each rank processes its subset using a local DataLoader, performing batched inference with a foundation model (e.g., HIPT, Virchow, H-Optimus-0). Depending on model capabilities, embeddings and attention tensors are extracted either in a single forward pass or in separate passes.

No collective communication (e.g., all-reduce) is used during inference. Instead, each rank accumulates results locally

# WSI → Multi-resolution Pyramid → Patches → Embeddings → Vector DB

Graphical abstract: patch-level embeddings of whole slide image with spatial mapping of different resolution and patient-level metadata , image-level metadata

![](images/30a37398f6fb24aa2e6fbccda0dec4e23fae7345eb32323cee375c3e3f00bef8.jpg)  
Fig. 1: Decoupled WSI embedding pipeline. The workflow is organized into three stages: (1) MPI-based patch generation and staging, (2) embarrassingly parallel embedding inference across GPUs, and (3) shard-parallel vector database ingestion. Each stage is parallelized according to its dominant constraint—storage, compute, and write throughput—enabling independent scaling. The pipeline produces a persistent, metadata-aware embedding database for downstream reuse.

```csv
Algorithm 1 Distributed Patch Generation with Cyclic Coor
dinate Partitioning
Require: WSI S, level L, patch size p, stride s, ranks
0, . . . , R − 1
1: Each rank opens S
2: G ← grid_coords(S, L, p, s)
3: G<sub>r</sub> ← {G[i] | i mod R = r}
4: for all (i, j, x<sub>L</sub>, y<sub>L</sub>) ∈ G<sub>r</sub> do
5: d ← level downsample factor
6: (x<sub>0</sub>, y<sub>0</sub>) ← (round(x<sub>L</sub> · d), round(y<sub>L</sub> · d))
7: P ← read_region(S, (x<sub>0</sub>, y<sub>0</sub>), L, (p, p))
8: ω ← white fraction of P
9: Save P to rank-local storage
10: Record metadata (x<sub>0</sub>, y<sub>0</sub>, x<sub>L</sub>, y<sub>L</sub>, ω, path)
11: end for
12: MPI Barrier
13: Rank 0 gathers metadata and writes index.csv
```

and serializes them to disk. Synchronization is deferred to a final consolidation step, where rank 0 gathers all rank outputs via file-based coordination, merges them, sorts patches by spatial coordinates, and writes per-WSI embedding files.

This stage exhibits embarrassingly parallel data parallelism. Each rank operates independently with no communication in the critical path. Unlike traditional distributed deep learning, there is no parameter synchronization or gradient exchange. The only coordination occurs at the end of execution, where outputs are merged via file-based aggregation.

## Stage 3: Vector Database Ingestion (Write-Parallel Sharding)

The final stage aggregates embeddings into a persistent vector database, enabling downstream retrieval and analytics. This stage is constrained primarily by write throughput and indexing overhead rather than compute. The shards are partitioned at file granularity to reduce coordination overhead. Vector database ingestion is implemented as streaming ingestion with batched inserts. We also have file-level checkpointing for fault tolerance.

This design exposes shard-level parallelism and can scale with the number of shards until limited by filesystem throughput while preserving spatial and metadata relationships. The embedding and metadata in vector database becomes a persistent resource rather than transient outputs, allowing downstream tasks to operate directly on stored representations without repeatedly accessing raw WSIs.

## IV. EXPERIMENTAL SETUP

## A. Dataset

We evaluate the proposed pipeline on 4,185 hematoxylin and eosin (H&E) whole-slide images from the Childhood Cancer Data Initiative Molecular Characterization Initiative, obtained via the National Cancer Institute Imaging Data Commons (CCDI MCI). The dataset spans a wide range of slide sizes and tissue content, yielding multi-resolution workloads from tens to more than $1 0 ^ { 5 }$ patches per WSI. This diversity enables evaluation across both small- and large-scale regimes and captures variability representative of real-world pathology cohorts.

Algorithm 2 SPMD Embedding Extraction with Rank-Local   
Shards   
Require: Patch index I, model $f _ { \theta } ,$ ranks $0 , \ldots , R - 1$   
1: Each rank loads I and constructs dataset D   
2: $\mathcal { D } _ { r }  \{ \mathcal { D } [ i ] \ | \ i$ mod $R = r \}$   
3: Initialize DataLoader over $\mathcal { D } _ { r }$   
4: for all mini-batch $B \subset { \mathcal { D } } _ { r }$ do   
5: Load patches and coordinates   
6: Apply preprocessing   
7: if model supports joint extraction then   
8: $( E , A )  f _ { \theta } ( B )$   
9: else   
10: $E  f _ { \theta } ^ { \mathrm { e m b e d } } ( B )$   
11: if attention supported then   
12: $A  f _ { \theta } ^ { \mathrm { a t t n } } ( B )$   
13: end if   
14: end if   
15: Append $( E , A ,$ coords) to local buffers   
16: end for   
17: Serialize local results to rank-specific file   
18: Write completion marker   
19: Rank 0 loads all rank outputs, merges and sorts by   
coordinates   
20: Write final embedding (and attention) files

Algorithm 3 Sharded Vector Database Ingestion   
Require: Embedding File set E, metadata M, shards K   
1: for all rank $r \in \{ 1 , \ldots , K \}$ in parallel do   
2: Initialize shard database $D _ { r }$   
3: Assign subset $( \mathcal { E } _ { r } , \mathcal { M } _ { r } )$   
4: for all $( e _ { i } , m _ { i } ) \in ( \mathcal { E } _ { r } , \mathcal { M } _ { r } )$ do   
5: Insert $( e _ { i } , m _ { i } )$ into $D _ { r }$   
6: end for   
7: Build local index on $D _ { r }$   
8: end for   
9: Parallel query across shards at application level or indi  
vidual shard can also be queried.

<table><tr><td rowspan=1 colspan=2>CCDI Dataset</td></tr><tr><td rowspan=1 colspan=1>No. of Tissue Samples</td><td rowspan=1 colspan=1>4185</td></tr><tr><td rowspan=1 colspan=1>No. of Patients</td><td rowspan=1 colspan=1>4054</td></tr><tr><td rowspan=1 colspan=1>No. of DICOMS</td><td rowspan=1 colspan=1>19,603</td></tr><tr><td rowspan=1 colspan=1>No. of Patches (256x256)</td><td rowspan=1 colspan=1>414M</td></tr><tr><td rowspan=1 colspan=1>No. of Non-White Patches</td><td rowspan=1 colspan=1>170M</td></tr></table>

TABLE II: CCDI Dataset Statistics

## B. Models

We consider three representative foundation models—HIPT, H-Optimus-0, and Virchow2—which differ in patch resolution, embedding dimensionality, and computational characteristics.

This variation allows us to evaluate whether observed system behavior is consistent across models with different compute intensity and output sizes.

## C. System and Hardware

All experiments are conducted on the Frontier system at Oak Ridge National Laboratory. Each compute node consists of a single 64-core AMD EPYC CPU paired with 8 AMD Instinct MI250X GPUs, interconnected via the Slingshot highspeed network.

The system is backed by the Orion parallel filesystem, a Lustre-based storage system designed for exascale workloads. Orion provides high aggregate bandwidth but is subject to metadata and small-file access constraints under high concurrency. These characteristics are particularly relevant for our workload, which generates and accesses millions of small patch files concurrently across nodes.

This hardware configuration directly influences observed performance behavior: while GPU compute capacity is abundant, effective throughput is governed by filesystem bandwidth and metadata scalability. As a result, the experiments capture realistic system-level bottlenecks encountered in data-intensive workloads on leadership-class HPC systems.

## D. Experimental Methodology

The evaluation covers all three stages of the pipeline. Patch generation is executed using MPI-based CPU parallelism over spatial coordinates, producing large collections of small files and exposing filesystem-level scaling behavior. Embedding extraction is evaluated under both multi-node distributed execution (for large workloads) and single-GPU per-WSI execution (for small workloads), enabling comparison across parallelization regimes. The vector database ingestion stage is evaluated using shard-parallel insertion into a distributed vector database, capturing write-intensive behavior at scale.

To isolate system bottlenecks in the embedding stage, we organize experiments into four setups. Setup A varies patch count for a fixed WSI and hardware configuration to characterize the transition from compute-bound to I/Odominated execution. Setup B performs strong-scaling analysis by increasing GPU count for a fixed workload, revealing the impact of storage contention on parallel efficiency. Setup C uses kernel-level profiling to verify that GPU execution remains efficient and that end-to-end limitations arise from data movement rather than compute. Setup D evaluates productionscale behavior across thousands of WSIs, capturing variability in workload size, runtime, and scheduling dynamics.

Across all stages, we measure runtime, throughput, speedup, and efficiency to distinguish compute scaling from storagelimited behavior. Stage-specific metrics include file counts, output size, average file size, and write throughput for patch generation; GPU utilization, I/O fraction, and per-WSI runtime for embedding extraction; and insertion rate, total runtime, and memory footprint for vector database ingestion.

## V. RESULTS

We evaluate the proposed pipeline to understand how performance evolves with scale and to identify the dominant bottlenecks at each stages. Across all experiments, a consistent pattern emerges: although each stage is embarrassingly parallel in isolation, end-to-end performance is ultimately limited by data movement and storage system behavior rather than compute or communication.

## A. Patch Generation Scaling

Table II shows patch-generation performance on a single node (56 MPI ranks). Increasing patch size from 224 to 256 reduces file count while keeping total output size nearly constant, resulting in consistently lower runtime and higher write throughput. For example, reducing file count by 25% yields proportional throughput improvements, despite a negligible change in total bytes written. This shows that performance is driven by the number of files (metadata overhead), not data volume .

Strong scaling results shown in Table III further support the conclusions that performance is bound by the number of files. Increasing ranks from 56 to 224 yields a 1.69× speedup (42.3% efficiency), while aggregate throughput increases sublinearly. Despite the absence of communication, scaling degrades due to shared filesystem contention. These results show that patch generation transitions from communication-free overhead to I/O-dominated performance as parallelism increases, with additional resources increasing storage pressure rather than improving performance.

## B. Embedding Inference Analysis

## Setup A: Patch Sweep — Identifying I/O-Dominated Behavior

Figure 2 evaluates performance as patch count increases for a fixed WSI and hardware configuration. At small scales, throughput increases with workload size and execution remains compute-dominated. As patch count increases, throughput plateaus while I/O wait-time increases, indicating that the movement of the data becomes the limiting factor (Figure 2b,Figure 2c).

GPU utilization remains modest at all scales (Figure 2a), despite sufficient computing capacity. This confirms that accelerators are underutilized due to input data latency rather than insufficient compute. As workload size increases, throughput flattens, I/O wait grows, and GPU utilization stays modest, indicating a transition to I/O-dominated execution even without communication overhead.

## Setup B: Strong Scaling — From Communication-Free to I/O-Dominated

Figure 3 shows strong scaling behavior across HIPT, H-Optimus-0, and Virchow2. Throughput increases with GPU count, but scaling is not linear, with efficiency dropping below 30–50% at higher concurrency. This degradation occurs despite fully independent execution without collective communication.

The key result is that communication-free execution (in critical path) does not guarantee scalability. Instead, performance is limited by shared storage bandwidth and metadata contention as all workers concurrently load patch data. Increasing parallelism saturates the I/O path, leading to diminishing returns and early efficiency collapse.

Setup C: Kernel-Level Analysis — Not Compute-Bound Kernel-level profiling (Fig. 4a) shows that model execution operates in compute-efficient regions relative to hardware limits. Across models, kernels achieve high arithmetic intensity and do not approach memory-bound ceilings, indicating that GPU execution itself is efficient.

However, kernel durations are short (Fig. 4b), exposing latency between batches. As a result, GPUs stall waiting for data despite efficient kernel execution. This confirms that the embedding stage is not compute-bound in practice, and that performance limitations arise from upstream data delivery rather than model inefficiency.

## Setup D: Production-Scale Behavior — Orchestration and Variability

Production-scale results across 4,185 WSIs (Fig. 5a) show that throughput increases with patch count due to amortization of fixed overheads. However, significant variance remains across runs, reflecting storage contention and heterogeneous slide characteristics. Larger WSIs benefit from better amortization, while smaller WSIs suffer from disproportionate I/O costs.

Per-worker completion times (Fig. 5b) reveal substantial spread, with stragglers determining overall runtime. This variability highlights the importance of orchestration and load balancing in large-scale deployments. These results confirm that I/O-dominated behavior persists at production scale and is amplified by workload heterogeneity and system-level contention.

## C. Vector Database Ingestion Analysis

Table VII shows strong scaling for VDB ingestion. Throughput increases from 22K to 188K rows/s (8.5× speedup), but efficiency peaks at intermediate scale (86.7% at 4 nodes) and declines at higher concurrency. This nonmonotonic behavior reflects the balance between parallelism and filesystem contention.

At low concurrency, available bandwidth is underutilized; at high concurrency, metadata and read contention dominate. Importantly, VDB ingestion remains I/O-bound, with performance determined by read bandwidth and write amplification during indexing rather than compute. This mirrors behavior observed in earlier stages.

Notably, the ingestion stage is embarrassingly parallel from a compute perspective: each rank independently processes a subset of embedding files and writes to a separate shard without coordination. However, all ranks share the underlying storage system, making I/O the dominant bottleneck. As concurrency increases, contention for read bandwidth and metadata access limits scalability, leading to diminishing returns.

<table><tr><td>Slide</td><td>Patch Size</td><td></td><td></td><td></td><td></td><td></td><td>Nodes MPI Ranks Runtime (s) Files Created Output Size (GB) Avg File Size (KB)</td><td>Write Throughput (MB/s)</td></tr><tr><td>ff93fcc3</td><td>224</td><td>1</td><td>56</td><td>19.82</td><td>93,696</td><td>2.42</td><td>25.8</td><td>116.22</td></tr><tr><td>ff93fcc3</td><td>256</td><td>1</td><td>56</td><td>17.21</td><td>71,773</td><td>2.40</td><td>33.4</td><td>133.03</td></tr><tr><td>d47c2741</td><td>224</td><td>1</td><td>56</td><td>40.67</td><td>184,785</td><td>6.75</td><td>36.5</td><td>158.20</td></tr><tr><td>d47c2741</td><td>256</td><td>1</td><td>56</td><td>34.11</td><td>141,561</td><td>6.71</td><td>47.4</td><td>187.60</td></tr><tr><td>ca6e4e7a</td><td>224</td><td>1</td><td>56</td><td>47.24</td><td>224,133</td><td>9.94</td><td>44.4</td><td>200.67</td></tr><tr><td>ca6e4e7a</td><td>256</td><td>1</td><td>56</td><td>42.02</td><td>171,702</td><td>9.88</td><td>57.5</td><td>224.27</td></tr></table>

TABLE III: Stage 1 patch-generation performance for three representative slides at 56 MPI ranks using patch sizes 224 and 256. The selected slides span smaller, medium, and larger output footprints in the benchmark set. Increasing patch size reduces the number of generated files while increasing average file size and effective write throughput.
<table><tr><td>Patch Size</td><td>Nodes</td><td>MPI Ranks</td><td>Runtime (s)</td><td>Speedup</td><td>Efficiency (%)</td><td>Files Created</td><td>Write Throughput (MB/s)</td></tr><tr><td>224</td><td>1</td><td>56</td><td>22.39</td><td>1.00×</td><td>100.0</td><td>93,696</td><td>102.66</td></tr><tr><td>224</td><td>2</td><td>112</td><td>18.85</td><td>1.19×</td><td>59.4</td><td>93,696</td><td>122.99</td></tr><tr><td>224</td><td>4</td><td>224</td><td>13.22</td><td>1.69×</td><td>42.3</td><td>93,696</td><td>177.50</td></tr><tr><td>256</td><td>1</td><td>56</td><td>16.89</td><td>1.00×</td><td>100.0</td><td>71,773</td><td>135.46</td></tr><tr><td>256</td><td>2</td><td>112</td><td>13.41</td><td>1.26×</td><td>63.0</td><td>71,773</td><td>171.57</td></tr><tr><td>256</td><td>4</td><td>224</td><td>9.56</td><td>1.77×</td><td>44.2</td><td>71,773</td><td>244.02</td></tr></table>

TABLE IV: Stage 1 strong-scaling results for one representative slide (ID ff93fcc3-7bc2-4656-8e2d-e379180fa042) using patch sizes 224 and 256 across 56, 112, and 224 MPI ranks. Speedup and parallel efficiency are computed relative to the 56-rank baseline for each patch size. The sublinear scaling and rising write throughput are consistent with the I/O-dominated behavior

<table><tr><td>Model</td><td>GPUs</td><td>Throughput</td><td>Speedup</td><td>Efficiency</td><td>I/O %</td></tr><tr><td>h-optimus-0</td><td>1</td><td>110.7</td><td>1.00x</td><td>100.0%</td><td>1.8</td></tr><tr><td>h-optimus-0</td><td>2</td><td>187.7</td><td>1.70x</td><td>84.8%</td><td>2.5</td></tr><tr><td>h-optimus-0</td><td>4</td><td>284.0</td><td>2.57x</td><td>64.1%</td><td>3.5</td></tr><tr><td>h-optimus-0</td><td>8</td><td>362.5</td><td>3.27x</td><td>40.9%</td><td>6.5</td></tr><tr><td>h-optimus-0</td><td>16</td><td>600.3</td><td>5.42x</td><td>33.9%</td><td>10.7</td></tr><tr><td>h-optimus-0</td><td>40</td><td>855.8</td><td>7.73x</td><td>19.3%</td><td>15.5</td></tr><tr><td>hipt</td><td>1</td><td>153.7</td><td>1.00x</td><td>100.0%</td><td>35.4</td></tr><tr><td>hipt</td><td>2</td><td>224.8</td><td>1.46x</td><td>73.1%</td><td>25.7</td></tr><tr><td>hipt</td><td>4</td><td>352.9</td><td>2.30x</td><td>57.4%</td><td>20.3</td></tr><tr><td>hipt</td><td>8</td><td>407.1</td><td>2.65x</td><td>33.1%</td><td>9.5</td></tr><tr><td>hipt</td><td>16</td><td>719.3</td><td>4.68x</td><td>29.2%</td><td>10.2</td></tr><tr><td>hipt</td><td>40</td><td>1104.1</td><td>7.18x</td><td>18.0%</td><td>15.0</td></tr><tr><td>virchow2</td><td>1</td><td>69.8</td><td>1.00x</td><td>100.0%</td><td>1.2</td></tr><tr><td>virchow2</td><td>2</td><td>125.5</td><td>1.80x</td><td>89.9%</td><td>1.8</td></tr><tr><td>virchow2</td><td>4</td><td>210.5</td><td>3.02x</td><td>75.4%</td><td>2.3</td></tr><tr><td>virchow2</td><td>8</td><td>318.9</td><td>4.57x</td><td>57.1%</td><td>5.1</td></tr><tr><td>virchow2</td><td>16</td><td>555.6</td><td>7.96x</td><td>49.7%</td><td>6.6</td></tr><tr><td>virchow2</td><td>40</td><td>826.9</td><td>11.85x</td><td>29.6%</td><td>12.9</td></tr></table>

TABLE V: Strong scaling of embedding inference across models. Throughput increases with GPU count, but efficiency degrades and I/O fraction rises with concurrency. This indicates that scaling is limited by shared storage bandwidth rather than compute or communication.

Memory usage remains stable across configurations (approximately 0.6 GB per rank), confirming that the ingestion process follows a streaming design without significant inmemory accumulation. This indicates that performance limitations arise from external I/O constraints rather than internal resource pressure.

Overall, these results demonstrate that vector database ingestion, like earlier pipeline stages, is fundamentally I/Obound. While shard-level parallelism enables substantial throughput gains, scalability is ultimately constrained by shared filesystem behavior. The 4-node configuration represents a practical operating point, balancing parallelism with manageable storage contention.

## VI. DISCUSSION

Our results consistently show that large-scale WSI embedding extraction deviates from classical compute-bound scaling assumptions and instead exhibits I/O-dominated behavior across all pipeline stages. While each stage (patch generation, embedding inference, and vector database ingestion) exhibits ideal parallel complexity in isolation, end-to-end performance is governed by data movement and storage system constraints.

From a scaling perspective, patch generation nominally follows $\mathcal { O } ( N / R )$ work distribution across ranks, but in practice incurs significant per-patch overhead due to small-file creation and metadata operations. As shown in Tables III, IV, increasing parallelism yields sublinear speedup despite the absence of communication, indicating that filesystem contention—not computation—limits scalability. This behavior is driven by the expansion of WSIs into millions of independent patches, where metadata overhead becomes comparable to data movement.

A similar pattern emerges in the embedding stage. Although model inference is computationally efficient—as confirmed by kernel-level analysis (Fig 4); GPU utilization remains modest and throughput plateaus as patch counts increase (Fig 2). The limiting factor is not compute capacity, but input data latency: GPUs stall between batches waiting for patch delivery. Strong-scaling results (Fig. 3) further reinforce this point—performance degrades with increasing GPU count despite fully independent execution, demonstrating that communication-free designs do not eliminate scaling bottlenecks when storage bandwidth is shared.

GPU Utilization vs Patch Count (8 GPUs, same WSI)  
![](images/4f65fa37a04bfd29c86d7e79a36628eac8de2cb027ec83fc35385d732dfcc662.jpg)  
(a) Mean GPU utilization.

![](images/29cdfebdf061fe2b38f84c6a016a187cb024f1101a6b59885f6f7facf39c06ed.jpg)  
(b) Throughput vs patch count.

Setup A: I/O vs Compute Breakdown vs Patch Count (8 GPUs, same WSI)  
![](images/cdeb73ddd98d86ba0c6d95dd5b384fe750a5fa4d43fdfcc62407cb71455f46f1.jpg)  
(c) Extraction-time breakdown by phase.

Fig. 2: Setup A (Patch Sweep). As patch count increases, throughput plateaus while I/O wait grows and GPU utilization remain modest, indicating a transition from compute-bound to I/O-dominated (storage-limited) execution. This shows that increasing workload size does not improve accelerator utilization when data delivery becomes the bottleneck.
<table><tr><td>Model</td><td>Total Rows</td><td>Avg Rate</td><td>Max Rate</td><td>Files</td><td>Mean File Time (s)</td><td>Total Time (min)</td><td>RSS Max (GB)</td></tr><tr><td>HIPT</td><td>6,143,506</td><td>2628.88</td><td>2775</td><td>14,499</td><td>4.27</td><td>1032.18</td><td>0.62</td></tr><tr><td>Optimus</td><td>2,298,630</td><td>258.71</td><td>321</td><td>4,052</td><td>55.59</td><td>3754.24</td><td>1.68</td></tr><tr><td>Virchow2</td><td>5,413,012</td><td>131.86</td><td>155</td><td>11,726</td><td>116.52</td><td>22772.80</td><td>2.74</td></tr></table>

TABLE VI: Performance comparison of large-scale VDB ingestion of embedding across three foundation models. Rows are sorted by average ingestion throughput (rows/sec), highlighting substantial performance differences across embedding backbones. HIPT achieves the highest throughput and lowest per-file processing time, while Virchow2 exhibits significantly slower ingestion despite processing a comparable number of rows. We also observe increased memory footprint and tota runtime for larger embedding sizes (e.g., Virchow2), emphasizing the importance of model efficiency in distributed vector database construction.

Vector database ingestion exhibits the same fundamental constraint. While shard-parallel insertion enables throughput improvements (Table VII), efficiency peaks at intermediate scale and declines under higher concurrency due to read contention and write amplification. This confirms that even downstream stages, which are conceptually independent, remain bounded by storage system behavior.

These observations can be summarized by a simple performance model:

$$
T ( R ) = \operatorname* { m a x } \left( \frac { N } { R } , \frac { N \cdot B } { { \bf B } { \bf W } _ { \mathrm { s t o r a g e } } } \right)
$$

where N is the total number of patches to process, R is the effective compute processing rate, B is the average number of bytes transferred per patch, and $\mathbf { B W _ { \mathrm { s t o r a g e } } }$ is the effective storage bandwidth. Increasing compute capacity reduces the compute term, $N / R _ { : }$ , but does not reduce the data-transfer term, $N B / \mathrm { B W _ { s t o r a g e } } .$ . This explains the observed plateau in throughput and collapse in parallel efficiency beyond moderate scale.

![](images/3a3328218009e5ccf0777a44922e013eb662c9478930b8cc177e3661e69f013c.jpg)

![](images/40c284ed340d371eb51b86ffff1b93e8c52931e4a609e2e9c714ef9105708e77.jpg)

Fig. 3: Setup B. Strong-scaling efficiency versus GPU count across models. Despite fully independent execution with no collective communication in the critical path, scaling deviates from linear and efficiency drops at higher concurrency. This demonstrates that performance is limited by shared storage bandwidth rather than communication overhead.
<table><tr><td>Nodes</td><td>Agg. Rate (rows/s)</td><td>Wall Time (min)</td><td>Speedup</td><td>Efficiency (%)</td><td>RSS Max (GB)</td></tr><tr><td>1</td><td>22,216*</td><td>127.9*</td><td>1.00×</td><td>100.0</td><td>0.58</td></tr><tr><td>2</td><td>40,694</td><td>83.84</td><td>1.53×</td><td>76.3</td><td>0.59</td></tr><tr><td>4</td><td>86,099</td><td>36.90</td><td>3.47×</td><td>86.7</td><td>0.60</td></tr><tr><td>8</td><td>188,020</td><td>20.79</td><td>6.15×</td><td>76.9</td><td>0.59</td></tr></table>

TABLE VII: Strong scaling ablation for HIPT embedding VDB ingestion on Frontier. Each configuration ingests the same ∼170M embedding rows into Milvus; we vary the number of nodes (16 ranks each) and report aggregate throughput, wallclock time, speedup, and parallel efficiency (speedup/N). <sup>∗</sup>Extrapolated from observed rate; the 1-node run did not complete within the 2-hour allocation. The 4-node configuration achieves the highest parallel efficiency (86.7%) and corresponds to the production run in Table VI.

Importantly, this I/O-dominated behavior is not merely a systems artifact but a direct consequence of the biomedical workload structure. WSIs are inherently high-resolution and large-scale, and meaningful analysis requires preserving patchlevel granularity. This transforms compact slide data into massive collections of small, independently processed objects, making data movement unavoidable. In addition, embeddings must remain linked to their originating patches and spatial metadata to support interpretability and downstream clinical use. As a result, eliminating data movement is neither feasible nor desirable.

Instead, our results suggest a shift in system design priorities: optimizing compute alone is insufficient. Scalable WSI pipelines must explicitly account for data orchestration, storage access patterns, and intermediate data representation. The decoupled pipeline design proposed in this work addresses this requirement by isolating I/O-heavy, compute-heavy, and write-heavy stages, enabling independent optimization and reducing cross-stage interference.

Overall, our findings reposition large-scale WSI embedding extraction as a data-centric systems problem, where performance is governed not by accelerator throughput or communication, but by the ability to efficiently generate, move, and persist massive intermediate datasets.

## VII. CONCLUSION

We presented a decoupled, I/O-aware pipeline for largescale whole-slide image embedding extraction that separates patch generation and staging, embarrassingly parallel inference, and sharded vector database ingestion. Across controlled benchmarks and production-scale runs, we showed that this workload is fundamentally constrained by data movement and orchestration, rather than by collective communication or peak accelerator throughput. Throughput plateaus, modest GPU utilization, rising I/O fractions under scaling, and substantial VDB ingestion costs all point to the same conclusion: largescale WSI embedding extraction is best understood as a datacentric systems problem.

Finally, the pipeline produces a persistent vector database in which embeddings are stored together with their associated metadata, including spatial and slide-level context. This representation allows downstream tasks to operate directly on embeddings without repeatedly accessing raw WSIs, avoiding redundant patch extraction and reducing I/O overhead. As a result, embedding extraction becomes a one-time data generation stage whose outputs can be reused across analyses. This is particularly important at scale, where data movement dominates runtime and repeated access to raw images is costly.

![](images/d3a651d377d866a6b212331e829ebe3705916b59b496ef586524467043e8120b.jpg)  
(a) Roofline positioning of model kernels against hardware compute/bandwidth ceilings. We include this to distinguish kernel-local limits from system-level limits. Even when kernels operate in compute-favorable regions, the overall pipeline can remain data-movement-bound due to upstream patch delivery constraints.

Setup C: GPU Kernel Time Breakdown by Category  
![](images/fe94bb22e9be94666853989949c861665b2cad82f4d5724b2c34a32c67a422e8.jpg)

![](images/7cfe89222b25f7ae7faed1e4bd2a183e69b96839420abece246ae3881a156f1a.jpg)

![](images/e33e9f7482d710475ed741a9aa5edff5237ff535fab67b55fd75e14483c8873a.jpg)  
Fig. 4: Setup C (Kernel-level analysis)  
(b) Kernel-category time composition (e.g., GEMM, convolution, elementwise) by model. We include this to explain modeldependent I/O sensitivity through internal compute structure: kernels may be efficient yet short enough that filesystem latency becomes exposed between batches

## VIII. ACKNOWLEDGMENTS

OLCF: This research used resources of the Oak Ridge Leadership Computing Facility, which is a US Department of Energy (DOE) Office of Science User Facility supported under Contract DE-AC05-00OR22725. RADIANCE: This work has been supported by DOE’s Office of Science, Office of Advanced Scientific Computing under award number DE-SC-KJ0403010. UT Battelle: This research has been authored by UT-Battelle LLC under contract number DE-AC05- 00OR22725 with the DOE.

## REFERENCES

[1] S. Atchley, C. Zimmer, J. Lange, D. Bernholdt, V. Melesse Vergara, T. Beck, M. Brim, R. Budiardja, S. Chandrasekaran, M. Eisenbach, T. Evans, M. Ezell, N. Frontiere, A. Georgiadou, J. Glenski, P. Grete, S. Hamilton, J. Holmen, A. Huebl, D. Jacobson, W. Joubert,

K. Mcmahon, E. Merzari, S. Moore, A. Myers, S. Nichols, S. Oral, T. Papatheodore, D. Perez, D. M. Rogers, E. Schneider, J.-L. Vay, and P. K. Yeung, “Frontier: Exploring exascale,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, ser. SC ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3581784.3607089

[2] R. J. Chen, M. Y. Lu, M. T. Shaban, T. Y. Chen, D. F. Williamson, and F. Mahmood, “Scaling vision transformers to gigapixel images via hierarchical self-supervised learning,” in Proceedings of CVPR, 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2206.02647

[3] G. Jaume et al., “A foundation model for clinical-grade computational pathology and rare cancers detection,” Nature Medicine, 2024. [Online]. Available: https://doi.org/10.1038/s41591-024-03141-0

[4] C. Saillard, R. Jenatton, F. Llinares-Lopez, Z. Mariet, D. Cahan´ e,´ E. Durand, and J.-P. Vert, “H-optimus-0,” 2024. [Online]. Available: https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0

[5] A. Goode, B. Gilbert, J. Harkes, D. Jukic, and M. Satyanarayanan, “Openslide: A vendor-neutral software foundation for digital pathology,” Journal of Pathology Informatics, vol. 4, 2013. [Online]. Available: https://doi.org/10.4103/2153-3539.119005

![](images/9c3a4226171a8b0f20db155372f09829434a64256f222119d98210f1e1b5e1e1.jpg)  
(a) Setup D. Throughput versus WSI patch count across production levels. We include this to test generalization beyond controlled sweeps. Larger WSIs amortize fixed overheads better, while residual variance captures storage jitter/contention and heterogeneous slide complexity seen in real deployments.

Setup D: Per-Worker Completion Times — WSI ced49a97... (Level 0, 5 nodes, 40 GPUs)  
![](images/9c4aa8658e9535ec940b2944d6d740629aa7fdbc45426a06ed0a99d496f8df18.jpg)  
(b) Setup D. Per-worker completion-time variance for representative runs. We include this because wall-clock stage time is determined by stragglers. Large spreads indicate non-trivial idle tails and motivate work-stealing and better load balancing for I/O-sensitive model settings.  
Fig. 5: Setup D (Production Scale Behavior)

[6] S. Mohanty et al., “Gems: Gpu-enabled memory-aware modelparallelism system for distributed dnn training,” in SC Conference, 2020. [Online]. Available: https://doi.org/10.1109/SC41405.2020.00049

[7] S. Dash, B. Hernandez, A. Tsaris, F. T. Alamudun, H.-J. Yoon, and´ F. Wang, “A scalable pipeline for gigapixel whole slide imaging analysis on leadership class hpc systems,” in 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2022, pp. 1266–1274. [Online]. Available: https://doi.org/10.1109/ IPDPSW55747.2022.00223

[8] E. Deelman, K. Vahi, G. Juve, M. Rynge, S. Callaghan, P. J. Maechling, R. Mayani, W. Chen, R. F. da Silva, M. Livny, and K. Wenger, “Pegasus: A workflow management system for science automation,” Future Generation Computer Systems, vol. 46, pp. 17–35, 2015. [Online]. Available: https://doi.org/10.1016/j.future.2014.10.008

[9] H. Shan, K. Antypas, and J. Shalf, “Characterizing and predicting the i/o performance of hpc applications using a parameterized synthetic benchmark,” in Proceedings of the 2008 ACM/IEEE Conference on Supercomputing, ser. SC ’08. IEEE Press, 2008.

[10] N. Liu, J. Cope, P. Carns, C. Carothers, R. Ross, G. Grider, A. Crume, and C. Maltzahn, “On the role of burst buffers in leadership-class storage systems,” in 2012 IEEE 28th Symposium on Mass Storage Systems and Technologies (MSST), 2012, pp. 1–11. [Online]. Available: https://doi.org/10.1109/MSST.2012.6232369

[11] J. Johnson, M. Douze, and H. Jegou, “Billion-scale similarity search with´ gpus,” IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2019. [Online]. Available: https://doi.org/10.1109/TBDATA.2019.2921572

[12] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu et al., “Milvus: A purpose-built vector data management system,” in Proceedings of the 2021 International Conference on Management of Data (SIGMOD ’21). New York, NY, USA: Association for Computing Machinery, 2021, pp. 2614–2627. [Online]. Available:

https://doi.org/10.1145/3448016.3457550

[13] Y. Zheng, E. Abila, E. Chrenkova, I. Buljan, J. Winkler, and´ A. F. Rendeiro, “Lazyslide: accessible and interoperable whole-slide image analysis,” Nature Methods, pp. 1–4, 2026. [Online]. Available: https://doi.org/10.1038/s41592-026-03044-7

[14] G. Campanella, M. G. Hanna, L. Geneslaw, A. Miraflor, V. Silva, K. J. Busam, E. Brogi, V. E. Reuter, D. S. Klimstra, and T. J. Fuchs, “Clinical-grade computational pathology using weakly supervised deep learning on whole slide images,” Nature Medicine, vol. 25, no. 8, pp. 1301–1309, 2019. [Online]. Available: 10.1038/s41591-019-0508-1

[15] M. Y. Lu, D. F. Williamson, T. Y. Chen, R. J. Chen, M. Barbieri, and F. Mahmood, “Data-efficient and weakly supervised computational pathology on whole-slide images,” Nature Biomedical Engineering, vol. 5, no. 6, pp. 555–570, 2021. [Online]. Available: https: //doi.org/10.1038/s41551-020-00682-w

[16] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2304.07193

[17] T. Ding, S. J. Wagner, A. H. Song, R. J. Chen, M. Y. Lu, A. Zhang, A. J. Vaidya, G. Jaume, M. Shaban, A. Kim et al., “A multimodal whole-slide foundation model for pathology,” Nature medicine, pp. 1–13, 2025. [Online]. Available: https://doi.org/10.1038/s41591-025-03982-3

[18] Y. Xu, Y. Wang, F. Zhou, J. Ma, C. Jin, S. Yang, J. Li, Z. Zhang, C. Zhao, H. Zhou et al., “A multimodal knowledge-enhanced wholeslide pathology foundation model,” Nature Communications, 2025. [Online]. Available: https://doi.org/10.1038/s41467-025-66220-x