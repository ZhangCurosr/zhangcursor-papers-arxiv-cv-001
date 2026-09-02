# Do Satellites See Commuters? A Critical Benchmark of Vision Foundation Models for Generative Urban Flow Networks

Ashiq Shukoor Iqbal   
The University of New South Wales Sydney, NSW, Australia   
ashiq\_shukoor.iqbal@student.unsw.edu.au u

Wilson Wongso The University of New South Wales Sydney, NSW, Australia w.wongso@unsw.edu.au

Flora Salim The University of New South Wales Sydney, NSW, Australia flora.salim@unsw.edu.au

## Abstract

Satellite foundation models ofer a globally available alternative to census data for commuting origin-destination (OD) generation, yet no study has systematically compared encoder paradigms within a single downstream pipeline. We ablate four satellite vision encoders: language-supervised (RemoteCLIP), self-supervised (DINOv3), and geographically grounded (SatCLIP, AlphaEarth) within an identical WeDAN graph difusion framework across 1,925 US counties, 325 UK districts, and 14 global cities under five random seeds. Three main findings emerge. First, language-supervised features achieve the strongest in-distribution performance (RemoteCLIP CPC 0.602), while geographically grounded encoders transfer more reliably zero-shot: AlphaEarth improves CPC by 33% over RemoteCLIP on UK districts. Second, pretraining corpus scale alone is insuficient: DINOv3, trained on a substantially larger satellite corpus, underperforms RemoteCLIP by 0.091 CPC in-distribution and collapses to CPC 0.022 globally. Third, no encoder transfers usefully to global cities (best CPC 0.122 for RemoteCLIP, 0.022 for DINOv3), confirming cross-continental OD generation remains an open problem. We additionally clarify the semantics of the census noise parameter �, whose ordering reverses under cross-continental evaluation, a distinction critical to correctly interpreting prior results. Training scripts and evaluation logs will be released.

## CCS Concepts

• Computing methodologies → Generative models; Computer vision; • Human-centered computing → Geographic information systems.

## Keywords

Generative graph difusion, satellite imagery, vision foundation models, origin-destination matrix generation, encoder ablation

## ACM Reference Format:

Ashiq Shukoor Iqbal, Wilson Wongso, and Flora Salim. 2026. Do Satellites See Commuters? A Critical Benchmark of Vision Foundation Models for Generative Urban Flow Networks. In The 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’26), November 03–06, 2026, Riverside, CA, USA. ACM, New York, NY, USA, 12 pages. https: //doi.org/10.1145/3841645.3842989

![](images/6d4afceef5d885b40bff48797231504bae3e14db7fcf83b41b3035f08666d615.jpg)

## 1 Introduction

Modeling urban mobility patterns via Origin-Destination (OD) matrices represents a cornerstone of contemporary civil infrastructure planning, transportation system optimization, and macro-level spatiotemporal analysis. Historically, the extraction of these granular travel flow networks has depended heavily on localized administrative census datasets, household travel surveys, or aggregated signaling logs sourced from proprietary mobile network operators. However, these traditional collection methods sufer from severe financial and operational bottlenecks: comprehensive census registries are typically updated only at decadal intervals, household surveys scale poorly due to low response frequencies, and telecommunication logs are strictly constrained by privacy regulations and commercial siloing. This pervasive data gap is particularly acute within medium-sized municipalities in emerging economies and rapidly transforming urban zones, where the absence of baseline demographic records leaves municipal planners without actionable structural insights.

To circumvent these multi-source data dependencies, recent deep generative architectures have reframed the problem of mobility network synthesis, modeling OD generation as a conditional graph denoising task. Architectures like GlODGen [16] completely eliminate the reliance on downstream tabular census parameters by utilizing high-resolution satellite imagery and globally uniform WorldPop statistics as their exclusive input modalities. By processing region-level satellite tiles through RemoteCLIP [12], a visionlanguage foundation model pre-trained on remote sensing imagery and natural language captions, and conditioning the underlying WeDAN graph difusion pipeline [15] on the resulting visual features, this framework demonstrates that satellite-derived signals are highly expressive of underlying human commuting patterns. Under in-distribution settings, this satellite-conditioned configuration achieves a Common Part of Commuters (CPC) score of 0.623 across 1,925 US counties, recovering 98.3% of full-census WeDAN performance (CPC 0.623 vs. 0.634). Our replication of GlODGen under identical conditions yields CPC = 0.602, confirming the encoder’s signal while establishing the ablation baseline.

Despite these strong results, existing frameworks treat the underlying satellite vision encoder backbone, RemoteCLIP, as an architectural primitive. Since the initial formulation of satellite-conditioned graph difusion, the geospatial foundation model landscape has expanded significantly, yielding distinct self-supervised paradigms optimized via fundamentally distinct pretraining objectives. For example, self-distillation networks such as DINOv3 [17] leverage massive, unlabeled image repositories to capture fine-grained pixel textures and geometric boundaries. Coordinate-contrastive models like SatCLIP [10] map localized visual patches directly to universal geographic coordinates to enforce explicit spatial grounding. Finally, multi-sensor frameworks such as AlphaEarth [1] combine diverse multi-spectral, radar, and elevation datastreams into highly compressed digital signatures. The current literature lacks a system atic comparison of vision encoders within a fixed generative graph pipeline, while these encoders difer in objective, dimension, and preprocessing. Consequently, it remains an open question whether scene-level text-aligned features via RemoteCLIP are a strict requirement for modeling human mobility, or if self-supervised structural representations can provide equivalent or superior downstream generalizability.

This paper directly addresses this research gap by presenting a systematic empirical evaluation of geospatial vision foundation models embedded within generative graph difusion architectures. Keeping the same downstream WeDAN graph transformer pipeline, we evaluate the representational fidelity of four vision encoder paradigms across a multi-scale benchmarking pipeline. Our experimental framework evaluates in-distribution performance using 1,925 county-level networks within the United States, followed by zero-shot cross-continental transferability testing across 325 Local Authority Districts in the United Kingdom and 14 morphologically distinct global metropolitan cities. Every configuration is evaluated across five independent random seeds, using paired �-tests over matched seeds to verify the statistical reliability of observed performance changes. Our main contributions are summarized as follows:

• We conduct the first controlled empirical ablation isolating the impact of visual representation spaces on generative OD graph difusion, evaluating text-aligned models (Remote-CLIP), self-distillation architectures (DINOv3), coordinatecontrastive models (SatCLIP), and multi-sensor physical embeddings (AlphaEarth).

• We formally clarify the semantics of the census noise parameter (�) in the WeDAN architecture, demonstrating through architectural analysis that it measures census data availabil ity rather than satellite feature noise, which establishes the correct operating point for cross-continental evaluation.

• We analyze a representational tradeofacross vision encoders that difer in pretraining corpus scale, objective, architecture, and preprocessing, highlighting that larger-scale pretraining does not necessarily translate to better representations for spatially aggregated tasks (DINOv3 achieving CPC = 0.511 vs. RemoteCLIP achieving CPC = 0.602, � < 0.001).

• We demonstrate that for cross-continental zero-shot transfer, geographically grounded encoders significantly outperform language-aligned features on the UK evaluation dataset, with multi-sensor physical embeddings (AlphaEarth) achieving a performance margin of 0.127 CPC over RemoteCLIP (� = 0.001, paired �-test).

The remainder of this paper is structured as follows. Section 2 reviews the historical and contemporary literature across mobility modeling, geospatial foundation models, and domain generalization boundaries. Section 3 formalizes the problem formulation, graph difusion mechanics, and the theoretical interpretation of the census noise parameter. Section 4 outlines our multi-scale dataset profiles, geospatial encoder adaptation mechanisms, and baseline implementation details. Section 5 presents our comprehensive empirical findings, representational analyses, and cross-continental scaling anomalies. Section 6 concludes with practical recommendations and future directions.

## 2 Background and Related Work

## 2.1 Deep Learning for Human Mobility

The mathematical modeling of Origin-Destination (OD) flow generation has shifted from rigid physics-inspired spatial interaction models to flexible, data-driven deep architectures. Traditional spatial interaction baselines are anchored by the Gravity model [24], which operates on an analogy to Newtonian mechanics where commuting flows are directly proportional to regional population masses and inversely proportional to a power function of geographic distance. Similarly, the Radiation model [19] eliminates parameter tuning by calculating flow systems based on job vacancy distributions and surrounding population densities. While these classical systems provide basic spatial priors, they are entirely macroscopic, with classic Gravity approaches limited to a lower performance baseline (CPC ≈ 0.32). They consistently fail to capture the highly localized, non-linear functional zoning dependencies, such as specialized commercial corridors or residential pockets, that govern modern metropolitan transit.

To bypass these strict parametric assumptions, researchers transitioned to pair-wise deep learning paradigms. Architectures like DeepGravity [18] and the Geo-contextual Multitask Embedding Learner (GMEL) [13] deploy deep feed-forward networks to map localized neighborhood land-use features, computing flow probabilities across decoupled origin and destination vector pairs. However, optimizing these independent pairs ignores the systemic, networkwide dynamics that dictate physical human transit; global OD marginal sums are conservation constraints that pair-wise models cannot enforce.

This critical systemic limitation forced the adoption of Graph Neural Networks (GNNs). Frameworks such as ODCRN [2] map urban environments as cohesive topological structures, tracking regional transit corridors and capturing complex spatial structural constraints through message-passing mechanics.

Most recently, generative graph difusion paradigms have redefined performance benchmarks. The WeDAN architecture [15] treats the generation of complete OD matrices as a parameterized reverse denoising process, leveraging continuous Gaussian noise corruption to capture both fine-grained local anomalies and macrolevel spatial structures, establishing a new state-of-the-art baseline (CPC ≈ 0.59 on the full 3,233-area LODES benchmark).

Building directly upon this difusion framework, GlODGen [16] introduced satellite-conditioned node embeddings to eliminate the downstream network’s historical reliance on highly localized, expensive, and non-generalizable tabular census registries. However, GlODGen permanently fixes RemoteCLIP as its vision backbone. Our work directly challenges this assumption, systematically evaluating whether alternative vision encoders yield stronger downstream predictive abilities and cross-continental robustness.

## 2.2 Spatial Vision Foundation Models

The rapid expansion of self-supervised computer vision has yielded highly specialized remote sensing foundation models. Rather than sharing a unified representational space, these models are optimized under fundamentally diferent pretraining paradigms, resulting in distinct representational properties. Table 1 outlines the core architectural profiles evaluated within our benchmark.

• Vision-Language Alignment: RemoteCLIP [12] utilizes a massive contrastive language-image pretraining (CLIP) objective. By aligning remote sensing imagery directly with corresponding descriptive natural language tokens, the model embeds high-level anthropogenic geography, functional zoning concepts, and socioeconomic semantics directly into its 1024-dimensional latent space.

• Self-Supervised Patch Discrimination: DINOv3 [17] opti mizes a vision transformer backbone using a teacher-student self-distillation objective over a corpus of493 million satellite images. This framework forces the network to specialize in localized patch discrimination, prioritizing high-frequency geometric features, building boundaries, and sharp surface textures over abstract semantic meanings.

• Coordinate-Contrastive Grounding: SatCLIP [10] discards language completely, deploying a coordinate-contrastive objective that forces the visual transformer to map satellite patches directly to their exact global GPS coordinates. The resulting 384-dimensional latent space is implicitly structured by continuous geographic distance, capturing spatial variations across the Earth’s surface.

• Multisensor Physical Grounding: AlphaEarth [1] shifts away from heavy visual transformers entirely, deploying a multi-datastream network natively hosted within the Google Earth Engine (GEE) cloud architecture. By assimilating six distinct sensor modalities across optical, radar, and elevation datastreams, it extracts explicit surface moisture, spectral indices, and vegetation canopy structure into a highly compact 64-dimensional feature vector.

## 2.3 The Challenge of Zero-Shot Cross-Continental Transfer

Deploying predictive mobility models across diferent domains and geographical regions remains an open challenge. Domain generalization for GNNs has been explored through distribution shift minimization [23], though these methods target node classification rather than generative flow estimation. Concurrently, the remote sensing community has established benchmarks like BigEarth-Net [21] to test the cross-domain transferability of representation layers across highly heterogeneous multi-spectral landscapes span ning multiple European nations.

Despite these advancements in visual classification and domaininvariant graph mining, the problem of direct cross-continental transfer in generative human mobility remains structurally unad dressed. Predicting commuting volumes across completely diferent continents introduces severe, compounding covariates. Local archi tectural forms, street network configurations, regional population densities, transit infrastructure designs, and socio-environmental zoning behaviors shift radically between North American developments and European historic grids.

When a model is trained exclusively on domestic distributions, its conditioning layers easily overfit to regional features. To our knowledge, no prior work evaluates satellite-conditioned OD generation under zero-shot cross-continental conditions: our two-stage transfer evaluation (US → UK and US → Global) serves as the first systematic benchmark to uncover how these shifting visual and structural topologies impact generative graph difusion networks.

## 2.4 Evaluation Protocols for OD Generation

Validating the systemic accuracy of a generated graph adjacency matrix requires mathematical evaluation protocols that can isolate relative spatial routing logic from absolute numerical scaling behavior. While RMSE and MAE are widely used in trafic forecasting and vehicle routing, these metrics operate purely on absolute arithmetic diferences. In zero-shot cross-continental settings, absolute magnitude estimation is highly fragile: subtle distribution shifts in a foundation model’s latent conditioning vector can cause a downstream graph transformer to output runaway volume scales, completely disrupting absolute error metrics even if the directional routing logic remains accurate.

To establish a scale-invariant validation framework, modern origin-destination literature prioritizes the Common Part of Com muters (CPC) metric [11]. CPC calculates the normalized overlap between the full ground-truth and generated OD distributions as a global coeficient over all origin-destination pairs. By explicitly penalizing misallocated spatial connections while remaining substantially more robust to absolute magnitude scaling than regressionbased metrics, CPC provides the stable, topology-focused signal required to evaluate true cross-continental alignment reliability.

## 3 Problem Formulation & Architecture

## 3.1 OD Generation as Conditional Graph Difusion

We model a city as a directed graph $G = ( V , E )$ , where nodes $v \in V$ represent geographic regions, such as census tracts or administrative zones, and edges $e _ { i j } \in E$ represent the commuting flow $F _ { i j }$ from origin � to destination �. The objective of the origin-destination (OD) generation task is to construct the complete flow adjacency matrix $\mathbf { F } \in \mathbb { R } ^ { | V | \times | V | }$ <sup>|</sup> . This matrix is generated by an architecture trained as a Denoising Difusion Probabilistic Model (DDPM) [7] and sampled at inference via a Denoising Difusion Implicit Model (DDIM) [20].

The forward difusion process systematically adds Gaussian noise to the true flow matrix $\mathbf { F } _ { 0 }$ over � discrete timesteps according to a predefined variance schedule $\beta _ { 1 } , \beta _ { 2 } , \ldots , \beta _ { T }$

$$
q ( \mathbf { F } _ { t } \mid \mathbf { F } _ { t - 1 } ) = N ( \mathbf { F } _ { t } ; { \sqrt { 1 - \beta _ { t } } } \mathbf { F } _ { t - 1 } , \beta _ { t } \mathbf { I } )\tag{1}
$$

Using the notation $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } } \end{array}$ , we can express the marginalized distribution at any arbitrary timestep � directly as:

$$
q ( \mathbf { F } _ { t } \mid \mathbf { F } _ { 0 } ) = N ( \mathbf { F } _ { t } ; { \sqrt { { \bar { \alpha } } _ { t } } } \mathbf { F } _ { 0 } , ( 1 - { \bar { \alpha } } _ { t } ) \mathbf { I } )\tag{2}
$$

Following the WeDAN architecture [15], the reverse denoising process employs a GraphTransformer network [4] to iteratively remove noise from the continuous flow matrix over a series ofdiscrete timesteps. Crucially, this generation process is conditioned on nodelevel feature vectors, ensuring that the unique socio-environmental and structural characteristics of each geographic region directly guide the predicted downstream mobility flows.

Table 1: Architectural and Pretraining Comparison of Evaluated Spatial Vision Foundation Models
<table><tr><td>Model</td><td>Architecture</td><td>Dim</td><td>Pretraining</td><td>Supervision</td></tr><tr><td>RemoteCLIP</td><td>ViT-L/14</td><td>1024</td><td>RS image-text pairs</td><td>Contrastive (VLM)</td></tr><tr><td>DINOv3</td><td>ViT-L/16</td><td>1024</td><td>493M satellite images</td><td>Self-distillation</td></tr><tr><td>SatCLIP</td><td>ViT-S/16</td><td>384</td><td>GPS-matched patches</td><td>Coordinate-contrastive</td></tr><tr><td>AlphaEarth</td><td>Multi-sensor GEE fusion</td><td>64</td><td>6-sensor fusion</td><td>Self-supervised (GEE)</td></tr></table>

## 3.2 Satellite-Conditioned Node Features

In traditional mobility frameworks, the node feature vector relies entirely on localized, tabular census statistics. In our satelliteconditioned paradigm, the base feature vector for a given region � is defined as:

$$
\mathbf { x } _ { r } = [ \varphi ( I _ { r } ) \parallel \log ( 1 + \mathbf { p } _ { r } ) ] \in \mathbb { R } ^ { d + 2 }\tag{3}
$$

where $\varphi ( I _ { r } ) \in \mathbb { R } ^ { d }$ represents the embedding vector produced by a spatial vision foundation model applied to the satellite imagery of the region, and $\mathbf { p } _ { r } \in \mathbb { R } ^ { 2 }$ represents globally available WorldPop statistics covering total population and land area. To ensure highly uniform spatial sampling across irregular administrative boundaries, each geographic region is partitioned using H3 resolution 9 hexagonal cells. The selected vision encoder processes each hexagonal tile independently, and the resulting individual embeddings are mean-pooled across all tiles contained within the region boundary to generate the final visual representation $\varphi ( I _ { r } )$ .

During model training, the final conditioning vector fuses the satellite representation ${ \bf x } _ { r }$ with a noise-controlled census vector $\tilde { \mathbf { a } } _ { r } { : }$

$$
\tilde { \mathbf { a } } _ { r } = \mathbf { a } _ { r } \cdot \boldsymbol \eta + \pmb \varepsilon \cdot \left( 1 - \eta \right)\tag{4}
$$

where ${ \mathbf { a } } _ { r } \in \mathbb { R } ^ { 9 7 }$ is the true localized census data vector and $\varepsilon \sim$ ${ \cal N } ( 0 , { \bf { I } } )$ represents standard Gaussian noise.

The full node vector fed to the GraphTransformer is then constructed as:

$$
\mathbf { n } _ { r } = \left[ \mathbf { x } _ { r } \parallel \tilde { \mathbf { a } } _ { r } \parallel \hat { \mathbf { a } } _ { r } \right] \in \mathbb { R } ^ { ( d + 2 ) + 9 7 + 9 7 }\tag{5}
$$

where $\hat { \mathbf { a } } _ { r }$ is a DenoisingTransformer prediction of census from satellite features. In all experiments, setting the configuration parameter if-ImgAttrAug=0 zeros the predicted vector $\hat { \mathbf { a } } _ { r }$ , meaning the efective input simplifies to:

$$
\begin{array} { r } { \mathbf { n } _ { r } = \big [ \mathbf { x } _ { r } \big \| \tilde { \mathbf { a } } _ { r } \big \| \mathbf { 0 } \big ] . } \end{array}\tag{6}
$$

This structural formulation explains why the downstream network input dimension scales directly with the choice of spatial foundation model, varying based on the underlying dimensionality � of the vision encoder.

## 3.3 GraphTransformer Denoising Mechanics and Training Loss

The GraphTransformer network updates the node states and processes edge relationships within the flow graph. Given the concatenated matrix of node feature vectors $\mathbf { H } ^ { ( 0 ) } = [ \mathbf { n } _ { 1 } , \mathbf { n } _ { 2 } , \ldots , \mathbf { n } _ { | V | } ] ^ { T }$ each transformer layer � applies a multi-head attention mechanism.

For a specific attention head $k ,$ the query, key, and value matrices are projected using learned parameter weights:

$$
{ \bf Q } _ { k } = { \bf H } ^ { ( l ) } { \bf W } _ { k } ^ { Q } , \quad { \bf K } _ { k } = { \bf H } ^ { ( l ) } { \bf W } _ { k } ^ { K } , \quad { \bf V } _ { k } = { \bf H } ^ { ( l ) } { \bf W } _ { k } ^ { V }\tag{7}
$$

The spatial interaction between node � and node $j$ is computed by taking the scaled dot product of their respective projections, incorporating a temporal step embedding t to track the difusion timeline:

$$
\mathbf { A } _ { k } ( i , j ) = \frac { ( \mathbf { Q } _ { k } ) _ { i } ( \mathbf { K } _ { k } ) _ { j } ^ { T } } { \sqrt { d _ { h e a d } } } + \psi _ { k } ( [ \mathbf { F } _ { t } ] _ { i j } , \mathbf { t } )\tag{8}
$$

where $\psi _ { k }$ represents an edge projection network that transforms the noisy flow state $\left[ \mathbf { F } _ { t } \right] _ { i j }$ into a structural bias modifier for the attention matrix. The updated node hidden representations are then compiled across all � heads using a linear projection layer and a feed-forward network:

$$
\mathbf { H } ^ { ( l + 1 ) } = \mathrm { F F N } \left( \left[ \mathrm { S o f t m a x } ( \mathbf { A } _ { 1 } ) \mathbf { V } _ { 1 } \parallel \mathbf { \dots } \parallel \mathrm { S o f t m a x } ( \mathbf { A } _ { K } ) \mathbf { V } _ { K } \right] \mathbf { W } ^ { O } \right)\tag{9}
$$

The network parameter optimization relies on a simplified mean squared error objective function. The loss function forces the Graph-Transformer to isolate and predict the precise Gaussian noise vector � injected into the flow matrix at any given difusion step �:

$$
\mathcal { L } _ { \mathrm { s i m p l e } } ( \theta ) = \mathbb { E } _ { t \sim \mathcal { U } \left\{ 1 , T \right\} , \mathbf { F } _ { 0 } \sim q , \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) } \left[ \left\| \epsilon - \epsilon _ { \theta } ( \mathrm { F } _ { t } , \mathbf { H } ^ { ( 0 ) } , \mathbf { t } ) \right\| _ { F } ^ { 2 } \right]\tag{10}
$$

By minimizing this objective across all discrete timesteps, � learns to reverse the forward corruption process. At inference, the trained GraphTransformer is deployed within DDIM sampling [20], which generates structured flow matrices via a deterministic 25-step trajectory, yielding a 10× reduction from the �=250 training steps. To reduce stochastic variance, the final output for each city is averaged over 50 independent DDIM runs. The node-level conditioning $\Breve { \mathbf { H } ^ { ( 0 ) } }$ , fixed at initialization, ensures that all generated flows remain anchored to the satellite and demographic representations of each geographic zone.

## 3.4 Semantics of the Census Noise Parameter

Our architectural analysis indicates that the census noise parameter (�) is best understood as governing the assumed availability of true census attributes, rather than acting as a noise-injection regularizer on satellite features. During training, � is resampled uniformly at random for each city in each batch, so the model observes the full range of census availability, from clean attributes through to a purely noisy channel. It is never exposed to an all-zero census vector, however: that would require $\mathbf { a } _ { r } = \mathbf { 0 }$ and $\eta = 1$ to coincide, which does not occur under continuous sampling of �. This insight reveals two distinct operational paradigms depending on the evaluation context:

![](images/be8fe0bf3ee3eb71a0b099f50ee6ed1872979be51425185f0a061fc18f1f2e8d.jpg)  
Figure 1: GlODGen pipeline. Satellite imagery and population data are preprocessed via H3 hexagonal tiling (resolution 9, ∼0.1 km<sup>2</sup> per cell) with per-tile mean pooling. A spatial vision encoder extracts zone-level embeddings conditioning the WeDAN GraphTransformer difusion model, which generates the OD flow matrix via DDIM sampling.

• In-Distribution Evaluation: When true localized census data a<sub>�</sub> is fully available, setting � = 1 provides the model with clean, multi-modal conditioning. Setting � = 0 replaces the census data entirely with Gaussian noise, forcing the architecture to rely solely on the satellite representation x<sub>�</sub> to generate flows.

• Zero-Shot Cross-Continental Transfer: In target regions where local census datasets are missing or structurally incompatible, true census data is unavailable; supplying a<sub>�</sub> = 0 as a surrogate alongside a setting of � = 1 forces an allzero vector into the network, yielding a˜<sub>�</sub> = 0. Because � is randomized throughout training, the model has repeatedly encountered a noise-dominated census channel, but never an all-zero one. That substitution is therefore genuinely out of distribution, whereas evaluating at � = 0 is not: it reproduces a conditioning regime the model was trained under. This asymmetry, rather than a generic distribution shift, is what separates the two substitutions.

Accordingly, we report all US in-distribution results at � = 1 and all UK and Global zero-shot results at � = 0, and we treat these as the declared operating points for every headline comparison. The encoders do not, however, respond uniformly to � on the UK data (Figure 3): AlphaEarth degrades sharply as � → 1 (0.515 to 0.249) while DINOv3 remains essentially flat. We do not claim a single mechanism for that spread.

## 4 Experimental Setup

## 4.1 Datasets and Evaluation Metrics

Our empirical evaluation spans three distinct geographic scales to rigorously stress-test both in-distribution performance profiles and zero-shot cross-continental generalization capabilities:

• US In-Distribution: This dataset serves as the foundational training and localized testing corpus, consisting of 1,925 counties retained from the full LODES pool following GlOD-Gen’s data filtering to include only areas with a complete OD matrix and satellite feature data. The ground-truth commuting configurations are sourced from the Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics (LODES) 2018 Census Bureau repository. For every evaluation seed, this dataset is partitioned into an 80/10/10 random split dedicated to model training, internal hyperparameter validation, and in-distribution testing, respectively.

• UK Zero-Shot Transfer: To evaluate strict zero-shot crosscontinental generalizability, we introduce an evaluation target incorporating 325 Local Authority Districts within the United Kingdom. Ground-truth cross-zonal commuting flows are taken from the UK benchmark released with GlOD-Gen [16], which derives population-level flows between census units from Ofice for National Statistics (ONS) registries. Under this cross-domain evaluation paradigm, the downstream generative model is trained exclusively on North

American US urban areas, completely omitting any targetdomain fine-tuning or structural adaptation before inference.

• Global Zero-Shot Evaluation: To assess unconstrained scalability across highly heterogeneous urban forms, this evaluation maps 14 morphologically diverse international metropolitan areas outside the US and UK core regions: Baoding, Beijing, Chengdu, Guangzhou, London, Namibia, Paris, Rio de Janeiro, Senegal, Shanghai, Shenzhen, Sydney, Tang shan, and Tokyo. The same US-trained architecture is frozen and applied directly to these international regions without weight optimization, representing the more distant of two zero-shot targets: US → Global. Ground-truth flow baselines for these global metropolises are aggregated from localized household travel surveys and anonymous mobile network signaling registries [16].

These three targets difer in reference year, zoning system, and collection method, and the satellite composites are not date-matched to any of them; cross-region diferences therefore reflect diferences in reference data as well as in urban morphology.

The primary metric utilized to quantify spatial allocation accuracy across all experimental boundaries is the Common Part of Commuters (CPC). Let $\mathbf { F } \in \mathbb { R } ^ { | V | \times | V | }$ represent the ground-truth origin-destination flow matrix, and let $\hat { \hat { \mathbf { F } } } \in \mathbb { R } ^ { | V | \times | V | }$ represent the corresponding matrix generated by the difusion network. The CPC score is mathematically formalized as:

$$
\mathrm { C P C } ( \mathbf { F } , \hat { \mathbf { F } } ) = \frac { 2 \sum _ { i = 1 } ^ { | V | } \sum _ { j = 1 } ^ { | V | } \operatorname* { m i n } ( F _ { i j } , \hat { F } _ { i j } ) } { \sum _ { i = 1 } ^ { | V | } \sum _ { j = 1 } ^ { | V | } F _ { i j } + \sum _ { i = 1 } ^ { | V | } \sum _ { j = 1 } ^ { | V | } \hat { F } _ { i j } }\tag{11}
$$

The CPC metric calculates the normalized overlap coeficient between the true and predicted mobility distributions, where CPC ∈ [0, 1]. A score of1 indicates perfect structural reconstruction, whereas a score of 0 denotes complete spatial disconnection. In addition to CPC, standard error scales including Root Mean Squared Error (RMSE), normalized Root Mean Squared Error (NRMSE), and Mean Absolute Error (MAE) are tracked to assess numeric scaling behavior.

## 4.2 Encoder Adaptation and Feature Extraction

High-resolution satellite imagery tiles are retrieved from the Esri World Imagery repository at zoom level 15 using the automated cloud pipelines of Google Earth Engine. To ensure highly uniform visual sampling across irregular administrative and political bound aries, each target geographic region is partitioned using an Uber H3 spatial index at resolution 9, generating uniform hexagonal cells with an average area of approximately 0.1 square kilometers.

Each spatial foundation model processes the individual hexagonal tiles falling within a region independently. The resulting collection of high-dimensional latent tile embeddings is subsequently mean-pooled across the region administrative boundary to construct a single, unified visual representation vector $\varphi ( I _ { r } )$ . The four underlying vision encoder adapt their internal feature extraction layers as follows:

• RemoteCLIP: We deploy the ViT-L/14 visual backbone to extract dense 1024-dimensional visual embeddings. This representation benefits directly from pre-aligned language supervision, capturing high-level anthropogenic land-use patterns.

• DINOv3: We use the satellite-pretrained dinov3-vitl16 -pretrain-sat493m checkpoint (ViT-L/16) to obtain 1024- dimensional embeddings. These features are trained via selfsupervised teacher-student distillation across a corpus of 493 million unannotated satellite images, rendering the latent features highly sensitive to micro-level surface geometry and texture.

• SatCLIP: This paradigm utilizes a ViT-S/16 backbone to generate 384-dimensional coordinate-aligned feature maps. The SatCLIP checkpoint’s patch embedding projection is pretrained on 13-band Sentinel-2 imagery [3], producing weights of shape [384, 13, 16, 16]. To adapt to 3-channel RGB tiles, we average these weights across the 13 spectral bands to obtain a single-channel kernel [384, 1, 16, 16], then tile it three times to form a [384, 3, 16, 16] projection. This preserves the spatial convolution structure while distributing spectral information uniformly across RGB channels; all subsequent transformer weights remain unchanged.

• AlphaEarth: This model entirely bypasses local GPU processing constraints by providing precomputed 64-dimensional multi-sensor pixel embeddings processed natively within Google Earth Engine, serving as an eficient baseline for resource-constrained large-scale deployments.

## 4.3 Implementation Details

To isolate the efect of the visual encoder, all competing foundation model configurations are mapped to an identical downstream WeDAN generative framework. The underlying GraphTransformer neural engine is configured with 4 multi-head attention layers, incorporating a fixed hidden layer dimensionality of 32. The forward corruption pipeline applies a parametric cosine noise schedule to step-by-step inject variance across $T = 2 5 0$ discrete difusion timesteps. The complete structural hyperparameter configuration utilized across all model variations is detailed in Table 2.

Table 2: Downstream GraphTransformer and Difusion Network Hyperparameter Specifications
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>GraphTransformer Layers</td><td>4</td></tr><tr><td>Attention Head Count (K)</td><td>4</td></tr><tr><td>Hidden Layer Dimension</td><td>32</td></tr><tr><td>Total Diffusion Training Steps (T)</td><td>250</td></tr><tr><td>Noise Schedule</td><td>Cosine Schedule</td></tr><tr><td>Primary Learning Rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Optimization Algorithm</td><td>AdamW</td></tr><tr><td>Batch Size</td><td>1</td></tr><tr><td>Dropout Regularization Rate</td><td>0</td></tr><tr><td>DDIM Sampling Steps</td><td>25</td></tr><tr><td>Monte Carlo Samples per City</td><td>50</td></tr></table>

During the inference phase, the iterative flow generation process is completed via deterministic DDIM sampling, traversing a compressed 25-step trajectory. To minimize stochastic variance and prevent single-run anomalies from distorting performance tracking, the final generated flow adjacency matrix for each independent city is calculated as the mean over 50 independent DDIM sampling runs. All model iterations are trained, validated, and evaluated across 5 random seeds {2024, 2025, 2026, 2027, 2028}. Because all encoders are evaluated on an identical set of seeds, splits, and target regions, statistical diferences are validated using two-tailed paired �-tests over the five matched seeds, with Holm–Bonferroni correction across the three reported comparisons.

## 5 Results and Analysis

Table 3 consolidates performance across all four encoders, averaged over five random seeds and all census-noise levels (� ∈ {0, 0.25, 0.5, 0.75, 1}). Tables 6–8 provide a granular breakdown of spatial allocation alignment and absolute error metrics across five discrete census availability tiers $( \eta \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \} )$ ).

## 5.1 US In-Distribution Performance

As shown in Table 3, RemoteCLIP achieves the highest overall mean CPC (0.598 ± 0.008). When isolating the primary clean evaluation point $( \eta = 1 ;$ , full census available), this performance reaches 0.602± 0.009, recovering 94.9% of the full-census WeDAN baseline (0.634) reported by GlODGen [16]. AlphaEarth and SatCLIP trail closely within the macro averages, while DINOv3 lags significantly with an overall mean CPC of 0.509 ± 0.009 (� < 0.001, paired �-test vs. RemoteCLIP).

This gap is particularly notable given DINOv3’s 493-millionimage pretraining corpus: raw scale does not compensate for misaligned objectives. The self-distillation objective specializes in microscale surface geometry and texture discrimination, whereas commuting OD generation requires semantic features of human mobil ity patterns, the precise signal that language-aligned contrastive supervision encodes. A plausible explanation, which we do not test directly, is an aggregation mismatch between DINOv3’s pretraining objective and the zone-level prediction task. DINOv3’s teacherstudent self-distillation is optimized to discriminate between indi vidual image patches at fine spatial scales. Each US county contains 50–200 H3 resolution-9 tiles, which are independently encoded and mean-pooled into a single zone embedding. Mean-pooling of patch-discriminative features destroys the very discriminability the encoder was trained to produce: if each tile captures locally distinctive surface geometry (a parking lot, a rooftop, a tree canopy), but the encoder has not learned to associate these patterns with zone-level commuting character, the pooled embedding carries near-zero relevant signal. RemoteCLIP’s contrastive language supervision, by contrast, produces scene-level embeddings aligned to high-level semantic categories (e.g., "dense urban commercial area", "low-density suburban residential") that remain meaningfully discriminative after mean-pooling because they already operate at the semantic granularity required for zone-level conditioning.

All four encoders generally exhibit a consistent increase in CPC as � increases from 0 to 1 (Figure 3), confirming that census features provide additive conditioning value when their distribution remains in-domain.

## 5.2 UK Zero-Shot Transfer

The ranking inverts sharply under zero-shot cross-continental transfer. At the declared UK operating point $( \eta ~ = ~ 0 ;$ , local census entirely unavailable), AlphaEarth reaches $0 . 5 1 5 \pm 0 . 0 1 5$ and SatCLIP $0 . 5 1 3 \pm 0 . 0 3 4 ,$ a diference that is not statistically significant $(  p = 0 . 9 1$ , paired �-test). RemoteCLIP attains only 0.388 and DINOv3 0.330. Averaged across all � levels (Table 3), SatCLIP and AlphaEarth exchange places $( 0 . 4 9 5 \pm 0 . 0 3 2$ vs. $0 . 4 5 5 \pm 0 . 1 1 8 )$ , reflecting AlphaEarth’s sharper degradation as $\eta  1$ ; RemoteCLIP and DINOv3 remain far behind at $0 . 3 4 9 \pm 0 . 0 5 8$ and $0 . 3 2 9 \pm 0 . 0 6 0$ The clear performance advantage of AlphaEarth and SatCLIP over RemoteCLIP constitutes a central empirical finding of this work: geographically grounded encoders generalize better across continents than language-alignedfeatures.

One explanation consistent with these results is distributional overfitting. RemoteCLIP’s language supervision encodes human mobility pattern semantics specific to North American urban morphology: low-density suburban sprawl, freeway-anchored commercial strips, which do not translate as efectively to European landuse configurations. Conversely, AlphaEarth’s multi-sensor physical embeddings (spectral indices, radar backscatter, elevation) and Sat-CLIP’s coordinate-contrastive pretraining may encode geophysical and geospatial priors that remain continuous across continental boundaries.

Within each independent seed, DINOv3’s performance is conspicuously insensitive to variation in � on UK data (Figure 3), despite a high cross-seed variance that results in an aggregate score of $0 . 3 2 9 \pm 0 . 0 6 0$ . This behavioral signature reveals that DINOv3’s geometric features carry minimal mobility signal when deployed out-of-distribution: the model’s pathway through the census conditioning channel contributes near-zero marginal information, making the network unresponsive to changes in demographic inputs.

## 5.3 Global Zero-Shot Generalization

Under the most distant transfer setting (US → Global), all encoders experience significant performance degradation. Remote-CLIP achieves the highest Global CPC $( 0 . 1 2 2 \pm 0 . 0 1 4 )$ , with AlphaEarth $( 0 . 1 1 1 \pm 0 . 0 1 7 )$ and SatCLIP (0.104 ± 0.024) following in close proximity. DINOv3 yields only $0 . 0 2 2 \pm 0 . 0 0 6$ , confirming that texture-specialized self-distillation features provide little viable cross-continental transfer signal. For reference, classical Gravity models report $\mathrm { C P C } \approx 0 . 3 2$ on comparable US commuting data [24]. Every satellite-conditioned configuration evaluated here falls far below that on the global benchmark, placing global performance beneath even a classical spatial-interaction prior.

The near-convergence of RemoteCLIP, AlphaEarth, and SatCLIP on the Global benchmark (ΔCPC < 0.025) suggests that at this extreme transfer distance, encoder choice becomes secondary to the fundamental distribution shift between training domains and morphologically heterogeneous global metropolises. Reported NRMSE values for Global are highly unstable across seeds (e.g., RemoteCLIP: 1927) due to normalization against city-level mean flows, which vary by orders of magnitude across the 14 target cities; CPC should be treated as the sole reliable metric on this benchmark.

Four compounding factors explain this systematic global collapse. (i) Morphological diversity: the 14 evaluation cities span radically distinct urban forms: Tokyo’s transit-oriented polycentric structure, Rio de Janeiro’s favela-formal city duality, and peri-urban sub-Saharan settlement patterns share no common visual or demographic signature with US county-level commuting distributions.

![](images/108a1f61ce8b82d9e28adc305efd95e8a6560422a2a8378bde42e7446de8241d.jpg)

Table 3: Mean (±std) performance across 5 seeds and all � levels. CPC (↑) represents the primary metric for spatial allocation alignment; RMSE, NRMSE, and MAE are lower-is-better (↓). <sup>†</sup>SatCLIP Global metrics are evaluated over �=4 seeds. <sup>‡</sup>Global NRMSE exhibits high scale-sensitivity; CPC serves as the primary metric.
<table><tr><td></td><td colspan="4">US (In-Distribution)</td><td colspan="4">UK (Zero-Shot Transfer)</td><td colspan="4">Global (Zero-Shot Transfer)</td></tr><tr><td>Model Paradigm</td><td>CPC</td><td>RMSE</td><td>NRMSE</td><td>MAE</td><td>CPC</td><td>RMSE</td><td>NRMSE</td><td>MAE</td><td>CPC</td><td>RMSE</td><td>NRMSE</td><td>MAE</td></tr><tr><td>RemoteCLIP</td><td> $\mathbf { 0 . 5 9 8 \pm 0 . 0 0 8 }$ </td><td>68.1</td><td>0.955</td><td>39.7</td><td> $0 . 3 4 9 \pm 0 . 0 5 8$ </td><td>104.5</td><td>2.08</td><td>71.5</td><td> $\mathbf { 0 . 1 2 2 \pm 0 . 0 1 4 }$ </td><td>632</td><td> $1 9 2 7 ^ { \ddagger }$ </td><td>205.8</td></tr><tr><td>AlphaEarth</td><td> $0 . 5 6 8 \pm 0 . 0 1 9$ </td><td>71.1</td><td>0.967</td><td>41.8</td><td> $0 . 4 5 5 \pm 0 . 1 1 8$ </td><td>79.7</td><td>1.55</td><td>54.1</td><td> $0 . 1 1 1 \pm 0 . 0 1 7$ </td><td>613</td><td>488</td><td>201.3</td></tr><tr><td>DINOv3</td><td> $0 . 5 0 9 \pm 0 . 0 0 9$ </td><td>77.4</td><td>1.037</td><td>43.7</td><td> $0 . 3 2 9 \pm 0 . 0 6 0$ </td><td>342.1</td><td>8.26</td><td>199.9</td><td> $0 . 0 2 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td>1083</td><td>10054</td><td>580.7</td></tr><tr><td> $\mathrm { S a t C L I P ^ { \dagger } }$ </td><td> $0 . 5 5 5 \pm 0 . 0 2 7$ </td><td>72.8</td><td>0.992</td><td>43.3</td><td> $\mathbf { 0 . 4 9 5 \pm 0 . 0 3 2 }$ </td><td>83.4</td><td>1.60</td><td>59.5</td><td> $0 . 1 0 4 \pm 0 . 0 2 4$ </td><td>666</td><td> $5 0 4 5 ^ { \ddagger }$ </td><td>214.0</td></tr></table>

![](images/0eef39a6c0a1b27ee86d02450e2cb21d50879941147df7abf650d382edfb9a9c.jpg)

![](images/ea02f80ec698f9f93c38e65022d907cbf43ce358a38e3b439ee777cf446708d1.jpg)

![](images/3fe14697b39b16d2869fb7dde1ea97aad81c9c17e60dc2fb09925139d7393365.jpg)  
Figure 2: Mean CPC, RMSE, and MAE per region across all � levels (mean ± std, 5 seeds), consistent with Table 3. RemoteCLIP leads in-distribution (US); SatCLIP and AlphaEarth dominate zero-shot UK transfer; all encoders converge on Global cities. DINOv3 collapses to near-zero performance globally (CPC = 0.022).  
Figure 3: CPC as a function of census noise parameter across all four encoders and three evaluation benchmarks (mean over 5 seeds). The dashed vertical line marks the primary evaluation point for each dataset $\scriptstyle ( \eta = 1$ for $\mathbf { U S } ; \boldsymbol { \eta } { = } 0$ for UK and Global). US performance increases monotonically with census availability, whereas UK and Global performance degrades as � → 1 due to the introduction of out-of-distribution zero-valued census vectors.

(ii) Scale mismatch: the training distribution contains cities with approximately 30-200 zones; London alone contains 932 zones, roughly five times the largest training city, placing its graph structure firmly out of distribution for the GraphTransformer’s attention layers. (iii) OD data heterogeneity: ground-truth flows for global cities are aggregated from heterogeneous sources, household travel surveys, call detail records, and mobile signaling registries, each introducing distinct noise floors and systematic collection biases that the CPC metric conflates with model error. (iv) Zero-census conditioning: all global evaluation operates at $\scriptstyle \eta = 0 ,$ already the weaker conditioning regime; combined with the domain shift in satellite appearance, neither modality provides suficient signal to anchor the difusion process to local mobility patterns. Taken together, these factors indicate that the global performance gap is not an encoder-selection problem but a fundamental distributional mismatch requiring either local fine-tuning data or geographically diverse training corpora.

## 5.4 Census Noise Sensitivity and Operational Implications

The �-sensitivity patterns across datasets empirically validate the architectural interpretation developed in Section 3. For US data, all encoders benefit from increasing � (more census signal). For UK and Global data, the opposite holds: RemoteCLIP and AlphaEarth CPC values degrade as $\eta  1$ because supplying zero-valued census vectors at $\eta = 1$ introduces an out-of-distribution input that the network was not exposed to during training (Figure 3). Practitioners deploying these models in census-absent regions must evaluate strictly at $\eta \ : = \ : 0$ to avoid silent performance degradation. This finding has direct operational consequences: tools and frameworks built on WeDAN should expose � as a deployment-time parameter rather than fixing it at training time, allowing practitioners to select the appropriate regime based on local census availability. This heterogeneity carries a direct deployment implication. Although AlphaEarth attains the best UK transfer at $\eta \ : = \ : 0 ,$ it is also the most sensitive to the census channel: its UK CPC falls from 0.515 at $\eta = 0 \ \mathrm { t o } \ 0 . 2 4 9$ at $\eta = 1$ , whereas SatCLIP degrades only from 0.512 to 0.455 over the same range (Table 7). A practitioner holding partial or unreliable census data should therefore prefer SatCLIP, whose accuracy is statistically indistinguishable from AlphaEarth’s at $\eta = 0$ but far more stable as census information is introduced; AlphaEarth is the better choice only when the census channel is guaranteed to be absent.

## 5.5 Practical Encoder Selection

Table 4 summarizes the computational profile of each encoder. AlphaEarth is the only configuration requiring no local GPU: all feature extraction runs natively within Google Earth Engine (GEE), making it the default choice for resource-constrained or cloudfirst deployments. SatCLIP’s ViT-S/16 backbone incurs roughly one-third the VRAM of the ViT-L models, enabling extraction on consumer-grade hardware. RemoteCLIP and DINOv3 share iden tical backbone size (ViT-L) and extraction cost; the performance gap between them is attributable entirely to pretraining objective rather than model capacity.

Table 4: Computational profiles of evaluated encoders. GEE = Google Earth Engine cloud extraction; H200 = NVIDIA H200 GPU.
<table><tr><td>Encoder</td><td>Backbone</td><td>Dim</td><td>Hardware</td><td>Params</td></tr><tr><td>RemoteCLIP</td><td>ViT-L/14</td><td>1024</td><td>H200 GPU</td><td>307M</td></tr><tr><td>DINOv3</td><td>ViT-L/16</td><td>1024</td><td>H200 GPU</td><td>307M</td></tr><tr><td>SatCLIP</td><td>ViT-S/16</td><td>384</td><td>H200 GPU</td><td>22M</td></tr><tr><td>AlphaEarth</td><td>GEE fusion</td><td>64</td><td>GEE (CPU)</td><td>一</td></tr></table>

Table 5: Encoder selection guide by deployment scenario.
<table><tr><td>Scenario</td><td>Encoder</td><td>Rationale</td></tr><tr><td>In-dist., census available</td><td>RemoteCLIP</td><td>Highest US CPC (0.602)</td></tr><tr><td rowspan="2">Zero-shot, Europe</td><td>AlphaEarth/</td><td>Tied at η=0 (0.515 vs.</td></tr><tr><td>SatCLIP</td><td>0.513, n.s.)</td></tr><tr><td rowspan="2">Zero-shot, no GPU Minimal embedding dim</td><td>AlphaEarth</td><td>GEE cloud extraction</td></tr><tr><td>AlphaEarth</td><td>64-dim; low memory</td></tr><tr><td>Diverse global cities</td><td>Any except DINOv3*</td><td>Gap &lt;0.025 CPC</td></tr></table>

<sup>∗</sup>These three converge near CPC ≈ 0.11 globally; DINOv3 reaches only 0.022.

Table 5 distills the empirical findings into deployment recommendations. No single encoder dominates across all settings; the optimal choice depends on deployment region, census data availability, and hardware constraints.

## 6 Limitations and Future Work

## 6.1 Limitations

Evaluation Limitations. Several structural constraints bound the scope of the present evaluation. First, all evaluated encoders operate on static annual satellite composites. Commuting flows, however, are inherently dynamic: rush-hour congestion patterns, seasonal employment shifts, and post-pandemic transit adjustments introduce temporal variance that no single static visual composite can capture.

Second, our 80/10/10 train/valid/test split reduces training data by approximately 190 cities relative to the GlODGen baseline (90/10), introducing a systematic 2–3% CPC deficit. Beyond this methodological gap, deploying models trained on North American urban morphology to UK LADs and global cities introduces domain shifts that zero-shot visual embeddings only partially bridge, evidenced by a 37% CPC drop from the US to the UK (e.g., 0.602 → 0.388 for RemoteCLIP).

Third, while SatCLIP demonstrated robust cross-continental resilience via coordinate-contrastive alignment, its architectural reliance on standard optical patches limits its ability to fully exploit the rich, multi-spectral geophysical signatures $( \mathrm { e . g . } ,$ radar backscatter, elevation) natively captured by AlphaEarth’s multi-sensor fusion.

Finally, the DINOv3 evaluation demonstrates that raw data scale does not guarantee task alignment. Despite a large-scale pretraining corpus, the model’s empirical performance remained constrained by its self-distillation objective, which over-indexes on patch-level texture discrimination rather than the zone-level functional land use required for OD flow generation.

OD Metric Limitations. A further limitation concerns evaluation granularity. Our primary metric, CPC, aggregates flow accuracy over all origin-destination pairs equally, masking systematic directional biases—for instance, consistent over-prediction of shortdistance intra-zone flows or under-prediction of long-distance commutes. Decomposing CPC by distance decay bin, flow magnitude quantile, or intra- versus inter-zone flows would provide a more diagnostically useful performance profile and expose encoder-specific failure modes that aggregate metrics obscure. Similarly, our Global results are reported as means over 14 cities, which masks substantial per-city variation; a per-city breakdown would be required to separate morphological efects from reference-data quality.

Backbone Architectural Constraints and Alternatives. A final limitation concerns the generative backbone itself. All experiments fix WeDAN as the graph difusion architecture, varying only the visual encoder. WeDAN’s GraphTransformer operates with a batch size of 1 (one city per gradient step), constraining cross-city structural learning. Its full-attention mechanism scales quadrati cally with node count, restricting deployment to cities with fewer than approximately 1,000 zones. Alternative generative backbones warrant investigation: score-based graph difusion via stochastic diferential equations [9] and variational flow matching formulations [5] ofer potentially superior scalability and generalization properties. Disentangling encoder quality from backbone choice remains an open question for future work.

Reproducibility Constraints. AlphaEarth embeddings depend on Google Earth Engine asset availability and API versioning, which may change over time. The SatCLIP 13-to-3 channel band weight averaging is a non-standard adaptation; exact reproduction requires the released extraction code. All runs depend on the frozen DenoisingTransformer checkpoint sourced from GlODGen [16].

## 6.2 Future Work

Several directions emerge directly from our findings.

Trafic-encoded node features. Static satellite composites cannot observe dynamic mobility signals such as peak-hour congestion, transit ridership, or road capacity utilization. Incorporating zonelevel trafic indicators, from loop detector networks (TMAS), GTFS transit feeds, or OSM road structural features, as additional node signals would provide employment-density proxies orthogonal to satellite appearance, potentially closing the gap between the �=0 and �=1 operating points.

Multi-temporal satellite features. Extending encoders to multitemporal composites (quarterly Sentinel-2 stacks, VIIRS nighttime light time series, or Sentinel-1 SAR coherence change) would capture seasonal land-use dynamics invisible to annual composites. Pretrained temporal encoders such as Prithvi-TS could be integrated with minimal architectural modification.

Architectural scaling and fine-tuning. Architecturally, WeDAN relies on a standard full-attention GraphTransformer, which scales quadratically with the number of nodes. Upgrading the backbone with sparse or flash attention mechanisms would enable the model to process significantly larger metropolitan graphs without memory bottlenecks, improving macro-scale flow estimation accuracy.

Furthermore, exploring end-to-end fine-tuning paradigms via Low-Rank Adaptation (LoRA) [8] could tailor the frozen spatial embeddings explicitly for systemic network flow rather than generic visual similarity.

Few-shot adaptation for global cities. To address the crosscontinental split discrepancy, future research should explore fewshot global adaptation. Rather than relying purely on zero-shot transfer, meta-learning approaches (e.g., MAML [6] or Reptile [14]) could enable fast adaptation by exposing the difusion model to a small held-out subset of target-region zones during fine-tuning, re-anchoring layout distributions with minimal target-domain supervision.

Uncertainty quantification. Finally, the generative nature of the difusion process natively afords uncertainty quantification. Future iterations can leverage the system’s existing generative capacity by exposing the per-sample variance across the 50 DDIM draws as calibrated confidence intervals. Reporting this uncertainty around predicted OD flow pairs would allow transport planners to identify unreliable flow estimates prior to downstream analysis.

## 7 Conclusion

We compared four spatial vision foundation models within a fixed generative OD graph difusion pipeline. Across 1,925 US counties, 325 UK districts, and 14 global cities, three core findings emerge. First, language-supervised representations (RemoteCLIP) capture structural dependencies accurately inside the training domain but exhibit fragile cross-continental generalization due to morphological distribution shifts; RemoteCLIP achieves a leading CPC of 0.602 on US counties, but drops sharply under transfer conditions. Second, geographical and physical grounding provides superior resilience for zero-shot transfer across continents; AlphaEarth leads UK zeroshot transfer at a peak CPC of 0.515 at �=0, representing a 33% relative improvement over RemoteCLIP at that operating point. Third, massive pretraining dataset volume alone cannot compensate for task-specific domain alignment, as evidenced by DINOv3’s empirical performance limitations and near-zero global CPC. These outcomes suggest that embedding coordinate-contrastive or geophysical structural priors within deep vision architectures is a promising direction for geographically transferable urban mobility generation at scale.

## Acknowledgments

We thank the support of the ARC Centre of Excellence for Automated Decision-Making and Society (CE200100005). This research includes computations using the computational cluster Katana supported by Research Technology Services at UNSW Sydney [22]. Satellite imagery was accessed via Google Earth Engine. US commuting data from LODES (US Census Bureau); UK and global city data from [16]: UK flows from Ofice for National Statistics (ONS) registries, global flows from the travel-survey and mobile-signalling sources documented therein. WorldPop population grids provided by the WorldPop Project (University of Southampton). The WeDAN architecture and GlODGen codebase are due to [16] (Tsinghua University FIB Lab).

## References

[1] Christopher F Brown, Michal R Kazmierski, Valerie J Pasquarella, William J Rucklidge, Masha Samsikova, Chenhui Zhang, Evan Shelhamer, Estefania Lahera Olivia Wiles, Simon Ilyushchenko, et al. 2025. Alphaearth foundations: An embedding field model for accurate and eficient global mapping from sparse label data.

[2] Jiayu Chang, Tian Liang, Wanzhi Xiao, and Li Kuang. 2023. Origin-Destination Convolution Recurrent Network: A Novel OD Matrix Prediction Framework. In International Conference on Collaborative Computing: Networking, Applications and Worksharing. Springer, 131–150.

[3] Michael Drusch, Umberto Del Bello, Sébastien Carlier, Olivier Colin, Veronica Fernandez, Ferran Gascon, Bianca Hoersch, Claudia Isola, Paolo Laberinti, Philippe Martimort, et al. 2012. Sentinel-2: ESA’s Optical High-Resolution Mission for GMES Operational Services. Remote Sensing ofEnvironment 120 (2012), 25–36.

[4] Vijay Prakash Dwivedi and Xavier Bresson. 2020. A Generalization of Trans former Networks to Graphs. arXiv preprint arXiv:2012.09699 (2020).

[5] Floor Eijkelboom, Grigory Bartosh, Christian A. Naesseth, Max Welling, and Jan-Willem van de Meent. 2024. Variational Flow Matching for Graph Generation. In Advances in Neural Information Processing Systems.

[6] Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. In International Conference on Machine Learning. PMLR, 1126–1135.

[7] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems, Vol. 33. 6840–6851.

[8] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

[9] Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. 2022. Score-based Generative Modeling of Graphs via the System of Stochastic Diferential Equations. In International Conference on Machine Learning. https://arxiv.org/abs/2202.02514

[10] Konstantin Klemmer, Esther Rolf, Caleb Robinson, Lester Mackey, and Marc Rußwurm. 2025. Satclip: Global, general-purpose location embeddings with satellite imagery. 4347–4355 pages.

[11] Maxime Lenormand, Sylvie Huet, Floriana Gargiulo, and Guillaume Defuant. 2012. A universal model of commuting networks. (2012).

[12] Fan Liu, Delong Chen, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, and Jun Zhou. 2024. Remoteclip: A vision language foundation model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing 62 (2024), 1–16.

[13] Zhicheng Liu, Fabio Miranda, Weiting Xiong, Junyan Yang, Qiao Wang, and Claudio Silva. 2020. Learning geo-contextual embeddings for commuting flow prediction. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 34. 808–816.

[14] Alex Nichol, Joshua Achiam, and John Schulman. 2018. On First-Order Meta-Learning Algorithms. arXiv preprint arXiv:1803.02999 (2018).

[15] Can Rong, Jingtao Ding, Zhicheng Liu, and Yong Li. 2023. Complexity-aware large scale origin-destination network generation via difusion model. arXiv preprint arXiv:2306.04873 (2023).

[16] Can Rong, Xin Zhang, Yanxin Xi, Hongjie Sui, Jingtao Ding, and Yong Li. 2026. Satellites reveal mobility: A commuting origin-destination flow generator for global cities. Advances in Neural Information Processing Systems 38 (2026)

[17] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Rama monjisoa, et al. 2025. Dinov3.

[18] Filippo Simini, Gianni Barlacchi, Massimilano Luca, and Luca Pappalardo. 2021. A deep gravity model for mobility flows generation. Nature communications 12, 1 (2021), 6576.

[19] Filippo Simini, Marta C González, Amos Maritan, and Albert-László Barabási. 2012. A universal model for mobility and migration patterns. Nature 484, 7392 (2012), 96–100.

[20] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising difusion implicit models. arXiv preprint arXiv:2010.02502.

[21] Gencer Sumbul, Marcela Charfuelan, Begüm Demir, and Volker Markl. 2019. Bigearthnet: A large-scale benchmark archive for remote sensing image understanding. arXiv preprint arXiv:1902.06148.

[22] UNSW Sydney. 2010. Katana. doi:10.26190/669x-a286

[23] Qi Zhu, Natalia Ponomareva, Jiawei Han, and Bryan Perozzi. 2021. Shift-robust gnns: Overcoming the limitations of localized graph training data. Advances in Neural Information Processing Systems 34, 27965–27977.

[24] George Kingsley Zipf. 1946. The P 1 P 2/D hypothesis: on the intercity movement of persons. American sociological review 11, 6 (1946), 677–686.

## A Complete Per-Region Results

Table 6: US performance metrics by � level (mean over seeds).
<table><tr><td>Encoder</td><td>η</td><td>CPC↑</td><td>RMSE↓</td><td>MAE↓</td></tr><tr><td rowspan="5">RemoteCLIP</td><td>0</td><td>0.590</td><td>69.0</td><td>40.1</td></tr><tr><td>0.25</td><td>0.596</td><td>68.2</td><td>39.9</td></tr><tr><td>0.5</td><td>0.600</td><td>67.9</td><td>39.6</td></tr><tr><td>0.75</td><td>0.602</td><td>67.4</td><td>39.4</td></tr><tr><td>1</td><td>0.602</td><td>67.8</td><td>39.6</td></tr><tr><td rowspan="5">AlphaEarth</td><td>0</td><td>0.549</td><td>73.0</td><td>42.8</td></tr><tr><td>0.25</td><td>0.556</td><td>72.6</td><td>43.0</td></tr><tr><td>0.5</td><td>0.568</td><td>71.5</td><td>42.5</td></tr><tr><td>0.75</td><td>0.580</td><td>70.1</td><td>41.4</td></tr><tr><td>1</td><td>0.588</td><td>68.2</td><td>39.0</td></tr><tr><td rowspan="5">SatCLIP</td><td>0</td><td>0.539</td><td>74.9</td><td>44.6</td></tr><tr><td>0.25</td><td>0.546</td><td>73.9</td><td>44.2</td></tr><tr><td>0.5</td><td>0.552</td><td>73.3</td><td>43.8</td></tr><tr><td>0.75</td><td>0.565</td><td>71.6</td><td>42.6</td></tr><tr><td>1</td><td>0.575</td><td>70.2</td><td>41.1</td></tr><tr><td rowspan="5">DINOv3</td><td>0</td><td>0.506</td><td>78.3</td><td>44.3</td></tr><tr><td>0.25</td><td>0.508</td><td>77.6</td><td>44.0</td></tr><tr><td>0.5</td><td>0.510</td><td>77.5</td><td>43.8</td></tr><tr><td>0.75</td><td>0.511</td><td>77.2</td><td>43.4</td></tr><tr><td>1</td><td>0.511</td><td>76.6</td><td>42.9</td></tr></table>

Table 8: Global performance metrics by � level (mean over seeds). <sup>†</sup>�=4 seeds; see Table 3.
<table><tr><td>Encoder</td><td>η</td><td>CPC↑</td><td>RMSE↓</td><td>MAE↓</td></tr><tr><td rowspan="5">RemoteCLIP</td><td>0</td><td>0.125</td><td>587.7</td><td>191.5</td></tr><tr><td>0.25</td><td>0.121</td><td>690.7</td><td>225.2</td></tr><tr><td>0.5</td><td>0.122</td><td>588.9</td><td>191.8</td></tr><tr><td>0.75</td><td>0.124</td><td>604.8</td><td>195.9</td></tr><tr><td>1</td><td>0.119</td><td>689.7</td><td>224.5</td></tr><tr><td rowspan="5">AlphaEarth</td><td>0</td><td>0.116</td><td>774.6</td><td>254.2</td></tr><tr><td>0.25</td><td>0.119</td><td>577.7</td><td>189.7</td></tr><tr><td>0.5</td><td>0.113</td><td>573.4</td><td>189.8</td></tr><tr><td>0.75</td><td>0.110</td><td>571.9</td><td>188.8</td></tr><tr><td>1</td><td>0.098</td><td>566.4</td><td>184.1</td></tr><tr><td rowspan="5">SatCLIP†</td><td>0</td><td>0.114</td><td>607.6</td><td>195.1</td></tr><tr><td>0.25</td><td>0.106</td><td>737.6</td><td>237.5</td></tr><tr><td>0.5</td><td>0.112</td><td>615.9</td><td>197.2</td></tr><tr><td>0.75</td><td>0.099</td><td>621.1</td><td>199.3</td></tr><tr><td>1</td><td>0.091</td><td>748.2</td><td>241.0</td></tr><tr><td rowspan="5">DINOv3</td><td>0</td><td>0.021</td><td>1045.8</td><td>583.9</td></tr><tr><td>0.25</td><td>0.022</td><td>1114.0</td><td>584.5</td></tr><tr><td>0.5</td><td>0.022</td><td>1012.4</td><td>550.3</td></tr><tr><td>0.75</td><td>0.022</td><td>1106.3</td><td>578.9</td></tr><tr><td>1</td><td>0.022</td><td>1135.9</td><td>606.0</td></tr></table>

Table 7: UK performance metrics by � level (mean over seeds).
<table><tr><td>Encoder</td><td>η</td><td>CPC↑</td><td>RMSE↓</td><td>MAE↓</td></tr><tr><td rowspan="5">RemoteCLIP</td><td>0</td><td>0.388</td><td>121.6</td><td>88.5</td></tr><tr><td>0.25</td><td>0.377</td><td>112.5</td><td>79.2</td></tr><tr><td>0.5</td><td>0.355</td><td>102.8</td><td>69.2</td></tr><tr><td>0.75</td><td>0.319</td><td>93.4</td><td>60.4</td></tr><tr><td>1</td><td>0.296</td><td>89.4</td><td>57.2</td></tr><tr><td rowspan="5">AlphaEarth</td><td>0</td><td>0.515</td><td>80.6</td><td>57.5</td></tr><tr><td>0.25</td><td>0.515</td><td>80.7</td><td>57.9</td></tr><tr><td>0.5</td><td>0.510</td><td>80.2</td><td>56.8</td></tr><tr><td>0.75</td><td>0.488</td><td>78.1</td><td>52.2</td></tr><tr><td>1</td><td>0.249</td><td>78.9</td><td>46.4</td></tr><tr><td rowspan="5">SatCLIP</td><td>0</td><td>0.512</td><td>81.4</td><td>58.1</td></tr><tr><td>0.25</td><td>0.503</td><td>84.6</td><td>62.2</td></tr><tr><td>0.5</td><td>0.495</td><td>84.6</td><td>61.5</td></tr><tr><td>0.75</td><td>0.476</td><td>83.9</td><td>59.2</td></tr><tr><td>1</td><td>0.455</td><td>82.8</td><td>56.7</td></tr><tr><td rowspan="5">DINOv3</td><td>0</td><td>0.330</td><td>338.2</td><td>197.0</td></tr><tr><td>0.25</td><td>0.329</td><td>341.9</td><td>199.9</td></tr><tr><td>0.5</td><td>0.329</td><td>342.7</td><td>200.4</td></tr><tr><td>0.75</td><td>0.329</td><td>344.4</td><td>201.6</td></tr><tr><td>1</td><td>0.330</td><td>343.4</td><td>200.6</td></tr></table>