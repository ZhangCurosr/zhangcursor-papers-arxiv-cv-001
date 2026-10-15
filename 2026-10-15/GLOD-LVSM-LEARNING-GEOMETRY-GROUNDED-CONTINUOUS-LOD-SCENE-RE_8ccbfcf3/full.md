# GLOD-LVSM: LEARNING GEOMETRY-GROUNDED CONTINUOUS-LOD SCENE REPRESENTATIONS FOR LARGE VIEW SYNTHESIS MODELS

Weikai Lin<sup>1,2</sup> Nicholas Milef<sup>2</sup>

Carl S Marshall<sup>2</sup> Grant Kaijuin Yang<sup>2</sup>

Todd Keeler<sup>2</sup>

Yuhao Zhu<sup>1</sup>

<sup>1</sup>University of Rochester

<sup>2</sup>Reality Labs, Meta

![](images/46af43a88e8ec6c74b0831c67b187c7b0793fc6b4eede9583cab76723fe2382a.jpg)  
Fig. 1: GLOD-LVSM learns a compact, geometry-grounded scene representation (anchors) for novel view synthesis. Left: View-synthesis supervision and compression alone yield emergent depth and self-organized, surface-aligned scene anchors whose density adapts to local texture complexity. Right: GLOD-LVSM achieves the best quality–scene-token trade-off across reference-view counts.

## ABSTRACT

Large view synthesis models (LVSMs) reconstruct scenes from sparse views and render novel views quickly, but they incur unnecessary storage and rendering costs due to token redundancy. We present GLOD-LVSM, a compact, geometrygrounded scene representation for LVSMs with continuous levels of detail. First, instead of learning a fixed set of scene anchor initializations, GLOD-LVSM learns a continuous field that initializes a flexible number of anchors to match the desired budget. A condenser then aggregates scene information into these anchors, with both the field and condenser trained end-to-end. Second, the condenser infers reference-token and anchor geometry to guide compression through geometrygrounded attention, improving rendering quality and enabling frustum culling during decoding. Compared with existing LVSM scene-compression baselines, GLOD-LVSM achieves the best trade-off between rendering quality and scene token count. Compared with the uncompressed base model, GLOD-LVSM reduces the scene-token count by 33.4% and accelerates rendering by 1.23× with no quality loss. Under a relaxed quality-loss budget, the reduction increases to 85.5%, with 3.00× rendering speedup. Further analysis reveals that view-synthesis supervision yields emergent scene depth and self-organized anchors that align with surfaces and adapt their density to local texture complexity. Beyond rendering, we further show that our scene representation also supports scene-semantic understanding, highlighting its potential for broader downstream tasks.

## 1 INTRODUCTION

Large view synthesis models (LVSMs) learn a reconstruction prior for novel view synthesis (NVS) (Jin et al., 2025; Kim et al., 2026; Szymanowicz et al., 2026), reducing scene reconstruction to a single feed-forward pass and enabling high-quality rendering from sparse reference views. Recent highquality LVSMs commonly adopt a highway representation, where each reference-image patch is directly mapped to a scene token (Szymanowicz et al., 2026; Kim et al., 2026). While this design preserves detailed appearance information, the resulting scene-token count grows with the number of reference views, increasing both storage and decoding costs. This tight coupling between image and scene-representation sizes becomes a bottleneck with more reference views.

In addition to scaling poorly with reference view count, highway representations allocate scene capacity inefficiently in two ways. First, overlapping 3D regions observed from different views are stored as separate tokens rather than consolidated. Second, every image patch receives the same one-token capacity regardless of texture complexity. An ideal LVSM representation should therefore (1) support flexible scene-token count independent of the input-view count, (2) consolidate multi-view observations into a unified, scene-centric 3D representation, and (3) allocate capacity adaptively to scene complexity. Thus, we propose GLOD-LVSM, which provides two key features: (1) a geometry-grounded scene representation that supports (2) continuous level-of-detail. We refer to scene tokens in GLOD-LVSM as anchors to emphasize their scene-centric 3D nature.

Different storage and rendering budgets require different scene-token numbers, which correspond to different levels of scene detail. To support arbitrary scene-token counts and thus continuous levels of detail (CLoD), we first design an anchor initialization scheme that can generate a flexible number of tokens. Specifically, we introduce a CLoD scene-anchor initialization that parameterizes anchor initialization as a continuous field (Sec. 3.2). Sampling this field at different densities generates different numbers of anchors. This decouples the scene-token count from the reference-view count and allows our model to operate under different storage and rendering budgets.

A compact scene representation should consolidate multi-view observations and allocate capacity where it is most useful. To achieve this, we (1) use unified, scene-centric 3D anchors to consolidate redundant multi-view tokens, and (2) learn the spatial allocation of these anchors end-to-end. Specifi cally, we use an end-to-end learned, geometry-grounded condenser that aggregates reference-view tokens into 3D anchors (Sec. 3.3). The condenser predicts geometry for both views and anchors and uses their geometric relationships to guide cross-attention. The resulting anchors have 3D positions, which further enable target-view frustum culling to reduce decoding cost (Sec. 3.4). Both the geometry and scene-complexity-aware anchor allocation emerge from the NVS objective (Sec. 4.6).

We evaluate GLOD-LVSM across datasets, reference-view counts and scene-token budgets. Quantitatively, GLOD-LVSM consistently achieves the best quality–token-count trade-off among the baselines. Qualitatively, the learned anchors emerge near scene surfaces and allocate higher density to regions with richer texture, suggesting that NVS supervision and compression alone can induce a geometry- and complexity-aware efficient scene representation. Beyond rendering, we further evaluate the learned representation on scene-semantic understanding, demonstrating its potential as a 3D scene representation for broader downstream tasks. Our contributions are threefold:

• Continuous-LoD scene representation. We introduce a scene-conditioned continuous query field that enables flexible scene anchor budgets independent of the reference-view count.

• Geometry-grounded scene condensation. We learn scene-centric 3D anchors end-to-end, enabling efficient multi-view consolidation and scene-adaptive capacity allocation.

• Efficient and generalizable scene anchors. GLOD-LVSM achieves the best quality–scenetoken trade-off for rendering and supports scene semantic understanding.

## 2 RELATED WORK

## 2.1 NOVEL VIEW SYNTHESIS

Novel view synthesis (NVS) is a widely studied problem and can be broadly categorized into perscene optimization methods and feed-forward generalizable methods. NeRF (Mildenhall et al., 2020)

and 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) achieve photorealistic novel-view synthesis by optimizing a radiance field or a set of Gaussian primitives for each scene. However, they typically require time-consuming per-scene optimization and dense image capture to achieve high rendering quality. In contrast, generalizable NVS methods (Sajjadi et al., 2022; Chen et al., 2024; Xu et al., 2025), including recent Transformer-based approaches such as LVSMs (Jin et al., 2025; Kim et al., 2026; Sun et al., 2026; Szymanowicz et al., 2026), learn reconstruction priors from large datasets and can reconstruct unseen scenes from sparse input views in a single feed-forward pass, avoiding costly dense image capture and per-scene optimization. Our work builds on Transformer-based generalizable NVS and focuses on learning a compact, geometry-grounded scene representation that reduces redundancy among scene tokens, thereby lowering storage and rendering costs.

## 2.2 COMPACT SCENE REPRESENTATION

Compact scene representations have been widely studied for efficient NVS. For example, 3DGS methods prune or compress Gaussian primitives to reduce representation size while preserving visual quality (Fan et al., 2024; Fang & Wang, 2024; Lin et al., 2025). Some methods further support multiple or continuous levels of detail (CLoD), allowing representation complexity to adapt to runtime rendering budgets (Milef et al., 2025; Cheng et al., 2026). Compact scene-token representations have also been explored for LVSMs. LVSM-ED (Jin et al., 2025) uses a fixed latent bottleneck and cannot support flexible compression rates. CLiFT (Wang et al., 2025) uses K-means to select representative reference-view tokens, relying on handcrafted clustering that may be suboptimal. Our work also targets compact scene representations for LVSMs. Unlike prior methods, we support CLoD and learn scene anchors end-to-end, enabling flexible compression with less handcrafted design.

## 3 GLOD-LVSM: GEOMETRY-GROUNDED CLOD SCENE REPRESENTATION

Fig. 2 illustrates GLOD-LVSM. Given V reference images and their camera parameters, GLOD-LVSM constructs a compact scene representation with a chosen budget of K anchors. A highway scene encoder first extracts reference-view tokens (Sec. 3.1). Our anchor initializer then generates K anchors from a continuous field (Sec. 3.2). These anchors are passed to our condenser, which produces scene-centric 3D anchors (Sec. 3.3). The anchors will be used to render the RGB images given the target views (Sec. 3.4). We train GLOD-LVSM end-to-end using the NVS loss (Sec. 3.5).

## 3.1 HIGHWAY SCENE ENCODER

We use the encoder from LagerNVS (Szymanowicz et al., 2026). The inputs are $\boldsymbol { \mathcal { X } } = \{ ( \mathbf { I } _ { v } , \mathcal { P } _ { v } ) \} _ { v = 1 } ^ { V } ,$ where I<sub>v</sub> is the v-th reference image and $\mathcal { P } _ { v }$ is its camera parameters. The encoder patchifies each reference image, embeds each patch as a token, embeds $\mathcal { P } _ { v }$ into a camera token $\mathbf { t } _ { v } ^ { \mathrm { c a m } }$ , and appends four register tokens $\mathbf { t } _ { 1 : 4 } ^ { \mathrm { r e g } }$ to each view (Darcet et al., 2024). For each view, input token sequence $\mathbf { T } _ { v } ^ { \mathrm { i n } } = \bar  [ \mathbf { t } _ { v } ^ { \mathrm { c a m } } , \mathbf { t } _ { 1 : 4 } ^ { \mathrm { r e g } }$ , PatchEmbed(I )]. The resulting tokens are jointly encoded across all V views:

$$
\mathbf { H } = \operatorname { E n c o d e r } ( \{ \mathbf { T } _ { v } ^ { \mathrm { i n } } \} _ { v = 1 } ^ { V } ) , \qquad Z = \mathbf { H } _ { \mathrm { p a t c h } } \mathbf { W } \in \mathbb { R } ^ { N \times D } .\tag{1}
$$

Here, W linearly maps the encoded patch tokens to a defined D-dimensional space, and N is the total number of input patches across all views. We refer to the output $z$ as reference-view tokens and retain the encoded register and camera tokens for anchor initialization (Sec. 3.2). This one-patch-one-token highway design causes storage and decoding costs to grow with reference-view count.

## 3.2 CONTEXTUALIZED CONTINUOUS FIELD FOR ANCHOR INITIALIZATION

Given reference-view tokens Z, our condenser compresses them into scene anchors. The first challenge is anchor initialization. Fixed learned vectors (Jin et al., 2025) support only a predefined anchor count. Clustering-based initialization (Wang et al., 2025) supports variable counts, but relies on handcrafted clustering that may be suboptimal and remains image-centric rather than scenecentric. We instead learn a scene-conditioned continuous initialization field end-to-end with minimal handcrafted design (Fig. 2(a)). Sampling the field at different densities yields different anchor budgets, enabling continuous levels of detail under varying storage and rendering constraints.

![](images/0f0f9d418555818d56d2262ce21693d1ef445262ae111a8858b4c30ee5b71e53.jpg)  
Fig. 2: Overview of GLOD-LVSM. A highway encoder extracts reference-view tokens and per-view context. (a) A scene-conditioned field uses box-filtered positional encodings to initialize the chosen number of anchors. (b) The condenser predicts geometry for both reference-view tokens and anchors and uses geometry-grounded bidirectional attention to condense reference-view tokens into anchors. The same scene anchors are reused to render different target views.

Continuous field parameterization. Given a total anchor budget K, we distribute K evenly across the V reference views, assigning $K _ { v }$ anchors to view v. For each view, our goal is to learn a D-dimensional continuous field that maps query indices $i \in \{ 0 , \ldots , K _ { v } - 1 \}$ to $K _ { v }$ vectors in the D-dimensional anchor space, with D set to the highway model’s scene-token dimension. We parameterize this continuous field as an MLP conditioned jointly on the anchor indices and the current scene/view context. Specifically, each anchor index i is mapped to a continuous coordinate $u _ { v , i }$ and encoded with our anti-aliased positional encoding $\bar { \gamma } ( u _ { v , i } )$ (Eqn. 3). To condition the field on the current input, we mean-pool the encoded registers and camera tokens (Sec. 3.1) of view v and map them through $\mathrm { M L P } _ { \mathrm { c o n t e x t } }$ to obtain a view- and scene-specific context vector $\mathbf { c } _ { v } . \bar { \gamma } ( u _ { v , i } )$ and $\mathbf { c } _ { v }$ are then concatenated and passed to ${ \mathrm { M L P } } _ { \mathrm { f i e l d } }$ to produce the D-dimensional anchor initialization $\mathbf { \alpha } _ { a _ { v , i } ; }$

$$
\begin{array} { r } { \pmb { a } _ { v , i } = \mathrm { M L P } _ { \mathrm { f i e l d } } \left( \left[ \overline { { \gamma } } ( u _ { v , i } ) , \mathbf { c } _ { v } \right] \right) , \qquad \mathbf { c } _ { v } = \mathrm { M L P } _ { \mathrm { c o n t e x t } } \left( \mathrm { M e a n P o o l } ( \left[ \mathbf { H } _ { v , 1 : 4 } ^ { \mathrm { r e g } } , \mathbf { H } _ { v } ^ { \mathrm { c a m } } \right] ) \right) } \end{array}\tag{2}
$$

Finally, we concatenate the initializations from all views to obtain $\begin{array} { r } { K = \sum _ { v } K _ { v } } \end{array}$ anchor initializations and feed them to the condenser. The field is optimized end-to-end through the NVS objective.

Anti-Aliased Positional Encoding Across Anchor Budgets. To sample the same continuous field at different anchor budgets, we map each discrete query index i to a continuous coordinate $u _ { v , i }$ in a fixed canonical domain $[ 0 , K _ { \mathrm { c a n o n i c a l } } ]$ , where $K _ { \mathrm { c a n o n i c a l } } = 2 0 0 0$ . For each view with budget $K _ { v } .$ , we divide the domain into $K _ { v }$ intervals. Changing $K _ { v }$ therefore changes the sampling density while preserving coverage of the domain. To help the field capture high-frequency variation, we follow NeRF-style positional encoding (PE) (Mildenhall et al., 2020) and encode each coordinate as $\begin{array} { r } { \gamma ( u ) = \frac { 1 } { \sqrt { F } } [ \overline { { \cos ( 2 \pi \mathbf { f } u ) } } , \sin ( 2 \pi \mathbf { f } u ) ] \stackrel { \smile } { \in } \mathbb { R } ^ { 2 F } } \end{array}$ . Here, $\mathbf { f } = \left( f _ { 0 } , \ldots , f _ { F - 1 } \right)$ is logarithmically spaced between $f _ { \mathrm { m i n } } = 1 / 2 0 0 0 0$ and $f _ { \mathrm { m a x } } = 0 . 5$ . We set $F = D / 2$ , where D is the dimension of anchor space. At small budgets, sparse sampling can alias the high-frequency components of PE. We therefore integrate PE over the sampling interval, analogous to box filtering (Barron et al., 2021):

$$
\bar { \gamma } _ { v , i } = \frac { 1 } { \Delta _ { v } } \int _ { I _ { v , i } } \gamma ( u ) d u , \quad I _ { v , i } = [ u _ { v , i } , u _ { v , i } + \Delta _ { v } ] , \quad u _ { v , i } = i \Delta _ { v } , \quad \Delta _ { v } = \frac { K _ { \mathrm { c a n o n i c a l } } } { K _ { v } } .\tag{3}
$$

As the anchor budget decreases, the intervals become wider and more strongly attenuate highfrequency components. The sine and cosine integrals have closed-form solutions, given in Supple-

mentary A.1. The resulting $\bar { \gamma } ( u _ { v , i } ) \in \mathbb R ^ { D }$ provides a D-dimensional encoding of each anchor index whose frequency content reflects the anchor budget, with lower budgets inducing stronger attenuation.

## 3.3 GEOMETRY-GROUNDED BIDIRECTIONAL CONDENSATION

As illustrated in Fig. 2(b), our condenser maps K initialized anchors and N reference-view tokens $z$ to K compact scene anchors. It uses $L = 8$ bidirectional Transformer blocks (Szymanowicz et al., 2026), where the two streams attend to each other. Content-only attention can mix spatially unrelated information because it lacks 3D awareness. We therefore predict geometry for both streams (Sec. 3.3.1) and use it to modulate the bidirectional cross-attention (Sec. 3.3.2). In the condenser, we retain the reference-view register and camera tokens and add four anchor registers. These tokens also participate in attention, but with no geometry modulation.

## 3.3.1 PREDICTING VIEW AND ANCHOR GEOMETRY

View Geometry. We predict geometry for each reference-view token directly from its feature. Following RayRoPE (Wu et al., 2026), a linear depth head maps token $z _ { j }$ to a log-depth center $\mu _ { j }$ and an uncertainty scale $\sigma _ { j }$ . Using the known camera parameters, we back-project the four corners of the corresponding image patch at depth $d _ { j } = e ^ { \mu _ { j } }$ to obtain its spatial span in 3D space $\mathbf { p } _ { j , 1 : 4 }$ . We also use the unit center-ray direction $\mathbf { v } _ { j }$ to encode its viewing direction. Together, $\{ \mathbf { p } _ { j , 1 : 4 } , \mathbf { v } _ { j } , \sigma _ { j } \}$ describes the spatial support and viewing direction of reference-view token $j .$ . The geometry is recomputed from the updated reference-view features at every condenser layer.

Anchor Geometry. Rather than directly regressing unconstrained 3D coordinates for each anchor, we derive its geometry by aggregating the geometry of the reference-view tokens. This introduces a useful prior: it constrains anchors to the convex hull of the observed geometry rather than allowing unconstrained 3D positions. Specifically, we use a lightweight single-head cross-attention to compute aggregation weights $\beta _ { i j }$ between anchor i and reference-view token j:

$$
\beta _ { i j } = \mathrm { s o f t m a x } _ { j } \left( \frac { \hat { \mathbf { q } } _ { i } ^ { \top } \hat { \mathbf { k } } _ { j } } { \sqrt { 6 4 } } \right) , \qquad \mathbf { p } _ { i } ^ { A } = \sum _ { j } \beta _ { i j } \bar { \mathbf { p } } _ { j } , \qquad \mathbf { v } _ { i } ^ { A } = \sum _ { j } \beta _ { i j } \mathbf { v } _ { j } ,\tag{4}
$$

where $\hat { \mathbf { q } } _ { i }$ and $\hat { \mathbf { k } } _ { j }$ are 64-dimensional query and key projections, and $\begin{array} { r } { \bar { \bf p } _ { j } = \frac { 1 } { 4 } \sum _ { c = 1 } ^ { 4 } { \bf p } _ { j , c } } \end{array}$ is the center of the back-projected view-token patch. The same weights aggregate both position and viewing direction, giving each anchor a 3D position $\mathbf { p } _ { i } ^ { A }$ and a viewing-direction descriptor $\mathbf { v } _ { i } ^ { A } . ~ \beta _ { i j }$ is normalized over the reference-view tokens. Importantly, we use the anchor geometry predicted at the final condenser block as the anchor’s final geometry for subsequent rendering.

## 3.3.2 GEOMETRY-GROUNDED BIDIRECTIONAL ATTENTION

After geometry prediction, each condenser block first applies self-attention to the anchor stream, followed by bidirectional cross-attention between anchors and reference-view tokens and MLPs. We inject the inferred geometry into this cross-attention using geometry-dependent RoPE.

We define the geometric descriptors for the two token streams as follows. A reference-view token $j$ corresponds to an image patch and is represented by its four back-projected patch corners $\mathbf { p } _ { j , 1 : 4 }$ viewing direction $\mathbf { v } _ { j }$ , and depth uncertainty $\sigma _ { j }$ . An anchor $i ,$ in contrast, has a single 3D position $\mathbf { p } _ { i } ^ { A }$ and viewing-direction descriptor $\mathbf { v } _ { i } ^ { A }$ . To match the geometric layout of the reference-view tokens, we repeat $\mathbf { p } _ { i } ^ { A }$ four times for each anchor. Following RayRoPE (Wu et al., 2026), we represent a 3D point using $\pi ( \mathbf { p } )$ , which first transforms the point into the reference camera frame to obtain $( X , Y , Z )$ , then computes $\pi ( \mathbf { p } ) = ( X / r , Y / r , 1 / | Z | )$ with $r = \sqrt { X ^ { 2 } + Y ^ { 2 } + Z ^ { 2 } }$ . The four points therefore provide 12 coordinates. Appending the 3D viewing-direction descriptor gives a 15-dimensional geometric descriptor for both reference-view tokens and anchors:

$$
\begin{array} { r } { \mathbf { g } _ { j } ^ { R } = [ \pi ( \mathbf { p } _ { j , 1 } ) , \ldots , \pi ( \mathbf { p } _ { j , 4 } ) , \mathbf { v } _ { j } ] , \qquad \mathbf { g } _ { i } ^ { A } = [ \pi ( \mathbf { p } _ { i } ^ { A } ) , \ldots , \pi ( \mathbf { p } _ { i } ^ { A } ) , \mathbf { v } _ { i } ^ { A } ] \in \mathbb { R } ^ { 1 5 } . } \end{array}\tag{5}
$$

For reference-view token geometry $\mathbf { g } _ { j } ^ { R }$ , we use RayRoPE’s expected RoPE as encoding to account for the predicted depth uncertainty $\sigma _ { j }$ (Sec. 3.3.1). For anchor geometry $\mathbf { g } _ { i } ^ { A }$ , we use standard RoPE encoding (Su et al., 2024). The encodings are then used to rotate the query, key, value, and output features in the bidirectional cross-attention, following the rotation mechanism in RayRoPE.

A key challenge is choosing the RoPE coordinate frame. When an anchor interacts with tokens from a reference view, we express RoPE in that view’s frame for both cross-attention directions. This avoids dependence on the arbitrary world coordinate system, similar to conventional graphics pipelines that transform scene geometry into camera space before rasterization (Akenine-Möller et al., 2016).

## 3.4 RUNTIME CULLING AND RENDERING

Culling. After scene reconstruction, the same set of anchors can be reused across different target views. We use the anchor geometry from the final condenser layer as the geometry of each anchor. Given a target camera $\mathcal { P } _ { t } ,$ we project each anchor position $\mathbf { p } _ { i } ^ { A }$ into the target camera frame and obtain its normalized image coordinate $( u _ { i } ^ { t } , v _ { i } ^ { t } )$ and camera-space depth $z _ { i } ^ { t }$ . We retain anchors that lie in front of the camera and inside a slightly padded image region, $\mathcal { K } _ { t } \dot { = } \{ i : z _ { i } ^ { t } > 0 , ~ ( u _ { i } ^ { t } , v _ { i } ^ { t } ) \in [ - \delta , 1 + \delta ] ^ { 2 } \}$ with $\delta = 0 . 1$ . The padding retains anchors close to the image boundary. Only anchors in $\textstyle { \mathcal { K } } _ { t }$ are passed to the decoder, reducing the number of scene tokens processed for each target view.

Novel-View Rendering. We use the bidirectional decoder of LagerNVS (Szymanowicz et al., 2026), modified with our geometry-grounded RoPE for anchor–target-token cross-attention (Sec. 3.3.2). The target camera is converted into a ray map and patch-embedded into target tokens, which predict their view geometry with a linear head (as for reference-view tokens) and interact with anchors through bidirectional cross-attention. Anchors use the fixed geometry from the final condenser layer. The target camera frame is used as the RoPE coordinate frame. Finally, a linear projection maps target tokens to RGB patches, which form the rendered image $\hat { \mathbf { I } } _ { t }$

## 3.5 TRAINING

We freeze the pretrained scene encoder and jointly optimize the continuous query field, condenser, and decoder. The decoder is supervised by the LagerNVS rendering loss $\mathcal { L } _ { \mathrm { r e n d e r } }$ (Szymanowicz et al., 2026). We additionally regularize the predicted depth uncertainty $\sigma _ { j } ^ { \ell }$ of reference-view token j at condenser layer ℓ (Sec. 3.3.1) to encourage sharp, localized geometry:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e n d e r } } + \lambda _ { \sigma } \mathcal { L } _ { \mathrm { d e c a y } } , \qquad \mathcal { L } _ { \mathrm { d e c a y } } = \frac { 1 } { L N } \sum _ { \ell = 0 } ^ { L - 1 } \sum _ { j = 1 } ^ { N } | \sigma _ { j } ^ { \ell } | , \qquad \lambda _ { \sigma } = 1 0 ^ { - 3 } .\tag{6}
$$

We first train without culling, then enable culling and perform culling-aware fine-tuning. During training, we sample 1–10 reference views and uniformly sample the anchor budget from 256 to the corresponding highway-token count N. No explicit depth or anchor-position supervision is used.

## 4 EVALUATION

## 4.1 EXPERIMENTAL SETUP

Datasets. We evaluate on RealEstate10K (Zhou et al., 2018) and DL3DV (Ling et al., 2024) at resolutions of 256 × 256 and 256 × 448, respectively. We study varying-reference-view settings from 2 to 10 views. Since the official protocols cover only 2 views for RealEstate10K and 2/4/6 views for DL3DV, we construct the higher-view settings using the method described in Supplementary A.2.

Training. We fine-tune the pretrained general LagerNVS checkpoint (Szymanowicz et al., 2026) on each dataset using its 1–10-view training protocol, obtaining LAGERNVS-FT, which is our base model. We initialize our decoder from LAGERNVS-FT and train the condenser and decoder, followed by culling-aware fine-tuning. See Supplementary A.3 for training details.

Baselines. We compare against two groups of baselines:

• System-level baselines. We compare GLOD-LVSM against CLIFT (Wang et al., 2025), LVSM (Jin et al., 2025), MVSPLAT (Chen et al., 2024), and DEPTHSPLAT (Xu et al., 2025).

![](images/d478c02d4ff288f9b084ba538f525b9ab3173cc3f5497e1450c2608d8538ffa6.jpg)  
Fig. 3: PSNR across datasets and view counts. Fig. 4: PSNR vs. scene-token counts. Shaded Shaded bands show 95% CI. bands show 95% CI.

• Condenser baselines. Alternative condensers using the same base model and training schedule:

– RANDOM PRUNING: randomly selects scene tokens to match the desired budget.

– K-MEANS PRUNING: sets the number of clusters to the desired token budget, clusters the reference-view token features, and retains only the token closest to each cluster centroid.

– LEARNED PRUNING: uses a trained token pruner to select 90% of the token budget, then fills the remaining 10% through feature-diversity-based sampling (Takezoe et al., 2026).

CLIFT-STYLE: uses a K-means-based Transformer condenser adapted from CLIFT (Wang et al., 2025), with the number of blocks increased to eight to match our condenser.

Metrics. We report novel-view PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018) under different reference view counts and scene token budgets. For all quality metrics, we report the mean and 95% confidence intervals (CIs) from 2,000 scene-level resamples with replacement.

## 4.2 COMPARING AGAINST OTHER SYSTEMS

Our base model is strong. To show that we build on a strong highway backbone, we first compare our base model, LAGERNVS-FT, against other systems. Fig. 3 shows the results, where filled markers denote view counts seen during training and hollow markers denote unseen counts. LAGERNVS-FT consistently achieves substantially higher PSNR than all compared systems<sup>1</sup>; SSIM and LPIPS show similar trends (Supplementary Fig. S4). It also improves as more reference views are provided. <sup>2</sup>

GLOD-LVSM achieves the best quality–token trade-offs. We compare the quality–scene-token trade-off with CLIFT because all other systems do not support flexible scene-token counts. Fig. 4 shows the results for different reference views. GLOD-LVSM achieves a better PSNR–tokencount trade-off than CLIFT. The advantage comes from both our strong base model and our better condenser design. SSIM and LPIPS show similar trends (see Supplementary Fig. S3).

## 4.3 COMPARING AGAINST DIFFERENT CONDENSER DESIGNS

To isolate the effect of the strong base model, we conduct controlled experiments using the same base model and training schedule across different condenser designs. Because the baseline condensers do not produce geometry-grounded tokens, applying frustum-culling is not straightforward. For a fair comparison, we disable culling for all methods and report the compressed scene-token count. Fig. 5 compares quality–token trade-offs with 2 and 6 reference views on both datasets; 4- and 10-view results are shown in Fig. 1. Our condenser achieves a better PSNR–token-count trade-off across datasets and reference-view counts. SSIM and LPIPS show similar trends (see Supplementary Figs. S5–S8). We also visualize novel-view predictions from all condensers in Fig. 6, which visually confirms our method’s quality advantage. More visualizations across scenes, reference-view counts, and compression rates are provided in Supplementary Figs. S13–S25.

![](images/d06b05be1f92dd5c68b802ca95cf39f7f5b9825287568e16e6c6de818f004662.jpg)  
Scene tokens

![](images/c97f6b11b3d60f67bc39e4732f4c7056bb02c7b4b346b05e17061fcc9e9288ef.jpg)  
Scene tokens

![](images/891efbebdf9f535afea05d8bab3ccc5bf1d9d5e50e0066c8dfa3497b7bfaccbe.jpg)  
Scene tokens

![](images/8f0f13a296b9ab3b008a3e0e40fad20dda44a4cf92895a4cd1fe155201324547.jpg)  
Scene tokens

Fig. 5: Comparison of condenser quality–token trade-offs using the same base model. Left: RealEstate10K with 2 and 6 reference views. Right: DL3DV with 2 and 6 reference views. The x-axis denotes the pre-culling scene-token count. Shaded bands indicate 95% confidence intervals.  
CLiFT-style  
![](images/7e627f866307699405070375de0e9b385bbc281fefeacacd7cdfaece668d60e5.jpg)  
Fig. 6: Visualization of novel-view predictions from different condensers using two reference views at the same budget of 311 scene tokens (80% reduction), shown on four DL3DV scenes.

## 4.4 TOKEN REDUCTION & RENDERING SPEEDUP OVER THE BASE MODEL

We now evaluate the token reduction and rendering speedup against the base model. Efficiency gains must be considered in the context of rendering quality. Using LAGERNVS-FT as the quality reference, we measure scene-token reduction under PSNR losses of $X \in \{ 2 , 1 , 0 . 5 , 0 \}$ dB. Fig. 7 shows the results. On 10-view RealEstate10K, GLOD-LVSM reduces the scene-token count by 85.5% at a 2 dB quality loss. Even with no quality loss (0 dB), it still reduces the token count by 33.4%, which we attribute to the more effective geometry-grounded representation. The corresponding rendering speedups are 3.00× and 1.23× over LAGERNVS-FT, respectively. With culling, the estimated speedups further increase to 3.20× and 1.55×. More speed results and evaluation detail are provided in Supplementary Fig. S1 and Supplementary A.5.

## 4.5 SCALING QUALITY WITH VIEWS AND BEYOND THE HIGHWAY TOKEN COUNT

We now highlight two properties of GLOD-LVSM. First, unlike highway designs whose scene tokens grow with reference-view count, GLOD-LVSM can exploit more views at a fixed scene-token budget. Fig. 8 shows that, at a fixed token budget, GLOD-LVSM generally improves with more views. For example, on RealEstate10K with a 2048-token budget, 10 views outperform 2 views by 2.8 dB. In contrast, a 10-view highway model requires over 13K tokens. At small capacity, such as 512 tokens, gains saturate/degrade beyond 6 views. See Supplementary Fig. S9 for more results.

Second, because GLOD-LVSM learns a continuous anchor-initialization field (Sec. 3.2), it can sample more anchors (K) than the original highway representation tokens (N), which baseline condensers cannot support. Fig. 9 reports the PSNR gain relative to GLOD-LVSM at the uncompressed budget. Although GLOD-LVSM exceeds the base-model quality at the same token budget, its quality continues to improve for $K > N$ . See Supplementary Fig. S10 for more results.

![](images/56dd31470bb37e7c2caee812460f4ceab65f0377b19c3add463027cdee8edd38.jpg)  
Fig. 7: Scene-token reductions at different PSNR loss budgets.

![](images/fbee5c3775c6b561d941f8f961604e3ef1b3ff6da75b6082b3e43da95090ca82.jpg)  
Fig. 8: Scaling quality with reference views on RealEstate10K.

![](images/0c9d0913a4d64beba2bca370b721aadf8e91806de731b05754fe69b80fec7e29.jpg)  
Fig. 9: Scaling beyond the base-model token budget on DL3DV.

![](images/eac994f5eb0dadbf997e5287dd4178e389b2d71fd8e2f2249a5e68653cd78b12.jpg)  
Fig. 10: DL3DV ablation studies using 6 reference views.

## 4.6 EMERGENT GEOMETRY AND ANCHOR ALLOCATION

To understand GLOD-LVSM, we visualize its representation using four reference views with 2048 scene anchors (Fig. 1). Without explicit geometry supervision, consistent scene depth emerges across reference views, while the learned anchors concentrate near scene surfaces. Projecting the anchors back onto the reference images further reveals scene-adaptive allocation: anchors are denser in texture rich regions and sparser in textureless regions. These results show that the NVS loss can induce an efficient geometry- and texture-aware scene representation. More visualizations are in Supplementary Figs. S26–S31. We process raw depth before point-cloud visualization; see Supplementary A.4.

## 4.7 ABLATION STUDIES

Fig. 10 compares quality–decoder-token trade-offs for our ablations on 6-view DL3DV. OURS denotes GLOD-LVSM without culling. OURS + CULLING enables our frustum-culling (Sec. 3.4). W/O GEO. removes geometry-dependent RoPE (Sec. 3.3.2) from both the condenser and decoder. W/O BOX. removes anti-aliasing from the positional encoding (Eqn. 3). W/O PE. removes positional encoding and instead feeds normalized anchor indices to the field MLP. W/O CONTEXT.<sup>3</sup> removes scene- and view-specific conditioning from the anchor initialization field (Eqn. 2). W. K-MEANS INIT. replaces our end-to-end learned anchor initialization field (Sec. 3.2) with Kmeans centroids of the reference-view tokens (Wang et al., 2025). The results show that each component contributes to the quality–token trade-off of GLOD-LVSM. Results with other referenceview counts show similar trends and are provided in Supplementary Figs. S11–S12.

## 4.8 GLOD-LVSM SUPPORTS SCENE SEMANTIC UNDERSTANDING

We further examine whether the learned anchors capture scene semantics beyond rendering. We use VLM-generated annotations for RealEstate10K (He et al., 2025; Li et al., 2023) to categorize scenes into eight semantic classes. For each scene, GLOD-LVSM compresses 10 reference views into 1,024 anchors, which are mean-pooled into a single 768-dimensional scene vector. We train only a linear probe on this vector for scene classification and achieve 80.40%/91.73% top-1/top-2 accuracy. This result shows that learned anchors already contain scene semantics. See Supplementary A.6 for evaluation protocol and Supplementary Fig. S2 for the confusion matrix.

## 5 CONCLUSION

We propose GLOD-LVSM, a compact, geometry-grounded LVSM representation with continuous levels of detail. GLOD-LVSM achieves a strong quality–token trade-off through two key designs: (1) an end-to-end learned continuous field with minimal handcrafted design, and (2) a condenser that consolidates multi-view observations into scene-centric, geometry-grounded anchors. Trained with the NVS objective, GLOD-LVSM learns geometry- and texture-aware anchor allocation. We further show its utility on scene semantic understanding, highlighting the potential beyond rendering.

## REFERENCES

T. Akenine-Möller, E. Haines, and N. Hoffman. Real-Time Rendering. CRC Press, 2016. ISBN 9781498785631. URL https://books.google.com/books?id=Xcb1CwAAQBAJ.

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5835–5844. IEEE, 2021.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European conference on computer vision, pp. 370–386. Springer, 2024.

Zhigang Cheng, Mingchao Sun, Liu Yu, Zengye Ge, Luyang Tang, Mu Xu, Yangyan Li, and Peng Pan. Clod-gs: Continuous level-of-detail via 3d gaussian splatting. In International Conference on Learning Representations, volume 2026, pp. 89275–89299, 2026.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International conference on learning representations, volume 2024, pp. 2632–2652, 2024.

Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, and Zhangyang Wang. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37:140138–140158, 2024.

Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European conference on computer vision, pp. 165–181. Springer, 2024.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for video diffusion models. In International Conference on Learning Representations, volume 2025, pp. 100433–100464, 2025.

Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=QQBPWtvtcn.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Evan Kim, Hyunwoo Ryu, Thomas W Mitchel, and Vincent Sitzmann. Scaling view synthesis transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28893–28902, 2026.

Johannes Kopf, Michael F Cohen, Dani Lischinski, and Matt Uyttendaele. Joint bilateral upsampling. ACM Transactions on Graphics (ToG), 26(3):96–es, 2007.

Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library for language-vision intelligence. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 31– 41, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https: //aclanthology.org/2023.acl-demo.3.

Weikai Lin, Yu Feng, and Yuhao Zhu. Metasapiens: Real-time neural rendering with efficiencyaware pruning and accelerated foveated rendering. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pp. 669–682, 2025.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22160–22169. IEEE, 2024.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

Nicholas Milef, Dario Seyb, Todd Keeler, Thu Nguyen-Phuoc, A Božic, Sushant Kondguli, andˇ Carl Marshall. Learning fast 3d gaussian splatting rendering using continuous level of detail. In Computer Graphics Forum, volume 44, pp. e70069. Wiley Online Library, 2025.

Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Luciˇ c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation transformer:´ Geometry-free novel view synthesis through set-latent scene representations. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6219–6228. IEEE, 2022.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Cheng Sun, Jaesung Choe, Min-Hung Chen, Ryo Hachiuma, and Yu-Chiang Frank Wang. Dvsm: Decoder-only view synthesis model done right. arXiv preprint arXiv:2605.29891, 2026.

Stanislaw Szymanowicz, Minghao Chen, Jianyuan Wang, Christian Rupprecht, and Andrea Vedaldi. Lagernvs: Latent geometry for fully neural real-time novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

Rinyoichi Takezoe, Yaqian Li, Zi-Hao Bo, Anzhou Hou, Mo Guang, and Kaiwen Long. Learnpruner: Rethinking attention-based token pruning in vision language models. In International Conference on Learning Representations, volume 2026, pp. 66381–66400, 2026.

Zhengqing Wang, Yuefan Wu, Jiacheng Chen, Fuyang Zhang, and Yasutaka Furukawa. Clift: Compressive light-field tokens for compute efficient and adaptive neural rendering. Advances in Neural Information Processing Systems, 38:8759–8787, 2025.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Yu Wu, Minsik Jeon, Jen-Hao Rick Chang, Oncel Tuzel, and Shubham Tulsiani. Rayrope: Projective ray positional encoding for multi-view attention. In European Conference on Computer Vision (ECCV), 2026.

Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16453–16463. IEEE, 2025.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pp. 586–595. IEEE, 2018.

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018.

## GLOD-LVSM: Supplementary Material

## A IMPLEMENTATION AND EVALUATION DETAILS

## A.1 CLOSED-FORM BOX-FILTERED FREQUENCY-BAND ENCODING

For query i in view $v ,$ let $a = u _ { v , i } , b = u _ { v , i } + \Delta _ { v } ,$ and $\Delta _ { v } = b - a .$ . A normalized box filter averages each sinusoidal component over this interval. Using the antiderivatives of cosine and sine, we obtain

$$
\begin{array} { l } { { \displaystyle \bar { C } _ { j } = \frac { 1 } { \Delta \nu } \int _ { a } ^ { b } \cos ( 2 \pi f _ { j } u ) d u = \left[ \frac { \sin ( 2 \pi f _ { j } u ) } { 2 \pi f _ { j } \Delta _ { v } } \right] _ { a } ^ { b } } } \\ { { \displaystyle ~ = \frac { \sin ( 2 \pi f _ { j } b ) - \sin ( 2 \pi f _ { j } a ) } { 2 \pi f _ { j } \Delta _ { v } } } , } \\ { { \displaystyle \bar { S } _ { j } = \frac { 1 } { \Delta \nu } \int _ { a } ^ { b } \sin ( 2 \pi f _ { j } u ) d u = \left[ - \frac { \cos ( 2 \pi f _ { j } u ) } { 2 \pi f _ { j } \Delta _ { v } } \right] _ { a } ^ { b } } } \\ { { \displaystyle ~ = \frac { \cos ( 2 \pi f _ { j } a ) - \cos ( 2 \pi f _ { j } b ) } { 2 \pi f _ { j } \Delta _ { v } } . } } \end{array}\tag{S1}
$$

Concatenating these coefficients and dividing by $\sqrt { F }$ gives $\bar { \gamma } _ { v , i }$

## A.2 REFERENCE-VIEW SELECTION BEYOND OFFICIAL SETTINGS

RealEstate10K. When we require more reference views beyond the official dataset settings, we retain the official reference views and incrementally add frames using farthest-point sampling in the temporal domain, which reduces redundancy among selected views. Let R denote the current set of selected reference-frame indices and C the candidate frames from the source video, which exclude the target views and the selected reference views. At each step, we select

$$
f ^ { * } = \arg \operatorname* { m a x } _ { f \in { \mathcal { C } } \setminus { \mathcal { R } } } \operatorname* { m i n } _ { r \in { \mathcal { R } } } | f - r | ,
$$

This process is repeated until the desired number of reference views is reached.

DL3DV. We use the frame indices provided by LagerNVS (Szymanowicz et al., 2026) to define each 50-frame window. Within each 50-frame window, we first use temporal farthest-point sampling to meet the desired number of reference views beyond the official setting. We then use the remaining frames in the window as target views, so V reference views leave 50 − V target views.

Fig. S9 instead uses a fixed target set across all reference-view counts for DL3DV. Under the original DL3DV protocol, the target set is defined as the complement of the reference views and therefore changes with view count. To enable direct quality comparison when scaling views, we use the same 40 targets per scene from the 10-view setting for all DL3DV results in this figure. This set is a subset of the original targets at every reference-view count.

## A.3 TRAINING DETAILS

We follow the 1–10-view training recipe of the LagerNVS general model (Szymanowicz et al., 2026), with a few modifications.

LagerNVS fine-tuning. We fine-tune LagerNVS for 100K iterations with a 1.2K-step warmup, using learning rates of $\bar { 1 0 } ^ { - 4 }$ on RealEstate10K and $1 0 ^ { - 5 }$ on DL3DV. We follow its dynamic batching strategy with a base batch size of 128, which decreases the batch size for larger reference-view counts.

GLOD-LVSM training. We train GLOD-LVSM for 50K iterations with a 1K-step warmup, using a learning rate of $1 0 ^ { - 4 }$ and a base batch size of 64. For the culling-aware model, we further fine-tune for 25K iterations with a 1K-step warmup, using a learning rate of $1 0 ^ { - 5 }$ and the same base batch size.

![](images/46af6f5b392595ed565d4ebc8cc52a334d5672b0223646b36f506955d5fb21cd.jpg)  
Fig. S1: Rendering FPS at 10 reference views on an A100 80GB with BF16. Left: RealEstate10K; right: DL3DV. Annotations show FPS and speedup over LAGERNVS-FT. For OURS + CULLING, FPS is estimated by measuring rendering speed at the average post-culling decoder-token count over the test set.

## A.4 DEPTH DENOISING FOR POINT-CLOUD VISUALIZATION

To visualize the noisy predicted reference-view depth as a point cloud, we apply ${ \mathrm { ~ a ~ 3 ~ } } \times { \mathrm { ~ 3 ~ } }$ median filter to the patch-level depth and upsample it to image resolution using joint bilateral upsampling (Kopf et al., 2007) guided by the corresponding RGB image. We then back-project the filtered depth using the camera intrinsics, transform the points into a shared scene coordinate system using the camera poses, and color them with RGB values. This post-processing is used only for point-cloud visualization; the displayed depth maps are raw predictions, and the model geometry, anchors, and rendering remain unchanged.

## A.5 RENDERING-SPEED EVALUATION

We report results for GLOD-LVSM both with and without culling in Fig. S1. We evaluate rendering speed on an NVIDIA A100 80GB GPU using BF16 with a batch size of 1 and 1 target view. For the baseline and our no-culling model, we directly benchmark rendering latency at fixed scene-token counts. For the culling-enabled model, the effective decoder-token count varies across scenes and target poses. We therefore estimate its speed by measuring rendering speed at the average post-culling decoder-token count over the test set. The estimated rendering latency includes the measured culling overhead.

## A.6 SCENE SEMANTIC UNDERSTANDING EVALUATION PROTOCOLS

We first test scene-semantic understanding using VLM-generated annotations for RealEstate10K (He et al., 2025; Li et al., 2023) to categorize scenes into eight semantic classes: living room, bedroom, kitchen, bathroom, dining room, hallway, staircase, and outdoor. Each scene receives one class derived from its caption. We filter out scenes that belong to multiple classes or do not belong to any of the predefined classes. This results in 40,857 training scenes and 4,606 validation scenes. We train for 500 epochs with a learning rate of $3 \times 1 0 ^ { - 2 }$ , batch size 256, and weight decay $1 0 ^ { - 4 }$ . We use cosine annealing to decay the learning rate to zero over training.

## B MORE RESULTS

Our Base Model is Strong We provide more results for Sec. 4.2, including PSNR, SSIM, and LPIPS comparisons with 2–16 reference views on both datasets (Fig. S4).

![](images/19bdcbabd3aa9eb37acf3cab3fe44127198446da158f10cae52b077061dc2f7b.jpg)  
Fig. S2: Row-normalized confusion (%) for our scene classification. Top-1/2 accuracy: 80.40/91.73% on 4,606 validation scenes.

GLOD-LVSM Achieves the Best Quality–Token-Count Trade-offs We provide more results for Sec. 4.2, including PSNR, SSIM, and LPIPS versus scene-token counts for GLOD-LVSM and CLIFT with 2, 4, 6, and 10 reference views on both datasets (Fig. S3).

Comparing against Different Condenser Designs We provide more results for Sec. 4.3, including PSNR, SSIM, and LPIPS comparisons across condenser designs with 2, 4, 6, and 10 reference views on both datasets (Figs. S5–S8). All methods use the same base model with culling disabled.

Scene-Token Reductions & Rendering Speedup at Fixed Quality Target. We provide more results for Sec. 4.4, including rendering FPS and speedups at 0, 0.5, 1, and 2 dB quality-loss budgets on both datasets with 10 reference views (Fig. S1). The evaluation protocol is described in A.5.

Scaling Quality with Reference Views and Token Budget We provide more results for Sec. 4.5, including PSNR, SSIM, and LPIPS on both datasets for increasing reference-view counts at fixed budgets of 512, 1,024, and 2,048 tokens (Fig. S9) and increasing anchor budgets from 100% to 150% of the input token count (Fig. S10). Culling is disabled in both experiments. For DL3DV with two reference views, the 2,048-anchor setting exceeds the corresponding highway token count of 1554. This is enabled by our continuous anchor field (Sec. 3.2), which can sample an arbitrary number of anchors beyond the input token count.

Ablation Studies We provide more results for Sec. 4.7, including PSNR, SSIM, and LPIPS for the same component ablations with 2, 4, 6, and 10 reference views on DL3DV (Figs. S11–S12).

GLOD-LVSM as a General Scene Representation We provide scene-classification confusion matrix (Fig. S2) for Sec. 4.8. The downstream evaluation protocols are given in A.6.

Visual Comparison across Different Condensers. We provide more results for Sec. 4.3, including 34 selected novel-view examples with 2 and 10 reference views at 0%, 20%, 40%, 60%, and 80% token reduction on both datasets (Figs. S13–S25). For compact figure files, we JPEG-compress the full images at quality 88 without chroma subsampling (4:4:4), while keeping the zoom-in crops losslessly encoded.

Qualitative Analysis of Emergent Geometry and Anchor Allocation We provide more results for Sec. 4.6, including emergent depth, point-cloud, 3D anchors, and projected-anchor visualizations for 12 RealEstate10K scenes and 15 DL3DV scenes using four reference views and 2,048 anchors (Figs. S26–S31).

![](images/c95d8898a5c01023e46b9b472a9cdcdedd6c61f80bd46947f4d9116a4ad77b4f.jpg)  
Fig. S3: System-level quality versus scene tokens. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Colors distinguish reference-view counts; solid lines denote GLOD LVSM and dashed lines denote official CLiFT. Shaded bands show 95% CIs.

![](images/2a03f4f3302a1546430e13772029d99bbf5ad88bc4ee79fa42f53f30ccd1cbb1.jpg)  
Fig. S4: Base-model comparison without our condenser. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% CIs. No official DL3DV checkpoint is available for LVSM.

![](images/0ae33128f6b02c3baedef4a983cd888bf96515372d8bed7c34bfca0f43db3c35.jpg)

![](images/8a7ff9a98531081ebf946f7a0ed3431880e8450a09ac9034c4e3661d6acad7e1.jpg)

![](images/19f2e95b8649d5207137c217a190b3a0aace0d972baf29141b9bafaf947811b7.jpg)

![](images/1069beb6af0149a90d37ead365693ccaa796d9111b3629c6471eabc042659762.jpg)

![](images/a27cb3e593b7f3beed93a80829f72c647ce2190e27792920cccc7f66a488e115.jpg)  
Fig. S5: Condenser quality–token trade-offs with 2 reference views and culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% CIs.  
Fig. S6: Condenser quality–token trade-offs with 4 reference views and culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% CIs.

![](images/e090f007b35225177d419b01051318e2804886ddfd6ad1273e88f864a2722695.jpg)  
Fig. S7: Condenser quality–token trade-offs with 6 reference views and culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% CIs.

![](images/3483c70841538a615f01c21a3e2105df9ce4c8fc29bf0227c2df6fa90fb29a7f.jpg)  
Fig. S8: Condenser quality–token trade-offs with 10 reference views and culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% CIs.

![](images/3b4d018f36f224ead9bf04a8fa23b7de1f92b142036c17f6bd98c4c0c689f474.jpg)  
Fig. S9: Fixed-budget view scaling with culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% scene-bootstrap CIs.<sup>a</sup>

<sup>a</sup>Under the original DL3DV protocol, the target set is the complement of the reference views and therefore changes with view count. To make scaling directly comparable, all DL3DV results in this figure use the same 40 targets per scene from the 10-view setting, a subset of the original target set at every reference-view count.

![](images/a6ad2090b328afbe6f4024d55e9113ce6b08edfcaa25f184f72d5bfa4085864e.jpg)  
Fig. S10: Paired changes relative to K = N with culling disabled. Top: RealEstate10K; bottom: DL3DV. Columns show ∆PSNR, ∆SSIM, and ∆LPIPS. Shaded bands show paired 95% scenebootstrap intervals. Negative ∆LPIPS indicates improvement.

![](images/6e6941424190d0f43bfea8ae2a2a930dca9a8a94ccb0a44eae74b8dbed97953a.jpg)  
Fig. S11: DL3DV ablations with 2 reference views (top) and 4 reference views (bottom). Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% scene-bootstrap CIs.

![](images/cd9c494836d48557973b2f7713ae9562e70f9e8bf64f830e64615e3cfcc4ed6a.jpg)  
Fig. S12: DL3DV ablations with 6 reference views (top) and 10 reference views (bottom). Columns show PSNR, SSIM, and LPIPS. Shaded bands show 95% scene-bootstrap CIs.

![](images/68d0475c3cb27f55a0bfe6cc388c607225654b1ed6cb67accf2f94c6105102ef.jpg)  
Fig. S13: Visual comparison of different condensers on DL3DV.

![](images/d23c4a9a9d247422899141e3ab80ba429bb542b1499a5159c3192a037a0c1572.jpg)  
Fig. S14: Visual comparison of different condensers on DL3DV.

![](images/bb150b131298978fae815686ad5557334b652f3811b7108caf656ce37736ece6.jpg)  
Fig. S15: Visual comparison of different condensers on DL3DV.

![](images/0a91c33e0fe75f63003ca94b5d2ece0b31032f016a7a75f69d41dd81b91bddf9.jpg)  
Fig. S16: Visual comparison of different condensers on DL3DV.

![](images/5947f2c08e61125a0c7238d2f5f11752a598121b2d1a3bbc9d9bfe90f3cd047a.jpg)  
Fig. S17: Visual comparison of different condensers on DL3DV.

![](images/dcf26b988db8d0ede847e9d1059a283a95cf92fd22528e64d887c0e474396386.jpg)  
Fig. S18: Visual comparison of different condensers on DL3DV.

![](images/b38df949f8b0f49fcbb1ebd47d1c5cbec3007abdb0e3a8b8a5a7c9858d9635fa.jpg)  
Fig. S19: Visual comparison of different condensers on DL3DV.

![](images/2ba808273670b449b837463972d8d47487406b3ca8df554d992a701f619dd290.jpg)  
Fig. S20: Visual comparison of different condensers on DL3DV.

![](images/c94ba8b36b7917d3d92c90ebdd9b1b059454a53ff02e004df9b903e798903d28.jpg)  
Fig. S21: Visual comparison of different condensers on RealEstate10K.

![](images/1f13cae61083318befc1d814261d17220a11e7fec8339590f43c222374b3d968.jpg)  
Fig. S22: Visual comparison of different condensers on RealEstate10K.

![](images/9a7410bcd84f7e19d3a04d9aa4ccbe5edf4eead309e7ea039c9f4ec416db2c1f.jpg)  
Fig. S23: Visual comparison of different condensers on RealEstate10K.

![](images/0e51f231603069a51d0131ba8a7d66b2d2fb5c899696aadb432cbf7442c34f8b.jpg)  
Fig. S24: Visual comparison of different condensers on RealEstate10K.

![](images/b1d4f2059b54fae09586bfbd1183cc13767bbb607cf8a38ff7f1741adc8313d6.jpg)  
Fig. S25: Visual comparison of different condensers on RealEstate10K.

Multi-view Point Cloud + Anchors

Projected Anchors & Density  
![](images/9028c6a3b7a45e43f52647ce1e354adbfa4d96ceab8ba197715bde614b943a6b.jpg)  
Fig. S26: Emergent geometry and geometry- and scene-complexity-aware anchor placement on RealEstate10K.

Multi-view Point Cloud + Anchors

Projected Anchors & Density  
![](images/1ebadab8dfd82865f85833c74abac11b86b6e4ef44780121a4386efff3ad1109.jpg)  
Fig. S27: Emergent geometry and geometry- and scene-complexity-aware anchor placement on RealEstate10K.

Projected Anchors & Density  
![](images/c6d9fb83738ed423c0cd8866402b53d422834b515e35581dffbffb87dea60dd6.jpg)  
Fig. S28: Emergent geometry and geometry- and scene-complexity-aware anchor placement on RealEstate10K.

![](images/57cbe9f0e5c05602f94f9b3b935c8e82be459153cd37c203bd1d4a714fb2b782.jpg)  
Fig. S29: Emergent geometry and geometry- and scene-complexity-aware anchor placement on DL3DV.

Projected Anchors & Density

![](images/c8b0fa33076e4d6db50b8cd88a5f01299faabe25eae72d43d1dc5ccf51237c78.jpg)  
Fig. S30: Emergent geometry and geometry- and scene-complexity-aware anchor placement on DL3DV.

![](images/b4d1ff71ef47d113351148dca9eb3c9c6925257e9629f77825b283a0eaec220e.jpg)  
Fig. S31: Emergent geometry and geometry- and scene-complexity-aware anchor placement on DL3DV.