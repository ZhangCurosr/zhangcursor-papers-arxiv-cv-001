# Data-eficient crack quantification in lithium-ion cathodes using foundation model transfer

Thorsten Tegetmeyer-Kleine<sup>a,b,c,∗</sup>, Thomas Schmitt<sup>g,h</sup>, Phillip Aquino<sup>f</sup>, Christiane Rahe<sup>b,c,d</sup>, Dirk Uwe Sauer<sup>b,c,d,e</sup>, Weihan Li<sup>a,b,c,∗</sup>

<sup>a</sup>Junior Professorship for Artificial Intelligence and Digitalization for Batteries, Institute for Power Electronics and Electrical Drives (ISEA), RWTH Aachen

<sup>b</sup>Center for Aging, Reliability and Lifetime Prediction of Electrochemical and Power Electronic Systems (CARL), RWTH Aachen University, Campus-Boulevard 89, Aachen, 52074, Germany

<sup>c</sup>Juelich Aachen Research Alliance, JARA-Energy, Templergraben 55, Aachen, 52056, Germany

<sup>d</sup>Chair for Electrochemical Energy Conversion and Storage Systems, Institute for Power Electronics and Electrical Drives (ISEA), RWTH Aachen University, Campus-Boulevard 89, Aachen, 52074, Germany

<sup>e</sup>Helmholtz Institute Muenster (HI MS), IMD-4, Forschungszentrum Juelich, Germany <sup>f</sup>Honda Research Institute USA, Inc., 21001 State Route 739, Raymond, OH, 43067, United States <sup>g</sup>Honda Research Institute Europe GmbH, Carl-Legien-Strasse 30, Ofenbach/Main, 63073, Germany <sup>h</sup>Zenules GmbH, Landgraf-Georg-Str. 4, Darmstadt, 64283, Germany

## Abstract

Battery lifetime is central to sustainable electrification, yet the particle crack ing that drives lithium-ion cathode aging is hard to measure: quantitative microscopy of this degradation is bottlenecked by annotation, because each destructive electron-microscopy cross-section spans hundreds of megapixels and pixel-level expert labelling requires hours per image. We show that a frozen self-supervised vision-transformer encoder, combined with a lightweight trainable decoder and iterative model-assisted annotation, turns this sparse labelling budget into population-scale degradation measurements. Applied to three 120- megapixel NMC cathode cross-sections representing initial, cycled-aged and calendar-aged states, the framework distinguishes intragranular cracks from early- and late-stage intergranular cracks and yields per-particle distributions of crack width, tortuosity and area fraction. Late intergranular crack coverage reaches 4.6% in the cycled sample versus 0.5% in the initial and calendaraged samples, forming more tortuous, higher-coverage networks, consistent with degradation from repeated electrochemical cycling rather than elevated-temperat

storage alone. A single destructive image yields the population-level statistics needed for lifetime-extending design, aging assessment and second-life decisions.

Keywords: data-eficient learning, foundation models, quantitative microscopy, crack morphometrics, NMC cathodes, model-assisted annotation

## 1. Introduction

Lithium-ion batteries are central to electrified transport and renewableenergy integration, yet electrochemical and chemo-mechanical degradation processes within their electrodes ultimately limit service life [1, 2]. Quantifying the mechanical cracking that accompanies this degradation is bottlenecked by destructive cross-sectional microscopy and labour-intensive expert annotation. In high-nickel layered oxide NMC cathodes, this degradation appears primarily as particle-level cracking: secondary particles composed of aggregated primary grains develop cracks that propagate along grain boundaries (intergranular) and through individual grains (intragranular) during cycling, calendar aging, and thermal stress [3]. These cracks expose fresh cathode surfaces to the electrolyte, catalyzing parasitic reactions, promoting cathode-electrolyte interphase growth, and accelerating transition-metal dissolution, collectively driving capacity fade and impedance rise [4, 5]. As cracking intensifies, secondary particles fragment entirely, causing loss of electronic contact and further capacity loss [6, 7]. Understanding crack initiation, propagation, and morphology is therefore central to diagnosing battery aging and guiding electrode design.

To quantify cracking reliably, one must first distinguish the underlying crack morphologies. Intergranular cracks propagate along grain boundaries between primary crystallites within polycrystalline secondary particles, yielding a fragmented, mosaic-like appearance, whereas intragranular cracks traverse individual grains and often align with the layered structure, running parallel to the (003) crystallographic planes [8]. Intragranular cracks are typically narrow (on the order of tens of nanometers) and may originate at dislocation cores before extending along mechanically weak lattice planes [9]. Under high-voltage operation, this intragranular mode is associated with pronounced surface reactivity, oxygen release, and the formation of a rock-salt surface layer in Ni-rich cathodes [10, 11]. Scanning electron microscopy (SEM) and X-ray microscopy can reveal such cracks ex situ, after cell disassembly and sample preparation (direct in situ imaging at comparable spatial resolution and contrast inside sealed cells is typically impractical). However, translating image contrast into quantitative, class-aware crack maps remains labor-intensive: large SEM images (e.g., $1 7 k \times 7 k ~ \mathrm { p x } )$ must be subdivided into tiles. In our experience, manual labelling can take up to 5 min per tile, amounting to several hours per image and introducing substantial inter-annotator variability [12, 13, 14].

Current imaging methods cannot simultaneously achieve high spatial resolution and track crack evolution over time. High-fidelity techniques such as FIB-SEM or nano-CT resolve fine microstructural details but are destructive, precluding subsequent cycling of the same specimen. Non-destructive modalities such as laboratory micro-CT enable repeated imaging over cycling but typically lack the spatial resolution needed to resolve small cracks. This tradeof motivates approaches that use information from destructive measurements to interpret non-destructive data.

Foundation-model transfer is well suited to this setting. Self-supervised vision-transformer encoders pretrained on natural-image corpora produce transferable mid-level features without domain-specific pretraining; recent encoders [15] preserve dense per-patch correlations and remain efective with a frozen backbone and a lightweight task-specific head [16]. In low-label scientific imaging, where expert annotation is the bottleneck, freezing the encoder concentrates the annotation budget on per-particle morphometric quantification rather than on encoder fine-tuning. The approach therefore fits destructive-microscopy domains in which each annotated tile is costly and the downstream question is morphometric rather than segmentation-overlap based.

The three crack classes correspond to morphological categories associated in the literature with distinct degradation mechanisms in NMC cathodes. Intragranular cracking has been linked to anisotropic lattice strain within primary particles during the high-state-of-charge H2→H3 phase transition, often aligning with the lamellar (003) structure of the layered oxide [3, 10]. Early intergranular cracking propagates along grain boundaries under repeated mechanical stress from volume changes during cycling, while late intergranular cracking represents the developed network state; wider crack openings infiltrated by electrolyte decomposition products and the rock-salt surface layer characteristic of Ni-rich degradation [11, 9]. Resolving these three regimes from a single SEM cross-section therefore captures the chemo-mechanical degradation history at the particle level and supports mechanistic interpretation beyond the binary “cracked / not cracked” framing of single-class segmentation.

Deep learning-based segmentation, including variational autoencoder (VAE)- style and U-Net approaches, has accelerated automated analysis of battery microstructures, and recent lithium-plating work suggests that latent representations from autoencoder-style models can encode diagnostically relevant post-mortem image features [17]. By contrast, foundation-model encoders pretrained at scale provide richer and more transferable representations for downstream dense prediction [18, 15, 19]. However, across the key studies cited here, most contributions either target a single crack morphology or limited volumes (e.g., sub-volume analysis [20]), or they primarily advance general segmentation strategies rather than broad, multi-class crack mapping (e.g., multi-phase 3D segmentation [21], general-purpose segmentation frameworks [22, 23], or foundational encoder-decoder architectures [24, 25]). Multi-class crack typing across hundred-megapixel SEM images therefore remains an open problem, and foundation-model transfer pipelines for multi-class segmentation at this resolution in electron microscopy have not been reported. Recent foundation models for segmentation, including the Segment Anything Model (SAM [26]) and its successor SAM-2 [27], as well as CellPose [28] for biological microscopy, offer zero- or few-shot segmentation via interactive prompting. However, these models do not natively support class-discriminative multi-label prediction for semantically distinct sub-structures (e.g., early vs. late intergranular cracks), and their applicability to narrow, high-aspect-ratio structures in electron microscopy without task-specific fine-tuning remains limited.

Here we introduce a sparse-label morphometric-inference framework that turns frozen foundation-model representations into per-particle crack morphometrics from few expert annotations, making population-scale degradation diagnostics practical for battery-lifetime research. We freeze a self-supervised vision-transformer encoder [15] and train only a lightweight decoder [29] on the SEM labels. An iterative model-assisted annotation loop grows 79 hand-labelled SEM tiles into a 482-tile expert-corrected reference set across hundred-megapixel cathode cross-sections at three aging states. We resolve intragranular cracking and two stages of intergranular cracking, three categories linked in the literature to distinct degradation mechanisms, and read per-particle width, tortuosity and area fraction from the resulting class maps. Cycled-aged NMC cathodes show an approximately ninefold increase in late intergranular cracking, accompanied by more tortuous, higher-coverage crack networks, whereas calendar-aged samples remain near baseline. These morphometric diferences are consistent with degradation associated with repeated cycling rather than elevated-temperature storage alone. Each cross-section yields hundreds of per-particle morphometric measurements, so the scientific findings come from these per-particle distributions rather than from pixel-overlap metrics.

## 2. Results

We analysed three SEM cross-section images of NMC cathodes representing distinct aging conditions: initial, cycled-aged, and calendar-aged. Each image spans approximately 17k×7k pixels (382×156 µm at 22.3 nm/px). Panels a–c of Fig. 1 show representative regions from the three cross-sections and illustrate the condition-dependent diferences in SEM contrast, brightness, and crack density. Pixel-level annotations define three crack classes: early intergranular (narrow boundary cracks at initial propagation stages, typically <120 nm width), late intergranular (wider, more developed boundary cracks, typically >150 nm width), and intragranular (cracks traversing individual grains). The distinction between early and late intergranular cracks is temporal; panel d of Fig. 1 summarises the crack taxonomy used throughout the study. In cross-sectional SEM images this progression manifests as crack-width diferences, which were assigned by expert visual assessment. Panel f of Fig. 1 shows the 1024×1024 px tiling strategy with 50% overlap used for seamless image reconstruction and image-scale inference. All pixel-level annotations originate from a single expert annotator; systematic biases (e.g., preferential labelling of high-contrast late intergranular cracks over subtle early or intragranular cracks) may afect per-class performance.

## Multi-class segmentation performance

We trained the multi-class model using the iterative annotation strategy in Fig. 2b, with the dataset composition shown in Fig. 1e. An initial model was trained on 79 expert-labelled seed tiles (1024 × 1024 px each) sampled from the cycled-aged image, achieving a mean Dice score of 0.49. This model was then applied to tiles from all three SEM images to generate provisional pixel-wise crack-class masks, which were expert-corrected and added to the training set. The expanded dataset comprised 482 expert-corrected tiles: 79 fully manual seed tiles (all from the cycled-aged image) and 403 model-assisted tiles distributed across the three SEM cross-sections (160 cycled-aged, 81 calendar-aged, 162 initial), yielding per-sample totals of 239 (cycled-aged), 81 (calendar-aged), and 162 (initial). Retraining on this dataset increased the mean Dice score to 0.58 (Fig. 2d), an 18% relative gain from the 79-tile seed model that expanded coverage to all three aging conditions.

a  
![](images/8cb3d77ae8d9d5ca500840d7d672cc01b6eb28f5bc6b8fd0345587e9c99cea6f.jpg)

b  
![](images/5fb350e7f170d01571c209a684cad63e56b9bec3af5243efbd44b7be5ac3a869.jpg)

c  
![](images/ef39457c012e156284346e43e4c02767640ef62acb58a1015db1aec54939cda8.jpg)

d  
![](images/bdcfda9162022800b31fef715b720b809d020cba52500d8825953c846160ce92.jpg)

e  
![](images/c443afbfe77cc6ac6b84c5dd260d5ed0ad412e14d25cee7c8f3b9dd62625e020.jpg)

f  
![](images/3f69ca7d5c5cb74ecf1326883be5393c194563e5f9b83b568a9a955f9a7aca18.jpg)  
Figure 1: Cathode degradation and multi-class crack dataset. (a–c) Representative regions from raw SEM images (ca. 17k × 7k px, 22.3 nm/px) of NMC cathode cross-sections: (a) initial, (b) cycled-aged, and (c) calendar-aged. The three samples exhibit visibly diferent SEM contrast, brightness, and crack density. (d) Crack classification scheme: intragranular cracks (within primary grains), early intergranular cracks (typically <120 nm at grain boundaries), and late intergranular cracks (typically >150 nm and often widened with high-contrast deposits). (e) Dataset composition: 482 expert-corrected tiles distributed as 79 fully manual seed tiles (all drawn from the cycled-aged image) plus 403 model-assisted tiles (160 cycledaged, 81 calendar-aged, 162 initial), across three samples and crack classes. (f) Tiling strategy with 1024×1024 px windows and 50% overlap for seamless image reconstruction.

Trainable parameters: \~17.8M Total: \~103.5 M (85.7 M frozen ViT-B/16 + 17.8 M decoder)  
a  
![](images/ed2c384ec6c18e74ee9ae0e86a8cc0780ad8c7d5e2a30771dde2cd776f8f7621.jpg)

![](images/5610f56ff0f2db555332ea2ba8cc14bb51b5342e844c17639d105e31090f7f0b.jpg)

![](images/5ee4c7d021242b5249a1291212203515c6a1a3c36d087c13db993b669b780882.jpg)

![](images/34b0a0312477a9a2ea5b213f859353d71341c7eedf1373dd4fd552c9acb30217.jpg)

b  
![](images/3bd842f1fd7cf7e0ec7d5a5cfd8edec1d6961fa6838f66ac9e012a3eaf87f516.jpg)

c  
![](images/e334cc148fee0a7f0587485d06353bf8fef372f3d0c493aaf5aed676d8dfbce2.jpg)

![](images/d63652faeaaaaaf8bc716e41e78e7650fae1410e5ed0403a10676ae9c1c2558a.jpg)

![](images/2d43d08c512a93d5929a460b48c45219c44c17263a1cd8b95f612495c095e4b3.jpg)

![](images/5bcb30c0cda3299baec727891bbf27f93244914cdea40c7496d469a3128cdbcf.jpg)

d  
![](images/ad3dd0cddb69316c7c3e924a88a7ce982c0707ffd104fb790a7b3365194d31f7.jpg)

e  
![](images/e2e90dcf5843f32bb107f97da457abe960f04fa9a6a35b552b12fee7f0e62e29.jpg)

![](images/472a959f46252196c5751b2077f784c96be2af6b9fd02235fa49b940b13f9e9f.jpg)

f  
![](images/842e6ad3709f5a0eb6d7a8e3b82e03510337fef12564a9a6158ccaed59635c69.jpg)

![](images/ee4279b8f2854e3fc3b1e9d6f4e90b3ecd8e187f85c8aedbc42ed7e7f2e40f96.jpg)  
Figure 2: Foundation model architecture and model-assisted annotation workflow. (a) Encoder-decoder architecture combining a frozen DINOv3 ViT-B/16 backbone with a trainable DeepLabV3+ decoder for 3-class semantic segmentation. The frozen encoder supplies dense per-patch features while the lightweight decoder adapts to the domain with minimal annotated data and reduced training compute. (b) Model-assisted labelling workflow: 79 expert hand-annotated tiles (cycled-aged image) train an initial model. The model then generates provisional pixel-wise crack-class masks for 403 additional tiles (160 cycled-aged, 81 calendar-aged, 162 initial), which are expert-corrected and added back into training to yield the final model. (c) Image-scale inference: sliding windows with 50% overlap produce per-tile 3-class sigmoid predictions (intragranular, early intergranular, late intergranular), stitched by visit-count normalisation to eliminate boundary artefacts. (d) Mean-Dice comparison between the seed model (79 manual tiles, 0.49) and the final model (482 expert-corrected tiles, 0.58), an 18% relative improvement $( \Delta = + 0 . 0 9 )$ . (e) Representative output linking segmentation to geometric quantification: raw SEM ROI, predicted late-intergranular mask overlay, and combined distance- and width-transform overlay (full ROI walkthrough in Fig. S1). (f) Perclass morphometric summary previewing the full analysis in Fig. 4: kernel density estimate of mean crack width and tortuosity distributions for intragranular (I), early intergranular (EI), and late intergranular (LI) cracks.

a  
![](images/5746c95706c196c22d3e8fcf8e0c97edee4d766770a7c33258450407b1c5fe0c.jpg)

b  
![](images/36b81c297e749dd09aea1ee59b1cdfd9f4640724a17cf306d70bd38eff592ba3.jpg)

c  
![](images/ae4e354a6b36e192d971878af8fb00fb986811ec3d73a63b73edcd65dba7d285.jpg)

![](images/a057a1cd9755391d8680a6db990c042af491e729fd636250729a11bee5d2f4c5.jpg)

![](images/f08410985c383d4cf01b628ac23a8496d21f2181f0c3fe961b5b0b688f933d91.jpg)

![](images/86113c57e1d019d57faf47959ec5eb2dec6fb33c580d8b4b14154f71d433219d.jpg)

e  
![](images/c18ff53cbb1d39b984243b7d0265fb3156e531a597e300aee137cda1d19f3aab.jpg)  
Figure 3: Multi-class segmentation performance. (a–c) Representative segmentation results for each sample condition showing input SEM images and model predictions (overlay). Despite the visible contrast and brightness diferences among the three SEM images, the model detects the crack classes in all conditions. (d) Normalized confusion matrix aggregated across the expert-corrected tile set (n=482 tiles; threshold τ=0.1). (e) Per-class segmentation metrics. The model achieves Dice scores of 0.575 (intragranular), 0.513 (early intergranular), and 0.653 (late intergranular) (mean Dice 0.580; mean clDice 0.587).

We froze a self-supervised vision-transformer encoder (facebook/dinov3- vitb16-pretrain-lvd1689m) and trained only a lightweight decoder (Fig. 2a). This reduced GPU memory and compute by backpropagating only through the decoder while using transferable self-supervised features [16]. Image-scale inference then followed the sliding-window visit-normalised stitching scheme shown in Fig. 2c.

On the expert-corrected tile set (n=482; τ=0.1), the final model achieved a mean Dice of 0.580 across the three classes (intragranular: 0.575, early intergranular: 0.513, late intergranular: 0.653; mean precision: 0.729, mean recall: 0.485; Fig. 3e). Representative outputs and the aggregated confusion matrix (Fig. 3a–d) show that the framework resolves all crack classes across the initial, cycled-aged and calendar-aged SEM cross-sections despite substantial contrast variation between samples. The operating threshold $\tau = 0 . 1$ was selected from the validation precision–recall curve to maximize class-averaged Dice under the strong foreground–background imbalance of thin crack structures (<1% foreground pixels; Fig. A.1). Morphometric sensitivity across τ ∈ {0.05, 0.1, 0.2, 0.3, 0.5} remained qualitatively stable (Fig. A.2), supporting the use of the segmentation outputs for downstream crack morphometrics.

Because many cracks are only 2–5 pixels wide, pixel-exact Dice is conservative: small boundary shifts can substantially reduce overlap without altering the underlying crack morphology [30, 31, 32]. Consistent with this, topology-aware centreline Dice (clDice) [33] reached a mean of 0.587 (intragranular: 0.564, early intergranular: 0.520, late intergranular: 0.678), while boundary F1 increased from 0.446 at d = 1 px tolerance to 0.600 at d = 2 px (Fig. S2), indicating that residual errors are dominated by small boundary misalignments. The segmentation outputs are therefore used as an intermediate representation for extracting downstream crack morphometrics rather than as the final scientific endpoint.

To assess whether domain-specific supervision is required for crack morphometrics, we evaluated the promptable foundation model SAM2 (Hiera-Large, sam2.1) [27] on the same 482-tile reference set using a binary crack-versusbackground target (Table A.1). In automatic mode (64 × 64 dense point grid, no prompts), SAM2 achieved a binary Dice score of 0.019 [95% CI 0.018–0.020]. Providing Otsu-derived dark-region centroids as point prompts increased Dice to 0.025 [95% CI 0.023–0.026], whereas the fine-tuned framework reached a binary Dice score of 0.77 on the same target (paired bootstrap p < 0.001, n = 1000).

Unlike the supervised framework, SAM2 is class-agnostic and therefore cannot directly resolve the intragranular and intergranular crack morphologies required for downstream morphometric analysis (Fig. A.3; further detail in Fig. S3).

## Per-particle morphometric quantification

Per-particle width, tortuosity, and area-fraction distributions extracted from the class maps resolve degradation morphologies that pixel-overlap metrics alone cannot capture.

We automatically identified regions of interest (ROIs) for morphometric analysis from the crack segmentation masks (Fig. S4a–d). For intergranular cracks (early and late), the ROIs approximate particle-scale regions derived from SEM contrast (97 initial, 69 cycled-aged, 49 calendar-aged; each particlescale ROI yields one early and one late measurement). Intragranular ROIs correspond to connected crack components within individual particles (446 initial, 610 cycled-aged, 623 calendar-aged). Within each ROI, crack centrelines were extracted by skeletonization and local widths were computed from the Euclidean distance transform (Fig. S1c–f; previewed in Fig. 2e). Each crack class was characterised by three skeleton-based descriptors (see Methods for definitions and Table A.2 for summary statistics): area fraction (percentage of ROI area occupied by crack pixels), mean width (average maximal-inscribed-disk diameter along the centreline), and geometric tortuosity (centreline arc length divided by the projected span along the principal crack axis).

Cycling strongly increases late-stage intergranular cracking relative to both the initial and calendar-aged states, consistent with progressive grain-boundary degradation during repeated electrochemical cycling. The cycled-aged crosssection shows a late intergranular area fraction of 4.58%, compared with 0.52% in the initial sample and 0.49% in the calendar-aged sample, corresponding to an approximately ninefold increase relative to the initial state. In contrast, the calendar-aged sample remains near baseline despite elevated-temperature storage. Early intergranular coverage remains comparatively stable across conditions (1.30–2.34%), with modest width diferences (∼91–99 nm), consistent with a broadly similar early-stage boundary-opening morphology.

Late intergranular cracks in the cycled-aged sample are markedly more tortuous than those in the initial and calendar-aged samples, indicating the development of more interconnected grain-boundary crack networks during electrochemical cycling. Mean late-crack width increases only modestly from 159 ± 22 nm in the initial sample to 167 ± 39 nm in the cycled-aged sample (not statistically significant; Fig. 5j), whereas tortuosity increases significantly from 1.55 ± 0.63 to 2.40 ± 1.00. ROI-level distributions of area fraction, mean width, tortuosity, and per-pixel width probability density are summarised in Fig. 4a–d (previewed in Fig. 2f). Cycled-aged late cracks occupy a broader range of area fractions, widths, and tortuosities than the initial and calendar-aged samples, with pairwise two-sided Mann–Whitney U tests indicating statistically significant distributional shifts and Clif’s δ values supporting substantial efect sizes. Intragranular morphometrics (Fig. 4e–h) remain comparatively similar across conditions, whereas combined-class distributions (Fig. 4i–l) retain the cycled aged shift toward larger widths and tortuosities, including a broader width tail extending beyond 200 nm for late intergranular cracks. These condition-level trends are preserved across threshold reruns (Fig. A.2).

a  
![](images/cd7c3d2761205ab80aeede920218f410b91c8dc410afe7b36009bcf421bf9104.jpg)  
b

![](images/97a6b9221442a0c041b399d441fdb765d352d3d3adf998eb14f220d60818bafa.jpg)  
c

d  
![](images/20a6f8b05f6ff7b91750e5b917cebad9a647362a367cbad5b15cac9cf1518454.jpg)

![](images/60f5f8d1ccd58c0e5d0a88dd462067987d5f890fabe4cf8ade098e829a63cd96.jpg)

![](images/36fb2e64681642771df0e94aa091f79c32df4e70d4fbf3424330430b6eb6fc92.jpg)  
f

![](images/59ec23fbc3852a88ac2c228fadee3d646144e4797a5491c1ac4086e331eb0650.jpg)  
g

![](images/b45e7ecae67017deeaf30df8eccb39754845425d64d234488aff73d2d4052eb1.jpg)  
h

![](images/a11628014e04dccd8b04bddb59e0cf7ee932d7d5085d4675f202b10cb201ddcf.jpg)

![](images/3d719688dc82a1db3dbf3bb221085da47f65409e25bc7fc3689f556a970446ed.jpg)

![](images/c5905ec500395e798bcdef5010aaf728c7d56041153601764fcbbf1b0bb47896.jpg)  
k

l  
![](images/efd5f77a9be097b14ebc763ba5898ac2f289c8a85abe33729d1b41ee9e8d1878.jpg)

![](images/b2a0d8ec6cc740e16ce6354177b9a7abbebe6e9790ba5d9151a65927a1958ccd.jpg)  
Figure 4: Automated morphometric quantification of crack characteristics. Three rows show intergranular cracks (a–d, combining early and late classes), intragranular cracks $\left( \mathrm { e - h } \right)$ , and all cracks combined (i–l). Each row contains four panels: violin plots of area fraction (a, e, i), mean crack width (b, f, j), tortuosity $( \mathrm { c } , \mathrm { g } , \mathrm { k } )$ , and width probability distributions (d, h, l). Brackets indicate statistically significant pairwise diferences (two-sided Mann– Whitney U test); significance levels: $* * * \ q < 0 . 0 0 1 , \ * * \ 0 . 0 0 1 \leq q < 0 . 0 1 , \ * \ 0 . 0 1 \leq q < 0 . 0 5$ ns $q \ge 0 . 0 5 ,$ , where q is the Benjamini–Hochberg false-discovery-rate-adjusted p-value across the family of 27 pairwise tests (3 subsets $\times ~ 3$ metrics × 3 condition pairs); see Methods. Cycled-aged samples show significantly higher late intergranular crack area fraction and tortuosity than initial and calendar-aged samples; mean crack width is only slightly larger and not statistically significant (panel j).

To summarise condition-dependent morphometric diferences, we compared ROI-level distributions using two-sided Mann–Whitney U tests with Clif’s $\delta$ efect sizes (see Methods). ECDFs, orientation roses, and area-fraction-versuswidth scatter plots for intergranular, intragranular, and combined populations are shown in Fig. 5a–i, with the intergranular Mann–Whitney U and Clif’s $\delta$ summary in Fig. 5j. For late intergranular area fraction, cycled-aged ROIs exceeded initial ROIs $( U = 5 8 6 , q < 0 . 0 0 1 , \delta = - 0 . 8 2 )$ , indicating a strong shift toward higher crack coverage under cycling, whereas the calendar-aged distribution remained similar to the initial state $( U = 2 . 4 1 9 , \ q = 0 . 8 5 , \ \delta \ =$ +0.02). Because each condition is represented by a single cross-section, these ROI-level statistics describe within-cross-section morphometric heterogeneity rather than cell- or lot-level population inference (see Discussion). Additional validation is provided by Fig. A.4 and Figs. S2 and S5.

## 3. Discussion

Cycling produces a strong enrichment of late intergranular cracking in NMC that is not observed in the calendar-aged sample under elevated-temperature storage, consistent with progressive grain-boundary degradation during repeated electrochemical lithiation–delithiation [1, 2]. The accompanying increase in crack area fraction and geometric tortuosity (Table A.2) indicates that cycling is associated with both greater grain-boundary crack coverage and the development of more interconnected crack networks relative to the initial and calendar-aged states.

The morphologies observed in the cycled-aged samples are consistent with the anisotropic lattice evolution of Ni-rich NMC during electrochemical cycling, in which the in-plane lattice contracts while the c-axis undergoes expansion followed by collapse near the H2→H3 transition at high state of charge [34, 35, 36]. Such anisotropic deformation has been associated with mechanically incompatible strain between diferently oriented primary grains and the development of intergranular cracking [37, 4, 38]. The more tortuous, higher-coverage intergranular crack networks observed here are also consistent with progressive grain-boundary degradation and surface reconstruction reported in Ni-rich cathodes, including rock-salt formation and lattice-oxygen-related surface reactivity [3, 11, 10, 9]. The accompanying increase in geometric tortuosity (2.40±1.00 vs. 1.55 ± 0.63) and mean crack length $( 1 4 . 2 \pm 1 0 . 5 \ \mu \mathrm { m \ v s . \ 1 . 5 \pm 3 . 4 \ \mu \mathrm { m } ) }$ further supports the development of increasingly interconnected crack networks under cycling [8, 39, 7]. In contrast, the calendar-aged sample shows comparatively limited crack evolution despite elevated-temperature storage, consistent with the absence of repeated electrochemical strain [40, 41, 42]. Figure 6 summarises the proposed progression of crack-network development and the associated morphometric trends. The skeleton-based tortuosity reported here describes individual crack-path geometry rather than efective pore-network transport tortuosity.

The methodological contribution of this work is to extract per-particle crack morphometrics from sparse and imperfect SEM annotations, concentrating the expert-label budget on the scientific question rather than on representation learning. Per-pixel class probabilities serve as an intermediate representation from which distributions of crack width, tortuosity and area fraction are derived. By freezing a self-supervised encoder [15] and training only a lightweight decoder [29], the framework leverages transferable visual representations despite the limited number of expert-corrected SEM tiles. This extends transferlearning approaches in low-data scientific imaging [16, 43] to multi-class morphometric analysis at the hundred-megapixel scale, beyond earlier batterymicrostructure studies focused on single crack classes or smaller imaging volumes [21, 20, 24]. The morphometric formulation is also relatively tolerant to local annotation uncertainty because width and tortuosity measurements average over crack geometry rather than relying on exact pixel overlap. Consistent morphometric trends across thresholds and descriptors further support the robustness of the extracted degradation signatures beyond pixel-level segmentation metrics.

![](images/eed8420c19e3e6511642f48bf2f19240253f1dd0d1f7d3bdfe3fee4b6dc16248.jpg)  
b

![](images/23660a01682d47cc1576c85ba06a971f96f2bb24c3ffb9497968632c51161104.jpg)

c  
![](images/3b6d8f3558ea96d37d2af4479b2f482cfc2c6af61b99d9dc610e669e791978a3.jpg)

d  
![](images/5215ac55061d7ed15fb6ef4d667fd03107800a9f777d90471e64e28a3a284ca1.jpg)  
e

![](images/746137fcf8487724d4b545f5f04b37b8c4bf11bfadcd7ca06ee26bb9ba3dbfac.jpg)

f  
![](images/d8c83fb031ab93133874eec87e535973eef4270214667e8b4ae643ad19fc6980.jpg)

g  
![](images/2a308557b971475680d5145e2771f2ed4e7e98568c2d819c3142560784fcccd9.jpg)  
h

![](images/8320f52a09032e12a0b71265599f125942bed25523c3b36d6dbdcaee2ccf1b46.jpg)

i  
![](images/7e8bdf0ef660d6075ca30b4b866da55e653ce08dd8b6a0c388e1ea707a6d5b8f.jpg)

j
<table><tr><td>Comparison Metric</td><td></td><td>U-statistic q-value</td><td></td><td></td><td>Cliff&#x27;s δ Effect size</td></tr><tr><td>Ini vs Cyc Ini vs Čal</td><td>Area fraction Area fraction</td><td>586 2,419</td><td>&lt; 0.001 0.85</td><td>-0.82 +0.02</td><td>Large Negligible</td></tr><tr><td>Cyc vs Cal</td><td>Area fraction</td><td>3,098</td><td>&lt; 0.001</td><td>+0.83</td><td>Large</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ini vs Cyc</td><td>Width (nm)</td><td>709</td><td>0.60</td><td>-0.09</td><td>Negligible</td></tr><tr><td>Ini vs Cal</td><td>Width (nm)</td><td>160</td><td>0.44</td><td>+0.21</td><td>Small</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cyc vs Cal</td><td>Width (nm)</td><td>482</td><td>0.15</td><td>+0.35</td><td>Medium</td></tr><tr><td>Ini vs Cyc</td><td>Tortuosity</td><td>352</td><td>&lt; 0.001</td><td>-0.55</td><td></td></tr><tr><td>Ini vs Cal</td><td>Tortuosity</td><td>143</td><td>0.77</td><td>+0.08</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $\mathrm { N e g l i g i b l e }$ </td></tr><tr><td> $\mathrm { C y c \ v s \ C a l }$ </td><td>Tortuosity</td><td>565</td><td>0.010</td><td>+0.58</td><td></td></tr></table>

Figure 5: Statistical analysis of degradation patterns. Rows: intergranular (a–c), intragranular (d–f), and all cracks combined (g–i). Columns: ECDFs of mean crack width with 95% Dvoretzky–Kiefer–Wolfowitz bands and median markers $( \operatorname { a } , \operatorname { d } , \operatorname { g } ) ;$ ; orientation rose diagrams $( \mathrm { b , e , h } ) ;$ area fraction vs. mean width (c, f, i). The cycled-aged width distributions (red) show only a slight, non-significant shift toward larger widths relative to initial (blue) and calendar-aged (green); the statistically significant cycled-aged increases are in area fraction and tortuosity (panel j). (j) Mann–Whitney U tests on late intergranular cracks with Clif’s δ; q-values are Benjamini–Hochberg-adjusted within the 9-test intergranular-late sub-family (see Methods and Supplementary Methods §S7).

![](images/77ca2ed07c4c000783094e1d48dbd62616af80e66f13ad5e12a907eaeef45a61.jpg)  
Figure 6: Proposed mechanism of crack evolution in NMC cathodes. (a–d) Schematic illustration of characteristic crack types, shown as a generalized progression; in practice, these mechanisms may overlap temporally and coexist depending on local microstructural conditions: (a) Initial polycrystalline NMC particle with coherent grain boundaries. (b) Intragranular cracks form within primary particles due to anisotropic lattice strain during H2→H3 phase transition. (c) Intergranular cracks nucleate at grain boundaries under repeated mechanical stress from volume changes. Intragranular cracks from (b) persist but are omitted for visual clarity. (d) Late-stage degradation with widened cracks (>150 nm), rock-salt phase formation, and electrolyte infiltration pathways. Intragranular cracks remain present but are not shown to highlight intergranular crack progression. (e) Generalized trend of crack morphology metrics correlated with electrochemical capacity retention, suggesting crack width as a quantitative degradation marker. The depicted sequence represents a predominant tendency rather than a strict temporal order.

Each destructive cross-section yields hundreds of per-particle morphometric measurements, enabling population-level characterization of crack morphology from sparse SEM annotation. More broadly, sparse-label morphometric inference may be applicable to other destructive-microscopy domains in which annotation cost limits throughput, including ceramic microstructure, geological thin-sections, polymeric composites, and biological tissues. Self-supervised encoders [15] and promptable segmenters [26, 27] have begun to demonstrate broad transfer in scientific imaging; the contribution here is to make that transfer quantitative in a morphometric-analysis setting rather than as a segmentation benchmark alone. Standardised morphology-based diagnostics built on this framework would support lithium-ion manufacturing quality assurance, aging diagnostics, and quantification of single-crystal NMC designs that suppress intergranular cracking [44, 45, 46]. Extensions to 3D imaging (XRM, nano-CT), multimodal fusion (EDS, EBSD), active-learning annotation, and coupling with electrochemical diagnostics for physics-based aging models are natural next steps.

Several limitations of the present case study should be noted. First, the focus of this work is the sparse-label morphometric-inference framework rather than optimisation of a specific segmentation architecture; the frozen visiontransformer encoder with a lightweight decoder represents one implementation of this approach. Systematic benchmarking across alternative self-supervised encoders, decoder architectures, and training seeds is reserved for future work. Second, only three SEM cross-sections (one per aging condition) were available because high-resolution destructive imaging is costly, so the reported ROI-level statistics describe within-cross-section morphometric heterogeneity rather than cell- or lot-level population inference. Third, two-dimensional sectioning cannot fully resolve out-of-plane crack morphology or pore–crack ambiguity, motivating future volumetric validation using FIB-SEM or X-ray microscopy. Finally, the framework was trained on NMC images acquired on a single instrument; extension to other cathode chemistries, imaging conditions or laboratories will likely require domain adaptation and larger multi-site reference datasets.

## 4. Conclusions

We introduced a sparse-label morphometric-inference framework that turns frozen self-supervised vision-transformer features into per-particle crack morphometrics from few expert annotations, concentrating the annotation budget on quantification rather than representation learning. An iterative modelassisted annotation loop grew 79 hand-labelled tiles into a 482-tile expertcorrected reference set across three hundred-megapixel NMC cathode crosssections, resolving intragranular as well as early- and late-stage intergranular cracking. The cycled-aged cathode exhibited an approximately ninefold increase in late intergranular cracking and more tortuous, higher-coverage crack networks, whereas the calendar-aged sample remained near baseline, consistent with degradation from repeated electrochemical cycling rather than elevatedtemperature storage alone. These condition-level trends were preserved across decision thresholds and topology-aware metrics, while a zero-shot SAM2 baseline failed on the thin crack structures, indicating that domain-specific supervision is needed for this task. By converting a single destructive cross-section into population-level crack statistics, the framework supports aging diagnostics, manufacturing quality assurance and second-life assessment, and extends naturally to other destructive-microscopy domains in which annotation cost limits throughput.

## 5. Methods

## Multi-class segmentation of crack types

Pixel-wise crack typing is cast as a three-class semantic segmentation problem with labels intragranular (reflecting defect-induced crack initiation within primary particles [47]), early intergranular, and late intergranular.

The frozen self-supervised DINOv3 encoder [15] provides dense textureand morphology-aware features. The checkpoint facebook/dinov3-vitb16- pretrain-lvd1689m is a ViT-B/16 backbone (85.7 M frozen parameters, 768- dimensional patch embeddings). Each greyscale 1024 × 1024 px SEM tile is replicated to three channels and tokenised on a 64 × 64 grid of $1 6 \times 1 6$ patches, yielding a $6 4 \times 6 4 \times 7 6 8$ feature map that is passed to a DeepLabV3+ decoder [29] (17.8 M trainable parameters). For an input image $I \in [ \bar { 0 } , 1 ] ^ { H \times W }$ , the network predicts per-pixel logits $\ell \in \overset { \prime } { \mathbb { R } } ^ { C \times H \times W }$ with C=3; per-class probabilities use independent sigmoids,

$$
p _ { c } ( x ) = \sigma \big ( \ell _ { c } ( x ) \big ) = \frac { 1 } { 1 + \exp \big ( - \ell _ { c } ( x ) \big ) } , \quad c \in \{ 1 , \ldots , C \} , x \in \Omega ,\tag{1}
$$

and binary masks follow by thresholding, $\hat { y } _ { c } ( x ) = \mathbf { 1 } [ p _ { c } ( x ) \geq \tau ]$ . The operating threshold $\tau { = } 0 . 1$ was selected on the validation precision–recall curve (architecture in Fig. 2a; sensitivity in Fig. A.2). Training minimised a weighted sum of binary cross-entropy, mean-squared-error, and Dice losses (Supplementary Methods §S1).

Tiles from each cross-section were randomly partitioned 80%/20% per image, yielding 96 held-out validation tiles from a 482-tile pool: 79 manual seed tiles (cycled-aged) and 403 model-assisted, expert-corrected tiles distributed as 160 (cycled-aged), 81 (calendar-aged), and 162 (initial). Because tiles share 50% spatial overlap by construction, pixel-level segmentation metrics are reported on the full 482-tile expert-corrected reference as operating-point calibration rather than out-of-distribution generalisation (Supplementary Methods §S5). The ROI-level morphometric distributions reported in the Results are computed on the full cross-sections and are independent of the tile-level split.

## Baseline evaluation using SAM2

The Segment Anything Model 2 (facebook/sam2.1-hiera-large) was evaluated in two configurations: automatic mask generation on a dense point grid, and prompted segmentation using Otsu-thresholded dark regions as point prompts. Because SAM2 is class-agnostic, the comparison was performed on a binary crack-versus-background target obtained by collapsing the three reference classes; on this binary metric the DINOv3 framework reaches Dice = 0.77 against the SAM2 values reported in Table A.1 and visualised in Fig. A.3. Prompt-grid spacing, intensity and geometric post-filters, and multi-mask selection are described in Supplementary Methods §S6.

Inference and overlay mask generation

Inference on full-resolution images $( \sim 1 7 , 0 0 0 \times 7 , 0 0 0 ~ \mathrm { p x } )$ used an overlapping sliding-window strategy, averaging per-class predictions across overlapping tiles (Supplementary Methods §S2). Final per-class binary masks were thresholded and rendered as alpha-blended overlays at native resolution for side-by-side comparison across the initial, calendar-aged, and cycled-aged probes.

## Segmentation mask analysis pipeline

The morphometric pipeline transforms each per-pixel likelihood map into a cleaned binary crack mask, extracts particle-level regions of interest (ROIs) corresponding to individual cathode secondary particles (or small clusters), and reduces each ROI to skeleton-based width and geometry distributions; all measurements are computed in pixel units and converted to physical units via SEM calibration. The crack likelihood map $m ( x ) \in [ 0 , 1 ]$ is thresholded at τ to give a binary field $B ( x ) = \mathbf { 1 } \{ m ( x ) \geq \tau \}$ , with small connected components filtered by area and skeleton length, and ROIs are obtained from the SEM backscatter contrast (panels a–e of Fig. S4; full procedure in Supplementary Methods §S2).

Within each ROI, the binary crack mask is reduced to one-pixel centrelines by skeletonisation, $S \subset \Omega$ . The Euclidean distance transform on the crack boundary ∂B (panels $\mathrm { c } ,$ d of Fig. S1),

$$
d ( x ) \ = \ \operatorname* { m i n } _ { y \in \partial B } \ \| x - y \| _ { 2 } ,\tag{2}
$$

defines a local crack width at each skeleton point $x _ { s } \in S$

$$
w ( x _ { s } ) = 2 d ( x _ { s } ) ,\tag{3}
$$

the diameter of the maximal inscribed disk centred at $x _ { s }$ . Non-finite or nonpositive samples are excluded from summary statistics but retained as zeros in visualisation maps.

## Geometric descriptors and width distributions

Panel c of Fig. S1 illustrates orientation, topology, and tortuosity for an example ROI. Let $\{ x _ { i } \} _ { i = 1 } ^ { n }$ be skeleton coordinates in image axes. We summarise:

• Crack length: $\textstyle L = \sum _ { x \in S } 1$ (in pixels).

• Area fraction: $| \{ x : B ( x ) = 1 \} | / | \Omega _ { \mathrm { R O I } } |$

• Topology: classify pixels by $3 \times 3$ neighbour degree (endpoints: degree 1; branch points: degree $\geq 3 ;$ isolated: degree 0).

• Orientation and anisotropy: with centered coordinates $X = [ x _ { i } - { \bar { x } } ]$ the covariance $\begin{array} { r } { \Sigma = \frac { 1 } { n - 1 } X ^ { \top } \bar { X } } \end{array}$ has eigenpairs $( \lambda _ { 1 } , { \mathbf { v } } _ { 1 } ) , ( \lambda _ { 2 } , { \mathbf { v } } _ { 2 } ) , \lambda _ { 1 } \geq \lambda _ { 2 } \geq$ 0. Orientation is $\angle \mathbf { v } _ { 1 } \in [ 0 ^ { \circ } , 1 8 0 ^ { \circ } )$ and anisotropy is

$$
\mathcal { A } = \frac { \lambda _ { 1 } } { \lambda _ { 2 } + \varepsilon } .\tag{4}
$$

• Geometric tortuosity: ratio of centreline length to projected span along $\mathbf { v } _ { 1 } ;$ a morphological descriptor, distinct from the efective transport tortuosity of the pore network:

$$
\mathcal T = \frac { L } { \mathrm { m a x } _ { i } \langle x _ { i } - \bar { x } , { \bf v } _ { 1 } \rangle - \mathrm { m i n } _ { i } \langle x _ { i } - \bar { x } , { \bf v } _ { 1 } \rangle + \varepsilon } .\tag{5}
$$

Because $L$ is a raw skeleton-pixel count rather than a ${ \sqrt { 2 } } .$ -weighted arc length, near-linear cracks can yield $\tau$ marginally below unity; such values reflect near-straight morphology rather than a measurement error.

• Transverse roughness: RMS deviation orthogonal to $\mathbf { v } _ { 1 }$ ,

$$
\rho = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle x _ { i } - \bar { x } , \mathbf { v } _ { 2 } \rangle ^ { 2 } } .\tag{6}
$$

Let $\{ w _ { i } \} _ { i = 1 } ^ { N }$ be valid widths sampled on S. We report descriptive statistics (min, max, mean, median, std) and visualize: (i) a histogram of $\{ w _ { i } \}$ and (ii) the empirical cumulative distribution function (ECDF), defined as

$$
{ \widehat { F } } ( t ) = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ w _ { i } \leq t \} ,\tag{7}
$$

i.e. the fraction of the centreline length with width $\leq t \left[ 4 8 \right]$ . Quartiles $( P _ { 2 5 } , P _ { 5 0 } , P _ { 7 5 } )$ are marked for cross-ROI comparison (panel f of Fig. S1; aggregated ECDFs in panels a, d, and g of Fig. 5).

## Statistical analysis

Per-ROI morphometric distributions (area fraction, mean width, geometric tortuosity) were compared between aging conditions using two-sided Mann– Whitney U tests with Clif’s δ as a non-parametric efect-size estimator and Benjamini–Hochberg correction for family-wise multiple comparisons. Testfamily structure, raw versus adjusted statistics, and the released TSV summary are detailed in Supplementary Methods §S7.

## Reporting and assumptions

Quantitative measurements derive from thresholded crack masks; reported quantities are in pixels and converted to physical units via SEM calibration. Width statistics depend on $\tau ,$ with sensitivity reported in Fig. A.2. Cell construction, aging protocols, post-mortem sample preparation, and SEM acquisition parameters are summarised in the Supplementary Information (Sample preparation and SEM imaging).

## CRediT authorship contribution statement

Thorsten Tegetmeyer-Kleine: Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Visualization, Writing – original draft. Thomas Schmitt: Conceptualization, Writing – review & editing. Phillip Aquino: Conceptualization, Writing – review & editing. Christiane Rahe: Investigation. Dirk Uwe Sauer: Resources, Writing – review & editing. Weihan Li: Supervision, Funding acquisition, Writing – review & editing. All authors reviewed and approved the final manuscript.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by Honda Research Institute Europe.

## Data availability

The datasets generated and analysed during this study, including the expertcorrected SEM tile image–mask pairs, SEM cross-sections, released model weights, validation artefacts, and morphometric analysis tables, are openly available on Figshare at https://doi.org/10.6084/m9.figshare.32975921.

## Code availability

The code used for annotation support, inference, validation, morphometric analysis, and figure generation in this study, together with the trained model weights, is openly available at https://git.rwth-aachen.de/thorsten.tegetmeyer-kleine hri-nmc-crack-segmentation.

![](images/2f6021937c874b80b289ca87476e34dff2c040ecc75efdd19d6903d2c2cebf26.jpg)

![](images/9adc3f8f3446ab22092af7f53300416781ccf384544991ff14c4a692d673af75.jpg)  
Figure A.1: | Precision–recall curves and threshold selection. (a) Per-class PR curves with average precision (AP) values; filled circles mark the operating point at τ=0.1. (b) Perclass and mean Dice score vs. decision threshold τ; the selected threshold is annotated. The low optimal τ reflects extreme foreground–background imbalance (<1% crack pixels).

Table A.1: | Zero-shot SAM2 baselines vs. fine-tuned DINOv3 (binary segmentation). SAM2 sam2.1-hiera-large in two configurations evaluated on the same 482-tile reference set as the main DINOv3 result. Binary Dice and IoU were computed after collapsing the 3-class reference to crack-vs-background. For each SAM2 configuration the higher of the with- and without-filter values was retained per tile. The 95% CIs are non-paired bootstrap with n = 1000 resamples.
<table><tr><td>Method</td><td>Binary Dice ↑</td><td>Binary IoU ↑</td><td>3-class capable?</td></tr><tr><td>DINOv3-DeepLabV3+ (ours, binary)</td><td>0.77</td><td>0.65</td><td>√ (via 3-class output)</td></tr><tr><td>SAM2-auto (zero-shot)</td><td>0.019</td><td>0.010</td><td></td></tr><tr><td>SAM2-prompted (Otsu)</td><td>0.025</td><td>0.013</td><td></td></tr></table>

![](images/207ca8dad9deb96ed662b5b7f9a2df6e7e32c8f976fd9925d44ebe2c196c9124.jpg)  
Figure A.2: | Morphometric threshold sensitivity. Area fraction, mean crack width, and tortuosity (mean ± std across ROIs) as a function of the decision threshold $\tau \in$ $\{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 \}$ for each crack class and sample condition. Area fraction decreases monotonically with increasing τ as stricter thresholds suppress low-confidence crack pixels; mean width and tortuosity are more stable because skeleton-based statistics average over retained crack pixels. The relative ordering of conditions is preserved across all τ tested: the cycled-aged cross-section consistently shows the highest late intergranular area fraction, width, and tortuosity, while calendar-aged late-crack coverage remains comparable to the initial state throughout. Within $\tau \in [ 0 . 0 5 , 0 . 2 ]$ the absolute morphometric values closely track the maintext numbers; outside this range the relative ordering is preserved but absolute values deviate more substantially. The operating point $\tau { = } 0 . 1$ used throughout this study is highlighted.

Table A.2: | Morphometric statistics by sample and crack class. Mean ± std across ROIs, supporting the per-condition morphometric comparisons in the main article. Bold: highest values within each metric column. Pixel size: 22.3 nm.
<table><tr><td>Sample</td><td>Class</td><td>ROIs</td><td>Area (%)</td><td></td><td>Width (nm) Length (µm) Tortuosity</td><td></td></tr><tr><td>Initial</td><td>Intragranular</td><td>446</td><td> $1 0 . 3 4 \pm 7 . 5 4$ </td><td> $1 0 7 \pm 3 3$ </td><td> $0 . 8 \pm 0 . 6$ </td><td> $0 . 9 5 \pm 0 . 3 1$ </td></tr><tr><td>Initial</td><td>Intergran. early</td><td>97</td><td> $1 . 6 6 \pm 0 . 7 6$ </td><td> $9 1 \pm 1 0$ </td><td> $8 . 6 \pm 5 . 7$ </td><td> $1 . 3 8 \pm 0 . 5 3$ </td></tr><tr><td>Initial</td><td>Intergran. late</td><td>97</td><td> $0 . 5 2 \pm 1 . 2 9$ </td><td> $1 5 9 \pm 2 2$ </td><td> $1 . 5 \pm 3 . 4$ </td><td> $1 . 5 5 \pm 0 . 6 3$ </td></tr><tr><td>Cycled-aged</td><td>Intragranular</td><td>610</td><td> $9 . 2 5 \pm 7 . 1 7$ </td><td> $1 0 7 \pm 3 2$ </td><td> $1 . 1 \pm 0 . 9$ </td><td> $0 . 9 7 \pm 0 . 3 1$ </td></tr><tr><td>Cycled-aged</td><td>Intergran. early</td><td>69</td><td> $1 . 3 0 \pm 1 . 2 3$ </td><td> $9 8 \pm 1 5$ </td><td> $7 . 3 \pm 7 . 4$ </td><td> $1 . 5 8 \pm 0 . 7 3$ </td></tr><tr><td>Cycled-aged</td><td>Intergran. late</td><td>69</td><td> $\mathbf { 4 . 5 8 \pm 3 . 6 0 }$ </td><td> ${ \bf 1 6 7 \pm 3 9 }$ </td><td> ${ \bf 1 4 . 2 \pm 1 0 . 5 }$ </td><td> $\mathbf { 2 . 4 0 \pm 1 . 0 0 }$ </td></tr><tr><td>Calendar-aged Intragranular</td><td></td><td>623</td><td> $9 . 5 1 \pm 7 . 9 6$ </td><td> $1 0 6 \pm 3 5$ </td><td> $0 . 9 \pm 0 . 7$ </td><td> $0 . 9 3 \pm 0 . 3 2$ </td></tr><tr><td></td><td>Calendar-aged Intergran. early</td><td>49</td><td> $2 . 3 4 \pm 1 . 0 1$ </td><td> $9 9 \pm 1 1$ </td><td> $1 0 . 2 \pm 5 . 1$ </td><td> $1 . 7 1 \pm 0 . 4 7$ </td></tr><tr><td></td><td>Calendar-aged Intergran. late</td><td>49</td><td> $0 . 4 9 \pm 1 . 1 6$ </td><td> $1 4 6 \pm 2 6$ </td><td> $1 . 4 \pm 3 . 3$ </td><td> $1 . 4 5 \pm 0 . 6 3$ </td></tr></table>

![](images/458575cfcd734c6b9c7e45b8f50a3f670aa4b3ffd59a89a4a8358e7627cf2b6a.jpg)  
Figure A.3: | Zero-shot SAM2 vs. fine-tuned DINOv3 on the NMC crack validation set. Columns: raw tile; expert 3-class reference (intragranular red, early intergranular green, late intergranular blue); SAM2 in automatic mode (64 × 64 point grid, no prompts); SAM2 prompted with Otsu-derived dark-region centroids and area-bounded lowest-mean-intensity multi-mask selection; fine-tuned DINOv3-DeepLabV3+. Per-prediction binary Dice values are annotated below each prediction column. The inset on the SAM2-auto column of row 1 shows the unfiltered output before geometric and intensity post-processing, illustrating the oversegmentation failure mode on textured SEM matrix. The 3-class reference is collapsed to a single crack vs. background target because SAM2 is class-agnostic; this typology gap is the structural reason fine-tuning is required for the cycled/initial late-intergranular comparison.

![](images/aacbeeba057a231d827fdef9c30f3898ec847322c42029bbce81f12da3e64114.jpg)  
Figure A.4: | Per-cross-section segmentation metrics by aging condition. Mean perclass scores for Dice, clDice, IoU, Precision, and Recall (with standard-deviation error bars) computed separately for the initial, cycled-aged, and calendar-aged cross-sections. Crack classes are abbreviated as I (intragranular), EI (early intergranular), and LI (late intergranular). The relative ranking of per-class scores is preserved across aging conditions: the frozen DINOv3 encoder transfers across the three aging states despite substantial diferences in SEM contrast, brightness, and late-crack density.

## Appendix B. Supplementary information

This appendix reproduces the Supplementary Information accompanying the manuscript. References to “the main article” below refer to Sections 1–5 of this preprint; Extended Data items are in Appendix A.

## Supplementary Notes

## Cell, aging, and post-mortem sample preparation

Aged samples were prepared from commercial Hunan LiFun Technology pouch cells (Model 103962) with a Ni-rich NCM cathode and graphite anode. Per the manufacturer datasheet (PS-FC-103962-01, rev. A0, 2018-11-05), each cell has nominal capacity 3400 mAh, nominal voltage 3.80 V, operating window 3.0–4.35 V, internal impedance ≤60 mΩ (50 % SoC, 1 kHz AC), and Al-laminate pouch dimensions $\leq 1 0 . 4 \times 3 9 . 0 \times 6 2 . 0$ mm at ${ \sim } 5 0 \mathrm { g }$ . The electrolyte is a $\mathrm { L i P F _ { 6 } }$ solution in $\mathrm { E C / E M C / D M C }$ , and the separator is a polyethylene film with a ceramic coating; the cells are qualified to GB/T 18287-2013 and GB 31241-2014.

Three cells were sampled to span representative aging states: an initial (formation-only) reference cell, a cycled-aged cell (K100, 100 charge–discharge cycles), and a calendar-aged cell (K187, stored at $4 5 ^ { \circ } \mathrm { C } )$ . One cell per condition was prepared for post-mortem cross-sectioning. Capacity retention at the end of aging was measured by reference performance tests (RPT) at $\mathrm { C / 3 }$ (≈1.13 A relative to the 3.4 Ah nominal capacity): the two characterised initial cells delivered 3.59 and 3.68 Ah (mean 3.64 Ah); K100 retained 3.12 Ah (86 % of initial mean); K187 retained 2.68 Ah (74 % of initial mean). After aging and the final RPT, cells were transferred to an argon-filled glovebox $\mathrm { ( O _ { 2 } }$ and $\mathrm { H _ { 2 } O < 0 . 5 p p m ) }$ disassembled, and the cathode coupons cross-sectioned and mounted for SEM imaging.

## Cross-section SEM imaging

Cross-section images of the three NMC cathodes were acquired on a ZEISS Supra 55 field-emission scanning electron microscope operated at 10 kV accelerating voltage using the backscattered electron (BSE) detector. Image tiles were assembled with ZEISS Atlas 5 (v5.3.5) with no line or frame averaging (LineAvg = 1). Pixel size was nominally 22.3 nm/px (per-sample: 22.29 nm/px for K100, 22.38 nm/px for K187, 22.28 nm/px for the initial sample). Final image dimensions were $1 7 , 0 8 8 \times 6 , 5 9 5 \mathrm { p x }$ (K100), $1 7 , 0 1 7 \times 6 , 8 4 8$ px (K187), and $^ { 1 7 , 1 3 7 \times 7 , 0 1 5 \mathrm { p x } }$ (initial). Acquisition parameters were identical across the three samples.

## S1. Training configuration

The encoder was kept fixed; only the decoder and output head were optimised during training. We used the AdamW optimiser [49] with learning rate $\eta { = } 1 0 ^ { - 4 } , \beta _ { 1 } { = } 0 . 9 , \beta _ { 2 } { = } 0 . 9 8 , \varepsilon { = } 1 0 ^ { - 9 }$ , and weight decay $\lambda { = } 1 0 ^ { - 3 }$ . A cosine-cyclical learning-rate schedule (4 warm-restart cycles, minimum learning rate $5 \times 1 0 ^ { - 5 } )$

was applied over the full training run. Training ran for up to 500 epochs with early stopping (patience = 20 epochs, monitoring validation loss). The batch size was 4 at 1024×1024 px in single precision (FP32), with a fixed random seed of 42. Model selection retained the checkpoint with the lowest validation loss; the selected checkpoint corresponds to epoch 465. Online data augmentation was applied during training to improve robustness to SEM acquisition variation while preserving crack topology; it comprised random afine transforms, flips, crop-and-resize, brightness/contrast jitter, additive Gaussian noise and blur, salt-and-pepper noise, pixel dropout, and grid masking, with full per-operation probabilities and ranges listed in Supplementary Section S8.

We minimised a per-channel weighted sum of binary cross-entropy (BCE), mean squared error (MSE), and Dice losses. Let $y _ { c } ( x ) \in \{ 0 , 1 \}$ denote the target mask and $p _ { c } ( x ) \in [ 0 , 1 ]$ the predicted probability for class c at pixel $x ,$ with class weights $w _ { c }$ and $\varepsilon > 0$ for numerical stability:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { \left| \Omega \right| } \sum _ { x \in \Omega } \sum _ { c = 1 } ^ { C } w _ { c } \Big [ y _ { c } ( x ) \log p _ { c } ( x ) + \left( 1 - y _ { c } ( x ) \right) \log \left( 1 - p _ { c } ( x ) \right) \Big ] ,\tag{B.1}
$$

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { | \Omega | } \sum _ { x \in \Omega } \sum _ { c = 1 } ^ { C } w _ { c } \big ( p _ { c } ( x ) - y _ { c } ( x ) \big ) ^ { 2 } ,\tag{B.2}
$$

$$
\mathcal { L } _ { \mathrm { D i c e } } = 1 - \frac { 1 } { \sum _ { c = 1 } ^ { C } w _ { c } } \sum _ { c = 1 } ^ { C } w _ { c } \frac { 2 \sum _ { x \in \Omega } p _ { c } ( x ) y _ { c } ( x ) + \varepsilon } { \sum _ { x \in \Omega } p _ { c } ( x ) + \sum _ { x \in \Omega } y _ { c } ( x ) + \varepsilon } ,\tag{B.3}
$$

with the total objective

$$
{ \mathcal { L } } = \lambda _ { \mathrm { B C E } } { \mathcal { L } } _ { \mathrm { B C E } } + \lambda _ { \mathrm { M S E } } { \mathcal { L } } _ { \mathrm { M S E } } + \lambda _ { \mathrm { D i c e } } { \mathcal { L } } _ { \mathrm { D i c e } } ,\tag{B.4}
$$

where $\lambda _ { \mathrm { B C E } } = \lambda _ { \mathrm { M S E } } = \lambda _ { \mathrm { D i c e } } = 1$ . Equal weights were used as a neutral initialisation, treating all three loss components as equally informative a priori. This symmetric combination is consistent with multi-term loss formulations commonly adopted in biomedical segmentation [24, 29].

## S2. Image-scale inference, stitching, and ROI extraction

We applied the trained model to full-resolution SEM images to generate per-pixel class-probability maps and corresponding overlay masks. Each image $( \sim 1 7 , 0 0 0 \times 7 , 0 0 0 ~ \mathrm { p x ) }$ was read in greyscale, rescaled to [0, 1], and processed in evaluation mode using the frozen DINOv3 encoder [15] with the trained DeepLabV3+ decoder [29]. Because the field of view exceeds GPU memory, inference was performed with an overlapping sliding-window strategy (50% overlap). Per-tile predictions $p _ { c } ^ { ( t ) } ( x )$ were stitched by visit-normalised averaging,

$$
\bar { p } _ { c } ( x ) = \frac { 1 } { N ( x ) } \sum _ { t \in \mathcal { T } } \mathbf { 1 } \{ x \in t \} p _ { c } ^ { ( t ) } ( x ) , \quad N ( x ) = \sum _ { t \in \mathcal { T } } \mathbf { 1 } \{ x \in t \} ,\tag{B.5}
$$

which suppresses artefacts at tile boundaries. To reduce false responses on bright foreground regions, we optionally gated the stitched maps using the greyscale

intensity $g ( x )$ and a background threshold $\tau _ { g } ,$ chosen either as a fixed value or from Otsu’s method $\tau _ { g } = \kappa \tau _ { \mathrm { O t s u } }$ [50]. Hard gating sets predictions outside the dark background to zero,

$$
\tilde { p } _ { c } ( x ) = \bar { p } _ { c } ( x ) { \bf 1 } \{ g ( x ) \leq \tau _ { g } \} ,\tag{B.6}
$$

whereas soft gating down-weights them smoothly:

$$
\tilde { p } _ { c } ( x ) = \bar { p } _ { c } ( x ) \mathrm { c l i p } \bigg ( \frac { \tau _ { g } - g ( x ) } { \mathrm { m a x } ( \tau _ { g } , \epsilon ) } , 0 , 1 \bigg ) .\tag{B.7}
$$

A light Gaussian blur on $g$ and brief morphological cleanup of the gate mask reduced artefacts without eroding thin cracks. Final per-class binary masks were obtained by global thresholding, $M _ { c } ( x ) = \mathbf { 1 } \{ \tilde { p } _ { c } ( x ) \geq \tau \}$ , and exported as perclass masks and composite overlays. Overlays were rendered by alpha blending at native resolution for side-by-side comparison across the initial, calendar-aged, and cycled-aged probes.

Following stitching, connected components in the binary maps are filtered by a minimum pixel area and a minimum skeleton length, suppressing isolated artefacts while preserving thin, elongated cracks. Particle-level ROIs are then extracted by intensity-thresholding the SEM backscatter contrast (cathode brighter than epoxy matrix), followed by morphological closing, area filtering, and padding, with each ROI clipped to an in-bounds square window. ROIs are ordered top-to-bottom and left-to-right for reproducible enumeration across cross-sections (panels a–e of Supplementary Figure S4).

## S3. Segmentation operating point and threshold selection

All quantitative results in the main article use the decision threshold $\tau { = } 0 . 1$ This operating point was selected on the validation precision–recall (PR) curve to maximise the class-averaged Dice score under the highly imbalanced foreground fractions characteristic of thin crack structures in SEM cross-sections (crack pixels constitute less than 1% of the tile area). At higher thresholds, recall collapses faster than the false-positive rate is reduced, so the class-averaged Dice peaks toward the low end of the threshold sweep. The PR-based threshold selection therefore yields a recall-sensitive operating point appropriate for sparse foreground structures; the corresponding PR curves and threshold sweep are shown in Fig. A.1 of the main article, and the resulting robustness of morphometric outputs to the choice of τ in the interval [0.05, 0.2] is documented in Fig. A.2 of the main article. Applying the optional intensity-based background gating (Eqs. B.6–B.7) at this operating point raises precision to 0.877 at the cost of recall (0.302), lowering the mean Dice to 0.434 and reflecting the false-positive/recall trade-of on bright cathode regions.

## S4. Supplementary morphometric definitions

The main Methods define the principal morphometric descriptors used in the analysis (crack length, area fraction, orientation/anisotropy, geometric tortuosity, and transverse roughness). For completeness, we record here the standardoperation conventions adopted by the pipeline.

Skeleton-pixel topology was classified by the local neighbourhood degree on the eight-connected $3 \times 3$ window centred on each foreground skeleton pixel. Pixels with degree 1 were labelled endpoints (crack termini), pixels with degree $\geq 3$ were labelled branch points (crack junctions), and pixels with degree 0 were labelled isolated (and excluded from path-length statistics). This convention is the standard medial-axis classification used throughout connected-component morphometry and corresponds to the topology counts reported in Supplementary Figure S1.

Width statistics derived from the Euclidean distance transform are reported using the empirical cumulative distribution function (ECDF)

$$
\widehat { F } ( t ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ w _ { i } \leq t \} ,
$$

i.e. the fraction of the centreline length with width $\leq t \ [ 4 8 ]$ . We summarise each ECDF by the first, second, and third quartiles $( P _ { 2 5 } , \ P _ { 5 0 } , \ P _ { 7 5 } )$ ; these robust descriptors are reported in preference to the mean and standard deviation for cross-ROI comparisons because the underlying width distributions are typically heavy-tailed and right-skewed by junction pixels.

## S5. Spatial overlap quantification

Validation tiles were quantified for spatial overlap with training tiles. Each validation tile v was assigned an overlap fraction

$$
o ( v ) \ = \ \operatorname* { m a x } _ { t } \ \frac { | v \cap t | } { | v | }
$$

taken over all training tiles t from the same SEM cross-section. By construction of the 50%-overlap sliding-window tiling, adjacent tiles share 50% of their area. Under the random 80/20 per-image partition, the probability that all four cardinal neighbours of a validation tile lie in the validation set is $0 . 2 ^ { 4 } \approx 0 . 0 0 2 .$ so the median overlap fraction $o ( v )$ across the 96 validation tiles is $\approx 0 . 5 0$ by construction. The reported pixel-level segmentation metrics are therefore most meaningfully read as operating-point calibration on the full 482-tile expertcorrected reference set rather than as out-of-distribution generalisation error; the morphometric distributions reported in the main article are computed at the ROI level on the full cross-sections and are independent of the tile-level split.

Image-disjoint splits were not used in this initial study because only three SEM cross-sections were available (one per aging condition), making a leave-onecondition-out split infeasible without losing condition coverage during training. Removing any single image from the training pool would deprive the model of one of the three target aging states and would prevent the validation set from sampling that condition altogether; the corresponding metric estimates would therefore conflate cross-condition generalisation error with the loss of indistribution training signal. A rotating image-disjoint sensitivity analysis, in which the model is retrained with each cross-section held out in turn and the remaining two used for both training and within-image validation, would require additional independent cross-sections per condition and is left for future work as more high-resolution SEM data become available.

## S6. SAM2 baseline details

The SAM2 (Hiera-Large, sam2.1) baseline was evaluated in two configurations. In automatic mode, a 64 × 64 dense point grid was used with no other prompts, and the union of SAM2’s returned masks was taken as the raw output. In prompted mode, point prompts were placed at the centroids of dark regions identified by Otsu thresholding (otsu\_factor = 1.5), with up to 128 prompts per tile. For each prompted point, the area-bounded candidate with the lowest mean intensity (candidate area $\leq 0 . 2 5$ of the tile) was selected from SAM2’s three multi-mask outputs. Geometric and intensity post-filters (mask elongation $\geq 2 . 0$ , solidity $\leq 0 . 8 5$ , mean intensit $\mathrm { y } \le 0 . 4 5 )$ were applied per tile in both configurations, and the higher of the with- and without-filter Dice was retained per tile.

The comparison was performed on a binary crack-versus-background target obtained by collapsing the three-class reference, mirroring SAM2’s class-agnostic output. A heuristic post-hoc width-based class assignment was considered and rejected, since it would conflate SAM2’s segmentation quality with the supervised classifier reported in the main article.

## S7. Statistical-analysis bookkeeping

The family of tests reported in the main article comprises 27 comparisons: 3 ROI subsets (intergranular, intragranular, all cracks) × 3 morphometric metrics (area fraction, mean width, geometric tortuosity) × 3 condition pairs (initial vs cycled-aged, initial vs calendar-aged, cycled-aged vs calendar-aged). Benjamini– Hochberg-adjusted q-values within this 27-test family appear as significance brackets in the morphometric figure of the main article (ROI counts: 194/138/98 for intergranular early + late, 446/610/623 for intragranular, initial / cycledaged / calendar-aged).

A separate 9-test sub-family is reported on the intergranular-late subset (97/69/49 ROIs), again Benjamini–Hochberg-adjusted within that sub-family. Both families are reported because the headline cycling-versus-baseline claim is tested under each scope and survives FDR adjustment at $q < 0 . 0 0 1$ in both.

Raw U-statistics, raw p-values, and adjusted q-values for all 27 comparisons are listed in the analysis bundle under fig5\_statistics\_pairwise\_stats.tsv in the released code repository.

## S8. Software environment and reproducibility

All models were trained and evaluated on a single NVIDIA A100 80 GB PCIe GPU (driver 550.163.01) under Ubuntu 22.04.5 LTS, using Python 3.10.13, Py-Torch 2.4.0 (CUDA 12.1, cuDNN 9.1.0), torchvision 0.19.0, PyTorch-Lightning 2.0.8, timm 0.9.6, scikit-image 0.25.2, SciPy 1.15.3, NumPy 1.26.4, and OpenCV

4.13. The frozen DINOv3 ViT-B/16 encoder fed a DeepLabV3+ decoder (ASPP dilation rates $6 / 1 2 / 1 8$ , low-level stride 2), giving 17.8 M trainable and 85.7 M frozen parameters.

The optimiser, learning-rate schedule, batch size, precision, and loss formulation are specified in Supplementary Section S1 (the Dice term used a smoothing constant ε=1). Online data augmentation was applied to the training split only; validation data were left unaltered. The training framework’s default pipeline was applied to every training sample, with each operation drawn independently at its own probability: random afine transforms (probability $0 . 5 ;$ rotation up to $\pm 4 5 ^ { \circ }$ , translation up to half the image extent, isotropic scaling in [0.9, 1.1]); random crop-and-resize (0.1; crop area fraction [0.6, 0.95]); horizontal and vertical flips (0.5 each); brightness and contrast jitter (0.1 each; factor [0.9, 1.1]); additive Gaussian noise (0.1; per-call mean in [0, 0.1] and standard deviation in [0, 0.2], clipped to [0, 1]); Gaussian blur (0.1; odd kernel size 3–7 px, $\sigma \in [ 0 . 1 , 2 . 0 ] )$ ; salt-and-pepper noise (0.1; salt and pepper fractions each in $\left[ 0 , 0 . 0 1 \right] )$ ; random pixel dropout (0.1; fraction in [0, 0.05]); and grid masking (0.1; up to 10 zeroed cells on a $1 0 \times 1 0 ~ \mathrm { g r i d } )$ . Geometric operations (afine, crop, flips) were applied jointly to the image and mask, and photometric operations to the image only; hue and saturation jitter present in the default configuration are inactive for the single-channel (greyscale) SEM input used here.

Inference and ROI extraction followed Supplementary Sections S2–S3 with the following operating values. Per-class probability maps were binarised at $\tau { = } 0 . 1$ . Particle ROIs were obtained by Otsu thresholding of the backscatter image after a $5 \times 5$ Gaussian pre-blur (no scaling factor), morphological closing with a $5 \times 5$ elliptical kernel, and a 50 px fragment-merge radius, retaining particle components with area ${ \geq } 1 0 , 0 0 0 \mathrm { p x }$ and crack components with area ${ \geq } 1 0 0 0 \mathrm { p x }$ and skeleton length $\geq 1 0 0 \mathrm { p x }$ , with zero ROI padding. Centrelines were extracted by Zhang–Suen thinning (skimage.morphology.skeletonize) and widths from the L2 distance transform as $w = 2 d .$ . The intensity-based background gating of Eqs. B.6–B.7 is an optional step that was disabled in the production pipeline. All code and trained weights are available as stated in the Code Availability section.

a  
![](images/7fe823f8af50ecacbf81b12c68d70250cf5fe59fba394089c18d766fb1698df7.jpg)

b  
![](images/c95fb47abbd1f3a959d619c0fc52aa0efef23e795de431f23a1ec9efc3f7312b.jpg)

c  
![](images/e846884232687f9d83adab8a744b440c90434d1ac82467cab76b4d719f9d733f.jpg)

d  
![](images/481a7c27c82dd4a79031c5c7d6dc1792988d28b9ee0c2c41b77acefd86ffba94.jpg)

![](images/d10ce0a27513839717164c33f2d605eb7c2f6a9b25a6db1e1c092b99f1ac316a.jpg)

![](images/6799363552f12bbec0b1e672b68062727152d1dbdfd7241a8023d20ee8fd63be.jpg)  
Supplementary Figure S1: Skeleton-based morphometric extraction. The skeletonbased extraction procedure applied within each ROI, illustrated for an example intergranular late ROI. (a) Raw SEM image of the region of interest. (b) Multi-class segmentation overlay fused with a semi-transparent highlight of the single-class crack mask (white: 16.1% of pixels, 17 666 px; black: 83.9%, 91 895 px). (c) Medial-axis skeleton overlaid on the raw SEM image, with branch points (orange) and endpoints (cyan) classified from the $3 \times 3$ neighbour-degree topology and the principal orientation (172.6°) derived from the leading eigenvector of the skeleton coordinate covariance matrix; the network comprises 74 branches and 30 endpoints, with tortuosity 5.39, anisotropy 1.51, and roughness $6 4 . 7 6 $ px. (d) Fused distance-transform map (cividis colour scale) and per-skeleton-pixel width heatmap $w ( x _ { s } ) = 2 d ( x _ { s } )$ (inferno colour scale), both overlaid on the raw ROI. (e) Crack width histogram (mean 10.69 px, median 10.39 px, std 4.43 px, range 2.00–27.57 px). (f) Empirical cumulative distribution function of crack width with $P _ { 2 5 } , P _ { 5 0 } ,$ , and $P _ { 7 5 }$ percentiles indicated. Together, panels $\mathrm { ( c ) - ( f ) }$ document that the pipeline reliably extracts centrelines even for branched, tortuous crack networks and yields stable width estimates along elongated crack segments.

![](images/69be3ccfa9ef3112bf9c7ae5bdf17e5e0ac89a690e239cd6851b6c24e72e228c.jpg)  
Supplementary Figure S2: Per-tile boundary-F1 distribution at 1 px and 2 px tolerance. Per-tile boundary-F1 (BF1) score distributions at 1-pixel and 2-pixel Euclidean boundary tolerance for each crack class. The median BF1 at 1-pixel tolerance (0.45–0.46 per class) rises to 0.58–0.62 at 2-pixel tolerance, approaching the clDice level. The improvement from 1 px to 2 px confirms that 1–2-pixel boundary misalignment dominates the moderate pixelexact Dice. For crack structures only 2–5 pixels wide at the imaging resolution of 22.3 nm/px, this boundary uncertainty corresponds to sub-50 nm localisation error, comparable to the precision achievable by human annotators tracing approximate crack centrelines.

![](images/3330cd2dceecb48ea87f19cf21a5ea8dcfba59598d21fede6493f6cea61f5e38.jpg)  
Supplementary Figure S3: Method-by-method diagnostic on tile 100\_0000\_00038 (the Fig. 1 sample tile). Row a (predictions): columns left-to-right are the raw tile, the 3-class expert reference (intragranular red, early intergranular green, late intergranular blue), SAM2-auto, SAM2-prompted, and fine-tuned DINOv3; each prediction cell is annotated with its binary Dice. Row b (error maps): same column order; cyan = true positive, magenta = false positive, amber = false negative, black = true negative. Row c (inputs): for each SAM2 column, the prompts the method sees: Otsu-derived dark mask overlay (column 1), Otsu centroids (column 2, SAM2-prompted inputs), the 64 × 64 dense point grid (column 3, SAM2-auto inputs), and one prompt zoom showing SAM2’s 3 multi-mask candidates with the area-bounded lowest-mean-intensity selection bordered in green (column 4); column 5 shows the DINOv3 sigmoid heatmap (maximum over the 3 crack classes) before binarisation, with the colour bar indicating P(any crack) per pixel.

d  
![](images/2f9b8c8db49f4f6b232fd6187ea2a8ac914ce298e53fd698d7cc6a4f700e67ef.jpg)

b  
![](images/e397b6345c7a2a1c923593e346c8e8da6e828b65b62431d552e0845dba63bb58.jpg)

![](images/08069018c0612e53167b0359d6287b3ddd9ec663042a724fe08f2207914d3606.jpg)

![](images/b9f2699c53ec8968e1e3d121799861472c5e9183a7bec5f29d9febc404c492dd.jpg)  
Supplementary Figure S4: Region-of-interest detection pipeline. The automated ROIextraction workflow that defines the spatial units used for the morphometric analysis in the main article, illustrated for the cycled-aged image (intergranular late class). (a) Binary threshold mask at decision threshold τ. (b) Post-merge mask after morphological closing merges spatially proximal crack fragments belonging to the same secondary particle. (c) Rejected sub-particle components below the minimum-area threshold, suppressed to remove artefacts. (d) Final detected ROI windows: each surviving component is enclosed by its axis-aligned bounding box, expanded by a fixed margin to ensure full particle coverage. The resulting ROI windows are sorted in raster order (top-to-bottom, left-to-right) and used as the spatial units for all downstream morphometric statistics reported in the main article.

![](images/3262548f44c51c67d0c4884246913aad9e95981c3bf36b941b286584488ff2b8.jpg)

b  
![](images/6f1cb03d40474152f70ffd38d377f62096962621c544c4d779c1f266e88adff0.jpg)

![](images/b2297c20217d8944681036d2ff51ffc211537b1ec1a889fbb0e2fb5655b11f37.jpg)

![](images/fdeddd1aa2c6427a4b724323e2a8e558d8b180f7fab453ca340b4e7a3a5def62.jpg)

e  
![](images/192924084f2c42dafd2944d0ed1dc0763b55a43178b1e734b2e74d24d1b7549c.jpg)  
f

![](images/d23be92d46fce6a6b224c3c9bb217cab5456302eaec457687c8a6611bfae7167.jpg)  
Supplementary Figure S5: Per-class segmentation metrics versus crack density (full panel set). Per-tile Dice (top row) and IoU (bottom row) plotted against the foreground pixel fraction for intragranular $( \mathbf { a } , \mathbf { d } ) ,$ intergranular early $( \mathbf { b } , \mathbf { e } ) .$ , and intergranular late $( \mathbf { c } , \mathbf { f } )$ cracks across all 482 expert-corrected tiles. Dashed lines show linear fits; Pearson correlation coeficients r are annotated in each panel $( r = 0 . 2 1 \mathrm { - } 0 . 3 2 )$ . The full multi-panel decomposition shown here supports the conclusion that pixel-exact agreement is principally limited by lowdensity tiles rather than by model failures on crack-rich regions, motivating the topology-aware and boundary-tolerant metrics (clDice, BF1) reported in the main article.

## References

[1] J. P. Pender, G. Jha, D. H. Youn, J. M. Ziegler, I. Andoni, E. J. Choi, A. Heller, B. S. Dunn, P. S. Weiss, R. M. Penner, C. B. Mullins, Electrode degradation in Li-ion batteries, ACS Nano 14 (2) (2020) 1243–1295. doi: 10.1021/acsnano.9b04365.

[2] J. S. Edge, S. O’Kane, R. Prosser, N. D. Kirkaldy, A. N. Patel, A. Hales, A. Ghosh, W. Ai, J. Chen, J. Yang, S. Li, M.-C. Pang, L. Bravo Diaz, A. Tomaszewska, M. W. Marzook, K. N. Radhakrishnan, H. Wang, Y. Patel, B. Wu, G. J. Ofer, Lithium-ion battery degradation: what you need to know, Phys. Chem. Chem. Phys. 23 (14) (2021) 8200–8221. doi: 10.1039/D1CP00359C.

[3] P. Yan, J. Zheng, M. Gu, J. Xiao, J.-G. Zhang, C.-M. Wang, Intragranular cracking as a critical barrier for high-voltage usage of layer-structured cathode for Li-ion batteries, Nature Communications 8 (2017) 14101. doi:10.1038/ncomms14101.

[4] T. M. M. Heenan, A. Wade, C. Tan, J. E. Parker, D. Matras, A. S. Leach, J. B. Robinson, A. Llewellyn, A. Dimitrijevic, R. Jervis, P. D. Quinn, D. J. L. Brett, P. R. Shearing, Identifying the origins of microstructural defects such as cracking within Ni-rich NMC811 cathode particles for lithium-ion batteries, Advanced Energy Materials 10 (47) (2020) 2002655. doi:10.1002/aenm.202002655.

[5] S. E. J. O’Kane, W. Ai, G. Madabattula, D. Alonso Alvarez, R. Timms, V. Sulzer, J. S. Edge, B. Wu, G. J. Ofer, M. Marinescu, Lithium-ion battery degradation: how to model it, Phys. Chem. Chem. Phys. 24 (13) (2022) 7909–7922. doi:10.1039/D2CP00417H.

[6] S. Navidi, A. Thelen, T. Li, C. Hu, Physics-informed machine learning for battery degradation diagnostics: A comparison of state-of-the-art methods, Energy Storage Materials 68 (2024) 103343. doi:10.1016/j.ensm.2024. 103343.

[7] L. Britala, M. Marinaro, G. Kucinskis, A review of the degradation mechanisms of NCM cathodes and corresponding mitigation strategies, Journal of Energy Storage 73 (2023) 108875. doi:10.1016/j.est.2023.108875.

[8] T. Li, X.-Z. Yuan, L. Zhang, D. Song, K. Shi, C. Bock, Degradation mechanisms and mitigation strategies of nickel-rich NMC-based lithiumion batteries, Electrochemical Energy Reviews 3 (1) (2020) 43–80. doi: 10.1007/s41918-019-00053-3.

[9] J. K. Morzy, W. M. Dose, P. E. Vullum, M. C. Lai, A. Mahadevegowda, M. F. L. De Volder, C. Ducati, Origins and importance of intragranular cracking in layered lithium transition metal oxide cathodes, ACS Applied Energy Materials 7 (9) (2024) 3945–3956. doi:10.1021/acsaem.4c00279.

[10] C. Xu, K. Märker, J. Lee, A. Mahadevegowda, P. J. Reeves, S. J. Day, M. F. Groh, S. P. Emge, C. Ducati, B. L. Mehdi, C. C. Tang, C. P. Grey, Bulk fatigue induced by surface reconstruction in layered Ni-rich cathodes for Li-ion batteries, Nature Materials 20 (1) (2021) 84–92. doi:10.1038/ s41563-020-0767-8.

[11] R. Jung, M. Metzger, F. Maglia, C. Stinner, H. A. Gasteiger, Oxygen release and its efect on the cycling stability of $\mathrm { \bar { \ t i N i _ { \it x } M n _ { \it y } C o _ { \it z } \bar { O } _ { \it 2 } } }$ (NMC) cathode materials for Li-ion batteries, Journal of The Electrochemical Society 164 (7) (2017) A1361–A1377. doi:10.1149/2.0021707jes.

[12] S. Lee, L. Su, A. Mesnier, Z. Cui, A. Manthiram, Cracking vs. surface reactivity in high-Ni cathodes for Li-ion batteries, Joule 7 (11) (2023) 2430– 2444. doi:10.1016/j.joule.2023.09.006.

[13] J. Hu, H. Wang, B. Xiao, P. Liu, T. Huang, Y. Li, X. Ren, Q. Zhang, J. Liu, X. Ouyang, X. Sun, Challenges and approaches of single-crystal Nirich layered cathodes in lithium batteries, National Science Review 10 (12) (2023) nwad252. doi:10.1093/nsr/nwad252.

[14] Y. Chen, Y. Yao, Z. Yao, W. Li, X. Shen, J. Song, W. Luan, H. Chen, S.-t. Tu, K. Wu, Fracture behaviour of NCM polycrystalline particles in lithium-ion batteries under extreme conditions, Nano Energy 141 (2025) 111104. doi:10.1016/j.nanoen.2025.111104.

[15] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, P. Bojanowski, Dinov3, arXiv preprint arXiv:2508.10104 (2025). arXiv:2508.10104, doi:10.48550/arXiv.2508.10104. URL https://arxiv.org/abs/2508.10104

[16] D. Karimi, S. K. Warfield, A. Gholipour, Transfer learning in medical image segmentation: New insights from analysis of the dynamics of model parameters and learned representations, Artificial Intelligence in Medicine 116 (2021) 102078. doi:10.1016/j.artmed.2021.102078.

[17] T. Tegetmeyer-Kleine, H. Ditler, G. Stahl, C. Rahe, D. U. Sauer, W. Li, Deep learning for decoding lithium plating in lithium-ion batteries from post-mortem images, Journal of Energy Storage 154 (2026) 121028. doi: 10.1016/j.est.2026.121028.

[18] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P.-Y. Huang, S.-W. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, P. Bojanowski, Dinov2: Learning robust visual features without supervision, arXiv (2023). arXiv:2304.07193, doi:10.48550/arXiv.

[19] Y. Chu, L. Zhou, G. Luo, K. Kang, S. Dong, Z. Han, L. Wu, X. Meng, C. Yang, X. Guo, Y. Cheng, Y. Qi, X. Liu, D. Xie, Y. Li, R. Henao, X. Xiao, S. Cao, G. Setti, Z. Qiu, X. Gao, HorusEye: a self-supervised foundation model for generalizable X-ray tomography restoration, Nature Computational Science 6 (4) (2026) 372–387. doi:10.1038/s43588-026-00973-3.

[20] T. Fu, F. Monaco, J. Li, K. Zhang, Q. Yuan, P. Cloetens, P. Pianetta, Y. Liu, Deep-learning-enabled crack detection and analysis in commercial Li-ion battery cathodes, Advanced Functional Materials 32 (39) (2022) 2203070. doi:10.1002/adfm.202203070.

[21] S. Müller, C. Sauter, R. Shunmugasundaram, N. Wenzler, V. De Andrade, F. De Carlo, E. Konukoglu, V. Wood, Deep learning-based segmentation of Li-ion battery microstructures enhanced by artificially generated electrodes, Nature Communications 12 (1) (2021) 6205. doi: 10.1038/s41467-021-26480-9.

[22] A. M. Boyce, E. Martínez-Pañeda, A. Wade, Y. S. Zhang, J. J. Bailey, T. M. M. Heenan, D. J. L. Brett, P. R. Shearing, Cracking predictions of lithium-ion battery electrodes by x-ray computed tomography and modelling, Journal of Power Sources 526 (2022) 231119. doi: 10.1016/j.jpowsour.2022.231119.

[23] M. Majurski, P. Bajcsy, Exact tile-based segmentation inference for images larger than GPU memory, Journal of Research of the National Institute of Standards and Technology 126 (2021) 126009. doi:10.6028/jres.126. 009.

[24] O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, in: N. Navab, J. Hornegger, W. M. Wells, A. F. Frangi (Eds.), Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, Vol. 9351 of Lecture Notes in Computer Science, Springer, Cham, 2015, pp. 234–241. doi:10.1007/978-3-319-24574-4\_28. URL https://link.springer.com/chapter/10.1007/ 978-3-319-24574-4\_28

[25] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, J. Sun, Unified perceptual parsing for scene understanding, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 432–448. doi:10.1007/978-3-030-01228-1\_26. URL https://openaccess.thecvf.com/content\_ECCV\_2018/html/ Tete\_Xiao\_Unified\_Perceptual\_Parsing\_ECCV\_2018\_paper.html

[26] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick, Segment

anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 3992–4003. doi:10.1109/ICCV51070. 2023.00371.

[27] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.- Y. Wu, R. Girshick, P. Dollár, C. Feichtenhofer, SAM 2: Segment anything in images and videos, arXiv (2024). arXiv:2408.00714, doi:10.48550/ arXiv.2408.00714. URL https://arxiv.org/abs/2408.00714

[28] C. Stringer, T. Wang, M. Michaelos, M. Pachitariu, Cellpose: a generalist algorithm for cellular segmentation, Nature Methods 18 (1) (2021) 100–106. doi:10.1038/s41592-020-01018-x.

[29] L. Chen, Y. Zhu, G. Papandreou, F. Schrof, H. Adam, Encoder–decoder with atrous separable convolution for semantic image segmentation, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 833–851. doi:10.1007/978-3-030-01234-2\_49. URL https://openaccess.thecvf.com/content\_ECCV\_2018/papers/ Liang-Chieh\_Chen\_Encoder-Decoder\_with\_Atrous\_ECCV\_2018\_paper. pdf

[30] L. Joskowicz, D. Cohen, N. Caplan, J. Sosna, Inter-observer variability of manual contour delineation of structures in CT, European Radiology 29 (3) (2019) 1391–1399. doi:10.1007/s00330-018-5695-5.

[31] L. Maier-Hein, A. Reinke, P. Godau, M. D. Tizabi, F. Buettner, E. Christodoulou, B. Glocker, F. Isensee, J. Kleesiek, M. Kozubek, M. Reyes, M. A. Riegler, M. Wiesenfarth, A. E. Kavur, C. H. Sudre, M. Baumgartner, M. Eisenmann, D. Heckmann-Nötzel, T. Rädsch, L. Acion, M. Antonelli, T. Arbel, S. Bakas, A. Benis, M. B. Blaschko, M. J. Cardoso, V. Cheplygina, B. A. Cimini, G. S. Collins, K. Farahani, L. Ferrer, A. Galdran, B. van Ginneken, R. Haase, D. A. Hashimoto, M. M. Hofman, M. Huisman, P. Jannin, C. E. Kahn, D. Kainmueller, B. Kainz, A. Karargyris, A. Karthikesalingam, F. Kofler, A. Kopp-Schneider, A. Kreshuk, T. Kurc, B. A. Landman, G. Litjens, A. Madani, K. Maier-Hein, A. L. Martel, P. Mattson, E. Meijering, B. Menze, K. G. M. Moons, H. Müller, B. Nichyporuk, F. Nickel, J. Petersen, N. Rajpoot, N. Rieke, J. Saez-Rodriguez, C. I. Sánchez, S. Shetty, M. van Smeden, R. M. Summers, A. A. Taha, A. Tiulpin, S. A. Tsaftaris, B. Van Calster, G. Varoquaux, P. F. Jäger, Metrics reloaded: Recommendations for image analysis validation, Nature Methods 21 (2) (2024) 195–212. doi:10.1038/s41592-023-02151-z.

[32] S. Nikolov, S. Blackwell, A. Zverovitch, R. Mendes, M. Livne, J. De Fauw, Y. Patel, C. Meyer, H. Askham, B. Romera-Paredes, C. Kelly, A. Karthikesalingam, C. Chu, D. Carnell, C. Boon, D. D’Souza, S. A. Moinuddin,

B. Garie, Y. McQuinlan, S. Ireland, K. Hampton, K. Fuller, H. Montgomery, G. Rees, M. Suleyman, T. Back, C. O. Hughes, J. R. Ledsam, O. Ronneberger, Clinically applicable segmentation of head and neck anatomy for radiotherapy: Deep learning algorithm development and validation study, Journal of Medical Internet Research 23 (7) (2021) e26151. doi:10.2196/26151.

[33] S. Shit, J. C. Paetzold, A. Sekuboyina, I. Ezhov, A. Unger, A. Zhylka, J. P. W. Pluim, U. Bauer, B. H. Menze, clDice – a novel topologypreserving loss function for tubular structure segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 16560–16569. doi:10.1109/CVPR46437.2021.01629.

[34] A. O. Kondrakov, A. Schmidt, J. Xu, H. Geßwein, R. Mönig, P. Hartmann, H. Sommer, T. Brezesinski, J. Janek, Anisotropic lattice strain and mechanical degradation of high- and low-nickel NCM cathode materials for Li-ion batteries, The Journal of Physical Chemistry C 121 (6) (2017) 3286–3294. doi:10.1021/acs.jpcc.6b12885.

[35] H.-H. Ryu, K.-J. Park, C. S. Yoon, Y.-K. Sun, Capacity fading of Ni-rich $\mathrm { L i } [ \mathrm { N i } _ { x } \mathrm { C o } _ { y } \mathrm { M n } _ { 1 - x - y } ] \mathrm { O } _ { 2 } \ ( 0 . 6 \leq \mathrm { x } \leq 0 . 9 5 )$ cathodes for high-energy-density lithium-ion batteries: Bulk or surface degradation?, Chemistry of Materials 30 (3) (2018) 1155–1163. doi:10.1021/acs.chemmater.7b05269.

[36] L. de Biasi, B. Schwarz, T. Brezesinski, P. Hartmann, J. Janek, H. Ehrenberg, Between scylla and charybdis: Balancing among structural stability and energy density of layered NCM cathode materials for advanced lithium-ion batteries, The Journal of Physical Chemistry C 121 (47) (2017) 26163–26171. doi:10.1021/acs.jpcc.7b06363.

[37] P. Yan, J. Zheng, J. Liu, B. Wang, X. Cheng, Y. Zhang, X. Sun, C. Wang, J.-G. Zhang, Tailoring grain boundary structures and chemistry of Ni-rich layered cathodes for enhanced cycle stability of lithium-ion batteries, Nature Energy 3 (2018) 600–605. doi:10.1038/s41560-018-0191-3.

[38] C. Liu, F. Roters, D. Raabe, Role of grain-level chemo-mechanics in composite cathode degradation of solid-state lithium batteries, Nature Communications 15 (1) (2024) 7970. doi:10.1038/s41467-024-52123-w.

[39] L. R. Brandt, J.-J. Marie, T. Moxham, D. P. Förstermann, E. Salvati, C. Besnard, C. Papadaki, Z. Wang, P. G. Bruce, A. M. Korsunsky, Synchrotron x-ray quantitative evaluation of transient deformation and damage phenomena in a single nickel-rich cathode particle, Energy & Environmental Science 13 (2020) 3556–3566. doi:10.1039/D0EE02290J.

[40] Y. Du, K. Fujita, S. Shironita, Y. Sone, E. Hosono, D. Asakura, M. Umeda, Capacity fade characteristics of nickel-based lithium-ion secondary battery after calendar deterioration at $8 0 ~ ^ { \circ } \mathrm { c } .$ , Journal of Power Sources 501 (2021) 230005. doi:10.1016/j.jpowsour.2021.230005.

[41] Z. Mao, M. Farkhondeh, M. Pritzker, M. Fowler, Z. Chen, Calendar aging and gas generation in commercial graphite/NMC-LMO lithium-ion pouch cell, Journal of The Electrochemical Society 164 (14) (2017) A3469–A3483. doi:10.1149/2.0241714jes.

[42] R. Jung, M. Metzger, F. Maglia, C. Stinner, H. A. Gasteiger, Chemical versus electrochemical electrolyte oxidation on NMC111, NMC622, NMC811, LNMO, and conductive carbon, The Journal of Physical Chemistry Letters 8 (19) (2017) 4820–4825. doi:10.1021/acs.jpclett.7b01927.

[43] R. Schäfer, T. Nicke, H. Höfener, A. Lange, D. Merhof, F. Feuerhake, V. Schulz, J. Lotz, F. Kiessling, Overcoming data scarcity in biomedical imaging with a foundational multi-task model, Nature Computational Science 4 (7) (2024) 495–509. doi:10.1038/s43588-024-00662-z.

[44] J. Li, A. R. Cameron, H. Li, S. L. Glazier, D. Xiong, M. Chatzidakis, J. Allen, G. A. Botton, J. R. Dahn, Comparison of single crystal and polycrystalline $\mathrm { L i N i _ { 0 . 5 } M n _ { 0 . 3 } C o _ { 0 . 2 } O _ { 2 } }$ positive electrode materials for high voltage li-ion cells, Journal of The Electrochemical Society 164 (7) (2017) A1534–A1544. doi:10.1149/2.0991707jes.

[45] G. Qian, Y. Zhang, L. Li, R. Zhang, J. Xu, Z. Cheng, S. Xie, H. Wang, Q. Rao, Y. He, Y. Shen, L. Chen, M. Tang, Z.-F. Ma, Single-crystal nickelrich layered-oxide battery cathode materials: Synthesis, electrochemistry, and intra-granular fracture, Energy Storage Materials 27 (2020) 140–149. doi:10.1016/j.ensm.2020.01.027.

[46] Z. Li, Y. Wang, J. Wang, C. Wu, W. Wang, Y. Chen, C. Hu, K. Mo, T. Gao, Y.-S. He, Z. Ren, Y. Zhang, X. Liu, N. Liu, L. Chen, K. Wu, C. Shen, Z.-F. Ma, L. Li, Gradient-porous-structured Ni-rich layered oxide cathodes with high specific energy and cycle stability for lithium-ion batteries, Nature Communications 15 (1) (2024) 10216. doi:10.1038/ s41467-024-54637-9.

[47] Q. Lin, W. Guan, J. Zhou, J. Meng, W. Huang, T. Chen, Q. Gao, X. Wei, Y. Zeng, J. Li, Z. Zhang, Ni–Li anti-site defect induced intragranular cracking in ni-rich layer-structured cathode, Nano Energy 76 (2020) 105021. doi:10.1016/j.nanoen.2020.105021.

[48] L. Wasserman, All of Statistics: A Concise Course in Statistical Inference, Springer, New York, 2004. doi:10.1007/978-0-387-21736-9.

[49] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7

[50] N. Otsu, A threshold selection method from gray-level histograms, IEEE Transactions on Systems, Man, and Cybernetics 9 (1) (1979) 62–66. doi: 10.1109/TSMC.1979.4310076.