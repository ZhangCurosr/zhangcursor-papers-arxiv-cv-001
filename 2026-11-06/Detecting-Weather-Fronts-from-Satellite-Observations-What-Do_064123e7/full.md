# Detecting Weather Fronts from Satellite Observations: What Does Reanalysis Add?

Ioannis Dravilas<sup>∗</sup> University of Amsterdam

Oriol Hinojo Comellas EUMETSAT

Miruna Stoicescu EUMETSAT

Sina Montazeri EUMETSAT

Roope Tervo EUMETSAT

Andrea Meraner EUMETSAT

Carlos Horn EUMETSAT

Ana Lucic University of Amsterdam

## Abstract

Weather fronts are transition regions between air masses with contrasting thermodynamic properties and are associated with mid-latitude precipitation and severe weather. We investigate machine learning (ML) methods for detecting weather fronts and quantify the contribution of reanalysis data to performance. We compare two configurations of the same ML model for pixel-wise frontal classification over the North Atlantic and Europe: (1) using only geostationary infrared satellite observations, and (2) additionally using 28 ERA5 reanalysis fields. The satelliteonly model achieves a neighbourhood critical success index at a 150 km tolerance (CSI@150) of 0.402, which lies within the range of agreement between analyses from different meteorological services. The satellite+ERA5 model achieves 0.575, an improvement of 0.173. Our analysis shows that low-level winds are the least predictable from the satellite representation (R<sup>2</sup> = 0.23, compared with 0.88 for temperature), while removing winds or moisture causes the largest decreases in detection performance.

## 1 Introduction

Weather fronts are narrow transition regions where air masses with contrasting thermodynamic properties meet, causing sharp changes in temperature, humidity, wind, and pressure [Hewson, 1998, Biard and Kunkel, 2019]. Detecting fronts matters for both weather forecasting and climate analysis, because a substantial fraction of extreme mid-latitude precipitation occurs near them [Catto and Pfahl, 2013]. Operational forecasters identify fronts manually [Biard and Kunkel, 2019, Justin et al., 2025] from surface observations, satellite imagery, and numerical weather prediction (NWP) output. Independent meteorological services often analyse the same synoptic situation differently, because their local information, analysis conventions, and regional priorities differ. Automated methods, from thermodynamic diagnostics on gridded fields [Hewson, 1998, Schemm et al., 2015] to deep learning models [Biard and Kunkel, 2019, Lagerquist et al., 2019, Niebler et al., 2022, Justin et al., 2025], offer objective and reproducible alternatives. Most rely on reanalysis, which combines historical observations with an NWP model to reconstruct past atmospheric states. We study the automated pixel-wise classification of cold, warm, and occluded fronts over the North Atlantic and Europe.

Detectors that rely on reanalysis or NWP fields lag behind the latest observations. Reanalyses such as ERA5 [Hersbach et al., 2020] become available after several days and operational NWP analyses typically after a few hours, while real-time forecasts contain no observations made after their initialisation. Geostationary satellites, in contrast, observe the atmosphere several times per hour, so a satellite-only detector can run in near real time. However, infrared radiances do not directly measure the temperature, humidity, and wind fields that characterise frontal zones. Reanalyses estimate these fields by data assimilation. This trade-off motivates our question: what does reanalysis add beyond satellite observations for frontal detection? We answer it by comparing satellite-only and satellite+ERA5 detectors through cross-modal probing, variable-group ablation, and single-group addition.

## 2 Related work

Traditional objective frontal detection applies rule-based thermodynamic diagnostics to gridded atmospheric fields [Hewson, 1998, Schemm et al., 2015]. The NWC SAF Automatic Satellite Image Interpretation (ASII) product [NWC SAF, 2022] detects fronts in geostationary imagery with rule-based morphological analysis, optionally adding NWP fields, and is our reference baseline.

Most deep learning approaches rely on reanalysis. Biard and Kunkel [2019] and Lagerquist et al. [2019] trained convolutional neural networks to detect atmospheric fronts over North America using reanalysis variables. Niebler et al. [2022] used multi-level ERA5 over Europe and North America and introduced label deformation for positional uncertainty, while Justin et al. [2025] developed FrontFinder AI (UNet3+), which is deployed operationally at NOAA. None of these studies evaluates a satellite-only configuration. Qin et al. [2025] combined geostationary imagery with ERA5-derived fields to detect cold fronts over East Asia, but did not consider other front types or isolate the contribution of ERA5 over satellite observations alone.

Prior work therefore evaluates a fixed input configuration [Biard and Kunkel, 2019, Lagerquist et al., 2019, Niebler et al., 2022, Justin et al., 2025, Qin et al., 2025], sometimes ranking individual variables by importance [Justin et al., 2025]. We instead compare satellite-only and satellite+ERA5 models directly and identify which ERA5 variables add information beyond the satellite representation. This matters when designing detectors capable of operating in near real time without relying on reanalysis data.

## 3 Methodology

## 3.1 Problem formulation

Given geostationary satellite observations over the North Atlantic and Europe, optionally augmented with ERA5 reanalysis fields, we classify each grid point into one of four classes: cold front, warm front, occluded front, or no front (example in Appendix F). We use UK Met Office surface-analysis charts as ground truth because they are available every six hours over the 2019–2024 study period and their graphical conventions allow more reliable automated digitisation than the charts of some other services. We use data from four additional services as independent baselines (Appendix B).

We train two models with a Swin-UNet [Liu et al., 2021, Cao et al., 2023] inspired by Aurora [Bodnar et al., 2025] (Appendix A) that differ only in their inputs. These are the single-service variants of the SATFRONT detector [Dravilas et al., 2026], trained only on UK Met Office labels. The main SATFRONT model instead uses labels from five services. The satellite-only model (OBS) uses 15 satellite-derived channels: eight infrared channels (3.8–13.3 µm) from the GEO-Ring geostationary data record [EUMETSAT, 2026] at approximately 5 km resolution on the equal-area HEALPix grid, and seven deterministic features derived from the brightness temperatures following the ASII scheme [NWC SAF, 2022]. The satellite+ERA5 model (HYB) adds 28 ERA5 fields [Hersbach et al., 2020]: six atmospheric variables at four pressure levels (1000–700 hPa) and four additional surface variables. For analysis, we group the ERA5 variables into winds (10 fields), moisture (8), temperature (5), and mass (5). Appendix A lists all input channels. Because ERA5 becomes available only after several days, we use HYB as a diagnostic upper bound on what reanalysis adds to this detector, not as an intended operational model. We train on 2019–2022, validate on 2023, and test on 2024.

## 3.2 Experimental setup

We first compare the satellite-only and satellite+ERA5 models, then use three analyses to identify which ERA5 variables explain the performance difference: (i) cross-modal probing tests which ERA5 fields the satellite-only encoder can already predict, (ii) variable-group ablation measures the effect of removing each ERA5 group from HYB, and (iii) single-group addition measures the gain from adding each ERA5 group alone to OBS.

Cross-modal probing. We freeze the OBS encoder and extract features from each of its four hierarchical stages. Successive stages reduce spatial resolution and increase channel dimension. At each stage, we fit per-pixel ridge regression to predict every ERA5 field from the encoder features. Because variables may be best resolved at different scales, we take each field’s highest coefficient of determination $( R ^ { 2 } )$ on the 2024 test set across the four stages and average it within each variable group. To separate what training adds from what the inputs or architecture provide, we repeat the procedure with the raw satellite input and a randomly initialised encoder (Appendix C).

Variable-group ablation. For each variable group, we replace its HYB input channels with values from a randomly selected date and measure the change in CSI@150 relative to the unmodified HYB model. The replacement keeps each field’s spatial distribution but breaks its temporal alignment with the frontal labels.

![](images/17780e1c03cc73acc759ff7a65650e91ff2546b84582fa74ef1f0a54edffae4c.jpg)

![](images/9a97bb17c75acd89c85036a9784b9086a13803169326467d3df8e2838e8739ea.jpg)  
Figure 1: Analysis of ERA5 variable groups. Top: linear predictability $( R ^ { 2 } )$ from the frozen satellite-only encoder (best stage) and gain over a randomly initialised encoder, with 95% CIs. Near-sfc: 1000/925 hPa and 10 m fields. Dashed: positive control. Bottom: drop in CSI@150 when a group is replaced by values from another date.

Single-group addition. We train three additional models, each combining the satellite inputs with one ERA5 variable group, using the same architecture and training procedure as OBS and HYB. We add temperature and mass as one group because, by the hypsometric equation, the geopotential thickness between two pressure levels is set by the mean virtual temperature of the layer [Holton and Hakim, 2013]. This grouping also limits the number of new training runs. We measure each model’s paired CSI@150 gain over OBS.

Evaluation. We report the critical success index at a 150 km tolerance (CSI@150), following Justin et al. [2025]. A ground-truth front pixel counts as detected if a predicted pixel of the same class lies within 150 km, and a predicted pixel counts as a false positive if no ground-truth pixel of the same class lies within 150 km. We compute CSI per class and average over the three classes. We estimate 95% confidence intervals for the difference $\Delta$ between two models (e.g., HYB − OBS) with a paired moving-block bootstrap (Appendix C) and report $P ( \Delta \leq 0 )$ , the fraction of bootstrap samples with zero or negative improvement. Every paired comparison with OBS, HYB, or the random encoder in Section 4 has $P ( \bar { \Delta } \leq 0 ) < 1 0 ^ { - 3 }$ . All results use a single training seed. We compare the models with the operational ASII product [NWC SAF, 2022] and with the agreement between analyses from different meteorological services, measured against the same ground truth (Appendix B).

## 4 Results and discussion

Comparison of satellite-only and satellite+ERA5 models. Table 1 compares OBS and HYB with two reference baselines (Appendix B). OBS reaches $\mathrm { C S I @ 1 5 0 } = 0 . 4 0 2$ , approximately three times the ASII score (0.143). The OBS score also lies within the range of agreement between analyses from different meteorological services (0.277 to 0.501), and Met Éireann agrees most closely with the UK Met Office. These comparisons provide context, not an upper bound. The services differ in local information, conventions, and geographic priorities, whereas we train and score OBS on the UK Met Office convention. HYB reaches 0.575 (∆ = +0.173, 95% CI [0.163, 0.183]), a 43% improvement over OBS. The gain is similar across front types $( + 0 . 1 6 4 , + 0 . 1 7 4$ , and +0.180 for cold, warm, and occluded fronts) and comes mainly from higher probability of detection (0.505 to 0.718).

Contribution of ERA5 variable groups. In the probe (Figure 1, top), temperature is the ERA5 group that the frozen OBS encoder predicts best $\bar { ( R ^ { 2 } } = 0 . \bar { 8 } 8 )$ , followed by moisture (0.60) and mass (0.59), while winds are the least predictable $( R ^ { 2 } = 0 . 2 3 )$ . High temperature predictability is consistent with the direct thermal information in infrared radiances, whereas the encoder can infer wind only indirectly from cloud and moisture patterns [Velden et al., 2005]. A randomly initialised encoder already shows the same ranking $( R ^ { 2 } \stackrel { . } { = } 0 . 6 9$ for temperature and 0.05 for winds). Training improves all groups by +0.18 to +0.28 but does not close the gap for winds.

Table 1: CSI@150 on the 2024 test year (Met Office ground truth). Overall is the mean over the three front types; brackets give 95% CIs. All per-class improvements exclude zero $( P ( \Delta \leq 0 ) < 1 0 ^ { - 3 } )$ ).
<table><tr><td></td><td>Overall</td><td>Cold</td><td>Warm</td><td>Occluded</td></tr><tr><td>Operational ASII product (satellite+NWP)</td><td>0.143</td><td>0.156</td><td>0.109</td><td>0.163</td></tr><tr><td>Closest human analysis (Met Éireann)</td><td>0.501</td><td>0.543</td><td>0.454</td><td>0.508</td></tr><tr><td>OBS (satellite-only)</td><td>0.402 [0.389, 0.414]</td><td>0.450</td><td>0.359</td><td>0.398</td></tr><tr><td>HYB (satellite+ERA5)</td><td>0.575 [0.563, 0.585]</td><td>0.615</td><td>0.532</td><td>0.578</td></tr><tr><td>ERA5 improvement ∆</td><td>+0.173 [0.163, 0.183]</td><td>+0.164</td><td>+0.174</td><td>+0.180</td></tr></table>

In the ablation (Figure 1, bottom), replacing winds or moisture lowers CSI@150 by 0.21 and 0.31, whereas replacing temperature or mass causes much smaller decreases (0.04 and 0.03). Replacing one group also breaks its correlations with the other groups, such as the thermal-wind relationship between winds and temperature. We therefore compare the wind-and-moisture block with the temperature-and-mass block and do not rank the groups within each block.

In the single-group addition experiments (Appendix D), adding each group alone to OBS yields similar gains: +0.118 for winds, +0.138 for moisture, and +0.132 for temperature and mass. Temperature and mass have little effect when ablated but provide comparable gains when added alone, consistent with strong correlations among the ERA5 groups. Winds are the clearest case in which low predictability from the satellite encoder coincides with a large ablation effect.

Effect on detection characteristics. The improvement from ERA5 grows with the tolerance, from +0.116 at 50 km to +0.173 at 150 km, while the gain in probability of detection stays nearly constant (Appendix E). This pattern is consistent with HYB detecting fronts that OBS misses. OBS also partly reproduces the annual Met Office frontal-frequency map (Pearson correlation 0.37), and ERA5 raises this correlation to 0.53 (Figure 2). Among front types, occluded fronts show the largest improvement in front placement in individual analyses (+0.180), but ERA5 reduces the mean-squared error (MSE) of their annual frequency map by only 5.5%. Cold and warm fronts improve on both measures, with MSE reductions of 47% and 39% for the single-service models used here. With the five-service models of Dravilas et al. [2026], the cold-front MSE instead rises by 3.5%.

![](images/f34965ab3d544690537bc23025733c2a77a58afa0a7f6668531a30d8e101daa5.jpg)

![](images/fa8777a53e0919020373c028a68b00d125225e5c449f47a5406fc6382eff6338.jpg)

Limitations and conclusion. Our results show that satellite observations alone provide substantial information for frontal detection, while adding ERA5 raises CSI@150 from 0.402 to 0.575. Our analyses help explain this gap. The probe and the ablation agree most clearly on low-level winds. Winds are the ERA5 variables least predictable from the satellite representation and, together with moisture, those whose replacement most reduces performance. For occluded fronts, ERA5 strongly improves detection on individual charts (CSI@150) but barely reduces the error of the annual frequency map, consistent with reanalysis providing instantaneous dynamical information that infrared imagery does not directly measure.

![](images/52f3d18798744d3d42ebff07775c8205b02143d4259b4c9929bae22c6d88f160.jpg)  
Figure 2: Annual frontal frequency (2024, all front types, smoothed at ∼150 km for display).

These conclusions remain subject to several limitations. Linear probing measures which information the representation encodes, not what

the decoder actually uses [Ravichander et al., 2021], and our ablations disrupt correlations among ERA5 variables. We also evaluate one domain, architecture, and training seed. Nevertheless, crossmodal probing of a frozen unimodal encoder can estimate how much of a candidate modality the existing representation already predicts, before any multi-modal model is trained. The same approach could be applied to other combinations of observations and variables in Earth observation.

## Acknowledgments and Disclosure of Funding

We thank the Finnish Meteorological Institute (FMI) for providing its frontal analyses. We also acknowledge the German Weather Service (DWD), the Royal Netherlands Meteorological Institute (KNMI), the Irish Meteorological Service (Met Éireann), and the UK’s national meteorological service (UK Met Office), whose surface-analysis charts we used in this work. We acknowledge EUMETSAT, Destination Earth, and the European Weather Cloud for providing compute and storage resources. The authors declare no competing interests.

## References

James C Biard and Kenneth E Kunkel. Automated detection of weather fronts using a deep learning neural network. Adv. Stat. Climatol. Meteorol. Oceanogr., 5(2):147–160, November 2019.

Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A Weyn, Haiyu Dong, Jayesh K Gupta, Kit Thambiratnam, Alexander T Archibald, Chun-Chieh Wu, Elizabeth Heider, Max Welling, Richard E Turner, and Paris Perdikaris. A foundation model for the Earth system. Nature, 641(8065):1180–1187, May 2025.

Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, and Manning Wang. Swin-Unet: Unet-like pure transformer for medical image segmentation. In Computer Vision – ECCV 2022 Workshops, volume 13803 of Lecture Notes in Computer Science, pages 205–218. Springer, 2023. doi: 10.1007/978-3-031-25066-8\_9.

J L Catto and S Pfahl. The importance of fronts for extreme precipitation. J. Geophys. Res. Atmos., 118(19):10,791–10,801, October 2013.

Ioannis Dravilas, Miruna Stoicescu, Roope Tervo, Carlos Horn, Oriol Hinojo Comellas, Sina Montazeri, Andrea Meraner, and Ana Lucic. SATFRONT: Detecting atmospheric fronts with satellite data using machine learning. arXiv:2610.42111 [physics.ao-ph], October 2026. URL https://arxiv.org/abs/2610.42111.

EUMETSAT. GEO-Ring test data. User guide, EUMETSAT User Portal, 2026. URL https: //user.eumetsat.int/resources/user-guides/geo-ring-test-data. Accessed 2026.

Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, András Horányi, Joaquín Muñoz-Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla, Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean Bidlot, Massimo Bonavita, Giovanna De Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger, Sean Healy, Robin J Hogan, Elías Hólm, Marta Janisková, Sarah Keeley, Patrick Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia de Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-Noël Thépaut. The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., 146(730): 1999–2049, July 2020.

T D Hewson. Objective fronts. Meteorol. Appl., 5(1):37–65, March 1998.

James R. Holton and Gregory J. Hakim. An Introduction to Dynamic Meteorology. Academic Press, 5th edition, 2013. doi: 10.1016/C2009-0-63394-8.

Andrew D Justin, Amy McGovern, and John T Allen. FrontFinder AI: Efficient identification of frontal boundaries over the continental United States and NOAA’s unified surface analysis domain using the UNET3+ model architecture. Artif. Intell. Earth Syst., 4(1), January 2025.

Ryan Lagerquist, Amy McGovern, and David John Gagne, II. Deep learning for spatially explicit prediction of synoptic-scale fronts. Weather Forecast., 34(4):1137–1160, August 2019.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision transformer using shifted windows. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9992–10002. IEEE, October 2021.

Stefan Niebler, Annette Miltenberger, Bertil Schmidt, and Peter Spichtinger. Automated detection and classification of synoptic-scale fronts from atmospheric data grids. Weather Clim. Dynam., 3 (1):113–137, February 2022.

NWC SAF. User manual for the automatic satellite image interpretation processor of the NWC/GEO: Science part. Technical Report NWC/CDOP3/GEO/ZAMG/SCI/UM/ASII, Issue 2.0.1, EUMETSAT Satellite Application Facility on Support to Nowcasting and Very Short-Range Forecasting (NWC SAF), 2022. URL https://www.nwcsaf.org/documents/27295/ 0/NWC-CDOP3-GEO-ZAMG-SCI-UM-ASII\_v2.0.1.pdf/.

Yujing Qin, Qian Liu, and Chuhan Lu. Cold front identification using the DETR model with satellite cloud imagery. Remote Sensing, 17(1):36, 2025.

Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. Probing the probing paradigm: Does probing accuracy entail task relevance? In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 3363–3377, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.295. URL https://aclanthology.org/2021.eacl-main.295/.

Sebastian Schemm, Irina Rudeva, and Ian Simmonds. Extratropical fronts in the lower troposphere– global perspectives obtained from two automated methods. Q. J. R. Meteorol. Soc., 141(690): 1686–1698, July 2015.

Christopher Velden, Jaime Daniels, David Stettner, David Santek, Jeff Key, Jason Dunion, Kenneth Holmlund, Gail Dengel, Wayne Bresky, and Paul Menzel. Recent innovations in deriving tropospheric winds from meteorological satellites. Bulletin of the American Meteorological Society, 86 (2):205–224, 2005. doi: 10.1175/BAMS-86-2-205.

## A Model architecture and input channels

Both configurations use a Swin-UNet encoder–decoder segmentation network [Liu et al., 2021, Cao et al., 2023], adapted to two dimensions from the three-dimensional Swin U-Net processor in Aurora [Bodnar et al., 2025]. The encoder uses a convolutional patch-embedding stem followed by four stages. After the first stage, each subsequent stage halves the spatial resolution and doubles the channel width. Each stage applies windowed self-attention with alternating shifted windows, allowing information to propagate across windows. A symmetric decoder with skip connections restores the original grid resolution. Unlike Aurora, our model does not use a three-dimensional pressure-level axis, Perceiver encoding, or foundation-model pre-training. Dravilas et al. [2026] give a more detailed description of the architecture. The single-service variants used here have one output head, whereas the five-service model SATFRONT has one per service.

The satellite input contains eight infrared brightness temperatures at approximate central wavelengths of 3.8, 6.2, 7.3, 8.6, 9.7, 11.0, 12.0, and 13.3 µm, together with seven auxiliary features. Following ASII [NWC SAF, 2022], we compute these features from the brightness temperatures of the 6.2, 11.0, and 12.0 µm channels. The first two are differences between channels and are sensitive to upper-atmospheric moisture and cloud properties. The remaining five are spatial filters of the 11.0 µm channel that highlight local anomalies, cloud texture, cloud edges, curved cloud structures, and largescale cloud bands. All seven features use satellite data only. The 28 ERA5 fields comprise geopotential height, specific humidity, relative humidity, temperature, and horizontal wind components (u, v) at four pressure levels (1000, 925, 850, 700 hPa), plus 2-metre temperature, mean sea level pressure, and 10-metre wind components (u , v ). OBS uses the 15 satellite channels, while HYB uses all 43 channels (15 satellite plus 28 ERA5 fields). We map the satellite observations from the HEALPix grid to a regular 0.05<sup>◦</sup> latitude–longitude grid using nearest-neighbour interpolation and bilinearly interpolate the ERA5 fields to the same grid.

## B Reference baselines

The two reference rows in Table 1 provide external context. We score the operational NWC SAF ASII product [NWC SAF, 2022], a rule-based satellite+NWP front detection product (the ASII-NWP variant), against the same 2024 Met Office charts (1453 of the 1462 test timestamps) using the same CSI@150 metric as for the learned models. The Met Éireann row shows the agreement between analyses from different meteorological services. To measure this agreement, we score the 2024 surface-analysis charts of four other services in place of the model predictions, against the charts of the UK’s national meteorological service (UK Met Office) used as ground truth throughout. The four services are the Finnish Meteorological Institute (FMI), the German Weather Service (DWD), the Royal Netherlands Meteorological Institute (KNMI), and the Irish Meteorological Service (Met Éireann). For each service, we evaluate only the overlap with the Met Office domain and timestamps for which both services provide an analysis (359 for Met Éireann, 366 for DWD, 893 for FMI, and 1460 for KNMI), so these scores do not use exactly the same samples as the models. Agreement ranges from 0.277 (FMI) to 0.501 (Met Éireann). This variation reflects not only differences in frontal positioning but also differences in analysis conventions, available local information, and geographic focus, as each service primarily analyses its own region of interest. The range therefore provides context for model performance but does not represent a ceiling on achievable detection skill. No directly comparable machine learning baseline exists because no prior detector has been trained and evaluated on the same geographic domain, time period, and labelling convention.

## C Representation analysis details

We freeze the OBS encoder and extract features from each of its four stages. At each stage, we fit a per-pixel ridge-regression model to predict each of the 28 ERA5 fields from the encoder features, using 500 training-year timestamps sampled at regular intervals. We pool each ERA5 target to the corresponding feature resolution using adaptive average pooling and subsample pixels by a factor of four during fitting. We standardise all features before regression. We select the ridge penalty from {0.1, 1, 10, 100} using a 20% held-out split of the training subset. We evaluate $R ^ { 2 }$ on the 2024 test year, take the maximum for each field across the four encoder stages so that each variable is evaluated at its most favourable resolution, and then average within each variable group. We compute all 95% confidence intervals in this paper using a paired moving-block bootstrap over the 1462 test timestamps (block length 20, $B = 1 0 0 0$ resamples). A value of $P ( \Delta \leq 0 ) < \bar { 1 } 0 ^ { - 3 }$ means that no resample gave $\Delta \leq 0 .$ . All results use a single training seed.

We repeat the same procedure with two controls. The first replaces the encoder features with the raw satellite input, pooled to each feature resolution. The second uses features from two independently randomly initialised encoders with the same architecture and averages $R ^ { 2 }$ across the two. These controls distinguish the contribution of the trained representation from that of the inputs or architecture alone. As a verification, predicting two of the model’s own input channels from its encoder features gives $R ^ { 2 } \approx 0 . 9 9$ . Table 2 reports the results by variable group.

Table 2: Predictability of ERA5 variables from the frozen OBS encoder, by variable group (2024 test year). $R ^ { 2 }$ is the maximum over encoder stages. The ∆ columns show the improvement over the randomly initialised encoder and the raw satellite input. All improvements exclude zero $( P ( \Delta \leq 0 ) < 1 0 ^ { \bar { - } 3 } )$ . 95% confidence intervals for $R ^ { 2 }$ and the improvement over the random encoder are shown in Figure 1.
<table><tr><td>Variable group</td><td> $R ^ { 2 }$  (max over stages)</td><td>∆ vs. random init.</td><td>∆ vs. raw input</td></tr><tr><td>Temperature</td><td>0.883</td><td>+0.188</td><td>+0.177</td></tr><tr><td>Moisture</td><td>0.600</td><td>+0.254</td><td>+0.244</td></tr><tr><td>Mass</td><td>0.585</td><td>+0.282</td><td>+0.234</td></tr><tr><td>Winds</td><td>0.232</td><td>+0.181</td><td>+0.189</td></tr></table>

## D Single-group addition experiments

Table 3 reports CSI@150 for the satellite-only model augmented with one ERA5 variable group, together with the paired improvement over OBS.

Table 3: Single-group addition experiments. CSI@150 of the satellite-only model augmented with one ERA5 variable group, and the paired improvement over OBS (2024 test year, 95% paired blockbootstrap confidence intervals). All three improvements have $P ( \Delta \leq 0 ) < \mathrm { i } 0 ^ { - 3 }$
<table><tr><td></td><td>OBS + winds</td><td>OBS + moisture</td><td>OBS + temp. &amp; mass</td></tr><tr><td>CSI@150</td><td>0.520 [0.506, 0.530]</td><td>0.540 [0.527, 0.550]</td><td>0.534 [0.521, 0.545]</td></tr><tr><td>∆ over OBS</td><td>+0.118 [0.108, 0.126]</td><td>+0.138 [0.129, 0.147]</td><td>+0.132 [0.122, 0.142]</td></tr></table>

![](images/3cbc9a23007447d67122638619db7c75a86bc7436293b7e0c0c627a499829c14.jpg)  
Figure 3: Example on one 2024 test timestamp (22 November, 00:00 UTC). Top left: Met Office analysis. Top right: satellite+ERA5 model (HYB). Bottom left: satellite-only model (OBS). Bottom right: the same OBS prediction after a morphological closing that bridges small gaps and removal of small regions, applied for visualisation only. All reported metrics use the unprocessed predictions. Cold, warm, and occluded fronts are shown in blue, red, and purple over the infrared background.

## E ERA5 improvement across tolerances

Table 4 reports the paired improvement in CSI from adding ERA5 (HYB − OBS) at each tolerance. The gain is smallest at 50 km (+0.116), peaks at 150 km (+0.173), and decreases slightly to +0.165 at 250 km. However, because both models achieve higher CSI at larger tolerances, the absolute gain alone cannot distinguish additional detections from improved positioning. The relative gain also decreases with tolerance, from 48% at 50 km to 43% at 150 km and 32% at 250 km. CSI combines two components, the probability of detection (the fraction of observed front pixels matched by predictions) and the success ratio (the fraction of predicted front pixels matching observed fronts). The gain in probability of detection remains nearly constant, between +0.20 and +0.21, across all tolerances from 50 to 250 km. If ERA5 primarily improved the positioning of fronts already detected by OBS, this gain would diminish at larger tolerances as displaced OBS predictions begin to match the observations. Instead, the persistent gain is consistent with HYB detecting fronts missed entirely by OBS, which count as additional detections regardless of tolerance. In contrast, the success-ratio gain decreases from +0.16 at 50 km to +0.05 at 250 km, indicating that HYB also places its predicted fronts closer to the analysed fronts.

Table 4: Paired ERA5 improvement in CSI (HYB − OBS) by tolerance, 2024 test year, with 95% paired block-bootstrap confidence intervals. All exclude zero $\dot { (} P ( \Delta \leq 0 ) < 1 0 ^ { - 3 } )$
<table><tr><td>Tolerance (km)</td><td>50</td><td>100</td><td>150</td><td>200</td><td>250</td></tr><tr><td>ΔCSI</td><td>+0.116</td><td>+0.160</td><td>+0.173</td><td>+0.172</td><td>+0.165</td></tr><tr><td>95% CI</td><td>[0.109, 0.124]</td><td>[0.151, 0.169]</td><td>[0.163, 0.183]</td><td>[0.162, 0.183]</td><td>[0.155, 0.177]</td></tr></table>

## F Qualitative example

The main-text evaluation reports quantitative results averaged over the full 2024 test year. Figure 3 instead illustrates OBS and HYB predictions for a single test timestamp. The top-left panel shows the Met Office analysis. The top-right panel shows HYB, which reproduces the frontal structure more completely and follows the Met Office analysis more closely. The bottom-left panel shows OBS, which detects the main systems but breaks many fronts into short, scattered segments and misses others. These fragments make the frontal structure hard to read, so the bottom-right panel shows the same OBS prediction after a morphological closing that bridges gaps of up to about 12 pixels, followed by removal of connected regions smaller than 120 pixels. We apply this step only to OBS, because the HYB predictions already form continuous fronts. This post-processing is for visualisation only (as in an operational display) and does not affect any reported metric, which we always compute on the unprocessed predictions of both models.