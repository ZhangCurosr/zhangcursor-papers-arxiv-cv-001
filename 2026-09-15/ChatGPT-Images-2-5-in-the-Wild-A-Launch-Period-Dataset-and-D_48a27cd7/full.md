# ChatGPT Images 2.5 in the Wild: A Launch-Period Dataset and Detector Evaluation

Dennis Ng<sup>\*</sup> Xingyu Shen<sup>\*</sup> Ankit Raj<sup>\*</sup> Kidus Zewde<sup>\*</sup> Tommy Duong<sup>\*</sup>

Yuchen Zhou<sup>\*</sup> Yuxin Zhang<sup>\*</sup> Neo Tiangratanakul<sup>\*</sup> Simiao Ren<sup>†</sup>

Scam.ai

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author: benren@scam.ai

Abstract. An image tool can change its underlying generator while retaining its public name, making version attribution from online posts ambiguous. We study this problem after the ChatGPT Images 2.5 launch. Our frozen collection contains 3,478 images from 2,440 posts across 8 sources. Recorded posting times fall within the first 51.1 hours after the announcement. It records three attribution tiers and retains standalone images after image-form filtering and targeted review. Caption claims and host records provide admission evidence, not independently verified generator identity.

The observed content profile depends on the source mixture: NightCafe supplies 39.0% of images but 77.0% of CLIPassigned fantasy scenes. We then evaluate six frozen detectors at thresholds calibrated to a 5% flag rate on reference photographs. Collection flag rates range from 3.7 to 56.4%, falling 42–81 percentage points below GenImage recall. Held-out artwork false-positive rates range from 1.5 to 96.5%, so a higher collection flag rate does not by itself establish better detection.

An exploratory X-only comparison with our April collection finds a higher September flag rate for Effort, and a suggestive difference for DoU, under fixed-threshold post-clustered bootstrap intervals. Attribution, content and processing differences prevent a causal interpretation of these contrasts. The collection supports analysis of reported model use during a product transition, with source and attribution evidence retained for interpretation.

Keywords: AI-generated image dataset; detector evaluation; attribution; C2PA; social media; ChatGPT Images 2.5

## 1 Introduction

A model-specific dataset collected from online posts depends on evidence that connects each image to the claimed generator. That link becomes ambiguous when a product changes its underlying model while retaining its public name. A caption naming the product may then refer to either version, and a post made after an upgrade can still contain an older image.

OpenAI released ChatGPT Images 2.5 on 8 September 2026 through gpt-image-2.5-flare and gpt-image-2.5-sunburst, alongside an upgrade to ChatGPT’s image tool [14]. Our April GPT-Image-2 collection [30] used X posts to study an earlier release. Repeating its version-silent queries would not reliably identify 2.5: in a launch-day pilot, 35 of 106 returned posts explicitly named GPT-Image-2. We therefore require explicit version evidence and retain its source.

Existing research already evaluates detection on socialmedia samples and under online image processing [5, 10, 11]. Here we examine a specific collection problem raised by a product transition: what evidence supports version attribution, and how do the resulting admission rules and source mixture shape the observations? A companion paper studies forgery tasks with known answers [18]; this paper concerns posted images whose production histories are incompletely observed.

We organize the study around three questions:

1. What evidence supports version attribution? We distinguish caption claims, caption claims with a creationlanguage classifier, and host model records. We document admission and curation decisions and inspect delivered content credentials (Sections 3.4, 3.5 and 3.6). These records make the evidence auditable without treating every admitted image as independently authenticated.

2. How does source composition shape the collection? We assemble 3,478 images from 2,440 posts across 8 sources and report content and collection differences by source (Sections 3.3 and 5.1). Query instruments, language metadata, admission rules and observed time coverage differ, so the source counts do not measure relative platform activity.

3. How do published detectors respond at fixed operating points? We check six frozen implementations against a labelled benchmark, then report collection flag rates beside held-out control FPRs (Section 6). An exploratory Xonly comparison with the April release extends this record across two collection windows (Section 6.4). It does not isolate a generator-version effect.

![](images/43811920693b171d81a32fab29bdfffde7637e5df71e5af78b60285cbc85d884.jpg)  
Figure 1: One illustrative image from each of the eight sources in the frozen collection. Labels give each source’s full image count. Examples are selected from the previously fixed illustration set and preserve their aspect ratios. This source-balanced display is not a random or proportionally representative sample; Figure 3 shows subject distributions within each source.

The resulting contribution is a documented collection of attributed launch-period images and a fixed-threshold evaluation of detector responses to it. Here, in the wild means images retrieved from online posts and galleries under the stated admission rules. The image-form filter excludes screenshots and composites, and the collection is not a census of online use. Historical audits assess agreement with another model; their coverage limits are reported in Section 4.

## 2 Related Work

Model-specific datasets. Broad web-image collections such as LAION-5B [22] are not organised around a specific generator. Targeted datasets focus on a generator or a defined roster, as in DiffusionDB [25] and GenImage [34]. For OpenAI models, GPT4o-Receipt [31] collected document images with direct API provenance. Our April dataset [30] curated 10,217 GPT-Image-2 images from 6,606 X posts under creator self-report, starting from 27,662 collected image records. That collection is larger than the present one but uses a single platform and attribution regime.

Two follow-up studies [24, 32] predate Images 2.5, while first-look studies of GPT-4o image generation [2, 29] illustrate how to characterise a newly released model. Our collection distinguishes explicit caption-version claims from hostrecorded attribution and applies an image-form filter across multiple sources. The gateway in Section 3.2 reduces the need for separate platform clients; it does not establish exhaustive coverage.

Evaluation in online settings. ITW-SM [10] collects real and generated images from four social platforms and evaluates how detector design and preprocessing affect performance. TrueFake [5] studies generated images shared through social networks, while RRDataset [11] examines scenario changes, internet transmission and re-digitization. These studies already establish that online conditions matter for detection. Our focus is attribution during a specific product transition: a tool can retain its name while its underlying generator changes. We record the version evidence available in captions and host fields, distinguish admission tiers, and examine the resulting collection with fixed detectors. The April comparison extends this observational record; it does not isolate the generator upgrade.

Provenance in delivered files. C2PA [4] uses a signed manifest to describe an asset’s production history. OpenAI reports signing outputs and adding an invisible watermark [13]. In our April study, embedded credentials did not survive the inspected X delivery path. Here, the inspected generatorversion field does not distinguish the tested endpoint versions (Section 3.6). X’s “Made with AI” label [27] and Pixiv’s AI-generation declaration [16] are additional platform signals. Section 7 explains their limitations for this collection.

Generated media and detection in practice. Self-reported AI images account for well under one percent of image posts in the Reddit art communities studied by Matatov et al. [12]. That is a rate of disclosed AI use; without independently identifying undisclosed AI images, it does not estimate disclosure recall. Studies of AI political imagery on X [3] and fact-checked media [6] document other settings in which generated media circulate.

Image detectors often generalise poorly across generator families [7, 21, 23]. Multimodal language models are inconsistent zero-shot detectors [20], and deployment studies find further performance losses outside laboratory benchmarks [19]. A collection attributed to a new generator, with screenshots and composites filtered out, supports evaluation in this less controlled setting. Its attribution and processing limits remain part of that evaluation.

## 3 Collection

We first define which posts and assets enter the collection, then distinguish version-attribution evidence from image form. These choices determine the population described in the content and detector analyses.

## 3.1 Launch anchor and funnel

Every source is filtered against a single timestamp, the @OpenAIDevs announcement post, whose snowflake identifier decodes to 2026-09-08T18:58:45Z. Images 2.5 outputs existed before it – arena evaluations were live by 2026-09-07 – but arena images are anonymous and pre-launch API access was private, so the announcement is a reproducible scope boundary rather than a claim about the earliest attributable generation. Official X queries pass it as start\_time; other sources take the coarsest server-side date filter they support and are filtered client-side against it (Appendix C). Gated collection and released subset share an end, 2026-09-10 22:06:41Z, roughly 51.1 hours after the anchor. Figure 2 traces the records from collection to the retained images.

## 3.2 Collection infrastructure

Five of the nine collection routes use one commercial gateway (Monid, CLI 0.1.7); Pixiv, note.com and NightCafe use public JSON endpoints directly. The gateway log records 932 calls across 5 endpoints. Endpoint details and cost accounting appear in Appendix B.

The official and gateway X routes provide partly overlapping observations: 502 of 1,524 gateway-admitted posts appear in retained official results, and 168 pass both admission rules. Different queries, schedules and admission rules prevent interpreting this difference as a controlled comparison of index coverage (Appendix D).

## 3.3 Sources, routes and yield

We collect X/Twitter through two routes. The official v2 recent-search endpoint [26] has a seven-day retention window and does not provide a census of public posts. A launch-day pilot and a day-one run used the same queries. The day-one run read 2,009 posts for \$10.05 and yielded 594 gate-passing records from 305 posts.

Table 1: Retained images by platform after deduplication, imageform filtering and targeted review. X combines both collection routes. The separate 12 videos are excluded.
<table><tr><td>Platform</td><td>Images</td></tr><tr><td>X/Twitter</td><td>1,501</td></tr><tr><td>NightCafe</td><td>1,358</td></tr><tr><td>Xiaohongshu</td><td>325</td></tr><tr><td>note.com</td><td>137</td></tr><tr><td>Pixiv</td><td>66</td></tr><tr><td>Weibo</td><td>40</td></tr><tr><td>Instagram</td><td>40</td></tr><tr><td>Reddit</td><td>11</td></tr><tr><td>Total</td><td>3,478</td></tr></table>

The gateway runs the same search but returns a partly overlapping set (Appendix D). Both routes deliver media from the same content delivery network at the same resolutions. We merge them on post and media identifiers. The remaining seven sources use either the gateway (Xiaohongshu, Weibo and Reddit through TikHub; Instagram through a resold Apify hashtag scraper) or direct access (Pixiv, note.com and Night-Cafe). Table 1 gives retained image counts.

The nine instruments differ in query, time filtering, timestamp resolution, served format, language metadata and creation-language requirements (Appendix C). Cross-source differences therefore reflect the instruments as well as the platforms. Reddit is mainly a discussion source in this sample: 78 of 88 qualifying posts have no downloadable image. Instagram differs in being accessed by hashtag through a third-party scraper rather than a search index (Appendix D).

## 3.4 The version gate

We record three attribution tiers. Caption-based admission requires a case-insensitive match to a 2.5 name, an API codename (Flare or Sunburst), or a 2.5 hashtag, with no accompanying 2.0 signal (Appendix F). Posts naming only 2.0, both generations, or neither are excluded from this route. A match establishes a caption claim, not the attachment’s true generator.

• Caption plus classifier includes 371 released images from official-API X records. Alongside the version match, these posts pass the April creation-language classifier. It uses creation phrases, shared prompts and showcase hashtags, while excluding specified announcement and promotional patterns.

• Caption-only includes 1,749 images from gateway X and the other caption-based sources.

• Host-recorded includes 1,358 NightCafe images admitted through the platform’s generator field. They need no version-bearing caption. Its 12 videos form a separate cohort.

![](images/28e31d07d66f6a17c0cc8bc4edc96bc7893baa77c39479d356e2b0a2294bdcf7.jpg)  
Figure 2: Collection funnel. Of 10,163 candidate asset records, 7,911 passed caption or host-record attribution; deduplication removed 274 exact (post, media, frame) repeats and 1,281 average-hash near-duplicates (Appendix G), leaving 6,356 gated assets from 3,624 posts Image-form filtering kept the 3,494 assets labelled standalone, and targeted review removed a further 16, leaving 3,478 retained images; 12 videos are recorded separately and excluded from image statistics. dedup\_audit.py replays the attribution and deduplication stages.

The tier field allows each population to be selected independently (Appendix F.1). Host records state what the platform reports; we did not independently verify endpoint execution.

The caption rule excludes version-silent posts and can also admit misleading or non-creation captions. Its errors are not one-sided. The earlier text audit assesses agreement about caption claims, with recall evidence restricted to recorded official-X strata (Section 4); it does not validate host attribution or identify the true generator from pixels.

## 3.5 Image-form filter and targeted review

Attribution alone does not establish image form. A caption can refer to Images 2.5 while its attachment is a screenshot; a host can identify the generator while serving a grid or composite. We therefore classify each gated image with gpt-5.4-mini from a 768-pixel thumbnail, at a cost of \$0.70 per thousand images.

The classifier assigns one of six image forms: a standalone image, chat or app screenshot, collage or grid, photograph of a screen, promotional graphic, or other. We retain only the standalone class. Appendix H gives the exact labels and prompt. A targeted visual review then removed 16 prompt sheets, editorial composites and end cards. Each exclusion has a recorded reason. This review corrects identified errors but does not estimate the remaining error rate.

Of 6,344 gated images, 3,494 are labelled standalone and 44.9% are not: 1,661 promotional graphics, 502 chat screenshots, 506 collages or grids, 64 photographs of screens and 117 other. The filter separates imageform from provenance: the excluded collages and promotional graphics hold an unknown number of real outputs in composite form, so exclusion is not a claim that an asset is not AI-generated. Per-class counts by source are in Appendix H.1.

## 3.6 Content credentials and version attribution

A local byte scan found C2PA markers in 31 of 66 released Pixiv files; 30 contain the searched generator-version, OpenAI and watermark strings. Marker presence is not signature validation, and the remaining marker-positive file is not included in the matching-string claim (Appendix K). Earlier pilot signature checks are reported separately. These observations do not establish preservation or stripping behaviour for every source or delivery path.

The API-side ledger covers 1,007 files generated for the benchmark on 2026-09-09: 371 Flare, 371 Sunburst and 265 GPT-Image-2. All inspected files contain the generatorversion string 2.0, including both 2.5 endpoints. This is a census of that logged file set, not of all API outputs. The supported result is that the inspected generator-version field does not distinguish these tested endpoint versions; caption and host attribution remain separate evidence sources.

## 3.7 Coverage beyond the retained sources

The retained sources do not exhaust the available online images. Launch-period probes also examined model-tagged galleries and API resellers (Appendix L). A generation service need not expose a public gallery, and a model filter need not return a model field for each image.

NightCafe illustrates the limits of a single probe. A 2026-09-09 sample of 240 creations contained no 2.5 records. On 2026-09-10, the same endpoint yielded 1,449 records with a 2.5 gptImageModel field. These samples came from a retrospectively paginated feed. They describe observed feed contents, not when the platform adopted the model. Empty results from other tested endpoints likewise establish only what those queries returned. Collection records therefore retain the endpoint, query and observation date as well as the attribution evidence.

## 4 Historical Audits of Automated Curation

We audited the version gate, screenshot filter and subject classifier on an earlier collection, before the day-two expansion. Seeded samples (seed 20260909) were relabelled by gpt-5.5, a different model from the screenshot filter’s gpt-5.4-mini. The audits measure agreement with another model; they provide neither human annotation nor verified generator attribution. They cost \$3.53.

The screenshot audit sampled a pool of 2,910 images, including 1,382 predicted standalone images. Neither its 300 items nor the 100 subject-classification items include Night-Cafe. These audits therefore do not validate the expanded release or its host-attributed tier. Appendix I gives the sampling design. The intervals below are descriptive Wilson 95% intervals for raw sample proportions, not uncertainty estimates adjusted for the weighted design.

Caption claims. Among 100 sampled admitted post texts, raw agreement on the text claim is 64.0% (54.2–72.7). Disagreements concern announcements, tutorials and marketing that name the version without claiming the poster’s own generation. Appendix I.1 gives language breakdowns and the limits of the recall analysis.

Image form. The auditor labelled 98.5% (91.9–99.7) of the 66 predicted standalone images as standalone. Overall raw agreement is 72.3%, with unweighted $\kappa = 0 . 6 5$ . The auditor treated some predicted collages and promotional graphics as standalone. Because the other stratum was not sampled, the audit does not identify full-pool recall (Appendix I.2).

Subject labels. On 100 earlier released images, CLIP and the auditor agree on 61.0% of labels (κ = 0.51). We therefore describe content as a distribution of CLIP assignments, not validated subject prevalence. Appendix I.3 reports class breakdowns and their limitations.

## 5 Content and Source Composition

We next describe the retained images, keeping source-specific observations separate from pooled summaries; Figure 1 shows one illustrative image per source. Content labels and post metadata are measurements of this collection, not estimates of platform-wide use.

## 5.1 Subject matter, text and faces

We apply the April study’s three analyses to 3,478 released images: zero-shot CLIP subject classification, OCR and face detection. Appendix I.3 gives the settings and explains why the April comparison is not controlled.

Source composition strongly affects the pooled subject profile (Figure 3). NightCafe supplies 39.0% of images but 77.0% of fantasy/surreal assignments. That class accounts for 53.6% of NightCafe images and 11.3% of X images. Removing NightCafe lowers the pooled fantasy share from 27.2% to 10.3%. These differences describe the sampled images. The small source cohorts shown in the figure cannot provide stable estimates of platform-wide preferences.

OCR detects text in 39.0% of images (1,357), compared with 82.0% in April. The median number of text regions per image is 0 (April: 29). Face detection finds at least one face in 59.0% of images (2,052), compared with 59.2% in April. These measurements characterise the collections; they do not establish changes in model use. Because face detectors are biased on non-photorealistic images and CLIP assigns much of this collection to non-photographic classes, we omit the demographic histograms used in the April paper.

## 5.2 Languages, timing and wording

The cohort contains 3,478 images from 2,440 posts. Recorded language labels span 16 languages, but their sources differ. For 1,501 X image rows, the label is the API’s detected lang. Another 568 rows (16.3%) use a platform default, and 1,409 rows (40.5%) have no language field (Appendix J). These labels describe post metadata, not text recognised in the images.

The most common recorded labels in the released cohort are Japanese (703), English (675) and Chinese (631). Chinese is most common in the gated pool (2,258), but its retention rate is 27.9%, compared with 55.6% for Japanese and 62.4% for English. Source composition and image-form filtering both contribute to this difference; these counts do not measure platform adoption.

Recorded posting timestamps span roughly 51.1 hours and differ from retrieval dates (Appendix J.1). Of 1,667 retained images posted from 2026-09-10 05:25Z onwards, 1,358 (81%) come from NightCafe and were collected on 2026-09-10. The final interval therefore reflects that source’s collection route rather than demonstrating an increase in posting activity.

We analyse wording in 1,082 caption-attributed posts with nonempty text. NightCafe is excluded because its text field contains creation titles rather than caption claims. English, Japanese and Chinese patterns match prompt vocabulary in 31.6% of eligible posts, editing in 10.5%, consistency in 6.8% and speed in 6.7%. Sunburst and Flare appear in 10.4% and 6.7%, respectively. These unvalidated keyword matches do not measure how often users edited images or disclosed prompts.

## 6 Detector Responses to the Collection

We evaluate six frozen image detectors: Community Forensics [15], B-Free [8], Effort [28], PGC [33] (2026, SD v1.4 checkpoint), PROBE-ResNet50 [1], and DoU [9]. None is trained or fine-tuned on this collection. Because generator identity is creator- or host-attributed, the outcome is the fraction of imagesflagged, not verified detection recall.

## 6.1 Controls and operating point

We select 400 reference images from each of five photograph pools: GeoDE, SUN397, FairFace, Pascal VOC and Food101. We separately select 400 WikiArt artworks. Source labels are inherited without new human adjudication. After exact-byte deduplication, deterministic hash ordering and alternating assignment produce equal calibration and test splits within each pool.

Each detector’s native-logit threshold is calibrated on the 1,000 calibration photographs to flag at most 5%. The 1,000 held-out photographs measure the realised false-positive rate (FPR). Artwork controls provide a separate diagnostic and do not set the thresholds. Every detector scores all 3,478 collection images and 2,400 controls without decoding or inference failures. DoU retains its stochastic evaluation forward pass with primary seed 42; Appendix O gives execution details and a second-seed check.

<table><tr><td></td><td rowspan=1 colspan=1>illustration</td><td rowspan=1 colspan=4>surreal  graphic  portraitUI mockup landscape</td><td></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=1>36.6%</td><td rowspan=1 colspan=1>11.3%</td><td rowspan=1 colspan=1>19.5%</td><td rowspan=1 colspan=1>11.5%</td><td rowspan=1 colspan=1>10.2%</td><td rowspan=1 colspan=1>4.7%</td><td rowspan=1 colspan=1>3.9%</td><td rowspan=1 colspan=1>2.2%</td></tr><tr><td></td><td rowspan=1 colspan=1>16.5%</td><td rowspan=1 colspan=1>53.6%</td><td rowspan=1 colspan=1>10.3%</td><td rowspan=1 colspan=1>9.7%</td><td rowspan=1 colspan=1>1.6%</td><td rowspan=1 colspan=1>5.7%</td><td rowspan=1 colspan=1>1.5%</td><td rowspan=1 colspan=1>1.0%</td></tr><tr><td></td><td rowspan=1 colspan=1>32.3%</td><td rowspan=1 colspan=1>10.5%</td><td rowspan=1 colspan=1>17.5%</td><td rowspan=1 colspan=1>9.5%</td><td rowspan=1 colspan=1>10.8%</td><td rowspan=1 colspan=1>7.4%</td><td rowspan=1 colspan=1>0.9%</td><td rowspan=1 colspan=1>11.1%</td></tr><tr><td></td><td rowspan=1 colspan=1>45.3%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>11.7%</td><td rowspan=1 colspan=1>21.9%</td><td rowspan=1 colspan=1>13.9%</td><td rowspan=1 colspan=1>3.6%</td><td rowspan=1 colspan=1>2.9%</td><td rowspan=1 colspan=1>0.7%</td></tr><tr><td rowspan=1 colspan=1>Pixiv (n=66)</td><td rowspan=1 colspan=1>98.5%</td><td rowspan=1 colspan=1>1.5%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td></tr><tr><td></td><td rowspan=1 colspan=1>55.0%</td><td rowspan=1 colspan=1>15.0%</td><td rowspan=1 colspan=1>2.5%</td><td rowspan=1 colspan=1>2.5%</td><td rowspan=1 colspan=1>22.5%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>2.5%</td></tr><tr><td></td><td rowspan=1 colspan=1>12.5%</td><td rowspan=1 colspan=1>12.5%</td><td rowspan=1 colspan=1>27.5%</td><td rowspan=1 colspan=1>30.0%</td><td rowspan=1 colspan=1>2.5%</td><td rowspan=1 colspan=1>2.5%</td><td rowspan=1 colspan=1>7.5%</td><td rowspan=1 colspan=1>5.0%</td></tr><tr><td></td><td rowspan=1 colspan=1>54.5%</td><td rowspan=1 colspan=1>18.2%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>18.2%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>9.1%</td><td rowspan=1 colspan=1>0.0%</td><td rowspan=1 colspan=1>0.0%</td></tr></table>

Figure 3: CLIP-assigned subjects by source. Cells give percentages within each source, and row labels give image counts. The eight-class assignments show how source composition changes the aggregate distribution. They are not validated estimates of platform preferences o model capabilities.

What the operating point establishes. Photograph calibration fixes a reproducible threshold for each detector. It does not establish a 5% FPR on every source or content type. The held-out controls measure how often reference photographs and artworks are incorrectly flagged; the collection rate measures how often attributed images are flagged. Deployment precision would additionally require representative controls and the prevalence of generated images in the target stream. These experiments therefore support comparisons of the frozen cohorts, not a deployment ranking.

## 6.2 Harness reproduction on GenImage

We check the six detector implementations on a labelled benchmark before interpreting their collection scores. Forward-pass agreement with official code and plausible photograph FPRs are insufficient: the former can share a preprocessing error, and the latter tests no generated images.

GenImage [34] provides a common reference. Community Forensics reports results on it, Effort’s released checkpoint targets it, and PROBE and PGC train on its SD v1.4 split. We score a frozen, balanced validation sample: 300 real and 300 generated images per generator across 7 generators, totalling 4,200 images. The sample is hash-selected before inference, with no fitting or tuning. This implementation check uses each detector’s own 0.5 boundary. The collection comparison in Section 6.3 instead applies the photograph-calibrated thresholds to both GenImage and the collected images. Appendix O.1 gives the freeze, per-generator comparisons and a separate Community Forensics evaluation.

All five detectors with a published reference agree within 3.5 accuracy points, and four agree within 1.7 (Table 3). Several per-generator patterns also reproduce. PROBE-ResNet50 reaches 57.0% versus the published 60.0% on ADM, and 49.8% versus 49.4% on BigGAN. B-Free agrees within 1.3 points on all seven generators.

PGC’s published accuracy is 100.0% on 6 generators. Big-GAN is its only nonsaturated reference cell and the one that disagrees. Across all 28 per-generator comparisons, three differences exceed three standard errors of our subset estimate, all with our result higher. The remainder lie within three standard errors, in both directions.

The benchmark results provide evidence against inverted scores or gross implementation failures. They do not prove every collection preprocessing path correct. PROBE-ResNet50’s reproduced benchmark weaknesses are consistent with its low collection flag rate of 3.7%, although that consistency alone does not identify the cause of the collection result.

## 6.3 Flag rates on the collection

Held-out photograph FPRs range from 3.5 to 5.9%, while artwork FPRs range from 1.5 to 96.5% (Table 2). PGC, for example, flags 96.5% of reference artworks and 5.1% of photographs. A high collection flag rate can therefore coexist with sensitivity to human artwork.

Table 2: Frozen-detector results at a 5% reference-photograph calibration FPR. Collection intervals are 95% post-clustered bootstrap intervals conditional on the fitted threshold. Held-out control FPRs are separated by content type. Collection flag rates are not independently verified recall. DoU uses seed 42; its intervals exclude latent-sampling variation.
<table><tr><td>Detector</td><td>Collection flagged  $( n = 3 , 4 7 8 )$ </td><td> $( n = 1 , 0 0 0 )$ </td><td>Photo FPR Artwork FPR  $( n = 2 0 0 )$ </td></tr><tr><td>Community Forensics</td><td>54.7% [52.8, 56.7]</td><td>5.9%</td><td>42.0%</td></tr><tr><td>B-Free</td><td>15.0% [13.4, 16.6]</td><td>4.2%</td><td>64.0%</td></tr><tr><td>Effort</td><td>43.7% [41.5, 46.2]</td><td>3.5%</td><td>8.5%</td></tr><tr><td>PGC (2026)</td><td>56.4% [54.2, 58.6]</td><td>5.1%</td><td>96.5%</td></tr><tr><td>PROBE-ResNet50</td><td>3.7% [2.6, 5.1]</td><td>5.3%</td><td>1.5%</td></tr><tr><td>DoU</td><td>17.4% [15.1, 19.5]</td><td>4.4%</td><td>11.0%</td></tr></table>

Table 3: Accuracy on a balanced 4,200-image GenImage validation sample at each detector’s own 0.5 boundary. Means use the same 7 generators; ∆ is ours minus published. <sup>†</sup>Community Forensics reports pooled accuracy over a set containing one additional generator, so its comparison is not exactly matched. <sup>‡</sup>B-Free’s reference is another group’s measurement. DoU reports GenImage results only graphically, so no tabulated reference is available.
<table><tr><td>Detector</td><td>Acc. (%)</td><td> $\mathrm { A P } \left( \% \right)$ </td><td>Published (%)</td><td> $\Delta$ </td></tr><tr><td>Community Forensics</td><td>95.0</td><td>99.8</td><td></td><td> $9 5 . 7 ^ { \dagger } \quad - 0 . 7 \qquad $ </td></tr><tr><td>B-Free</td><td>87.2</td><td>96.9</td><td></td><td> $8 7 . 3 ^ { \ddagger } \quad - 0 . 1 \quad$ </td></tr><tr><td>Effort</td><td>92.1</td><td>98.3</td><td></td><td> $8 8 . 7 \quad + 3 . 4 \hphantom { 0 0 0 }$ </td></tr><tr><td>PGC (2026)</td><td>99.7</td><td>100.0</td><td></td><td> $9 8 . 1 \quad + 1 . 6 \phantom { 0 }$ </td></tr><tr><td>PROBE-ResNet50</td><td>74.4</td><td>86.2</td><td></td><td> $7 5 . 5 \quad \textrm { -- } 1 . 1 $ </td></tr><tr><td>DoU</td><td>88.7</td><td>99.8</td><td>not reported</td><td>一</td></tr></table>

At the same thresholds, the six detectors recall 47–100% of GenImage’s generated images with at most 4.0% false positives. They flag 3.7–56.4% of the collection. For every detector, the latter rate is lower, by 42–81 percentage points.

Figure 4 places these rates beside the held-out photograph and artwork controls. It makes both observations visible: collection flag rates are low relative to benchmark recall, while artwork FPRs can be high. Interpreting either requires the corresponding population and denominator.

The benchmark check supports the implementation for 4 detectors with their own published reference results (Section 6.2). It does not exclude preprocessing faults specific to platform-recompressed images: GenImage uses small benchmark PNGs. Three per-generator comparisons exceed sampling error, all with our accuracy above the published value. Those discrepancies widen the observed gap.

3. Content differs. GenImage is photographic and ImageNetderived; CLIP assigns much of this collection to non-

Interpreting the gap. Five additional differences prevent treating this gap as an isolated measure of detector generalization:

2. Processing differs. Some delivery routes recompress images, and we have not bounded the resulting effect on detector scores.

1. Attribution is unverified. Images incorrectly attributed to Images 2.5 can change the collection flag rate.

photographic classes. The artwork controls show changes of tens of percentage points between the two control types; their content and processing are not matched.

4. Benchmark exposure differs. Effort’s checkpoint is selected for GenImage, while PROBE and PGC train on its SD v1.4 split. Removing the two SD-family generators changes the gap range to 22–80 pp, mostly through PROBE-ResNet50.

5. Some images may be edits. Editing vocabulary appears in 10.5% of 1,082 caption-attributed posts. These unvalidated keyword matches exclude NightCafe and are measured per post, so they cannot estimate or bound the share of edited images. Edits may retain real reference content and need not resemble wholly generated benchmark images.

We report the observed gaps without assigning their magnitude to any one cause.

Attribution and source sensitivity. Restricting the collection to its most restrictive caption-based tier does not narrow the gap consistently. Among 371 official-X images admitted by both caption matching and the creation classifier, the gap widens for 5 of six detectors and ranges from 45 to 82 pp. The gaps therefore persist in this subset. The restriction does not validate attribution or isolate its effect: the tier is X-only, so tier and platform remain confounded.

Flag rates also vary by source (Appendix Table 10). Content, attribution and processing differences are intertwined in those comparisons. These results show why source and attribution fields are needed when reusing the dataset for detection studies.

![](images/5f5465b1e9faacf28b3b7d0dff060a08635fd461318d29b8a15a7346470561f0.jpg)  
Figure 4: Detector responses at the same photograph-calibrated thresholds. Left: labelled GenImage generated-image recall (squares) and attributed collection flag rates (circles). Right: false-positive rates on held-out reference photographs and artworks. Collection bars are 95% post-clustered bootstrap intervals; control bars are 95% Wilson intervals. GenImage points show the frozen sample estimates. Sample sizes appear in the legends. The populations differ in content, provenance and processing; connecting their outcomes to a generator effect would require a controlled experiment.

## 6.4 Comparison with our April GPT-Image-2 release

Design. We score a fixed 3,000-image sample from 2,542 posts in the April GPT-Image-2 release [30]. Its parent release contains 10,217 images from 6,606 X posts. We use the same six checkpoints, preprocessing and score definitions as for September, whose scores are already complete. A deterministic hash ranking selects April images before inference. Each detector retains its photograph-calibration threshold; neither cohort sets the operating point.

The primary September cohort comprises 1,501 X images from 861 posts. The pooled release is a secondary view. The cohorts are unpaired, and 95% percentile intervals use 50,000 post-clustered bootstrap draws per cohort. A 100-image pilot from the same ranking found no separation and resolved differences only to about ±10 pp. We then extended the prefix. This added records without reselecting them, but the expansion decision followed the pilot results. The reported intervals do not account for that look, so these contrasts are exploratory. A held-out April check appears below; Appendix O.2 gives coverage and sensitivity details.

Comparability. April admitted generic AI-badge and nameonly claims without an image-form filter. Thus, even within X, the cohorts differ in attribution rules, screenshots, content and collection window. Detector-training exposure to either release is unverified. This design compares two collections and cannot isolate the effect of upgrading the generator.

Findings. The fixed-threshold 95% intervals indicate higher September X flag rates for Effort and DoU (Table 4, Figure 5). The differences are +6.1 pp [+2.7, +9.6] for Effort and +3.6 pp [+1.0, +6.2] for DoU. Both remain positive after a Bonferroni adjustment for 6 contrasts: Effort [+1.6, +10.8] and DoU [+0.2, +7.2]. We use 50,000 bootstrap draws to estimate these tail bounds.

DoU’s adjusted interval only narrowly excludes zero and omits its latent-sampling variability. A second seed shifts the pooled September flag rate by 0.43 pp at the same threshold. We therefore treat its contrast as suggestive. The other four detectors show no detected difference, with interval half-widths at most 3.9 pp. These intervals quantify precision but do not establish equivalence.

No detector’s interval lies wholly below zero. All point estimates except Community Forensics’ are positive. These are differences between the sampled X collections; they do not establish whether the upgraded generator is intrinsically easier or harder to detect.

Check with held-out April images. We re-test the two positive contrasts on 7,217 April images from 5,153 posts at ranks beyond the analysed sample. These images were not used in the sample-expansion decision. Thresholds, score definitions and September scores remain fixed, and only Effort and DoU are evaluated.

Effort’s held-out contrast is +5.8 pp [+2.6, +9.1]. DoU replicates at +4.7 pp [+2.3, +7.3] against +3.6 pp. Both contrasts remain positive under the same family-wise adjustment, with lower bounds of +2.1 and +2.0 pp. Only April is held out. Reusing the same 1,501 September images makes the estimates correlated, so this check replicates the April side of the comparison rather than the full contrast independently. Because selection is by image rank, 1,089 posts contribute images to both April samples, including 1,894 held-out images; the held-out images are therefore not independent of the analysed sample at the post level.

Table 4: X-only flag rates at identical detector-specific thresholds. Entries are percentages with 95% post-clustered bootstrap intervals; differences are September minus April in percentage points (pp). April uses 3,000 of 10,217 released images; September X contains 1,501. The contrasts do not identify causal generator-version effects. DoU uses seed 42.
<table><tr><td>Detector</td><td>April X (%)</td><td>September X (%)</td><td>Difference (pp)</td></tr><tr><td>Community Forensics</td><td>44.8 [42.9, 46.7]</td><td>44.3 [41.2, 47.5]</td><td>-0.5 [-4.1, +3.2]</td></tr><tr><td>B-Free</td><td>18.1 [16.6, 19.5]</td><td>19.3 [16.8, 22.0]</td><td>+1.3 [-1.7, +4.3]</td></tr><tr><td>Effort</td><td>22.7 [21.1, 24.3]</td><td>28.8 [25.8, 31.9]</td><td>+6.1 [+2.7, +9.6]</td></tr><tr><td>PGC (2026)</td><td>42.4 [40.5, 44.2]</td><td>43.2 [39.8, 46.7]</td><td>+0.8 [-3.1, +4.7]</td></tr><tr><td>PROBE-ResNet50</td><td>2.9 [2.3, 3.5]</td><td>3.8 [2.8, 4.9]</td><td>+0.9 [-0.3, +2.2]</td></tr><tr><td>DoU</td><td>7.6 [6.6, 8.6]</td><td>11.1 [8.8, 13.6]</td><td>+3.6 [+1.0, +6.2]</td></tr></table>

Calibration uncertainty. A further post hoc analysis resamples the calibration photographs within their five source pools, refits each threshold, and resamples posts in both cohorts. The six-detector Bonferroni intervals remain positive for Effort [+1.7, +11.3] and DoU [+0.3, +7.4] pp. The other four include zero (Appendix Table 12). This check propagates calibration sampling but still omits DoU’s latent-sampling variability.

Composition sensitivity. Within September, widening from X to the pooled collection changes flag rates by up to 14.9 pp, in both directions. For Community Forensics, Effort, PGC and DoU, this shift exceeds every April–September difference in Table 4. Narrowing both cohorts to metadata-based attribution subsets (April confirmed images and September official-X classifier admissions) leaves the Effort and DoU differences positive; only PROBE-ResNet50, whose X difference is near zero, changes sign. Only the X-only comparison has a difference interval; pooled and attribution-subset views are descriptive.

## 7 Limitations

Exploratory detector evaluation. Collection labels record attribution. Control labels are inherited from existing datasets, without matching their content or processing to the collected images. Detector-training overlap and residual perceptual duplicates are unverified, and six checkpoints cover only part of the detector landscape. Primary intervals condition on the calibrated thresholds; an additional sensitivity analysis resamples calibration photographs. DoU also samples latent noise: we fix its seed, and the reported intervals exclude that source of variation.

Scope of the implementation checks. For the five detectors with a published GenImage reference, our accuracies agree within 3.5 percentage points; DoU has no tabulated reference.

![](images/0a66fbe4662f55771c618ba9bb1fc1fcfb35b5188cfebf00438ecc9968217010.jpg)  
Figure 5: Flag-rate differences between the September and April collections at fixed detector-specific thresholds. Positive values mean September X images are flagged more often. Bars are pointwise 95% post-clustered bootstrap intervals; those excluding zero are green. They omit DoU’s latent-sampling variability. Per-collection rates appear in Table 4.

This agreement provides evidence against gross scoring errors, such as inverted logits, on that benchmark. It cannot distinguish a harness fault from a difference between the released checkpoint and the one used for the paper’s table. Nor does it validate preprocessing for the platform-processed collection. B-Free’s reference comes from another group.

Unmatched cross-release cohorts. The April and September cohorts are not matched by image, prompt or creator. Exact-byte and decoded-pixel checks found no shared images; creator and prompt overlap was not established. Even within X, the cohorts differ in admission rules, image-form filtering and collection window as well as attributed generator version. The pooled September view additionally changes the platform mix. The 2 contrasts excluding zero survive a Bonferroni adjustment for 6 detectors, but they identify collection differences. The observed source shifts are large enough to be plausible alternative explanations. Effort has both the largest cross-release difference and the largest deviations from its published per-generator results. Content-dependent implementation differences would not necessarily cancel between these cohorts.

Appendix O.3 details the collection’s remaining limitations: attribution, historical audit coverage, selection, source comparability, near-duplicate removal and snapshot coverage. Platform AI labels likewise do not resolve generator version or establish coverage of all relevant posts.

## 8 Conclusion and Release

This study documents the attribution evidence available when collecting online images after an image tool is upgraded without a new public name. The frozen snapshot contains 3,478 images from 2,440 posts across 8 sources, plus 12 separately labelled videos. Caption claims and host records provide different admission evidence, and the inspected C2PA generatorversion field does not distinguish the tested 2.5 endpoints from GPT-Image-2. None of these observations independently authenticates every image’s production history.

Observed content profiles and detector flag rates differ across sources. These results motivate retaining source and attribution fields when reusing the collection. They do not estimate platform-wide preferences or activity. Historical model-based audits leave a need for human review of the frozen cohort’s evidence and curation decisions; visual review alone cannot authenticate a generator version.

At fixed thresholds, collection flag rates are lower than Gen-Image recall for all six detectors. The gaps remain compatible with attribution, content and processing differences. In the exploratory April–September X comparison, fixed-threshold intervals support a higher September flag rate for Effort and suggest one for DoU. Both contrasts persist with held-out April images, which share posts with the analysed sample, and with calibration resampling; DoU’s latent-sampling variation remains excluded. Matched prompts, verified generator records and controlled image processing would be needed to identify an upgrade effect.

Prepared artifacts and planned release. The local snapshot contains media, metadata.csv and a datasheet (Appendix M). The collection and curation code records queries, endpoints, gate expressions, deduplication and the screenshot prompt. We plan to publish the dataset and code; the hosting location and distribution terms are not yet finalized.

## References

[1] Zijie Cao, Weijie Tu, Yao Xiao, Weijian Deng, Liang Lin, and Pengxu Wei. Where detectors fail: Probing generative space for generalizable AI-generated image detection. arXiv preprint arXiv:2605.24906, 2026.

[2] Sixiang Chen, Jinbin Bai, Zhuoran Zhao, Tian Ye, Qingyu Shi, Donghao Zhou, Wenhao Chai, Xin Lin, Jianzong Wu, Chao Tang, Shilin Xu, Tao Zhang, Haobo Yuan, Yikang Zhou, Wei Chow, Linfeng Li, Xiangtai Li, Lei Zhu, and Lu Qi. An

empirical study of GPT-4o image generation capabilities. arXiv preprint arXiv:2504.05979, 2025.

[3] Zhiyi Chen, Jinyi Ye, Beverlyn Tsai, Emilio Ferrara, and Luca Luceri. Synthetic politics: Prevalence, spreaders, and emotional reception of AI-generated political images on X. arXiv preprint arXiv:2502.11248, 2025.

[4] Coalition for Content Provenance and Authenticity. C2PA technical specification, version 2.1. Technical report, C2PA, 2024. https://c2pa.org/specifications/spec ifications/2.1/specs/C2PA\_Specification. html; accessed September 2026.

[5] Stefano Dell’Anna, Andrea Montibeller, and Giulia Boato. TrueFake: A real world case dataset of last generation fake images also shared on social networks. arXiv preprint arXiv:2504.20658, 2025.

[6] Nicholas Dufour, Arkanath Pathak, Pouya Samangouei, Nikki Hariri, Shashi Deshetti, Andrew Dudfield, Christopher Guess, Pablo Hernández Escayola, Bobby Tran, Mevan Babakar, and Christoph Bregler. AMMeBa: A large-scale survey and dataset of media-based misinformation in-the-wild. arXiv preprint arXiv:2405.11697, 2024.

[7] Diego Gragnaniello, Davide Cozzolino, Francesco Marra, Giovanni Poggi, and Luisa Verdoliva. Are GAN generated images easy to detect? A critical analysis of the state-of-the-art. In IEEE International Conference on Multimedia and Expo (ICME), 2021.

[8] Fabrizio Guillaro, Giada Zingarini, Ben Usman, Avneesh Sud, Davide Cozzolino, and Luisa Verdoliva. A bias-free training paradigm for more general AI-generated image detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18685–18694, 2025.

[9] Qinghui He, Haifeng Zhang, Qiao Qin, Bo Liu, Xiuli Bi, and Bin Xiao. Diversity over uniformity: Rethinking representation in generated image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 40407–40417, 2026.

[10] Despina Konstantinidou, Dimitrios Karageorgiou, Christos Koutlis, Olga Papadopoulou, Emmanouil Schinas, and Symeon Papadopoulos. Navigating the challenges of AI-generated image detection in the wild: What truly matters? In Proceedings of the ACM International Workshop on Multimedia AI against Disinformation (MAD), 2026.

[11] Chunxiao Li, Xiaoxiao Wang, Meiling Li, Boming Miao, Peng Sun, Yunjian Zhang, Xiangyang Ji, and Yao Zhu. Bridging the gap between ideal and real-world evaluation: Benchmarking AI-generated image detection in challenging scenarios. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 20379–20389, 2025.

[12] Hana Matatov, Marianne Aubin Le Quéré, Ofra Amir, and Mor Naaman. Examining the prevalence and dynamics of AI-generated media in art subreddits. arXiv preprint arXiv:2410.07302, 2024.

[13] OpenAI. ChatGPT Images 2.5 system card. OpenAI Deployment Safety Hub, 2026. https://deploymentsafe ty.openai.com/chatgpt-images-2-5; accessed September 2026.

[14] OpenAI. Introducing ChatGPT Images 2.5. OpenAI announcement, 2026. https://openai.com/index/intro ducing-chatgpt-images-2-5/; accessed September 2026.

[15] Jeongsoo Park and Andrew Owens. Community forensics: Using thousands of generators to train fake image detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8245–8257, 2025.

[16] pixiv. What is the “AI-generated” option in my work settings? pixiv App Help Center. https://app.pixiv.help /hc/en-us/articles/11877605105433; accessed September 2026.

[17] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), pages 8748–8763, 2021.

[18] Ankit Raj, Yuxin Zhang, Kidus Zewde, Tommy Duong, Jiaqi Gan, Xingyu Shen, Yuchen Zhou, Huaiyu Guo, Siyu Zhang, and Simiao Ren. ChatGPT Images 2.5 on forgery tasks: Testing advertised improvements against known answers, 2026. Companion paper, in preparation.

[19] Simiao Ren, Disha Patil, Kidus Zewde, Tsang (Dennis) Ng, Hengwei Xu, Shengkai Jiang, Ramini Desai, Ning-Yau Cheng, Yining Zhou, and Ragavi Muthukrishnan. Do deepfake detectors work in reality? In Proceedings of the 4th Workshop on Security Implications of Deepfakes and Cheapfakes, pages 21–26, 2025.

[20] Simiao Ren, Yao Yao, Kidus Zewde, Zisheng Liang, Tsang (Dennis) Ng, Ning-Yau Cheng, Xiaoou Zhan, Qinzhe Liu, Yifei Chen, and Hengwei Xu. Can multi-modal (reasoning) LLMs work as deepfake detectors? arXiv preprint arXiv:2503.20084, 2025.

[21] Simiao Ren, Yuchen Zhou, Xingyu Shen, Kidus Zewde, Tommy Duong, George Huang, Hatsanai (Neo) Tiangratanakul, Tsang (Dennis) Ng, En Wei, and Jiayu Xue. How well are open sourced AI-generated image detection models out-ofthe-box: A comprehensive benchmark study. arXiv preprint arXiv:2602.07814, 2026.

[22] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION 5B: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

[23] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A. Efros. CNN-generated images are surprisingly easy to spot. . . for now. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8695–8704, 2020.

[24] Yijin Wang, Shuyi Wang, Wenhan Zhang, and Yuqi Ouyang. TextRich: A multi-domain benchmark for detecting AIgenerated text-rich images from GPT-Image-2. arXiv preprint arXiv:2606.19259, 2026.

[25] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pages 893–911, 2023.

[26] X Corp. Twitter API v2 documentation: Recent search endpoint. X Developer Platform documentation, 2024. https://do cs.x.com/x-api/posts/recent-search; accessed September 2026.

[27] X Corp. X media literacy action plan. X Help Center, 2026. https://help.x.com/en/rules-and-policies/ media-literacy-plan; accessed September 2026.

[28] Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-Yue Zhang, Chengchun Liu, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, and Li Yuan. Orthogonal subspace decomposition for generalizable AI-generated image detection. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 70268–70288. PMLR, 2025.

[29] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. GPT-ImgEval: A comprehensive benchmark for diagnosing GPT4o in image generation. arXiv preprint arXiv:2504.02782, 2025.

[30] Kidus Zewde, Simiao Ren, Xingyu Shen, Jiaqi Wu, Yuchen Zhou, Tommy Duong, Zikang Zhang, Ethan Traister, and Kewen Xie. GPT-Image-2 in the wild: A Twitter dataset of self-reported AI-generated images from the first week of deployment. arXiv preprint arXiv:2604.25370, 2026.

[31] Yan Zhang, Simiao Ren, Ankit Raj, En Wei, Dennis Ng, Alex Shen, Jiayu Xue, Yuxin Zhang, and Evelyn Marotta. GPT4o-Receipt: A dataset and human study for AI-generated document forensics. arXiv preprint arXiv:2603.11442, 2026.

[32] Zirui Zhang, Yinbo Yu, Donghai Guan, Chunwei Tian, Daoqiang Zhang, and Qi Zhu. A benchmark dataset for MLLMgenerated image detection: GPT Image2 & Nano Banana2. arXiv preprint arXiv:2608.01258, 2026.

[33] Xiaoyu Zhou, Jianwei Fei, Peipeng Yu, Jingchang Xie, Chong Cheng, and Zhihua Xia. PGC: Peak-guided calibration for generalizable AI-generated image detection. arXiv preprint arXiv:2605.21207, 2026.

[34] Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, and Yunhe Wang. GenImage: A million-scale benchmark for detecting AI-generated image. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2023.

## Appendix

The appendix provides four groups of supporting material:

• Collection design: launch context (A), gateway accounting (B), differences among nine instruments (C), route overlap (D), queries (E) and attribution rules (F).

• Curation and audits: deduplication (G), the screenshot prompt (H) and historical audits (I).

• Dataset evidence: languages, timing and geometry (J), content credentials (K), the source landscape (L), prepared release files (M) and aggregate subject shares (N).

• Evaluation: detector execution, benchmark checks, crossrelease comparisons and collection limitations (O).

## A Model context at launch

On the Arena.ai blind-vote leaderboards snapshotted on 2026-09-07, a day before the announcement, both 2.5 models sat above GPT-Image-2 on editing (1520±9 and 1491±9 against 1461±3 Elo, on 6,704, 5,676 and 235,928 votes, for Sunburst, Flare and GPT-Image-2 respectively). On text-toimage, both also sat above GPT-Image-2, Flare by a narrow margin (1421±13 and 1399±13 against 1381±4). The system card [13] reported a lower unsafe-presented rate (1.09% and 1.41% against 1.64%).

## B The billed collection stack

The April collection used one scraper on one platform. Extending it required additional clients, credentials, pagination logic and rate-limit handling. Here, five of nine routes share a commercial gateway (Monid, CLI 0.1.7). Each call specifies a provider, endpoint and JSON query, reducing the client code needed for another platform.

The logs contain 932 calls across 5 endpoints, with latency and return code recorded per call. Table 5 lists counts and prices. Pixiv, note.com and NightCafe instead use public JSON endpoints directly, without a gateway charge.

Unit economics. The official X recent-search endpoint bills \$0.005 per post read; the same search through the gateway bills \$0.0015 for a page of twenty posts, about 67× cheaper per post. Xiaohongshu is \$0.015 per twenty notes and Weibo and Reddit \$0.0015 per page. These are list-price ratios for fully populated pages, before repeated posts, failed calls and deduplication; they are not ratios per unique released image. The first multi-platform crawl exhausted its \$0.97 balance, and Reddit and Instagram were added after a top-up (observed balance \$100). The official day-one X run cost \$10.05 for 2,009 posts. These costs reflect the recorded call schedule and may differ for another collector.

Observed route overlap. Of 1,524 gateway-admitted posts, 502 appear in the retained official result files; 168 pass both admission rules, and 334 were returned officially but rejected there. The remaining 1,022 gateway-admitted posts were not observed in the retained official results. Those files do not preserve every billed read, and query schedules and admission rules differ (Appendix D). These counts therefore describe recorded overlap, not a causal decomposition of index coverage.

Combining routes increases observed coverage. Reproduction still depends on the gateway’s prices, endpoint availability and retrieval behaviour. Direct public endpoints avoid that intermediary where they are available.

The gateway calls in Table 5 were logged by crawl/crawl\_all.py and support Section 3.2.

## C What differs between the nine instruments

Table 6 compares the routes, including query type, time filtering, media format and language metadata. These differences limit cross-platform comparisons even when the launch cutoff is shared.

## D The two X routes, and the two sources unlike the rest

The retained official results contain 502 of the 1,524 gatewayadmitted posts, or 33%. This is a lower bound on returned overlap: the curated files retain 1,292 posts with media, not all 2,009 billed reads.

The official creation-language classifier rejects 334 of those returned posts, including 318 that pass the version regex alone (Section 3.4). Both routes admit 168 posts: 55% of 305 official posts and 11% of gateway posts.

A further 1,022 gateway-admitted posts do not appear in the retained official files. Thus, the records distinguish 334 returned-but-rejected posts from 1,022 unobserved posts. They cannot uniquely assign unmatched posts to index coverage, because retention is incomplete and query schedules differ. Neither route is a census. All 139 gate-passing pilot posts also occur in the day-one run.

Reddit mainly contributes discussion in this sample. Across four queries, the crawl read 193 posts, of which 176 were post-launch and 88 named 2.5. Only 10 had a directly downloadable image. The other 78 (89%) were link posts, workflow write-ups or excluded preview.redd.it re-encodes. The 11 retained images document this route’s yield but do not provide a comparable platform sample.

On Instagram, 65 of 97 returned posts named 2.5, and all of those carried images. Access through a third-party hashtag scraper, however, gives it a different sampling frame from full-text search.

Table 5: The billed collection stack, from data/crawl/crawl\_log.jsonl: every gateway call, its endpoint, its measured mean wall-clock latency and the vendor’s list price. Pixiv, note.com and NightCafe are absent: they were collected directly, without gateway charges.
<table><tr><td>Source</td><td>Provider and endpoint</td><td>Calls</td><td></td><td>Mean s List price</td></tr><tr><td>X/Twitter</td><td>tikhub/twitter/web/fetch_search_timeline</td><td>273</td><td>6.6</td><td>$0.0015/20 posts</td></tr><tr><td>Reddit</td><td>tikhub/reddit/app/fetch_dynamic_search</td><td>116</td><td>4.7</td><td>$0.0015/page</td></tr><tr><td>Xiaohongshu</td><td>tikhub/xiaohongshu/app_v2/search_notes</td><td>212</td><td>3.9</td><td>$0.015/20 notes</td></tr><tr><td>Weibo</td><td>tikhub/weibo/web_v2/fetch_realtime_search</td><td>324</td><td>3.8</td><td>$0.0015/page</td></tr><tr><td>Instagram</td><td>apify  $/ a \mathrm { p i f y / i n s t a g r a m - h a s h t a g - s c r a p e r }$ </td><td>7</td><td>11.9</td><td>per actor run</td></tr><tr><td>Total</td><td></td><td>932</td><td></td><td></td></tr></table>

Table 6: The launch-time cutoff, deduplication and image-form filter are shared. Caption and host attribution use different admission rules; query instrument, time filter, timestamp resolution, served format, language field and creation-language requirement are not, so cross-source comparisons in the body are comparisons of instruments as much as of platforms. A hashtag scraper and a model-tagged gallery have different sampling frames from full-text search.
<table><tr><td>Source</td><td>Query</td><td>Server time filter</td><td>Timestamp</td><td>Served as</td><td>Language</td><td>Creation lang.</td></tr><tr><td>X, official</td><td>6 families, full text</td><td>start_time = anchor</td><td>second</td><td>resampled JPEG</td><td>API lang</td><td>required</td></tr><tr><td>X, gateway</td><td>6 strings, full text</td><td>since: day</td><td>second</td><td>resampled JPEG</td><td>API lang</td><td>not required</td></tr><tr><td>Xiaohongshu</td><td>4 strings, note search</td><td>one-week filter</td><td>second</td><td>transcoded webp</td><td>constant zh</td><td>not required</td></tr><tr><td>Weibo</td><td>keyword, realtime search</td><td>none</td><td>minute, relative</td><td>CDN JPEG</td><td>constant zh</td><td>not required</td></tr><tr><td>note.com</td><td>4 hashtags</td><td>none (newest first)</td><td>second</td><td>re-encoded by note</td><td>constant ja</td><td>not required</td></tr><tr><td>Pixiv</td><td>5 strings, caption search</td><td>none (newest first)</td><td>second</td><td>original bytes</td><td>constant ja</td><td>not required</td></tr><tr><td>Reddit</td><td>4 strings, post search</td><td>one-week filter</td><td>second</td><td>Reddit CDN JPEG</td><td>none (no field)</td><td>not required</td></tr><tr><td>Instagram</td><td>3 hashtags + 1 keyword</td><td>none (scraper walk)</td><td>second</td><td>Instagram CDN JPEG</td><td>none (no field)</td><td>not required</td></tr><tr><td>NightCafe</td><td>newest creations feed</td><td>local anchor cutoff</td><td>millisecond</td><td>CDN assets</td><td>none (no field)</td><td>host model field</td></tr></table>

## E Query families, dropped families and animated posts

The official-API query set descends from the six April families, adapted for a name with a decimal point: hashtags cannot contain a dot, so the community settled on compressed forms such as #GPTImage25, and the dotted name appears only as phrase-quoted free text. Four April families were kept (hashtags; English creation language with a name variant; Japanese creation signals on “generated with” and “tried making”; Chinese signals on “prompt” and “generated”), two were dropped, and two were added: a codename family targeting Flare and Sunburst, minus announcement phrasing, because those tokens are dominated by third-party platforms announcing availability, and a broad anti-noise family matching name variants minus comparison and release language.

The gateway and non-X sources use shorter query strings (Appendix C); the collection code records every string.

The two families dropped after the pilot are the featurename queries, which carry no version signal, and the versionsilent family pairing ChatGPT with prompt-sharing language— April’s largest source of images—which is unusable for a silently upgraded model: 35 of the 106 posts it returned in the pilot named GPT Image 2 explicitly. Dropping this family reduces coverage but avoids admitting posts without an explicit version signal.

Animated posts are kept: X serves GIFs as MP4, from which we extract frames at four per second, drop consecutive near-duplicates and cap each animation at twelve frames. The release contains 69 frames from 17 animations (one to 11 each)

alongside 3,409 stills; frames are marked in the manifest and correlated within an animation.

## F Version attribution: caption expressions and three tiers

Caption-attributed records are scored by the following caseinsensitive regular expressions on post text. Host-attributed NightCafe records instead require an allowed value in the platform’s model field. Both signals are given in Figure 6. The 2.0 signal is written in three alternatives: the prefixed form, whose negative lookahead keeps “Images 2.5” from firing as 2.0; a bare “2.0”, which in a corpus already naming 2.5 is the prior model and is what catches Chinese and Japanese comparison posts, where no English prefix precedes the number; and the compact gptimage2, guarded against “GPTimage2.5”.

The bare and compact 2.0 alternatives exclude comparison posts that name both versions. The stored caption records were re-gated with these expressions; regate.py applies the rule by attribution route: caption records use these expressions (and the stored classifier decision for official X), while NightCafe records retain admission only when the recorded endpoint, field and model value agree with an allowed hostmodel record. Host titles are not used as a fallback version signal.

## F.1 Three attribution tiers

The release identifies three tiers in gate\_tier: official-X caption attribution plus the April creation-language classifier

2.5signal: 2.5 signal:

```javascript
(image|images|gpt|chatgpt)[\s\-_]<sub>*</sub>2\.5|2\.5[\s\-_]<sub>*</sub>(flare|sunburst)|gptimage25|images25|
image25|sunburst|flare|image2_5|2_5
```

2.0 signal:

```javascript
(gpt[\s\-_]<sub>*</sub>image|images?)[\s\-_]<sub>*</sub>2(\.0)?(?![\.\d_])|(?<![\d.])2\.0(?![\d])|
gpt[\s\-_]<sub>*</sub>images?2(?![\d.\w])
```

Figure 6: Case-insensitive version-gate expressions applied to post text for caption-attributed records. Each signal is a single expression;   
the line break is typographic. Host-attributed NightCafe records are gated on the platform’s model field instead.

(371 images); caption-regex-only attribution (1,749 images); and host-recorded model attribution (1,358 images). The last tier also contains 12 separately labelled videos, excluded from image analyses.

NightCafe records are admitted when gptImageModel names gpt-image-2-5-sunburst or gpt-image-2-5-flare; caption silence is permitted in that tier. The host field is a platform assertion, not independent verification that the endpoint executed. The collector excludes records with its checked NSFW or underage moderation flags before curation.

Re-scoring the 1,749 caption-only images with the official tier’s text classifier admits 372 (21.3%); 1,296 lack creation language it recognises. This is a comparison of automated rules, not a validation: the classifier’s phrase lists are Englishand Japanese-led, and missing language fields limit its use. The three tiers must therefore remain selectable rather than interpreted as interchangeable evidence of generator identity.

## G Deduplication and cross-source overlap

Runs are merged in the order pilot, official day one, gateway X, Xiaohongshu, Weibo, Pixiv, note.com, Reddit, Instagram and NightCafe; the first occurrence wins. Exact repeats are removed on the (post, media, frame) key (274); near-duplicates are removed by a 64-bit average hash (8×8 grayscale) at Hamming distance ≤4 against every image kept so far (1,281).

Replaying that pass over the run directories splits the 1,281 into 570 images whose surviving match came from the same post and 711 from a different post, 27 of them GIF frames.

Of the 570 same-post drops, 302 are the same post re-fetched through a second run (gateway rows carry no media\_key, so only the hash stage catches them); the remaining 268 (255 stills, 13 GIF frames) are within-run samepost near-duplicates, possible candidates for a before-and-after pair posted as two attachments, although only 38 of those stills are X rows, the rest being carousel near-duplicates from the gallery-shaped sources.

The pass did not exempt any of them. Treating every such X still as an edit-pair member gives an unverified upper-bound scenario of 38 losses (Appendix O.3).

The merge order also explains why the day-one run’s 305 gate-passing posts become 160 in the merged manifest: 139 were already carried by the pilot and lose the tie to it, and 6 lost every image to hash collisions. The merged manifest therefore records no cross-run overlap; the pre-dedup overlap of Appendix D comes from the dedup\_audit.py script in the collection code.

The pre-dedup overlap across sources is measured separately: analysis/also\_seen\_in.py recomputes which other sources hold a near-duplicate of each released image, using the same 64-bit average hash at Hamming distance ≤4 the merge itself uses, so it reproduces the merge’s similarity criterion, not semantic identity.

259 of 3,478 released images (7.4%) were also seen on at least one other source: 217 on two, 36 on three, 4 on four and 2 on five. That script computes cross-source matches for the released images and stores them in a separate analysis output; these matches are not a column in metadata.csv.

## H The screenshot filter: prompt, cost and labels by source

Every gated image is labelled with gpt-5.4-mini through the Responses API (single-pass labels with default sampling, accumulated through the snapshot) at detail=low on a 768-pixel JPEG thumbnail, which cost 4,955,460 input and 188,971 output tokens over 6,490 calls (762 input tokens per labelled row; the call count exceeds the 6,344 gated images because the re-gate discarded rows the filter had already labelled).

At the model’s list price of \$0.75 per million input and \$4.50 per million output tokens, that is \$4.57 for the whole pass, or \$0.70 per thousand images. The fixed prompt asks for exactly one of six labels:

You are curating a dataset of images generated by   
ChatGPT Images 2.5. Classify this image into   
exactly one label: standalone\_generated: a   
single image as the model would output it. No   
app chrome, no browser, no message bubbles, not a   
phone/desktop screenshot. chat\_screenshot: a   
screenshot of ChatGPT / a chat app / a browser /   
a phone UI that contains a generated image inside   
it. collage\_or\_grid: two or more separate   
images stitched together (before/after, 2x2 grid,   
candidates side by side). photo\_of\_screen: a   
camera photo of a monitor or phone displaying an   
image. promo\_graphic: a marketing/announcement   
graphic (platform logo, “now available”, pricing   
tables, feature lists). other: anything else.   
Respond with JSON only: {"label": ...,   
"confidence": "high|medium|low", "why": ...}

## H.1 Labels by source

Table 7 gives the full breakdown summarised in Section 3.5. Instagram sits near the Xiaohongshu end (140 of 193 promotional). The collection design does not establish whether this reflects platform composition, the hashtag query, or other selection effects. Because the sources were queried with different instruments (Appendix C), that contrast describes what our queries returned in the collection window, not the platforms populations. Within X the official tier is also 11.1% promotional against 21.4% for the gateway rows, so part of every cross-source promotional contrast is gate tier, not platform.

## I Historical audit design and detail

The audits concern the 9 September curation, before the 10 September NightCafe expansion. Retained per-item files record seed 20260909, identifiers, sources and strata. They lack exact run timestamps and a complete immutable sampling frame. We therefore report stored populations rather than reconstructing them from the current manifest.

The screenshot frame contains 2,910 images: 1,382 standalone, 309 chat screenshots, 302 collages, 818 promotional graphics, 38 screen photographs and 61 other. Respective sample sizes are 66, 65, 65, 66, 38 and zero. Within each sampled stratum, the recorded random-sampling rule gives inclusion probability $n _ { h } / N _ { h }$

The 300 sampled images come from X (217), Xiaohongshu (59), Instagram (9), note.com (6), Weibo (5) and Pixiv (4). None comes from NightCafe. The taxonomy sample also excludes NightCafe and contains 90 X images among 100 items. Its full historical frame size cannot be recovered from the summary alone.

The hashes and reconstructed stratum metadata in audit-provenance.json identify retained evidence, not the missing historical frame. Neither audit establishes performance on the current cohort.

## I.1 Version gate, per language

The gate audit sampled 25 posts in each language-by-decision stratum. For English, Japanese, Chinese and other languages, respectively, the stored admitted stratum sizes are 377, 454, 309 and 100; rejected sizes are 510, 174, 221 and 82. Sampling probabilities are therefore 25 divided by these sizes.

Rejected posts exist only for official X. Within the admitted samples, the numbers of official-X posts are 11, 8, 3 and 2.

The stored summary’s official-X recall calculation reweights these realised subsets, although the original random draw used combined-tier language-by-decision strata. We do not report that quantity as a Horvitz–Thompson estimate. A design-consistent domain analysis with uncertainty, and a sufficiently sized official-X sample, would be needed before interpreting recall. Other tiers have no sampled rejected population.

Of 25 rejected texts per language, the auditor calls 10 English, 15 Japanese, 12 Chinese and 4 other-language texts creation claims. These judgements are about what text claims, not which generator produced the attached pixels.

## I.2 Screenshot filter, per class

Among images predicted standalone in the historical sample, model disagreement is 1.5% (0.3–8.1). Within other sampled classes, label agreement is 69.2% (57.2–79.1) for collage\_or\_grid, 62.1% (50.1–72.9) for promo\_ graphic, 90.8% for chat\_screenshot and 18.4% for the 38-image photo\_of\_screen census. The intervals describe model-label proportions in the sampled strata, not accuracy against ground truth.

Weighting by historical stratum sizes gives standalone recall only over the five sampled strata. The unsampled 61- image other stratum could add between zero and 61 auditorstandalone images; its contribution is unmeasured. We do not interpret the stored whole-pool implied share, which assigns this stratum no standalone contribution, as an identified population estimate.

## I.3 Subject taxonomy, per class

Over 100 historically sampled images, anime/illustrated agrees with the auditor on 68.2% of 44 images; text-graphic agrees on 23.5% and product/UI mockup on 37.5%. CLIP assigns text-graphic 17 times versus the auditor’s 6, and photorealistic portrait 5 times versus 29. These sample contrasts motivate caution about class prevalence; they neither calibrate the expanded release nor bound its true category shares.

Content-analysis settings. Subject classification uses CLIP ViT-L/14 [17] zero-shot over the eight April category names with newly authored templates; the original April templates were unavailable, so the backbone and category names do not establish an identical classifier.

EasyOCR is routed by post-language metadata: Japanese labels use Japanese and English, Chinese labels use simplified Chinese and English, and all other or missing labels use English only, so the 1,409 images without language metadata all receive English-only OCR. Nonempty detections count at confidence at least 0.3. Faces use InsightFace buffalo\_l.

Summaries cover successfully analysed released non-video images; video decoding failures are excluded rather than counted as zero detections. April’s analyses ran on a set that still contained screenshots and comparison grids, both text-bearing by construction, so every April comparison is between a cleaned and an uncleaned set under differing CLIP templates.

## I.4 Language field and creation claims

A separate historical text audit sampled 50 released posts per language in English, Japanese and Chinese, using gpt-5.5 and seed 20260909. The auditor agrees with the recorded language on 99.3% of 150 sampled posts (English 100.0%, Japanese 100.0%, Chinese 98.0%). It reads 67.3% (59.5– 74.3) as the author presenting an image they made: Japanese

Table 7: Screenshot-filter labels by source over all gated images; only the standalone column gives automated admissions before the 16 targeted review exclusions; the last column is that automated share. X is split by gate tier: official-API rows also passed the creation-language classifier, gateway and other caption-attributed rows passed the version regex alone; NightCafe uses the host model field. Videos are excluded. Reddit’s 18 gated images are too few to read as a rate.
<table><tr><td>Source</td><td>Gated</td><td>Standalone</td><td>Chat</td><td>Collage</td><td>Promo</td><td>Screen photo</td><td>Other</td><td>Standalone %</td></tr><tr><td>X, official caption</td><td>548</td><td>371</td><td>34</td><td>72</td><td>61</td><td>3</td><td>7</td><td>67.7</td></tr><tr><td>X, caption regex</td><td>2,091</td><td>1,130</td><td>195</td><td>246</td><td>447</td><td>31</td><td>42</td><td>54.0</td></tr><tr><td>Xiaohongshu</td><td>1,594</td><td>340</td><td>198</td><td>118</td><td>865</td><td>23</td><td>50</td><td>21.3</td></tr><tr><td>Weibo</td><td>165</td><td>40</td><td>26</td><td>10</td><td>80</td><td>4</td><td>5</td><td>24.2</td></tr><tr><td>note.com</td><td>262</td><td>138</td><td>48</td><td>25</td><td>43</td><td>2</td><td>6</td><td>52.7</td></tr><tr><td>Pixiv</td><td>72</td><td>66</td><td>0</td><td>2</td><td>3</td><td>0</td><td>1</td><td>91.7</td></tr><tr><td>Instagram</td><td>193</td><td>40</td><td>0</td><td>11</td><td>140</td><td>0</td><td>2</td><td>20.7</td></tr><tr><td>Reddit</td><td>18</td><td>11</td><td>1</td><td>2</td><td>3</td><td>1</td><td>0</td><td>61.1</td></tr><tr><td>NightCafe</td><td>1,401</td><td>1,358</td><td>0</td><td>20</td><td>19</td><td>0</td><td>4</td><td>96.9</td></tr><tr><td>Total</td><td>6,344</td><td>3,494</td><td>502</td><td>506</td><td>1,661</td><td>64</td><td>117</td><td>55.1</td></tr></table>

82.0%, English 66.0%, Chinese 54.0% (40.4–67.0). These are balanced-sample results; the pooled rate does not estimate the current release’s creation-claim share. The audit does not cover records without a language field, including NightCafe, and it does not validate the wording regexes as measures of user intent.

## J Languages, timing and image geometry

Table 8 gives the language distribution of the gated pool against the release. The 1,501 X rows carry the API’s detected lang; the 568 rows (16.3%) from Xiaohongshu and Weibo (365, zh) and Pixiv and note.com (203, ja) are assigned a platform default by construction; the 1,409 Instagram and Reddit and NightCafe rows carry none, their collectors returning no such field.

Comparison with April’s 32.8% Japanese share is affected by four construction choices: differential filter retention, the addition of two Japanese-first and two Chinese-first sources, platform-default labels, and missing language fields on 40.5% of rows. Changes in early adoption cannot be separated from these choices. The field also describes the post, not the image.

## J.1 Timing and wording

Table 9 reports the earliest and latest posting timestamps among each source’s released images. These observed ranges do not establish complete query coverage. For NightCafe, created\_at is normalised from the host’s postedDate; retrieval and generation times are different quantities. The 10 September collection paginated the newest-items feed retrospectively, so its retrieval date does not date every retrieved post. Figure 7 shows images by posting hour and caption wording by language.

## J.2 Image geometry as served

Served dimensions are recorded for 371 official-API rows (10.7%). The following statistics apply only to that subset. Its images occupy 91 pixel sizes. Of these rows, 62 match an API-native size (1024 × 1024, 1024 × 1536 or 1536 × 1024), and 85 match a ChatGPT export size. Portrait images account for 51.5%, landscape for 23.7% and square for 24.8%.

Dimensions alone identify neither the original output nor later transformations. Other sizes may result from editing, export or platform resampling, while native sizes do not establish unchanged bytes. Generator-level frequency analysis requires a verified delivery path and transformation history.

## K Content credentials in detail

A pilot inspected 5 Pixiv originals: 2 PNGs had manifests whose signature chains were verified as OpenAI’s, declaring trained algorithmic media and a watermark assertion. The remaining three were JPEG re-saves with no manifest detected. The correspondence between the two verified pilot files and the current release has not been established, so the pilot does not supply a verified-signature count for the current subset.

A byte scan of all 66 released Pixiv files found C2PA markers in 31 (47%). Of these, 30 matched the searched gpt-image version 2.0 string, an OpenAI string and a watermark string. One marker-positive file lacked that combination and remains uncharacterised. Byte-marker and string matches do not verify a manifest or its signature.

The separate API ledger covers 1,007 files from the three recorded endpoints on 2026-09-09; the inspected generatorversion field is 2.0 throughout that ledger. This establishes a limitation of that field in those files, not all provenance assertions or all endpoints.

Because the inspected field does not distinguish versions, it cannot confirm that the caption-attributed Pixiv images were generated by 2.5. Their version attribution remains captionbased. Manifest presence is not an admission signal or a released CSV field. The expanded NightCafe cohort has not received the same credential inspection, so we do not infer whether it preserves or strips C2PA.

## L The source landscape, platform by platform

Social and creator platforms. X, Reddit, Weibo, Xiaohongshu, Instagram and Threads expose caption-based evidence.

Table 8: Language distribution, gated pool against release. “Platform default” counts released rows whose language is a per-source constant rather than a detected field; the “no field” row includes NightCafe, Reddit and Instagram. Retention is the released share of the gated count.
<table><tr><td>Language</td><td>Gated</td><td>Released</td><td>of which X (detected)</td><td>of which platform default</td><td>Retention</td></tr><tr><td>English (en)</td><td>1,082</td><td>675</td><td>675</td><td>0</td><td>62.4%</td></tr><tr><td>Japanese (ja)</td><td>1,265</td><td>703</td><td>500</td><td>203</td><td>55.6%</td></tr><tr><td>Chinese (zh)</td><td>2,258</td><td>631</td><td>266</td><td>365</td><td>27.9%</td></tr><tr><td>Other (French 19, 12 others, 1 undetermined)</td><td>127</td><td>60</td><td>60</td><td>0</td><td>47.2%</td></tr><tr><td>No field (Instagram and Reddit and NightCafe)</td><td>1,612</td><td>1,409</td><td>0</td><td>0</td><td>87.4%</td></tr><tr><td>Total</td><td>6,344</td><td>3,478</td><td>1,501</td><td>568</td><td>54.8%</td></tr></table>

Table 9: Observed posting ranges among released images by source (UTC). Ranges describe retained records, not known complete collection windows or generation times.
<table><tr><td>Source</td><td>Images</td><td>First observed posting (UTC)</td><td>Last observed posting (UTC)</td></tr><tr><td>X</td><td>1,501</td><td>2026-09-0818:58:45</td><td>2026-09-1021:47:00</td></tr><tr><td>Xiaohongshu</td><td>325</td><td>2026-09-0819:24:47</td><td>2026-09-1012:28:43</td></tr><tr><td>Weibo</td><td>40</td><td>2026-09-09 00:26:00</td><td>2026-09-1010:15:00</td></tr><tr><td>note.com</td><td>137</td><td>2026-09-09 02:15:11</td><td>2026-09-1018:05:19</td></tr><tr><td>Pixiv</td><td>66</td><td>2026-09-08 21:11:51</td><td>2026-09-1013:57:04</td></tr><tr><td>Instagram</td><td>40</td><td>2026-09-08 20:12:43</td><td>2026-09-1017:53:47</td></tr><tr><td>Reddit</td><td>11</td><td>2026-09-09 02:36:14</td><td>2026-09-1018:13:34</td></tr><tr><td>NightCafe</td><td>1,358</td><td>2026-09-10 05:25:55</td><td>2026-09-10 22:06:41</td></tr></table>

Our source observations come from probes on 9–10 September recorded in our collection notes, not an exhaustive survey. Pixiv contributes 66 released images from 18 works; all 66 carry its structured AI declaration. The declaration is still supplied by the uploader and does not name a generator version. Pixiv originals offer a byte-preserving delivery route in the inspected files, whereas note.com and social CDN paths can re-encode images. We do not extrapolate a tested delivery path to every asset on a platform. Credential inspection coverage is given in Appendix K.

Model-tagged galleries. A host-recorded model field provides an attribution route distinct from caption claims. Night-Cafe’s public creations endpoint exposes gptImageModel; the 9 September probe observed no 2.5 items among 240 newest creations. The 10 September crawl collected 1,449 creations bearing allowed 2.5 values. These snapshots show a change in observed feed contents, not a precisely measured adoption time.

The frozen snapshot includes 1,358 NightCafe images and 12 separate videos. A caption-recall study could compare text against these host labels within this gallery, but would still depend on their validity and would not estimate recall on other platforms.

On 10 September, SeaArt’s 2 model cards reported 1,631 and 1,161 generation tasks but 0 published works. The imagine.art probe returned no items for the 2.5 style identifiers 41901 and 41902, nor among 500 newest posts. Its style\_id filter returned expected matches on control identifiers; a separate model\_id parameter did not change the results and cannot be used as attribution evidence. These are query-specific observations of no items, not evidence of no platform adoption.

Mage.space’s public model counts rose from 117 on 2026-09-09 to 221 on 2026-09-10 (149 Sunburst, 72 Flare), while its per-image feed remained behind login and was not collected.

API resellers and aggregators. The same survey found third-party access routes including fal.ai, Kie.ai, OpenArt, Mage.space and Krea. Availability of generation does not establish a public, attributable gallery. OpenArt’s observed model-filter results did not consistently carry a corresponding per-item model field, so we did not admit them on that basis. Intermediary re-encoding is a separate provenance concern that must be checked against the actual returned bytes.

Collection implications. Collect accessible social and creator sources in parallel, recording their distinct gates and delivery paths. Poll model-tagged galleries during the launch week: NightCafe demonstrates that a single empty probe can miss a substantial later cohort. Preserve dates and query parameters for negative observations, and keep host attribution separate from caption attribution. Choice of an official API or gateway should depend on the observed yield, cost assumptions and reproducibility of each route rather than an assumed universal ranking of their coverage.

## M What the release directory contains

The prepared local release directory contains the image cohort and separate video assets, a flat metadata.csv with 22 columns, and a datasheet. Each row carries a content-hash identifier, post identifier and URL (the legacy names tweet\_id and tweet\_url apply across sources), recorded posting time, language, media type, frame index, served dimensions, confirmation route, version signal, screenshot label, query family, source\_platform, pixiv\_ai\_type, like count and post text (the host’s public creation title for host-attributed rows, not a verified generation prompt).

![](images/48aaf15b040d3ffebb05b1b89f77d60b319f03b724235df9dd7d560f032231ea.jpg)

![](images/27f342b7db5a4fd40d7ebe66e506fad9e8277b30f2395736a285a982ff724bd8.jpg)  
Figure 7: Left: released images by recorded posting hour in the 2026-09-08 18:58:45Z–2026-09-10 22:06:41Z window (51.1 hours), stacked by recorded post language, with missing language shown separately. Differing source coverage prevents an adoption-rate interpretation. Right: share of released image posts with nonempty text whose text matches prompt-sharing, editing, speed, consistency and codename patterns, for English (364 posts), Japanese (360) and Chinese (299); these are descriptive text-pattern matches, and the other recorded languages have too few posts to plot.

The selectable gate\_tier separates official-X caption plus classifier, caption-regex-only, and host-recorded model attribution. The source\_run field records the surviving collection run. Host records additionally carry model\_id, attribution\_endpoint and attribution\_field. No author profile fields are exported. The datasheet states the frozen window, image/video counts and three admission rules. Public hosting is planned. Analyses of images exclude the video rows using media\_type.

C2PA signature status, per-item human adjudication and full reconstruction of the historical audit sampling frames are not provided. Cross-source hash matches remain analysis outputs rather than an attribution guarantee. These limitations should be retained when selecting a cohort or interpreting a released label.

## N Aggregate subject comparison

Figure 8 compares CLIP-assigned subject shares with the three categories reported in April. The differing templates and image-form filters limit this descriptive comparison.

## O Detector Evaluation Details

Selection and controls. The experiment uses the public Community Forensics 384-pixel checkpoint, B-Free’s BFREE\_ dino2reg4/model\_epoch\_best.pth, and Effort’s genimage\_checkpoint.pth, plus PGC’s PGC\_ train\_sdv1\_4\_ckpt.pth, PROBE’s ResNet50\_ best\_model\_step\_19343.pth, and DoU’s released best.pth. Checkpoints are selected before inspecting collection scores. Photograph pools derive from MLap/GeoDE, tanganke/sun397, HuggingFaceM4/FairFace, nateraw/pascal-voc-2012, and ethz/food101; artwork controls derive from huggan/wikiart. Selection orders candidate paths by SHA256 of the fixed prefix paper-w-controls-v1: followed by the path, retains the first 400 byte-distinct images per source, and alternates calibration/test assignment. Manifests retain source, split, image identity, and byte hashes.

![](images/41697a90d94eccba2d9373fa500e5e3aa434be4d23b20e096daefcc2b6df8333.jpg)  
Figure 8: Subject-matter distribution of the release under CLIP ViT-L/14 zero-shot classification into the eight April categories using newly authored templates. April shares are shown for the three classes that paper reported, computed on an uncleaned set that still contained screenshots and grids. Read the shares as CLIP’s: an earlier-cohort audit, excluding NightCafe, found κ = 0.51 and substantial disagreements on text-graphic and photorealistic portrait (Section 4).

Execution and checks. All inference is FP32 with TF32 disabled. Community Forensics uses its official image processor at 384 pixels; B-Free preserves native patch processing; Effort uses RGB conversion, OpenCV linear resizing to 224 × 224, and CLIP normalization. B-Free’s batched adapter and Effort’s folded frozen SVD weights matched upstream CPU forward outputs exactly on the recorded validation cases; Effort preprocessing also matched upstream on 112 pilot images. Models run sequentially on one RTX 3090, using batches of 64, 8, and 32 and respectively six, four, and four preprocessing workers with pinned memory and prefetching.

PGC preserves the official native-pixel padding and center crop at 224 pixels, appends three quantization-residual channels, and normalizes RGB and residual channels separately. It uses batches of 32 with four preprocessing workers. Its strict load consumed all 563 checkpoint keys, and its transform and forward path matched the upstream evaluator exactly on four recorded input images.

PROBE-ResNet50 strictly loads its 320 checkpoint keys and retains native 224-pixel patch extraction, averaging patch logits before sigmoid. Small inputs use the official black-padding policy. Image batches of eight feed GPU microbatches of 32 patches.

DoU retains its official CLIP-based transform and stochastic forward, using batches of 32. Both use four preprocessing workers. Their adapters were checked against upstream transforms and forwards.

The September-plus-controls runs take approximately 67, 644, 78, 78, 154, and 75 seconds. All 35,268 primary scores are finite, with exact expected image-ID coverage. Recorded checkpoint, code, and manifest hashes accompany the score files.

Calibration and uncertainty. For sorted calibration logits $s _ { ( 1 ) } \leq \cdots \leq s _ { ( n ) }$ , the threshold is set to the next representable double above $^ { S } ( n - \lfloor 0 . 0 5 n \rfloor ) \dot { , }$ scores at or above the threshold are flagged. This rule conservatively handles ties and flags exactly 50 of 1,000 calibration photographs for each detector. Table 2 and the source-level rates use 1,000 post-clustered bootstrap draws; the cross-cohort comparison uses 50,000 draws per cohort.

Held-out photograph FPR Wilson intervals are 4.6–7.5%, 3.1–5.6%, 2.5–4.8%, 3.9–6.6%, 4.1–6.9%, and 3.3–5.9%; artwork intervals are 35.4–48.9%, 57.1–70.3%, 5.4–13.2%, 93.0–98.3%, 0.5–4.3%, and 7.4–16.1%, in table order. All intervals condition on the fitted calibration thresholds.

Overlap sensitivity. No byte-identical overlap was found between collection and control inputs, and no decoded-pixel identity was found across the control calibration/test splits. A dHash distance screen $( \le 3 )$ identified three cross-split Fair-Face candidate pairs, not established duplicates. Removing all six candidate images and recalibrating on 997 photographs yields held-out photograph FPRs of 6.02%, 4.21%, 3.51%, 5.02%, 5.12%, and 4.31% on 997 images, respectively. Residual perceptual overlap and detector-training overlap are not excluded by these checks.

Exploratory content restriction. Restricting the collection to 736 images assigned the existing CLIP categories portrait, food, nature/landscape, or architecture gives AUROCs of 0.841, 0.535, 0.929, 0.846, 0.538, and 0.695 against the 1,000 held-out reference photographs. These are classifierdefined content proxies, not human-validated photographic images or semantically matched controls. The comparison remains secondary to the source-resolved flag rates and does not independently validate image origin.

DoU stochastic evaluation. DoU samples latent Gaussian noise in its published evaluation forward pass. We preserve this behavior, using seed 42 for the primary run and recording batch size 32. A second full pass with seed 43 checks sensitivity without changing the primary result. With each pass calibrated on the same reference photograph split, collection flag rates are 17.4% and 17.9%, and held-out photograph FPRs are 4.4% and 6.9%.

Holding the primary threshold fixed instead gives a seed-43 flag rate of 17.8% and photograph FPR of 6.6%; 235 of 3,478 collection images change verdict.

These two realizations do not estimate the full distribution over seeds. Post-clustered bootstrap intervals omit this additional stochastic variation. Partial resume is rejected rather than silently reseeding and skipping previously scored images.

## O.1 Harness reproduction on GenImage

Purpose of the benchmark check. Forward-pass parity and control FPRs test different parts of the implementation, but neither validates the full scoring pipeline. Each adapter matches the official forward pass bit for bit on identical tensors. Both sides, however, receive already-preprocessed inputs, so this check cannot detect a shared resizing error.

Held-out photograph FPRs of 3.5–5.9% show that the scores can be calibrated on real photographs. They include no generated image with a known label. Together, these checks can still miss errors such as inverted score direction or an omitted centre crop, which can yield plausible flag rates on an unlabelled collection.

Sample and comparability. GenImage [34] is a common benchmark for these detectors. Community Forensics reports results on it, Effort’s checkpoint targets it, and PROBE and PGC train on its SD v1.4 split. We evaluate a frozen validation sample with 300 real and 300 generated images from each of 7 available generators (4,200 images). It is intended to detect large implementation errors while accepting the sampling uncertainty of a small subset. No fitting or tuning uses this sample.

Table 10: Collection flag rates (%) by source, with half-widths of 95% post-clustered bootstrap intervals. CF is Community Forensics and PROBE is PROBE-ResNet50. Each detector uses the same photograph-calibrated threshold across all sources. Intervals resample posts and condition on the fitted threshold. Small sources are thin: Reddit contributes 11 images from 4 posts. Differences describe the collected sample, not the causal effect of publishing on a platform. DoU uses seed 42; intervals omit variability from its latent-noise sampling.
<table><tr><td>Source</td><td>Images</td><td>Posts</td><td>CF</td><td>B-Free</td><td>Effort</td><td>PGC</td><td>PROBE</td><td>DoU</td></tr><tr><td>X</td><td>1,501</td><td>861</td><td> $4 4 . 3 \pm 3$ </td><td> $1 9 . 3 \pm 3$ </td><td> $2 8 . 8 \pm 3$ </td><td> $4 3 . 2 \pm 3$ </td><td> $3 . 8 \pm 1$ </td><td> $1 1 . 1 \pm 2$ </td></tr><tr><td>NightCafe</td><td>1,358</td><td>1,358</td><td> $6 4 . 1 \pm 3$ </td><td> $8 . 2 \pm 1$ </td><td> $5 2 . 6 \pm 3$ </td><td> $5 6 . 0 \pm 3$ </td><td> $2 . 4 \pm 1$ </td><td> $3 . 5 \pm 1$ </td></tr><tr><td>Xiaohongshu</td><td>325</td><td>133</td><td> $5 0 . 8 \pm 6$ </td><td> $2 0 . 3 \pm 5$ </td><td> $6 1 . 5 \pm 8$ </td><td> $9 6 . 9 \pm 2 $ </td><td> $3 . 1 \pm 3$ </td><td> $5 3 . 8 \pm 6$ </td></tr><tr><td>note.com</td><td>137</td><td>29</td><td> $7 0 . 1 \pm 1 3$ </td><td> $6 . 6 \pm 5$ </td><td> $7 8 . 1 \pm 1 1$ </td><td> $9 2 . 7 \pm 6 $ </td><td> $1 . 5 \pm 2$ </td><td> $9 3 . 4 \pm 5 $ </td></tr><tr><td>Pixiv</td><td>66</td><td>18</td><td> $8 1 . 8 \pm 1 6$ </td><td> $4 5 . 5 \pm 3 1$ </td><td> $3 7 . 9 \pm 2 5$ </td><td> $8 4 . 8 \pm 1 2$ </td><td> $3 1 . 8 \pm 3 1$ </td><td> $6 0 . 6 \pm 3 6$ </td></tr><tr><td>Instagram</td><td>40</td><td>18</td><td> $3 5 . 0 \pm 1 9$ </td><td> $1 0 . 0 \pm 1 5$ </td><td> $4 7 . 5 \pm 2 2$ </td><td> $4 7 . 5 \pm 2 4 $ </td><td> $1 0 . 0 \pm 1 3$ </td><td> $1 7 . 5 \pm 1 4$ </td></tr><tr><td>Weibo</td><td>40</td><td>19</td><td> $7 0 . 0 \pm 1 8$ </td><td> $2 2 . 5 \pm 1 6$ </td><td> $4 5 . 0 \pm 2 5$ </td><td> $6 2 . 5 \pm 2 2$ </td><td> $5 . 0 \pm 8$ </td><td> $7 7 . 5 \pm 2 2$ </td></tr><tr><td>Reddit</td><td>11</td><td>4</td><td> $9 0 . 9 \pm 1 5 $ </td><td> $0 . 0 \pm 0$ </td><td> $4 5 . 5 \pm 5 0$ </td><td> $8 1 . 8 \pm 3 0$ </td><td> $0 . 0 \pm 0$ </td><td> $8 1 . 8 \pm 3 0$ </td></tr><tr><td>All sources</td><td>3,478</td><td>2,440</td><td> $5 4 . 7 \pm 2 $ </td><td> $1 5 . 0 \pm 2$ </td><td> $4 3 . 7 \pm 2$ </td><td> $5 6 . 4 \pm 2 $ </td><td> $3 . 7 \pm 1$ </td><td> $1 7 . 4 \pm 2$ </td></tr></table>

Frozen sample and metric. Within each generator and class, files are ranked by SHA256 of paper-w-genimage-v1: followed by generator, class and filename. We retain the first 300. Selection reads filenames only and finishes before inference. Our copy contains 7 of the eight published subsets; SD v1.4 is absent. The training-domain column for PROBE and PGC is therefore not included.

Accuracy uses each detector’s own 0.5 boundary. Equal real and generated counts make accuracy identical to balanced accuracy within each generator. We recompute published means over the same available generators. Community Forensics is an exception: its published aggregate spans all eight, including the missing SD v1.4 subset, so that comparison is not exactly matched.

Published references. Community Forensics, PROBE, PGC and Effort references come from their authors: [15] Table 2, [1] Table 2, [33] Table 2 and [28] Table 9, respectively. B-Free reports only averages on a recompressed GenImage variant; its reference comes from the PGC study, [33] Table 2. DoU reports GenImage results only graphically.

Residual differences. Each generator contributes 600 images, compared with twelve thousand in the full validation split. Our estimates are therefore noisier and use a different composition. Differences above one point run both ways: ours is higher in six cells, mostly Effort, and lower in five, mostly PROBE-ResNet50. The generated report records pergenerator values, differences and standard errors.

Reference provenance matters. Third-party Effort figures are more than ten points below its own reported results. Using them would imply a roughly fifteen-point discrepancy where the comparison with the authors’ result is much closer.

Community Forensics evaluation set. We also evaluate Community Forensics on a separately released, labelled heldout set [15]. A frozen balanced sample contains 10,000 images (5,000 per class) across 21 generators. Files are copied byte for byte from shards containing 51,836 released images.

The harness reaches pooled average precision (AP) 0.989 and accuracy 89.8%, compared with published mean AP 0.994 and accuracy 0.912. The release does not fully specify the published AP aggregation. Averaging per-generator AP on our sample gives 0.947, rather than pooled AP 0.989. Each generator is then compared with the entire real pool, producing generated prevalence of roughly 5%. Because AP depends on prevalence, this is not directly comparable with the balanced published evaluation.

Pooled AP on the balanced sample is the closest comparison we can construct. It falls about half an AP point below the published value, which lies outside our bootstrap interval [0.987, 0.990]. Accuracy likewise falls short, and the residual gap is unresolved. Averaging within released shards gives 0.989 and does not close the gap.

The harness calls the authors’ image processor in test mode, so preprocessing is not independently reimplemented. Our sampling and the published pairing of real and generated images remain possible sources of difference. On this same set, the other five detectors reach AUROCs of at least 0.850. Those out-of-distribution results lack comparable published references. We report them only as evidence against inverted or chance-level scoring.

Metric computation. AP is pooled unless otherwise stated. Community Forensics evaluation intervals use 2,000 imagelevel bootstrap draws with seed 20260912. This set has no post-level grouping analogous to the online collection. Comparisons with balanced published sets use balanced samples because AP depends on prevalence.

The AP and AUROC implementations are checked against scikit-learn in our test suite, using continuous, integertied and rounded scores. Ties matter for saturated detectors: cumulative-sum AP that arbitrarily separates equal scores can disagree with the standard metric. Such an error would otherwise resemble disagreement with a published result.

Table 11: Per-generator accuracy (%) on the frozen GenImage sample and published reference values. Agreement in per-generator patterns provides additional evidence beyond the mean; it does not prove checkpoint identity.
<table><tr><td>Detector</td><td></td><td>ADM</td><td>BigGAN</td><td>Midjourney</td><td>SD1.5</td><td>VQDM</td><td>Wukong</td><td>Glide</td><td>Mean</td></tr><tr><td rowspan="2">Community Forensics</td><td>ours</td><td>93.0</td><td>87.5</td><td>85.3</td><td>100.0</td><td>100.0</td><td>99.7</td><td>99.7</td><td>95.0</td></tr><tr><td>published</td><td>一</td><td>一</td><td></td><td></td><td>一</td><td></td><td>一</td><td></td></tr><tr><td rowspan="2">B-Free</td><td>ours</td><td>78.0</td><td>67.7</td><td>95.8</td><td>99.2</td><td>88.3</td><td>97.5</td><td>84.2</td><td>87.2</td></tr><tr><td>published</td><td>78.1</td><td>68.6</td><td>94.9</td><td>98.9</td><td>88.3</td><td>98.8</td><td>83.8</td><td>87.3</td></tr><tr><td rowspan="2">Effort</td><td>ours</td><td>84.8</td><td>92.7</td><td>84.2</td><td>98.3</td><td>93.7</td><td>98.0</td><td>93.2</td><td>92.1</td></tr><tr><td>published</td><td>78.7</td><td>77.6</td><td>82.4</td><td>99.8</td><td>91.7</td><td>97.4</td><td>93.3</td><td>88.7</td></tr><tr><td rowspan="2">PGC (2026)</td><td>ours</td><td>100.0</td><td>98.2</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.7</td></tr><tr><td>published</td><td>100.0</td><td>86.7</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>98.1</td></tr><tr><td rowspan="2">PROBE-ResNet50</td><td>ours</td><td>57.0</td><td>49.8</td><td>92.0</td><td>99.3</td><td>62.3</td><td>99.7</td><td>60.8</td><td>74.4</td></tr><tr><td>published</td><td>60.0</td><td>49.4</td><td>90.7</td><td>99.2</td><td>66.9</td><td>99.4</td><td>63.0</td><td>75.5</td></tr><tr><td rowspan="2">DoU</td><td>ours</td><td>83.3</td><td>75.3</td><td>72.3</td><td>100.0</td><td>98.2</td><td>99.5</td><td>92.5</td><td>88.7</td></tr><tr><td>published</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr></table>

Scope. A released checkpoint can differ from the one used for a paper’s table. These checks cannot separate that possibility from a harness fault, so agreement supports implementation correctness without proving it. They also do not validate generator attribution or platform-specific preprocessing in the collected images.

## O.2 Historical-cohort comparison

Frozen April input. We retrieve the published GPT-Image-2 dataset at revision a8e3f74cd02edcab94d8c61ae0 439a9dc5d78468, preserving all 10,217 metadata rows. The archive places 219 images in images\_overflow/ although metadata paths point to images/. Unique filenames resolve them without substituting files. The manifest retains release order, post IDs, admission routes, languages, classifications and both byte and decoded-RGB hashes.

Neither the sample nor the complete 10,217-image release overlaps September images or reference controls at either hash level. We do not apply September’s screenshot filter retrospectively. Original classifications are 4,959 confirmed and 5,258 uncertain images; recorded routes include 4,750 generic AI-badge admissions and 508 name-only admissions. These labels do not authenticate generator identity.

Posts span 21–28 April 2026. Recomputed image-row language counts are 4,117 English, 3,355 Japanese and 1,959 Chinese: 40.3%, 32.8% and 19.2% of 10,217 rows. These reproduce the April shares cited in this appendix. We do not pool them with September language fields, whose sources, defaults and missingness differ.

Sample selection. All six detectors use the same 3,000 images from 2,542 posts. We rank the release by SHA256 of paper-w-april-pilot100-v1: followed by image ID and retain the first 3,000 distinct records. The prefix is inherited from the initial pilot and kept unchanged for reproducibility. Inference follows original release order.

Selection precedes scoring and uses no quotas or scorebased exclusions. It samples image records without replacement but does not guarantee balanced content or languages.

Parent and sample manifest hashes and selected IDs are recorded. The report checks score and manifest hashes, checkpoint identity, preprocessing, score definitions, package versions and batch configuration against September. Thresholds come from the saved September calibration report and are never refitted on April.

Intervals and multiplicity. Each bootstrap draw samples source–post pairs with replacement within a cohort and retains all images of each selected post. The rate is the resampled flag count divided by the resampled image count. Cohorts are resampled independently, and difference intervals use percentiles of draw-wise differences. The report uses seed 20260912, deterministic detector-specific streams and 50,000 draws to estimate the family-wise tail bounds.

The six contrasts are exploratory. We report pointwise intervals and, for the two excluding zero, Bonferroni-adjusted intervals for the family of 6. Shared creators and residual image similarity can produce dependence beyond the post clusters.

Attribution sensitivity. The metadata-only view restricts April to sampled records labelled confirmed, a label held by 4,959 of 10,217 parent images. September is restricted to 371 X images in official\_caption\_classifier. These admission rules are neither matched nor independently verified. We do not compute difference intervals for this view. Per-cohort intervals can rely on as few as seven PROBE-ResNet50 flags, for which percentile-bootstrap coverage is unreliable.

Duplicate and seed sensitivity. The 3,000 records contain 17 byte-duplicate groups with 19 extra images; the parent release has 122 groups and 145 extra images. The pre-specified firstoccurrence sensitivity retains 2,981 records. It changes every flag rate by less than 0.1 pp: Community Forensics 44.80 to 44.78, Effort 22.70 to 22.61, and DoU 7.57 to 7.48. No conclusion changes.

DoU uses primary seed 42 on April. The September-only second-seed check does not quantify variability in the crosscohort difference. Neither a favourable seed nor an attribution subset replaces the primary comparison.

Table 12: Sensitivity to calibration sampling. Differences are September X minus April X, in percentage points. Fixed-threshold intervals reproduce the primary analysis; the other intervals additionally resample calibration photographs within source and refit the threshold. Both analyses resample whole posts and use 50,000 draws. The 99.17% intervals adjust for six detectors; DoU’s latent-sampling variability is excluded.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Difference (pp)</td><td rowspan="2">Fixed threshold 95% interval</td><td colspan="2">Resampled calibration</td></tr><tr><td></td><td>95% interval 99.17% interval</td></tr><tr><td>Community Forensics</td><td>-0.5</td><td>[-4.1, +3.2]</td><td>[-4.0, +3.4]</td><td>[-5.3, +4.6]</td></tr><tr><td>B-Free</td><td>+1.3</td><td>[-1.7, +4.3]</td><td>[-1.8, +4.2]</td><td>[-2.9, +5.3]</td></tr><tr><td>Effort</td><td>+6.1</td><td>[+2.7, +9.6]</td><td>[+2.8, +10.0]</td><td>[+1.7, +11.3]</td></tr><tr><td>PGC (2026)</td><td>+0.8</td><td>[-3.1, +4.7]</td><td>[-2.9, +5.4]</td><td>[-4.4, +6.8]</td></tr><tr><td>PROBE-ResNet50</td><td>+0.9</td><td>[-0.3, +2.2]</td><td>[-0.3, +2.3]</td><td>[-0.7, +2.8]</td></tr><tr><td>DoU</td><td>+3.6</td><td>[+1.0, +6.2]</td><td>[+1.1, +6.3]</td><td>[+0.3, +7.4]</td></tr></table>

Resampling the calibration photographs. We add a post hoc sensitivity analysis after the primary comparison. Each of 50,000 draws resamples 200 calibration photographs within each of the five source pools, preserving the fixed source allocation, and refits the same conservative 5% threshold. It independently resamples April and September posts, retaining every image in each sampled post. Both cohorts use the same refitted threshold in that draw. Seed 20260913 and separate detector-specific streams make the calculation reproducible. The point estimates remain those at the original fitted thresholds.

Table 12 reports all six detectors, with pointwise 95% and Bonferroni 99.17% intervals. Effort and DoU remain positive under the family-wise adjustment; all other intervals include zero. This analysis addresses calibration sampling conditional on the five chosen photograph pools. It does not account for alternative control sources, DoU’s stochastic forward pass, selection bias or attribution error.

## O.3 Further limitations of the collection

Attribution remains unverified. Captions and host fields can misname the generator; the inspected C2PA generatorversion field does not resolve that uncertainty. X’s rendered “Made with AI” badge was unavailable in the collection environment. Pixiv’s AI declaration, present for 66 images, identifies AI use rather than generator version.

Historical audits do not validate the current cohort. The three automated stages were assessed by another model without human adjudication. Correlated errors can remain. Screenshot and taxonomy audits contain no NightCafe examples; the caption audit cannot assess host attribution. Language quotas and unequal class sampling also limit pooled proportions. Section 4 specifies the populations and estimands. A human audit covering the frozen cohort and all attribution tiers remains future work.

Selection and source comparability. Caption admission omits version-silent posts; host-attributed galleries use a different selection rule. Queries, time filters, budgets and imageform retention differ by route, so source yields do not measure relative platform activity. Four platforms have default language labels and three have none. Source and attribution fields permit restricted analyses but do not remove these confounders.

Near-duplicate removal may discard distinct edits. Average-hash similarity does not establish semantic identity. Treating all 38 within-run, same-post X still-image drops as edit-pair members gives an unverified upper-bound scenario for that subset (Appendix G). Actual edit losses and gallery drops remain unreviewed. Caption self-report also does not reliably distinguish generation from editing.

Snapshot coverage. Neither X route, nor their union, is a census. The gateway’s prices, endpoints and retrieval behaviour can change. Reddit and Instagram were collected later, and some qualifying posts lack downloadable media. NightCafe was added later still: its host posting timestamps span only the final 16.7 hours (2026-09-10 05:25Z to 2026-09-10 22:06Z). It supplies 1,358 of 1,667 images posted in that span. These are publication times, not retrieval or verified generation times. The window ends at 2026-09-10 22:06:41Z, about 51.1 hours after the anchor, and covers only part of launch week.

Content measurement. Summaries exclude 12 videos and failed image analyses. April comparisons mix source populations, image-form filters and newly authored CLIP templates, despite a shared backbone and category names. OCR readers are selected from post-language metadata; missing or other languages receive English-only OCR. Post language can differ from image script, so unsupported scripts may be missed. The detected share is not a validated estimate of text prevalence.