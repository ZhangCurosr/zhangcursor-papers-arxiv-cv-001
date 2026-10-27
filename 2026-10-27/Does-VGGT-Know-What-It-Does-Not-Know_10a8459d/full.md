# Does VGGT Know What It Does Not Know?

Adityan Thirumani Srinivasan University of Wisconsin–Madison

David Nordstrom¨ Chalmers University of Technology

VGGT is confidently wrong ...

![](images/6a6425f324fbe62e5dc1aa52d6e9ca7451dd76e9714800d6b727e1fdcc9db7b8.jpg)  
VGGT's confidence: higher than on 74% of pairs  
true error: 141 px | probe on its features: failure (0.83)

## ... but its features know

![](images/e5dc6300d83caa2070fdee6b0c8fa5928bc7046d55b771eb3e89f6f848288729.jpg)  
Figure 1. VGGT is confidently wrong, but its features know. Left: an aerial and a ground view of the same gate. VGGT is more confident on this pair than on 74% of HardMatch pairs, yet the epipolar lines implied by its predicted cameras miss the annotated points by 141 px; a linear probe on its frozen features flags the failure. Right: ROC-AUC for predicting VGGT’s pose failures on held-out HardMatch categories, from VGGT’s own confidence and from the probe, on all pairs and on the half of the pairs VGGT is most confident about.

## Abstract

Feed-forward 3D reconstruction models such as VGGT regress cameras and geometry in a single forward pass, but they report no uncertainty for the camera poses they predict: an incorrect relative pose looks no different from a correct one. We ask whether the frozen model nonetheless knows when its pose is wrong. On HardMatch, a benchmark of 1,000 deliberately difficult image pairs on which VGGT fails on 35% of pairs, a linear probe on VGGT’s frozen features predicts pose failure on held-out categories with ROC-AUC 93.5%, compared with 79.7%for VGGT’s own point-map confidence and 77.7%for a classifier trained on all of VGGT’s outputs. The probe catches failures VGGT is confident about, lets a downstream system abstain on the pairs it distrusts, and transfers without retraining to rotation failures on PhotoTourism and ScanNet-1500, beyond what co-visibility explains. This knowledge is absent from the per-image encoder: it is computed partway through the layers that fuse the two views, and the same holds for a second model, $\pi ^ { 3 } .$ . In essence, we find that VGGT knows more about its own failures than it reports.

## 1. Introduction

Feed-forward reconstruction models [12, 14, 15, 23–26] predict cameras, depth and point maps from a set of images in a single forward pass, replacing much of the classical Structure-from-Motion (SfM) [7] pipeline. VGGT [23] is the most widely used of these models. It outputs a per-pixel confidence for depth and point maps, but no uncertainty for the camera poses it regresses. Downstream systems, such as SLAM, mapping, novel-view synthesis, therefore cannot distinguish a reliable relative pose from a badly wrong one.

Two observations make this gap worth studying. First, these models do fail: on HardMatch [18], a recent benchmark of hand-annotated, deliberately difficult image pairs, VGGT’s predicted geometry is grossly wrong (median epipolar error above 50 px) on 35% of pairs. Second, the large-language-model literature has shown that internal representations often encode whether a model’s own output is correct [2, 4, 11], even when the model’s stated confidence does not. This motivates the question in our title: does a feed-forward 3D model know when its pose is wrong, and is that knowledge linearly readablefrom itsfrozenfeatures?

We answer it with linear probes [1, 3] on frozen VGGT activations at every block, evaluated with category-grouped cross-validation so that every test category is unseen during training, and compared against everything VGGT itself outputs. The main contributions are as follows:

• We show that VGGT’s frozen features encode its own pose failures far more clearly than anything it outputs (ROC-AUC 93.5% vs. 79.7% for its own confidence on HardMatch; Fig. 1), including failures it is confident about.

• We locate where this knowledge forms: it is absent from the per-image encoder and emerges partway through the layers that fuse the views, in VGGT and in a second model, $\pi ^ { 3 }$

• We show that the signal is useful beyond the benchmark it was trained on: it supports selective prediction and transfers without retraining to rotation failures on PhotoTourism and ScanNet-1500.

## 2. Related Work

Feed-forward 3D reconstruction. DUSt3R [25] regresses aligned point maps for an image pair and MASt3R [14] adds dense matching. VGGT [23] predicts cameras, depth, point maps and tracks for many views in one pass through alternating frame-wise and global attention; VGGT-Ω [24], $\pi ^ { 3 }$ [26], Depth Anything 3 [15] and MapAnything [12] extend the recipe. These models expose per-pixel confidence for geometry, learned with an uncertainty-weighted loss, but no uncertainty for camera poses.

Uncertainty of feed-forward reconstruction. Hillemann et al. [9] evaluate how well VGGT’s per-pixel confidence tracks depth and point errors on DTU and note that VGGT provides no pose uncertainty. ReVar3R [16] estimates per-point uncertainty by perturbing the inputs of a frozen model. Both target per-point geometry; we target the correctness of the predicted relative pose, and we read it from internal features rather than outputs or input perturbations.

What 3D foundation models encode. Zhang et al. [27] find an emergent ability to reason about non-overlapping views; Stary´ et al. [21] trace how relative pose forms across the layers of a DUSt3R variant; Nordstrom¨ et al. [19] study what feed-forward 3D models know about image matching. Closest to us, Co-VGGT [28] probes VGGT’s layers for covisibility. Co-visibility is a property of the scene; failure is a property of the prediction. HardMatch pairs are co-visible by construction, and on PhotoTourism our probe predicts failure beyond ground-truth co-visibility (Sec. 4.3).

Probing, self-knowledge and failure prediction. Linear probes measure what is linearly available in frozen features [1, 3]. For language models, internal states encode whether the model’s own answer is correct [2, 4, 11]. Failure prediction and selective prediction [6, 8, 13] typically train dedicated heads; we keep the model frozen and compare against the confidence it already emits.

## 3. Method

Model and inputs. We use VGGT [23] (1B) frozen, with no fine-tuning. Each image pair is processed in one forward pass of two frames; we study pairs because HardMatch defines its ground truth, and its official evaluation, per pair, and test up to 16 views in Sec. 4.3. Images are passed as RGB in [0, 1] (VGGT normalises internally), resized so that the longer side is 518 px with the shorter side a multiple of 14, and centred on a square canvas. From the predicted pose encoding we decode extrinsics $[ \mathbf { R } _ { i } | \mathbf { t } _ { i } ]$ and intrinsics $\mathbf { K } _ { i }$ for both frames. As a second model we use $\pi ^ { 3 }$ [26] with the same inputs; it predicts camera-to-world poses but no intrinsics, so we fit a focal length by least squares to its loca point maps. We hook its encoder blocks 6, 12 and 23, all 36 decoder blocks and the camera head’s features.

## 3.1. Defining failure

Pairs without ground-truth poses (HardMatch). Hard-Match [18] provides, for each pair, a set of hand-annotated correspondences (median 11 per pair) but no camera poses. Following the official protocol, which evaluates any method through the fundamental matrix it implies, we form VGGT’s own relative pose ${ \bf R } = { \bf R } _ { 2 } { \bf R } _ { 1 } ^ { \top } , { \bf t } = { \bf t } _ { 2 } - { \bf R } { \bf t } _ { 1 }$ and fundamental matrix

$$
\mathbf { F } = \mathbf { H } _ { 2 } ^ { \top } \mathbf { K } _ { 2 } ^ { - \top } [ \mathbf { t } ] _ { \times } \mathbf { R } \mathbf { K } _ { 1 } ^ { - 1 } \mathbf { H } _ { 1 } ,\tag{1}
$$

where $\mathbf { H } _ { i }$ maps original-image pixels to the network’s input canvas. For every annotated correspondence we compute the symmetric epipolar distance [17], take the median over the pair, and call the pair a failure if it exceeds τ. We use $\tau = 5 0$ px (about 5% of the image width) as the main threshold and report 10, 20 and 100 px. As in the official code, we keep pairs with at least 8 correspondences and drop pairs labelled dynamic, leaving 903 pairs in 97 categories.

Pairs with ground-truth poses (PhotoTourism, ScanNet-1500). Here we compare VGGT’s relative pose with ground truth directly: the geodesic rotation error, the angle between translation directions (sign-agnostic, as in the ScanNet-1500 protocol [22]), and the pose error max $( e _ { R } , e _ { t } )$

## 3.2. Probing for failure

Features. We hook seven depths of the frozen network: blocks 6, 12 and 23 of the DINOv2-initialised [20] ViT encoder, which processes each image independently; the output of the aggregator’s last frame-attention block and last global (cross-frame) attention block; the normalised camera token that enters the camera head; and the input of the camera head’s trunk at its final refinement iteration. For the layer analysis we additionally pool the output of every one of the 24 encoder and 48 aggregator blocks. Tokens are mean-pooled per image, and a pair is represented by the mean of its two image vectors.

Probe and protocol. A probe is an $\ell _ { 2 } \cdot$ -regularised logistic regression on standardised features reduced to 128 dimensions by PCA, with the regularisation strength chosen by an inner 5-fold cross-validation and balanced class weights. All in-domain numbers use 10-fold crossvalidation grouped by HardMatch category, so every test pair comes from a category the probe never saw. We report ROC-AUC with 95% confidence intervals from a cluster bootstrap that resamples whole categories (1,000 resamples). For transfer, a probe is trained on all HardMatch pairs and applied unchanged to the other datasets.

Baselines from VGGT’s own outputs. We compare against signals a user of VGGT already has, each with a direction fixed in advance: the lower of the two images’ mean point-map confidence, the lower mean depth confidence, image sharpness (variance of the Laplacian), the predicted relative rotation angle, and the ratio of predicted focal lengths. We also train classifiers, with the same crossvalidation as the probes, on (i) all confidence statistics and (ii) everything VGGT outputs: confidence statistics, both extrinsics, focal lengths and relative rotation. Baseline (ii) asks whether the probe merely re-reads VGGT’s outputs.

## 4. Experiments

## 4.1. Setup

Table 1 summarises the three datasets. They play different roles. HardMatch is where VGGT fails often enough to study failure in depth, so all probes are trained and analysed there. PhotoTourism and ScanNet-1500 have groundtruth poses but few failures; we use them to ask whether the HardMatch probe transfers, without retraining, to other data and to a label defined directly on the rotation.

Protocol. All probes follow Sec. 3: frozen VGGT, meanpooled tokens, a linear classifier on 128 PCA components, and cross-validation grouped by HardMatch category, so that every reported number is measured on categories the probe never saw. Unless stated otherwise, the probe reads the last global-attention block (global last), a choice we fix in advance. Code will be released publicly.

Table 1. Datasets. Ground truth (hand-annotated correspondences, about 11 per pair, or poses), the failure definition used for each, and VGGT’s failure rate on image pairs.
<table><tr><td>Dataset</td><td>Pairs GT</td><td>Failure</td><td>Rate</td></tr><tr><td>HardMatch [18]</td><td>903 corresp.</td><td> $\mathrm { \ e p i . > 5 0 p x }$ </td><td>34.9%</td></tr><tr><td>PhotoTourism [10]</td><td>1,262 SfM</td><td> $e _ { R } > 1 5 ^ { \circ }$ </td><td>1.7%</td></tr><tr><td>ScanNet-1500 [22] 1,500</td><td>RGB-D</td><td> $e _ { \mathrm { p o s e } } > 1 5 ^ { \circ }$ </td><td>7.7%</td></tr></table>

## 4.2. HardMatch: does VGGT encode its own failures?

After the official filtering, HardMatch contains 903 valid pairs in 97 categories (830 in the official test split), including large viewpoint changes, aerial-to-ground pairs, illumination and season changes, drawings and repeating structure. Judged by the geometry its own cameras imply, VGGT fails on 34.9% of pairs at 50 px (63.7%, 51.5% and 25.1% at 10, 20 and 100 px, respectively), and reaches mAA@10 px of 19.9% on the official test split. Figure 2 shows typical pairs.

The features know more than the outputs. Table 2 compares what a user of VGGT can already observe with what is linearly readable inside it. The most informative output is the point-map confidence (79.7%), followed by depth confidence (66.5%); the predicted pose itself is nearly uninformative (relative rotation 62.9%, focal ratio 51.1%). A classifier trained on everything VGGT outputs, including both cameras and every confidence statistic, reaches only 77.7%, so the outputs contain little failure information beyond the point-map confidence. In contrast, a linear probe on any of the four late layers in Tab. 2 reaches 93–94%, and adding the confidence statistics to the probe leaves it unchanged (93.5%). Thresholded at 0.5, the probe labels 87.5% of pairs correctly, catching 87.6% of failures at 78.9% precision; VGGT’s confidence, thresholded to flag about as many pairs, reaches 74.0% accuracy, catching 68.3% of failures at 61.4% precision.

The gap has a plausible explanation. VGGT’s confidence is trained per pixel to down-weight hard-to-regress points; it describes how certain each depth or point is, not whether the two cameras are placed consistently. The fused features, by contrast, are the very representation from which the camera head regresses the relative pose, and they retain evidence that the two views were not registered consistently, which the camera head discards.

(a) confident, wrong probe: flagged

![](images/ab6725e23a087bbf7287c8746ebc575b67c0e733200e68138df084d2ef10222c.jpg)

conf. 74th pct · probe 0.83 median error 141 px

(b) confident, wrong probe: flagged

![](images/5dc6f6da02a058a93cb3580fb8c7c9fd859aff241595419de9ebe237fc255d1e.jpg)  
conf. 61th pct · probe 0.79 median error 401 px

(c) confident, wrong probe: flagged

![](images/2ef9e74976cdd55fee36aa6a66ff2c572e30dd5691aed8c929fa2d91a69251de.jpg)  
conf. 58th pct · probe 0.92 median error 1181 px

(d) unconfident, correct probe: not flagged

(e) confident, wrong probe: missed

![](images/9ca337afe6dfc476ea212a3576a0bb921a9ebcc3cc2a9747d5935587dc95d4f7.jpg)  
conf. 7th pct · probe 0.12 median error 9 px

![](images/9a9315880fdde561239d2351b70c3a585c10a0b2d78c5bf578269dc6f223872d.jpg)  
conf. 92th pct · probe 0.17 median error 115 px

Figure 2. The probe catches failures VGGT is confident about. Top: first image with hand-annotated points. Bottom: second image with the epipolar lines implied by VGGT’s own predicted cameras (same colour as the point); for a correct pose each point lies on its line. “conf.”: percentile of VGGT’s point-map confidence among all pairs (higher = more confident); “probe”: out-of-fold failure probability (a)–(c) VGGT is confident but wrong (aerial vs. ground view, large viewpoint change, photograph vs. drawing) and the probe flags it. (d) VGGT is unconfident but correct, and the probe does not flag it. (e) A miss: extreme zoom on a confidently wrong pair.

Robustness. Table 3 rules out the obvious artefacts. Shuffled labels give chance (51.6%), so the grouped crossvalidation does not leak; image size and aspect ratio alone give chance (49.8%), so the padding of the input canvas does not leak the label. Pairs on which VGGT predicts an almost-zero baseline make the fundamental matrix illconditioned and could produce spurious failures, but the predicted baseline does not predict failure (49.9%) and removing those pairs leaves the probe at 93.4%. The advantage over confidence holds at every threshold, on the official test split, and for every pair representation and PCA size we tried. Combining layers does not help, and combining the three backbone blocks still gives only 64.1%.

Where the signal emerges. To locate the signal we probe every block of the network (Fig. 3a). The per-image encoder never exceeds 65.2%: what makes a pair hard is not visible in either image alone. More surprisingly, the first half of the aggregator adds nothing either, although it already alternates frame and global attention: through block 10 the probe stays between 63% and 66%. It then rises abruptly, from 69% after block 11 to 81% after block 12, 88% after block 13 and 93% after block 15, and stays at 93–94% to the last block (peak 94.0%, block 18). The failure signal is therefore not a by-product of the two views meeting; it is computed in a narrow band of middle aggregator blocks. A training-free reading appears later still. The cosine similarity between the two views’ pooled features stays at the encoder’s level (61–69%) through block 15 and then climbs to 83.7% at the last block, above VGGT’s own confidence: in its final layers, VGGT’s representations of the two views stop agreeing when its pose is wrong. At the camera token this similarity falls back to 71.0% while a trained probe stays at 93.2%; the information is still present, but no longer as a simple disagreement.

A second model: $\pi ^ { 3 } .$ $\pi ^ { 3 }$ [26] also alternates frame-wise and global attention, but has no reference view and predicts camera-to-world poses and local point maps, from which we recover the focal length by least squares. We run it on the same pairs with the same failure definition. It fails on 34.6% of them (mAA@10 18.4% on the test split) and agrees with VGGT’s failure label on 81%. Table 4 repeats the main comparison. $\pi ^ { 3 } \mathrm { { ^ , } }$ own confidence is a weaker failure signal than VGGT’s (62.2%), and on its most confident half, where 27% of pairs still fail, it is at chance (52.8%). A probe on its last decoder block nonetheless reaches 91.7%, against 76.9% for a classifier trained on all of its outputs, and 92.7% on that most confident half. The layer profile repeats as well (Fig. 3b): flat through the encoder and the first 21 decoder blocks (≤ 67%), then a steep rise to 92% by block 33. One difference is instructive: in $\pi ^ { 3 }$ the trainingfree cosine never rises (≤ 69%). Failure is linearly decodable in both models, but the simple reading that the two views stop agreeing is specific to VGGT.

![](images/784d78632b170bd080ff105f18a1a865c956c9f152a2d28cab6de79866ec25f5.jpg)  
Figure 3. Failure awareness emerges abruptly in the middle of the network, in both models. ROC-AUC for predicting VGGT’s pose failures on HardMatch from every block of (a) VGGT and $( { \mathsf { b } } ) \ \pi ^ { 3 }$ , with a linear probe (blue) and with one training-free number, the cosine similarity between the two views’ pooled features (orange). Dashed: the model’s own confidence. Shaded: the per-image encoder. The probe is flat through the encoder and the first half of the cross-view layers, then rises within a few blocks; only in VGGT does the training-free similarity follow.

Table 2. VGGT’s features know more about its failures than anything it outputs. Predicting VGGT’s pose failures on Hard-Match (failure: median epipolar error > 50 px). ROC-AUC (%) with 95% category-bootstrap CI, and average precision (AP, %). All learned rows use the same category-grouped cross-validation.
<table><tr><td>Predictor</td><td>ROC-AUC (%)</td><td>AP (%)</td></tr><tr><td>VGGT&#x27;s own outputs Point-map conf. (min of pair) Depth conf. (min of pair) 57.7 [53.2, 62.1]</td><td>79.7 [76.6, 82.4] 66.5 [62.8, 69.9]</td><td>60.4 47.5 44.0</td></tr><tr><td>Predicted rel. rotation Predicted focal ratio Classifiers trained on VGGT&#x27;s outputs</td><td>62.9 [58.5, 67.1] 51.1 [46.6, 55.4]</td><td>43.1 37.1</td></tr><tr><td>All confidence statistics All VGGT outputs</td><td>75.2 [71.9, 78.7] 77.7 [73.8, 80.8]</td><td>58.4 59.5</td></tr><tr><td>Linear probes on frozen features</td><td></td><td></td></tr><tr><td>ViT block 6</td><td>59.3 [54.6, 64.0]</td><td></td></tr><tr><td>ViT block 12</td><td>64.6 [59.9, 69.4]</td><td>43.8</td></tr><tr><td>ViT block 23 Aggregator, last frame-attn</td><td>65.0 [60.3, 69.6] 93.7 [92.0, 95.4]</td><td>48.7 48.3</td></tr></table>

Confidently wrong predictions. Confidence and failure are correlated at the level of the whole benchmark, but how much does confidence tell us once we already know it? We split the pairs into quartiles of point-map confidence. The failure rate falls from 64% in the least confident quartile to 6% in the most confident, yet within each quartile confidence ranks failures close to chance (Fig. 4) (AUC 47%, 61%, 64%, 57%) while the probe stays high (90%, 91%, 92%, 82%). On the most confident half of the pairs, where 12.2% still fail, the probe reaches 90.0% [85.4, 94.2] against 69.9% [61.2, 77.8]. These are the failures a downstream system cannot see. Figure 2 shows examples: an aerial-to-ground pair, a large viewpoint change and a photograph matched to a drawing, all of which VGGT places confidently but wrongly and the probe flags, together with a correct pair on which confidence raises a false alarm and an extreme-zoom pair the probe misses.

Table 3. The result is robust to the failure threshold and to every control we tried (HardMatch, ROC-AUC, %). Top: failure threshold. Bottom: controls and probe variants (global last unless noted).
<table><tr><td>Threshold</td><td>Fail rate</td><td>Probe</td><td>Point conf.</td><td>All outputs</td></tr><tr><td>10 px</td><td>64%</td><td>91.1</td><td>72.8</td><td>73.8</td></tr><tr><td>20 px</td><td>51%</td><td>92.9</td><td>76.0</td><td>76.6</td></tr><tr><td>50 px</td><td>35%</td><td>93.5</td><td>79.7</td><td>77.7</td></tr><tr><td>100 px</td><td>25%</td><td>92.4</td><td>77.8</td><td>76.9</td></tr><tr><td colspan="3">Controls and probe variants</td><td></td><td>Probe</td></tr><tr><td colspan="3">Shuffled labels (control)</td><td></td><td>51.6</td></tr><tr><td colspan="3">Image size and aspect only (control)</td><td></td><td>49.8</td></tr><tr><td colspan="3">5 folds / official test split only</td><td></td><td>93.4 / 93.7</td></tr><tr><td colspan="3">Without 45 near-zero-baseline pairs</td><td></td><td>93.4</td></tr><tr><td colspan="3">All seven layers / frame-attn + global-attn</td><td></td><td>93.2 / 93.4</td></tr><tr><td colspan="3">Concatenated views / mean + |A — B|</td><td></td><td>93.4 / 93.7</td></tr><tr><td colspan="3">PCA 32 / 64 / 256 dims</td><td></td><td>93.4 / 93.7 / 93.5</td></tr><tr><td colspan="3">ViT blocks 6+12+23 combined</td><td></td><td>64.1</td></tr></table>

Table 4. A second model, $\pi ^ { 3 } ,$ shows the same gap between its features and its confidence. HardMatch, same pairs, failure definition and protocol. ROC-AUC (%) unless stated; “most confident half”: the half of the pairs on which the model’s own confidence is highest.
<table><tr><td></td><td>VGGT</td><td> $\pi ^ { 3 }$ </td></tr><tr><td>Failure rate (&gt; 50 px)</td><td>34.9%</td><td>34.6%</td></tr><tr><td>mAA@10 px (test split)</td><td>19.9</td><td>18.4</td></tr><tr><td>Own confidence (min of pair)</td><td>79.7</td><td>62.2</td></tr><tr><td>All outputs (learned)</td><td>77.7</td><td>76.9</td></tr><tr><td>Probe, best of encoder blocks 6/12/23</td><td>65.0</td><td>66.3</td></tr><tr><td>Probe, last attention block Probe, camera head</td><td>93.5</td><td>91.7</td></tr><tr><td></td><td>93.2</td><td>88.7</td></tr><tr><td>Cross-view cosine, best block</td><td>83.7</td><td>68.7</td></tr><tr><td>Most confident half: failure rate</td><td>12.2%</td><td>27.3%</td></tr><tr><td>own confidence</td><td>69.9</td><td>52.8</td></tr><tr><td>probe</td><td>90.0</td><td>92.7</td></tr></table>

![](images/018c99c154565ed4df11cc15bf763b041835b1d2e59d321455afd4159b868271.jpg)  
Figure 4. Within each level of confidence, VGGT’s confidence is near chance; the probe is not. HardMatch pairs split into quartiles of VGGT’s point-map confidence. Bars: VGGT’s failure rate. Lines: ROC-AUC for predicting failure within each quartile.

Per category. Failure rates vary widely across Hard-Match’s categories, from 27% on building and texture pairs to 74% on celestial ones (Fig. 5). The probe exceeds VGGT’s confidence in every category with enough pairs except doppelgangers, where both reach¨ 92%; the largest gains are on the categories where appearance changes most, namely celestial (82% vs. 60%), drawing (95% vs. 77%) and illumination (92% vs. 71%).

Using the signal: selective prediction. A system that can abstain on some pairs, for example by requesting another view or falling back to a slower pipeline, needs a ranking rather than a calibrated probability. Keeping the 50% of pairs the probe trusts most yields a 3.8% failure rate and mAA@10 of 34.9%; ranking by VGGT’s confidence yields 12.4% and 29.7%, and a perfect oracle 0% and 38.3% (Fig. 6). Over all coverages, the area under the risk– coverage curve is 0.096 for the probe, 0.154 for confidence, 0.169 for all outputs and 0.070 for the oracle, so the probe closes about 70% of the gap between VGGT’s confidence and a perfect ranking.

![](images/848635cc5e72b5d0a6c364ac9501bfbb9a0a0d6c52b12352b7cbf79da3e37ad1.jpg)  
Figure 5. The probe beats VGGT’s confidence in every Hard-Match category but one. Bars: VGGT’s failure rate. Markers: AUC of the probe (filled) and of point-map confidence (open).

![](images/c52cec78942755dd3280bd5728e0ebc594237b45d994dd73f8851696324e9cb6.jpg)

![](images/a7b0817ca7cccf138d4cd4497c0008b6bd2cabf10a120095b35898bbaed23bc6.jpg)  
Figure 6. Abstaining on the pairs the probe distrusts removes most of VGGT’s failures. Keeping the pairs ranked most reliable by each score: failure rate (left) and mAA@10 px (right) of the kept pairs. Oracle: ranked by true error.

A negative result: routing to a matcher. We also tried to route each pair either to VGGT or to the dedicated matcher LoMa [18]. On HardMatch, LoMa (mAA@10 50.8% on the test split) almost never does worse than VGGT (19.9%): even an oracle that always picks the better of the two reaches only 52.1%, and a learned router that uses the probe reaches 51.1%. With so little complementarity there is nothing for the probe to exploit, and we make no routing claim.

## 4.3. PhotoTourism: transfer to rotation failures

HardMatch defines failure through epipolar error, which mixes rotation, translation and intrinsics errors, and its probe is trained on HardMatch’s own categories. Photo-Tourism [10] lets us test both limitations at once: its poses come from large-scale structure-from-motion, so failure can be defined on the rotation alone, and none of its images were used to train the probe. We sample 1,262 pairs from 16 landmarks, stratified into four bins of ground-truth covisibility, and apply the HardMatch probe unchanged.

Table 5. The HardMatch probe transfers to rotation failures on PhotoTourism, beyond co-visibility. Top: VGGT’s rotation failures by co-visibility. Bottom: ROC-AUC (%) for rotation failures of the HardMatch probe (no retraining), VGGT’s confidence, ground-truth co-visibility (not available at test time) and a probe trained on PhotoTourism itself (scene-grouped CV). “low covis.”: pairs with co-visibility < 0.02 only.
<table><tr><td>Co-visibility</td><td> $< 0 . 0 2$ </td><td>0.02-0.1</td><td>0.1-0.3</td><td>&gt; 0.3</td></tr><tr><td>Pairs</td><td>302</td><td>320</td><td>320</td><td>320</td></tr><tr><td> $e _ { R } > 1 5 ^ { \circ }$ </td><td>6.6%</td><td>0.3%</td><td>0.0%</td><td>0.3%</td></tr><tr><td></td><td colspan="2">all pairs</td><td colspan="2">low covis.</td></tr><tr><td></td><td> $e _ { R } > 1 0 ^ { \circ }$ </td><td> $e _ { R } > 1 5 ^ { \circ }$ </td><td> $e _ { R } > 1 0 ^ { \circ }$ </td><td> $e _ { R } > 1 5 ^ { \circ }$ </td></tr><tr><td>#failures</td><td>25</td><td>22</td><td>21</td><td>20</td></tr><tr><td>HardMatch probe</td><td>94.6</td><td>97.5</td><td>86.9</td><td>89.5</td></tr><tr><td>VGGT confidence</td><td>88.5</td><td>90.6</td><td>79.5</td><td>79.2</td></tr><tr><td>GT co-visibility</td><td>87.0</td><td>89.8</td><td>74.3</td><td>75.7</td></tr><tr><td>In-domain probe</td><td>95.1</td><td>89.1</td><td>一</td><td>一</td></tr></table>

VGGT rarely fails on landmark photographs. Rotation errors above 15<sup>◦</sup> occur on only 1.7% of the pairs, and almost all of them on pairs that barely overlap: 6.6% of pairs with co-visibility below 0.02 fail, against at most 0.3% in every other bin (Tab. 5, top). At the looser threshold of 5<sup>◦</sup>, 6.8% of pairs fail. These pairs are much easier than Hard-Match, which is why we use PhotoTourism as a transfer test rather than as a training set.

The HardMatch probe transfers. Without seeing a single PhotoTourism image, the probe predicts rotation failures with AUC 94.6% at 10<sup>◦</sup> and 97.5% at 15<sup>◦</sup> (87.1% at 5<sup>◦</sup>), above VGGT’s confidence at every threshold and on par with a probe trained on PhotoTourism itself (Tab. 5, bottom). The score is also related to the size of the error, not just to a threshold: over all 1,262 pairs its Spearman correlation with the log rotation error is 0.43, against 0.33 for confidence. The intervals are wide because there are few failures (25 at 10<sup>◦</sup>), but the ordering is the same at every threshold.

It is not just co-visibility. Because the failures concentrate on barely overlapping pairs, a probe that merely detected low overlap, as Co-VGGT’s [28] does by design, would also score well. Three observations argue against this explanation. First, ground-truth co-visibility itself, which a deployed system would not have, predicts the failures less well than the probe (87.0% and 89.8%). Second, among only the barely overlapping pairs, where co-visibility is nearly constant, the probe still separates failures from successes (86.9% and 89.5%) better than confidence (79%) and co-visibility (74–76%). Third, with co-visibility partialled out, the probe’s rank correlation with the rotation error remains 0.23, twice that of confidence (0.11). The probe detects whether VGGT got the rotation wrong, not merely whether the views overlap.

Table 6. More views remove most failures; the few left are hard to rank for both the probe and confidence (PhotoTourism, 96 sets of 16 images). Top: failure rate of the same 2,304 pairs when VGGT sees 2, 4, 8 or 16 views per forward pass. Bottom: all pairs of each setting; ROC-AUC (%) for rotation errors above $5 ^ { \circ }$ of the HardMatch probe (trained on two-view pairs, no retraining) and of VGGT’s confidence. With 28–83 failures per setting, the 95% confidence intervals span more than 40 points.
<table><tr><td>Views per forward pass</td><td>2</td><td>4</td><td>8</td><td>16</td></tr><tr><td>Same pairs:  $e _ { R } > 5 ^ { \circ }$ </td><td>3.6%</td><td>1.4%</td><td>0.5%</td><td>0.3%</td></tr><tr><td>Same pairs:  $e _ { R } > 1 5 ^ { \circ }$ </td><td>0.9%</td><td>0.3%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>All pairs</td><td>2,304</td><td>2,304</td><td>5,376</td><td>11,520</td></tr><tr><td>failures  $( e _ { R } > 5 ^ { \circ } )$ </td><td>83</td><td>33</td><td>28</td><td>35</td></tr><tr><td>HardMatch probe (2-view)</td><td>81.5</td><td>82.6</td><td>74.2</td><td>86.3</td></tr><tr><td>VGGT confidence</td><td>91.2</td><td>90.4</td><td>92.9</td><td>82.1</td></tr></table>

Multi-view graphs: no downstream gain. We also built ten 16-image graphs from PhotoTourism scenes and ran VGGT on all 120 pairs of each. On these 1,200 edges the HardMatch probe again separates rotation failures $( e _ { R } ~ >$ $1 5 ^ { \circ }$ , 1.75% of edges) better than confidence (98.6% vs. 92.7%). We then asked whether discarding distrusted edges improves rotation averaging. Robust averaging over all edges already reduces the mean absolute-rotation error from 1.27<sup>◦</sup> (least squares) to 0.70<sup>◦</sup>; dropping the 10% of edges the probe distrusts most gives 0.66<sup>◦</sup>, dropping by confidence 0.67<sup>◦</sup>, and even an oracle that removes every edge whose true error exceeds 15<sup>◦</sup> gives 0.67<sup>◦</sup>. With fewer than 2% of edges failing, robust averaging already absorbs them; showing a downstream benefit needs graphs with far denser failures than these.

More than two views. HardMatch defines its ground truth per pair, so our main study uses pairs. To test whether this matters, we sample 96 sets of 16 images from the PhotoTourism scenes and run VGGT on each set as one 16-view pass, as two 8-view passes, as four 4-view passes, and on every pair inside the 4-view groups, so that the same 2,304 pairs are predicted with 2, 4, 8 and 16 views (Tab. 6). Extra views fix most failures: the share of these pairs with a rotation error above $5 ^ { \circ }$ falls from 3.6% with two views to 1.4%,

0.5% and 0.3% with 4, 8 and 16, and with 8 or more views none exceeds 10<sup>◦</sup>. The probe trained on two-view Hard-Match pairs still ranks the remaining failures well above chance without retraining (74–86% for 4–16 views), and VGGT’s own confidence does comparably (82–93%); with 28–35 failures per setting, neither is clearly better. Even with two views, confidence is ahead for small errors (above 5<sup>◦</sup>: 91.2% vs. 81.5%) and the probe for larger ones (above 10<sup>◦</sup>: 95.7% vs. 93.6%). Extra views thus mainly remove the failures the probe is meant to catch; failure awareness for many-view inputs needs a benchmark on which such inputs still fail.

## 4.4. ScanNet-1500: failure modes differ

ScanNet-1500 [5, 22] is the standard indoor benchmark for relative pose: 1,500 pairs from 100 rooms, with poses from RGB-D reconstruction. Indoor scenes add low texture, repeated structure and, unlike PhotoTourism, many pairs taken from nearby positions.

VGGT’s indoor failures are translation failures. VGGT’s rotations on ScanNet are accurate (median error 1.2<sup>◦</sup>; only 2.1% of pairs exceed 5<sup>◦</sup>). Its pose errors come almost entirely from the direction of the translation: 113 of the 115 pairs with pose error above $1 5 ^ { \circ }$ have a rotation error below $5 ^ { \circ }$ . These failures are concentrated on short baselines (Tab. 7, top): the translation direction is wrong by more than 15<sup>◦</sup> on 47% of pairs whose cameras are less than 10 cm apart, but on fewer than 1% of pairs more than 40 cm apart. This is a known property of two-view geometry rather than a peculiarity of VGGT: when the cameras barely move, the direction of their small displacement is weakly constrained by the images.

Table 7. On ScanNet, the probe transfers to rotation failures but not to short-baseline translation failures. Top: VGGT’s errors by true camera baseline; translation-direction errors dominate at short baselines. Bottom: ROC-AUC (%) of the Hard-Match probe (no retraining), VGGT’s confidence and a probe trained on ScanNet itself (scene-grouped CV), by failure type. $e _ { \mathrm { p o s e } } = \operatorname* { m a x } ( e _ { R } , e _ { t } )$
<table><tr><td>True baseline</td><td>&lt; 10 cm</td><td>10–20 cm</td><td>20-40 cm</td><td>&gt; 40 cm</td></tr><tr><td>Pairs</td><td>83</td><td>173</td><td>292</td><td>952</td></tr><tr><td> $e _ { t } > 1 5 ^ { \circ }$ </td><td>47.0%</td><td>27.2%</td><td>6.8%</td><td>0.9%</td></tr><tr><td> $e _ { R } > 5 ^ { \circ }$ </td><td>0.0%</td><td>0.0%</td><td>0.3%</td><td>3.3%</td></tr><tr><td>Failure</td><td>#fail</td><td>HM probe</td><td>Confidence</td><td>In-domain</td></tr><tr><td> $e _ { R } > 5 ^ { \circ }$ </td><td>32</td><td>79.5</td><td>58.7</td><td>82.3</td></tr><tr><td> $e _ { \mathrm { p o s e } } > 1 0 ^ { \circ }$ </td><td>202</td><td>47.6</td><td>61.0</td><td>90.5</td></tr><tr><td> $e _ { \mathrm { p o s e } } > 1 5 ^ { \circ }$ </td><td>115</td><td>44.2</td><td>63.6</td><td>91.0</td></tr></table>

Transfer depends on the failure mode. The few rotation failures behave differently: they occur at the longest baselines (3.3% of pairs above 40 cm, none below 20 cm), the regime closest to HardMatch’s wide-baseline pairs. For these, the HardMatch probe transfers: 79.5% against 58.7% for confidence, close to a probe trained on ScanNet itself (82.3%). For the translation-dominated failures it does not (44–48%), and VGGT’s confidence is only marginally better (61–64%). Yet a probe trained on ScanNet detects these failures with AUC 90.5–91.0%. VGGT’s fused features therefore encode the short-baseline failures as well, but along a direction that a probe trained on HardMatch’s gross failures does not learn. Failure awareness is not a single scalar inside the model: one linear readout carries across datasets for wrong orientation, and ill-conditioned translation calls for its own.

## 5. Limitations

Two models. We study VGGT and $\pi ^ { 3 } ;$ larger or newer models (VGGT-Ω, MapAnything) may behave differently, and the training-free cosine reading already differs between the two. Labels. HardMatch has no ground-truth poses, so failure is defined through the epipolar error of about 11 annotated correspondences per pair, which mixes rotation, translation and intrinsics errors; the ground-truth datasets confirm the result for rotation but have few failures (PhotoTourism: 22 above 15<sup>◦</sup>). Two views. We study image pairs, because HardMatch defines its ground truth per pair. With more views, failures on PhotoTourism become rare (0.3% with 16 views) and the two-view probe is no longer clearly better than VGGT’s confidence on them; filtering edges by the probe also did not improve rotation averaging (Sec. 4.3). Supervision. The probe is trained with failure labels, whereas VGGT’s confidence is free. We therefore compare against classifiers trained on VGGT’s outputs under the same protocol, and test the probe on data it was not trained on. Failure modes. A single probe does not cover short-baseline translation errors (Sec. 4.4).

## 6. Conclusion

VGGT knows more about its own failures than it reports. A linear readout of its frozen features predicts when its relative pose is wrong far better than its own confidence or anything else it outputs, including when it is confident; it supports selective prediction and transfers to rotation failures on other datasets. The signal emerges abruptly in the middle of the network, in both VGGT and $\pi ^ { 3 }$ . Costing one linear layer on features the model already computes, it is a practical reliability signal for systems built on feed-forward reconstruction, and evidence that self-knowledge, studied extensively in language models, is also present in 3D foundation models.

## References

[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In ICLR, 2017. Workshop Track. 2

[2] Amos Azaria and Tom Mitchell. The internal state of an LLM knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP, 2023. 1, 2

[3] Yonatan Belinkov. Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics, 48(1), 2022. 2

[4] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In ICLR, 2023. 1, 2

[5] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 8

[6] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In NeurIPS, 2017. 2

[7] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003. 1

[8] Dan Hendrycks and Kevin Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In ICLR, 2017. 2

[9] Markus Hillemann, Robert Langendorfer, Steven Landgraf,¨ and Markus Ulrich. Uncertainty quality of VGGT: An analysis on the DTU benchmark dataset. arXiv:2606.16479, 2026. 2

[10] Addison Howard, Eduard Trulls, Kwang Moo Yi, Dmitry Mishkin, Sohier Dane, and Yuhe Jin. Image matching challenge 2022, 2022. 3, 6

[11] Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022. 1, 2

[12] Nikhil Keetha, Norman Muller, Johannes Sch¨ onberger,¨ Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan, Sebastian Scherer,\` and Peter Kontschieder. MapAnything: Universal feedforward metric 3D reconstruction, 2025. arXiv preprint arXiv:2509.13414. 1, 2

[13] Alex Kendall and Yarin Gal. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? In NeurIPS, 2017. 2

[14] Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Ground-ˆ ing image matching in 3D with MASt3R. In ECCV, 2024. 1, 2

[15] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 1, 2

[16] Sammam Mahdi, Fariha Binta Salim, Rakin Bin Rabbani, and Aniqua Nusrat Zereen. ReVar3R: Gauge-aware perturbation uncertainty for feed-forward 3D reconstruction. arXiv:2610.07883, 2026. 2

[17] Dmytro Mishkin, Jiri Matas, Michal Perdoch, and Karel Lenc. WxBS: Wide baseline stereo generalizations. In BMVC, 2015. 2

[18] David Nordstrom, Johan Edstedt, Georg B¨ okman, Jonathan¨ Astermark, Anders Heyden, Viktor Larsson, Marten˚ Wadenback, Michael Felsberg, and Fredrik Kahl. LoMa:¨ Local feature matching revisited. In Proceedings of the Eu ropean Conference on Computer Vision (ECCV), 2026. 1, 2, 3, 6

[19] David Nordstrom, Xinyue Zhang, Thibaut Loiseau, Vincent¨ Lepetit, and Fredrik Kahl. RoMa-Ω: What feed-forward 3D models know about image matching. arXiv:2609.09507, 2026. 2

[20] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V.´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. In TMLR, 2024. 3

[21] Michal Stary, Julien Gaubil, Ayush Tewari, and Vincent Sitz-´ mann. Understanding Multi-View Transformers. In ICCV 2025 E2E3D Workshop, 2025. 2

[22] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free local feature matching with transformers. In CVPR, 2021. 2, 3, 8

[23] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025. 1, 2

[24] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut, Piotr Bo-¨ janowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2

[25] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D vision made easy. In CVPR, 2024. 2

[26] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chun hua Shen, and Tong He. π<sup>3</sup>: Permutation-Equivariant Visual Geometry Learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2026. To appear. 1, 2, 4

[27] Yiwen Zhang, Joseph Tung, Ruojin Cai, David Fouhey, and Hadar Averbuch-Elor. Emergent extreme-view geometry in 3D foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 2

[28] Filippo Ziliotto, Luciano Serafini, Lamberto Ballan, and Tommaso Campari. What VGGT knows about over lap: Probing geometric foundation models for co-visibility. arXiv:2607.09503, 2026. 2, 7