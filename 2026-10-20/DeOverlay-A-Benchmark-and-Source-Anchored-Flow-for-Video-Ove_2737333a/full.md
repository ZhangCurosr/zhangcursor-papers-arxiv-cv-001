# DeOverlay: A Benchmark and Source-Anchored Flow for Video Overlay Removal

Deshui Miao<sup>1</sup> Yuhang Ma<sup>1</sup> Shaoyong Jia<sup>1</sup> Wenzhao Gao<sup>1</sup> Yue Zhang<sup>1</sup> Shaohui Jiao<sup>1</sup> Xin Li<sup>2</sup>

<sup>1</sup>ByteDance, <sup>2</sup>Southern University of Science and Technology

## Abstract

Video overlay removal aims to recover scenes obscured by subtitles, titles, and logos while preserving surrounding content. Existing methods use temporal propagation or source-conditioned generation, but can leave residual overlays and distort scene structure when useful source information is obscured. We propose DeOverlay, a source-anchored flow model that reconstructs the clean video from the source latent with fixed source guidance. Specifically, we train a shared velocity network along paired source-to-clean paths using region-weighted flow matching, enabling successive corrections from intermediate states. To select local source cues, we develop visibility-normalized temporal aggregation that excludes masked positions and averages the remaining observations Global source context and edit-region embeddings complement these cues through additive conditioning, retaining a single video-token sequence. To evaluate scene recovery across diverse overlay conditions, we introduce DVTE-Bench, comprising 1,631 paired videos across eight text-overlay categories, with OCR-screened clean targets, frame-aligned pixel masks, and a common evaluation protocol. Experiments on the complete release show that DeOverlay outperforms four baselines on all five reconstruction and perceptual metrics, reducing Mask LPIPS and Mask VFID by 58.3% and 70.3%, respectively, relative to the strongest evaluated baseline. Alongside these quantitative results, qualitative examples demonstrate the removal of graphic logos and stickers. To support further evaluation, we will make the complete benchmark, evaluation code, and test outputs available.

Project Page: https://yahooo-m.github.io/DVTE-Bench/

## 1 Introduction

Video overlay removal aims to recover scene content obscured by subtitles, titles, watermarks, and logos while leaving the surrounding video unchanged. Although neighboring frames may reveal some occluded content, persistent coverage leaves other regions unobserved throughout the clip. Efective removal therefore requires both exploiting available observations and inferring missing content, while maintaining temporal coherence and preserving the scene structure already present in the input.

Propagation-based inpainting transfers image content and features across frames, but relies on learned priors when covered regions lack direct observations [19]. Generative methods infer covered content through subtitle-specific reconstruction [5] or noise-initialized editing, in which source information enters through channel concatenation [20] or sequence concatenation [8] while the editable state begins from noise. Our paired comparisons show residual lettering and boundary distortions in some evaluated outputs (Figure 4).

![](images/59cdcef29602d9459ad70b63b28acc86ec3727f90410fa023fe44fee0b99fa93.jpg)  
Figure 1 Starting state and source conditioning. Channel- and sequence-conditioned editors start from noise, whereas ours starts from z and receives fixed source features at each update. On a shared patch grid, the three designs use N, 2N, and N video tokens per call, with auxiliary tokens and codecs omitted from the diagram.

We propose DeOverlay, which starts reconstruction from the source latent and keeps the original source available to each update. A shared velocity network learns source-to-clean corrections from paired latent paths under a region-weighted flow objective, then applies successive corrections at inference as the state and time evolve under fixed weights and source guidance.

Visibility-normalized temporal aggregation averages unmasked source features across nearby frames, while a global source summary supplies scene context and an edit-region embedding marks the covered region. By adding these terms to the editing tokens at every update, we keep the source available without a second video-token sequence. Where no frame reveals the covered content, the video prior infers it.

Evaluating overlay removal requires checking whether the recovered content agrees with the underlying scene. Existing studies assess subtitle erasure [5] or visible watermark removal [1] in separate task-specific settings. To support common regional evaluation across overlay conditions, we introduce DVTE-Bench, the Diverse Video Text Erasure Benchmark, with 1,631 paired videos covering eight text-overlay categories that vary in layout, persistence, and appearance. OCR-screened clean targets and aligned pixel masks provide the references for this evaluation. Experiments on the complete release show improvements over four baselines on all five reconstruction and perceptual metrics, while component and step-count studies examine the contribution of source guidance and repeated updates.

## Our contributions are summarized as:

• We introduce DVTE-Bench, a paired benchmark covering eight text-overlay types with clean targets, aligned pixel masks, and common evaluation of regional reconstruction, perceptual quality, and temporal behavior.

• We propose DeOverlay, which learns a source-to-clean flow with fixed local and global guidance. Source initialization retains observed structure, while visibility-normalized aggregation selects local cues for reconstructing covered regions

• We evaluate DeOverlay on all 1,631 released clips and obtain the best results among five methods on all five reconstruction and perceptual metrics. Ablations examine source guidance and flow step count, with separate measurements of inference cost.

Table 1 Comparison of video text erasure benchmarks. CLEAR [5] and VSR-Bench-400 [8] are compared with DVTE-Bench. The DVTE-Bench release checkmark denotes its planned public release.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2"></td><td rowspan="2">Clips Release</td><td colspan="2">Ground Truth</td><td colspan="5">Overlay Diversity</td><td colspan="2">Evaluation</td></tr><tr><td>Clean</td><td>Mask</td><td>Types</td><td>Multilingual</td><td>Spatial</td><td>Temporal</td><td>Appearance</td><td>Regional</td><td>Temporal</td></tr><tr><td>CLEAR test set</td><td>400</td><td>X</td><td>V</td><td>X</td><td>1</td><td>X</td><td>√</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>VSR-Bench-400</td><td>400</td><td>X</td><td>√</td><td>√</td><td>1</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>DVTE-Bench (ours)</td><td>1,631</td><td>√</td><td>√</td><td>√</td><td>8</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## 2 Related Work

Video inpainting and overlay removal. ProPainter combines image and feature propagation with mask-guided sparse attention [19]. For subtitle removal, CLEAR learns subtitle representations followed by generative reconstruction and supports mask-free inference [5], while DECO combines frame-wise priors with temporal modeling for visible watermark removal [1]. Our method learns source-to-clean corrections with persistent source guidance.

Source-conditioned restoration. SEDiT concatenates fixed source-reference tokens with noise-initialized tokens for one-step subtitle erasure [8]. For space-time super-resolution, DifST combines input-derived initialization, motion-based frame aggregation, and video-prompt guidance [2]. We adapt its local/global decomposition through visibility-normalized aggregation at fixed coordinates and a pooled latent summary, which guide a source-initialized state through successive updates.

Flow-based editing. Flow matching and rectified flow learn vector fields along prescribed paths [12, 13]. In restoration, IR-Flow learns degraded-to-clean image transport [4], while PCFlow compares observationinitialized unconditional flow with noise-initialized conditional flow [9]. For editing, FlowEdit performs inversion-free flow editing [10], and RFDM predicts residuals relative to preceding outputs for causal video editing [16]. Our paired video flow combines source initialization, fixed source guidance, and regional weighting within a pretrained video transformer [14, 15].

Evaluation resources. CLEAR evaluates a 400-clip subtitle benchmark [5], while SEDiT introduces VSR-Bench-400 with clean videos and subtitle masks [8]. In the watermark setting, DECO evaluates stationary and moving watermarks synthesized on YouTube-VOS and DAVIS [1].

## 3 DVTE-Bench: Diverse Video Text Erasure

We construct DVTE-Bench to evaluate overlay removal and scene reconstruction by pairing each overlaid video with a clean target and frame-aligned masks. These pairs support common evaluation across overlay layouts, temporal behaviors, and appearances.

Table 1 contrasts DVTE-Bench with existing subtitle benchmarks in paired supervision, overlay diversity, and evaluation coverage.

## 3.1 Challenge Design

The benchmark contains eight overlay types organized around three aspects of reconstruction (Table 7). Spatial layout determines which scene regions must be recovered, including vertical text, non-bottom placement, and multiple simultaneous regions. Temporal behavior afects whether nearby frames expose the covered content, as illustrated by moving text and persistent watermarks. Appearance and coverage vary through standard subtitles and artistic titles with display typography, outlines, or glow, while speech-aligned captions add audio-derived content and timing. Because these factors interact within each category, category comparisons measure coverage across conditions rather than isolated efects.

(a) Video and text planning  
(b) Type-controlled rendering (c) Target screening  
![](images/f84cbf5a595e980ef360dfc2d7fbf73489affa53bb383e25421c74bfd6682e47.jpg)  
Figure 2 DVTE-Bench construction. Text plans and type-controlled rendering produce aligned source videos, clean targets, and overlay masks. Target-frame OCR screening rejects 569 of 2,200 candidates and retains 1,631 clips. The eight thumbnails show distinct released clips, while the lower-right source, target, and mask form one aligned example.

## 3.2 Paired Data and Target Quality

The release contains 1,631 clips, 136,090 frames, and 91.1 minutes of video, with 1,085 portrait and 546 landscape clips. As outlined in Figure 2, scene-aware text plans or speech transcripts supply overlay content for type-controlled rendering. This rendering retains the original video as the reconstruction target and produces aligned overlay masks, including masks that follow moving text. Language metadata has 11 labels, dominated by Chinese, English, Korean, and Japanese, with the released distributions shown in Figure 6.

Because text in a reference can be mistaken for an unerased overlay, we screen every target frame with OCR and reject candidates with confirmed text, removing 569 of 2,200 candidates and retaining 1,631 (74.1%). Preservation of natural scene text requires a separate evaluation setting.

## 3.3 Regional Evaluation Protocol

All five methods are evaluated on the same 1,631 released clips, with scoring based on pixel-accurate GT masks from the released NPZ archives independently of the bounding-box masks supplied to mask-conditioned models.

Mask PSNR measures pixel fidelity, and Mask SSIM [17] averages the SSIM map over GT mask pixels. Mask LPIPS [18], DISTS [3], and VFID measure perceptual agreement with clean targets. For these perceptual metrics, exterior prediction pixels are replaced by the target, and a shared crop covers the union of mask locations over time to focus evaluation on the overlay region. Frame measurements are averaged within videos and then equally across videos, whereas VFID is a global distance between video-feature distributions. Infinite-PSNR videos are omitted from the finite PSNR mean and counted as $N _ { \infty }$

Temporal evaluation reports mask-restricted TWE for predictions and clean targets, their aggregate gap, and auxiliary TC and flow statistics using the released Farneback protocol. These diagnostics complement targetreferenced fidelity because temporally smooth outputs can still reconstruct the wrong content. Appendix C describes the regional computations and aggregation.

Shared weights · fixed source · evolving state  
(a) Source-anchored flow  
![](images/7ec83161dc73a9aa98ccd17f94309b878d4564ed7261cdfe3a5bbe7c9d55ee54.jpg)

![](images/91fc7b9239b68de0e02b0e372e9f78114cd57a0a727a63db2306a22a69002391.jpg)  
Figure 3 DeOverlay overview. (a) Shared-weight Euler updates start from the source latent, with the upper bypass carrying the current state and the lower loop updating state and time with fixed source cues. (b) Visibility-aware source aggregation (VSA), global source conditioning (GSC), and mask embeddings form the source condition. (c) Source cues are added to the current state tokens before joint video–text processing. Tensor grids and intermediate states are schematic, whereas RGB frames are matched source/output pairs.

## 4 DeOverlay: Source-Anchored Flow

Given an overlaid video, a removal instruction, and an edit mask, we encode the source as the initial editing state and use a shared velocity network to predict corrections with fixed local and global source guidance. A video decoder then maps the updated latent to the reconstructed video (Figure 3).

## 4.1 Source-Initialized Continuous Flow

To learn corrections from an overlaid video to its clean counterpart, we connect their latent representations with a paired flow path [12, 13]. Let $X , Y \in \mathbb { R } ^ { T \times H \times W \times 3 }$ denote the aligned source and target, with M marking afected pixels, including partially transparent regions. A video VAE encodes $z _ { s } = \mathcal { E } ( X )$ and $z _ { c } = \mathcal { E } ( Y )$ in $\mathbb { R } ^ { C \times T _ { \ell } \times \bar { H _ { \ell } } \times W _ { \ell } }$ , with D denoting decoding and c the removal instruction. The training path is

$$
a = z _ { s } + \sigma \epsilon , \quad \quad z _ { \tau } = ( 1 - \tau ) z _ { c } + \tau a , \quad \quad \epsilon \sim \mathcal { N } ( 0 , I ) , \quad \tau \in [ 0 , 1 ] ,\tag{1}
$$

with $\sigma = 0 . 0 5$ , the source endpoint at $\tau = 1$ , and the clean endpoint at $\tau = 0$ , giving the target velocity

$$
v ^ { * } = \partial z _ { \tau } / \partial \tau = a - z _ { c } .\tag{2}
$$

We sample intermediate τ so that the predictor $v _ { \theta } ( z _ { \tau } , \tau , \mathbf { c } ; z _ { s } , M _ { \mathrm { m o d e l } } )$ learns corrections at diferent positions along the path from the current state and time together with the original source and conditioning mask. The perturbation augments training, while inference starts directly at $z _ { s }$

## 4.2 Persistent Source Guidance

As editing changes the reconstruction, fixed source guidance keeps the original observations available. We combine local source features, a global summary, and the edit mask with the patchified state by addition:

$$
h _ { \tau } = P ( z _ { \tau } ) + A ( z _ { s } , M _ { \mathrm { m o d e l } } ) + G ( z _ { s } ) + m _ { \mathrm { t o k } } e _ { M } .\tag{3}
$$

Here $P$ embeds the state, $m _ { \mathrm { t o k } }$ is the token-grid mask, and $e _ { M }$ is a learned vector. All terms have the transformer width, with G broadcast over tokens, and their addition preserves the number of video tokens.

Visibility-aware source aggregation. Direct temporal averaging can mix overlay features with useful scene information. Inspired by the local/global decomposition in DifST [2], our visibility-aware source aggregation (VSA) excludes masked positions and normalizes by the number of visible observations. Let $s _ { t , p }$ be the source latent average-pooled to the token grid and $m _ { t , p }$ its max-pooled mask. Within $\mathcal { N } ( t ) = \{ t - 2 , \ldots , t + 2 \}$ , we compute

$$
\bar { s } _ { t , p } = \frac { \frac { 1 } { 5 } \sum _ { j \in \mathcal { N } ( t ) } ( 1 - m _ { j , p } ) s _ { j , p } } { \operatorname* { m a x } \Big ( \frac { 1 } { 5 } \sum _ { j \in \mathcal { N } ( t ) } ( 1 - m _ { j , p } ) , \delta \Big ) } , \qquad \delta = 1 0 ^ { - 4 } .\tag{4}
$$

The denominator prevents masked positions from attenuating the visible-feature mean, and we project s¯ to transformer width with a two-layer pointwise network before gating it to the edit region with a dilated mask. Video attention supplies broader context, but fixed-coordinate aggregation can mix structures under motion and provides no local observations in fully masked windows.

Global source conditioning. To provide scene-level context when persistent coverage leaves local observations unavailable, our global source conditioning (GSC) projects the source’s spatiotemporal mean over both overlaid and uncovered positions through a two-layer MLP. The resulting summary $G$ complements local features, while $m _ { \mathrm { t o k } } e _ { M }$ marks the edit region. Output projections and $e _ { M }$ start at zero, with further implementation details in Appendix A.

## 4.3 Learning the Velocity Field

To emphasize overlay correction, we assign larger loss weights to covered regions. Let $\widetilde { M }$ be the annotated mask resized to the latent grid and dilated by one cell. With $q _ { i }$ excluding overlap-conditioning elements, we minimize

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \frac { \sum _ { i } q _ { i } ( 1 + \lambda \widetilde { M } _ { i } ) ( v _ { \theta , i } - v _ { i } ^ { * } ) ^ { 2 } } { \operatorname* { m a x } ( \sum _ { i } q _ { i } , 1 ) } \right] , \qquad \lambda = 5 .\tag{5}
$$

The index spans channel, time, space, and batch, with covered elements receiving weight six and exterior elements weight one, while the denominator counts supervised elements without weighting.

We adapt the dense 1.3B LingBot-Video backbone [14, 15] with rank-256 LoRA [7], directly training the patch embedding and source-guidance modules while freezing the VAE and text encoder. Conditioned video tokens and projected instruction tokens undergo joint attention. With probability 0.2, missing-mask training sets $M _ { \mathrm { m o d e l } } = 0$ , zeroing local guidance and the edit embedding while retaining global guidance. The annotated mask continues to determine loss weights, and Table 6 records the full configuration.

## 4.4 Shared-Weight Flow Updates

Starting at $z ^ { ( 0 ) } = z _ { s }$ , we use a descending schedule $1 = \tau _ { 0 } > \cdot \cdot \cdot > \tau _ { K } = 0 \mathrm { { : } }$

$$
z ^ { ( k + 1 ) } = z ^ { ( k ) } + \Delta \tau _ { k } \widehat v _ { \theta } ( z ^ { ( k ) } , \tau _ { k } , \mathbf { c } ; z _ { s } , M _ { \mathrm { m o d e l } } ) , \qquad \Delta \tau _ { k } = \tau _ { k + 1 } - \tau _ { k } < 0 .\tag{6}
$$

The state and time change while weights, source, and mask remain fixed, so one checkpoint supports multiple step counts. We use a linear–quadratic schedule with a resolution-dependent shift and apply classifier-free guidance [6] at scale $g = 2$ with two network evaluations per step, so six steps require 12 evaluations. The core output is $\widehat { Y } = \mathcal { D } ( z ^ { ( K ) } )$ , with chunked inference and output processing specified in Appendix A.

Although each training pair defines a straight latent path, the learned velocity can vary with the state and time during inference. We therefore evaluate how the number of flow updates afects reconstruction quality in Section 5.3. Appendix B analyzes the integration and endpoint errors.

Table 2 DVTE-Bench results on all 1,631 videos, using the released pixel-mask protocol. (a) PSNR<sup>†</sup> excludes infinite values, and VFID is a global video-feature distance. (b) TWE Gap is $| \overline { { \mathrm { T W E } } } _ { \mathrm { p r e d } } - \overline { { \mathrm { T W E } } } _ { \mathrm { G T } } |$ , with clean-target TWE 9.3228 and diagnostics defined in Appendix I. Bold marks best values in (a), while bold/underline in (b) mark the closest/second-closest TWE Pred to GT and the lowest/second-lowest values otherwise.
<table><tr><td colspan="6">(a) Regional reconstruction and perceptual quality</td></tr><tr><td>Method</td><td>PSNR†↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>VFID↓</td></tr><tr><td>CLEAR</td><td>16.02</td><td>0.5172</td><td>0.3209</td><td>0.2158</td><td>1.4736</td></tr><tr><td>ProPainter</td><td>25.90</td><td>0.7832</td><td>0.1166</td><td>0.0957</td><td>0.3879</td></tr><tr><td>MiniMax-Remover</td><td>26.17</td><td>0.8044</td><td>0.0977</td><td>0.0856</td><td>0.2806</td></tr><tr><td>DiffuEraser</td><td>24.84</td><td>0.7723</td><td>0.1130</td><td>0.0952</td><td>0.3285</td></tr><tr><td>DeOverlay, K = 6</td><td>30.80</td><td>0.8822</td><td>0.0408</td><td>0.0507</td><td>0.0835</td></tr></table>

<table><tr><td colspan="6">(b) Regional temporal diagnostics</td></tr><tr><td>Method</td><td>TWE Pred</td><td>TWE Gap</td><td>TC</td><td>Flow Mean</td><td>Flow Var</td></tr><tr><td>CLEAR</td><td>10.2373</td><td>0.9144</td><td>11.0261</td><td>0.9014</td><td>2.8169</td></tr><tr><td>ProPainter</td><td>8.1428</td><td>1.1801</td><td>4.0349</td><td>0.6496</td><td>1.5601</td></tr><tr><td>MiniMax-Remover</td><td>8.7095</td><td>0.6133</td><td>4.5586</td><td>0.7991</td><td>2.3072</td></tr><tr><td>DiffuEraser</td><td>10.1853</td><td>0.8624</td><td>5.3240</td><td>0.6690</td><td>1.9385</td></tr><tr><td>DeOverlay</td><td>9.3110</td><td>0.0118</td><td>4.2498</td><td>0.3333</td><td>0.5921</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Training, inference, and baselines. Training uses 350,000 aligned overlaid/clean clip pairs with masks: 250,000 base samples, 50,000 stylized-text samples, and 50,000 mirrored-text samples. These clips are disjoint from DVTE-Bench, with corpus construction and optimization settings detailed in Appendix A. Full-release evaluation uses the 40K checkpoint, GT bounding-box input masks, K = 6, guidance scale g = 2 (12 network evaluations), and 12-pixel seamless paste-back. We compare against ProPainter [19], CLEAR [5], MiniMax-Remover [20], and DifuEraser [11], spanning propagation, subtitle-specific, and generative removal, with non-subtitle cases testing CLEAR’s transfer beyond its original task. All methods share GT scoring masks, while input-mask access and postprocessing follow their released pipelines (Appendix D). SEDiT [8] is compared conceptually because an executable model was unavailable for our matched evaluation. Ablations use 200 randomly sampled videos from DVTE-Bench.

## 5.2 Reconstruction across Diverse Overlays

On all 1,631 released videos, DeOverlay achieves the best results on all five reconstruction and perceptual metrics (Table 2). Compared with MiniMax-Remover, the strongest baseline on these metrics, it improves Mask PSNR from 26.17 to 30.80 dB and SSIM from 0.8044 to 0.8822, while reducing LPIPS, DISTS, and VFID by 58.3%, 40.8%, and 70.3%, respectively. These improvements support more faithful covered-region reconstruction.

Coverage across overlay conditions. DeOverlay achieves the best PSNR, SSIM, LPIPS, and DISTS across all eight overlay types (Appendix D). Artistic titles produce its lowest PSNR/SSIM (26.24 dB/0.8409), whereas vertical text produces its highest LPIPS/DISTS (0.0975/0.1015). The most dificult category therefore depends on whether pixel accuracy or perceptual similarity is measured.

Temporal behavior. Table 2(b) reports the released temporal diagnostics alongside reconstruction fidelity, with DeOverlay obtaining TWE Pred of 9.3110 and TC of 4.2498, compared with ProPainter’s 8.1428 and 4.0349. All methods use the same mask-restricted computation and clean-target reference of 9.3228. Under this computation, ProPainter and MiniMax-Remover score below the reference despite weaker reconstruction, illustrating why the diagnostic is interpreted together with clean-target fidelity. Appendix I specifies the temporal computations.

Source

CLEAR

ProPainter

MiniMax-Remover

DiffuEraser

DeOverlay

Ground truth

![](images/0eb5dadebccb26e2c2c1c9b5d1934b4e087ac3b375771031f5b14d7ab9209cd1.jpg)

Figure 4 Paired benchmark comparisons. Five frame-40 examples with shared yellow crop boxes compare four baselines, DeOverlay, and clean targets, with inference settings and expanded seven-category comparisons in Appendix E.  
![](images/863a7cbebec6884f5ed37db0e746b46ad95f1ce87fe953e7af78af43ca2806fe.jpg)  
Figure 5 Removal across overlay appearances. Twelve matched source/output crops cover subtitles, titles, watermarks, a sticker, and a television logo, with full frames and crop boxes in Appendix E.

DeOverlay has an aggregate TWE Gap of 0.0118, versus 0.6133 for MiniMax-Remover and 1.1801 for ProPainter, where the gap compares dataset-level means under the aggregation rule in Appendix I. The five target-referenced metrics in Table 2(a) therefore provide the primary reconstruction comparison, while Flow Mean, Flow Var, and TC retain their released labels and scales as supplementary descriptions of the evaluated outputs.

Qualitative reconstruction. Across the five paired examples with clean targets in Figure 4, residual text, blurred structures, and altered colors distinguish the outputs, while DeOverlay more closely preserves the target appearance. In the artistic-title example, the restored car and road remain aligned with the clean target. The multiple-region and non-bottom-text examples also make color shifts and damaged background boundaries visible around the building and clouds, complementing the aggregate scores by showing which structures are recovered within the covered regions. Figure 5 adds twelve application examples spanning subtitle languages and density, title layouts, watermarks, a sticker, and a television logo. Appendix E provides seven-category comparisons, application full frames, and additional successes and failures.

Table 3 Core design ablations on 200 randomly sampled videos. Component variants use $K = 4 ,$ and the direct-residual reference uses $K = 1$ . PSNR and MAE use overlay pixels, with ∆PSNR measured relative to the full four-step model. VSA supplies local source context and GSC supplies global context, while the noise-start variant concatenates noise and source video tokens. Bold marks the best value in each quality column.
<table><tr><td>Variant</td><td>PSNR↑</td><td>MAE↓</td><td>Crop SSIM↑</td><td>Outside MAE↓</td><td>TWE gap↓</td><td>∆PSNR</td></tr><tr><td>DeOverlay,  $K = 4$ </td><td>32.461</td><td>4.299</td><td>0.92765</td><td>2.541</td><td>0.180</td><td>0.000</td></tr><tr><td>Direct residual,  $K = 1$ </td><td>31.300</td><td>4.520</td><td>0.91120</td><td>2.548</td><td>0.195</td><td>-1.161</td></tr><tr><td>Noise init. + source tokens</td><td>29.600</td><td>6.500</td><td>0.89200</td><td>3.050</td><td>0.320</td><td>-2.861</td></tr><tr><td>Mask only (no VSA/GSC)</td><td>32.230</td><td>4.440</td><td>0.92590</td><td>2.552</td><td>0.197</td><td>-0.231</td></tr><tr><td>Ordinary temporal mean</td><td>32.360</td><td>4.360</td><td>0.92725</td><td>2.542</td><td>0.183</td><td>-0.101</td></tr><tr><td>Without VSA</td><td>32.370</td><td>4.350</td><td>0.92705</td><td>2.543</td><td>0.185</td><td>-0.091</td></tr><tr><td>Without GSC</td><td>32.290</td><td>4.410</td><td>0.92645</td><td>2.550</td><td>0.193</td><td>-0.171</td></tr></table>

Crop SSIM denotes MaskCrop SSIM, and TWE gap denotes the TWE absolute gap.

Table 4 Flow step count on 200 randomly sampled videos. Only the inference update count changes under a fixed checkpoint, with latency normalized to $K = 4$ and the best quality metric marked in bold.
<table><tr><td>K</td><td>Mask PSNR↑</td><td>Mask MAE↓</td><td>MaskCrop SSIM↑</td><td>TWE abs. gap↓</td><td>Rel. latency</td></tr><tr><td>2</td><td>32.410</td><td>4.330</td><td>0.92710</td><td>0.184</td><td>0.82×</td></tr><tr><td>4</td><td>32.461</td><td>4.299</td><td>0.92765</td><td>0.180</td><td>1.00×</td></tr><tr><td>6</td><td>32.467</td><td>4.294</td><td>0.92772</td><td>0.178</td><td>1.25×</td></tr><tr><td>8</td><td>32.468</td><td>4.293</td><td>0.92774</td><td>0.178</td><td>1.50×</td></tr></table>

## 5.3 Ablation Studies

Ablations use 200 randomly sampled DVTE-Bench videos, whereas Table 2 uses all 1,631 videos. Component variants use $K = 4 .$ with a direct-residual reference at K = 1, and the step sweep holds the checkpoint fixed. We measure ablation SSIM with MaskCrop SSIM.

Efect of source representation and guidance. Replacing source initialization and additive guidance with noise initialization and concatenated source tokens decreases Mask PSNR by 2.861 dB (Table 3). Because the starting state and conditioning change together, this comparison evaluates their combined design.

Within the source-initialized model, keeping only the mask embedding reduces PSNR by 0.231 dB, while removing VSA or GSC reduces it by 0.091 or 0.171 dB. These changes show that both local observations and global context contribute to the measured reconstruction quality. Replacing visibility-normalized aggregation with an ordinary temporal mean reduces PSNR by 0.101 dB and raises MAE to 4.360, supporting the use of visibility selection in local aggregation.

Efect of flow step count. Increasing from two to four steps within the same checkpoint improves Mask PSNR from 32.410 to 32.461 dB, lowers MAE from 4.330 to 4.299, and improves Crop SSIM and TWE gap (Table 4). Eight steps add only 0.007 dB over four at 1.50× latency, favoring four steps for these component studies.

Training formulation and repeated updates. Four-step flow exceeds the separately trained direct-residual reference by 1.161 dB in Mask PSNR, with lower MAE and higher Crop SSIM (Table 3). Each flow update recomputes the velocity from the current state and time with fixed source cues, allowing diferent inference budgets without retraining.

## 5.4 Efficiency

End-to-end speed. The comparison in Table 5 uses eight shared clips (650 frames), run serially on one A100, with timing that includes decoding, postprocessing, and encoding but excludes model loading. One-step DeOverlay reaches 3.0847 FPS, 1.17× MiniMax-Remover’s throughput, while four steps reach 2.0712 FPS. Both configurations have a sampled peak memory of 12.236 GiB, compared with the lowest value of 10.776 GiB for MiniMax-Remover. Appendix G specifies the measurement boundaries.

Table 5 Speed and memory on the same A100. All rows process the same eight clips (650 frames), with E2E denoting end-to-end time excluding model initialization. Inference time follows each runner’s CUDA-synchronized interval, and peak memory uses a common external GPU-memory sampler. Bold marks the best value in each column.
<table><tr><td>Method</td><td>E2E s/frame↓</td><td>E2E FPS↑</td><td>Inference s/frame↓</td><td>Peak GiB↓</td></tr><tr><td>CLEAR</td><td>0.6030</td><td>1.6584</td><td>0.5983</td><td>18.354</td></tr><tr><td>ProPainter</td><td>0.4271</td><td>2.3413</td><td>0.4134</td><td>34.101</td></tr><tr><td>MiniMax-Remover</td><td>0.3787</td><td>2.6405</td><td>0.3530</td><td>10.776</td></tr><tr><td>DiffuEraser</td><td>1.1677</td><td>0.8564</td><td>1.1524</td><td>23.950</td></tr><tr><td>DeOverlay, K = 1</td><td>0.3242</td><td>3.0847</td><td>0.3001</td><td>12.236</td></tr><tr><td>DeOverlay, K = 4</td><td>0.4828</td><td>2.0712</td><td>0.4587</td><td>12.236</td></tr></table>

Cost of a second token stream. With the same 32K weights and four transformer forwards, adding a sourcetoken stream increases tokens from 13,346 to 26,594, median time by 2.485×, and incremental peak memory by 1.995× (Table 9). These measurements isolate transformer cost, whereas end-to-end timing also includes video-processing overhead. Additive guidance preserves the video sequence length, while increasing K repeats the predictor.

## 6 Limitations

DVTE-Bench focuses on synthetic text overlays, leaving graphic logos, stickers, and preservation of natural scene text for broader paired evaluation. When neighboring frames provide no visible evidence for a fully covered background, reconstruction depends on the video prior. The application examples also include residual graphics and local texture artifacts (Appendix E).

## 7 Conclusion

We introduced DVTE-Bench and DeOverlay for reconstructing scenes beneath video text overlays, with DVTE-Bench unifying eight overlay categories in 1,631 paired videos. Clean targets, aligned pixel masks, and a common evaluation protocol support comparisons across overlay layouts, temporal behavior, and appearance.

DeOverlay combines source initialization with visibility-normalized local aggregation and global source guidance, keeping the source available through additive conditioning throughout shared-weight flow updates using one video-token sequence. It outperforms four baselines on all five reconstruction and perceptual metrics, and on the 200-video subset, four-step flow exceeds the separately trained direct-residual reference by 1.161 dB in Mask PSNR. Component ablations support both guidance branches, while category-level evaluation separates the challenges of pixel reconstruction and perceptual quality, with artistic titles and vertical text stressing diferent metrics. Together, the benchmark and method enable joint study of scene recovery and inference cost.

## References

[1] Junye Chen, Chaowei Fang, Jichang Li, Yicheng Leng, and Guanbin Li. Decouple and couple: Exploiting prior knowledge for visible video watermark removal. IEEE Transactions on Image Processing, 34:1192–1203, 2025. doi: 10.1109/TIP.2025.3534033. URL https://doi.org/10.1109/TIP.2025.3534033.

[2] Zheng Chen, Ruofan Yang, Jin Han, Dehua Song, Zichen Zou, Chunming He, Yong Guo, and Yulun Zhang. DifST: Spatiotemporal-aware difusion for real-world space-time video super-resolution. arXiv preprint arXiv:2605.13182, 2026. URL https://arxiv.org/abs/2605.13182.

[3] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2022. doi: 10.1109/TPAMI.2020.3045810. URL https://doi.org/10.1109/TPAMI.2020.3045810.

[4] Zihao Fan, Xin Lu, Jie Xiao, Dong Li, Jie Huang, and Xueyang Fu. IR-Flow: Bridging discriminative and generative image restoration via rectified flow. Machine Intelligence Research, 2026. doi: 10.1007/s11633-026-1679-0. URL https://arxiv.org/abs/2604.19680.

[5] Qingdong He, Chaoyi Wang, Peng Tang, Yifan Yang, and Xiaobin Hu. CLEAR: Context-aware learning with end-to-end mask-free inference for adaptive subtitle removal. In International Conference on Machine Learning, 2026. URL https://icml.cc/virtual/2026/poster/65005.

[6] Jonathan Ho and Tim Salimans. Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598, 2022. URL https://arxiv.org/abs/2207.12598.

[7] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2106.09685.

[8] Zheng Hui and Yunlong Bai. SEDiT: Mask-free video subtitle erasure via one-step difusion transformer. arXiv preprint arXiv:2605.14894, 2026. URL https://arxiv.org/abs/2605.14894.

[9] Sangwoo Jo, Donggeun Ko, Jayeon Kang, Youngsang Kwak, Jaehwa Kwak, and Sungjoon Choi. Flow straight to reality: Perceptually consistent flow matching for eficient image restoration. In Computer Vision – ECCV 2026, volume 17066 of Lecture Notes in Computer Science, pages 223–239. Springer, 2026. doi: 10.1007/ 978-3-032-37718-0\_14. URL https://link.springer.com/chapter/10.1007/978-3-032-37718-0\_14.

[10] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. FlowEdit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. URL https://arxiv.org/abs/2412.08629.

[11] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. DifuEraser: A difusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025. URL https://arxiv.org/abs/2501.10018.

[12] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=PqvMRDCJT9t.

[13] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023. URL https://arxiv.org/ abs/2209.03003.

[14] Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang, Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Ka Leong Cheng. Scaling mixture-of-experts video pretraining for embodied intelligence. arXiv preprint arXiv:2607.07675, 2026. URL https://arxiv.org/abs/2607.07675.

[15] Robbyant. LingBot-Video. GitHub repository, 2026. URL https://github.com/Robbyant/lingbot-video. Accessed September 24, 2026.

[16] Mohammadreza Salehi, Mehdi Noroozi, Luca Morreale, Ruchika Chavhan, Malcolm Chadwick, Alberto Gil Couto Pimentel Ramos, and Abhinav Mehrotra. RFDM: Residual flow difusion models for video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. URL https: //cvpr.thecvf.com/virtual/2026/poster/38778.

[17] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861. URL https://www.cns.nyu.edu/pub/lcv/wang03-reprint.pdf.

[18] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. URL https://arxiv.org/abs/1801.03924.

[19] Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, and Chen Change Loy. ProPainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10477–10486, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Zhou\_ProPainter\_ Improving\_Propagation\_and\_Transformer\_for\_Video\_Inpainting\_ICCV\_2023\_paper.html.

[20] Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. MiniMax-Remover: Taming bad noise helps video object removal. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-2536. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/hash/6d255ddeb1f4a5ebb8e448b3998bb8e5-Abstract-Conference.html.

## Appendix Contents

A Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12   
B Euler Error and Endpoint Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13   
C Metric Definitions and Reporting Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15   
D Overlay Categories and Evaluation Coverage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15   
E Qualitative Exports and Case Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17   
F Ablation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25   
G Timing and Optimization Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25   
H Complete Per-Category Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27   
I Temporal Diagnostic Computation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

## A Implementation Details

Training corpus. The training corpus contains 350,000 paired samples: 250,000 base samples, 50,000 stylizedtext samples, and 50,000 mirrored-text samples, each providing an overlaid input, its aligned clean target, and an overlay mask. All training samples are separate from the 1,631 DVTE-Bench clips, including its 68 speech-aligned subtitle clips. We sample the base, stylized, and mirrored subsets with weights 4 : 2 : 1, respectively, which specify sampling frequencies rather than corpus sizes.

Table 6 gives the multistep configuration.

Table 6 Configured model and optimization settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>LingBot-Video dense 1.3B configuration</td></tr><tr><td>Latent channels / patch size</td><td>16 / (1, 2, 2)</td></tr><tr><td>Transformer blocks / width / heads</td><td>24 / 2048 / 16</td></tr><tr><td>Feed-forward width / text embedding width</td><td>6144 / 2560</td></tr><tr><td>LoRA rank / scale</td><td>256  /  1.0</td></tr><tr><td>Local / global branch hidden width</td><td>512 / 512</td></tr><tr><td>Local temporal window</td><td>5 latent positions</td></tr><tr><td>Local mask dilation radius</td><td>1 token-grid cell</td></tr><tr><td>Loss-mask dilation radius</td><td>1 latent-grid cell</td></tr><tr><td>Model-mask dropout probability</td><td>0.2</td></tr><tr><td>Source-endpoint perturbation σ</td><td>0.05 (training)</td></tr><tr><td>Overlay-region loss coefficient λ</td><td>5.0</td></tr><tr><td>Pixel-crop / high-frequency loss weights</td><td>0 /0</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW  $/ \ 5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Betas / weight decay</td><td>(0.9, 0.95) / 0</td></tr><tr><td>Warm-up / maximum configured updates</td><td>300 / 60,000</td></tr><tr><td>Precision / gradient clipping</td><td>bfloat16 / norm 1.0</td></tr><tr><td>Preview steps / batch inference default</td><td>4/6</td></tr><tr><td>DVTE-Bench checkpoint / mask / steps</td><td>40K /  GT / 6</td></tr><tr><td>Guidance scale</td><td></td></tr><tr><td>VAE posterior mode</td><td>2.0 Sample</td></tr></table>

Token-grid conditioning. Before local context aggregation, the transformer averages the source latent over each spatial patch and max-pools the model mask to the same grid, with a temporal patch size of one. The local branch therefore operates on patch-grid source features, while the global branch averages the original latent over all temporal and spatial positions. Accordingly, mask dilation in the local branch and dilation in the loss use their respective resolutions.

Source-guidance projections. The local projection uses two $1 \times 1 \times 1$ convolutions with hidden width 512:

$$
A _ { t , p } = \mathrm { d i l a t e } ( m ) _ { t , p } W _ { A , 2 } ( \mathrm { S i L U } ( W _ { A , 1 } \bar { s } _ { t , p } + b _ { A , 1 } ) ) + \mathrm { d i l a t e } ( m ) _ { t , p } b _ { A , 2 } .\tag{7}
$$

Dilation has radius one on the token grid, and padded temporal positions contribute zero to both sums in Eq. 4. The global source branch uses a two-layer MLP with hidden width 512:

$$
G ( z _ { s } ) = W _ { G , 2 } \mathrm { S i L U } \left( W _ { G , 1 } \frac { 1 } { T _ { \ell } H _ { \ell } W _ { \ell } } \sum _ { t , p } z _ { s , t , p } + b _ { G , 1 } \right) + b _ { G , 2 } .\tag{8}
$$

Both guidance output projections and the mask embedding are initialized to zero, and joint video–text self-attention uses three-dimensional rotary embeddings, time-dependent modulation, and dense feed-forward layers. LoRA of rank 256 adapts eligible linear layers, while the patch embedding, guidance modules, and mask embedding are trained directly. The VAE and text encoder remain fixed, with posterior sampling used in the VAE and zero weight assigned to pixel-crop and high-frequency losses.

Guidance and deployment. For classifier-free guidance [6], positive and negative text conditions give

$$
\widehat { v } _ { \theta } = v _ { \theta } ^ { - } + g ( v _ { \theta } ^ { + } - v _ { \theta } ^ { - } ) .\tag{9}
$$

With the source and conditioning mask retained in both evaluations, the default $g = 2$ requires two network evaluations per Euler step. The decoder maps the final latent to the output video, and long videos use overlapping chunks in which previous outputs supply latent boundary conditions. Deployment derives text masks from OCR, while removing non-text overlays requires a mask covering the designated graphic region. Full-benchmark evaluation uses 12-pixel seamless pixel paste-back, with the method-specific output processing specified in Appendix D.

Resolution and time sampling. The configured data buckets use 41 frames at a short-side resolution of 768 for videos with original long side at least 1920, 65 frames at 768 for long side at least 1280, and 121 frames at 608 for smaller inputs. The decoder samples a random starting frame with unit frame stride. The training sampler uses the continuous shifted-mode distribution with mode scale 1.1. The configured resolution shift ranges from 1 to 5 over the selected token-count range, and inference also applies the resolution shift to the linear–quadratic schedule.

Training and inference procedure. Algorithm 1 describes one training draw and core inference, omitting batch padding and chunk overlap for clarity. When overlap conditioning is active, the corresponding latent elements are supplied as conditions and excluded from the target mask q in Eq. 5.

## B Euler Error and Endpoint Prediction

Repeated updates change both the states seen by the predictor and the numerical approximation of its flow. For a known training pair, the displacement in Eq. 2 is constant, and an oracle reaches the clean latent from the perturbed source endpoint in one step. The network estimates this displacement from the current state and time, allowing its prediction to change along the trajectory.

Consider the learned guided field $f ( z , \tau ) = \widehat { v } _ { \theta } ( z , \tau , \mathbf { c } ; z _ { s } , M _ { \mathrm { m o d e l } } )$ with fixed conditions, and let $z _ { f } ( \tau )$ be its exact ODE solution initialized at $z _ { s } , \mathrm { ~ H ~ } f$ is Lipschitz in z and suficiently smooth along this solution, Euler integration on a fixed interval has global error of order h, where $h = \operatorname* { m a x } _ { k } \left| \Delta \tau _ { k } \right|$ . The endpoint reconstruction error can be decomposed as

$$
\begin{array} { r } { \| z ^ { ( K ) } - z _ { c } \| \leq \underbrace { \| z ^ { ( K ) } - z _ { f } ( 0 ) \| } _ { \mathrm { i n t e g r a t i o n ~ e r r o r } } + \underbrace { \| z _ { f } ( 0 ) - z _ { c } \| } _ { \mathrm { l e a r n e d - f l o w ~ e n d p o i n t ~ e r r o r } } . } \end{array}\tag{10}
$$

Reducing the maximum step size h tightens the integration-error bound under these assumptions. Endpoint accuracy also depends on the learned field and uncertainty about hidden content, so integration accuracy alone does not establish improved detail. The following derivation gives the error recursion and explains the state-dependent predictor. The shared-checkpoint study in Section 5.3 tests reconstruction quality directly as K varies.

Algorithm 1 DeOverlay training and inference   
Require: Paired X, Y, M, instruction c, velocity network $v _ { \theta }$   
Training   
1: Encode $z _ { s } \gets \mathcal { E } ( X )$ and $z _ { c } \gets \mathcal { E } ( Y )$   
2: Sample $\epsilon \sim \mathcal { N } ( 0 , I )$ and continuous time $\tau$   
3: $a  z _ { s } + 0 . 0 5 \epsilon ; z _ { \tau }  ( 1 - \tau ) z _ { c } + \tau a$   
4: Set $M _ { \mathrm { m o d e l } }  0$ with probability 0.2, otherwise M   
5: Predict $v  v _ { \theta } ( z _ { \tau } , \tau , \mathbf { c } ; z _ { s } , M _ { \mathrm { m o d e l } } )$   
6: Update trainable parameters using Eq. 5, target $\boldsymbol { a } - \boldsymbol { z } _ { c } ,$ and true mask M   
Inference   
7: Encode $z _ { s } \gets \mathcal { E } ( X ) ;$ set $z ^ { ( 0 ) } \gets z _ { s }$   
8: Construct descending schedule $\tau _ { 0 } = 1 > \dots > \tau _ { K } = 0$   
9: for $k = 0 , \ldots , K - 1$ do   
10: Evaluate guided velocity $\widehat { v } _ { \theta }$ with fixed source and mask   
11: $z ^ { ( k + 1 ) } \gets \overline { { z } } ^ { ( k ) } + ( \tau _ { k + 1 } - \tau _ { k } ) \widehat { v } _ { \theta }$   
12: end for   
13: return $\mathcal { D } ( \boldsymbol { z } ^ { ( K ) } )$

To derive the standard Euler approximation bound for the learned field, set $u = 1 - \tau$ and $F ( z , u ) = - f ( z , 1 - u )$ so integration proceeds forward from $u = 0$ to $u = 1$ . Assume F is Lipschitz in z with constant L on a region containing the exact and numerical trajectories, and that the exact solution satisfies $\| d ^ { 2 } z _ { F } / d u ^ { 2 } \| \leq B$ there. Taylor expansion over an interval $h _ { k }$ then gives

$$
\begin{array} { r } { z _ { F } ( u _ { k + 1 } ) = z _ { F } ( u _ { k } ) + h _ { k } F ( z _ { F } ( u _ { k } ) , u _ { k } ) + r _ { k } , \qquad \| r _ { k } \| \leq \frac { 1 } { 2 } B h _ { k } ^ { 2 } . } \end{array}\tag{11}
$$

Subtracting this update from Euler and using the Lipschitz condition yields

$$
\begin{array} { r } { e _ { k + 1 } \leq ( 1 + L h _ { k } ) e _ { k } + \frac { 1 } { 2 } B h _ { k } ^ { 2 } , \qquad e _ { k } = \| z ^ { ( k ) } - z _ { F } ( u _ { k } ) \| . } \end{array}\tag{12}
$$

With the same initial condition, $e _ { 0 } = 0 ,$ , and setting $h = \operatorname* { m a x } _ { k } h _ { k }$ gives the iterated bound, for $L > 0$

$$
e _ { K } \leq \frac { B h } { 2 L } ( e ^ { L } - 1 ) ,\tag{13}
$$

and the $L = 0$ limit is $B h / 2$ . Here, L and B are assumed constants rather than measured properties of the trained network, and the bound concerns approximation of its ODE rather than the distance between the ODE endpoint and the clean target.

Why a straight paired path can yield a varying predictor. The population minimizer of an unweighted squared velocity loss is the conditional mean

$$
v ^ { * } ( z , \tau , \xi ) = \mathbb { E } [ a - z _ { c } \mid z _ { \tau } = z , \tau , \xi ] , \qquad \xi = ( z _ { s } , M _ { \mathrm { m o d e l } } , \mathbf { c } ) .\tag{14}
$$

For the coordinate-weighted loss in Eq. 5, the corresponding minimizer is the weighted conditional mean, $\mathbb { E } [ w _ { i } ( a _ { i } - z _ { c , i } ) \mid z , \tau , \xi ] / \mathbb { E } [ w _ { i } \mid z , \tau , \xi ]$ , when the denominator is positive and the coordinate’s valid-sample weighting is absorbed into $w _ { i } .$ . Although a particular paired displacement is constant, the conditional distribution over displacements can depend on the observed intermediate state, which explains why evaluating the field again can change the predicted correction. This dependence does not ensure faithful recovery of a multimodal hidden background, an outcome tested directly by the matched-checkpoint experiment.

## C Metric Definitions and Reporting Protocol

Tables 2 and 8 use the complete 1,631-video release and pixel-accurate mask\_packed GT masks, with scores transcribed from the released machine-readable records. Except for global VFID and the aggregate TWE gap, frame or frame-pair measurements are averaged within videos and then equally across videos. For the finite PSNR mean, infinite-valued video scores are omitted and counted separately.

Pixel and structural fidelity. Mask PSNR is computed from errors over GT mask pixels in each frame, while Mask SSIM averages the spatial SSIM map over those pixels rather than over the enclosing rectangle. Input masks and scoring masks are separate, so a pipeline may receive bounding boxes while its output is scored on the common pixel mask.

Perceptual fidelity. For LPIPS and DISTS, form $\widehat { Y } _ { t } ^ { M } = M _ { t } \odot \widehat { Y } _ { t } + ( 1 - M _ { t } ) \odot Y _ { t }$ to replace the exterior prediction with clean-target pixels, then use a temporally stable crop enclosing the union of mask locations over the video for both prediction and target. Active-mask frames are resized to a short side of 256 pixels with the long side capped at 512. For VFID, the same GT-composited crop video supplies RGB-I3D features of dimension 1,024 through the ProPainter evaluator, and the global Fréchet distance is computed between feature distributions rather than averaged from per-video or per-type distances. These constructions isolate errors around the overlay and do not measure exterior preservation.

Released temporal diagnostics. Optical flow uses Farneback with maximum image side 320, and the temporal region is the union of the current GT mask and the previous mask transported to the current frame by GT backward flow. Flow statistics instead use the previous mask union the current mask sampled into previous-frame coordinates by GT forward flow. TWE uses the sampling operation defined in Appendix I and restricts error aggregation to the mask region, with RGB errors on the [0, 255] scale. TC, Flow Mean, and Flow Var retain the release’s named diagnostics and numerical scale, as reported in Table 2 using this released computation.

The released TWE gap is computed after aggregation:

$$
\Delta _ { \mathrm { T W E } } = \left| \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { T W E } ( \widehat { Y } _ { i } ) - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { T W E } ( Y _ { i } ) \right| , \qquad N = 1 , 6 3 1 .\tag{15}
$$

With a common target reference of 9.3228, this dataset-level statistic accompanies the target-referenced reconstruction metrics, and Appendix I specifies its interpretation.

## D Overlay Categories and Evaluation Coverage

Table 7 DVTE-Bench coverage and complete evaluation. All released clips are evaluated by five methods under the protocol defined in Section 3.3.

<table><tr><td>Challenge type</td><td>Overlay characteristic</td><td>Release</td><td>Evaluated</td></tr><tr><td>Standard subtitle</td><td>Lower-third captions and style variation</td><td>232</td><td>232</td></tr><tr><td>Artistic title</td><td>Display typography, outlines, and glow</td><td>218</td><td>218</td></tr><tr><td>Moving text</td><td>Frame-varying trajectories</td><td>232</td><td>232</td></tr><tr><td>Vertical text</td><td>Vertical writing at varied positions</td><td>215</td><td>215</td></tr><tr><td>Non-bottom layout</td><td>Placement outside the bottom region</td><td>223</td><td>223</td></tr><tr><td>Multi-region</td><td>Distributed simultaneous text regions</td><td>225</td><td>225</td></tr><tr><td>Persistent watermark</td><td>Persistent semi-transparent text marks</td><td>218</td><td>218</td></tr><tr><td>ASR subtitle</td><td>Speech-aligned captions</td><td>68</td><td>68</td></tr><tr><td colspan="2">Full release</td><td>1,631</td><td>1,631</td></tr></table>

Overlay function and benchmark challenge type describe diferent properties: subtitles, text watermarks, titles, and name or role captions describe function, while position, motion, opacity, and afected area describe appearance and layout. A vertical name caption, for example, combines a function with a spatial layout, although the current benchmark assigns each clip one challenge type. Its clean-target OCR policy excludes confirmed text, so preservation of natural scene text requires a separate evaluation setting.

![](images/f80d4c7d4d3ae6df53acaf3014931457c86c3e430f04472b8899d0edee7d1099.jpg)

![](images/2a5f4a3ddaf497dbcfc1a48f2f82bccd6ce2450ff61df24ecb8d4b1a3a86f309.jpg)

![](images/470483602e4812788934d140ee45cdeb7a42b2b2c52b963c3fe4754fc69748a6.jpg)  
Figure 6 DVTE-Bench release distributions. All panels count the 1,631 released clips: (a) assigned overlay type, (b) language metadata, (c) frame orientation, and (d) the per-clip number of rendered overlay elements. “Other labels” combines Italian (2) and mixed, Spanish, Portuguese, Polish, and Welsh (1 each). Overlay-element count does not measure spatial regions or pixel coverage.

Dataset composition. The release contains 1,631 clips across eight overlay types, evaluated under the protocol defined in Section 3.3. Figure 6 summarizes the manifest counts, including 1,085 portrait and 546 landscape clips. Language metadata records Chinese (845), English (350), Korean (201), Japanese (178), Thai (50), Italian (2), and one clip each labeled mixed, Spanish, Portuguese, Polish, and Welsh. These counts describe coverage, but the rare labels do not support reliable per-language comparisons. Every target frame is scanned under the OCR rejection policy described in Section 3.

DVTE-Bench organizes overlay conditions by layout, motion, typography, and caption timing, with Table 8 reporting DeOverlay across all eight types and 1,631 clips. Aggregates weight videos equally rather than assigning equal weight to each type, and DeOverlay leads Mask PSNR, SSIM, LPIPS, and DISTS among the five evaluated methods in each type.

Table 8 DeOverlay across all eight DVTE-Bench types. All 1,631 clips are included, with the 40K checkpoint using GT bounding-box masks, six steps, and the released seamless paste-back setting. Scoring uses pixel masks, and $N _ { \infty }$ counts clips omitted only from finite-mean PSNR.
<table><tr><td>Text type</td><td>Clips</td><td>PSNR† ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>N∞</td></tr><tr><td>Standard subtitle</td><td>232</td><td>30.39</td><td>0.8718</td><td>0.0285</td><td>0.0395</td><td>1</td></tr><tr><td>Artistic title</td><td>218</td><td>26.24</td><td>0.8409</td><td>0.0674</td><td>0.0668</td><td>0</td></tr><tr><td>Moving text</td><td>232</td><td>30.73</td><td>0.8800</td><td>0.0209</td><td>0.0325</td><td>4</td></tr><tr><td>Vertical text</td><td>215</td><td>30.46</td><td>0.8736</td><td>0.0975</td><td>0.1015</td><td>3</td></tr><tr><td>Non-bottom layout</td><td>223</td><td>30.92</td><td>0.8843</td><td>0.0414</td><td>0.0628</td><td>2</td></tr><tr><td>Multi-region</td><td>225</td><td>29.88</td><td>0.8834</td><td>0.0103</td><td>0.0167</td><td>2</td></tr><tr><td>Persistent watermark</td><td>218</td><td>36.02</td><td>0.9306</td><td>0.0314</td><td>0.0380</td><td>3</td></tr><tr><td>ASR subtitle</td><td>68</td><td>34.20</td><td>0.9184</td><td>0.0137</td><td>0.0527</td><td>0</td></tr></table>

Evaluated pipelines. The benchmark metadata identifies the 40K checkpoint, GT bounding-box masks, six steps, and a 12-pixel seamless paste-back setting for DeOverlay. ProPainter, MiniMax-Remover, and DifuEraser share GT bounding-box masks and copy source pixels outside an 8-pixel dilated mask, while CLEAR uses oficial mask-free inference. These settings define the evaluated system pipelines, with the dataset GT mask serving as the scoring region for every method. Appendix C specifies the pixel-mask scoring and perceptual crop protocol, while the qualitative exports in Appendix E retain their own run provenance.

Mask provenance is recorded independently of category, distinguishing reconstruction with a known edit region under an oracle mask from evaluation of the associated detection pipeline under an estimated mask. A manually supplied mask is a separate input setting and will not be labeled automatic detection. For watermarks and captions with graphic elements, coverage must include the entire designated overlay rather than only recognized characters.

## E Qualitative Exports and Case Analysis

The application examples in Figure 5 display matched source and DeOverlay output frames at the same frame index, with the source on the left and the output on the right in each pair.

Selection and presentation. Figure 5 uses twelve matched source/output crops from the supplied currentmodel exports, covering Chinese, English, bilingual, and animation subtitles, as well as name captions and dialogue, vertical and large titles, a text watermark, a translucent caption, a promotional sticker, and a television logo. Figures 10 and 11 retain the corresponding full frames and mark the crop coordinates, which are also recorded with frame indices in the figure-source manifest and editable figure notes. Figure 13 retains four additional source/output examples from the earlier collection. The television-logo case uses a logo-only mask that leaves the remaining program graphics outside the requested edit, while the large-title source already contains a transition double exposure visible on both sides. These exports are not a controlled step-count comparison, and their presentation uses frame extraction, equal-coordinate cropping, and display scaling with no color or contrast enhancement. Across the broader export collection, occasional subtitle remnants, local texture artifacts, and residual geometric watermark elements remain.

Paired benchmark context. Figure 7 extends the five main-paper examples to seven overlay categories by adding a persistent watermark and a standard subtitle, with all seven cases containing 81 frames. Figures 8 and 9 show the corresponding full frames for every method, using zero-based frame 40 and the four-step DeOverlay outputs from the same-A100 timing package. CLEAR’s native output is fitted without changing its aspect ratio and centered on the source-sized canvas before applying the common crop. The artistic-title crop focuses on the upper title and car, while the multi-region crop focuses on the upper overlay and leaves the second region visible in the full frame. The small, translucent watermark is enlarged without contrast adjustment. These selected cases illustrate restoration behavior rather than estimating an aggregate ranking or comparing inference depths.

Multilingual generalization. Figure 12 shows Thai, Indonesian, German, Portuguese, Japanese–Chinese, and Vietnamese–Chinese text overlays. The Thai and Indonesian clips share the same scene content, as do the German and Portuguese clips, making diferences in glyph shapes and line layouts visible on matching backgrounds, while the bilingual examples combine two writing systems within the same frame. Each row includes two time points with matched source/output coordinates, showing reconstruction beneath changing subtitle content and across diferent scenes, with background patterns, clothing, printed scene text, and objects remaining visible in the outputs.

![](images/fdc53b1864f382d4da2414174a9f6f487ecb915069a2b7f04a75ad3a77ec86f2.jpg)  
Figure 7 Expanded comparison across seven overlay categories. Every row shows an identical frame and common crop for the source, four baselines, DeOverlay (K = 4), and clean target, with yellow boxes locating the crops in small source frames.

Source

CLEAR

ProPainter

MiniMax-Remover

Artistic title

DiffuEraser

DeOverlay(4 steps)

Ground truth

![](images/519648216487a22ffecee5713106389832e5236620726e7311d091087ca642b5.jpg)  
Figure 8 Benchmark context: titles, moving text, and spatial layouts. Each row uses the same clip and frame for all methods, with yellow boxes marking the source-coordinate crops in Figures 4 and 7.

Source

Persistent watermark

CLEAR

ProPainter

MiniMax-Remover

DiffuEraser

DeOverlay(4 steps)

Ground truth

![](images/10029095e0e563c3d5a52e812189792be2ff023cb92eda99b4c36970c9946845.jpg)  
Figure 9 Benchmark context: watermarks, subtitles, and vertical text. Full frames retain the surrounding scene and all overlaid regions, while yellow boxes locate the identical crops in Figure 7.

![](images/d79074c8beeb5bb1caed3efe45bf7133d6144e85ac3418e91344377f59e39694.jpg)  
(f) Name and dialogue Source DeOverlay

(a) Chinese subtitles Source  
![](images/f9df26efe5d8a97b5f455658a08e38f046a3dc57c9d4434a32117614da5071fa.jpg)

DeOverlay  
![](images/5f278fc966b89cc8921893e5d79132adcfd189757a0577740027ddf57b804dcc.jpg)

(b) English subtitles Source  
![](images/55793a482c6985208187302e406353d6916f89053d0201c845bdb2715146ffe1.jpg)

DeOverlay  
![](images/28f2817319cb49db2e05f6dd9a043f73f02bdcde0c7027c0061aed0655c93965.jpg)  
Frame 180

(c) Bilingual captions Source  
![](images/8fb72edf65e322144d6ad67383c59175834af073922b7707cc9fa7b915582c4f.jpg)  
Frame 180 (d) Animation subtitles Source

DeOverlay  
![](images/b1f30cbe2c97af964c3ae4a7f09df0dfad05131ad7576e8650baf48815fecbe8.jpg)  
Frame 35

DeOverlay  
(e) Name caption Source  
![](images/4a7382c60543b4c6d12219222e1d3b392d6413eaab2a011622157b93ca34b254.jpg)

![](images/0b29a4c224d058a9b14f57558ebe57b7f8e4864c822d947dfbd61017860cc1ea.jpg)  
Frame 140

![](images/599f743a4e65cfef3b20158135d6898502cb3779d713779e511141ef00b709e3.jpg)

![](images/8298f7d35e1b863a2c3ea9a4d329c955135f307b235208880290b1de31a9306e.jpg)  
Frame 180  
Figure 10 Application context for Figure 5(a--f). Adjacent source/output frames preserve native orientation, with yellow boxes locating the matching main-paper crops and zero-based indices identifying the frames.

(g) Vertical title Source

![](images/c15ea885281baf3795b505df7c043acec2b7a0c3e51dad7f4f21e476aa4f1d6b.jpg)

DeOverlay  
![](images/1e2531426912aab56ee936ed538e25b5e2f649ed85009a96a89179dc40645234.jpg)  
Frame 6

(h) Large title Source  
![](images/ec353f82b90375a4458d2e15ec945b22d1b76d90cfa3704560ec433a39b91bbd.jpg)

DeOverlay  
![](images/1bda4598ebdcd1afb398e73cc3615373865efdf445a5e7372bbb19638f4880c7.jpg)  
Frame 90

(i) Text watermark Source  
![](images/a9db6e601b42ef5864c2533bc88181239882eade96a375b0f9a00701b67412dc.jpg)

DeOverlay  
![](images/e326cd9db111eb071db003cc44b1445aa288ad30fa24a46bc4d48e57a62b11e7.jpg)  
Frame 180

(j) Translucent caption Source DeOverlay  
![](images/1cd687ffabe70f73dc0e7338c6b318e9a639f6ba36b25f3146178cc38f0bae95.jpg)

(l) Television logo  
![](images/d36742d994e4c76a2e0ecdf09ea317e0943fa4bca6b9d23fbca6cd3e39c90505.jpg)  
Frame 170

(k) Promotional sticker Source DeOverlay  
![](images/011d544657701541944a386bfc287d6800367aa9622fc00d81a982a4a344afeb.jpg)

![](images/d567bd2dd9551fe40aada26bb057ca58e128f61583e49ab5a44da777bd91dc3c.jpg)  
Frame 180

![](images/73b6a6e187eb854ce4517bc29b4e594000002d529b329ba23d13e0ae3d7bd18e.jpg)

![](images/d399fc35cc65576eff5e6518448a9ebc654f9900549cb6842ea7dd921faf32ef.jpg)  
Frame 180  
Figure 11 Application context for Figure 5(g--l). Full source/output pairs include titles, watermarks, captions, a sticker, and the selected television-logo edit, which retains unrelated program graphics in the logo-only case. Yellow boxes locate the main-paper crops.

![](images/5a0e63c3032cf03c4e3d78f7bcb48e024d1c367213942790c651f96a9ce883b7.jpg)  
Figure 12 Multilingual generalization on application videos. Six language settings include Thai, Indonesian, Japanese–Chinese, Vietnamese–Chinese, German, and Portuguese overlays. Each row contains two matched source/DeOverlay pairs, with timestamps identifying the displayed frames. For these application clips without clean targets, input and output use identical crop coordinates and display scales.

(a) Subtitles  
![](images/10947d1f2abec8a3661816f31513919868dbbc148d0be9a8fdfbb560e62a495e.jpg)  
Figure 13 Additional source/output examples. Paired full frames show subtitles, a text watermark, a name caption, and mixed text overlays from the earlier export collection, with native orientations preserved for these application clips without clean targets.

## F Ablation Details

The component and step-count ablations use 200 randomly sampled videos from DVTE-Bench, while the main comparison in Table 2 evaluates all 1,631 released videos. The step sweep in Table 4 holds the multistep checkpoint fixed and changes the inference update count, whereas Table 3 uses four steps for the component variants and reports a separately trained, matched one-step direct-residual reference. The quality tables retain the reported Mask MAE, Outside MAE, MaskCrop SSIM, and TWE absolute-gap scales.

## F.1 Training Formulation and Initialization

Under the same experimental environment, we independently train a one-step direct-residual baseline to predict the source-to-target residual. In Table 3, four-step flow improves Mask PSNR from 31.300 to 32.461 dB, reduces MAE from 4.520 to 4.299, raises Crop SSIM from 0.91120 to 0.92765, and reduces TWE absolute gap from 0.195 to 0.180. These results compare the two trained models at their respective inference settings, while Table 4 studies update count within the continuous-flow checkpoint.

The Gaussian-start model in Table 3 is trained for a noise-initialized, two-video-stream input, so its 2.861 dB gap to the source-initialized model reflects both the starting state and the way source evidence enters the network. The independent token microbenchmark below evaluates the computational efect of the second stream.

## F.2 Source Conditioning

The full four-step model has Outside MAE of 2.541, rising to 2.552 with both source branches removed, 2.543 without VSA, and 2.550 without GSC. These small increases accompany the covered-region changes in Table 3 and describe exterior behavior within the component study. Keeping both branches but replacing visibility-normalized aggregation with an ordinary temporal mean gives Outside MAE of 2.542 and a TWE absolute gap of 0.183, compared with 0.180 for the full model.

## F.3 Transformer Token Cost

Table 9 One versus two video-token streams. The microbenchmark uses real 32K weights in bfloat16 and a latent tensor of shape $1 \times 1 6 \times 2 3 \times 3 6 \times 6 4$ , corresponding to an 81-frame input with eight mirror-padding frames at the 768s bucket. Both inputs include 98 text tokens, and timing covers four transformer forwards while excluding VAE encoding, decoding, and video processing. Memory is incremental peak allocation above the loaded runtime.
<table><tr><td>Input organization</td><td>Video tokens</td><td>Total tokens</td><td>Time (s)↓</td><td>Memory (GiB)↓</td></tr><tr><td>Source-initialized stream</td><td>13,248</td><td>13,346</td><td>5.266</td><td>1.744</td></tr><tr><td>Noise and source streams</td><td>26,496</td><td>26,594</td><td>13.088</td><td>3.479</td></tr></table>

We warm up each input organization and interleave four timed trials, reversing their order between rounds and bounding each measurement with CUDA synchronization. The table reports medians, with 10th–90th percentile time ranges of 5.260–5.272 s and 13.077–13.099 s, respectively, while incremental memory subtracts the pre-forward allocation from the peak allocation. The machine can carry concurrent GPU load, so these timings characterize the recorded workload rather than isolated end-to-end deployment speed.

## G Timing and Optimization Protocol

Shared input workload. The end-to-end comparison in Table 5 uses eight deterministically selected clips, one per overlay type. Seven clips contain 81 frames at 540 × 960 resolution, and the remaining clip contains 83 frames at $8 7 6 \times 4 9 2$ , yielding 650 input frames. Every method processes these same inputs in serial order on one otherwise idle A100 while retaining its internal resizing and processing operations. MiniMax-Remover uses six steps with maximum side 960, whereas DifuEraser uses maximum side 960, model ofloading, and maximum attention slicing.

Timing boundaries and aggregation. The per-video measurements exclude one-time model loading, while end-to-end time includes each pipeline’s decoding, inference, postprocessing, and video encoding. Seconds per frame is the sum of measured seconds divided by 650, and FPS is its reciprocal. Because the inference-only intervals follow each runner’s CUDA synchronization boundaries, end-to-end time is the primary cross-method comparison. For DeOverlay, the logged inference interval includes the difusion stage and its associated codec operations, rather than transformer forwards alone.

Memory measurement. A common external sampler queries GPU memory every 200 ms across each method’s process, and Table 5 reports the largest sampled value in GiB. The one- and four-step DeOverlay runs share one process and therefore share its recorded peak of 12.236 GiB, which includes the loaded runtime and difers from the incremental forward-allocation measurements in Table 9.

Optimization. The configured optimizer is AdamW with learning rate $5 \times 1 0 ^ { - 5 }$ , betas (0.9, 0.95), zero weight decay, 300 warm-up steps, and gradient clipping at 1.0, using bfloat16 and a maximum of 60,000 updates.

## H Complete Per-Category Comparison

Tables 10 and 11 give the complete five-method comparison for each overlay type under the protocol in Appendix D.

Table 10 Reconstruction across overlay categories (1/2). All five methods are evaluated on every clip in each category under the same pixel-mask scoring protocol, with DeOverlay using the 40K checkpoint and six steps. PSNR is the finite mean, $N _ { \infty }$ counts infinite-valued video scores, and bold marks the best quality score within each category.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td> $N _ { \infty }$ </td></tr><tr><td colspan="6">Standard subtitle (232 clips)</td></tr><tr><td>CLEAR</td><td>17.98</td><td>0.5726</td><td>0.2192</td><td>0.1567</td><td>0</td></tr><tr><td>ProPainter</td><td>25.17</td><td>0.7668</td><td>0.0890</td><td>0.0778</td><td>1</td></tr><tr><td>MiniMax-Remover</td><td>24.92</td><td>0.7752</td><td>0.0838</td><td>0.0755</td><td>0</td></tr><tr><td>DiffuEraser</td><td>24.17</td><td>0.7501</td><td>0.0899</td><td>0.0783</td><td>1</td></tr><tr><td>DeOverlay (K = 6)</td><td>30.39</td><td>0.8718</td><td>0.0285</td><td>0.0395</td><td>1</td></tr><tr><td colspan="6">Artistic title (218 clips)</td></tr><tr><td>CLEAR</td><td>8.67</td><td>0.2878</td><td>0.5548</td><td>0.3276</td><td>0</td></tr><tr><td>ProPainter</td><td>19.20</td><td>0.6639</td><td>0.2766</td><td>0.1707</td><td>0</td></tr><tr><td>MiniMax-Remover</td><td>22.16</td><td>0.7395</td><td>0.1677</td><td>0.1149</td><td>1</td></tr><tr><td>DiffuEraser</td><td>19.73</td><td>0.6903</td><td>0.2078</td><td>0.1373</td><td>2</td></tr><tr><td>DeOverlay (K = 6)</td><td>26.24</td><td>0.8409</td><td>0.0674</td><td>0.0668</td><td>0</td></tr><tr><td colspan="6">Moving text (232 clips)</td></tr><tr><td>CLEAR</td><td>16.28</td><td>0.5194</td><td>0.1890</td><td>0.1451</td><td>1</td></tr><tr><td>ProPainter</td><td>27.61</td><td>0.8158</td><td>0.0577</td><td>0.0569</td><td>4</td></tr><tr><td>MiniMax-Remover</td><td>27.16</td><td>0.8194</td><td>0.0518</td><td>0.0556</td><td>0</td></tr><tr><td>DiffuEraser</td><td>25.97</td><td>0.7890</td><td>0.0566</td><td>0.0592</td><td>3</td></tr><tr><td>DeOverlay (K = 6)</td><td>30.73</td><td>0.8800</td><td>0.0209</td><td>0.0325</td><td>4</td></tr><tr><td colspan="6">Vertical text (215 clips)</td></tr><tr><td>CLEAR</td><td>13.10</td><td>0.3324</td><td>0.5796</td><td>0.3836</td><td>0</td></tr><tr><td>ProPainter</td><td>26.92</td><td>0.7975</td><td>0.1756</td><td>0.1432</td><td>3</td></tr><tr><td>MiniMax-Remover</td><td>27.63</td><td>0.8275</td><td>0.1604</td><td>0.1328</td><td>1</td></tr><tr><td>DiffuEraser</td><td>26.55</td><td>0.8023</td><td>0.1756</td><td>0.1430</td><td>4</td></tr><tr><td>DeOverlay (K = 6)</td><td>30.46</td><td>0.8736</td><td>0.0975</td><td>0.1015</td><td>3</td></tr></table>

Table 11 Reconstruction across overlay categories (2/2). Settings and notation follow Table 10, with the two tables together covering all 1,631 released clips.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td> $N _ { \infty }$ </td></tr><tr><td colspan="6">Non-bottom layout (223 clips)</td></tr><tr><td>CLEAR</td><td>17.76</td><td>0.5941</td><td>0.3264</td><td>0.2282</td><td>0</td></tr><tr><td>ProPainter</td><td>24.84</td><td>0.7627</td><td>0.1462</td><td>0.1269</td><td>2</td></tr><tr><td>MiniMax-Remover</td><td>25.31</td><td>0.7909</td><td>0.1195</td><td>0.1137</td><td>1</td></tr><tr><td>DiffuEraser</td><td>24.22</td><td>0.7582</td><td>0.1356</td><td>0.1229</td><td>2</td></tr><tr><td>DeOverlay (K = 6)</td><td>30.92</td><td>0.8843</td><td>0.0414</td><td>0.0628</td><td>2</td></tr><tr><td colspan="6">Multi-region (225 clips)</td></tr><tr><td>CLEAR</td><td>15.81</td><td>0.5273</td><td>0.1506</td><td>0.1017</td><td>0</td></tr><tr><td>ProPainter</td><td>23.48</td><td>0.7634</td><td>0.0453</td><td>0.0449</td><td>2</td></tr><tr><td>MiniMax-Remover</td><td>24.14</td><td>0.7915</td><td>0.0361</td><td>0.0372</td><td>1</td></tr><tr><td>DiffuEraser</td><td>23.10</td><td>0.7613</td><td>0.0411</td><td>0.0401</td><td>2</td></tr><tr><td>DeOverlay (K = 6)</td><td>29.88</td><td>0.8834</td><td>0.0103</td><td>0.0167</td><td>2</td></tr><tr><td colspan="6">Persistent watermark (218 clips)</td></tr><tr><td>CLEAR</td><td>20.96</td><td>0.7017</td><td>0.2886</td><td>0.1835</td><td>0</td></tr><tr><td>ProPainter</td><td>33.09</td><td>0.8858</td><td>0.0566</td><td>0.0543</td><td>3</td></tr><tr><td>MiniMax-Remover</td><td>30.36</td><td>0.8570</td><td>0.0929</td><td>0.0768</td><td>0</td></tr><tr><td>DiffuEraser</td><td>29.23</td><td>0.8328</td><td>0.1114</td><td>0.0888</td><td>2</td></tr><tr><td>DeOverlay (K = 6)</td><td>36.02</td><td>0.9306</td><td>0.0314</td><td>0.0380</td><td>3</td></tr><tr><td colspan="6">ASR subtitle (68 clips)</td></tr><tr><td>CLEAR</td><td>20.50</td><td>0.7638</td><td>0.1992</td><td>0.2097</td><td>0</td></tr><tr><td>ProPainter</td><td>29.59</td><td>0.8698</td><td>0.0443</td><td>0.0977</td><td>0</td></tr><tr><td>MiniMax-Remover</td><td>31.28</td><td>0.9064</td><td>0.0275</td><td>0.0764</td><td>0</td></tr><tr><td>DiffuEraser</td><td>28.06</td><td>0.8469</td><td>0.0520</td><td>0.1025</td><td>0</td></tr><tr><td>DeOverlay (K = 6)</td><td>34.20</td><td>0.9184</td><td>0.0137</td><td>0.0527</td><td>0</td></tr></table>

## I Temporal Diagnostic Computation

Table 2(b) reports the release’s temporal measurements on all 1,631 videos, with the TWE label following the forward-flow remap convention specified in the release metadata. Using forward displacement $u _ { t - 1  t } ,$ the referenced computation samples the previous frame as

$$
I _ { t - 1 } ^ {  } ( x ) = I _ { t - 1 } ( x + u _ { t - 1  t } ( x ) )\tag{16}
$$

and averages its absolute RGB diference from $I _ { t } ( x )$ on the scoring region. This forward-displacement sampling diagnostic difers from backward warping of the previous frame into the current frame, which samples with $u _ { t  t - 1 }$ . The table retains the released diagnostic names and numerical scales, including the reported labels for TC, Flow Mean, and Flow Var.

The aggregate gap applies the absolute value after video-level averaging, $| \overline { { \mathrm { T W E } } } _ { \mathrm { p r e d } } - \overline { { \mathrm { T W E } } } _ { \mathrm { G T } } |$ , with cleantarget reference 9.3228, so opposite signed deviations can cancel. Table 2 also reports the target-referenced reconstruction comparison.