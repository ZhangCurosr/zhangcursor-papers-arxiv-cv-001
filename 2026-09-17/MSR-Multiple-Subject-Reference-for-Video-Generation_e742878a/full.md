# MSR: Multiple Subject Reference for Video Generation

Guannan Li Jiaji Chen Jingyuan Liao<sup>∗</sup> Yu Geng<sup>∗</sup> Baolan Qiu<sup>∗</sup> Licon Studio lign2@hotmail.com

September 2026

## Abstract

Conditioning a video generator on multiple images requires preserving appearance while associating each reference with its intended role. We present MSR (Multiple Subject Reference), a slot-aware conditioning scheme for LTX-based video generation. Each reference image is independently encoded as a static clip and represented by a separate latent-token group. A compact Fourier-feature multilayer perceptron adds a numeric slot embedding, while slot-dependent temporal ofsets modify the group’s rotary coordinates. The reference groups are prepended to noisy target tokens and serve as clean context during target-only flow-matching training. We implement this scheme through low-rank adaptation and release the resulting weights and inference workflows. Qualitative examples demonstrate compositions containing distinct characters and referenced environments in realistic and stylized scenes. Development observations suggest reduced reference confusion relative to an earlier continuous-reference baseline, while similar clothing, complex garments, and viewpoint changes remain challenging. We describe the conditioning mechanism, the retained training configuration, and the observed strengths and limitations of the released system. A supplementary audio-reference experiment adds voice conditioning while keeping the visual parameters frozen.

Model and workflows · Interactive demonstration

## 1 Introduction

Reference-conditioned video generation is useful when a scene must contain specified visual elements rather than arbitrary instances of a text description. In a two-character interaction, for example, a prompt may need to preserve each character’s face, hairstyle, and clothing, place both in a referenced environment, and maintain the assignment of attributes as the camera changes viewpoint. Producing a plausible frame is insuficient if one character acquires the other’s jacket or the referenced object changes ownership.

The problem is not limited to facial identity. A reference may specify a complete character, a garment, a prop, or a setting. Several views can also be arranged within a single reference image. Such an image still constitutes one input source; its constituent views are not automatically separate subjects. A useful interface should preserve the association between an input image and the role assigned to it in the prompt, while allowing the generator to create new poses, compositions, and interactions.

An early version of our system assembled reference images into a continuous pseudo-video and concatenated the resulting reference representation with the generation context. This provides a simple route for reusing a video model, but it does not explicitly identify independent reference sources. In particular, adjacent frames of the pseudo-video can depict unrelated subjects even though a video encoder is designed to process temporally related content. This observation motivates separating image encoding from the organization of the reference context.

Multiple Subject Reference (MSR) represents each input image as an independently encoded token group. It supplies two complementary source cues: a learned numeric slot embedding in latent-token space, and a deterministic shift of the group’s temporal positional coordinates. The former changes the reference features; the latter changes their positional relationships inside the existing transformer. Ordinary text conditioning specifies the roles and actions associated with the input images. We adapt the LTX model family [4] using LoRA [5] and a small slot module, without adding a separate face-recognition or multimodal-language-model conditioning branch.

This paper makes three contributions. First, it specifies an implementable slot-aware conditioning scheme, including the exact slot representation, token organization, and positional convention. Second, it documents the corresponding visual training configuration and provides publicly accessible weights and workflows. Third, it analyzes qualitative behavior across multi-reference cases, development comparisons, and observed failure modes

The main study concerns visual multi-subject conditioning. A subsequent audio-reference extension freezes the visual parameters and trains audio-related modules. We describe it in the appendix to separate the visual method from additional audio capabilities.

## 2 Related Work

Video foundations and eficient adaptation. LTX-2 provides a joint audio–video generation architecture with interacting visual and audio streams [4]. Our implementation adapts a pretrained backbone from the LTX model family. LoRA introduces low-rank updates to pretrained weight matrices [5], and flow matching provides the general velocity-regression formulation used for generation [7]. We retain these foundations and focus on representing multiple reference sources within the visual conditioning sequence.

Subject-conditioned video generation. Ingredients combines facial extraction, projection, and identity routing to condition video difusion transformers on multiple human identities [3]. MAGREF addresses heterogeneous references through composite reference layouts, masked guidance, and channel-wise conditioning; its subject disentanglement formulation also associates semantic subject information with visual regions [2]. BindWeave uses multimodal language-model hidden states to connect reference subjects with prompt semantics before conditioning a video generator [6]. MSR instead exposes independently encoded visual token groups to the existing transformer and retains its text-conditioning pathway.

Reference identity and positional conditioning. Rotary position embeddings introduce position-dependent relationships in attention [8]. As a separate related method, ID-LoRA uses negative reference temporal positions for LTX-based audiovisual personalization [1]. Aura combines reference-token concatenation, learned category information, rotary-coordinate shifts, and a VLM-based conditioning pathway [9]. MSR shares the motivation of distinguishing reference sources, but uses a numeric image-slot MLP and a selected-slot temporal ofset within an LTX adaptation. Our contribution is a concrete slot-indexed realization of these conditioning principles for parameter-eficient LTX adaptation.

## 3 Method

## 3.1 Task and reference representation

Let � be a text prompt and $\{ I _ { s _ { i } } \} _ { i = 1 } ^ { K }$ a set of supplied images ordered by their slot identifiers $s _ { i }$ . The task is to generate a video whose content follows � and preserves the relevant appearance and role of each reference. A slot is an input-source identifier, not a semantic class or a guarantee of one entity per image. The interface supports human and stylized characters as well as clothing, objects, and scenes, with the intended use specified in text.

For each image, we form a static clip by repeating only that image for $F _ { r }$ frames and apply the backbone’s video VAE encoder � independently:

$$
R _ { i } = P \big ( E \big ( \mathrm { r e p e a t } ( I _ { s _ { i } } , F _ { r } ) \big ) \big ) \in \mathbb { R } ^ { N _ { i } \times d _ { r } } ,\tag{1}
$$

where � denotes latent patchification and $N _ { i }$ is the number of tokens for that reference. Diferent images never share an encoder input clip. A multi-view collage already contained in one image remains a single reference. The archived training preprocessing uses $F _ { r } = 2 5 $ at 25 frames per second. The reference token dimension for the released slot module is $d _ { r } = 1 2 8$

Independent encoding separates the VAE inputs of unrelated references. It does not prevent interactions between them after token concatenation, and it does not impose an attention mask. That downstream interaction is necessary for composing a scene from several input sources.

## 3.2 Numeric slot embeddings

Each reference group receives an embedding determined by its positive integer image-slot identifier. For $u _ { s } = s / 1 6$ and $\omega _ { j } = 2 ^ { 4 j / 1 5 } , j \in \{ 0 , \ldots , 1 5 \}$ , we define

$$
\phi ( s ) = \big [ u _ { s } , \{ \sin ( u _ { s } \omega _ { j } ) \} _ { j = 0 } ^ { 1 5 } , \{ \cos ( u _ { s } \omega _ { j } ) \} _ { j = 0 } ^ { 1 5 } \big ] \in \mathbb { R } ^ { 3 3 } ,\tag{2}
$$

$$
\begin{array} { r } { e _ { s } = W _ { 2 } \operatorname { S i L U } ( W _ { 1 } \phi ( s ) + b _ { 1 } ) + b _ { 2 } \in \mathbb { R } ^ { 1 2 8 } , } \end{array}\tag{3}
$$

$$
\widehat { R } _ { i } = R _ { i } + \mathbf 1 _ { N _ { i } } e _ { s _ { i } } ^ { \mathsf T } .\tag{4}
$$

The MLP has dimensions 33→256→128, with biases in both layers and no normalization layer. It contains 41,600 learned parameters; its 16 frequency values are fixed bufers. The embedding is added before the transformer’s input projection, so every token in a reference group carries the same source tag without changing its sequence length. Target tokens do not receive a reference-slot embedding.

The module encodes an index rather than a learned category such as “person” or “background.” The text prompt supplies these roles. Although the MLP can be evaluated for integer IDs beyond those encountered in training, this algebraic property is not evidence of reliable generation with an unlimited number of references.

## 3.3 Slot-dependent temporal ofsets

The video transformer also receives spatial–temporal coordinates used by its positional encoding. Let $\bar { \tau } _ { i , n }$ be a temporal coordinate for reference token � after the implementation’s reference-to-target alignment, and let $f _ { \nu }$ denote the target frame rate. For � selected references in natural slot order, we use

$$
\Delta _ { i } = - \frac { K - i + 1 } { f _ { \nu } } , \qquad \tau _ { i , n } ^ { \prime } = \bar { \tau } _ { i , n } + \Delta _ { i } , \quad i = 1 , \ldots , K .\tag{5}
$$

The shift is applied to both temporal endpoints when a token is represented by a coordinate interval. Spatial coordinates retain the reference-to-target scale adjustment. The embedding in equation (3) uses the original slot ID $s _ { i }$ , whereas equation (5) uses the compact order � among selected references. They agree for consecutively numbered inputs.

These are negative ofsets, not disjoint negative temporal windows. For three references at 25 fps, the ofsets are −0.12, −0.08, and −0.04 seconds; a reference with nonzero temporal extent can still have positive coordinates after translation. The shift distinguishes positional context but does not constrain an attention matrix or guarantee identity separation. Positional coordinates also difer from difusion timesteps: the former indicate where a token lies in the model’s coordinate system, while the latter indicate its noise level.

![](images/234424679143259aa5577ed936c097bece37b400d50bf7fd8f4fa63f7a1f8f82.jpg)  
Figure 1: MSR visual conditioning. Each reference is encoded separately, tagged in latent-token space, and assigned a slot-dependent positional ofset. Clean reference tokens precede noisy target tokens. Only the target sufix contributes to the training loss. The diagram abstracts the existing transformer and its text-conditioning path.

## 3.4 Clean context and target-only training

Given target video tokens $z _ { 0 } \in \mathbb { R } ^ { N _ { \nu } \times d _ { r } }$ and Gaussian noise �, we sample a noise level $\sigma$ and interpolate

$$
z _ { \sigma } = ( 1 - \sigma ) z _ { 0 } + \sigma \epsilon , \qquad \nu ^ { \star } = \epsilon - z _ { 0 } .\tag{6}
$$

The transformer receives the concatenated sequence

$$
X _ { \sigma } = [ \widehat { R } _ { 1 } ; . . . . ; \widehat { R } _ { K } ; z _ { \sigma } ] .\tag{7}
$$

Reference tokens remain clean and are assigned difusion timestep zero; target tokens receive timestep $\sigma .$ Conditioning includes the text representation and the spatial–temporal coordinates described above. We retain the predicted target sufix and minimize the velocity-regression loss

$$
\mathcal { L } = \mathbb { E } _ { z _ { 0 } , \epsilon , \sigma } \left[ \frac { 1 } { N _ { \nu } d _ { r } } \left\| \nu _ { \theta } ( X _ { \sigma } , p ) _ { \mathrm { t a r g e t } } - ( \epsilon - z _ { 0 } ) \right\| _ { F } ^ { 2 } \right] .\tag{8}
$$

This is the all-target-token case of the implementation’s normalized masked loss, following the flow-matching formulation [7]. Reference tokens provide conditioning but no reconstruction-loss terms.

Training updates LoRA adapters on the visual self-attention and text cross-attention query, key, value, and output projections, as well as the visual feed-forward input and output projections. The reference-slot MLP is trainable. The underlying backbone weights remain frozen. At inference, the reference context and slot correspondence are constructed before denoising; a normal text prompt identifies the input images and the desired scene.

## 3.5 Relationship to the earlier baseline

Table 1 distinguishes the earlier continuous-reference representation from the released visual MSR. The changes form a combined method comparison: independent image encoding, slot embeddings, and positional ofsets change together. Consequently, a favorable version-level observation cannot isolate the gain from any single component.

## 4 Experiments and Qualitative Analysis

## 4.1 Training data and implementation

The documented visual training stage uses examples with English captions, a target video, and numbered reference images. The training mixture comprises approximately 28.91% from MSR\_batch2\_V2, 36.61% from MSR\_training\_data\_V6, and 34.47% from short\_drama\_V2; MSR\_batch4\_V1 is excluded from this stage. Each caption describes the supplied sources and the requested scene or actions. The records cover diferent reference counts and visual roles. Table 2 reports the percentage of examples at each nonempty reference-image slot count in the metadata entries matched to this manifest.

<table><tr><td>Property</td><td>Earlier MSR V1/V2</td><td>Released visual MSR</td></tr><tr><td>Training backbone</td><td>LTX</td><td>LTX</td></tr><tr><td>Reference encoding</td><td>Continuous pseudo-video of input images</td><td>Independent static clip per image</td></tr><tr><td>Learned image-slot MLP</td><td>Absent</td><td>Present</td></tr><tr><td>Slot-dependent negative offset</td><td>Absent</td><td>Present</td></tr></table>

Table 1: Version definitions. Both variants use the same LTX training backbone. Comparisons in development used the same inference backbone; this table describes the combined version changes, not isolated ablations.
<table><tr><td>Reference images per example</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Share of examples (%)</td><td>20.35</td><td>25.99</td><td>29.53</td><td>23.89</td><td>0.12</td><td>0.08</td><td>0.04</td></tr></table>

Table 2: Distribution of reference-image counts in the visual-stage manifest, expressed as percentages of examples. Reference counts refer to image slots, not the number of subjects or views inside an image. Percentages are rounded to two decimal places and may not sum to 100%. Unique file paths do not establish content-level deduplication.

Reference preprocessing repeats an image for 25 frames at 25 fps and independently encodes it using the video VAE. The archived scripts provide landscape, portrait, and square resolution buckets. Target videos are processed at their source frame rates and trimmed to a supported 8� + 1 frame count without temporal padding. The documented preprocessing job and metadata schema use up to eight reference-image fields, while the released visual interface exposes up to five input images. The few training examples with six to eight images do not constitute an evaluation of generation quality at those counts.

Table 3 summarizes the retained configuration for the visual stage. The public visual adapter is an intermediate checkpoint from that stage. Its exact step number was not retained locally, so neither the later saved 13,000-step checkpoint nor the 20,000-step configuration ceiling is used as the release’s training duration. The retained configuration includes a subsequent resume; the table documents that stage rather than reconstructing an exact schedule for the released intermediate file.

## 4.2 Evaluation scope

Our evaluation is qualitative. It consists of public reference–video pairs and observations made during method development. We show cases explicitly associated with the released visual adapter in its model card, rather than infer model identities from unrelated local filenames. For each displayed public clip, we sample three frames at 15%, 50%, and 85% of its 12.04-second duration. Selection is illustrative, not random, and three frames cannot establish consistency at every instant of a video.

The earlier and newer adapters were compared on the same inference backbone during development. However, a complete archive of paired runs, random seeds, and per-trial judgments was not retained. We therefore summarize these observations without a numerical success rate, statistical significance claim, or quantitative ranking against other methods. An additional local demonstration collection contains selected complete cases, with multiple outputs for some inputs; it is not an exhaustive test set. We do not use that collection to infer a population success rate.

<table><tr><td>Setting</td><td>Retained visual-stage configuration</td></tr><tr><td>Backbone</td><td>LTX (22B)</td></tr><tr><td>Text encoder</td><td>Gemma 3 12B IT</td></tr><tr><td>LoRA rank / scaling α / dropout</td><td>128 / 128 / 0</td></tr><tr><td>Slot MLP</td><td>33 → 256 → 128, SiLU, 41,600 parameters</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW / 10−4</td></tr><tr><td>Schedule / precision</td><td>Cosine / BF16</td></tr><tr><td>Parallelism / batch</td><td>8 devices; 1 example per device; accumulation 1</td></tr><tr><td>Gradient clipping / checkpointing</td><td>Norm 1.0 / enabled</td></tr><tr><td>Sampling</td><td>No bucket-aware sampling</td></tr><tr><td>Noise sampling</td><td>Two-region shifted-logit-normal mixture (Appendix A)</td></tr><tr><td>Configured seed</td><td>42</td></tr></table>

Table 3: Settings recorded for the stage that produced the released visual adapter. The exact public checkpoint step and a full historical schedule are not reconstructed from a later resume configuration.

## 4.3 Composing distinct character and scene references

Figure 2 shows a public example with two separate character references and a nightclub reference. Across the sampled frames, the dark jacket of one character and the silver outfit and transparent outer garment of the other remain distinguishable, while the composition changes from a wider view to closer interaction. The nightclub lighting and booth environment are also reflected in the generated frames. This illustrates the intended task: several referenced elements participate in a newly composed interaction rather than appearing as independent still images.

Figure 3 uses a diferent visual style: two stylized characters and a forest setting. The sampled output includes a wide environment view and closer two-character views. Hair color, garment color, and distinctive head features provide visible cues for separating the two references. Together, the two cases demonstrate that the interface can condition on more than facial portraits: each character image includes overall appearance, and the third image supplies an environment.

Scene layout, pose, perspective, and illumination are recomposed by the generator. The figures illustrate reference-conditioned composition rather than exact replication of the input images.

## 4.4 Development observations and failure modes

In development comparisons against the earlier continuous-reference V1/V2 adapters, we observed improved visual coherence, fewer clothing-color confusions, and stronger facial consistency in the tested cases. Since the versions change several design elements together, these observations concern the combined representation and do not identify the contribution of independent encoding, the MLP, or the temporal ofsets individually. All compared variants were trained on the same LTX backbone, so the version comparison does not involve a change of training backbone.

The same testing revealed recurring limitations, summarized in table 4. Simple clothing, uncomplicated scene references, and objects were easier to maintain in the tested examples. Visually similar characters and garments remained more dificult. For example, two same-gender characters wearing similarly shaped short jackets could exhibit appearance blending or clothing exchange, even when jacket colors difered. A gray-jacket character could become inconsistent after turning or after a cut away and back. Elaborate dresses with overlapping sleeve and skirt structures were another observed failure case. These are development observations; their frequencies were not logged.

The failures show why a slot tag should not be interpreted as a hard identity constraint. It helps identify the source of conditioning tokens, but the transformer still has to associate that source with the correct generated region and preserve fine structure during motion. Source indexing and successful visual binding are distinct

Reference 1  
![](images/9a379d8715916cf7f23e830541d2de477ad33037a50bd9c9b38ca4c06099946d.jpg)

Reference 2  
![](images/926c9795729a6cdf0beceb7649cefa0d516ce17edf4599055d80f81aaf339a4c.jpg)

Reference 3  
![](images/b4dbce8806f9d02815acc4a0b83179ff63cf0c0549d3e830d3a91195d9004353.jpg)

Generated: 1.83 s  
![](images/cea7a5712aef867189adc8c72b010575d3542d9d52816684abd22edac3349441.jpg)

Generated: 6.04 s  
![](images/09a3f169a4ccd989db3196a88d959b8ec4c2962fd7583cb0d8f0d23b7ffcdcf4.jpg)

![](images/f6c203ffe517ac4bff34cdcdb061e8c076ef589a5f137e231df9f99c1d7c6905.jpg)  
Figure 2: Public visual MSR case 03. Top: two character references and a scene reference. Bottom: full generated frames sampled at approximately 1.83, 6.04, and 10.25 seconds. The images show distinguishable appearance cues and the referenced setting across changes in framing. Frames are not cropped; the composite views in each character reference were already present in the input image.

Reference 1  
Reference 2  
Reference 3  
![](images/888e8bc250d77c0ce7f435086c6accd3674a2ca6d5fe407b4db5a51f7eee53e5.jpg)  
Figure 3: Public visual MSR case 06. Top: two stylized-character references and a forest reference. Bottom: the same relative sampling times as in figure 2. The sampled views show the contrast between the orange-haired character in a light dress and the silver-haired character in a green outfit while changing camera distance.

<table><tr><td>Condition</td><td>Observed behavior</td></tr><tr><td>Simple clothing, props, scenes</td><td>Reference appearance was often maintained in tested examples.</td></tr><tr><td>clothing</td><td>Similar character appearance or Subject blending and clothing exchange could occur.</td></tr><tr><td>Turning or returning after a cut structures</td><td>Appearance could become inconsistent after the viewpoint change Complex overlapping garment Sleeve and skirt structures could merge or be misinterpreted.</td></tr></table>

Table 4: Qualitative development observations. No event counts or failure probabilities were retained, so the rows do not imply measured rates.

requirements.

## 4.5 Supplementary audio-reference experiment

We also extend MSR with audio-reference conditioning as a supplementary experiment. The additional stage freezes the visual parameters, including the visual slot MLP, and trains audio-related adapters and an audio-slot module. This adds voice-reference conditioning to the visual multi-subject system without updating its visual weights. In exploratory tests, explicit text assignments associated two speakers with two reference voices. The tests also revealed voice-characteristic mixing, inherited reference noise, and artifacts during emotional speech No obvious visual change was observed in comparisons using the same images, prompts, and inference backbone, although random seeds were not matched. These observations are qualitative and do not establish speaker-similarity scores or strict visual equivalence. Appendix C details the audio conditioning, training setup, and observed limitations.

## 5 Discussion and Limitations

What the representation provides. Independent encoding makes reference boundaries explicit before transformer processing. The numeric MLP contributes a shared source signal to each group, and the temporal ofset changes its positional context. The cues operate at feature and positional levels. They identify conditioning sources but do not prescribe the spatial regions occupied by generated subjects; assigning reference attributes to those regions remains a learned part of generation.

Evaluation and reproducibility. The present qualitative study lacks a retained seed-matched benchmark, isolated retraining ablations, a formal user study, and a quantitative comparison with other systems. The development corpus is not presented as a benchmark with a documented identity-disjoint test split. The released intermediate checkpoint also lacks a recoverable exact training-step index. These limitations restrict both statistical conclusions and exact reproduction of its training history. A future evaluation should separately measure appearance fidelity, role assignment, temporal behavior, and scene composition on a fixed set of inputs, while preserving outputs and run configurations.

Scope of reference counts and prompting. The public workflow accepts up to five reference images; general ization to larger numbers has not been established. Indexing also does not eliminate the need to describe intended roles. In practice, prompts identify each image, establish the scene and actor relationships, and describe actions and camera changes in order. These are useful interface conventions rather than a claim that one exact prompt syntax is necessary. The model can still misinterpret a complex garment or confuse similar-looking subjects even when the references are explicitly named.

Responsible use. Reference-conditioned synthesis can be used to depict recognizable people or imitate distinctive creative assets. Users should have appropriate permission to use and distribute the supplied references and should identify generated material when presenting it as such. The model’s ability to reproduce appearance does not establish consent, ownership, or factual authenticity. Audio-reference use additionally requires care with speaker consent. This paper makes no claim that the training assets are unrestricted for redistribution.

## 6 Conclusion

We presented MSR, a slot-aware conditioning scheme for multi-subject reference video generation. Independently encoded reference groups receive numeric slot embeddings and temporal-coordinate ofsets, then provide clean context for target-only flow-matching training. Qualitative examples show the composition of distinct characters and referenced environments in realistic and stylized videos. Development observations suggest reduced reference confusion relative to an earlier continuous-reference design, with similar clothing, complex garments, and viewpoint changes remaining challenging. A supplementary audio-reference extension adds voice conditioning through audio-related training with frozen visual parameters. The released adapter and workflows provide an implementation for further evaluation of reference fidelity, role assignment, and temporal consistency.

Availability. The visual adapter and ComfyUI workflows are available at the public MSR model repository. The earlier adapters are available at the earlier MSR model repository. A hosted inference interface is available through the hosted MSR demonstration. The adapters described here are based on LTX.

## References

[1] Aviad Dahan, Moran Yanuka, Noa Kraicer, Lior Wolf, and Raja Giryes. ID-LoRA: Identity-driven audio-video personalization with in-context LoRA, 2026. URL https://arxiv.org/abs/2603.10256.

[2] Yufan Deng, Yuanyang Yin, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Shenghai Yuan, Yiding Yang, Angtian Wang, Bo Liu, Haibin Huang, and Chongyang Ma. MAGREF: Masked guidance for any-reference video generation with subject disentanglement, 2025. URL https://arxiv.org/abs/2505.23742.

[3] Zhengcong Fei, Debang Li, Di Qiu, Changqian Yu, and Mingyuan Fan. Ingredients: Blending custom photos with video difusion transformers, 2025. URL https://arxiv.org/abs/2501.01790.

[4] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. LTX-2: Eficient joint audio-visual foundation model, 2026. URL https://arxiv.org/abs/2601.03233.

[5] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.

[6] Zhaoyang Li, Dongjun Qian, Kai Su, Qishuai Diao, Xiangyang Xia, Chang Liu, Wenfei Yang, Tianzhu Zhang, and Zehuan Yuan. BindWeave: Subject-consistent video generation via cross-modal integration, 2025. URL https://arxiv.org/abs/2510.00438.

[7] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2022. URL https://arxiv.org/abs/2210.02747.

[8] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding, 2021. URL https://arxiv.org/abs/2104.09864.

[9] Zixiang Zhou, Zhentao Yu, Yifeng Ma, Hongmei Wang, Wenqing Yu, Cong Wang, Zilin Yang, Rui Chen, Jiarong Ou, Yezhou Liu, Yuan Zhou, and Qinglin Lu. Aura: Consistent multi-subject video generation via VLM-grounded semantic alignment, 2026. URL https://arxiv.org/abs/2607.04311.

## A Implementation and Training Details

Reference preprocessing. The archived image-preprocessing script specifies buckets of $1 2 8 0 \times 7 0 4 , 7 0 4 \times 1 2 8 0$ 704 × 704, and 1280 × 1280, each with 25 repeated image frames. It uses centered resizing/cropping, VAE tiling, and separate latent directories for numbered image slots. The target-video path preserves source frame rates and selects supported 8� + 1 temporal lengths. The released example workflow and hosted interface default to 33 repeated frames per reference, whereas the documented training preprocessing uses 25. The exact settings of each published demonstration were not retained in its paired asset files. These are preprocessing settings, not a claim that every generated example uses the same output resolution or duration. Some source images are multi-view sheets; a sheet occupies one slot and is not split into multiple identities by the method.

Temporal alignment. For the archived training implementation, a reference coordinate $\tau _ { i , n } ^ { \mathrm { r e f } }$ is formed from the causal VAE grid at the stored reference frame rate. Before applying equation (5), the implementation uses

$$
\bar { \tau } _ { i , n } = \operatorname* { m a x } \left\{ \tau _ { i , n } ^ { \mathrm { r e f } } - \frac { S _ { i } - 1 } { f _ { \nu } } , 0 \right\} ,\tag{9}
$$

when $S _ { i } > 1$ , where $S _ { i }$ is the positive rounded ratio between target and reference latent temporal-group counts after excluding the first causal group. For $S _ { i } = 1$ , the nonnegative reference coordinates are unchanged. This operation applies to both interval endpoints. Generic inference helpers additionally account for temporal scaling while constructing the grid. Thus equation (5) describes the shared slot translation applied to the pre-aligned grid, rather than asserting identical grid construction for every training and inference resize configuration.

Noise sampling. The retained visual-stage configuration uses a two-region sampler. Let $U \in [ 0 , 1 ]$ be drawn from the implementation’s length-aware, stretched and clipped shifted-logit-normal sampler with a 10% uniform mixture, standard deviation 1, and endpoint parameter $\varepsilon = 0 . 0 0 1$ . It maps the draw into a low- or high-noise region as

$$
\sigma = \left\{ \begin{array} { l l } { 0 . 4 5 U , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 6 , } \\ { 0 . 4 5 + 0 . 5 5 U , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 4 . } \end{array} \right.\tag{10}
$$

The sampler changes the distribution of training noise levels; it does not introduce an additional per-token loss weight. We record it as an implementation setting and do not claim a separately measured gain from this choice.

Checkpoint and release correspondence. The public visual weight is an intermediate checkpoint of the documented visual stage. Its original local file was removed after release. The later retained checkpoint and resume configuration are useful training records, but they are not substitutes for the exact release history. The public weight contains five tensors for the slot module, including its fixed frequency bufer. The two earlier adapters do not contain this learned slot module. All versions were trained using the same LTX backbone.

## B Public Examples and Prompt Organization

Figures 2, 3, and 4 use cases 03, 06, and 07 from the visual model repository’s validition\_V1 directory. The assets were retrieved at revision 9a053e6d63e54cd6970b1cda9419f1339457bd8c. Reference images are shown as supplied; generated frames are sampled without cropping by selecting the first source frame at or after 15%, 50%, and 85% of each 12.04-second clip. The nominal sampling times are 1.806, 6.020, and 10.234 seconds; the selected source frames occur at 1.833333, 6.041667, and 10.250000 seconds. The original videos and prompts provide context beyond the sampled frames.

Figure 4 shows another public case with two character references and an interior scene. The sampled frames alternate the visible focal character. The clothing colors and the red headband remain recognizable visual cues. Subtitles are already present in the source video and are retained in the figure; we do not use their appearance as an evaluation of speech, transcription, or audio-reference conditioning.

A useful prompt identifies each image and its role, establishes the setting and initial actor positions, and describes actions, object ownership, and camera changes in sequence. Image numbers refer to input sources rather than identity-recognition outputs. We used this organization during development, but did not establish that every clause is necessary through independently retained deletion experiments.

![](images/118dbdab4b42fe4790b7c9ec3dcab7364153075c5bf8e6b326d234a16bf98fda.jpg)  
Figure 4: Additional public visual MSR case 07. Top: reference images. Bottom: three full frames from the published clip. Framing changes emphasize diferent characters, while several salient appearance cues remain visible. The subtitles are part of the supplied output. The figure does not evaluate the later audio-reference extension.

## C Supplementary Audio-Reference Extension

The subsequent audio-reference stage is a separate extension of the visual system. It freezes visual parameters, including the visual slot MLP, and trains audio self-attention, audio text cross-attention, audio feed-forward adapters, video-to-audio attention adapters, and an audio-slot module. Video is conditioning-only during this training stage; the audio stream is generated and supplies the optimization loss. The recorded audio-stage training mixture comprises approximately 64.56% from short\_drama\_audio and 35.44% from audio\_reference\_v2. This distribution is reported separately from the visual-stage mixture.

Audio references use a separate temporal arrangement. The configuration allocates five-second audio slots with an end margin of 0.04 seconds and truncates overlong references. These audio windows should not be confused with the small visual ofsets in equation (5). Joint appearance and voice personalization is also studied by ID-LoRA [1]; our main contribution and qualitative figures concern the visual reference scheme.

In exploratory tests, two speakers were associated with two reference voices using explicit text assignments. With a missing voice reference, the model generated audio from the remaining context. Observed issues included mixtures of reference and model-generated voice characteristics, inherited reference noise, and synthetic or intermittent electronic-sounding artifacts during emotional speech. Cross-language observations were not suficiently retained to support a specific conclusion, and generalization to unseen voices was not evaluated in the recorded tests. We therefore do not report a speaker-similarity or lip-synchronization score.

Comparisons with the visual version used the same images, prompts, and inference backbone but did not fix identical random seeds. No obvious visual change was observed in those tests. Freezing visual weights does not mathematically imply identical generated videos, because audiovisual inference can still couple the streams. The observation is consequently not a claim of strict visual equivalence or guaranteed absence of degradation.