# Direct Conditioning of Audio Diffusion Transformers on fMRI Reveals Cortical Contributions to Sound Reconstruction

Matteo Ciferri Department of Biomedicine and Prevention University of Rome Tor Vergata, Italy matteo.ciferri@students.uniroma2.eu

Matteo Ferrante Tether Evo, El Salvador Department of Biomedicine and Prevention University of Rome Tor Vergata, Italy

Tonio Weidler Maastricht Centre for Systems Biology (MaCSBio) Faculty of Science and Engineering Maastricht University, The Netherlands

Nicola Toschi<sup>∗</sup>   
Department of Biomedicine   
and Prevention   
University of Rome Tor Vergata, Italy   
Athinoula A. Martinos Center   
for Biomedical Imaging   
Massachusetts General Hospital   
and Harvard Medical School, USA Elia Formisano<sup>∗</sup> Maastricht Brain Imaging Center   
Department of Cognitive Neuroscience   
Faculty of Psychology and Neuroscience   
Maastricht University, The Netherlands

## Abstract

Reconstructing audio from brain activity remains challenging. Existing brain-toaudio pipelines typically map neural responses to predefined stimulus representations before synthesis. While recent work in the visual domain has explored direct conditioning on brain activity, this approach has not yet been extended to audio diffusion models. We introduce MAAS (Multi-Area Auditory Synthesis), an end-to-end framework that conditions a pretrained audio diffusion transformer on ROI-wise fMRI responses without first supervising a decoder to predict an external stimulus representation. MAAS uses region-specific projections and learned brain tokens to condition Stable Audio through a dedicated cross-attention pathway. On a 7T fMRI dataset, we demonstrate consistent improvements over baselines, including two-stage fMRI-to-latent conditioning, across both acoustic and semantic evaluation metrics. In addition, we introduce a fidelity metric grounded in a computational model of auditory neural processing, which allows us to quantify the contribution of individual auditory cortical regions to reconstructed audio representations.

## 1 Introduction

Reconstructing audio from functional Magnetic Resonance Imaging (fMRI) measurements of brain activity offers a unique window into the representational structure of auditory cortex, yet remains tech nically challenging. While recent advances in the visual domain have demonstrated that photorealistic images can be synthesized from fMRI [Ferrante et al., 2024a, Ozcelik and VanRullen, 2023, Luo et al., 2023, Chen et al., 2023], progress in the auditory domain has been comparatively limited. This gap reflects several compounding factors: the hemodynamic response introduces temporal blurring on the order of seconds [Kay et al., 2013], auditory datasets with sufficient diversity remain scarce, and sound reconstruction imposes strict spectrotemporal constraints.

Audio generative models impose a restrictive conditioning interface, typically accepting semantic embeddings (e.g., text or audio-language representations). Recent work tried to extend text-conditioned audio generation with additional, non-text conditioning signals. AP-Adapter [Tsai et al., 2024] augments AudioLDM2 [Liu et al., 2024] with a parallel cross-attention pathway, allowing the model to use acoustic information alongside the text prompt for sound editing. MuseControlLite [Tsai et al., 2025] applies decoupled cross-attention to Stable Audio [Evans et al., 2025] to incorporate time-varying musical attributes, such as melody, rhythm, and dynamics, as well as reference-audio signals, enabling temporally controlled generation and audio inpainting or outpainting.

Existing brain-to-audio pipelines address the conditioning mismatch through a two-stage strategy, mapping neural activity onto intermediate deep neural network (DNN) embeddings, such as CLAP [Elizalde et al., 2023] or YamNet [Hershey et al., 2017], which are then used to guide a generative model. While effective, this approach introduces a key limitation: it constitutes an information bottleneck [Shirakawa et al., 2025, Feng et al., 2026], and reliance on intermediate embeddings may discard fine-grained spectrotemporal structure. Recent work in decoding of visual stimuli from brain activity has begun to explore an alternative paradigm based on direct conditioning, where generative models are conditioned on brain representations without intermediate embedding prediction [Feng et al., 2026]. Related ideas have also been explored in language domain, where neural signals are directly integrated into autoregressive generation [Ye et al., 2025]. However, these approaches are tailored to specific architectures and do not transfer directly to current audio diffusion models.

Here, we introduce MAAS (Multi-Area Auditory Synthesis), an end-to-end framework for conditioning a pretrained audio diffusion transformer on ROI-wise fMRI responses. Central to our approach is a transformer-compatible adapter that injects neural representations into the cross-attention layers of a diffusion transformer (DiT), while remaining compatible with pretrained models such as Stable Audio [Evans et al., 2025]. Importantly, MAAS preserves anatomical organization at the region-of-interest (ROI) level by conditioning on ROI-specific representations. This enables an interpretable mapping between brain regions and generated audio, allowing the contribution of distinct cortical areas to be explicitly probed. This is important because different regions of auditory cortex are thought to encode complementary aspects of sound, and separating their contributions helps reveal how complex auditory representations are organized in the brain.

We instantiate this framework for brain-to-audio reconstruction from fMRI. Early studies demonstrated that spectrotemporal features of natural sounds could be predicted from fMRI patterns using multivariate regression [Santoro et al., 2017], though resulting reconstructions lacked perceptual fidelity. More recent approaches have improved realism by decoding high-level DNN features and mapping them to a codebook-based audio generator [Park et al., 2025]. Denk et al. [2025] and Ciferri et al. [2025] have pursued decoding of musical stimuli from neural signals, both routing through latent representations before synthesis. In these pipelines, the intermediate representation also serves as a denoising stage, leveraging regularization to suppress measurement noise before synthesis [Ferrante et al., 2024b].

In contrast, our setting leverages the signal-to-noise ratio of fMRI responses acquired at ultra-high magnetic field strength (7 Tesla) [Santoro et al., 2017, Giordano et al., 2023], which exceeds that of conventional field strengths. This enables robust estimation of voxel-wise responses to individual sounds, substantially reducing measurement noise.

MAAS reconstructs natural sounds from fMRI with improved acoustic and semantic fidelity. The structured ROI-based conditioning enables analysis of the representational contribution of individual cortical regions via an ablation-based fidelity metric computed in a biologically grounded auditory representation space. We further characterize how brain-derived signals are guided through the generative process via attention routing.

Our contributions are: (i) an end-to-end framework for conditioning a pretrained audio diffusion transformer on ROI-wise fMRI responses, without an intermediate decoder trained to predict a predefined stimulus representation; (ii) an evaluation on a 7T dataset, including controlled twostage and caption-based baselines, and a neural-information controls; (iii) an analysis of ROI-level perceptual contributions using a biologically grounded auditory representation.

## 2 Related Work

Brain decoding with generative models. Recent work has shown that perceptual stimuli can be reconstructed from fMRI by conditioning generative models on features decoded from brain activity. In vision, latent diffusion models combined with DNN feature decoding (e.g., CLIP[Radford et al., 2021], DINO [Oquab et al., 2023]) have enabled high-quality image reconstruction [Ferrante et al., 2024a, Takagi and Nishimoto, 2023, Scotti et al., 2024]. In the auditory domain, early work decoded spectrotemporal features directly [Santoro et al., 2017], while more recent approaches using audio-AI models have similarly relied on intermediate representations to drive generative models [Park et al., 2025, Denk et al., 2025, Ciferri et al., 2025]. In contrast, our approach removes this intermediate stage by directly conditioning the generative model on neural signals.

Direct conditioning on neural representations. A small number of recent works have explored conditioning generative models directly on brain activity. NeuroAdapter [Feng et al., 2026] conditions a visual latent diffusion model, based on a UNet architecture [Rombach et al., 2022], on parcellated fMRI representations bypassing intermediate feature spaces. Our work extends this paradigm to audio diffusion transformers, which differ in architecture and conditioning interfaces.

Conditioning mechanisms for diffusion models. Several methods have been proposed to incorporate auxiliary inputs into diffusion models. IP-Adapter [Ye et al., 2023] introduces a lightweight mechanism for injecting external representations into cross-attention layers via key-value modulation, enabling flexible conditioning beyond text. AP-Adapter and MuseControlLite [Tsai et al., 2025] extend text-conditioned audio generation by incorporating reference-audio features or timevarying musical controls via decoupled cross-attention. We extend the adapter concept to diffusion transformer (DiT)–based audio models, enabling conditioning on continuous signals such as fMRI.

Interpreting brain contributions. While prior decoding work focuses primarily on reconstruction quality, fewer studies address how different brain regions contribute to the generated outputs [Santoro et al., 2017]. By conditioning on ROI–specific signals, our framework enables direct analysis of regional contributions through ablation, providing a more interpretable link between neural activity and generated audio.

## 3 Methods

We describe the neural data, the voxel selection procedure, the generative backbone, and the architecture of MAAS. Our framework is depicted in Figure 1.

## 3.1 Experimental Setup

We use the 7T fMRI dataset from Santoro et al. [2017], Giordano et al. [2023]. Five participants listened to 288 natural sounds (1 s duration each, divided into four non-overlapping sets of 72 sounds each) spanning six semantic categories (human non-speech vocalizations, speech, animal cries, musical instruments, nature scenes, and tool sounds) while fMRI data were acquired using a clustered EPI sequence (TR = 2600 ms, TA = 1200 ms, TE = 19 ms, voxel size = 1.5 mm isotropic, 7T Siemens). Stimuli were presented in the silent gap between volume acquisitions in a fast event-related design. Each sound was repeated three times across two scanning sessions. Single-sound BOLD responses were estimated via a two-step procedure a within four-fold cross-validation across the four stimulus sets (3 sets for training, 1 set for testing). First, using the training data, a voxel-specific hemodynamic response function (HRF) was estimated using finite impulse response (FIR) modeling [Kay et al., 2013, Santoro et al., 2014]. Second, sound-evoked responses were estimated by fitting fMRI responses, separately for training and test data, using the estimated HRFs and regularised linear regression (ridge). This procedure yields one response estimate (beta) per sound per voxel. Combined with the high functional contrast-to-noise ratio of 7T fMRI, this procedure provides a robust representation of auditory cortical activity [Santoro et al., 2017].

![](images/32c6b43ad8a824962494764cfbb8ce49d284a6a2959cb1aa249e2648380d3ad6.jpg)  
Figure 1: General overview of our approach. fMRI responses from auditory ROIs are projected and transformed into brain tokens, which are injected into the cross-attention layers of a Diffusion Transformer via a dedicated adapter. The model is conditioned solely on brain activity (empty text prompt) and generates audio through a frozen latent diffusion backbone.

For each participant, responses were extracted from six bilateral (left and right hemispheres combined) anatomical ROIs covering the auditory cortex: Heschl’s Gyrus (HG), Planum Temporale (PT), Planum Polare (PP), and posterior, middle, and anterior Superior Temporal Gyrus (pSTG, mSTG, aSTG). Each ROI was manually delineated on individual cortical surfaces in the original study. The final input to MAAS is a matrix $\check { \mathbf { X } } \in \mathbb { R } ^ { R \times V }$ per sample, where $R = 6$ is the number of ROIs and $V = 1 0 2 \bar { 4 }$ is the number of selected voxels per ROI (see below).

Since the number of voxels per ROI differs across participants, direct pooling across subjects is not possible. We address this by selecting the V = 1024 most responsive voxels per ROI per participant, quantified as the prediction accuracy (Pearson r) of a ridge regression encoding model trained to predict voxel responses from YamNet embeddings [Hershey et al., 2017], fitted exclusively on the training split (in cross-validation) of each fold to avoid circularity. This yields fixed-dimensional ROI representations that can be pooled across participants and fed jointly to MAAS, with the first module (Projector) learning shared weights across subjects. Note that YamNet serves solely as a voxel selection criterion, not as a conditioning signal for generation.

## 3.2 Generative Backbone: Stable Audio

We build on Stable Audio [Evans et al., 2025], a latent diffusion model designed for text-conditioned audio synthesis. The model operates in a compressed latent space obtained via a variational autoencoder (VAE), which maps waveforms $\mathbf { w } \in \mathbb { R } ^ { 1 \times T _ { w } }$ to latent representations $\mathbf { z } = \boldsymbol { \mathcal { E } } ( \mathbf { w } ) \in \mathbb { R } ^ { C \times T _ { z } }$ where C is the number of latent channels and $T _ { z }$ is the temporal length of the latent sequence. A Diffusion Transformer (DiT) performs iterative denoising in this latent space, conditioned on textual inputs encoded by the flan T5 model [Chung et al., 2024]. The resulting conditioning sequence $\mathbf { c } _ { \mathrm { t e x t } } \in \mathbb { R } ^ { L \times D }$ (with L = 128 token length, $D = 7 6 8$ feature dimension) is injected via cross-attention layers, while duration information is provided through global conditioning embeddings.

A key limitation of Stable Audio, and audio diffusion models more broadly, is that conditioning is restricted to text embeddings. The architecture does not provide a native mechanism for incorporating structured, non-linguistic signals such as neural activity. While auxiliary conditioning methods have been explored in image diffusion models (e.g., NeuroAdapter [Feng et al., 2026]), these approaches are tightly coupled to UNet-based architectures [Rombach et al., 2022] and do not transfer directly to transformer-based diffusion backbones. In particular, Stable Audio’s DiT departs from UNet-based designs in several key aspects: it applies rotary positional embeddings (RoPE) [Su et al., 2024] only to latent self-attention queries, uses grouped-query attention (GQA) [Ainslie et al., 2023] with fewer key/value than query heads, and exposes conditioning solely as token-level cross-attention rather than spatial feature maps. These choices break the assumptions of UNet-oriented adapters such as IP-Adapter, which rely on spatial attention blocks and matched key-value head structure. Direct reuse is therefore non-trivial: non-text tokens must be separated from text tokens, projected with compatible key/value heads, and routed without disrupting the pretrained denoising interface. MAAS addresses this gap with a dedicated brain pathway for DiT cross-attention.

## 3.3 MAAS Architecture

Our goal is to map fMRI signals directly into the conditioning space of the generative model, without relying on intermediate embeddings. MAAS comprises three components: (i) a ROI Projector that maps voxel-level activity to a shared embedding space, (ii) a Brain Token Generator that transforms region-wise representations into a compact sequence of conditioning tokens, and (iii) a transformercompatible adapter that injects these tokens into the diffusion backbone. During training, the standard text pathway is neutralized by providing an empty prompt to the T5 encoder. This ensures that all generative information is conveyed through the brain-derived conditioning stream, enabling a direct analysis of how neural signals influence synthesis.

The ROI Projector. For each ROI $r \in \{ 1 , \ldots , R \}$ (with $R = 6$ bilateral auditory regions), the corresponding voxel activity vector $\mathbf { x } _ { r } \in \mathbb { R } ^ { V }$ (with $V = 1 0 2 4$ selected voxels) is projected to the backbone feature space with dimension $D = 7 6 8$ via a region-specific linear map: $\mathbf { h } _ { r } =$ $\mathbf { x } _ { r } \mathbf { W } _ { r } + \mathbf { b } _ { r } \in \mathbb { R } ^ { D }$ , where $\mathbf { W } _ { r } ~ \in ~ \mathbb { R } ^ { V \times D }$ and $\mathbf { b } _ { r } \in \mathbb { R } ^ { D }$ are learnable parameters. Stacking across regions yields $\mathbf { H } \in \mathbb { R } ^ { R \times D }$ . Unlike approaches that collapse brain activity into a single global representation, this design preserves region-specific structure and allows the model to learn heterogeneous mappings across anatomically distinct auditory areas.

The Brain Token Generator. Because the number of ROIs is small, directly conditioning on H would limit expressive capacity. To address this, we transform the ROI embeddings into a richer set of conditioning tokens using a learned query-based aggregation mechanism.

Specifically, we employ a transformer decoder with learned queries:

$$
\mathbf { C } _ { \mathrm { b r a i n } } = \mathrm { C r o s s A t t n } ( \mathbf { Q } _ { \mathrm { l e a r n } } , \mathbf { H } + \mathbf { E } _ { \mathrm { r o i } } ) \in \mathbb { R } ^ { Q \times D } ,\tag{1}
$$

where $\mathbf { Q } _ { \mathrm { l e a r n } } \in \mathbb { R } ^ { Q \times D }$ are $Q$ learnable query embeddings and $\mathbf { E } _ { \mathrm { r o i } } \in \mathbb { R } ^ { R \times D }$ are learned positional embeddings encoding ROI identity.

This mechanism maps a small set of anatomically grounded signals into a higher-capacity token sequence compatible with the diffusion transformer. Learned queries allow flexible aggregation of information across regions, rather than relying on fixed pooling schemes.

Conditioning the DiT: fMRI Adapter for Audio. We integrate brain-derived tokens into the diffusion process through a parallel attention pathway. In Stable Audio, each cross-attention layer computes: $\mathbf { h } ^ { \prime } = \mathrm { A t t n } ( \mathbf { q } , \mathbf { k } _ { \mathrm { t e x t } } , \mathbf { v } _ { \mathrm { t e x t } } )$ , where $\begin{array} { r } { \dot { \mathbf { q } } = \mathbf { h } \dot { \mathbf { W } } ^ { Q } , \mathbf { k } _ { \mathrm { t e x t } } = \mathbf { c } _ { \mathrm { t e x t } } \mathbf { W } ^ { K } , \mathbf { v } _ { \mathrm { t e x t } } = \mathbf { c } _ { \mathrm { t e x t } } \mathbf { W } ^ { V } } \end{array}$

q are derived from latent features, and keys/values are obtained from the text conditioning sequence.

In our setting, $\mathbf { c _ { \mathrm { t e x t } } }$ is fixed to the T5 encoding of an empty string. We introduce a parallel "brain" branch with dedicated key and value projections: $\mathbf { k } _ { \mathrm { a d a p } } \doteq \mathbf { \bar { C } } _ { \mathrm { b r a i n } } \mathbf { \bar { W } } _ { \mathrm { a d a p } } ^ { K } , \mathbf { v } _ { \mathrm { a d a p } } = \mathbf { C } _ { \mathrm { b r a i n } } \mathbf { W } _ { \mathrm { a d a p } } ^ { V } .$

The output of the attention layer becomes a weighted combination of the two branches: $\mathbf { h } ^ { \prime } \mathbf { \Sigma } =$ $\mathrm { A t t n } ( { \bf q } , { \bf k } _ { \mathrm { t e x t } } , { \bf v } _ { \mathrm { t e x t } } ) { \bf \phi } + { \bf \nabla } \lambda \cdot \mathrm { A t t n } ( { \bf q } , { \bf k } _ { \mathrm { a d a p } } , { \bf v } _ { \mathrm { a d a p } } )$ , where λ is a learnable scalar that calibrates the strength of the brain pathway against the pretrained stream, since their optimal relative scale is not known a priori.

To preserve compatibility with the original model interface, we construct an augmented conditioning sequence

$$
\tilde { \mathbf { c } } = [ \mathbf { c _ { \mathrm { e m p t y } } } ; ~ \mathbf { c _ { \mathrm { s t a r t } } } ; ~ \mathbf { c _ { \mathrm { e n d } } } ; ~ \boldsymbol { \lambda } \cdot \mathbf { C _ { \mathrm { b r a i n } } } ] \in \mathbb { R } ^ { ( L + 2 + Q ) \times D } ,\tag{2}
$$

where $\mathbf { c } _ { \mathrm { s t a r t } } , \mathbf { c } _ { \mathrm { e n d } } \in \mathbb { R } ^ { 1 \times D }$ are the duration embeddings required by Stable Audio, and $L = 1 2 8$ is the length of the empty T5 sequence. Inside the processor, the last $\dot { Q }$ tokens are split off as $\mathbf { C } _ { \mathrm { b r a i n } }$ and routed to the Adapter branch; the remaining $L + 2$ tokens are routed to the standard text branch. This keeps the DiT interface unchanged and makes the pipeline drop-in compatible with the Stable Audio denoising loop.

Adapting this mechanism to a transformer-based diffusion model requires addressing two architectural constraints. First, Stable Audio uses rotary positional embeddings (RoPE), which are applied to queries but not to conditioning keys. We preserve this asymmetry and avoid applying RoPE to $\mathbf { k } _ { \mathrm { a d a p } }$ as brain tokens do not possess a meaningful temporal ordering relative to the latent sequence. Second, the model employs grouped-query attention (GQA), in which the number of key-value heads $H _ { k v }$ is smaller than the number of query heads $H _ { q } .$ To ensure compatibility, both text and brain branches replicate key-value heads via repeat interleaving to match the query head dimensionality before attention computation.

## 3.4 Training Objective

The DiT weights are frozen except for the cross-attention layers (keys, values, and layer norms of attn2), which are fine-tuned jointly with the adapter. The VAE weights remain fully frozen throughout training. Ground-truth waveforms are encoded into latents $\mathbf { z } _ { 0 }$ by the frozen VAE. At each training step, a random timestep t is sampled, noise ϵ is added to produce $\mathbf { z } _ { t } .$ , and the model is trained to predict the v-prediction target $\mathbf { v } _ { t }$ of Stable Audio’s cosine noise schedule, using an MSE loss weighted by Min-SNR-γ [Hang et al., 2023], which down-weights the contribution of low-SNR timesteps and stabilises training.

## 3.5 Evaluation

We evaluate reconstruction quality using four complementary metrics. Mel-SSIM and log-mel distance assess acoustic (spectral) fidelity. CLAP similarity measures cosine similarity in the CLAP embedding space [Elizalde et al., 2023], capturing high-level semantic correspondence between reconstructed and ground-truth audio. Pairwise identification accuracy follows the approach of Park et al. [2025]. We also compare our results with model baselines. The Brain-to-text baseline predicts flan-T5 text embeddings (obtained with sound captions) from brain activity via ridge regression and conditions the DiT on the resulting estimated sequence, replicating the standard two-stage paradigm. Brain-only passes $\mathbf { C } _ { \mathrm { b r a i n } }$ directly as the full conditioning sequence in place of the T5 embeddings. Brain-plus-empty adds $\mathbf { C } _ { \mathrm { b r a i n } }$ as a scaled residual matched to the lambda Adapter on top of the empty-prompt T5 embedding (without the Adapter routing). We also evaluate caption-based conditions. Caption-only conditions Stable Audio on the ground-truth test caption without brain input; this oracle condition measures generation from semantic information unavailable to the brain decoder. Caption-plus-brain (zero-shot) supplies the caption alongside brain tokens to the original MAAS model, which was trained with an empty text prompt.

Then we assess neural information over generative prior using two controls. First, MAAS is trained on 30 independently shuffled brain–stimulus assignments per condition, either unrestricted or within category, and evaluated by held-out pairwise identification. Second, ridge regression predicts audio embeddings from each ROI or all ROIs jointly, in a retrieval setting without audio generation [Ciferri et al., 2026]. We report Top-1 and Top-5 retrieval among 72 held-out stimuli; RidgeCV selects alpha from [0.01, 0.1, 1, 10, 100, 1000] by internal cross-validation on the training set.

Finally, to quantify the contribution of individual auditory ROIs, we introduce an ablation-based fidelity metric grounded in a computational model of auditory cortical processing. Reconstructions are processed using the NSL Auditory-Cortical Toolbox [Chi et al., 2005], which models the transformation of sound along the auditory pathway from the cochlea to primary auditory cortex and produces a multiscale spectro-temporal cortical representation for details. For each sound, an auditory spectrogram was first estimated using 128 cochlear frequency channels, with centre frequencies defined relative to 440 Hz. Spectrotemporal modulation energy was then computed with using 32 temporal modulation rates logarithmically spaced between 1 and 128 Hz and 32 spectral modulation scales logarithmically spaced between 0.2 and 10 cycles/octave. Upward and downward temporal-rate channels were averaged over time and converted to absolute modulation energy, yielding for each sound a 3D representation of size 32 scales × 64 rates × 128 frequency channels (See Appendix A for details). For each test sound, we compute the Pearson correlation r between the flattened NSL representations of the reconstruction and of the ground truth. Correlations are averaged over the sounds of each category within each fold, and then across folds in Fisher-z space $( \bar { r } = \operatorname { t a n h } ( \overline { { \operatorname { a r t a n h } r } } ) )$ . The relative fidelity loss for ROI i is $\Delta _ { i } = ( \bar { r } _ { \mathrm { r e m } } - \bar { r } _ { \mathrm { a l l } } ) / \bar { r } _ { \mathrm { a l l } }$ , where $\bar { r } _ { \mathrm { a l l } }$ and $\bar { r } _ { \mathrm { r e m } }$ refer to reconstructions using all ROIs and with ROI i removed or retained in isolation. Negative $\Delta _ { i }$ indicates lower fidelity than the full model.

Table 1: Reconstruction performance. Higher is better for CLAP similarity, pairwise identification (ID) accuracy, and Mel-SSIM; lower is better for log-mel distance. Caption-based conditions use ground-truth test captions.
<table><tr><td>Condition</td><td>CLAP sim. ↑</td><td>ID acc. ↑</td><td>Mel-SSIM ↑</td><td>Log-mel dist. ↓</td></tr><tr><td>Brain-conditioning baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>MAAS (our adapter)</td><td> $0 . 6 1 4 \pm 0 . 1 3$ </td><td> $8 5 . 4 \pm 4 . 2 \%$ </td><td> $0 . 2 9 7 \pm 0 . 0 9$ </td><td> $0 . 1 5 2 \pm 0 . 0 4$ </td></tr><tr><td>MAAS (no backbone fine-tuning)</td><td> $0 . 5 4 8 \pm 0 . 1 7$ </td><td> $7 3 . 1 \pm 4 . 8 \%$ </td><td> $0 . 2 8 3 \pm 0 . 0 8$ </td><td> $0 . 1 7 1 \pm 0 . 0 4$ </td></tr><tr><td>Brain-plus-empty</td><td> $0 . 4 3 6 \pm 0 . 1 4$ </td><td> $6 3 . 1 \pm 5 . 7 \%$ </td><td> $0 . 2 7 3 \pm 0 . 1 0$ </td><td> $0 . 1 8 5 \pm 0 . 0 5$ </td></tr><tr><td>Brain-only</td><td> $0 . 4 0 3 \pm 0 . 1 5$ </td><td> $5 8 . 3 \pm 5 . 9 \%$ </td><td> $0 . 2 2 6 \pm 0 . 0 8$ </td><td> $0 . 1 9 7 \pm 0 . 0 6$ </td></tr><tr><td>Two-stage fMRI-to-latent</td><td> $0 . 5 0 9 \pm 0 . 1 6$ </td><td> $7 0 . 1 \pm 5 . 9 \%$ </td><td> $0 . 2 7 0 \pm 0 . 1 0$ </td><td> $0 . 1 7 7 \pm 0 . 0 8$ </td></tr><tr><td>Caption-based conditions</td><td></td><td></td><td></td><td></td></tr><tr><td>Caption-only</td><td> $0 . 5 7 2 \pm 0 . 1 2$ </td><td> $8 7 . 9 \pm 5 . 4 \%$ </td><td> $0 . 1 6 6 \pm 0 . 0 8$ </td><td> $0 . 2 4 9 \pm 0 . 0 7$ </td></tr><tr><td>Caption-plus-brain (zero-shot)</td><td> $0 . 6 1 0 \pm 0 . 1 4$ </td><td> $8 4 . 6 \pm 5 . 7 \%$ </td><td> $0 . 2 7 5 \pm 0 . 0 9$ </td><td> $0 . 1 6 7 \pm 0 . 0 6$ </td></tr><tr><td>Caption-plus-brain (retrained)</td><td> $0 . 5 8 4 \pm 0 . 1 5$ </td><td> $8 2 . 4 \pm 5 . 1 \%$ </td><td> $0 . 3 1 1 \pm 0 . 1 1$ </td><td> $0 . 1 4 3 \pm 0 . 0 4$ </td></tr></table>

Table 2: Neural-information controls $( \mathrm { m e a n } \pm \mathrm { S D } , \% )$ . Top: pairwise identification after training with correct or shuffled brain–stimulus pairings (within-category or unrestricted stimulus identity shuffled). Bottom: retrieval from fMRI without generation (72 test stimuli).
<table><tr><td colspan="2">Training pairing</td><td>Correct</td><td colspan="3">Within-category shuffle</td><td colspan="2">Unrestricted shuffle</td><td>Chance</td></tr><tr><td colspan="2">Pairwise ID</td><td> $8 5 . 4 \pm 4 . 2$ </td><td></td><td> $7 3 . 6 \pm 4 . 0$ </td><td></td><td> $5 3 . 4 \pm 2 . 7$ </td><td></td><td>50.0</td></tr><tr><td>Retrieval</td><td>HG</td><td>PT</td><td>PP</td><td>mSTG</td><td>pSTG</td><td>aSTG</td><td>All ROIs</td><td>Chance</td></tr><tr><td>Top-1</td><td> $1 2 . 5 \pm 2 . 9$ </td><td> $1 4 . 2 \pm 3 . 7$ </td><td> $8 . 3 \pm 2 . 9$ </td><td> $1 2 . 2 \pm 3 . 3$ </td><td> $1 0 . 1 \pm 2 . 8$ </td><td> $4 . 9 \pm 1 . 6$ </td><td> $1 7 . 5 \pm 3 . 6$ </td><td>1.39</td></tr><tr><td>Top-5</td><td> $4 4 . 8 \pm 4 . 9$ </td><td> $4 7 . 9 \pm 4 . 7$ </td><td> $3 9 . 2 \pm { 3 . 8 }$ </td><td> $4 3 . 4 \pm 4 . 3$ </td><td> $3 9 . 9 \pm { 3 . 8 }$ </td><td> $2 6 . 4 \pm 3 . 1$ </td><td> $5 6 . 4 \pm 4 . 8$ </td><td>6.94</td></tr></table>

## 4 Results

We evaluate MAAS on brain-to-audio reconstruction from fMRI. We first assess reconstruction performance against baselines, then examine qualitative reconstruction behaviour, and finally analyze the contribution of individual auditory cortical regions using our ablation-based fidelity metric.

## 4.1 Quantitative Evaluation

Table 1 reports reconstruction performance on the held-out test set (72 sounds per fold), aggregated across four cross-validation folds and participants. Among brain-conditioning models, MAAS achieved the highest CLAP similarity (0.614), pairwise identification accuracy (85.4%), and Mel-SSIM, while also yielding the lowest log-mel distance. Performance degraded progressively as components of the conditioning mechanism were removed, confirming the importance of both the adapter and backbone fine-tuning, despite routing through a semantically structured embedding space.

Caption-only conditioning achieves high pairwise identification accuracy, as expected because the oracle caption explicitly provides semantic information. However, it performs substantially worse on both acoustic metrics than MAAS. This suggests that the caption and generative prior can produce a semantically compatible sound but do not recover the stimulus-specific spectrotemporal structure available in the neural responses. Adding captions only at inference to the original MAAS model constitutes an out-of-distribution condition because our model was trained exclusively with an empty prompt. Consistent with this, the zero-shot combination provides no systematic improvement over MAAS.

To address the possibility that reconstruction performance primarily reflects the generative prior, we performed two complementary controls (Table 2). Above-chance retrieval without audio generation demonstrates that stimulus information is decodable from the neural responses alone. The reduction in reconstruction accuracy after shuffling brain–stimulus pairings, including within semantic categories, further indicates that MAAS uses stimulus-specific neural information. Together, these controls support a neural contribution to reconstruction that cannot be explained by the generative prior alone.

## 4.2 Qualitative Reconstruction

MAAS produced high-quality sound-reconstructions from fMRI responses. We provide audio examples in the following site: https://fidelioc55.github.io/brainSounds-website/.

Figure 2 illustrates the denoising trajectory for two representative test samples. The final reconstructions capture key spectrotemporal characteristics of the ground-truth signals, including harmonic structure in speech and vocalizations and broadband dynamics in environmental sounds. Intermediate denoising steps reveal a structured progression: early steps capture coarse temporal envelopes, while later stages refine spectral detail. This behavior is consistent with the coarse-to-fine nature of diffusion-based generation [Ho et al., 2020].

We further analyzed attention weights during inference. The evolution of ROI-specific attention across denoising steps shows that different cortical regions contribute at different stages of synthesis. In particular, a marked reweighting occurs around steps 60–70, corresponding to the transition from low to high signal-to-noise ratio in the cosine schedule [Nichol and Dhariwal, 2021]. This pattern suggests that the adapter dynamically changes how it routes brain-derived information as generation progresses. See Appendix A for a full description of attention decomposition.

Importantly, the attention profiles were strongly sound-dependent: different stimuli elicit distinct patterns of ROI weighting throughout the denoising process. This indicates that the contribution of individual cortical regions is not fixed, but adapts to the specific acoustic content of each sound. This observation is consistent with the ROI-level fidelity analysis in Section 4.3, where different regions contribute differentially across sound categories.

![](images/ecde13d8e4a8e7d452e2d42a76bebbc189e6bff3fb67b57f273992e8915a5e61.jpg)  
Figure 2: Analysis of cortical contributions in MAAS. Evolution of per-ROI transported attention ${ { \bar { \alpha } } _ { t } } ( r )$ across 100 denoising steps for two test samples. Each row shows the reconstructed spectrogram at a given denoising step alongside the ground truth and the corresponding ROI attention profile.

## 4.3 Cortical Contribution Analysis

We examined the contribution of auditory ROIs using condition-specific models trained and evaluated with identical fixed ROI masks in all four folds (Figure 3a). Removing a single ROI produced comparatively modest changes in fidelity (averaged across categories, between −2.8% and +1.7% per ROI), whereas models conditioned on a single ROI lost up to 83% of the full-model fidelity in individual folds (removal vs. single-ROI, $p < 0 . 0 0 1$ ; repeated-measures ANOVA across the four folds on Fisher-z differences from the all-ROI model). Thus, for the reconstruction model, information across ROIs is largely redundant, and the remaining regions can typically compensate for an omitted one. Within this smaller range, the leave-one-ROI-out analysis showed an effect of ROI $( p = 0 . 0 4 4 )$ whose ordering mirrored the single-ROI analysis: removing HG or PT produced the largest losses on average (−2.8% and −2.5%), whereas removing STG regions left fidelity close to the all-ROI reference (−0.3% to +1.7%). The cost of removal was most evident for music, where every removal reduced fidelity on average and removing HG and PT produced the largest losses (−10.4% and $- 8 . 4 \% )$ , and for nature sounds $( \mathrm { H G } ; - 5 . 3 \% , \mathrm { P T } ; - 7 . 0 \% )$ . Single-ROI fidelity varied systematically across regions (effect of $\mathrm { R O I } , p < 0 . 0 0 1 )$ . Early supratemporal-plane ROIs retained most of the fidelity in isolation $( \mathrm { P T } ; - 1 2 . 2 \% , \mathrm { H G } ; - 1 3 . 1 \% , \dot { \mathrm { P P } } ; \bar { - } 1 9 . 7 \% )$ , whereas lateral STG regions retained less (pSTG: −26.9%, mSTG: −28.4%, aSTG: −45.9%; supratemporal vs. STG, $p = 0 . 0 0 3$ , paired t-test across folds), with the largest loss in the most anterior region (posterior-to-anterior trend, $p = 0 . 0 1 5 .$ , t-test on the fold-wise slopes). Together with the leave-one-out ordering, this pattern is consistent with spectrotemporal detail being most directly available in early auditory areas and with increasingly abstract representations along the STG [Norman-Haignere and McDermott, 2018, Giordano et al., 2023]. The effect of ROI also varied across categories, and we describe these categoryspecific patterns qualitatively. For speech, every removal reduced fidelity on average $( - 2 \%  t 0 - 3 \% )$ and all single-ROI models, including aSTG, performed similarly $( - 6 \% 1 0 - 1 1 \% )$ , consistent with distributed encoding of speech across the STG [Hamilton et al., 2021]. For non-speech vocalizations, removing any ROI slightly increased fidelity on average (+4% to +12%), indicating that additional inputs do not invariably improve reconstruction under this metric. Finally, the individual examples in Figure 3b complement these quantitative results by illustrating how different ROI conditions alter the reconstructed spectrotemporal structure.

![](images/257fe3c07516e52d05c513a4f8d01797496c6e999c9f5a7b774ab9dcdf7032ff.jpg)

a)  
![](images/4870d1a0fcfda9cd676142cb49d305522b6dfaeb89546fa4d1645a08d196a10e.jpg)

![](images/34434c191af4cd367ce7cc1c52764f49d72422c73236d9961e19334d0fcb0050.jpg)

b)  
![](images/abc1b229f10e3411108b32e67253ff52ee830c56c5a25c130e04ea4d5b03fd57.jpg)  
Figure 3: (a) Relative change in NSL-based reconstruction fidelity, $\Delta _ { i } ,$ by ROI and semantic category, computed from Fisher-z averaged correlations across the four cross-validation folds; brackets show the range [min, max] of fold-wise $\Delta _ { i }$ (%). Dots mark cells in which all four folds agree in sign. Left: leave-one-ROI-out analysis. Right: single-ROI analysis. Each partial-ROI condition uses a separately trained model with the same fixed mask during training and testing; negative values indicate lower fidelity. (b) NSL-derived spectrotemporal representations for two individual stimuli, woman speech and monkey call. Columns compare the ground truth, all-ROI model, and models trained without HG or mSTG. Rows show scale–rate, frequency–rate, and frequency–scale projections, respectively, obtained by averaging over the remaining dimension of the time-averaged representation.

## 5 Discussion

MAAS demonstrates that direct conditioning of a latent audio diffusion transformer on fMRI signals is both feasible and advantageous. The consistent performance gains across semantic and perceptual metrics suggest that bypassing intermediate feature prediction preserves information that is otherwise lost in the regression bottleneck, a finding that aligns with recent evidence in the visual domain [Feng et al., 2026] and extends it to the architecturally distinct setting of DiT-based audio generation.

The additional controls suggest that neural conditioning contributes to resolving the acoustic ambiguity inherent in semantic descriptions. A caption can identify a sound source while remaining compatible with many different acoustic realizations. The higher acoustic fidelity obtained with brain conditioning, including when oracle captions are available, suggests that fMRI responses provide constraints that help the generator recover aspects of the particular sound that was heard. The performance reduction under within-category shuffling further supports this interpretation: preserving category correspondence alone does not reproduce the performance obtained with correctly paired neural responses. Together, these findings suggest that MAAS uses neural information beyond broad semantic compatibility to guide the acoustic realization of generated sounds. The trade-off observed under joint caption and brain conditioning also indicates that semantic recognizability and acoustic fidelity capture distinct aspects of reconstruction.

From a technical standpoint, the adapter design addresses a non-trivial compatibility problem: injecting continuous, non-linguistic representations into a model built exclusively around text conditioning, while preserving RoPE asymmetry and GQA structure. The fact that fine-tuning only the crossattention layers is necessary to achieve strong performance suggests that the generative prior of Stable Audio is preserved, with the brain pathway acting as a targeted guidance.

The ablation analysis reveals a hierarchical behaviour that mirrors what is known about the functional organization of the human auditory cortex [Kell et al., 2018, Moerel et al., 2014, Rolls et al., 2023]: early areas (HG, PT) carry the least redundant and most self-sufficient spectrotemporal information, while higher-order STG regions are largely redundant with them and individually insufficient for faithful reconstruction of non-vocal sounds. This is consistent with evidence that primary auditory cortex responses are largely explained by spectrotemporal modulations [Santoro et al., 2014, 2017], whereas non-primary regions encode higher-order structure that goes beyond what modulation statistics can capture [Norman-Haignere and McDermott, 2018, Landemard et al., 2021], representations that are categorically informative but perceptually incomplete without the low-level substrate provided by earlier areas. The fact that higher-order regions alone produce the largest fidelity drops suggests that their intermediate acoustic-to-semantic representations [Giordano et al., 2023], while sufficient for recognition, do not carry enough fine-grained spectro-temporal detai to constrain the generative process toward perceptually faithful synthesis.

Several limitations should be acknowledged. The dataset comprises only five participants and 288 stimuli of 1 second duration, which constrains the generalizability of the reconstruction results. fMRI carries inherent limitations in temporal resolution: the BOLD signal integrates neural activity over several seconds and, with clustered acquisition and GLM-based beta estimation, the input to MAAS reflects a temporally blurred snapshot of auditory cortical processing. That the model achieves the reported reconstruction quality under these constraints is encouraging, and suggests that the spatial pattern of activation across ROIs carries substantial generative information even in the absence of fine-grained temporal dynamics, a hypothesis that could be tested with higher-temporal-resolution modalities. While the attention decomposition provides a mechanistic window into the generative process, the relationship between attention routing and ablation outcomes suggests that the adapter’s internal representations warrant further investigation.

## 6 Conclusion

We presented Multi-Area Auditory Synthesis (MAAS), a framework for directly conditioning a latent audio diffusion transformer on fMRI signals. The proposed adapter provides a mechanism for integrating neural representations into DiT-based generative models. Beyond reconstruction performance, MAAS offers a principled framework for probing the functional contribution of individual auditory cortical regions to sound perception. By conditioning an end-to-end generative model on structured brain representations, our approach enables a generative perspective on cortical information content, complementing traditional encoding and decoding analyses.

Looking forward, the direct conditioning paradigm opens several promising directions. In the clinical domain, the ability to reconstruct perceived audio from cortical activity could inform the development of auditory brain-computer interfaces for patients with severe motor impairments or disorders of consciousness, where passive listening paradigms are the only viable option.

## Acknowledgements

This work was funded by the Dutch Research Council (NWO Grant 406.20.GO.030) and the European Research Council through ERC-2024-SyG NASCE (grant agreement 101167313). This work was also supported by NEXTGENERATIONEU (NGEU) and funded by the Italian Ministry of University and Research (MUR), National Recovery and Resilience Plan (NRRP), project MNESYS (PE0000006) (to NT)– A Multiscale integrated approach to the study of the nervous system in health and disease (DN. 1553 11.10.2022); by the MUR-PNRR M4C2I1.3 PE6 project PE00000019 Heal Italia (to NT).

## Competing Interests

M.F. is employed by Tether Evo. The company had no role in the design, execution, or reporting of this work. The other authors declare no competing interests.

## References

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.

Zijiao Chen, Jiaxin Qing, and Juan Helen Zhou. Cinematic mindscapes: High-quality video reconstruction from brain activity. Advances in Neural Information Processing Systems, 36:24841–24858, 2023.

Taishih Chi, Powen Ru, and Shihab A Shamma. Multiresolution spectrotemporal analysis of complex sounds. The Journal ofthe Acoustical Society ofAmerica, 118(2):887–906, 2005.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal ofMachine Learning Research, 25(70):1–53, 2024.

Matteo Ciferri, Matteo Ferrante, and Nicola Toschi. Reconstructing music perception from brain activity using a prior guided diffusion model. Scientific Reports, 15(1):42108, 2025.

Matteo Ciferri, Matteo Ferrante, and Nicola Toschi. R&b-rhythm and brain: Cross-subject decoding of music from human brain activity. Neural Networks, page 109195, 2026.

Timo I Denk, Yu Takagi, Takuya Matsuyama, Andrea Agostinelli, Tomoya Nakai, Christian Frank, and Shinji Nishimoto. Text-to-music generation models capture musical semantic representations in the human brain. Nature Communications, 2025.

Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Oscar Esteban, Christopher J Markiewicz, Ross W Blair, Craig A Moodie, A Ilkay Isik, Asier Erramuzpe, James D Kent, Mathias Goncalves, Elizabeth DuPre, Madeleine Snyder, et al. fmriprep: a robust preprocessing pipeline for functional mri. Nature methods, 16(1):111–116, 2019.

Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

Pinyuan Feng, Hossein Adeli, Wenxuan Guo, Fan Cheng, Ethan Hwang, and Nikolaus Kriegeskorte. Towards interpretable visual decoding with attention to brain representations. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=YWlYITAhMC.

Matteo Ferrante, Tommaso Boccato, Stefano Bargione, and Nicola Toschi. Decoding visual brain representations from electroencephalography through knowledge distillation and latent diffusion models. Computers in Biology and Medicine, 178:108701, 2024a.

Matteo Ferrante, Tommaso Boccato, Furkan Ozcelik, Rufin VanRullen, and Nicola Toschi. Through their eyes: Multi-subject brain decoding with simple alignment techniques. Imaging Neuroscience, 2:1–21, 05 2024b. ISSN 2837-6056. doi: 10.1162/imag\_a\_00170. URL https://doi.org/10. 1162/imag\_a\_00170.

Bruno L Giordano, Michele Esposito, Giancarlo Valente, and Elia Formisano. Intermediate acousticto-semantic representations link behavioral and neural responses to natural sounds. Nature Neuroscience, 26(4):664–672, 2023.

Liberty S Hamilton, Yulia Oganian, Jeffery Hall, and Edward F Chang. Parallel and distributed encoding of speech across human auditory cortex. Cell, 184(18):4626–4639, 2021.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 7441–7451, 2023.

Shawn Hershey, Sourish Chaudhuri, Daniel PW Ellis, Jort F Gemmeke, Aren Jansen, R Channing Moore, Manoj Plakal, Devin Platt, Rif A Saurous, Bryan Seybold, et al. Cnn architectures for large-scale audio classification. In 2017 ieee international conference on acoustics, speech and signal processing (icassp), pages 131–135. IEEE, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

Kendrick Kay, Ariel Rokem, Jonathan Winawer, Robert Dougherty, and Brian Wandell. Glmdenoise: a fast, automated technique for denoising task-based fmri data. Frontiers in neuroscience, 7:54426, 2013.

Alexander JE Kell, Daniel LK Yamins, Erica N Shook, Sam V Norman-Haignere, and Josh H McDermott. A task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical processing hierarchy. Neuron, 98(3):630–644, 2018.

Nina Kowalski, Didier A Depireux, and Shihab A Shamma. Analysis of dynamic spectra in ferret primary auditory cortex. i. characteristics of single-unit responses to moving ripple spectra. Journal ofneurophysiology, 76(5):3503–3523, 1996.

Agnès Landemard, Célian Bimbard, Charlie Demené, Shihab Shamma, Sam Norman-Haignere, and Yves Boubenec. Distinct higher-order representations of natural sounds in human and ferret auditory cortex. Elife, 10:e65566, 2021.

Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:2871–2883, 2024.

Andrew Luo, Maggie Henderson, Leila Wehbe, and Michael Tarr. Brain diffusion for visual exploration: Cortical discovery using large scale generative models. Advances in Neural Information Processing Systems, 36:75740–75781, 2023.

Michelle Moerel, Federico De Martino, and Elia Formisano. An anatomical and functional topography of human auditory cortical areas. Frontiers in neuroscience, 8:225, 2014.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.

Sam V Norman-Haignere and Josh H McDermott. Neural responses to natural and model-matched stimuli reveal distinct computations in primary and nonprimary auditory cortex. PLoS biology, 16 (12):e2005127, 2018.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Furkan Ozcelik and Rufin VanRullen. Natural scene reconstruction from fmri signals using generative latent diffusion. Scientific Reports, 13(1):15666, Sep 2023. ISSN 2045-2322. doi: 10.1038/ s41598-023-42891-8. URL https://doi.org/10.1038/s41598-023-42891-8.

Jong-Yun Park, Mitsuaki Tsukamoto, Misato Tanaka, and Yukiyasu Kamitani. Natural sounds can be reconstructed from human neuroimaging data using deep neural network representation. PLoS biology, 23(7):e3003293, 2025.

Jacob S Prince, Ian Charest, Jan W Kurzawski, John A Pyles, Michael J Tarr, and Kendrick N Kay. Improving the accuracy of single-trial fmri response estimates using glmsingle. eLife, 11:e77599, nov 2022. ISSN 2050-084X. doi: 10.7554/eLife.77599. URL https://doi.org/10.7554/ eLife.77599.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Edmund T Rolls, Josef P Rauschecker, Gustavo Deco, Chu-Chung Huang, and Jianfeng Feng. Auditory cortical connectivity in humans. Cerebral Cortex, 33(10):6207–6227, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Roberta Santoro, Michelle Moerel, Federico De Martino, Rainer Goebel, Kamil Ugurbil, Essa Yacoub, and Elia Formisano. Encoding of natural sounds at multiple spectral and temporal resolutions in the human auditory cortex. PLoS computational biology, 10(1):e1003412, 2014.

Roberta Santoro, Michelle Moerel, Federico De Martino, Giancarlo Valente, Kamil Ugurbil, Essa Yacoub, and Elia Formisano. Reconstructing the spectrotemporal modulations of real-life sounds from fmri response patterns. Proceedings of the National Academy of Sciences, 114(18):4799–4804, 2017.

Paul S Scotti, Mihir Tripathy, Cesar Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh Narang, Charan Santhirasegaran, Jonathan Xu, Thomas Naselaris, Kenneth A Norman, et al. Mindeye2: Shared-subject models enable fmri-to-image with 1 hour of data. arXiv preprint arXiv:2403.11207, 2024.

Ken Shirakawa, Yoshihiro Nagano, Misato Tanaka, Shuntaro C Aoki, Yusuke Muraki, Kei Majima, and Yukiyasu Kamitani. Spurious reconstruction from brain activity. Neural Networks, 190: 107515, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Yu Takagi and Shinji Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14453–14463, 2023.

Fang-Duo Tsai, Shih-Lun Wu, Haven Kim, Bo-Yu Chen, Hao-Chung Cheng, and Yi-Hsuan Yang. Audio prompt adapter: Unleashing music editing abilities for text-to-music with lightweight finetuning. arXiv preprint arXiv:2407.16564, 2024.

Fang-Duo Tsai, Shih-Lun Wu, Weijaw Lee, Sheng-Ping Yang, Bo-Rui Chen, Hao-Chung Cheng, and Yi-Hsuan Yang. Musecontrollite: Multifunctional music generation with lightweight conditioners. arXiv preprint arXiv:2506.18729, 2025.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Ziyi Ye, Qingyao Ai, Yiqun Liu, Maarten de Rijke, Min Zhang, Christina Lioma, and Tuukka Ruotsalo. Generative language reconstruction from brain recordings. Communications Biology, 8 (1):346, 2025.

## A Technical appendices and supplementary material

We provide code for the reproducibility of our pipeline at the following link: https://anonymous.   
4open.science/r/brainSounds\_nips-DFC1.

Attention decomposition. MAAS exposes two distinct loci of attention over auditory ROIs, which can be composed to obtain a single ROI-level influence map. Let $M ( k , r )$ denote the cross-attention weight assigned by the Brain Token Generator from query token k to ROI r, and let $B _ { t } ( l , k )$ denote the attention weight assigned by the fMRI Adapter branch from latent position l to brain token k at denoising timestep t. The net influence of ROI r on latent position l is obtained by composing the two maps:

$$
{ \mathrm { t r a n s p o r t } } _ { t } ( l , r ) = \sum _ { k } B _ { t } ( l , k ) \cdot M ( k , r ) .\tag{3}
$$

Averaging over latent positions yields a per-ROI influence score $\begin{array} { r } { \bar { \alpha } _ { t } ( r ) = \frac { 1 } { L } \sum _ { l } } \end{array}$ transport (l, r).

Hyperparameter selection. We performed a grid search over key hyperparameters on the validation split. The explored ranges and final selected configuration are summarized in Table A1. To enable classifier-free guidance at inference time, we apply conditioning dropout with probability $p = 0 . 0 5$ by zeroing $\mathbf { C } _ { \mathrm { b r a i n } }$ during training. The model is optimised with AdamW $( \mathrm { l r = \dot { 1 } 0 ^ { - 4 } }$ , weight decay $\dot { = } 1 0 ^ { - 4 } )$ , cosine annealing to $\mathrm { l r } _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ and gradient clipping at norm 1.0. Experiments were run on a shared server equipped with eight NVIDIA A100 GPUs (80 GB each), 256 CPU threads, and 2 TB of system memory. Each training run used a single GPU for 60 epochs, and the main configuration updated 181M trainable parameters.

<table><tr><td>Hyperparameter</td><td>Selected</td><td>Search range</td></tr><tr><td>Number of brain queries Q</td><td>16</td><td>{6, 16, 64, 128}</td></tr><tr><td>Conditioning strategy</td><td>adapter</td><td>{brain-only, residual, adapter}</td></tr><tr><td>BrainTokenGen. layers</td><td>1</td><td>{1, 2, 4}</td></tr><tr><td>Transformer dimension</td><td>1024</td><td>{512, 1024, 2048}</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $\{ 1 \cdot 1 0 ^ { - 5 } , 5 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - 4 } \}$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $\{ \tilde { 0 } , 1 \cdot 1 0 ^ { - 5 } , 1 \cdot 1 0 ^ { - 4 } , 1 \cdot 1 0 ^ { - \tilde { 3 } } \}$ </td></tr><tr><td>Effective batch size</td><td>64</td><td>{16, 32, 64}</td></tr><tr><td>Min-SNR γ</td><td>3.0</td><td>{0, 1, 3, 5}</td></tr><tr><td>CFG dropout</td><td>0.05</td><td>{0, 0.05, 0.1, 0.3}</td></tr><tr><td>Guidance scale</td><td>3.0</td><td>{1.0, 2.0, 3.0, 5.0}</td></tr></table>

Table A1: Hyperparameter selection. We report the final configuration and the grid explored during validation.

External validation on an independent 3T dataset. To assess whether the applicability of MAAS extends beyond the original 7T acquisition, we evaluated it on the public 3T dataset of Park et al. [2025], comprising five independently acquired participants and natural sounds of 8 or 10 seconds. This experiment introduces variation in field strength, stimulus duration, participants, and acquisition protocol, providing an external assessment of the brain-conditioning approach under different measurement conditions. We considered two strategies for estimating stimulus-related fMRI responses. The first followed the delayed-volume averaging procedure of the original study: BOLD volumes were temporally aligned with the stimuli by accounting for the hemodynamic delay (2 seconds), divided into overlapping windows, and then averaged to obtain stimulus-related response patterns. For the second strategy, we estimated stimulus-evoked beta coefficients via GLMsingle [Prince et al., 2022]. Unlike delayed-volume averaging, this approach models the BOLD time series using the experimental design to estimate response amplitudes associated with stimulus presentations. In both strategies we used fMRIprep library [Esteban et al., 2019] to process neural data. The resulting estimates were organized into the same six auditory ROI inputs used by MAAS (HG, PT, PP, mSTG, pSTG, and aSTG). Using stimulus-wise beta representations improves comparability with the neural input format of our 7T experiments, while differences in acquisition and response-estimation procedures remain.

MAAS achieved pairwise identification accuracies of $8 3 . 9 \pm 3 . 8 \%$ with delayed-volume averaging and 86.1 ± 3.5% with GLMsingle. The higher point estimate with GLMsingle suggests that response estimation can affect reconstruction quality, while the performance obtained with both strategies supports the applicability of MAAS across different fMRI input representations. However, Park et al. [2025] identification score uses a different evaluation representation, and their models are subjectspecific whereas MAAS uses a cross-subject approach. These values therefore provide contextual benchmarks rather than a controlled comparison of reconstruction methods. Overall, the experiment extends the evaluation of MAAS to an independent 3T dataset without establishing superiority or equivalence to the published pipeline.

Auditory-Cortical Representation. The perceptual fidelity metric $\Delta _ { i }$ is computed in the representational space of the NSL Auditory-Cortical Toolbox [Chi et al., 2005], which models sound processing along the auditory pathway in two stages.

A waveform is decomposed into frequency-specific temporal envelopes using a bank of 128 asymmetric, constant-Q cochlear filters, spaced at 24 channels per octave relative to a 440 Hz reference frequency. Each channel then undergoes compression, lateral inhibition, and leaky temporal integration [Chi et al., 2005], yielding the auditory spectrogram $y ( t , x )$ , a time–frequency representation reflecting auditory-nerve firing rates rather than raw acoustic energy. The auditory spectrogram is then analysed by a bank of two-dimensional filters tuned to spectrotemporal modulations, modelling the selectivity of neurons in primary auditory cortex [Kowalski et al., 1996]. Each filter is characterised by a temporal modulation rate ω and a spectral modulation scale Ω:

$$
z ( t , x ; \omega , \Omega ) = y ( t , x ) * _ { t , x } \left[ g _ { w } ( t ; \omega ) h _ { w } ( x ; \Omega ) \right] .\tag{4}
$$

Upward and downward temporal-rate channels are averaged and integrated over time, yielding a three dimensional modulation transfer function (MTF) of size 32 scales×64 rates×128 frequency channels per sound. For $\Delta _ { i } .$ , Pearson correlation is computed between the flattened three-dimensional representations of reconstructed and ground-truth audio.

For visualization, each three-dimensional MTF is projected into three two-dimensional summaries: rate × scale (Figures A3), averaged over frequency; rate × frequency (Figures A2), averaged over scale; and frequency × scale (Figures A1), averaged over rate. These projections are computed for ground-truth sounds, the full model, and ablated or ROI-specific conditions, and visualised either for individual sounds or as category-averaged responses across the six semantic classes. Colour scaling is applied using the 1st–99th percentile range, either globally across panels or normalised per panel to the ground-truth range.

Frequency x Scale  
![](images/a2746bd17fa79d83b24cf1156b225791bb87cf3f74350deced0c5a635fca5bca.jpg)  
Figure A1: Category-averaged frequency–scale projections of NSL-derived spectrotemporal representations, obtained by averaging over temporal modulation rate. Rows correspond to semantic categories; columns show the ground-truth audio, full-model reconstructions, and reconstructions from models trained without HG or mSTG, respectively. The horizontal axis denotes spectral modu lation scale (cycles/octave), and the vertical axis denotes characteristic frequency (Hz).

Rate x Frequency  
![](images/34f308055fafd4b6ff2b116916524bf5f33adda1ce12bcb5cc884915706fa629.jpg)  
Figure A2: Category-averaged rate–frequency projections of NSL-derived spectrotemporal representations, obtained by averaging over spectral modulation scale. Rown and columns follow the same structure as frequency × scale. The horizontal axis denotes temporal modulation rate (Hz), and the vertical axis denotes characteristic frequency (Hz).

Rate x Scale  
![](images/908ec5228f199a643a3dcd4f3b7c18bc38efc72736da00d87a692a2bc964a43a.jpg)  
Figure A3: Category-averaged rate–scale projections of NSL-derived spectrotemporal representations, obtained by averaging over characteristic frequency. The horizontal axis denotes temporal modulation rate (Hz), and the vertical axis denotes spectral modulation scale (cycles/octave).