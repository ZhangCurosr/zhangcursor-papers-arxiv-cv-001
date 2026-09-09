# Canonical Color as a Lens into Concept Decodability in Vision Encoders and VLMs

Xiaofu Chen<sup>1</sup> Stella Frank<sup>2</sup> Yova Kementchedjhieva<sup>1</sup>

<sup>1</sup>MBZUAI <sup>2</sup>Technical University of Denmark

{xiaofu.chen, yova.kementchedjhieva}@mbzuai.ac.ae stefra@dtu.dk

## Abstract

Visual encoders construct a representation of the image input for Vision-Language models. How much conceptual, as opposed to immediately visible, information does this representation contain? We use canonical color as a controlled test case to ask whether vision encoders make canonical-color information linearly accessible, even when color is removed from the input image. We construct a dataset of objects with canonical colors, and probe vision encoders for both color and object identity using color and grayscale images. We find that canonical color remains decodable from grayscale images, and is tied to predicted object identity, indicating a conceptual link. Extending this analysis to full VLMs, we find that VLM post-training can have a surprisingly large effect on color decodability in the vision encoder. Overall, canonical color provides a usefully controllable lens for tracing objectlevel conceptual semantic information in vision encoders and VLMs.

## 1 Introduction

Humans associate object concepts with typical attributes, including canonical color. Such attributes are part of conceptual representations, and color can be especially diagnostic for some object categories (McRae et al., 2005; Tanaka and Presnell, 1999). Human studies also show that object knowledge can affect perceived color (Hansen et al., 2006; Witzel et al., 2011). Thus, knowing that bananas are yellow is different from seeing yellow pixels in a particular image. For text-only models, acquiring such typical knowledge can be difficult because text often states unusual rather than obvious properties, a problem known as reporting bias (Gordon and Van Durme, 2013). Multimodal semantic work has therefore used visual grounding to complement linguistic evidence (Bruni et al., 2014; Kiela and Bottou, 2014; Silberer and Lapata, 2014). Vision encoders have direct access to object colors during training, but it remains unclear whether they encode canonical color as object-level conceptual knowledge or simply expose chromatic cues from the input during inference.

Recent work has begun to address this question with representation-level probing. Linear probes are widely used to study what information is accessible in intermediate representations, although their results depend on the probe and control setting (Alain and Bengio, 2017; Hewitt and Liang, 2019; Belinkov, 2022). Onea¸ta et al.˘ (2025) show that frozen vision encoders encode many semantic attributes, including color. However, their stimuli are full-color RGB images, so color prediction may reflect visible pixels rather than canonical color knowledge. In VLMs, final answers can also be shaped by language priors, dataset bias, hallucination, or weak object–attribute binding (Goyal et al., 2017; Rohrbach et al., 2018; Li et al., 2023; Yuksekgonul et al., 2023). Golovanevsky et al. (2025) study such conflicts with counterfactual images, e.g., a blue strawberry, but full-model behavior does not reveal where canonical color information is represented within the vision–language pipeline.

In this work, we use canonical color as a controlled probe of object-level semantic information in vision encoders and VLMs. Color is a useful case because, unlike many other visual attributes, it can be directly removed from the image input, by gray-scaling the image, while leaving the object structure intact. Because canonical color is defined at the object-category level, its relation to object identity is part of the question we study, rather than a confound. We ask whether canonical color remains linearly accessible through object-level representations after visible color is removed, and whether object recognition alone accounts for its decodability. In other words, we probe the encoder: How is a black-and-white banana like black-andwhite daffodils and lemons?

Furthermore, if visual encoders represent canonical colors as part of their object concept representations, how is this information passed on to the language backbone in VLM architectures? Does VLM training preserve the conceptual organization in the visual encoder, or is there a conceptual reorganization towards the language model?

To answer these questions, we construct a dataset of object classes annotated with basic canonical color labels and train linear probes on frozen representations. RGB inputs provide an upper-bound setting with visible color cues, while our Grayscale setting removes chromatic information and reduces simple luminance cues through grayscale conversion and histogram equalization. We validate this control with Visual-CounterFact (Golovanevsky et al., 2025) by comparing actual-color prediction from recolored images with canonical-color prediction from both original and counterfactual images. We then extend the probing logic to VLMs by comparing matched pre- and post-VLM vision towers and decoder-side visual-token representations.

Our results show that canonical color remains linearly decodable even after direct color cues are removed. Counterfactual validation indicates that this residual signal is better aligned with canonical object color than with actual surface color. We also find that canonical color is related to objectclass information, but the two signals are only partially aligned. Finally, VLM post-training changes where object and color information are most linearly accessible: in some models, this information becomes less accessible in the standalone vision tower but reappears after the visual interface or inside decoder-side visual-token states.

## 2 Related Work

Semantic feature norms describe object concepts through typical properties, including perceptual attributes such as color (McRae et al., 2005; Devereux et al., 2014; Bannert and Bartels, 2013). Recent work has used probing to ask whether pretrained representations encode such attributes. Onea¸ta et al.˘ (2025) probe image encoders, multimodally trained image encoders, and language-only models for perceptual, functional, and encyclopedic attributes. Our work follows this representationlevel view, but focuses on canonical color as a controlled case study. Color is a special attribute because it is directly visible in RGB images: a successful color probe may rely on pixel-level chromatic cues rather than object-level canonical color.We therefore introduce input-side controls to remove chromatic information and further reduce the influence of luminance cues.

Linear probes are useful for tracing what information is accessible in frozen representations (Alain and Bengio, 2017), but probe accuracy alone does not show that a model uses that information in its final predictions. Prior work therefore emphasizes careful probe design, control tasks, and cautious interpretation (Hewitt and Liang, 2019; Belinkov, 2022). We follow this view by treating probing as a measure of linear accessibility, and by validating our color-controlled setting with counterfactual images.

Work on VLMs further shows that model predictions can be shaped by language priors, dataset bias, hallucination, and weak object–attribute binding (Goyal et al., 2017; Rohrbach et al., 2018; Li et al., 2023; Zhao et al., 2022; Yuksekgonul et al., 2023). More directly related to color, Tang et al. (2023) identify Concept Association Bias, where VLMs fill in strongly associated missing concepts across modalities. Liang et al. (2025) show that color perception, reasoning, and robustness remain challenging for modern VLMs. Golovanevsky et al. (2025) further study conflicts between visual evidence and memorized priors using Visual CounterFact, where objects are recolored to non-canonical colors. Unlike these behavioral studies, we localize where object and canonical color information are linearly decodable across vision encoders, VLM-trained vision towers, visual interfaces, and decoder-side visual-token representations.

## 3 Data and Probing Setup

This section describes our controlled probing setup. We first construct a dataset of object–color pairs, where each object class is associated with a canonical color label. We then apply image-side controls to separate visible pixel color from canonical object color. Finally, we pass the images through frozen vision encoders, extract layer-wise representations, and train linear probes to predict canonical color and object identity from these representations.

## 3.1 Canonical Color Dataset

We construct a dataset of 708 object classes, each associated with one of 10 canonical-color labels, with a target of five images per class.

Our dataset extends the object–color pairs compiled by Golovanevsky et al. (2025) with additional pairs sourced from Wikidata. We retain entries whose structured has-color property contains a single value, which provides the canonical-color label. To reduce the label-space complexity, we map these values to ten basic colors: black, blue, brown, gray, green, pink, purple, red, white, and yellow. The resulting object–color pairs are manually screened for validity.

For each class, we target five images by combining those linked from Wikidata with additional Google Search results, retaining up to five images that clearly depict the object and are consistent with its canonical color.

Because Pokémon were over-represented, we downsample them to 50 classes while prioritizing low-frequency colors (gray, brown, and red). The resulting color distribution is shown in Figure 7; white is the most frequent label with 99 classes and gray the least frequent with 48.

## 3.2 Input-side Controls for Color Probing

A probe that predicts the canonical color of an object from a vision encoder’s representation may rely on color information that is directly present in the input image. RGB images contain real pixellevel color, so high probing accuracy in this setting does not by itself show that the encoder stores color as concept-level knowledge. We therefore treat RGB as an upper-bound setting and remove this direct color channel in the controlled settings.

To remove pixel color, we first convert each image to grayscale by using a single luminance channel and replicating it across three channels. We then apply per-image histogram equalization to reduce brightness-based cues that may still correlate with color. We refer to this processed input as grayscale throughout the paper. Thus, grayscale denotes grayscale conversion followed by per-image histogram equalization.

## 3.3 Probing Tasks and Evaluation

We probe the internal representations of frozen vision encoders to ask two questions. First, do these representations contain information that can predict an object’s canonical color? Second, how does this color information relate to object-class information? To answer these questions, we use two linear probing tasks. The color probe predicts one of the 10 color classes in our dataset, while the objectclass probe predicts the object identity among 708 object classes.

We compare the layer-wise trajectories of the two probes to test whether canonical color decodability is associated with category-level object semantics. Because the two tasks have different label spaces, 10 color classes versus 708 object classes, their absolute accuracies are not directly comparable. We therefore focus on how their accuracies change across layers and whether the two trends become aligned.

Canonical color labels are defined over object categories rather than over individual image instances. For example, the label “yellow” for an image containing a banana reflects the canonical color associated with the object category banana. Thus, a color probe that predicts “yellow” for such an image may rely on representations that support category-level identification of the object as a banana, or on other image-level cues that correlate with canonical color. A color signal that appears when object-class recognition is still weak may reflect lower-level visual regularities or coarse object information. A color signal that strengthens together with object-class recognition is more likely to be linked to category-level object semantics.

4 Controlled Probing of Canonical Color in Vision Encoders

## 4.1 Experimental Setup

Vision Encoders. We evaluate five vision encoders that cover different supervision signals and architectural designs. CLIP (Radford et al., 2021) and SigLIP (Zhai et al., 2023) are languagesupervised models trained with image–text alignment objectives, so their visual representations are shaped by natural language supervision. DI-NOv2 (Oquab et al., 2024) and ViT-MAE (He et al., 2022) are self-supervised vision models, but they use different objectives: DINOv2 learns through self-distillation, whereas ViT-MAE learns by reconstructing masked image patches. We also include Swin-V2 (Liu et al., 2022), a hierarchical vision transformer, to test whether the observed patterns hold beyond standard ViT-style backbones. All models are used at the ViT-Base scale, or at the corresponding hierarchical scale for Swin-V2. Model repositories are listed in Appendix A.

Features. For each input image, we cache hidden states from all transformer blocks and use meanpooled patch features for all main probing results. For Swin-V2, adjacent blocks within each stage are averaged to form 12 layer-indexed features, matching the ViT-Base models.

![](images/1cbe18b1dc9772b72e2613994a179616f3877e2f18037b777f814ed5e7d5a8bb.jpg)

Probing. For each vision encoder, input setting, layer, and representation type, we train an independent linear probe on the cached image-level features. Each probe is an L2-regularized multinomial logistic regression classifier trained for up to 1000 optimization iterations. The vision encoders remain frozen throughout probing.

For color probing, we use 5-fold object-classlevel cross-validation to avoid leakage across images of the same object class. We split the 708 object classes into five disjoint folds. In each fold, the probe is trained on all images from the training object classes and evaluated on the held-out object classes. At test time, we average the predicted probability vectors across all images of each test object class and take the argmax, producing one color prediction per object class. We report accuracy across the five folds.

For object-class recognition, we use a per-class image hold-out setting, since holding out entire object classes would remove the target labels from training. For each object class with at least two images, we hold out one image for testing and use the remaining images for training. Object classes with only one image are used for training only. This task tests whether different images of the same object class are grouped in the representation space.

## 4.2 Validating the Color-controlled Setting

Before using grayscale as our main controlled setting, we first validate what this preprocessing removes and what it preserves (Figure 1). All experiments use patch-averaged representations from the same five frozen encoders used in the main experiments, with results averaged across encoders. First, color probing accuracy drops sharply from RGB to grayscale, confirming that RGB probing is strongly driven by visible pixel color. Second, object-class recognition also drops under grayscale, but remains clearly decodable and continues to improve with depth. Thus, grayscale removes direct color cues while still preserving enough object-level structure for semantic probing.

These two panels do not tell us whether the remaining color signal reflects the actual surface color in the image or the canonical color of the object. To separate these possibilities, we repeat the probing analysis on Visual-CounterFact, which contains paired original and recolored images of

Figure 1: Validation of the grayscale control. Left: color probing drops after visible color cues are removed. Middle: object-class recognition remains decodable. Right: Visual-CounterFact separates actual surface color from canonical object color. Dashed lines show majority baselines; shaded bands show ±1 standard deviation across encoders.

the same object (Golovanevsky et al., 2025). We apply the same grayscale preprocessing to all images. The actual-color probe on counterfactual images stays close to its majority baseline, indicating that the counterfactual surface color is mostly removed. In contrast, the two canonical-color probes, one on counterfactual images and one on original images, are both well above their majority baseline and show similar layer-wise trends. The actual-color probe and the canonical-color probes use different label spaces. We therefore use the actual-color probe only as a check that the counterfactual surface color is not recoverable after grayscale preprocessing. The main validation comes from comparing the canonical-color probes on the original and the counterfactual images, which use the same canonical-color label space. Their similar layerwise behavior shows that canonical color remains decodable from both counterfactual and original images after visible color cues are removed.

Overall, these controls address two types of residual cues. First, grayscale conversion removes chromatic cues, histogram equalization reduces simple brightness cues, and Visual-CounterFact tests whether the counterfactual surface color can still be decoded after preprocessing. The actualcolor probe remains near the majority baseline, while the canonical-color probes on the original and counterfactual images remain above baseline. Second, grayscale preserves object-correlated structure, such as texture, material, and luminance layout. We do not treat this structure as a confound: our claim is not that canonical color can be decoded independently of object identity, but that it can be linearly decoded through object-level representations, as examined in Section 4.4. In short, the control separates canonical color from surface color, not canonical color from object identity.

![](images/578eb518d6e085b0a71fd4dc0201b93feb5e27727c5df5ec86b904f1dfabf354.jpg)  
Figure 2: Layer-wise color probing accuracy under grayscale inputs using patch-averaged representations. The figure compares five vision encoders: CLIP, SigLIP, DINOv2, ViT-MAE, and Swin-V2.

## 4.3 Canonical Color Decodability Across Encoders

We next ask whether canonical color remains linearly decodable across different vision encoders after visible color cues have been removed. Figure 2 shows that all five encoders remain above chance and the 13.98% majority-class baseline under grayscale inputs. Most models improve from early to middle layers, suggesting that canonicalcolor-predictive information becomes more accessible in later representations. Detailed best-layer results, confidence intervals, and significance tests are reported in Table 4 in Appendix B.

Across encoders, differences in peak accuracy are modest. DINOv2 reaches the highest overall peak, while SigLIP and Swin-V2 also perform strongly in the middle layers. CLIP improves across layers but does not clearly outperform the self-supervised encoders, suggesting that the signal does not depend exclusively on image–text supervision. Several models, especially ViT-MAE and to a lesser extent SigLIP and DINOv2, show some degradation in their final layers, which may reflect increasing specialization for their pretraining objectives.

Because this cross-encoder comparison holds model scale approximately fixed, we additionally conduct a controlled within-family comparison of

<table><tr><td>Model</td><td>Params</td><td>RGB</td><td>Gray</td><td>Gray + HE</td></tr><tr><td>DINOv2-S</td><td>22M</td><td>71.1</td><td>46.2</td><td>41.8</td></tr><tr><td>DINOv2-B</td><td>86M</td><td>77.0</td><td>48.6</td><td>43.8</td></tr><tr><td>DINOv2-L</td><td>304M</td><td>79.4</td><td>51.0</td><td>46.0</td></tr><tr><td>DINOv2-g</td><td>1.1B</td><td>80.6</td><td>51.8</td><td>48.0</td></tr></table>

Table 1: Best-layer canonical-color accuracy (%) across different DINOv2 model scales. All results are obtained using the same object-class-level five-fold protocol and mean-pooled patch features as in the main experiments. HE denotes histogram equalization.  
![](images/be7e0a8d811ce8869b37b3ac79bf4c8a7fe3ed3952f74fdda42935c505fbee0a.jpg)  
Figure 3: Layer-wise object-class recognition and color probing under grayscale inputs. Solid and dashed lines show object-class and color probing accuracy, respectively. Each subplot reports the Spearman correlation between the two layer-wise curves.

DINOv2-S/B/L/g using the same probing protocol.

Canonical-color accuracy increases monotonically with scale in all three input settings (Spearman’s $\rho = 1 . 0 0 )$ : from 46.2% to 51.8% for Gray and from 41.8% to 48.0% for Gray+HE. The RGB– Gray gap remains broadly stable across sizes, while entity CLS accuracy follows the same monotonic trend. Thus, canonical-color decodability under grayscale inputs is not specific to a single Basescale operating point and becomes stronger with model scale.

Together, the cross-encoder and scale comparisons establish that canonical color remains linearly accessible after visible color cues are removed. They do not, however, determine whether this accessibility is mediated by object-level recognition; the next section examines this relationship directly.

## 4.4 Relation to Object-Class Recognition

We analyze how color probe and object recognition probe results relate. Both probes are trained on the same patch-averaged representations at each layer, under the grayscale setting. Figure 3 shows that object-class recognition improves with depth for all models, although ViT-MAE remains lower in absolute accuracy. Color probing also tends to improve from early to middle layers, but its layer-wise trajectory does not always match object-class recognition. For example, some models show high color accuracy before object recognition reaches its peak, while others show a late-layer drop in color probing despite continued gains in object recognition. This suggests that color-predictive information is associated with object-level representations, but is not fully determined by object-class recognition accuracy alone.

![](images/24d5afd7e49e6659891985c18726a80f93c8e8edd40b252d97dfa060873257fb.jpg)  
Figure 4: Predicted-object color accuracy under grayscale inputs. Bars report the percentage of test entities for which the color probe predicts the canonical color of the object predicted by the object probe. Error bars show ±1 standard deviation across five folds.

We measure Spearman correlation as a measure of layer-wise association between the two accuracy curves (shown in Figure 3), to establish whether layers with higher object-class recognition accuracy also tend to have higher color probing accuracy. All models show a strong positive correlation, with CLIP and DINOv2 having the strongest layerwise alignment. This suggests that color-predictive information and object-class information become more accessible in similar parts of the encoder, although the two signals are not perfectly aligned.

Figure 4 measures color accuracy under the object class assumed by the object probe. For each test entity, let oˆ be the object predicted by the object probe, cˆ be the color predicted by the color probe, and g(ˆo) be the canonical color of the predicted object. We count the prediction as correct when ${ \hat { c } } = g ( { \hat { o } } )$ . This metric does not require oˆ to match the ground-truth object; it asks whether the color prediction is correct relative to the model’s own object prediction.

The score increases in later layers for most encoders, reaching around 55–65% for CLIP, SigLIP, $\mathrm { D I N O v } 2 .$ and Swin-V2, with ViT-MAE lower. This provides stronger evidence that the color probe often tracks object-level canonical color information, rather than behaving independently of object identity. At the same time, the scores are not near ceiling, which suggests that the encoders do not expose canonical color knowledge equally well for all object classes.

## 5 VLM Post-training and the Location of Semantic Decodability

We next ask whether VLM post-training changes where canonical color and object information are linearly decodable. We do not evaluate final VLM answers, since they can be confounded by decoderside world knowledge. Instead, we probe frozen representations across matched pre-/post-VLM vision towers and decoder-side visual-token states.

## 5.1 Model-dependent changes in the vision tower

We first test whether post-training changes linear accessibility in the standalone vision tower. We compare three matched pre-/post-VLM pairs: OpenAI CLIP ViT-L/14@336px versus the CLIP vision tower in Molmo (Deitke et al., 2025), OpenAI CLIP ViT-L/14@224 versus the CLIP vision tower in mPLUG-Owl (Ye et al., 2023), and SigLIP-So400m@224 versus the SigLIP vision tower in PaliGemma (Beyer et al., 2024). Within each pair, the pre- and post-VLM encoders are evaluated at the input resolution used by the VLM; across pairs, we compare the direction of change rather than absolute accuracy. Model repositories are provided in Appendix A.

All models are evaluated under the same grayscale setting. For each model and layer, we extract patch-averaged features and train linear probes on frozen representations. Canonical-color probing uses object-class-level cross-validation, while object recognition uses the same image-level entityrecognition protocol as before.

The red- and blue-circle curves in Figure 5 show that post-training has model-dependent effects on vision-tower decodability. In the two CLIP-based pairs, Molmo and mPLUG-Owl, the post-VLM vision towers yield lower linear-probe accuracy than the corresponding original CLIP encoders. The reduction is moderate for canonical color but substantially larger for object recognition. Thus, in these two models, post-training makes object-level information less linearly accessible from the standalone vision tower.

![](images/e16dcd43f8c632921c8ccdc485acb1c766de6f23003754bb67a83a1c59a08c78.jpg)  
Figure 5: Probing semantic decodability across vision towers and decoder visual-token states under grayscale inputs. Columns show matched pre-/post-VLM pairs: CLIP–Molmo, CLIP–mPLUG-Owl, and SigLIP–PaliGemma. Top row: canonical-color probing. Bottom row: object recognition. Red circles show the pre-VLM vision encoder. Blue circles show the post-VLM vision tower. Blue squares show the decoder visual-avg readout. Dotted gray lines show blank-image baselines; shaded bands show ±1 standard deviation across folds.

The SigLIP–PaliGemma pair behaves differently. Under the same grayscale probing setup, the post-VLM PaliGemma vision tower closely follows the original SigLIP encoder for both canonical color and object recognition. Reduced standalone vision-tower decodability is therefore not a universal consequence of VLM post-training: the two CLIP-based VLMs show clear reductions, whereas PaliGemma largely preserves the original SigLIP pattern. Detailed numerical comparisons and paired significance tests are reported in Table 5 in Appendix B.

## 5.2 Visual-interface and decoder-side probing

To test the reformatting hypothesis, we ask whether information weakly decodable from the post-VLM vision tower becomes linearly accessible again downstream. Keeping each VLM frozen, we run the same grayscale images through the model, cache hidden states after the visual interface and at each decoder layer, and apply the same linear probes as before. At each decoder layer, we average the hidden states at the visual-token positions passed to the language model, yielding the visualtoken average (visual-avg), the closest decoderside analogue of the patch-averaged vision features.

Decoder layers $d _ { 1 } , \ldots , d _ { N }$ are plotted after the vision layers only to indicate where information becomes linearly accessible in the full stack. A blankimage baseline tests whether the recovered signal depends on visual input.

The blue-square curves in Figure 5 show that object and canonical-color information can become linearly accessible again in decoder-side representations. In Molmo, canonical-color decodability returns close to the original CLIP level from the first decoder layer, while object recognition increases toward the original CLIP peak in later layers. mPLUG-Owl shows the same pattern: despite weak standalone-tower decodability, its decoderside representations recover much of the reduction for both tasks. PaliGemma behaves somewhat differently: its post-VLM vision tower already closely tracks the original SigLIP encoder, so its decoderside readout remains broadly comparable rather than showing a substantial recovery.

To locate where the recovery enters the stack, we additionally probe the mPLUG-Owl visual interface for object recognition.

Figure 6 shows that the post-VLM ViT readout remains low, whereas accuracy is already substantially higher at the Q-Former-style visual abstractor output, before the visual tokens enter the LLM decoder. The first decoder layer changes this value only slightly, while later layers further refine it, suggesting that the main increase occurs at the visual interface rather than in the language decoder alone.

![](images/8e09a666b52c55770558b30be56ba5df71157412c9c66f78609d9a6a4c6d0386.jpg)  
Figure 6: Object-recognition probing along the mPLUG-Owl visual path under grayscale inputs. Red shows the original CLIP encoder; blue shows the post-VLM path through the visual abstractor and LLaMA decoder. The main increase occurs at the visual interface.

This pattern reflects a change in linear accessibility, not proof that the post-VLM ViT contains no object information; a nonlinear probe might recover additional information. We focus on information directly accessible to a simple linear readout to remain consistent with the earlier experiments. The blank-image baseline remains near the majorityclass baseline, suggesting that the recovered signal depends on visual input.

Overall, these results are consistent with reformatting rather than removal. In Molmo and mPLUG-Owl, object and canonical-color information becomes less linearly accessible in the standalone tower but reappears after the learned visual interface and in decoder-side representations. PaliGemma, whose vision tower changes less substantially, largely preserves the original decodability pattern.

## 5.3 Prompt-level behavior under color-controlled inputs

As a final behavioral check, we query the VLMs with color- and object-related prompts. This experiment is not intended to localize color information; instead, it illustrates how final answers can mix visible evidence, object recognition, canonical-color priors, and output formatting.

We evaluate the same 708 object classes after converting all images to grayscale. We use three prompts: Visual Color, “What color is the object in this image? Answer with just one word,” for which the expected answer is gray; Canonical Color, “What is the typical real-world color of this object? Answer with just one word,” which targets typical rather than visible color; and Object, “What object is shown in this image? Answer with one or two words,” which tests object identification.

<table><tr><td>Metric</td><td>Molmo</td><td>mPLUG</td><td>PaliGemma</td></tr><tr><td>Visual color</td><td>23.4</td><td>14.9</td><td>17.3</td></tr><tr><td>Canonical color</td><td>44.5</td><td>25.6</td><td>4.1</td></tr><tr><td>Object category</td><td>53.0</td><td>57.1</td><td>61.3</td></tr><tr><td>Dec. color probe</td><td>42.7</td><td>40.6</td><td>43.2</td></tr><tr><td>Dec. object probe</td><td>59.7</td><td>57.5</td><td>69.9</td></tr></table>

Table 2: Prompt-level results on grayscale images. The first three rows report VLM answer accuracy; the last two report peak decoder visual-avg probe accuracy.

For the two color prompts, we obtain class-level predictions by majority vote over images of the same class. Because exact matching to Wikidata names is too strict for fine-grained, scientific, and proper names, we evaluate object recognition using manually verified acceptable labels derived from the entity name and Wikidata description. VLM outputs are not used to construct these label sets.

Table 2 shows that final answers are not a clean measure of visual color perception. All models perform poorly on Visual Color despite gray being the expected answer, suggesting effects from object-level color priors or response preferences. Canonical Color improves performance substantially for Molmo and also for mPLUG-Owl, while PaliGemma performs poorly. Category-level object recognition is higher but remains imperfect under grayscale inputs. Thus, final answers conflate visual evidence, object recognition, priors, and output normalization, whereas internal visual-token probes provide a cleaner view of where object and canonical-color information are linearly decodable.

## 6 Conclusion

Using canonical color as a controlled probe, we find that it remains linearly decodable from grayscale images even though RGB probing is strongly influenced by visible color. Counterfactual and object probes show that this signal aligns more closely with canonical than surface color and is related to, but not fully determined by, object identity. VLM post-training can reduce tower-level decodability while the information reappears after the visual interface or within decoder-side states, a pattern consistent with representational reformatting rather than removal and illustrating how canonical color information traces object-level semantics across the vision–language stack.

## Limitations

Our study focuses on canonical color as a controlled case study. Color is useful because visible chromatic cues can be removed with grayscale conversion and histogram equalization, but this also means that our empirical conclusions should not be directly generalized to all visual attributes. Other attributes, such as material, texture, function, or affordance, may require different controls and evaluation protocols.

Our analysis is also based on linear probing. This allows us to compare representations across models and layers in a simple and consistent way, but it only measures information that is linearly accessible from the chosen readout. A nonlinear probe or a downstream task-specific head might recover additional information. Therefore, our results should be interpreted as evidence about linear decodability, rather than as a complete measure of all information contained in the representations.

## Acknowledgments

Stella Frank was funded by NNF project 0094281.

## References

Guillaume Alain and Yoshua Bengio. 2017. Understanding intermediate layers using linear classifier probes. In International Conference on Learning Representations Workshop Track.

Michael M Bannert and Andreas Bartels. 2013. Decoding the yellow of a gray banana. Current Biology, 23(22):2268–2272.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, and 1 others. 2024. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726.

Elia Bruni, Nam Khanh Tran, and Marco Baroni. 2014. Multimodal distributional semantics. Journal ofArtificial Intelligence Research, 49:1–47.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, and 1 others. 2025. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 91–104.

Barry J. Devereux, Lorraine K. Tyler, Jeroen Geertzen, and Billi Randall. 2014. The centre for speech, language and the brain (CSLB) concept property norms. Behavior Research Methods, 46(4):1119–1127.

Michal Golovanevsky, William Rudman, Michael A Lepori, Amir Bar, Ritambhara Singh, and Carsten Eickhoff. 2025. Pixels versus priors: Controlling knowledge priors in vision-language models through visual counterfacts. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24848–24863.

Jonathan Gordon and Benjamin Van Durme. 2013. Reporting bias and knowledge acquisition. In Proceedings ofthe 2013 workshop on Automated knowledge base construction, pages 25–30.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Thorsten Hansen, Maria Olkkonen, Sebastian Walter, and Karl R. Gegenfurtner. 2006. Memory modulates color appearance. Nature Neuroscience, 9(11):1367– 1368.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. 2022. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743, Hong Kong, China. Association for Computational Linguistics.

Douwe Kiela and Léon Bottou. 2014. Learning image embeddings using convolutional neural networks for improved multi-modal semantics. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 36–45, Doha, Qatar. Association for Computational Linguistics.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, Singapore. Association for Computational Linguistics.

Yijun Liang, Ming Li, Chenrui Fan, Ziyue Li, Dang Nguyen, Kwesi Cobbina, Shweta Bhardwaj, Jiuhai Chen, Fuxiao Liu, and Tianyi Zhou. 2025. Colorbench: Can vlms see and understand the colorful world? a comprehensive benchmark for color perception, reasoning, and robustness. arXiv preprint arXiv:2504.10514.

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. 2022. Swin transformer v2: Scaling up capacity and resolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12009– 12019.

Ken McRae, George S. Cree, Mark S. Seidenberg, and Chris McNorgan. 2005. Semantic feature production norms for a large set of living and nonliving things. Behavior Research Methods, 37(4):547–559.

Dan Onea¸ta, Desmond Elliott, and Stella Frank. 2025.˘ Seeing what tastes good: Revisiting multimodal distributional semantics in the billion parameter era. In Findings of the Association for Computational Linguistics: ACL 2025, pages 24174–24191.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, and 7 others. 2024. DINOv2: Learning robust visual features without supervision. In Transactions on Machine Learning Research.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium. Association for Computational Linguistics.

Carina Silberer and Mirella Lapata. 2014. Learning grounded meaning representations with autoencoders. In Proceedings of the 52nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 721–732, Baltimore, Maryland. Association for Computational Linguistics.

James W. Tanaka and Lynn M. Presnell. 1999. Color diagnosticity in object recognition. Perception & Psychophysics, 61(6):1140–1153.

Yingtian Tang, Yutaro Yamada, Yoyo Zhang, and Ilker Yildirim. 2023. When are lemons purple? the concept association bias of vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14333–14348, Singapore. Association for Computational Linguistics.

Christoph Witzel, Hanna Valkova, Thorsten Hansen, and Karl R. Gegenfurtner. 2011. Object knowledge modulates colour appearance. i-Perception, 2(1):13– 49.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, and 1 others. 2023. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178.

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. 2023. When and why vision-language models behave like bags-of-words, and what to do about it? In International Conference on Learning Representations.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. 2022. VL-CheckList: Evaluating pre-trained visionlanguage models with objects, attributes and relations. arXiv preprint arXiv:2207.00221.

## A Additional Dataset and Model Details

This appendix provides additional details on dataset construction, label sources, evaluation splits, and the model repositories used in our experiments.

## A.1 Dataset Construction and Label Source

The dataset contains 708 object concepts. Canonical-color labels are obtained from the Wikidata color property (P462). We retain only entities associated with a single color value and map the source values to ten basic categories: black, blue, brown, gray, green, pink, purple, red, white, and yellow. This reduced label space makes the probing task tractable and allows direct comparison across object classes. The canonical-color labels are therefore derived from structured Wikidata entries rather than assigned through free-form human annotation.

For each object concept, we collect images linked from its Wikidata entry and supplement them with Google Search results, with a target of five images per concept. Manual inspection is used only for image quality control and does not assign or revise the canonical-color labels. We remove images that do not clearly depict the target object or are inconsistent with its canonical color. Because filtering is performed after image collection, some concepts contain fewer than five images.

## A.2 Class Distribution and Sampling

We retain the resulting imbalance across canonicalcolor labels rather than forcing all categories to have the same size. During collection, Pokémon formed a large subset of the candidate concepts. We therefore cap this subset at 50 concepts while prioritizing the less frequent gray, brown, and red categories. The final dataset contains 69 black, 69 blue, 53 brown, 48 gray, 73 green, 81 pink, 65 purple, 61 red, 99 white, and 90 yellow concepts. White is the most frequent label and gray is the least frequent.

## A.3 Evaluation Splits and Variable Image Counts

For color probing, we use object-class-level fivefold cross-validation. All images of the same object concept are assigned to the same fold. In each run, the probe is trained on four folds and evaluated on object concepts from the held-out fold. The test classes are therefore unseen during training, so this setting evaluates canonical-color generalization to new object concepts rather than to new images of known concepts.

![](images/acf3ec4f59fe689daafd4fbebc2ac919da9c6e1e4bc9189c7cb738a2d6a0561a.jpg)  
Figure 7: Distribution of the 708 object concepts across the ten canonical-color labels.

At test time, we average the predicted colorprobability vectors over all available images of each object concept and take the class with the highest average probability. This produces one color prediction per object concept regardless of its number of images. We report results against the approximately 14% majority-class baseline. The cross-validation procedure defines the held-out object concepts but does not rebalance the color-label distribution.

For object recognition, we instead use a withinclass image hold-out because holding out an entire object class would remove its target label from training. For each concept with at least two images, we hold out one image for testing and use the remaining images for training. Concepts represented by a single image are used for training only.

## A.4 Model Repositories

## B Numerical Results and Statistical Tests

This section provides numerical summaries of the layer-wise comparisons in the main text. All results use histogram-equalized grayscale inputs and five folds. For the encoder comparison, we report bestlayer accuracy with 95% confidence intervals, onesample t-tests against the 13.98% majority baseline, and paired t-tests for the RGB-to-grayscale difference. For the matched VLM pairs, we report best-layer pre-/post-training differences and paired t-tests over the five folds.

Vision encoders. Table 4 reports the best-layer canonical-color results for the five vision encoders. All grayscale accuracies are significantly above the majority baseline, while all RGB-to-grayscale reductions are also significant.

<table><tr><td>Role</td><td>Model</td><td>Hugging Face repository</td></tr><tr><td>Vision encoder</td><td>CLIP ViT-B/32</td><td>https://huggingface.co/openai/clip-vit-base-patch32</td></tr><tr><td>Vision encoder</td><td>SigLIP Base</td><td>https://huggingface.co/google/siglip-base-patch16-224</td></tr><tr><td>Vision encoder</td><td>DINOv2 Base</td><td>https://huggingface.co/facebook/dinov2-base</td></tr><tr><td>Vision encoder</td><td>ViT-MAE Base</td><td>https://huggingface.co/facebook/vit-mae-base</td></tr><tr><td>Vision encoder</td><td>Swin-V2 Base</td><td>https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k</td></tr><tr><td>Pre-VLM vision tower</td><td>CLIP ViT-L/14@336px</td><td>https://huggingface.co/openai/clip-vit-large-patch14-336</td></tr><tr><td>Pre-VLM vision tower Pre-VLM vision tower</td><td>CLIP ViT-L/14@224px SigLIP-So400m@224px</td><td>https://huggingface.co/openai  $/ { \mathsf { c l i p - v i t - l a r g e - p a t c h } } 1 4$ </td></tr><tr><td></td><td></td><td>https://huggingface.co/google/siglip-so400m-patch14-224</td></tr><tr><td>VLM</td><td>Molmo-7B-D</td><td>https://huggingface.co/allenai/Molmo-7B-D-0924</td></tr><tr><td>VLM VLM</td><td>mPLUG-Owl</td><td>https://huggingface.co/MAGAer13/mplug-owl-1lama-7b</td></tr><tr><td></td><td>PaliGemma 3B Mix 224</td><td>https://huggingface.co/google/paligemma-3b-mix-224</td></tr></table>

Table 3: Hugging Face model repositories used in our experiments.
<table><tr><td>Encoder</td><td>Gray Acc.</td><td>95% CI</td><td>vs. 13.98%</td><td>RGB→Gray Drop</td></tr><tr><td>CLIP</td><td>41.5%</td><td>[38.4, 44.6]</td><td> $p < . 0 0 1$ </td><td> $3 4 . 9 \mathrm { p p } ( p < . 0 0 1 )$ </td></tr><tr><td>DINOv2</td><td>43.8%</td><td>[37.0, 50.7]</td><td> $p < . 0 0 1$ </td><td> $3 2 . 8 \mathrm { p p } ( p < . 0 0 1 )$ </td></tr><tr><td>SigLIP</td><td>42.4%</td><td>[37.7, 47.2]</td><td> $p < . 0 0 1$ </td><td> $3 9 . 5 \mathrm { p p } ( p < . 0 0 1 )$ </td></tr><tr><td>ViT-MAE</td><td>40.3%</td><td>[35.3, 45.4]</td><td> $p < . 0 0 1$ </td><td> $3 5 . 3 \mathrm { p p } ( p < . 0 0 1 )$ </td></tr><tr><td>Swin-V2</td><td>40.2%</td><td>[35.8, 44.6]</td><td> $p < . 0 0 1$ </td><td> $3 7 . 1 \mathrm { p p } ( p < . 0 0 1 )$ </td></tr></table>

Table 4: Best-layer canonical-color probing under histogram-equalized grayscale inputs. Confidence intervals are computed over five folds. Significance relative to the 13.98% majority baseline is assessed using onesample t-tests; RGB-to-grayscale differences use paired t-tests. Here, pp denotes percentage points.
<table><tr><td>VLM Pair</td><td>Task</td><td>Pre-VLM</td><td>Post-VLM</td><td>Change</td><td>Paired Test</td></tr><tr><td>CLIP→Molmo</td><td>Color</td><td>43.9%</td><td>36.8%</td><td>-7.1 pp</td><td> $p < . 0 5$ </td></tr><tr><td>CLIP→Molmo</td><td>Object</td><td>65.0%</td><td>26.2%</td><td>-38.8 pp</td><td> $p < . 0 0 1$ </td></tr><tr><td>CLIP→mPLUG-Owl</td><td>Color</td><td>43.0%</td><td>30.8%</td><td>-12.3 pp</td><td> $p < . 0 5$ </td></tr><tr><td>CLIP→mPLUG-Owl</td><td>Object</td><td>64.4%</td><td>7.0%</td><td>-57.4pp</td><td> $p < . 0 0 1$ </td></tr><tr><td>SigLIP→PaliGemma</td><td>Color</td><td>43.3%</td><td>42.9%</td><td>−0.4 pp</td><td>n.s.</td></tr><tr><td>SigLIP→PaliGemma</td><td>Object</td><td>73.3%</td><td>71.5%</td><td>−1.9 pp</td><td> $p < . 0 0 1$ </td></tr></table>

Table 5: Best-layer grayscale probing accuracy for matched standalone vision towers before and after VLM post-training. Significance is assessed using paired t-tests over five folds. Statistical significance should be interpreted together with effect magnitude: the PaliGemma object difference is consistent across folds but only 1.9 percentage points.

Pre-/post-VLM vision towers. Table 5 compares the standalone vision towers before and after VLM post-training. The two CLIP-based pairs show substantial reductions, particularly for object recognition, whereas the SigLIP–PaliGemma pair changes little in absolute magnitude.