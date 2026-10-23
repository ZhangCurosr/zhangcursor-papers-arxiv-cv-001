# Class-Grounded Batik Captioning under Scarce Text Annotation: Label Retention and Source-Specific Error Propagation

Novanto Yudistira, Muhammad Dzaka Aufa Fadhillah, and Irawati Nurmala Sari

Universitas Brawijaya, Malang, Indonesia

Preprint. 22 October 2026.

Abstract. Class labels ofer compact supervision for describing culturalheritage images when expert text annotations are scarce. We investigate a batik-captioning cascade that integrates a LoRA-adapted DINOv2 classifier with Qwen2.5-VL-7B, adapted using LoRA and loaded in 4-bit NF4. A 3 × 2 evaluation combines three captioner conditions with two inference prompts to examine adaptation and predicted-class injection. Across 150 caption-evaluation images, all three injected conditions achieve a ground-truth class-label mention rate of 0.9067. The classifier’s separate test yields 96.10% pooled accuracy, with 98.33% on 300 public-source images and 75.76% on 33 local images. Supported-class macro-F1 and balanced accuracy on the local subset are 0.8493 and 0.8302, respectively. The classifier and captioner use separate partitions, and their training– evaluation overlap remains unquantified. Diferences in label semantics and training support also afect interpretation of the source-specific accuracy gap. Reference-based caption metrics and image–text compatibility scores produce diferent model rankings. Qualitative examples show both retention of correctly predicted identities and propagation of classification errors into fluent descriptions. These findings characterize how class conditioning preserves label information and transfers upstream errors into batik captions under scarce text annotation.

Keywords: batik · cultural heritage · image captioning · class injection · evaluation · vision–language models

## 1 Introduction

Cultural-heritage digitization frequently produces image collections with category labels and limited descriptive text. Batik archives exemplify this setting: a catalogue may identify a regional class or motif while providing sparse information about visible colors, ornaments, and spatial arrangement. Generating useful descriptions therefore requires a model to connect visual appearance with collection-specific terminology.

Figure 1 introduces three batik images with their annotated regional classes and caption excerpts generated using predicted-class conditioning. The examples connect the captioning task to visible fabric patterns and illustrate how a classifier prediction can determine the cultural identity expressed in the text.

(a) Tasik Madu  
![](images/232a6200b0dc6492d0872097319af3bb89231ae95496f3e73fa26e38d97dc8f9.jpg)  
Annotated class: Malang

(b) Bunga Nirwana 2  
![](images/ee10886eddbb74fb0678e1928b61cf612d80477cace4fe5b94d6d24fbd25e808.jpg)  
(c) Melati Deret  
Annotated class: Tulungagung

![](images/b7cdbdadaa417ff93549e8fc9deb8d72b53540b9ec1969476040be4bfc5271dc.jpg)  
Annotated class: Lamongan

Generated caption excerpt: “the batik fabric features the malang motif, characterized by white motifs against a deep blue background. the dominant motif elements include stylized bees and honeycomb structures.”

Generated caption   
excerpt:   
“the batik fabric features   
the trenggalek motif,   
characterized by red, blue,   
and white motifs. the   
dominant motif elements   
appear to be stylized leaves   
and flowers.”

Generated caption excerpt: “the batik fabric features the trenggalek motif, characterized by deep red motifs. the dominant motif elements appear to be small, intricate designs that resemble traditional indonesian patterns.”

Fig. 1. Batik images and generated caption excerpts from Dataset A. Bold class names indicate the model’s cultural attribution. The Malang identity in (a) agrees with the annotation; the Trenggalek identities in (b) and (c) conflict with the annotated Tulungagung and Lamongan classes. All outputs use the captioner adapted with class labels and a predicted-class inference prompt (CIFT-CI, defined in Section 4.2).

A vision–language model (VLM) can generate fluent descriptions, whereas reliable cultural attribution depends on recognizing the relevant class. These requirements motivate a two-stage framework in which a visual classifier predicts a cultural class and a captioner uses that prediction as contextual information. Previous studies have examined batik recognition and synthesis [7,14,18], culturally informed captioning [20], and classifier-guided radiology captioning [5].

Class conditioning also links the errors of the two stages. A correct prediction can introduce a collection-specific identity into the caption, while an incorrect prediction can produce a fluent description with an erroneous identity. Evaluation consequently needs to distinguish label retention, agreement with reference text, visual compatibility, and factual correctness. Repeated reference templates introduce a further challenge because lexical similarity can reward recurring expressions across visually diferent images.

We investigate these efects through a DINOv2–Qwen2.5-VL cascade. The evaluation crosses three captioner conditions with two inference prompts and examines class-label mention, caption metrics, classifier performance, and quali tative error patterns. The study contributes:

a six-condition evaluation of captioner adaptation and inference-time class injection under scarce text annotation;

an analysis of source-specific and class-specific classifier performance with explicit metric-averaging conventions; and

an examination of label retention, caption-metric disagreement, downstream identity errors, and resource use.

The evaluation focuses on the behavior of class-conditioned caption generation under independently constructed classifier and captioner partitions. Section 5 defines this evaluation scope, and Section 7 discusses its implications for completecascade generalization.

## 2 Related Work

Batik recognition research has explored transfer learning [7], while difusion and GAN-based approaches address motif synthesis [14,18]. Indonesian image captioning has also been studied using ViT–IndoBERT on general image–text data [2]. Our study examines the connection between cultural-class recognition and generated descriptions of batik images.

Culturally aware captioning incorporates external contextual information [20]. Classifier-guided captioning provides a related interface by passing compact domain information to a language-generation stage [5]. We use this interface to analyze the retention of predicted identities and the propagation of classification errors.

DINOv2 provides transferable visual representations [15], and Qwen2-VL and Qwen2.5-VL support multimodal instruction following [19,3]. LoRA reduces the number of parameters updated during adaptation [9], while quantized adaptation supports lower-memory model loading [6]. These components enable the classifier and captioner to be adapted separately within the cascade.

Reference-based metrics quantify lexical or semantic agreement with target captions [16,11,4,1,21]. CLIPScore evaluates image–text compatibility [8]. Research on hallucination further motivates direct examination of the visual assertions made in generated captions [17,10]. Together, these perspectives support evaluating cultural identity and visible attributes as distinct aspects of caption content.

## 3 Data and Task

## 3.1 Image collections and cultural labels

Dataset A contains 346 locally curated East-Java images assigned to four regional classes: Lamongan, Malang, Trenggalek, and Tulungagung. Human-curated metadata describes the class, colors, visually grounded theme, and pattern shape. Reference captions are constructed from these fields using a recurring template.<sup>1</sup> The reference text is therefore template-based, with descriptive content limited to the selected metadata fields. Historical interpretation, symbolism, fabric type, dyes, and production techniques are excluded from the caption target.

Dataset B comprises 3,000 images from 20 classes, with 150 images per class. The data documentation identifies Kaggle as the host platform; the repository identifier, creator, release, and license remain unspecified. This incomplete provenance limits reconstruction of the collection. Appendix A lists the 20 class names.

The combined inventory contains 24 cultural classes. Dataset A uses regional labels, while Dataset B primarily uses motif names together with several regional categories. Both collections contribute to training. Comparisons between sources consequently reflect diferences in label semantics, sample support, and potentially acquisition conditions.

Table 1 summarizes the collection sizes, label inventories, supervision, and component-specific data allocations.

Table 1. Data allocation and supervision for the classifier and captioner. The two components use separately constructed partitions.
<table><tr><td colspan="3">Source Images Labels Allocation / supervision</td></tr><tr><td>A</td><td>346</td><td>4 Captioner: 163 train, 33 valida- tion, 150 test; metadata-derived</td></tr><tr><td>B</td><td>3,000</td><td>references. Classifier test: 33. 20 Captioner: 2,700 train, 300 vali- dation; pseudo-captions. Classifier test: 300.</td></tr></table>

## 3.2 Class-conditioned caption generation

Figure 2 illustrates the classifier–captioner cascade and its optional class input. For image x, the classifier predicts $\hat { y } = \arg \operatorname* { m a x } _ { y } p _ { \theta } ( y \mid x )$ . The captioner generates $c = g _ { \phi } ( x , p )$ using either a generic prompt p<sub>0</sub> in the no-class-injection (NCI) condition or a class-conditioned prompt $p ( \hat { y } )$ in the class-injection (CI) condition. Paired prompt conditions use the same image and captioner. The resulting comparison characterizes the efect of predicted-class conditioning on generated text.

![](images/c88557d1e72681fe000d3d0abeb233f8e2fab48409a06e43c0bf403f8c3202b4.jpg)  
Fig. 2. Batik-captioning cascade with optional class injection. The classifier prediction provides contextual information to the captioner and forms a pathway for downstream identity errors.

## 4 Methodology

## 4.1 Pseudo-caption generation

For Dataset B, zero-shot pseudo-captions are generated with Qwen2.5-VL-7B-Instruct conditioned on each image and its class label before domain adaptation. The generation prompt requests dominant colors, visible elements, spatial arrangement, and visible isen-isen, while excluding unsupported historical or symbolic interpretations. It also requires the class label in the opening sentence (Appendix B), directly encouraging identity mention.

Manual review removes or regenerates captions with incorrect colors, classinconsistent descriptions, or unsupported cultural assertions. The resulting corpus contains 3,000 image–caption pairs, comprising 2,700 training pairs and 300 validation pairs.

Pseudo-caption generation disables sampling and uses a maximum of 256 new tokens and a repetition penalty of 1.1. Inference-time caption generation enables sampling with a temperature of 0.3 and a maximum of 256 new tokens.

## 4.2 Classifier and captioner adaptation

The DINOv2 backbone is frozen except for LoRA modules and a trained 24- way linear classification head. LoRA targets the query, key, value, and dense projections, using rank 32, scaling 64, and dropout 0.1. Training uses AdamW [13], a learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 4 }$ , 10 epochs, cosine scheduling,

10% warmup, batch size 16, and gradient accumulation over two batches. The objective is a weighted focal-style loss with γ = 2 [12].

Qwen2.5-VL-7B-Instruct is loaded with NF4 and double quantization. LoRA targets the {q, k, v, o, gate, up, down} projections, with rank 16, scaling 32, and dropout 0.05. Adaptation uses AdamW, a learning rate of $1 0 ^ { - 4 }$ , 10 epochs, cosine scheduling, 5% warmup, batch size 8, and gradient accumulation over four batches. Instruction and image tokens are masked from the answer-token cross-entropy objective. Dynamic-resolution processing uses a 262,144-pixel ceiling.

The three captioner conditions are Baseline, the pretrained model; PFT, adapted using training prompts that omit class labels; and CIFT, adapted using training prompts that include class labels. Crossing these conditions with NCI and CI produces the six-condition evaluation.

## 5 Experimental Setup and Evaluation Protocol

## 5.1 Data partitions and evaluation scope

The classifier uses an approximately 80/10/10 stratified split of the combined collection, with 333 test images. The captioner uses the allocations in Table 1, including 150 Dataset-A images for caption evaluation. These partitions were constructed independently, creating a risk that the classifier encountered some caption-evaluation images during training. Image-level partition assignments and duplicate-group identifiers are unavailable, leaving the extent of this overlap unquantified.

The 150-image caption evaluation characterizes generation under the implemented cascade protocol. Its interpretation as complete-cascade generalization requires a shared partition that excludes evaluation images from both components. This also requires exact- and near-duplicate grouping before assignment, followed by training and model selection using the corresponding training and validation partitions.

## 5.2 Evaluation metrics

Classifier accuracy measures the fraction of correct class predictions. Supportedclass macro-F1 averages per-class F1 over labels with positive ground-truth support in each evaluated source. Predictions into labels outside that supported set remain classification errors. Balanced accuracy averages recall over the supported labels. Class supports and the averaging set are reported alongside the aggregate metrics.

BLEU-1/4, ROUGE-L, METEOR, SPICE, and BERTScore quantify agreement with the reference captions. The recurring reference template makes these measures sensitive to shared syntax and repeated nouns. Metric-library versions, tokenization, BLEU smoothing and aggregation, the BERTScore backbone, and normalization details remain unspecified, limiting independent reproduction of the descriptive scores.

Motif Mention Rate (MMR) quantifies ground-truth class-label mention:

$$
\mathrm { M M R } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \bf 1 } \{ \mathrm { c l a s s ~ s t r i n g ~ } y _ { i } \mathrm { ~ o c c u r s ~ i n ~ c a p t i o n ~ } c _ { i } \} .\tag{1}
$$

Each image contributes a binary outcome. Under class injection, MMR reflects both upstream label correctness and the captioner’s retention of the predicted label. Its scope is class-string inclusion; factual evaluation of the remaining caption content requires attribute-level judgments. Reproduction of MMR also requires explicit string normalization and class-alias rules.

CLIPScore is used as an image–text compatibility diagnostic. Its encoder checkpoint and score scaling remain unspecified. Rare local class names can be weakly represented in the embedding space, so interpretation emphasizes score disagreement alongside direct inspection of caption content.

## 5.3 Computing environment and variability

The computing environment comprises an NVIDIA L40, an AMD EPYC 7413, PyTorch 2.5.1, Transformers 4.52.1, PEFT 0.10.0, and BitsAndBytes 0.49.2. Latency is measured on 30 images and summarized using component means and standard deviations across images. These standard deviations describe image-level runtime variability. Complete per-image caption outputs and repeated-training summaries are unavailable; the analysis consequently uses descriptive comparisons, with paired uncertainty and training-run variability remaining open evaluation requirements.

## 6 Results

## 6.1 Classifier performance across sources

Table 2 reports pooled and source-specific classifier accuracy. The classifier reaches pooled accuracy of 96.10%. Dataset B contributes 300 of the 333 test images and attains 98.33% accuracy, while Dataset A attains 75.76% on 33 images.

Table 2. Classifier accuracy by source. Both image sources are represented during training, and the source subsets use diferent cultural-label inventories.
<table><tr><td colspan="3">Subset Test images Correct Accuracy</td></tr><tr><td>Pooled</td><td>333 320</td><td>0.9610</td></tr><tr><td>B</td><td>300 295</td><td>0.9833</td></tr><tr><td>A</td><td>33 25</td><td>0.7576</td></tr></table>

The diference between the displayed source-specific percentages is 22.57 percentage points. Its interpretation depends on the difering label semantics, class supports, and acquisition conditions of the two collections.

Table 3 gives class-level metrics computed from the Dataset-A confusion matrix. Supported-class macro-F1 is 0.8493, weighted F1 is 0.8063, and balanced accuracy is 0.8302.

Table 3. Dataset-A metrics computed from the confusion matrix. The averaging set comprises the four supported regional classes; the final rows summarize predictions into other labels and the total error count.
<table><tr><td>True label</td><td>Support Precision Recall</td><td></td></tr><tr><td>Lamongan</td><td>10</td><td>0.8889 0.8000 0.8421</td></tr><tr><td>Malang</td><td>16</td><td>0.9167 0.6875 0.7857</td></tr><tr><td>Trenggalek</td><td>6</td><td>0.71430.8333 0.7692</td></tr><tr><td>Tulungagung</td><td>1</td><td>1.0000 1.0000 1.0000</td></tr><tr><td>Supported-class mean</td><td>33</td><td>0.8800 0.8302 0.8493</td></tr><tr><td colspan="3">Predictions to B labels: Betawi 1, Dayak 1, Parang 1, Sekarjagad 1. Other errors: four confusions among A labels. Total errors: 8/33.</td></tr></table>

The four additional predicted labels, Betawi, Dayak, Parang, and Sekarjagad, have zero ground-truth support in this subset. Including their four zero F1 values in the averaging set produces an eight-label macro-F1 of 0.4246, or 0.42 at twodecimal precision. These values illustrate the efect of the averaging inventory when a source subset receives predictions from a larger class space.

Tulungagung contributes a single test image, making its perfect score sensitive to individual outcomes. Appendix A gives the Dataset-B per-class results at the two-decimal precision of the classification report.

## 6.2 Captioner adaptation and class-label retention

Table 4 compares reference-based caption metrics across the six conditions.   
PFT-NCI achieves the highest scores on all six measures.

Table 4. Reference-based caption metrics on 150 Dataset-A images. Columns compare captioner adaptation and inference-time class injection under the evaluation protocol in Section 5.
<table><tr><td rowspan="2">Metric</td><td colspan="2">Baseline</td><td colspan="2">PFT</td><td colspan="2">CIFT</td></tr><tr><td>NCI</td><td>CI</td><td>NCI</td><td>CI</td><td>NCI</td><td>CI</td></tr><tr><td>BLEU-1</td><td>.4531</td><td>.4438</td><td>.6058</td><td>.4490</td><td>.4444</td><td>.4447</td></tr><tr><td>BLEU-4</td><td>.2004</td><td>.2639</td><td>.4415</td><td>.2675</td><td>.2020</td><td>.2698</td></tr><tr><td>ROUGE-L</td><td>.4296</td><td>.4674</td><td>.6775</td><td>.4696</td><td>.4218</td><td>.4761</td></tr><tr><td>METEOR</td><td>.5419</td><td>.5502</td><td>.7307</td><td>.5781</td><td>.5424</td><td>.5738</td></tr><tr><td>SPICE</td><td>.3152</td><td>.3701</td><td>.5249</td><td>.3761</td><td>.3160</td><td>.3714</td></tr><tr><td>BERTScore</td><td>.9108</td><td>.9147</td><td>.9363</td><td>.9102</td><td>.9051</td><td>.9110</td></tr></table>

Table 5 reports class-label mention and image–text compatibility. All three CI conditions attain MMR of 0.9067. Class-label mention is therefore invariant across the captioner variants under the injected prompt. This pattern is consistent with an interface that strongly encourages reproduction of the predicted label.

Table 5. Class-label mention and image–text compatibility on the caption-evaluation set. MMR measures ground-truth class-string inclusion; CLIPScore is shown in the evaluation’s numerical scale.
<table><tr><td rowspan="2">Diagnostic</td><td colspan="2">Baseline</td><td colspan="2">PFT</td><td colspan="2">CIFT</td></tr><tr><td>NCI</td><td>CI</td><td>NCI</td><td>CI</td><td>NCI</td><td>CI</td></tr><tr><td>MMR</td><td>.0000</td><td>.9067</td><td>.3800</td><td>.9067</td><td>.0000</td><td>.9067</td></tr><tr><td>CLIPScore</td><td>.3396</td><td>.3375</td><td>.3233</td><td>.3303</td><td>.3393</td><td>.3310</td></tr></table>

CIFT-CI exceeds Baseline-CI in BLEU-4 by 0.0059, while its CLIPScore is lower by 0.0065. These small diferences describe changes in reference agreement and image–text compatibility. Paired uncertainty estimates and independent attribute-level judgments would clarify their relationship to factual correctness.

PFT-NCI’s leading reference-based scores in Table 4 accompany the lowest CLIPScore in Table 5. This disagreement shows that model rankings depend on the evaluation criterion. The qualitative examples provide a complementary view of the visual assertions and cultural identities expressed in the generated text.

## 6.3 Qualitative analysis of caption content and error propagation

Figure 3 presents three image–caption cases across PFT-NCI, CIFT-NCI, and CIFT-CI. The excerpts permit direct comparison of class identity and visible attributes.

![](images/f042773665c9eb16b7fdbcb4bc72156e4fc8244b1d288d4a9da61ca238a9e9a7.jpg)  
Fig. 3. Generated caption excerpts across three model conditions. Tasik Madu illustrates correct identity retention under CI and unsupported visual nouns under PFT-NCI. Bunga Nirwana 2 and Melati Deret illustrate propagation of an incorrect Trenggalek prediction. The cases are selected to illustrate these mechanisms.

Tasik Madu visibly contains bees and honeycomb-like shapes, whereas the PFT-NCI output names lotus, jasmine, and a mask. Its CIFT-CI output includes the correct Malang identity together with bee and honeycomb descriptions.

In the other two cases, CI reproduces the classifier’s Trenggalek prediction despite diferent reference identities. The corresponding NCI excerpts use broader descriptions of flowers, leaves, dots, or layout.

These cases illustrate how class conditioning can preserve an identity or propagate an upstream error. Their selection supports analysis of the mechanisms involved, while limiting conclusions about the frequency of these behaviors across the evaluation set.

## 6.4 Training dynamics with pseudo-caption supervision

Training loss decreases from approximately 2.10 to 0.35 for adaptation using Dataset A alone and from 0.92 to 0.12 for adaptation using A+B. The curves show convergence on both training corpora. Diferences in corpus composition and caption distributions limit interpretation of their absolute loss values as evidence of improved held-out caption quality.

## 6.5 Resource measurements

Table 6 summarizes memory allocation, numerical status, and component latency. Model allocation is 5.4946 GB for NF4 and 9.6551 GB for the BF16/hybrid configuration, a diference of approximately 43%. Peak allocations are 7.2415 GB and 13.1161 GB, respectively.

Table 6. Memory allocation and component latency. NF4 completes inference, while the BF16/hybrid configuration encounters a NaN failure. Latency is summarized as mean and standard deviation across 30 images.
<table><tr><td>Memory (GB)</td><td>NF4 BF16 / hybrid</td></tr><tr><td>Model allocation 5.4946 Peak allocation 7.2415</td><td>9.6551 13.1161</td></tr><tr><td>Numerical status Completed NF4 latency, mean ± SD seconds per image:</td><td>NaN failure</td></tr><tr><td>Image  $\mathrm { I } / \mathrm { O }$ </td><td> $0 . 2 3 0 3 \pm 0 . 1 0 2 6$ </td></tr><tr><td>Classifier</td><td> $0 . 6 2 2 5 \pm 0 . 3 1 1 0$ </td></tr><tr><td>Captioner</td><td></td></tr><tr><td>End-to-end</td><td> $2 1 . 7 5 4 1 \pm 3 . 5 5 1 4$   $2 2 . 6 0 6 9 \pm 3 . 6 8 9 2$ </td></tr></table>

The BF16/hybrid run encounters a NaN failure and includes 134 tensors on a meta/ofloaded path. Caption-quality comparison between the two loading configurations therefore requires a numerically stable BF16 run.

The NF4 cascade takes a mean of 22.6069 seconds per image, of which the captioner accounts for 21.7541 seconds, or approximately 96.23%. Generation is thus the dominant component of the measured runtime. Decoding uses a ceiling of 256 new tokens, and image processing uses a 262,144-pixel cap.

## 7 Discussion and Limitations

All three CI variants attain MMR of 0.9067 despite their diferent adaptation conditions. This invariance is consistent with strong retention of the predicted class under the injected prompt. The qualitative cases demonstrate the dependency on classifier accuracy: the correct Malang prediction enters the Tasik Madu caption, while incorrect Trenggalek predictions enter the descriptions of Bunga Nirwana 2 and Melati Deret.

Label retention and visual factuality describe diferent aspects of caption quality. MMR measures the presence of a class string, while colors, ornaments, and spatial arrangement concern the visual assertions made about the fabric. Tasik Madu illustrates this distinction: the PFT-NCI caption includes the correct regional identity alongside lotus, jasmine, and mask descriptions that conflict with its visible bee and honeycomb forms. The three selected cases support analysis of such errors, while their selection limits inference about the prevalence of factual inaccuracies across the full evaluation set.

The disagreement between reference-based metrics and CLIPScore further qualifies the model rankings. PFT-NCI leads the reference-based measures while attaining the lowest image–text compatibility score. Because the reference captions share a recurring template, agreement in syntax and frequently repeated nouns can contribute to high lexical scores. Image–text compatibility provides a complementary view, although its interpretation depends on the embedding model’s representation of local cultural terminology. These measures therefore capture distinct properties of the generated descriptions.

The independently constructed classifier and captioner partitions restrict conclusions about complete-cascade generalization. Some caption-evaluation images may have entered classifier training, and the overlap is unquantified. The caption results consequently characterize generation under the implemented partition scheme, with independence from both adapted components unresolved. The classifier’s source-specific accuracy gap also combines diferences in regional and motif-based taxonomies, class support, and acquisition conditions. Dataset A’s 33-image classifier test, including a single Tulungagung image, further limits the stability of class-level estimates.

The A-only and A+B loss trajectories characterize adaptation to diferent caption corpora. Lower loss on the expanded corpus is compatible with easier prediction of its textual regularities and provides limited evidence about visual grounding. The contribution of pseudo-caption supervision to held-out generation remains unresolved by the loss comparison. This limits attribution of the sixcondition caption results to the addition of Dataset B.

NF4 supports inference in the evaluated environment, while the BF16/hybrid NaN failure limits caption-quality comparisons. The 30-image latency sample characterizes runtime on the L40 and depends on generated length and image processing. Incomplete Dataset-B provenance and metric-configuration details constrain reproducibility. In collection-management workflows, generated captions serve as draft metadata for expert review of cultural attribution and visible attributes.

## 8 Conclusion

This study examines class-conditioned batik captioning through a DINOv2– Qwen2.5-VL cascade and a six-condition evaluation. Predicted-class injection yields a class-label mention rate of 0.9067 across all three captioner conditions. Source-specific classifier results show a 22.57-percentage-point accuracy gap, whose interpretation depends on the collections’ diferent label inventories and training support. Class-level analysis further demonstrates the influence of the averaging convention on macro-F1.

Caption metrics produce diferent model rankings, and qualitative examples show both correct identity retention and propagation of classification errors. Together, these findings identify the classifier–captioner interface as a key point for evaluating cultural attribution. The results support assessing class-label retention alongside the factual grounding of the visual descriptions, within the generalization limits imposed by the component-specific data partitions.

## References

1. Anderson, P., Fernando, B., Johnson, M., Gould, S.: Spice: Semantic propositional image caption evaluation. In: European Conference on Computer Vision. pp. 382– 398 (2016). https://doi.org/10.1007/978-3-319-46454-1\_24

2. Apandi, A., Mutiara, A.B., Dharmayanti, D.: Generating image captions in indonesian using a deep learning approach based on vision transformer and indobert architectures. Journal of Applied Data Sciences 6(2), 1190–1201 (2025). https://doi.org/10.47738/jads.v6i2.672

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

4. Banerjee, S., Lavie, A.: Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In: ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization. pp. 65–72 (2005)

5. Chowdhury, R.N., Hoque, M., Hasan, M.R., Oluwafemi, E.P.O., Rahman, M.M.: Modality-guided radiology caption prediction with small vision-language models and image classifier. In: CLEF 2025 Working Notes, CEUR Workshop Proceedings. vol. 4038, pp. 2415–2425 (2025)

6. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: Qlora: Eficient finetuning of quantized llms. In: Advances in Neural Information Processing Systems. vol. 36, pp. 10088–10115 (2023)

7. Gultom, Y., Masikome, R.J., Arymurthy, A.M.: Batik classification using deep convolutional network transfer learning. Jurnal Ilmu Komputer dan Informasi 11(2), 59–66 (2018). https://doi.org/10.21609/jiki.v11i2.507

8. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Conference on Empirical Methods in Natural Language Processing. pp. 7514–7528 (2021). https://doi.org/10.18653/ v1/2021.emnlp-main.595

9. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)

10. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. In: Conference on Empirical Methods in Natural Language Processing. pp. 292–305 (2023). https://doi.org/10.18653/ v1/2023.emnlp-main.20

11. Lin, C.Y.: Rouge: A package for automatic evaluation of summaries. In: ACL Workshop on Text Summarization Branches Out. pp. 74–81 (2004)

12. Lin, T.Y., Goyal, P., Girshick, R., He, K., Doll´ar, P.: Focal loss for dense object detection. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 2980–2988 (2017). https://doi.org/10.1109/ICCV.2017.324

13. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

14. Octadion, O., Yudistira, N., Kurnianingtyas, D.: Synthesis of batik motifs using a difusion–generative adversarial network. Multimedia Tools and Applications 84(7), 3407–3438 (2025). https://doi.org/10.1007/s11042-025-20620-9

15. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., et al.: Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024)

16. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: A method for automatic evaluation of machine translation. In: Annual Meeting of the Association for Computational Linguistics. pp. 311–318 (2002). https://doi.org/10.3115/1073083. 1073135

17. Rohrbach, A., Hendricks, L.A., Burns, K., Darrell, T., Saenko, K.: Object hallucination in image captioning. In: Conference on Empirical Methods in Natural Language Processing. pp. 4035–4045 (2018). https://doi.org/10.18653/v1/D18-1437

18. Wahid, R.D.I., Yudistira, N., Dewi, C., Sari, I.N., Pradhikta, D., Fatmawati: Prompt conditioned batik pattern generation using lora weighted difusion model with classifier-free guidance. IEEE Access 13, 2436–2448 (2025). https://doi.org/ 10.1109/ACCESS.2024.3523494

19. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

20. Yun, Y., Kim, J.: Cic: A framework for culturally-aware image captioning. In: Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. pp. 1625–1633 (2024). https://doi.org/10.24963/ijcai.2024/180

21. Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y.: Bertscore: Evaluating text generation with bert. In: International Conference on Learning Representations (2020)

## A Class Inventory and Per-Class Performance

Dataset A comprises Lamongan, Malang, Trenggalek, and Tulungagung. Class indices are assigned by ordering the class labels; reproducing the mapping requires the original label encoding. Table 7 lists the 20 Dataset-B classes and their per-class results.

Table 7. Dataset-B precision, recall, and F1 at the two-decimal precision of the classification report. Each class contains 15 test images.
<table><tr><td>Class</td><td>P</td><td>R F1</td><td>Class</td><td>P</td><td>R F1</td></tr><tr><td>Betawi</td><td>1</td><td>1 1</td><td>Sidoluhur</td><td></td><td>1.87.93</td></tr><tr><td>Bokor Kencono</td><td>1</td><td>1 1</td><td>Sidomukti</td><td>1</td><td>1 1</td></tr><tr><td>Buketan</td><td>1</td><td>1 1</td><td>Sidomulyo</td><td>.88 .93 .90</td><td></td></tr><tr><td>Dayak</td><td>.94</td><td>1.97</td><td>Singa Barong</td><td>1</td><td>1 1</td></tr><tr><td>Jlamprang</td><td>.94</td><td>1.97</td><td>Srikaton</td><td>1</td><td>1 1</td></tr><tr><td>Kawung</td><td></td><td>1.93.97</td><td>Tribusono</td><td>1</td><td>1 1</td></tr><tr><td>Liong</td><td>1</td><td>1 1</td><td>Tujuh Rupa</td><td>1</td><td>1 1</td></tr><tr><td>Mega Mendung</td><td>1</td><td>1 1</td><td>Truntum</td><td>1</td><td>1 1</td></tr><tr><td>Parang</td><td>1</td><td>1 1</td><td>Wahyu Tumurun</td><td>1</td><td>1 1</td></tr><tr><td>Sekarjagad</td><td></td><td>1.93.97</td><td>Wirasat</td><td>.94</td><td>1.97</td></tr></table>

## B Pseudo-Caption Generation Prompt

The following instruction conditions pseudo-caption generation on each image’s class label. The exact evaluation prompts for every NCI and CI condition remain unspecified.

You are an expert batik annotator. Your task is to generate a precise, objective, and highly   
accurate visual description of the provided batik image.   
This batik is classified as the ’{kelas}’ motif.   
Instructions:   
1. Start the first sentence with exactly: ’The batik fabric features the {kelas} motif,   
characterized by [list 2-4 dominant motif colors] motifs.’   
List only the 2-4 most visually dominant motif colors. Do NOT enumerate all visible colors.   
2. In the second sentence, identify the dominant motif elements (e.g. fern plants, fish,   
geometric shapes, floral patterns).   
3. In the third sentence, describe how the motifs are arranged across the fabric (e.g. non  
geometric, diagonal, repeating grid, symmetrical). Include spacing, density, and   
orientation if clearly visible.   
4. If isen-isen (small dots, short lines, or fillers) are clearly visible, include them.   
5. Describe only what is directly visible in the image. Do NOT invent colors not clearly   
present. Do NOT include cultural meanings, historical context, or symbolic   
interpretations.   
6. Output strictly ONE coherent paragraph of 3-4 sentences, without any introductory or   
concluding remarks.