# Counterfactual Analysis of Extent Leakage from Annotation-Derived Prompts in Vision-Language Medical Image Segmentation

Niran Nataraj<sup>a,</sup>∗, Ankita Das<sup>b</sup>, C. Krishna Mohan<sup>b</sup>, Kenji Kawashima<sup>a</sup>

<sup>a</sup>Department ofInformation Physics and Computing, Graduate School ofInformation Science and Technology, The University ofTokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo, 113-8656, Japan

<sup>b</sup>Department of Computer Science and Engineering, Indian Institute of Technology Hyderabad, Kandi, Sangareddy, Telangana, 502284, India

## Abstract

Text-guided segmentation models receive an image and a short caption about the lesion, and on public benchmarks the caption improves accuracy considerably. On colonoscopy and breast ultrasound, however, the size, count and location words of each caption were computed from the mask that later scores the prediction, and the chest X-ray captions were written by benchmark annotators. We tested what two published models take from them. On colonoscopy and chest X-ray, it is mostly the lesion’s extent. Keeping only the size word or the lung-zone phrase recovers 84 to 98 percent of the improvement over an empty caption, and removing it loses 52 to 77 percent. The rest matters little, and a polyp called a rock is segmented just as well. We call this extent leakage. On breast ultrasound, nearly half the captions also name the diagnosis, which carries part of the benefit. Retraining on three datasets shows that models pick up this habit from their training captions, though only partly on breast ultrasound, and that it weakens their segmentation from the image alone. The leak can be measured without training a model, since caption words alone predict lesion area (coeficient of determination 0.54 to 0.77 on six of seven test sets). Without the test mask, the best method we tested, which<sub>V</sub> estimates extent from the image, reaches 74.59 Dice on colonoscopy against the published 77.23. We propose an evaluation protocol that keeps what a caption says about the image apart from what it copies from the label.s

Keywords: Evaluation methodology, Label leakage, Medical image segmentation, Shortcut learning, Vision-language modelse e

## 1. Introduction1

70 Tell a published text-guided segmenter that the polyp in a colonoscopy frame is small, and it outlines a small polyp al-<sub>9</sub> most exactly. Tell it that the same polyp is large, and it buries3 the polyp in false positives (Fig. 1). A model that follows its1 prompt is exactly what text-guided segmentation sets out to build1 (Huang et al., 2025; Koleilat et al., 2025, 2026; Li et al., 2024; Zhong et al., 2023). Alongside the image, the model receives<sup>6</sup> a short text prompt describing the target, such as the number of lesions, their size and their position; we call such prompts captions. Captions provide information about the target and guide the model toward accurate segmentation. Reported results suggest that text can substantially influence segmentation.a MedCLIPSeg, for example, reports that an under-specified caption lowers its overall Dice score by 26.9 points (Koleilat et al., 2026). On its published checkpoints, we find that an empty caption costs 9.11 to 24.61 points, depending on the test set.

Where do these captions come from? For chest X-ray, the authors of LViT (Li et al., 2024) annotated the QaTa-COV19 dataset (Degerli et al., 2021, 2022) with the afected side of the chest, the number of afected regions and the lung zones

![](images/4f63e3866a416a86bcd187d77923cf376d8becee27054dce0b2e761fabc223ca.jpg)  
Figure 1: A published text-guided segmenter reads the lesion’s extent from its caption. Each image is shown with the empty caption and with its own caption in which only the extent has been changed. Top: the size word of a colonoscopy caption, which is computed from the mask. Bottom: the lung-zone phrase of a chest X-ray caption, which was written by annotators. Numbers are Dice scores.

involved. For colonoscopy and breast ultrasound, the captions released with MedCLIPSeg are, image for image, the prompts that Poudel et al. (2024) generated with a fixed template. The template computes the count, size and location words from the reference mask (Fig. 2). A typical caption reads “one small pink circle polyp, located in right of the image”, where small indicates that the mask covers less than 8% of the image. The same captions are given to the model at test time. Its input therefore contains information computed from the very mask that its output is scored against, information that does not exist in clinical use, where the mask is what the model is asked to produce.

![](images/d60c78f88a127e1b597ed0d3657810ad0c415848a5fb0008c93f54614080cddf.jpg)  
Figure 2: How a benchmark caption is written. (a) A CVC-ColonDB test image and its caption. A template fills in the count, size and location words from the reference mask, and the colour and shape words from visual-question-answering (VQA) queries on the image (Poudel et al., 2024). (b) The size word of each single-lesion caption of Section 3.2 against the true lesion area. Shading marks the band that the 8% and 30% thresholds of Eq. (1) give each word; almost every caption lies in its band.

So what information do the models actually use from these captions? To find out, we edit the captions provided to the published checkpoints of MedCLIPSeg and RecLMIS, one part at a time, across seven test sets from three imaging modalities (Fig. 3; RecLMIS on chest X-ray only). These counterfactual edits show what each part of a caption contributes. To see where the behaviour comes from, we retrain MedCLIPSeg with edited captions on three training sets and evaluate the results against criteria that we fixed in advance (Appendix A). Finally, we measure how much the captions alone reveal about the masks, and we test methods that never see the test annotation.

On colonoscopy and chest X-ray, the models take mainly one thing: the lesion’s extent. We call the part of the caption that states it the extent attribute: the size word for colonoscopy and breast ultrasound, and the lung-zone phrase for chest X ray. On these two modalities, the extent attribute alone recovers most of the segmentation benefit provided by the caption. If the captions improved segmentation by describing the lesion, their other words should also contribute. They hardly do: on colonoscopy, colour, shape and count each recover little or none of the benefit, and calling the polyp a rock does not reduce performance (Section 5.1). Retraining shows that the training captions teach this: a model trained without the extent attribute ignores it, and on colonoscopy and chest X-ray a model trained on the extent attribute alone behaves like one trained on the full captions (Section 5.2). The lesson is not free. Models trained on the full captions segment less accurately when the caption is withheld, whereas models trained on captions that do not state the image’s own extent show no such loss (Section 5.3).

The models resemble Clever Hans, the horse that seemed to do arithmetic but read its answers from cues its questioners gave without knowing it (Pfungst, 1911): they answer correctly by reading a cue that the benchmark supplies. Most medicalimaging studies do not test for such Clever Hans predictors (Lapuschkin et al., 2019; Vásquez-Venegas et al., 2025). We call this form of label leakage extent leakage.

Extent leakage is visible in the captions themselves, before any model is trained: a regression that reads only the caption explains most of the variation in lesion area on every test set except CVC-300 (Section 5.4). How far can a model get without the test mask? Our best answer is ExtentSeg, an extent-aware segmenter built for this study, which estimates the extent from the image instead of reading it from a caption. It comes within 2.63 Dice points of the published colonoscopy result and gains 1.94 points over models trained the same way without text. Given the true extent, it matches our retrained models that read the mask-derived caption (Section 5.5). The benefit of text on these benchmarks has therefore been overestimated, and the overestimate can be measured. Our contributions are:

1. caption edits on two published models and three modalities, showing that on six of the seven test sets (colonoscopy and chest X-ray) the benefit of text is carried by an extent attribute that was computed from the mask or written by the benchmark’s annotators, and that on breast ultrasound the diagnosis word shares this benefit;

2. a retraining study on three training sets, with criteria fixed in advance, showing that the training captions create this dependence and that it weakens segmentation from the image;

3. an evaluation protocol, including a caption-only leakage score that needs no segmentation model, and reference results without the mask, among them those of ExtentSeg.

Code and the edited caption sets will be made public on acceptance (see Data availability). A related manuscript by the first author, in preparation, studies how well the training set covers lesion sizes in image-only polyp segmenters; it uses the same public splits and shares no results with this paper.

![](images/ded1f92b282389e8e9905168574602484d2347d429342c7477b5ea3b4edd6fd3.jpg)  
Figure 3: The caption edits of Section 4.1 on one CVC-ClinicDB test image, with the published MedCLIPSeg checkpoint. Words that difer from the matched caption are in red; a struck-through word has been deleted. Size removed, size only and size forced to large are the ablated, isolated and forced captions for the size word. Numbers are Dice scores

## 2. Related work

## 2.1. Text-guided medical image segmentation

Existing text-guided medical image segmentation methods incorporate language into segmentation networks in diferent ways. LViT (Li et al., 2024) fuses text and image features in a hybrid convolutional and Transformer network, whereas RecLMIS (Huang et al., 2025) adds a cross-modal reconstruction objective. MedCLIPSeg (Koleilat et al., 2026) adds probabilistic vision-language adapters to frozen encoders from contrastive language-image pretraining (CLIP; Radford et al., 2021), in this case UniMed-CLIP (Khattak et al., 2024). The chest X-ray captions studied here are the descriptions that the LViT authors wrote for QaTa-COV19; they also described the MosMedData+ computed tomography dataset (Morozov et al., 2020), which we do not use. Zhong et al. (2023) split these annotations into three parts of increasing positional detail and found that the most detailed part, the lung zones, gives their model the same Dice as the full annotation (89.8 points, against 84.1 without text). They read this as the value of detailed position information. Our Zone-only retraining agrees with their result; we add that the zone phrase is an annotation of the lesion’s extent, and that relying on it costs accuracy when segmenting from the image. Later segmenters adopted these annotations as their text (Bhardwaj et al., 2025; Chen et al., 2024b, 2026; Lin et al., 2026; Rahman and Hammond, 2026; Yu et al., 2025); other methods use text only during training (Shan et al., 2025; Ye et al., 2024, 2025; Zhao et al., 2024, 2025a) or train with incomplete prompts (Wang et al., 2025).

Promptable models such as the Segment Anything Model (SAM; Kirillov et al., 2023) and MedSAM (Ma et al., 2024) are trained and evaluated with box or point prompts simulated from the reference annotation (Mazurowski et al., 2023). There, the simulated prompt stands for a real user action: at deployment, a user draws the box. A caption computed from the mask, by contrast, stands for a clinical description that is uncertain and often not recorded (Section 6.1). Large text-prompted medical segmenters take the name of the target, such as an object type or anatomical term, as the prompt (Zhao et al., 2025b,c); such a prompt says what to segment but nothing about the particular mask.

Prompts computed from the mask have been used before. Poudel et al. (2024) computed the number, size and location words of their prompts from the mask. They found that replacing these words at inference barely changes one fine-tuned model (CLIPSeg) but degrades another (CRIS), most for size and location, and read this as evidence of which models use language. Their prompts are the colonoscopy and breast-ultrasound captions of the MedCLIPSeg benchmarks, and adapters for CLIP-based segmenters have been trained and evaluated on them (Dhakal et al., 2024). Liu et al. (2026) trained segmenters with five vision-language encoders on these captions and removed or altered the text at test time, much as we do. They concluded that the efect of text depends on the dataset and acts on the whole image rather than on particular regions, judging this through a decoder that applies text by modulating whole feature channels; they did not examine how the captions were built. We show that (i) the dependence rests on the extent attribute, shared on breast ultrasound with the diagnosis word; (ii) the extent does not merely shift a global threshold but removes false positives away from the lesion; (iii) the training captions teach this dependence at a cost to segmentation from the image; and (iv) a caption-only score measures it without training a segmenter.

## 2.2. Shortcuts, leakage and partial-input baselines

Shortcuts are decision rules that work on standard benchmarks but fail in real-world conditions (Geirhos et al., 2020). They are well documented in medical imaging, where they can inflate reported performance (DeGrave et al., 2021; Ong Ly et al., 2024; Zech et al., 2018), and leakage of information that is available only at evaluation time is one oftheir most direct causes (Kapoor and Narayanan, 2023; Kaufman et al., 2012). Evaluation pitfalls in medical imaging have also been documented for challenge rankings (Maier-Hein et al., 2018) and metrics (Maier-Hein et al., 2024; Reinke et al., 2024). Partial-input baselines, such as hypothesis-only (Gururangan et al., 2018; Poliak et al.,

2018) and question-only (Goyal et al., 2017) models, exposed annotation artefacts in natural language inference and language priors in VQA (Agrawal et al., 2018). They now measure how much of a multimodal benchmark can be solved without the image (Brown II et al., 2026; Chen et al., 2024a), including medical multiple-choice benchmarks, where text alone reaches most of the full accuracy (Wang et al., 2026). Such baselines give only a lower bound on the information available (Feng et al., 2019), and a high score does not show that a model uses it (Srikanth and Rudinger, 2022); we therefore pair our caption-only score with retraining. Referring-expression models have also been shown to ignore the structure of their text (Cirik et al., 2018), and multimodal classifiers often gain little from combining modalities (Hessel and Lee, 2020). Counterfactual benchmarks for segmentation edit the image so that the prompted object is absent (Li et al., 2026), or pair an unchanged mask with a misleading prompt (Liang et al., 2026), and find that models often produce a mask anyway. Our edits instead change single attributes of the caption, at test time and in training.

## 2.3. Modality imbalance and size priors

Networks trained on several modalities together can underlearn one of them, sometimes ending up worse than a singlemodality network (Du et al., 2023; Huang et al., 2022; Wang et al., 2020; Wu et al., 2022). Gradient modulation was proposed to counter this (Peng et al., 2022), and modality dropout (Neverova et al., 2016) to counter the fragility of multimodal Transformers when a modality is missing (Ma et al., 2022). In these studies every modality is a legitimate input. Here, one input carries the extent of the label. Learning using privileged information (Vapnik and Izmailov, 2015) describes the role such a caption ought to play: information available in training but not at test time. Lesion size is also a familiar prior in segmentation (Kervadec et al., 2019b). A network can regress the size of the target from the image to constrain its segmentation (Kervadec et al., 2019a), a known or predicted size can constrain segmentation in a new domain (Bateson et al., 2021), and a class-ratio prior can guide source-free adaptation (Bateson et al., 2022). TGANet (Tomar et al., 2022) predicts the polyp count and a three-level size label from the image to weight its attribute embeddings, and FANetv2 (Tomar et al., 2025) predicts the same two attributes as an auxiliary task. ExtentSeg (Section 4.8, Fig. 4) belongs to this family. Feeding a conditioned segmenter a wrong value of its condition is an established test: Lemay et al. (2021) gave a U-Net conditioned by feature-wise linear modulation (FiLM) each tumour type in turn and found that a wrong type changed the predicted tumour core, most strongly when the type implied a diferent size. Cao et al. (2026) tested phase-conditioned echocardiography segmenters with the true, an estimated and a random phase, and called the gap between true and deployable conditioning conditioning-availability bias. We test ExtentSeg with the true extent and with its own estimate in the same spirit.

Table 1: Benchmarks and the extent attribute of their captions.
<table><tr><td>Dataset</td><td>Modality</td><td>Total</td><td>Train</td><td>Val</td><td>Test</td><td>Extent attribute*</td></tr><tr><td>Kvasir-SEG</td><td>colonoscopy</td><td>1000</td><td>800</td><td>100</td><td>100</td><td>size word</td></tr><tr><td>CVC-ClinicDB</td><td>colonoscopy</td><td>612</td><td>(490)</td><td>(61)</td><td>61</td><td>size word</td></tr><tr><td>CVC-300</td><td>colonoscopy</td><td>66</td><td>(6)</td><td>=</td><td>60</td><td>size word</td></tr><tr><td>CVC-ColonDB</td><td>colonoscopy</td><td>380</td><td>(20)</td><td>=</td><td>360</td><td>size word</td></tr><tr><td>BKAI-IGH</td><td>colonoscopy</td><td>999</td><td>(799)</td><td>(100)</td><td>100</td><td>size word</td></tr><tr><td>BUSI</td><td>ultrasound</td><td>780</td><td>624</td><td>78</td><td>78</td><td>size word</td></tr><tr><td>QaTa-COV19</td><td>chest X-ray</td><td>9258</td><td>5716</td><td>1429</td><td>2113</td><td>lung zones</td></tr></table>

Image counts of the MedCLIPSeg release. Of the colonoscopy datasets, only Kvasir-SEG is used for training; splits in parentheses are not used. ∗Size word: the mask area thresholded by Eq. (1); lung zones: written by annotators (Section 3.2).

## 3. Benchmarks, captions and models

## 3.1. Benchmarks

We use the seven captioned test sets released with Med-CLIPSeg (Table 1; Koleilat, 2026a). The colonoscopy model is trained on Kvasir-SEG (Jha et al., 2020) and tested on its test split and on four datasets from other centres, which we call the unseen centres: CVC-ClinicDB (Bernal et al., 2015), CVC-300 (Vázquez et al., 2017), CVC-ColonDB (Bernal et al., 2012; Tajbakhsh et al., 2016) and BKAI-IGH (Ngoc Lan et al., 2021). The ultrasound model is trained and tested on the Breast Ultrasound Images dataset (BUSI; Al-Dhabyani et al., 2020) and the chest X-ray model on QaTa-COV19 (Degerli et al., 2021, 2022; Li et al., 2024), both with their released splits. All seven datasets are public; no new data were collected from human subjects.

Test images rarely duplicate training images, except on BUSI. Searching for identical or near-identical images (a correlation above 0.98 between 64  64 grey-level thumbnails), we found one near-duplicate (in Kvasir-SEG) among the 681 colonoscopy test images and none in the unseen centres. On BUSI, 20 of the 78 test images (15 with a lesion) nearly duplicate a training or validation image, probably from the same patient. No QaTa-COV19 test subject appears in training or validation, although 2354 training and 597 validation images lack a subject identifier. Two further issues concern the test data alone: 55 of the 60 CVC-300 test images also appear in CVC-ColonDB, so pooled colonoscopy results count them twice, and the captions of three BKAI-IGH test images say “no polyp” although their masks contain one. Appendix D repeats the analyses without these images; for the BKAI-IGH images it does so only for the published models.

## 3.2. Construction of the captions

Fig. 2 shows how a colonoscopy caption is written. The count, size and location words are computed from the reference mask <sub>�</sub>, and the colour and shape words come from VQA queries on the image (Poudel et al., 2024). Every colonoscopy and BUSI caption released with MedCLIPSeg is one of the prompts that Poudel et al. (2024) released for the same image (their template P6; Poudel, 2024; see Appendix C). The size word sorts the lesion’s area fraction $a ( y ) = | y | / | \Omega |$ , the share of the image domain Ω covered by the mask, into three levels:

$$
w ( y ) = \left\{ \begin{array} { l l } { s m a l l , } & { a ( y ) < 0 . 0 8 , } \\ { m e d i u m , } & { 0 . 0 8 \leq a ( y ) < 0 . 3 0 , } \\ { l a r g e , } & { a ( y ) \geq 0 . 3 0 . } \end{array} \right.\tag{1}
$$

In the five caption files we checked (training captions of Kvasir-SEG, CVC-ClinicDB, BKAI-IGH and BUSI, test captions of CVC-ColonDB), this rule reproduces 2617 of the 2633 singlelesion captions; the other 16 lie within 0.006 of a threshold, as expected when areas are measured on resized masks. The size word is therefore a function of the reference mask. The MedCLIPSeg supplement lists the attributes of its templates but no size attribute or threshold, although its example captions contain size words (Koleilat et al., 2026); we recovered Eq. (1) from the released captions.

In QaTa-COV19, the list of involved lung zones (e.g., “all left lung and all right lung”) plays the role of the size word. LViT’s annotators wrote it for the benchmark with the help of professionals and checked it against the masks (Li et al., 2024). It is not a fixed function of the mask: the best rule based on how much of each lung zone the mask covers reproduces the named zones for only 51.5% of test images, and a classifier on maskshape features for 65.1%, whereas the stated side and number of regions agree with the mask far more often (Appendix E). Sides are named by image position, the opposite of radiological convention, so the phrase is not taken from a radiology report, as Zhong et al. (2023) also note. MedCLIPSeg’s captions reword the LViT annotations without changing their content, so LViT, RecLMIS and MedCLIPSeg receive the same extent information.

## 3.3. Models and evaluation

MedCLIPSeg keeps a UniMed-CLIP (Khattak et al., 2024) image encoder (a Vision Transformer, ViT-B/16; Dosovitskiy et al., 2021) and a PubMedBERT (Gu et al., 2022) text encoder frozen, and trains probabilistic adapters inserted into ten encoder layers. With the released test captions, its published checkpoints for Kvasir-SEG, BUSI and QaTa-COV19 (Koleilat, 2026b) reproduce the reported Dice on all seven test sets to within 0.02 points (Table 2). Our second model, RecLMIS (Huang et al., 2025), was developed independently; its published checkpoint also reproduces its reported result (85.21 against 85.22). We use published for the authors’ trained checkpoints and released for their captions and code.

We follow the released evaluation code. The probability map is averaged over 30 stochastic forward passes, thresholded at 0.5 and compared with the mask by the Dice coeficient (Dice, 1945), in points from 0 to 100; an empty prediction on an empty mask scores 100. Every image is resized, without cropping, to 224 224 pixels, the input size of the frozen encoders. The prediction is scaled back to the native resolution of the mask (nearest neighbour) before scoring, so every Dice value in this paper, including those printed on example images, is computed at native resolution. Native test images range from 319 316 (BUSI) to 1348 1070 pixels (Kvasir-SEG); QaTa-COV19 im ages are natively 224  224. Example images in the figures are shown as released, resized but not cropped.

RecLMIS is run with its released test script (one forward pass in evaluation mode, threshold 0.5, native 224 224 resolution); the small changes needed to run it are listed in Appendix C. Its matched caption is the LViT wording it was trained on (Li, 2026). With MedCLIPSeg’s rewording of the same content, it gains 13.34 points instead of 14.68. Its mismatched captions are those given to MedCLIPSeg, in MedCLIPSeg’s wording.

## 4. Methodology

Sections 4.1 to 4.3 define the caption edits and the measures computed from them, Sections 4.4 and 4.5 the retraining, and Sections 4.6 to 4.8 the methods that measure or avoid the leakage without the test mask.

## 4.1. Caption edits

Let $f ( x , t ) \in [ 0 , 1 ] ^ { H \times W }$ be the probability map predicted for image � with caption �, and let $\mathrm { D } _ { i } ( t )$ be the Dice score of test image $x _ { i } , i = 1 , \ldots , n .$ , with caption �. A caption is made of attributes: the count, size, colour, shape and location words, the diagnosis word that some BUSI captions contain, and, on QaTa-COV19, the side, the number of regions and the zone phrase. Starting from each image’s released caption $t _ { i } ^ { \star }$ , we build the following captions by editing the text (Fig. 3):

• matched: $t _ { i } ^ { \star }$ itself;

• empty: $t _ { \emptyset } .$ , the fixed text “a medical image”;

• mismatched: $t _ { \sigma ( i ) } ^ { \star } .$ , the caption ofanother test image, where � is a random reordering of the test images in which no image keeps its own caption, drawn once per test set and shared by all models;

• ablated: $t _ { i } ^ { \star } \setminus a$ , with attribute � deleted;

• isolated: $t _ { i } ^ { \star } | _ { a }$ , with every attribute except � deleted;

• forced: $t _ { i } ^ { \star } [ a \gets \nu ]$ , with � replaced by the value �.

The same edits produce the training captions of Section 4.4. Because QaTa-COV19 has only 191 distinct test captions, 230 of its 2113 test images (10.9%) receive a mismatched caption identical to their own.

## 4.2. Decomposing the benefit of the caption

The quantity usually reported is the improvement of the matched caption over the empty caption, which we call the text gain,

$$
G = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \mathbf { D } _ { i } ( t _ { i } ^ { \star } ) - \mathbf { D } _ { i } ( t _ { \emptyset } ) \right] .\tag{2}
$$

Part of this gain need not depend on the image at all, because any caption from the benchmark may push the prediction towards typical lesion sizes. We estimate this dataset-level part, the

![](images/62f4529b0e991fbe08eb3abc604c965d384b97e50f4507dd8da5b2d5c0f5275e.jpg)  
Figure 4: ExtentSeg. Eight extent tokens, Eq. (10), pass through twelve trained residual stages and reach the frozen image encoder through ten probabilistic vision-language (PVL) adapters. A first pass with the uniform extent u feeds the extent head. Its estimate ${ \hat { \mathbf { p } } } ( x )$ builds the tokens of the second pass, which produces the segmentation. Nothing derived from the mask enters at test time.

prior gain �, with the mismatched caption, and express it as a fraction of the text gain, the prior share �:

$$
P = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl [ \mathsf { D } _ { i } ( t _ { \sigma ( i ) } ^ { \star } ) - \mathsf { D } _ { i } ( t _ { \circ } ) \bigr ] , \qquad \pi = \frac { P } { G } .\tag{3}
$$

The prior share is negative when another image’s caption is worse than no caption. The rest of the gain, $G - P .$ , is specific to the image (Fig. 5(a)). For an attribute �, the suficiency $S _ { a }$ is the fraction of � recovered when the caption keeps only �, and the necessity $N _ { a }$ is the fraction lost when only � is deleted:

$$
S _ { a } = \frac { 1 } { n G } \sum _ { i = 1 } ^ { n } \bigl [  { \mathbf { D } } _ { i } ( t _ { i } ^ { \star } | _ { a } ) -  { \mathbf { D } } _ { i } ( t _ { \emptyset } ) \bigr ] ,\tag{4}
$$

$$
N _ { a } = \frac { 1 } { n G } \sum _ { i = 1 } ^ { n } \bigl [  { \mathbf { D } } _ { i } ( t _ { i } ^ { \star } ) -  { \mathbf { D } } _ { i } ( t _ { i } ^ { \star } \setminus a ) \bigr ] .\tag{5}
$$

If other parts of the caption partly repeat an attribute’s information, its necessity is lower than its suficiency. We write $S _ { \mathrm { e x t } }$ and $N _ { \mathrm { e x t } }$ for the suficiency and necessity of the extent attribute (the size word or the lung-zone phrase).

## 4.3. Response to the size word

Let $A _ { i } ( t ) = | \{ p \in \Omega : f ( x _ { i } , t ) _ { p } > 0 . 5 \} | / | \Omega |$ be the fraction of image �<sub>�</sub> predicted as lesion with caption �. We write � both for the size-word attribute and, as in Eq. (1), for its value. The size response is the mean change in this fraction when the size word is forced from small to large:

$$
\Delta _ { \mathrm { s i z e } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Bigl [ A _ { i } \left( t _ { i } ^ { \star } [ w \gets l a r g e ] \right) - A _ { i } \left( t _ { i } ^ { \star } [ w \gets s m a l l ] \right) \Bigr ] .\tag{6}
$$

On QaTa-COV19 the zone phrase takes the place of the size word, and the size response compares the one-zone phrase “the lower right lung” with the six-zone phrase “all left lung and all right lung”.

## 4.4. Retraining with edited captions

Editing captions at test time describes a trained model; retraining shows where its behaviour comes from. We retrain MedCLIPSeg on Kvasir-SEG, BUSI and QaTa-COV19 with the published recipe: 100 epochs of Adam (Kingma and Ba, 2015), a learning rate decayed from $3 \times 1 0 ^ { - 4 }$ to $1 0 ^ { - \bar { 4 } }$ on a cosine schedule (Loshchilov and Hutter, 2017), and batch size 24. We use five sets of training captions:

• Original: the released captions;

• Size-blind: $t _ { i } ^ { \star } \mid$ �, the size word deleted;

• Size-only: $t _ { i } ^ { \star } | _ { w } ,$ only the size word kept;

• Mismatched: each image paired with another image’s caption, by one fixed reordering as in Section 4.1;

• Text-free: the empty caption $t _ { \emptyset }$ for every image.

On BUSI the Size-only caption also drops the diagnosis word, benign or malignant, and the 107 training images without a lesion keep their caption “No tumor in the breast ultrasound image.” in the Original, Size-blind and Size-only sets. On QaTa-COV19 the zone phrase plays the role of the size word (Zone-blind, Zone-only); the Zone-blind captions still state the side and the number of afected regions, and no Mismatched models were trained.

All retrained models are evaluated at their last epoch, with the released test captions: the colonoscopy models on the five colonoscopy test sets $( n = 6 8 1 )$ , the QaTa-COV19 models on its 2113 test images, and the BUSI models on all 78 BUSI test images. For BUSI, criteria R1-R3 and $\rho$ (below) use only the 65 test images with a lesion, because only their captions have a size word. $G _ { \mathcal { T } }$ is the text gain of the models trained on caption set , with each image’s scores averaged over seeds.

We fixed five criteria in advance. For BUSI and QaTa-COV19 they were committed to version control before training began; for Kvasir-SEG they were written on the day the first model was trained and committed before the three-seed results that decide them (Appendix A):

(a) one test image under three captions  
![](images/8d2f5a0407aa37a4946ed4ab4a27fb3d51fa807a31111e1a466745d15b7e3a32.jpg)

(b) each test set, published models  
![](images/3676050e4448f41b982ec9b056731d8197d210f1f37911ad0b629647084ffb81.jpg)  
Figure 5: Decomposition of the text gain. (a) The test image of Fig. 3 with the empty, mismatched and matched captions. Another image’s caption already gives the prior gain �, Eq. (3); only the remainder $G - P$ needs the image’s own caption. (b) The same decomposition for each test set, with the text gain � (tick), its 95% confidence interval (CI) and the prior share �. On BUSI, � is negative (hatched).

R0 Reproduction: with the matched caption, the mean Dice of the Original models lies within 1.5 points of the published checkpoint.

R1 Size response: the size response of the Size-blind models is below 25% of that of the Original models.

R2 Necessity: the Size-blind models keep at most half of the Original text gain, $G _ { \mathrm { S i z e - b l i n d } } / G _ { \mathrm { O r i g i n a l } } \ \leq \ 5 0 \%$ , with the upper confidence bound below 50%.

R3 Suficiency: the Size-only models keep at least three quarters of it, $G _ { \mathrm { S i z e - o n l y } } / G _ { \mathrm { O r i g i n a l } } \geq 7 5 \%$ , with the lower confidence bound above 75%.

R4 Uninformative captions: the text gain of the Mismatched models, and the diference between their empty-caption Dice and that of the Text-free models, both lie within 1 point.

Before any BUSI or QaTa-COV19 model was trained, we amended the criteria for these two datasets in three ways: R4 also requires both 95% CIs to lie within 2 points; $\rho _ { \mathrm { S i z e } } .$ -blind $> 0$ (Section 4.5), with an interval excluding zero, becomes a sixth criterion; and a result whose point estimate is on the required side but whose interval reaches the bar is called consistent.

## 4.5. Segmentation without a caption

A model that reads the extent from the caption may learn less about it from the image. We therefore compare, with the empty caption, the models trained on each caption set $\mathcal { T }$ with the Original models:

$$
\rho _ { \mathcal { T } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bigl [ \bar { \mathbf { D } } _ { i } ^ { \mathcal { T } } ( t _ { \emptyset } ) - \bar { \mathbf { D } } _ { i } ^ { \mathrm { O r i g i n a l } } ( t _ { \emptyset } ) \bigr ] ,\tag{7}
$$

where $\bar { \mathsf { D } } _ { i } ^ { \mathcal { T } }$ is averaged over seeds. A positive empty-caption diference $\rho \tau$ means that the models trained on $\mathcal { T }$ segment better from the image alone.

## 4.6. Caption-only leakage score

How much a set of captions reveals about the masks can be measured without any segmentation model. For each image with a lesion, a ridge regression (Hoerl and Kennard, 1970) predicts the log lesion area $z _ { i } = \log a ( y _ { i } )$ from the words and word pairs of its caption, with five-fold cross-validation (scikitlearn; Pedregosa et al., 2011): each image’s prediction $\hat { z } _ { i }$ comes from a regression fitted on the other four folds. We report the coeficient of determination of these predictions,

$$
R _ { \mathrm { c a p } } ^ { 2 } = 1 - { \frac { \sum _ { i } ( z _ { i } - \hat { z } _ { i } ) ^ { 2 } } { \sum _ { i } ( z _ { i } - \bar { z } ) ^ { 2 } } } ,\tag{8}
$$

where <sub>�</sub>¯ is the mean of the $z _ { i }$ . Shufling the captions across images gives the chance level. The folds are drawn over images, not over videos or subjects.

Because $R _ { \mathrm { c a p } } ^ { 2 }$ gives a fit but no significance test, we add a model-free measure, $I _ { \mathrm { e x t } } = ( \hat { I } - \mathbb { E } _ { \mathrm { p e r m } } \hat { I } ) / \hat { H } ( b )$ . Here � is the tercile of the lesion area (whether the lesion is among the smallest, middle or largest third of the lesions scored), <sup>ˆ</sup>� is the mutual information between $b$ and the caption’s extent token $e$ (with the Miller-Madow correction; Miller, 1955), $\mathbb { E } _ { \mathrm { p e r m } } \hat { I }$ is its mean over 200 random permutations of $^ { e , }$ and $\hat { H } ( \boldsymbol { b } )$ is the entropy of the terciles. In words, $I _ { \mathrm { e x t } }$ is the fraction of the uncertainty about the tercile that the extent token removes beyond chance: about 0 when the token says nothing about the area, and close to 1 when it fixes the tercile. The same permutations give a onesided <sub>�</sub>-value, whose smallest possible value is $1 / 2 0 1 \approx 0 . 0 0 5$ The extent token is the size word or, for caption sets without one (QaTa-COV19 and Size-blind captions) and for the edited BUSI and QaTa-COV19 training-caption sets of Section 4.4, the whole caption.

## 4.7. Annotation-free methods

We call a method annotation-free if nothing derived from the test annotation enters its input. We test four such methods.

Edge-aligned threshold. The published model is run with three fixed captions, “a small region in the image.” and its medium and large variants, and the three probability maps are averaged into a score map $s ( x )$ . Let $\Omega _ { k }$ be the � highest-scoring pixels. Among 96 candidate areas between 0.1% and 60% of the image, the rule picks the one whose boundary lies where � changes most steeply:

$$
k ^ { * } = \arg \operatorname* { m a x } _ { k } \frac { 1 } { \left| \partial \Omega _ { k } \right| } \sum _ { p \in \partial \Omega _ { k } } \left\| \nabla s ( p ) \right\| ,\tag{9}
$$

where is a $3 \times 3$ Sobel gradient and $\partial \Omega _ { k }$ is the boundary of $\Omega _ { k } ; \mathrm { i f } \ s \le 0 . 5$ everywhere, the mask is empty. On QaTa-COV19, � averages seven fixed captions that cover one to six lung zones and three size words (Appendix C). This rule was the best of twelve label-free rules on the pooled colonoscopy test sets, though not on every single test set or on BUSI (Table D.5), so choosing it used the test labels. If the rule for each test set is instead chosen on the remaining test sets, pooled Dice changes by only $- 0 . 2 6 \ [ - 0 . 6 6 , + 0 . 0 9 ]$ . On QaTa-COV19, three of the seven rules evaluated there score higher than this one (at best 76.22 against 75.66).

Image-only describer. This method predicts the caption from the image and gives the predicted caption to the published model. The prediction uses the image features of the published model run with the empty caption. A ridge regression estimates the log area, which is turned into small, medium or large at the terciles of the Kvasir-SEG training lesion areas, and logistic classifiers predict the count and location words; colour and shape are not predicted. All three are fitted on the Kvasir-SEG training split. Cutting at the thresholds of Eq. (1) instead changes its Dice by only 0.19 [ 0.58, +0.12].

The other two methods are the Text-free models of Section 4.4 and ExtentSeg.

## 4.8. An extent-aware segmenter

If the useful content of the caption is the extent, a segmenter can estimate the extent from the image and condition on that estimate instead. ExtentSeg replaces MedCLIPSeg’s text branch with learned extent tokens and keeps the image encoder, adapters and mask head (Fig. 4). Extent is divided into $K = 2 0$ bins at the quantiles of the Kvasir-SEG training lesion areas, and a probability distribution $\mathbf { p } \in \Delta ^ { K - 1 }$ over the bins is mapped to � = 8 tokens

$$
\mathbf { T } ( \mathbf { p } ) = \mathbf { B } + \mathbf { 1 } _ { M } ( \mathbf { p } ^ { \top } \mathbf { E } ) ,\tag{10}
$$

where ${ \bf 1 } _ { M }$ is a vector of � ones, $\mathbf { E } \in \mathbb { R } ^ { K \times 7 6 8 }$ holds one learned embedding per bin, $\mathbf { B } \in \mathbb { R } ^ { M \times 7 6 8 }$ is a learned base, and $\Delta ^ { K - 1 }$ is the probability simplex. Each token is thus a learned base vector plus the average of the bin embeddings, weighted by p. In place of the language layers, twelve residual stages $\mathbf { T } $ ${ \bf T } + \mathrm { L i n } _ { \ell } ( \mathrm { L N } ( { \bf T } ) )$ refine the tokens; each is a layer normalisation (LN; Ba et al., 2016) followed by a zero-initialised linear layer, so every stage starts as the identity.

An extent head estimates pˆ � from the image alone, using a first forward pass conditioned on the uniform distribution u; a second pass, conditioned on pˆ � , produces the segmentation. The head’s loss does not flow back into the image features, so it updates only the head. During training, each sample is conditioned on the true extent (slightly blurred), on the model’s own estimate, or on no extent information:

$$
\mathbf { p } = \left\{ \begin{array} { l l } { \tilde { \mathbf { q } } ( k _ { i } ) } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 4 5 , } \\ { \operatorname { s g } [ \hat { \mathbf { p } } ( x _ { i } ) ] } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 4 5 , } \\ { \mathbf { u } } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 1 0 , } \end{array} \right.\tag{11}
$$

where $k _ { i }$ is the bin of training image �’s mask, q˜ � is a Gaussian over bins centred at � (standard deviation one bin), and sg is a stop-gradient. The loss combines Dice and binary cross-entropy (BCE) segmentation losses, a cross-entropy (CE) term for the extent head and an area term:

$$
\begin{array} { r l } & { \mathcal { L } = \frac { 1 } { 2 } \mathcal { L } _ { \mathrm { D i c e } } + \frac { 1 } { 2 } \mathcal { L } _ { \mathrm { B C E } } + \lambda _ { e } \mathbf { C E } \big ( \tilde { \mathbf { q } } ( k _ { i } ) , \hat { \mathbf { p } } \big ) } \\ & { \quad \quad \quad + \lambda _ { a } \big ( \log \bar { m } _ { i } - \log \bar { a } _ { i } \big ) ^ { 2 } , } \end{array}\tag{12}
$$

where $\bar { m } _ { i }$ is the mean predicted probability over image � and $\bar { a } _ { i } = \tilde { \mathbf { q } } ( k _ { i } ) ^ { \top } \mathbf { c }$ is the expected area under the soft target, with c the bin centres (Appendix C). The area term is used only when a sample is conditioned on the true extent and its lesion is larger than the first bin edge. We set $\lambda _ { e } = \lambda _ { a } = 0 . 1$ ; the variant without the area loss sets $\lambda _ { a } = 0$ . The extent-free control is ExtentSeg with its extent input removed; its area term uses the true extent of every training image whose lesion is larger than the first bin edge. All variants use the training recipe of Section 4.4.

At test time ExtentSeg is conditioned on its own estimate pˆ � . Conditioning it instead on q˜ � for the bin � of the test mask gives an upper reference that is not annotation-free.

## 4.9. Statistical analysis

CIs are 95% percentile intervals from a paired bootstrap over test images (Efron and Tibshirani, 1993) with $1 0 ^ { 4 }$ resamples (random seed 0); for ratios, numerator and denominator are resampled together. Every retrained model, as well as ExtentSeg and its two variants, was trained with three seeds (1, 2 and 3), and each image’s scores are averaged over the seeds before resampling. Where stated, seed-resampled intervals also resample the three seeds of each model in every bootstrap draw. Pooled results are means over all images of the pooled test sets. Diferences and ratios are computed from unrounded values, so they can difer in the last digit from calculations on the printed values.

Not all test images are independent: frames from one colonoscopy video are correlated, and QaTa-COV19 contains several images per subject. As sensitivity analyses, two cluster bootstraps resample whole video sequences (273 clusters of the 681 colonoscopy test images, built from the images alone) or the 474 QaTa-COV19 test subjects instead of images; Appendix D describes the clustering and reports the results.

(a) keep one attribute of the caption, delete the rest  
![](images/f24c73266038bb692c1ea834c79b55f4a874e27bae55fb68cd2693b575b3f2a1.jpg)  
Figure 6: Which caption attribute carries the text gain. (a) Suficiency of each attribute, Eq. (4): the share of the text gain recovered when every test caption keeps only that attribute (100: as good as the whole caption; 0: no better than the empty caption). For QaTa-COV19 the extent attribute is the lung-zone phrase; -: not in the template. BUSI: the 65 images with a lesion. (b) The extent column with 95% CIs.

## 5. Results

## 5.1. What the models takefrom the caption

## 5.1.1. Dataset-level and image-specific benefit

Part of the reported benefit is not specific to the image. The text gains of the published checkpoints range from 9.11 to 24.61 points (Table 2, Fig. 5). On CVC-ClinicDB, CVC-300 and BKAI-IGH, a caption from a diferent image already recovers most of this gain; on Kvasir-SEG, the training domain, it recovers none. Pooled over the five colonoscopy test sets, the prior share is 39.7% [27.3, 50.8]. This interval treats video frames as independent; when whole video sequences are resampled it widens to [4.9, 58.6], and under the coarsest clustering it includes zero (Table D.2).

The prior gain mainly corrects how much of the image the model labels as lesion. Without a caption, the colonoscopy model labels on average 12.3% of an unseen-centre image as polyp, nearly twice the true 6.8%; with another image’s caption, it labels 5.7%. On BUSI the prior share is negative, because 13 test images contain no lesion, and another image’s caption may describe a lesion that is absent or deny one that is present. Since � ranges from strongly negative to over 90%, a gain over the empty caption alone says little about how much of the benefit is specific to the image. The picture is the same on the normalised surface Dice (Nikolov et al., 2021), a boundary-based score (Table D.3), and without the duplicated test images (Table D.1). Without the QaTa-COV19 images whose mismatched caption is identical to their own (Section 4.1), � falls from 45.0% to 36.2% for MedCLIPSeg and from 14.5% to 0.0% for RecLMIS.

## 5.1.2. Which attribute carries the benefit

The image-specific part of the gain is carried by the extent attribute. On every colonoscopy test set, the extent attribute alone recovers 84.1-98.3% of the text gain, and deleting it loses 51.8-77.2% (Table 2). No other attribute recovers much on its own on these test sets (Fig. 6), and colour, the word that most plainly describes the lesion’s appearance, is worse than no caption at all. The count and location words are also computed from the mask, yet the models barely use them: deleting the location word costs 11.3% [7.9, 15.1] of the gain, and deleting the count word costs nothing.

Table 2: Benefit of the caption for the published checkpoints.
<table><tr><td>Test set</td><td>Empty</td><td></td><td>Mism. Match.</td><td>G</td><td>π</td><td> $S _ { \mathrm { e x t } }$ </td><td> $N _ { \mathrm { e x t } }$ </td><td> $R _ { \mathrm { c a p } } ^ { 2 }$ </td></tr><tr><td>MedCLIPSeg</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kvasir-SEG</td><td>81.04</td><td>81.04</td><td>90.15</td><td>9.11</td><td>-0.1</td><td>87.7</td><td>51.8</td><td>0.70</td></tr><tr><td>CVC-ClinicDB</td><td>69.92</td><td>76.06</td><td>80.80</td><td>10.87</td><td>56.5</td><td>84.1</td><td>67.4</td><td>0.54</td></tr><tr><td>CVC-300</td><td>65.32</td><td>79.49</td><td>80.84</td><td>15.53</td><td>91.3</td><td>85.9</td><td>67.2</td><td>0.24</td></tr><tr><td>BKAI-IGH</td><td>54.55</td><td>73.75</td><td>79.16</td><td>24.61</td><td>78.0</td><td>92.0</td><td>77.2</td><td>0.59</td></tr><tr><td>CVC-ColonDB</td><td>59.17</td><td>60.96</td><td>71.89</td><td>12.72</td><td>14.1</td><td>98.3</td><td>68.2</td><td>0.71</td></tr><tr><td>BUSI†</td><td>71.73</td><td>59.50</td><td>85.73</td><td>14.00</td><td>-87.4</td><td>30.6</td><td>57.2</td><td>0.60</td></tr><tr><td>QaTa-COV19</td><td>60.24</td><td>70.67</td><td>83.41</td><td>23.17</td><td>45.0</td><td>90.9</td><td>63.4$</td><td>0.77</td></tr><tr><td>RecLMIS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QaTa-COV19π</td><td>70.53</td><td>72.66</td><td>85.21 14.68</td><td></td><td>14.5</td><td>87.8</td><td>58.0</td><td></td></tr></table>

Dice with the empty, mismatched and matched captions; �, Eq. (2); �, �<sub>ext</sub> and $N _ { \mathrm { e x t } } .$ Eqs. (3)-(5), in percent; $R _ { \mathrm { c a p } } ^ { 2 } ,$ Eq. (8), of the test captions (at most 0.00 with shufled captions). $^ \dag S _ { \mathrm { e x t } }$ and $N _ { \mathrm { e x t } }$ on the 65 images with a lesion. ‡36.2% without the images whose mismatched caption equals their own. §Zone phrase removed from every caption directly, because deleting it through the template misses it in 443 captions. ¶LViT wording; other conditions in Appendix D.

On QaTa-COV19, the zone phrase alone recovers 90.9% ofthe gain for MedCLIPSeg and 87.8% for RecLMIS, while deleting the stated count or side costs at most 1.8% (no other attribute was tested alone). Deleting the zone phrase costs RecLMIS 58.0% of its gain in its own wording. Our registered test deleted it from MedCLIPSeg’s wording instead; there the cost is only 43.6% [41.3, 45.8], below the 50% bar set in advance (Table B.1).

On BUSI, the caption leaks a second label: 30 of the 65 lesion captions name the diagnosis, benign or malignant, always correctly. Extent and diagnosis share the gain. On these 65 images, deleting the size word loses 57.2% of the gain; the size word alone recovers 30.6%, and together with the diagnosis 69.7%. (The other BUSI cells of Fig. 6(a) also keep the diagnosis word where the caption has one, so they are not comparable with the colonoscopy cells.)

Three controls show that the models respond to this content, not to medical vocabulary. Replacing polyp with rock (“one small pink circle rock, located in right of the image”) costs nothing on the colonoscopy test sets (+0.22 points). A made-up word (“a zlorb region in the image.”) recovers only 1.55 of the 14.02 points of text gain. Rewriting the captions in other styles, from bare lists to clinical sentences, changes the pooled result by at most 2.1 points (up to 6.8 on single test sets; Table D.4); only the list style, which is not a sentence, misses our registered bar of 1.0 point from the terse rewrite (Table B.1). Keeping the new wording but emptying its content loses 16.1 to 24.0 points and leaves the result within 4.7 points of the empty caption.

(a)  
![](images/e5edbc93c439f9167ad3ed54608c032a2446c139637704ff3c57f3b5d2ab3f0f.jpg)

(b)  
![](images/60e560946dfc6b81015c930be41203ca035720e4da16fe31d0b005521e381dc7.jpg)

![](images/68d26b16b6b7a61d7b65e47a50a51e333a1570981de7befb47a1a49d46318dd9.jpg)  
Figure 7: Predictions of the published models when only the extent in the caption is changed. (a) Colonoscopy and breast ultrasound, with the size word replaced (percentage: true lesion area). (b) Chest X-ray, with the lung-zone phrase replaced by one covering 1, 2, 4 or 6 zones. Orange frames mark the image’s true extent; numbers are Dice scores. (c) Mean predicted area over the 681 colonoscopy test images for one caption with diferent size words, including words not seen in training and the made-up word zlorb.

![](images/b43516de98eb30e6b815569249e51d1fd74d5ecc2c546467ed2cc2577924acc9.jpg)  
Figure 8: Text gain of the published models against lesion area, over the 746 colonoscopy and BUSI test images with a lesion (deciles of the Kvasir-SEG training areas; band: 95% CI). Shading marks the size word that the template assigns to each area. The gain is largest where the caption says small.

## 5.1.3. How the size word acts on the prediction

In efect, the size word tells the model how much of the image to segment (Fig. 7). Told small, the model segments a small polyp almost exactly; told large, it buries the same polyp in false positives. A polyp covering most of the image can vanish when the caption says small, although this is rare (1 of 30 lesions that the template calls large). The leaked size word can also erase a lesion: with the matched caption the prediction is empty for 22 of the 681 colonoscopy lesions (21 in CVC-ColonDB), against none with the empty caption. On colonoscopy, the mean predicted area roughly doubles from small to medium and again to large, and words never seen in training, such as tiny, huge and enormous, fall on the same scale; only minuscule is out of place (Fig. 7(c)). The chest X-ray model responds to the lung zones in the same way (Fig. 7(b)).

This is not just a shift of the decision threshold. Choosing a threshold for each image from an estimate of lesion volume is an established remedy for volume bias (Valcarcel et al., 2020), but it does not explain the efect here. Adding a fixed ofset to the output logits for each size word (in efect, one threshold per word), fitted on the training images, recovers only 44.7% [37.7, 51.7] of the word’s benefit on the unseen centres. Even an ofset tuned for each test image on its own mask reaches only 70.98 Dice, against 74.49 with the size word alone (� = 578, without the three “no polyp” captions). The word suppresses false positives far from small lesions (Fig. 7(a), first row), which no threshold can remove without also removing the lesion.

As a result, the text gain is concentrated on small lesions (Fig. 8). Over the 746 colonoscopy and BUSI test images with a lesion, it is +19.13 points [+17.02, +21.26] for lesions under 7.0% of the image (the smallest 30% of Kvasir-SEG training lesions), +4.91 between 7.0 and 17.9%, and +2.68 [ 0.79, +6.31] above 17.9% (the largest 30%). Much of the benefit of text is a rescue of small lesions, and it works because the caption says that they are small.

## 5.2. Where the dependence comes from

The training captions create the dependence on extent. On colonoscopy, all five criteria are met (Table 3; Table B.1 lists every registered criterion and its verdict). The Original models reproduce the published checkpoint to within 1.22 points (R0). Models trained with the size word follow a forced size word even when it is wrong, while the Size-blind and Text-free models ignore it (R1; Fig. 9). Deleting the size word from the training captions leaves 37.4% of the text gain (R2). Keeping only the size word, which leaves just 19 distinct training captions such as “small polyp, located in of the image” (only the template’s fixed words remain), keeps 89.7% (R3). Captions unrelated to their images teach almost nothing (R4). The Size-blind models keep a small gain (2.31 points) because their remaining words still carry some extent information $( R _ { \mathrm { c a p } } ^ { 2 } = 0 . 3 7 ;$ ; Table 4).

![](images/92b19e71ce00c48e525949b06f8893166d859759c91a75f8e4d76b867213aee4.jpg)

![](images/4adb2ada4ec84723c3f4e21896ee4c47d2d9d1130320973039264278bb203deb.jpg)  
Figure 9: Retraining MedCLIPSeg on edited Kvasir-SEG captions. (a) Four retrained models (seed 1); each image is given its own caption with the size word forced. (b)-(d) All five models on the five colonoscopy test sets (dots: seeds; lines: means; shaded: models trained with the size word): (b) size response, Eq. (6); (c) text gain; (d) empty-caption Dice and �, Eq. (7).

These verdicts use the image bootstrap that the criteria specify, and they are less certain once seeds and video sequences are taken into account. R2 holds in all nine pairings of a Size-blind seed with an Original seed, but R3 holds in only six of the nine, because one Size-only seed gains just 3.1 points. When the seeds are also resampled, both intervals cross their bars, and when video sequences are resampled, so does the upper bound of R2 (Appendix D, Table D.2). Both point estimates stay on the required side.

BUSI reproduces the size response and the cost to segmentation from the image, but not every criterion (Table 3). The Original models reproduce the published checkpoint (R0), and the Size-blind models ignore the size word (R1). On the 65 lesion images, the Size-blind models keep 39.1% of the text gain: below the R2 bar, but with an interval that reaches it (consistent). On a second population specified in advance, the 130 lesion images of the test and validation sets, they keep 28.5% [16.1, 40.7], and R2 is met. R3 is not met, because BUSI captions carry more than the extent: the Size-only captions also lose the diagnosis word, and these models keep only 49.9% of the gain. With the matched caption they score about as well as the Original models, but they score higher with the empty caption, so their share of the gain is small (Table D.6). On BUSI, then, the size word is one of several cues that the Original models rely on. R4 is not met either, but only because of the lesion-free images: in 2 of 39 evaluations of such images (13 images, three seeds), the Mismatched models predict a lesion although the caption says “No tumor in the breast ultrasound image.” This accounts for 0.85 of their 1.13 points; on the 65 lesion images their gain is 0.33 [ 0.86, +0.10], within the bar. With the seeds also resampled, the BUSI R2 and R3 intervals both cross their bars (Appendix D).

Chest X-ray meets every criterion that was run (Table 3). The Original models reproduce the published checkpoint (R0). The Zone-blind models ignore the zone phrase (R1) and keep only 9.0% of the gain (R2), although their captions still state the side and number of regions $( R _ { \mathrm { c a p } } ^ { 2 } = 0 . 4 7 )$ . The zone phrase alone teaches 91.0% of the gain (R3). With the seeds also resampled, R2 stays well below its bar, but R3 holds in only six of the nine seed pairings and its interval crosses the bar (Appendix D). Resampling subjects instead of images changes no conclusion.

On colonoscopy and chest X-ray, the published checkpoints depend on the caption even more than our retrained models. Their text gains are larger (14.02 against 6.18 points on colonoscopy, 23.17 against 15.77 on chest X-ray), mainly because they score lower with the empty caption. Our retraining results therefore describe models trained with the published recipe; the stronger dependence of the published checkpoints themselves is shown only by the test-time edits of Section 5.1. Text gains also vary between checkpoints trained the same way, from 5.15 to 7.53 points across our three Original seeds on colonoscopy and from 10.85 to 19.10 on chest X-ray (Table D.6), so they should be reported over several checkpoints. Every causal comparison in this paper is between models trained with the same recipe; on colonoscopy, these comparisons also hold when checkpoints are selected by validation Dice instead of taken at the last epoch (Appendix D).

## 5.3. The cost to segmentationfrom the image

Learning the leakage weakens segmentation from the image (Fig. 9(d)). We found this on colonoscopy and confirmed it on breast ultrasound and chest X-ray, where it was specified in advance. On colonoscopy, with the empty caption, every Sizeblind model beats every Original model, by $\rho _ { \mathrm { S i z e - b l i n d } } = + 2 . 3 9$ points on average (Table 3), and the Text-free models lead by $\rho _ { \mathrm { T e x t - f r e e } } ~ = ~ + 2 . 8 2 ~ [ + 2 . 1 1 , + 3 . 5 3 ]$ in all nine seed pairings. The Size-only models are no better than the Original models $( \rho _ { \mathrm { S i z e - o n l y } } = - 0 . 0 9 [ - 0 . 6 6 , + 0 . 4 8 ] )$ , so the size word alone produces the deficit. Both leads stay above zero when seeds or video sequences are also resampled (Appendix D).

Table 3: Retraining with edited captions, against the registered criteria.
<table><tr><td></td><td>Criterion</td><td> $\mathrm { K v a s i r - S E G } \left( n = 6 8 1 \right)$ </td><td> $\mathsf { B U S I } \left( n = 7 8 ; \mathrm { R 1 } \mathrm { - R 3 } , \rho ; 6 5 \right)$ </td><td> $\mathrm { Q a T a - C O V 1 9 } \left( n = 2 1 1 3 \right)$ </td></tr><tr><td></td><td>R0 Original within ±1.5 of the published checkpoint</td><td> $7 6 . 0 0 \mathrm { v s . 7 7 . } 2 3 ( - 1 . 2 2 ) \mathrm { y e s }$ </td><td> $8 5 . 2 6 ~ \mathrm { v s } . 8 5 . 7 3 ~ ( - 0 . 4 7 ) ~ \mathrm { y e s }$ </td><td>83.66 vs. 83.41 (+0.25) yes</td></tr><tr><td></td><td>R1 Size-blind size response  $< 2 5 \%$  of Original</td><td> $1 . 2 \% ( 0 . 1 3 \mathrm { v s . 1 1 . 1 p p ) y e s }$ </td><td>1.3% (0.11 vs. 8.0 pp) yes</td><td>0.0% (0.00 vs. 7.9 pp) yes</td></tr><tr><td></td><td>R2 Size-blind retains ≤ 50% of G (CI)</td><td>f 37.4% [28.5,46.8] yes</td><td>39.1% [21.9, 58.3] consistent</td><td>9.0% [7.1, 11.0] yes</td></tr><tr><td></td><td>R3 Size-only retains ≥ 75% of G (CI)</td><td> $8 9 . 7 \% [ 8 0 . 4 , 9 9 . 8 ] \mathrm { y e s }$ </td><td> $4 9 . 9 \% [ 3 7 . 6 , 6 0 . 7 ] \mathrm { n o }$ </td><td>91.0% [88.8, 93.2] yes</td></tr><tr><td></td><td>R4 Mismatched:  $| G | \leq 1 ;$  empty-caption score within</td><td> $G = - 0 . 5 3 [ - 0 . 7 8 , - 0 . 2 8 ] ; + 0 . 3 1$   $\left[ - 0 . 1 8 , + 0 . 8 0 \right] \mathrm { y e s }$ </td><td> $G = - 1 . 1 3 [ - 2 . 5 2 , - 0 . 1 0 ] ; + 0 . 5 4$   $[ - 0 . 4 6 , + 1 . 7 4 ] \ \mathrm { n o }$ </td><td></td></tr><tr><td>ρ</td><td>±1 of Text-free Size-blind better from the image alone  $( \rho > 0 , \mathrm { C I } )$ </td><td>+2.39 [+1.62, +3.16] (not registered)</td><td> $+ 8 . 4 3 \ [ + 4 . 3 8 , + 1 3 . 0 2 ] \ \mathrm { y e s }$ </td><td>+11.28 [+10.55, +12.00] yes</td></tr></table>

MedCLIPSeg, three seeds per model; the Kvasir-SEG models are tested on the five colonoscopy test sets. On QaTa-COV19 the extent attribute is the zone phrase (Zone-blind and Zone-only models). CI: image bootstrap over seed-averaged scores, as registered; seed-resampled intervals are given in Appendix D. pp: percentage points of image area.

On BUSI the deficit is larger. On the 65 lesion images, $\rho _ { \mathrm { S i z e - b l i n d } } = + 8 . 4 3$ <sup>points,</sup> �Text-free <sup>=</sup> +<sup>13</sup>.<sup>43</sup> <sup>[+8.41,</sup> <sup>+18.74]</sup> and $\rho _ { \mathrm { S i z e - o n l y } } = + 6 . 7 3 ~ [ + 4 . 3 8 , + 9 . 3 8 ]$ . Even the Size-only models segment better from the image than the Original models, so here the size word produces only part of the deficit (and the Sizeonly interval includes zero when seeds are resampled). Trained and tested without any text, the BUSI models score 80.65 on these images, against 82.31 for the Original models with the matched caption.

On chest X-ray, the Zone-blind models lead the Original models by $\rho _ { \mathrm { Z o n e - b l i n d } } = + 1 1 . 2 8$ points (Table 3), by at least 6.45 in every seed pairing, and match the Text-free models (79.17 against 79.20 Dice with the empty caption). The Zone-only models are no better than the Original models $( \rho _ { \mathrm { Z o n e - o n l y } } ~ = ~ - 0 . 3 9$ $[ - 0 . 7 0 , - 0 . 0 7 ] )$ . Again, the extent attribute alone produces the deficit.

The deficit is not caused by the caption simply being absent (Ma et al., 2022). The Size-blind and Mismatched models never saw an empty caption in training either, yet they show no deficit: the Mismatched models lead the Original models by $\rho _ { \mathrm { M i s m a t c h e d } } = + 3 . 1 4 $ points [+2.41, +3.86], in all nine seed pairings. What sets the Original models apart is a caption that states the image’s own extent. We read this as a label-derived form of modality imbalance (Du et al., 2023; Huang et al., 2022; Wang et al., 2020; Wu et al., 2022): when the caption supplies the extent, the model learns less about it from the image.

## 5.4. Detecting the leakage from the captions alone

The leakage can be measured before any model is trained. From the caption alone, the regression predicts lesion area with

Table 4: Caption-only leakage of the released and edited caption sets.
<table><tr><td></td><td> $R _ { \mathrm { c a p } } ^ { 2 }$ </td><td colspan="2"> $I _ { \mathrm { e x t } }$ </td></tr><tr><td>Captions</td><td>train</td><td>train</td><td>test</td></tr><tr><td>Kvasir-SEG</td><td>0.77</td><td>0.56</td><td>0.57</td></tr><tr><td>CVC-ClinicDB</td><td>0.68</td><td>0.44</td><td>0.48</td></tr><tr><td>CVC-300</td><td></td><td></td><td>0.02*</td></tr><tr><td>BKAI-IGH</td><td>0.48</td><td>0.18</td><td>0.28</td></tr><tr><td>CVC-ColonDB</td><td></td><td></td><td>0.35</td></tr><tr><td>BUSI</td><td>0.72</td><td>0.48</td><td>0.47</td></tr><tr><td>QaTa-COV19†</td><td>0.74</td><td>0.50</td><td>0.52</td></tr><tr><td colspan="4">Kvasir-SEG training-caption sets of Section 4.4$</td></tr><tr><td>Size-only</td><td>0.76</td><td>0.56</td><td></td></tr><tr><td>Size-blind†</td><td>0.37</td><td>0.16</td><td></td></tr><tr><td>Mismatched</td><td>-0.01</td><td>0.00</td><td>=</td></tr><tr><td>Text-free</td><td>0.00</td><td>0.00</td><td></td></tr><tr><td colspan="4">BUSI training-caption sets$¶</td></tr><tr><td>Size-only</td><td>0.66</td><td>0.45</td><td></td></tr><tr><td>Size-blind†</td><td>0.29</td><td>0.21</td><td>=</td></tr><tr><td>Mismatched</td><td>-0.01</td><td>0.00</td><td></td></tr><tr><td colspan="4"> $Q a T a – C O V I 9$  training-caption</td></tr><tr><td> ${ \bf Z o n e - o n l y ^ { \dag } }$ </td><td>0.71</td><td>0.47</td><td></td></tr><tr><td> ${ \mathrm { Z o n e - b l i n d } } ^ { \dagger }$ </td><td>0.47</td><td>0.19</td><td>=</td></tr></table>

Images with a lesion; test-set $R _ { \mathrm { c a p } } ^ { 2 }$ in Table 2. With shufled captions, $R _ { \mathrm { c a p } } ^ { 2 } \ \leq$ 0.00. $I _ { \mathrm { e x t } } \colon$ mutual information between extent token and area tercile, relative to the tercile entropy (Section 4.6); permutation $p = 0 . 0 0 5$ except for CVC-300 $( ^ { * } p = 0 . 3 1 )$ , Mismatched and Text-free. †No size word: the whole caption is the extent token. §Training $R _ { \mathrm { c a p } } ^ { 2 } { \mathrm { : } }$ x-axis of Fig. 10(d). ¶Whole caption as extent token, so $I _ { \mathrm { e x t } }$ is not comparable with the BUSI row above. -: not scored.

$R _ { \mathrm { c a p } } ^ { 2 }$ between 0.54 and 0.77 on six of the seven test sets (Table 2) and between 0.48 and 0.77 on the five training-caption sets (Table 4); shufled captions score at most zero. The predictions fall into one band per size word (Fig. 10(a)-(c)). The modelfree measure $I _ { \mathrm { e x t } }$ agrees: it lies between 0.18 and 0.57 for every released caption set except CVC-300, each with the smallest possible <sub>�</sub>-value (Table 4).

CVC-300 scores low because 58 of its 60 captions say small, so the size word can explain at most 0.24 of the variance in log lesion area, and its $I _ { \mathrm { e x t } } ( 0 . 0 2 , p = 0 . 3 1 )$ cannot be told apart from chance. Yet the model still reads the size word $( S _ { \mathrm { e x t } } = 8 5 . 9 \% )$ .

![](images/9e1f12105bed641aa14e4cd42e95c145345e31bcffb71eef6cd664893b682ed7.jpg)

(d) leakage of the training captions and text gain  
![](images/35ac35de46c837ad3507f343b52c70226f88b3a7fd73fe04237731abf8317036.jpg)  
Figure 10: Caption-only leakage score. (a)-(c) Lesion area predicted from the caption words alone (out-of-fold ridge regression) against the true area; dashed line: identity. (d) The score of each training-caption set against the text gain of the models retrained on it (three-seed means, 95% CIs). Extent only: Size-only and Zone-only; extent removed: Size-blind and Zone-blind. The Mismatched and Text-free sets overlap near zero.

A low score therefore does not certify that a benchmark is clean (Feng et al., 2019).

Within each training dataset, the score ranks the edited caption sets of Section 4.4 in the same order as the text gains of the models trained on them, and puts the uninformative sets at zero (Fig. 10(d)): it predicts what a set of captions will teach. Across datasets, it does not. Kvasir-SEG’s Size-blind captions score 0.37 and leave 37.4% of the gain, while QaTa-COV19’s Zoneblind captions score 0.47 and leave only 9.0%. For template captions the score merely rediscovers Eq. (1); it is most useful for human-written captions, such as those of QaTa-COV19, and for benchmarks whose construction is not documented.

## 5.5. Performance without the test mask

No annotation-free method we tested reaches the reported performance (Table 5). Each scores below the published model with its matched caption (77.23 Dice), and the 95% interval of each diference excludes zero; this also holds without the duplicated CVC-300 images, with video sequences resampled and, where scored, on the normalised surface Dice (Appendix D). On colonoscopy, the edge-aligned threshold, which trains nothing but whose rule was chosen on the test sets, beats the Text-free models, and predicting the caption from the image is no better than training without text. On QaTa-COV19, without retuning, the threshold reaches 75.66, between the published model’s 60.24 with the empty caption and 83.41 with the matched one; the Text-free models reach 79.20.

ExtentSeg is the best annotation-free method (74.59 Dice), 1.94 points [+1.36, +2.55] above the Text-free models and 2.63 below the published model. On the unseen centres, the area loss adds 0.84 points [+0.23, +1.43] (ExtentSeg against ExtentSeg without it), and the estimated extent adds 0.59 points [ 0.02, +1.18] (ExtentSeg against the extent-free control). The second gain is positive in seven of nine seed pairings, but its interval includes zero, so it misses its registered bar (Table B.1). Given the true extent, ExtentSeg matches the retrained Original models with the matched caption (75.85 against 76.00; diference 0.16 [ 0.84, +0.52]). For models trained the same way, a twenty-level extent code with no language model therefore carries the benefit of the caption. About half of ExtentSeg’s gap to the published result (1.26 of 2.63 points) is the cost of estimating the extent from the image. The rest roughly equals the gap between the published checkpoint and our retrained Original models (1.22 points), which is indistinguishable from zero when video sequences are resampled ([ 2.61, +0.25]; Table D.2).

Table 5: Methods with and without access to the test mask.
<table><tr><td colspan="2"></td><td>Dice</td><td></td></tr><tr><td colspan="2">Method</td><td></td><td>all unseen ∆, all (95% CI)</td></tr><tr><td colspan="2">Annotation-free</td><td></td><td></td></tr><tr><td>Published, empty caption</td><td>63.21</td><td></td><td>60.14 -14.02 [-15.63, -12.41]</td></tr><tr><td>Edge-aligned threshold</td><td>73.84</td><td></td><td>71.41 -3.39 [-4.45, -2.39]</td></tr><tr><td>Image-only describer</td><td>72.38</td><td></td><td>69.71 -4.84 [-5.91, -3.80]</td></tr><tr><td>Text-free (trained without text)</td><td>72.65</td><td></td><td>70.14 -4.57 [-5.67, -3.52]</td></tr><tr><td>Extent-free control</td><td>74.09</td><td></td><td>71.67 −3.14 [-4.19, -2.10]</td></tr><tr><td>ExtentSeg without area loss</td><td>73.85</td><td></td><td>71.42 -3.38 [-4.39, -2.35]</td></tr><tr><td>ExtentSeg</td><td>74.59</td><td></td><td>72.26 -2.63 [-3.62, -1.68]</td></tr><tr><td colspan="2">Uses the test mask</td><td></td><td></td></tr><tr><td>ExtentSeg, true extent†</td><td>75.85</td><td></td><td>73.64 -1.38 [-2.27, -0.50]</td></tr><tr><td>Retrained, matched caption†</td><td>76.00</td><td></td><td>73.65 -1.22 [−2.01, −0.44]</td></tr><tr><td>Published, matched caption†</td><td>77.23</td><td>75.00-</td><td></td></tr></table>

All five colonoscopy test sets (� = 681) and the unseen centres (� = 581). Δ: paired diference from the published model with its matched caption. Trained models: mean of three seeds. Best annotation-free result in bold. †Reads the test mask, directly or through its caption.

## 6. Discussion

## 6.1. What the reported numbers measure

The scores reported with mask-derived captions mix three things: segmentation from the image, a dataset-level correction of how much the model segments, and the lesion extent read from the caption. Only the first is what a segmentation benchmark sets out to measure. The second needs no caption of the image itself, as the mismatched caption shows. The third is not available in deployment, where the mask is what the model must produce, yet it accounts for most of what the caption contributes on colonoscopy and chest X-ray; on BUSI, the captions also carry the diagnosis and whether a lesion is present.

Text can still help. A clinician’s description could convey extent, ExtentSeg shows that a model can use such information, and segmentation guided by the findings of routine radiology reports is now benchmarked at scale (Xi et al., 2026). The problem is where these benchmarks get the extent from. A clinician’s “large mass” is an uncertain judgement that is often not recorded at all. The benchmark’s size word is an exact threshold of the scoring mask, the zone phrase was written for the benchmark from the same images, and both are present for every test image. We did not test clinician-written descriptions; on colonoscopy and chest X-ray, the gains measured here are carried by the benchmark’s extent attribute.

## 6.2. Recommendations

We make four recommendations. (1) Captions given to a model at test time should never be computed from the test annotation. (2) Results should be reported with the empty, mismatched and matched captions, together with the prior share and, for template captions, the suficiency and necessity of each attribute, alongside the metric guidance of Reinke et al. (2024) and Maier-Hein et al. (2024). (3) The caption-only score should be computed whenever a captioned benchmark is released, as Brown II et al. (2026) propose for multimodal question answering. (4) Text-guided models should be compared with a text-free model trained with the same recipe, over more than one checkpoint.

## 6.3. Scope and threats to validity

The published models cover two architectures and three modalities; the retraining covers MedCLIPSeg on three training sets. RecLMIS was tested only at test time and only on QaTa-COV19, and LViT’s code release has no QaTa-COV19 checkpoint (Li, 2026); retraining other architectures is the next step. We showed that models trained with the extent attribute segment worse from the image, but not where in the network the deficit arises. ExtentSeg was evaluated on colonoscopy only.

The main results hold without the duplicated or mislabelled test images and when video sequences or subjects are resampled (Appendix D); the quantities that become less certain are reported with their results in Section 5. The caption-only score was not recomputed with folds grouped by video or subject. On BUSI, the published model’s extent shares rest on 65 images and change noticeably without the 15 near-duplicate lesion images $( S _ { \mathrm { e x t } } = 2 0 . 4 \% , \ : N _ { \mathrm { e x t } } = 6 7 . 7 \% )$ , while the retraining verdicts stay the same. Demographics and acquisition devices are not reported for most of the datasets, so we could not analyse those biases. Every model reads 224  224 inputs, the size fixed by the frozen encoders of the audited checkpoints, so higher input resolutions were not tested.

## 7. Conclusion

The benchmarks audited here hand text-guided segmenters, at test time, a caption that was computed from the scoring mask or written by the benchmark’s annotators, and the models have learned to read the lesion’s extent from it. On colonoscopy and chest X-ray, the benefit attributed to language is largely the benefit of knowing the size of the answer in advance; on breast ultrasound, the diagnosis word shares it. The habit is not free: on all three training sets, models trained on the full captions segment worse from the image alone than models trained with out the extent attribute, and without the test mask no method we tested reaches the reported scores. Telling language from leakage takes three checks, which we recommend as standard practice: caption edits at test time, a caption-only score, and a text-free model trained with the same recipe.

## CRediT authorship contribution statement

Niran Nataraj: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review & editing. Ankita Das: Methodology, Validation, Investigation, Writing - original draft, Writing - review & editing. C. Krishna Mohan: Resources, Supervision, Project administration, Writing - review & editing. Kenji Kawashima: Resources, Supervision, Project administration, Writing - review & editing.

## Declaration of competing interest

Kenji Kawashima holds stock in RiverField Inc., a surgical robotics company. Niran Nataraj, Ankita Das and C. Krishna Mohan declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Data availability

All seven datasets are public (Kvasir-SEG, CVC-ClinicDB, CVC-300, CVC-ColonDB, BKAI-IGH, BUSI, QaTa-COV19), as are the captions and checkpoints released with MedCLIPSeg (Koleilat, 2026a,b) and RecLMIS (Huang, 2025). The edited caption sets, per-image score tables, the analysis code that generates every number and figure in this article, and the criteria of Appendix A will be made public at https://github.com/ felirox/ExtentProbe on acceptance; until then the reposi tory is available to the editors and reviewers on request. The authors’ code is released under the PolyForm Noncommercial License 1.0.0; the patch to the MedCLIPSeg code keeps that code’s Apache License 2.0, and the derived caption sets keep the licences of the captions they are built from (MedVLSM and LViT, MIT License). The dataset images are not redistributed; they are obtained from their original sources.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Grammarly to improve the clarity, readability, and academic presentation of the English text. After using this service, the authors reviewed and edited the manuscript as needed and take full responsibility for the content of the published article.

Table A.1: Registration record of the criteria in Table B.1.
<table><tr><td>Criteria</td><td>Written (dated record)</td><td>First commit containing them</td><td>First run</td><td>First result seen</td><td>Deciding result</td></tr><tr><td>Retraining, Kvasir-SEG (R0-R4)</td><td>research plan, dated by its author 22 Sep about 16:30 IST, while the first Original</td><td>e19ee2a, 22 Sep 23:06 IST (audit script with the bars)</td><td>22 Sep 16:16 IST (first Original run, by the project log; its log was</td><td>seed 1: R0 22 Sep 18:40, R1-R2 21:12, R3 23:35</td><td>three seeds: R0-R3 23 Sep 13:02-15:03 IST; R4 23 Sep 02:10 IST IST; R4 7 Oct</td></tr><tr><td>Retraining, BUSI and</td><td>run was training declaration, 5 Oct; Amendment 1 (R4 intervals,</td><td>9f31c22/e2c2051, 5 Oct 16:14-16:15 JST;</td><td>last written at 17:20 IST) queue started 5 Oct 16:20 JST with two Kvasir-SEG</td><td>BUSI 7 Oct;</td><td>BUSI 7 Oct;</td></tr><tr><td></td><td></td><td></td><td>Mismatched seeds; first BUSI run after 17:15 JST; first QaTa-COV19 run 6 Oct first run queued 23 Sep seed 1: 23 Sep 11:40 IST three seeds: 24 Sep</td><td></td><td></td></tr><tr><td>published model, and its area loss ExtentSeg against its</td><td>01:28 IST, before any output IST of the model research plan, 23 Sep,</td><td>d7160da, 23 Sep 21:13</td><td>01:31 IST control built 23 Sep</td><td>seed 1: 23 Sep 21:20 IST</td><td>01:40 IST three seeds: after 24</td></tr><tr><td>extent-free control RecLMIS</td><td>control research plan, 22 Sep,</td><td>bar) e19ee2a, 22 Sep 23:06</td><td></td><td></td><td>Sep 01:25 IST, when seeds 2-3 of the control were launched 22 Sep</td></tr><tr><td>(reproduction, extent necessity) Caption style</td><td>before either was run research plan, 22 Sep,</td><td>IST</td><td>checkpoint) - (published</td><td>22 Sep</td><td></td></tr></table>

## Appendix A. Registration record

Here, registration means that a criterion was written in the project’s dated research record before the results it decides were seen, and was committed to version control; no external registry was used. For each family of registered criteria reported in this paper, Table A.1 gives where the criteria were written, the first commit that contains them, and when the first and the deciding results were seen. Commit times are those recorded by version control, in India Standard Time (IST, UTC+5:30, where UTC is Coordinated Universal Time) for September 2026 and Japan Standard Time (JST, UTC+9) for October 2026; other times come from the dated research record (the project’s wiki and run logs). The text of every registered criterion, with its commit time, is included in the code release (see Data availability).

For BUSI and QaTa-COV19 the record is complete in version control: the criteria and their amendment were committed before the first of these models began training. For the other families, the dated research record places the criteria before the results they decide (for Kvasir-SEG, after the first Original run had started and before it finished), but the first commit that contains them came later. The Kvasir-SEG criteria were first committed after the single-seed results of R0-R2 had been seen and before those of R3 and R4 and every three-seed result; the criteria for ExtentSeg against the published model, after the seed-1 result and before the three-seed result; and those for RecLMIS and the caption-style test, after the results themselves. The bar for ExtentSeg against its extent-free control was committed before its first result. For RecLMIS, the matched caption in MedCLIPSeg’s wording had been partly run by mistake, and its running mean seen, before the criteria were written; that condition is therefore reported as an observation (Appendix D), not as a test.

## Appendix B. Registered criteria

Table B.1 lists every registered criterion whose analysis appears in this paper, with its bar, its result and its verdict. Criteria registered while ExtentSeg was being developed, each tested on one seed, are not in the table and are summarised here. Two concerned caption-based variants that are not reported; neither was met. Six concerned ExtentSeg itself, its decoding or its controls. One of them was met: ExtentSeg without the area loss outscores the best caption-based variant. The other five were not met ExtentSeg did not beat a fixed ofset of the output logits for each extent bin by two points, so most of what its extent input provides acts like such an ofset. Re-estimating the extent from the predicted mask, mixing the outputs of the conditioned and unconditioned passes, and an extent head that also reads the area of its own map each failed to improve Dice. Finally, the area loss did not improve the extent-free control on its first seed.

Table B.1: Registered criteria reported in this paper.
<table><tr><td>Criterion</td><td>Bar</td><td>Result</td><td>Verdict</td></tr><tr><td>R0-R4 and ρ, retraining on three training sets</td><td>Section 4.4</td><td>Table 3</td><td>Kvasir-SEG: R0-R4 met (ρ not registered). QaTa-COV19: every criterion run was met. BUSI: R0, R1 and ρ met; R2 consistent, and met on the second population specified in</td></tr><tr><td>ExtentSeg against the published model</td><td>ExtentSeg without area loss (the registered model) minus the published model with its matched caption, unseen centres: interval above zero</td><td>-3.58 [-4.76, -2.36]; with the area loss, -2.74 [-3.84, -1.61]</td><td>not met</td></tr><tr><td>Area loss of ExtentSeg</td><td>ExtentSeg minus ExtentSeg without area loss, unseen centres: at least +0.5, interval excluding zero</td><td>+0.84 [+0.23, +1.43]</td><td>met</td></tr><tr><td>Extent input of ExtentSeg</td><td>ExtentSeg minus its extent-free control, unseen centres: at least +0.5, interval excluding zero</td><td>+0.59 [−0.02, +1.18]</td><td>not met: positive in seven of nine seed pairings, interval includes zero (Section 5.5)</td></tr><tr><td>RecLMIS reproduction</td><td>Dice with its own captions within 1.0 of the reported 85.22</td><td>85.21</td><td>met</td></tr><tr><td>RecLMIS extent necessity</td><td>deleting the zone phrase costs at least 50% of the 43.6% [41.3,45.8] text gain, deletion written in MedCLIPSeg&#x27;s wording</td><td></td><td>not met; in RecLMIS&#x27;s own wording the necessity is 58.0% (Table 2)</td></tr><tr><td>Caption style</td><td>the four restyles of the same content (terse, verbose, clinical, list) each within 1.0 point of the Table D.4) terse one, pooled over the seven test sets, each paired interval within ±1.5</td><td>largest difference 1.5 points (list style; not met</td><td></td></tr></table>

Retrained models and ExtentSeg: three-seed means; RecLMIS and caption style: published checkpoints; image bootstrap throughout. The RecLMIS necessity criterion deleted the zone phrase in MedCLIPSeg’s wording, not RecLMIS’s own (Table 2 gives the matched value). The caption-style criterion missed only through the list style, which is not a sentence.

## Appendix C. Implementation and reproducibility

All retrained models start from the released UniMed-CLIP ViT-B/16 weights (SHA-256 prefix bfdce18ca2473c56, where SHA-256 is the 256-bit Secure Hash Algorithm) and are trained with the MedCLIPSeg code (Koleilat, 2026c) at commit 31d2220 with a local patch that adds ExtentSeg (patch digest 4bdb08c88d52534b, SHA-256 prefix). The released version of the patch (digest 7de4a5d0d2762fb6) keeps only the code these runs use: with ExtentSeg of it gives the outputs of the published models bit for bit (32 test images each of Kvasir-SEG and BUSI), and on the same inputs and seeds it reproduces the probability maps of the original patch and, after one training epoch, its weights bit for bit; the training-caption sets are passed through the released configuration options. Models are trained and run with PyTorch (Paszke et al., 2019); the regressions and classifiers of Sections 4.6 and 4.7 and of Appendix E use scikit-learn (Pedregosa et al., 2011). Inference with a fixed seed is deterministic on a given machine: on eight CVC-300 images, two runs of the 30-pass stochastic inference with the same seed give bitwise identical probability maps and a diferent seed changes them; across machines the maps can difer slightly. The SHA-256 digests of the published checkpoints and of the last-epoch checkpoint of every retrained model are part of the code release (paper/data/ckpt\_digests.json; see Data availability); the digests of the four published checkpoints equal those of the files distributed by their authors (checked on 22 October 2026).

The provenance of the colonoscopy and BUSI captions (Section 3.2) was checked by comparing the caption files released with MedCLIPSeg with the prompt files of Poudel et al. (2024) (commit dc36a3a; Poudel, 2024): all 3852 rows of the original training, validation and test caption files of the six colonoscopy and BUSI sets equal a P6 prompt of the same image (for BUSI, one of the two P6 variants stored for most images).

The BUSI and QaTa-COV19 retraining and the Kvasir-SEG Mismatched seeds 2 and 3 ran on an NVIDIA RTX 5000 Ada graphics processing unit; one-epoch timings on that card put a 100-epoch run at about 1.0-1.2 h for Kvasir-SEG, 0.8-0.9 h for BUSI and 6.5 h for QaTa-COV19, with a peak memory of about 9.6 GB (project run log of 5 October 2026). The other Kvasir-SEG retrained models, ExtentSeg and its controls were trained on one NVIDIA Quadro RTX 6000 (24 GB). All runs used Python 3.13.9, PyTorch 2.12.0 built for CUDA 12.6, Transformers 4.56.0 and NumPy 2.5.3.

RecLMIS is run from its released code (Huang, 2025) at commit d3265c8 with the released QaTa-COV19 checkpoint and test script (Section 3). Two changes, neither on the segmentation path, were needed: the attention module imports a softmax from fairseq, which is not among the released requirements and does not build here, so that one call is inlined as torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32); and the image file names difer between the RecLMIS and MedCLIPSeg distributions of the dataset, which is handled with symbolic links rather than by editing the data. A third change makes the test script write an optional per-image score file. The code changes and the script that creates the links are part of the code release (see Data availability).

Further method details. The released MedCLIPSeg evaluation configuration loads the latest checkpoint file (TEST.USE\_LATEST); the epoch of the published checkpoints is not stated. The ridge penalty of the caption-only score (Section 4.6) is chosen by leave one-out cross-validation inside each training fold, and its chance level averages five shufles. On QaTa-COV19 the edge-aligned threshold averages seven fixed captions: “Bilateral pulmonary infection involving two regions with involvement of . . . ” completed with each of “the lower right lung”, “the lower left lung and the lower right lung”, “the middle lower left lung and the middle lower right lung” and “all left lung and all right lung”, and “a small portion of the lungs” with each of its three size words. In ExtentSeg, c holds the geometric centres of the bins, with the outer bins closed at half the first edge and at 1.5 times the last; the area term excludes the 40 smallest of the 800 training lesions (those below the first bin edge, whatever rotation the training augmentation applies).

## Appendix D. Sensitivity analyses

Each analysis in this appendix recomputes quantities of Sections 5.1-5.5 from the same per-image scores, with the bootstrap of Section 4.9; none trains a new model.

Video-sequence clusters. Colonoscopy clusters are formed from the images alone, without reference to any score. CVC-ColonDB and CVC-ClinicDB test frames are chained in file order, and a new cluster starts at a gap of more than three file numbers or where the correlation of adjacent 32 32 grey-level images falls below 0.6 (0.5 and 0.7 in Table D.2). CVC-300 images that repeat a CVC-ColonDB image join its cluster, the other CVC-300 images are chained in the same way, and clusters that share a byte-identica image are merged. Kvasir-SEG and BKAI-IGH have no sequence information, so each of their images is its own cluster. Clusters are resampled within each test set, and a cluster that spans CVC-300 and CVC-ColonDB is drawn with CVC-ColonDB.

Duplicated and mislabelled test images. Table D.1 removes the duplicated, near-duplicated and mislabelled test images of Section 3.1. Without the 55 duplicated CVC-300 images, no Δ of Table 5 moves by more than 0.33 points, and ExtentSeg remains the best annotation-free method (74.12 Dice, $\Delta = - 2 . 7 0 [ - 3 . 7 3 , - 1 . 6 8 ] )$ .

Video sequences and subjects. Table D.2 resamples colonoscopy video sequences instead of images. Resampling the 474 QaTa-COV19 test subjects instead of the 2113 images moves no interval bound of the retrained or published MedCLIPSeg models by more than 1.1 points: R2 is [6.8, 11.4]%, R3 [88.5, 93.7]%, <sub>�Zone-blind</sub> [+10.31, +12.24], <sub>�Zone-only</sub> [ 0.79, +0.002] (its upper bound now just above zero) and � [40.2, 49.4]%. For RecLMIS, the prior-share interval widens most, from [8.9, 19.8]% to [5.1, 22.8]%, and $S _ { \mathrm { e x t } }$ is [85.3, 90.1]%.

Training seeds. When the three seeds of each model are resampled together with the images, the colonoscopy empty-caption diferences stay above zero: $\rho _ { \mathrm { S i z e - b l i n d } }$ is [+1.12, +3.67] and $\rho _ { \mathrm { T e x t - f r e e } } \left[ + 1 . 5 4 , + 4 . 1 1 \right]$ , while R2 is [26.0, 51.3]% and R3 [50.7, 132.9]%, so both intervals cross their bars. On BUSI, R2 is [18.1, 79.7]% and R3 [12.7, 124.4]% on the 65 lesion images, so both intervals cross their bars (R2 meets its bar in six of the nine seed pairings, R3 in two). <sub>�Size-blind</sub>, <sub>�Mismatched</sub> and <sub>�Text-free</sub> stay above zero ([+1.12, +17.65], [+5.81, +23.10] and [+5.73, +23.00]), while $\rho _ { \mathrm { S i z e - o n l y } }$ includes zero $( [ - 3 . 0 1 , + 1 7 . 0 1 ] )$ . On QaTa-COV19, R2 is [6.3, 13.1]% and R3 [64.8, 133.1]%, so only R3 crosses its bar; $\rho _ { \mathrm { Z o n e - b l i n d } }$ is [+6.77, +14.59] and $\rho _ { \mathrm { Z o n e - o n l y } }$ [ 5.24, +4.15]. Table D.6 gives the retrained models seed by seed.

Checkpoint selection. When each Kvasir-SEG run’s checkpoint is selected by validation Dice instead of taken at the last epoch, $G _ { \mathrm { O r i g i n a l } } = 5 . 9 9 [ 5 . 1 0 , 6 . 9 0 ]$ , R2 is 36.2% [26.8, 45.6], R3 100.1% [90.3, 111.3] and $\rho _ { \mathrm { T e x t - f r e e } } = + 2 . 4 9 \ [ + 1 . 8 4 , + 3 . 1 5 ]$

Other scores and tests. Table D.3 repeats Table 2 on the normalised surface Dice. On that score ExtentSeg is again the best of the annotation-free methods of Table 5 that were scored on it (77.36, against 80.12 for the published model with its matched caption). Table D.4 gives the caption-style test, and Table D.5 the label-free rules from which the edge-aligned threshold was chosen.

RecLMIS. On QaTa-COV19 (� = 2113), RecLMIS scores 83.87 Dice with the matched caption in MedCLIPSeg’s wording, against 85.21 in its own. In the LViT wording, it scores 83.43 with the zone phrase alone $( S _ { \mathrm { e x t } } ~ [ 8 6 . 1 , 8 9 . 4 ] \% )$ and 76.70 with the zone phrase deleted $( N _ { \mathrm { e x t } } [ 5 5 . 8 , 6 0 . 2 ] \% ) ;$ the registered deletion in MedCLIPSeg’s wording gives 78.82. Forcing the zone phrase to one or to six zones gives 75.99 and 71.54. With matched and mismatched captions both taken from MedCLIPSeg’s set, its prior share is 15.9% [9.9, 21.7].

Mismatched captions identical to the image’s own. Outside QaTa-COV19 such captions are rare (at most 16 test images per test set). Removing them changes � by at most 3.1 points on any colonoscopy test set (pooled: 36.6% [23.1, 48.2]) and moves BUSI further below zero ( 106.6%). QaTa-COV19 is reported in Section 5.1. The excluded images are those whose caption is shared with other test images, not a random subset, so these values show where the prior gain comes from; they do not estimate it for a permutation of caption texts.

Table D.1: Main quantities without duplicated or mislabelled test images.
<table><tr><td></td><td colspan="2">Colonoscopy, all (n = 626)</td><td colspan="2">Unseen centres (n = 526)</td><td colspan="2"> $\mathrm { W i t h o u t } \ ^ { \cdot \cdot } n o p o l y p ^ { \prime \cdot } \left( n = 6 7 8 \right)$ </td><td colspan="2">BUSI lesions (n = 50)</td></tr><tr><td>Quantity</td><td>value (95% CI)</td><td>change</td><td>value (95% CI)</td><td>change</td><td>value (95% CI)</td><td>change</td><td>value (95% CI)</td><td>change</td></tr><tr><td>G (points)</td><td>13.92 [12.20, 15.69]</td><td>-0.10</td><td>14.84 [12.90, 16.79]</td><td>-0.03</td><td> $1 4 . 0 3 [ 1 2 . 4 0 , 1 5 . 7 1 ]$ </td><td>+0.01</td><td>10.13 [5.32, 15.37]</td><td>-0.51</td></tr><tr><td>P (points)</td><td>+4.86 [+2.65, +7.03]</td><td>-0.71</td><td>+5.78 [+3.30, +8.25]</td><td>-0.74</td><td> $+ 5 . 6 2 \ [ + 3 . 5 4 , + 7 . 6 5 ]$ </td><td>+0.06</td><td> $- 0 . 6 3 \ : [ - 7 . 5 1 , + 5 . 9 4 ]$ </td><td>+3.27</td></tr><tr><td>π(%)</td><td>34.9 [20.7, 47.0]</td><td>-4.8</td><td>39.0 [24.2, 51.6]</td><td>-4.9</td><td>40.1 [27.4, 51.0]</td><td>+0.4</td><td> $- 6 . 3 \ : [ - 1 0 1 . 1 , 5 4 . 2 ]$ </td><td>+30.5</td></tr><tr><td> $S _ { \mathrm { e x t } } \left( \% \right)$ </td><td>94.3 [90.7,97.9]</td><td>+0.8</td><td>95.0 [91.1, 99.0]</td><td>+0.9</td><td>94.1 [90.7,97.4]</td><td>+0.6</td><td>20.4 [–53.2, 63.2]</td><td>-10.2</td></tr><tr><td> $N _ { \mathrm { e x t } } ( \% )$ </td><td>69.0 [63.1, 74.6]</td><td>+0.2</td><td>71.0 [64.7, 77.1]</td><td>+0.4</td><td>69.1 [63.5, 74.4]</td><td>+0.3</td><td>67.7 [35.4, 138.6]</td><td>+10.6</td></tr><tr><td> ${ \mathrm { R } } 2 \ ( { \% } )$ </td><td>39.2 [30.1, 49.2]</td><td>+1.8</td><td>36.7 [27.1, 46.8]</td><td>+1.8</td><td></td><td></td><td>43.2 [21.4, 68.1]</td><td>+4.1</td></tr><tr><td>R3 (%)</td><td>90.1 [80.6, 100.6]</td><td>+0.4</td><td>89.1 [79.1, 100.4]</td><td>+0.4</td><td></td><td></td><td>52.7 [37.5, 65.6]</td><td>+2.8</td></tr><tr><td>ρSize-blind</td><td> $+ 2 . 1 0 \ [ + 1 . 2 7 , + 2 . 9 3 ]$ </td><td>-0.29</td><td>+2.39 [+1.45, +3.32]</td><td>-0.31</td><td></td><td></td><td> $+ 7 . 0 8 \ [ + 2 . 5 1 , + 1 2 . 2 1 ]$ </td><td>-1.34</td></tr><tr><td> $\rho _ { \mathrm { S i z e - o n l y } }$ </td><td> $- 0 . 0 2 \ [ - 0 . 6 1 , + 0 . 5 9 ]$ </td><td>+0.07</td><td> $+ 0 . 0 7 \ [ - 0 . 6 1 , + 0 . 7 7 ]$ </td><td>+0.09</td><td></td><td></td><td> $+ 5 . 6 1 \ [ + 3 . 0 3 , + 8 . 4 7 ] - 1 . 1 2$ </td><td></td></tr><tr><td>ρMismatched</td><td> $+ 2 . 9 8 \ [ + 2 . 1 9 , + 3 . 7 8 ]$ </td><td>-0.16</td><td> $+ 3 . 2 0 \ [ + 2 . 3 2 , + 4 . 1 2 ]$ </td><td>-0.17</td><td></td><td></td><td> $+ 1 2 . 3 8 \ [ + 6 . 9 2 , + 1 8 . 2 8 ] - 1 . 1 9$ </td><td></td></tr><tr><td>ρText-free</td><td>+2.68 [+1.92, +3.47]</td><td>-0.15</td><td> $+ 2 . 9 5 \ [ + 2 . 1 0 , + 3 . 8 2 ]$ </td><td>-0.15</td><td></td><td></td><td> $+ 1 2 . 2 3 \ [ + 6 . 5 1 , + 1 8 . 3 7 ]$ </td><td>-1.20</td></tr></table>

Recomputed on reduced populations with the image bootstrap. Colonoscopy: without the 55 CVC-300 images that are also CVC-ColonDB images. “no polyp”: without the three BKAI-IGH images so captioned despite a polyp in the mask. BUSI: lesion images without the 15 near-duplicates of training or validation images. Change: value minus the full-population value (percentage points for �, shares and ratios).

Table D.2: Colonoscopy intervals when video sequences are resampled instead of images.
<table><tr><td>Quantity (n = 681)</td><td>Estimate</td><td>Images</td><td>Sequences, 0.6</td><td>Sequences, 0.5</td><td>Sequences, 0.7</td></tr><tr><td>G (points)</td><td>14.02</td><td>[12.41, 15.63]</td><td>[11.32, 16.57]</td><td>[11.11,16.80]</td><td>[11.52, 16.39]</td></tr><tr><td>P (points)</td><td>+5.56</td><td>[+3.53,+7.61]</td><td> $[ + 0 . 5 9 , + 9 . 1 7 ]$ </td><td>[-0.08, +9.56]</td><td>[+1.20, +8.92]</td></tr><tr><td>π (%)</td><td>39.7</td><td>[27.3,50.8]</td><td>[4.9,58.6]</td><td>[-0.7,59.6]</td><td>[10.0,58.0]</td></tr><tr><td>Sext (%)</td><td>93.5</td><td>[90.0,97.0]</td><td>[88.9,99.4]</td><td>[88.6, 100.1]</td><td>[88.7,99.5]</td></tr><tr><td> $N _ { \mathrm { e x t } } ( \% )$ </td><td>68.8</td><td>[63.3,74.2]</td><td>[62.2,77.1]</td><td>[61.7,77.2]</td><td>[62.1,75.3]</td></tr><tr><td>R2 (%)</td><td>37.4</td><td>[28.5,46.8]</td><td>[26.3,51.5]</td><td>[26.7,51.4]</td><td>[25.0,53.1]</td></tr><tr><td>R3 (%)</td><td>89.7</td><td>[80.4,99.8]</td><td>[77.9, 101.8]</td><td>[79.2, 101.9]</td><td>[78.0, 101.4]</td></tr><tr><td>ρSize-blind</td><td>+2.39</td><td>[+1.62, +3.16]</td><td> $[ + 0 . 7 9 , + 3 . 4 6 ]$ </td><td>[+0.75,+3.39]</td><td>[+0.82, +3.67]</td></tr><tr><td>ρSize-only</td><td>-0.09</td><td>[-0.66, +0.48]</td><td> $[ - 1 . 0 0 , + 0 . 9 3 ]$ </td><td> $[ - 1 . 1 7 , + 0 . 9 4 ]$ </td><td>[-0.96, +0.97]</td></tr><tr><td>ρMismatched</td><td>+3.14</td><td>[+2.41, +3.86]</td><td> $[ + 2 . 0 6 , + 3 . 9 6 ]$ </td><td>[+1.98, +4.00]</td><td>[+2.00, +4.11]</td></tr><tr><td>ρText-free</td><td>+2.82</td><td>[+2.11,+3.53]</td><td> $[ + 1 . 6 6 , + 3 . 6 2 ]$ </td><td>[+1.67,+3.61]</td><td>[+1.71,+3.77]</td></tr><tr><td colspan="6">∆ of Table 5, all five test sets</td></tr><tr><td>Published, empty caption</td><td>-14.02</td><td>[-15.63,-12.41]</td><td>[-16.57,-11.32]</td><td>[-16.80,-11.11]</td><td>[-16.39,-11.52]</td></tr><tr><td>Edge-aligned threshold</td><td>-3.39</td><td>[-4.45, -2.39]</td><td>[-4.73,-1.73]</td><td>[-4.66, -1.89]</td><td>[-4.94,-1.75]</td></tr><tr><td>Image-only describer</td><td>-4.84</td><td>[-5.91,-3.80]</td><td>[-7.71,-2.91]</td><td>[-7.97,-2.85]</td><td>[-7.33,-3.10]</td></tr><tr><td>Text-free (trained without text)</td><td>-4.57</td><td>[-5.67, -3.52]</td><td> $[ - 6 . 5 4 , - 3 . 1 7 ]$ </td><td> $[ - 6 . 6 4 , - 3 . 0 8 ]$ </td><td>[-6.24,-3.20]</td></tr><tr><td>Extent-free control</td><td>-3.14</td><td>[-4.19,-2.10]</td><td>[-4.91,-1.85]</td><td>[-5.04, -1.82]</td><td>[-4.77,-1.77]</td></tr><tr><td>ExtentSeg without area loss</td><td>-3.38</td><td>[-4.39, -2.35]</td><td>[-4.89, -2.29]</td><td>[-4.87,-2.25]</td><td>[-4.73,-2.20]</td></tr><tr><td>ExtentSeg</td><td>-2.63</td><td>[-3.62, -1.68]</td><td>[-3.95,-1.46]</td><td>[-4.04, -1.38]</td><td>[-3.86,-1.45]</td></tr><tr><td>ExtentSeg, true extent†</td><td>-1.38</td><td>[-2.27,-0.50]</td><td>[-2.38, -0.31]</td><td>[-2.46, -0.17]</td><td>[-2.37, -0.26]</td></tr><tr><td>Retrained, matched caption†</td><td>-1.22</td><td>[-2.01, -0.44]</td><td>[-2.61,+0.25]</td><td>[-2.51,+0.26]</td><td>[-2.54, +0.22]</td></tr></table>

95% intervals. Sequences: clusters at a correlation threshold of 0.6 (primary; 273 clusters), 0.5 (266) and 0.7 (298), resampled within each test set; the estimates do not change. †Reads the test mask.

Table D.3: Table 2 on the normalised surface Dice, and RecLMIS on the intersection over union.
<table><tr><td>Test set</td><td>Empty</td><td>Mism.</td><td>Match.</td><td>G (95% CI)</td><td>π (95% CI)</td><td> $S _ { \mathrm { e x t } } \left( 9 5 \% \mathrm { C I } \right)$ </td><td> $N _ { \mathrm { e x t } } ( 9 5 \% \mathrm { C I } )$ </td></tr><tr><td colspan="8">MedCLIPSeg, normalised surface Dice</td></tr><tr><td>Kvasir-SEG</td><td>82.84</td><td>83.10</td><td>92.33</td><td>9.49 [5.88, 13.16]</td><td>2.7 [–55.0, 36.7]</td><td>88.1 [78.5,95.3]</td><td>52.3 [32.1, 69.8]</td></tr><tr><td>CVC-ClinicDB</td><td>73.20</td><td>80.20</td><td>85.16</td><td>11.96 [6.16, 18.30]</td><td>58.6 [8.9,91.4]</td><td>84.1 [64.5,95.2]</td><td>69.7 [43.1, 86.1]</td></tr><tr><td>CVC-300</td><td>68.07</td><td>82.94</td><td>84.28</td><td>16.20 [12.43, 20.18]</td><td>91.8 [81.4, 100.9]</td><td>85.8 [78.0, 92.9]</td><td>67.3 [54.6, 79.2]</td></tr><tr><td>BKAI-IGH</td><td>55.55</td><td>75.41</td><td>80.92</td><td>25.37 [20.10, 30.92]</td><td>78.3 [65.7,90.4]</td><td>92.2 [86.7, 96.9]</td><td>77.5 [67.5, 87.0]</td></tr><tr><td>CVC-ColonDB</td><td>61.36</td><td>63.74</td><td>74.95</td><td>13.59 [11.25, 15.92]</td><td>17.5 [-7.3,37.2]</td><td>97.6 [92.1, 103.3]</td><td>68.5 [60.0, 76.6]</td></tr><tr><td>BUSI</td><td>74.33</td><td>61.92</td><td>88.36</td><td>14.03 [8.36, 20.50]</td><td>-88.4 [-208.4, -22.1]</td><td>30.1 [-29.5, 64.6]</td><td>58.8 [33.0, 111.0]</td></tr><tr><td>QaTa-COV19</td><td>66.23</td><td>75.80</td><td>89.17</td><td>22.94 [22.00, 23.89]</td><td>41.7 [37.7,45.5]</td><td>89.5 [87.7,91.2]</td><td>62.4 [59.9, 65.0]</td></tr><tr><td>Colonoscopy, all</td><td>65.31</td><td>71.46</td><td>80.12</td><td>14.80 [13.10, 16.50]</td><td>41.6 [29.6,52.3]</td><td>93.2 [89.8, 96.6]</td><td>69.2 [63.7,74.5]</td></tr><tr><td>Unseen centres</td><td>62.30</td><td>69.46</td><td>78.01</td><td>15.72 [13.81, 17.67]</td><td>45.6 [33.1, 56.6]</td><td>93.8 [90.1,97.3]</td><td>71.0 [65.4, 76.4]</td></tr><tr><td colspan="8">RecLMIS, intersection over union</td></tr><tr><td>QaTa-COV19</td><td>57.49</td><td>62.03</td><td>77.00</td><td>19.50 [18.64, 20.33]</td><td>23.3 [18.6, 27.6]</td><td>87.1 [85.5, 88.6]</td><td>59.2 [57.2, 61.3]</td></tr></table>

Normalised surface Dice at a tolerance of 2 pixels; populations and captions as in Table 2. ‡� and � on all 78 images, $S _ { \mathrm { e x t } }$ and $N _ { \mathrm { e x t } }$ on the 65 with a lesion. RecLMIS has no per-image surface score, so its intersection over union is shown.

Table D.4: MedCLIPSeg under captions rewritten in other styles (Dice).
<table><tr><td></td><td></td><td></td><td colspan="3">Same content, other style</td><td colspan="4">Content slots emptied</td></tr><tr><td>Test set</td><td></td><td>n Released</td><td>Terse</td><td>Verbose</td><td>Clinical</td><td></td><td>List Clinical</td><td>List</td><td>Empty</td></tr><tr><td>Kvasir-SEG</td><td>100</td><td>90.15</td><td>90.04</td><td>90.04</td><td>88.97</td><td>89.45</td><td>80.52</td><td>83.73</td><td>81.04</td></tr><tr><td>CVC-ClinicDB</td><td>61</td><td>80.80</td><td>80.17</td><td>80.34</td><td>81.18</td><td>80.85</td><td>68.73</td><td>72.22</td><td>69.92</td></tr><tr><td>CVC-300</td><td>60</td><td>80.84</td><td>80.84</td><td>80.95</td><td>77.14</td><td>78.27</td><td>63.08</td><td>67.76</td><td>65.32</td></tr><tr><td>BKAI-IGH</td><td>100</td><td>79.16</td><td>79.37</td><td>79.26</td><td>75.64</td><td>77.12</td><td>54.18</td><td>60.02</td><td>54.55</td></tr><tr><td>CVC-ColonDB</td><td>360</td><td>71.89</td><td>71.86</td><td>72.68</td><td>72.81</td><td>73.33</td><td>57.25</td><td>61.98</td><td>59.17</td></tr><tr><td>BUSI</td><td>78</td><td>85.73</td><td>85.39</td><td>85.41</td><td>81.52</td><td>78.93</td><td>71.91</td><td>73.73</td><td>71.73</td></tr><tr><td>QaTa-COV19</td><td>2113</td><td>83.41</td><td>82.65</td><td>82.50</td><td>82.10</td><td>80.81</td><td>67.08</td><td>54.69</td><td>60.24</td></tr><tr><td>Pooled</td><td>2872</td><td>82.01</td><td>81.43</td><td>81.42</td><td>80.81</td><td>79.94</td><td>65.95</td><td>57.96</td><td>61.25</td></tr><tr><td colspan="5">Pooled – released</td><td></td><td>- -0.58 [-0.70, −0.45] −0.59 [−0.82, -0.35] -1.20 [−1.48, −0.91] -2.06 [-2.39, −1.73]</td><td>-16.06</td><td>-24.04-20.75</td><td></td></tr><tr><td colspan="5">Pooled – terse</td><td>-0.01 [−0.26, +0.25] -0.62 [−0.90, −0.34] -1.49 [-1.80, −1.18]</td><td colspan="5"></td></tr></table>

Each caption is parsed into its content slots and rendered again in four styles: terse (LViT style), verbose (MedCLIPSeg style), clinical (a findings sentence) and list (no sentence). Terse is the reference of the registered criterion, because the parser drops text outside the slots. Content slots emptied: the wording is kept and the slots are emptied. Last rows: paired diferences of pooled means (95% CI).

Table D.5: Label-free rules for cutting the averaged score map of the edge-aligned threshold (Dice).
<table><tr><td>Rule</td><td>Kvasir-SEG</td><td>CVC-ClinicDB</td><td>CVC-300</td><td>BKAI-IGH</td><td>CVC-ColonDB</td><td>Colonoscopy, all</td><td>Unseen</td><td>BUSI</td><td>QaTa-COV19</td></tr><tr><td>half</td><td>85.03</td><td>75.79</td><td>71.01</td><td>62.98</td><td>64.84</td><td>69.06</td><td>66.31</td><td>63.06</td><td>76.04</td></tr><tr><td>Otsu</td><td>82.28</td><td>70.55</td><td>65.94</td><td>51.08</td><td>58.25</td><td>62.50</td><td>59.10</td><td>62.68</td><td>76.22</td></tr><tr><td>drop</td><td>82.42</td><td>74.44</td><td>77.42</td><td>77.62</td><td>63.25</td><td>70.42</td><td>68.36</td><td>62.84</td><td>75.05</td></tr><tr><td>plateau</td><td>2.67</td><td>4.44</td><td>6.35</td><td>5.08</td><td>7.07</td><td>5.83</td><td>6.38</td><td>4.07</td><td>5.29</td></tr><tr><td>agree</td><td>85.11</td><td>75.13</td><td>70.88</td><td>62.96</td><td>65.02</td><td>69.09</td><td>66.33</td><td>62.92</td><td>76.13</td></tr><tr><td>grad*</td><td>87.93</td><td>77.64</td><td>74.30</td><td>77.19</td><td>68.27</td><td>73.84</td><td>71.41</td><td>63.95</td><td>75.66</td></tr><tr><td>compact</td><td>73.26</td><td>62.55</td><td>68.95</td><td>55.25</td><td>60.94</td><td>62.77</td><td>60.96</td><td>36.47</td><td></td></tr><tr><td>gradimg</td><td>75.67</td><td>56.98</td><td>37.66</td><td>52.66</td><td>43.15</td><td>50.07</td><td>45.67</td><td>57.01</td><td></td></tr><tr><td>gradcomp</td><td>88.21</td><td>75.91</td><td>74.97</td><td>75.78</td><td>68.60</td><td>73.75</td><td>71.26</td><td>65.81</td><td></td></tr><tr><td>contrast</td><td>79.03</td><td>69.73</td><td>78.92</td><td>75.86</td><td>60.08</td><td>67.71</td><td>65.76</td><td>60.21</td><td></td></tr><tr><td>gradwin</td><td>86.38</td><td>75.94</td><td>73.17</td><td>65.77</td><td>66.94</td><td>70.98</td><td>68.33</td><td>62.56</td><td></td></tr><tr><td>gdrop</td><td>85.11</td><td>76.28</td><td>76.06</td><td>77.76</td><td>65.05</td><td>71.84</td><td>69.56</td><td>63.66</td><td>75.40</td></tr><tr><td>Published, empty caption</td><td>81.04</td><td>69.92</td><td>65.32</td><td>54.55</td><td>59.17</td><td>63.21</td><td>60.14</td><td>71.73</td><td>60.24</td></tr><tr><td>Published, matched caption†</td><td>90.15</td><td>80.80</td><td>80.84</td><td>79.16</td><td>71.89</td><td>77.23</td><td>75.00</td><td>85.73</td><td>83.41</td></tr><tr><td>Cut at the true area</td><td>91.20</td><td>85.83</td><td>82.47</td><td>79.96</td><td>73.75</td><td>79.08</td><td>76.99</td><td>85.30</td><td>81.93</td></tr><tr><td>Best threshold†</td><td>91.74</td><td>87.12</td><td>83.90</td><td>81.59</td><td>75.86</td><td>80.75</td><td>78.86</td><td>86.40</td><td>83.72</td></tr></table>

Each rule sets how many top-scoring pixels to keep. half: � > 0.5; Otsu: Otsu (1979); drop: steepest fall of the sorted scores; plateau: area most stable to the threshold; agree: best agreement with the single-caption masks; grad: steepest mean boundary gradient, Eq. (9); compact: compactness times mean score; gradimg: image gradient on the boundary; gradcomp: grad times compactness; contrast: mean score inside minus outside; gradwin: grad within a factor of four of the empty-caption area; gdrop: geometric mean of the grad and drop areas. ∗Rule used in the paper. -: not evaluated. †Uses the test mask.

## Appendix E. Provenance of the QaTa-COV19 extent phrase

The QaTa-COV19 captions were written for the benchmark by LViT’s annotators (Li et al., 2024); the dataset contains no radiology reports. To test whether the zone phrase is nonetheless a fixed function of the mask, we compared it with zone sets computed from the mask and a lung field. The lung field comes from the ChestX-Det lung segmentation network of TorchXRayVision (Cohen et al., 2022; Lian et al., 2021), with image thirds as a fallback. The analysis plan was fixed before any rule or classifier was compared with a test caption.

Sides. In training captions that name one side, “left” captions place 4.7% of the mask pixels in the image-right half, and “right” captions 87.5%. The annotations therefore name sides by image position, the opposite of the radiological convention; test images with a visible cardiac silhouette and side markers confirm this.

Zones. The overlap rule selected on the training split (a zone counts as involved when the mask covers at least 10% of it) reproduces the full zone set on 51.5% [47.3, 55.6] of the 2113 test images, against 32.0% for the most frequent set (intervals resample test subjects). Its mean Jaccard overlap with the caption is 0.77, and the per-zone Cohen’s � ranges from 0.56 to 0.91: the upper zones agree almost perfectly, and where the two disagree, the rule mostly calls a middle or lower zone involved that the caption does not name. A gradient-boosting classifier on 106 mask-shape features reaches 65.1% [61.0, 69.1]. The side (92.7%) and the stated number of regions (98.4%, against connected components) can almost always be derived from the mask; the zone set cannot. LViT’s own captions, which RecLMIS reads, parse to the same zones as MedCLIPSeg’s on all 2113 test images.

Interpretation. In an analysis not planned in advance, the number of named zones correlates with mask area with a Spearman rank correlation of $r _ { s } ~ = 0 . 8 9$ on the 2113 test images. The zone phrase is therefore a human annotation of extent that carries about as much extent information as the computed size word $( R _ { \mathrm { c a p } } ^ { 2 } 0 . 7 7$ against 0.70 on Kvasir-SEG; Table 2). Our analysis plan reserved leakage by construction for captions computed from the mask, and the paper keeps that distinction: only the colonoscopy and BUSI size words are functions of the mask. We still use extent leakage for QaTa-COV19, because its zone phrase, written for the benchmark and checked against the masks, gives the model at test time extent information that does not exist at deployment.

Table D.6: Per-seed results of the retrained models.
<table><tr><td rowspan="2">Model</td><td colspan="3">Dice, empty caption</td><td colspan="3">Dice, matched caption</td><td colspan="3">Text gain G</td></tr><tr><td>seed 1</td><td>seed 2</td><td>seed 3</td><td>seed 1</td><td>seed 2</td><td>seed 3</td><td>seed 1</td><td>seed 2</td><td>seed 3</td></tr><tr><td colspan="10">Kvasir-SEG, five colonoscopy test sets (n = 681)</td></tr><tr><td>Original</td><td>69.54</td><td>69.20</td><td>70.74</td><td>77.07</td><td>75.04</td><td>75.90</td><td>7.53</td><td>5.84</td><td>5.15</td></tr><tr><td>Size-blind</td><td>72.18</td><td>71.79</td><td>72.67</td><td>74.54</td><td>74.16</td><td>74.88</td><td>2.36</td><td>2.37</td><td>2.21</td></tr><tr><td>Size-only</td><td>69.79</td><td>68.94</td><td>70.49</td><td>75.88</td><td>76.41</td><td>73.54</td><td>6.09</td><td>7.47</td><td>3.06</td></tr><tr><td>Mismatched</td><td>72.86</td><td>73.13</td><td>72.90</td><td>72.42</td><td>72.83</td><td>72.06</td><td>-0.44</td><td>-0.30</td><td>-0.84</td></tr><tr><td>Text-free</td><td>73.36</td><td>72.27</td><td>72.33</td><td>73.32</td><td>72.28</td><td>72.21</td><td>-0.03</td><td>0.01</td><td>-0.12</td></tr><tr><td colspan="10">BUSI, 65 test images with a lesion</td></tr><tr><td>Original</td><td>74.20</td><td>67.75</td><td>59.71</td><td>82.17</td><td>82.83</td><td>81.94</td><td>7.97</td><td>15.08</td><td>22.23</td></tr><tr><td>Size-blind</td><td>76.05</td><td>74.92</td><td>75.97</td><td>80.78</td><td>82.09</td><td>81.76</td><td>4.73</td><td>7.16</td><td>5.80</td></tr><tr><td>Size-only</td><td>77.67</td><td>78.05</td><td>66.13</td><td>80.37</td><td>82.38</td><td>81.69</td><td>2.70</td><td>4.33</td><td>15.56</td></tr><tr><td>Mismatched</td><td>81.71</td><td>80.57</td><td>80.09</td><td>81.39</td><td>80.73</td><td>79.26</td><td>-0.31</td><td>0.16</td><td>-0.83</td></tr><tr><td>Text-free</td><td>81.04</td><td>80.43</td><td>80.48</td><td>81.04</td><td>80.36</td><td>80.56</td><td>0.00</td><td>-0.06</td><td>0.08</td></tr><tr><td colspan="10">QaTa-COV19 (n = 2113)</td></tr><tr><td>Original</td><td>66.42</td><td>72.61</td><td>64.63</td><td>83.78</td><td>83.46</td><td>83.73</td><td>17.36</td><td>10.85</td><td>19.10</td></tr><tr><td>Zone-blind</td><td>79.06</td><td>79.06</td><td>79.38</td><td>80.50</td><td>80.62</td><td>80.65</td><td>1.43</td><td>1.55</td><td>1.27</td></tr><tr><td>Zone-only</td><td>69.82</td><td>63.80</td><td>68.89</td><td>80.98</td><td>82.45</td><td>82.15</td><td>11.16</td><td>18.65</td><td>13.26</td></tr><tr><td>Text-free</td><td>79.17</td><td>79.39</td><td>79.04</td><td>79.18</td><td>79.40</td><td>79.05</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td colspan="10">ExtentSeg and its controls, five colonoscopy test sets: : Dice under the conditioning named</td></tr><tr><td>ExtentSeg, own estimate</td><td>75.07</td><td>74.11</td><td>74.59</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ExtentSeg without area loss, own estimate</td><td>73.92</td><td>73.79</td><td>73.83</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Extent-free control</td><td>73.85</td><td>74.15</td><td>74.26</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ExtentSeg, true extent†</td><td>76.07</td><td>75.39</td><td>76.07</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Last-epoch checkpoints, each scored on the full test population. †Reads the test mask.

## References

Agrawal, A., Batra, D., Parikh, D., Kembhavi, A., 2018. Don’t just assume; look and answer: Overcoming priors for visual question answering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4971-4980. doi:10.1109/CVPR.2018.00522.

Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A., 2020. Dataset of breast ultrasound images. Data Brief 28, 104863. doi:10.1016/j.dib.2019.104863.

Ba, J.L., Kiros, J.R., Hinton, G.E., 2016. Layer normalization. arXiv preprint arXiv:1607.06450. doi:10.48550/arXiv. 1607.06450.

Bateson, M., Dolz, J., Kervadec, H., Lombaert, H., Ben Ayed, I., 2021. Constrained domain adaptation for image segmentation. IEEE Trans. Med. Imaging 40, 1875-1887. doi:10.1109/TMI.2021.3067688.

Bateson, M., Kervadec, H., Dolz, J., Lombaert, H., Ben Ayed, I., 2022. Source-free domain adaptation for image segmentation. Med. Image Anal. 82, 102617. doi:10.1016/j.media. 2022.102617.

Bernal, J., Sánchez, F.J., Fernández-Esparrach, G., Gil, D., Rodríguez, C., Vilariño, F., 2015. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Comput. Med. Imaging Graph. 43, 99-111. doi:10.1016/j.compmedimag.2015.02.007.

Bernal, J., Sánchez, J., Vilariño, F., 2012. Towards automatic polyp detection with a polyp appearance model. Pattern Recognit. 45, 3166-3182. doi:10.1016/j.patcog.2012. 03.002.

Bhardwaj, R., Tambe, U.Y., Neog, D.R., 2025. ViTexNet: Vision-text guided dynamic convolution network for medical image segmentation, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2025, Springer Nature Switzerland, Cham. pp. 690-699. doi:10.1007/ 978-3-032-04981-0\_65.

Brown II, E.L., Yang, J., Yang, S., Fergus, R., Xie, S., 2026. Benchmark designers should “train on the test set” to expose exploitable non-visual shortcuts. Conference on Language Modeling (COLM). URL: https://openreview. net/forum?id=ggtdDv1l9s. arXiv:2511.04655v2.

Cao, D.P.M., Pham, H.D., Pham, H., 2026. When oracle conditioning misleads deployment: Conditioningavailability bias in echocardiographic segmentation. arXiv preprint arXiv:2608.03342. doi:10.48550/arXiv.2608. 03342. Accepted at the MICCAI 2026 Workshop on Fairness of AI in Medical Imaging (FAIMI).

Chen, G., Pan, H., Tian, Y., Tian, Z., Wu, X., 2026. Textguided multi-stage cross-perception network for medical image segmentation. J. Imaging Inform. Med. doi:10.1007/ s10278-025-01811-0. Online ahead of print.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., Zhao, F., 2024a. Are we on the right way for evaluating large vision-language models?, in: Advances in Neural Information Processing Systems, pp. 27056-27087. doi:10.52202/079017-0850.

Chen, Y., Wei, M., Zheng, Z., Hu, J., Shi, Y., Xiong, S., Zhu, X.X., Mou, L., 2024b. CausalCLIPSeg: Unlocking CLIP’s potential in referring medical image segmentation with causal intervention, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2024, Springer. pp. 77-87. doi:10.1007/978-3-031-72384-1\_8.

Cirik, V., Morency, L.P., Berg-Kirkpatrick, T., 2018. Visual referring expression recognition: What do systems actually learn?, in: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 2 (Short Papers), pp. 781-787. doi:10.18653/v1/ N18-2123.

Cohen, J.P., Viviano, J.D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., Lungren, M.P., Chaudhari, A., Brooks, R., Hashir, M., Bertrand, H., 2022. TorchXRayVision: A library of chest X-ray datasets and models, in: Proceedings of the International Conference on Medical Imaging with Deep Learning (MIDL), pp. 231-249. URL: https: //proceedings.mlr.press/v172/cohen22a.html.

Degerli, A., Ahishali, M., Yamac, M., Kiranyaz, S., Chowdhury, M.E.H., Hameed, K., Hamid, T., Mazhar, R., Gabbouj, M., 2021. COVID-19 infection map generation and detection from chest X-ray images. Health Inf. Sci. Syst. 9, 15. doi:10. 1007/s13755-021-00146-8.

Degerli, A., Kiranyaz, S., Chowdhury, M.E.H., Gabbouj, M., 2022. OSegNet: Operational segmentation network for COVID-19 detection using chest X-ray images, in: Proceedings of the IEEE International Conference on Image Processing (ICIP), pp. 2306-2310. doi:10.1109/ICIP46576. 2022.9897412.

DeGrave, A.J., Janizek, J.D., Lee, S.I., 2021. AI for radiographic COVID-19 detection selects shortcuts over signal. Nat. Mach. Intell. 3, 610-619. doi:10.1038/s42256-021-00338-7.

Dhakal, M., Adhikari, R., Thapaliya, S., Khanal, B., 2024. VLSM-Adapter: Finetuning vision-language segmentation eficiently with lightweight blocks, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2024, Springer Nature Switzerland, Cham. pp. 712-722. doi:10. 1007/978-3-031-72114-4\_68.

Dice, L.R., 1945. Measures of the amount of ecologic association between species. Ecology 26, 297-302. doi:10.2307/ 1932409.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N., 2021. An

image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR). URL: https://openreview.net/forum? id=YicbFdNTTy.

Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., Zhao, H., 2023. On uni-modal feature learning in supervised multi-modal learning, in: Proceedings of the 40th International Conference on Machine Learning (ICML), pp. 8632- 8656. URL: https://proceedings.mlr.press/v202/ du23e.html.

Efron, B., Tibshirani, R.J., 1993. An Introduction to the Bootstrap. Monographs on Statistics and Applied Probability, Chapman & Hall, New York, NY, USA. doi:10.1007/ 978-1-4899-4541-9.

Feng, S., Wallace, E., Boyd-Graber, J., 2019. Misleading failures of partial-input baselines, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 5533-5538. doi:10.18653/ v1/P19-1554.

Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A., 2020. Shortcut learning in deep neural networks. Nat. Mach. Intell. 2, 665-673. doi:10. 1038/s42256-020-00257-z.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., Parikh, D., 2017. Making the V in VQA matter: Elevating the role of image understanding in visual question answering, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6325-6334. doi:10.1109/CVPR.2017.670.

Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., Poon, H., 2022. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans. Comput. Healthc. 3, 1-23. doi:10. 1145/3458754.

Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., Smith, N.A., 2018. Annotation artifacts in natural language inference data, in: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 2 (Short Papers), pp. 107-112. doi:10.18653/v1/N18-2017.

Hessel, J., Lee, L., 2020. Does my multimodal model learn cross-modal interactions? It’s harder to tell than you might think!, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 861- 877. doi:10.18653/v1/2020.emnlp-main.62.

Hoerl, A.E., Kennard, R.W., 1970. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics 12, 55-67. doi:10.1080/00401706.1970.10488634.

Huang, X., 2025. RecLMIS: Code repository. GitHub repository ShawnHuang497/RecLMIS, commit d3265c8. URL: https: //github.com/ShawnHuang497/RecLMIS. [software] (accessed 21 October 2026).

Huang, X., Li, H., Cao, M., Chen, L., You, C., An, D., 2025. Cross-modal conditioned reconstruction for language-guided medical image segmentation. IEEE Trans. Med. Imaging 44, 1821-1835. doi:10.1109/TMI.2024.3523333.

Huang, Y., Lin, J., Zhou, C., Yang, H., Huang, L., 2022. Modality competition: What makes joint training of multi-modal network fail in deep learning? (Provably), in: Proceedings of the 39th International Conference on Machine Learning (ICML), pp. 9226-9259. URL: https://proceedings. mlr.press/v162/huang22e.html.

Jha, D., Smedsrud, P.H., Riegler, M.A., Halvorsen, P., de Lange, T., Johansen, D., Johansen, H.D., 2020. Kvasir-SEG: A segmented polyp dataset, in: MultiMedia Modeling - MMM 2020, Springer International Publishing, Cham. pp. 451-462. doi:10.1007/978-3-030-37734-2\_37.

Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4, 100804. doi:10.1016/j.patter.2023.100804.

Kaufman, S., Rosset, S., Perlich, C., Stitelman, O., 2012. Leakage in data mining: Formulation, detection, and avoidance. ACM Trans. Knowl. Discov. Data 6, 1-21. doi:10.1145/ 2382577.2382579.

Kervadec, H., Dolz, J., Granger, É., Ben Ayed, I., 2019a. Curriculum semi-supervised segmentation, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2019, Springer International Publishing, Cham. pp. 568-576. doi:10.1007/978-3-030-32245-8\_63.

Kervadec, H., Dolz, J., Tang, M., Granger, É., Boykov, Y., Ben Ayed, I., 2019b. Constrained-CNN losses for weakly supervised segmentation. Med. Image Anal. 54, 88-99. doi:10.1016/j.media.2019.02.009.

Khattak, M.U., Kunhimon, S., Naseer, M., Khan, S., Khan, F.S., 2024. UniMed-CLIP: Towards a unified image-text pretraining paradigm for diverse medical imaging modalities. arXiv preprint arXiv:2412.10372. doi:10.48550/arXiv. 2412.10372.

Kingma, D.P., Ba, J., 2015. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR). URL: https://arxiv.org/abs/1412. 6980.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R., 2023. Segment anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3992-4003. doi:10.1109/ICCV51070. 2023.00371.

Koleilat, T., 2026a. MedCLIPSeg datasets: Images, masks and text prompts. Hugging Face dataset repository TahaKoleilat/MedCLIPSeg, revision b52c8f5. URL: https://huggingface.co/datasets/ TahaKoleilat/MedCLIPSeg. [dataset] (accessed 21 October 2026).

Koleilat, T., 2026b. MedCLIPSeg model checkpoints. Hugging Face model repository TahaKoleilat/MedCLIPSeg, revision 69bfb8f. URL: https://huggingface.co/ TahaKoleilat/MedCLIPSeg. [software] (accessed 21 October 2026).

Koleilat, T., 2026c. MedCLIPSeg: Oficial code repository. GitHub repository HealthX-Lab/MedCLIPSeg, commit 31d2220. URL: https://github.com/HealthX-Lab/ MedCLIPSeg. [software] (accessed 21 October 2026).

Koleilat, T., Asgariandehkordi, H., Nejatimanzari, O., Barile, B., Xiao, Y., Rivaz, H., 2026. MedCLIPSeg: Probabilistic vision-language adaptation for data-eficient and generalizable medical image segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1406- 1417. URL: https://openaccess.thecvf.com/ content/CVPR2026/html/Koleilat\_MedCLIPSeg\_ Probabilistic\_Vision-Language\_Adaptation\_ for\_Data-Efficient\_and\_Generalizable\_Medical\_ Image\_CVPR\_2026\_paper.html. arXiv:2602.20423.

Koleilat, T., Asgariandehkordi, H., Rivaz, H., Xiao, Y., 2025. MedCLIP-SAMv2: Towards universal text-driven medical image segmentation. Med. Image Anal. 106, 103749. doi:10. 1016/j.media.2025.103749.

Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., Müller, K.R., 2019. Unmasking Clever Hans predictors and assessing what machines really learn. Nat. Commun. 10, 1096. doi:10.1038/s41467-019-08987-4.

Lemay, A., Gros, C., Vincent, O., Liu, Y., Cohen, J.P., Cohen-Adad, J., 2021. Benefits of linear conditioning for segmentation using metadata, in: Proceedings of the Fourth Conference on Medical Imaging with Deep Learning (MIDL), pp. 416-430. URL: https://proceedings.mlr.press/ v143/lemay21a.html.

Li, X., Juvekar, A., Zhang, J., Liu, X., Wahed, M., Nguyen, K.A., Shen, Y., Yu, T., Lourentzou, I., 2026. Counterfactual segmentation reasoning: Diagnosing and mitigating pixel-grounding hallucination, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings Track (CVPRF), pp. 7453-7463. doi:10.1109/CVPRF71909.2026.00715.

Li, Z., 2026. LViT: Code repository and QaTa-COV19 and MosMedData+ text annotations. GitHub repository HUANGLIZI/LViT, commit ba90775. URL: https:// github.com/HUANGLIZI/LViT. [dataset] (accessed 21 October 2026).

Li, Z., Li, Y., Li, Q., Wang, P., Guo, D., Lu, L., Jin, D., Zhang, Y., Hong, Q., 2024. LViT: Language meets vision transformer in medical image segmentation. IEEE Trans. Med. Imaging 43, 96-107. doi:10.1109/TMI.2023.3291719.

Lian, J., Liu, J., Zhang, S., Gao, K., Liu, X., Zhang, D., Yu, Y., 2021. A structure-aware relation network for thoracic diseases detection and segmentation. IEEE Trans. Med. Imaging 40, 2042-2052. doi:10.1109/TMI.2021.3070847.

Liang, S., Wang, Z., Li, Y., Liu, X., Wang, H., 2026. From pixels to concepts: Do segmentation models understand what they segment? arXiv preprint arXiv:2605.09591. doi:10.48550/ arXiv.2605.09591.

Lin, G., Zhao, H., Xiong, Y., Zhang, L., Du, B., Zhu, W., 2026. TGC-Net: A structure-aware and semanticallyaligned framework for text-guided medical image segmentation. Inf. Fusion 135, 104438. doi:10.1016/j.inffus. 2026.104438.

Liu, Z., Zhu, Z., Shi, X., 2026. Characterizing text branch sensitivity in medical vision-language segmentation via evidence decoupling. arXiv preprint arXiv:2609.02663. doi:10. 48550/arXiv.2609.02663.

Loshchilov, I., Hutter, F., 2017. SGDR: Stochastic gradient descent with warm restarts. International Conference on Learning Representations (ICLR). URL: https://arxiv.org/ abs/1608.03983.

Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B., 2024. Segment anything in medical images. Nat. Commun. 15, 654. doi:10. 1038/s41467-024-44824-z.

Ma, M., Ren, J., Zhao, L., Testuggine, D., Peng, X., 2022. Are multimodal transformers robust to missing modality?, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18156-18165. doi:10.1109/CVPR52688.2022.01764.

Maier-Hein, L., Eisenmann, M., Reinke, A., Onogur, S., Stankovic, M., Scholz, P., Arbel, T., Bogunovic, H., Bradley, A.P., Carass, A., et al., 2018. Why rankings of biomedical image analysis competitions should be interpreted with care. Nat. Commun. 9, 5217. doi:10.1038/ s41467-018-07619-7.

Maier-Hein, L., Reinke, A., Godau, P., Tizabi, M.D., Buettner, F., Christodoulou, E., Glocker, B., Isensee, F., Kleesiek, J., Kozubek, M., et al., 2024. Metrics reloaded: Recommendations for image analysis validation. Nat. Methods 21, 195-212. doi:10.1038/s41592-023-02151-z.

Mazurowski, M.A., Dong, H., Gu, H., Yang, J., Konz, N., Zhang, Y., 2023. Segment anything model for medical image analysis: An experimental study. Med. Image Anal. 89, 102918. doi:10.1016/j.media.2023.102918.

Miller, G.A., 1955. Note on the bias of information estimates, in: Quastler, H. (Ed.), Information Theory in Psychology: Problems and Methods. Free Press, Glencoe, IL, USA, pp. 95-100.

Morozov, S.P., Andreychenko, A.E., Blokhin, I.A., Gelezhe, P.B., Gonchar, A.P., Nikolaev, A.E., Pavlov, N.A., Chernina, V.Y., Gombolevskiy, V.A., 2020. MosMedData: Data set of 1110 chest CT scans performed during the COVID-19 epidemic. Digit. Diagn. 1, 49-59. doi:10.17816/dd46826.

Neverova, N., Wolf, C., Taylor, G., Nebout, F., 2016. Mod-Drop: Adaptive multi-modal gesture recognition. IEEE Trans. Pattern Anal. Mach. Intell. 38, 1692-1706. doi:10. 1109/TPAMI.2015.2461544.

Ngoc Lan, P., An, N.S., Hang, D.V., Long, D.V., Trung, T.Q., Thuy, N.T., Sang, D.V., 2021. NeoUNet: Towards accurate colon polyp segmentation and neoplasm detection, in: Advances in Visual Computing - ISVC 2021, Springer International Publishing, Cham. pp. 15-28. doi:10.1007/ 978-3-030-90436-4\_2.

Nikolov, S., Blackwell, S., Zverovitch, A., Mendes, R., Livne, M., De Fauw, J., Patel, Y., Meyer, C., Askham, H., Romera-Paredes, B., et al., 2021. Clinically applicable segmentation of head and neck anatomy for radiotherapy: Deep learning algorithm development and validation study. J. Med. Internet Res. 23, e26151. doi:10.2196/26151.

Ong Ly, C., Unnikrishnan, B., Tadic, T., Patel, T., Duhamel, J., Kandel, S., Moayedi, Y., Brudno, M., Hope, A., Ross, H., McIntosh, C., 2024. Shortcut learning in medical AI hinders generalization: Method for estimating AI model generalization without external data. npj Digit. Med. 7, 124. doi:10.1038/s41746-024-01118-4.

Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE Trans. Syst. Man Cybern. 9, 62-66. doi:10. 1109/TSMC.1979.4310076.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S., 2019. PyTorch: An imperative style, high-performance deep learning library, in: Advances in Neural Information Processing Systems, pp. 8026-8037.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikitlearn: Machine learning in Python. J. Mach. Learn. Res. 12, 2825-2830. URL: https://jmlr.org/papers/v12/ pedregosa11a.html.

Peng, X., Wei, Y., Deng, A., Wang, D., Hu, D., 2022. Balanced multimodal learning via on-the-fly gradient modulation, in: Proceedings of the IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition (CVPR), pp. 8228- 8237. doi:10.1109/CVPR52688.2022.00806.

Pfungst, O., 1911. Clever Hans (The Horse of Mr. von Osten): A Contribution to Experimental Animal and Human Psychology. Henry Holt and Company, New York, NY, USA. doi:10.5962/bhl.title.56164. Translated from the German by C. L. Rahn.

Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., Van Durme, B., 2018. Hypothesis only baselines in natural language inference, in: Proceedings of the Seventh Joint Conference on Lexical and Computational Semantics (\*SEM), pp. 180-191. doi:10.18653/v1/S18-2023.

Poudel, K., 2024. MedVLSM: Exploring transfer learning in medical image segmentation using vision-language models, oficial code and prompt files. GitHub repository naamiinepal/medvlsm, commit dc36a3a. URL: https:// github.com/naamiinepal/medvlsm. [software and data] (accessed 21 October 2026).

Poudel, K., Dhakal, M., Bhandari, P., Adhikari, R., Thapaliya, S., Khanal, B., 2024. Exploring transfer learning in medical image segmentation using vision-language models, in: Proceedings of the International Conference on Medical Imaging with Deep Learning (MIDL), pp. 1142-1165. URL: https: //proceedings.mlr.press/v250/poudel24a.html.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I., 2021. Learning transferable visual models from natural language supervision, in: Proceedings of the 38th International Conference on Machine Learning (ICML), pp. 8748-8763. URL: https://proceedings. mlr.press/v139/radford21a.html.

Rahman, M.M., Hammond, T., 2026. Dual-domain cross-modal decoding for clinical text-guided medical image segmentation, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2026, Springer Nature Switzerland, Cham. pp. 135-145. doi:10.1007/978-3-032-38082-1\_ 13.

Reinke, A., Tizabi, M.D., Baumgartner, M., Eisenmann, M., Heckmann-Nötzel, D., Kavur, A.E., Rädsch, T., Sudre, C.H., Acion, L., Antonelli, M., et al., 2024. Understanding metricrelated pitfalls in image analysis validation. Nat. Methods 21, 182-194. doi:10.1038/s41592-023-02150-0.

Shan, D., Li, Z., Li, Y., Li, Q., Tian, J., Hong, Q., 2025. STPNet: Scale-aware text prompt network for medical image segmentation. IEEE Trans. Image Process. 34, 3169-3180. doi:10.1109/TIP.2025.3571672.

Srikanth, N., Rudinger, R., 2022. Partial-input baselines show that NLI models can ignore context, but they don’t, in: Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pp. 4753- 4763. doi:10.18653/v1/2022.naacl-main.350.

Tajbakhsh, N., Gurudu, S.R., Liang, J., 2016. Automated polyp detection in colonoscopy videos using shape and context information. IEEE Trans. Med. Imaging 35, 630-644. doi:10.1109/TMI.2015.2487997.

Tomar, N.K., Jha, D., Bagci, U., Ali, S., 2022. TGANet: Textguided attention for improved polyp segmentation, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2022, Springer Nature Switzerland, Cham. pp. 151-160. doi:10.1007/978-3-031-16437-8\_15.

Tomar, N.K., Jha, D., Biswas, K., Bagci, U., 2025. Transformerenhanced iterative feedback mechanism for polyp segmentation, in: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1-5. doi:10.1109/ICASSP49660.2025.10890567.

Valcarcel, A.M., Muschelli, J., Pham, D.L., Martin, M.L., Yushkevich, P., Brandstadter, R., Patterson, K.R., Schindler, M.K., Calabresi, P.A., Bakshi, R., Shinohara, R.T., 2020. TAPAS: A thresholding approach for probability map automatic segmentation in multiple sclerosis. Neuroimage Clin. 27, 102256. doi:10.1016/j.nicl.2020.102256.

Vapnik, V., Izmailov, R., 2015. Learning using privileged information: Similarity control and knowledge transfer. J. Mach. Learn. Res. 16, 2023-2049. URL: https://jmlr. org/papers/v16/vapnik15b.html.

Vásquez-Venegas, C., Wu, C., Sundar, S., Prôa, R., Beloy, F.J., Medina, J.R., McNichol, M., Parvataneni, K., Kurtzman, N., Mirshawka, F., Aguirre-Jerez, M., Ebner, D.K., Celi, L.A., 2025. Detecting and mitigating the Clever Hans efect in medical imaging: A scoping review. J. Imaging Inform. Med. 38, 2563-2579. doi:10.1007/s10278-024-01335-z.

Vázquez, D., Bernal, J., Sánchez, F.J., Fernández-Esparrach, G., López, A.M., Romero, A., Drozdzal, M., Courville, A., 2017. A benchmark for endoluminal scene segmentation of colonoscopy images. J. Healthc. Eng. 2017, 4037190. doi:10.1155/2017/4037190.

Wang, B., Zhang, Y., Yu, J., Ong, C.S., Huang, J., Li, Z., Zhang, Z., Cohan, A., Yu, H., Yao, Z., 2026. MedQA-MM: Shortcuts behind medical visual reasoning, in: Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Budapest, Hungary. pp. 10708-10758. URL: https://aclanthology.org/2026.emnlp-main.467/.

Wang, Q., Lin, X., Yan, Z., 2025. Towards robust medical image referring segmentation with incomplete textual prompts, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2025, Springer Nature Switzerland, Cham. pp. 636-646. doi:10.1007/978-3-032-04981-0\_60.

Wang, W., Tran, D., Feiszli, M., 2020. What makes training multi-modal classification networks hard?, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12692-12702. doi:10.1109/CVPR42600.2020.01271.

Wu, N., Jastrzebski, S., Cho, K., Geras, K.J., 2022. Characterizing and overcoming the greedy nature of learning in multimodal deep neural networks, in: Proceedings of the 39th International Conference on Machine Learning (ICML), pp. 24043-24055. URL: https://proceedings.mlr.press/ v162/wu22d.html.

Xi, S., Hu, S., Wang, S., Ding, H., Li, Y., He, W., Zhang, K., Del Balzo, L., Zhong, C., Hu, M., Liu, C.C., Liu, Y., Yang, X., 2026. Grounding radiology report findings into medical image segmentation. npj Digit. Med. doi:10.1038/ s41746-026-03051-0.

Ye, S., Meng, M., Li, M., Feng, D., Kim, J., 2024. Enabling text-free inference in language-guided segmentation of chest X-rays via self-guidance, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2024, Springer Nature Switzerland, Cham. pp. 242-252. doi:10. 1007/978-3-031-72111-3\_23.

Ye, S., Naseem, U., Meng, M., Kim, J., 2025. Alleviating textual reliance in medical language-guided segmentation via prototype-driven semantic approximation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22316-22326. doi:10.1109/ICCV51701. 2025.02072.

Yu, B., Yang, J., Du, Z., Huang, Y., Li, C., Wang, L., 2025. Frequency-domain multi-modal fusion for languageguided medical image segmentation, in: Medical Image Computing and Computer Assisted Intervention - MICCAI 2025, Springer Nature Switzerland, Cham. pp. 278-288. doi:10.1007/978-3-032-05114-1\_27.

Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K., 2018. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLoS Med. 15, e1002683. doi:10.1371/journal.pmed.1002683.

Zhao, P., Hou, Y., Yan, Z., Huo, S., 2025a. Text-driven medical image segmentation with text-free inference via knowledge distillation. IEEE Trans. Instrum. Meas. 74, 1-15. doi:10. 1109/TIM.2025.3545506.

Zhao, T., Gu, Y., Yang, J., Usuyama, N., Lee, H.H., Kiblawi, S., Naumann, T., Gao, J., Crabtree, A., Abel, J., et al., 2025b. A foundation model for joint segmentation, detection and recognition of biomedical objects across nine modalities. Nat. Methods 22, 166-176. doi:10.1038/ s41592-024-02499-w.

Zhao, W., Li, F., Diao, Y., Fan, P., Chen, Z., 2024. Cap2Seg: Leveraging caption generation for enhanced segmentation of COVID-19 medical images. Front. Phys. 12, 1439122. doi:10.3389/fphy.2024.1439122.

Zhao, Z., Zhang, Y., Wu, C., Zhang, X., Zhou, X., Zhang, Y., Wang, Y., Xie, W., 2025c. Large-vocabulary segmentation for medical images with text prompts. npj Digit. Med. 8, 566. doi:10.1038/s41746-025-01964-w.

Zhong, Y., Xu, M., Liang, K., Chen, K., Wu, M., 2023. Ariadne’s thread: Using text prompts to improve segmentation of infected areas from chest X-ray images, in: Medical Image Computing and Computer Assisted Intervention - MIC-CAI 2023, Springer Nature Switzerland, Cham. pp. 724-733. doi:10.1007/978-3-031-43901-8\_69.