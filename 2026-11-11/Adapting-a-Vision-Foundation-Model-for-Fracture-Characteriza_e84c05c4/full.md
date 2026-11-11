# Adapting a Vision Foundation Model for Fracture Characterization in Acoustic Image Logs

Guangyu Wang<sup>a,b</sup>, Xiaodong Ma<sup>a,b,∗</sup> and Xinming Wu<sup>a,b,∗</sup>

<sup>a</sup>State Key Laboratory of Precision Geodesy, School of Earth and Space Sciences, University of Science and Technology of China, Hefei, 230026, China

<sup>b</sup>Mengcheng National Geophysical Observatory, University of Science and Technology of China, Mengcheng, 233500, China

## A R T I C L E I N F O

Keywords:6   
Acoustic image log2 Natural fracture   
<sup>Drilling-induced</sup> <sup>tensile</sup> <sup>fracture</sup>2 Vision foundation model DINOv3 Transfer learningo

## A BS T R AC T

Natural fractures (NFs) and drilling-induced tensile fractures (DITFs) observed in acoustic image logs provide important information about rock mass structure and in situ stress conditions. However, characterizing their locations and orientations requires time-consuming expert interpretation. Deep learning ofers a promising way to automate this process, but its application is constrained by the scarcity of expert-annotated fracture data. Pretrained vision foundation models (VFMs) may help address this limitation by providing transferable visual representations that can be adapted with relatively few task-specific annotations. In this study, we investigate whether adapting a pretrained VFM improves fracture characterization compared with training task-specific models from scratch on the same limited labeled dataset. Using only 80 annotated acoustic image samples, we adapted a DINOv3 model pretrained on satellite imagery for fracture characterization and evaluated it using data from four boreholes independent of the fine-tuning dataset. The fine-tuned DINOv3 model characterized NF and DITF parameters at a level comparable to human interpretation. Compared with U-Net, YOLOv8, and Swin Transformer models trained from scratch, the fine-tuned DINOv3 produced fewer false positives and false negatives and more accurate estimates of fracture counts and orientations. However, experiments with progressively fewer training samples indicate that DINOv3’s advantage over the task-specific models generally diminishes as the number of training/fine-tuning samples decreases. We also examined how decoder design, fine-tuning strategy, and image enhancement afect fracture characterization performance. Among the configurations examined, a progressive upsampling decoder combined with full fine-tuning yielded the best DINOv3 performance, whereas image enhancement intended to emphasize fracture appearance did not necessarily improve fracture characterization accuracy. These findings support VFM adaptation as an efective strategy for characterizing fractures in acoustic image logs when labeled data are limited. The proposed workflow may also provide a useful reference for applying VFMs to other geoscientific image analysis tasks with limited annotations.

## 1. Introduction

Understanding subsurface geological features is a fundamental objective in georesource development and geoscientific investigation. Among the features accessible at the borehole scale, fractures are of particular importance. Natural fractures (NFs) record the deformation history of the rock mass (Wenning et al., 2017: Massiot et al., 2017a) and impose a major influence on its mechanical and hydraulic properties (Massiot et al., 2017b; Afshari Moein et al., 2018). Drilling-induced tensile fractures (DITFs) develop in response to stress concentration around the borehole wall and provide constraints on stress orientation (Brudy and Zoback, 1999; Zoback et al., 2003). Accurate characterization of fracture locations and orientations is therefore essential for hydrocarbon reservoir evaluation (Milad et al., 2018; Lai et al., 2021), enhanced geothermal system development (Valley and Evans, 2009; Chen et al., 2026), and in situ stress assessment (Nian et al., 2016; Shahri et al., 2016).

Acoustic image logs provide direct visualization of fractures on borehole walls. An acoustic image log is acquired through emitting focused ultrasonic pulses toward the borehole wall and recording the amplitude and travel time of the reflected signal. In this study, acoustic image log only refers to its amplitude component. As shown in Figure 1, NFs typically appear as sinusoidal traces whose geometry encodes the fracture orientation.

![](images/a26bb418e3442e93b05b42bfbeb2a5a646f5cf068bb4d69843554feb45aff57d.jpg)  
Figure 1: Workflow for fracture characterization through DINOv3 adaptation.NF denotes natural fracture and DITF represent drilling-induced tensile fracture. The number beside each fracture trace represent the confidence score, ranging from 0 to 1. Fracture traces with confidence score less than 0.5 are not visualized.

DITFs commonly appear as axial or en echelon traces, often occurring in pairs approximately $1 8 0 ^ { \circ }$ apart in azimuth. The azimuth of DITFs reflects the maximum horizontal stress $( S _ { \mathrm { H m a x } } )$ direction. Conventional fracture characterization involves identifying these fracture traces and fitting their geometry manually. The repetitive nature of this work and the considerable time it requires have driven the development of automated methods.

Early attempts at automated fracture characterization from borehole images employs classical image processing techniques, including edge detection operators (e.g., Hall et al., 1996; Wang, 2005), Hough and Radon transforms for sinusoidal curve detection (e.g., Glossop et al., 1999; Kherroubi, 2008), mathematical morphology (Xavier et al., 2015), and phase congruency methods (Assous et al., 2014). While these approaches ofered improved eficiency over manual picking, they generally struggled with the highly variable image quality, complex geological backgrounds, and diverse morphological expressions of fractures encountered in real-world logging data (Han et al., 2023).

The development of deep learning has brought significant advances in automated fracture characterization. Convolutional neural networks (CNNs), particularly the U-Net architecture and its variants, have been widely adopted for pixel-level fracture segmentation from borehole image logs (e.g., Cruz et al., 2017; Li et al., 2025). Object detection frameworks such as Faster R-CNN, Mask R-CNN, and YOLOv8 have also been applied to fracture localization and instance segmentation (e.g., Dias et al., 2020; Liu et al., 2022; Han et al., 2025). More recently, transformer-based architectures have been explored for fracture segmentation (Wang and Zhou, 2023) and direct estimation of fracture locations and orientations (Nasim et al., 2025), leveraging their capacity to model long-range dependencies in image features.

Despite these advances, training deep learning models for image log interpretation remains challenging. Annotating fractures takes substantial time, so relatively few labeled acoustic images are available for training. Models trained from scratch on such small datasets may struggle to recognize fractures in new boreholes. Previous studies have addressed this problem through unsupervised domain adaptation, transferring information from labeled road crack images to unlabeled borehole images (Zhang et al., 2021, 2022). These studies demonstrate the potential value of knowledge learned from other image types. We therefore investigate whether a pretrained vision foundation model can be adapted with a small number of labeled acoustic images to improve fracture characterization.

Vision foundation models (VFMs) ofer a fundamentally diferent paradigm for addressing the label scarcity challenge. These large-scale neural networks, pretrained on massive image collections through self-supervised learning, develop rich and generalizable visual feature representations that can be efectively transferred to diverse downstream tasks with minimal task-specific adaptation (Bommasani et al., 2021). The DINO family of models (Caron et al., 2021; Oquab et al., 2024; Siméoni et al., 2025) exemplifies this approach. By learning visual representations through discriminative self-supervised objectives on datasets comprising hundreds of millions to billions of natural images, these models acquire hierarchical feature representations that capture both global semantic context and local details without any human annotation. These representations have proven transferable to downstream tasks across diverse specialized domains. In medical image segmentation, fine-tuned DINOv2 consistently outperforms fully supervised U-Net baselines under limited annotation conditions, with the performance gap widening as training data decrease (Kundu et al., 2025). In industrial anomaly detection, DINOv2 achieves state-of-the-art one-shot defect localization on standard benchmarks without domain-specific fine-tuning (Damm et al., 2025). In geophysical image analysis, DINOv2 fine-tuned with parameter-eficient methods outperforms task-specific U-Net models on seismic facies classification, fault detection, and crater identification tasks, even with as few as 50 training samples (Guo et al., 2025). Together, these results indicate that pretrained visual representations generalize efectively across imaging domains far removed from the natural image distribution on which they were trained.

In this study, we adapt a DINOv3 encoder pretrained on satellite imagery using a small set of labeled acoustic images (Figure 1). The adapted model segments the images into background, NF, and DITF classes. Geometric processing then converts the predicted masks into fracture depths, orientations, and morphology-based confidence scores. We compare this approach with U-Net, YOLOv8, and Swin Transformer models trained from scratch on the same labeled samples, evaluating both image segmentation and fracture parameters against manual interpretations. We also examine how the number of training samples, decoder design, fine-tuning strategy, and image enhancement afect performance. By comparing the approaches quantitatively with limited labeled data, this study assesses whether fine-tuning a VFM yields better fracture characterization than training task-specific models from scratch. The findings also inform the choice of decoder and fine-tuning strategy and may guide similar studies beyond the task of image log fracture characterization.

## 2. Method

## 2.1. Network architecture and fine-tuning strategy

Our workflow first identifies fracture pixels in an acoustic image and then calculates fracture parameters represented by those pixels. For the first step, we combine a pretrained DINOv3 image encoder, which extracts visual patterns, with a decoder that produces a pixel-level map (Figure 2). The encoder is a vision transformer initialized from SAT-493M satellite imagery (Siméoni et al., 2025). The decoder assigns each pixel in a 256 × 256 image to background, NF, or DITF class.

Our main configuration uses a progressive upsampling (PUP) decoder, which gradually expands the encoder’s image features into a full-resolution fracture map (Zheng et al., 2021). We update both the pretrained encoder and the decoder using labeled acoustic images. This fine-tuning strategy is referred to as full fine-tuning or Unfrozen in the comparisons below. Additional comparisons of decoder designs and fine-tuning strategies are presented in Section 4.3.

The acoustic amplitude values are standardized, and the single image channel is copied three times to match the encoder’s input format. During training, we apply augmentation techniques to the input image logs, including to reverse images along depth, reflect them along azimuth, and shift them around the azimuthal image boundary. Each transformation is also applied to its label. These transformations create varied training views but do not add new, independently labeled samples. The training combines cross-entropy and Dice losses to compare predicted fracture regions with the labels. It also uses Same Class Neighbor Penalization (SCNP) to discourage breaks in narrow fracture masks (Valverde et al., 2026).The network is trained for 500 epochs using AdamW optimization method. When processing a continuous log, we divide it into overlapping images along depth and combine predictions in the shared regions. A 20% overlap is applied to help avoid artificial breaks at image boundaries.

We compare DINOv3 with three models trained from scratch. These are U-Net (Ronneberger et al., 2015), a YOLOv8-based segmentation model (Jocher et al., 2023), and a Swin Transformer segmentation model (Liu et al., 2021). The YOLOv8 model predicts fracture pixels rather than bounding boxes. All models use the same labeled samples, training duration, batch size, and SCNP setting.

![](images/9afeb0db781b1ed9f61e84963cf54d2e9d5c9ec1b110fecf71f96efa285042f2.jpg)  
Figure 2: Segmentation architecture and adaptation strategy. The pretrained DINOv3 encoder supplies spatial features to a progressive upsampling (PUP) decoder, which produces pixel-wise background, NF, and DITF predictions. Pretrained weights of DINOv3 are used to initialize the encoder, and both encoder and decoder are updated in during fine-tuning. The acoustic image log is replicated across the three input channels.

## 2.2. Extracting fracture depths and orientations

The models described above identify NF and DITF pixels, but fracture characterization requires measurements of individual traces. We therefore process the two fracture types separately. In an image log, the vertical direction represents depth and the horizontal direction represents azimuth, or position around the borehole wall. We group connected fracture pixels into traces, extract a centerline for each trace, and convert its image positions to measured depths and azimuths.

On an image log, an NF typically appears as a sinusoidal trace. We fit a sinusoid to its centerline. The fitted curve’s vertical midpoint gives the fracture’s central depth. Its amplitude, together with the borehole diameter, determines the fracture dip. The azimuth of the sinusoid’s deepest point indicates the dip azimuth of the fracture.We also assign each NF a confidence score, giving 70% weight to how far the trace extends around the borehole and 30% to how closely it follows the fitted curve. Thus, a trace that covers more of the borehole circumference and fits the curve more closely receives a higher score. The confidence score reflects how closely a detected trace matches the expected fracture geometry. A higher score indicates a more credible fracture pick, whereas a lower score suggests that the trace may be noise.

In contrast, DITFs often appear as nearly vertical or en echelon traces. We fit a straight line to the overall trend of each DITF centerline and use it to estimate the trace’s central depth, azimuth, length, and tilt relative to the borehole axis. We also assign each DITF a confidence score. Because DITFs often extend along depth and occur in pairs on approximately opposite sides of the borehole, the score gives 30% weight to depth persistence and 70% to evidence of a counterpart at a similar depth. The pairing assessment considers both azimuthal separation from 180<sup>◦</sup> and depth overlap. Under this confidence scoring criterion, some short, unpaired false-positive noise features receive relatively low confidence scores. However, a real DITF may nevertheless receive a low score if its counterpart is faint or absent.

80 fine-tuning and 20 validation samples from 4 boreholes  
![](images/6b8d406e6a89020a512d1fe55070ffafa03194a8b3e5ae0d145b72373498996c.jpg)  
20 test samples from 6 boreholes

![](images/4e67bd166ba3c211344b8dd7593160e3b6753057031242bc1600d109ea29569d.jpg)

![](images/f0320319539be824527364ab2ca3c5190abc26bb548b398401b6be6dab18ea07.jpg)  
Figure 3: Acoustic image log dataset used for DINOv3 fine-tuning and evaluation. Representative images show the range of fracture expressions and backgrounds, with one example per borehole. Fine-tuning and validation images originate from four boreholes, and test images from six other boreholes. Histograms summarize the numbers of NF-only, DITF-only, and mixed images.

## 2.3. Dataset and evaluation design

The acoustic image logs used in this study were obtained from boreholes associated with geothermal projects, scientific drilling, and other subsurface investigations. They cover diferent rock types and imaging conditions and show a range of NF and DITF appearances. Each labeled sample is processed as a 256 × 256-pixel acoustic-amplitude image. NF and DITF regions are marked at the pixel level, with the remaining pixels assigned to background. Figure 3 shows representative images and their labels.

The dataset contains 120 labeled images. 80 images were used for training and 20 for validation, all from four boreholes. The remaining 20 images were reserved for testing and came from six other boreholes that were not used for training or validation. This separation allows us to assess segmentation performance on images from previously unseen boreholes. Figure 3 also shows how many images contain only NFs, only DITFs, or both. The validation and test sets include images containing only NFs, only DITFs, and combinations of both, providing a comprehensive evaluation of the model’s ability to segment diferent types of fractures.

We evaluate segmentation with intersection over union (IoU). For each image, we divide the number of pixels marked as fracture by both the model and the expert by the number marked as fracture by either one. We then report the mean across images. IoU measures how well the predicted fracture regions agree with the manual labels. However, good agreement at the pixel level does not necessarily mean that the resulting fracture depths and orientations are accurate. We therefore also evaluate the picks and extracted parameters for NFs and DITFs separately.

To assess whether the predicted masks lead to useful fracture measurements, we compare automated picks from continuous logs with manual interpretations. The following matching tolerances allow for small diferences in position or orientation while still requiring the picks to describe approximately the same fracture. For NFs, central depths must difer by no more than 0.1 m and orientations by no more than 15<sup>◦</sup>. For DITFs, azimuths must difer by less than 30<sup>◦</sup>, and the depth intervals must overlap by more than 80% of at least one trace’s length.

We use the false-negative rate (FNR) to measure the fraction of manual picks missed by the model, so a lower value indicates more complete detection. The false-positive rate (FPR) is used to reflect the fraction of automated picks with no manual match, so a lower value indicates fewer unmatched candidates. FNR follows its usual definition, but the reported FPR difers from the conventional false-positive rate because its denominator is the number of automated picks, not the number of negative cases. Manual picks serve as the reference for these comparisons, but it does not represent an error-free record.

We also compare summary properties in consecutive 5 m depth intervals. These metrics show whether the model reproduces the amount and prevailing orientation of fractures along a log, even when individual picks are imperfect. For NFs, we calculate mean absolute diferences in fracture count, mean dip azimuth, and mean dip. For DITFs, we compare total trace length, mean azimuth, and mean tilt in the same way. A trace is assigned to the interval containing its central depth. Azimuths near 0<sup>◦</sup> and 360<sup>◦</sup> are treated as close directions, while DITF directions 180<sup>◦</sup> apart are treated as equivalent. Geometric errors are calculated only where the required values are available. They must therefore be read alongside the missed- and unmatched-pick fractions.

To make fracture-picking accuracy easier to compare across methods, we combine detection and geometric accuracy in one summary score. Detection is represented by the fractions of manual picks recovered and automated picks matched. Geometry is represented by the three errors described above. Detection and geometry each contribute half of the score. To combine errors measured in diferent units, we compare each one with a fixed reference value. For each evaluated log � with complete measurements, the score is

$$
S _ { b } = 2 5 \big [ ( 1 - \mathrm { F N R } _ { b } ) + ( 1 - \mathrm { F P R } _ { b } ) \big ] + \frac { 5 0 } { 3 } \sum _ { j = 1 } ^ { 3 } \exp ( - e _ { b j } / m _ { b j } ) .\tag{1}
$$

Here $e _ { b j }$ is geometric error $j .$ Its reference value $m _ { b j }$ is the median for that error across all compared methods and remains fixed throughout the analyses. NF and DITF scores are averaged separately across the logs evaluated for each fracture type. Higher scores indicate better overall picking performance.

## 3. Results

In this section, we first compare how accurately the fine-tuned DINOv3 model and the models trained from scratch segment fractures in individual images. We then assess whether the predicted traces produce useful NF and DITF picks and fracture parameters in continuous logs from four boreholes not used for segmentation training or validation. Together, the image-level and log-level results show where improved segmentation translates into better fracture characterization and where errors remain.

## 3.1. Fracture segmentation

For fracture segmentation, the fine-tuned DINOv3 achieves the highest mean IoU on both the validation and borehole-separated test sets (Figure 4a for validation, Figure 4b for testing, and Table 1). On the test set, its mean IoU is 0.624, compared with 0.545 for U-Net, the strongest model trained from scratch. This is a relative improvement of 14.5%. DINOv3 also leads on the validation set, with a mean IoU of 0.655 versus 0.611 for U-Net.

All models perform less well on the test set than on validation, showing that segmentation remains sensitive to diferences between boreholes. The mean IoU decreases for DINOv3 are less than the decreases for the models trained from scratch. This pattern is consistent with better transfer to the test boreholes in this comparison, although 20 test images cannot represent all logging conditions. Selected examples in Figure 5 show how these numerical diferences appear in the predicted fracture traces.

In these examples, DINOv3 generally recovers more continuous annotated traces while producing fewer responses to background features than the comparison models. Errors remain around faint or intersecting traces and complex backgrounds. The examples illustrate the segmentation result but do not establish whether every predicted trace yields a correct fracture pick. We therefore examine the extracted NF and DITF parameters separately below.

![](images/4781d879426fb9d6a9b09678bf7508e2bb9188ea324fe166afe5dbc99573ef29.jpg)

![](images/4a2c445e39a1df984dbf83e901f1d8f865242483f3b7cb28a04dd2c2c750eca9.jpg)  
Figure 4: IoU distributions of NFs and DITFs for (a) validation and (b) test images. U-Net, YOLOv8, and Swin Transformer are trained from scratch, whereas DINOv3 is fully fine-tuned.

Mean IoU scores of NFs and DITFs for the validation and test dataset, respectively.
<table><tr><td>Model and learning strategy</td><td>Validation</td><td>Test</td></tr><tr><td>U-Net, from scratch</td><td>0.611</td><td>0.545</td></tr><tr><td>YOLOv8, from scratch</td><td>0.604</td><td>0.522</td></tr><tr><td>Swin Transformer, from scratch</td><td>0.597</td><td>0.502</td></tr><tr><td>DINOv3, full fine-tuning</td><td>0.655</td><td>0.624</td></tr></table>

## 3.2. Natural fracture characterization

We evaluate NF picking along two continuous logs, designated Boreholes A and B. When reviewing a selected interval, confidence scores help interpreters identify fitted traces that more closely follow the expected NF geometry. In Figure 6, darker curves represent higher confidence scores and paler curves represent lower scores, allowing interpreters to focus first on the more plausible candidates while checking lower-scoring picks against the image.

The selected intervals show how image quality afects the recovery of sinusoidal traces. In Borehole A (Figure 6a), the relatively smooth borehole wall produces a less noisy image, although pronounced tool eccentering creates two dark vertical bands. DINOv3 follows several of the manually interpreted traces despite these bands, whereas the models trained from scratch miss more traces or add unmatched picks. In Borehole B (Figure 6b), the rougher borehole wall produces a speckled image in which fracture traces are harder to distinguish from background texture. DINOv3 recovers more of the main interpreted traces in this example, although some of its picks remain unmatched. These contrasting examples motivate the full-log comparison below.

Across the full logs, confidence scores helps interpreters assess whether depth and orientation patterns are supported by higher-scoring picks or involve many lower-scoring candidates. With this visual guide, DINOv3’s results show principal NF orientation patterns similar to those in the manual interpretations (Figures 7a and 7b). In Borehole A, the central dip azimuth and mean dip are about $3 1 0 ^ { \circ }$ and $6 0 ^ { \circ }$ for DINOv3, versus $3 0 9 ^ { \circ }$ and $6 1 ^ { \circ }$ for manual interpretation. In Borehole B, the corresponding values are $3 3 8 ^ { \circ }$ and $5 5 ^ { \circ }$ versus $3 3 8 ^ { \circ }$ and $5 7 ^ { \circ }$ . This agreement in overall orientation does not imply that every individual pick matches the manual reference.

At the individual-pick level, DINOv3 misses fewer manually identified NFs and produces fewer unmatched automated picks than the models trained from scratch (Figure 7c). In Borehole A, its FNR is 0.15, compared with 0.49 for U-Net, the lowest FNR among the from-scratch models. This represents a relative reduction of about 69%.

![](images/94ee6c710cc5eded3c0b1a68c6ec7bed99ac817b1484be1b6c758ffbf7df8ed7.jpg)  
Figure 5: Segmentation of four selected test images by the fine-tuned DINOv3 model and the three from-scratch baselines (U-Net, YOLOv8, and Swin Transformer). NF and DITF labels are shown in red and blue, respectively, with manual segmentation as the reference.

DINOv3’s FPR is 0.43, compared with 0.66 for YOLOv8, the lowest FPR among the from-scratch models. The same pattern holds in Borehole B, where DINOv3 has an FNR of 0.19 and an FPR of 0.34, both lower than those of any model trained from scratch.

DINOv3 also gives the lowest mean errors in NF count, dip azimuth, and dip across 5 m intervals in both boreholes (Figures 7d–f). The improvement is especially clear in Borehole B, where its count error is 0.71, 42.7% below YOLOv8’s 1.24, and its dip error is 3.76<sup>◦</sup>, compared with 10.17<sup>◦</sup> for YOLOv8. For dip azimuth, the errors are 13.78<sup>◦</sup> in Borehole A and 10.47<sup>◦</sup> in Borehole B, below the lowest from-scratch values of 23.64<sup>◦</sup> for YOLOv8 and 34.24<sup>◦</sup> for Swin Transformer, respectively.

## 3.3. Drilling-induced tensile fracture characterization

We evaluate DITF picking along two other continuous logs, designated Boreholes C and D. Confidence scores also guide the review of individual DITF candidates by indicating how well their depth persistence and pairing match the expected geometry. Figure 8 also uses darker fitted traces for higher confidence scores and lighter traces for lower scores, helping interpreters prioritize the more plausible candidates and identify lower-scoring picks for closer inspection.

The selected interval from Borehole C (Figure 8a) shows the efects of tool eccentering and logging noise, including a prominent dark band along the right edge of the image. Despite this interference, DINOv3 recovers more of the manually interpreted DITF pair than the models trained from scratch, although some unmatched picks remain. In the selected interval from Borehole D (Figure 8b), geological bedding creates curved and oblique features that complicate

![](images/b9927582a6675a99aa452add4dcc4b388e9e407114d91031e13ad0a9f4ee0302.jpg)  
Figure 6: NF picking in selected 5 m intervals of (a) Borehole A and (b) Borehole B. Green regions denote predicted NF masks, and red curves denote fitted traces, with intensity indicating confidence. Side markers identify missed manual picks (red), unmatched automated picks (yellow), and matched picks (green) under the depth and orientation tolerances.

DITF picking. DINOv3 follows more of the two manually interpreted axial traces than the comparison models, but also detects additional segments.

For complete DITF logs, confidence scores also provide a way to limit the influence of potential false positives on the interpreted depth distributions and prevailing azimuths. We therefore display only DINOv3 picks with confidence greater than 0.5 in the depth–azimuth plots and rose diagrams (Figures 9a and 9b). Figures A1 shows the same comparisons without confidence score filtering. The threshold applies only to Figure 9. The FNR, FPR, and parameter errors reported below are calculated from the full set of picks.

In both Boreholes C and D, the displayed DINOv3 picks closely reproduce the overall depth distribution and azimuth trends with depth of the manual picks. In Borehole C, the rose diagrams give a dominant DITF azimuth of N53<sup>◦</sup>E for DINOv3 and N45<sup>◦</sup>E for manual interpretation, with azimuthal standard deviations of $3 9 ^ { \circ }$ and 51<sup>◦</sup>, respectively. In Borehole D, the corresponding dominant azimuths are $\mathrm { N } 9 0 ^ { \circ } \mathrm { E }$ and ${ \mathrm { N } } 8 8 ^ { \circ } { \mathrm { E } } ,$ with standard deviations of $6 3 ^ { \circ }$ and $4 9 ^ { \circ }$ . The close agreement in DITF azimuths suggests that stress orientations inferred from DINOv3 picks would be consistent with those inferred manually.

(a)  
![](images/c41cb824a5432aa725cca8b51a76d43504fdf9c26dfdd0529bf7aea12de3cd61.jpg)  
Borehole A

![](images/aa209745b15e86d2d82049517212e83da64feb6cea3f0c8411deae5d3749488d.jpg)

(b)  
Borehole B  
![](images/e09f33dde53d70cbc162b1620060d0e8dc810ee520e309579599755ac9e23207.jpg)

![](images/41f3af1c0ccf68ec37a3046443e0aad511317507c86f3545a60791c456405918.jpg)

![](images/d9d2a4b1a8f25ad62d669d74c9b2e6d86bb2b4d39d2cc0a2b3c4db526ea6c467.jpg)

![](images/5f67e0f60e0d240c35d888c8cf063210b06432fda98fe6a333de278b086c07b0.jpg)

![](images/85b1cb56f882a8c06a3269bde426b3d038748e8777ee9936b5cdd12f7495f3ab.jpg)

![](images/da7a0b9fdddb5608d7caa22c387486c29fbc5a06df2479f83e73b13c3b3771df.jpg)  
(f)

(c) FNR & FPR of Al-picked NFs in full logging interval  
(d)  
![](images/d339e6cbf7542eaa00d634848a570196b591f6844f2f52397145f878257cbb14.jpg)

Mean NF count error in 5 m intervals  
![](images/8d308d18b03fc77d5376fe3c52627b199f42c47b77aaf919586c8dda12a0896d.jpg)

(e) Mean NF dip azimuth error in 5 m intervals  
![](images/448ed91a1063665d6487d40b3e80c97d33e9f595e627bf8ad5aa1e6fa8b1fc2a.jpg)

Mean NF dip error in 5 m intervals  
![](images/5293b5607300855500556f6131668f433334c3706242babd3eaba36878b2ee94.jpg)  
Figure 7: NF characterization in Boreholes A and B. (a,b) DINOv3 and manual interpretations shown as tadpole and lower-hemisphere stereographic plots. Automated-pick shading denotes confidence. (c) Full-interval FPR and FNR of the picked NFs. (d–f) Mean absolute 5 m-window errors in fracture count, mean dip azimuth, and mean dip.

At the individual-pick level, DINOv3 misses fewer manually identified DITFs than the models trained from scratch (Figure 9c). Its FNR is 0.33 in Borehole C, compared with 0.67 for YOLOv8, the lowest from-scratch FNR. In Borehole D, DINOv3’s FNR is 0.30 versus 0.67 for Swin Transformer. However, this improvement in detection completeness is accompanied by many unmatched candidates. DINOv3’s FPR remains 0.83 in Borehole C and 0.88 in Borehole D, which is not the lowest in either borehole.

![](images/3fefec3559ae18db50ab43e942479929a4003f6f48da2d5e1155449c3a33bb50.jpg)  
Figure 8: DITF picking in selected 5 m intervals of (a) Borehole C and (b) Borehole D. Predicted masks and fitted centerlines show the recovered structures. Trace intensity indicates confidence. Red, yellow, and green side markers denote missed manual, unmatched automated, and matched picks, respectively.

Beyond the number of detected DITFs, we assess how accurately the picks describe their total length and orientation within 5 m intervals. DINOv3 gives the lowest mean error in total DITF length in both boreholes (Figure 9d). Its errors of 1.98 m in Borehole C and 1.40 m in Borehole D are 12.8% and 45.1% below Swin Transformer, the best from-scratch model for length in both boreholes. Orientation estimates are less consistently improved (Figures 9e and 9f). DINOv3 has the lowest azimuth error in Borehole C and the lowest tilt error in Borehole D, but U-Net gives a smaller azimuth error in Borehole D, at 6.96<sup>◦</sup> versus 12.55<sup>◦</sup> for DINOv3. Its tilt error in Borehole C is also higher than those of the three from-scratch models. Thus, DINOv3’s advantage in detecting DITFs and estimating their total length does not extend to every orientation measure.

(a)  
![](images/9cf2f5b0e7d506f6d7706e02f58a04051354f98ea2957a404a5bcba110b51bcd.jpg)  
Borehole C

![](images/e98473c7ff07b1650492bbe5e1082430ac81a27e3a9c7b24dbac1cf030d6fd65.jpg)

(b)  
![](images/870bf85af6dd935f1bd42a52754be49f9bdf727061d0e97e63532a0c8e45361a.jpg)  
Borehole D

![](images/61527cae6462c48b04f76e53fae70b3163f40b09c3a58615c6cbcf9e888b4038.jpg)

![](images/2ff8f43e04a914a700fbe1ee342618b2089dad6302c836ae7b5fac0d9b38d36b.jpg)

![](images/a1d1262e8a8793e43912d64a8537a5364e91f3d54b6b50dd71ce1af92e035f66.jpg)

![](images/ff90ca012f8fea84843eca48fd3b9d1d7db31324d33a0b51bc3b90fd67cf2d37.jpg)  
(c) FNR & FPR of Al-picked DITFs in whole logging interval

![](images/6baf5f612eaf2738a126adc0c15fb9393617b28e0b8ab7fa1117ed8449844cc5.jpg)

(d)  
![](images/aa9cf1ed54a9c1676b6cd5caeb9f16f6587327ce819b3bede279a2f03dfd6952.jpg)

Mean DITF length error in 5 m intervals  
![](images/4bd6736eb4e5c81da151953c76178b005253690df81ed0e7ea262b907a5a14c3.jpg)

(f)  
(e) Mean DITF azimuth error in 5 m intervals  
![](images/9f04b0ecf3b646b9e24ae5023282212e94f0a382acd8e6f114893cb89f81498a.jpg)

Mean DITF tilt error in 5 m intervals  
![](images/287b4c6f4c5db3d69f8679e5ecfc0c1a5a0c388692934f8df7b20aab455e0b85.jpg)  
Figure 9: DITF characterization in Boreholes C and D. (a,b) Depth–azimuth and rose diagram comparisons of DINOv3 and manual interpretations. Only DITFs with confidence scores higher than 0.5 are displayed. Bars represent DITF trace spans and dots represent central azimuths. (c) Full-interval FNR and FPR of the picked DITFs, and (d–f) mean absolute errors in DITF length, mean azimuth, and mean tilt within 5 m windows. Note that panels (c–f) evaluate the full picking products rather than only the displayed high-confidence subset.

## 4. Discussion

Using VFM adaptation in practice requires understanding both the basis of its performance and the choices that afect it when labeled data are limited. Therefore, we first examine whether DINOv3 responds to fracture traces before fine-tuning. Then, we assess how sample size, decoder design, fine-tuning strategy, and image enhancement afect performance. These analyses help identify suitable adaptation settings and clarify how much their benefits depend on the available data. We also discuss the limits of the current evaluation and their implications for geological and stress interpretation.

## 4.1. DINOv3 responses to fracture traces before fine-tuning

Fine-tuned DINOv3 generally outperformed the models trained from scratch. A possible explanation is its selfsupervised pretraining on a large collection of satellite images. Although satellite images difer from acoustic image logs, pretraining may have enabled DINOv3 to learn visual patterns useful for identifying fractures. We therefore examine whether DINOv3 already responds to fracture traces in acoustic images before fine-tuning. Figure 10 displays these responses as color images. Principal component analysis (PCA) summarizes the model’s responses at each image location in three values, which are displayed as red, green, and blue. Within each image, similar colors generally indicate similar model responses, whereas contrasting colors highlight diferences. A fracture trace with a diferent color from the surrounding background therefore indicates that the model responds diferently to these regions. No color is assigned a fixed meaning such as NF, DITF, or background, and the colors do not represent confidence scores.

In several examples, curved bands in the color images follow the sinusoidal NF traces, while near-vertical bands coincide with DITFs. These responses appear in images with diferent background textures and brightness patterns. Their clarity varies across examples, and the color bands are broader and less detailed than the original fracture traces. Background regions also produce changes in color. The pretrained model therefore responds to some fracture shapes, but these responses do not by themselves identify which structures should be labeled as NFs or DITFs.

A possible reason for these responses is that satellite images and acoustic image logs share visual patterns such as edges, elongated structures, and changes in texture. Learning these patterns during pretraining may give DINOv3 a useful starting point for locating fractures, even though satellite images do not teach it the geological meaning of an acoustic fracture trace. Fine-tuning can then link the existing responses to the NF and DITF labels and adjust them to the appearance of fractures in acoustic images. This provides a possible explanation for how the model can benefit from a small labeled dataset, since it can build on patterns learned before the fracture annotations are introduced.

![](images/971e3f5ec18d293af825473670db50efb99147a975b6a2d7f34aafecd3e9ccc4.jpg)  
Figure 10: DINOv3 responses to acoustic image logs before fine-tuning. PCA summarizes the responses at each image location in three values displayed as red, green, and blue. Within each image, similar colors generally indicate similar model responses, while contrasting colors indicate diferent responses. Examples are drawn from the validation and test sets.

## 4.2. Influence of training sample size

To assess whether DINOv3’s advantage persists with fewer than 80 labeled images, we repeat the comparison using smaller training sets. Reducing the number of training images afects segmentation and fracture picking diferently (Figure 11). For segmentation (Figure 11a), DINOv3 gives the highest mean test IoU from 20 samples onward, but U-Net and Swin Transformer perform better with five and 10 samples, respectively. With 40 images, DINOv3 achieves an IoU of 0.571, exceeding the best result obtained by a model trained from scratch with 80 images, U-Net’s 0.545. This comparison shows that DINOv3 can achieve better segmentation with half as many labeled images, although its advantage does not persist at the smallest sample sizes.

DINOv3’s advantage is more consistent for NF picking, where it leads at every tested sample size (Figure 11b). With only five labeled images, its score reaches 47.47, exceeding the scores of all three models trained from scratch with 80 images, which are no higher than 33.14. This difers from the segmentation ranking at the smallest sample size because IoU and fracture picking measure diferent aspects of the predictions. IoU is sensitive to trace thickness and boundary placement, whereas fracture depths and orientations depend on the position and shape of the extracted centerlines. A model can therefore produce useful NF measurements without achieving the highest IoU.

For DITF picking, DINOv3’s advantage is less consistent when very few labeled images are available (Figure 11c). YOLOv8 performs better with five images, and both U-Net and YOLOv8 perform better with 10. DINOv3 leads from 20 samples onward. DITF traces vary in shape and can be dificult to distinguish from noise, so adapting the model may require enough examples to cover this variation. The picking scores should also be read alongside the missed-pick fractions and parameter errors, because favorable parameter errors can partly ofset missed detections in the combined score.

These results show that the advantage of VFM adaptation does not necessarily grow as the size of dataset becomes smaller. With very few training images, DINOv3’s advantages in segmentation and DITF picking weaken or disappear, although its NF picking advantage persists. The irregular changes in some curves may reflect diferences in the selected images and variation during training. Nevertheless, within the tested range, larger training sets generally improve DINOv3’s performance and make its advantage over models trained from scratch clearer. These results support VFM adaptation as an efective approach when labeled data are limited, while showing that additional fracture annotations can further improve performance and strengthen its advantage over training from scratch.

## 4.3. Influence of decoder and fine-tuning strategy

Adapting DINOv3 requires choosing both how to turn its image responses into fracture regions and how much of the pretrained model to update. These choices matter with limited data because training must preserve useful patterns while learning the appearance of fractures in acoustic images. We select three established decoder designs that difer in which encoder outputs they use and how they combine them.

The progressive upsampling (PUP) decoder uses the final encoder output and enlarges it step by step to produce a fracture map. The multi-level feature aggregation (MLA) decoder combines outputs from several encoder layers before producing the final prediction. Both decoders follow the designs of Zheng et al. (2021). The dense prediction transformer (DPT)-based decoder also uses several encoder layers, but merges their outputs progressively as the prediction is enlarged (Ranftl et al., 2021). These designs represent diferent ways of recovering image detail from the encoder output. Comparing them tests whether combining information from several layers improves fracture characterization over using the final layer alone.

Each decoder is tested with three fine-tuning strategies (Figure 12). Frozen keeps the pretrained encoder unchanged, whereas Unfrozen, or full fine-tuning, updates the entire model. LoRA keeps the encoder’s original weights fixed and learns a small set of added adjustments (Hu et al., 2022). The decoder is trained in every case, and all comparisons use the same pretrained encoder and numbers of labeled images.

PUP with full fine-tuning gives the strongest overall results with 80 training images. Its test IoU reaches 0.624, exceeding the best result from the other decoders (Figure 12a). It also gives the highest NF and DITF picking scores (Figures 12b and 12c). With the PUP decoder held constant, full fine-tuning outperforms both freezing and LoRA on all three measures at this sample size. Its benefit therefore extends from locating fracture pixels to obtaining useful fracture measurements.

Full fine-tuning is nevertheless not the best choice for every decoder and sample size. With only five training images, frozen PUP gives a slightly higher NF picking score than fully fine-tuned PUP. Frozen MLA also exceeds fully fine-tuned MLA in test IoU and DITF picking with 80 images. These exceptions suggest that updating more of the model can help or hinder performance depending on the decoder and the available examples. Similarly, a more complex decoder does not necessarily produce better fracture measurements. LoRA reduces the number of parameters updated during training, but this alone does not guarantee higher accuracy, and the present experiments do not compare training time or computing cost.

![](images/e5bef5856e56fbad159fefb8050c0507090a1250763750c2151bde77d31d0022.jpg)

![](images/85696f239f422ebe0f2ff025afe9f09d14940c9d663837ef481eb08efc5bf8ef.jpg)

![](images/f11afcaa9e10daf498574ca570001f6891c42d33af6f2bd003fe03f19b9fa0b7.jpg)  
Figure 11: Efect of training/fine-tuning sample amount on (a) mean IoU score for the test set, (b) mean NF picking score for Boreholes A and B, and (c) mean DITF picking score for Boreholes C and D. Picking scores combine detection and parameter accuracy as defined in Eq. 1.

PUP gradually enlarges the encoder’s output to produce a fracture map, while full fine-tuning allows the pretrained model to adjust to acoustic-image patterns. This combination may help retain fracture positions and shapes while learning to distinguish them from the background. The experiments show that limited labeled data do not automatically favor freezing the encoder or updating only a small part of it. For the tested dataset with 80 labeled images, the relatively simple PUP decoder combined with full fine-tuning gives the best overall fracture characterization.

## 4.4. Efects of image enhancement on fracture characterization

Fracture traces can be faint or obscured by background texture, suggesting that making them more visible might improve automated characterization. We therefore test whether image enhancement helps the models recover fractures and estimate their parameters more accurately. The processing combines automatic gain control (AGC), which adjusts local brightness and contrast, with ridge enhancement, which emphasizes narrow, elongated structures. The ridgeenhancement procedure includes the Sato line filter (Sato et al., 1998). For DINOv3, AGC-adjusted and ridge-filtered images replace the repeated copies of raw amplitude used in the original input. The comparison tests the combined processing, and each model uses the same processing during training and prediction.

Figure 13 illustrates why a more visible fracture does not necessarily lead to a better prediction. Contrast adjustment can reveal faint traces, and ridge filtering can make narrow features more prominent. The same processing can also strengthen background features that resemble fractures or remove brightness diferences that help distinguish fractures from their surroundings. These competing efects ofer a possible explanation for the mixed prediction results, although the combined experiment does not separate the contributions of AGC and ridge filtering.

The quantitative comparison confirms that enhancement is not consistently beneficial (Figure 14). With 80 training images, it slightly improves segmentation IoU for all three models trained from scratch, but DINOv3’s IoU falls from 0.624 to 0.607 (Figure 14a). Even with enhancement, the models trained from scratch remain less accurate than DINOv3 using raw amplitude images. Enhancement does improve $\mathsf { D I N O v } 3 ^ { \circ } \mathrm { s }$ IoU with 20 training images, showing that its efect can change with sample size. Improving visual contrast therefore does not ensure better segmentation, even for the same model.

The fracture-picking results show the same dependence on the model and fracture type. With 80 training images, enhancement improves U-Net’s NF picking score but reduces DINOv3’s by 12.95 points (Figure 14b). For DITFs, YOLOv8 and Swin Transformer benefit, whereas DINOv3’s score decreases (Figure 14c). Some enhanced U-Net experiments with very few training images produce no comparable DITF estimates for calculating certain errors. Their scores therefore include fewer measures and should not be interpreted as equivalent to scores calculated from every measure. For the tested data, the clearest practical choice is to fine-tune DINOv3 fully with the PUP decoder using raw amplitude images. Making fractures more visible through image enhancement does not guarantee more accurate picks and was not needed to obtain the best overall performance.

![](images/570ed221c80e583715092e60a8f79383a809089cecd0f465d462c9fc4b1ad1b9.jpg)

(b) Mean NF picking scores of borehole A and B  
![](images/1b082320a2889e926e6368353566e533c96c6a5659b0e421282208c0f5428e8f.jpg)

(c) Mean DITF picking scores of borehole C and D  
![](images/73fb1b8e8696444dd36eb71208f55fda4bbbde82f6d725d5a6e68a9ea138d66a.jpg)  
Figure 12: Efects of decoder choice and fine-tuning strategy. Columns compare PUP, MLA, and DPT decoders. Curves compare Frozen, Unfrozen, and LoRA fine-tuning strategies. Rows show (a) mean IoU for the test set, (b) mean NF picking score for Boreholes A and B, and (c) mean DITF picking score for Boreholes C and D.

![](images/b6987851ff731a9f785cefbeedc025bb131c55d6e37d68fb3148a57695de68d8.jpg)  
Figure 13: Efect of image enhancement on DINOv3 fracture segmentation in selected test images. Columns show raw amplitude, the AGC-adjusted image, the ridge-filtered image, the prediction using raw amplitude, the prediction using the combined AGC and ridge-filtered images, and the manual labels.

## 4.5. Practical implications and limitations

The main practical benefit is the ability to develop a fracture-characterization workflow from a small set of expertlabeled acoustic images. This ofers a way to apply automated interpretation in projects without a large existing collection of fracture annotations. The improved NF picks and orientation estimates can support interpretation of rock mass structure along continuous logs. DITF picks can also help identify prevailing fracture azimuths for stressorientation assessment, while confidence scores help prioritize candidates for review against the original images.

The usefulness of these outputs is nevertheless limited by what can be distinguished in the acoustic images. In particular, logging noise, tool eccentering, rough borehole walls, and geological bedding can obscure fractures or produce similar-looking patterns. As a result, some traces remain ambiguous even during expert review, and manual interpretations can also be uncertain. Agreement with manual picks therefore indicates consistency with an interpretation reference, but does not confirm the presence or absence of every physical fracture.

The parameter estimates also depend on simplified descriptions of fracture shape. A sinusoid summarizes an NF trace, and a straight line summarizes the overall trend of a DITF, but irregular or discontinuous traces may not follow these shapes closely. Image alignment, depth resolution, and borehole geometry further limit the precision of the fitted depths and orientations. Confidence scores inherit these geometric assumptions. For example, a true DITF can receive a low score if its opposite trace is not visible, while a background feature with a plausible shape may receive a higher score. Nevertheless, confidence scores remain useful for identifying more plausible fracture candidates and guiding their review against the original images.

(a)

(c)  
![](images/584a6847676b159c125fc32e2e94ac8517e3095ac21a651da3bb0eced4ffb28a.jpg)

![](images/db78bf4d56d131e4385fddb34e2f0f473196290e4aecc596a664cae87b26df53.jpg)

![](images/64e2eef023f4479607e90517bbedd2e07ffb05697e3392c49ee74ee20bfe0e68.jpg)  
Figure 14: Changes caused by image enhancement in (a) mean IoU score for the test set, (b) NF picking score for Boreholes A and B, and (c) DITF picking score for Boreholes C and D. Values are calculated as the enhanced-input result minus the raw-input result. Positive values favor enhancement and negative values favor raw amplitude.

The practical role of the workflow is therefore to assist geological and stress interpretation. It automates trace detection and fitting and supplies candidate fractures and parameter estimates for review. Expert judgment remains essential for ambiguous traces and for assessing the geological meaning of the resulting fracture patterns.

## 5. Conclusions

This study investigated whether adapting a pretrained VFM improves fracture characterization in acoustic image logs compared with training task-specific models from scratch on the same limited labeled dataset. We adapted a DINOv3 model pretrained on satellite imagery and combined fracture segmentation with geometric processing to estimate NF and DITF depths and orientations.

With only 80 labeled images, fine-tuned DINOv3 achieved higher segmentation accuracy and more accurate NF detection, counts, and orientations than the models trained from scratch. For DITFs, detection and length estimates also improved, although unmatched picks remained common and orientation gains were less consistent. However, with very few labeled images, its segmentation and DITF picking advantages weakened or disappeared, whereas its NF picking advantage persisted. Further comparisons identified PUP with full fine-tuning using raw amplitude images as the best overall combination. In contrast, image enhancement did not consistently improve fracture characterization and was unnecessary for obtaining the best results.

Under the tested conditions, fine-tuning DINOv3 yielded better overall fracture characterization than training taskspecific models from scratch on the same limited labeled dataset. These results support VFM adaptation as an efective strategy for acoustic image log interpretation with limited annotations. The resulting fracture measurements can support geological and stress analysis. However, expert review remains necessary, particularly for fractures with relatively low confidence scores. Finally, this study provides a useful reference for adapting VFMs to other geoscientific image analysis tasks with limited labeled data, particularly for selecting decoders and fine-tuning strategies.

## CRediT authorship contribution statement

Guangyu Wang: Writing – original draft, Visualization, Methodology, Formal analysis, Data curation. Xiaodong Ma: Writing – review & editing, Validation, Supervision, Resources, Methodology, Investigation, Funding acquisition, Formal analysis, Conceptualization. Xinming Wu: Writing – review & editing, Validation, Supervision, Methodology, Investigation, Formal analysis, Conceptualization.

## Declaration of generative AI use

During the preparation of this manuscript, the authors used ChatGPT to improve language clarity and readability. All AI-assisted content was reviewed and revised by the authors, who take full responsibility for the manuscript.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This work is financially supported by the National Science and Technology Major Project of China (Grant No. 2024ZD1000705).

## Appendix A. DITF picks without confidence score filtering

![](images/f221ee8b9da769d9edb69592d01686832fb899b8842112935b5d628f7253e299.jpg)  
Figure A1: Depth–azimuth plots and rose diagrams comparing DINOv3 and manual DITF picks in (a) Borehole C and (b) Borehole D. These panels correspond to Figures 9a and 9b but include all DINOv3 picks regardless of confidence score.

## Data and code availability

The source code and dataset used in this study will be made publicly available upon acceptance of the manuscript.

## References

Afshari Moein, M.J., Somogyvári, M., Valley, B., Jalali, M., Loew, S., Bayer, P., 2018. Fracture network characterization using stress-based tomography. Journal of Geophysical Research: Solid Earth 123, 9324–9340. doi:10.1029/2018JB016438.

Assous, S., Elkington, P., Clark, S., Whetton, J., 2014. Automated detection of planar geologic features in borehole images. Geophysics 79, D11–D19. doi:10.1190/geo2013-0189.1.

Bommasani, R., Hudson, D.A., Adeli, E., et al., 2021. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258. doi:10.48550/arXiv.2108.07258.

Brudy, M., Zoback, M.D., 1999. Drilling-induced tensile wall-fractures: Implications for determination of in-situ stress orientation and magnitude. International Journal of Rock Mechanics and Mining Sciences 36, 191–215. doi:10.1016/S0148-9062(98)00182-X.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging properties in self-supervised vision transformers, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9650–9660. doi:10.1109/ICCV48922.2021. 00951.

Chen, D., Zhang, E., Niu, X., Wu, H., Zhang, H., Li, Z., Jin, X., Liu, D., Niu, Z., Wang, H., 2026. The fracture characteristics and dynamic response to hydraulic fracturing in the Gonghe enhanced geothermal system (egs) field, Qinghai province, northwest China. Geothermics 137, 103623. doi:10.1016/j.geothermics.2026.103623.

Cruz, R.A.Q., Cacau, D.C., dos Santos, R.M., Pereira, E.J.R., Leta, F.R., Clua, E.W.G., 2017. Improving accuracy of automatic fracture detection in borehole images with deep learning and GPUs, in: 2017 30th SIBGRAPI Conference on Graphics, Patterns and Images, pp. 345–350. doi:10.1109/SIBGRAPI.2017.52.

Damm, S., Laszkiewicz, M., Lederer, J., Fischer, A., 2025. Anomalydino: Boosting patch-based few-shot anomaly detection with dinov2, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE. pp. 1319–1329. doi:10.1109/WACV61041.2025.00136.

Dias, L.O., Bom, C.R., Faria, E.L., Valentín, M.B., Correia, M.D., de Albuquerque, M.P., de Albuquerque, M.P., Coelho, J.M., 2020. Automatic detection of fractures and breakouts patterns in acoustic borehole image logs using fast-region convolutional neural networks. Journal of Petroleum Science and Engineering 191, 107099. doi:10.1016/j.petrol.2020.107099.

Glossop, K., Lisboa, P.J.G., Russell, P.C., Siddans, A., Jones, G.R., 1999. An implementation of the Hough transformation for the identification and labelling of fixed period sinusoidal curves. Computer Vision and Image Understanding 74, 96–100. doi:10.1006/cviu.1999.0747.

Guo, Z., Wu, X., Liang, L., Sheng, H., Chen, N., Bi, Z., 2025. Cross-domain foundation model adaptation: Pioneering computer vision models for geophysical data analysis. Journal of Geophysical Research: Machine Learning and Computation 2, e2025JH000601. doi:10.1029/ 2025JH000601

Hall, J., Ponzi, M., Gonfalini, M., Maletti, G., 1996. Automatic extraction and characterisation of geological features and textures from borehole images and core photographs, in: SPWLA 37th Annual Logging Symposium.

Han, J., Zhang, X., Qi, Y., Liu, L., 2025. Intelligent identification of fractures and holes in ultrasonic logging images based on the improved YOLOv8 model. Artificial Intelligence in Geosciences 6, 100167. doi:10.1016/j.aiig.2025.100167.

Han, S., Xiao, X., Song, B., Guan, T., Zhang, Y., Lyu, M., 2023. Automatic borehole fracture detection and characterization with tailored Faster R-CNN and simplified Hough transform. Engineering Applications of Artificial Intelligence 126, 107024. doi:10.1016/j.engappai.2023. 107024.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2022. LoRA: Low-rank adaptation of large language models, in: International Conference on Learning Representations. URL: https://openreview.net/forum?id=nZeVKeeFYf9, doi:10.48550/arXiv. 2106.09685.

Jocher, G., Chaurasia, A., Qiu, J., 2023. Ultralytics YOLOv8. URL: https://github.com/ultralytics/ultralytics.

Kherroubi, J., 2008. Automatic extraction of natural fracture traces from borehole images, in: Proceedings of the 19th International Conference on Pattern Recognition (ICPR), pp. 1–4. doi:10.1109/ICPR.2008.4761203.

Kundu, B., Khanal, B., Simon, R., Linte, C., 2025. Assessing the performance of the dinov2 self-supervised learning vision transformer model for the segmentation of the left atrium from mri images, in: Medical Imaging 2025: Image-Guided Procedures, Robotic Interventions, and Modeling, SPIE. pp. 145–150. doi:10.1117/12.3048192.

Lai, J., Chen, K., Xin, Y., Wu, X., Chen, X., Yang, K., Song, Q., Wang, G., Ding, X., 2021. Fracture characterization and detection in the deep Cambrian dolostones in the Tarim Basin, China: Insights from borehole image and sonic logs. Journal of Petroleum Science and Engineering 196, 107659. doi:10.1016/j.petrol.2020.107659.

Li, Z., Zhang, G., Zhang, X., Zhang, X., Long, Y., Sun, Y., Lin, C., 2025. EILNet: An intelligent model for the segmentation of multiple fracture types in karst carbonate reservoirs using electrical image logs. Natural Gas Industry B 12, 158–173. doi:10.1016/j.ngib.2025.03.002.

Liu, Y., Liao, G., Xiao, L., Liang, Z., Zhang, J., Zhang, X., Zhang, Z., Zhou, J., Li, G., 2022. Automatic fracture segmentation and detection from image logging using Mask R-CNN, in: SPWLA 63rd Annual Logging Symposium. doi:10.30632/SPWLA-2022-0115.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B., 2021. Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022. doi:10.1109/ICCV48922.2021.00986.

Massiot, C., Nicol, A., McNamara, D.D., Townend, J., 2017a. Evidence for tectonic, lithologic, and thermal controls on fracture system geometries in an andesitic high-temperature geothermal field. Journal of Geophysical Research: Solid Earth 122, 6853–6874. doi:10.1002/2017JB014121.

Massiot, C., Townend, J., Nicol, A., McNamara, D.D., 2017b. Statistical methods of fracture characterization using acoustic borehole televiewer log interpretation. Journal of Geophysical Research: Solid Earth 122, 6836–6852. doi:10.1002/2017JB014115.

Milad, B., Ghosh, S., Slatt, R., Marfurt, K., 2018. Upscaled DFN models to understand the efects of natural fracture properties on fluid flow in the Hunton Group tight limestone, in: Proceedings of the Unconventional Resources Technology Conference (URTeC). doi:10.15530/ urtec-2018-2903038.

Nasim, M.Q., Maiti, T., Mosavat, N., Grech, P.V., Singh, T., Singha Roy, P.N., 2025. Automated detection of geological features: Leveraging deep learning for beddings and fractures identification in image logs. SPE Journal doi:10.2118/223976-PA.

Nian, T., Wang, G., Xiao, C., Zhou, L., Deng, L., Li, R., 2016. The in situ stress determination from borehole image logs in the Kuqa Depression. Journal of Natural Gas Science and Engineering 34, 1077–1084. doi:10.1016/j.jngse.2016.08.005.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2024. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research doi:10.48550/arXiv.2304. 07193.

Ranftl, R., Bochkovskiy, A., Koltun, V., 2021. Vision transformers for dense prediction, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12179–12188. URL: https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl\_Vision\_ Transformers\_for\_Dense\_Prediction\_ICCV\_2021\_paper.html.

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 234–241. doi:10.1007/978-3-319-24574-4\_28.

Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S., Koller, T., Gerig, G., Kikinis, R., 1998. Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images. Medical image analysis 2, 143–168. doi:10.1016/S1361-8415(98) 80009-1.

Shahri, M.P., Chok, H., Huang, J., Elkington, P., 2016. Stress inversion via borehole image log and fracturing data: Integrated approach, in: Proceedings of the Unconventional Resources Technology Conference (URTeC). doi:10.15530/urtec-2016-2461241.

Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al., 2025. DINOv3. arXiv preprint arXiv:2508.10104. doi:10.48550/arXiv.2508.10104. meta AI Research.

Valley, B., Evans, K.F., 2009. Stress orientation to 5 km depth in the basement below Basel (Switzerland) from borehole failure analysis. Swiss Journal of Geosciences 102, 467. doi:10.1007/s00015-009-1335-z.

Valverde, J.M., Papadopoulos, D.P., Larsen, R., Dahl, A.B., 2026. Towards high-quality image segmentation: Improving topology accuracy by penalizing neighbor pixels. arXiv preprint arXiv:2603.18671 doi:10.48550/arXiv.2603.18671. accepted to CVPR 2026.

Wang, W., 2005. An edge based segmentation algorithm for rock fracture tracing, in: Proceedings of the 2nd International Conference on Computer Graphics, Imaging and Visualization (CGIV), pp. 43–48. doi:10.1109/CGIV.2005.16.

Wang, W., Zhou, L., 2023. Fracture extraction from logging image using a dual encoder-decoder architecture with Swin Transformer. Petrophysics 64, 38–49. doi:10.30632/PJV64N1-2023a3.

Wenning, Q.C., Berthet, T., Ask, M., Zappone, A., Rosberg, J.E., Almqvist, B.S.G., 2017. Image log analysis of in situ stress orientation, breakout growth, and natural geologic structures to 2.5 km depth in central Scandinavian Caledonides: Results from the COSC-1 borehole. Journal of Geophysical Research: Solid Earth 122, 3999–4019. doi:10.1002/2016JB013776.

Xavier, A., Guerra, C.E., Andrade, A., 2015. Fracture analysis in borehole acoustic images using mathematical morphology. Journal of Geophysics and Engineering 12, 492–501. doi:10.1088/1742-2132/12/3/492.

Zhang, W., Li, Z., Wu, T., Yao, Z., Qiu, A., Li, Y., Shi, Y., 2022. Fracture identification in well logging images: Two-stage adaptive network. IEEE Transactions on Instrumentation and Measurement 71, 5003112. doi:10.1109/TIM.2021.3130671.

Zhang, W., Wu, T., Li, Z., Liu, S., Qiu, A., Li, Y., Shi, Y., 2021. Fracture recognition in ultrasonic logging images via unsupervised segmentation network. Earth Science Informatics 14, 955–964. doi:10.1007/s12145-021-00605-6.

Zheng, S., Lu, J., Zhao, H., Zhu, X., Luo, Z., Wang, Y., Fu, Y., Feng, J., Xiang, T., Torr, P.H., et al., 2021. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6881–6890. doi:10.1109/CVPR46437.2021.00681.

Zoback, M.D., Barton, C.A., Brudy, M., Castillo, D.A., Finkbeiner, T., Grollimund, B.R., Moos, D.B., Peska, P., Ward, C.D., Wiprut, D.J., 2003 Determination of stress orientation and magnitude in deep wells. International Journal of Rock Mechanics and Mining Sciences 40, 1049–1076. doi: / 7 .