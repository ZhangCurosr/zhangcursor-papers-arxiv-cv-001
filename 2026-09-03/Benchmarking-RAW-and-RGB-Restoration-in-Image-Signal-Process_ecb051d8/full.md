# Benchmarking RAW and RGB Restoration in Image Signal Processors

Zihao Lu   
zihao.lu@stud-mail.uni-wuerzburg.de   
Radu Timofte   
6radu.timofte@uni-wuerzburg.de

<sup>2</sup>Marcos V. Conde <sup>†</sup> marcos.conde@uni-wuerzburg.de

Computer Vision Laboratory CAIDAS and IFI University of Würzburg Würzburg, Germany

<sup>†</sup> Corresponding Author https://github.com/mv-lab/AISP

![](images/ab39dc301db23a1d867a13a8454e6e32b2ff4f23d64dcce96b13288b68318b64.jpg)  
<sub>1</sub>Figure 1: Where should restoration happen in camera pipelines? Considering the limited 3computational budget in mobile devices, we compare two strategies around a fixed Image <sup>8</sup>Signal Processor: (A) pre-ISP RAW restoration and (B) post-ISP RGB restoration.

## Abstract

Modern cameras transform RAW sensor measurements into sRGB images through an image signal processor (ISP). We benchmark two placements for blind restoration around a fixed ISP: (A) pre-ISP restoration in the RAW domain and (B) post-ISP restoration in the sRGB domain. The benchmark covers four smartphone device groups, two learned ISPs, three degradation regimes—noise, blur, and joint noise and blur—, and several representative RAW and RGB restoration models.

Our results show that placement alone does not determine performance. The RAW restoration strategy outperforms the best generic RGB restoration models. However, RGB restoration models trained considering the ISP transformations, achieve the best overall performance. Our novel benchmark demonstrates that the image reconstruction performance strongly depends on the alignment between the restoration model and the target imaging pipeline. We consequently recommend reporting restoration placement and ISP-aware supervision as key experimental factors.

![](images/7741ee10ff7291efa2b41f373af6f811a712673c591cc397f24653848564cd28.jpg)  
Figure 2: A classical Image Signal Processor [11].

## 1 Introduction and Motivation

Image restoration aims to recover high-quality images from degraded observations, such as noise, blur, and their joint effects. Most existing restoration methods operate in the RGB domain, where large-scale datasets and mature network architectures have led to strong performance on standard restoration benchmarks. However, RGB images are not direct sensor measurements. Digital cameras and smartphones employ a dedicated Image Signal Processor (ISP) to transform RAW sensor measurements into visually pleasing RGB images.

A simplified ISP pipeline can be expressed mathematically as:

$$
\mathbf { y } _ { r g b } = \mathrm { I S P } ( \mathbf { x } _ { r a w } ) = \gamma ( \mathbf { C } ( \mathbf { W } ( \mathcal { D } ( \mathbf { x } _ { r a w } ) ) ) )\tag{1}
$$

where $\mathbf { x } _ { r a w } \in \mathbb { R } ^ { H \times W }$ represents mosaiced RAW sensor data (typically following a Bayer RGGB pattern), and $\mathbf { y } _ { r g b } \in \mathbb { R } ^ { H \times W \times 3 }$ is the final sRGB image. The core operations include (i) demosaicing D(·) to interpolate missing color values, (ii) white balancing W to correct color temperature, (iii) a color correction matrix (CCM) C for device-specific color calibration, and (iv) tone mapping $\gamma ( \cdot )$ to compress dynamic range and apply gamma correction. Figure 2 illustrates this camera pipeline.

In practice, white balance and CCM parameters are typically calibrated or tuned based on sensor characterization and color charts [19]. The tone mapping γ(·) is highly nonlinear (often a power function with $\gamma \approx 1 / 2 . 2$ plus S-curves), which can amplify noise in darker regions and create complex interactions between degradations. Consequently, post-ISP restoration must handle artifacts that have been transformed by the camera pipeline. Conversely, a post-ISP model trained on outputs from the target pipeline can learn these ISP-specific artifacts directly.

These observations motivate a controlled comparison of restoration before and after the ISP. Existing comparisons often confound restoration domain with degradation synthesis, training data, camera pipeline, or evaluation protocol. It therefore remains unclear whether performance differences arise from the RAW or RGB domain itself or from alignment between the restoration training distribution and the target pipeline.

To compare RAW pre-processing and RGB post-processing, we assume that the ISP is fixed and treated as a black box: its weights are never updated jointly with the restoration model, and the restoration model does not use ISP parameters or gradients. This reflects proprietary camera pipelines whose internal operations are inaccessible. Recent work has explored learning ISP transformations end-to-end [6, 15, 28], but practical challenges remain: (i) large-scale paired RAW–RGB training data are expensive and device-specific; (ii) real camera ISPs are commonly proprietary and non-differentiable; and (iii) their behavior varies with lighting, scene content, and exposure. A restoration model may nevertheless be trained on examples from the output distribution of a fixed target ISP without accessing its internal parameters and configuration.

As shown in Figure 1, Strategy A always denotes pre-ISP RAW restoration, whereas Strategy B always denotes post-ISP RGB restoration. Independently of placement, we distinguish generic models, which are not trained on outputs from the target ISP, from ISPaware models, which are trained using examples from the target fixed ISP’s output distribution. Under this terminology, RAW restoration is stronger than generic RGB restoration (using all-in-one pre-trained models), while ISP-aware RGB restoration performs best in the sensor-specific comparison.

Contributions In this paper, we introduce a systematic comparison framework for evaluating pre-ISP RAW restoration and post-ISP RGB restoration with fixed ISPs. The framework separates restoration placement from the training regime, allowing us to distinguish domain effects from alignment with the target ISP distribution.

Our work fills this gap through a multi-camera evaluation spanning four smartphone groups and three degradation levels (denoising, deblurring, and joint restoration), together with real-world examples.

Our experimental validation shows that restoration performance depends strongly on the training distribution. Generic pre-trained RGB methods struggle with ISP-transformed degradations, whereas an RGB model trained using outputs from the fixed target ISP can recover both input degradations and pipeline-specific artifacts. This finding qualifies any universal claim that either RAW or RGB restoration is intrinsically preferable.

## 2 Related Work

## 2.1 Learned Image Signal Processing

Traditional ISPs rely on carefully tuned pipelines with handcrafted parameters [19]. Recent work has explored replacing these with end-to-end learned models that directly map RAW sensor data to sRGB images [6, 23, 30, 33, 35]. While these methods demonstrate impressive results under controlled conditions, they face practical limitations: training requires extensive paired RAW-RGB datasets that are sensor-specific [15, 16], and, crucially, robustness to input degradations is rarely evaluated. Our work instead treats the ISP as a fixed component and benchmarks modular restoration before and after it.

## 2.2 Image Restoration

RGB Image Restoration Deep learning has enabled remarkable progress in blind image restoration. Task-specific methods target denoising [37, 40], deblurring [14, 25], and super-resolution [39], while all-in-one networks [8, 20, 21, 27] handle multiple types of degradation using learned prompts or contrastive learning.

When applied to camera pipelines, generic RGB restoration methods face a distribution mismatch. They operate on nonlinear sRGB images after tone mapping and other device-specific processing, which changes the appearance of noise, blur, and content interactions [2, 11]. A model trained on conventional RGB degradations may therefore generalize poorly to outputs from a particular ISP. This motivates both restoration before the ISP, where degradations are better characterized, and target-ISP training for restoration after the ISP.

RAW Image Restoration Most prior works focus on RAW denoising [1, 2, 13, 32] or deblurring in controlled settings [22], while RawSR [34] and BSRAW [9] tackle RAW superresolution with degradation models. Although these works demonstrate benefits of RAWdomain processing, they do not systematically compare RAW pre-processing and RGB postprocessing for smartphone ISPs under different training regimes. Existing RAW restoration work also commonly assumes knowledge of the downstream ISP or jointly trains the restoration and ISP. We instead compare modular restoration around fixed ISPs and analyze how the target training distribution changes the observed RAW–RGB ranking.

## 3 Methodology

## 3.1 Dataset

Our goal is blind restoration of RAW images captured by smartphone cameras. Because obtaining pixel-aligned degraded–clean RAW pairs in real scenes is difficult, we synthesize training data by applying controlled noise and blur to clean RAW captures.

We use the RawIR dataset [5, 10], comprising Vivo X90 Pro, iPhone XS, Samsung S9/S21, and Google Pixel 7–9 images. The fixed train/test counts are 452/10, 955/17, 465/9, and 220/11, respectively, totaling 2,092 training and 47 test images. The test images were sampled approximately in proportion to each device subset and manually checked for image quality and scene and content diversity. All methods and degradation levels use the same split, preventing test-set differences from affecting the comparisons. The relatively small test set remains a limitation when interpreting broad cross-sensor generalization.

The RAW pre-processing pipeline consists of three steps: (i) we normalize each RAW image using its sensor-specific black level and bit depth (typically 10–14 bits); (ii) we pack the mosaiced data into a four-channel RGGB representation, preserving the color filter array (CFA) structure; and (iii) following established RAW-processing practice [2, 34], we crop non-overlapping $5 1 2 \times 5 1 2 \times 4 \mathrm { R A W }$ patches, corresponding to $1 0 2 4 \times 1 0 2 4 \times 3$ RGB regions, for training.

Three-Level Degradation Protocol. Our benchmark includes test data at three degradation levels, each evaluating a specific restoration capability (see Table 1):

$$
\mathbf { y } = ( \mathbf { x } \otimes \mathbf { k } ) + \mathbf { n }\tag{2}
$$

• Level 1, Denoising: Noise only: $\mathbf { y } = \mathbf { x } + \mathbf { n }$ , where n is heteroscedastic Gaussian or real sensor noise modeled as $n \sim \mathcal { N } ( 0 , \alpha x + \sigma _ { r } ^ { 2 } )$ as previous works [2, 13, 35]

• Level 2, Deblurring: Blur only: $\mathbf { y } = \mathbf { x } \otimes \mathbf { k }$ , where k represents a point-spread function (PSF) that models different types of defocus or motion blur [14, 25, 26].

• Level 3, Joint: Blur and noise: $\mathbf { y } = ( \mathbf { x } \otimes \mathbf { k } ) + \mathbf { n }$ , representing the most challenging real-world scenario [12, 34].

During training, we randomize the degradation parameters, apply blur with probability 0.5, and sample the noise types with equal probability. We follow the degradation pipeline of prior works such as RawIR [5, 10], using diverse noise profiles, PSFs, and blur kernels.

![](images/0f481bb9d23049dd94495b5e1d3bac662affc7d8a75f14c7814fd997efee08ba.jpg)  
(a) Original Sample 1

![](images/5870766be39da79f2e8ad078646dd589449442530cc81816548b7f7558b150f7.jpg)  
(b.1) Shot & read noise

![](images/c1da9592a57ced65811caa178d3dee202272a129e3a1fa819214e147cc2127c5.jpg)  
(b.2) Gaussian noise

![](images/4b3c8669e2271e494149054732108f102872417a468193af8b7cc0bcc529cbb8.jpg)

![](images/0e8fd308ac02b1aa18e01066e231bb35ba68ac14a95a074d9be9a8945ed1f368.jpg)

![](images/f633b4e611ad7cbdf100ee7e7368efc28462f7a9d5eff096dbeac9b095c16161.jpg)

(c.1) Linear Motion Blur  
![](images/4d9a0d238d1a81bf2e7333e2b6a99f361c92a99360110a2b454b46050f2e0b6e.jpg)

![](images/601cf41b73566c4799340ef686dbfc08378341037933abe3b7a80b9bab66afd7.jpg)

(c.3) Linear Motion Blur  
![](images/521810a22ff40f2429d6b1ce3ca4a9da74d2532f7895766c0e2d5e3852ae56a6.jpg)

(d.1) Complex Motion Blur  
(d.2) Complex Motion Blur  
![](images/377fe496b9043c46c517cf0418d266a549608eca9674fe825c99399e7e6a5b95.jpg)

![](images/acc8c0c3f056f60c4fb17390e925d04e2583a1aac1289e23fbeb2e7807a07814.jpg)

(d.3) Complex Motion Blur  
![](images/da80e57ec8ac3d26d9326b1365ad4f45f3fe9d43bf969628ab3e431c535b9a7b.jpg)  
(f.2) Linear Motion Blur

(e) Original Sample 2  
(f.1) Linear Motion Blur  
![](images/f345de7f66a3fd48843787acd461625c59cae965b425a9414f97a2239c141aa3.jpg)  
(g.1) Complex Motion Blur

![](images/205dc5619d08e29e009f787bd6599e4cc458eef0867d0d3741664506804506c4.jpg)  
(g.2) Complex Motion Blur

![](images/fb5f0b946a7e64b05cda70f2f20692417a3f6f5d2927a506a5ea9840c9b2357a.jpg)  
(g.3) Complex Motion Blur  
Figure 3: Examples of synthetic degradations. We apply diverse blur kernels and PSFs, visualized in the upper-right corner of each image, to approximate common blur patterns. Groups (a) and (b) illustrate a clean image and its noisy variants. Groups (c) and (f) show anisotropic Gaussian-like blur, whereas groups (d) and (g) show more complex motion blur. The RAW images are visualized after ISP processing.

![](images/4e83b9ca3b79254e41445fc3ae00a9cec8f54ed0798e2a2acdda3e45928cec16.jpg)  
ISP v1  
ISP v2  
Figure 4: Two neural ISP architectures used as fixed pipeline proxies. Left: a winning architecture from the Learned Smartphone ISP Challenge [16]. Right: a winning architecture from the Mobile AI 2025 challenge [17].

<table><tr><td rowspan=1 colspan=1>Deg. Level</td><td rowspan=1 colspan=1>ISP</td><td rowspan=1 colspan=1>Vivo X90</td><td rowspan=1 colspan=1>Google Pixel</td><td rowspan=1 colspan=1>Samsung S9</td><td rowspan=1 colspan=1>iPhone XS</td></tr><tr><td rowspan=2 colspan=1>Clean</td><td rowspan=2 colspan=1>ISP v1ISP v2</td><td rowspan=1 colspan=1>22.60 / 0.90</td><td rowspan=1 colspan=1>22.31 / 0.88</td><td rowspan=1 colspan=1>29.67 / 0.96</td><td rowspan=1 colspan=1>17.56 / 0.81</td></tr><tr><td rowspan=1 colspan=1>18.97 / 0.81</td><td rowspan=1 colspan=1>19.75 / 0.80</td><td rowspan=1 colspan=1>27.95 / 0.94</td><td rowspan=1 colspan=1>17.58 / 0.77</td></tr><tr><td rowspan=2 colspan=1>Level 1</td><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>21.52 / 0.77</td><td rowspan=1 colspan=1>21.04 / 0.75</td><td rowspan=1 colspan=1>26.31 / 0.83</td><td rowspan=1 colspan=1>17.08 / 0.66</td></tr><tr><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>21.26 / 0.82</td><td rowspan=1 colspan=1>19.52 / 0.77</td><td rowspan=1 colspan=1>25.72 / 0.84</td><td rowspan=1 colspan=1>17.45 / 0.73</td></tr><tr><td rowspan=2 colspan=1>Level 2</td><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>20.90 / 0.79</td><td rowspan=1 colspan=1>19.17 / 0.67</td><td rowspan=1 colspan=1>22.48 / 0.81</td><td rowspan=1 colspan=1>16.01 / 0.67</td></tr><tr><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>20.10 / 0.79</td><td rowspan=1 colspan=1>17.83 / 0.64</td><td rowspan=1 colspan=1>22.08 / 0.81</td><td rowspan=1 colspan=1>17.11 / 0.70</td></tr><tr><td rowspan=2 colspan=1>Level 3</td><td rowspan=2 colspan=1>ISP v1ISP v2</td><td rowspan=2 colspan=1>19.37 / 0.6319.43 / 0.70</td><td rowspan=1 colspan=1>17.67 / 0.49</td><td rowspan=1 colspan=1>22.31 / 0.73</td><td rowspan=1 colspan=1>15.85 / 0.55</td></tr><tr><td rowspan=1 colspan=1>17.58 / 0.57</td><td rowspan=1 colspan=1>21.86 / 0.74</td><td rowspan=1 colspan=1>16.94 / 0.66</td></tr></table>

Table 1: Evaluation of neural ISPs under different degradation levels. No restoration model is used. We report PSNR/SSIM metrics in the RGB domain. Best results are shown in bold; the same convention is used in the following tables.

## 3.2 Neural ISP Models

The goal of this work is to improve the robustness of an Image Signal Processor using a restoration network. Designing and training neural ISPs is a separate research problem [16, 17]; here, the restoration models are treated as modular blocks placed before or after the ISP.

Because real smartphone ISPs are proprietary, we use learned ISP models as controlled proxies for generating and evaluating RGB outputs. Once trained, these proxy ISPs remain fixed in every restoration experiment.

We train two neural ISP architectures from the MAI Learned ISP Challenges [16, 17] to render sRGB images from RAW inputs. Figure 4 illustrates both architectures. Each ISP is trained for a specific camera sensor using curated, aligned RAW–sRGB pairs, where the sRGB targets are produced by the corresponding phone pipeline.

Table 1 shows that the neural ISPs are sensitive to degraded RAW inputs. On Vivo X90, for example, ISP v1 reaches 22.60 dB on clean inputs. Its performance falls to 21.52 dB with noise, 20.90 dB with blur, and 19.37 dB with joint noise and blur, corresponding to drops of 1.08, 1.70, and 3.23 dB, respectively.

## 4 Experimental Results

Throughout our experiments, the neural ISP is fixed: no restoration method updates it or receives gradients through it. Strategy A restores the RAW input before this ISP, and Strategy

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>UNet [29]</td><td rowspan=1 colspan=1>NAFNet [0]</td><td rowspan=1 colspan=1>PMRID[B2]</td><td rowspan=1 colspan=1>MOFA []</td><td rowspan=1 colspan=1>MFDNet [[8]</td></tr><tr><td rowspan=1 colspan=1>PSNRSSIM</td><td rowspan=1 colspan=1>37.440.962</td><td rowspan=1 colspan=1>39.700.972</td><td rowspan=1 colspan=1>38.430.965</td><td rowspan=1 colspan=1>38.710.966</td><td rowspan=1 colspan=1>37.100.957</td></tr><tr><td rowspan=1 colspan=1>Params. (M)MACs (G)</td><td rowspan=1 colspan=1>0.2662.23</td><td rowspan=1 colspan=1>1.1303.99</td><td rowspan=1 colspan=1>1.0321.21</td><td rowspan=1 colspan=1>0.9711.14</td><td rowspan=1 colspan=1>2.1796.88</td></tr></table>

Table 2: Benchmark of pre-ISP RAW restoration (RAWRes) models.

B restores its RGB output. We use ISP-aware (“target-ISP-trained”) only when the restoration training pairs contain outputs from the same fixed ISP used at evaluation time.

## 4.1 RAW Restoration Prior

To benchmark Strategy A, RAW-domain restoration, we consider architectures ranging from foundational baselines to recent efficient restoration models [3, 4, 18, 24, 29, 32]. We include UNet [29], which is widely used in image restoration [2, 7, 38], and PMRID [32], which targets sensor-specific RAW denoising.

We also train MFDNet [18], NAFNet [3], and MOFA [4]. To keep NAFNet comparable in size to the other baselines, we use a reduced version with three encoder blocks, one middle block, three decoder blocks, and simplified skip connections. All models are trained on degraded–clean RAW pairs generated by the controlled degradation pipeline and organized by camera sensor. They receive normalized four-channel RGGB inputs corrected for black and white levels and produce restored RAW images, which are then passed to the fixed ISPs.

Table 2 demonstrates that NAFNet and MOFA outperform other methods while maintaining reasonable parameter counts and computational complexity, so we use these two methods in the subsequent evaluation.

## 4.2 RGB Restoration: Pre-trained Models

To benchmark generic Strategy B (post-ISP restoration), we employ state-of-the-art all-inone RGB restoration models that can handle multiple types of degradation simultaneously.

We select three representative methods: AirNet [21]; PromptIR [27], which is an all-inone model based on Restormer [36]; and MiOIR [20]. These models receive degraded 8-bit sRGB images produced by the fixed ISP and return restored sRGB outputs. At this stage, ISP operations have transformed the original RAW-domain degradation characteristics.

These RGB models were pre-trained on large-scale RGB datasets with conventional synthetic degradations, rather than on outputs from our target ISPs. Despite their larger model sizes (15–30M parameters), they underperform the RAW models trained on our benchmark degradations across the evaluated levels (Table 4). This is a practical comparison of generic pre-trained RGB models against benchmark-trained RAW models, rather than a trainingmatched isolation of the restoration domain.

These results indicate a gap between ISP-transformed degradations and the conventional RGB degradations used to pre-train these models. We therefore next evaluate training on a specific target pipeline.

![](images/24eb51f387febd8da0f5d5a85f08e9c7df9967db7755f917c28de74679699363.jpg)

<table><tr><td>Deg. Level</td><td>Res. Model</td><td>ISP</td><td>Vivo X90</td><td>Google Pixel</td><td>Samsung S9</td><td>iPhone XS</td></tr><tr><td rowspan="4">Level 1</td><td>MOFA</td><td>ISP v1</td><td>22.06 / 0.85</td><td>21.40 / 0.80</td><td>27.70/ 0.91</td><td>16.11 /0.71</td></tr><tr><td>NAFNet</td><td>ISP v1</td><td>22.00 / 0.85</td><td>21.48 / 0.81</td><td>27.93 / 0.92</td><td>16.16 / 0.72</td></tr><tr><td>MOFA</td><td>ISP v2</td><td>21.39 /0.85</td><td>19.43 / 0.76</td><td>26.57 / 0.90</td><td>17.42 / 0.74</td></tr><tr><td>NAFNet</td><td>ISP v2</td><td>21.35 / 0.85</td><td>19.48 / 0.77</td><td>26.80 / 0.91</td><td>17.30/ 0.74</td></tr><tr><td rowspan="4">Level 2</td><td>MOFA</td><td>ISP v1</td><td>21.54 / 0.81</td><td>19.71 / 0.69</td><td>24.58 / 0.85</td><td>15.75 / 0.67</td></tr><tr><td>NAFNet</td><td>ISP v1</td><td>21.93 / 0.83</td><td>20.21 / 0.72</td><td>25.99 / 0.88</td><td>15.78 / 0.68</td></tr><tr><td>MOFA</td><td>ISP v2</td><td>20.77 / 0.81</td><td>18.42 / 0.67</td><td>24.03 / 0.84</td><td>17.14/0.71</td></tr><tr><td>NAFNet</td><td>ISP v2</td><td>21.12/0.83</td><td>18.81 / 0.70</td><td>25.25 / 0.87</td><td>17.18 / 0.71</td></tr><tr><td rowspan="4">Level 3</td><td>MOFA</td><td>ISP v1</td><td>20.76 / 0.78</td><td>19.12 / 0.65</td><td>24.43 / 0.83</td><td>15.60 / 0.65</td></tr><tr><td>NAFNet</td><td>ISP v1</td><td>21.10 / 0.80</td><td>19.77 / 0.69</td><td>25.41 / 0.85</td><td>15.66 / 0.66</td></tr><tr><td>MOFA</td><td>ISP v2</td><td>20.32 / 0.79</td><td>18.26 / 0.64</td><td>23.76 / 0.83</td><td>17.00 / 0.69</td></tr><tr><td>NAFNet</td><td>ISP v2</td><td>20.55 / 0.80</td><td>18.57 / 0.67</td><td>24.58 / 0.85</td><td>17.01 / 0.70</td></tr></table>

Table 3: Benchmark of Strategy A: RAW restoration followed by a fixed ISP. The restoration models are trained on degraded–clean RAW pairs without using downstream ISP outputs during training. Compared with Table 1, they improve robustness across the evaluated degradation levels.

## 4.3 Sensor-Specific Models

We additionally train comparable NAFNet [3] models on the Vivo X90 data. The sensorspecific Strategy A model learns degraded-to-clean RAW restoration and is then followed by the fixed ISP. The ISP-aware Strategy B model learns from RGB pairs drawn from the output distribution of that same fixed ISP; it does not access ISP parameters or gradients. Figure 6 shows that the latter achieves higher RGB PSNR. Because the two losses operate in different domains, this experiment should not be interpreted as isolating domain alone. Instead, it demonstrates the benefit of training directly on the target output distribution.

## 4.4 Quantitative Results

In the quantitative study, we use NAFNet [3] and MOFA [4] as our pre-ISP models–RAWRes– for Strategy A. We consider three generic all-in-one RGB restoration models (MiOIR [20], AirNet [21], and PromptIR [27]) for Strategy B, together with the two fixed neural ISPs.

The benchmark results in Tables 3 and 4 show that the benchmark-trained RAW models outperform the generic pre-trained RGB models. Figure 5 summarizes these tables by averaging PSNR across the four device groups and two fixed ISPs for each degradation level. This comparison demonstrates the limitations of transferring conventional RGB restoration directly to ISP outputs; it does not establish an intrinsic advantage of the RAW domain.

For example, at Level 2 on Samsung S9, NAFNet+ISPv1 reaches 25.99 dB, compared with 22.49 dB for ISPv1+MiOIR, a gain of 3.50 dB. The performance gap against these generic RGB baselines generally widens for the joint degradation at Level 3, indicating that their training distribution does not transfer well to the more complex ISP-transformed artifacts.

![](images/bee5c7e188f2f9182d87b6660d5960c7b36f6cbff181e617124410207643d379.jpg)

<table><tr><td rowspan=1 colspan=1>Deg. Level</td><td rowspan=1 colspan=1>Res. Model</td><td rowspan=1 colspan=1>ISP</td><td rowspan=1 colspan=1>Vivo X90</td><td rowspan=1 colspan=1>Google Pixel</td><td rowspan=1 colspan=6>Samsung S9</td><td rowspan=1 colspan=1>iPhone XS</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>PromptIRAirNet</td><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>20.46 / 0.75</td><td rowspan=1 colspan=1>20.58 / 0.74</td><td rowspan=1 colspan=6>25.36 / 0.82</td><td rowspan=1 colspan=1>16.63 / 0.64</td></tr><tr><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>20.87 / 0.76</td><td rowspan=1 colspan=1>20.90 / 0.75</td><td rowspan=1 colspan=6>25.65 / 0.82</td><td rowspan=1 colspan=1>16.55 / 0.64</td></tr><tr><td rowspan=4 colspan=1>Level 1</td><td rowspan=4 colspan=1>MiOIRPromptIRAirÑetMiOIR</td><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>21.55 / 0.78</td><td rowspan=1 colspan=1>20.09 / 0.67</td><td rowspan=1 colspan=6>26.15 / 0.84</td><td rowspan=1 colspan=1>17.14 / 0.67</td></tr><tr><td rowspan=2 colspan=1>ISP v2ISP v2</td><td rowspan=1 colspan=1>20.30 / 0.80</td><td rowspan=1 colspan=1>19.47 / 0.77</td><td rowspan=2 colspan=6>23.77 / 0.8025.03 / 0.83</td><td rowspan=2 colspan=1>15.16 / 0.6816.22 / 0.70</td></tr><tr><td rowspan=1 colspan=1>20.52 / 0.81</td><td rowspan=1 colspan=1>19.32 / 0.77</td></tr><tr><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>20.57 / 0.72</td><td rowspan=1 colspan=1>19.38 / 0.72</td><td rowspan=1 colspan=6>25.92 / 0.85</td><td rowspan=1 colspan=1>16.52 / 0.57</td></tr><tr><td rowspan=6 colspan=1>Level 2</td><td rowspan=3 colspan=1>PromptIRAirNetMiOIR</td><td rowspan=2 colspan=1>ISP v1ISP v1</td><td rowspan=2 colspan=1>19.85 / 0.7719.25 / 0.75</td><td rowspan=1 colspan=1>18.81 / 0.67</td><td rowspan=1 colspan=6>20.71 / 0.75</td><td rowspan=1 colspan=1>14.28 / 0.62</td></tr><tr><td rowspan=1 colspan=1>18.09 / 0.64</td><td rowspan=1 colspan=5>19.95 / 0.71</td><td rowspan=2 colspan=5>22.49 / 0.81</td><td rowspan=2 colspan=1>12.87 / 0.5916.11 /0.67</td></tr><tr><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>20.86 / 0.79</td><td rowspan=1 colspan=1>19.22 / 0.67</td><td></td></tr><tr><td rowspan=3 colspan=1>PromptIRAirNetMiOIR</td><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>19.62 / 0.78</td><td rowspan=1 colspan=1>17.75 / 0.64</td><td rowspan=1 colspan=6>21.34 / 0.79</td><td rowspan=1 colspan=1>13.98 / 0.64</td></tr><tr><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>17.87 / 0.73</td><td rowspan=1 colspan=1>17.05 / 0.62</td><td rowspan=1 colspan=6>19.62 / 0.73</td><td rowspan=1 colspan=1>14.06 / 0.64</td></tr><tr><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>20.07 / 0.79</td><td rowspan=1 colspan=1>17.24 / 0.63</td><td rowspan=1 colspan=6>22.10 / 0.81</td><td rowspan=1 colspan=1>17.05 / 0.70</td></tr><tr><td rowspan=6 colspan=1>Level 3</td><td rowspan=3 colspan=1>PromptIRAirÑetMiOIR</td><td rowspan=1 colspan=1>ISP v1</td><td rowspan=1 colspan=1>18.70 / 0.61</td><td rowspan=1 colspan=1>17.36 / 0.48</td><td rowspan=1 colspan=6>21.51 /0.71</td><td rowspan=1 colspan=1>15.60 / 0.53</td></tr><tr><td rowspan=2 colspan=1>ISP v1ISP v1</td><td rowspan=2 colspan=1>18.81 / 0.6219.41 / 0.65</td><td rowspan=1 colspan=1>17.48 / 0.48</td><td rowspan=1 colspan=6>21.45 / 0.72</td><td rowspan=1 colspan=1>15.36 / 0.53</td></tr><tr><td rowspan=1 colspan=1>17.58 / 0.47</td><td rowspan=1 colspan=3>21.83 / 0</td><td rowspan=1 colspan=2>10.</td><td rowspan=1 colspan=2>.74</td><td></td><td rowspan=1 colspan=1>15.35 / 0.55</td></tr><tr><td rowspan=2 colspan=1>PromptIRAirNet</td><td rowspan=2 colspan=1>ISP v2ISP v2</td><td rowspan=2 colspan=1>18.58 / 0.6818.81 / 0.69</td><td rowspan=1 colspan=1>17.49 / 0.57</td><td rowspan=1 colspan=2>20.82</td><td rowspan=2 colspan=5>20.82 / 0.7021.13 /0.73</td><td rowspan=1 colspan=1>70</td></tr><tr><td rowspan=1 colspan=1>17.34 / 0.56</td><td></td><td rowspan=1 colspan=1>15.61 / 0.63</td></tr><tr><td rowspan=1 colspan=1>MiOIR</td><td rowspan=1 colspan=1>ISP v2</td><td rowspan=1 colspan=1>18.97 / 0.61</td><td rowspan=1 colspan=1>17.35 / 0.52</td><td rowspan=1 colspan=6>21.94 / 0.75</td><td rowspan=1 colspan=1>16.45 / 0.57</td></tr></table>

Table 4: Benchmark of generic Strategy B: a fixed ISP followed by RGB restoration models pre-trained on conventional RGB degradations. Some models improve the degraded ISP outputs in Table 1, but they remain below the benchmark-trained Strategy A models in Table 3. This comparison is not training-matched.

Comparing Table 1 with Table 3, we observe that pre-ISP RAW restoration brings the ISP performance closer to the clean baseline. For example, on Vivo X90 at Level 1, the degraded ISP drops from 22.60 dB to 21.52 dB, yet NAFNet+ISPv1 recovers to 22.00 dB–only 0.60 dB below the clean performance. This shows that RAW restoration effectively “shields” the ISP from input degradations.

Furthermore, the compact RAW models (approximately 1–2 million parameters) obtain these improvements with lower model capacity than the pre-trained RGB baselines (15–30 million parameters), which is favorable for mobile deployment.

Target-ISP Training Figure 6 shows a different regime of Strategy B: the post-ISP model is trained on the target ISP output distribution. With comparable model size, this model exceeds the sensor-specific Strategy A result across all three degradation levels. It can learn artifacts present in the final RGB output while the ISP remains fixed and is treated as a black box. Together with the generic-model comparison, this reversal supports our main conclusion that training-distribution alignment, rather than domain alone, determines the observed ranking.

![](images/c9af45dff0cb63fbceb511bfb848cb20b94019dcda0396b0bb48fca6a2b63131.jpg)  
Figure 5: Generic-model comparison, averaged over four device groups and two fixed ISPs. Strategy A (blue) is trained on clean-noisy RAW images with calibrated degradations, whereas generic Strategy B (orange) uses RGB models pre-trained for all-in-one restoration. This practical transfer comparison is not training-matched.

![](images/44a05fbe1026da34c719fb7d62ff8f31944c68bbd4e694cbc2baea424588a0cb.jpg)  
Figure 6: Target-distribution comparison on Vivo X90 using fixed ISP v1. Blue curves are Strategy A RAW models evaluated after the ISP; the green curve is Strategy B trained directly on RGB outputs from that ISP. Sensor-specific RAW training changes performance only slightly, whereas ISP-aware RGB restoration yields the largest gain.

![](images/888a460b21420c59b7b7cbe36fe3ac5e7b13028bf61eca992842b2219244d4a1.jpg)  
Figure 7: RAW–RGB qualitative comparison on the synthetic test set. We show clean and degraded RAW inputs together with RGB restoration results. The generic MiOIR model [20] receives degraded RGB produced by the fixed ISP, whereas our Strategy A model restores RAW before the same ISP.

## 4.5 Qualitative Results

In the qualitative studies, we use NAFNet [3] as the backbone of the RAWRes pre-ISP block because it is the strongest RAW baseline in Table 2.

Figures 7 and 8 compare our trained models with the generic RGB restoration baseline MiOIR [20], which is generally the strongest pre-trained RGB method in Table 4. We use the same fixed ISP v1 for these comparisons.

The visual examples show that both our pre-ISP and post-ISP-aware models reduce noise and motion blur more effectively than MiOIR [20] on the selected samples.

Moreover, as discussed in Sec. 4.4, the target-ISP-trained post-ISP model can restore artifacts present after ISP processing. This result is specific to training on the target output distribution and should not be generalized to arbitrary RGB restoration models, as illustrated by the weaker generic baselines.

Real-World Evaluation We apply our pipeline to real-world RAW images captured using a Vivo X90 phone in challenging conditions. We obtain the sRGB images directly from the smartphone’s ISP. In Figure 9, the original phone output exhibits strong sensor noise and blur. Our ISP-aware RGB restoration model produces images with less visible noise and clearer details. We also compare with RealESRGAN [31], which produces oversmoothed textures in these examples. This illustrates a potential risk of deploying generative restoration models in photography pipelines.

![](images/1ae7bd45495ba937a215f9883d1b8ab4e91f7f8b1de66c3943676f1a0cb1a4f7.jpg)  
Figure 8: Qualitative comparison on the synthetic test set. MiOIR [20] receives RGB images produced by passing degraded RAW data through the fixed ISP. Both our pre-ISP RAW model and our target-ISP-trained post-ISP RGB model reduce visible blur and noise. In these examples, target-ISP-trained RGB restoration recovers more color and detail.

![](images/c7722552c01a8cb9861d5180f6c8962ea47618d159d3f2de47b78b0b989288ee.jpg)  
Our Strategy B: ISP-aware RGB restoration method  
Figure 9: Real-world qualitative results. We use the same RAW images and camera pipeline for all comparisons. Our target-ISP-trained RGB restoration reduces visible noise while preserving fine details in these examples.

## 4.6 Limitations and Future Work

First, although the RAW degradation pipeline aims to approximate real-world degradations, synthetic noise and blur cannot capture every property of smartphone sensors and lenses. Together with the 47-image test split, this limits broad claims about generalization.

Second, the generic comparison is not training-matched, and the sensor-specific RAW and RGB models optimize losses in different domains. The reported reversal therefore supports the importance of target-distribution alignment, but it does not determine the intrinsic performance ceiling of either domain.

Third, we do not investigate Color Correction Matrix-aware or target-ISP-aware RAW objectives. Injecting CCM priors or optimizing RAW restoration through a suitable approximation of the downstream pipeline could improve Strategy A and change the observed ranking. We consider this an important direction for future work rather than a conclusion supported by the current benchmark.

Finally, larger real-world test sets and additional modern cameras are needed to assess generalization beyond the evaluated sensors and pipeline proxies.

## 5 Conclusion

We benchmark deep restoration before and after a fixed image signal processor using four smartphone device groups, two learned ISP proxies, and representative RAW and RGB restoration methods. Benchmark-trained RAW restoration outperforms generic pre-trained RGB restoration in most evaluated configurations, whereas target-ISP-trained RGB restoration performs best in the Vivo X90 pipeline-specific experiment. These results do not identify a universally preferable restoration domain. Instead, they show that matching the restoration training distribution to the distribution induced at the model’s insertion point in the target imaging pipeline is central to robust final-image quality.

## Acknowledgments

This work was partly supported by the Alexander von Humboldt Foundation.

## References

[1] Abdelrahman Abdelhamed, Stephen Lin, and Michael S Brown. A high-quality de noising dataset for smartphone cameras. In CVPR, pages 1692–1700, 2018.

[2] Tim Brooks, Ben Mildenhall, Tianfan Xue, Jiawen Chen, Dillon Sharlet, and Jonathan T Barron. Unprocessing images for learned raw denoising. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11036–11045, 2019.

[3] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. In European conference on computer vision, pages 17–33. Springer, 2022.

[4] Xiangyu Chen, Ruiwen Zhen, Shuai Li, Xiaotian Li, and Guanghui Wang. Mofa: A model simplification roadmap for image restoration on mobile devices. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1322–1332, 2023.

[5] Marcos Conde, Radu Timofte, Zihao Lu, Xiangyu Kong, Xiaoxia Xing, Fan Wang, Suejin Han, MinKyu Park, Tianyu Hao, Yuhong He, et al. Ntire 2025 challenge on raw image restoration and super-resolution. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1148–1171, 2025.

[6] Marcos V Conde, Steven McDonagh, Matteo Maggioni, Ales Leonardis, and Eduardo Pérez-Pellitero. Model-based image signal processors via learnable dictionaries. In

Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 481– 489, 2022.

[7] Marcos V Conde, Florin Vasluianu, Javier Vazquez-Corral, and Radu Timofte. Perceptual image enhancement for smartphone real-time applications. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1848–1858, 2023.

[8] Marcos V Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In European Conference on Computer Vision, pages 1–21. Springer, 2024.

[9] Marcos V Conde, Florin Vasluianu, and Radu Timofte. Bsraw: Improving blind raw image super-resolution. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 8500–8510, 2024.

[10] Marcos V Conde, Florin Vasluianu, and Radu Timofte. Toward efficient deep blind raw image restoration. In 2024 IEEE International Conference on Image Processing (ICIP), pages 1725–1731. IEEE, 2024.

[11] Mauricio Delbracio, Damien Kelly, Michael S Brown, and Peyman Milanfar. Mobile computational photography: A tour. Annual review of vision science, 7(1):571–604, 2021.

[12] Michael Elad and Arie Feuer. Restoration of a single superresolution image from several blurred, noisy, and undersampled measured images. IEEE Transactions on Image Processing, 6(12):1646–1658, 1997.

[13] Samuel W Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams, Jonathan T Barron, Florian Kainz, Jiawen Chen, and Marc Levoy. Burst photography for high dynamic range and low-light imaging on mobile cameras. ACM Transactions on Graphics (ToG), 35(6):1–12, 2016.

[14] Mahdi S Hosseini and Konstantinos N Plataniotis. Convolutional deblurring for natural imaging. IEEE Transactions on Image Processing, 29:250–264, 2019.

[15] Andrey Ignatov, Luc Van Gool, and Radu Timofte. Replacing mobile camera isp with a single deep learning model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 536–537, 2020.

[16] Andrey Ignatov, Radu Timofte, Shuai Liu, Chaoyu Feng, Furui Bai, Xiaotao Wang, Lei Lei, Ziyao Yi, Yan Xiang, Zibin Liu, et al. Learned smartphone isp on mobile gpus with deep learning, mobile ai & aim 2022 challenge: report. In European Conference on Computer Vision, pages 44–70. Springer, 2022.

[17] Andrey Ignatov, Georgii Perevozchikov, Radu Timofte, Cheng Li, Lian Liu, Jun Cao, Heng Sun, Wu Pan, Song Wang, KeQiang Yu, et al. Learned smartphone isp on mobile gpus, mobile ai 2025 challenge: Report. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1934–1946, 2025.

[18] Yiguo Jiang, Xuhang Chen, Chi-Man Pun, Shuqiang Wang, and Wei Feng. Mfdnet: Multi-frequency deflare network for efficient nighttime flare removal. The Visual Computer, 40(11):7575–7588, 2024.

[19] Hakki Can Karaimer and Michael S Brown. A software platform for manipulating the camera imaging pipeline. In ECCV, pages 429–444, 2016.

[20] Xiangtao Kong, Chao Dong, and Lei Zhang. Towards effective multiple-in-one image restoration: A sequential and prompt learning strategy. arXiv preprint arXiv:2401.03379, 2024.

[21] Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, and Xi Peng. All-in-one image restoration for unknown corruption. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 17452–17462, 2022.

[22] Chih-Hung Liang, Yu-An Chen, Yueh-Cheng Liu, and Winston H. Hsu. RAW image deblurring. IEEE Transactions on Multimedia, 24:61–72, 2022. doi: 10.1109/TMM. 2020.3045303.

[23] Zhetong Liang, Jianrui Cai, Zisheng Cao, and Lei Zhang. Cameranet: A two-stage framework for effective camera isp learning. IEEE Transactions on Image Processing, 30:2248–2262, 2021.

[24] Zhuoqun Liu, Meiguang Jin, Ying Chen, Huaida Liu, Canqian Yang, and Hongkai Xiong. Lightweight network towards real-time image denoising on mobile devices. In 2023 IEEE International Conference on Image Processing (ICIP), pages 2270–2274. IEEE, 2023.

[25] Seungjun Nah, Tae Hyun Kim, and Kyoung Mu Lee. Deep multi-scale convolutional neural network for dynamic scene deblurring. In IEEE Conference on Computer Vision and Pattern Recognition, pages 3883–3891, 2017.

[26] Seungjun Nah, Sanghyun Son, Suyoung Lee, Radu Timofte, Kyoung Mu Lee, Liangyu Chen, Jie Zhang, Xin Lu, Xiaojie Chu, Chengpeng Chen, et al. Ntire 2021 challenge on image deblurring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 149–165, 2021.

[27] Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan. PromptIR: Prompting for all-in-one blind image restoration. In Advances in Neural Information Processing Systems, volume 36, 2023.

[28] Abhijith Punnappurath, Abdullah Abuolaim, Abdelrahman Abdelhamed, Alex Levinshtein, and Michael S Brown. Day-to-night image synthesis for training nighttime neural isps. In CVPR, pages 10769–10778, 2022.

[29] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computerassisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

[30] Eli Schwartz, Raja Giryes, and Alex M Bronstein. Deepisp: Toward learning an endto-end image processing pipeline. IEEE Transactions on Image Processing, 28(2): 912–923, 2018.

[31] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2021.

[32] Yuzhi Wang, Haibin Huang, Qin Xu, Jiaming Liu, Yiqun Liu, and Jue Wang. Practical deep raw image denoising on mobile devices. In European Conference on Computer Vision, pages 1–16. Springer, 2020.

[33] Yazhou Xing, Zian Qian, and Qifeng Chen. Invertible image signal processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6287–6296, 2021.

[34] Xiangyu Xu, Yongrui Ma, and Wenxiu Sun. Towards real scene super-resolution with raw images. In CVPR, 2019.

[35] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao. Cycleisp: Real image restoration via improved data synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2696–2705, 2020.

[36] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5728–5739, 2022.

[37] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE transactions on image processing, 26(7):3142–3155, 2017.

[38] Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6360–6376, 2021.

[39] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4791–4800, 2021.

[40] Kai Zhang, Yawei Li, Jingyun Liang, Jiezhang Cao, Yulun Zhang, Hao Tang, Deng-Ping Fan, Radu Timofte, and Luc Van Gool. Practical blind image denoising via Swin-Conv-UNet and data synthesis. Machine Intelligence Research, 20:822–836, 2023.