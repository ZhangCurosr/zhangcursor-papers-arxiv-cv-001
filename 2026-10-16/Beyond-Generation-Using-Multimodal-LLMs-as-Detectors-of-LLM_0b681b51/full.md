# Beyond Generation: Using Multimodal LLMs as Detectors of LLM-Generated Deepfake Facial Images

OLIVIA SALAMON, VIRGINIA N. L. FRANQUEIRA, and SHUJUN LI,

Institute of Cyber Security for Society (iCSS) & School of Computing, University of Kent, London and South East University Group, Canterbury, UK

Background/Aim: As image generation models become increasingly capable of producing highly realistic content, traditional forensic techniques are struggling to keep up. This paper investigates whether MLLMs can instead be used as a form of forensic analysis by identifying synthetic media. Specifically, the study aims to determine whether these models detect fake images by recognising physical inconsistencies, or whether they rely on digital artefacts such as embedded watermarks.

Methods: Six MLLMs: GPT-4o, GPT-5, GPT-5.2, Gemini 2.5, Gemini 3, and Seed 1.6, are benchmarked using a dataset of 400 images. Fifty images are authentic photographs, while the remaining 350 are generated using seven leading image models: Gemini 3, Gemini 2.5, Flux.2 Pro, GPT-Image-1, GPT-Image-1.5, Seedream 4.0, and Seedream 4.5. The evaluation is structured in three stages: single-image classification, pairwise comparison, and a focused “self-detection” experiment on Gemini. This final stage tests five image groups, including clean controls, visible logos, and images containing invisible SynthID watermarks, to explore how digital signals influence detection accuracy.

Results: Several MLLMs achieved a high detection accuracy, with GPT-5.2 reaching a peak of 97.9% in Experiment 2, the pairwise experiment, against specific generators and an overall mean pairwise accuracy of 82.5% by identifying structural or physical inconsistencies within images. However, results from the self-detection experiment indicate that the presence of invisible SynthID watermarks leads to a 10% reduction in Gemini’s detection accuracy compared to the clean control group. One possible explanation is that SynthID embedding reduces some of the visible artefacts the models rely on for detection

Conclusion: These results imply that MLLMs can detect structural artefacts without relying on watermarks, but performance varies substantially across model generations. The findings from comparison C5 also suggest that SynthID embedding may suppress some of the structural artefacts used during detection.

Additional Key Words and Phrases: Deepfake detection, Multimodal Large Language Models (MLLMs), Watermarking, SynthID

## ACM Reference Format:

Olivia Salamon, Virginia N. L. Franqueira, and Shujun Li. 2026. Beyond Generation: Using Multimodal LLMs as Detectors of LLM-Generated Deepfake Facial Images. 1, 1 (October 2026), 25 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

## 1.1 Background and Motivation

Generative tools such as OpenAI’s GPT-series and Google’s Gemini-series have become increasingly advanced in their image generation capabilities. Throughout this study, the term deepfake refers to fully synthetic facial images generated

Authors’ Contact Information: Olivia Salamon; Virginia N. L. Franqueira, V.Franqueira@kent.ac.uk; Shujun Li, S.J.Li@kent.ac.uk , Institute of Cyber Security for Society (iCSS) & School of Computing, University of Kent, London and South East University Group, Canterbury, UK from text prompts using the seven generative models: Gemini 3, Gemini 2.5, Flux.2 Pro, GPT-Image-1, GPT-Image-1.5, Seedream 4.0, and Seedream 4.5. This study focuses specifically on fully synthetic deepfakes, images generated entirely from text prompts, rather than face-swap or identity-manipulation techniques, which involve modifying existing footage. While generative models are increasingly capable of both approaches, this study isolates text-to-image synthesis as the primary forensic threat. These models now produce images that are dificult to distinguish from real photographs. Traditional forensic methods are becoming inadequate for this task; methods that once relied on pixel-level inconsistencies [36] are becoming obsolete. Researchers have increasingly turned to MLLMs for deepfake detection, through their abilities to “justify” their detection classifications. Rather than searching for pixel artefacts, research has shown that MLLMs can identify semantic inconsistencies within images such as irregular lighting or inconsistent background geometry [19]. This study evaluates whether recent models such as GPT-5.2 demonstrate similar or improved capability.

While MLLM decision justification represents a promising step for Explainable AI (XAI), these models remain unpredictable in their classification behaviour. Lapuschkin et al. [21] and Maheronnaghsh [29] highlight the “Clever Hans” efect, when a model achieves high accuracy by exploiting shortcuts within the image rather than a genuine forensic understanding. These shortcuts often stem from the generator’s traces such as visible logos or pixel-embedded watermarks such as Google’s SynthID.

This study assesses the reliability of MLLMs as forensic detectors. By isolating and removing generator-related traces within Experiment 3, and comparing detection performance across conditions in Experiments 1 and 2, this study examines the extent to which MLLM accuracy may depend on structural image artefacts or digital shortcuts.

Central to this research is the audit of MLLM decision-making. It remains unclear whether MLLMs genuinely reason about image structure or whether high accuracy reflects exploitation of digital shortcuts, such as metadata cues or pixel-embedded watermarks, rather than genuine forensic analysis [18, 29]. Current evidence is not yet suficient to answer this question, motivating the controlled experimental design used in this study

## 1.2 Research Hypotheses

To achieve this, the following four Hypotheses have been established:

• H1 (Self- and Cross-Model Detection): Generative ecosystems may have an advantage in detecting their own generated images, however, detection performance against images from external models is expected to vary significantly across architectures [18].

• H2 (Pairwise Image Advantage): Detection accuracy is theoretically higher in a pairwise experiment, where the MLLM is presented with one real and one fake image simultaneously, than in a single-image experiment [3, 19].

• H3 (Efects of Watermarks): The presence or absence of watermark signals will have a measurable efect on detection accuracy [16, 26].

• H4 (Justification Reliability): MLLM natural-language justifications may not accurately reflect the reasoning process, raising concerns about whether MLLM justifications are genuine structural analysis [29].

## 1.3 Research Objectives

These hypotheses are further reinforced within the following objectives evaluated throughout the study:

Manuscript submitted to ACM

• Dataset Construction: Building a custom 400-image corpus, 50 authentic photographs from the Flickr-Faces-HQ dataset [20] in conjunction with 350 images from seven top-ranked generators, including the Gemini Ecosystem, GPT family, Flux and SeeDream.

• Detector Performance: Evaluating six MLLM detectors across single-image and pairwise classification experiments.

• Gemini Watermark Experiments: Performing an “ablation” experiment on the Gemini generated images by removing their visible logos and SynthID embedded watermarks to measure how this afects detection accuracy.

• Model Justification Audit: Manually validating the natural-language justifications provided by the model to determine if their justifications are reasonable or if they are “hallucinating” artefacts within the image.

## 1.4 Report Structure

• Section 2: Literature Review - Identifies the current gaps discovered around how MLLMs justify their decisions.

• Section 3: Methodology - Details the technical framework established for the study. This includes the development of the image dataset used and the experiment designs for the single-image analysis, pairwise-image analysis and ablation experiments.

• Section 4: Results - Presents the core findings of the study and comparative performance of the chosen MLLM detectors.

• Section 5: Discussion - Discusses the key findings, positions them within the literature, and addresses limitations and future directions.

• Section 6: Conclusion - Final closing statement of the study.

## 2 Literature Review

Generative AI ecosystems such as Gemini and the GPT family can now produce media that is genuinely dificult to detect. Successful detection now requires techniques such as the use of MLLMs as opposed to conventional pixel-based algorithms.

## 2.1 Evolution of Generative AI

Within the early days of generative media, it relied heavily on deep learning architectures such as Generative Adversarial Networks (GANs) [10] which produced media that was easily detectable both through the human eye and AI. The introduction of pre-trained language models such as BERT [7] set the foundation for advanced language models. Thi was then further developed through Latent Difusion Models (LDMs) [35] which produced high-resolution images through a denoising technique and Large Vision Models (LVMs), which extended this to visual understanding. This ha led to the formation of Multimodal Large Language Models (MLLMs) which are capable of processing both text and image inputs.

## 2.2 Historical Overview of Deepfakes

Deepfakes have evolved through several generations, the first being video-based face-swap techniques [36] which produced visible artefacts detectable by pixel-level analysis. The second generation improved realism through GANbased models to generate synthetic facial images. The third generation, introduced by [35], generated high-fidelity images from textual prompts with a significant decline in detectable inconsistencies, enabled by LDMs. The most recent generation involves MLLMs generating images from multimodal prompts, further reducing image detectability [34].

This study focuses on the third and fourth generations of synthetic media however, deepfake videos remain a significant concern.

## 2.3 Automated Deepfake Detection

As generative quality has significantly improved, detection strategies have shifted to high-level semantic evaluation. MLLMs have increasingly been applied to zero-shot deepfake classification. Jia et al. [19] demonstrated that models within the GPT and Gemini families could identify artificial media by identifying inconsistencies within the image, rather than depending on pixel-level artefacts. These models may detect issues such as inconsistent lighting or unrealistic object structures.

Although the idea of using MLLMs as deepfake detectors initially seemed promising, newer studies have produced contradictory results. Large-scale testing has revealed that although they show some baseline detection capability, models experience a large drop in accuracy when tested against high-quality, real, difusion images [3, 32]. In addition, Jia et al. [19] argued that the current MLLMs are not yet reliable enough for forensic detection, demonstrating that even a slight image alteration can drastically alter their judgement. These inconsistent findings suggest that the models tested were likely looking for visual shortcuts rather than genuine forensic reasoning.

Specifically, no recent study has used a controlled ablation design to isolate whether the accuracy of MLLM detection reflects genuine structural reasoning or the exploitation of watermarks, which this study directly addresses within Experiment 3.

## 2.4 Watermarks and Their Importance

Due to the increased challenges created by modern generative models, digital watermarks are increasingly promoted as a practical detection solution. Traditional image metadata can be easily manipulated by the user. Therefore, image watermarking techniques work through embedding signals directly into the image architecture [8].

Google DeepMind’s SynthID is a leading example of this technology. Gowal et al.[13] outlined how the tool embeds an invisible digital signature directly into the image’s frequency domain, theoretically overcoming the traditional trade-of between visual quality and signal strength ensuring the watermark persists even through common image transformations such as cropping or compression. According to DeepMind’s developers, this ensured that even if an image is altered, the watermark remained detectable [5].

However, recent security audits have shown that embedded signals are far more fragile than the industry seems to claim. Liu et al. [26] found that invisible watermarks, such as systems similar to SynthID, could still be targeted by attackers through a “controllable regeneration” attack. This method removes hidden patterns without compromising the quality of the image. Han et al. [16] explained this weakness, finding that even basic image alterations can cause SynthID signals to deteriorate. Relying on embedded watermarks alone therefore creates a vulnerability to the Trace Removal Attack (TRA) described by Liu et al.[25], enabling attackers to generate seemingly real fakes that bypass modern detection systems.

## 2.5 Explainability and Justification Quality

MLLM usage has noticeably increased due to their Explainable AI (XAI) capabilities, providing natural language justifications alongside their classifications [23]. Knowing why a classification was made is arguably as important as the classification itself, especially in forensic scenarios [31].

Manuscript submitted to ACM

However, analysing these explanations exposes a major flaw: the “Clever Hans” efect. Maheronnaghsh and Alvanagh [29] argued that MLLMs often achieve high accuracy scores not because of their capabilities to recognise facial anomalies but instead “cheat” through finding visual shortcuts within the testing data to find the correct answer. Ji et al. [18] demonstrated self-detection bias, explained that while these models had higher success in recognising fakes built through their own architecture, they struggle against previously unseen difusion models and rely on irrelevant visual traces to make a guess

Beyond the models taking shortcuts, a second problem has emerged: LLM hallucinations. When asked to justify a decision, they can often fabricate supporting evidence. Both Sun et al. [37] and Li et al. [22] demonstrated that models will confidently output factually incorrect descriptions. An example seen is the model flagging a non-existent distortion that simply is not evident within the image. Rather than the model performing true analysis, it has actually constructed a believable but fabricated justification. To extensively test the model’s reasoning capabilities, a Two-Alternative Forced Choice (TAFC) framework [14] has been adopted within this study.

## 2.6 Ethics, Diversity and Industry Context

How a dataset is constructed directly afects what a detection model learns to look for [39]. Dataset bias is a relevant problem within digital forensics as it limits the models’ ability to evaluate diverse demographic groups accurately [24]. Widely available datasets such as FFHQ [20] remain the industry standard for training Generative AI; researchers must also acknowledge that FFHQ does not accurately represent certain age groups or ethnic backgrounds with approximately 70% of images featuring Caucasian individuals [27]. Bias towards certain age brackets, ethnic backgrounds, and environmental lighting directly leads to higher false-positive rates for under-represented groups.

The synthetic corpus was designed using a prompt matrix to ensure demographic variance across age, gender, ethnicity and professional job roles to reduce the possibility of stereotypical outputs. Similarly, the FFHQ control group images were manually selected to ensure the evaluation framework did not favour any particular demographic profile. Together, the diverse prompt matrix and the selection of FFHQ images were implemented to avoid demographic shortcuts that can make MLLMs unreliable forensic tools

## 3 Methodology and Implementation

This section outlines the methodology and implementation choices applied across three experiments. Each experiment is designed to address one or more of the four hypotheses established in Section 1. A concern across all experiments is the distinction between self-model detection, where a detector evaluates images generated by its own ecosystem, and cross-model detection, where a detector evaluates images from an external generator. This self-/cross-model comparison is integrated into experiments 1 and 2 through the full 6×7 detector-generator matrix, and is the primary focus of H1.

The structure of this methodology is illustrated below within Figure 1.

![](images/51920f863140cb7bfbd5e7e18126a60c351967dd7c003ba9202c3c2edcd6adda.jpg)  
Fig. 1. Methodological Framework: The logical progression from data sourcing (FFHQ and AI Generators) through the three-tiered experimental structure, concluding with the qualitative audit for Experiment 1 justifications.

## 3.1 Dataset Construction

All materials, including the datasets, used in this study are available at: https://github.com/OliviaSalamon/dissertationcorpus.

Prior testing for this study revealed a significant problem to overcome: images generated by older models, such as Gemini 1.5, contained inconsistencies that would result in the MLLMs classifying them as AI-generated, such as overly smooth skin textures or unusually perfectly glossed image backgrounds [35].

The dataset was constructed in three stages. First, a synthetic corpus of 350 images was generated across seven leading generators, each generator providing a pool of 50 images, using a structured prompt matrix. Second, the prompt matrix was designed using a diversity-guided Equality, Diversity, and Inclusion (EDI) framework to prevent stereotypica outputs. Third, a manually selected control group of 50 authentic photographs was sourced from FFHQ [20].

3.1.1 Synthetic Corpus Generation. To overcome this limitation, a custom 400-image dataset was constructed, 350 of which were generated using seven state-of-the-art text-to-image generators. The models selected for this study were ranked among the top performers on the Artificial Analysis Text to Image Leaderboard [1] at the time of dataset creation (December 2025). The seven generators used within this study were: Gemini 3, Gemini 2.5, Flux.2 Pro, GPT-Image-1, GPT-Image-1.5, Seedream 4.0, and Seedream 4.5.

All 350 images were generated via each generator’s oficial developer API, the generation scripts are provided in the project corpus

(01\_Pre\_Experimental\_Setup/generate\_images\_master\_script.py). Generating these images through APIs ensured a baseline without UI-specific watermarks, user-interface alterations, or image compression that can result from web interface generation. The only exception to this API-only pipeline occurs in Experiment 3, where additional images were specifically generated using the Gemini Web Interface to capture the visible logo watermark for comparative ablation testing.

Manuscript submitted to ACM

3.1.2 Prompt Matrix Design. A structured prompt matrix (full prompt matrix is available in the project corpus at (01\_Pre\_Experimental\_Setup/project\_prompts.csv), was constructed to ensure tests across the diferent models remain consistent. This was established to prevent stereotypical outputs that would directly impact detection rates of the images [27]. To mitigate this, a dynamic prompt structure was used to ensure contextual variance within variables such as age, gender, physical traits, ethnicity, and professional roles were embedded into each of the 50 prompts within the prompt matrix. For example, prompts included: ‘A professional headshot of a woman in her 50s with grey hair, natural window light, centred composition’ and ‘A teenage boy with South Asian features, casual outdoor setting, overcast lighting.’ In addition to contextual variance, photograph background bias was further mitigated by applying consistent photographic qualifiers, such as “natural window light” and “centred composition”. The prompt matrix was designed to prevent any demographic shortcuts the generator models might rely on.

3.1.3 Real Image Control Group. A control group of 50 authentic images of human faces was sourced from the Flickr-Faces-HQ (FFHQ) dataset [20]. FFHQ was selected for a few main reasons: its images are taken under real-world conditions at high resolution (1024×1024) making them a representative and authentic baseline. In addition, its portrait composition is directly comparable to AI-generated head shots. However, the base FFHQ dataset is known to be heavily demographically biased, consisting of nearly 70% Caucasian and only 4.2% Black individuals [27]. To mitigate this, images were manually selected to guarantee a balance of age groups, genders and ethnicities. This purposeful sampling prevented the AI detectors from using demographic features as a classification shortcut.

## 3.2 Experimental framework

To evaluate the forensic capabilities ofthe chosen MLLMs, a three-tiered framework was developed with each experiment designed to address specific hypotheses. Experiment 1 (single-image zero-shot classification) provides the baseline detection data required to evaluate H1 (Self- and Cross-Model Detection) and H4 (Justification Reliability). Experiment 2, pairwise TAFC classification, is designed to test H2 (Pairwise Image Advantage) by comparing accuracy against the Experiment 1 baseline. Experiment 3 (Gemini watermark ablation) directly addresses H3 (Efects of Watermarks) by isolating the contribution of visible and invisible watermark signals to detection accuracy.

## 3.2.1 Experiment 1: Single-Image Classification.

Phase A: Quantitative Zero-Shot Baseline. The first experiment was designed to establish a zero-shot baseline. In this experiment, each MLLM was presented with a single image, randomly drawn from either the 50 authentic FFHQ images or from one of seven generator-specific pools, each containing 50 images produced by a single generator.

Phase B: Qualitative Forensic Reasoning. Building directly onto the single-image classification within Experiment 1 and to move testing beyond a binary result (1 for correct classification, 0 for incorrect classification) this phase of Experiment 1 focuses on an automation script to investigate the “black box” of AI decision-making [31]. Phase B extracts the natural-language justifications provided by each MLLM, forming the evidence base for the H4 (Justification Reliability) evaluation. The validity of these justifications is assessed in Phase C. This consisted of a subset of 10 fixed images (5 real images from the FFHQ dataset and 5 AI-generated images). This subset size was selected to allow five independent classification runs per image per model, producing 50 data points per model for consistent analysis while remaining feasible for the Manual Justification Audit process outlined below in Phase C. The five AI-generated images were drawn from four distinct generator sources: Flux.2 Pro, GPT-Image-1.5, Gemini 3, and GPT-Image-1, to ensure the reasoning evaluation was carried across the top modern generation architectures. By presenting the same ten images Manuscript submitted to ACM

to each model across five independent runs, this phase tested whether each model produced consistent reasoning for identical inputs; inconsistent justifications across runs would highlight inaccuracies within detection rather than genuine structural analysis. As the MLLMs outputted their reasoning in a structured JSON format, the script allowed for analysis to check whether visible or invisible watermark signals were referenced in the model’s text outputs.

Phase C: Manual Justification Audit. To conclude the single-image analysis experiment, the final phase involved a Manual Justification Audit of the natural-language justifications produced in Phase B. Since MLLMs are known to ‘hallucinate’ plausible but factually incorrect justifications [2, 18], the outputs from Phase B were manually audited to evaluate whether the reasoning provided was accurate. Raw justifications were cross-referenced against a manua visual audit. The MLLM responses were categorised as ‘Correct + Reasonable’ if they referenced correct cues, or ‘Hallucination’ if the classification was correct but supported by non-existent or incorrect artefacts. Manually validating the justifications ensured the study’s conclusions reflected genuine forensic evaluations by the MLLM rather than a statistical guess.

3.2.2 Experiment 2: Pairwise Image Analysis (Contextual Advantage). The second experiment introduced contextual reasoning through a TAFC evaluation protocol [14]. To ensure the results were not afected by positional bias, the placement of the authentic and synthetic images as either “Image A” or “Image B” was fully randomised.

In this part of the experiment, the MLLMs were provided with two randomly selected images, one real image from the FFHQ control dataset and one synthetic image generated by one of the 7 generators at random from the 7 pools of 50 images. The models were prompted with “You are presented with two images, labelled IMAGE A and IMAGE B. One is a REAL photograph, and the other is AI-GENERATED. Your task is to identify which image is the AI-generated fake”, in order to identify which of the two images was AI-generated and which was a real photograph, providing a binary classification and a justification for their decision

This methodology, which focuses on the comparison was directly designed to test the Contextual Advantage Hypothesis (H2). The experiment measured whether accuracy improved when models were provided with a real reference image alongside the synthetic one, testing whether comparative context benefits detection performance. Justification analysis was not conducted for Experiment 2, as reasoning auditing was limited to Phase B and C of Experiment 1.

## 3.2.3 Experiment 3: Targeted Watermark Ablation Study (Gemini/SynthID).

Phase A: Forensic Rationale and Dataset Construction. By introducing an ablation approach, cropping visible watermarks and resampling invisible ones through pixel resampling, this experiment investigates the true detection capabilities of MLLMs. The objective is to determine whether these models genuinely recognise deepfake artefacts, or if their accuracy is influenced by generator traces; a methodology that aligns with the framework established by Gandh and Jain [9].

The five-dataset comparison framework within this study isolates each signal: visible logo, invisible SynthID, and their combination, to determine whether MLLM accuracy reflects genuine forensic reasoning or watermark exploitation This decision to perform an ablation study on the Gemini ecosystem was driven by the integration of SynthID, Google DeepMind’s invisible Watermarking tool. According to oficial documentation, SynthID “embeds a digital watermark directly into the pixels of an image, making it imperceptible to the human eye, but detectable for identification” [12].

Preliminary research for this study revealed a double layer forensic system within Gemini: images generated via the Gemini Web Interface not only include the invisible SynthID embedded watermark but also a visible watermark on the Manuscript submitted to ACM

lower right hand corner of the generated image, a small, four-pointed “sparkle” icon. Images sourced directly from the Gemini API (Imagen 3) do not generate the visible logo yet only with SynthID embedded within the image. This provided an opportunity to study the importance of each marker in a controlled environment, forming the basis for the five-dataset ablation structure.

To investigate this, five distinct datasets (Pools) were constructed:

• D1 (Gemini-API): Raw Gemini API output (Imagen 3) — SynthID only, no logo. These images form the Gemini generator pool used in Experiments 1 and 2.

• D2 (Gemini-Web): Raw Gemini Web Interface output — SynthID + visible sparkle logo.

• D3 (Gemini-Web-Cropped): D2 with visible logo manually cropped.

• D4 (Gemini-Web-Ablated): D2 subjected to LDM reconstruction — logo remains, SynthID disrupted.

• D5 (Gemini-Full-Ablation): D2 cropped and subjected to LDM reconstruction — no logo, SynthID disrupted.

Phase B: Dataset Structure and Ablation Design. To complete the ablation experiment, the neutralisation of the SynthID signal in Datasets 4 and 5 was essential. This was achieved through an Image-to-Image (img2img) LDM reconstruction pass utilising the Hugging Face Difusers library [38]. Specifically, the pipeline employed the Realistic Vision V5.1 model [6], which was selected for its ability to maintain high photographic quality during pixel resynthesis.

By altering the pixels at a tested 0.15 strength, a value supported by Zhao et al. [40] for difusion-based image alterations, the high-frequency digital watermarks were successfully disrupted while still maintaining high visual quality of the image’s original state. This specific strength was selected to minimise degradation of the subject’s facial biometrics and keeping the image metadata such as the background and lighting as similar as possible to the originally generated image. This could be a tactic a malicious actor might use to bypass AI detection by running an image through a secondary generative pass. The complete watermark removal script is provided in the corpus at 04\_Experiment\_3\_Watermark\_Ablation/remove\_synthid\_dataset4.py.

Phase C: Comparison Matrix. To evaluate the impact of specific forensic markers (H3), five distinct dataset comparisons were conducted. This matrix isolates the presence of invisible SynthID signals against visible “sparkle” watermarks to determine their individual and combined efects on detection accuracy. The raw dataset comparison results are archived within the corpus at 04\_Experiment\_3\_Watermark\_Ablation/Dataset\_Comparison\_Results/.

• C1 (API vs. Cropped Web): D1 (Gemini-API) vs. D3 (Gemini-Web-Cropped). Evaluates consistency across API and Web-generated sources when visible markers are absent.

• C2 (Visible Logo Efect): D2 (Gemini-Web) vs. D3 (Gemini-Web-Cropped). Measures the accuracy gain provided by the visible logo when the invisible signal is present.

• C3 (Logo-Only Detection): D4 (Gemini-Web-Ablated) vs. D5 (Gemini-Full-Ablation). Assesses if the detector relies on the visible logo once the invisible signal is neutralised.

• C4 (Invisible Signal Efect): D2 (Gemini-Web) vs. D4 (Gemini-Web-Ablated). Isolates the efectiveness of the SynthID watermark while the visible logo remains constant.

• C5 (Invisible-Only vs. Control): D3 (Gemini-Web-Cropped) vs. D5 (Gemini-Full-Ablation). Evaluates detection performance based solely on the invisible signal against a fully ablated control.

Figure 2 below provides a visual comparison of the five datasets within the ablation study.

Manuscript submitted to ACM

![](images/1f455c80920b743d9d00c586dc20892730f0a04fe2d91c455316554817480964.jpg)  
D1: SynthID Only

![](images/ba222b5c5772c87a622620a23a7c2f07b45b5d83d692d3d615de95c1132f2089.jpg)  
D2: Logo + SynthID

![](images/9f256c9697a67f2b1fd6ad67b99096061b056ca54057c23c0cee73a7633e64b4.jpg)  
D3: Logo Removed

![](images/a53998507816de0643a43a33217500a8d5eedcf5571b0e546ad04a35c369c155.jpg)  
D4: SynthID Disrupted

![](images/ea205c97595817aaf3fbd27573bd34b494f5276a36ba09fda06636e4a6acc2a9.jpg)  
D5: Full-ablation  
Fig. 2. Targeted Watermark Ablation Series: Visual comparison of the five Gemini dataset pools (D1–D5) used to isolate the forensic impact of visible logos and invisible digital signatures.

## 3.3 Implementation Framework and Parameter Design

3.3.1 Automation Pipeline. The experimental phases were executed through a custom Python pipeline interacting with the OpenAI [30], Google Gemini [11], and Replicate [33] APIs. The single-image, pairwise, and justification evaluation scripts are provided in the corpus at

02\_Experiment\_1\_Single\_Image/ and 03\_Experiment\_2\_Pairwise/. Replicate was specifically utilised to host the LDM reconstruction models required for the watermark ablation study in Experiment 3. Findings were automatically extracted via JSON schemas into a binary matrix (1 for correct, 0 for incorrect) for statistical analysis. The architectura logic of this automated framework and the data extraction process are visualized in Figure 3.

![](images/9a35e1305063ea32676b27af1a842c4503648f1987379375272a61362d01d34b.jpg)  
Fig. 3. Automation Pipeline: Workflow illustrating the interaction between the custom Python controller, the four external API services (including the Seed 1.6 detector), and the automated data extraction process.

3.3.2 Parameter Standardisation and Bias Control. Temperature settings were applied diferently across generators and detectors. For image generation, a temperature of 0.0 was set where the API permitted within an adjustable parameter to produce deterministic outputs. For the six MLLM detectors, models were kept at their default settings (typicall 1.0) to represent realistic conditions. The temperature diferences between models were a deliberate choice to allow detectors to operate under their standard conditions. Not all MLLM APIs expose a temperature parameter; where this setting was unavailable, the model’s default configuration was used. Gemini’s detection configuration was explicitly set Manuscript submitted to ACM

to 0.0 to ensure consistent justification outputs across the five repeated runs in Phase B of Experiment 1. Instructions and system prompts remained uniform across all platforms to prevent any potential prompt bias.

For Experiment 2, the pipeline utilised a shufle randomisation algorithm [15] to randomise the positioning of images (Position A vs. Position B). This neutralised positional bias inherent in TAFC protocols [17, 28].

## 3.4 Statistical Methodology

The independent variables across the three experiments were:

• Detector Model: The six MLLMs under evaluation, including GPT-5.2, GPT-5, GPT-4o, Gemini 3, Gemini 2.5, and Seed 1.6.

• Generator Source: the generator source of the evaluated image (the seven generators plus FFHQ authentic images)

• Experimental Condition: the experimental condition: single-image, pairwise, or ablation dataset tier (D1–D5).

The dependent variable in all three experiments was binary classification accuracy: 1 for a correct classification and 0 for an incorrect one. Since binary classification scores (0 or 1) follow a Bernoulli distribution and do not meet the normality assumptions required for parametric testing, a non-parametric framework was adopted to evaluate the hypotheses at a significance level of � < 0.05:

• Mann-Whitney U Test: Applied to independent samples to compare baseline detection rates across distinct model families (e.g., OpenAI vs. Google).

• Wilcoxon Signed-Rank Test: Applied to dependent, paired samples to measure how an individual model’s performance changed under specific experimental variables, such as pairwise presentation or watermark ablation. The Python scripts implementing these tests are provided at 04\_Experiment\_3\_Watermark\_Ablation/Gemini\_Statistics.py, with raw outputs at

04\_Experiment\_3\_Watermark\_Ablation/wilcoxon\_manwhitneyu\_results.txt.

Finally, to ensure that any statistically significant findings held practical weight in an experimental context, the Rank-Biserial Correlation (�) was used to calculate the efect size.

## 4 Results

## 4.1 Overview of Experimental Results

Section 3 described image generators as Large Vision Models (LVMs) while the six MLLM detectors: GPT-5.2, GPT-5, GPT-4o, Gemini 3, Gemini 2.5 and Seed 1.6 represent a separate set of models. During the time of evaluation, GPT-4o and Seed 1.6 were included as detectors because they represent the most accessible commercial MLLMs from their ecosystems. This enabled cross-generation comparison within the GPT family and cross-ecosystem comparison with ByteDance’s Seed platform.

Table 1 provides a cross-experiment overview of pairwise versus single-image accuracy shifts across all six detectors, directly relevant to H2 (Pairwise Image Advantage). Figure 4 visualises the single-image versus pairwise accuracy gap across all six detectors.

Manuscript submitted to ACM

Table 1. Summary of detector model performance across single-image and pairwise conditions. Models are ranked in descending order according to their mean single-image accuracy.
<table><tr><td>Model</td><td>Single Mean (%)</td><td>Pairwise Mean (%)</td><td>Context Impact</td><td>SD (Single)</td></tr><tr><td>GPT-5.2</td><td>78.8</td><td>82.5</td><td>+3.7</td><td>3.41</td></tr><tr><td>Gemini 3</td><td>69.5</td><td>64.5</td><td>-5.0</td><td>4.67</td></tr><tr><td>GPT-5</td><td>64.5</td><td>69.4</td><td>+4.9</td><td>5.23</td></tr><tr><td>Gemini 2.5</td><td>62.6</td><td>45.5</td><td>-17.1</td><td>4.59</td></tr><tr><td>GPT-40</td><td>58.3</td><td>61.3</td><td>+3.0</td><td>4.46</td></tr><tr><td>Seed 1.6</td><td>56.8</td><td>55.7</td><td>-1.1</td><td>6.70</td></tr></table>

![](images/83b45b3c4a55bb8531905896d985a99eb8380ca02396bd354a15c67b69950cb3.jpg)  
Fig. 4. Comparative performance of detector models. Results are ranked by baseline single-image mean from best to worst. Note the performance gains in the GPT family compared to the accuracy decline observed in the Gemini ecosystem under pairwise conditions.

## 4.2 Experiment 1 Results: Single-Image Baseline and Qualitative Reasoning

4.2.1 Quantitative Zero-Shot Baseline. Tier 1’s primary objective was to determine whether MLLMs are capable of identifying AI-generated images without any external references. The performance matrix within Table 2 highlights that some models demonstrated clear forensic capability; the remaining models struggled to exceed even the 50% Manuscript submitted to ACM

baseline for this study. Based on the single-image analysis results, the most efective detector was GPT-5.2 achieving a mean accuracy of 78.8%. GPT-5 and Gemini 3 also performed well with levels above 60% (full data within the corpus 02\_Experiment\_1\_Single\_Image/single\_image\_analysis\_phaseA.xlsx).

Table 2. Single-image detection accuracy (%) per detector model per generator source. Models are grouped by vendor ecosystem. Shaded cells indicate same-ecosystem detector-generator pairs (self-detection), relevant to H1.
<table><tr><td>Detector</td><td>Flux.2 Pro</td><td>GPT-Img-1</td><td>GPT-Img-1.5</td><td>Gem-2.5</td><td>Gem-3</td><td>SD-4.0</td><td>SD-4.5</td></tr><tr><td>GPT-5.2</td><td>87.9</td><td>84.3</td><td>72.1</td><td>84.6</td><td>65.8</td><td>78.6</td><td>78.6</td></tr><tr><td>GPT-5</td><td>88.6</td><td>51.4</td><td>50.7</td><td>91.9</td><td>45.0</td><td>52.1</td><td>71.4</td></tr><tr><td>GPT-40</td><td>78.6</td><td>60.7</td><td>37.9</td><td>76.5</td><td>36.2</td><td>56.4</td><td>61.4</td></tr><tr><td>Gemini 3</td><td>72.1</td><td>74.3</td><td>72.1</td><td>67.1</td><td>67.8</td><td>68.6</td><td>64.3</td></tr><tr><td>Gemini 2.5</td><td>82.9</td><td>80.0</td><td>44.3</td><td>79.9</td><td>40.3</td><td>54.3</td><td>56.4</td></tr><tr><td>Seed 1.6</td><td>73.6</td><td>45.7</td><td>50.0</td><td>76.5</td><td>47.7</td><td>47.1</td><td>57.1</td></tr><tr><td>Mean</td><td>80.6</td><td>66.1</td><td>54.5</td><td>79.4</td><td>50.4</td><td>59.5</td><td>64.9</td></tr></table>

However, the results revealed a significant performance gap with both Seed 1.6 (56.8%) and GPT-4o (58.3%) hovering just above the 50% baseline used. These models often failed to identify structural artefacts within the images, suggesting that their detection capabilities are not precise enough to be used within a forensic setting. In addition the Standard Deviation (SD) for Seed 1.6 (6.70) was the highest figure amongst the models. This indicated that not only was the model’s performance low but also inconsistent compared to the other models tested across the ten experimental runs. Therefore, while H1 is supported for the more advanced architectures, detection capability is not consistent across the current MLLMs. The results further revealed that detection accuracy was significantly influenced by the generator source of the AI-generated image. As shown in Table 2, the detection rates varied considerably depending on which of the seven generators produced the image that was being evaluated. These results suggest that generator architecture has influence on how detectable the image is.

With a cross-model mean detection rate of 80.6%, Flux.2 Pro was the most easily identifiable generator. The results imply that Flux.2 Pro likely produces more detectable artefacts. In contrast, Gemini 3 generated images were the most dificult to detect by the six detectors within the study, with a mean detection rate of just 50.4%, at the boundary of random chance and therefore indicating that the most recent generative architectures are reaching a point where standalone zero-shot single-image classification is becoming an unreliable forensic methodology.

A Mann-Whitney U test comparing self-model detection accuracy (same-ecosystem detector-generator pairs) against cross-model accuracy revealed $\dot { p } = 0 . 3 8 4 , r = 0 . 1 8 8$ , confirming that ecosystem alignment was not a statistically significant predictor of detection accuracy.

The diagonal efect, where same-ecosystem detector-generator pairs show varying performance, is visible in the highlighted cells of Table 2 and is discussed in the H1 verdict.

4.2.2 Qualitative Reasoning. Phase B investigated the forensic reasoning provided by each model through an evaluation focused on a subset of ten images, each analysed across five separate iterations. Using a Manual Justification Audit, jus tifications were sorted into specific categories: “Correct + Reasonable” were assigned when a model identified authentic structural indicators, while “Incorrect Detection” was used for failed classifications. The compiled results are presented in Table 3; the complete audit spreadsheet is provided in the corpus at 02\_Experiment\_1\_Single\_Image/Manual\_audit.xlsx.

Table 3. Phase B Manual Justification Audit results: frequency of correct and reasonable justifications across 48 evaluated cases per model (10 fixed images × 5 runs = 50 classification events per model; 2 cases excluded due to Gemini 3’s content safety block on one authentic image, yielding 48 usable responses) Models are presented in a fixed order consistent with baseline performance rankings.
<table><tr><td>Model</td><td>Correct + Reasonable</td><td>Incorrect Detection</td><td>Accuracy (%)</td></tr><tr><td>GPT-5.2</td><td>45</td><td>3</td><td>93.8</td></tr><tr><td>Gemini 3</td><td>37</td><td>11</td><td>77.1</td></tr><tr><td>GPT-5</td><td>31</td><td>17</td><td>64.6</td></tr><tr><td>Gemini 2.5</td><td>37</td><td>11</td><td>77.1</td></tr><tr><td>GPT-40</td><td>28</td><td>20</td><td>58.3</td></tr><tr><td>Seed 1.6</td><td>29</td><td>19</td><td>60.4</td></tr></table>

Leading the qualitative evaluation with 93.8% was GPT-5.2, citing structural artefacts correctly such as unusual object geometry, unnatural image texture and blurred image backgrounds. GPT-4o had a considerably weaker score of 58.3% despite sharing the same architectural lineage as GPT-5 and GPT-5.2. This suggests that model version, rather than the generator ecosystem, is the strongest factor of forensic reasoning quality. Gemini 3 triggered a content safety block on one authentic FFHQ image, returning no classification and therefore recording it as an automatic detection failure. The Manual Justification Audit revealed that no hallucinations occurred in any of the models and that every error recorded was simply a misclassification rather than a correct guess. Sun et al. [37] and Li et al. [22] predicted hallucinations would be common; however, no hallucinations were recorded. Every misclassification was accompanied by structurally coherent reasoning, the models were wrong, but their logic was reasonable due to image inconsistencies. A clear illustration of this is img\_0017.png ,as shown below within Figure 5, a GPT-Image-1.5 generated image with four out of six detector models classified as real across all five runs. Only Gemini 3 and GPT-5.2 correctly identified it as an AI-generated image by observing inconsistent hair blurring and implausible background geometry.

![](images/9d535798788df0735c59f48c70268b61a0e782ffabf96eb4b376da607a828c06.jpg)  
Fig. 5. img\_0017.png, GPT-Image-1.5 generated image with four out of six detector models classified as real across all five runs.

4.2.3 Hypothesis Verdicts: H1 and H4. Tier 1 experiment results ofer only partial support for Hypothesis 1 (H1), this is seen in GPT-5.2 detecting GPT-Image-1 with 84.3% detection accuracy and Gemini 2.5 detecting Gemini AI-generated images at 79.9% detection accuracy. Neither figure supports a consistent advantage over performance against external generator models. For example, GPT-5.2 achieved 87.9% detection against Flux.2 Pro generated images, a completely diferent generation ecosystem. It is possible that the quality and artefacts of the generated image appear to be the strongest factors of detection dificulty than specific ecosystem alignment between the detector and the generator (see Appendix A.2.1 for full test output).

H4 is partially rejected as no hallucinations were recorded across any model, suggesting these models engage in genuine forensic reasoning rather than the shortcut-driven pattern predicted by recent literature.

The Manual Justification Audit was conducted by a single researcher, which introduces a confirmation bias risk. A future approach should involve multiple auditors to cross-verify the AI’s reasoning patterns to ensure the nohallucination finding is not a product of one opinion.

## Hypothesis Verdicts: H1 and H4

H1 (Self-Detection): Partially Supported — GPT-5.2 and Gemini 2.5 showed modest same-ecosystem advantages, but generator image quality proved a far stronger predictor of detection dificulty than ecosystem alignment.

H4 (Justification Reliability): Partially Rejected — No hallucinations were recorded across any model; all errors were clean misclassifications, suggesting genuine forensic reasoning rather than shortcut-driven pattern matching.

## 4.3 Experiment 2 Results: Pairwise Classification and Contextual Advantage

The primary objective of tier 2 was to determine whether the presence of a real image during the evaluation would improve detection accuracy across the six detector models. Table 4 presents the full pairwise experiment detection matrix across all detector and generator combinations.

Table 4. Pairwise detection accuracy (%) per detector model per generator source. Performance change compared to Table 2’s single-image baseline is shown in parentheses. Models are grouped by vendor ecosystem.
<table><tr><td>Detector</td><td>Flux.2 Pro</td><td>GPT-Img-1</td><td>GPT-Img-1.5</td><td>Gem-2.5</td><td>Gem-3</td><td>SD-4.0</td><td>SD-4.5</td></tr><tr><td>GPT-5.2</td><td>97.9 (+10.0)</td><td>88.6 (+4.3)</td><td>62.1 (-10.0)</td><td>94.0 (+9.4)</td><td>80.0 (+14.2)</td><td>72.1 (-6.5)</td><td>82.9 (+4.3)</td></tr><tr><td>GPT-5</td><td>87.1 (-1.5)</td><td>71.4 (+20.0)</td><td>50.7 (0.0)</td><td>92.7 (+0.8)</td><td>37.3 (-7.7)</td><td>72.1 (+20.0)</td><td>74.3 (+2.9)</td></tr><tr><td>GPT-40</td><td>85.7 (+7.1)</td><td>59.3 (-1.4)</td><td>40.0 (+2.1)</td><td>88.7 (+12.2)</td><td>26.0 (-10.2)</td><td>60.0 (+3.6)</td><td>69.3 (+7.9)</td></tr><tr><td>Gemini 3</td><td>81.4 (+9.3)</td><td>67.1 (-7.2)</td><td>43.6 (-28.5)</td><td>83.3 (+16.2)</td><td>58.0 (-9.8)</td><td>50.7 (-17.9)</td><td>67.1 (+2.8)</td></tr><tr><td>Gemini 2.5</td><td>64.3 (-18.6)</td><td>56.4 (-23.6)</td><td>33.6 (-10.7)</td><td>71.3 (-8.6)</td><td>21.3 (-19.0)</td><td>35.0 (-19.3)</td><td>36.4 (-20.0)</td></tr><tr><td>Seed 1.6</td><td>85.0 (+11.4)</td><td>43.6 (-2.1)</td><td>35.7 (-14.3)</td><td>84.0 (+7.5)</td><td>38.7 (-9.0)</td><td>50.0 (+2.9)</td><td>52.9 (-4.2)</td></tr><tr><td>Mean</td><td>83.6 (+3.0)</td><td>64.4 (-1.7)</td><td>44.3 (-10.2)</td><td>85.7 (+6.3)</td><td>43.6 (-6.8)</td><td>56.7 (-2.8)</td><td>63.8 (-1.1)</td></tr></table>

GPT-5.2 achieved the strongest pairwise performance with a mean accuracy of 82.5%, reaching a peak of 97.9% against Flux.2 Pro generated images, the highest single detection rate recorded across the entire study. All GPT models improved under pairwise conditions with GPT-5 gaining 4.9 percentage points and GPT-4o gaining 3.0 points compared to their single-image baselines. All three GPT models gained accuracy within the pairwise experiment, suggesting that a detector benefits from a comparative reference rather than being confused by it.

Wilcoxon Signed-Rank tests comparing each detector’s pairwise accuracy against its single-image baseline did not confirm a statistically significant improvement for GPT-5.2 $( \mathrm { p } = 0 . 4 0 6 , \mathrm { r } = 0 . 3 9 3 )$ despite its descriptive gains, but did confirm a highly significant decline for Gemini 2.5 $( \mathrm { p } = 0 . 0 1 6 , \mathrm { r } = - 1 . 0 0 0$ , see Appendix A.2.2 for full test results).

4.3.1 Unexpected Pairwise Behaviour in Gemini Models. An important realisation within the Tier 2 experiment was that Gemini models recorded lower accuracy, not higher, under pairwise conditions. Gemini 2.5 recorded the steepest drop of any model in the study, falling 17.1 percentage points from 62.6% to 45.5%, below the random chance threshold of 50%. Gemini 3 similarly declined by 5.0 points to 64.5%. This pattern contradicts H2 for the Gemini ecosystem and was an unexpected finding of the study.

One explanation for this behaviour is “comparison noise” which is when the introduction of a real image appears to interfere with Gemini’s classification process rather than assist detection by providing a comparative reference point. Gemini models may be exhibiting a comparative bias that causes them to doubt the structural cues they previously detected in the single-image analysis. This is most evident in Gemini 2.5’s performance against Gemini 3 generated images, which declined to just 21.3% within the pairwise analysis compared to 40.3% in the single-image analysis, significantly below the random chance percentage and suggesting misclassification rather than simple uncertainty.

However, “comparison noise” remains a suggested explanation rather than a confirmed justification. Another possibility would be that Gemini’s training might use diferent logic when comparing images side-by-side than when judging a single image alone. This could be tested by pairing two synthetic images together rather than one real and one fake as an extension of experiment 2.

4.3.2 Generator Dificulty Under Pairwise Conditions. The patterns observed within Tier 1 continued to persist under pairwise analysis. Gemini 3 images remained the most dificult to detect with a cross-model mean of 43.6%, while Flux.2 Pro remained the most detectable generator at 83.6%. However, a shift was observed as Gemini 2.5 images became the easiest to detect under the pairwise-image experiment with a mean of 85.7%, overtaking Flux.2 Pro.

4.3.3 Hypothesis Verdict: H2. Overall, the pairwise-image results provide partial support for Hypothesis 2 (H2). The GPT-series showed consistent descriptive advantage, though these did not reach statistical significance. Through the unexpected findings regarding the Gemini family, H2 is rejected for the Gemini family, where pairwise presentation was associated with a significant decline in performance.

## Hypothesis Verdict: H2

H2 (Contextual Advantage): Partially Supported — The GPT family showed consistent descriptive gains (+3.0 to +4.9pp) that were not statistically significant, but Gemini 2.5 showed a statistically significant decline $( p = 0 . 0 1 6 , r = - 1 . 0 0 0 )$

## 4.4 Experiment 3 Results: Watermark Ablation Study

4.4.1 Ablation Study Overview. The third experiment within this study investigates how Gemini’s detection accuracy either increases or decreases with the presence of digital watermarking (SynthID) embedded within its own generated images. Five image datasets were constructed to test the efect of the two Gemini watermarks present within their images: the visible sparkle logo present on web-generated Gemini images, and the invisible SynthID signal embedded within the image’s pixels. Five comparisons were conducted across these datasets using Wilcoxon Signed-Rank tests Manuscript submitted to ACM

for paired comparisons and a Mann-Whitney U test for the independent stability check, with efect sizes reported as Rank-Biserial Correlation (r). Table 5 presents the detection rates and statistical results across all comparisons. The complete Wilcoxon output for all ablation comparisons is provided in Appendix A.3. Figure 6 also provides a visual representation of the detection accuracy across tiers D1–D5 as described above.  
Table 5. Watermark ablation detection rates and statistical results across five Gemini image dataset comparisons.
<table><tr><td>Comp.</td><td>Datasets</td><td> $\mathbf { D _ { A } } \left( \% \right)$ </td><td> $\bf { D _ { B } } \left( \% \right)$ </td><td> $\scriptstyle p - \mathbf { v a l u e }$ </td><td>r</td><td>Effect Size</td></tr><tr><td>C1</td><td>D1 vs D3</td><td>94.0</td><td>90.0</td><td>0.4354</td><td>0.078</td><td>Negligible</td></tr><tr><td>C2</td><td>D2 vs D3</td><td>100.0</td><td>90.0</td><td>0.0075</td><td>0.378</td><td>Medium</td></tr><tr><td>C3</td><td>D4 vs D5</td><td>100.0</td><td>100.0</td><td>1.0000</td><td>0.000</td><td>None</td></tr><tr><td>C4</td><td>D2 vs D4</td><td>100.0</td><td>100.0</td><td>0.0455</td><td>0.283</td><td>Small</td></tr><tr><td>C5</td><td>D3 vs D5</td><td>90.0</td><td>100.0</td><td>0.0020</td><td>0.437</td><td>Medium-Large</td></tr></table>

Note: Efect size (�) interpreted as: negligible $r < 0 . 1$ , small $r \geq 0 . 1$ , medium $r \geq 0 . 3 .$ , large $r \geq 0 . 5$ [4].

![](images/c7738b22e43b02d806aef074475e298a69040c74bb544f9192d6cd3a4cdc725e.jpg)  
Fig. 6. Detection accuracy across tiers D1–D5. The drop at D3 highlights the masking efect where results fall below the 100% baseline. (D1 = Gemini-API, D2 = Gemini-Web, D3 = Gemini-Web-Cropped, D4 = Gemini-Web-Ablated, D5 = Gemini-Full-Ablation).

4.4.2 Stability and Baseline (C1). Comparison C1 compared D1 (Gemini-API) against D3 (Gemini-Web-Cropped), returning $ { p } = 0 . 4 3 5 4$ with a negligible efect size of $r = 0 . 0 7 8 .$ This suggests no significant diference between dataset groups, validating that subsequent comparisons reflect watermark manipulation rather than image content variability.

4.4.3 Visible Watermark Efect (C2). Comparison C2 explored the impact of the sparkle logo within Gemini generated images by comparing D2 (Gemini-Web), whose images contained both the visible logo and SynthID against D3 (Gemini-Web-Cropped), containing SynthID only with the visible logo cropped. The Wilcoxon Signed-Rank test returned Manuscript submitted to ACM $p = 0 . 0 0 7 5$ with a medium efect size of $r = 0 . 3 7 8 ,$ showing that the visible logo has a meaningful impact on detection consistency. Upon removal of the visible logo, detection dropped from 100% to 90%, indicating that the sparkle icon can be identified as a forensic shortcut that significantly improves accuracy when present. This indicates that removing a visible watermark leads to lower detection reliability.

4.4.4 SynthID as a Secondary Signal (C4). Comparison C4 examined D2 (Gemini-Web) whose images contained both watermarks, visible and invisible, against D4 (Gemini-Web-Ablated) where SynthID had been neutralised but the visible logo remained. The Wilcoxon test returned $\begin{array} { r } { p = 0 . 0 4 5 5 } \end{array}$ with a small efect size of $r = 0 . 2 8 3$ and both datasets remained at 100% binary detection for its Gemini AI-generated images. The diference in detection probability scores suggests that SynthID provides a second layer of forensic data even when the visible watermark is already present. This mino diference in efect size indicates the visible logo provides the primary forensic evidence, which restricts the additional impact SynthID can provide. Even $s 0 ,$ its role remains statistically significant and should be recognised as more than just a redundant feature.

4.4.5 The Ceiling Efect (C3). Comparison C3 compared D4 (Gemini-Web-Ablated) containing the visible sparkle logo but no SynthID, against D5 (Gemini-Full-Ablation) which was the fully ablated pure control containing no watermarks of any kind. The Wilcoxon test returned $p = 1 . 0 0 0$ with an efect size of $r = 0 . 0 0 0$ , with both datasets recording 100% detection when asked to detect the AI-generated image. The significance in C3 is Gemini’s ability to detect its own images through structural analysis, not through watermarks. It confirms that even when all watermark signals have been removed, Gemini still achieves perfect detection of its own generated images. This is attributed to Gemini detecting native artefacts by identifying structural inconsistencies within the image without relying on embedded watermarking signals. These results represent the ceiling efect as detection has already reached 100% through structural reasoning; watermarks within the image could not mathematically improve the score further.

4.4.6 Pure SynthID Impact (C5). Comparison C5 reveals the impact of SynthID which compares D3 (Gemini-Web-Cropped), containing only the invisible SynthID signal with the logo removed, against D5 (Gemini-Full-Ablation), the fully ablated control. The Wilcoxon test returned $p = 0 . 0 0 2 0$ with a medium-large efect size of $r = 0 . 4 3 7$ which confirms that SynthID is statistically significant and a powerful forensic signal, the strongest invisible cue for the detector when no other hints are present. However after careful analysis: detection reached only 90% in the SynthID-only group, whereas it hit 100% in the fully ablated control. This contradiction may reflect a masking phenomenon observed during evaluation. The reconstruction method to neutralise SynthID in D5 (Gemini-Full-Ablation) likely amplified native structural artefacts. In contrast, the embedding process for SynthID in D3 (Gemini-Web-Cropped) may have slightly diminished the high-frequency details that aid structural identification. SynthID serves as both a powerful forensic signal and a subtle inhibitor of the native structural artefacts that underpin successful detection.

4.4.7 Hypothesis Verdict: H3. Following the full ablation experiments, the results partially reject hypothesis 3. The Hypothesis predicted that the presence of any watermarks would increase detection accuracy however the data exposes the impact of image watermarking. Visible watermarks do provide a significant detection boost (C2: $p = 0 . 0 0 7 5 ,$ $r = 0 . 3 7 8 )$ , and SynthID functions as a strong independent forensic signal in isolation (C5: $\dot { p } = 0 . 0 0 2 0 , r = 0 . 4 3 7 )$ However, when all watermarks are removed entirely from the AI-generated images, detection still remains at 100% through native artefact analysis. This implies that watermarks are not essential to MLLM-based forensic detection. In addition the masking phenomenon identified in C5 further reveals that future invisible watermarks should be designed with careful consideration of their potential to suppress the structural signals that MLLM forensic tools depend upon. Manuscript submitted to ACM

## Hypothesis Verdict: H3

H3 (Efects of Watermarks): Partially Rejected — Visible watermarks provided a significant boost $\begin{array} { r } { ( \rho = 0 . 0 0 7 5 , } \end{array}$ $r = 0 . 3 7 8 )$ and SynthID is a strong independent signal $( p = 0 . 0 0 2 0 , r = 0 . 4 3 7 )$ , but native structural artefact analysis alone achieved 100% detection, suggesting watermarks are supplementary and not essential.

## 5 Discussion

## 5.1 Revisiting the Research Question

This study addressed the following research question: can MLLMs function as reliable deepfake detectors of MLLMgenerated images, and do their classifications and justifications reflect genuine reasoning or do they exploit shortcuts such as image watermarking?

From the findings of this study, the answer remains partial. Same-ecosystem detection showed no consistent advantage as GPT-5.2 detected external Flux.2 Pro images better (87.9%) than its own GPT-Image-1.5 (72.1%), confirming that generator image quality matters more than whether the image is a product of the same ecosystem generator.

Beyond the detection baseline, the most unexpected finding was the masking phenomenon in C5, a result that was never hypothesised and carries direct implications for watermark design.

## 5.2 Key Findings

GPT-5.2 was the leading detector in every part of this study, achieving 78.8% accuracy in single-image analysis and 93.8% in the qualitative audit. GPT-5.2 demonstrated that it does not simply guess but rather identifies the right image artefacts within its justifications. This contrasts with GPT-4o which comes from the same family but scored 58.3% accuracy within the audit, suggesting forensic capability is more dependent on model generation than ecosystem membership. Since no consistent self-detection advantage was observed across the full 6×7 detector-generator matrix, H1 is only partially supported within this study. The “Gemini pairwise paradox” was the study’s most unexpected result. Gemini 2.5’s accuracy reduced by 17.1%, falling below the random chance threshold of 50%, when a real photo was presented alongside the generated one. This suggests that, for certain AI architectures, a second image actually creates “comparison noise” which can confuse the model rather than assisting it.

Experiment 3 provided clarity on the efects of digital watermarking through the “ceiling efect”. Once both visible and invisible watermarks were stripped from the Gemini images, detection still reached 100%. Detection at 100% even without any watermarks shows that Gemini is able to recognise its own generative artefacts.

The C5 masking phenomenon was an unexpected finding. When SynthID was present but the visible logo removed (D3), detection fell to 90%, lower than the fully ablated control (D5) at 100%. An invisible watermark promoted as a detection tool is statistically associated with reduced detection accuracy $( p = 0 . 0 0 2 0 , r = 0 . 4 3 7 )$ compared to images carrying no watermark at all and this directly challenges the idea of invisible watermarking systems. While embedding a signal into the image frequency ensures it survives cropping and compression [13], it consequently modifies the high-frequency pixel structure that the MLLMs use for watermark detection. While previous researchers have discovered resistance to removal [16, 26], this study demonstrates that watermarking systems can no longer be evaluated purely on their resistance to removal as a highly durable watermark is counterproductive if it accidentally hides the generative flaws needed to detect it as AI-generated.

The absence of hallucinations across the six models within the Manual Justification Audit directly challenges the predictions of Sun et al. [37] and Li et al. [22]. Within the literature it was believed that MLLMs would “hallucinate” forensic justifications when classifying images. This pattern was not observed in this study. Every incorrect classification was a justified misclassification, models were wrong; however their errors remained grounded in identifiable structural reasoning rather than made-up justifications.

## 5.3 Relation to Existing Literature

Detection dificulty increased with each generation of generator architecture, with Gemini 3 images achieving a cross-model detection mean of 50.4%, close to the boundary of random chance. This directly reflects what Rombach et al. [35] and Rössler et al. [36] predicted: as generative architectures advance, detection strategies that once worked stop working. Liu et al. Liu et al. [26] and Han et al. Han et al. [16] also raised concerns about the fragility around image watermarking and this can be partially supported from the ablation study results. The C2 comparison showed that cropping a visible logo reduced detection from 100% to 90%, confirming that once visible markers are removed, images are more likely to be vulnerable to trivial attacks. However, the “ceiling efect” slightly contradicts this as detection remained at 100% even after all watermarks were removed.

Sun et al. Sun et al. [37] and Li et al. Li et al. [22] predicted that MLLMs would fabricate justifications however the results do not align with the hallucination concerns raised in the literature. This gap highlights that current literature overestimates how common hallucinations are. It is highly likely that the studio-style, high-quality portraits used within this study do not trigger the hallucination errors. Alternatively, the result might simply show that newer model generations have genuinely improved, outgrowing the baseline reasoning flaws reported in older studies.

Similarly, the “Clever Hans” efect predicted by Maheronnaghsh and Alvanagh Maheronnaghsh and Alvanagh [29] was only partially observed. While the visible watermarks definitely served as shortcuts for the detectors, the Manual Justification Audit demonstrates that models such as GPT-5.2 classify images through their structural reasoning.

## 5.4 Limitations of the Study

Whilst the dataset was designed to be demographically balanced, it relies exclusively on high-quality, studio style portraits from the FFHQ and state-of-the-art generative sources. In a real world forensic scenario, synthetic imagery often undergoes “post-generation degradation,” including adversarial noise, low-resolution cropping, or significant social media compression. It remains unclear whether the “semantic reasoning” proficiency of detectors such as GPT-5.2 would maintain this performance under true forensic settings.

The dataset of n=400 images is suficient for a controlled comparison but small relative to the training corpus of the MLLMs themselves. This creates a risk that the accuracy results are based upon generative “flaws” in current versions of Gemini or GPT that might have been resolved by a larger dataset incorporating newer model versions.

Relying on the model’s own natural-language justifications introduces the “black box” problem. The internal logic the models use to reach those conclusions remains inaccessible which raises questions about the reliability of the justifications.

Another limitation within this study is that the FFHQ dataset itself could have been part of the training data for some of the MLLMs used within this study. This could mean detectors have prior exposure to control group images, potentially impacting accuracy.

Manuscript submitted to ACM

Finally, the models selected were state-of-the-art at time of evaluation (December 2025), however, newer releases of the ecosystems have been released, raising questions about whether the specific accuracy figures reported here remain current.

## 5.5 Future Work

Within future work, the most immediate priority would be stress-testing MLLM detectors against lower-quality imagery. Future studies should implement datasets featuring heavy JPEG compression and noise injection to identify the specific threshold at which a model’s structural reasoning begins to fail. This would provide a more realistic assessment of how these models perform in “in-the-wild” forensic settings, where high-fidelity cues are often obscured. As this study provides empirical evidence of a masking efect within invisible watermarking, expanding the ablation corpus would confirm if these findings have direct implications for forensic watermark design.

A future priority is to address the qualitative gap within the pairwise analysis. A dedicated “Pairwise Reasoning Audit” should use “Ablated Anchors” (degraded reference images). This would reveal whether the performance drop within the tier 2 experiment results are a result of the image quality, or the models ability to perform pairwise detection.

Additionally, future research should incorporate a human-baseline comparison to provide an insight into how the MLLMs and their justifications map against human forensic experts. Finally, to address any human bias within the Manual Justification Audit, future audits should employ multiple auditors to review the MLLMs justifications and then get the average for the final verdict. This would make the no-hallucination finding considerably harder to dismiss. Comparison noise remains a proposed explanation rather than a confirmed mechanism. One alternative is that Gemini’s instruction-tuning causes it to assess visual diferences between paired images diferently than it processes a single image in isolation, a hypothesis that could be isolated by replacing the FFHQ control group image with a second synthetic image in a controlled follow-up experiment.

## 6 Conclusion

Based on the results, structural reasoning proves more reliable for forensic detection than pixel-level or metadata-based approaches. While the “Gemini Paradox” and inconsistent performance accuracy across model generations show that these tools are not yet robust, the Manual Justification Audit proves that their ability to identify physical inconsistencies is a capability that improves meaningfully with model generation.

While MLLMs should not yet replace human forensic experts, they do provide auditable natural-language justifications for each classification decision, which diferentiates them from traditional CNN detectors. The masking phenomenon identified within this study demonstrates that the forensic community should reevaluate how invisible watermarking systems are used, not only for their durability against adversarial attacks but also for their impact on the structural signals that many MLLM detectors depend upon.

## References

[1] Artificial Analysis. 2025. Text to Image Leaderboard. https://artificialanalysis.ai/image/leaderboard/text-to-image Accessed: December 2025.

[2] Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, Quynh Van Do, Yan Xu, and Pascale Fung. 2023. A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity. In Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers). 675–718. doi:10.18653/v1/2023.ijcnlp-main.45

[3] Nuria Alina Chandra, Ryan Murtfeldt, Lin Qiu, Arnab Karmakar, Hannah Lee, Emmanuel Tanumihardja, Kevin Farhat, Ben Cafee, Sejin Paik Changyeon Lee, Jongwook Choi, Aerin Kim, and Oren Etzioni. 2025. Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark. arXiv preprint arXiv:2503.02857 (2025). doi:10.48550/arXiv.2503.02857

Manuscript submitted to ACM

[4] Jacob Cohen. 1988. Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates, Hillsdale, NJ.

[5] Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, Jamie Hayes, Nidhi Vyas, Majd Al Merey, Jonah Brown-Cohen, Rudy Bunel, Borja Balle, Taylan Cemgil, Zahra Ahmed, Kitty Stacpoole, Ilia Shumailov, Ciprian Baetu, Sven Gowal, Demis Hassabis, and Pushmeet Kohli. 2024. Scalable watermarking for identifying large language model outputs. Nature 634, 8035 (2024), 818–823. doi:10.1038/s41586-024-08025-4

[6] Riccardo Davì. 2023. Realistic Vision V5.1. https://huggingface.co/SG161222/Realistic\_Vision\_V5.1\_noVAE. https://huggingface.co/SG161222/ Realistic\_Vision\_V5.1\_noVAE Accessed: December 2025.

[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805 (2019). doi:10.48550/arXiv.1810.04805

[8] Oleg Evsutin and Kristina Dzhanashia. 2022. Watermarking schemes for digital images: Robustness overview. Signal Processing: Image Communication 100, Article 116523 (2022), 14 pages. doi:10.1016/j.image.2021.116523

[9] Apurva Gandhi and Shomik Jain. 2020. Adversarial Perturbations Fool Deepfake Detectors. In Proceedings ofthe 2020 International Joint Conference on Neural Networks. IEEE, 1–8. doi:10.1109/IJCNN48605.2020.9207034

[10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generativ Adversarial Networks. arXiv preprint arXiv:1406.2661 (2014). doi:10.48550/arXiv.1406.2661

[11] Google. 2025. Google Gemini API Documentation. https://ai.google.dev/docs Accessed: December 2025.

[12] Google DeepMind. 2023. Identifying AI-generated content with SynthID. https://www.deepmind.com/technologies/synthid Accessed: December 2025.

[13] Sven Gowal, Rudy Bunel, Florian Stimberg, David Stutz, Guillermo Ortiz-Jimenez, Christina Kouridi, Mel Vecerik, Jamie Hayes, Sylvestre-Alvise Rebufi, Paul Bernard, Chris Gamble, Miklós Z. Horváth, Fabian Kaczmarczyck, Alex Kaskasoli, Aleksandar Petrov, Ilia Shumailov, Meghana Thotakuri, Olivia Wiles, Jessica Yung, Zahra Ahmed, Victor Martin, Simon Rosen, Christopher Savčak, Armin Senoner, Nidhi Vyas, and Pushmeet Kohli. 2025. SynthID-Image: Image watermarking at internet scale. arXiv preprint arXiv:2510.09263 (2025). doi:10.48550/arXiv.2510.09263

[14] David Marvin Green and John Arthur Swets. 1966. Signal Detection Theory and Psychophysics. Wiley, New York. https://archive.org/details/ signaldetectiont00gree

[15] Farrokh Habibzadeh and Parham Habibzadeh. 2015. Shufle Randomization. Croatian Medical Journal 56, 4 (2015), 383–384. doi:10.3325/cmj.2015.56.383

[16] Xia Han, Qi Li, Jianbing Ni, and Mohammad Zulkernine. 2025. Robustness Assessment and Enhancement of Text Watermarking for Google’s SynthID. arXiv preprint arXiv:2508.20228 (2025). doi:10.48550/arXiv.2508.20228

[17] Michael J. Hautus, Neil A. Macmillan, and C. Douglas Creelman. 2021. Detection Theory: A User’s Guide (3 ed.). Routledge. doi:10.4324/9781003203636

[18] Jiazhou Ji, Jie Guo, Weidong Qiu, Zheng Huang, Yang Xu, Xinru Lu, Xiaoyu Jiang, Ruizhe Li, and Shujun Li. 2025. “I know myself better, but not really greatly”: How Well Can LLMs Detect and Explain LLM-Generated Texts? arXiv preprint arXiv:2502.12743 (2025). doi:10.48550/arXiv.2502.12743

[19] Shan Jia, Reilin Lyu, Kangran Zhao, Yize Chen, Zhiyuan Yan, Yan Ju, Chuanbo Hu, Xin Li, Baoyuan Wu, and Siwei Lyu. 2024. Can ChatGPT Detect DeepFakes? A Study of Using Multimodal Large Language Models for Fake Image Detection. arXiv preprint arXiv:2403.14077 (2024) doi:10.48550/arXiv.2403.14077

[20] Tero Karras, Samuli Laine, and Timo Aila. 2019. A Style-Based Generator Architecture for Generative Adversarial Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4401–4410. doi:10.1109/CVPR.2019.00453

[21] Sebastian Lapuschkin, Stephan Wäldchen, Alexander Binder, Grégoire Montavon, Wojciech Samek, and Klaus-Robert Müller. 2019. Unmasking Clever Hans predictors and assessing what machines really learn. Nature Communications 10, 1, Article 1096 (2019), 10 pages. doi:10.1038/s41467-019-08987-4

[22] Yixuan Li, Xuelin Liu, Xiaoyang Wang, Bu Sung Lee, Shiqi Wang, Anderson Rocha, and Weisi Lin. 2025. FakeBench: Probing Explainable Fake Imag Detection via Large Multimodal Models. IEEE Transactions on Information Forensics and Security 20 (2025), 8730–8745. doi:10.1109/TIFS.2025.359721

[23] Yixuan Li, Yu Tian, Yipo Huang, Wei Lu, Shiqi Wang, Weisi Lin, and Anderson Rocha. 2025. FakeScope: Large Multimodal Expert Model for Transparent AI-Generated Image Forensics. arXiv preprint arXiv:2503.24267 (2025). doi:10.48550/arXiv.2503.24267

[24] Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, and Shu Hu. 2024. Preserving Fairness Generalization in Deepfake Detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 16815–16825. doi:10.1109/CVPR52733.2024.01592

[25] Chi Liu, Huajie Chen, Tianqing Zhu, Jun Zhang, and Wanlei Zhou. 2022. Making DeepFakes More Spurious: Evading Deep Face Forgery Detection via Trace Removal Attack. IEEE Transactions on Dependable and Secure Computing 20, 6 (2022), 5182–5196. doi:10.1109/TDSC.2022.3207213

[26] Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu. 2024. Image Watermarks are Removable Using Controllable Regeneration from Clean Noise. arXiv preprint arXiv:2410.05470 (2024). doi:10.48550/arXiv.2410.05470

[27] Alexandra Sasha Luccioni, Christopher Akiki, Margaret Mitchell, and Yacine Jernite. 2023. Stable Bias: Analyzing Societal Representations in Difusion Models. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 36. 64333–64345. doi:10.48550/arXiv.2302.10893

[28] Neil A. Macmillan and C. Douglas Creelman. 2004. Detection Theory: A User’s Guide. Psychology Press, New York. https://www.routledge.com/ Detection-Theory-A-Users-Guide/Macmillan-Creelman/p/book/9780805842319

[29] Fatemeh Maheronnaghsh and Armin Sabouri Alvanagh. 2025. Detection and Explanation of AI-Generated Facial Images via Large Multimodal Models. arXiv preprint arXiv:2501.04128 (2025). doi:10.48550/arXiv.2501.04128

[30] OpenAI. 2025. OpenAI API Documentation. https://platform.openai.com/docs Accessed: December 2025.

[31] Hanwei Qian, Lingling Xia, Ruihao Ge, Yiming Fan, Qun Wang, and Zhengjun Jing. 2025. From Black Boxes to Glass Boxes: Explainable AI for Trustworthy Deepfake Forensics. Cryptography 9, 4 (2025), 61. doi:10.3390/cryptography904006

Manuscript submitted to ACM

[32] Simiao Ren, Yao Yao, Kidus Zewde, Zisheng Liang, Tsang Ng, Ning-Yau Cheng, Xiaoou Zhan, Qinzhe Liu, Yifei Chen, and Hengwei Xu. 2025. Can Multi-Modal (Reasoning) LLMs Work as Deepfake Detectors? arXiv preprint arXiv:2503.20084 (2025). doi:10.48550/arXiv.2503.20084

[33] Replicate. 2025. Replicate API Documentation. https://replicate.com/docs Accessed: December 2025.

[34] J. Ricker, S. Damm, T. Holz, and A. Fischer. 2024. Towards the Detection of Difusion Model Deepfakes. In Proceedings ofthe 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2024), Vol. 4. SciTePress, 446–457. doi:10.5220/0012422000003660

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 10684–10695. doi:10.1109/CVPR52688.2022. 01042

[36] Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. 2019. FaceForensics++: Learning to Detect Manipulated Facial Images. In Proceedings ofthe IEEE/CVFInternational Conference on ComputerVision (ICCV). 1119–1128. doi:10.1109/ICCV.2019.00009

[37] Haoran Sun, Chen Cai, Huiping Zhuang, Kong Aik Lee, Lap-Pui Chau, and Yi Wang. 2025. EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning. arXiv preprint arXiv:2510.16442 (2025). doi:10.48550/arXiv.2510.16442

[38] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. 2022. Difusers: State-of-the-art difusion models. https://github.com/huggingface/difusers. https://github.com/huggingface/difusers Accessed: December 2025.

[39] Yichi Zhang, Yao Huang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Yifan Wang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, and Jun Zhu. 2024. MultiTrust: A Comprehensive Benchmark Towards Trustworthy Multimodal Large Language Models. In Advances in Neural Information Processing Systems, Vol. 37. 4112–4133. doi:10.48550/arXiv.2406.07057

[40] Xuandong Zhao, Kexun Zhang, Zihao Huang, Jinghuai Peng, Weili Nie, Chaowei Xiao, Bo Li, and Yu-Nan Chen. 2023. Invisible Image Watermark Are Not Robust to Difusion-based Purification. arXiv preprint arXiv:2311.12871 (2023). doi:10.48550/arXiv.2311.12871

## A Appendix

## A.1 System Prompt

The following system prompt was provided to all seven MLLMs via their respective APIs:

“You are an expert digital forensics analyst. Analyse the following single image. Determine if it is a REAL photograph or an AI-GENERATED fake.”

Figure 7 shows the prompt within the detection Python script.

# - System Prompts   
OUTPUT SCHEMA = {   
"classification": "string [FAKE or REAL]",   
"confidence": "string [High/Medium/Low]"   
}   
SYSTEM PROMPT = (   
"Analyse the following single image. Determine if it is a REAL photograph or an AI-GENERATED fake. "   
f"Return strictly in JSON: {ison.dumps(OUTPUT SCHEMA)}. "  
Fig. 7. System prompt within the detection Python script.

## A.2 Statistical Results (Experiments 1 and 2)

This section presents the complete statistical results for the baseline performance metrics ofExperiment 1 and Experiment 2.

A.2.1 Experiment 1: Same- vs. Cross-Ecosystem Performance (H1). A Mann-Whitney U test was conducted to determine whether same-ecosystem detector-generator pairs (� = 10) performed significantly diferently from cross-ecosystem pairs (� = 32) in zero-shot baseline detection accuracy (Table 6).

Manuscript submitted to ACM

Table 6. Mann-Whitney U Test Results for Ecosystem Alignment (H1)
<table><tr><td>Metric</td><td>Symbol</td><td>Value</td><td>Forensic Interpretation</td></tr><tr><td>U-Statistic</td><td>U</td><td>130.0000</td><td>Rank sum difference between evaluation groups</td></tr><tr><td>p-value</td><td>p</td><td>0.3835</td><td>Not significant  $( p > 0 . 0 5 ) ;$  rejects systematic advantage</td></tr><tr><td>Effect Size</td><td>r</td><td>0.1875</td><td>Negligible-to-small rank-biserial correlation</td></tr></table>

A.2.2 Experiment 2: Baseline vs. Pairwise Context Performance (H2). Wilcoxon Signed-Rank tests were performed to compare the pairwise (TAFC) accuracy of each detector against its corresponding single-image baseline accuracy across the $N = 7$ generator sources (Table 7).

Table 7. Wilcoxon Signed-Rank Test Results for Contextual Advantage (H2)
<table><tr><td>Detector Model</td><td>W-Stat (W)</td><td>p-value</td><td>Effect Size (r)</td><td>Mean Diff.</td><td>Statistical Outcome</td></tr><tr><td>GPT-5.2</td><td>8.5</td><td>0.4062</td><td>0.3929</td><td>+3.69%</td><td>Not Statistically Significant  $( p > 0 . 0 5 )$ </td></tr><tr><td>GPT-5</td><td>6.0</td><td>0.4375</td><td>0.4286</td><td>+4.93%</td><td>Not Statistically Significant  $( p > 0 . 0 5 )$ </td></tr><tr><td>GPT-40</td><td>7.0</td><td>0.2969</td><td>0.5000</td><td>+3.03%</td><td>Not Statistically Significant  $( p > 0 . 0 5 )$ </td></tr><tr><td>Gemini 3</td><td>9.0</td><td>0.4688</td><td>-0.3571</td><td>-5.00%</td><td>Not Statistically Significant  $( p > 0 . 0 5 )$ </td></tr><tr><td>Gemini 2.5</td><td>0.0</td><td>0.0156</td><td>-1.0000</td><td>-17.09%</td><td>Statistically Significant Decline  $( p < 0 . 0 5 )$ </td></tr><tr><td>Seed 1.6</td><td>12.0</td><td>0.8125</td><td>-0.1429</td><td>-1.13%</td><td>Not Statistically Significant  $( p > 0 . 0 5 )$ </td></tr></table>

A.2.3 Description of Statistical Outcomes. The Mann-Whitney U test results above shows no statistically significant advantage for same-ecosystem pairs over cross-ecosystem pairings. With $ p = 0 . 3 8 3 5$ and a weak correlation $( r = 0 . 1 8 7 5 )$ models being part of the same ecosystem are not a reliable predictor of baseline accuracy.

In Experiment 2 (the Wilcoxon results above), descriptive gains across the GPT family were not statistically significant (such as ${ \mathrm { G P T ^ { - } } } 5 . 2 \colon p = 0 . 4 0 6 2 , r = 0 . 3 9 2 9 , + 3 . 6 9 \%$ . Gemini 2.5 had a significant performance decline $( W = 0 . 0 , p = 0 . 0 1 5 6 ,$ $r = - 1 . 0 0 0 0 )$ . The −17.09% accuracy drop indicates that a comparative reference baseline can act as ‘comparison noise which leads to a decline in performance accuracy (Figure 8).

Manuscript submitted to ACM

## A.3 Ablation Study Statistical Results

Dissertation Statistical Analysis: Hybrid Non-Parametric Framework   
Test: 1 vs 3: Stability Check (Independent Sets)   
Method: Mann-Whitney U (Unpaired)   
Sample Size: n\_a=50, n\_b=50   
P-Value: 0.4354   
Interpretation: Not Significant   
Test: 2 vs 3: Role of Visible Watermark (Both vs Invisible Only)   
Method: Wilcoxon Signed-Rank (Paired)   
Sample Size: n=50   
P-Value: 0.0075   
Interpretation: Significant   
Test: 2 vs 4: Role of SynthID (Both vs Visible Only)   
Method: Wilcoxon Signed-Rank (Paired)   
Sample Size: n=50   
P-Value: 0.0455   
Interpretation: Significant   
Test: 4 vs 5: Role of Visible Watermark (Visible Only vs Control)   
Method: Wilcoxon Signed-Rank (Paired)   
Sample Size: n=50   
P-Value: 1.0000   
Interpretation: Not Significant   
Test: 3 vs 5: Role of SynthID (Invisible Only vs Control)   
Method: Wilcoxon Signed-Rank (Paired)   
Sample Size: n=50   
P-Value: 0.0020   
Interpretation: Significant  
Fig. 8. Full non-parametric Wilcoxon and Mann-Whitney U statistical results for the Gemini ablation study.