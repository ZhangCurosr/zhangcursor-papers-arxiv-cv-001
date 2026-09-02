# An Intelligent Decision Support System for Emotion Monitoring using Microscopic Fixational Dynamics

Xiangyu Shen<sup>a,∗</sup>, Feiyang Deng<sup>b</sup>, Zijian Dai<sup>a</sup>, Aibin Chen<sup>a</sup>, Jizheng Yi<sup>a</sup>, Jie Li<sup>c</sup> and Hongbo Jiang<sup>d</sup>

<sup>a</sup>College ofAdvanced Interdisciplinary Studies, Central South University ofForestry and Technology, Changsha, 410004, Hunan, China

<sup>b</sup>Chow Yei Ching School of Graduate Studies, City University of Hong Kong, Hong Kong, 999077, Hong Kong

<sup>c</sup>College ofComputer and Mathematics, Central South University ofForestry and Technology, Changsha, 410004, Hunan, China

<sup>d</sup>College of Computer Science and Electronic Engineering, Hunan University, Changsha, 410082, Hunan, China

## A R T I C L E I N F O

Keywords:   
Emotion monitoring   
Afective computing   
Fixational micro-movements   
Multi-head attention   
Support vector machine   
Explainable artificial intelligence

## A BS T RA C T

The rising prevalence of psychological disorders necessitates efective emotion monitoring, yet current methods relying on facial or physiological signals often sufer from intrusiveness and privacy issues. This paper proposes an intelligent decision support system and pervasive edge-computing framework that leverages smart glasses and a companion smartphone to infer emotional states from microscopic visual fixation patterns. Moving beyond traditional macroscopic gaze metrics, the proposed system extracts and decomposes three distinct neurophysiological micro-movements: microsaccades, ocular drifts, and ocular microtremors. We introduce an interpretable hybrid artificial intelligence pipeline—combining a multi-head attention mechanism, extreme gradient boosting, and a support vector machine—to extract deep temporal features, quantify their physiological importance, and perform eficient on-device classification. Through an extensive evaluation involving 60 volunteers, we rigorously validate the framework under a strict leave-one-subject-out cross-validation protocol across both controlled and naturalistic mobile scenarios. Ablation studies unequivocally demonstrate that these fixational micro-movements are substantially more discriminative for emotion inference than traditional macroscopic features. Furthermore, aligned with contemporary afective science, the system incorporates a few-shot personalization mechanism to bridge universal physiological baselines with individual emotional heterogeneity, achieving a highly robust personalized F1-score of 83.6%. This work establishes a physiologically interpretable, unobtrusive, and deployable paradigm for continuous real-time emotion monitoring.

## 1. Introduction

Emotion plays a fundamental role in human behavior and interaction, influencing various physiological responses, including heart rate, pupil dilation, and eye movement(Khare, Blanes-Vidal, Nadimi and Acharya, 2024). Emotion recognition technology has become increasingly important for enhancing user experience, improving mental health monitoring, and enabling adaptive applications in human-computer interaction. However, traditional methods of emotion recognition, such as facial expression recognition and voice analysis, present challenges in privacy and practicality for realworld use. These methods often require direct observation of facial or vocal cues, which may be intrusive and limit user privacy(Canal, Müller, Matias, Scotton, de Sa Junior, Pozzebon and Sobieranski, 2022; Li, Zhang, Tiwari, Song, Hu, Yang, Zhao, Kumar and Marttinen, 2022; Zhang, Yang, Chen, Zhang, Leng and Zhao, 2024; Liu, Gao, Li, Fu and Ding, 2023b).

In this context, we raise a question: can we capture users’ emotions through spontaneous physiological activities without violating their privacy? Recent studies indicate that when humans are in a state of fixation, their eyes are not entirely stationary; instead, they exhibit three types of subtle eye movements—microsaccades, ocular drifts, and ocular microtremors—that are closely associated with autonomic emotional states (Rucci and Poletti, 2015; Alexander and Martinez-Conde, 2019; Rucci, McGraw and Krauzlis, 2015; Yang, Huang, Chen, Li and Hasegawa, 2023). In other words, these previously overlooked fixation patterns contain emotion-related physiological markers that can be extracted and analyzed. Simultaneously, the ubiquity of smart glasses is growing rapidly. According to the International Data Corporation (IDC), global smart glasses shipments in the first quarter of 2025 reached 1.487 million units, an 82.3% yearon-year increase (IDC, 2025). Therefore, we consider leveraging the built-in cameras of smart glasses to record users microscopic fixation patterns during daily wear, enabling unobtrusive emotion monitoring. If successfully implemented, this fixation-pattern-based mobile paradigm ofers unique advantages:

• Non-intrusive and privacy-preserving: Unlike methods relying on facial expressions or voice recordings, analyzing subtle fixational micro-movements avoids capturing sensitive environmental or identifiable biometric data, providing a highly privacy-conscious approach.

• Fine-grained, physiologically-grounded detection: Macroscopic behaviors (e.g., facial expressions) can be consciously suppressed or faked. In contrast, microscopic eye movements are involuntary autonomic reflexes. Analyzing these patterns enables the detection of nuanced physiological shifts in the valencearousal space (Kollias, 2022; Yik, Mues, Sze, Kuppens, Tuerlinckx, De Roover, Kwok, Schwartz, Abu-Hilal, Adebayo et al., 2023; Martínez, Molero, González, Conde, Brysbaert and Reviriego, 2025).

• Real-world mobile applicability: Utilizing commodity smart glasses allows for continuous emotion monitoring in diverse everyday contexts, from casual social interactions to immersive virtual experiences, without requiring highly specialized or intrusive medicalgrade sensors.

Achieving this goal, however, entails significant technical and theoretical challenges. First, from a systems perspective, extracting millisecond-level micro-movements typically requires heavy computational resources, which contradicts the severe power and thermal constraints of wearable devices. Second, from an algorithmic perspective, deep learning models often act as "black boxes," making it difficult to interpret which specific physiological features actually drive the emotion classification. Finally, contemporary afective science—such as the Theory of Constructed Emotion (Barrett, 2017)—posits that emotions are not strictly universal but exhibit significant inter-individual variability, complicating the deployment of "one-size-fits-all" models in the wild.

To address these challenges, we propose EmoGaze, an intelligent edge-based expert system that pioneers the extraction and classification of microscopic fixation patterns for emotion monitoring. Unlike conventional pervasive sensors, EmoGaze is designed as an Intelligent Decision Support System (IDSS) tailored for domains requiring unobtrusive afective profiling, such as clinical psychological assessment, driver vigilance monitoring, and adaptive humancomputer interaction. To ensure deployability, EmoGaze partitions the workload: the smart glasses function purely as a lightweight sensor node, transmitting video streams to a companion computing hub. Algorithmically, we engineer an interpretable expert pipeline combining a Multi-Head Attention (MHA) mechanism, XGBoost, and a Support Vector Machine (SVM). This hybrid architecture not only achieves high accuracy but also acts as an "explainable AI (XAI)" module, quantitatively revealing the physiological importance of each micro-movement modality to domain experts.

We have prototyped EmoGaze and conducted extensive evaluations involving 60 volunteers across both controlled and naturalistic mobile scenarios. Evaluated under a rigorous LOSO-CV protocol, EmoGaze demonstrates robust zero-shot generalization to unseen individuals, with performance improving significantly to an F1-score of 83.6% <sup>1</sup> upon applying minimal few-shot personalized calibration. In summary, our main contributions are as follows:

• Conceptual Innovation via Microscopic Signal Definition: This research fundamentally reframes the input signal for emotion recognition. By decomposing fixation into its constituent neurophysiological components (microsaccades, ocular drifts, and microtremors), and validating their superiority over macroscopic gaze features through extensive modality ablation studies, we establish that this overlooked data stream contains robust, highly discriminative markers for complex emotional states.

• Edge-Centric, Interpretable Hybrid Architecture: We introduce a novel MHA-XGBoost-SVM pipeline meticulously designed for mobile edge computing. This architecture optimally balances deep non-linear representation power with deployment eficiency. Crucially, by utilizing XGBoost for feature importance ranking, we add a transparent layer of physiological interpretability to the "black box" attention features, empirically mapping specific micro-movements to high- and low-arousal states.

• Explainable Expert System Paradigm: Challenging the prevailing trend of "black-box" deep learning, EmoGaze demonstrates a highly interpretable expert system paradigm. We rigorously evaluate the system under a strict LOSO-CV protocol. By incorporating a few-shot personalization mechanism and explicitly extracting physiological features, our framework aligns with contemporary afective science, effectively bridging universal physiological baselines with individual emotional heterogeneity. This provides domain experts (e.g., clinicians, HCI designers) with transparent, data-driven insights into human affective states.

The rest of this paper is organized as follows. We review related work in Sec. 2. The physiological relationship between fixation patterns and emotions is explored in Sec. 3. We detail the EmoGaze system architecture in Sec. 4. Performance and system overhead evaluations are presented in Sec. 5. Limitations and future directions are discussed in Sec. 6, and we conclude in Sec. 7.

## 2. Related Work

In this section, we review the eforts of researchers in mobile emotion detection and provide a comprehensive overview of EmoGaze’s advantages compared to state-ofthe-art works.

## 2.1. Diverse Emotion Detection

In recent years, emotion detection has gained considerable attention, particularly in contexts like humancomputer interaction, mental health monitoring, and user experience optimization (Acheampong, Nunoo-Mensah and Chen, 2021; Mao, Liu, He, Li and Cambria, 2022; Nie, Bao, Zhao and Liu, 2023). Researchers have explored various approaches to detect emotions, utilizing modalities such as facial expressions, speech, physiological signals, eye movements, and multimodal data (Deng and Ren, 2020; Awais,

Raza, Singh, Bashir, Manzoor, Islam and Rodrigues, 2020; Zhang, Li, Ying, Li, Tang and Lu, 2020; Chen, Cao, Yao, Lu, Peng, Mei and Liu, 2021b; Arun, Maheswaravenkatesh and Jayasankar, 2023; Kulkarni and Dixit, 2025).

Facial expression recognition remains one of the most widely used methods, leveraging computer vision techniques such as convolutional neural networks (CNNs) (Zhang, 2018; Li, Jin, Zhou, Kubota and Ju, 2020). However, facial expression-based methods face severe limitations when dealing with face occlusion, varied lighting conditions, or individuals intentionally suppressing their expressions (Zhang, 2018). Speech-based emotion recognition utilizes features like tone and pitch (Khalil, Jones, Babar, Jan, Zafar and Alhussain, 2019). Yet, it is continuously challenged by environmental background noise, dialect variations, and the lack of continuous verbal communication in daily life. Physiological signal-based methods (e.g., ECG, EMG, EEG) ofer high precision by directly measuring bodily responses (Song, Lu and Yan, 2020). However, they are inherently intrusive, requiring specialized medical-grade equipment that severely limits their deployment in pervasive mobile applications (Yin, Zhao, Wang, Yang and Zhang, 2017).

Eye-tracking data has emerged as a highly promising modality, providing unobtrusive insights into emotional states. Recent works have demonstrated that macroscopic eye movement features, such as fixation duration, saccade amplitude, and pupil dilation, are sensitive indicators of emotional responses (Abdou, Sood, Müller and Bulling, 2022). However, macroscopic gaze metrics are often heavily confounded by external visual tasks, ambient lighting (afecting the pupil), and cognitive load, which complicates reliable emotion detection in unconstrained real-world settings (Wu, Dey, Shi, Crespo and Sherratt, 2021). Furthermore, multimodal approaches (Zhang et al., 2024; Lv, Chen, Huang, Duan and Lin, 2021; Middya, Nag and Roy, 2022) attempt to fuse these diverse signals to improve robustness. Unfortunately, collecting and synchronously processing data from multiple high-bandwidth sensors (e.g., cameras and EEG) poses prohibitive computational overhead for mobile and edge computing platforms.

In contrast to the aforementioned methods, this work introduces a novel approach based on the fine-grained microscopic patterns of visual fixations. Crucially, while macroscopic expressions (e.g., facial muscle movements or large saccades) can be consciously controlled or easily disrupted by task demands, the microscopic dynamics within a single fixation are involuntary autonomic reflexes. By extracting these subtle, task-independent physiological markers, we aim to establish a robust and computationally lightweight framework for continuous, on-the-go emotion monitoring.

## 2.2. Connecting Fixation with Emotion and Contemporary Afective Science

In recent years, the relationship between eye movement patterns and emotional states has gained considerable attention. While several studies have explored gaze patterns (Hadders-Algra, 2022; Cuve, Castiello, Shiferaw,

Ichijo, Catmur and Bird, 2021; Nag, Haber, Voss, Tamura, Daniels, Ma, Chiang, Ramachandran, Schwartz, Winograd et al., 2020; Black, Chen, Lipp, Bölte and Girdler, 2020; Le, Kou, Zhao, Fu, Zhang, Becker and Kendrick, 2020), most focus on macroscopic, behavioral gaze trajectories. A key contribution in this domain is the work by Yitzhak et al. (Yitzhak, Pertzov, Guy and Aviezer, 2022), demonstrating that emotional states modulate overall visual attention and saccadic patterns.

However, mapping complex emotions directly from macroscopic behaviors is being increasingly challenged by contemporary afective science. Notably, the Theory of Constructed Emotion (Barrett, 2017) argues that emotions are not universally fixed, rigid templates (e.g., a specific facial expression always meaning "anger"). Instead, emotions are highly individualized, context-dependent phenomena constructed from deeper, low-level physiological states of arousal and valence.

Addressing this paradigm shift, our approach deliberately moves away from macroscopic gaze behavior. Instead, we hypothesize that the eye undergoes subtle yet measurable involuntary movements—namely microsaccades, ocular drifts, and ocular microtremors—that serve as the fundamental physiological substrate for arousal and valence. Microsaccades, small involuntary jerks, have been linked to cognitive load and autonomic arousal shifts (Liu, Nobre and van Ede, 2022; Yu, Herman, Katz and Krauzlis, 2022; Liu, Nobre and van Ede, 2023a). Ocular drifts have been shown to correlate with perceptual processing and attentional disengagement (Malevich, Buonocore and Hafed, 2020; Khademi, Zhang, Baumann, Malevich, Yu and Hafed, 2024). Ocular microtremors are believed to be a direct byproduct of the central nervous system’s regulation (Graham, Das, Vitorio, McDonald, Walker, Godfrey, Morris and Stuart, 2023; Leigh and Martinez-Conde, 2013; Bolger, Bojanic, Sheahan, Coakley and Malone, 1999).

By decomposing and analyzing these three fundamental micro-movements, EmoGaze captures the purest, low-level physiological indicators of emotion. This aligns seamlessly with modern afective theories: rather than assuming a "onesize-fits-all" mapping, we extract robust underlying physiological markers that, when coupled with brief personalized calibration, can accurately capture the user’s uniquely constructed emotional state.

## 2.3. Gaze-based Emotion Recognition

To provide a clear overview of EmoGaze in relation to recent advancements in gaze-based emotion recognition, Table 1 presents a comparative summary. It highlights key aspects such as core methodologies, specific data modalities, and architectural limitations. Due to considerable heterogeneity in experimental designs (e.g., many studies lacking strict subject-independent cross-validation), direct comparison of accuracy figures is omitted. Instead, the primary aim of this table is to delineate the unique contributions of EmoGaze: it is a pioneering single-modality system that extracts deep temporal features from previously overlooked microscopic fixational dynamics, specifically engineered for deployment on mobile edge-computing platforms (e.g., smart glasses and companion smartphones).

Sadness  
Pleasure  
Anger  
Gaze-based Emotion Recognition Methods
<table><tr><td>Related Works</td><td>Method</td><td colspan="2">Data Modality</td><td>Limitations/Advantages</td></tr><tr><td>Ashwaq Alhargan et al. (Alhargan, Cooke and Binjammaz, 2017)</td><td>Support Vector Machine + multimodal fusion</td><td colspan="2">Eye tracking Speech</td><td>Needs speech; shallow features</td></tr><tr><td>Bere (Zhu, Wang, Darmanola, Guo, Jin and Li, 2024)</td><td>Graph Convolutional Network Cross-Modal Domain Adaptation</td><td colspan="2">Eye tracking Head movement</td><td>Non-emotion focus; lacks fine-grained eye modeling</td></tr><tr><td>VREED (Tabbaa, Searle, Bafti, Hossain, Intarasisrisawat, Glancy and Ang, 2021)</td><td>Statistics + traditional machine learning</td><td colspan="2">Eye tracking ECG GSR</td><td>Multimodal required; severe wearable hardware burden</td></tr><tr><td>Vehlen et al. (Vehlen, Kellner, Normann, Heinrichs and Domes, 2023)</td><td>Clinical study and oxytocin intervention</td><td colspan="2">Eye tracking</td><td>No algorithmic modeling: clinical environment only</td></tr><tr><td>EyeSyn (Lan, Scargill and Gorlatova, 2022)</td><td>Psychology-driven synthetic modeling</td><td colspan="2">Synthetic eye tracking EEG</td><td>Synthetic data only; lacks real-world evaluation</td></tr><tr><td>Wang Kay Ngai et al. (Ngai, Xie, Zou and Chou, 2022)</td><td>CNN with multimodal fusion</td><td colspan="2">Facial images Eye tracking</td><td>Heavy model; requires multiple intrusive sensors</td></tr><tr><td>EmoGaze (Ours)</td><td>Positional encoding + multi-head attention</td><td>Fixation</td><td>OMT</td><td>Lightweight and deployable on mobile edge devices; Single-modality (dual-eye cameras) non-intrusive setup; Fine-grained global fixation modeling via attention</td></tr></table>

![](images/073eef828f8c5265dcb0e03782a6e6808b8f846ae515880442c0b1387ea6fb18.jpg)

![](images/06aa2bbf102223f98861620b34a6e5861f2b6a0a088464c522f87236c17973b0.jpg)

![](images/8ded053f07258f0757d36b6cad197ff15d940ac0c73696864617e9a12e6ce777.jpg)

![](images/c465fb3b79735c64a51f6b15da741c4f321a0bc5906bff90939068266eddd123.jpg)

![](images/bf581e2d76814f97b8add28a4bd7d6f9bd7f40d5e7c8d93b2e68a4ed1641708b.jpg)

![](images/6fc3c1d3e0f6be6d3886b77ac3bf9a05e6a3b399ae5d817d65a0a2964ebb17bc.jpg)

![](images/b3aac5a3166b1179a51fc425daf18dddf1516b22a28473b7027a9bd90d993519.jpg)

![](images/a6a07d074e7278304f375bf936f869ec0a4299423c160b06ac2d02338e19c4bf.jpg)  
Figure 1: Amplitude (degree) and frequency (Hz) of microsaccade (MS), ocular drift (OD), and ocular microtremor (OMT) under four basic emotions of joy, anger, sadness and pleasure. The frequency and amplitude of eye movement signals vary with diferent emotional states. High-frequency and high-amplitude signals are observed during anger, indicating heightened activity, while sadness shows lower frequency and amplitude, reflecting reduced activity. Joy and pleasure states have moderate signal levels, showing stable eye movements. The variability in eye movement signals is higher in anger, joy, and pleasure, but more stable in sadness.

## 3. Motivation

In this section, we explore the intricate relationship between eye fixation patterns and emotional states from physiological and neuroscientific perspectives, conducting a qualitative pilot study to establish this connection.

Emotional cognition research posits that ocular motion metrics—such as microsaccades, ocular drifts, and ocular microtremors—ofer significant insights into the underlying states of autonomic arousal and valence (Kashihara, Okanoya and Kawai, 2014; Krejtz, Żurawska, Duchowski and Wichary, 2020; Lin, Intoy, Clark, Rucci and Victor, 2023; Bolger, Bojanic, Sheahan, Coakley and Malone, 2001; Simola, Le Fevre, Torniainen and Baccino, 2015). For instance, microsaccades, which are rapid, minute eye movements, exhibit increased frequency and amplitude during states of heightened emotional arousal, typically associated with anger and joy (Kashihara et al., 2014; Chen, Yep, Hsu, Cherng and Wang, 2021a; Krejtz et al., 2020). This increase reflects the autonomic nervous system’s response to intense stimuli, serving as physiological correlates that mirror the central nervous system’s activation. Conversely, ocular drifts—slow, involuntary eye movements—are observed to be more pronounced during emotional states characterized by low arousal and valence, such as sadness (Aston-Jones and Cohen, 2005; Russell, 1980; Nolen-Hoeksema, Wisco and Lyubomirsky, 2008). This phenomenon may be attributed to a reduced capacity for visual fixation control in depressed emotional states, reflecting a visual disengagement from external stimuli and a decrease in attentional focus.

Ocular microtremors, though less extensively studied compared to other ocular metrics, represent continuous, extremely small, high-frequency oscillations of the eyes. These involuntary movements are believed to reflect underlying cortical activity, suggesting a potential association with emotional fluctuations. Given their link to central nervous system processes, ocular microtremors provide valuable insight into the subtle physiological dynamics accompanying shifts in emotional states. Preliminary evidence indicates that changes in the frequency and amplitude of these tremors can be indicative of shifts in arousal and attentional focus (Graham et al., 2023; Robertson and Timmons, 2007; Bolger et al., 2001; Ryle, Ryle, Al-Kalbani, Collins, Gopinathan, Boyle, Coakley and Sheridan, 2009).

While contemporary afective science, such as the Theory of Constructed Emotion (Barrett, 2017), emphasizes that final emotional experiences are context-dependent and exhibit inter-individual variability, these autonomic ocular micro-movements provide a universally grounded physiological substrate. By systematically mining these intricate physiological behaviors, we can capture high-fidelity markers of arousal and valence, setting a robust baseline that can subsequently accommodate personalized emotion construction.

To empirically illustrate how these fixational micropatterns correlate with emotional states, we recruited an independent cohort of 14 volunteers (7 males and 7 females, aged from 20 to 53; completely distinct from the 60 subjects used in the main system evaluation to strictly prevent data leakage) to measure their fixation patterns while experiencing discrete emotions.

Specifically, we observe that in the anger state, the frequency and amplitude of all three eye movement signals are relatively high, reflecting the heightened physiological activity during extreme arousal. In contrast, during the sadness state, the frequency and amplitude of the signals are significantly lower, indicating a reduction in eye movement stability during emotional downturns. In thejoy and pleasure states, the signals maintain moderate levels, representing stable ocular patterns under positive conditions. Moreover, the width of the box plots (i.e., variance) demonstrates that signal variability is higher in high-arousal states (anger, joy, pleasure), whereas sadness exhibits a more suppressed and concentrated eye movement behavior.

This analysis confirms a definitive correlation between these three micro-eye movement signals and underlying emotional states, validating our premise that microscopic fixational dynamics contain rich, discriminative emotion indicators. Therefore, it is theoretically feasible and scientifically rigorous to infer users’ emotions through deep algorithmic mining of their fixation patterns.

In the following section, we elaborate on the design of EmoGaze, an edge-centric hybrid deep-learning framework tailored to extract and classify these subtle fixation micromovements.

## 4. System Design

## 4.1. Design Overview

The system overview of EmoGaze is illustrated in Fig. 2. To align with the power and thermal constraints of wearable technology, EmoGaze adopts an edge-computing architecture that partitions tasks between a sensor node and a computational hub. The framework consists of five main modules: Data Collection, Data Preprocessing, Fixation Decomposition, Feature Extraction & Ranking, and Emotion Classification.

In Data Collection, the raw ocular data are acquired by our custom-designed smart glasses (Fig. 3). Functioning purely as a high-fidelity sensor node, the glasses capture cropped Region of Interest (ROI) infrared video streams of both eyes. These video streams are transmitted via highbandwidth Wi-Fi Direct (802.11ax) to a companion smartphone acting as the edge hub. In Data Preprocessing, the smartphone first utilizes its Digital Signal Processor (DSP) to extract the X, Y coordinates and rotational angles of the pupils using a sub-pixel tracking algorithm. The system then filters outliers, removes unreasonable fixation distances, applies low-pass filtering, and labels the data. Building on this, in Fixation Decomposition, specific algorithmic strategies extract the three subtle eye movement signals: ocular drifts, ocular microtremors, and microsaccades. In Feature Extraction & Ranking, positional encoding and a multihead attention mechanism extract deep temporal features from these sequences. An XGBoost model is then utilized to evaluate feature importance and assign corresponding weights. Finally, in Emotion Classification, a Support Vector Machine (SVM) utilizes these weighted features to infer emotional states. We elaborate on the algorithmic modules executed on the edge device in the following sections.

![](images/9b3ab8690007a55e0f85556a020e22128941ab0bf9e68b39ec9009adaee15de3.jpg)  
Figure 2: EmoGaze involves five modules. Data Collection: dual-eye ROI infrared video streams are captured by our customdesigned eye-tracking glasses and transmitted to the edge device. Data Preprocessing: X, Y coordinates are extracted via DSP, and raw fixation data are filtered according to physiological characteristics. Fixation Decomposition: the preprocessed fixation sequences are decomposed into three subtle eye movement signals by our detection algorithm. Feature Extraction & Ranking: features are extracted through positional encoding and multi-head attention mechanisms, and weighted by XGBoost. Emotion Classification: emotions are classified by an SVM based on the weighted features.

## 4.2. Data Preprocessing

After obtaining the fixation data, EmoGaze further performs data preprocessing, which includes removing outliers of left and right pupil angles, eliminating points with unreasonable fixation distances, applying low-pass filtering, and labeling the data.

## 4.2.1. Initial Removal of Outliers

Since our device can obtain the rotational angle data of both left and right eyes (as shown by � and � in Fig. 2), we can remove abnormal rotational angle values. As illustrated in Fig. 2, based on the triangle’s exterior angle being equal to the sum of the non-adjacent interior angles, � must be greater than �. Therefore, we can remove fixation point data where $\alpha \leq \beta$ to initially eliminate outliers.

## 4.2.2. Fixation Distance Filtering

As shown in Fig. 2, we calculate the distance � from the fixation point to the eyes using the parallax method commonly used in astronomy(Yu, Li, Gao, Yan, Li, Wang and Sang, 2024). Assuming the interocular distance � is known, based on the properties of triangles, we can obtain $\begin{array} { r } { D = { \frac { p \cdot \tan ( \alpha ) \cdot \tan ( \beta ) } { \tan ( \alpha ) + \tan ( \beta ) } } } \end{array}$ . Therefore, we remove points with unreasonable fixation distances (those exceeding three times the standard deviation). This is because the human fixation point cannot undergo very large sudden changes.

## 4.2.3. Low-Pass Filtering

Since the frequency of eye movement signals is generally below 104Hz(Barea, Boquete, Mazo and López, 2002), we set a low-pass filter to reset the frequency coeficients above 104Hz to 0, thereby filtering out high-frequency noise. The signal is then converted from the frequency domain back to the time domain.

## 4.2.4. Labeling and Expert Ground Truth Protocol

To ensure the clinical and psychological validity of the emotion labels, participants completed the Positive and Negative Afect Schedule (PANAS) (Watson, Clark and Tellegen, 1988) alongside the Self-Assessment Manikin (SAM) (Bradley and Lang, 1994) arousal scale immediately following each stimulus trial. PANAS provides two independent scores (Positive Afect, PA, and Negative Afect, NA), while SAM quantifies the physiological arousal level.

To map these continuous scores to our four discrete emotion categories, we utilize a mapping protocol informed by the Circumplex Model of Afect (Russell, 1980). Using a subject-specific median split to establish thresholds (for practical deployment involving completely new users, a global population median is utilized during the initial coldstart phase before transitioning to individualized thresholds during few-shot calibration), the emotions are categorized as follows: Joy is characterized by high PA and high arousal; Pleasure represents high PA but low/moderate arousal; Anger maps to high NA and high arousal; and Sadness corresponds to high NA and low arousal. To eliminate ambiguity, participants also provided a forced-choice categorical label (Joy, Pleasure, Anger, Sadness). Samples where the categorical label contradicted the PANAS/SAM dimensional mapping are marked as ambiguous and discarded, yielding a highly reliable, expert-grade ground truth dataset for model training.

## 4.3. Fixation Decomposition and Sensing Fidelity

In Sec. 3, we introduce the three types of eye movement signals associated with fixation: microsaccades, ocular drifts, and ocular microtremors. In this section, we detail the separation of these signals from the raw fixation sequence and rigorously validate the sensing fidelity of our decomposition pipeline.

## 4.3.1. Sensing Fidelity and Noise Floor Calibration

A critical challenge in tracking fine-grained micro-eye movements on mobile wearables is distinguishing genuine physiological signals from hardware noise, especially given the minute amplitudes of Ocular Microtremors (OMT). While our cameras operate at a nominal resolution of 360p to conserve power and bandwidth, we employ a widelyvalidated sub-pixel center-of-mass pupil tracking algorithm (Kassner, Patera and Bulling, 2014; Świrski, Bulling and Dodgson, 2012) that enhances the efective angular resolution to less than 0.05<sup>◦</sup>.

Furthermore, to unequivocally prove that the extracted high-frequency components are physiological rather than instrumental artifacts, we conduct a baseline noise floor calibration using a stationary artificial eye model under identical lighting conditions. Spectral analysis revealed that the systemic hardware and algorithmic jitter produced a baseline noise floor Power Spectral Density (PSD) of $<$ $1 0 ^ { - 4 } ~ \mathrm { d e g } ^ { 2 } / \mathrm { H z }$ . In contrast, analysis of our in-vivo data confirmed that the extracted OMT (40–100Hz) exhibited peak PSDs in the range of $1 0 ^ { - 2 } ~ \mathrm { d e g } ^ { 2 } / \mathrm { H z }$ . This yields a Signal-to-Noise Ratio (SNR) exceeding 20 dB, empirically confirming that the kinematic amplitudes of EmoGaze’s extractions capture genuine autonomic ocular reflexes rather than noisy proxies.

## 4.3.2. Ocular Drifts and Microtremors Extraction

Having verified the signal fidelity, we first extract the ocular drifts (OD) and ocular microtremors (OMT). Ocular drifts are characterized by slow, smooth motions generally occurring within the 0–40Hz frequency range, whereas ocular microtremors are high-frequency, low-amplitude oscillations typically falling within the 40–100Hz range (Ahissar, Arieli, Fried and Bonneh, 2016). Because our 500fps camera yields a Nyquist frequency of 250Hz, it safely encompasses these physiological bands. We apply dedicated, high-order Butterworth bandpass filters to the preprocessed positional signals to isolate these respective components without phase distortion.

## 4.3.3. Microsaccade Extraction

Microsaccades are rapid, corrective eye movements that maintain visual clarity (Martinez-Conde, Macknik, Troncoso and Hubel, 2009). We extract them based on precise velocity and acceleration thresholding. This process is fundamentally inspired by established microsaccade detection baselines (e.g., the robust Engbert & Kliegl algorithm principles (Engbert and Kliegl, 2003)) and is optimized for our mobile pipeline:

1) Velocity Calculation: Eye movement velocity is obtained by diferentiating the position signals. The instantaneous velocity $v _ { t }$ is calculated as Eq. 1 shows:

$$
v _ { t } = \frac { \sqrt { \left( x _ { t } - x _ { t - 1 } \right) ^ { 2 } + \left( y _ { t } - y _ { t - 1 } \right) ^ { 2 } } } { \Delta t } ,\tag{1}
$$

where $x _ { t } , y _ { t }$ and $x _ { t - 1 } , y _ { t - 1 }$ represent the fixation coordinates at the current moment � and preceding moment $t - 1 , \Delta t$ is the reciprocal of the sampling rate. The velocity is mapped to angular degrees per second ${ \bf \Xi } ( { \bf \Lambda } ^ { \circ } / s )$ . This methodology adheres to the widely adopted velocity threshold (I-VT) principles (Olsen and Matos, 2012).

2) Velocity Filtering: To further isolate genuine microsaccades from transient high-frequency noise (e.g., blink artifacts), a sliding-window mean filter is applied to smooth the raw velocity data, defined in Eq. 2:

$$
v _ { \mathrm { f i l t e r d } } = \frac { 1 } { N } \sum _ { i = 0 } ^ { N - 1 } v _ { t - i } ,\tag{2}
$$

where � denotes the sliding window sample size (Suthaharan, MW, Zhang and Rossi, 2023).

3) Event Screening and Detection: We utilize a peakfinding algorithm to detect local velocity maxima. Because genuine physiological microsaccades exhibit very specific and short temporal durations (typically 40–70 ms (Taylor, Buonocore and Fracasso, 2024)), detected peaks failing to meet this strict temporal duration constraint are discarded.

4) Event Verification: To strictly diferentiate microsaccades from deliberate macroscopic saccades, we evaluate instantaneous acceleration $\left( a _ { t } \right)$ , computed via second-order diferentiation (Eq. 3):

$$
a _ { t } = \frac { v _ { t } - v _ { t - 1 } } { \Delta t } .\tag{3}
$$

Microsaccades exhibit smoother acceleration profiles, whereas macroscopic saccades produce abrupt spikes. Applying a slope threshold efectively separates these distinct mechanisms (Møller, Laursen, Tygesen and Sjølie, 2002). As an external validation of our pipeline, the amplitude and duration distributions of the extracted MS perfectly align with established physiological norms (e.g., amplitudes < 1<sup>◦</sup>). Fig. 4 illustrates the successful decomposition of these signals.

## 4.4. Feature Extraction & Ranking

## 4.4.1. Feature Extraction

Although the coarse relationship between the three eye movement signals and emotional states has been mentioned in Sec. 3, it is challenging to directly determine emotional states through the sequence of these three eye movement signals. Therefore, we intend to explore the use of feature extraction techniques to identify features related to emotional states.

Since the three eye movement sequences obtained earlier are essentially time series, and the multi-head attention mechanism is well-suited for processing time series, the multi-head attention mechanism is proposed to extract emotion-related features from the three eye movement sequences.

![](images/2952c2490fbcff967c20b27364dd28e52239ffe483d6e74afb88020a7b8b584b.jpg)  
Figure 3: Eye tracking equipment and corresponding software.

![](images/7931228982965bfb8038b25ee37b1067553562198d25b010a2a23faf3d6e26c9.jpg)

![](images/580ec8db93f1cf66918c8d6ab13779c69efa8f44520cd41edaaf3a6b29078045.jpg)

![](images/ca823976eefa445abd7860165060efee608026461571e2976567732b4b3530f1.jpg)

![](images/af570d2bc5ea7b7522d5ed26b1d95a3ad908d8766d1157394b2081b04670d5a2.jpg)

![](images/1718d5250fe991bf63ee591aa7cd918e05fce3c9cee70a3c487a4108afa65439.jpg)

![](images/afa2ae9a86024840fa1d74288fb61821bded08ee6b0698b5ac4058c4f0a41084.jpg)  
Figure 4: Enhanced illustration of microsaccade detection, showcasing the efectiveness of our designed algorithm in distinguishing between microsaccades and saccades across multiple subfigures (a-f).

As shown in Fig. 5, we propose a customized feature extraction framework that mining the three decomposed physiologically meaningful micro-movement components:

Ocular Drift (OD), Microsaccades (MS), and Ocular Microtremor (OMT). This decomposition enables fine-grained modeling of distinct motion patterns within fixation signals.

Specifically, given an input fixation sequence of length �, we first decompose it into three independent feature matrices: $F _ { O D } \in \mathbb { R } ^ { n \times d _ { O D } } , F _ { M S } \in \mathbb { R } ^ { n \times d _ { M S } } , F _ { O M T } \in \mathbb { R } ^ { n \times d _ { O M T } }$ where $d _ { O D } , d _ { M S }$ and $d _ { O M T }$ represent the original feature dimension of OD, MS and OMT components respectively. To fully utilize the complementary information among these components, we perform a concatenation operation along the feature dimension:

![](images/44c785ad5ebad59177d5bee1dafcd334c53f0c8f0b021aaa6390b607a945ab87.jpg)  
Figure 5: Detailed illustration of the feature extraction process based on multi-head attention. The framework maps input data to a higher-dimensional space, adds positional encoding, and processes the sequence through two stacked Transformer encoder layers. Each layer utilizes multi-head attention (4 attention heads, denoted by ×4 in the figure) to capture global dependencies. After stabilizing training with residual connections and layer normalization, the features are aggregated into a global representation.

$$
F = \mathrm { C o n c a t } \left( F _ { O D } , F _ { M S } , F _ { O M T } \right) .\tag{4}
$$

The concatenated features are then projected into a unified embedding space through a linear transformation:

$$
X = F W _ { e } + b _ { e } ,\tag{5}
$$

where $W _ { e } ~ \in ~ \mathbb { R } ^ { \left( d _ { O D } + d _ { M S } + d _ { O M T } \right) \times d }$ maps the combined features to the embedding dimension �. To preserve the temporal characteristics of each component, we inject sine position encoding $P \colon X ^ { \prime } = X + P , P \in \mathbb { R } ^ { n \times d }$ , where $X ^ { \prime }$ represents the sequence of positional encoded component.

Considering the diferences in the physical properties of the diferent micromovement components, we have designed a dedicated multi-head attention mechanism. Specifically, we generate the queries, keys, and values for OD, MS, and OMT components through independent learnable matrices $W _ { Q } { } ^ { O D } , W _ { Q } { } ^ { M S }$ and $W _ { O } { } ^ { \overbar { O } M T }$ , respectively. The keys matrices $W _ { K }$ and values matrices $W _ { V }$ are similarly constructed. Then, for each head $h ,$ we calculate the scaled dot-product attention scores as Eq. 6 shows:

$$
{ \mathrm { A t t e n t i o n ~ } } \left( Q _ { h } , K _ { h } , V _ { h } \right) = { \mathrm { S o f t m a x } } \left( { \frac { Q _ { h } { K _ { h } } ^ { T } } { \sqrt { d _ { k } } } } \right) V _ { h } ,\tag{6}
$$

where $Q _ { h } , K _ { h }$ and $V _ { h }$ are learned from the sequence of positional encoded component $X ^ { \prime }$ . This enables the model to capture the global dependencies among OD, MS, and OMT sequences. The outputs of all heads are then combined and projected to generate attention features $Z .$ . After residual connections and layer normalization, the features are further refined through a feed-forward neural network as Eq. 7 shows:

$$
F F N ( Z ) = G E L U \left( Z W _ { 1 } + b _ { 1 } \right) W _ { 2 } + b _ { 2 } .\tag{7}
$$

where � and � denote the weights and biases of the feedforward neural network.

The choice of extracting a 128-dimensional global feature representation is a deliberate architectural decision. Empirical grid search indicates that dimensions below 64 result in feature under-representation, while dimensions above 256 lead to severe overfitting on physiological datasets of this scale and increased mobile memory overhead. To balance representational capacity with edge-device constraints, our MHA module comprises 2 Transformer encoder layers with 4 attention heads, a hidden dimension of 128, and a dropout rate of 0.3 to mitigate overfitting.

During the ofline training phase, the MHA network is optimized using the Cross-Entropy Loss function. We employ the AdamW optimizer with an initial learning rate of $\bar { 1 } \times 1 0 ^ { - 3 }$ and a weight decay of $1 \times 1 0 ^ { - 4 }$ . A cosine annealing learning rate scheduler is utilized over 100 epochs with a batch size of 64. Ultimately, this stage outputs a 128- dimensional dense vector rich in emotion-related semantics, providing the foundational knowledge base for the subsequent expert decision-making modules.

## 4.4.2. Feature Importance Ranking

To bridge the gap between "black-box" deep learning features and physiological interpretability, we pass the extracted 128-dimensional MHA features through an XGBoost (Extreme Gradient Boosting) model. This stage primarily serves to assess the relative information gain of specific attention-derived features, pinpointing exactly which micropatterns (MS, OD, or OMT) are most critical for distinguishing emotional arousal and valence (Sagi and Rokach, 2021). The procedure is outlined as follows:

The category-specific feature vectors derived from the multi-head attention mechanisms are concatenated to form a unified 128-dimensional feature array, where each dimension traces back to its source modality (MS, OD, or OMT). This array, paired with discrete emotion labels, is fed into the XGBoost architecture (configured with a maximum tree depth of 6 and a learning rate of 0.1). Upon completion of the training folds under the strict LOSO-CV protocol, the Information Gain metric $I _ { j }$ for each feature dimension � ∈ [1, 128] is extracted.

Rather than treating the deep learning pipeline as an opaque black box, we implement an XAI strategy to interpret the physiological drivers behind the model’s decisions. We extract the class-specific information gain from the XG-Boost trees to isolate the relative percentage contribution of MS, OD, and OMT features for each distinct emotional state. By aggregating the dimension-wise scores $I _ { j }$ back to their respective source modalities, we can quantitatively map specific autonomic fixational behaviors to high-arousal and low-arousal states, providing a transparent, physiologically grounded interpretation for domain experts (the detailed neurophysiological analysis of which is presented in Sec. 5.3.2).

Beyond providing expert interpretability, XGBoost serves as an intermediate soft-gating mechanism. However, we deliberately ofload the final classification to a downstream SVM rather than relying on XGBoost’s raw outputs. This architectural decoupling is theoretically motivated: treebased ensembles like XGBoost intrinsically construct axisaligned, orthogonal decision boundaries, which can be susceptible to high variance and struggle with extrapolation when encountering out-of-distribution physiological signals from completely unseen users (Hastie, Tibshirani and Friedman, 2009). Conversely, an SVM with a non-linear RBF kernel guarantees a maximally smooth, maximummargin hyperplane. This property has been extensively validated to demonstrate superior generalization and structural robustness on high-dimensional, continuous physiological data (Lotte, Bougrain, Cichocki, Clerc, Congedo, Rakotomamonjy and Yger, 2018).

To seamlessly bridge the two modules, the raw information gain scores $I _ { j }$ are subjected to Min-Max normalization to derive the final scaling weights $w _ { j } .$

$$
w _ { j } = 0 . 1 + 0 . 9 \times \frac { I _ { j } - \operatorname* { m i n } ( I ) } { \operatorname* { m a x } ( I ) - \operatorname* { m i n } ( I ) }\tag{8}
$$

The baseline shift of 0.1 ensures that low-contributing features are smoothly suppressed but not entirely zeroed out. The 128-dimensional attention features are element-wise multiplied by these weights $w _ { j }$ before being projected into the SVM classifier.

## 4.5. Emotion Classification

The final emotion inference is executed by a Support Vector Machine (SVM) utilizing a Radial Basis Function (RBF) kernel. The 128-dimensional attention-derived features are first scaled by their respective XGBoost-derived importance weights (Yan, Lin, Lin and Vucetic, 2023; Liu, Cheng and Lee, 2020).

This deliberate MHA-XGBoost-SVM hybrid pipeline balances deep representation power with edge-deployment eficiency. The RBF kernel adeptly models the highly nonlinear decision boundaries of physiological data by projecting the weighted features into a separable space. Crucially, while SVM training is computationally intensive, its inference complexity is merely $O ( N _ { s v } \cdot d )$ . This lightweight inference phase makes the model highly viable for near real-time, continuous deployment on a companion mobile edge device (e.g., a smartphone) without draining battery resources.

The final classification process is meticulously designed to prevent any data leakage. All feature standardizations, XGBoost weight extractions, and SVM training steps are encapsulated strictly within the training folds. The 128- dimensional attention features are first standardized using Zscore normalization fitted solely on the training data. Subsequently, these normalized features are element-wise multiplied by the fold-specific XGBoost-derived scaling weights $w _ { j }$ . This sequential order is mathematically critical, ensuring that the standardization process does not inadvertently neutralize the relative physiological importance assigned by the XAI soft-gating mechanism. To further address potential class imbalances, class-weight penalties inversely proportional to the emotion class frequencies are incorporated into the SVM margin formulation.

To rigorously adhere to the Leave-One-Subject-Out (LOSO) protocol, the SVM hyperparameters—specifically the penalty parameter � and the RBF kernel coeficient �—are not globally fixed. Instead, we implement a Nested Cross-Validation strategy. For each LOSO iteration (where Subject � is the completely unseen test set), an internal 5-fold cross-validation is performed strictly within the remaining 59 training subjects. A grid search is executed over the hyperparameter space: $C \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 \}$ and $\gamma \in$ {0.001, 0.01, 0.1, 1, ’scale’}, where the ’scale’ parameter dynamically calculates � based on the inverse of the feature variance to handle high-dimensional physiological inputs. The internally optimal parameters (typically converging around $C = 1 0 , \gamma = 0 . 0 1 )$ are then utilized to train the final SVM for that fold, which is finally evaluated on Subject �.

This rigorous methodology guarantees that the reported zero-shot predictive performance is an uninflated reflection of EmoGaze’s generalizability in real-world deployments. Furthermore, because the SVM relies on support vectors to define its decision boundaries, it is inherently well-suited for incremental learning. For personalized real-world deployments, these boundaries can be eficiently fine-tuned using a minimal subset of the target user’s physiological data, forming the algorithmic foundation for the few-shot personalization mechanism evaluated in Sec. 5.2.3.

## 5. Evaluation

We have prototyped the data collection module of EmoGaze using our custom-designed smart glasses eye tracker, as shown in Fig. 3. To rigorously validate both the algorithmic eficacy and the system’s viability as a pervasive mobile framework, our evaluation architecture is designed as a two-stage pipeline. The ofline model training, rigorous subject-independent validation, and multi-head attention optimizations are conducted on a workstation equipped with an AMD Ryzen 9 4900HS CPU and 16 GB RAM. Crucially, to demonstrate its readiness for everyday pervasive use, the lightweight feature extraction and SVM inference modules are subsequently deployed and profiled on a commercial mobile device (Android smartphone), ensuring that users high-framerate eye movement video streams are processed locally and in real-time.

The remainder of the evaluation is structured as follows to systematically address the system’s performance, interpretability, and ecological validity. We first present the experimental setup and evaluation metrics in Sec. 5.1. Following this, our evaluations are organized into three primary dimensions:

• System Accuracy and Personalization (Sec. 5.2): We estimate the overall performance of EmoGaze using a rigorous Leave-One-Subject-Out Cross-Validation (LOSO-CV) to evaluate true generalization to unseen individuals, complemented by a few-shot personalization analysis informed by contemporary emotion theories.

• Ablation Study and Feature Importance (Sec. 5.3): To substantiate our central hypothesis that micro-level eye movements encode rich emotional information, we conduct comprehensive ablation studies isolating the contributions of ocular drifts (OD), microsaccades (MS), and ocular microtremors (OMT). We also report the XGBoost-derived feature importance rankings.

Table 2  
Demographics of volunteers in the experiment
<table><tr><td>Gender</td><td>No.</td><td>Age range</td><td>No.</td><td>Acting experience</td><td>No.</td></tr><tr><td>Female</td><td>30</td><td>18-31</td><td>20</td><td>With</td><td>40</td></tr><tr><td>Male</td><td>30</td><td>32-45</td><td>16</td><td>Without</td><td>20</td></tr><tr><td></td><td></td><td>46-59</td><td>14</td><td></td><td></td></tr><tr><td></td><td></td><td>60-73</td><td>10</td><td></td><td></td></tr></table>

• Mobile Viability and Naturalistic Robustness (Sec. 5.4): We investigate EmoGaze’s robustness across naturalistic, unconstrained visual tasks and varying environmental factors. Finally, we report the system’s endto-end latency, CPU usage, and memory overhead on mobile hardware to validate its real-world deployability.

## 5.1. Experiment Setup and Metrics

## 5.1.1. Experimental setting

A total of 60 volunteers (30 females and 30 males) have participated in the evaluation. The demographic distribution is detailed in Table 2, indicating a diverse age group ranging from 18 to 73 years. To rigorously evaluate the model’s robustness against individual variations in emotional expressivity, our participant pool intentionally encompasses varying levels of expressiveness, including 40 individuals with formal acting/dramatic training and 20 without. Rather than utilizing trained individuals to generate exaggerated or "posed" signals, this demographic composition allows us to systematically investigate how inherent emotional expressiveness impacts autonomic fixational micro-movements as a demographic sub-group (analyzed in Sec. 5). Participants have normal or corrected-to-normal vision. All collected data are kept strictly anonymous, and the Institutional Review Board (IRB) of our university authorized all study procedures.

Apparatus and High-Fidelity Sensing. Participants wear our custom-designed smart glasses equipped with two built-in eye-tracking cameras, one for each eye. The cameras capture the participants’ eye area at a resolution of 360p and a high frame rate of 500 fps. This strict 500 Hz sampling rate safely satisfies the Nyquist sampling theorem to capture high-frequency ocular microtremors (OMT, typically 40– 100Hz). Furthermore, to guarantee sensing fidelity at a 360p resolution, we employ a sub-pixel pupil center estimation algorithm, ensuring that even micrometer-level physiological eye movements are faithfully recorded above the camera’s baseline noise floor.

Standardized Emotion Elicitation and Ground Truth Protocol. To eliminate the bias of artificial or posed emotions, we implement a rigorous emotion elicitation paradigm. We design a two-phase experimental protocol to bridge the gap between laboratory baselines and unconstrained ecological validity:

• Phase 1: Controlled Baseline Task. Participants are seated in a quiet ofice and asked to fixate on a target point. We utilize standardized, validated multimodal stimuli (e.g., emotionally charged film clips from established afective computing databases) to elicit genuine internal states of Anger, Joy, Sadness, and Pleasure.

• Phase 2: Naturalistic Mobile Task (In-the-wild). To account for varying attention and visual task demands, participants subsequently perform unconstrained daily activities. Elicitation stimuli are embedded directly into naturalistic mobile interfaces, such as reading polarized news articles on a smartphone or watching short-form emotional vlog clips in a breakroom environment.

Immediately following each stimulus trial, ground truth is established using the widely-adopted Positive and Negative Afect Schedule (PANAS). This ensures that the emotion labels strictly reflect the participants’ genuine, self-reported afective states rather than our subjective assumptions about the stimuli. Furthermore, the end-to-end processing pipeline is profiled on a commercial Android smartphone (Snapdragon 8 Gen 2) to evaluate real-world system latency.

Evaluation Strategy and Theoretical Grounding. To rigorously evaluate the model’s generalization capability to unseen individuals and prevent any potential identity data leakage, we discard traditional random-split crossvalidation. Instead, we adopt a strict LOSO-CV. In each iteration, data from 59 subjects are used for training, and the remaining 1 subject’s data is held out exclusively for testing.

Moreover, aligned with the contemporary Theory of Constructed Emotion (Barrett, 2017), which posits that emotional physiological responses are context-dependent and exhibit significant inter-individual variability, we acknowledge that universal discrete emotion mapping has limitations in real-world deployments. Therefore, alongside the zeroshot LOSO-CV, we introduce a few-shot personalization evaluation metric. In this setup, a minimal fraction (e.g., 10%) of the unseen test user’s data is utilized to calibrate the model, demonstrating how EmoGaze adapts to individual physiological heterogeneity in everyday mobile scenarios.

## 5.1.2. Evaluation metrics

To comprehensively assess EmoGaze, we define the following multi-dimensional evaluation metrics:

1) Classification Performance: We evaluate the core subject-independent classification accuracy using macroaveraged precision, recall, and F1-Score. These metrics are exclusively computed under a strict LOSO-CV protocol to reflect true generalization to unseen users.

2) Few-Shot Personalization Gain: To quantify the system’s adaptability to individual physiological diferences, we measure the personalization gain. This is defined as the absolute improvement in F1-score when the baseline LOSO-CV model is fine-tuned using a minimal subset (e.g., 5% − 10%) of an unseen user’s data.

Prediction accuracy of EmoGaze under rigorous LOSO-CV
<table><tr><td>Emotion</td><td>Precision(%)</td><td>Recall(%)</td><td>F1-score(%)</td><td>Support</td></tr><tr><td>Joy</td><td>75.3</td><td>70.5</td><td>72.8</td><td>637</td></tr><tr><td>Pleasure</td><td>70.6</td><td>73.9</td><td>72.2</td><td>643</td></tr><tr><td>Sadness</td><td>80.5</td><td>74.6</td><td>77.4</td><td>732</td></tr><tr><td>Anger</td><td>83.2</td><td>76.4</td><td>79.7</td><td>706</td></tr><tr><td>Avg/total</td><td>77.4</td><td>73.9</td><td>75.5</td><td>2718</td></tr></table>

3) Feature Importance and Ablation: We extract the information gain from the XGBoost module to precisely quantify the contribution of each micro-movement modality. For the modality ablation study, we use precision, recall, and F1-score to benchmark our full fusion (MS+OD+OMT) against macroscopic baselines and partial micro-movement subsets.

4) Mobile Edge System Overhead: To assess real-world deployability, we profile the complete edge-computing pipeline (including both high-framerate image-to-coordinate extraction and SVM inference) on a commercial Android smartphone. We specifically report the End-to-End Processing Latency (ms) and CPU/Memory Utilization(%).

## 5.2. System Accuracy and Personalization

We first evaluate the fundamental emotion recognition performance of EmoGaze under strict subject-independent conditions, followed by an exploration of personalized calibration based on contemporary emotion theories.

## 5.2.1. Subject-Independent Prediction Accuracy

To evaluate the true generalization capability of EmoGaze, we utilize the data collected from the 60 volunteers. The emotion labels reported via PANAS serve as the ground truth, resulting in 2,718 valid samples. Crucially, addressing the methodological pitfalls of standard random-split crossvalidation (which often leads to data leakage and inflated accuracies in physiological computing), we apply a rigorous LOSO-CV.

As shown in Table 3 and Fig. 6, under the zero-shot LOSO-CV protocol (testing on completely unseen individuals), EmoGaze achieves a macro-averaged precision of 77.4%, a recall of 73.9%, and an F1-score of 75.5%. While lower than traditional random-split accuracies, this performance firmly demonstrates the robust baseline capability of our micro-movement features in predicting emotional states across diverse and previously unseen physiological profiles.

To further elucidate the inter-class classification dynamics and error modalities, Fig. 7 presents the aggregate confusion matrix derived from the LOSO-CV evaluation. The matrix reveals highly distinct classification boundaries for extreme states. Notably, Anger and Sadness demonstrate minimal inter-class confusion due to their polarized arousal profiles, which are cleanly separated by the amplitude variations in our extracted OMT and OD features.

A minor degree of expected misclassification occurs between Joy and Pleasure (e.g., a small fraction of Joy instances predicted as Pleasure). This is physiologically anticipated given their shared positive valence and proximity within the arousal continuum. Nevertheless, the deep temporal features extracted by the multi-head attention module efectively disentangle even these closely related states in the vast majority of cases, underscoring the robustness of microscopic fixational indicators over traditional macroscopic gaze tracking.

![](images/73dbe48651848a5c66ae55e00b3b784d63c806aeeed9a13142608c242327ad5f.jpg)

![](images/48cb9cc83d3b92f46991c309f7c500088d3fdb8f9464e415059d7561fb85baa9.jpg)

![](images/b58198fde9fc0147ee0b8b1d9d0bf21aeda487c0eb423fc30e4b2e3fd5010caf.jpg)

![](images/6550149d9c331e5bccdbbbea615f8047b8d0da53acc8a2b78ce78858eea1b75e.jpg)  
Figure 6: EmoGaze’s zero-shot performance under diferent emotions. CDF plot of (a) Joy, (b) Anger, (c) Sadness, and (d) Pleasure.

![](images/04900348cccbb4894cf216244c039ac4388cf351eed24ca54f13aa2f19b1031f.jpg)  
Figure 7: Aggregate confusion matrix across the LOSO-CV folds. The matrix highlights strong diagonal performance, with minor predictable confusion between neighboring valence-arousal states (e.g., Joy vs. Pleasure).

## 5.2.2. Demographic Variability

Contemporary afective science, particularly the Theory of Constructed Emotion (Barrett, 2017), posits that emotions are highly context-dependent and exhibit significant physiological variability across individuals. To systematically investigate this inherent heterogeneity, we first evaluate EmoGaze’s performance across three specific demographic dimensions: individuals with and without acting experience, male and female participants, and diferent age cohorts.

Following a group-specific evaluation protocol (i.e., training and testing the model exclusively on data from the same demographic group), the results in Fig. 8 demonstrate that EmoGaze remains robustly efective across all subsets.

Notably, the algorithm achieves higher accuracy for individuals with acting experience. This aligns with findings in (Zhao, Adib and Katabi, 2016), suggesting that actors are more adept at consistently managing and "reproducing" target emotions, thereby generating more discriminative micro-eye movement patterns. Meanwhile, the performance exhibits only marginal fluctuations across gender and age groups, underscoring the general stability of our extracted fixational features.

## 5.2.3. Few-Shot Personalization and Calibration Sensitivity

While demographic analysis confirms the baseline stability of our fixational features, contemporary afective science emphasizes that final emotional experiences are highly individualized. To bridge the gap between universal physiological baselines and individual emotional heterogeneity, EmoGaze incorporates a highly eficient few-shot personalization mechanism.

A critical challenge for personalized expert systems is the "cold-start" problem—how the system performs before acquiring any user-specific data. For a completely new user, EmoGaze seamlessly defaults to the generalized zero-shot LOSO-CV boundaries, ensuring immediate out-of-the-box usability with a robust baseline F1-score of 75.5%. As the user interacts with the system and provides sporadic groundtruth labels (e.g., via occasional Ecological Momentary Assessment (EMA) prompts on the companion smartphone), the system transitions to a personalized mode.

Emotions  
Emotions  
Emotions  
Emotions  
![](images/42aa549984b22894e5cb7db95ef9e5a91e3577e94bdd45539795b631928ab9a0.jpg)

![](images/03b5cefca2b0aafcc19525fa966a92c445c32927040810b74e05aea40ef7f93a.jpg)

![](images/55759f748f2e4e990ff43c7cb14f745dd85c71bd67ee204d21919aa67d925b51.jpg)

![](images/26ff2179d3fb0a073858474bd16e3d082b9c0e53bb542bb961a4b3b4db3a23d9.jpg)

![](images/8c9476bf76204e1c1856e756743750c61956423df3de82ce93d21b7b0dabcfa0.jpg)

![](images/c0185bb3c003476994a2d1a327bb326fd6b1f7a541813e335622e7a5e9d28213.jpg)

![](images/f15f6cfc4089d4052a95f93badd09bc2029608d391c1ab17d13910954f7de2d7.jpg)

![](images/7fc1c1fe763eb3cc033937f174b71290a7b3e203a101126c547868d666650f13.jpg)

![](images/7b9886418ad6822dfec572341a51feac9f8a372f4ce245e9e361c17222de6593.jpg)  
Figure 8: Performance across demographic dimensions using group-specific modeling. (a) Actor vs. Non-actor. (b) Male vs. Female. (c) Age groups (18-45 vs. 46-73).

Algorithmically, instead of computationally intensive retraining of the entire pipeline from scratch, EmoGaze employs a lightweight Support Vector Transfer strategy. The pre-computed support vectors from the global model are retained to anchor the latent space. The new user-specific calibration samples are then introduced into the SVM training phase with elevated sample-weight penalties. This elegantly and eficiently shifts the non-linear RBF decision hyperplanes to tightly encompass the new user’s specific physiological distribution, requiring minimal computational overhead on the edge device.

To rigorously evaluate the practical feasibility and the minimal sample size required for this personalization, we conduct a sensitivity analysis. We systematically vary the proportion of the unseen test user’s data used for fine-tuning: 0% (Cold-start), 1%, 5%, 10%, and 20%.

As illustrated in Table 4, even an extreme few-shot calibration scenario (utilizing merely 1% of the target user’s data) yields a noticeable personalization gain (+2.3%). The performance trajectory climbs steeply to 81.4% at a 5% data proportion and reaches an optimal eficacy trade-of at 10% (achieving 83.6% F1-score). Beyond 10%, the performance improvement plateaus (84.2% at 20%), exhibiting clear diminishing returns relative to the increased user calibration burden.

This sensitivity curve provides crucial deployment guidelines for expert systems: EmoGaze requires only a negligible calibration footprint. Given that our dataset contains an average of ≈ 45.3 one-second sample windows per user, a 10% proportion equates to merely 4.5 seconds of active physiological data. Therefore, transitioning from a robust generalized baseline to an expert-level, highly personalized emotion inference engine requires less than one minute of user interaction for label acquisition.

## 5.2.4. Classifier Eficiency Justification

Tojustify our selection of the SVM classifier for a mobile framework, we benchmark it against mainstream and deep learning architectures (DT, RF, k-NN, LSTM, BiLSTM, Transformer-Encoder (Zerveas, Jayaraman, Patel, Bhamidipaty and Eickhof, 2021), and CNN-LSTM (Zha, Liu, Wan,

Sensitivity analysis of Few-Shot Personalization across varying calibration data proportions
<table><tr><td>Calibration Proportion</td><td>F1-score (%)</td><td>Personalization Gain</td><td>Practical Feasibility</td></tr><tr><td>0% (Cold-start)</td><td>75.5</td><td></td><td>Immediate out-of-the-box usability</td></tr><tr><td>1%</td><td>77.8</td><td>+2.3%</td><td>Extremely low effort (e.g., a few seconds)</td></tr><tr><td>5%</td><td>81.4</td><td>+5.9%</td><td>Low effort</td></tr><tr><td>10%</td><td>83.6</td><td>+8.1%</td><td>Optimal trade-off point</td></tr><tr><td>20%</td><td>84.2</td><td>+8.7%</td><td>Diminishing returns; higher user burden</td></tr></table>

Luo, Li, Yang and Xu, 2022)). As shown in Fig. 9, SVM achieves comparable accuracy to heavy deep learning models for classifying high-dimensional extracted features, while maintaining a remarkably lower computational complexity. This balance is critical for real-time mobile scenarios.

## 5.2.5. Comparison with State-of-the-Art Gaze Baselines

To rigorously evaluate the external competitiveness of EmoGaze, we benchmark it against representative state-ofthe-art (SOTA) gaze-based emotion recognition paradigms. While direct comparison with heavily multi-modal methods (e.g., requiring clinical EEG or facial cameras) is restricted by hardware modalities, we re-implement and evaluate the following prominent gaze-only baselines on our comprehensive 60-subject dataset under the same strict LOSO-CV protocol:

• Macro-Gaze SOTA (Statistical + DL): Representing traditional gaze-based afective computing methods (e.g., (Lu, Zheng, Li and Lu, 2015)), this baseline extracts macroscopic behavioral features (e.g., total fixation duration, saccade amplitude, blink rate, and pupil diameter) and classifies them using a deep neural network.

• Raw-Gaze E2E (End-to-End Deep Learning): Representing modern brute-force sequence modeling (e.g., (Zemblys, Niehorster and Holmqvist, 2019)), this baseline directly feeds the raw, unfiltered (�, �) coordinate and velocity time-series into a deep neural network, bypassing any explicit physiological decomposition.

As presented in Table 5, EmoGaze significantly outperforms all external baselines. The Macro-Gaze SOTA achieves a modest F1-score of 62.1%, demonstrating that macroscopic behaviors alone are insuficient for fine-grained emotion classification due to their susceptibility to varying task demands. Interestingly, the Raw-Gaze E2E model (68.4%) also falls notably short of our EmoGaze zeroshot baseline (75.5%). This exposes a critical limitation of generic end-to-end deep learning: without explicit neurophysiological decomposition, even powerful deep networks struggle to implicitly separate subtle micro-eye movements (MS, OD, OMT) from dominant macroscopic gaze shifts and background hardware noise.

To validate the robustness of these results, we conduct a paired t-test across the LOSO-CV folds. The zero-shot performance improvement of EmoGaze over both the Macro-Gaze SOTA and Raw-Gaze E2E baselines is statistically significant $( p < 0 . 0 1 )$ . This formally corroborates that our explicit algorithmic mining of autonomic fixational micromovements is not merely an engineering alternative, but a fundamentally superior signal paradigm for mobile afective computing.

## 5.3. Ablation Study and Feature Importance

A central claim of this paper is that microscopic fixational dynamics provide superior emotional discriminability compared to macroscopic gaze features. To empirically validate this, we conduct both an ablation study and a feature importance analysis.

## 5.3.1. Modality Ablation Study

To empirically validate our central argument—that microlevel eye movements are substantially more informative for emotion recognition than macro-level gaze features— we conduct a comprehensive ablation study. As presented in Table 6, we benchmark our proposed full fusion model (MS+OD+OMT) against a baseline relying solely on Macroscopic Features (e.g., fixation duration, pupil size, and saccade amplitude), as well as single and dual micro-movement combinations.

The results reveal a fascinating neurophysiological phenomenon: the partial representation paradox. Relying on any single micro-modality (e.g., Only MS: 47.3%) or dual combinations (e.g., MS+OD: 54.6%) yields performance inferior to the Macroscopic baseline (62.1%). This is logically sound: macroscopic features provide a holistic, albeit lowresolution, summary of the user’s state. In contrast, microscopic signals are highly specialized, narrow-band physiological reflexes. For instance, MS captures arousal spikes, while OD predominantly reflects valence drops. Devoid of cross-modality context, an isolated micro-signal acts as a weak, incomplete classifier, sufering from the "blind men and the elephant" problem.

However, this drastic performance gap implies a profound synergistic efect. When all three fundamental micromovements are fused through our multi-head attention architecture (MS+OD+OMT), the cross-attention mechanism successfully reconstructs the holistic autonomic neurophysiological state. This synergy overcomes the macroscopic baseline’s plateau, achieving a significantly superior F1- score of 75.5%. This rigorously proves that while individual micro-patterns are fragmented, their deep algorithmic fusion forms an unprecedented, expert-level data stream for emotion inference.

Table 5  
Comparison with SOTA Gaze-based Emotion Recognition Baselines
<table><tr><td>Method / Paradigm</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td></tr><tr><td>Macro-Gaze SOTA (Statistical + DL) (Lu et al., 2015)</td><td>60.7</td><td>63.5</td><td>62.1</td></tr><tr><td>Raw-Gaze E2E (Deep Sequence Modeling) (Zemblys et al., 2019)</td><td>67.8</td><td>69.1</td><td>68.4</td></tr><tr><td>EmoGaze (Zero-shot LOSO-CV)</td><td>77.4</td><td>73.9</td><td>75.5</td></tr><tr><td>EmoGaze (Few-shot Personalized)</td><td>84.1</td><td>83.2</td><td>83.6</td></tr></table>

Table 6

Modality ablation results demonstrating the synergistic efect of micro-movements
<table><tr><td>Feature Combination</td><td>Precision  $\overline { { ( \% ) } }$ </td><td>Recall (%)</td><td>F1-score  $\overline { { ( \% ) } }$ </td></tr><tr><td>Macroscopic Features</td><td>60.7</td><td>63.5</td><td>62.1</td></tr><tr><td>Only MS</td><td>48.5</td><td>46.2</td><td>47.3</td></tr><tr><td>Only OD</td><td>45.2</td><td>44.8</td><td>45.0</td></tr><tr><td>Only OMT</td><td>41.8</td><td>40.5</td><td>41.1</td></tr><tr><td>OD+OMT</td><td>57.9</td><td>56.8</td><td>57.3</td></tr><tr><td>MS+OMT</td><td>51.3</td><td>52.5</td><td>51.9</td></tr><tr><td> ${ \mathsf { M S } } { + } \mathsf { O D }$ </td><td>55.2</td><td>54.1</td><td>54.6</td></tr><tr><td> $\mathbf { M S + O D + O M T }$  (Full Fusion)</td><td>77.4</td><td>73.9</td><td>75.5</td></tr></table>

## 5.3.2. Feature Importance Ranking and Physiological Interpretability

To explicitly address the physiological validity of our multi-head attention mechanisms and interpret the "black box" of our deep learning pipeline, we extract the classspecific information gain from the XGBoost training phase. Instead of merely reporting an aggregate score, we isolate the relative percentage contribution of MS, OD, and OMT features for each discrete emotional state.

As illustrated in Fig. 10, the relative feature importance exhibits stark, physiologically meaningful variations across diferent emotions, directly corroborating our theoretical qualitative analysis presented in Sec. 3.

Specifically, for high-arousal states such as anger and joy, the rapid, high-frequency components dominate the classification decision. In the case of anger, MS and OMT features account for the vast majority of the decision weight (e.g., 55.2% and 30.5%, respectively). This aligns perfectly with our initial observation that MS exhibits increased frequency and amplitude during heightened emotional arousal, serving as a direct physiological correlate to the autonomic nervous system’s response to intense stimuli.

Conversely, for low-valence and low-arousal states such as sadness, the reliance on feature modalities shifts dramatically towards the OD branch (contributing to 63.1% of the importance score). Consistent with our motivation analysis, sadness manifests with significantly lower frequency and amplitude across all signals. Consequently, the slow, involuntary ocular drifts become the most discriminative signature, efectively capturing the user’s reduced capacity for visual fixation control and visual disengagement from external stimuli during emotional downturns. Moderate positive states like pleasure show a more balanced feature distribution, leveraging both OD and stable MS activity.

By deconstructing the feature importance per emotion, we demonstrate that EmoGaze successfully learns and leverages genuine, fine-grained neurophysiological ocular behaviors. This mapping from deep attention weights back to established cognitive psychology not only addresses the "black box" nature of typical deep learning systems but also validates our core premise: specific fixation micromovements contain distinct, robust markers for real-time emotion monitoring.

## 5.4. Mobile Viability and Naturalistic Robustness

To ensure EmoGaze functions reliably outside the laboratory, we investigate its robustness against environmental artifacts and user states, and finally assess its real-world system overhead on mobile hardware.

## 5.4.1. Impact ofEnvironmental Factors (Light and Noise)

We test EmoGaze under varying light intensities. Light intensity is controlled by toggling three ofice lights: one on for weak light intensity $( L i _ { w } , \ 1 1 0 – 1 5 0 \ \mathrm { l u x } )$ , two for medium light intensity $( L i _ { m } , 2 2 0 - 2 6 0$ lux), and three for strong light intensity $( L i _ { s } , 2 8 0 - 3 0 0 \ \mathrm { l u x } )$ . As illustrated in Fig. 11, precision mildly improves under stronger light $( L i _ { s } )$ due to clearer camera captures, though the system remains robust under weak light $( L i _ { w } )$

We further comprehensively investigate ambient noise based on standard acoustic measurement principles (Jacobsen and de Bree, 2005), systematically assessing the impact of sound types, intensities, and source directions.

![](images/88fd6bd9c064cee72dd62d074ee271999b57ded98aa808d29a26feb3a3275f6e.jpg)  
Figure 9: Performance comparison of EmoGaze utilizing SVM vs. Deep Learning models.

![](images/464a4b20506fddbadbb96d0aecca023e0a4662e33f59de4d0b7cf0fd318ad5fd.jpg)  
Figure 10: Class-specific feature importance of micro-eye movements. (Detailed numerical breakdown: For Anger, MS=55.2%, OMT=30.5%, OD=14.3%; For Sadness, OD=63.1%, MS=21.4%, OMT=15.5%; For Joy, MS=48.7%, OD=28.5%, OMT=22.8%; For Pleasure, OD=45.2%, MS=36.8%, OMT=18%.

• Sound Types: As shown in Fig. 12, EmoGaze maintains relatively high accuracy under natural environmental sounds (e.g., wind, rain) and everyday activity sounds (e.g., background conversations). However, sudden sounds (e.g., alarms, telephone rings) cause a slight accuracy drop. This occurs because sudden noises are rare and abruptly capture users’ attention, causing them to temporarily lose focus on the target.

• Sound Intensity: Fig. 13 reveals that system accuracy significantly decreases as sound intensity increases from low (30–50 dB) to high (80–90 dB) levels. While typical ambient noise ranges from 30 to 50 dB, sound levels above 60 dB can actively interfere with neural activity (Röhl and Uppenkamp, 2012; Herrmann, Augereau and Johnsrude, 2020). This physiological interference acts as a confounder to baseline emotional states, subsequently reducing the accuracy of EmoGaze.

• Sound Direction: Regarding source direction (Fig. 14), sounds originating from the front or back have minimal impact on system performance. In contrast, sounds from the side exhibit a minor disruptive efect.

This is attributed to the fact that lateral sounds reflexively attract visual attention, prompting involuntary eye shifts that disrupt stable fixational behavior.

## 5.4.2. Naturalistic Tasks and User States

The experimental results are depicted in Fig. 15. The accuracy remains highly comparable across seated (SIT), controlled ambulation (CA), and free indoor/outdoor (FI/OA) tasks, proving that the micro-movement patterns hold true even during dynamic, everyday visual tasks. A noticeable decline occurs only during intensive running (RUN), attributed to severe physical vibrations and motion artifacts (Neumann and Piercy, 2013).

Crucially, rather than viewing this degradation as a systemic failure, this outcome precisely delineates the operating envelope of EmoGaze. It demonstrates that the system is exceptionally robust during stationary and routine ambulatory activities—which constitute the vast majority of an individual’s daily routine—but becomes less reliable during vigorous physical exertion. This empirical boundary informs a pragmatic deployment strategy: EmoGaze should operate as an opportunistic sensing framework, actively monitoring emotions during normal daily routines while intelligently suspending inference during intense activities to prevent false positives and conserve battery life.

Emotions  
Emotions  
![](images/6c59b6d9a940f29e28b53503dd6d5c3e932b5190c732e570027b61086d73077a.jpg)

![](images/abafb320e418124e75278eba3dcb1c729bb5b98e03cdc1bc36a7bd5a542d0c2b.jpg)

![](images/b6c68a57ae844c48dbeff432dd6ec14302fcb91d58dc16826933af6521c7f2fe.jpg)

Figure 11: Performance under diferent light conditions $\left( L i _ { w } , L i _ { m } , L i _ { s } \right)$  
![](images/099b72a9ad84c3ed6a6b633a92416d95337d494741ac4e37136d28fa71ef6bb1.jpg)

![](images/004dc6c5fb2e911a14760f39860e8855ff3bde28a4a20612fee6fcc0d6e06f0b.jpg)

![](images/5c755e2413e7c8e2be43fe0b3b6afbd8f7867d5cee51098c16190f1419e1f76c.jpg)  
Figure 12: Performance under NES, EAS and sudden sounds (SS).

## 5.4.3. System Overhead on Mobile Hardware

To unequivocally validate EmoGaze’s pervasive viability and address the severe power and thermal constraints inherent in head-mounted wearables, we design an edgecomputing architecture leveraging a commercial Android smartphone (Snapdragon 8 Gen 2 processor, 12GB RAM) as the primary computational hub.

![](images/c1a3193a98eccf2866bed7ddbdac1a1eb1567e021c1e8f6daf969076ef231d17.jpg)

![](images/5e8f9596cacdc8e2d4a1665340ee38d5b125be8820ef0e4e8ee9eae0719f99fe.jpg)

![](images/e60188eff9b8114c49ee7ea5a85f6156d704ac8e3a8858ed664fbe4e22f7b634.jpg)  
Figure 13: Performance under varied sound intensities.

![](images/f31e0d2c5202b03cc53b13eefc1e4ad2f398244e3a89ab53c483d5d857d4515b.jpg)

![](images/1eab47b9da5c0483e5d1d446e687cd12fc6c2cd197e4899634397b4a4821bd2a.jpg)

![](images/7d1ae83d5816a81d08993c58692597e37e77c07151af6f200418f452864bcbfd.jpg)  
Figure 14: Performance under varied sound directions.

![](images/23a99a1a9fb59286a6f8fbd53e7c79eb408e1d38302e94e4d86762be02bd3e6e.jpg)  
User States

![](images/666c5f23563fd1afb58acbd469e404374bf179fd36377efd52b3c0af34bf261a.jpg)  
User States

![](images/c03ff30eeb720fa7c4d5a5c3af8158d4f4afc6bf2851d6907c692cb351b24b9d.jpg)  
User States

![](images/2a1df3e730ee301b5cefdcbe1ab41ff7920d8f02e7083cc8e9e6904687878dd5.jpg)  
User States  
Figure 15: Performance across dynamic user states and naturalistic tasks (SIT, CA, FI/OA, RUN).

In this architecture, task partitioning is critical. Offloading high-framerate (500 fps) raw image processing to the cloud introduces unpredictable network latency and jitter, which severely degrades the temporal fidelity required to capture millisecond-level micro-eye movements. Conversely, executing complex computer vision tasks directly on the smart glasses causes rapid battery depletion. Therefore, we adopt a local edge-ofloading strategy where the smart glasses function purely as a high-fidelity sensor node. They capture dual-eye infrared video streams and transmit the cropped Region of Interest (ROI) locally to the companion smartphone via a high-bandwidth Wi-Fi Direct (802.11ax) connection, with an optional USB-C tether for zero-latency scenarios.

Upon receiving the video stream, the smartphone executes the complete EmoGaze pipeline. It first utilizes its highly optimized Digital Signal Processor (DSP) to perform real-time image thresholding and pupil center extraction. The resulting coordinate time-series is then fed directly into our emotion inference pipeline, which encompasses the multi-head attention feature extraction, XGBoost weighting, and SVM classification.

Empirical profiling demonstrates the remarkable eficiency of this mobile setup. For a continuous 1-second sliding window at 500 fps, the entire end-to-end processing latency—from image-to-coordinates to feature-to-emotion— is approximately 38.5 ms, ensuring seamless real-time responsiveness. Furthermore, the background CPU utilization on the mobile processor remains modest at 4.2%, and the memory footprint is bounded to 125 MB, primarily allocated for video frame bufers and attention weights. These hardware-in-the-loop metrics systematically confirm that EmoGaze successfully circumvents the unreliability of cloud ofloading and the hardware limits of smart glasses, resulting in a lightweight, robust framework suitable for continuous everyday emotion monitoring.

## 6. Discussions

In this section, we discuss the current limitations of EmoGaze and outline critical pathways for future research in mobile afective computing.

• Embracing Context-Dependent Emotion Construction. Currently, EmoGaze classifies emotional states based on four fundamental categories within the discrete valence-arousal model. However, aligning with contemporary afective science—specifically the Theory of Constructed Emotion (Barrett, 2017)—we acknowledge that emotions are not merely universal, static, or isolated physiological reflexes. Rather, they are context-dependent phenomena constructed from a combination of physiological signals, cognitive states, and environmental surroundings. While our few-shot personalization approach successfully mitigated baseline individual variability, future work must transition from discrete category classification to modeling continuous, context-aware afective trajectories, dynamically adjusting to the user’s daily life context.

• Longitudinal "In-the-Wild" Deployments. Although our evaluation extended beyond controlled laboratory settings to include naturalistic mobile tasks (e.g., reading, free-viewing, and walking), true ecological validity requires prolonged, unsupervised deployment in the wild. Future research will focus on longitudinal studies spanning weeks or months. This scale of deployment will allow us to capture a broader spectrum of naturalistic emotion elicitation and investigate how fixational micro-movement patterns evolve over time under varying levels of fatigue, stress, and circadian rhythms.

• Hardware Robustness in Dynamic Contexts. To capture ultra-fine microsaccades and microtremors, EmoGaze utilizes a 500 fps dual-infrared camera setup. While this high temporal resolution is a significant technical achievement for a mobile framework, it introduces challenges in extremely dynamic environments. For instance, infrared tracking is notoriously susceptible to interference from direct sunlight in outdoor scenarios. Our current evaluations are conducted under indoor illuminance levels (110– 300 lux); however, direct outdoor sunlight (10,000– 100,000 lux) will severely saturate infrared sensors, establishing a strict operational limitation for current optical hardware. Future iterations will explore adaptive sampling rates (e.g., dynamically down-clocking the camera during vigorous movement) and enhanced optical sensor integration to balance fidelity, power consumption, and environmental robustness.

• Secondary Privacy Implications of Gaze Tracking. While EmoGaze successfully circumvents the immediate privacy concerns of recording facial or audio data, continuous ocular monitoring presents secondary privacy risks. Gaze patterns and micromovements can inadvertently reveal sensitive physiological or cognitive traits, such as underlying neurological conditions (e.g., ADHD, Parkinson’s) or specific reading habits. Future deployment of such expert systems must incorporate robust on-device data anonymization and strict consent protocols to mitigate the exposure of inferred biometric profiles.

• Operating Envelope and Opportunistic Sensing. A common pitfall in wearable afective computing is overclaiming "24/7 continuous monitoring" without regard for severe environmental confounders. Our evaluations explicitly establish the operating envelope of EmoGaze: it performs robustly under routine lighting and ambulatory daily activities, but experiences signal degradation during vigorous exercise (e.g., running) and under extreme high-decibel sudden noises. Therefore, we position EmoGaze not as an unconditional continuous monitor, but as an opportunistic sensing framework. In future real-world deployments, EmoGaze can leverage the companion smartphone’s built-in IMU and ambient light sensors to dynamically detect boundary conditions (e.g., vigorous running or blinding direct sunlight). Upon detecting these extremes, the system can intelligently suspend the eyetracking camera to conserve power and avoid generating low-confidence emotional predictions, seamlessly resuming monitoring once the user returns to the established operating envelope.

• Multimodal and Context-Aware Fusion. While EmoG demonstrates that unimodal microscopic fixation data contains rich emotional markers, emotions are inherently multi-dimensional. Relying exclusively on eye-tracking poses challenges when ocular responses are suppressed or ambiguous. A promising future direction involves integrating complementary sensors readily available on smart glasses, such as IMUs for head-movement tracking, microphones for voice prosody analysis, and outward-facing scene cameras to understand the context of the emotion (e.g., what the user is looking at). Fusing this multimodal context with our micro-fixation algorithms will provide a more comprehensive, resilient, and accurate understanding of the user’s emotional state in unconstrained everyday scenarios.

## 7. Conclusion

In this paper, we present EmoGaze, a pervasive edgecomputing framework that pioneers the use of microscopic fixational dynamics for real-time, unobtrusive emotion recognition. Moving beyond traditional macroscopic gaze tracking, EmoGaze decomposes visual fixations into three distinct neurophysiological micro-movements: microsaccades, ocular drifts, and ocular microtremors. By deploying a custom-designed smart glasses prototype coupled with a companion smartphone acting as an edge-computing hub, we ensure the millisecond-level temporal fidelity required for micro-movement analysis while strictly preserving user privacy through local processing. Through a rigorous, subjectindependent evaluation (LOSO-CV) involving 60 diverse volunteers, we systematically demonstrate the system’s robust capabilities across both controlled and naturalistic mobile scenarios. Crucially, our extensive ablation studies and feature importance analyses unequivocally validate our core hypothesis: micro-level eye movements are substantially more informative and discriminative for emotion inference than macroscopic features. Furthermore, by introducing a few-shot personalization approach, EmoGaze successfully bridges universal physiological baselines with the profound individual variability emphasized by contemporary afective science. Ultimately, EmoGaze establishes a novel, physiologically interpretable, and highly deployable paradigm for continuous emotion monitoring, paving the way for nextgeneration afective wearables in human-computer interaction and ubiquitous mental health applications.

## References

Abdou, A., Sood, E., Müller, P., Bulling, A., 2022. Gaze-enhanced crossmodal embeddings for emotion recognition. Proceedings of the ACM on Human-Computer Interaction 6, 1–18.

Acheampong, F.A., Nunoo-Mensah, H., Chen, W., 2021. Transformer models for text-based emotion detection: a review of bert-based approaches. Artificial Intelligence Review 54, 5789–5829.

Ahissar, E., Arieli, A., Fried, M., Bonneh, Y., 2016. On the possible roles of microsaccades and drifts in visual perception. Vision research 118, 25–30.

e<sub>Alexander, R.G., Martinez-Conde, S., 2019. Fixational eye movements.</sub> Eye movement research: An introduction to its scientific foundations and applications , 73–115.

Alhargan, A., Cooke, N., Binjammaz, T., 2017. Multimodal afect recognition in an interactive gaming environment using eye tracking and speech signals, in: Proceedings of the 19th ACM international conference on multimodal interaction, pp. 479–486.

Arun, A., Maheswaravenkatesh, P., Jayasankar, T., 2023. Facial micro emotion detection and classification using swarm intelligence based modified convolutional network. Expert Systems with Applications 233, 120947.

Aston-Jones, G., Cohen, J.D., 2005. An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance. Annu. Rev. Neurosci. 28, 403–450.

Awais, M., Raza, M., Singh, N., Bashir, K., Manzoor, U., Islam, S.U., Rodrigues, J.J., 2020. Lstm-based emotion detection using physiological

signals: Iot framework for healthcare and distance learning in covid-19. IEEE Internet of Things Journal 8, 16863–16871.

Barea, R., Boquete, L., Mazo, M., López, E., 2002. System for assisted mobility using eye movements based on electrooculography. IEEE transactions on neural systems and rehabilitation engineering 10, 209– 218.

Barrett, L.F., 2017. The theory of constructed emotion: an active inference account of interoception and categorization. Social cognitive and afective neuroscience 12, 1–23.

Black, M.H., Chen, N.T., Lipp, O.V., Bölte, S., Girdler, S., 2020. Complex facial emotion recognition and atypical gaze patterns in autistic adults. Autism 24, 258–262.

Bolger, C., Bojanic, S., Sheahan, N.F., Coakley, D., Malone, J.F., 1999. Ocular microtremor in patients with idiopathic parkinson’s disease. Journal of Neurology, Neurosurgery & Psychiatry 66, 528–531.

Bolger, C., Bojanic, S., Sheahan, N.F., Coakley, D., Malone, J.F., 2001. Effect of age on ocular microtremor activity. The Journals of Gerontology Series A: Biological Sciences and Medical Sciences 56, M386–M390.

Bradley, M.M., Lang, P.J., 1994. Measuring emotion: the self-assessment manikin and the semantic diferential. Journal of behavior therapy and experimental psychiatry 25, 49–59.

Canal, F.Z., Müller, T.R., Matias, J.C., Scotton, G.G., de Sa Junior, A.R., Pozzebon, E., Sobieranski, A.C., 2022. A survey on facial emotion recognition techniques: A state-of-the-art literature review. Information Sciences 582, 593–617.

Chen, J.T., Yep, R., Hsu, Y.F., Cherng, Y.G., Wang, C.A., 2021a. Investigating arousal, saccade preparation, and global luminance efects on microsaccade behavior. Frontiers in Human Neuroscience 15, 602835.

Chen, Z., Cao, Y., Yao, H., Lu, X., Peng, X., Mei, H., Liu, X., 2021b. Emojipowered sentiment and emotion detection from software developers’ communication data. ACM Transactions on Software Engineering and Methodology (TOSEM) 30, 1–48.

Cuve, H.C., Castiello, S., Shiferaw, B., Ichijo, E., Catmur, C., Bird, G., 2021. Alexithymia explains atypical spatiotemporal dynamics of eye gaze in autism. Cognition 212, 104710.

Deng, J., Ren, F., 2020. Multi-label emotion detection via emotion-specified feature extraction and emotion correlation learning. IEEE Transactions on Afective Computing 14, 475–486.

Engbert, R., Kliegl, R., 2003. Microsaccades uncover the orientation of covert attention. Vision research 43, 1035–1045.

Graham, L., Das, J., Vitorio, R., McDonald, C., Walker, R., Godfrey, A., Morris, R., Stuart, S., 2023. Ocular microtremor: a structured review. Experimental Brain Research 241, 2191–2203.

Hadders-Algra, M., 2022. Human face and gaze perception is highly context specific and involves bottom-up and top-down neural processing. Neuroscience & Biobehavioral Reviews 132, 304–323.

Hastie, T., Tibshirani, R., Friedman, J., 2009. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. 2nd ed., Springer Science & Business Media.

Herrmann, B., Augereau, T., Johnsrude, I.S., 2020. Neural responses and perceptual sensitivity to sound depend on sound-level statistics. Scientific Reports 10, 9571.

IDC, 2025. Idc predicts that the market landscape will be turbulent, with china’s smart glasses market expected to grow by 116.1% year-on-year in the first quarter of 2025. https://my.idc.com/getdoc.jsp?containerId= prCHC53610525.

Jacobsen, F., de Bree, H.E., 2005. A comparison of two diferent sound intensity measurement principles. The Journal of the Acoustical Society of America 118, 1510–1517.

Kashihara, K., Okanoya, K., Kawai, N., 2014. Emotional attention modulates microsaccadic rate and direction. Psychological research 78, 166– 179.

Kassner, M., Patera, W., Bulling, A., 2014. Pupil: an open source platform for pervasive eye tracking and mobile gaze-based interaction, in: Proceedings of the 2014 ACM international joint conference on pervasive and ubiquitous computing: Adjunct publication, pp. 1151–1160.

Khademi, F., Zhang, T., Baumann, M.P., Malevich, T., Yu, Y., Hafed, Z.M., 2024. Visual feature tuning properties of short-latency stimulusdriven ocular position drift responses during gaze fixation. Journal of Neuroscience 44.

Khalil, R.A., Jones, E., Babar, M.I., Jan, T., Zafar, M.H., Alhussain, T., 2019. Speech emotion recognition using deep learning techniques: A review. IEEE access 7, 117327–117345.

Khare, S.K., Blanes-Vidal, V., Nadimi, E.S., Acharya, U.R., 2024. Emotion recognition and artificial intelligence: A systematic review (2014–2023) and research recommendations. Information fusion 102, 102019.

Kollias, D., 2022. Abaw: Valence-arousal estimation, expression recognition, action unit detection & multi-task learning challenges, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2328–2336.

Krejtz, K., Żurawska, J., Duchowski, A.T., Wichary, S., 2020. Pupillary and microsaccadic responses to cognitive efort and emotional arousal during complex decision making. Journal of eye movement research 13, 10–16910.

Kulkarni, D., Dixit, V.V., 2025. Hybrid classification model for emotion detection using electroencephalogram signal with improved feature set. Biomedical Signal Processing and Control 100, 106893.

Lan, G., Scargill, T., Gorlatova, M., 2022. Eyesyn: Psychology-inspired eye movement synthesis for gaze-based activity recognition, in: 2022 21st ACM/IEEE international conference on information processing in sensor networks (IPSN), IEEE. pp. 233–246.

Le, J., Kou, J., Zhao, W., Fu, M., Zhang, Y., Becker, B., Kendrick, K.M., 2020. Oxytocin biases eye-gaze to dynamic and static social images and the eyes of fearful faces: associations with trait autism. Translational psychiatry 10, 142.

Leigh, R.J., Martinez-Conde, S., 2013. Tremor of the eyes, or of the head, in parkinson’s disease? Movement Disorders 28, 691.

Li, J., Jin, K., Zhou, D., Kubota, N., Ju, Z., 2020. Attention mechanismbased cnn for facial expression recognition. Neurocomputing 411, 340– 350.

Li, X., Zhang, Y., Tiwari, P., Song, D., Hu, B., Yang, M., Zhao, Z., Kumar, N., Marttinen, P., 2022. Eeg based emotion recognition: A tutorial and review. ACM Computing Surveys 55, 1–57.

Lin, Y.C., Intoy, J., Clark, A.M., Rucci, M., Victor, J.D., 2023. Cognitive influences on fixational eye movements. Current Biology 33, 1606– 1612.

Liu, B., Nobre, A.C., van Ede, F., 2022. Functional but not obligatory link between microsaccades and neural modulation by covert spatial attention. Nature Communications 13, 3503.

Liu, B., Nobre, A.C., van Ede, F., 2023a. Microsaccades transiently lateralise eeg alpha activity. Progress in Neurobiology 224, 102433.

Liu, S., Gao, P., Li, Y., Fu, W., Ding, W., 2023b. Multi-modal fusion network with complementarity and importance for emotion recognition. Information Sciences 619, 679–694.

Liu, X., Cheng, X., Lee, K., 2020. Ga-svm-based facial emotion recognition using facial geometric features. IEEE Sensors Journal 21, 11532–11542.

Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., Yger, F., 2018. A review of classification algorithms for eegbased brain–computer interfaces: a 10 year update. Journal of neural engineering 15, 031005.

Lu, Y., Zheng, W.L., Li, B., Lu, B.L., 2015. Combining eye movements and eeg to enhance emotion recognition., in: IJCAI, Buenos Aires. pp. 1170–1176.

Lv, F., Chen, X., Huang, Y., Duan, L., Lin, G., 2021. Progressive modality reinforcement for human multimodal emotion recognition from unaligned multimodal sequences, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2554– 2562.

Malevich, T., Buonocore, A., Hafed, Z.M., 2020. Rapid stimulus-driven modulation of slow ocular position drifts. Elife 9, e57595.

Mao, R., Liu, Q., He, K., Li, W., Cambria, E., 2022. The biases of pretrained language models: An empirical study on prompt-based sentiment analysis and emotion detection. IEEE transactions on afective computing 14, 1743–1753.

Martínez, G., Molero, J.D., González, S., Conde, J., Brysbaert, M., Reviriego, P., 2025. Using large language models to estimate features of multi-word expressions: Concreteness, valence, arousal. Behavior Research Methods 57, 1–11.

Martinez-Conde, S., Macknik, S.L., Troncoso, X.G., Hubel, D.H., 2009. Microsaccades: a neurophysiological analysis. Trends in neurosciences 32, 463–475.

Middya, A.I., Nag, B., Roy, S., 2022. Deep learning based multimodal emotion recognition using model-level fusion of audio–visual modalities. Knowledge-Based Systems 244, 108580.

Møller, F., Laursen, M., Tygesen, J., Sjølie, A., 2002. Binocular quantification and characterization of microsaccades. Graefe’s archive for clinical and experimental ophthalmology 240, 765–770.

Nag, A., Haber, N., Voss, C., Tamura, S., Daniels, J., Ma, J., Chiang, B., Ramachandran, S., Schwartz, J., Winograd, T., et al., 2020. Toward continuous social phenotyping: analyzing gaze patterns in an emotion recognition task for children with autism through wearable smart glasses. Journal of medical Internet research 22, e13810.

Neumann, D.L., Piercy, A., 2013. The efect of diferent attentional strategies on physiological and psychological states during running. Australian Psychologist 48, 329–337.

Ngai, W.K., Xie, H., Zou, D., Chou, K.L., 2022. Emotion recognition based on convolutional neural networks and heterogeneous bio-signal data sources. Information Fusion 77, 107–117.

Nie, W., Bao, Y., Zhao, Y., Liu, A., 2023. Long dialogue emotion detection based on commonsense knowledge graph guidance. IEEE Transactions on Multimedia 26, 514–528.

Nolen-Hoeksema, S., Wisco, B.E., Lyubomirsky, S., 2008. Rethinking rumination. Perspectives on psychological science 3, 400–424.

Olsen, A., Matos, R., 2012. Identifying parameter values for an i-vt fixation filter suitable for handling data sampled with various sampling frequencies, in: proceedings of the symposium on Eye tracking research and applications, pp. 317–320.

Robertson, J., Timmons, S., 2007. Non-invasive brainstem monitoring: The ocular microtremor. Neurological Research 29, 709–711.

Röhl, M., Uppenkamp, S., 2012. Neural coding of sound intensity and loudness in the human auditory system. Journal of the Association for Research in Otolaryngology 13, 369–379.

Rucci, M., McGraw, P.V., Krauzlis, R.J., 2015. Fixational eye movements and perception. Vision research 118, 1.

Rucci, M., Poletti, M., 2015. Control and functions of fixational eye movements. Annual review of vision science 1, 499–518.

Russell, J.A., 1980. A circumplex model of afect. Journal of personality and social psychology 39, 1161.

Ryle, J.P., Ryle, J.P., Al-Kalbani, M., Collins, N., Gopinathan, U., Boyle, G., Coakley, D., Sheridan, J.T., 2009. Compact portable ocular microtremor sensor: design, development and calibration. Journal of Biomedical Optics 14, 014021–014021.

Sagi, O., Rokach, L., 2021. Approximating xgboost with an interpretable decision tree. Information sciences 572, 522–542.

Simola, J., Le Fevre, K., Torniainen, J., Baccino, T., 2015. Afective processing in natural scene viewing: Valence and arousal interactions in eye-fixation-related potentials. NeuroImage 106, 21–33.

Song, T., Lu, G., Yan, J., 2020. Emotion recognition based on physiological signals using convolution neural networks, in: Proceedings of the 2020 12th international conference on machine learning and computing, pp. 161–165.

Suthaharan, S., MW, L.D., Zhang, M., Rossi, E.A., 2023. Microsaccade segmentation using directional variance analysis and artificial neural networks, in: 2023 IEEE 24th International Conference on Information Reuse and Integration for Data Science (IRI), IEEE. pp. 1–6.

Świrski, L., Bulling, A., Dodgson, N., 2012. Robust real-time pupil tracking in highly of-axis images, in: Proceedings of the symposium on eye tracking research and applications, pp. 173–176.

Tabbaa, L., Searle, R., Bafti, S.M., Hossain, M.M., Intarasisrisawat, J., Glancy, M., Ang, C.S., 2021. Vreed: Virtual reality emotion recognition dataset using eye tracking & physiological measures. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies

5, 1–20.

Taylor, R., Buonocore, A., Fracasso, A., 2024. Saccadic “inhibition” unveils the late influence of image content on oculomotor programming. Experimental Brain Research 242, 2281–2294.

Vehlen, A., Kellner, A., Normann, C., Heinrichs, M., Domes, G., 2023. Reduced eye gaze during facial emotion recognition in chronic depression: efects of intranasal oxytocin. Journal of Psychiatric Research 159, 50– 56.

Watson, D., Clark, L.A., Tellegen, A., 1988. Development and validation of brief measures of positive and negative afect: the panas scales. Journal of personality and social psychology 54, 1063.

Wu, Q., Dey, N., Shi, F., Crespo, R.G., Sherratt, R.S., 2021. Emotion classification on eye-tracking and electroencephalograph fused signals employing deep gradient neural networks. Applied Soft Computing 110, 107752.

Yan, X., Lin, Z., Lin, Z., Vucetic, B., 2023. A novel exploitative and explorative gwo-svm algorithm for smart emotion recognition. IEEE Internet of Things Journal 10, 9999–10011.

Yang, B., Huang, J., Chen, X., Li, X., Hasegawa, Y., 2023. Natural grasp intention recognition based on gaze in human–robot interaction. IEEE Journal of Biomedical and Health Informatics 27, 2059–2070.

Yik, M., Mues, C., Sze, I.N., Kuppens, P., Tuerlinckx, F., De Roover, K., Kwok, F.H., Schwartz, S.H., Abu-Hilal, M., Adebayo, D.F., et al., 2023. On the relationship between valence and arousal in samples across the globe. Emotion 23, 332.

Yin, Z., Zhao, M., Wang, Y., Yang, J., Zhang, J., 2017. Recognition of emotions using multimodal physiological signals and an ensemble deep learning model. Computer methods and programs in biomedicine 140, 93–110.

Yitzhak, N., Pertzov, Y., Guy, N., Aviezer, H., 2022. Many ways to see your feelings: Successful facial expression recognition occurs with diverse patterns of fixation distributions. Emotion 22, 844.

Yu, G., Herman, J.P., Katz, L.N., Krauzlis, R.J., 2022. Microsaccades as a marker not a cause for attention-related modulation. Elife 11, e74168.

Yu, X., Li, J., Gao, X., Yan, B., Li, H., Wang, Y., Sang, X., 2024. Smooth motion parallax method for 3d light-field displays with a narrow pitch based on optimizing the light beam divergence angle. Optics Express 32, 9857–9866.

Zemblys, R., Niehorster, D.C., Holmqvist, K., 2019. gazenet: End-to-end eye-movement event detection with deep neural networks. Behavior research methods 51, 840–864.

Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., Eickhof, C., 2021. A transformer-based framework for multivariate time series representation learning, in: Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2114–2124.

Zha, W., Liu, Y., Wan, Y., Luo, R., Li, D., Yang, S., Xu, Y., 2022. Forecasting monthly gas field production based on the cnn-lstm model. Energy 260, 124889.

Zhang, S., Yang, Y., Chen, C., Zhang, X., Leng, Q., Zhao, X., 2024. Deep learning-based multimodal emotion recognition from audio, visual, and text modalities: A systematic review of recent advancements and future prospects. Expert Systems with Applications 237, 121692.

Zhang, T., 2018. Facial expression recognition based on deep learning: a survey, in: Advances in Intelligent Systems and Interactive Applications: Proceedings of the 2nd International Conference on Intelligent and Interactive Systems and Applications (IISA2017), Springer. pp. 345– 352.

Zhang, X., Li, W., Ying, H., Li, F., Tang, S., Lu, S., 2020. Emotion detection in online social networks: a multilabel learning approach. IEEE Internet of Things Journal 7, 8133–8143.

Zhao, M., Adib, F., Katabi, D., 2016. Emotion recognition using wireless signals, in: Proceedings of the 22nd annual international conference on mobile computing and networking, pp. 95–108.

Zhu, H., Wang, C., Darmanola, V.R., Guo, H., Jin, W., Li, M., 2024. Bere: A novel video recommender system for virtual reality using human behavioral signals, in: Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pp. 770–784.