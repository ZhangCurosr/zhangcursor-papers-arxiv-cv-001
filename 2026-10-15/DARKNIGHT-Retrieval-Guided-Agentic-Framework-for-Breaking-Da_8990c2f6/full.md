# DARKNIGHT: Retrieval-Guided Agentic Framework for Breaking Dark Web CAPTCHA Gates

Binghui Wu, Yeo Zi Yi, Darren Tan, Manit Baser

National University of Singapore

Singapore

Levente Csikor, Dinil Mon Divakaran

A\*STAR Institute of Advanced Intelligence and Compute

Singapore

Abstract—Cyber Threat Intelligence (CTI) investigators who monitor dark web marketplaces for leaked credentials and breach listings must pass the diverse CAPTCHAs that operators deploy to block automated collection. Existing solvers either need unscalable per-site fine-tuning or rely on commercial Vision-Language Model (VLM) APIs, which surrender operational sovereignty: provider policies, Tor-exit blocking, government regulation, or model updates outside the investigator’s control can halt or alter the solver overnight, and per-call costs recur with every session and retry. We present DARKNIGHT, a self-improving agentic framework designed to run on a single workstation GPU inside an investigator’s Tor crawler and solve CAPTCHAs with small, locally hosted open-weight VLMs that no provider decision can withdraw, without per-market training or third-party services. For each CAPTCHA, DARKNIGHT retrieves successful and failed past attempts on similar images (dual-signal RAG), composes an image-specific preprocessing pipeline, and learns from the site’s pass/fail verdict. In a case study on four real-world dark web marketplaces, agentic tool selection outperformed parameter scaling. A deployable open-weight VLM with DARKNIGHT reached 83.2% on the most heavily defended market, 37.2 percentage points above its base model, outperforming even the frontier Claude Opus 5 (44%). DARKNIGHT recovered 48.2% of baseline failures at a 7.6% degeneration cost, indicating that selective tool use can avoid much of the degradation caused by uniform preprocessing.

Index Terms—Dark web, VLM, Agentic, Threat Intelligence

## I. INTRODUCTION

Dark web marketplaces demonstrate resilience against law enforcement efforts, with new platforms rapidly emerging to replace those that are disrupted [1]. The scale of illicit activity continues to grow: data breaches posted on underground forums increased by 43% in 2024, compromised credentials rose by 34%, and the number of ransomware group leak sites grew by 53% over the same period [2]. Beyond data and credentials, dark web forums host exploit listings that are subsequently linked to known vulnerabilities in production systems [3]. Phishing kits are openly traded and deployed at industrial scale [4], [5]. The emergence of malicious LLM services, sold openly on underground markets to lower the barrier to crafting phishing lures and malware, further amplifies the threat [6]. Cryptocurrency facilitates pseudonymous settlement of nearly all transactions across these markets [7].

Security teams monitor dark web marketplaces to detect, as early as possible, newly listed data breaches, emerging ransomware strains, and exposed customer records before they are exploited [1], [8]. The value of this monitoring scales with coverage and speed, which makes automated, large-scale collection a requirement rather than a convenience. To resist such automation, marketplaces deploy several defensive layers, including headless-browser blocking, dynamic link generation, request-rate limiting, and CAPTCHAs [9]–[11]. Most of these layers can be circumvented through careful crawler engineering [11], [12]. CAPTCHAs are the exception. Deployed at registration, login, and checkout, they are designed to admit humans and turn away bots [13], [14]. It is the chokepoint that if overcome cheaply and at scale, makes automated dark web measurement feasible; otherwise collection stalls behind manual intervention [10].

What sits at that gate is not a clear web CAPTCHA. Mainstream services such as reCAPTCHA, hCaptcha, and Cloudflare Turnstile are largely absent from dark web sites for structural reasons [13]–[16]. They require the client to contact a third-party endpoint the site operator does not control. On a hidden service, that adds the risky de-anonymization layer and logging surface; neither do legit CAPTCHA provider serve the dark web market. Moreover, those endpoints also degrade or block Tor exit traffic. For these anonymity reasons mainstream CAPTCHAs are not used on the dark web [14]. The visitors compound the issue: many Tor users disable JavaScript to shrink their fingerprinting surface, which makes the behavioral, slider, and interactive challenges common on the clear web impractical to deploy. Dark web site operators therefore build their own [9], [10], [13]; the result is a population of bespoke, locally hosted, server-rendered CAPTCHAs [13]. Due to the lack of JavaScript, they tend to be less diverse than commercial offerings using standard image libraries, random fonts and additive noise. The answer is bound to a server-side session, often at low character entropy. The dominant form is the OCR text CAPTCHA. It appears in a few recurring variants such as distorted Latin alphanumeric strings, image-rendered arithmetic, and non-Latin script. Non-Latin rendering is a low-cost anti-OCR measure on regionally targeted markets. It defeats common off-the-shelf recognizers trained predominantly on Latin text.

Existing empirical evaluations of modern CAPTCHAs study the clear web systems above [17], [18]. They explicitly place dark web challenges out of scope [19], leaving this design space comparatively unmeasured. Prior works approach solving dark web CAPTCHAs from two directions, neither of which fits a dark web Cyber Threat Intelligence (CTI)

deployment. The first trains a recognizer per CAPTCHA type. CNN (Convolutional Neural Network) and GAN (Generative Adversarial Network) solvers reach high accuracy on the styles they are trained on [10], [13], [14]. However, they require labeled data and retraining for each new style. Besides, they do not generalize across the heterogeneous, churning set of markets a collector must track. The second is general visual solving with vision-language models (VLMs). Currently, they can solve a range of clear web CAPTCHAs [17], [20], [21]. On the dark web, a VLM has so far been used only as a component: an off-the-shelf model embedded in a singlemarket navigation tool [11]. There, solving is a step in data acquisition, not the object of study. Consequently, prior darkweb work has not characterised how VLM-based solvers behave across model scales, preprocessing pipelines, or agent designs. These are the questions a measurement study of CAPTCHA solving must answer. Neither direction studies dark web CAPTCHA solving as a problem in itself. The unstudied space spans heterogeneous market styles, including the non-Latin scripts of regionally targeted markets. It sits under the operational constraints of CTI deployment. And it calls for a solver that adapts without per-type retraining.

Frontier commercial LLM APIs are not good options for a CTI solver, because they leave the investigator without operational AI sovereignty: control over whether, how, and at what cost the solver keeps running. Even a frontier model that solves every CAPTCHA today would have to be removed from the pipeline by decisions the investigator does not control. First, provider usage policies prohibit using the models to circumvent CAPTCHAs, and such policies can be tightened at any time. Second, regulation can cut access without warning: in June 2026, a US export-control directive led Anthropic to suspend access to two of its frontier models for all customers worldwide for about three weeks [22]. Third, providers decide which networks they serve. When the collection host routes its traffic through Tor, major providers challenge or block Tor exit traffic with access-denied errors and forced humanverification prompts, so reachability cannot be assumed [23]. Moreover, Tor rotates its exit endpoint regularly; to a provider this resembles an account hopping over IP addresses, which injects additional verification mid-run and stalls an unattended pipeline until a human intervenes. Fourth, providers update and retire model versions on their own schedule, and solving behavior changes with them: on our most heavily defended dataset, four Claude models range from 1.4% to 44% accuracy (cf. App. D). Finally, the per-call cost recurs with every session and retry, since several attempts may be needed before a page opens. Closed APIs are therefore not a practical option (nor do they perform well; see App. D); the deployed solver must be open-weight and self-hosted, so that the investigator runs a fixed model checkpoint that no provider decision can withdraw or alter.

Two further requirements follow from the field setting. The solver must be small enough to run on a single workstation GPU (see Sec. VI-A for details) for CTI deployments (≤ 30B models) rather than a GPU cluster in a datacenter. The solver has one hard timing constraint and one soft one. The hard constraint is server-side session expiry. A correct answer arriving after the session has timed out is rejected, and there is no client-side workaround [13]. The soft constraint is submissionrate detection. Markets that escalate on rapid submission can be defeated by holding the solver output and submitting after a randomised delay. The hard constraint binds the solver design. The soft constraint binds only the submission policy.

We identify three gaps that this paper addresses.

Small open-weight VLMs for solving CAPTCHAs from the dark web within the operation constraints are understudied. Prior dark web CAPTCHA solvers use perstyle trained models that require labelled data per market [10], [14]. Other work treats solving as a downstream step in a navigation pipeline [11]. Modern VLM-based solvers are evaluated on the clear web, predominantly with frontier-closed or largeparameter models [20], [21]. The capability of small openweight VLMs in solving real dark web CAPTCHAs under time-bounded operational constraints remains understudied.

G<sub>2</sub> Understudied deployable enhancement methods for open-weight VLMs. Prior work can be categorized broadly into two approaches: per-style retraining and frontier-API agentic solving. No prior method combines agentic enhancement with a deployable open-weight backbone.

G<sub>3</sub> Non-Latin and heterogeneous-style CAPTCHAs from dark web are yet to be addressed. Existing dark web CAPTCHA work targets English-language, Western markets, and Latin-script inputs [10], [13], [14]. Recent threat intelligence reports identify active Chinese-language underground markets as hubs that preserve operational anonymity across clearnet, darknet, and messaging platforms [24]. The study of any solver, classical or VLM-based, on non-Latin dark web CAPTCHAs is missing in literature.

We argue that these gaps leave us with a single viable profile to address the challenges: an open-weight solver that runs locally, stays small, and operates within a timing window. We hypothesize that the bottleneck is not model capacity but toolselection strategy, informed by past solves.

To this, we design and develop DARKNIGHT, a selfimproving agentic solver. DARKNIGHT is given a small library of image-preprocessing tools (see Sec. V). For each CAPTCHA, DARKNIGHT perceives the input and retrieves past attempts on visually similar images. Then, it composes a preprocessing pipeline, solves the cleaned image, and updates from the site’s verdict [25]. The same open-weight VLM does both planning and solving, with no additional model to host. That closed verdict-feedback loop enables stream-level improvement without retraining.

We present this work as a case study of one framework on four marketplaces, evaluated offline; where we generalize, we show what is feasible, not what holds for every market or model. Our primary contributions are summarized as follows:

1) We design DARKNIGHT, an agentic enhancement layer for open-weight small VLMs operating under dark web deployment constraints. DARKNIGHT combines chain-ofthought visual analysis, a dual-signal RAG knowledge database (retains both successful and failed records of CAPTCHA solving, §V-C), five preprocessing operations, and automated ground-truth feedback (§V-E). Across 2,000 CAPTCHAs from four real-world dark web marketplaces and nine model backbones, DARKNIGHT improves accuracy in 30 out of 32 (93.75%) configurations, with an average absolute improvement of 13.33%. In particular, on the most challenging dataset $( \mathcal { D } _ { 1 } )$ , the average improvement across all backbones is +30.48%. Such improvement on all four marketplaces is achieved without any per-site retraining, with a single configuration handling alphanumeric, math, and Chinese CAPTCHAs within the same mixed-style market.

2) We present the first systematic evaluation of small openweight VLMs (1B–30B) on real dark web CAPTCHAs under operational constraints. We show that a generalpurpose 27B model equipped with DARKNIGHT (83.2%) on $\mathcal { D } _ { 1 }$ outperforms the raw baselines of Step3-VL-10B [26] (63.6%), the strongest bare model we evaluated, a larger general-purpose vision language model, Qwen3- VL-30B-A3B (42.4%), and even a frontier model Claude Opus 5 (44%), showing that adding DARKNIGHT to a lightweight model beats scaling up the bare model.

3) We identify and quantify the degeneration-recovery effect in CAPTCHA preprocessing. Across all Planner decisions, including cases where no preprocessing is selected, DARKNIGHT recovers 48.2% of baseline failures while degrading only 7.6% of correct predictions, yielding a 5.8× recovery-to-degeneration ratio. The Planner selects no preprocessing in 86.9% of cases on $\mathcal { D } _ { 2 }$ and only 26.6% on the heavily defended $\mathcal { D } _ { 1 }$ , indicating that it intervenes selectively rather than uniformly.

To support reproducibility, we publicly release the complete source code, experiment configurations, and all four annotated CAPTCHA datasets<sup>1</sup>.

## II. RELATED WORK

## A. Dark Web Analytics and Anti-Crawling Mechanisms

Dark web marketplaces continuously deploy robust anticrawling defenses to shield illicit trade operations from automated CTI monitoring [27]. Unlike the clear web, where operators frequently deploy sophisticated, behaviorally driven validation modules (e.g., reCAPTCHA or hCaptcha), dark web administrators operate under strict Tor network constraints that fundamentally preclude third-party JavaScript execution to prevent user de-anonymization [13]–[16]. Consequently, marketplaces rely predominantly on bespoke, locally hosted, and server-rendered CAPTCHAs. These mechanisms are characterized by aggressive background noise, hyper-kerning, variable character lengths, and mathematical puzzles intentionally engineered to defeat conventional scraping tools [9].

## B. Classical ML-Based CAPTCHA Solvers

Early automated CAPTCHA resolution relied on specialized computer vision and deep learning pipelines [28], [29]. Even prior to the emergence of multimodal foundation models, researchers demonstrated that machine learning systems could successfully attack visual reasoning CAPTCHAs that were explicitly designed to resist automated solving [30]. For dark web CTI, frameworks like DW-GAN [10] and “Tick Tock Break the Clock” [14] achieve high success against bespoke challenges. However, they are operationally brittle, as they overfit to specific data distributions and require labor-intensive manual retraining whenever marketplace administrators update their visual obfuscation logic. We do not compare against them on accuracy. Given a labeled training set for a target style these solvers score highly, but that assumption is exactly what fails in CTI: a collector tracking churning, heterogeneous markets has no per-style labels at the moment of solving and cannot retrain as styles change. DARKNIGHT targets this regime, achieving deployable accuracy without per-style supervision.

## C. Generalized VLM Solvers and Agentic Frameworks

Recent works have demonstrated the capability of VLMs in zero-shot multimodal reasoning. Recent benchmark suites such as MCA-Bench [17] and CAPTURE [18] systematically evaluate the effectiveness of modern VLMs against diverse CAPTCHA schemes. MCA-Bench [17] demonstrates that fine-tuned VLMs can consistently overcome cross-modal verification challenges, while CAPTURE [18] provides a dedicated evaluation framework for assessing LVLM performance on CAPTCHA-solving tasks. More recently, Cognition provides a large-scale evaluation of multimodal LLM CAPTCHA-solving capabilities and investigates defensive strategies against emerging MLLM-based attackers [31]. Stateof-the-art agentic frameworks have further operationalized these capabilities against complex security mechanisms. Halligan is an agentic VLM framework, which reduces intricate visual CAPTCHAs into heuristic search problems overlaid onto graphical user interface action spaces [20]. Concurrently, the Oedipus framework utilizes an LLM-enhanced reasoning pipeline to parse and bypass advanced logicbased CAPTCHAs using a specialized Domain Specific Language [21]. Despite their adaptability, these frontier agentic solvers uniformly rely on commercial cloud APIs for inference. This reliance violates the strict operational requirements of dark web CTI deployments discussed above. As with the trained solvers above, we therefore cannot run these frontier-API agents under our deployment constraints, and exclude them from our evaluation.

## III. THREAT MODEL

Setting. We consider a security investigator who has to automate CAPTCHA solving on dark web marketplaces for scalable CTI gathering (see Fig. 1). The CAPTCHA’s purpose is to gate automated access; the investigator’s goal is to bypass that gate at scale. We study the defender (the site’s CAPTCHA system) by playing the attacker (the automated solver), for the larger social good. From the marketplace operator’s perspective, the investigator is a bot. We discuss ethical considerations in Appendix A.

![](images/b91ca779a5767a1bd52ee1da1e2032aff2fca4c73c9cb1aeff235f2689ceb509.jpg)  
Fig. 1: Threat model

Investigator. The investigator operates an automated crawler over Tor. The CAPTCHA solver interacts with the target site through a standard browser session. It receives CAPTCHA images as rendered in the browser and submits candidate text solutions. Each submission yields binary feedback: the protected page loads (solution accepted) or a new CAPTCHA is served (solution rejected). The solver has no view of the CAPTCHA generation algorithm, character set, font library, or distortion parameters. No labelled dataset of marketplacespecific image and solution pairs is available at the start. The solver may, however, accumulate experience over time. It may record candidate solutions, outcomes, and preprocessing strategies that succeeded or failed on visually similar images. However, this memory is feedback-derived, not a labelled training set. Per Sec. I, the solver must be open-weight, locally hosted, and resource-bounded.

Site operator. The site operator controls the full CAPTCHA stack: generation algorithm, fonts, distortion, script choice, length, and presentation. The operator also controls the serverside access-control policy. The operator enforces requestfrequency thresholds. Exceeding a threshold triggers escalating countermeasures. These include serving harder CAPTCHAs, invalidating sessions, and blocking the current Tor exit IP [23]. Exit-IP blocking forces the client to rotate circuits, since circuit control belongs to the client. The operator may also update CAPTCHA designs over time in response to observed bypassing attacks.

Scope and assumptions. This work targets text-based CAPTCHAs, the dominant variant on dark-web marketplaces [10], [13]. The text-based scope spans Latin alphanumeric, arithmetic, and non-Latin (e.g., Chinese) variants observed in our data. For this current work, we exclude slider puzzles, image-selection grids, audio CAPTCHAs, and any challenge requiring JavaScript execution beyond text rendering. We assume the Tor Browser renders CAPTCHA images faithfully; i.e., what the solver receives matches what a human user would see. We assume the operator does not adversarially adapt CAPTCHA generation to our specific system. Tor network availability and anonymity are out of scope. A solve is successful when the candidate solution is accepted and the protected page loads. All experimental results are obtained offline against a static labeled dataset (Section VI-B). Livedeployment characteristics that depend on operator adaptation are not exercised in this evaluation.

## IV. DESIGN RATIONALE

Below we present the design principles behind DARK NIGHT (Sec. V).

1 From Monolithic Solvers to Decomposable Modules. A single VLM invocation that simultaneously analyzes a CAPTCHA to perform multiple operations (e.g., removing background noise and OCR), treats solving as a monolithic task. A CAPTCHA-solving framework should therefore decompose the task into distinct perception, planning, and execution stages, allowing each component to focus on a welldefined objective. This modular design improves controllability over individual components and enables targeted error diagnosis. This principle motivates the three-phase perceiveplan-execute architecture described in Sec. V.

2 Utilizing Transformation Experts. Beyond staging the pipeline, the question is who performs each step. Identifying visual characteristics is a perception task suited to VLMs; the resulting image transformation, by contrast, is typically deterministic. VLM should be used to analyze visual distortions and interference patterns, while preprocessing operations such as denoising, line removal, and binarization should be delegated to classical computer vision techniques. This principle motivates the separation between the Planner’s visual analysis component (Sec. V-B) and the deterministic preprocessing toolbox (Sec. V-D).

3 Deterministic Tasks are Programmable. Separately from image preprocessing, the symbolic steps after recognition are also deterministic. Once text is extracted, many downstream operations such as evaluating an arithmetic expression can be solved programmatically rather than by the model. Programmable tasks thus should be delegated to dedicated tools, reserving the VLM for perception and interpretation while avoiding unnecessary sources of stochastic error. This principle motivates the Executor’s use of a dedicated arithmetic solver described in Sec. V-D.

4 Learn from Experience. CAPTCHAs from the same marketplace often share recurring visual characteristics, so a preprocessing strategy that succeeds or fails on one is likely to behave the same on similar ones. A solver should therefore retain both successful and unsuccessful attempts, and retrieve visually similar records when handling a new CAPTCHA, reusing what worked and avoiding what degraded performance. This principle motivates the dual-signal RAG knowledge base (Sec. V-C).

5 Plug-and-Play for Scalability. Dark web CAPTCHA styles evolve continuously, making repeated model retraining impractical for operational deployments. The preprocessing capabilities should comprise modular tools rather than learned model behaviors. The framework is designed so that a new tool can be added without modifying the underlying model weights; we leave empirical validation of adding new tools to future work. This plug-and-play design may facilitate rapid adaptation to emerging defenses, and enable scaling without incurring the cost of continual retraining. This principle motivates the modular toolbox and feedback-driven adaptation mechanisms described in Sec. V-D and V-E.

## V. THE DARKNIGHT FRAMEWORK

## A. System Overview

Fig. 2 illustrates the architecture of DARKNIGHT. Given a CAPTCHA image from a dark web marketplace, the system operates in a three-phase loop: plan, execute, and record.

i) In the plan phase, the Planner Agent receives the raw CAPTCHA image and produces a structured visual analysis describing the image’s properties. This analysis is used to query the RAG Knowledge Base, which retrieves the most relevant past solving records (both successes and failures) based on visual and semantic similarity. The planner agent then uses the retrieved records alongside its own visual analysis and composes a preprocessing pipeline tailored to the current image. ii) In the execute phase, The Executor runs this pipeline step-by-step from the toolbox, passes the processed image to the OCR engine and submits the recognized text (or the solution of an equation) to the target website. iii) In the record phase, the website returns a binary pass/fail verdict. The complete attempt, including image, visual analysis, pipeline, OCR output, and outcome, is written back to the RAG Knowledge Base regardless of the result, closing the feedback loop.

![](images/edf24a143041f7eaa3a228db5c9f492774afc51911d49e18af1c7269cd800603.jpg)  
Fig. 2: Architecture Diagram of the pipeline

As the system encounters more CAPTCHAs from the same marketplace, the RAG store accumulates site-specific experience. This enables the Planner to progressively make better tool-selection decisions based on the past experience, with no model retraining or fine-tuning. This is the key design property that distinguishes DARKNIGHT from prior per-style trained solvers. A critical design choice is that the Planner VLM serves dual roles: it performs the visual analysis and planning in the perceive phase, and also serves as the OCR engine in the execute phase. Since the solver is already loaded, routing planning through it adds no memory overhead. This keeps the system lightweight since only a single model is needed to be loaded, even if VLM invocation happens twice, resulting in the same memory requirements as a naive single VLM execution pipeline. The full agentic loop introduces additional local inference time overhead. Section VII-E quantifies this computational cost across the evaluated VLMs.

## B. Planner Agent

The Planner Agent is a lightweight, open-weight VLM that operates in three stages to perform visual analysis, experience retrieval, and pipeline planning.

Stage 1: Visual Analysis. The Planner receives a raw CAPTCHA image and outputs a structured JSON describing eight visual properties shown in Table I. This structured output serves two purposes: i) it informs the RAG retrieval query (Stage 2) and ii) it provides the Planner with an explicit representation of the image’s properties for reasoning in Stage 3.

TABLE I: Visual features extracted by Stage 1 visual analysis.
<table><tr><td>Features</td><td>Values</td></tr><tr><td>captcha_type char_count noise</td><td>alphanumeric |math |chinese Integer count of CAPTCHA characters none |color_lines: multiple coloured wavy/diagonal</td></tr><tr><td>background</td><td>curved or arc-shaped bands | vertical_lines: thin dark vertical strokes through characters | dot s: scattered isolated dot clusters | salt_pepper: fine pixel-level random grain</td></tr><tr><td>distortion</td><td>clean | gradient |noisy | patterned nonewave |blur|perspective</td></tr><tr><td>char_quality</td><td>normal |hollow | faded | thin | broken | thick</td></tr><tr><td>is_math,</td><td>Boolean flag if capt cha_type is math</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Boolean flag if capt cha_type is chinese</td></tr><tr><td>is_chinese</td><td></td></tr><tr><td></td><td></td></tr><tr><td>description</td><td>Free-text sentence for RAG text-embedding similarity</td></tr></table>

Stage 2: Experience Retrieval. Using the structured analysis from Stage 1, the system queries the RAG Knowledge Base (Sec. V-C) for past solving records on visually similar CAPTCHAs. The query uses two signals: text similarity of the image description and perceptual hash [32] similarity of the image itself. The top-k records, annotated with their preprocessing pipelines and website pass/fail outcomes, are appended to the Planner’s context for Stage 3. This stage is automated and does not require additional VLM inference.

Stage 3: Pipeline Planning. The Planner receives its own visual analysis (Stage 1) and the retrieved records (Stage 2) as context, and produces a complete processing plan. The plan specifies two components: (i) preprocess\_pipeline, an ordered list of tools to apply from the toolbox (Sec. V-D), each with its parameters (e.g., binarize with method=otsu); and (ii) post\_ocr, a flag indicating whether to activate the math expression solver after OCR. The Planner is prompted with a system instruction that defines the available tools, their expected inputs and effects, and the output JSON format. When retrieved records are available, the Planner is instructed to prefer strategies that previously succeeded on similar images and avoid strategies that previously failed. Table XV (Appendix. F) summarizes the API parameters for VLM calls.

## C. Dual-signal RAG Knowledge Base

The optimal preprocessing strategy for a given CAPTCHA cannot be deduced from static visual analysis alone. Hence, DARKNIGHT maintains a persistent dual-signal RAG knowledge base of past execution records, which the Planner queries dynamically at inference time. Such RAG knowledge base retains both successful and failed solving attempts in distinct partitions (Table II), bounded to 100 records per experimental configuration. Successful records function as positive blueprints, guiding the Planner to replicate proven pipelines when encountering matching visual features. Failed records serve as negative constraints, explicitly warning the Planner against repeating historically destructive tool sequences. This dual-signal retrieval design directly addresses the degeneration-recovery trade-off (Sec. VII-B). Rather than blindly applying an operation and discovering at test time that it degrades a specific CAPTCHA archetype, the Planner actively retrieves prior failure evidence to proactively prune hazardous tools from its execution plan. In all experiments, the RAG store applies a cold-start protocol. It begins empty and is populated incrementally as each dataset is processed in a single sequential pass: the first image is solved with no retrieved experience, and each subsequent image benefits from the records accumulated by all preceding attempts.

1) Storage Layout: The RAGStore adopts a file-percombination storage scheme: one JSON file is created for each unique [dataset, planner model, ocr model] triplet, stored under rag\_store/ with the naming convention:

$$
\{ { \mathrm { d a t a s e t } } \} \_ { \mathrm { - } } \{ \mathrm { p l a n n e r } \} \_ { \mathrm { - } } \{ \mathrm { o c r } \} \cdot \mathrm { j s o n }
$$

For the primary per-dataset experiments, this storage scheme ensures isolation between different configurations, and no cross-dataset retrieval is performed. Each file holds up to 200 records, distributed across two independent slots as shown in Table II: up to 100 successful and 100 failed records. This design allows us to measure the effect of DARKNIGHT independently on each marketplace. In Sec. VII-D, we additionally evaluate a mixed-site setting in which CAPTCHAs from all four marketplaces are processed sequentially using a single shared RAG store for each evaluated backbone.

TABLE II: RAGStore slot structure.
<table><tr><td>Slot</td><td>Content</td><td>Capacity</td><td>Eviction Policy</td></tr><tr><td>correct</td><td>Successful predictions</td><td>100 records</td><td>Least-retrieved first</td></tr><tr><td>failed</td><td>Failed predictions</td><td>100 records</td><td>Least-retrieved first</td></tr></table>

2) Quality-Based Eviction: Each record tracks a \_retrieval\_count field that is incremented each time the record is surfaced to the planner. When a slot reaches capacity, the system applies a least frequently used (LFU) eviction policy, displacing the record with the lowest cumulative retrieval count [33]. This policy is self-reinforcing as records that frequently match new queries survive indefinitely, while records that are never retrieved and thus never contribute to planning, are replaced by more relevant newcomers. The net effect is a memory that organically specializes toward the most frequently encountered CAPTCHA configurations.

3) Deduplication: On every add() call, the system computes the weighted similarity between the incoming record and all existing records in the target slot. If any existing record has a similarity $\mathrm { s c o r e } \geq 0 . 9 9$ , the insert is suppressed. This prevents near-duplicate examples from crowding out diverse records and preserves the breadth of the memory.

4) Similarity Metric: The composite similarity score combines two signals:

$$
s = 0 . 7 \cdot \cos ( \mathbf { f } _ { q } , \mathbf { f } _ { r } ) + 0 . 3 \cdot \cos ( \mathbf { e } _ { q } , \mathbf { e } _ { r } )\tag{1}
$$

where $\mathbf { f } \in \{ 0 , 1 \} ^ { 1 9 2 }$ is the perceptual hash (pHash-RGB) of the image and $\textbf { e } \in \mathbb { R } ^ { 3 8 4 }$ is the sentence-transformer embedding of the free-text description. The pHash is computed over the RGB channels of the image (pHash-RGB) [32], which captures structural and color gradient patterns without storing the full image. The free-text description is embedded using $\mathtt { a l 1 - M i n i L M - L 6 - v 2 }$ [34], [35], a lightweight sentencetransformer that maps visual descriptions to a dense semantic space. Feature cosine (weight 0.7) dominates because it is deterministic and encodes the exact visual properties that drive tool selection. Text cosine (weight 0.3) contributes semantic context: two CAPTCHAs described as “wavy blue lines over Chinese characters on a gradient background” may share only partial feature overlap but should still trigger similar preprocessing decisions. Records whose weighted similarity falls below τ = 0.7 (RAG\_SIMILARITY\_THRESHOLD) are dropped before being returned to the planner.

5) Split Retrieval: The query\_split(description, features, top\_k) method searches the correct and failed slots independently and returns the top $k / 2$ records from each. We set k=5 throughout all experiments, retrieving up to two successful and two failed records per query (with one additional record from whichever slot has the higher top match). This guarantees balanced exposure to positive and negative examples regardless of the relative sizes of the two slots. Each retrieved record is presented to the planner with the following fields:

• similarity score: the weighted score from Equation (1).

• description: the free-text visual description.

• tools applied: the ordered list of preprocessing tools used in that run.

• slot: correct or failed, indicating the outcome.

• note: a natural-language annotation “✓ succeeded replicate if features match” or “× failed avoid the same tools on similar images”.

• ocr result: the raw OCR output, included only for failed records to assist diagnosis.

Relevance thresholds. Table III summarizes the threshold policy applied to retrieved RAG records. Correct-slot records are adopted, verified, or ignored according to their similarity to the query; failed-slot records serve as negative signals regardless of similarity, and a tool that appears repeatedly in failed records is treated as banned for that CAPTCHA style. Positive and negative records. A distinguishing feature of the RAG store is that it retains both successful and failed attempts. Failed records are explicitly valuable: when the Planner sees that a particular pipeline failed on a visually similar CAPTCHA, it avoids repeating the same strategy. In the prompt, failed records are formatted as “the following pipeline was tried on a similar image and FAILED: [...]” to clearly signal that the strategy should be avoided.

TABLE III: Planner policy for retrieved RAG records.
<table><tr><td>Slot</td><td>Similarity</td><td>Action</td></tr><tr><td rowspan="3">Correct</td><td>≥ 0.8</td><td>Adopt pipeline if features match</td></tr><tr><td>0.6-0.8</td><td>Verify each feature before copying</td></tr><tr><td>&lt; 0.6</td><td>Ignore entirely</td></tr><tr><td rowspan="3">Failed</td><td>≥ 0.6</td><td>Avoid same tools</td></tr><tr><td>0.5-0.6</td><td>Still a negative signal</td></tr><tr><td>Repeated</td><td>Treat that tool as banned</td></tr></table>

## D. Executor and Toolbox

The Executor receives the Planner’s pipeline specification and runs it in three phases.

Phase 1: Image Preprocessing. The toolbox provides five preprocessing operations to neutralise a specific class of visual degradation commonly encountered in CAPTCHA images. Table IV summarises the tools, their target degradation, and the conditions under which each should be applied.

TABLE IV: Preprocessing tools and their intended use cases.
<table><tr><td>Tool</td><td>Description and Usage Condition</td></tr><tr><td>color_extract</td><td>Removes colored interference lines by iso- lating non-color pixels. Use when wavy or diagonal lines of a distinct color cross the text region.</td></tr><tr><td>sharpen</td><td>Restores character edge crispness via un- sharp masking. Use when blur distortion has softened character boundaries.</td></tr><tr><td>denoise</td><td>Removes small connected components be- low an area threshold. Use to clean residual noise dots or on patterned backgrounds.</td></tr><tr><td>remove_line</td><td>Detects and removes a single thin dark diagonal or near-horizontal line via the Hough transform. Use when such a line crosses all characters.</td></tr><tr><td>binarize</td><td>Converts a faded-grayscale image to black- and-white using Otsu&#x27;s global thresh- old. Use only when background = &quot;clean&quot;.</td></tr></table>

Three hard ordering constraints govern pipeline composition and are surfaced as explicit rules in the Planner prompt:

1) color\_extract must precede sharpen and binarize, as both destroy the color information on which color-based line removal relies.

2) denoise is applied after color\_extract to eliminate residual dot artefacts produced by color separation.

3) binarization is used only when background = ‘‘clean’’; applying Otsu’s global threshold to a structured or gradient background corrupts the result.

The Planner may select zero, one, or several tools depending on the visual analysis, and the tools are designed to be freely composable within these constraints.

As shown in Fig. 3, for instance, color\_extract → denoise → binarize is a valid three-step pipeline for a clean-background CAPTCHA overlaid with colored lines and residual dot noise. Conversely, misapplied preprocessing operations cause more visual obfuscation, making the CAPTCHA even harder for the underlying VLM (see the comparison in Fig. 4). These cases concretely illustrate design principle 4 (Sec. IV): the same tool can help one CAPTCHA archetype and harm another, so tool selection must be guided by retrieved success and failure records rather than applied uniformly. This is the trade-off the dual-signal RAG store (Sec. V-C) is built to manage, and which we quantify in Sec. VII-B.

![](images/d0f1344b5f9418c2638b8ccaf224a9cf2f81224e6920e2d2f084f07104540983.jpg)

![](images/b49b4367b6bee1edc7700c16de136b89d4c9e8faea6a4207b0ccb62c24e8db41.jpg)

(a) Color Extract on D<sub>1</sub> Chinese CAPTCHA.  
![](images/2f05e06e3f7035674445fc798d97dc685ac77d8f56dd82fe8a3b4bdf93edac2a.jpg)

![](images/bd018e3b2dfff6d661eb768effe6e51179f6459770dafa3dda5293c188029d05.jpg)  
(b) Denoise on $\mathcal { D } _ { 4 }$  
Fig. 3: Effective use of preprocessing tools.

![](images/f0aae6f6ddc444de07a053a3986d9e1abd0e40de4afd2f6ef9ecec70a411fe4b.jpg)

![](images/cc48422367925f98c79808bcdcd21b8a6be80b11e1d220f9b3e345569a965333.jpg)

![](images/e99b1da80b92092a5455d5f0de99e42ede603a398239e34b59107bcf30215bbd.jpg)

$$
\mathcal { D } _ { 2 }
$$

![](images/84d62f4a153df133274e95f5813699d1b9fbcb97ad19b695ff3cd6a59ebf813c.jpg)  
(b) Failed Binarize on $\mathcal { D } _ { 2 }$  
Fig. 4: Effective vs. degenerative use of preprocessing tools.

Phase 2: OCR engine. After preprocessing, the Executor invokes the same VLM used for planning, now with a typespecific OCR prompt:

• Alphanumeric prompt: instructs the model to output exactly the characters visible in the image, handling mixed letters and digits.

• Math prompt: instructs the model to extract the arithmetic expression as written (e.g., 3+5=?), without solving it.

• Chinese prompt: instructs the model to output each character independently, without inferring semantic meaning between characters.

The prompt type is determined by the captcha\_type field from the Planner’s Stage 1 analysis.

Phase 3: Post-processing. For math CAPTCHAs, the OCR output (e.g., 3+5=?) is passed to a rule-based arithmetic solver. Before evaluation, the system validates the expression: it must contain at least one arithmetic operator and

(a) D<sub>1</sub>

Alphanumeric, colored interference lines, wave distortion.

an equals sign or question mark, and must not contain alphabetic characters. If validation fails, the system reclassifies the CAPTCHA as alphanumeric and re-runs OCR with the alphanumeric prompt. This fallback catches misclassifications where the Planner incorrectly identified a CAPTCHA as a math expression.

One additional recovery mechanism: If the Planner labels a CAPTCHA as alphanumeric but the OCR output contains = or ?, the executor automatically attempts to solve the math before returning a result.

## E. Ground-truth Feedback Loop and Self-Improvement

After each solve attempt, the Executor writes a complete record to the RAG Knowledge Base:

$$
\mathsf { \{ i m a g e \_ h a s h , ~ d e s c r i p t i o n , ~ p i p e l i n e , ~ p a r a m s , } 
$$

This write occurs for both passed and failed attempts. The resulting feedback loop drives DARKNIGHT’s self-improvement through three mechanisms:

Positive reinforcement. When a pipeline succeeds, future CAPTCHAs with similar visual properties will retrieve this record as a high-confidence reference, biasing the Planner toward the same strategy. Over consecutive images from the same site, successful strategies are reinforced and converge.

Negative avoidance. When a pipeline fails, the record is stored with a FAIL label. Future retrievals surface this record alongside successful ones, explicitly informing the Planner that the failed strategy should be avoided. This is particularly important for addressing the degeneration-recovery trade-off: if a preprocessing step degraded a similar image in the past, the Planner learns to skip it.

Cold start and warm-up. When DARKNIGHT first encounters a new marketplace, the RAG store contains no records for that site’s CAPTCHA style. In this cold-start scenario, the Planner relies solely on its own visual analysis (Stage 1) without retrieved experience. As attempts accumulate, typically within 5-10 images as per our observation, the RAG store builds sufficient site-specific history for retrieval to become effective.

Cross-site adaptation. Because retrieval is based on visual similarity rather than site identity, experience can be shared across marketplaces with similar CAPTCHA characteristics. We empirically evaluate this capability using the mixed-site experiment in Sec. VII-D.

## VI. EXPERIMENT SETUP

## A. Models and Hardware Configurations

We evaluate nine models across three categories: one OCR specialist, five general-purpose instruct models, and three general-purpose thinking models.

HunyuanOCR [36] is a 1B end-to-end OCR specialist covering multiple languages and document types. It is the lowest-cost point in our comparison and represents the OCRspecialist category.

The Qwen3-VL family [37] contributes three instruct models at 4B, 8B, and 30B-A3B. The first two are dense; the third model is a mixture-of-experts model with 3B active parameters per forward pass.

Gemma4-E4B-it and Gemma4-12B-it [38] are instruct models having 8B total parameters with 4.5B effective and 12B parameters (all effective), respectively.

Qwen3.5-9B [39] and Qwen3.6-27B [40] are native multimodal thinking models that integrate vision and language through early fusion.

Step3-VL-10B [26] is a 10B thinking model pairing a Perception Encoder with a Qwen3-8B backbone. It introduces Parallel Coordinated Reasoning (PaCoRe) for test-time compute scaling.

All experiments were conducted on a single NVIDIA RTX Pro 6000 GPU (96 GB VRAM) running Debian 13 (trixie) with NVIDIA driver 610.57.04. To simulate a resourceconstrained production environment, we capped GPU memory at 30% of its total capacity ( 29 GB), using vLLM (v0.25.1) to enforce this limit. All models were evaluated in their official, unmodified form: Qwen3-VL and Qwen3.5 (Qwen), Gemma4 (Google), and Step3-VL (StepFun). No third-party quantization, fine-tuning, or other modifications were applied.

## B. Datasets

We evaluate DARKNIGHT on four CAPTCHA datasets collected from real dark web marketplaces, namely ChangAn $( \mathcal { D } _ { 1 } )$ , Freecity $( \mathcal { D } _ { 2 } )$ , Russian market $( \mathcal { D } _ { 3 } )$ , Mobilestore $( \mathcal { D } _ { 4 } )$ CAPTCHA images were collected via automated visits to each marketplace’s login page, rate-paced with randomized delays to stay below detection thresholds. Each visit yielded a fresh CAPTCHA, which was downloaded and stored. For each marketplace, we sampled 500 CAPTCHA instances, resulting in 2,000 images in total. Table V summarizes the resulting dataset structure and visual characteristics. Dataset availability and responsible-use safeguards are discussed in Appendix B.

![](images/65b53cd1edffb2d15efd9b3ba1645f5ad1bd2fc07dc6298e810085c3131fb7ed.jpg)

![](images/421e070d21643aecec3a0dd6c29dde506a46129255d7f03cb940153a82439be3.jpg)  
(b) D<sub>2</sub>

![](images/4735f1d98f154c32f9363dc5840a26a44d9f90690cc20d766cc92c1fc5bd58b7.jpg)

![](images/3a6ebdd87026b0d6d31ddb09afc0300b7b3a9ee5bdee1296a171cfb99d42000f.jpg)  
Alphanumeric, gradient background, wave distortion and dot noise.  
(d) D<sub>4</sub>  
Alphanumeric, patterned mesh background, 60×30 px.  
Fig. 5: Representative samples from each dataset.

$\mathcal { D } _ { 1 }$ is the most diverse dataset, containing CAPTCHAs spanning three types: alphanumeric, math equation, and Chinese characters. The images exhibit a range of visual defenses, e.g., colored interference lines, wave-distorted characters, overlapping strokes, and mixed background textures, making it the primary benchmark for evaluating the Planner’s ability to select appropriate tools for heterogeneous inputs.

TABLE V: Summary of dataset visual properties.
<table><tr><td>Dataset</td><td>Type</td><td>Sample size</td><td>Background</td><td>Noise</td><td>Distortion</td><td>Char Quality</td><td>Difficulty</td><td>Dark web site</td></tr><tr><td> $\mathcal { D } _ { 1 }$ </td><td>Alpha / Math / Chinese</td><td>500</td><td>Mixed</td><td>colored lines</td><td>Wave</td><td>Mixed</td><td>High</td><td>ChangAn</td></tr><tr><td> $\mathcal { D } _ { 2 }$ </td><td>Alphanumeric</td><td>500</td><td>Clean</td><td>Strikethrough line</td><td>None</td><td>Normal</td><td>Low</td><td>Freecity</td></tr><tr><td> $\mathcal { D } _ { 3 }$ </td><td>Alphanumeric</td><td>500</td><td>Gradient</td><td>Dots/specks</td><td>Wave</td><td>Normal</td><td>Medium</td><td>Russian market</td></tr><tr><td> $\mathcal { D } _ { 4 }$ </td><td>Alphanumeric</td><td>500</td><td>Patterned</td><td>None (mesh texture)</td><td>None</td><td>Thick</td><td>Medium</td><td>Mobilestore</td></tr></table>

$\mathcal { D } _ { 2 }$ contains alphanumeric CAPTCHAs with clean backgrounds and a dark line crossing the text. This dataset specifically tests whether DARKNIGHT correctly applies minimal or no preprocessing to easy images. A well-functioning Planner should recognize that preprocessing is unnecessary here and leave the image untouched, directly evaluating the system’s handling of the degeneration-recovery effect.

$\mathcal { D } _ { 3 }$ contains alphanumeric CAPTCHAs with moderate noise and distortion, serving as an intermediate-difficulty benchmark between $\mathcal { D } _ { 2 }$ and $\mathcal { D } _ { 1 }$

$\mathcal { D } _ { 4 }$ contains alphanumeric CAPTCHAs with thick characters on a patterned mesh background.

Dataset Annotation and Verification. Ground-truth labels were produced independently prior to the model experiments, ensuring that annotation was not influenced by model predictions. Three annotators participated in the annotation process and inter-annotator reliability was measured before any reconciliation using pairwise exact-string agreement (ESA) and character error rate (CER). Disagreements were subsequently reviewed and manually resolved through consensus to establish the final ground-truth labels. More details and dataset-level agreement statistics are reported in Appendix C.

## C. Baselines and Models

We evaluate nine open-weight models, with model size spanning from 1B to 30B parameters (Table VI). Each model is tested in two configurations:

Baseline (raw VLM). A CAPTCHA image is passed directly to the model with one generic prompt for all types. There is no preprocessing, no type-specific prompting, and no RAG retrieval involved. This set of evaluations demonstrates what each model achieves on its own.

DARKNIGHT (agentic pipeline). The DARKNIGHT system is applied on top of the same model, with visual analysis, RAG retrieval, selective preprocessing, type-specific prompting, and feedback recording.

Throughout this paper, backbone refers to the underlying base VLM on top of which the DARKNIGHT agentic pipeline is applied.

Eight of the nine evaluated models are general-purpose VLMs. We include HunyuanOCR as our sole OCR-specialized baseline to determine whether dedicated text-recognition finetuning is sufficient on its own. The selected models span 1B to 30B parameters across four distinct architectures, to understand if our findings generalize across different model families and training paradigms. A key distinction among these models is their inference mode (Table VI). Instruct-only models (marked Inst) execute structured commands without internal chain-of-thought deliberation, prioritizing speed over planning reliability. Conversely, thinking-enabled models (marked Think) perform extended internal reasoning, yielding more deliberate tool-selection decisions at the expense of higher latency. This dichotomy is critical: DARKNIGHT requires the backbone to both plan (reason about tool selection) and read (execute OCR). These are two distinct tasks that inherently benefit from different inference modes.

TABLE VI: Baseline models evaluation: Parameter size, type, domain, and development source.
<table><tr><td>Model Name</td><td>Size</td><td>Mode</td><td>Domain</td><td>Date</td><td>Source</td></tr><tr><td>HunyuanOCR</td><td>1B</td><td>Inst</td><td>OCR</td><td>Nov 2025</td><td>Tencent</td></tr><tr><td>Gemma4-E4B-it</td><td>8B</td><td>Inst</td><td>General</td><td>Apr 2026</td><td>Google</td></tr><tr><td>Qwen3-vl-4B</td><td>4B</td><td>Inst</td><td>General</td><td>Oct 2025</td><td>Alibaba</td></tr><tr><td>Qwen3-vl-8B</td><td>8B</td><td>Inst</td><td>General</td><td>Oct 2025</td><td>Alibaba</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>Think</td><td>General</td><td>Mar 2026</td><td>Alibaba</td></tr><tr><td>Step3-VL-10B</td><td>10B</td><td>Think</td><td>General</td><td>Jan 2026</td><td>Stepfun</td></tr><tr><td>Gemma4-12B-it</td><td>12B</td><td>Inst</td><td>General</td><td>Jul 2026</td><td>Google</td></tr><tr><td>Qwen3.6-27B</td><td>27B</td><td>Think</td><td>General</td><td>Apr 2026</td><td>Alibaba</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>30B</td><td>Inst</td><td>General</td><td>Oct 2025</td><td>Alibaba</td></tr></table>

## D. Evaluation Metrics

Exact-match accuracy. The primary metric: the fraction of CAPTCHAs for which the website returns “pass” upon submission. Matching is case-insensitive for alphabetic characters, following standard practice [20], which we also experimented on the markets we gathered our dataset from. For math CAPTCHAs, the numeric result must match exactly.

Inference time. Wall-clock time per CAPTCHA on local hardware, covering all pipeline stages (visual analysis, RAG query, preprocessing, OCR, and result handling). This measures local compute only; the Tor network round-trip is excluded.

Degeneration and recovery rates. For each preprocessing tool, we measure: (i) degeneration rate: fraction of images correct without the tool but incorrect with it (correct → wrong); (ii) recovery rate: fraction incorrect without the tool but correct with it (wrong → correct). Statistical significance is assessed via McNemar’s test $( p < 0 . 0 5 )$

## VII. RESULTS AND ANALYSES

## A. CAPTCHA-solving Accuracy

Fig. 6 presents the full empirical results across all four datasets; the corresponding exact numerical values are compiled in Table XIII (Appendix G).

Baseline models performance. Without any agentic assistance, open-weight VLMs exhibit highly variable performance across dark web CAPTCHA datasets. On $\mathcal { D } _ { 1 } .$ , the most diverse dataset, containing alphanumeric, math, and Chinese CAPTCHAs with heavy visual defenses, baseline accuracy is strikingly low, ranging from 6% with Gemma4-e4b-it (see Table XIII) to 64% (achieved by Step3-VL-10B). Most generalpurpose VLMs cluster around 22–46%, indicating that dark web CAPTCHAs with multiple defense features remain challenging for current models. Conversely, on the less defended alphanumeric CAPTCHAs of $\mathcal { D } _ { 4 }$ , several models exceed 80% baseline accuracy. Models also perform reliably on $\mathcal { D } _ { 2 } { ^ { \circ } } \mathrm { { s } }$ clean, minimal-noise backgrounds (Step3-VL-10B: 98.6%). $\mathcal { D } _ { 3 }$ represents an intermediate difficulty, with baselines falling between 27–67%.

![](images/82f60891133f1aa5174b9f3d7ec5e76da79969ff71e236ccd52daec400799326.jpg)  
Fig. 6: Baseline vs. DARKNIGHT: accuracy (%) across four benchmark datasets.

Model scale does not predict accuracy. On D , parameter count is a poor predictor of baseline accuracy. The dense Qwen3.6-27B (46.0%) improves on Qwen3-VL-8B (39.6%) by only 6.4 pp despite a 3.4× larger parameter count, and despite also being a newer model with built-in reasoning. The mixture-of-experts Qwen3-VL-30B-A3B (42.4%), which activates only 3B parameters per token, performs on par with its dense 8B sibling. Conversely, Step3-VL-10B reaches 63.6% on $\mathcal { D } _ { 1 }$ , outperforming every larger open-weight model we evaluate, while Gemma4-12B-IT (22.4%) falls below the 4B Qwen3-VL (28.0%). However, pure OCR systems lacking visual reasoning struggle on complex datasets. HunyuanOCR (cf. Table XIII) achieves 90.0% on the simple $\mathcal { D } _ { 4 }$ but collapses to 12.4% on $\mathcal { D } _ { 1 }$

Agentic Tool Selection vs. Model Scaling. Beyond baseline performance, in our evaluation, DARKNIGHT’s agentic framework yields higher accuracy gains than scaling model parameters. Within the dense Qwen3-VL family, doubling parameters from 4B to 8B raises $\mathcal { D } _ { 1 }$ accuracy by 11.6 pp (28.0%→39.6%), whereas adding DARKNIGHT to the 4B model raises it by 27.6 pp (to 55.6%), so the 4B model with

DARKNIGHT exceeds the base 8B by 16 pp. This improvement holds across model families: even Qwen3-VL-30B-A3B rises from 42.4% to 80.6% with DARKNIGHT (+38.2 pp). Gemma4- E4B-IT with DARKNIGHT (46.8%) also slightly exceeds the bare Qwen3-VL-30B-A3B (42.4%).

We additionally evaluate two conventional OCR systems and four frontier cloud VLMs as reference baselines; although the latter remains outside our target deployment setting as outlined in Sec. I. In the first group, PaddleOCR [41] substantially outperforms Tesseract [42], particularly on the less distorted datasets, but remains considerably weaker on the heavily defended $\mathcal { D } _ { 1 }$ dataset (below 5%). Recent frontier API models from Claude should provide stronger general-purpose visual reasoning, although their performance was below expectations; even with the Claude Opus 5, the accuracy is less than 50% (see Appendix D for more details).

## B. The Degeneration-Recovery Effect

A central design premise of DARKNIGHT is that preprocessing must be applied selectively rather than uniformly. To validate this, we trace every tool-selection decision made by the Planner across the evaluation suite and classify its ultimate effect on the prediction. By comparing the DARKNIGHT pipeline’s outcome against the raw VLM baseline, we map each attempt to one of the four transition states: correct→correct (C→C, preserving a correct prediction), correct→wrong (C→W, degeneration), wrong→correct (W→C, recovery), and wrong→wrong (W→W, no effect). Based on these transition states, we derive three core metrics to quantify the Planner’s efficacy: Degen% (the fraction of originally correct baseline predictions degraded by unnecessary tools), Recov% (the fraction of originally incorrect baseline predictions successfully recovered), and Net (the overall performance gain, calculated as Recov% − Degen%).

TABLE VII: Per-tool degeneration-recovery profile across all models and datasets. Tools ordered by net effect.
<table><tr><td>Tool</td><td>N</td><td>%</td><td>C→C</td><td>C→W</td><td>W→C</td><td>W→W</td><td>Degen%</td><td>Recov%</td><td>Net</td></tr><tr><td>(no tool)</td><td>4406</td><td>72.8</td><td>2357</td><td>159</td><td>733</td><td>1157</td><td>6.3</td><td>38.8</td><td>+32.5</td></tr><tr><td>color_extract</td><td>1216</td><td>20.1</td><td>371</td><td>58</td><td>508</td><td>279</td><td>13.5</td><td>64.5</td><td>+51.0</td></tr><tr><td>sharpen</td><td>197</td><td>3.3</td><td>66</td><td>3</td><td>90</td><td>38</td><td>4.3</td><td>70.3</td><td>+66.0</td></tr><tr><td>remove_straight_line</td><td>184</td><td>3.0</td><td>111</td><td>9</td><td>53</td><td>11</td><td>7.5</td><td>82.8</td><td>+75.3</td></tr><tr><td>denoise</td><td>43</td><td>0.7</td><td>17</td><td>12</td><td>6</td><td>8</td><td>41.4</td><td>42.9</td><td>+1.5</td></tr><tr><td>binarize</td><td>3</td><td>0.0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>1</td><td>0.0</td><td></td></tr><tr><td>Total</td><td>6049</td><td>100</td><td>2922</td><td>241</td><td>1390</td><td>1496</td><td>7.6</td><td>48.2</td><td>+40.5</td></tr></table>

Table VII summarises the effect of each preprocessing tool applied by DARKNIGHT across three operational models (Qwen3-VL-4B, Qwen3.5-9B, and Qwen3-VL-30B-A3B; see Sec. VII-F) and all datasets. The bottom row reports the pooled degeneration and recovery rates across all Planner decisions, including cases in which no preprocessing tool was selected.

Across all Planner decisions, DARKNIGHT recovers 48.2% of baseline failures at a cost of 7.6% degeneration, yielding 1,390 recoveries against only 241 degenerations (a 5.8× recovery-to-degeneration ratio).

The preprocessing tools exhibit different risk-reward profiles. remove\_straight\_line achieves the strongest net effect (+75.3), recovering 82.8% of baseline failures while degrading 7.5% of correct predictions. sharpen provides the next strongest profile (+66.0), with a low degeneration rate of 4.3% and a recovery rate of 70.3%. color\_extract, the most frequently selected preprocessing tool, achieves a +51.0 net effect, recovering 64.5% of failures at a 13.5% degeneration rate. In contrast, denoise provides only a marginal net benefit (+1.5), while binarize is selected only three times and therefore does not provide sufficient evidence for a meaningful risk-reward estimate.

TABLE VIII: Planner tool-selection behavior per dataset. Nopreproc rate is the fraction of decisions where the Planner selects an empty pipeline.
<table><tr><td></td><td> $\mathcal { D } _ { 1 }$ </td><td> $\mathcal { D } _ { 2 }$ </td><td> $\mathcal { D } _ { 3 }$ </td><td> $\mathcal { D } _ { 4 }$ </td></tr><tr><td>No-preproc rate</td><td>26.6%</td><td>86.9%</td><td>97.0%</td><td>56.7%</td></tr><tr><td>Avg baseline acc</td><td>35.7%</td><td>67.7%</td><td>50.5%</td><td>70.6%</td></tr><tr><td>Avg ∆ with DARKNIGHT</td><td>+34.2</td><td>+21.5</td><td>+6.5</td><td>+9.5</td></tr></table>

Table VIII demonstrates that the Planner’s intervention behavior varies substantially across datasets. On $\mathcal { D } _ { 3 }$ , the Planner selects no preprocessing in 97.0% of cases, while the nopreprocessing rate remains high at 86.9% on $\mathcal { D } _ { 2 }$ . Conversely, on $\mathcal { D } _ { 1 }$ , which contains substantially heavier visual obfuscation, the Planner intervenes in 73.4% of cases. This indicates that tool selection is driven by the visual characteristics of the CAPTCHA rather than by a uniform preprocessing policy.

Importantly, selecting no preprocessing is itself an explicit Planner decision. When intervention is unnecessary, withholding tools avoids potential degeneration. Overall, in our evaluation, effective planning required both selecting appropriate transformations and knowing when not to intervene.

## C. Effect of Retrieval Memory

To isolate the contribution of retrieval memory, we compare the raw VLM baseline, the DARKNIGHT pipeline without retrieval, and the full DARKNIGHT framework with retrieval. As shown in Fig. 7, enabling retrieval consistently improves accuracy over the no-retrieval pipeline across all evaluated configurations. This demonstrates that the gains of DARK-NIGHT are not solely attributable to preprocessing and prompting. Retrieving prior successful and failed experiences provides additional benefit by guiding tool selection and avoiding previously harmful actions. We further examine the sensitivity to RAG memory size in App. E, where a memory size of 100 achieves the best average performance across the evaluated settings and is adopted as the default configuration.

## D. Cross-Site Adaptation

To evaluate whether DARKNIGHT can operate across heterogeneous marketplaces using a shared experience memory, we construct a mixed-site stream from all four datasets. Specifically, we sample 125 CAPTCHAs from each of $\mathcal { D } _ { 1 }$ to $\mathcal { D } _ { 4 }$ , combine them into a merged set with 500 images, and randomly shuffle the samples. The resulting stream is processed sequentially using a single shared RAG store, allowing experience accumulated from one marketplace to be retrieved when solving visually similar CAPTCHAs originating from another. Table IX reports the results. DARKNIGHT improves accuracy for all four evaluated backbones in this mixedsite setting. Qwen3-VL-8B improves from 65.0% to 75.0%, while Qwen3.6-27B improves from 65.0% to 78.0%. We also observe substantial improvements for Gemma4-E4B and Qwen3.5-9B. These results indicate that DARKNIGHT can maintain a shared experience memory across heterogeneous CAPTCHA sources rather than requiring an independently engineered solver for each marketplace.

![](images/dbf72421efed3d6a99123807b35364b173de2efa39757ba374f2a2a8d5264501.jpg)  
Fig. 7: Ablation of retrieval memory using Qwen3-VL-4B and Qwen3-VL-8B across $\mathcal { D } _ { 1 } - \mathcal { D } _ { 4 }$

TABLE IX: Accuracy (%) in the cross-site mixed-stream experiment.
<table><tr><td>Model</td><td>Baseline</td><td>DARKNIGHT</td><td> $\overline { { \Delta } }$ </td></tr><tr><td>Qwen3-VL-8B</td><td>65.0</td><td>75.0</td><td>+10.0</td></tr><tr><td>Gemma4-E4B</td><td>50.8</td><td>60.8</td><td>+10.0</td></tr><tr><td>Qwen3.5-9B</td><td>16.5</td><td>80.4</td><td>+63.9</td></tr><tr><td>Qwen3.6-27B</td><td>65.0</td><td>78.0</td><td>+13.0</td></tr></table>

## E. Inference Time and Computational Overhead

Table X presents the end-to-end inference time across all datasets. As expected, integrating the DARKNIGHT agentic pipeline inherently increases latency compared to raw baselines. This delay represents the computational cost of the perceive-plan-execute loop, which involves dual VLM invocations and intermediate RAG retrieval. However, this overhead varies substantially across the evaluated VLMs.

The Accuracy-Latency Sweet Spot. Among the evaluated models, Qwen3-VL-30B-A3B-Instruct provides the strongest accuracy–latency trade-off. With DARKNIGHT, it achieves an average accuracy of 86.5% across the four datasets while requiring only 2.64 s per CAPTCHA on average. In comparison, Qwen3.6-27B reaches a similar average accuracy of 86.1% but requires 20.77 s, while Step3-VL-10B achieves 78.8% at

TABLE X: Mean end-to-end inference time (in seconds) per CAPTCHA across dark web datasets.
<table><tr><td>Model</td><td rowspan="2">Params</td><td colspan="2">D1</td><td colspan="2"> $\mathbf { \Delta } _ { \mathbf { B } } ^ { \mathcal { D } _ { 4 } } \mathbf { \Delta } _ { \mathbf { D } \mathbf { K } }$ </td><td colspan="2">D2</td><td colspan="2">D3</td></tr><tr><td></td><td>B</td><td>DK</td><td></td><td></td><td>B</td><td>DK</td><td>B</td><td>DK</td></tr><tr><td>HunyuanOCR</td><td>1B</td><td>1.82</td><td>一</td><td>0.10</td><td>一</td><td>0.07</td><td>一</td><td>0.13</td><td>一</td></tr><tr><td>Gemma4-E4B-IT</td><td>8B</td><td>0.156</td><td>3.81</td><td>0.154</td><td>3.25</td><td>0.137</td><td>3.12</td><td>0.157</td><td>2.97</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>4B</td><td>0.122</td><td>2.72</td><td>0.102</td><td>2.21</td><td>0.120</td><td>2.57</td><td>0.108</td><td>2.34</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>0.170</td><td>4.26</td><td>0.164</td><td>3.72</td><td>0.273</td><td>3.89</td><td>0.165</td><td>3.67</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>0.168</td><td>5.38</td><td>0.162</td><td>4.70</td><td>0.163</td><td>5.17</td><td>0.166</td><td>4.43</td></tr><tr><td>Step3-VL-10B</td><td>10B</td><td>4.500</td><td>68.19</td><td>1.026</td><td>44.50</td><td>2.203</td><td>48.20</td><td>0.971</td><td>50.11</td></tr><tr><td>Gemma4-12B-IT</td><td>12B</td><td>0.156</td><td>4.44</td><td>0.164</td><td>5.47</td><td>0.144</td><td>4.03</td><td>0.172</td><td>3.77</td></tr><tr><td>Qwen3.6-27B</td><td>27B</td><td>2.247</td><td>23.03</td><td>1.714</td><td>18.34</td><td>1.898</td><td>21.00</td><td>1.732</td><td>20.70</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>30B</td><td>0.170</td><td>2.92</td><td>0.155</td><td>2.11</td><td>0.148</td><td>3.16</td><td>0.162</td><td>2.38</td></tr></table>

52.75 s. These results indicate that efficient model architecture and agentic orchestration matter more than model size alone. The Latency Bottleneck of Heavy Reasoning. Conversely, some models incur higher local inference cost when integrated with DARKNIGHT. Step3-VL-10B is the most expensive, requiring 44.50–68.19 s per CAPTCHA across the four datasets, with an average of 52.75 s. Qwen3.6-27B also incurs considerably higher latency, averaging 20.77 s, despite achieving strong accuracy. In contrast, Qwen3-VL-30B-A3B-Instruct achieves comparable or higher accuracy with an average latency of only 2.64 s. These results show that the agentic overhead depends on model architecture and reasoning behavior rather than parameter count.

For the backbones we evaluated, effective CAPTCHA solving did not depend on parameter scaling alone. Instead, the best accuracy-latency trade-off came from combining an efficient VLM backbone with selective agentic orchestration.

## F. Cross-Model Accuracy-Latency Analysis

Table XI shows that the benefit introduced by DARKNIGHT varies substantially across backbones and is not determined simply by model size or inference mode. Qwen3-VL-4B obtains the largest incremental improvement (+24.9%), showing that a relatively weak raw backbone can benefit substantially from agentic assistance. In contrast, Qwen3-VL-30B-A3B-Instruct achieves the highest final average accuracy (86.5%), closely followed by Qwen3.6-27B (86.1%). These results also highlight the distinction between pipeline-induced gain and absolute performance, a large improvement does not necessarily imply the highest final accuracy.

TABLE XI: Cross-model accuracy and latency summary. Avg B and Avg DK denote the average raw-baseline and DARK-NIGHT accuracy across the four datasets; Avg ∆ denotes the incremental improvement introduced by DARKNIGHT, and t<sub>DK</sub> denotes the average local inference time per CAPTCHA.
<table><tr><td>Model</td><td>Params</td><td>Avg B</td><td>Avg DK</td><td>Avg ∆</td><td>tDK (s)</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>4B</td><td>29.9</td><td>54.8</td><td>+24.9</td><td>2.46</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>36.3</td><td>48.3</td><td>+12.0</td><td>3.89</td></tr><tr><td>Gemma4-E4B-IT</td><td>8B</td><td>50.8</td><td>59.9</td><td>+9.1</td><td>3.29</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>64.6</td><td>80.9</td><td>+16.3</td><td>4.92</td></tr><tr><td>Step3-VL-10B</td><td>10B</td><td>74.2</td><td>78.8</td><td>+4.6</td><td>52.75</td></tr><tr><td>Gemma4-12B-IT</td><td>12B</td><td>54.9</td><td>67.4</td><td>+12.6</td><td>4.43</td></tr><tr><td>Qwen3.6-27B</td><td>27B</td><td>71.4</td><td>86.1</td><td>+14.7</td><td>20.77</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>30B</td><td>73.9</td><td>86.5</td><td>+12.6</td><td>2.64</td></tr></table>

The latency results further show that parameter count alone is a poor predictor of computational cost. Qwen3-VL-30B-

A3B-Instruct achieves the highest average DARKNIGHT accuracy while requiring only 2.64 s per CAPTCHA, providing the strongest accuracy–latency trade-off among the evaluated backbones. By comparison, Qwen3.6-27B reaches a similar accuracy of 86.1% but requires 20.77 s, while Step3-VL-10B requires 52.75 s despite achieving a lower final accuracy of 78.8%. Overall, the results suggest that selecting an efficient backbone is more important than either parameter scaling or model category alone when deploying DARKNIGHT.

## VIII. CONCLUSION

In this work, we introduced DARKNIGHT, an autonomous, self-improving agentic framework that redefines the approach to defeating dark web CAPTCHAs. By shifting from per-site model retraining and large parameter scaling toward dynamic, experience-driven tool orchestration, DARKNIGHT enables lightweight, open-weight VLMs to reach high accuracy among deployable open-weight solvers. Equipping a small 9B VLM with DARKNIGHT yields a +36.8% accuracy gain on the most heavily defended dataset, with the local agentic loop completing in 5.38s per CAPTCHA. Furthermore, the dual-signal RAG architecture navigates the degeneration-recovery tradeoff by learning to avoid preprocessing steps that previously degraded similar CAPTCHAs. In our evaluation, the agent’s efficacy rests as much on restraint, knowing when to bypass preprocessing, as on active intervention.

Implications. Beyond our specific tool, we see three implications; the last two are speculative. First, for CTI teams: in our case study, one workstation GPU, an open-weight model, and five classical image tools solved most CAPTCHAs without labelled data or per-market training. Practitioners can start from our results on backbone choice (the mixture-ofexperts Qwen3-VL-30B-A3B gave the best accuracy–latency trade-off, Sec. VII-E), on failure memory (Sec. VII-B and VII-C), and on per-market warm-up (Sec. V-E). Second, server-rendered text CAPTCHAs, on the dark or clear web, may offer limited protection against low-cost local models; we expect operators to move to other challenge types or stricter rate limits, and collectors should plan for this. Third, learning from failures when not to act may benefit other agentic tooluse settings where a tool can degrade its input.

## ACKNOWLEDGMENT

This research/project is supported by the National Research Foundation, Singapore, and the Cyber Security Agency of Singapore under the National Cybersecurity R&D Programme and the CyberSG R&D Programme Office (Award CRPO-GC2-ASTAR-001). Any opinions, findings, conclusions, or recommendations expressed in these materials are those of the author(s) and do not reflect the views of the National Research Foundation, Singapore, the Cyber Security Agency of Singapore, or the CyberSG R&D Programme Office.

[1] Europol, The evolving threat landscape: how encryption, proxies and AI are expanding cybercrime – Internet Organised Crime Threat Assessment (IOCTA). Luxembourg: Publications Office of the European Union, 2026. [Online]. Available: https://www.europol. europa.eu/cms/sites/default/files/documents/IOCTA-2026.pdf

[2] D. Lerner, “State of the underground 2025: Key trends shaping cyber risk today,” May 2025. [Online]. Available: https://www.bitsight.com/ blog/state-of-the-underground-2025

[3] S. Samtani, Y. Chai, and H. Chen, “Linking exploits from the dark web to known vulnerabilities for proactive cyber threat intelligence: An attention-based deep structured semantic model,” MIS Quarterly, vol. 46, no. 2, pp. 911–946, 06 2022. [Online]. Available: https://doi.org/10.25300/MISQ/2022/15392

[4] H. Bijmans, T. Booij, A. Schwedersky, A. Nedgabat, and R. van Wegberg, “Catching Phishers By Their Bait: Investigating the Dutch Phishing Landscape through Phishing Kit Detection,” in USENIX Security, Aug. 2021, pp. 3757–3774. [Online]. Available: https: //www.usenix.org/conference/usenixsecurity21/presentation/bijmans

[5] D. Kim, H. Zhao, D. Kim, and S. Hwang, “Unveiling the Underground Phishing Ecosystem: A 12-Year Longitudinal Study of Deep and Dark Web Forums,” in ACM Web Conference, 2026, p. 3531–3542. [Online]. Available: https://doi.org/10.1145/3774904.3792737

[6] Z. Lin, J. Cui, X. Liao, and X. Wang, “Malla: Demystifying Real-world Large Language Model Integrated Malicious Services,” in USENIX Security, Aug. 2024, pp. 4693–4710. [Online]. Available: https: //www.usenix.org/conference/usenixsecurity24/presentation/lin-zilong

[7] T. E. Dearden and S. E. Tucker, “Follow the Money: Analyzing Darknet Activity Using Cryptocurrency and the Bitcoin Blockchain,” Journal of Contemporary Criminal Justice, vol. 39, no. 2, Mar. 2023.

[8] S. L. Schroer, N. Canevascini, I. Pekaric, P. Widmer, and P. Laskov,¨ “The dark side of the web: Towards understanding various data sources in cyber threat intelligence,” in IEEE European Symposium on Security and Privacy Workshops (EuroS&PW), 2025, pp. 79–89.

[9] Y. Wang, B. Arief, and J. Hernandez-Castro, “Secure in the dark? An in-depth analysis of dark web markets security,” International Journal of Information Security, vol. 24, no. 3, p. 107, 2025.

[10] N. Zhang, M. Ebrahimi, W. Li, and H. Chen, “Counteracting Dark Web Text-Based CAPTCHA with Generative Adversarial Learning for Proactive Cyber Threat Intelligence,” ACM Trans. Manage. Inf. Syst., vol. 13, no. 2, Mar. 2022. [Online]. Available: https://doi.org/10.1145/3505226

[11] M. Vibhute, N. Gutierrez, K. Radivojevic, and P. R. Brenner, “Multimodal Web Agents for Automated (Dark) Web Navigation,” in ICAART, 2025, pp. 437–444.

[12] D. De Pascale, G. Cascavilla, D. A. Tamburri, and W. J. Van Den Heuvel, “CRATOR a CRAwler for Tor: Turning Dark Web Pages into Open Source INTelligence,” in 9th European Symposium on Research in Computer Security. Springer-Verlag, 2024, p. 144–161. [Online]. Available: https://doi.org/10.1007/978-3-031-70890-9 8

[13] Y. Yannikos and J. Heeger, “Captchas on Darknet Marketplaces: Overview and Automated Solvers,” Electronic Imaging, vol. 36, no. 4, pp. 330–1–330–1, 2024.

[14] D. H. Audran, M. B. Andersen, M. H. Hansen, M. M. Andersen, T. B. Frederiksen, K. H. Hansen, D. Georgoulias, and E. Vasilomanolakis, “Tick Tock Break the Clock: Breaking CAPTCHAs on the Darkweb,” in SECRYPT, 2022, pp. 357–365.

[15] L. von Ahn, B. Maurer, C. McMillen, D. Abraham, and M. Blum, “reCAPTCHA: Human-Based Character Recognition via Web Security Measures,” Science, vol. 321, no. 5895, pp. 1465–1468, 2008.

[16] M. Guerar, L. Verderame, M. Migliardi, F. Palmieri, and A. Merlo, “Gotta CAPTCHA ’Em All: A Survey of 20 Years of the Human-orcomputer Dilemma,” ACM Comput. Surv., vol. 54, no. 9, Oct. 2021. [Online]. Available: https://doi.org/10.1145/3477142

[17] Z. Wu, Y. Xue, Y. Feng, X. Wang, and Y. Song, “MCA-Bench: A multimodal benchmark for evaluating CAPTCHA robustness against VLM-based attacks,” in AAAI Conference on Artificial Intelligence, vol. 40, no. 44, 2026, pp. 38 039–38 047. [Online]. Available: https://doi.org/10.1609/aaai.v40i44.41141

[18] J. Zhang, Z. Zhou, X. Ji, S. Liu, and Z. Zhao, “CAPTURE: A Benchmark and Evaluation for LVLMs in CAPTCHA Resolving,” arXiv preprint arXiv:2512.11323, 2025.

[19] A. Searles, Y. Nakatsuka, E. Ozturk, A. Paverd, G. Tsudik, and A. Enkoji, “An empirical study & evaluation of modern CAPTCHAs,” in USENIX Security, 2023.

[20] X. Teoh, Y. Lin, S. Li, R. Liu, A. Sollomoni, Y. Harel, and J. S. Dong, “Are CAPTCHAs still bot-hard? Generalized visual CAPTCHA solving with agentic vision language model,” in USENIX Security, 2025, pp. 3747–3766.

[21] G. Deng, H. Ou, Y. Liu, J. Zhang, T. Zhang, and Y. Liu, “Oedipus: LLM-enchanced Reasoning CAPTCHA Solver,” in ACM CCS, 2025, p. 6–20. [Online]. Available: https://doi.org/10.1145/3719027.3744872

[22] Anthropic, “Statement on the US government directive to suspend access to Fable 5 and Mythos 5,” https://www.anthropic.com/news/ fable-mythos-access, Jun 2026, accessed: 2026-10-14.

[23] Z. Zhang, W. Zhou, and M. Sherr, “Bypassing Tor Exit Blocking with Exit Bridge Onion Services,” in ACM CCS, 2020, p. 3–16. [Online]. Available: https://doi.org/10.1145/3372297.3417245

[24] CrowdStrike, “CrowdStrike 2025 APJ eCrime Landscape Report: Chinese Underground Marketplaces Drive Billions in Illicit Transactions; AI-accelerated Ransomware Surges,” 2025. [Online]. Available: https://ir.crowdstrike.com/news-releases/news-release-details/ crowdstrike-2025-apj-ecrime-landscape-report-chinese-underground

[25] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in ICLR, 2023.

[26] StepFun AI, “Step3-VL-10B: Compact Yet Frontier Multimodal Intelligence,” arXiv preprint arXiv:2601.09668, 2026. [Online]. Available: https://arxiv.org/abs/2601.09668

[27] D. Georgoulias, J. M. Pedersen, M. Falch, and E. Vasilomanolakis, “A qualitative mapping of darkweb marketplaces,” in APWG eCrime, 2021, pp. 1–15.

[28] E. Bursztein, S. Bethard, C. Fabry, J. C. Mitchell, and D. Jurafsky, “How good are humans at solving CAPTCHAs? A large scale evaluation,” in IEEE Security and Privacy, 2010, pp. 399–413.

[29] H. Gao, J. Yan, F. Cao, Z. Zhang, L. Lei, M. Tang, P. Zhang, X. Zhou, X. Wang, and J. Li, “A simple generic attack on text captchas.” in NDSS, 2016.

[30] Y. Gao, H. Gao, S. luo, Y. Zi, S. Zhang, W. Mao, P. Wang, Y. Shen, and J. Yan, “Research on the security of visual reasoning CAPTCHA,” in USENIX Security, Aug. 2021, pp. 3291–3308. [Online]. Available: https://www.usenix.org/conference/usenixsecurity21/presentation/gao

[31] J. Wang, C. Zhu, Y. Zhou, L. Li, X. He, M. Wei, and J. Xiong, “COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers,” arXiv preprint arXiv:2512.02318, 2025.

[32] C. Zauner, “Implementation and Benchmarking of Perceptual Image Hash Functions,” Master’s thesis, Upper Austria University of Applied Sciences, Hagenberg Campus, Hagenberg, Austria, 2010. [Online]. Available: https://www.phash.org/docs/pubs/thesis zauner.pdf

[33] W. F. King, “Analysis of demand paging algorithms,” Information Processing, vol. 71, pp. 485–490, 1971.

[34] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 5776–5788.

[35] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” in EMNLP-IJCNLP. Hong Kong, China: Association for Computational Linguistics, 2019, pp. 3982–3992.

[36] Hunyuan Vision Team et al., “HunyuanOCR Technical Report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.19575

[37] Qwen Team, “Qwen3-VL Technical Report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[38] Google DeepMind, “Gemma 4,” Hugging Face, 2026, accessed: 2026- 06-12. [Online]. Available: https://huggingface.co/google/gemma-4-E4B

[39] Qwen Team, “Qwen3.5: Towards Native Multimodal Agents,” Qwen Blog, Feb 2026, accessed: 2026-08-12. [Online]. Available: https: //qwen.ai/blog?id=qwen3.5

[40] ——, “Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model,” Qwen Blog, Apr 2026, accessed: 2026-06-12. [Online]. Available: https://qwen.ai/blog?id=qwen3.6-27b

[41] Y. Du, C. Li, R. Guo, X. Yin, W. Liu, J. Zhou, Y. Bai, Z. Yu, Y. Yang, Q. Dang, and H. Wang, “PP-OCR: A practical ultra lightweight OCR system,” 2020, paddleOCR, PP-OCRv4 recognition model: https://github.com/PaddlePaddle/PaddleOCR. [Online]. Available: https: //arxiv.org/abs/2009.09941

[42] R. Smith, “An overview of the Tesseract OCR engine,” in Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), vol. 2. IEEE, 2007, pp. 629–633.

## APPENDIX

## A. Ethical Considerations

This work studies CAPTCHA robustness in the context of dark web threat intelligence collection. While CAPTCHAsolving techniques can be misused to automate access to protected services (e.g., on the clear web), our objective is dark web analytics for the larger social good. We therefore release the research artifacts used in this study, including the annotated datasets, evaluation code, preprocessing tools, agentic pipeline, experiment configurations, and reproduction scripts. To reduce unnecessary exposure of operational infrastructure, the released datasets contain no onion addresses, credentials, or other marketplace access information.

## B. Data Availability and Responsible Release

To support reproducibility and future research, we publicly release the complete source code, experiment configurations, reproduction scripts, and all four annotated CAPTCHA datasets at: https://github.com/Binghui99/DarKnight. The released data contain no onion addresses or other marketplace access information. We also withhold operational solver assets and deployment tooling that could directly facilitate automated CAPTCHA bypass at scale.

## C. Inter-Annotator Agreement

Table XII reports inter-annotator reliability for CAPTCHA transcription. Three annotators participated in the annotation process, two of whom are Chinese-speaking, resulting in two annotators labelled 2,000 CAPTCHAs, while the third annotator skipped the 100+ Chinese CAPTCHAs from $\mathcal { D } _ { 1 }$ All agreement statistics were computed on independently produced transcriptions before any disagreement resolution. We report exact-string agreement (ESA), which measures the proportion of independently produced transcription pairs that match exactly, and character error rate (CER), which quantifies character-level disagreement using edit distance. For samples annotated by three annotators, the reported statistics aggregate agreement across all annotator pairs; for Chinese-script samples, agreement is evaluated between the two Chinesespeaking annotators. After agreement measurement, disagreements were reviewed and resolved by consensus to establish the final ground-truth labels.

TABLE XII: Inter-annotator agreement for CAPTCHA transcription. Agreement statistics were calculated before consensus resolution.
<table><tr><td>Dataset</td><td>Samples</td><td>ESA (%)</td><td>CER (%)</td><td>Resolution</td></tr><tr><td> $\overline { { \mathcal { D } _ { 1 } } }$ </td><td>500</td><td>90.4</td><td>2.90</td><td>Consensus</td></tr><tr><td> $\mathcal { D } _ { 2 }$ </td><td>500</td><td>99.4</td><td>0.16</td><td>Consensus</td></tr><tr><td> $\mathcal { D } _ { 3 }$ </td><td>500</td><td>96.4</td><td>0.70</td><td>Consensus</td></tr><tr><td> $\mathcal { D } _ { 4 }$ </td><td>500</td><td>94.4</td><td>1.29</td><td>Consensus</td></tr></table>

The results show high annotation agreement across all four datasets. D2 exhibits the highest reliability (99.4% ESA and 0.16% CER), consistent with its relatively clean visual structure, whereas $\mathcal { D } _ { 1 }$ shows the largest disagreement (90.4% ESA and 2.90% CER), reflecting its greater visual and script complexity. Importantly, even for $\mathcal { D } _ { 1 }$ , character-level disagreement remains low, indicating that most transcription differences involve only a small number of characters.

## D. Additional Baseline Comparison

To provide additional reference points against established OCR systems and frontier VLMs, we evaluate two locally hosted OCR baselines (PaddleOCR [41] and Tesseract [42]) and four frontier Claude models on the ChangAn $( \mathcal { D } _ { 1 } )$ dataset. The Claude models are evaluated only as reference baselines and are not considered deployment candidates under the operational constraints discussed in Sec. I.

![](images/18105216148228e98cdeb85cdbd2b3718b0755038c0ae20a3d8201973ac9cae8.jpg)

(a) Classical OCR and VLM baselines across $\mathcal { D } _ { 1 } - \mathcal { D } _ { 4 }$  
![](images/3fe4fc4c646ea54eed074b39e1bf326fe5138422c4f816667e03a2fb37f1df99.jpg)  
(b) Frontier Claude models on $\mathcal { D } _ { 1 }$ , with and without DARKNIGHT. Fig. 8: Comparison with other baselines.

## E. Sensitivity to RAG Memory Size

We further evaluate the sensitivity of DARKNIGHT to the per-slot RAG memory capacity using Qwen3-VL-4B on $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ . Here, a RAG size of N denotes a capacity of $N$ successful and N failed records, for a maximum of 2N records in total. As shown in Table XIV, moderate memory sizes generally provide strong performance, while increasing the capacity does not consistently yield further improvements. Among the evaluated settings, a per-slot capacity of 100 (up to 200 records in total) achieves the highest average accuracy across the two datasets and is therefore adopted as the default configuration.

TABLE XIII: Accuracy (%) on dark web CAPTCHA datasets. B = baseline (raw VLM), DK = DARKNIGHT (agentic pipeline). ∆ = absolute change. Best agentic accuracy per dataset in bold. Models are ordered by size. HunyuanOCR is a pure OCR system that cannot be used as a Planner and therefore has no agentic results.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td rowspan="2">B</td><td rowspan="2"> $\mathcal { D } _ { 1 }$  DK</td><td rowspan="2"> $\Delta$ </td><td rowspan="2">B</td><td colspan="2"> $\mathcal { D } _ { 2 }$ </td><td rowspan="2"> $\Delta$ </td><td rowspan="2">B</td><td rowspan="2"> $\mathcal { D } _ { 3 }$  DK</td><td rowspan="2"> $\Delta$ </td><td rowspan="2">B</td><td rowspan="2"> $\mathcal { D } _ { 4 }$  DK</td><td rowspan="2"></td></tr><tr><td> $\mathbf { D K }$ </td></tr><tr><td>HunyuanOCR</td><td>1B</td><td>12.4</td><td></td><td></td><td>61.4</td><td></td><td></td><td>45.7</td><td></td><td></td><td>90.0</td><td></td><td></td></tr><tr><td>Gemma4-E4B-IT</td><td>8B</td><td>5.8</td><td>46.8</td><td>+41.0</td><td>71.4</td><td>65.8</td><td>-5.6</td><td>51.5</td><td>56.5</td><td>+5.0</td><td>74.4</td><td>70.4</td><td>-4.0</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>4B</td><td>28.0</td><td>55.6</td><td>+27.6</td><td>30.8</td><td>73.4</td><td>+42.6</td><td>26.8</td><td>35.3</td><td>+8.5</td><td>33.8</td><td>54.8</td><td>+21.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>39.6</td><td>66.6</td><td>+27.0</td><td>43.4</td><td>49.0</td><td>+5.6</td><td>33.4</td><td>38.3</td><td>+4.9</td><td>28.8</td><td>39.4</td><td>+10.6</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>36.6</td><td>73.4</td><td>+36.8</td><td>78.2</td><td>94.2</td><td>+16.0</td><td>58.1</td><td>63.4</td><td>+5.3</td><td>85.4</td><td>92.4</td><td>+7.0</td></tr><tr><td>Step3-VL-10B</td><td>10B</td><td>63.6</td><td>76.0</td><td>+12.4</td><td>98.6</td><td>100.0</td><td>+1.4</td><td>53.6</td><td>57.0</td><td>+3.4</td><td>80.8</td><td>82.0</td><td>+1.2</td></tr><tr><td>Gemma4-12B-IT</td><td>12B</td><td>22.4</td><td>46.0</td><td>+23.6</td><td>65.0</td><td>81.0</td><td>+16.0</td><td>43.0</td><td>47.7</td><td>+4.7</td><td>89.0</td><td>95.0</td><td>+6.0</td></tr><tr><td>Qwen3.6-27B</td><td>27B</td><td>46.0</td><td>83.2</td><td>+37.2</td><td>85.2</td><td>100.0</td><td>+14.8</td><td>63.6</td><td>68.0</td><td>+4.4</td><td>90.6</td><td>93.0</td><td>+2.4</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>30B</td><td>42.4</td><td>80.6</td><td>+38.2</td><td>94.0</td><td>100.0</td><td>+6.0</td><td>66.7</td><td>72.3</td><td>+5.6</td><td>92.6</td><td>93.0</td><td>+0.4</td></tr></table>

TABLE XIV: Accuracy (%) for varying RAG memory size.
<table><tr><td>RAG Size</td><td>D1</td><td> $\overline { { \mathcal { D } _ { 2 } } }$ </td></tr><tr><td>No RAG</td><td>52.8</td><td>51.0</td></tr><tr><td>10</td><td>57.2</td><td>53.0</td></tr><tr><td>20</td><td>56.8</td><td>73.2</td></tr><tr><td>50</td><td>56.0</td><td>73.2</td></tr><tr><td>80</td><td>57.2</td><td>64.8</td></tr><tr><td>100</td><td>57.8</td><td>73.4</td></tr><tr><td>150</td><td>56.8</td><td>73.2</td></tr><tr><td>200</td><td>59.4</td><td>59.0</td></tr></table>

The results in Fig. 8 show that conventional OCR systems perform poorly on ChangAn $( \mathcal { D } _ { 1 } )$ , with PaddleOCR achieving 3.4% and Tesseract 0%. The frontier Claude models provide substantially stronger raw performance, reaching up to 44.0%. DARKNIGHT further improves all four Claude models, with final accuracy reaching up to 81.0%.

## F. API parameters for VLM calls

Tab. XV specifies the exact hyperparameter configurations and inference payloads used for the two distinct VLM invocations (planning and execution) within the framework.

TABLE XV: API parameters for the two VLM calls per sample.
<table><tr><td>Parameter</td><td>visual_analysis &amp; plan</td><td>run_ocr</td></tr><tr><td>max_tokens</td><td>2048</td><td>1024</td></tr><tr><td>temperature</td><td>0.0</td><td>0.1</td></tr><tr><td>Input</td><td>image+text+RAG</td><td>processed image+text</td></tr></table>

## G. Detailed results

This section provides the complete granular dataset for the evaluation discussed in Sec. VII. Table XIII details the exact accuracy percentages and absolute changes (∆) for all evaluated VLMs across the four dark web datasets, serving as the numerical foundation for Fig. 6.

## H. Discussion

Local CAPTCHA Solving for CTI. Historically, defeating specially designed dark web CAPTCHAs has required either labor-intensive per-site engineering or increasingly large vision models. DARKNIGHT demonstrates that strong performance can instead be achieved by combining open-weight VLMs with adaptive, experience-driven tool orchestration. In particular, our results show that model size alone does not determine either accuracy or inference cost; an efficient backbone paired with DARKNIGHT can provide a favorable accuracy–latency trade-off without relying on third-party VLM APIs. This makes local CAPTCHA solving more accessible for automated CTI collection.

The Necessity of Negative Memory in Visual Agents. A core insight from our evaluation is that agentic visual systems fail not just from an inability to act, but from an inability to restrain themselves. When applied uniformly, aggressive preprocessing consistently degrades clean inputs. DARKNIGHT overcomes this via its dual-signal RAG architecture. By treating historical failures as explicit negative constraints rather than discarding them, the Planner learns a conservative execution policy. This suggests that multimodal agents deployed in volatile security contexts may benefit from incorporating explicit failurememory mechanisms to prevent self-induced degradation.

Limitations and Adversarial Adaptation. While our evaluation uses real-world dark web CAPTCHA datasets, a live deployment faces dynamic adversarial conditions that are not fully captured by an offline study. First, marketplace operators may introduce targeted perturbations or entirely new CAPTCHA mechanisms designed to disrupt VLM-based solvers. Although DARKNIGHT can adapt its preprocessing strategy through accumulated experience, its formal robustness against targeted adversarial examples remains an open question. Second, our latency measurements capture local computational overhead only. In a live deployment, end-toend latency would additionally depend on variable Tor network delays and crawler request pacing, which we did not measure. Future work will integrate DARKNIGHT into a live adaptive crawler to evaluate its long-term behavior under operator countermeasures and network variability.