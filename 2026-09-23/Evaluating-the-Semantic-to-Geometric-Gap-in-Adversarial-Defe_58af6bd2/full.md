# Evaluating the Semantic-to-Geometric Gap in Adversarial Defenses Against Vision-Language Model-Based Plagiarism

Christopher Burger Pelican Quantitative chrisb@pelicanqc.com

Christina Trotter The University of Mississippi cjtrotte@olemiss.edu

Joseph Carlisle The University of Mississippi jcarlis1@olemiss.edu

Charles Walter The University of Mississippi cwwalter@olemiss.edu

## Abstract

The rapidly advancing capabilities of vision-language models (VLMs) present a systemic challenge to academic integrity. VLMs now allow students to bypass meaningful engagement by capturing and submitting graphical problems as singular images, a practice we define as “trivial plagiarism.” To provide educators with actionable data on VLM limitations, we investigate the efficacy of heuristic adversarial image transformations designed to degrade model performance while remaining human-interpretable. Through a two-phase evaluation of introductory assessments, we manually assess baseline VLM performance on circuit diagrams, followed by an automated large-scale evaluation of topological structures (logic gates) and coordinate geometry (Karnaugh maps). We find that while highly capable VLMs can exhibit appreciable robustness, all models suffer vulnerability to adversarial perturbations. We conclude that while visual perturbations act as a viable near-term stopgap, long-term assessment security requires educators to reapproach assessment design given continually increasing VLM performance.

Keywords: Vision-Language Models, Academic Integrity, Adversarial Examples, Assessment Design

## 1. Introduction

The integration of vision-language models (VLMs) into academic settings has created a tension between their potential as powerful learning aids and their use as tools for academic dishonesty. While text-based plagiarism via large-language models (LLMs) is a recognized issue (Cotton et al., 2024; Teel et al., 2023; Toba et al., 2023; Wahle et al., 2022), the recent introduction of multimodal models capable of interpreting images represents a substantially lower barrier for plagiarism (Miles et al., 2022; Peters & Angelov, 2025). This capability allows students to circumvent the learning process entirely by submitting a raw image of a problem, a method we define as trivial plagiarism. Trivial plagiarism requires minimal effort by the student, who no longer needs to understand (at least some aspect of) the problem to obtain a solution through appropriately crafted prompts. Rather, the student needs only take a picture of the problem (or wear smart glasses with cameras recording) to obtain a solution, not thinking or understanding any aspect of the problem or solution.

Traditional plagiarism required some understanding of the content being assessed to remain undetected (Peters & Angelov, 2025). Previously, obtaining correct results from an LLM often required careful prompt engineering, a task that necessitated a degree of patience and some level of familiarity with the subject matter. This interaction, while potentially aimed at academic misconduct, created a soft barrier that could still allow for some incidental learning. However, the effectiveness of current VLM models on direct image inputs has largely removed this barrier. A student can now upload a photograph or screenshot of a graphical problem, from a circuit diagram to a chemical equation, and receive a correct solution with no real intellectual engagement. This shift separates traditional plagiarism from trivial plagiarism, as there is no understanding from the student but detection based only on the answer is difficult or impossible.

Our work explores a defensive strategy against this form of plagiarism by leveraging adversarial examples. While prior research on adversarial defenses in the context of academic integrity has focused predominantly on text inputs (Salim et al., 2024), image-based inputs remain a relatively unexplored domain. We argue that image perturbations represent a more robust and efficient defensive mechanism. Unlike perturbations in natural language, which must carefully preserve syntactic and semantic meaning to remain coherent (Burger et al., 2023), image transformations can tolerate a greater degree of alteration before a human observer’s understanding of the content is compromised (Ahn et al., 2003; Geirhos et al., 2018; Marr, 2010).

The goal of this paper is to evaluate the plausibility of computationally efficient image transformations as a deterrent to trivial plagiarism, and to investigate the limitations of VLMs when processing visual assessments. We present a two-phase evaluation. Phase 1 manually evaluates a dataset of introductory circuit analysis problems containing both text and diagrams to establish the baseline effectiveness of heuristic visual obfuscation (e.g., pixelated noise and affine warping). Phase 2 introduces a fully automated pipeline evaluating purely visual topological structures (logic diagrams) and geometric structures (Karnaugh maps). This automated approach allows for extended and repeated testing to confirm model efficacy given that there exists unavoidable randomness when using modern LLMs. Ultimately, we argue that while visual perturbations act as a viable near-term stop-gap by degrading model accuracy, the long-term solution for assessment security requires exploiting these underlying spatial and geometric blind spots rather than engaging in a futile arms race of image degradation.

## 2. Background & Related Work

The integration of LLMs into educational environments has invoked a significant debate regarding academic integrity and the validity of traditional assessments (Cotton et al., 2024; Jost et al.,ˇ 2024; Nikolic et al., 2024). While LLMs undoubtedly offer benefits, especially through asynchronicity and personalization of learning, the question of how to prevent the circumvention of assessments becomes more prominent as LLMs transition beyond text into multimodal VLMs, which are capable of accepting a growing variety of inputs. Attempting to algorithmically detect LLM-generated output has proven systematically unreliable; recent studies demonstrate that AI detectors yield appreciable false positive rates and can fail to catch sophisticated AI use (Burger et al., 2026; Ofqual, 2024; Perkins et al., 2023; Weber-Wulff et al., 2023). This detection collapse necessitates a pivot toward proactive, input-level defenses.

In an attempt to maintain academic integrity, defense mechanisms involving adversarial perturbations (a known weakness of many classic deep learning models) have been developed. The closest directly related to our work being Salim et al. (2024), which focuses on text-based perturbation to corrupt code generation for programming assignments. However, we seek to deal with VLMs and image-based input. Much of the adversarial machine learning literature focused on images has predominantly utilized gradient-based attacks, such as the Fast Gradient Sign Method (Goodfellow et al., 2014). However, these attacks require full “white-box” access, meaning the attacker needs complete visibility and control over the model’s proprietary architecture, internal weights, and gradients. Because students rely on closed-source, black-box platforms (e.g., ChatGPT), executing white-box attacks is impossible for educators, as those resources are exclusively held by the parent tech corporations. Even if such access were available, mathematically complex gradient attacks remain far too computationally expensive to be integrated into high-volume educational content management systems.

Vision transformers, which power modern state-of-the-art VLMs, have also demonstrated inherent robustness against traditional black-box adversarial attacks (Mahmood et al., 2021), necessitating an even greater computational budget for classic adversarial methods. Consequently, instead of relying on computationally prohibitive deep-learning exploits, we investigate heuristic, visually perceptible perturbations based on natural forms of image degradation.

Our defense strategy is informed by recent findings regarding the “cognitive” and spatial limitations of VLMs. While vision-language models have what amounts to a holistic comprehension of visual environments, recent benchmarks reveal deficiencies in genuine spatial reasoning; models suffer from a fundamental “semantic-to-geometric gap” where qualitative object recognition fails to translate into high-fidelity geometric computation. Specifically, VLMs perform adequately on static spatial tasks or basic topological connections, but their performance drops appreciably on tasks requiring dynamic transformations, mental rotation, or strict coordinate geometry (Deng et al., 2026; Stogiannidis et al., 2025). By targeting this spatial weakness with heuristic visual obfuscation, we aim to bridge the gap between adversarial machine learning and practical assessment design. We acknowledge that these perturbations are fundamentally crude; they represent a necessary trade-off: optimizing for rapid, automated pipeline integration rather than effectively human undetectable perturbations. Ultimately, while this visual disruption serves as an effective near-term stop-gap, the long-term assurance of academic integrity will likely require educators to rethink assessment redesign rather than simply obscuring the input.

## 3. Methodology

To be practically useful in an educational setting, an adversarial defense mechanism must satisfy two criteria. First, it must be computationally efficient. We avoid strictly defining efficient, but seek methods with generation times that would not feel burdensome if integrated in a content management system. For example, a few minutes to generate quality perturbations with an acceptable likelihood of defeating VLM models may be considered reasonable, whereas hours of computation is not. Second, it must maintain human interpretability; the perturbations must degrade the model’s performance without altering the fundamental problem or burdening the student with excessive artifacts. We do not seek to find mathematically optimal, white-box adversarial noise, but rather to determine if fast, heuristic methods can induce erroneous VLM outputs while keeping the perturbed document legible.

## 3.1. Perturbation Methods

The methods used in the perturbation process were chosen to mimic common causes of image degradation. The goal here is not to find an “optimal” perturbation from an iterative process, but instead to determine if fast and efficient methods can create what looks to be “natural” (or, at least, human interpretable) imperfections in an image representation of a problem that induce erroneous output from a VLM.

We utilize three main perturbations that emulate common negative conditions when reproducing an image. These perturbations are: Warping, Pixel Noise, and Lines with random start and end points. When we use the term “random”, we mean the values are generated arbitrarily and uniformly. Additionally, we introduce a multi-perturbation method to simulate a Photocopy as a perturbation that may result from normal use of material.

Warping: Used to simulate physical paper distortion (like crinkling, folding, etc). We apply a piecewise affine transformation by first applying a grid of control points over the image. Then, the locations of these points are shifted randomly, with a defined maximum displacement. A piecewise affine transform function then calculates the distortion needed to move the grid points from the original to the new location. Ultimately, this results in stretching or compressing different parts of the image locally.

Pixel Noise: Used to simulate imaging sensor noise, dust / debris, or print imperfections. Some proportion of pixels are randomly selected across the image for perturbation. Each selected pixel is then filled with a randomly chosen shade of grey.

Lines: Used to simulate scratches on lenses or scanner glass, fold lines or creases in paper, streaks from faulty copier drums or sensor elements, etc. We generate a small, random number of straight lines (3 to 7), with random start and end points, random shade of gray, and a random small width (1 or 2 pixels). The line perturbations are applied only to the problem text and never the figure as the addition of a line can fundamentally alter the structure of the circuit.

Photocopy: Used to emulate the effect of degradation from the digitization/image conversion process when a document is repeatedly copied, like say a photocopy of a photocopy, or a picture of a photocopy. This method has seven stages applied in sequence and then the result is fed back into the method for some number of iterations to simulate the progressive degradation.

1. Greyscale conversion: Converts the image back to greyscale to disable any unexpected color additions from prior perturbations.

2. Contrast adjustment: Randomly increases the contrast by a small factor. Useful for simulating loss of detail from digitization, e.g., ‘crushed blacks’, or ‘blown-out whites’. Default parameter set to the range of 1.1 to 1.4.

3. Brightness adjustment: Random increases or decreases the brightness by a small factor. Helps simulate variations in exposure settings or lamp age for a scanner / camera. Default parameter set to the range of 0.95 to 1.05.

4. Blurring: A Gaussian blur filter is applied with a small, random radius. Used to provide a small, cumulative loss of sharpness and fine detail. Default parameter set to range 0.3 to 0.8.

5. Skew: Applied a slight rotation (up to 1 degree). Used to mimic misalignment. Default parameter set to active.

6. Pixel noise: Functionally equivalent to the pixel noise perturbation described above. Default parameter set to active.

7. Compression artifacts: Performs a save to a temporary JPEG file with a random quality level (50 to 90), the file is then reloaded and used for future iterations or permanently saved. Used to simulate the degradation of repeated digitization in lossy formats. Default parameter range set to 50% to 90%.

![](images/219baf37c538c8c25d914e79760ad2e48932a0be33983cf56b828fc7424bf25d.jpg)  
(a) Original, unperturbed image.

![](images/7d437b631cfd801daa68e162107469efdfc7a4d7944cfcdad72ba9775cdb07ee.jpg)  
(b) Perturbed but legible.

![](images/aa1958a8003de0434fba5483eb0ddbefde28900e06bb78f74bab29404c89066f.jpg)  
(c) Perturbed and illegible.  
Figure 1. Legibility comparison between perturbed problem images and the original.

Finally, we include an option to control where the above perturbations are applied. By default the entire image is perturbed, but we allow for the perturbation of only the question text or the associated figure.

While the effectiveness of the perturbations is clearly important, this work is concentrated on exploring if naive selections of the above perturbations demonstrate enough efficacy to justify the time and computational expense to optimize this process. We note that with the above perturbation methods all of our practicality conditions are met. First, the code needed to generate these perturbations is simple with well specified parameters that will allow future tuning of the perturbations. And second, the process is fast in generating and applying these perturbations. It takes only fractions of a second per image on a modest desktop computer for all options enabled at any reasonable value (that is, values that do not degrade the image to the point of being unrecognizable).

## 3.2. Phase 1: Manual Evaluation (Circuit Diagrams)

Our evaluation is divided into two distinct phases. Phase 1 serves as an initial heuristic baseline, evaluating problems that contain both necessary textual components and visual diagrams. We selected introductory DC circuit analysis as the domain, curating a dataset of instructor resources with provided (and verified) solutions where a short text prompt (10-20 words) and a simple schematic (restricted to resistor networks) are both required to solve for a specific value (e.g., current or voltage). Our reasoning behind the choice of DC circuit problems is to choose subject matter that involves both text and graphics simultaneously while being challenging enough (to VLMs) to increase the likelihood of efficacy of relatively minor perturbations (and so hopefully less disruptive to viewers).

This phase utilizes a manual evaluation process via the models’ standard web interfaces to simulate the exact workflow of a cheating student. We apply the perturbations (Section 3) selectively: to the text only, to the figure only, or to the entire combined image, allowing us to isolate which region the model relies on most heavily. Because this process relies on both manual web-interface input and processing of the results, the sample size is constrained, but it provides a necessary qualitative baseline for real-world VLM interaction.

The evaluation process was performed as follows:

1. All prompts used the identical structure:

Can you solve the problem in the attached image?

2. Errors in processing (e.g. network timeouts) resulted in the creation of a new conversation as to not bias future output.

3. Complete output of all responses was recorded and compared to known correct solutions from guaranteed human generated content.

## 3.3. Phase 2: Automated Evaluation (Topological and Geometric Rigidity)

Phase 2 transitions to a fully automated pipeline designed to evaluate the limits of VLM spatial reasoning at scale. Unlike Phase 1, the problem text is provided directly in the system prompt (which remains identical for each distinct problem type), and so the adversarial perturbations are applied only to the image. This phase reduces the stochasticity of web interfaces by utilizing direct API access, fixing the generation temperature to 0.0, and evaluating each image three separate times to generate a robust “Pass@3” majority-vote metric.

To ensure consistency and correct evaluation, the automated prompting and evaluation pipeline is as follows:

![](images/a383b1e96b906fb32a35b56cbf8dea591fe1ea4a721609609d65b5b3af3ce7fd.jpg)  
(a) Original

![](images/6c901aadab12ff38446885d6abb89c73b932315b123a89a67f74d8573c65d2d4.jpg)

![](images/902fd840982ae583e4c1472c2c8bff5ffb782347f2b0a4e4c99b0e3947d6fcc1.jpg)

![](images/a482cbd8f604513f867e9addf23bc136f1fb3af03ee9783237735ae5f71c9fbc.jpg)  
(d) All perturbations on full image

![](images/c477bbb3c4fa1e84bad0e1a0cfa7dac03c08acc2c2ecd340f36562f1662b6e1f.jpg)  
(e) All perturbations on text prompt only

![](images/a350a5ba5fb8ce5bdeae5b7bbaff8d0352f814b1042fec1b9dd0f2f5713fb2d8.jpg)  
(f) All perturbations on figure only  
Figure 2. Perturbations and their valid regions on the image. Note that the intensity of the perturbations here is deliberately high to emphasize their effect. Subfigures (b) and (d) are pushing the boundary of what we would consider to be reasonably comprehensible.

The models were instructed using a highly constrained zero-shot template. To prevent standard textbook syntax from interfering with evaluation, the prompt demanded a specific output format (FINAL ANSWER: <expression>) and enforced explicit negative constraints (e.g., instructing the model to use ˜ for NOT).

Degraded images were passed directly to the model APIs alongside the system prompt. To prevent data loss or false negatives caused by server-side stochasticity, network timeouts and rate-limit errors (e.g., HTTP 429) triggered an pause-and-retry loop.

The raw text output from the VLM was parsed using regular expressions to isolate the final answer block. A programmatic string-cleaning safety net was applied to standardize common LLM formatting inconsistencies that bypassed prompt constraints (e.g., automatically replacing a rogue + sign with the requested | operator for a logical OR).

Rather than relying on brittle, exact-string matching, the sanitized boolean expression was passed into SymPy, a Python library for symbolic mathematics. The model’s logic was evaluated against the known ground-truth expression using a strict symbolic equivalence check. This ensured that functionally identical answers written in varying associative orders (e.g., A & B versus B & A) were correctly counted as equivalent.

Phase 2 then tests two distinct visual structures:

Topological Routing (Digital Logic Circuits): A dataset of mathematically unique logic gate diagrams (e.g., AND, OR, NOT cascades) (Figure 3). This tests whether models can trace continuous paths (edges) between nodes, even when lines are broken by pixel noise or affine warping.

Coordinate Geometry (Karnaugh Maps): A dataset of 4-variable K-Maps, including both randomized distributions and specific pedagogical edge cases (e.g., four-corner wrap-around groupings) (Figure 3). This tests the models’ reliance on spatial-coordinate rigidity, as minor visual warping can shift binary values across rigid cell boundaries.

## 3.4. Model Selection

The selection of VLMs is dictated by accessibility; we evaluate models comparable to those available to standard undergraduate students, bypassing specialized or open-source models that require local deployment. For the manual evaluation in Phase 1, we utilize the flagship models available via free or university-provided web interfaces: Google’s Gemini 2.5 Pro and Gemini 2.5 Flash, Anthropic’s Claude Sonnet 4, and OpenAI’s GPT-4 Turbo (via Microsoft Copilot). For the automated evaluation in Phase 2, we target the “lightweight flagship” tier via developer APIs. We utilize Gemini 2.5 Flash,

Gemini 2.5 Pro, OpenAI’s GPT-4o-mini, and Anthropic’s Claude 4.5 Haiku to provide the closest match to the models used in Phase 1. Additionally, these models represent a balance between capability, speed, and scope of deployment (that is, models commonly used by third parties as the backend for their services) for multimodal architectures currently available to the general public.

## 4. Results

Our evaluation is structured around the two experimental phases. Phase 1 details the manual evaluation of regional perturbations on introductory circuit analysis problems. Phase 2 presents the automated evaluation of specific spatial and geometric vulnerabilities.

## 4.1. Phase 1: Manual Evaluation

To establish a baseline for real-world VLM interaction, we manually evaluated a curated subset of 25 introductory circuit problems. Authors ensured that each perturbed image tested remained legible enough to complete the problem successfully. From this baseline, a representative sample of seven images was selected to apply the heuristic transformations. This subset was chosen to encompass a range of difficulties, with the difficulty increasing by question number.

Table 1 summarizes the efficacy of perturbations across the question text, the figure, and the full combined image. We define a successful perturbation as one that alters the output of the model, a positive perturbation as a shift from an incorrect to a correct answer, and a negative perturbation as a shift from correct to incorrect. In general, the perturbation process produced some level of effectiveness across all model types, with certain models being more susceptible to specific perturbation types over others. For example, Claude was most susceptible to the photocopy perturbation alone, with enabling the remaining noise, lines, and warping perturbations actually reducing overall success. Gemini as a whole was more robust, with 2.5 Pro maintaining strong, but not complete, resistance to adversarial perturbation. The most interesting finding was the presence of positive perturbations, but due to the stochasticity of the web-interface and the time requirements for manually performing the experimentation, this cannot be proven to be an artifact of the perturbation process itself.

To mitigate the issue of stochasticity observed in the web-interface trials, we constructed an automated pipeline allowing us to perform a much larger, repeated set of experiments. This necessitated a transition from standard circuit diagrams as algorithmically generating complex, readable circuit diagrams with verified ground-truth solutions at scale was infeasible since it requires extensive manual verification. By shifting to well-defined mathematical structures like logic gates and coordinate maps, we could fully automate the generation and evaluation loop, enabling larger samples sizes to more confidently measure model robustness.

![](images/4e04dc7871b8a61e1f959a49b9806bc6c5899830260aabac8118c8598d322c6a.jpg)  
(a) Logic Diagram

![](images/9543d81b56771a0d1062c8cae14abe93948678d55690fe23a117f3ca6fad1944.jpg)  
(b) Perturbed Figure 3a

![](images/9d28e16f1b2b97fe9f36fadf15c2b876535ac26fc6cebc13d0c1fd397a158b34.jpg)

![](images/d6e4ca30ac35ee0684b1f157f93203651fe5828939f54d450862f9a0052404be.jpg)  
(c) Karnaugh Map  
(d) Perturbed Figure 3c  
Figure 3. Phase 2 automated evaluation datasets, contrasting continuous topological routing (a, b) with geometric alignment (c, d) under baseline and perturbed conditions.

## 4.2. Phase 2: Automated Evaluation

In Phase 2, we evaluated the models’ structural vulnerabilities by automating the analysis of two distinct datasets: 80 Logic Diagrams (testing topological routing) and 30 Karnaugh Maps (testing coordinate geometry). To account for stochasticity, we evaluated each image three times, recording Pass@Any (at least 1 of 3 passes correct), Majority Vote (at least 2 of 3 correct), and Strict Consistency (3 of 3 correct). We established a clean baseline with no perturbations, followed immediately by an evaluation using the perturbation suite of Warping, Noise, Lines, Photocopy. We specifically constrained Phase 2 to this “full suite” approach due to practical resource limitations; conducting ablation studies on individual perturbations would require many millions of additional API tokens and substantial time. Testing this upper bound allowed us to observe whether the models could withstand maximum adversarial degradation.

The results reveal an appreciable disparity in geometric reasoning capabilities across the tested models. GPT-4o-mini experienced complete failure, achieving 0.0% accuracy across all tiers on both the clean and perturbed datasets. This suggests a fundamental inability of the model’s vision encoder to maintain the rigid

Table 1. Comparison of Model Robustness Under Text, Figure or Both (Full) Perturbations
<table><tr><td></td><td></td><td colspan="4">Claude Sonnet 4</td><td colspan="4">Gemini 2.5 Pro</td><td colspan="4">Gemini 2.5 Flash</td><td colspan="4">GPT-4</td></tr><tr><td>Config</td><td>Q</td><td>Base</td><td>Txt</td><td>Fig</td><td>Full</td><td>Base</td><td>Txt</td><td>Fig</td><td>Full</td><td>Base</td><td>Txt</td><td>Fig</td><td>Full</td><td>Base</td><td>Txt</td><td>Fig</td><td>Full</td></tr><tr><td rowspan="8">All</td><td>1567</td><td>X</td><td></td><td></td><td></td><td></td><td>XX</td><td></td><td></td><td>X</td><td>XX</td><td>XX</td><td>XX</td><td>XX√</td><td>X</td><td>XX</td><td>XX√</td></tr><tr><td></td><td></td><td>XX</td><td>XX</td><td></td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td><td>&gt;&gt;&gt;</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>12</td><td></td><td></td><td>&gt;</td><td></td><td>XX√√&gt;√√</td><td></td><td></td><td>&gt;</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>13 14</td><td></td><td>√x√√</td><td>√X</td><td>Xx√√√√√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√X</td><td>√X</td><td></td><td>VX</td><td>√X</td><td>√X</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>&gt;&gt;√x</td><td></td><td></td><td></td></tr><tr><td>Acc</td><td>6/7</td><td>4/7</td><td>4/7</td><td>5/7</td><td>5/7</td><td>5/7</td><td>6/7</td><td>6/7</td><td>6/7</td><td>5/7</td><td>5/7</td><td>5/7</td><td>4/7</td><td>5/7</td><td>3/7</td><td>4/7</td></tr><tr><td rowspan="8">Photo</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>x&gt;&gt;&gt;√</td><td></td><td>XX</td><td></td><td></td><td></td><td>X</td><td></td><td>X√</td></tr><tr><td>156712</td><td></td><td>XX</td><td>XX√</td><td></td><td>Xx√√&gt;√√</td><td></td><td></td><td></td><td></td><td></td><td>√X</td><td>XX</td><td></td><td></td><td>XX</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>VX</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>&gt;&gt;</td><td></td><td></td><td></td><td></td></tr><tr><td>13</td><td></td><td>√</td><td>√XXX</td><td>XX√√XX√</td><td></td><td></td><td></td><td>&gt;&gt;</td><td></td><td>√X</td><td>√X</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>14</td><td></td><td></td><td></td><td></td><td></td><td></td><td>V</td><td></td><td>X&lt;&gt;&gt;&gt;&gt;√</td><td></td><td></td><td>&gt;&gt;</td><td>XX√√&gt;√x</td><td></td><td></td><td></td></tr><tr><td>Acc</td><td>6/7</td><td>4/7</td><td>2/7</td><td>3/7</td><td>5/7</td><td>7/7</td><td>6/7</td><td>6/7</td><td>6/7</td><td>4/7</td><td>5/7</td><td>5/7</td><td>4/7</td><td>6/7</td><td>5/7</td><td>5/7</td></tr><tr><td rowspan="8">NLW</td><td></td><td></td><td>X</td><td>X</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>X√</td></tr><tr><td>1567</td><td>X&gt;&gt;&gt;&gt;&gt;</td><td></td><td></td><td></td><td></td><td></td><td>√X</td><td></td><td></td><td>√X</td><td>XX√</td><td>√x√</td><td></td><td>X√</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>12</td><td></td><td>V</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>V</td><td></td><td></td><td></td><td></td><td>√XX</td></tr><tr><td>13</td><td></td><td></td><td></td><td>Xx√v√√&gt;</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√X</td><td></td><td>XX√√√√X</td><td>√X</td><td></td><td></td></tr><tr><td>14</td><td></td><td>V</td><td>L</td><td></td><td>XX√√&gt;√√</td><td></td><td>1</td><td>XX√√√√X</td><td>x&lt;&gt;&gt;&gt;</td><td></td><td></td><td>√</td><td></td><td></td><td>X</td><td></td></tr><tr><td>Acc</td><td>6/7</td><td>6/7</td><td>6/7</td><td>5/7</td><td>5/7</td><td>7/7</td><td>6/7</td></table>

Note: Base denotes baseline model performance (no perturbations). Txt and Fig denote performance under text prompts and image modifications respectively. NLW denotes the combination of the Noise, Lines and Warping perturbations. Cases where the model recovers from a baseline failure (X → ✓) indicate counter-intuitive helper perturbations.

2D grid alignment required for K-Map minimization, possibly flattening the visual matrix into a linear sequence rather than processing it as a coordinate map. Claude 4.5 Haiku demonstrated limited baseline capability, achieving a 16.7% Majority Vote on clean maps. Upon introducing adversarial perturbations, this performance degraded by more than half, dropping to a 6.7% Majority Vote.

Gemini 2.5 Flash established the highest cognitive baseline, achieving an 80.0% Majority Vote and 73.3% Strict Consistency on clean images. However, the introduction of visual noise successfully degraded this performance, reducing the Majority Vote to 66.7% and Strict Consistency to 53.3%. While the model remained robust enough to solve the majority of the perturbed problems, the 20-point drop in Strict Consistency demonstrates that visual obfuscation successfully introduces instability into the model’s reasoning paths. Ultimately, these results confirm that while perturbations act as a viable near-term stop-gap by degrading model reliability, robust models can still circumvent them, necessitating a structural shift in assessment design.

Gemini 2.5 Pro represented the most capable model in our evaluation. On the clean Karnaugh Map dataset, it achieved a 90.0% Strict Consistency baseline. Curiously, while the introduction of visual noise paradoxically increased its Majority Vote accuracy to a perfect 100%, its Strict Consistency simultaneously dropped to 70.0%. This confirms that while the heaviest models possess excellent overall performance, the perturbations can still be applied successfully.

Conversely, the evaluation of the Logic Diagrams dataset focuses on the models’ capacity for topological routing. As detailed in Table 2, Gemini 2.5 Flash exhibited excellent baseline robustness, achieving a 95.0% Strict Consistency on clean graphs. The introduction of the full perturbation suite successfully degraded this metric to 67.5%. Gemini 2.5 Pro demonstrated similar foundational strength, achieving a 92.5% Strict Consistency on clean graphs, which was successfully reduced to 71.2%. Claude 4.5 Haiku and GPT-4o-mini demonstrated limited foundational capability on these topological tasks, achieving only single-digit Strict Consistency scores under perturbation. Given Gemini’s ability to maintain a strong majority pass rate under heavy visual degradation we conjecture that sufficiently capable VLMs are inherently more resilient to noise when tracing continuous topological connections (wires) than when interpreting geometric alignments (K-Maps).

Table 2. Phase 2 Automated Evaluation: VLM Accuracy on Topological vs. Geometric Structures
<table><tr><td rowspan="2">Model</td><td rowspan="2">Condition</td><td colspan="3">Logic Diagrams (Topological)</td><td colspan="3">Karnaugh Maps (Geometric)</td></tr><tr><td>Pass@Any</td><td>Majority Vote</td><td>Strict (3/3)</td><td>Pass@Any</td><td>Majority Vote</td><td>Strict (3/3)</td></tr><tr><td rowspan="2">Gemini 2.5 Pro</td><td>Clean</td><td>100%</td><td>97.5%</td><td>92.5%</td><td>100%</td><td>90%</td><td>90%</td></tr><tr><td>Perturbed</td><td>82.5%</td><td>76.2%</td><td>71.2%</td><td>100%</td><td>100%</td><td>70%</td></tr><tr><td rowspan="2">Gemini 2.5 Flash</td><td>Clean</td><td>96.2%</td><td>96.2%</td><td>95.0%</td><td>86.7%</td><td>80.0%</td><td>73.3%</td></tr><tr><td>Perturbed</td><td>76.2%</td><td>70.0%</td><td>67.5%</td><td>86.7%</td><td>66.7%</td><td>53.3%</td></tr><tr><td rowspan="2">Claude 4.5 Haiku</td><td>Clean</td><td>17.5%</td><td>15.0%</td><td>12.5%</td><td>26.7%</td><td>16.7%</td><td>10.0%</td></tr><tr><td>Perturbed</td><td>7.5%</td><td>6.2%</td><td>5.0%</td><td>13.3%</td><td>6.7%</td><td>3.3%</td></tr><tr><td rowspan="2">GPT-40-mini</td><td>Clean</td><td>10.0%</td><td>8.8%</td><td>6.2%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>Perturbed</td><td>5.0%</td><td>3.8%</td><td>2.5%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr></table>

Note: Pass@Any indicates $\overline { { \geq } }$ 1 correct inference across three trials. Majority Vote indicates $\overline { { \geq 2 } }$ correct inferences. Strict requires identical results for all three trials. Best results for both Clean and Perturbed are show in bold.

## 5. Discussion

The results of Phase 1 (Table 1) and Phase 2 (Table 2) demonstrate that VLMs can possess substantially different effectiveness even on the unperturbed questions. Overall the Gemini family demonstrated good performance on both the original and perturbed questions, while models like GPT-4o-mini were largely incapable of solving even the original questions.

## 5.1. Model Complexity and Paradoxical Improvement

In general, less complex models (Claude 4.5 Haiku, GPT-4o-mini) demonstrated proportionally more vulnerability to spatial perturbation, compared to larger frontier models (Gemini 2.5 Pro). However, multi-pass evaluation still revealed instability within these larger models. Across both the Phase 1 manual circuit evaluations and the Phase 2 Karnaugh Maps, Gemini 2.5 Pro occasionally exhibited “positive perturbations”, where the introduction of noise shifted an incorrect baseline output to a correct one. In the Phase 2 K-Map dataset, perturbation increased the model’s Majority Vote accuracy from 90.0% to 100%. We hypothesize that this is an instance of “paradoxical improvement”, where a model performs better after the input data has been degraded, transformed, or otherwise made “worse” from a human perspective (Benzi et al., 1981; Zhai et al., 2023). We do note that the model’s Strict Consistency simultaneously degraded from 90.0% down to 70.0%. This confirms that while the model was overall successful, the perturbations do have some limited effect.

## 5.2. Efficacy of Perturbation Methods and Locations

Across all model types, the “Photocopy” perturbation proved more effective than the isolated Noise, Lines, and

Warping (NLW) perturbations. In Phase 1, the photocopy method generated 15 successful negative output shifts, compared to only 7 from the standard suite. As the Photocopy perturbation on its own is generally less visually disruptive than the NLW, this was unexpected behavior as intuition would align with the more severe the perturbation, the worse the results. We hypothesize that this may be another case of paradoxical improvement.

We also note that isolating the location of the attack yielded counter-intuitive results. Perturbations applied solely to the question text were the least effective at forcing an incorrect output, while perturbations applied to the full combined image were occasionally less effective than perturbations applied exclusively to the figure region. We hypothesize that introducing noise to the text may trigger an “error-correction” state within the VLM’s language pathway, causing it to expend greater computational effort to align the text with the image, thereby inadvertently improving its performance.

## 5.3. Human-Unrecognizable Inputs and the Limits of Obscurity

While heuristic visual perturbations serve as a functional near-term deterrent for trivial plagiarism, initial threshold testing revealed a likely time limit on this methodology. Current vision transformer models demonstrated an alarming capacity to solve circuit problems that were degraded well beyond the threshold of human legibility (Figure 1c). Because VLMs can already parse certain heavily corrupted images better than humans, attempting to secure visual assessments purely through image degradation is likely a futile arms race. Long-term academic integrity must rely on alternative methods to secure assessments, rather than relying on visual obscurity.

## 6. Limitations and Future Work

While this work establishes a framework for evaluating VLM spatial reasoning, several limitations remain. First, Phase 2 utilized a combined “full suite” perturbation approach due to the computational and token costs of testing at scale. Consequently, we did not perform an exhaustive ablation study isolating the efficacy of individual perturbations (e.g., affine warping versus pixel noise on Karnaugh Maps). We also caution against cross-phase comparison. Future work should conduct granular parameter tuning to identify the minimal perturbation required to induce model failure, as excess perturbation can increase model performance.

Second, model selection was intentionally constrained to represent the threat landscape most accessible to undergraduate students: free-tier web interfaces and optimized, low-cost “lightweight flagship” APIs. We did not evaluate bleeding-edge frontier models such as the Gemini 3 or GPT-5 families. Given the observed capability gap between Gemini 2.5 Flash and Pro, higher-capacity models may possess greater resilience. Future research should benchmark these models to forecast the lifespan of visual obfuscation defenses.

Third, degraded problem sets may create accessibility challenges for students, particularly those with vision impairments or who rely on assistive technologies, potentially limiting the effectiveness of these perturbations as a learning aid. Instructor workload may also increase when selecting perturbations that degrade VLM performance while remaining accessible to students. This challenge will grow as VLMs evolve, making these perturbations a stopgap until VLM performance is empirically evaluated. Future work should examine perturbations alongside modified assessment designs to determine whether they are as effective as converting existing content to new assessment methods.

Finally, although outside the scope of our computational evaluation, the impact of adversarial perturbations on student behavior warrants investigation. Encountering degraded assignment images may reduce trust in VLM output, potentially encouraging students to verify the model’s logic. A human-computer interaction study examining how visual perturbations affect trivial plagiarism and trust in model output may prove valuable.

## 7. Conclusion

This research demonstrates that modern Vision-Language Models present a significant challenge to academic integrity through “trivial plagiarism,” while also possessing exploitable weaknesses. Through a two-phase evaluation of introductory engineering and computing problems, we showed that heuristic image perturbations, such as affine warping and simulated photocopy degradation, can measurably degrade model accuracy in most configurations without destroying human interpretability. Testing also revealed appreciable capability differences between models, although even highly capable frontier models suffered measurable losses when evaluating perturbed problems.

Ultimately, computationally efficient visual perturbations provide a plausible near-term stopgap for educators seeking to deter low-effort academic dishonesty. However, as VLM capabilities increase, relying on “security by visual obscurity” will likely become infeasible. To ensure the long-term validity of visual assessments, the academic community must shift toward assessment designs inherently resistant to VLM-based plagiarism.

Data Availability Statement: All materials used to generate the results in this paper are openly available at: https://github.com/christopherburger/AMVT

## References

Ahn, L., Blum, M., Hopper, N., & Langford, J. (2003). Captcha: Using hard ai problems for security. Advances in Cryptology, Eurocrypt, 2656, 294–311. https://doi.org/10.1007/3- 540-39200-9 18

Benzi, R., Sutera, A., & Vulpiani, A. (1981). The mechanism of stochastic resonance. Journal of Physics A: mathematical and general, 14(11), L453–L457.

Burger, C., Chen, L., & Le, T. (2023). “Are Your Explanations Reliable?” Investigating the Stability of LIME in Explaining Text Classifiers by Marrying XAI and Adversarial Attack. The 2023 Conference on Empirical Methods in Natural Language Processing.

Burger, C., Talley, K., & Trotter, C. (2026). “can ai recognize its own reflection? self-detection performance of llms in computing education”. Proceedings of the 59th Hawaii International Conference on System Sciences.

Cotton, D. R. E., Cotton, P. A., & Shipway, J. R. (2024). Chatting and cheating: Ensuring academic integrity in the era of chatgpt. Innovations in Education and Teaching International, 61(2), 228–239. https://doi.org/10.1080/14703297. 2023.2190148

Deng, N., Gu, L., Ye, S., He, Y., Chen, Z., Li, S., Wang, H., Yin, J., Wei, Q., Yang, T., et al. (2026). Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. International Conference on Learning Representations, 2026, 50773–50814.

Geirhos, R., Temme, C. R. M., Rauber, J., Schutt, H. H.,¨ Bethge, M., & Wichmann, F. A. (2018). Generalisation in humans and deep neural networks. Proceedings of the 32nd International Conference on Neural Information Processing Systems, 7549–7561.

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.

Jost, G., Taneski, V., & Karakatiˇ c, S. (2024). The impactˇ of large language models on programming education and student learning outcomes. Applied Sciences, 14(10). https://doi.org/10. 3390/app14104115

Mahmood, K., Mahmood, R., & Van Dijk, M. (2021). On the robustness of vision transformers to adversarial examples. Proceedings of the IEEE/CVF international conference on computer vision, 7838–7847.

Marr, D. (2010). Vision. MIT Press.

Miles, P. J., Campbell, M., & Ruxton, G. D. (2022). Why students cheat and how understanding this can help reduce the frequency of academic misconduct in higher education: A literature review. Journal of Undergraduate Neuroscience Education, 20(2), A150.

Nikolic, S., Sandison, C., Haque, R., Daniel, S., Grundy, S., Belkina, M., Lyden, S., Hassan, G. M., & Neal, P. (2024). Chatgpt, copilot, gemini, scispace and wolfram versus higher education assessments: An updated multi-institutional study of the academic integrity impacts of generative artificial intelligence (genai) on assessment, teaching and learning in engineering. Australasian Journal of Engineering Education, 29(2), 126–153. https: //doi.org/10.1080/22054952.2024.2372154

Ofqual. (2024). Principles of ai use in assessment and marking (tech. rep.). Office of Qualifications and Examinations Regulation, UK. https://www. gov.uk/government/publications/principles-ofai-use-in-marking

Perkins, M., Roe, J., Postma, D., McGaughran, J., & Hickerson, D. (2023). Detection of gpt-4 generated text in higher education: Combining academic judgement and software to identify generative ai tool misuse. Journal of Academic

Ethics, 22, 1–25. https : / / doi . org / 10 . 1007 / s10805-023-09492-6

Peters, M., & Angelov, D. (2025). Redefining assessment tasks to promote students’ creativity and integrity in the age of generative artificial intelligence. International Journalfor Educational Integrity, 21(1), 25.

Salim, S. I., Yang, R. Y., Cooper, A., Ray, S., Debray, S., & Rahaman, S. (2024). Impeding llm-assisted cheating in introductory programming assignments via adversarial perturbation. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 445–463.

Stogiannidis, I., McDonagh, S., & Tsaftaris, S. A. (2025). Mind the gap: Benchmarking spatial reasoning in vision-language models. https://arxiv.org/ abs/2503.19707

Teel, Z. A., Wang, T., & Lund, B. (2023). Chatgpt conundrums: Probing plagiarism and parroting problems in higher education practices. College & Research Libraries News, 84, 205–208. https: //doi.org/10.5860/crln.84.6.205

Toba, H., Karnalim, O., Johan, M. C., Tada, T., Djajalaksana, Y. M., & Vivaldy, T. (2023). Inappropriate benefits and identification of chatgpt misuse in programming tests: A controlled experiment. https://arxiv.org/abs/ 2309.16697

Wahle, J. P., Ruas, T., Kirstein, F., & Gipp, B. (2022, December). How large language models are transforming machine-paraphrase plagiarism. In Y. Goldberg, Z. Kozareva, & Y. Zhang (Eds.), Proceedings of the 2022 conference on empirical methods in natural language processing (pp. 952–963). Association for Computational Linguistics. https : / / doi . org / 10.18653/v1/2022.emnlp-main.62

Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltynek, T., Guerrero-Dib, J., Popoola, O.,´ Sigut, P., & Waddington, L. (2023). Testing<sup>ˇ</sup> of detection tools for ai-generated text. International Journal for Educational Integrity, 19(1). https://doi.org/10.1007/s40979- 023- 00146-z

Zhai, Z.-M., Kong, L.-W., & Lai, Y.-C. (2023). Emergence of a resonance in machine learning. Physical Review Research, 5(3), 033127.