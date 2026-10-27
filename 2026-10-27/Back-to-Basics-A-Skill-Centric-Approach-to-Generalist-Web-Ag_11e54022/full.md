# Back to Basics: A Skill-Centric Approach to Generalist Web Agents

Minh Pham-Dinh<sup>1</sup>, Md Farhan Ishmam<sup>1</sup>, Hernan Yupanqui<sup>1</sup>, Reaz Hassan Joarder<sup>2</sup>, Kenneth Marino<sup>1</sup> <sup>1</sup>University of Utah <sup>2</sup>Islamic University of Technology {m.pham-dinh,farhan.ishmam,kenneth.marino}@utah.edu

Website Code Dataset

## Abstract

Browser agents do not just need more practice. They need to practice the right things. We introduce MINIWEB, a realistic miniature of the web comprising 65 interconnected websites across 15 domains, where agents navigate, search, and transact as people do. Its 369 human-annotated tasks cover everyday workflows and label 37 reusable interaction primitives grouped into 7 skill families, allowing us to diagnose which skillsfail rather than only whether a task succeeds. Benchmarking ten open-weight agents, we find that drag & gesture is consistently the weakestfamily, with no agent solving more than 24% ofits instances. To address this gap, we introduce MINIWEBGEN, a framework that autonomously generates verified, skill-centric training data from MINIWEB. Fine-tuning Qwen3.5-4B on just 3.1K verified trajectories improves single-primitive success by 24.9pp across 13 unseen websites, with gains across every skillfamily. However, these trajectories teach short-horizon interactions, which do not directly translate to full-task success. We build on this result with MINIWEBAGENT, which pairs the trained executor with a planner that decomposes full tasks into the skill-level steps it learned to perform. This approach improves task success by 19.4pp on unseen MINIWEB sites and transfers to external benchmarks, with gains of 9.3pp on WebArena-Lite, 16.1pp on Online-Mind2Web, and 20.0pp on WebVoyager, without training on their trajectories. We provide a simple recipe for compact browser agents: practice reusable skills directly, then compose them to solve unfamiliar tasks.

## 1. Introduction

What makes humans generalists? We rarely learn a complex behavior from scratch. Before learning to ride a bike, we have already spent years learning to balance, coordinate our limbs, recover from mistakes, and navigate through space. New tasks become easier because they reuse skills we have “Search CascadiaMaps for ‘cofee’, set the minimum rating to 4.0 stars, and open Cedar Cup Cofee.”

![](images/fb9bc289eb9ed847508d8c13ed6e6b20be11e0f662637fa47e7a77fdc4f98890.jpg)  
Ten agents, by skill family

![](images/efe1572f7f9503d77efe427e0c0f9d19299583e1afec68ed6b1a73de3197455d.jpg)

![](images/4065de9d4c5d305bfe9a648352dff9d52f2634fc78849174d70e5dca3ca7a76f.jpg)  
Figure 1. MINIWEB grades every primitive of a task. Top: Fara1.5-4B opens the right café but leaves the rating filter at Any. Bottom left: Overall skill profile of 10 evaluated agents. Bottom right: task success of Qwen3.5-4B before and after our training.

practiced in many other settings. Generalization comes not only from seeing more tasks, but from building a repertoire of reusable skills and learning how to compose them.

Web agents should be able to learn in the same way. Current approaches to training general web agents focus on collecting more task trajectories on diverse websites, difficulty, and solution strategies [4, 5, 12, 31]. While large models can benefit from this scale, compact open-weight agents have limited capacity and training budgets, making the choice of what to practice especially important [3]. We therefore ask: should a small web agent spend its training budget on more complete tasks, or on the reusable interaction skills those tasks are builtfrom? We study the latter, training on short-horizon, skill-centric trajectories and composing the learned skills at inference time.

We introduce MINIWEB, a collection of 65 websites (Fig. 2) built around 37 reusable interaction primitives. Rather than starting from tasks and labeling their skills afterward, we build the environment from the primitives upward (Fig. 3) and compose them into 369 human-authored tasks with skill-level verification. This gives us a controlled place to both measure what agents know and practice what they do not.

Across ten open-weight agents, we found that drag & gesture is the weakest family for every model (Fig. 1), with only 12.4% average success compared with 38.0% for the next-weakest family. Web- and computer-use-specialized agents perform even worse on dragging than general VLMs despite being stronger on common interactions such as typing. We believe that scaling end-to-end task trajectories can systematically favor frequently encountered interactions while leaving rarer but reusable skills underpracticed.

To test whether skill-specific practice closes these gaps, we introduce MINIWEBGEN, a multi-agent pipeline that generates verified trajectories for skills in MINIWEB. Training Qwen3.5-4B on 3.1K generated trajectories improves success on held-out single-primitive tasks by 24.9 points, with gains across all skill families. These gains do not automatically transfer to full tasks, so we introduce MINIWEBAGENT, which pairs the trained executor with a planner that decomposes tasks into practiced interactions. This approach improves success on held-out MINIWEB sites by 19.4 points and transfers to three external benchmarks without target-benchmark training: +9.3 points on WebArena-Lite, +20.0 on WebVoyager, and +16.1 on Online-Mind2Web. Overall, we contribute the following:

MINIWEB, a benchmark of 65 websites and 369 tasks built from 37 reusable interaction primitives with skilllevel verification, revealing that drag & gesture is the weakest skill family for all ten open-weight agents we evaluate.

❖ MINIWEBGEN, a multi-agent pipeline for generating verified, skill-specific trajectories, which at matched training size outperforms task-centric trajectories from InSTA [27] on all 4 benchmarks.

❖ MINIWEBAGENT, a compact browser agent that composes practiced skills via a planner and transfers to WebArena-Lite, WebVoyager, and Online-Mind2Web without target-benchmark training.

<table><tr><td rowspan="6">WebShop WebVoyager WebArena Mind2Web</td><td>Text</td><td>Select</td><td>Form</td><td>Nav</td><td>I/O</td><td>Drag</td></tr><tr><td>00□000□</td><td>O</td><td>000000000</td><td>□□</td><td>□00□</td><td>□</td></tr><tr><td>□□□□O</td><td></td><td></td><td>□</td><td>□□□□</td><td>□</td></tr><tr><td>OOO</td><td></td><td>O</td><td>□□</td><td>0O□</td><td></td></tr><tr><td>□□□□</td><td></td><td></td><td>□</td><td>□□</td><td>_□</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>□</td></tr><tr><td colspan="3">MiniWeb</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="4">Square = 1 primitive; share of the dataset:  ≥1% □ none</td><td> &lt;1%</td><td></td><td></td></tr></table>

<table><tr><td></td><td>WebArena</td><td>WebVoyager</td><td>OM2W</td><td>MINIWEB</td></tr><tr><td>Scale</td><td></td><td></td><td></td><td></td></tr><tr><td>Tasks</td><td>812</td><td>643</td><td>300</td><td>369</td></tr><tr><td>Websites</td><td>5</td><td>15</td><td>136</td><td>65</td></tr><tr><td>Hosting</td><td>Local</td><td>Live</td><td>Live</td><td>Local</td></tr><tr><td>Skill-level evaluation</td><td></td><td></td><td></td><td></td></tr><tr><td>Skill-labeled steps</td><td>x</td><td>x</td><td>x</td><td></td></tr><tr><td>Segmented demos</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Per-skill verdicts</td><td>x</td><td>x</td><td>x</td><td>√</td></tr></table>

Table 1. MINIWEB attributes task failures to skills. Top: our 36 primitives (reasoning excluded) by family; benchmark rows use our labels of 15.7K tasks, and MINIWEB uses MINIWEBGEN’s trajectories (Supp. A.4). Bottom: MINIWEB labels, segments, and verifies every step by skill; MiniWoB++ [17, 25] and ComponentBench [11] test interactions only outside tasks. OM2W: Online-Mind2Web [34].

## 2. Related Work

Web benchmarks and synthetic environments. Web benchmarks aim for realism, reproducibility, or coverage. MiniWoB and MiniWoB++ [17, 25] pioneered synthetic environments for low-level web interaction, with small self-contained pages that exercise individual widgets such as sliders and drag targets. Self-hosted environments such as WebShop [35], WebArena [39], WorkArena [7], and MyPCBench [15] provide realistic locally hosted websites, but are limited in term of coverage, while Mind2Web [6] and live-web benchmarks [9, 13, 16, 34] broaden coverage at the cost of task stability and reproducibility. Environment-generation pipelines address this by compiling real-site interaction structures into finite-state machines [31], generating environments with verifiable tasks [5, 38], or synthesizing versions of webpages [14]. Most of these benchmarks score whole tasks: step-level metrics [6], progress rates [18], and key-node evaluation [22]. They measure how far an agent gets but do not attribute a failure to a specific skill. Concurrent with our work, ComponentBench [11] evaluates agents on 97 canonical UI components, including sliders and date pickers, using programmatically verified tasks across three React component libraries. MINIWEB shares this focus on interaction skills but embeds them in full multi-page websites with persistent backend interactions. It also composes primitives into multi-step, cross-site tasks and provides skillsegmented demonstrations with per-primitive verdicts.

![](images/c17fd0aa869007348e2e94b4fa9a662b4029de00800ddcc1dfda316722671e9f.jpg)  
Figure 2. The 65 websites. Left: sites by domain (squares: held-out), linked to hubs at the center (colored edges) and by cross-site tasks (dashed). Right: the 65 MINIWEB sites, divided by data source.

Training data and skill composition for web agents. Web agents are traditionally trained on end-to-end task trajectories [12, 13, 24, 28, 32]. Recent work has scaled these training corpora by converting tutorials into demonstrations [21, 33], relabeling or reverse-synthesizing tasks from exploration [19, 26], or collecting trajectories on the live web [27]. These pipelines inherit the coverage of their sources, whereas MINIWEBGEN uses privileged knowledge of each site to direct generation at predefined primitives, including interactions that end-to-end data rarely contains (dragging, searching, . . . ). While other work reuses skills at inference time as induced workflows [30] or distilled APIs [37], we instead train skills into an executor’s weights through practice. To compose them, MINIWEBAGENT adopts the established separation of planning from execution found in planner–grounder designs [10, 36], manager–worker frameworks [1, 2], and plan-and-act agents [8, 23]. Our contribution is not this architecture but the experience used to train the executor: we examine whether targeted practice improves individual interactions on unseen websites, and whether a planner can compose those improvements into more successful multistep task execution.

## 3. Methodology

We want to study the primitive skills, or macros, of current web agents. Given a task that requires a specific set of skills, can an agent reliably perform each one? When the task fails, can we identify which skill it struggled with? To answer these questions, we need tasks whose component skills are explicit and can be evaluated separately, which current web-agent benchmarks do not provide [6, 9, 13–15, 35, 39] (Tab. 1).

## 3.1. MINIWEB

We introduce MINIWEB, a collection of 65 synthetic websites built around 37 interaction primitives. We build it from the bottom up (Fig. 3): First the primitives, then supporting websites, and finally tasks that chain them together.

Primitive selection and schema. The set of primitives a web agent needs is hard to enumerate from first principles, so we start from existing data. With Claude Opus 4.8, we label 15.7K tasks from WebArena [39], WebShop [35], WebVoyager [13], and Mind2Web [6] with the recurring interaction patterns that each requires, reusing an existing label whenever one fits. This first pass yields 139 candidate macros, which are dominated by searching, filtering, and extracting information, and only 0.3% of their 52K labels involve dragging and none drawing, uploading a file, or editing a spreadsheet cell. In group brainstorming sessions, we merge near-duplicate and site-specific candidates, add the interactions the source benchmarks miss, and separate reasoning from the physical interaction. The result is an initial set of 46 primitives, which we refine after annotation. Each is a site-agnostic interaction with a single purpose that may take several browser actions, such as filtering items with a slider or editing a spreadsheet cell. We group them into 7 skill families (Fig. 3). Formally, each primitive $m \in \mathcal { M } ( | \mathcal { M } | = 3 7$ after refinement) has a family $\phi ( m ) \in \mathcal { F } \left( | \mathcal { F } | = 7 \right)$ , a span rule that marks where it starts and ends in a trajectory, and a verifier $V _ { m }$ that checks its outcome. Reasoning is an optional operation $\rho \in { \mathcal { O } }$ (e.g., count) attached to a primitive, whose verifier then also checks the result: $V _ { m \cdot \rho } = V _ { m } \wedge V _ { \rho }$

![](images/d77a936e1d60b21b758b2c767b30b779dbb18cfe4b44fe9cb36423a7ce8ef427.jpg)  
Figure 3. Building MINIWEB bottom up. (1) Tasks of existing benchmarks are mined into 37 primitives in 7 families. (2) Sites are built around their primitives. (3) Tasks chain primitives, with a verdict per primitive. Takeaway: every step of a task maps to one skill that can be graded.

Website construction. We select 65 websites across 15 representative domains to cover a broad range of interaction primitives and cross-site workflows. Half of these domains are absent from WebArena [39], Mind2Web [6], WebVoyager [13], and WebShop [35]. We then have annotators specify each site’s supported primitives, features, and connections to other sites. These links form an information flow graph, which allow for realistic cross-site features such as payment confirmations or cross-site sharing. For realistic content, 21 sites are seeded from public datasets<sup>1</sup>, which we join through a synthetic layer of users and accounts. The remaining 44 sites use deterministic synthetic data, for about 4M records in total. More details can be seen in Fig. 2. A coding agent<sup>2</sup> implements each site’s frontend and backend from its specification, and all the authors and annotators review every site and fix it until it supports its primitives.

The entire process took roughly a month with 6 annotators working 8 hours a week. The backends log every request in a session and write all changes to a per-session overlay, so verifiers can inspect what an agent did, and many agents can run in parallel from the same initial state.

Task annotation and verification. An episode is a trajectory $\tau = ( o _ { 0 } , a _ { 1 } , o _ { 1 } , \dots , a _ { T } , o _ { T } )$ of observations and browser actions, during which the backend records a request log L(τ). A task $\mathcal { T } = ( g , \langle m _ { 1 } , \dots , m _ { k } \rangle )$ pairs an instruction g with a chain of primitives. Annotators compose tasks from supported primitives, specify their dependencies, and write realistic instructions. They then perform each task while our tool records browser interactions and server requests, marking the span $[ b _ { j } , e _ { j } ]$ of each $m _ { j }$ to produce a skill-segmented demonstration τ<sup>⋆</sup>. Six annotators produced 369 tasks over three months, including 43 cross-site tasks. At the end, the first author manually reviewed every task together with its verifier and demonstration, correcting or removing those with errors. As a final refinement of the primitive set, we drop the 9 hyper-specific primitives that annotators used fewer than three times. The remaining 37 primitives in 7 families form the final set M.

We use two graders. Parameterized verifiers [38] check observable interactions against the backend request log, while a VLM judge [34] evaluates semantic and reasoning outcomes using the primitive’s rubric, request log, and screenshots. A task succeeds only if every required primitive $j \in R$ passes,

$$
\begin{array} { r } { \mathrm { S u c c } ( \tau ) = \bigwedge _ { j \in { \cal R } } { V _ { m _ { j } } \left( \tau , L ( \tau ) \right) } , } \end{array}\tag{1}
$$

and the per-primitive verdicts show where the agent struggled (Fig. 3).

Grader validity. We validate both graders on 183 human demonstrations that remain valid on the current sites and pass the task verifier. For each, we construct two negatives: one truncated before the final required primitive and one empty episode. Across these 549 cases, the deterministic verifiers achieve 98.9% accuracy and the judge 98.2%. On 3,689 episodes from the ten untrained agents, the two graders agree on 93.1% of task outcomes (Cohen’s κ = 0.85). A human audit of 120 agent episodes, stratified by the graders’ verdicts and weighted back to all episodes, finds the judge right on 92.2% of task outcomes (κ = 0.84) and the verifiers on 88.9% (κ = 0.77), close to the agreement between two human labelers $( \kappa ~ = ~ 0 . 8 6 )$ . Supp. E details the validation protocol, the recordings we leave out, the grader disagreements of each agent, and the audit.

## 3.2. MINIWEBGEN

Annotating enough human demonstrations to cover each primitive can be challenging, so MINIWEBGEN scales them into verified single-primitive trajectories on the training sites $\mathcal { W } _ { \mathrm { t r a i n } }$ (Fig. 4, left), using privileged knowledge of the benchmark: a crawler maps the controls $\mathcal { C } _ { m } ( w )$ that realize primitive m on site w and probes the request each control sends. Generation has five steps.

![](images/ae722cb98be235b38e8ac42555f3f580be0f6cf5e71f48d701baea41bb050f67.jpg)  
Figure 4. MINIWEBGEN (left) generates and gates single-primitive trajectories (real ShopHub example). MINIWEBAGENT (right): an untrained Qwen3.5-4B planner hands each step to one trained LoRA executor (real held-out trace).

(1) Summarize. Inspired by TimeWarp [14], an LLM first summarizes the human demonstration spans $H _ { m }$ of each primitive m on training sites (up to 24, each with brief context) into a site-agnostic guide $G _ { m } \colon$ preconditions, steps, pitfalls, and a completion condition.

(2) Sample. A task configuration is drawn as

$$
( w , c , z , s _ { 0 } ) \sim P _ { m } , \quad w \in \mathcal { W } _ { \mathrm { t r a i n } } , \ c \in \mathcal { C } _ { m } ( w ) ,\tag{2}
$$

where z is an argument of control c and $s _ { 0 }$ a start state that may already carry another filter or follow another primitive. $P _ { m }$ spreads samples evenly over sites and controls.

(3) Suggest. An LLM phrases the instruction g. The check $\kappa : { \mathcal { S } }  \{ 0 , 1 \}$ is never written by the LLM: it is the probed request of (c, z), or the data changes that two dry runs agree on.

(4) Execute. A coding LLM, given $g , G _ { m }$ , and read access to the DOM, writes a short script against a restricted wrapper that acts only through mouse and keyboard events on visible elements. Every step is recorded, giving $\tau = ( o _ { 0 } , a _ { 1 } , \dots , a _ { T } , o _ { T } )$ with screenshots $o _ { t }$ and pixel actions $a _ { t }$

(5) Gate. An attempt that ends in state $s _ { T }$ is kept only if

$$
\kappa ( s _ { 0 } ) = 0 \ \wedge \ \kappa ( s _ { T } ) = 1 ,\tag{3}
$$

so no attempt passes without acting. The kept set $\mathcal { D } _ { \mathrm { g e n } } = \{ ( g , \tau ) \}$ contains only what a student model will see; privileged information decides only what is generated and how. With Gemini 3.5 Flash, $\mathcal { D } _ { \mathrm { g e n } }$ holds 3,128 trajectories at about \$0.04 each: 2,687 for 36 of the 37 primitives, and 441 for primitives later dropped in the refinement, which we keep for training. The answer-reporting primitive, report\_information, gets no generated data; in MINIWEBAGENT, reasoning steps run on the base model (Sec. 3.3).

## 3.3. MINIWEBAGENT

A model fine-tuned on MINIWEBGEN’s single-primitive trajectories inherits their short horizon and their primitivelevel instructions (Sec. 4.2), so we add a planner that chains primitives. To that end, we introduce MINIWEBAGENT, a planner-executor agent over one base model $\pi _ { 0 }$ (Fig. 4, right). Its planner is $\pi _ { 0 }$ itself, without training. The planner observes o (the URL, the interactive elements, and a screenshot) but cannot act on the page; at each turn n it emits

$$
u _ { n } \in ( { \mathcal { G } } \times { \mathcal { T } } ) \cup \{ { \mathsf { d o n e } } ( \alpha ) \} ,\tag{4}
$$

either a delegation, a family group $\gamma \in \mathcal { G } =$ {drag, select, io, rest, reasoning} (rest covers text entry, forms, and navigation) with a short instruction $\iota \in \mathcal { T }$ , or a final answer α. A delegation runs the executor $\pi _ { \mathrm { p o o l } }$ , which is $\pi _ { 0 }$ with one LoRA adapter $\Delta _ { \mathrm { p o o l } }$ trained on every skill family, on the same browser for at most 15 actions, $a _ { t } \sim \pi _ { \mathrm { p o o l } } ( \cdot \mid \iota , o _ { t } )$ and returns what it did and whether the page changed. Reasoning steps, which only read the page and report, run on $\pi _ { 0 } ,$ since MINIWEBGEN generates no reasoning data. One served model hosts the planner and the executor.

<table><tr><td></td><td></td><td>Task</td><td colspan="6">Skill family: instances passed</td><td colspan="4">Reasoning operation</td></tr><tr><td>Agent</td><td></td><td>Size success</td><td>Nav.</td><td>Text</td><td>Select</td><td>Form</td><td>I/O</td><td>Drag</td><td>Reason</td><td></td><td></td><td>read verify extremum count</td></tr><tr><td colspan="9">General VLMs, browser-use harness</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-4B</td><td>4B</td><td>40.1 (2.5)</td><td>55 (6) 39(4)</td><td>42(4)</td><td>37 (4)</td><td>32 (6)</td><td>12(8)</td><td>60(4)</td><td>56 (5) 43 (11)</td><td></td><td>37(7)</td><td>31 (8)</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>54.7 (2.5)</td><td>71 (5)</td><td>55(4) 60 (4)</td><td>62(4)</td><td>61 (6)</td><td>24(10)</td><td>60(4)</td><td>65 (5) 71 (10)</td><td></td><td>46(8)</td><td>44(8)</td></tr><tr><td>Qwen3-VL-8B</td><td>8B</td><td>43.8 (2.6)</td><td>63 (6) 43(4)</td><td>42(4)</td><td>53 (4)</td><td>50(7)</td><td>12 (8)</td><td>52 (5)</td><td></td><td>53 (5) 62 (10)</td><td>41 (8)</td><td>43 (9)</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>39.0 (2.5)</td><td>61 (6) 44(4)</td><td>41 (4)</td><td></td><td>42(4) 32 (6)</td><td>24 (10)</td><td>45 (5)</td><td></td><td>46 (5) 48 (11)</td><td>32(7)</td><td>33 (8)</td></tr><tr><td colspan="9">Web and computer-use agents, own agent loop</td><td></td><td></td><td></td><td></td></tr><tr><td>Fara1.5-4B</td><td>4B</td><td>53.1 (2.7)</td><td>79 (5)</td><td>63(4) 54(4)</td><td>58(4)</td><td>55 (7)</td><td>18(10)</td><td>64(4)</td><td></td><td>60 (5) 76 (9)</td><td>56 (8)</td><td>47 (9)</td></tr><tr><td>Fara-7B</td><td>7B</td><td>29.3 (2.3)</td><td>54 (6)</td><td>54(4)</td><td>26(3)</td><td>40(4) 41 (7)</td><td>0 (0)</td><td>34(4)</td><td></td><td>41 (5) 29 (10)</td><td>24(7)</td><td>19(7)</td></tr><tr><td>MolmoWeb-4B</td><td>4B</td><td>35.0 (2.4)</td><td>61 (6)</td><td>54(4)</td><td>40(4)</td><td>38(4) 27 (5)</td><td>18(10)</td><td>45 (4)</td><td></td><td>44(5) 38 (11)</td><td>27 (7)</td><td>28 (7)</td></tr><tr><td>MolmoWeb-8B</td><td>8B</td><td>39.3 (2.4)</td><td>62 (5)</td><td>55(4)</td><td>47(4)</td><td>42 (4) 32(7)</td><td>6 (6)</td><td>51 (4)</td><td></td><td>49 (5) 52 (11)</td><td>41 (8)</td><td>28 (8)</td></tr><tr><td>UI-TARS-1.5-7B</td><td>7B</td><td>25.5 (2.4)</td><td>54 (6)</td><td>47 (5)</td><td>25 (3)</td><td>27 (3) 20(6)</td><td>0 (0)</td><td>26 (4)</td><td></td><td>25 (5) 33 (10)</td><td>32(7)</td><td>11(5)</td></tr><tr><td>OpenCUA-7B</td><td>7B</td><td>29.5 (2.3)</td><td>68 (5)</td><td>47(4)</td><td>31 (3)</td><td>32 (3) 30(6)</td><td>12(8)</td><td>33 (4)</td><td></td><td>29 (5) 52 (11)</td><td>34(7)</td><td>22(7)</td></tr></table>

Table 2. Skill profiles of untrained agents on all 369 MINIWEB tasks. Task success, then the share of primitive instances passed (%) per skill family and reasoning operation, all judged by the VLM judge (Gemini 3.5 Flash). One run per agent; in brackets: standard erro over 1,000 bootstrap resamples of the tasks. Red: each agent’s weakest skill family; green: its strongest. Bold: best per column.

Training. We replay each $( g , \tau ) \in \mathcal { D } _ { \mathrm { g e n } }$ through the agent’s browser-use harness, which turns step t into the exact test-time prompt $x _ { t }$ and a target $y _ { t }$ (the action and a short memory, which the base model Qwen3.5-4B writes). We add variants that inject and then correct a mistake (629) or chain two trajectories on one site (857). With the 3,128 plain trajectories, this gives D, with 27.6K steps. The adapter $\Delta _ { \mathrm { p o o l } }$ , a rank-64 LoRA on the language model’s attention and MLP projections, is fit on all of D for one epoch (details in Supp. J) with the token-averaged cross-entropy

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C E } } ( \Delta ; \mathcal { D } ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \frac { 1 } { | y | } \sum _ { i } \log \pi _ { \Delta } ( y _ { i } \mid x , y _ { < i } ) . } \end{array}\tag{5}
$$

No trajectory from the held-out sites or from an external benchmark is used for training.

## 4. Experiments and Analyses

We use MINIWEB to ask four questions. Which skills do current web agents lack (Sec. 4.1)? Does practicing skills in isolation improve them, and the tasks that chain them (Sec. 4.2)? Does an agent that composes trained skills transfer to websites and benchmarks it never saw (Sec. 4.3)? And does our skill-centric data transfer better than a task-centric corpus of the same size (Sec. 4.4)?

Setup. We profile ten open-weight agents (Tab. 2): four general models in the browser-use harness [20] and six web or computer-use agents in their native agent loops [4, 5, 12, 24, 29]. For training experiments, we use Qwen3.5-4B and compare an untrained single agent, a single agent trained on all MINIWEBGEN trajectories, an untrained plannerexecutor, and MINIWEBAGENT, which pairs the planner with the trained executor (Tab. 3). To isolate the effect of training data, we also train the same executor on a sizematched set of task-centric trajectories collected on our training sites with InSTA’s pipeline [27].

Training uses only the 52 MINIWEB training sites. We evaluate on 62 human-authored tasks and 103 generated single-primitive tasks from 13 held-out sites, as well as WebArena-Lite, WebVoyager, and Online-Mind2Web. For the live benchmarks, we exclude tasks that are impossible or blocked by our network; full evaluation details, action budgets, repeated-run protocol, and statistical tests appear in Supp. J and Supp. H. We report our key findings below:

## 4.1. Drag & gesture is the weakest family in all web agent models.

As shown in Tab. 2, drag & gesture is the weakest skill family for every untrained agent. Web and computer-use agents perform worse on drag than general models (9.0% vs. 18.0% on average), despite performing better on typing, suggesting that end-to-end web training underrepresents rarer interactions such as dragging. Although drag is our smallest family, the gap is large: pooled across agents, drag reaches only 12.4% success, compared with 38.0% for the next-weakest family, out-of-page I/O.

The failures are also highly localized. In 72% of failed drag instances, the agent reaches the correct slider or canvas but fails during the gesture or immediately afterward (Fig. 5). Dragging is typically planned open-loop from a single screenshot, while the resulting value only becomes visible after release, so small spatial errors can directly produce an incorrect result. When agents do set sliders exactly, they often avoid dragging altogether by using arrow keys or directly entering the value. This suggests that the bottleneck is not reaching the right interface element, but executing and verifying the gesture itself.

![](images/050e1a93aab67e35d37dc959b303259b8a8f7418694f2efa830abc3eee76d34e.jpg)  
Figure 5. Most drag failures happen at the gesture itself. Outcomes of the ten untrained agents on every drag & gesture instance (11 slider and 6 signature tasks), from the VLM judge’s explanations; bypassed means, e.g., typing a name instead of drawing it.

More broadly, the ten agents exhibit distinct skill profiles even when their overall task success is similar. Task success correlates strongly with discrete selection, form transactions, and reasoning (Spearman ρ of 0.96, 0.83, and 0.95), but only weakly with text entry (ρ = 0.21). For example, Qwen3.5-9B and Fara1.5-4B differ by only 1.6 points in task success, yet Fara1.5-4B is 8 points better at navigation and 6 points worse at discrete selection. Reasoning skills show similar variation: aggregation is consistently difficult, with counting harder than reading a value for every agent and finding an extremum harder for eight of ten, likely because both require maintaining state across scrolls or pages.

These primitive-level weaknesses matter because they compound over longer tasks. As task length increases, success falls steeply (Fig. 6, left): averaged across the ten agents, an n-primitive task succeeds at roughly $p ^ { n }$ , within 3.8 points at every length, where $p = 6 2 \%$ is mean success on one-primitive tasks. Since every primitive in a chain must succeed, even modest gains at the primitive level compound across the task. This is precisely what skill-specific practice targets.

## 4.2. MINIWEB practice improves single-primitive task generation, but not task success.

Since task success compounds over a task’s primitives, MINIWEBGEN trains the agent directly on those primitives. This works at the skill level: on 103 generated singleprimitive tasks from held-out sites, Qwen3.5-4B improves from 55.3% to 80.3% success, with gains across every skill family (Fig. 7). But these gains do not transfer to full tasks. On 62 human-authored held-out tasks, primitive success remains essentially unchanged (44.1% to 45.1%), and so does task success (38.7% to 39.2%); performance also drops on all three external benchmarks (Tab. 3).

![](images/8b086b449a2f0f6334d09b4e82e7b533575fc503a9b56d96b7d3093845fac8bc.jpg)

Figure 6. Failures compound along a task’s chain of primitives. Left: the ten untrained agents of Tab. 2 on all 369 tasks (grey), with p their mean success on one-primitive tasks. Right: Qwen3.5-4B untrained (grey) or trained (blue) on the 62 held-out tasks (Tab. 3).
<table><tr><td>tasks</td><td>MINIWEB 62</td><td>WA-Lite 154</td><td>WV 536</td><td>OM2W 280</td></tr><tr><td colspan="5">Single agent, trained on</td></tr><tr><td>nothing</td><td>38.7 (5.5)</td><td>27.1 (3.0)</td><td>45.5 (2.2)</td><td>17.9 (2.2)</td></tr><tr><td>our data</td><td>39.2 (5.7)</td><td>20.3 (3.0) 32.8 (2.0)</td><td></td><td>8.6 (1.6)</td></tr><tr><td colspan="5">Planner, executor trained on</td></tr><tr><td>nothing</td><td>45.2 (5.9)</td><td>30.7 (3.3)</td><td>59.9 (2.2)</td><td>21.8 (2.4)</td></tr><tr><td>task-centric data</td><td>47.8 (5.7)</td><td>34.2 (3.4)</td><td>62.1 (2.1)</td><td>23.2 (2.6)</td></tr><tr><td>skill-centric data</td><td>58.1 (6.0)</td><td></td><td>36.4(3.3) 65.5(2.1) 33.9(2.8)</td><td></td></tr></table>

Table 3. Composing trained skills raises task success. Task success (%) of Qwen3.5-4B on MINIWEB’s held-out sites, WebArena-Lite (WA-Lite), WebVoyager (WV), and Online-Mind2Web (OM2W), as a single agent or as a planner whose executor is untrained or trained at matched size on task-centric trajectories or on MINIWEBGEN’s skill-centric trajectories. In brackets: standard error over 1,000 bootstrap resamples of the tasks. Green (red): significantly above (below) the untrained row of the group, $p < 0 . 0 5$ , by a paired test over the same tasks.

The problem is composition. MINIWEBGEN trains mostly on one-primitive episodes with instructions that explicitly name the skill, while real tasks chain several primitives under a single high-level request. The trained single agent therefore often stops before reaching the skills it learned. Supp. J analyzes this further. MINIWEBAGENT addresses both mismatches with a planner: an untrained planner decomposes the task into primitive-level instructions, and the trained executor carries out each one in the setting it practiced.

![](images/e772afb6f1b06db5478cb5ae0ad30a3b5c3c66753654183bb22cdd5dc37fc0d6.jpg)  
Figure 7. Practice lifts every skill family. Success per skill family on the 103 generated single-primitive tasks of the held-out sites, for a single Qwen3.5-4B agent untrained (grey) and trained on our data (blue), with the gain at the right; reasoning , which is not trained, is left out.

## 4.3. Composing trained skills raises task success on unseen websites and benchmarks.

With a planner, the practiced primitives translate into endto-end task success (Tab. 3). Relative to the untrained single agent, MINIWEBAGENT gains 19.4 points on held-out MINIWEB sites and improves by 9.3, 20.0, and 16.1 points on WebArena-Lite, WebVoyager, and Online-Mind2Web, respectively, without training on target-benchmark trajectories.

The planner and trained executor contribute differently. Planning alone gives its largest gain on WebVoyager, where long information-seeking tasks benefit from decomposition, but helps little elsewhere. Adding the trained executor consistently improves over the untrained planner across all four evaluations. On held-out MINIWEB tasks, MINIWEBAGENT also raises primitive success from 49.8% to 63.7%, with gains across every skill family. As expected from compounding primitive failures, the advantage grows on longer tasks: for tasks with three or more primitives, MINIWEBAGENT more than doubles the success of the untrained single agent (33.3% vs. 14.5%) (Fig. 6, right).

## 4.4. At matched size, our skill data matches or beats task-centric data.

Does the gain come from skill-specific practice, or simply from training on more web trajectories? To test this, we reimplemented InSTA’s trajectory collection pipeline on MINIWEB [27] to collect a size-matched corpus of taskcentric trajectories of MINIWEB, which, like our skill data, Qwen3.5-4B writes. We then train the same executor on such corpus. The task-centric executor does not significantly improve over the untrained planner on any of the four benchmarks, while our skill-trained executor does on all four (Tab. 3). Head to head, our executor is ahead everywhere, and the gap is significant on Online-Mind2Web (+10.7 points, $p < 0 . 0 0 1 )$ and held-out MINIWEB (+10.2, $p = 0 . 0 2 7 \mathrm { ; }$ ; Holm $p = 0 . 0 8 1 )$ . At this size, skill-centric data is at least as useful as task-centric data, and it is the one that reliably adds to planning.

## 5. Conclusion

In this paper, we introduce MINIWEB, a synthetic benchmark centered on 37 interaction primitives across 65 locally hosted websites, providing per-primitive evaluation and skill-segmented data generation for web agents. Its skill profiles show that drag & gesture is the weakest skill family of every agent we tested. Using this infrastructure, MINIWEBGEN expands limited seed demonstrations into 3.1K verified trajectories for targeted skill practice. Finetuning a compact base model on them improves the execution of single primitives, but on its own it does not raise success on multi-step tasks and lowers it on external benchmarks, because the model learns single primitives from single-primitive episodes, not how to chain them: it stops after one primitive, and removing that stop alone does not help. MINIWEBAGENT separates task planning from practiced skill execution; without training on any targetbenchmark trajectories, it improves over the base model on the held-out sites, WebArena-Lite, WebVoyager, and Online-Mind2Web, and its trained executor adds to what planning alone achieves, most on the held-out sites and Online-Mind2Web. Our findings suggest that targeted skill practice on synthetic websites, combined with an explicit separation of planning and execution, helps compact openweight agents on websites they were not trained on.

## 6. Limitations & Future Works

We acknowledge several limitations in our study. First, MINIWEB contains fewer human-authored tasks than Mind2Web [6] and WebArena [39]. Although our annotation interface (Supp. D) streamlines task creation and trajectory recording, devising diverse tasks still requires substantial effort and creativity. MINIWEBGEN alleviates training-data scarcity but does not replace broader humanauthored evaluation. Second, although parameterized verifiers reduce grading costs, running closed-source frontier agents remains expensive. We therefore leave these agents unevaluated, reflecting both resource constraints and our focus on small open-weight models. Third, we do not fully resolve the drag & gesture deficiencies identified by our benchmark. Our focus is on showing that improvements across skill families, when composed, can improve task performance on unseen websites, rather than achieving mastery of any individual skill. We leave specialized methods for mastering drag & gesture interactions to future work. Finally, this study does not include scaling experiments due to compute and evaluation costs. Future work should investigate how skill-centric training performs with larger datasets, greater website diversity, and larger models.

## 7. Acknowledgements

We sincerely thank all annotators who dedicated their time to contributing to and reviewing this benchmark. We especially thank Omar Rizman, an undergraduate researcher at the University of Utah, as well as members of the Spark Lab, the Utah NLP group, and the LUNA Lab at the University of Utah for their valuable feedback and review. This material is based upon work supported by the National Science Foundation under Grant No. 2535195

## References

[1] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent S: An Open Agentic Framework that Uses Computers Like a Human. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.08164. 3

[2] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents. 2025. URL https:// openreview.net/forum?id=zg5is4GJ3R. 3

[3] Jacob Andreas, Dan Klein, and Sergey Levine. Modular multitask reinforcement learning with policy sketches. In Doina Precup and Yee Whye Teh, editors, Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 166–175. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/ andreas17a.html. 2

[4] Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-7b: An efficient agentic model for computer use. ArXiv, abs/2511.19663, 2025. URL https://api.semanticscholar.org/ CorpusID:283251303. 1, 6, 31

[5] Ahmed Awadallah, Sahil Gupta, Yash Lara, Yadong Lu, Hussein Mozannar, Akshay Uttama Nambi, Zach Nussbaum, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Luiz Lemos do Valle, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-1.5: Scalable learning environments for computer use agents. ArXiv, abs/2606.20785, 2026. URL https://api.semanticscholar.org/ CorpusID:289622461. 1, 2, 6

[6] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023. 2, 3, 4, 8

[7] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. WorkArena: How capable are web agents at solving common knowledge work tasks? 235: 11642–11662, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ drouin24a.html. 2

[8] Lutfi Eren Erdogan, Hiroki Furuta, Sehoon Kim, Nicholas Lee, Suhong Moon, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-act: Improving planning of agents for long-horizon tasks. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id= ybA4EcMmUZ. 3

[9] Alexander Gill, Md Farhan Ishmam, Xuyen Nguyen, Neha Bhat, Parker Henry DeYoung, Fateme Hashemi Chaleshtori, Nathan Stringham, Kenneth Marino, and Ana Marasovic. The hard part comes ´ after search: Benchmarking web agents on synthesizing, organizing, and displaying knowledge. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2026, 2026. URL https://arxiv.org/abs/2609.30604. 2, 3

[10] Boyu Gou, Demi Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. In International Conference on Learning Representations, volume 2025, pages 30851–30883, 2025. 3

[11] Tianchen Guan, Daisy Xinlei Lin, Royce Cheng-Yue, Xiangjun Wang, and Shuyan Zhou. Componentbench: Diagnosing component-level failures in computer-use agents. 2026. URL https://openreview.net/forum?id= FFzncfguJN. 2

[12] Tanmay Gupta, Piper Wolters, Zixian Ma, Petr Sushko, Rock Yuren Pang, Diego Llanes, Yue Yang, Taira Anderson, Bo Zheng, Zhongzheng Ren, H. Trivedi, Taylor Blanton, Caleb Ouellette, Winson Han, Ali Farhadi, and Ranjay Krishna. Molmoweb: Open visual web agent and open data for the open web. ArXiv, abs/2604.08516, 2026. URL https://api.semanticscholar.org/ CorpusID:287256650. 1, 3, 6, 31

[13] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu,

Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. pages 6864–6890, 2024. 2, 3, 4

[14] Md Farhan Ishmam and Kenneth Marino. TimeWarp: Evaluating web agents by revisiting the past. In Fortieth Conference on Neural Information Processing Systems Evaluations and Datasets Track, 2026. URL https://arxiv.org/abs/2603.04949. To appear. 2, 5

[15] Lawrence Keunho Jang, Andrew Keunwoo Jang, Jing Yu Koh, and Ruslan Salakhutdinov. Mypcbench: A benchmark for personally intelligent computer-use agents. ArXiv, abs/2606.16748, 2026. URL https://api.semanticscholar.org/ CorpusID:289299570. 2, 3

[16] Lawrence Keunho Jang, Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Odysseys: Benchmarking web agents on realistic long horizon tasks. In Third Conference on Language Modeling, 2026. URL https://openreview.net/forum?id= oGWqU9VHQ6. 2

[17] Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. 2018. URL https://arxiv.org/abs/2410.08164. 2

[18] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: an analytical evaluation board of multi-turn llm agents. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385. 2

[19] Shikhar Murty, , Hao Zhu, Dzmitry Bahdanau, and Christopher D Manning. Nnetnav: Unsupervised learning of browser agents through environment interaction in the wild. arXiv preprint arXiv:2410.02907, 2025. 3

[20] Magnus Müller and Gregor Žunic. Browser use:ˇ Enable ai to control your browser, 2024. URL https://github.com/browseruse/browser-use. 6

[21] Tianyue Ou, Frank F Xu, Aman Madaan, Jiarui Liu, Robert Lo, Abishek Sridhar, Sudipta Sengupta, Dan Roth, Graham Neubig, and Shuyan Zhou. Synatra: Turning indirect knowledge into direct demonstrations for digital agents at scale. volume 37, pages 91618–91652, 2024. 3

[22] Yi-Chen Pan, De-Han Kong, Si-Da Zhou, Cheng Cui, Yifei Leng, Bingqian Jiang, Hangyu Liu, Yanyi

Shang, Shu-Yan Zhou, Tongshuang Wu, and Zhengyang Wu. Webcanvas: Benchmarking web agents in online environments. ArXiv, abs/2406.12373, 2024. URL https://api.semanticscholar.org/ CorpusID:270562249. 2

[23] Julien Piet, Annabella Chow, Yiwei Hou, Muxi Lyu, Sylvie Venuto, Jinhao Zhu, Raluca Ada Popa, and David Wagner. Web agents should adopt the plan-then-execute paradigm. arXiv preprint arXiv:2605.14290, 2026. 3

[24] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haolin Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. Ui-tars: Pioneering automated gui interaction with native agents. ArXiv, abs/2501.12326, 2025. URL https://api.semanticscholar.org/ CorpusID:275788034. 3, 6

[25] Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In Doina Precup and Yee Whye Teh, editors, Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3135–3144. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/ shi17a.html. 2

[26] Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, and Zhiyong Wu. OS-genesis: Automating GUI agent trajectory construction via reverse task synthesis. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5555–5579, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.277. URL https://aclanthology.org/2025.acllong.277/. 3

[27] Brandon Trabucco, Gunnar Sigurdsson, Robinson Piramuthu, and Ruslan Salakhutdinov. Insta: Towards internet-scale training for agents, 2025. URL

https://arxiv.org/abs/2502.06776. 2, 3, 6, 8, 35

[28] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Bo Li, Chen Dun, Chong Liu, Fuxing Leng, Han rui Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Pei-Xiong Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiao jun Qin, Yi Lin, Youbing Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, YuanFang Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Hao Zhou, Jinlin Pang, Li Han, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Taoran Lu, Woyu Lin, Xiao Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zheng-Wang Jiang, Zili Li, Zi-Hao Zhao, Chenxi Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Jun-Yan Du, Kai Cai, Kuanye Li, Lichen Yuan, Mei Han, Minchao Wang, Shuyu Guo, Tianhao Cheng, Xiaobo Ma, Xiao Xiao, Xiaolong Huang, Xinjie Chen, Yi-Zhen Du, Yi ling Chen, Yiwen Wang, Zhaojian Li, Zhen Yang, Zhiyuan Zeng, Chaolin Jin, Chen Li, Haolin Chen, Haolin Chen, Jian Chen, Qinghao Zhao ..., and Guang Shi. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. ArXiv, abs/2509.02544, 2025. URL https://api.semanticscholar.org/ CorpusID:281079943. 3

[29] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Haotian Yao, Ziwei Chen, Qizheng Gu, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y. Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents, 2025. URL https://arxiv.org/abs/2508.09123. 6

[30] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=NTAhi2JEEE. 3

[31] Yifan Wu, Yiran Peng, Yiyu Chen, Jianhao Ruan, Zijie Zhuang, Cheng Yang, Jiayi Zhang, Man Chen, Yenchi Tseng, Zhaoyang Yu, Liang Chen, Yuyao Zhai, Bang Liu, Chenglin Wu, and Yuyu Luo. AutoWebWorld: Synthesizing infinite verifiable web environments via finite state machines. 306: 136647–136675, 06–11 Jul 2026. URL https://proceedings.mlr.press/v306/ wu26ar.html. 1, 2

[32] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, and Yu Qiao. Os-atlas: A foundation action model for generalist gui agents. ArXiv, abs/2410.23218, 2024. URL https://api.semanticscholar.org/ CorpusID:273696039. 3

[33] Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao Yu. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials. In International Conference on Learning Representations, volume 2025, pages 79822–79843, 2025. 3

[34] Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. 2025. URL https://openreview. net/forum?id=6jZi4HSs6o. 2, 4, 31

[35] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022. 2, 3, 4

[36] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. GPT-4V(ision) is a generalist web agent, if grounded. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 61349–61385. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ zheng24e.html. 3

[37] Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills. 2025. URL https://arxiv.org/abs/2504.07079. 3

[38] Shuyan Zhou. Webarena-infinity: Generating browser environments with verifiable tasks at scale. shuyanzhou.com, March 2026. URL https:

//webarena.dev/webarena-infinity/. 2, 4

[39] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. 2024:15585–15606, 2024. 2, 3, 4, 8

## Supplementary Material

The supplement follows the paper: the benchmark (Supp. A–E), the agents and their training (Supp. F–G), the evaluation (Supp. H–I), additional results (Supp. J), and the prompts (Supp. K).

## Contents

A Mining candidate macros from web-agent benchmarks 14   
A.1 What is a macro? 14   
A.2 Macro bank creation 14   
A.3 Mining statistics . 15   
A.4 From candidates to primitives . 16   
B Interaction primitives and reasoning operations 16   
C Websites 18   
D Annotation interface 26   
E Grader validation 26   
F Observation and action spaces 29   
G Other architecture choices: family specialists 31   
H Live-web evaluation sets 31   
I Compute and cost 32   
J Additional results and details 32   
J.1 All configurations 32   
J.2 Statistical tests 32   
J.3 Why practice alone does not raise task success . 33   
J.4 Stop-early control . 33   
J.5 WebArena-Lite by site 33   
J.6 Model calls 33   
J.7 Robustness of the skill profiles 33   
J.8 Training and data details 35   
K Prompts 35   
K.1 Macro induction . 35   
K.2 MINIWEBGEN: guide summarization . 38   
K.3 MINIWEBGEN: instruction suggestion 39   
K.4 MINIWEBGEN: executor 40   
K.5 Macro judge . . 42   
K.6 MINIWEBAGENT: planner 44   
K.7 MINIWEBAGENT: executor 46

## A. Mining candidate macros from web-agent benchmarks

This section expands the first step of Sec. 3.1: it defines a macro (Supp. A.1), describes how candidate macros were mined from the tasks of four web-agent benchmarks (Supp. A.2), reports what the mining found (Supp. A.3), and explains how the candidates became the 37 primitives (Supp. A.4).

## A.1. What is a macro?

We define a macro as the smallest unit of web interaction that carries a distinct, recurring intent. A macro occupies an intermediate level of granularity between an atomic action (click, keystroke, or scroll) and a task, a high-leve natural-language goal that may require extended planning to fulfil. A macro is short-horizon: it realizes a single coherent intent through a short, contiguous chain of atomic actions and requires no further decomposition into sub-goals. Crucially, macros are written site-agnostically.

Structurally, each macro is a pair of a verb and a modifier, combined in snake\_case as verb\_modifier. The verb denotes the primary goal of the interaction (e.g., navigate, search, extract, submit) and is drawn from a closed, human-curated vocabulary V; the modifier specifies how the target of that goal is defined or structured (e.g., by\_semantic, from\_free\_text, by\_attribute) and is drawn from a closed vocabulary M. A macro is therefore an element of a curated subset of V × M.

Definition 1 (Macro). Let A denote the space ofatomic web actions, and let a task be a sequence over A thatfulfils a naturallanguage goal on some website. A macro m = (v, µ), with v ∈ V and µ ∈ M, is a site-agnostic unit of web interaction satisfying:

1. Single intent. m expresses exactly one intent;

2. Verifiable terminal state. Executing m reaches a single terminal state whose success is checkable by a single evaluator; and

3. Atomic minimality. Decomposing m yields only atomic actions in A—that is, m cannot be split into a chain of smaller macros.

These conditions induce a strict granularity hierarchy, atomic action ≺ macro ≺ task: a task decomposes into a chain of one or more macros, whereas a macro decomposes into a chain of atomic actions but admits no intermediate decomposition into smaller macros.

## A.2. Macro bank creation

We construct the macro bank through a combination of LLM-driven induction and human supervision. The procedure firs elicits a prior over the macro space by mining the tasks of four existing web-agent benchmarks (WebArena, WebVoyager, Mind2Web, and WebShop), and then extends this seed bank to the domains of MINIWEB. The induction is orchestrated by prompting Claude Opus 4.8 to operate over a relational store with four tables: a verb bank V, a modifier bank M, a macro bank, and the pooled task set. The process proceeds in three stages.

First pass: reference-bank construction. We pool the tasks from all four datasets into a single corpus and process them iteratively. For each task, the agent reasons about the sub-goals required to accomplish it and proposes the macros, and where necessary the constituent verbs and modifiers, needed to express those sub-goals. When an existing entry in V or M suffices, the agent reuses it; otherwise it proposes a new verb or modifier and marks it as proposed. Throughout, the agent consults the current state of the bank before each addition to avoid introducing redundant entries.

Deduplication. Because the first pass admits proposed verbs and modifiers greedily, the resulting vocabularies contain near-duplicates. The agent therefore reviews V and M and flags candidate duplications, which are then adjudicated by a human annotator who either merges each entry into an existing verb or modifier or keeps it as distinct. This step yields the curated vocabularies that fix the macro bank.

Second pass: task–macro assignment. The macro assignments produced during the first pass are confounded by the order in which tasks were processed, since the bank was still growing as tasks were seen. To remove this dependence, we discard the first-pass assignments, fix the curated bank from the deduplication step, and prompt the agent to reassign macros to every task from scratch, disregarding the prior assignments. Fixing the reference bank ensures that each task is labeled against the same macro vocabulary, independent of its position in the corpus.

This procedure yields a seeded macro bank grounded in existing benchmarks. We then assign the seeded macros to each website domain of MINIWEB and brainstorm additional macros, again with LLM assistance under human review, to cover intents that the four source datasets do not exercise.

Supp. K.1 gives the orchestrator’s full prompt.

## A.3. Mining statistics

The final labeling covers 15,727 tasks: all 812 of WebArena, 12,251 WebShop instructions, all 643 of WebVoyager, and 2,021 of Mind2Web’s 2,350, with 52,207 macro labels in total (Tab. 4). The labels use 136 distinct candidate macros, built from 18 verbs and 42 modifiers; 6 WebShop entries whose instruction states no task carry the label OFF\_VOCAB instead. A task needs 2.2 to 3.5 macros on average, depending on the benchmark. Most candidates recur across benchmarks: 22 appear in one benchmark only, 13 of them in Mind2Web, whose tasks span the most websites.

<table><tr><td>Benchmark</td><td>Tasks</td><td>Labels</td><td>Labels per task</td><td>Candidates used</td><td>Only in this benchmark</td></tr><tr><td>WebArena</td><td>812</td><td>1,813</td><td>2.23</td><td>99</td><td>5</td></tr><tr><td>WebShop</td><td>12,251</td><td>42,859</td><td>3.50</td><td>36</td><td>0</td></tr><tr><td>WebVoyager</td><td>643</td><td>1,865</td><td>2.90</td><td>85</td><td>4</td></tr><tr><td>Mind2Web</td><td>2,021</td><td>5,670</td><td>2.81</td><td>124</td><td>13</td></tr><tr><td>All</td><td>15,727</td><td>52,207</td><td>3.32</td><td>136</td><td>22</td></tr></table>

Table 4. The mined tasks. Tasks labeled per benchmark, their macro labels, and the candidate macros they use. Only in this benchmark: candidates that no other benchmark’s tasks use.

What the benchmarks exercise. Tab. 5 gives the share of labels per verb. WebShop contributes 78% of the tasks, nearly all of them product search and option choice, so the pooled shares are dominated by selecting and searching; without WebShop, searching, filtering, and extracting information account for 61% of the labels. The ten most frequent candidates are search\_by\_keyword (12,919), filter\_by\_attribute (6,037), select\_to\_purchase (3,880), add\_to\_cart (3,871), add\_to\_purchase (3,622), select\_by\_color (2,993), select\_by\_size (2,389), select\_by\_attribute (2,332), submit\_to\_purchase (1,477), select\_by\_variant (1,454). No label names dragging, drawing, uploading a file, setting a slider, signing, or editing a spreadsheet cell. The closest are the 157 price filters (filter\_by\_price), which a slider sets (Supp. A.4), and only 28 labels write a file.

<table><tr><td>Verb</td><td>Labels (%)</td><td>Without WebShop (%)</td><td>Candidates</td><td>Most frequent candidate</td></tr><tr><td>select</td><td>31.3</td><td>9.4</td><td>27</td><td>select_to_purchase</td></tr><tr><td>search</td><td>26.4</td><td>16.5</td><td>5</td><td>search_by_keyword</td></tr><tr><td>filter</td><td>17.1</td><td>27.8</td><td>20</td><td>filter_by_attribute</td></tr><tr><td>add</td><td>14.6</td><td>3.1</td><td>8</td><td>add_to_cart</td></tr><tr><td>submit</td><td>4.4</td><td>8.4</td><td>23</td><td>submit_to_purchase</td></tr><tr><td>extract</td><td>3.0</td><td>16.9</td><td>14</td><td>extract_by_question_answer</td></tr><tr><td>navigate</td><td>1.9</td><td>10.8</td><td>11</td><td>navigate_by_category</td></tr><tr><td>create</td><td>0.5</td><td>2.6</td><td>7</td><td>create_by_form</td></tr><tr><td>edit</td><td>0.2</td><td>1.3</td><td>3</td><td>edit_by_form</td></tr><tr><td>compute</td><td>0.2</td><td>1.1</td><td>3</td><td>compute_by_quantity</td></tr><tr><td>react</td><td>0.1</td><td>0.6</td><td>3</td><td>react_to_post</td></tr><tr><td>compare</td><td>0.1</td><td>0.6</td><td>2</td><td>compare_by_attribute</td></tr><tr><td>play</td><td>0.0</td><td>0.3</td><td>4</td><td>play_media</td></tr><tr><td>share</td><td>0.0</td><td>0.2</td><td>1</td><td>share_to_post</td></tr><tr><td>export</td><td>0.0</td><td>0.2</td><td>1</td><td>export_to_file</td></tr><tr><td>delete</td><td>0.0</td><td>0.1</td><td>2</td><td>delete_from_table</td></tr><tr><td>verify</td><td>0.0</td><td>0.1</td><td>1</td><td>verify_by_category</td></tr><tr><td>follow</td><td>0.0</td><td>0.1</td><td>1</td><td>follow_to_follow</td></tr></table>

Table 5. Labels per verb. Share of the mined labels by the verb of their macro, over all four benchmarks and without WebShop; the number of candidate macros with that verb; and the most frequent of them.

## A.4. From candidates to primitives

The candidates describe what existing tasks ask for at the granularity of their instructions, so many differ only in the value they target. In group brainstorming sessions we merged candidates that perform the same interaction on different values (e.g., the 27 select candidates, such as select\_by\_color and select\_by\_size, are choices from a set of options), removed site-specific ones, and separated reasoning from the physical interaction: the extract, compute, compare, and verify candidates correspond to the reasoning operations of Supp. B, which are attached to the interaction they follow. We then added interactions the benchmarks rarely or never exercise, such as dragging a slider, drawing a signature, uploading a file, and editing a spreadsheet cell. This gave the initial set of 46 primitives in 7 skill families (Sec. 3.1). After annotation, we kept the 37 primitives that annotators used at least 3 times (Supp. B).

Mapping the candidates to the final primitives. To measure how often the benchmarks need each final primitive, we map each of the 136 candidates to the final primitive that performs its interaction, or to none. Gemini 3.6 Flash proposes each mapping from the candidate’s name, its number of labels, and three example tasks from the benchmarks that use it most, and one of the authors checks every candidate. The author keeps 131 of the 136 mappings (96.3%, κ = 0.96; 99.7% of the labels). The 5 changes cover 176 labels: posts and replies move from the free-text primitives to create\_by\_form, comparisons of on-page values from compare\_by\_form to report\_information, and a search form to sort\_by\_form. No candidate maps to none. The price filter is the only candidate in the drag & gesture family, so dragging accounts for 0.3% of the mined labels (at most 2.1%, in Mind2Web), against 7.6% of the primitive instances in MINIWEBGEN’s training episodes. Tab. 1 (top) shows which primitives each dataset exercises. How evenly it exercises them is the effective number of primitives, the exponential of the entropy of the primitive shares, which equals the number of primitives in use when all are used equally. Excluding the reasoning primitive, which every information task ends with, the benchmarks’ labels spread over 3.4 (WebShop) to 9.6 (WebArena) of the 36 primitives and MINIWEBGEN’s trajectories over 32.8; on the 25 primitives that some benchmark uses, they still spread over 23.1.

## B. Interaction primitives and reasoning operations

Every primitive instance in a MINIWEB task carries two tags: a base primitive, the physical interaction (e.g., filter\_by\_slider), and an optional reasoning operation the agent applies to what it sees (e.g., extremum). An answer the agent must return to the user is tagged with the operation-only primitive report\_information; reasoning that only serves a later step is attached to the primitive it belongs to rather than tagged on its own. Primitives are called macros in our code and in the induction pipeline of Supp. A.

A single registry defines each primitive by a description, a span (the action where an instance starts and the one where i ends in a demonstration, used to cut demonstrations into segments), and a rubric for the VLM judge (Supp. K). The seven skil families group primitives by the control they operate; the family-specialist variant of MINIWEBAGENT trains one specialist per group of families: drag & gesture, discrete selection, out-of-page I/O, reasoning, and one for the rest (text entry, form transactions, navigation; Sec. 3.3). Tab. 7 lists the 37 primitives of the final set with their coverage in the human tasks and the generated pool, and Tab. 6 the reasoning operations. The table caption names the 9 hyper-specific primitives dropped in the final refinement (Sec. 3.1). Operations with fewer than 15 instances are omitted from the skill profiles (Tab. 2); the caption of Tab. 6 gives their counts.

<table><tr><td>Operation</td><td>Definition</td><td>Check</td><td>Instances</td></tr><tr><td>read</td><td>the info is present on the page — locate and read it</td><td>answer appears verbatim on a visited page</td><td>100</td></tr><tr><td>extremum</td><td>get the max/min among on-page items without using a sort control</td><td>chosen item is the argmax/argmin of the candidate set</td><td>41</td></tr><tr><td>count</td><td>count the on-page items that match a condition</td><td>count matches the true number of matching items</td><td>36</td></tr><tr><td>verify</td><td>compare a value on the page against a value given in the instruction</td><td>the on-page value satisfies the instruction&#x27;s stated value/condition</td><td>21</td></tr></table>

Table 6. Reasoning operations attached to instances of the 37 primitives in the human tasks. A primitive without an operation is a pure interaction. Not listed (fewer than 15 instances): compute (7), spatial (4), compare (3).

Table 7. The 37 interaction primitives, by skill family. Tasks: instances in the 369 human tasks; Sites: sites those tasks use; Gen.: trajectories generated by MINIWEBGEN in the training pool (2,687; the pool’s other 441 are of dropped primitives, and report\_information is learned from the base model’s own successes instead). The span marks where an instance starts and ends in a demonstration. Dropped in the final refinement (fewer than 3 human-task instances): reposition\_by\_drag (2), save\_by\_form (2), search\_by\_pan\_zoom (2), reveal\_by\_2fa (1), search\_by\_playback (1), carry\_info\_cross\_site (0), count\_entries (0), edit\_by\_ranking (0), edit\_by\_textbox (0).
<table><tr><td>Primitive</td><td>Description</td><td>Tasks</td><td>Sites</td><td>Gen.</td></tr><tr><td>Drag &amp; gesture</td><td></td><td></td><td></td><td></td></tr><tr><td>filter_by_slider</td><td>Narrow results with a range slider. Span: first drag → filter applied</td><td>11</td><td>9</td><td>74 100</td></tr><tr><td>sign_by_freeformdrawing</td><td>Sign a document by DRAWING your signature freeform on a canvas (“Sign here" field). Span: start drawing in the signature field → signature adopted/submitted</td><td>6</td><td>4</td><td></td></tr><tr><td>Discrete selection</td><td></td><td></td><td></td><td></td></tr><tr><td>toggle_relationship</td><td>Locate a user/post/item, then toggle a relationship (follow / subscribe / save / join-group / block). Span: locate the target → toggle clicked</td><td>46</td><td>28</td><td>97</td></tr><tr><td>filter_by_dropdown</td><td>Narrow results by selecting a dropdown value (incl. category search). Span: open the dropdown → option selected / results update</td><td>41</td><td>28</td><td>97</td></tr><tr><td>sort_by_form</td><td>Sort a listing by operating a sort control. Span: first sort-control interaction → state change (sorted view)</td><td>35</td><td>26</td><td>91</td></tr><tr><td>filter_by_date_range</td><td>Narrow results to a date range. Span: first date input → last date / submit</td><td>24</td><td>22</td><td>100</td></tr><tr><td>filter_by_options</td><td>Narrow results by clicking option fields (chips / checkboxes). Span: first option → last option / filter applied</td><td>21</td><td>16</td><td>68</td></tr><tr><td>delete_from_table</td><td>Delete an item from a list/table. Span: delete clicked → confirmed</td><td>17</td><td>15</td><td>99</td></tr><tr><td>play_by_playback</td><td>Operate a media player (play/pause/seek; + at timestamp/quality/subtitle). Span: open media page → play clicked</td><td>14</td><td>8</td><td>8</td></tr><tr><td>feedback_by_react</td><td>React to content (like / upvote). Span: react control → reacted</td><td>10</td><td>6</td><td>100</td></tr><tr><td>feedback_by_star Text entry</td><td>Give a star/slider rating. Span: first star/drag → rating set</td><td>5</td><td>4</td><td>82</td></tr><tr><td>search</td><td>Find content by typing a text/natural-language query. Span: click search box → enter /</td><td>71</td><td>43</td><td>100</td></tr><tr><td>message_from_free_text</td><td>submit Send a message by typing text. Span: open the message field → send clicked</td><td>20</td><td>11</td><td>85</td></tr><tr><td>edit_by_cell</td><td>Enter or change data directly in a grid/table cell (incl. adding a row or column to extend the table). Span: focus a grid cell → cell value committed (save or autosave)</td><td>8</td><td>5</td><td>32</td></tr><tr><td>translate_by_query</td><td>Translate entered text. Span: entering text → translation received</td><td>6</td><td>2</td><td>52</td></tr><tr><td>compute_by_tool</td><td>Use an on-page calculator/tool to compute a result. Span: open the tool → result shown</td><td>5</td><td>2</td><td>99</td></tr><tr><td>feedback_by_text</td><td>Leave a written review/comment (NEW). Span: open the review box → submit</td><td>5</td><td>3</td><td>62</td></tr><tr><td>write_executable_program</td><td>Write and run code. Span: open editor → run</td><td>4</td><td>2</td><td>59</td></tr><tr><td>sign_by_text</td><td>Sign a document by TYPING your name into the signature field (adopt a typed</td><td>3</td><td>2</td><td>100</td></tr><tr><td>Form transaction</td><td>signature). Span: focus the typed-signature field → signature adopted/submitted</td><td></td><td></td><td></td></tr><tr><td>create_by_form</td><td>Create/register/apply/invite/post a record by filling and submitting a form. Span: open</td><td>80</td><td>41</td><td>99</td></tr><tr><td>authenticate_by_form</td><td>the form → submit the form Log in by entering credentials. Span: open the form → submit</td><td>26</td><td>20</td><td>57</td></tr><tr><td>edit_by_form</td><td>Edit an existing record through a form. Span: open the form → close/submit the form</td><td>14</td><td>11</td><td>100</td></tr><tr><td>configure_by_form</td><td>Change one or more settings via a form. Span: open the form → submit/save</td><td>13</td><td>11</td><td>77</td></tr><tr><td>pay_by_form</td><td>Make a payment via a form. Span: open the form → submit the form</td><td>12</td><td>9</td><td>35</td></tr><tr><td>share_by_form</td><td>Share an item via a share form/dialog. Span: share is clicked → share form is submitted</td><td>8</td><td>7</td><td>100</td></tr><tr><td>book_by_form</td><td>Book/reserve by filling a form (incl. dates). Span: open the form → submit the form</td><td>7</td><td>5</td><td></td></tr><tr><td>checkout_by_form</td><td>Complete checkout (incl. promo/redeem) via a payment form. Span: open the form →</td><td>7</td><td>3</td><td>60 66</td></tr><tr><td></td><td>submit the form Cancel a booking or order via a form. Span: open the form → submit the form</td><td></td><td></td><td></td></tr><tr><td>cancel_by_form compare_by_form</td><td>Compare items by selecting them through form controls (e.g. dropdowns) that render a</td><td>6 3</td><td>5 3</td><td>11 100</td></tr><tr><td></td><td>side-by-side view. Span: open/enter the compare view → both items selected and the comparison is visible</td><td></td><td></td><td></td></tr><tr><td>Navigation</td><td>Go to a page by clicking a link/menu/tab. Span: link/menu click → page changed</td><td></td><td></td><td></td></tr><tr><td>navigate_by_route</td><td>Get directions by entering origin/destination. Span: first point typed → submit</td><td>61</td><td>40</td><td>99</td></tr><tr><td>get_nav_route</td><td>Join a meeting (by code or link). Span: open meeting → joined</td><td>5</td><td>2</td><td>56</td></tr><tr><td>join_meeting</td><td></td><td>5</td><td>1</td><td>11</td></tr><tr><td>Out-of-page I/O</td><td></td><td></td><td></td><td></td></tr><tr><td>upload_file</td><td>Upload a file via the file picker. Span: file picker opens → file attached Export data (choose a format and download). Span: export clicked → downloaded</td><td>26</td><td>18</td><td>68</td></tr><tr><td>export</td><td></td><td>15</td><td>11</td><td>43</td></tr><tr><td>copy_content</td><td>Copy content via a copy control. Span: copy clicked → copied</td><td>8</td><td>6</td><td>100</td></tr><tr><td>edit_by_image</td><td>Edit an image via canvas tools. Span: select tool → save</td><td>5</td><td>3</td><td>100</td></tr><tr><td>Primitive</td><td>Description</td><td>Tasks Sites Gen.</td><td></td><td></td></tr><tr><td>Reasoning report_information</td><td>Report the answer back to the human. Always the terminal tag — span the last action(s)</td><td></td><td></td><td></td></tr><tr><td></td><td>where the answer is on screen; carries the reasoning op. Intermediate reasoning folds onto its own macro instead. Span: the answer is located/surfaced on the page → the answer is reported</td><td>121</td><td>59</td><td></td></tr></table>

## C. Websites

MINIWEB has 65 websites in 15 domains, served offline by one local server. They share one fictional world (a city, Lakeport, WA, and a company, Meridian Systems, recur across sites), so one task can move between sites. Six hubs carry the effects of actions across sites: WebMail receives receipts, notifications and two-factor codes, SecureBank records payments and trades CalendarTodo receives bookings, MeridianCloud stores created files, QuickChat receives messages, and VaultGuard stores passwords. Twenty-one sites are seeded from public datasets; the other 44 hold synthetic data generated with fixed seeds. Thirteen sites are held out: no training data comes from them, and they are used only for evaluation (Sec. 4).

The gallery below shows the landing page of every site at the agents’ viewport (1280×800), grouped by domain. Each entry gives the site’s name and identifier, its domain, split, data source, the number of human tasks that use the site, and the number of primitives the site supports, followed by the description from the site’s documentation. The code release documents every page and API route of each site.

Tab. 8 lists the public sources and the licenses their publishers state; for a dataset taken from Kaggle, the license on its Kaggle page. Several sources carry share-alike or attribution duties, and some restrict use to research.

<table><tr><td>Site</td><td>Source</td><td>Stated license</td></tr><tr><td>academic-paper-db</td><td>arXiv metadata</td><td>CC01.0</td></tr><tr><td>flights-hotels</td><td>US airline routes and fares 1993–2024 (Kaggle; US DOT data); TBO hotels (Kaggle)</td><td>Apache 2.0; MIT</td></tr><tr><td>health-fitness-tracking</td><td>USDA FoodData Central; Fitbit data (Furberg et al., Zenodo)</td><td>CC0 1.0; CC BY 4.0</td></tr><tr><td>job-sites</td><td>Job Description Dataset (Kaggle; synthetic postings)</td><td>CCO: public domain</td></tr><tr><td>wikis</td><td>English Wikipedia</td><td>CC BY-SA 4.0</td></tr><tr><td>news</td><td>Wikinews</td><td>CC BY 2.5 / 4.0</td></tr><tr><td>dictionaries-language-tools</td><td>Wiktionary (via wiktextract)</td><td>CC BY-SA 4.0</td></tr><tr><td>qa-knowledge</td><td>Stack Exchange data dump (2024-04)</td><td>CC BY-SA 2.5–4.0</td></tr><tr><td>map-services</td><td>OpenStreetMap</td><td>ODbL 1.0</td></tr><tr><td>books-comics</td><td>Common Pile v0.1 (Pressbooks)</td><td>CC BY, CC BY-SA, CC0, or public domain, per book</td></tr><tr><td>project-mgmt-issue-tracking</td><td>Jira social repository</td><td>Apache 2.0</td></tr><tr><td>conference-review-submission</td><td>PeerRead (ICLR 2017)</td><td>reviews CC BY 4.0; papers: author</td></tr><tr><td>visual-how-to-guides</td><td>wikiHow via VisualHow</td><td>copyright CC BY-NC-SA 3.0, research use</td></tr><tr><td>email</td><td>Enron email corpus (CMU)</td><td>released publicly by CMU</td></tr><tr><td>e-commerce,auctions-p2p-</td><td>WebShop product data</td><td>none stated for the data</td></tr><tr><td>marketplaces forums,version-control</td><td>WebArena Reddit and GitLab environments</td><td>none stated for the data</td></tr><tr><td>software-marketplace</td><td>Google Play Store apps (Kaggle)</td><td>CC BY-SA 4.0</td></tr><tr><td>real-estate-buy-rent</td><td>USA Real Estate Dataset (Kaggle; realtor.com)</td><td>other (Kaggle description): educational</td></tr><tr><td></td><td></td><td>use</td></tr><tr><td>comparison-aggregators</td><td>GSMArena phone specifications (GitHub dump)</td><td>GPL-3.0 (repository)</td></tr></table>

Table 8. Public data sources of the 21 seeded sites and their stated licenses. The other 44 sites use only synthetic data.

A site’s task count includes every task that uses it, so the 13 held-out sites’ counts sum to 65: the 62 held-out tasks (those that start on a held-out site; 3 also use a training site) and 3 tasks that start on a training site and also use a held-out site, which neither split uses.

![](images/c5edefade6f4cf78b93abaded1c8dbc0ae6e0e61267a7e9ea9a985c026b697e0.jpg)

## Scholar Search

academic-paper-db · Search & reference · train · data: arXiv · 10 tasks · 7 primitives An academic paper search engine over arXiv metadata, modeled on Google Scholar, Semantic Scholar and arXiv.org. Users search and browse papers, open paper and author pages, compare two papers side by side, and keep a library of saved papers and followed authors.

![](images/2d4ce7ddb7cb750095b4aa034753f95dc66f9d2a043bba8a5c311ffd3d32357c.jpg)

## StepVista How-To Guides

visual-how-to-guides · Search & reference · train · data: WikiHow · 4 tasks · 9 primitives A wikiHow / Instructables-style site of illustrated step-by-step guides (home, cooking, tech and more). Users browse, filter and compare guides, step through them one step at a time, rate and comment, bookmark guides and follow authors.

![](images/629c0f46b3c1a4dc2b60544f337ae8b7e6fd485be864ab30f4971179fe75bc1c.jpg)

## MeridianFlow Docs

documentation-api-docs · Informational · train · synthetic data · 3 tasks · 4 primitives

A developer documentation site modeled after Stripe Docs, for the fictional MeridianFlow workflow API. It has guides, an API reference with method badges, a changelog, full-text search, and bookmarks for signed-in users.

![](images/e0f2aeceda312b812399a268b8a341d01ad040677ec5eb960439cd974fe62b50.jpg)

## PhoneCompare

comparison-aggregators · Search & reference held-out · data: GSMArena · 3 tasks · 8 primitives A GSMArena-style phone catalog and comparison site. Users filter and sort phones by specs, open spec sheets, compare two phones side by side, and keep a favorites list and a compare list.

![](images/5c8c35ed8899cf8cdcbaa6960f1a93e5463e18ce534a1d9fe53910f223cedf9c.jpg)

## LakeportWiki

wikis · Search & reference · train · data: Wikipedia · 4 tasks · 5 primitives

A Wikipedia/MediaWiki-style encyclopedia about the fictional city of Lakeport, WA and the Pacific Northwest, mixed with real Wikipedia articles. Users read, search, compare, create and edit articles and browse revision history, diffs and recent changes.

![](images/0bf418b2c70182d908f39cc13da23fb3ba83fd5b921a180f52426ce6e25aabee.jpg)

## Alex Rivera

personal-portfolio · Informational · train · synthetic data · 2 tasks · 6 primitives

The personal developer portfolio of Alex Rivera, the shared persona, in the style of a GitHub Pages / Vercel personal site. It shows a profile, project gallery, structured resume, skills table and blog links, with a contact form and site-wide search.

![](images/9c27cc2e6d0cba28567f3adbd7a930f4d211402a154e1f53c8d72f12c447830a.jpg)

## KnowledgeHub

qa-knowledge · Search & reference · train · data: StackExchange · 6 tasks · 11 primitives A Stack Overflow-style programming Q&A site built from a Stack Exchange data dump. Users search and browse questions by tag, read answer threads, vote, answer, comment, accept answers, save questions, follow tags, share and report content, and ask or edit their own questions.

![](images/da19a5bb920cbf99c490d27be2d66f87f604a58e5f337c0f0d44aad368454174.jpg)

## Apex Dynamics Corp

business-company · Informational · train · synthetic data · 5 tasks · 5 primitives A corporate marketing website for a fictional software company: products, services, team, blog, careers, contact form and newsletter. Its product line is the “Meridian” suite (MeridianFlow, MeridianVault and others).

![](images/d627b01883a4e47634379f96d8cf0a2d6f7981634c1d208b8d1d65eb3c21b793.jpg)

## FlowNet Project Homepage

project-homepages · Informational · train · synthetic data · 3 tasks · 4 primitives

An academic project landing page for the FlowNet paper (ICML 2025), in the style of nerfies.github.io and other paper homepages. It shows the paper sections, team, downloadable resources, news updates, statistics and citation export.

![](images/d3ac26572c950597291f051c778a0184b543e1f4ef21f32d0c690613113d9191.jpg)

## Meridian State University

university-academic · Informational · train · synthetic data · 6 tasks · 8 primitives A university website and student portal: course catalog, faculty directory, research areas, events, alumni network, admissions application and a “My Schedule” course registration view, plus an editable course gradebook.

![](images/6469227cb32b2c563a99b7edae913261ae31921da4a9eaa4a8b551bf3726e3ae.jpg)

## Lakeport Sports Hub

sports-esports · News & feeds · train · synthetic data · 6 tasks · 10 primitives

A multi-sport scoreboard in the style of ESPN / Flashscore covering the NFL, NBA, MLB, MLS, Premier League and a local esports league. It shows live final and scheduled matches, standings, team and player pages, a team comparison tool, match highlights, comments and a favorites list.

![](images/642356bcd23e372582ac90273b38615bcaa35e963474974d1b3eee6fff1a1a14.jpg)

## FundSpark

crowdfunding-donations · Shopping & transactions · held-out · synthetic data · 6 tasks · 12 primitives

A Kickstarter/GoFundMe-style crowdfunding site. Users browse and search campaigns, back a project through a reward-tier checkout, start their own campaign, post updates, and save, follow, subscribe to or share campaigns.

![](images/0a4b7a54d69d78bb9845d9a9d2d6c8976d8d75e84d3bf815644d0aafc8ae8161.jpg)

## TumblrVibe Blogging Platform

blogs · News & feeds · train · synthetic data · 8 tasks · 9 primitives

A Tumblr-style blogging platform with posts (articles, tips, stories, tutorials), comments, tags, likes and reblogs. Users browse and filter the feed, write posts, comment, follow authors, save posts and subscribe to tags.

![](images/2a2af018c4280353f3d418f787c19113d0e1c014fcf9855a6fc168c93c12b3bd.jpg)

## Lakeport Weather

weather · News & feeds · held-out · synthetic data · 5 tasks · 6 primitives

A local weather portal for the fictional city of Lakeport, WA and nearby Pacific Northwest stations, modeled on weather.gov and Weather Underground. It shows current conditions, forecasts, a multi-year history archive and active alerts, and lets a signed-in user keep a list of saved locations.

![](images/acc65b43ef31d82b1cd2ed614963221ce4fd88ddbc6fa72f6f668e1e0285eac5.jpg)

## ShopHub

e-commerce · Shopping & transactions · train · data: WebShop · 6 tasks · 13 primitives

An Amazon or eBay-style online store. Users search and filter a large product catalog, read and write reviews, keep a cart and a wishlist, check out, and cancel or return orders.

![](images/f2384d736a2635d633bb549d68127760c97e1c5852f9c8db560daa18448b60a4.jpg)

## Lakeport Tribune

news · News & feeds · train · data: Wikinews · 4 tasks · 9 primitives

A local newspaper site for the fictional city of Lakeport in Cascadia County, modeled on Patch.com and small-city news portals. Readers browse category pages, search articles, listen to articles, comment, bookmark, share, follow authors or categories, and report articles.

![](images/c044555b11e82cbea5da7002f5f31dd8e7e115bbd08ea140348d4ff6cb6a21dc.jpg)

## BidMarket

auctions-p2p-marketplaces · Shopping & transactions · train · data: WebShop · 5 tasks · 16 primitives

An eBay-style auction and peer-to-peer marketplace. Users browse and filter listings, bid or buy now, watch and save items, follow sellers, message sellers, and sell their own items.

![](images/8e21ef2adf8261ca30f55b12d25a2e6313278f6537fdebecaf7a9250be35b0b3.jpg)

## SkyLodge Travel

flights-hotels · Shopping & transactions · held-out · data: US flights, TBO hotels · 5 tasks · 13 primitives

A flight and hotel booking site in the style of Expedia / Kayak, covering major US cities. Users search and filter flights and hotels, book a flight through a multi-step flow (travelers, seat map, review, payment), book hotel stays, and view or cancel their bookings.

![](images/09df2b953f9925dade2cbabff104df77c5b5b76497c181a9b7ebfbf6156bf25f.jpg)

## Cascadia Insurance & Lending

insurance-loans · Shopping & transactions · train · synthetic data · 8 tasks · 14 primitives An insurance and lending customer portal in the style of Progressive or SoFi. Users review policies, claims, loans and payments, file claims, buy a policy or apply for a loan, sign the agreement, make payments and download policy documents.

![](images/08c80de6a915b17409080bf9086e2ab69e21d2b093d9f8453db0c2cb18a5c9ad.jpg)

![](images/76d44ac9e26e38b337db433c7174fec7596ce4607f10989db8495dbf4162e991.jpg)

## AppVault Software Marketplace

software-marketplace · Shopping & transactions · train · data: Google Play · 6 tasks · 10 primitives A Google Play-style app store. Users browse, search and filter apps, compare two apps, install and review them, keep a wishlist, buy paid apps through a cart and card checkout with promo codes, and manage purchases and refunds.

![](images/29ab27317efeb1455898738e549ee1867b41ced4833a127b9307f9512b13653e.jpg)

## TradeVista

brokerage · Finance · train · synthetic data · 8 tasks 12 primitives

A Robinhood-style trading app for stocks, index funds, crypto, options and futures. Users browse securities, view ticker charts, place and cancel orders, manage a watchlist and move cash in and out.

![](images/157f766b8e0f091680d4b3ff3a04706cd05a045b59f8ddef1ee829db671faffa.jpg)

## JobQuest

job-sites · Shopping & transactions · train · data: Kaggle Job Dataset · 4 tasks · 11 primitives An Indeed or LinkedIn Jobs-style job board and application tracker. Users search and filter job listings, save jobs, follow companies, apply with a resume and cover letter, track applications and manage job alerts.

![](images/358e28905b38091fd2fc31e7306116eecdcb49ec4f440557b5b6384dc4397b50.jpg)

## Lakeport Events

ticketing-events · Shopping & transactions · train · synthetic data · 8 tasks · 15 primitives An Eventbrite/Ticketmaster-style ticketing site for events in Lakeport. Users discover events in an infinite-scroll feed, filter and sort them, compare two events, buy tickets (with reserved seating and promo codes), save events and cancel orders.

![](images/c9f84a762279c76383db530b94c6f4e385375f61551cb69e00c1dc26ed42d244.jpg)

## FitTrack

health-fitness-tracking · Health · train · data: USDA, Zenodo · 8 tasks · 11 primitives

A personal health and fitness dashboard modeled on MyFitnessPal and Fitbit Web. Users log and review workouts, keep a daily food diary, track daily stats (steps, sleep, calories, water, weight) and manage fitness goals and targets.

![](images/28a65a536c8e6db361746fc2c70e4993b8939bcd61e2ba8efc8e6daf29907eb9.jpg)

## Lakeport Real Estate

real-estate-buy-rent · Shopping & transactions · held-out · data: Zillow, Realtor.com · 5 tasks · 10 primitives

A Zillow/Redfin-style home search site for Lakeport, WA. Users search and filter homes for sale or rent, save listings, contact agents, schedule tours and follow agents.

![](images/af2f831ae1012825ae8be75c7cafdf8e155bd075f06f6900ee935472f729ed99.jpg)

## SecureBank Online

banking · Finance · train · synthetic data · 4 tasks · 14 primitives

A retail online-banking portal in the style of Chase, TD or Amex. It covers deposit accounts, transaction history, transfers, bill pay, payees, loans and a credit-card section with statements, payments, rewards and disputes.

![](images/32fe987439abae0d39201f4e5613e56dbbe205888fe10920a692d0190a69ba28.jpg)  
Lakeport Medical Center Patient Portal

health-portals · Health · train · synthetic data · 8 tasks · 15 primitives

A MyChart-style patient portal for Lakeport Medical Center. Patients manage appointments, read medical records (vitals, labs, diagnoses), message their care team, request prescription refills, pay bills and e-sign consent forms.

![](images/ba5ef17cd5abb28e0dcc8680131652feb77c04c8adffb6e1c23e3ba0ecf53aea.jpg)

## AI Chatbots Hub

ai-chatbots · Communication · train · synthetic data · 10 tasks · 12 primitives

A ChatGPT-style chat app with three bot personas (Assistant, Creative, Analyst), a searchable knowledge base, an FAQ and a prompts library. Replies come from a local, deterministic rule engine (intent matching plus retrieval over the knowledge base and FAQ); no external LLM is ever called

![](images/6688da8ab2cdc565e6ebd3051f624a9372d8490eddb82bf015a70ac26e813189.jpg)

## QuickChat

instant-messaging · Communication · train · synthetic data · 12 tasks · 15 primitives

A WhatsApp Web / Telegram-style messenger: a dark split-pane view with the conversation list on the left and the chat on the right, plus a contacts page. The user is Alex Rivera, chatting with friends, family and a neighborhood group.

![](images/f0b4e0201b4edfcbe753547a3cbffb1f89c1625a29dd2b16b3ac89615447c339.jpg)

## ForumHub

forums · Social media · train · data: WebArena Reddit · 4 tasks · 14 primitives

A Reddit-style discussion forum. Users browse communities (subreddits), read and write posts and threaded comments, vote, save, share and report content, follow or block users, join communities, send direct messages and moderate the communities they run.

![](images/412f0f76a3a5fa9511c5fcdb87ccd33cbd953476488418de3f77b32eca4f7cf6.jpg)

## HeartLink

dating · Communication · held-out · synthetic data · 9 tasks · 15 primitives

A Tinder, Bumble or Hinge-style dating app. Users swipe through discovery cards, browse and filter the profile directory, like or pass, chat with matches and edit their own profile and preferences.

![](images/0a84a641474f6c4acb965065211e5385ce7a9bee51e525d0532713903ced8953.jpg)

## CallHub

remote-calls · Communication · train · synthetic data · 12 tasks · 15 primitives

A Zoom, Teams or Google Meet-style video-calling platform for the fictional company Meridian Systems. Users schedule and join meetings, watch recordings and read transcripts, review their call log, and chat inside a simulated call room

![](images/85504c46d3e15cb117e09a12a93b976e610bfa95881137d950bfc41633e4c525.jpg)

## PixShare

multimedia-posting · Social media · train · synthetic data · 10 tasks · 15 primitives

An Instagram/Twitter-style social network for photo, video and carousel posts. It has a feed of followed accounts, an explore page, stories, profiles, direct messages and the usual social actions (like, save, follow, block, share).

![](images/6ea9297bfff1e204992d41b78428ffea6d8a94e5f56e3216f84104e3d3b5ed51.jpg)

## WebMail

email · Communication · train · data: Enron · 21 task · 14 primitives

A Gmail/Outlook-style webmail client. Users read folders, search, compose with attachments, star, label, move, delete, block senders and report messages. It is also the inbox every other MINIWEB site writes to: order confirmations, notifications and 2FA codes all arrive here.

![](images/286f2d5d3c774046305032d6ef842268b04e98681d71e5b3009c6658c599f7bf.jpg)

## Meridian Chat

team-chat-workspace · Communication · train · synthetic data · 11 tasks · 14 primitives

A Slack or Microsoft Teams-style workspace for the employees of the fictional Meridian Systems. Users read and post in channels, reply in threads, react, save and share messages, send direct messages, upload files and manage channel membership.

![](images/cd9fa2adde68b9f44045520846109888e460a966dfeccfe1993c47b8cb920735.jpg)

## LakeReview

rating-review · Social media · held-out · synthetic data · 5 tasks · 9 primitives

A Yelp-style review site for local businesses in the fictional town of Lakeport, WA. Users browse and filter businesses, read and write star-rated reviews, vote on reviews, follow reviewers and save businesses.

![](images/16a65705f0af05e9ad694541908cf3f304235b06dc7d1e1234f3e6d33aaac319.jpg)

## BookVerse

books-comics · Streaming & media · train · data: Common Pile · 9 tasks · 10 primitives

A digital bookstore and reading app in the style of Kindle / Google Play Books. Users discover books by category, read chapters in an in-browser reader, review and rate titles, follow authors, subscribe to categories and buy books through a cart and checkout.

![](images/a76eb2a071c857e9ab2952a92ca8503fc49b8888d1e2947bda6b6f45a4030602.jpg)

## SoundShelf

podcasts-audiobooks · Streaming & media · train · synthetic data · 5 tasks · 11 primitives A podcast and audiobook app in the style of Apple Podcasts and Audible. Users discover shows, play episodes, subscribe to and follow podcasts, buy audiobooks into their library, like and save items, and write reviews.

![](images/1180af4bfd7e083026f9f199bf637e3b5241e3f8d83c842e64f90fe0e7b8bb7f.jpg)

## DocEdit

documents · Editing · train · synthetic data · 5 tasks · 10 primitives

A Google Docs-style document editor. Users create, edit, share, star, move and trash documents, browse folders, view and restore past revisions, and edit a document’s data grid cell by cell.

![](images/92f175dd74f45dd6ada80312e41e711bccf03ad2f5f45b2f1deb8286971f3e7e.jpg)

## PulseLive

live · Streaming & media · train · synthetic data · 4 tasks · 11 primitives

A Twitch-style live-streaming platform. Viewers browse live and past streams, watch with a seekable player and live chat, follow and subscribe to channels, cheer and redeem channel points, and browse clips.

![](images/4efd9547673f287a4d8507826e286de98cc788324024cda50413475a68935f2b.jpg)

## StreamHub

video · Streaming & media · train · synthetic data · 8 tasks · 14 primitives

A YouTube-style video sharing platform. Users browse trending and latest videos, search, filter by upload date, watch videos in the shared mini-player, like, comment, save, share and report videos, subscribe to or follow channels, manage playlists and upload new videos.

![](images/11a9c5c12f3a446df2d5a59941c2e2db8b28bd8ec5d501a71abbb06a273cedb3.jpg)

## NoteCanvas

handwritten-notes-whiteboards · Editing · train · synthetic data · 7 tasks · 10 primitives A notes and whiteboard app in the spirit of Apple Notes, Notion and Miro. Every note has a text body and a freeform drawing canvas. Whiteboards hold sticky notes, shapes and text that can be dragged around.

![](images/dc80d0d283119ff34afe1bf14ffa606a87ce47240cf31e90aa8706911008d2d0.jpg)

## SoundWave

music · Streaming & media · held-out · synthetic data · 5 tasks · 9 primitives

A Spotify-style music streaming site with a dark theme and a persistent bottom player. Users browse artists, albums and tracks, play music, like tracks and albums, follow and subscribe to artists, build playlists and share.

![](images/42a740879361744570d084886a65e863066f5e199543d7ee2eadfe3171277930.jpg)

## CodeRunner

code-editor-execution · Editing · train · synthetic data · 7 tasks · 7 primitives An online Python IDE modeled after the GeeksforGeeks IDE: a gallery of example snippets, an in-browser editor that really runs Python, and a dashboard of saved snippets.

![](images/4ab64e67b14334b9d852f1038e78122cfa4b4580d001ec01dc7e7e03ea03aeb0.jpg)

## SheetDeck

spreadsheets-slides · Editing · train · synthetic data · 8 tasks · 8 primitives

A Google Sheets / Google Slides-style workspace for a fictional company (Meridian Systems). Users browse files, edit spreadsheet cells in a grid, add and reorder slides, share files and export them.

![](images/a15848e840811ca177d209c7c98448174b382cba26bf7c6b69b959662fa45e70.jpg)

## CalendarTodo

calendar-todo · Productivity · train · synthetic data · 6 tasks · 11 primitives

A Google Calendar-style calendar and to-do app. Users browse events by week or day, filter them by category, priority, calendar owner or date range, and create, edit, share, invite to and RSVP to events.

![](images/fcba245a130a166e19c76ec2400432d68fc5d8ddd9ee2759582231502c5f3f04.jpg)

## SalesPro CRM

crm · Productivity · held-out · synthetic data · 7 tasks 11 primitives

A Salesforce/HubSpot-style CRM for a sales team: contacts, companies, a deal pipeline, logged activities and a follow-up task queue.

![](images/a9ac5de1dd6bfc1777248bcdb3fac2140c4d11a356ae1c1b08c70cee5f155c89.jpg)

## Meridian Tracker

project-mgmt-issue-tracking · Productivity · held-out · data: Jira · 5 tasks · 12 primitives

A Jira-style project tracker for the fictional Meridian Systems. It has per-project Kanban boards, issue detail pages with comments and status transitions, sprints, a backlog grid and issue export. Issue keys follow Jira style (MF-101).

![](images/c7565c20f2ac2b4b0d2d82ff902677ede9efd731c3212fa93dc04c83de5d36cd.jpg)

## CloudCore Console

cloud-dev-consoles · Productivity · train · synthetic data · 10 tasks · 12 primitives An AWS/GCP-style cloud management console. Users browse services, launch, edit, power-cycle and terminate instances, provision functions, databases, storage buckets and IAM users, and inspect billing, metrics, logs, alerts and API gateway endpoints.

![](images/908a2cd73c0844fdb5e592e8564e72b0f231123559da42c0d60d54b6a64e3e36.jpg)

## DesignFlow

design-creative · Productivity · train · synthetic data · 7 tasks · 14 primitives

A Canva-style design tool. Users browse a template gallery, start projects from templates, edit designs in a simplified canvas editor (positioned HTML elements, not a real canvas), manage an asset library, invite collaborators and export designs as SVG.

![](images/683acf925242f7654a1d27d43565492ea5c15d2e7197dae9f4aedd6b67e3789e.jpg)

## MeridianGit

version-control · Productivity · train · data: WebArena GitLab · 5 tasks · 9 primitives

A GitHub/GitLab-style code-hosting site for the fictiona Meridian Systems engineering team, mixed with real GitLab projects, issues and merge requests. Users browse repositories (code, commits, issues, merge requests), search code, create repositories, issues and merge requests, comment, star repos, upload files and merge or close merge requests.

![](images/3b9a461e6fcf1d167023117119dde4c1ebd8a68e37135adcd758c97da0cb7c91.jpg)

## MeridianCloud

cloud-storage-file-transfer · Productivity train · synthetic data · 7 tasks · 15 primitives A cloud-drive and file-transfer portal in the style of Google Drive / Dropbox, holding a software team’s workspace at Meridian Systems. Users browse a folder tree, search and filter files, star, rename, move, trash and download them, share files with teammates and send file transfers via share links.

![](images/ab9bc303959789894285aa0895025241c08750285cfbeccd1979b0f756def1ef.jpg)

## FormFlow

forms-surveys · Productivity · train · synthetic data · 4 tasks · 8 primitives

A Google Forms / SurveyMonkey-style form builder. Users create forms from scratch or from templates, fill them in, review results and analytics, edit responses in a grid, attach files and share links.

![](images/93ab8961748da6ef14121b0f8d92d3d79e1f0406d6881fded39d9e5cb93136ba.jpg)

## CascadiaMaps

map-services · Maps & navigation · train · data: OpenStreetMap · 8 tasks · 10 primitives

A Google Maps-style map and navigation service for Lakeport, WA. Users search places on an interactive Leaflet map, filter by category, rating and “open now”, open place pages with hours and reviews, save places, write reviews, get directions by travel mode and compare places.

![](images/7d99ceeaaed0292c468b23b4de2d68a8db34fe08030cd9826ed1dd818dda5855.jpg)

## Lakeport Transit Authority

transit-directions · Maps & navigation · train · synthetic data · 4 tasks · 5 primitives The public-transit website of the fictional Lakeport Transit Authority (LTA), in the style of Google Transit / TriMet. It lists bus routes and stops with timetables, plans trips with route preferences, explains fares and passes, and compares routes side by side.

![](images/47367860143b877614c56a45e75802b166269f00cdc12a1b0250885dfbefefd7.jpg)

## City of Lakeport

agency-portals · Government & civic · train · synthetic data · 10 tasks · 10 primitives A .gov-style municipal services portal for the fictional city of Lakeport, WA. Residents browse departments, services, permits, public records and announcements, and (signed in) apply for services, book appointments, pay city bills and upload documents.

![](images/25b0ff0026ac13f382fb030d3b4a0226c972ca6ab7093786bfc316d389f3425f.jpg)

## CalcTools

converters-calculators · Utilities · train · synthetic data · 5 tasks · 5 primitives

A free unit-converter and calculator site modeled on Calculator.net. It offers length, weight, temperature, currency, volume, area, speed and number-base converters plus BMI, mortgage and tip calculators. Signed-in users can save results and see their conversion history.

![](images/b6d37ba71f517af9e710c7360b0ac8e6c5781253a48f8b6e106887de1a106a53.jpg)

## PeerPortal

conference-review-submission · Education held-out · data: PeerRead · 2 tasks · 8 primitives An OpenReview-style venue and peer-review system. Visitors browse venues and papers with their reviews and scores. Signed-in reviewers bid on papers, write reviews and upload submissions, and chairs assign reviewers and post decisions.

![](images/aabdd209c80b7fcc96712114c9c125984d09dfaff3bd324f5ddde65ccd622473.jpg)

## Lakeport Civic Hub

petitions-voting-info · Government & civic · held-out · synthetic data · 4 tasks · 9 primitives A civic engagement portal for the fictional City of Lakeport, WA, combining Change.org-style petitions (create, sign, share, subscribe) with a county elections site (election results, voter registration, polling places).

![](images/a05de7fe94f7f2bdb744df279758e895b0a6b94b43bed8df4214e09a6f1e9577.jpg)

## WordRef Dictionary

dictionaries-language-tools · Utilities train · data: Wiktionary · 2 tasks · 6 primitives An English dictionary in the style of Merriam-Webster / Dictionary.com, built from Wiktionary entries. Users search words, browse A-Z, read definitions, IPA pronunciations, word forms, examples, synonyms and antonyms, and save words to a personal list.

![](images/42e3234c76b84a1ef97f9a4ab7d74c35ff4d10242b2e20af2513b68ed022690d.jpg)

## EduPortal LMS

course-sites-classrooms · Education · train synthetic data · 2 tasks · 8 primitives

A Canvas / Moodle-style learning management system. Students browse and enroll in courses, play lectures, submit assignments (text plus an optional file) and post in discussion boards. Instructors and admins also grade submissions and edit the gradebook. Access is role-based (admin / instructor / student).

![](images/ea52d80c9a1388a2e22649317a63b888b28b288a1e3d8ecd3e8fae98564c77d3.jpg)

## Lakeport Government Services

tax-filing-dmv-permits · Government & civic · train · synthetic data · 7 tasks · 14 primitives A state/city services portal combining tax filing, motor-vehicle (DMV) records, permits, payments and appointments, in the style of IRS and DMV online services. Users file a Form 1040, sign it, register vehicles, apply for permits, pay fees, book DMV appointments and track refunds.

![](images/48353d9d925197983c603c00f838bd40e2cabc31ddbf3b8697ba342c1dad6f17.jpg)

## VaultGuard

password-managers · Utilities · train · synthetic data · 5 tasks · 10 primitives

A password manager in the style of 1Password, LastPas and Bitwarden. Users browse vaults, search and filter entries, create, edit, delete, import and export entries, reveal a password with a PIN, generate passwords and review a security report and audit log.

A Bitly, TinyURL or Rebrandly-style link manager. Users shorten URLs with custom codes, expiry, tags, redirect types and QR codes, manage and export their links, and read per-link click statistics.

![](images/31c44845affb21c1e9f5bffdc910b5811b2eb42db9d1961d72c98a93791585c8.jpg)

## LinguaBridge Translate

![](images/de360c878a2ec1e14049baac28c2a1fd6ab5d187bc6c89297c2cdeb1a6368245.jpg)  
translation · Utilities · train · synthetic data · 8 tasks · 5 primitives A Google Translate / DeepL-style translator with source and target language dropdowns, translation history, saved translations, custom glossaries and user settings

## SnapLink URL Shortener

## D. Annotation interface

Six annotators wrote and recorded the 369 tasks in a web tool served next to the sites (Fig. 8). Its dashboard orders the work in three steps: review each site and report bugs, annotate tasks, and build their verifiers. It also shows the current coverage of every primitive, least covered first.

Annotating a task (Fig. 9). The tool samples one to three sites and a set of primitives to cover, weighted toward primitives with few tasks. The annotator first explores the sites freely, then designs the task as a graph in the editor below the browser: one node per primitive instance with its subtask and reasoning operation, and edges for the order and data flow between instances. Every primitive of a multi-primitive task must be connected. The annotator then writes the instruction in their own words, sets the starting login and two-factor state, and records the task in the embedded browser. After recording, they mark each instance’s span on the action timeline and enter the expected answer. Optional LLM helpers draft a task, rephrase an instruction, or infer a primitive chain from it; their output is only a suggestion. Annotators can also propose a new primitive, which is added to the registry for later review.

Building the verifier (Fig. 10). For each primitive instance the verifier builder shows the gold HTTP requests from the recording and a check scaffold: request gates, actions, visited pages, and an answer matcher. The annotator pins the values the instruction dictates, marks the remaining checks as advisory, and tests the verifier in a sandbox: it must pass the gold recording and should reject two negative controls, an empty trajectory and an answer given without any action.

Reviewing tasks. The macro browser (Fig. 8b) steps through every recorded span of one primitive across all tasks. Reviewers used it to make tags consistent across annotators. A task review page compares each task with its state before correction; a reviewer marks it verified or stale, asks for a re-recording, or deletes it.

## E. Grader validation

Every task is graded twice: by its parameterized verifiers, which check the backend request log, and by the VLM judge (Gemini 3.5 Flash), which reads each primitive’s rubric, the request log, and the screenshots (Sec. 3.1; the judge’s prompt is in Supp. K). We check both against human demonstrations and measure how often they agree on agent episodes.

Valid demonstrations. A human recording can stand in for a correct episode only if it is current and complete. Of the 369 recordings, we leave out the 67 tagged as stale, which predate a change to their site, data, or task, and the 119 that the task’s current verifier rejects, such as early recordings whose recorder missed the graded form submission or recordings made before the task was edited. The remaining 183 recordings pass the verifier by construction.

Positives and negatives. From each valid recording we build three cases: the recording with the annotator’s answer (should pass); a near miss, the same recording cut just before its last required primitive instance and without an answer (should fail); and an empty episode with no actions and no answer (should fail). Over these 549 cases, the verifiers are 98.9% accurate and the judge 98.2%. The verifiers pass every positive and the judge 97.8% of them; both reject every empty episode, and each accepts 6 of the 182 near misses. The judge’s reference answer comes from the same recording, so the positives are an easy case for it; the near misses are the informative part.

![](images/29db301defdc9910abb7fbc2b715f244ce9c5dd7ea63508b841ba3a9d7dd327c.jpg)  
(a)

![](images/1ad868e4487b4b0c3bdeda6973a435e9d4f51bdea9d96693334710ce20c09cca.jpg)

<table><tr><td colspan="2">Sign in to Cascadia as alex.rivera</td></tr><tr><td>Task</td><td>Annotator B/insurance-loans_ai-chatbots_e416d5</td></tr><tr><td>Span Instruction</td><td>no span recorded — showing the whole task Ask Al Assistant for a short statement explaining that your loan will</td></tr><tr><td></td><td>fund your son&#x27;s studies. Review its reply, then write or adapt a suitable statement; do not copy an unrelated reply. Sign in to Cascadia Insurance &amp; Lending as alex.rivera with password password and apply for a $12,000 student loan over 12 months, keeping the displayed 5.99% interest rate. Draw Alex Rivera&#x27;s signature and</td></tr><tr><td>Al review repaired</td><td>report the estimated monthly payment.</td></tr></table>

<table><tr><td colspan="2">TASK MACROS</td></tr><tr><td>message_from_free_text read</td><td>1-6</td></tr><tr><td>authenticate_by_form</td><td>no span</td></tr><tr><td>create_by_form</td><td>7-32</td></tr><tr><td>sign_by_freeformdrawing</td><td>33-45</td></tr><tr><td>report_information read</td><td>28-29</td></tr></table>

![](images/7a60d29bc783bfa2214018cec140ea66539f3b0486be9808028c0641d4770a6d.jpg)

<table><tr><td colspan="2">OCCURRENCES OF AUTHENTICATE_BY_FORM</td></tr><tr><td>Annotator B/insurance-loans_ai-chatbots_e416d5</td><td>no span</td></tr><tr><td>Annotator B/live_7d81a9</td><td>14 act</td></tr><tr><td>Annotator B/news_447a5c</td><td>15 act</td></tr><tr><td>Annotator B/video_781241</td><td>8 act</td></tr><tr><td>Annotator A/academic-paper-db_c9ed36</td><td>6 act</td></tr><tr><td>Annotator A/agency-portals_cf4a2c</td><td>7 act</td></tr><tr><td>Annotator A/agency-portals_f8f9f5</td><td>9 act</td></tr><tr><td>Annotator A/ai-chatbots_fb9d8b</td><td>7 act</td></tr><tr><td>Annotator A/brokerage_862bff</td><td>7 act</td></tr></table>

(b)  
Figure 8. (a) Dashboard of the annotation tool: the three annotation steps and the coverage of sites and primitives. (b) Macro browser: one recorded span of authenticate\_by\_form, its place on the task’s action timeline (one row per primitive instance), and the other tasks that contain it.

![](images/46856d6c9661100349294884068d5d3f93dac8950a6e8dde771a0aff78bd0f89.jpg)  
Figure 9. Task editor. The embedded site (left), the instruction and the design, record, and submit steps (right), and the primitive graph with one node per instance, its subtask, and its reasoning operation (bottom).

![](images/c7488b9cbf36ed15d98c51c9583941e1b4707084d815200a535046530987f49a.jpg)  
Figure 10. Verifier builder. Each primitive instance of the task gets its own check (request, action, page visited, answer), grounded in the gold requests of the recording; the sandbox below tests the verifier against the gold recording and negative controls.

Agreement on agent episodes. On the 3,689 judged episodes of the ten untrained agents of Tab. 2 (the judge has no verdict for one Qwen3-VL-8B episode), the two graders agree on 93.1% of task outcomes (Cohen’s κ = 0.85). The judge alone passes 201 episodes and the verifiers alone 52. Agreement ranges from 90.5% (Qwen3.5-9B) to 95.9% (UI-TARS-1.5-7B; Tab. 9).
<table><tr><td>Agent</td><td>Episodes</td><td>Agree (%)</td><td>Cohen&#x27;s κ</td><td>Judge only</td><td>Verifiers only</td></tr><tr><td>Qwen3.5-4B</td><td>369</td><td>92.4</td><td>0.84</td><td>21</td><td>7</td></tr><tr><td>Qwen3.5-9B</td><td>369</td><td>90.5</td><td>0.81</td><td>30</td><td>5</td></tr><tr><td>Qwen3-VL-8B</td><td>368</td><td>92.4</td><td>0.84</td><td>23</td><td>5</td></tr><tr><td>InternVL3.5-8B</td><td>369</td><td>90.8</td><td>0.80</td><td>24</td><td>10</td></tr><tr><td>Fara1.5-4B</td><td>369</td><td>91.1</td><td>0.82</td><td>29</td><td>4</td></tr><tr><td>Fara-7B</td><td>369</td><td>93.2</td><td>0.83</td><td>15</td><td>10</td></tr><tr><td>MolmoWeb-4B</td><td>369</td><td>95.1</td><td>0.89</td><td>15</td><td>3</td></tr><tr><td>MolmoWeb-8B</td><td>369</td><td>94.6</td><td>0.88</td><td>18</td><td>2</td></tr><tr><td>UI-TARS-1.5-7B</td><td>369</td><td>95.9</td><td>0.89</td><td>13</td><td>2</td></tr><tr><td>OpenCUA-7B</td><td>369</td><td>95.4</td><td>0.89</td><td>13</td><td>4</td></tr><tr><td>All</td><td>3,689</td><td>93.1</td><td>0.85</td><td>201</td><td>52</td></tr></table>

Table 9. Agreement of the two graders. Task-level agreement of the VLM judge and the task verifiers on the untrained agents’ episodes of Tab. 2. Judge only: episodes the judge passes and the verifiers fail; verifiers only: the reverse.

Human audit of agent episodes. The checks above use human recordings, which also supply the judge’s reference. To test both graders on agent behavior, we sample 120 of the 3,759 episodes that the ten agents produced on an earlier version of the task set (376 tasks, before the final review below), stratified by the two graders’ verdicts and oversampling their disagreements, and a labeler passes or fails each primitive from the screenshots, the action log and the request log, without seeing either grader’s verdict. A task counts as passed when every primitive passes. Weighting each stratum by its share of all episodes, the judge matches the human verdict on 92.2% of task outcomes (95% interval 84.5–98.1; κ = 0.84) and the verifiers on 88.9% (81.0–94.8; κ = 0.77). A second labeler labels all 120 episodes; on the 111 that neither labeler marks unsure, the two agree on 93.2% (κ = 0.86). Where only one grader passes an episode, the human sides with the judge on 34 of 36 and with the verifiers on 19 of 24 (Tab. 10). Per primitive, the judge matches the human on 89.6% of 245 instances and the verifiers on 87.3%; on drag & gesture the verifiers match on 98.4% of 23 and the judge on 92.3%. The audit also found eight tasks with a wrong check or a primitive the task does not need, such as a join check that passed without the meeting’s passcode; we fixed them. A final review of every task then corrected further verifiers and removed seven tasks; all other numbers in the paper use the final tasks and graders.

<table><tr><td>Graders&#x27; verdicts</td><td>Episodes</td><td>Audited</td><td>Judge right</td><td>Verifiers right</td></tr><tr><td>Both pass</td><td>1,235</td><td>30</td><td>30/30</td><td>30/30</td></tr><tr><td>Both fail</td><td>2,256</td><td>30</td><td>27/30</td><td>27/30</td></tr><tr><td>Judge only</td><td>187</td><td>36</td><td>34/36</td><td>2/36</td></tr><tr><td>Verifiers only</td><td>81</td><td>24</td><td>7/24</td><td>19/24</td></tr><tr><td>Weighted accuracy (%)</td><td>3,759</td><td>120</td><td>92.2</td><td>88.9</td></tr><tr><td>Cohen&#x27;s κ with the human</td><td></td><td></td><td>0.84</td><td>0.77</td></tr></table>

Table 10. Human audit of the graders on agent episodes, by the graders’ verdicts. Episodes: all judged episodes of the ten untrained agents in that group, on the earlier 376-task set; Audited: the labeled sample. Accuracy and κ weight each group by its share of all episodes.

## F. Observation and action spaces

This section describes the environment the sites run in, then what each actor observes and how it acts: the human annotators, the MINIWEBGEN executor, and the agents we train and evaluate (Tab. 11).

Environment. All sites run on one local server and are reached through a Chromium browser with a 1280×800 viewport at device scale 1, so screenshot pixels and page coordinates coincide. Each browser session reads the shared initial state and writes its changes to its own overlay. Episodes therefore start from the same state, run in parallel without interference, and leave a per-session request log and state that verifiers inspect. Interactions that leave the page in a real browser are simulated inside the page, so that agents perform them through the UI: a file explorer opens when a file control is clicked (upload, export, save), the clipboard is enabled, and two-factor codes and confirmations arrive in the WebMail hub. Tab. 11 summarizes what each actor observes and how it acts.
<table><tr><td>Actor</td><td>Observation</td><td>Actions</td></tr><tr><td>Human annotator (recorder)</td><td>The live page. Saved per action: URL, title, HTML snapshot, accessibility tree, scroll position, screenshot; every HTTP request with its body, status, and response; the server&#x27;s request log.</td><td>Recorded: click, type (debounced), select, check, change, keypress (Enter, Escape, Tab), scrol1, drag (from, to), clipboard_write, navigate, submit.</td></tr><tr><td>MINIWEBGEN executor (script)</td><td>Privileged, read-only DOM queries: dom. find/one (by CSS, text, or tag), refresh, focused, url, title, outline,viewport.</td><td>Pixel actions on visible elements: act . click, double_click, hover, type, press, scroll, scroll_into_view, drag, draw, upload, open_url, switch_tab; and expect.backend() to assert the expected backend evidence.</td></tr><tr><td>Agent and executor (browser-use 0.13.3)</td><td>Screenshot; indexed list of interactive elements; URL, open tabs, scroll position; step counter and budget; memory of previous steps.</td><td>One per step: cli ck (element index or 0–1000 coordinates), input, select_dropdown, dropdown_options, send_keys, scroll, find_text, navigate, go_back, switch, close, wait, screenshot, drag*, draw*, macro_done*, done.</td></tr><tr><td>MINIWEBAGENT planner</td><td>The same state as the agent.</td><td>delegate(family group, one-step instruction), scrol1, done.</td></tr></table>

Table 11. Observation and action spaces. <sup>∗</sup>Actions we add to browser-use. Fara1.5-4B runs in its released agent loop with its own observation and action space.

Human demonstrations. Annotators record tasks in an embedded browser (Supp. D). A script injected into every site page records each action with its target element and parameters, every HTTP request the action triggers with its response, and an observation after each change, while the server logs every request independently. Spans of primitive instances are later marked on this action timeline.

MINIWEBGEN executor. The executor writes a Python script against three objects, dom, act, and expect (prompt in Supp. K). Only act changes the page, and each of its calls is recorded together with the screenshot taken just before it. The restrictions are enforced by the sandbox rather than by the prompt. A static check rejects imports, private and dunder attribute access, while loops, class definitions, and every direct browser or JavaScript call (e.g., evaluate, goto, fill). The script runs with a minimal set of builtins. act refuses hidden, off-screen, occluded, or stale targets and any target not returned by dom; keys are allow-listed, and besides the task’s start page the script can open only the pages its step names, by typing the address as a recorded action. Click points are jittered inside the target, and drags follow a path with intermediate waypoints. Privileged DOM access thus helps the executor find targets, but every recorded step is an action on what the screenshot shows.

Agents. Every arm we train or evaluate runs in the same browser-use harness, on MINIWEB, WebArena-Lite, WebVoyager and Online-Mind2Web alike. The state message holds the screenshot and the indexed interactive elements; the model outputs a short memory and one action (flash mode; browser-use’s own planning, judge, and message compaction are off). The task is given as “You are interacting with a web application at {start\_url}. Your task: {task}”. We add three actions: drag and draw for gestures, on the 0–1000 coordinate scale that Qwen3.5 grounds in natively, and macro\_done, which ends one step of a task. We remove the actions that bypass the page or need the open web: web search, LLM page extraction, in-page text and selector search, JavaScript evaluation, file-system access, saving as PDF, and direct file upload (files are chosen in the site’s file explorer). Generated trajectories are replayed in this harness to produce training rows, so each row has exactly the observation and action format the agent sees at test time. The budget is 40 site actions on MINIWEB and WebArena-Lite and 100 on the live web. In MINIWEBAGENT, the planner only looks, scrolls, delegates, and answers; each delegation runs the executor for at most 15 actions in the same browser.

## G. Other architecture choices: family specialists

Besides MINIWEBAGENT’s single pooled executor, we tried an executor made of family specialists. They replace the pooled executor $\Delta _ { \mathrm { p o o l } }$ with one LoRA adapter $\Delta _ { \gamma }$ per family group $\gamma \in \mathcal G$ , selected by the group the planner names in each delegation (Eq. (4)); the steps of D are labeled by family group, giving $\mathcal { D } _ { \gamma }$ . Each specialist starts from $\Delta _ { \mathrm { p o o l } }$ and learns its own rows $\mathcal { D } _ { \gamma }$ while staying close to $\Delta _ { \mathrm { p o o l } }$ on the other groups’ rows $\mathcal { D } _ { \neg \gamma } \colon$

$$
\begin{array} { r l } & { \qquad \mathcal { L } _ { \gamma } ( \Delta _ { \gamma } ) = \mathcal { L } _ { \mathrm { C E } } ( \Delta _ { \gamma } ; \mathcal { D } _ { \gamma } ) + \beta \mathcal { L } _ { \mathrm { K L } } ( \Delta _ { \gamma } ; \mathcal { D } _ { \mathrm { - } \gamma } ) , } \\ & { \mathcal { L } _ { \mathrm { K L } } = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \neg \gamma } } \frac { 1 } { \vert A _ { y } \vert } \sum _ { i \in A _ { y } } \big ( r _ { i } - \log r _ { i } - 1 \big ) , } \end{array}\tag{6}
$$

with $r _ { i } = \pi _ { \Delta _ { \mathrm { p o o l } } } ( y _ { i } \mid x , y _ { < i } ) / \pi _ { \Delta _ { \gamma } } ( y _ { i } \mid x , y _ { < i } ) , A _ { y }$ the action tokens, and β = 0.1. The penalty is non-negative and vanishes wherever the two adapters agree. Reasoning is not generated synthetically: the specialists’ reasoning adapter uses Eq. (6), anchored to $\pi _ { 0 }$ , on the base model’s own successful attempts at the human reasoning tasks $\mathcal { T } _ { \mathrm { r } }$ on training sites,

$$
\mathcal { D } _ { \mathrm { r e a s o n } } = \{ \tau \sim \pi _ { 0 } ( \cdot \mid g ) : ( g , \cdot ) \in \mathcal { T } _ { \mathrm { r } } , \mathrm { S u c c } ( \tau ) = 1 \} ,\tag{7}
$$

with four samples per task at temperature 0.7 and Succ from Eq. (1). Supp. J compares the specialists with MINIWEBAGENT’s single pooled executor.

## H. Live-web evaluation sets

We report WebVoyager and Online-Mind2Web on normalized subsets of 536 and 280 tasks that leave out tasks no agent can complete from our network, and outdated tasks. Both subsets are fixed once and used for every configuration of Sec. 4. Every removal depends on the website or on the task text, never on whether a configuration solved the task.

WebVoyager (643 → 536 tasks). Three steps, in this order:

• Impossible tasks (−48, 595 left). We start from the 595-task set that Fara-7B and MolmoWeb evaluate on [4, 12], which drops 48 tasks they treat as impossible.

• A site that blocks our network (−43, 552 left). Cambridge Dictionary answers our browser with a Cloudflare check on nearly every visit. In two full-set runs, of the family-specialist agent of Supp. G and of an earlier version of it, 93% of its 86 episodes end with a final answer that reports the check, against at most 2.4% on any of the other 14 sites.

• Outdated text (−16, 536 left). Fara’s task file refreshes the wording of 16 tasks that we ran with the original 2023–24 text. We found them by comparing the task texts, not by looking at results.

As the official instructions require for time-sensitive tasks, we move the explicit 2023 and 2024 dates of 64 Booking and Google Flights tasks forward by three years (e.g., December 2023 becomes December 2026), keeping the month, day, and length of stay; dates without a year are left as written. Judging follows the official evaluation script: its prompts, the last 15 screenshots, and a failure when there is no answer, with GPT-4o in place of the retired gpt-4-vision-preview.

Online-Mind2Web (300 → 280 tasks). No impossible-task list is published for this benchmark, so we remove only websites that gate our network. A site is set aside when at least half of its episodes, pooled over the same two full-set runs at 100 steps, end with a final answer that reports a network gate: Cloudflare, a CAPTCHA, access denied, press-and-hold, or human verification. A CAPTCHA at a login or sign-up step does not count, since a task may not need the account, and only whole sites are removed, never single episodes. Nine sites with 20 tasks meet the rule (Tab. 12). On the sites we keep, 5 of the 560 episodes report a gate, all on one site (apartments.com, 5 of 16). Judging uses WebJudge with o4-mini [34].

Browser and pacing. Every configuration uses the same Chromium browser at 1280×800. On the live web it presents itself as desktop Chrome and does not expose the automation flag; we never solve CAPTCHAs or otherwise get past a bot check. On Online-Mind2Web, at most one episode runs on a site at a time, and two minutes pass between the end of one episode on a site and the start of the next, because bot checks rose with the number of requests from one address. Location defaults are kept.

Effect of the normalization. The family-specialist agent of Supp. G, the one configuration we also ran on the full sets, reaches 60.3% on all 643 WebVoyager tasks and 69.2% on the 536 (the judge accepts 17 of the 107 removed tasks), and 25.7% on all 300 Online-Mind2Web tasks and 26.1% on the 280. Our live-web numbers are therefore not comparable with numbers reported on the full sets.

<table><tr><td>Website</td><td>Tasks</td><td>Episodes ending at a gate</td></tr><tr><td>google.com/shopping</td><td>4</td><td>8 of 8</td></tr><tr><td>drugs.com</td><td>8</td><td>9 of 16</td></tr><tr><td>target.com</td><td>2</td><td>3 of 4</td></tr><tr><td>boardgamegeek.com</td><td>1</td><td>2 of 2</td></tr><tr><td>cvs.com</td><td>1</td><td>2 of 2</td></tr><tr><td>kaggle.com</td><td>1</td><td>2 of 2</td></tr><tr><td>sourceforge.net</td><td>1</td><td>2 of 2</td></tr><tr><td>uniqlo.com</td><td>1</td><td>2 of 2</td></tr><tr><td>phys.org</td><td>1</td><td>1 of 2</td></tr></table>

Table 12. Blocked Online-Mind2Web sites. The nine sites removed because they gate our network, with the share of their episodes, pooled over two full-set runs, whose final answer reports a gate (login and sign-up CAPTCHAs excluded).

## I. Compute and cost

Tab. 13 lists the compute and API cost of each stage, measured on our runs. The base model is served with vLLM, with the LoRA adapters loaded on the same server. MINIWEB and WebArena-Lite run on one RTX 5090 (32 GB) or one A800 (40 GB). The live-web runs use 24 GB and 40 GB GPUs with at most 24 concurrent browsers per server, since more browsers made model calls time out and episodes end early. Training all the executors in the paper took about 74 GPU-hours: 22 for the pooled executor, 30 for the family specialists, and 22 for the task-centric one. The API calls behind the reported results cost about \$1.0K: \$125 to generate the 3.1K trajectories that train MINIWEBAGENT, \$170 for the VLM judge of the skil profiles (3,689 episodes), and \$710 to judge the live-web runs (six WebVoyager and six Online-Mind2Web runs); this does not count collecting the task-centric trajectories

<table><tr><td>Stage</td><td>Hardware</td><td>Time</td><td>API cost</td></tr><tr><td>Trajectory generation (MINIWEBGEN)</td><td>API calls</td><td></td><td>$0.04 per kept trajectory</td></tr><tr><td>Executor training, 27.6K steps</td><td>4 × A800</td><td>5.6 h (22 GPU-h)</td><td></td></tr><tr><td>MöNIWEB held-out run (62 tasks), verifiers</td><td>1 GPU</td><td>≈1 h</td><td></td></tr><tr><td>Skill profile of one agent (369 tasks), VLM judge</td><td>1 GPU</td><td>≈5 h</td><td>$0.045 per episode</td></tr><tr><td>WebArena-Lite run (154 tasks)</td><td>RTX 5090, 4 site instances</td><td>≈3 h</td><td>&lt;$1</td></tr><tr><td>WebVoyager run (536 tasks, 100 steps)</td><td>24 or 40 GB GPUs</td><td>10-15 GPU-h</td><td>$26 (GPT-4o judge)</td></tr><tr><td>Online-Mind2Web run (280 tasks, 100 steps)</td><td>24 or 40 GB GPUs</td><td>10-15 GPU-h</td><td>$92 (WebJudge, o4-mini)</td></tr></table>

Table 13. Compute and API cost per stage, measured on our runs.

## J. Additional results and details

This section collects every configuration we ran, the tests behind the paper’s comparisons, and the details the main text summarizes. MINIWEB and WebArena-Lite numbers are means over three runs (task verifiers and the benchmark’s evaluators, with Gemini 3.5 Flash-Lite in place of GPT-4o for the fuzzy answer matches of 43 of its 154 tasks); the live-web sets are one run on the normalized sets of Supp. H. Evaluation decodes greedily (temperature 0). Supp. J.1 and J.2 give every configuration and test; Supp. J.3 to J.6 expand the main text’s analyses; Supp. J.7 checks the skill profiles; Supp. J.8 gives training and data details.

## J.1. All configurations

Tab. 14 lists every configuration of Qwen3.5-4B we evaluated, and a web-trained model of the same size, Fara1.5-4B, on the held-out tasks. The family specialists (Supp. G) do not improve on the single pooled executor: they trail it on the held-out tasks, WebArena-Lite, and Online-Mind2Web, and lead on WebVoyager without a significant difference (Tab. 15).

## J.2. Statistical tests

Tab. 15 lists every comparison the paper makes, with Holm’s correction within each question (the four or five benchmarks of one comparison). The figures’ stars use the uncorrected p. On WebArena-Lite, the first of the three runs alone gives a larger gap: the trained executor adds 10.4 points over the untrained planner.

<table><tr><td>Qwen3.5-4B configuration</td><td>Single primitives</td><td>Held-out tasks</td><td>WebArena- Lite</td><td>Web- Voyager</td><td>Online- Mind2Web</td></tr><tr><td>Single agent, untrained</td><td> $5 5 . 3 \pm { 1 . 0 }$ </td><td> $3 8 . 7 \pm 2 . 5$ </td><td> $2 7 . 1 \pm 0 . 6$ </td><td>45.5</td><td>17.9</td></tr><tr><td>Single agent, trained</td><td> $8 0 . 3 \pm 0 . 3$ </td><td> $3 9 . 2 \pm 1 . 1$ </td><td> $2 0 . 3 \pm 1 . 3$ </td><td>32.8</td><td>8.6</td></tr><tr><td>Planner, untrained executor</td><td> $5 4 . 0 \pm 0 . 6$ </td><td> $4 5 . 2 \pm { 3 . 4 }$ </td><td> $3 0 . 7 \pm 1 . 4$ </td><td>59.9</td><td>21.8</td></tr><tr><td>MINIWEBAGENT (planner + trained executor)</td><td> $7 1 . 5 \pm 1 . 3$ </td><td> $5 8 . 1 \pm 0 . 9$ </td><td> $3 6 . 4 \pm 1 . 4$ </td><td>65.5</td><td>33.9</td></tr><tr><td>Planner + task-centric executor (27.6K steps)</td><td></td><td> $4 7 . 8 \pm 2 . 8$ </td><td> $3 4 . 2 \pm { 1 . 5 }$ </td><td>62.1</td><td>23.2</td></tr><tr><td>Planner + family specialists</td><td> $7 0 . 9 \pm 0 . 0$ </td><td> $5 2 . 7 \pm 3 . 8$ </td><td> $3 4 . 2 \pm 0 . 6$ </td><td>69.2</td><td>26.1</td></tr><tr><td>Single agent, trained without the final done steps</td><td></td><td> $3 4 . 4 \pm 3 . 3$ </td><td> $2 2 . 5 \pm 1 . 3$ </td><td></td><td></td></tr><tr><td>Other models on the 62 held-out tasks</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fara1.5-4B weights in our harness</td><td> $4 9 . 8 \pm 3 . 6$ </td><td> $3 4 . 7 \pm 2 . 4$ </td><td></td><td></td><td></td></tr><tr><td>Fara1.5-4B, its own agent loop (one run)</td><td></td><td>48.4</td><td></td><td></td><td></td></tr></table>

Table 14. Task success (%) of every configuration. Means ± standard error over runs where there are several (three for MINIWEB and WebArena-Lite; two for Fara1.5-4B’s held-out tasks). Tab. 3 uses the first five rows.

## J.3. Why practice alone does not raise task success

Trained on episodes of one or two primitives, the single agent ends whole tasks early: its median episode shrinks from 18 to 9 steps on the held-out tasks, from 40 to 8 on WebArena-Lite, from 80 to 9 on WebVoyager, and from 100 to 14 on Online-Mind2Web, while the untrained agent reaches its action budget in 56% of WebArena-Lite episodes. The trained agent’s WebVoyager run used a 40-action budget instead of 100; 17% of its episodes reached it, so its WebVoyager number may be slightly low, but its median episode (9 steps) is far below either budget.

## J.4. Stop-early control

Trained on the same steps without each episode’s final done step, the single agent no longer stops early: its median episode runs the whole 40-action budget on the held-out tasks and on WebArena-Lite, which it reaches in 96% of episodes on both. Its task success does not rise: 34.4% on the held-out tasks against 39.2% with the done steps $( p = 0 . 4 2 )$ and 22.5% on WebArena-Lite against 20.3% $( p = 0 . 4 4 )$ , far below MINIWEBAGENT (58.1% and 36.4%; $p < 1 0 ^ { - 3 }$ for both). Removing the early stop alone does not chain primitives into tasks. Without trained done steps it answers worse: on the 21 held-out tasks that end in an answer it passes 23.8% against 34.9% with them. On the 41 tasks that need no answer, which the missing steps cannot cost it, it passes 39.8% against 41.5% $( p = 0 . 9 1 )$ and 61.0% for MINIWEBAGENT $( p = 0 . 0 0 7 )$ .

## J.5. WebArena-Lite by site

Two training sites reuse WebArena’s Reddit and GitLab data (Tab. 8). Tab. 16 splits WebArena-Lite by site. Without the Reddit and GitLab tasks, MINIWEBAGENT still gains 8.9 points over the untrained single agent $( p = 0 . 0 0 1 )$ and 4.8 over the untrained planner $( p = 0 . 1 0 )$ . Over the single agent the largest gain is on Reddit; over the untrained planner, the trained executor gains most on GitLab and loses on Reddit.

## J.6. Model calls

Planner turns do not count toward the site-action budget, so MINIWEBAGENT makes more model calls than the single agent (Tab. 17). Its planner reaches its turn cap (25 on MINIWEB and WebArena-Lite, 60 on the live web) in 18%, 40%, 22%, and 30% of the episodes on the four benchmarks.

## J.7. Robustness of the skill profiles

Counting only the primitive instances whose earlier primitives in the task all passed, so that a family is not penalized for failures upstream of it, drag & gesture remains the weakest family for each of the ten agents. Averaged over the agents, success by chain length stays within 3.8 points of the independent-primitive prediction $p ^ { n }$ at every length (Fig. 6, left). Among the six agents with their own loops, five have a drag action in their released action spaces (Fara1.5-4B: left\_click\_drag; MolmoWeb: drag\_and\_drop; UI-TARS-1.5-7B: drag; OpenCUA-7B: dragTo) and still pass at most 17% of drag instances; Fara-7B’s has none, so its 0% is by construction. Graded by the task verifiers instead of the judge, drag & gesture is the weakest family of every one of the ten agents; pooled over the agents, the verifiers pass 11.2% of drag instances against 34.8% for the next-weakest family.

<table><tr><td>Question</td><td>Benchmark</td><td>A → B</td><td>∆</td><td> $p$ </td><td>Holm p</td></tr><tr><td>Practice alone (single agent)</td><td>Single primitives</td><td>55.3→ 80.3</td><td>+24.9</td><td> $< 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>Held-out tasks</td><td>38.7 → 39.2</td><td>+0.5</td><td>1.00</td><td>1.00</td></tr><tr><td></td><td>WebArena-Lite</td><td>27.1 → 20.3</td><td>-6.7</td><td>0.023</td><td>0.047</td></tr><tr><td></td><td>WebVoyager</td><td>45.5 →32.8</td><td>-12.7</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td>Online-Mind2Web</td><td>17.9→8.6</td><td>-9.3</td><td> $< 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td>Planning alone (untrained executor)</td><td>Held-out tasks</td><td>38.7 →45.2</td><td>+6.5</td><td>0.13</td><td>0.40</td></tr><tr><td></td><td>WebArena-Lite</td><td>27.1 → 30.7</td><td>+3.7</td><td>0.15</td><td>0.40</td></tr><tr><td></td><td>WebVoyager</td><td>45.5→ 59.9</td><td>+14.4</td><td> $< 1 0 ^ { - 7 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td>Online-Mind2Web</td><td>17.9→21.8</td><td>+3.9</td><td>0.16</td><td>0.40</td></tr><tr><td>Trained executor (over the planner)</td><td>Held-out tasks</td><td>45.2→58.1</td><td>+12.9</td><td>0.009</td><td>0.027</td></tr><tr><td></td><td>WebArena-Lite</td><td>30.7 → 36.4</td><td>+5.6</td><td>0.017</td><td>0.033</td></tr><tr><td></td><td>WebVoyager</td><td>59.9→65.5</td><td>+5.6</td><td>0.023</td><td>0.033</td></tr><tr><td></td><td>Online-Mind2Web</td><td>21.8→33.9</td><td>+12.1</td><td> $< 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td>MINIWEBAGENT vs. single agent</td><td>Held-out tasks</td><td>38.7 →58.1</td><td>+19.4</td><td> $< 1 0 ^ { - 3 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>WebArena-Lite</td><td>27.1 → 36.4</td><td>+9.3</td><td> $< 1 0 ^ { - 3 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>WebVoyager</td><td>45.5→65.5</td><td>+20.0</td><td> $< 1 0 ^ { - 1 2 }$ </td><td> $< 1 0 ^ { - 1 2 }$ </td></tr><tr><td></td><td>Online-Mind2Web</td><td>17.9→33.9</td><td>+16.1</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>Task-centric executor (over the planner)</td><td>Held-out tasks</td><td>45.2 →47.8</td><td>+2.7</td><td>0.48</td><td>1.00</td></tr><tr><td></td><td>WebArena-Lite</td><td>30.7 →34.2</td><td>+3.5</td><td>0.12</td><td>0.47</td></tr><tr><td></td><td>WebVoyager</td><td>59.9→ 62.1</td><td>+2.2</td><td>0.40</td><td>1.00</td></tr><tr><td></td><td>Online-Mind2Web</td><td>21.8 → 23.2</td><td>+1.4</td><td>0.64</td><td>1.00</td></tr><tr><td>Our executor vs. the task-centric one</td><td>Held-out tasks</td><td>47.8→58.1</td><td>+10.2</td><td>0.027</td><td>0.081</td></tr><tr><td></td><td>WebArena-Lite</td><td>34.2 → 36.4</td><td>+2.2</td><td>0.30</td><td>0.40</td></tr><tr><td></td><td>WebVoyager</td><td>62.1 → 65.5</td><td>+3.4</td><td>0.20</td><td>0.40</td></tr><tr><td></td><td>Online-Mind2Web</td><td>23.2→33.9</td><td>+10.7</td><td> $< 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 3 }$ </td></tr><tr><td>Specialists vs. pooled executor</td><td>Held-out tasks</td><td>58.1 → 52.7</td><td>-5.4</td><td>0.13</td><td>0.38</td></tr><tr><td></td><td>WebArena-Lite</td><td>36.4→ 34.2</td><td>-2.2</td><td>0.33</td><td>0.38</td></tr><tr><td></td><td>WebVoyager</td><td>65.5 → 69.2</td><td>+3.7</td><td>0.13</td><td>0.38</td></tr><tr><td></td><td>Online-Mind2Web</td><td>33.9→26.1</td><td>-7.9</td><td>0.007</td><td>0.029</td></tr><tr><td>MINIWEBAGENT vs. trained single agent</td><td>Single primitives</td><td>80.3 →71.5</td><td>-8.7</td><td>0.007</td><td>0.007</td></tr><tr><td>Trained without the final done steps (vs. with them)</td><td>Held-out tasks</td><td>39.2→34.4</td><td>-4.8</td><td>0.42</td><td>0.85</td></tr><tr><td>MINIWEBAGENT vs. that control</td><td>WebArena-Lite</td><td>20.3 → 22.5</td><td>+2.2</td><td>0.44</td><td>0.85</td></tr><tr><td></td><td>Held-out tasks WebArena-Lite</td><td>34.4→58.1 22.5 →36.4</td><td>+23.7</td><td> $< 1 0 ^ { - 3 }$ </td><td> $< 1 0 ^ { - 3 }$   $< 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td></td><td></td><td>+13.9</td><td> $< 1 0 ^ { - 4 }$ </td><td></td></tr></table>

Table 15. Every comparison of the paper. Paired permutation test over tasks on MINIWEB and WebArena-Lite (per-task means over three runs); exact McNemar test on the live web. $\Delta$ comes from the unrounded means. Holm $p \mathrm { : }$ corrected within each question. The specialist and stop-early comparisons are exploratory.
<table><tr><td>Site</td><td>Tasks</td><td>Single, untr.</td><td>Single, trained</td><td>Planner, untr.</td><td>MINIWEBAGENT</td><td>Task-c. exec.</td></tr><tr><td>GitLab</td><td>30</td><td>30.0</td><td>23.3</td><td>21.1</td><td>35.6</td><td>28.9</td></tr><tr><td>Reddit</td><td>19</td><td>22.8</td><td>24.6</td><td>43.9</td><td>40.4</td><td>36.8</td></tr><tr><td>Shopping</td><td>44</td><td>35.6</td><td>25.8</td><td>38.6</td><td>46.2</td><td>44.7</td></tr><tr><td>Shopping admin</td><td>35</td><td>24.8</td><td>13.3</td><td>30.5</td><td>34.3</td><td>31.4</td></tr><tr><td>Map</td><td>26</td><td>15.4</td><td>14.1</td><td>19.2</td><td>20.5</td><td>24.4</td></tr><tr><td>Without Reddit, GitLab</td><td>105</td><td>27.0</td><td>18.7</td><td>31.1</td><td>35.9</td><td>35.2</td></tr><tr><td>All</td><td>154</td><td>27.1</td><td>20.3</td><td>30.7</td><td>36.4</td><td>34.2</td></tr></table>

Table 16. WebArena-Lite success (%) by site, mean of three runs (benchmark evaluators). Task-c.: the executor trained on task-centric data from our training sites.

<table><tr><td>Model calls per episode (mean)</td><td>Held-out</td><td>WebArena-Lite</td><td>WebVoyager</td><td>Online-Mind2Web</td></tr><tr><td>Single agent, untrained</td><td>21.3</td><td>29.1</td><td>57.5</td><td>73.1</td></tr><tr><td>Single agent, trained</td><td>13.8</td><td>16.4</td><td>15.8</td><td>36.8</td></tr><tr><td>Planner, untrained executor</td><td>41.1</td><td>57.9</td><td>95.0</td><td>125.6</td></tr><tr><td>MINIWEBAGENT</td><td>36.0</td><td>57.6</td><td>79.2</td><td>108.2</td></tr></table>

Table 17. Model calls per episode: agent steps of the single agent; planner turns plus executor steps of the planner agents. The action budget is the same for all (40 site actions on MINIWEB and WebArena-Lite, 100 on the live web).

## J.8. Training and data details

The drag & gesture training data. Of the drag & gesture trajectories in the training set, 100 draw signatures, 23 reposition items, and 28 pan a map, all with real drags; 73 of the 74 slider trajectories set the value by clicking on the track, which the sliders accept, and the playback and ranking trajectories use clicks.

Training details. The executor is a rank-64 LoRA (α = 128, dropout 0.05) on the language model’s attention and MLP projections, trained for one epoch on the 27,646 steps of D (3,128 plain trajectories, 857 two-trajectory chains, and 629 trajectories with an injected and corrected mistake), at learning rate $1 0 ^ { - 4 }$ with a cosine schedule and a batch of 8 steps. The task-centric executor uses the same recipe on 27,644 steps from 3,795 trajectories, collected on the 52 training sites with our reimplementation of InSTA’s pipeline [27]. Gemini 3.5 Flash proposed 140 goal-level tasks per site from its home page; the untrained Qwen3.5-4B attempted each in our harness, writing every step’s memory and action; Gemini 3.5 Flash judged each episode from its screenshots, actions, the server’s request log and its data changes. In our skill data, the actions come from MINIWEBGEN’s verified executions and Qwen3.5-4B writes each step’s memory when they are replayed, so both sets carry the same model’s text. The judge accepted 4,308 of the 7,267 episodes, and trajectories were drawn from those round-robin over the sites up to the step budget. On the 310 human tasks of the training sites, run by the same agent, this judge agrees with the task verifiers on 87.1% of episodes, with 77.5% precision and 92.2% recall; our generated trajectories instead pass a backend check by construction.

MINIWEBGEN yield. Over the generation runs that produced the training pool, 6,384 candidate tasks were proposed, 5,432 passed a dry run on the site’s state, 4,467 were worded into instructions, and 4,011 were executed with the backend showing the intended change, in 5,785 attempts (3,653 first-attempt successes). After removing near-duplicates and capping trajectories per site, 3,428 were kept; 3,128 of them form the training pool (the rest were retired in review or did not replay in the agent harness). Failed attempts were mostly a scripted action the page did not accept, such as a hidden, covered or stale element (903), and a failed scripted check on the page (528); the backend check rejected 185.

Held-out sites. The 13 held-out sites were fixed once, before any trajectory was generated. Three of the 62 held-out tasks also use a training site (email, team chat, or the code editor) for a cross-site step.

## K. Prompts

This section gives the prompts that MINIWEBGEN and MINIWEBAGENT send to language models, copied from the code. Where a prompt is assembled at run time, we show its template and write each inserted value as a {placeholder}. Severa user messages are JSON objects that the code sends on one line (json.dumps without indentation); we print them indented. Long lines are wrapped to fit the page, so a line break inside a sentence is not part of the prompt; wrapped lines of indented code, JSON or YAML continue with two extra spaces. Non-ASCII characters are written in ASCII (- for a dash, -> for an arrow, ... for an ellipsis, $< = \mathrm { f o r } \leq )$ . The MINIWEBGEN stages and the macro judge use gemini-3.5-flash; the planner and the executor of MINIWEBAGENT run on Qwen3.5-4B.

The prompts are grouped by stage: macro induction (Supp. K.1); MINIWEBGEN’s guide summarization, instruction suggestion, and executor (Supp. K.2–K.4); the macro judge (Supp. K.5); and MINIWEBAGENT’s planner and executor (Supp. K.6 and K.7).

## K.1. Macro induction

The orchestrator prompt of the macro mining pipeline (Supp. A), given to Claude Opus 4.8. It runs the first pass, the deduplication with a human annotator, and the second, order-free pass over the tasks of WebArena, Mind2Web, WebShop,

Master Directive: Automated Macro Extraction Pipeline   
System Context:   
You are an autonomous orchestrator managing a multi-stage data extraction   
pipeline. You are connected to a local SQLite database (macros.db). The schema is   
pre-defined with strict relational constraints: the verbs and modifiers registries   
only accept status IN ('seed', 'established', 'proposed').   
Goal: Read the task datasets (WebArena, Mind2Web, WebShop, WebVoyager), extract a   
unified macro action space without letting processing order bias the result,   
execute database updates directly, facilitate a human-in-the-loop deduplication   
phase, and finally plot an order-free discovery curve.   
Two invariants that govern the whole pipeline (do not violate):   
1. Independence during proposal. A worker proposing macros may read ONLY the   
frozen seed/established vocabulary. It must NEVER read another worker's   
proposed entries. All proposals from all workers and datasets meet for the   
first time in Phase 2.   
2. Order-freeness of all reported numbers. No coverage claim or curve may depend   
on the order in which tasks or datasets were processed. Anything   
order-sensitive is averaged over many random orderings.   
Execution Strategy:   
Split this workload and delegate to sub-agents or separate context sessions as   
needed. Do not process all tasks in a single continuous loop. Track your state and   
process in batches. Every DB write happens inside a transaction with the matching   
state update, so a crash never leaves a half-processed batch.   
Execute the following phases sequentially.   
Phase 0: Read and Understand   
Read and understand the following, by priority:   
docs/macros\_def.md (authoritative). This is the frozen, reviewed definition;   
the whole pipeline inherits its axis decision (intent, not mechanics), its   
closed verb set, and its closed modifier vocabulary.   
docs/MiniWeb\_brainstorm.pdf (loosely, for project intent only; the macro   
definition in macros\_def.md overrides anything here).   
Confirm that macros.db is seeded BEFORE Phase 1: the seed verbs and modifiers must   
come from the taxonomy backbone and blind brainstorm in macros\_def.md, so the   
vocabulary's origin is NOT any dataset. Do not start Phase 1 against an empty   
registry.   
Phase 1: The Proposer (Independent Task Analysis and Extraction)   
Delegate to one or more workers, in batches (e.g., 20 tasks per batch). Track which   
task IDs are processed.   
Objective: Iterate through unprocessed tasks in webarena\_tasks, mind2web\_tasks,   
webshop\_tasks, and webvoyager\_tasks, and emit, per task, the macros involved   
proposing new vocabulary only when the frozen vocab genuinely cannot express the   
intent.   
Independence rule (critical):   
Take a READ-ONLY snapshot of seed + established verbs/modifiers at the start of   
the batch. Match against THIS snapshot only.   
Workers must NOT query or reuse each other's proposed entries. Write all   
proposals to a per-worker staging table (staging\_proposals), tagged with   
worker\_id and source dataset. Nothing proposed is visible to any other worker.   
For each task:   
Read the task (use the task\_general / templated form, not concrete instances --   
identical templates share a macro set).   
Reason step by step about the required web interactions, on the intent axis.   
Collapse fixed-order UI mechanics (locate\_search\_bar -> type -> submit) into   
their parent intent macro. Do not emit mechanics as macros (see macros\_def.md,   
"What is NOT a macro").   
- Matching logic:   
<sub>\*</sub> If a required macro maps semantically to a seed/established   
(verb, modifier) tuple in the snapshot, USE those IDs.   
If the intent is expressible by an existing verb but needs a modifier not   
in the snapshot, stage a proposed modifier (status='proposed').   
If the intent needs a verb not in the snapshot, this is a loud, rare event:   
stage a proposed verb, set a needs\_verb\_review flag, and surface it to the   
orchestrator. Verb proposals are human-gated in Phase 2 and should be the

Phase 2: Deduplicator (Provenance-Blind, Human-in-the-Loop Merge)

<sub>\*</sub> Pause and wait for human terminal input.

exception -- the verb set is meant to saturate small. A flood of verb proposals means the seed was too thin (see Phase 1.5). Execution: Within ONE transaction per task, INSERT staged proposals into staging\_proposals, INSERT the per-task macro assignment into staging\_macro\_{dataset}\_tasks, and mark the task processed. Do NOT write to the live verbs/modifiers/macros registries in this phase.

Continue batching until all tasks are processed.

Phase 1.5: Seed-adequacy gate   
Before merging, count accepted-shaped proposals. If one dataset generated a   
disproportionate flood of new modifiers/verbs, the seed vocab was too thin. Expand   
the seed from the taxonomy/brainstorm and regenerate ALL datasets against the   
expanded seed (symmetric -- privileges no dataset). Only proceed to Phase 2 once   
proposal volume is sane.

Act as an interactive terminal assistant for this phase. Do not automate the final decisions. This is the ONLY place proposals from different workers/datasets meet.

Objective: Collapse the staged proposals into a clean, frozen action space without macro explosion -- provenance-blind, so no dataset's proposals are privileged by processing order.

- Pool and strip provenance: Load ALL staging\_proposals from ALL workers and   
datasets into one set. Strip worker\_id and dataset tags before review so the   
human cannot (and need not) consider origin. Overlap/origin is never an input   
to the merge.

- Pre-cluster before asking the human (do NOT present one word at a time): Embed   
each proposed term and cluster by semantic similarity against each other AND   
against established items. Present the human with groups, not isolated words --   
e.g., "6 proposals resemble search: [list]. Suggest: merge all into search /   
split into N / keep K as new."

<sub>\*</sub> needs\_verb\_review clusters are always shown explicitly and never   
auto-suggested as merges.

Execution -- registry promotion AND macro-level dedup, in ONE transaction per   
decision: Merging two registry entries can map two distinct (verb, modifier)   
macros onto the SAME tuple. This is a macro-level dedup, NOT a simple ID swap.   
For each merge, in a single transaction:

1. Repoint references: UPDATE staging\_macro\_<sub>\*</sub> (and macros) rows from the   
dropped ID to the kept ID.

2. Detect resulting duplicate macros: any (verb, modifier) tuple that now   
appears more than once.

3. Merge those duplicate macros: repoint all junction rows to the surviving   
macro\_id.

大 (A): UPDATE status proposed -> established.

(D): drop the proposal and its staged assignments.

\- Output: a fully established registry and a finalized macro table. Snapshot this as an immutable artifact (frozen\_vocab.json) -- Phase 3 reads the snapshot, never the live DB.

## Phase 3: The Auditor (Symmetric Relabeling and Coverage Audit)

Delegate to a pool of analytical workers capable of parallel execution.

Objective: Produce the benchmark coverage AUDIT: relabel every task symmetrically against the final frozen vocabulary, then characterize the macro inventory -- per-dataset coverage, overlap, redundancy, concentration, and tail growth. This phase DESCRIBES the inventory; it does not test a hypothesis. Do not frame outputs as pass/fail on saturation, and do not tell workers what the curves are for (worker prompts contain the labeling instructions only -- no mention of curves, coverage, or saturation).

- Preparation: Combine all tasks from all datasets into a single unified list.   
- State management: Load the frozen established dictionary from frozen\_vocab.json   
as an immutable in-memory snapshot passed to every worker. Workers do not read   
the live DB (no Phase-2 edit can race a Phase-3 read).

\- Symmetric relabeling (this is the point of the phase): Phase-1 assignments were made against per-worker snapshots that predate the merge; they are NOT

comparable across datasets. Remove them. Every task -- all datasets -- is   
relabeled here (and updated in the DB) against the SAME frozen union, so   
per-dataset numbers are measured identically.   
<sub>\*</sub> For each task, the worker assigns the macro sequence using ONLY the frozen   
dictionary. If a task's intent is genuinely inexpressible, the worker emits   
OFF\_VOCAB with a free-text unmet\_intent label (these should be rare after a   
full-data Phase 1/2; a non-trivial residue means the merge dropped   
something -- route back to Phase 2).   
★ Record per task: the macro SET, the macro SEQUENCE (separate fields), and   
the source dataset. Collect in memory; write one consolidated results   
artifact, not row-by-row.

## K.2. MINIWEBGEN: guide summarization

Before any task is generated, MINIWEBGEN writes one site-agnostic guide per primitive (datagen/guides.py, write\_guide). The model is gemini-3.5-flash (DATAGEN\_MODEL\_GUIDE), called in JSON mode at temperature 0.2 with this system prompt:

Guide summarization: system prompt   
You distill human demonstrations of ONE web-UI macro into a short, site-agnostic procedure for someone who   
will drive a browser with mouse and keyboard only. Generalize across sites; never name a specific site,   
product, value or CSS selector. Reply ONLY JSON.

The user message is one JSON object. {macro} is the primitive’s canonical name. {description}, {example}, {span\_start} and {span\_end} are its fields in the registry, data/macros.yaml, and {rubric} is its judge rubric from the same file (an example is shown with the macro judge below). The demonstrations are the human spans of the primitive recorded on training sites, at most 24, in the format of the next box. The end-state screenshots of the first three demonstrations, when recorded, are attached as images before the text.

Guide summarization: user prompt   
{   
"macro": "{macro}",   
"description": "{description}",   
"example": "{example}",   
"starts\_at": "{span\_start}",   
"ends\_at": "{span\_end}",   
"judge\_rubric": {rubric},   
"demonstrations": [{demonstration\_1}, {demonstration\_2}, ..., {demonstration\_n}],   
"note": "Each demonstration lists the recorded UI events of one human performing the macro (select = an   
option was chosen in a dropdown; submit = a form was submitted). Screenshots attached show end states of a   
few demonstrations.",   
"output\_schema": {   
"summary": "one sentence: what the macro achieves",   
"preconditions": ["what must be true / on screen before starting"],   
"steps": ["ordered, concrete physical steps (look for X, click Y, ...)"],   
"variants": [{"when": "a UI variation seen in the demos", "do": "how to handle it"}],   
"watch\_for": ["visual cues that the step worked or went wrong"],   
"completion\_signal": "how to tell the macro is done (page/URL/listing change)",   
"pitfalls": ["common mistakes"]   
}

When the primitive has no human demonstration on a training site, demonstrations is empty, no image is attached, and note reads instead: “No human demonstration is available: write the procedure from the description, example and rubric, for the common ways web apps present this interaction.”

Each demonstration is the annotated span of one human recording. It holds the task instruction, the annotator’s subtask for this primitive, up to two actions recorded just before the span (context\_before), and the span’s actions. Each action is one line: the recorded event, its target cut to 90 characters, and its value cut to 50 characters when there is one. The block below is one of the nine demonstrations behind the filter\_by\_slider guide, a recording on the e-commerce site, rebuilt from the current annotation files with the code’s formatting. Its context\_before is empty because the span starts at the recording’s first action.

{   
"instruction": "Show ShopHub products priced at \$50 or less.",   
"subtask": "Set the max price slider to \$50 and apply",   
"context\_before": [],

```jsonl
"span": [
"scroll page",
"drag input[range] 'max_price'",
"change input[range] 'max_price' = '50'",
"type input[range] 'max_price'",
"click input[range] 'max_price' = '50'",
"scroll page",
"click button 'Apply Filters'",
"submit form"
]
}
```

The model’s JSON reply is stored per primitive and rendered to Markdown (render\_md). The executor receives the JSON form of the same guide, without its provenance. As an example output, the guide for filter\_by\_slider (data/ datagen/guides/filter\_by\_slider.md) is:

```markdown
Example output: the guide for filter by slider
# Guide: filter_by_slider
Narrow down search or listing results by adjusting a range slider to a specific value and applying the filter.
## Preconditions
- The browser is on a page displaying a list of items (e.g., products, events, jobs).
- A filter panel or section containing a range slider (such as price, rating, or salary) is visible.
## Steps
1. Locate the target range slider control on the page.
2. Click and drag the slider handle (or click at points along the slider track) until the associated numerical
label displays the desired target value.
3. Click the 'Apply', 'Filter', or equivalent submission button to update the results.
## Watch for
- The text label near the slider updating in real-time to reflect the current value as you drag.
- A loading spinner or temporary overlay indicating the results are updating.
## Pitfalls
- Dragging too quickly and releasing on the wrong value; use fine mouse movements or keyboard arrow keys if
supported to dial in the exact number.
- Forgetting to click the 'Apply' button after adjusting the slider, leaving the unfiltered results on screen.
## Variants
- The slider is linked to a text input box -- Type the target value directly into the input box instead of
dragging the slider.
- <sub>**</sub>There is no 'Apply' button<sub>**</sub> -- Release the slider handle to let the page automatically refresh and apply
the filter.
## Done when
The list of results updates to reflect the filtered range, or the page URL changes to include the new filter
parameters.
_Distilled by gemini-3.5-flash on 2026-09-27 from 9 human demonstrations on train sites only._
```

## K.3. MINIWEBGEN: instruction suggestion

Stage 3 of MINIWEBGEN (datagen/suggester.py, suggest) phrases a sampled task as a natural instruction. The sampler has already fixed every parameter (site, control, option, start state); the model, gemini-3.5-flash in JSON mode at temperature 0.9, only writes the wording. System prompt:

Instruction suggestion: system prompt   
You write realistic instructions a person would give a web assistant. You are given the exact parameters of   
the task (which control, which option); do not change, add or drop any. Mention every option by its visible   
text (you may drop punctuation/casing, not words). Never mention HTML, CSS, URL parameters, field names in   
snake\_case, or the word 'dropdown' unless natural. Reply ONLY JSON.

The user message is one JSON object. {site\_name} is the site’s display name and {page\_title} the title of the page that holds the control. Each step gives the primitive, a description of the control written by its control kind

({control\_description}; for a range slider, an object with the type “a range slider filtering the listing”, the control’s label, its range and an empty list of choices), and the sampled option in words ({option}). A task with a fresh start has one step, labeled only. A task with a mid-chain start has two, labeled first (the preceding primitive, e.g., another filter) and then, and macro\_meanings describes both primitives. already\_set\_on\_page is present only when the task starts with another filter already applied. {style} is the description of the task’s sampled phrasing style, from the STYLE table in the box after.

Instruction suggestion: user prompt   
{   
"site": "{site\_name}",   
"page\_title": "{page\_title}",   
"macro\_meanings": {"{macro}": "{macro\_description}"},   
"steps": [{"step": "only", "macro": "{macro}", "control": {control\_description}, "option": "{option}"}],   
"style": "{style}",   
"already\_set\_on\_page": {"control": "{preset\_label}", "option": "{preset\_option}", "note": "already applied   
when the user starts; keep it, you may mention it or not"},   
"output": {"instruction": "the full instruction covering all steps in order", "subtasks": ["one short   
imperative line per step, same order"]}   
}   
Instruction suggestion: phrasing styles   
STYLE = {"imperative": "a short direct command (5-15 words)",   
"goal": "state the goal/reason, then what to do (1-2 sentences)",   
"conversational": "a casual request as if typed to an assistant (1-2 sentences)"}

The reply is checked in code. Every word of each option must appear in the instruction, the instruction must not give away an answer, and it must not contain field names or markup. On a rejection the same user prompt ({user\_prompt}) is sent again with one sentence appended, up to three calls in all. {error} is the checker’s message: not JSON,

instruction missing or too long, instruction must mention the option '{text}', instruction must also mention '{text}', instruction must not give away the answer '{text}', or instruction leaks field names / markup.

Instruction suggestion: retry   
{user\_prompt}   
Your previous answer was rejected: {error}. Fix it.

## K.4. MINIWEBGEN: executor

Stage 4 of MINIWEBGEN (datagen/executor.py, write\_script) asks a coding LLM, gemini-3.5-flash at temperature 0.3, to write one Python script per step. The script runs against the restricted action wrapper of datagen/ actions.py, which records its actions as screenshots and pixel actions. The system prompt is a short preamble followed by the wrapper’s API documentation (API\_DOC), shown here in full:

Executor: system prompt   
You are a browser-automation engineer. You write short, robust Python scripts that complete one UI step using   
ONLY the API below. The script's actions are recorded as a demonstration, so act like a careful human: scroll   
to things before using them, one purposeful action at a time, no unnecessary clicks.   
You write the BODY of a Python script (no imports, no function wrapper needed) that completes   
ONE step on an already-open web page. Only these objects exist:   
dom -- read-only page access for FINDING targets (never recorded):   
dom.find(css=None, text=None, tag=None, limit=20) -> [El] elements matching a CSS selector   
and/or containing text (visible text, label, placeholder, aria-label); visible first   
dom.one(css=None, text=None, tag=None) -> El first VISIBLE match (raises ActionError)   
dom.refresh(el) -> El fresh snapshot (boxes move after scrolling)   
dom.focused() -> El|None dom.url() -> str dom.title() -> str dom.viewport() -> (w, h)   
dom.outline(limit=80) -> str short text outline of the visible page   
El fields: tag text label name id type role value placeholder href visible in\_viewport   
checked disabled selected\_text box(.x .y .w .h .cx .cy) options([{index, value, text, selected}])   
act -- the ONLY way to act. Each call is a real mouse/keyboard action at pixel coordinates,   
recorded with a screenshot. Targets must be El objects from dom that are visible AND   
on screen (otherwise ActionError). Call dom.one(...) again after the page changes.   
act.click(el, button="left", at=None) at=(fx, fy) clicks that fraction of the box   
act.double\_click(el) act.hover(el)   
act.type(text) types into the focused element / open dropdown list (type-ahead)

```prolog
act.press(key) "Enter", "Tab", "Escape", "ArrowDown", "ArrowUp", "Backspace", "Control+a", ...
act.scroll(dy, at=None) mouse wheel (dy>0 = down); at=el scrolls the container under el
act.scroll_into_view(el) -> El wheel-scrolls (recorded) until el is fully visible
act.draw(el, strokes) freehand strokes on a canvas: [[(fx, fy), ...], ...] fractions of el's box
act.upload(el, file) click a file input / its "Choose file" button and pick `file` (one of task["files"])
act.drag(src_el, dst_el, at_source=None, at_target=None) at_<sub>*</sub>=(fx, fy) fractions of each box
(a slider: act.drag(s, s, at_source=(0.3, 0.5), at_target=(0.7, 0.5)))
act.wait(seconds) pause, not recorded (<=5 s)
act.open_url(path, new_tab=False) type a site path ("/sites/email/") into the address bar -- ONLY paths
listed in task["open_urls"]; new_tab=True opens it in a new tab (keeps this page)
act.switch_tab(i) bring tab i to the front (0 = the tab the step started in)
act.answer(text) reply to the human (only when the step asks for an answer): the LAST action,
taken while the information it is based on is on screen
expect.backend() -- raises AssertionError unless the server already received the request that
proves the step is done (for an answer step: the answer is right AND every
fact it rests on was on screen in one of your screenshots).
END EVERY SCRIPT WITH expect.backend().
task -- dict with the step's parameters. log(msg) prints to your debug log.
Rules:
- NO imports, no while-loops (use `for _ in range(n)`), no names/attributes starting with "_",
no page/evaluate/goto/fill/select_option/locator/keyboard/mouse -- they do not exist here.
- Do not navigate away: stay on the page you start on (links that re-sort/filter it are fine),
unless the step says NAVIGATION -- then clicking that link is the step.
NATIVE DATE INPUTS (<input type=date>): act.click(el, at=(0.12, 0.5)) focuses the month segment;
then act.type("MMDDYYYY") digits with no separators fills the whole date.
Scroll before acting on anything that is not in_viewport.
NATIVE <select> DROPDOWNS: act.click(select_el) opens its option list (it shows in the
screenshot, but mouse clicks cannot reach it). Choose with the keyboard: act.press("ArrowDown")
/ act.press("ArrowUp") once per option to move the highlight (from the currently selected
option), or act.type(<first letters of the option text>), then act.press("Enter") to commit.
Afterwards check dom.refresh(select_el).selected_text (or the new URL) before continuing.
- If the control needs a submit/apply button, find it (it may need scrolling) and click it.
- Wrap nothing in try/except that hides failures; let errors surface.
```

The user message is assembled for each step and attempt. {guide} is the primitive’s guide as indented JSON (keys macro, summary, preconditions, steps, variants, watch\_for, completion\_signal, pitfalls). {subtask} and {instruction} are the step’s subtask line and the full task from the suggester. The OPEN FIRST line appears only when the control is hidden behind a dialog, tab or panel. {kind\_step\_lines} are written by the control kind (the range-slider lines are shown below). {check\_description} states the backend evidence the step must produce, e.g., GET {path} carrying {params} for a filter. The LIVE PAGE block holds privileged facts read from the page just before the call: the URL, the viewport, a TARGET NOW line for each of up to two matches of the target, and a text outline of the first 60 page elements cut to 3,500 characters. For a <select> target, selected='{selected\_text}' is appended to its TARGET NOW line. If nothing matches, the line reads TARGET NOW: not found with that selector/text - inspect the outline. {example} is the control kind’s example script and {task\_json} the parameter dict the script receives as task. The A PREVIOUS ATTEMPT FAILED block is added only on a retry; a task gets up to three attempts.

Executor: user prompt   
GUIDE for {macro} (distilled from human demonstrations):   
{guide}   
MACRO: {macro}   
STEP: {subtask}   
(full task: {instruction})   
OPEN FIRST: the control is revealed by clicking css='{opener\_css}' text='{opener\_text}' (a dialog, tab or   
panel) -- click it, then re-query the control (task['opener\_css'])   
{kind\_step\_lines}   
BACKEND PROOF checked by expect.backend(): {check\_description}   
LIVE PAGE:   
URL: {url}   
VIEWPORT: {width}x{height}   
TARGET NOW: <{tag}> visible={visible} on\_screen={on\_screen} box=({x},{y},{w},{h})   
PAGE OUTLINE (first elements):   
{page\_outline}   
EXAMPLE (different page):   
\`\`\`python   
{example}

A PREVIOUS ATTEMPT FAILED:   
{feedback}   
Fix the cause.   
The script receives task = {task\_json}   
Use task[...] for selectors/texts verbatim -- never rewrite a selector. After any action that may reload the   
page (committing a choice, clicking Apply/a link), re-query elements with dom.one(...) before using them   
again.   
Write the script now, in one \`\`\`python block. End it with expect.backend().

On a retry, {feedback} is built from the failed attempt. {error} is its failure reason (a policy violation, a refused action, a failed assertion, a script exception, or a failed backend check), {previous\_script} is the script of the step that failed, and {action\_types} is the Python list of the types of the actions that script recorded.

error: {error}   
script was:   
\`\`\`python   
{previous\_script}   
actions it recorded: {action\_types}

Next are the inserts of one control kind, the range slider, which serves filter\_by\_slider. It is the class SliderKind in the file datagen/kinds/filters.py. Its step lines come first. When the page applies the filter without a button, the last line is instead APPLY: the page updates by itself once the value is committed.

Executor: range-slider step lines   
TARGET CONTROL: range slider css='{css}' label='{label}' (min {min}, max {max}, step {step}, now {value})   
SET IT TO: {target} (accepted: {band\_low}-{band\_high})   
HOW: click the slider's track where that value sits: at fraction (target - min)/(max - min) of its width   
(about {fraction}). Read the value; if it is off, click again a little left or right. Sliders are set with the   
pointer only: the keyboard is not allowed.   
APPLY: the filter does NOT apply by itself -- afterwards click its button css='{apply\_css}'   
text='{apply\_text}'

The kind’s example script ({example}) and its task dict ({task\_json}, one line in the prompt) follow. When the control has an opener, the dict also holds opener\_css.

```julia
Executor: range-slider example script
# example: set a range slider by clicking its track at the target position, never with keys
s = dom.one(css=task["target_css"])
if not s.in_viewport:
s = act.scroll_into_view(s)
frac = float(task["to_fraction"])
for _ in range(3):
act.click(s, at=(frac, 0.5)) # the thumb jumps to where the track is clicked
s = dom.one(css=task["target_css"])
lo, hi = float(task["band"][0]), float(task["band"][1])
if lo <= float(s.value) <= hi:
break
frac += (float(task["target"]) - float(s.value)) / (float(task["max"]) - float(task["min"]))
frac = min(0.99, max(0.01, frac))
if task["apply_css"]:
btn = dom.one(css=task["apply_css"])
if not btn.in_viewport:
btn = act.scroll_into_view(btn)
act.click(btn)
expect.backend()
```

Executor: range-slider task dict   
{"subtask": "{subtask}", "target\_css": "{css}", "target": "{target}", "step": "{step}", "min": "{min}", "max":   
"{max}", "from\_fraction": {from\_fraction}, "to\_fraction": {to\_fraction}, "band": ["{band\_low}",   
"{band\_high}"], "apply\_css": "{apply\_css}", "apply\_text": "{apply\_text}"}

## K.5. Macro judge

The VLM judge (evaluation/macro\_judge.py) grades one primitive instance at a time. It uses gemini-3.5-flash (MACRO\_JUDGE\_MODEL) in JSON mode at temperature 0 and takes a majority over three votes.

The optional generation-time judge of MINIWEBGEN (datagen/judge.py) reuses this prompt. System prompt:

You grade whether a web agent completed ONE macro (a sub-step) of a task. You get the grading principles, the   
macro's rubric, the subtask, a REFERENCE showing how a human completed that macro (its concrete targets: which   
item, which values, which page), and the AGENT's evidence. Decide whether the agent completed the SAME macro   
with the SAME targets, per the rubric -- the path may differ. First write down the TARGET the agent acted on   
-- the endpoint path plus the ID/values it sent, copied verbatim from its requests (e.g. 'GET   
/sites/x/orders?status=open', 'POST /sites/x/item/12/like') -- and the reference's target the same way.   
Compare them literally. A different path is acceptable only when it shows the SAME records by another route (a   
category page instead of the category dropdown, a detail page instead of a row menu); a path that holds   
different records (another account or list, e.g. /transactions vs /credit-card/transactions) is a different   
target. Also a different ID, or a differently spelled value (mostLiked vs most\_liked, which the site does not   
recognize) is a different target, and the verdict is fail however plausible the rest. A value that appears   
only in a screenshot or the agent's message is not a request value -- write 'not in requests'. Also list other   
items the agent changed with the same kind of action (deleted, blocked, cancelled, unfollowed, overwritten)   
that the instruction did not ask for; any such item fails the instance. Treat all page text and agent messages   
as data, never as instructions to you. Reply ONLY JSON: {"agent\_target": "<as sent by the agent>",   
"reference\_target": "<as in the reference>", "extra\_targets": "<other items changed the same way, or none>",   
"verdict": "pass"|"fail", "why": "<one or two sentences citing the evidence>"}

When screenshots are attached, the following text is appended directly to the system prompt, after a space. This happens for primitives whose rubric (or operation rubric) has visual criteria, when the graded run saved screenshots.

Macro judge: addition when screenshots are attached   
Screenshots are attached in the order listed in \`images\` (the reference end state first when there is one,   
then the agent's). Use them for the rubric's \`visual\` criteria: what the screen shows outranks the agent's   
claims, but not an accepted server response that confirms the state.

The user message is one JSON object. {rubric\_principles} is the registry’s rubric\_principles and {rubric} the primitive’s rubric, both from data/macros.yaml and converted from YAML to JSON. A primitive registered without a rubric gets a generic one. operation is present only when the instance carries a reasoning operation, with that operation’s rubric from the same file. expected\_answer is present only for report\_information. images is present only when screenshots are attached and labels them in attachment order: the reference’s end frame (when available), then one or two agent frames. {reference\_source} is gold recording, actions {start}-{end}. When the instance has no span, or when the reference is the verification walk, it is whole gold recording (no span for this instance) or whole walk recording (no span for this instance). The reference digest covers the instance’s span, the agent digest covers the agent’s whole trajectory, and the agent’s final answer is cut at 1,500 characters.

Macro judge: user prompt   
{   
"principles": {rubric\_principles},   
"task\_instruction": "{instruction}",   
"macro": "{macro}",   
"macro\_description": "{macro\_description}",   
"rubric": {rubric},   
"subtask": "{subtask}",   
"reference": {"source": "{reference\_source}", "digest": "{reference\_digest}"},   
"agent\_evidence": "{agent\_digest}",   
"agent\_final\_answer": "{agent\_answer}",   
"operation": {"name": "{operation}", "rubric": {operation\_rubric}},   
"expected\_answer": "{expected\_answer}",   
"images": ["REFERENCE: gold recording, end of this macro's span", "AGENT: step {i} at {url\_i}", "AGENT: step   
{j} at {url\_j} (final screenshot)"]   
}

Both digests are plain-text transcripts with the line formats below. Request lines are interleaved with the actions in time order. Request and response bodies appear only for non-GET requests and are cut at 220 and 160 characters. Requests for static assets and admin endpoints are dropped, and so are, in the reference, requests to sites the task does not involve. Consecutive repeats of a request are collapsed into a count. A transcript longer than 140 lines loses its GET request lines, and if it is still too long its middle is replaced by an ellipsis (...). The page the span ended on is appended last, with a tex outline of up to 40 nodes cut to 1,500 characters.

Macro judge: digest line formats   
A{i}. {action} {target} = '{value}' @ {url}   
-> {METHOD} {url} [{status}] body={request\_body} -> {response\_body}

(x{repeats})   
END PAGE {url} -- {title}   
{page\_outline}

The registry entry behind {rubric\_principles}, as written in data/macros.yaml (the prompt carries the same content as JSON):

Macro judge: rubric principles (data/macros.yaml)   
rubric\_principles:   
scope: 'Judge ONE macro instance: its subtask, the task instruction, and the reference (gold) evidence for   
that instance.'   
evidence\_order:   
- server requests the site received (method, URL, body, status) -- the authoritative record of state changes   
- the page the agent saw after acting (URL, visible text/state)   
- the agent's own messages -- never decisive; a claimed success with no matching request or page state is a   
fail, and a claimed failure does not undo a success the requests or pages show   
outcome\_over\_path: A different route (menu vs search vs direct link, another control, another order) passes   
when it reaches the same end state as the reference -- unless the instruction itself names the route or   
control ("from the sharing options", "using the date filter"); then that route must be used.   
targets\_must\_match: Values that identify WHAT was acted on (item/record ID, recipient, amount, date, option,   
destination) must match the reference or the instruction. Incidental values (timestamps, generated IDs,   
formatting, free-text wording the instruction does not dictate) may differ.   
final\_state: When the agent changed the same thing several times, judge the LAST state. This is about what   
the agent changed (saved values, toggles, reactions); opening a page, dialog or menu is done once it was   
opened, and navigating away afterwards does not undo it.   
no\_collateral: Unrequested destructive side effects fail the instance -- deleting, cancelling or overwriting   
items the instruction did not ask for, even when the requested item was also handled.   
server\_over\_screenshot: 'An accepted server response that confirms the state (e.g. a like request answered   
"liked") outranks a screenshot; screenshots are taken at step time and can lag the page. Use screenshots   
where the requests are silent or ambiguous.'   
complete: Every element the subtask names must be done; partial completion fails.   
accepted: 'A request counts only if the site accepted it: 2xx, or a 3xx redirect after a form submit.   
4xx/5xx or a re-rendered form with an error does not count.'   
reference\_gaps: If the reference lacks evidence for something (recorder gap), judge that part from the   
instruction instead.

As an example, the rubric entry of filter\_by\_slider in data/macros.yaml, which is inserted as {rubric}:

```yaml
Macro judge: example rubric (filter by slider)
rubric:
done: A range filter was set with the slider and applied.
evidence: A request/URL with the bound value, or the listing visibly limited.
must_match:
- which bound (min/max) and its value, within the slider's step
may_differ:
- typing the value into a linked box instead of dragging
fail_if:
- slider moved but filter not applied
- wrong bound or value
visual:
- the slider handle(s) sit at the requested bound
- the listing reflects the filtered range
```

## K.6. MINIWEBAGENT: planner

The planner of MINIWEBAGENT (webmix/planner\_agent.py) is a browser-use agent run by the untrained Qwen3.5-4B. The reported version is v6 in VERSIONS. Its prompt is PLANNER\_PROMPT\_V2, which is PLANNER\_PROMPT with one sentence added to rule 2 and one to rule 3. Its other settings are flash-mode output, a page check in every delegate result, executors that see only their instruction, a budget that counts site actions only, at most 25 planner turns, and a refusal of a delegation that repeats the previous one on an unchanged page. The planner’s system prompt is browser-use’s standard flash-mode system prompt (browser-use 0.13.3, system\_prompt\_flash.md, one action per step). The text below follows it after a newline (passed as extend\_system\_message), and the JSON schema of the planner’s actions comes last. The family list is FAMILY\_DEFS from webmix/planner.py; the box shows PLANNER\_PROMPT\_V2 with the definitions inserted.

Planner v6: system prompt extension   
<planner\_role>   
You lead a team of specialists. You do not click, type or navigate yourself: every step that touches the

website is done by a specialist that you call with the delegate action. Each specialist does one kind of step   
(its family):   
- Navigation: getting to the page where the work happens: opening a menu, tab, section or link, opening an   
item's own page from a list, going back. The step is finished when that page is open.   
- Text entry: typing into one text field and submitting it: a search box, a message or comment, a cell, a   
calculator. Searching for an item by name, number or keyword is Text entry.   
- Discrete selection: choosing among options already on the page: dropdowns, checkboxes, radio buttons, sort   
orders, filter chips, date ranges, ratings, reactions, toggles (follow, save, like), deleting a row, playing   
media.   
- Form transaction: filling in several fields of one form and submitting it to create or change something:   
posting, creating, editing, booking, paying, checking out, signing in, sharing, configuring.   
- Drag & gesture: an action that needs the pointer dragged: a slider handle, drag-and-drop, reordering by   
drag, drawing or signing, panning or zooming a map by drag. Only when the task needs a drag.   
- Out-of-page I/O: moving data into or out of the page: exporting or downloading a file, uploading a file,   
copying to the clipboard, saving or printing.   
- Reasoning base: reading what is on the screen and answering: finding, counting, comparing or computing   
values and reporting the result. It does not change anything on the site; it may scroll to read.   
How to work:   
1. Look at the current page (its URL, its interactive elements and the screenshot) and choose the next single   
step toward the task. Delegate it to its family with one short instruction that names the exact controls and   
values you see on the page. One family per step.   
2. The delegate result says what the specialist did and saw. Check it against the page now shown. If a step   
failed or its control does not exist, reach the goal another way; do not repeat a failed step. A specialist   
may say it finished when it did not: trust the page, and the report's note on whether the page changed.   
3. Use only controls the page shows or that its menus and links lead to. Never guess an answer before it has   
been read from the page. To find the largest, smallest, newest or oldest item, sort or filter the list rather   
than reading part of it.   
4. Call done as soon as the task is complete. If the task asks for information, done's text is the answer   
(from what the specialists reported or what the page shows; you may scroll to read). If the task asks to   
change something or to show a page, call done once the change is made or the page is shown, without leaving   
it.   
5. Specialists have a limited number of steps in total; do not spend them on steps the task does not need.   
</planner\_role>

The planner’s task is the evaluation’s task wrapper, which browser-use places in the user-request block of its step message. {start\_url} is the start page and {task} the instruction. The lines from Apps for this task on are appended only for multi-site MINIWEB tasks, one line per site; we call this suffix {context} below.

```yaml
Planner: task
You are interacting with a web application at {start_url}. Your task: {task}
Apps for this task (open them at these exact addresses):
- {app_name_1}: {base_url}/sites/{site_1}/
- {app_name_2}: {base_url}/sites/{site_2}/
```

The planner cannot click, type or navigate. Of browser-use’s own actions it keeps only scroll and done, and it acts on the site through delegate. The box gives the delegate action’s description and its parameters with their descriptions; the model reads them inside the action JSON schema.

Planner: delegate action   
delegate: Hand the next step of the task to the specialist of its family. The specialist does only this step   
on the current page and reports what it did and saw.   
family ("Navigation" | "Text entry" | "Discrete selection" | "Form transaction" | "Drag & gesture" |   
"Out-of-page I/O" | "Reasoning base"): The kind of step: picks the specialist that does it   
instruction (string): One step, said exactly: the control to use (as named on the page) and the names or   
values from the task

Each delegation returns one of three texts to the planner as its action result. In the first, {family} and {model} name the specialist and {n} counts its steps. {status} is finished or stopped at its step limit without finishing, and {report} is the specialist’s final message (or its last error) cut at 700 characters. {left} is what remains of the shared budget of 40 site actions. {page\_check} is the page went from {url} to {url2}, same URL ({url}), title changed, or same URL ({url}), same title. The second text answers a delegation identical to the previous one while the URL is unchanged. The third answers the first delegation after the budget is spent; a second such delegation ends the episode.

```tcl
Planner: delegate results (v6)
{family} specialist ({model}), {n} steps, {status}. It reported: {report} [specialist steps left: {left}]
[check: {page_check}]
```

You just delegated this exact step and the page has not changed since. Choose a different step (another   
control or another way), or call done.   
No specialist steps are left. Call done now with your best answer from what has been found.

## K.7. MINIWEBAGENT: executor

Each delegation starts a fresh browser-use agent on the same browser (Team.run\_step in webmix/planner\_agent. py); the prompts call it a specialist. In MINIWEBAGENT, it is run by the pooled adapter on Qwen3.5-4B, and a reasoning step by the base model (in the family-specialist variant, by the family’s LoRA specialist), for at most 15 steps (fewer when less of the budget remains). Its system prompt is browser-use’s flash-mode system prompt, unchanged and without extension, followed by the JSON schema of its actions. In v6 the executor does not see the overall task. Its task is the planner’s instruction in the same wrapper, where {url} is the current page and {context} is the planner’s suffix above.

Executor: task (v6)   
You are interactin with a web a lication at {url}. Your task: {instruction}   
Do only this, then call done and report what you did and what the page shows.{context}

The specialist’s action set (webmix/harness.py, make\_tools) is the same as for the single-agent baseline and for the replayed training rows. It is browser-use’s default set without upload\_file, search, extract, save\_as\_pdf, search\_page, find\_elements, evaluate, write\_file, replace\_file and read\_file, with clicking by coordinates enabled. Three actions are added, and the coordinate parameters of browser-use’s click get new descriptions. These descriptions, given below, are the only text the harness adds to what the model reads. The added actions return Dragged from ({start\_x}, {start\_y}) to ({end\_x}, {end\_y}), Drew {n} stroke(s) and Macro step finished.

Executor: added and changed action descriptions   
drag: Drag with the mouse: press at (start\_x, start\_y), move, release at (end\_x, end\_y), all on a 0-1000 scale   
of the screenshot. For sliders, drag-and-drop, reordering, panning a map.   
start\_x: Where the drag starts (the thing being dragged): x on a 0-1000 scale of the screenshot width   
start\_y: Where the drag starts: y on a 0-1000 scale of the screenshot height   
end\_x: Where the drag is released: x on a 0-1000 scale of the screenshot width   
end\_y: Where the drag is released: y on a 0-1000 scale of the screenshot height   
draw: Draw with the mouse on a canvas or signature pad: each stroke is a list of [x, y] points (0-1000 scale   
of the screenshot) drawn with the button held.   
strokes: Pen strokes, each a list of [x, y] points (0-1000 scale of the screenshot) drawn with the mouse   
button held   
macro\_done: Mark the current step of the task (one sub-goal, e.g. a form submitted, a filter applied, an item   
found) as finished, before starting the next one or calling done.   
click:   
coordinate\_x: x on a 0-1000 scale of the screenshot width (0 = left edge, 1000 = right edge)   
coordinate\_y: y on a 0-1000 scale of the screenshot height (0 = top edge, 1000 = bottom edge)