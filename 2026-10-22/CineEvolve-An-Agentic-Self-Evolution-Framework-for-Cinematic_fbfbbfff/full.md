# CineEvolve: An Agentic Self-Evolution Framework for Cinematic Multi-Shot Video Generation

Jingzhe Zhang<sup>#1</sup> Peixuan Zhang<sup>#1,5</sup> Genying Li<sup>1</sup> Bing Ma<sup>5</sup> Shuchen Weng<sup>2,4</sup> Si Li<sup>∗1,6</sup> Boxin Shi<sup>2,3</sup>

<sup>1</sup>School of Artificial Intelligence, Beijing University of Posts and Telecommunications

<sup>2</sup>State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University

<sup>3</sup>National Engineering Research Center of Visual Technology, School of Computer Science, Peking University

<sup>4</sup>Beijing Academy of Artificial Intelligence <sup>5</sup>Tencent

<sup>6</sup>Beijing Key Laboratory of Multimodal Data Intelligent Perception and Governance

{zhangjingzhe,pxzhang,lisi}@bupt.edu.cn {shuchenweng,shiboxin}@pku.edu.cn

![](images/cf59987dac9b986279fd04436733c0b1cb8a1ea03c5a794ea0b7c4bb5eb816f0.jpg)  
Figure 1. Illustration of our proposed CineEvolve for multi-shot video generation. (a) Existing visual-based and agentic-based methods suffer from narrative reasoning deficiency and static workflow rigidity. (b) CineEvolve co-evolves Director and Critic in a shared vision-language policy, enabling cognitive synergy and feedback refinement.

## Abstract

Cinematic multi-shot video generation requires both narrative planning and iterative refinement. Existing methods either lack narrative reasoning or rely onfeed-forward workflows without policy updates. In this paper, we propose CineEvolve, an agentic self-evolution framework for cinematic multi-shot video generation that unifies a Director and a Critic within a single vision-language policy. Emulating real-worldfilmmaking that couples iterative direction with critical feedback, we train CineEvolve via alternating reinforcement learning, where the Director surfacesflaws to sharpen the Critic, and the updated Critic guides the Director’s targeted revisions. To stabilize co-evolution, we introduce Evolutionary GRPO (E-GRPO), leveraging historical anchoring and revision safeguarding to mitigate advantage

inflation. Additionally, we contribute CineTraj, a 40K cinematic trajectory dataset spanning creation, critique, and revision. Extensive experiments demonstrate that CineEvolve outperforms existing approaches. Our code will be available at this url.

## 1. Introduction

Video generation is rapidly advancing from isolated single clips toward long-horizon cinematic storytelling [18]. In visual storytelling, narratives unfold through sequences of interconnected shots capturing dynamic events over shifting viewpoints and scenes. Such multi-shot sequences demand cross-shot consistency to preserve characters and environments, alongside inter-shot coherence to ensure narrative progression [2, 32]. Satisfying these constraints requires coupling narrative planning with visual synthesis.

To address these challenges, existing visual-based methods use video generation models to produce shots, relying on extended attention mechanisms or historical conditioning [17, 27, 28]. In contrast, agentic-based methods use multimodal foundation models to plan the story structure and invoke external visual tools to render each shot [10, 11]. Despite these advancements, visual-based methods lack the narrative reasoning to rectify semantic errors, while agenticbased methods typically follow feed-forward workflows or use feedback for inference-time refinement, without updating their policies through training (Fig. 1 (a)).

To overcome this static limitation, a natural solution is to emulate real-world filmmaking, where directors iteratively refine defective shots through trial-and-error reflection. However, visual closed-loop refinement poses practical challenges: external tools may subtly deviate from the narrative plan, and separate evaluators often struggle to assess these flaws without direct access to the creator’s intent. Because cinematic planning and defect diagnosis share an understanding of cinematic grammar, we formulate multi-shot creation as an agentic self-evolution process with two roles: a Director for initial creation and targeted revision, and a Critic for defect diagnosis. By unifying both roles within a single vision-language policy, the model can introspect against its own narrative goals, fostering cognitive synergy across both capabilities while enabling systematicfeedback refinement to perfect multi-shot narratives (Fig. 1 (b)).

In this paper, we present CineEvolve, an agentic selfevolution framework that uses external visual generation tools for cinematic multi-shot video synthesis through a unified vision-language policy acting as both Director and Critic. Following the supervised warm-up, the unified policy evolves through alternating reinforcement learning, where the Director exposes failure cases to train the Critic, while the updated Critic provides diagnostic feedback to guide the Director. Specifically, the Critic is optimized for dy namic storyboard rollouts using composite critic rewards to sharpen its diagnostic precision. Meanwhile, the Director is trained across initial creation and targeted revision, guided by the updated Critic’s self-evaluative quality scores alongside narrative coherence and tool-executability rewards. To mitigate intra-group advantage inflation during co-evolution, we propose Evolutionary GRPO (E-GRPO), which calibrates advantage estimation via revision safeguarding and historical anchoring to ensure steady improvement.

To support training and evaluation, we curate CineTraj, a cinematic trajectory dataset spanning initial creation, defect critique, and targeted revision, yielding 30K samples for supervised warm-up and 10K for alternating reinforcement learning. Our contributions are summarized as follows:

• We propose CineEvolve, an agentic self-evolution framework that unifies the Director and Critic in a single policy, driving multi-shot video generation via co-evolution.

• We design critic and director rewards for closed-loop optimization, alongside E-GRPO to mitigate advantage inflation via revision safeguarding and historical anchoring.

• We develop a trajectory curation pipeline and construct CineTraj, which contains 40K trajectory-derived cinematic samples for multi-shot generation and refinement.

## 2. Related Work

## 2.1. Multi-Shot Video Generation

Driven by the advancements of AI-assisted filmmaking, video synthesis is expanding from isolated clips toward multi-shot storytelling. Multi-shot video generation broadly follows two paradigms: visual-based and agentic-based approaches. Visual-based methods enforce continuity by synthesizing consecutive shots within unified spatio-temporal latents (e.g., HoloCine [18], CineTrans [28], MultiShotMaster [27]) or conditioning on preceding visual frames (e.g., OneStory [2], ShotStream [17]). While preserving visual persistence, they struggle to diagnose or rectify semantic narrative flaws. Conversely, agentic methods leverage multimodal foundation models to schedule generative tools via structured planning $( e . g .$ , STAGE [32], ViMax [11]) or cinematic retrieval (e.g., FilMaster [10]). However, their feed-forward execution cannot adaptively rectify execution failures. Although CineCrew [6] introduces diagnostic feedback for rule-based refinement, it lacks policy-level learning to evolve from errors. In contrast, CineEvolve formalizes multi-shot synthesis as a closed-loop trial-and-error process, unifying Director and Critic within a single policy that co-evolves via reinforcement learning.

## 2.2. Agentic Self-Evolution Mechanisms

Agentic self-evolution empowers autonomous agents to continuously improve their capabilities via experiential trial-and-error and environmental feedback. The vast majority of existing works confine self-evolution to nonvisual domains, developing reflective strategies for textual reasoning, symbolic manipulation, and algorithmic tasks through external memory updates (e.g., ReasoningBank [20], ACE [34], RMM [25]), prompt optimization (e.g., GEPA [1], ERM [29]), self-improving code synthesis (e.g., Darwin Gödel Machine [31], Gödel Agent [30]), or synthetic adversarial reasoning (e.g., R-Zero [9], PasoDoble [35]). Recently, a handful of pioneering studies have begun exploring self-evolution in multimodal settings; however, these efforts remain largely limited to isolated 2D perception or static image interactions (e.g., VisPlay [8], Agent0-VL [16], ECHO [13], JarvisEvo [15]). To our knowledge, CineEvolve represents the first endeavor to extend agentic self-evolution into multi-shot video generation, coupling creative direction and diagnostic critique to achieve iterative self-improvement in complex cinematic synthesis.

## 3. Dataset

Generating multi-shot videos demands global visual planning and reflection across shot transitions. Yet, existing datasets predominantly provide text-to-video pairs, lacking corrective demonstrations that are critical for models to autonomously plan and revise. To address this limitation, we propose CineTraj, a dataset that captures the trajectory of multi-shot video generation (Fig. 2). We use Gemini 3.1 Pro [7] as the teacher VLM, Director, and Critic, with the teacher Critic also annotating refreshed replay data.

Scene curation. To construct high-quality scenes with coherent narrative arcs, we collect cinematic footage from Condensed Movies [4] and CineBench [33] via a four-step curation pipeline: (i) Deduplication: performing movielevel filtering to eliminate overlapping titles between the two sources; (ii) Boundary detection: segmenting shot transitions across videos using TransNetV2 [24]; (iii) Shot grouping: grouping consecutive shots under native scene boundaries via VLM-assessed narrative continuity to preserve complete context; and (iv) Quality screening: removing black borders and watermarks [26], filtering out clips with low LAION aesthetic scores [22], and pruning incoherent shot sequences via Gemini-3.1-Pro [7]. This yields a high-fidelity corpus of verified source scenes with coherent visual storytelling.

Initial generation. To bridge narrative intent with visual layout, we construct initial generation trajectories by transforming curated source scenes into structured production plans and visual storyboards. Taking each curated scene S as input, a teacher VLM first annotates a global narrative prompt x alongside ordered shot descriptions $\mathcal { D } _ { \mathrm { s h o t } }$ detailing visual events and character dynamics. Conditioned on $( x , \mathcal { D } _ { \mathrm { s h o t } } )$ , a teacher Director then generates a hierarchical plan P: (i) a global plan outlining narrative progression; and (ii) shot-level specifications designating camera framing, subject attributes, and tool-call prompts. Subsequently, an external visual generation model [5] executes these tool calls to render the multi-shot storyboard frames $B = ( b _ { j } ) _ { j = 1 } ^ { N }$ , where N is the number of shots. This process outputs structured tuples $( x , P , B )$ , which serve as foundational demonstrations for defect critique and targeted revision.

Trajectory construction. To supervise defect diagnosis and iterative refinement, we synthesize verified closed-loop trajectories by pairing generation rollouts with teacher-guided critique and correction. Given the source scene S, shot descriptions $\mathcal { D } _ { \mathrm { s h o t } }$ , and initial generation tuple $( x , P , B )$ this stage proceeds in three steps: (i) Defect diagnosis: the teacher Critic compares the initial storyboard B against S given $( x , \mathcal { D } _ { \mathrm { s h o t } } )$ , focusing on cross-shot consistency and inter-shot coherence, and outputs a structured diagnosis $c = ( \mathcal { P } , \mathbf { d } , \{ \hat { \ell } _ { j } \} _ { j \in \mathcal { P } } )$ that details shot-level defect locations, textual issues, and severity ratings $( i . e .$ , minor, moderate, or severe); (ii) Directorial revision: conditioned on $( x , P , B , c )$ the teacher Director formulates a targeted revision plan $P ^ { \prime }$ with shot-level actions $( i . e . ,$ keep, regenerate, or edit) and executes visual tool calls to render the revised storyboard $B ^ { \prime } ;$ and (iii) Outcome verification: the teacher Critic re-evaluates the entire revised storyboard B<sup>′</sup>. We retain the demonstration only if its severity-weighted defect score strictly decreases, discarding failed or regressive attempts. Consequently, this process outputs successful closed-loop trajectory instances $( x , P , B , c , P ^ { \prime } , B ^ { \prime } )$ , which populate the demonstration corpus CineTraj.

![](images/0e6793fa366c43fd87d7cae482e6619ca32cd659d40210e1235b3831eb5f9748.jpg)  
Figure 2. Overview of the CineTraj construction pipeline. (a) Movie scenes are curated through deduplication, shot segmentation and grouping, and quality filtering. (b) The teacher Director converts annotated stories and shot descriptions into production plans and initial storyboards. (c) Teacher-guided critique, revision, and re-evaluation produce closed-loop trajectories.

Data Summary. The constructed closed-loop trajectory dataset CineTraj provides structured supervision across three functional training objectives: (i) defect critique instances $( x , B , c )$ for training the Critic to predict shot defects and severity ratings; (ii) initial creation instances $( x , P , B )$ for supervising directorial global planning and tool execution; and (iii) targeted revision instances $( x , P , B , c , P ^ { \prime } , B ^ { \prime } )$ for training directorial defect-driven repair. We balance these three tasks with a sample distribution ratio of 4 : 2 : 4, placing greater emphasis on critique and refinement. In total, CineTraj comprises 30,000 verified trajectories for super vised policy warm-up and 10,000 diverse trajectories curated for alternating reinforcement learning<sup>1</sup>. Furthermore, we curate an independent benchmark of 100 manually verified movie sequences with strong narrative coherence, strictly excluded from all training data.

## 4. Method

This section begins with an introduction to the training and inference paradigm (Sec. 4.1). Then, we detail the Critic policy optimization to provide fine-grained visual critique via composite critic rewards (Sec. 4.2). Next, we elaborate on the Director policy optimization to guide generative planning via multimodal director rewards (Sec. 4.3). Finally, we present Evolutionary GRPO (E-GRPO) to stabilize alternating updates and promote mutual improvement between the Director and Critic (Sec. 4.4).

## 4.1. Training and Inference Paradigm

Stage I: Supervised Policy Warm-up. Directly applying reinforcement learning across multi-shot planning and diagnosis yields intractable action exploration and sparse rewards. To establish a stable behavioral prior, we cold-start the student policy $\pi _ { \theta }$ on CineTraj, which contains expert demonstration trajectories collected from the proprietary teacher Director and Critic. The supervised warm-up objective is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( z , y ) \sim \mathcal { D } _ { \mathrm { S F T } } } \left[ \log \pi _ { \theta } ( y \mid z ) \right] , } \end{array}\tag{1}
$$

where $\mathcal { D } _ { \mathrm { S F T } }$ denotes the SFT demonstration subset of Cine-Traj, and $( z , y )$ represents a role-conditioned multimodal input and its corresponding ground-truth response. This stage yields the warm-up policy checkpoint $\theta _ { 0 }$ for subsequent evolutionary training.

Stage II: Alternating Reinforcement Learning. Starting from $\theta _ { 0 } ,$ , we co-evolve the student Critic and Director across K alternating rounds to transcend teacher imitation. In round $k \in \{ 0 , \ldots , K - 1 \}$ , starting from $\theta _ { k } ,$ , π<sub>θ</sub> is first optimized as the Critic via critic rewards, yielding an intermediate checkpoint $\theta _ { k , \mathrm { C } }$ . A frozen snapshot of $\theta _ { k , \mathrm { C } }$ then serves as an internal evaluator, providing self-evaluation rewards that are combined with external narrative coherence rewards to update $\pi _ { \theta }$ as the Director, producing $\theta _ { k + 1 }$ . This interactive cotraining under a shared policy $\pi _ { \theta }$ induces cognitive synergy: the Critic’s sharpened diagnosis informs and regularizes the Director’s generative planning. By feeding the updated Director’s rollouts back into $\mathcal { D } _ { \mathrm { C } , k + 1 }$ as dynamic diagnostic environments for the subsequent round, this paradigm forms an evolutionary flywheel.

Inference. At test time, the autonomous policy $\pi _ { \theta }$ executes an iterative generate-critique-revise loop without external guidance. Given a narrative prompt x, π<sub>θ</sub> first acts as the Director to formulate an initial multi-shot storyboard. It then switches to the Critic to diagnose this storyboard. Upon diagnosing defects, $\pi _ { \theta }$ alternates back as the Director to perform targeted shot-level revisions, iterating up to a maximum budget of $T _ { \mathrm { m a x } } = 1 0$ revision steps. Once approved, the final storyboard conditions the downstream video backbone to render the cinematic multi-shot video.

## 4.2. Critic Policy Optimization

Critic policy optimization aims to sharpen the shared policy under its Critic role, enabling it to deliver fine-grained visual critiques rather than scalar ratings (Fig. 3 (a)).

Critic formulation. Given the narrative prompt x and the candidate storyboard comprising rendered visual frames $B = ( b _ { j } ) _ { j = 1 } ^ { N }$ , the Critic evaluates cross-shot consistency and narrative coherence. The Critic output is structured into three components: defective shot indices ${ \mathcal P } ,$ corresponding defect descriptions d, and severity ratings $\hat { \ell } _ { j } \in$ {minor, moderate, severe} for each defective shot. For defect-free storyboards, the Critic outputs none for both defect claims and severity, yielding an empty defective set $\mathcal { P } = \emptyset$ . During training, we draw $G = 4$ Critic rollouts for each sample in the replay pool $\mathcal { D } _ { \mathrm { C } , k }$ , which is initialized from CineTraj, and compute learning signals against teacher annotations.

Critic rewards. To jointly encourage structural compliance, semantic fidelity, and severity calibration, we formulate a composite critic reward across three complementary objectives<sup>2</sup>: (i) Format compliance $F _ { \mathrm { C } } \in \{ 0 , 1 \}$ enforces output schema validity, where any parsing failure invalidates the rollout and directly zeroes out the reward $( r _ { \mathrm { { C } } } = 0 ) ;$ (ii) Semantic fidelity $M \in [ 0 , 1 ]$ uses an F1-style formulation to penalize both false alarms and missed shots, scoring how accurately predicted defect descriptions match reference semantic units via an LLM judge on correctly located shots; and (iii) Severity calibration $V \in [ 0 , 1 ]$ measures agreement on defect severity levels, strictly conditioned on a high-fidelity subset of correctly identified shots whose descriptions faithfully cover reference units without spurious claims. Therefore, for valid rollouts, the overall critic reward is formulated as:

$$
r _ { \mathrm { C } } = 0 . 1 0 F _ { \mathrm { C } } + 0 . 7 5 M + 0 . 1 5 V .\tag{2}
$$

Buffer refreshing. As the Director policy advances, its failure distributions shift, making static diagnostic data unrepresentative of its failure patterns. To track these emerging weaknesses, we dynamically refresh the replay pool $\mathcal { D } _ { \mathrm { C } , k }$ across rounds. After each Director optimization phase, the updated policy generates fresh storyboards, which are diagnosed by the external teacher model. We randomly retain 50% of the trajectories from the previous pool $\mathcal { D } _ { \mathrm { C } , k }$ and add 50% newly generated samples to form $\mathcal { D } _ { \mathrm { C } , k + 1 }$ . This refreshing mechanism maintains a fixed pool capacity while adapting the Critic to the Director’s latest failure modes.

## 4.3. Director Policy Optimization

Director policy optimization aims to train the shared policy under its Director role to generate coherent storyboards based on diagnostic feedback, using multimodal director rewards (Fig. 3 (b)).

![](images/30c4f9b4fc48f3a0ccc07b556ef06445b678222696dca4a8fb259cae4d5abadb.jpg)  
Figure 3. Overview of our proposed CineEvolve framework. (a) The Critic optimizes its diagnostic accuracy against reference defect annotations on an evolving pool of Director-generated storyboards. (b) The Director learns to draft and revise storyboards guided by self-evaluation rewards from the updated Critic, alongside narrative and visual feedback. (c) Revision safeguarding and historical anchoring are integrated to stabilize policy updates and prevent performance collapse during co-evolution.

Director formulation. We jointly optimize the shared policy for initial creation and targeted revision, as both tasks unify cinematic planning with tool execution. Given a narrative prompt x, initial creation generates a global plan with shot-level tool specifications. For iterative revision, the policy conditions on the story x, rendered storyboard B, and Critic diagnosis c to output a revision plan and actions that selectively keep, regenerate, or edit shots. In both tasks, a frozen visual backbone executes the dispatched generation and editing calls. Within round k, policy optimization follows a two-stage sequential schedule starting from $\boldsymbol { \theta } _ { k , \mathrm { C } }$ . The policy is first updated on initial creation; an intermediate frozen checkpoint of this updated policy then synthesizes parent storyboards to formulate the revision task, further optimizing the same policy into $\theta _ { k + 1 }$ . Throughout this process, the frozen snapshot $\boldsymbol { \theta } _ { k , \mathrm { C } }$ serves a dual role: it provides textual diagnoses to guide revision rollouts and self-evaluation rewards for $G = 4$ sampled trajectories per input.

Director rewards. To balance action executability with visual narrative quality, we formulate a composite multimodal director reward. Specifically, the reward unifies structural validity and visual storytelling across four complementary terms<sup>3</sup>: (i) Format compliance $F _ { \mathrm { D } } \in \{ 0 , 1 \}$ checks adherence to the structured output schema; (ii) Tool executability $T \in \{ 0 , 1 \}$ verifies tool-call executability and imagereference validity, where violating either $ { F _ { \mathrm { D } } }$ or T invalidates the rollout and yields zero reward $( r _ { \mathrm { D } } = 0 ) ;$ (iii) Critic quality $Q \in [ 0 , 1 ]$ uses the frozen Critic snapshot $\theta _ { k , \mathrm { C } }$ to assess shot-level visual defect severities, where globally reevaluating all shots during revision prevents repairing local flaws at the cost of overall quality; and (iv) Narrative coherence $H \in [ 0 , 1 ]$ employs an external VLM judge to score cross-shot consistency and inter-shot coherence against a curated source scene S, which serves purely as a narrative anchor and is strictly excluded from policy inputs.

## 4.4. Evolutionary GRPO

Standard GRPO [23] estimates advantages through intragroup normalization, without explicitly referencing parentstoryboard or historical policy performance. However, in alternating co-evolution, this relative comparison allows an inferior rollout to receive a falsely positive advantage whenever the entire group degrades, reinforcing counterproductive revisions. To mitigate intra-group advantage inflation, we propose Evolutionary GRPO, which equips GRPO with revision safeguarding and historical anchoring (Fig. 3 (c)).

Revision safeguarding. To prevent the policy from adopting destructive edits, each rollout is benchmarked against its parent storyboard reward $p _ { g } .$ . At the reward level, any edit introducing new Critic-verified severe defects has its reward $r _ { i }$ capped at $p _ { g } .$ . For each rollout group g, given the resulting rewards $\{ r _ { i } \} _ { i = 1 } ^ { \overline { { G } } }$ , we compute the group advantage $a _ { i } = ( r _ { i } - \mu _ { g } ) / \operatorname* { m a x } ( \sigma _ { g } , \epsilon _ { \mathrm { s t d } } )$ , where $\mu _ { g }$ and $\sigma _ { g }$ denote the empirical mean and standard deviation of $\{ r _ { i } \} _ { i = 1 } ^ { G }$ , and $\epsilon _ { \mathrm { s t d } } = 0 . 0 5$ is the standard-deviation floor. However, to ensure that sub-par edits never receive positive learning signals even if they rank high in a low-quality group, we enforce an advantage safeguard: for revision rollouts with $r _ { i } \le p _ { g }$ , we set $\widetilde { \boldsymbol { a } } _ { i } = \operatorname* { m i n } ( \boldsymbol { a } _ { i } , 0 )$ ; otherwise, $\widetilde { a } _ { i } = a _ { i }$

Historical anchoring. To encourage steady improvement across rounds, we anchor advantages to historical performance rather than relying solely on within-group relative comparisons. For each task (initial creation, defect critique, and targeted revision), we sample one response per input on 256 fixed calibration inputs and take the mean reward under the current evaluator as $b _ { k }$ . Setting $p _ { g } = 0$ for non-revision tasks, we formulate the anchored advantage as $u _ { i } = ( r _ { i } - \operatorname* { m a x } ( \mu _ { g } , b _ { k } , p _ { g } ) ) / \operatorname* { m a x } ( \sigma _ { g } , \epsilon _ { \mathrm { s t d } } )$ and compute the final advantage via bounded interpolation:

$$
A _ { i } = ( 1 - \alpha ) \widetilde { a } _ { i } + \alpha u _ { i } ,\tag{3}
$$

where $\alpha \in [ 0 , 1 ]$ is the largest value retaining at least half the positive advantage mass and at most doubling the negative advantage magnitude relative to $\widetilde { a } _ { i }$ , weighted by trainable token counts over valid responses in the full optimization batch.

## 5. Experiments

## 5.1. Training Details

We initialize the shared policy with Qwen3-VL-8B-Instruct [3], fine-tuning its language backbone while keeping the visual encoder frozen. For generative execution, we use the four-step distilled FLUX.2-klein-9B [5] for storyboard generation and editing, and the eight-step distilled MiniMax-H3 Turbo [14] for video generation, with both visual backbones kept frozen. Supervised policy warm-up is conducted for two epochs with a learning rate of $1 0 ^ { - 5 }$ and an effective batch size of 64. Subsequently, we perform five rounds of alternating reinforcement learning, with 64 optimization steps allocated per role in each round. We use a learning rate of $1 0 ^ { - 6 }$ and a batch size of 16. All experiments are conducted on 16 NVIDIA A100 GPUs.

## 5.2. Evaluation Metrics

We comprehensively evaluate CineEvolve across three aspects<sup>4</sup>: (i) Visual Quality and Alignment: we assess aesthetic appeal, technical fidelity, and text and video alignment using Aesthetic Quality (AQ), Imaging Quality (IQ), and Overall Consistency (OC) metrics from VBench [12]. (ii) Within-Shot Consistency: we evaluate the temporal stability of subjects and scene backgrounds across successive frames using Subject Consistency (SC) and Background Consistency (BC) metrics from VBench [12]. (iii) Cross-Shot Coherence: we evaluate visual continuity and narrative transitions across shots through four complementary metrics. Specifically, Entity Appearance Consistency (EAC) and Transition Contrast Agreement (TCA) measure cross-shot entity similarity and visual transition fidelity using DINOv2 [19] and CLIP [21] features. Additionally, Cross-shot Consistency (CSC) and Transition Logic Consistency (TLC) leverage a Vision-Language Model [7] evaluator to assess fine-grained entity attributes and inter-shot narrative progression, referencing the original story and source scene.

## 5.3. Comparison with Existing Methods

We compare CineEvolve with six representative multi-shot video generation baselines, categorized into direct visual synthesis methods (HoloCine [18], MultiShotMaster [27], and ShotStream [17]) and agentic methods (STAGE [32], ViMax [11], and CineCrew [6]). All methods are evaluated on the proposed CineTraj benchmark (Sec. 3).

Fair Comparisons. To ensure a fair evaluation, all baselines share identical backbones with CineEvolve: Qwen3-VL-8B-Instruct serves as the VLM for agentic baselines, while FLUX.2-klein-9B and MiniMax-H3 Turbo are adopted as the unified image and video backbones. For visual-based baselines, we supply detailed shot-level scripts pre-expanded by Qwen3-VL-8B-Instruct to ensure fairness.

Qualitative Comparisons. In Fig. 4, we present visual quality comparison results with the aforementioned methods. ShotStream [17] fails to strictly adhere to numerical constraints specified in the prompt (e.g., Fig. 4 first row, it violates the entity count by depicting multiple fish in shot 5, undermining prompt fidelity). HoloCine [18] suffers from poor narrative progression and pacing (e.g., Fig. 4 second row, it abruptly omits the initial scene setup in shot 1, lead ing to a confusing storyline). MultiShotMaster [27] exhibits notable visual artifacts and attribute instability (e.g., Fig. 4 third row, the character’s clothing color drastically alters alongside severe facial distortions across shot 3 and shot 4). STAGE [32] suffers from severe attire drift across long-range transitions (e.g., Fig. 4 fourth row, the protagonist’s outfit inconsistently morphs between shot 1 and shot 6, breaking identity persistence). CineCrew [6] opens too abruptly with

A young man in a faded blue shirt and an older man with a gray beard and a beige vest fish from a small white boat on a calm sea. A large bluefish takes the bait and leaps from the water, scattering spray. The young man stands and reels it toward the boat, his rod bending under the strain. The older man leans over the side with a landing net and lifts the fish aboard. They hold the net together, admiring the fish and talking about their catch. With the fish resting on the deck, they sit back and exchange satisfied smiles.

![](images/3ce0ef014e415a578e2f9195b3a984d56d84458e815dcba7f99bf95fdf922bf6.jpg)  
Figure 4. Qualitative comparisons with existing multi-shot video generation methods.

Table 1. Quantitative comparisons and ablation results. ↑ (↓) means higher (lower) is better. The best performances are highlighted in bold.
<table><tr><td>Method</td><td>AQ↑</td><td>IQ↑</td><td>OC↑</td><td>SC↑</td><td>BC↑</td><td>EAC ↑</td><td>TCA↑</td><td>CSC ↑</td><td>TLC ↑</td></tr><tr><td colspan="10">Comparison with existing methods</td></tr><tr><td>ShotStream</td><td>0.5789</td><td>0.6881</td><td>0.2393</td><td>0.9040</td><td>0.9279</td><td>0.6889</td><td>0.8144</td><td>0.8467</td><td>0.7966</td></tr><tr><td>STAGE</td><td>0.5859</td><td>0.6805</td><td>0.2522</td><td>0.9285</td><td>0.9189</td><td>0.5906</td><td>0.8335</td><td>0.7600</td><td>0.7309</td></tr><tr><td>CineCrew</td><td>0.6250</td><td>0.7022</td><td>0.2469</td><td>0.8596</td><td>0.9092</td><td>0.7296</td><td>0.7990</td><td>0.8013</td><td>0.7943</td></tr><tr><td>HoloCine</td><td>0.5398</td><td>0.5612</td><td>0.2212</td><td>0.9118</td><td>0.9270</td><td>0.5746</td><td>0.8156</td><td>0.7226</td><td>0.7386</td></tr><tr><td>ViMax</td><td>0.6264</td><td>0.7010</td><td>0.2492</td><td>0.9322</td><td>0.9355</td><td>0.6073</td><td>0.7914</td><td>0.8060</td><td>0.6939</td></tr><tr><td>MultiShotMaster</td><td>0.5621</td><td>0.5764</td><td>0.2452</td><td>0.9559</td><td>0.9528</td><td>0.5642</td><td>0.8216</td><td>0.7833</td><td>0.7592</td></tr><tr><td colspan="10">Ablation studies</td></tr><tr><td>SFT only</td><td>0.5912</td><td>0.6828</td><td>0.2454</td><td>0.9299</td><td>0.9291</td><td>0.6223</td><td>0.8171</td><td>0.7952</td><td>0.7529</td></tr><tr><td>Fixed Critic</td><td>0.6093</td><td>0.6859</td><td>0.2472</td><td>0.9393</td><td>0.9320</td><td>0.6782</td><td>0.8255</td><td>0.8103</td><td>0.7765</td></tr><tr><td>Separate Policies</td><td>0.6185</td><td>0.6896</td><td>0.2477</td><td>0.9452</td><td>0.9400</td><td>0.7040</td><td>0.8184</td><td>0.8213</td><td>0.7902</td></tr><tr><td>Original GRPO</td><td>0.6206</td><td>0.6916</td><td>0.2513</td><td>0.9495</td><td>0.9421</td><td>0.7179</td><td>0.8307</td><td>0.8350</td><td>0.7881</td></tr><tr><td>CineEvolve (Ours)</td><td>0.6391</td><td>0.7155</td><td>0.2599</td><td>0.9621</td><td>0.9507</td><td>0.7587</td><td>0.8456</td><td>0.8893</td><td>0.8156</td></tr></table>

Two friends wearing aprons prepare six small bread rolls in a sunlit kitchen. One shapes the dough into rolls while the other arranges them on a metal baking tray. She then slides the tray into the oven to bake. Later, she lifts out the freshly baked golden rolls with oven mitts while her friend watches beside her. Once the bread has cooled slightly, they tear a roll in half and share it.

![](images/bfb0fca6db9734305981b75f1e6bd2b2cc1d439c5aa50084959c54134edf548b.jpg)  
Figure 5. Ablation study results with different variants of our CineEvolve framework.

out establishing adequate character context (e.g., Fig. 4 fifth row, the absence of an establishing shot disrupts the cinematic pacing). ViMax [11] struggles with environmental consistency across continuous scenes (e.g., Fig. 4 sixth row, an unnatural lighting discontinuity occurs between daylight and sunset across shot 4 to shot 6, creating an incoherent visual tone). In contrast, CineEvolve preserves character identity and lighting consistency across all frames while depicting a coherent progression from the initial fishing scene to landing the catch and the final conversation, delivering a highly cinematic and coherent sequence.

Quantitative Comparisons. We present quantitative comparisons in Tab. 1, where CineEvolve consistently outperforms existing multi-shot generation approaches across eight metrics. This demonstrates its superior ability to create multi-shot cinematic videos with high aesthetic appeal and technical fidelity (AQ, IQ), faithful story-to-video alignment (OC), robust intra-shot temporal stability (SC, BC), reliable cross-shot visual persistence (EAC, CSC), and coherent narrative progression with logical transitions (TCA, TLC).

User Study. We evaluate human preferences across four dimensions<sup>5</sup>: Visual Quality (VQ), Text Alignment (TA),

Table 2. Human preference shares (%) across four evaluation dimensions. Higher is better.
<table><tr><td>Method</td><td>VQ</td><td>TA</td><td>CC</td><td>IC</td></tr><tr><td>ShotStream</td><td>7.2</td><td>6.6</td><td>6.0</td><td>6.6</td></tr><tr><td>STAGE</td><td>5.8</td><td>6.0</td><td>6.2</td><td>6.8</td></tr><tr><td>CineCrew</td><td>7.2</td><td>5.8</td><td>8.4</td><td>8.8</td></tr><tr><td>HoloCine</td><td>10.2</td><td>10.2</td><td>5.4</td><td>5.6</td></tr><tr><td>ViMax</td><td>6.8</td><td>7.0</td><td>9.6</td><td>9.6</td></tr><tr><td>MultiShotMaster</td><td>10.6</td><td>10.4</td><td>5.8</td><td>5.8</td></tr><tr><td>CineEvolve (Ours)</td><td>52.2</td><td>54.0</td><td>58.6</td><td>56.8</td></tr></table>

Cross-shot Consistency (CC), and Inter-shot Coherence (IC). We randomly select 25 test stories and recruit 20 volunteers from Amazon Mechanical Turk (AMT) to independently select their preferred video among the seven methods for each dimension. As shown in Table 2, CineEvolve receives the highest preference share in all four dimensions.

## 5.4. Ablation Studies

To validate the effectiveness of our key design and training paradigms, we construct four ablation variants. The evaluation scores and visual results of the ablation study are presented in Table 1 and Fig. 5, respectively.

SFT Only. We remove alternating reinforcement learning to retain only the supervised warm-up policy. Without iterative trial-and-error reflection, the model fails to continuously refine its directorial planning and diagnostic precision, yielding degraded performance across all metrics. As shown in Fig. 5 (row 1), the generated sequence suffers from chronological reversals and environmental drift, where a baked roll prematurely appears in Shot 2 before the baking stage.

Fixed Critic. We freeze the Critic after supervised warmup while optimizing only the Director during reinforcement learning. Lacking dynamic diagnostic adaptation, the static Critic fails to provide informative feedback across directorial rollouts. Thus, this variant degrades cross-shot consistency (lower EAC and CSC scores). As shown in Fig. 5 (row 2), it fails to preserve object-level consistency, resulting in changes in the count and spatial arrangement of bread rolls on the tray between Shot 2 and Shot 4.

Separate Policies. We optimize the Director and Critic separately rather than within a unified model. Without unified parameter sharing, the model loses cognitive synergy in cinematic grammar, resulting in degraded transition coherence (lower TLC score). As shown in Fig. 5 (row 3), it ambiguously conflates loading and unloading the oven in Shot 4, disrupting the transition from raw dough to baked bread.

Original GRPO. We replace E-GRPO with standard GRPO, removing revision safeguarding and historical anchoring. Without baseline calibration, standard GRPO falsely rewards defective candidates that rank highest in a poor group, encouraging flawed revisions that harm entity consistency and transition logic (lower CSC and TLC scores). As shown in Fig. 5 (row 4), it exhibits identity and state flickering, where the woman’s attire abruptly changes from long to short sleeves between Shot 2 and Shot 3.

## 6. Conclusion

In this paper, we present CineEvolve, an agentic selfevolution framework for cinematic multi-shot video generation. By unifying the Director and Critic roles within a single vision-language policy, CineEvolve realizes iterative multi-shot synthesis via alternating reinforcement learning. To sustain stable co-evolution, we introduce role-specific rewards that couple fine-grained diagnostic fidelity with self evaluative feedback, complemented by E-GRPO to calibrate policy optimization. Furthermore, we construct CineTraj, a large-scale cinematic trajectory dataset comprising 40K samples for cinematic multi-shot video generation. Extensive experiments demonstrate that CineEvolve significantly outperforms existing approaches in both narrative coherence and visual execution, establishing a promising paradigm for autonomous cinematic creation.

## Limitations

CineEvolve primarily performs defect diagnosis and targeted revision at the storyboard level. Although multi-shot narrative consistency and cinematic composition are predominantly governed by keyframe storyboards, subtle motion artifacts can still arise during subsequent video synthesis, which remain outside the current feedback loop. We expect these residual artifacts to diminish naturally as foundational video generation backbones continue to advance.

## References

[1] Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. In International Conference on Learning Representations, pages 8479–8565, 2026. 2

[2] Zhaochong An, Menglin Jia, Haonan Qiu, Zijian Zhou, Xiaoke Huang, Zhiheng Liu, Weiming Ren, Kumara Kahatapitiya, Ding Liu, Sen He, Chenyang Zhang, Tao Xiang, Fanny Yang, Serge Belongie, and Tian Xie. OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16173–16184, 2026. 1, 2

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025. 6

[4] Max Bain, Arsha Nagrani, Andrew Brown, and Andrew Zisserman. Condensed Movies: Story based retrieval with con textual embeddings. In Computer Vision – ACCV 2020, pages 460–479, Cham, 2021. Springer International Publishing. 3

[5] Black Forest Labs. FLUX.2 [klein] 9B, 2026. Model card. Accessed October 13, 2026. 3, 6

[6] Jiaben Chen, Sixun Dong, Qinhong Zhou, Raine Ma, Zhiyang Dou, Wojciech Matusik, and Chuang Gan. Better Call CineCrew: Consistent Ultra-Long Narrative-to-Film Generation. In Computer Vision – ECCV 2026, pages 338–355, Cham, 2026. Springer Nature Switzerland. 2, 6

[7] Google DeepMind. Gemini 3.1 Pro Model Card, 2026. Pub lished February 19, 2026. 3, 6

[8] Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. VisPlay: Self-Evolving Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26274–26284, 2026. 2

[9] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-Evolving Reasoning LLM from Zero Data. In International Conference on Learning Representations, pages 130770–130790, 2026. 2

[10] Kaiyi Huang, Yukun Huang, Xintao Wang, Zinan Lin, Xuefei Ning, Pengfei Wan, Di Zhang, Yu Wang, and Xihui Liu. FilMaster: Bridging Cinematic Principles and Generative AI for Automated Film Generation. In International Conference on Learning Representations, pages 155439–155482, 2026. 2

[11] Lingxuan Huang, Sizhe He, Hengji Zhou, Liqiang Nie, Lianghao Xia, and Chao Huang. ViMax: Agentic Video Generation. arXiv preprint arXiv:2606.07649, 2026. 2, 6, 8

[12] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, 2024. 6

[13] Zhicong Li, Lingjie Jiang, Yulan Hu, Xingchen Zeng, Yixia Li, Xiangwen Zhang, Guanhua Chen, Zheng Pan, Xin Li, and Yong Liu. No More Stale Feedback: Co-Evolving Critics for Open-World Agent Learning. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12643–12660. Association for Computational Linguistics, 2026. 2

[14] LightX2V Team. MiniMax-H3 Turbo, 2026. Model card. Accessed October 13, 2026. 6

[15] Yunlong Lin, Linqing Wang, Kunjie Lin, Zixu Lin, Kaixiong Gong, Wenbo Li, Bin Lin, Zhenxi Li, Shiyi Zhang, Yuyang Peng, Wenxun Dai, Xinghao Ding, Chunyu Wang, and Qinglin Lu. JarvisEvo: Towards a Self-Evolving Photo Editing Agent with Synergistic Editor-Evaluator Optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27291– 27302, 2026. 2

[16] Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, and Huaxiu Yao. Agent0-VL: Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning. In Proceedings of the 43rd International Conference on Machine Learning, pages 79198–79215. PMLR, 2026. 2

[17] Yawen Luo, Xiaoyu Shi, Junhao Zhuang, Yutian Chen, Quande Liu, Xintao Wang, Pengfei Wan, and Tianfan Xue. ShotStream: Streaming Multi-shot Video Generation for Interactive Storytelling. In Computer Vision – ECCV 2026, pages 193–209, Cham, 2026. Springer Nature Switzerland. 2, 6

[18] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Shuailei Ma, Yixuan Li, Cheng Chen, Yanhong Zeng, Xing Zhu, Yujun Shen,

and Huamin Qu. HoloCine: Holistic Generation of Cinematic Multi-Shot Long Video Narratives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 461–471, 2026. 1, 2, 6

[19] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research, 2024. 6

[20] Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xi angru Tang, Vishy Tirumalashetty, George Lee, Mahsan Ro fouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory. In International Conference on Learning Representations, pages 94327–94354, 2026. 2

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 6

[22] Christoph Schuhmann. LAION-Aesthetics, 2022. LAION technical blog, August 16, 2022. 3

[23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300, 2024. 5

[24] Tomáš Soucek and Jakub Lokoˇ c. TransNet V2: An effectiveˇ deep network architecture for fast shot transition detection. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11218–11221. Association for Computing Machinery, 2024. 3

[25] Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8416–8439. Association for Computational Linguistics, 2025. 2

[26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,

Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025. 3

[27] Qinghe Wang, Xiaoyu Shi, Baolu Li, Weikang Bian, Quande Liu, Huchuan Lu, Xintao Wang, Pengfei Wan, Kun Gai, and Xu Jia. MultiShotMaster: A Controllable Multi-Shot Video Generation Framework. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16268–16278, 2026. 2, 6

[28] Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models. In International Conference on Learning Representations, pages 60198–60230, 2026. 2

[29] Cilin Yan, Jingyun Wang, Lin Zhang, Ruihui Zhao, Xiaopu Wu, Kai Xiong, Qingsong Liu, Guoliang Kang, and Yangyang Kang. Efficient and accurate prompt optimization: the benefit of memory in exemplar-guided reflection. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 753–779. Association for Computational Linguistics, 2025. 2

[30] Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. Gödel Agent: A selfreferential agent framework for recursively self-improvement. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 27890–27913. Association for Computational Linguistics, 2025. 2

[31] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents. In International Conference on Learning Representations, pages 104223–104294, 2026. 2

[32] Peixuan Zhang, Zijian Jia, Kaiqi Liu, Shuchen Weng, Si Li, and Boxin Shi. STAGE: Storyboard-Anchored Generation for Cinematic Multi-shot Narrative. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 659–669, 2026. 1, 2, 6

[33] Peixuan Zhang, Chang Zhou, Ziyuan Zhang, Hualuo Liu, Chunjie Zhang, Jingqi Liu, Xiaohui Zhou, Xi Chen, Shuchen Weng, Si Li, and Boxin Shi. A Benchmark and Multi-agent System for Instruction-Driven Cinematic Video Compilation. In Computer Vision – ECCV 2026, pages 213–230, Cham, 2026. Springer Nature Switzerland. 3

[34] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. In International Conference on Learning Representations, pages 86069–86100, 2026. 2

[35] Zhengxin Zhang, Chengyu Huang, Aochong Oliver Li, and Claire Cardie. Better LLM reasoning via dual-play. In Findings ofthe Associationfor Computational Linguistics: ACL