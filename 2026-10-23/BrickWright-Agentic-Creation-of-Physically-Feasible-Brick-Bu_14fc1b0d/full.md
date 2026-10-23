# BrickWright: Agentic Creation of Physically Feasible Brick Builds

Ava Pun<sup>1</sup> Kevin Dai<sup>1</sup> Sally Chen<sup>1</sup> Maxwell Jones<sup>1</sup> Ruixuan Liu<sup>1</sup> Kangle Deng<sup>2</sup> Changliu Liu<sup>1</sup> Maneesh Agrawala<sup>3</sup> Deva Ramanan<sup>1</sup> Jun-Yan Zhu<sup>1</sup> <sup>1</sup>Carnegie Mellon University <sup>2</sup>Independent Researcher <sup>3</sup>Stanford University

![](images/6ea0c43e01fb24b183a9a858c9cc6c268bc70f3e8eec9e938d9c02d9d6dd5e27.jpg)

![](images/72dd94c8d5f1def4360df95e3aaf1f63ddbf6665a6f15e0f06b29ce7eebfb533.jpg)  
Figure 1. BrickWright overview. BrickWright instructs pre-trained LLM agents to generate physically feasible brick builds by writing code using a Python SDK, BrickLayer, in a plan-execute-validate loop utilizing BrickLayer’s physical validation tools (a). BrickWright accepts text or image inputs and can produce articulated and large-scale builds, all physically valid. Agents equipped with BrickWright produce higher-quality builds with fewer physical violations (b). Red outlines in insets indicate mesh intersections.

## Abstract

Frontier coding agents are becoming increasingly adept at 3D modeling. Yet, they still face challenges when building objects meant to be assembled in the physical world, often creating invalid assemblies with floating, colliding, or disconnected parts. We argue that providing agents with tools for assembling builds and verifying their physical feasibility is sufficient for them to produce physically valid assemblies that still adhere to input prompts. Focusing on the domain of LEGO<sup>®</sup> brick builds, we present BrickWright, a framework for generating physically feasible brick builds from text or image prompts using pre-trained agents, where a build is physically feasible if it uses only real parts, has no colliding orfloating parts, andforms a connected structure if possible. BrickWright guides agents to create builds by writing code with BrickLayer, a Python toolkit with two main components: an ergonomic brick assembly API that

halves inference cost compared to working in raw LDraw, and a validator that provides feedback on physical feasibility. Whereas baseline agents achieve physical validity rates rangingfrom only 9% (GPT-6 Luna) to 69% (GPT-6 Astra), BrickWright-guided agents achieve 100% validity on a sample of 512 prompts, while improving prompt adherence for lightweight agents and maintaining it for stronger agents. Additionally, they outperform specialized models on the text-to-brick task. Code is available on our project page: avalovelace1.github.io/brickwright.

## 1. Introduction

Large language model (LLM) agents are becoming increasingly adept at 3D modeling. Without additional fine-tuning, commercially available agents can successfully produce articulated objects [45, 59], scenes [52], and constructive solid geometry models [27]. However, they still face challenges when building objects for the physical world. We focus on the task of creating multi-part assemblies: real-world objects made from a set of prefabricated parts joined by fixed connectors, such as LEGO<sup>®</sup> builds. Multi-part assemblies must obey multiple constraints: all parts must be drawn from a preset library, parts must not collide or float, and the joined parts typically form a single connected component.

![](images/fac5bb5273d242628ed0eb11a38783fa60f0582a669f505fae75a8ab838f4c28.jpg)  
Figure 2. Physical violations in builds by Codex (GPT-6 Astra, high reasoning). Red outlines in insets show mesh intersections.

We argue that designing a toolkit for multi-part assembly creation is sufficient to enable agents to create physically valid builds. This toolkit should include (1) an assembly API, which provides an ergonomic programmatic interface for 3D creation; and (2) a validator, which provides feedback on a finished build’s physical validity. The need for a toolkit is independent of how powerful a coding agent is. We observe that GPT-6 Astra with Codex, when asked to make brick builds, already writes scripts to generate and validate its output more than 80% of the time (Sec. D). For complex physical domains, this self-validation is often not enough, resulting in builds that do not satisfy physical constraints, containing colliding, floating, or disconnected parts (Fig. 2). Even if an agent is powerful enough to write a perfect toolkit, separately developing a toolkit remains important to reduce token usage and avoid repeated work. Existing toolkits for agentic 3D generation [45, 59] are usually intended for digital asset creation; as they do not consider physical constraints, they cannot be repurposed as-is to create real-world objects.

We focus on the domain of LEGO<sup>®</sup> generation as it is a commonly available yet complex system, consisting of thousands of parts that connect in dozens of ways. Our key insight is to merge agentic code-as-3D-models with the regularity of brick parts. Parts can be modeled as geometries with localized connectors that have programmatic rules for assembly (e.g., studs interlock with antistuds). From this perspective, we leverage the physical production rules for snap-fit brick builds [6] to create an ergonomic assembly API. Then, given an image or text prompt, our task is to generate a program using this API that, when compiled, produces a LEGO<sup>®</sup> brick build in the standard LDraw format [21]. This build should include a list of library part placements, and it must be physicallyfeasible, having no intersecting or floating parts and forming a connected structure if possible. While there are several specialized brick generation methods [10, 11, 19, 36, 37, 48, 50, 53], some of which consider physical constraints, they generally exhibit worse generalization and lower fidelity than commercial LLM agents on the same task. Hence, we wish to achieve this task agentically.

Toward this goal, we introduce BrickLayer, a Python package for assembling and validating brick builds, alongside BrickWright, a framework that guides LLM agents through creating validated brick builds with BrickLayer. Agents equipped with BrickWright create higher-quality builds with fewer physical violations; on a test set of 512 prompts, all agents—including lightweight models like GPT-6 Luna and Claude Haiku 5.5—achieve a 100% validity score without a significant cost increase or drop in fidelity. BrickWright is especially useful for lightweight models: it improves Luna’s performance from 9% to 100% valid builds and boosts its prompt adherence at 2% the price of frontier models. BrickWright additionally outperforms specialized brick generation models on validity and fidelity. Code is available on our project page: avalovelace1.github.io/brickwright.

## 2. Related work

Brick Build Generation. Early work treats brick build generation as an optimization problem over a target shape. Legolization [34] voxelizes a 3D mesh, merges voxels into standard bricks, and iteratively re-merges regions identified as unstable. Subsequent work captures finer geometric detail through feature-aware shape approximation and nonrectangular bricks [57, 58], assembles bricks sequentially via Bayesian optimization [17], or optimizes Technic assemblies to match annotated sketches [49]. Retrieval-based methods map images to figurines assembled from prebuilt templates [9, 56]. These methods are limited by small brick vocabularies, explicit geometric guidance, or fixed templates.

Learning-based methods generate brick structures directly from data. These approaches adapt graph generative models [41], predict voxel representations that are assembled into bricks algorithmically [4, 24], learn sequential construction policies [7], or use diffusion to predict labeled voxels within a single category [10]. Autoregressive methods generate bricks conditioned on text or images. BrickGPT [37] fine-tunes an LLM to generate stable structures, but is restricted to eight rectangular brick types across 21 object categories. LEGO-Maker [11] generates builds from images, but supports only axis-aligned orientations and two categories. LegoACE [50] and BrickNet [19] scale to openvocabulary generation by training on large collections of human-designed builds. However, LegoACE learns connectivity implicitly and can produce disconnected components, while BrickNet represents assemblies as connector graphs but accumulates collisions over longer build sequences. Brick-Composer [30] instead approaches assembly from target views as sequential brick selection and pose estimation, training a multimodal LLM on simulator-generated data and feedback. Other methods improve buildability through preference optimization or reinforcement learning [36, 48]. In contrast, BrickWright utilizes pre-trained models to build arbitrary objects without task-specific training, verifies collisions and connectivity in a feedback loop, and supports articulation. Concurrent with our work, BrickBench [20] introduces a toolkit for agentic brick build creation, focusing on benchmarking the performance of different agents.

![](images/c0c943e71ffc54ad3b96183f31a488d5d707d226b8e25ab64018b9f518269d0b.jpg)  
Figure 3. Method overview. (a) BrickLayer supports catalog discovery, compositional construction with typed connectors, and physical validation. (b) BrickWright prompts a pre-trained LLM agent to write and revise build.py, which contains brick build creation code and agent-defined tests. The validate tool runs fixed collision and floating checks alongside the agent’s tests, while the render too provides views for visual inspection. The agent calls finalize once all checks pass. The agent may optionally run explore at any time to investigate the BrickLayer API and written brick build. Arrows summarize information flow rather than a fixed tool-call sequence.

Agentic 3D Generation. LLM agents have been used successfully to create 3D scenes. Recent works use a planexecute-critique loop [15, 29] or reconstruct scenes from images through iterative code execution and visual inspection [13, 52]. Other works fine-tune LLMs to reason directly over 3D spatial inputs [44] or to generate tool-call trajectories [14]. These methods place objects in continuous space and rely on optimization or visual judgment to resolve collisions and floating objects. In BrickWright, explicit connector rules enable deterministic checks of attachment validity.

Representing shapes as programs provides structured descriptions of geometry and part relationships [16, 39, 42]. Separately, LLMs have been shown to reason more reliably when they write code and offload exact computation to an interpreter [8]. Building on both ideas, agents have been applied to modeling individual objects as code, typically through plan-execute-critique loops. Prior work augments this loop with API retrieval [32], domain-specific languages [45], or procedural part-based modeling [27], with some supporting articulation [23, 27, 45, 59]. In CAD, LLMs have been fine-tuned to generate parametric modeling code [12, 18, 47], while tool-augmented agents generate and refine parts and assemblies [1, 35, 40, 55]. Articraft [59] generates articulated assets using a custom SDK and a minimal harness that provides textual feedback from tools that probe the model code. Unlike these systems, where the agent generates geometry or infers inter-part relationships, Brick-Wright assembles fixed parts whose connectors determine how they attach and move.

Physics-Aware Generation. To improve physical validity, prior work refines shapes or layouts after generation [5, 25, 29, 51, 54], fine-tunes generators on simulator feedback [26], or guides diffusion sampling with physical constraints [33]. Geometric analysis has also been used to plan LEGO assembly sequences without explicit connector information [2]. For brick structures, stability has been assessed through force-based equilibrium analysis [31, 34, 37] or physics simulation [17, 46]. In contrast, BrickWright uses an executable verifier to check for collisions, floating parts, and connectivity, allowing a pre-trained agent to iteratively repair builds without task-specific training. Incorporating force-based stability analysis [31, 34] is a natural extension.

## 3. BrickWright

Given a natural-language or image description, our task is to generate a physically feasible brick build from a fixed part catalog. A brick build is defined as an ordered sequence of placements. Each placement consists of an element or a nested brick build, together with a rigid transformation relative to the containing build. An element specifies a single part drawn from the catalog and a color. Elements expose typed connectors, such as studs, whose compatibility and relative alignment determine valid connections. A build is valid if it uses only catalog parts and contains no colliding (intersecting) or floating parts. It forms a connected component if one can draw a path of connections between any two elements in the build. We define a build to be physicallyfeasible if it is valid and contains as few connected components as possible. More complex feasibility considerations, such as ensuring a build is stable and has an executable assembly sequence, are left to future work. Note that although having a single connected component is desirable as it makes a build less likely to break apart in the physical world, builds with multiple components are still necessary in some cases: for example, two separate vehicles in a car chase scene.

We introduce BrickWright (Fig. 3), a framework for generating physically feasible brick builds using LLM agents. Using a lightweight harness, BrickWright guides a single agent to write Python code that produces a brick build using a custom toolkit called BrickLayer. If the build passes a set of predefined and agent-written tests, it is accepted; otherwise, the agent makes edits informed by the test results, following the commonly used plan-execute-validate loop.

## 3.1. BrickLayer

LDraw [21], the standard text format for LEGO<sup>®</sup> builds, includes no physical validation and is not very ergonomic: it only allows users to place parts and sub-models by their absolute 3D poses. This makes it difficult to mate parts by their connectors or place them relative to others, which can lead to wasted tokens and misplaced components. To provide agents with an ergonomic interface for creating and validating brick builds, we introduce BrickLayer (Listing 1), a Python package that includes functions for part discovery, assembly, and validation. BrickLayer provides a unified programmatic representation for brick builds, enabling them to be easily understood and edited by humans and agents; builds can be translated from BrickLayer to LDraw and vice versa. See Sec. B for BrickLayer documentation.

Discovery. BrickLayer allows users to search the LDraw catalog [22], which contains 17k+ parts. This search accepts keywords matched against the part’s official LDraw name. Results are ordered by their BM25 score [38] multiplied by a popularity factor that scales logarithmically with the number of official LEGO<sup>®</sup> sets containing the part. LDraw names are parsed into fields such as category, subcategory, and dimensions; users may target specific fields with their keyword search and filter results using exact field matches.

Assembly. The central objects of BrickLayer are Element, representing a single colored part, and LegoBuild, representing a brick build. Both objects support a unified LegoGeometry interface; this interface exposes dimensions and available connectors, which can be filtered by type and position. We use BrickNet’s connector annotations [19], which mark the local frames of the connectors on each part, along with compatible connector types and the joints they form: fixed, revolute, ball, prismatic, or cylindrical. For example, studs and antistuds join with a revolute joint. As BrickNet annotations do not differentiate between prismatic and cylindrical joints, we determine a sliding joint’s type by rotating it 15<sup>◦</sup> about its axis in one direction; if a collision occurs, we mark the joint prismatic.

```python
b = LegoBuild() # Initialize build
socket = b.add(Element("14418", Color.ORANGE))
ball = Element("3170", Color.MEDIUM_AZURE)
b.connect( # Search for and mate compatible connectors
ball,
ball.find_connector("towball",
toward=Direction.RIGHT),
to=socket.find_connector("towball_socket")
)
assert (not b.colliding_placements # Check feasibility
and not b.floating_placements
and len(b.connected_components == 1)
write_ldraw(b, "build.ldr") # Export to LDraw
Output LDraw:
1 25 20 -8 10 1 0 0 0 1 0 0 0 1 14418.dat
1 322 80 -8 10 1 0 0 0 1 0 0 0 1 3170.dat
```  
Listing 1. BrickLayer usage. Users can add elements, join them via selected connectors, and assess feasibility. The BrickLayer script is easier to understand than its corresponding LDraw output.

LegoBuilds have a recursive structure: they may contain both Elements and other LegoBuilds, allowing complex models to be built from smaller components. Placements can be made at arbitrary poses, fixed to a grid, or by mating specified connectors. Users may name placements, allowing them to place components relative to or directly target the connectors of a named component. Users may also explicitly mark movable components with articulation limits corresponding to their joint type.

For convenience, BrickLayer includes several high-level construction functions, introduced after noting repeated patterns in agent-generated code. These include brick, plate, tile, slope, and inverted slope to instantiate common parts by specifying their dimensions; minifigure, door, and window to create common multi-component sub-builds; mirror to reflect a build along a given axis; and fill box, fill slope, and fill inverted slope to fill a cuboid or wedge in a manner guaranteed to be valid and connected if possible.

Validation. A LegoBuild provides methods to list collisions, floating elements, and disconnected components. To determine whether two elements collide, we inset each mesh by 0.1 mm before checking for intersections, following BrickNet; snap connections between bricks are so tight that a naive mesh intersection check would report a false positive. The floater check first translates a build so its lowest point is at the ground plane. Then, it marks an element floating if it cannot reach a grounded element through mated connectors or supporting surface contacts. A surface contact is supporting if it has an upward component; lateral contact alone is insufficient. Two connectors form a connection if their local frames align, depending on their joint type; for example, a fixed joint requires the two frames to align exactly, while a revolute joint additionally admits a rotational degree of freedom. For builds with user-annotated articulations, a posed method allows an agent to create a snapshot of the build with joints set to specified articulation parameters, to validate physical feasibility under joint motion. Posing works with models whose kinematic structure forms a tree.

Table 1. Quantitative results on BrickNet-512. Valid, No col., and No fl. are valid, collision-free, and floater-free build rates. Comp., Parts, and Unique are mean component, part, and distinct part-ID counts. PE: Perception Encoder. SL2: SigLIP 2. Checks: mean validate calls. Bold/underline mark best/second-best results within each agent group. Costs use October 2026 token pricing, including cache savings.
<table><tr><td>Method</td><td>Valid↑ No col.↑ No fl.↑</td><td></td><td></td><td>Comp.↓</td><td>VQA↑ PE↑</td><td>SL2↑</td><td>Parts</td><td>Unique</td><td>Checks↓</td><td>Time↓ Cost↓</td></tr><tr><td>BrickGPT [37]</td><td>0.990</td><td>1.000</td><td>0.990</td><td>2.67</td><td>0.0658 0.1547</td><td>0.0514144.8</td><td>7.1</td><td></td><td>247s</td><td></td></tr><tr><td>BrickNet [19]</td><td>0.318</td><td>0.320</td><td>0.943</td><td>1.03</td><td>0.6077 0.2846</td><td>60.6407 31.8</td><td>11.5</td><td></td><td>5s</td><td></td></tr><tr><td>Claude Code 5</td><td>0.328</td><td>0.428</td><td>0.387</td><td>3.59</td><td>0.6032 0.2963</td><td>0.7286 12.8</td><td>5.3</td><td></td><td>168s</td><td>$0.04</td></tr><tr><td>5. Ours w/o assembly API 0.822</td><td></td><td>0.889</td><td>0.828</td><td>2.07</td><td>0.5744 0.2940 0.7009</td><td>16.6</td><td>4.8</td><td>9.29</td><td>279s</td><td>$0.13</td></tr><tr><td>Haku Ours w/o validator</td><td>0.586</td><td>0.744</td><td>0.775</td><td>2.99</td><td>0.6390 0.3012</td><td>0.7481 18.3</td><td>7.4</td><td></td><td>119s</td><td>$0.03</td></tr><tr><td>Ours (full)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.00</td><td>0.6523 0.3056</td><td>0.7660 21.5</td><td>7.6</td><td>2.89</td><td>143s</td><td>$0.05</td></tr><tr><td>Codex</td><td>0.086</td><td>0.131</td><td>0.135</td><td>10.11</td><td>0.6504 0.3025 0.7593</td><td>21.1</td><td>6.1</td><td></td><td>150s</td><td>$0.02</td></tr><tr><td>qn9 Ours w/o assembly API</td><td>0.996</td><td>1.000</td><td>0.996</td><td>1.04</td><td>0.3342 0.2586 0.4972</td><td>7.2</td><td>2.0</td><td>15.25</td><td>405s</td><td>$0.06</td></tr><tr><td>Ours w/o validator</td><td>0.529</td><td>0.561</td><td>0.840</td><td>2.03</td><td>0.6975 0.3111</td><td>0.7920 28.5</td><td>8.5</td><td></td><td>510s</td><td>$0.03</td></tr><tr><td>Ours (full)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.00</td><td>0.6945 0.3105</td><td>0.7766 27.4</td><td>7.8</td><td>4.62</td><td>415s</td><td>$0.03</td></tr><tr><td>Codex</td><td>0.641</td><td>0.654</td><td>0.781</td><td>2.00</td><td>0.7610 0.3241</td><td>0.8700 75.9</td><td>15.8</td><td></td><td>346s</td><td>$0.29</td></tr><tr><td>So Ours w/o assembly API 1.000</td><td></td><td>1.000</td><td>1.000</td><td>1.01</td><td>0.7623 0.3228</td><td>0.8573 70.1</td><td>13.9</td><td>10.02</td><td>459s</td><td>$0.40</td></tr><tr><td>Ours w/o validator 6.</td><td>0.779</td><td>0.787</td><td>0.990</td><td>1.03</td><td>0.7611 0.3240 0.8664</td><td>86.0</td><td>19.6</td><td></td><td>328s</td><td>$0.20</td></tr><tr><td>Ours (full)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.00</td><td>0.7610 0.3243 0.8669</td><td>75.8</td><td>16.6</td><td>2.48</td><td>251s</td><td>$0.19</td></tr><tr><td>Astra Codex</td><td>0.693</td><td>0.709</td><td>0.822</td><td>1.87</td><td>0.7609 0.3237</td><td>70.8691 75.5</td><td>15.3</td><td></td><td>244s</td><td>$1.21</td></tr><tr><td>Articraft [59]</td><td>0.684</td><td>0.693</td><td>0.844</td><td>1.77</td><td>0.7690 0.3249</td><td>0.8785 63.2</td><td>16.0</td><td></td><td>426s</td><td>$3.00</td></tr><tr><td>Ours (full) 6</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.00</td><td>0.7695 0.3243</td><td>0.8698 89.3</td><td>18.9</td><td>2.40</td><td>267s</td><td>$1.28</td></tr></table>

## 3.2. Building with BrickLayer

BrickWright is a framework for agentic brick build creation using BrickLayer. It provides a single agent with a set of tools and prompts it to create a build by writing and revising a single Python file in a plan-execute-validate loop, inspired by Articraft [59]. The BrickWright environment exposes the tools validate, which runs agent-written tests alongside fixed collision and floating checks; render, which renders the build into a multi-view sheet or from a single specified view; and finalize, which marks the build as finished.

In its system prompt, BrickWright instructs an agent to make a short internal plan, then write a Python file, build.py, containing two functions: create build(), which returns a LegoBuild, and run tests(), which runs a set of agent-written tests against the completed build. After creating build.py, the agent is instructed to iteratively call validate and edit the build file until all tests pass, then call render; if it judges the render satisfactory, it should then call finalize; otherwise, it should revise build.py and repeat validation. Besides these instructions, BrickWright includes only the BrickLayer documentation in its system prompt, and the input text or image in a user prompt. It does not provide the agent with detailed planning advice, complete build examples, or building tips beyond the plan-execute-validate loop; the loop alone is sufficient for even lightweight agents to produce a feasible build.

BrickWright can be used alongside commercial frameworks such as Codex, which allows general file access and script execution. However, to demonstrate its utility alone, we provide the agent with only two additional tools during experiments: read / write / insert / edit build to read and modify build.py, and explore to run a Python snippet with BrickLayer and the written LegoBuild in scope, returning its output as text.

## 4. Experiments

We compare BrickWright against two types of baselines: specially trained text-to-brick models (BrickGPT [37] and Brick-Net [19]) and pre-trained agentic methods (Articraft [59], OpenAI’s Codex, and Anthropic’s Claude Code). We evaluate the physical feasibility and prompt adherence of the output brick builds. Throughout evaluation, we use the BrickNet-512 [19] dataset, which contains 512 text prompts created by VLM captioning of human-made brick builds.

![](images/1fcb0ecc51ea02a0012ee9de4b61019bc8083478d5d0368f98fc5121391ecd32.jpg)  
Figure 4. Qualitative ablation results. Red outlines in insets indicate mesh intersections. See Sec. C for input text prompts.

Technical details. To test agents with a range of capabilities, we run experiments with Claude Haiku 5.5 and GPT-6 Luna (lightweight), GPT-6.1 Sol (balanced), and GPT-6 Astra (frontier). All agents use High reasoning. Figure examples use Astra unless otherwise specified. Agents are instructed to create builds with no collisions or floaters, and to ensure the build is connected if it comprises one object. Codex and Claude Code baselines may read, write, and execute arbitrary files in a workspace directory, which initially contains only the LDraw documentation [21] and the library of LDraw parts [22], including part meshes. BrickWright agents are only allowed to read and modify the output build.py and execute text-output-only Python snippets; they have no access to the LDraw documentation or LDraw part library beyond the searchable parts list exposed by BrickLayer. Agentic methods have a budget of \$5 per sample; for samples that reach the budget, we use the latest completed build, whether valid or invalid. Only Luna w/o assembly API (2 samples) and Astra with Codex (1 sample), Articraft (34 samples), and full BrickWright (2 samples) reached this limit.

As Articraft is intended for primitive-based 3D generation, we minimally modify it to use a library of LDraw part meshes rather than mesh primitives. Invalid operations on physical parts (e.g., scaling) are disabled, as are collision checks, which yield false positives for coupled bricks, and watertightness checks, which fail on library meshes. We adjust Articraft’s system prompt to reword phrases that do not work with brick builds, such as allowing geometry overlaps.

Metrics. We report metrics on physical feasibility as defined in Sec. 3; prompt adherence as judged by VQAScore [28], Perception Encoder [3], and SigLIP 2 [43]; average brick count per build; and average time and cost per prompt. Validity and connectivity are first evaluated using BrickLayer’s validator. Then, to independently verify the metrics’ accuracy, we flag potential false positives for review by a human experienced with LEGO<sup>®</sup> design. Potential false positives include collisions resolved by a tested shift of at most 0.2 mm and disconnected components containing parts without connector annotations. Human review increases collisionfree rates by 0.2 percentage points for BrickNet and Claude Code (Haiku), 1.4 for Codex (Sol), and 1.0 for Codex (Astra). It reduces mean component counts by 0.13 for Codex (Sol), 0.10 for Codex (Astra), and 0.12 for Articraft; changes for the remaining methods are below 0.02 components per build. Numbers in Tab. 1 are post-review.

## 4.1. Baseline comparison

Compared to vanilla agent baselines, builds produced by BrickWright-equipped agents exhibit greatly improved physical feasibility with broadly comparable token costs (Tab. 1). Impressively, the lightweight model Luna, which has only a 9% baseline validity rate, achieves 100% validity when used with BrickWright. This increase in physical validity is not accompanied by a significant decrease in complexity or fidelity; on average, BrickWright agents use more bricks and more unique parts than vanilla agent baselines, while improving prompt adherence for lightweight models (Haiku, Luna) and maintaining it for powerful models (Sol, Astra). These gains hold even with BrickWright’s minimal harness, whereas Codex and Claude Code can read, write, and execute arbitrary files. BrickWright-equipped agents additionally outperform specially trained text-to-brick models in overall feasibility and prompt adherence. The Articraft baseline, which is intended for digital 3D assets, yields comparable prompt adherence but worse physical validity at more than twice the cost of BrickWright. The visual results in Fig. 6 show how BrickWright reduces physical artifacts for al agent backends and improves the coherence of builds created by lightweight agents. On average, most BrickWright agents use 2–3 validate calls; Luna uses 4–5 (Tab. 1).

## 4.2. Ablation study

We ablate BrickLayer’s two key components: the assembly API and the validator. In our first ablation, “w/o assembly API”, agents write a program that creates an LDraw file without using BrickLayer’s assembly functions. The LegoBuild abstraction is removed from BrickLayer, leaving only part search and physical validation, and its docu-

(e) Constrained generation

(a) Image-conditioned generation

![](images/89943520ecdb97e8cb75885c45683e164bfcb33b0985052cfd0dd5419fd7f52a.jpg)  
(b) Indoor and outdoor scenes, large-scale builds

![](images/2d33bb8f67cba9ccd8ac2112c904076836533a30cc4e66f9be08c4d212c7d96b.jpg)  
“A cozy ramen bar…”

![](images/80970408902fdb7f0283c166b494f041c050ead514ecfe742fbd64df7cf905b4.jpg)  
“An Arctic observatory…”

![](images/b816e84eb37197c18d41ae51cd0b9e53ddc0e58cbebf66ebf7c97225ce02eb82.jpg)  
(c) Articulated builds

![](images/97926b64b40d8cf38d7ce36c7e265af33fea1372eed3e11dcb4731ddf94cbb53.jpg)

![](images/52116d1d75a9f39870270ed503eb7e99605c138b681a72e0b9ea87f275d25487.jpg)

![](images/9164ccb32a0aa88b41b95d2169aa56801ab7a4708a312518a569a4557ea0138c.jpg)  
Figure 5. Extensions. BrickWright naturally supports imageconditioned generation, scenes, articulated builds, editing, and constrained generation. All examples are physically valid and connected. See Sec. C for full input prompts.

mentation is replaced by LDraw documentation of similar length. In our second ablation, “w/o validator”, agents are still told to make their output physically valid but cannot use the validate and render tools or BrickLayer’s physical validation functions. They are allowed to run their own validations using explore. We run ablations using Haiku, Luna, and Sol; we exclude Astra due to budget constraints.

BrickLayer’s assembly API greatly reduces token usage and makes it easier to create valid builds; when forced to work in LDraw rather than using LegoBuild, agents spend around twice the token cost and three times as many validate calls (Tab. 1). Though stronger agents (Sol) still achieve 100% validity with no assembly API, weaker agents (Luna and Haiku) sometimes give up without finalizing the build, leading to validity scores below 100%. Without an assembly API, Luna’s prompt-similarity scores degrade significantly; it often produces simple structures (Fig. 4) and its average part count drops from 27.4 to 7.2. Interestingly, merely using BrickLayer’s API without a provided validator boosts validity by up to 44 percentage points from Codex/- Claude Code baselines! The automatic grid-snapping and connector-mating functions provided by BrickLayer help agents, especially lightweight ones, produce valid builds more easily. However, the validator remains a crucial component; without it, all agents score below 79% validity (Tab. 1).

## 4.3. Extensions

BrickWright works with image conditions, indoor and outdoor scenes, large-scale builds of more than 29,000 parts, articulated objects, build editing by providing a BrickLayer script or LDraw file as input, and special requirements such as being constrained to a given set of bricks (Fig. 5). This functionality naturally arises from pre-trained agent capabilities, without needing to update BrickWright’s toolkit.

## 5. Discussion

Frontier coding agents are capable of producing aesthetically pleasing LEGO<sup>®</sup> brick builds from text or image prompts. However, even the most capable agents fail to produce physically feasible builds much of the time. Our framework, BrickWright, closes this gap by introducing a toolkit for brick build creation and validation. Equipped with Brick-Wright, even lightweight agents achieve 100% validity on 512 samples tested, while improving prompt adherence at a similar cost to agents without BrickWright. We hope this work highlights the importance of well-written construction APIs and physical validation in improving the performance of LLM agents on building tasks.

Limitations and future work. Our definition of physical feasibility does not cover valid assembly sequences or stability analysis, which we leave to future work. Though agentic methods outperform specially trained models on brick build generation, they are generally slower and more costly. Effectively distilling the capabilities of powerful agents into smaller, faster models remains an interesting future direction.

## Acknowledgments

This work is partly supported by the Packard Foundation, Sloan Foundation, and IITP grant funded by the Korean Government (MSIT) (No. RS-2024-00457882, National AI Research Lab Project). Ava Pun is supported by the Meshy Fellowship, Tata Consultancy Services Presidential Fellowship, and NSERC CGRS D. Maxwell Jones is supported by the Rales Fellowship.

![](images/3bf08ac6a8a0d94eee45805bd7573046dc211bd8f9011a00b25dfeee366910bc.jpg)  
Figure 6. Qualitative baseline comparison. BrickWright improves physical feasibility across all agents and visibly boosts output fidelity for lightweight agents (Haiku, Luna), while outperforming specialized brick generation models [19, 37]. See Sec. C for text prompts and Fig. 7 for more results. Red outlines in insets indicate mesh intersections.

## References

[1] Mitch Adler, Matthew Russo, and Michael Cafarella. Agentaided design for dynamic CAD models. In CAIS, 2026. 3

[2] Maneesh Agrawala, Doantam Phan, Julie Heiser, John Haymaker, Jeff Klingner, Pat Hanrahan, and Barbara Tversky. Designing effective step-by-step assembly instructions. ACM TOG, 22(3):828–837, 2003. 3

[3] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Bangalath, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Shang-Wen Li, Piotr Dollar, and Christoph Feichtenhofer. Perception Encoder:´ The best visual embeddings are not at the output of the network. In NeurIPS, pages 60884–60937, 2025. 6

[4] Xilun Chen, Hanwen Wan, Yusong Zhao, Zexin Lin, Ruixiang Liao, and Xiaoqiang Ji. ResemBrick: Brick reconstruction from photographs with perceptual fidelity and buildability. arXiv preprint arXiv:2608.09597, 2026. 2

[5] Yunuo Chen, Tianyi Xie, Zeshun Zong, Xuan Li, Feng Gao, Yin Yang, Ying Nian Wu, and Chenfanfu Jiang. Atlas3D: Physically constrained self-supporting text-to-3D for simulation and fabrication. In NeurIPS, pages 102501–102530, 2024. 3

[6] Godtfred Kirk Christiansen. Toy building brick. U.S. Patent 3,005,282, 1961. Filed July 28, 1958; granted October 24, 1961. 2

[7] Hyunsoo Chung, Jungtaek Kim, Boris Knyazev, Jinhwi Lee, Graham W. Taylor, Jaesik Park, and Minsu Cho. Brick-by-Brick: Combinatorial construction with deep reinforcement learning. In NeurIPS, pages 5745–5757, 2021. 2

[8] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. In ICML, pages 10764– 10799, 2023. 3

[9] Jiahao Ge, Mingjun Zhou, Wenrui Bao, Hao Xu, and Chi-Wing Fu. Creating LEGO® figurines from single images. ACM TOG, 43(4):153:1–153:16, 2024. 2

[10] Jiahao Ge, Mingjun Zhou, and Chi-Wing Fu. Learn to create simple LEGO® micro buildings. ACM TOG, 43(6):249:1– 249:13, 2024. 2

[11] Jiahao Ge, Mingjun Zhou, Hanyou Zheng, Hao Xu, and Chi-Wing Fu. LEGO®-Maker: Autoregressive image-conditioned LEGO® model creation. ACM TOG, 44(6):201:1–201:15, 2025. 2

[12] Yandong Guan, Xilin Wang, XiMing Xing, Jing Zhang, Dong Xu, and Qian Yu. CAD-Coder: Text-to-CAD generation with chain-of-thought and geometric reward. In NeurIPS, pages 59765–59789, 2025. 3

[13] Guangzhao He, Rundong Luo, Wei-Chiu Ma, and Hadar Averbuch-Elor. Thinking in Blender: Staged executable inverse graphics with vision-language models. arXiv preprint arXiv:2606.02580, 2026. 3

[14] Yun He, Kelin Yu, and Matthias Zwicker. SceneOrchestra: Efficient agentic 3D scene synthesis via full tool-call trajectory generation. In ECCV, 2026. 3

[15] Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A. Ross, Cordelia Schmid, and Alireza Fathi.

SceneCraft: An LLM agent for synthesizing 3D scenes as Blender code. In ICML, pages 19252–19282, 2024. 3

[16] R. Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J. Mitra, and Daniel Ritchie. ShapeAssembly: Learning to generate programs for 3D shape structure synthesis. ACM TOG, 39(6):234:1–234:20, 2020. 3

[17] Jungtaek Kim, Hyunsoo Chung, Jinhwi Lee, Minsu Cho, and Jaesik Park. Combinatorial 3D shape generation via sequential assembly. In NeurIPS Workshop on Machine Learning for Engineering Modeling, Simulation, and Design, 2020. 2, 3

[18] Mingi Kim, Yongjun Kim, Jungwoo Kang, and Hyungki Kim. BrepCoder: A unified multimodal large language model for multi-task B-rep reasoning. In ECCV, pages 428–445, 2026. 3

[19] Peter Kulits and Cordelia Schmid. BrickNet: Graph-backed generative brick assembly. In CVPR, pages 39252–39261, 2026. 2, 4, 5, 8

[20] Peter Kulits, Yiqing Xu, R. Kenny Jones, Cordelia Schmid, and Jiajun Wu. BrickBench: Evaluating agentic brick design. arXiv preprint arXiv:2610.12452, 2026. 3

[21] LDraw.org. LDraw documentation. Online documentation, . Accessed: 2026-10-14. 2, 4, 6

[22] LDraw.org. LDraw parts library. Online part library, . Accessed: 2026-10-14. 4, 6

[23] Long Le, Jason Xie, William Liang, Hung-Ju Wang, Yue Yang, Yecheng Jason Ma, Kyle Vedder, Arjun Krishna, Dinesh Jayaraman, and Eric Eaton. Articulate-Anything: Automatic modeling of articulated objects via a vision-language foundation model. In ICLR, 2025. 3

[24] Kyle Lennon, Katharina Fransen, Alexander O’Brien, Yumeng Cao, Matthew Beveridge, Yamin Arefeen, Nikhil Singh, and Iddo Drori. Image2Lego: Customized LEGO set generation from images. arXiv preprint arXiv:2108.08477, 2021. 2

[25] Qixuan Li, Chao Wang, Zongjin He, and Yan Peng. PhiP-G: Physics-guided text-to-3D compositional scene generation. arXiv preprint arXiv:2502.00708, 2025. 3

[26] Ruining Li, Chuanxia Zheng, Christian Rupprecht, and Andrea Vedaldi. DSO: Aligning 3D generators with simulation feedback for physical soundness. In ICCV, pages 6772–6783, 2025. 3

[27] Youtian Lin, Yikang Yang, Zhanpeng Hu, Mengqi Zhou, Feihu Zhang, Xun Cao, Jiaheng Liu, and Yao Yao. Procedura: Agentic 3D modeling with procedural control. arXiv preprint arXiv:2608.26238, 2026. 1, 3

[28] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In ECCV, pages 366–384, 2024. 6

[29] Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Zhaoshuo Li. Scenethesis: A language and vision agentic framework for 3D scene generation. In ICLR, 2026. 3

[30] Jiateng Liu, Bingxuan Li, Zhenhailong Wang, Rushi Wang, Kaiwen Hong, Cheng Qian, Jiayu Liu, Denghui Zhang,

Katherine Driggs-Campbell, Manling Li, and Heng Ji. Brick-Composer: Using MLLMs for assembly with diverse bricks. arXiv preprint arXiv:2606.05445, 2026. 2

[31] Ruixuan Liu, Kangle Deng, Ziwei Wang, and Changliu Liu. StableLego: Stability analysis of block stacking assembly. IEEE RA-L, 9(11):9383–9390, 2024. 3

[32] Sining Lu, Guan Chen, Nam Anh Dinh, Itai Lang, Ari Holtzman, and Rana Hanocka. LL3M: Large language 3D modelers. arXiv preprint arXiv:2508.08228, 2025. 3

[33] Rundong Luo, Haoran Geng, Congyue Deng, Puhao Li, Zan Wang, Baoxiong Jia, Leonidas Guibas, and Siyuan Huang. PhysPart: Physically plausible part completion for interactable objects. In ICRA, 2025. 3

[34] Sheng-Jie Luo, Yonghao Yue, Chun-Kai Huang, Yu-Huan Chung, Sei Imai, Tomoyuki Nishita, and Bing-Yu Chen. Legolization: Optimizing LEGO designs. ACM TOG, 34 (6):222:1–222:12, 2015. 2, 3

[35] Dimitrios Mallis, Ahmet Serda Karadeniz, Sebastian Cavada, Danila Rukhovich, Niki Foteinopoulou, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. CAD-Assistant: Toolaugmented VLLMs as generic CAD task solvers. In ICCV, pages 7284–7294, 2025. 3

[36] Zhengyang Ni, Feng Yan, Yu Guo, and Fei Wang. Brick-Anything: Geometry-conditioned buildable brick generation with structure-aware tokenization. arXiv preprint arXiv:2605.26182, 2026. 2, 3

[37] Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, and Jun-Yan Zhu. Generating physically stable and buildable brick structures from text. In ICCV, pages 14798–14809, 2025. 2, 3, 5, 8

[38] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. 4

[39] Gopal Sharma, Rishabh Goyal, Difan Liu, Evangelos Kalogerakis, and Subhransu Maji. CSGNet: Neural shape parser for constructive solid geometry. In CVPR, pages 5515–5523, 2018. 3

[40] Yuan Shui, Yandong Guan, Zhanwei Zhang, Juncheng Hu, Jing Zhang, Dong Xu, and Qian Yu. ArtiCAD: Articulated CAD assembly design via multi-agent code generation. arXiv preprint arXiv:2604.10992, 2026. 3

[41] Rylee Thompson, Elahe Ghalebi, Terrance DeVries, and Graham W. Taylor. Building LEGO using deep generative models of graphs. In NeurIPS Workshop on Machine Learning for Engineering Modeling, Simulation, and Design, 2020. 2

[42] Yonglong Tian, Andrew Luo, Xingyuan Sun, Kevin Ellis, William T. Freeman, Joshua B. Tenenbaum, and Jiajun Wu. Learning to infer and execute 3D shape programs. In ICLR, 2019. 3

[43] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Henaff, Jeremiah Harmsen, Andreas Steiner,´ and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 6

[44] Cheng Wan, Yongsen Mao, Wenzheng Wu, Yuxuan Xie, Chucheng Xiang, Runze Wang, Xiang Zhang, Zhongyuan Liu, Rushi Dai, and Yuan Liu. NaLA: A 3D native LLM layout agent for high-quality 3D scene generation. In ECCV, 2026. 3

[45] Rui-Huan Wang, Si-Tong Wei, Jia-Qi He, Heng-Yi Wei, Baoquan Chen, and Peng-Shuai Wang. aDSL: Agentic 3D creation via joint agent-program design. arXiv preprint arXiv:2608.17975, 2026. 1, 2, 3

[46] Haowei Wen, Ruixuan Liu, Weiyi Piao, Siyu Li, and Changliu Liu. BrickSim: A physics-based simulator for manipulating interlocking brick assemblies. arXiv preprint arXiv:2603.16853, 2026. 3

[47] Haoyang Xie and Feng Ju. Text-to-CadQuery: A new paradigm for CAD generation with scalable large model capabilities. arXiv preprint arXiv:2505.06507, 2025. 3

[48] Chenhui Xu, Ziyue Bai, Fuxun Yu, Heng Huang, and Jinjun Xiong. Rollback-free stable brick structures generation. arXiv preprint arXiv:2605.06947, 2026. 2, 3

[49] Hao Xu, Ka-Hei Hui, Chi-Wing Fu, and Hao Zhang. Computational LEGO Technic design. ACM TOG, 38(6):196:1– 196:14, 2019. 2

[50] Hao Xu, Yuqing Zhang, Yiqian Wu, Xinyang Zheng, Yutao Liu, Xiangjun Tang, Yunhan Yang, Ding Liang, Yingtian Liu, Yuanchen Guo, Yanpei Cao, and Xiaogang Jin. LegoACE: Autoregressive construction engine for expressive LEGO® assemblies. In SIGGRAPH Asia Conference Papers, pages 40:1–40:11, 2025. 2

[51] Qingshan Xu, Jiao Liu, Melvin Wong, Caishun Chen, and Yew Soon Ong. Looks great, functions better: Physics compliance text-to-3D shape generation. In IJCNN, 2025. 3

[52] Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning. In ECCV, pages 557–575, 2026. 1, 3

[53] Yuhuan Yuan, Zhouliang Yu, Minghao Liu, Weiyang Liu, and Ge Lin Kan. Sample-efficient post-training for LEGO spatial-physics reasoning. arXiv preprint arXiv:2606.07602, 2026. 2

[54] Xiao Zhan, Clement Jambon, Evan Thompson, Kenney Ng,´ and Mina Konakovic Lukovi´ c. PhysiOpt: Physics-driven´ shape optimization for 3D generative models. In SIGGRAPH Asia Conference Papers, pages 109:1–109:11, 2025. 3

[55] Keyou Zheng, Xuyang Su, and Jiewu Leng. SPADA: A verifiable test-driven agent for controllable parametric CAD assembly generation. In ICML, pages 163547–163572, 2026. 3

[56] Guyue Zhou, Liyi Luo, Hao Xu, Xinliang Zhang, Haole Guo, and Hao Zhao. Brick yourself within 3 minutes. In ICRA, pages 6261–6267, 2022. 2

[57] Jie Zhou, Xuejin Chen, and Ying-Qing Xu. Automatic generation of vivid LEGO architectural sculptures. Comput. Graph. Forum, 38(6):31–42, 2019. 2

[58] Mingjun Zhou, Jiahao Ge, Hao Xu, and Chi-Wing Fu. Computational design of LEGO® sketch art. ACM TOG, 42(6): 201:1–201:15, 2023. 2

[59] Matt Zhou, Ruining Li, Xiaoyang Lyu, Zhaomou Song, Zhening Huang, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi, and Shangzhe Wu. Articraft: An agentic system for scalable articulated 3D asset generation. arXiv preprint arXiv:2605.15187, 2026. 1, 2, 3, 5

# BrickWright: Agentic Creation of Physically Feasible Brick Builds

Supplementary Material

## A. Additional results

Table 3 reports “valid@k” metrics: how often agents reach a passing validate call within their first k calls, together with mean token usage. With the assembly API, BrickWright agents pass validation in far fewer calls than when writing LDraw directly; for example, 81% of GPT-6.1 Sol builds pass within two calls, compared with 37% without the API.

Table 2 breaks down the reasons for failed validate calls. Physical checks account for most failures (63–85%), followed by runtime errors (11–28%) such as selecting a nonexistent connector or specifying an invalid part ID. For GPT agents, most physical failures are due to collisions, while Haiku’s main failure mode is connectivity. The more capable an agent, the more its physical failure distribution shifts toward collisions, away from floaters and connectivity.

Additional qualitative comparisons are given in Fig. 7.

## B. System prompt and BrickLayer reference

Listing 2 shows the system prompt used for BrickWright agents, and Listing 3 shows the user prompt template used for the Codex and Claude Code baselines. The BrickWright prompt also includes the BrickLayer API reference; Listings 4 to 8 show excerpts of the reference documenting the LegoBuild, Element, Placement, LegoGeometry, and Connector classes.

## C. Example prompts

We provide full prompts for all examples used in the figures.

Teaser (Fig. 1). Starry Night: “Recreate this painting as a sculptural 3D LEGO scene using varied pieces. No mosaics.” Combine: “Match the image as closely as possible.” Seattle Needle: “Seattle’s Space Needle, with tapered legs, a saucer observation deck, and its slender spire.” Crab: “A poseable crab with eight jointed legs, opening claws, and raised eyestalks.” Harbour: “Hong Kong’s Victoria Harbour in 10,000+ parts, with dense skyscrapers and a redsailed junk.” Spaceship: “This is a white and black LEGO spaceship with a blue tinted canopy, featuring a blue-suited minifigure pilot, rear thrusters, and side-mounted weapons.”

Baseline comparison (Fig. 6). ATV: “This is a LEGO build of a blue, grey, and black all-terrain vehicle (ATV) or quad bike, featuring large treaded wheels, a roll cage, and translucent yellow headlights.” Copter: “This is a LEGO build of a blue and white utility helicopter, featuring an open cockpit, black landing skids, and a tail rotor.” Bug: “This is a custom-built LEGO build of a fantastical, insectoid creature, constructed primarily from dark blue and black pieces with vibrant magenta accents for its claws, tail stinger, and decorative elements, showcased from multiple angles to highlight its articulated limbs and unique design.” Fox: “This is a stylized, geometric LEGO build of a sitting fox, constructed primarily from orange bricks with white accents on the chest and snout, black pieces for the paws and nose, and smal orange slopes for ears.” Ninja: “This is a blocky, stylized LEGO minifigure of a ninja or martial artist, constructed primarily from black bricks with white and brown accents for its headband and torso, and posed dynamically as if in mid-action.” Flowers: “This LEGO build is a modular, multi-angle representation of a cluster of white flowers with green leaves and stems, constructed from various specialized plant-themed pieces.” Scene: “This LEGO build depicts a small, fenced-in backyard scene featuring a brown shed with an attached silo, a blue pond, green landscaping, and a colorful, abstract sculpture.” Coffee Machine: “This LEGO build is a detailed, dual-grouphead espresso machine with an attached coffee grinder, all mounted on a brown baseplate.”

Table 2. Validation failure types for BrickWright agents on BrickNet-512. Failure types are given as fractions of failed calls. Runtime errors are exceptions while running create build() or run tests(); custom test failures pass all physical checks but fail an agent-written test. No calls failed for other reasons. Indented rows break down physical failures as fractions of physical failures; a call can fail several physical checks, so they may sum to > 1.
<table><tr><td></td><td>Haiku 5.5</td><td>6 Luna</td><td>6.1 Sol</td><td>6 Astra</td></tr><tr><td>validate calls Failed calls</td><td>1,481 805</td><td>2,364 1,598</td><td>1,269 477</td><td>1,230 393</td></tr><tr><td>Syntax error</td><td>0.000</td><td>0.003</td><td>0.000</td><td>0.003</td></tr><tr><td>Runtime error</td><td>0.143</td><td>0.281</td><td>0.130</td><td>0.109</td></tr><tr><td>Physical test</td><td>0.820</td><td>0.635</td><td>0.813</td><td>0.847</td></tr><tr><td>Collision</td><td>0.505</td><td>0.796</td><td>0.899</td><td>0.895</td></tr><tr><td>Floater</td><td>0.439</td><td>0.317</td><td>0.170</td><td>0.105</td></tr><tr><td>Connectivity</td><td>0.767</td><td>0.557</td><td>0.363</td><td>0.297</td></tr><tr><td>Custom test</td><td>0.037</td><td>0.081</td><td>0.057</td><td>0.041</td></tr></table>

Additional baseline comparison (Fig. 7). Car: “This is a red, open-top LEGO sports car featuring black racing stripes, yellow accents on the front and rear, and translucent orange headlights.” Boat: “This is a LEGO model of a small workboat or tugboat, featuring a red hull, an orange and white cabin with blue windows, a grey deck, and a black articulated crane on the stern.” Train: “This is a small, stylized LEGO model of a yellow and black steam locomotive, featuring a prominent smokestack, a curved boiler, a cab, and simple wheels, presented from multiple angles against a white background.” Snake: “This is a modular, dark green LEGO model of a snake or serpent, constructed from multiple articulated segments that allow it to be posed in various curved and straight configurations.” Squirrel: “This is a LEGO model of a black squirrel, constructed with dark grey and light grey bricks, featuring a large bushy tail, prominent white eyes, and holding two small, round objects in its paws.” Dragon: “This is a custom-built, orange LEGO model of a stylized, blocky creature resembling a dragon or dinosaur, featuring translucent flame elements at its rear and amber pieces for eyes.” Station: “This LEGO model is a white, multi-level medical or scientific station featuring an articulated lamp with a yellow lens, a control panel, and a cabinet with a yellow door.” Garden: “This LEGO model depicts a charming, rotating diorama of a small garden scene, featuring a pumpkin patch with a white fence on one side and a cylindrical well or shed with a ladder on the other, all built upon interconnected green baseplates.”

Table 3. Validation success and token usage on BrickNet-512. valid@k is the fraction of builds whose first k validate calls include at least one passing call (all physical checks and custom tests pass); it applies only to methods with access to validate. Tokens are mean thousands of tokens per prompt; total tokens are the sum of input tokens (including cached input) and output tokens (including reasoning). Bold marks the best valid@k within each agent group.
<table><tr><td></td><td colspan="5">valid@k↑</td><td colspan="3">Tokens (K)↓</td></tr><tr><td>Method</td><td>k=1</td><td>k=2</td><td>k=3</td><td>k=4</td><td>k=5</td><td>Total</td><td>Input</td><td>Output</td></tr><tr><td>5 Claude Code</td><td></td><td></td><td></td><td></td><td></td><td>603.2</td><td>570.0</td><td>33.2</td></tr><tr><td>Ours w/o assembly API</td><td>0.143</td><td>0.324</td><td>0.439</td><td>0.527</td><td>0.613</td><td>1,900.7</td><td>1,852.2</td><td>48.5</td></tr><tr><td>Ha Ours w/o validator</td><td></td><td></td><td></td><td></td><td></td><td>562.2</td><td>540.8</td><td>21.3</td></tr><tr><td>Ours (full)</td><td>0.320</td><td>0.637</td><td>0.801</td><td>0.908</td><td>0.959</td><td>846.4</td><td>821.7</td><td>24.7</td></tr><tr><td>Codex</td><td></td><td></td><td></td><td></td><td></td><td>495.5</td><td>482.6</td><td>12.9</td></tr><tr><td>Ours w/o assembly API</td><td>0.217</td><td>0.334</td><td>0.482</td><td>0.596</td><td>0.678</td><td>3,324.8</td><td>3,288.5</td><td>36.4</td></tr><tr><td>69 Ours w/o validator</td><td></td><td></td><td></td><td></td><td></td><td>719.9</td><td>678.6</td><td>41.3</td></tr><tr><td>Ours (full)</td><td>0.268</td><td>0.500</td><td>0.646</td><td>0.750</td><td>0.814</td><td>1,282.0</td><td>1,249.8</td><td>32.2</td></tr><tr><td>Codex</td><td></td><td></td><td></td><td></td><td></td><td>654.3</td><td>644.6</td><td>9.6</td></tr><tr><td>So Ours w/o assembly API</td><td>0.164</td><td>0.369</td><td>0.535</td><td>0.658</td><td>0.732</td><td>1,825.0</td><td>1,810.6</td><td>14.4</td></tr><tr><td>61 Ours w/o validator</td><td></td><td></td><td></td><td></td><td></td><td>334.7</td><td>321.3</td><td>13.4</td></tr><tr><td>Ours (full)</td><td>0.475</td><td>0.807</td><td>0.932</td><td>0.977</td><td>0.992</td><td>528.3</td><td>519.0</td><td>9.4</td></tr><tr><td>Codex</td><td></td><td></td><td></td><td></td><td></td><td>458.5</td><td>452.1</td><td>6.3</td></tr><tr><td>Astra Articraft [59]</td><td></td><td></td><td></td><td></td><td></td><td>1,540.2</td><td>1,528.0</td><td>12.1</td></tr><tr><td>Ours (full) 6</td><td>0.545</td><td>0.879</td><td>0.953</td><td>0.984</td><td>0.996</td><td>544.5</td><td>533.5</td><td>10.9</td></tr></table>

Ablation study (Fig. 4). Piano: “This LEGO model is a compact, white grand piano with tan and black keys, featuring a hinged lid and built from various brick types including technic pieces.” Scene: “This LEGO model depicts a dilapidated, single-room shelter or bunker with crumbling brown and tan walls, containing a simple bed, a rifle, a sack, and a small cache of money.”

Extensions (Fig. 5). Image-conditioned generation: All three examples use the prompt “Match the image as closely as possible.” Ramen bar: “A cozy ramen bar interior with an open kitchen and diners.” Arctic observatory: “An Arctic observatory beneath the aurora.” Mountainside city: “A mountainside city in 15,000+ parts, with terraced houses, winding stairs, bridges, and waterfalls.” Build editing: Starting prompt—“A compact timber-framed cottage with a steep red roof.” Edit 1—“Double the cottage’s depth; preserve its width, height, and timber-framed style.” Edit 2—“Reshape its roof into a dark-blue gambrel, keeping the stretched walls and chimney.”

## D. Brick build generation with vanilla agents

We give a short analysis of the brick build creation process of baseline GPT-6 Astra, used with Codex. We inspect the Python programs produced and executed by the agent during generation (whether inline or from a file) and classify them as generators and/or validators by using heuristics that trace the program’s abstract syntax tree (AST). If a program calls write text() or write bytes() on a model.ldr file or opens the file in a mode with write access, we classify it as a generator. If a program reads model.ldr and includes an assertion or error involving values derived from model.ldr, we classify it as a validator. Out of 512 builds created using prompts from BrickNet-512, 471 (92%) use a generator and 424 (83%) use a validator.

![](images/780a05aaa347efa65fc71ce0fac39e42a8378dcc31065189d5844d2c610e77a4.jpg)  
Figure 7. Additional qualitative baseline comparison. See Sec. C for text prompts. Red outlines in insets indicate mesh intersections.

```markdown
# BrickWright Instructions
You are BrickWright, an expert LEGO builder. Edit ‘build.py‘ so ‘create_build()‘ returns a ‘LegoBuild‘ faithful to
the request and ‘run_tests()‘ returns its ‘TestContext‘ report. The BrickLayer API reference below describes
construction, coordinates, connectors, parts, and colors; use it to resolve API questions.
The finished build must meet these requirements:
- No collisions: parts may touch and mate, but must not interpenetrate.
<sub>**</sub>No floating parts:<sub>**</sub> every part must reach the ground through mated connections or supporting surface contacts.
Resting on tiles, ledges, and slopes counts; side contact alone does not. Sliding and tipping are not checked.
- <sub>**</sub>Connected:<sub>**</sub> one connected assembly, unless the request calls for separate pieces. Face contact alone does not
connect parts. Use compatible connectors: studs/antistuds, pins/holes, axles, clips/bars, ball joints, and others.
Stagger ordinary brick layers or bridge adjacent pieces with a shared plate.
- <sub>**</sub>Faithful:<sub>**</sub> preserve the requested identity, proportions, features, and detail at a believable LEGO scale. Do not
remove requested features merely to pass checks.
## Checks in build.py
The harness independently enforces no floating parts and no collisions on the built model. Connectivity and
request-specific checks come from ‘run_tests()‘. Keep ‘check_no_floating()‘, ‘check_no_collisions()‘, and
‘check_connected()‘ in ‘run_tests()‘. Change ‘expected_components‘ only when the request genuinely calls for multiple
assemblies. Add ‘tests.check(name, ok, detail)‘ for measurable re uirements; ins ect failures and re air the build
rather than weakening those requirements.
‘‘‘python
import bricklayer as bl
from brickwright import TestContext
def create_build():
build = bl.LegoBuild()
build.add(bl.Element("3001", bl.Color.RED))
return build
def run_tests():
build = create_build()
tests = TestContext(build)
tests.check_no_floating()
tests.check_no_collisions()
tests.check_connected(expected_components=1)
# Add checks for this request, e.g. tests.check("over 100 parts", build.brick_count > 100).
return tests.report()

For articulated builds, declare intended movement with ‘MotionRange‘ on the relevant connections. Optionally add
‘tests.check_motion(samples=5)‘ to sample the annotated ranges for collisions, floating parts, and connector
disengagement. Other connections stay rigid, which can produce false positives if they would also move in reality.
This check is optional and covers sampled poses, not every intermediate position. ‘build.check_motion()‘ exposes its
detailed report for investigation. Static validity checks do not test motion.
## Working method
1. Plan the defining features, rough size, support structure, and specialized parts.
2. Look up catalog parts before using remembered IDs. Use ‘Element(part_id, color)‘ as the normal constructor; size
helpers are conveniences. Check their listed argument combinations: dimensions are width-first, and changing
orientation requires rotation, not swapping dimensions to an unsupported size.
3. Build in manageable increments. Validate, read each failure’s detail, and repair it. Use helpers and sub-models
for repeated or meaningful assemblies.
4. When rendering is available, inspect the build while working and after the final edit. Physical validity alone
does not establish visual fidelity.
5. Run the checks on the finished source. Keep ‘create_build()‘ deterministic and free of file writes so checks,
previews, and exports represent the same build.
{docs}
```  
Listing 2. BrickWright system prompt. {docs} is replaced by tool documentation (e.g. for validate) and the BrickLayer reference.

Build a LEGO model matching the following description. Save the complete LDraw code in model.ldr in the current   
workspace.   
The model must be an assembly of real, existing LEGO parts from the LDraw parts catalogue, referenced by their exact   
.dat filenames. Use only type-0 comments and metadata and type-1 part or submodel references. Do not emit type-2,   
type-3, type-4, or type-5 geometry, invent part IDs, or define custom part geometry. If you use MPD submodels, embed   
them in model.ldr and build each entirely from catalogued parts. Every leaf reference must resolve to a real   
catalogue part.   
Avoid colliding or floating parts. If the model depicts a single object, all its parts must form one connected   
component through valid LEGO connections.   
Description:   
{caption}  
Listing 3. Codex/Claude Code baseline prompt. {caption} is replaced with the text prompt.

```ini
#### class ‘LegoBuild(LegoGeometry)‘
‘LegoBuild(placements: Iterable[Placement] = (), names: Mapping[str, Placement] | None = None, steps: Iterable[int] |
None = None, , joints: Iterable[Joint] = ()) -> None‘
A build or reusable sub-assembly, edited in place.
Validity queries update automatically after edits, including edits to nested models. ‘is_valid‘ checks collisions and
ground support; ‘is_connected‘ checks whether every part joins through compatible connectors.
Properties
‘bounding_box: tuple[Position, Position]‘ -- The physical bounding box spanning every part’s, in the model’s frame.
A sub-model holding no parts spans nothing, so a model with no parts returns ‘‘(Position.ZERO, Position.ZERO)‘‘.
‘brick_count: int‘ -- The total number of bricks in the model.
‘colliding_placements: tuple[tuple[Placement[Element], Placement[Element]], ...]‘ -- Pairs of placements whose
collision volumes overlap.
‘connected_components: tuple[tuple[Placement[Element], ...], ...]‘ -- The placements grouped by connected component.
Each group holds the parts joined to one another through couplings, with the groups and the parts within them in
placement order. An empty model has no components.
‘connectors: frozenset[Connector]‘ -- Unoccupied connector points in this model’s frame.
A connector disappears here when it mates with another part, including across nested sub-model boundaries. Covering
studs can therefore change the model’s alignment box and how subsequent parts stack on it.
‘flattened_joints: tuple[Joint, ...]‘ -- All joint occurrences, with endpoint paths rooted in this model.
A repeated sub-model contributes distinct records for each occurrence. Use these records as keys to ‘‘posed‘‘ when
moving a nested joint.
‘flattened_placements: tuple[Placement[Element], ...]‘ -- Every element placement, flattened into this model’s own
frame.
Flattening reaches through nested sub-models, so each leaf holds an :class:‘Element‘.
‘floating_placements: tuple[Placement[Element], ...]‘ -- Parts without a path to the ground through connectors or
contact.
The lowest physical extent defines the ground, as in ‘translate_to_ground‘. The model’s poses remain unchanged, and
returned placements are the original leaves.
Mated connectors transmit support in both directions. Surface contacts transmit it from the supporting part to the
part resting on it, including on tiles, elevated ledges, and slopes. Sliding and tipping are not checked.
‘joints: tuple[Joint, ...]‘ -- Connections recorded in this model; nested records stay in their owner.
Removing either endpoint, including a nested leaf, deletes its record.
‘names: Mapping[str, Placement]‘ -- The model’s named placements, keyed by the name each was registered under.
A live read-only view: names appear as parts are added.
‘num_connected_components: int‘ -- The number of connected components of the model.
‘placements: tuple[Placement, ...]‘ -- Every top-level placement in the model, in insertion order.
```  
Listing 4. LegoBuild documentation. A LegoBuild represents a multi-part assembly containing placed elements or sub-builds. It supports adding/removing placements and validating the build’s physical feasibility. Excerpt from the BrickLayer API reference included in the BrickWright system prompt. Continued on the next two pages.

![](images/f91bd99a81764b4f653b2ef058294ee7a29884625d5bc39e4f360f3f6a0dee89.jpg)  
LegoBuild documentation (continued).

‘copy() -> LegoBuild‘ -- A shallow copy holding the same placements under the same names.   
Placements and nested sub-models are shared. Build-step markers carry over. Adding to either outer model leaves the   
other alone; editing a shared sub-model affects both. Use :meth:‘deepcopy‘ to isolate those edits.   
‘deepcopy() -> LegoBuild‘ -- Copy this model and nested sub-models, isolating subsequent edits.   
Names, poses, and build steps are preserved. Repeated sub-models stay shared within the copy, but no mutable   
content is shared with the original.   
‘extend(other: LegoBuild, position: Position | tuple[float, float, float] = Position(0, 0, 0), rotation: Rotation =   
Rotation.DEG\_0, anchor: Anchor = Anchor.FRONT\_RIGHT) -> tuple[Placement, ...]‘ -- Splice ‘other‘’s current top-level   
placements into this model.   
The resulting geometry matches ‘add(other, position, rotation, anchor)‘, but without a wrapper sub-model or its   
names. Returns the new placements in order. Later additions to ‘other‘ do not carry over; nested content remains   
shared. Raises ‘ValueError‘ for off-grid positions or cyclic nesting.   
‘is\_connected() -> bool‘ -- Whether the model has a single connected component.   
‘is\_valid() -> bool‘ -- Whether the model has no collisions or floating parts.   
‘joint(child: Placement) -> Joint‘ -- The locally recorded joint that placed ‘‘child‘‘ using ‘‘connect‘‘.   
‘list\_elements() -> Counter[Element]‘ -- How many of each element -- part and color -- the model uses.   
- ‘name\_of(placement: Placement) -> str | None‘ -- The name ‘‘placement‘‘ was registered under, or ‘‘None‘‘ if   
unnamed.   
‘named(name: NameRef) -> Placement‘ -- Look up a name or nested path such as ‘[’tower’, ’roof’]‘.   
Returns a placement in this model’s frame, suitable for ‘relative\_to‘ or ‘on‘. That placement remains usable after   
removal, but looking up its removed name raises ‘KeyError‘. An empty path or traversal through an element raises   
‘ValueError‘.   
‘posed(positions: Mapping[Joint, JointCoordinates]) -> LegoBuild‘ -- A flat geometry snapshot at joint coordinates,   
leaving this build intact.   
Keys come from ‘‘joints‘‘ or ‘‘flattened\_joints‘‘. Unspecified positions stay at zero. Static connections and   
attached rigid assemblies follow the moving parts. The snapshot has no joint records; evaluate every pose from the   
original build. Closed motion loops raise ‘‘ValueError‘‘.   
‘remove(target: int | str | Placement) -> Placement‘ -- Remove and return a top-level placement by index, name, or   
identity.   
Negative indices count from the end. Repeated placements lose their first occurrence; the name is freed after the   
last. Build steps track the removal. Nested name paths are not accepted: edit the sub-model itself, which affects   
all models sharing it. Missing indices/names raise ‘IndexError‘/‘KeyError‘; unknown placements raise ‘ValueError‘   
and unsupported target types ‘TypeError‘.   
‘set\_motion(joint: Joint | Placement, motion: MotionRange | None) -> Joint‘ -- Replace a local joint’s range;   
‘‘None‘‘ makes it static. Return the record.   
‘stack(content: Element | LegoBuild, on: NameRef | Placement, offset: Position | tuple[float, float, float] =   
Position(0, 0, 0), rotation: Rotation = Rotation.DEG\_0, anchor: Anchor = Anchor.FRONT\_RIGHT, target\_anchor: Anchor =   
Anchor.FRONT\_RIGHT, <sub>\*</sub>, name: str | None = None, index: int | None = None) -> Placement‘ -- Place content on top of   
‘on‘ (a name, name path, or placement).   
The new part’s alignment-box bottom rests at the reference’s alignment-box top. ‘offset‘ shifts from there in   
studs/bricks and must be on the grid. Use ‘anchor=Anchor.CENTER, target\_anchor=Anchor.CENTER‘ to center footprints;   
anchors, absolute rotation, shared content, naming, and insertion behave as in ‘add‘.   
A concrete reference is read in this model’s frame even if removed or from another model. Later changes to that   
reference do not move this placement.  
LegoBuild documentation (continued).

![](images/e33b3ac70df410baabc57a13528e8dea9d14fb96a32583210cbe265b63047a81.jpg)  
Listing 5. Element documentation. An Element represents a single colored part. Excerpt from the BrickLayer API reference.

```markdown
#### class ‘Placement(LegoGeometry)‘
An element or sub-model placed in a containing model.
‘content‘ and ‘pose‘ are fixed. Geometry is expressed in the containing model’s frame and updates if the content is a
mutable sub-model. Placements compare by identity; two placements can hold identical elements at different positions.
Properties
‘bounding_box: tuple[Position, Position]‘ -- The content’s physical bounding box, in world coordinates.
‘connectors: frozenset[Connector]‘ -- The content’s connector points, in world coordinates.
‘pose: Pose‘ -- Maps the content’s local frame into the model’s.
Methods
- ‘to_world(p: Position | tuple[float, float, float]) -> Position‘ -- Map a position from the content’s local frame
to world coordinates.
```

Listing 6. Placement documentation. A Placement pairs an element or nested build with a rigid transformation. Excerpt from the BrickLayer API reference.

#### class ‘LegoGeometry‘   
Geometry shared by elements, placements, and models.   
Elements use local coordinates; placements use the containing model’s frame. Models use the frame they were built in.   
Measurements update when content changes.   
<sub>\*\*</sub>Properties<sub>\*\*</sub>   
‘alignment\_box: tuple[Position, Position]‘ -- The ‘(min, max)‘ box used for grid placement and stacking.   
Grid-aligned connectors contribute 1x1 stud footprints and determine the bottom. The highest upward stud sets the   
top; without one (e.g. a tile), the physical bounding box sets the top. Models use only their unoccupied connectors.   
Without grid-marked connectors, it is derived from the physical bounding box. Width/depth are whole studs and   
height is in whole plates; thin parts can have zero alignment extent. Inspect the box rather than guessing offsets.   
‘alignment\_depth: int‘ -- Extent along Y (studs) of the alignment box. Always an integer.   
‘alignment\_height: float‘ -- Extent along Z (bricks) of the alignment box. Always an integer multiple of PLATE.   
‘alignment\_width: int‘ -- Extent along X (studs) of the alignment box. Always an integer.   
‘center: Position‘ -- The midpoint of the bounding box, in BrickLayer coordinates.   
Empty geometry returns ‘‘Position(0, 0, 0)‘‘.   
‘depth: int | float‘ -- Extent along Y (studs) of the bounding box.   
‘height: int | float‘ -- Extent along Z (bricks) of the bounding box.   
‘width: int | float‘ -- Extent along X (studs) of the bounding box.   
Methods   
‘find\_connector(connector\_type: ConnectorType | str | Iterable[ConnectorType | str] | None = None, toward: Position   
| tuple[float, float, float] | Direction | None = None, , facing: Direction | Vec3 | None = None) -> Connector‘ --   
Find one connector using the filters and ordering of :meth:‘find\_connectors‘.   
With ‘‘toward‘‘, return the first ranked match. Otherwise require exactly one match. Raises ‘‘ValueError‘‘ if none   
match or the choice is ambiguous. ‘‘connect‘‘ takes a world-frame target connector from a placement.   
‘find\_connectors(connector\_type: ConnectorType | str | Iterable[ConnectorType | str] | None = None, toward:   
Position | tuple[float, float, float] | Direction | None = None, , facing: Direction | Vec3 | None = None, count:   
int | None = None) -> list[Connector]‘ -- The geometry’s connectors, filtered and ordered by the given criteria.   
‘‘connector\_type‘‘ filters types; ‘‘facing‘‘ filters axis direction. ‘‘toward‘‘ orders nearest a position or   
furthest in a direction; ‘‘None‘‘ uses a stable geometric order. ‘‘count‘‘ caps the results (default: all).   
Coordinates are local for an element, world for a placement.   
‘rotated\_footprint(rotation: Rotation) -> tuple[int, int]‘ -- The alignment box’s ‘‘(width, depth)‘‘ in studs after   
‘‘rotation‘‘.   
An odd quarter turn swaps the two; a half turn leaves them as they are.  
Listing 7. LegoGeometry documentation. LegoGeometry is the base class of Element and LegoBuild, exposing dimension and connector information. Excerpt from the BrickLayer API reference.

![](images/619c9e6153f437d4fa44e098a7bd6e2f256ca4f60f54cd5a227d176ff409f17a.jpg)  
Listing 8. Connector documentation. A Connector is a typed mating point on an element or build. Connectors come with a local frame and length; compatibility and relative frame alignment determine valid connections. Excerpt from the BrickLayer API reference.