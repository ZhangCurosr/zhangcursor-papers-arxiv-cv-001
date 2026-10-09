Model A pelican riding a bicycle.

# BRICKBENCH: EVALUATING AGENTIC BRICK DESIGN

Peter Kulits<sup>1,2</sup> Yiqing Xu<sup>1</sup> R. Kenny Jones<sup>1</sup> Cordelia Schmid<sup>3</sup> Jiajun Wu<sup>1</sup> <sup>1</sup>Stanford University <sup>2</sup>Max Planck Institute for Intelligent Systems <sup>3</sup>Inria

![](images/d054361ee87c4d1556f604a1975829a5af09c8f4dc0b278b49b50d0ef486b8a4.jpg)

![](images/978481d09674be9fa9e349e43b9ae654d43b6889e42bfe9c4def89fc54ee023a.jpg)  
Set A brown dog with floppy black ears and a small brown hat sits at a table in a burning room. A white cofee mug rests in front of it. Flames climb the curtains and surround the chair, but the dog sits upright with its paws resting calmly on the tabletop. The front of the room is open to reveal the scene.

![](images/58c54ac0f9f82ac79070e70a85ced1de2f610271b872296e92e944b198a5a39d.jpg)  
Alt-Build A cowboy in a wide-brimmed hat sits on a saddled horse beside a cactus with two raised arms. He holds a looped lasso above his head. The horse’s front legs stand on a low rock, while its rear legs remain on the sandy ground.

GPT-6 Astra Claude Opus 5.5 Qwen 3.8 Flash Gemini 3.8 Flash Muse Spark 1.3

Figure 1: We evaluate coding agents on their ability to design LEGO assemblies from a text prompt. We introduce three settings with varying constraints: Model (≤400 parts), Set (400– 4000 parts), and Alt-Build (the part inventory of set 10698). Each assembly above can be built.

## ABS TR AC T

We propose BrickBench, a benchmark for agentic text-conditioned LEGO-set design. Given a prompt, an agent is tasked with producing an assembly that not only satisfies semantic and design criteria, but that can also be physically built. To do so, it must select parts from a discrete library and reason jointly about local and global constraints. We score validity, alignment, and design across three settings that vary in scale and part availability. We provide BrickAgent, an environment for coding agents to construct, inspect, and validate their designs. We find that leading agents largely satisfy verifiable physical and semantic requirements, but fall short of human designs. We release our benchmark and environment at http://www.brickben.ch

## 1. Introduction

Designing a LEGO assembly is deceptively difficult. A designer must select from thousands of parts, arrange them into a recognizable form, and ensure that the structure holds together. Computational generative approaches to brick assembly have relied on specialized models and carefully engineered representations to handle these challenges (Chung et al., 2021; Pun et al., 2025; Kulits & Schmid, 2026). But coding agents offer a different approach: they can write programs, execute them, inspect the results, and revise. This shift is already transforming software engineering (Jimenez et al., 2024), graphics (Gu et al., 2025), and CAD (Zhang et al., 2026). We find that a general coding agent, without task-specific training, substantially outperforms specialized LEGO-generation models on the BrickNet benchmark (Kulits & Schmid, 2026), approaching the semantic alignment of its reference assemblies (Fig. 2).

Yet does this mean agents have learned to design LEGO? Existing evaluations leave two important challenges unexplored. First, physical complexity: BrickNet evaluates small objects of at most 100 parts, far from the hundreds or thousands of interconnected pieces and inventory constraints of a full LEGO set. Second, and more fundamentally, design quality: making an assembly that works is not the same as making an assembly worth building. The appeal of LEGO lies not merely in reproducing an object, but in the creativity of its construction: clever part usage, shapes, proportions, and details. Two builds can be equally valid and faithful to a description, yet differ dramatically in how well they are designed. Unlike physical validity or semantic alignment with a prompt, design quality has no definitive pass/fail test.

We introduce BrickBench, a benchmark for agentic text-conditioned LEGO-set design (Fig. 1). BrickBench contains 300 prompts across three settings: Model (at most 400 parts), Set (400– 4000 parts), and Alt-Build (a fixed inventory of parts). To support complex construction, we introduce BrickAgent, an environment for agentic LEGO design including construction tools and interpretable validation of connectivity, collisions, and stability. Rather than prescribing how to design an assembly, BrickAgent provides the building blocks, constraints, and feedback, leaving the agent to devise its own solution. To evaluate what validity and semantic alignment miss, we measure design quality through pairwise judgments, validated against human preferences.

We evaluate eleven frontier agents and two data-driven baselines. With BrickAgent, leading agents produce valid assemblies for nearly every prompt, and the best satisfies approximately 95% of semantic requirements. Removing the environment sharply reduces physical validity, showing that reliable construction depends on grounded tools and feedback. Design quality, however, separates agents far more strongly, and human raters identify human-designed assemblies in 323 of 360 comparisons. These findings reveal a shift in the challenge of LEGO design: Agents are learning to build what works, but not yet what makes a great design.

Our contributions are as follows.

1. BrickBench, a benchmark of 300 prompts in three settings for agentic brick design, with metrics for physical validity (connections, collisions, and stability under simulation), semantic alignment (VQA, over yes/no questions decomposed from each prompt), and design quality (ELO, from pairwise judgments by a VLM validated against human raters).

2. BrickAgent, an environment enabling coding agents to programmatically construct, inspect, and validate complex LEGO assemblies, with part search, placement by connectors, subassemblies, rendering, and a validator that names parts at fault. Without it, the share of valid assemblies falls from 100% to 40% for GPT-6 Astra and to under 1% for GPT-5.6 Luna.

3. A systematic evaluation of eleven frontier agents and two data-driven baselines, with human studies that validate the design metric and measure the gap to human design.

## 2. Related Work

Agentic Benchmarks. Benchmarks for coding agents validate an agent’s work by executing it. SWE-bench (Jimenez et al., 2024) runs a repository’s tests on a patch an agent supplied, and OSWorld (Xie et al., 2024) tracks state through execution. SceneCraft (Hu et al., 2024) writes Blender scripts, and CAD-Assistant (Mallis et al., 2025) writes CAD programs. Benchmarks then execute the program and score the result. BlenderGym (Gu et al., 2025) compares the result with a goal scene, and BenchCAD (Zhang et al., 2026) executes CadQuery programs for industrial parts. BuildArena (Xia et al., 2026) scores machines assembled from modular parts by their simulated behavior, and PhyBlock (Ma et al., 2025) scores block placements for plausibility in a physics simulator. BrickBench executes the agent’s program into an assembly of real LEGO parts and validates connections, collisions, and stability under simulation.

LEGO and Evaluation. Prior work generates LEGO assemblies with learned models and uses LEGO tasks to evaluate others. BrickGPT (Pun et al., 2025) fine-tunes a language model to place eight brick types on a 20×20×20 grid from a caption. BrickNet (Kulits & Schmid, 2026), predicts connections rather than positions and autoregressively generates structures through typed connectors. LEGO-Maker (Ge et al., 2025) produces building facades conditioned on an image. Other work evaluates agents and multimodal models on LEGO tasks. Break and Make (Walsman et al., 2022) gives an agent an LDraw model in a simulator to test whether it can inspect it, take it apart, and rebuild it from scratch. LEGO-Puzzles (Tang et al., 2026) presents multimodal models spatial questions about assembly steps. BrickBench does both: general coding agents generate set-scale assemblies from a text prompt, while we score the assemblies on physical validity, prompt alignment, and design.

## 3. Preliminaries

## 3.1. LDraw

LDraw is a community-maintained part library and standard for digital LEGO modeling<sup>1</sup>. Each library part is intended to mirror a real-world counter-part. Assemblies are represented as text files of instances defined by their part type, color, and 6D pose. LDraw does not itself, however, determine whether an assembly holds together or could be built.

## 3.2. Connectivity and Physical Validity

BrickNet (Kulits & Schmid, 2026) annotates each part in the LDraw library with typed connectors. Each connector instance has a type, a subtype, a polarity, and a frame. The subtype names the connector’s geometry within its type and fixes what it mates with: a stud mates with a hole or a tube, and a bar with a clip. The polarity distinguishes the two halves of a hinge, ball, or fixed joint, which must be opposite to connect. The frame is the connector’s position and orientation in the coordinates of its part, with one axis along the direction of mating. Two part instances are connected when a compatible pair of their connectors is aligned up to tolerance.

There are five connector types. A stud connector allows rotation about the connection axis, a hinge allows rotation and a possible flip, and an axle additionally allows a slide along its axis. A ball allows free rotation, and a fixed-type connector allows no motion.

The connections between instances of an assembly form a graph over parts. An assembly may have more than one component to represent a scene with multiple (disconnected) objects. We define an assembly as stable if it remains at rest under gravity. We treat connected components as rigid bodies of the assembly and simulate them as single rigid bodies of uniform density in PyBullet (Coumans & Bai, 2016–2021). We set a high coefficient of friction and no restitution, so that inter-component contacts neither slide nor bounce. A component then fails by falling when it is not supported. We consider an assembly stable if no point of any component moves more than three quarters the height of the stud of a brick, 3 LDraw Units (LDU). Effects such as the holding force of stud connections and the play of axles are not modeled, so an assembly that balances as a rigid body passes even when a built one might sag or fall apart. Evaluating this would require a structural simulation, with either a force limit at each connection or a finite-element model of the parts, which we discuss in Sec. 6. Two parts collide when their geometry overlaps by more than a tolerance. An assembly with collisions cannot be built, and is determined invalid prior to simulation.

![](images/3e60b29264179c9eff34e4a03ad7ddb683524b685dce121824d78b19a20d723d.jpg)  
(a)

<table><tr><td colspan="3">Method PE↑ SigLIP 2↑ VQAScore ↑</td></tr><tr><td>BrickGPT 0.157</td><td>0.052</td><td>0.050</td></tr><tr><td>BrickNet-0.6B 0.279</td><td>0.603</td><td>0.557</td></tr><tr><td>BrickNet-1.7B 0.282</td><td>0.631</td><td>0.593</td></tr><tr><td>BrickNet-4B 0.283</td><td>0.639</td><td>0.615</td></tr><tr><td>BrickNet-8B 0.284</td><td>0.647</td><td>0.608</td></tr><tr><td>BrickNet-14B 0.283</td><td>0.625</td><td>0.602</td></tr><tr><td>GT 0.315</td><td>0.826</td><td>0.748</td></tr><tr><td>GPT-5.6 Luna 0.313</td><td>0.818</td><td>0.732</td></tr></table>

(b)  
Figure 2: BrickNet validation. We evaluate GPT-5.6 Luna on the BrickNet text-to-assembly validation set (Kulits & Schmid, 2026). While it was not specialized for this dataset, we find it exceeds task-specific baselines including BrickGPT (Pun et al., 2025) across metrics and nears the alignment of the ground-truth objects. (a) Sample inputs and model outputs. (b) Caption alignment following BrickNet’s evaluation protocol. PE (Bolya et al., 2025) and SigLIP 2 (Tschannen et al., 2025) score the similarity between the embeddings of the renders and of the caption, and VQAScore (Lin et al., 2024) the probability that a VQA model answers yes when asked whether the renders show the caption. See Kulits & Schmid (2026) for details on metric computation.

## 3.3. Text-Conditioned Assembly

In text-conditioned assembly, a system receives a text prompt and is tasked with producing an assembly that depicts it. The system must select part types from the library and assign each instance a color, position, and 3D rotation.

Prior work BrickNet trains a language model for this task on human-designed assemblies collected from online. Its captions are generated by a vision–language model over renders of existing assemblies. The validation set contains 512 such held-out captions from single objects of at most 100 parts each.

We evaluate GPT-5.6 Luna, a general coding agent, on the BrickNet validation prompts (Fig. 2). The agent is given the environment described in Sec. 4.3. To match BrickNet, we limit the agent to 100 parts. An output is scored by the alignment between its renders and the prompt, measured with PE (Bolya et al., 2025), SigLIP 2 (Tschannen et al., 2025), and VQAScore (Lin et al., 2024).

Without having been trained on the dataset, the agent substantially outperforms task-specific baselines BrickNet and BrickGPT (Pun et al., 2025), scoring within 0.02 of the held-out (GT) assemblies on each metric and effectively saturating the evaluation. This motivates our design of a new benchmark.

## 4. BrickBench

BrickBench is a benchmark of agentic text-conditioned LEGO-set design. Given a text prompt, an agent must produce an assembly that satisfies semantic and design criteria and that can be physically built. The benchmark consists of 300 prompts in three settings (Sec. 4.1), an environment in which a coding agent can build, and an evaluation protocol for validity, semantic alignment, and design quality (Sec. 4.4). Except for the ablation in Sec. 5.2 and the two datadriven baselines, the agents work inside of the BrickAgent environment described in Sec. 4.3.

## 4.1. Settings

The settings differ in the parts that an agent builds with. In the Model setting, an agent may use at most 400 parts, while the Set setting requires 400–4000 parts be used. Both draw from the full LDraw library as supported by BrickNet, 14,441 part types. In the Alt-Build setting, an agent may use only the 783 pieces, of 104 part types, of the retail set 10698,<sup>2</sup> containing mostly basic bricks and plates with a few windows, doors, wheels, and decorative elements. An assembly that violates these limits fails the validity check.

The settings are intended to evaluate different abilities. Model serves as the base evaluation and tests the creation of assemblies consisting primarily of a single component. Set, whose size is chosen to mirror the requirements of the BrickLink Designer Program,<sup>3</sup> tests whether an agent can build an assembly on the scale of a full-size retail set. Alt-Build evaluates agent ability to design given a scarce, fixed part inventory.

## 4.2. Prompts

Each evaluation setting has 100 prompts, with ten in each of ten topic categories. They follow the BrickLink Designer Program, except that we split the Floral / Nature / Animal category into Floral / Nature and Animal. The categories are Medieval / Castle, Vehicle / Boat / Airplane, Train, Space / Sci-Fi / Fantasy, Art / Object, Building, Floral / Nature, Animal, Pirate, and History / Period. See Figs. 1 and 4 for prompt samples by setting. The prompts were constructed through prompting of an LLM and were manually validated for conformity. Prompts average 31 words in Alt-Build, 45 in Model, and 57 in Set.

We decompose each generation prompt into a question graph in the style of DSG (Cho et al., 2024), which evaluates text-to-image generation by turning a prompt into atomic yes/no questions with an LLM and answering them with a VLM after organizing them by graph dependency. The questions cover the entities named by the prompt, their attributes, and the relations among them. With the graph we can avoid evaluating properties invalidated by other failures, such as checking the color of a knight’s armor when the presence of a knight was not yet established. See Fig. A1 for their distribution by type and Fig. A2 for a full example graph, both in Sec. A.

## 4.3. BrickAgent

We introduce BrickAgent, an environment in which the agents build. It contains the LDraw library, the BrickNet connector system, and tools for authoring, validating, and viewing an assembly programmatically. The tools include functions to search the library for parts, to place parts by their positions or by connectors, to mirror and measure them, to define subassemblies and combine them, and to render an assembly from any viewpoint. The provided validator includes the connectivity, collision, and stability tests of Sec. 3.2 and interpretably reports each failure with the parts involved. This enables an agent to validate the physical requirements of an assembly prior to submission.

The prompt explains the environment with worked examples and states the setting’s part requirements. It informs the agent that experienced LEGO designers will judge the assembly against others by the standards of a published set. In the ablation of Sec. 5.2 the agent operates in the same container, with the harness CLI, LDraw library, and task requirements, but without the tools of BrickAgent. We find that without these tools, frontier coding agents cannot reliably construct valid builds.

## 4.4. Metrics

We score each system per setting, and additionally aggregate across all three (Overall). An assembly is judged as valid if it meets the part requirements of its setting and is both collision-free and stable under the tests of Sec. 3.2. Valid is a proportion and is scored on every prompt.

VQA measures bounded semantic alignment with the prompt, following DSG. We render the assembly from eight views using BrickNet’s renderer, with consistent render settings. We then ask Gemma 4 31B (Gemma Team, 2026) each question of the graph in one call, with instructions to judge on what the views show and to answer yes or no. The score of an assembly is the fraction of its questions satisfied, and VQA is the mean over assemblies.

ELO measures relative semantic alignment and design quality against other systems. For each prompt, we form every pairing of assemblies from different systems and show the judge four views of each. We ask two questions of each pair: 1) The alignment question gives the judge the prompt and asks which assembly matches it better. 2) The design question withholds the prompt and asks which assembly is better designed by the standard of an official set, judged on how the pieces are used rather than the subject. Each pair is asked in both orderings to eliminate position bias. Fig. A3 in Sec. B contains the judge prompts. We fit a Bradley–Terry model (Bradley & Terry, 1952) to the wins from each question and report them as ELO. We anchor to a fixed set of references as noted in Sec. 5. ELO is the average of Align ELO and Design ELO after rescaling to a common spread over the anchors.

VQA and ELO are scored on the assemblies generated by a system, without considering whether they are Valid. Cost is the list price of the tokens consumed by a system for an assembly in US dollars. For the two trained baselines it is the serving cost per assembly, at serverless token rates for the base model of BrickNet and as wall-clock time on a rented GPU instance for BrickGPT.

## 5. Evaluations

We evaluate eleven agents and two data-driven baselines. The agents include GPT-6 Astra, GPT-6.1 Sol, GPT-5.6 Sol, GPT-5.6 Luna, Claude Opus 5.5, Claude Opus 5, Gemini 3.8 Flash, Qwen 3.8 Flash, DeepSeek V4.1 Flash, GLM 5.3 Flash, and Muse Spark 1.3. Opus 5 and Opus 5.5 run in Claude Code, and the others in Codex CLI using the vendors’ respective APIs. Astra, both Opus models, and both Sol models are evaluated at high reasoning to bound cost while

Table 1: BrickBench. We report metrics for each of the three evaluation settings and in the aggregate (Overall). Valid requires that the assembly satisfies the part requirement of the setting and is both collision-free and stable. VQA is the proportion of VQA questions satisfied (Sec. 4.4). ELO, Align ELO, and Design ELO are ratings estimated from pairwise VLM judgments (Sec. 4.4). Cost is the API cost in US dollars, or its equivalent for BrickGPT (Pun et al., 2025), of producing one assembly. \*The Valid metric of BrickNet-14B (Kulits & Schmid, 2026) doesn’t include stability as the model doesn’t produce assemblies in a canonical frame and the direction of gravity is undefined.
<table><tr><td>System</td><td>Valid ↑</td><td>VQA↑</td><td>ELO↑</td><td>Align ELO ↑</td><td>Design ELO ↑</td><td>Cost ($) ↓</td></tr><tr><td>Overall</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-6 Astra</td><td>1.00</td><td>0.954</td><td>1297 ±22</td><td>1295 ±27</td><td>1299 ±27</td><td>4.06</td></tr><tr><td>S GPT-6.1 Sol</td><td>1.00</td><td>0.945</td><td>1293 ±23</td><td>1301 ±27</td><td>1286 ±26</td><td>0.87</td></tr><tr><td>A Claude Opus 5.5</td><td>0.99</td><td>0.923</td><td>1249 ±23</td><td>1212 ±26</td><td>1287 ±27</td><td>6.33</td></tr><tr><td>A Claude Opus 5</td><td>1.00</td><td>0.908</td><td>1101 ±17</td><td>1095 ±20</td><td>1108 ±20</td><td>16.47</td></tr><tr><td>女 Qwen 3.8 Flash</td><td>0.90</td><td>0.852</td><td>1048 ±17</td><td>1040 ±21</td><td>1057 ±20</td><td>1.75</td></tr><tr><td>S GPT-5.6 Sol</td><td>1.00</td><td>0.859</td><td>1015 ±16</td><td>1041 ±19</td><td>988 ±19</td><td>1.02</td></tr><tr><td>Gemini 3.8 Flash</td><td>0.98</td><td>0.856</td><td>1014 ±17</td><td>1005 ±21</td><td>1024 ±20</td><td>3.70</td></tr><tr><td>Q DeepSeek V4.1 Flash</td><td>0.94</td><td>0.813</td><td>1006 ±17</td><td>998 ±20</td><td>1015 ±20</td><td>0.71</td></tr><tr><td>GPT-5.6 Luna</td><td>1.00</td><td>0.792</td><td>898 ±16</td><td>917 ±19</td><td>879 ±20</td><td>0.75</td></tr><tr><td>∞Muse Spark 1.3</td><td>0.80</td><td>0.718</td><td>868 ±20</td><td>859 ±24</td><td>878 ±23</td><td>6.68</td></tr><tr><td>ZGLM 5.3 Flash</td><td>0.92</td><td>0.590</td><td>752 ±23</td><td>750 ±25</td><td>753 ±29</td><td>0.54</td></tr><tr><td>Model</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-6 Astra</td><td>1.00</td><td>0.966</td><td>1266 ±38</td><td>1277 ±45</td><td>1254 ±45</td><td>4.08</td></tr><tr><td>S GPT-6.1 Sol</td><td>1.00</td><td>0.969</td><td>1269 ±36</td><td>1272 ±45</td><td>1265 ±42</td><td>0.92</td></tr><tr><td>AI Claude Opus 5.5</td><td>1.00</td><td>0.952</td><td>1236 ±40</td><td>1216 ±46</td><td>1252 ±42</td><td>5.90</td></tr><tr><td>AI Claude Opus 5</td><td>1.00</td><td>0.917</td><td>1083 ±30</td><td>1075 ±36</td><td>1089 ±31</td><td>16.72</td></tr><tr><td>女 Qwen 3.8 Flash</td><td>0.95</td><td>0.921</td><td>1012 ±27</td><td>1030 ±30</td><td>996 ±32</td><td>1.59</td></tr><tr><td>S GPT-5.6 Sol</td><td>1.00</td><td>0.908</td><td>1032 ±28</td><td>1051 ±34</td><td>1015 ±31</td><td>1.14</td></tr><tr><td>Gemini 3.8 Flash 1</td><td>0.98</td><td>0.894</td><td>1012 ±34</td><td>995 ±40</td><td>1028 ±37</td><td>4.05</td></tr><tr><td>Q DeepSeek V4.1 Flash</td><td>0.95</td><td>0.857</td><td>1007 ±30</td><td>1013 ±33</td><td>1001 ±33</td><td>0.65</td></tr><tr><td>GPT-5.6 Luna</td><td>1.00</td><td>0.821</td><td>892 ±30</td><td>897 ±34</td><td>889 ±37</td><td>0.99</td></tr><tr><td>∞Muse Spark 1.3</td><td>0.86</td><td>0.811</td><td>848 ±31</td><td>845 ±38</td><td>851 ±36</td><td>5.99</td></tr><tr><td>Z GLM 5.3 Flash</td><td>0.90</td><td>0.658</td><td>848 ±37</td><td>816 ±42</td><td>878 ±43</td><td>0.69</td></tr><tr><td>BrickNet-14B</td><td>0.26</td><td>0.180</td><td>731 ±53</td><td>485 ±62</td><td>952 ±50</td><td>0.0006</td></tr><tr><td>BrickGPT</td><td>1.00</td><td>0.024</td><td>338 ±45</td><td>129 ±64</td><td>528 ±60</td><td>0.04</td></tr><tr><td>Set</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S GPT-6 Astra</td><td>1.00</td><td>0.933</td><td>1305 ±35</td><td>1291 ±42</td><td>1316 ±46</td><td>5.15</td></tr><tr><td>S GPT-6.1 Sol</td><td>1.00</td><td>0.920</td><td>1295 ±41</td><td>1293 ±47</td><td>1292 ±45</td><td>1.05</td></tr><tr><td>A Claude Opus 5.5</td><td>0.99</td><td>0.892</td><td>1240 ±44</td><td>1174 ±45</td><td>1315 ±53</td><td>8.08</td></tr><tr><td>AI Claude Opus 5</td><td>1.00</td><td>0.890</td><td>1100 ±27</td><td>1092 ±31</td><td>1109 ±34</td><td>18.80</td></tr><tr><td>女 Qwen 3.8 Flash</td><td>0.79</td><td>0.801</td><td>1090 ±38</td><td>1054 ±44</td><td>1131 ±41</td><td>2.26</td></tr><tr><td>S GPT-5.6 Sol</td><td>1.00</td><td>0.832</td><td>993 ±27</td><td>1019 ±32</td><td>963 ±32</td><td>1.16</td></tr><tr><td>Gemini 3.8 Flash </td><td>0.97</td><td>0.806</td><td>1012 ±31</td><td>1010 ±36</td><td>1013 ±36</td><td>4.26</td></tr><tr><td>DeepSeek V4.1 Flash Q</td><td>0.87</td><td>0.786</td><td>1005 ±35</td><td>979 ±39</td><td>1036 ±41</td><td>0.87</td></tr><tr><td>GPT-5.6 Luna S</td><td>1.00</td><td>0.758</td><td>929 ±26</td><td>952 ±31</td><td>902 ±35</td><td>0.85</td></tr><tr><td>∞ Muse Spark 1.3</td><td>0.68</td><td>0.661</td><td>871 ±38</td><td>864 ±46</td><td>882 ±42</td><td>8.54</td></tr><tr><td>Z GLM 5.3 Flash Alt-Build</td><td>0.89</td><td>0.523</td><td>695 ±33</td><td>738 ±41</td><td>649 ±45</td><td>0.61</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-6 Astra S GPT-6.1 Sol</td><td>1.00</td><td>0.961</td><td>1325 ±37</td><td>1314 ±51</td><td>1336 ±44</td><td>2.95</td></tr><tr><td>A Claude Opus 5.5</td><td>1.00</td><td>0.946</td><td>1326 ±35</td><td>1339 ±43</td><td>1312 ±46</td><td>0.64</td></tr><tr><td>Claude Opus 5 AI</td><td>0.99</td><td>0.925</td><td>1277 ±37</td><td>1248 ±41</td><td>1307 ±45</td><td>5.01</td></tr><tr><td></td><td>1.00</td><td>0.917</td><td>1126 ±31</td><td>1120 ±36</td><td>1132 ±36</td><td>13.90</td></tr><tr><td>女 Qwen 3.8 Flash</td><td>0.97</td><td>0.827</td><td>1051 ±27</td><td>1038 ±35</td><td>1064 ±36</td><td>1.39</td></tr><tr><td>S GPT-5.6 Sol</td><td>1.00</td><td>0.836</td><td>1021 ±31</td><td>1054 ±36</td><td>986 ±33</td><td>0.76</td></tr><tr><td>1 Gemini 3.8 Flash</td><td>1.00</td><td>0.867</td><td>1023 ±25</td><td>1010 ±31</td><td>1036 ±31</td><td>2.79</td></tr><tr><td>Q DeepSeek V4.1 Flash</td><td>0.99</td><td>0.797</td><td>1007 ±29</td><td>1002 ±33</td><td>1014 ±34</td><td>0.60</td></tr><tr><td>GPT-5.6 Luna S</td><td>1.00</td><td>0.797</td><td>875 ±28</td><td>903 ±39</td><td>845 ±37</td><td>0.40</td></tr><tr><td>∞Muse Spark 1.3</td><td>0.86</td><td>0.678</td><td>887 ±30</td><td>871 ±36</td><td>904 ±37</td><td>5.52</td></tr><tr><td>Z GLM 5.3 Flash</td><td>0.97</td><td>0.591</td><td>686 ±36</td><td>688 ±47</td><td>684 ±47</td><td>0.31</td></tr></table>

![](images/1184802c891d15364df09341f2acfa6fbe9073e76ee662c6a7bb1cb7cd5885ae.jpg)  
Figure 3: BrickBench summary. We report summary metrics averaged across the three evaluation settings. See Tab. 1 for disaggregated results. The ELO-plot lines represent 95% intervals.

Luna runs at max and the others at xhigh. Each agent is given a budget of 300 turns per prompt. The baselines evaluated are BrickNet-14B (Kulits & Schmid, 2026) and BrickGPT (Pun et al., 2025), both restricted to the Model setting which is the only one applicable. Nine of the agents form the reference set of the ELO scale (Sec. 4.4). The baselines and ablated runs of Sec. 5.2 are rated against them and do not affect the scale. Opus 5.5 and GPT-6.1 Sol were released after the human studies of Secs. 5.3 and 5.4 were run, so they are scored on the automated metrics only. Like the baselines, they are rated against the reference set, which leaves every other rating unchanged, and both fall within the range of ratings on which the judge is validated.

## 5.1. Results

We report each metric, per setting and overall, in Tab. 1, and plot the overall columns in Fig. 3. We observe that validity is nearly solved when provided the environment. Five agents deliver a valid assembly for each prompt and none fall below a Valid proportion of 0.80 overall. Of the 137 invalid assemblies across the reference set, 44 are prompts with no delivered assembly. The others fail on stability (41), collisions (31), and the part requirement (21).

VQA separates the reference set gently, from 0.59 for GLM to 0.95 for Astra. Design ELO does so more sharply, with 545 ELO points between GLM and Astra, a gap that can be interpreted as a judge preferring Astra’s assembly in around 96% of pairs. Within the reference set, Astra leads on every metric, and Opus 5 is ranked second while incurring four times the cost. Sol, Gemini, Qwen, and DeepSeek form a middle within 41 ELO points of one another. Of the two later agents, GPT-6.1 Sol matches Astra on ELO (1293 against 1297, intervals overlapping) at a fifth of its cost, and Opus 5.5 rates 148 points above Opus 5 at 40% of its cost, inside Astra’s interval on Design ELO but 83 points lower on Align ELO. Alignment varies more with the kind of question than with the subject (Fig. A7 in Sec. C). No prompt category is consistently harder than the others, and the two top agents score at least 0.90 in every one. Every agent satisfies at least 91% of the questions about color, but texture is satisfied 49–83% of the time, and object type and state are the next hardest. The stronger agents also build larger and more varied assemblies (Tab. A1 in Sec. C). In Model, Astra averages 217 parts of 35 types in 9 colors, and GLM 27 parts of 10 types in 5 colors. In Set, Qwen, DeepSeek, Muse, and GLM average 23 to 40 connected components per assembly, against 2.2 to 4.2 for Astra and both Sol and Opus models, so their scenes consist of far more disconnected pieces. Figs. A4 and A5 in Sec. C plot Tabs. 1 and A1 by setting.

The reference set spans a factor of 30 in cost, and rank does not follow price. Muse, for instance, costs nine times what DeepSeek costs and rates 138 points lower. See Fig. 5a for a plot of ELO against Cost. While Align ELO and Design ELO agree for most agents, Sol scores 53 points higher on alignment than design. Both trained baselines fall drastically below every agent.

![](images/c0c8506d4471770ef5b39d92831a54838fff75ef9a2bd2d316e153362b262535.jpg)  
Figure 4: BrickBench samples. We show one assembly from each of the eleven agents for three prompts in each setting. We observe the agents vary in their interpretation of the prompts. One assembly is missing for Muse because it did not deliver one within the turn budget.

![](images/d315126bff1b33029a6cdfa4edd19f43c4f6c6f308a7c6ff4ed8fcb790d96048.jpg)  
(a) ELO vs. Cost

![](images/e98a00b44658f67276323b4841451850d9e566a39b0fadd43056837d4d6b5eab.jpg)  
(b) Validation of Design ELO  
Figure 5: ELO. (a) We plot ELO against mean assembly cost. (b) To validate our model-based Design ELO computation, we perform a perceptual study where participants are tasked with judging design quality between a pair of assemblies for a given prompt. The prompt isn’t given to the judge, and they make their decision based only on the images. We find that the rankings agree on 34/36 agent pairs (Kendall � = 0.89), suggesting the metric is an effective proxy for human judgment. The study covers the nine reference agents; Opus 5.5 and GPT-6.1 Sol were evaluated after it.

BrickNet-14B is collision-free on only 26% of prompts, while BrickGPT, though valid on each prompt, satisfies 2% of the questions. See Fig. 4 for sample visuals, and Fig. A6 in Sec. C for ten more prompts. We observe that the settings differ in what breaks. In terms of validity, Set is the hardest, with validity dropping to 0.68 for Muse, 0.79 for Qwen, and 0.87 for DeepSeek. Alignment falls for every agent, and part counts crowd the 400-part floor (Fig. A8 in Sec. C), so many agents build the smallest scene allowed by the setting. Alt-Build costs the least and maintains the highest validity. Alignment is highest in Model, the only setting in which the baselines apply.

## 5.2. Environment Ablation

We ablate the BrickAgent environment in Tab. 2, with the same agent CLI (Codex CLI), container, and task requirements, but with only the LDraw library and a primer on the LDraw file format in place of its tools. Astra remains valid on 40% of prompts while Luna produces only one valid build in 300. Their assemblies average 7 and 156 colliding pairs, respectively, and fall apart into 6 and 83 connected components against 2 and 5 with the environment.

Alignment and design do not fail, however. Astra’s VQA drops by 0.01 and its ELO moves within the interval. Both of Luna’s metrics rise, by 0.02 in VQA and by 62 ELO points. Without the restriction of buildability, Luna makes assemblies that depict the prompt as well and look better, but that cannot be built, demonstrating the importance of relevant verifiers. Doing so costs Astra nothing on alignment or design, but costs Luna a little of both.

Table 2: Environment ablation. We ablate the effect of the BrickAgent environment, pooled over the three evaluation settings (Overall; 300 prompts). While both agents consistently produce valid assemblies when given tools, they struggle without them to varying degrees: GPT-6 Astra is somewhat robust but its assemblies drop to being valid only 40% of the time while GPT-5.6 Luna’s fall to less than a percent. The environment improves the VQA of Astra but lowers its ELO within noise. Without it, Luna improves on both, suggesting the buildability constraint limits expressivity.
<table><tr><td>System</td><td>Valid ↑</td><td>Collisions ↓</td><td>Conn. Components</td><td>VQA↑</td><td>ELO↑</td><td>Cost ($)</td></tr><tr><td>GPT-6 Astra</td><td>1.00</td><td>0.00</td><td>2.21</td><td>0.954</td><td>1297 ±22</td><td>4.06</td></tr><tr><td>without BrickAgent</td><td>0.40</td><td>7.19</td><td>6.12</td><td>0.940</td><td>1309 ±24</td><td>4.66</td></tr><tr><td>GPT-5.6 Luna</td><td>1.00</td><td>0.00</td><td>4.60</td><td>0.792</td><td>898 ±16</td><td>0.75</td></tr><tr><td>without BrickAgent</td><td>&lt;0.01</td><td>155.51</td><td>83.00</td><td>0.813</td><td>960 ±20</td><td>0.60</td></tr></table>

![](images/389b400c960d56cf4d244921c982d8c0356123e256c99180a253f559b30290a3.jpg)  
(a)  
(b)  
Figure 6: Human-design evaluation. We perform a perceptual study comparing agent designs with human designs. For each assembly in the Model and Set settings, we select a random assembly from the BrickNet dataset with a comparable number of parts and ask a human rater to identify the one made by a person. We find that raters are able to distinguish between the agent-designed and human-designed assemblies (a); the study covers the nine reference agents and the two baselines. While agent-designed assemblies often satisfy the prompt criteria, there is a clear gap in fine-grained design to those produced by humans (b).

## 5.3. Validation of the Design Judge

To confirm the validity of Design ELO, which rests on a VLM judge, we evaluate against human ratings. We recruit participants on Prolific and show each 40 pairs of same-prompt assemblies, with four views per assembly. We collect 4,080 judgments from 102 raters. We fit the human judgments with the same Bradley–Terry model as ELO (Sec. 4.4), and compare the ratings against Design ELO in Fig. 5b. The rankings agree on 34 of the 36 pairs among the agents, with a Kendall � of 0.89. This follows findings from prior work which demonstrated alignment between VLM and human preferences on text-to-3D generation (Wu et al., 2024). The judge is more decisive than the rater pool, so we align the two by a constant factor. Opus 5.5 and GPT-6.1 Sol were released after this study and are not part of it.

## 5.4. Comparison with Human Designs

We asked people familiar with LEGO design to tell agent builds from human ones (Fig. 6). For assemblies in the Model and Set settings, we drew part-count-matched human-designed models from the BrickNet dataset, and asked which of the two a person made. The drawn samples are not prompt-matched. No human designs exist for the BrickBench prompts, so the pairs are matched on part count alone, and Fig. 6b shows human models of other subjects. A prompt-matched comparison would require commissioning designs, which we leave to future work. As in Sec. 5.3, the study predates Opus 5.5 and GPT-6.1 Sol. Five raters provided 360 ratings altogether, and selected the human model in 323 of them. No agent was taken for a human designer more than one time in five (Fig. 6a), and every agent’s rate falls significantly below chance. GPT-6 Astra, the agent most often taken for a person, was chosen in 4 of its 21 pairs. We show reference examples of human-designed models in Fig. 6b. While agent assemblies satisfy the prompt well, they lack the part usage, proportion, and surface detail that mark human design.

## 6. Discussion and Limitations

We report four findings from our evaluations. First, we find that data-driven models built for the task are subsumed by general agents. A general coding agent exceeds models on their own benchmark (Sec. 3.3) and on ours. Cost, however, remains an axis for future work. BrickNet-14B produces an assembly for six hundredths of a cent, while the Astra does so in \$4.06. Second, prompt adherence saturates. The best-performing agent satisfies 95% of the VQA questions asked of it, and absolute alignment does not strongly separate the top performers. Headroom lies instead in design quality. Third, agent performance is dependent on the environment. The same model produces consistently buildable assemblies with BrickAgent, and unbuildable ones without it (Tab. 2). Fourth, a gap remains to humans. Participants familiar with LEGO design filtered agent-designed assemblies from human ones nine out of ten times (Fig. 6). We interpret this as a measure of design quality: agents satisfy requirements but fall short of the standards of a designed set.

Our simulator of physical validity is a proxy and not a complete measure of structural stability. Connections are not tested for strength, so an assembly that balances passes even when a physical one might sag or come apart (Sec. 3.2). Some works exist that attempt to build force models over stud connections Waßmann & Weicker (2012) to evaluate FEM (Pletz & Drvoderic, 2023). None, however, is general, and integrating such would severely limit the available part catalog. Until a general one exists, our bar for physical evaluation is necessary but not sufficient.

## 7. Conclusion

We introduced BrickBench, a benchmark for agentic text-conditioned LEGO-set design. The domain gives the task a floor that a simulator can verify and an open ceiling on design quality. It includes 300 prompts in three settings with different constraints, a reference environment in which agents build, BrickAgent, and a set of metrics that score build validity, alignment, and design confirmed by human raters. Nine frontier agents build valid assemblies for most prompts and satisfy the majority of what each requests, displacing task-specific data-driven models at an increase of 900–6,800 times the cost. While doing so, we highlight the open challenge of narrowing the gap to human design.

## References

Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. Perception encoder: The best visual embeddings are not at the output of the network. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In International Conference on Learning Representations (ICLR), 2024.

Hyunsoo Chung, Jungtaek Kim, Boris Knyazev, Jinhwi Lee, Graham W. Taylor, Jaesik Park, and Minsu Cho. Brick-by-Brick: Combinatorial construction with deep reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Erwin Coumans and Yunfei Bai. PyBullet, a Python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016–2021.

Jiahao Ge, Mingjun Zhou, Hanyou Zheng, Hao Xu, and Chi-Wing Fu. LEGO-maker: Autoregressive image-conditioned LEGO model creation. ACM Transactions on Graphics (TOG), 44(6), 2025.

Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026.

Yunqi Gu, Ian Huang, Jihyeon Je, Guandao Yang, and Leonidas Guibas. BlenderGym: Benchmarking foundational model systems for graphics editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A. Ross, Cordelia Schmid, and Alireza Fathi. SceneCraft: An LLM agent for synthesizing 3D scenes as Blender code. In International Conference on Machine Learning (ICML), 2024.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations (ICLR), 2024.

Peter Kulits and Cordelia Schmid. BrickNet: Graph-backed generative brick assembly. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 39252–39261, June 2026.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision (ECCV), 2024.

Liang Ma, Jiajun Wen, Min Lin, Rongtao Xu, Xiwen Liang, Bingqian Lin, Jun Ma, Yongxin Wang, Ziming Wei, Haokun Lin, Mingfei Han, Meng Cao, Bokui Chen, Ivan Laptev, and Xiaodan Liang. PhyBlock: A progressive benchmark for physical understanding and planning via 3D block assembly. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025.

Dimitrios Mallis, Ahmet Serdar Karadeniz, Sebastian Cavada, Danila Rukhovich, Niki Foteinopoulou, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. CAD-assistant: Tool-augmented VLLMs as generic CAD task solvers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Martin Pletz and Matthias Drvoderic. BrickFEM: An automated finite element model for static and dynamic simulations of simple Lego sets. engrXiv preprint, 2023. doi: 10.31224/2898.

Ava Pun, Kangle Deng, Ruixuan Liu, Deva Ramanan, Changliu Liu, and Jun-Yan Zhu. Generating physically stable and buildable brick structures from text. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14798–14809, October 2025.

Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kai Chen, and Kaifeng Lyu. LEGO-puzzles: How good are MLLMs at multi-step spatial reasoning? In European Conference on Computer Vision (ECCV), 2026.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Aaron Walsman, Muru Zhang, Klemen Kotar, Karthik Desingh, Ali Farhadi, and Dieter Fox. Break and make: Interactive structural understanding using LEGO bricks. In European Conference on Computer Vision (ECCV), 2022.

Martin Waßmann and Karsten Weicker. Maximum flow networks for stability analysis of LEGO structures. In European Symposium on Algorithms (ESA), 2012.

Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. GPT-4V(ision) is a human-aligned evaluator for text-to-3D generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Tian Xia, Tianrun Gao, Wenhao Deng, Long Wei, Xiaowei Qian, Chenglei Yu, and Tailin Wu. BuildArena: A physics-aligned interactive benchmark of LLMs for engineering construction. In International Conference on Machine Learning (ICML), 2026.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Jing Hua Toh, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Haozhe Zhang, Kaichen Liu, Miaomiao Chen, Lei Li, Shaojie Yang, Cheng Peng, and Hanjie Chen. BenchCAD: A comprehensive, industry-standard benchmark for programmatic CAD. arXiv preprint arXiv:2605.10865, 2026.

# BRICKBENCH: EVALUATING AGENTIC BRICK DESIGN

## Appendix

## A. Prompts and Question Graphs

Fig. A1 shows the share of each setting’s questions by kind and subtype. Set graphs are the largest, at 24 questions per prompt on average, because their prompts often describe scenes with multiple objects.

![](images/c7e6017272801bded88519f67fec486d915ac64641ba67875d3b04cd45c230ce.jpg)  
Figure A1: VQA question types. We plot the share of each setting’s questions by DSG type.

Fig. A2 shows one Set graph in full, with the prompt it was generated from. Its roots are the entities the prompt names, and every attribute and relation question hangs from the questions that establish what it refers to, so a missing entity fails every subsequent question about it.

## B. Judge Prompts

Fig. A3 reproduces the three prompts of Sec. 4.4 as the judge receives them. The two pairwise prompts differ in what the judge is told. The alignment prompt contains the prompt and asks which assembly covers more of it, whatever its build quality. The design prompt withholds the prompt and asks which assembly is better designed, whatever it depicts. The VQA prompt lists every question of the graph, tells the judge to answer from the views alone, and requires a yes or a no on each.

Set A brown dog with floppy black ears and a small brown hat sits at a table in a burning room. A white cofee mug rests in front of it. Flames climb the curtains and surround the chair, but the dog sits upright with its paws resting calmly on the tabletop. The front of the room is open to reveal the scene.

![](images/042263a6f87814e129024273d15a54aacefaec7e2fe440bd0e7ad0467705d876.jpg)  
Figure A2: Question-graph sample. We plot a sample question graph for a set prompt. An arrow from one question to another indicates the second is satisfied only if the judge answers yes to the first.

## C. Additional Results

Tab. A1 reports build statistics per setting. Astra uses the most parts in every setting and Qwen the second most. In Set, Qwen, DeepSeek, Muse, and GLM build scenes of 20 to 40 disconnected components on average, while Astra, Opus, Gemini, and Sol keep theirs under five.

Figs. A4 and A5 plot Tabs. 1 and A1 by setting in the layout of Fig. 3, and Fig. A6 shows one assembly per agent for further prompts in each setting.

Fig. A7 breaks VQA down by build category and by attribute type.

Fig. A8 plots the distribution of part counts in Model and Set. Most agents peak near the 400-part floor of Set, and Astra is the exception.

Table A1: Further assembly statistics. We report further statistics of the assemblies per setting. Parts is the average number of parts in an assembly. Unique is the (average) number of unique parts. Colors is the number of colors. Collisions is the number of parts that nonphysically collide (overlap) with one another. Stable is the proportion of assemblies that remain static under simulation, with every part moving under 3 LDU. Conn. Components is the number of graph components after BrickNet parsing; a valid assembly may have multiple components in order to represent multiple objects.
<table><tr><td>System</td><td>Parts</td><td>Unique</td><td>Colors</td><td>Collisions ↓</td><td>Stable ↑</td><td>Conn. Components</td></tr><tr><td colspan="7">Model</td></tr><tr><td>GPT-6 Astra</td><td>217</td><td>35.4</td><td>9.4</td><td>0.00</td><td>1.000</td><td>1.6</td></tr><tr><td>S GPT-6.1 Sol</td><td>230</td><td>36.4</td><td>9.1</td><td>0.00</td><td>1.000</td><td>1.2</td></tr><tr><td>AI Claude Opus 5.5</td><td>182</td><td>38.2</td><td>9.1</td><td>0.00</td><td>1.000</td><td>1.3</td></tr><tr><td>Claude Opus 5 AI</td><td>110</td><td>27.7</td><td>7.7</td><td>0.00</td><td>1.000</td><td>1.3</td></tr><tr><td>Qwen 3.8 Flash 女</td><td>125</td><td>23.8</td><td>8.3</td><td>0.10</td><td>0.979</td><td>4.8</td></tr><tr><td>S GPT-5.6 Sol</td><td>83</td><td>20.4</td><td>8.0</td><td>0.00</td><td>1.000</td><td>1.2</td></tr><tr><td>Gemini 3.8 Flash </td><td>95</td><td>25.6</td><td>9.6</td><td>0.00</td><td>0.990</td><td>1.2</td></tr><tr><td>DeepSeek V4.1 Flash Q</td><td>88</td><td>19.3</td><td>7.0</td><td>0.56</td><td>0.969</td><td>3.1</td></tr><tr><td>GPT-5.6 Luna S</td><td>41</td><td>14.8</td><td>6.8</td><td>0.00</td><td>1.000</td><td>2.0</td></tr><tr><td>∞ Muse Spark 1.3</td><td>56</td><td>14.5</td><td>6.7</td><td>0.46</td><td>0.925</td><td>2.0</td></tr><tr><td>Z GLM 5.3 Flash</td><td>27</td><td>10.3</td><td>4.9</td><td>0.86</td><td>0.928</td><td>1.7</td></tr><tr><td>BrickNet-14B</td><td>40</td><td>14.5</td><td>4.8</td><td>24.80</td><td></td><td>1.0</td></tr><tr><td>BrickGPT</td><td>139</td><td>7.1</td><td>1.0</td><td>0.00</td><td>1.000</td><td>2.4</td></tr><tr><td colspan="7">Set</td></tr><tr><td>S GPT-6 Astra</td><td>1511</td><td>46.8</td><td>17.0</td><td>0.00</td><td>1.000</td><td>2.2</td></tr><tr><td>S GPT-6.1 Sol</td><td>1625</td><td>49.1</td><td>16.7</td><td>0.00</td><td>1.000</td><td>3.5</td></tr><tr><td>Claude Opus 5.5 AI</td><td>1112</td><td>68.5</td><td>17.5</td><td>0.00</td><td>1.000</td><td>3.8</td></tr><tr><td>Claude Opus 5 AI</td><td>865</td><td>48.6</td><td>14.6</td><td>0.00</td><td>1.000</td><td>4.2</td></tr><tr><td>Qwen 3.8 Flash 女</td><td>1118</td><td>38.8</td><td>17.4</td><td>0.02</td><td>0.929</td><td>32.8</td></tr><tr><td>GPT-5.6 Sol S</td><td>698</td><td>29.9</td><td>14.8</td><td>0.00</td><td>1.000</td><td>3.3</td></tr><tr><td>Gemini 3.8 Flash </td><td>581</td><td>36.5</td><td>17.3</td><td>0.02</td><td>0.980</td><td>1.7</td></tr><tr><td>DeepSeek V4.1 Flash Q</td><td>726</td><td>33.5</td><td>12.9</td><td>6.94</td><td>0.888</td><td>36.7</td></tr><tr><td>GPT-5.6 Luna</td><td>511</td><td>26.2</td><td>13.2</td><td>0.00</td><td>1.000</td><td>10.2</td></tr><tr><td>∞Muse Spark 1.3</td><td>500</td><td>19.2</td><td>11.0</td><td>9.96</td><td>0.800</td><td>22.6</td></tr><tr><td>Z GLM 5.3 Flash</td><td>568</td><td>16.5</td><td>9.0</td><td>0.80</td><td>0.908</td><td>40.2</td></tr><tr><td colspan="7">Alt-Build</td></tr><tr><td>GPT-6 Astra</td><td>110</td><td>30.0</td><td>15.4</td><td>0.00</td><td>1.000</td><td>2.9</td></tr><tr><td>GPT-6.1 Sol</td><td>110</td><td>30.8</td><td>15.4</td><td>0.00</td><td>1.000</td><td>2.0</td></tr><tr><td>A Claude Opus 5.5</td><td>103</td><td>27.5</td><td>14.2</td><td>0.00</td><td>1.000</td><td>3.3</td></tr><tr><td>A Claude Opus 5</td><td>86</td><td>25.1</td><td>12.9</td><td>0.00</td><td>1.000</td><td>1.8</td></tr><tr><td>女 Qwen 3.8 Flash</td><td>105</td><td>23.3</td><td>16.0</td><td>0.00</td><td>1.000</td><td>5.1</td></tr><tr><td>S GPT-5.6 Sol</td><td>52</td><td>19.3</td><td>12.1</td><td>0.00</td><td>1.000</td><td>1.4</td></tr><tr><td>Gemini 3.8 Flash</td><td>64</td><td>24.7</td><td>14.2</td><td>0.00</td><td>1.000</td><td>1.4</td></tr><tr><td>DeepSeek V4.1 Flash</td><td>73</td><td>18.1</td><td>12.2</td><td>0.00</td><td>1.000</td><td>4.7</td></tr><tr><td>GPT-5.6 Luna S</td><td>40</td><td>17.9</td><td>11.2</td><td>0.00</td><td>1.000</td><td>1.6</td></tr><tr><td>∞ Muse Spark 1.3</td><td>49</td><td>14.8</td><td>11.0</td><td>0.63</td><td>0.945</td><td>2.0</td></tr><tr><td>GLM 5.3 Flash Z</td><td>25</td><td>11.1</td><td>9.4</td><td>0.01</td><td>0.980</td><td>1.5</td></tr></table>

![](images/7d027367e13b8bddf743be767761b8b8742efc701fefcedccc185e4f144a4400.jpg)  
Figure A3: Judge prompts. The three prompts of Sec. 4.4 as the judge receives them. For the two pairwise questions, the four views of each assembly are tiled into one image and the two images precede the text, and the design question never contains the prompt. For VQA, the eight views precede the text as separate images, and the bracketed fields are filled in for each assembly.

![](images/fc432e0f4777990b2db3ebcf50a433b88fe86f19dce029590139cb6f98e270fa.jpg)  
(a)

![](images/d4ccd0ee5455d633b9d608dbfcd5cf7d7970bfbb0918ba5139ec1dba6a66fbf3.jpg)  
(b)  
Figure A4: BrickBench. We visualize disaggregated BrickBench quantitative results. The baselines BrickNet-14B (Kulits & Schmid, 2026) and BrickGPT (Pun et al., 2025) are only evaluated on the Model setting and the Valid metric reported of BrickNet-14B does not include stability.

![](images/1317bd3fdf7db877d3a4b8f7ca7ffa4f5e75fc6076a9b525afb16b3a3d4ccafc.jpg)  
(a)

Model Set Alt-Build  
![](images/e21d1883bf1e9884c4c201309c56841351dec2c63570dbd4162f03520393636b.jpg)  
(b)  
Figure A5: Further assembly statistics. We plot the statistics of Tab. A1 by setting.

![](images/f3b7092f579c751df9038e4039afeb9728597bf91ccc3625a97d19b5924b091c.jpg)  
Figure A6: Additional BrickBench samples. We show further assemblies from the eleven agents in the three settings.

![](images/1efb2150bd8f99158304dbd7a6b2fb6cea2768223e429b113465ad465ba4eb01.jpg)

![](images/a8ad639099f1e1c20d51d3d3b5282fa09a4f519fee99c9de7666101ebb3d2358.jpg)  
(b) By attribute type  
Figure A7: VQA breakdowns. We report score by prompt category (a) and attribute type (b).

![](images/cc8684d20eca56e001f6792173f9c4fffa5f1639ba17714f690e7520e36838a7.jpg)  
(a) Model

![](images/9313ba3fba01dd3b0aaab924af2bb75adf7af2d46420a9d811f77a2abd61b32e.jpg)  
(b) Set  
Figure A8: Part-count distributions. We report the density of part counts in the Model (≤400) and Set (400–4000) settings for each agent. Most agents peak near 400 in Set. GPT-6 Astra uses considerably more parts on average in both settings.