# ARCH-B: Architectural Representation, Comprehension and Hierarchy Benchmark

Kieran Sagar Parikh Harvard University kieran parikh@mde.harvard.edu

Jose Luis Garcia del Castillo y Lopez Northeastern University Harvard University j.garciadelcastillo@northeastern.edu

## Abstract

Multimodal models increasingly interpret visual environments, but their ability to recognize the same building across photographs, floor plans, elevations, sections, and renderings remains poorly characterized. We introduce ARCH-B, a benchmark of354four-choice questions across 11 cross-representational archetypes, constructed from a building-linked corpus of 3.9 million architectural images using visually similar distractors, model-guided difficulty screening, and manual validation. We evaluate 25 multimodal models and collect 5,830 responses from nonexpert human participants. Model accuracy ranges from 10.45% to 83.90%, compared with a human baseline of 35.35%. Models perform comparatively well on mixedrepresentation outlier detection and photograph matching, but remain weaker on floorplan-to-photograph correspondence. Human and model difficulty across archetypes is only weakly correlated (Spearman’s ρ = 0.33). Heldout evaluation confirms that the difficulty identified during screening generalizes beyond the curation models. ARCH-B provides a diagnostic evaluation of visual correspondence and representation transfer across architectural media.

## 1. Introduction

Architectural visual understanding is inherently crossrepresentational. The same building may appear as a photograph, a floor plan, an elevation, a section, a project rendering, a small set of partial views or a combination thereof that must be recognized as belonging to the same object. For humans, these transitions are routine. Architects move constantly between abstract drawings and embodied views of space, and non-architect occupants do something similar whenever they use maps, plans, signage, or partial visual cues to orient themselves in the built environment. For multimodal large language models, however, this family of abilities remains difficult to evaluate systematically; existing multimodal benchmarks probe expert reasoning, core visual perception, and 2D and 3D spatial understanding [5, 12, 16, 18, 19], but do not directly test whether models can preserve building identity across photographs and architectural drawings.

We present ARCH-B, a multimodal benchmark of crossrepresentational architectural comprehension. We define cross-representational and hierarchical architectural comprehension as the selection and alignment of visual evidence for building identity across changes in viewpoint, abstraction, and representational convention. The hierarchy in ARCH-B refers to movement across levels of architectural representation, from local perspectival views of interiors and exteriors to whole-building drawings such as elevations, sections, and floor plans.

ARCH-B consists of 354 multiple-choice question samples, each belonging to one of 11 question archetypes. The 11 question archetypes cover image-drawing correspondence, floor-plan matching, mixed-representation outlier detection, interior-exterior correspondence, and samebuilding recognition across photographs and drawings. Recent architectural benchmarks emphasize floor-plan semantics, drawing literacy, higher-level spatial cognition, or principle-grounded engineering reasoning [3, 6, 9, 15]; ARCH-B instead isolates correspondence and identity across heterogeneous representations of the same building. Our premise is that an architectural benchmark should test not only whether a model can describe a scene, but whether it can align representations that differ in viewpoint, abstraction, and representational conventions.

The images used for the questions were sourced from a large architectural image corpus that we assembled from real-estate listings and architectural project pages. Question samples were generated programmatically from that corpus, using different algorithms for each question archetype. Samples were selected for inclusion in the final benchmark through LLM-in-the-loop difficulty selection against four strong multimodal models, followed by manual validation and balancing.

We used the benchmark to evaluate 25 multimodal models, with Gemini 3.1 Pro Preview achieving a top score of 83.90%. The leaderboard spans a wide range of performance, and the strongest proprietary models substantially outperform both chance and the non-expert human baseline. We also conducted a human study through a human-subjects crowdsourcing platform, collecting 5,830 valid responses across the 354 benchmark questions. The response-weighted human baseline was 35.35%: above random chance, but far below the best evaluated models.

The main contributions of this paper are threefold. First, we introduce and publicly release ARCH-B, a 354-question benchmark for evaluating cross-representational architectural comprehension across photographs, floor plans, elevations, sections, and renderings. Second, we describe a scalable generation and curation pipeline that combines a large architectural image corpus, embedding-based candidate construction, LLM-in-the-loop difficulty selection, and manual validation by researchers. Third, we provide a broad empirical evaluation of 25 multimodal models alongside a non-expert human baseline, showing that current frontier models can perform strongly on these tasks while substantial variation remains across archetypes and models.

## 2. Related Work

## 2.1. Multimodal visual and spatial reasoning benchmarks

Visual question answering benchmarks have progressed from object recognition and language-conditioned prediction toward compositional and expert-level reasoning. GQA uses scene graphs and functional programs to test structured reasoning over real images [7], while MMMU evaluates models using diagrams, charts, maps, and other expert materials across multiple disciplines [19]. Because such broad evaluations can conflate perception, knowledge, and reasoning, vision-centric benchmarks isolate perceptual capabilities more directly. MMVP exposes visual distinctions poorly represented by common vision–language encoders [17]; BLINK reformulates classic computer-vision tasks, including visual correspondence, relative depth, and multiview reasoning, as multiple-choice evaluations [5]; and CV-Bench tests 2D relations and counting alongside 3D depth and distance judgments [16].

Recent benchmarks extend spatial evaluation across views and over time. VSI-Bench tests whether models can recover and query a representation of an indoor environment from video [18], while 3DSRBench evaluates objectcentered 3D relations and robustness to unusual viewpoints [12]. ARCH-B addresses a complementary problem: the identity that must persist is an entire building, and observations vary in both viewpoint and representational convention. Matching photographs, floor plans, elevations, sections, and renderings requires transferring evidence between projective images and abstract drawings rather than reasoning within a single visual formalism.

## 2.2. Architectural imagery, drawings, and multimodal evaluation

Architectural vision datasets have largely supported specialized prediction tasks. CubiCasa5K, CubiGraph5K, and FloorPlanCAD provide raster, graph, and vector floorplan representations for parsing and symbol recognition [4, 8, 11]. ZInD aligns panoramas, room layouts, and 2D and 3D floor plans to support layout estimation and multiview registration [2], while WAFFLE collects nearly 20,000 in-the-wild floor plans and associated metadata across diverse building types [6]. These resources advance parsing, layout recovery, and geometric registration, but do not evaluate identity across heterogeneous architectural representations.

Recent benchmarks also evaluate general-purpose multimodal models on architecture and engineering. AECV-Bench tests counting, OCR, and drawing-grounded question answering on floor plans and engineering drawings [9]. ArchSIBench evaluates architectural spatial intelligence through tasks involving perception, navigation, transformation, configuration, circulation, and functional zoning [15]. MMArch instead requires models to combine technical figures with architecture and civil-engineering knowledge [3]. These benchmarks emphasize drawing literacy, broader spatial cognition, or professional knowledge; ARCH-B focuses specifically on recognizing multiple visual representations as evidence of the same built object.

## 2.3. Benchmark construction and difficulty control

Automatically generated benchmarks provide scale but risk artifacts, trivial distractors, and misaligned difficulty. GQA uses programmatic generation from scene graphs [7]; adversarial filtering and AFLite use model behavior to reduce exploitable biases [10]; and MMMU-Pro removes questions answerable without images and expands answer sets to reduce shortcuts [20]. ARCH-B combines structured generation with visually similar distractors and model-guided difficulty filtering for building-level correspondence. Manual validation and held-out model evaluation further address item validity and test whether the selected difficulty generalizes.

## 3. Methods

We constructed ARCH-B by assembling an architectural image corpus, generating candidate questions, filtering them with four multimodal models, and manually validating the selected questions.

![](images/e586c184fc3d578736dcec4cd300fa15ca3d2c11188f8ee783e8b56cad34173d.jpg)  
Figure 1. Example question for archetype 2. The correct answer is outlined in green.

## 3.1. Corpus of Architectural Imagery

We built the source corpus by scraping architectural imagery from two source families: real-estate listings and architectural project pages. Real-estate listings were sourced from Zillow and Realtor.com, while architectural projects were sourced from ArchDaily and Deezeen. We scraped daily from November 2024 until October 2025, producing a corpus of 3.9 million images, including 58,000 floor plans, drawn from 95,000 real-estate listings and 41,000 architectural projects. Crucially, we chose to preserve relationships among images and their parent listing or project, rather than flattening the corpus into isolated files, enabling us to query images by listing or project during programmatic question sample generation. We refer to this grouping of images and associated metadata as building record.

We augmented this corpus with vector embeddings that would be needed for programmatic question sample generation, detailed in Sec. 3.2. We encoded each image using OpenCLIP (an open-source implementation of CLIP that learns a shared representation space for images and natural-language text) using the ViT-B/32 architecture and the laion2b s34b b79k pretrained checkpoint, producing a 512-dimensional embedding for each image [1, 13, 14]. These embeddings can be used to identify visually related images using the cosine similarity between two images’ embeddings, as well as to perform zero shot image classification by embedding a label string using the same OpenCLIP encoder and measuring the cosine similarity of the image embedding and the label string embedding.

## 3.2. Question Sample Generation

Every benchmark question sample follows the same format: a short natural-language instruction, optionally one or more reference images in the question stem, and four answer choices with exactly one correct answer. Answer choices do not include descriptive text beyond an answer ID. All benchmark questions use a four-alternative forcedchoice format with exactly one correct answer, so chance performance is 25%.

Each question sample belongs to one of 11 question archetype. The question archetypes were designed by the authors to sample all possible cross-representational and outlier permutations of representations and hierarchy types. The archetype to which a question belongs defines its natural-language instruction and the types of images that appear as reference images and answer choices. For example, question samples belonging to archetype 1 will all have the natural-language instruction “Choose the photograph that corresponds to the building shown in this/these floorplan/s,” and will have floorplans as reference images and photographs as answer choices (Fig. 1). A complete list of archetype metadata, including a representative question sample for each archetype, is provided in Sec. A.

Given that each question archetype can be defined by a set of rules, we designed generation algorithms to generate question samples for each archetype. This approach follows the broader use of structured annotations and rule-based generation to construct diagnostic visual-reasoning questions [7]. Each archetype has its own generation algorithm, but all generation algorithms share the same shape. Generation begins by querying the corpus for building records that satisfy the image-type requirements of a target archetype. Image types (floorplan, interior, exterior, etc) are identified using zero shot image classification on the multimodal OpenCLIP embeddings described in Sec. 3.1. For example, a question sample for archetype 1 (“match photograph to floorplan”) can only be created from a building record that contains at least one floor plan and at least one photograph, while a mixed-drawing question sample (archetype 11) requires a building record that contains a floor plan, an elevation, a section, and at least one photograph. Once a building record has been selected, question images and the correct answer choice are sampled randomly from its eligible images, as determined by the precomputed image-type labels. Incorrect answer choices are sampled randomly from all eligible images of the required image type across the corpus.

However, after an initial round of question sample generation using random image selection , we observed that many generated question samples were either too easy or were unsolvable. We therefore introduced two embedding-based methods to improve image selection during question sample generation: diverse image selection within a property using agglomerative clustering, and distractor (incorrect answer choice) selection via visual similarity.

When selecting multiple representative images of a building record, we found that random selection would sometimes produce a set of images that were similar to each other, limiting the visual information about the building as a whole. We therefore developed a method to perform diverse image selection within a property using agglomerative clustering. We cluster candidate images with a distance threshold of 0.25, discard singleton clusters, and select centroidnearest representatives that maximize visual diversity. This makes the reference set less redundant and forces the mode to compare distinct views of the same property rather than nearly duplicated frames. If too few multi-image clusters survive, we fall back to round-robin selection across available clusters, starting with centroid-nearest images and then sampling additional images as needed.

![](images/5ea2dd94c504ec5f8eef35228bb01d5ed0e1e9c0270fbef72fe328f0b8d67f34.jpg)  
Figure 2. Cluster-based reference image selection, promoting both representativeness and visual diversity; selected images are outlined in green.

![](images/4ad1a380306cabdef71fbdaa36e1e1a72d8f56ae5749c3db6ce07b04cb184e08.jpg)  
Figure 3. Random versus OpenCLIP similarity answer selection. Random selection produces visually unrelated distractors, whereas OpenCLIP similarity selection yields more plausible and visually similar answer choices, increasing the question’s difficulty. The question depicted is the same as in Fig. 1.

We used image embeddings to improve our image selection for incorrect answer choices. Once the correct answer image for a question sample was selected, we selected the 3 nearest images in the vector space based on the cosine similarity of the OpenCLIP image embeddings, filtering out any images that came from the same building record as the correct answer image. This procedure enabled us to select visually similar incorrect answer images, thereby increasing the difficulty of the resultant question samples (Fig. 3).

## 3.3. LLM-in-the-loop Difficulty Selection

Automated question sample generation enabled us to generate a large number of candidate question samples, but not all generated question samples were suitable for inclusion in the final benchmark. We therefore required an automated method to filter the generated question sample set. Model-assisted filtering has previously been used to reduce benchmark shortcuts and retain more discriminative examples [10, 20]. Agreement among the screening models provided a difficulty signal: items answered by all models were likely to be easy, whereas items missed by all models were potentially difficult, ambiguous, or invalid and therefore required manual review.

We generated 2,200 candidate question samples (200 for each archetype) and evaluated them against four state-ofthe-art multimodal models: Claude Sonnet 4, Gemini 2.5 Pro, Pixtral Large, and OpenAI GPT-4.1. Questions answered correctly by at most half of the four screening models were treated as the primary pool for manual review. The model-difficulty-filtered pool contained 1,013 question samples at or below the 50% threshold: 147 at 0% model accuracy, 256 at 25%, and 610 at 50%. These 1,013 question samples were subsequently reviewed manually.

## 3.4. Manual Curation for Final Benchmark Composition

The model-difficulty-filtered pool of 1,013 question samples was evaluated manually to select a final set of questions for the benchmark.

We built a dedicated validation interface for reviewing generated questions alongside aggregated model results. Reviewers knew the coarse screening bucket for the question (0%, 25%, or 50% model accuracy), but they did not see model-specific answer choices or per-model correctness while making their initial judgment. Reviewers could first answer the question themselves and assess whether the item was solvable under the benchmark interface. They could then mark the question as valid or invalid.

Every question sample in the released benchmark was manually reviewed by two researchers. In practice, inclusion required that reviewers could identify a single correct answer, confirm that the images were of the correct type for the intended archetype, and determine that the item was human-solvable under the benchmark interface. Questions were rejected if they contained duplicate images, if the images were not of the correct type for the archetype, or if either reviewer judged the item unsolvable. When the two researchers initially disagreed, they reviewed the question together and reached consensus on whether to include it.

## 3.5. Release Protocol

We have released ARCH-B as a JSON dataset containing the finalized question samples, answer choices, archetype metadata, and links to the publicly hosted images at their original source locations. For each image, we include a cryptographic hash of the exact source bytes used in our evaluation. We will not redistribute third-party image files, but will retain archival copies for integrity and preservation; researchers may contact the authors regarding unavailable assets, subject to applicable rights and institutional policies. The released question set will be preserved as a fixed benchmark version, with subsequent corrections or removals documented in a changelog and issued as new versions.

To support reproducible model evaluation, we have also released the benchmark runner code used to download images, serialize each JSON question, construct multimodal prompts, query models, parse responses, and score outputs. The runner will verify image hashes and flag unretrievable question sample images. Finally, we provide a link to the hosted human-study interface so that readers can interactively attempt representative questions and better understand the visual reasoning tasks evaluated by the benchmark (available at https://archbench.codecolab.org/). The released benchmark dataset and runner code accompany this preprint as arXiv ancillary files.

## 4. Results

## 4.1. Benchmark Composition

The finalized benchmark contained 354 questions spanning 11 archetypes. Sec. A summarizes the final composition by archetype, including the final question count and dominant task for each archetype. The benchmark question samples were generated and curated according to the methods in Secs. 3.1 to 3.4.

## 4.2. Benchmark Evaluation Protocol

The evaluation pipeline loaded each question from the benchmark JSON and converted it into a single ordered multimodal user message containing the question text, any reference images, and four answer choices labeled with unique IDs. Before transmission, images were downloaded, decoded, resized to a maximum dimension of 1,024 pixels while preserving aspect ratio, and re-encoded as JPEG at 90% quality. All models received the answer choices in the same order and were instructed to return a structured output containing an answer choice id and a brief explanation.

Each model–question pair contributed exactly one scored outcome. Responses were processed using a deterministic parser, and an item was scored as correct only when the parsed answer ID exactly matched the ground-truth ID. The accompanying explanation was not used for scoring. If a model failed to return a response because of a timeout or terminal provider error, or returned an output that did not yield a syntactically valid answer ID, the outcome was counted as incorrect.

<table><tr><td>Rank Model</td><td></td><td>Family</td><td>Accuracy</td><td>Valid Response Rate</td></tr><tr><td>1</td><td></td><td>Google</td><td>83.90%</td><td>97.46%</td></tr><tr><td>2</td><td>gemini-3.1-pro-preview gemini-3.5-flash</td><td>Google</td><td>83.05%</td><td>98.87%</td></tr><tr><td>3</td><td>claude-fable-5</td><td>Anthropic</td><td>80.23%</td><td>98.87%</td></tr><tr><td>4</td><td>gpt-4.1</td><td>OpenAI</td><td>77.97%</td><td>100.00%</td></tr><tr><td>5</td><td>gpt-5.5 reasoning</td><td>OpenAI</td><td>76.55%</td><td>99.72%</td></tr><tr><td>6</td><td>gemini-3-flash-preview</td><td>Google</td><td>75.99%</td><td>98.02%</td></tr><tr><td>7</td><td>gpt-5.4 reasoning</td><td>OpenAI</td><td>74.29%</td><td>99.72%</td></tr><tr><td>8</td><td>gemini-2.5-pro</td><td>Google</td><td>71.47%</td><td>100.00%</td></tr><tr><td>9</td><td>gpt-5.5</td><td>OpenAI</td><td>70.90%</td><td>99.72%</td></tr><tr><td>10</td><td>claude-opus-4-6-v1</td><td>Anthropic</td><td>70.34%</td><td>100.00%</td></tr><tr><td>11</td><td>claude-opus-4-5-20251101-v1.0</td><td>Anthropic</td><td>65.25%</td><td>99.15%</td></tr><tr><td>12</td><td>gpt-5.4</td><td>OpenAI</td><td>64.41%</td><td>98.87%</td></tr><tr><td>13</td><td>grok-4.20-0309-reasoning</td><td>xAI</td><td>57.06%</td><td>99.15%</td></tr><tr><td>14</td><td>grok-4.3</td><td>xAI</td><td>50.28%</td><td>98.87%</td></tr><tr><td>15</td><td>gpt-5.4-mini reasoning</td><td>OpenAI</td><td>50.00%</td><td>100.00%</td></tr><tr><td>16</td><td>gpt-5.4-mini</td><td>OpenAI</td><td>46.61%</td><td>99.72%</td></tr><tr><td>17</td><td>gemma3 27b</td><td>Google</td><td>38.98%</td><td>99.72%</td></tr><tr><td>18</td><td>claude-sonnet-4-6</td><td>Anthropic</td><td>37.85%</td><td>60.45%</td></tr><tr><td>19</td><td>claude-sonnet-4-5-20250929-v1.0</td><td>Anthropic</td><td>34.46%</td><td>68.08%</td></tr><tr><td>20</td><td>pixtral-large-2502-v1.0</td><td>Mistral</td><td>32.77%</td><td>97.18%</td></tr><tr><td>21</td><td>gpt-5.4-nano reasoning</td><td>OpenAI</td><td>29.10%</td><td>99.72%</td></tr><tr><td>22</td><td>claude-haiku-4-5-20251001-v1.0</td><td>Anthropic</td><td>17.51%</td><td>94.63%</td></tr><tr><td>23</td><td>gemini-2.5-flash</td><td>Google</td><td>17.51%</td><td>24.29%</td></tr><tr><td>24</td><td>gpt-5.4-nano</td><td>OpenAI</td><td>16.95%</td><td>99.44%</td></tr><tr><td>25</td><td>gemma3 4b</td><td>Ollama</td><td>10.45%</td><td>100.00%</td></tr></table>

Table 1. Main model leaderboard on ARCH-B.

Across the 8,850 model-question-sample evaluations, 37 responses (0.42%) ended in terminal inference failures and 559 responses (6.32%) did not yield a valid answer ID. Per model valid response rates are reported alongside overall accuracy in Tab. 1. Exact provider model identifiers, access dates, inference settings, and per-model failure counts are provided in the supplemental material. The complete evaluation runner, including prompt construction, image preprocessing, response parsing, and scoring, is released with the benchmark question set.

## 4.3. Main Leaderboard

We evaluated 25 multimodal models on the full 354- question benchmark. Tab. 1 reports the exact leaderboard scores. Overall accuracy ranged from 10.45% to 83.90%, indicating that ARCH-B separates current systems across a wide performance band rather than saturating at either floor or ceiling. The strongest system was Gemini 3.1 Pro Preview, which answered 297 of 354 questions correctly (83.90%). It was followed closely by Gemini 3.5 Flash (83.05%), Claude Fable 5 (80.23%), GPT-4.1 (77.97%), and GPT-5.5 reasoning (76.55%). Valid-response rates were above 94% for most evaluated models, but substantially lower for Gemini 2.5 Flash (24.29%), Claude Sonnet 4.6 (60.45%), and Claude Sonnet 4.5 (68.08%). Consequently, the low aggregate accuracies of these models partly reflect failure to produce an answer in the required format rather than incorrect selections among the four choices.

![](images/3c8c90f5e3a73487b6cc1ea030b562236c33bab1aeab43500999c1e219cb6f32.jpg)  
Figure 4. Per-question model accuracy across the 354-question benchmark. Each bar is one question, ordered and grouped by archetype; bar height gives the percentage of the 25 evaluated models that answered the question correctly.

## 4.4. Performance by Archetype

Tab. 2 summarizes model performance by archetype. Models are strongest on mixed-representation outlier detection, same-building photograph matching, view selection from mixed drawings, and elevation-to-photo matching, while their weakest categories include floorplan-to-photo matching and photograph-only outlier detection.

## 4.5. Per-Question Difficulty Distribution

Fig. 4 presents a per-question model-accuracy bar chart, showing that ARCH-B is not saturated for current multimodal models. Mean per-question model accuracy was 53.36%, median accuracy was 56.00%, and individual questions ranged from 4.00% to 96.00%. Crucially, there were no question samples answered correctly by every model (too easy) and no questions missed by every model (impossible). 145 of 354 questions were answered correctly by at most half of the evaluated models, 36 questions were at or below chance-level model accuracy, and 299 of 354 questions were answered correctly by at most 75% of models rather than clustering near universal success. In combination with the manual validation described in Sec. 3.4, this suggests that the final benchmark contains a graded spectrum of difficult but solvable cross-representational correspondence problems.

## 4.6. Held-Out Model Analysis

To test whether LLM-in-the-loop selection introduced circularity, we grouped questions by how many of the four screening models answered them correctly and measured accuracy across all remaining models. Held-out accuracy increased monotonically from 28.65% for questions answered by 0/4 screening models $( n = 3 3 )$ , to 41.38% for 1/4 $( n = 7 7 )$ , and 59.02% for 2/4 $( n = 2 4 4 )$ . Thus, the difficulty trends identified by the screening models generalized to the broader model set rather than reflecting modelspecific weaknesses.

## 5. Human Study

## 5.1. Human Study Design

We conducted a human study to establish a human baseline score on the benchmark question sample set. Participants were recruited through Prolific, an online platform that connects researchers with verified study participants. The benchmark question sample set and evaluation interface were developed and hosted by us.

The study recruited English-speaking adults in the United States through the platform and forwarded them to our hosted evaluation interface. Each participant was presented with 10 question samples belonging to one archetype. The study was conducted in June 2026 and participants were paid \$2 per session. We recorded 583 completed sessions, resulting in 16 to 18 human responses per question sample. Our human study design was reviewed and approved by our institution’s IRB.

## 5.2. Results

We collected 5,830 valid human responses across the 354 benchmark questions, with 16–18 responses per question. Participants answered 2,061 responses correctly, yielding a response-weighted accuracy of 35.35% and a macroaverage question accuracy of 35.37%. Both measures exceed the four-choice chance level of 25%. Tab. 2 compares human accuracy with two summaries of model performance for each archetype: mean accuracy across all 25 evaluated models and the highest accuracy achieved by any evaluated model.

The mean-model and best-observed results describe different aspects of model performance. Mean model accuracy indicates how difficult an archetype was across the evaluated model set. Best observed accuracy reveals state of the art performance per archetype. Gemini 3.1 Pro Preview produced the best result for four archetypes, Gemini 3.5 Flash for three, GPT-4.1 for two, and Claude Fable 5, Claude Opus 4.5, and Gemini 3 Flash Preview for one each.

Human accuracy was highest for elevation-tophotograph matching (56.15%) and selecting a view from mixed drawings (42.44%), and lowest for photographonly outlier detection (21.61%) and exterior-from-interior matching (29.70%). Mean model accuracy was highest for mixed-representation outlier detection (64.22%) and lowest for photograph-only outlier detection (41.22%) and floorplan-to-photograph matching (44.14%). The best observed model accuracy ranged from 76.79% to 100%, exceeding human accuracy in every archetype by between 40.38 and 68.00 percentage points.

The relative ordering of archetypes differs between humans and models. Human accuracy and mean model accuracy have a weak rank correlation across archetypes (Spearman’s $\rho ~ = ~ 0 . 3 3 )$ , indicating limited agreement in which tasks are comparatively easy or difficult. Sec. 6.2 considers the possible failure modes underlying these differences.

<table><tr><td>Archetype</td><td>Task family</td><td>Best acc.</td><td>Best Model for Archetype</td><td>Mean model acc.</td><td>Human acc.</td><td>Best-human gap</td></tr><tr><td>0</td><td>Match floor plan with images</td><td>95.65</td><td>gemini-3.1-pro-preview</td><td>58.78</td><td>34.74</td><td>60.91</td></tr><tr><td>1</td><td>Match image with floor plan</td><td>76.79</td><td>gpt-4.1</td><td>44.14</td><td>36.41</td><td>40.38</td></tr><tr><td>2</td><td>Choose outlying image</td><td>84.78</td><td>gemini-3.1-pro-preview</td><td>41.22</td><td>21.61</td><td>63.17</td></tr><tr><td>3</td><td>Choose inlying image</td><td>100.00</td><td>gpt-4.1</td><td>60.17</td><td>32.00</td><td>68.00</td></tr><tr><td>4</td><td>Identify interior from exterior</td><td>96.00</td><td>gemini-3.1-pro-preview</td><td>52.64</td><td>35.85</td><td>60.15</td></tr><tr><td>5</td><td>Identify exterior from interior</td><td>85.00</td><td>claude-opus-4-5-20251101-v1.0</td><td>50.00</td><td>29.70</td><td>55.30</td></tr><tr><td>6</td><td>Identify elevation from floor plan</td><td>90.62</td><td>gemini-3.5-flash</td><td>50.25</td><td>36.95</td><td>53.67</td></tr><tr><td>7</td><td>Identify floor plan from elevation</td><td>90.00</td><td>gemini-3.5-flash</td><td>57.87</td><td>32.25</td><td>57.75</td></tr><tr><td>8</td><td>Identify outlier from mixed representations</td><td>100.00</td><td>claude-fable-5</td><td>64.22</td><td>33.93</td><td>66.07</td></tr><tr><td>9</td><td>Identify view from mixed drawings</td><td>96.55</td><td>gemini-3-flash-preview</td><td>59.86</td><td>42.44</td><td>54.11</td></tr><tr><td>10</td><td>Identify view from elevations</td><td>96.88</td><td>gemini-3.5-flash</td><td>61.00</td><td>56.15</td><td>40.73</td></tr></table>

Table 2. Archetype-level accuracy.

## 6. Discussion

## 6.1. Benchmark Saturation and Continued Use

Although the strongest evaluated model scores above 80%, ARCH-B still provides value as a comparative benchmark because performance is far from uniform across models and task families. Overall leaderboard accuracy spans a wide range, from 10.45% to 83.90%, and the archetypelevel results show that models differ not only in aggregate strength but in the kinds of architectural comprehension they handle well. This makes the benchmark useful as a way to measure relative capability: whether a model is strong on mixed-representation outlier detection, plan-to-elevation correspondence, interior-exterior matching, or floorplan-tophotograph transfer. In this sense, the benchmark’s value comes from its breadth and diagnostic structure. A high topline score does not eliminate the need to understand which representational transitions are easy, which remain brittle, and whether improvements in one task family generalize to others.

The question generation pipeline also has value beyond leaderboard evaluation; the same pipeline could be used to generate larger quantities of annotated data for targeted training or fine-tuning. For example, if a model is weak on floorplan-to-photograph matching or elevation-to-plan correspondence, the system could generate additional supervised examples for that specific task family rather than treating architectural visual understanding as a single undifferentiated capability.

## 6.2. Human and Model Failure Modes

The archetype-level results indicate that humans and models have overlapping but non-identical patterns of difficulty.

Photograph-only outlier detection was the weakest archetype for both groups, with 21.61% human accuracy and 41.22% mean model accuracy. Unlike positive matching, this task requires establishing consistency among several views while rejecting a visually similar photograph; agreement in materials, style, or context may therefore be mistaken for building identity. Interior-to-exterior matching was also difficult, particularly for humans (29.70%), plausibly because an interior provides only local and often nondistinctive evidence about the building’s exterior form.

Models exhibited a directional asymmetry that was largely absent for humans: mean model accuracy was 58.78% when selecting a floor plan from photographs, but only 44.14% when selecting a photograph from floor plans, whereas human accuracy was similar in the two directions (34.74% and 36.41%). This suggests that models can use the stable topology of candidate plans to explain observed views more readily than they can project an abstract plan into a plausible visual appearance. Conversely, humans performed best when matching elevations to photographs (56.15%), where facade proportions, window rhythms, and massing provide comparatively direct visual correspondences. The model advantage on mixedrepresentation outlier detection (64.22% versus 33.93%) may reflect a greater ability to pool weak cues across several representations, while the lower non-expert human result may partly reflect unfamiliarity with architectural drawings. These directional asymmetries are consistent with prior evidence that multimodal performance on visual correspondence depends strongly on viewpoint and representational form [5, 12, 18].

## 6.3. Human Study Improvements and Future Work

The human study is best interpreted as a non-expert baseline for the benchmark rather than a general comparison between people and models under all scenarios. Participants completed short sessions and were paid for completion rather than accuracy, so low performance may reflect architectural inexperience, limited attention, or unfamiliarity with plans and elevations in addition to item difficulty; related evaluations likewise find differences associated with architectural training [15]. Future studies should compare non-experts, architecture students, and practitioners, record response time and confidence, and test whether incentives or brief instruction improve performance.

![](images/390ecff849bd33a5eb6c251c83feb7db328c95bb06fdca5376af66aea68b6329.jpg)  
10. Identify view from elevations

![](images/395d9204f90a43f71adf80c5e4472c789c98eef370b773c01791f12b81dadc9f.jpg)

![](images/18058feb01e5f640b2dda4d0ddc2aea01bb1b65493590f2dec1d3efa78a0723f.jpg)  
n: 29  
n: 24

Future versions of ARCH-B could expand beyond photographs, plans, elevations, sections, and renderings to include axonometric drawings, diagrams, site plans, construction documents, three-dimensional models, and images from different stages of construction or use. They could also move beyond fixed answer choices to require localization, spatial ordering, evidence-based explanations, or reconstruction of relationships among rooms, floors, facades, and buildings. Controlled variation in distractor similarity, drawing style, image quality, building type, and geographic source would help distinguish robust representation transfer from reliance on superficial cues.

## 7. Conclusion

ARCH-B evaluates whether multimodal models can align photographs, plans, elevations, sections, and mixed evidence as representations of the same building. Evaluation across 25 models and a non-expert human baseline reveals substantial differences among systems and uneven performance across task types and individual questions. ARCH-B is therefore not a general measure of architectural intelligence, but a reusable stress test for visual correspondence and representation transfer across architectural media.

## A. Question Archetypes

![](images/51b34e35b34dac5e2474b2bb1d7d9247e1d76e9ad4c50a9bdd0a34671d896d5b.jpg)  
Question: Choose the floorplan that corresponds to the building shown in these photographs. Refs: 2 to 4 Photographs Choices: 4 Floor Plans

![](images/c504aae82a61aeb9600f93f951e45cd28535c04e06b8fd90d9140b7ddd5273bc.jpg)  
Question: Choose the photograph that corresponds to the building shown in this/these floorplan/s. Refs: 1 to 4 Floor Plans Choices: 4 Photographs

![](images/58593cb5cc4c0ed6106b25114ab09dae56912f65885acf99d9ac87bd1a9f5a7c.jpg)

## 2. Choose outlying image

Question: One of these photographs does NOT belong to the same building as the other three. Choose the OUTLIER Refs: None Choices: 4 Photographs

![](images/25a99e9e23d5b41e739f1ac10c04009d1310e0147a26cf00e8b025c7c4c5104b.jpg)

![](images/2b0a820e3719bbd7adfa3adf6abb60fa4d7367a9dbd4becc3f32492cc54f195f.jpg)

![](images/dc36bb2a3a83160892d2ff8079b627281c20e68cc0e87e0d8719fed1382e5ad4.jpg)

![](images/4e09581037b2df80896279896477cedc94e096a4850cb4c71f7a9486f3597f9b.jpg)  
Question: Choose the photograph that corresponds to the same building as these three. Refs: 3 Photographs Choices: 4 Photographs  
n: 25  
Question: Choose the interior photograph that corresponds to the same building as these exterior ones. Refs: 4 Exterior Photographs Choices: 4 Interior Photographs

![](images/e3188f85c32ebddb9ba3c65452d53df0abbc54f23dc0b024b1bcf178b777ffc0.jpg)

![](images/1038042be8acfa23c0cf2901a1739ac1697f9e37a1875454c3ec7a7aa732e37a.jpg)

![](images/4a7edf3633c14a3f1fd2e34dd0b604a7cff9e12b2b93d143c061e3396aa26116.jpg)

![](images/4e31ffbe7bf45402def4831eda10d2064ea096d254672d904338654c28e86a74.jpg)  
5. Identify exterior from interior Question: Choose the exterior photograph that corresponds to the same building as these interior ones. Refs: 4 Interior Photographs Choices: 4 Exterior Photographs

![](images/7db5a021f735f348dc8aa344b57ba389772ffebb035af9afebdddaf7962c03aa.jpg)  
6. Identify elevation from floor plan  
Question: Choose the elevation drawing that corresponds to the same building as this/these plan/s. Refs: 1 to 4 Floor Plans Choices: 4 Elevation Drawings

![](images/cafbfbdfa2f3f5a05c58eac0ba03458bb827adb12f60b5321b3646f317d7c7da.jpg)

![](images/841411d1645cb2a1446d27d864eb37852f6d2b0d096b91931c012417624d4388.jpg)  
Question: Choose the plan drawing that corresponds to the same building as this/these elevation/s. Refs: 1 to 4 Elevation Drawings Choices: 4 Floor Plans

![](images/afa4c08aac5e442abfdb06228d417978a2a3b4bc5118d70242e2efbb86ebe0c9.jpg)

![](images/2b0d6ecf15ed31804a4f580ccea6fe5946e157352930655da0fd4072397ded48.jpg)

![](images/c594e9598587c67a06f21a662044e867e28dc1064f69acd9c9abf116ccedf775.jpg)  
Question: Choose the photograph that corresponds to the building shown in these drawings. Refs: 1 Floor Plan, 1 Elevation Drawing, and 1 Section Drawing Choices: 4 Photographs  
Question: Choose the photograph that corresponds to the building shown in these elevations. Refs: 2 Elevation Drawings Choices: 4 Exterior Photographs

## References

[1] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2829, 2023. 3

[2] Steve Cruz, Will Hutchcroft, Yuguang Li, Naji Khosravan, Ivaylo Boyadzhiev, and Sing Bing Kang. Zillow indoor dataset: Annotated floor plans with 360-degree panoramas and 3d room layouts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2133–2143, 2021. 2

[3] Chenxu Du, Kang An, Tengyue Wang, Zhongyu Yang, Xinqi Yang, Yuanchi Zhu, Hebao Zhu, Ziliang Wang, Faqiang Qian, Yunli Yang, and Qibing Ren. MMArch: Benchmarking multimodal reasoning grounded in architectural evidence. arXiv preprint arXiv:2608.09281, 2026. 1, 2

[4] Zhiwen Fan, Lingjie Zhu, Honghua Li, Xiaohao Chen, Siyu Zhu, and Ping Tan. FloorPlanCAD: A large-scale CAD drawing dataset for panoptic symbol spotting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10128–10137, 2021. 2

[5] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal large language models can see but not perceive. In Computer Vision – ECCV 2024, pages 148–166. Springer, 2024. 1, 2, 7

[6] Keren Ganon, Morris Alper, Rachel Mikulinsky, and Hadar Averbuch-Elor. WAFFLE: Multimodal floorplan understanding in the wild. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 1488–1497, 2025. 1, 2

[7] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6700–6709, 2019. 2, 3

[8] Ahti Kalervo, Juha Ylioinas, Markus Haiki¨ o, Antti Karhu,¨ and Juho Kannala. CubiCasa5K: A dataset and an improved multi-task model for floorplan image analysis. In Image Analysis: 21st Scandinavian Conference, SCIA 2019, pages 28–40. Springer, 2019. 2

[9] Aleksei Kondratenko, Mussie Birhane, Houssame E. Hsain, and Guido Maciocci. AECV-Bench: Benchmarking multimodal models on architectural and engineering drawings understanding. arXiv preprint arXiv:2601.04819, 2026. 1, 2

[10] Ronan Le Bras, Swabha Swayamdipta, Chandra Bhagavatula, Rowan Zellers, Matthew Peters, Ashish Sabharwal, and Yejin Choi. Adversarial filters of dataset biases. In Proceedings of the 37th International Conference on Machine Learning, pages 1078–1088. PMLR, 2020. 2, 4

[11] Yueheng Lu, Runjia Tian, Ao Li, Xiaoshi Wang, and Jose Luis Garcia del Castillo y Lopez. Organizational graph generation for structured architectural floorplan dataset. In Proceedings of the 26th International Conference of the

Association for Computer-Aided Architectural Design Research in Asia (CAADRIA), CUMINCAD, 2021. 2

[12] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3DSR-Bench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025. 1, 2, 7

[13] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 3

[14] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In Advances in Neural Information Processing Systems, pages 25278–25294, 2022. 3

[15] Qirui Shen, Wenda Wang, Jiachen Lu, Zilong Huang, Jin Bai, Lei He, Hongxuan Chen, and Weixin Huang. ArchSIBench: Benchmarking the architectural spatial intelligence of vision-language models. arXiv preprint arXiv:2605.20837, 2026. 1, 2, 7

[16] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In Advances in Neural Information Processing Systems, 2024. 1, 2

[17] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024. 2

[18] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10632– 10643, 2025. 1, 2, 7

[19] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024. 1, 2

[20] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge

Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 2, 4