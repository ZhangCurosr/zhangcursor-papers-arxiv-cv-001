# BENCHMARKING VISION-LANGUAGE MODELS ON SYNAPSE DETECTION AND PROOFREADING IN CONNECTOMICS

Yicong Li<sup>1,∗</sup>, Junjie Wang<sup>2</sup>, Leander Lauenburg<sup>1</sup>, Ella Hugie<sup>1</sup> Alexandra Irger<sup>1</sup>, Wanhua Li<sup>3</sup>, Donglai Wei<sup>2</sup>, Hanspeter Pfister<sup>1,∗</sup>

<sup>1</sup>Harvard University, Cambridge, MA, USA <sup>2</sup>Boston College, Chestnut Hill, MA, USA <sup>3</sup>Nanyang Technological University, Singapore

## ABSTRACT

We benchmarked vision-language models (VLMs) on the decisions annotators take when inspecting electron microscopy images in connectomics: synapse detection (presence and polarity) and proofreading (split errors and merge errors). For synapse detection, we evaluated 19 open and 2 closed models across various architectures and sizes under zero-shot, four-shot in-context learning and LoRA settings, against specialist models, on datasets constructed by us using public resources. For proofreading, we evaluated 3 open and 2 closed models on the ConnectomeBench2 dataset, with crossspecies transfer from fly and mouse to human and zebrafish. Most models were at chance zero-shot; a few examples helped mainly the closed and largest open ones. LoRA on a few thousand labels brought open models level with specialist models. When evaluated on unseen species, the best adapted VLMs outperformed specialist models trained on the same data in identifying merge errors. The project will be publicly available upon acceptance.

Index Terms— Connectomics, electron microscopy, visionlanguage models, synapse detection, proofreading

## 1. INTRODUCTION

Connectomics aims to reconstruct the wiring diagram of the brain: every neuron and every synapse between them. Currently, electron microscopy (EM), while expensive [1, 2], is the primary imaging method that resolves both, at nanometer resolution over volumes that now reach peta-byte scale [3, 4], and the field has drawn considerable attention as these datasets appeared [5]. With the development of computer vision techniques, the reconstruction pipeline is mostly automatic in its two main stages: segmentation traces each neuron through the volume [6, 7], and synapse detection finds the contacts between them and assigns their direction [8, 9]. However, both stages still need human eyes. Synapse detectors are validated and corrected by annotators who judge, section by section, whether a contact is a synapse and which side is presynaptic. Segmentations are proofread, because a split or merge error propagates through every circuit the neuron takes part in; FlyWire took about three million proofreading edits [3, 10], and MICrONS proofreading continues years after imaging [4]. Each of these checks is a small visual decision taken on a few EM sections.

Vision-language models (VLMs) answer visual questions in natural language and can be steered by a few examples or a short finetuning run, yet on general microscopy benchmarks they still struggle even with modality identification [11]. Whether they can make an annotator’s decision from raw EM, and what it takes to get them there, is largely unknown.

Prior work on these two tasks is supervised and task-specific. Synapse detection has been posed as voxel classification [8], as partner assignment from point annotations [9, 12], and as direct mask generation for the pre- and postsynaptic partners of a given cleft [13]. On the proofreading side, guided proofreading trains CNNs to flag split and merge errors and to propose the correction [14]. More recently, ConnectomeBench [15] evaluated several multimodal LLMs on proofreading tasks, but only on mesh renderings rather than EM images. Its successor, ConnectomeBench2 [16] released expert-labelled sites with EM slices from four species and trained a ViT to compare with human performance, without evaluating language models.

To the best of our knowledge, no prior study has evaluated a range of VLMs on synapse detection from EM, and none has benchmarked VLMs on proofreading tasks using ConnectomeBench2 [16], whose only published results are for a supervised ViT. Concretely, our contributions are:

• We built a synapse detection benchmark from public fly and mouse connectomes and evaluated 19 open and two closed VLMs on it zero-shot, with four in-context examples, and with LoRA, against specialist networks trained on the same images.

• We presented the first evaluation of VLMs on ConnectomeBench2: three open models were run on it in full against the dataset’s own ViT, and the same models were compared with two closed VLMs and three specialist networks on mouse and fly subsets.

• We found that most open models were at chance zero-shot, many returning one fixed answer, while the closed models read proofreading above chance. In-context examples helped the closed models and the largest open ones, and LoRA finetuning brought open models level with the specialists. Additionally, VLMs seem to be more resilient to cross-species transfer compared with specialist models.

## 2. BENCHMARK

## 2.1. Task 1: Synapse detection

We built the synapse detection tasks from two public sources with ground-truth synapse annotations, CREMI [17] for fly and MI-CrONS [4] for mouse. Every item is a single EM section of 256×256 pixels at 8 nm per pixel, a 2.05 �m field of view. For the presence subtask, a yellow circle of 224 nm radius marks the query point and the model is asked whether a chemical synapse is present there. For the polarity subtask, two markers, A and B, are placed inside the two partner processes of a synapse, with the letters assigned at random, and the model is asked which marker is presynaptic.

![](images/519464005499f69cfb0b8d5b6e4b50844589d071aaa8a3bf27e84adf8941347a.jpg)  
Fig. 1. Synapse detection example inputs of fly. Left: presence, the query point marked by the yellow circle (answer: yes). Right: polarity, markers A and B inside the two partner processes (answer: B).

Fig. 1 shows one item of each subtask and below is the accompanying prompt:

Preamble (both subtasks): This is a 2 um x 2 um electron micrograph of brain tissue (adult Drosophila brain), imaged at 8 nm per pixel. Dark membranes outline neuronal processes; mitochondria appear as dark oval organelles; synaptic vesicles are small round ∼40 nm blobs clustered inside axon terminals. Presence: A yellow circle marks the location of interest. Is there a chemical synapse (a synaptic cleft with a presynaptic vesicle cluster on one side and a postsynaptic density on the other) at the marked location? Answer with exactly one word: yes or no. Polarity: Two markers, A and B, are placed inside two adjacent neuronal processes that form a chemical synapse at their shared membrane. Which marker is inside the PRESYNAPTIC process, i.e. the side that contains the cluster of synaptic vesicles? Answer with exactly one letter: A or B.

For fly we used the three CREMI volumes [17], adult Drosophila brain at 4 × 4 × 40 nm with hand-labelled synaptic clefts and their pre- and postsynaptic partners. We assigned volume A to training, B to validation and C to testing, so the three splits come from different brain regions. Positives samples for the presence subtask were created by placing one query at the centroid of every cleft, in the section where the cleft is largest. For each positive we then drew one negative at a random location that lies at least 256 nm from any cleft voxel in its section and 600 nm from every annotated synapse, and we kept the query point at the same offset from the crop centre as in the paired positive, so that the offset alone carries no information. For polarity we took the annotated partner pairs and placed the markers 72 to 240 nm from the cleft and at least 112 nm apart. This procedure yielded 246, 262 and 330 presence items for training, validation and test, half of them positive, and 170, 275 and 413 polarity pairs.

For mouse we used the public MICrONS cubic-millimetre volume of visual cortex (minnie65) [4], available at 8 × 8 × 40 nm in its public mirror, which comes with a cleft segmentation and an automatically generated synapse table. Because the table is automatic, we kept only entries of at least 300 voxels whose cleft is confirmed at the crop centre, and we drew negatives at random points with no cleft within 256 nm. To keep the splits apart, we divided the volume into contiguous blocks along one axis, separated by gaps, so that training, validation and test tissue never overlap. This gave 6000, 1000 and 3000 samples for the presence subtask, half positive, and 3000, 500 and 1500 polarity pairs.

(a)  
![](images/d111878b28e72c98cdbf7a304578345eaa37ce16633873e962b28d4dff1cde93.jpg)

(b)  
![](images/04887fb14eb785278039b99f7bdc13125738e8936e748cd334d1403ef132a888.jpg)  
Fig. 2. Proofreading example inputs of fly. (a) A split-error site: the two segments are tinted red and blue; the proofreaders merged them, so the answer is yes. (b) A merge-error site: the segment is tinted red; the proofreaders split it, so the answer is no.

## 2.2. Task 2: Proofreading

For proofreading we did not build a new dataset but adopted ConnectomeBench2 (CB2) [16], which released proofreading sites from the edit histories of expert proofreaders, together with matched controls, in mouse, fly, human and zebrafish. Each site comes as four EM views taken before the edit, the ��, �� and �� planes and an oblique plane, at 224 × 224 pixels, together with the masks of the segments involved; the cutouts span 2.5 �m in mouse and human and 1.5 �m in fly and zebrafish. We kept the released images unchanged and only tinted the masks, so that the model knows which segments the question is about. In the split-error subtask, the two segments are tinted red and blue in all four planes, and the answer is yes when the proofreaders merged them, that is, at a split error. In the merge-error subtask, the union of the segments is tinted red in the three planes, and the answer is no when the proofreaders split it, that is, at a merge error. We dropped the oblique plane for this task because its orientation is chosen from the post-split masks and would leak the answer, as the dataset card itself warns.

Fig. 2 shows one item of each subtask and below is the accompanying prompt:

Split error: The four panels are electron-microscopy slices through the same location in adult Drosophila brain, each about 1.5 um across: top-left xy, top-right xz, bottom-left yz, bottom-right an oblique plane. Two segments from an automatic neuron segmentation are tinted RED and BLUE and meet near the centre. Automatic segmentations sometimes wrongly split one neuron into two segments, whereas two different cells are separated by a continuous dark membrane. Do the red and blue segments belong to the SAME neuron, so that they should be merged? Answer with exactly one word: yes or no. Merge error: The three panels are electronmicroscopy slices (xy, xz, yz) through the same location in adult Drosophila brain, each about 1.5 um across. One segment from an automatic neuron segmentation is tinted RED. Automatic segmentations sometimes wrongly merge two different cells across a membrane near the centre of the view. Is the red segment a SINGLE neuron here, with no merge error at the centre? Answer with exactly one word: yes or no.

Table 1. Synapse detection, balanced accuracy (%) on the test splits. P: presence; Pol: polarity. Bold: best VLM per column; underlined: second best; Italic rows: specialist models, not ranked; –: not applicable.
<table><tr><td></td><td colspan="4">zero-shot</td><td colspan="4">4-shot</td><td colspan="4">LoRA</td></tr><tr><td>Model</td><td>fly P</td><td>fly Pol</td><td>mouse P</td><td>mouse Pol</td><td>fly P</td><td>fly Pol</td><td>mouse P</td><td>mouse Pol</td><td>fly P</td><td>fly Pol</td><td>mouse P</td><td>mouse Pol</td></tr><tr><td>Qwen2.5-VL-3B</td><td>49.1</td><td>50.0</td><td>49.5</td><td>50.0</td><td>50.9</td><td>50.0</td><td>53.9</td><td>50.0</td><td>50.3</td><td>57.8</td><td>74.4</td><td>76.4</td></tr><tr><td>Qwen2.5-VL-7B</td><td>50.0</td><td>49.0</td><td>49.6</td><td>49.2</td><td>48.8</td><td>49.1</td><td>48.2</td><td>49.3</td><td>71.8</td><td>70.0</td><td>86.3</td><td>81.4</td></tr><tr><td>Qwen2.5-VL-32B</td><td>48.2</td><td>49.1</td><td>46.2</td><td>49.8</td><td>55.2</td><td>49.4</td><td>52.2</td><td>50.3</td><td>68.2</td><td>61.7</td><td>83.5</td><td>77.4</td></tr><tr><td>Qwen3-VL-2B</td><td>51.2</td><td>49.1</td><td>48.3</td><td>50.9</td><td>50.0</td><td>48.7</td><td>54.1</td><td>50.0</td><td>69.7</td><td>56.9</td><td>81.0</td><td>79.9</td></tr><tr><td>Qwen3-VL-4B</td><td>53.3</td><td>48.2</td><td>47.5</td><td>51.9</td><td>59.1</td><td>51.7</td><td>48.8</td><td>49.8</td><td>72.4</td><td>56.3</td><td>83.7</td><td>82.1</td></tr><tr><td>Qwen3-VL-8B</td><td>52.4</td><td>56.8</td><td>47.1</td><td>49.1</td><td>57.6</td><td>54.7</td><td>56.5</td><td>49.9</td><td>73.0</td><td>66.0</td><td>83.1</td><td>87.8</td></tr><tr><td>Qwen3-VL-32B</td><td>50.0</td><td>54.9</td><td>49.9</td><td>51.3</td><td>71.2</td><td>58.4</td><td>56.1</td><td>50.7</td><td>71.5</td><td>76.2</td><td>82.9</td><td>89.2</td></tr><tr><td>Qwen3.5-2B</td><td>53.0</td><td>56.9</td><td>51.7</td><td>49.7</td><td>57.0</td><td>52.1</td><td>55.9</td><td>48.7</td><td>75.5</td><td>65.1</td><td>87.1</td><td>87.6</td></tr><tr><td>Qwen3.5-4B</td><td>50.0</td><td>50.5</td><td>50.0</td><td>50.1</td><td>52.1</td><td>64.2</td><td>51.0</td><td>50.3</td><td>80.0</td><td>74.6</td><td>87.9</td><td>90.1</td></tr><tr><td>Qwen3.5-9B</td><td>50.0</td><td>71.9</td><td>50.0</td><td>52.5</td><td>75.8</td><td>72.4</td><td>56.5</td><td>49.4</td><td>82.7</td><td>75.4</td><td>89.0</td><td>89.9</td></tr><tr><td>Qwen3.5-27B</td><td>68.5</td><td>58.5</td><td>55.4</td><td>51.8</td><td>73.0</td><td>69.0</td><td>63.9</td><td>52.4</td><td>82.7</td><td>72.0</td><td>87.5</td><td>89.3</td></tr><tr><td>InternVL3-2B</td><td>49.7</td><td>50.2</td><td>49.3</td><td>50.1</td><td>50.0</td><td>48.0</td><td>49.9</td><td>46.0</td><td>77.3</td><td>61.2</td><td>88.8</td><td>77.2</td></tr><tr><td>InternVL3-8B</td><td>51.5</td><td>45.2</td><td>48.4</td><td>48.4</td><td>67.6</td><td>46.1</td><td>47.1</td><td>46.3</td><td>79.1</td><td>73.4</td><td>89.8</td><td>87.0</td></tr><tr><td>InternVL3-14B</td><td>51.5</td><td>47.1</td><td>50.4</td><td>48.4</td><td>60.3</td><td>48.4</td><td>52.5</td><td>48.2</td><td>80.0</td><td>72.3</td><td>89.8</td><td>86.9</td></tr><tr><td>InternVL3-38B</td><td>50.6</td><td>47.2</td><td>56.1</td><td>49.5</td><td>65.2</td><td>56.3</td><td>62.2</td><td>51.2</td><td>80.6</td><td>75.6</td><td>91.6</td><td>88.6</td></tr><tr><td>Gemma 4 E2B</td><td>52.1</td><td>48.4</td><td>47.3</td><td>49.8</td><td>51.2</td><td>49.6</td><td>49.8</td><td>52.7</td><td>60.9</td><td>53.6</td><td>79.4</td><td>76.2</td></tr><tr><td>Gemma 4 E4B</td><td>50.0 60.6</td><td>52.1 75.0</td><td>49.5</td><td>50.2</td><td>65.5</td><td>59.0</td><td>48.1</td><td>48.2</td><td>72.1</td><td>59.0</td><td>83.2</td><td>75.5</td></tr><tr><td>Gemma 4 12B</td><td>73.6</td><td>70.1</td><td>51.8 60.9</td><td>61.5</td><td>72.1</td><td>83.2</td><td>56.4</td><td>61.5</td><td>81.5</td><td>91.1</td><td>93.5</td><td>94.5</td></tr><tr><td>Gemma 431B</td><td></td><td></td><td></td><td>66.6</td><td>81.8</td><td>76.0</td><td>64.4</td><td>66.9</td><td>85.5</td><td>75.7</td><td>88.0</td><td>88.1</td></tr><tr><td>GPT-5.6 Sol</td><td>59.7</td><td>77.2</td><td>45.0</td><td>56.8</td><td>79.4</td><td>83.1</td><td>56.3</td><td>60.9</td><td>1 1</td><td>1</td><td>1</td><td></td></tr><tr><td>Claude Opus 5</td><td>59.7</td><td>63.4</td><td>57.2</td><td>52.0</td><td>68.2</td><td>65.7</td><td>62.2</td><td>53.7</td><td></td><td></td><td></td><td></td></tr><tr><td>ResNet-50</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>83.8</td><td>75.9</td><td>95.6</td><td>95.3</td></tr><tr><td>ConvNeXt-T ViT-S/14 (DINOv2)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>85.6 79.7</td><td>82.6 60.5</td><td>96.4 95.3</td><td>95.6 92.2</td></tr></table>

We evaluated proofreading on three different setups. The first is the complete test split of the dataset’s second release, 101 929 operations with their natural class mix, namely 20 879 split-error and 22 094 merge-error operations in mouse, 24 065 and 24 203 in fly, 5 351 and 60 in human, and 5 223 and 54 in zebrafish; models were trained on its complete training and validation split of 531 734 and 82 822, respectively. The second is a balanced subset of mouse and fly with 3000, 600 and 1000 sites per species and task for training, validation and test, on which the models can be compared with closed-source VLMs and specialist networks under our budget. The third measures transfer to the species that no model developed using the second setup ever saw in training: namely, human and zebrafish.

## 2.3. Evaluation protocol

We report balanced accuracy throughout and each model saw one image and one prompt per item; the prompt states the image scale, the marker convention and the definition of the target, and asks for exactly one word or letter. We decoded greedily with at most 64 new tokens and took the last yes/no or A/B in the reply as the decision; any other reply counts as wrong. All three regimes shared the same images and prompts. In the zero-shot regime the model received only the test sample. In the 4-shot regime we prepended four training examples of the same species and subtask as prior turns, two for each answer. In the LoRA regime the model was fine-tuned with the recipe given in Sec. 2.4. As references, we trained specialist networks on exactly the same rendered images and the same label pools as the adapters.

## 2.4. Models

For synapse detection, we evaluated several open families of instruction-tuned VLMs across their size ladders: Qwen2.5-VL (3B, 7B, 32B) [18], Qwen3-VL (2B, 4B, 8B, 32B) [19], Qwen3.5 (2B, 4B, 9B, 27B) [20], InternVL3 (2B, 8B, 14B, 38B) [21] and Gemma 4 (E2B, E4B, 12B, 31B) [22]. For proofreading, due to budget constraints, we kept three mid-sized models, one each from the Qwen, InternVL, and Gemma lines: Qwen3.5-9B, InternVL3-8B and Gemma 4 12B. We also evaluated two closed models, GPT-5.6 Sol and Claude Opus 5, as zero-shot and in-context reference rows.

To fine-tune the open VLMs we used LoRA [23] with one recipe throughout: rank-16 adapters with � = 32 and dropout 0.05 on the attention and MLP projections of the language model, the vision encoder and the bfloat16 base frozen, AdamW at a learning rate of 10<sup>−4</sup>, a short linear warm-up followed by a cosine schedule, and only the answer token and the end-of-turn token supervised.

As references we fine-tuned three specialist networks: ResNet-50 [24] and ConvNeXt-Tiny [25], initialized from ImageNet, and a ViT-S/14 initialized from DINOv2 [26]. Each was trained end to end on exactly the rendered images and the label pool of the corresponding adapters, one model per subtask covering both species.

Table 2. Proofreading on the complete ConnectomeBench2, balanced accuracy (%) on the test split: all 101,929 samples across 4 species. S: split-error subtask, M: merge-error subtask. Bold: best VLM per column; Italic row: the dataset’s own reported EM-only ViT-B; –: not applicable.
<table><tr><td></td><td colspan="2">zero-shot</td><td colspan="2">4-shot</td><td colspan="2">LoRA</td></tr><tr><td>Model</td><td>S</td><td>M</td><td>S</td><td>M</td><td>S</td><td>M</td></tr><tr><td>Qwen3.5-9B</td><td>50.0</td><td>52.9</td><td>58.7</td><td>57.3</td><td>94.6</td><td>90.2</td></tr><tr><td>InternVL3-8B</td><td>51.0</td><td>50.0</td><td>49.0</td><td>57.3</td><td>93.9</td><td>90.1</td></tr><tr><td>Gemma 412B</td><td>50.1</td><td>50.0</td><td>62.7</td><td>58.0</td><td>95.0</td><td>90.9</td></tr><tr><td>CB2 ViT-B, EM only</td><td></td><td></td><td></td><td></td><td>94.2</td><td>91.1</td></tr></table>

Table 3. Proofreading on the subset of ConnectomeBench2 (Sec. 2.2), balanced accuracy (%). S: split-error subtask, M: mergeerror subtask. Bold: best VLM per row; italic columns: specialist models, not ranked; –: not applicable.
<table><tr><td></td><td>-9B</td><td>Int--8B</td><td>Gm12B</td><td>GP-5S0I1</td><td>5 ; ude pps</td><td>Res-50</td><td>Connvt-T</td><td>VT-S14</td></tr><tr><td colspan="9">zero-shot</td></tr><tr><td>mouse S</td><td>50.0</td><td>53.4</td><td>50.4</td><td>67.2</td><td>73.1</td><td></td><td></td><td></td></tr><tr><td>mouse M</td><td>59.3</td><td>50.0</td><td>50.0</td><td>67.6</td><td>67.3</td><td></td><td></td><td></td></tr><tr><td>fly S</td><td>50.0</td><td>52.0</td><td>50.0</td><td>63.0</td><td>64.7</td><td></td><td></td><td></td></tr><tr><td>fly M</td><td>51.0</td><td>50.0</td><td>50.0</td><td>57.0</td><td>57.0</td><td></td><td></td><td></td></tr><tr><td colspan="9">4-shot</td></tr><tr><td>mouse S</td><td>60.2</td><td>49.7</td><td>64.5</td><td>74.1</td><td>76.4</td><td></td><td></td><td></td></tr><tr><td>mouse M</td><td>69.1</td><td>54.1</td><td>57.7</td><td>65.7</td><td>68.0</td><td></td><td></td><td></td></tr><tr><td>fly S</td><td>60.4</td><td>48.9</td><td>64.7</td><td>74.4</td><td>66.4</td><td></td><td></td><td></td></tr><tr><td>flyM</td><td>61.7</td><td>50.0</td><td>60.0</td><td>67.7</td><td>61.9</td><td></td><td></td><td></td></tr><tr><td colspan="9">LoRA, trained on mouse + fly</td></tr><tr><td>mouse S</td><td>93.3</td><td>89.3</td><td>92.5</td><td></td><td></td><td>94.9</td><td>94.9</td><td>85.3</td></tr><tr><td>mouse M</td><td>84.6</td><td>84.4</td><td>83.4</td><td></td><td></td><td>86.1</td><td>85.9</td><td>82.0</td></tr><tr><td>fly S</td><td>86.6</td><td>83.1</td><td>85.2</td><td></td><td></td><td>89.2</td><td>90.1</td><td>80.8</td></tr><tr><td>flyM</td><td>80.1</td><td>78.8</td><td>74.5</td><td></td><td></td><td>76.7</td><td>79.7</td><td>74.2</td></tr><tr><td colspan="9">transfer to unseen species</td></tr><tr><td>human S</td><td>83.1</td><td>72.2</td><td>79.4</td><td></td><td></td><td>81.9</td><td>85.4</td><td>65.7</td></tr><tr><td>human M</td><td>71.9</td><td>67.6</td><td>77.3</td><td></td><td></td><td>59.8</td><td>66.9</td><td>57.9</td></tr><tr><td>zebrafish S</td><td>88.0</td><td>79.0</td><td>85.4</td><td></td><td></td><td>87.3</td><td>89.3</td><td>78.8</td></tr><tr><td>zebrafish M</td><td>77.3</td><td>73.5</td><td>82.7</td><td></td><td></td><td>70.5</td><td>74.7</td><td>65.6</td></tr></table>

## 3. RESULTS

## 3.1. Synapse detection

Table 1 shows the synapse detection results. Zero-shot, most open models couldn’t interpret the images: across the 76 cells of the 19 open models the median score was 50.0 and 62 cells fell below 55. The exceptions were the two largest Gemma 4 and Qwen3.5 models, mostly on fly, where Gemma 4 31B reached 73.6 on presence and Gemma 4 12B 75.0 on polarity; on mouse no open model passed 67. The closed models stayed at or near chance on mouse, but GPT-5.6 Sol read fly polarity at 77.2, the highest zero-shot value in the table, while staying at 59.7 on fly presence.

Four in-context examples raised the performance: Gemma 4 12B reached 83.2 on fly polarity, above the best specialist at 82.6, Gemma 4 31B 81.8 on fly presence, and Qwen3.5-9B 75.8 on fly presence, whereas the best mouse cells stayed at 64.4 and 66.9. GPT-5.6 Sol reached 79.4 and 83.1 on the two fly subtasks, and Claude

Opus 5 68.2 and 65.7.

LoRA changed the picture entirely. 63 of 76 cells reached 70, and the best adapter per cell reached 85.5 on fly presence, 91.1 on fly polarity, 93.5 on mouse presence and 94.5 on mouse polarity, ahead of or level with the specialists on fly and within three points of them on mouse. Size mattered less once adapted: Gemma 4 12B beat Gemma 4 31B on three of the four cells, and Qwen3.5-9B matched or beat Qwen3.5-27B on all four.

## 3.2. Proofreading

Full ConnectomeBench2 evaluation. Table 2 reports the three open models evaluated on the complete ConnectomeBench2 (CB2) under the dataset’s own protocol. Zero-shot, all three sat at or near chance. Four in-context examples lifted the split-error subtask scores of Qwen3.5-9B and Gemma 4 12B to 58.7 and 62.7 and the merge-error scores of all three to 57 to 58. LoRA brought all three level with the dataset’s own EM-only ViT-B, at 94.2 and 91.1: Gemma 4 12B reached 95.0 and 90.9, and the other two were within about one point on both tasks.

Balanced subsets, closed models, and transfer. Table 3 compares the same three models with the two closed models and the specialist models on the balanced mouse and fly subset of CB2, and shows how their adapters transfer. Zero-shot, the open models were again at or near chance, apart from Qwen3.5-9B at 59.3 on mouse merge errors, whereas the closed models performed better: Claude Opus 5 scored 73.1 and 64.7 on split errors in mouse and fly, GPT-5.6 Sol 67.2 and 63.0. With four examples the closed models rose to 76.4 and 74.4 at best, Gemma 4 12B and Qwen3.5-9B reached 58 to 69, and InternVL3-8B barely moved. LoRA took the open models to 83.1 to 93.3 on split errors and 74.5 to 84.6 on merge errors, within 1.6 points of the references on mouse, 3.5 points behind on fly split errors, and ahead on fly merge errors, 80.1 against 79.7. Applied unchanged to human and zebrafish, the adapters kept most of this accuracy. On split errors the best adapter transferred about as well as the best specialist network, Qwen3.5-9B at 83.1 and 88.0 against 85.4 and 89.3 for ConvNeXt-Tiny. On merge errors the best adapter beat the best network on both species, Gemma 4 12B reaching 77.3 and 82.7 against 66.9 and 74.7, and on human all three adapters did.

## 4. DISCUSSION AND CONCLUSION

We benchmarked vision-language models for synapse detection and proofreading tasks on EM images in connectomics. Without adaptation, most open models are at or near chance and cannot be dropped into an annotation pipeline; the closed models performed better under zero-shot and gained from in-context examples, but stay well below the adapted open models. A few thousand labels with LoRA bring an open model level with a purpose-trained network, and the adapters carry to species never seen in training, the best of them beating the specialist networks on merge errors. What makes the VLM worth its cost is not accuracy in the trained setting, where a small network is about as good and cheaper per image, but what happens away from it. One adapter per task answered both questions in both species, where the specialist models needed one network per question, and the adapters generally lost less accuracy on unseen species, which suggests that large-scale pretraining on images and text supplies context the small models lack. A VLM is therefore the right tool when the decisions are many or when a volume comes from a species with no training data; a purpose-trained network remains the right tool for one fixed decision on one well-labelled dataset.

## 5. ACKNOWLEDGEMENTS

This research is supported by NSF grant NCS-FO-2124179 and NIH grant R01HD104969. The authors have no relevant financial or nonfinancial interests to disclose. The use of AI systems is only for editing and grammar enhancement.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study used only publicly released, de-identified datasets, CREMI, MICrONS and ConnectomeBench2, under their open licenses, including the human tissue sites in ConnectomeBench2’s public release. No new human or animal data was collected and ethical approval was not required.

## 7. REFERENCES

[1] Yicong Li, Yaron Meirovitch, Aaron T. Kuan, Jasper S. Phelps, Alexandra Pacureanu, Wei-Chung Allen Lee, Nir Shavit, and Lu Mi, “X-Ray2EM: Uncertainty-aware cross-modality image reconstruction from X-ray to electron microscopy in connectomics,” in IEEE International Symposium on Biomedical Imaging (ISBI), 2023.

[2] Aaron T Kuan, Jasper S Phelps, Logan A Thomas, Tri M Nguyen, Julie Han, Chiao-Lin Chen, Anthony W Azevedo, John C Tuthill, Jan Funke, Peter Cloetens, et al., “Dense neuronal reconstruction through x-ray holographic nanotomography,” Nature neuroscience, vol. 23, no. 12, pp. 1637– 1643, 2020.

[3] Sven Dorkenwald, Arie Matsliah, Amy R. Sterling, et al., “Neuronal wiring diagram of an adult brain,” Nature, vol. 634, pp. 124–138, 2024.

[4] The MICrONS Consortium, “Functional connectomics spanning multiple areas of mouse visual cortex,” Nature, vol. 640, pp. 435–447, 2025.

[5] Nature Methods, “Method of the Year 2025: electron microscopy-based connectomics,” Nature Methods, vol. 22, no. 12, 2025.

[6] Jan Funke, Fabian Tschopp, William Grisaitis, et al., “Large scale image segmentation with structured loss based deep learning for connectome reconstruction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 7, pp. 1669–1680, 2019.

[7] Kisuk Lee, Jonathan Zung, Peter Li, Viren Jain, and H. Sebastian Seung, “Superhuman accuracy on the SNEMI3D connectomics challenge,” arXiv preprint arXiv:1706.00120, 2017.

[8] Larissa Heinrich, Jan Funke, Constantin Pape, Juan Nunez-Iglesias, and Stephan Saalfeld, “Synaptic cleft segmentation in non-isotropic volume electron microscopy of the complete Drosophila brain,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2018. 2018, vol. 11071 of Lecture Notes in Computer Science, pp. 317–325, Springer.

[9] Julia Buhmann, Arlo Sheridan, Caroline Malin-Mayor, et al., “Automatic detection of synaptic partners in a whole-brain Drosophila electron microscopy data set,” Nature Methods, vol. 18, pp. 771–774, 2021.

[10] Philipp Schlegel, Yijie Yin, Alexander S. Bates, Sven Dorkenwald, Katharina Eichler, Paul Brooks, et al., “Whole-brain annotation and multi-connectome cell typing of Drosophila,” Nature, vol. 634, pp. 139–152, 2024.

[11] Alejandro Lozano, Jeffrey Nirschl, James Burgess, Sanket Rajan Gupte, Yuhui Zhang, Alyssa Unell, and Serena Yeung-Levy, “Micro-bench: A microscopy benchmark for visionlanguage understanding,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2024.

[12] Yicong Li, Wanhua Li, Qi Chen, Wei Huang, Yuda Zou, Xin Xiao, Kazunori Shinomiya, Pat Gunn, Nishika Gupta, Alexey Polilov, et al., “Waspsyn: A challenge for domain adaptive synapse detection in microwasp brain connectomes,” IEEE transactions on medical imaging, vol. 43, no. 11, pp. 3719– 3730, 2024.

[13] Nicholas Turner, Kisuk Lee, Ran Lu, Jingpeng Wu, Dodam Ih, and H. Sebastian Seung, “Synaptic partner assignment using attentional voxel association networks,” in IEEE International Symposium on Biomedical Imaging (ISBI), 2020.

[14] Daniel Haehn, Verena Kaynig, James Tompkin, Jeff W. Lichtman, and Hanspeter Pfister, “Guided proofreading of automatic segmentations for connectomics,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 9319–9328.

[15] Jeff Brown, Andrew Kirjner, Annika Vivekananthan, and Ed Boyden, “ConnectomeBench: Can LLMs proofread the connectome?,” in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2025.

[16] Jeff Brown, Tim Farkas, Gleb Razgar, and Edward S. Boyden, “ConnectomeBench2: A unified benchmark for automated connectomic proofreading,” arXiv preprint arXiv:2606.21116, 2026.

[17] Jan Funke, Stephan Saalfeld, Davi D. Bock, Srinivas C. Turaga, and Eric Perlman, “CREMI: MICCAI challenge on circuit reconstruction from electron microscopy images,” 2016.

[18] Shuai Bai, Keqin Chen, Xuejing Liu, et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[19] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[20] Qwen Team, “Qwen3.5: Accelerating productivity with native multimodal agents,” Qwen blog, https://qwen.ai/ blog?id=qwen3.5, 2026.

[21] Jinguo Zhu, Weiyun Wang, Zhe Chen, et al., “InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models,” arXiv preprint arXiv:2504.10479, 2025.

[22] Gemma Team, “Gemma 4 technical report,” arXiv preprint arXiv:2607.02770, 2026.

[23] Edward J. Hu, Yelong Shen, Phillip Wallis, et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.

[25] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie, “A ConvNet for the 2020s,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[26] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, et al.,´ “DINOv2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.