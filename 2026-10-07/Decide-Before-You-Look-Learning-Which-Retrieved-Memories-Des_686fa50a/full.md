# Decide Before You Look: Learning Which Retrieved Memories Deserve Pixels

Youxing LI The Chinese University of Hong Kong

rickliyouxing1103@gmail.com

![](images/38164566f97fc2d23e3632df9afb213ee01279db460ec01a9a34edb385ec6770.jpg)

![](images/859ef76ac5af82c9656c2c01c55842c285c5d70ede875dcc663bc9456f59eec0.jpg)  
Figure 1: Left: the answer is visible only in the photo, and PixelTriage opens the right one at 10% of the visual tokens. Right: accuracy relative to opening all retrieved images at budgets k=1, 2, 3 (7B answering model).

## Abstract

Multimodal assistants answer questions from long-term memories that contain images. After retrieval, each retrieved image reaches the answering model either as pixels, at about a thousand visual tokens per image, or as a stored text proxy that often misses the detail the question asks about. We find that the benefit of pixels usually comes from one or two retrieved memories, and that it can be predicted before the answering model runs, without reading any full-resolution image. In PixelTriage, a plug-in placed after retrieval, a small model that does not generate text reads the dialogue, a short note and a thumbnail of each retrieved memory and predicts how much its pixels would add. It is trained on synthetic memory episodes labeled by a frozen 27B model that answers each question with and without each memory’s pixels. With a 7B answering model, PixelTriage lies on the accuracy–cost frontier of M<sup>3</sup>Exam, DMV and MemEye and uses 11–23% of the visual tokens without a significant loss of accuracy. On DMV it answers 2.9 times faster than opening all images. It outperforms retrieval order and uniform down-sizing at equal budgets and transfers to other memory systems and to a 397B answering model.

## 1 Introduction

Assistants built on multimodal large language models keep long-term memories of what their users share, such as photos, screenshots, product pages and documents [3, 9, 13, 20]. A memory system stores each image as pixels and as a short text proxy, for example a caption. After retrieving a few memories for a question, it must decide for each retrieved image whether the answering model sees the pixels or only the text.

Both defaults are costly. Ten retrieved images take about 12k visual tokens on DMV-Bench, which slows every answer, and the cost grows with retrieval depth. Text proxies drop details that nobody wrote down, like the orchid stem in Fig. 1. On DMV, answering from proxies alone loses 10.3 accuracy points with a 7B answering model and 22.4 points with a 397B model, and recent work finds that delivery can matter more than retrieval [17].

Existing systems control this choice either with fixed rules or with a generative model. Rules are cheap but choose for the wrong reason. Retrieval order measures how related a mem ory is to the question, which is not what its proxy is missing, and down-sizing every image blurs the detail in question. Asking the answering model to pick images is flexible but adds a generative call to every question before answering begins. Methods that inspect the evidence first, by scoring full-resolution images with a surrogate model [21] or by escalating to images after a text-only answer [13], work at a later stage and can run after an earlier, cheaper decision.

We view delivery through the dual-process distinction between fast, automatic decisions and slow, deliberate reasoning [18]. Choosing which retrieved memories to open is a structured, high-frequency decision with a bounded output, a probability per memory. It needs semantic understanding of the question and the memories but no text generation, so it fits a fast System One. Composing the answer from the delivered evidence remains the work of a System Two, the answering model, which then runs once. Decision models that output typed probabilities instead of text [29] make such a System One practical, and two observations tell us what it should predict and how to supervise it without annotation.

First, the benefit of pixels is concentrated: in our development labels, 87% of the questions where pixels help have at most two of 20 candidates with more than half of the largest gain, and on three benchmarks one image picked by our model recovers most of the gain of opening all retrieved images (Fig. 5). System One therefore only needs to find one or two memories. Second, the gain of a memory’s pixels can be measured directly, by swapping its proxy for its pixels in front of a frozen answering model, so training labels need no annotation.

PixelTriage (Fig. 2) implements this as a plug-in between retrieval and answering. Its System One, a 2B visionlanguage model, reads the dialogue text, a short note and a 336-pixel thumbnail of each retrieved memory and predicts a pixel gain per memory in one forward pass. We train it on 9.8k synthetic multi-session episodes built from singleimage VQA and labeled by a frozen Qwen3.5-27B (Section 5). Among 27 delivery policies on M<sup>3</sup>Exam, DMV and MemEye, PixelTriage lies on the accuracy–cost frontier of every benchmark. On M<sup>3</sup>Exam and MemEye, one opened image is more accurate than opening all images at 23% and 11% of the visual tokens, and on DMV PixelTriage answers 2.9 times faster than opening all ten images, including its own cost.

Our contributions are as follows.

• We cast the delivery of retrieved images as a pixelbudget decision made before answering and show that the benefit of pixels concentrates in one or two memories.

• We introduce PixelTriage, a non-generative System One that decides delivery without reading full-resolution images, trained with counterfactual labels from a frozen answering model on memory episodes built from singleimage VQA.

• With a 7B answering model, PixelTriage cuts visual tokens to 11–23% without a significant accuracy loss and outperforms retrieval order, self-selection and downsizing at equal budgets. It transfers to two other memory systems and to a 397B answering model.

## 2 Related Work

Multimodal memory systems and benchmarks. Multimodal agents store images with the dialogue and retrieve them later. MuRAG [6] retrieves image–text memories for generation, UniversalRAG [31] routes retrieval across corpora of different modalities and granularities, V-Mem [19] routes retrieval by the modality of the target evidence, and M3-Agent [20] builds long-term memory from continuous multimodal input. Benchmarks such as Mem-Gallery [3], MemEye [9], M<sup>3</sup>Exam [13], DMV-Bench [27] and SMM-Bench [4] test whether agents recall visual details from long histories. PixelTriage takes the retrieved set of such a system as given and decides only in which form each retrieved image reaches the answering model.

Delivery and selection of retrieved visual evidence. DeliverMem [17] separates delivery from retrieval in multimodal memory and finds that delivering pixels matters more than improving retrieval. It delivers the pixels of every retrieved memory. Utility-oriented evidence selection [21] argues that the utility of visual evidence differs from its relevance and estimates it with a surrogate multimodal model that reads each candidate image at full resolution, and drops the images it does not select. M3Proctor [13] answers from text surrogates first and escalates to raw images when its confidence is low, deciding per question with additional answering calls. Submodular selection [25] chooses image subsets for multiimage question answering by relevance. VisRAG [32] and MRAG-Bench [12] show that pixel evidence can be more useful than its textual description. These methods refine the evidence after reading full-resolution images or after a first answer. PixelTriage decides earlier, per memory and before the answering model runs, from a thumbnail and a short note of each memory instead of the full-resolution image. It keeps the text proxy of every unopened memory and learns its decision from counterfactual responses. The two stages are complementary, and PixelTriage can run first to limit which images the later methods read.

Visual token reduction. FastV [5] and SparseVLM [35] prune visual tokens inside the model, RUTA [36] allocates token budgets across image–query pairs with a rate–utility objective, and consequence-sensitive compression [30] adapts the compression rate of each image to the question. They shrink every delivered image, whereas PixelTriage chooses which retrieved memories receive pixels and leaves the rest as text. The two can be combined, and the matched-budget down-sizing baseline in Section 6.2 compares the two allocations at equal token counts.

Adaptive retrieval and decision models. Self-RAG [1] and Adaptive-RAG [15] decide whether and how to retrieve for each query. Non-generative decision models that separate typed decisions from slow generation have been used to control the construction and retrieval of text memories [16], and Valen [29] extends such decision models to visual input and provides our backbone and initialization. We apply this separation at a different step, after retrieval and before answering, and train the decision on counterfactual responses of a frozen answering model, which general decision data does not provide (Section 7.2).

## 3 Problem Setting

Delivery after retrieval. A host memory system stores every image memory $m _ { i }$ in two forms: its pixels $x _ { i }$ and a text proxy $t _ { i } ,$ such as a caption or a note written when the image was saved, together with the dialogue text $u _ { i }$ of the turn in which it was shared. Given a question $q ,$ the host retrieves n memories $R ( q ) = \{ m _ { 1 } , . . . , m _ { n } \}$ . Before the answering model runs, a delivery $d \in \{ 0 , 1 \} ^ { n }$ fixes the form of every retrieved memory: if $d _ { i } ~ = ~ 1$ , memory i enters the prompt with its pixels, placed next to its text proxy unless the host’s prompt format replaces the proxy, and otherwise it enters as its text proxy alone. The answering model, which we call System Two, is then called once on $( q , R ( q ) , d )$ . We write acc(d) for the judged accuracy of its answer and define the visual cost of a delivery as

$$
c ( d ) = \sum _ { i = 1 } ^ { n } d _ { i } \tau ( x _ { i } ) ,\tag{1}
$$

where $\tau ( x _ { i } )$ is the number of visual tokens System Two spends on image $x _ { i }$ . Opening all images $( d = { \bf 1 } )$ and opening none $( d = { \bf 0 } )$ are the two defaults, and a pixel budget k restricts a delivery to $\| d \| _ { 1 } \leq k$

Evaluation by the accuracy–cost frontier. A delivery policy maps each question to a delivery. We summarize a policy on a benchmark by its mean visual cost, reported relative to opening all images, and its mean accuracy. Policy A dominates policy B if A costs no more and is at least as accurate, with one of the two strict. The undominated policies form the accuracy–cost frontier. The cost counts the visual tokens of the answering call. Extra model calls that a policy makes to choose its images, and the compute of our decision model, are reported separately (Section 6.4).

Counterfactual pixel gain. The quantity a delivery decision needs is how much one memory’s pixels add over its text proxy for the question at hand. Let $A ( d ) \in [ 0 , 1 ]$ be the probability that a fixed answering model answers q correctly under delivery $d ,$ and let $e _ { i }$ open only memory i. The counterfactual pixel gain of memory i is

$$
g _ { i } = A ( e _ { i } ) - A ( \mathbf { 0 } ) .\tag{2}
$$

It depends on what the proxy $t _ { i }$ omits and on what $q$ asks, so two memories that are equally relevant to q can have very different gains. At the question level, pixels are needed when the text-only delivery fails and opening the right memory succeeds.

The decision problem. System One predicts the gains and the need for pixels from the dialogue text, a short note and a thumbnail of each memory, never from the full-resolution image, and a policy turns the predictions into d without calling System Two.

![](images/e713bf8d5c3f835fbd1e124f12ed3a5e083aee16b5218429f149c74a5adfea58.jpg)  
Figure 2: PixelTriage. System One scores the retrieved memories in one forward pass, the top-k are delivered as pixels, and System Two answers once. Offline, a frozen System Two provides the training labels.

## 4 PixelTriage: A Delivery Plug-in

PixelTriage sits between the retrieval of a host memory system and its answering model (Fig. 2). For every question it decides which retrieved memories reach the answering model as pixels and which as text, using a System One that predicts the pixel gain of each memory from cheap views of it. Section 5 describes how System One is trained.

## 4.1 Interface with the host

From the host, PixelTriage takes the question, the retrieved memories and, for every retrieved image, its pixels, its stored text proxy and the dialogue turn in which it was shared. It returns a delivery d (Section 3). The host’s storage, index, retrieval, prompt template and answering model stay unchanged.

The only addition is a write-time cache. When the host stores an image, PixelTriage keeps a 336-pixel thumbnail and a short note keyed by the image hash. The note is written by a frozen Qwen3.5-27B [24] without seeing any question and keeps a short label and the first sentence of a caption, about 15 words. The cache is built once per image, at 2.4 to 3.4 s per image in our measurements (Section D), and a host that already stores such notes only adds the thumbnails. At answering time System One reads this cache and never the full-resolution image.

## 4.2 System One: one forward pass over the retrieved set

System One outputs a question-level probability $p _ { \mathrm { v i s } }$ that at least one retrieved image must be opened, and a per-memory gain $\hat { g } _ { i }$ that estimates the counterfactual pixel gain of Eq. (2) on a [0, 1] scale. The delivery policy ranks memories by ${ \hat { g } } _ { i } .$ A per-memory relevance $\boldsymbol { { \hat { r } } } _ { i }$ , the probability that memory i is the one the question refers to, is trained as an auxiliary output and is not used at test time.

Backbone and decision head. We build System One on Qwen3.5-2B [24] with the non-generative decision head of Valen [29] in place of the language-model head. A decision position $h _ { \mathrm { d } }$ is scored against candidate positions $h _ { j }$ with a

bilinear form,

$$
\ell _ { j } = \frac { ( W _ { \mathrm { d } } h _ { \mathrm { d } } ) ^ { \top } ( W _ { \mathrm { c } } h _ { j } ) } { \sqrt { D } } ,\tag{3}
$$

with projections of dimension $D = 2 5 6$ . The candidates of the question-level decision are the answers true andfalse.

One state, all memories. The input lists the question and then, for each retrieved memory, its session date, the dialogue text in which it was shared, its note and its thumbnail. Each memory ends with a marker token whose hidden state $h _ { i }$ represents it. The decision questions follow the memories, so one forward pass yields every output. Each permemory question also contains a fixed candidate, none, and every memory marker is scored against it,

$$
\begin{array} { r } { \hat { g } _ { i } = \sigma ( \ell _ { i } - \ell _ { \mathrm { n o n e } } ) . } \end{array}\tag{4}
$$

A memory therefore receives a high gain only when it outscores none, independently of the other memories, which makes gains comparable across memories and questions. On the benchmarks a state has 2.2k to 2.4k tokens.

## 4.3 Delivery decision

After retrieval, System One runs once on the question and the cached views of the retrieved memories. The policy $\pi _ { k }$ marks the k memories with the highest $\hat { g } _ { i }$ for pixels and the others for text, and opens every image when fewer than k exist. The budget k is the operating knob between accuracy and visual cost and needs no retraining. A gated variant also uses $p _ { \mathrm { v i s } } \colon$ it opens images only for questions whose $p _ { \mathrm { v i s } }$ ranks among a chosen fraction of recent questions, which turns $p _ { \mathrm { v i s } }$ into a question-level budget (Section 7.4). Neither variant calls the answering model to decide, and the answering model then runs once on the delivered evidence. A forward pass of System One costs about 10 TFLOP, while the prefill of an answering call that reads all retrieved images costs 97 to 670 TFLOP in our settings (Section 6.4).

## 4.4 Plugging into different hosts

PixelTriage applies d in the host’s own prompt format. We consider three cases, all evaluated in Section 6.

Hosts that deliver pixels. Multimodal memory systems such as MuRAG [6] hand every retrieved image to the answering model. PixelTriage keeps the pixels of the selected memories and lets the others fall back to their text proxies, which in MuRAG’s prompt are the image identifier and caption. Whether an opened image keeps its caption or replaces it follows the host’s format. Hosts that route retrieval across modalities, such as UniversalRAG [31], are handled in the same way on the images they return.

Text-only hosts. A text memory system never shows pixels. PixelTriage leaves its retrieved text and prompt unchanged and appends the pixels of the selected memories, which gives the system visual evidence at the cost of k images.

Other answering models. System One is trained once, on labels from a 27B answering model, and is used unchanged with other answering models, such as the 7B and 397B models of Section 6.3. Only the host’s answering call changes.

## 5 Training System One Without Annotation

System One predicts counterfactual quantities, the gain of a memory’s pixels over its proxy and whether a question needs pixels at all. No annotation of these exists, and memory benchmarks are small and reserved for testing. We therefore build training data from single-image visual question answering and label it with a frozen answering model in five steps (Fig. 3). Steps 1 to 4 turn a VQA question into a memory episode (Section 5.1), step 5 labels every memory (Section 5.2), and Section 5.3 trains System One.

## 5.1 Memory episodes from single-image VQA

Step 1: source questions. We start from the singleimage multiple-choice questions of Valen-General-100k [29], taking training questions from GQA [14], VQAv2 [8], TextVQA [26] and DocVQA [23]. ChartQA [22] and ScreenQA [10] are held out as a cross-source development set, and a hashed slice of training-source images forms an insource development set. Splits are made at the image level, and every distractor memory comes from the same split as its question.

Step 2: write-time notes. As at deployment (Section 4.1), the frozen Qwen3.5-27B [24] writes for every image, without seeing any question, a label of at most five words, a twoto-four-sentence caption and an OCR transcript. The stored text proxy keeps only the label and the first caption sentence, about 15 words, so that fine detail remains in the pixels. Each image also gets a 336-pixel thumbnail, a SigLIP 2 [28] image embedding and a Qwen3 [34] embedding of its caption, which step 3 uses to find neighbors.

Step 3: memory episodes. Each query receives 20 candi date memories spread over four to eight dated sessions. Besides the image the question is about, there are eight nearest visual neighbors, six nearest caption neighbors and five random images, with near-duplicates (cosine similarity above 0.97) excluded. Every memory is a dated user turn that shares its image.

Step 4: recall questions and options. The original question is rewritten as a recall question that points to the memory by its session date (40% of queries), by a content phrase written by the 27B model from the label and caption without revealing the answer (40%), or through another memory shared in the same session (20%, two hops). The 27B model also replaces the original distractor options, which often differ in type from the answer, by three options of the answer’s type and format. In 25% of queries one option states that no such image was shared, an answer that lossy proxies often lead answering models to choose. As a leakage check, the 27B model answers from the question and options alone: the correct option receives probability 0.37 on average, against 0.25 for chance. The result is 9,824 training queries and 2,470 development queries (1,474 in-source, 996 cross-source).

![](images/ccb6c6984ccd3842522089a73d4b212710608bae97dd92588abe6b1b2bd49d40.jpg)  
Figure 3: The training data pipeline on one training query. A frozen 27B model writes the notes and options and labels each memory by the change in the probability of the correct answer when that memory’s pixels are added.

## 5.2 Counterfactual labels

In step 5, the frozen 27B model answers each query under controlled contexts. In the reference context T every candidate appears through its text proxy, and $T + x _ { i }$ adds the pixels of candidate i, once for each candidate. As in Eq. (2), A(·) is the probability of the correct answer, here read as the probability of the correct option letter, renormalized over the four letters, with the assistant turn prefilled by “Answer:” so that the next token is a letter. The soft targets are

$$
g _ { i } = \operatorname* { m i n } \bigl ( 1 , \operatorname* { m a x } ( 0 , A ( T + x _ { i } ) - A ( T ) ) \bigr ) ,\tag{5}
$$

$$
p _ { \mathrm { v i s } } = \left( 1 - A ( T ) \right) A ( T + x _ { \mathrm { r e f } } ) ,\tag{6}
$$

$$
r _ { i } = \mathbf { 1 } [ i \mathrm { ~ i s ~ t h e ~ r e f e r e n c e d ~ m e m o r y } ] ,\tag{7}
$$

where $x _ { \mathrm { r e f } }$ is the image of the referenced memory. g<sub>i</sub> is the expected accuracy gain from opening candidate i, and $p _ { \mathrm { v i s } }$ is the probability that the proxy-only answer is wrong while the answer with the referenced image is right. Hard labels remeasured under a permuted option order agree in only about 82% of cases, so we train on soft targets, and development labels average two option orders. Labeling takes about 26 answering calls per query. On the training set the referenced image raises A from 0.47 to 0.81 on average, 39.5% of the queries need pixels under the hard criterion, and a distractor’s pixels raise the answer probability by more than 0.1 in 13.9% of cases and lower it by more than 0.1 in 11.8%.

## 5.3 Objective and training

Each query becomes one state with the three decision questions. The per-memory outputs use binary cross-entropy against the soft targets and $p _ { \mathrm { v i s } }$ uses Valen’s decision loss. System One starts from a Valen decision model fine-tuned on Valen-General-100k (our reproduction) and is trained for two epochs with LoRA [11] (rank 32) on all linear layers of the language model, together with the vision merger and the decision head. On the development set, $p _ { \mathrm { v i s } }$ reaches an AUROC of 0.821 with an expected calibration error of 0.048, and the memory with the highest $\hat { g } _ { i }$ is the referenced one in 87.8% of queries.

## 6 Experiments

## 6.1 Setup

Plug-in protocol. All experiments run inside the Mem-Gallery evaluation harness [3]. The host is MuRAG [6] with GME-Qwen2-VL-2B [33] retrieval of the top 10 memories, plugged in as described in Section 4.4, and the answering model is Qwen2.5-VL-7B [2], called once with the harness prompt. The benchmark caption of each image is the host’s text proxy. An opened image is placed next to its caption, except on M<sup>3</sup>Exam, whose captions are mostly empty, where we keep the harness default of replacing the caption. Other caption and delivery conditions are in Section E. System One was trained only on the synthetic episodes of Section 5.

Benchmarks. We use benchmarks in which visual detail matters, selected by a criterion fixed before running Pixel-Triage: opening all retrieved images must improve accuracy over text proxies by at least 5 points. M<sup>3</sup>Exam [13] (654 open-ended questions about two public user personas, 4.2 retrieved images per question), DMV-Bench [27] (1,000 recall probes over 200 replayed shopping chains, with the queried cue only in a product photo, 10 retrieved images) and Mem-Eye [9] (371 open-ended questions, 8.8 retrieved images) pass with gains of 7.5, 10.3 and 11.1 points. DMV-Bench is interactive in its original form. We replay its browsing chains as static dialogue, so our numbers are not comparable to the original leaderboard. Mem-Gallery [3] (1.7 points) and SMMBench [4] (at most 3.7) do not pass and are analyzed as boundary cases (Section 7.5).

Scoring. M<sup>3</sup>Exam answers are judged by Qwen2.5-VL-32B with a five-level rubric, and DMV and MemEye answers by Qwen2.5-72B with the Mem-Gallery rubric (0, 0.5 or 1), which reproduces the official harness scores on 99.6% of questions. Scores are percentages. Visual cost counts Qwen2.5-VL tokens of the opened images (28-pixel patches) relative to opening all retrieved images. Paired bootstrap intervals over questions [7] for all comparisons are in Section A.

Policies at budget k. Retrieval order opens the top-k retrieved images, and random opens k at random. The general decision model is the Valen checkpoint from which System One is initialized, given the same input and asked which images to open, which isolates the effect of our supervision. Two self-selection baselines ask the answering model to choose up to k memories in an extra call, from the captions alone or from the captions plus the same thumbnails System One reads. The second one is a training-free selector with exactly System One’s input. Down-sizing opens all retrieved images, each resized so that their total equals the tokens PixelTriage spends at budget k for that question.

## 6.2 PixelTriage lies on the accuracy–cost frontier

Fig. 4 places all 27 evaluated policies by mean visual cost and accuracy. On every benchmark the frontier contains PixelTriage configurations, and all 27 configurations of retrieval order, random selection and the general decision model are dominated. On M<sup>3</sup>Exam, the single image chosen by PixelTriage reaches 75.73 at 23% of the visual tokens, above opening all images (75.08), and on MemEye it reaches 49.06 at 11%, against 48.52, so on both it dominates opening all images. On DMV, opening all images remains the most accurate policy (64.75). PixelTriage with two images reaches 63.35 at 20% of the tokens, 98% of that accuracy, and the difference is not significant, whereas one image is 2.45 points lower. The other frontier points are variants of PixelTriage, self-selection with its extra call, and down-sizing on DMV, which is 0.1% cheaper than PixelTriage and 2.3 points less accurate.

Table 1 compares the policies at equal budgets. PixelTriage is the most accurate budgeted policy in all nine columns. Its margin over retrieval order is largest on DMV, 4.2 points at k=1, where similar products make the top retrieved image a poor guess. Self-selection by the answering model trails even when it sees the same thumbnails as System One, by 3.5 points on DMV at k=1, although it calls a model three times larger.

Choosing images versus down-sizing all of them. Downsizing keeps every retrieved image visible at the same token budget. PixelTriage outperforms it in all nine budget settings, most clearly at the smallest budget, by 3.1 points on M<sup>3</sup>Exam and 2.3 on DMV at k=1, where down-sizing loses 2.5 and 4.7 points to opening all images. As the budget grows, each down-sized image keeps more detail and the gap narrows, to less than half a point on M<sup>3</sup>Exam and DMV at k=3. A question usually asks about a detail in one image, and spreading the tokens over all images blurs that detail. Opening PixelTriage’s top image at full resolution and the others downsized performs on par with PixelTriage at about twice the cost.

## 6.3 Transfer across memory systems and answering models

Memory systems. We plug PixelTriage into Universal-RAG [31], which routes retrieval across modalities, and into NaiveRAG, a text-only memory system to which the plug-in adds pixels (Section 4.4 and Table 2). With both systems, PixelTriage’s single image outperforms the top retrieved image by 3.6 to 4.5 points on DMV and MemEye. For the textonly system, one image chosen by PixelTriage raises DMV accuracy from 55.05 to 66.65, close to adding all ten images (68.10), so the plug-in also gives a text memory system access to visual evidence at a tenth of the visual cost. On M<sup>3</sup>Exam neither system passes the 5-point criterion (gains of 2.5 and 1.8 points), and there PixelTriage’s single image stays within 0.3 points of opening all images.

A larger answering model. With Qwen3.5-397B answering and System One unchanged, opening images matters more: all images add 12.8 points on M<sup>3</sup>Exam and 22.4 on DMV. PixelTriage’s image outperforms the top retrieved image by 1.6 points on M<sup>3</sup>Exam and 6.5 on DMV, and a random image by 6.2 and 11.8 points (Fig. 5), so labels from a 27B model transfer to a stronger answering model. One image, however, no longer matches opening all images (1.4 and 6.7 points lower). Budgets above one image were not evaluated with this model.

## 6.4 Cost

System One is a 2.25B-parameter model that runs one forward pass of about 10 TFLOP per question. It adds 0.32 to 0.34 s in the measurement of Table 3 and 0.17 to 0.23 s after warm-up. Including this overhead, answering a DMV question with one opened image is 2.9 times faster than opening all ten with the 7B answering model and 2.6 times faster with the 27B model, and total prefill compute falls to 27–28% (Table 3). On M<sup>3</sup>Exam, where only 4.2 images are retrieved on average, compute falls to 55–60% but the latency difference is not significant. Retrieval order needs no decision model and answers faster than PixelTriage (0.39 against 0.71 s on DMV with the 7B model) at 4.15 points lower accuracy. Writing the note of an image takes 2.4 to 3.4 s once, at storage time (Section D).

## 7 Analysis

## 7.1 Pixel gains concentrate in one or two memories

Fig. 5 measures how much of the gain of opening every retrieved image one opened image recovers. With the 7B an swering model, the image chosen by PixelTriage recovers more than the full gain on M<sup>3</sup>Exam and MemEye and 76% on DMV, while a random image recovers at most 23%. Because these shares use our own selector, they bound from be low how much of the gain one image can carry, and the gap to retrieval order (36% on DMV) shows that which image is opened decides most of the gain. DMV is the hardest case for retrieval order: its retrieved memories are similar products whose text never names the cue, so retrieval similarity says little about which photo shows it. The concentration in the development labels (Section 1) partly reflects their construction with one referenced image per query (Section 5).

![](images/3b955256421d5d06c0b3873643ec00f391fa1a4c1ab1f9a3951bc76e98d87ce0.jpg)  
Figure 4: Accuracy versus visual tokens for the evaluated delivery policies (7B answering model). Black line: accuracy–cost frontier. Self selection makes one extra call, not counted here.

Table 1: Accuracy at a budget of k opened images (7B answering model, 10 retrieved memories). <sup>†</sup>Extra answering call. Bold: best budgeted policy. Shaded : at least as accurate as opening all.
<table><tr><td></td><td colspan="3">M3Exam</td><td colspan="3">DMV</td><td colspan="3">MemEye</td></tr><tr><td>Delivery policy</td><td>k=1</td><td>k=2</td><td>k=3</td><td>k=1</td><td>k=2</td><td>k=3</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td>Text proxies only</td><td></td><td>67.58</td><td></td><td></td><td>54.45</td><td></td><td></td><td>37.47</td><td></td></tr><tr><td>Open all retrieved images</td><td></td><td>75.08</td><td></td><td></td><td>64.75</td><td></td><td></td><td>48.52</td><td></td></tr><tr><td>Random</td><td>68.96</td><td>71.06</td><td>72.36</td><td>55.90</td><td>57.85</td><td>59.60</td><td>40.03</td><td>41.64</td><td>42.59</td></tr><tr><td>Retrieval order</td><td>73.47</td><td>74.50</td><td>74.73</td><td>58.15</td><td>59.55</td><td>59.55</td><td>47.30</td><td>47.17</td><td>47.30</td></tr><tr><td>General decision model</td><td>69.30</td><td>72.13</td><td>73.47</td><td>57.00</td><td>57.40</td><td>58.85</td><td>43.94</td><td>46.77</td><td>47.98</td></tr><tr><td>Self-selection: captions†</td><td>71.41</td><td>71.83</td><td>71.64</td><td>55.55</td><td>56.80</td><td>56.75</td><td>46.90</td><td>48.52</td><td>48.11</td></tr><tr><td>Self-selection: + thumbnails† Down-size all images</td><td>73.43 72.59</td><td>74.16</td><td>73.85 75.04</td><td>58.80</td><td>60.40</td><td>61.60</td><td>47.84 47.84</td><td>47.30</td><td>47.57</td></tr><tr><td></td><td></td><td>74.96</td><td></td><td>60.05</td><td>61.50</td><td>63.00</td><td></td><td>48.79</td><td>48.52</td></tr><tr><td>PixelTriage (ours)</td><td>75.73</td><td>75.42</td><td>75.46</td><td>62.30</td><td>63.35</td><td>63.45</td><td>49.06</td><td>50.27</td><td>50.54</td></tr><tr><td>visual tokens vs. open all</td><td>23%</td><td>44%</td><td>61%</td><td>10%</td><td>20%</td><td>30%</td><td>11%</td><td>22%</td><td>33%</td></tr></table>

Table 2: Transfer to other memory systems (7B answering model) and to a 397B answering model. NaiveRAG stores text only. Bold: better of Rank@1 and Ours@1.
<table><tr><td>Setting</td><td>Benchmark</td><td>Text</td><td>Rank@1</td><td>Ours@1</td><td>All</td></tr><tr><td>UniversalRAG</td><td>DMV</td><td>19.45</td><td>20.60</td><td>25.05</td><td>25.55</td></tr><tr><td>UniversalRAG</td><td>MemEye</td><td>32.61</td><td>39.08</td><td>42.72</td><td>41.91</td></tr><tr><td>NaiveRAG</td><td>DMV</td><td>55.05</td><td>62.90</td><td>66.65</td><td>68.10</td></tr><tr><td>NaiveRAG</td><td>MemEye</td><td>31.94</td><td>36.25</td><td>40.16</td><td>41.51</td></tr><tr><td>Qwen3.5-397B</td><td>M³Exam</td><td>67.2</td><td>77.1</td><td>78.6</td><td>80.0</td></tr><tr><td>Qwen3.5-397B</td><td>DMV</td><td>62.2</td><td>71.5</td><td>77.9</td><td>84.6</td></tr></table>

![](images/fc6b871e49b58f949988356b87244587f574506833b33607e31bc7833206f6b0.jpg)  
Figure 5: Share of the all-image gain recovered by opening one image.

## 7.2 The ability comes from counterfactual supervision

The general decision model has the same backbone, decision head and input as System One and was trained on general decision data only. On the synthetic development set its question-level AUROC is 0.46, close to chance, and opening its top image yields an expected answering accuracy of 52.7, against 78.8 for System One. On the benchmarks it trails PixelTriage by 5.1 to 6.4 points at k=1 and falls below retrieval order (Table 1). The choice of which memory deserves pixels comes from the counterfactual labels. The auxiliary relevance output helps this learning: removing it lowers the development top-1 hit rate of gˆ from 88.4% to 80.3% (Section C).

Table 3: Online cost per question at k=1, including System One, over 60 questions per benchmark without prefix caching. The time ratio is open all over ours, and compute is prefill FLOPs of ours over open all, extrapolated for 397B.
<table><tr><td></td><td>Answering</td><td>Open all</td><td>Ours@1</td><td>Ratio</td></tr><tr><td rowspan="2">DMV time (s)</td><td>7B</td><td>2.05</td><td>0.71</td><td>2.9×</td></tr><tr><td>27B</td><td>2.77</td><td>1.07</td><td>2.6×</td></tr><tr><td rowspan="2">M³Exam time (s)</td><td>7B</td><td>0.99</td><td>0.90</td><td>1.1×</td></tr><tr><td>27B</td><td>2.62</td><td>2.41</td><td>1.1×</td></tr><tr><td rowspan="3">DMV TFLOP</td><td>7B</td><td>238</td><td>65</td><td>27%</td></tr><tr><td>27B</td><td>670</td><td>179</td><td>27%</td></tr><tr><td>397B-A17B</td><td>422</td><td>117</td><td>28%</td></tr><tr><td rowspan="3">M³Exam TFLOP</td><td>7B</td><td>97</td><td>58</td><td>60%</td></tr><tr><td>27B</td><td>296</td><td>162</td><td>55%</td></tr><tr><td>397B-A17B</td><td>186</td><td>106</td><td>57%</td></tr></table>

Table 4: Input ablation on the development set (mean ± s.d. over three training seeds). Ref. helps: AUROC of gˆ for whether opening the referenced image helps.
<table><tr><td>System One input pvis AUROC</td><td></td><td>Top-1 is ref. (%)</td><td>Ref. helps</td></tr><tr><td>Memory headers</td><td>0.718±.003</td><td>41.1±0.4</td><td>0.666±.005</td></tr><tr><td>+ text notes</td><td>0.809±.005</td><td>75.4±0.8</td><td>0.782±.002</td></tr><tr><td>+ thumbnails</td><td> $\mathbf { 0 . 8 3 5 \bot . 0 0 1 }$ </td><td> ${ \bf 8 6 . 7 \pm } 3 . 4 $ </td><td> $\mathbf { 0 . 8 4 7 \pm . 0 0 5 }$ </td></tr></table>

## 7.3 What System One needs to see

Table 4 trains System One with three inputs, three seeds each. With only the memory headers (session date and dialogue text), the memory with the highest gˆ is the referenced one in 41% of development queries. Adding the text notes raises this to 75%, and adding the thumbnails to 87%. The thumbnails also help beyond identifying the memory. Among referenced images, the AUROC of gˆ for whether opening the image helps rises from 0.78 with notes to 0.85 with thumbnails. This ablation was run on an earlier version of the episodes, in which questions refer to memories by date only.

## 7.4 Deciding whether to look

The question-level probability $p _ { \mathrm { v i s } }$ decides whether any image is opened. We use it as a question-level budget: Pixel-Triage’s top image is opened only for the fraction of questions with the highest $p _ { \mathrm { v i s } }$ , and the rest are answered from text proxies. The threshold is a quantile of $p _ { \mathrm { v i s } }$ and needs no answers. Estimating it on a disjoint half of the questions, or on only 50 unlabeled questions, changes the average accuracy by at most 0.4 points compared with estimating it on all test questions (Section B). On M<sup>3</sup>Exam, where text proxies already answer most questions, opening images for 30% of the questions reaches 75.12 with 6% of the visual tokens, the accuracy of opening all images. Choosing the same share of questions at random gives 70.03, and opening for 50% reaches 75.69 with 11% of the tokens. On DMV and Mem Eye most questions need pixels and a question-level gate has little to skip. At a 30% opening rate it is 0.35 and 1.05 points above random gating and 7.6 and 6.5 points below opening all images. A probability threshold fixed on the synthetic development set does not transfer across benchmarks (it opens 25%, 77% and 83% of the questions), so the gate is set by its opening rate.

## 7.5 When PixelTriage does not help

When text suffices. On Mem-Gallery the benchmark captions already carry the needed detail. Opening all images adds 1.7 points, and PixelTriage’s single image is 1.05 points below the top retrieved image. SMMBench shows at most 3.7 points of headroom. A delivery decision has little to gain in such settings, and comparing opening all images with opening none on a sample of questions identifies such settings.

When one image is not enough. Matching the accuracy of opening all images also requires informative proxies for the unopened memories. On MemEye with our short notes or no captions, one opened image is 4.45 and 4.85 points below opening all images, although still above the top retrieved image. On DMV, replacing an opened image’s caption by its pixels deletes text that often names the cue, and one opened image then scores 53.1, below text only (54.4), with retrieval order dropping in the same way (Section E). Among DMV questions that text alone gets wrong, PixelTriage’s image is fully correct where the top retrieved image is not in 39 cases, and the reverse holds in 10. A typical loss follows a word in a proxy, such as a wind chime with “blue chimes” opened for a blue orchid stem.

## 8 Conclusion

A small model that never reads a full-resolution image can decide, before the answering model runs, which retrieved mem ories deserve pixels. Because the benefit of pixels concentrates in one or two memories, opening the right ones keeps the accuracy of opening all images at a fraction of the visual tokens, and a frozen answering model supplies the counterfactual supervision needed to find them. The decisions transfer across memory systems and to a larger answering model, for which one image no longer matches opening all images. The same kind of decision could be made for other modalities that have a cheap proxy and an expensive raw form, such as audio and video.

## References

[1] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, gen-

erate, and critique through Self-Reflection. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.11511.

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.

[3] Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann, Jingrui He, and Hanghang Tong. Mem-Gallery: Benchmarking multimodal Long-Term conversational memory for MLLM agents. arXiv preprint arXiv:2601.03515, 2026.

[4] Huacan Chai, Yukai Wang, Yingxuan Yang, Dan Peng, Yuanyi Song, Zhihui Fu, Weiwen Liu, Jianghao Lin, Jun Wang, and Weinan Zhang. SMMBench: A benchmark for Source-Distributed multimodal agent memory. arXiv preprint arXiv:2605.15710, 2026.

[5] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-Play inference acceleration for large Vision-Language models. In European Conference on Computer Vision (ECCV), 2024. arXiv:2403.06764.

[6] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. MuRAG: Multimodal Retrieval-Augmented generator for open question answering over images and text. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022. arXiv:2210.02928.

[7] Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap. Chapman and Hall/CRC, 1993.

[8] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. arXiv:1612.00837.

[9] Minghao Guo, Qingyue Jiao, Zeru Shi, Yihao Quan, Boxuan Zhang, Danrui Li, Liwei Che, Wujiang Xu, Shilong Liu, Zirui Liu, Mubbasir Kapadia, Vladimir Pavlovic, Jiang Liu, Mengdi Wang, Yiyu Shi, Dimitris N. Metaxas, and Ruixiang Tang. MemEye: A Visual-Centric evaluation framework for multimodal agent memory. arXiv preprint arXiv:2605.15128, 2026.

[10] Yu-Chung Hsiao, Fedir Zubach, Gilles Baechler, Srinivas Sunkara, Victor Carbune, Jason Lin, Maria Wang, Yun Zhu, and Jindong Chen. ScreenQA: Large-Scale Question-Answer pairs over mobile app screenshots. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2025. arXiv:2209.08199.

[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. arXiv:2106.09685.

[12] Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu,

Kai-Wei Chang, and Nanyun Peng. MRAG-Bench: Vision-Centric evaluation for Retrieval-Augmented multimodal models. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.08182.

[13] Zhengjun Huang, Wenxuan Liu, Zhoujin Tian, Wei Chen, Junle Chen, Yuqian Wu, Fangyuan Zhang, Qintian Guo, and Xiaofang Zhou. M<sup>3</sup>exam: Benchmarking multimodal memory for realistic User-Agent interactions. arXiv preprint arXiv:2606.07402, 2026.

[14] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for Real-World visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. arXiv:1902.09506.

[15] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-RAG: Learning to adapt Retrieval-Augmented large language models through question complexity. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024. arXiv:2403.14403.

[16] Dongming Jiang, Yi Li, and Bingzhe Li. Jev-Mem: System-One-Controlled agentic memory for efficient AI agents. arXiv preprint arXiv:2609.23986, 2026.

[17] Yuhang Jiang, Qingwei Liao, Kaize Yin, Xingling Liu, Luca Cuomo, and Silvio Bacci. Retrieved but not delivered: Multimodal memory delivery for Long-Term agents. arXiv preprint arXiv:2609.32590, 2026.

[18] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011.

[19] Dingyi Kang, Dongming Jiang, Yi Li, Guanpeng Li, and Bingzhe Li. V-Mem: Modality-Routed retrieval for Long-Term multimodal agentic memory. arXiv preprint arXiv:2608.01543, 2026.

[20] Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remembering, and reasoning: A multimodal agent with Long-Term memory. arXiv preprint arXiv:2508.09736, 2025.

[21] Weiqing Luo, Zongye Hu, Xiao Wang, Zhiyuan Yu, Haofeng Zhang, and Ziyi Huang. Utility-Oriented visual evidence selection for multimodal Retrieval-Augmented generation. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2026. arXiv:2605.13277.

[22] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Associationfor Computational Linguistics (ACL), 2022. arXiv:2203.10244.

[23] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A dataset for VQA on document images. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2021. arXiv:2007.00398.

[24] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, February 2026.

[25] Aaryan Sharma, Shivansh Gupta, Samar Agarwal, Vishak Prasad C., and Ganesh Ramakrishnan. Enhanc-

ing Multi-Image question answering via submodular subset selection. arXiv preprint arXiv:2505.10533, 2025.

[26] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. arXiv:1904.08920.

[27] Yujin Tang, Chenming Shang, Ruize Xu, and Nikhil Singh. DMV-Bench: Diagnosing Long-Horizon multimodal agents visual memory with incidental cue injection. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026. arXiv:2606.27499.

[28] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual Vision-Language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[29] Valen Team. Valen. https://github.com/ Liuziyu77/Valen, 2026. Multimodal System One decision model.

[30] Jingbo Wen, Liang He, Mingyu Cao, Haoyu Wang, Minxuan Hu, Kangning Cui, and Xilu Wang. Not all visual tokens are equally safe to remove: Consequence-Sensitive visual token compression. arXiv preprint arXiv:2608.09176, 2026.

[31] Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. UniversalRAG: Retrieval-Augmented generation over corpora of diverse modalities and granularities. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2026. arXiv:2504.20734.

[32] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. VisRAG: Vision-based retrievalaugmented generation on multi-modality documents. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.10594.

[33] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. GME: Improving universal multimodal retrieval by multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2412.16855.

[34] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

[35] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. SparseVLM: Visual token sparsification for efficient Vision-Language model inference. In International Conference on Machine Learning (ICML), 2025. arXiv:2410.04417.

[36] Jian Zou, Xiaoyu Xu, Zhihua Wang, Yilin Wang, Balu Adsumilli, and Kede Ma. RUTA: Principled visual to-

ken allocation via Rate-Utility optimization. arXiv preprint arXiv:2608.04132, 2026.

## A Significance tests

Table A1 lists paired bootstrap intervals over questions for every comparison in Table 1 and for the variants on the frontier of Fig. 4. Table A2 gives the intervals for the transfer experiments of Table 2. With the 397B model, Ours@1 outperforms Random@1 by 6.2 [+4.1, +8.3] points on M<sup>3</sup>Exam and $1 1 . 8 [ + 9 . 6 , + 1 4 . 0 ]$ on DMV.

## B Question-level gating

Setting the opening rate without the test questions. The gated points in Fig. 4 open PixelTriage’s top images for the 30% of questions with the highest $p _ { \mathrm { v i s } }$ , the quantile being taken over all test questions without their answers. Table A3 estimates the quantile instead on questions that are not eval uated: on one random half, applied to the other half and vice versa (cross-fit), or on 50 random questions, applied to the rest. Accuracies are simulated from the recorded answers (an opened question takes its PixelTriage@1 answer, a closed one its text-only answer), which reproduces the recorded 30% gate within 0.1 to 0.5 points. Each estimate is repeated over 1,000 random draws. The held-out estimates match the transductive ones, and the gate clearly outperforms random questions only on M<sup>3</sup>Exam.

A fixed probability threshold. Table A4 reports the gate with the threshold fixed on the development set $( p _ { \mathrm { { v i s } } } ~ \geq$ 0.209), opening PixelTriage’s top image for the questions above the threshold. The random gate opens the top retrieved image for the same number of randomly chosen questions. The same threshold opens images for very different shares of the questions on the three benchmarks, which is why the main text reports gating by opening rate.

## C Relevance as an auxiliary output

We trained two models from the same initialization, data and seed, with and without the relevance output ${ \hat { r } } _ { i } ,$ on 32 GPUs with the same global batch. With relevance, the memory with the highest $\hat { g } _ { i }$ is the referenced one in 88.38% of development queries, against 80.28% without it (+8.10 [+6.72, +9.47]). The gap is 10.65 points in-source and 4.32 points crosssource. The AUROC of $p _ { \mathrm { v i s } }$ is unchanged (0.822 against $0 . 8 1 7 , + 0 . 0 0 5 \ [ - 0 . 0 0 2 , + 0 . 0 1 2 ] )$ ). As a plug-in on M<sup>3</sup>Exam at k=1, the model with relevance scores 75.84 and the model without it $7 5 . 3 1 \left( + 0 . 5 4 \left[ - 0 . 1 1 , + 1 . 2 6 \right] \right)$ . Removing the relevance question also changes the averaging of the per-question losses from three terms to two, so this comparison removes a task rather than isolating its loss weight.

## D Write-time cost

The plug-in caches, for every stored image, a thumbnail and a note written by Qwen3.5-27B. We re-ran note writing for all 1,231 distinct images of DMV and M<sup>3</sup>Exam, one request at a time with prefix caching disabled: 2.42 s and 91 output tokens per image on DMV (967 images) and 3.39 s and 134 output tokens on M<sup>3</sup>Exam (264 images). This cost is paid once per image, when it is stored, and online answering does not repeat it. A memory system that already stores notes of this kind adds only the thumbnails.

## E Additional conditions

Down-sizing without captions on M<sup>3</sup>Exam. On M<sup>3</sup>Exam an opened image replaces its caption, so down-sizing replaces every caption by a small image while PixelTriage keeps the captions of unopened memories. With all captions removed for both policies, PixelTriage outperforms down-sizing by 3.13 [+1.72, +4.59] points at k=1, 0.80 [−0.27, +1.87] at k=2 and 0.80 [−0.23, +1.87] at k=3, the same pattern as in Table 1 (3.13, 0.46 and 0.42).

Caption and delivery conditions. Table A5 varies what the unopened memories carry (the benchmark captions, our 27B short notes, or nothing) and whether an opened image keeps its caption (kept) or replaces it (replaced). The main results use the first row of each benchmark. PixelTriage’s image outperforms the top retrieved image in ten of the eleven conditions. The exception is DMV with replaced captions (−0.6 [−2.6, +1.4]). Whether one image matches opening all images depends on what the unopened memories keep. On DMV without any captions, one image outperforms all ten (64.15 against 59.85, +4.3 [+1.5, +7.3]). On MemEye with our short notes or no captions, one image is 4.45 and 4.85 points below opening all images, and three images close most of the gap. On DMV with replaced captions, the benchmark captions, which are not filtered for the cue, often name it. Replacing the opened image’s caption deletes that text, and both retrieval order and PixelTriage fall below text-only answering at k=1.

## F Protocol details

Benchmarks. M<sup>3</sup>Exam is converted to the Mem-Gallery dialogue format. A round that shares several images or a PDF (rendered page by page) keeps its text with the first image and adds one image-only round per further image. The file-name questions are removed because the memory view never shows file names, leaving 654 open-ended questions. DMV-Bench probes are replayed from its generator: each chain has five sessions, a session browses about twelve products of one category in 22 to 28 steps, and every view becomes one memory round with the product name, price and storefront text, which do not mention the cue, and the product photo, which shows it. The benchmark’s own captions are kept as text proxies. Answers are product names. MemEye uses its open-ended form with absolute image paths.

Judges. The Mem-Gallery rubric is the harness prompt with JSON output, rounded to 0, 0.5 or 1, and our concurrent implementation agrees with the official harness on 99.6% of questions. ${ \bf M } ^ { 3 }$ Exam uses the five-level rubric with Qwen2.5- VL-32B, the judge model of its release.

Table A1: Ours@k minus each policy, in accuracy points, with 95% paired bootstrap intervals (10,000 resamples). Bold: interval excludes zero.
<table><tr><td>Ours@k minus</td><td>k</td><td> $\mathbf { M } ^ { 3 } \mathrm { E x a m }$ </td><td>DMV</td><td>MemEye</td></tr><tr><td>Open all retrieved images</td><td>1</td><td> $+ 0 . 6 5 [ - 0 . 3 4 , + 1 . 6 8 ]$ </td><td>-2.45 [—4.55, -0.30]</td><td>+0.54 [−3.10, +4.18]</td></tr><tr><td></td><td>2</td><td>+0.34 [−0.54, +1.26]</td><td>-1.40 [-3.25, +0.40]</td><td>+1.75 [−1.48, +5.12]</td></tr><tr><td></td><td>3</td><td>+0.38 [-0.27, +1.11]</td><td>-1.30 [-3.10, +0.45]</td><td>+2.02 [−1.08, +5.12]</td></tr><tr><td>Retrieval order</td><td>1</td><td>+2.26[+1.07, +3.52]</td><td>+4.15 [+2.45, +5.90]</td><td>+1.75 [−1.21, +4.72]</td></tr><tr><td></td><td>2</td><td>+0.92 [−0.19, +2.06]</td><td>+3.80 [+2.20, +5.45]</td><td>+3.10 [+0.00, +6.33]</td></tr><tr><td></td><td>3</td><td>+0.73 [−0.19, +1.64]</td><td>+3.90 [+2.30, +5.55]</td><td>+3.23 [+0.40, +6.06]</td></tr><tr><td>Random</td><td>1</td><td>+6.77 [+4.89, +8.75]</td><td>+6.40 [+4.35, +8.50]</td><td>+9.03 [+5.39, +12.80]</td></tr><tr><td></td><td>2</td><td>+4.36 [+2.71, +6.04]</td><td>+5.50 [+3.45, +7.60]</td><td>+8.63 [+4.85, +12.40]</td></tr><tr><td></td><td>3</td><td>+3.10 [+1.80, +4.51]</td><td>+3.85 [+2.00, +5.80]</td><td>+7.95[+4.18, +11.86]</td></tr><tr><td>General decision model</td><td>1</td><td>+6.42 [+4.66, +8.33]</td><td>+5.30 [+3.30, +7.35]</td><td> $+ 5 . 1 2 \left[ + 1 . 2 1 , + 9 . 0 3 \right]$ </td></tr><tr><td></td><td>2</td><td>+3.29 [+1.83, +4.89]</td><td>+5.95 [+3.95, +8.00]</td><td> $+ 3 . 5 0 [ + 0 . 0 0 , + 7 . 1 4 ]$ </td></tr><tr><td></td><td>3</td><td>+1.99 [+0.65, +3.36]</td><td>+4.60 [+2.75, +6.50]</td><td>+2.56 [−0.67, +5.80]</td></tr><tr><td>Self-selection from captions</td><td>1</td><td>+4.32 [+2.75, +5.96]</td><td>+6.75 [+4.95, +8.60]</td><td>+2.16[−1.08, +5.39]</td></tr><tr><td></td><td>2</td><td>+3.59 [+1.95, +5.35]</td><td>+6.55 [+4.55, +8.65]</td><td>+1.75 [−1.48, +4.99]</td></tr><tr><td></td><td>3</td><td>+3.82 [+2.22, +5.58]</td><td>+6.70 [+4.65, +8.80]</td><td>+2.43 [−0.67, +5.53]</td></tr><tr><td>Self-selection from captions + thumbnails</td><td>1</td><td>+2.29 [+1.15, +3.52]</td><td>+3.50 [+1.75, +5.30]</td><td>+1.21 [−1.75, +4.18]</td></tr><tr><td></td><td>2</td><td>+1.26[-0.04, +2.64]</td><td>+2.95 [+1.40, +4.55]</td><td>+2.96 [+0.13, +5.93]</td></tr><tr><td></td><td>3</td><td>+1.61 [+0.27, +3.02]</td><td> $+ 1 . 8 5 \ [ + 0 . 3 5 , + 3 . 4 0 ]$ </td><td>+2.96 [+0.00, +6.06]</td></tr><tr><td>All images down-sized</td><td>1</td><td>+3.13 [+1.64, +4.63]</td><td>+2.25 [+0.25, +4.30]</td><td>+1.21 [−2.29, +4.72]</td></tr><tr><td></td><td>2</td><td> $+ 0 . 4 6 \ [ - 0 . 5 7 , + 1 . 4 9 ]$ </td><td> $+ 1 . 8 5 \left[ + 0 . 0 5 , + 3 . 7 0 \right]$ </td><td>+1.48 [−1.89, +4.85]</td></tr><tr><td></td><td>3</td><td>+0.42 [−0.61, +1.49]</td><td>+0.45 [−1.15, +2.10]</td><td>+2.02 [−1.08, +5.26]</td></tr><tr><td>Its top-1 + others down-sized</td><td>1</td><td>-0.15 [−1.15, +0.88]</td><td>-0.75 [−2.60, +1.10]</td><td>-0.13 [-3.50, +3.23]</td></tr><tr><td>With question gate (30%)</td><td>1</td><td>+0.54 [−0.19, +1.30]</td><td> $+ 5 . 0 5 \ [ + 3 . 3 5 , + 6 . 8 5 ]$ </td><td> $+ 6 . 7 4 \ [ + 3 . 2 3 , + 1 0 . 2 4 ]$ </td></tr><tr><td></td><td>2</td><td> $+ 0 . 4 6 [ - 0 . 3 8 , + 1 . 3 0 ]$ </td><td> $+ 5 . 1 5 \left[ + 3 . 3 5 , + 7 . 0 5 \right]$ </td><td> $+ 7 . 1 4 \left[ + 3 . 2 3 , + 1 1 . 0 5 \right]$ </td></tr><tr><td></td><td>3</td><td> $+ 0 . 1 9 \ [ - 0 . 6 5 , + 1 . 0 7 ]$ </td><td> $+ 5 . 4 5 \ [ + 3 . 6 0 , + 7 . 3 5 ]$ </td><td> $+ 7 . 4 1 \ [ + 3 . 5 0 , + 1 1 . 3 2 ]$ </td></tr></table>

Table A2: Paired differences in the transfer experiments, with 95% bootstrap intervals.
<table><tr><td>Setting</td><td>Benchmark</td><td> $\operatorname { O u r s } @ 1 - \operatorname { R a n k } @ 1$ </td><td>Ours@1 - All</td></tr><tr><td>UniversalRAG</td><td>DMV</td><td> $+ 4 . 4 5 \ [ + 3 . 0 0 , + 5 . 9 5 ]$ </td><td>-0.50</td></tr><tr><td>UniversalRAG</td><td>MemEye</td><td> $+ 3 . 6 4 [ + 0 . 4 0 , + 6 . 8 7 ]$ </td><td>+0.81</td></tr><tr><td>NaiveRAG</td><td>DMV</td><td> $+ 3 . 7 5 \ [ + 2 . 0 5 , + 5 . 5 5 ]$ </td><td>-1.45</td></tr><tr><td>NaiveRAG</td><td>MemEye</td><td> $+ 3 . 9 1 \ [ + 0 . 9 4 , + 7 . 0 1 ]$ </td><td>-1.35</td></tr><tr><td>Qwen3.5-397B</td><td>M³Exam</td><td> $+ 1 . 6 \ [ + 0 . 2 , + 3 . 0 ]$ </td><td> $- 1 . 4 \left[ - 2 . 5 , - 0 . 3 \right]$ </td></tr><tr><td>Qwen3.5-397B</td><td>DMV</td><td> $+ 6 . 5 \ [ + 4 . 6 , + 8 . 3 ]$ </td><td> $- 6 . 7 \ [ - 8 . 2 , - 5 . 3 ]$ </td></tr></table>

Self-selection baselines. The answering model receives the retrieved memories with their captions (and, for the thumbnail variant, the same 336-pixel thumbnails System One reads) and is asked for the identifiers of at most k memories whose images it needs. It may name fewer. At k=1 it opens 0.42, 0.71 and 0.91 images on average on M<sup>3</sup>Exam, DMV and MemEye from captions, and 0.56, 0.78 and 0.94 with thumbnails.

Down-sizing. For each question the budget is the number of visual tokens of the images PixelTriage opens at k. Every retrieved image is resized, preserving its aspect ratio, to a multiple of the 28-pixel patch so that the images share the budget equally, and an image already below its share is kept. The mean token count of the down-sized deliveries is within 6.2% of PixelTriage’s in every setting.

Table A3: Opening rate set without the evaluated questions (k=1): mean accuracy and 2.5–97.5 percentiles over 1,000 draws. Open all: 75.08 / 64.75 / 48.52.
<table><tr><td></td><td>Rate</td><td>All test</td><td>Cross-fit</td><td></td><td>50 unlabeled</td><td>Random</td><td>Cost</td></tr><tr><td rowspan="2">M3Exam</td><td>30%</td><td>75.08</td><td></td><td>75.12 [75.00, 75.23]</td><td>75.17 [74.38, 76.03]</td><td>70.03</td><td>6.0%</td></tr><tr><td>50%</td><td>75.65</td><td></td><td>75.69 [75.65, 75.80]</td><td>75.70 [75.04, 76.37]</td><td>71.66</td><td>10.8%</td></tr><tr><td rowspan="2">DMV</td><td>30%</td><td>57.15</td><td></td><td>57.15 [56.95, 57.40]</td><td>57.25 [55.84, 58.53]</td><td>56.80</td><td>3.0%</td></tr><tr><td>50%</td><td>58.65</td><td></td><td>58.72 [58.50, 58.95]</td><td>58.71 [57.84, 59.63]</td><td>58.38</td><td>5.0%</td></tr><tr><td rowspan="2">MemEye</td><td>30%</td><td>41.78</td><td></td><td>41.99 [41.37, 43.00]</td><td>42.18 [38.78, 45.64]</td><td>40.94</td><td>3.4%</td></tr><tr><td>50%</td><td>45.69</td><td></td><td>45.54 [45.01, 46.09]</td><td>45.64 [43.30, 47.98]</td><td>43.26</td><td>5.6%</td></tr></table>

Table A4: Gating with a threshold fixed on the development set.
<table><tr><td></td><td>Opened</td><td>Cost</td><td>Gate</td><td>Random gate</td><td>All</td><td>Gate — random</td></tr><tr><td>M³Exam</td><td>25%</td><td>4.9%</td><td>75.00</td><td>69.63</td><td>75.08</td><td> $+ 5 . 3 7 \ [ + 3 . 9 0 , + 6 . 9 1 ]$ </td></tr><tr><td>DMV</td><td>77%</td><td>7.7%</td><td>61.40</td><td>60.48</td><td>64.75</td><td> $+ 0 . 9 2 \ [ + 0 . 0 8 , + 1 . 7 3 ]$ </td></tr><tr><td>MemEye</td><td>83%</td><td>9.4%</td><td>47.57</td><td>47.12</td><td>48.52</td><td> $+ 0 . 4 5 \ [ - 1 . 0 5 , + 1 . 8 5 ]$ </td></tr></table>

## G Implementation details

System One. Qwen3.5-2B with the Valen decision head (bilinear projections of dimension 256), initialized from a Valen checkpoint fine-tuned on Valen-General-100k. Training updates LoRA adapters (rank 32, α=64, learning rate $5 \times 1 0 ^ { - 5 } )$ on all linear layers of the language model, the vision merger $( 1 0 ^ { - 5 } )$ and the decision head $( 2 \times 1 0 ^ { - 4 } )$ for two epochs, with 28 states per optimizer step on four A100 GPUs and a maximum state length of 16,384 tokens. The hybrid linearattention layers of Qwen3.5 do not support custom attention masks, so the decision questions are concatenated after the memories in a fixed order, identically in training and inference.

Table A5: Caption and delivery conditions (7B answering model). Kept or replaced: whether an opened image keeps its caption.
<table><tr><td></td><td>Captions</td><td>Text</td><td>All</td><td>Rank@1</td><td>Ours@1</td><td>Ours@2</td><td>Ours@3</td></tr><tr><td rowspan="3">M3Exam</td><td rowspan="3">benchmark, replaced our notes, replaced</td><td>67.58</td><td>75.08</td><td>73.47</td><td>75.73</td><td>75.42</td><td>75.46</td></tr><tr><td>67.97</td><td>75.04</td><td>73.43</td><td>75.42</td><td>75.31</td><td>75.11</td></tr><tr><td>67.01</td><td>75.00</td><td>73.39</td><td>75.69</td><td>75.84</td><td>75.80</td></tr><tr><td rowspan="4">DMV</td><td rowspan="4">benchmark, kept benchmark, replaced</td><td>54.45 64.75</td><td></td><td>58.15</td><td>62.30</td><td>63.35</td><td>63.45</td></tr><tr><td>54.40</td><td>60.00</td><td>53.70</td><td>53.10</td><td>56.35</td><td>57.20</td></tr><tr><td>our notes, replaced</td><td>28.45 60.00</td><td>49.30</td><td>60.15</td><td>61.70</td><td>60.35</td></tr><tr><td>1.35</td><td>59.85</td><td>49.80</td><td>64.15</td><td>63.35</td><td>59.40</td></tr><tr><td rowspan="4">MemEye</td><td rowspan="4">benchmark, kept benchmark, replaced our notes, replaced</td><td>37.47</td><td>48.52</td><td>47.30</td><td>49.06</td><td>50.27</td><td>50.54</td></tr><tr><td>37.87</td><td>49.19</td><td>45.42</td><td>47.44</td><td>47.84</td><td>49.87</td></tr><tr><td>26.42</td><td>49.06</td><td>40.30</td><td>44.61</td><td>47.17</td><td>49.19</td></tr><tr><td>23.72</td><td>49.33</td><td>41.24</td><td>44.47</td><td>47.30</td><td>48.38</td></tr></table>

Labels. The answering model’s turn is prefilled with “Answer:” so that the next token is an option letter. Without the prefill, contexts with images often begin with a sentence, which depresses the letter probabilities unevenly across contexts. Training labels use one option order, and development labels average two orders.

## H Limitations

Training questions are synthetic multiple-choice recall questions built from single-image VQA, and the labels come from one answering model. Answers are scored by languagemodel judges, DMV-Bench is evaluated as a static replay of its interactive protocol, and M<sup>3</sup>Exam covers the two released personas. The experiments cover text and images.