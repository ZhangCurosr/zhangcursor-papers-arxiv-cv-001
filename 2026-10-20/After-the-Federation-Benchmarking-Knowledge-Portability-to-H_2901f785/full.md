# After the Federation: Benchmarking Knowledge Portability to Heterogeneous Newcomers

Xinyuan Zhao Sun Yat-sen University

## Abstract

Once historical clients become unavailable, a federation must serve newcomers through the artifact it leaves behind. How much of its knowledge remains useful in an architecture chosen afterward? We introduce FEDNBO, a sevendataset benchmark that separates knowledge supply, policy identification, and private-model compilation under disjoint data roles. Controlled studies expose inflated oracle gains, architecture-dependent transfer reversals, and a gap between teacher accuracy and private-model utility. Ordinary public-response distillation provides a competitive reference without public task labels or newcomer samples: gains over strengthened local training are 23.52 and 16.62 percentage points on 300 CelebA identities and 300 FedEMNIST writers, respectively. In a complete-pipeline FedEMNIST comparison, this reference reaches 97.13% accuracy with 12.4 seconds ofcompilation per architecture, versus 97.03% and 67.5 secondsfor a HeteroFedTTA reconstruction at the same epoch budget. Equal-count public-input interventions extend support sensitivity to trainable visual backbones. Together, these results establish an auditable benchmark and a strong referencefor evaluating the utility, cost and reliability ofpost-federation knowledge transfer.

## 1. Introduction

Federations must serve clients that arrive after historical participants become unavailable. New hospitals, devices and organizations may expose different modalities and choose architectures absent from training; restarting federated learning (FL) [18] may be impossible. The relevant endpoint is therefore the utility of the frozen artifact in a model chosen later.

Prior work handles late unlabeled clients, generates models in predefined spaces, or distills public responses into newcomer-chosen architectures [1, 14, 25, 26]. We connect these capabilities to three deployment questions (Fig. 1): does useful knowledge exist, can allowed evidence identify it, and can the private model preserve it? Query-label reuse inflates opportunity; changing architecture reverses transfer on the same clients; sparse handoffs leave a 27.17-point teacher–private-model gap.

FEDNBO contributes: (1) a seven-dataset, role-disjoint benchmark with declared information access, unseen private architectures, accuracy–cost comparisons and client harm; (2) a supply–identification–compilation decomposition with corresponding identifiability boundaries; and (3) controlled evidence for a strong public-response reference. Ordinary KD gains +23.52 pp on CelebA and +16.62 pp on FedEM-NIST over strengthened local training. Complete-pipeline comparisons quantify its cost advantage, while equal-count interventions expose public-support sensitivity in trainable visual models. These results establish useful portability under explicit conditions rather than a universal transfer guarantee.

## 2. Related Work

Future clients and architectures. OD-PFL and EPFL generate joining-client models; UPFL/FedTTA meta-learns adaptation, and HeteroFedTTA distills public-data ensemble responses into newcomer-chosen models [1, 14, 25]. FLowDUP generates models in a trained parameter subspace [26]. pFedHN, PeFLL, HyperFLoRA and Hyper-FedZero use predefined parameter families [16, 20, 21, 27]; HAFL-GHN initializes withheld architectures from known layer types [13]. ATP adapts compatible models, and TFL studies out-of-federation generalization [3, 17]. Our contract overlaps HeteroFedTTA’s base-response pathway; the contribution is its controlled deployment evaluation and failure decomposition.

Response interfaces and evidence. FedMD exchanges public responses among participating private models [12]; FedMBridge and FedAFD address heterogeneous multimodal participants [7, 23]. We evaluate a frozen endpoint through complete pipelines and same-bank ablations, separating source effort, payload and newcomer cost. Active Testing supplies a generic label-efficient selector [11]. Domainadaptation theory motivates our distinction between response recovery and benefit under unconstrained target labels [4].

![](images/5dea5dd9992188481483d7af0e9e3d56171d299ad85b7b93ce4279e7fc583fc4.jpg)  
Figure 1. The FEDNBO protocol. (a) A frozen bank exports public inputs/responses. (b) Private architectures learn from unlabeled responses; disjoint response validation selects checkpoints. Ridge is a separate control. (c) Shadow policies and newcomer selection-label methods are alternative routes; local wins ties and query labels are evaluation-only. (d) Gain separates supply, identification and compilation. Target utilities are hidden at deployment; glyphs are schematic.

The supplement maps each method’s information and architecture requirements.

## 3. Problem Formulation

A completed federation exposes a frozen bank $B ;$ historical clients cannot be contacted or retrained. Newcomer c has modality set $M _ { c }$ and private hypothesis class $\mathcal { H } _ { c } ,$ possibly unseen during federation training. Source and newcomer share label semantics and metric, but need not share representations or parameters. Fit labels $D _ { c } ^ { \mathrm { f i t } }$ train local model $f _ { c , 0 } .$ Bank policies $h _ { c , 1 } , \ldots , h _ { c , K }$ are compiled into $f _ { c , k } \in \mathcal { H } _ { c } ;$ directly deployable policies have $f _ { c , k } = h _ { c , k }$

A selector A chooses $k \in \{ 0 , \ldots , K \}$ using declared onboarding evidence, with $h _ { c , 0 } = f _ { c , 0 }$ . Query labels $D _ { c } ^ { \mathrm { q r y } }$ are evaluation-only. For higher-is-better utility U, client gain is

$$
G _ { k } ( c ) = U ( f _ { c , k } ; c ) - U ( f _ { c , 0 } ; c ) \in [ - 1 , 1 ] .\tag{1}
$$

Negative transfer means $G _ { A } ( c ) < 0$ . Fit and selection labels serve different roles: zero selection labels does not make a few-label-trained model zero-shot.

Public-response compilation. Identity-disjoint, unlabeled public anchors $A = \{ x _ { i } \} _ { i = 1 } ^ { n }$ receive frozen logits $t ( x _ { i } )$ for each supported modality subset. Only $( x _ { i } , t ( x _ { i } ) )$ crosses the interface. We fit fixed-feature ridge $( \lambda = 1 )$ or ordinary neural response KD [9], including trainable late visual blocks; held-out responses select checkpoints without task labels or newcomer inputs. A compiled model is reusable across clients choosing the same architecture–modality interface. Historical models are absent at inference.

Three separable bottlenecks. Teacher-side supply is $\begin{array} { r } { S _ { c } = \operatorname* { m a x } _ { k } U ( h _ { c , k } ; c ) - U ( f _ { c , 0 } ; c ) } \end{array}$ . Identification loss is $\begin{array} { r } { I _ { c } = \operatorname* { m a x } _ { k } U ( h _ { c , k } ; c ) - U ( h _ { c , A } ; c ) } \end{array}$ , and compilation loss is $C _ { c } = U ( h _ { c , A } ; c ) - U ( f _ { c , A } ; c )$ . Thus

$$
G _ { A } ( c ) = S _ { c } - I _ { c } - C _ { c } .\tag{2}
$$

Compilation addresses $C _ { c }$ , not the sign of $S _ { c } - I _ { c }$ . A privatecandidate oracle mixes these terms and is optimistic if query labels both select and score a policy; we also use labeldisjoint selection/evaluation. All terms are diagnostic because target utility is hidden at deployment.

## 4. Identifiability Boundary

Proposition 1 (unlabeled benefit is not identifiable). Let $f _ { 0 }$ be local and $f _ { 1 } , \ldots , f _ { K }$ fixed candidates. A selector observes unlabeled target inputs, candidate predictions and their statistics. If it selects $f _ { j } \neq f _ { 0 }$ with positive probability where they disagree, then over unrestricted $P ( Y \mid X )$ it cannot guarantee both nonnegative transfer and strictly positive gain. The same observations admit opposite label conditionals; a positive guarantee needs assumptions or label-linked evidence.

Proposition 2 (anchor-span identifiability). Let $z ( x )$ = $[ \phi _ { c } ( x ) ; 1 ] \in \mathbb { R } ^ { d }$ be a fixed representation with bias, and $Z _ { A }$ its anchor matrix. If teacher scores satisfy $t ( x ) = w _ { * } ^ { \top } z ( x )$ on anchors and targets, anchor responses determine all target scores iff

$$
\operatorname { s p a n } \{ z ( x ) : x \in \operatorname { s u p p } ( P _ { c } ) \} \subseteq \operatorname { r o w s p a n } ( Z _ { A } ) .\tag{3}
$$

Recovery error scales with the coefficient norm in $z ( x ) =$ $Z _ { A } ^ { \top } \alpha _ { x } ;$ regularization also requires a representation-residual term. This is a fixed-feature response result, not a labelaccuracy or neural-optimization guarantee. Exact recovery can copy a harmful teacher.

Proposition 3 (shadow mean-gain certificate). Condition on a frozen bank and policies, F. Suppose m labeled shadow clients and a future client are conditionally independent draws from the same declared stratum. Policies are fixed before shadow labels are opened and $G _ { k } ( c ) \in [ - 1 , 1 ]$ With probability at least $1 - \delta ,$ simultaneously for K candidates,

$$
\begin{array} { r } { \mathbb { E } [ G _ { k } ( C ) \mid \mathcal { F } ] \ge \bar { G } _ { k } - \sqrt { 2 \log ( K / \delta ) / m } . } \end{array}\tag{4}
$$

Hoeffding concentration and a union bound give the result [10]. A positive bound certifies expected future-client gain, not individual safety. For $m = 3 , K = 5 , \delta = . 0 5$ , the radius is 1.75. Proofs and precise scope are supplementary.

## 5. The FEDNBO Benchmark

## 5.1. Role-disjoint protocol

Source clients train the bank; shadow (meta) clients select policies before held-out test scoring (Table 1). Prospective studies freeze plans before test access. Labeled revisions reuse evaluated cohorts but make new choices on shadowdevelopment clients. The five-dataset policy audit repeats optimization with five seeds and linear/deep heads; UCI HAR uses linear models and temporal CNNs. Seeds are not independent clients. MELD assigns modalities by speaker and preserves dialogue separation. UCF51-20 preserves indivisible video groups; its clients are simulated. TCGA uses eligible Tissue Source Sites and class-stratified 60/20/20 splits. UCI HAR withholds gyroscope from half the subjects; FedEMNIST preserves writer identities and official within-writer partitions. Cohort and freeze receipts are supplementary.

## 5.2. Candidates and baselines

Local uses newcomer fit labels. Joint KD adds response KL with weights 0.1, 1, 10; pure KD omits fit supervision. Candidates share private initialization within each cell. CREMA-D also tests global, clustered and uniform ensembles. Fit budgets are 32 labels on MELD, 16 on TCGA, two per class on HAR, and the existing private training partition on UCF51-20.

CelebA’s label-blind hash prefix selects 4,096 images from 3,137 public identities outside evaluation clients. ResNet-18 responses train frozen-feature MobileNetV3- Small/EfficientNet-B0 ridge heads or late visual blocks; 512 held-out responses select neural checkpoints. Hard-label ridge and soft prototypes share anchors. The image pool is 28.2 MB; two float32 response tables total 64 KiB. FedEM-NIST uses 4,096 images from 500 public writers to compile unseen pixel-MLP/CNN models. Neither compiler uses public task labels or newcomer inputs; the primary decentralized revision removes the historical public-label progression gate.

Selectors include fixed policies, empirical selection-label argmax, Active Testing (fixed ensemble surrogate, expectedloss acquisition, 10% uniform floor and LURE correction [11]), PCV (one shadow-selected policy, requiring nonnegative mean gain for every shadow client), and Eq. 4. Local wins ties; same-query oracles are diagnostic. HAR additionally executes PeFLL-compatible generation from 12 labeled descriptors and ATP-compatible one-step entropy adaptation from 20 unlabeled inputs [3, 20]; their supported interfaces and references differ.

## 5.3. Metrics and statistics

Gain is in percentage points (pp). A cell is a client– architecture–seed combination. Historical client harm averages architectures/seeds first; complete-pipeline harm is architecture-specific after seed averaging. Reported 95% percentile bootstrap intervals resample clients, conditional on trained models. Reused-cohort revisions, post-hoc controls and prospective confirmations retain their distinct evidence status.

## 6. Results

## 6.1. Useful transfer across tasks and domains

Strengthened controls. CelebA uses a client-uniform FedSGD ResNet-18 bank trained on 878 identities; labelindependent identity hashing assigns landmark availability. Private MobileNetV3/EfficientNet architectures are absent from federation training. Independent 300-identity confirmations find $+ 1 6 . 6 3 \mathrm { p p }$ for ridge and 86.46% for trainable blocks, the latter adding 5.19 pp over frozen ridge.

Table 1. Benchmark tasks. TCGA-BRCA uses balanced accuracy; the others use accuracy. UCF51-20 clients are group-preserving simulations; all other client units are natural.
<table><tr><td>Dataset</td><td>classes</td><td>source/shadow/test</td><td>client unit</td><td>modalities</td><td>frozen bank → private models</td></tr><tr><td>CREMA-D [6]</td><td>6</td><td>60/15/16</td><td>actor (natural)</td><td>visual/audio features; some audio missing</td><td>global/cluster FedAvg MLPs → linear/three- layer MLP</td></tr><tr><td>MELD [19]</td><td>4</td><td>15/3/3</td><td>speaker (natural)</td><td>text/audio; dialogues re- main disjoint</td><td>GloVe/audio MLPs → MobileBERT-feature linear/MLP</td></tr><tr><td>UCF51-20 [22]</td><td>51</td><td>12/4/4</td><td>group silo (simulated)</td><td>video/audio features</td><td>FedAvg MLPs → linear/MLP</td></tr><tr><td>TCGA-BRCA [24]</td><td>2</td><td>7/3/3</td><td>Tissue Source Site (natural)</td><td>WSI embedding/RNA expression</td><td>FedAvg MLPs → linear/MLP</td></tr><tr><td>UCI HAR [2]</td><td>6</td><td>12/6/12</td><td>subject (natural)</td><td>raw accelerome- ter/gyroscope windows</td><td>flattened-window MLPs → raw ear/temporal CNN</td></tr><tr><td>CelebA [15]</td><td>2</td><td>878/310/291; +4×300 confirm</td><td>identity (natural)</td><td>raw RGB/facial land- marks</td><td>FedSGD ResNet-18 → MobileNetV3/EfficientNet-B0</td></tr><tr><td>FedEMNIST [5, 8]</td><td>10</td><td>1000/400/300</td><td>writer (natural)</td><td>raw grayscale image</td><td>residual CNN → pixel MLP/small CNN</td></tr></table>

Table 2. Public-response utility and cost. A: strengthened local controls on 300 clients per dataset (reused-cohort revisions). CelebA matches late blocks and eight fit labels; FedEMNIST uses 20 labels/1,440 local updates. B: separate 100-epoch, three-seed pipeline comparison; harm is MLP/CNN after seed averaging. Source costs differ; the full comparator reconstructs HeteroFedTTA.

A. Benefit over strengthened local training
<table><tr><td>Dataset</td><td>local</td><td>compiled</td><td>gain [95% CI]</td></tr><tr><td>CelebA</td><td>62.94</td><td>86.46</td><td>23.52 [21.69,25.38]</td></tr><tr><td>FedEMNIST</td><td>80.27</td><td>96.89</td><td>16.62 [15.35,17.86]</td></tr></table>

B. Complete pipeline: accuracy, harm and cost
<table><tr><td>Pipeline</td><td>acc. (%) harm MLP/CNN time (s)</td><td></td></tr><tr><td>Original bank + KD</td><td>97.13 12.0/2.0</td><td>12.4</td></tr><tr><td>Meta bank + KD</td><td>97.09 12.7/2.3</td><td>11.9</td></tr><tr><td>Meta bank + dual</td><td>97.03 12.3/2.7</td><td>67.5</td></tr></table>

Accuracy is %, gain is pp. Panel A FedEMNIST compilers use 50 epochs; B uses 100, with 200-epoch sensitivity in Table 6. Compilation time is per architecture on one RTX 4090. Intervals in A resample clients after averaging architectures and FedEMNIST seeds.

The primary revision matches local ImageNet initialization and trainable blocks, selecting its optimization on shadow identities; compilation retains +23.52 pp (Table 2A). Even a query-selected local ridge/finetuning oracle leaves +18.52 pp; this is diagnostic only.

FedEMNIST’s primary revision uses three client-local FedSGD teachers with GroupNorm and response-only compilation. On 300 writers it reaches 96.89% versus 80.27% for 20-label local training with shadow-selected learning rates and 1,440 updates. Seed standard deviation is 0.16 pp; three models do not make 900 independent writers. Architectureaveraged harm is 1.7–3.0%. The centralized-source audit remains a historical reference.

Task and supervision controls. A correlated-attribute confirmation gains 19.53 pp with 7.7% harm. A later, disjoint 300-identity study selects attributes with |corr(Y, Smiling)| ≤ .10 and pairwise correlation ≤ .20 using metadata: mean gain is 9.66 pp [8.09,11.20], but harm is 19.7%. Figure 2A retains the negative Necklace result; these cohorts do not identify a paired decorrelation effect. Separately supervised public-label controls (Fig. 2B) yield response-minus-control differences of −0.35, +0.40, −1.08, +2.09 pp across ridge, binary, correlated and decorrelated studies. Responses are competitive, not uniformly superior to public labels.

## 6.2. Honest policy identification

Same-query versus label-disjoint gain falls from +2.49 to −1.01 pp on CREMA-D and +22.98 to −0.08 pp on TCGA (Fig. 3A); UCF51-20 retains +2.12 pp from +3.56 pp. Candidate existence and policy identification differ: all fixed CREMA-D policies are harmful, all MELD policies help, and HAR retains positive fixed gains despite PCV abstention (Table 3). TCGA has a useful fixed KD-1 policy (+17.62 pp), but small, imbalanced selection cohorts fail to identify it per client.

More labels help selectively. Four Active Testing labels recover 13.68 pp on MELD and 1.14 pp on UCF51-20; additional labels improve both (Fig. 3B). CREMA-D stays negative with all selection labels, while TCGA reaches only 1.67 pp. Labels cannot create transferable candidates or make a shifted cohort representative.

Shadow evidence and shift. PCV abstains on CREMA-D, UCF51-20 and HAR. On TCGA it selects KD-0.1 and gains 13.33 pp, yet harms site A2 and has a client interval crossing zero. Three nonnegative shadow sites did not protect every future site. The formal LCB abstains throughout: a true gain of .20 requires $m \geq 2 3 1$ for a positive bound at the mean. In 50,000 bounded-gain trials, deployment reaches 99.8% at m = 384 under future mean .20; shifting that mean by −.25

![](images/8a3db25afe147c8b581e773bc3429ca36d07a03a231976221210e0a655a39feb.jpg)

![](images/e96252b139e3635b870d334a1a769577ba20691d2e2ee56dac9096cc9a9fb2f4.jpg)  
Figure 2. Task and supervision controls. A: Attribute gains on 300 decorrelated CelebA identities include the negative Necklace result. B: Response-minus-public-label-control gains in separate historical studies. These post-confirmation supervised controls are not compile inputs. Bars: 95% client-bootstrap intervals; cohorts and local references differ across conditions.

## A Selection optimism

![](images/7709a1bb6dd63c98f4a6253f00b6e5bc275e49277bb7c33180209b57c315dbf9.jpg)

![](images/d75b5dbc2d162b3f0da69d5b5e19c3867d2e2aadde591960da3de3f3064ce9d1.jpg)  
D Architecture on individual subjects

![](images/70b19cd8b1e87c6e99a6e075ac11ae94983f0a01563ea0b2278cba233aca2c45.jpg)

![](images/b3ecb0ecb367ce34249fb0295f8bd961c8ff77893649bc5c6c78c98f6af280cf.jpg)  
Figure 3. Four onboarding audits. A: Same-query versus label-disjoint selection (95% newcomer intervals). B: Active Testing across labe budgets. C: Mean selection and LCB power/harm under declared or shifted populations (50,000 trials). D: Fixed KD-1 gains on 12 HAR subjects, averaged over seeds within each architecture.

causes harmful deployment in 99.8% of trials (Fig. 3C). This is a population-assumption audit, not another benchmark.

## 6.3. Private architecture and handoff loss

HAR freezes roles, missingness, five policies and budgets before opening 12 test subjects; a shadow-only amendment sets two fit labels per class. Fixed KD-1 gains 3.88 pp overall but loses 1.73 pp in linear models and gains 9.49 pp in temporal CNNs (Fig. 3D). The paired interaction is 11.21 pp [7.57,15.17], p = .0005 by exact sign-flip. Six disagreement labels halve harm from 16.7% to 8.3%; 24 retain that harm and raise gain to 4.47 pp (Table 4A). Neither removes negative linear-stratum gain. Architecture routing discovered here remains retrospective.

Table 3. Fixed policies and shadow-selected deployment. Gain/CI: pp; harm: %. Policies 1–4 are global/cluster 0/cluster 1/uniform on CREMA-D and KD 0.1/1/10/pure KD elsewhere. PCV uses only shadow labels; all LCB rules abstain. Test cohorts: MELD/TCGA, 3 clients; UCF, 4 simulated silos.
<table><tr><td></td><td colspan="4">Fixed-policy gain</td><td colspan="3">PCV deployment</td></tr><tr><td>Dataset</td><td>1</td><td>2</td><td>3</td><td>4</td><td>Choice</td><td>Gain [95% CI]</td><td>Harm</td></tr><tr><td>CREMA-D</td><td>-7.08</td><td>-6.78</td><td>-8.80</td><td>-6.87</td><td>local</td><td>0.00 [0.00,0.00]</td><td>0.0</td></tr><tr><td>MELD</td><td>+11.16</td><td> $+ 1 6 . 0 5$ </td><td> $+ 1 7 . 6 6$ </td><td> $+ 1 7 . 8 9$ </td><td>pure KD</td><td> $+ 1 7 . 8 9 \ [ 1 5 . 3 0 , 2 2 . 0 3 ]$ </td><td>0.0</td></tr><tr><td>UCF51-20</td><td> $+ 1 . 9 0$ </td><td> $+ 1 . 5 7$ </td><td>-1.18</td><td> $- 1 . 4 5$ </td><td>local</td><td> $0 . 0 0 \left[ 0 . 0 0 , 0 . 0 0 \right]$ </td><td>0.0</td></tr><tr><td>TCGA-BRCA</td><td> $+ 1 3 . 3 3$ </td><td> $+ 1 7 . 6 2$ </td><td> $+ 1 7 . 1 8$ </td><td> $+ 1 4 . 2 1$ </td><td>joint KD 0.1</td><td>+13.33[-1.31,30.00]</td><td>33.3</td></tr><tr><td>UCI HAR</td><td> $+ 3 . 2 3$ </td><td> $+ 3 . 8 8$ </td><td>+1.98</td><td> $+ 1 . 6 9$ </td><td>local</td><td>0.00 [0.00,0.00]</td><td>0.0</td></tr></table>

Table 4. HAR architectures, selection labels and compatible baselines. A: sealed 12-subject audit; gain/CI: pp, harm: %. Oracle is diagnostic. B: separate supported slices with different references; PeFLL uses 12 labeled descriptors, ATP an unlabeled batch of 20. Accuracy: %.

A. Fixed transfer and disagreement-label audit
<table><tr><td>Policy</td><td>Gain [95% CI]</td><td>Harm</td><td>Linear gain</td><td>CNN gain</td></tr><tr><td>Fixed joint KD 1</td><td>+3.88 [2.19,5.51]</td><td>16.7</td><td>-1.73</td><td>+9.49</td></tr><tr><td>Disagreement audit, 6 labels</td><td>+3.22 [1.83,4.64]</td><td>8.3</td><td>-1.08</td><td>+7.52</td></tr><tr><td>Disagreement audit, 24 labels</td><td> $+ 4 . 4 7 \ [ 2 . 7 1 , 6 . 3 4 ]$ </td><td>8.3</td><td>-0.93</td><td>+9.87</td></tr><tr><td>Same-query oracle (diagnostic)</td><td>+8.03</td><td>0.0</td><td>+2.89</td><td>+13.18</td></tr></table>

B. Named methods on supported interfaces
<table><tr><td>Method</td><td>Reference</td><td>Ref. acc.</td><td>Method acc.</td><td>Gain [95% CI]</td><td>Harm</td></tr><tr><td>PeFLL</td><td>FedAvg + fit (12 labels)</td><td>44.4</td><td>51.2</td><td>+6.79 [3.35,9.95]</td><td>16.7</td></tr><tr><td>ATP</td><td>Compatible global model</td><td>46.02</td><td>45.97</td><td>-0.05 [-0.69,0.59]</td><td>50.0</td></tr></table>

Compatible PeFLL and ATP slices have different local references (Table 4B): PeFLL gains 6.79 pp but harms 16.7%; ATP changes mean accuracy little (−0.05 pp) while harming 50.0%. They are not a common-baseline ranking. On 291 sealed CelebA identities, sparse prototype handoff reaches 65.23% versus 63.49% local and 92.40% teacher, leaving a 27.17 pp [25.17,29.17] portability gap and 37.5% harm. A six-transformation control does not resolve it (−0.47 pp [−2.19,1.09]).

## 6.4. Public-input quantity, support and compression

On 270 shadow identities, ridge accuracy rises from 50.44% at 16 public anchors to 85.28% at 4,096, versus 63.31% local and 91.57% teacher (Fig. 4A). Controlling budget, backbone and modality removes the coverage–gain association (ρ = −.012 [−.095,.070]). Blurring only public inputs reduces 4,096-anchor gain from 21.97 to 8.01 pp at radius 4 and raises harm from 5.6% to 24.8% (Fig. 4B).

Support beyond count. An equal-count, fixed-feature intervention compares broad inputs with the upper quartile of pose asymmetry. Broad support adds 4.61 pp [3.67,5.56] at 64 anchors and 3.90 pp [2.92,4.90] at 128 across three draws. Holding initialization, 336 neural updates and 512 validation responses fixed extends this comparison to trainable blocks (Table 5; Fig. 4C). At 512 inputs, harm falls from 13.7/13.0% to 10.4/8.5% for MobileNet/EfficientNet. EfficientNet at 128 remains inconclusive despite worse response KL. These neural findings are empirical, not consequences of Proposition 2.

Matched and post-hoc controls. With identical initial FP32 features, inputs and λ = 1, late-block training exceeds closed-form ridge by 5.52, 6.51, 6.47 pp at 64/128/512 inputs. On separate cohorts, hard-label/response ridge reach 80.25/80.85%, while soft prototypes reach 65.83%. Neither pivoted QR nor target-nearest anchor selection consistently beats a hash prefix. A shadow-selected one-bit quantizer gives validation accuracy 84.91% versus float32 85.28%; on a reused 300-identity cohort it gives 85.71%, +22.56 pp and 7.0% harm with a 1,024-byte response digest. The 28.2- MB image pool remains required; this post-hoc compression audit implies no privacy guarantee. Full control grids are supplementary.

![](images/d40f5443d17b65346440e29d28f2adecde3ee7b54b07d02759670d90f433f568.jpg)

![](images/0b176d4b3343634b7961c9b635c82bc0cca37c55f4cf13c912bbf7b2bed692c2.jpg)

![](images/21e4ad8a32524ea914b807a9fa4424473c530d1219dec344044bbcb80cde45bc.jpg)  
Figure 4. Public inputs and private transfer. On the same 270 shadow identities: A: anchor-count frontier; B: blur applied only to public inputs; C: broad-minus-restricted neural support with equal counts, 336 updates and 512 extra validation responses. Bars in C are paired 95% client intervals conditional on three draws; EfficientNet at 128 is inconclusive.

Table 5. Equal-count public-support ablation. Broad/restricted pools share teacher, initialization, 336 updates, 512 validation responses and 270 identities. Accuracy/harm: %; differences: pp. CIs resample clients after averaging three draws. ∆KL is broad minus restricted; harm is broad/restricted.
<table><tr><td>Private backbone</td><td>Fit inputs</td><td>Broad acc.</td><td>Restr. acc.</td><td>Difference [95% CI]</td><td>∆KL</td><td>Harm</td></tr><tr><td>MobileNet blocks</td><td>64</td><td>72.07</td><td>69.01</td><td>3.06 [1.84,4.32]</td><td>-0.0490</td><td>27.4/32.2</td></tr><tr><td></td><td>128</td><td>78.46</td><td>75.42</td><td>3.04 [1.93,4.20]</td><td>-0.0426</td><td>17.8/24.8</td></tr><tr><td></td><td>512</td><td>85.32</td><td>80.94</td><td>4.38 [3.26,5.54]</td><td>-0.0861</td><td>10.4/13.7</td></tr><tr><td>EfficientNet blocks</td><td>64</td><td>75.96</td><td>70.12</td><td>5.83 [4.51,7.21]</td><td>-0.0242</td><td>25.6/35.2</td></tr><tr><td></td><td>128</td><td>82.19</td><td>82.04</td><td>0.15 [–0.76,1.10]</td><td>+0.0248</td><td>14.1/14.4</td></tr><tr><td></td><td>512</td><td>87.82</td><td>84.94</td><td>2.89 [1.84,3.94]</td><td>-0.0667</td><td>8.5/13.0</td></tr></table>

## 6.5. Complete-pipeline cost and client harm

At 100 epochs, original-bank KD reaches 97.13% with 12.4 s compilation per architecture versus 97.03% and 67.5 s for the HeteroFedTTA reconstruction (Table 2B). Full-minusminimal gain is −0.11 pp [−0.28,0.07]; at 200 epochs it is 0.04 pp [−0.12,0.21], not evidence of equivalence. Table 6 retains both architectures and budgets: doubling optimization roughly doubles compilation without uniform accuracy gain. Same-bank KD separates source improvement from target adaptation; at 200 epochs its advantage over originalbank KD is 0.15 pp [0.04,0.27].

Amortization and source effort. For J interfaces and N arrivals, stage costs satisfy

$$
C _ { \mathrm { t o t a l } } = C _ { \mathrm { b a n k } } + J \bar { C } _ { \mathrm { c o m p i l e } } + N \bar { C } _ { \mathrm { t a r g e t } } .\tag{5}
$$

Bank cost includes export; communication is separate. Original source training takes 85.7–91.6 s, selected meta checkpoints add 2,656.9–10,668.1 s, and source-budget search costs 12,463.9–43,954.6 s. Uncached minimal/full payloads are 3.22/3.38 MiB (responses: 160/320 KiB). These are matched onboarding budgets, not equal source effort or distributed latency; one reconstructed source run reaches its cap before plateau. Reuse determines the tradeoff.

Who is harmed? Original-bank KD harms 12.0% of MLP writers versus 2.0% of CNN writers. In the paired CelebA historical-ridge audit, updating blocks reduces compilerinduced harm from 6.3% to 2.0% and compiler loss from 6.90 to 1.71 pp; inherited harm stays near 4%. Image-only and image–landmark gains remain 24.74/22.03 pp. The decorrelated task retains 19.7% harm (10.3% inherited, 9.3% compilation): Necklace’s 1.19 pp teacher supply cannot absorb 3.60 pp compilation loss. Query-set rates are not individual-risk guarantees.

A diagnostic CREMA-D public-actor audit instead has insufficient supply: local/teacher/64-anchor accuracy is 80.68/55.19/53.13%. Broad-minus-restricted gains at 64/96 inputs cross zero, and both pools harm all ten validation actors. Only three have audio. Better response recovery cannot rescue a harmful teacher. Primary mean evidence comes from 270–300-client image cohorts; HAR and the remaining tasks expose mechanism and selection boundaries.

## 7. Scope and Implications

Report teacher utility against a credible local baseline, separate selection from scoring, and test the intended private architecture at a declared budget. CelebA covers image-only and image-derived landmarks; FedEMNIST tests unimodal portability. General independent-sensor multimodal transfer remains distinct. The fednbo.py entry point binds cohort identities, protocols, result hashes and reproduction commands.

Table 6. FedEMNIST compiler-budget sensitivity. Entries are 100/200 epochs with matched inputs, initialization, optimizer and batch order. Accuracy/harm: %; harm averages three seeds by architecture. Meta + dual reconstructs HeteroFedTTA. Timings exclude I/O/loading; source effort differs and is reported separately.
<table><tr><td>Architecture</td><td>Pipeline</td><td>Accuracy</td><td>Harm</td><td>Compile (s)</td><td>Target (ms)</td></tr><tr><td>MLP</td><td>Original + KD</td><td>95.87/95.82</td><td>12.0/12.0</td><td>11.8/23.1</td><td>0.71/0.66</td></tr><tr><td></td><td>Meta + KD</td><td>95.74/95.97</td><td>12.7/11.0</td><td>11.0/22.4</td><td>0.66/0.66</td></tr><tr><td></td><td>Meta + dual</td><td>95.76/95.82</td><td>12.3/12.0</td><td>67.1/132.2</td><td>6.11/6.23</td></tr><tr><td>CNN</td><td>Original + KD</td><td>98.39/98.33</td><td>2.0/2.3</td><td>12.9/25.7</td><td>0.79/0.77</td></tr><tr><td></td><td> $\mathrm { M e t a + K D }$ </td><td>98.44/98.49</td><td>2.3/2.7</td><td>12.8/25.3</td><td>0.69/0.69</td></tr><tr><td></td><td> $\mathrm { \ M e t a + d u a l }$ </td><td>98.29/98.42</td><td>2.7/2.0</td><td>67.9/136.5</td><td>42.88/10.94</td></tr></table>

## 8. Limitations

MELD/TCGA have three test clients, UCF51-20 clients are simulated, and four tasks use frozen features. The image studies use in-domain anchors; cross-domain support, response privacy and prospective individual safety remain open. Proposition 2 concerns fixed linear representations. The HeteroFedTTA reconstruction has finite source budgets; revisions reuse evaluated cohorts. The supplement preserves superseded centralized FedEMNIST and public-label-gate protocols alongside the client-local replications.

## 9. Conclusion

FEDNBO makes post-federation portability measurable through knowledge supply, policy identification and privatemodel compilation. Controlled experiments reveal where apparent opportunities disappear and where a simple publicresponse handoff succeeds. Its strong image-task results, competitive complete-pipeline accuracy and lower compilation cost establish a useful reference for future methods. Public-support interventions and client-level harm identify conditions those methods must confront to improve deployment outcomes.

## References

[1] Ohad Amosy, Gal Eyal, and Gal Chechik. Late to the party? on-demand unlabeled personalized federated learning. In WACV, pages 2184–2193, 2024. 1

[2] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz. A public domain dataset for human activity recognition using smartphones. In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, pages 437–442, 2013. 4

[3] Wenxuan Bao, Tianxin Wei, Haohan Wang, and Jingrui He. Adaptive test-time personalization for federated learning. In NeurIPS, 2023. 1, 3

[4] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine Learning, 79: 151–175, 2010. 1

[5] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konecnˇ y, H. Brendan McMahan, Virginia Smith,´ and Ameet Talwalkar. LEAF: A benchmark for federated settings. arXiv preprint arXiv:1812.01097, 2018. 4

[6] Houwei Cao, David G. Cooper, Michael K. Keutmann, Ruben C. Gur, Ani Nenkova, and Ragini Verma. CREMA-D: Crowd-sourced emotional multimodal actors dataset. IEEE Transactions on Affective Computing, 5(4):377–390, 2014. 4

[7] Jiayi Chen and Aidong Zhang. FedMBridge: Bridgeable multimodal federated learning. In ICML, pages 7667–7686, 2024. 1

[8] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre´ van Schaik. EMNIST: Extending MNIST to handwritten letters. In International Joint Conference on Neural Networks, pages 2921–2926, 2017. 4

[9] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS Deep Learning Workshop, 2015. 2

[10] Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. 3

[11] Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active testing: Sample-efficient model evaluation. In ICML, pages 5753–5763, 2021. 1, 3

[12] Daliang Li and Junpu Wang. FedMD: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581, 2019. 1

[13] Or Litany, Haggai Maron, David Acuna, Jan Kautz, Gal Chechik, and Sanja Fidler. Federated learning with heterogeneous architectures using graph hypernetworks. arXiv preprint arXiv:2201.08459, 2022. 1

[14] Yuange Liu, Daobin Luo, Weishan Zhang, Wei Jiang, Chaoqun Zheng, Xiaohui Sun, Yuru Liu, Qiao Qiao, Tao Chen, Hongwei Zhao, Su Yang, and Fei-Yue Wang. EPFL: Toward elastic personalized federated learning with seamless client joining and quitting. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 55(12):9324–9338, 2025. 1

[15] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision, pages 3730–3738, 2015. 4

[16] Qikai Lu, Di Niu, Mohammadamin Samadi Khoshkho, and Baochun Li. HyperFLoRA: Federated learning with instantaneous personalization. In SDM, pages 824–832, 2024. 1

[17] Mengmeng Ma, Tang Li, and Xi Peng. Beyond the federation: Topology-aware federated learning for generalization to unseen clients. In ICML, pages 33794–33810, 2024. 1

[18] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In AISTATS, 2017. 1

[19] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In ACL, pages 527–536, 2019. 4

[20] Jonathan Scott, Hossein Zakerinia, and Christoph H. Lampert. PeFLL: Personalized federated learning by learning to learn. In ICLR, 2024. 1, 3

[21] Aviv Shamsian, Aviv Navon, Ethan Fetaya, and Gal Chechik. Personalized federated learning using hypernetworks. In ICML, pages 9489–9502, 2021. 1

[22] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. Technical Report CRCV-TR-12-01, University of Central Florida, 2012. 4

[23] Min Tan, Junchao Ma, Yinfu Feng, Jiajun Ding, Wenwen Pan, Tingting Han, Qian Zheng, Zhenzhong Kuang, and Zhou Yu. FedAFD: Multimodal federated learning via adversarial fusion and distillation. In CVPR, pages 3400–3409, 2026. 1

[24] The Cancer Genome Atlas Research Network. Machine learning identifies stemness features associated with oncogenic dedifferentiation. Cell, 173(2):338–354.e15, 2018. 4

[25] Tiandi Ye, Cen Chen, Yinggui Wang, Xiang Li, and Ming Gao. UPFL: Unsupervised personalized federated learning towards new clients. In SDM, pages 851–859, 2024. 1

[26] Hossein Zakerinia, Jonathan Scott, and Christoph H. Lampert. Federated learning with unlabeled clients: Personalization can happen in low dimensions. In ICML, 2026. 1

[27] Yuhao Zhou, Jindi Lv, Yuxin Tian, Dan Si, Qing Ye, and Jiancheng Lv. Deploying models to non-participating clients in federated learning without fine-tuning: A hypernetworkbased approach. In ICLR, 2026. 1

# Supplementary Material: After the Federation

## 1. Protocol, Cohorts and Evidence Status

This supplement follows the four experimental questions in the main paper: selection, private-model realization, public-input support, and cost/benefit. The main paper presents the principal policy, architecture, support, and budget contrasts. The consolidated tables here retain their original context and additional shadow, historical-control, and source-search rows. Full grids and individual seeds remain in the accompanying JSON/CSV files.

Roles and uncertainty. Source clients train the bank; shadow-development clients choose recipes; shadow-validation and test clients evaluate frozen decisions. Revisions explicitly reuse previously evaluated cohorts. The primary revised CelebA and FedEMNIST results are not additional prospective confirmations. Confidence intervals resample natural clients after averaging architectures and seeds, unless architecture is a displayed stratum. They quantify client sampling conditional on the trained models. Three teacher seeds do not turn 300 writers into 900 clients. MELD and TCGA have only three test clients each; UCF51-20 uses four simulated test silos. These studies diagnose boundary cases rather than establish population-wide success.

Natural-client protocols. The original CelebA protocol has 1,479 eligible identities: 878 source, 310 shadow (40 development, 270 validation) and 291 test. Its metadata-only v2 amendment increased the minimum class count from eight to ten within already frozen roles before test-pixel access. Each query has four examples per class; selection has two. Local fitting uses four labels per class. Landmark availability is assigned by an identity hash independently of labels. Subsequent ridge, trainable-block, correlated-attribute and decorrelated-attribute confirmations each use 300 new identities, disjoint from all preceding cohorts and public identities. Their image counts are 6,881, 6,959, 6,468 and 8,016. The current primary baseline revision uses the trainable-block cohort. The registry verifies exact identities and result cells for all five cohorts.

FedEMNIST uses the digits-only TensorFlow Federated release. An eligible writer has two training examples per digit and at least seven represented digits in its official test split. A metadata hash assigns 1,000 source, 500 public, 100 development, 300 validation and 300 test writers from 2,582 eligible writers. The 20 local labels are fixed training examples; officia per-writer test images are query-only. Public-image ordering excludes task labels. UCF groups remain indivisible across data roles; the earlier clip-split UCF20 result is excluded from the benchmark.

Frozen chronology. Table S1 binds each original confirmation policy to the first complete result artifact. The guarded runner records the consumed policy hash; the verifier checks policy, receipt and result agreement. Local filesystem times (UTC+8) are repository records, not independent trusted timestamps. The original FedEMNIST progression gate used public labels and a centralized teacher; the later primary replications remove both dependencies. Task amendments, historical results and strengthened-baseline revisions remain distinguishable in the release.

Table S1. Original confirmation chronology. Hash prefixes identify full policy hashes in the receipts. Revised baselines reuse these cohorts rather than create new confirmations.
<table><tr><td>Study</td><td>roles</td><td>frozen choice</td><td>policy/result time</td><td>policy SHA</td></tr><tr><td>CelebA sparse handoff</td><td>878/310/291 identities</td><td>fit pseudo-label prototype; 8 labels</td><td>09-2623:53/09-2623:55</td><td>9f2392c1</td></tr><tr><td>CelebA ridge</td><td>new 300 identities</td><td>4,096 anchors; ridge alpha 1</td><td>09-2701:26/09-27 01:27</td><td>59f616ab</td></tr><tr><td>CelebA trainable blocks</td><td>second new 300</td><td>late blocks; 8 epochs; response selection</td><td>09-27 02:07/09-27 02:09</td><td>561559fa</td></tr><tr><td>CelebA 3-label</td><td>third new 300</td><td>3 incidence labels; late blocks</td><td>09-2710:39/09-2710:40</td><td>a0b9c7e6</td></tr><tr><td>CelebA decorrelated 3-label</td><td>fourth new 300</td><td>correlation rule; inherited blocks</td><td>09-2712:24/09-2712:26</td><td>9a2de6ac</td></tr><tr><td>FedEMNIST</td><td>1000/400/300 writers</td><td>4,096 anchors; 2 architectures; 20-label local</td><td>09-2713:53/09-2714:02</td><td>6f66f2a4</td></tr></table>

Initial policy-audit optimization. CREMA-D, MELD, UCF51-20 and TCGA use seeds 4201/4301/4401/4501/4601, linear and three-layer private heads, and respectively 160/200/200/240 updates. AdamW rates are $3 \times 1 0 ^ { - 3 }$ for linear and $1 0 ^ { - 3 }$ for deep heads; hidden width is 64 with normalization and GELU. TCGA uses balanced loss and accuracy. UCI HAR uses seeds 6101/6201/6301/6401/6501, 40 FedAvg rounds, flattened-window teachers and linear/temporal-CNN private models trained for 150 steps on two labels per class. The natural client, rather than the optimization cell, is the resampling unit.

## 2. Useful Transfer and the Historical Controls

Main Table 2 contains the strengthened primary baselines and the complete-pipeline comparison. Table S2 consolidates the earlier confirmation endpoints, so that the baseline changes and independent cohorts can be inspected without six separate resul tables. These are paired gains within rows; improvements across different rows cannot be attributed to a single intervention.

Table S2. Historical cohort endpoints, distinct from the strengthened references in main Table 2. Each row retains its own local baseline; rows are not paired interventions. Accuracy is %, gain is pp; the two attribute studies use balanced accuracy. Harm averages architectures; decorrelated ties use the recorded correction. The original FedEMNIST teacher was centralized.
<table><tr><td>Study (clients)</td><td>Local</td><td>Private</td><td>Teacher</td><td>Gain [95% CI]</td><td>Harm (%)</td></tr><tr><td>CelebA sparse (291)</td><td>63.49</td><td>65.23</td><td>92.40</td><td>1.74 [0.15,3.22]</td><td>37.5</td></tr><tr><td>CelebA ridge (300)</td><td>64.23</td><td>80.85</td><td>88.13</td><td>16.63 [14.63,18.63]</td><td>15.3</td></tr><tr><td>CelebA blocks (300)</td><td>63.15</td><td>86.46</td><td>88.17</td><td>23.31 [21.33,25.27]</td><td>6.0</td></tr><tr><td>CelebA correlated (300)</td><td>55.95</td><td>75.48</td><td>76.44</td><td>19.53 [17.92,21.19]</td><td>7.7</td></tr><tr><td>CelebA decorrelated (300)</td><td>56.40</td><td>66.07</td><td>67.19</td><td>9.66 [8.09,11.20]</td><td>19.7</td></tr><tr><td>FedEMNIST original (300)</td><td>77.90</td><td>96.75</td><td>99.34</td><td>18.85 [17.49,20.20]</td><td>1.3</td></tr></table>

Historical and private visual models. A client-uniform FedSGD ResNet-18 starts from ImageNet weights and updates layer 4 plus its image/landmark head; BatchNorm statistics remain frozen. Its selected source-query accuracy is 91.51%. The original sparse handoff forms prototypes from hard teacher labels on local fit images, overwriting the eight true labeled examples. It never calls the teacher on query inputs. On the 270 validation identities it gains +2.41 points, while direct teache accuracy is 91.57%; its sealed 291-identity endpoint appears in Table S2. A six-transformation development control changes gain by −0.47 pp [−2.19,1.09], leaving the sparse-handoff limitation unresolved.

Public ridge fits historical log-odds with fixed λ = 1 on frozen MobileNetV3-Small or EfficientNet-B0 features. The neural compiler updates MobileNet blocks 9–12 or EfficientNet blocks 6–8 plus their landmark-aware heads, leaving earlier ImageNe blocks fixed. Each public image supplies image-only and image–landmark views. BCE matches historical probabilities; 3,584 images fit and 512 response-only images select checkpoints. The eight-epoch visual-block recipe and checkpoints were fixed before the second confirmation; no public task label or newcomer sample enters compilation. Compilers are shared within an architecture/modality interface.

Strengthened local controls. The current CelebA local reference has the identical ImageNet initialization and trainable blocks, a ridge-initialized head and eight labels. Forty development identities select learning rates $1 0 ^ { - 4 } / 1 0 ^ { - 3 }$ , 10/30/100 updates and horizontal flips with consistent landmark remapping. A query-label-privileged local oracle choosing between ridge and finetuning still leaves +18.52 pp [16.69,20.38] for compilation; it is a diagnostic, not a deployment rule. Three of 600 original ridge cells change near-zero predictions between half- and single-precision features. The revision keeps the archived compiler endpoint fixed and records this numerical amendment separately.

The primary FedEMNIST teachers use three independent seeds, 1,200 FedSGD rounds, eight writers per round and 64 examples per writer. Each forward/backward pass accesses one writer; the server averages gradients and uses AdamW. GroupNorm avoids cross-writer BatchNorm statistics; source-query accuracy selects the teacher. Fifty-epoch private KD uses AdamW $\mathrm { a t } 8 \times 1 0 ^ { - 4 }$ and response-only validation. A 100-writer development grid first tested 120/360/720 local steps and rates $5 \times 1 0 ^ { - 4 } / 2 \times 1 0 ^ { - 3 }$ . A logged upper-bound amendment added 1,440 steps before the new local test summary was inspected. MLP/CNN select 1,440 steps at these two respective rates. The three compiled accuracies are 96.92/97.03/96.71%, with gains 16.65/16.76/16.44 points; mean gain is 16.62 with seed SD 0.16 points. Individual-run harm and model records remain in the numeric export.

Task and supervision controls. The incidence-selected correlated attributes are High Cheekbones, Mouth Slightly Open and Oval Face; source correlations with Smiling are .657/.572/.117. The later task requires |corr| ≤ .10 with Smiling and pairwise $| \mathrm { c o r r } | \le . 2 0$ , selecting Arched Eyebrows, Wavy Hair and Wearing Necklace from incidence metadata. Their gains are $+ 1 4 . 7 0 \mathrm { p p } [ 1 1 . 9 9 , 1 7 . 3 8 ] , + 1 6 . 6 9 \mathrm { p p } [ 1 4 . 1 8 , 1 9 . 2 4 ]$ , and $- 2 . 4 0 \mathrm { p p } \left[ - 4 . 6 7 , - 0 . 1 0 \right]$ . Local references use at most 12 labeled images and development-selected nearest centroids. The corresponding 20/75 and 20/91 development/validation cohorts never join confirmation roles.

Public-label oracle controls are separate supervised experiments, performed after the corresponding CelebA confirmations. For response ridge, binary blocks, correlated blocks and decorrelated blocks, oracle accuracies are 81.21/86.06/76.56/63.98%; response-minus-oracle differences are $- 0 . 3 5 \mathrm { p p } [ - 1 . 0 0 , 0 . 3 3 ] , + 0 . 4 0 \mathrm { p p } [ - 0 . 4 0 , 1 . 1 9 ] , - 1 . 0 8 \mathrm { p p } [ - 1 . 9 2 , - 0 . 2 4 ] ,$ , and +2.09 pp [0.98,3.15]. The original FedEMNIST supervised public model reaches 95.54%; response KD exceeds it by 1.21 points [0.82,1.61]. These controls constrain the role of public supervision, without changing the information available to the response method.

## 3. Can Allowed Evidence Select a Useful Policy?

Table S3. Fixed-policy opportunity and deployable population selection. The four gains are global/cluster 0/cluster 1/uniform for CREMA-D and joint KD 0.1/1/10/pure KD elsewhere. PCV uses shadow labels only; local has zero gain. All LCB decisions abstain. TCGA and MELD have only three test clients.
<table><tr><td>Dataset</td><td>Four fixed gains (pp)</td><td>PCV choice</td><td>PCV gain [95% CI]</td><td>Harm</td></tr><tr><td>CREMA-D</td><td>-7.08 / -6.78 / -8.80 / -6.87</td><td>local</td><td>0.00 [0.00,0.00]</td><td>0.0%</td></tr><tr><td>MELD</td><td>+11.16/+16.05/+17.66 /+17.89</td><td>pure KD</td><td>+17.89 [15.30,22.03]</td><td>0.0%</td></tr><tr><td>UCF51-20</td><td> $+ 1 . 9 0 / + 1 . 5 7 / - 1 . 1 8 / - 1 . 4 5$ </td><td>local</td><td>0.00 [0.00,0.00]</td><td>0.0%</td></tr><tr><td>TCGA-BRCA</td><td> $+ 1 3 . 3 3 / + 1 7 . 6 2 / + 1 7 . 1 8 / + 1 4 . 2 1$ </td><td>joint KD 0.1</td><td>+13.33[-1.31,30.00]</td><td>33.3%</td></tr><tr><td>UCI HAR</td><td> $+ 3 . 2 3 / + 3 . 8 8 / + 1 . 9 8 / + 1 . 6 9$ </td><td>local</td><td>0.00 [0.00,0.00]</td><td>0.0%</td></tr></table>

Oracle and active-label controls. For accuracy, repeated random query halves alternate candidate selection and scoring. Balanced-accuracy splits are stratified; when a TCGA minority class has only one query example, a separate selection cohort chooses the policy and query labels remain evaluation-only. The exception is logged per cell. Active Testing uses the candidate-vote surrogate, expected zero-one-loss acquisition with a 0.1 uniform mixture, LURE correction, 200 acquisition repetitions and 5,000 client-cluster bootstrap replicates. TCGA uses class-conditional LURE ratio estimates and abstains until both classes are sampled. The main paper gives the label-budget curves and fixed-policy deployment table; Table S3 retains the consolidated values alongside the detailed selection protocol.

Conditional evidence requirements. The simultaneous Hoeffding radius yields $m \ge \lceil 2 \log ( K / \delta ) / \gamma ^ { 2 } \rceil$ for radius at most mean gain γ. At $K = 5 , \delta = . 0 5$ , a gain of .20 requires 231 independent same-population shadow clients; three to fifteen clients cannot certify moderate gains. In a 50,000-trial bounded-gain diagnostic, deployment reaches 99.8% at $m = 3 8 4$ when the future mean is .20, yet becomes harmful in 99.8% of trials when the future mean shifts down by .25. Larger cohorts address power under the population assumption; they cannot establish that assumption.

## 4. Private Architectures and Compatible Prior Methods

The paired temporal-CNN-minus-linear interaction for fixed joint KD 1 is +10.73 pp [3.21,17.73] on six shadow subjects (exact $p = . 1 2 5 ) \mathrm { a n d } + 1 1 . 2 1 \mathrm { p p } [ 7 . 5 7 , 1 5 . 1 7 ]$ on 12 test subjects $( p = . 0 0 0 5 )$ . Each subject averages seeds before comparison. A retrospective architecture-specific shadow rule gains 3.06 points but harms 25.0% of test subjects. Leave-one-subject-out selection over all 18 target subjects gains 4.30 points with 11.1% harm when stratified, versus 3.06 points and 27.8% when pooled. These analyses motivate a future prospective rule; they do not validate one on their hypothesis-discovery cohort.

PeFLL and ATP interfaces. PeFLL uses a learned example encoder to average 12 labeled examples into a descriptor, then generates all private parameters with a hypernetwork. Source inner updates provide first-order parameter-delta training signals. Separate modality/architecture banks restrict output to predefined linear/CNN spaces. All supervised controls receive the same labels, and development selects FedAvg-plus-fit as the reference. PeFLL gains 4.11/9.48 points on linear/CNN and 6.04/7.54 on fused/acceleration-only clients. ATP uses compatible FedAvg models, learns one adaptation rate per parameter tensor on source-selection clients, and restores the global state before one entropy step per unlabeled target batch of 20. Its architecture gains are −.06/−.05 points, and modality gains are −.53/+.43. This implementation omits custom differentiable BatchNorm-statistic rates because the HAR models do not expose them.

Table S4. UCI HAR controls. Accuracy is %; gains and intervals are pp. Panel A uses the sealed 12-subject study; Panel B uses compatible predefined models and each method’s own reference. PeFLL receives 12 descriptor labels; ATP receives an unlabeled query batch. Panels do not share a baseline.  
A. Selection and architecture sensitivity
<table><tr><td>Policy</td><td>Gain [95% CI]</td><td>Harm</td><td>Linear</td><td>CNN</td></tr><tr><td>fixed joint KD 1</td><td>+3.88 [2.19,5.51]</td><td>16.7%</td><td>-1.73</td><td>+9.49</td></tr><tr><td>disagreement audit, 6 labels</td><td>+3.22 [1.83,4.64]</td><td>8.3%</td><td>-1.08</td><td>+7.52</td></tr><tr><td>disagreement audit, 24 labels</td><td>+4.47 [2.71,6.34]</td><td>8.3%</td><td>-0.93</td><td>+9.87</td></tr><tr><td>same-query oracle (diagnostic)</td><td>+8.03-</td><td>0.0%</td><td>+2.89</td><td>+13.18</td></tr></table>

B. Named compatible slices
<table><tr><td>Method</td><td>Role</td><td>Reference</td><td>Ref. acc.</td><td>Method acc.</td><td>Gain [95% CI]</td><td>Harm</td></tr><tr><td>PeFLL</td><td>shadow</td><td>FedAvg + fit (12 labels)</td><td>45.9%</td><td>49.3%</td><td>+3.38 [0.81, 6.06]</td><td>16.7%</td></tr><tr><td>PeFLL</td><td>test</td><td>FedAvg + fit (12 labels)</td><td>44.4%</td><td>51.2%</td><td>+6.79 [3.35, 9.95]</td><td>16.7%</td></tr><tr><td>ATP</td><td>shadow</td><td>Compatible global model</td><td>42.48%</td><td>41.92%</td><td>-0.55 [-1.39, 0.00]</td><td>83.3%</td></tr><tr><td>ATP</td><td>test</td><td>Compatible global model</td><td>46.02%</td><td>45.97%</td><td>-0.05 [-0.69, 0.59]</td><td>50.0%</td></tr></table>

HeteroFedTTA reconstruction boundary. The linked anonymous implementation was unavailable when this study was built. We therefore reconstruct Algorithm 2 and Eq. 14. On the historical HAR slice, 12 source clients produce base/adapted responses on 754 source-query inputs, with labels excluded from the response loss. Its squared-mean adaptation loss approximates the native batch $\ell _ { 2 }$ norm; the newer FedEMNIST reconstruction uses the native norm. At 300 steps, HAR shadow accuracy is 42.08% for base-response KD, 41.39% before dual-model adaptation, and 37.70% afterward. Extending the compiler to 1,000 steps gives a best difference of −1.34 pp [−4.12,1.41]; at 3,000 steps the compiled bases differ by only .09 points but target adaptation loses 5.53 points. This branch leaves test unopened. It is a bounded compatible-slice audit, not a claim about the published method’s optimal performance.

The original FedEMNIST target-only TTA control selects entropy or fixed-pseudo-label updates, rates $1 0 ^ { - 5 } – 1 0 ^ { - 3 }$ and 1/5/20 steps on 300 shadow writers. On the already evaluated test cohort, accuracy changes from 96.75% to 96.65%, −0.10 pp [−.26,.05]. This historical post-hoc control uses a centralized source bank; the complete-pipeline study below is the primary nearest-method comparison.

## 5. What Public Inputs Enable Compilation?

Shared resources and quantity. The 4,096 CelebA public images come from 3,137 identities outside all federation and evaluation clients, chosen by label-blind path-hash order. Anchor selection and response training do not use task labels The historical teacher writes two class logits for each declared modality subset. Public-response validation uses its own disjoint inputs. The main figure displays the count frontier and controlled blur grid once; complete accuracy, gain, harm, agreement, coefficient and RMSE grids are in the released result files. At 4,096 anchors, teacher agreement is 85.37%, coefficient norm-squared .074 and log-odds RMSE 3.307. The coefficient–gain association is −.462 pooled across budgets but −.012 [−.095,.070] after removing budget/backbone/modality means, and .001 [−.075,.075] at 4,096. It cannot reliably route individual clients.

Support interventions. The pose-restricted CelebA pool is the upper quartile of normalized nose displacement from the eye midpoint within the 3,584-image fit pool. Three seeds sample equal fit counts from broad and restricted pools. Neura models use 336 updates, batches of 32 images with two modality views, fixed late-block/head rates $1 0 ^ { - 4 } / 5 \times 1 0 ^ { - 4 }$ , FP32 cached prefixes and frozen BatchNorm. Response BCE on the common 512 validation images selects checkpoints every 56 updates. Table S5 joins response error, accuracy and harm in one view. EfficientNet at 128 images has uncertain accurac change despite worse KL; all conditions and both signs remain visible.

The short-step head-only control often selects its last checkpoint and is not used to establish representation superiority. After observing this, an explicitly post-hoc closed-form ridge control uses λ = 1, identical FP32 initial features, the same fit anchors and both modality views. On broad support, late blocks exceed this control by 5.52/6.51/6.47 points at 64/128/512 images, with intervals [4.53,6.49]/[5.52,7.49]/[5.49,7.45]. The earlier fixed-ridge pose intervention has different feature handling and local references, so its rows are a separate study. Original head-only and ridge cells, including unfavorable conditions, remain in the numeric export.

Table S5. Equal-count support interventions. Accuracy is %; differences and intervals are pp. Broad/restricted conditions share teacher, clients and training budget within each study. Intervals resample clients after averaging three draws. Restricted means pose-tail on CelebA and four scripts on CREMA-D. Harm columns are broad/restricted; KL difference is broad minus restricted; lower KL is better. Studies use different representations and local references.
<table><tr><td>Study/model</td><td>Fit</td><td>Broad acc.</td><td>Restr. acc.</td><td>Difference [95% CI]</td><td>∆KL</td><td>Harm (%)</td></tr><tr><td>MobileNet blocks</td><td>64</td><td>72.07</td><td>69.01</td><td>3.06 [1.84,4.32]</td><td>-0.0490</td><td>27.4/32.2</td></tr><tr><td>MobileNet blocks</td><td>128</td><td>78.46</td><td>75.42</td><td>3.04 [1.93,4.20]</td><td>-0.0426</td><td>17.8/24.8</td></tr><tr><td>MobileNet blocks</td><td>512</td><td>85.32</td><td>80.94</td><td>4.38 [3.26,5.54]</td><td>-0.0861</td><td>10.4/13.7</td></tr><tr><td>EfficientNet blocks</td><td>64</td><td>75.96</td><td>70.12</td><td>5.83 [4.51,7.21]</td><td>-0.0242</td><td>25.6/35.2</td></tr><tr><td>EfficientNet blocks</td><td>128</td><td>82.19</td><td>82.04</td><td>0.15 [-0.76,1.10]</td><td>+0.0248</td><td>14.1/14.4</td></tr><tr><td>EfficientNet blocks</td><td>512</td><td>87.82</td><td>84.94</td><td>2.89 [1.84,3.94]</td><td>-0.0667</td><td>8.5/13.0</td></tr><tr><td>CelebA fixed ridge</td><td>64</td><td>70.86</td><td>66.26</td><td>4.61 [3.67,5.56]</td><td>-0.0730</td><td>30.7/42.2</td></tr><tr><td>CelebA fixed ridge</td><td>128</td><td>75.05</td><td>71.16</td><td>3.90 [2.92,4.90]</td><td>-0.0604</td><td>22.2/30.0</td></tr><tr><td>CREMA-D fixed features</td><td>64</td><td>53.13</td><td>51.30</td><td>1.83 [-2.41,6.53]</td><td>-0.0001</td><td>100.0/100.0</td></tr><tr><td>CREMA-D fixed features</td><td>96</td><td>52.47</td><td>53.41</td><td>-0.93 [-4.87,2.06]</td><td>+0.0044</td><td>100.0/100.0</td></tr></table>

For CREMA-D, six paired public actors are removed before training 54-client source banks. Five shadow actors select local settings and ten evaluate; only three evaluation actors have audio. The 478-clip public pool reserves 64 response-validation clips; four fixed scripts leave 114 eligible restricted-fit clips. A metadata-only amendment changes budgets from 64/128 to 64/96 before outcomes. Conditions share initialization, normalization and at most 720 updates. The loader materializes public annotations, but labels do not enter selection, loss or checkpointing. At 64 broad anchors, local/teacher/compiler accuracies are 80.68/55.19/53.13%: the dominant 25.49-point deficit is already in supply. This is a small boundary case, not a second positive multimodal confirmation.

Alternative support and response controls. On 40 development identities, pivoted QR selects rank-diverse anchors but reaches 53.75% versus hash-prefix 69.06% at 64 inputs and 67.50% versus 80.00% at 256. Target-nearest selection reaches 63.91/77.97% at those budgets; its .94-point advantage at 1,024 has an interval crossing zero. Soft prototypes reach 65.83% at 4,096 validation anchors; on the ridge confirmation, hard-label ridge is 80.25% versus response ridge 80.85%. The evidence favors the public interface more than a unique soft-logit algorithm. A shadow-selected one-bit quantizer gives 84.91% versus float32 85.28%, then 85.71% on a previously evaluated 300-identity cohort: +22.56 pp over local, with 7.0% harm. Its two-modality digest is 1,024 bytes (1-KB); this is post-hoc compression robustness, not privacy protection or a new prospective cohort.

## 6. Complete-Pipeline Costs and Budget Sensitivity

The comparison reuses three independently trained FedSGD source checkpoints, then forms an original bank, a meta-trained base bank, and a dual-response bank. The latter source stage persists client parameters across full writer sweeps. Each writer takes 20 public digest and 20 labeled meta-revisit steps; both response tables are updated after each sweep. A 32-writer pilot and 100 development writers select source inner rate .1 from .1/.5. Source budgets 1/2/4/8/16 and dual weights .2/.8/1 are chosen on development; the compiler retains inner rate .5. A plateau requires ordinary-KD and full-method developmen accuracy to vary by at most .25 points over the last three budgets, with at least four rounds. One seed reaches 16 rounds without meeting this criterion.

New compilers share initialization, optimizer, minibatch order and 3,584/512 public fit/validation images at 100/200 epochs. Full-method target rates .05/.1/.5, 1/5/20 steps and a no-update option are development-selected. Each adapted model retains the lowest-entropy checkpoint including step zero. Table S6 reports per-architecture sensitivity and the source search. At 100/200 epochs, full-minus-minimal differences are −.11 [−.28,.07] and .04 [−.12,.21] points. At 200 epochs the meta-bank KD ablation gains .15 [.04,.27] over the original bank; adaptation does not inherit an automatic advantage from the bank improvement. Neither zero-crossing intervals nor an epoch cap establish equivalence or convergence.

The common FedSGD stage uses 614,400 labeled presentations. Each additional source round uses 1,280,000 labeled revisit presentations, 1,280,000 public digest presentations and 4,096,000 ensemble evaluations. Search time includes compile

Table S6. Complete-pipeline sensitivity and source effort. Panel A reports 100/200 epochs in each paired entry, retaining architecture-specific harm after seed averaging. Target time excludes I/O and loading. Panel B reports each source seed; the selected checkpoint and the ful search are different costs. Times are sequential simulator measurements, not distributed latency.

A. Private architecture and compiler budget
<table><tr><td>Arch.</td><td>Pipeline</td><td>Accuracy (%)</td><td>Harm (%)</td><td>Compile (s)</td><td>Target (ms)</td></tr><tr><td>MLP</td><td> $\mathrm { O r i g i n a l + K D }$ </td><td>95.87/95.82</td><td>12.0/12.0</td><td>11.8/23.1</td><td>0.71/0.66</td></tr><tr><td>MLP</td><td> $\mathrm { M e t a + K D }$ </td><td>95.74/95.97</td><td>12.7/11.0</td><td>11.0/22.4</td><td>0.66/0.66</td></tr><tr><td>MLP</td><td> $\mathrm { \ M e t a + d u a l }$ </td><td>95.76/95.82</td><td>12.3/12.0</td><td>67.1/132.2</td><td>6.11/6.23</td></tr><tr><td>CNN</td><td> $\mathrm { O r i g i n a l + K D }$ </td><td>98.39/98.33</td><td>2.0/2.3</td><td>12.9/25.7</td><td>0.79/0.77</td></tr><tr><td>CNN</td><td> $\mathrm { M e t a + K D }$ </td><td>98.44/98.49</td><td>2.3/2.7</td><td>12.8/25.3</td><td>0.69/0.69</td></tr><tr><td>CNN</td><td> $\mathrm { \ M e t a + d u a l }$ </td><td>98.29/98.42</td><td>2.7/2.0</td><td>67.9/136.5</td><td>42.88/10.94</td></tr></table>

B. Source-bank construction and search
<table><tr><td>Seed</td><td>source rounds</td><td>FedSGD (s)</td><td>extra source (s)</td><td>budget search (s) plateau</td><td></td></tr><tr><td>81011</td><td>2</td><td>91.6</td><td>5349.6</td><td>12463.9</td><td>yes</td></tr><tr><td>81012</td><td>4</td><td>87.3</td><td>10668.1</td><td>23589.8</td><td>yes</td></tr><tr><td>81013</td><td>1</td><td>85.7</td><td>2656.9</td><td>43954.6</td><td>no</td></tr></table>

trials and development scoring, excluding final validation/test scoring. Original-bank export additionally takes .38/.10/.08 seconds across seeds; regenerated responses exactly match the stored tables. Source computation differs and is charged explicitly. Compilation is reusable for a chosen architecture; adaptation recurs per newcomer. Per-seed harm is retained in the CSV/JSON export, including MLP original-KD rates 12.3/13.7/14.0% at 100 epochs.

FedEMNIST public pixels occupy 3,211,264 bytes; each ten-class float32 response table occupies 163,840 bytes (160 KiB). Uncached original/full handoffs are 3,375,104/3,538,944 bytes, a 4.85% difference, even though response-only payload doubles. CelebA’s JPEG pool occupies 28.2 MB and its two binary-response tables total 64 KiB. No distributed network latency was measured. Onboarding timings exclude data preparation and loading as declared; sequential simulator timings do not measure distributed deployment speed.

## 7. Where Does Client Harm Come From?

For a fixed teacher, $S _ { c } = U ( h _ { c } ) - U ( f _ { c , 0 } ) , C _ { c } = U ( h _ { c } ) - U ( f _ { c } )$ and $G _ { c } = S _ { c } - C _ { c }$ . Inherited harm is $S _ { c } < 0 , G _ { c } < 0 ;$ compiler-induced harm is $S _ { c } \geq 0 , G _ { c } < 0$ . These events partition observed harm. Table S7 refers to the historical loca references in Table S2; it must not be mixed with the strengthened or different-budget endpoints in main Table 2.

Table S7. Historical harm decomposition using each cohort’s original local reference. Supply, loss and gain are pp; harm is a percentage after architecture averaging. All quantities use query labels and are diagnostic.
<table><tr><td>Study</td><td>Supply</td><td>Compile loss</td><td>Gain</td><td>Inherited</td><td>Compiler</td><td>Total</td></tr><tr><td>CelebA ridge</td><td>+23.90</td><td>7.27</td><td>+16.63</td><td>4.0%</td><td>11.3%</td><td>15.3%</td></tr><tr><td>CelebA blocks</td><td>+25.02</td><td>1.71</td><td>+23.31</td><td>4.0%</td><td>2.0%</td><td>6.0%</td></tr><tr><td>CelebA decorrelated</td><td>+10.78</td><td>1.12</td><td>+9.66</td><td>10.3%</td><td>9.3%</td><td>19.7%</td></tr><tr><td>FedEMNIST</td><td>+21.44</td><td>2.59</td><td>+18.85</td><td>0.3%</td><td>1.0%</td><td>1.3%</td></tr></table>

For the decorrelated cohort, the original strict floating-point sign gave 20.7% harm. Treating absolute differences below $1 0 ^ { - 1 2 }$ as ties gives 59/300 (19.7%), split into 31 inherited and 28 compiler-induced cases. The original raw result remains unchanged. On the same binary block cohort, frozen-feature versus updated-block compilation losses are 6.90/1.71 points and compiler-induced harm is 6.3/2.0%. Missing landmarks do not drive this improvement: image-only and image–landmark gains are 24.74/22.03 points, with compiler-induced harm 2.1/1.9%. These paired results must not be compared as interventions against the separate ridge cohort.

On the original FedEMNIST model, per-architecture harm is 6.0% for MLP and 1.0% for CNN, while averaging architectures before checking sign gives 1.3%. All 234 writers with at least 95% teacher agreement have nonnegative observed gain, but this threshold was discovered post hoc. Exact response recovery would set compilation loss to zero while leaving teacher benefit unidentified. None of these observations certifies every future client’s risk.

![](images/44281bc441f77fc42e4f8f6b442055fc828071e388154fd15f070cef7f646c37.jpg)

![](images/2e5daecd7d403401be1d7c7aeb8fc8e496a0bb2b2f9e4564c081f9d9985a02a2.jpg)

![](images/3c41d0927d93b9933a58db5c8e348f61e5a7f2aea07138675051adec908c85b5.jpg)  
Figure S1. Utility, candidate opportunity and private architecture. A: Strengthened local controls versus compilation on separate CelebA and FedEMNIST cohorts. B: Four fixed policies on each dataset. Policies 1–4 denote global, cluster 0, cluster 1 and uniform on CREMA-D; elsewhere they denote KD weights 0.1, 1, 10 and pure KD. These are dataset-specific candidate menus. C: UCI HAR gains by private architecture under fixed KD, 6- and 24-label disagreement audits, and a same-query diagnostic oracle. The oracle is not an allowed deployment method. Each comparison retains its original local reference

![](images/da3c90296c386845b51fbf1264b41ae6e21767a34037cdd1d53580d964b2e20a.jpg)

![](images/487c7b3effddcdb541db5a159614a4e35e507fa0b42192d664e886c73afa5b17.jpg)

![](images/2ea9c5e894a2a07066aed72a03ede33f17dfd471eba10cb1fe58c16648d3f903.jpg)

![](images/743bd262de603a9ec8254c53c2d319e414b7fa57857b5971bb8c786b1faa5409.jpg)  
Figure S2. Complete-pipeline budget sensitivity. Accuracy, client harm, compilation time and target inference time at 100 and 200 compilation epochs. M/C denote pixel-MLP/CNN; Orig, Meta and Dual denote original+KD, meta+KD and meta+dual pipelines. Points are the reported three-seed means. Timing is sequential simulator time and excludes loading and I/O; it is not a parallel deployment latency estimate.

## 8. Visual Comparisons of Reported Results

Figures S1–S3 visualize results already tabulated in the main paper and this supplement. They add no experimental runs.   
Points use the published rounded means; only the paired differences in the main support figure carry confidence intervals.

## 9. Identifiability Proofs and Their Scope

For completeness, let $Z _ { A } ~ \in ~ \mathbb { R } ^ { n \times d }$ contain the augmented private anchor features as rows. Under linear realizability, $t ( x ) = w _ { * } ^ { \top } z ( x ) . \mathrm { I f } z ( x ) = Z _ { A } ^ { \top } \alpha _ { x }$ for every target input, then any wˆ satisfying $Z _ { A } \hat { w } = Z _ { A } w$ <sub>∗</sub> also satisfies

$$
z ( x ) ^ { \top } \hat { w } = \alpha _ { x } ^ { \top } Z _ { A } \hat { w } = \alpha _ { x } ^ { \top } Z _ { A } w _ { * } = t ( x ) .\tag{1}
$$

![](images/d79cb79843cd9815abbb574c98561e6904b187182a76eb55beedeb5d2588e6b1.jpg)

![](images/39a35c3516e16e25cd3a1728b39098f66338e8d74e967aa5bf558f81a773b85a.jpg)  
Figure S3. Equal-count neural public-support comparison. Broad versus pose-restricted public inputs at 64, 128 and 512 fit images. M/E denote MobileNetV3-Small/EfficientNet-B0. Each setting uses 336 neural updates, 512 additional shared validation responses, three anchor draws and the same 270 shadow identities. Panels show accuracy and architecture-specific client harm; the main support figure reports paired accuracy differences with client-bootstrap intervals.

Conversely, if a target $z ( x _ { 0 } )$ is outside the row span of $Z _ { A }$ , the fundamental theorem of linear algebra gives a $v \in \mathrm { n u l l } ( Z _ { A } )$ for which $v ^ { \top } z ( x _ { 0 } ) \ne 0$ . The two heads $w _ { * } + v$ and $w _ { * } - v$ produce identical anchor responses and different target scores, so no response-only compiler can identify which is correct. This establishes necessity and sufficiency.

With imperfect fitting and $z ( x ) = Z _ { A } ^ { \top } \alpha _ { x }$ , Cauchy–Schwarz gives

$$
| z ( x ) ^ { \top } ( \hat { w } - w _ { * } ) | ^ { 2 } \leq \| \alpha _ { x } \| _ { 2 } ^ { 2 } \| Z _ { A } ( \hat { w } - w _ { * } ) \| _ { 2 } ^ { 2 } .\tag{2}
$$

For the empirical ridge diagnostic we use the regularized coefficient

$$
\alpha _ { x } = Z _ { A } ( Z _ { A } ^ { \top } Z _ { A } + \lambda I ) ^ { - 1 } z ( x ) ,\tag{3}
$$

For $\lambda > 0$ , this coefficient generally does not satisfy $Z _ { A } ^ { \top } \alpha _ { x } = z ( x )$ . Defining $\begin{array} { r } { r _ { \lambda } ( x ) = z ( x ) - Z _ { A } ^ { \top } \alpha _ { x } } \end{array}$ and $\Delta w = \hat { w } - w ,$ gives the valid bound

$$
| z ( \boldsymbol { x } ) ^ { \top } \Delta \boldsymbol { w } | \leq \| \alpha _ { x } \| _ { 2 } \| Z _ { A } \Delta \boldsymbol { w } \| _ { 2 } + \| r _ { \lambda } ( \boldsymbol { x } ) \| _ { 2 } \| \Delta \boldsymbol { w } \| _ { 2 } .\tag{4}
$$

We report $\mathbb { E } \Vert \alpha _ { X } \Vert _ { 2 } ^ { 2 }$ only as a regularized geometric diagnostic, not an exact span certificate. The exact bound above uses a coefficient satisfying the representation equality (e.g., a pseudoinverse solution when feasible). Neither diagnostic implies label accuracy under misspecification.

For C classes, apply the same statement to each logit contrast against one fixed reference class. The required span condition remains a property of the newcomer features; the response target becomes a $C - 1$ column matrix. The binary experiment uses the single class-1 versus class-0 log-odds contrast.

Unlabeled risk-order indistinguishability. Fix $P _ { X }$ and all candidate classifiers. In world P, se $Y = f _ { 0 } ( X )$ everywhere, making the local classifier error-free and every genuinely disagreeing candidate no better. In world Q, set $Y = f _ { j } ( X )$ on $D _ { j } = \{ x : f _ { j } ( x ) \neq f _ { 0 } ( x ) \}$ and keep $Y = f _ { 0 } ( X )$ elsewhere. The marginal inputs and all observable predictions are identical in the two worlds. If $P _ { X } ( D _ { j } ) > 0$ , selecting $f _ { j }$ has strictly negative gain in P and strictly positive gain in Q. Since the selector sees identical evidence, it cannot guarantee safe nontrivial transfer over unrestricted label conditionals. This argument requires disagreement on positive target mass, not merely different parameter values. The anchor-span proof and regularized residua are given above.

Conditional mean-gain concentration. Condition on the frozen bank and policies F. For independently drawn same-stratum shadow clients and fixed candidate k, gains lie in $[ - 1 , 1 ]$ , so Hoeffding gives

$$
\operatorname* { P r } ( \bar { G } _ { k } - \mathbb { E } [ G _ { k } ( C ) \mid \mathcal { F } ] \ge \epsilon \mid \mathcal { F } ) \le \exp ( - m \epsilon ^ { 2 } / 2 ) .
$$

A union bound over K fixed candidates and $\epsilon = \sqrt { 2 \log ( K / \delta ) / m }$ yields the stated simultaneous bound. Candidate or bank selection using the same shadow labels invalidates this fixed-policy argument unless independently accounted for. The guarantee concerns population mean gain under the declared distribution, not each client or an arbitrarily shifted future population.

## 10. Related Interfaces and Reproduction Entry Points

Table S8 records task-level applicability rather than assigning artificial failure scores to methods whose required interface is absent. A check means that the cited work explicitly treats the capability; parentheses mark a restricted form. “Frozen” means that onboarding does not update the completed federation. “Private output” means that improvement is realized in a newcomer-side model rather than only a server/global model.

Table S8. Novelty and applicability boundary.
<table><tr><td>Method</td><td>unseen</td><td>frozen</td><td>multimodal</td><td>unseen arch.</td><td>private output</td></tr><tr><td>OD-PFL [1]</td><td>√</td><td>√</td><td>一</td><td></td><td>(predefined)</td></tr><tr><td>UPFL/HeteroFedTTA [15]</td><td>√</td><td>√</td><td>1</td><td>√</td><td>√</td></tr><tr><td>FLowDUP [16]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(predefined)</td></tr><tr><td>EPFL [7]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(predefined)</td></tr><tr><td>PeFLL [12]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(predefined)</td></tr><tr><td>pFedHN [13]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(predefined)</td></tr><tr><td>HyperFLoRA [8]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(adapter)</td></tr><tr><td>HyperFedZero [17]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(predefined)</td></tr><tr><td>ATP [2]</td><td>√</td><td>√</td><td>一</td><td>一</td><td>(global copy)</td></tr><tr><td>TFL [9]</td><td>√</td><td>√</td><td>一</td><td>一</td><td></td></tr><tr><td>TGFed [11]</td><td>√</td><td>√</td><td></td><td>一</td><td>(compatible)</td></tr><tr><td>COTE [18]</td><td>√</td><td>√</td><td>(V-L)</td><td>一</td><td>(prompt)</td></tr><tr><td>DP2FL [3]</td><td>√</td><td>(inference)</td><td>(V-L)</td><td></td><td>(prompt)</td></tr><tr><td>HAFL-GHN [6]</td><td>√</td><td>√</td><td>一</td><td>(graph family)</td><td>√</td></tr><tr><td>FedMBridge [4]</td><td>一</td><td>一</td><td>√</td><td>(training clients)</td><td>√</td></tr><tr><td>FedAFD [14]</td><td>一</td><td>一</td><td>√</td><td>(training clients)</td><td>√</td></tr><tr><td>FedMD [5]</td><td>一</td><td>一</td><td>一</td><td>√</td><td>√</td></tr><tr><td>pFedDSH [10]</td><td>√</td><td></td><td></td><td>一</td><td>(hypernetwork)</td></tr><tr><td>FEDNBO task</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

PeFLL and ATP are executed on their compatible interfaces in Section 4; HeteroFedTTA is evaluated through explicitly scoped reconstructions. Response distillation itself is established prior work. The contribution is the post-federation benchmark, its information-access contract and controlled evidence about benefit and portability. The strongest confirmations address visual architecture portability; image-derived landmarks do not establish independent-sensor multimodal success.

Reproduction and full numeric records. Run python3 fednbo.py verify for registered hashes, exact CelebA cohort identities and role separation; verify --data checks local assets. run ... --dry-run prints registered training commands, and results locates current summaries. The README maps cohorts and allowed information. Main and supplementary tables are generated from recorded evidence; paper cvpr/compose evidence.py --check verifies the compact displays. The directory paper\_cvpr/evidence\_tables/ contains machine-readable full condition grids, contrasts and the previous table archive. Existing per-client JSONL records retain individual seeds, interfaces and unfavorable results.

The primary evidence resides in results/revision\_20261007/; fixed-feature support and source-budget audits in results/acceptance\_revision\_20261008/; complete-pipeline and neural-support records in results/ completion\_revision\_20261009/. The registry binds the original policy studies, confirmation cohorts and compatible baselines. These grouped entry points replace a repeated list of every intermediate file. The original compiler source was recovered from its creation patch and matches its frozen hash; later quantization extensions are kept separately. Full neural reproduction requires separately prepared public datasets and weights. The evidence bundle verifies records and code without redistributing raw data or checkpoints.

Ethics and data governance. The study uses public benchmark data and previously extracted features. TCGA site identifiers define experimental clients and do not assess institutions. IDC/ILC classification is a methodological probe, not a clinica diagnostic system. Response release is assumed permitted; differential privacy and leakage protection are outside the evaluated contract.

## References

[1] Ohad Amosy, Gal Eyal, and Gal Chechik. Late to the party? on-demand unlabeled personalized federated learning. In WACV, pages 2184–2193, 2024. 9

[2] Wenxuan Bao, Tianxin Wei, Haohan Wang, and Jingrui He. Adaptive test-time personalization for federated learning. In NeurIPS, 2023. 9

[3] Ying Chang, Xiaohu Shi, Xiaohui Zhao, Zhaohuang Chen, and Deyin Ma. Dual prompt personalized federated learning in foundation models. Scientific Reports, 15:28026, 2025. 9

[4] Jiayi Chen and Aidong Zhang. FedMBridge: Bridgeable multimodal federated learning. In ICML, pages 7667–7686, 2024. 9

[5] Daliang Li and Junpu Wang. FedMD: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581, 2019. 9

[6] Or Litany, Haggai Maron, David Acuna, Jan Kautz, Gal Chechik, and Sanja Fidler. Federated learning with heterogeneous architectures using graph hypernetworks. arXiv preprint arXiv:2201.08459, 2022. 9

[7] Yuange Liu, Daobin Luo, Weishan Zhang, Wei Jiang, Chaoqun Zheng, Xiaohui Sun, Yuru Liu, Qiao Qiao, Tao Chen, Hongwei Zhao Su Yang, and Fei-Yue Wang. EPFL: Toward elastic personalized federated learning with seamless client joining and quitting. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 55(12):9324–9338, 2025. 9

[8] Qikai Lu, Di Niu, Mohammadamin Samadi Khoshkho, and Baochun Li. HyperFLoRA: Federated learning with instantaneous personalization. In SDM, pages 824–832, 2024. 9

[9] Mengmeng Ma, Tang Li, and Xi Peng. Beyond the federation: Topology-aware federated learning for generalization to unseen clients. In ICML, pages 33794–33810, 2024. 9

[10] Thinh Nguyen, Le Huy Khiem, Van-Tuan Tran, Khoa D. Doan, Nitesh V. Chawla, and Kok-Seng Wong. Onboarding without forgetting: Hypernetwork personalization with data-free replay for personalized federated learning. In CVPR Findings, 2026. 9

[11] Ke Niu, Jiuyun Cai, Wenjuan Tai, Yijie Pan, and Kaize Shi. TGFed: Transferability-guided federated learning for unseen client adaptation. Expert Systems with Applications, 299:129949, 2026. 9

[12] Jonathan Scott, Hossein Zakerinia, and Christoph H. Lampert. PeFLL: Personalized federated learning by learning to learn. In ICLR, 2024. 9

[13] Aviv Shamsian, Aviv Navon, Ethan Fetaya, and Gal Chechik. Personalized federated learning using hypernetworks. In ICML, pages 9489–9502, 2021. 9

[14] Min Tan, Junchao Ma, Yinfu Feng, Jiajun Ding, Wenwen Pan, Tingting Han, Qian Zheng, Zhenzhong Kuang, and Zhou Yu. FedAFD: Multimodal federated learning via adversarial fusion and distillation. In CVPR, pages 3400–3409, 2026. 9

[15] Tiandi Ye, Cen Chen, Yinggui Wang, Xiang Li, and Ming Gao. UPFL: Unsupervised personalized federated learning towards new clients. In SDM, pages 851–859, 2024. 9

[16] Hossein Zakerinia, Jonathan Scott, and Christoph H. Lampert. Federated learning with unlabeled clients: Personalization can happen in low dimensions. In ICML, 2026. 9

[17] Yuhao Zhou, Jindi Lv, Yuxin Tian, Dan Si, Qing Ye, and Jiancheng Lv. Deploying models to non-participating clients in federated learning without fine-tuning: A hypernetwork-based approach. In ICLR, 2026. 9

[18] Rui Zhu, Liang Bai, Yanming Guo, Yirun Ruan, Tianyuan Yu, and Zhihe Lu. Controllable federated prompt learning at test time. In CVPR, pages 39455–39465, 2026. 9