# Measuring Robustness and Efficiency in a Connectome-Constrained Fly Visual System Model: Linking an Adversarial and a Perceptual Illusion

Alexandre Garcia-Duran1,2,\* Alejandro Rodriguez-Garcia3,\* Manuel Molano-Mazón1,2 Alexandre Hyafil1 Srikanth Ramaswamy³ Janne K. Lappalainen⁴

1Centre de Recerca Matemàtica, Barcelona, Spain 2Universitat Politècnica de Catalunya, Barcelona, Spain 3Newcastle University, Newcastle upon Tyne, United Kingdom 4Allen Institute, Seattle, WA, United States janne.lappalainen@alleninstitute.org These authors contributed equally.

## Abstract

Biological visual systems compute reliably under tight metabolic and wiring constraints, whereas artificial vision systems consume far more energy and can be sensitive to adversarial perturbations. Whether networks with brain-like wiring are more robust at lower computational cost is unclear. We address this by benchmarking a task-optimized, continuous-time recurrent neural network of the fly visual system constrained by its synaptic connectome (FlyNet) against artificial neural networks. We compare it against CNNs, RAFT, and continuous-time recurrent neural networks without detailed connectome constraints (sCTRNN, CTRNN) on a collision-detection task within a differentiable 3D rendering pipeline. FlyNet is adversarially vulnerable but, relative to its compute, more robust to common video corruptions than RAFT, while using up to five orders of magnitude fewer FLOPs. The CTRNNs remain more robust than FlyNet across almost all tested perturbations, suggesting that connectome constraints benefit efficiency, through their low edge count, over robustness. Recurrence improves robustness to white-box attacks in RAFT and temporal context improves robustness to common corruptions. FlyNet's position and velocity representations degrade under adversarial textures but recover under counterfactual textures that restore correct classification while staying equally far from training, ruling out unfamiliar textures as the cause of failure. The reverse-phi illusion similarly disrupts these representations and inverts the same low-level motion statistic as the adversarial textures. Optimizing this input statistic alone, without access to the network, suffices to induce classification failure, attributing both perturbations to the same circuit vulnerability. Connectomeconstrained models can thus surface adversarial failure mechanisms as testable hypotheses about biological perceptual vulnerabilities.

## 1 Introduction

Biological neural systems compute reliably despite stochastic components and variable inputs [1], and do so under tight metabolic and wiring constraints [2, 3]. But these systems are not perturbation-proof: Classical visual illusions expose the limits of computational mechanisms of biological perception [4]. Artificial neural networks, by contrast, consume vastly more energy [5] and can fail under various perturbations [6–9]. Do artificial and biological neural networks specialized for the same task share the same failure modes? Linking both has been hard because the systems one can attack with gradients are not biologically wired. To our knowledge, no perceptual illusion has yet been tied to an adversarial stimulus in the same system at the level of a computation both exploit.

Robustness — the ability to maintain reliable computation under perturbations — of a biological neural network can now directly be evaluated with FlyNet [10], a fly visual system model constrained by real synaptic wiring. It was trained on motion detection, an ecologically essential task for the fly, using the Sintel optical flow dataset [11] and recapitulates single-neuron response properties across three decades of experimental research, without training on that data. In artificial networks, robustness is linked to specific architectural features — recurrence [12–14], global processing [15, 16], and continuous-time dynamics [17, 18]. FlyNet has little of these: it uses continuous-time dynamics but has sparse recurrence and primarily local connectivity, reflecting the predominantly feedforward organization of the fly optic lobe. Any robustness it exhibits might therefore arise from properties other than recurrence or global integration, possibly shaped through evolution under tight metabolic constraints [2, 3]. Does biological wiring contribute to robustness, to efficiency, or to both?

We designed a behaviorally grounded, looming-based collision-detection task to evaluate robustness in both connectome-constrained networks and conventional optic-flow models. We implemented the task in a fully differentiable 3D rendering pipeline, in which textured objects follow parametric trajectories toward the observer. Rather than optimizing perturbations in pixel-space — where they lose effectiveness under geometric transformations [19, 20] — we optimize textures directly in 3D scene space, ensuring physical consistency across frames as objects scale during looming. We evaluate models under both adversarial attacks and common video perturbations (e.g., flicker, fog), and pair each robustness measurement with computational cost in FLOPs per inference to characterize robustness relative to efficiency.

The main contributions of our work are:

• A differentiable 3D rendering pipeline for a binary collision detection task (approaching objects that hit or miss the observer) in which adversarial attacks optimize object textures for differentiable motion-perception models.

• A direct comparison of gradient-based adversarial attacks and the reverse-phi illusion on the same connectome-constrained, task-optimized model of the fly visual system (FlyNet).

• Using counterfactual restoration, we find textures equally far from the training distribution that recover classification accuracy and position and velocity readouts, linking failure to motion-code disruption, not unfamiliar textures.

• We identify the computation both perturbations exploit: Adversarial textures and reverse-phi both invert the Hassenstein-Reichardt correlation between radially neighboring inputs, and optimizing this statistic alone is sufficient to induce or reverse classification failure.

• We benchmark robustness-efficiency trade-offs across architectures with different inductive biases (recurrence, global signals, continuous-time dynamics) under adversarial attacks and common video perturbations.

• We find that FlyNet and sCTRNN have higher robustness per order of magnitude of compute than every other architecture on common video corruptions of the collision-detection task.

## 2 Background

Visual motion detection in the fly and FlyNet In the Drosophila optic lobe (Supp. Fig. 1a), visual motion detection is decomposed into parallel ON and OFF pathways, mediated by the T4 and T5 neurons, respectively [21]. These neurons act as local moving-edge detectors, with T4 and T5 selective for ON and OFF edges, each with subtypes (a, b, c, d) selective for the four cardinal directions of motion, forming a distributed basis for local motion encoding [22]. These local directional signals are subsequently integrated by Lobula Plate-Lobula Columnar neurons, such as LPLC2, which are selectively tuned to radial expansion patterns associated with looming, and then project to giant fiber descending neurons driving escape behaviors such as jumping [23–25]. This hierarchical integration encodes collision-relevant variables —including radial expansion, object position, and velocity— establishing a shared representational substrate from which multiple taskrelevant variables can be decoded. There has been extensive computational work applying structural constraints from the fruit fly on neural networks [10, 26–28]. Here we use FlyNet from [10], which provides a computational instantiation of the early motion detection circuit (excluding LPLC2, and giant fiber neurons), modeling neurons as passive-point units governed by

![](images/ad033240c56b7c819b0076b0c0b6fa69a1ed8fda4c866dbb04e6a66428250143.jpg)  
Figure 1: Fully differentiable 3D geometric framework for motion-based collision detection. (a) Processing differentiable pipeline. (b) Geometric setup of looming stimuli, parameterized by polar coordinates (R, θ, z0). Bottom: example rendering in the image frame $F _ { t }$ and corresponding fly visual input $V _ { t } .$ (c) Generated dataset: example trajectories, shapes and textures. Examples of hit and miss trajectories in 3D space and their temporal distance profiles.

$$
\tau _ { i } \dot { h } _ { i } = - h _ { i } + \sum _ { j } w _ { i j } f ( h _ { j } ) + h _ { i } ^ { \mathrm { r e s t } } + l _ { i } ,\tag{1}
$$

where synaptic weights $w _ { i j }$ are constrained by connectomic measurements while neuronal and synaptic parameters are optimized for motion detection using an optic flow dataset (see App. A.1.1).

Visual motion illusion in the fly: Reverse-phi Biological visual systems estimate motion accurately under natural conditions, but illusory stimuli that deviate from natural scene statistics elicit systematic perceptual errors, reflecting adaptations of neural circuitry to natural environments [4]. A canonical example is the reverse-phi illusion, when the contrast of a moving pattern is inverted between successive frames —bright regions become dark and vice-versa— the perceived direction of motion reverses, in humans [29–31], macaques [32], and flies [31, 33]. Flies, when presented with a reverse-phi stimulus, turn their body in the direction opposite to the physical motion, a reflex that reflects the inversion of direction-selective T4/T5 signals [33-36].

Adversarial attacks in optical flow Neural networks are vulnerable to small, targeted perturbations that induce misclassification [6]. Such perturbations can be designed under worst-case gradient-based attacks (PGD [8], CW [37]), gradient-free methods (NES [38]), and common video perturbations such as noise, flicker, and fog [9, 39]. Such adversarial examples often transfer across models, revealing shared vulnerabilities [40]. In optical flow specifically, patch-based adversarial attacks can severely corrupt motion estimation [41, 42], and global norm-constrained perturbations show that state-of-the-art networks are particularly vulnerable [43]. However, pixel-space and patch-based perturbations often lose effectiveness under minor geometric transformations [19, 20] as they do not capture how perturbations evolve under coherent object motion. Optimizing textures over 3D objects addresses this in still images [44]. In optical flow, adversarial particles rendered consistently in 3D degrade methods that are robust to per-pixel attacks [45], but adversarial textures on moving objects remain unexplored.

## 3 Methods

## 3.1 A differentiable 3D rendering pipeline for a looming motion perception task

We assess motion perception as a binary decision task in a visual looming paradigm, where an approaching object must be classified as hit or miss [46]. This task is a canonical, behaviorally grounded paradigm in Drosophila [25, 47]. To generate stimuli, we develop a differentiable 3D rendering pipeline implemented in pytorch3d [48], which provides full control over scene geometry and ensures that physical perturbations evolve consistently with the underlying dynamics across time (Figure 1). We render grayscale sequences at $4 1 6 \times 4 1 6$ pixels and 100 Hz, and, when required, map them onto a hexagonal visual sampler before feeding them to each model. The resulting model activity is then used by a Multi-Layer Perceptron (MLP) decoder to produce the final decision (Figure 1a). All stimuli are defined in the reference frame of the fly-observer, modeled as a pointparticle camera fixed at the origin. We initialize objects at distance $z _ { \mathrm { 0 } }$ from the origin, with positions parameterized by polar coordinates (R, θ) in the plane orthogonal to the z-axis, enabling isotropic sampling of trajectories across the visual field (Figure 1b). Each trajectory consists of a linear translation in 3D space over duration $T _ { \mathrm { t r a j } } .$ , directed either toward the origin (hit) or toward a nearby offset endpoint (miss). This construction yields controlled deviations from collision while preserving comparable motion statistics across conditions. The resulting trajectories span angular sizes from ${ \sim } 5 { - } \bar { 1 } 1 0 ^ { \circ }$ , with peak angular expansion speeds of up to $1 , 4 2 0 ^ { \circ } / \mathrm { s } ,$ consistent with the looming-escape literature: 5–90° [25] and $6 5 0 { - } 2 , 1 0 0 ^ { \circ } / \mathrm { { s } } \ [ 4 7 ]$ . We define each stimulus by its trajectory parameters $( R , \theta , z _ { 0 } , T _ { \mathrm { t r a j } } )$ and object appearance parameters, including shape {Sphere, Cube} and texture $X \in$ {Black, White, Checkerboard, Wavy, Random}, with $X \in \mathbb { R } ^ { 1 2 8 \times 1 2 8 }$ (Figure 1c). Random textures are generated from pixelwise uniform noise, offset by a uniformly sampled constant to keep the local mean different from background.

## 3.2 Architectures

Continuous-time recurrent neural networks FlyNet is derived from the connectome-constrained recurrent model of the Drosophila visual system [10], with 45,669 neurons and 1.51M anatomically constrained edges, trained on Sintel optical flow [11] with only 734 free parameters. We edited the weights of two pretrained model instances to create a hybrid model, exhibiting accurate direction selectivity for all T4 and T5 subtypes (App. A.1.1, Supp. Fig. 1). As a control, we evaluate CTRNN, a FlyNet variant with the same dynamics, input, and cell-type connectivity, but without single-cell connectivity, synapse counts and synapse signs from the connectome, trained under the same Sintel protocol. It has 7.51M edges, connecting cells of the 604 connected cell-type pairs through isotropic convolutional connectivity over a hexagonal kernel of 19 columns (five columns across), yielding 11,593 free parameters [10]. As an edge-count-matched control, we additionally evaluate a sparsitymatched CTRNN variant (sCTRNN). Starting from the trained CTRNN, we untied the shared filter weights, pruned per-edge by the mean absolute injected current, down to FlyNet's 1.51M connections, and fine-tuned on Sintel. It recovered optical-flow performance to an EPE 2% below FlyNet's on held-out Sintel samples. All models here are evaluated using instantaneous T4/T5 responses (5,768 units) as decoder input (see App. A.1.1, A.1.2, and A.1.3 for details). Their dynamics follow Eq. 1, integrated using Euler with $d t = 0 . 0 1$ . They receive as input the luminance of the visual stimulus sequence, sampled to a regular hexagonal lattice of 721 columns.

Optical flow networks We evaluate four feedforward CNN baselines following a FlowNet-inspired architecture [49] (\~414k parameters), pretrained on Sintel and using the same hexagonal visual sampling as FlyNet. The vanilla CNN (vCNN) processes 2 consecutive frames using local receptive fields, while its global variant (gCNN) incorporates channel-wise global pooling into the second hidden layer (Supp. Fig. 3) to probe the role of global integration. We also train two vCNN variants using 5 and 10 input frames, respectively, while keeping the remaining architecture unchanged; to assess the effect of temporal context. For all CNN variants, features are extracted from the final hidden layer (7,688 activations). As a higher-performance baseline, we evaluate RAFT [50], pretrained and fine-tuned on Sintel (\~5.3M parameters). We extract pooled features from the final hidden representation (6,802 activations) and vary recurrent iterations (1, 6, 12) to probe the role of recurrence. We also evaluate RAFT-small (RAFTsm, \~1M parameters) as a lower-parameter baseline. These models receive the original 416 × 416 luminance frames replicated across three channels. To control for differences in input representation and training regime, we additionally trained RAFT hex-matched variants exclusively on Sintel using the same hexagonally sampled input as FlyNet and the CNN baselines (hexRAFT; Supp. Fig. 4).

## 3.3 Optimizations and attacks

Decoders Task-specific MLPs are trained on frozen network representations on \~600,000 samples from 6,000 trials, and tested on \~40,000 held-out samples from 400 trials. Each trial contributes

50–150 timepoints (100 Hz, durations 0.5–1.5 s), treated as independent samples. We train a binary hit/miss classifier (512 hidden units, tanh) and a continuous position/velocity regressor (128 – 16 hidden units, ReLU) (App. A.7, Supp. Fig. 5, 6, 7, 8). Both trained with Adam [51] $( \mathrm { l r } = 1 0 ^ { - 3 }$ batch size 1024). Our decoder architecture is inspired by how the fly visual system is organized for collision-detection: The fly's LPLC2 neurons detect radial expansion patterns and, by nonlinearly integrating dendritic inputs from T4 and T5 neurons (spatially), trigger the animal's escape response [24]. Since this mechanism integrates spatial patterns, we used a nonlinear, spatial-only decoder architecture to abstract LPLC2 computation.

Adversarial optimization We optimize texture X on hit trials to suppress the decoder's predicted probability of a collision, $p ( \mathrm { h i t } )$ . As in targeted adversarial optimization $[ 6 , 3 7 ]$ , the loss combines the cross-entropy $\mathcal { L } _ { C E }$ toward the target label with an $L _ { 2 }$ penalty on the distance to a baseline texture $X _ { \mathrm { i n i t } } \mathrm { : }$

$$
\mathcal { L } ( X ) = \mathcal { L } _ { C E } ( X , y _ { \mathrm { t a r g e t } } = 0 ) + \lambda _ { \mathrm { i n i t } } \Vert X - X _ { \mathrm { i n i t } } \Vert _ { 2 } .\tag{2}
$$

The resulting texture, $X _ { \mathrm { i l l u s i o n } } .$ , minimizes the above loss.

Counterfactual restoration We find the nearest texture to $X _ { \mathrm { i l l u s i o n } }$ that recovers correct classification, akin to optimization-based counterfactual explanations [52], minimizing:

$$
\mathcal L ( X ) = \mathcal L _ { C E } ( X , y _ { \mathrm { t a r g e t } } = 1 ) + \lambda _ { \mathrm { i n i t } } \Vert X - X _ { \mathrm { i l l u s i o n } } \Vert _ { 2 } .\tag{3}
$$

Both stages use Adam [51] with learning rate 0.01 and $\lambda _ { \mathrm { i n i t } } = 0 . 0 2$

Hassenstein-Reichardt looming detector (HR) Given $C _ { i } ( t ) = V _ { t } ^ { i } - \bar { V } ( t )$ the input at hexal i minus the mean over all hexals, and $\mathcal { P }$ the set of radially neighboring hexal pairs $( i , { \bar { j } } )$ , with i the inner and $j$ the outer one, we compute the HR correlation [53] for a given delay τ as (Supp. Fig. 16)

$$
g ( \tau ) \equiv g ( X , \tau ) \ = \ \big \langle C _ { i } ( t - \tau , X ) C _ { j } ( t , X ) \ - \ C _ { j } ( t - \tau , X ) C _ { i } ( t , X ) \big \rangle _ { t , ( i , j ) \in \mathcal { P } } \ ,\tag{4}
$$

where $g ( \tau ) > 0$ indicates looming, $g ( \tau ) < 0$ contraction, and $g ( \tau ) = 0$ static or non-directional input. Note that the computation depends only on the input, making it agnostic to the model. The optimization process finds the texture that minimizes or maximizes Eq. 4.

Adversarial attacks All attacks optimize the logit of the object texture, on 100 time-steps hit trajectories with Black initial texture. For PGD [8], we use 100 steps with step size $\varepsilon / 1 0 ,$ an adaptive step-size scheme [54], 2 random restarts, and 15 independent attacks per condition; resulting textures are applied directly to other models to assess transfer attacks. For CW [37], we minimize a distance plus misclassification objective. For NES [38], we estimate gradients via Gaussian perturbations. See App. B.6.1 for details.

Video perturbations We test six common video perturbations [9, 39]: additive white noise (AWN) on the texture, AWN on the frames, shot (Poisson) noise, frame flickering, motion jitter, and contrast reduction (fog), with noise power matched across architectures (see App. B.6.2 for details; Supp. Fig. 24). AWN, shot noise, flickering, and motion jitter are averaged over 10 trajectories with 10 noise realizations each, fog over 20 trajectories.

Robustness R For each attack, we sweep a parameter defining the magnitude of the perturbation and define robustness as the area under the accuracy curve, which we call R [55].

## 4 Results

## 4.1 Adversarial failures arise from disrupted motion representations

FlyNet maintains hit/miss classification accuracy under common video corruptions – additive white noise, shot noise, flicker, motion jitter, and contrast reduction (Fig. 2a; Supp. Fig. 23, 24) – but accuracy plummets under optimized adversarial attacks targeting the looming object's texture (Fig. 2b; Supp. Fig. 18, 19, 21, 22). We asked what the adversarial perturbations are exploiting. Do they disrupt the motion representations that FlyNet was trained to compute, or do they push the input outside the training texture distribution?

a  
![](images/f9ad135fe050515574051fa250ccb0222e39663d4083c2c29caf524f11d1b39a.jpg)

b  
![](images/30f14b477059bc341dce3ae93cfed5f8528548b7748d1ef29b0282f67afbfe3f.jpg)

C  
![](images/2f904ce6f2d21370fc1f53b452e0a02c21eb2da1355a293fe85e870d8b9a781f.jpg)

d  
![](images/40a2a47a098df51749e5efee9455ab407d0485c34ea28f7a9cac8878c16326de.jpg)

![](images/08b29cc22b1509162d21afd4313321b1efff515084ac8fcb3a608f8cca18b281.jpg)  
Figure 2: FlyNet is robust to flicker but sensitive to adversarial and reverse-phi attacks. (a) p(hit) under flicker $( \varepsilon = 0 . 0 5 )$ and accuracy vs. flicker strength (10 trajectories × 10 noise samples). (b) p(hit) under an $L _ { \mathrm { 2 } } { \mathrm { - } } \mathrm { P G D }$ adversarial texture $( \varepsilon = 2 0 0 0 )$ and accuracy vs. perturbation size (15 trajectories × 2 restarts). (c) p(hit) under reverse-phi $( T = 0 . 3 3 0 s )$ and average $p ( \mathrm { h i t } )$ vs. contrastinversion period (50 trajectories). (d) Position and velocity decoding errors under reverse-phi vs. original texture $( T = 0 . 3 3 0 s ;$ 50 trajectories). Gray regions indicate contrast-inversion windows; error bars and shaded curves denote standard error of the mean (SEM).

Reverse-phi disrupts motion representations with familiar textures Reverse-phi is a known illusion in real flies that disrupts motion representations. It inverts the contrast between successive frames, evoking illusory reversed motion through inversion of T4/T5 direction-selective signals [33–35]. Using familiar textures, we constructed reverse-phi looming stimuli by inverting the object's contrast every T seconds. Under this stimulus, p(hit) drops sharply at each contrast inversion and recovers between them (Fig. 2c), mirroring the optomotor inversion seen in real flies [33]. Position and velocity decoded from FlyNet activity show the same dependence: decoding errors spike at each inversion (Fig. 2d) and depend on T. Short T gives the network too little time to recover, long T returns both p(hit) and decoding errors toward baseline (Supp. Fig. 9). Reverse-phi establishes that motion representations can be disrupted without any input-texture distribution shift.

Counterfactual restoration rules out texture novelty as the cause of adversarial failure To test whether adversarial failure arises from the same kind of within-distribution disruption, we optimized a texture to suppress p(hit) on a hit trajectory (Eq. 2), then optimized the nearest texture that recovers correct classification (Eq. 3). The restored texture remains as far from the training distribution as the illusion texture throughout optimization (Supp. Fig. 11). If unfamiliar textures were the cause of failure, restored textures should disrupt accuracy as strongly, which they don't. Position and velocity readouts recover alongside p(hit) (Fig. 3a,b; Supp. Fig. 12) consistent with the adversarial restoration recovering the same motion code that reverse-phi disrupts. Single T4/T5 traces show that the illusion texture shifts activity in the same direction as reverse-phi during the inversion phase, consistent with both perturbations engaging T4/T5 motion detectors in the same direction (Supp. Fig. 14). To quantify this effect at the population level, we compared responses across all 5,768 T4/T5 neurons in a 330-ms window aligned to reverse-phi contrast inversions $( n = 3 0$ trajectories). Activity induced by the adversarial illusion was the closest to reverse-phi's, compared to both the original and the restored conditions (all metrics $p < 0 . 0 0 5 ;$ Supp. Fig. 15; see App. B.4), suggesting that adversarial textures and reverse-phi may exploit similar computational vulnerabilities.

Inverting a second-order motion statistic suffices to induce failure We hypothesized that both perturbations act on the same second-order motion statistic: The delayed correlation between neighboring inputs that Hassenstein-Reichardt correlators read out [53], to which T4/T5 direction selectivity is sensitive [34], and which reverse-phi is known to invert [33]. To test this, at each step of counterfactual restoration, we computed the HR correlator (Eq. 4). We found that it closely tracks the motion decoding errors (Fig. 3c; Supp. Fig. 17a). Supporting our hypothesis, when the network is maximally fooled, HR reaches values close to the ones from reverse-phi (Fig. 3c, Supp. Fig. 16b). To test whether disrupting this statistic is sufficient for failure, we separately minimized and maximized $g ( \tau = 8 \mathrm { { f r a m e s } = 8 0 \mathrm { { m s } ) ( E q . 4 ) } }$ , without access to the network. Under minimization p(hit) drops $\mathrm { t o } \sim 0 . 3$ , whereas under maximization it stays above 0.8 (Fig. 3d; Supp. Fig. 17b, 13). Manipulating this statistic alone is thus sufficient to induce the failure, while increasing it leaves classification intact suggesting that the adversarial attacks indeed exploit this canonical, well-studied motion statistic.

![](images/38e7a9075d891b746a3b86310773f07e82df1320e8820b2c3b1a39a63305b5bc.jpg)

![](images/c8d604b4364dbc0dfb77824546d24c20a35a01fbd27cc335020d8c340772d81e.jpg)

b  
![](images/ec058bad7e77fd579655a01534f5d0a045144e36a59e5fa9e4ba87c9024a7ffd.jpg)  
C

![](images/252cbc345b2ed14018712a0d8ba71f39bc00cc00daa45b539cf8dd579b3c11b7.jpg)

![](images/29e7a945e33d16bf1deb8d23f9b406a58a303ccd48019c15dc9dc3cc05e746a4.jpg)

![](images/ccc721747c369791d161d83f7378f422b5c01a4aa1d4db1dc212ff1228b59d67.jpg)

d  
![](images/3034fb2a263fea6a64ac6a2fce360550fc14080b48b06fab958edc72cce0bd72.jpg)

![](images/377a548d346526788f842926ef276f7582ead1c75ddf9d19580711ee5f1cdc69.jpg)  
Figure 3: A static adversarial texture and a dynamic illusion exploit the same low-level motion computation. (a) p(hit) and position and velocity decoding errors across adversarial optimization and counterfactual restoration (30 trajectories). Stars mark initial (black), illusion (red), and restored (blue) textures; texture evolution is shown above. (b) Position and velocity decoding errors at the three counterfactual restoration checkpoints. (c) HR correlator $g ( \tau = 8 )$ (Eq. 4) across counterfactual restoration steps (left), and at the three checkpoints (right). (d) Minimization (left) and maximization (right) of $g ( \tau = 8 )$ (top) and corresponding p(hit) (bottom) across optimization steps. Shaded area represents SEM; asterisks indicate significance from paired t-test $( { } ^ { * * } \colon p < 0 . 0 1 , { } ^ { * * * } \colon p < 0 . 0 0 1 )$

Adversarial failures in FlyNet are therefore not explained by unfamiliar texture statistics. Adversarial textures and reverse-phi invert the same second-order correlation and shift T4/T5 population activity in the same direction, attributing both perturbations to a common vulnerability in the motion representations of FlyNet.

## 4.2 Robustness profiles are shaped by structural characteristics

FlyNet's architecture is predominantly local and feedforward, reflecting the columnar organization of the fly optic lobe [56]. This is at odds with what has been typically used in ANNs to provide robustness: recurrence and global integration [12, 13, 15–18]. We asked whether these ingredients are required for robustness in general, and which best account for FlyNet's robustness profile.

Global integration helps conditionally We compared two feedforward CNN baselines that differ only in global pooling: a vanilla CNN (vCNN) with purely local receptive fields, and a variant with an additional channel-wise global pooling pathway that injects a spatially averaged scene representation into the network (gCNN; Supp. Fig. 3). gCNN's pooled global signal helps when it carries perturbation-relevant information that local features lack – additive noise on frames, flicker and high-photon shot noise are all spatially distributed and gCNN beats vCNN under each (Fig. 4a; Supp. Fig. 23). It hurts when the global signal carries little perturbation-relevant information but dominates a weakened local signal – fog, low-photon shot noise, and adversarial textures all reduce local contrast against a still-present scene background, and gCNN underperforms vCNN under each (Fig. 4a; Supp. Fig. 18, 19; Supp. Fig. 23). Global pooling therefore buys robustness conditionally: it does not provide unconditional robustness, and it introduces specific failure modes where the pooled scene average can overwhelm weak local signals.

Recurrence improves robustness to white-box attacks The RAFT model [50] allows applying an arbitrary number of recurrent refinement iterations. Increasing recurrence depth (1, 6, and

a  
![](images/2203b33bd9fed5887fc8464ed2a831da0fb6b0fb60c48dbe5e1c30b13614cb4f.jpg)  
Figure 4: Global integration, recurrence, temporal context, and connectivity shape robustness under adversarial and common video perturbations. (a) Effect of global signal integration (vCNN vs. gCNN) and temporal context (vCNN(2 frames) vs. vCNN(5 frames) vs. vCNN(10 frames)). (b) Effect of recurrence depth (RAFT family). (c) Effect of connectivity (FlyNet vs. sCTRNN vs. CTRNN). Top row: accuracy under PGD attacks $( L _ { 2 }$ and $L _ { \infty } )$ . Bottom row: two representative common video perturbations (AWN on frames and fog). Error bars represent SEM.

12) in RAFT improved robustness in white-box attack conditions: CW attack success decreased monotonically, and accuracy under PGD increased substantially from RAFT-6 to RAFT-12, with RAFT-1 and RAFT-6 similar. RAFT-12 was also the most robust under additive noise on frames, shot noise, and flicker (Fig. 4b; Supp. Fig. 18, 21, 23; Table 2). Robustness was unchanged under the black-box NES and transfer attacks and under noise on texture, and decreased under motion jitter and fog, where RAFT-12 was the most vulnerable of the family (Supp. Fig. 19, 20, 23; Table 2). In hexRAFT, recurrence had little or inconsistent effect. Recurrent refinement is therefore a candidate for robustness to white-box attacks in artificial networks. FlyNet is also recurrent, but it updates its state once per frame rather than refining its state over repeated iterations on the same input, so the RAFT result cannot directly carry over to FlyNet.

Temporal integration supports robustness to common video corruptions FlyNet, CTRNN, and sCTRNN integrate their input through continuous-time dynamics (Eq. 1), which low-pass filter perturbations that change from frame to frame and provide more temporal context than the two-frame input of CNN and RAFT baselines. All three continuous-time models are highly robust to motion jitter, flicker and additive noise on frames, above most baselines (Fig. 4c; Supp. Fig. 23; Table 2), consistent with this low-pass filtering. To manipulate temporal context directly in a network without continuous-time dynamics, we retrained vCNN with 5 and 10 input frames instead of 2. Robustness was higher with 5 or 10 frames than with 2 under most perturbation types, and under all common video corruptions (Fig. 4a; Supp. Fig. 23; Table 2), showing that temporal context improves robustness to common video corruptions in a feedforward network.

No single connectivity property orders robustness among continuous-time models The three continuous-time models share dynamics and input but differ in connectivity. FlyNet's connectomederived structure is the most local: The influence-weighted mean distance between perturbed input columns and responding neurons (the projective counterpart [57] of the effective receptive field [58]; see App. A.3) is λ ≈ 1.5 hexagonal columns, compared to 2.2 for sCTRNN and 2.8 for CTRNN (2.7 for vCNN and 4.3 for gCNN). CTRNN, with \~5× more edges, is more robust than FlyNet under most perturbations tested, although the gap is small on common video corruptions (Fig. 4c; Supp. Fig. 18, 19, 20, 21, 23; Table 2). Pruning CTRNN to FlyNet's edge count and fine-tuning it (sCTRNN) increased robustness under PGD ${ \bar { L } } _ { 2 }$ and fog beyond both CTRNN and FlyNet, and matched CTRNN under PGD $L _ { \infty }$ (Fig. 4c), but reduced it below CTRNN under $L _ { \infty }$ transfer attacks, and below both CTRNN and FlyNet under flicker (Supp. Fig. 19, 20, 23; Table 2). The ordering therefore depends on the perturbation, and neither the number of free parameters (Table 1), edge count nor locality explain it consistently across these conditions. Under every adversarial attack, FlyNet, the most local model with the fewest free parameters, is the least robust of the three.

a  
![](images/b0a3930746bdcc0077175b9c3f4bbc535bb93e2b8b7492af65a840f2329d69e3.jpg)

![](images/badec0a93fb1caef9ca96fa1ce724d90b2319bf23ada6ddaa46eb9e4bf7b48c2.jpg)

b  
![](images/208006a55f2cbcb6edb3d7f0a4995c91807e12379119a9b569342d40940cddb2.jpg)  
Figure 5: Robustness-efficiency frontier across perturbation regimes. (a) Robustness R vs. computational cost $\mathcal { E } ,$ measured in FLOPs, for each network and perturbation type. Left: common video perturbations, including texture noise, frame noise, Poisson noise, flicker, jitter and fog. Right: adversarial attacks, including PGD, transfer and NES attacks. Inset: zoom for the CNN family. (b) LERS computed for all networks and perturbation types. Common video perturbations (left group): AWN texture, motion jitter, shot noise, AWN frames, flicker, fog. Adversarial attacks (right group): transfer $L _ { \infty }$ , transfer $L _ { 2 }$ , PGD $L _ { \infty }$ , NES, PGD $L _ { 2 }$

Together, these results show that robustness to common video corruptions is consistent with temporal integration and increases when temporal context is added to vCNN, whereas no single tested connectivity property explains the differences among the continuous-time models.

## 4.3 Robustness-efficiency trade-offs

Architectures span five orders of magnitude in compute cost: FlyNet runs at 0.186 GFLOPs per inference, RAFT-12 at 27,402 GFLOPs, with CNN baselines and CTRNN in between (Fig. 5a; Table 2). Compute does not buy proportional robustness. Within the RAFT family, robustness rises with cost under white-box attacks; but vCNN and gCNN sit at comparable FLOPs and differ substantially in robustness, and on common video corruptions the fly-inspired models achieve high robustness at a fraction of the cost of any other architecture tested. How much robustness does each architecture buy per unit of compute?

Log-Efficiency Robustness Score We measure the robustness-efficiency trade-off as

$$
\mathrm { L E R S } = \frac { \mathscr { R } } { \log _ { 1 0 } \mathscr { E } } ,\tag{5}
$$

where E is FLOPs per inference. The logarithmic normalization reflects diminishing returns in robustness from added compute [59], and is conservative – it penalizes compute less than linear normalizations would, so any trade-off advantage observed for low-compute architectures would only grow under stricter normalization. LERS represents robustness per order of magnitude of compute (Fig. 5b, Table 2).

Sparsity places FlyNet in a favorable LERS regime on common video corruptions We computed LERS for each model and perturbation type (Fig. 5b, Table 2). Within the CNN and RAFT families, compute differs by less than 4% in $\log _ { 1 0 } \mathcal { E }$ , so LERS follows the robustness rankings reported above. Across families, compute spans five orders of magnitude, and on all six common video corruptions the three continuous-time models have the highest LERS of all models. The one within-group comparison in which LERS departs from robustness is FlyNet versus CTRNN. On common video corruptions, robustness is high for all three continuous-time models $( \mathcal { R } \geq 0 . 8 7 )$ , and CTRNN's \~4× higher compute raises its $\log _ { 1 0 } \mathcal { E } \mathrm { b y } \sim 7 . 5 \%$ , more than its robustness lead over FlyNet; FlyNet therefore has higher LERS than CTRNN on common corruptions, although CTRNN is more robust. FlyNet and sCTRNN, with nearly identical compute, are ordered by R alone and reach comparable LERS. Under PGD $L _ { 2 }$ and NES, CTRNN's robustness lead exceeds the compute difference, and both CTRNNs rank above FlyNet. sCTRNN is more robust than FlyNet at slightly lower compute and has the highest LERS of all models under PGD and transfer attacks. Together, these results show that FlyNet's low edge count, and hence low compute, gives it higher LERS than CTRNN and all baselines other than sCTRNN on common video corruptions, despite CTRNN's higher raw robustness. sCTRNN, matched to FlyNet's edge count, reaches comparable LERS without FlyNet's synapse-level wiring.

## 5 Discussion

Robust and efficient sensory processing is a hallmark of biological neural circuits, yet the architectural principles that support this combination remain poorly understood. Here, we asked whether synaptic wiring derived from the connectome contributes to either property beyond what standard artificial architectures and task training achieve. To answer this, we built a differentiable 3D rendering pipeline for looming-based collision detection and benchmarked a connectome-constrained, task-optimized model of the fly visual system (FlyNet) against unconstrained CNN, globally pooled CNN, CTRNN, and RAFT-based baselines under adversarial attacks and common video perturbations. Recurrence (in RAFT), global context, and temporal context each support robustness under different perturbation regimes. FlyNet's connectome constraint, however, does not seem to contribute to robustness — CTRNN and sCTRNN, which lack detailed connectome constraints, are more robust than FlyNet under almost every perturbation tested. What the connectome seems to contribute is efficiency through sparsity: FlyNet matches CTRNN closely on common video corruptions while using a fraction of the synapses and compute, yielding higher LERS than any architecture other than sCTRNN. Our comparison to sCTRNN suggests that this efficiency advantage is not unique to the connectome's specific wiring. While FlyNet is robust to the tested common video corruptions it remains sensitive to optimized adversarial attacks (gradient-based PGD, gradient-free NES) indicating that biological structure alone does not imply general adversarial robustness. The fly itself is susceptible to a structured stimulus: Reverse-phi inverts the contrast of moving patterns and produces illusory motion in real flies — a perceptual error that reflects a specific computational vulnerability of the T4/T5 motion-detection circuit. FlyNet inherits this vulnerability, and our analysis shows that adversarial optimization discovers stimuli that exploit the same failure mode: Both invert the Hassenstein– Reichardt correlation between neighboring inputs, and manipulating that second-order correlation statistic alone suffices to induce the failure.

Limitations In the fly, downstream looming-selective neurons, such as LPLC2 and LC4, project to escape pathways [24, 25] that are not present in FlyNet. Beyond connectivity, biological circuits compute under additional constraints such as neuromodulation, synaptic plasticity, synaptic stochasticity [14], and spiking — none of which FlyNet models. Some asymmetries complicate cross-architecture comparison. First, decoder input dimensionality differs across architectures, which could confound cross-architecture robustness comparisons: a higher-dimensional readout may itself contribute to measured robustness rather than the backbone alone, leaving this as an axis for future work. Second, sCTRNN and FlyNet share edge count and compute but beyond wiring differ by more than three orders of magnitude in trainable parameters (1,513,361 vs 734; Table 1; Supp. Fig. 2), as well as in training pipeline (sCTRNN is fine-tuned), so differences between them cannot be attributed to synapse-level connectome constraints alone. Third, our robustness and efficiency results hold for the binary looming task, while generalization to other visual tasks, such as object tracking or navigation, and under more naturalistic scenes, remains untested and left as future work. Fourth, each architecture is represented by a single trained instance, so error bars reflect stimulus and attack variability, not variability across trained networks. Finally, our compute estimate uses FLOPs, which enables consistent cross-architecture comparison but does not measure the metabolic cost of biological computation [2, 3, 60, 61].

Conclusion and future direction In general, adversarial vulnerability appears to be a consequence of specialized computational strategies: Networks become robust to the perturbations they were trained on and vulnerable to perturbations designed against their architecture. Adversarial training is the field's canonical answer, raising the question whether the connectome's wiring already sits closer to an adversarially trained solution than an artificial neural network trained from scratch. Differentiable connectome-constrained models also turn adversarial attacks into a tool that can interface with experiments in biology. Future connectome-constrained models that incorporate downstream circuits and greater biological realism will generate more faithful predictions of adversarial stimuli for experimental tests. Because the correlation-based motion motif is shared with mammalian motion vision [23, 62], the approach could extend to larger connectome models as a route to candidate illusions for mammalian circuits. Our work suggests adversarial analysis as a tool for probing the limits of biological computation, while also informing the design of specialized neural networks that achieve favorable robustness-efficiency trade-offs. Immediate future work will focus on isolating the role of the connectome in robustness and address the generalization of these findings across more tasks.

## Acknowledgments and Disclosure of Funding

We thank the organizers and participants of the Cajal NeuroAI summer school 2025 at the Champalimaud Foundation, Lisbon, where this project was initiated. We also thank Srini Turaga, Mason McGill, and Eyal Gruntman for the discussions and helpful comments. This work was supported by MCIN/AEI/10.13039/501100011033 and FSE+ (grants PID2022-143311NB-I00, PRE2022- 101492, and RYC2022-037595-I); by the Spanish State Research Agency, through the Severo Ochoa and María de Maeztu Program for Centers and Units of Excellence in R&D (CEX2020- 001084-M); by a Newcastle University Academic Track (NUAcT) Fellowship and PhD studentship; by the Lister Institute Prize Fellowship; by the Air Force Office of Scientific Research (AFOSR, FA9550-23-1-0533); by the Leverhulme Research Leadership Award; by the Shanahan Foundation Fellowship at the Interface of Data and Neuroscience (Shanahan Family Foundation and Allen Institute); by the Howard Hughes Medical Institute (HHMI), Janelia Research Campus (we thank Srini Turaga for compute support); by the European Union (ERC, DeepCoMechTome, 101089288 to Jakob Macke). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them; and by the AC-CESS program (project: CIS261112, allocation: Anvil, Purdue University [63]), supported by NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. We thank CERCA Programme/Generalitat de Catalunya for institutional support. The code and networks associated with this work will be available at https://github.com/alexgarciaduran/flynet-robustness, https://huggingface.co/alejandrorg00/flynet-robustness, and the flyvis fork https: //github.com/lappalainenj/flyvis (release neurips2026).

## Author contributions

Conceptualization, methodology: A.G.-D., A.R.-G. and J.K.L. Software and investigation: A.G.-D., A.R.-G. and J.K.L. Analysis: A.G.-D., A.R.-G. and J.K.L. Writing: A.G.-D., A.R.-G. and J.K.L. Writing (review and editing): M.M.-M., A.H., S.R. Supervision: J.K.L. Funding: M.M.-M., A.H., S.R. and J.K.L.

## Competing interests

The authors declare no competing interests.

## References

[1] A Aldo Faisal, Luc P J Selen, and Daniel M Wolpert. Noise in the nervous system. Nat. Rev. Neurosci., 9 (4):292–303, April 2008.

[2] D Attwell and S B Laughlin. An energy budget for signaling in the grey matter of the brain. J. Cereb. Blood Flow Metab., 21(10):1133–1145, October 2001.

[3] Simon B. Laughlin, Rob R. de Ruyter van Steveninck, and John C. Anderson. The metabolic cost of neural information. Nature Neuroscience, 1(1):36–41, May 1998. ISSN 1546-1726. doi: 10.1038/236

[4] D M Eagleman. Visual illusions and neurobiology. Nat. Rev. Neurosci., 2(12):920–926, December 2001.

[5] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv [cs.LG], April 2021.

[6] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks, 2014. URL https://arxiv.org/abs/1312. 6199.

[7] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples, 2015. URL https://arxiv.org/abs/1412.6572.

[8] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.

[9] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv [cs.LG], March 2019.

[10] Janne K. Lappalainen, Fabian D. Tschopp, Sridhama Prakhya, Mason McGill, Aljoscha Nern, Kazunori Shinomiya, Shin-ya Takemura, Eyal Gruntman, Jakob H. Macke, and Srinivas C. Turaga. Connectomeconstrained networks predict neural activity across the fly visual system. Nature, 634(8036):1132–1140, 2024. doi: 10.1038/s41586-024-07939-3. URL https://doi.org/10.1038/s41586-024-07939-3.

[11] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In A. Fitzgibbon et al. (Eds.), editor, European Conf. on Computer Vision (ECCV), Part IV, LNCS 7577, pages 611–625. Springer-Verlag, October 2012.

[12] Kohitij Kar, Jonas Kubilius, Kailyn Schmidt, Elias B Issa, and James J DiCarlo. Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior. Nat. Neurosci., 22 (6):974–983, June 2019.

[13] Courtney J Spoerer, Tim C Kietzmann, Johannes Mehrer, Ian Charest, and Nikolaus Kriegeskorte. Recurrent neural networks can explain flexible trading of speed and accuracy in biological vision. PLoS Comput. Biol., 16(10):e1008215, October 2020

[14] Antonino Greco, Marco D'Alessandro, Karl J. Friston, Giovanni Pezzulo, and Markus Siegel. Robust inference via top-down feedback and neural variability in recurrent vision models. Patterns, page 101693, 2026. ISSN 2666-3899. doi: https://doi.org/10.1016/j.patter.2026.101693. URL https://www.sciencedirect.com/science/article/pii/S2666389926002023.

[15] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv [cs.CV], November 2018.

[16] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. arXiv [cs.CV], May 2021.

[17] Hanshu Yan, Jiawei Du, Vincent Y F Tan, and Jiashi Feng. On robustness of neural ordinary differential equations. arXiv [cs.LG], October 2019.

[18] Joel Dapello, Tiago Marques, Martin Schrimpf, Franziska Geiger, David Cox, and James J DiCarlo. Simulating a primary visual cortex at the front of CNNs improves robustness to image perturbations. Advances in Neural Information Processing Systems, 33:13073–13087, 2020.

[19] Yan Luo, Xavier Boix, Gemma Roig, Tomaso Poggio, and Qi Zhao. Foveation-based mechanisms alleviate adversarial examples, 2016. URL https://arxiv.org/abs/1511.06292.

[20] Jiajun Lu, Hussein Sibai, Evan Fabry, and David Forsyth. No need to worry about adversarial examples in object detection in autonomous vehicles, 2017. URL https://arxiv.org/abs/1707.03501.

[21] Matthew S. Maisak, Juergen Haag, Georg Ammer, Etienne Serbe, Matthias Meier, Aljoscha Leonhardt, Tabea Schilling, Armin Bahl, Gerald M. Rubin, Aljoscha Nern, Barry J. Dickson, Dierk F. Reiff, Elisabeth Hopp, and Alexander Borst. A directional tuning map of drosophila elementary motion detectors. Nature, 500(7461):212–216, August 2013. ISSN 1476-4687. doi: 10.1038/nature12320.

[22] Juergen Haag, Alexander Arenz, Etienne Serbe, Fabrizio Gabbiani, and Alexander Borst. Complementary mechanisms create direction selectivity in the fly. eLife, 5:e17421, aug 2016. ISSN 2050-084X. doi: 10.7554/eLife.17421. URL https://doi.org/10.7554/eLife.17421.

[23] Alexander Borst and Moritz Helmstaedter. Common circuit design in fly and mammalian motion vision. Nature Neuroscience, 18(8):1067–1076, June 2015. ISSN 1546-1726. doi: 10.1038/nn.4050.

[24] Nathan C. Klapoetke, Aljoscha Nern, Martin Y. Peek, Edward M. Rogers, Patrick Breads, Gerald M. Rubin, Michael B. Reiser, and Gwyneth M. Card. Ultra-selective looming detection from radial motion opponency. Nature, 551(7679):237–241, November 2017. ISSN 1476-4687. doi: 10.1038/nature24626.

[25] Jan M. Ache, Jason Polsky, Shada Alghailani, Ruchi Parekh, Patrick Breads, Martin Y. Peek, Davi D. Bock. Catherine R. von Reyn, and Gwyneth M. Card. Neural Basis for Looming Size and Velocity Encoding in the Drosophila Giant Fiber Escape Pathway. Current Biology, 29(6):1073–1081.e4, March 2019. ISSN 0960-9822. doi: 10.1016/j.cub.2019.01.079. URL https://doi.org/10.1016/j.cub.2019.01.079.

[26] Fabian David Tschopp, Michael B Reiser, and Srinivas C Turaga. A Connectome Based Hexagonal Lattice Convolutional Network Model of the Drosophila Visual System. arXiv preprint arXiv: 1806. 04793, 2018.

[27] Omer Mano, Matthew S. Creamer, Bara A. Badwan, and Damon A. Clark. Predicting individual neuron responses with anatomically constrained task optimization. Current Biology, 31(18):4062–4075.e4, September 2021. ISSN 0960-9822. doi: 10.1016/j.cub.2021.06.090. URL https://doi.org/10.1016/ j.cub.2021.06.090.

[28] Jacob Morra, Andrew Flynn, Andreas Amann, and Mark Daley. Multifunctionality in a connectome-based reservoir computer. In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 4961–4966, 2023. doi: 10.1109/SMC53992.2023.10394668.

[29] S.M. Anstis. Phi movement as a subtraction process. Vision Research, 10(12):1411–IN5, 1970. ISSN 0042-6989. doi: https://doi.org/10.1016/0042-6989(70)90092-1. URL https://www.sciencedirect. com/science/article/pii/0042698970900921.

[30] Stuart M. Anstis and Brian J. Rogers. Illusory reversal of visual depth and movement during changes of contrast. Vision Research, 15(8):957–IN6, 1975. ISSN 0042-6989. doi: https://doi.org/10.1016/0042-6989(75) 90236-9. URL https://www.sciencedirect.com/science/article/pii/0042698975902369.

[31] Margarida Agrochao, Ryosuke Tanaka, Emilio Salazar-Gatzimas, and Damon A. Clark. Mechanism for analogous illusory motion perception in flies and humans. Proceedings of the National Academy of Sciences, 117(37):23044–23053, 2020. doi: 10.1073/pnas.2002937117. URL https://www. pnas.org/ doi/abs/10.1073/pnas.2002937117.

[32] Margaret S Livingstone, Christopher C Pack, and Richard T Born. Two-Dimensional Substructure of MT Receptive Fields. Neuron, 30(3):781–793, May 2001. ISSN 0896-6273. doi: 10.1016/S0896-6273(01) 00313-0. URL https://doi.org/10.1016/S0896-6273(01)00313-0.

[33] John C. Tuthill, M. Eugenia Chiappe, and Michael B. Reiser. Neural correlates of illusory motion perception in drosophila. Proceedings of the National Academy of Sciences, 108(23):9685–9690, 2011. doi: 10.1073/pnas.1100062108.

[34] Damon A. Clark, Limor Bursztyn, Mark A. Horowitz, Mark J. Schnitzer, and Thomas R. Clandinin. Defining the computational structure of the motion detector in drosophila. Neuron, 70 (6):1165–1177, 2011. IS\$N 0896-6273. doi: https://doi.org/10.1016/j.neuron.2011.05.023. URL https://www.sciencedirect.com/science/article/pii/S0896627311004417.

[35] Aljoscha Leonhardt, Matthias Meier, Etienne Serbe, Hubert Eichner, and Alexander Borst. Neural mechanisms underlying sensitivity to reverse-phi motion in the fly. PLOS ONE, 12(12):e0189019, December 2017. ISSN 1932-6203. doi: 10.1371/journal.pone.0189019.

[36] Eyal Gruntman, Pablo Reimers, Sandro Romani, and Michael B Reiser. Non-preferred contrast responses in the drosophila motion pathways reveal a receptive field structure that explains a common visual illusion. Curr. Biol., 31(23):5286–5298.e7, December 2021.

[37] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks, 2017. URL https://arxiv.org/abs/1608.04644.

[38] Daan Wierstra, Tom Schaul, Jan Peters, and Jürgen Schmidhuber. Natural evolution strategies. In Proceedings of the IEEE Congress on Evolutionary Computation (CEC), pages 3381–3387, Piscataway, NJ, USA, 2008. IEEE. doi: 10.1109/CEC.2008.4631255.

[39] Victor Oei, Jenny Schmalfuss, Lukas Mehl, Madlen Bartsch, Shashank Agnihotri, Margret Keuper, Andreas Bulling, and Andres Bruhn. Robustspring: Benchmarking robustness to image corruptions for optical flow, scene flow and stereo. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=RebPBMrMmk.

[40] Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. Transferability in machine learning: from phenomena to black-box attacks using adversarial samples, 2016. URL https://arxiv. org/abs/1605. 07277.

[41] Anurag Ranjan, Joel Janai, Andreas Geiger, and Michael J. Black. Attacking optical flow, 2019. URL https://arxiv.org/abs/1910.10053.

[42] Simon Schrodi, Tonmoy Saikia, and Thomas Brox. Towards understanding adversarial robustness of optical flow networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[43] Jenny Schmalfuss, Philipp Scholze, and Andrés Bruhn. A perturbation-constrained adversarial attack for evaluating the robustness of optical flow. In Proc. European Conference on Computer Vision (ECCV), pages 183–200. Springer, 2022.

[44] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples 2018. URL https://arxiv.org/abs/1707.07397.

[45] Jenny Schmalfuss, Lukas Mehl, and Andrés Bruhn. Distracting downpour: Adversarial weather attacks for motion estimation. In Proc. International Conference on Computer Vision (ICCV). IEEE/CVF, 2023.

[46] Baohua Zhou, Zifan Li, Sunnie Kim, John Lafferty, and Damon A Clark. Shallow neural networks trained to detect collisions recover features of visual loom-selective neurons. eLife, 11:e72067, jan 2022. ISSN 2050-084X. doi: 10.7554/eLife.72067. URL https://doi.org/10.7554/eLife.72067.

[47] Florian T. Muijres, Michael J. Elzinga, Johan M. Melis, and Michael H. Dickinson. Flies evade looming targets by executing rapid visually directed banked turns. Science, 344(6180):172–177, 2014. doi: 10.1126/science.1248955. URL https://www.science.org/doi/abs/10.1126/science.1248955.

[48] Nikhila Ravi, Jeremy Reizenstein, David Novotny, Taylor Gordon, Wan-Yen Lo, Justin Johnson, and Georgia Gkioxari. Accelerating 3d deep learning with pytorch3d. arXiv:2007.08501, 2020.

[49] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2015.

[50] Zachary Teed and Jia Deng. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In European Conference on Computer Vision (ECCV), pages 402–419, 2020.

[51] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. 3rd International Conference on Learning Representations (ICLR), 2015.

[52] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the gdpr, 2018. URL https://arxiv.org/abs/1711.00399.

[53] Bernhard Hassenstein and Werner Reichardt. Systemtheoretische analyse der zeit-, reihenfolgen-und vorzeichenauswertung bei der bewegungsperzeption des rüsselkäfers chlorophanus. Z. Naturforsch. B J. Chem. Sci., 11(9-10):513–524, 1956.

[54] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International Conference on Machine Learning (ICML), 2020.

[55] Owen Kunhardt, Arturo Deza, and Tomaso Poggio. The effects of image distribution and task on adversarial robustness, 2021. URL https://arxiv.org/abs/2102.10534.

[56] K F Fischbach and A P M Dittrich. The optic lobe of Drosophila melanogaster. I. A Golgi analysis of wild-type structure. Cell Tissue Res., 258(3):441–475, 1989.

[57] Sidney R. Lehky and Terrence J. Sejnowski. Network model of shape-from-shading: neural function arises from both receptive and projective fields. Nature, 333(6172):452–454, June 1988. ISSN 1476-4687. doi: 10.1038/333452a0. URL https://doi.org/10.1038/333452a0.

[58] Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel. Understanding the effective receptive field in deep convolutional neural networks. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/paper/2016/file/ c8067ad1937f728f51288b3eb986afaa-Paper.pdf.

[59] Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, and Bhavya Kailkhura. Adversarial robustness limits via scaling-law and human-alignment studies. In 2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML 2024), 2024. URL https://openreview.net/forum?id=IpMM192TJA.

[60] Biswa Sengupta, Martin Stemmler, Simon B. Laughlin, and Jeremy E. Niven. Action potential energy efficiency varies among neuron types in vertebrates and invertebrates. PLOS Computational Biology, 6(7): e1000840, 2010. doi: 10.1371/journal.pcbi.1000840.

[61] Michael Deistler, Jakob H Macke, and Pedro J Gonçalves. Energy-efficient network activity from disparate circuit parameters. Proc. Natl. Acad. Sci. U. S. A., 119(44):e2207632119, November 2022.

[62] Damon A Clark, James E Fitzgerald, Justin M Ales, Daryl M Gohl, Marion A Silies, Anthony M Norcia, and Thomas R Clandinin. Flies and humans share a motion estimation strategy that exploits natural scene statistics. Nature Neuroscience, 17(2):296–303, February 2014. ISSN 1546-1726. doi: 10.1038/nn.3600. URL https://doi.org/10.1038/nn.3600.

[63] X. Carol Song, Preston Smith, Rajesh Kalyanam, Xiao Zhu, Eric Adams, Kevin Colby, Patrick Finnegan, Erik Gough, Elizabett Hillery, Rick Irvine, Amiya Maji, and Jason St. John. Anvil - system architecture and experiences from deployment and early user operations. In Practice and Experience in Advanced Research Computing 2022: Revolutionary: Computing, Connections, You, PEARC '22, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450391610. doi: 10.1145/3491418.3530766. URL https://doi.org/10.1145/3491418.3530766.

[64] Alexander Borst, Jürgen Haag, and Alex S. Mauss. How fly neurons compute the direction of visual motion. Journal of Comparative Physiology A, 206:109–124, 2020. doi: 10.1007/s00359-019-01375-9.

[65] Nikolaus Kriegeskorte, Marieke Mur, and Peter A. Bandettini. Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, Volume 2 - 2008, 2008. ISSN 1662-5137. doi: 10.3389/neuro.06.004.2008. URL https://www.frontiersin.org/ journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008.

## A Additional information

## A.1 Architectures

## A.1.1 FlyNet

FlyNet [10] is a differentiable neural network whose architecture is derived from a synapse-level connectome reconstruction of the Drosophila optic lobe, covering 65 cell types. The network topology, which neurons are connected and with what anatomical synapse counts, is fixed by connectomic measurements and not modified during training. What is learned are the unknown biophysical parameters: the single-neuron time constants $\tau _ { i }$ , resting potentials $h _ { i } ^ { r e s t }$ , and synaptic strength parameters that scale the fixed anatomical weights, yielding only 734 free parameters in total. These parameters are optimized using standard deep learning techniques to minimize an optical flow prediction loss on the Sintel dataset [11]. Crucially, no electrophysiological data are used during training: the model recovers single-neuron response properties consistent with three decades of experimental measurements as an emergent consequence of fitting the motion task under the connectome constraint. FlyNet has 45,669 neurons and 1.51M synaptic connections, and is governed by the continuous-time dynamics in Eq. 1, where $f$ is a ReLU nonlinearity applied to pre-synaptic voltages, and $l _ { i }$ represents the luminance input to photoreceptors (for all other cell types, $l _ { i } = 0 )$

b  
![](images/93a1e5600d9771c14aa422ee24eb737579a7615cefb9278daceb4e47654e1fd8.jpg)

![](images/cbffc21f4c2e999b8fc9c0c4d02933e86a8b7f551fd40a4db46573c5607ed475.jpg)  
Figure 1: FlyNet visual circuit. (a) Schematic of the fly optic lobe [64], from retina to lobula highlighting ON and OFF motion-selective T4/T5 pathways used for motion estimation by the decoder. (b) Contrast preference of the T4 and T5 subtypes, measured by the flash response index (FRI; $0 \mathrm { N } > 0 , 0 \mathrm { F F } \bar { < } 0 )$ , for the best-loss model, the second-best model (T4d donor), and the hybrid model. The parameter substitution reverses T4d from OFF to the expected ON preference and preserves the contrast preference of all other subtypes. (c) Directional tuning of T4 (ON-edge) and T5 (OFF-edge) neurons across the three models (solid lines), compared with experimental data (dashed lines).

We specifically decode T4 and T5 motion pathways (Supp. Fig. 1). The original work reports neural tuning averaged over clusters of 50 equivalently configured and stochastically trained model instances and individual instances can deviate from known tuning properties. However, no single pretrained model exhibits correct direction selectivity for all T4 and T5 subtypes simultaneously. In the model with the best Sintel validation loss, T4d prefers OFF instead of ON contrast and is only weakly direction selective, whereas all other T4 and T5 subtypes have the expected contrast preference and preferred direction. We therefore constructed a hybrid model: starting from the best-loss model, we replaced the resting potential and time constant of T4d, and the synaptic strengths of all ten connection types onto T4d, with those of the second-best model, whose T4d prefers ON contrast

After this substitution, all T4 and T5 subtypes have the expected contrast preference and preferred direction, and T4d becomes strongly direction selective (Supp. Fig. 1b,c).

## A.1.2 CTRNN

CTRNN follows the same continuous-time dynamics as FlyNet (Eq. 1), receives the same hexagonal input, but its weights $w _ { i j }$ are not constrained by the connectome. CTRNN learns weights of free signs for all connected cell-type pairs over a support of a 19-column hexagonal neighborhood (five columns across), increasing edge count from 1.51M to 7.51M and increasing the number of free parameters from 734 to 11,593. CTRNN thus serves as an ablation of the synapse-level connectome prior while holding the dynamical framework constant. CTRNN is the best-Sintel-validation network from Extended Data Fig. 2, DMN[11,593] in [10].

## A.1.3 sCTRNN

sCTRNN is obtained from the trained CTRNN by untying the shared filter weights and pruning per edge by the mean absolute injected current, down to FlyNet's 1.51M connections, followed by fine-tuning on Sintel. sCTRNN matches FlyNet's edge count of 1,513,231.

Supp. Fig. 2a,b characterize remaining differences in mean degree per cell type between sCTRNN, CTRNN, and FlyNet. CTRNN and sCTRNN degree distributions differ from FlyNet's $( r = 0 . 1 3$ in, 0.38 out). Supp. Fig. 2c, d characterize within and across cell-type in-degree and out-degree variation in each network.

![](images/fb5fb950b24265847ee74f2a951c114db5e8ff9a3f5e823d3d9d81f8a8fcc263.jpg)  
FlyNet mean in-degree per cell type

![](images/1d5ca42f908ec134f42601bdc9fd1a6a64b5f65c1ce14969321573cdb850be97.jpg)

![](images/fdf651f6fb2a285f65e130fc0c32f4819f206c2e50b98b5bcb42c4bd43cc30da.jpg)  
FlyNet mean out-degree per cell type

![](images/25b63fce37c554e7c63a40d93a3aac8884b921d1b0731a8581b8c11894f78f01.jpg)  
Figure 2: Comparison of in-degree and out-degree statistics between FlyNet, CTRNN, and sCTRNN. (a) Mean in-degree per cell type in CTRNN and sCTRNN against FlyNet (each point one of 65 cell types; dashed line: identity). r: Pearson correlation with FlyNet's per-type mean degree. (b) Same for out-degree. (c) Coefficient of variation of degree within a cell type's interior cells (interior meaning 10 of 15 columns across to remove boundary effects), median across types: FlyNet, CTRNN, and sCTRNN share one filter support by construction. sCTRNN's differs by a few edges, because of the remainder from pruning the isotropic convolutional filter support from 7.51M to 1.51M edges. (d) Coefficient of variation of degree across cell types within each interior column, median across columns. CTRNN's dense filters flatten degree differences between types.

## A.1.4 CNNs

We considered unconstrained CNN baselines trained on Sintel. The vanilla CNN consists of five layers with 32, 92, 136, 8, and 2 channels, respectively. All layers use a kernel size of 5, except for the last layer, which uses a kernel size of 1. Each layer consists of a convolution, batch normalization, and an exponential linear unit activation, except for the final layer, which applies only a convolution. The global pooling network uses the same architecture, but includes a global average pooling operation between the second and third layers, providing the network with access to global spatial context (Supp. Fig. 3). Among our CNNs, only vCNN(2f) is one of five networks from Extended Data Fig. 2 vanilla CNN in [10].

## A.1.5 RAFT

We use RAFT as a high-performance optical flow baseline, pretrained on standard benchmarks, and fine-tuned on Sintel. We extract feature maps from the final hidden representation prior to flow output. Given its size $1 2 8 \times H / 8 \times W / 8$ (we use $H = W = 4 1 6 )$ , we apply 2D average pooling (kernel size 7, stride 7), resulting in 6, 802 features for decoding.

b  
a  
![](images/e4257ab976787d35e3e3e12cca43ab28e8483450d6226c8f5c25e158fc6cf96a.jpg)

![](images/269c938f9b81c15bd9694f537085fc472e436c131a21bec0f64f5e983fcface4.jpg)  
Figure 3: vCNN (a) and gCNN (b) architectures.

To control for differences in visual sampling, we additionally trained a FlyNet-matched RAFT model (hexRAFT) using the same hexagonal sampling geometry as FlyNet (Supp. Fig. 4). Sintel frames were converted to luminance and sampled through an array of 721 hexagonally arranged photoreceptors, each performing spatial averaging over a $1 3 \times 1 3$ pixel receptive field. The resulting samples were mapped onto a 31 × 31 Cartesian grid, zero-padded to $3 2 \times 3 2$ , and bilinearly upsampled to $2 5 6 \times 2 5 6$ to produce RAFT-compatible inputs. The same sampling procedure was applied to the ground-truth optical flow, without rescaling flow magnitudes, and a validity mask excluded unsampled grid locations from the training loss. We trained the model on the Clean and Final passes of Sintel using the same scene-disjoint training split, for 50, 000 iterations with a batch size of 8, using AdamW $( \mathrm { l r } = 1 0 ^ { - 4 } , $ weight decay $\bar { \cdot } = 1 0 ^ { - 5 } )$ and 12 recurrent update iterations.

![](images/7afbcdbc5bcc0e340c673ec7f4a97ccb5bd4dd4891cfb8bf6855dbac8bafc89d.jpg)  
Figure 4: Hexagonal sampling pipeline for hexRAFT Sintel. Sintel RGB frames (top) and groundtruth optical flow (bottom) are shown in their original resolution (left), after hexagonal sampling with 721 receptors (middle), and after conversion to the 256 × 256 Cartesian representation used by RAFT (right). This matches the hexagonal sampling geometry of FlyNet, as do the other CTRNN and CNN benchmark families.

## A.2 Simulation

We used pytorch3d [48] to render the stimulus videos. The camera is fixed at the origin $\vec { r } _ { c } = ( 0 , 0 , 0 )$ . Objects are initialized at distance $z _ { 0 }$ along the z-axis, with positions in the plane parameterized by polar coordinates $( R , \theta ) \colon \vec { r _ { 0 } } = ( R$ cos θ, $R \sin \theta , z _ { 0 } )$ . Each trajectory is a linear translation of duration $T _ { \mathrm { t r a j } }$ from $\vec { r } _ { 0 }$ to a final point $\vec { r _ { f } }$ . For hit trials, $\vec { r _ { f } } = \left( 0 , 0 , z _ { \mathrm { o f f s e t } } \right)$ , where ${ \cal Z } _ { \mathrm { o f f s e t } }$ is a small displacement along $\boldsymbol { \hat { k } } = ( 0 , 0 , 1 )$ to prevent the object from intersecting the camera. For miss trials, the endpoint is shifted to the antipodal point on the circle: $\vec { r } _ { f } = ( - \alpha x _ { 0 } + \kappa _ { 1 } , ~ - \alpha y _ { 0 } + \kappa _ { 2 }$ , Zoffset), with $\alpha = 0 . 2 5 , \kappa _ { 1 } = \cos ( \pi + \mathrm { a t a n 2 } ( y _ { 0 } , x _ { 0 } ) )$ , and $\kappa _ { 2 } = \sin ( \pi + \mathrm { a t a n 2 } ( y _ { 0 } , x _ { 0 } ) )$ . Trajectory parameters were sampled uniformly: $\theta \in [ 0 , 2 \pi )$ $R \in [ 3 , 6 ] , z _ { 0 } \in [ 8 , 2 2 ]$ , and $T _ { \mathrm { t r a j } }$ took one of 9 equally spaced values in [0.5, 1.5] s.

## A.3 Functional locality

We ask how far a network carries the effect of a single input column — the projective counterpart [57] of the effective receptive field [58]. Let i index the units of a network, each assigned the coordinate pi (in units of the nearest-neighbor column spacing) of the hexal it sits above. Perturbing hexal j by 0.25 above a uniform input of 0.5 and time-averaging each unit's response, we define the influence of j on unit $i , \delta _ { i } ^ { ( j ) }$ , as the absolute change from the unperturbed run. Repeating this for every input column $j ,$ locality is the influence-weighted mean distance $\begin{array} { r } { \bar { d } = \sum _ { i , j } d _ { i } ^ { ( j ) } \delta _ { i } ^ { ( j ) } \big / \sum _ { i , j } \delta _ { i } ^ { ( j ) } } \end{array}$ , with $d _ { i } ^ { ( j ) } = \| \mathbf { p } _ { i } - \mathbf { p } _ { j } \|$ the distance from the perturbed hexal. It is the center of mass of the perturbation's effect: small when a column only moves its neighbors, larger as it spreads across the lattice.

## A.4 Robustness quantification

For each attack type, we sweep a parameter defining the magnitude of the perturbation and define robustness as the area under the accuracy curve R [55]:

$$
\mathcal { R } = \frac { 1 } { x _ { m a x } - x _ { m i n } } \int _ { x _ { \operatorname* { m i n } } } ^ { x _ { \operatorname* { m a x } } } \operatorname { A c c } ( x ) d x , \quad \operatorname { A c c } ( x ) = \frac { 1 } { N T } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \mathbf { 1 } [ \hat { y } _ { i , t } ( x ) = y _ { i , t } ] .
$$

For CW attacks [37], we quantify robustness as the minimum perturbation magnitude across values of c satisfying an attack success criterion of at least 80% misclassified frames:

$$
\delta _ { r } ^ { * } = \operatorname* { m i n } _ { c : \mathrm { s u c c e s s } _ { r , c } \geq 0 . 8 } \delta _ { r , c } ,
$$

where $\delta _ { r , c }$ is the perturbation magnitude $( L _ { 2 }$ or $L _ { \infty } )$ for run r and trade-off coefficient c.

## A.5 FLOPs quantification

We estimated inference FLOPs of all networks using the PyTorch profiler (torch.profiler).   
FLOPs are computed for 100-frame random videos.

## A.6 Compute

We ran experiments on Google Cloud Vertex AI using NVIDIA L4 and A100 GPUs, and on Purdue University's Anvil cluster using NVIDIA A100 GPUs, as computationally required.

## A.7 Decoding

Both decoders are trained with Adam [51] $( \mathrm { l r } { = } 1 0 ^ { - 3 }$ , batch size 1,024). The binary hit/miss classifier uses 512 hidden units, tanh activation, cross-entropy loss, L2 regularization $\alpha { = } 0 . 1$ , max 50 epochs, and early stopping with patience 10. The continuous position/velocity regressor uses 128–16 hidden units, ReLU activation, MSE loss, max 200 epochs, and early stopping with patience 50 and min improvement $1 0 ^ { - 5 }$

## A.7.1 Hit/Miss

![](images/0c63970e29e9adacdfa5556fe0fbf09a9858a53d7b81c85de11eee2cff0aed3a.jpg)  
Figure 5: Training dynamics of the hit/miss decoder across model architectures. Training loss (top) and validation accuracy (bottom) as a function of training epoch for FlyNet, CTRNN, sCTRNN, CNN-based models, and Cartesian and hexagonally sampled RAFT variants. Models are grouped by architecture across columns, with each curve representing a different model configuration.

![](images/4a49037c3481197e03289d7999a546f259a9633f916624376dede005db99c61b.jpg)  
Figure 6: Decoder test accuracy over time for all CTRNN and CNN networks, with normalized time (dividing by its total duration), conditioning on parameters of the simulation. Numbers in the legend correspond to ascending bins of the uniformly sampled parameters.

![](images/d06e45a79da5ab36180e794a972b8cfb1f0d7841acf0cc73dfd996da2271f548.jpg)  
Figure 7: Decoder test accuracy over time for all RAFT networks, with normalized time (dividing by its total duration), conditioning on parameters of the simulation. Numbers in the legend correspond to ascending bins of the uniformly sampled parameters.

## A.7.2 Motion

![](images/5abe0a2fd59dfad67dd26f2ab80ba1a7fd182f49e41275bd533e12f123576d49.jpg)

![](images/1a4b3db510b4f6a74ac5eb18f8dffa7a194854c97fc7ca6735585a623ae7f3f6.jpg)

![](images/2e56cf223c86e94a64e84bc66ecf717376403678d9f24a1bbe78585f1ad5ef85.jpg)

![](images/c9e28ebe0bda84c9921d1e6c15a87f470569415149c057e48eb30c933a70d33f.jpg)

![](images/b920f68d32d44372ca0ccc69de01302a2dc8d6df64e094e7202725974e00000d.jpg)

![](images/3e4dfd2a8a873c9541296b3a357c0de145508972df52127f583048a158c33f46.jpg)  
Figure 8: FlyNet decoder predictions versus ground truth for position (top) and velocity (bottom) on the test set. Columns correspond to the (x), (y), and (z) components. Pearson's (r) and (R2) are reported.

## B Additional results

## B.1 Reverse-phi

![](images/bfed78a5fd04ce45b85e9739bed92ce0a203e0a68a60337c123d583de6ea9bcb.jpg)

![](images/119603cf1e7fbf110a198ecd421622d5e9c5aa54c29a137e99e66b36d331bbfc.jpg)

![](images/b1f3a15782167d54223f12764bb3d1b6936777893afb8112d9325752fe2ad28d.jpg)

b  
![](images/17b8dc91321eadd18e4f7b2dc3712f1363bcdf37a0c33341e215c2ece3ef3cc6.jpg)

C  
![](images/84c404363aa7d0d6ea8856dac584d50fc161e4975d3a083ada8ca12ef166bdb9.jpg)

![](images/fb5e933fdd9cda2a27fa7685f645dc2363309580cb88147941d61240a79b250f.jpg)

d  
![](images/8b966c4b5143a97acff5da87a2f9272abfccb87bfd3dc359d386c4253e716d99.jpg)

![](images/4a495e6407fc4dc6dfac41e3ea91f00f9471d88a98ad6ff73c8834557125269a.jpg)  
Figure 9: FlyNet's representations of velocity consistently change sign during the reverse-phi. (a) Sign agreement, i.e. the average product of true and predicted velocity sign across $N = 5 0$ trajectories, given by $\begin{array} { r } { \frac { 1 } { N } \sum _ { N } \mathrm { s g n } ( v _ { \mathrm { t r u e } } ) ^ { \sum } \cdot \mathrm { s g n } ( \hat { v } ) } \end{array}$ . Left: original trajectory. Middle: contrast inversion period of $T = 0 . 0 5 s$ .Right: contrast inversion period of $T = 0 . 3 3 s$ . Average and s.e.m. (shaded area) across 50 trajectories, gray shaded area represents the contrast flip. (b) Time-averaged sign agreement vs period duration $\begin{array} { r } { \dot { T } . } \end{array}$ Dashed lines represent the original trajectory values. (c) Time evolution of the position (left) and velocity (right) decoding error for reverse-phi with $T = 0 . 0 5 s$ inversion frames. (d) Time-averaged position (left) and velocity (right) decoding errors vs period duration T.

## B.2 Parameter comparison across networks

<table><tr><td>Network</td><td>Activations</td><td>Free parameters</td></tr><tr><td>vCNN(2f)</td><td>276,480</td><td>414,064</td></tr><tr><td>vCNN(5f)</td><td>276,480</td><td>414,698</td></tr><tr><td>vCNN(10f)</td><td>276,480</td><td>414,858</td></tr><tr><td>gCNN(2f)</td><td>276,480</td><td>414,064</td></tr><tr><td>RAFTsm-12 [50]</td><td>432,640</td><td>1.0M</td></tr><tr><td>RAFT-1 [50]</td><td>697,632</td><td>5.3M</td></tr><tr><td>RAFT-6 [50]</td><td>697,632</td><td>5.3M</td></tr><tr><td>RAFT-12 [50]</td><td>697,632</td><td>5.3M</td></tr><tr><td>hexRAFT-1</td><td>264,192</td><td>5.3M</td></tr><tr><td>hexRAFT-6</td><td>264,192</td><td>5.3M</td></tr><tr><td>hexRAFT-12</td><td>264,192</td><td>5.3M</td></tr><tr><td>FlyNet [10]</td><td>45,669</td><td>734</td></tr><tr><td>CTRNN [10]</td><td>45,669</td><td>11,593</td></tr><tr><td>sCTRNN</td><td>45,669</td><td>1.51M</td></tr></table>

Table 1: Parameter comparison across networks. In the case of sCTRNN, parameters comprise 1,513,231 per-edge weights, 65 resting potentials, and 65 time constants, yielding 1,513,361 free parameters.

![](images/c70e66875cbb5bb088c0e17d57aee9d6dbf660e0eb714ffd2855a5bde3a8ea8b.jpg)  
Figure 10: Robustness versus free parameters across models under common video corruptions (left) and adversarial attacks (right).

## B.3 Optimization

## B.3.1 Counterfactual restoration

a  
![](images/a8d370802274b44ef7a88cf9bfa97712ad414da98aa218ca72325650a1c9d50f.jpg)

![](images/e4ae6993fe6ba5d5cd86202acee22158029f11a941fff6c29968fdef2448c783.jpg)

b  
![](images/80048afa3c12e2acf4ec07346848825ffd6c5afd6b829032d417e31a643d80f3.jpg)

![](images/3ad745d401b6be7c0465fa2045eac4e10f89386194713b1510eb17ffa80ed85d.jpg)  
Figure 11: $L _ { 2 }$ distances of the textures during counterfactual restoration. (a) $L _ { 2 }$ distance distributions between the Initial, Illusion, and Restored textures (left), and the minimum distance between such textures and the training set textures (right). The base texture is sampled from the training set, so the distance is 0. (b) Same as a but median evolution over steps, with shaded IQR.

![](images/60dfeea48b8a84b403e8baebc72329bbfd8e055c8b09fe2e6f141d1034e55456.jpg)  
Figure 12: p(hit), position and velocity decoding errors over steps during counterfactual restoration. Top 3: Example trajectories. Bottom: Average across 30 trajectories.

![](images/393e86dba3c02d6e6f3f93df5789d897554ab5042697935830c905ba96a29835.jpg)  
b

![](images/0fcabee36bd45817d91be98f224a8d4139142fef1aa04fd16fa2d3280d0bac28.jpg)

![](images/74bfd9a2ba59f33672328244d49c6e35d924a44800256aaddd97e18272ed4db6.jpg)

d  
e  
![](images/d234143f391e4d94e5868523246976c53a001e5c54ba4129c952e3476dd68d98.jpg)

![](images/a00ff0640557bebf1bd5414dfc8176850a6b149f8e87bcecd112d8a3ba00fce6.jpg)  
Figure 13: Optimizing the input HR correlation alone produces the illusion and restores the hit response. Textures were optimized to minimize $g ( \tau = 8 )$ (Eq. 4) for 300 steps and then to maximize it for 300 steps, with a single Adam optimizer and an anchor penalty $( \lambda = 0 . 1 ;$ 30 trajectories). (a) p(hit) across optimization steps. Stars mark initial (black), illusion (red), and restored (blue) textures. (b) p(hit) at the three checkpoints. ( $: ) \ g ( \tau = 8 )$ across optimization steps. (d) $g ( \tau = 8 )$ at the three checkpoints. (e) p(hit) against $g ( \tau = 8 )$ along the optimization, during minimization (red) and maximization (blue); dot opacity and arrowheads indicate optimization steps Lines show the mean; shaded area represents SEM; asterisks indicate significance from paired t-test $( ^ { \ast \ast \ast \ast } ; p < 0 . 0 0 1 )$ 1

![](images/5316a842afc1848e4aef2fa04ad98b37dca366535e4e94bc7480c73ead7467c3.jpg)

b  
![](images/923e586776de7ae33926886f0877793656d5afd2f086b7937f09b7265f9bc42e.jpg)

![](images/5a1597fc68e76d1c260ea402a870b955823f3ac0b9fba7dfc63bcc0485359a18.jpg)  
Figure 14: (a) Single neuron activity (each panel is a neuron) over time, for a trajectory, under all conditions: original texture, reverse-phi, illusion texture and restored texture (from counterfactual restoration). (b) Maximum single-neuron activity perturbation, defined as $\Delta = \mathrm { A c t }$ condition — Act. original. Taken across trajectories, computed in the window of contrast change for reverse-phi —for reverse-phi, illusion and restored textures (from counterfactual restoration)—. Neurons sorted according to the maximum perturbation given by illusory texture. Left: predicted positive by the illusory texture. Right: predicted negative by the illusory texture. Mean rolling window of 200 neurons for smoothing

## B.4 Representational Similarity Analysis (RSA)

We use representational similarity analysis [65] to quantify how each condition's T4/T5 population response relates to reverse-phi. We represent every condition by the window-mean activity of its 5768 T4/T5 units over the full 330 ms first-contrast-flip window, and compare these population vectors, $\mathbf { r } _ { i } ,$ taking reverse-phi as the reference. We use three metrics: the Pearson correlation across units, the Euclidean distance (their $\ell _ { 2 }$ distance normalized by $\sqrt { n _ { \mathrm { u n i t s } } } )$ , and a signed projection.

Relative to reverse-phi, the adversarial representation showed higher correlation (0.874) than the restored (0.864) and original (0.834) representations, and lower Euclidean distance (0.121 vs. 0.125 and 0.140, respectively; Supp. Fig. 15a,b). A signed projection onto the original-reverse-phi axis further confirmed this shift (0.203 for adversarial vs. 0.149 for restored, with original defined as 0; Supp. Fig. 15c).

The signed projection computes

$$
S _ { i j } = \frac { \| \mathbf { d } _ { i } \| } { \| \mathbf { d } _ { j } \| } \cos \theta _ { i j } ,
$$

the scalar projection of $\mathbf { d } _ { i }$ onto $\mathbf { d } _ { j } ,$ where $\mathbf { d } _ { i } = \mathbf { r } _ { i } - \mathbf { r } _ { \mathrm { o r i g } }$ is the displacement of condition i from the original. The original sits at 0 and condition j at 1. $\theta _ { i j }$ is the angle between the two representation displacements. The headline comparison (Supp. Fig. 15c) takes j = reverse-phi.

Both adversarial and restored align with reverse-phi above chance $( \theta \approx 5 8 ^ { \circ } )$ . Their shifts differ in magnitude: the adversarial shift is the longer of the two, so it projects farther onto the original→reverse-phi axis (0.20 vs. 0.15) while reverse-phi overshoots both (1.34, 1.96), placing the adversarial response between the original and reverse-phi and closer to reverse-phi than the restored one. This is consistent with single-neuron traces (Supp. Fig. 14a), where the illusion response departs from the original in the same direction as reverse-phi but at reduced amplitude, while the restored response tracks the original.

All three metrics showed significant differences between the adversarial representation and both the restored and the original one (paired t-tests across the 30 trajectories, all $p \ < \ 0 . 0 0 5 )$ . The condition-by-condition matrices for the three metrics are shown in Supp. Fig. 15d–f.

a  
![](images/e308e4b44ff39c3d68859c6a63225e8b5fb7e2846d70be026339380864f3f824.jpg)

b  
![](images/75452d95bdfc4ef636ab42e875104859fbe3569df39ab152095b42cac5f73c91.jpg)

C  
![](images/6fae81c2f591f807282f1d22a6ac64c365b819027cb0c0c0358729e6cfaba8cb.jpg)

d  
![](images/18b49848d612dbaa83b0261a84853ad42c94e5b3fdab2d196bce2ceb7f6d1e2d.jpg)

![](images/5cf4278fc0ad2b6fd7152598cff9b1f8d61e5aea85da960d56b36fc0416d73f7.jpg)

![](images/cfc4075f75cba25af9f89c23fcfa1290c8e306e245233ca18010a80d373f0c75.jpg)  
Figure 15: Representational similarity to reverse-phi. (a-c) Closeness of each condition's T4/T5 population response to reverse-phi, under three metrics: Pearson correlation across units (a), Euclidean distance (b), and the signed projection onto the original→reverse-phi axis (c), on which the original sits at 0 and reverse-phi at 1. Bars are means over the 30 trajectories, error bars s.e.m., dots individual trajectories; brackets give paired t-tests $( ^ { * * } p < 0 . 0 1 , ^ { * * * } p ^ { ^ { \bullet } } < 0 . 0 0 1 )$ . The adversarial texture is closer to reverse-phi than the restored one on every metric. (d) The full condition-bycondition matrix for each metric; panels a-c are its reverse-phi column. Each matrix is computed within a trajectory and averaged over the 30 trajectories. In signed projection entry $( i , j )$ gives how far condition i lies along the original→ j axis, so the original row and column are 0 by construction; reverse-phi overshoots the adversarial texture along the original→adversarial axis (1.34), placing the adversarial response between the original and reverse-phi rather than beyond it.

## B.5 HR correlator

a  
![](images/acbe4061d9af30f08bfdfc7469433c9d6d4b82e9186975c1e78717e7a699e4fc.jpg)

![](images/2128fb1881b75b6573bcf230364aba7a5dfc012ef60a3faee831ab13849fe2ce.jpg)

![](images/a5f55ce69f9e242d600cab32f19188cafacfe3d8e2fe9f9a564240f3936ce458.jpg)

b  
![](images/cfb1e58bbbf9482e38ab0c270b529c6c3d53c1c51625977b155e79181576774e.jpg)

C  
![](images/c40e1e2e31b4a5cecc96cea064a6e3ded129e080d6246be5f7ace7bd4e254afa.jpg)

![](images/c0549f85d52b51657cdded17806ac6916f6bfbd67697f0781bf699fda62e60f2.jpg)

![](images/94f28c463aabf2343307b1b790a22873c6608405c5a4affecbd48feb623a38f7.jpg)

![](images/eff6e95e971c63bed748d99d11369cea3cff0682ce62697221b28db7b58d2456.jpg)

![](images/d1b0b4f4d5274dcb66e160f20bb11f44d66f9ed3d6b70d4c4f194d5bfe1d17f0.jpg)  
Figure 16: The Hassenstein-Reichardt looming correlation. (a) Cartoon, two consecutive frames of a looming object (black ball) approaching the camera. Two neighboring pixels (in, out) are highlighted, with the activity of the highlighted neighboring pixels, and a delayed example. (b) Dependence of $g ( \tau )$ on $\tau$ for FlyNet, for the 4 conditions (all averaged at the same window as reverse-phi). Average across 30 trajectories, shade indicates standard error of the mean. (c) Real example, pixel pair highlighted. Traces of the pixels (in delayed) for the 4 conditions: Original, Adversarial illusion, Restored, Reverse-phi. Pearson correlation of the traces is reported on top.

a  
![](images/81bab49705742fe55850375fd094e9ca7a7cb172d152649d23129ada5fcb17c5.jpg)

b  
![](images/ce12b25bf3989f1a78fecfbb3e05f548459f43789d969e35d5e70bd9e380852f.jpg)

![](images/1ab9934dd05b5d5c2b51e89cece69b7ca10129f9dd259cb83eb49fe54683a70a.jpg)

![](images/7730f1f56c3e54922b1dc6977274385d3671d86aa8eb55c00f53aa3a36633709.jpg)  
Figure 17: Comparison between counterfactual restoration and $g ( \tau )$ optimization on FlyNet. Top hex-panels: evolution over time of the input for the 3 conditions. Middle panel: $p ( \mathrm { h i t } )$ over time. Bottom panel: $g ( \tau )$ over time. (a) Counterfactual restoration: original (black), adversarial illusion (red) and restored (blue). (b) Optimizing $g ( \tau )$ : original (black), minimizing (red) and maximizing (blue) $g ( \tau )$

## B.6 Attacks

## B.6.1 Adversarial attacks

Methods We perform untargeted adversarial attacks over the object texture by optimizing a free logit variable, ensuring valid pixel intensities through a sigmoid transformation. All adversarial evaluations are performed on 100 time-steps hit trajectories with Black initial texture where baseline accuracy exceeds 90%, ensuring that failures arise from perturbations rather than pre-existing model errors. For PGD [8], we iteratively update the texture to maximize misclassification under $L _ { \infty }$ and $L _ { 2 }$ constraints via projection, using 100 steps with step size $\alpha = \varepsilon / 1 0$ . During optimization, we employ an adaptive step-size scheme inspired by [54]: at fixed checkpoints of 20 steps, if the loss fails to improve, the step size is halved and the iterate is reset to the best solution found so far; otherwise, optimization proceeds with the current step size. We sweep over $\varepsilon _ { L _ { \infty } } = [ 1 , 6 , 8 , 1 0 , 1 5 ]$ and $\varepsilon _ { L _ { 2 } } = [ 1 0 0 , 2 5 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 ]$ . Recall we optimize the texture in logit space. Each attack uses $^ 2$ random restarts from noise within the constraint set, and we run 15 independent attacks per condition. We report the highest label-flip rate across restarts. We then use the resulting PGD adversarial textures to assess transfer by applying them directly to the other models. For CW attacks [37], we optimize the same variable using Adam to minimize a distance term plus a misclassification objective, performing a grid search over $\overline { { c } } = [ 1 , 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } ]$ with 150 steps and $\mathrm { l r } = 1 0 ^ { - 1 }$ evaluating both $L _ { \infty }$ and $L _ { 2 }$ distances. Finally, for the gradient-free NES method [38], we estimate the update direction using symmetric Gaussian perturbations, sweeping over $\sigma = [ 1 , 3 , 6 , 9 , 1 2 ]$ , using 100 optimization steps with 15 gradient-estimation samples per step and a normalized update, and running 15 attacks per condition.

a  
b  
![](images/44f92835c75464516b7e7dba55c12fa324aa3bd68469985ab7a7cd3f22ecd787.jpg)  
Figure 18: Robustness of models under projected gradient descent (PGD) attacks. (a) $\mathrm { P G D - } L _ { \infty }$ and (b) PGD. $. L _ { 2 }$ attacks across model architectures. Classification accuracy and change in loss $( L = L _ { \mathrm { e n d } } - L _ { \mathrm { i n i } } )$ are shown as functions of perturbation size $\varepsilon .$ Error bars indicate variability across samples.

![](images/864ccfd66ffa6c3f606162bd536cee4101badfd50c862adb1e8d34aeb9078eac.jpg)  
Figure 19: Robustness of models under natural evolution strategies (NES) attacks. Classification accuracy and change in loss $( L = L _ { e n d } - L _ { i n i } )$ are shown as functions of the Gaussian perturbation standard deviation σ across model architectures. Error bars indicate variability across samples.

![](images/3980b2386b72fccd42a0e7a1214ad32d127a127308126fa498d75b0592cef9b7.jpg)

![](images/247580933ace6fc332ee0182219edd1d66fbdf427e09201a9ffc0cec8885f8ce.jpg)

![](images/3c3a2aa0bd5a749ccc3ec17602d09d45e81ed57633c7411b02094700c4adfad2.jpg)  
Figure 20: (a) Transfer attack AUC matrices for PGD illusions. Rows: Source of illusion. Columns: Target of the illusion. Top: $L _ { 2 }$ norm. Bottom: $L _ { \infty }$ norm. (b) Transfer attack curves for flynet as source (left) and target (right). Top: $L _ { 2 }$ norm. Bottom: $L _ { \infty }$ norm.

a  
b  
![](images/6bd0a3e07a860a9ff6e8406defb7253e2379a15827c91d08487006731d2d703c.jpg)  
Figure 21: Robustness of models under Carlini-Wagner (CW) attacks. (a) $\mathrm { C W } { \cdot } L _ { \infty }$ and (b) $\mathrm { C W } { \cdot } L _ { 2 }$ attacks across model architectures. Attack success rates and mean perturbation distances are shown as functions of the attack parameter (c), together with attack success as a function of perturbation distance. Error bars indicate variability across samples.

![](images/f2e3dcde7c8eced2d35f8b5f10eefc0613ceec3df9833f72807fe375c5bc3dfd.jpg)  
Figure 22: Examples of adversarial textures found for FlyNet. Adversarial textures optimized for the synthesized FlyNet model under PGD, CW, and NES attacks. Columns show increasing perturbation strength, either ε for PGD, c for CW, or σ for NES, illustrating the distinct texture structures induced by each attack objective and constraint.

## B.6.2 Common video corruptions

Additive White Noise (AWN) on Texture: It simulates sensor noise or surface irregularities on the looming object. All frames are given by adding zero-mean independent white noise with variance $\sigma ^ { 2 }$ to the texture.

AWN on Frames: We test robustness to global unstructured perturbations across the entire visual field. simulating temporal sensor grain and atmospheric degradation. At each frame, we add zero-mean independent white noise with variance $\sigma ^ { 2 }$ to every input pixel. For FlyNet, CNNs, and hexRAFT we add it to every pixel of the hexagonal input of $\bar { N _ { \mathrm { F l y N e t } } } = 7 2 1$ columns $V _ { t } .$ , for RAFT to every pixel of the full image $F _ { t }$ . Noise power is matched across architectures by scaling $\sigma ( \mathrm { a r c h . } ) ~ = ~$ $\sigma \sqrt { N _ { \mathrm { F l y N e t } } / N _ { \mathrm { a r c h . } } }$

Shot (Poisson) noise: Similar to the previous perturbation but with normalized Poisson noise instead of Gaussian, modulated by the photon count $c .$ Noise power is matched across architectures by scaling the photon count as $c ( \mathrm { a r c h . } ) = c \cdot N _ { \mathrm { a r c h . } } / N _ { \mathrm { F l y N e t } }$

Frame Flickering: We apply random, frame-wise gain fluctuations $( G _ { t } = 1 + \xi _ { t } )$ , modulated by the flickering strength ε as $\xi _ { t } \sim \mathcal { U } ( - \varepsilon , \varepsilon )$

Motion jitter: We apply random, frame-wise shifts of the whole image in both vertical and horizontal dimensions. We change the maximum amount of shifted pixels by the jitter j magnitude.

Contrast Sensitivity (Fog): We reduce stimulus contrast against the background $( \hat { \mathbf { F } } = ( \mathbf { F } - b ) \alpha + b )$ α modulates the amount of contrast scale difference between the object and the background, $\alpha = 1$ is the original video and α = 0 are just gray frames.

Results for AWN, shot noise, flickering, and motion jitter are averaged over 10 trajectories with 10 noise realizations per trajectory, while fog results are averaged over 20 trajectories.

![](images/cf3f2ab5a711276a3cd125740e8c9a7a684a213583a377364b2c0d58987f94f6.jpg)  
Figure 23: Accuracy of networks under common video perturbations, for each network family. Order (columns): Additive White Noise (AWN) on the texture, AWN on the frames, Fog/Reduced contrast, Flicker, Motion Jitter and Shot (Poisson) noise. The first five consist on the average across 10 trajectories and 10 noise samples for each trajectory. The last consists on the average across 20 trajectories. Each row corresponds to a network family.

![](images/89127207d05e79868fe69470cbc7fd06ceb79b15eb5a4d419e5b7db73a8d9e9e.jpg)  
Figure 24: Examples of frames over time of each different video corruption

B.6.3 Summary table
<table><tr><td></td><td></td><td>FlyNet</td><td>sCTRNN</td><td>CTRNN</td><td>vCNN(10f)</td><td>vCNN(5f)</td><td>vCNN(2f)</td><td>gCNN(2f) hexRAFT-12</td><td>hexRAFT-6</td><td>hexRAFT-1</td><td>RAFT-12</td><td></td><td>RAFT-6</td><td>RAFT-1</td><td>RAFTsm-12</td></tr><tr><td>E (GFLOPs)</td><td></td><td>0.186</td><td>0.183</td><td>0.783</td><td>71.7</td><td>75.6</td><td>78.8</td><td>78.8</td><td>10,290</td><td>6,495</td><td>3,332</td><td>27,402</td><td>17,380</td><td>9,029</td><td>6,138</td></tr><tr><td></td><td>R</td><td>0.379 ± 0.040</td><td>0.616 ± 0.068</td><td>0.537 ± 0.058</td><td>0.448 ± 0.032</td><td>0.320 ± 0.034</td><td>0.385 ± 0.052</td><td>0.306 ± 0.022</td><td>0.524 ± 0.019</td><td>0.516 ± 0.026</td><td>0.531 ± 0.019</td><td>0.766 ± 0.021</td><td>0.657 ± 0.033</td><td>0.625 ± 0.014</td><td>0.548 ± 0.031</td></tr><tr><td>PGD L2</td><td>LERS</td><td>0.046 ± 0.005</td><td>0.075 ± 0.008</td><td>0.060 ± 0.007</td><td>0.041 ± 0.003</td><td>0.029 ± 0.003</td><td>0.035 ± 0.005</td><td>0.028 ± 0.002 0.040 ± 0.001</td><td></td><td>0.040 ± 0.002 0.042 ± 0.002</td><td>0.057 ± 0.002</td><td></td><td>0.050 ± 0.002 0.048 ± 0.001</td><td></td><td>0.043 ± 0.002</td></tr><tr><td></td><td>R</td><td>0.569 ± 0.014</td><td>0.675 ± 0.029</td><td>0.658 ± 0.027</td><td>0.605 ± 0.009</td><td>0.584 ± 0.015</td><td>0.592 ± 0.027</td><td>0.571 ± 0.012</td><td>0.632 ± 0.009</td><td>0.651 ± 0.015 0.631 ± 0.013</td><td></td><td>0.793 ± 0.015</td><td>0.703 ± 0.018</td><td>0.674 ± 0.012</td><td>0.636 ± 0.017</td></tr><tr><td>PGD L∞</td><td>LERS</td><td>0.069 ± 0.002</td><td>0.082 ± 0.003</td><td>0.074 ± 0.003</td><td>0.056 ± 0.001 0.054</td><td>± 0.001</td><td>0.054 ± 0.003</td><td>0.052 ± 0.001</td><td>0.049 ± 0.001 0.051</td><td> ± 0.001</td><td>0.050 ± 0.001</td><td>0.059 ± 0.001</td><td>0.053 ± 0.001</td><td>0.052 ± 0.001</td><td>0.050 ± 0.001</td></tr><tr><td></td><td>R</td><td>0.675 ± 0.049</td><td>0.783 ± 0.043</td><td>0.783 ± 0.031</td><td>0.720 ± 0.039</td><td>0.764 ± 0.066</td><td>0.683 ± 0.037</td><td>0.619 ± 0.044</td><td>0.526 ± 0.023 0.522</td><td>± 0.020</td><td>0.541 ± 0.018</td><td>0.686 ± 0.018</td><td>0.686 ± 0.023</td><td>0.760 ± 0.019</td><td>0.835 ± 0.032</td></tr><tr><td>Transfer L2</td><td>LERS</td><td>0.082 ± 0.006</td><td>0.095 ± 0.005</td><td>0.088 ± 0.003</td><td>0.066 ± 0.004</td><td>0.070 ± 0.006</td><td>0.063 ± 0.003</td><td>0.057 ± 0.004</td><td>0.040 ± 0.002</td><td>0.041 ± 0.002</td><td>0.043 ± 0.001</td><td>0.051 ± 0.001</td><td>0.052 ± 0.002</td><td>0.059 ± 0.001</td><td>0.065 ± 0.003</td></tr><tr><td>Transfer L∞</td><td>R</td><td>0.711 ± 0.026</td><td>0.749 ± 0.024</td><td>0.782 ± 0.025</td><td>0.785 ± 0.014</td><td>0.778 ± 0.022</td><td>0.751 ± 0.019</td><td>0.726 ± 0.022</td><td>0.650 ± 0.014</td><td>0.636 ± 0.011</td><td>0.649 ± 0.011</td><td>0.741 ± 0.010</td><td>0.744 ± 0.014</td><td>0.793 ± 0.015</td><td>0.847 ± 0.018</td></tr><tr><td></td><td>LERS</td><td>0.086 ± 0.003</td><td>0.091 ± 0.003</td><td>0.088 ± 0.003</td><td>0.072 ± 0.001 0.071</td><td>± 0.002</td><td>0.069 ± 0.002 0.067</td><td>± 0.002</td><td>0.050 ± 0.001 0.050</td><td>± 0.001</td><td>0.052 ± 0.001</td><td>0.055 ± 0.001</td><td>0.056 ± 0.001</td><td>0.061 ± 0.001</td><td>0.066 ± 0.001</td></tr><tr><td>Adveal atccks NES</td><td>R</td><td>0.444 ± 0.053</td><td>0.562 ± 0.088</td><td>0.614 ± 0.043</td><td>0.600 ± 0.030</td><td>0.610 ± 0.059</td><td>0.634 ± 0.079</td><td>0.366 ± 0.044</td><td>0.726 ± 0.026</td><td>0.681 ± 0.021</td><td>0.655 ± 0.018</td><td>0.803 ± 0.014</td><td>0.732 ± 0.033</td><td>0.784 ± 0.030</td><td>0.599 ± 0.020</td></tr><tr><td></td><td>LERS</td><td>0.054 ± 0.006</td><td>0.068 ± 0.011</td><td>0.069 ± 0.005</td><td>0.055 ± 0.003</td><td>0.056 ± 0.005</td><td>0.058 ± 0.007</td><td>0.034 ± 0.004</td><td>0.056 ± 0.002</td><td>0.053 ± 0.002</td><td>0.052 ± 0.001</td><td>0.060 ± 0.001</td><td>0.055 ± 0.003</td><td>0.061 ± 0.002</td><td>0.047 ± 0.002</td></tr><tr><td>CW</td><td>|∥|δtex|∞</td><td>0.940 ± 0.039</td><td>0.966 ± 0.018</td><td>0.969 ± 0.018</td><td>0.947 ± 0.027</td><td>0.968 ± 0.012</td><td>0.964 ± 0.020</td><td>0.914 ± 0.043</td><td>0.980 ± 0.006</td><td>0.978 ± 0.009</td><td>0.975 ± 0.013</td><td></td><td>0.836 ± 0.000</td><td>0.717 ± 0.116</td><td>0.967 ± 0.010</td></tr><tr><td rowspan="4">AWN (texture)</td><td>||δtex||2</td><td>16.10 ± 2.41</td><td>22.97 ± 8.30</td><td>23.95 ± 7.95</td><td>17.14 ± 5.32</td><td>20.49 ± 7.64 20.21 ± 3.78</td><td>13.77 ± 3.30</td><td>26.14 ± 6.41</td><td>24.87 ± 9.59</td><td>24.51</td><td>± 5.31 +</td><td></td><td>4.00 ± 0.00</td><td></td><td></td></tr><tr><td>R</td><td>0.989 ± 0.005</td><td>0.985 ± 0.003</td><td>0.991 ± 0.004</td><td>0.993 ± 0.005 0.992</td><td>± 0.007</td><td>0.931 ± 0.032</td><td>0.904 ± 0.036</td><td>0.970 ± 0.008</td><td>0.976 ± 0.007</td><td>0.962 ± 0.014</td><td>0.986 ± 0.005</td><td>0.991 ± 0.004</td><td>4.64 ± 4.20</td><td>16.35 ± 0.00</td></tr><tr><td>LERS</td><td>0.120 ± 0.001</td><td>0.119 ± 0.000</td><td>0.111 ± 0.000</td><td>0.091 ± 0.000 0.091</td><td>± 0.001</td><td>0.085 ± 0.003 0.083</td><td>± 0.003</td><td>0.075 ± 0.001</td><td>0.076 ± 0.001 0.077 ± 0.001</td><td></td><td>0.073 ± 0.000</td><td>0.075 ± 0.000</td><td>0.993 ± 0.003</td><td>0.914 ± 0.034</td></tr><tr><td>R AWN (frames)</td><td>0.942 ± 0.013</td><td>0.957 ± 0.008</td><td>0.972 ± 0.010</td><td>0.930 ± 0.013</td><td>0.896 ± 0.022</td><td>0.441 ± 0.014</td><td>0.670 ± 0.023</td><td>0.103 ± 0.004</td><td>0.088 ± 0.002</td><td>0.199 ± 0.018</td><td>0.993 ± 0.003</td><td>0.950 ± 0.012</td><td>0.077 ± 0.000 0.921 ± 0.020</td><td>0.071 ± 0.003</td></tr><tr><td>vieieoo veons</td><td>LERS</td><td>0.114 ± 0.002</td><td>0.116 ± 0.001</td><td>0.109 ± 0.001</td><td>0.086 ± 0.001 0.082</td><td>± 0.002</td><td>0.041 ± 0.001 0.062</td><td>± 0.002 0.008 ± 0.000</td><td>0.007 ± 0.000</td><td>0.016 ± 0.001</td><td>0.074 ± 0.000</td><td>0.072 ± 0.001</td><td></td><td>0.868 ± 0.016</td><td></td></tr><tr><td></td><td>R</td><td>0.974 ± 0.005</td><td>0.981 ± 0.003</td><td>0.985 ± 0.004</td><td>0.963 ± 0.011 0.964</td><td>± 0.009</td><td>0.685 ± 0.027 0.890</td><td>± 0.015</td><td>0.142 ± 0.009 0.153</td><td>± 0.011</td><td>0.225 ± 0.028</td><td>0.995 ± 0.002</td><td>0.984 ± 0.002</td><td>0.071 ± 0.002</td><td>0.068 ± 0.001</td></tr><tr><td>Shot noise</td><td>LERS</td><td>0.118 ± 0.001</td><td>0.119 ± 0.000</td><td>0.111 ± 0.000</td><td>0.089 ± 0.001 0.089</td><td>± 0.001</td><td>0.063 ± 0.003 0.082</td><td>± 0.001</td><td>0.011 ± 0.001 0.012</td><td>± 0.001</td><td>0.018 ± 0.002</td><td>0.074 ± 0.000</td><td>0.074 ± 0.000</td><td>0.979 ± 0.004</td><td>0.944 ± 0.007</td></tr><tr><td></td><td>R</td><td>0.925 ± 0.015</td><td>0.880 ± 0.012</td><td>0.968 ± 0.010</td><td>0.871 ± 0.014</td><td>0.890 ± 0.017</td><td>0.662 ± 0.018</td><td>0.708 ± 0.020</td><td>0.989 ± 0.004 0.992</td><td>± 0.003</td><td>0.961 ± 0.007</td><td>0.663 ± 0.028</td><td>0.420 ± 0.032</td><td>0.076 ± 0.000 0.435 ± 0.023</td><td>0.074 ± 0.001</td></tr><tr><td>Flicker</td><td>LERS</td><td>0.112 ± 0.002</td><td>0.107 ± 0.001</td><td>0.109 ± 0.001</td><td>0.080 ± 0.001 0.082</td><td>± 0.002</td><td>0.061 ± 0.002 0.065</td><td>± 0.002</td><td>0.076 ± 0.000</td><td>0.077 ± 0.000</td><td>0.077 ± 0.001</td><td>0.049 ± 0.002</td><td>0.032 ± 0.002</td><td>0.034 ± 0.002</td><td>0.308 ± 0.013 0.024 ± 0.001</td></tr><tr><td>Coomonn</td><td>R</td><td>0.986 ± 0.003</td><td>0.984 ± 0.004</td><td>0.988 ± 0.004 0.111 ± 0.000</td><td>0.646 ± 0.020</td><td>0.856 ± 0.009</td><td>0.605 ± 0.016</td><td>0.626 ± 0.021</td><td>0.561 ± 0.021</td><td>0.592 ± 0.018</td><td>0.585 ± 0.014</td><td>0.504 ± 0.010</td><td>0.603 ± 0.011</td><td>0.555 ± 0.008</td><td>0.705 ± 0.008</td></tr><tr><td>Motion jitter</td><td>LERS</td><td>0.119 ± 0.000 0.875 ± 0.018</td><td>0.119 ± 0.000 0.955 ± 0.010</td><td>0.928 ± 0.010</td><td>0.060 ± 0.002 0.943 ± 0.013</td><td>0.079 ± 0.001</td><td>0.056 ± 0.002 0.810 ± 0.024</td><td>0.057 ± 0.002</td><td>0.043 ± 0.002 0.975 ± 0.002</td><td>0.046 ± 0.001</td><td>0.047 ± 0.001</td><td>0.037 ± 0.001</td><td>0.046 ± 0.001</td><td>0.043 ± 0.001</td><td>0.055 ± 0.001</td></tr><tr><td rowspan="2">Fog</td><td>R</td><td></td><td>0.116 ± 0.001</td><td>0.104 ± 0.001</td><td>0.929 ± 0.001 0.085</td><td>± 0.013 ± 0.001</td><td>0.782 0.074 ± 0.002 0.072</td><td>± 0.028 ± 0.003</td><td>0.075 ± 0.000</td><td>0.977 ± 0.001</td><td>0.954 ± 0.009</td><td>0.984 ± 0.005</td><td>0.997 ± 0.001 0.999 ± 0.000</td><td></td><td>0.925 ± 0.014</td></tr><tr><td>LERS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.077 ± 0.000</td><td></td></tr><tr><td></td><td></td><td>0.106 ± 0.002</td><td></td><td></td><td>0.087</td><td></td><td></td><td></td><td></td><td>0.076 ± 0.000</td><td>0.076 ± 0.001</td><td>0.073 ± 0.000</td><td>0.075 ± 0.000</td><td></td><td>0.072 ± 0.001</td></tr></table>

Table 2: Robustness R and log-efficiency robustness score (LERS) across networks and perturbation types. Models are grouped by architecture family; the vCNN variants use 2-, 5-, and 10-frame temporal contexts, respectively. hexRAFT denotes the FlyNet-matched RAFT variants. For each row, the best-performing mean is bold and underlined, the second-best is bold, and the third-best is italicized; the corresponding uncertainty receives the same formatting. Tied values share the same rank. The smallest GFLOPs value is highlighted in bold. The RAFT family is from [50], and FlyNet from [10]. For CW, we report the mean and standard deviation of the minimum perturbation magnitude across runs attaining at least 80% misclassified frames. Empty cells indicate missing results; † indicates that CW data are present but no qualifying run was found.