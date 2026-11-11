# Mechanistic Interpretability of Deep Neural Networks: Methods, Frameworks, and Perspectives

Lorenzo Cirillo , Irene Amerini , Andrea Cavallaro

Deep neural networks underpin fundamental tasks in a wide range of signal domains, such as image, audio, and language processing. However, their internal mechanisms are notably difficult to interpret. Internal activations are structured signals transformed across layers that may encode a multitude of interdependent relationships, as individual neurons frequently represent multiple underlying concepts. Mechanistic interpretability aims to uncover the internal mechanisms of neural networks by identifying components whose activations are causally responsible for specific behaviors. In such a context, this article provides an overview of advances in mechanistic interpretability for the analysis of causal inference in deep neural networks.

## INTRODUCTION

Deep neural networks have achieved unprecedented performance across several signal domains, including image, audio, and language processing [1], [2]. These networks perform complex tasks such as classification, detection, generation, and forecasting. However, their internal processing mechanisms are difficult to interpret [3]. Uncovering causal relationships in deep models is fundamental for improving reliability, controllability, safety, alignment, and mitigating the risk of undesirable behaviors of learning systems [4]. Building on this, traditional interpretability and mechanistic interpretability aim to make the decision-making process of neural networks understandable, and their characteristics are compared in Table I.

Traditional interpretability techniques focus on input-output relationships and high-level summaries, operate on the relevance of the input features to the predicted final output, and identify which input features most influence the model output. Notable approaches include integrated gradients [5] and GradCAM [6]. Interpretability techniques can be model-agnostic, such as integrated gradients, which apply to any network regardless of its components, or model-specific, such as GradCAM, which is tailored for convolutional neural networks. Although these methods consider the gradients of the network to attribute input feature importance, they do not explicitly examine the activations of internal model components, offering limited insights into how latent representations are constructed through the model architecture.

In contrast, mechanistic interpretability aims to identify the internal mechanisms of neural networks, relying on model components responsible for specific behaviors [7]. A model component is an entity within the neural network that takes a signal as input, processes it, and outputs a corresponding activation. Examples of model components include neurons, layers, and modules. The components analyzed by a mechanistic interpretability method define its granularity, which refers to the resolution at which internal mechanisms are identified. At the finest scale, mechanistic methods target the causal contribution of specific neurons. At coarser scales, the focus shifts to aggregated transformations applied by entire architectural modules or layers to predict the model output. An internal mechanism refers to the causal relationship between model components and inputs for a given model behavior. For instance, a set of components that activate for specific inputs and trigger the model behavior constitutes an internal mechanism. Mechanistic interpretability techniques, such as activation maps applied to internal activations of the model [8] and logit lens [9], allow the observation of model activations without modifying any parameters. Sparse autoencoders [10] and task vectors [11] isolate distinct model components correlated to particular input concepts. Activation patching [12] tests causal hypotheses by replacing specific components to measure their direct impact on the signal flow. Causal testing manipulates internal signals such as activations, hidden states, and attention patterns of neural networks and evaluates through targeted interventions the resulting changes in model behavior [12]. These techniques have been extended to identify circuits [3], [13], namely subgraphs within the computational graph of the neural network that are causally responsible for specific behaviors or functional properties of the model. The computational graph is a directed acyclic graph where nodes correspond to the operations performed during the model forward pass, and edges represent the activations flowing sequentially from one node to the next. Circuit discovery combines different mechanistic interpretability methodologies to find causal relationships between model components and shows the most promising results in terms of model interpretability [3]. While previous surveys on mechanistic interpretability [7], [14] mainly focus on mechanistic understanding within specific neural networks or emphasize the context of safety and robustness, we introduce a distinct perspective. We focus on how these diverse techniques are combined into interpretability frameworks to achieve a causal understanding of deep neural networks. Moreover, we provide examples of mechanistic methods for audio classification and image reconstruction.

TABLE I  
COMPARISON OF TRADITIONAL AND MECHANISTIC INTERPRETABILITY.
<table><tr><td></td><td>TRADITIONAL INTERPRETABILITY</td><td>MECHANISTIC INTERPRETABILITY</td></tr><tr><td>GOAL</td><td>Identify input features that most contribute to the model final output</td><td>Identify model components that are causally responsi- ble for the model behavior</td></tr><tr><td>GRANULARITY</td><td>Input-output relationship</td><td>Model inner mechanisms</td></tr><tr><td>COMPUTATION COST</td><td>Medium-low (generally a single forward pass and backward pass)</td><td>Medium-high (sparse autoencoder training, circuit dis- covery across all model components)</td></tr><tr><td>SCALABILITY</td><td>Medium-high (analysis on model input and final out- put)</td><td>Medium-low (causal testing iterating across all model components)</td></tr><tr><td>LIMITATIONS</td><td>Approximate, limited analysis of how the model pro- duces internal activations</td><td>Poor performance under data distribution shift</td></tr></table>

![](images/5c2ac30183522e1686b46b78d553de5a9517d016b61865f1a1df660ec5b524b0.jpg)  
Fig. 1. Categorization of mechanistic interpretability, along with representative methods under each category.

We categorize techniques that support mechanistic interpretability frameworks (Figure 1) into four groups: representation visualization, representation disentanglement, causal intervention, and circuit tracking. Representation visualization techniques observe and measure the information encoded in activation signals without altering them, through probes or visualization tools [8]. Techniques in this category generate hypotheses about which signals may contribute to specific model behaviors. Representation disentanglement factors activations into interpretable components or subspaces to isolate distinct features through dedicated methods such as sparse autoencoders [10]. Techniques in this category enable the identification of candidate neural variables that may serve as causal factors in the model internal computations. Causal intervention operates on internal activations to test causal hypotheses about how the model processes its internal signals through intervention techniques such as activation patching [12]. Techniques in this category causally verify the relevance of specific components to the model computational flow. Finally, circuit tracking techniques identify and map the flow of information, finding specific pathways, circuits, or attention patterns in the network through, for example, automated circuit discovery [3] or edge attribution patching [13]. Techniques in this category reconstruct causal chains linking internal activations to observed model behaviors.

## REPRESENTATION VISUALIZATION

Representation visualization aims to observe and measure the activations without altering them [8]. Activation maps are used to visualize the activations of model components. Unlike traditional interpretability techniques, which compute activation maps by considering the final output of the model, mechanistic approaches exploit these maps to measure responses of internal model components to specific inputs. Visualizing the internal computations is a preliminary step to understanding the model behavior [15]. An activation map $A ( \mathbf { x } )$ is computed as the gradient of the activation $f ( \mathbf { x } ) \in \mathbb { R } ^ { n }$ with respect to the input $\mathbf { x } \in \mathbb { R } ^ { d }$ , with $n > d { : }$

$$
A ( \mathbf { x } ) = \frac { \partial f ( \mathbf { x } ) } { \partial \mathbf { x } } .\tag{1}
$$

Figure 2 illustrates an end-to-end application of representation visualization using activation maps. We analyze a Vision Transformer (ViT-B/16), pretrained on ImageNet-1K and fine-tuned for 12 epochs on the ESC-50 dataset [16] for audio classification, achieving 88% test accuracy. The one-dimensional audio signals from the ESC-50 dataset are transformed into spectrograms to serve as inputs to the model. In the input spectrogram x, pixel intensity corresponds to acoustic energy magnitude. We extract the raw activation tensor $f ( \mathbf { x } )$ from layer 6. The activation map $A ( \mathbf { x } )$ is obtained by computing the gradient of the raw activation $f ( \mathbf { x } )$ with respect to the input spectrogram x. After normalizing the values of $A ( \mathbf { x } )$ to [0, 1], the resulting map highlights the regions of the input that most influence the internal state of the layer. The highlighted regions are the time-frequency bins where small perturbations would cause the most substantial changes in the output of layer $6 , A ( \mathbf { x } )$ reveals that layer 6 localizes specific acoustic events occurring in mid-to-high frequency bands (bins 130 − 200) at distinct time frames.

In addition to activation maps, which visualize the model prediction layer by layer, the logit lens [9] converts the logits of hidden states into the output space of the model. This is useful to observe how the model processes its input till the final prediction. Representation visualization methods have been exploited to provide descriptions of neuron activations [15]. By collecting the images that most activate the neuron of interest, captions for such images are generated, along with images corresponding to each generated caption. The neuron is described by selecting the best caption based on the corresponding image on which the neuron activates the most. Observing model activations reveals that scaled models are not easier to interpret than original models [17]. The process involves the selection of random neurons in the scaled and non-scaled models and visualizing the corresponding activations. Then, some participants are presented with a pair of strongly and weakly activating images for the target neuron and are asked to identify the strongly activated one. Participants showed higher identification accuracy when evaluating the non-scaled neural network. In the context of neural network alignment, investigating the model internal mechanisms ensures harmlessness following safety fine-tuning by visualizing neuron activations within transformer blocks [18]. This specific training stage aims to mitigate toxic behavior without degrading overall performance capabilities. When presented with a malicious input, an aligned model must recognize the hazard and modulate its output accordingly. By comparing internal representations triggered by benign versus harmful prompts, the fine-tuning

![](images/305c2fda659b12a1db6cfdfedb30cdfe8f09242db27277608e5e599ed93b3770.jpg)  
Fig. 2. Activation map computation for ViT-B/16 audio spectrogram classification task at layer granularity. The activation map is computed for layer 6.

process separates safe and unsafe activations.

Representation visualization methods identify the internal mechanisms of Low-Rank Adaptation (LoRA) fine-tuning within neural networks for speech emotion recognition [19]. This analysis leverages the logit lens, which inspects the network by projecting intermediate layer representations into the final classification space. The representation extracted from a given layer is passed through the model to generate intermediate logits. By evaluating these intermediate predictions against the final output using the Kullback-Leibler (KL) divergence, it is possible to show that early encoder layers preserve general features, while deeper layers contain the most significant contributions of LoRA.

## REPRESENTATION DISENTANGLEMENT

A neural network can learn more concepts than its number of neurons by encoding features in overlapping activation subspaces. This phenomenon is known as the superposition hypothesis [20]: while this dense representation increases model capacity, it introduces polysemanticity. Polysemanticity is the phenomenon where a single neuron responds to multiple concepts [21]. A polysemantic neuron confounds the causal pathways of all the distinct concepts it simultaneously encodes, thus making it difficult to perform interventional tests. This behavior is a major obstacle to causality, since the same activations can reflect several patterns depending on context, thus preventing a direct mapping between neurons and input concepts. A polysemantic network can be conceptualized as a highly compressed approximation of a much larger monosemantic model, where each neuron acts as a dedicated channel activating exclusively for a single input feature.

Disentanglement is the process of converting a polysemantic signal into a monosemantic component and producing monosemantic representation spaces. Monosemanticity is the phenomenon where each neuron corresponds to a single concept. To interpret deep neural networks and deal with the superposition hypothesis [20], internal neural components must be disentangled into interpretable monosemantic components or subspaces to isolate distinct features. Upon this, task vectors [11] are linear directions within a neural network parameter space that isolate discrete learned behaviors. Defined as the difference between the weights of a task-specific fine-tuned model and its pre-trained base model, these vectors embed input concepts as nearly orthogonal separable features within a high-dimensional continuous space. Sparse autoencoders can be used to project polysemantic activations into a monosemantic space [10]. A sparse autoencoder takes as input an activation $\mathbf { x } \in \mathbb { R } ^ { d }$ of a model component. The encoder $\mathbf { W } _ { \mathrm { e n c } } \in \mathbb { R } ^ { n \times d }$ projects x into a high-dimensional space $\mathbb { R } ^ { n }$ , where $n > d ,$ and computes the monosemantic features activations $\mathbf { h } \in \mathbb { R } ^ { n }$

$$
\mathbf { h } = \mathrm { R e L U } ( \mathbf { W } _ { \mathrm { e n c } } \mathbf { x } + \mathbf { b } _ { \mathrm { e n c } } )\tag{2}
$$

where $\mathbf { b } _ { \mathrm { e n c } }$ is the encoder bias, and ${ \mathrm { R e L U } } ( \mathbf { x } ) = \operatorname* { m a x } ( 0 , \mathbf { x } )$ is the Rectified Linear Unit function ensuring that any negative values become zero. The decoder $\mathbf { W } _ { \mathrm { d e c } } \in \mathbb { R } ^ { d \times n }$ reconstructs the original activation $\hat { \mathbf { x } } \in \mathbb { R } ^ { d }$ from the monosemantic features:

$$
\hat { \mathbf { x } } = \mathbf { W } _ { \mathrm { d e c } } \mathbf { h } + \mathbf { b } _ { \mathrm { d e c } }\tag{3}
$$

where $\mathbf { b } _ { \mathrm { d e c } } \in \mathbb { R } ^ { d }$ is the decoder bias. The loss function $\mathcal { L }$ to be minimized is:

$$
\mathcal { L } = | | \mathbf { x } - \hat { \mathbf { x } } | | _ { 2 } ^ { 2 } + \lambda | | \mathbf { h } | | _ { 1 }\tag{4}
$$

where $| | \mathbf { x } - \hat { \mathbf { x } } | | _ { 2 } ^ { 2 }$ is the mean squared error that forces the sparse autoencoder to accurately reconstruct the original activation, and $\lambda | | \mathbf { h } | | _ { 1 }$ is the regularization term forcing the reconstruction to use as few monosemantic features as possible. The reconstructed input $\hat { \bf x }$ is exploited to train the sparse autoencoder and compute the monosemantic features h. Figure 3 shows the functionality of a sparse autoencoder, whose monosemantic features activate based on the input concepts.

Sparse autoencoders have been employed to ensure that representations learned by foundation models align with the ontological structure of the ImageNet dataset [22]. This ontology is organized in sets of synonymous terms that represent a single concept. The sparse autoencoder is trained on the activation of the foundation model layers. The activation maps computed on the hidden states of the sparse autoencoder with respect to the input images highlight input features corresponding to hierarchical concepts, confirming that the model effectively reveals the hierarchical structure. Furthermore, sparse autoencoders have been adopted in diffusion models for concept unlearning [23]. By disentangling model components and analyzing the corresponding activations under specific concepts, activations responsible for undesired concepts are scaled with a negative multiplier at inference time to control the generation process. Similarly, uncovering internal mechanisms of bias generation in diffusion models supports bias mitigation [24]. Representation disentanglement provides the identification of neurons correlated with bias generation. In this way, it is possible to control the bias level in the generated content by adjusting the scale of such identified neurons. Moreover, to disentangle convolutional neural networks, inserting middle layers with monosemantic channels within the model allows for the decomposition of polysemantic channels into monosemantic ones. By tuning the parameters of the added disentangled channels at training time, the new network still has comparable performance to the original one [25].

![](images/00f0debb8cd9df9a9a26362a4b640410a0f35cfa30c95120ba8bc669c2fb6b13.jpg)  
Fig. 3. Sparse autoencoder applied to the linear module activation of a transformer block at module granularity. Each monosemantic feature in the hidden space of the sparse autoencoder reacts to the input concept it is associated with.

Training a sparse autoencoder requires the activations extracted from the specific model components under analysis, depending on the chosen granularity. Given that the encoder projects the signal into a higher dimensional space, the sparse autoencoder training is generally limited to the considered model activations [24]. However, there is no guarantee that the high-dimensional space of a sparse autoencoder will contain the specific concept to be analyzed. In applications such as concept unlearning [23] or bias mitigation [24], whose aim is to penalize activations responsible for specific concepts or bias in the neural networks, it becomes particularly challenging to penalize components related to undesired model behaviors if those concepts are absent from the learned representations of the sparse autoencoder.

## CAUSAL INTERVENTION

Causal intervention techniques modify activations to verify their contribution to the internal processing mechanisms. To determine the causal impact of a specific component on the model behavior, activation patching [12] is a causal tool that relies on ablation to measure whether the model output changes accordingly. If the prediction shifts, the patched activation is deemed causally responsible for that behavior.

Given the activation $f ( \mathbf { x } ) \in \mathbb { R } ^ { n }$ of a model component $f$ when the input is $\mathbf { x } \in \mathbb { R } ^ { d } .$ , with $n > d ,$ the most used ablation techniques are to replace $f ( \mathbf { x } )$ with a zero-tensor of the same dimension, $f ( \mathbf { x } ) \gets 0$ (zero ablation); with its average value across a dataset D, $f ( \mathbf { x } ) \gets \mathbb { E } _ { x \sim \mathcal { D } } [ f ( \mathbf { x } ) ]$ (mean ablation); or with its version coming from a different input $\tilde { \mathbf { x } }$ to test which part of the network carries specific information, $f ( \mathbf { x } ) \gets f ( \tilde { \mathbf { x } } )$ , (interchange ablation).

Figure 4 reports an end-to-end application of activation patching via interchange ablation for audio spectrogram classification, utilizing the same ViT-B/16 model with the ESC-50 dataset [16] used for Figure 2. In the base run, the model processes and correctly classifies a vacuum cleaner spectrogram. In the target run, it correctly classifies a rain spectrogram. We apply activation patching to identify which feed forward modules within the transformer blocks are causally responsible for the vacuum cleaner prediction. To this end, we perform a causal intervention during the target run, replacing the feed forward activation with the corresponding activation saved from the base run. We then observe the resulting changes in the output logits and the final predicted class. If this intervention causes the vacuum cleaner logit to increase enough to change the model prediction, we can conclude that the patched activation drives the model toward the base class. As depicted in the third step of Figure 4(a), patching the feed forward module in the first transformer block causes the vacuum cleaner logit in the target run to increase from 0.002 to 7.753, restoring the base prediction. Furthermore, Figure 4(b) details the logit values for both classes when this intervention is applied independently across the feed forward module of each block. The vacuum cleaner logits surpass the rain logits when patching any of the first three feed forward modules, showing that these early layers are causally responsible for the vacuum cleaner prediction.

![](images/5193dcb1b2868996cf11842d560921aee3c6d802815a22dbc832f6bc98045e8b.jpg)

![](images/51c9b2b7d2cbb3faaf69d56d8a2ac0037c087b80a441c40e6fb8d1aca9ef4d32.jpg)  
(b)  
Fig. 4. Activation patching for ViT-B/16 audio spectrogram classification at module granularity. (a) Interchange ablation on the feed forward module of the first transformer block. Injecting the base run activation into the target run alters the logits and restores the vacuum cleaner prediction. (b) Logits for vacuum cleaner ( ) and rain ( ) during activation patching applied to every feed forward module separately. Patching any of the first three feed forward modules raises the vacuum cleaner logit, reverting the prediction from rain.

Activation patching with interchange ablation requires a separate forward pass for every component intervened upon. Attribution patching [26] provides a computationally efficient approximation of the causal effect of an activation on a performance measure M. The complete procedure to apply attribution patching is detailed in Algorithm 1. Attribution patching requires two forward passes (one to store the activations from the base run and one from the target run) and a single backward pass to compute the model gradient. Attribution patching estimates the change in the measure, denoted as the scalar $\Delta M .$ , using a first-order Taylor expansion around the base activation. Let $f ( \mathbf { x } ) \in \mathbb { R } ^ { n }$ denote the activation of a model component f given a base input x, and $f ( \tilde { \mathbf { x } } ) \in \mathbb { R } ^ { n }$ denote the activation given a target input x˜. Let $M : \mathbb { R } ^ { n }  \mathbb { R }$ be a scalar measure evaluating the model output, such as loss or logit difference. Attribution patching estimates the effect of intervening on f as:

$$
\Delta M = \frac { \partial M } { \partial f ( \mathbf { x } ) } \cdot ( f ( \tilde { \mathbf { x } } ) - f ( \mathbf { x } ) ) ,\tag{5}
$$

where $\frac { \partial M } { \partial f ( \mathbf { x } ) } \in \mathbb { R } ^ { n }$ is the gradient of the performance measure with respect to the base activation, and the · operator denotes the dot product over all dimensions of the activation.

Causal intervention verifies whether internal model components function as causal mediators between input data and model outputs. In vision-language models, causal interventions have been used to determine the cause-effect relationship between specific visual tokens and object identification accuracy [27]. After representation visualization identifies candidate visual tokens correlated with a target concept, counterfactual intervention via mean ablation is performed on those tokens. If model performance degrades when the activation of these components is altered, their causal responsibility in producing the final prediction is established. In diffusion models, cause-effect hypotheses on feature disentanglement are validated using interchange ablation [28]. By replacing internal representations with corrupted activations derived from perturbed inputs, researchers evaluate whether restoring specific disentangled features causally steers the generated output toward the target distribution. Furthermore, causal intervention can identify the internal mechanisms responsible for behavioral safety triggers in vision-language models exposed to toxic inputs [29]. When facing a toxic input, if the safety mechanism is activated, the model refuses the instructions and apologizes to the user. Therefore, a reply with an apology is a signal of the safety mechanism. Based on that, selecting samples that successfully enable the safety mechanism enables the analysis of the triggered model activations. Layers responsible for the safety mechanism exhibit attention peaks when the input is toxic. This is validated by masking the toxic information and testing the changes in the safety capabilities of the neural network.

Algorithm 1 Attribution Patching   
Input: Base input x, Target input $\tilde { \mathbf { x } } \in \mathbb { R } ^ { d } ,$ , Model component f, Scalar performance measure M   
Output: Estimated change in measure $\Delta M$   
1: Forward Passes: Store base activation $f ( \mathbf { x } ) \in \mathbb { R } ^ { n }$ , base measure M, and target activation $f ( \tilde { \mathbf { x } } ) \in \mathbb { R } ^ { n }$   
2: Backward Pass: $\begin{array} { r } { \mathbf { g }  \frac { \partial M } { \partial f ( \mathbf { x } ) } } \end{array}$ // Compute gradient w.r.t base activation   
3: Attribution: $\Delta M \gets \mathbf { g } \cdot ( f ( \tilde { \mathbf { x } } ) - f ( \mathbf { x } ) )$ // Compute linear approximation   
4: return ∆M

## CIRCUIT TRACKING

Circuit tracking aims to identify model components and connections, and reconstruct circuits that link internal activations to observed behaviors [3].

Model components are sequentially subjected to causal interventions, such as zero, mean, or interchange ablation, to verify their contribution to a specific model behavior and discover circuits. Unlike causal intervention techniques where components are checked independently, circuit tracking iteratively applies interventions to identify a circuit. Automated circuit discovery [3], whose procedure is detailed in Algorithm 2, simplifies neural networks by applying activation patching to the edges of the model computational graph, which correspond to the activations passed between network components. Processing the graph in reverse topological order, the algorithm removes edges whose ablation only marginally degrades the model output. The removal criterion is based on the KL divergence between the original model prediction and the prediction from the ablated circuit. Let $C _ { i - 1 }$ be the circuit at the previous iteration $i - 1$ and $C _ { i - 1 } \backslash \{ f _ { i } \} ( \mathbf { x } )$ denote the output distribution of the circuit when the activation $f _ { i } ( \mathbf { x } ) \in \mathbb { R } ^ { n }$ of component $f _ { i }$ is ablated. For an input $\mathbf { x } \in \mathbb { R } ^ { d }$ and the predefined threshold $\tau > 0$ , the circuit is updated as follows:

Algorithm 2 Automated Circuit Discovery   
Input: Model F, input $\mathbf { x } \in \mathbb { R } ^ { d }$ , threshold $\tau > 0 .$ , components $\{ f _ { 1 } , \dotsc , f _ { E } \}$ in reverse topological order   
Output: Circuit C   
1: $C _ { 0 } \gets F$ // Initialize the circuit   
2: for $i = 1$ to E do   
3: $C _ { i } \gets C _ { i - 1 } \setminus \{ f _ { i } \}$ // Temporarily remove $f _ { i }$   
4: $D \gets D _ { K L } ( F ( \mathbf { x } ) | | C _ { i } ( \mathbf { x } ) )$ // Compute the KL Divergence   
5: if $D < \tau$ then   
6: Keep $f _ { i }$ removed from $C _ { i }$   
7: else   
8: $C _ { i } \gets C _ { i - 1 }$ // Retain f<sub>i</sub>   
9: end if   
10: end for   
11: return $C  C _ { E }$

$$
\begin{array}{c} C _ { i } = \left\{ C _ { i - 1 } \setminus \{ f _ { i } \} \quad { \mathrm { i f } } \ D _ { K L } ( F ( \mathbf { x } ) \parallel C _ { i - 1 } \setminus \{ f _ { i } \} ( \mathbf { x } ) ) < \tau  \\ { C _ { i - 1 } \qquad \mathrm { o t h e r w i s e } } \end{array} \right. \quad { \mathrm { f o r } } \ i = 1 , \cdots , E ,\tag{6}
$$

where $F ( \mathbf { x } )$ is the output distribution of the original neural network and E is the number of edges.

However, iterating over all the components of a deep neural network requires high computational time, especially when executing interchange ablation, where a base and a target run are required for each iteration (Figure 4). Therefore, edge attribution patching [13] aims to discover circuits by employing attribution patching [26] across all the model activations and sorting them in descending order based on the $\Delta M$ scores (Equation (5)). The top k activations will be part of the final circuit. Figure 5 illustrates an end-to-end application of edge attribution patching with zero ablation for the image reconstruction task. We analyze a convolutional autoencoder with 6 layers, which was trained for 15 epochs on the CIFAR-10 dataset, achieving a peak signalto-noise ratio of 24 and Mean Squared Error (MSE) of 0.004. The images from CIFAR-10 are perturbed by exploiting zero-mean Gaussian noise with standard deviation of 0.15. As depicted in Figure 5(a), we report the minimum, maximum, and mean values of the first layer activation, although $\Delta M$ is estimated for every layer. We save all the activations and the performance measure M during the forward pass. We use the MSE as the measure M. We adopt zero ablation and replace the corrupted activation with a zero tensor of the same dimension, and compute the $\Delta M$ score as in Equation (5). In Figure 5(b), we plot the estimated $\Delta M$ for each layer. For this experiment, we choose to keep the top-3 layers in the circuit responsible for the reconstructed image, since the scores for these layers are much higher than those of the other layers. Therefore, the final circuit is made up of layer 1 $( \Delta M = 0 . 0 0 0 5 8 )$ , layer $5 ~ ( \Delta M = 0 . 0 0 0 5 9 )$ , and layer 6 $( \Delta M = 0 . 0 0 0 8 9 )$

Beyond empirical extraction, circuit tracking can establish lower bounds on model performance by ablating specific attention matrices to infer the causal importance of individual tokens [30]. Furthermore, circuit tracking analyzes whether neural networks can generalize across data distribution shifts [31]. Activation patching is applied to verify if the target state is altered when providing the model with the perturbed input. If the state is altered, then it will be part of the final circuit made up of nodes and connections responsible for generalization. Interestingly, models trained with grokking, consisting of an extended training far beyond overfitting [31], have higher generalization performance to data unseen during training than models trained for fewer epochs.

①Save activations and metric M during the forward pass  
![](images/b51596103b76ecebc2379773f9fff230c5f858fb950d09bb17f5bcd27f1bd56d.jpg)  
Fig. 5. Edge attribution patching with zero ablation for the convolutional autoencoder at layer granularity. (a) Estimation of $\Delta M$ for the first encoder layer. The performance metric M is the Mean Squared Error (MSE). A scalar $\Delta M$ score is obtained by summing all elements of the resulting $\Delta M$ tensor. (b) $\Delta M$ for each layer. The top-k layers ( ) with the highest $\Delta M$ scores $( k = 3$ is chosen for this implementation) constitute the circuit responsible for reconstructing the input image. The remaining layers ( ) are excluded from the circuit.

## INTERPRETABILITY FRAMEWORKS

Frameworks combine interpretability methods to observe model activations, project them into monosemantic spaces, test the causal importance of these isolated features through targeted amplification or suppression, and track the interconnected pathways to identify circuits responsible for a particular behavior. This can be used to steer the model towards target behaviors, such as removing concepts that contribute to harmful consequences, reinforcing stereotypes, and exacerbating inequalities in various social contexts [24]. As shown in Figure 6, mechanistic interpretability categories interact within frameworks. Model activations observed through representation visualization are disentangled and tested through causal intervention. Then, if the causal intervention reveals such an activation as responsible for the model behavior, it will be part of the related circuit.

Certain interpretability frameworks bypass explicit feature disentanglement, directly linking representation visualization to causal intervention [27]–[29]. Rather than relying on monosemantic projection, these methods directly verify the functional relevance of specific components, such as vision transformer attention heads, by replacing them with corrupted activations derived from noisy inputs [32], [33]. While replacing an activation with the version from another forward pass enables the impact measurement of a component on model behavior, steering the corrupted activation toward its clean state while patching subsequent activations with the corresponding corrupted activations helps to visualize the component activation [28]. By passing such an isolated activation to an image generator, it is possible to produce an image containing the concepts learned by that component.

![](images/dd7601fdf4e0ecb62a6ff50d15ab4c8071fe5f79bc9031e245ba0b3c492c01c5.jpg)  
Fig. 6. Combining mechanistic interpretability categories at neuron granularity: activations are observed through visualization, isolated through disentanglement, tested for behavioral impact through causal intervention, and are part of the circuit responsible for the model behavior if the causal intervention on that activation changes the prediction.

Linear probing and activation patching can be used to uncover the internal mechanisms of automatic speech recognition models [34]. By training linear probes on intermediate activations, encoders for speech recognition build semantic representations, distinguishing between word categories with 85.6% accuracy by the final encoder layer. Probing the decoder residual stream enables the prediction of hallucinations with up to 93.4% accuracy. To investigate acousticcontextual trade-offs, activation patching is used to inject white noise into activations of the encoder. This intervention restores acoustic accuracy in 85% of error cases.

Pure representation visualization is fundamentally constrained by polysemanticity [21]. Because individual neurons frequently encode multiple, superimposed concepts, activation maps often yield ambiguous interpretations. Consequently, while representation visualization is effective for generating causal hypotheses about internal signals, verifying these hypotheses requires transitioning to active interventional techniques and representation disentanglement. Other interpretability frameworks rely on representation disentanglement combined with causal intervention to dissect and control model behavior [23], [24], [35]. These configuration maps embed polysemantic activations into isolated monosemantic features to identify the ones that drive specific behaviors. Subsequently, scaling the underlying signal amplitudes facilitates the mitigation of undesirable outputs. Because interventional signal modeling validates whether hypothesized monosemantic features are necessary components of the model internal processing mechanisms, the causal intervention approach is coupled with circuit tracking to construct reliable circuits [3], [13], [30], [36]. Furthermore, to build these causal chains from the ground up, representation visualization frequently precedes both causal intervention and circuit tracking to preliminarily detect candidate signals of interest [31]. This comprehensive integration allows candidate features to be isolated via initial visual probing and tested through interventional ablation.

## PERSPECTIVES

Based on the above discussions, we finally present open challenges in causality research for deep neural networks, which include generalization, alignment of internal representations with human-interpretable concepts, model robustness, efficiency, and sensitivity.

## Generalization

Mechanistic interpretability may help address the performance degradation of deep neural networks when employed in out-of-distribution scenarios [37]. This generalization problem occurs when deep models learn dataset artifacts to minimize the loss function. Through targeted causal interventions, model activations that respond when the model is employed in out-of-distribution scenarios can be identified, constraining the forward pass to rely exclusively on these causal subgraphs. Such identified activations can also be rewarded at training time through tailored loss regularizations, steering the model to rely on the internal components that increase generalization performance. This would prevent the network from exploiting training data artifacts, providing better performance on out-of-distribution scenarios.

## Alignment of internal representations with human-interpretable concepts

While representation disentanglement techniques can produce monosemantic components [10], [25], these extracted signals may lack semantic grounding [4]. To map model components to human-understandable concepts, structured concept supervision could be integrated to steer the discovery of interpretable features. To validate these mappings and ensure that an isolated signal aligns with a human concept, causal interventions could then be integrated with human-in-theloop feedback or automated concept probers [8].

## Robustness

Certified robustness techniques aim to provide the accuracy lower bound of a model against any adversarial attack in a predefined perturbation region of the input [39]. Interval bound certification methods propagate this perturbation region layer by layer [38]. Since they require a single forward pass, these methods are computationally efficient but suffer from propagation errors on the estimated intervals, resulting in inaccurate bounds as network depth increases. Randomized smoothing methods can deliver accurate certifications for deeper architectures, but are characterized by high computational cost due to the extensive number of forward passes required to produce the certification for a single input [39]. Mechanistic interpretability potentially supports a certification process by isolating specific model components responsible for certification failure. Causal intervention techniques can map the propagation of adversarial noise through the neural network and isolate the specific model components that amplify malicious signals. Targeted interventions can be applied to these vulnerable components, improving the model certified robustness without incurring the high computational costs of smoothing the entire network.

## Efficiency

Model pruning identifies components of a deep neural network with low impact on accuracy [40]. To isolate prunable components, traditional magnitude-based methods remove trainable parameters relying on their norm. Although such methods can yield significant reductions in parameter count and computational overhead, performance may degrade accordingly. Causal intervention can help analyze such causal impact to remove only the activation of a component identified as prunable. This would facilitate the generation of architecturally efficient models. For example, automated circuit discovery [3] removes edges in the model computational graph based on their causal contribution to the model behavior to obtain the representative circuit. This procedure could be directly applied to the weights of the neural network. However, iteratively testing the causal impact of each model component requires high computational time. As already mentioned, attribution patching [13] deals with this efficiency problem by computing the gradient of the performance measure we want to consider with respect to the model activations, limiting the accuracy in the identification of activations that are causally responsible for the model behavior.

## Sensitivity

The sensitivity of mechanistic techniques to implementation choices is an open challenge. The causal responsibility attributed to a specific component often depends on the data used during intervention. A component identified as responsible for a behavior under one set of inputs may exhibit negligible causal impact under different data [31]. Circuit tracking approaches inherit this data sensitivity. Since circuit extraction relies on causal intervention, the resulting circuit may vary across input data. Furthermore, circuit discovery methods rely on thresholds to determine component inclusion in the circuit [3], [13]. Small perturbations to these thresholds can yield different structures. Addressing these limitations is an important research direction for the mechanistic interpretability of deep neural networks.

## REFERENCES

[1] V. Monga, Y. Li, and Y. C. Eldar, ”Algorithm unrolling: interpretable, efficient deep learning for signal and image processing,” IEEE Signal Process. Mag., vol. 38, no. 2, pp. 18-44, 2021, doi: 10.1109/MSP.2020.3016905.

[2] J. -M. Lemercier, J. Richter, S. Welker, E. Moliner, V. Valim¨ aki, and T. Gerkmann, ”Diffusion mod-¨ els for audio restoration: a review,” IEEE Signal Process. Mag., vol. 41, no. 6, pp. 72-84, 2024, doi: 10.1109/MSP.2024.3445871.

[3] A. Conmy, A. N. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso, “Towards automated circuit discovery for mechanistic interpretability,” in Proc. Adv. Neural Inf. Process. Syst., New Orleans, LA, USA, 2023.

[4] J. Ji et al., ”AI alignment: a contemporary survey”, ACM Comput. Surv., 58, 5, Article 132, 2025, doi: 10.1145/3770749.

[5] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in Proc. 34th Int. Conf. Mach. Learn., Sydney, NSW, Australia, 2017, pp. 3319–3328, doi: 10.5555/3305890.3306024.

[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ”Grad-CAM: visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput Vis., Venice, Italy, 2017, pp. 618-626, doi: 10.1109/ICCV.2017.74.

[7] L. Bereska and E. Gavves, ”Mechanistic interpretability for AI safety–a review”, Trans. Mach. Learn. Res, 2024.

[8] D. Bau, B. Zhou, A. Khosla, A. Oliva, and A. Torralba, ”Network Dissection: Quantifying Interpretability of Deep Visual Representations,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Honolulu, HI, USA, 2017, pp. 6541–6549, doi:10.1109/CVPR.2017.354

[9] Nostalgebraist, “Interpreting GPT: the logit lens,” AI Alignment Forum, 2020. [Online]. Available: https://www. lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

[10] R. Huben et al., ”Sparse autoencoders find highly interpretable features in language models”, in Proc. Int. Conf. Learn. Represent., Vienna, Austria, 2024.

[11] G. Ilharco, M. T. Ribeiro, M. Wortsman, S. Gururangan, L. Schmidt, H. Hajishirzi, and A. Farhadi, ”Editing models with task arithmetic,” in Proc. Int. Conf. Learn. Represent., Kigali, Rwanda, 2023.

[12] F. Zhang and N. Nanda, “Towards best practices of activation patching in language models: metrics and methods,” in Proc. Int. Conf. Learn. Represent., Vienna, Austria, 2024.

[13] A. Syed, C. Rager, and A. Conmy, “Attribution patching outperforms automated circuit discovery,” in Proc. BlackboxNLP Workshop, Miami, FL, USA, 2024, pp. 407–416, doi:10.18653/v1/2024.blackboxnlp-1.25.

[14] S. Somvanshi et al., ”Bridging the black box: a survey on mechanistic interpretability in AI,” ACM Comput. Surv., vol. 58, no. 8, 2026, doi:10.1145/3787104.

[15] N. Bai, R. A. Iyer, T. Oikarinen, and T. W. Weng, “Interpreting neurons in vision networks with language models,” Trans. Mach. Learn. Res., 2025.

[16] K. J. Piczak, “ESC: Dataset for Environmental Sound Classification,” In Proc. Ann. ACM Conf. on Multimedia, Brisbane, Australia, 2015, doi:10.1145/2733373.2806390.

[17] R. S. Zimmermann, T. Klein, and W. Brendel, “Scale alone does not improve mechanistic interpretability in vision models,” in Proc. Adv. Neural Inf. Process. Syst., New Orleans, LA, USA, 2023, doi: 10.5555/3666122.3668645.

[18] S. Jain, E. S. Lubana, K. Oksuz, T. Joy, P. H. S. Torr, A. Sanyal, and P. K. Dokania, “What makes and breaks safety fine-tuning? A mechanistic study,” in Proc. Adv. Neural Inf. Process. Syst., Vancouver, BC, Canada, 2024, doi:10.5555/3737916.3740879.

[19] Y. Ma, X. Lu, J. Sang, X. Jiang, and R. Li, ”Behind the scenes: Mechanistic Interpretability of lora-adapted whisper for speech emotion recognition,” In Proc. IEEE Int. Conf. Acoust., Speech and Signal Process., Barcelona, Spain, 2026, pp. 5286-5290. doi:10.1109/ICASSP55912.2026.11464049.

[20] N. Elhage et al., ”Toy models of superposition,” 2022, arXiv:2209.10652.

[21] C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter, ”Zoom in: an introduction to circuits,” Distill, 2020, doi: 10.23915/distill.00024.001.

[22] M. L. Olson et al., “Analyzing hierarchical structure in vision models with sparse autoencoders,” in Proc. Comput. Vis. Pattern Recognit. Conf. Workshops, Nashville, TN, USA, 2025, pp. 4835-4839, doi:10.1109/CVPRW67362.2025.00473.

[23] B. Cywinski and K. Deja, “SAeUron: interpretable concept unlearning in diffusion models with sparse´ autoencoders,” in Proc. Int. Conf. Mach. Learn., 2025.

[24] Y. Shi et al., “Dissecting and mitigating diffusion bias via mechanistic interpretability,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Nashville, TN, USA, 2025, pp. 8192–8202, doi:10.1109/CVPR52734.2025.00767.

[25] R. Hesse, J. Fischer, S. Schaub-Meyer, and S. Roth, “Disentangling polysemantic channels in convolutional neural networks,” in Proc. Comput. Vis. Pattern Recognit. Conf. Workshops, Nashville, TN, USA, 2025, pp. 4799-4803, doi:10.1109/CVPRW67362.2025.00466.

[26] N. Nanda, ”Attribution patching: activation patching at industrial scale,” 2023. [Online]. Available: https://www. neelnanda.io/mechanistic-interpretability/attribution-patching.

[27] C. Neo, L. Ong, P. Torr, M. Geva, D. Krueger, and F. Barez, “Towards interpreting visual information processing in vision-language models,” in Proc. Int. Conf. Learn. Represent., Singapore, 2025.

[28] R. Takatsuki, S. Joseph, I. Fujisawa, and R. Kanai, “Decoding vision transformers: the diffusion steering

lens,” in Proc. Comput. Vis. Pattern Recognit. Conf. Workshops, Nashville, TN, USA, 2025, pp. 4819-4824, doi: 10.1109/CVPRW67362.2025.00470.

[29] S. Xu, L. Pang, Y. Zhu, H. Shen, and X. Cheng, “Cross-modal safety mechanism transfer in large vision-language models,” in Proc. Int. Conf. Learn. Represent., Singapore, 2025.

[30] J. Gross et al., “Compact proofs of model performance via mechanistic interpretability,” in Proc. Adv. Neural Inf. Process. Syst., Vancouver, BC, Canada, 2024, doi:10.5555/3737916.3740439.

[31] B. Wang, X. Yue, Y. Su, and H. Sun, “Grokking of implicit reasoning in transformers: A mechanistic journey to the edge of generalization,” in Proc. Adv. Neural Inf. Process. Syst., Vancouver, BC, Canada, 2024, doi:10.5555/3737916.3740933.

[32] Q. Li, Z. Ye, X. Feng, W. Zhong, W. Ma, and X. Feng, “Causal tracing of object representations in large vision language models: mechanistic interpretability and hallucination mitigation,” in Proc. AAAI Conf. Artif. Intell., Singapore, 2026, vol. 40, no. 37, pp. 31645-31653, doi:10.1609/aaai.v40i37.40431.

[33] J. Chen, X. Wang, Z. Yao, Y. Bai, L. Hou, and J. Li, “Towards understanding safety alignment: a mechanistic perspective from safety neurons,” in Proc. Adv. Neural Inf. Process. Syst., San Diego, CA, USA, 2025.

[34] N. Glazer et al., ”Beyond transcription: Mechanistic interpretability in ASR,” In Proc. AAAI Conf. Artif. Intell., Singapore, 2026, vol. 40, no. 44, pp. 37407–37416. doi:10.1609/aaai.v40i44.41073.

[35] X. Chen, A. Plaat, and N. van Stein, “How does chain of thought think? Mechanistic interpretability of chainof-thought reasoning with sparse autoencoding,” in Proc. AAAI Conf. Artif. Intell., Singapore, 2026, vol. 40, no. 36, pp. 30297-30305, doi: 10.1609/aaai.v40i36.40281.

[36] R. Gupta, I. A. Moreno, T. Kwa, and A. Garriga-Alonso, “Interpbench: semi-synthetic transformers for evaluating mechanistic interpretability techniques,” in Proc. Adv. Neural Inf. Process. Syst., Vancouver, BC, Canada, 2024, doi:10.5555/3737916.3740866.

[37] Y. -F. Zhang, Z. Zhang, D. Li, Z. Jia, L. Wang, and T. Tan, ”Learning domain invariant representations for generalizable person re-identification,” IEEE Trans. Image Process., vol. 32, pp. 509-523, 2023, doi: 10.1109/TIP.2022.3229621.

[38] Z. Shi, Y. Wang, H. Zhang, J. Yi, and C.-J. Hsieh, ”Fast certified robust training with short warmup,” In Proc. Adv. Neural Inf. Process. Syst., Online, 2021, doi:10.5555/3540261.3541663.

[39] J. Cohen, E. Rosenfeld, and Z. Kolter, ”Certified adversarial robustness via randomized smoothing,” in Proc. Int. Conf. Mach. Learn., Long Beach, CA, USA, 2019.

[40] A. Brateanu, R. Balmez, A. Avram, C. Orhei, and C. Ancuti, ”LYT-NET: lightweight YUV transformer-based network for low-light image enhancement,” IEEE Signal Process. Lett., vol. 32, pp. 2065-2069, 2025, doi: 10.1109/LSP.2025.3563125.

## BIOGRAPHIES

Lorenzo Cirillo received the B.S. degree in computer and control engineering, the M.S. degree in artificial intelligence and robotics, earning the Excellent Student Award, from the Sapienza University of Rome, Italy, in 2022 and 2024, respectively, where he is currently pursuing the Ph.D. degree in cybersecurity at the AlcorLab. In 2023, he was an intern researcher at the Bank of Italy, working on knowledge graphs and graph databases. He is a member of the EURASIP Biometrics, Data Forensics, and Security Technical Area Committee. His research interests include deep learning, computer vision, adversarial machine learning, and explainable artificial intelligence.

Irene Amerini is an Associate Professor at DIAG, Sapienza University of Rome, where she leads the Computer Vision and Multimedia Forensics Research Team at ALCORLab. She was an Assistant Professor at Sapienza and a postdoctoral researcher at the Image Forensics and Security Lab, University of Florence. She received her Ph.D. from the University of Florence in 2011 and held visiting positions at Binghamton University and Charles Sturt University. She has served in IEEE, EURASIP, and IAPR committees, and is Senior Area Editor for IEEE Signal Processing Letters. Her research focuses on trustworthy AI, computer vision, adversarial learning, multimedia forensics.

Andrea Cavallaro is a Full Professor at EPFL and a Fellow of the IAPR, ELLIS, and the Higher Education Academy. His research focuses on machine learning for multimodal perception, computer vision, machine listening, and information privacy. He served as an IEEE Signal Processing Society Distinguished Lecturer and as Chair of the IEEE Image, Video and Multidimensional Signal Processing Technical Committee. Furthermore, he has served as an Area, Associate, and Guest Editor for numerous leading journals and magazines, including IEEE Transactions on Image Processing, Signal Processing, and Multimedia, and the IEEE Signal Processing Magazine

November 11, 2026