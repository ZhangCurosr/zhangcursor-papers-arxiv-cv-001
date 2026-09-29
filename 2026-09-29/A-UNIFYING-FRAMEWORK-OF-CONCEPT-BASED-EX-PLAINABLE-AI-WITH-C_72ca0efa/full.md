# A UNIFYING FRAMEWORK OF CONCEPT-BASED EX-PLAINABLE AI WITH COMPLETENESS GUARANTEES

Vojtech Kˇ ur˚ Adam Kukuckaˇ

Toma´s Brˇ azdil´ V´ıt Musil

Masaryk University, Faculty of Informatics, Brno, Czechia

## ABSTRACT

Concept-based explanations describe neural network predictions through humanunderstandable properties of inputs called concepts. The field encompasses approaches that differ in how they define and represent concepts and connect them to model predictions. We introduce a theoretical framework that describes these approaches in a common mathematical language and supports a shared analysis of their properties. For concept discovery, which identifies concepts automatically within a latent space of a trained model, we employ a concept autoencoder view. An encoder extracts concept representations from the model’s latent space, and a decoder uses them to reconstruct the original latent representation. The autoencoder’s reconstruction error measures how accurately its decoder recovers the original latent representation. We revisit model completeness: how well the concepts can reproduce the model’s outputs. We show that model incompleteness of the concepts can be bounded by the autoencoder’s reconstruction error. The autoencoder view also provides a common way to define individual concept attributions, which measure each concept’s contribution to a prediction. We establish when these attributions sum to the model’s prediction, and bound the discrepancy otherwise, thus providing attribution completeness guarantees.

## 1 INTRODUCTION

Deep learning has driven substantial advances across many tasks through its ability to learn rich representations directly from data (LeCun et al., 2015). However, these representations are not inherently aligned with human-understandable concepts, making the resulting predictions difficult to interpret (Samek et al., 2017). Explainable artificial intelligence (XAI) seeks to make model behavior and predictions understandable to humans. Concept-based XAI (C-XAI) approaches this goal through properties of inputs called concepts, such as textures, shapes, or object parts (Poeta et al., 2025). For example, the presence of wheels may help explain why an image is classified as a vehicle.

C-XAI encompasses methods that differ in what a concept is, how it is represented, and how it relates to predictions (Poeta et al., 2025). These different uses and representations of concepts have motivated mathematical frameworks for relating and comparing concept-based methods (Fel et al., 2023a; Li et al., 2024; Poche et al.´ , 2025). We focus on two main problems in bridging C-XAI together: a) a common definition of a concept, and b) a suitably general definition of concept extraction. In the following, we present our framework of C-XAI.

We define a concept to be a function c : $\mathcal { X }  [ 0 , 1 ]$ , where  is the input space. Binary values indicate absence or presence of the concept in the input, while intermediate values allow graded descriptions. We write a model as $f = g \circ h \colon \mathcal { X } \stackrel { h } { \to } \mathcal { Z } \stackrel { g } { \to } \mathcal { Y }$ , where h maps inputs to latent representations and g maps these representations to predictions. We define a concept extractor as a map e : ${ \bar { z } }  [ 0 , 1 ]$ , so that $c = e \circ h$ defines a concept on inputs. However, existing methods can represent a single concept through a vector or a collection of spatial responses, containing information beyond its scalar presence (Vielhaben et al.,

![](images/41be51b1a566eadcb07f696fd4b703b2814cd70680807aca93bd29bf529048ec.jpg)  
Figure 1: Concept extraction.

2023; Fel et al., 2023b). We therefore separate the extracted representation from its activation by writing $e = \sigma \circ \pi ;$ a probe π : $\mathcal { Z }  \mathcal { V }$ produces the concept representation, while an activation function σ : $\nu  [ 0 , 1 ]$ measures its presence. This allows to be multidimensional while ensuring that $c = \sigma \circ \pi \circ$ h remains a concept under our input-level definition (Fig. 1).

![](images/7c8991936c7f0bb35517b05d54826d16cc9bc9f1554f07d6f366ac999c3d35de.jpg)  
Figure 3: Overview of the framework for a scalar model output. The decomposition $f = g \circ h$ connects inputs, latent representations, and predictions. Concept probes $\pi _ { i }$ and activation functions $\sigma _ { i }$ form extractors $e _ { i } = \sigma _ { i } \circ \pi _ { i }$ that estimate input-level concepts $c _ { i } .$ . The concept autoencoder $A = ( E , D )$ induces the reconstructed predictor $f _ { A }$ , while the attribution model $f _ { \varphi }$ adds the concept contributions to a baseline b. Solid arrows denote function evaluations; dashed links compare the indicated functions by RMSE over the input distribution. Model completeness instead considers the best recovery of f from the concept representations over prediction heads.

Unsupervised concept discovery methods learn probes $\pi _ { 1 } , \ldots , \pi _ { k }$ that together form an encoder $E ( z ) = ( \pi _ { 1 } ( z ) , \ldots , \pi _ { k } ( z ) ) $ . The encoder maps the model’s latent space  to the joint concept representation space $\begin{array} { r } { \mathcal { V } \stackrel { - } { = } \mathcal { V } _ { 1 } \times \cdots \times \mathcal { V } _ { k } = \bar { \prod _ { i } } \mathcal { V } _ { i } } \end{array}$ . Following earlier frameworks (Fel et al., 2023a; Poche et al.´ , 2025), we pair it with a decoder $D \colon \mathcal { V }  \mathcal { Z }$ that reconstructs the latent representation, forming a concept autoencoder $A = ( E , D )$ . The autoencoder also defines the induced concept autoencoder model (ICAM) $f _ { A } = g \circ D \circ E \circ h$ . Denoting $\gamma = g \circ D$ and $C = E \circ h .$ , we have $f _ { A } = \gamma \circ C ( \operatorname { F i g } . 2 )$

![](images/f560d07e74d6a2a5f08eb6605dda1f6e6b970dc16506ddc83e7ebc282fa61583.jpg)  
Figure 2: Concept autoenc.

We measure the reconstruction error RE(A) by the root mean squared We measure the reconstruction error RE(A) by the root mean squared error (RMSE) between h and $D \circ E \circ h$ over a distribution of inputs. Following ICE’s comparison of reconstructed and original predictions (Zhang et al., 2021), we measure fidelity error $\operatorname { F E } ( A )$ by the RMSE between $f$ and $f _ { A }$ . We revisit model completeness: whether the concept representations $E ( h ( x ) )$ retain enough information to recover the model’s outputs $f ( x )$ (Yeh et al., 2020). We define the model completeness error MCE(A) as the infimum of the RMSE between f and $\eta \circ C .$ , over all eligible η : $\mathcal { V } \stackrel { - } {  } \mathcal { V } , \mathtt { i . e . }$

$$
\operatorname { M C E } ( A ) = \operatorname* { i n f } _ { \eta } \operatorname { R M S E } ( f , \eta \circ C ) .\tag{1}
$$

Choosing $\eta = \gamma \mathrm { g i v e s } f _ { A }$ , so fidelity error bounds MCE. If g is $L _ { g ^ { - } }$ Lipschitz continuous, a change in the latent representation produces at most a proportional change in the output. Together this makes the following chain of inequalities:

$$
\operatorname { M C E } ( A ) \leq \operatorname { F E } ( A ) \leq L _ { g } \operatorname { R E } ( A ) .\tag{2}
$$

For a scalar output $\begin{array} { r } { \mathcal { V } = \mathbb { R } \left( \mathrm { e . g . } \right. } \end{array}$ , a class score), an attributionfunction $\varphi = ( \varphi _ { 1 } , \ldots , \varphi _ { k } ) \colon \mathcal { X } \to \mathbb { R } ^ { k }$ assigns each concept a score intended to describe its contribution to the prediction (Fel et al., 2023a; Poche et al.´ , 2025). We also revisit attribution completeness: whether concept attributions together with a baseline sum to $f ( x )$ (Sundararajan et al., 2017; Vielhaben et al., 2023). We measure the discrepancy by the attribution error $\operatorname { A T E } ( A , \varphi )$ , the RMSE between $f$ and $f _ { \varphi }$ where $\begin{array} { r } { f _ { \varphi } ( x ) = b + \sum _ { i = 1 } ^ { k } \dot { \varphi } _ { i } ( \dot { x } ) } \end{array}$ and b is a baseline value. We consider three attribution rules: occlusion and gradient-times-input, as defined by Fel et al. (2023a); Poche et al.´ (2025), and insertion.

The three attribution rules are defined using the concept autoencoder A. For example, occlusion measures the difference between $f _ { A } ( x )$ and the same value with the i-th concept set to zero. We make the relationship of $\varphi$ to A explicit using the triangle inequality

$$
\mathrm { A T E } ( A , \varphi ) = \mathrm { R M S E } ( f , f _ { \varphi } ) \leq \mathrm { R M S E } ( f , f _ { A } ) + \mathrm { R M S E } ( f _ { A } , f _ { \varphi } ) = \mathrm { F E } ( A ) + \mathrm { A D D } ( A , \varphi ) ,\tag{3}
$$

where $\mathrm { A D D } ( A , \varphi ) = \mathrm { R M S E } ( f _ { A } , f _ { \varphi } )$ is the additivity error. In the case that $\gamma$ is affine, for example when D is linear and g is a final linear layer, we show that all three rules are equal and $\mathrm { A D D } ( A , \varphi ) =$ 0. This means that the attribution rules perfectly attribute the ICAM, and their error with respect to the original model can be bounded in the same way as MCE in Eq. (2). Finally, we also show a bound on the additivity error when $\gamma$ is not affine using a bound on the curvature of $\gamma$ . The key terms are summarized in Fig. 3.

Contributions We present a unifying mathematical framework for concept-based explanation methods. This abstraction enables general completeness guarantees for the concept representations and attributions, under explicit assumptions. Our contribution is the shared formulation and analysis that support these guarantees, rather than a new explanation method. Specifically, our contributions are threefold:

1. A common formulation of concept-based explanation. We distinguish concepts on inputs, their potentially multidimensional representations, and scalar activations within a shared formulation of concept testing, discovery, and bottleneck models.

2. Reconstruction-based model completeness guarantees. We show that fidelity error bounds model completeness error and is itself bounded by reconstruction error under a Lipschitz assumption on the prediction head. Exact reconstruction yields zero fidelity and model completeness errors.

3. Attribution completeness through fidelity and additivity. We bound attribution error by fidelity error plus additivity error within the ICAM. For affine decoded prediction heads, insertion, occlusion, and gradient-times-input give identical contributions with zero additivity error. For nonaffine decoded heads with a Lipschitz gradient, we bound additivity error using curvature and the magnitude of the concept representations.

## 2 PROPOSED FRAMEWORK

In this section, we give the formal description of our framework.

Concept A concept is a function $\mathcal { X }  [ 0 , 1 ]$ , where  is an input space. In the following, we assume that is fixed.

Model A model is a function $\mathcal { X }  \mathcal { V }$ , where is the output space. A decomposition of a model f is $( \mathcal { Z } , h , g )$ where  is a latent space, $h \colon \mathcal { X } \to \mathcal { Z }$ is a latent extractor, $g \colon { \mathcal { Z } } \to \mathcal { V }$ is a prediction head, and $f = g \circ h$ . In the following, we assume the model and its decomposition are fixed.

Concept Extractor A concept extractor is a function ${ \mathcal { Z } }  [ 0 , 1 ]$ . A decomposition of a concept extractor e is $( \nu , \pi , \sigma )$ , where  is a representation space, $\pi \colon \mathcal { Z }  \mathcal { V }$ is a concept probe, $\sigma : \mathcal { V } $ [0, 1] is a concept activation function, and $e = \sigma \circ \pi$

Concept Autoencoder A k-concept autoencoder is $( E , D )$ , where $E = ( \pi _ { 1 } , \ldots , \pi _ { k } ) \colon \mathcal { Z }  \mathcal { V }$ is a k-concept encoder, $\mathcal { V } = \mathcal { V } _ { 1 } \times \cdots \times \mathcal { V } _ { k }$ is a product representation space, $\pi _ { i } \colon \mathcal { Z }  \mathcal { V } _ { i }$ is the i-th concept probe, and $D \colon \mathcal { V } \to \mathcal { Z }$ is a k-concept decoder. An induced concept autoencoder model (ICAM) $f _ { A }$ of a concept autoencoder $A = ( E , { \bar { D } } )$ is $g \circ D \circ E \circ h$

## 3 OVERVIEW OF C-XAI

Concept-based explanations differ in how concepts enter the model, how they are obtained, and what the explanation describes (Poeta et al., 2025). Concepts may participate in prediction (antehoc) or be used to analyze a trained model (post-hoc). They may be specified through annotations or descriptions (supervised), or discovered without concept labels (unsupervised). In this section, we show how these methods can be described using our framework; Tab. 1 in Sec. A summarizes the instantiations.

## 3.1 EXPLAINABLE-BY-DESIGN MODELS

Concept-based models use concepts directly in prediction. We distinguish their architectures independently of how the concepts are obtained or the models are trained.

A concept bottleneck model (CBM) is a model f paired with a decomposition $( \mathcal { Z } , h , g )$ of f where $\mathcal { Z } ~ = ~ [ 0 , 1 ] ^ { \hat { k } }$ and $\begin{array} { r l } { h } & { { } = } \end{array}$ $( c _ { 1 } , \ldots , c _ { k } )$ . Hence, each $\begin{array} { l } { { \displaystyle c _ { i } ~ = ~ \pi _ { i } \circ h } } \end{array}$ is a concept, where $\pi _ { i } ( z ) = z _ { i }$ is the i-th coordinate projection, and the prediction head $g \colon [ 0 , 1 ] ^ { k }  \mathcal { V }$ receives their activations (Fig. 4).

![](images/e5575cb6159a063b550daf34773545d6cf6b31cb94f6c341064a775cb0677c5e.jpg)  
Figure 4: Bottleneck model.

A concept embedding model (CEM) is a model f paired with a decomposition $( \mathcal { Z } , h , g )$ where $\mathcal { Z } = \mathcal { V } _ { 1 } \times \cdots \times \mathcal { V } _ { k }$ and k concept extractors $e _ { i }$ with decompositions $( \nu _ { i } , \pi _ { i } , \sigma _ { i } )$ , where $\pi _ { i } ( z ) = z _ { i } .$ Each probe $\pi _ { i }$ selects a concept representation $\pi _ { i } ( h ( x ) ) \in \mathcal { V } _ { i } ,$ and $\sigma _ { i } \colon \mathcal { V } _ { i } \  \ [ 0 , 1 ]$ maps it to a scalar activation. Thus $c _ { i } =$ $\sigma _ { i } \circ \pi _ { i } \circ h$ is a concept (Fig. 5). The prediction head g receives the representations $h ( x )$ themselves. Every CBM is a CEM with $\mathcal { V } _ { i } = \left[ 0 , 1 \right]$ and identity activations.

![](images/58e77a4c2235f23906854b5e0e51b27827e00b428d868f3a37f740044bb7ff37.jpg)  
Figure 5: Embedding model.

The probability-based variants of Koh et al. (2020) and the prototype-presence constructions of ProtoTree and PIP-Net give CBM decompositions (Nauta et al., 2021; 2023). Concept logits, as used in the sequential and joint classification variants of Koh et al. (2020), illustrate scalar CEM representations. This construction also accommodates the scalar projections of Post-hoc CBMs without a residual branch and Label-free CBMs (Yuksekgonul et al., 2023; Oikarinen et al., 2023), and the similarity scores of ProtoPNet and LaBo (Chen et al., 2019; Yang et al., 2023). The vector-valued embeddings of Espinosa Zarlenga et al. (2022) illustrate multidimensional CEM representations, which can retain predictive information beyond their scalar concept activations.

Example: Supervised CBMs and CEMs We express the classification models of Koh et al. (2020) in our framework. The user specifies k binary concepts and supplies training data $\{ ( x _ { j } , y _ { j } , a _ { j } ) \} _ { j = 1 } ^ { n }$ , where $x _ { j } \in { \mathcal { X } }$ is an input, $y _ { j } \in \{ 1 , \ldots , N \}$ is its class label, and $a _ { j } \in \{ 0 , 1 \} ^ { k }$ contains its concept annotations. The entry $a _ { j , i }$ indicates whether concept i is present in input $x _ { j }$

In the probability-bottleneck variant, the method learns concept predictors ${ \hat { c } } _ { i } \colon \mathcal { X } \to [ 0 , 1 ]$ and a prediction head $\dot { g } \colon [ 0 , 1 ] ^ { k } \to \mathbb { R } ^ { N }$ producing class logits, so $\mathcal { V } \overset { \mathbf { \bar { \rho } } } { = } \mathbb { \bar { R } } ^ { N }$ . Concept annotations encourage $\hat { c } _ { i } ( x _ { j } ) \approx a _ { j , i } ,$ while class labels encourage $g ( \hat { c } _ { 1 } ( x _ { j } ) , \dots , \hat { c } _ { k } ( x _ { j } ) )$ to predict $y _ { j } .$ . The resulting model predicts entirely through these concept probabilities and therefore has the CBM decomposition $( [ \overbrace { 0 } , 1 ] ^ { k } , ( \hat { c } _ { 1 } , \ldots , \overbar { { c } } _ { k } ) , g )$

The other variants instead learn a concept-logit predictor $h \colon \mathcal { X } \ \to \ \mathbb { R } ^ { k }$ and a prediction head $g \colon \mathbb { R } ^ { k } \ \to \ \mathbb { R } ^ { N }$ that receives the logits directly. This gives the model decomposition $( \mathbb { R } ^ { k } , h , g )$ The architecture uses the fixed sigmoid $s ( t ) \stackrel { . } { = } ( 1 + \exp ( - t ) ) ^ { - 1 }$ to obtain concept probabilities from the logits. In our notation, $\begin{array} { r } { \check { \nu _ { i } } = \mathbb { R } . } \end{array}$ , the probe $\pi _ { i } ( z ) = z _ { i }$ selects the representation of concept i, and $\sigma _ { i } = s$ gives its activation. Concept supervision encourages $e _ { i } ( h ( x _ { j } ) ) = s ( h ( x _ { j } ) _ { i } ) \approx a _ { j , i } $ while the prediction head receives $h ( x _ { j } )$ itself. The decomposition together with these extractors is therefore a CEM in our terminology.

## 3.2 POST-HOC CONCEPT-BASED EXPLANATIONS

Post-hoc methods explain a fixed model f with decomposition $( \mathcal { Z } , h , g )$ by identifying concepts in its latent representations and examining their relation to predictions.

## 3.2.1 CONCEPT DETECTION

Concept detection asks whether a trained model’s latent representations $h ( x )$ retain information about a concept of interest. For a user-specified concept $c \colon \mathcal { X }  [ 0 , 1 ]$ , we seek an extractor $e =$ $\sigma \circ \pi$ such that $e ( h ( x ) ) \approx c ( x )$ . Annotated examples allow this correspondence to be learned and evaluated.

The most direct approach associates concepts with existing model units. For $\mathcal { Z } = \Pi _ { i } \mathcal { V } _ { i }$ , projection probes $\pi _ { i } ( z ) = z _ { i }$ select neurons, spatial feature maps, or groups of units, whose concept responses are described by $\sigma _ { i }$ . This gives a CEM interpretation of the existing model decomposition. Network Dissection evaluates channel–concept associations against annotated spatial masks (Bau et al., 2017). CRP similarly treats channels as concept representations (Achtibat et al., 2023).

More generally, learned probes $\pi _ { i } \colon \mathcal { Z }  \mathcal { V } _ { i }$ can use the full latent representation. We can compare learned probes to study relationships between concepts, as in Net2Vec (Fong & Vedaldi, 2018), or evaluate class-score sensitivity along concept directions, as in TCAV (Kim et al., 2018). CAR extends detection to nonlinear probes and studies concept–class associations and input features supporting concept detection (Crabbe & Schaar´ , 2022).

Example: TCAV The user supplies a trained classifier $f \colon \mathcal { X } \to \mathbb { R } ^ { N }$ and selects a layer giving the decomposition $( \mathbb { R } ^ { d } , h , g )$ . They specify a binary concept c through labeled examples $\{ ( x _ { j } , a _ { j } ) \} _ { j = 1 } ^ { n } ,$ where $a _ { j } ~ = ~ c ( x _ { j } ) ~ \in ~ \{ 0 , 1 \}$ . In our framework, TCAV (Kim et al., 2018) uses an affine probe $\pi ( z ) = \mathbf { \bar { \langle } } w , z \mathbf { \rangle } + \mathbf { \bar { \boldsymbol { \beta } } }$ with representation space $\nu = \mathbb { R }$ . Taking $\sigma ( t ) = \mathbf { 1 } \{ t > 0 \}$ gives the concept extractor $e = \sigma \circ \pi$ . The method learns w and $\beta$ so that $e \circ h$ is as close as possible to c on the labeled examples.

## 3.2.2 UNSUPERVISED CONCEPT DISCOVERY

Unsupervised concept discovery constructs concept representations without annotations specifying their meanings. We use our framework to describe methods that admit reconstruction, followed by a concrete example using ICE. For a fixed model decomposition $( \mathcal { Z } , h , g )$ , these methods construct probes $E = ( \pi _ { 1 } , \cdot \cdot \cdot , \pi _ { k } )$ and a decoder D, forming a concept autoencoder $A = ( E , D )$ (Fel et al., 2023a; Poche et al.´ , 2025). Many methods fit these functions by minimizing reconstruction error together with regularization or constraints (Fel et al., 2023a; Bhusal et al., 2025). Other constructions provide exact reconstruction algebraically (Vielhaben et al., 2023). The discovered representations are then interpreted through representative images, spatial responses, or concept-specific saliency maps, allowing users to identify and name the concepts (Zhang et al., 2021; Fel et al., 2023b).

The decoder also identifies the predictor whose concept contributions are being explained. Writing $C = E \circ h$ and $\gamma = g \circ D$ , the reconstructed predictor is the ICAM $f _ { A } = \gamma \circ C$ . When activation functions $\sigma _ { i }$ are specified, this gives the CEM decomposition $( \nu , C , \gamma )$ , with input-level concepts $c _ { i } = \sigma _ { i } \circ \pi _ { i } \circ h$ . Standard feature-attribution methods can therefore treat each concept representation as a feature block of this predictor. For example, insertion and occlusion modify blocks of $C ( x )$ before evaluating $\gamma ,$ while gradient-times-input combines these blocks with derivatives of $\gamma$ (Fel et al., 2023a; Poche et al.´ , 2025). These attributions describe the reconstructed prediction $f _ { A } ( x )$ which may differ from the original prediction $f ( x )$ . The next section uses this distinction to study how well the concept representations and their attributions recover the original model’s outputs.

Example: ICE We express ICE’s NMF construction in our framework (Zhang et al., 2021) (see also Fig. 6 in Sec. A). The user supplies a trained classifier with $N$ classes, selects a class $r \in$ $\{ 1 , \ldots , N \}$ , provides inputs $X = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n } $ from that class, and chooses the number of concepts k. We explain the class logit $f \colon \mathcal { X }  \mathbb { R }$ using the nonnegative feature maps preceding global average pooling and the final affine classifier. This gives the decomposition $( \bar { \mathcal { Z } } , \bar { h } , g )$ with $\bar { \mathcal { Z } } = \mathbb { R } _ { \geq 0 } ^ { S \times d }$ , where S is the number of spatial locations and d is the number of feature channels. The prediction head is $\begin{array} { r } { g ( z ) = b + \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \langle w , z _ { s , : } \rangle } \end{array}$ , where $w \in \mathbb { R } ^ { d }$ and $b \in \mathbb { R }$ are the trained classifier’s parameters for class r. The original model remains fixed.

ICE stacks the supplied feature maps into $Z \in \mathbb { R } _ { > 0 } ^ { n S \times d }$ and fits a dictionary $U \in \mathbb { R } _ { > 0 } ^ { k \times d }$ and coefficients $H \in \mathbb { R } _ { \geq 0 } ^ { k \times n S }$ by minimizing $\| Z - H ^ { \top } U \| _ { F } ^ { 2 }$ . Each row of $U$ is a learned concept direction. Once U is fixed, a new input receives one coefficient per concept and spatial location. We therefore take $\mathcal { V } _ { i } = \mathbb { R } _ { > 0 } ^ { S }$ and identify $\begin{array} { r } { \mathcal { V } = \prod _ { i } \mathcal { V } _ { i } } \end{array}$ with $\mathbb { R } _ { \geq 0 } ^ { k \times S }$ by placing concept representations in rows. ICE’s reconstruction operation is our decoder $D ( V ) { \overset { - } { = } } V ^ { \top } U$ . Its numerical procedure for fitting coefficients with U fixed gives our encoder $E ( z )$ , seeking to minimize $\| z - D ( \dot { V } ) \| _ { F } ^ { 2 }$ over $V \in \mathcal V$ . The probes are $\pi _ { i } ( z ) = E ( z ) _ { i , : }$ , so each concept representation is an entire spatial coefficient map. ICE uses these maps to highlight image regions and their spatial means to select representative images for interpreting the concepts.

For an input x, ICE multiplies the spatial mean of $\pi _ { i } ( h ( x ) )$ by the class-specific weight $( U w ) _ { \ l }$ to obtain concept i’s contribution. Its explanation displays these quantities as the concept’s similarity score, weight, and contribution. Our general results in Sec. 4.3 explain why these contributions, together with the classifier bias, recover the reconstructed logit and how reconstruction quality controls their agreement with the original prediction.

## 4 COMPLETENESS GUARANTEES

We connect reconstruction quality to fidelity, model completeness, and attribution completeness. The results apply to the fitted encoder and decoder, independently of their learning objective or optimization procedure.

Common Assumptions and Notation Fix a model $f$ with decomposition $( \mathcal { Z } , h , g )$ , a concept autoencoder $A = \mathsf { \bar { ( } } E , D )$ , and a probability distribution p on $\mathcal { X }$ . Write $C = E \circ h$ and $\gamma = g \circ D$ thus $f _ { A } = \gamma \circ C$ . Assume that $\dot { \mathcal { y } } = \mathbb { R } ^ { N } , \mathbf { \bar { \mathcal { Z } } } \subseteq \mathbb { R } ^ { d }$ , that all relevant functions are measurable, and that the errors being bounded are finite. For functions $f _ { 1 } , f _ { 2 } \colon \mathcal { X }  \mathbb { R } ^ { m }$ , define

$$
\mathrm { R M S E } ( f _ { 1 } , f _ { 2 } ) = \sqrt { \mathbb { E } _ { x \sim p } \big [ \| f _ { 1 } ( x ) - f _ { 2 } ( x ) \| _ { 2 } ^ { 2 } \big ] } .\tag{4}
$$

All errors use this distribution; choosing p uniform on a dataset gives empirical guarantees on that dataset. Further assumptions are introduced where needed.

We use RMSE because it aggregates discrepancies across inputs while retaining the scale of the quantities being compared. It satisfies ${ \mathrm { R M S E } } ( f _ { 1 } , f _ { 2 } ) = \| f _ { 1 } - f _ { 2 } \| _ { L ^ { 2 } ( p ) }$ , where the $L ^ { 2 } ( p )$ norm is defined on square-integrable functions identified up to equality p-almost surely. We can therefore use properties of this norm, including the triangle inequality. The Euclidean norms measure discrepancies between individual representations in $\bar { z }$ or outputs in $\mathcal { V } ,$ while the $L ^ { 2 } ( p )$ norm turns these pointwise discrepancies into distances between functions. More general norms could be used with corresponding regularity assumptions; we adopt this setting to keep the statements and proofs easy to follow.

## 4.1 FIDELITY

The fidelity error measures output disagreement between $f$ and $f _ { A }$ . Thus we define $\operatorname { F E } ( A ) =$ RMSE $\left( f , f _ { A } \right)$ . The reconstruction error measures the disagreement between z and $D ( E ( z ) )$ where $z = h ( x )$ , so we define $\mathrm { R E } ( A ) = \mathrm { R M S E } ( h , D \circ C )$

Now observe that if $h ( x ) = D ( E ( h ( x ) ) ,$ ), then

$$
f ( x ) = g ( h ( x ) ) = g ( D ( E ( h ( x ) ) ) ) = f _ { A } ( x ) .\tag{5}
$$

From this, we can easily see that if $\operatorname { R E } ( A ) \ = \ 0$ , then $\operatorname { F E } ( A ) \ = \ 0$ as well. For approximate reconstruction, assume that $g$ is $L _ { g } .$ -Lipschitz for a finite constant $L _ { g } \geq 0 \colon$

$$
\| g ( z ) - g ( z ^ { \prime } ) \| _ { 2 } \leq L _ { g } \| z - z ^ { \prime } \| _ { 2 } \qquad \mathrm { f o r ~ a l l } ~ z , z ^ { \prime } \in \mathcal { Z } .\tag{6}
$$

This controls how strongly the prediction head can amplify reconstruction errors.

Theorem 1 (Reconstruction controls fidelity). Under the Lipschitz assumption in Eq. (6),

$$
\operatorname { F E } ( A ) \leq L _ { g } \operatorname { R E } ( A ) .\tag{7}
$$

For a proof see Sec. B.1.

This result reflects the construction of the ICAM: $f _ { A }$ replaces $h ( x )$ by its reconstruction $D ( E ( h ( x ) ) )$ before applying the same prediction head $g .$ The usefulness of this dwells in the formalization, as we show this is a stepping stone to more interesting results. The Lipschitz assumption holds for many common neural-network heads, including finite compositions of affine maps and Lipschitz activations such as ReLU. For deeper compositions, products of the individual layer bounds give a valid but potentially conservative constant.

## 4.2 MODEL COMPLETENESS

Model completeness asks how well the original outputs can be recovered from $C ( x )$ when the prediction head is allowed to vary. Let then Γ contain all functions $\nu \to \mathbb { R } ^ { N }$ . We define the model completeness error as $\begin{array} { r } { \mathrm { M C E } ( \boldsymbol { \dot { A } } ) = \operatorname* { i n f } _ { \eta \in \Gamma } \mathrm { R M S E } ( f , \eta \circ C ) } \end{array}$

Now, observe that $g \circ D \in \Gamma$ is an admissible head. The definition of the infimum therefore gives $\operatorname { M C E } ( A ) \leq \operatorname { R M S E } ( f , g \circ D \circ C ) = \operatorname { F E } ( A )$ . Combining this with Theorem 1, we get the following result.

Theorem 2 (Reconstruction controls completeness). Under the Lipschitz assumption in Eq. (6),

$$
\operatorname { M C E } ( A ) \leq \operatorname { F E } ( A ) \leq L _ { g } \operatorname { R E } ( A ) .\tag{8}
$$

The connection is not immediately obvious, as completeness concerns only the concept representations $E ( h ( x ) )$ , while reconstruction concerns both E and D. In particular, if $\mathrm { R E } ( A ) { \stackrel { - } { = } } 0$ , then the concept representations are complete. In this view, the reason is clear: E retains enough information for D to perfectly reconstruct the latent representation almost surely, and thus recover the original output. This exact-reconstruction conclusion does not require the Lipschitz assumption. The theorem also gives an additional interpretation to minimizing reconstruction error when learning A: under the Lipschitz assumption, it lowers an upper bound on the incompleteness of the concept representations.

## 4.3 ATTRIBUTION COMPLETENESS

Take scalar outputs, $N = 1$ , and assume that each $\mathcal { V } _ { i } \subseteq \mathbb { R } ^ { m _ { i } }$ contains its ambient zero vector $0 _ { i } .$ where $m _ { i } \geq 1$ . Equip $\begin{array} { r } { \mathcal { W } = \prod _ { i } \mathbb { R } ^ { m _ { i } } } \end{array}$ <sup>i</sup> with its Euclidean inner product and norm. Write $b = \gamma ( 0 \nu )$ where $0 _ { \mathcal { V } } = ( 0 _ { 1 } , \ldots , 0 _ { k } )$ . The isolation map $P _ { i } ( v _ { 1 } , \ldots , v _ { k } ) = ( 0 _ { 1 } , \ldots , 0 _ { i - 1 } , v _ { i } , 0 _ { i + 1 } , \ldots , 0 _ { k } )$ retains concept $i ;$ both $P _ { i } v$ and the representation $v - P _ { i } v$ with that concept removed belong to .

An attribution is a function $\varphi = ( \varphi _ { 1 } , \ldots , \varphi _ { k } ) \colon \mathcal { X } \to \mathbb { R } ^ { k }$ that assigns each concept a contribution to a prediction. We study three choices:

$$
\varphi _ { i } ( x ) = \gamma ( P _ { i } C ( x ) ) - b
$$

$$
( { \mathrm { I n s e r t i o n } } ) ,\tag{9}
$$

$$
\varphi _ { i } ( x ) = \gamma ( C ( x ) ) - \gamma ( C ( x ) - P _ { i } C ( x ) )
$$

$$
( { \mathrm { O c c l u s i o n } } ) ,\tag{10}
$$

$$
\varphi _ { i } ( x ) = \langle \nabla \gamma ( C ( x ) ) , P _ { i } C ( x ) \rangle
$$

$$
( \mathrm { G r a d i e n t - t i m e s - i n p u t } ) .\tag{11}
$$

Insertion adds a concept to the baseline, occlusion removes it from the full representation, and gradient-times-input pairs its coordinates with their output sensitivities. For the last rule, $\gamma$ must also be defined and differentiable on an open neighborhood of each $C ( x )$ in $\mathcal { W } ,$ , agreeing with $g \circ D$ on its intersection with .

An attribution function $\varphi$ induces an attribution model $f _ { \varphi } \colon \mathcal { X } \to \mathbb { R }$ , defined by

$$
f _ { \varphi } ( x ) = b + \sum _ { i = 1 } ^ { k } \varphi _ { i } ( x ) .\tag{12}
$$

The attribution error measures output disagreement between $f$ and $f _ { \varphi } { \mathrm { : } }$ , so we define $\operatorname { A T E } ( A , \varphi ) =$ $\operatorname { R M S E } ( f , f _ { \varphi } )$ . Similarly, the additivity error measures output disagreement between $f _ { A }$ and $f _ { \varphi } , { \mathrm { s o } }$ we define $ { \mathrm { A D D } } ( A , \varphi ) \stackrel { \cdot } { = }  { \mathrm { R M S E } } ( f _ { A } , \dot { f } _ { \varphi } )$

Observe that attributions computed through $\gamma = g \circ D$ concern the ICAM $f _ { A }$ , whose predictions may differ from those of the original model $f .$ . Our attribution model $f _ { \varphi }$ makes this distinction explicit: its agreement with $f _ { A }$ is measured by additivity error, while its agreement with $f$ is measured by attribution error. Applying the triangle inequality for RMSE to $f , f _ { A }$ , and $f _ { \varphi }$ gives

$$
\operatorname { A T E } ( A , \varphi ) \leq \operatorname { F E } ( A ) + \operatorname { A D D } ( A , \varphi ) .\tag{13}
$$

If $\mathrm { A D D } ( A , \varphi ) = 0$ , then $f _ { \varphi } = f _ { A }$ almost surely, and hence $\operatorname { A T E } ( A , \varphi ) = \operatorname { F E } ( A )$ Thus, even attributions that perfectly recover the ICAM output retain its disagreement with the original model. This lets us study additivity within the ICAM separately and use fidelity to connect the resulting guarantees to the original predictions.

In the following proposition, we provide a general case where $\mathrm { A D D } ( A , \varphi ) = 0$

Proposition 1 (Complete attributions for affine decoded heads). Suppose that $\gamma$ is affine, i.e., there are $a _ { i } \in \mathbb { R } ^ { m _ { i } }$ <sup>i</sup> such that $\begin{array} { r } { \gamma ( v _ { 1 } , \dots , v _ { k } ) = b + \sum _ { i = 1 } ^ { k } \langle a _ { i } , v _ { i } \rangle } \end{array}$ for all $v \in \mathcal V$ . Then the attribution $\varphi _ { i } ( x ) = \langle a _ { i } , \pi _ { i } ( h ( x ) ) \rangle$ has zero additivity error, i.e., $\mathrm { \bar { A D D } } ( A , \varphi ) = 0$

For a proof see Sec. B.2. Insertion, occlusion, and gradient-times-input all give precisely this attribution for an affine decoded head. We verify this in Sec. B.3. Thus, all three rules have zero additivity error, and their attribution error equals fidelity error.

This is an important case because many concept discovery methods use affine decoders. $\operatorname { I f } g$ is also affine, as when explaining a class logit through the final affine layer, then $\gamma = g \circ D$ is affine even when the encoder is nonlinear. Theorem 1 therefore gives

$$
\operatorname { A T E } ( A , \varphi ) = \operatorname { F E } ( A ) \leq L _ { g } \operatorname { R E } ( A )\tag{14}
$$

for all three rules. For a scalar head $g ( z ) = \langle w , z \rangle + \beta _ { : }$ , we may take $L _ { g } = \| w \| _ { 2 }$ , so this bound can be evaluated directly from the classifier weights and reconstruction error. Small reconstruction error together with a small Lipschitz constant therefore guarantees that the attributions approximately recover the original model output, and exact reconstruction makes them complete. This conclusion applies to the affine class scores; applying softmax generally makes the decoded head nonaffine.

Bounding the Nonaffinity of $\gamma$ When explaining an earlier layer, the remaining prediction head g may be nonlinear, and a nonlinear decoder can also make $\gamma = g \circ D$ nonaffine. The three attribution rules may then give different contributions with nonzero additivity error. To extend the affine case, we control how far $\gamma$ departs from an affine approximation by bounding the variation of its gradient.

Assume that $\gamma$ is also defined and differentiable on an open set $\Omega \subseteq \mathcal W$ containing the segments from zero to $\dot { C } ( \boldsymbol { x } )$ $P _ { i } C ( x )$ , and from $C ( x )$ to $C ( x ) \stackrel { \cdot } { - } P _ { i } C ( x )$ , for every x and i. Its values on $\Omega \cap \mathcal { V }$ agree with $g \circ D$ . Assume that $\mathbb { E } _ { x \sim p } \big [ \| C ( x ) \| _ { 2 } ^ { 4 } \big ] <$ < and that the gradient of γ is M-Lipschitz for a finite $M \geq { \bar { 0 } } !$

$$
\begin{array} { r } { \| \nabla \gamma ( v ) - \nabla \gamma ( w ) \| _ { 2 } \leq M \| v - w \| _ { 2 } \qquad \mathrm { f o r ~ a l l ~ } v , w \in \Omega . } \end{array}\tag{15}
$$

Theorem 3 (Curvature bounds additivity error). Under these assumptions, insertion, occlusion, and gradient-times-input satisfy

$$
\mathrm { A D D } ( A , \varphi ) \leq M { \sqrt { \mathbb { E } _ { x \sim p } [ \| C ( x ) \| _ { 2 } ^ { 4 } ] } } .\tag{16}
$$

A proof is in Sec. B.4.

## 4.4 EMPIRICAL TIGHTNESS

To assess how conservative these bounds are, we evaluate them on a ResNet-50 trained on ImageNet 1k, split after its last residual stage and before its final affine layer, where the prediction head is affine and $L _ { g }$ is available in closed form. We fit concept autoencoders with PCA, NMF, k-means, a sparse autoencoder, and an autoencoder with a nonlinear decoder, for $k \in \{ 5 , 2 5 , 5 0 \}$ concepts and three seeds; details and all results are in Sec. C. All bounds hold in every configuration, but their tightness differs considerably.

The fidelity bound of Theorem 1 is conservative: the fidelity error is about a quarter of $L _ { g } \operatorname { R E } ( A )$ before the final layer and about a tenth after the last residual stage, almost independently of the method and of $k .$ . The slack has two sources. First, reconstruction errors lie mostly in directions to which the prediction head is insensitive, whereas $L _ { g }$ accounts for its most sensitive direction. Second, after the last residual stage, the errors at different spatial locations partly cancel when the head averages over them.

For model completeness, heads fitted to the concept representations reduce the error of the decoded head $g \circ D$ by at most about $9 \% .$ so $g \circ D$ is already close to the best head we could find.

For attributions, the tightness depends on the decoder. With the four affine decoders, insertion, occlusion, and gradient-times-input are additive up to floating-point error, so the attribution error equals the fidelity error and the attribution bound is attained, as Proposition 1 predicts. With the nonlinear decoder, the attribution bound, which adds the curvature term of Theorem 3 to the fidelity error, still holds but exceeds the attribution error by two to three orders of magnitude.

## 5 RELATED WORK

Concepts from Dictionary Learning Fel et al. (2023a) unify k-means, PCA, NMF, and sparse autoencoders through dictionary learning, abstracting methods such as ACE (Ghorbani et al., 2019), ICE (Zhang et al., 2021), and CRAFT (Fel et al., 2023b). The same encoder–decoder formulation also accommodates FACE (Bhusal et al., 2025). Their coefficient inference is our encoder E, their dictionary defines $D ( v ) = B v ,$ and their reconstructed predictor is our ICAM. Allowing a decode bias also accommodates Anthropic’s sparse autoencoders (Bricken et al., 2023; Templeton et al., 2024). Our guarantees apply to these fitted constructions under the stated assumptions, independently of the learning objective or whether optimization reaches a global optimum. For affine $^ { g , }$ as in the holistic framework’s penultimate-layer analysis, $\gamma = g \circ D$ is affine, so our results apply to insertion and to the occlusion and gradient-times-input rules they consider. Our analysis complement their optimality guarantees for importance rankings and FACE’s bounds on predictive disagreement by connecting reconstruction to model completeness and distinguishing attribution completeness from fidelity. The formulation also permits nonlinear decoders, with approximate guarantees under the corresponding Lipschitz and curvature assumptions.

Multidimensional Concept Discovery MCD (Vielhaben et al., 2023) represents concepts by subspaces of feature channels. Its component projections are our probes, and its decoder sums the components: $\begin{array} { r } { D ( v _ { 1 } , . . . , v _ { k } ) = \sum _ { i } v _ { i } } \end{array}$ . For the complete basis decomposition, including the residual, $D \circ E = \operatorname { i d } _ { \mathcal { Z } }$ . Together with its affine prediction head and blockwise relevance scores, this makes MCD’s attribution-completeness relation a special case of exact reconstruction and Proposition 1. Our formulation identifies the structural conditions behind this result and additionally establishes model completeness of the full decomposition for any prediction head. Our approximate bounds address discarded components and nonlinear heads under the corresponding regularity assumptions. These distribution-dependent errors complement MCD’s geometric completeness score based on classifier weights. Our product representation spaces also accommodate HU-MCD (Grobrugge¨ et al., 2025) and the vector-valued concept blocks of subspace-aware sparse autoencoders (Dalili & Mahdavi, 2026), without requiring the same discovery procedure or exact reconstruction.

Completeness and Evaluation Yeh et al. (2020) measure completeness through normalized classification accuracy against ground-truth labels, optimizing a decoder for thresholded, normalized concept projections while retaining the original prediction head. Our model completeness error instead measures recovery of the original outputs $f ( x )$ over all measurable heads on $C ( x )$ , separating information retained by the encoder from the performance of a particular decoder. Their ConceptSHAP distributes a global completeness score among concepts, whereas our attribution error concerns recovery of individual model outputs. Other evaluations address semantic readability and perturbation-based faithfulness (Li et al., 2024), or whether explanations enable an LLM to simulate model predictions, as in ConSim (Poche et al.´ , 2025). SAEBench further shows that improvements in sparse-autoencoder proxy metrics need not improve practical interpretability evaluations (Karvonen et al., 2025). Our contribution is a common formulation and conditional guarantees for existing constructions; we do not introduce a discovery algorithm, and our guarantees do not establish se mantic readability or successful simulation.

## 6 CONCLUSION

We introduced a common language for concept-based prediction, detection, and discovery, distinguishing input-level concepts, potentially multidimensional representations, and scalar activations. For concept autoencoders, exact reconstruction gives zero fidelity and model completeness errors, while a Lipschitz prediction head provides quantitative guarantees for approximate reconstruction. We also bound attribution error by fidelity error plus additivity error within the ICAM. Insertion, occlusion, and gradient-times-input have zero additivity error for affine decoded heads, and our curvature bound extends the analysis to smooth nonaffine heads.

These guarantees apply to existing methods through their encoder–decoder structure, independently of their training objectives. They concern output recovery and attribution completeness; the semantic meaning of concepts requires separate evaluation.

## AI USE STATEMENT

We used generative AI tools to assist with developing and refining the conceptual framework, mathematical statements, and proofs. We used a generative AI coding assistant to help implement, debug, and optimize the evaluation code. We also used generative AI to support literature exploration, assist with analyzing experimental results, prepare figures, and draft and revise the structure, language, and formatting of the paper.

The authors reviewed the AI-assisted mathematical arguments, figures, code, and text. The code was tested on small configurations, and its outputs were checked against theoretical predictions. AIassisted text was read and revised by the authors, and every reported number was checked against the logged results. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

All assumptions of our theoretical results are stated explicitly in Sec. 4 and the proofs are either in the main text or in Sec. B. For the empirical evaluation, Sec. C describes the model, data, boundaries, concept discovery methods, prediction heads, and the estimation of M, and reports all results as means over three seeds together with their standard deviations. The evaluation uses only public resources: the torchvision ResNet-50 weights (IMAGENET1K V2) and the ImageNet-1k validation set. We provide the source code as supplementary material; each run is fully specified by a configuration file and a seed, which determines the sampled images, the split into fitting and evaluation halves, and the initialization of all methods and heads.

## ACKNOWLEDGMENTS

The work was supported by the Czech Science Foundation (GACR) grant no. 26-23981S. Computa- <sup>ˇ</sup> tional resources were provided by the e-INFRA CZ project (ID: 90140), supported by the Ministry of Education, Youth, and Sports of the Czech Republic.

## REFERENCES

Reduan Achtibat, Maximilian Dreyer, Ilona Eisenbraun, Sebastian Bosse, Thomas Wiegand, Wojciech Samek, and Sebastian Lapuschkin. From attribution maps to human-understandable explanations through Concept Relevance Propagation. Nature Machine Intelligence, 5(9):1006– 1019, September 2023. ISSN 2522-5839. doi: 10.1038/s42256-023-00711-8. URL https: //www.nature.com/articles/s42256-023-00711-8. Publisher: Nature Publishing Group.

David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. URL https://arxiv.org/abs/ 1704.05796.

Dipkamal Bhusal, Michael Clifford, Sara Rampazzi, and Nidhi Rastogi. FACE: Faithful automatic concept extraction. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/3e59c3d09840b4e40a21a1f2ea771f89-Abstract-Conference.html.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.

Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: Deep learning for interpretable image recognition. In Advances in Neural Informa-

tion Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/ paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html.

Jonathan Crabbe and Mihaela Van Der Schaar. Concept activation regions: A generalized framework´ for concept-based explanations. In Advances in Neural Information Processing Systems 35, pp. 2590–2607. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. ISBN 978-1-7138-7108-8. doi: 10.52202/068431-0188. URL http://www.proceedings.com/ 068431-0188.html.

Seyed Arshan Dalili and Mehrdad Mahdavi. Subspace-aware sparse autoencoders for effective mechanistic interpretability, 2026. URL https://arxiv.org/abs/2606.06333. Preprint.

Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, Pietro Lio, and Mateja Jamnik. Concept embedding models: Beyond the accuracy-explainability\` trade-off. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv.org/abs/2209.09056.

Thomas Fel, Victor Boutin, Louis Bethune, Remi Cadene, Mazda Moayeri, L´ eo And´ eol, Math-´ ieu Chalvidal, and Thomas Serre. A holistic approach to unifying automatic concept extraction and concept importance estimation. In Advances in Neural Information Processing Systems, volume 36, 2023a. URL https://papers.neurips.cc/paper\_files/paper/2023/ hash/abf3682c9cf9245a0294a4bebe4544ff-Abstract-Conference.html.

Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, R´ emi´ Cadene, and Thomas Serre. CRAFT: Concept recursive activation FacTorization for explainabil-\` ity. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023b. URL https://arxiv.org/abs/2211.10154.

Ruth Fong and Andrea Vedaldi. Net2Vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. URL https://arxiv.org/abs/1801.03454.

Amirata Ghorbani, James Wexler, James Y Zou, and Been Kim. Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/ 77d2afcb31f6493e350fca61764efb9a-Abstract.html.

Arne Grobrugge, Niklas K ¨ uhl, Gerhard Satzger, and Philipp Spitzer. Towards human-¨ understandable multi-dimensional concept discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. URL https://arxiv.org/abs/ 2503.18629.

Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Isaac Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum Stuart Mcdougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. SAEBench: A comprehensive benchmark for sparse autoencoders in language model interpretability. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 29223–29264. PMLR, 2025. URL https://proceedings.mlr.press v267/karvonen25a.html.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory´ Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2668–2677. PMLR, 2018. URL https://proceedings.mlr.press/v80/kim18d.html.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5338–5348. PMLR, 2020. URL https://proceedings.mlr.press/v119/koh20a.html.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436– 444, 2015. doi: 10.1038/nature14539. URL https://www.nature.com/articles/ nature14539.

Meng Li, Haoran Jin, Ruixuan Huang, Zhihao Xu, Defu Lian, Zijia Lin, Di Zhang, and Xiting Wang. Evaluating readability and faithfulness of concept-based explanations. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 607–625, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.36. URL https://aclanthology.org/2024.emnlp-main.36/.

Meike Nauta, Ron van Bree, and Christin Seifert. Neural prototype trees for interpretable finegrained image recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. URL https://arxiv.org/abs/2012.02046.

Meike Nauta, Jorg Schl¨ otterer, Maurice van Keulen, and Christin Seifert. PIP-Net: Patch-¨ based intuitive prototypes for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2744–2753, 2023. URL https://openaccess.thecvf.com/content/CVPR2023/html/Nauta\_ PIP-Net\_Patch-Based\_Intuitive\_Prototypes\_for\_Interpretable\_ Image\_Classification\_CVPR\_2023\_paper.html.

Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. In International Conference on Learning Representations, 2023. URL https: //arxiv.org/abs/2304.06129.

Antonin Poche, Alon Jacovi, Agustin Martin Picard, Victor Boutin, and Fanny Jourdan. Con-´ Sim: Measuring concept-based explanations’ effectiveness with automated simulatability. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5594–5615, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.279. URL https://aclanthology.org/2025.acl-long.279/.

Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor, Tania Cerquitelli, and Elena Baralis. Conceptbased explainable artificial intelligence: A survey. ACM Comput. Surv., November 2025. ISSN 0360-0300. doi: 10.1145/3774643. URL https://doi.org/10.1145/3774643. Just Accepted.

Wojciech Samek, Thomas Wiegand, and Klaus-Robert Muller. Explainable artificial intelli-¨ gence: Understanding, visualizing and interpreting deep learning models. arXiv preprint arXiv:1708.08296, 2017. URL https://arxiv.org/abs/1708.08296.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 3319–3328. PMLR, 2017. URL https://proceedings. mlr.press/v70/sundararajan17a.html.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html.

Johanna Vielhaben, Stefan Blucher, and Nils Strodthoff. Multi-dimensional concept discovery¨ (MCD): A unifying framework with completeness guarantees. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id= KxBQPz7HKh.

Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19187–19197, 2023. URL https://arxiv.org/abs/2211.11158.

Chih-Kuan Yeh, Been Kim, Sercan O. Arık, Chun-Liang Li, Tomas Pfister, and Pradeep<sup>¨</sup> Ravikumar. On completeness-aware concept-based explanations in deep neural networks. In Advances in Neural Information Processing Systems, volume 33, pp. 20554– 20565, 2020. URL https://papers.nips.cc/paper\_files/paper/2020/file/ ecb287ff763c169694f682af52c1f309-Paper.pdf.

Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=nA5AZ8CEyow.

Ruihan Zhang, Prashan Madumal, Tim Miller, Krista A. Ehinger, and Benjamin I. P. Rubinstein. Invertible concept-based explanations for CNN models with non-negative concept activation vectors. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11682– 11690, 2021. doi: 10.1609/aaai.v35i13.17389. URL https://ojs.aaai.org/index. php/AAAI/article/view/17389.

## A INSTANTIATIONS OF CONCEPT-BASED METHODS

Tab. 1 summarizes how the methods discussed in Sec. 3 and 5 fit our framework. For explainableby-design models, the latent space is the representation space, $\mathcal { Z } = \nu .$ , and the probes $\pi _ { i } ( z ) = z _ { i }$ form the identity encoder. The trivial concept autoencoder $E = D =$ id then gives $f _ { A } = f$ and $\gamma = g$ , so $\operatorname { M C E } ( A ) = \operatorname { F E } ( A ) = 0$ , and attribution completeness depends only on whether $g$ is affine. Concept detection methods learn probes for individual user-specified concepts without a decoder, so the guarantees of Sec. 4 do not apply to them directly. All concept discovery methods in the table use affine decoders, so γ is affine whenever g is.

Table 1: Instantiations of concept-based methods in our framework: CBMs (Koh et al., 2020), PCBM (Yuksekgonul et al., 2023), LF-CBM (Oikarinen et al., 2023), CEM (Espinosa Zarlenga et al., 2022), TCAV (Kim et al., 2018), CAR (Crabbe & Schaar´ , 2022), ACE (Ghorbani et al., 2019), CRAFT (Fel et al., 2023b), ICE (Zhang et al., 2021), SAE (Bricken et al., 2023), and MCD (Vielhaben et al., 2023). Columns give the latent space , the concept representation space , the probe $\pi _ { i } ,$ , the activation function $\sigma _ { i } .$ , the decoder D, and whether the decoded head $\gamma = g \circ D$ is affine; “if $g ^ { , , }$ means that $\gamma$ is affine whenever g is. A dash marks a component the method does not specify. s is the logistic sigmoid, U is a dictionary with concept directions $u _ { i }$ in its rows, NMF encoders compute nonnegative least-squares coefficients with U fixed, and $Q _ { i }$ projects channels onto the i-th MCD subspace, with the residual subspace included as a concept.
<table><tr><td>Method</td><td>Z</td><td>Vi</td><td> $\pi _ { i } ( z )$ </td><td> $\sigma _ { i }$ </td><td>D</td><td>γ affine</td></tr><tr><td colspan="7">Explainable by design</td></tr><tr><td>CBM, probabilities</td><td> $[ 0 , 1 ] ^ { k }$ </td><td>[0, 1]</td><td>Zi</td><td>id</td><td>id</td><td>if g</td></tr><tr><td>CBM, logits</td><td> $\dot { \mathbb { R } } ^ { k }$ </td><td>R</td><td>Zi</td><td>S</td><td>id</td><td> $\operatorname { i f } g$ </td></tr><tr><td>PCBM, LF-CBM</td><td> $\mathbb { R } ^ { k }$ </td><td>R</td><td>Zi</td><td>一</td><td>id</td><td>yes</td></tr><tr><td>CEM</td><td> $\Pi _ { i } \mathbb { R } ^ { 2 m }$ </td><td> $\mathbb { R } ^ { 2 m }$ </td><td>Zi</td><td> $s ( \langle w , v \rangle + \beta )$ </td><td>id</td><td>no</td></tr><tr><td colspan="7">Concept detection</td></tr><tr><td>TCAV</td><td> $\mathbb { R } ^ { d }$ </td><td>R</td><td> $\langle w , z \rangle + \beta$ </td><td> $\mathbf { 1 } \{ t > 0 \}$ </td><td></td><td></td></tr><tr><td>CAR</td><td> $\mathbb { R } ^ { d }$ </td><td>R</td><td>kernel SVM</td><td> $\mathbf { 1 } \{ t > 0 \}$ </td><td></td><td></td></tr><tr><td colspan="7">Concept discovery</td></tr><tr><td>ACE</td><td> $\mathbb { R } ^ { d }$ </td><td>{0, 1}</td><td> $\begin{array} { r } { \mathbf { 1 } \{ i = \arg \operatorname* { m i n } _ { j } \left\| z - u _ { j } \right\| \} } \end{array}$ </td><td></td><td> $U ^ { \top } v$ </td><td>if g</td></tr><tr><td>CRAFT</td><td> $\mathbb { R } _ { > 0 } ^ { d }$ </td><td> $\mathbb { R } _ { \geq 0 }$ </td><td> $E ( z ) _ { i } ~ ( \mathrm { N M F } )$ </td><td></td><td> $U ^ { \top } v$ </td><td>if g</td></tr><tr><td>ICE</td><td> $\mathbb { R } _ { > 0 } ^ { \boldsymbol { \bar { S } } \times d }$ </td><td> $\mathbb { R } _ { \geq 0 } ^ { S }$ </td><td> $E ( z ) _ { i , : } \left( \mathrm { N M F } \right)$ </td><td></td><td> $V ^ { \top } U$ </td><td>yes</td></tr><tr><td>SAE</td><td> $\mathbb { R } ^ { d }$ </td><td> $\mathbb { R } _ { \geq 0 }$ </td><td> $\mathrm { R e L U } ( \langle w _ { i } , z \rangle + \beta _ { i } )$ </td><td></td><td> $U ^ { \top } v + \beta _ { D }$ </td><td>if g</td></tr><tr><td>MCD</td><td> $\mathbb { R } ^ { S \times d }$ </td><td> $\mathbb { R } ^ { \overline { { S } } \times d }$ </td><td> $z Q _ { i } ^ { \top }$ </td><td></td><td> $\textstyle \sum _ { i } v _ { i }$ </td><td>if g</td></tr></table>

For CEM, we take as the representation of concept i the pair of positive and negative embeddings $( \hat { c } _ { i } ^ { + } , \hat { c } _ { i } ^ { - } ) \in \mathbb { R } ^ { 2 m }$ , the shared scoring function as $\sigma _ { i }$ , and include the probability-weighted mixing of the two embeddings in $^ { g ; }$ this mixing makes $g$ nonaffine. Following Fel et al. (2023a), we read ACE’s clustering of segment activations as k-means dictionary learning with one-hot coefficients.

ICE Fig. 6 summarizes the ICE example of Sec. 3.2.2. Since $D$ and $g$ are both affine, so is the decoded head,

$$
\gamma ( V ) = g ( V ^ { \top } U ) = b + \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \sum _ { i = 1 } ^ { k } V _ { i , s } \langle u _ { i } , w \rangle = b + \sum _ { i = 1 } ^ { k } ( U w ) _ { i } \bar { v } _ { i } , \qquad \bar { v } _ { i } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } V _ { i , s } ,\tag{17}
$$

where $u _ { i }$ is the i-th row of $U .$ . This is the affine form of Proposition 1 with $a _ { i } = S ^ { - 1 } ( U w ) _ { i } { \bf 1 } _ { S }$ . Insertion, occlusion, and gradient-times-input therefore all return ICE’s contributions $\varphi _ { i } ( x ) = ( U w ) _ { i } { \bar { v } } _ { i }$ which sum with b to $f _ { A } ( x )$ , so $\operatorname { A T E } ( A , \varphi ) = \operatorname { F E } ( A )$ . The head g is Lipschitz with $L _ { g } = \| w \| _ { 2 } / \sqrt { S }$ and Theorem 1 gives $\mathrm { F E } ( A ) \leq \| w \| _ { 2 } \mathrm { R E } ( A ) / \sqrt { S } .$

$$
\begin{array} { r l } { \mathcal { X } \xrightarrow { h } } & { \mathbb { R } _ { \geq 0 } ^ { S \times d } \xrightarrow { g } \mathbb { R } } \\ & { \xrightarrow { E } \Bigg \downarrow \Bigg \uparrow D } \\ { \mathbb { R } _ { \geq 0 } ^ { S }  \pi _ { i } } & { \mathbb { R } _ { \geq 0 } ^ { k \times S } \xrightarrow [ \Phi ] { g } \mathbb { R } ^ { k } } \end{array} \Bigg \downarrow { \overset { { } } { } }
$$

Figure 6: ICE (Zhang et al., 2021) as a concept autoencoder. The encoder E computes nonnegative coefficients $V = E ( z )$ for the fixed dictionary $U \in \mathbb { R } _ { > 0 } ^ { k \times d }$ , and the decoder is $D ( V ) = V ^ { \top } U$ . The probe $\pi _ { i }$ returns the spatial coefficient map of concept i, which ICE uses to highlight image regions. The prediction head is the class logit $\begin{array} { r } { g ( z ) = b + \frac { 1 } { S } \sum _ { s } \langle w , z _ { s , : } \rangle } \end{array}$ , and $\Phi ( V ) = \left( \left( U w \right) _ { i } \bar { v } _ { i } \right) _ { i = 1 } ^ { k }$ returns the concept contributions. The right square commutes, $g \circ D = b + \textstyle \sum _ { i } \Phi _ { i }$ , so the contributions sum to $f _ { A } ( x )$ exactly; since $D \circ E$ only approximates the identity, they recover $f ( x )$ up to $\operatorname { F E } ( A )$

## B PROOFS FOR SEC. 4

We use the notation and assumptions of Sec. 4.

## B.1 PROOF OF THEOREM 1

Proof. For every $x \in \mathcal { X }$ , Lipschitz continuity gives

$$
\| f ( x ) - f _ { A } ( x ) \| _ { 2 } = \| g ( h ( x ) ) - g ( D ( C ( x ) ) ) \| _ { 2 } \leq L _ { g } \| h ( x ) - D ( C ( x ) ) \| _ { 2 } .\tag{18}
$$

Squaring and taking expectations preserves the inequality by monotonicity of expectation. Linearity of expectation allows $\dot { L _ { g } } ^ { 2 }$ to be taken outside, and taking square roots gives the result.

## B.2 PROOF OF PROPOSITION 1

Proof. For every $x \in \mathcal X$ , the definitions of $f _ { \varphi }$ and $C$ give

$$
f _ { \varphi } ( x ) = b + \sum _ { i = 1 } ^ { k } \varphi _ { i } ( x ) = b + \sum _ { i = 1 } ^ { k } \langle a _ { i } , \pi _ { i } ( h ( x ) ) \rangle = \gamma ( C ( x ) ) = f _ { A } ( x ) .
$$

Thus $f _ { \varphi } = f _ { A }$ , and consequently $\mathrm { A D D } ( A , \varphi ) = \mathrm { R M S E } ( f _ { A } , f _ { \varphi } ) = 0$

## B.3 ATTRIBUTION RULES FOR AFFINE DECODED HEADS

Suppose that $\gamma$ has the affine form in Proposition 1. For insertion, only block i is retained, so

$$
\gamma ( P _ { i } C ( x ) ) - b = \langle a _ { i } , \pi _ { i } ( h ( x ) ) \rangle .
$$

For occlusion, subtracting the prediction with block i removed cancels the baseline and all other block contributions:

$$
\gamma ( C ( x ) ) - \gamma ( C ( x ) - P _ { i } C ( x ) ) = \langle a _ { i } , \pi _ { i } ( h ( x ) ) \rangle .
$$

For gradient-times-input, we use the affine formula on , whose gradient is the constant vector $( a _ { 1 } , \ldots , a _ { k } )$ . Consequently,

$$
\langle \nabla \gamma ( C ( x ) ) , P _ { i } C ( x ) \rangle = \langle a _ { i } , \pi _ { i } ( h ( x ) ) \rangle .
$$

All three rules therefore satisfy the attribution formula of Proposition 1.

## B.4 PROOF OF THEOREM 3

Proof. Choose $\boldsymbol { a } ( \boldsymbol { x } ) = \nabla \gamma ( 0 \nu )$ for insertion and $\boldsymbol { a } ( \boldsymbol { x } ) = \nabla \gamma ( C ( \boldsymbol { x } ) )$ for occlusion and gradient times-input. For each input x, define the affine approximation

$$
\gamma _ { \mathrm { a f f } , x } ( v ) = b + \langle a ( x ) , v \rangle , \qquad v \in \mathcal { W } .\tag{19}
$$

The bias is fixed at $b ,$ so every approximation agrees with $\gamma$ at zero. Define $f _ { \mathrm { a f f } } ( x ) = \gamma _ { \mathrm { a f f } , x } ( C ( x ) )$ The triangle inequality for RMSE gives

$$
\mathrm { A D D } ( A , \varphi ) \leq \mathrm { R M S E } ( f _ { A } , f _ { \mathrm { a f f } } ) + \mathrm { R M S E } ( f _ { \mathrm { a f f } } , f _ { \varphi } ) .\tag{20}
$$

The standard Taylor bound for a function with an M-Lipschitz gradient gives

$$
| \gamma ( v ) - \gamma ( w ) - \langle \nabla \gamma ( w ) , v - w \rangle | \leq \frac { M } { 2 } \Vert v - w \Vert _ { 2 } ^ { 2 }\tag{21}
$$

whenever the segment from w to v lies in Ω. We use this bound to control both terms in Eq. (20).

For the first term, apply Eq. (21) from zero to $C ( x )$ for insertion, and from $C ( x )$ to zero for the other two rules. Using $b = \gamma ( 0 \nu )$ and the chosen $a ( x )$ , both cases give

$$
| f _ { A } ( x ) - f _ { \mathrm { a f f } } ( x ) | = | \gamma ( C ( x ) ) - b - \langle a ( x ) , C ( x ) \rangle | \leq { \frac { M } { 2 } } \| C ( x ) \| _ { 2 } ^ { 2 } .\tag{22}
$$

Squaring and taking expectations preserves the inequality by monotonicity of expectation. By linearity of expectation, the constant $\dot { M } ^ { 2 } / 4$ can be taken outside, and taking square roots yields

$$
\mathrm { R M S E } ( f _ { A } , f _ { \mathrm { a f f } } ) \leq \frac { M } { 2 } \sqrt { \mathbb { E } _ { x \sim p } \big [ \| C ( x ) \| _ { 2 } ^ { 4 } \big ] } .\tag{23}
$$

For the second term, fix $x \in \mathcal { X }$ . The isolated blocks satisfy

$$
C ( x ) = \sum _ { i = 1 } ^ { k } { P _ { i } C ( x ) } , \qquad \| C ( x ) \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { k } { \| P _ { i } C ( x ) \| _ { 2 } ^ { 2 } } .\tag{24}
$$

By linearity of the inner product, the affine prediction therefore decomposes as

$$
f _ { \mathrm { a f f } } ( x ) = b + \sum _ { i = 1 } ^ { k } \langle a ( x ) , P _ { i } C ( x ) \rangle .\tag{25}
$$

We compare each attribution $\varphi _ { i } ( x )$ with its corresponding affine contribution $\langle a ( x ) , P _ { i } C ( x ) \rangle$

For insertion, applying Eq. (21) from zero to $P _ { i } C ( x )$ gives

$$
| \gamma ( P _ { i } C ( x ) ) - b - \langle \nabla \gamma ( 0 _ { \mathcal { V } } ) , P _ { i } C ( x ) \rangle | \leq \frac { M } { 2 } \| P _ { i } C ( x ) \| _ { 2 } ^ { 2 } .
$$

For occlusion, applying the same bound from $C ( x )$ to $C ( x ) - P _ { i } C ( x )$ and rearranging gives

$$
\begin{array} { r l } & { | \gamma ( C ( x ) ) - \gamma ( C ( x ) - P _ { i } C ( x ) ) - \langle \nabla \gamma ( C ( x ) ) , P _ { i } C ( x ) \rangle | } \\ & { \qquad \leq \displaystyle \frac { M } { 2 } \| P _ { i } C ( x ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

For gradient-times-input, the attribution equals $\left. a ( x ) , P _ { i } C ( x ) \right.$ by definition. Thus, for all three rules,

$$
| \varphi _ { i } ( x ) - \langle a ( x ) , P _ { i } C ( x ) \rangle | \leq \frac { M } { 2 } \| P _ { i } C ( x ) \| _ { 2 } ^ { 2 } .\tag{26}
$$

Using the common baseline b and summing over concepts gives

$$
\begin{array} { r l } & { | f _ { \mathrm { a f f } } ( x ) - f _ { \varphi } ( x ) | = \displaystyle \left| \sum _ { i = 1 } ^ { k } ( \langle a ( x ) , P _ { i } C ( x ) \rangle - \varphi _ { i } ( x ) ) \right| } \\ & { \qquad \leq \displaystyle \sum _ { i = 1 } ^ { k } \langle a ( x ) , P _ { i } C ( x ) \rangle - \varphi _ { i } ( x ) | } \\ & { \qquad \leq \displaystyle \frac { M } { 2 } \sum _ { i = 1 } ^ { k } \| P _ { i } C ( x ) \| _ { 2 } ^ { 2 } = \frac { M } { 2 } \| C ( x ) \| _ { 2 } ^ { 2 } . } \end{array}
$$

The inequalities use the triangle inequality and Eq. (26), and the last equality uses Eq. (24). Taking the root mean square as above gives

$$
\mathrm { R M S E } ( f _ { \mathrm { a f f } } , f _ { \varphi } ) \leq \frac { M } { 2 } \sqrt { \mathbb { E } _ { x \sim p } \big [ \| C ( x ) \| _ { 2 } ^ { 4 } \big ] } .\tag{27}
$$

Substituting Eqs. (23) and (27) into Eq. (20) proves

$$
\mathrm { A D D } ( A , \varphi ) \leq M { \sqrt { \mathbb { E } _ { x \sim p } [ \| C ( x ) \| _ { 2 } ^ { 4 } ] } } .
$$

## C EVALUATION OF TIGHTNESS OF THE BOUNDS

We evaluate how closely the bounds of Sec. 4 are attained by concept autoencoders obtained with common concept discovery methods.

Setup We use a ResNet-50 trained on ImageNet-1k and, for each of three random seeds, 25,000 images drawn from the ImageNet-1k validation set, which define the empirical distribution p. Half of the images are used to fit the concept autoencoders and prediction heads, and all reported errors are computed on the other half. We split the model at two boundaries: after the last residual stage (layer4, $\mathcal { Z } = \mathbb { R } ^ { S \times d }$ with $S = 4 9$ and $\bar { d } = 2 0 4 8 )$ and before the final affine layer (penultimate, $\mathcal { Z } = \mathbb { R } ^ { d } )$ . At both boundaries the prediction head is affine, $g ( z ) = W \bar { z } + \beta .$ , where $\bar { z } = S ^ { - 1 } \sum _ { s = 1 } ^ { S } z _ { s }$ is the spatial average at layer4 and $\bar { z } = z$ at the penultimate boundary. Consequently, the Lipschitz constant of Theorem 1 is available in closed form: $L _ { g } = \| W \| _ { 2 }$ (the largest singular value of $W )$ at the penultimate boundary, and $L _ { g } = \| W \| _ { 2 } / \sqrt { S }$ at layer4, since averaging over S locations shrinks the Euclidean norm on $\mathbb { R } ^ { S \times d }$ by at least a factor ${ \sqrt { S } } ,$ , with equality for errors that are identical at every location. We extract $k \in \{ 5 , 2 5 , 5 0 \}$ concepts with five methods that cover different decoder types: PCA (orthogonal linear decoder), NMF (nonnegative linear decoder), k-means (centroid decoder with soft assignments as concept representations), a sparse autoencoder (affine decoder), and an autoencoder with a nonlinear decoder. Methods acting on spatial latents are applied to each spatial feature vector separately, so each concept has one coefficient per location. PCA, NMF, and k-means use their scikit-learn implementations. Both autoencoders share the encoder $\mathbb { R } ^ { d }  \mathbb { R } ^ { 4 k }  \mathbb { R } ^ { k }$ with a ReLU after each layer; the SAE has the affine decoder $\mathbb { R } ^ { k } \to \mathbb { R } ^ { d }$ and an $\ell _ { 1 }$ penalty on the concept representation, and the nonlinear autoencoder has the decoder $\mathbb { R } ^ { k }  \mathbb { R } ^ { 4 k } \stackrel { - } {  } \mathbb { R } ^ { d }$ with a ReLU after the hidden layer.

Tightness Ratios Since the errors differ by orders of magnitude across boundaries and methods, we report each bound through the ratio of its left-hand side to its right-hand side,

$$
\rho _ { \mathrm { F E } } = \frac { \mathrm { F E } ( A ) } { L _ { g } \mathrm { R E } ( A ) } , \qquad \rho _ { \mathrm { A T E } } = \frac { \mathrm { A T E } ( A , \varphi ) } { \mathrm { F E } ( A ) + M \sqrt { \mathbb { E } _ { x \sim p } \left[ \Vert C ( x ) \Vert _ { 2 } ^ { 4 } \right] } } .\tag{28}
$$

A ratio of one means that the bound is attained, and smaller ratios indicate slack. The corresponding ratio $\mathrm { M C E } ( A ) / \mathrm { F E } ( A )$ for model completeness cannot be computed, since MCE is an infimum over all heads, and we report the upper estimate $\hat { \rho } _ { \mathrm { M C E } }$ described below. Tab. 2 reports the fidelity and model completeness ratios for every boundary, method, and k, and Tab. 3 reports the attribution ratios for the nonlinear autoencoder, the only method for which they differ from one.

Table 2: Tightness of the fidelity and model completeness bounds, mean over three seeds; the standard deviation over seeds is at most 0.012 for ρˆ and at most 0.002 for the other ratios. $\rho _ { \mathrm { F E } } = \alpha \kappa$ is the tightness of Theorem 1, with alignment α and spatial coherence κ from Eq. (30). ρˆ<sub>MCE</sub> upperbounds the tightness of Theorem 2.
<table><tr><td>Boundary</td><td>Method</td><td>k</td><td>ρFE</td><td>α</td><td>κ</td><td>MCE</td></tr><tr><td rowspan="18">layer4</td><td rowspan="2">PCA</td><td>5</td><td>0.106</td><td>0.270</td><td>0.394</td><td>0.967</td></tr><tr><td>25</td><td>0.103</td><td>0.266</td><td>0.386</td><td>0.927</td></tr><tr><td rowspan="2"></td><td>50</td><td>0.099</td><td>0.261</td><td>0.381</td><td>0.939</td></tr><tr><td>5</td><td>0.106</td><td>0.269</td><td>0.395</td><td>0.975</td></tr><tr><td rowspan="2">NMF</td><td>25</td><td>0.102</td><td>0.264</td><td>0.389</td><td>0.960</td></tr><tr><td>50</td><td>0.101</td><td>0.263</td><td>0.385</td><td>0.964</td></tr><tr><td rowspan="3">k-means</td><td>5</td><td>0.105</td><td>0.267</td><td>0.394</td><td>0.993</td></tr><tr><td>25</td><td>0.103</td><td>0.266</td><td>0.388</td><td>0.983</td></tr><tr><td>50</td><td>0.102</td><td>0.264</td><td>0.385</td><td>0.978</td></tr><tr><td rowspan="3">SAE</td><td>5</td><td>0.106</td><td>0.270</td><td>0.392</td><td>0.968</td></tr><tr><td>25</td><td>0.102 0.099</td><td>0.265</td><td>0.386</td><td>0.934</td></tr><tr><td>50</td><td></td><td>0.261</td><td>0.381</td><td>0.943</td></tr><tr><td rowspan="3">Nonlinear AE</td><td>5</td><td>0.104</td><td>0.267</td><td>0.390</td><td>0.985</td></tr><tr><td>25 50</td><td>0.096 0.092</td><td>0.256 0.250</td><td>0.375</td><td>0.979</td></tr><tr><td></td><td></td><td></td><td>0.367</td><td>0.971</td></tr><tr><td rowspan="9">penultimate</td><td rowspan="3">PCA</td><td>5</td><td>0.270</td><td>0.270</td><td>1</td><td>0.956</td></tr><tr><td>25</td><td>0.265</td><td>0.265</td><td>1</td><td>0.908</td></tr><tr><td>50</td><td>0.261</td><td>0.261</td><td>1</td><td>0.916</td></tr><tr><td rowspan="3">NMF</td><td>5</td><td>0.270</td><td>0.270</td><td>1</td><td>0.969</td></tr><tr><td>25</td><td>0.264</td><td>0.264</td><td>1</td><td>0.944</td></tr><tr><td>50</td><td>0.261</td><td>0.261</td><td>1</td><td>0.949</td></tr><tr><td></td><td>5 0.267</td><td>0.267</td><td>1</td><td>1.000</td></tr><tr><td rowspan="3">k-means</td><td>25</td><td>0.264</td><td>0.264</td><td>1</td><td>1.000</td></tr><tr><td>50</td><td>0.262</td><td>0.262</td><td>1</td><td>1.000</td></tr><tr><td>5</td><td>0.269</td><td>0.269</td><td>1</td><td>0.978</td></tr><tr><td rowspan="3">SAE</td><td>25</td><td>0.265</td><td>0.265</td><td>1</td><td>0.916</td></tr><tr><td>50</td><td>0.264</td><td>0.264</td><td>1</td><td>0.916</td></tr><tr><td>5</td><td>0.268</td><td>0.268</td><td>1</td><td>0.975</td></tr><tr><td rowspan="3">Nonlinear AE</td><td>25</td><td>0.257</td><td>0.257</td><td>1</td><td>0.969</td></tr><tr><td></td><td>0.250</td><td>0.250</td><td></td><td></td></tr><tr><td>50</td><td></td><td></td><td>1</td><td>0.980</td></tr></table>

Fidelity The fidelity bound holds in every configuration, with $\rho _ { \mathrm { F E } }$ between 0.092 and 0.106 at layer4 and between 0.250 and 0.270 at the penultimate boundary. The slack has a structural explanation. Write $\varepsilon ( x ) = h ( x ) - D ( C ( x ) )$ ) for the reconstruction error and $\bar { \varepsilon } ( x )$ for its spatial average, so that $f ( x ) - { f } _ { A } ( x ) = W { \bar { \varepsilon } } ( x )$ . The bound then follows from two inequalities,

$$
\operatorname { F E } ( A ) = { \sqrt { \operatorname { \mathbb { E } } \| W { \bar { \varepsilon } } \| _ { 2 } ^ { 2 } } } \leq \| W \| _ { 2 } { \sqrt { \operatorname { \mathbb { E } } \| { \bar { \varepsilon } } \| _ { 2 } ^ { 2 } } } \leq { \frac { \| W \| _ { 2 } } { \sqrt { S } } } { \sqrt { \operatorname { \mathbb { E } } \| \varepsilon \| _ { 2 } ^ { 2 } } } = L _ { g } \operatorname { R E } ( A ) .\tag{29}
$$

The first holds because $W$ stretches no vector by more than its largest singular value $\| W \| _ { 2 }$ , and the second because averaging over S locations shrinks the Euclidean norm by at least a factor $\sqrt { S }$ The tightness ratios of the two steps, the alignment α and the spatial coherence κ, multiply to the tightness ratio of the bound,

$$
\rho _ { \mathrm { F E } } = \alpha \kappa , \qquad \alpha = \frac { \sqrt { \mathbb { E } \| W \bar { \varepsilon } \| _ { 2 } ^ { 2 } } } { \| W \| _ { 2 } \sqrt { \mathbb { E } \| \bar { \varepsilon } \| _ { 2 } ^ { 2 } } } , \qquad \kappa = \frac { \sqrt { S \mathbb { E } \| \bar { \varepsilon } \| _ { 2 } ^ { 2 } } } { \sqrt { \mathbb { E } \| \varepsilon \| _ { 2 } ^ { 2 } } } ,\tag{30}
$$

and both lie in $[ 0 , 1 ]$ by Eq. (29). The alignment α measures how much of the averaged error lies along the directions that $\dot { W }$ amplifies most, its top right singular vectors, and the spatial coherence κ measures how much of the error survives averaging over locations, with $\kappa = 1$ at the penultimate boundary. Both factors equal one for an error that is identical at every location and points along the direction that W amplifies most, so the constant $L _ { g }$ in Theorem 1 cannot be improved without assumptions on the error. Empirically, the alignment is nearly the same at both boundaries and for all methods, $\alpha \approx 0 . 2 6$ , so the reconstruction error is spread over many directions rather than concentrated along the directions that $W$ amplifies most. At layer4, averaging over locations removes a further part of the error, $\kappa \approx 0 . 3 8$ , which explains the additional slack at this boundary. As k grows, RE and FE decrease for all methods, and the ratio $\rho _ { \mathrm { F E } }$ decreases slightly, on average from 0.105 to 0.099 at layer4 and from 0.269 to 0.260 at the penultimate boundary between $k = 5$ and $k = 5 0$

Table 3: Tightness of the attribution bound for the nonlinear autoencoder, mean over three seeds. Bound is $\mathrm { F E } ( A ) + M \sqrt { \mathbb { E } \big [ \lVert C ( x ) \rVert _ { 2 } ^ { 4 } \big ] }$ with FE on the explained scores, shared by insertion (Ins.), occlusion (Occ.), and gradient-times-input (G I), and $\rho _ { \mathrm { A T E } }$ is the tightness of the attribution bound obtained by combining Eq. (13) with Theorem 3. The bound, and with it $\rho _ { \mathrm { A T E } } .$ , varies across seeds by up to 40% because M is estimated by sampling, whereas ADD varies by at most 0.4. For the four methods with affine decoders, ADD $< 2 \cdot 1 0 ^ { - 5 }$ and $\rho _ { \mathrm { A T E } } = 1$ for all three attribution functions.
<table><tr><td colspan="3"></td><td colspan="3">ADD(A, φ)</td><td colspan="3"> $\rho _ { \mathrm { A T E } } \times 1 0 ^ { 3 }$ </td></tr><tr><td>Boundary</td><td>k</td><td>Bound</td><td>Ins.</td><td>Occ.</td><td>G×I</td><td>Ins.</td><td>Occ.</td><td>G×I</td></tr><tr><td rowspan="3">layer4</td><td>5</td><td>2281</td><td>1.31</td><td>1.39</td><td>0.10</td><td>3.0</td><td>3.0</td><td>2.9</td></tr><tr><td>25</td><td>2381</td><td>2.34</td><td>2.92</td><td>0.06</td><td>2.9</td><td>1.9</td><td>2.4</td></tr><tr><td>50</td><td>2749</td><td>3.49</td><td>4.20</td><td>0.06</td><td>2.7</td><td>1.4</td><td>2.0</td></tr><tr><td rowspan="3">penultimate</td><td>5</td><td>2593</td><td>0.78</td><td>0.87</td><td>0.25</td><td>3.0</td><td>2.8</td><td>2.9</td></tr><tr><td>25</td><td>1909</td><td>1.70</td><td>3.32</td><td>0.13</td><td>3.5</td><td>2.2</td><td>2.9</td></tr><tr><td>50</td><td>1726</td><td>2.40</td><td>4.46</td><td>0.09</td><td>3.9</td><td>2.2</td><td>2.9</td></tr></table>

Model Completeness Model completeness error is an infimum over all heads and cannot be computed exactly. Every fitted head γˆ is admissible, so its held-out error $\widehat { \mathrm { M C E } } = \mathrm { R M S E } ( f , \widehat { \gamma } \circ C )$ estimates an upper bound on $\operatorname { M C E } ( A )$ . We fit two heads on the fitting half. The first refits the decoded head within its own family, starting from $g \circ D \colon$ when D is affine on each location, the functions $g \circ D$ are exactly the affine maps of the averaged representation v¯, so we fit $\hat { \gamma } ( v ) = \hat { W } \bar { v } + \hat { \beta }$ by least squares, and for the nonlinear decoder we fine-tune copies of D and $g$ by gradient descent. The second adds a learned correction to the decoded head, $\hat { \boldsymbol { \gamma } } = \boldsymbol { g } \circ \boldsymbol { D } + \boldsymbol { r }$ , where r is a multilayer perceptron initialized to zero. Both heads start from $g \circ D ,$ and we select them by validation error, so they do not exceed FE(A) beyond sampling error. We report the smaller of the two as $\widehat { \mathrm { M C E } }$ and $\hat { \rho } _ { \mathrm { M C E } } = \widetilde { \mathrm { M C E } } / \mathrm { F E } ( A )$ , which upper-bounds the true tightness of Theorem 2. Consequently, our results can certify slack in this bound, since $\mathrm { F E } ( A ) - \widehat { \mathrm { M C E } }$ lower-bounds $\operatorname { F E } ( A ) - \operatorname { M C E } ( A )$ , but they cannot certify its tightness. The gap is modest throughout: ρˆ ranges from 0.91 to 1.00. It is largest for PCA and the SAE, and a nonlinear correction reduces the error by at most 9% over the refitted decoded head. For k-means at the penultimate boundary, $\hat { \rho } _ { \mathrm { M C E } } = 1 \mathrm { : }$ : its assignments are nearly one-hot, and $g \circ D$ maps each cluster to the mean of $f$ over that cluster, which is already the best any head can do.

Attribution We explain the score of the predicted class for each input and evaluate all three attribution functions of Eqs. (9) to (11). The theory assumes a scalar output, whereas the explained class varies between inputs. The bounds still apply, because the triangle inequality in Eq. (13) and the Taylor bound in the proof of Theorem 3 hold for each input and every class separately; we therefore compute FE on the explained scores only and take $M$ as the largest curvature constant over the explained classes. For the four methods with affine decoders, $\gamma = g \circ D$ is a composition of affine maps, since $g$ is affine at both boundaries, and is therefore affine even though the SAE and k-means have nonlinear encoders. In agreement with Proposition 1, the additivity error vanishes up to floating-point error for all three attribution functions, $\mathrm { A D D } ( A , \varphi ) < 2 \cdot 1 0 ^ { - 5 }$ compared with FE  6 on the explained scores, and the attribution error coincides with the fidelity error, so $\rho _ { \mathrm { A T E } } = 1$ . Only the nonlinear decoder produces curvature. We estimate M from the largest ratio of gradient differences over sampled pairs of nearby concept representations, which lower-bounds the true constant, so the reported bound is itself an estimate. The curvature term does not depend on the scale of the concept representations: replacing E by λE and D by $D ( \cdot / \lambda )$ for $\lambda > 0$ leaves $f _ { A }$ and the attributions unchanged, scales M by $\lambda ^ { - 2 }$ , and scales $\sqrt { \mathbb { E } [ \| C ( x ) \| _ { 2 } ^ { 4 } ] }$ by $\lambda ^ { 2 }$ , so the looseness of the bound reflects the curvature of γ rather than the units of the concept representations. For the nonlinear autoencoder (Tab. 3), both bounds hold for all three attribution functions but are loose, with $\rho _ { \mathrm { A T E } }$ between 0.001 and 0.004. Gradient-times-input is nearly additive, with $\mathrm { A D D } \leq 0 . 2 5$ whereas insertion and occlusion reach 3.5 and 4.5 at $k = 5 0$ . Better fidelity does not imply better attributions: as k grows, FE on the explained scores decreases while ATE for insertion stays nearly constant, and for occlusion ATE even falls below FE because the additivity error partly cancels the fidelity error.

Limitations of the Evaluation All errors are estimated on a finite held-out sample. Model completeness error is only bounded from above, and M is estimated from below. The closed-form Lipschitz constant restricts the evaluation to boundaries with affine prediction heads, and at earlier boundaries $L _ { g }$ would itself have to be bounded.