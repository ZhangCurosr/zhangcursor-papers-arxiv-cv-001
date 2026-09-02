# Denoising Difusion Generative Models Secretly Calculate Attentions

Farzan Haddadi†, Leila Monfared†, Ebrahim Rezaii†, Mohammadreza Malek-Mohammadi<sup>\*</sup>, Pejman Zakalvand†, Narges Mokhtari†

Abstract—Denoising difusion models are the dominant architecture for image generation, whereas most natural language generation and modeling are primarily handled by wellknown transformer architectures employing attention mechanism. Here, we show that difusion models also inherently use an attention mechanism very similar to that of transformers. Therefore, attention emerges as a universal machine learning principle, based on a general training objective. We also show similarities in basic functional principle of auto-encoders and attention-based models. These equivalences allows us to interchange these designs based on practical requirements. As an example, we can reformulate the difusion framework to reduce the lengthy training process and computation-intensive image generation. Using this approach, a simplified algorithm is proposed for image generation which is based on attention mechanism. Results show that the attention-based implementation achieves comparable performance with significantly less efort and computational resources.

Index Terms—Difusion model, attention mechanism, autoencoder, data manifold, latent space, interpolation.

## I. Introduction

A <sup>RTIFICIAL</sup> <sup>intelligence</sup> <sup>(AI)</sup> <sup>encompasses</sup> <sup>a</sup> <sup>multi-</sup><sub>tude of proposed architectures for diferent tasks.</sub> Throughout the history of AI, it was a surprise that gradually eficient unified system architectures replaced this multitude of systems for various applications. An important turning point was the advent of the dot-productbased attention mechanism [1]. Transformers were first proposed for tasks of natural language processing (NLP) such as question-answering and chatbots [2]. Soon, it was realized that transformers also have stellar performance in vision applications with the advent of the vision transformer (ViT) model [3].

In image generation, difusion systems are arguably the best option [4]. Yet, for the best performance they are used in tandem with transformer-based components. Specifically, difusion principle is used to determine the loss function and high-level train and test schemes, while in network level, an auto-encoder is trained which in turn uses attention mechanism and transformer architecture as its main component [5].

Here, we show that the difusion principle is equivalent to the attention mechanism. Difusion denoising probabilistic generative model (DDPM) [4], trains a neural network to reconstruct the original image from its noisy copies. Then, starting from pure noise, the system can generate a realistic image based on the learned distribution of the dataset.

Detecting the presence of a signal from its noisy version is a classic statistical problem [6]. Specifically, when the noise is Gaussian, the solution is an innerproduct detector. While in the context of machine learning theory, we train a neural network to solve this problem, we will show that this problem enjoys a closed-form solution which is basically the attention mechanism. This not only shows that the difusion system is identical to the attention mechanism, but also provides a previously known statistical basis for the very attention mechanism as the closed form solution to the signal recovery problem in the Gaussian noise [7].

The closed form solution of the difusion system was first derived in [8]. The authors calculate optimum score function of a difusion system for a finite dataset. However, they did not notice the relation of their result to the well-known attention mechanism. Then [8] extends the result to include additional structures like CNN network which is used in the DDPM implementation and develops a creativity theory for CNN difusion systems based on locality and positional invariance properties. Here we use a diferent approach based on the difusion loss function and convex optimization to reach the same closed form solution while extending the analysis to include resemblance to transformer and autoencoders. The equivalence between difusion and transformer principles have broad theoretical and practical consequences in machine learning theory.

It is not suficient to understand the core difusion principle in the forms of what we investigate as closed-form one-step solution or convergence trajectory. In practice, difusion systems are implemented in the inner latent space of a Variational Auto-Encoder (VAE) [9]. Also, in practice the main neural network that performs denoising in the difusion system is an AE. We argue that the same equivalence exists between attention and AE, in that both systems eventually project a random input on a learned data manifold [10], [11]. Therefore, we reach to an equivalence between three of the most important modern AI systems, namely: attention, difusion , and AE. This enables us to interchange these systems depending on the situation.

DDPM difusion system sufers from heavy computations in both training and generation which requires running an encoder hundreds of times on a random input.

Some eforts have been made to reduce computational complexity specially in generation. A trend in the literatur is to reduce the steps for image generation using non-Markovian statistical models as in Difusion Denoising Implicit Model (DDIM) [12].

Regarding the taining, DDIM is equivalent to DDPM and very computationally complex. It requires adding noise for hundreds of times to each image in dataset to teach network to denoise images with varying degree of noise efectively. Therefore, reducing computational complexity in the training phase is highly important. Here, we calculate a closed form denoiser which reduces the need for training a network on the dataset.

As a result of the equivalence of difusion and attention, we conclude that deliberate interpolation between neighbor real images in the latent space results is an approximately realistic image. This interpolation can be refined and projected on the real image manifold by a limited recursive application of the outer AE. Overall, the proposed system can generate a realistic image without training an inner AE network, and just by using the dataset images. The results show good quality with considerably lower computations.

The manuscript is organized as follows: The main result and discussion is presented in Sec. II, where a background review, discussion on equivalence between difusion and attention, and convergence topics are covered. AutoEncoder equivalence with attention and projection on data manifold are addressed in Sec. III. A data structure for fast attention calculation is discussed in Sec. IV. The proposed low complexity generative network is demonstrated in Sec. ${ \mathrm { V } } ,$ while simulation results are exhibited in Sec. VI.

## II. Difusion Systems Calculate Attention

## A. Background

A difusion system is trained to denoise noisy images in a recurssive manner. In the forward process, Gaussian noise is gradually added to the true image $x _ { 0 }$ in hundreds or thousands of steps [13]:

$$
\begin{array} { l } { \displaystyle x _ { t } = \sqrt { \alpha } \ x _ { t - 1 } + \sqrt { 1 - \alpha } \ n _ { t } } \\ { \displaystyle n _ { t } \sim \mathcal { N } ( \boldsymbol { 0 } , \boldsymbol { I } ) } \end{array}
$$

for $\alpha < 1$ . Therefore, each image in the forward process can be written based on the true initial image as:

$$
x _ { t } = \sqrt { \alpha ^ { t } } \ x _ { 0 } + \sqrt { 1 - \alpha ^ { t } } \ n _ { t } ^ { \prime }\tag{1}
$$

This eventually leads to a noise-only image at the final stage of the forward process since lim $_ { 1 \ i  + \infty } \alpha ^ { t } = 0$ . We train a neural network to recover the true image $x _ { 0 }$ from each noisy image $x _ { t }$ . Theoretically, this results in a neural network that can produce a realistic image from a pure noise and based on the true distribution of the data which is represented by the dataset used for training.

## B. Prior Art

In practice, we train the denoiser neural network on a finite dataset $\mathcal { D } = \{ s _ { 1 } , . . . , s _ { N } \}$ . This finite dataset training situation was first analyzed in [8]. Adding Gaussian noise incurs a mixture of Gaussian distribution on dataset points in signal space:

$$
p _ { t } ( y ) \ = \ \frac { 1 } { | \mathcal { D } | } \sum _ { s \in \mathcal { D } } \mathcal { N } ( y \mid \sqrt { \alpha ^ { t } } s \ , ( 1 - \alpha ^ { t } ) I )\tag{2}
$$

Then [8] uses a flow matching continuous-time model which is solved backward in time for signal generation:

$$
- \dot { y } _ { t } = \gamma _ { t } \left( y _ { t } + s _ { t } ( y _ { t } ) \right)\tag{3}
$$

in which $\gamma _ { t }$ is a constant and $s _ { t } ( y ) : = \nabla _ { y } \log p _ { t } ( y )$ is the score function. The score function of the mixture distribution in (2) is then [8]:

$$
s _ { t } ( y ) = \frac { 1 } { 1 - \alpha ^ { t } } \sum _ { s \in \mathcal { D } } ( \sqrt { \alpha ^ { t } } s - y ) W _ { t } ( y , s )\tag{4}
$$

$$
W _ { t } ( y , s ) = \frac { \mathcal { N } ( y \mid \sqrt { \alpha ^ { t } } s , ( 1 - \alpha ^ { t } ) I ) } { \sum _ { s ^ { \prime } \in \mathcal { D } } \mathcal { N } ( y \mid \sqrt { \alpha ^ { t } } s ^ { \prime } , ( 1 - \alpha ^ { t } ) I ) }\tag{5}
$$

Then [8] continues the analysis to develop a theory for combinatorial creativity of difusion models when the denoiser is a CNN network.

## C. Difusion is Attention

Our analysis starts from the same finite dataset training assumption and then proceeds in a diferent direction to find the relation of the difusion model to transformers and autoencoders as current dominant models in machine learning research.

When we train the network on the finite dataset ${ \mathcal { D } } ,$ the network learns to recover each $s _ { i }$ from its noisy versions. We guess that when sampling the system for a new signal, although we use a random noise as the input, the system still outputs just an interpolation of the signals it had seen in the dataset since it has not seen anything else.

For signal $s _ { i }$ and Gaussian noise n, DDPM minimizes Mean-Squared Error (MSE) loss function:

$$
\mathrm { D D P M } : \quad \underset { \theta } { \operatorname* { m i n } } \parallel \hat { x } _ { 0 } ( y , t ; \theta ) - s _ { i } \parallel ^ { 2 } \quad : \quad y = s _ { i } + n\tag{6}
$$

in which $x _ { 0 }$ is the real signal, $\scriptstyle { \hat { x } } _ { 0 }$ its estimate, $t \in$ $\{ 0 , \ldots , T \}$ time index, and θ is the neural network’s parameters. If we infinitely train the network with fixed $s _ { i }$ and $y ,$ it learns to convert input $y$ to output $s _ { i }$

But we do not train the network just for one sample image. Adding Gaussian noise to any signal $s _ { i }$ for suficiently many times, can reach any point y in the signal space many times. A geometrical scheme of the problem is depicted in Fig. 1. Reaching any fixed point y from various signals $s _ { i }$ occurs in accordance with the Gaussian distribution of the noise term n. Therefore, if we fix $y ,$ the system is trained with an average loss function:

![](images/6f959cdaa6ca941b2ca0c101dbb046371ddafe6a566698f3dcc96044a6d268d1.jpg)  
Fig. 1. Geometry of difusion system. Three signals $s _ { 1 } , \ s _ { 2 } ,$ and s<sub>3</sub> are used for system training. We are interested in calculating the output for random input y. We show that the output is the attention combination of training signals.

$$
\operatorname* { m i n } _ { \theta } \quad \sum _ { i = 1 } ^ { N } p ( y | s _ { i } ) \| \hat { x } _ { 0 } ( y , t ; \theta ) - s _ { i } \| ^ { 2 }
$$

$$
p ( y | s _ { i } ) = \frac { 1 } { \sqrt { 2 \pi } \sigma } \exp \left. - \frac { \| y - s _ { i } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right.\tag{7}
$$

For fixed y, (7) is a Quadratic Programming (QP) convex optimization problem:

$$
\operatorname* { m i n } _ { z } \quad \sum _ { i = 1 } ^ { N } p _ { i } \| z - s _ { i } \| ^ { 2 }\tag{8}
$$

$$
p _ { i } \geqslant 0 \quad : \quad i = 1 , \cdot \cdot , N\tag{9}
$$

Where (9) is a conic criterion for the strictly convex optimization in (8). The optimal solution for the above optimization problem can be derived using gradients [14]:

$$
\nabla _ { z } \sum _ { i } p _ { i } \| z - s _ { i } \| ^ { 2 } = 2 \sum _ { i } p _ { i } z - 2 \sum _ { i } p _ { i } s _ { i } = 0\tag{10}
$$

$$
z ^ { * } = \frac { \sum _ { i = 1 } ^ { N } p _ { i } s _ { i } } { \sum _ { i = 1 } ^ { N } p _ { i } } = \frac { \sum _ { i = 1 } ^ { N } \exp \left\{ - \frac { \| y - s _ { i } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} s _ { i } } { \sum _ { i = 1 } ^ { N } \exp \left\{ - \frac { \| y - s _ { i } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} }\tag{11}
$$

Assume that σ is suficiently small. Then just those signals $s _ { i }$ that are in the neighborhood of y are efective in the convex combination in (11). Therefore, if N is large enough and the space is suficiently populated with data points, we can safely assume that ∥s<sub>i</sub>∥’s are approximately constant. Other situations also may hold the condition of approximately constant $\| s _ { i } \|$ like in image dataset when total energy of most of the images are in the same range. In this case we will have:

$$
z ^ { * } \simeq \frac { \sum _ { i = 1 } ^ { N } \exp \left\{ \frac { < y , s _ { i } > } { \sigma ^ { 2 } } \right\} s _ { i } } { \sum _ { i = 1 } ^ { N } \exp \left\{ \frac { < y , s _ { i } > } { \sigma ^ { 2 } } \right\} } = \mathrm { A t t e n t i o n } \{ y , \left\{ s _ { i } \right\} \}\tag{12}
$$

in which $< \cdot , \cdot >$ denotes the inner vector product.

As we can see in (12), the optimum solution to the input signal y in a difusion system with fixed σ is the attention vector. But this is just one step of the overall difusion system since each step in the forward difusion process has a distinct increasing value of σ. We will discuss this convergence issue in the next section II-D.

The derivation resulting in (12) is also a mathematical basis for the very attention system. It shows that attention is the optimum solution for recovering signals in Gaussian noise as it was shown earlier in [7]. Note that in the conventional attention system [1], the noise variance $\sigma ^ { 2 } = { \sqrt { p } } .$ , where $p$ is the dimension of the space.

This derivation also shows a basis for the softmax layer. The exponential form in (12) with $\sigma = 1$ , is the softmax output when $s _ { i }$ is a standard one-hot label vector $\mathbb { I } _ { i }$

$$
z _ { i } ^ { * } = \mathrm { S o f t m a x } _ { i } \{ y \} = \frac { e ^ { y _ { i } } } { \sum _ { j = 1 } ^ { N } e ^ { y _ { j } } }\tag{13}
$$

$$
z ^ { * } = \frac { \sum _ { i = 1 } ^ { N } e ^ { < y , \mathbb { I } _ { i } > } \mathbb { I } _ { i } } { \sum _ { i = 1 } ^ { N } e ^ { < y , \mathbb { I } _ { i } > } } = \mathrm { A t t e n t i o n } \{ y , \{ \mathbb { I } _ { i } \} \}\tag{14}
$$

which shows the equivalence of softmax and attention layers:

$$
\operatorname { S o f t m a x } \{ y \} = \operatorname { A t t e n t i o n } \{ y , \{ \mathbb { I } _ { i } \} \}\tag{15}
$$

Note that the result in (11) can be deduced using the results in [8], replacing (4) and (5) in (3). However, the authors in [8] did not proceed to deduce the attention mechanism from their closed-form score function solution of the difusion system in (4).

## D. Convergence

In Sec. II-C, it was shown that each step of a difusion system calculates attention for a predefined σ value. The reverse generative process in DDPM starts with a random Gaussian y and refines it with denoising network trained with noisy images from dataset. Theoretically, there should be hundreds of denoising networks each trained for denoising with a specific σ. But usually a shared network is trained and used for every values of σ to reduce complexity. Therefore, output of each step is then recursively fed in the network to generate another less noisy output image. This is repeated hundreds of times to reach a high quality image.

As can be seen from $( 1 ) , \sigma _ { t } ^ { 2 } = 1 - \alpha ^ { t }$ is increasing with t in the forward process. This will result in a decreasing $\sigma _ { t }$ in the backward generative process. We start with a Gaussian random y. In the early stages when $\sigma$ is large, diferences in distances $\| y - s _ { i } \|$ in (11) make negligible efect. Therefore, the output sequence converges to the mean value of the nearby vectors.

When σ gets small, diferences in distances are promoted in (11) and the results quickly converge to the nearest signal, with some contribution from nearby signals. This is a sparse representation, as the simulation of (11) for a simple setup shows in Fig. 2. In this simulation, $\sigma _ { t } = 0 . 9 6 \sigma _ { t + 1 }$ is a decreasing sequence which is decreased gradually from 1 to 0.37 in the reverse process. Although, as we will discuss later in Sec. II-E, it seems that σ does not converge to zero because of practical limitations in the denoising neural network design and training.

![](images/2c8194799aeb23f1424f2432e181a7fb763b35dabc3206da50e59dd5112dccc8.jpg)  
Fig. 2. Convergence of a random point y to a sparse solution through sequential attention calculation when σ is decreasing. M is the mean of the $s _ { i }$ signals. Starting point y passes near M when σ is large in the first stages.

The difusion process initially converges to the mean point of the distribution. Then in the long term, it converges to the vicinity of some dense region of the dataset. This is in accordance with the fact that the difusion process simulates the real data distribution as reflected in the dataset.

Knowledge about the convergence of the difusion process enables us to circumvent the whole convergence process by directly using a convex combination of some nearby dataset signals as output of the generation process. This basically reduces not only the lengthy generation process, but also the very lengthy training process of denoising neural network.

## E. Finite Training

In this manuscript, we have generally assumed infinite training of denoising network. This enables the training process to cover every y point in the space. It also enables the network through many gradient descent updates, to achieve the ground-truth closed form solution of the training process optimization.

This is rarely the case in practice. Due to limited resources like silicon, time, and energy, we usually perform minimal training up to the point of acceptable network performance. Here we discuss the efect of non-asymptotic training on the network behavior.

First, the network does not experience every y in the space during the training. It only sees finite adjacent inputs $\{ y _ { 1 } , \ldots , y _ { k } \}$ . It learns the correct direction of move from these inputs to outputs $\{ o _ { 1 } , \ldots , o _ { k } \}$ , respectively. Any new y is sorrounded by multiple seen points. Since the network is a continouos system, we expect an average behavior in $y$ if the training set is suficiently dense. This guides $y$ to regions of higher density which are experienced more in training and gradually we experience

better refinements.

$$
o _ { i } = \mathbb { D } \left( \mathbb { E } \left( y _ { i } \right) \right)\tag{16}
$$

$$
y = \sum _ { i = 1 } ^ { k } \alpha _ { i } y _ { i } \quad  \quad \mathbb { D } ( \mathbb { E } ( y ) ) = \sum _ { i = 1 } ^ { k } \beta _ { i } o _ { i }\tag{17}
$$

where $\left\{ \alpha _ { i } \right\}$ and $\{ \beta _ { i } \}$ are two convex coeficient sets, E is the encoder function and D is the decoder.

The second consequence of finite training is that the network does not learn the optimal solution of the optimization in (7) for each training input $y _ { i }$ . We can model this training error as an additive noise in the output of the network. The same is true for the unseen input error and we can model the error in D $\left( \mathbb { E } \left( y _ { i } \right) \right)$ also as an additive noise.

When noise is added to the output of the denoiser network in a difusion system, the output will no longer be a sparse attention combination even when $\sigma$ of the difusion system is very small. This is in fact a new additive noise in the backward process:

$$
z _ { t - 1 } = z _ { t } + n _ { t } ^ { \prime \prime } \quad \Rightarrow \quad x _ { t } = \sqrt { \alpha ^ { t } } x _ { 0 } + \sqrt { 1 - \alpha ^ { t } } n _ { t } ^ { \prime } - n _ { t } ^ { \prime \prime }\tag{18}
$$

where $n _ { t } ^ { \prime \prime }$ is the new noise term indicating errors emanating from finite training process.

Therefore, the overall efect of finite training can be modeled as a higher value of σ which increases the number of intermittent signals. This means that although in the difusion framework, theoretically σ gradually converges to zero, practical limitations leads to a stable higher limiting value for σ. This prevents the optimization process to achieve a very sparse solution and the final output will be a convex combination of many signals.

## III. Auto Encoders

In practice, difusion systems use auto-encoders as both the outer encoder layer that the difusion system operates inside [9], and the inner recurrent denoising network. Therefore, it is also crucial to understand working principles of AE to reduce computational complexity of the difusion systems. In this section we use a diferent perspective to explain AE and attention system similarity.

## A. Auto-encoders are attentions

As we showed in Sec. II, there is a deep relationship between difusion generative systems and attention mechanism. Here, we also show a strong resemblance between AEs, attention systems, and difusion systems. This will present a great unity between three of the most important artificial intelligence systems currently in use.

First of all, notice that the derivations in Sec. II-C is not limited to a difusion system. In fact, it is a widespread setting in unsupervised learning systems. Assume we have an AE network which as usual use MSE loss function to reconstruct input images or other signals in an unsupervised manner. Regardless of the AE’s inner structure, we assume perfect training of the system. Most of the times, an augmentation technique of adding Gaussian noise to limited dataset signals is used for better training and performance. Now, we can see that the mathematical setup is identical to the optimization in (7). Therefore, the same solution as in (11) applies to the problem. This certifies that such an AE system is equivalent to an attention system in (12) with fixed σ in the infinite training setup. Attention parameter σ in (11) can be tuned according to the structural parameters of the AE for a satisfactory equivalence.

![](images/03c82493ff583b737d9949694af7849929f21de0ba0d016ba846af26c6730a40.jpg)  
Fig. 3. An auto-encoder with m-dimensional input/output vectors and k-dimensional code layer.

The equivalence between AE and attention is doublesided. It means that we can in principle replace current attention systems with AEs. A good example for such a replacement is the linear attention [15], which is known to be a finite-state attention mechanism with compressed rather than full state. This is reminiscent of recursive autoencoder structure [16], especially when we consider the causal implementation of the linear attention.

## B. Auto-encoders and Manifolds

Assume an auto-encoder with m-dimensional inputoutput and k-dimensional code layer as in Fig. 3. Manifold hypothesis assumes that data from the dataset is located on a high-dimensional manifold in input space [10]. It is well known that the autoencoder estimates the data manifold during the training phase [11]. We will show that AE projects the m-dimensional input signal on to a k-dimensional manifold which represents locus of data in the dataset and their meaningful augmentations in the space.

Assume that the AE is fully trained with the dataset and therefore, it can now regenerate signal s in the output approximately. In the input space, we add a diferential length vector $v _ { 1 }$ to s. With high probability, no ReLU-like switch in the network is on its change position. Therefore, for any diferential change in the input, the system is linear. Then, changing the input s with $v _ { 1 }$ results in a diferential $c _ { 1 }$ vector change in the code layer:

$$
\mathbb { E } \left( s + v _ { 1 } \right) = \mathbb { E } ( s ) + c _ { 1 }\tag{19}
$$

Continue these changes with orthogonal diferential vectors $v _ { i }$ that results each in code layer change of $c _ { i } .$ . In the $k + 1 \mathrm { { t h } }$ step, the set of $\{ c _ { 1 } , \ldots , c _ { k + 1 } \}$ is a linearly dependent set since the code layer vector space is $k -$ dimensional. Therefore, a linear combination of these vectors sums to zero:

$$
\sum _ { i = 1 } ^ { k + 1 } \alpha _ { i } c _ { i } = 0\tag{20}
$$

The same set of $\alpha _ { i }$ coeficients can be used to find a direction in the input space that keeps the code layer, and therefore the output constant:

$$
\mathbb { D } \left( \mathbb { E } \left( s + \sum _ { i = 1 } ^ { k + 1 } \alpha _ { i } v _ { i } \right) \right) = \mathbb { D } \left( \mathbb { E } ( s ) + \sum _ { i = 1 } ^ { k + 1 } \alpha _ { i } c _ { i } \right) = s\tag{21}
$$

Therefore, we showed that any $k + 1$ vectors $v _ { i }$ in the input space can form a direction with null efect in the output. This direction is:

$$
w = \sum _ { i = 1 } ^ { k + 1 } \alpha _ { i } v _ { i }\tag{22}
$$

The local null space of the encoder at s is defined as the local Afine vector space of all such w vectors:

$$
\mathbb { N } ( s ) : = \{ s + w \mid \mathbb { E } ( s + w ) = \mathbb { E } ( s ) \ , \ \| w \| < \delta \}\tag{23}
$$

where $\delta$ is a small constant. The null space $\mathbb { N } ( s )$ is local to s and is efective just in a δ-neighborhood of s. At most k input vectors $v _ { i }$ can be chosen that does not lie in this local null space. Therefore:

$$
\dim ( \mathbb { N } ( s ) ) = m - k\tag{24}
$$

Suppose $\mathbb { M } ( s )$ is the k-dimensional local Afine space orthogonal to the local null space:

$$
\begin{array} { c }  \mathbb { M } ( s ) : = \left\{ \begin{array} { l } { s + v \mid < v , w > = 0 \ : } \\ { \qquad \forall w : s + w \in \mathbb { N } ( s ) \ , \ \lVert v \rVert < \delta \right\} } \end{array} \end{array}\tag{25}
$$

In the vicinity of $s ,$ any displacement along the null space $\mathbb { N } ( s )$ does not change the values of the code layer and therefore, the output. But displacement along manifold $\mathbb { M } ( s )$ translates to changes in code layer and output. Therefore, the AE will tune its null space $\mathbb { N } ( s _ { i } )$ such that it direct farthest from the neighbors of $s _ { i } .$ . This will set the estimated data subspace $\mathbb { M } ( s _ { i } )$ in the nearest position that best cover the neighbors of $s _ { i }$ and orthogonal to the local null space. This is in accordance with autoencoder estimating a manifold with good fit to neighboring data points [11] with extensive training on suficiently dense datasets.

Here we assume that the dataset is dense enough that the continuity of the data manifold can be tracked by the AE as in Fig. 4 . We also assume that the AE does not have a specific structure (like convolutional or any other weight sharing scheme). This helps the AE to follow any direction in the data space without any bias. It is wellknown that networks like convolutional networks better estimate the data manifold. This is because some of the data structure, like image content translation invariance is embedded in such networks. This is equivalent to more data points which helps the network better estimate the data manifold.

![](images/656619b34b04298aa8bade79661b0f28f88abc5d67096d303dbbe5a19eeb2ebf.jpg)  
Fig. 4. The dataset  is assumed dense enough that the data manifold F can be followed by the auto encoder after suficient training. Input y is projected onto the data manifold in a curved path following the local null spaces N and eventually arrives at F in a orthogonal direction.

Any input y outside the manifold defines a specific local null space $\mathbb { N } ( y )$ which aggregates directions of moves that does not change the code layer and output layers values. Therefore, AE moves y in a curved path along local null spaces to finally reach the AE estimated data manifold F in h. Since $\mathbb { N } ( h ) \perp \mathbb { M } ( h )$ we reach to the AE manifold perpendicularly in the last touch as is shown in Fig. 4.

## C. Attention and manifold

Here, we show that the attention mechanism also projects data on a low dimensional manifold. This is in-line with our core insight that basically these three structures: AE, attention, and difusion systems are similar.

Assume that y is near the data manifold. Dataset satisfies conditions that we can use (11) in place of the better-known attention formula in (12). Also, assume that $\sigma$ is small enough that only k nearest signals:

$$
A _ { i } : = \{ s _ { i _ { j } } \} _ { j = 1 } ^ { k }\tag{26}
$$

are efective in the attention sum in (11). Then, the attention coeficients are:

$$
p _ { i } \simeq \frac { \exp \left\{ - \frac { \| y - s _ { i } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} } { \sum _ { s \in A _ { i } } \exp \left\{ - \frac { \| y - s \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} } = : p _ { i } ( y , A _ { i } )\tag{27}
$$

The adjacency set $A _ { i }$ forms a k − 1-dimensional Afine subspace. Assume that the orthogonal projection of $y$ onto this subspace is h as in Fig. 5. Then:

$$
\| y - s _ { i } \| ^ { 2 } = \| y - h \| ^ { 2 } + \| h - s _ { i } \| ^ { 2 }\tag{28}
$$

Then $\| y - h \| ^ { 2 }$ is shared in numerator and denominator of (27), eliminating it:

$$
p _ { i } = \frac { \exp \left\{ - \frac { \| h - s _ { i } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} } { \sum _ { s \in A _ { i } } \exp \left\{ - \frac { \| h - s \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} } = p _ { i } ( h , A _ { i } )\tag{29}
$$

![](images/a99b4b34e26ff5cc0ed452be3b26c24c6b598f867a1b84325bd03bd31905c824.jpg)  
Fig. 5. Geometry of attention mechanism. Point y with a small ofset from the data manifold F is projected orthogonally to the Afine subspace spanned by nearby signals on the data manifold by attention mechanism.

Output of h from the attention mechanism is a convex combination of $s \in A _ { i }$ and therefore is on their Afine subspace. Also, the fact that the output of $y$ is the same as the output for h shows that there is a local Afine null space orthogonal to the local afine space of nearby signals in $A _ { i }$ that does not change the output of the attention.

## D. Auto-encoders resemble attention

The attention mechanism when act on a point y near the data manifold F, orthogonally projects it on the manifold at point h, and then transform h to another point in the convex hull of signals in $A _ { i }$ in the output. Overall, we can conclude that the attention mechanism projects any point y near the data manifold to some point on the manifold too and this is what also auto encoders do.

The attention mechanism depends on the parameter $\sigma ,$ while the AE is characterized by its code layer dimension k. The equivalence between attention and AE requires fine tuning these parameters. As stated before, setting a small $\sigma$ is required in order that approximately $k + 1$ nearby signals are efective in the attention. This will result in the estimation of a k-dimensional manifold for data similar to an AE with k-dimensional code layer. This is not an exact equivalence because the required σ parameter may not be constant in diferent parts of the input space.

In a conventional difusion system, we have an outer autoencoder which compresses data to a lower dimension. Then a recursive inner autoencoder denoises an input noise. We have shown that not only the high level difusion framework is equivalent to the attention mechanism, but also both autoencoders in the network structure realization perform in a similar way to the attention mechanism. Therefore, we can alternate between these three structures based on the situation.

If attention and auto encoder structures are basically similar, what is their diference? The diference lies in where the computational load is concentrated. In an auto encoder, we use vast amount of data in training phase while the test phase is as simple as one forward calculation of the network. The attention mechanism is reverse. Calculating attention does not require any training while all the computational load is concentrated in the test time where the forward pass of the network requires vast amount of product calculations with a long context of vectors. Therefore, we may be able to reduce the complex training phase of the auto encoder and difusion systems by using some simpler form of the attention mechanism.

![](images/be27deeb2a6a68f8fcc820b98f2b8e3038a64c1c53304df93c6a1adcee442bf8.jpg)  
Fig. 6. Proposed attention-based generative architecture.

## IV. Tree Cross Attention

The problem with the attention mechanism is that it involves calculating the dot product of the input y with every image in the dataset. These image datasets are usually very large in industry-level network training. Since usually σ parameter of attention is small, it sufices to calculate the attention just for nearby signals. But finding the nearbies requires calculating the distances to all images in the dataset which is equivalent to inner product computation.

Here, a data structure can lead to considerable reduction in the computational complexity. Calculating the dot product of y with every images in the dataset is of $\mathcal { O } ( | \mathcal { D } | )$ complexity. But since there is a great deal of redundancy in this calculation, we can save some information for reuse.

Data can be order in a tree structure based on similarity [17]. Each node in the tree is an image and any new image is tested with every nodes in each tree level to see with which branch it is more similar. If there was a similar branch, the new image goes down to that branch’s next level children. If there were no similar branch, make a new branch in the current level.

In this way, a balanced tree can be constructed in which similar and nearby images are clustered in a branch. Then for each new input signal y, we need a few inner product calculations of ${ \mathcal { O } } ( \log | { \mathcal { D } } | )$ which is the tree depth to find its neighbors for eficient attention calculations. This method is called tree cross attention and is presented in the literature [18], for the situations where base attention vectors are constant. Therefore, attention calculation is not restrictive in an image generation scenario.

## V. Proposed Generation Algorithm

Theoretical equivalence of difusion and attention systems helps to reduce the computational complexity of image generation. We have seen in Sec. II-D that any noise input to a difusion system experiences a long road to its final position in latent space.

Since the solution to difusion is an approximately sparse one (Fig. 2), we can start image synthesize from one of dataset images. Since attention to latent signals is approximately constant in the neighborhood of the image latent, we can calculate the attention of the latent signal to other neighbor signals.

This attention calculation helps the resulting signal to reside on or very near the data manifold since it is equivalent to passing the signal once through the hypothetical difusion trained AE denoiser. Therefore, we can pass the resulting latent through the decoder and reach to a high quality innovative image. Since there may be slight disposition from the data manifold, we can pass the resulting image a few times through the AE to better fit it to the data manifold.

Therefore, there will be no need to train a denoiser auto encoder as the working engine of the difusion system. We just need to train an outer AE which projects the input images to the low-dimensional latent space. Then in the generation phase, we just rely on attention calculation near the signal latents. Therefore, we not only reduced the training phase calculations, but also in the test phase, we reduce the long journey of the input noise to the distribution mean value and then to nearby of one of the signals. We just perform the final step with calculating the attention of the signal to its neighbbors including itself. The block diagram of the proposed image generation algorithm is depicted in Fig. 6.

The generation process is done in the learned latent space of trained VAE. As described in Alg. 1, for a random latent query q, we find its K nearest neighbors $\{ \mathbf { s } _ { k } \} _ { k = 1 } ^ { K } .$ New latent representation is generated through attention interpolation between the query and its neighbors. The generated latent representation is then passed through the VAE decoder to obtain a new image. Since nearby latent representations correspond to semantically related data, their interpolation produce meaningful intermediate representations.

Although using plain attention between querry and its neighbor images produce meaningful images, we use a fine grain spatial attention mechanism to get better quality. Spatial attention is calculated based on the distance of each pixel depth vector in querry to its corresponding pixel depth vector in a neighbor image as presented in Alg. 2. This generates a spatial attention map that determines the contribution of each pixel to the generated sample. Example spatial attention maps in Fig. 7 illustrate how the proposed method selectively exploits spatial information from neighbors during latent-space sampling. Note that spatial attention resembles the patch similarity selection mechanism that is developed as the theoretical solution of a convolutional difusion network in [8].

Algorithm 1 Attention-Based Image Generation   
1: Input: Latents $\{ \mathbf { s } _ { i } \} _ { i = 1 } ^ { N }$ , query q, number of neighbors   
$K ,$ attention parameter σ, noise scale η   
2: Output: Generated image x   
3: $\begin{array} { r } { r  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \mathbf { s } _ { i } \| _ { 2 } } \end{array}$   
4: $d _ { i }  \| \mathbf { q } - \mathbf { s } _ { i } \| _ { 2 } \quad \forall i$   
5: $S \gets \{ \mathbf { s } _ { i } \ | \ i \in \mathrm { T o p K } ( - d _ { i } ) \}$   
6: W ← SpatialAttention $( \mathbf { q } , { \mathcal { S } } , \sigma )$   
7: $\begin{array} { r } { \mathbf { z }  \sum _ { \mathbf { s } _ { i } \in \mathcal { S } } \mathbf { W } _ { i } \odot \mathbf { s } _ { i } } \end{array}$   
8: $\mathbf { z } \gets r \frac { \mathbf { z } } { \lVert \mathbf { z } \rVert _ { 2 } + \epsilon }$   
9: $\mathbf { z } \gets \mathbf { z } + \eta \dot { \epsilon } , \quad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$   
10: $\mathbf { z } _ { q } \gets \mathrm { Q }$ uantize(z)   
11: $\mathbf { x } \gets \mathbb { D } ( \mathbf { z } _ { q } )$   
12: for $j = 1 , \dots , 3$ do   
13: x ← AE (x)   
14: end for   
15: return x   
Algorithm 2 Spatial Attention   
1: Input: Query $\mathbf { q } \in \mathbb { R } ^ { D \times H \times W }$ , neighbor latents ${ \boldsymbol { S } } =$   
$\{ \mathbf { s } _ { i } \} _ { i = 1 } ^ { K } .$ , scale parameter σ   
2: Output: Attention weights $\mathbf { W } \in \mathbb { R } ^ { K \times H \times W }$   
3: for $i = 1 , \ldots , K$ do   
4: s ← s<sub>i</sub>   
5: for all $j , k$ do   
6: $d _ { i j k } \gets \| \mathbf { q } _ { : j _ { k } } - \mathbf { s } _ { : j k } \| _ { 2 }$   
7: $p _ { i j k }  - \frac { d _ { i j k } ^ { 2 } } { \sigma ^ { 2 } + \epsilon }$   
8: $w _ { i j k }$ ← softmax ${ \bf \chi } _ { i } ( p _ { i j k } )$   
9: end for   
10: end for   
11: return $\mathbf { W } \gets \{ w _ { i j k } \}$

## VI. Simulation Results

We train the model using a P100 GPU on the MNIST, Fashion-MNIST, FFHQ, and CelebA datasets with an image size of 28 × 28 for MNIST and Fashion-MNIST, 64 × 64 for FFHQ and 128 × 128 for CelebA. The Adam optimizer is used with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } ~ = ~ 0 . 9 9$ , and a learning rate of $1 \times 1 0 ^ { - 4 }$ . For outer VAE training, we use a combination of $L _ { 1 }$ loss, perceptual loss based on

![](images/ba7b000cf7ee4feff6093b2636b89b2355f727f98e0244e1e689787026bfb23a.jpg)  
Fig. 7. Spatial attention weight maps obtained for the top-3 nearest neighbors of the query on the FFHQ dataset. Each map illustrates the spatial contribution of the corresponding neighbor region to the generated latent representation.

TABLE I  
Training configurations of the baseline difusion models.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MNIST</td><td rowspan=1 colspan=1>FashionMNIST</td><td rowspan=1 colspan=1>FFHQ</td><td rowspan=1 colspan=1>CelebA</td></tr><tr><td rowspan=1 colspan=1>f</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Diffusion steps</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>1000</td></tr><tr><td rowspan=1 colspan=1>Noise schedule</td><td rowspan=1 colspan=1>linear</td><td rowspan=1 colspan=1>linear</td><td rowspan=1 colspan=1>linear</td><td rowspan=1 colspan=1>linear</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>Transformer</td><td rowspan=1 colspan=1>CNN</td></tr><tr><td rowspan=1 colspan=1>#Params</td><td rowspan=1 colspan=1>15.7M</td><td rowspan=1 colspan=1>15.7M</td><td rowspan=1 colspan=1>130M</td><td rowspan=1 colspan=1>47M</td></tr><tr><td rowspan=1 colspan=1>Attention resolution</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>16,8</td></tr><tr><td rowspan=1 colspan=1>#Heads</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Iterations</td><td rowspan=1 colspan=1>30k</td><td rowspan=1 colspan=1>36k</td><td rowspan=1 colspan=1>134k</td><td rowspan=1 colspan=1>51k</td></tr><tr><td rowspan=1 colspan=1>Learning rate</td><td rowspan=1 colspan=1>10−4</td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1>10⁻4</td></tr></table>

VGG network and an adversarial (GAN) loss of StyleGAN discriminator with the following architecture:

$$
L _ { \mathrm { r e c } } = L _ { \mathrm { 1 } } + L _ { \mathrm { p e r c e p t u a l } } + L _ { \mathrm { a d v } }\tag{30}
$$

In our approach, generation is performed by local interpolation which is substantially faster than iterative reverse difusion. The difusion method requires 0.17 second versus the proposed method just 0.02 second to generate an image on the FFHQ dataset, hence a 8.5× speedup.

Example generated images are presented in Fig. 8 for FFHQ, MNIST, and Fashion-MNIST datasets. The first column from left in each dataset is the chosen query, the second is the generated image, and the last three are nearest neighbors, respectively. Generated images are quite natural and innovative with overall similarity to the query image while they receive special features from neighbors.

More generated sample images is shown in Fig. 9. It can be seen that most of the images are natural and of good quality. The higher resolution of 128×128 with CelebA dataset is also examined in Fig. 10.

Baseline difusion model configurations are listed in Table I. A diferent setting is used for each dataset used for comparison. Similar settings of the proposed method is also listed in Table II. Table III compares FID of the proposed method versus the baseline difusion model. While our proposed model has much less parameters, it achieves better FID in MNIST and comparable FID in Fashio MNIST and CelebA, while the performance is lower in FFHQ dataset. Table IV shows an optimization study which shows that lower σ parameter of the attention mechanism will result in better FID metric.

![](images/131d4b23b31960837fcc534bfbc1b142efc013278795b52fe5a7f9dfd5806498.jpg)  
Fig. 8. Qualitative comparison on FFHQ, MNIST, and Fashion-MNIST. From left to right, each row presents the query image, the generated image, and the Top-3 nearest neighbors retrieved from the latent space.

![](images/87e0fe1412a0df0db686253d4a59a9933cf409cf07148dc854dbf531277e8f02.jpg)  
Fig. 9. Generated samples on FFHQ 64 64

![](images/3df050215eec62275d94d1bddd7b681eb5e405faacc868a25052142e95ec704b.jpg)  
Fig. 10. Generated samples on CelebA 128 128

## References

[1] A. Vaswani, et al., “Attention is all you need,” in Proc. Ann. Conf. Neur. Inf. Process. Syst., Long Beach, CA, USA, pp. 5998–6008, 2017.

[2] W. Sun, J. Hu, Y. Zhou, et al., “Speed always wins: A survey on eficient architectures for large language models,” arXiv:2508.09834, 2025.

[3] A. Dosovitskiy, et al. “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929. 2020.

[4] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” In International Conference on Machine Learning, pages 2256–2265, PMLR, 2015.

[5] W. Peebles, and S. Xie, “Scalable difusion models with transformers,” In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195-4205. 2023.

[6] H. V. Poor, An introduction to signal detection and estimation, Springer, 2013.

[7] J. Henderson and F. Fehr, “A VAE for Transformers with Nonparametric Variational Information Bottleneck,” in Proceedings of International Conference on Machine Learning ICLR, 2023.

[8] M. Kamb and S. Ganguli, “An analytic theory of creativity in convolutional difusion models,” In Proceedings of the 42<sup>nd</sup> International Conference on Machine Learning, Vancouver, Canada, 2025.

[9] R. Rombach, et al. “High-resolution image synthesis with latent difusion models,” Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022.

[10] J. Braunsmann, M. Rajkovic , M Rumpf, and B. Wirth,

TABLE II  
Architecture and training configurations of the proposed latent-space models.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MNIST</td><td rowspan=1 colspan=1>FashionMNIST</td><td rowspan=1 colspan=1>FFHQ</td><td rowspan=1 colspan=1>CelebA</td></tr><tr><td rowspan=1 colspan=1>f</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td></tr><tr><td rowspan=1 colspan=1>z-shape</td><td rowspan=1 colspan=1>3 × 7 × 7</td><td rowspan=1 colspan=1> $3 \times 7 \times 7$ </td><td rowspan=1 colspan=1>16 × 8 × 8</td><td rowspan=1 colspan=1>4 × 16 × 16</td></tr><tr><td rowspan=1 colspan=1>Latent Signals</td><td rowspan=1 colspan=1>1024</td><td rowspan=1 colspan=1>1024</td><td rowspan=1 colspan=1>2048</td><td rowspan=1 colspan=1>=</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>CNN</td></tr><tr><td rowspan=1 colspan=1>#Params</td><td rowspan=1 colspan=1>2.95M</td><td rowspan=1 colspan=1>2.95M</td><td rowspan=1 colspan=1>64.8M</td><td rowspan=1 colspan=1>53M</td></tr><tr><td rowspan=1 colspan=1>Attention resolution</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Encoder: 8Decoder: 8, 16</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>#Heads</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Iterations</td><td rowspan=1 colspan=1>13.6k</td><td rowspan=1 colspan=1>31.3k</td><td rowspan=1 colspan=1>44k</td><td rowspan=1 colspan=1>51k</td></tr><tr><td rowspan=1 colspan=1>Learning rate</td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $1 0 ^ { - 4 }$ </td></tr></table>

TABLE III

Quantitative comparison of the baseline and proposed methods in terms of FID.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>FID</td></tr><tr><td rowspan=2 colspan=1>MNIST</td><td rowspan=1 colspan=1>Diffusion</td><td rowspan=1 colspan=1>19.30</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>8.60</td></tr><tr><td rowspan=2 colspan=1>FMNIST</td><td rowspan=1 colspan=1>Diffusion</td><td rowspan=1 colspan=1>13.00</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>14.60</td></tr><tr><td rowspan=2 colspan=1>FFHQ</td><td rowspan=1 colspan=1>Diffusion</td><td rowspan=1 colspan=1>8.60</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>16.58</td></tr><tr><td rowspan=2 colspan=1>CelebA</td><td rowspan=1 colspan=1>Diffusion</td><td rowspan=1 colspan=1>30.49</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>32.15</td></tr></table>

“Convergent autoencoder approximation of low bending and low distortion manifold embeddings,” Mathematical Modelling and Numerical Analysis, vol. 58, pp. 335-361, 2024.

[11] Y. Lee, “A geometric perspective on autoencoders,” available online: https://arxiv.org/abs/2309.08247

[12] J. Song, C. Meng, and S. Ermon. “Denoising difusion implicit models,” arXiv preprint arXiv:2010.02502, 2020

[13] C. Luo, “Understanding difusion models: A unified perspective,” arXiv preprint arXiv:2208.11970, 2022

[14] S Boyd and L. Vandenberghe, Convex optimization, Cambridge Univ. Press, 2004.

[15] A. Katharopoulos, et al., “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention,” in Proceedings of the 37th International Conference on Machine Learning, pp. 5156–5165, 2020.

[16] J. Zhang, R. Su, C. Liu, et al, “A Survey of Eficient Attention Methods: Hardware-eficient, Sparse, Compact, and Linear Attention,” 2025.

[17] R. Sedgewick and K. Wayne, Algorithms, Addison-Wesley, 4’th ed., 2011.

[18] L. Feng, et al. “Tree cross attention,” arXiv preprint, arXiv:2309.17388, 2023.

TABLE IV  
Ablation study of the proposed sampling method under diferent σ and Top-K settings on the FFHQ dataset.
<table><tr><td rowspan=1 colspan=1>σ</td><td rowspan=1 colspan=1>Top-K</td><td rowspan=1 colspan=1>Noise scale</td><td rowspan=1 colspan=1>FID</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>27.43</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>34.33</td></tr><tr><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>16.69</td></tr><tr><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>16.58</td></tr></table>