# Controlled Binary Assignment Flows

Paul Fröhling<sup>1</sup>, Christoph Schnörr<sup>1,2</sup>

<sup>1</sup> Institute for Mathematics, Image and Pattern Analysis Group, Heidelberg University, Germany <sup>2</sup> Research Station Geometry and Dynamics, Heidelberg University

Abstract. This paper introduces a geometric approach to the representation and learning of time-variant parametrizations of deep assignment flows, for binary data labelings on graphs. Each edge carries an optimally controlled parameter flow in a 2-dimensional totally geodesic submanifold, based on objectives derived from a non-local linear scale space representation of the input data. The aggregation of triplets of such flows in 5-dimensional totally geodesic submanifolds is learned from data. The fusion of these flows determines the trajectory in the highdimensional symmetric parameter space and by geometric integration the layers of binary deep assignment flows. Numerical results demonstrate the expressivity and the limitation of our highly parallel geometric local synthesis approach to high-dimensional deep assignment flow parametrizations.

Keywords: image labeling, foreground-background separation, positive definite parameter manifold, totally geodesic submanifolds, discrete optimal control, deep assignment flows, geometric deep learning

## 1 Introduction

Geometric deep learning [4] constitutes an active and promising research direction. The fast growing literature includes works with a focus on manifold-valued data [5,9], on the equivariance properties of mappings realized by deep networks [6,14,8] or on the application-specific sensitivity to spatial structure in terms of geometric control through sub-Riemannian geometry [15,3]. Apparently, less work has been devoted to studying the geometric role of the network parameters themselves for particular classes of deep networks.

This paper takes a step along this direction for the specific problem class of binary data labeling on arbitrary graphs. The basis are assignment flows [2,13] of the form

$$
\dot { S } ( t ) = R _ { S ( t ) } [ \Omega ( t ) S ( t ) ] , \quad S ( 0 ) = S _ { 0 } \in S _ { c } ^ { n } , \quad \mathrm { w h e r e }\tag{1.1a}
$$

$$
S = ( S _ { 1 } , \ldots , S _ { n } ) ^ { \top } \in S _ { c } ^ { n } ,\tag{1.1b}
$$

$$
S _ { c } : = \mathrm { r i n t } ( \Delta _ { c } ) = \left\{ s \in \Delta _ { c } \colon s > 0 \right\} \quad \mathrm { ( r e l . ~ i n t e r i o r : ~ \Delta ~ ) } \mathrm { p r o b . ~ s i m p l e x } )\tag{1.1c}
$$

$$
S _ { c } ^ { n } : = S _ { c } \times \cdots \times S _ { c }
$$

$$
( n \ \mathrm { f a c t o r s } )\tag{1.1d}
$$

$$
( R _ { S } [ \Omega S ] ) _ { i } : = R _ { S _ { i } } ( \Omega S ) _ { i }\tag{1.1e}
$$

$$
: = \mathrm { D i a g } ( S _ { i } ) ( \Omega S ) _ { i } - \langle S _ { i } , ( \Omega S ) _ { i } \rangle S _ { i } , \quad i \in [ n ] ,\tag{row index}
$$

(1.1f)

with time-variant symmetric positive-definite parameter matrix

$$
\Omega ( t ) \in { \mathcal { P } } ^ { n } : = \{ A \in \mathbb { R } ^ { n \times n } : A = A ^ { \top } , ~ A \succ 0 \} , \quad t \geq 0 .\tag{1.1g}
$$

The latter constraint satisfies a suficient condition that the flow (1.1a) converges to integral labelings [20].

![](images/b4b02968ebaef476a9f25302b78222c8dcec417dbc6bf9e8a73e5d38ac067174.jpg)  
Fig. 1.1: Illustration of our geometric local-to-global synthesis approach of high-dimensional parameter trajectories (1.1g) for deep binary assignment flows, via the optimal control of trajectories and their aggregation in totally-geodesic submanifolds $\mathcal { P } _ { 1 } ^ { 2 } , \mathcal { P } _ { 1 } ^ { 3 }$

Figure 1.1 illustrates the first two stages of our 3-stage approach for representing and learning from data the parameter trajectory (1.1g) in the high-dimensional symmetric space $\mathcal P ^ { n }$ : (1) each edge $i j$ of the underlying graph indexes a control parameter trajectory $u ^ { i j } ( t )$ in the 2-dimensional totally geodesic submanifold $\mathcal { P } _ { 1 } ^ { 2 }$ of positive-definite symmetric $2 \times 2$ matrices with determinant 1, which are globally optimized in parallel on all edges through discrete optimal control. (2) Triplets of such flows indexed by triangular subgraphs are aggregated via machine learning to form time-variant $3 \times 3$ parameter matrices in the 5-dimensional totally geodesic submanifold $\mathcal { P } _ { 1 } ^ { 3 }$ . (3) The third ingredient (not shown in Fig. 1.1) is a small finite subset of the nonlocal linear scale space representation of the input data, used to define the objectives of discrete optimal control processes at stage (1).

Altogether, vertices $\mathcal { V } = \{ \dots , i , j , k , \dots \}$ and edges $\mathcal { E } = \{ \dots , i j , i k , j k , \dots \}$ of the underlying graph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } )$ determine a map $u ( t ) \mapsto \Omega ( t ) \in \mathcal P ^ { n }$ and by geometric integration [19] of (1.1a) the resulting deep assignment flow network.

Section 2 specializes the general set-up (1.1) to binary labelings. The ingredients (1), (2), (3) of our approach sketched above are introduced in Section 3. Numerical results illustrate in Section 4 both the expressivity and the limitation of our geometric approach to the parametrization of high-dimensional binary labeling via assignment flows. Our work paves the way for future research indicated in Section 5.

## 2 Binary Assignment Flows

We consider the specific case of two labels $c = 2 .$ , i.e. binary data labeling, and reformulate the general assignment flow equation (1.1) accordingly. Since $\Delta _ { 2 } \cong [ 0 , 1 ]$ , any probability vector $p = ( p _ { 1 } , 1 - p _ { 1 } ) \in \Delta _ { 2 }$ is determined just by its first component, so is componentwise any assignment vector $S \in S _ { 2 } ^ { n }$ . As a result, we have the following.

## Proposition 1 (binary assignment flow equation).

Let $c = 2$ and define the map

$$
s \colon S _ { 2 } ^ { n } \to ( 0 , 1 ) ^ { n } , \qquad s ( S ) = ( s _ { 1 } , \ldots , s _ { n } ) ^ { \top } : = ( S _ { 1 1 } , \ldots , S _ { n 1 } ) ^ { \top } .\tag{2.1}
$$

Then the assignment flow equation (1.1) is equivalent to the system

$$
\begin{array} { r l } & { S ( t ) = \left( \left( \begin{array} { c } { \sigma \big ( 2 v _ { 1 } ( t ) \big ) } \\ { 1 - \sigma \big ( 2 v _ { 1 } ( t ) \big ) } \end{array} \right) , \dots , \left( \begin{array} { c } { \sigma \big ( 2 v _ { n } ( t ) \big ) } \\ { 1 - \sigma \big ( 2 v _ { n } ( t ) \big ) } \end{array} \right) \right) ^ { \top } } \\ & { \qquad = : s ^ { - 1 } \big ( \sigma [ 2 v ( t ) ] \big ) , } \end{array}\tag{2.2a}
$$

(2.2b)

where $\sigma ( x ) = ( 1 { + } e ^ { - x } ) ^ { - 1 }$ is the logistic sigmoidfunction and $\mathbf { \boldsymbol { \upsilon } } ( t ) = \left( v _ { 1 } ( t ) , \ldots , v _ { n } ( t ) \right) ^ { \top }$ ∈ $\mathbb { R } ^ { n }$ solves

$$
\dot { v } = \frac { 1 } { 2 } \Omega \operatorname { t a n h } [ v ] , \qquad v ( 0 ) = v _ { 0 } : = \frac { 1 } { 2 } \log \left[ \frac { ( S _ { 0 } ) _ { , 1 } } { \mathbb { 1 } _ { n } - ( S _ { 0 } ) _ { , 1 } } \right] \in \mathbb { R } ^ { n } .\tag{2.2c}
$$

As a consequence of Proposition 1, it sufices to solve the nonlinear system of ODEs (2.2c) in order to determine the binary label assignment by substitution into (2.2a) followed by trivial rounding.

The binary assignment flow network corresponding to (2.2) is derived by basic geometric integration. Following [19] and using a discrete layer index also denoted by $t \in [ T ] = [ 1 , 2 , \dotsc , T ] , T \in \mathbb { N }$ , the application of the explicit geometric Euler scheme with step sizes $h _ { t }$ yields the

$$
S _ { t + 1 } = s ^ { - 1 } ( \sigma [ 2 v _ { t + 1 } ] ) , ~ ( \mathrm { b i n a r y ~ a s s i g n m e n t ~ f l o w ~ n e t w o r k } ) ~ ( 2 . 3 \mathrm { a } )
$$

$$
v _ { t + 1 } = v _ { t } + h _ { t } \Omega _ { t } \operatorname { t a n h } [ v _ { t } ] , \quad t \in [ T ] , \quad \Omega _ { t } \in \mathcal { P } ^ { n } ,\tag{2.3b}
$$

where the step sizes $h _ { t }$ absorb the factor $\frac { 1 } { 2 }$ from (2.2c). The network input is $v _ { 1 }$ , the network output is $S _ { T + 1 }$ . Inspecting (2.3) reveals the basic network structure:

– each network layer $t \in [ T ]$ comprises a tanh activation function and a manifoldvalued parameter matrix $\Omega _ { t }$ corresponding to (1.1g);

– each layer $t \in [ T + 1 ]$ outputs the state $S _ { t } ,$ including the initial layer with output equal to the input $S _ { 1 }$ encoded by input vector $v _ { 1 }$

We point out that this structure is a consequence of the information geometry underlying the assignment flow equation (1.1), rather than set up beforehand in an ad-hoc manner. In Section 3, we study the problem to learn the parameters $\Omega _ { t } \in \mathcal { P } ^ { n } , ~ t \in [ T ]$ of the network (2.3).

## 3 Approach

This section details our approach illustrated by Figure 1.1. Each of the three stages is described below in a corresponding subsection.

## 3.1 Optimal Parameter Trajectories on Edges

We consider the restriction of the discrete-time flow (2.3) to the edges $i j \in \mathcal { E }$ as minimal basic components of assignment flow networks and call them $2 \times 2$ systems. Each $2 \times 2$

systems couples two discrete binary probability distributions. For each � and edge ��, we impose the constraint

$$
\Omega _ { t } ^ { ( i j ) } \in { \mathcal P } _ { 1 } ^ { 2 } : = \{ A \in { \mathcal P } ^ { 2 } \colon \operatorname* { d e t } ( A ) = 1 \} , \qquad \forall t \in [ T ] , \quad \forall i j \in { \mathcal E } ( { \mathcal G } ) ,\tag{3.1}
$$

in terms of the totally geodesic submanifold $\mathcal { P } _ { 1 } ^ { 2 }$ of the positive definite manifold (1.1g) for $n = 2$ . We take this constraint into account by the tangent space parametrization, utilizing the matrix exponential $\operatorname { e x p } _ { m }$ as global difeomorphism

$$
\mathbb { S } _ { 0 } ^ { 2 } : = T _ { \Omega _ { t } } \mathcal { P } _ { 1 } ^ { 2 } \ni u _ { t } ^ { ( i j ) } = \left( u _ { i } \right) _ { t } \mapsto \left( u _ { i } ^ { \phantom { 2 } } u _ { j } ^ { \phantom { 2 } } \right) _ { t } \xrightarrow { \mathbb { e x p } _ { \mathrm { m } } } \Omega _ { t } ^ { ( i j ) } = \Omega ( u _ { t } ^ { ( i j ) } ) \in \mathcal { P } _ { 1 } ^ { 2 }\tag{3.2a}
$$

$$
\mathbb { S } _ { 0 } ^ { 2 } : = \{ U \in \mathbb { R } ^ { 2 \times 2 } : U ^ { \top } = U , \operatorname { t r } U = 0 \} .\tag{3.2b}
$$

Using the shorthands

$$
\begin{array} { r } { \boldsymbol { u } _ { [ T ] } : = ( \boldsymbol { u } _ { t } ) _ { t \in [ T ] } = ( \boldsymbol { u } _ { 1 } , \ldots , \boldsymbol { u } _ { T } ) , } \end{array}\tag{3.3a}
$$

$$
v _ { [ T + 1 ] } : = ( v _ { t } ) _ { t \in [ T + 1 ] } = ( v _ { 1 } , \ldots , v _ { T + 1 } ) ,\tag{3.3b}
$$

we call optimal a sequence $u _ { [ T ] }$ if it minimizes the objective function

$$
J ( u _ { [ T ] } , v _ { [ T + 1 ] } ) = \sum _ { t \in [ T ] } \ell _ { t + 1 } ( v _ { t } , u _ { t } )\tag{3.4}
$$

given in terms of loss functions $\ell _ { 2 } , \ldots , \ell _ { T + 1 }$ to be specified.

local binary assignment flow

![](images/9a7d7bfcc90bde7c3df41f419eeb2566770826c0b101ee653bc73af7b03ad857.jpg)  
Fig. 3.1: Left: Illustration of the layer mapping and its parametrization of the local binary assignment network (2.3b), for an arbitrary edge $i j \in { \mathcal { E } } ( { \mathcal { G } } )$ of the underlying graph (dropped above from the notation) and the corresponding pairs of input feature vectors and output assignment vectors. The control parameters (3.2) which determine the parameter matrix, determine in turn the flow. We learn local binary assignment flows by determining optimal trajectories of these parameters by discrete optimal control. Right: The feasible set for vectors of the sequences (3.3b). Regions near the origin corresponding to a large uncertainty of local label assignments and hence are discretized more finely, whereas regions farther from the origin are discretized more coarsely.

Determining optimal parameter sequences (3.3a) is accomplished by dynamic programming [11]. Consider the minimization problem (3.4) for any given initial vector

$v _ { 1 } \in \mathbb { R } ^ { 2 }$ and a target vector $v _ { T + 1 }$ corresponding to a prescribed target state. Let the costto-go functions $\mathcal { T } _ { [ t , T ] } , t \in [ T ]$ be defined recursively backward for $t = T , T - 1 , \dots , 1$ by

$$
\mathcal { J } _ { [ T , T ] } ( v _ { T } ) : = \operatorname* { m i n } _ { u _ { T } \in \mathbb { U } ^ { 2 } } \ell _ { T + 1 } ( v _ { T } , u _ { T } )\tag{3.5a}
$$

and the Bellman equation

$$
\mathcal { T } _ { [ t , T ] } ( v _ { t } ) : = \operatorname* { m i n } _ { u _ { t } \in \mathbb { V } ^ { 2 } } \big \{ \ell _ { t + 1 } ( v _ { t } , u _ { t } ) + \mathcal { T } _ { [ t + 1 , T ] } \big ( \Phi ( v _ { t } , u _ { t } ) \big ) \big \} , \quad t = T - 1 , \ldots , 2 , 1 .\tag{3.5b}
$$

on the finite feasible set $\mathbb { V } ^ { 2 } \subset \mathbb { R } ^ { 2 }$ depicted by Fig. 3.1 for each vector $v _ { t } , \ t \in [ T ]$ , and on a finite regular grid $\mathbb { U } ^ { 2 } \subset \mathbb { R } ^ { 2 }$ of feasible control parameter vectors of the sequence (3.3a). The function Φ in (3.5b), defined by

$$
\Phi ( v _ { t } , u _ { t } ) = \mathrm { c l i p } _ { \mathbb { V } } \bigl ( v _ { t } + \Omega ( u _ { t } ) \operatorname { t a n h } ( v _ { t } ) \bigr ) , \qquad \mathrm { c l i p } _ { \mathbb { V } } \colon \mathbb { R } ^ { 2 } \to \mathbb { V } ^ { 2 }\tag{3.6}
$$

maps to the feasible set by rounding. For our implementation, we have chosen the weighted loss functions

$$
\ell _ { t + 1 } ( v _ { t } , u _ { t } ) : = \lambda _ { t } \| \Phi ( v _ { t } , u _ { t } ) - v _ { T + 1 } \| ^ { 2 } , \qquad t \in [ T ] ,\tag{3.7}
$$

where the weights $\lambda _ { t } > 0$ increase with � in order to assign more importance to states closer to the terminal target.

Dynamic Programming solves the optimization problem (3.4) by working recursively backward, starting at the target. This leads to the policy mapping

$$
\pi _ { v _ { T + 1 } ^ { * } } \colon \mathbb { V } ^ { 2 } \times [ T ] \to \mathbb { U } ^ { 2 } , \qquad ( v _ { t } , t ) \mapsto u _ { t } ^ { * }\tag{3.8}
$$

returning an optimal $\boldsymbol { u } _ { t } ^ { * }$ for a current state $( v _ { t } , t )$ and a given target $v _ { T + 1 } . \mathrm { { I f } . }$ for each � ∈ $[ T ]$ , the optimality of $\dot { \boldsymbol { u } } _ { t } ^ { * }$ is achieved in (3.5), then the entire sequence $\boldsymbol { u } _ { [ T ] } ^ { * } = ( \boldsymbol { u } _ { 1 } ^ { * } , \ldots , \boldsymbol { u } _ { T } ^ { * } )$ given by (3.8) is optimal. Using the resulting policy $\pi _ { v _ { T + 1 } ^ { * } }$ , optimal trajectories emanating from any feasible point can be determined for a given target $v _ { T + 1 } ^ { * }$

## 3.2 Learning Flow Aggregation on Triangular Subgraphs

In this section, as sketched by Figure 3.2, we describe the aggregation of $3 \times 3$ flows on triangular subgraphs based on three local $2 \times 2$ flows on the corresponding edges, each determined via dynamic programming as described in Section 3.1.

Geometric Ansatz. We consider any triangle $T _ { i j k }$ and denote the vertices $\{ i , j , k \}$ by {1, 2, 3}, for simplicity. For the edges $i j = 1 2 , i k = 1 3$ and $j k = 2 3$ , we denote the optimal parameter sequences by

$$
u _ { [ T ] } ^ { ( 1 2 ) } , \quad u _ { [ T ] } ^ { ( 1 3 ) } , \quad u _ { [ T ] } ^ { ( 2 3 ) } .\tag{3.9}
$$

The control parameters of the $2 \times 2$ edge flows are embedded into $3 \times 3$ parameter matrices via the map $\iota \colon \mathbb { R } ^ { 2 }  \mathbb { S } _ { 0 } ^ { 3 } .$

$$
\iota ( u _ { t } ^ { ( 1 2 ) } ) = \left( \begin{array} { l l l } { u _ { 1 } } & { u _ { 2 } } & { 0 } \\ { u _ { 2 } } & { - u _ { 1 } } & { 0 } \\ { 0 } & { 0 } & { 0 } \end{array} \right) _ { t } , \quad \iota ( u ^ { ( 1 3 ) } ) = \left( \begin{array} { l l l } { u _ { 1 } } & { 0 } & { u _ { 3 } } \\ { 0 } & { 0 } & { 0 } \\ { u _ { 3 } } & { 0 } & { - u _ { 1 } } \end{array} \right) _ { t } , \quad \iota ( u ^ { ( 2 3 ) } ) = \left( \begin{array} { l l l } { 0 } & { 0 } & { 0 } \\ { 0 } & { u _ { 2 } } & { u _ { 3 } } \\ { 0 } & { u _ { 3 } } & { - u _ { 2 } } \end{array} \right) _ { t }\tag{3.10}
$$

![](images/d476dbf4e3cf0694be3f4829bbbf868d557b6cf306681ce0f4d7057cd13bd3de.jpg)  
Fig. 3.2: Left: A triangle (minimally complete) subgraph. Each edge indexes an optimally controlled binary assignment flow network (2.3b). Fusing these minimal networks into a larger network creates conflicts because each controlled variable participates in two local systems, as indicated by the above right-most panel and the application of the injection map (3.10) to the three sequences (3.9). This problem is resolved by a machine learning component (center panel) with learnable parameters $\theta _ { t } ^ { e } , e \in \{ i j , i k , j k \}$

The image of the parameter space $\mathbb { R } ^ { 2 }$ under the composition of this embedding and the exponential map,

$$
\exp _ { m } \circ \iota ( \mathbb { R } ^ { 2 } ) \subset \mathcal P _ { 1 } ^ { 3 } ,\tag{3.11}
$$

forms a totally geodesic submanifold in the manifold $\mathcal { P } _ { 1 } ^ { 3 }$ , since the linear subspace $\iota ( \mathbb { R } ^ { 2 } )$ forms a Lie triple system [10,16]: evaluating the double-commutator on matrices parametrized by arbitrary parameter vectors $u _ { t } ^ { ( 1 2 ) } , \overline { { v _ { t } ^ { ( 1 2 ) } } } , w _ { t } ^ { ( 1 2 ) } \in \mathbb { R } ^ { 2 }$ yields again a matrix in the submanifold,

$$
\begin{array} { r } { \left[ \iota ( u _ { t } ^ { ( 1 2 ) } ) , [ \iota ( v _ { t } ^ { ( 1 2 ) } ) , \iota ( w _ { t } ^ { ( 1 2 ) } ) ] \right] \in \iota ( \mathbb { R } ^ { 2 } ) , \qquad \forall u _ { t } ^ { ( 1 2 ) } , v _ { t } ^ { ( 1 2 ) } , w _ { t } ^ { ( 1 2 ) } \in \mathbb { R } ^ { 2 } , } \end{array}\tag{3.12}
$$

i.e. there is a vector $z _ { t } ^ { ( 1 2 ) }$ depending on $u _ { t } ^ { ( 1 2 ) } , v _ { t } ^ { ( 1 2 ) } , w _ { t } ^ { ( 1 2 ) }$ such that the double-commutator evaluates to $\iota ( z _ { t } ^ { ( 1 2 ) } )$ .

![](images/27e42d7da0da817aa77bbd45892ddc174c731e98bb36b30cf4a7eff4104bea9a.jpg)  
Fig. 3.3: Performance of locally labeled triangular subgraphs, based on local geometric averaging (3.13) of three $2 \times 2$ flows indexed by the edges. Each point indicates an initial point around 0, mapped to a small sphere for visualization. Its color displays if the trajectory (2.3b) reaches the target octant $( - \cdot - + )$ (green) or makes a single (blue), two (black) or three (red) labeling errors at the vertices. Errors occur when the starting and target octants difer by $\geq 2$ binary labels.

As a consequence and in view of the optimality of the parameter sequences (3.9), it appears natural to define the sequence $\Omega _ { [ T ] } \subset \mathcal { S } _ { 1 } ^ { 3 }$ for the triangular subgraph by averaging the tangent matrices corresponding to the three edges and subsequent exponentiation [1],

$$
\mathcal { P } _ { 1 } ^ { 3 } \supset \Omega _ { [ T ] } \ni \Omega _ { t } : = \exp _ { \mathrm { m } } \Big ( \frac { 1 } { 2 } \big ( \iota ( u _ { t } ^ { ( 1 2 ) } ) + \iota ( u _ { t } ^ { ( 1 3 ) } ) + \iota ( u _ { t } ^ { ( 2 3 ) } ) \big ) \Big ) , \quad t \in [ T ] .\tag{3.13}
$$

For a numerical evaluation of this Ansatz, the starting points $\{ v _ { 1 } \in \mathbb { V } \colon \| v \| _ { \infty } < 0 . 7 \}$ were selected and every vector $v _ { T + 1 }$ in the orthant $( - , - , + )$ was considered as admissible target. For each pair $( v _ { 1 } , v _ { T + 1 } )$ , three optimal edge flows were constructed that achieve the mappings $( v _ { 1 ; 1 } , v _ { 1 ; 2 } ) \to ( - , - ) , ( v _ { 1 ; 1 } , v _ { 1 ; 3 } ) \to ( - , + ) , ( v _ { 1 ; 2 } , v _ { 1 ; 3 } ) \to ( - , + )$ corresponding to the restrictions of the desired $3 \times 3$ flow. Empirically, it turned out that for the resulting set of 1728 test flows, the aggregation approach (3.13) by plain geometric averaging only achieved ∼ 62% correct $3 \times 3$ flows, caused by local conflicts illustrated by Figure 3.2, right panel.

Additional learning component. We refine the Ansatz (3.13) by replacing each vector $u _ { t } ^ { ( i j ) }$ by the vectors

$$
\widetilde { u } _ { t } ^ { ( i j ) } = R ( \theta _ { t } ^ { ( i j ) } ) u _ { t } ^ { ( i j ) } , \qquad R ( \theta _ { t } ^ { ( i j ) } ) = \left( \begin{array} { l } { \cos ( \theta _ { t } ^ { ( i j ) } ) , - \sin ( \theta _ { t } ^ { ( i j ) } ) } \\ { \sin ( \theta _ { t } ^ { ( i j ) } ) , \cos ( \theta _ { t } ^ { ( i j ) } ) } \end{array} \right) \in \mathrm { S O } ( 2 ) ,\tag{3.14}
$$

each adjusted by a planar time-variant rotations with learnable parameter $\theta _ { t } ^ { ( i j ) } \in [ 0 , 2 \pi )$ These parameters are learned by a neural network which realizes the mapping

$$
F _ { \theta } ( \xi ) : = ( \theta _ { t } ^ { ( i j ) } , \theta _ { t } ^ { ( i k ) } , \theta _ { t } ^ { ( j k ) } ) , \xi = \{ \xi _ { 1 } , \xi _ { 2 } , . . . , \xi _ { n } \} .\tag{3.15}
$$

This mapping is used not only to align local edge flows in order to avoid to labeling errors displayed by Figure 3.3, but also to reorient flows, provided the non-local context (cf. Section 3.3) suggests to change the target labeling at some point of time $t \in [ T ]$

The architecture of the neural network consists of two branches (Fig. 3.2, center panel): one for static features and one for time-dependent features. The static branch is a three-layer neural network. The temporal branch consists of an LSTM [18] for processing the � and $v -$ trajectories of the three edge flows of a triangular subgraph.

As input $\xi$ of the network (3.15), we use

– static features: initial pixel positions and values $v _ { 1 }$ per incident triangular subgraph; – temporal features: �-trajectories, �-trajectories for every edge, current time (layer index) $t ,$ current state in $\mathbb { R } ^ { 3 }$ (starting point $v _ { 1 }$ or latest prediction $\widetilde { v } _ { t }$ for $t > 1 )$ .

Minimizing the loss function (�: margin, $\mathcal { T } \backslash$ : set of triangles)

$$
\mathcal { L } ( \xi ) = \frac { 1 } { | \mathcal { T } | } \sum _ { \tau \in \mathcal { T } } \sum _ { i \in \tau } \log \Big ( 1 + \exp \big ( m - \mathrm { s g n } ( v _ { i ; T + 1 } ^ { * } ) \widetilde { v } _ { i ; T + 1 } ( \xi ) \big ) \Big )\tag{3.16}
$$

with margin parameter � $> 0$ and dependency $\xi \mapsto \widetilde { v } _ { i ; T + 1 } ( \xi )$ given by (2.3b), (3.14) and (3.15), enforces that each aggregated flow for a triangle reaches the prescribed octant. After training, repeating the local labeling of triangular subgraphs (Fig. 3.3) achieves 100% performance.

## 3.3 Context Representation via a Linear Scale Space

We supply the triangular subgraphs with a sequence of initial data corresponding to convolutions with the Gaussian kernel

$$
G ( x , \tau ) = \frac { \tau } { 4 \pi } e ^ { - \frac { \tau } { 4 } \| x \| ^ { 2 } } , \qquad \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { D }\tag{3.17}
$$

and an increasing sequence of parameter values corresponding to a coarse-to-fine transition of spatial scale and structure; see Figure 3.4 for an illustration. The approach from Section 3.2 is applied to each transition $\tau _ { d }  \tau _ { d + 1 }$ . The changing target labels induced by the linear scale space provide information about the spatial context at varying scales, which facilitates both learning and inference.

![](images/732ce6892de2747680bbb6b35be25b8857b03df98218d0d7b59ce563c8b8bb52.jpg)  
Fig. 3.4: Image labeling via a linear scale space, exemplified for a corrupted handwritten digit $\mathbf { \tilde { \Sigma } } ^ { 6 6 } 8 \mathbf { \mathit { \Omega } } ^ { \prime }$ . Top row: Scale-space representations $v _ { 1 } \mapsto G ( \cdot , \tau _ { d } ) * v _ { 1 }$ <sub>,</sub> � $\in \ [ D ]$ , of the input data. Bottom row: In the training phase, analogous representations of the target labeling $v _ { T + 1 } ^ { * } \mapsto G ( \cdot , \tau _ { d } ) * v _ { T + 1 } ^ { * }$ are provided as well. Center rows, inference: The learned sequence of parameter matrices $\Omega _ { t } , t \in [ T ]$ enables the binary assignment flow network (2.3) to infer the uncorrupted labelings (3rd row; green arrows) from the corrupted labelings (2nd row; yellow arrows) across the scale space.

Thus, this process is structured by two nested parameter sequences: The assignment flow time (network layer index) $t \in \{ 1 , . . . , T + 1 \}$ } for each transition $\tau _ { d }  \tau _ { d + 1 }$ and the linear scale space parameters $d \in \{ 1 , . . , D \}$ . Accordingly, the loss (3.16) is extended to $\begin{array} { r } { \mathcal { L } = \sum _ { d = 1 } ^ { D } \mathcal { L } ( \xi _ { d } ) } \end{array}$

## 4 Experiments and Discussion

## 4.1 Data Sets

We used the following data sets illustrated by Figure 4.1.

Rotated MNIST: a publicly available well-known data set [12] which generalizes the original MNIST data set. Data points were additionally corrupted (i) by selecting randomly two rows containing foreground pixels and changing all pixels in these rows to foreground; (ii) by interchanging fore- and background pixels afterwards. Thus the task is to simultaneously invert fore- and background and to correct false labelings.

Dino scene: Images of computer-generated 3D scenes, turned into binary input labelings by thresholding. The task is to infer the foreground regions just from the irregular black-/white regions and the spatial context provided by the entire graph and local-to-global information aggregation, without using explicit shape information as provided e.g. by the silhouettes. This experiment mimics the famous psychophysical Dalmation dog experiment [7, pp.20] which demonstrates the ability of the human visual system to infer object shapes from seemingly unorganized binary spatial patterns.

![](images/09495f6ad41203f472d85bf4203939367df30657d064b77764e497f56321f782.jpg)  
Fig. 4.1: Left: Examples of input images (top) corrupted initial labelings (mid) and target labelings (bottom) from the Rotated MNIST data base. Right: Examples of generating images, initial labelings and target labelings for the Dino scene.

## 4.2 Implementation Details

Network architecture. The network architecture comprises 186.995 trainable parameters. The temporal branch processes the time-dependent features via an LSTM cell with 56 hidden units, followed by a linear projection to 56 dimensions. The static branch computes a 56-dimensional embedding of the input through a multilayer perceptron with hidden dimensions 256 and 512 and SiLU activations. The two embeddings are added elementwise and passed through a 56-dimensional fusion layer and a final linear layer predicting the three ��(2) rotation angles in (3.15). Since inference is done across the linear scale space (Section 3.3), the corresponding scale index � ∈ [�] was used as additional input datum.

Graph structure. The set of edges added to the underlying grid graph in order to form triangular subgraphs, was manually determined for each data set in an ad-hoc way in view of the inference task, as specified by Figure 4.2. Ideally, a suitable structure for the problem class at hand should be a variable and adaptively inferred as well, which is beyond the scope of this paper and an attractive perspective for future work.

![](images/ba4f482591f85f577a915ae4b58250289ae0d9235714c9aa3131f114f90db5f6.jpg)  
Fig. 4.2: Triangular subgraphs. Rotated MNIST (leftmost 2 panels): Vertices associated with a center vertex (marked red, left panel), used to define triangular subgraphs (right panel, diferent vertex embedding for visualization). Dino scene (rightmost 2 panels): Vertices associated with a center vertex (marked red, left panel), used to define triangular subgraphs (right panel, diferent vertex embedding for visualization). For both scenes, every vertex becomes a center vertex (red) and thus defines a local collection of triangular subgraphs.

Training. The model for Rotated MNIST was trained on 500 images (50 per digit class) of size $2 8 \times 2 8 .$ , with 2.240 triangles complementing the corresponding grid graph (cf. Fig. 4.2). We used three difusion steps. As for the Dino scene, 140 images downsampled to $6 4 \times 6 4$ were used for training, with 61.752 triangles complementing the underlying grid graph. We used four difusion steps. The computation was done on a desktop computer with a NVIDIA GeForce RTX 5070 Ti ofering 16GB of VRAM.

The training time for the Rotated MNIST model was 8:58min for 45 epochs (all images are processed within one epoch). Training the model on the Dino dataset took 29:30min for 70 epochs. Applying the Rotated MNIST model to a test image takes 0.02 sec, whereas the model trained for the Dino scene needs 0.1 sec for inference.

<table><tr><td colspan="3">Rotated MNIST</td></tr><tr><td>Diff. Stage</td><td>Foreground Acc.</td><td>Background Acc.</td></tr><tr><td>0</td><td>97.26 %</td><td>98.83 %</td></tr><tr><td>1</td><td>96.67 %</td><td>98.62 %</td></tr><tr><td>2</td><td>92.83 %</td><td>98.34 %</td></tr></table>

<table><tr><td colspan="3">Cow/Dino</td></tr><tr><td>Diff. Stage</td><td>Foreground Acc.</td><td>Background Acc.</td></tr><tr><td>0</td><td>97.35 %</td><td>99.90 %</td></tr><tr><td>1</td><td>97.32 %</td><td>99.93 %</td></tr><tr><td>2</td><td>94.02 %</td><td>99.80 %</td></tr><tr><td>3</td><td>91.32 %</td><td>99.59 %</td></tr></table>

Table 1: Prediction performance for independent test data for Rotated MNIST (left) and the Dino scene (right). Recall from Fig. 3.4 that during inference, the sequence of difusion stages is $2  1  0$ . The accuracy measurements have been averaged over the entire test data set.

Testing. We computed the minimum Hamming distance between each predicted test labeling �<sup>test</sup> and both the ground truth test labeling and the closest labeling in the training set $\mathcal { D } _ { \mathrm { t r a i n } } .$ . For Rotated MNIST, the predicted test labelings difer from the closest training labelings on average at 10.25% of the vertices, and by 2.74% for the Dino data set. This shows that the model generalizes to previously unseen data rather than merely reproducing labelings from the training set. Table 1 reports the prediction performance for the independent test sets.

![](images/c2c0edd98980beab1c592ab89d53f5571e1b0886bff38573eade7f0a46b2e7b8.jpg)  
Fig. 4.3: Predicted labelings for independent test data for Rotated MNIST (left) and the Dino scene (right). Pink pixels indicate vertices with wrong label assignments.

## 5 Conclusion and Outlook

We introduced controlled binary assignment flows based on geometrically fusing optimal parameter trajectories in low-dimensional totally geodesic submanifolds. Our strategy of using cyclic subgraphs for representing high-dimensional point in the symmetric space (1.1g) complements established strategies using acyclic subgraphs, or chordal subgraphs with low tree-width [17], which may be of independent interest.

Regarding future work, the following topics are of particular interest.

– Adaptive triangular subgraphs. Ideally, the two subgraphical structures depicted by Figure 4.2 should be learned from data, to enhance the adaptivity to diferent scenarios and to further improve prediction performance.

– Adaptive scale space representation. Providing non-local contextual information eases both learning and inference and enables to reduce the number of parameters. Using adaptive scale selection and possibly non-linear scale spaces might be beneficial in this respect.

– Using symmetry. The scenario ‘binary labeling’ and our approach to using a symmetric space entails symmetries of the parameter representation (cf. Figure 3.3), which have not yet been fully explored. Corresponding reductions might enable to consider large-scale scenarios eficiently regarding memory requirements and computation time.

– Non-binary labelings. The non-binary case requires various generalizations. Yet, a better understanding of the preceding points should pave the way for tackling arbitrary labeling scenarios in a similar way.

Acknowledgements. This work is funded by the Deutsche Forschungsgemeinschaft (DFG) under Germany’s Excellence Strategy EXC-2181/1 - 390900948 (the Heidelberg STRUCTURES Excellence Cluster). This work is funded by the Deutsche Forschungsgemeinschaft (DFG), grant SCHN 457/17-2, within the priority programme SPP 2298: Theoretical Foundations of Deep Learning.

## References

1. Arsigny, V., Fillard, P., Pennec, X., Ayache, N.: Geometric means in a novel vector space structure on symmetric positive-definite matrices. SIAM Journal on Matrix Analysis and Applications 29(1), 328–347 (2007)

2. Åström, F., Petra, S., Schmitzer, B., Schnörr, C.: Image Labeling by Assignment. Journal of Mathematical Imaging and Vision 58(2), 211–238 (2017)

3. Bellard, G., Bon, D.L.J., Pai, G., Smets, B.M.N., Duits, R.: Analysis of (sub-)Riemannian PDE-G-CNNs. J. Math. Imaging Vision 65, 819–843 (2023)

4. Bronstein, M.M., Bruna, J., Cohen, T., Veličković, P.: Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv 2104.13478 (2021)

5. Bronstein, M.M., Bruna, J., LeCun, Y., Szlam, A., Vandergheynst, P.: Geometric Deep Learn ing: Going beyond Euclidean data. IEEE Signal Processing Magazine 34(4), 18–42 (2017)

6. Cohen, T., Welling, M.: Group Equivariant Convolutional Networks. In: ICML (2016)

7. Frisby, J.: Seeing: Illusion, Brain and Mind. Oxford University Press (1980)

8. Gerken, J.E., Aronsson, J., Carlsson, O., Linander, H., Ohlsson, F., Petersson, C., Persson, D.: Geometric Deep Learning and Equivariant Neural Networks. Artificial Intelligence Review 56, 14605–14662 (2023)

9. Hanik, M., Steidl, G., von Tycowicz, C.: Manifold GCN: Difusion-based Convolutional Neural Network for Manifold-valued Graphs. International Journal of Computer Vision 134(315) (2026)

10. Helgason, S.: Diferential Geometry, Lie Groups, and Symmetric Spaces. Academic Press (1978)

11. Hernández-Lerma, O., Laura-Guarachi, L., Mendoza-Palacios, S., González-Sánchez, D.: An Introduction to Optimal Control Theory: The Dynamic Programming Approach, Texts in Applied Mathematics, vol. 76. Springer, Cham (2023)

12. Larochelle, H., Erhan, D., Courville, A., Bergstra, J., Bengio, Y.: An empirical evaluation of deep architectures on problems with many factors of variation. In: ICML (2007)

13. Savarino, F., Schnörr, C.: Continuous-Domain Assignment Flows. Europ. J. Appl. Math. 32(3), 570–597 (2021)

14. Smets, B.M.N., Portegies, J., Bekkers, E.J., Duits, R.: PDE-Based Group Equivariant Convolutional Neural Networks. J. Math. Imaging Vision 65, 209–239 (2023)

15. Sommer, S., Svane, A.: Modelling anisotropic covariance using stochastic development and sub-Riemannian frame bundle geometry. J. Geom. Mechanics 9(3), 391–410 (2017)

16. Tumpach, A.B., Larotonda, G.: Totally Geodesic Submanifolds in the Manifold SPD of Symmetric Positive-Definite Real Matrices. Information Geometry 7, 913–942 (2024)

17. Vandenberghe, L., Andersen, M.S.: Chordal Graphs and Semidefinite Optimization. Found. Trends Optimiz. 1(4), 241–433 (2014)

18. Yu, Y., Si, X., Hu, C., Zhang, J.: A Review of Recurrent Neural Networks: LSTM Cells and Network Architectures. Neural Computation 31, 1235–1270 (2019)

19. Zeilmann, A., Savarino, F., Petra, S., Schnörr, C.: Geometric Numerical Integration of the Assignment Flow. Inverse Problems 36(3), 034004 (33pp) (2020)

20. Zern, A., Zeilmann, A., Schnörr, C.: Assignment Flows for Data Labeling on Graphs: Convergence and Stability. Information Geometry 5, 355–404 (2022)