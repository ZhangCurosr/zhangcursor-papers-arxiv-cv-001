# DECOY AND DISCLOSURE RADII OF INVARIANT SHAPE DESCRIPTORS

T. SHASKA AND L. BESHAJ

Abstract. A recognizer that compares rotation-invariant descriptors sees a surface only up to the fiber of the descriptor. We measure this fiber by its radius in the orbit distance from the enrolled surface. A large radius admits decoys, that is, distant shapes that pass the matcher. A small radius discloses the enrolled shape to anyone who captures the stored value. For star-shaped surfaces truncated to spherical harmonics of degree at most $L ,$ with n coeficients, a descriptor of generic rank r has generic fibers of dimension $n - 3 - r$ modulo rotations. The standard pool of band powers, even bispectra, and three invariants of the degree-three band therefore admits decoy families of dimension 5, 13, 20 at $L = 4 , 6 , 8$ . Its rank first reaches $n - 3$ at $L = 1 6 .$ , and a mirror decoy remains at every L. The odd bispectra remove the mirror decoy generically for $L \geq 4 .$ Yet at fixed mean radius the same pool determines the enclosed volume exactly, and it does not determine whether a surface meets a clearance requirement. We certify two cases by exact and interval arithmetic. At $L = 6 \mathrm { ~ a ~ }$ decoy matches all 32 invariants to relative precision $2 \cdot 1 0 ^ { - 1 8 }$ at orbit distance at least 0.87 times the norm of the enrolled tuple. For the radar shape model of asteroid (101955) Bennu, the pool recovers the modeled volume, misses the handedness, and leaves the keep-out radius uncertain by more than 7 m.

## 1. Introduction

Consider a recognizer that identifies three-dimensional objects from their geometry and must ignore the pose of the object. A standard way to do this is to replace the surface by a descriptor that is invariant under rotations, and to compare the descriptor with an enrolled library. The recognizer then sees only the descriptor. Two surfaces with the same descriptor are indistinguishable from that representation alone. The set of such surfaces is the fiber of the descriptor, and we measure it modulo rotations. We analyze this fiber from two adversarial perspectives: decoy construction and disclosure from a released descriptor.

The first adversary builds decoys. A decoy is a surface that is not a rotated copy of an enrolled object but has the same descriptor, up to the tolerance of the recognizer. At zero tolerance a decoy lies in the exact fiber; at positive tolerance it lies in a thickened fiber. Large fibers therefore permit geometrically distinct accepted shapes. A fiber of positive dimension modulo rotations is a whole family of decoys; this ambiguity is intrinsic to the representation and exists independently of classifier training. Every classifier that acts on the descriptor is constant on its fibers, so no training removes them. The second adversary obtains the stored descriptor of an enrolled object. A captured descriptor is consistent with every surface in its fiber, so small fibers disclose the enrolled geometry. We measure both threats by one number, the radius of the fiber in the orbit distance, centered at the enrolled surface. For a fixed admissible set and tolerance, this radius, centered at the enrolled point, has a decoy interpretation and a disclosure interpretation. An observer who knows only the released value faces a diferent uncertainty, which is governed by the orbit diameter, or by the corresponding minimax radius, of the set of surfaces consistent with that value. The diameter lies between the radius and twice the radius (Prop. 3.3). This statement concerns the shape as a whole. It does not address the accuracy of a particular task, and it is not a statistical notion of privacy.

The setting is that of [15, 16]. A star-shaped surface is given by its radial function on the sphere, truncated to spherical harmonics of degree at most $L .$ Rotations act band by band on the coeficients, and descriptors are polynomial invariants of the coeficient tuple. A descriptor that separates orbits admits no exact decoys; conversely, an exact released value determines the enrolled orbit. On a compact semialgebraic admissible set, the radius of any descriptor at tolerance ε exceeds its exact radius by at most $C \varepsilon ^ { \alpha }$ , with constants that may depend on the enrolled tuple. The stability inequality of [16] bounds the orbit distance by a H¨older power of the descriptor discrepancy. It was proved there to certify the descriptor. Read as a disclosure statement, it says that a released descriptor known to accuracy ε locates the enrolled surface within orbit distance $C \varepsilon ^ { \alpha }$ . On the dipole and quadrupole bands the separation proof of [16] is constructive and yields an explicit reconstruction.

A descriptor that does not separate admits decoys, and the constant-rank theorem measures how many. Let the descriptor have generic rank r on a coeficient space of dimension n. Modulo rotations, the part of its fiber in the free maximal-rank locus is a manifold of dimension $n - 3 - r$ If $r = n - 3$ , that regular part is a finite union of orbits. If $r < n - 3 ,$ , the recognizer accepts a family of decoys of positive dimension near every generic enrolled surface. A descriptor with fewer than $n - 3$ components is always in the second case, whatever its components are.

The standard spectral pool consists of the band powers of [10], the even bispectra of [9], and three invariants of the band-three sextic. It is the invariant block used in [15]. We compute its generic rank exactly. For $2 \le L \le 1 5$ the rank equals the size of the pool, which is less than $n - 3$ . At the cutofs $L = 4 , 6 , 8$ the decoy families have dimension 5, 13, and 20. At $L = 1 6$ the rank reaches $n - 3$ for the first time, and the continuous ambiguity disappears on the free maximal-rank locus. A discrete ambiguity remains. The pool consists of true scalars, so a surface and its mirror image have the same pool. For every surface that is not a rotated copy of its mirror image, the mirror image is an exact decoy, at every $L ,$ and at $L = 1 6$ the finite ambiguity still contains the mirror pair. The odd bispectra are pseudoscalars. For $L \geq 4 .$ adjoining them removes the mirror decoy for generic surfaces; the augmented pool first saturates at $L = 1 0$ . The latter threshold also follows from the degree-three orbit-recovery theorem of $[ 7 ]$

A certified $L = 6$ benchmark complements the dimension count: an integer tuple and a rational tuple agree in all 32 invariants to relative precision $2 \cdot 1 0 ^ { - 1 8 }$ and have orbit distance at least 0.87 times the enrolled tuple norm. The rational comparison is exact, and the distance bounds are established by interval arithmetic through a validated covering criterion given in Sec. 7.

On the quadrupole band, the decoy radius of the power can be computed exactly. The fiber of the quadrupole power $\| Q \| ^ { 2 }$ is a sphere. The Hofman–Wielandt inequality of [8] turns the orbit distance on it into the distance between sorted spectra. The decoy radius at Q is $2 r \sin ( \operatorname* { m a x } \{ \theta , \pi / 3 - \theta \} / 2 )$ , where $r = \| Q \|$ and cos $3 \theta = 3 \sqrt { 6 }$ det $Q / r ^ { 3 }$ . It lies between ${ \frac { { \sqrt { 6 } } - { \sqrt { 2 } } } { 2 } } r$ and $r ,$ and adjoining the cubic invariant det $Q$ makes it zero. With the power known exactly, the radius of $( \Vert Q \Vert ^ { 2 }$ , det $Q )$ at determinant tolerance ε grows linearly at matrices with distinct eigenvalues and like $\varepsilon ^ { 1 / 2 }$ at matrices with a repeated eigenvalue. On the quadrupole band, the passage from the power spectrum to the bispectrum is the passage from decoys to disclosure.

Whole-shape ambiguity does not settle task utility. We therefore introduce task intervals and prove an exact attribute-disclosure result: at a fixed constant radial term, the even pool determines enclosed volume. An explicit pair of positive radial surfaces has identical descriptors and volumes but opposite outcomes for a specified clearance requirement. The example can be checked by hand, so the conflict is exhibited for a stated task and is not merely inferred from a diference in shape. In Sec. 11 the same analysis is applied to the radar shape model of the near-Earth asteroid (101955) Bennu, with certified descriptor, distance, positivity, and task bounds for the rounded $L = 6$ model.

The coeficient model assumes a fully observed closed surface that is star-shaped about a specified origin. A centroid convention is an additional constraint. It follows neither from positivity nor from omitting the constant harmonic, and Sec. 10.1 states it explicitly. Surfaces of higher genus and partial views are outside this model. The discussion of applications in Sec. 10.4 proposes a validation setting; it makes no claim about any deployed system.

## 2. Descriptors, fibers, and radii

Let $\mathbb { R } ^ { 3 }$ carry the Euclidean inner product $\langle x , y \rangle = x _ { 1 } y _ { 1 } + x _ { 2 } y _ { 2 } + x _ { 3 } y _ { 3 }$ and the norm $| x | =$ $\langle x , x \rangle ^ { 1 / 2 }$ . Let $S ^ { 2 } = \{ x \in \mathbb { R } ^ { 3 } : | x | = 1 \}$ be the unit sphere, and let the special orthogonal group $\mathrm { S O ( 3 ) }$ be the group of real $3 \times 3$ matrices R with $R ^ { \top } R = I$ and det $R = 1$ , where I is the identity matrix. Every $R \in \mathrm { S O ( 3 ) }$ maps $S ^ { 2 }$ onto $S ^ { 2 }$

Let $l \geq 1$ be an integer. A polynomial $P \in \mathbb { R } [ x _ { 1 } , x _ { 2 } , x _ { 3 } ]$ is harmonic if $\Delta P = 0$ , where $\Delta = \partial _ { 1 } ^ { 2 } + \partial _ { 2 } ^ { 2 } + \partial _ { 3 } ^ { 2 }$ . Let $\mathcal { H } _ { l }$ be the real vector space of the functions $S ^ { 2 } \to \mathbb { R }$ that are restrictions of homogeneous harmonic polynomials of degree $l .$ Its elements are the real spherical harmonics of degree l. The space $\mathcal { H } _ { l }$ has dimension $2 l + 1 ;$ see [16, Sec. 2.1] for further details.

Let $R \in \mathrm { S O ( 3 ) }$ and $f \in \mathcal { H } _ { l }$ . Define $R \cdot f \colon S ^ { 2 } \to \mathbb { R }$ by

$$
( R \cdot f ) ( p ) = f ( R ^ { - 1 } p ) , \qquad p \in S ^ { 2 } ,
$$

where $R ^ { - 1 } p$ is the product of the matrix $R ^ { - 1 } = R ^ { \top }$ with the vector $p .$ Let P be a homogeneous harmonic polynomial of degree l whose restriction to $S ^ { 2 }$ is $f .$ . Then $R \cdot f$ is the restriction of $P \circ R ^ { - 1 }$ . This polynomial is homogeneous of degree $l ,$ and it is harmonic because $\Delta$ commutes with orthogonal changes of variables. Hence $R \cdot f \in { \mathcal { H } } _ { l }$

The group SO(3) acts on $\mathcal { H } _ { l }$ via

$$
\begin{array} { c } { \mathrm { S O ( 3 ) } \times \mathcal { H } _ { l } \to \mathcal { H } _ { l } , } \\ { ( R , f ) \mapsto R \cdot f . } \end{array}
$$

Then $I \cdot f = f$ and $( R _ { 1 } R _ { 2 } ) \cdot f = R _ { 1 } \cdot ( R _ { 2 } \cdot f )$ for all $R _ { 1 } , R _ { 2 } \in \mathrm { S O } ( 3 )$ and $f \in \mathcal { H } _ { l }$ , and the action is linear in $f .$ . Hence the map

$$
\rho _ { l } \colon \mathrm { S O } ( 3 )  \mathrm { G L } ( \mathcal { H } _ { l } ) ,
$$

defined by $\rho _ { l } ( R ) ( f ) = R \cdot f$ , is a group homomorphism, that is, a representation of $\mathrm { S O ( 3 ) }$ on $\mathcal { H } _ { l }$ By [16, Thm. 3.8], the complexification $\mathcal { H } _ { l } \otimes \mathbb { C }$ is equivariantly isomorphic to the space of binary forms of degree 2l, which is an irreducible $\mathrm { S L _ { 2 } ( \mathbb { C } ) }$ -module. The group $\mathrm { S O ( 3 ) }$ acts on it through the covering $\mathrm { S U } ( 2 )  \mathrm { S O } ( 3 )$ , and SU(2) is Zariski dense in $\mathrm { S L _ { 2 } ( \mathbb { C } ) }$ , so a subspace invariant under $\mathrm { S O ( 3 ) }$ is invariant under $\mathrm { S L _ { 2 } ( \mathbb { C } ) }$ . Hence $\mathcal { H } _ { l } \otimes \mathbb { C }$ is irreducible, and every $\mathrm { S O ( 3 ) }$ -equivariant linear map $\mathcal { H } _ { l } \to \mathcal { H } _ { l }$ is a real multiple of the identity. For ${ \mathit { l } } \neq { \mathit { l } } ^ { \prime }$ the representations on $\mathcal { H } _ { l }$ and $\mathcal { H } _ { l ^ { \prime } }$ are not isomorphic, because their dimensions $2 l + 1$ and $2 l ^ { \prime } + 1$ difer.

Remark 2.1 (Julia reduction). Let $l \geq 1$ and $f \in \mathcal { H } _ { l }$ . Let $F _ { f }$ be the binary form of degree 2l that corresponds to $f$ under the isomorphism above. Assume that $F _ { f }$ has $2 l$ distinct roots. Let $J _ { F _ { f } }$ be the Julia quadratic of $F _ { f }$ , and let $\xi ( J _ { F _ { f } } ) \in \mathbb { H } ^ { 3 }$ be its Julia point, as defined in [4, Sec. 4.1] and [3]. Then $\xi ( J _ { F _ { f } } ) = j$ , where $j = ( 0 , 1 )$

Indeed, put $\sigma ( z _ { 0 } , z _ { 1 } ) = ( - \bar { z } _ { 1 } , \bar { z } _ { 0 } )$ . The Veronese map v of [16] satisfies $v ( \sigma ( z ) ) = - \overline { { v ( z ) } }$ Since f is real and homogeneous of degree l, we have $F _ { f } ( \sigma ( z ) ) = ( - 1 ) ^ { l } \overline { { F _ { f } ( z ) } }$ . Hence the set of roots of $F _ { f }$ is invariant under $\alpha \mapsto - 1 / \bar { \alpha }$ . The Julia point is $\mathrm { S L _ { 2 } ( \mathbb { C } ) }$ -equivariant by [4, Thm. 3.4, Lem. 4.4]. The group ${ \mathrm { S U } } ( 2 )$ fixes $j ,$ so after a rotation we may assume that no root of $F _ { f }$ is at infinity. By [4, Thm. 4.1], the Julia point is determined by the roots, and conjugating the roots replaces $( z , t )$ by $( \bar { z } , t )$ . The map $\alpha \mapsto - 1 / \bar { \alpha }$ extends to the isometry

$$
( z , t ) \mapsto \left( { \frac { - z } { | z | ^ { 2 } + t ^ { 2 } } } , { \frac { t } { | z | ^ { 2 } + t ^ { 2 } } } \right)
$$

of $\mathbb { H } ^ { 3 }$ . Its only fixed point is $j .$ . The stabilizer of $j$ in $\mathrm { P S L _ { 2 } ( C ) }$ is $\mathrm { P S U } ( 2 ) \cong \mathrm { S O } ( 3 )$ . Hence the reduction theory of binary forms in [3, 4] gives no normalization of pose on a single real band.

Let $\mathcal { H } _ { 0 }$ be the space of constant functions on $S ^ { 2 }$ . Let $d \omega$ be the surface measure on $S ^ { 2 }$ and $d \mu = d \omega / ( 4 \pi )$ the round probability measure. For real continuous functions $f , g$ on $S ^ { 2 }$ put

$$
\langle f , g \rangle = \int _ { S ^ { 2 } } f g d \mu .
$$

The angle brackets thus denote the Euclidean product for vectors and this product for functions; the arguments determine which is meant. The measure $\mu$ is invariant under $\mathrm { S O ( 3 ) }$ , so $\langle R \cdot f , R$ $g \rangle = \langle f , g \rangle$ for every $R \in \mathrm { S O ( 3 ) }$ . The spaces $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 } , \mathcal { H } _ { 2 } , . . .$ . are pairwise orthogonal for this product; see [16, Sec. 2.1]. We call $\mathcal { H } _ { l }$ the band of degree l.

A closed surface $\Sigma \subset \mathbb { R } ^ { 3 } \setminus \{ 0 \}$ is star-shaped about the origin if every ray $\{ t p : t > 0 \}$ with $p \in S ^ { 2 }$ meets Σ in exactly one point. Let Σ be such a surface, and for $p \in S ^ { 2 }$ let $\rho ( p ) > 0$ be the unique number with $\rho ( p ) p \in \Sigma$ . The function $\rho \colon S ^ { 2 } \to$ R is the radial function of $\Sigma .$ , and $\Sigma = \{ \rho ( p ) p : p \in S ^ { 2 } \}$ . We assume that $\rho$ is continuous. Conversely, every positive continuous function $\rho \colon S ^ { 2 } \to$ R is the radial function of the star-shaped surface $\{ \rho ( p ) p : p \in S ^ { 2 } \}$ . This surface is the boundary of the solid $\{ t p : p \in S ^ { 2 } , \ 0 \leq t \leq \rho ( p ) \}$ , which contains the segment from the origin to each of its points.

Let $L \geq 1$ . For $0 \leq l \leq L$ let $f _ { l }$ be the orthogonal projection of $\rho$ onto $\mathcal { H } _ { l }$ . The truncation of $\rho$ at degree L is

$$
\rho _ { L } = f _ { 0 } + f _ { 1 } + \cdot \cdot \cdot + f _ { L } .
$$

The band $\mathcal { H } _ { \mathrm { 0 } }$ is omitted from the coeficient space below because $\mathrm { S O ( 3 ) }$ acts trivially on it. Put

$$
U _ { L } = \bigoplus _ { l = 1 } ^ { L } \mathcal { H } _ { l } , \qquad n = \dim U _ { L } = \sum _ { l = 1 } ^ { L } ( 2 l + 1 ) = L ^ { 2 } + 2 L .
$$

We write $c = ( f _ { 1 } , \ldots , f _ { L } )$ for a point of $U _ { L }$ . The point c built from the projections of $\rho$ is the coeficient tuple of $\Sigma$ at degree L. The group $\mathrm { S O ( 3 ) }$ acts on $U _ { L }$ diagonally by

$$
R \cdot c = ( R \cdot f _ { 1 } , \ldots , R \cdot f _ { L } ) .
$$

Rotating the surface by $R \in \mathrm { S O ( 3 ) }$ replaces $\rho$ by $R \cdot \rho ,$ where $( R \cdot \rho ) ( p ) = \rho ( R ^ { - 1 } p )$ . This action preserves each $\mathcal { H } _ { l }$ and the product $\langle \cdot , \cdot \rangle$ , so it commutes with the orthogonal projections. Hence the rotation replaces each $f _ { l }$ by $R \cdot f _ { l } ,$ , and the coeficient tuple c by $R \cdot c .$ . When comparing physical radial solids, the constant $f _ { 0 }$ must be fixed or retained separately; throughout the volume and clearance statements it is a known constant $\rho _ { 0 }$

Let $c \in U _ { L }$ . The orbit of c is the set $\mathrm { S O ( 3 ) } \cdot c = \{ R \cdot c : R \in \mathrm { S O ( 3 ) } \}$ . Two points of $U _ { L }$ lie in the same orbit if and only if one is obtained from the other by a rotation. The orbits partition $U _ { L }$ , and the set of orbits is denoted $U _ { L } / \mathrm { S O ( 3 ) }$

An inner product on $U _ { L }$ is $\mathrm { S O ( 3 ) }$ -invariant if $\langle R \cdot c , R \cdot c ^ { \prime } \rangle = \langle c , c ^ { \prime } \rangle$ for all $R \in \mathrm { S O ( 3 ) }$ and $c , c ^ { \prime } \in U _ { L }$ . The bands are irreducible, their complexifications are irreducible, and distinct bands are not isomorphic. By Schur’s lemma, every $\mathrm { S O ( 3 ) }$ -invariant inner product on $U _ { L }$ therefore has the form

$$
\langle c , c ^ { \prime } \rangle = \sum _ { l = 1 } ^ { L } w _ { l } \langle f _ { l } , f _ { l } ^ { \prime } \rangle ,
$$

where $\langle \cdot , \cdot \rangle$ on $\mathcal { H } _ { l }$ is the product defined above and $w _ { l } > 0$ . We fix such an inner product and its norm ∥·∥. Let $c , c ^ { \prime } \in U _ { L }$ . The function $R \mapsto \| c - R \cdot c ^ { \prime } \|$ is continuous on the compact group $\mathrm { S O ( 3 ) }$ , so it attains its minimum. The orbit distance is

$$
\delta ( c , c ^ { \prime } ) = \operatorname* { m i n } _ { R \in \mathrm { S O ( 3 ) } } \| c - R \cdot c ^ { \prime } \| .
$$

It is a metric on $U _ { L } / \mathrm { S O ( 3 ) }$ , and $\delta ( c , c ^ { \prime } ) = 0$ if and only if $c ^ { \prime } \in \mathrm { S O } ( 3 ) \cdot c$ . It is continuous and semialgebraic on $U _ { L } \times U _ { L }$ . These statements are [16, Thm. 7.3], whose proof uses only the invariance of the norm and the compactness of $\mathrm { S O ( 3 ) }$

The space $U _ { L }$ is a real vector space of dimension n. A map $\varphi \colon U _ { L } \to \mathbb { R }$ is a polynomial if it is a polynomial in the coordinates of $U _ { L }$ with respect to some basis. A change of basis is linear, so this property does not depend on the basis. A map

$$
\Phi = ( \Phi _ { 1 } , \dots , \Phi _ { N } ) \colon U _ { L } \to \mathbb { R } ^ { N }
$$

is a polynomial map if each component $\Phi _ { j }$ is a polynomial. A map $\Phi \colon U _ { L } \to \mathbb { R } ^ { N }$ is ${ \mathrm { S O } } ( 3 ) .$ invariant if

$$
\Phi ( R \cdot c ) = \Phi ( c )
$$

for every $R \in \mathrm { S O ( 3 ) }$ and every $c \in U _ { L }$ . Equivalently, Φ is constant on every orbit, so it induces a map $U _ { L } / \mathrm { S O } ( 3 ) \to \mathbb { R } ^ { N }$

Definition 2.2. A descriptor is an SO(3)-invariant polynomial map Φ : $U _ { L } \to \mathbb { R } ^ { N }$ . Let $c \in U _ { L }$ The fiber of Φ through c is the set $\Phi ^ { - 1 } ( \Phi ( c ) )$ of all points x of $U _ { L }$ with $\Phi ( x ) = \Phi ( c )$ . The descriptor Φ separates orbits if every fiber of Φ is a single orbit.

Every fiber contains the orbit of each of its points, so it is a union of orbits. The descriptor Φ separates orbits if and only if $\Phi ( c ) = \Phi ( c ^ { \prime } )$ implies $c ^ { \prime } \in \mathrm { S O } ( 3 ) \cdot c$ . We use the Euclidean norm |·| on $\mathbb { R } ^ { N }$

Definition 2.3. Let $\Phi \colon U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor, let $K \subset U _ { L }$ be compact, let $c \in K$ , and let $\varepsilon \geq 0$ . The radius of Φ at c with tolerance ε is

$$
r _ { \Phi } ^ { \varepsilon } ( c ; K ) = \operatorname* { m a x } \{ \delta ( c , c ^ { \prime } ) : c ^ { \prime } \in K , \ | \Phi ( c ^ { \prime } ) - \Phi ( c ) | \leq \varepsilon \} .
$$

We write $r _ { \Phi } ( c ; K ) = r _ { \Phi } ^ { 0 } ( c ; K )$

The maximum exists. The set over which it is taken is closed in K, hence compact, and it contains c. The function δ is continuous. The radius is nondecreasing in ε. By the properties of $\delta , r _ { \Phi } ( c ; K ) = 0$ if and only if the fiber of Φ through c meets K only in $\mathrm { S O } ( 3 ) \cdot c .$

The radius has two interpretations. In the first, an adversary builds decoys against a recognizer. The recognizer holds $\Phi ( c )$ for the coeficient tuple c of an enrolled surface and accepts $c ^ { \prime }$ when

$$
| \Phi ( c ^ { \prime } ) - \Phi ( c ) | \leq \varepsilon .
$$

A decoy for c is an accepted point $c ^ { \prime } \notin \mathrm { S O } ( 3 ) \cdot c .$ . Then $r _ { \Phi } ^ { \varepsilon } ( c ; K )$ is the largest orbit distance from c of a decoy in K, and it is zero if there is none. We call it the decoy radius. In the second interpretation, an adversary has obtained $\Phi ( c )$ and knows that $c \in K$ . Every point of the fiber of Φ through c that lies in K is consistent with this information, and $r _ { \Phi } ( c ; K )$ is the largest orbit distance from c among them. A large radius serves the first adversary. A small radius serves the second. Fig. 1 shows two surfaces whose band powers coincide. Each is a decoy for the other under the descriptor $P = ( P _ { 1 } , P _ { 2 } )$ of Sec. 5.

For comparison across components of diferent degrees, fix a nonsingular matrix $W \in \mathbb { R } ^ { N \times N }$ and use the discrepancy $| W ( \Phi ( c ^ { \prime } ) - \Phi ( c ) )$ |. This is equivalent to replacing Φ by the descriptor

WΦ. It leaves exact fibers and Jacobian ranks unchanged, but in general it changes positivetolerance radii. The matrix must be fixed when the tolerance is specified. In particular, a componentwise relative error corresponds to

$$
W = \mathrm { d i a g } ( 1 / | \Phi _ { 1 } ( c ) | , \dots , 1 / | \Phi _ { N } ( c ) | ) ,
$$

which depends on c and is undefined when a component vanishes; it requires a positive absolute floor $\eta _ { j }$ , that is, the entries $1 / \operatorname* { m a x } \{ | \Phi _ { j } ( c ) | , \eta _ { j } \}$ .

Definition 2.4. Let $\Phi \colon U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor, let $W \in \mathbb { R } ^ { N \times N }$ be nonsingular, let $K \subset U _ { L }$ be compact, let $\boldsymbol { y } \in \mathbb { R } ^ { N }$ , and let $\varepsilon \geq 0$ . The observation-consistent set is

$$
\mathcal { F } _ { \Phi , W } ^ { \varepsilon } ( y ; K ) = \{ x \in K : | W ( \Phi ( x ) - y ) | \leq \varepsilon \} .
$$

When this set is nonempty, its orbit diameter is

$$
D _ { \Phi , W } ^ { \varepsilon } ( y ; K ) = \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { F } _ { \Phi , W } ^ { \varepsilon } ( y ; K ) } \delta ( x , x ^ { \prime } ) .
$$

Let $c \in K$ . If $y = \Phi ( c )$ , the largest distance from c in this set is $r _ { W \Phi } ^ { \varepsilon } ( c ; K )$ . If both

$$
| W ( y - \Phi ( c ) ) | \leq \varepsilon \quad \mathrm { a n d } \quad x \in \mathcal { F } _ { \Phi , W } ^ { \varepsilon } ( y ; K ) ,
$$

then $| W ( \Phi ( x ) - \Phi ( c ) ) | \le 2 \varepsilon$ by the triangle inequality. An empty consistent set signals incompatibility among the observation, the error budget, and $K ;$ it is not a certificate about an enrolled shape. All information statements below are conditional on the specified K. A catalog or a geometric prior can make the part of a fiber that lies in K much smaller than the whole fiber.

## 3. Separating descriptors disclose

Separating descriptors exist on every $U _ { L }$ . Real forms of the generators of the invariant ring separate the $\mathrm { S O ( 3 ) }$ -orbits by [16, Thm. 4.1]. Recovery of an orbit from the values of invariants, estimated from noisy samples, is the subject of [1]. Let I be a separating descriptor and let $K \subset U _ { L }$ be compact and semialgebraic. By [16, Thm. $7 . 3 ( \mathrm { i i i } ) ]$ there are constants $C _ { K } > 0$ and $\alpha _ { K } \in ( 0 , 1 ]$ such that

$$
\delta ( c , c ^ { \prime } ) \leq C _ { K } | I ( c ) - I ( c ^ { \prime } ) | ^ { \alpha _ { K } } , \qquad c , c ^ { \prime } \in K .
$$

The proof is the Lojasiewicz inequality for the continuous semialgebraic functions $\boldsymbol { \delta } ( \boldsymbol { c } , \boldsymbol { c } ^ { \prime } )$ and $| I ( c ) - I ( c ^ { \prime } ) |$ | on $K \times K$ , which have the same zero set. It applies verbatim on $U _ { L }$ with any invariant inner product. In [16] the inequality certifies the descriptor. The same argument bounds the positive-tolerance radius of every descriptor by its exact radius plus a H¨older error term.

Proposition 3.1. Let Φ: $U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor, let $K \subset U _ { L }$ be compact and semialgebraic, and let $c \in K$ . Then there are $C > 0$ and $\alpha \in ( 0 , 1 ]$ such that

$$
r _ { \Phi } ^ { \varepsilon } ( c ; K ) \leq r _ { \Phi } ( c ; K ) + C \varepsilon ^ { \alpha }
$$

for every $\varepsilon \geq 0$

Proof. Let F be the set of $x \in K$ with $\Phi ( x ) = \Phi ( c )$ . It is compact and semialgebraic, and it contains c. The functions $d ( x ) = \mathrm { m i n } _ { y \in F } \| x - y \|$ and $h ( x ) = | \Phi ( x ) - \Phi ( c ) |$ are continuous and semialgebraic on $K .$ , with the same zero set F. The Lojasiewicz inequality in [5] gives $C > 0$ and $\alpha \in ( 0 , 1 ]$ with $d ( x ) \leq C h ( x ) ^ { \alpha }$ on K. Let $c ^ { \prime } \in K$ with $h ( c ^ { \prime } ) \leq \varepsilon ,$ , and choose $x \in F$ with $\| c ^ { \prime } - x \| = d ( c ^ { \prime } )$ . Then

$$
\delta ( c , c ^ { \prime } ) \leq \delta ( c , x ) + \delta ( x , c ^ { \prime } ) \leq r _ { \Phi } ( c ; K ) + \| x - c ^ { \prime } \| \leq r _ { \Phi } ( c ; K ) + C \varepsilon ^ { \alpha } .
$$

Taking the maximum over $c ^ { \prime }$ proves the claim.

The constants depend on $c .$ For a separating descriptor $r _ { \Phi } ( c ; K ) = 0$ , and the inequality of [16] makes the constants uniform on $K .$ . The next proposition gives the corresponding bounds under both interpretations.

Proposition 3.2. Let I : $U _ { L } \to \mathbb { R } ^ { N }$ be a separating descriptor. Let $K \subset U _ { L }$ be compact and semialgebraic, and let $C _ { K }$ and $\alpha _ { K }$ be as above. Then the following hold.

(1) For every $c \in K$ and every $\varepsilon \ge 0 , r _ { I } ^ { \varepsilon } ( c ; K ) \le C _ { K } \varepsilon ^ { \alpha _ { K } }$ . In particular $r _ { I } ( c ; K ) = 0$

(2) Let $c \in K , l e t \varepsilon \geq 0 ,$ , and let $\boldsymbol { y } \in \mathbb { R } ^ { N }$ satisfy $| y - I ( c ) | \leq \varepsilon$ . Then every $c ^ { \prime } \in K$ with $| I ( c ^ { \prime } ) - y | \leq \varepsilon \ s a t i s f i e s \ \delta ( c , c ^ { \prime } ) \leq C _ { K } ( 2 \varepsilon ) ^ { \alpha _ { K } }$

Proof. Let $c ^ { \prime } \in K$ with $| I ( c ^ { \prime } ) - I ( c ) | \leq \varepsilon$ . The displayed inequality gives $\delta ( c , c ^ { \prime } ) \leq C _ { K } \varepsilon ^ { \alpha _ { K } }$ Taking the maximum over c<sup>′</sup> proves (1). For $\varepsilon = 0$ the bound is zero. In (2) the triangle inequality gives $| I ( c ^ { \prime } ) - I ( c ) | \leq 2 \varepsilon$ , and (1) applies with $2 \varepsilon$ □

Part (1) says that a separating descriptor admits no decoys at tolerance zero, and only decoys within $C _ { K } \varepsilon ^ { \alpha _ { K } }$ at tolerance ε. Part (2) gives the disclosure interpretation: if a released value $y$ is known with error $\varepsilon ,$ then the enrolled tuple lies within orbit distance $C _ { K } ( 2 \varepsilon ) ^ { \alpha _ { K } }$ of every point of $K$ whose descriptor lies within ε of $y .$ . That set of points contains $c ,$ and membership in it is decided by evaluating I. The diferential of I annihilates the orbit directions, so its rank never exceeds $n - 3$ on the free locus. Where the Jacobian of I restricted to a slice transverse to the orbit has full rank $n - 3$ on the principal stratum, the exponent can be taken equal to one locally [16, Thm. $7 . 3 ( \mathrm { i v } ) ]$

On $U _ { 2 }$ the disclosure is explicit. Identify $\mathcal { H } _ { 1 }$ with $\mathbb { R } ^ { 3 }$ and $\mathcal { H } _ { 2 }$ with the traceless symmetric $3 \times 3$ matrices by $f _ { 1 } ( x ) = \langle a , x \rangle$ and $f _ { 2 } ( x ) = x ^ { \top } Q x$ . Then $R \cdot ( a , Q ) = ( R a , R Q R ^ { \top } )$ . By [16, Thm. 4.9] the invariants

$$
| a | ^ { 2 } , \quad \mathrm { t r } Q ^ { 2 } , \quad \mathrm { d e t } Q , \quad a ^ { \top } Q a , \quad a ^ { \top } Q ^ { 2 } a , \quad \mathrm { d e t } ( a , Q a , Q ^ { 2 } a )
$$

separate the $\mathrm { S O ( 3 ) }$ -orbits in $U _ { 2 }$ . The proof in [16] is constructive and yields an explicit reconstruction. The eigenvalues of $Q$ are the roots of $t ^ { 3 } - { \textstyle \frac { 1 } { 2 } } ( \mathrm { t r } Q ^ { 2 } ) t - \mathrm { d e t } Q$ . Let $a _ { 1 } , a _ { 2 } , a _ { 3 }$ be the coordinates of a in a positively oriented orthonormal eigenbasis of $Q ,$ with eigenvalues $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ . Then

$$
\operatorname * { d e t } ( a , Q a , Q ^ { 2 } a ) = a _ { 1 } a _ { 2 } a _ { 3 } \prod _ { i < j } ( \lambda _ { j } - \lambda _ { i } ) .
$$

When the eigenvalues are distinct, the squares $a _ { i } ^ { 2 }$ solve the Vandermonde system $\textstyle \sum _ { i } \lambda _ { i } ^ { k } a _ { i } ^ { 2 } =$ $a ^ { \top } Q ^ { k } a$ for $k = 0 , 1 , 2$ , and the displayed identity fixes the sign of $a _ { 1 } a _ { 2 } a _ { 3 }$ . Thus, on the locus where $Q$ has distinct eigenvalues, the six numbers determine $( a , Q )$ up to rotation by one cubic equation and one linear system. The cases of a repeated eigenvalue are treated separately in the proof of [16, Thm. 4.9], and separation holds on all of $U _ { 2 }$

Proposition 3.3 (Reconstruction from a released value). Let $\mathcal { F } = \mathcal { F } _ { \Phi , W } ^ { \varepsilon } ( y ; K )$ be nonempty and put $D = D _ { \Phi , W } ^ { \varepsilon } ( y ; K )$ . Define the deterministic worst-case reconstruction error $b y$

$$
\mathcal { R } ^ { * } ( y ; K ) = \operatorname* { m i n } _ { \widehat { c } \in K } \operatorname* { m a x } _ { x \in \mathcal { F } } \delta ( \widehat { c } , x ) .
$$

Then $D / 2 \le \mathcal { R } ^ { \ast } ( y ; K ) \le D$ . For every $c \in \mathcal { F } , \ i f r _ { c } = \operatorname* { m a x } _ { x \in \mathcal { F } } \delta ( c , x )$ , then $r _ { c } \le D \le 2 r _ { c }$

Proof. Compactness and continuity give all maxima. The function of $\widehat { c }$ being minimized is continuous, since its change is at most $\delta ( \widehat { c } , \widehat { c } ^ { \prime } ) \leq \| \widehat { c } - \widehat { c } ^ { \prime } \|$ , so the minimum also exists. For any $x , x ^ { \prime } \in { \mathcal { F } }$ and ${ \widehat { c } } \in K$ , the triangle inequality gives $\delta ( x , x ^ { \prime } ) \leq \delta ( x , { \widehat { c } } ) + \delta ( { \widehat { c } } , x ^ { \prime } )$ . Taking maxima proves the lower bound. Taking $\widehat { c }$ to be any point of $\mathcal { F }$ proves the upper bound and $r _ { c } \le D$ The same triangle inequality through c gives $D \leq 2 r _ { c } .$ □

This is an information bound. It provides neither a reconstruction algorithm nor a probabilistic privacy guarantee: it assumes no distribution on K and gives no bound on the computational efort needed to find a consistent point. A large shape diameter need not hide a particular attribute; Secs. 8 and 9 make that distinction explicit.

## 4. Fibers of descriptors

Let $L \geq 2 .$ . For $c \in U _ { L }$ let $( a , Q )$ correspond to $( f _ { 1 } , f _ { 2 } )$ as in Sec. 3, and put

$$
\Omega _ { L } = \{ c \in U _ { L } : \operatorname * { d e t } ( a , Q a , Q ^ { 2 } a ) \neq 0 \} .
$$

Lemma 4.1. Let $L \geq 2$ . The set $\Omega _ { L }$ is open, dense, and $\mathrm { S O ( 3 ) }$ -invariant, and $\mathrm { S O ( 3 ) }$ acts freely on it.

Proof. For $R \in \mathrm { S O ( 3 ) }$

$$
\operatorname* { d e t } ( R a , R Q R ^ { \top } R a , R Q ^ { 2 } R ^ { \top } R a ) = \operatorname* { d e t } R \cdot \operatorname* { d e t } ( a , Q a , Q ^ { 2 } a ) = \operatorname* { d e t } ( a , Q a , Q ^ { 2 } a ) .
$$

So $\Omega _ { L }$ is invariant. The polynomial de $; ( a , Q a , Q ^ { 2 } a )$ is not zero: for $a = ( 1 , 1 , 1 )$ and $Q =$ diag $( 1 , 0 , - 1 )$ it equals −2. So $\Omega _ { L }$ is the complement of the zero set of a nonzero polynomial, and it is open and dense. Let $c \in \Omega _ { L }$ , and let $\lambda _ { i }$ and $a _ { i }$ be as in Sec. 3. The identity $\begin{array} { r } { \operatorname* { d e t } ( a , Q a , Q ^ { 2 } a ) = a _ { 1 } a _ { 2 } a _ { 3 } \prod _ { i < j } ( \lambda _ { j } - \lambda _ { i } ) } \end{array}$ shows that the $\lambda _ { i }$ are distinct and that every $a _ { i }$ is nonzero. Let $R \cdot c = c$ . Then $\bar { R } Q R ^ { \top } = Q$ , so R preserves each eigenline of $Q .$ . In the eigenbasis R is diagonal with entries ±1. Since $R a = a$ and every $a _ { i }$ is nonzero, all entries equal 1. Hence $R = I$ □

Theorem 4.2. Let $L \geq 2$ and let Φ: $U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor. Let r be the maximum over $U _ { L }$ of the rank of dΦ, and let $U ^ { \prime } = \{ c \in \Omega _ { L }$ : rank $d \Phi _ { c } = r \}$ . Then the following hold.

(1) The set $U ^ { \prime }$ is open, dense, and $\mathrm { S O } ( 3 )$ -invariant, and $r \leq \mathrm { m i n } \{ N , n - 3 \}$

(2) For every $c \in U ^ { \prime }$ the set $M _ { c } = \Phi ^ { - 1 } ( \Phi ( c ) ) \cap U ^ { \prime }$ is an SO(3)-invariant embedded submanifold $o f U _ { L }$ of dimension $n - \boldsymbol { r } _ { \mathrm { { \scriptsize ~ i } } }$ and ${ M _ { c } } / \mathrm { S O ( 3 ) }$ is a smooth manifold of dimension $n - 3 - r$

(3) $I f r < n - 3$ , then $r _ { \Phi } ( c ; K ) > 0$ for every $c \in U ^ { \prime }$ and every compact neighborhood K $o f c .$

(4) $I f r = n - 3$ , then $M _ { c }$ is a finite union $o f \mathrm { S O ( 3 ) }$ -orbits for every $c \in U ^ { \prime }$

Proof. (1) $\mathrm { I f } \ r = 0$ , then Φ is constant, the set where dΦ has rank r is $U _ { L }$ , and $U ^ { \prime } = \Omega _ { L }$ . Assume $r > 0$ . The Jacobian of Φ has rank at most r everywhere. So the set where it has rank r is the complement of the common zero set of its $r \times r$ minors. These minors are polynomials, and one of them is not identically zero. Hence this set is a nonempty Zariski open subset of $U _ { L } . \mathrm { \textrm { B y } }$ Lem. 4.1 so is $\Omega _ { L }$ . Their intersection $U ^ { \prime }$ is nonempty because $U _ { L }$ is irreducible, so it is open and dense. Invariance of Φ gives $d \Phi _ { R \cdot c } \circ R = d \Phi _ { c }$ . So the rank of dΦ is constant on orbits, and $U ^ { \prime }$ is invariant. Clearly $r \leq N$ . Let $c \in U ^ { \prime }$ . The action is free at $c ,$ so the orbit ${ \mathrm { S O } } ( 3 ) \cdot c$ is an embedded submanifold of dimension 3. The map Φ is constant on it, so its tangent space at c lies in the kernel of $d \Phi _ { c } .$ Hence $r \leq n - 3$

(2) The map Φ has constant rank r on the open set $U ^ { \prime }$ . By the constant-rank level set theorem in [11], $M _ { c }$ is a properly embedded submanifold of $U ^ { \prime }$ of dimension $n - r$ . It is invariant because Φ and $U ^ { \prime }$ are. The compact group $\mathrm { S O ( 3 ) }$ acts smoothly and freely on $M _ { c }$ . By the quotient manifold theorem in $[ 1 1 ] , M _ { c } / \mathrm { S O } ( 3 )$ is a smooth manifold of dimension $n - r - 3$

(3) Let $r < n - 3$ , and let K be a compact neighborhood of c. Choose $\eta > 0$ such that the closed ball of radius η about c lies in K. The intersection O of $M _ { c }$ with the open ball of radius η about c is a nonempty open subset of $M _ { c }$ . It has dimension $n - r > 3$ . The orbit $\mathrm { S O } ( 3 ) \cdot c$ is a submanifold of dimension 3, so it has measure zero in $M _ { c }$ and does not contain O. Choose $c ^ { \prime } \in \mathcal { O }$ outside ${ \mathrm { S O } } ( 3 ) \cdot c .$ Then $c ^ { \prime } \in K , \Phi ( c ^ { \prime } ) = \Phi ( c )$ , and $\delta ( c , c ^ { \prime } ) > 0$ . Hence $r _ { \Phi } ( c ; K ) \ge \delta ( c , c ^ { \prime } ) > 0$

(4) Let $r = n - 3$ . Then $M _ { c }$ is a manifold of dimension 3. Every orbit in $M _ { c }$ is a compact embedded submanifold of dimension 3, so it is open and closed in $M _ { c }$ . Since $\mathrm { S O ( 3 ) }$ is connected, the orbits in $M _ { c }$ are its connected components. The set $U ^ { \prime }$ is semialgebraic: it is the intersection of $\Omega _ { L }$ with the set where the sum of the squares of the $r \times r$ minors is nonzero. So $M _ { c }$ is semialgebraic, and it has finitely many connected components; see [5]. Hence $M _ { c }$ is a finite union of orbits. □

We call $U ^ { \prime }$ the free maximal-rank locus of Φ. For $c \in U ^ { \prime }$ we call $M _ { c }$ the regular fiber of Φ through c.

Corollary 4.3. Let $L \geq 2$ and let $\Phi \colon U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor with $N < n - 3$ . Then Φ does not separate orbits. Moreover, there is an open dense set $U ^ { \prime } \subset U _ { L }$ such that $r _ { \Phi } ( c ; K ) > 0$ for every $c \in U ^ { \prime }$ and every compact neighborhood K of c.

Proof. $\mathrm { B y }$ Thm. 4.2, $r \leq N < n - 3$ . Part (3) of Thm. 4.2 gives the second claim. It also gives a point of a fiber outside the orbit of $c ,$ which proves the first claim. □

Only the number of components of Φ enters the corollary; the components themselves play no role. A recognizer built on fewer than $n - 3$ invariants admits decoys near every generic enrolled surface, at every tolerance.

Corollary 4.4 (Combined releases). Let $L \geq 2$ , let $\Phi _ { 1 } , \ldots , \Phi _ { m }$ be descriptors on $U _ { L }$ , and let $\Psi = \left( \Phi _ { 1 } , \ldots , \Phi _ { m } \right)$ . Then

$$
\Psi ^ { - 1 } ( \Psi ( c ) ) = \bigcap _ { j = 1 } ^ { m } \Phi _ { j } ^ { - 1 } ( \Phi _ { j } ( c ) ) .
$$

On the open dense free maximal-rank locus for $\Psi$ , the regular fiber modulo rotations has dimension $n - 3 - r _ { \Psi }$ , where

$$
r _ { \Psi } = \operatorname* { m a x } _ { c \in U _ { L } } \mathrm { r a n k } \left( { \binom { d ( \Phi _ { 1 } ) _ { c } } { \vdots } } \right) .
$$

For separately specified observation error budgets, the jointly consistent set is the intersection of the individual consistent sets, and when nonempty, its diameter cannot exceed any individual diameter.

Proof. Equality of the concatenated values is equivalent to equality of each component value, and the displayed matrix is $d \Psi _ { c }$ . Apply Thm. 4.2. The last assertion follows from set inclusion and from the monotonicity of the maximum that defines the diameter. □

Thus separate releases about the same tuple must be audited jointly. Their ranks need not add, because their diferentials can be dependent. The conclusion applies to releases about one underlying tuple. It does not apply to measurements of unrelated objects.

## 5. Band powers

The band powers $P _ { l } ( c ) = \| f _ { l } \| ^ { 2 }$ form the descriptor $P = ( P _ { 1 } , \dots , P _ { L } )$ of [10]. The fiber of $P$ through c is a product of spheres. It contains $( R _ { 1 } \cdot f _ { 1 } , \ldots , R _ { L } \cdot f _ { L } )$ for all $R _ { 1 } , \ldots , R _ { L } \in \mathrm { S O } ( 3 )$ The gradient of $P _ { l }$ is a nonzero multiple of $f _ { l } ,$ so $P$ has rank L wherever every $f _ { l }$ is nonzero. By Thm. 4.2, for $L \geq 2$ the fiber of $P$ through a generic point is, modulo rotations, a manifold of dimension $L ^ { 2 } + L - 3$

On $\mathcal { H } _ { 2 }$ we use the matrix $Q$ of Sec. 3 and the Frobenius norm $\lVert Q \rVert$ . The invariant inner product on $\mathcal { H } _ { 2 }$ is unique up to a positive factor. Any other choice multiplies all radii on $\mathcal { H } _ { 2 }$ by one constant. In the next two statements the definitions of Sec. 2 are applied with $\mathcal { H } _ { 2 }$ in place of $U _ { L } ;$ ; the orbit distance on $\mathcal { H } _ { 2 }$ is the restriction of the orbit distance on $U _ { 2 }$ to the pairs with $a = a ^ { \prime } = 0$

Proposition 5.1. Let $r > 0$ and let $Q \in \mathcal { H } _ { 2 }$ with $\| Q \| = r$ . Let $\theta \in [ 0 , \pi / 3 ]$ satisfy cos $3 \theta =$ $3 \sqrt { 6 }$ det $Q / r ^ { 3 }$ . Let $K \subset \mathcal { H } _ { 2 }$ be compact and contain the sphere $\{ Q ^ { \prime } : \| Q ^ { \prime } \| = r \}$ . Then

$$
r _ { P _ { 2 } } ( Q ; K ) = 2 r \sin \frac { \operatorname* { m a x } \{ \theta , \pi / 3 - \theta \} } { 2 } .
$$

In particular $\begin{array} { r } { \frac { \sqrt { 6 } - \sqrt { 2 } } { 2 } r \leq r _ { P _ { 2 } } ( Q ; K ) \leq r } \end{array}$ . The lower bound is attained $i f$ and only if det $Q = 0$ The upper bound is attained $i f$ and only $i f Q$ has a repeated eigenvalue.

Proof. For $\varphi \in \mathbb { R }$ put

$$
\mu ( \varphi ) = r \sqrt { 2 / 3 } \left( \cos \varphi , \ \cos ( 2 \pi / 3 - \varphi ) , \ \cos ( 2 \pi / 3 + \varphi ) \right) .
$$

The identity $\begin{array} { r } { \sum _ { k = 0 } ^ { 2 } \cos ( x + 2 \pi k / 3 ) \cos ( y + 2 \pi k / 3 ) = \frac { 3 } { 2 } \cos ( x - y ) ~ \mathrm { g i v e s } ~ \langle \mu ( \varphi ) , \mu ( \varphi ^ { \prime } ) \rangle = r ^ { 2 } \cos ( \varphi - \varphi ^ { 2 } ) , } \end{array}$ $\varphi ^ { \prime } )$ . Hence

$$
\| \mu ( \varphi ) - \mu ( \varphi ^ { \prime } ) \| = 2 r \sin \frac { | \varphi - \varphi ^ { \prime } | } 2 , \qquad | \varphi - \varphi ^ { \prime } | \leq \pi .
$$

So $\mu$ traces the circle of radius $r$ in the plane of vectors with coordinate sum zero, at unit angular speed. The coordinates of $\mu ( \varphi )$ are in decreasing order exactly on the arc $\varphi \in [ 0 , \pi / 3 ]$ So the spectra, in decreasing order, of the matrices $Q ^ { \prime } \in \mathcal { H } _ { 2 }$ with $\| Q ^ { \prime } \| = r$ are exactly the vectors $\mu ( \varphi )$ with $\varphi \in [ 0 , \pi / 3 ]$

Let $\lambda ( A )$ denote the spectrum of a symmetric matrix A in decreasing order. By the Hofman– Wielandt inequality in [8] and the rearrangement inequality, $\lVert Q - U Q ^ { \prime } U ^ { \top } \rVert \ge \lVert \lambda ( Q ) - \lambda ( Q ^ { \prime } ) \rVert$ for every $U \in \mathrm { O } ( 3 )$ . Equality holds when $U$ carries an orthonormal eigenbasis of $Q ^ { \prime }$ to one of $Q$ with the eigenvalues in the same order. Replacing U by $- U$ does not change $U Q ^ { \prime } U ^ { \top }$ , so the minimum over $\mathrm { S O ( 3 ) }$ equals the minimum over $\mathrm { O ( 3 ) }$ . Hence $\delta ( Q , Q ^ { \prime } ) = \lVert \lambda ( Q ) - \lambda ( Q ^ { \prime } ) \rVert$

Write $\lambda ( Q ) = \mu ( \theta ^ { \prime } )$ with $\theta ^ { \prime } \in [ 0 , \pi / 3 ]$ . The identity $\begin{array} { r } { \prod _ { k = 0 } ^ { 2 } \cos ( \theta ^ { \prime } + 2 \pi k / 3 ) = \frac { 1 } { 4 } \cos 3 \theta ^ { \prime } } \end{array}$ gives

$$
\operatorname* { d e t } Q = r ^ { 3 } \Big ( \frac { 2 } { 3 } \Big ) ^ { 3 / 2 } \frac { \cos 3 \theta ^ { \prime } } { 4 } = \frac { r ^ { 3 } \cos 3 \theta ^ { \prime } } { 3 \sqrt { 6 } } .
$$

Since $3 \theta ^ { \prime } \in [ 0 , \pi ]$ , this determines $\theta ^ { \prime } { } _ { 3 }$ and $\theta ^ { \prime } = \theta$ . The fiber of $P _ { 2 }$ through $Q$ is the sphere $\{ \| Q ^ { \prime } \| = r \}$ , which lies in $K$ . Therefore

$$
r _ { P _ { 2 } } ( Q ; K ) = \operatorname* { m a x } _ { \varphi \in [ 0 , \pi / 3 ] } 2 r \sin \frac { | \theta - \varphi | } { 2 } = 2 r \sin \frac { \operatorname* { m a x } \{ \theta , \pi / 3 - \theta \} } { 2 } ,
$$

because sin is increasing on $[ 0 , \pi / 6 ]$ . The quantity $\operatorname* { m a x } \{ \theta , \pi / 3 - \theta \}$ lies in $[ \pi / 6 , \pi / 3 ]$ . It equals $\pi / 6$ exactly when $\theta = \pi / 6 .$ , that is, when det $Q = 0$ . It equals $\pi / 3$ exactly when $\theta \in \{ 0 , \pi / 3 \}$ , that is, when two eigenvalues of $Q$ coincide. The values 2r sin $\begin{array} { r } { . ( \pi / 1 2 ) = \frac { \sqrt { 6 } - \sqrt { 2 } } { 2 } r } \end{array}$ and $2 r \sin ( \pi / 6 ) = r$ complete the proof. □

Adjoining det $Q$ to $P _ { 2 }$ gives a separating descriptor on $\mathcal { H } _ { 2 }$ . The numbers tr $Q ^ { 2 }$ and det $Q$ determine the spectrum of $Q ,$ and symmetric matrices with the same spectrum are conjugate by a rotation. So the radius of $( P _ { 2 } , \operatorname* { d e t } Q )$ vanishes at every point. The invariant cubic forms on $\mathcal { H } _ { 2 }$ are the multiples of det $Q$ . So det $Q$ is, up to a nonzero factor, the bispectrum of the quadrupole with itself in the sense of [9].

At positive tolerance the radius of the separating pair is again explicit. Knowledge of the power is side information, recorded by the set $K$

Corollary 5.2. Let $r > 0 .$ , let $S _ { r } = \{ Q ^ { \prime } \in \mathcal { H } _ { 2 } : \| Q ^ { \prime } \| = r \}$ , and let $I = ( P _ { 2 } , \mathrm { d e t } )$ . Let $Q \in S _ { r }$ let θ be as in Prop. 5.1, let $\varepsilon \geq 0$ , and put $\eta = 3 \sqrt { 6 } \varepsilon / r ^ { 3 }$ and

$$
\begin{array} { l } { \varphi _ { - } = \frac { 1 } { 3 } \operatorname { a r c c o s } \big ( \operatorname* { m i n } \{ 1 , \cos 3 \theta + \eta \} \big ) , } \\ { \varphi _ { + } = \frac { 1 } { 3 } \operatorname { a r c c o s } \big ( \operatorname* { m a x } \{ - 1 , \cos 3 \theta - \eta \} \big ) . } \end{array}
$$

Then

$$
r _ { I } ^ { \varepsilon } ( Q ; S _ { r } ) = 2 r \sin \frac { \operatorname* { m a x } \{ \theta - \varphi _ { - } , ~ \varphi _ { + } - \theta \} } { 2 } .
$$

$\begin{array} { r } { A s \ \varepsilon  \ 0 , \ r _ { I } ^ { \varepsilon } ( Q ; S _ { r } ) = \frac { r \eta } { 3 \sin 3 \theta } + O ( \eta ^ { 2 } ) \ i f \ 0 < \theta < \pi / 3 } \end{array}$ , and $\begin{array} { r } { r _ { I } ^ { \varepsilon } ( Q ; S _ { r } ) = \frac { r \sqrt { 2 \eta } } { 3 } + O ( \eta ^ { 3 / 2 } ) } \end{array}$ if $\theta \in \{ 0 , \pi / 3 \}$

Proof. For $Q ^ { \prime } \in S _ { r }$ write $\lambda ( Q ^ { \prime } ) = \mu ( \varphi )$ with $\varphi \in [ 0 , \pi / 3 ]$ , as in the proof of Prop. 5.1. The power is constant on $S _ { r }$ , and det $\boldsymbol { Q } ^ { \prime } \ : = \ : r ^ { 3 }$ cos $3 \varphi / ( 3 \sqrt { 6 } )$ . So $Q ^ { \prime }$ is accepted if and only if |cos $3 \varphi - \cos 3 \theta | \leq \eta$ . Since cos $3 \varphi$ decreases on $[ 0 , \pi / 3 ]$ , this holds if and only if $\varphi \in [ \varphi _ { - } , \varphi _ { + } ]$ By the proof of Prop. 5.1, $\delta ( Q , Q ^ { \prime } ) = 2 r \sin ( | \bar { \theta ^ { - } } - \varphi | / 2 )$ . Its maximum over $[ \varphi _ { - } , \varphi _ { + } ]$ is the stated value. For $0 < \theta < \pi / 3$ the derivative of arccos at cos 3θ is $- 1 /$ sin 3θ, which gives $\varphi _ { \pm } =$ $\theta \pm \eta / ( 3 \sin 3 \theta ) + O ( \eta ^ { 2 } )$ . For $\theta = 0$ we have $\varphi _ { - } = 0$ and $\begin{array} { r } { \varphi _ { + } = \frac { 1 } { 3 } \operatorname { a r c c o s } ( 1 - \eta ) = \frac { 1 } { 3 } \sqrt { 2 \eta } + O ( \eta ^ { 3 / 2 } ) } \end{array}$ The case $\theta = \pi / 3$ is symmetric. □

The maximum over the fixed-power sphere is also explicit:

$$
\operatorname* { m a x } _ { Q \in S _ { r } } r _ { I } ^ { \varepsilon } ( Q ; S _ { r } ) = 2 r \sin \left( \frac { 1 } { 6 } \operatorname { a r c c o s } \left( 1 - \operatorname* { m i n } \{ \eta , 2 \} \right) \right) .\tag{1}
$$

To see this, for $0 \leq a \leq b \leq \pi$ the identity co $~ \textsf { s a } - \cos b = 2 \sin ( ( a + b ) / 2 ) \sin ( ( b - a ) / 2 )$ gives cos $a - \cos b \geq 1 - \cos ( b - a )$ . Apply this with $a , b$ the ordered angles $3 \theta , 3 \varphi ;$ equality is attained when one angle is an endpoint. Thus the uniform exponent $1 / 2$ is attained and cannot be increased. At a fixed spectrum with $0 < \theta < \pi / 3$ the radius is linear to first order. These statements assume exact knowledge of the power and do not describe joint noise in both invariants.

Proposition 5.3. Let $L = 2$ and use on $U _ { 2 }$ the invariant norm $\| ( a , Q ) \| ^ { 2 } = | a | ^ { 2 } + \| Q \| ^ { 2 }$ . Let $r _ { 1 } , r _ { 2 } > 0$ , let $F = \left\{ ( a , Q ) : | a | = r _ { 1 } , \ \| Q \| = r _ { 2 } \right\}$ be the corresponding fiber of $P = ( P _ { 1 } , P _ { 2 } )$ , and put

$$
c _ { \pm } = \Big ( r _ { 1 } e _ { 1 } , ~ \pm \frac { r _ { 2 } } { \sqrt { 6 } } \mathrm { d i a g } ( 2 , - 1 , - 1 ) \Big ) .
$$

Then $c _ { \pm } \in F$ and

$$
\delta ( c _ { + } , c _ { - } ) ^ { 2 } = \left\{ \begin{array} { l l } { 2 r _ { 1 } ^ { 2 } + r _ { 2 } ^ { 2 } - \frac { r _ { 1 } ^ { 4 } } { 3 r _ { 2 } ^ { 2 } } , } & { r _ { 1 } ^ { 2 } \le 3 r _ { 2 } ^ { 2 } , } \\ { 4 r _ { 2 } ^ { 2 } , } & { r _ { 1 } ^ { 2 } \ge 3 r _ { 2 } ^ { 2 } . } \end{array} \right.
$$

Consequently $r _ { P } ( c ; K ) \ge { \textstyle { \frac { 1 } { 2 } } } \delta ( c _ { + } , c _ { - } )$ for every $c \in F$ and every compact $K \supset F$

Proof. Put $s = r _ { 2 } / \sqrt { 6 }$ and $E = 3 e _ { 1 } e _ { 1 } ^ { \top } - I = \mathrm { d i a g } ( 2 , - 1 , - 1 )$ , so $c _ { \pm } = ( r _ { 1 } e _ { 1 } , \pm s E )$ . Both points lie in F because $\| E \| ^ { 2 } = 6$ . Let $R \in \mathrm { S O ( 3 ) }$ , put $u = R e _ { 1 }$ , and put $t = \langle u , e _ { 1 } \rangle$ . Then

$$
\begin{array} { r l } & { \| r _ { 1 } e _ { 1 } - R ( r _ { 1 } e _ { 1 } ) \| ^ { 2 } = 2 r _ { 1 } ^ { 2 } ( 1 - t ) , } \\ & { ~ s E - R ( - s E ) R ^ { \top } = s \big ( 3 e _ { 1 } e _ { 1 } ^ { \top } + 3 u u ^ { \top } - 2 I \big ) . } \end{array}
$$

The symmetric matrix $e _ { 1 } e _ { 1 } ^ { \top } + u u ^ { \top }$ has eigenvalues $1 + t , 1 - t ,$ , and 0. So the second matrix has eigenvalues $s ( 1 + 3 t )$ , $s ( 1 - 3 t )$ , and $- 2 s$ , and its squared norm is $s ^ { 2 } ( 6 + 1 8 t ^ { 2 } ) = r _ { 2 } ^ { 2 } ( 1 + 3 t ^ { 2 } )$ Hence

$$
\| c _ { + } - R \cdot c _ { - } \| ^ { 2 } = g ( t ) , \qquad g ( t ) = 2 r _ { 1 } ^ { 2 } ( 1 - t ) + r _ { 2 } ^ { 2 } ( 1 + 3 t ^ { 2 } ) .
$$

Every $t \in [ - 1 , 1 ]$ occurs. The function g is convex, with minimum at $t _ { 0 } = r _ { 1 } ^ { 2 } / ( 3 r _ { 2 } ^ { 2 } ) > 0$ . If $t _ { 0 } \leq 1$ , then $\delta ( c _ { + } , c _ { - } ) ^ { 2 } = g ( t _ { 0 } ) = 2 r _ { 1 } ^ { 2 } + r _ { 2 } ^ { 2 } - r _ { 1 } ^ { 4 } / ( 3 r _ { 2 } ^ { 2 } )$ . If $t _ { 0 } \geq 1$ , then g is decreasing on $[ - 1 , 1 ]$ and $\delta ( c _ { + } , c _ { - } ) ^ { 2 } = g ( 1 ) = 4 r _ { 2 } ^ { 2 }$ . For $c \in F$ the triangle inequality gives max $\{ \delta ( c , c _ { + } ) , \delta ( c , c _ { - } ) \} \ge$ $\begin{array} { r } { \frac { 1 } { 2 } \delta ( c _ { + } , c _ { - } ) } \end{array}$ , and both $c _ { \pm }$ lie in $F \subset K$ □

For the norm $w _ { 1 } | a | ^ { 2 } + w _ { 2 } \| Q \| ^ { 2 }$ , replace $r _ { 1 }$ and $r _ { 2 }$ by $\sqrt { w _ { 1 } } r _ { 1 }$ and $\sqrt { w _ { 2 } } r _ { 2 }$ . The map $( a , Q ) \mapsto$ $( { \sqrt { w _ { 1 } } } a , { \sqrt { w _ { 2 } } } Q )$ is an equivariant isometry onto $U _ { 2 }$ with the norm of Prop. 5.3, and it maps fibers of $P$ to fibers of $P .$

![](images/092147bb2ae46ead67f6eff0be1d1e17974445c09186a35102ca6c0d6ba1e8f1.jpg)  
Figure 1. The surfaces $c _ { + } \ ( \mathrm { l e f t } )$ and $c _ { - } \ \mathrm { ( r i g h t ) }$ of Prop. 5.3, drawn as radial surfaces $\rho _ { 0 } + f _ { 1 } + f _ { 2 }$ with $\rho _ { 0 } = 1 , r _ { 1 } = 0 . 2 5$ , and $r _ { 2 } = 0 . 3 5$ , where $f _ { 1 } ( p ) = \langle a , p \rangle$ and $f _ { 2 } ( p ) = p ^ { \top } Q p$ . Their band powers coincide. Their orbit distance is $\delta ( c _ { + } , c _ { - } ) \approx 0 . 4 8 7$

## 6. The spectral pool

For binary forms F and G of degrees m and $m ^ { \prime }$ in $\left( z _ { 0 } , z _ { 1 } \right)$ , the transvectant of order k is

$$
( F , G ) _ { k } = { \frac { ( m - k ) ! ( m ^ { \prime } - k ) ! } { m ! m ^ { \prime } ! } } \sum _ { i = 0 } ^ { k } ( - 1 ) ^ { i } { \binom { k } { i } } { \frac { \partial ^ { k } F } { \partial z _ { 0 } ^ { k - i } \partial z _ { 1 } ^ { i } } } { \frac { \partial ^ { k } G } { \partial z _ { 0 } ^ { i } \partial z _ { 1 } ^ { k - i } } } ,
$$

as in [16, Eq. (18)]. By [16, Thm. 3.8], restriction to the null cone gives an $\mathrm { S L _ { 2 } ( \mathbb { C } ) }$ -equivariant linear isomorphism $\Psi _ { l }$ from $\mathcal { H } _ { l } \otimes \mathbb { C }$ onto the binary forms of degree 2l. Write $F _ { l } = \Psi _ { l } ( f _ { l } )$ and

$$
\begin{array} { r } {  { \mathcal { A } } ( L ) = \{ ( l _ { 1 } , l _ { 2 } , l _ { 3 } ) \in  { \mathbb { Z } } ^ { 3 } : 1 \le l _ { 1 } \le l _ { 2 } \le l _ { 3 } \le L , \ l _ { 3 } \le l _ { 1 } + l _ { 2 } , \ l _ { 1 } + l _ { 2 } + l _ { 3 } \ \mathrm { e v e n } \} , } \end{array}
$$

$$
\begin{array} { r } { A ^ { - } ( L ) = \{ ( l _ { 1 } , l _ { 2 } , l _ { 3 } ) \in \mathbb { Z } ^ { 3 } : 1 \leq l _ { 1 } < l _ { 2 } < l _ { 3 } \leq L , \ l _ { 3 } \leq l _ { 1 } + l _ { 2 } , \ l _ { 1 } + l _ { 2 } + l _ { 3 } \ \mathrm { o d d } \} . } \end{array}
$$

For $L \geq 2$ the pool $\Phi _ { L }$ has the following components.

(1) The powers $P _ { l } = ( F _ { l } , F _ { l } ) _ { 2 l }$ for $1 \leq l \leq L$

(2) The bispectra $B _ { l _ { 1 } l _ { 2 } l _ { 3 } } = \left( ( F _ { l _ { 1 } } , F _ { l _ { 2 } } ) _ { l _ { 1 } + l _ { 2 } - l _ { 3 } } , F _ { l _ { 3 } } \right) _ { 2 l _ { 3 } } { \mathrm { ~ f o r ~ } } ( l _ { 1 } , l _ { 2 } , l _ { 3 } ) \in A ( L ) .$

(3) For $L \geq 3 ,$ the invariants $J _ { 4 } , \ J _ { 6 } , \ J _ { 1 0 }$ of the sextic $F _ { 3 } .$ They are obtained from its Igusa–Clebsch invariants by $[ 1 6 , \ \mathrm { E q s . } \ ( 2 0 ) - ( 2 1 ) ] , \ J _ { 2 } \ = \ I _ { 2 } / 8 , \ J _ { 4 } \ = \ ( 4 J _ { 2 } ^ { 2 } - I _ { 4 } ) / 9 6$ $J _ { 6 } = ( 8 J _ { 2 } ^ { 3 } - 1 6 0 J _ { 2 } J _ { 4 } - I _ { 6 } ) / 5 7 \bar { 6 } .$ and $J _ { 1 0 } = I _ { 1 0 } / 4 0 9 6 .$

The number of components is $N ( 2 ) = 4$ and $N ( L ) = L + \left| \mathcal { A } ( L ) \right| + 3$ for $L \geq 3$ . The odd bispectra $B _ { l _ { 1 } l _ { 2 } l _ { 3 } }$ with $( l _ { 1 } , l _ { 2 } , l _ { 3 } ) \in \mathcal { A } ^ { - } ( L )$ are given by the same formula. In the symmetric $3 j$ normalization, an invariant trilinear contraction changes by $( - 1 ) ^ { l _ { 1 } + l _ { 2 } + l _ { 3 } }$ under interchange of two factors; another nonzero normalization gives the same vanishing criterion. Thus an odd bispectrum with two equal bands vanishes when evaluated on the same band vector twice; equivalently, after reordering the contraction, $( F , F ) _ { k } = 0$ for odd k. The pool $\Phi _ { L } ^ { + }$ consists of $\Phi _ { L }$ and the odd bispectra divided by $i ,$ and it has $N ^ { + } ( L ) = N ( L ) + \vert A ^ { - } ( L ) \vert$ | components. We call $\Phi _ { L }$ the spectral pool, or the even pool when it is contrasted with $\Phi _ { L } ^ { + }$ , and we call $\Phi _ { L } ^ { + }$ the augmented pool.

The powers are the power spectrum of [10] and the $B _ { l _ { 1 } l _ { 2 } l _ { 3 } }$ are bispectra in the sense of [9]. The pool is the radial invariant block of [15, Sec. 5.2]. There the power of band l is $\| f _ { l } \| ^ { 2 }$ Its complexification is a nonzero multiple of $P _ { l }$ , because an invariant quadratic form on the irreducible module of binary forms of degree 2l is unique up to a scalar. Let $d _ { l }$ be the degree of a component in $F _ { l } .$ . For every component of $\Phi _ { L }$ the sum $\sum _ { l } l d _ { l }$ is even, and for every odd bispectrum it is odd. So the components of $\Phi _ { L }$ are real on real harmonics and the odd bispectra are purely imaginary by [16, Lem. 3.6(iii)]. Hence $\Phi _ { L }$ and $\Phi _ { L } ^ { + }$ are descriptors on $U _ { L }$ . The block of [15] does not contain the odd bispectra.

<table><tr><td>L</td><td>n</td><td> $n - 3$ </td><td> $N ( L )$ </td><td> $r ( L )$ </td><td> $n - 3 - r ( L )$ </td><td> $N ^ { + } ( L )$ </td><td> $r ^ { + } ( L )$ </td><td> $n - 3 - r ^ { + } ( L )$ </td></tr><tr><td>2</td><td>8</td><td>5</td><td>4</td><td>4</td><td>1</td><td>4</td><td>4</td><td>1</td></tr><tr><td>3</td><td>15</td><td>12</td><td>10</td><td>10</td><td>2</td><td>10</td><td>10</td><td>2</td></tr><tr><td>4</td><td>24</td><td>21</td><td>16</td><td>16</td><td>5</td><td>17</td><td>17</td><td>4</td></tr><tr><td>5</td><td>35</td><td>32</td><td>22</td><td>22</td><td>10</td><td>24</td><td>24</td><td>8</td></tr><tr><td>6</td><td>48</td><td>45</td><td>32</td><td>32</td><td>13</td><td>37</td><td>37</td><td>8</td></tr><tr><td>7</td><td>63</td><td>60</td><td>42</td><td>42</td><td>18</td><td>50</td><td>50</td><td>10</td></tr><tr><td>8</td><td>80</td><td>77</td><td>57</td><td>57</td><td>20</td><td>71</td><td>71</td><td>6</td></tr><tr><td>9</td><td>99</td><td>96</td><td>72</td><td>72</td><td>24</td><td>92</td><td>92</td><td>4</td></tr><tr><td>10</td><td>120</td><td>117</td><td>93</td><td>93</td><td>24</td><td>123</td><td>117</td><td>0</td></tr><tr><td>11</td><td>143</td><td>140</td><td>114</td><td>114</td><td>26</td><td>154</td><td>140</td><td>0</td></tr><tr><td>12</td><td>168</td><td>165</td><td>142</td><td>142</td><td>23</td><td>197</td><td>165</td><td>0</td></tr><tr><td>13</td><td>195</td><td>192</td><td>170</td><td>170</td><td>22</td><td>240</td><td>192</td><td>0</td></tr><tr><td>14</td><td>224</td><td>221</td><td>206</td><td>206</td><td>15</td><td>297</td><td>221</td><td>0</td></tr><tr><td>15</td><td>255</td><td>252</td><td>242</td><td>242</td><td>10</td><td>354</td><td>252</td><td>0</td></tr><tr><td>16</td><td>288</td><td>285</td><td>287</td><td>285</td><td>0</td><td>427</td><td>285</td><td>0</td></tr></table>

Table 1. Sizes, generic ranks, and generic fiber dimensions of the pools $\Phi _ { L }$ and $\Phi _ { L } ^ { + }$

Theorem 6.1. Let $2 \le L \le 1 6$ , and let $r ( L )$ and $r ^ { + } ( L )$ be the generic ranks of $\Phi _ { L }$ and $\Phi _ { L } ^ { + }$ Then $r ( L ) = N ( L ) < n - 3$ for $L  \leq 1 5$ , and $r ( 1 6 ) = n - 3$ . Moreover $r ^ { + } ( L ) = N ^ { + } ( L ) < n - 3$ for $L \leq 9 .$ , and $r ^ { + } ( L ) = n - 3$ for $1 0 \leq L \leq 1 6$ . Let Φ be $\Phi _ { L }$ or $\Phi _ { L } ^ { + }$ , and let $U ^ { \prime }$ and $M _ { c }$ be as in Thm. 4.2. Then the following hold.

(1) If the generic rank of Φ is less than $n - 3$ and $c \in U ^ { \prime }$ , then ${ M _ { c } } / \mathrm { S O ( 3 ) }$ is a smooth manifold of positive dimension, listed in Tab. 1, and $r _ { \Phi } ( c ; K ) > 0$ for every compact neighborhood K of c.

(2) If the generic rank of Φ equals $n - 3$ and $c \in U ^ { \prime }$ , then $M _ { c }$ is a finite union of $S \mathrm { O } ( 3 ) – o r b i t s$ Proof. $\mathrm { B y }$ Thm. 4.2, the generic rank is at most the number of components and at most $n - 3$ The counts in Tab. 1 give $N ( L ) < n - 3$ for $L  \leq 1 5$ and $N ( 1 6 ) > n - 3$ , and $N ^ { + } ( L ) < n - 3$ for $L \leq 9$ and $N ^ { + } ( L ) > n - 3$ for $L \geq 1 0$ . It remains to bound the ranks from below.

The coeficients of the forms $F _ { l }$ are complex linear coordinates on $U _ { L } \otimes \mathbb { C }$ . In these coordinates the components of $\Phi _ { L }$ , and the odd bispectra up to the constant i, are polynomials with rational coeficients, and they define the complexifications. A minor of the Jacobian that vanishes on $U _ { L }$ vanishes on $U _ { L } \otimes \mathbb { C }$ , because a polynomial that vanishes on a real form of a complex vector space is zero. So each generic rank equals that of the complexification. It is at least the rank of the Jacobian at any point of $U _ { L } \otimes \mathbb { C }$

Put $i = ( F _ { 3 } , F _ { 3 } ) _ { 4 }$ and $\Delta = ( i , i ) _ { 2 }$ . In this paragraph i and ∆ denote these covariants, and not the imaginary unit and the Laplacian. In the rows of the Jacobian we replace $J _ { 4 } , J _ { 6 } , J _ { 1 0 }$ by the invariants $( i , i ) _ { 4 } , ( i , \Delta ) _ { 4 }$ , and the discriminant of $F _ { 3 } .$ , of degrees 4, 6, 10. Together with $P _ { 3 }$ , these generate the same algebra as $P _ { 3 } , J _ { 4 } , J _ { 6 } , J _ { 1 0 }$ . Indeed, the Clebsch invariants $( F _ { 3 } , F _ { 3 } ) _ { 6 } , ( i , i ) _ { 4 }$ ， $( i , \Delta ) _ { 4 }$ and the Clebsch invariant of degree ten generate the algebra of invariants of even degree of the sextic. The invariants $I _ { 2 } , I _ { 4 } , I _ { 6 } , I _ { 1 0 }$ are polynomials in them, and $I _ { 1 0 }$ is the discriminant up to a nonzero factor, with a nonzero coeficient on the degree-ten generator. Each set is therefore a polynomial function of the other. By the chain rule, each Jacobian row space is contained in the other at every point; hence the two row spaces coincide.

We evaluate the Jacobian at a point whose coordinates in the monomial bases are integers in $[ - 9 , 9 ]$ . The coordinates are produced, band by band and coeficient by coeficient, by the linear congruential generator $x \mapsto ( 1 6 6 4 5 2 5 x + 1 0 1 3 9 0 4 2 2 3 )$ mod $2 ^ { 3 2 }$ , started at $x = 1$ , with coordinate $\left( \lfloor x / 2 ^ { 1 6 } \rfloor \right)$ mod $1 9 ) - 9$ . The same point serves for every $L$ . The entries of the Jacobian are rational numbers. Their denominators divide products of factorials of integers at most $2 L \leq 3 2$ . So they are prime to the prime $p = 2 ^ { 3 1 } - 1$ . The rank of the reduction modulo $p$ is at most the rank over Q. At this point the reductions have the ranks listed in Tab. 1. This gives the lower bounds. Parts (1) and (2) follow from Thm. 4.2. □

The accompanying program rank witness.py performs this computation exactly modulo $p$ for $2 \le L \le 1 6$ . For each L and each pool it prints the evaluation point, the labels of a maximal linearly independent set of Jacobian rows, the pivot columns of that row set, and the determinant of the resulting square minor modulo p. A nonzero determinant certifies the rank. For $L  \leq 1 5$ the minor of $\Phi _ { L }$ has size $N ( L )$ , and for $L = 1 6$ it has size 285. The starting value of the generator is a parameter of the program; the starting values 2, 3 and 7 were also tested and produced the same ranks. At the cutofs $L \in \{ 4 , 6 , 8 \}$ of [15, Sec. 7.3], the fiber dimensions are 5, 13, 20 for $\Phi _ { L }$ and $4 , 8 , 6$ for $\Phi _ { L } ^ { + }$

We compare these ranks with known orbit-recovery results. For one spherical function including the constant band, [7, Thm. 4.1 and Cor. 4.2] proves finite generic recovery from invariants of degree at most three for every $L \geq 1 0$ . On $\mathcal { H } _ { 0 } \oplus U _ { L }$ the invariants of degree at most three are polynomials in $f _ { 0 }$ , the powers, and the even and odd bispectra. Finite generic recovery gives their map the generic rank $( L + 1 ) ^ { 2 } - 3$ . Fixing $f _ { 0 }$ lowers the rank by at most one, so the powers and bispectra have generic rank at least $n - 3$ on $U _ { L } ;$ ; adjoining the sextic invariants cannot lower it. In particular, $r ^ { + } ( L ) = n - 3$ holds for every $L \geq 1 0$ , and the saturation of $\Phi _ { L } ^ { + }$ at $L = 1 0$ is not a new generic-recovery threshold. The comparison made here concerns the even pool $\Phi _ { L }$ , its radii and mirror ambiguity, and the information retained for particular tasks.

For R spherical shells, [6, Cor. 5.1] gives recovery under $L \geq 3$ and $R \ge L + 2$ , while [2, Thm. 2.2] gives generic separation from degree-at-most-three invariants for $R \geq 3$ . The radial function considered here has one copy of each harmonic band, whereas those results require several independently sampled shells.

The odd bispectra change sign under reflection, and $\Phi _ { L }$ does not change. Let $\tau \colon U _ { L } \to U _ { L }$ be $\tau ( f _ { 1 } , \dots , f _ { L } ) = ( - f _ { 1 } , f _ { 2 } , \dots , ( - 1 ) ^ { L } f _ { L } )$ , so that τ replaces the radial function by $p \mapsto \rho ( - p )$ Put $\chi ( c ) = \operatorname * { d e t } ( a , Q a , Q ^ { 2 } a )$ , as in Sec. 4.

Proposition 6.2. Let $L \geq 2$ . Then the following hold.

(1) The map τ commutes with the action $o f \mathrm { S O ( 3 ) }$ . Moreover $\Phi _ { L } \circ \tau = \Phi _ { L } , B \circ \tau = - B$ for every odd bispectrum $B _ { i }$ , and $\chi \circ \tau = - \chi$

(2) For every $c \in \Omega _ { L }$ , the point τc lies in the fiber of $\Phi _ { L }$ through c and not in $\mathrm { S O } ( 3 ) \cdot c$ . In particular $\Phi _ { L }$ does not separate orbits.

(3) Let $U ^ { \prime }$ and $M _ { c }$ be as in Thm. $ \mathbf { \nabla } _ { 4 } . \mathcal { Q } ~ f o r ~ \Phi = \Phi _ { L }$ . Then $\tau ( U ^ { \prime } ) = U ^ { \prime }$ and $\tau ( M _ { c } ) = M _ { c }$ for every $c \in U ^ { \prime }$ . For $L = 1 6$ and $c \in U ^ { \prime }$ , the number of orbits in $M _ { c }$ is even, and at least two.

Proof. (1) For $R \in { \mathrm { S O } } ( 3 ) , ( R \cdot \tau f ) ( p ) = f ( - R ^ { - 1 } p ) = ( \tau ( R \cdot f ) ) ( p )$ . The map $\Psi _ { l }$ is linear, so τ multiplies $F _ { l } \ \mathrm { b y } \ ( - 1 ) ^ { l } . \mathrm { ~ A ~ }$ component of degree $d _ { l }$ in $F _ { l }$ is multiplied by $( - 1 ) \sum _ { l } l d _ { l }$ , which is 1 for the components of $\Phi _ { L }$ and −1 for the odd bispectra. The map $\tau$ fixes $Q$ and negates $^ { a , }$ so $\chi ( \tau c ) = \operatorname * { d e t } ( - a , - Q a , - Q ^ { 2 } a ) = - \chi ( c )$

(2) By $( 1 ) , \Phi _ { L } ( \tau c ) = \Phi _ { L } ( c )$ . The function $\chi$ is constant on orbits by the proof of Lem. 4.1, and $\chi ( \tau c ) = - \chi ( c ) \neq \chi ( c )$ for $c \in \Omega _ { L }$ . Hence $\tau c \notin \mathrm { S O } ( 3 ) \cdot c$ . The set $\Omega _ { L }$ is not empty, so $\Phi _ { L }$ does not separate orbits.

(3) By (1), τ is linear with $\Phi _ { L } \circ \tau = \Phi _ { L } .$ , so $d ( \Phi _ { L } ) _ { \tau c } \circ \tau = d ( \Phi _ { L } ) _ { c }$ , and the rank is the same at c and τ c. By (1) also $\tau ( \Omega _ { L } ) = \Omega _ { L }$ . Hence $\tau ( U ^ { \prime } ) = U ^ { \prime }$ and $\tau ( M _ { c } ) = M _ { c }$ . By (1), τ permutes the orbits in $M _ { c } .$ , and $\tau ^ { 2 } = 1$ . No orbit is fixed, because $\chi$ is constant and nonzero on each orbit in $M _ { c }$ and changes sign under $\tau .$ . For $L = 1 6$ the orbits in $M _ { c }$ are finite in number by Thm. $6 . 1 .$ so their number is even. It is at least two, since $M _ { c }$ contains c and $\tau c .$ □

The point τ c matches every component of $\Phi _ { L }$ exactly. It is a distinct mirror decoy precisely when $\tau c \notin \mathrm { S O } ( 3 ) \cdot c ,$ equivalently when the surface has no orientation-reversing orthogonal symmetry about the chosen origin. Such a symmetry need not be reflection in a plane. On $\Omega _ { L }$ the two rotation classes are always distinct. At $L = 1 6$ , continuous ambiguity disappears on the free maximal-rank locus, but the mirror pair remains. The proposition neither counts all other finite ambiguities nor asserts that singular parts of a fiber have been classified.

At $L = 2$ the pool consists, up to nonzero factors, of $| a | ^ { 2 } , \mathrm { t r } Q ^ { 2 } , a ^ { \top } Q a$ , and det $Q$ . Its fibers have dimension one. By [16, Thm. 4.9], adjoining $a ^ { \top } Q ^ { 2 } a$ and $\chi = \operatorname * { d e t } ( a , Q a , Q ^ { 2 } a )$ gives a separating descriptor. These two invariants have degrees four and six, and every bispectrum is cubic. The second is odd under τ.

For the two adversaries of Sec. 2 these results read as follows. For $2 \le L \le 1 5$ and every $c \in U ^ { \prime }$ the decoy radius $r _ { \Phi _ { L } } ^ { \varepsilon } ( c ; K )$ is positive for every tolerance $\varepsilon$ and every compact neighborhood $K$ of c. A recognizer built on $\Phi _ { L }$ accepts, near every generic enrolled surface, a family of decoys of the dimension listed in Tab. 1, and it accepts τ c whenever τ c $\# \mathrm { S O } ( 3 ) \cdot c ,$ at every $L .$ . For $2 \le L \le 1 5$ , a captured value of $\Phi _ { L }$ at a point of $U ^ { \prime }$ is consistent with a family of rotation classes of positive dimension, and that family contains the mirror class. For $L = 1 6 .$ within $U ^ { \prime } .$ a captured value is consistent with finitely many rotation classes, including at least the mirror pair; additional finite ambiguities are not excluded. For $L \geq 4$ the odd bispectrum $B _ { 2 3 4 }$ is a nonzero trilinear invariant, so it is nonzero on an open dense subset and distinguishes c from τc there. There are no odd bispectra in these pools for $L = 2 , 3$ . Neither the addition of odd bispectra nor full generic rank proves global separation of all orbits.

## 7. An explicit finite-tolerance benchmark

In this section $w _ { l } = 1$ for all $l ,$ so $\lVert \cdot \rVert$ is the $L ^ { 2 }$ norm for the round probability measure. For $\theta \in \mathbb { R } ^ { 3 }$ let $R ( \theta ) \in \mathrm { S O } ( 3 )$ be the rotation with rotation vector θ. The closed ball of radius π is mapped onto $\mathrm { S O ( 3 ) }$

Lemma 7.1. Let $c , c ^ { \prime } \in U _ { L }$ with $c ^ { \prime } = ( f _ { 1 } ^ { \prime } , \ldots , f _ { L } ^ { \prime } )$ , and put

$$
K ^ { \prime } = \Big ( \sum _ { l = 1 } ^ { L } l ^ { 2 } \| f _ { l } ^ { \prime } \| ^ { 2 } \Big ) ^ { 1 / 2 } .
$$

Then the following hold.

(1) $| \| c - R ( \theta _ { 1 } ) \cdot c ^ { \prime } \| - \| c - R ( \theta _ { 2 } ) \cdot c ^ { \prime } \| \leq K ^ { \prime } | \theta _ { 1 } - \theta _ { 2 } | \ f o r \ a l l \ \theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { 3 }$

(2) For $R _ { 1 } , R _ { 2 } \ \in \ \mathrm { S O } ( 3 )$ let $\phi ( R _ { 1 } , R _ { 2 } ) ~ \in ~ [ 0 , \pi ]$ be the rotation angle of $R _ { 1 } ^ { - 1 } R _ { 2 }$ . Then $| \| c - R _ { 1 } \cdot c ^ { \prime } \| - \| c - R _ { 2 } \cdot c ^ { \prime } \| | \leq K ^ { \prime } \phi ( R _ { 1 } , R _ { 2 } ) .$

Proof. (2) Write $R _ { 1 } ^ { - 1 } R _ { 2 } = R ( \phi w )$ with a unit vector w and $\phi = \phi ( R _ { 1 } , R _ { 2 } )$ . On $\mathcal { H } _ { l }$ the generator of the rotations about a unit axis is skew, with eigenvalues im for $| m | \leq l .$ So the curve $t \mapsto R ( t \phi w ) \cdot c ^ { \prime }$ has speed at most $\phi K ^ { \prime }$ , and $\| R _ { 1 } \cdot c ^ { \prime } - R _ { 2 } \cdot c ^ { \prime } \| = \| c ^ { \prime } - \dot { R } _ { 1 } ^ { - 1 } R _ { 2 } \cdot c ^ { \prime } \| \le K ^ { \prime } \phi$ . The triangle inequality completes the proof.

(1) Put $\theta ( t ) = \theta _ { 1 } + t ( \theta _ { 2 } - \theta _ { 1 } )$ and $R _ { t } = R ( \theta ( t ) )$ for $t \in [ 0 , 1 ]$ . Then $\dot { R } _ { t } = R _ { t } \widehat { \omega ( t ) }$ , where $\widehat { \omega }$ is the skew matrix of ω $\in \mathbb { R } ^ { 3 }$ . The derivative of the exponential map of $\mathrm { S O ( 3 ) }$ at θ is $\exp ( { \widehat { \theta } } )$ composed with a normal operator on $\mathbb { R } ^ { 3 }$ whose eigenvalues are 1 and $( 1 - e ^ { \mp i | \theta | } ) / ( \pm i | \theta | )$ . These have modulus at most 1, so $| \omega ( t ) | \leq | \theta _ { 2 } - \theta _ { 1 } |$ . On $\mathcal { H } _ { l }$ the generator of the rotations about a unit axis is skew, with eigenvalues im for $| m | \leq l .$ Hence

$$
\left\| \frac { d } { d t } \big ( R _ { t } \cdot c ^ { \prime } \big ) \right\| \leq | \omega ( t ) | \Big ( \sum _ { l = 1 } ^ { L } l ^ { 2 } \| f _ { l } ^ { \prime } \| ^ { 2 } \Big ) ^ { 1 / 2 } \leq K ^ { \prime } | \theta _ { 2 } - \theta _ { 1 } | ,
$$

because $R _ { t }$ acts isometrically. Integration gives $\lVert R ( { \boldsymbol { \theta } } _ { 1 } ) \cdot { \boldsymbol { c } } ^ { \prime } - R ( { \boldsymbol { \theta } } _ { 2 } ) \cdot { \boldsymbol { c } } ^ { \prime } \rVert \leq K ^ { \prime } | { \boldsymbol { \theta } } _ { 1 } - { \boldsymbol { \theta } } _ { 2 } |$ , and the triangle inequality completes the proof. □

A second parametrization avoids trigonometric functions. For a unit quaternion $q =$ $\left( q _ { 0 } , q _ { 1 } , q _ { 2 } , q _ { 3 } \right)$ let $R ( q ) \in \mathrm { S O } ( 3 )$ be the rotation it represents; the entries of $R ( q )$ are quadratic forms in $q ,$ and $R ( q ) = R ( - q )$ . For $u \in \mathbb { R } ^ { 3 }$ and $k \in \{ 0 , 1 , 2 , 3 \}$ let $S _ { k } ( u ) = R ( q )$ , where $q$ is the unit quaternion whose k-th coordinate is $( 1 + | u | ^ { 2 } ) ^ { - 1 / 2 }$ and whose remaining three coordinates, in order, are the coordinates of $( 1 + | u | ^ { 2 } ) ^ { - 1 / 2 } u$ . The entries of $S _ { k } ( u )$ are rational functions of u with denominator $1 + | u | ^ { 2 }$

Lemma 7.2. Every rotation is $S _ { k } ( u )$ for some k and some $u \in [ - 1 , 1 ] ^ { 3 }$ . For $u , u ^ { \prime } \in [ - 1 , 1 ] ^ { 3 }$ with $| u - u ^ { \prime } | < \sqrt { 2 }$

$$
\phi \big ( S _ { k } ( u ) , S _ { k } ( u ^ { \prime } ) \big ) \leq \frac { \pi } { \sqrt { 2 } } | u - u ^ { \prime } | .
$$

Proof. Let $q _ { k }$ be a coordinate of largest absolute value of a unit quaternion $q ;$ then $q _ { k } \neq 0 .$ Replacing $q \ \mathrm { b y } \ - q .$ , we may take $q _ { k } > 0 ;$ ; then $q = q _ { k } ( 1 , u )$ up to the position of the coordinate 1, with $| u _ { j } | = | q _ { j } | / q _ { k } \le 1$ . This proves the first claim. Let $q$ and $q ^ { \prime }$ be the unit quaternions of $S _ { k } ( u )$ and $S _ { k } ( u ^ { \prime } )$ . The map $v \mapsto v / | v |$ has diferential of norm at most $1 / | v | ,$ , so on the segment from $( 1 , u )$ to $( 1 , u ^ { \prime } )$ , where $| v | \geq 1$ , it is 1-Lipschitz; hence $| q - q ^ { \prime } | \leq | u - u ^ { \prime } | < \sqrt { 2 }$ Put $\psi = \operatorname { a r c c o s } \langle q , q ^ { \prime } \rangle$ . From $| q - q ^ { \prime } | ^ { 2 } = 2 - 2 \langle q , q ^ { \prime } \rangle < 2$ we get $\langle q , q ^ { \prime } \rangle > 0$ and $\psi \in [ 0 , \pi / 2 )$ . The rotation ${ \cal R } ( q ) ^ { - 1 } { \cal R } ( q ^ { \prime } )$ is represented by the quaternion ${ \bar { q } } q ^ { \prime }$ , whose real part is $\left. q , q ^ { \prime } \right. =$ cos $\psi .$ so its rotation angle is $2 \psi$ . Since sin is concave on $[ 0 , \pi / 2 ]$ , sin $( \psi / 2 ) \ge ( 2 \sqrt { 2 } / \pi ) ( \psi / 2 )$ for $\psi / 2 \in [ 0 , \pi / 4 ]$ . Hence $| q - q ^ { \prime } | = 2 \sin ( \psi / 2 ) \geq ( { \sqrt { 2 } } / \pi ) ( 2 \psi )$ , which is the claim. □

Corollary 7.3 (Validated rotation covering). Let $c , c ^ { \prime } \in U _ { L }$ , let $K ^ { \prime }$ be as in Lem. $\ 7 . 1 ,$ and let $\overline { { K } } \geq K ^ { \prime }$ . Then the following hold.

(1) Let B be a finite collection of boxes covering $[ - \pi , \pi ] ^ { 3 }$ . For $B \in B$ let $\theta _ { B }$ be its center, let $h _ { B }$ bound the distance from the center to every point of the box, and let $\ell _ { B } \ \leq$ $\| c - R ( \theta _ { B } ) \cdot c ^ { \prime } \|$ be a validated lower bound. Then

$$
\delta ( c , c ^ { \prime } ) \geq \operatorname* { m a x } \{ 0 , \operatorname* { m i n } _ { B \in \mathcal { B } } ( \ell _ { B } - \overline { { K } } h _ { B } ) \} .
$$

(2) For $k = 0 , 1 , 2 , 3$ let $\boldsymbol { B } _ { k }$ be a finite collection of boxes covering $[ - 1 , 1 ] ^ { 3 }$ . For $B \in B _ { k }$ let u<sub>B</sub> be its center, let $h _ { B } < \sqrt { 2 }$ bound the distance from the center to every point of the box, and let $\ell _ { B } \leq \| c - S _ { k } ( u _ { B } ) \cdot c ^ { \prime } \|$ be a validated lower bound. Then

$$
\delta ( c , c ^ { \prime } ) \geq \operatorname* { m a x } \biggr \{ 0 , \operatorname* { m i n } _ { k } \operatorname* { m i n } _ { B \in \mathcal { B } _ { k } } \Big ( \ell _ { B } - \frac { \pi } { \sqrt { 2 } } \overline { { K } } h _ { B } \Big ) \biggr \} .
$$

Any validated upper bound on $\| c - R _ { 0 } \cdot c ^ { \prime } \|$ at a single rotation $R _ { 0 }$ is an upper bound on $\boldsymbol { \delta } ( \boldsymbol { c } , \boldsymbol { c } ^ { \prime } )$

Proof. (1) For $\theta \in B$ , Lem. 7.1(1) gives $\| c - R ( \theta ) \cdot c ^ { \prime } \| \geq \ell _ { B } - \overline { { K } } h _ { B }$ . The boxes cover all rotation vectors of length at most π, which represent every rotation. Taking the minimum over all boxes gives the stated bound. (2) For $u \in B \in B _ { k }$ , Lem. 7.1(2) and Lem. 7.2 give $\| c - S _ { k } ( u ) \cdot c ^ { \prime } \| \geq \ell _ { B } - ( \pi / \sqrt { 2 } ) \overline { { K } } h _ { B }$ , and by Lem. 7.2 the four families cover $\mathrm { S O ( 3 ) }$ . The upper bound follows by evaluating the function at one rotation. □

For the computation we parametrize the null cone of $\mathbb { C } ^ { 3 }$ by $( x , y , z ) = ( z _ { 0 } ^ { 2 } - z _ { 1 } ^ { 2 } , \ i ( z _ { 0 } ^ { 2 } +$ $z _ { 1 } ^ { 2 } ) , \ - 2 z _ { 0 } z _ { 1 } )$ and take $F _ { l }$ to be the restriction of $f _ { l }$ . Two parametrizations difer by a linear substitution in $\left( z _ { 0 } , z _ { 1 } \right)$ . Under it every component of $\Phi _ { L }$ is multiplied by a nonzero constant, because transvectants are relative invariants. Relative diferences of components do not depend on the choice. We fix bases of $\mathcal { H } _ { 1 } , \ldots , \mathcal { H } _ { 6 }$ consisting of harmonic polynomials with integer coeficients, listed in the data file sh-183-benchmark.json that accompanies this paper together with the programs used below. The enrolled tuple is

$$
f _ { 1 } = ( - 2 , - 3 , 0 ) , \qquad f _ { 2 } = ( - 3 , 3 , 3 , 3 , 2 ) , \qquad f _ { 3 } = ( - 1 , - 3 , 3 , - 4 , 2 , 2 , - 4 ) ,
$$

$$
f _ { 4 } = ( 3 , 0 , - 1 , - 3 , 1 , - 4 , - 4 , - 4 , 4 ) , \qquad f _ { 5 } = ( - 4 , 2 , - 1 , 2 , - 4 , 4 , - 1 , 3 , 3 , 4 , - 1 ) ,
$$

$$
f _ { 6 } = ( 1 , - 1 , - 1 , 3 , 0 , - 4 , 2 , 4 , - 3 , - 2 , 0 , - 3 , 1 ) ,
$$

in coordinates with respect to these bases, and $\| c \| ^ { 2 } = 1 6 3 3 3 6 1 / 4 5 0 4 5$ , so $\| c \| < 6 . 0 2 1 7$ . The decoy $c ^ { \prime }$ has coordinates in $1 0 ^ { - 2 0 } \mathbb { Z } ;$ they are listed in the data file.

Example 7.4 (Certified $L = 6$ benchmark). The tuples $c , c ^ { \prime } \in U _ { 6 }$ above satisfy the following.

(1) For every component $\varphi$ of $\Phi _ { 6 } , | \varphi ( c ^ { \prime } ) - \varphi ( c ) | \le 2 \cdot 1 0 ^ { - 1 8 } | \varphi ( c ) |$ . In particular $| \varphi ( c ^ { \prime } ) - \varphi ( c ) | \leq$ $1 0 ^ { - 1 4 } | \varphi ( c ) |$

(2) $2 1 / 4 \le \delta ( c , c ^ { \prime } ) \le 5 . 3 6 4 3$ . In particular $\delta ( c , c ^ { \prime } ) \geq 0 . 8 7 \| c \| .$

(3) $4 . 5 \leq \delta ( c , \tau c ) \leq 4 . 8 0 4 8$ and $4 . 3 \le \delta ( \tau c , c ^ { \prime } ) \le 4 . 6 0 3 5$ (4) Let $\varepsilon = 1 0 ^ { - 1 4 } | \Phi _ { 6 } ( c ) |$ , and let $K \subset U _ { 6 }$ be compact and contain c and $c ^ { \prime } .$ . Then $r _ { \Phi _ { 6 } } ^ { \varepsilon } ( c ; K ) \ge 2 1 / 4$

(5) $s _ { c } > - 2 0$ and $s _ { c ^ { \prime } } > - 2 0$ on $S ^ { 2 }$ , where $s _ { c } = f _ { 1 } + \cdot \cdot \cdot + f _ { 6 }$ for $c = ( f _ { 1 } , \ldots , f _ { 6 } )$ and $s _ { c ^ { \prime } } = f _ { 1 } ^ { \prime } + \cdot \cdot \cdot + f _ { 6 } ^ { \prime } ~ \mathrm { f o r } ~ c ^ { \prime } = ( f _ { 1 } ^ { \prime } , \cdot \cdot \cdot , f _ { 6 } ^ { \prime } )$

Computation. (1) The components are evaluated at c and $c ^ { \prime }$ in exact arithmetic over $\mathbb { Q } ( i )$ with the invariants $J _ { 4 } , J _ { 6 } , J _ { 1 0 }$ computed from the Clebsch covariants of $F _ { 3 }$ . All 32 values are real and nonzero, and the largest relative diference is $1 . 5 4 \cdot 1 0 ^ { - 1 8 }$

(2) The norms are computed from the exact Gram matrices of the bases. The upper bound is the exact value of $\| c - R _ { 0 } \cdot c ^ { \prime } \|$ at the rational rotation $R _ { 0 } ~ = ~ R ( q _ { 0 } )$ of the quaternion $q _ { 0 } = ( 1 4 2 7 / 9 0 5 0 7 4$ , 290689/653469, $- 2 9 2 4 6 7 / 3 3 0 4 2 2 , \ 1 )$ , normalized; $\| c - R _ { 0 } \cdot c ^ { \prime } \| ^ { 2 }$ is a rational number smaller than $2 8 . 7 7 5 5 6 < 5 . 3 6 4 3 ^ { 2 }$ . The lower bound is Cor. $7 . 3 ( 2 )$ . Each cube $[ - 1 , 1 ] ^ { 3 }$ is divided into $8 ^ { 3 }$ boxes. For a box B the value $\| c - S _ { k } ( u _ { B } ) \cdot c ^ { \prime } \| ^ { 2 } = \| c \| ^ { 2 } + \| c ^ { \prime } \| ^ { 2 } - 2 \langle c , S _ { k } ( u _ { B } ) \cdot c ^ { \prime } \rangle$ is enclosed by evaluating $\langle c , S _ { k } ( u _ { B } ) \cdot c ^ { \prime } \rangle$ as an explicit polynomial in the nine entries of $S _ { k } ( u _ { B } )$ in interval arithmetic, in IEEE double precision with rounding to nearest and with outward rounding by one unit in the last place after every operation. With $\overline { { K } } = 2 2 . 8 0 6 > K ^ { \prime }$ and $h _ { B }$ the half-diagonal of B, a box with $\ell _ { B } - ( \pi / \sqrt { 2 } ) \overline { { K } } h _ { B } \geq 2 1 / 4$ is discarded, and every other box is divided into eight. All boxes are discarded after at most eight levels, with 990 624 cente evaluations in total. The ratio follows from $( 2 1 / 4 ) ^ { 2 } > 0 . 8 7 ^ { 2 }$ · 1633361/45045.

(3) The same procedure, applied to the pairs $( c , \tau c )$ and $( \tau c , c ^ { \prime } )$ with the thresholds $9 / 2$ and $4 3 / 1 0$ , discards all boxes after 249 744 and 221 200 evaluations. The upper bounds come from the rational quaternions listed in the data file.

(4) By (1), $| \Phi _ { 6 } ( c ^ { \prime } ) - \Phi _ { 6 } ( c ) | \le \varepsilon$ . By (2), c<sup>′</sup> ∈/ SO(3) · c and $\delta ( c , c ^ { \prime } ) \geq 2 1 / 4$

(5) Write $p = v / | v |$ with v on a face of the cube $[ - 1 , 1 ] ^ { 3 }$ , so that $\begin{array} { r } { s _ { c } ( p ) = \sum _ { l } f _ { l } ( v ) / | v | ^ { l } } \end{array}$ . Each face is covered by boxes, and $s _ { c }$ is enclosed on each box by interval arithmetic as in (2), with boxes subdivided until the lower bound exceeds −20.

The distance bounds distinguish $c ^ { \prime }$ from both c and τ c. The mirror match itself is an exact algebraic identity by Prop. 6.2. The rational inequality in part (1) is a finite-tolerance match: it does not establish that $c ^ { \prime }$ belongs to the exact fiber or to a particular connected component of that fiber.

We found $c ^ { \prime }$ by numerical continuation along the descriptor level set, moving in the quotient tangent space of Thm. 6.1, which has dimension 13 at a free maximal-rank point, with Gauss– Newton correction of the descriptor residual, followed by Newton iteration in 60-digit arithmetic and rounding of the coordinates to $1 0 ^ { - 2 0 } \mathbb { Z }$ . The continuation is not part of the certificate; parts (1)–(5) are verified from the final rational pair alone.

Part (1) is a componentwise relative check. No component of $\Phi _ { 6 } ( c )$ vanishes, so it is the discrepancy of Sec. 2 with $W$ the diagonal matrix of the reciprocals $1 / | \varphi ( c ) |$ and with the maximum norm in place of the Euclidean norm. A matcher on $\Phi _ { 6 }$ with componentwise relative tolerance $1 0 ^ { - 1 4 }$ therefore accepts $c ^ { \prime }$ , and the Euclidean tolerance ε of part (4) is a consequence. The band powers of c and $c ^ { \prime }$ agree to the same precision, so their energies agree to that precision in every band. By (5), the radial functions $2 4 + s _ { c }$ and $2 4 + s _ { c ^ { \prime } }$ exceed 4 on the sphere. Fig. 2 shows both as radial surfaces with constant band $f _ { 0 } = 2 4$

Remark 7.5 (Certification status). Part (1) is established in exact rational arithmetic. Parts (2), (3) and (5) are established by interval arithmetic in IEEE double precision in round-to-nearest mode, with every elementary result widened by one unit in the last place in each direction; this covers the rounding error of correctly rounded addition, subtraction, multiplication, division and square root. The constant $\pi / { \sqrt { 2 } }$ is enclosed from above by the double-precision quotient of the nearest doubles to π and ${ \sqrt { 2 } } .$ , increased by two units in the last place; this exceeds $\pi / { \sqrt { 2 } }$ because the nearest double to π lies below π by less than $1 . 3 \cdot 1 0 ^ { - 1 6 }$ and the nearest double to $\sqrt { 2 }$ lies above $\sqrt { 2 }$ by less than $1 0 ^ { - 1 6 }$ . The entries of $S _ { k } ( u _ { B } )$ have absolute value at most 1 and all coeficients are bounded, so no overflow occurs, and results in the subnormal range remain correctly rounded. The certificate depends on the correctness of this floating-point arithmetic and of the interval routines in the accompanying program certify.py, and on nothing else. The exact value of $\| c - R _ { 0 } \cdot c ^ { \prime } \| ^ { 2 }$ , the rational quaternions, the box counts and the Lipschitz constants are recorded by the program.

The factor 0.87 is normalized by the nonconstant coeficient tuple. For the rendered model with $\rho _ { 0 } = 2 4 $

$$
\| \rho _ { c } \| _ { L ^ { 2 } ( d \mu ) } = \sqrt { 5 7 6 + 1 6 3 3 3 6 1 / 4 5 0 4 5 } \approx 2 4 . 7 4 4 .
$$

The distance interval $[ 5 . 2 5 , 5 . 3 6 4 3 ]$ corresponds to approximately 0.2122–0.2168 of that full radial norm. Both normalizations are radial $L ^ { 2 }$ quantities. They are neither Hausdorf distances nor violations of a dimensional tolerance.

![](images/71db8952388d0119fd7cccf88694da271fcbefa6a8b540e0cfb2a1796849fe77.jpg)  
Figure 2. The enrolled surface c (left) and the decoy c<sup>′</sup> (right), drawn with $f _ { 0 } = 2 4 ; c ^ { \prime }$ is rotated by the rotation $R _ { 0 }$ of Ex. 7.4(2). All 32 components of $\Phi _ { 6 }$ agree within relative tolerance $2 \cdot 1 0 ^ { - 1 8 }$ ; see Rem. 7.5.

## 8. Task-specific certification and disclosure

A shape collision is not necessarily an error for every task. An inspection quantity can be constant on a fiber that contains many noncongruent shapes. Conversely, a shape descriptor can fail to determine geometry while revealing an attribute exactly. The distinction can be stated without choosing a classifier.

Definition 8.1. Let T: $U _ { L } \ \to \ \mathbb { R }$ be a continuous rotation-invariant task quantity. For a nonempty set $\mathcal { F } = \mathcal { F } _ { \Phi , W } ^ { \varepsilon } ( y ; K )$ , define

$$
\begin{array} { c } { { T _ { - } ( y ) = \displaystyle \operatorname* { m i n } _ { x \in \mathcal { F } } T ( x ) , \qquad T _ { + } ( y ) = \displaystyle \operatorname* { m a x } _ { x \in \mathcal { F } } T ( x ) , } } \\ { { \ A _ { \Phi , W , T } ^ { \varepsilon } ( y ; K ) = T _ { + } ( y ) - T _ { - } ( y ) . } } \end{array}
$$

We call $[ T _ { - } ( y ) , T _ { + } ( y ) ]$ the task interval and $A _ { \Phi , W , T } ^ { \varepsilon } ( y ; K )$ its width.

The interval is an enclosure of the attainable task values; these values need not fill it when the consistent set is disconnected. For a requirement $T \leq b ,$ an established upper bound $t _ { \mathrm { u p } } \geq T _ { + } ( y )$ with $\boldsymbol { t } _ { \mathrm { u p } } \le \boldsymbol { b }$ certifies compliance. An established lower bound $t _ { \mathrm { l o w } } \le T _ { - } ( y )$ with $\smash { t _ { \mathrm { l o w } } > b }$ certifies noncompliance. To demonstrate ambiguity one needs two feasible shapes with validated task values on opposite sides of b. A loose enclosing interval that crosses b does not establish such a pair.

Proposition 8.2 (Task information). With the notation of Def. 8.1, the following hold.

(1) The smallest worst-case error of a scalar estimate based on y is

$$
\operatorname* { m i n } _ { t \in \mathbb { R } } \operatorname* { m a x } _ { x \in \mathcal { F } } \lvert t - T ( x ) \rvert = \frac { T _ { + } ( y ) - T _ { - } ( y ) } { 2 } .
$$

The midpoint of the task interval attains this value.

(2) $I f \left| T ( x ) - T ( x ^ { \prime } ) \right| \leq M _ { T } \delta ( x , x ^ { \prime } )$ for all $x , x ^ { \prime } \in K$ , then

$$
A _ { \Phi , W , T } ^ { \varepsilon } ( y ; K ) \leq M _ { T } D _ { \Phi , W } ^ { \varepsilon } ( y ; K ) .
$$

In particular, for $y = \Phi ( c )$ ，

$$
\operatorname* { m a x } _ { x \in \mathcal { F } } \lvert T ( x ) - T ( c ) \rvert \le M _ { T } r _ { W \Phi } ^ { \varepsilon } ( c ; K ) .
$$

(3) At exact tolerance, every task interval is a singleton $i f$ and only $i f$ there is a map $g \colon \Phi ( K ) \to \mathbb { R }$ such that $T | _ { K } = g \circ \Phi | _ { K }$ . In that case $g$ is continuous.

Proof. For (1), both endpoints are attained, so every estimate has error at least half their diference. Every attainable value lies between them, so their midpoint has error at most half their diference. For (2), apply the assumed inequality to a maximizing pair and, for the second assertion, to c and a feasible point. For (3), singleton intervals say exactly that T is constant on each fiber in $K$ , making $g ( \Phi ( x ) ) = T ( x )$ well-defined. Conversely, such a factorization makes the intervals singletons. The continuous surjection $\Phi | _ { K }$ from a compact space onto the Hausdorf space $\Phi ( K )$ is a quotient map, so continuity of $T | _ { K }$ gives continuity of $g .$ □

For exact descriptor collisions, every deterministic predictor h ◦ Φ takes the same value at both shapes. Thus if the required task values difer, no such predictor can be correct at both. At positive tolerance, closeness of two descriptors alone does not imply that an arbitrary predictor assigns them the same label. The corresponding impossibility statement concerns a common observation $y \colon$ if both shapes lie in its consistent set, no decision based only on that observation can be guaranteed correct for both under the stated error model.

A local test uses the same Jacobian calculation as the fiber theorem.

Proposition 8.3 (Diferential test for task ambiguity). Let $L \ge 2$ , let $\Phi \colon U _ { L } \to \mathbb { R } ^ { N }$ be a descriptor, and let $U ^ { \prime }$ and $M _ { c }$ be as in Thm. $4 . 2 .$ . Let $c \in U ^ { \prime }$ , and let T : $U _ { L } \to \mathbb { R }$ be $a ~ C ^ { 1 }$ rotation-invariant function. $I f$

$$
\mathrm { r a n k } d ( \Phi , T ) _ { c } > \mathrm { r a n k } d \Phi _ { c } ,
$$

then every neighborhood $o f c$ in the regular fiber $M _ { c }$ contains points $c _ { - }$ and $c _ { + }$ with

$$
\begin{array} { r } { T ( c _ { - } ) < T ( c ) < T ( c _ { + } ) . } \end{array}
$$

The same conclusion holds inside an admissible set K whenever $K$ contains a neighborhood of c in that fiber.

Proof. The rank inequality is equivalent to the existence of $v \in$ ker $d \Phi _ { c }$ with $d T _ { c } ( v ) \neq 0$ . By the constant-rank theorem, ker $d \Phi _ { c }$ is the tangent space to the regular fiber. Choose a $C ^ { 1 }$ curve in that fiber through c with derivative v. The derivative of T along the curve is nonzero at $c ,$ so suficiently small parameters of opposite signs give the two inequalities. □

The rank test detects first-order variation only. It does not measure the size of the variation, and at a single point it is suficient but not necessary. It also cannot exclude a jump between disconnected components. For example, let c lie in the free maximal-rank locus of $\Phi _ { 1 6 }$ . The diferential of every invariant annihilates the three-dimensional tangent space of the orbit, so adjoining $\chi$ does not increase the rank at c. Yet χ takes opposite signs at c and $\tau c ,$ which $\Phi _ { 1 6 }$ does not distinguish. Full local rank therefore does not imply that a task quantity is determined globally.

## 9. Volume disclosed by the even spectral pool

$$
s _ { c } ( p ) = \sum _ { l = 1 } ^ { L } f _ { l } ( p ) , \qquad \rho _ { c } ( p ) = \rho _ { 0 } + s _ { c } ( p ) ,
$$

Fix a known constant radial term $\rho _ { 0 } > 0$ and use the round probability measure $d \mu = d \omega / ( 4 \pi )$ on $S ^ { 2 }$ . Put

and suppose $\rho _ { c } > 0$ . The enclosed radial solid is $\{ t p : p \in S ^ { 2 } , \ 0 \leq t \leq \rho _ { c } ( p ) \}$ . Its volume is

$$
V ( c ) = \frac { 4 \pi } { 3 } \int _ { S ^ { 2 } } \rho _ { c } ( p ) ^ { 3 } d \mu ( p ) .\tag{2}
$$

For a positive continuous radial function $\rho$ on $S ^ { 2 }$ we write $\begin{array} { r } { V ( \rho ) ~ = ~ \frac { 4 \pi } { 3 } \int _ { S ^ { 2 } } \rho ^ { 3 } d \mu } \end{array}$ , so that $V ( c ) = V ( \rho _ { c } )$ . This is the volume enclosed by the modeled surface. For a hollow object it is neither the material volume nor a measure of mass.

Theorem 9.1 (Exact volume disclosure). For $L \ \geq \ 2$ and fixed $\rho _ { 0 }$ , there is an explicitly computable vector $\beta ( \rho _ { 0 } )$ such that every $c \in U _ { L }$ with $\rho _ { c } > 0$ satisfies

$$
V ( c ) = \frac { 4 \pi } { 3 } \rho _ { 0 } ^ { 3 } + \beta ( \rho _ { 0 } ) ^ { \top } \Phi _ { L } ( c ) .\tag{3}
$$

The entries of $\beta$ corresponding to $J _ { 4 } , J _ { 6 } , J _ { 1 0 }$ are zero. The same identity holds for $\Phi _ { L } ^ { + }$ with zero coeficients on the odd bispectra. Consequently, at fixed $\rho _ { 0 ; }$ , every exact fiber intersected with the set $\{ c : \rho _ { c } > 0 \}$ has zero volume-task width, irrespective of its shape dimension or diameter.

Proof. Every $f _ { l }$ has mean zero. Orthogonality gives

$$
\int \rho _ { c } ^ { 3 } d \mu = \rho _ { 0 } ^ { 3 } + 3 \rho _ { 0 } \sum _ { l } \| f _ { l } \| _ { L ^ { 2 } ( d \mu ) } ^ { 2 } + \sum _ { i , j , k = 1 } ^ { L } \int f _ { i } f _ { j } f _ { k } d \mu .
$$

For each $l ,$ the quadratic integral is a nonzero fixed multiple $a _ { l } P _ { l }$ . A triple integral vanishes when the sum of the degrees is odd, by parity, or when the triangle condition fails. For an ordered even triangle triple, the invariant trilinear space is one-dimensional: in the Clebsch– Gordan decomposition, $\mathcal { H } _ { k }$ occurs with multiplicity one in $\mathcal { H } _ { i } \otimes \mathcal { H } _ { j }$ . The triple integral and the iterated transvectant are proportional. The Gaunt formula shows that the former is nonzero for these triples; see [12, Eq. (34.3.5) and §34.3(vii)]. Thus there are nonzero constants $b _ { i j k }$ with $\begin{array} { r } { \int f _ { i } f _ { j } f _ { k } d \mu = b _ { i j k } B _ { i j k } } \end{array}$ . Collecting the ordered triples gives

$$
V ( c ) = \frac { 4 \pi } { 3 } \left( \rho _ { 0 } ^ { 3 } + 3 \rho _ { 0 } \sum _ { l = 1 } ^ { L } a _ { l } P _ { l } ( c ) + \sum _ { ( i , j , k ) \in A ( L ) } m _ { i j k } b _ { i j k } B _ { i j k } ( c ) \right) ,\tag{4}
$$

where $m _ { i j k } = 1 , 3 .$ , 6 according as all three indices, exactly two indices, or no indices coincide. This gives the stated vector and shows that the other invariants are unnecessary for volume. □

9.1. Normalization and exact coeficient computation. The constants must match the descriptor normalization; norm powers cannot be substituted directly for the transvectant powers. For the null-cone parametrization of Sec. $^ { 7 , }$ let $\mathcal { P } _ { l }$ be the Legendre polynomial with $\mathcal { P } _ { l } ( 1 ) = 1$ . The solid harmonic extending $\mathcal { P } _ { l } ( p _ { z } )$ restricts to

$$
q _ { l } ( z _ { 0 } , z _ { 1 } ) = ( - 1 ) ^ { l } { \binom { 2 l } { l } } z _ { 0 } ^ { l } z _ { 1 } ^ { l } .
$$

Indeed only the leading term in $\mathcal { P } _ { l }$ survives restriction to $x ^ { 2 } + y ^ { 2 } + z ^ { 2 } = 0$ , and $z = - 2 z _ { 0 } z _ { 1 }$ Consequently

$$
\begin{array} { c } { \displaystyle a _ { l } = \frac { 1 } { ( 2 l + 1 ) ( - 1 ) ^ { l } \binom { 2 l } { l } } , } \\ { \displaystyle b _ { i j k } = \frac { G _ { i j k } } { \big ( ( q _ { i } , q _ { j } ) _ { i + j - k } , q _ { k } \big ) _ { 2 k } } , \qquad G _ { i j k } = \frac { 1 } { 2 } \int _ { - 1 } ^ { 1 } \mathcal { P } _ { i } ( t ) \mathcal { P } _ { j } ( t ) \mathcal { P } _ { k } ( t ) d t . } \end{array}\tag{5}
$$

The denominator is nonzero for $( i , j , k ) \in \mathcal { A } ( L )$ . With $s = ( i + j + k ) / 2$ , the zonal Gaunt coeficient is

$$
G _ { i j k } = \frac { ( s ! ) ^ { 2 } ( 2 s - 2 i ) ! ( 2 s - 2 j ) ! ( 2 s - 2 k ) ! } { ( s - i ) ! ^ { 2 } ( s - j ) ! ^ { 2 } ( s - k ) ! ^ { 2 } ( 2 s + 1 ) ! } ,
$$

by [12, Eq. (34.3.5) and $\ S 3 4 . 3 ( \mathrm { v i i } ) ]$ . Thus the calibration uses rational arithmetic only. For example,

$$
a _ { 1 } = - \frac { 1 } { 6 } , \qquad a _ { 2 } = \frac { 1 } { 3 0 } ,
$$

$$
b _ { 1 1 2 } = \frac { 1 } { 3 0 } , \qquad b _ { 2 2 2 } = - \frac { 1 } { 1 0 5 } .
$$

The accompanying program volume from pool.py implements these formulas, evaluates volume as an exact rational multiple of $\pi$ when the supplied descriptor is rational, and propagates componentwise absolute error bounds. It does not test realizability of an arbitrary descriptor or validity of a supplied measurement-error bound.

## 9.2. Descriptor and radial-model error.

Corollary 9.2. Let $L \geq 2$ , let $c \in U _ { L }$ with $\rho _ { c } > 0$ , let $W$ be a nonsingular matrix, and put $\begin{array} { r } { \widehat { V } ( y ) = \frac { 4 \pi } { 3 } \rho _ { 0 } ^ { 3 } + \beta ( \rho _ { 0 } ) ^ { \top } y . \ I f \left| W ( \Phi _ { L } ( c ) - y ) \right| \leq \varepsilon _ { 1 } } \end{array}$ , then

$$
| V ( c ) - \widehat { V } ( y ) | \leq | W ^ { - \top } \beta ( \rho _ { 0 } ) | \varepsilon .\tag{6}
$$

If instead each component satisfies $| \Phi _ { L , j } ( c ) - y _ { j } | \le e _ { j }$ , the right-hand side may be replaced by $\sum _ { j } | \beta _ { j } | e _ { j }$

Suppose also that the actual radial function $\rho$ and the modeled radial function $\rho _ { c }$ satisfy $0 < \rho , \rho _ { c } \le \rho _ { \mathrm { m a x } }$ and $\| \rho - \rho _ { c } \| _ { L ^ { 2 } ( d \mu ) } \leq \gamma$ . Then

$$
\begin{array} { r } { \vert V ( \rho ) - \widehat V ( y ) \vert \leq \vert W ^ { - \top } \beta ( \rho _ { 0 } ) \vert \varepsilon + 4 \pi \rho _ { \operatorname* { m a x } } ^ { 2 } \gamma . } \end{array}\tag{7}
$$

The componentwise version holds with the same additional term.

Proof. Subtract Eq. (3) from the definition of $\widehat { V }$ and apply Cauchy–Schwarz to

$$
( W ^ { - \top } \beta ) ^ { \top } W ( \Phi _ { L } ( c ) - y ) .
$$

The componentwise estimate is the triangle inequality. For $a , b \in [ 0 , \rho _ { \operatorname* { m a x } } ]$ we have

$$
| a ^ { 3 } - b ^ { 3 } | \leq 3 \rho _ { \mathrm { m a x } } ^ { 2 } | a - b | .
$$

Therefore Eq. (2) and the fact that $\mu$ is a probability measure give

$$
| V ( \rho ) - V ( c ) | \leq 4 \pi \rho _ { \operatorname* { m a x } } ^ { 2 } \| \rho - \rho _ { c } \| _ { L ^ { 1 } ( d \mu ) } \leq 4 \pi \rho _ { \operatorname* { m a x } } ^ { 2 } \gamma .
$$

Another triangle inequality proves Eq. (7).

An uncertain constant term must be accounted for: its contribution can be included in the radial error $\gamma$ or bounded separately. Similarly, a descriptor of a truncated tuple determines the modeled volume. It determines the physical volume only up to an estimate of the truncation and acquisition errors.

9.3. An exact clearance test with identical volume. The following example exhibits a task conflict in closed form, without a numerical search. The reference origin is fixed; no centroid normalization is imposed.

Proposition 9.3. Let $p = ( x , y , z ) \in S ^ { 2 }$ and define

$$
\begin{array} { l } { { \rho _ { A } ( p ) = 3 + \frac { 2 } { 3 } x + \frac { 1 } { 3 } y + \frac { 2 } { 3 } z + x ^ { 2 } - z ^ { 2 } , } } \\ { { \rho _ { B } ( p ) = 3 + y + x ^ { 2 } - z ^ { 2 } . } } \end{array}
$$

Both radial functions are at least 1. Their coeficient tuples $c _ { A } , c _ { B } \in U _ { 2 }$ , taken with $\rho _ { 0 } = 3 { } _ { ; }$ have exactly the same even pool $\Phi _ { 2 }$ , and their enclosed volumes are both $2 1 6 \pi / 5$ . Nevertheless, for $T _ { \mathrm { o u t } } ( c ) = \operatorname* { m a x } _ { p \in S ^ { 2 } } \rho _ { c } ( p )$

$$
T _ { \mathrm { o u t } } ( c _ { B } ) = \frac { 1 7 } { 4 } < \frac { 9 } { 2 } < \frac { 1 4 } { 3 } \leq T _ { \mathrm { o u t } } ( c _ { A } ) .
$$

Thus the same exact descriptor supports incompatible decisions for containment in the origincentered ball of radius $9 / 2$ , while it determines the enclosed volume exactly.

Proof. Both have $Q = \mathrm { d i a g } ( 1 , 0 , - 1 )$ , with $a _ { A } = ( 2 / 3 , 1 / 3 , 2 / 3 )$ and $a _ { B } = ( 0 , 1 , 0 )$ . Then

$$
\begin{array} { c } { { | a _ { A } | ^ { 2 } = | a _ { B } | ^ { 2 } = 1 , \quad \mathrm { t r } Q ^ { 2 } = 2 , \quad \mathrm { d e t } Q = 0 , } } \\ { { { a } _ { A } ^ { \top } Q a _ { A } = a _ { B } ^ { \top } Q a _ { B } = 0 . } } \end{array}
$$

These are the four invariants of $\Phi _ { 2 }$ , up to its fixed nonzero normalization factors. In the null-cone convention of Sec. 7, the ordered values $\left( P _ { 1 } , P _ { 2 } , B _ { 1 1 2 } , B _ { 2 2 2 } \right)$ are $( - 2 , 8 , 0 , 0 )$ for both tuples. Also $| \langle a , p \rangle | \leq 1$ for $a \in \{ a _ { A } , a _ { B } \}$ and $| p ^ { \top } Q p | \leq 1$ , so each radius is at least $3 - 1 - 1 = 1$ . Applying Eq. (4) with the constants in Eq. (5) gives

$$
V = { \frac { 4 \pi } { 3 } } \left( 2 7 + 9 \left( { \frac { 1 } { 3 } } + { \frac { 4 } { 1 5 } } \right) \right) = { \frac { 2 1 6 \pi } { 5 } } .
$$

For $\rho _ { B }$ , use $x ^ { 2 } - z ^ { 2 } \leq 1 - y ^ { 2 }$ to obtain

$$
\rho _ { B } \leq 4 + y - y ^ { 2 } \leq \frac { 1 7 } { 4 } .
$$

Equality is attained at $y = 1 / 2 , z = 0 , x = \sqrt { 3 } / 2$ . For $\rho _ { A }$ , evaluation at $p = ( 1 , 0 , 0 )$ gives $1 4 / 3$ The displayed strict inequalities follow. □

The two objects cannot be congruent even under $\mathrm { O ( 3 ) }$ , since $T _ { \mathrm { o u t } }$ is invariant under that group and their values difer. The failure is therefore not solely a handedness ambiguity. At the common descriptor value the task interval for any admissible K containing both tuples has width at least $1 4 / 3 - 1 7 / 4 = 5 / 1 2$ . By Prop. 8.2, every scalar estimator based on that value has worst-case outer-radius error at least $5 / 2 4$ on these two objects. No comparable error is forced for volume, whose exact task width is zero.

## 10. Admissible shapes and a representation audit

10.1. Origin, scale, and the feasible domain. The set K records geometric side information for both recognition and disclosure. One concrete model, with fixed $\rho _ { 0 }$ , is

$$
K = \{ c \in U _ { L } : \| c \| \leq M , \quad \rho _ { \operatorname* { m i n } } \leq \rho _ { 0 } + s _ { c } ( p ) \leq \rho _ { \operatorname* { m a x } } { \mathrm { ~ f o r ~ a l l ~ } } p \in S ^ { 2 } \} ,\tag{8}
$$

where $0 < \rho _ { \mathrm { m i n } } < \rho _ { \mathrm { m a x } }$ . It is compact and rotation-invariant. It is semialgebraic, because each harmonic is the restriction of a polynomial and the quantified conditions on $p \cdot p = 1$ can be eliminated over the reals; see [5]. Positivity of this kind is an assumption on the truncated model: the truncation of a positive function need not be positive.

A declared reference origin and a centroid convention are diferent assumptions. For the enclosed solid, the volume centroid is at the origin precisely when

$$
\int _ { S ^ { 2 } } p \rho _ { c } ( p ) ^ { 4 } d \mu ( p ) = 0 ,
$$

since its unnormalized first moment is $\textstyle { \frac { 1 } { 4 } } \int p \rho _ { c } ^ { 4 } d \omega$ . A surface-area centroid imposes a diferent condition. Recentering changes the radial function and generally its harmonic coeficients. The exact example of Prop. 9.3 uses the declared origin; it is not a counterexample to an unspecified recentered pipeline.

More generally, if equality constraints restrict coeficients to a smooth rotation-invariant manifold M of dimension d on which the action is free, the local regular-fiber dimension is

$$
d - 3 - \mathrm { r a n k } \big ( d \Phi _ { c } | _ { T _ { c } , M } \big ) ,
$$

where the restricted rank is assumed constant in a neighborhood. This follows from the same constant-rank and quotient-manifold argument as Thm. 4.2. The ambient count $n - 3 - r$ does not carry over after such a normalization. Physical size must also be retained when the task depends on dimensions; removing $\mathcal { H } _ { \mathrm { 0 } }$ for orbit analysis does not make it irrelevant to inspection.

## 10.2. A computable geometric bound.

Proposition 10.1. For the weighted norm of Sec. 2, put

$$
\Gamma _ { L , w } = \left( \sum _ { l = 1 } ^ { L } \frac { 2 l + 1 } { w _ { l } } \right) ^ { 1 / 2 } .
$$

Then $\| s _ { c } \| _ { \infty } \leq \Gamma _ { L , w } \| c \|$ for every $c \in U _ { L }$ . At a fixed constant term, the outer-radius task satisfies

$$
| T _ { \mathrm { o u t } } ( c ) - T _ { \mathrm { o u t } } ( c ^ { \prime } ) | \le \Gamma _ { L , w } \delta ( c , c ^ { \prime } )
$$

for all $c , c ^ { \prime } \in U _ { L }$ . For $w _ { l } = 1 , \Gamma _ { L , w } = \sqrt { L ^ { 2 } + 2 L }$

Proof. Choose an orthonormal basis of each band for $L ^ { 2 } ( d \mu )$ . The addition theorem makes the sum of its squared values at any point equal to $2 l + 1$ . Weighted Cauchy–Schwarz therefore gives $| s _ { c } ( p ) | \leq \Gamma _ { L , w } \Vert c \Vert$ . Apply this to $c - R \cdot c ^ { \prime }$ and use invariance of the maximum under rotations. Minimizing over R proves the task bound. □

In particular, $\rho _ { 0 } - \Gamma _ { L , w } M \geq \rho _ { \operatorname* { m i n } }$ is a simple suficient positivity condition on a coeficient ball. It is not necessary. The task bound yields conservative clearance certificates from an upper bound on the shape radius; optimizing the task itself can be substantially tighter.

10.3. An error budget for the actual representation. A numerical matching tolerance is not a sensor specification. Suppose c and a fitted tuple bc belong to a convex bounded region on which $\| W d \Phi _ { x } \| _ { \mathrm { o p } } \leq M _ { \Phi , W }$ , with the domain norm fixed as in Sec. 2. If $\| { \widehat { c } } - c \| \leq \sigma$ and the stored value satisfies $| W ( y - \Phi ( \widehat { c } ) ) | \leq \eta$ , integration along the line segment gives

$$
| W ( y - \Phi ( c ) ) | \leq M _ { \Phi , W } \sigma + \eta .
$$

Thus $\varepsilon = M _ { \Phi , W } \sigma + \eta$ is a suficient descriptor-error budget under these assumptions. Truncation error in a geometric quantity must still be propagated separately, as in Cor. 9.2. If the acquisition bounds hold only with a stated probability, the resulting certificate holds with that probability and is not unconditional.

10.4. Use in independent evaluation. The framework is intended primarily for evaluators who select or test geometric representations, not for operators who apply a radius to an unspecified field observation. The input is one documented representation, one admissible shape model, a calibrated error budget, and a task requirement. The output is a task certificate, a verified conflicting pair, or an unresolved case. A failure to find a conflicting shape is not a certificate of its absence. Similarly, a numerical distance computed at one aligning rotation bounds the orbit distance from above only; Cor. 7.3 gives the lower bound needed to certify geometric separation.

One such setting is an independent validation study of scan-based identification or inspection of equipment or calibration objects, for instance in defense or aerospace programs. Such a study is relevant only if the representation under test actually discards geometry in the manner modeled here. A comparison against complete geometry does not automatically inherit a collision of $\Phi _ { L }$

A focused validation would first reproduce the exact quadrupole formulas and Prop. 9.3. It would then use a controlled collection of fully observed radial objects with independently measured reference geometry, evaluate band powers, the even pool, and a task-augmented representation under the same acquisition and preprocessing conditions, and report task-error bounds, conflicting decisions, the fraction of unresolved cases, computation time, and descriptor storage. The high-band benchmark of Sec. 7 is a certified calibration case for such a comparison, and Sec. 11 applies the same audit to a space object.

For shared records, the same audit specifies what is released: at fixed known $\rho _ { 0 }$ , the modeled enclosed volume is recoverable by Thm. 9.1, and multiple records about one tuple must be assessed jointly by Cor. 4.4. An application need not demand complete shape reconstruction. It may be better served by retaining one additional task measurement than by enlarging a global invariant pool. That decision depends on the required task and on which attributes may be disclosed.

## 11. A case study: asteroid (101955) Bennu

Rotation-invariant descriptors are natural for a tumbling object of unknown attitude. We apply the framework to the radar shape model of [13] for the near-Earth asteroid (101955) Bennu. It is archived in the Planetary Data System as the file data/101955bennu.tab of the bundle urn:nasa:pds:ast-bennu.radar.shape-model, version 1.1; see [14]. The model is a closed triangulated surface with 1348 vertices and 2692 facets. Its origin is the center of mass, and its axes are the principal axes. We use meters.

The polyhedral volume of the model is $V _ { \mathrm { m e s h } } = 6 . 2 2 6 5 6 \cdot 1 0 ^ { 7 } \mathrm { m ^ { 3 } }$ , and its equivalent-sphere diameter is 491.8 m. The facets are oriented so that the signed volume is positive. Every facet has outward normal ν with $\langle \nu , a \rangle > 0$ at its vertices ${ a ; }$ the smallest value of $\langle \nu , a \rangle / ( | \nu | | a | )$ is 0.667. So the origin lies in the open inner half-space of every facet plane, that is, in the interior of the kernel of the polyhedron, and the surface is star-shaped about the origin. As an independent check, the radial projection maps every facet onto a positively oriented spherical triangle, and the solid angles of these triangles sum to 4π; so the projection has degree one, and a regular direction is covered exactly once. The radial function $\rho$ is positive and Lipschitz. The vertices of the model lie at distances between 224.2 m and 289.8 m from the center of mass. Since |x| is convex, $\rho \le 2 8 9 . 8  { \mathrm { m } }$ on $S ^ { 2 }$

The constant term $\rho _ { 0 }$ and the bands $f _ { 1 } , \ldots , f _ { L }$ approximate the orthogonal projections of $\rho$ for the round probability measure; they are computed using a product Gauss rule with 200 × 400 nodes, without a validated quadrature bound, since $\rho$ is only piecewise smooth. For $L = 6$ the coordinates of $f _ { 1 } , \ldots , f _ { 6 }$ in the bases of Sec. 7 are rounded to $1 0 ^ { - 4 } \mathrm { m }$ , and $\rho _ { 0 }$ is rounded to

245.2830 m. The rounded tuple is denoted $c \in U _ { 6 } ;$ it is listed in the data file sh-183-bennu.json. All statements about $L = 6$ concern c and its radial model $\rho _ { c } = \rho _ { 0 } + s _ { c }$ . We have $\| c \| < 1 1 . 5 2 \mathrm { m }$
<table><tr><td>L</td><td>components</td><td> $V _ { \mathrm { p o o l } } ~ \mathrm { ( k m ^ { 3 } ) }$ </td><td> $| V _ { \mathrm { p o o l } } - V _ { L } | / V _ { \mathrm { m e s h } }$ </td><td> $( V _ { \mathrm { p o o l } } - V _ { \mathrm { m e s h } } ) / V _ { \mathrm { m e s h } }$ </td><td>γL (m)</td></tr><tr><td>2</td><td>4</td><td>0.061969</td><td> $1 . 1 \cdot 1 0 ^ { - 1 5 }$ </td><td>-0.477%</td><td>9.71</td></tr><tr><td>3</td><td>7</td><td>0.062009</td><td> $8 . 4 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.412%</td><td>9.01</td></tr><tr><td>4</td><td>13</td><td>0.062186</td><td> $3 . 6 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.127%</td><td>4.97</td></tr><tr><td>5</td><td>19</td><td>0.062204</td><td> $2 . 4 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.098%</td><td>4.32</td></tr><tr><td>6</td><td>29</td><td>0.062227</td><td> $< 1 0 ^ { - 1 6 }$ </td><td>-0.063 %</td><td>3.45</td></tr><tr><td>7</td><td>39</td><td>0.062232</td><td> $3 . 6 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.054%</td><td>3.18</td></tr><tr><td>8</td><td>54</td><td>0.062248</td><td> $3 . 6 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.029 %</td><td>2.31</td></tr><tr><td>9</td><td>69</td><td>0.062250</td><td> $3 . 6 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.026 %</td><td>2.17</td></tr><tr><td>10</td><td>90</td><td>0.062253</td><td> $1 . 2 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.021%</td><td>1.95</td></tr><tr><td>11</td><td>111</td><td>0.062254</td><td> $3 . 6 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.019%</td><td>1.88</td></tr><tr><td>12</td><td>139</td><td>0.062256</td><td> $4 . 8 \cdot 1 0 ^ { - 1 6 }$ </td><td>-0.016 %</td><td>1.73</td></tr></table>

Table 2. Volume of Bennu from the even pool. $\overline { { V _ { \mathrm { p o o l } } } }$ is the right-hand side of Eq. (4) evaluated on the powers and even bispectra; $V _ { L }$ is the volume of the truncated radial model; $\gamma _ { L } = \| \rho - \rho _ { 0 } - f _ { 1 } - \cdot \cdot \cdot - f _ { L } \| _ { L ^ { 2 } ( d \mu ) }$ is the truncation error of the radial function. The column “components” counts the powers and even bispectra.

Example 11.1 (Bennu). Let $c \in U _ { 6 }$ be the tuple of Bennu above, let $T _ { \mathrm { o u t } } ( x ) = \rho _ { 0 } + \operatorname* { m a x } _ { S ^ { 2 } } s _ { x }$ be the keep-out radius about the fixed reference origin, the center of mass of the radar model, and let $V ( x )$ be the volume of the radial model $\rho _ { 0 } + s _ { x }$ . There are tuples $d _ { + } , d _ { - } \in U _ { 6 }$ with coordinates in $1 0 ^ { - 2 0 } \mathbb { Z }$ such that the following hold.

(1) $\Phi _ { 6 } ( \tau c ) = \Phi _ { 6 } ( c )$ . The five odd bispectra of c are nonzero, and each changes sign under τ . Moreover $5 \leq \delta ( c , \tau c ) \leq 5 . 5 1 3 8 \mathrm { m }$

(2) For $2 \leq L \leq 1 2$ , the volume obtained from the even pool by Thm. 9.1 equals the volume of the truncated model. In floating-point arithmetic the two agree to relative precision $2 \cdot 1 0 ^ { - 1 5 }$ , and $V _ { \mathrm { p o o l } }$ difers from $V _ { \mathrm { m e s h } }$ by the amounts in Tab. 2. This part is numerical and is not certified.

(3) For every component $\varphi$ of $\Phi _ { 6 } , | \varphi ( d _ { \pm } ) - \varphi ( c ) | \le 4 \cdot 1 0 ^ { - 1 9 } | \varphi ( c ) |$ , and $| V ( d _ { \pm } ) - V ( c ) | \le$ $2 \cdot 1 0 ^ { - 1 6 } \mathrm { { m } ^ { 3 } }$

(4) The keep-out radii satisfy

$$
\begin{array} { r l } & { \mathrm { 2 7 2 . 1 6 1 } \leq T _ { \mathrm { o u t } } ( d _ { - } ) \leq 2 7 2 . 1 7 2 , } \\ & { \mathrm { 2 7 4 . 3 2 1 } \leq T _ { \mathrm { o u t } } ( c ) \leq 2 7 4 . 3 3 2 , } \\ & { \mathrm { 2 7 9 . 5 8 8 } \leq T _ { \mathrm { o u t } } ( d _ { + } ) \leq 2 7 9 . 5 9 9 , } \end{array}
$$

in meters. In particular $d _ { + }$ and d<sub>−</sub> lie neither in O(3) · c nor in each other’s O(3)-orbit. (5) $4 \leq \delta ( c , d _ { + } ) \leq 4 . 2 9 3 0 \mathrm { m }$

(6) $s _ { c } , s _ { d _ { + } }$ and $s _ { d _ { - } }$ exceed −40 m on $S ^ { 2 }$ . Hence each of the radial functions $\rho _ { 0 } + s _ { c } , \rho _ { 0 } + s _ { d _ { + } }$ and $\rho _ { 0 } + s _ { d }$ exceeds 205 m in every direction.

Computation. (1) The values of $\Phi _ { 6 }$ at c and τc are computed in exact arithmetic over Q(i) and coincide; this is also $\mathrm { P r o p . ~ 6 . 2 ( 1 ) }$ . The odd bispectra $B _ { 2 3 4 } , B _ { 2 4 5 } , B _ { 2 5 6 } , B _ { 3 4 6 } , B _ { 4 5 6 }$ are computed exactly at c and τ c. The distance bounds are obtained as in Ex. $7 . 4 ( 2 )$ , with 64 176 boxes, and the upper bound at a rational rotation listed in the data file.

(2) The identity Eq. (4) is exact. In floating-point arithmetic both sides agree to the relative accuracy in Tab. 2. The last two columns of the table are numerical: they combine the truncation error of the radial model with the quadrature error of the projection, and neither is enclosed by a validated bound. Cor. 9.2 with $\rho _ { \mathrm { m a x } } = 2 8 9 . 8$ m and $\gamma = \gamma _ { 6 }$ bounds the volume error at $L = 6$ by 5.8 % of $V _ { \mathrm { m e s h } } ;$ the actual error is 0.063 %.

(3) The tuples $d _ { \pm }$ are found by numerical continuation along the level set of $\Phi _ { 6 }$ through $c ,$ in the direction of the projection onto the quotient tangent space of the gradient of $\pm ( \int ( \rho _ { 0 } +$ $s _ { x } ) ^ { 2 4 } d \mu ) ^ { 1 / 2 4 }$ , followed by Newton iteration in 60-digit arithmetic and rounding. The relative diferences of the components are computed exactly, and are at most $3 . 7 \cdot 1 0 ^ { - 1 9 }$ . The volume bound is the componentwise estimate of Cor. 9.2, computed exactly.

(4) Each lower bound is the exact value of $\rho _ { 0 } + s _ { x }$ at a rational point of $S ^ { 2 }$ of the form $( 2 u , 2 v , \pm ( 1 - u ^ { 2 } - v ^ { 2 } ) ) / ( 1 + u ^ { 2 } + v ^ { 2 } )$ with $u , v \in \mathbb { Q }$ . Each upper bound is established by the interval cover of Ex. $7 . 4 ( 5 )$ applied to $- s _ { x }$ . The function $T _ { \mathrm { o u t } }$ is invariant under $\mathrm { O ( 3 ) }$ , which gives the last assertion.

(5) As in Ex. 7.4(2), with 43 392 boxes.

(6) As in Ex. 7.4(5).

![](images/983f6848d1224119b045df4ef5143e6bc4518a7b0a49dab404b5ce9662b6f73f.jpg)  
Figure 3. Bennu: the radar shape model of [13], its $L = 6$ model c, the mirror image τc, and the decoy $d _ { + }$ , labeled in each panel. The last two are rotated to their numerically best alignment with c. All four are drawn at the same scale. The color shows the distance from the origin in meters; see the color bar.

Ex. 11.1 has the following implications for the two adversarial perspectives and for the keep-out task. The even pool at $L = 6$ determines the modeled volume of Bennu, and hence its mass for a given bulk density. It does not determine the handedness of the asteroid: the mirror image matches all 32 components exactly and lies at orbit distance at least 5 m. The odd bispectra do determine it. The even pool does not determine the keep-out radius either. The requirement $T _ { \mathrm { o u t } } \leq 2 7 7$ m holds for c and fails for $d _ { + }$ . The requirement $T _ { \mathrm { o u t } } \le 2 7 3$ m holds for $d _ { - }$ and fails for c. For every admissible set $K$ containing $c , d _ { + }$ and $d _ { - } .$ , the task interval of $T _ { \mathrm { o u t } }$ at the value $\Phi _ { 6 } ( c )$ with componentwise relative tolerance $1 0 ^ { - 1 4 }$ has width at least $2 7 9 . 5 8 8 - 2 7 2 . 1 7 2 = 7 . 4 1 6 \mathrm { m }$ . By Prop. 8.2, every estimate of the keep-out radius from that value has worst-case error at least 3.708 m on these three tuples. The modeled volumes of $d _ { + }$ and $d _ { - }$ difer from that of c by at most $2 \cdot 1 0 ^ { - 1 6 } \mathrm { m } ^ { 3 }$ . The continuation that produced $d _ { \pm }$ imposes no centroid constraint. The example therefore audits the representation with the center-of-mass origin of the radar model held fixed, as in Sec. 10.1, and not a pipeline that recenters every candidate shape.

These statements concern the degree-six model of a radar shape model with a vertex spacing of about 25 m; see [14]. They do not describe the surface of Bennu at finer scales.

## 12. Conclusion

The fiber radius measures geometric ambiguity, but does not by itself determine task utility or attribute confidentiality. Continuous ambiguity, discrete mirror ambiguity, finite-precision instability, and task ambiguity are distinct. The spectral pool gives a concrete example: its fibers can leave the shape undetermined while fixing enclosed volume exactly. The explicit clearance pair shows, without a numerical search, that a second geometric decision can remain impossible from the same descriptor. In an audit seeded by the radar shape model of Bennu, these phenomena appear together: the even pool fixes the modeled volume, misses the handedness, and, at componentwise relative descriptor tolerance $1 0 ^ { - 1 4 }$ , leaves the keep-out radius uncertain by more than 7 m. Together with task intervals, observer-centered reconstruction bounds, and the analysis of combined releases, this gives a precise basis for evaluating what a retained geometric representation can certify and what sharing it discloses.

## References

[1] A. S. Bandeira, B. Blum-Smith, J. Kileel, A. Perry, J. Niles-Weed, and A. S. Wein, Estimation under group actions: recovering orbits from invariants, Appl. Comput. Harmon. Anal. 66 (2023), 236–319. ↑6

[2] T. Bendory, D. Edidin, J. Katz, and S. Kreymer, Orbit recovery for spherical functions, 2025. arXiv:2508.02674. ↑14

[3] L. Beshaj, Reduction theory of binary forms, Advances on superelliptic curves and their applications, 2015, pp. 84–116. arXiv:1502.06289. ↑3, 4

[4] L. Beshaj, Integral binary forms with minimal height, ProQuest LLC, Ann Arbor, MI, 2016. Thesis (Ph.D.)–Oakland University. MR3579531 ↑3, 4

[5] J. Bochnak, M. Coste, and M.-F. Roy, Real algebraic geometry, Ergebnisse der Mathematik und ihrer Grenzgebiete (3), vol. 36, Springer, Berlin, 1998. ↑6, 9, 23

[6] D. Edidin and M. Satriano, Orbit recovery for band-limited functions, SIAM J. Appl. Algebra Geom. 8 (2024), 733–755. ↑14

[7] Z. Fan, R. R. Lederman, Y. Sun, T. Wang, and S. Xu, Maximum likelihood for high-noise group orbit estimation and single-particle cryo-EM, Ann. Statist. 52 (2024), no. 1, 52–77. arXiv:2107.01305. ↑2, 14

[8] A. J. Hofman and H. W. Wielandt, The variation of the spectrum of a normal matrix, Duke Math. J. 20 (1953), 37–39. ↑2, 10

[9] R. Kakarala, The bispectrum as a source of phase-sensitive invariants for Fourier descriptors: a grouptheoretic approach, J. Math. Imaging Vision 44 (2012), no. 3, 341–353. ↑2, 10, 13

[10] M. Kazhdan, T. Funkhouser, and S. Rusinkiewicz, Rotation invariant spherical harmonic representation of 3D shape descriptors, Symposium on Geometry Processing (SGP 2003), 2003, pp. 156–164. ↑2, 9, 13

[11] J. M. Lee, Introduction to smooth manifolds, Second, Graduate Texts in Mathematics, vol. 218, Springer, New York, 2013. ↑8

[12] L. C. Maximon, 3j, 6j, 9j symbols. §34.3, https://dlmf.nist.gov/34.3. ↑21, 22

[13] M. C. Nolan, C. Magri, E. S. Howell, L. A. M. Benner, J. D. Giorgini, C. W. Hergenrother, R. S. Hudson, D. S. Lauretta, J.-L. Margot, S. J. Ostro, and D. J. Scheeres, Shape model and surface properties of the OSIRIS-REx target asteroid (101955) Bennu from radar and lightcurve observations, Icarus 226 (2013), no. 1, 629–640. ↑25, 27

[14] , Asteroid (101955) Bennu radar shape model V1.1, 2021. ↑25, 27

[15] T. Shaska, Exact Euclidean-invariant surface descriptors for data-eficient learning from segmented anatomy, 2026. Preprint. ↑2, 13, 14

[16] T. Shaska and M.-R. Siadat, Invariant shape analysis of surfaces with spherical topology, 2026. arXiv:2609.39567. ↑2, 3, 4, 5, 6, 7, 12, 13, 15

Department of Computer Science and Engineering<sub>,</sub> Oakland University<sub>,</sub> Rochester<sub>,</sub> MI 48309

Department of Mathematical Sciences<sub>,</sub> United States Military Academy<sub>,</sub> West Point<sub>,</sub> NY 10996