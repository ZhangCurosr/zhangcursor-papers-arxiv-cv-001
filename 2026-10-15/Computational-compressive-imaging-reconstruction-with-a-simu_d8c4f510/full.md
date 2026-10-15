# Computational compressive imaging reconstruction with a simulated-annealing-based Ising solver

Ze Zheng <sup>1,2</sup>, Xiaoyan Wu<sup>1</sup>, Hang Xu <sup>1,2</sup>, Jingzheng Huang<sup>1,2,3</sup>, Tailong Xiao<sup>1,2,3\*</sup>, and Guihua Zeng<sup>1,2,3</sup>

<sup>1</sup> Institute for Quantum Sensing and Information Processing, State Key Laboratory of Photonics and Communications, Shanghai Jiao Tong University, Shanghai 200240, China <sup>2</sup>Hefei National Laboratory, Hefei 230088, China <sup>3</sup>Shanghai Research Center for Quantum Sciences, Shanghai 201315, China

<sup>\*</sup>Corresponding author. Email: tailong\_shaw@sjtu.edu.cn

Computational compressive imaging, such as single-pixel imaging and lensless imaging, is a promising imaging technique, enabling the reconstruction of objects from under-sampled measurements through compressed sensing algorithms or deep learning methods. However, the demand for electronic computational resources grows exponentially when recovering highresolution/multi-dimensional images and training neural networks with numerous parameters, which would become a bottleneck of this technique for widespread adoption. Here, we propose a physics-driven reconstruction framework, which formulates the imaging inverse problems as Ising models and directly recovers target objects through the highly efficient self-evolution toward the minimum Hamiltonian via non‑von Neumann Ising machines. We employ a simulated-annealingbased Ising solver to demonstrate this proof-of-concept by reconstructing the objects for experiments of both single-pixel imaging and a lensless diffuser camera at low-sampling rates (ranging from 1 % to 25 %). By further combining the foreground-sparsity constraint and the smoothness constraint, the reconstruction quality of our method is comparable to or even better than that of the mainstream algorithms. This approach has the potential to drastically reduce the electronic computational burden of inverse reconstruction, paving the way for real-time, low-power computational imaging frameworks.

## Introduction

Computational compressive imaging recovers an object from far fewer measurements than the pixel number of the image to be reconstructed, which shifts complexity from the optical front-end to a computational back-end [1-3]. In single-pixel imaging, a sequence of structured patterns illuminates the scene and the resulting intensities are integrated by a single bucket detector, and the image is reconstructed from the correlation between the patterns and the measurements [4-14]. In lensless imaging, such as coded-aperture imaging [15-20] and diffuser camera [21-26], the imaging lens is replaced by a custom mask or a scattering medium, and the sensor captures a multiplexed measurement that contains the multi-dimensional information of the scene. The image is computationally reconstructed from the correlation between the point spread functions (PSFs) and the under-sampled measurement [27, 28]. The forward model [29] is used to describe both optical encoding processes, and the imaging reconstruction is an ill-posed inverse problem as the number of measurements is smaller than the pixel number of the target image. Compressed sensing (CS) algorithms [4, 5] and deep learning methods [30-33] are employed to invert the forward process, however, the required memory, floating-point operations, and energy of both routes grow rapidly with image resolution and dimensionality [34] based on von Neumann electronic computers. CS algorithms typically require thousands of forward and adjoint operations to recover a mediumquality image at medium-sampling rates. Data-driven models should be trained on large datasets with millions of parameters and retrained whenever the optical encoding changes. This back-end cost has become a practical bottleneck that limits the deployment of compressive imaging in realtime, low-power, and embedded scenarios.

Ising machine (IM) offers a physical hardware solution scheme to combinatorial optimization problems, which has shown orders-of-magnitude advantages in time-to-solution [35-37] or energyto-solution [38-40] than traditional von Neumann schemes for specific problems, such as Max-cut problems [41] and traffic optimization problems [42]. The Ising Hamiltonian is defined as $\begin{array} { r } { H _ { I s i n g } ( s ) = - \sum _ { 1 \leq \mathrm { i } < \mathrm { j } \leq \mathrm { N } } J _ { \mathrm { i j } } s _ { \mathrm { i } } s _ { \mathrm { j } } - \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } h _ { \mathrm { i } } s _ { \mathrm { i } } } \end{array}$ with binary spins $s _ { \mathrm { i } } = \pm 1$ , where i and j are element index, and N is the spin number, which is a universal target for any problem in the complexity class nondeterministic polynomial time (NP) problems [43]. The solution of the original problem corresponds to the lowest-energy spin configuration. Ising machines embody the spin system in physical hardware and let it evolve toward the ground state, thereby circumventing the von Neumann bottleneck of continuously shuttling data between memory and processor [44]. Existing Ising machine implementations span superconducting-circuit quantum annealers [45], coherent Ising machines (CIMs) built from networks of degenerate optical parametric oscillators [46], spatialphotonic Ising machines (SPIMs) that encode thousands of spins into a modulated wavefront [47- 50], as well as spintronic, memristive, and CMOS annealing architectures [51]. Ising machines have recently been applied in solving L0-regularized compressed sensing [52, 53] and tomographic reconstruction problems [54]. Nevertheless, these demonstrations remain largely restricted to synthetic one-dimensional sparse signals or magnetic-resonance data on a small scale, and a general Ising reconstruction framework of computational compressive imaging systems has not yet been established.

In this paper, we propose a computational compressive imaging reconstruction framework driven by an Ising machine, which would assist in real-time, energy-efficient computational imaging. We demonstrate that the imaging inverse reconstruction problem, as well as prior constraints, can be formulated as an Ising model, followed by uploading into the Ising machine to evolve to the lowest-energy spin configuration. The reconstructed image is then retrieved from this spin configuration. We employ a simulated-annealing-based Ising solver to perform the evolution, which is the common strategy of SPIMs [47-50] and baseline competitors from other Ising machines [35- 37], due to the available Ising machines being unable to meet the requirement of bit resolution of our applications [55-57]. It should be note that various developing Ising machines exhibit solution quality better than simulated-annealing-based algorithms, alongside reduced time consumption and energy efficiency, for solving combinatorial optimization problems. This underpins our rationale that the proposed Ising reconstruction framework can possess practical advantages with advanced Ising machines, provided the coding requirement of bit resolution is satisfied [35-42]. We demonstrate the reconstruction of experiments of both single-pixel imaging and a diffuser camera, and the peak signal-to-noise ratios (PSNRs) and structural similarity index measures (SSIMs) are comparable to or better than the mainstream algorithms. Our proposed framework is also suitable for other computational imaging techniques incorporating multiple imaging modalities, and would further inspire the development of intelligent machine vision.

## Methods

As shown in Fig. 1(a), in computational imaging systems, the forward model is usually used to describe the optical front-end mapping from the object field to the measured signals, as below:

$$
y = A x ,\tag{1}
$$

where $y \in \mathbb { R } ^ { \mathbf { M } \times 1 }$ is the measurement vector, $A \in \mathbb { R } ^ { \mathrm { M \times N } }$ is the measurement matrix, and � ∈ $\mathbb { R } ^ { \mathrm { { N \times 1 } } }$ is the vectorized image of the object. M is the number of measurements, and N is the pixel number of the reconstructed image. In single-pixel imaging, each row of A represents an illumination pattern. In the diffuser camera, A corresponds to the linear transmission matrix of the diffuser. Within the range of the optical memory effect, A is a circulant matrix constructed from the PSF. In computational compressive imaging, the number of measurements M is fewer than the image's pixel number N, and Eq. (1) becomes an underdetermined system of equations and admits no unique exact solution. To solve this ill-posed problem, we take the minimization of the ℓ<sub>2</sub>-norm as our optimization objective to minimize the data fidelity loss, as shown in Eq. (2):

$$
\hat { x } = a r g m i n _ { x \geq 0 } \| y - A x \| _ { 2 } ^ { 2 } .\tag{2}
$$

![](images/c96a9dbe9d6c4d929dc65754670a95c3d8c8c0b8a5739d67c48a9b40a9466bf9.jpg)  
Figure 1. Concept and principle of the proposed imaging reconstruction framework driven by an Ising machine. (a) The general optical front-end encoding process, which is described by the forward model. The object X is illuminated by incoherent light, followed by encoding by an optical modulator. A represents the measurement matrix. The detector measures the one- or twodimensional intensity Y of the resulting light field corresponding to different imaging techniques. (b) The reconstruction framework using the Ising machine. The image to be reconstructed, represented by minimizing the $\ell _ { 2 } { \mathrm { - n o r m } }$ , can be further modeled as the minimization of the Ising Hamiltonian, followed by uploading into the Ising solver for self-evolution. The reconstructed image can be recovered directly by reshaping the spin configuration corresponding to the Ising minimum Hamiltonian (i.e., the green star).

By expanding Eq. (2) and discarding the constant term $y ^ { \mathrm { T } } y ,$ , since the constant term does not shift the position of the minimum, we obtain the equivalent formulation:

$$
{ \widehat { x } } = \underset { x \geq 0 } { \operatorname { a r g m i n } } [ x ^ { \operatorname { T } } ( A ^ { \operatorname { T } } A ) x - 2 ( A ^ { \operatorname { T } } y ) ^ { \operatorname { T } } x ] .\tag{3}
$$

For a binary target object, x degenerates to the binary variable $\sigma \in \{ 0 , 1 \} ^ { \mathrm { N } }$ , and Eq. (3) simplifies to Eq. (4), which corresponds to the standard form of quadratic unconstrained binary optimization (QUBO) problem with linear terms $\hat { \sigma } = \arg \operatorname* { m i n } _ { \sigma } ( \sigma ^ { \mathrm { T } } Q \sigma + 2 c ^ { \mathrm { T } } \sigma )$ , where the quadratic coefficient matrix is $Q = A ^ { \mathrm { T } } A$ and the linear coefficient vector is $c = - 2 ( A ^ { \mathrm { T } } y ) ^ { \mathrm { T } }$ . For grayscale objects, x should be decomposed into a set of binary variables which can also be rearranged into the standard QUBO form in Eq. (4). This work aims to propose a computational compressive reconstruction framework driven by an Ising machine, therefore, we adopt binary objects to simplify the reconstruction demonstrations of single-pixel imaging and diffuser camera experiments.

$$
\hat { \boldsymbol { \sigma } } = \underset { \boldsymbol { \sigma } \in \{ 0 , 1 \} } { \operatorname { a r g m i n } } [ \boldsymbol { \sigma } ^ { \mathrm { T } } ( \boldsymbol { A } ^ { \mathrm { T } } \boldsymbol { A } ) \boldsymbol { \sigma } - 2 ( \boldsymbol { A } ^ { \mathrm { T } } \boldsymbol { y } ) ^ { \mathrm { T } } \boldsymbol { \sigma } ] .\tag{4}
$$

We further map $\sigma _ { 1 }$ to binary spin $s _ { i } \in \{ - 1 , 1 \}$ via the following transformation:

$$
\begin{array} { r } { \sigma _ { \mathrm { { i } } } = \frac { 1 + s _ { \mathrm { { i } } } } { 2 } . } \end{array}\tag{5}
$$

Equation (4) is transformed according to Eq. (5) into:

$$
\begin{array} { r l r } {  { \hat { s } = \underset { s \in \{ - 1 , 1 \} } { \mathrm { a r g m i n } } [ ( \frac { I + s } { 2 } ) ^ { \mathrm { T } } A ^ { \mathrm { T } } A ( \frac { I + s } { 2 } ) - 2 ( A ^ { \mathrm { T } } y ) ^ { \mathrm { T } } ( \frac { I + s } { 2 } ) ] } } \\ & { = \underset { s \in \{ - 1 , 1 \} } { \mathrm { a r g m i n } } [ ( \frac { 1 } { 4 } s ^ { \mathrm { T } } A ^ { \mathrm { T } } A s + \frac { 1 } { 2 } I ^ { \mathrm { T } } A ^ { \mathrm { T } } A s + \frac { 1 } { 4 } I ^ { \mathrm { T } } A ^ { \mathrm { T } } A I ) - ( A ^ { \mathrm { T } } y ) ^ { \mathrm { T } } I - ( A ^ { \mathrm { T } } y ) ^ { \mathrm { T } } s ] , } \end{array}\tag{6}
$$

where I is an $\mathrm { N } \times 1$ all-ones vector. After discarding the constant term $\begin{array} { r } { \left[ \frac { 1 } { 4 } I ^ { \mathrm { T } } A ^ { \mathrm { T } } A I - ( A ^ { \mathrm { T } } y ) ^ { \mathrm { T } } I \right] } \end{array}$ and rearrangement $\operatorname { E q . } \left( 6 \right)$ , we obtain Eq. (7):

$$
\hat { s } = \underset { s \in \{ - 1 , 1 \} } { \operatorname { a r g m i n } } \Bigl [ \frac { 1 } { 4 } s ^ { \mathrm { T } } A ^ { \mathrm { T } } A s + \textstyle \frac { 1 } { 2 } ( A ^ { \mathrm { T } } A I - 2 A ^ { \mathrm { T } } y ) ^ { \mathrm { T } } s \Bigr ] .\tag{7}
$$

Element-wise expansion of Eq. (7) into summation yields:

$$
\begin{array} { r } { \hat { s } = \underset { s \in \{ - 1 , 1 \} } { \operatorname { a r g m i n } } \Big \{ - \sum _ { 1 \leq i < j \leq \mathrm { N } } \Big ( - \frac { 1 } { 2 } \sum _ { \mathrm { k } = 1 } ^ { \mathrm { M } } A _ { \mathrm { k i } } A _ { \mathrm { k j } } \Big ) s _ { i } s _ { j } - \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } - \frac { 1 } { 2 } \big ( \sum _ { \mathrm { k } = 1 } ^ { \mathrm { M } } A _ { \mathrm { k i } } \sum _ { \mathrm { j } = 1 } ^ { \mathrm { N } } A _ { \mathrm { k j } } - 2 \sum _ { \mathrm { k } = 1 } ^ { \mathrm { M } } A _ { \mathrm { k i } } Y _ { \mathrm { k } } \big ) s _ { \mathrm { i } } \Big \} , } \end{array}\tag{8}
$$

where k $\in \{ 1 , 2 , 3 , \cdots , \mathbb { M } \}$ represents the k-th measurement (i.e., the raw index of A), and $\mathrm { i , j } \in$ $\{ 1 , 2 , 3 , \cdots , \mathrm { N } \}$ represent the pixels' index of the image (i.e., the column indexes of $A )$ . The summation of diagonal elements of $A ^ { \mathrm { T } } A$ is a constant term (i.e., $\textstyle \sum _ { \mathrm { i = 1 } } ^ { \mathrm { N } } \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } A _ { \mathrm { k i } } ^ { 2 } )$ and has been discarded during expansion. Equation (8) is equivalent to the standard Ising model with the coefficient definitions given below:

$$
\small \left\{ \begin{array} { c } { J _ { \mathrm { i j } } = - \frac 1 2 { \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } } A _ { \mathrm { k i } } A _ { \mathrm { k j } } } \\ { h _ { \mathrm { i } } = - \frac 1 2 \bigl ( { \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } } A _ { \mathrm { k i } } { \sum _ { \mathrm { j = 1 } } ^ { \mathrm { N } } } A _ { \mathrm { k j } } - 2 { \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } } A _ { \mathrm { k i } } Y _ { \mathrm { k } } \bigr ) } \end{array} \right.\tag{9}
$$

The diagonal elements $J _ { \mathrm { i i } }$ contribute only a constant and may be set to zero without loss of generality. The image reconstruction task can be modeled as an Ising formulation shown in Eq. (9), followed by image recovery via optical Ising machine evolution, as illustrated in Fig. 1(b). We name this framework of Eq. (9) as Ising-SA. To improve the reconstruction quality of the algorithm, we further incorporate a foreground-sparsity constraint and a smoothness term into Eq. (4), as shown below:

$$
\boldsymbol { \hat { \sigma } } = \underset { \sigma \in \{ 0 , 1 \} } { \arg \operatorname* { m i n } } \big [ \sigma ^ { \mathrm { T } } ( \boldsymbol { A } ^ { \mathrm { T } } \boldsymbol { A } ) \sigma - 2 ( \boldsymbol { A } ^ { \mathrm { T } } \boldsymbol { y } ) ^ { \mathrm { T } } \sigma + \lambda _ { 1 } \mathcal { L } _ { s p a r s e } + \lambda _ { 2 } \mathcal { L } _ { s m o o t h } \big ] ,\tag{10}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are the corresponding coefficients, and

$$
\left\{ \begin{array} { c } { \mathcal { L } _ { s p a r s e } = \| \sigma \| _ { 1 } = \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \sigma _ { \mathrm { i } } } \\ { \left\{ \mathcal { L } _ { s m o o t h } = \sum _ { 1 \leq \mathrm { i } < \mathrm { j } \leq \mathrm { N } } d _ { \mathrm { i j } } \| \sigma _ { \mathrm { i } } - \sigma _ { \mathrm { j } } \| _ { 2 } ^ { 2 } = \sum _ { 1 \leq \mathrm { i } < \mathrm { j } \leq \mathrm { N } } d _ { \mathrm { i j } } ( \sigma _ { \mathrm { i } } + \sigma _ { \mathrm { j } } - 2 \sigma _ { \mathrm { i } } \sigma _ { \mathrm { j } } ) \right. } \end{array} \right.\tag{11}
$$

Given that $\sigma _ { \mathrm { i } } \in \{ 0 , 1 \}$ , there has $\sigma _ { \mathrm { i } } ^ { 2 } = \sigma _ { \mathrm { i } }$ . The $d _ { \mathrm { i j } }$ is the element in the i-th row and j-th column of the pixel adjacency matrix $D \in \mathbb { R } ^ { \mathrm { N \times N } }$ , which is pre-generated according to the specific problem and defined as:

$$
\begin{array} { r } { d _ { \mathrm { i j } } = \left\{ \begin{array} { c l } { 1 , \mathrm { w h e n } \sigma _ { \mathrm { i } } \mathrm { a n d } \sigma _ { \mathrm { j } } \mathrm { a r e n e i g h b o r s } } \\ { 0 , \mathrm { e l s e } } \end{array} \right. . } \end{array}\tag{12}
$$

We can also define the degree deg(σ<sub>i</sub>) of the i-th pixel σ<sub>i</sub> as the number of its neighbors, which can be pre-calculated:

$$
\begin{array} { r } { \deg ( \sigma _ { \mathrm { i } } ) = \sum _ { \mathrm { j } = 1 } ^ { \mathrm { N } } d _ { \mathrm { i j } } . } \end{array}\tag{13}
$$

We consider nearest-neighbor connections for each specific pixel, where the four adjacent pixels (up, down, left, and right) are defined as its neighbors, with deg(σ<sub>i</sub>) set to 4. Then, we simplify the smoothness term as:

$$
\begin{array} { r } { \mathcal { L } _ { s m o o t } \ = \sum _ { 1 \leq \mathrm { i } < \mathrm { j } \leq \mathrm { N } } - 2 d _ { \mathrm { i j } } \sigma _ { \mathrm { i } } \sigma _ { \mathrm { j } } + \sum _ { \mathrm { i } = 1 } ^ { \mathrm { N } } \deg ( \sigma _ { \mathrm { i } } ) \sigma _ { \mathrm { i } } . } \end{array}\tag{14}
$$

Combining the transformation in Eq. (5), we can obtain the updated coefficients for the Ising machine:

$$
\small \left\{ \begin{array} { c } { J _ { \mathrm { i j } } = - \frac { 1 } { 2 } \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } A _ { \mathrm { k i } } A _ { \mathrm { k j } } + \frac { \lambda _ { 2 } } { 2 } d _ { \mathrm { i j } } } \\ { h _ { \mathrm { i } } = - \frac { 1 } { 2 } \Big ( \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } A _ { \mathrm { k i } } \sum _ { \mathrm { j = 1 } } ^ { \mathrm { N } } A _ { \mathrm { k j } } - 2 \sum _ { \mathrm { k = 1 } } ^ { \mathrm { M } } A _ { \mathrm { k i } } Y _ { \mathrm { k } } \Big ) - \frac { \lambda _ { 1 } } { 2 } } \end{array} \right.\tag{15}
$$

The additional constant terms do not affect the search for the ground-state spin configuration. We name the improved framework of Eq. (15) as Ising-SA-D. Since the elements of $J \in \mathbb { R } ^ { \mathrm { N \times N } }$ and $h \in \mathbb { R } ^ { \mathrm { N } }$ require high-precision encoding. The 8-bit encoding precision of currently available Ising machines leads to ill-conditioning and invalidates the equivalence with the original problem. Hence, we solve the problem using a simulated-annealing Ising solver with double-precision floating-point arithmetic on an electronic computer. Nevertheless, the modulation and encoding precision of emerging Ising machines is continuously improving, which is promising to combine with our approach to realize high-efficient image reconstruction in the future.

## Simulations

(a)  
![](images/5d6f09a9434deda2e5c5cfcb17261fb9901bcd9a8274d7341e53915a8cc17bcd.jpg)

![](images/95e2b292a28a68c9a0df3fa84afc0582af3285cfe3f28f8ae466e887f69d3a29.jpg)

(b) Annealing evolution (sampling rate: 2 %)  
![](images/a9dddda093c5f6ed84b132d0fdd0e371b926976f0055da74eaf646e0918436db.jpg)

![](images/bb544287292661bfb0b98dde8eacd3ec238946f6f2246d482c6664f31ea7886b.jpg)

![](images/42b1621ddf10d7d7be17fa4cbe19a8060206b4e8e5491e7b3b219af6aa7b297e.jpg)  
Figure 2. The simulation results of single-pixel imaging with different reconstruction methods. (a) The experimental setup of single-pixel imaging using the cake-cutting ordered Hadamard patterns, which is consistent with the simulation setup. The target object is the handwritten digit "5", and the sampling rate in this simulation is 2 %. (b) The annealing evolution of the Ising-SA-D framework, including the Ising Hamiltonian (in gray), PSNR (in red), and SSIM (in blue) of the reconstructed images varying with iterations. The ground truth image is inserted in (b). (c) The reconstructed images recovered by the Ising-SA reconstruction framework at 10,000, 20,000, 50,000, and 100,000 iterations. (d) The reconstructed images recovered by the Ising-SA-D reconstruction framework at 10,000, 20,000, 50,000, and 100,000 iterations. The corresponding iteration numbers are displayed at the top-left corner of each reconstructed image in (c–d). (e) The reconstruction image from the employed 8-bit commercial Ising machine. (f–h) The reconstruction images recovered by DGI, ADMM-TV, and GIDC, respectively. The corresponding PSNR and SSIM values are shown at the bottom-left corner of each reconstructed image in (c–d, f–h). Among these values, the highest values are bolded, and the second-highest are underlined, respectively. The resolution of the reconstructed images is 64×64 pixels.

We demonstrate the results of the single-pixel imaging simulations at a sampling rate of 2 % with our proposed frameworks (i.e., Ising-SA, and Ising-SA-D), commercial Ising machine with 8- bit encoding precision (i.e., 8-bit Ising) [56], differential ghost imaging (DGI) [58], alternating direction method of multipliers with total variation regularization (ADMM-TV) [59, 60], and ghost imaging using deep neural-network constraint (GIDC) [61] in Fig. 2. As shown in Fig. 2(b), the Ising Hamiltonian of Ising-SA-D framework decreases with iterations, while the PSNR and SSIM of the corresponding reconstructed images gradually increase. It should be noted that setting the iteration range from 0 to 100,000 does not represent the time-consuming bottleneck of our framework. Instead, this setup is intended to acquire a relatively optimal solution via the simulated annealing (SA) algorithm, for which slower annealing leads to results closer to the ground truth [62- 64]. This allows us to estimate the reconstruction performance achievable by physical hardwarebased Ising machines in the future more accurately, since many existing ultrafast Ising machines have been demonstrated to find superior solutions compared with the SA algorithm [35-36, 42, 55].

Figure 2(c-d) also demonstrate that the quality of reconstructed images improves with increasing iterations visually. Ising-SA-D framework incorporating sparse and smooth constraints achieves higher reconstruction quality than the Ising-SA framework with only the data-fidelity term at the same iteration, which implies the potential advantages of introducing additional constraints and regularization terms into the Ising-SA framework for other scenarios and applications. Figure 2(e) shows the reconstruction result of the commercial Ising machine, which yields an all-white image (i.e., an all-ones solution vector), and obviously fails to match the ground-truth image. This demonstrates that the 8-bit precision is insufficient to properly encode this compressive image reconstruction problem. Figures 2 (f–h) present the reconstruction images of three mainstream algorithms under the same sampling rate of 2%, whose metrics are inferior to those reconstructed by the Ising-SA-D framework at 100,000 iterations.

## Experiments

(a)  
![](images/90bdae890d8b9bb0b25581715c92094409b5a5c203ff09a5d2ce7cc8c7625204.jpg)

![](images/08a207210e2533ee9d66c286ed2dd84342192838609315c319b67b7404ce5a05.jpg)

(c)  
![](images/638a01f9d3fdc25fc209c146d82601f3248cfc23154ed30e7cde686d10d36c46.jpg)  
Figure 3. The results of single-pixel imaging experiments with different reconstruction methods. (a), (b) The PSNRs and SSIMs of Ising-SA (in brown), Ising-SA-D (in red), DGI (in blue), ADMM-TV (in green), and GIDC (in purple) at different sampling rates (1 %, 5 %, 10 %, 15 %, 20 %, and 25 %) are shown, respectively. The ground truth of the employed object "sailboat" is inserted in (b). (c) The corresponding reconstructed images of different methods at different sampling rates. The resolution of the reconstructed images is 64×64 pixels, and the scale bar is 0.5 cm.

Figure 3 demonstrates the reconstructed results of single-pixel imaging experiments. The experimental setup is consistent with Fig. 2(a). A parallel coaxial LED light (LB-CLS-50, LBTEK) illuminates an amplitude-type spatial light modulator (A-SLM, TSLM023-A, Cas Microstar), which is employed to project the object. The object is imaged to the plane of the digital micro-mirror device (DMD, V-7001-VIS, ViALUX) via a lens (OLC320754M, JCOPTiX). The cake-cutting ordered Hadamard patterns are projected on the DMD. The reflected light from the DMD is focused by a condenser lens (OLS300434-T2M, JCOPTiX), followed by integrating measurement with a singlepixel detector (PDAPC2, Thorlabs) after switching. A data acquisition device (USB-3133A, ART Technology) is utilized for signal acquisition. The target object images are custom-designed in-

house.

The Ising-SA-D framework achieves the best image reconstruction quality at each sampling rate, followed by the Ising-SA framework, with both set for 100,000 iterations. The reconstruction quality of all methods improves as the sampling rates increase. Although mainstream algorithms can visually recover the target objects, they exhibit high-level background noise, yielding lower SNR and metric values (PSNRs and SSIMs) compared with our proposed frameworks. We also conduct the same experimental acquisition and reconstruction on other target objects to demonstrate the generalization and effectiveness of our approach, and the corresponding results are presented in Appendix A.

![](images/ddf13105e52a9007c819ad7380665235ae8182b5118e579b724dcd633a74ab3a.jpg)  
Figure 4. The results of the experiments of the diffuser camera. (a) The experimental setup. An LED light illuminates the structured object, followed by capture by the custom-designed diffuser camera. (b) The pre-calibrated PSF of the system at the object position. (c-d) The captured raw data, the reconstructed images, and the metric values (PSNRs and SSIMs) of the 3D-printed transmitted objects "H" and "Z" with different methods at the sampling rates of 8 % and 16 %, respectively. The sampling region within each raw data is marked by a dashed box. This square region is centered at the raw-data center, and its size varies with different sampling rates. The highest metric values for each target object at each sampling rate are bolded, and the second-highest values are underlined. The resolution of the reconstructed images is 64×64 pixels, and the scale bar is 1 cm.

We also applied the proposed framework to inverse reconstruct the images of diffuser camera experiments, as shown in Fig. 4. Figure 4(a) presents the experimental setup. A monochromatic red LED light (GCI-060401, DHC), which is collimated by a lens (MCX10610, LBTEK), illuminates the 3D-printed transmission-structured objects (i.e., "H" and "Z"), which are placed within the range of the optical memory effect. The transmitted light is modulated by a diffuser (GCL-201101, DHC), followed by capture by a two-dimensional sensor (daA2500-14um, Basler) which is placed 1 cm behind. The PSF is pre-calibrated by replacing the object with a 100 μm pinhole (GCO-P100A, DHC) at the same position, as presented in Fig. 4(b).

The vectorized measurement Y of the forward model is composed of the values of the sampling region in each raw data, and the measurement matrix A is constructed by the PSF. As shown in Fig. 4(c-d), our proposed Ising-SA and Ising-SA-D frameworks with 100,000 iterations perform better than other methods at low-sampling rates, which implies the application potential of our proposed framework in other computational compressive imaging techniques with limited raw measurements. Although the demonstration is performed numerically with double-precision solvers, advances in hardware encoding precision of physical Ising machines will enable on-chip, low-latency image reconstruction in future work. The presented strategy can also be extended to other inverse problems beyond diffuser-camera and single-pixel imaging.

## Conclusion and Discussion

In summary, we have proposed and demonstrated the compressive imaging reconstruction framework, which models the reconstruction task derived from the optical forward model, consisting of a data-fidelity term, a sparsity constraint term and a smoothness constraint term, into the Ising model formulation. We have demonstrated the effectiveness of this framework with a simulated-annealing-based Ising solver, and have achieved the higher reconstruction quality than the mainstream algorithms in the simulation and experiments of single-pixel imaging and lensless diffuser camera. More imaging techniques and modalities are potentially inspired to be incorporated into this framework [65-67]. The proposed approach is further going to be deployed to the developing physical Ising machines to achieve the ultrafast and energy-efficient imaging reconstruction.

We should note that, besides high-precision encoding requirements, the proposed method demands fully-connected Ising machine spin resources scaling as O(N<sup>2</sup>) and O(N<sup>4</sup>) couplings for reconstructing binary images with a resolution of N×N pixels. For grayscale images with the same resolution, the required spins and couplings scale as O(aN<sup>2</sup>) and O(a<sup>2</sup>N<sup>4</sup>), respectively, where a represents the gray level count (a equals 256 for 8-bit grayscale images). We reconstruct 64×64 pixels binary images with 4096+64=4160 virtual spins in our demonstrations. Large-scale fullyconnected Ising machines are difficult to realize on physical platforms including coherent Ising machines [35, 36, 46]. Spatial photonic Ising machines have been proved to possess intrinsic full connectivity [68] and are progressing toward larger size [48, 69], complex couplings [48] and improved precision [70]. Dedicated reconstruction frameworks designed for spatial photonic Ising machines may greatly reduce the hardware resource cost, making this approach a promising route toward rapid real-world implementation.

Appendix A: Reconstruction of other objects in experimental single-pixel imaging  
(a)  
![](images/322f6a65baa7d712459d2ba16b96a09311375925ae39ebff5b9d6991d7f06b56.jpg)

![](images/a579cde787626be247118be1f62de64db485683299a863bea3b7e7f5478c830b.jpg)

(c)  
![](images/895b2703938227ce383f983c586a882331307834a5f9690937ca4902a0108333.jpg)  
Figure 5. The results of single-pixel imaging experiments with different reconstruction methods. (a), (b) The PSNRs and SSIMs of Ising-SA (in brown), Ising-SA-D (in red), DGI (in blue), ADMM-TV (in green), and GIDC (in purple) at different sampling rates (1 %, 5 %, 10 %, 15 %, 20 %, and 25 %) are shown, respectively. The ground truth of the employed object "running man" is inserted in (a). (c) The corresponding reconstructed images of different methods at different sampling rates. The resolution of the reconstructed images is 64×64 pixels, and the scale bar is 0.5 cm.

(a)  
![](images/4d21c82cfc0fe56aad1117ea8e7bb433910d2a37e0a4aaa9a972858923c888c9.jpg)

![](images/4dad44d3818a4181b0f2b1fe07f390d810a95eaecc8d52dc1861ae68a84ecfa9.jpg)

(c)  
![](images/396e60e20c6ac292772ac53579d70e5560c7e2a7d68774ad7154a4b7d70f1107.jpg)  
Figure 6. The results of single-pixel imaging experiments with different reconstruction methods. (a), (b) The PSNRs and SSIMs of Ising-SA (in brown), Ising-SA-D (in red), DGI (in blue), ADMM-TV (in green), and GIDC (in purple) at different sampling rates (1 %, 5 %, 10 %, 15 %, 20 %, and 25 %) are shown, respectively. The ground truth of the employed object "fish" is inserted in (b). (c) The corresponding reconstructed images of different methods at different sampling rates. The resolution of the reconstructed images is 64×64 pixels, and the scale bar is 0.5 cm.

## Acknowledgments

This work was supported by the National Key R&D Program of China (No. 2025YFF0515504), the National Natural Science Foundation of China (Nos. 62401359 and 62471289), the State Key Laboratory of Photonics and Communications, the Quantum Science and Technology - National Science and Technology Major Project (No. 2021ZD0300703), the Shanghai Municipal Science and Technology Major Project (No. 2019SHZDZX01), and the CCF‑QBoson Fund for Quantum‑Computing Application Innovation (No. CCF‑QBoson202406).

## Author Contributions

G.Z. conceived the research project. Z.Z. designed the scheme, implemented the algorithms, developed the simulation, and performed the experiments. G.Z. and T.X. supervised the project. Z.Z. analyzed all the results and prepared the manuscript, with assistance from X.W.. All the authors participated in the discussion and confirmed the final manuscript.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

The data that support the findings of this study are available from the corresponding author upon reasonable request

## References

[1] M. F. Duarte, M. A. Davenport, D. Takhar, J. N. Laska, T. Sun, K. F. Kelly, and R. G. Baraniuk, "Single-pixel imaging via compressive sampling," IEEE Signal Process. Mag. 25, 83–91 (2008). https://doi.org/10.1109/MSP.2007.914730

[2] J. H. Shapiro, "Computational ghost imaging," Phys. Rev. A 78, 061802 (2008). https://doi.org/10.1103/PhysRevA.78.061802

[3] A. Wang, J. Zhu, J. Wu, L. Ye, X. Liu, Y. Jiang, Y. Wang, Y. Liu, R. Ge, J. Du, and Z. Zhang, "Compressed computational imaging based on optical differentiation," Opt. Lett. 49, 6988–6991 (2024). https://doi.org/10.1364/OL.535063

[4] E. J. Candès, J. Romberg, and T. Tao, "Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information," IEEE Trans. Inf. Theory 52, 489–509 (2006). https://doi.org/10.1109/TIT.2005.862083

[5] D. L. Donoho, "Compressed sensing," IEEE Trans. Inf. Theory 52, 1289–1306 (2006). https://doi.org/10.1109/TIT.2006.871582

[6] M. P. Edgar, G. M. Gibson, and M. J. Padgett, "Principles and prospects for single-pixel imaging," Nat. Photonics 13, 13–20 (2018). https://doi.org/10.1038/s41566-018-0300-7

[7] D. Wang, B. Liu, J. Song, Y. Wang, X. Shan, X. Zhong, and F. Wang, "Dual-mode adaptive-SVD ghost imaging," Opt. Express 31, 14225–14239 (2023). https://doi.org/10.1364/OE.486290

[8] W. Jiang, Y. Yin, J. Jiao, X. Zhao, and B. Sun, "2,000,000 fps 2D and 3D imaging of periodic or reproducible scenes with single-pixel detectors," Photon. Res. 10, 2157–2164 (2022). https://doi.org/10.1364/PRJ.461064

[9] E. Hahamovich, S. Monin, Y. Hazan, and A. Rosenthal, "Single pixel imaging at megahertz switching rates via cyclic Hadamard masks," Nat. Commun. 12, 4516 (2021). https://doi.org/10.1038/s41467-021-24850-x

[10] J. Song, B. Liu, Y. Wang, C. Chen, X. Shan, X. Zhong, L. Wu, and F. Wang, "Computational and dark-field ghost imaging with ultraviolet light," Photon. Res. 12, 226–234 (2024). https://doi.org/10.1364/PRJ.503974

[11] Z. Gao, X. Wu, X. Zhai, Z. Zheng, J. Shi, J. Huang, and G. Zeng, "Physics-informed two-stage neural network enables computational ghost imaging through unknown scattering media," Appl. Phys. Lett. 127, 071103 (2025). https://doi.org/10.1063/5.0283491

[12] Y. Zhang, Y. Li, Z. Liu, F. Wang, G. Situ, M. K. Chen, H. Wang, and Z. Geng, "Physics and data-driven alternative optimization enabled ultra-low-sampling single-pixel imaging," Adv. Photon. Nexus 4, 036005 (2025). https://doi.org/10.1117/1.APN.4.3.036005

[13] Z.-Q. Zhao, Y.-X. Zhang, J.-Q. Song, M.-F. Li, and L.-A. Wu, "Photon-counting single-pixel camera based on a fast spinning coding disk," Opt. Lett. 50, 169–172 (2025). https://doi.org/10.1364/OL.546034

[14] H. Shao, H. Huang, Y.-X. Wei, H.-J. Zhang, Z.-H. Yang, and Y.-J. Yu, "Two-stage training method for high-quality reconstruction in single-pixel imaging," Chin. Phys. Lett. 41, 124202 (2024). https://doi.org/10.1088/0256-307X/41/12/124202

[15] M. J. Cieślak, K. A. A. Gamage, and R. Glover, "Coded-aperture imaging systems: past, present and future development—a review," Radiat. Meas. 92, 59–71 (2016). https://doi.org/10.1016/j.radmeas.2016.08.002

[16] H. Arguello and G. R. Arce, "Rank minimization code aperture design for spectrally selective compressive imaging," IEEE Trans. Image Process. 22, 941–954 (2013).

[17] T. Kim, K. C. Lee, N. Baek, H. Chae, and S. A. Lee, "Aperture-encoded snapshot hyperspectral imaging with a lensless camera," APL Photon. 8, 066109 (2023). https://doi.org/10.1063/5.0150797

[18] Z. Tian, L. Li, J. Ma, L. Cao, and P. Su, "CFZA camera: a high-resolution lensless imaging technique based on compound Fresnel zone aperture," Opt. Lett. 49, 3532–3535 (2024). https://doi.org/10.1364/OL.527533

[19] X. Zhang, B. Wang, S. Li, K. Liang, H. Guan, Q. Chen, and C. Zuo, "Lensless imaging with a programmable Fresnel zone aperture," Sci. Adv. 11, eadt3909 (2025). https://doi.org/10.1126/sciadv.adt3909

[20] N. Joshi, V. Tiwari, T. Kahro, A. P. I. Xavier, T. Tahara, A. Kasikov, K. Kukli, S. Juodkazis, A. Tamm, J. Rosen, and V. Anand, "Interferenceless coded aperture correlation holography for fivedimensional imaging of 3D space, spectrum and polarization," J. Phys. Photonics 7, 025004 (2025). https://doi.org/10.1088/2515-7647/ada997

[21] N. Antipa, G. Kuo, R. Heckel, B. Mildenhall, E. Bostan, R. Ng, and L. Waller, "DiffuserCam: lensless single-exposure 3D imaging," Optica 5, 1–9 (2018). https://doi.org/10.1364/OPTICA.5.000001

[22] K. Monakhova, K. Yanny, N. Aggarwal, and L. Waller, "Spectral DiffuserCam: lensless snapshot hyperspectral imaging with a spectral filter array," Optica 7, 1298–1307 (2020). https://doi.org/10.1364/OPTICA.397214

[23] S. K. Sahoo, D. Tang, and C. Dang, "Single-shot multispectral imaging with a monochromatic camera," Optica 4, 1209–1213 (2017). https://doi.org/10.1364/OPTICA.4.001209

[24] Z. Zheng, B. Liu, J. Song, L. Ding, X. Zhong, L. Chang, X. Wu, D. McGloin, and F. Wang, "Temporal compressive edge imaging enabled by a lensless diffuser camera," Opt. Lett. 49, 3058– 3061 (2024). https://doi.org/10.1364/OL.515429

[25] Z. Cai, J. Chen, G. Pedrini, W. Osten, X. Liu, and X. Peng, "Lensless light-field imaging through diffuser encoding," Light Sci. Appl. 9, 143 (2020). https://doi.org/10.1038/s41377-020- 00380-x

[26] Z. Zheng, B. Liu, J. Song, M. Zhu, C. Han, Y. Wang, M. Tian, Y. Xiong, Z. Yang, X. Zhong, D. McGloin, and F. Wang, "Exploiting scattering-based point spread functions for snapshot 5D and modality-switchable lensless imaging," Laser Photon. Rev. 20, e01888 (2025). https://doi.org/10.1002/lpor.202501888

[27] V. Boominathan, J. K. Adams, M. S. Asif, B. W. Avants, J. T. Robinson, R. G. Baraniuk, A. C. Sankaranarayanan, and A. Veeraraghavan, "Lensless imaging: a computational renaissance," IEEE Signal Process. Mag. 33, 23–35 (2016). https://doi.org/10.1109/MSP.2016.2581921

[28] V. Boominathan, J. T. Robinson, L. Waller, and A. Veeraraghavan, "Recent advances in lensless imaging," Optica 9, 1–6 (2021). https://doi.org/10.1364/OPTICA.431361

[29] J. W. Goodman, Introduction to Fourier Optics (McGraw-Hill, New York, 1968). ISBN: 9780070237766.

[30] K. Song, Y. Bian, D. Wang, R. Li, K. Wu, H. Liu, C. Qin, J. Hu, and L. Xiao, "Advances and challenges of single-pixel imaging based on deep learning," Laser Photon. Rev. 19, 2401397 (2025). https://doi.org/10.1002/lpor.202401397

[31] L. Huang, R. Luo, X. Liu, and X. Hao, "Spectral imaging with deep learning," Light Sci. Appl. 11, 61 (2022). https://doi.org/10.1038/s41377-022-00743-6

[32] A. Sinha, J. Lee, S. Li, and G. Barbastathis, "Lensless computational imaging through deep

learning," Optica 4, 1117–1125 (2017). https://doi.org/10.1364/OPTICA.4.001117

[33] G. Barbastathis, A. Ozcan, and G. Situ, "On the use of deep learning for computational imaging," Optica 6, 921–943 (2019). https://doi.org/10.1364/OPTICA.6.000921

[34] X. Yuan, D. J. Brady, and A. K. Katsaggelos, "Snapshot compressive imaging: theory, algorithms, and applications," IEEE Signal Process. Mag. 38, 65–88 (2021). https://doi.org/10.1109/MSP.2020.3023869

[35] T. Honjo, T. Sonobe, K. Inaba, T. Inagaki, T. Ikuta, Y. Yamada, T. Kazama, K. Enbutsu, T. Umeki, R. Kasahara, K.-I. Kawarabayashi, and H. Takesue, "100,000-spin coherent Ising machine," Sci. Adv. 7, eabh0952 (2021). https://doi.org/10.1126/sciadv.abh0952

[36] H. Takesue, K. Inaba, T. Honjo, Y. Yamada, T. Ikuta, Y. Yonezu, T. Inagaki, T. Umeki, and R. Kasahara, "Finding independent sets in large-scale graphs with a coherent Ising machine," Sci. Adv. 11, eads7223 (2025). https://doi.org/10.1126/sciadv.ads7223

[37] W. Yue, T. Zhang, Z. Jing, K. Wu, Y. Yang, Z. Yang, Y. Wu, W. Bu, K. Zheng, J. Kang, Y. Lin, Y. Tao, B. Yan, R. Huang, and Y. Yang, "A scalable universal Ising machine based on interactioncentric storage and compute-in-memory," Nat. Electron. 7, 904–913 (2024). https://doi.org/10.1038/s41928-024-01228-7

[38] B. Wu, W. Zhang, S. Zhang, H. Zhou, Z. Ruan, M. Li, D. Huang, J. Dong, and X. Zhang, "A monolithically integrated optical Ising machine," Nat. Commun. 16, 4296 (2025). https://doi.org/10.1038/s41467-025-59537-0

[39] J. Si, S. Yang, Y. Cen, J. Chen, Y. Huang, Z. Yao, D.-J. Kim, K. Cai, J. Yoo, X. Fong, and H. Yang, "Energy-efficient superparamagnetic Ising machine and its application to traveling salesman problems," Nat. Commun. 15, 3457 (2024). https://doi.org/10.1038/s41467-024-47818-z

[40] Y. Hou, Z. Xiao, X. Wu, Z. Tong, S. Wang, M. Chen, C. H. Chan, Y. Lu, Y. Zhang, C.-C. Chen, Z. Zhu, A. Lee, D. Wu, H. Cai, and Q. Shao, "Fully analog in-memory annealing Ising machine with unified deterministic and probabilistic computing array using CMOS-integrated VCMA-MTJ," in 2025 IEEE International Electron Devices Meeting (IEDM) (2025), pp. 1–4. https://doi.org/10.1109/IEDM50572.2025.11353790

[41] N. Al-Kayed, C. St-Arnault, H. Morison, A. Aadhi, C. Huang, A. N. Tait, D. V. Plant, and B. J. Shastri, "Programmable 200 GOPS Hopfield-inspired photonic Ising machine," Nature 648, 576– 584 (2025). https://doi.org/10.1038/s41586-025-09838-7

[42] Z. Hu, Y. Ren, Y. Meng, T. Wang, Y. Jiang, M. Wei, Y. Xiao, Z. Li, and M. Li, "Programmable optoelectronic Ising machine for optimization of real-world problems," Light Sci. Appl. 15, 6 (2026). https://doi.org/10.1038/s41377-025-02100-9

[43] A. Lucas, "Ising formulations of many NP problems," Front. Phys. 2, 5 (2014). https://doi.org/10.3389/fphy.2014.00005

[44] N. Mohseni, P. L. McMahon, and T. Byrnes, "Ising machines as hardware solvers of combinatorial optimization problems," Nat. Rev. Phys. 4, 363–379 (2022). https://doi.org/10.1038/s42254-022-00440-8

[45] M. W. Johnson, M. H. S. Amin, S. Gildert, T. Lanting, F. Hamze, N. Dickson, R. Harris, A. J. Berkley, J. Johansson, P. Bunyk, et al., "Quantum annealing with manufactured spins," Nature 473, 194–198 (2011). https://doi.org/10.1038/nature10012

[46] P. L. McMahon, A. Marandi, Y. Haribara, R. Hamerly, C. Langrock, S. Tamate, T. Inagaki, H. Takesue, S. Utsunomiya, K. Aihara, et al., "A fully programmable 100-spin coherent Ising machine with all-to-all connections," Science 354, 614–617 (2016). https://doi.org/10.1126/science.aah5178

[47] D. Veraldi, D. Pierangeli, S. Gentilini, M. C. Strinati, J. Sakellariou, J. S. Cummins, A. Kamaletdinov, M. Syed, R. Z. Wang, N. G. Berloff, et al., "Fully programmable spatial photonic Ising machine by focal plane division," Phys. Rev. Lett. 134, 063802 (2025). https://doi.org/10.1103/PhysRevLett.134.063802

[48] Z. Zheng, Y. Li, H. Xu, J. Huang, T. Xiao, and G. Zeng, "Incorporating rank-free coupling and external field via an incoherent modulated spatial photonic Ising machine," arXiv:2512.21587 (2025). https://doi.org/10.48550/arXiv.2512.21587

[49] Z. Zheng, R. Ni, J. Zhao, X. Hu, W. Jiang, Y. Li, H. Xu, T. Xiao, and G. Zeng, "Accelerating ground state search of spatial photonic Ising machines with genetic-simulated annealing hybrid algorithm," arXiv:2605.23295 (2026). https://doi.org/10.48550/arXiv.2605.23295

[50] L. Luo, Z. Mi, J. Huang, and Z. Ruan, "Wavelength-division multiplexing optical Ising simulator enabling fully programmable spin couplings and external magnetic fields," Sci. Adv. 9, eadg6238 (2023). https://doi.org/10.1126/sciadv.adg6238

[51] C. Duffee, J. Athas, Y. Shao, N. D. Melendez, E. Raimondo, J. A. Katine, K. Y. Camsari, G. Finocchio, and P. Khalili Amiri, "An integrated-circuit-based probabilistic computer that uses voltage-controlled magnetic tunnel junctions as its entropy source," Nat. Electron. 8, 784–793 (2025). https://doi.org/10.1038/s41928-025-01439-6

[52] M. D. S. H. Gunathilaka, S. Kako, Y. Inui, K. Mimura, M. Okada, Y. Yamamoto, and T. Aonishi, "Effective implementation of L0-regularised compressed sensing with chaotic-amplitude-controlled coherent Ising machines," Sci. Rep. 13, 16140 (2023). https://doi.org/10.1038/s41598-023-43364- 8

[53] T. Aonishi, K. Mimura, M. Okada, and Y. Yamamoto, "L0 regularization-based compressed sensing with quantum–classical hybrid approach," Quantum Sci. Technol. 7, 035013 (2022). https://doi.org/10.1088/2058-9565/ac6f19

[54] K. Dremel, D. Prjamkov, M. Firsching, M. Weule, T. Lang, A. Papadaki, S. Kasperl, M. Blaimer, and T. O. J. Fuchs, "Utilizing quantum annealing in computed tomography image reconstruction," IEEE Trans. Quantum Eng. 6, 3100810 (2025). https://doi.org/10.1109/TQE.2025.3549305

[55] H. Wei, C. Ai, P. Guo, B. Jia, L. Yuan, H. Song, S. Chen, C. Cao, J. Wu, C. Ju, et al., "A versatile coherent Ising computing platform," Light Sci. Appl. 15, 74 (2026). https://doi.org/10.1038/s41377- 025-02178-1

[56] Beijing Ising Intelligent Technology Co., Ltd., "Ising Cloud Platform," https://console.isingq.com/ (accessed 2026-10-13).

[57] Beijing Bose Quantum Technology Co., Ltd., "QBoson Ising quantum computing cloud platform," https://platform.qboson.com/ (accessed 2026-10-13).

[58] F. Ferri, D. Magatti, L. A. Lugiato, and A. Gatti, "Differential ghost imaging," Phys. Rev. Lett. 104, 253603 (2010). https://doi.org/10.1103/PhysRevLett.104.253603

[59] L. I. Rudin, S. Osher, and E. Fatemi, "Nonlinear total variation based noise removal algorithms," Physica D 60, 259–268 (1992). https://doi.org/10.1016/0167-2789(92)90242-F

[60] M. V. Afonso, J. M. Bioucas-Dias, and M. A. T. Figueiredo, "An augmented Lagrangian approach to the constrained optimization formulation of imaging inverse problems," IEEE Trans. Image Process. 20, 681–695 (2011). https://doi.org/10.1109/TIP.2010.2076294

[61] F. Wang, C. Wang, M. Chen, W. Gong, Y. Zhang, S. Han, and G. Situ, "Far-field superresolution ghost imaging with a deep neural network constraint," Light Sci. Appl. 11, 1 (2022). https://doi.org/10.1038/s41377-021-00680-w

[62] S. Kirkpatrick, C. D. Gelatt Jr., and M. P. Vecchi, "Optimization by simulated annealing," Science 220, 671–680 (1983). https://doi.org/10.1126/science.220.4598.671

[63] S. Geman and D. Geman, "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images," IEEE Trans. Pattern Anal. Mach. Intell. 6, 721–741 (1984). https://doi.org/10.1109/TPAMI.1984.4767596

[64] B. Hajek, "Cooling schedules for optimal annealing," Math. Oper. Res. 13, 311–329 (1988). https://doi.org/10.1287/moor.13.2.311

[65] Y. Ma, Y. Gao, and L. Cao, "Lensless Kramers–Kronig holography with nonlocal flat optics," Optica 13, 1845 (2026). https://doi.org/10.1364/OPTICA.607488

[66] N. Liu, K. Liu, Y. Yang, Y. Peng, and L. Cao, "Propagation-adaptive 4K computer-generated holography using physics-constrained spatial and Fourier neural operator," Nat. Commun. 16, 62997 (2025). https://doi.org/10.1038/s41467-025-62997-z

[67] Y. Gao and L. Cao, "Model-based deep learning enables time-resolved computational microscopy," PhotoniX 7, 22 (2026). https://doi.org/10.1186/s43074-025-00222-2

[68] D. Pierangeli, G. Marcucci, and C. Conti, "Large-scale photonic Ising machine by spatial light modulation," Phys. Rev. Lett. 122, 213902 (2019). https://doi.org/10.1103/PhysRevLett.122.213902

[69] J. Yao, R. Zhu, and J. Yu, "Enhanced, fully connected 360 000-spin spatial photonic Ising machine," Phys. Rev. Appl. 24, 044087 (2025). https://doi.org/10.1103/c792-d3n9

[70] H. Li, G. Wang, E. Hasman, B. Wang, and X. Chen, "Erdos-Turan photonic Ising machines with record-high coupling resolution," arXiv:2510.01998 (2025). https://doi.org/10.48550/arXiv.2510.01998