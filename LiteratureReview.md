# Literature review: digital simulation of a boundary-damped \(XX/XY\) chain by Hamiltonian dilation

**Literature cutoff:** 9 September 2026  
**Repository problem reviewed:** `OpenQuantumSystem`, Model 1, with emphasis on Experiments 1–3

## Executive assessment

This repository studies the Markovian dynamics of an \(N\)-qubit \(XX\) (often called isotropic \(XY\)) chain with amplitude loss at its two ends. Its quantum algorithm replaces a short Lindblad time step by unitary evolution of the system and a reusable three-level ancilla, encoded in two qubits, and then discards or resets that ancilla. The system Hamiltonian is compiled into commuting even and odd bond layers of `XXPlusYYGate` operations. Two implementations are of particular interest: a fully coherent circuit that preserves the ancilla's three-state jump label until the end of a step, and a dynamic circuit that measures one ancilla bit after the first boundary-jump dilation and substitutes classical feed-forward for a nonlocal quantum control on the second boundary.

The literature establishes essentially every *individual ingredient* of this program. The GKSL equation, Kraus representations, and Stinespring dilation are foundational results.[^1][^2][^3][^4] Repeated-interaction or collision models have long generated Markovian channels by coupling to fresh ancillas and tracing them out.[^5][^6] Lie–Trotter decompositions of local Liouvillians and even–odd product formulas are standard and have rigorous error theories.[^7][^8] Most importantly, the first-order \((J+1)\)-level block-Hamiltonian construction used here already appears in Cleve and Wang's 2017 work and in subsequent quantum-computing demonstrations by de Jong and collaborators.[^9][^10][^11][^12] Ding, Li, and Lin's 2024 contribution is not the invention of that first-order block alone; it is a systematic stochastic-numerics-derived construction of arbitrarily high-order Hamiltonian dilations, with rigorous global error and ancilla bounds.[^13]

The *physical model* also has a very close—and, after parameter specialization, exact—classical precedent. Yamanaka and Sasamoto solve the real-time Lindbladian dynamics of an open \(XX\) chain with general gain and loss at both boundaries. Setting both reservoir polarizations to the pure-loss limit leaves precisely one lowering operator at each end, as in this repository.[^14] Therefore neither the boundary-loss model nor its classical solution should be presented as new. The model should also be called **boundary damped** or **two-ended loss**, not generically **boundary driven**: two lowering jumps feed no excitations into the chain, the vacuum is absorbing, and there is no reservoir bias sustaining a nonequilibrium current. The large boundary-driven transport literature usually studies unequal source and sink reservoirs, often with both raising and lowering jumps.[^15]

The closest end-to-end circuit precedent located is the May 2026 preprint by Tirado *et al.*[^16] It simulates interacting qubit chains with coherent \(XY\) exchange, ancilla-assisted dissipative channels, even/odd scheduling, mid-circuit measurement, reset, conditional gates, hardware-aware routing, and IBM hardware at a scale of 86 system qubits and 129 total qubits. That overlap is substantial and must be acknowledged prominently. Nevertheless, Tirado *et al.* use distributed pairwise ancillas and exact amplitude-damping/collective-decay channel blocks; they do not use the Ding–Li–Lin shared jump-label qutrit, do not specialize dissipation to two boundary losses, and do not implement the repository's particular conversion of the shared-qutrit inter-boundary control into a classical branch.

The resulting conservative novelty statement is:

> **No exact precedent was located in the documented search for the complete combination of a general-\(N\) boundary-loss \(XX\) chain, the first-order shared-\((J+1)\)-level Hamiltonian dilation with \(J=2\), even–odd `XXPlusYY` synthesis, and a dynamic-circuit transformation that measures the first jump label and classically conditions the second boundary dilation.**

That is an implementation-level and resource-engineering gap, not a claim of a new Lindblad model, a new first-order dilation formula, or a new use of reset and feed-forward. Its scientific value will depend on demonstrating a real advantage—compiled two-qubit depth, routing, error, or accessible system size—against the fully coherent shared-ancilla circuit, independent-ancilla amplitude-damping circuits, and the recent Tirado architecture.

## 1. The problem being simulated

### 1.1 Lindblad model

In the repository's current notation, the system Hamiltonian is

\[
H_S=\frac{J}{2}\sum_{n=0}^{N-2}
\left(X_nX_{n+1}+Y_nY_{n+1}\right)
+\frac{h}{2}\sum_{n=0}^{N-1}Z_n,
\]

and the two jump operators are

\[
V_1=\sqrt{\gamma}\,\sigma^-_0,
\qquad
V_2=\sqrt{\gamma}\,\sigma^-_{N-1},
\qquad
\sigma^- = \frac{X-iY}{2}.
\]

The density operator obeys

\[
\frac{d\rho}{dt}=\mathcal L(\rho)
=-i[H_S,\rho]
+\sum_{j=1}^{2}\left(
V_j\rho V_j^\dagger
-\frac12\{V_j^\dagger V_j,\rho\}
\right).
\tag{1}
\]

Equation (1) is a finite-dimensional GKSL master equation and therefore generates a completely positive, trace-preserving semigroup.[^1][^2] The \(XX\) interaction conserves excitation number, whereas both dissipators remove excitations. Consequently the all-zero state is absorbing; for generic finite parameters it is the natural long-time state. This differs qualitatively from a source–sink transport setup in which one bath injects and another removes excitations.

The nomenclature matters because “boundary-driven chain” commonly denotes a nonequilibrium system coupled to reservoirs of different polarizations, chemical potentials, or temperatures. Such models typically use four boundary jumps,

\[
\sqrt{\Gamma_L^+}\,\sigma_0^+,
\quad \sqrt{\Gamma_L^-}\,\sigma_0^-,
\quad \sqrt{\Gamma_R^+}\,\sigma_{N-1}^+,
\quad \sqrt{\Gamma_R^-}\,\sigma_{N-1}^-,
\]

and can support a steady current.[^15] The repository occupies the pure-loss corner \(\Gamma_L^+=\Gamma_R^+=0\), \(\Gamma_L^-=\Gamma_R^-=\gamma\). “Open \(XX\) chain with boundary dissipation” or “two-boundary amplitude damping” is therefore the precise description.

### 1.2 The computational target

For small \(N\), direct exponentiation of the \(4^N\times4^N\) Liouvillian or numerical integration of Eq. (1) supplies an exact classical benchmark. For general \(N\), storing the full density matrix is exponentially expensive. The repository instead asks whether local structure can be retained in a gate circuit with only \(N+2\) qubits: \(N\) system qubits and two qubits encoding a shared three-state environment. The intended workflow has three distinguishable errors:

1. **Dilation discretization error:** a finite-\(\Delta t\) channel approximates \(e^{\Delta t\mathcal L}\).
2. **Product-formula error:** noncommuting pieces of the dilated Hamiltonian, and the even and odd system layers, are ordered rather than jointly exponentiated.
3. **Implementation error:** finite shots, transpilation/routing, readout and reset errors, decoherence, and feed-forward latency alter the executed channel.

Those errors should be reported separately. Agreement on an ideal Aer simulator validates algebra and sampling; it is not evidence that reset or feed-forward will outperform a coherent circuit on a device.

## 2. Foundations: channels, dilations, and repeated interactions

### 2.1 From a GKSL generator to a short-time channel

Every finite-dimensional quantum channel has a Kraus representation

\[
\Phi(\rho)=\sum_{\mu=0}^{r-1}A_\mu\rho A_\mu^\dagger,
\qquad
\sum_\mu A_\mu^\dagger A_\mu=I,
\]

and can be realized as unitary evolution on a larger Hilbert space followed by a partial trace.[^3][^4] For a small GKSL step, one may choose to leading order

\[
A_0=I-i\Delta t H_S-
\frac{\Delta t}{2}\sum_jV_j^\dagger V_j+O(\Delta t^2),
\qquad
A_j=\sqrt{\Delta t}\,V_j+O(\Delta t^{3/2}).
\tag{2}
\]

Substitution into the Kraus sum gives

\[
\Phi_{\Delta t}(\rho)
=\rho+\Delta t\,\mathcal L(\rho)+O(\Delta t^2).
\tag{3}
\]

The square-root scaling in Eq. (2) is essential. A bounded system–ancilla interaction applied for time \(\Delta t\) ordinarily changes survival probabilities only quadratically at short time. A coupling of effective action \(O(\sqrt{\Delta t})\), followed by discarding a fresh ancilla, creates an \(O(\Delta t)\) dissipative probability. Repeated-interaction analyses make this weak-coupling/short-collision scaling explicit and bound the accumulated error.[^5]

### 2.2 Collision models and reset

Repeated-interaction models describe a system colliding sequentially with independently prepared environment units. Each collision is unitary; tracing out the used unit yields a channel, and replacing it with a fresh copy supplies the Markov property. Collision models can reproduce arbitrary multipartite Markovian dynamics with polynomial resources under locality assumptions.[^6] They also give a physical interpretation to the reset operation: reset is not a cosmetic gate but an entropy-export step that destroys system–environment memory.

For an ancilla \(A\) reused after each step, the ideal operation is

\[
\mathcal R_A(\rho_{SA})
=\operatorname{Tr}_A(\rho_{SA})\otimes|0\rangle\!\langle0|_A.
\tag{4}
\]

Omitting Eq. (4) generally changes the model from a memoryless semigroup to a finite-environment, potentially non-Markovian dynamics. Conversely, measuring the environment may produce a quantum-trajectory unraveling; averaging over all outcomes recovers the unconditioned channel when the measurement basis resolves the Kraus labels.[^17]

Collision-model circuits were demonstrated on IBM devices before the present project. García-Pérez *et al.* implemented one- and two-qubit open dynamics, including collisional models, on the IBM Q platform.[^18] Cattaneo *et al.* later demonstrated collective dissipation with collision-model circuits on IBM hardware.[^19] These works establish ancilla interaction, discard/reset, and hardware execution as prior art, but they use different channel decompositions and physical models.

## 3. Product formulas for open and closed dynamics

### 3.1 Liouvillian splitting

If a Liouvillian decomposes as \(\mathcal L=\sum_\alpha\mathcal L_\alpha\), a first-order channel product formula is

\[
e^{\Delta t\mathcal L}
=\prod_\alpha e^{\Delta t\mathcal L_\alpha}+O(\Delta t^2),
\]

with a global fixed-time error of first order in \(\Delta t\). Kliesch *et al.* proved local-Liouvillian Trotter bounds and used them to establish efficient unitary-circuit simulation of local Markovian dynamics.[^7] Later work developed universal channel decompositions, randomized formulas, and commutator-sensitive bounds.[^20][^21] A 2026 analysis obtains \(O(\sqrt N)\) Trotter-step scaling for locally interacting Lindbladians under its assumptions, illustrating why norm-only worst-case estimates can be overly pessimistic.[^22]

An important distinction is that a higher-order Hamiltonian Suzuki formula can contain negative time coefficients, whereas a generic Lindblad semigroup cannot be evolved for negative time as a CPTP map. Higher-order *channel* simulation therefore requires care; one cannot simply import every closed-system Suzuki formula. Ding–Li–Lin construct higher-order CPTP approximations through higher-order Kraus/Stinespring data, rather than by composing unphysical negative dissipative times.[^13]

### 3.2 Even–odd synthesis of the \(XX\) chain

Define bond terms

\[
H_n=\frac J2(X_nX_{n+1}+Y_nY_{n+1}),
\]

and split

\[
H_{\rm even}=\sum_{n\ \mathrm{even}}H_n,
\qquad
H_{\rm odd}=\sum_{n\ \mathrm{odd}}H_n.
\tag{5}
\]

Terms within either sum act on disjoint pairs and commute, so each layer exponentiates exactly and its two-qubit gates can execute in parallel. Only the even and odd layers fail to commute. Thus

\[
e^{-i\Delta t(H_{\rm even}+H_{\rm odd})}
=e^{-i\Delta tH_{\rm odd}}
e^{-i\Delta tH_{\rm even}}+O(\Delta t^2),
\tag{6}
\]

or, symmetrically,

\[
e^{-i\Delta tH_S}
=e^{-i\Delta tH_{\rm even}/2}
e^{-i\Delta tH_{\rm odd}}
e^{-i\Delta tH_{\rm even}/2}+O(\Delta t^3),
\tag{7}
\]

apart from the commuting field layer. Modern Trotter-error theory specifically shows that commutator structure and even–odd ordering in one-dimensional spin chains produce much tighter bounds than a naive sum-of-norms analysis.[^8]

Qiskit's `XXPlusYYGate(\theta,0)` uses

\[
R_{XX+YY}(\theta,0)
=\exp\!\left[-i\frac{\theta}{4}(X\otimes X+Y\otimes Y)\right].
\tag{8}
\]

Therefore a bond evolution under \(H_n\) for time \(\Delta t\) requires \(\theta_H=2J\Delta t\). A field term \((h/2)Z_n\) requires `RZ(h Δt)`. These convention checks are important: a factor-of-two error can preserve a visually plausible oscillation while simulating the wrong physical time.[^23]

The repository's Experiment 3 uses one even–odd ordering inside each outer dilation step. That is a coherent design choice: the outer discretization already controls \(\Delta t\), so an independent multi-step inner Trotter loop would introduce a second refinement parameter without changing the formal first-order target. It remains useful experimentally to separate the errors by comparing: exact \(e^{-i\Delta tH_S}\), the even–odd system formula alone, the unsplit dilated Hamiltonian, and the complete split channel.

## 4. The shared-ancilla Hamiltonian dilation

### 4.1 First-order block Hamiltonian

Let the ancilla have basis \(\{|0\rangle,|1\rangle,\ldots,|J\rangle\}\), where \(|0\rangle\) is the no-jump state and \(|j\rangle\) labels jump \(V_j\). The first-order dilated Hamiltonian can be written

\[
\widetilde H_{\Delta t}
=\sqrt{\Delta t}\,|0\rangle\!\langle0|\otimes H_S
+\sum_{j=1}^{J}
\left(
|j\rangle\!\langle0|\otimes V_j
+|0\rangle\!\langle j|\otimes V_j^\dagger
\right).
\tag{9}
\]

The unitary applied during one step is

\[
U_{\Delta t}=e^{-i\sqrt{\Delta t}\widetilde H_{\Delta t}}
=e^{-iK_{\Delta t}},
\]

where it is often clearer for circuit synthesis to absorb the outer factor and write

\[
K_{\Delta t}
=\Delta t\,|0\rangle\!\langle0|\otimes H_S
+\sqrt{\Delta t}\sum_{j=1}^{J}
\left(
|j\rangle\!\langle0|\otimes V_j
+\mathrm{h.c.}
\right).
\tag{10}
\]

Starting the ancilla in \(|0\rangle\), the block matrix elements of \(U_{\Delta t}\) are

\[
\begin{aligned}
\langle0|U_{\Delta t}|0\rangle
&=I-i\Delta tH_S-
\frac{\Delta t}{2}\sum_jV_j^\dagger V_j
+O(\Delta t^2),\\
\langle j|U_{\Delta t}|0\rangle
&=-i\sqrt{\Delta t}\,V_j+O(\Delta t^{3/2}).
\end{aligned}
\tag{11}
\]

Tracing the ancilla gives Eq. (3), hence local channel error \(O(\Delta t^2)\) and global fixed-time error \(O(T\Delta t)\). There is no postselection: all ancilla outcomes are retained.

### 4.2 Historical attribution

The block structure in Eqs. (9)–(11) is not unique to the 2024 Ding–Li–Lin paper. Cleve and Wang explicitly analyzed a \((J+1)\times(J+1)\) block Hamiltonian whose off-diagonal first column contains the Lindblad operators, evolved it for \(\sqrt{\Delta t}\), and traced the ancilla.[^9] They used this construction partly as a foil: repeated first-order dilation has \(O(T^2/\epsilon)\)-type step overhead, whereas their channel-LCU algorithm attains nearly linear time and polylogarithmic precision dependence under oracle assumptions. De Jong *et al.* then used closely related block/Stinespring constructions in IBM demonstrations and open-Schwinger-model studies.[^10][^11][^12]

Ding, Li, and Lin supply the rigorous systematic generalization. For order \(k\), they construct a dilated Hamiltonian with at most \((J+1)^{k+1}\) nonzero block components and

\[
a_k\le \left\lceil(k+1)\log_2(J+1)\right\rceil
=O(k\log(J+1))
\]

ancilla qubits, achieving global error

\[
O\!\left(T\,\|\mathcal L\|_{\rm be}^{k+1}\Delta t^k\right)
\]

under their stated bounded-input assumptions.[^13] They also cover time-dependent Lindbladians and demonstrate orders up to three numerically. The correct attribution for this repository is therefore: **Cleve–Wang-type first-order block dilation, placed in the Ding–Li–Lin arbitrary-order framework.** If only first order is implemented, the paper should not imply that the primitive itself first appeared in 2024.

### 4.3 Specialization to two boundary jumps

For \(J=2\), encode the qutrit in two ancilla qubits \(a_0,a_1\):

\[
|0\rangle_A=|00\rangle,
\qquad |1\rangle_A=|01\rangle,
\qquad |2\rangle_A=|10\rangle,
\]

leaving \(|11\rangle\) unused. The two jump generators are

\[
G_1=\sqrt\gamma
\left(|01\rangle\!\langle00|\otimes\sigma^-_0+
|00\rangle\!\langle01|\otimes\sigma^+_0\right),
\]

\[
G_2=\sqrt\gamma
\left(|10\rangle\!\langle00|\otimes\sigma^-_{N-1}+
|00\rangle\!\langle10|\otimes\sigma^+_{N-1}\right).
\tag{12}
\]

Within the allowed ancilla subspace, \(G_1\) is an exchange between \(a_0\) and system qubit \(0\), controlled on \(a_1=0\); \(G_2\) is an exchange between \(a_1\) and system qubit \(N-1\), controlled on \(a_0=0\). Since

\[
\sigma_A^+\sigma_S^-+\sigma_A^-\sigma_S^+
=\frac12(X_AX_S+Y_AY_S),
\]

each exchange is an `XXPlusYYGate` with angle

\[
\theta_J=2\sqrt{\gamma\Delta t},
\tag{13}
\]

plus the appropriate open control. The controls are not optional in a fully coherent qutrit implementation: without them, an exchange can act on the unused \(|11\rangle\) sector or mix one jump-label sector with another.

The system block in Eq. (10) is \(P_0\otimes H_S\), with \(P_0=|00\rangle\!\langle00|\). A literal coherent implementation therefore controls every system layer on both ancillas being zero. This is algebraically natural but hardware-expensive: all nominally parallel bond gates share the same controls, and routing controls from a shared ancilla to both ends of a long chain can dominate the compiled circuit.

## 5. Lie and Strang circuit constructions

Let

\[
K_H=\Delta t\,P_0\otimes H_S,
\qquad
K_1=\sqrt{\Delta t}\,G_1,
\qquad
K_2=\sqrt{\Delta t}\,G_2.
\]

The exact first-order dilation channel would use \(e^{-i(K_H+K_1+K_2)}\) before tracing the ancilla. A Lie product step uses the chronological circuit

\[
K_H\ \longrightarrow\ K_1\ \longrightarrow\ K_2
\ \longrightarrow\ \mathcal R_A,
\tag{14}
\]

meaning the total unitary is \(e^{-iK_2}e^{-iK_1}e^{-iK_H}\). Inside \(K_H\), the chain is itself ordered into even and odd layers. This nesting does not require multiple inner steps: both approximation levels use the same \(\Delta t\), while their commutators contribute to the one-step error coefficient.

A time-symmetric Strang step should place the system in the middle, as adopted in Experiment 2:

\[
\frac{K_1}{2}
\ \longrightarrow\
\frac{K_2}{2}
\ \longrightarrow\
K_H
\ \longrightarrow\
\frac{K_2}{2}
\ \longrightarrow\
\frac{K_1}{2}
\ \longrightarrow\ \mathcal R_A.
\tag{15}
\]

The palindrome cancels odd ordering asymmetries at the unitary product-formula level. However, the accuracy label for the *reduced Lindblad channel* should be justified by a channel expansion, not inferred solely from the visual symmetry. A first-order dilated Hamiltonian can remain the ultimate accuracy bottleneck even when its exponent is more accurately split. Genuine second-order Lindblad simulation in Ding–Li–Lin includes correction blocks beyond merely applying a Strang formula to the first-order Hamiltonian.[^13]

For resource reporting, “gate count” should include every circuit operation that occupies the schedule or implements the channel, including measurement and reset. Useful separate metrics are: total operation count, one-qubit count, entangling-gate count, measurement count, reset count, classical-control operations, logical depth, two-qubit depth, and scheduled duration. Pre- and post-transpilation figures should be shown both for one step and for the full repeated circuit, because cancellation and routing do not scale linearly from a single isolated step.

## 6. Dynamic-circuit optimization

### 6.1 Why the controls can be changed

The dynamic Lie circuit exploits facts that hold at a specific point in the algorithm, rather than replacing controlled gates by uncontrolled gates globally.

First, Eq. (4) guarantees that every step begins in ancilla state \(|00\rangle\). Because \(K_H\) is applied first in Eq. (14), the projector \(P_0\) is certain on input. The ancilla controls on the system even/odd and field layers can therefore be removed without changing the state reached in that step.

Second, immediately before \(K_1\), ancilla \(a_1\) is still certainly zero. The open control of the first exchange on \(a_1=0\) can likewise be removed. After \(K_1\), the joint state has support only in the \(|00\rangle\) and \(|01\rangle\) sectors.

Third, the remaining control for \(K_2\) is essential: \(G_2\) must act on the no-jump sector \(|00\rangle\), not on the first-jump sector \(|01\rangle\). The dynamic construction measures \(a_0\) after \(K_1\). If the result is \(m=0\), it applies the otherwise uncontrolled second exchange between \(a_1\) and system qubit \(N-1\); if \(m=1\), it skips that exchange. Both ancilla qubits are then reset.

This is not merely heuristic. Write the post-\(K_1\) density operator in qutrit blocks,

\[
\rho_{SA}=\sum_{r,s\in\{0,1\}}|r\rangle\!\langle s|_A\otimes\rho_{rs}.
\]

The unitary \(e^{-iK_2}\) is block diagonal with respect to “sector 1” versus the \(\{0,2\}\) subspace: it is identity on label 1 and acts only between labels 0 and 2. Measurement of whether the ancilla is in sector 1 deletes \(\rho_{01}\) and \(\rho_{10}\). Those coherences make no contribution after the final ancilla trace in the coherent circuit. Applying \(K_2\) only on the \(m=0\) branch and averaging the classical outcomes therefore produces the same reduced system channel as the coherently controlled \(K_2\), in the ideal model. The equivalence would fail if a later gate recombined the jump-label sectors before the trace, if the measurement coarse-grained incompatible sectors, or if the measurement record were postselected rather than averaged.

### 6.2 Architectural benefit

The coherent qutrit encoding creates a particularly awkward geometry. The control \(a_1=0\) for the left-boundary jump and \(a_0=0\) for the right-boundary jump couple a shared register across opposite ends of the system chain. The projector control on \(K_H\) also touches every bond. On restricted-connectivity hardware, decomposition and SWAP routing can erase the abstract advantage of using only two ancilla qubits.

The dynamic transformation removes both controls from the system evolution, removes the unused control from \(K_1\), and replaces the inter-boundary quantum control of \(K_2\) by a classical bit. That permits \(a_0\) to be placed near the left boundary and \(a_1\) near the right boundary. In an ideal connectivity model it reduces controlled-gate complexity; in a heavy-hex layout it may additionally eliminate a long routing path. This is the repository's most defensible implementation-level contribution.

The cost is that measurement and classical control are physical operations. A fair comparison must use scheduled duration and observed error, not gate count alone. Mid-circuit readout can dephase spectator qubits, create measurement-induced entangling errors, and expose the system to idle \(T_1/T_2\) decay.[^24][^25] Assignment error selects the wrong feed-forward branch; reset error corrupts the next collision; and residual photons or leakage can affect neighboring qubits. Readout-error mitigation for feed-forward circuits can reduce branch errors without increasing two-qubit depth, but it adds calibration and sampling/post-processing requirements.[^26] Dynamical decoupling may protect idle system qubits, although both benchmarking work and Tirado *et al.* show that its benefit can be strongly qubit dependent.[^16][^25]

IBM's current dynamic-circuit stack supports mid-circuit measurement and `if_test` control flow, but backend-dependent restrictions, controller memory, branch size, measurement buffering, and latency remain relevant.[^27] Hardware documentation reports substantial recent latency improvements, yet software-level depth alone does not encode that latency.[^28] The correct prospective experiment is therefore a crossover study: determine the \(N\), layout, and calibration regime in which avoided coherent controls and SWAPs outweigh measurement, reset, and idle errors.

### 6.3 Aer as a verification environment

Qiskit Aer supports statevector, density-matrix, matrix-product-state, and tensor-network simulation methods, and its shot-branching option is explicitly designed for dynamic circuits containing measurement, reset, and classical branches.[^29] Aer can verify:

- equality of the coherent and dynamic reduced channels in the noiseless limit;
- shot convergence and classical-branch frequencies;
- sensitivity to a backend-derived noise model;
- circuit functionality after transpilation.

It cannot by itself establish hardware advantage. Backend noise models may omit nonstationary drift, measurement-induced spectator errors, correlated reset effects, and exact feed-forward timing. Results should be labeled “ideal Aer,” “Aer with specified noise model,” or “IBM hardware,” never simply “quantum simulation.”

## 7. Physical-model literature

### 7.1 Exact solution of the same loss model

Quadratic fermionic Lindbladians can be solved by “third quantization”: a quadratic Hamiltonian with Lindblad operators linear in fermionic variables reduces to diagonalization of a \(4N\times4N\) structure matrix.[^30] The \(XX\) chain maps by Jordan–Wigner transformation to free fermions, and boundary raising/lowering operators are linear at the ends. Yamanaka and Sasamoto exploit this to derive exact time-dependent magnetization and current formulas for an open \(XX\) chain with general boundary dissipation.[^14]

Their model includes independently tunable boundary coupling strengths and reservoir polarizations, giving raising and lowering operators at each end. Setting their \(\mu_{\mathrm L}=\mu_{\mathrm R}=-1\) eliminates the raising jumps; then choosing equal boundary strengths leaves precisely the two equal-rate lowering jumps in Eq. (1), up to the paper's Hamiltonian and rate conventions. They further reduce the \(XX\) problem to an \(N\times N\) non-Hermitian matrix and derive an \(O(N^{-3})\) Liouvillian-gap scaling.

This precedent is valuable rather than damaging. It supplies a scalable exact or semianalytic benchmark beyond brute-force density matrices. A strong validation program should compare quantum-circuit observables with the Yamanaka–Sasamoto solution at \(N\) where the full \(4^N\) Liouvillian is already inconvenient. Because the model is free-fermionic, however, it is not a plausible near-term quantum-advantage target by itself. The value of the experiment is as a controlled benchmark for a general Lindblad-circuit method that can later be applied to interactions, disorder, nonquadratic jumps, or higher-dimensional geometries.

### 7.2 Boundary loss versus boundary drive

Boundary-driven \(XX/XY/XXZ\) chains are a large literature covering ballistic or diffusive transport, nonequilibrium steady states, rectification, and phase transitions.[^15] Exact steady-state matrix-product solutions and repeated-interaction derivations are well established.[^31][^32] Those results should not be imported uncritically into the repository's loss-only case. With two pure sinks and no source, any transient current reflects redistribution and escape of the initial excitation content, not stationary transport between biased reservoirs.

This distinction also affects experimental observables. Appropriate quantities include total excitation survival, site occupations, boundary emission probabilities, light-cone arrival times, transient current, and approach to vacuum. A nonequilibrium steady-state current should not be advertised unless raising terms or coherent pumping are added. Extending the code to four gain/loss jumps would connect directly to the canonical boundary-driven setting but would increase the shared ancilla from five logical labels (one no-jump plus four jumps), requiring at least three ancilla qubits in a literal first-order encoding.

## 8. Digital open-system simulation: closest method families

### 8.1 Block/Stinespring hardware demonstrations

De Jong *et al.* presented an IBM-device demonstration of open-system evolution motivated by heavy-ion physics, using ancilla-based dilation and error mitigation.[^10] Follow-up work on the Schwinger model used Stinespring dilation for thermal and dissipative dynamics and analyzed Trotter error.[^11][^12] These papers are closer to the repository's mathematical primitive than collision circuits built from a separately derived exact amplitude-damping channel. Their system Hamiltonians, jump structures, and circuit optimizations differ, and they do not report the two-boundary shared-qutrit dynamic conversion developed here.

Mohammadipour and Li directly analyze first-order Kraus and dilated-Hamiltonian Lindblad integrators and use step-size extrapolation to trade multiple circuit ensembles for much smaller maximum depth. Their published bounds reduce polynomial precision dependence of maximum depth to polylogarithmic dependence while retaining the usual \(1/\epsilon^2\) sampling scale.[^33] This is especially relevant because it can be layered on top of the repository without changing the one-step construction: execute several \(\Delta t\) grids and extrapolate observables. It is a more principled depth-reduction baseline than simply increasing a nominal “Trotter order” without including the higher-order dilation corrections.

### 8.2 Exact amplitude-damping channel circuits

For a single qubit, amplitude damping over time \(\Delta t\) has probability

\[
p=1-e^{-\gamma\Delta t}
\]

and a one-ancilla Stinespring circuit can implement that channel exactly. Digital spin-boson and collision-model studies alternate such dissipative blocks with coherent evolution and reset.[^34] If the dissipators are split from the Hamiltonian, the repository's two boundary losses can likewise be implemented with one reusable ancilla—or two boundary-local ancillas for parallelism—without a shared qutrit. The dissipative substep is then exact in \(p\), while Hamiltonian/dissipator splitting remains approximate.

This is a critical baseline. The shared-qutrit Hamiltonian dilation is general and aligns with Ding–Li–Lin, but for two simple amplitude-damping jumps it may not minimize hardware resources. A convincing study should compare:

- shared-qutrit coherent dilation;
- shared-qutrit dynamic feed-forward dilation;
- sequential exact amplitude-damping channels using one reusable ancilla;
- two parallel boundary-local amplitude-damping ancillas;
- a channel-level classical reference.

The comparison should hold the physical splitting order and target error fixed.

### 8.3 Repeated interactions and stochastic trajectories

Pocrnic, Segal, and Wiebe give rigorous error and resource bounds for Lindbladian simulation through repeated interactions, including Trotter–Suzuki and qubitization implementations.[^5] Di Bartolomeo *et al.* propose stochastic simulation with a single environmental qubit and sampling overhead independent of system size.[^35] Garg *et al.* compare Hamiltonian-simulation strategies for collision models and include a ten-qubit \(XX\)-Heisenberg chain under amplitude damping, using at most \(n+2\) qubits.[^36] These are close in resource philosophy but not exact precedents: their ancilla organization, stochastic averaging, bath model, or damping placement differs.

Quantum trajectories offer another natural route. Rather than store a mixed state coherently, each shot follows a pure-state non-Hermitian evolution interrupted by stochastic jumps, and observables are recovered by ensemble averaging. Recent algorithms improve query complexity for suitable Lindbladians.[^37] Trajectories can reduce quantum width but move cost into the number of circuit realizations and may require nonunitary-state preparation or QITE-like steps. They are best treated as a complementary accuracy–sampling tradeoff, not as the same algorithm.

### 8.4 Variational, QITE, randomized, and algebraic methods

Kamakari *et al.* adapt quantum imaginary-time evolution to vectorized and variational representations of Lindblad dynamics and demonstrate spontaneous emission and a dissipative transverse-field Ising model on IBM hardware.[^38] Chen *et al.* develop an adaptive variational simulator and validate it on IBM processors.[^39] These approaches can produce shorter circuits but introduce ansatz expressivity, optimization, and measurement overheads absent from a direct product formula.

Randomized schemes decompose a Lindbladian into easily simulated single-jump pieces and sample them in a qDRIFT-like manner.[^40] Other recent algorithms use random compilation, two-ancilla logarithmic-precision constructions, mixed-unitary adjoint channels, or minimal-ancilla randomized dissipation.[^41][^42][^43] Their asymptotic objectives are important for fault-tolerant or early-fault-tolerant machines, but their oracle and sampling assumptions do not automatically translate into shallower near-term circuits for this small, local jump set.

QFlux, released as a 2026 preprint/tutorial toolkit, includes damped spin-chain examples and quantum circuits based on Kraus extraction and Sz.-Nagy dilation.[^44] Its spin-chain example is physically adjacent, but it diagonalizes a finite-time propagator/Choi representation and dilates individual nonunitary operators. It is not the local shared-jump-label Hamiltonian construction in Eqs. (9)–(13), and its generic circuit synthesis has different scaling and locality properties.

## 9. The closest dynamic-circuit precedent: Tirado et al. (2026)

Tirado *et al.* deserve a direct comparison because a keyword-only novelty search could easily miss how close their circuit architecture is.[^16] Their preprint considers a one-dimensional chain of interacting emitters with onsite terms and coherent \(XY\)-type exchange. Markovian evolution is Trotterized into coherent and dissipative pieces. The dissipative circuit contains local amplitude damping and nearest-neighbor collective/cross-decay channels. Disjoint dissipative blocks are scheduled in even and odd layers. Ancillas are measured and reset; mid-circuit outcomes control later gates. Hardware-aware layout, dynamical decoupling, and biased Clifford data regression are used, and results are validated with a Monte Carlo–TEBD method that treats resets through stochastic trajectories.

The reported scale is far beyond this repository's current Aer experiments: up to 86 system qubits on IBM System Two `ibm_basquecountry`, 129 total qubits including ancillas, and roughly 8000 two-qubit gates in the largest circuits. This is direct evidence that open-system dynamic circuits, reset, conditional operations, even/odd scheduling, and interacting chains can be combined on contemporary IBM hardware.

There are nevertheless four structural differences.

1. **Dilation:** Tirado *et al.* compile exact local amplitude-damping and correlated dissipative channel blocks. The repository derives a short-time channel from the Cleve–Wang/Ding–Li–Lin Hamiltonian dilation.
2. **Ancilla organization:** Tirado *et al.* distribute one ancilla across a local system pair and reduce the total to roughly \(\lfloor N/2\rfloor\). The repository uses exactly two ancilla qubits to encode one shared qutrit, independent of \(N\), but pays for nonlocal controls.
3. **Dissipators:** Tirado *et al.* model local decay throughout the chain and collective nearest-neighbor decay. The repository has two independent loss jumps only at sites \(0\) and \(N-1\).
4. **Feed-forward identity:** Tirado *et al.* dynamically simplify controlled channel blocks. The repository's specific identity measures the first shared jump label and classically guards the second, spatially distant boundary exchange.

Thus Tirado *et al.* are a **near precedent**, not an exact precedent. They sharply reduce the breadth of any novelty claim: dynamic dissipative chains on IBM hardware are established. The remaining question is narrower and testable—whether a shared qutrit can be made hardware-competitive by exploiting reset-certainty and replacing its long-range quantum sector control with classical feed-forward.

## 10. Precedent matrix

“Exact overlap” below means overlap with the complete repository combination, not merely with its physical model or one circuit primitive.

| Work | Physical model | Dilation/channel type | Ancilla organization | System-size scaling | Product formula | Reset/trace | Feed-forward | Platform | Exact overlap with repository |
|---|---|---|---|---|---|---|---|---|---|
| Cleve & Wang (2017)[^9] | General Lindblad | First-order block Hamiltonian plus channel-LCU algorithm | Shared \((J+1)\)-level label register | General \(n,J\) | Time slicing | Yes | No | Algorithm/theory | **Exact first-order primitive**, not the chain or dynamic circuit |
| de Jong *et al.* (2021)[^10] | Heavy-ion-inspired open system | Block/Stinespring Hamiltonian dilation | Shared jump-label ancilla | Small demonstration | Hamiltonian/dissipative splitting | Yes | No | IBM hardware | Same method family; different model and optimization |
| de Jong *et al.* (2022); Lee *et al.* (2023)[^11][^12] | Open Schwinger model | Stinespring/block dilation | Ancilla register | Small-\(N\) studies | Trotterized \(H\) and dissipation | Yes | No | IBM simulator/device and resources | Same method family; different many-body model |
| Ding, Li & Lin (2024)[^13] | General, including time dependent | Arbitrary-order Hamiltonian dilation | \(O(k\log(J+1))\) ancillas | General | Dilation-derived high order | Trace/measure | No specific hardware feed-forward | Theory/numerics | The repository instantiates its **first-order framework** |
| Yamanaka & Sasamoto (2023)[^14] | \(N\)-site \(XX\), general gain/loss at both boundaries | Classical third quantization | None | \(N\times N\) reduction for \(XX\) | None | N/A | N/A | Classical exact solution | **Exact physical model after pure-loss specialization**, no circuit |
| Cattaneo *et al.* (2021, 2023)[^6][^19] | General multipartite; collective dissipation demo | Collision model | Fresh/reused local ancillas | Polynomial; small hardware demo | Collision ordering | Yes | Not the repository identity | IBM hardware/theory | Reset and dissipative channels established; different dilation/model |
| Pocrnic, Segal & Wiebe (2025)[^5] | General Lindblad | Repeated interaction | Bath units/fresh ancillas | General bounds | Trotter or qubitization | Trace after collision | No | Theory/numerics | Similar weak-coupling logic, different construction |
| Garg *et al.* (2025)[^36] | Ten-qubit \(XX\)-Heisenberg with amplitude damping | Collision maps + Hamiltonian simulation | At most \(n+2\) total qubits | General \(n\); \(n=10\) comparison | Several Hamiltonian formulas | Yes | No | Early-FT resource study | Very close components; damping and ancilla structure differ |
| Mohammadipour & Li (2025)[^33] | General Lindblad examples | First-order Kraus and Hamiltonian-dilation primitives | Primitive dependent | General bounds | Step-size grids + extrapolation | Yes | No | Theory/noisy numerics | Directly applicable accuracy/depth enhancement, not the chain circuit |
| QFlux Part IV (2026 preprint)[^44] | Spin chains with sitewise damping/dephasing | Choi/Kraus extraction + Sz.-Nagy dilation | Per nonunitary operator | Demonstrated \(N=3\) spin chain | Closed-system Trotter plus channel workflow | Measurement/postselection workflow | No | Simulator/toolkit | Adjacent model and dilation, not shared qutrit/local DLL block |
| Pillay *et al.* (2026)[^45] | \(XX\) chain with four boundary gain/loss jumps and local dephasing | Liouvillian product formulas | Method dependent | General resource bounds | Deterministic/randomized first/second order | Channel dependent | No | Classical resource optimization | Exact \(XX\) layer context, but driven/dephasing model and no shared-qutrit circuit |
| Li, Li & Yan (2026)[^46] | Two qubits in common environment; two-body dissipator | Collision-model dissipator engineering | Effective multilevel environment | Two system qubits | Stroboscopic collision sequence | Environment traced/reset conceptually | No matching shared-label branch | Superconducting-circuit simulation | Multilevel/dissipator component only; not boundary loss or general \(N\) |
| Tirado *et al.* (2026 preprint)[^16] | Interacting chain, local and collective decay | Exact ancilla-assisted dissipative channels | Distributed, one ancilla per active pair | 86 system / 129 total qubits | Coherent and dissipative even/odd layers | Measure and reset | **Yes** | IBM Heron hardware | **Closest end-to-end near precedent**, but not shared qutrit or boundary-only DLL construction |

## 11. Tiered novelty verdict

### Tier 1 — Exact precedent

No exact end-to-end precedent was located in the documented search. In particular, no primary paper was found that simultaneously uses:

- the pure-loss \(N\)-site \(XX\) chain with exactly two boundary lowering jumps;
- a single shared three-level no-jump/jump-label ancilla encoded in two qubits;
- the first-order block-Hamiltonian dilation;
- direct \(XX+YY\) synthesis and even–odd system layers;
- ancilla reuse by measurement/reset at every step; and
- measurement of the first jump label to classically condition the opposite-boundary second jump.

This is a search result, not a proof of nonexistence. The defensible wording is “no exact precedent located in the documented search through 9 September 2026.”

### Tier 2 — Near precedents

Tirado *et al.* are the closest circuit-level precedent because they combine interacting \(XY\) chains, dissipative channel Trotterization, even/odd scheduling, measurement, reset, feed-forward, and IBM hardware at utility scale.[^16] Garg *et al.* combine an \(XX\)-Heisenberg chain, amplitude damping, collision maps, product-formula Hamiltonian simulation, and an \(n+2\)-qubit resource ceiling.[^36] De Jong and collaborators use the same broad block/Stinespring Hamiltonian-dilation lineage on IBM hardware.[^10][^11][^12] QFlux supplies a generic damped-spin-chain dilation toolkit, albeit through global Kraus/Sz.-Nagy synthesis.[^44]

### Tier 3 — Established ingredients

The following are established and should not be claimed independently as novel:

- the GKSL model, Kraus form, and unitary dilation;
- boundary-dissipated \(XX/XY\) chains and the loss-only parameter limit;
- the first-order \((J+1)\)-level block-Hamiltonian dilation and \(\sqrt{\Delta t}\) coupling;
- arbitrary-order Hamiltonian dilation in the Ding–Li–Lin framework;
- Lie, Strang, and even–odd product formulas;
- exchange-gate realization of \(XX+YY\) evolution;
- ancilla trace/reset between Markovian steps;
- mid-circuit measurement and conditional gates for dissipative channels;
- Aer simulation and IBM execution of open-system circuits.

### Tier 4 — Defensible remaining gap

The defensible gap is a **hardware-aware specialization and dynamic-circuit equivalence**: derive the shared-qutrit circuit for two spatially separated boundary jumps, identify which controls are redundant because reset fixes the incoming ancilla sector, replace the remaining cross-boundary sector control with measurement and classical feed-forward, and quantify the resulting resource/error tradeoff as \(N\) grows.

That gap becomes a research contribution only if supported quantitatively. A useful claim would be:

> For boundary-local jump operators represented by a shared jump-label ancilla, early measurement of a completed jump-label sector can convert a nonlocal coherent exclusivity control into local quantum gates plus feed-forward, preserving the reduced channel while reducing a specified compiled resource on restricted-connectivity hardware.

This statement is narrower, clearer, and more testable than “a new algorithm for open quantum systems.”

## 12. Recommended validation and research program

### 12.1 Prove channel equality explicitly

Give Kraus operators for both the coherent and dynamic one-step maps and prove equality after outcome averaging. The proof should specify the qutrit encoding, unused \(|11\rangle\) sector, gate ordering, classical condition, and reset map. A numerical Choi-matrix comparison to machine precision for random input states should accompany the algebra.

### 12.2 Resolve the error budget

For each \((N,\Delta t,T)\), compare:

1. exact classical \(e^{T\mathcal L}\) or the free-fermion solution;
2. exact exponentiation of the complete first-order dilated Hamiltonian per step;
3. Lie versus palindromic split dilation;
4. even–odd system synthesis;
5. coherent versus dynamic circuit on ideal Aer;
6. shot-based Aer;
7. backend-noise Aer; and
8. IBM hardware.

Report trace distance or diamond/Choi distance for one-step channel tests and observable error for long-time tests. Fit convergence slopes rather than identifying order from circuit symmetry alone.

### 12.3 Compare architectures fairly

The strongest baselines are exact boundary amplitude-damping channels, because the repository's jumps are unusually simple. Report \(N+2\) shared-qutrit width against \(N+1\) sequential and \(N+2\) parallel boundary-local damping circuits. Compare pretranspilation operation counts, post-transpilation native entangling counts, two-qubit depth, SWAP count, measurement/reset count, classical branch count, and scheduled duration on the *same backend calibration and placement policy*.

### 12.4 Use the model's solvability as an asset

The loss-only \(XX\) chain is classically tractable, so it is an excellent calibration ladder rather than a quantum-advantage endpoint. Increase \(N\) while retaining exact two-point-function or third-quantization benchmarks; test light-cone propagation, excitation survival, and the \(N^{-3}\) slow relaxation scale. After the implementation is validated, add a \(ZZ\) interaction, nonlinear jump, disorder, or a genuine source–sink boundary bias to leave the free-fermion regime.

### 12.5 Hardware-specific dynamic tests

Before a full dynamics run, benchmark isolated primitives on candidate layouts:

- coherent open-controlled exchange versus measurement-conditioned exchange;
- spectator dephasing during boundary measurement;
- branch-assignment and reset fidelity;
- feed-forward latency and scheduled idle duration;
- dynamical-decoupling benefit on each spectator qubit;
- repeated reset drift over the intended number of steps.

This will reveal whether the dynamic circuit is actually favorable on a given calibration day. The answer may differ by system size and layout, which is itself a meaningful result.

## 13. Search audit and limitations

The search began with the repository's `writeup.pdf`, Experiment 1 mathematical reference, and Experiment 2/3 circuit descriptions. An initial search-and-reading-only interval of more than 30 minutes preceded drafting. Backward and forward searches were then performed around Ding–Li–Lin's PRX Quantum paper, including the APS cited-by index available at the cutoff. Primary publisher pages and/or arXiv records were checked for the close works in the precedent matrix. Sources consulted included APS journals, Springer Nature, SciPost, IOP/Quantum, Dagstuhl proceedings, arXiv, institutional repositories, IBM Research/Quantum documentation, Qiskit Aer documentation, and general scholarly web indexes.

Query families combined terms from five groups:

- `Lindblad`, `GKSL`, `Hamiltonian dilation`, `block Hamiltonian`, `Stinespring`, `Kraus`, `shared ancilla`, `qutrit`, `sqrt(dt)`;
- `XX chain`, `XY chain`, `boundary dissipation`, `boundary loss`, `amplitude damping`, `two boundary jumps`, `boundary driven`;
- `Lie Trotter`, `Strang`, `Suzuki`, `even odd`, `commutator bounds`, `XXPlusYYGate`;
- `collision model`, `repeated interaction`, `quantum trajectory`, `LCU`, `QITE`, `variational`, `randomized Lindblad`;
- `mid-circuit measurement`, `reset`, `feed-forward`, `dynamic circuit`, `spectator error`, `IBM hardware`, and `Aer`.

Inclusion favored original papers that established a method, analyzed the exact physical model, demonstrated related hardware circuits, or materially constrained novelty. Reviews were used for orientation and terminology, not as the sole support for a substantive priority claim. Preprints are labeled. Works dated after 9 September 2026 were excluded. “No exact precedent located” is qualified because scholarly indexes can miss non-English papers, theses, workshop material, unpublished code, or papers whose abstracts use unrelated terminology. No patent search was performed. The 2026 Tirado preprint and other very recent papers may change during peer review.

## References

[^1]: G. Lindblad, “On the Generators of Quantum Dynamical Semigroups,” *Communications in Mathematical Physics* **48**, 119–130 (1976), DOI: [10.1007/BF01608499](https://doi.org/10.1007/BF01608499).

[^2]: V. Gorini, A. Kossakowski, and E. C. G. Sudarshan, “Completely Positive Dynamical Semigroups of \(N\)-Level Systems,” *Journal of Mathematical Physics* **17**, 821–825 (1976), DOI: [10.1063/1.522979](https://doi.org/10.1063/1.522979).

[^3]: W. F. Stinespring, “Positive Functions on \(C^*\)-Algebras,” *Proceedings of the American Mathematical Society* **6**, 211–216 (1955), DOI: [10.1090/S0002-9939-1955-0069403-4](https://doi.org/10.1090/S0002-9939-1955-0069403-4).

[^4]: K. Kraus, “General State Changes in Quantum Theory,” *Annals of Physics* **64**, 311–335 (1971), DOI: [10.1016/0003-4916(71)90108-4](https://doi.org/10.1016/0003-4916(71)90108-4).

[^5]: M. Pocrnic, D. Segal, and N. Wiebe, “Quantum Simulation of Lindbladian Dynamics via Repeated Interactions,” *Journal of Physics A: Mathematical and Theoretical* **58**, 305302 (2025), DOI: [10.1088/1751-8121/adebc4](https://doi.org/10.1088/1751-8121/adebc4), [arXiv:2312.05371](https://arxiv.org/abs/2312.05371).

[^6]: M. Cattaneo, G. De Chiara, S. Maniscalco, R. Zambrini, and G. L. Giorgi, “Collision Models Can Efficiently Simulate Any Multipartite Markovian Quantum Dynamics,” *Physical Review Letters* **126**, 130403 (2021), DOI: [10.1103/PhysRevLett.126.130403](https://doi.org/10.1103/PhysRevLett.126.130403), [arXiv:2010.13910](https://arxiv.org/abs/2010.13910).

[^7]: M. Kliesch, T. Barthel, C. Gogolin, M. Kastoryano, and J. Eisert, “Dissipative Quantum Church–Turing Theorem,” *Physical Review Letters* **107**, 120501 (2011); erratum **109**, 119904 (2012), DOI: [10.1103/PhysRevLett.107.120501](https://doi.org/10.1103/PhysRevLett.107.120501), [arXiv:1105.3986](https://arxiv.org/abs/1105.3986).

[^8]: A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, “Theory of Trotter Error with Commutator Scaling,” *Physical Review X* **11**, 011020 (2021), DOI: [10.1103/PhysRevX.11.011020](https://doi.org/10.1103/PhysRevX.11.011020), [arXiv:1912.08854](https://arxiv.org/abs/1912.08854).

[^9]: R. Cleve and C. Wang, “Efficient Quantum Algorithms for Simulating Lindblad Evolution,” in *44th International Colloquium on Automata, Languages, and Programming (ICALP 2017)*, LIPIcs **80**, Article 17 (2017), DOI: [10.4230/LIPIcs.ICALP.2017.17](https://doi.org/10.4230/LIPIcs.ICALP.2017.17), [arXiv:1612.09512](https://arxiv.org/abs/1612.09512).

[^10]: W. A. de Jong, M. Metcalf, J. Mulligan, M. Płoskoń, F. Ringer, and X. Yao, “Quantum Simulation of Open Quantum Systems in Heavy-Ion Collisions,” *Physical Review D* **104**, L051501 (2021), DOI: [10.1103/PhysRevD.104.L051501](https://doi.org/10.1103/PhysRevD.104.L051501), [arXiv:2010.03571](https://arxiv.org/abs/2010.03571).

[^11]: W. A. de Jong, K. Lee, J. Mulligan, M. Płoskoń, F. Ringer, and X. Yao, “Quantum Simulation of Nonequilibrium Dynamics and Thermalization in the Schwinger Model,” *Physical Review D* **106**, 054508 (2022), DOI: [10.1103/PhysRevD.106.054508](https://doi.org/10.1103/PhysRevD.106.054508), [arXiv:2106.08394](https://arxiv.org/abs/2106.08394).

[^12]: K. Lee, J. Mulligan, F. Ringer, and X. Yao, “Liouvillian Dynamics of the Open Schwinger Model: String Breaking and Kinetic Dissipation in a Thermal Medium,” *Physical Review D* **108**, 094518 (2023), DOI: [10.1103/PhysRevD.108.094518](https://doi.org/10.1103/PhysRevD.108.094518), [arXiv:2308.03878](https://arxiv.org/abs/2308.03878).

[^13]: Z. Ding, X. Li, and L. Lin, “Simulating Open Quantum Systems Using Hamiltonian Simulations,” *PRX Quantum* **5**, 020332 (2024), DOI: [10.1103/PRXQuantum.5.020332](https://doi.org/10.1103/PRXQuantum.5.020332), [arXiv:2311.15533](https://arxiv.org/abs/2311.15533).

[^14]: K. Yamanaka and T. Sasamoto, “Exact Solution for the Lindbladian Dynamics for the Open XX Spin Chain with Boundary Dissipation,” *SciPost Physics* **14**, 112 (2023), DOI: [10.21468/SciPostPhys.14.5.112](https://doi.org/10.21468/SciPostPhys.14.5.112), [arXiv:2104.11479](https://arxiv.org/abs/2104.11479).

[^15]: G. T. Landi, D. Poletti, and G. Schaller, “Nonequilibrium Boundary-Driven Quantum Systems: Models, Methods, and Properties,” *Reviews of Modern Physics* **94**, 045006 (2022), DOI: [10.1103/RevModPhys.94.045006](https://doi.org/10.1103/RevModPhys.94.045006), [arXiv:2104.14350](https://arxiv.org/abs/2104.14350).

[^16]: B. Tirado, J. Fraxanet, A. Juan-Delgado, J. Aizpurua, and R. Esteban, “Utility-Scale Quantum Experiments Using Dynamic Circuits to Address Collective Dissipation in Interacting Qubits,” arXiv preprint (2026), [arXiv:2605.25830](https://arxiv.org/abs/2605.25830).

[^17]: F. Ciccarello, S. Lorenzo, V. Giovannetti, and G. M. Palma, “Quantum Collision Models: Open System Dynamics from Repeated Interactions,” *Physics Reports* **954**, 1–70 (2022), DOI: [10.1016/j.physrep.2022.01.001](https://doi.org/10.1016/j.physrep.2022.01.001), [arXiv:2106.11974](https://arxiv.org/abs/2106.11974).

[^18]: G. García-Pérez, M. A. C. Rossi, B. Sokolov, E.-M. Borrelli, and S. Maniscalco, “IBM Q Experience as a Versatile Experimental Testbed for Simulating Open Quantum Systems,” *npj Quantum Information* **6**, 1 (2020), DOI: [10.1038/s41534-019-0235-y](https://doi.org/10.1038/s41534-019-0235-y), [arXiv:1909.07068](https://arxiv.org/abs/1909.07068).

[^19]: M. Cattaneo, M. A. C. Rossi, G. García-Pérez, R. Zambrini, and S. Maniscalco, “Quantum Simulation of Dissipative Collective Effects on Noisy Quantum Computers,” *PRX Quantum* **4**, 010324 (2023), DOI: [10.1103/PRXQuantum.4.010324](https://doi.org/10.1103/PRXQuantum.4.010324), [arXiv:2201.11597](https://arxiv.org/abs/2201.11597).

[^20]: R. Sweke, I. Sinayskiy, D. Bernard, and F. Petruccione, “Universal Simulation of Markovian Open Quantum Systems,” *Physical Review A* **91**, 062308 (2015), DOI: [10.1103/PhysRevA.91.062308](https://doi.org/10.1103/PhysRevA.91.062308), [arXiv:1503.05028](https://arxiv.org/abs/1503.05028).

[^21]: X. Li and C. Wang, “Simulating Markovian Open Quantum Systems Using Higher-Order Series Expansion,” in *50th International Colloquium on Automata, Languages, and Programming (ICALP 2023)*, LIPIcs **261**, Article 87 (2023), DOI: [10.4230/LIPIcs.ICALP.2023.87](https://doi.org/10.4230/LIPIcs.ICALP.2023.87), [arXiv:2212.02051](https://arxiv.org/abs/2212.02051).

[^22]: X. Wang, S. Zhou, X. Wang, Y.-C. Zheng, S. Zhang, and T. Li, “Lindbladian Simulation with Commutator Bounds,” arXiv preprint, version 2 (2026), [arXiv:2603.28602](https://arxiv.org/abs/2603.28602).

[^23]: IBM Quantum, “XXPlusYYGate,” *Qiskit API documentation*, accessed 9 September 2026, [documentation](https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.circuit.library.XXPlusYYGate). For the software context, see A. Javadi-Abhari *et al.*, “Quantum Computing with Qiskit,” arXiv preprint (2024), [arXiv:2405.08810](https://arxiv.org/abs/2405.08810).

[^24]: L. C. G. Govia *et al.*, “A Randomized Benchmarking Suite for Mid-Circuit Measurements,” *New Journal of Physics* **25**, 123016 (2023), DOI: [10.1088/1367-2630/ad0e19](https://doi.org/10.1088/1367-2630/ad0e19), [arXiv:2207.04836](https://arxiv.org/abs/2207.04836).

[^25]: A. Shirizly, L. C. G. Govia, and D. C. McKay, “Characterizing Errors on Qubit Operations via Iterative Midcircuit Measurement Benchmarking,” *Physical Review A* **111**, 012611 (2025), DOI: [10.1103/PhysRevA.111.012611](https://doi.org/10.1103/PhysRevA.111.012611), [arXiv:2408.07677](https://arxiv.org/abs/2408.07677).

[^26]: J. M. Koh, D. E. Koh, and J. Thompson, “Readout Error Mitigation for Mid-Circuit Measurements and Feedforward,” *PRX Quantum* **7**, 010317 (2026), DOI: [10.1103/cj89-4h5t](https://doi.org/10.1103/cj89-4h5t), [arXiv:2406.07611](https://arxiv.org/abs/2406.07611).

[^27]: IBM Quantum, “Hardware Considerations and Limitations for Classical Feedforward and Control Flow,” accessed 9 September 2026, [documentation](https://docs.quantum.ibm.com/guides/dynamic-circuits-considerations); IBM Quantum, “Execute Dynamic Circuits,” accessed 9 September 2026, [documentation](https://quantum.cloud.ibm.com/docs/en/guides/execute-dynamic-circuits).

[^28]: IBM Quantum, “Utility-Scale Dynamic Circuits Now Available for All Users,” IBM Quantum blog (2025), [article](https://www.ibm.com/quantum/blog/utility-scale-dynamic-circuits).

[^29]: Qiskit Aer Development Team, “AerSimulator,” Qiskit Aer 0.17.1 documentation, accessed 9 September 2026, [documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html).

[^30]: T. Prosen, “Third Quantization: A General Method to Solve Master Equations for Quadratic Open Fermi Systems,” *New Journal of Physics* **10**, 043026 (2008), DOI: [10.1088/1367-2630/10/4/043026](https://doi.org/10.1088/1367-2630/10/4/043026), [arXiv:0801.1257](https://arxiv.org/abs/0801.1257).

[^31]: D. Karevski and T. Platini, “Quantum Nonequilibrium Steady States Induced by Repeated Interactions,” *Physical Review Letters* **102**, 207207 (2009), DOI: [10.1103/PhysRevLett.102.207207](https://doi.org/10.1103/PhysRevLett.102.207207), [arXiv:0904.3527](https://arxiv.org/abs/0904.3527).

[^32]: D. Karevski, V. Popkov, and G. M. Schütz, “Exact Matrix Product Solution for the Boundary-Driven Lindblad XXZ Chain,” *Physical Review Letters* **110**, 047201 (2013), DOI: [10.1103/PhysRevLett.110.047201](https://doi.org/10.1103/PhysRevLett.110.047201), [arXiv:1211.7010](https://arxiv.org/abs/1211.7010).

[^33]: P. Mohammadipour and X. Li, “Reducing Circuit Depth in Lindblad Simulations via Step-Size Extrapolation,” *Physical Review A* **112**, 062206 (2025), DOI: [10.1103/dzpc-m1p4](https://doi.org/10.1103/dzpc-m1p4), [arXiv:2507.22341](https://arxiv.org/abs/2507.22341).

[^34]: S. Tornow, N. Kanazawa, W. E. Shanks, and D. J. Egger, “Digital Quantum Simulation of the Spin-Boson Model under Markovian Open-System Dynamics,” *Quantum Science and Technology* **7**, 045018 (2022), DOI: [10.1088/2058-9565/ac8eaf](https://doi.org/10.1088/2058-9565/ac8eaf), [arXiv:2012.11598](https://arxiv.org/abs/2012.11598).

[^35]: G. Di Bartolomeo, M. Vischi, T. Feri, A. Bassi, and S. Donadi, “Efficient Quantum Algorithm to Simulate Open Systems through a Single Environmental Qubit,” *Physical Review Research* **6**, 043321 (2024), DOI: [10.1103/PhysRevResearch.6.043321](https://doi.org/10.1103/PhysRevResearch.6.043321), [arXiv:2311.10009](https://arxiv.org/abs/2311.10009).

[^36]: K. Garg, Z. Ahmed, S. Mitra, and S. Chakraborty, “Simulating Quantum Collision Models with Hamiltonian Simulations Using Early Fault-Tolerant Quantum Computers,” *Physical Review A* **112**, 022425 (2025), DOI: [10.1103/3trk-smbh](https://doi.org/10.1103/3trk-smbh), [arXiv:2504.21564](https://arxiv.org/abs/2504.21564).

[^37]: E. Borras and M. Marvian, “Quantum Simulation Algorithms Based on Quantum Trajectories,” *Quantum* **10**, 2063 (2026), DOI: [10.22331/q-2026-04-13-2063](https://doi.org/10.22331/q-2026-04-13-2063), [arXiv:2509.10425](https://arxiv.org/abs/2509.10425).

[^38]: H. Kamakari, S.-N. Sun, M. Motta, and A. J. Minnich, “Digital Quantum Simulation of Open Quantum Systems Using Quantum Imaginary-Time Evolution,” *PRX Quantum* **3**, 010320 (2022), DOI: [10.1103/PRXQuantum.3.010320](https://doi.org/10.1103/PRXQuantum.3.010320), [arXiv:2104.07823](https://arxiv.org/abs/2104.07823).

[^39]: H. Chen, N. Gomes, S. Niu, and W. A. de Jong, “Adaptive Variational Simulation for Open Quantum Systems,” *Quantum* **8**, 1252 (2024), DOI: [10.22331/q-2024-02-13-1252](https://doi.org/10.22331/q-2024-02-13-1252), [arXiv:2305.06915](https://arxiv.org/abs/2305.06915).

[^40]: H. Chen, B. Li, J. Lu, and L. Ying, “A Randomized Method for Simulating Lindblad Equations and Thermal State Preparation,” *Quantum* **9**, 1917 (2025), DOI: [10.22331/q-2025-11-20-1917](https://doi.org/10.22331/q-2025-11-20-1917), [arXiv:2407.06594](https://arxiv.org/abs/2407.06594).

[^41]: E. Borras and M. Marvian, “Quantum Algorithm to Simulate Lindblad Master Equations,” *Physical Review Research* **7**, 023076 (2025), DOI: [10.1103/PhysRevResearch.7.023076](https://doi.org/10.1103/PhysRevResearch.7.023076).

[^42]: H.-Y. Liu, T.-P. Sun, Y.-C. Wu, and G.-P. Guo, “Simulation of Open Quantum Systems on Universal Quantum Computers,” *Quantum* **9**, 1765 (2025), DOI: [10.22331/q-2025-06-05-1765](https://doi.org/10.22331/q-2025-06-05-1765), [arXiv:2405.20712](https://arxiv.org/abs/2405.20712).

[^43]: J. Kato, K. Wada, K. Ito, and N. Yamamoto, “Exponentially Accurate Open Quantum Simulation via Randomized Dissipation with Minimal Ancilla,” *PRX Quantum* **7**, 033046 (2026), DOI: [10.1103/91yq-swlv](https://doi.org/10.1103/91yq-swlv), [arXiv:2412.19453](https://arxiv.org/abs/2412.19453).

[^44]: X. Dan, S. Shivpuje, Y. Wang, D. G. A. Cabral, B. C. Allen, P. Khazaei, A. V. Soudackov, Z. Hu, N. Lyu, E. Geva, S. Kais, and V. S. Batista, “QFlux: An Open-Source Toolkit for Quantum Dynamics Simulations on Quantum Computers. Part IV—Dilation Method for Open Quantum Systems,” ChemRxiv preprint (2026), DOI: [10.26434/chemrxiv.10001768/v1](https://doi.org/10.26434/chemrxiv.10001768/v1), [project documentation](https://qflux.batistalab.com/qflux/Open_Systems/spinchainOpen/).

[^45]: S. M. Pillay, I. J. David, I. Sinayskiy, and F. Petruccione, “Optimising Trotter-Suzuki Simulations of Markovian Open Quantum Systems via Classical Search,” *Quantum Information Processing* **25**, 267 (2026), DOI: [10.1007/s11128-026-05267-1](https://doi.org/10.1007/s11128-026-05267-1), [arXiv:2607.27060](https://arxiv.org/abs/2607.27060).

[^46]: X. Li, Y. Li, and Y. Yan, “Signatures of Environment-Induced Quantum Synchronization Transitions via Two-Body Dissipator Engineering,” *Chinese Physics Letters* **43**, 020302 (2026), DOI: [10.1088/0256-307X/43/2/020302](https://doi.org/10.1088/0256-307X/43/2/020302), [arXiv:2506.07580](https://arxiv.org/abs/2506.07580).
