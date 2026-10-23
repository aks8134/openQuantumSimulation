# Model3 circuit construction: a mathematical derivation

This document derives the randomized Lie–Trotter circuit. It connects the physical Lindblad equation to the Hamiltonian gates, boundary ancilla interactions, resets, trajectory averaging, and final observable measurements. 

The derivation describes the ideal logical circuit for **Model3, corresponding to Section 4.9, Experiment B: nonuniform stationary populations from unequal reservoirs**, in the modified [writeup.pdf](../writeup.pdf#page=11). The saved cosine study is a longer-chain, different-rate instance of the same model family; its parameters are distinguished explicitly below.

## 1. Model3 and Section 4.9 of the modified writeup

We follow Eqs. (1)–(9), (34), and (55)–(72) of [writeup.pdf](../writeup.pdf). The primary reservoir parameters are the preferred excitation populations $p_L,p_R$ and total boundary rates $\Gamma_L,\Gamma_R$, as introduced in Section 4.7 and used in Section 4.9. The physical timestep is $\Delta t$, jump operators are $V_\alpha$, and selection probabilities in the randomized circuit are $w_\alpha$.

The computational convention is

$$
Z|0\rangle=+|0\rangle,\qquad Z|1\rangle=-|1\rangle,
\qquad
\sigma^-=|0\rangle\langle1|=\frac{X+iY}{2},
\qquad
\sigma^+=|1\rangle\langle0|=\frac{X-iY}{2}.
$$

There are $n$ system sites, numbered $\ell=1,\ldots,n$, with occupation operators $n_\ell=(I-Z_\ell)/2$. The implementation uses zero-based indices. Tensor products involving a bath are written **ancilla first**, and $\hbar=1$.

### Hamiltonian and four boundary jumps

The Hamiltonian is the $\Delta=0$ specialization of Eq. (34):

$$
H=H_{\mathrm{XX}}=\frac J2\sum_{\ell=1}^{n-1}
(X_\ell X_{\ell+1}+Y_\ell Y_{\ell+1}).
$$

Using Eq. (55), with $s_L=1$ and $s_R=n$, the boundary operators are

$$
\begin{aligned}
V_{L,+}&=\sqrt{\Gamma_Lp_L}\,\sigma_1^+,
&V_{L,-}&=\sqrt{\Gamma_L(1-p_L)}\,\sigma_1^-,\\
V_{R,+}&=\sqrt{\Gamma_Rp_R}\,\sigma_n^+,
&V_{R,-}&=\sqrt{\Gamma_R(1-p_R)}\,\sigma_n^-.
\end{aligned}
$$

Write the master equation explicitly, as in Eq. (3):

$$
\frac{d\rho}{dt}=\mathcal L(\rho)
=-i[H,\rho]
+\sum_{\alpha\in\{L+,L-,R+,R-\}}
\left(V_\alpha\rho V_\alpha^\dagger
-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right),
\qquad \{A,B\}=AB+BA.
$$

All dissipation is at the endpoints. Section 4.9's model is not the quantum-detailed-balance thermalization problem called “Model problem III” in Section 5 of the PDF; repository labels and PDF model-problem numbering differ.

### Exact benchmark versus the saved cosine extension

Section 4.9 retains the Hamiltonian, size, and total rates from Eq. (64), changing the reservoirs to those in Eq. (66):

$$
n=3,\qquad J=1,\qquad \Delta=0,\qquad
\Gamma_L=\Gamma_R=2,\qquad p_L=0.9,\quad p_R=0.1.
$$

This gives precisely Eqs. (67)–(68):

$$
V_{L,+}=\sqrt{1.8}\,\sigma_1^+,
\quad V_{L,-}=\sqrt{0.2}\,\sigma_1^-,
\quad V_{R,+}=\sqrt{0.2}\,\sigma_3^+,
\quad V_{R,-}=\sqrt{1.8}\,\sigma_3^-.
$$

| Parameter | Section 4.9 / Model3 Experiment1 benchmark | Saved 72-step cosine extension |
|---|---:|---:|
| System sites $n$ | 3 | 10 |
| Exchange coupling $J$ | 1 | 1 |
| $ZZ$ interaction $\Delta$ | 0 | 0 |
| $\Gamma_L=\Gamma_R=\Gamma$ | 2 | 1 |
| $(p_L,p_R)$ | $(0.9,0.1)$ | $(0.9,0.1)$ |

The existing runner uses a common boundary-rate parameter, `gamma`, so $\Gamma_L=\Gamma_R=\Gamma=\texttt{gamma}$. The unequal-rate formulas above state the general physical model; the current command-line interface supplies the common-rate case. The 72-step result does not establish a 72-step solution for Section 4.9's exact three-site, $\Gamma=2$ benchmark.

For comparison with the earlier bias notation in Eqs. (36)–(37), the special case $p_L+p_R=1$ has $p_L=(1+\mu)/2$, $p_R=(1-\mu)/2$. Here $\mu=0.8$. It is not needed as a primary parameter in Section 4.9. With the stated basis convention, the isolated bath targets are $\langle Z_1\rangle_*=1-2p_L$ and $\langle Z_n\rangle_*=1-2p_R$, consistent with Eq. (56) and the corrected text in the modified writeup.

For the gate derivation, write $V_\alpha=\sqrt{\lambda_\alpha}A_\alpha$:

| Branch $\alpha$ | $A_\alpha$ | $\lambda_\alpha$ |
|---|---|---|
| $L,+$ | $\sigma_1^+$ | $\Gamma_Lp_L$ |
| $L,-$ | $\sigma_1^-$ | $\Gamma_L(1-p_L)$ |
| $R,+$ | $\sigma_n^+$ | $\Gamma_Rp_R$ |
| $R,-$ | $\sigma_n^-$ | $\Gamma_R(1-p_R)$ |

The quantities $J$, $\Gamma_b$, and $\lambda_\alpha$ have units of inverse time. The timestep has units of time, and all gate angles are dimensionless. As in Sections 4.7–4.9, we retain both excitation current $I_\ell$ and magnetization current $j_\ell$, with $I_\ell=-j_\ell/2$.

### Relation to the writeup's joint-ancilla dilation

The parent construction of Eqs. (40)–(42) is

$$
\widetilde H_{\Delta t}^{\mathrm{bdry}}
=\sqrt{\Delta t}\,|0\rangle\langle0|_a\otimes H+B,
\qquad
B=\sum_\alpha\left(
|\alpha\rangle\langle0|_a\otimes V_\alpha
+|0\rangle\langle\alpha|_a\otimes V_\alpha^\dagger\right),
$$

$$
K_{\Delta t}^{\mathrm{bdry}}=\sqrt{\Delta t}\,\widetilde H_{\Delta t}^{\mathrm{bdry}},
\qquad U_{\Delta t}^{\mathrm{bdry}}=e^{-iK_{\Delta t}^{\mathrm{bdry}}},
\qquad R=|0\rangle\langle0|_a\otimes\rho.
$$

The channel is $\Phi_{\Delta t}^{\mathrm{joint}}(\rho)=\operatorname{Tr}_a(U_{\Delta t}^{\mathrm{bdry}}R U_{\Delta t}^{\mathrm{bdry}\dagger})$. To identify its generator directly, use

$$
e^{-iK}Re^{iK}=R-i[K,R]-\frac12[K,[K,R]]+\cdots.
$$

The needed partial traces are

$$
\begin{aligned}
\operatorname{Tr}_a[B,R]&=0,\\
\operatorname{Tr}_a(BRB)&=\sum_\alpha V_\alpha\rho V_\alpha^\dagger,\\
\operatorname{Tr}_a(B^2R)&=\sum_\alpha V_\alpha^\dagger V_\alpha\rho,\\
\operatorname{Tr}_a(RB^2)&=\sum_\alpha\rho V_\alpha^\dagger V_\alpha.
\end{aligned}
$$

Substitution gives

$$
\Phi_{\Delta t}^{\mathrm{joint}}(\rho)
=\rho+\Delta t\left[-i[H,\rho]
+\sum_\alpha\left(V_\alpha\rho V_\alpha^\dagger
-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right)\right]
+O((\Delta t)^2).
$$

Terms with an odd number of the off-diagonal couplings $B$ have zero ancilla trace for this initial bath state; this removes the possible half-integer powers through order $(\Delta t)^{3/2}$ in the reduced channel.

The joint ancilla has five active states. The implemented circuit is a **randomized, reset-and-reuse variant**: it chooses one probability-compensated boundary block per step. It has the same first-order physical generator, but its finite-step channel differs from the joint unitary and from Eq. (44)'s four-block symmetric formula. The following derivation describes the implemented variant.

## 2. Deriving the Hamiltonian exchange gate

For a single bond, define

$$
H_{\ell,\ell+1}=\frac J2(X_\ell X_{\ell+1}+Y_\ell Y_{\ell+1}).
$$

In the ordered basis $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$,

$$
XX+YY=
\begin{pmatrix}
0&0&0&0\\
0&0&2&0\\
0&2&0&0\\
0&0&0&0
\end{pmatrix}.
$$


Qiskit's zero-phase exchange gate uses the convention

$$
G(\varphi)\equiv\operatorname{XXPlusYYGate}(\varphi,0)
=\exp\!\left[-\frac{i\varphi}{4}(XX+YY)\right].
$$

Matching the exponents gives

$$
U_{\ell,\ell+1}(\Delta t) = e^{-iH_{\ell,\ell+1}\Delta t}=G(2J\Delta t),
\qquad
\varphi_{\mathrm{bond}}=2J\Delta t.
$$

This factor follows from the [documented Qiskit gate definition](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.XXPlusYYGate).

### Two disjoint bond layers

Split the Hamiltonian into

$$
H_A=\sum_{\ell\in\{1,3,5,\ldots\}}H_{\ell,\ell+1},
\qquad
H_B=\sum_{\ell\in\{2,4,6,\ldots\}}H_{\ell,\ell+1}.
$$

Terms within each sum act on disjoint pairs, so their exponentials factor exactly. Applying layer $A$ first and then $B$ produces

$$
U_{\mathrm T}(\Delta t)=e^{-i\Delta t H_B}e^{-i\Delta t H_A}
=e^{-i\Delta t H}+O((\Delta t)^2).
$$

## 3. A system–bath unitary for one boundary

Introduce a bath qubit $a$ beside boundary system qubit $S$. Define

$$
W(\theta)
=\exp\!\left[-\frac{i\theta}{2}(X_a\otimes X_S+Y_a\otimes Y_S)\right]
=G(2\theta).
$$

## 4. Loss: the Lindbladian corresponding to $\sigma^-$

Consider a single spin with $V_-=\sqrt{\lambda_-}\,\sigma^-$ and density matrix

$$
\rho=\begin{pmatrix}\rho_{00}&\rho_{01}\\\rho_{10}&\rho_{11}\end{pmatrix}.
$$

The two ingredients in the loss term are

$$
\sigma^-\rho\sigma^+
=\begin{pmatrix}\rho_{11}&0\\0&0\end{pmatrix},
\qquad
\sigma^+\sigma^-=|1\rangle\langle1|
=\begin{pmatrix}0&0\\0&1\end{pmatrix}.
$$

Therefore

$$
\{\sigma^+\sigma^-,\rho\}
=\begin{pmatrix}0&\rho_{01}\\\rho_{10}&2\rho_{11}\end{pmatrix},
$$

and the loss Lindbladian is

$$
\boxed{
\mathcal L_-(\rho)
=\lambda_-\left(\sigma^-\rho\sigma^+
-\frac12\{\sigma^+\sigma^-,\rho\}\right)
=\lambda_-\begin{pmatrix}
\rho_{11}&-\rho_{01}/2\\-\rho_{10}/2&-\rho_{11}
\end{pmatrix}.}
$$

For loss alone, $d\rho/dt=\mathcal L_-(\rho)$ implies

$$
\frac{d\rho_{11}}{dt}=-\lambda_-\rho_{11},
\qquad
\frac{d\rho_{00}}{dt}=\lambda_-\rho_{11},
\qquad
\frac{d\rho_{01}}{dt}=-\frac{\lambda_-}{2}\rho_{01}.
$$

Thus occupation decays as $e^{-\lambda_-t}$ and coherence decays as $e^{-\lambda_-t/2}$.

For the circuit, initialize the bath in $|0\rangle_a$ and apply the exchange $W(\theta)$ from Section 3. Direct multiplication followed by the bath trace gives

$$
\begin{aligned}
\mathcal E_{-,\theta}(\rho)
&=\operatorname{Tr}_a\!\left[W(\theta)
(|0\rangle\langle0|_a\otimes\rho)W(\theta)^\dagger\right]\\
&=\begin{pmatrix}
\rho_{00}+\sin^2\theta\,\rho_{11}&\cos\theta\,\rho_{01}\\
\cos\theta\,\rho_{10}&\cos^2\theta\,\rho_{11}
\end{pmatrix}.
\end{aligned}
$$

Expanding $\sin^2\theta=\theta^2+O(\theta^4)$ and $\cos\theta=1-\theta^2/2+O(\theta^4)$ reproduces the matrix in $\mathcal L_-$ with coefficient $\theta^2/\lambda_-$. Section 6 determines how to choose $\theta$ for a physical timestep and random branch selection.

## 5. Pumping: the Lindbladian corresponding to $\sigma^+$

Now let $V_+=\sqrt{\lambda_+}\,\sigma^+$. Using the same density matrix,

$$
\sigma^+\rho\sigma^-
=\begin{pmatrix}0&0\\0&\rho_{00}\end{pmatrix},
\qquad
\sigma^-\sigma^+=|0\rangle\langle0|
=\begin{pmatrix}1&0\\0&0\end{pmatrix},
$$

$$
\{\sigma^-\sigma^+,\rho\}
=\begin{pmatrix}2\rho_{00}&\rho_{01}\\\rho_{10}&0\end{pmatrix}.
$$

Consequently,

$$
\boxed{
\mathcal L_+(\rho)
=\lambda_+\left(\sigma^+\rho\sigma^-
-\frac12\{\sigma^-\sigma^+,\rho\}\right)
=\lambda_+\begin{pmatrix}
-\rho_{00}&-\rho_{01}/2\\-\rho_{10}/2&\rho_{00}
\end{pmatrix}.}
$$

For pumping alone,

$$
\frac{d\rho_{11}}{dt}=\lambda_+\rho_{00},
\qquad
\frac{d\rho_{00}}{dt}=-\lambda_+\rho_{00},
\qquad
\frac{d\rho_{01}}{dt}=-\frac{\lambda_+}{2}\rho_{01}.
$$

For a normalized state, $\rho_{11}(t)=1-[1-\rho_{11}(0)]e^{-\lambda_+t}$. The local occupation tends to one.

In the implemented circuit, prepare the bath in $|1\rangle_a$ using X, then apply the same exchange gate:

$$
\begin{aligned}
\mathcal E_{+,\theta}(\rho)
&=\operatorname{Tr}_a\!\left[W(\theta)
(|1\rangle\langle1|_a\otimes\rho)W(\theta)^\dagger\right]\\
&=\begin{pmatrix}
\cos^2\theta\,\rho_{00}&\cos\theta\,\rho_{01}\\
\cos\theta\,\rho_{10}&\rho_{11}+\sin^2\theta\,\rho_{00}
\end{pmatrix}.
\end{aligned}
$$

Its small-angle expansion gives the pumping Lindbladian above. For the full chain, the raising and lowering operators act on the chosen endpoint and as identity on all other sites.

### Combining pump and loss at a boundary

For reservoir $b$, set $\lambda_+=\Gamma_bp_b$ and $\lambda_-=\Gamma_b(1-p_b)$. Adding the two local generators gives

$$
\frac{d\rho_{11}}{dt}
=\Gamma_bp_b\rho_{00}-\Gamma_b(1-p_b)\rho_{11}
=\Gamma_b(p_b-\rho_{11}),
\qquad
\frac{d\rho_{01}}{dt}=-\frac{\Gamma_b}{2}\rho_{01}.
$$

This derives Eq. (56): an isolated boundary spin tends to occupation $p_b$ and magnetization $1-2p_b$. For $p_L=0.9$ and $p_R=0.1$, the reservoir preferences differ; this is the driving mechanism of Section 4.9.

### Connection to the writeup's initially empty ancilla

The writeup always initializes the ancilla in $|0\rangle_a$. For a selected branch with probability $w_\alpha>0$, its probability-compensated coupling operator can be written

$$
F_\alpha=\frac{1}{\sqrt{w_\alpha}}
\left(|1\rangle\langle0|_a\otimes V_\alpha
+|0\rangle\langle1|_a\otimes V_\alpha^\dagger\right).
$$

With the Pauli conventions of the writeup,

$$
F_- =\frac{\sqrt{\lambda_-}}{2\sqrt{w_-}}
(X_a\otimes X_S+Y_a\otimes Y_S),
\qquad
F_+ =\frac{\sqrt{\lambda_+}}{2\sqrt{w_+}}
(X_a\otimes X_S-Y_a\otimes Y_S).
$$

Thus loss is a direct exchange with an empty ancilla. The empty-ancilla pumping unitary instead satisfies

$$
e^{-i\sqrt{\Delta t}F_+}
=X_a W(\theta_{+,\Delta t})X_a,
\qquad
\theta_{+,\Delta t}=\sqrt{\lambda_+\Delta t/w_+}.
$$

The first X prepares an occupied bath. The final X acts only on the discarded bath and can be omitted before reset without affecting the reduced system channel. This derives the equivalence between the writeup's reset-zero pumping dilation and the implemented X-then-exchange block.

## 6. Deriving the square-root timestep and probability compensation

Choose one branch $\alpha$ with probability $w_\alpha$, where $\sum_\alpha w_\alpha=1$. Every nonzero-rate branch must have positive selection probability. Use the initially empty ancilla form from Section 5, and define

$$
Q_\alpha=|1\rangle\langle0|_a\otimes A_\alpha
+|0\rangle\langle1|_a\otimes A_\alpha^\dagger,
\qquad R=|0\rangle\langle0|_a\otimes\rho.
$$

For $e^{-i\theta Q_\alpha}$, expand the joint density operator:

$$
e^{-i\theta Q_\alpha}R e^{i\theta Q_\alpha}
=R-i\theta[Q_\alpha,R]
-\frac{\theta^2}{2}[Q_\alpha,[Q_\alpha,R]]+\cdots.
$$

The first-order commutator has zero bath trace. For the second-order term,

$$
\begin{aligned}
\operatorname{Tr}_a(Q_\alpha RQ_\alpha)&=A_\alpha\rho A_\alpha^\dagger,\\
\operatorname{Tr}_a(Q_\alpha^2R)&=A_\alpha^\dagger A_\alpha\rho,\\
\operatorname{Tr}_a(RQ_\alpha^2)&=\rho A_\alpha^\dagger A_\alpha.
\end{aligned}
$$

Odd powers also have vanishing bath trace, so

$$
\mathcal E_{\alpha,\theta}(\rho)
=\rho+\theta^2\left(A_\alpha\rho A_\alpha^\dagger
-\frac12\{A_\alpha^\dagger A_\alpha,\rho\}\right)+O(\theta^4).
$$

Since $V_\alpha=\sqrt{\lambda_\alpha}A_\alpha$, the desired contribution in the branch average is $\Delta t\lambda_\alpha$ times the expression in parentheses. Therefore require

$$
w_\alpha\theta_{\alpha,\Delta t}^2=\lambda_\alpha\Delta t,
\qquad
\boxed{\theta_{\alpha,\Delta t}=\sqrt{\frac{\lambda_\alpha\Delta t}{w_\alpha}}.}
$$

The selected channel obeys

$$
\mathcal E_{\alpha,\Delta t}(\rho)
=\rho+\frac{\Delta t}{w_\alpha}
\left(V_\alpha\rho V_\alpha^\dagger
-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right)+O((\Delta t)^2).
$$

Averaging gives the intended dissipative part of the physical generator:

$$
\sum_\alpha w_\alpha\mathcal E_{\alpha,\Delta t}(\rho)
=\rho+\Delta t\sum_\alpha
\left(V_\alpha\rho V_\alpha^\dagger
-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right)+O((\Delta t)^2).
$$

The square root appears because the first surviving contribution after tracing out the bath is quadratic in $\theta$. The factor $1/\sqrt{w_\alpha}$ compensates for applying the selected branch only a fraction $w_\alpha$ of the time. One branch is applied per step; the resulting reduced evolution includes both transferred and untransferred population.

## 7. Specialization to the implemented rate-weighted sampling

For equal total rates $\Gamma_L=\Gamma_R=\Gamma$, as in Section 4.9 and the saved cosine extension, the sum of the four rates is $2\Gamma$. The default policy chooses

$$
w_\alpha=\frac{\lambda_\alpha}{2\Gamma}.
$$

In branch order $(L,+),(L,-),(R,+),(R,-)$,

$$
\boldsymbol w=\frac12(p_L,1-p_L,p_R,1-p_R).
$$

For Section 4.9's $p_L=0.9$ and $p_R=0.1$, this is $(0.45,0.05,0.05,0.45)$. Every active branch satisfies $\lambda_\alpha/w_\alpha=2\Gamma$, hence

$$
\boxed{\theta_{\Delta t}=\sqrt{2\Gamma \Delta t},
\qquad
\varphi_{\mathrm{bath}}=2\theta_{\Delta t}=2\sqrt{2\Gamma \Delta t}.}
$$

All active branches therefore use the same exchange angle. The selected endpoint, initial bath bit, and selection frequencies encode the unequal reservoirs. A branch with zero physical rate has zero probability under this policy and is never selected; its angle need not be evaluated as a ratio of two zeros.

For example, with $J=\Gamma=1$ and $\Delta t=0.1$,

$$
\varphi_{\mathrm{bond}}=0.2,
\qquad
\theta_{\Delta t}=\sqrt{0.2}\simeq0.447214,
\qquad
\varphi_{\mathrm{bath}}\simeq0.894427.
$$

Changing to uniform branch sampling would retain the formula $\theta_{\alpha,\Delta t}=\sqrt{\lambda_\alpha \Delta t/w_\alpha}$ with $w_\alpha=1/4$; the four angles would generally differ.

## 8. Complete timestep, ancilla reset, and trajectory averaging

For a chosen branch $\alpha$, define its boundary site and initial bath occupation by

$$
s(\alpha)=\begin{cases}1,&\alpha=(L,\pm),\\n,&\alpha=(R,\pm),\end{cases}
\qquad
b_\alpha=\begin{cases}1,&\text{pump},\\0,&\text{loss}.\end{cases}
$$

First apply the two Hamiltonian layers. Then interact the selected endpoint with its bath. Tracing out that bath gives the system update

$$
\mathcal B_{\alpha,\Delta t}(\rho)
=\operatorname{Tr}_a\!\left[
W_{a,s(\alpha)}(\theta_{\alpha,\Delta t})
\left(|b_\alpha\rangle\langle b_\alpha|_a\otimes
U_{\mathrm T}(\Delta t)\rho U_{\mathrm T}(\Delta t)^\dagger\right)
W_{a,s(\alpha)}(\theta_{\alpha,\Delta t})^\dagger
\right].
$$

Equivalently, $\mathcal B_{\alpha,\Delta t}=\mathcal E_{\alpha,\Delta t}\circ\mathcal U_{\mathrm T,\Delta t}$, where $\mathcal U_{\mathrm T,\Delta t}(\rho)=U_{\mathrm T}(\Delta t)\rho U_{\mathrm T}(\Delta t)^\dagger$.

The actual ancilla reset acts on a joint state $\tau_{aS}$ as

$$
\mathcal R_a(\tau_{aS})
=|0\rangle\langle0|_a\otimes\operatorname{Tr}_a(\tau_{aS}).
$$

It discards the bath's previous state and correlations, supplying a fresh bath for the next interaction. No bath outcome is postselected. The implementation normally uses separate left and right bath qubits to keep both endpoint interactions local; only the selected one interacts during a given step.

The averaged timestep is

$$
\Phi_{\Delta t}=\sum_\alpha w_\alpha\mathcal B_{\alpha,\Delta t}.
$$

Combining the Hamiltonian and dissipator expansions yields

$$
\begin{aligned}
\Phi_{\Delta t}(\rho)
&=\rho+\Delta t\left(-i[H,\rho]+\sum_\alpha\left(V_\alpha\rho V_\alpha^\dagger-\frac12\{V_\alpha^\dagger V_\alpha,\rho\}\right)\right)+O((\Delta t)^2)\\
&=\rho+\Delta t\mathcal{L}(\rho)+O((\Delta t)^2)
=e^{\Delta t\mathcal{L}}(\rho)+O((\Delta t)^2).
\end{aligned}
$$

This establishes first-order consistency of the implemented randomized step with the target Lindblad equation, for fixed model parameters and branch probabilities.

### What a saved trajectory means

A saved trajectory fixes a branch sequence $\omega=(\alpha_0,\ldots,\alpha_{N_{\mathrm{steps}}-1})$. Its state obeys

$$
\rho_{k+1}^{(\omega)}
=\mathcal B_{\alpha_k,\Delta t_k}(\rho_k^{(\omega)}).
$$

Independent branch draws at each step imply

$$
\overline\rho_{k+1}=\Phi_{\Delta t_k}(\overline\rho_k),
\qquad
\overline\rho_{N_{\mathrm{steps}}}=
\Phi_{\Delta t_{N_{\mathrm{steps}}-1}}\circ\cdots\circ\Phi_{\Delta t_0}(\rho_0).
$$

Each $\rho_k^{(\omega)}$ already averages over unobserved bath outcomes. Thus the saved branch sequence is not a pure-state trajectory conditioned on every transfer event. On hardware, multiple shots of its compiled circuit repeat the same saved branch sequence, while preparation and final measurement outcomes fluctuate.

## 9. Why finite timesteps produce bias

The reduced-channel formulas in Sections 4 and 5 are exact for the implemented unitary and reset at any real angle. Their identification with Lindblad evolution is a small-$\Delta t$ approximation:

$$
q_{\alpha,\Delta t}=\sin^2\theta_{\alpha,\Delta t}
=\frac{\lambda_\alpha \Delta t}{w_\alpha}
-\frac13\left(\frac{\lambda_\alpha \Delta t}{w_\alpha}\right)^2
+O((\Delta t)^3).
$$

For comparison, exponentiating a loss-only or pump-only dissipator of rate $\lambda_\alpha/w_\alpha$ exactly for time $\Delta t$ would give

$$
q_{\mathrm{exp}}=1-e^{-\lambda_\alpha \Delta t/w_\alpha}.
$$

Both agree to first order, but not at finite timestep. Moreover, the implemented coherence multiplier is $\cos\theta$, which can be negative at coarse timesteps. It must not be replaced by $\sqrt{1-q}=|\cos\theta|$ when describing the actual circuit.

The finite-step effect can also be seen directly in the averaged boundary population. Let $\widetilde\nu_L$ be the left occupation expectation immediately after the Hamiltonian layers. With rate-weighted sampling,

$$
\begin{aligned}
\nu_L'
&=\widetilde\nu_L
+\frac{p_L}{2}\sin^2\theta_{\Delta t}(1-\widetilde\nu_L)
-\frac{1-p_L}{2}\sin^2\theta_{\Delta t}\,\widetilde\nu_L\\
&=\widetilde\nu_L+\frac{\sin^2\!\sqrt{2\Gamma \Delta t}}{2}
(p_L-\widetilde\nu_L).
\end{aligned}
$$

The other-endpoint branches leave this population unchanged. In the exact reservoir-only evolution, the corresponding update would instead be

$$
\nu_L(\Delta t)=\widetilde\nu_L+(1-e^{-\Gamma \Delta t})
(p_L-\widetilde\nu_L).
$$

This comparison concerns the dissipative stage alone. Together with the noncommuting Hamiltonian layers and the Hamiltonian–dissipator splitting, it explains why a valid coarse-step circuit can have a biased stationary state. Trace preservation and positivity do not by themselves establish accuracy relative to the physical Lindblad model.

## 10. Initial mixed-state preparation

For an intended local occupation $r_\ell$, start from $|0\rangle$ and apply

$$
R_y(2\arcsin\sqrt{r_\ell})|0\rangle
=\sqrt{1-r_\ell}|0\rangle+\sqrt{r_\ell}|1\rangle.
$$

An unconditioned Z measurement applies the dephasing channel

$$
\mathcal M_Z(\rho)=\sum_{b=0}^1|b\rangle\langle b|\rho|b\rangle\langle b|.
$$

Therefore the prepared local state, when the outcome is ignored, is

$$
\rho_\ell=(1-r_\ell)|0\rangle\langle0|+r_\ell|1\rangle\langle1|,
\qquad
\rho_0=\bigotimes_{\ell=1}^n\rho_\ell.
$$

For the half-mixed preparation, $r_\ell=1/2$ at every site. The code uses H followed by an unconditioned Z measurement, giving

$$
\rho_0=\left(\frac I2\right)^{\otimes n}.
$$

The preparation record is not used to select or discard shots. The system must retain its random prepared bit after measurement; resetting it immediately would replace the intended mixture by $|0\rangle\langle0|$.

## 11. Population and current observables of Sections 4.7–4.9

Use the magnetization $m_\ell$ and current operator $j_\ell$ of Eqs. (35) and (52):

$$
m_\ell=\langle Z_\ell\rangle,
\qquad
j_\ell=J(X_\ell Y_{\ell+1}-Y_\ell X_{\ell+1}).
$$

The excitation-current operator of Eq. (57) is

$$
I_\ell=\frac J2(Y_\ell X_{\ell+1}-X_\ell Y_{\ell+1})=-\frac12j_\ell.
$$

To derive the sign and normalization, use $[X,Z]=-2iY$ and $[Y,Z]=2iX$ on the right bond:

$$
\begin{aligned}
i[H_{\ell,\ell+1},Z_\ell]
&=\frac{iJ}{2}\left([X_\ell,Z_\ell]X_{\ell+1}
+[Y_\ell,Z_\ell]Y_{\ell+1}\right)\\
&=J(Y_\ell X_{\ell+1}-X_\ell Y_{\ell+1})=-j_\ell.
\end{aligned}
$$

The left bond supplies $+j_{\ell-1}$, so the interior continuity equation is

$$
\frac{dm_\ell}{dt}=\langle j_{\ell-1}\rangle-\langle j_\ell\rangle,
\qquad 2\leq\ell\leq n-1.
$$


The primary population continuity equations of Section 4.7, Eqs. (58)–(60), follow from $n_\ell=(I-Z_\ell)/2$ and the current relation above:

$$
\begin{aligned}
\frac{d\langle n_1\rangle}{dt}
&=\Gamma_L(p_L-\langle n_1\rangle)-\langle I_1\rangle,\\
\frac{d\langle n_\ell\rangle}{dt}
&=\langle I_{\ell-1}\rangle-\langle I_\ell\rangle,
\qquad 2\leq\ell\leq n-1,\\
\frac{d\langle n_n\rangle}{dt}
&=\langle I_{n-1}\rangle+\Gamma_R(p_R-\langle n_n\rangle).
\end{aligned}
$$

Equivalently, the boundary magnetizations satisfy

$$
\frac{dm_1}{dt}=-\langle j_1\rangle-\Gamma_L[m_1-(1-2p_L)],
\qquad
\frac{dm_n}{dt}=\langle j_{n-1}\rangle-\Gamma_R[m_n-(1-2p_R)].
$$

### Mapping existing population and excitation-current outputs

The saved simulator outputs can be converted to the writeup's observables without changing any dynamics:

$$
\langle n_\ell\rangle=\frac{1-m_\ell}{2},
\qquad
\langle I_\ell\rangle=\frac J2\left(\langle Y_\ell X_{\ell+1}\rangle
-\langle X_\ell Y_{\ell+1}\rangle\right)
=-\frac12\langle j_\ell\rangle.
$$

For the ten-site cosine extension, the saved steady excitation current $\langle I_\ell\rangle_*=0.32$ for $J=\Gamma=1$ corresponds to $\langle j_\ell\rangle_*=-0.64$. This is a sign and normalization conversion, not a change in the physical state.

### Section 4.9 stationary targets

At stationarity the two three-site bond-current expectations agree; denote their common value by $I_*$. For equal rates $\Gamma_L=\Gamma_R=\Gamma$, Eqs. (69)–(70) give

$$
I_* = \frac{2J^2\Gamma}{\Gamma^2+4J^2}(p_L-p_R),
\qquad
\langle n_1\rangle_*=p_L-\frac{I_*}{\Gamma},
\quad
\langle n_2\rangle_*=\frac{p_L+p_R}{2},
\quad
\langle n_3\rangle_*=p_R+\frac{I_*}{\Gamma}.
$$

For Section 4.9's $J=1$, $\Gamma=2$, and $(p_L,p_R)=(0.9,0.1)$, these give exactly Eqs. (71)–(72):

$$
(\langle n_1\rangle_*,\langle n_2\rangle_*,\langle n_3\rangle_*)=(0.7,0.5,0.3),
$$

$$
(\langle Z_1\rangle_*,\langle Z_2\rangle_*,\langle Z_3\rangle_*)=(-0.4,0,0.4),
\qquad
\langle I_1\rangle_*=\langle I_2\rangle_*=0.4,
\qquad
\langle j_1\rangle_*=\langle j_2\rangle_*=-0.8.
$$

These are the targets for the literal Section 4.9 benchmark. The saved ten-site, $\Gamma=1$ cosine extension instead has endpoint populations 0.58 and 0.42, flat interior populations 0.5, and excitation current 0.32. Neither target set should be substituted for the other. Increasing the homogeneous XX chain length does not by itself create a bulk population gradient, as noted at the end of Section 4.9.

### Three final measurement settings

All populations and nearest-neighbor currents are covered by

$$
Z\,Z\,Z\,Z\cdots,
\qquad
X\,Y\,X\,Y\cdots,
\qquad
Y\,X\,Y\,X\cdots.
$$

A basis rotation $B$ before computational-basis measurement measures $B^\dagger ZB$. The implemented native rotations are

$$
B_X=R_z(\pi/2)\,SX\,R_z(\pi/2),
\qquad
B_Y=R_z(\pi/2)\,SX,
$$

where $SX$ is the square-root-X gate and operations act right to left. They satisfy $B_X^\dagger ZB_X=X$ and $B_Y^\dagger ZB_Y=Y$.

Let $C_\ell^{XY}$ and $C_\ell^{YX}$ denote the measured neighboring-outcome correlations in the two alternating settings, not fixed Pauli labels on every bond. With the one-based indexing used here,

$$
\langle j_\ell\rangle
=J(-1)^{\ell-1}\left(C_\ell^{XY}-C_\ell^{YX}\right),
\qquad
\langle I_\ell\rangle=-\frac12\langle j_\ell\rangle.
$$

The alternating sign is necessary: on odd bonds the first setting measures $XY$, while on even bonds it measures $YX$. Final observables use only the final outcome register; initial preparation bits are ignored.

## 12. Substituting the variable cosine schedule

Let $N_{\mathrm{steps}}$ denote the number of preparation steps, distinguishing it from the system size $n$ and the dilation exponent $K_{\Delta t}^{\mathrm{bdry}}$. For $N_{\mathrm{steps}}>1$, the cosine scheduler supplies

$$
\Delta t_k=\Delta t_f+\frac{\Delta t_0-\Delta t_f}{2}
\left[1+\cos\left(\frac{\pi k}{N_{\mathrm{steps}}-1}\right)\right],
\qquad k=0,\ldots,N_{\mathrm{steps}}-1.
$$

For $N_{\mathrm{steps}}=1$, it uses $\Delta t_f$. The schedule changes the following angles at each step:

$$
\boxed{
\varphi_{\mathrm{bond},k}=2J\Delta t_k,
\qquad
\varphi_{\mathrm{bath},k}=2\sqrt{2\Gamma \Delta t_k}
}
$$

The common bath angle assumes the rate-weighted policy. The branch probabilities remain fixed because the physical rates remain fixed.

For the saved ten-site, $J=\Gamma=1$ cosine result (the extension distinguished in Section 1),

$$
N_{\mathrm{steps}}=72,\qquad
\Delta t_0=0.576267723009639,\qquad
\Delta t_f=0.05733068985582982.
$$

Every trajectory uses the same timestep list and its own random branch sequence. The deterministic mean channel selects the schedule offline; hardware measurement outcomes do not adjust it. Classical forward-validation steps are not appended to the preparation circuit.

## 13. Direct correspondence with the implementation

| Mathematical operation | Circuit implementation |
|---|---|
| Diagonal product preparation | Local H or RY, followed by unconditioned Z measurements |
| First bond layer, then second bond layer | `XXPlusYYGate(2*J*dt)` on alternating disjoint bonds |
| Choose one boundary branch | Saved `jump_choices[trajectory][step]` |
| Prepare an occupied bath for a pump | X on the selected reset bath |
| Boundary exchange | `XXPlusYYGate(2*sqrt(lambda*dt/w))` |
| Discard bath state and correlations | Reset selected bath |
| Measure populations and currents | Z and the two alternating X/Y readouts |

Source files:

- [Physical benchmark: modified writeup.pdf, Section 4.9, Eqs. (64), (66)–(72)](../writeup.pdf#page=11).
- [Reservoir and current conventions: Section 4.7, Eqs. (55)–(60)](../writeup.pdf#page=10).
- [Parent dilation: Eqs. (1)–(9), (34)–(44)](../writeup.pdf).
- [Logical preparation, timestep, ancilla handling, and readout](Experiment5/circuit_depth.py).
- [Shared mean-channel and sampled covariance implementation](../library/reservoir_scaling.py).
- [Cosine schedule selection](Experiment7/schedules.py).
- [Experiment1 normalization and three-site physical targets](Experiment1/Experiment1.md).
- [Variable-schedule results and circuit resources](../report.md).
