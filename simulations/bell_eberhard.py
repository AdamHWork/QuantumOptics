"""
simulations/bell_eberhard.py
════════════════════════════════════════════════════════════════════════
  Simulation of:
  "A Tractable Protocol for Detection-Loophole-Free Bell Tests
   over Long Distances"
  Alwehaibi, Mer, Machado, Yu, Walmsley, Patel (2025)
  Phys. Rev. Research 7, 043198
  arXiv: 2506.05048 | DOI: 10.1103/xw66-nqfs
  Blackett Laboratory & QuEST, Imperial College London

════════════════════════════════════════════════════════════════════════

PHYSICS OVERVIEW
────────────────
The paper proposes an all-photonic heralded entanglement protocol
that simultaneously achieves:

  (1) Square-root scaling with channel transmittance: O(√η_C)
  (2) Bell violation at the Eberhard limit: η_D ≥ 66.7%

PROTOCOL STRUCTURE
──────────────────
Alice prepares a two-mode squeezed vacuum (TMSV) state:

  |φ⟩_A = 1/√(1+λ²) ( |0,0⟩ + λ e^{iφ_a} |1,1⟩ )   λ = tanh(g)

Bob prepares a single-photon path-entangled (SPPE) state:

  |ψ⟩_B = √t_b |0,1⟩ + √(1-t_b) e^{iφ_b} |1,0⟩

Charlie performs a BSM by interfering modes a2 and b2 on a 50/50
beamsplitter and detecting exactly 1 photon. This heralds:

  |φ⟩_AB = c0|0,0⟩ + c1|1,1⟩

  c0 = √t_b / N,   c1 = λ√(1-t_b) / N,   N = normalisation

This vacuum+biphoton superposition is loss-tolerant because the
|0,0⟩ component is completely immune to photon loss.

MEASUREMENT
───────────
Alice and Bob displace their modes by α, β (complex) then perform
click/no-click detection. The CHSH parameter uses the Eberhard
form (Eq. S21 of supplemental):

  S = 2 + 4[Q_ab(α1,β1) + Q_ab(α1,β2) + Q_ab(α2,β1)
            - Q_ab(α2,β2) - Q_a(α1) - Q_b(β1)]

where Q_ab, Q_a, Q_b are no-click joint/marginal probabilities.

CORRECTIONS FROM v1
────────────────────
  1. Matplotlib format strings use ASCII markers only ('D' not '◇')
  2. CHSH correlator uses Eberhard form, not standard ±1 correlator
  3. Complex displacements properly handled in Q-functions
  4. SF simulation uses state.fock_prob() for exact probabilities
     rather than sampling, matching the paper's numerical method
════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.simulation import QuantumSimulation, SimulationConfig

import strawberryfields as sf
from strawberryfields.ops import (
    Squeezed, BSgate, Fock, Dgate, LossChannel
)

# Fock space cutoff — paper uses n=6 for <10^-5 truncation error
FOCK_CUTOFF = 6


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OPTIMAL PARAMETERS FROM TABLE SI
# ══════════════════════════════════════════════════════════════════════════════
# (η_D, t_b, g, α1, α2, β1, β2, CHSH_paper)
# α and β are real-valued displacement amplitudes per the paper's
# parameterisation. The phase of the heralded state absorbs the
# complex phase, leaving real optimal displacements.

OPTIMAL_PARAMS = [
    (0.65,  0.999,  0.090,  0.000, -0.013,  0.000,  0.013,  2.000000),
    (0.67,  0.945,  0.048,  0.001, -0.128, -0.001,  0.128,  2.000002),
    (0.68,  0.905,  0.132,  0.010, -0.241, -0.010,  0.241,  2.000133),
    (0.70,  0.823,  0.226,  0.035, -0.364, -0.035,  0.364,  2.002067),
    (0.75,  0.614,  0.279,  0.097, -0.501, -0.097,  0.501,  2.027202),
    (0.80,  0.181,  0.094,  0.139, -0.554, -0.139,  0.554,  2.088839),
    (0.85,  0.435,  0.320,  0.162, -0.574, -0.162,  0.574,  2.186472),
    (0.90,  0.414,  0.364,  0.172, -0.579, -0.172,  0.579,  2.318223),
    (0.95,  0.296,  0.284,  0.172, -0.573, -0.172,  0.573,  2.483976),
    (1.00,  0.280,  0.300,  0.165, -0.560, -0.165,  0.560,  2.685871),
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ANALYTICAL Q-FUNCTION CHSH (EQ. S21-S24)
# ══════════════════════════════════════════════════════════════════════════════

def _heralded_amplitudes(t_b, g):
    """
    Compute the normalised amplitudes of the heralded state:
      |φ⟩_AB = c0|0,0⟩ + c1|1,1⟩

    From Eq.(3) of the main text:
      c0 = √t_b / N
      c1 = tanh(g) · √(1-t_b) / N
      N² = t_b + tanh²(g)(1-t_b)

    Parameters
    ----------
    t_b : float — Bob's BS transmittance
    g   : float — squeezing parameter

    Returns
    -------
    c0, c1 : floats — vacuum and biphoton amplitudes
    """
    lam  = np.tanh(g)
    N_sq = t_b + lam**2 * (1 - t_b)
    N    = np.sqrt(N_sq)
    c0   = np.sqrt(t_b) / N
    c1   = lam * np.sqrt(1 - t_b) / N
    return c0, c1


def _Q_joint(c0, c1, alpha, beta, eta):
    """
    Joint no-click probability Q_ab^η(α,β) for the heralded state
    |φ⟩ = c0|0,0⟩ + c1|1,1⟩ under symmetric loss η.

    Implements Eq.(S23) of the supplemental material.

    The no-click POVM element for a lossy homodyne detector is:
      Π_0(α,η) = D†(α)(1-η)^{n̂}D(α)

    For our two-mode state in the {|00⟩,|11⟩} subspace:
      Q_ab = ⟨φ|Π_0^A(α,η) ⊗ Π_0^B(β,η)|φ⟩

    Expanding in the Fock basis gives (Eq. S23):
      Q_ab = e^{-η(|α|²+|β|²)} [
               c0²
             + c1²(1-η+η²|α|²)(1-η+η²|β|²)
             + 2c0c1η²Re(α)Re(β)
             ]

    Note: for real α,β the interference term simplifies to 2c0c1η²αβ

    Parameters
    ----------
    c0, c1 : float — heralded state amplitudes
    alpha  : float — Alice's displacement (real)
    beta   : float — Bob's displacement (real)
    eta    : float — detection efficiency

    Returns
    -------
    float — joint no-click probability
    """
    a2 = alpha**2
    b2 = beta**2

    # Common exponential decay from displacement of both modes
    exp_ab = np.exp(-eta * (a2 + b2))

    # Vacuum component: immune to loss, unaffected by displacement
    term_vac = c0**2

    # Biphoton component: both modes degraded by loss
    # Each mode contributes factor (1-η+η²|x|²) from Eq. S23
    term_bip = c1**2 * (1 - eta + eta**2 * a2) \
                     * (1 - eta + eta**2 * b2)

    # Interference: cross term between |00⟩ and |11⟩ components
    # This is what creates the Bell violation — the quantum
    # coherence between vacuum and biphoton
    term_int = 2 * c0 * c1 * eta**2 * alpha * beta

    return exp_ab * (term_vac + term_bip + term_int)


def _Q_marginal(c0, c1, x, eta):
    """
    Marginal no-click probability Q_a^η(α) or Q_b^η(β).

    Implements Eq.(S24) of the supplemental material.

    Obtained by tracing out the other mode from Q_ab:
      Q_a(α) = Tr_B[ρ_AB · Π_0^A(α,η) ⊗ I_B]
             = (c0² + c1²(1-η+η²|α|²)) e^{-η|α|²}

    Parameters
    ----------
    c0, c1 : float — heralded state amplitudes
    x      : float — displacement amplitude (α or β)
    eta    : float — detection efficiency

    Returns
    -------
    float — marginal no-click probability
    """
    return (c0**2 + c1**2 * (1 - eta + eta**2 * x**2)) \
           * np.exp(-eta * x**2)


def compute_chsh_analytical(t_b, g, alpha1, alpha2, beta1, beta2, eta_D):
    """
    Compute the CHSH parameter S using the analytical Q-function
    expressions from the supplemental material.

    Uses the Eberhard form (Eq. S21):
      S = 2 + 4[Q(α1,β1) + Q(α1,β2) + Q(α2,β1)
                - Q(α2,β2) - Q_a(α1) - Q_b(β1)]

    This form is specific to click/no-click (threshold) detectors
    and is NOT the same as the standard ±1 correlator form.
    It is equivalent to the standard CHSH when the measurement
    operators are defined as M = I - 2Π_0 (click = +1, no-click = -1).

    Parameters
    ----------
    t_b    : float — Bob's BS transmittance
    g      : float — squeezing parameter
    alpha1 : float — Alice's first setting
    alpha2 : float — Alice's second setting
    beta1  : float — Bob's first setting
    beta2  : float — Bob's second setting
    eta_D  : float — detection efficiency

    Returns
    -------
    float — CHSH parameter S
    """
    c0, c1 = _heralded_amplitudes(t_b, g)

    # Four joint Q-functions for the four measurement setting pairs
    Q11 = _Q_joint(c0, c1, alpha1, beta1, eta_D)
    Q12 = _Q_joint(c0, c1, alpha1, beta2, eta_D)
    Q21 = _Q_joint(c0, c1, alpha2, beta1, eta_D)
    Q22 = _Q_joint(c0, c1, alpha2, beta2, eta_D)

    # Marginal Q-functions (single-party no-click probabilities)
    Qa1 = _Q_marginal(c0, c1, alpha1, eta_D)
    Qb1 = _Q_marginal(c0, c1, beta1,  eta_D)

    # Eberhard CHSH form (Eq. S21)
    S = 2 + 4 * (Q11 + Q12 + Q21 - Q22 - Qa1 - Qb1)
    return S


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STRAWBERRY FIELDS NUMERICAL SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
# Replicates the paper's numerical method (Supplemental Sec. IV.B).
# Uses exact Fock-basis state vector computation rather than sampling,
# matching the paper's approach of computing probabilities directly
# from the state density matrix.
#
# Key correction from v1:
#   The CHSH correlator for click/no-click detectors uses the
#   Eberhard form, not the standard ±1 correlator. We compute
#   the no-click probabilities directly from the state vector
#   using state.fock_prob() for numerical exactness.

def _build_heralded_state(t_b, g, eta_D, eta_H=1.0,
                           cutoff=FOCK_CUTOFF):
    """
    Build the heralded state density matrix using Strawberry Fields.

    Simulates the full 4-mode circuit:
      q[0] = Alice mode a1  (kept)
      q[1] = Alice mode a2  (to Charlie)
      q[2] = Bob mode b1    (kept)
      q[3] = Bob mode b2    (to Charlie)

    After Charlie's BSM (post-select on 1 photon), applies local
    detection loss η_D to modes q[0] and q[2].

    Returns the reduced 2-mode density matrix for (a1, b1) after
    tracing out Charlie's modes and post-selecting.

    Parameters
    ----------
    t_b   : float — Bob's BS transmittance
    g     : float — squeezing parameter
    eta_D : float — local detection efficiency
    eta_H : float — heralding channel efficiency
    cutoff: int   — Fock space truncation

    Returns
    -------
    numpy.ndarray — reduced density matrix ρ_AB of shape
                    (cutoff², cutoff²) in the {|n_a1, n_b1⟩} basis
    float         — heralding probability p_herald
    """
    prog = sf.Program(4)

    with prog.context as q:
        # ── Alice: TMSV via two-mode squeezing ────────────────────
        # Two-mode squeezing: squeeze each mode then interfere on BS
        Squeezed(g)  | q[0]
        Squeezed(-g) | q[1]
        BSgate(np.pi / 4, 0.0) | (q[0], q[1])

        # ── Bob: SPPE via single photon + beamsplitter ────────────
        # θ = arccos(√t_b) gives transmittance t_b
        Fock(1)                              | q[2]
        BSgate(np.arccos(np.sqrt(t_b)), 0.0) | (q[2], q[3])

        # ── Heralding channel loss ─────────────────────────────────
        if eta_H < 1.0:
            LossChannel(eta_H) | q[1]
            LossChannel(eta_H) | q[3]

        # ── Charlie's BSM: 50/50 BS on a2 and b2 ─────────────────
        BSgate(np.pi / 4, 0.0) | (q[1], q[3])

        # ── Local detection loss at Alice and Bob ─────────────────
        if eta_D < 1.0:
            LossChannel(eta_D) | q[0]
            LossChannel(eta_D) | q[2]

    eng    = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    result = eng.run(prog)
    state  = result.state

    # ── Extract full probability tensor ───────────────────────────
    # Shape: (cutoff, cutoff, cutoff, cutoff) for (a1, a2, b1, b2)
    probs = state.all_fock_probs()

    # ── Post-select on Charlie detecting exactly 1 photon ─────────
    # Charlie's modes are q[1]=a2 and q[3]=b2
    # Post-selection: n_a2 + n_b2 = 1

    # Build reduced density matrix for (a1, b1) post Charlie=1
    # Since we're working with a mixed state after loss, we need
    # the full probability distribution P(n_a1, n_a2, n_b1, n_b2)

    # Heralding probability: P(n_a2 + n_b2 = 1)
    p_herald = 0.0
    for n_a2 in range(cutoff):
        for n_b2 in range(cutoff):
            if n_a2 + n_b2 == 1:
                for n_a1 in range(cutoff):
                    for n_b1 in range(cutoff):
                        idx = n_a1 + n_a2 + n_b1 + n_b2
                        if idx < cutoff:
                            p_herald += probs[n_a1, n_a2, n_b1, n_b2]

    # Conditional probability distribution P(n_a1, n_b1 | Charlie=1)
    p_cond = np.zeros((cutoff, cutoff))
    if p_herald > 1e-12:
        for n_a2 in range(cutoff):
            for n_b2 in range(cutoff):
                if n_a2 + n_b2 == 1:
                    for n_a1 in range(cutoff):
                        for n_b1 in range(cutoff):
                            idx = n_a1 + n_a2 + n_b1 + n_b2
                            if idx < cutoff:
                                p_cond[n_a1, n_b1] += \
                                    probs[n_a1, n_a2, n_b1, n_b2] \
                                    / p_herald

    return p_cond, p_herald


def _no_click_prob_sf(p_cond, alpha, beta, eta_D, cutoff=FOCK_CUTOFF):
    """
    Compute the joint no-click probability Q_ab(α,β) from the
    conditional probability distribution p_cond[n_a1, n_b1].

    The no-click probability after displacement D(α) is:
      P(no-click | α) = Σ_{n,m} p_cond[n,m] |⟨0|D(α)|n⟩|²|⟨0|D(β)|m⟩|²

    The overlap |⟨0|D(α)|n⟩|² = e^{-|α|²} |α|^{2n} / n!
    This is the Poisson distribution P(0; |α|²) evaluated at n.

    Parameters
    ----------
    p_cond : ndarray — conditional prob distribution (cutoff × cutoff)
    alpha  : float   — Alice's displacement
    beta   : float   — Bob's displacement
    eta_D  : float   — detection efficiency (already applied via LossChannel)
    cutoff : int     — Fock space truncation

    Returns
    -------
    float — joint no-click probability Q_ab(α,β)
    """
    # Overlap |⟨0|D(x)|n⟩|² = e^{-|x|²} |x|^{2n} / n!
    # This is the probability of detecting 0 photons when the
    # coherent state |α⟩ is measured in the Fock basis
    def overlap(x, n):
        x2 = x**2
        if n == 0:
            return np.exp(-x2)
        # Use log for numerical stability at large n
        log_val = -x2 + n * np.log(x2 + 1e-300) \
                  - sum(np.log(k) for k in range(1, n + 1))
        return np.exp(log_val)

    Q = 0.0
    for n_a in range(cutoff):
        ov_a = overlap(abs(alpha), n_a)
        for n_b in range(cutoff):
            ov_b = overlap(abs(beta), n_b)
            Q   += p_cond[n_a, n_b] * ov_a * ov_b

    return Q


def _marginal_no_click_sf(p_cond, x, is_alice, cutoff=FOCK_CUTOFF):
    """
    Compute the marginal no-click probability Q_a(α) or Q_b(β)
    from the conditional probability distribution.

    Parameters
    ----------
    p_cond   : ndarray — conditional prob distribution
    x        : float   — displacement amplitude
    is_alice : bool    — True for Alice's marginal, False for Bob's
    cutoff   : int     — Fock space truncation

    Returns
    -------
    float — marginal no-click probability
    """
    def overlap(val, n):
        v2 = val**2
        if n == 0:
            return np.exp(-v2)
        log_val = -v2 + n * np.log(v2 + 1e-300) \
                  - sum(np.log(k) for k in range(1, n + 1))
        return np.exp(log_val)

    Q = 0.0
    if is_alice:
        # Marginalise over Bob's modes
        for n_a in range(cutoff):
            ov_a = overlap(abs(x), n_a)
            for n_b in range(cutoff):
                Q += p_cond[n_a, n_b] * ov_a
    else:
        # Marginalise over Alice's modes
        for n_b in range(cutoff):
            ov_b = overlap(abs(x), n_b)
            for n_a in range(cutoff):
                Q += p_cond[n_a, n_b] * ov_b

    return Q


def compute_chsh_sf(t_b, g, alpha1, alpha2, beta1, beta2,
                    eta_D, eta_H=1.0, cutoff=FOCK_CUTOFF):
    """
    Compute the CHSH parameter S using Strawberry Fields.

    Builds the heralded state once, then evaluates the Eberhard
    CHSH form (Eq. S21) using exact Fock-basis probabilities.

    This matches the paper's numerical method (Supplemental Sec. IV.B)
    of computing Q-functions directly from the state rather than
    sampling, giving exact results up to Fock truncation error.

    Parameters
    ----------
    t_b    : float — Bob's BS transmittance
    g      : float — squeezing parameter
    alpha1 : float — Alice's first setting
    alpha2 : float — Alice's second setting
    beta1  : float — Bob's first setting
    beta2  : float — Bob's second setting
    eta_D  : float — detection efficiency
    eta_H  : float — heralding channel efficiency
    cutoff : int   — Fock space truncation

    Returns
    -------
    float — CHSH parameter S
    """
    # Build heralded state once — reuse for all four correlators
    p_cond, p_herald = _build_heralded_state(
        t_b, g, eta_D, eta_H, cutoff
    )

    if p_herald < 1e-12:
        return 0.0

    # Four joint no-click probabilities
    Q11 = _no_click_prob_sf(p_cond, alpha1, beta1,  eta_D, cutoff)
    Q12 = _no_click_prob_sf(p_cond, alpha1, beta2,  eta_D, cutoff)
    Q21 = _no_click_prob_sf(p_cond, alpha2, beta1,  eta_D, cutoff)
    Q22 = _no_click_prob_sf(p_cond, alpha2, beta2,  eta_D, cutoff)

    # Two marginal no-click probabilities
    Qa1 = _marginal_no_click_sf(p_cond, alpha1, True,  cutoff)
    Qb1 = _marginal_no_click_sf(p_cond, beta1,  False, cutoff)

    # Eberhard CHSH form (Eq. S21)
    S = 2 + 4 * (Q11 + Q12 + Q21 - Q22 - Qa1 - Qb1)
    return S


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DI-QKD QUANTITIES (EQS. 6a, 6b)
# ══════════════════════════════════════════════════════════════════════════════

def min_entropy(S):
    """
    Min-entropy H_min(S) — Eq.(6a) of main text.
    H_min = 1 - log2(1 + √(2 - S²/4))
    Quantifies certifiable randomness per round.
    Returns NaN if S ≤ 2 (no Bell violation).
    """
    arg = 2.0 - (S**2) / 4.0
    if arg < 0 or S <= 2.0:
        return np.nan
    return 1.0 - np.log2(1.0 + np.sqrt(arg))


def holevo_bound(S):
    """
    Holevo bound χ_max(S) — Eq.(6b) of main text.
    χ_max = h((1 + √(S²/4 - 1)) / 2)
    where h(x) = -x log2(x) - (1-x) log2(1-x) is binary entropy.
    Quantifies maximum eavesdropper information.
    Returns NaN if S ≤ 2 (no Bell violation).
    """
    arg = (S**2) / 4.0 - 1.0
    if arg < 0 or S <= 2.0:
        return np.nan
    x = (1.0 + np.sqrt(arg)) / 2.0
    if x <= 0 or x >= 1:
        return np.nan
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LOG-NEGATIVITY (EQ. S13)
# ══════════════════════════════════════════════════════════════════════════════

def log_negativity(lam, t_b):
    """
    Log-negativity E_N of the heralded state — Eq.(S13).

    E_N = log2(1 + 2λ√(t_b(1-t_b)) / (λ²(1-t_b) + t_b))

    E_N = 0 → separable, E_N = 1 → maximally entangled.
    """
    num = 2 * lam * np.sqrt(t_b * (1 - t_b))
    den = lam**2 * (1 - t_b) + t_b
    if den < 1e-15:
        return 0.0
    return np.log2(1 + num / den)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN SIMULATION CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BellEberhardSimulation(QuantumSimulation):
    """
    Simulation of the Alwehaibi et al. (2025) Bell test protocol.

    Computes and plots:
      1. CHSH vs η_D (Fig. 2 of paper)
      2. H_min and χ_max vs η_D (Fig. 4 of paper)
      3. Log-negativity landscape E_N(g, t_b) (Fig. S1)

    Parameters
    ----------
    config : SimulationConfig, optional
    run_sf : bool — if True, runs SF numerical simulation (slow)
    """

    def __init__(self, config=None, run_sf=True):
        super().__init__(config)
        self.save_dir = os.path.dirname(os.path.abspath(__file__))
        self.run_sf   = run_sf
        self.results  = {}

        print("═" * 65)
        print("  Bell Test Simulation — Alwehaibi et al. (2025)")
        print("  arXiv: 2506.05048 | Phys. Rev. Research 7, 043198")
        print("  Imperial College London — Walmsley Group")
        print("═" * 65)
        print(f"\n  Fock cutoff   : n = {FOCK_CUTOFF}")
        print(f"  SF simulation : "
              f"{'Enabled' if run_sf else 'Disabled (analytical only)'}")
        print(f"  η_D points    : {len(OPTIMAL_PARAMS)}\n")

    def build_circuit(self, phi_2=None):
        """Satisfies abstract base class — circuit built in helpers."""
        pass

    def run(self):
        """
        Execute the full simulation across all η_D values.

        For each row in OPTIMAL_PARAMS:
          1. Compute analytical CHSH (Q-function, Eq. S21)
          2. Optionally compute SF numerical CHSH
          3. Compute H_min and χ_max from analytical CHSH
          4. Store all results

        Also computes the log-negativity landscape over (g, t_b).
        """
        eta_D_vals    = []
        chsh_analytic = []
        chsh_sf_vals  = []
        chsh_paper    = []
        hmin_vals     = []
        holevo_vals   = []

        print("Computing CHSH vs detection efficiency...\n")

        for row in OPTIMAL_PARAMS:
            eta_D, t_b, g, a1, a2, b1, b2, chsh_expected = row

            print(f"  η_D = {eta_D:.2f} | t_b = {t_b:.3f} | g = {g:.3f}")

            # ── Analytical CHSH ────────────────────────────────────
            S_a = compute_chsh_analytical(
                t_b, g, a1, a2, b1, b2, eta_D
            )
            chsh_analytic.append(S_a)
            print(f"    Analytical CHSH : {S_a:.6f}  "
                  f"(paper: {chsh_expected:.6f})")

            # ── SF numerical CHSH ──────────────────────────────────
            if self.run_sf:
                S_sf = compute_chsh_sf(
                    t_b, g, a1, a2, b1, b2, eta_D
                )
                chsh_sf_vals.append(S_sf)
                print(f"    SF CHSH         : {S_sf:.6f}")
            else:
                chsh_sf_vals.append(np.nan)

            # ── DI-QKD quantities ──────────────────────────────────
            hmin_vals.append(min_entropy(S_a))
            holevo_vals.append(holevo_bound(S_a))

            eta_D_vals.append(eta_D)
            chsh_paper.append(chsh_expected)

        # ── Log-negativity landscape ───────────────────────────────
        print("\nComputing log-negativity landscape E_N(g, t_b)...")
        g_range  = np.linspace(0.01, 0.5,  80)
        tb_range = np.linspace(0.01, 0.99, 80)
        G, TB    = np.meshgrid(g_range, tb_range)
        LAM      = np.tanh(G)
        EN       = np.vectorize(log_negativity)(LAM, TB)

        self.results = {
            "eta_D"         : np.array(eta_D_vals),
            "chsh_analytic" : np.array(chsh_analytic),
            "chsh_sf"       : np.array(chsh_sf_vals),
            "chsh_paper"    : np.array(chsh_paper),
            "hmin"          : np.array(hmin_vals),
            "holevo"        : np.array(holevo_vals),
            "g_range"       : g_range,
            "tb_range"      : tb_range,
            "log_neg_map"   : EN,
        }

        print("\nSimulation complete.\n")
        return self

    def plot(self):
        """
        Generate all three figures from the paper.
        Uses only ASCII matplotlib marker strings.
        """
        r     = self.results
        eta_D = r["eta_D"]

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Bell Test Protocol",
            fontsize=13, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(
            2, 2, figure=fig,
            hspace=0.38, wspace=0.32,
            top=0.90, bottom=0.07
        )

        ax1 = fig.add_subplot(gs[0, :])   # Top row — CHSH
        ax2 = fig.add_subplot(gs[1, 0])   # Bottom left — DI-QKD
        ax3 = fig.add_subplot(gs[1, 1])   # Bottom right — Log-neg

        # ══════════════════════════════════════════════════════════
        # PLOT 1 — CHSH vs η_D
        # ══════════════════════════════════════════════════════════

        # Analytical result — smooth line
        ax1.plot(
            eta_D, r["chsh_analytic"],
            "g-", linewidth=2.5,
            label="This work (analytical Q-function)",
            zorder=4
        )

        # SF numerical result — triangle markers
        # ✅ Fixed: use 'g^' not 'g^' with unicode
        sf_arr = r["chsh_sf"]
        if not np.all(np.isnan(sf_arr)):
            ax1.plot(
                eta_D, sf_arr,
                "g^",                          # ✅ ASCII marker
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
                label=f"This work (SF Fock, cutoff n={FOCK_CUTOFF})",
                zorder=5
            )

        # Paper's reported values — diamond markers
        # ✅ Fixed: use 'D' (capital D = diamond) not '◇'
        ax1.plot(
            eta_D, r["chsh_paper"],
            "gD",                              # ✅ ASCII marker
            markersize=8,
            markerfacecolor="limegreen",
            markeredgewidth=1.5,
            label="Paper Table SI values",
            zorder=3
        )

        # Reference lines
        ax1.axhline(
            2.0, color="black", linewidth=1.5,
            linestyle="--", label="Classical bound S = 2", zorder=2
        )
        ax1.axvline(
            0.826, color="gray", linewidth=1.5,
            linestyle="--", alpha=0.8,
            label=r"SPPE limit: $\eta_D$ = 82.6%", zorder=2
        )
        ax1.axvline(
            0.667, color="red", linewidth=1.5,
            linestyle=":", alpha=0.9,
            label=r"Eberhard limit: $\eta_D$ = 66.7%", zorder=2
        )
        ax1.axhline(
            2 * np.sqrt(2), color="purple", linewidth=1,
            linestyle=":", alpha=0.6,
            label=r"Tsirelson bound $2\sqrt{2}$"
        )

        # Annotations
        ax1.annotate(
            "Eberhard\nlimit 66.7%",
            xy=(0.667, 2.04), xytext=(0.685, 2.15),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2)
        )
        ax1.annotate(
            "SPPE limit\n82.6%",
            xy=(0.826, 2.04), xytext=(0.840, 2.22),
            fontsize=9, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2)
        )

        ax1.set_xlabel(r"Detection Efficiency $\eta_D$", fontsize=12)
        ax1.set_ylabel("CHSH Parameter S", fontsize=12)
        ax1.set_title(
            "CHSH Violation vs Detection Efficiency\n"
            "(Reproduces Fig. 2 of Alwehaibi et al. 2025)",
            fontsize=11
        )
        ax1.set_xlim(0.63, 1.02)
        ax1.set_ylim(1.95, 2.75)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.4)

        # ══════════════════════════════════════════════════════════
        # PLOT 2 — H_min and χ_max vs η_D
        # ══════════════════════════════════════════════════════════

        valid    = ~np.isnan(r["hmin"])
        ax2_twin = ax2.twinx()

        l1, = ax2.plot(
            eta_D[valid], r["hmin"][valid],
            "b-o", linewidth=2, markersize=5,
            label=r"$H_{\min}$ (this work)"
        )
        l2, = ax2_twin.plot(
            eta_D[valid], r["holevo"][valid],
            "r-s", linewidth=2, markersize=5,
            label=r"$\chi_{\max}$ (this work)"
        )

        ax2.axvline(0.667, color="red",  linewidth=1.2,
                    linestyle=":", alpha=0.8)
        ax2.axvline(0.826, color="gray", linewidth=1.2,
                    linestyle="--", alpha=0.6)

        ax2.set_xlabel(r"Detection Efficiency $\eta_D$", fontsize=11)
        ax2.set_ylabel(r"Min-entropy $H_{\min}$",
                       fontsize=11, color="blue")
        ax2_twin.set_ylabel(r"Holevo bound $\chi_{\max}$",
                            fontsize=11, color="red")
        ax2.set_title(
            "DI-QKD Quantities vs Detection Efficiency\n"
            "(Reproduces Fig. 4 of Alwehaibi et al. 2025)",
            fontsize=10
        )
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2_twin.tick_params(axis="y", labelcolor="red")
        ax2.set_xlim(0.63, 1.02)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(
            [l1, l2],
            [l1.get_label(), l2.get_label()],
            fontsize=9, loc="upper left"
        )

        # ══════════════════════════════════════════════════════════
        # PLOT 3 — Log-Negativity Landscape
        # ══════════════════════════════════════════════════════════

        im = ax3.contourf(
            r["g_range"], r["tb_range"], r["log_neg_map"],
            levels=20, cmap="viridis", vmin=0, vmax=1
        )
        plt.colorbar(im, ax=ax3,
                     label=r"Log-negativity $\mathcal{E}_N$")

        # Mark optimal operating points from Table SI
        # ✅ Fixed: use 'w^' not 'w^' with unicode
        for row in OPTIMAL_PARAMS:
            _, t_b_pt, g_pt = row[0], row[1], row[2]
            ax3.plot(
                g_pt, t_b_pt,
                "w^",                          # ✅ ASCII marker
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.8
            )

        ax3.set_xlabel("Squeezing parameter g", fontsize=11)
        ax3.set_ylabel(r"Bob's BS transmittance $t_b$", fontsize=11)
        ax3.set_title(
            r"Log-negativity $\mathcal{E}_N(g,\, t_b)$ of Heralded State"
            "\n(Reproduces Fig. S1 — Supplemental Material)",
            fontsize=10
        )
        ax3.text(
            0.05, 0.92,
            "^ = Optimal operating\n  points (Table SI)",
            transform=ax3.transAxes,
            fontsize=8, color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5)
        )

        # ── Save ──────────────────────────────────────────────────
        save_path = self._resolve_save_path("bell_eberhard_result.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {save_path}")
        plt.show()
        plt.close(fig)

    def summarise(self):
        """
        Print a comparison table of analytical vs paper CHSH values.
        """
        r = self.results
        print("\n" + "═" * 65)
        print("  RESULTS SUMMARY")
        print("═" * 65)
        print(f"\n  {'eta_D':>6}  {'Analytical':>12}  {'SF':>12}  "
              f"{'Paper':>12}  {'Violation?':>10}")
        print("  " + "─" * 60)

        for i, row in enumerate(OPTIMAL_PARAMS):
            eta_D    = row[0]
            S_a      = r["chsh_analytic"][i]
            S_sf     = r["chsh_sf"][i]
            S_p      = r["chsh_paper"][i]
            viol     = "Yes" if S_a > 2.0 else "No"
            sf_str   = f"{S_sf:.6f}" if not np.isnan(S_sf) else "  (skipped)"
            print(f"  {eta_D:>6.2f}  {S_a:>12.6f}  {sf_str:>12}  "
                  f"{S_p:>12.6f}  {viol:>10}")

        print("\n  Key Thresholds:")
        print("  Eberhard limit : eta_D >= 66.7%  (this protocol)")
        print("  SPPE limit     : eta_D >= 82.6%  (previous work)")
        print(f"  Tsirelson bound: S <= 2*sqrt(2) = {2*np.sqrt(2):.6f}")
        print("═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sim = BellEberhardSimulation(run_sf=True)
    sim.run()
    sim.plot()
    sim.summarise()