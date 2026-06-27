# Initialise terminal using .\venv\Scripts\Activate.ps1
# run using python michelson_interferometer.py
"""
=============================================================
  Single-Photon Michelson Interferometer Simulation
  Using Strawberry Fields (Xanadu)
  https://strawberryfields.readthedocs.io/en/latest/
=============================================================

Physics Overview:
-----------------
A Michelson Interferometer splits a single photon at a 50/50
beamsplitter (BS). Each arm acquires a phase shift (φ_1, φ_2)
simulating the path-length difference due to mirror positions.
The photon then recombines at the same BS.

In the quantum circuit model:
  - Mode 0 = "input arm" (photon enters here)
  - Mode 1 = "vacuum arm" (vacuum enters here)
  - BSgate(π/4, 0) → 50/50 beamsplitter
  - Rgate(φ)       → phase shift per arm (mirror displacement)
  - BSgate(π/4, 0) → recombination at BS (return trip)

The output photon probability at each port is measured via
Fock-state sampling across a sweep of phase differences Δφ.

Expected Result:
  P_detector0 = cos²(Δφ/2)   (constructive/destructive fringes)
  P_detector1 = sin²(Δφ/2)
=============================================================
"""

# ══════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════
# numpy       : Maths — arrays, π, linspace, cos, sin
# matplotlib  : Plotting the final interference fringe results
# strawberryfields : The quantum optics simulation engine
# Fock        : Prepares a quantum state with exactly n photons
# BSgate      : Beamsplitter gate — splits and recombines light
# Rgate       : Phase rotation gate — simulates mirror displacement
# MeasureFock : Measures the photon number in each mode
# os          : Used to build an absolute save path for the PNG

import os
import numpy as np
import matplotlib.pyplot as plt
import strawberryfields as sf
from strawberryfields.ops import Fock, BSgate, Rgate, MeasureFock


# ══════════════════════════════════════════════════════════════
# SECTION 2 — SAVE PATH
# ══════════════════════════════════════════════════════════════
# os.path.abspath(__file__) gets the full absolute path of this
# script file, regardless of where the terminal is pointed.
# os.path.dirname() strips the filename, leaving just the folder.
# os.path.join() then appends the PNG filename to that folder.
# This guarantees the PNG always saves next to this script file.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SCRIPT_DIR, "michelson_interferometer_result.png")


# ══════════════════════════════════════════════════════════════
# SECTION 3 — SIMULATION PARAMETERS
# ══════════════════════════════════════════════════════════════
# N_PHASES : How many phase values to sweep from 0 → 2π.
#            More points = smoother fringe curve.
#
# N_SHOTS  : How many independent quantum measurements to take
#            at each phase point. Quantum measurement is
#            probabilistic — a single shot only returns 0 or 1.
#            More shots = closer to the true probability.
#
# PHI_1    : The fixed phase on arm 1 (the reference arm).
#            In a real Michelson this is the stationary mirror.
#
# PHI_RANGE: Array of 200 evenly spaced phase values for arm 2.
#            This simulates sweeping the movable mirror position.

N_PHASES = 200
N_SHOTS = 500
PHI_1 = 0.0
PHI_RANGE = np.linspace(0, 2 * np.pi, N_PHASES)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — RESULT STORAGE
# ══════════════════════════════════════════════════════════════
# Two arrays to accumulate the detection probability at each
# port across the full phase sweep. Initialised to all zeros.

prob_port0 = np.zeros(N_PHASES)   # Probability photon exits port 0
prob_port1 = np.zeros(N_PHASES)   # Probability photon exits port 1

print("Running Single-Photon Michelson Interferometer simulation...")
print(f"  Phase sweep : 0 → 2π  ({N_PHASES} points)")
print(f"  Shots/point : {N_SHOTS}\n")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN PHASE SWEEP LOOP
# ══════════════════════════════════════════════════════════════
# Outer loop : steps through each of the 200 phase values.
# Inner loop : runs N_SHOTS independent experiments at that phase.
#
# At each phase point we count how many times the photon was
# detected at port 0 vs port 1, then divide by N_SHOTS to get
# the empirical probability for that phase value.

for idx, phi_2 in enumerate(PHI_RANGE):

    count_port0 = 0
    count_port1 = 0

    for _ in range(N_SHOTS):

        # ── SECTION 5a — CREATE THE QUANTUM CIRCUIT ───────────────
        # sf.Program(2) creates a 2-mode quantum optical circuit.
        # Each "mode" is a beam path:
        #   q[0] = mode 0 → Arm 1 (reference arm, fixed mirror)
        #   q[1] = mode 1 → Arm 2 (swept arm, movable mirror)

        prog = sf.Program(2)

        with prog.context as q:

            # ── SECTION 5b — STATE PREPARATION ───────────────────
            # Fock(1) injects exactly 1 photon into mode 0.
            # Mode 1 is left in the vacuum state |0⟩ (no photons).
            # The initial 2-mode state is therefore:
            #
            #   |ψ_in⟩ = |1⟩₀ ⊗ |0⟩₁
            #
            # The photon enters from the source, just like a laser
            # pulse entering the beamsplitter in a real Michelson.

            Fock(1) | q[0]

            # ── SECTION 5c — FIRST BEAMSPLITTER (SPLITTING) ──────
            # BSgate(θ, φ_bs) applies a beamsplitter transformation.
            #   θ = π/4  → 50/50 split (equal transmission/reflection)
            #   φ_bs = 0 → no additional phase from the BS itself
            #
            # After this gate the single photon is in a quantum
            # superposition across both arms simultaneously:
            #
            #   |ψ⟩ = (1/√2)(|1⟩₀|0⟩₁ + i|0⟩₀|1⟩₁)
            #
            # This is purely quantum — the photon does not "choose"
            # an arm. It genuinely exists in both arms at once.

            BSgate(np.pi / 4, 0.0) | (q[0], q[1])

            # ── SECTION 5d — ARM PHASE SHIFTS (MIRROR POSITIONS) ─
            # Rgate(φ) applies a phase rotation to that mode.
            # This simulates the round-trip optical path length
            # each arm accumulates before returning to the BS.
            #
            # In a real Michelson, moving a mirror by λ/4 changes
            # the round-trip path by λ/2, adding a phase of π.
            #
            #   Mode 0 (arm 1): fixed reference phase φ_1 = 0
            #   Mode 1 (arm 2): swept phase φ_2 (current loop value)
            #
            # Mathematically each Rgate acts as:
            #   |1⟩ → e^(iφ)|1⟩

            Rgate(PHI_1) | q[0]   # Fixed reference arm
            Rgate(phi_2) | q[1]   # Swept movable arm

            # ── SECTION 5e — SECOND BEAMSPLITTER (RECOMBINATION) ─
            # The two arms recombine at the same beamsplitter.
            # This is where quantum interference occurs.
            #
            # The phase difference Δφ = φ_2 - φ_1 determines
            # whether the interference is constructive or destructive:
            #
            #   Δφ = 0   → constructive at port 0 (photon always → port 0)
            #   Δφ = π   → destructive at port 0  (photon always → port 1)
            #   Δφ = 2π  → constructive again at port 0
            #
            # This is the same fringe pattern seen in a real
            # Michelson interferometer with a classical laser.

            BSgate(np.pi / 4, 0.0) | (q[0], q[1])

            # ── SECTION 5f — MEASUREMENT ──────────────────────────
            # MeasureFock() collapses the quantum state and returns
            # the photon number at each port.
            #
            # Since we started with exactly 1 photon, the result
            # is always either (1, 0) or (0, 1) — the photon exits
            # one port or the other, never both, never neither.
            # This is photon anti-bunching — a quantum effect.

            MeasureFock() | q[0]
            MeasureFock() | q[1]

        # ── SECTION 5g — RUN THE ENGINE ───────────────────────────
        # sf.Engine("fock") simulates the circuit in the photon-
        # number (Fock) basis. This is an exact simulation method
        # for discrete photon states — no approximations.
        #
        # cutoff_dim=3 truncates the Hilbert space at 3 photons
        # (states: |0⟩, |1⟩, |2⟩). Since we only ever have 1
        # photon in the system, this is more than sufficient and
        # keeps computation fast.

        eng = sf.Engine("fock", backend_options={"cutoff_dim": 3})
        result = eng.run(prog)

        # ── SECTION 5h — READ MEASUREMENT RESULTS ─────────────────
        # result.samples[0] contains the measurement outcomes for
        # this shot as a list: [photon_count_mode0, photon_count_mode1]
        # We check if each mode registered exactly 1 photon and
        # increment the appropriate counter.

        m0 = result.samples[0][0]   # Photon count at port 0
        m1 = result.samples[0][1]   # Photon count at port 1

        count_port0 += int(m0 == 1)
        count_port1 += int(m1 == 1)

    # ── SECTION 5i — NORMALISE TO PROBABILITY ─────────────────────
    # Dividing the raw counts by N_SHOTS converts them into
    # empirical probabilities for this phase value.
    # As N_SHOTS → ∞ these approach the exact quantum prediction.

    prob_port0[idx] = count_port0 / N_SHOTS
    prob_port1[idx] = count_port1 / N_SHOTS

    if (idx + 1) % 20 == 0:
        print(f"  Progress: {idx + 1}/{N_PHASES} phase points done.")

print("\nSimulation complete. Generating plots...\n")


# ══════════════════════════════════════════════════════════════
# SECTION 6 — ANALYTICAL REFERENCE CURVES
# ══════════════════════════════════════════════════════════════
# These are the exact theoretical predictions from quantum optics.
# The simulated dots should closely follow these smooth curves.
# Any deviation is quantum shot noise from the finite N_SHOTS.
#
#   P_port0 = cos²(Δφ/2)   — constructive interference fringe
#   P_port1 = sin²(Δφ/2)   — destructive interference fringe
#
# Note: P_port0 + P_port1 = 1 always (photon must go somewhere).

delta_phi = PHI_RANGE - PHI_1
theory_port0 = np.cos(delta_phi / 2) ** 2
theory_port1 = np.sin(delta_phi / 2) ** 2


# ══════════════════════════════════════════════════════════════
# SECTION 7 — PLOTTING
# ══════════════════════════════════════════════════════════════
# Two subplots sharing the same x-axis (phase difference).
# Each subplot overlays the simulated dots on the theory curve.
#
# IMPORTANT — ORDER OF OPERATIONS:
#   1. plt.savefig() MUST come before plt.show().
#      plt.show() clears the figure from memory after displaying
#      it. Calling savefig() after show() saves a blank image.
#   2. plt.close(fig) releases the figure from memory cleanly
#      after both saving and displaying are complete.

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle(
    "Single-Photon Michelson Interferometer\n"
    "Strawberry Fields — Fock Backend",
    fontsize=14, fontweight="bold"
)

# — Port 0 (Constructive Interference Arm) —
axes[0].plot(
    np.degrees(PHI_RANGE), prob_port0,
    "o", markersize=3, alpha=0.6, color="#1f77b4", label="Simulated (Port 0)"
)
axes[0].plot(
    np.degrees(PHI_RANGE), theory_port0,
    "-", linewidth=2, color="#1f77b4", alpha=0.9, label=r"Theory: $\cos^2(\Delta\phi/2)$"
)
axes[0].set_ylabel("Detection Probability", fontsize=11)
axes[0].set_title("Port 0 — Constructive Interference Arm", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_ylim(-0.05, 1.15)
axes[0].grid(True, linestyle="--", alpha=0.5)

# — Port 1 (Destructive Interference Arm) —
axes[1].plot(
    np.degrees(PHI_RANGE), prob_port1,
    "o", markersize=3, alpha=0.6, color="#d62728", label="Simulated (Port 1)"
)
axes[1].plot(
    np.degrees(PHI_RANGE), theory_port1,
    "-", linewidth=2, color="#d62728", alpha=0.9, label=r"Theory: $\sin^2(\Delta\phi/2)$"
)
axes[1].set_xlabel(r"Phase Difference $\Delta\phi$ (degrees)", fontsize=11)
axes[1].set_ylabel("Detection Probability", fontsize=11)
axes[1].set_title("Port 1 — Destructive Interference Arm", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.05, 1.15)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()

# ✅ Save FIRST (before show clears the figure)
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Plot saved → {SAVE_PATH}")

# ✅ Then display
plt.show()

# ✅ Then clean up memory
plt.close(fig)


# ══════════════════════════════════════════════════════════════
# SECTION 8 — CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════
# Prints key statistics to the terminal to give a quick sanity
# check that the simulation produced physically correct results.
#
# At Δφ = 0°   → P(Port 0) should be ≈ 1.0, P(Port 1) ≈ 0.0
# At Δφ = 180° → P(Port 0) should be ≈ 0.0, P(Port 1) ≈ 1.0

print("\n── Key Results ──────────────────────────────────────────────────────")
print(f"  Max P(Port 0): {prob_port0.max():.3f}  (expect ≈ 1.0 at Δφ = 0°)")
print(f"  Min P(Port 0): {prob_port0.min():.3f}  (expect ≈ 0.0 at Δφ = 180°)")
print(f"  Max P(Port 1): {prob_port1.max():.3f}  (expect ≈ 1.0 at Δφ = 180°)")
print(f"  Min P(Port 1): {prob_port1.min():.3f}  (expect ≈ 0.0 at Δφ = 0°)")
print("─────────────────────────────────────────────────────────────────────\n")
