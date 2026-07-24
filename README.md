# QuantumOptics

Quantum optics simulations built on [Strawberry Fields](https://strawberryfields.ai/), covering single-photon interference and a full reproduction of a recent loophole-free Bell test protocol.

## Simulations

### 1. Single-Photon Michelson Interferometer (`simulations/michelson.py`)

A single photon enters a 50/50 beamsplitter, is split into a superposition across two arms, picks up a relative phase shift (simulating a mirror displacement), and recombines at a second beamsplitter before detection.

- Circuit: `|1⟩|0⟩ → BS(π/4) → R(φ₁)R(φ₂) → BS(π/4) → MeasureFock`
- Sweeps the arm-2 phase φ₂ over 0 → 2π and estimates detection probabilities at each port from repeated shots
- Validates against the analytical fringe pattern, P = cos²(Δφ/2) and sin²(Δφ/2)
- Outputs a two-panel plot comparing simulated points against the theory curves

Run with:
```bash
python run.py michelson
```

### 2. Bell-Eberhard Test (`simulations/bell_eberhard.py`)

A reproduction of the detection-loophole-free Bell test protocol from Alwehaibi, Mer, Machado, Yu, Walmsley & Patel, *"A Tractable Protocol for Detection-Loophole-Free Bell Tests over Long Distances,"* Phys. Rev. Research 7, 043198 (2025), [arXiv:2506.05048](https://arxiv.org/abs/2506.05048).

The simulation computes the CHSH parameter S as a function of detector efficiency η_D, using two independent methods that are cross-checked against each other and against the paper's published values:

- **Analytical path** — derives the heralded two-mode state from the paper's closed-form expressions and evaluates the CHSH correlator directly from the Husimi Q-function (Eqs. S21–S24).
- **Numerical path** — builds the corresponding circuit in Strawberry Fields (two-mode squeezing, beamsplitter, loss channels, Fock-basis measurement with photon-number-resolving detector model) and extracts the heralded density matrix from the simulated state.

Also computed:
- Device-independent QKD quantities (min-entropy and Holevo bound) as functions of η_D
- The log-negativity of the heralded state across the (squeezing, beamsplitter transmittance) parameter space, reproducing the paper's entanglement landscape

Output is a three-panel figure: CHSH violation vs. η_D (reproducing the paper's Fig. 2, with the Eberhard and SPPE efficiency thresholds and the Tsirelson bound marked), the DI-QKD quantities (Fig. 4), and the log-negativity landscape (Fig. S1).

Run with:
```bash
python run.py bell_eberhard
```

**Note on η_D = 0.80**: the paper's Table SI entry at this efficiency doesn't reproduce the stated CHSH value when checked analytically (a transcription error). The code re-optimises the free parameters at runtime via Nelder-Mead to recover a consistent operating point, and flags this in the console output.

## Project structure

```
QuantumOptics/
├── run.py                       # entry point — python run.py <simulation_name>
├── base/
│   ├── simulation.py             # abstract QuantumSimulation base class
│   └── plotter.py                 # shared matplotlib plotting for interference-type results
├── simulations/
│   ├── michelson.py               # single-photon Michelson interferometer
│   └── bell_eberhard.py           # Bell-Eberhard CHSH violation reproduction
└── requirements.txt
```

All simulations share a common `QuantumSimulation` abstract base class (`base/simulation.py`), which handles engine configuration, phase sweeps, progress reporting, and output paths, so each simulation only needs to implement `build_circuit()`, `run()`, and `summarise()`.

## Requirements

- Python 3
- [Strawberry Fields](https://strawberryfields.ai/) 0.23
- [The Walrus](https://the-walrus.readthedocs.io/) 0.19 (used for exact Fock-basis displacement operators in the Bell-Eberhard POVM)
- NumPy, SciPy, Matplotlib

Install with:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py michelson        # single-photon interferometer
python run.py bell_eberhard    # Bell-Eberhard CHSH reproduction
```

Each run prints a results summary to the terminal and saves a plot alongside the simulation script.
