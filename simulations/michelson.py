"""
simulations/michelson.py
────────────────────────
Single-Photon Michelson Interferometer Simulation.
Inherits all shared logic from QuantumSimulation base class.

Physics:
  A single photon enters a 50/50 beamsplitter and is split
  into a superposition across two arms. Each arm acquires a
  phase shift simulating mirror displacement. The photon
  recombines at the BS and is detected at one of two ports.

  P_port0 = cos²(Δφ/2)   — constructive interference
  P_port1 = sin²(Δφ/2)   — destructive interference
"""

from base.plotter import InterferencePlotter
from base.simulation import QuantumSimulation, SimulationConfig
from strawberryfields.ops import Fock, BSgate, Rgate, MeasureFock
import strawberryfields as sf
import sys
import os
import numpy as np

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MichelsonInterferometer(QuantumSimulation):
    """
    Single-photon Michelson Interferometer simulation.

    Extends QuantumSimulation with:
      - A 2-mode Fock-basis quantum circuit
      - A full phase sweep from 0 → 2π
      - Comparison against cos²/sin² theory curves
      - Automatic plot generation and saving

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation parameters. Uses defaults if not provided.
    """

    def __init__(self, config=None):
        # Initialise the base class (sets up phi_range,
        # save_dir, config, and empty results dict)
        super().__init__(config)

        # Override save_dir to point to THIS file's directory
        # so the PNG saves next to michelson.py, not base/
        self.save_dir = os.path.dirname(os.path.abspath(__file__))

        print("Michelson Interferometer Simulation")
        print(f"  Phase sweep : 0 → 2π  ({self.config.n_phases} points)")
        print(f"  Shots/point : {self.config.n_shots}\n")

    def build_circuit(self, phi_2: float) -> sf.Program:
        """
        Build the Michelson interferometer quantum circuit
        for a single shot at arm-2 phase phi_2.

        Circuit structure:
          |1⟩|0⟩ → BS(π/4) → R(φ₁) R(φ₂) → BS(π/4) → Measure

        Parameters
        ----------
        phi_2 : float
            Phase applied to arm 2 for this shot (radians).

        Returns
        -------
        sf.Program
            The fully configured 2-mode quantum circuit.
        """
        prog = sf.Program(2)

        with prog.context as q:
            # ── State Preparation ──────────────────────────
            # Inject 1 photon into mode 0; mode 1 = vacuum
            Fock(1) | q[0]

            # ── Beamsplitter — Splitting ───────────────────
            # 50/50 split: θ=π/4, no extra BS phase: φ=0
            # Creates superposition across both arms
            BSgate(np.pi / 4, 0.0) | (q[0], q[1])

            # ── Arm Phase Shifts ───────────────────────────
            # Simulate round-trip optical path in each arm
            # Mode 0: fixed reference arm (φ_1)
            # Mode 1: swept movable arm (φ_2)
            Rgate(self.config.phi_1) | q[0]
            Rgate(phi_2) | q[1]

            # ── Beamsplitter — Recombination ───────────────
            # Interference occurs here. Δφ determines whether
            # constructive or destructive at each port.
            BSgate(np.pi / 4, 0.0) | (q[0], q[1])

            # ── Measurement ───────────────────────────────
            # Collapse to photon number — always (1,0) or (0,1)
            MeasureFock() | q[1]
            MeasureFock() | q[0]

        return prog

    def run(self):

        # Name these according to physical behaviour, not mode number
        # Under SF's BSgate convention, mode 1 is constructive
        prob_constructive = np.zeros(self.config.n_phases)
        prob_destructive = np.zeros(self.config.n_phases)

        for idx, phi_2 in enumerate(self.phi_range):

            count_constructive = 0
            count_destructive = 0

            for _ in range(self.config.n_shots):
                prog = self.build_circuit(phi_2)
                result = self._run_circuit(prog)

                m0, m1 = result.samples[0]

                # m1 = constructive port under SF BSgate convention
                # m0 = destructive port under SF BSgate convention
                count_constructive += int(m1 == 1)
                count_destructive += int(m0 == 1)

            prob_constructive[idx] = count_constructive / self.config.n_shots
            prob_destructive[idx] = count_destructive / self.config.n_shots

            self._report_progress(idx)

        # Store with physically meaningful keys
        self.results["prob_port0"] = prob_constructive   # cos² port
        self.results["prob_port1"] = prob_destructive    # sin² port

        print("\nSimulation complete. Generating plots...\n")
        return self

    def plot(self):
        """
        Generate and save the interference fringe plot.
        Must be called after run().

        Overlays simulated dots on analytical theory curves
        for both output ports across the full phase sweep.
        """
        delta_phi = self.phi_range - self.config.phi_1
        theory_port0 = np.cos(delta_phi / 2) ** 2
        theory_port1 = np.sin(delta_phi / 2) ** 2

        save_path = self._resolve_save_path(
            "michelson_interferometer_result.png"
        )

        plotter = InterferencePlotter(
            title="Single-Photon Michelson Interferometer\n"
                  "Strawberry Fields — Fock Backend",
            save_path=save_path
        )

        plotter.plot(
            phi_range=self.phi_range,
            prob_port0=self.results["prob_port0"],
            prob_port1=self.results["prob_port1"],
            theory_port0=theory_port0,
            theory_port1=theory_port1,
        )

    def summarise(self):
        """
        Print key simulation statistics to the terminal.
        Provides a quick sanity check against known physics.

        Expected values:
          Max P(Port 0) ≈ 1.0  at Δφ = 0°
          Min P(Port 0) ≈ 0.0  at Δφ = 180°
          Max P(Port 1) ≈ 1.0  at Δφ = 180°
          Min P(Port 1) ≈ 0.0  at Δφ = 0°
        """
        p0 = self.results["prob_port0"]
        p1 = self.results["prob_port1"]

        print("\n── Key Results ──────────────────────────────────────────")
        print(f"  Max P(Port 0): {p0.max():.3f}  (expect ≈ 1.0 at Δφ = 0°)")
        print(f"  Min P(Port 0): {p0.min():.3f}  (expect ≈ 0.0 at Δφ = 180°)")
        print(f"  Max P(Port 1): {p1.max():.3f}  (expect ≈ 1.0 at Δφ = 180°)")
        print(f"  Min P(Port 1): {p1.min():.3f}  (expect ≈ 0.0 at Δφ = 0°)")
        print("─────────────────────────────────────────────────────────\n")
