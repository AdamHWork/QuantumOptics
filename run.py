"""
run.py
──────
Entry point for all QuantumOptics simulations.
Run any simulation by name from the terminal:

    python run.py michelson
    python run.py hong_ou_mandel      ← future
    python run.py boson_sampling      ← future

Each simulation can also be configured by editing the
SimulationConfig passed to it here.
"""

import sys
from base.simulation import SimulationConfig
from simulations.michelson import MichelsonInterferometer

# ── Available Simulations ─────────────────────────────────────
SIMULATIONS = {
    "michelson": MichelsonInterferometer,
    # "hong_ou_mandel" : HongOuMandel,    ← add future sims here
    # "boson_sampling" : BosonSampling,
}

def main():
    # Default to michelson if no argument given
    name = sys.argv[1] if len(sys.argv) > 1 else "michelson"

    if name not in SIMULATIONS:
        print(f"Unknown simulation '{name}'.")
        print(f"Available: {list(SIMULATIONS.keys())}")
        sys.exit(1)

    # ── Configure ─────────────────────────────────────────────
    # Edit these values to change simulation parameters
    config = SimulationConfig(
        n_phases = 200,
        n_shots  = 500,
        phi_1    = 0.0,
        cutoff   = 3,
        backend  = "fock"
    )

    # ── Run ───────────────────────────────────────────────────
    sim = SIMULATIONS[name](config)
    sim.run()
    sim.plot()
    sim.summarise()

if __name__ == "__main__":
    main()