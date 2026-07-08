"""
run.py
──────
Entry point for all QuantumOptics simulations.
Run any simulation by name from the terminal:

    python run.py michelson

Each simulation can also be configured by editing the
SimulationConfig passed to it here.
"""

import sys
from base.simulation import SimulationConfig
from simulations.michelson     import MichelsonInterferometer
from simulations.bell_eberhard import BellEberhardSimulation, FOCK_CUTOFF

SIMULATIONS = {
    "michelson"     : MichelsonInterferometer,
    "bell_eberhard" : BellEberhardSimulation,
}

def main():
    """Main entry point for running simulations."""
    name = sys.argv[1] if len(sys.argv) > 1 else "michelson"

    if name not in SIMULATIONS:
        print(f"Unknown simulation '{name}'.")
        print(f"Available: {list(SIMULATIONS.keys())}")
        sys.exit(1)

    config = SimulationConfig(
        n_phases = 200,
        n_shots  = 500,
        phi_1    = 0.0,
        cutoff   = FOCK_CUTOFF if name == "bell_eberhard" else 3,
        backend  = "fock"
    )

    # Bell-Eberhard can be run analytics-only for speed
    if name == "bell_eberhard":
        sim = BellEberhardSimulation(config, run_sf=True)
    else:
        sim = SIMULATIONS[name](config)

    sim.run()
    sim.plot()
    sim.summarise()

if __name__ == "__main__":
    main()