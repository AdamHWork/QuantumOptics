"""
base/simulation.py
──────────────────
Abstract base class for all quantum optics simulations.
Every simulation in this project inherits from QuantumSimulation
and must implement:
  - build_circuit() : defines the SF quantum circuit
  - run()           : executes the sweep and stores results
  - summarise()     : prints key results to the terminal
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import os
import numpy as np


@dataclass
class SimulationConfig:
    """
    Holds all configurable parameters for a simulation.
    Using a dataclass gives us clean, typed, self-documenting
    configuration with no loose global variables.

    Attributes
    ----------
    n_phases  : Number of phase points to sweep (resolution)
    n_shots   : Measurement shots per phase point (statistics)
    phi_1     : Fixed reference arm phase in radians
    cutoff    : Fock space Hilbert space truncation dimension
    backend   : Strawberry Fields backend to use
    """
    n_phases : int   = 200
    n_shots  : int   = 500
    phi_1    : float = 0.0
    cutoff   : int   = 3
    backend  : str   = "fock"


class QuantumSimulation(ABC):
    """
    Abstract base class for all quantum optics simulations.

    Subclasses must implement:
      - build_circuit(phi_2) : returns a configured sf.Program
      - run()                : executes the full phase sweep
      - summarise()          : prints terminal summary of results

    Shared functionality provided by this base class:
      - Engine creation and execution
      - Phase range generation
      - Result storage initialisation
      - Save path resolution
      - Progress reporting
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialise the simulation with a config.
        If no config is provided, defaults are used.

        Parameters
        ----------
        config : SimulationConfig, optional
            Simulation parameters. Uses defaults if not provided.
        """
        # Use provided config or fall back to defaults
        self.config = config or SimulationConfig()

        # Generate the phase sweep array from 0 → 2π
        self.phi_range = np.linspace(0, 2 * np.pi, self.config.n_phases)

        # Resolve the save directory to the folder of the
        # script that instantiates the subclass — ensures the
        # PNG always saves next to the calling script
        self.save_dir = os.path.dirname(os.path.abspath(__file__))

        # Result storage — subclasses populate these in run()
        self.results = {}

    @abstractmethod
    def build_circuit(self, phi_2: float):
        """
        Build and return the Strawberry Fields Program for a
        single shot at a given arm-2 phase value.

        Parameters
        ----------
        phi_2 : float
            The phase applied to arm 2 (radians) for this shot.

        Returns
        -------
        sf.Program
            The fully configured quantum circuit ready to run.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Execute the full phase sweep.
        Must populate self.results with simulation output.
        """
        pass

    @abstractmethod
    def summarise(self):
        """
        Print a summary of key results to the terminal.
        """
        pass

    def _run_circuit(self, prog):
        """
        Internal helper — runs a given sf.Program on the
        configured backend and returns the result object.

        This is shared by all subclasses so engine creation
        is never duplicated across simulation files.

        Parameters
        ----------
        prog : sf.Program
            The quantum circuit to execute.

        Returns
        -------
        sf.Result
            The measurement result from the engine.
        """
        import strawberryfields as sf
        eng = sf.Engine(
            self.config.backend,
            backend_options={"cutoff_dim": self.config.cutoff}
        )
        return eng.run(prog)

    def _resolve_save_path(self, filename: str) -> str:
        """
        Resolves an absolute save path for output files.
        Places files in the same directory as the subclass
        script that calls this method.

        Parameters
        ----------
        filename : str
            The output filename (e.g. 'result.png')

        Returns
        -------
        str
            The full absolute path to save the file to.
        """
        return os.path.join(self.save_dir, filename)

    def _report_progress(self, idx: int):
        """
        Prints a progress update every 20 phase points.

        Parameters
        ----------
        idx : int
            The current phase index (0-based).
        """
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{self.config.n_phases} phase points done.")