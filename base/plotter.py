"""
base/plotter.py
───────────────
Shared plotting utilities for all quantum optics simulations.
Encapsulates the matplotlib boilerplate so simulation classes
only need to provide their data and labels.
"""

import numpy as np
import matplotlib.pyplot as plt


class InterferencePlotter:
    """
    Handles all plotting for two-port interference simulations.

    Produces a two-subplot figure showing:
      - Top    : Port 0 simulated dots vs theory curve
      - Bottom : Port 1 simulated dots vs theory curve

    Parameters
    ----------
    title     : Main figure title string
    save_path : Absolute path to save the PNG to
    """

    def __init__(self, title: str, save_path: str):
        self.title = title
        self.save_path = save_path

    def plot(
        self,
        phi_range: np.ndarray,
        prob_port0: np.ndarray,
        prob_port1: np.ndarray,
        theory_port0: np.ndarray,
        theory_port1: np.ndarray,
        label0: str = "Port 0",
        label1: str = "Port 1",
        theory_label0: str = r"Theory: $\cos^2(\Delta\phi/2)$",
        theory_label1: str = r"Theory: $\sin^2(\Delta\phi/2)$",
        subtitle0: str = "Port 0 — Constructive Interference Arm",
        subtitle1: str = "Port 1 — Destructive Interference Arm",
    ):
        """
        Generate, save, and display the interference fringe plot.

        Parameters
        ----------
        phi_range    : Phase sweep array in radians
        prob_port0   : Simulated probabilities for port 0
        prob_port1   : Simulated probabilities for port 1
        theory_port0 : Analytical theory curve for port 0
        theory_port1 : Analytical theory curve for port 1
        label0/1     : Legend labels for simulated data
        theory_label0/1 : Legend labels for theory curves
        subtitle0/1  : Subplot titles
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(self.title, fontsize=14, fontweight="bold")

        # ── Port 0 ────────────────────────────────────────────────
        axes[0].plot(
            np.degrees(phi_range), prob_port0,
            "o", markersize=3, alpha=0.6,
            color="#1f77b4", label=f"Simulated ({label0})"
        )
        axes[0].plot(
            np.degrees(phi_range), theory_port0,
            "-", linewidth=2, color="#1f77b4",
            alpha=0.9, label=theory_label0
        )
        axes[0].set_ylabel("Detection Probability", fontsize=11)
        axes[0].set_title(subtitle0, fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].set_ylim(-0.05, 1.15)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # ── Port 1 ────────────────────────────────────────────────
        axes[1].plot(
            np.degrees(phi_range), prob_port1,
            "o", markersize=3, alpha=0.6,
            color="#d62728", label=f"Simulated ({label1})"
        )
        axes[1].plot(
            np.degrees(phi_range), theory_port1,
            "-", linewidth=2, color="#d62728",
            alpha=0.9, label=theory_label1
        )
        axes[1].set_xlabel(
            r"Phase Difference $\Delta\phi$ (degrees)", fontsize=11
        )
        axes[1].set_ylabel("Detection Probability", fontsize=11)
        axes[1].set_title(subtitle1, fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].set_ylim(-0.05, 1.15)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        # ✅ Save FIRST, then show, then clean up
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {self.save_path}")
        plt.show()
        plt.close(fig)
