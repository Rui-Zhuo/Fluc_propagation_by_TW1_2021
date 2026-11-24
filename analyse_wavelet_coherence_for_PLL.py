"""Compatibility entry point; implementation is in tw1.workflows.analyse_wavelet_coherence_for_PLL."""
from tw1.workflows.analyse_wavelet_coherence_for_PLL import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('analyse_wavelet_coherence_for_PLL')
