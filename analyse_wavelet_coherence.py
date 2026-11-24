"""Compatibility entry point; implementation is in tw1.workflows.analyse_wavelet_coherence."""
from tw1.workflows.analyse_wavelet_coherence import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('analyse_wavelet_coherence')
