"""Compatibility entry point; implementation is in tw1.workflows.statisticize_wavelet_coherence_inward."""
from tw1.workflows.statisticize_wavelet_coherence_inward import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('statisticize_wavelet_coherence_inward')
