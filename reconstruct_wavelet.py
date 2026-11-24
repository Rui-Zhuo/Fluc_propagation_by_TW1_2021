"""Compatibility entry point; implementation is in tw1.workflows.reconstruct_wavelet."""
from tw1.workflows.reconstruct_wavelet import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('reconstruct_wavelet')
