"""Compatibility entry point; implementation is in tw1.workflows.calc_density_fluctuation."""
from tw1.workflows.calc_density_fluctuation import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('calc_density_fluctuation')
