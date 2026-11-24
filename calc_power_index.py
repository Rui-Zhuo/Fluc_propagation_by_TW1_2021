"""Compatibility entry point; implementation is in tw1.workflows.calc_power_index."""
from tw1.workflows.calc_power_index import main

if __name__ == "__main__":
    from tw1.cli import run_workflow
    run_workflow('calc_power_index')
