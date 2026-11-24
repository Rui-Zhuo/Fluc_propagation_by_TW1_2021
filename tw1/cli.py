"""One local command for discovery, input checks and analysis."""
import argparse
from importlib import import_module
from .cases import presets, get_case

WORKFLOWS = (
    'analyse_wavelet_coherence', 'analyse_wavelet_coherence_for_phase',
    'analyse_wavelet_coherence_for_PLL', 'reconstruct_wavelet', 'calc_power_index',
    'plot_wavelet_spectrum', 'calc_density_fluctuation', 'plot_wavelet_reconstruction',
    'plot_density_fluctuation', 'statisticize_wavelet_coherence',
    'statisticize_wavelet_coherence_inward', 'analyse_multiple_baseline',
    'analyse_multiple_baseline_for_spikes', 'read_PLL_data',
)


def run_workflow(workflow=None, argv=None):
    parser = argparse.ArgumentParser(description='Local Tianwen-1 analysis (UTC times)')
    if workflow is None:
        parser.add_argument('workflow', choices=WORKFLOWS, nargs='?')
    parser.add_argument('--list', action='store_true', help='List workflows or cases without running analysis')
    parser.add_argument('--check', action='store_true', help='Check all selected preset files')
    parser.add_argument('--case', type=int)
    parser.add_argument('--start', help='UTC HH:MM[:SS] or YYYYDDDHHMMSS')
    parser.add_argument('--end', help='UTC HH:MM[:SS] or YYYYDDDHHMMSS')
    parser.add_argument('--file1', help='Exact filename for first station')
    parser.add_argument('--file2', help='Exact filename for second station')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--plot', type=int, choices=(0, 1), help='0=Plotly, 1=Matplotlib')
    parser.add_argument('--show', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    workflow = workflow or args.workflow
    catalog = presets()
    if args.list:
        if workflow in catalog:
            for number, item in catalog[workflow].items():
                print(f'{number:>2}  {item["data_relative"]}  {item["description"]}')
        else:
            print('\n'.join(WORKFLOWS))
        return
    if args.check:
        if workflow and workflow not in catalog:
            parser.error('--check is available for workflows with observation presets')
        failures = []
        for name, cases in catalog.items():
            if workflow and name != workflow:
                continue
            for number in cases:
                if args.case is not None and int(number) != args.case:
                    continue
                try:
                    get_case(name, number, start=args.start, end=args.end, file1=args.file1, file2=args.file2)
                except (FileNotFoundError, ValueError) as error:
                    failures.append(f'{name} / {number}: {error}')
        print('\n'.join(failures) if failures else 'All selected preset files exist.')
        if failures:
            raise SystemExit(1)
        return
    if workflow is None:
        parser.error('Select a workflow or use --list / --check')
    if workflow in ('analyse_multiple_baseline', 'analyse_multiple_baseline_for_spikes', 'read_PLL_data'):
        if any(value is not None for value in (args.case, args.start, args.end, args.file1, args.file2, args.save, args.plot)):
            parser.error('This workflow uses its own main() parameters; see README.md')
        if workflow == 'read_PLL_data':
            import_module(workflow).main()
        else:
            import_module(workflow).main(show=args.show)
    else:
        import_module('tw1.workflows.' + workflow).main(
            case=args.case, start=args.start, end=args.end, save=args.save,
            plot=args.plot, file1=args.file1, file2=args.file2, show=args.show)
