"""Declarative observation presets, independent of plotting/scientific dependencies."""
from dataclasses import dataclass
import json
from pathlib import Path
from .paths import data_path, result_path, result_directory

PRESET_FILE = Path(__file__).with_name('cases.json')


def presets():
    with PRESET_FILE.open(encoding='utf8') as stream:
        return json.load(stream)


@dataclass(frozen=True)
class ObservationCase:
    workflow: str
    number: int
    data_relative: str
    result_relative: str
    description: str
    parameters: dict

    @property
    def input_directory(self):
        return data_path(self.data_relative).as_posix().rstrip('/') + '/'

    def output_directory(self, create=False):
        if create:
            return result_directory(self.result_relative)
        return result_path(self.result_relative).as_posix().rstrip('/') + '/'

    def selected_files(self):
        return [data_path(self.data_relative) / value for key, value in self.parameters.items()
                if key in ('file_name', 'file1_name', 'file2_name', 'file3_name')]

    def validate_files(self):
        missing = [str(path) for path in self.selected_files() if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing selected observation files (no variant substitution):\n' + '\n'.join(missing))


def get_case(workflow, number, *, start=None, end=None, file1=None, file2=None, validate=True):
    """Load a preset; overrides never mutate the saved configuration."""
    try:
        values = presets()[workflow][str(number)].copy()
    except KeyError as error:
        raise ValueError(f'Unknown case {number} for {workflow}') from error
    metadata = {key: values.pop(key) for key in ('data_relative', 'result_relative', 'description')}
    for key, value in (('file1_name', file1), ('file2_name', file2)):
        if value is not None:
            if key not in values:
                raise ValueError(f'{workflow} does not support a station-pair override')
            if Path(value).name != value:
                raise ValueError('Select an exact filename within the preset observation directory')
            if value not in values.get('time_formats', {}) and (data_path(metadata['data_relative']) / value).is_file():
                raise ValueError('Declare the new filename and its time format in cases.json first')
            values[key] = value
    if 'time_beg' in values:
        from .time import observation_time, TimeWindow
        values['time_beg'] = observation_time(start, values['time_beg']) if start is not None else values['time_beg']
        values['time_end'] = observation_time(end, values['time_end']) if end is not None else values['time_end']
        TimeWindow(values['time_beg'], values['time_end'])
        if 'time_lb' in values and not (values['time_beg'] <= values['time_lb'] < values['time_ub'] <= values['time_end']):
            raise ValueError('The spectral fitting interval must lie within the selected time interval; edit time_lb/time_ub in cases.json')
    elif start is not None or end is not None:
        raise ValueError('This workflow uses sample-based selection; see its implementation')
    case = ObservationCase(workflow, int(number), parameters=values, **metadata)
    if validate:
        case.validate_files()
    return case
