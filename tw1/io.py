"""Observation readers and explicit PLL conversion."""
from pathlib import Path
import numpy as np
from .paths import data_path
from .time import convert_to_second_of_day


def read_series(path, *, time_format='utc_code', column=1, sign=1):
    """Read a station series, returning original time, seconds/day and values."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[0] < 2 or data.shape[1] <= column:
        raise ValueError(f'Insufficient samples/columns: {path}')
    time = data[:, 0]
    if time_format not in ('utc_code', 'sod'):
        raise ValueError(f'Unknown observation time format: {time_format}')
    seconds = time if time_format == 'sod' else convert_to_second_of_day(time)
    return time, seconds, sign * data[:, column]


def convert_pll(observation, station, *, channel='chan3', output=None):
    """Read Fre_Pha_data and write the chosen text product; return its path."""
    from scipy.io import loadmat
    directory = data_path(observation)
    source = directory / station / channel / 'Time_Frequency_Phase.mat'
    values = loadmat(source)['Fre_Pha_data']
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f'Invalid Fre_Pha_data shape in {source}: {values.shape}')
    target = Path(output) if output is not None else directory / f'{station}{station}{channel}_1frephase.dat'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, values, fmt='%f')
    return target
