"""UTC observation time conversion and interval selection."""
from dataclasses import dataclass
from datetime import datetime
import numpy as np


def convert_to_second_of_day(time_array):
    """Convert scalar/array YYYYDDDHHMMSS, preserving fractional seconds."""
    values = np.atleast_1d(np.asarray(time_array, dtype=float))
    return (values % 1000000 // 10000) * 3600 + (values % 10000 // 100) * 60 + values % 100


def convert_to_HHMM(sod):
    return f'{int(sod) // 3600:02d}{int(sod) % 3600 // 60:02d}'


def observation_time(value, reference):
    """Accept YYYYDDDHHMMSS or HH:MM[:SS] on the preset's UTC date."""
    if isinstance(value, str) and ':' in value:
        clock = datetime.strptime(value, '%H:%M:%S' if value.count(':') == 2 else '%H:%M')
        return int(reference) // 1000000 * 1000000 + clock.hour * 10000 + clock.minute * 100 + clock.second
    value = int(value)
    datetime.strptime(str(value), '%Y%j%H%M%S')
    return value


@dataclass(frozen=True)
class TimeWindow:
    start: int
    end: int

    def __post_init__(self):
        for value in (self.start, self.end):
            datetime.strptime(str(value), '%Y%j%H%M%S')
        if self.start // 1000000 != self.end // 1000000 or self.start >= self.end:
            raise ValueError('Select an increasing interval within one UTC day')

    def mask(self, seconds):
        """Use open endpoints, as in the original analysis."""
        lower, upper = convert_to_second_of_day([self.start, self.end])
        return (np.asarray(seconds) > lower) & (np.asarray(seconds) < upper)
