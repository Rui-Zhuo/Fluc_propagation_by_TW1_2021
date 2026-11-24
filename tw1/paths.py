"""Local layout. Observation processing variants are never substituted."""
from pathlib import Path
import os
import re

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
RESULT_DIR = PROJECT_DIR / 'RESULT'
DATA_DIR = Path(os.environ.get('TW1_DATA_DIR', 'E:/Research/Data/Tianwen-1'))
GEOMETRY_DIR = CODE_DIR / 'Sun_Mars_Earth'
EXTERNAL_UTILS_DIR = Path(os.environ.get('TW1_UTILS_DIR', 'E:/Research/Program/[else]'))


def data_path(relative=''):
    """Map observation-relative names to year/observation, including outdated/[phase]."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Data paths must be relative to TW1_DATA_DIR')
    if path.parts and path.parts[0] not in ('2021', '2023'):
        observation = next((p for p in path.parts if re.match(r'^[ms][13]', p)), None)
        if observation:
            path = Path('2023' if observation[1] == '3' else '2021') / path
    return DATA_DIR / path


def data_directory(relative=''):
    """Compatibility string for legacy filename concatenation."""
    return data_path(relative).as_posix().rstrip('/') + '/'


def result_path(relative=''):
    """Return a result path without creating a directory."""
    return RESULT_DIR / relative


def result_directory(relative=''):
    """Create an output directory and return a slash-terminated string."""
    directory = result_path(relative)
    directory.mkdir(parents=True, exist_ok=True)
    return directory.as_posix().rstrip('/') + '/'
