"""Regression checks for layout migration, intervals and import-safe entry points."""
import importlib
import unittest
from unittest.mock import patch
import numpy as np
from tw1.cases import get_case, presets
from tw1.paths import DATA_DIR, data_path
from tw1.time import TimeWindow, convert_to_second_of_day, observation_time


class LocalStructureTests(unittest.TestCase):
    def test_year_and_processing_variant(self):
        for relative, expected in (
            ('m1a04x_renew', '2021/m1a04x_renew'),
            ('outdated/m1a04x_PLL', '2021/outdated/m1a04x_PLL'),
            ('[phase]/m1a01x', '2021/[phase]/m1a01x'),
            ('m3b14x', '2023/m3b14x'),
            ('2023/m3b14x', '2023/m3b14x'),
        ):
            self.assertEqual(data_path(relative), DATA_DIR / expected)
        self.assertEqual(data_path(), DATA_DIR)

    def test_all_presets_reference_existing_exact_files(self):
        for workflow, cases in presets().items():
            for number in cases:
                with self.subTest(workflow=workflow, number=number):
                    get_case(workflow, number).validate_files()

    def test_default_observation_and_immutable_overrides(self):
        changed = get_case('analyse_wavelet_coherence', 9, start='08:05', end='08:15')
        self.assertEqual(changed.parameters['time_beg'], 2021277080500)
        original = get_case('analyse_wavelet_coherence', 9)
        self.assertEqual(original.parameters['time_beg'], 2021277080000)
        self.assertEqual(original.parameters['file1_name'], 'JsJschan3_1frephase1s.dat')
        self.assertEqual(original.parameters['time_formats'][original.parameters['file1_name']], 'sod')

    def test_numpy_scalars_fractions_and_open_endpoints(self):
        self.assertEqual(convert_to_second_of_day(np.int64(2021277080000))[0], 28800)
        self.assertAlmostEqual(convert_to_second_of_day(2021277080000.25)[0], 28800.25)
        mask = TimeWindow(2021277080000, 2021277080100).mask([28800, 28830, 28860])
        np.testing.assert_array_equal(mask, [False, True, False])
        self.assertEqual(observation_time('08:01:05', 2021277080000), 2021277080105)

    def test_invalid_selection_fails_before_analysis(self):
        with self.assertRaises(ValueError):
            get_case('analyse_wavelet_coherence', 9, start='09:00', end='08:00')
        with self.assertRaises(ValueError):
            get_case('analyse_wavelet_coherence', 999)
        with self.assertRaises(FileNotFoundError):
            get_case('analyse_wavelet_coherence', 9, file1='missing_variant.dat')

    def test_workflow_imports_do_not_read_observation_files(self):
        with patch('numpy.loadtxt', side_effect=AssertionError('Import attempted data access')):
            for workflow in presets():
                module = importlib.import_module('tw1.workflows.' + workflow)
                self.assertTrue(callable(module.main))

    def test_station_time_format_and_polarity(self):
        from tw1.io import read_series
        with patch('numpy.loadtxt', return_value=np.array([[28800., 2.], [28801., 3.]])):
            time, seconds, frequency = read_series('station.dat', time_format='sod', sign=-1)
        np.testing.assert_array_equal(time, seconds)
        np.testing.assert_array_equal(frequency, [-2., -3.])

    def test_density_fit_uses_arguments_without_workflow_globals(self):
        from tw1.density import cal_beta
        frequency = np.geomspace(0.001, 0.1, 100)
        beta = cal_beta(frequency ** -1.7, frequency, 0.001, 0.1, show=False)
        self.assertAlmostEqual(beta, 1.7)


if __name__ == '__main__':
    unittest.main()
