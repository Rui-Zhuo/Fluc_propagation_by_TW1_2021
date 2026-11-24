"""Scientific regression checks; run with python -m unittest test_multibaseline_fit."""
import unittest
import numpy as np
from multibaseline_fit import fit_velocity, correct_vel_direction, signed_radial_angle


class PropagationFitTests(unittest.TestCase):
    def test_recover_known_signed_plane_wave(self):
        phi = np.radians([10., 65., 140.])
        directions = np.column_stack((np.cos(phi),np.sin(phi)))
        speed, theta = 123.4, np.radians(35.7)
        observed = speed/np.cos(theta-phi)
        saved = observed.copy()
        fit = fit_velocity(directions,observed)
        self.assertAlmostEqual(fit['speed'],speed,places=5)
        self.assertAlmostEqual(fit['theta'],theta,places=7)
        self.assertLess(fit['rmse'],1e-5)
        np.testing.assert_array_equal(observed,saved)
        np.testing.assert_allclose(fit['prediction'],observed,atol=1e-5)

    def test_rotation_through_zero_and_scale_uncertainty(self):
        phi = np.radians([-30.,20.,60.])
        directions = np.column_stack((np.cos(phi),np.sin(phi)))
        observed = 145./np.cos(np.radians(359.3)-phi)+np.array([2.,-3.,1.])
        fit = fit_velocity(directions,observed)
        rotation = np.radians(25.)
        rotated = np.column_stack((np.cos(phi+rotation),np.sin(phi+rotation)))
        turned = fit_velocity(rotated,observed)
        scaled = fit_velocity(directions,observed*1.2)
        self.assertAlmostEqual(turned['speed'],fit['speed'],places=4)
        diff=np.arctan2(np.sin(turned['theta']-fit['theta']-rotation),np.cos(turned['theta']-fit['theta']-rotation))
        self.assertAlmostEqual(diff,0,places=6)
        self.assertAlmostEqual(scaled['speed'],fit['speed']*1.2,places=4)
        self.assertAlmostEqual(scaled['speed_std'],fit['speed_std']*1.2,places=4)
        self.assertAlmostEqual(scaled['theta_std_deg'],fit['theta_std_deg'],places=4)
        self.assertAlmostEqual(fit['rmse'],np.linalg.norm(fit['residual'])/np.sqrt(3),places=10)

    def test_sign_correction_does_not_mutate(self):
        phi=np.array([0.,1.,2.]); vel=np.array([-100.,200.,-300.])
        angles,positive=correct_vel_direction(phi,vel)
        np.testing.assert_array_equal(phi,[0.,1.,2.])
        np.testing.assert_array_equal(vel,[-100.,200.,-300.])
        np.testing.assert_array_equal(positive,[100.,200.,300.])
        np.testing.assert_allclose(np.cos(angles)*positive,np.cos(phi)*vel)

    def test_radial_sides(self):
        self.assertAlmostEqual(signed_radial_angle([1,1],[1,0]),45.)
        self.assertAlmostEqual(signed_radial_angle([1,-1],[1,0]),-45.)
        self.assertAlmostEqual(signed_radial_angle([-1,1],[1,0]),135.)
        self.assertAlmostEqual(signed_radial_angle([-1,-1],[1,0]),-135.)

    def test_incompatible_signs_fail_explicitly(self):
        phi=np.radians([0.,120.,240.])
        with self.assertRaisesRegex(ValueError,'No propagation direction'):
            fit_velocity(np.column_stack((np.cos(phi),np.sin(phi))),[100.,100.,100.])

    def test_speed_boundary_is_flagged(self):
        phi=np.radians([-10.,0.,10.])
        fit=fit_velocity(np.column_stack((np.cos(phi),np.sin(phi))),800./np.cos(phi))
        self.assertTrue(fit['speed_boundary'])
        self.assertTrue(fit['interval_boundary68'])


if __name__ == '__main__':
    unittest.main()
