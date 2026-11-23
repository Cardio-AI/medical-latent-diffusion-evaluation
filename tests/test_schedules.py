"""Reject unsupported strided DDPM schedules."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dit'))
from src.models.ddpmscheduler import DDPMScheduler
from src.models.ddimscheduler import DDIMScheduler


class ScheduleTest(unittest.TestCase):
    def test_ddpm_requires_consecutive_transitions(self):
        ddpm = DDPMScheduler(num_train_timesteps=10)
        ddpm.set_timesteps(10)
        self.assertEqual(ddpm.timesteps.tolist(), list(range(9, -1, -1)))
        for steps in (0, 5, 11):
            with self.assertRaisesRegex(ValueError, 'consecutive timesteps'):
                ddpm.set_timesteps(steps)

    def test_ddim_allows_shorter_schedules(self):
        ddim = DDIMScheduler(num_train_timesteps=10)
        ddim.set_timesteps(5)
        self.assertEqual(ddim.timesteps.tolist(), [8, 6, 4, 2, 0])
        for steps in (0, 11):
            with self.assertRaises(ValueError):
                ddim.set_timesteps(steps)


if __name__ == '__main__':
    unittest.main()
