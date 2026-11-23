"""Data-free checks of the retained perturbations and distribution metrics."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from benchmark.conditions import all_conditions
from benchmark.transforms import Sample, apply_condition, load_luna
from evaluation.metrics import distribution_stack


ROOT = Path(__file__).resolve().parents[2]


class EvaluationCoreTests(unittest.TestCase):
    def test_source_identity_and_condition_registry(self):
        vendor = ROOT / "evaluation/vendor/medgen_eval_lab/medgen_eval_lab/metrics"
        for path, expected in [
            (vendor / "distribution/kernels.py", "10f24b7d7e8c8919f2560b123b55df44ba33036f2e05382f3fc96d21d08e8493"),
            (vendor / "prdc/toy.py", "ff96095fcfbf86b8bc0ddf67fd547bf7688faddded089c9b48c290d05b0c18e7"),
            (ROOT / "evaluation/wad_div_cache.py", "d9b2cb3d8e78f61208f2071d355c6f2722e6bb1e7a02202bb1df2c9510f69378"),
        ]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        conditions = all_conditions()
        self.assertEqual(len(conditions), 233)
        self.assertEqual(len({c.condition_id for c in conditions}), 233)

    def test_physical_gaussian_and_high_frequency(self):
        data = np.zeros((33, 33), dtype=np.float32)
        data[16, 16] = 1
        sample = Sample(data, "chestxray", "sample", "", "", (0.5, 1.0), {})
        values = all_conditions()
        blur = next(c for c in values if c.dataset == "chestxray" and c.subtype == "gaussian_blur" and c.severity_value == 1.0)
        actual = apply_condition(sample, blur)
        expected = gaussian_filter(data, sigma=(2.0, 1.0), mode="nearest")
        np.testing.assert_array_equal(actual, expected)
        low = next(c for c in values if c.dataset == "chestxray" and c.subtype == "high_frequency_attenuation" and c.severity_value == 0.0)
        lowpass = gaussian_filter(data, sigma=(4.0, 2.0), mode="nearest")
        np.testing.assert_array_equal(apply_condition(sample, low), lowpass)

    def test_ct_shape_and_spacing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "volume.nii.gz"
            affine = np.diag([0.7, 0.8, 1.25, 1.0])
            nib.save(nib.Nifti1Image(np.zeros((8, 9, 10), dtype=np.float32), affine), path)
            sample = load_luna(str(path), "sample")
            self.assertEqual(sample.data.shape, (10, 8, 9))
            np.testing.assert_allclose(sample.spacing, (1.25, 0.7, 0.8))

    def test_identical_feature_cohorts(self):
        x = np.random.default_rng(1).normal(size=(20, 4)).astype(np.float32)
        result = distribution_stack(x, x)
        self.assertAlmostEqual(result["fid"], 0.0, places=8)
        self.assertAlmostEqual(result["wad_div_empirical"]["wad_div"], 0.0, places=8)
        self.assertEqual(result["prdc"]["precision"], 1.0)
        self.assertEqual(result["prdc"]["recall"], 1.0)

    def test_metrics_cli_emits_prdc_calibration_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.save(path / "real.npy", np.random.default_rng(3).normal(size=(5, 4)))
            np.save(path / "generated.npy", np.random.default_rng(4).normal(size=(5, 4)))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(
                [sys.executable, "-m", "evaluation.metrics", "--real", str(path / "real.npy"),
                 "--generated", str(path / "generated.npy"), "--output", str(path / "result.json")],
                cwd=ROOT, env=env, check=True, capture_output=True, text=True,
            )
            result = json.loads((path / "result.json").read_text())
            self.assertEqual(result["prdc_calibration"]["status"], "threshold-not-crossed")
            self.assertEqual(result["prdc_calibration"]["selected_k"], result["prdc_k"])
            self.assertEqual(result["prdc_k"], 1)
            self.assertTrue(result["prdc_calibration"]["candidate_table"])
            self.assertTrue(any("did not both cross threshold" in warning
                                for warning in result["prdc_warnings"]))


if __name__ == "__main__":
    unittest.main()
