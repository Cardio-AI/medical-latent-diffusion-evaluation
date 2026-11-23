"""Check PNG axis conventions with an asymmetric synthetic input."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'dit'))
from training.data import encoding_transform
from training.sample_ldm import xray_pixels
from src.scripts.sample_dit import _save_decoded


class OrientationTest(unittest.TestCase):
    def test_input_and_legacy_conventions(self):
        pixels = np.array([[0, 13, 40, 90], [150, 200, 240, 255], [23, 71, 99, 121]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.png'
            Image.fromarray(pixels).save(path)
            model_array = encoding_transform('xray')({'image': str(path)})['image'][0].numpy()
            expected = {'input': pixels, 'legacy': pixels[:, ::-1]}
            for convention in ('input', 'legacy'):
                np.testing.assert_array_equal(xray_pixels(model_array, convention), expected[convention])
                out = Path(tmp) / f'{convention}.png'
                _save_decoded(torch.as_tensor(model_array)[None, None], out, 2, None, convention)
                with Image.open(out) as result:
                    np.testing.assert_array_equal(np.asarray(result), expected[convention])


if __name__ == '__main__':
    unittest.main()
