"""Optional-FRD checks with newly generated images only."""

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk
from PIL import Image

from evaluation.frd import __main__ as frd_cli
from evaluation.frd import core


class ColorHandlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    def test_rgba_conversion_and_legacy_scalar_read(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "synthetic-rgba.png"
            Image.new("RGBA", (1024, 1024), (90, 170, 40, 255)).save(image_path)
            corrected, corrected_repair = core.read_image(
                image_path, dataset="ChestXray", generated=False, color_handling="grayscale"
            )
            legacy, legacy_repair = core.read_image(
                image_path, dataset="ChestXray", generated=False, color_handling="legacy-as-run"
            )
            old_reader = sitk.ReadImage(str(image_path), outputPixelType=sitk.sitkFloat32)
            self.assertEqual(corrected.GetPixel(0, 0), 131.0)
            self.assertEqual(corrected_repair, "multichannel_to_l_pillow")
            self.assertEqual(legacy.GetPixel(0, 0), old_reader.GetPixel(0, 0))
            self.assertEqual(legacy_repair, "none")
            self.assertNotEqual(legacy.GetPixel(0, 0), corrected.GetPixel(0, 0))
            self.assertEqual(corrected.GetSize(), (1024, 1024))
            self.assertEqual(corrected.GetSpacing(), (1.0, 1.0))

    def test_grayscale_does_not_quantize_scalar_16bit_png(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "synthetic-16bit.png"
            Image.fromarray(np.full((1024, 1024), 4095, dtype=np.uint16)).save(image_path)
            image, repair = core.read_image(image_path, dataset="ChestXray", generated=False)
            self.assertEqual(image.GetPixel(0, 0), 4095.0)
            self.assertEqual(repair, "none")

    def test_cli_reports_color_mode_and_repairs(self):
        class FakeExtractor:
            def execute(self, image, mask):
                return {"original_firstorder_Mean": float(sitk.GetArrayViewFromImage(image)[0, 1, 1])}

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image_path = folder / "synthetic-rgba.png"
            Image.new("RGBA", (1024, 1024), (90, 170, 40, 255)).save(image_path)
            manifests = []
            for role in ("real", "generated"):
                manifest = folder / f"{role}.csv"
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["image"])
                    writer.writeheader()
                    writer.writerows([{"image": image_path.name}, {"image": image_path.name}])
                manifests.append(manifest)
            for mode, repair in (("grayscale", "multichannel_to_l_pillow"),
                                 ("legacy-as-run", "none")):
                output = folder / f"{mode}.json"
                args = ["evaluation.frd", "--dataset", "chestxray", "--real-csv", str(manifests[0]),
                        "--generated-csv", str(manifests[1]), "--output", str(output),
                        "--color-handling", mode]
                with mock.patch.object(sys, "argv", args), \
                        mock.patch.object(core, "get_extractor", return_value=FakeExtractor()), \
                        mock.patch("evaluation.frd.aggregation.compute_pair", return_value={}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    frd_cli.main()
                result = json.loads(output.read_text())
                self.assertEqual(result["color_handling"], mode)
                self.assertEqual(result["real_repair_counts"], {repair: 2})
                self.assertEqual(result["generated_repair_counts"], {repair: 2})


if __name__ == "__main__":
    unittest.main()
