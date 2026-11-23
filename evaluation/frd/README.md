# Optional Fréchet Radiomics Distance

FRD uses a separate Python 3.10 environment because its pinned image-processing dependencies differ from the main environment. Create and activate a new environment, then install the pinned dependencies:

```sh
python3.10 -m venv .venv-frd
. .venv-frd/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r evaluation/frd/requirements.txt
```

PyRadiomics is built from its pinned Git revision and requires a C/C++ build toolchain. The user supplies all images; no pretrained weights or patient data are included. The optional dependencies are [frd-score](https://github.com/RichardObi/frd-score) commit `bf05c71fc44e461ab123bd273e58390b90f9cf54` (Apache-2.0) and [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) commit `8ed579383b44806651c463d5e691f3b2b57522ab` (BSD-3-Clause), with SimpleITK, PyWavelets and OpenCV installed as separately licensed dependencies.

`core.py` adapts parts of frd-score by Richard Osuala, Nick Konz and Preeti Verma with the study-specific image handling described below. The upstream material is distributed under the [Apache-2.0 license](LICENSE-APACHE-2.0); `aggregation.py` uses the same upstream implementation for normalization and distance.

Run from the repository root with two CSV files, each containing an `image` column. Relative image paths resolve from the corresponding CSV. Each cohort needs at least two images:

```sh
python -m evaluation.frd --dataset chestxray --real-csv real.csv --generated-csv generated.csv --output frd.json
# or: --dataset luna for 3D CT
```

The default `paper` profile uses the retained study geometry. Chest X-rays must be 1024 × 1024 PNGs with unit archive spacing and identity direction. Its default `--color-handling grayscale` converts multichannel sources to 8-bit Pillow grayscale before extraction; `--color-handling legacy-as-run` retains the earlier scalar-reader behavior, which can give different values. The JSON records the selected behavior and per-cohort repair counts. CT real scans must have 0.7 × 0.7 × 1.25 mm spacing and identity direction. Generated CT must be 512 × 512 × 256 with unit archive spacing and SimpleITK direction `(-1, 0, 0, 0, -1, 0, 0, 0, 1)`; both are clipped to [-1000, 1000] in the study's HU-like numerical convention, centered/cropped or padded to 512 × 512 × 256, and evaluated with 0.7 × 0.7 × 1.25 mm spacing. Generated values are not guaranteed to be calibrated HU. The exact checks and conversions are in `core.py`. Each image uses a whole-image mask with one background corner voxel.

The optional `--profile upstream` delegates image handling to the pinned frd-score implementation and is useful for other geometries or its test fixtures. Results produced with that profile are not automatically comparable to the paper's preprocessing. Full-field 256³ ablation outputs need their separately specified trilinear resizing and are not handled by this command.

The pinned FRDv1 configuration enables Original, Laplacian-of-Gaussian (sigma 2, 3, 4 and 5) and wavelet image types; first-order, GLCM (selected features), GLRLM, GLSZM and NGTDM feature classes; per-image intensity normalization with scale 100; B-spline resampling to [2, 2, 0] for 2D and [2, 2, 2] for 3D; and intensity bin width 5. The 2D profile forces 2D extraction on a singleton third axis; PyRadiomics may skip filters unsuitable for that image geometry. Consult the pinned upstream configurations for all settings.

Aggregation drops rows containing NaNs, normalizes both cohorts using the real cohort, drops resulting nonfinite columns, and reports `frd_paper_log_distance = 0.5 × log(squared Fréchet distance)`. A zero or nonfinite squared distance is rejected. At runtime the optional path checks only the pinned `frd.py` and two upstream YAML configuration hashes; `--allow-other-frd-source` deliberately overrides that check. The installation recipe pins PyRadiomics and other dependencies, but their installed bytes are not hash-verified by this command.

Matching historical results also requires the original cohort members, order, image bytes, archived geometry, software versions and numerical environment. They are not distributed here, so the provided command does not by itself establish exact reproduction of any study result. Extraction and covariance calculations can be resource intensive, especially for full-size CT.
