# Evaluation and controlled benchmark

Run commands from the repository root. Use Python 3.10 or later and install a compatible PyTorch 2.10.0 / torchvision 0.25.0 pair for your hardware, then `python -m pip install -r evaluation/requirements.txt`. No checkpoint or image data are included.

## Feature and distribution metrics

Make a CSV with an `image` column containing PNG paths for `chestxray`, or NIfTI paths for `luna`; relative paths are resolved from the CSV. Files produced by the benchmark command below can also be read directly. Extract two cohorts with the same feature model, then compare their `.npy` feature matrices:

```sh
python -m evaluation.features --dataset chestxray --extractor radimagenet --manifest real.csv --weights /path/to/radimagenet_resnet50.pt --output real_features.npy
python -m evaluation.features --dataset chestxray --extractor radimagenet --manifest generated.csv --weights /path/to/radimagenet_resnet50.pt --output generated_features.npy
python -m evaluation.metrics --real real_features.npy --generated generated_features.npy --output metrics.json
```

The extractor options are `radimagenet` or `inception` for 2D, and `medicalnet` for 3D. MedicalNet additionally requires `--medicalnet-repo /path/to/MedicalNet-models`, a trusted local checkout of [Warvito/MedicalNet-models](https://github.com/Warvito/MedicalNet-models) at commit `a76edd352306a7fb73f1c66b02d86fc8585e25da`:

```sh
git clone https://github.com/Warvito/MedicalNet-models.git /path/to/MedicalNet-models
git -C /path/to/MedicalNet-models checkout a76edd352306a7fb73f1c66b02d86fc8585e25da
```

The expected `hubconf.py` SHA-256 is `1709645c653b42dc5c16b58c398d51fc21bf889f4a656e2a2987db6f8a8da5e4`; `medicalnet_models/models/resnet.py` is `26e487efeddabe0f8859b6bbb4cbef68bfaf34774420751a42683cb62b828979`. An alternative checkout requires `--allow-other-medicalnet-source`. The extractor uses a local checkpoint and never downloads one. Expected study checkpoint hashes are listed in `evaluation/features.py`; an alternative requires `--allow-other-weights` and changes the feature space. The user is responsible for obtaining and checking the applicable checkpoint terms.

Defaults use the study's 2D direct 224/299 resize or 3D 512 × 512 × 256 crop followed by 112³ resize. The MedicalNet crop and resize can be changed explicitly; doing so changes the evaluation protocol. The FID, KID, MMD, C2ST, PRDC and WAD-DIV command expects two finite 2D arrays of features in the **same** space, at least five rows each. It uses five-fold C2ST, calibrated PRDC and WAD-DIV with k = 3; pairwise computations may require substantial memory for large cohorts. The JSON includes the PRDC calibration status, threshold, candidate table and warnings so a threshold failure or fallback is visible.

## Paired reconstruction metrics

The training reconstruction command writes `paired.csv` with an `array` column. Each referenced `.npz` holds float `reference` and `reconstruction` arrays in the same model intensity domain. Compute per-image L1, PSNR and NMSE, plus optional SSIM:

```sh
python -m evaluation.paired --dataset chestxray --manifest /path/to/reconstructions/paired.csv --output paired_metrics.csv --ssim
```

Use `--dataset luna` for 3D. `--ms-ssim` requires sufficiently large images. `--lpips-squeezenet-weights /path/to/squeezenet1_1.pth` additionally computes LPIPS using the locally provided ImageNet SqueezeNet 1.1 trunk and the LPIPS package's calibration; no weights are downloaded. 3D LPIPS evaluates evenly spaced slices from all three axes.

## Controlled transformations

`python -m benchmark.run --list` prints the condition IDs. The source CSV has `image` and optional unique `source_id` columns. For radiographic Gaussian blur or high-frequency attenuation, also provide `original_width`, `original_height`, `original_spacing_x_mm`, `original_spacing_y_mm`; `--allow-unit-spacing` explicitly assumes 1 mm/pixel when those fields are unavailable. CT uses NIfTI header geometry. Conditions requiring an anatomical mask need a `mask` column with a matching NIfTI mask.

```sh
python -m benchmark.run --dataset chestxray --condition chestxray__local_degradation__gaussian_blur__s2_1p0 --manifest xray.csv --output-dir corrupted_xray
python -m benchmark.run --dataset luna --condition luna__local_degradation__gaussian_blur__s2_1p0 --manifest ct.csv --output-dir corrupted_ct
python -m evaluation.features --dataset chestxray --extractor radimagenet --manifest corrupted_xray/manifest.csv --weights /path/to/radimagenet_resnet50.pt --output corrupted_features.npy
```

Outputs are float `.npz` arrays and a relative-path manifest. `--condition near_copy` transforms donor images for the near-copy lane. `python -m benchmark.composition --help` builds a deterministic source/donor position mapping for cohort-composition conditions. Use `python -m benchmark.compose_features --help` with feature matrices and their matching ID manifests to build the mapped cohort matrix, then compare it with reference features using `evaluation.metrics`. For near-copy injection, extract features from the near-copy donor arrays. This package does not include a memorisation detector, patient data, pretrained weights or private cohort splits.

Run data-free core checks with `python -m unittest discover -s evaluation/tests -v`.
