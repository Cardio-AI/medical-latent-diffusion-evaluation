# Autoencoder and U-Net LDM

Use Python 3.10 in a dedicated environment. Install a PyTorch/torchvision build appropriate for the hardware, then install `training/requirements.txt`. For the example U-Net configurations, install a compatible `xformers` build or pass `--disable-flash-attention`. The optional `--mlflow-experiment` switch on AE training additionally requires MLflow and an explicitly configured tracking destination. The pretrained SqueezeNet/LPIPS weights used by the default perceptual loss must be locally available or acquired under their own terms.

Each training or validation CSV has an `image` column with a local PNG path (X-ray) or NIfTI path (CT). Relative paths are resolved relative to the CSV file. Inputs must be single-channel 2D images or 3D volumes; the loader selects the first PNG channel. X-rays are min–max scaled to `[-1,1]`. CTs are clipped to `[-1000,1000]`, min–max scaled to `[-1,1]`, and spatially cropped for training. Paths and data must be provided by the user.

The AE optimizer learning rates come from the configuration. Loss weights are command-line options: `--adv-weight` and `--perceptual-weight` default to 0.01, `--spectral-weight` to 1, and `--entropy-weight` to 0. The `adv_weight` and `perceptual_weight` fields in the example YAMLs are not used by the training command. Use `--adv-start` and `--warmup-epochs` to choose a delayed or gradual adversarial schedule. The quantizer requires the total number of latent positions per batch to be a positive multiple of 32.

Example workflow from the repository root (select the corresponding `*_ct.yaml` configurations and `--dataset ct` for CT):

```sh
python training/train_ae.py --dataset xray --train-csv data/train.csv --val-csv data/validation.csv \
  --config training/configs/vqgan_xray.yaml --output runs/ae --batch-size 1 --epochs 300
python training/encode_latents.py --dataset xray --ids-csv data/train.csv --split train \
  --config training/configs/vqgan_xray.yaml --checkpoint runs/ae/final_model.pth --output runs/latents/train
python training/encode_latents.py --dataset xray --ids-csv data/validation.csv --split validation \
  --config training/configs/vqgan_xray.yaml --checkpoint runs/ae/final_model.pth --output runs/latents/validation
python training/train_ldm.py --train-csv runs/latents/train/train.csv \
  --val-csv runs/latents/validation/validation.csv --config training/configs/ldm_xray.yaml \
  --output runs/ldm --batch-size 1 --epochs 300 --disable-flash-attention
python training/sample_ldm.py --dataset xray --ae-config training/configs/vqgan_xray.yaml \
  --ae-checkpoint runs/ae/final_model.pth --ldm-config training/configs/ldm_xray.yaml \
  --ldm-checkpoint runs/ldm/final_model.pth --latent-shape 8 256 256 \
  --output runs/samples --steps 300 --disable-flash-attention
python training/reconstruct_ae.py --dataset xray --ids-csv data/validation.csv \
  --config training/configs/vqgan_xray.yaml --checkpoint runs/ae/final_model.pth --output runs/reconstructions
```

All commands provide `--help`. Set `--device cpu` for CPU use. The model config and actual image size determine the latent shape; the example above assumes 1024 × 1024 inputs and a factor-4 codec. The default CT encoding size is 512 × 512 × 256; the example factor-4 CT latent shape is `8 128 128 64`. The encoder writes float, channel-first `.nii.gz` latents and a CSV of relative file paths. Set the same `--scale-factor` in LDM training and sampling. The DDPM sampler uses consecutive timesteps; `--steps` must equal the configuration's `num_train_timesteps` (300 in the examples).

Reconstruction exports a `paired.csv` whose `array` column points to `.npz` files containing float32 `reference` and `reconstruction` arrays in the model's `[-1,1]` domain. PNG previews are uint8; CT reconstructions are floating-point NIfTI files in model intensity units. Generated CT samples use a synthetic identity affine and map normalized model outputs to integers in `[-1000,1000]`; they do not encode patient geometry or establish calibrated Hounsfield units. The model examples are starting configurations and do not establish an exact recreation of a historical training run.

For X-ray sampling, `--xray-orientation legacy` (the default) retains the original clockwise rotation used by this sampling lane. This is a horizontal reflection relative to the input-oriented reconstruction preview. Use `--xray-orientation input` to write generated PNGs in the same axis convention as the input PNGs and reconstruction previews. The DiT sampler supports the same choice. Record the convention when comparing cohorts.
