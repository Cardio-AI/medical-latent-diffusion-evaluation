# Medical-image autoencoders and latent diffusion

Code for training a VQ-GAN autoencoder, training and sampling a U-Net or DiT latent diffusion model, and evaluating medical-image cohorts. The examples cover chest X-rays (2D) and lung CT (3D). Data, trained weights and cohort splits are not included.

## Installation

Use Python 3.10 in a fresh environment. For a CPU installation:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple 'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'
python -m pip install -r training/requirements.txt -r dit/requirements.txt -r evaluation/requirements.txt
```

For GPU use, install a compatible PyTorch 2.10.0 and torchvision 0.25.0 pair for the hardware before installing the requirements. Example model configurations are research-scale; adjust them to available hardware. The default autoencoder perceptual loss needs pretrained SqueezeNet weights; see [training/README.md](training/README.md). Load checkpoints and `.pt` files only from trusted sources.

## Train and generate

Supply training and validation CSVs with an `image` column. Paths can be absolute or relative to the CSV. Use PNG images for X-ray and NIfTI volumes for CT. From the repository root, an X-ray workflow is:

```sh
python training/train_ae.py --dataset xray --train-csv data/train.csv --val-csv data/validation.csv \
  --config training/configs/vqgan_xray.yaml --output runs/ae --batch-size 1 --epochs 300 --device cpu
python training/encode_latents.py --dataset xray --ids-csv data/train.csv --split train \
  --config training/configs/vqgan_xray.yaml --checkpoint runs/ae/final_model.pth --output runs/latents/train --device cpu
python training/encode_latents.py --dataset xray --ids-csv data/validation.csv --split validation \
  --config training/configs/vqgan_xray.yaml --checkpoint runs/ae/final_model.pth --output runs/latents/validation --device cpu
python training/train_ldm.py --train-csv runs/latents/train/train.csv --val-csv runs/latents/validation/validation.csv \
  --config training/configs/ldm_xray.yaml --output runs/ldm --batch-size 1 --epochs 300 --disable-flash-attention --device cpu
python training/sample_ldm.py --dataset xray --ae-config training/configs/vqgan_xray.yaml \
  --ae-checkpoint runs/ae/final_model.pth --ldm-config training/configs/ldm_xray.yaml \
  --ldm-checkpoint runs/ldm/final_model.pth --latent-shape 8 256 256 \
  --output runs/samples --steps 300 --disable-flash-attention --device cpu
```

This latent shape assumes 1024 × 1024 input images and a factor-4 autoencoder. For CT configurations, reconstruction, preprocessing, loss options and CPU execution see [training/README.md](training/README.md). Commands accept `--help`.

## Train and sample a DiT

The same encoded-latent CSVs can be used with the DiT. The following example matches the factor-4 X-ray autoencoder and assumes 1024 × 1024 images:

```sh
python dit/src/scripts/train_dit.py --config dit/configs/xray_ds4_best_l1.yaml \
  --output_dir runs/dit --run_name xray --training_ids runs/latents/train/train.csv \
  --validation_ids runs/latents/validation/validation.csv
python dit/src/scripts/sample_dit.py --stage1_cfg training/configs/vqgan_xray.yaml \
  --stage1_ckpt runs/ae/final_model.pth --diff_cfg dit/configs/xray_ds4_best_l1.yaml \
  --diff_ckpt runs/dit/xray/last_checkpoint.pth --latent_shape 256 256 \
  --output_dir runs/dit_samples --n_samples 4 --timesteps 300 --device cpu
```

To train DiT on CPU, set `CUDA_VISIBLE_DEVICES=''`. See [dit/README.md](dit/README.md) for configuration, checkpoint and output details. The DiT model includes non-commercial third-party material: see [dit/NOTICE.md](dit/NOTICE.md) and [dit/LICENSE](dit/LICENSE).

## Evaluate and perturb cohorts

The evaluation tools extract local RadImageNet, Inception or MedicalNet features, compute distribution and paired reconstruction metrics, and apply specified controlled transformations. A short example using supplied feature matrices is:

```sh
python -m evaluation.metrics --real real_features.npy --generated generated_features.npy --output metrics.json
python -m benchmark.run --list
```

For data manifests, local feature weights, extraction commands, condition generation and metric details see [evaluation/README.md](evaluation/README.md). The optional [FRD evaluator](evaluation/frd/README.md) uses its own pinned environment. Pretrained weights, CT-CLIP and memorisation detectors are not bundled.

## Check the installation

`python tests/smoke_cpu.py` runs an offline check using tiny randomly generated 2D and 3D inputs. It exercises the AE, both diffusion models, sampling, reconstruction, metrics and a controlled blur. It disables the adversarial and pretrained perceptual losses; those paths need their own validation for a real run.

## Scope and rights

The included scripts were checked with tiny synthetic 2D and 3D data on CPU. These checks do not establish full-scale training, GPU requirements, exact reproduction of historical results, clinical validity or safe deployment. Generated outputs are research outputs.

Third-party terms and attributions are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and component notices. No repository-wide grant of rights for project-authored code or model weights is implied; the rights holder must specify those terms before public distribution.
