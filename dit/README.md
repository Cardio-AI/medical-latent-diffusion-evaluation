# Diffusion transformer

The DiT component contains adapted Meta DiT material under CC BY-NC 4.0. Read [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE) before use. The root README gives an example with encoded X-ray latents. The `configs/` directory contains example 2D and 3D configurations for each compression factor. Choose a configuration whose spatial size and channels match the encoded latents and autoencoder; the sample command takes spatial dimensions only, with channels read from the DiT configuration.

The encoder in `training/encode_latents.py` writes channel-first NIfTI latents and CSVs whose paths are relative to the CSV. DiT training can use those CSVs directly. The configurations enable EMA and save periodic and final checkpoints. `sample_dit.py` uses EMA weights when a full checkpoint contains them; `final_model.pth` contains raw weights. Only load checkpoints or `.pt` latent files from a trusted source.

To use CPU for training set `CUDA_VISIBLE_DEVICES=''` before launching. For sampling, pass `--device cpu`. If flash attention is unavailable, set `dit.params.flash_attention: false` in a copy of the configuration. The example configurations were designed for large research workloads; tiny CPU checks do not establish that they fit a particular GPU.

The xformers attention path retains the original training implementation, which does not apply attention-weight dropout. The standard PyTorch path applies the configured dropout during training, so the two backends need not produce the same training trajectory. Set `flash_attention: false` to use the standard path consistently. This release has not been tested on GPU. Attention masks are not supported by the xformers path.

The DDPM sampler uses consecutive reverse transitions and requires `--timesteps` to equal the configuration's `num_train_timesteps`. To use fewer inference steps, select `--scheduler ddim`.

For 3D outputs, the sampler maps model values in `[-1,1]` to an integer `[-1000,1000]` range and uses an identity affine unless `--reference_nii` is supplied. These are synthetic output conventions; they do not establish patient geometry or calibrated Hounsfield units. The input transforms use per-image intensity scaling, so this mapping is not generally an inverse of input preprocessing.

For 2D output, `--xray-orientation legacy` retains the original clockwise-rotation export; `--xray-orientation input` instead matches the input-oriented PNG convention used for reconstruction previews. The legacy version is reflected horizontally relative to that convention. Use and record the same choice across cohorts.
