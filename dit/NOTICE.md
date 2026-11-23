# Third-party notices for DiT

`src/models/dit.py` is adapted from [Meta DiT](https://github.com/facebookresearch/DiT), Copyright Meta Platforms, Inc. and affiliates, under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the full upstream text is in `LICENSE`. It was adapted for 3D in [Cardio-AI/VolDiT](https://github.com/Cardio-AI/voldit) and further modified here to support 2D radiographs. The upstream non-commercial restriction remains applicable to the adapted model. No exact Meta DiT source revision was established.

The VolDiT source used here is based on revision `76c7063a1d51884dfeb7cd51c63d4191b5358839`; selected files were modified for this release. VolDiT provides an Apache-2.0 notice, reproduced in `LICENSE-APACHE-2.0`. The model's Meta-derived portions remain subject to their own terms above.

`src/models/attention.py` and the scheduler files include adaptations of [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels), Copyright MONAI Consortium, and [Diffusers](https://github.com/huggingface/diffusers), Copyright 2022 UC Berkeley Team and The HuggingFace Team, Apache-2.0. The attention code also follows [timm](https://github.com/huggingface/pytorch-image-models), Copyright 2019 Ross Wightman, Apache-2.0. `vqvae.py` is from VolDiT and uses MONAI components. Individual source headers remain applicable.

The DiT timestep implementation references [OpenAI GLIDE](https://github.com/openai/glide-text2im), Copyright 2021 OpenAI, MIT License; the full notice is in `LICENSE-GLIDE-MIT`. The positional-embedding implementation references [Meta MAE](https://github.com/facebookresearch/mae), under CC BY-NC 4.0.

These notices describe third-party material; they do not grant rights in other authors' contributions or in model weights and datasets.
