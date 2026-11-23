# Third-party notices

- `training/models/` contains adapted [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels) code. Its Apache-2.0 license is reproduced in `training/LICENSE-MONAI`; individual headers retain MONAI and Diffusers / UC Berkeley Team / Hugging Face copyright notices. The VQ-VAE and quantizer source also cite the relevant SynthAnatomy and DeepMind Sonnet implementations.
- Training code builds on work in [Warvito/generative_brain_controlnet](https://github.com/Warvito/generative_brain_controlnet), distributed by that project under Apache-2.0.
- `evaluation/frd/core.py` adapts [frd-score](https://github.com/RichardObi/frd-score) source under Apache-2.0; see the FRD component's README and license text. FRD also uses the separately installed PyRadiomics package under its own BSD terms.
- `dit/src/models/dit.py` includes adapted [Meta DiT](https://github.com/facebookresearch/DiT) material under CC BY-NC 4.0, via [Cardio-AI/VolDiT](https://github.com/Cardio-AI/voldit). `dit/NOTICE.md` and the license files in `dit/` contain the applicable details, including notices for MONAI, Diffusers and timm components.
- Python dependencies, datasets, pretrained model weights and external MedicalNet code have their own terms; they are not relicensed by this repository.

These notices do not grant rights in project-authored contributions or any dataset or model checkpoint.
