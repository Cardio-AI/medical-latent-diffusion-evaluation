"""Run tiny offline 2D and 3D end-to-end checks with synthetic data only."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def run(*args):
    command = [sys.executable, *map(str, args)]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=ENV, check=True)


def write_csv(path, image):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image'])
        writer.writeheader()
        writer.writerow({'image': image})


def config_files(work, dims):
    ae = {
        'stage1': {'base_lr': .0001, 'disc_lr': .0001, 'params': {
            'spatial_dims': dims, 'in_channels': 1, 'out_channels': 1,
            'num_channels': [8, 16], 'num_res_channels': [8, 16],
            'num_res_layers': 1, 'downsample_parameters': [[2, 4, 1, 1]] * 2,
            'upsample_parameters': [[2, 4, 1, 1, 0]] * 2,
            'num_embeddings': 16, 'embedding_dim': 4, 'ddp_sync': False}},
        'discriminator': {'params': {'spatial_dims': dims, 'num_channels': 8,
            'num_layers_d': 1, 'in_channels': 1, 'out_channels': 1}},
        'perceptual_network': {'params': {'spatial_dims': dims, 'network_type': 'squeeze'}},
    }
    ldm = {'ldm': {'base_lr': .0001, 'params': {
        'spatial_dims': dims, 'in_channels': 4, 'out_channels': 4,
        'num_channels': [8, 16], 'attention_levels': [False, False],
        'num_head_channels': [4, 4], 'num_res_blocks': 1, 'norm_num_groups': 4,
        'use_flash_attention': False, 'with_conditioning': False},
        'scheduler': {'num_train_timesteps': 2, 'prediction_type': 'v_prediction', 'schedule': 'cosine'}}}
    dit = {'spatial_dims': dims, 'dit': {'params': {'input_size': [8] * dims,
        'patch_size': 2, 'in_channels': 4, 'hidden_size': 24, 'depth': 1,
        'num_heads': 4, 'mlp_ratio': 2., 'class_dropout_prob': 0.,
        'num_classes': 0, 'learn_sigma': False, 'flash_attention': False},
        'scheduler': {'num_train_timesteps': 2, 'prediction_type': 'v_prediction', 'schedule': 'cosine'}},
        'training': {'n_epochs': 1, 'eval_freq': 1, 'checkpoint_freq': 1, 'batch_size': 1,
        'gradient_accumulation_steps': 1, 'num_workers': 0, 'roi_size': [32] * dims,
        'use_precomputed_latents': True, 'preload_latents': False,
        'scale_factor': 1., 'use_ema': False}, 'optim': {'lr': .0001, 'lr_gamma': .999}}
    paths = []
    for name, contents in [('ae', ae), ('ldm', ldm), ('dit', dit)]:
        path = work / f'{name}.yaml'
        path.write_text(yaml.safe_dump(contents), encoding='utf-8')
        paths.append(path)
    return paths


def modality(base, dataset, dims, rng):
    work = base / dataset
    work.mkdir()
    for split in ('train', 'validation'):
        if dims == 2:
            name = f'{split}.png'
            Image.fromarray(rng.integers(0, 256, size=(32, 32), dtype=np.uint8)).save(work/name)
        else:
            name = f'{split}.nii.gz'
            nib.save(nib.Nifti1Image(rng.uniform(-900, 500, size=(32, 32, 32)).astype('float32'), np.eye(4)), work/name)
        write_csv(work/f'{split}.csv', name)
    ae, ldm, dit = config_files(work, dims)
    crop = ['--ct-roi', '32', '32', '32', '--ct-val-roi', '32', '32', '32'] if dims == 3 else []
    size = ['--ct-spatial-size', '32', '32', '32'] if dims == 3 else []
    run('training/train_ae.py', '--dataset', dataset, '--train-csv', work/'train.csv',
        '--val-csv', work/'validation.csv', '--config', ae, '--output', work/'ae',
        '--batch-size', '1', '--epochs', '1', '--eval-every', '1', '--workers', '0',
        '--device', 'cpu', '--adv-weight', '0', '--perceptual-weight', '0', *crop)
    for split in ('train', 'validation'):
        run('training/encode_latents.py', '--dataset', dataset, '--ids-csv', work/f'{split}.csv',
            '--split', split, '--config', ae, '--checkpoint', work/'ae/final_model.pth',
            '--output', work/'latents'/split, '--workers', '0', '--device', 'cpu', *size)
    train_csv = work/'latents/train/train.csv'
    val_csv = work/'latents/validation/validation.csv'
    run('training/train_ldm.py', '--train-csv', train_csv, '--val-csv', val_csv,
        '--config', ldm, '--output', work/'ldm', '--batch-size', '1', '--epochs', '1',
        '--eval-every', '1', '--workers', '0', '--device', 'cpu', '--disable-flash-attention')
    run('training/sample_ldm.py', '--dataset', dataset, '--ae-config', ae,
        '--ae-checkpoint', work/'ae/final_model.pth', '--ldm-config', ldm,
        '--ldm-checkpoint', work/'ldm/final_model.pth', '--latent-shape', '4', *(['8']*dims),
        '--output', work/'ldm_samples', '--count', '1', '--steps', '2', '--device', 'cpu', '--disable-flash-attention')
    run('training/reconstruct_ae.py', '--dataset', dataset, '--ids-csv', work/'validation.csv',
        '--config', ae, '--checkpoint', work/'ae/final_model.pth', '--output', work/'reconstruction',
        '--workers', '0', '--device', 'cpu', *size)
    tag = 'chestxray' if dims == 2 else 'luna'
    run('-m', 'evaluation.paired', '--dataset', tag, '--manifest', work/'reconstruction/paired.csv',
        '--output', work/'paired_metrics.csv', '--ssim')
    with (work/'paired_metrics.csv').open(newline='') as stream:
        row = next(csv.DictReader(stream))
        assert all(np.isfinite(float(row[k])) for k in ('L1', 'PSNR', 'NMSE', 'SSIM'))
    run('dit/src/scripts/train_dit.py', '--config', dit, '--output_dir', work/'dit_runs', '--run_name', 'tiny',
        '--training_ids', train_csv, '--validation_ids', val_csv)
    run('dit/src/scripts/sample_dit.py', '--stage1_cfg', ae, '--stage1_ckpt', work/'ae/final_model.pth',
        '--diff_cfg', dit, '--diff_ckpt', work/'dit_runs/tiny/final_model.pth', '--latent_shape', *(['8']*dims),
        '--output_dir', work/'dit_samples', '--n_samples', '1', '--timesteps', '2', '--scheduler', 'ddim', '--device', 'cpu')
    extension = '.png' if dims == 2 else '.nii.gz'
    for directory in ('ldm_samples', 'dit_samples'):
        assert (work/directory/f'sample_0000{extension}').is_file()
    run('-m', 'benchmark.run', '--dataset', tag, '--condition',
        f'{tag}__local_degradation__gaussian_blur__s2_1p0', '--manifest', work/'train.csv',
        '--output-dir', work/'corrupted', *(['--allow-unit-spacing'] if dims == 2 else []))
    with np.load(work/'corrupted/000000.npz', allow_pickle=False) as data:
        assert data['data'].ndim == dims and np.isfinite(data['data']).all()
    print(f'{dataset}: AE, LDM, DiT, paired metrics and controlled blur passed', flush=True)


def main():
    run('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py')
    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory(prefix='media-smoke-') as temp:
        base = Path(temp)
        for name, dims in [('xray', 2), ('ct', 3)]:
            modality(base, name, dims, rng)
        np.save(base/'real.npy', rng.normal(size=(12, 8)).astype('float32'))
        np.save(base/'generated.npy', rng.normal(size=(12, 8)).astype('float32'))
        run('-m', 'evaluation.metrics', '--real', base/'real.npy', '--generated', base/'generated.npy',
            '--output', base/'metrics.json')
        results = json.loads((base/'metrics.json').read_text())
        assert all(np.isfinite(results[k]) for k in ('fid', 'kid', 'mmd', 'c2st_absolute_balanced_accuracy_gap'))
        print('Synthetic distribution metrics passed', flush=True)
    print('All CPU smoke checks passed; no patient data or pretrained weights were used.', flush=True)


if __name__ == '__main__':
    main()
