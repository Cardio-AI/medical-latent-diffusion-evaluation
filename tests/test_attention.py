"""Check that both CPU attention paths attend across tokens, not heads."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dit'))
from src.models.attention import Attention


class AttentionTest(unittest.TestCase):
    def test_reference_and_cross_token_effect(self):
        torch.manual_seed(7)
        layer = Attention(dim=12, num_heads=3, use_flash_attention=False).eval()
        x = torch.randn(2, 5, 12)
        q, k, v = layer.qkv(x).reshape(2, 5, 3, 3, 4).unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * layer.scale
        expected = layer.proj((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(2, 5, 12))

        for fused in (True, False):
            layer.fused_attn = fused
            output = layer(x)
            torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
            changed = x.clone()
            changed[:, 1] += 3
            self.assertGreater(float((layer(changed)[:, 0] - output[:, 0]).detach().abs().max()), 1e-5)

    def test_boolean_mask_matches_reference(self):
        torch.manual_seed(8)
        layer = Attention(dim=8, num_heads=2, use_flash_attention=False).eval()
        x = torch.randn(1, 4, 8)
        mask = torch.tensor([[True, True, False, True]]).expand(4, 4)
        layer.fused_attn = True
        fused = layer(x, mask)
        layer.fused_attn = False
        torch.testing.assert_close(layer(x, mask), fused, atol=1e-6, rtol=1e-5)
        layer.use_flash_attention = True
        with self.assertRaisesRegex(ValueError, 'masks are not supported'):
            layer(x, mask)


if __name__ == '__main__':
    unittest.main()
