import tempfile
from pathlib import Path
import unittest
import torch
from levo_harmony.codec import ExportedMixedCodec
from levo_harmony.contracts import CodecContract


class StubDit(torch.nn.Module):
    def forward(self, codes, latent, timestep, tail):
        return torch.zeros_like(latent)


class StubVae(torch.nn.Module):
    def forward(self, latent):
        return latent[:, :2].repeat_interleave(1920, dim=-1)


class BadVae(torch.nn.Module):
    def forward(self, latent):
        return latent[:, :2]


class CodecTests(unittest.TestCase):
    def test_export_contract_roundtrip_and_shape_rejection(self):
        # These are identity/stub modules, deliberately not music checkpoints.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.jit.script(StubDit()).save(str(root / 'dit.pt'))
            torch.jit.script(StubVae()).save(str(root / 'vae.pt'))
            codec = ExportedMixedCodec(root / 'dit.pt', root / 'vae.pt', CodecContract('test-only'))
            x = torch.randn(1, 64, 3)
            pcm, latent = codec.decode_chunk(torch.tensor([[1, 2, 3]]), x, x[:, :, :1], 2)
            self.assertEqual(pcm.shape, (1, 2, 5760))
            torch.testing.assert_close(latent, x)
            torch.jit.script(BadVae()).save(str(root / 'bad.pt'))
            bad = ExportedMixedCodec(root / 'dit.pt', root / 'bad.pt', CodecContract('test-only'))
            with self.assertRaisesRegex(RuntimeError, 'VAE'):
                bad.decode_chunk(torch.tensor([[1, 2, 3]]), x, x[:, :, :1], 2)


if __name__ == '__main__': unittest.main()
