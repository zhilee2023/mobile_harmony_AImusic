import tempfile
import unittest
from pathlib import Path
import numpy as np
from levo_harmony.deployment.dit_layout import export_ffn_layouts, to_btc


class DitLayoutTests(unittest.TestCase):
    def test_layout_axes_preserved(self):
        x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        packed = np.ascontiguousarray(x.transpose(0, 2, 1)[:, :, None, :])
        np.testing.assert_array_equal(to_btc(packed.ravel(), packed.shape, batch=2, frames=3, channels=4), x)
        with self.assertRaises(ValueError):
            to_btc(x.ravel(), (3, 2, 4), batch=2, frames=3, channels=4)

    def test_onnx_variants_match_and_share_weights(self):
        # Synthetic values only exercise the exporter; production probes require real captures.
        rng = np.random.default_rng(71)
        x = rng.normal(size=(2, 3, 4)).astype(np.float32)
        w1, w2 = rng.normal(size=(5, 4)).astype(np.float32), rng.normal(size=(4, 5)).astype(np.float32)
        b1, b2 = rng.normal(size=5).astype(np.float32), rng.normal(size=4).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'candidate'
            report, _ = export_ffn_layouts(out, x, w1, b1, w2, b2)
            self.assertEqual(len(report['cases']), 12)
            self.assertTrue(all(r['relativeL2'] < 1e-5 for r in report['cases']))
            self.assertEqual((out / 'weights.f32').stat().st_size, (w1.size + w2.size + b1.size + b2.size) * 4)
            self.assertEqual([r['transposeNodes'] for r in report['cases']], [0] * 9 + [2] * 3)
            with self.assertRaises(FileExistsError):
                export_ffn_layouts(out, x, w1, b1, w2, b2)


if __name__ == '__main__':
    unittest.main()
