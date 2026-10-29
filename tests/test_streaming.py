import unittest
from dataclasses import replace
import numpy as np
from levo_harmony.streaming import TokenWindows, PCMOverlap


class StreamingTests(unittest.TestCase):
    def render(self, length):
        planner = TokenWindows()
        stitch = PCMOverlap(samples_per_frame=2)
        outputs, chunks = [], []

        def consume(chunk):
            self.assertEqual(chunk.tokens[:chunk.valid_frames],
                             tuple(i % 16384 for i in range(chunk.start_frame, chunk.start_frame+chunk.valid_frames)))
            chunks.append(chunk)
            positions = np.arange(chunk.start_frame*2, (chunk.start_frame+256)*2)
            expected_pcm = np.tile(np.sin(positions * .01)[None, :], (2, 1)).astype(np.float32)
            outputs.append(stitch.accept(chunk, expected_pcm))

        # Irregular token arrival batches exercise incremental delivery.
        for start in range(0, length, 17):
            for chunk in planner.push([i % 16384 for i in range(start, min(start+17, length))]):
                consume(chunk)
        final = planner.finish()
        if final is not None:
            consume(final)
        outputs.append(stitch.finish())
        result = np.concatenate(outputs, axis=1)
        reference = np.tile(np.sin(np.arange(length*2) * .01)[None, :], (2, 1)).astype(np.float32)
        np.testing.assert_allclose(result, reference, atol=2e-7, rtol=2e-7)
        self.assertLessEqual(planner.peak_buffer_frames, 256)
        self.assertEqual(len(planner.buffer), 0)
        self.assertEqual(stitch.tail.shape[1], 0)
        return chunks

    def test_boundaries_and_4000_tokens(self):
        for length in (0, 1, 63, 64, 255, 256, 257, 447, 448, 449, 4000):
            with self.subTest(length=length):
                chunks = self.render(length)
                if length == 4000:
                    self.assertEqual(len(chunks), 21)
                    self.assertEqual(chunks[-1].valid_frames, 160)

    def test_eos_and_invalid_ids(self):
        planner = TokenWindows()
        with self.assertRaises(ValueError):
            list(planner.push([16384]))
        planner.finish()
        with self.assertRaises(ValueError):
            list(planner.push([1]))

    def test_nonfinite_and_out_of_order(self):
        chunk = next(TokenWindows().push([0] * 256))
        pcm = np.zeros((2, 512), dtype=np.float32)
        pcm[0, 0] = np.nan
        with self.assertRaises(ValueError):
            PCMOverlap(samples_per_frame=2).accept(chunk, pcm)
        with self.assertRaises(ValueError):
            PCMOverlap(samples_per_frame=2).accept(replace(chunk, start_frame=192, context_frames=64),
                                                  np.zeros((2,512),dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
