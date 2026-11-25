import unittest

from tools.build_vae_primitive_graph import parse_convolution, parse_deconvolution


class DeconvolutionTraceTests(unittest.TestCase):
    def test_keeps_phase_wall_times_distinct_from_worker_times(self):
        observed = (
            'LEVO_VAE_DECONV 128 128 64 512 2 2 8 12 4 2 1 1024 64 256 3000000 900000 '
            '1000 2000 3000 4000 5000 6000 11000 12000 13000 14000 15000 16000'
        )
        row, = parse_deconvolution(observed, 1)
        self.assertEqual(row['privateScatterBytes'], 1024)
        self.assertEqual(row['scratchBytes'], 1664)
        self.assertEqual(row['stageWallMs'], [3., .9])
        self.assertAlmostEqual(row['workerMs'][1]['PackA'], .012)
        with self.assertRaises(ValueError):
            parse_deconvolution(observed, 2)
        with self.assertRaises(ValueError):
            parse_deconvolution(observed.rsplit(' ', 1)[0], 1)

    def test_conv_keeps_dispatch_and_serial_and_parallel_costs(self):
        row, = parse_convolution(
            'LEVO_VAE_DENSE_CONV 512 128 128 8 1 0 0 1 8 8 8 2 2 3 1 1000000 '
            '1 200 0 2 0 500 2 0 900 0 0 0 0 0 0 0 0 0', 1)
        self.assertTrue(row['dynamicQuant'])
        self.assertTrue(row['quantBeforePacking'])
        self.assertEqual(row['weightBits'], 8)
        self.assertEqual(row['stages']['PackA']['calls'], 2)
        self.assertAlmostEqual(row['stages']['PackA']['slowestWorkerMs'], .0005)
