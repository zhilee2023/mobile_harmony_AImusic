import unittest

from tools.vae_storage_graph import storage_peak


class VaeStorageGraphTest(unittest.TestCase):
    def test_alias_keeps_storage_until_last_consumer_and_includes_scratch(self):
        graph = {
            'operators': [{'id': i, 'local': {'op': str(i), 'executorMeasurement': {'scratchBytes': 13 if i == 2 else 0}}}
                          for i in range(3)],
            'tensors': [
                {'tensor': 0, 'storageBytes': 8, 'firstObserved': 0, 'consumers': [1], 'isGraphOutput': False},
                {'tensor': 1, 'storageBytes': 8, 'firstObserved': 1, 'consumers': [2], 'isGraphOutput': False},
            ],
            'copyAliases': [{'name': 'view', 'sourceTensor': 0, 'destinationTensor': 1}],
        }
        self.assertEqual([r['liveTensorBytes'] for r in storage_peak(graph, [])['timeline']], [8, 16, 8])
        aliased = storage_peak(graph, ['view'])
        self.assertEqual([r['liveTensorBytes'] for r in aliased['timeline']], [8, 8, 8])
        self.assertEqual(aliased['peak']['modeledBytes'], 21)
