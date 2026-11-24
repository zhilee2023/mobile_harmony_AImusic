import unittest
from levo_harmony.operator_lab.candidates import mulsilu_tile_source

def source_fixture():
    includes=['CPUBinary.hpp','CPUBinaryInt8.hpp','CPUBackend.hpp','compute/CommonOptFunction.h','compute/ConvOpt.h','BinaryUtils.hpp']
    return '\n'.join('#include "'+x+'"' for x in includes)+'''\nnamespace MNN {
auto i1 = input1->host<int8_t>();
mSilu((float*)out, (float*)inp1, realSize);
                mMul((float*)out, (float*)out, (float*)inp, realSize, -1);
}
'''

class CandidateRecipeTests(unittest.TestCase):
    def test_changed_upstream_arithmetic_is_rejected(self):
        with self.assertRaises(ValueError):mulsilu_tile_source(source_fixture().replace('realSize, -1','realSize, 0'),4096)
    def test_ambiguous_duplicate_patch_point_is_rejected(self):
        with self.assertRaises(ValueError):mulsilu_tile_source(source_fixture()+source_fixture(),4096)
    def test_only_registered_tiles_and_immutable_input(self):
        original=source_fixture();saved=original
        with self.assertRaises(ValueError):mulsilu_tile_source(original,32)
        generated=mulsilu_tile_source(original,4096)
        self.assertEqual(original,saved);self.assertNotEqual(generated,original)
        self.assertEqual(generated,mulsilu_tile_source(original,4096))
if __name__=='__main__':unittest.main()
