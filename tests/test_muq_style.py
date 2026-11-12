import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('muq_style_tool',Path(__file__).resolve().parents[1]/'tools/extract_muq_text_style.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)

class MuqStyleExportTests(unittest.TestCase):
    def test_style_round_trip_records_text_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'warm.npy';text='温暖的钢琴与木吉他'
            vector=np.zeros(512,dtype=np.float32);vector[2]=1
            result=tool.save_style(vector,text,path,{'modelRevision':'pinned-test'})
            value=np.load(path,allow_pickle=False)
            self.assertEqual(value.shape,(512,));self.assertEqual(value.dtype,np.dtype('<f4'))
            np.testing.assert_array_equal(value,vector)
            self.assertEqual(path.with_suffix('.txt').read_text(encoding='utf-8'),text+'\n')
            self.assertEqual(result['npySha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['modality'],'text')

    def test_rejects_wrong_shape_nonfinite_and_unnormalized(self):
        cases=[np.zeros(256),np.full(512,np.nan),np.zeros(512),np.ones(512)]
        with tempfile.TemporaryDirectory() as folder:
            for i,vector in enumerate(cases):
                path=Path(folder)/f'{i}.npy'
                with self.subTest(i=i),self.assertRaises(ValueError):tool.save_style(vector,'style',path,{})
                self.assertFalse(path.exists())

    def test_preserves_existing_style_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            vector=np.zeros(512);vector[0]=1;path=Path(folder)/'style.npy'
            tool.save_style(vector,'first',path,{})
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):tool.save_style(vector,'second',path,{})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(path.with_suffix('.txt').read_text(encoding='utf-8'),'first\n')

if __name__=='__main__':unittest.main()
