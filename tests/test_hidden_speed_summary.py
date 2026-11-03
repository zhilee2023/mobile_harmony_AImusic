"""A faster hidden handoff must preserve even the prefill-derived first frame."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.summarize_hidden_speed import summarize


class HiddenSpeedSummaryTests(unittest.TestCase):
    def test_missing_or_changed_first_hidden_is_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for name in ('1-logits-t4','2-buffer-t4','3-allocated-t4'):
                (folder/name).mkdir()
            logits=np.zeros(256*16385,dtype=np.float32)
            hidden=np.zeros(256*1024,dtype=np.float32)
            logits.tofile(folder/'1-logits-t4/logits.f32')
            for name in ('2-buffer-t4','3-allocated-t4'):
                shutil.copyfile(folder/'1-logits-t4/logits.f32',folder/name/'logits.f32')
                hidden.tofile(folder/name/'hidden.f32')
            run={'complete':True,'frames':256,'runId':'test','suite':'outputs','binarySha256':'test',
                 'cases':[{'name':n,'displayStable':True,'result':{'success':True,'mode':m}}
                          for n,m in [('1-logits-t4','logits'),('2-buffer-t4','buffer'),('3-allocated-t4','allocated')]]}
            path=folder/'run.json';path.write_text(json.dumps(run))
            self.assertFalse(summarize([path],[])['wholeChainRtfMeasured'])
            # A lost/misaligned h0 must fail, even if every later frame agrees.
            with (folder/'3-allocated-t4/hidden.f32').open('r+b') as stream:
                stream.write(np.float32(1).tobytes())
            with self.assertRaisesRegex(ValueError,'changed fixed-token'):
                summarize([path],[])
            (folder/'3-allocated-t4/hidden.f32').unlink()
            with self.assertRaisesRegex(ValueError,'missing its numerical snapshot'):
                summarize([path],[])

    def test_incomplete_suite_cannot_be_ranked(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'1-logits-t4').mkdir();(folder/'2-buffer-t4').mkdir()
            np.zeros(256*16385,dtype=np.float32).tofile(folder/'1-logits-t4/logits.f32')
            np.zeros(256*1024,dtype=np.float32).tofile(folder/'2-buffer-t4/hidden.f32')
            path=folder/'run.json';path.write_text(json.dumps({'complete':False,'frames':256}))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                summarize([path],[])


if __name__=='__main__':
    unittest.main()
