import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import merge_stream_runtime_optimizations as merge


class RuntimeReportTests(unittest.TestCase):
    def test_background_completion_cannot_be_published_as_controlled_rtf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case = root/'case'; case.mkdir()
            raw = {'status':'completed', 'wholeRtf':2., 'foregroundGapObserved':True,
                   'generation':{'runId':'one', 'generatedSamples':48000}}
            (case/'run.json').write_text(json.dumps(raw), encoding='utf-8')
            with patch.object(merge, 'ROOT', root): result = merge.duration_case(Path('case'))
            self.assertNotIn('wholeRtf', result)
            self.assertEqual(result['observedWholeRtf'], 2.)
            self.assertFalse(result['controlledTimingAccepted'])
            self.assertEqual(result['status'], 'completed-with-foreground-gap')

    def test_playback_acceptance_uses_only_the_measured_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case = root/'case'; case.mkdir()
            raw = {'status':'completed', 'wholeRtf':2., 'controlledTimingAccepted':True,
                   'generation':{'runId':'one', 'generatedSamples':48000}}
            (case/'run.json').write_text(json.dumps(raw), encoding='utf-8')
            events = [{'runId':'other', 'kind':'playback-error', 'playedMs':99999},
                      {'runId':'one', 'kind':'playback', 'playedMs':1000}]
            (case/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events), encoding='utf-8')
            with patch.object(merge, 'ROOT', root): result = merge.duration_case(Path('case'))
            self.assertEqual(result['wholeRtf'], 2.)
            self.assertEqual(result['playback']['maxPlayedMs'], 1000)
            self.assertEqual(result['playback']['errorCount'], 0)


if __name__ == '__main__':
    unittest.main()
