import json
import unittest
from levo_harmony.telemetry import decode_sample, summarize


def row(time=0, stage='prefill', native=0):
    return {'observerElapsedSeconds':time,'rawReadings':['85000','350','4000000','-1500','NA','NA','2000000','2400000','Charging','1'],
            'progress':json.dumps({'stage':stage,'seconds':native})}


class TelemetryTests(unittest.TestCase):
    def test_units_and_bidirectional_current_magnitude(self):
        sample=decode_sample(row())
        self.assertEqual(sample['socNodeC'],85);self.assertEqual(sample['batteryC'],35)
        self.assertEqual(sample['batteryExchangeMagnitudeW'],6)
        micro=row();micro['rawReadings'][3]='-1500000';micro['units']={'batteryCurrent':'uA'}
        self.assertEqual(decode_sample(micro)['batteryExchangeMagnitudeW'],6)

    def test_missing_optional_vendor_nodes_preserve_temperature_and_vi(self):
        sample=decode_sample(row());self.assertEqual(sample['batteryExchangeMagnitudeW'],6)
        missing=row();missing['rawReadings'][3]='NA';missing['rawReadings'][0]='NA'
        sample=decode_sample(missing);self.assertIsNone(sample['batteryExchangeMagnitudeW']);self.assertIsNone(sample['socNodeC']);self.assertEqual(sample['batteryC'],35)

    def test_native_stage_clock_reset_does_not_reset_sensor_time(self):
        result=summarize([row(1,'prefill',1),row(6,'real-phone-music-codes-2250',436),row(11,'chunk-1-dit-0',1)])
        self.assertEqual(result['observedSpanSeconds'],10);self.assertEqual(result['observedIntervalSeconds']['median'],5)
        self.assertEqual(result['sampleCount'],3);self.assertIn('not App',result['powerScope'])

    def test_missing_data_is_not_reported_as_zero_power(self):
        self.assertFalse(summarize([])['available'])
        missing=row();missing['rawReadings']=None;missing['progress']='bad json'
        self.assertIsNone(summarize([missing])['batteryExchangeMagnitudeW'])
        unknown=row();unknown['units']={'batteryCurrent':'unknown'}
        self.assertIsNone(decode_sample(unknown)['batteryExchangeMagnitudeW'])
        nonobject=row();nonobject['progress']='null';nonobject['units']=None
        self.assertIsNone(decode_sample(nonobject)['stage'])


if __name__=='__main__':unittest.main()
