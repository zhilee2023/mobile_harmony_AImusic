"""Read the current phone's observed sensor units, preserving missing values."""
import json
import math
import statistics


POWER_SCOPE = 'Battery-terminal |V*I| magnitude under observed USB/charging conditions; not App, SoC or whole-device power'


def decode_sample(row):
    raw = row.get('rawReadings', [])
    if not isinstance(raw,list): raw=[]
    def value(index, scale=1):
        try:
            number = float(raw[index]) / scale
            return number if math.isfinite(number) else None
        except (ValueError, TypeError, IndexError):
            return None
    try:
        progress = json.loads(row.get('progress', '{}'))
    except (ValueError, TypeError):
        progress = {}
    if not isinstance(progress, dict):
        progress = {}
    units = row.get('units') or {}
    voltage = value(2, 1e6); unit=units.get('batteryCurrent','mA')
    current = value(3,1000 if unit=='uA' else 1) if unit in ('mA','uA') else None
    power = abs(voltage * current / 1000) if voltage is not None and current is not None else None
    return {'observerSeconds': row.get('observerElapsedSeconds'), 'stage': progress.get('stage'),
            'nativePhaseSeconds': progress.get('seconds'), 'socNodeC': value(0, 1000), 'batteryC': value(1, 10),
            'batteryVoltageV': voltage, 'batteryCurrentMa': current, 'batteryExchangeMagnitudeW': power,
            'cpuPolicy2GHz': value(6, 1e6), 'cpuPolicy3GHz': value(7, 1e6),
            'batteryStatus': raw[8] if len(raw)>8 else None, 'usbOnline': raw[9] if len(raw)>9 else None}


def summarize(rows):
    samples = [decode_sample(row) for row in rows]
    def stats(name):
        values = [sample[name] for sample in samples if sample[name] is not None]
        return {'validSamples': len(values), 'min': min(values), 'max': max(values), 'sampleMean': statistics.mean(values)} if values else None
    times = [s['observerSeconds'] for s in samples if isinstance(s['observerSeconds'], (int,float))]
    gaps = [b-a for a,b in zip(times,times[1:]) if b>a]
    return {'available': bool(samples), 'sampleCount': len(samples),
            'observedAppPids': sorted({str(r['observedAppPid']) for r in rows if r.get('observedAppPid')}),
            'runIds': sorted({str(r['runId']) for r in rows if r.get('runId')}),
            'observedSpanSeconds': times[-1]-times[0] if len(times)>1 else 0 if times else None,
            'observedIntervalSeconds': {'min': min(gaps), 'max': max(gaps), 'median': statistics.median(gaps)} if gaps else None,
            'socNodeC': stats('socNodeC'), 'batteryC': stats('batteryC'), 'batteryExchangeMagnitudeW': stats('batteryExchangeMagnitudeW'),
            'usbOnlineObserved': any(s['usbOnline']=='1' for s in samples), 'powerScope': POWER_SCOPE,
            'limits': ['SoC thermal node is a coarse internal reading, not skin/case temperature',
                       'Sample mean is not a time-weighted whole-run average or incremental model power',
                       'Current sysfs units use the verified current phone profile (uV, mA); recheck units after changing devices',
                       'Progress is read from a path; historical progress events have no embedded writer PID',
                       'Observed peaks can miss changes shorter than the sampling interval'], 'samples': samples}
