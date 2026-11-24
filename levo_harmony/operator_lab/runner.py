import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from .device import Hdc


class DeviceLock:
    def __init__(self,path):self.path=Path(path);self.owned=False
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            with self.path.open('x') as out:json.dump({'pid':os.getpid(),'createdUtc':now()},out)
        except FileExistsError as error:raise RuntimeError('Device campaign lock already exists; check its owner instead of running concurrent inference') from error
        self.owned=True;return self
    def __exit__(self,*_):
        if self.owned:self.path.unlink()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def snapshot(hdc):
    raw=hdc.shell('cat /sys/class/power_supply/Battery/temp; hidumper -s PowerManagerService -a -a')
    lines=raw.splitlines();temp=next((float(s.strip())/10 for s in lines if re.fullmatch(r'\d+',s.strip())),None)
    return {'utc':now(),'batteryC':temp,'awake':'Current State: AWAKE' in raw,
            'lastScreenOff':next((s.strip() for s in lines if 'Last Screen Off:' in s),'')}

def ensure_idle(hdc,package,root):
    pid=hdc.shell('pidof '+package+' || true').strip()
    if not pid:return
    raw=hdc.shell('test -f '+shlex.quote(root+'/last-state.json')+' && cat '+shlex.quote(root+'/last-state.json')+' || echo null')
    try:state=json.loads(raw)
    except ValueError as error:raise RuntimeError('Cannot establish idle state; leave the running App untouched') from error
    if not isinstance(state,dict) or state.get('busy') is not False:
        raise RuntimeError('App is active or its idle state is unknown; no force-stop or overlapping benchmark')

def ensure_no_cli(hdc):
    if hdc.shell('pidof llm-operator-baseline.bin || true').strip():raise RuntimeError('A prior CLI experiment is still active; leave it untouched')

def compare_tensors(path,reference):
    import numpy as np
    a=np.fromfile(reference,dtype='<f4').astype(np.float64);b=np.fromfile(path,dtype='<f4').astype(np.float64)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid regression output dimensions or nonfinite values')
    return {'relativeL2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),'maxAbs':float(np.abs(b-a).max())}

def _run_device(hdc,repo,remote_root,model,options,output,*,transport='app',binary=None,cooling_seconds=0,max_start_battery_c=None,reference=None,epsilon=1e-3):
    if transport not in {'app','cli'}:raise ValueError('Unknown transport')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote_root) or '..' in remote_root.split('/'):raise ValueError('Unsafe remote data root')
    tag=options.get('runTag','')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',tag):raise ValueError('A safe unique runTag is required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    record={'schemaVersion':1,'status':'preparing','transport':transport,'createdUtc':now(),'options':options,'controls':[]}
    target_package='com.zhilee.llmoperatorlab'
    def save():
        (output/'llm-operator-report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    with DeviceLock(Path(repo)/'reports/local'/('operator-device-'+hdc.device+'.lock')):
        ensure_no_cli(hdc)
        source_root='/data/app/el2/100/base/com.zhilee.levostreamlab/haps/entry/files'
        ensure_idle(hdc,'com.zhilee.levostreamlab',source_root)
        ensure_idle(hdc,target_package,'/data/app/el2/100/base/'+target_package+'/haps/entry/files')
        manifest=hdc.json(remote_root+'/cases/'+('large' if model else 'small')+'/manifest.json')
        for file in manifest['files']:
            if not re.fullmatch(r'[A-Za-z0-9_./-]+',file['relative']) or '..' in file['relative'].split('/'):raise ValueError('Unsafe manifest file path')
            if hdc.sha256(remote_root+'/cases/'+('large' if model else 'small')+'/'+file['relative'])!=file['sha256']:raise ValueError('Baseline asset changed after deployment')
        if options.get('decodeSteps',64)>manifest['fixture']['availableDecodeSteps']:raise ValueError('Requested decode exceeds real fixture history')
        record['caseSignature']=manifest['caseSignature'];record['modelManifest']=manifest;record['status']='cooling';save()
        problem={'profile':manifest['profile'],'fixture':manifest['fixture'],'files':manifest['files'],
                 'precision':options['precision'],'threads':options['threads'],'decodeSteps':options['decodeSteps']}
        record['measurementSignature']=hashlib.sha256(json.dumps(problem,sort_keys=True).encode()).hexdigest()
        tick=time.monotonic()
        while True:
            s=snapshot(hdc);record['controls'].append(s);save()
            if time.monotonic()-tick>=cooling_seconds and (max_start_battery_c is None or s['batteryC'] is not None and s['batteryC']<=max_start_battery_c):break
            if time.monotonic()-tick>3600:record['status']='awaiting-temperature';save();raise RuntimeError('Temperature not reached; no inference started')
            time.sleep(5)
        if not s['awake']:record['status']='awaiting-unlock';save();raise RuntimeError('Manually unlock the phone; no inference started')
        before=s;record['coolingSecondsObserved']=time.monotonic()-tick
        ensure_no_cli(hdc)
        ensure_idle(hdc,'com.zhilee.levostreamlab',source_root)
        if transport=='app':
            if remote_root!='/data/app/el2/100/base/'+target_package+'/haps/entry/files':raise ValueError('App uses its own sandbox root')
            ensure_idle(hdc,target_package,remote_root)
            record['hapSha256']=hdc.sha256('/data/app/el1/bundle/public/'+target_package+'/entry.hap')
            hdc.shell('aa force-stop '+target_package)
            command=f'aa start -a EntryAbility -b {target_package} --ps operatorLabTest {options["mode"]} --pi operatorLabModel {model} --pi operatorLabSteps {options["decodeSteps"]} --pi operatorLabRounds {options["rounds"]} --pi operatorLabThreads {options["threads"]} --ps operatorLabPrecision {options["precision"]} --ps operatorLabRunTag {tag}'
            hdc.shell(command);process=None
        else:
            if binary is None:raise ValueError('CLI transport needs a built binary')
            remote_binary='/data/local/tmp/llm-operator-baseline.bin';hdc.send(binary,remote_binary);hdc.shell('chmod 755 '+remote_binary)
            record['binarySha256']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            local_options=output/'options.json';local_options.write_text(json.dumps(options),encoding='utf-8');remote_options=remote_root+'/options-'+tag+'.json';hdc.send(local_options,remote_options)
            log=(output/'native.log').open('w',encoding='utf-8')
            process=subprocess.Popen([hdc.executable,'-t',hdc.device,'shell',remote_binary+' '+shlex.quote(remote_root)+' '+str(model)+' '+shlex.quote(remote_options)],stdout=log,stderr=subprocess.STDOUT)
        record['status']='running';record['inferenceStartedUtc']=now();save();start=time.monotonic();last_print=-30
        result=None
        while time.monotonic()-start<3600:
            s=snapshot(hdc)
            if transport=='app':
                missions=hdc.shell('aa dump -l');m=re.search(r'app name \['+re.escape(target_package)+r'\].*?state #([A-Z]+)',missions,re.S)
                s['foregroundState']=m.group(1) if m else 'UNKNOWN'
            record['controls'].append(s)
            try:
                current=hdc.json(remote_root+'/llm-operator-report.json')
                if current.get('runTag')==tag and current.get('status') in {'completed','cancelled','error'}:result=current
            except (ValueError,RuntimeError):pass
            if result is None and transport=='app':
                try:
                    current=hdc.json(remote_root+'/last-state.json')
                    if current.get('runTag')==tag and current.get('status') in {'cancelled','error'}:
                        result=current
                except (ValueError,RuntimeError):pass
            if result is not None and (process is None or process.poll() is not None):break
            if process is not None and process.poll() is not None:
                record['status']='process-exited-without-result';save();raise RuntimeError('CLI process exited without a matching final report')
            if time.monotonic()-start-last_print>=30:
                print(f'{tag}: {time.monotonic()-start:.0f}s elapsed, battery {s["batteryC"]}C',flush=True);last_print=time.monotonic()-start
            save();time.sleep(5)
        if result is None:
            record['status']='observer-timeout';save();raise RuntimeError('No final result; leave active inference untouched')
        if process is not None:log.close()
        record['native']=result;record['status']=result['status'];record['finishedUtc']=now()
        active=[s for s in record['controls'] if s['utc']>=record['inferenceStartedUtc']]
        record['screenAccepted']=all(s['awake'] and s['lastScreenOff']==before['lastScreenOff'] for s in active)
        record['foregroundAccepted']=transport=='app' and all(s.get('foregroundState')=='FOREGROUND' for s in active)
        record['controlledAppTimingAccepted']=result['status']=='completed' and record['screenAccepted'] and record['foregroundAccepted']
        save()
        if result['status']=='completed':
            hdc.receive(remote_root+'/runs/'+result['runId']+'/reference-logits.f32',output/'reference-logits.f32')
            if model:hdc.receive(remote_root+'/runs/'+result['runId']+'/reference-hidden.f32',output/'reference-hidden.f32')
        if reference is not None and result['status']=='completed':
            prior=json.loads((Path(reference)/'llm-operator-report.json').read_text(encoding='utf-8'));prior_native=prior['native']
            for field in ['profile','fixture','precision','threads','requestedDecodeSteps']:
                if prior_native[field]!=result[field]:raise ValueError('Regression signature mismatch: '+field)
            prior_files={f['relative']:f['sha256'] for f in prior['modelManifest']['files']}
            for f in manifest['files']:
                if f['relative']!='lm/llm.mnn' and prior_files.get(f['relative'])!=f['sha256']:raise ValueError('Weights, vocabulary or fixed inputs changed during operator comparison')
            record['regression']={'logits':compare_tensors(output/'reference-logits.f32',Path(reference)/'reference-logits.f32')}
            if model:record['regression']['hidden']=compare_tensors(output/'reference-hidden.f32',Path(reference)/'reference-hidden.f32')
            record['regressionPassed']=all(v['relativeL2']<=epsilon for v in record['regression'].values())
        save()
        return record

def run_device(hdc,repo,remote_root,model,options,output,**kwargs):
    try:return _run_device(hdc,repo,remote_root,model,options,output,**kwargs)
    except Exception as error:
        path=Path(output)/'llm-operator-report.json'
        if path.is_file():
            record=json.loads(path.read_text(encoding='utf-8'));record['observerError']=str(error)
            if record['status'] in {'running','starting'}:record['status']='observer-failed-active-task-left-untouched'
            path.write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise
