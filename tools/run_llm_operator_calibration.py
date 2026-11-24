"""Run isolated reference operator and nominal-bandwidth probes on an idle phone."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import DeviceLock,ensure_idle,ensure_no_cli,snapshot

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--root',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--binary',default=str(ROOT/'native-build/llm_operator_calibration'))
    p.add_argument('--hdc',default=r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe');a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',a.tag) or not re.fullmatch('/[A-Za-z0-9_./-]+',a.root) or '..' in a.root.split('/'):p.error('Unsafe run tag or remote root')
    out=ROOT/'reports/local'/a.tag;out.mkdir(exist_ok=False);hdc=Hdc(a.hdc,a.device)
    remote=a.root+'/calibration-'+a.tag+'.json'
    with DeviceLock(ROOT/'reports/local'/('operator-device-'+a.device+'.lock')):
        ensure_no_cli(hdc)
        if hdc.shell('pidof llm-operator-calibration.bin || true').strip():raise RuntimeError('Prior calibration still active; leave it untouched')
        for package in ['com.zhilee.levostreamlab','com.zhilee.llmoperatorlab']:ensure_idle(hdc,package,'/data/app/el2/100/base/'+package+'/haps/entry/files')
        before=snapshot(hdc)
        if not before['awake']:raise RuntimeError('Manually unlock phone; no inference started')
        for name in ['small','large']:
            m=hdc.json(a.root+'/cases/'+name+'/manifest.json')
            if hdc.sha256(a.root+'/cases/'+name+'/fixtures/prefix.f32')!=m['fixture']['prefixSha256']:raise ValueError('Fixture was modified')
        binary=Path(a.binary).resolve();hdc.send(binary,'/data/local/tmp/llm-operator-calibration.bin');hdc.shell('chmod 755 /data/local/tmp/llm-operator-calibration.bin')
        with (out/'native.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([hdc.executable,'-t',a.device,'shell','/data/local/tmp/llm-operator-calibration.bin '+a.root+' '+remote+' 4'],stdout=log,stderr=subprocess.STDOUT)
            controls=[before];tick=time.monotonic()
            while process.poll() is None:
                time.sleep(2);controls.append(snapshot(hdc))
                if time.monotonic()-tick>180:raise RuntimeError('Observer timeout; active process left untouched')
        hdc.receive(remote,out/'native.json');n=json.loads((out/'native.json').read_text())
        record=dict(native=n,controls=controls,screenAccepted=all(x['awake'] and x['lastScreenOff']==before['lastScreenOff'] for x in controls),transport='cli',binarySha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        (out/'llm-operator-report.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(dict(success=n['success'],rows=len(n['rows']),screenAccepted=record['screenAccepted'],rssPeakMiB=n['rssPeakMiB']),indent=2))
        return 0 if n['success'] and record['screenAccepted'] else 1
if __name__=='__main__':raise SystemExit(main())
