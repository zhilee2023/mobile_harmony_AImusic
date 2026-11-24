"""HDC transport with explicit argument arrays; no app credentials are copied."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


class Hdc:
    def __init__(self, executable, device):
        if not re.fullmatch(r'[A-Za-z0-9]+',device):raise ValueError('Invalid device identifier')
        self.executable=str(Path(executable).resolve());self.device=device

    def call(self,*args,timeout=120):
        result=subprocess.run([self.executable,'-t',self.device,*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        if result.returncode:raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def shell(self,command,timeout=120):return self.call('shell',command,timeout=timeout)
    def send(self,source,destination):return self.call('file','send',str(Path(source).resolve()),destination)
    def receive(self,source,destination):return self.call('file','recv',source,str(Path(destination).resolve()))
    def json(self,path):return json.loads(self.shell('cat '+shlex.quote(path)))
    def sha256(self,path):
        result=self.shell('sha256sum '+shlex.quote(path),timeout=240).strip().split()
        if not result or not re.fullmatch('[a-fA-F0-9]{64}',result[0]):raise RuntimeError('Device did not return a valid SHA256')
        return result[0].lower()


def latest_fixture(root,model):
    selected=[];expected='tanhai.npy' if model else 'muq-text-warm-pop.npy'
    for file in (Path(root)/'reports/local').glob('*/run.json'):
        try:
            record=json.loads(file.read_text(encoding='utf-8-sig'));n=record['generation']
            if record['status']!='completed' or n['model']!=model or Path(n.get('npy','')).name!=expected:continue
            prefix=next((file.parent/'artifacts').rglob('prefix.f32'));codes=prefix.parent/'codes.json'
            if prefix.stat().st_size!=513*(2560 if model else 1024)*4 or not codes.is_file():continue
            selected.append((record.get('finishedUtc',''),prefix.parent))
        except (KeyError,ValueError,StopIteration,OSError):continue
    if not selected:raise ValueError('No completed actual style-matched prefix/code fixture; supply or generate one in the existing Stream App')
    return max(selected,key=lambda p:p[0])[1]


def prepare_case(hdc,repo,remote_root,model,fixture,*,mnn_sha256):
    """Copy exact existing phone LM files into a new immutable case directory.

    Private app copies have independent ownership. Never rewrite source bundles,
    sign with another app's profile, silently crop a new head or manufacture data.
    """
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote_root) or '..' in remote_root.split('/'):
        raise ValueError('Unsafe remote data root')
    case='large' if model else 'small';dest=remote_root+'/cases/'+case
    prior=json.loads(hdc.shell('test -f '+shlex.quote(dest+'/manifest.json')+' && cat '+shlex.quote(dest+'/manifest.json')+' || echo null'))
    if prior is not None:
        for f in prior['files']:
            if hdc.sha256(dest+'/'+f['relative'])!=f['sha256']:raise ValueError('Existing case was modified; preserve it and use a new destination')
        return prior
    old='/data/app/el2/100/base/com.zhilee.levostreamlab/haps/entry/files'
    source=old+('/model-bundles/larkstudio-base-v2/lm' if model else '/experimental/music-head-audio-r2')
    if hdc.shell('test -e '+shlex.quote(dest)+' && echo EXISTS || true').strip():raise ValueError('Incomplete prior case exists; preserve it and choose another data root')
    hdc.shell('mkdir -p '+shlex.quote(dest+'/lm')+' '+shlex.quote(dest+'/fixtures'))
    files=[]
    for name in ['llm.mnn','llm.mnn.weight','llm_config.json','tokenizer.mtok','embeddings_bf16.bin']:
        original=source+'/'+name;target=dest+'/lm/'+name;digest=hdc.sha256(original)
        size=int(hdc.shell('stat -c %s '+shlex.quote(original)).strip())
        hdc.shell('cp '+shlex.quote(original)+' '+shlex.quote(target)+' && chmod 644 '+shlex.quote(target),timeout=240)
        if hdc.sha256(target)!=digest:raise RuntimeError('Model copy hash mismatch')
        files.append({'relative':'lm/'+name,'bytes':size,'sha256':digest})
    fixture=Path(fixture)
    for name in ['prefix.f32','codes.json']:
        p=fixture/name;raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();hdc.send(p,dest+'/fixtures/'+name)
        if hdc.sha256(dest+'/fixtures/'+name)!=digest:raise RuntimeError('Fixture copy hash mismatch')
        files.append({'relative':'fixtures/'+name,'bytes':len(raw),'sha256':digest})
    codes=json.loads((fixture/'codes.json').read_text());width=2560 if model else 1024
    if (fixture/'prefix.f32').stat().st_size!=513*width*4 or not codes or not all(isinstance(c,int) and 0<=c<16384 for c in codes):raise ValueError('Incorrect real fixture dimensions/IDs')
    profile={'layers':36 if model else 28,'hiddenWidth':width,'classes':16385,'audioInputBase':1 if model else 151830,'bodyWeightBits':4,'headWeightBits':8,'groupSize':64,'dynamicOption':0 if model else 2,'defaultPrecision':'high','defaultThreads':4}
    fixture_meta={'positions':513,'availableDecodeSteps':len(codes),'mode':'actual-fixed-teacher-code-history','styleFile':'tanhai.npy' if model else 'muq-text-warm-pop.npy','prefixSha256':files[-2]['sha256'],'codesSha256':files[-1]['sha256']}
    manifest={'schemaVersion':1,'modelId':'larkstudio-base-v2' if model else 'larkstudio-small-0.6b-v1','profile':profile,'fixture':fixture_meta,'files':files,'mnnArchiveSha256':mnn_sha256,'origin':'Existing trained-declared music bundle, not stock language-Qwen or random weights'}
    manifest['caseSignature']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    local=Path(repo)/'reports/local/llm-operator-lab-design'/f'{case}-manifest.json';local.parent.mkdir(parents=True,exist_ok=True);local.write_text(json.dumps(manifest,indent=2))
    hdc.send(local,dest+'/manifest.json');hdc.shell('chmod -R a+rX '+shlex.quote(dest))
    return manifest
