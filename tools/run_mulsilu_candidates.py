"""Compile and replay a small candidate pool without changing pinned MNN or Apps."""
import argparse,hashlib,json,random,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import DeviceLock,ensure_idle,ensure_no_cli,snapshot
from levo_harmony.operator_lab.candidates import mulsilu_tile_source

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-report',required=True);p.add_argument('--device',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--sdk',default='C:/Program Files/Huawei/DevEco Studio/sdk/default')
    p.add_argument('--mnn-root',default='C:/cosyvoice-harmony/third_party/MNN')
    p.add_argument('--mnn-library',default='C:/cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a');a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',a.tag):p.error('Unsafe run tag')
    work=ROOT/'reports/local'/a.tag;work.mkdir(exist_ok=False)
    capture=Path(a.capture_report).resolve().parent;record=json.loads(Path(a.capture_report).read_text(encoding='utf-8'))
    fixture=record['native']['operators']['capture']
    if set(fixture['phases'])!={'prefill','decode'}:raise ValueError('Need both actual capture phases')
    sdk=Path(a.sdk);cmake=sdk/'openharmony/native/build-tools/cmake/bin/cmake.exe'
    mnn=Path(a.mnn_root);library=Path(a.mnn_library);original=(mnn/'source/backend/cpu/CPUBinary.cpp').read_text(encoding='utf-8')
    variants={'stock':ROOT/'native-build/llm_operator_mulsilu'}
    for tile in [0,256,1024,4096,8192]:
        name='tile'+str(tile);folder=work/name;folder.mkdir();src=folder/'candidate.cpp';src.write_text(mulsilu_tile_source(original,tile),encoding='utf-8')
        build=folder/'build';args=[str(cmake),'-S',str(ROOT/'native-bench'),'-B',str(build),'-G','Ninja',
            '-DCMAKE_MAKE_PROGRAM='+str(sdk/'openharmony/native/build-tools/cmake/bin/ninja.exe'),
            '-DCMAKE_TOOLCHAIN_FILE='+str(sdk/'hms/native/build/cmake/hmos.toolchain.bisheng.cmake'),
            '-DOHOS_SDK_NATIVE='+str(sdk/'openharmony/native'),'-DMNN_ROOT='+str(mnn),'-DMNN_STATIC_LIBRARY='+str(library),
            '-DOPERATOR_CANDIDATE_SOURCE='+str(src),'-DOPERATOR_VARIANT_ID='+name]
        with (folder/'build.log').open('w',encoding='utf-8') as log:
            subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
            subprocess.run([str(cmake),'--build',str(build),'--target','llm_operator_mulsilu','llm_operator_baseline','-j','4'],check=True,stdout=log,stderr=subprocess.STDOUT)
        variants[name]=build/'llm_operator_mulsilu';print('Compiled',name,flush=True)
    hdc=Hdc(str(sdk/'openharmony/toolchains/hdc.exe'),a.device);remote='/data/local/tmp/'+a.tag
    report=dict(scope='Isolated existing MUL_SILU cache tiles; actual activations; not global fusion or whole-LLM speedup',captureSignature=record['caseSignature'],variants={},trials=[])
    with DeviceLock(ROOT/'reports/local'/('operator-device-'+a.device+'.lock')):
        ensure_no_cli(hdc)
        for package in ['com.zhilee.levostreamlab','com.zhilee.llmoperatorlab']:ensure_idle(hdc,package,'/data/app/el2/100/base/'+package+'/haps/entry/files')
        if not snapshot(hdc)['awake']:raise RuntimeError('Phone asleep; no candidate inference started')
        hdc.shell('mkdir -p '+remote);f=work/'fixtures.json';f.write_text(json.dumps(fixture),encoding='utf-8');hdc.send(f,remote+'/fixtures.json')
        for phase in fixture['phases'].values():
            for item in phase.values():
                name=item['file']
                if not re.fullmatch('[A-Za-z0-9_-]+\\.f32',name):raise ValueError('Unsafe private capture filename')
                file=capture/'captures'/name
                if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Capture changed')
                hdc.send(file,remote+'/'+name)
        for name,binary in variants.items():
            hdc.send(binary,remote+'/'+name+'.bin');hdc.shell('chmod 755 '+remote+'/'+name+'.bin')
            report['variants'][name]=dict(binarySha256=hashlib.sha256(binary.read_bytes()).hexdigest(),sourceSha256=None if name=='stock' else hashlib.sha256((work/name/'candidate.cpp').read_bytes()).hexdigest())
        for repeat in range(3):
            order=list(variants);random.Random(42+repeat).shuffle(order)
            for name in order:
                before=snapshot(hdc)
                if not before['awake']:raise RuntimeError('Phone slept; prior trials retained')
                path=remote+f'/{repeat}-{name}.json';hdc.shell(remote+'/'+name+'.bin '+remote+' '+path+' 4',timeout=90)
                after=snapshot(hdc);r=hdc.json(path)
                if r['status']!='completed' or r['variantId']!=('baseline' if name=='stock' else name) or (name!='stock' and r['candidateCalls']<=0):raise RuntimeError('Candidate identity, dispatch or numerical failure')
                report['trials'].append(dict(repeat=repeat,variant=name,native=r,before=before,after=after,screenAccepted=after['awake'] and before['lastScreenOff']==after['lastScreenOff']))
                (work/'llm-operator-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(repeat,name,flush=True)
if __name__=='__main__':main()
