"""Concrete Harmony MNN backend-candidate stages for operator_lab.py candidate.

The candidate is linked into a new CLI binary. Its setup/counter ABI proves
dispatch; correctness and graph stages run the entire fixed LLM workload.
This adapter is not original MoKA or a standalone fusion-kernel profiler.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from levo_harmony.operator_lab.device import Hdc
from levo_harmony.operator_lab.runner import run_device


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['compile','correctness','device','graph'],required=True)
    p.add_argument('--task',required=True);p.add_argument('--workspace',required=True);p.add_argument('--result',required=True);a=p.parse_args()
    task=json.loads(Path(a.task).read_text());workspace=Path(a.workspace).resolve();output=Path(a.result).resolve()
    output.relative_to(workspace)
    identifier=task['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',identifier):raise ValueError('Invalid candidate ID')
    reference=Path(task['baselineReport']).resolve();baseline=json.loads((reference/'llm-operator-report.json').read_text())
    if baseline['measurementSignature']!=task['signature']:raise ValueError('Task differs from fixed baseline signature')
    sdk=Path(task.get('sdk',r'C:\Program Files\Huawei\DevEco Studio\sdk\default'))
    cmake=sdk/'openharmony/native/build-tools/cmake/bin/cmake.exe';ninja=sdk/'openharmony/native/build-tools/cmake/bin/ninja.exe'
    mnn=Path(task.get('mnnRoot','C:/cosyvoice-harmony/third_party/MNN')).resolve()
    library=Path(task.get('mnnLibrary','C:/cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a')).resolve()
    build=workspace/'build';binary=build/'llm_operator_baseline'
    candidate=task.get('candidateSource');source=None
    if candidate:
        source=(workspace/candidate).resolve();source.relative_to(workspace)
        if not source.is_file():raise ValueError('Agent candidate source is missing')
    result={'signature':task['signature'],'stage':a.stage,'identityValidationOnly':source is None,'outputScope':'whole_llm_fixed_teacher_workload'}
    if a.stage=='compile':
        args=[str(cmake),'-S',str(ROOT/'native-bench'),'-B',str(build),'-G','Ninja',
              '-DCMAKE_MAKE_PROGRAM='+str(ninja),'-DCMAKE_TOOLCHAIN_FILE='+str(sdk/'hms/native/build/cmake/hmos.toolchain.bisheng.cmake'),
              '-DOHOS_SDK_NATIVE='+str(sdk/'openharmony/native'),'-DMNN_ROOT='+str(mnn),'-DMNN_STATIC_LIBRARY='+str(library),
              '-DOPERATOR_VARIANT_ID='+identifier,'-DOPERATOR_CANDIDATE_SOURCE='+str(source or '')]
        subprocess.run(args,check=True);subprocess.run([str(cmake),'--build',str(build),'--target','llm_operator_baseline','-j','4'],check=True)
        result['binarySha256']=hashlib.sha256(binary.read_bytes()).hexdigest();result['passed']=True
    else:
        if not binary.is_file():raise ValueError('Compile stage did not produce a binary')
        options=dict(baseline['options']);options.update(mode='baseline',rounds=1 if a.stage=='correctness' else 3,
            runTag=identifier+'-'+a.stage,requireCandidateCalls=source is not None)
        trial=workspace/(a.stage+'-device')
        report=run_device(Hdc(task.get('hdc',r'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'),task['device']),ROOT,task['remoteRoot'],task['model'],options,trial,
                          transport='cli',binary=binary,reference=reference,epsilon=task.get('epsilon',1e-3),cooling_seconds=task.get('coolingSeconds',0),max_start_battery_c=task.get('maxStartBatteryC'))
        if report['native']['variantId']!=identifier:raise ValueError('Built candidate identity was not observed')
        errors=report.get('regression',{});result['relativeL2']=max([v['relativeL2'] for v in errors.values()] or [float('inf')])
        result['passed']=report['status']=='completed' and report.get('regressionPassed',False) and report['screenAccepted']
        result['candidateCalls']=report['native']['candidateCalls']
        rows=report['native']['rounds'];warm=rows[1:] or rows;latency=statistics.median(r['prefillMs']+r['decodeMs'] for r in warm)
        if a.stage=='device':result['medianMs']=latency
        if a.stage=='graph':result['wholeGraphMs']=latency
        if not result['passed']:output.write_text(json.dumps(result,indent=2));return 1
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    return 0
if __name__=='__main__':raise SystemExit(main())
