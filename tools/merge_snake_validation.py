"""Record the bounded Oct30 Snake validation without inventing an RTF."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
RUN='snake-shared-pool-20261030-r1'
def read(name):return (REPORTS/name).read_text(encoding='utf-8-sig')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    raw=read(RUN+'.log')
    exits=re.findall(r'^LEVO_PROBE_EXIT (\d+)\s*$',raw,re.M)
    assert exits==['124'], 'This record documents a confirmed timeout only'
    assert not any(line.startswith('LEVO_SNAKE_FUSION ') for line in raw.splitlines())
    stages=[line.removeprefix('LEVO_SNAKE_STAGE ') for line in raw.splitlines() if line.startswith('LEVO_SNAKE_STAGE ')]
    assert stages==['warm-capture-start','warm-capture-complete','immutable-coefficients-ready','fused-window-1-start']
    observation=json.loads(read(RUN+'-observation.json'))
    assert observation['durationCapSeconds']==45 and observation['wallSeconds']>=45
    assert all('Current State: AWAKE' in read(RUN+suffix+'.log') for suffix in ('-before','-after'))
    hashes=read(RUN+'-shas.log')
    binary=ROOT/'native-build/levo_vae_snake_fusion'
    coefficient=ROOT/'models/vae-stream-256/snake-coefficients.json'
    assert sha(binary) in hashes and sha(coefficient) in hashes
    section={
        'asOfUtc':datetime.now(timezone.utc).isoformat(),
        'humanAuthorizationScope':'Continue the bounded Snake fusion validation after manually waking the phone; no new open-ended experiment campaign',
        'status':'timeout','programExitCode':124,'observation':observation,
        'stages':stages,'numericEquivalenceValidated':False,'performanceValidated':False,
        'newRtf':None,'productionAppChanged':False,'bothEndpointPowerStates':'AWAKE',
        'phoneProbeObservedTerminal':True,
        'sourceSha256':{name:sha(ROOT/name) for name in ('native-bench/VaeSnakeFusion.cpp','native-bench/CMakeLists.txt')},
        'verifiedProbeBinarySha256':sha(binary),'verifiedImmutableCoefficientSha256':sha(coefficient),
        'sourceLogSha256':sha(REPORTS/(RUN+'.log')),
        'previousAttempt':{'name':'snake-immutable-20261030-r1','structuredResultPresent':False,
            'programExitCode':None,'transportExitIsNotProgramExit':True,
            'sourceLogSha256':sha(REPORTS/'snake-immutable-20261030-r1.log')},
        'limits':['Timeout is a failure of this callback implementation; it does not measure the speed of a correct fused kernel',
            'Final stage marker precedes input copying; finer markers are needed before attributing the stall solely to the fused kernel or thread pool',
            'Graph workspaces remain allocated; no peak-memory reduction established',
            'Independent kernel correctness and an explicit CPU Execution/graph rewrite remain unvalidated'],
        'nextAction':'Split input-copy and callback/kernel phases; validate the fused kernel independently before integrating a graph rewrite'}
    path=REPORTS/'levo-test-report.json';report=json.loads(path.read_text(encoding='utf8'))
    report['vae_snake_validation_2026_10_30']=section
    report['vae_snake_immutable_followup_2026_10_29']['status']='Original immutable-constant version attempted Oct30 without structured result; shared-pool followup timed out; see vae_snake_validation_2026_10_30'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':section['status'],'programExitCode':124,'newRtf':None,'stages':stages}))
if __name__=='__main__':main()
