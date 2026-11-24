"""Provider-independent compile -> correctness -> device -> graph gate.

Commands are explicit argv arrays in a user-supplied local configuration. The
default baseline does not invoke an agent, spend API tokens or patch MNN.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time


class Budget:
    def __init__(self,compile_attempts=100,device_profiles=100,wall_seconds=3600,agent_tokens=None):
        self.maximum_compile=compile_attempts;self.maximum_device=device_profiles;self.maximum_wall=wall_seconds;self.maximum_tokens=agent_tokens
        self.compile_attempts=0;self.device_profiles=0;self.agent_tokens=0;self.tokens_known=True;self.start=time.monotonic()

    def check(self):
        if time.monotonic()-self.start>=self.maximum_wall:raise RuntimeError('Development wall-time budget exhausted')
        if self.maximum_tokens is not None and (not self.tokens_known or self.agent_tokens>=self.maximum_tokens):raise RuntimeError('Agent token budget exhausted or usage unknown')

    def reserve_compile(self):
        self.check()
        if self.compile_attempts>=self.maximum_compile:raise RuntimeError('Compilation-attempt budget exhausted')
        self.compile_attempts+=1

    def reserve_device(self):
        self.check()
        if self.device_profiles>=self.maximum_device:raise RuntimeError('Device-profile budget exhausted')
        self.device_profiles+=1

    def usage(self):return {'compilationAttempts':self.compile_attempts,'deviceProfilingCount':self.device_profiles,'agentTokens':self.agent_tokens if self.tokens_known else None,'elapsedSeconds':time.monotonic()-self.start}


def execute_candidate(task,commands,workspace,budget,*,epsilon=1e-3,timeout=600):
    workspace=Path(workspace).resolve();workspace.mkdir(parents=True,exist_ok=False)
    task_path=workspace/'task.json';task_path.write_text(json.dumps(task,indent=2),encoding='utf-8')
    if not task.get('signature'):raise ValueError('Candidate task needs fixed device/input/precision signature')
    result={'candidate':task.get('id'),'status':'starting','stages':[],'latencyMs':None,'wholeGraphLatencyMs':None,'deployable':False,'optimizationClaimed':False}
    for stage in ['agent','compile','correctness','device','graph']:
        if stage not in commands:
            if stage in ['compile','correctness','device']:raise ValueError('Required stage missing: '+stage)
            continue
        try:
            budget.check()
            if stage=='compile':budget.reserve_compile()
            if stage in ['correctness','device','graph']:budget.reserve_device()
        except RuntimeError as error:
            result['status']='budget_exhausted';result['error']=str(error);break
        output=workspace/(stage+'-result.json')
        substitutions={'task':str(task_path),'workspace':str(workspace),'result':str(output),'python':sys.executable,'repo':str(Path(__file__).resolve().parents[2])}
        argv=commands[stage]
        if not isinstance(argv,list) or not argv or not all(isinstance(a,str) for a in argv):raise ValueError('Stages use nonempty argv arrays, not shell strings')
        for token,value in substitutions.items():argv=[a.replace('{'+token+'}',value) for a in argv]
        tick=time.monotonic()
        try:
            remaining=max(.001,budget.maximum_wall-(time.monotonic()-budget.start))
            done=subprocess.run(argv,cwd=workspace,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=min(timeout,remaining))
            (workspace/(stage+'.log')).write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
            row={'stage':stage,'seconds':time.monotonic()-tick,'exitCode':done.returncode};result['stages'].append(row)
            if done.returncode:result['status']=stage+'_failed';break
            payload=json.loads(output.read_text(encoding='utf-8'))
            if payload.get('signature')!=task['signature']:raise ValueError('Stage result signature differs from baseline task')
            if stage=='agent':
                tokens=payload.get('totalTokens')
                if not isinstance(tokens,int) or tokens<0:budget.tokens_known=False
                else:budget.agent_tokens+=tokens
                budget.check()
            if stage in ['correctness','graph']:
                error=payload.get('relativeL2')
                if payload.get('passed') is not True or not isinstance(error,(int,float)) or not math.isfinite(error) or error>epsilon or error<0:
                    result['status']=stage+'_failed';break
            if stage=='device':
                if payload.get('passed') is not True:raise ValueError('Device stage did not pass its runtime controls')
                latency=payload.get('medianMs')
                if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:raise ValueError('Missing finite device medianMs')
                result['latencyMs']=latency
            if stage=='graph':
                latency=payload.get('wholeGraphMs')
                if not isinstance(latency,(int,float)) or not math.isfinite(latency) or latency<0:raise ValueError('Missing actual full-graph measurement')
                result['wholeGraphLatencyMs']=latency;result['deployable']=True
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
            result['status']=stage+'_failed';result['error']=str(error);break
    else:result['status']='passed'
    # A kernel-only success cannot become a measured end-to-end speedup.
    if result['status']!='passed':result['latencyMs']=None;result['wholeGraphLatencyMs']=None;result['deployable']=False
    result['budget']=budget.usage();(workspace/'pipeline-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
