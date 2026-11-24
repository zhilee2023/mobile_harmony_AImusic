"""Bounded read-only delegation through the user's installed Claude CLI."""
import json
from pathlib import Path
import shutil
import subprocess
import os


def decode_output(raw):
    try:return raw.decode('utf-8-sig')
    except UnicodeDecodeError:return raw.decode('gb18030')


def invoke(argv,prompt,*,timeout,cwd):
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=cwd)
    try:
        stdout,stderr=process.communicate(prompt,timeout=timeout)
    except subprocess.TimeoutExpired as error:
        # Killing cmd.exe alone leaves its native Claude child and pipe alive.
        if os.name=='nt':subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15)
        else:process.kill()
        process.communicate(timeout=15)
        raise RuntimeError('Claude task timed out; owned process tree stopped; token usage unknown') from error
    return subprocess.CompletedProcess(argv,process.returncode,stdout,stderr)


def review(context,output,*,model='haiku',timeout=150,budget_usd=1,executable=None):
    if model not in {'haiku','sonnet','opus'} or not 0<timeout<=300 or not 0<budget_usd<=5:
        raise ValueError('Bounded model alias, timeout and CLI budget required')
    output=Path(output)
    if output.exists():raise ValueError('Keep previous delegation result')
    cli=executable or shutil.which('claude.cmd') or shutil.which('claude')
    if not cli:raise RuntimeError('Install/configure Claude CLI locally; no credentials are embedded')
    prompt=('Read-only delegated operator review. Use only the supplied context, no tools, no file access or subagents. '
            'Return concise JSON with priorityCandidates and caveats. Do not invent measurements, speedups or hardware bandwidth. '
            'Graph callback times are not standalone latency; logical IO is not measured DRAM. Do not infer all arithmetic is FP32 from coreBytes4.\n'
            +json.dumps(context,ensure_ascii=False,allow_nan=False))
    done=invoke([cli,'-p','--model',model,'--safe-mode','--strict-mcp-config','--tools=',
                         '--output-format','json','--no-session-persistence','--max-budget-usd',str(budget_usd)],
                        prompt.encode('utf-8'),timeout=timeout,cwd=output.parent)
    text=decode_output(done.stdout)
    if done.returncode:raise RuntimeError('Claude CLI failed; credentials are never copied into the report')
    value=json.loads(text)
    if value.get('is_error') or value.get('subtype')!='success':raise RuntimeError('Claude CLI did not complete successfully')
    record=dict(cliResponse=value,requestedAlias=model,scope='One read-only delegated review, not generated/validated kernel code',
                cliBudgetUsd=budget_usd,billingScope='CLI-reported usage; third-party model billing and hard-limit enforcement are not independently verified')
    with output.open('x',encoding='utf-8') as out:json.dump(record,out,ensure_ascii=False,indent=2)
    return record
