param([Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9]+$')][string]$Device,
      [string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio')
$ErrorActionPreference='Stop';$project=Split-Path -Parent $PSScriptRoot
$hap=Join-Path $project 'llm-operator-app/entry/build/default/outputs/default/entry-default-signed.hap'
if(!(Test-Path -LiteralPath $hap)){throw 'Build with its own DevEco signing profile for com.zhilee.llmoperatorlab; do not reuse another package Profile'}
$hdc=Join-Path $DevEcoRoot 'sdk/default/openharmony/toolchains/hdc.exe'
$running=(& $hdc -t $Device shell 'pidof com.zhilee.llmoperatorlab || true') -join ''
if($running.Trim()){
 $state=((& $hdc -t $Device shell 'cat /data/app/el2/100/base/com.zhilee.llmoperatorlab/haps/entry/files/last-state.json') -join "`n") | ConvertFrom-Json
 if($null -eq $state -or $state.busy -cne $false){throw 'Existing experiment is active or idle state is unknown; leave it untouched'}
}
& $hdc -t $Device file send $hap /data/local/tmp/llm-operator-lab.hap
if($LASTEXITCODE -ne 0){throw 'HAP transfer failed'}
$status=(& $hdc -t $Device shell 'bm install -p /data/local/tmp/llm-operator-lab.hap -r') -join "`n"
Write-Output $status;if($status -notmatch 'successfully'){throw 'Independent App installation failed'}
& $hdc -t $Device shell 'aa start -a EntryAbility -b com.zhilee.llmoperatorlab'
if($LASTEXITCODE -ne 0){throw 'App launch failed'}
Write-Output 'Old applications were not replaced. Deploy cases into the new App sandbox with prepare_llm_operator_lab.py.'
