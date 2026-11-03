param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [ValidateRange(2,256)][int]$Frames=256,
  [ValidateSet('outputs','threads','threads-confirm')][string]$Suite='outputs',
  [string]$Hdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
)
# Bounded, separate CLI processes. No app restart, settings changes or model writes.
$ErrorActionPreference='Stop'
$probeRoot=Split-Path -Parent $PSScriptRoot
$probeLocal=Join-Path $probeRoot "reports/local/hidden-speed/$RunId"
if(Test-Path -LiteralPath $probeLocal){throw 'Keep existing results; choose a new RunId'}
New-Item -ItemType Directory -Path $probeLocal -Force | Out-Null
$probeRemote="/data/local/tmp/levo-hidden-speed-$RunId"
$probeModels='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files/chain-comparison/small'
$probeBinary=Join-Path $probeRoot 'native-build/levo_hidden_condition_bench'
if(!(Test-Path -LiteralPath $probeBinary)){throw 'Build native target levo_hidden_condition_bench first'}
$probeRecord=[ordered]@{runId=$RunId;suite=$Suite;scope='Phone CLI, fixed-token student LLM only; no music or whole-chain RTF';binarySha256=(Get-FileHash -LiteralPath $probeBinary).Hash.ToLower();frames=$Frames;complete=$false;cases=@()}
function Save-Probe { [IO.File]::WriteAllText((Join-Path $probeLocal 'run.json'),($probeRecord|ConvertTo-Json -Depth 50),[Text.UTF8Encoding]::new($false)) }
function Probe-State { return ((& $Hdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n") }
function Probe-Temperature { return (((& $Hdc -t $Device shell 'cat /sys/class/power_supply/Battery/temp') -join '').Trim()) }
try {
  if((Probe-State) -notmatch 'Current State: AWAKE'){throw 'Keep phone awake before starting'}
  & $Hdc -t $Device shell "mkdir -p $probeRemote"
  & $Hdc -t $Device file send $probeBinary "$probeRemote/bench"
  if($LASTEXITCODE -ne 0){throw 'Benchmark transfer failed'}
  & $Hdc -t $Device shell "chmod 755 $probeRemote/bench"
  $probeCases=@(@('logits',4),@('buffer',4),@('allocated',4),@('buffer',6),@('buffer',4),@('logits',4))
  if($Suite -eq 'threads'){$probeCases=@(@('buffer',3),@('buffer',4),@('buffer',2),@('buffer',4))}
  if($Suite -eq 'threads-confirm'){$probeCases=@(@('buffer',3),@('buffer',4),@('buffer',3),@('buffer',4))}
  $probeIndex=0
  foreach($probeCase in $probeCases){
    $probeIndex++;$probeMode=$probeCase[0];$probeThreads=$probeCase[1]
    $probeName="$probeIndex-$probeMode-t$probeThreads";$probeWork="$probeRemote/$probeName"
    $probeBefore=Probe-State
    if($probeBefore -notmatch 'Current State: AWAKE'){throw 'Phone slept; completed cases preserved, do not automatically rerun'}
    $probeTempBefore=Probe-Temperature
    Write-Output "Start $probeName"
    & $Hdc -t $Device shell "mkdir -p $probeWork"
    & $Hdc -t $Device shell "$probeRemote/bench $probeModels $probeWork $probeMode $probeThreads $Frames $probeWork/result.json >$probeWork/console.log 2>&1"
    $probeAfter=Probe-State;$probeTempAfter=Probe-Temperature
    $probeCaseLocal=Join-Path $probeLocal $probeName;New-Item -ItemType Directory -Path $probeCaseLocal | Out-Null
    & $Hdc -t $Device file recv "$probeWork/result.json" (Join-Path $probeCaseLocal 'result.json') | Out-Null
    & $Hdc -t $Device file recv "$probeWork/console.log" (Join-Path $probeCaseLocal 'console.log') | Out-Null
    $probeResult=Get-Content -Raw -LiteralPath (Join-Path $probeCaseLocal 'result.json') | ConvertFrom-Json
    $probeOffBefore=[regex]::Match($probeBefore,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $probeOffAfter=[regex]::Match($probeAfter,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $probeRecord.cases+=@{name=$probeName;displayStable=($probeAfter -match 'Current State: AWAKE') -and $probeOffBefore -eq $probeOffAfter;temperatureBeforeRaw=$probeTempBefore;temperatureAfterRaw=$probeTempAfter;result=$probeResult}
    Save-Probe
    if(!$probeResult.success){throw "Probe failed: $($probeResult.error)"}
    & $Hdc -t $Device file recv "$probeWork/logits.f32" (Join-Path $probeCaseLocal 'logits.f32') | Out-Null
    if($probeMode -ne 'logits'){ & $Hdc -t $Device file recv "$probeWork/hidden.f32" (Join-Path $probeCaseLocal 'hidden.f32') | Out-Null }
    Write-Output "$probeName prefill=$($probeResult.rounds[1].prefillSeconds) decode/s=$($probeResult.rounds[1].decodeForwardsPerSecond) handoff=$($probeResult.rounds[1].hiddenHandoffSeconds)"
    if(!$probeRecord.cases[-1].displayStable){throw 'Phone screen state changed; preserve this case as uncontrolled'}
  }
  $probeRecord.complete=$true;Save-Probe
} catch { $probeRecord.error=$_.Exception.Message;Save-Probe;throw }
