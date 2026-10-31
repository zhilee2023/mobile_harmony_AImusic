param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [Parameter(Mandatory=$true)][ValidateSet('kv:persistent','kv:persistent-high','kv:persistent-kv8','kv:persistent-kv4','kv:full-reprefill','kv:window64','chain:eager','chain:overlap','chain:compute')][string]$Mode,
  [ValidateSet('dit','small-dit')][string]$Dit='small-dit',
  [ValidateSet(4,6)][int]$Threads=4
)
$ErrorActionPreference='Stop'
$levoRoot=Split-Path -Parent $PSScriptRoot
$levoHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$levoRemote='/data/local/tmp/levo-speed-20261031'
$levoJson=Join-Path $levoRoot "reports\scheduling-$RunId.json"
$levoLog=Join-Path $levoRoot "reports\scheduling-$RunId.log"
if((Test-Path -LiteralPath $levoJson) -or (Test-Path -LiteralPath $levoLog)){throw 'Preserve previous result; select a new RunId'}
$levoBinaryHash=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'native-build\levo_scheduling_bench') -Algorithm SHA256).Hash.ToLower()
$levoChainHash=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'native-bench\ChainBench.cpp') -Algorithm SHA256).Hash.ToLower()
$levoSchedulerHash=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'native-bench\SchedulingBench.cpp') -Algorithm SHA256).Hash.ToLower()
$levoStart=(& $levoHdc shell 'hidumper -s PowerManagerService -a "-s"') -join "`n"
if($levoStart -notmatch 'Current State: AWAKE'){throw 'Keep phone awake before starting the bounded experiment'}
$levoRows=@()
& $levoHdc shell "timeout 360 $levoRemote/levo_scheduling_bench $levoRemote $Mode $Dit $Threads" | Tee-Object -FilePath $levoLog | ForEach-Object {
  if($_ -like 'LEVO_CHAIN_PROGRESS *'){
    $levoStage=$_.Substring(20) | ConvertFrom-Json
    Write-Output "$RunId : $($levoStage.stage); t=$([math]::Round($levoStage.seconds,2))s; RSS=$([math]::Round($levoStage.rssKiB/1024,1))MiB"
  }
}
$levoRaw=Get-Content -LiteralPath $levoLog | Where-Object {$_ -like 'LEVO_SCHEDULING_REPORT *'} | Select-Object -Last 1
if(!$levoRaw){throw 'No structured result: preserve log and inspect phone; no automatic retry'}
$levoResult=$levoRaw.Substring(23) | ConvertFrom-Json -AsHashtable
$levoEnd=(& $levoHdc shell 'hidumper -s PowerManagerService -a "-s"') -join "`n"
$levoOffBefore=[regex]::Match($levoStart,'Last Screen Off:\s*(\d+)').Groups[1].Value
$levoOffAfter=[regex]::Match($levoEnd,'Last Screen Off:\s*(\d+)').Groups[1].Value
$levoDisplayStable=($levoEnd -match 'Current State: AWAKE') -and ($levoOffBefore -ne '') -and ($levoOffBefore -eq $levoOffAfter)
$levoRecord=[ordered]@{
  runId=$RunId;date='2026-10-31';execution='CLI native executable, not installed signed App';mode=$Mode;dit=$Dit;threads=$Threads
  binarySha256=$levoBinaryHash;chainSourceSha256=$levoChainHash;schedulerSourceSha256=$levoSchedulerHash
  screenBefore=$levoStart;screenAfter=$levoEnd;displayStableAtEndpoints=$levoDisplayStable;result=$levoResult
}
$levoRecord | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $levoJson -Encoding utf8
$levoResult | Select-Object success,error,mode,forwardSeconds,prefillSeconds,decodeTokensPerSecond,pipelineSecondsThroughPcm,pipelineRtfThroughPcm,firstNumericalPacketSeconds,sampledPeakRssKiB,strictNumericalParityPassed | ConvertTo-Json -Compress | Write-Output
if(!$levoResult.success){throw 'Bounded experiment returned failure; result preserved'}
if(!$levoDisplayStable){Write-Warning 'Screen-off transition detected; preserve numerical result but exclude its timing from speed comparisons'}
