param([Parameter(Mandatory=$true)][string]$Device,[string]$RunId='20261102-ui-aba-r1',[switch]$ObserveOnly)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[a-zA-Z0-9-]+$'){throw 'Invalid RunId'}
$uiHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$uiRoot=Split-Path -Parent $PSScriptRoot
$uiRemote='/data/app/el2/100/base/com.zhilee.levomusic/haps/entry/files/music-models'
$uiPath=Join-Path $uiRoot "reports\$RunId.json"
if(Test-Path -LiteralPath $uiPath){throw 'Choose a new RunId to preserve results'}
$uiBefore=(& $uiHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
if($uiBefore -notmatch 'Current State: AWAKE'){throw 'Keep phone awake before testing'}
$uiPrevious=(& $uiHdc -t $Device shell "cat $uiRemote/last-ui-benchmark.json") -join "`n"
if(!$ObserveOnly){
  & $uiHdc -t $Device shell 'aa force-stop com.zhilee.levomusic'
  & $uiHdc -t $Device shell "aa start -a EntryAbility -b com.zhilee.levomusic --ps musicAutoRun llm-ui-aba-20261102"
}
$uiClock=[System.Diagnostics.Stopwatch]::StartNew();$uiPid='';$uiLast='';$uiFinished=$false
while($uiClock.Elapsed.TotalSeconds -lt 420){
  $live=(& $uiHdc -t $Device shell 'pidof com.zhilee.levomusic').Trim()
  if(!$uiPid){$uiPid=$live};if($live -ne $uiPid){throw 'App process changed; inspect before restarting'}
  $line=(& $uiHdc -t $Device shell "tail -n 1 $uiRemote/progress.jsonl") -join "`n"
  try{$stage=$line|ConvertFrom-Json}catch{$stage=$null}
  if($stage -and $stage.stage -like 'ui-benchmark-*'){
    if($stage.stage -ne $uiLast){$uiLast=$stage.stage;Write-Output "$uiLast $([math]::Round($stage.seconds,1))s RSS=$([math]::Round($stage.rssKiB/1024))MiB"}
    $raw=(& $uiHdc -t $Device shell "cat $uiRemote/last-ui-benchmark.json") -join "`n"
    try{$report=$raw|ConvertFrom-Json}catch{$report=$null}
    if($report -and ($ObserveOnly -or $raw -ne $uiPrevious) -and ($report.success -or $report.error)){
      $uiAfter=(& $uiHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
      $beforeOff=[regex]::Match($uiBefore,'Last Screen Off:\s*(\d+)').Groups[1].Value
      $afterOff=[regex]::Match($uiAfter,'Last Screen Off:\s*(\d+)').Groups[1].Value
      $hashes=(& $uiHdc -t $Device shell "sha256sum $uiRemote/lm/generation-embeddings-bf16.bin $uiRemote/lm/llm.mnn.weight $uiRemote/phone-generated-codes.json") -join "`n"
      $archive=[ordered]@{runId=$RunId;date='2026-11-02';execution='signed LeVo Music App';observeOnly=[bool]$ObserveOnly;displayStableDuringObservation=($beforeOff -eq $afterOff -and $uiAfter -match 'Current State: AWAKE');inputHashes=$hashes;result=$report}
      [System.IO.File]::WriteAllText($uiPath,($archive|ConvertTo-Json -Depth 80),[System.Text.UTF8Encoding]::new($false))
      $report.rows|ForEach-Object{[pscustomobject]@{phase=$_.phase;animation=$_.animationEnabled;prefill=$_.prefillSeconds;decode=$_.decodeSeconds;token_s=$_.decodeTokensPerSecond;L2=$_.maxLogitsL2VsA1}}|Format-Table
      if(!$report.success){throw "Benchmark stopped: $($report.error)"};$uiFinished=$true;break
    }
  }
  Start-Sleep -Seconds 3
}
if(!$uiFinished){throw 'Observer timeout; do not automatically restart existing work'}
