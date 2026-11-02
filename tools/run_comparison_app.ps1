param(
  [Parameter(Mandatory=$true)][ValidateSet('large-2','small-2')][string]$Variant,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device
)
$ErrorActionPreference='Stop'
$cmpHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$cmpRoot=Split-Path -Parent $PSScriptRoot
$cmpReport=Join-Path $cmpRoot "reports\comparison-app-$RunId.json"
if(Test-Path -LiteralPath $cmpReport){throw 'Preserve existing report; choose a new RunId'}
$cmpRemote='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files'
$cmpKey=if($Variant -eq 'small-2'){'comparisonSmall2'}else{'comparisonLarge2'}
$cmpModel=if($Variant -eq 'small-2'){'small'}else{'public'}
$cmpBefore=(& $cmpHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
if($cmpBefore -notmatch 'Current State: AWAKE'){throw 'Phone is asleep; keep it awake before running'}
& $cmpHdc -t $Device shell 'aa force-stop com.zhilee.levoharmony'
& $cmpHdc -t $Device shell "aa start -a EntryAbility -b com.zhilee.levoharmony --ps levoDiagnostic chain-comparison-$Variant --ps levoDiagnosticRunId $RunId"
$cmpPid=(& $cmpHdc -t $Device shell 'pidof com.zhilee.levoharmony').Trim()
if($cmpPid -notmatch '^\d+$'){throw 'No unique live App PID'}
$cmpClock=[System.Diagnostics.Stopwatch]::StartNew();$cmpLast=''
while($cmpClock.Elapsed.TotalSeconds -lt 800){
  $raw=(& $cmpHdc -t $Device shell "cat $cmpRemote/levo-test-report.json") -join "`n"
  try{$archive=$raw | ConvertFrom-Json -AsHashtable}catch{$archive=$null}
  if($archive -and $archive.lastRunId -eq $RunId){
    $cmpAfter=(& $cmpHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
    $offBefore=[regex]::Match($cmpBefore,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $offAfter=[regex]::Match($cmpAfter,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $record=[ordered]@{runId=$RunId;date='2026-11-02';variant=$Variant;execution='Installed signed LeVo Harmony App';
      hapSha256=(Get-FileHash -LiteralPath (Join-Path $cmpRoot 'app\entry\build\default\outputs\default\entry-default-signed.hap') -Algorithm SHA256).Hash.ToLower();
      displayStable=($cmpAfter -match 'Current State: AWAKE') -and $offBefore -eq $offAfter;
      observedSeconds=$cmpClock.Elapsed.TotalSeconds;result=$archive[$cmpKey]}
    [System.IO.File]::WriteAllText($cmpReport,($record | ConvertTo-Json -Depth 70),[System.Text.UTF8Encoding]::new($false))
    $record.result | Select-Object success,error,stageSeconds,stageRtf,memory | ConvertTo-Json -Depth 8
    if(-not $record.result.success){throw 'Comparison stopped; result preserved'}
    return
  }
  $live=(& $cmpHdc -t $Device shell 'pidof com.zhilee.levoharmony').Trim()
  if($live -ne $cmpPid){throw 'App process ended or changed; do not automatically restart'}
  $progress=(& $cmpHdc -t $Device shell "tail -n 1 $cmpRemote/chain-comparison/$cmpModel/progress.jsonl") -join "`n"
  try{$state=$progress|ConvertFrom-Json;if($state.stage -ne $cmpLast){$cmpLast=$state.stage;Write-Output "$RunId $cmpLast t=$([math]::Round($state.seconds,1))s RSS=$([math]::Round($state.rssKiB/1024))MiB"}}catch{}
  Start-Sleep -Seconds 3
}
throw 'Observer timeout; inspect existing process before any restart'
