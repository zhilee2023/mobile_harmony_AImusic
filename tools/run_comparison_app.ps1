param(
  [Parameter(Mandatory=$true)][ValidateSet('large-2','small-2','large-2-60','small-2-60','large-2-120','small-2-120','large-2-120-low','large-codec-120','large-codec-90','large-2-90','small-2-90','small-vae-90')][string]$Variant,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device,
  [switch]$RecordSensors=$true,
  [ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$')][string]$ModelBundle='',
  [ValidateRange(1,60)][double]$SensorIntervalSeconds=5,
  [string]$Python='python'
)
$ErrorActionPreference='Stop'
$cmpHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$cmpRoot=Split-Path -Parent $PSScriptRoot
$cmpHapSha=(Get-FileHash -LiteralPath (Join-Path $cmpRoot 'app\entry\build\default\outputs\default\entry-default-signed.hap') -Algorithm SHA256).Hash.ToLower()
$cmpReport=Join-Path $cmpRoot "reports\comparison-app-$RunId.json"
if(Test-Path -LiteralPath $cmpReport){throw 'Preserve existing report; choose a new RunId'}
$cmpRemote='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files'
$cmpKey=switch($Variant){'small-vae-90'{'comparisonSmallVae90'} 'small-2'{'comparisonSmall2'} 'large-2'{'comparisonLarge2'} 'small-2-60'{'comparisonSmall60'} 'large-2-60'{'comparisonLarge60'} 'small-2-120'{'comparisonSmall120'} 'large-2-120'{'comparisonLarge120'} 'large-2-120-low'{'comparisonLarge120'} 'large-codec-120'{'comparisonLargeCodec120'} 'large-codec-90'{'comparisonLargeCodec90'} 'small-2-90'{'comparisonSmall90'} 'large-2-90'{'comparisonLarge90'}}
$cmpModel=if($Variant.StartsWith('small')){'small'}else{'public'}
if($ModelBundle.Contains('..')){throw 'Invalid model bundle ID'}
$cmpProgress=if($ModelBundle){"$cmpRemote/model-bundles/$ModelBundle/progress.jsonl"}else{"$cmpRemote/chain-comparison/$cmpModel/progress.jsonl"}
$cmpBefore=(& $cmpHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
if($cmpBefore -notmatch 'Current State: AWAKE'){throw 'Phone is asleep; keep it awake before running'}
& $cmpHdc -t $Device shell 'aa force-stop com.zhilee.levoharmony'
$cmpWant="aa start -a EntryAbility -b com.zhilee.levoharmony --ps levoDiagnostic chain-comparison-$Variant --ps levoDiagnosticRunId $RunId"
if($ModelBundle){$cmpWant+=" --ps levoModelBundle $ModelBundle"}
& $cmpHdc -t $Device shell $cmpWant
$cmpPid=(& $cmpHdc -t $Device shell 'pidof com.zhilee.levoharmony').Trim()
if($cmpPid -notmatch '^\d+$'){throw 'No unique live App PID'}
$cmpSensorJob=$null
if($RecordSensors){$cmpSensorJob=Start-Job -ScriptBlock {param($script,$device,$run,$model,$bundle,$interval,$stop); & $script -Device $device -RunId $run -Model $model -BundleId $bundle -IntervalSeconds $interval -StopFile $stop} -ArgumentList (Join-Path $PSScriptRoot 'observe_power_thermal.ps1'),$Device,$RunId,$cmpModel,$ModelBundle,$SensorIntervalSeconds,$cmpReport}
$cmpClock=[System.Diagnostics.Stopwatch]::StartNew();$cmpLast='';$cmpForegroundStable=$true;$cmpForegroundSamples=@()
try {
while($cmpClock.Elapsed.TotalSeconds -lt 1500){
  $raw=(& $cmpHdc -t $Device shell "cat $cmpRemote/levo-test-report.json") -join "`n"
  try{$archive=$raw | ConvertFrom-Json}catch{$archive=$null}
  if($archive -and $archive.lastRunId -eq $RunId){
    $cmpAfter=(& $cmpHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
    $offBefore=[regex]::Match($cmpBefore,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $offAfter=[regex]::Match($cmpAfter,'Last Screen Off:\s*(\d+)').Groups[1].Value
    $record=[ordered]@{runId=$RunId;date=(Get-Date -Format 'yyyy-MM-dd');variant=$Variant;modelBundle=$ModelBundle;observedAppPid=$cmpPid;execution='Installed signed LeVo Harmony App';
      hapSha256=$cmpHapSha;appVersion=$archive.version;
      displayStable=($cmpAfter -match 'Current State: AWAKE') -and $offBefore -eq $offAfter;
      foregroundStable=$cmpForegroundStable;foregroundSamples=$cmpForegroundSamples;
      observedSeconds=$cmpClock.Elapsed.TotalSeconds;result=$archive.$cmpKey}
    [System.IO.File]::WriteAllText($cmpReport,($record | ConvertTo-Json -Depth 70),[System.Text.UTF8Encoding]::new($false))
    $record.result | Select-Object success,error,stageSeconds,stageRtf,memory | ConvertTo-Json -Depth 8
    if(-not $record.result.success){throw 'Comparison stopped; result preserved'}
    return
  }
  $live=((& $cmpHdc -t $Device shell 'pidof com.zhilee.levoharmony') -join '').Trim()
  if($live -ne $cmpPid){throw 'App process ended or changed; do not automatically restart'}
  $mission=(& $cmpHdc -t $Device shell 'aa dump -l') -join "`n"
  $match=[regex]::Match($mission,'(?s)app name \[com\.zhilee\.levoharmony\].*?state #([A-Z]+)')
  $stateName=if($match.Success){$match.Groups[1].Value}else{'UNKNOWN'}
  $cmpForegroundSamples+=@{seconds=$cmpClock.Elapsed.TotalSeconds;state=$stateName}
  if($stateName -ne 'FOREGROUND'){$cmpForegroundStable=$false;Write-Output "Foreground validation failed: $stateName; timings must not be treated as a foreground baseline"}
  $progress=(& $cmpHdc -t $Device shell "tail -n 1 $cmpProgress") -join "`n"
  try{$state=$progress|ConvertFrom-Json;if($state.stage -ne $cmpLast){$cmpLast=$state.stage;Write-Output "$RunId $cmpLast t=$([math]::Round($state.seconds,1))s RSS=$([math]::Round($state.rssKiB/1024))MiB"}}catch{}
  Start-Sleep -Seconds 3
}
throw 'Observer timeout; inspect existing process before any restart'
} catch {
  if(!(Test-Path -LiteralPath $cmpReport)){
    $partial=[ordered]@{runId=$RunId;date=(Get-Date -Format 'yyyy-MM-dd');variant=$Variant;modelBundle=$ModelBundle;
      observedAppPid=$cmpPid;hapSha256=$cmpHapSha;complete=$false;observerError=$_.Exception.Message;
      foregroundSamples=$cmpForegroundSamples;lastObservedProgressText=$progress;
      scope='Observer stopped without an archived native completion; not a completed E2E result'}
    [IO.File]::WriteAllText($cmpReport,($partial|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false))
  }
  throw
} finally {
  if($cmpSensorJob){
    Wait-Job $cmpSensorJob -Timeout 10 | Out-Null;Stop-Job $cmpSensorJob;Remove-Job $cmpSensorJob
    try {
      & $Python -X utf8 (Join-Path $PSScriptRoot 'summarize_telemetry.py') --log (Join-Path $cmpRoot "reports/power-thermal-$RunId.jsonl") --attach-report $cmpReport
      if($LASTEXITCODE -ne 0){Write-Warning 'Sensor summary was not attached; preserve the raw sensor log'}
    } catch {Write-Warning "Sensor summary failed: $($_.Exception.Message); native test report is preserved"}
  }
}
