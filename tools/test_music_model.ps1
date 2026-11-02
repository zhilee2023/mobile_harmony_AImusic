param([Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device,
      [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$')][string]$ModelId,
      [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
      [switch]$Attach)
$ErrorActionPreference='Stop'
if($ModelId.Contains('..')){throw 'Invalid model ID'}
$testHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$testRoot=Split-Path -Parent $PSScriptRoot
$testOutput=Join-Path $testRoot "reports\music-model-$RunId.json"
if(Test-Path -LiteralPath $testOutput){throw 'Preserve previous test; choose a new run ID'}
$testFiles='/data/app/el2/100/base/com.zhilee.levomusic/haps/entry/files'
$testModel=if($ModelId -eq 'builtin'){"$testFiles/music-models"}else{"$testFiles/model-bundles/$ModelId"}
$testBefore=(& $testHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
if($testBefore -notmatch 'Current State: AWAKE'){throw 'Keep phone awake and LeVo Music in foreground'}
$testPreviousId='';$testPreviousText=''
if(!$Attach){
  try{$oldConfig=((& $testHdc -t $Device shell "cat $testModel/lm/config-phone.json") -join "`n")|ConvertFrom-Json;$testPreviousId=$oldConfig.levo_run_id}catch{}
  $testPreviousText=(& $testHdc -t $Device shell "cat $testModel/last-music-result.json") -join "`n"
  & $testHdc -t $Device shell 'aa force-stop com.zhilee.levomusic'
  & $testHdc -t $Device shell "aa start -a EntryAbility -b com.zhilee.levomusic --ps levoSelectModel $ModelId --ps musicAutoRun first-song-20261102"
}
$testPid=((& $testHdc -t $Device shell 'pidof com.zhilee.levomusic') -join '').Trim()
if($testPid -notmatch '^\d+$'){throw 'No unique Music App process'}
$testClock=[Diagnostics.Stopwatch]::StartNew();$testSamples=@();$testFirstProgress=$null;$testLast=''
$testConfigId='';$testHap=(Get-FileHash -LiteralPath (Join-Path $testRoot 'music-app\entry\build\default\outputs\default\entry-default-signed.hap') -Algorithm SHA256).Hash.ToLower()
while($testClock.Elapsed.TotalSeconds -lt 900){
  $testMission=(& $testHdc -t $Device shell 'aa dump -l') -join "`n"
  $match=[regex]::Match($testMission,'(?s)app name \[com\.zhilee\.levomusic\].*?state #([A-Z]+)')
  $testState=if($match.Success){$match.Groups[1].Value}else{'UNKNOWN'}
  $testSamples+=@{seconds=$testClock.Elapsed.TotalSeconds;state=$testState}
  $progress=(& $testHdc -t $Device shell "tail -n 1 $testModel/progress.jsonl") -join "`n"
  try{$row=$progress|ConvertFrom-Json;if(!$testFirstProgress){$testFirstProgress=$row};if($row.stage -ne $testLast){$testLast=$row.stage;Write-Output "$testLast t=$([math]::Round($row.seconds,1))s"}}catch{}
  if(!$testConfigId){try{$config=((& $testHdc -t $Device shell "cat $testModel/lm/config-phone.json") -join "`n")|ConvertFrom-Json;if($Attach -or $config.levo_run_id -ne $testPreviousId){$testConfigId=$config.levo_run_id}}catch{}}
  $result=$null;$resultText=(& $testHdc -t $Device shell "cat $testModel/last-music-result.json") -join "`n";try{$result=$resultText|ConvertFrom-Json}catch{}
  $newError=$result -and $result.error -and ($Attach -or $resultText -ne $testPreviousText)
  if($result -and (($result.lm.runId -eq $testConfigId -and $testConfigId) -or $newError)){
    $selection=((& $testHdc -t $Device shell "cat $testFiles/last-model-selection.json") -join "`n")|ConvertFrom-Json
    $after=(& $testHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n"
    $record=[ordered]@{runId=$RunId;modelId=$ModelId;hapSha256=$testHap;attachedToRunningGeneration=[bool]$Attach;
      firstObservedProgress=$testFirstProgress;foregroundSamples=$testSamples;foregroundStableDuringObservation=(@($testSamples|Where-Object {$_.state -ne 'FOREGROUND'}).Count -eq 0);
      screenRemainedOnDuringObservation=($after -match 'Current State: AWAKE') -and ([regex]::Match($testBefore,'Last Screen Off:\s*(\d+)').Groups[1].Value -eq [regex]::Match($after,'Last Screen Off:\s*(\d+)').Groups[1].Value);
      selection=$selection;result=$result;scope='Installed Music App model-selection and real generation acceptance, not a controlled speed comparison'}
    [IO.File]::WriteAllText($testOutput,($record|ConvertTo-Json -Depth 80),[Text.UTF8Encoding]::new($false))
    if(!$result.success){throw "Generation failed; report preserved: $($result.error)"}
    if($selection.result.activeId -ne $ModelId){throw 'Selected model differs from requested model'}
    Write-Output "Completed $($result.durationSeconds)s audio in $($result.totalSeconds)s; RTF=$($result.wholePhoneRtf)"
    return
  }
  $live=((& $testHdc -t $Device shell 'pidof com.zhilee.levomusic') -join '').Trim()
  if($live -ne $testPid){throw 'App exited; partial evidence retained in its model directory'}
  Start-Sleep -Seconds 3
}
throw 'Generation observer deadline; inspect existing process before restarting'
