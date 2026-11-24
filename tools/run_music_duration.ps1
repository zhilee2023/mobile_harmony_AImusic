param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9]+$')][string]$Device,
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,65}$')][string]$RunId,
 [Parameter(Mandatory=$true)][ValidateSet(15,30,60,90,120,240,300)][int]$DurationSeconds,
 [ValidateSet(0,1)][int]$ModelChoice=0,
 [ValidateSet(0,2)][int]$DynamicOption=2,
 [ValidateSet(0,1)][int]$VaeFusion=1,
 [ValidateRange(0,600)][int]$CoolingSeconds=180,
 [switch]$ResumePrepared,
 [string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio'
)
$ErrorActionPreference='Stop'
if($ModelChoice -eq 1 -and ($DurationSeconds -gt 120 -or $DynamicOption -ne 0)){throw 'Large-model probe max120s; explicitly use -DynamicOption0 for its existing LM recipe'}
$musicProject=Split-Path -Parent $PSScriptRoot
$musicBundleName=(Get-Content -LiteralPath (Join-Path $musicProject 'stream-app/AppScope/app.json5') -Raw -Encoding UTF8 | ConvertFrom-Json).app.bundleName
if($musicBundleName -notmatch '^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$'){throw 'Invalid App bundleName'}
$musicHdc=Join-Path $DevEcoRoot 'sdk/default/openharmony/toolchains/hdc.exe'
$musicDir=Join-Path $musicProject "reports/local/$RunId"
$musicFiles="/data/app/el2/100/base/$musicBundleName/haps/entry/files"
if(Test-Path -LiteralPath $musicDir){if(!$ResumePrepared){throw 'Preserve previous measurements; choose a new RunId'}}
elseif($ResumePrepared){throw 'No prepared run to resume'}
else{$null=New-Item -ItemType Directory -Path $musicDir}
function Music-Hdc([string[]]$MusicArguments){
 $musicText=& $musicHdc -t $Device @MusicArguments
 if($LASTEXITCODE -ne 0){throw 'HDC operation failed; preserve partial result'}
 return ($musicText -join "`n")
}
function Music-ReadState {
 $musicPath=Join-Path $musicDir 'latest-state.json'
 & $musicHdc -t $Device file recv "$musicFiles/last-state.json" $musicPath *> $null
 if($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $musicPath)){return $null}
 try{return (Get-Content -LiteralPath $musicPath -Raw -Encoding UTF8 | ConvertFrom-Json)}catch{return $null}
}
function Music-Sample {
 $musicRaw=Music-Hdc @('shell','for p in /sys/class/power_supply/Battery/temp /sys/class/thermal/thermal_zone0/temp; do cat "$p" 2>/dev/null || echo NA; done')
 $musicValues=@($musicRaw -split "`n" | ForEach-Object {$_.Trim()})
 return @{utc=[DateTimeOffset]::UtcNow.ToString('o');batteryC=if($musicValues[0] -match '^\d+$'){[double]$musicValues[0]/10}else{$null};socNodeC=if($musicValues.Count -gt 1 -and $musicValues[1] -match '^\d+$'){[double]$musicValues[1]/1000}else{$null}}
}
$musicRecord=@{runTag=$RunId;bundleName=$musicBundleName;modelChoice=$ModelChoice;requestedDurationSeconds=$DurationSeconds;dynamicOption=$DynamicOption;vaeFusion=[bool]$VaeFusion;status='preparing';coolingSecondsRequested=$CoolingSeconds;
 scope='Actual selected-model App fixed-length generation with playback; W4 body/W8 head, DiT CPU2steps CFG1.5, CPU Low VAE; dynamic option and fusion recorded explicitly; EOS suppressed for timing';
 samples=@();createdUtc=[DateTimeOffset]::UtcNow.ToString('o')}
if($ResumePrepared){
 $musicPrepared=Get-Content -LiteralPath (Join-Path $musicDir 'run.json') -Raw -Encoding UTF8 | ConvertFrom-Json
 if($musicPrepared.status -ne 'awaiting-unlock' -or [int]$musicPrepared.modelChoice -ne $ModelChoice -or $musicPrepared.requestedDurationSeconds -ne $DurationSeconds -or $musicPrepared.dynamicOption -ne $DynamicOption -or [bool]$musicPrepared.vaeFusion -ne [bool]$VaeFusion){throw 'Only the same prepared, not-yet-started measurement may resume'}
 $musicRecord=@{};foreach($musicProperty in $musicPrepared.psobject.Properties){$musicRecord[$musicProperty.Name]=$musicProperty.Value}
 $musicRecord.resumedUtc=[DateTimeOffset]::UtcNow.ToString('o')
}
function Music-Save {[IO.File]::WriteAllText((Join-Path $musicDir 'run.json'),($musicRecord | ConvertTo-Json -Depth 100),[Text.UTF8Encoding]::new($false))}
Music-Save
$musicExisting=Music-ReadState
$musicPidBefore=Music-Hdc @('shell',"pidof $musicBundleName || true")
 if($musicExisting -and $musicExisting.busy -and $musicPidBefore.Trim()){
 $musicPreviousDone=$false
 if($musicExisting.status -in @('completed','stopped','error') -and $musicExisting.runId -match '^\d+$'){
  $musicPreviousPath=Join-Path $musicDir 'previous-native-result.json'
  & $musicHdc -t $Device file recv "$musicFiles/stream-runs/$($musicExisting.runId)/result.json" $musicPreviousPath *> $null
  if($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $musicPreviousPath)){
   try{$musicPrevious=Get-Content -LiteralPath $musicPreviousPath -Raw -Encoding UTF8 | ConvertFrom-Json;$musicPreviousDone=(!$musicPrevious.busy -and $musicPrevious.runId -eq $musicExisting.runId)}catch{}
  }
 }
 if(!$musicPreviousDone){throw 'An App task is still active; do not kill it or start a second generation'}
}
if(!$ResumePrepared){
 $musicRecord.status='cooling';$musicCoolStart=[DateTimeOffset]::UtcNow
 while(([DateTimeOffset]::UtcNow-$musicCoolStart).TotalSeconds -lt $CoolingSeconds){
  $musicRecord.samples+=@(Music-Sample);Music-Save
  $musicRemaining=$CoolingSeconds-([DateTimeOffset]::UtcNow-$musicCoolStart).TotalSeconds
  if($musicRemaining -gt 0){Start-Sleep -Milliseconds ([int]([Math]::Min(30,$musicRemaining)*1000))}
 }
 $musicRecord.coolingSecondsObserved=([DateTimeOffset]::UtcNow-$musicCoolStart).TotalSeconds
}
$musicPower=Music-Hdc @('shell','hidumper -s PowerManagerService -a -a')
[IO.File]::WriteAllText((Join-Path $musicDir 'power-before.txt'),$musicPower,[Text.UTF8Encoding]::new($false))
if($musicPower -notmatch 'Current State: AWAKE'){$musicRecord.status='awaiting-unlock';Music-Save;throw 'Manually unlock the phone; no inference started'}
$musicOffBefore=[regex]::Match($musicPower,'Last Screen Off:\s*(\d+)').Groups[1].Value
$musicInstalledHashText=Music-Hdc @('shell',"sha256sum /data/app/el1/bundle/public/$musicBundleName/entry.hap")
$musicRecord.installedHapSha256=[regex]::Match($musicInstalledHashText,'^[a-fA-F0-9]{64}').Value.ToLowerInvariant()
if(!$musicRecord.installedHapSha256){throw 'Cannot verify installed HAP hash; no inference started'}
$musicBeforeStart=Music-ReadState
if($musicBeforeStart -and $musicBeforeStart.busy){$musicRecord.status='blocked-active-task-after-cooling';Music-Save;throw 'Another generation became active during cooling; App left untouched'}
$null=Music-Hdc @('shell',"aa force-stop $musicBundleName") # Confirmed idle above, one fresh process per duration.
$musicRecord.samples+=@(Music-Sample);$musicRecord.status='starting';Music-Save
$null=Music-Hdc @('shell',"aa start -a EntryAbility -b $musicBundleName --ps streamLabTest duration-benchmark --pi streamLabModel $ModelChoice --pi streamLabSeconds $DurationSeconds --pi streamLabDynamicOption $DynamicOption --pi streamLabVaeFusion $VaeFusion --ps streamLabRunTag $RunId")
$musicClock=[DateTimeOffset]::UtcNow;$musicLastAnnounce=0
while(([DateTimeOffset]::UtcNow-$musicClock).TotalSeconds -lt 3600){
 Start-Sleep -Seconds 5
 $musicState=Music-ReadState
 $musicObservation=Music-Sample
 $musicMissions=Music-Hdc @('shell','aa dump -l')
 $musicForeground=[regex]::Match($musicMissions,('(?s)app name \['+[regex]::Escape($musicBundleName)+'\].*?state #([A-Z]+)'))
 $musicRecord.foregroundObserved=($musicForeground.Success -and $musicForeground.Groups[1].Value -eq 'FOREGROUND')
 $musicObservation.foregroundState=if($musicForeground.Success){$musicForeground.Groups[1].Value}else{'UNKNOWN'}
 $musicRecord.samples+=@($musicObservation)
 if(!$musicRecord.foregroundObserved){
  $musicRecord.foregroundGapObserved=$true
  $musicDiagnosticName='foreground-'+[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()+'.txt'
  [IO.File]::WriteAllText((Join-Path $musicDir $musicDiagnosticName),$musicMissions,[Text.UTF8Encoding]::new($false))
 }
 if($musicState -and $musicState.runTag -eq $RunId){
  $musicRecord.status='running';$musicRecord.lastProgress=@{runId=$musicState.runId;phase=$musicState.phase;conditionFrames=$musicState.conditionFrames;generatedSamples=$musicState.generatedSamples;rssPeakMiB=$musicState.rssPeakMiB};Music-Save
  if(!$musicState.busy){$musicRecord.generation=$musicState;$musicRecord.status=$musicState.status;$musicRecord.completionSource='App checkpoint';break}
  # A final native result is written only after model/KV destruction. It also
  # recovers measurements from the v0.1.8 UI checkpoint transition bug.
  if($musicState.status -in @('completed','stopped','error') -and $musicState.runId -match '^\d+$'){
   $musicFinalPath=Join-Path $musicDir 'native-result.json'
   & $musicHdc -t $Device file recv "$musicFiles/stream-runs/$($musicState.runId)/result.json" $musicFinalPath *> $null
   if($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $musicFinalPath)){
    try{$musicFinal=Get-Content -LiteralPath $musicFinalPath -Raw -Encoding UTF8 | ConvertFrom-Json}catch{$musicFinal=$null}
    if($musicFinal -and !$musicFinal.busy -and $musicFinal.runTag -eq $RunId){$musicRecord.generation=$musicFinal;$musicRecord.status=$musicFinal.status;$musicRecord.completionSource='Native final result';break}
   }
  }
 }
 $musicPid=Music-Hdc @('shell',"pidof $musicBundleName || true")
 if(!$musicPid.Trim()){$musicRecord.status='process-exited';Music-Save;throw 'App process exited; last progress retained, not classified as OOM without evidence'}
 $musicWall=([DateTimeOffset]::UtcNow-$musicClock).TotalSeconds
 if($musicWall-$musicLastAnnounce -ge 30){$musicLastAnnounce=$musicWall;Write-Output "${DurationSeconds}s target: $([Math]::Round($musicWall))s elapsed, phase $($musicState.phase), frames $($musicState.conditionFrames)"}
}
if(!$musicRecord.ContainsKey('generation')){$musicRecord.status='observer-timeout';Music-Save;throw 'No complete observed result; App left running, do not restart automatically'}
$musicPowerAfter=Music-Hdc @('shell','hidumper -s PowerManagerService -a -a')
[IO.File]::WriteAllText((Join-Path $musicDir 'power-after.txt'),$musicPowerAfter,[Text.UTF8Encoding]::new($false))
$musicRecord.lastScreenOffUnchanged=([regex]::Match($musicPowerAfter,'Last Screen Off:\s*(\d+)').Groups[1].Value -eq $musicOffBefore)
$musicRecord.screenAwakeAtEnd=($musicPowerAfter -match 'Current State: AWAKE')
$musicActual=[double]$musicRecord.generation.generatedSamples/48000
$musicRecord.effectiveAudioSeconds=$musicActual
$musicRecord.controlledTimingAccepted=($musicRecord.status -eq 'completed' -and $musicActual -eq $DurationSeconds -and !$musicRecord.foregroundGapObserved -and $musicRecord.lastScreenOffUnchanged -and $musicRecord.screenAwakeAtEnd)
if($musicRecord.status -eq 'completed'){
 $musicRecord.observedWholeRtf=[double]$musicRecord.generation.elapsedSeconds/$musicActual
 if($musicRecord.controlledTimingAccepted){$musicRecord.wholeRtf=$musicRecord.observedWholeRtf}
}
Music-Save
$null=Music-Hdc @('file','recv',"$musicFiles/stream-runs/$($musicRecord.generation.runId)",(Join-Path $musicDir 'artifacts'))
# Leave up to one final10.24s chunk to drain; do not force-stop playback.
Start-Sleep -Seconds 12
$null=Music-Hdc @('file','recv',"$musicFiles/stream-events.jsonl",(Join-Path $musicDir 'events.jsonl'))
$musicHap=Join-Path $musicProject 'stream-app/entry/build/default/outputs/default/entry-default-signed.hap'
$musicRecord.hapSha256=$musicRecord.installedHapSha256
if(Test-Path -LiteralPath $musicHap){$musicRecord.localBuildHapSha256=(Get-FileHash -LiteralPath $musicHap -Algorithm SHA256).Hash.ToLowerInvariant()}
$musicRecord.finishedUtc=[DateTimeOffset]::UtcNow.ToString('o');Music-Save
if($musicRecord.status -ne 'completed' -or $musicActual -ne $DurationSeconds){throw 'Incomplete or wrong-duration generation; not an accepted full-length RTF'}
if(!$musicRecord.controlledTimingAccepted){throw 'Audio completed with background/screen interruption; timing quarantined, do not accept as controlled RTF'}
Write-Output "Completed $musicActual seconds: elapsed $($musicRecord.generation.elapsedSeconds), RTF $($musicRecord.wholeRtf), RSS $($musicRecord.generation.rssPeakMiB) MiB"
