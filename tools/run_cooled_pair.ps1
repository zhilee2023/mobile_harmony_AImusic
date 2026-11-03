param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [ValidateRange(1,120)][int]$CoolingMinutes=15,
  [ValidateSet(30,60)][int]$DurationSeconds=60,
  [string]$Python='python',
  [switch]$Resume
)
# One paired experiment, not a recurring schedule. Do not wake the phone or change its settings.
$ErrorActionPreference='Stop'
$pairRoot=Split-Path -Parent $PSScriptRoot
$pairHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$pairOutput=Join-Path $pairRoot "reports/cooled-pair-$RunId.json"
$pairLarge=Join-Path $pairRoot "reports/comparison-app-$RunId-large.json"
$pairSmall=Join-Path $pairRoot "reports/comparison-app-$RunId-small.json"
function Save-Pair { [IO.File]::WriteAllText($pairOutput,($script:pair|ConvertTo-Json -Depth 100),[Text.UTF8Encoding]::new($false)) }
function Phone-Awake { return (((& $pairHdc -t $Device shell 'hidumper -s PowerManagerService -a -s') -join "`n") -match 'Current State: AWAKE') }
function Phone-Ready {
  if(!(Phone-Awake)){return $false}
  $missions=(& $pairHdc -t $Device shell 'aa dump -l') -join "`n"
  $match=[regex]::Match($missions,'(?s)app name \[com\.zhilee\.levoharmony\].*?state #([A-Z]+)')
  return $match.Success -and $match.Groups[1].Value -eq 'FOREGROUND'
}
function Read-Cooling {
  $readings=@(& $pairHdc -t $Device shell 'for p in /sys/class/power_supply/Battery/temp /sys/class/thermal/thermal_zone0/temp /sys/class/power_supply/Battery/voltage_now /sys/class/power_supply/Battery/current_now /sys/class/power_supply/Battery/status; do cat "$p" 2>/dev/null || echo NA; done')
  return @{utc=[DateTimeOffset]::UtcNow.ToString('o');rawReadings=$readings;units=@('0.1 C','milli C','uV','mA','status')}
}
function Save-Wave($result,[string]$family) {
  $wave=$result.wave
  if(!$wave -or $wave -notmatch '^/data/(app/el2/100/base/com\.zhilee\.levoharmony|storage/el2/base)/haps/entry/files/chain-comparison/(public|small)/songs/[a-zA-Z0-9-]+\.wav$'){throw 'Expected a generated WAV in the comparison model song directory'}
  $transferPath=$wave -replace '^/data/storage/el2/base/','/data/app/el2/100/base/com.zhilee.levoharmony/'
  $directory=Join-Path $pairRoot "reports/local/cooled-pair/$RunId"
  New-Item -ItemType Directory -Force -Path $directory|Out-Null
  $destination=Join-Path $directory "$family-$DurationSeconds.wav"
  if(Test-Path -LiteralPath $destination){throw 'Preserve existing local audio'}
  & $pairHdc -t $Device file recv $transferPath $destination | Out-Host
  if($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $destination)){throw 'Generated audio transfer failed'}
  return @{path=$destination;sha256=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower();bytes=(Get-Item -LiteralPath $destination).Length}
}
if($Resume){
  if(!(Test-Path -LiteralPath $pairOutput)){throw 'No pair checkpoint to resume'}
  $script:pair=Get-Content -LiteralPath $pairOutput -Raw|ConvertFrom-Json -AsHashtable
  if($pair.runId -ne $RunId -or $pair.state -notin @('cooling','awaiting-awake','large-failed')){throw 'Only cooling/awaiting-awake or completed-public recovery checkpoints can resume; never rerun a completed model automatically'}
  $CoolingMinutes=[int]$pair.coolingMinutes
  $DurationSeconds=[int]$pair.durationSeconds
  $pair.resumedUtc=[DateTimeOffset]::UtcNow.ToString('o')
  if($pair.state -eq 'large-failed'){
    if(!$pair.large.result.success -or $pair.Contains('coolingStartUtc')){throw 'Only recover already completed public output; do not rerun failed inference'}
    $pair.workflowRecovery=@{utc=[DateTimeOffset]::UtcNow.ToString('o');priorObserverError=$pair.observerError;
      action='Export already completed public output through its physical storage path; no model inference rerun'}
    $pair.largeAudio=Save-Wave $pair.large.result 'public'
    & $pairHdc -t $Device shell 'aa force-stop com.zhilee.levoharmony'
    if($LASTEXITCODE -ne 0){throw 'Cannot confirm App stopped'}
    $pair.coolingStartUtc=[DateTimeOffset]::UtcNow.ToString('o');$pair.state='cooling';Save-Pair
  } else {$pair.coolingMonitoringGapPossible=$true}
} else {
  if((Test-Path -LiteralPath $pairOutput) -or (Test-Path -LiteralPath $pairLarge) -or (Test-Path -LiteralPath $pairSmall)){throw 'Preserve previous test files; choose a new RunId'}
  if(!(Phone-Ready)){throw 'Manually unlock the phone and open LeVo Harmony in foreground before starting'}
  $script:pair=[ordered]@{runId=$RunId;state='large-running';durationSeconds=$DurationSeconds;coolingMinutes=$CoolingMinutes;
    protocol="Public Mixed ${DurationSeconds}s -> stop inference -> at least $CoolingMinutes minutes idle -> student ${DurationSeconds}s; same research HAP and legacy shared VAE";
    coolingMonitoringGapPossible=$false;createdUtc=[DateTimeOffset]::UtcNow.ToString('o');coolingSamples=@();
    limits=@('Small music adapters and DiT untrained','USB battery-terminal readings are not isolated whole-device power','Idle interval is not proof of identical starting temperature','No screen wake or system setting changes')}
  Save-Pair
  try {
    $largeVariant=if($DurationSeconds -eq 60){'large-2-60'}else{'large-2'}
    & (Join-Path $PSScriptRoot 'run_comparison_app.ps1') -Device $Device -Variant $largeVariant -RunId "$RunId-large" -Python $Python
    $pair.large=Get-Content -LiteralPath $pairLarge -Raw|ConvertFrom-Json -AsHashtable
    if(!$pair.large.result.success){throw 'Public model did not complete; do not automatically run the student'}
    if($pair.large.result.durationSeconds -ne $DurationSeconds){throw 'Public audio duration differs from the paired protocol'}
    $pair.largeAudio=Save-Wave $pair.large.result 'public'
    & $pairHdc -t $Device shell 'aa force-stop com.zhilee.levoharmony'
    if($LASTEXITCODE -ne 0){throw 'Cannot confirm model App stopped; do not begin a cooling claim'}
    $pair.coolingStartUtc=[DateTimeOffset]::UtcNow.ToString('o');$pair.state='cooling';Save-Pair
  } catch {$pair.state='large-failed';$pair.observerError=$_.Exception.Message;Save-Pair;throw}
}
$coolingStart=[DateTimeOffset]::Parse($pair.coolingStartUtc)
while(([DateTimeOffset]::UtcNow-$coolingStart).TotalSeconds -lt $CoolingMinutes*60){
  $pair.coolingSamples+=@(Read-Cooling)
  $remaining=[math]::Max(0,$CoolingMinutes*60-([DateTimeOffset]::UtcNow-$coolingStart).TotalSeconds)
  Save-Pair;Write-Output "Cooling; $([math]::Ceiling($remaining)) seconds remain"
  if($remaining -gt 0){Start-Sleep -Milliseconds ([int][math]::Ceiling([math]::Min(30,$remaining)*1000))}
}
$pair.coolingSamples+=@(Read-Cooling)
$pair.observedCoolingSeconds=([DateTimeOffset]::UtcNow-$coolingStart).TotalSeconds
if(!(Phone-Ready)){
  $pair.state='awaiting-awake';Save-Pair
  Write-Output 'Cooling complete. Manually unlock and open LeVo Harmony in foreground, then resume this same RunId with -Resume. Public model will not rerun.'
  return
}
$pair.smallStartedUtc=[DateTimeOffset]::UtcNow.ToString('o');$pair.state='small-running';Save-Pair
try {
  $smallVariant=if($DurationSeconds -eq 60){'small-2-60'}else{'small-2'}
  & (Join-Path $PSScriptRoot 'run_comparison_app.ps1') -Device $Device -Variant $smallVariant -RunId "$RunId-small" -Python $Python
  $pair.small=Get-Content -LiteralPath $pairSmall -Raw|ConvertFrom-Json -AsHashtable
  if(!$pair.small.result.success){throw 'Student did not complete'}
  if($pair.small.result.durationSeconds -ne $DurationSeconds){throw 'Student audio duration differs from the paired protocol'}
  if($DurationSeconds -eq 60){$pair.smallAudio=Save-Wave $pair.small.result 'student-diagnostic'}
  if($pair.large.hapSha256 -ne $pair.small.hapSha256){throw 'HAP changed between paired tests'}
  $pair.state='complete';$pair.finishedUtc=[DateTimeOffset]::UtcNow.ToString('o');Save-Pair
  Write-Output "Paired results including thermal samples: $pairOutput"
} catch {$pair.state='small-failed';$pair.observerError=$_.Exception.Message;Save-Pair;throw}
