param([Parameter(Mandatory=$true)][string]$Device,[Parameter(Mandatory=$true)][string]$RunId,
      [Parameter(Mandatory=$true)][ValidateSet('public','small')][string]$Model,
      [string]$BundleId='',
      [ValidateSet('research','music')][string]$App='research',
      [ValidateRange(1,60)][double]$IntervalSeconds=5,
      [string]$StopFile='')
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[a-zA-Z0-9-]+$'){throw 'Invalid RunId'}
$sensorHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$sensorRoot=Split-Path -Parent $PSScriptRoot
$sensorLog=Join-Path $sensorRoot "reports\power-thermal-$RunId.jsonl"
if(Test-Path -LiteralPath $sensorLog){throw 'Preserve previous sensor data'}
$sensorBundle=if($App -eq 'music'){'com.zhilee.levomusic'}else{'com.zhilee.levoharmony'}
$sensorFiles="/data/app/el2/100/base/$sensorBundle/haps/entry/files"
if($BundleId -and ($BundleId -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$' -or $BundleId.Contains('..'))){throw 'Invalid bundle ID'}
$sensorProgress=if($App -eq 'music' -and (!$BundleId -or $BundleId -eq 'builtin')){"$sensorFiles/music-models/progress.jsonl"}elseif($BundleId){"$sensorFiles/model-bundles/$BundleId/progress.jsonl"}else{"$sensorFiles/chain-comparison/$Model/progress.jsonl"}
$sensorPaths=@('/sys/class/thermal/thermal_zone0/temp','/sys/class/power_supply/Battery/temp',
 '/sys/class/power_supply/Battery/voltage_now','/sys/class/power_supply/Battery/current_now',
 '/sys/class/power_supply/charge_power/sys_power_now','/sys/class/power_supply/charge_power/charge_loss_now',
 '/sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq','/sys/devices/system/cpu/cpufreq/policy3/scaling_cur_freq',
 '/sys/class/power_supply/Battery/status','/sys/class/power_supply/USB/online')
$sensorPid=((& $sensorHdc -t $Device shell "pidof $sensorBundle") -join '').Trim()
if($sensorPid -notmatch '^\d+$'){throw 'No benchmark process'}
$sensorTime=[System.Diagnostics.Stopwatch]::StartNew()
while($sensorTime.Elapsed.TotalSeconds -lt 1500){
  if($StopFile -and (Test-Path -LiteralPath $StopFile)){break}
  $sampleStart=$sensorTime.Elapsed.TotalSeconds
  $live=((& $sensorHdc -t $Device shell "pidof $sensorBundle") -join '').Trim();if($live -ne $sensorPid){break}
  $readCommand='for sensor_path in '+($sensorPaths -join ' ')+'; do cat "$sensor_path" 2>/dev/null || echo NA; done'
  $raw=@(& $sensorHdc -t $Device shell $readCommand)
  $progress=(& $sensorHdc -t $Device shell "tail -n 1 $sensorProgress") -join "`n"
  $row=[ordered]@{observerElapsedSeconds=$sensorTime.Elapsed.TotalSeconds;runId=$RunId;observedAppPid=$sensorPid;model=$Model;app=$App;modelBundle=$BundleId;requestedIntervalSeconds=$IntervalSeconds;
    units=@{socTemperature='millidegreeC';batteryTemperature='decidegreeC';batteryVoltage='uV';batteryCurrent='mA';cpuFrequency='kHz';vendorPower='unverified raw'};
    powerScope='Battery-terminal |V*I| magnitude; USB input/charging present, not isolated App or whole-device power';
    paths=$sensorPaths;rawReadings=$raw;progress=$progress}
  [System.IO.File]::AppendAllText($sensorLog,($row|ConvertTo-Json -Depth 6 -Compress)+"`n",[System.Text.UTF8Encoding]::new($false))
  if($App -eq 'research'){
    $archive=(& $sensorHdc -t $Device shell "cat $sensorFiles/levo-test-report.json") -join "`n"
    try{$report=$archive|ConvertFrom-Json;if($report.lastRunId -eq $RunId){break}}catch{}
  }
  $remaining=$IntervalSeconds-($sensorTime.Elapsed.TotalSeconds-$sampleStart)
  if($remaining -gt 0){Start-Sleep -Milliseconds ([int][math]::Ceiling($remaining*1000))}
}
Write-Output 'Power/thermal recording finished; USB charging and voltage/current are not isolated App power.'
