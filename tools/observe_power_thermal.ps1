param([Parameter(Mandatory=$true)][string]$Device,[Parameter(Mandatory=$true)][string]$RunId,
      [Parameter(Mandatory=$true)][ValidateSet('public','small')][string]$Model)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[a-zA-Z0-9-]+$'){throw 'Invalid RunId'}
$sensorHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$sensorRoot=Split-Path -Parent $PSScriptRoot
$sensorLog=Join-Path $sensorRoot "reports\power-thermal-$RunId.jsonl"
if(Test-Path -LiteralPath $sensorLog){throw 'Preserve previous sensor data'}
$sensorFiles='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files'
$sensorPaths=@('/sys/class/thermal/thermal_zone0/temp','/sys/class/power_supply/Battery/temp',
 '/sys/class/power_supply/Battery/voltage_now','/sys/class/power_supply/Battery/current_now',
 '/sys/class/power_supply/charge_power/sys_power_now','/sys/class/power_supply/charge_power/charge_loss_now',
 '/sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq','/sys/devices/system/cpu/cpufreq/policy3/scaling_cur_freq',
 '/sys/class/power_supply/Battery/status','/sys/class/power_supply/USB/online')
$sensorPid=((& $sensorHdc -t $Device shell 'pidof com.zhilee.levoharmony') -join '').Trim()
if($sensorPid -notmatch '^\d+$'){throw 'No benchmark process'}
$sensorTime=[System.Diagnostics.Stopwatch]::StartNew()
while($sensorTime.Elapsed.TotalSeconds -lt 1500){
  $live=((& $sensorHdc -t $Device shell 'pidof com.zhilee.levoharmony') -join '').Trim();if($live -ne $sensorPid){break}
  $raw=@(& $sensorHdc -t $Device shell ('cat '+($sensorPaths -join ' ')))
  $progress=(& $sensorHdc -t $Device shell "tail -n 1 $sensorFiles/chain-comparison/$Model/progress.jsonl") -join "`n"
  $row=[ordered]@{observerElapsedSeconds=$sensorTime.Elapsed.TotalSeconds;model=$Model;paths=$sensorPaths;rawReadings=$raw;progress=$progress}
  [System.IO.File]::AppendAllText($sensorLog,($row|ConvertTo-Json -Depth 6 -Compress)+"`n",[System.Text.UTF8Encoding]::new($false))
  $archive=(& $sensorHdc -t $Device shell "cat $sensorFiles/levo-test-report.json") -join "`n"
  try{$report=$archive|ConvertFrom-Json;if($report.lastRunId -eq $RunId){break}}catch{}
  Start-Sleep -Seconds 5
}
Write-Output 'Power/thermal recording finished; USB charging and voltage/current are not isolated App power.'
