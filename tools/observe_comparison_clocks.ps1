param([Parameter(Mandatory=$true)][string]$Device,[Parameter(Mandatory=$true)][string]$RunId)
$ErrorActionPreference='Stop'
if($RunId -notmatch '^[a-zA-Z0-9-]+$'){throw 'Invalid RunId'}
$clockHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$clockRoot=Split-Path -Parent $PSScriptRoot
$clockPath=Join-Path $clockRoot "reports\clocks-$RunId.jsonl"
if(Test-Path -LiteralPath $clockPath){throw 'Preserve previous clock observations'}
$clockFiles='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files'
$clockPaths=@('/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq','/sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq',
 '/sys/devices/system/cpu/cpufreq/policy2/scaling_cur_freq','/sys/devices/system/cpu/cpufreq/policy3/scaling_cur_freq',
 '/sys/devices/system/cpu/cpufreq/policy2/cpuinfo_max_freq','/sys/devices/system/cpu/cpufreq/policy3/cpuinfo_max_freq',
 '/sys/class/thermal/thermal_zone0/type','/sys/class/thermal/thermal_zone0/temp')
$clockPid=(& $clockHdc -t $Device shell 'pidof com.zhilee.levoharmony').Trim()
$clockTimer=[System.Diagnostics.Stopwatch]::StartNew()
while($clockTimer.Elapsed.TotalSeconds -lt 1500){
  $live=((& $clockHdc -t $Device shell 'pidof com.zhilee.levoharmony') -join '').Trim();if($live -ne $clockPid){break}
  $raw=@(& $clockHdc -t $Device shell ('cat '+($clockPaths -join ' ')))
  $progress=(& $clockHdc -t $Device shell "tail -n 1 $clockFiles/chain-comparison/public/progress.jsonl") -join "`n"
  $row=[ordered]@{observerElapsedSeconds=$clockTimer.Elapsed.TotalSeconds;paths=$clockPaths;rawReadings=$raw;progress=$progress}
  [System.IO.File]::AppendAllText($clockPath,($row|ConvertTo-Json -Depth 6 -Compress)+"`n",[System.Text.UTF8Encoding]::new($false))
  $archive=(& $clockHdc -t $Device shell "cat $clockFiles/levo-test-report.json") -join "`n"
  try{$report=$archive|ConvertFrom-Json;if($report.lastRunId -eq $RunId){break}}catch{}
  Start-Sleep -Seconds 20
}
Write-Output 'Clock observation finished; no device settings changed.'
