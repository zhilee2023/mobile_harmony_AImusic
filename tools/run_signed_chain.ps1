param(
  [Parameter(Mandatory=$true)][ValidateSet('serial','keep-llm','all','all-w4','llm-short','llm-long','chunks-5','chunks-5-tuned')][string]$Policy,
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId
)
$ErrorActionPreference='Stop'
$levoHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$levoRoot=Split-Path -Parent $PSScriptRoot
$levoOutput=Join-Path $levoRoot "reports\signed-$RunId.json"
if (Test-Path -LiteralPath $levoOutput) { throw 'Preserve existing diagnostic result; choose another RunId' }
$levoRemote='/data/app/el2/100/base/com.zhilee.levoharmony/haps/entry/files'
$levoKey=@{'serial'='chainSerial';'keep-llm'='chainKeepLlm';'all'='chainAll';'all-w4'='chainAllW4';'llm-short'='llmShort';'llm-long'='llmLong';'chunks-5'='chunks5';'chunks-5-tuned'='chunks5Tuned'}[$Policy]
$levoReceipt=[ordered]@{runId=$RunId;policy=$Policy;hapSha256=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'app\entry\build\default\outputs\default\entry-default-signed.hap') -Algorithm SHA256).Hash.ToLower();chainSourceSha256=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'native-bench\ChainBench.cpp') -Algorithm SHA256).Hash.ToLower();napiSourceSha256=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'app\entry\src\main\cpp\napi_init.cpp') -Algorithm SHA256).Hash.ToLower();scope='Local package installed by HDC immediately before the experiment group'}
$levoReceipt | ConvertTo-Json | Set-Content -LiteralPath ($levoOutput.Replace('.json','-receipt.json')) -Encoding utf8
& $levoHdc shell 'aa force-stop com.zhilee.levoharmony'
& $levoHdc shell "aa start -a EntryAbility -b com.zhilee.levoharmony --ps levoDiagnostic chain-$Policy --ps levoDiagnosticRunId $RunId"
$levoPid=(& $levoHdc shell 'pidof com.zhilee.levoharmony').Trim()
if ($levoPid -notmatch '^\d+$') { throw 'No unique live App process after launch' }
Write-Output "Started $RunId; PID=$levoPid"
$levoWatch=[System.Diagnostics.Stopwatch]::StartNew()
$levoLastStage=''
while ($levoWatch.Elapsed.TotalSeconds -lt 240) {
  $levoRaw=(& $levoHdc shell "cat $levoRemote/levo-test-report.json") -join "`n"
  try { $levoReport=$levoRaw | ConvertFrom-Json -AsHashtable } catch { $levoReport=$null }
  if ($null -ne $levoReport -and $levoReport.lastRunId -eq $RunId) {
    [System.IO.File]::WriteAllText($levoOutput,$levoRaw,[System.Text.UTF8Encoding]::new($false))
    $levoCase=$levoReport[$levoKey]
    $levoSummary=[ordered]@{run=$RunId;success=$levoCase.success;error=$levoCase.error;rtf=$levoCase.pipelineRtfThroughPcm;pipelinePeakMiB=($levoCase.pipelinePeakRssKiB/1024);allPeakMiB=($levoCase.sampledPeakRssKiB/1024);decodeTps=$levoCase.decodeTokensPerSecond;warmDecodeTps=$levoCase.warmDecodeTokensPerSecond;numericParity=$levoCase.strictNumericalParityPassed}
    Write-Output ($levoSummary | ConvertTo-Json -Compress)
    if (-not $levoCase.success) { throw "App diagnostic stopped: $($levoCase.error)" }
    exit 0
  }
  $levoLive=(& $levoHdc shell 'pidof com.zhilee.levoharmony').Trim()
  if ($levoLive -ne $levoPid) { throw 'App process ended or changed; no automatic restart' }
  $levoProgress=@(& $levoHdc shell "tail -n 1 $levoRemote/three-module/last-chain-progress.jsonl")
  try {
    $levoStage=($levoProgress -join "`n") | ConvertFrom-Json
    if ($levoStage.stage -ne $levoLastStage) {
      $levoLastStage=$levoStage.stage
      Write-Output "$RunId : $levoLastStage ; RSS=$([math]::Round($levoStage.rssKiB/1024,1))MiB ; t=$($levoStage.seconds)"
    }
  } catch { }
  Start-Sleep -Seconds 3
}
throw 'Observation timeout; do not restart automatically. Inspect the existing App PID and progress file.'
