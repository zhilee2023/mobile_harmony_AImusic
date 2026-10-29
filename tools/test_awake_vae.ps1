param(
    [Parameter(Mandatory=$true)][ValidateSet(64,256)][int]$Frames,
    [Parameter(Mandatory=$true)][ValidateRange(1,2)][int]$Repeat
)
$ErrorActionPreference = 'Stop'
$levoRoot = Split-Path -Parent $PSScriptRoot
$levoHdc = 'C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$levoName = "phone-vae-awake-$Frames-low-6-high-r$Repeat"
$levoReportDir = Join-Path $levoRoot 'reports'
if (Test-Path -LiteralPath (Join-Path $levoReportDir "$levoName.log")) { throw 'Preserve existing run; use a new experiment name instead of overwriting' }
$levoBefore = & $levoHdc shell 'hidumper -s PowerManagerService -a "-s"'
if ($LASTEXITCODE -ne 0) { throw 'Cannot read device state' }
$levoBefore | Set-Content -LiteralPath (Join-Path $levoReportDir "$levoName-before.log") -Encoding utf8
if (($levoBefore -join "`n") -notmatch 'Current State:\s*AWAKE\b') { throw 'Device is not awake; no benchmark started' }
$levoBinary = '/data/local/tmp/levo-vae-stream-64/levo_vae_bench-affinity'
& $levoHdc shell "sha256sum $levoBinary /data/local/tmp/levo-vae-stream-$Frames/fixture.json" |
    Set-Content -LiteralPath (Join-Path $levoReportDir "$levoName-identity.log") -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'Identity read failed' }
& $levoHdc shell "$levoBinary /data/local/tmp/levo-vae-stream-$Frames low 6 high" |
    Tee-Object -FilePath (Join-Path $levoReportDir "$levoName.log")
$levoBenchmarkExit = $LASTEXITCODE
& $levoHdc shell 'hidumper -s PowerManagerService -a "-s"' |
    Set-Content -LiteralPath (Join-Path $levoReportDir "$levoName-after.log") -Encoding utf8
if ($levoBenchmarkExit -ne 0) { throw "Benchmark transport/exit failure: $levoBenchmarkExit" }
$levoLines = @(Get-Content -LiteralPath (Join-Path $levoReportDir "$levoName.log") | Where-Object { $_.StartsWith('LEVO_VAE_REPORT ') })
if ($levoLines.Count -ne 1) { throw 'Missing unique completed benchmark record' }
$levoResult = $levoLines[0].Substring(16) | ConvertFrom-Json
if (-not $levoResult.success) { throw "Benchmark failed: $($levoResult.error)" }
Write-Output "Completed $levoName : RTF=$($levoResult.decoderOnlyComputeRtf), peakKiB=$($levoResult.sampledPeakRssKiB)"
