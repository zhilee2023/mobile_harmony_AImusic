param(
  [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9-]+$')][string]$RunId,
  [ValidateSet('w4','smooth-a4','smooth-a8','smooth-noact')][string]$Variant='w4',
  [ValidateSet(4,6)][int]$Threads=4,
  [ValidateSet(8,14)][int]$AttentionMode=8,
  [ValidateSet(0,8)][int]$DynamicOption=0,
  [ValidateSet('low','high')][string]$Precision='low',
  [switch]$StageFiles
)
$ErrorActionPreference='Stop'
$levoRoot=Split-Path -Parent $PSScriptRoot
$levoHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$levoRemote='/data/local/tmp/levo-qwen-language-20261031'
$levoJson=Join-Path $levoRoot "reports\qwen-language-$RunId.json"
$levoLog=Join-Path $levoRoot "reports\qwen-language-$RunId.log"
if((Test-Path -LiteralPath $levoJson) -or (Test-Path -LiteralPath $levoLog)){throw 'Use a new RunId to preserve prior measurements'}
if($StageFiles){
  & $levoHdc shell "mkdir -p $levoRemote/$Variant $levoRemote/fixture"
  if($LASTEXITCODE -ne 0){throw 'Device staging directory failed'}
  $levoFiles=@('config.json','llm_config.json','llm.mnn','llm.mnn.weight','embeddings_bf16.bin','tokenizer.mtok','language-probe.json')
  foreach($levoName in $levoFiles){
    & $levoHdc file send (Join-Path $levoRoot "models/qwen-language-$Variant/$levoName") "$levoRemote/$Variant/$levoName"
    if($LASTEXITCODE -ne 0){throw "Model transfer failed: $levoName"}
  }
  foreach($levoFile in (Get-ChildItem -LiteralPath (Join-Path $levoRoot 'models/qwen-language-fixture') -File)){
    & $levoHdc file send $levoFile.FullName "$levoRemote/fixture/$($levoFile.Name)"
    if($LASTEXITCODE -ne 0){throw 'Fixture transfer failed'}
  }
  & $levoHdc file send (Join-Path $levoRoot 'native-build/levo_qwen_language_bench') "$levoRemote/levo_qwen_language_bench"
  if($LASTEXITCODE -ne 0){throw 'Binary transfer failed'}
  & $levoHdc shell "chmod 700 $levoRemote/levo_qwen_language_bench"
}
$levoBinaryHash=(Get-FileHash -LiteralPath (Join-Path $levoRoot 'native-build/levo_qwen_language_bench') -Algorithm SHA256).Hash.ToLower()
$levoDeviceHash=(& $levoHdc shell "sha256sum $levoRemote/levo_qwen_language_bench") -join ''
if($levoDeviceHash -notlike "$levoBinaryHash *"){throw 'Device binary does not match local executable'}
$levoManifest=Get-Content -LiteralPath (Join-Path $levoRoot "models/qwen-language-$Variant/language-probe.json") -Raw | ConvertFrom-Json -AsHashtable
foreach($levoName in $levoManifest.files.Keys){
  $levoExpected=$levoManifest.files[$levoName].sha256
  $levoActual=(& $levoHdc shell "sha256sum $levoRemote/$Variant/$levoName") -join ''
  if($levoActual -notlike "$levoExpected *"){throw "Device model hash mismatch: $levoName"}
}
foreach($levoFile in (Get-ChildItem -LiteralPath (Join-Path $levoRoot 'models/qwen-language-fixture') -File)){
  $levoExpected=(Get-FileHash -LiteralPath $levoFile.FullName -Algorithm SHA256).Hash.ToLower()
  $levoActual=(& $levoHdc shell "sha256sum $levoRemote/fixture/$($levoFile.Name)") -join ''
  if($levoActual -notlike "$levoExpected *"){throw "Device fixture hash mismatch: $($levoFile.Name)"}
}
$levoStart=(& $levoHdc shell 'hidumper -s PowerManagerService -a "-s"') -join "`n"
if($levoStart -notmatch 'Current State: AWAKE'){throw 'Keep phone awake for comparable timing'}
& $levoHdc shell "timeout 240 $levoRemote/levo_qwen_language_bench $levoRemote $Variant $Threads $AttentionMode $DynamicOption $Precision" | Tee-Object -FilePath $levoLog | ForEach-Object {
  if($_ -like 'LEVO_CHAIN_PROGRESS *'){
    $levoStage=$_.Substring(20) | ConvertFrom-Json
    Write-Output "$RunId : $($levoStage.stage); t=$([math]::Round($levoStage.seconds,2))s; RSS=$([math]::Round($levoStage.rssKiB/1024,1))MiB"
  }
}
$levoRaw=Get-Content -LiteralPath $levoLog | Where-Object {$_ -like 'LEVO_LANGUAGE_REPORT *'} | Select-Object -Last 1
if(!$levoRaw){throw 'No structured result; preserve raw log, no automatic retry'}
$levoResult=$levoRaw.Substring(21) | ConvertFrom-Json -AsHashtable
$levoEnd=(& $levoHdc shell 'hidumper -s PowerManagerService -a "-s"') -join "`n"
$levoOffBefore=[regex]::Match($levoStart,'Last Screen Off:\s*(\d+)').Groups[1].Value
$levoOffAfter=[regex]::Match($levoEnd,'Last Screen Off:\s*(\d+)').Groups[1].Value
$levoStable=($levoEnd -match 'Current State: AWAKE') -and ($levoOffBefore -ne '') -and ($levoOffBefore -eq $levoOffAfter)
[ordered]@{runId=$RunId;date='2026-10-31';binarySha256=$levoBinaryHash;displayStable=$levoStable;screenBefore=$levoStart;screenAfter=$levoEnd;result=$levoResult} | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $levoJson -Encoding utf8
$levoCompactRounds=@($levoResult.rounds | ForEach-Object {$_ | Select-Object round,prefillSeconds,prefillTokensPerSecond,decodeTokensPerSecond,teacherGreedyAgreement,referenceContinuationMeanNll})
[ordered]@{success=$levoResult.success;error=$levoResult.error;variant=$levoResult.variant;threads=$levoResult.threads;kvBits=$levoResult.kvBits;dynamicOption=$levoResult.dynamicOption;peakRssMiB=$levoResult.sampledPeakRssKiB/1024;displayStable=$levoStable;rounds=$levoCompactRounds} | ConvertTo-Json -Depth 6 -Compress | Write-Output
if(!$levoResult.success){throw 'Language probe failed; result retained'}
if(!$levoStable){Write-Warning 'Display state changed; exclude timing from speed comparisons'}
