param([Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9]+$')][string]$Device)
$ErrorActionPreference='Stop'
$labRoot=Split-Path -Parent $PSScriptRoot
$labHdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
$labHap=Join-Path $labRoot 'stream-app/entry/build/default/outputs/default/entry-default-signed.hap'
if(!(Test-Path -LiteralPath $labHap)){throw 'Complete DevEco automatic signing and build the independent stream-app'}
& $labHdc -t $Device file send $labHap /data/local/tmp/levo-stream-lab.hap
if($LASTEXITCODE -ne 0){throw 'HAP transfer failed'}
$labInstall=(& $labHdc -t $Device shell 'bm install -p /data/local/tmp/levo-stream-lab.hap -r') -join "`n"
Write-Output $labInstall
if($labInstall -notmatch 'successfully'){throw 'Independent App install failed'}
& $labHdc -t $Device shell 'aa start -a EntryAbility -b com.zhilee.levostreamlab'
$labFiles='/data/app/el2/100/base/com.zhilee.levostreamlab/haps/entry/files'
$labOwner=''
for($labAttempt=0;$labAttempt -lt 15;$labAttempt++){
 $labOwner=((& $labHdc -t $Device shell "stat -c %u:%g $labFiles") -join '').Trim()
 if($labOwner -match '^\d+:\d+$'){break}
 Start-Sleep -Milliseconds 200
}
if($labOwner -notmatch '^\d+:\d+$'){throw 'New App files directory was not initialized'}
$labSource='/data/app/el2/100/base/com.zhilee.larkstudiomusic/haps/entry/files/model-bundles'
& $labHdc -t $Device shell "mkdir -p $labFiles/model-bundles"
foreach($labId in @('larkstudio-small-0.6b-v1','larkstudio-base-v2')){
 Write-Output "Copy independent model $labId"
 & $labHdc -t $Device shell "if [ ! -d $labFiles/model-bundles/$labId ]; then cp -R $labSource/$labId $labFiles/model-bundles/$labId; fi"
 & $labHdc -t $Device shell "chown -R $labOwner $labFiles/model-bundles/$labId"
 & $labHdc -t $Device shell "chmod -R a+rX $labFiles/model-bundles/$labId"
 # Compare every copied asset against the signed-in user's source manifest.
 $labRaw=(& $labHdc -t $Device shell "cat $labFiles/model-bundles/$labId/model-bundle.json") -join "`n"
 $labManifest=$labRaw | ConvertFrom-Json
 foreach($labAsset in $labManifest.assets){
  $labRelative=[string]$labAsset.relative
  if($labRelative -notmatch '^[A-Za-z0-9_./-]+$' -or $labRelative.Contains('..')){throw 'Invalid asset path'}
  $labDigest=((& $labHdc -t $Device shell "sha256sum $labFiles/model-bundles/$labId/$labRelative") -join '')
  if(!$labDigest.StartsWith([string]$labAsset.sha256)){throw "Copied model hash mismatch: $labId/$labRelative"}
 }
 Write-Output "${labId}: all asset SHA256 verified"
}
$labStyles='/data/app/el2/100/base/com.zhilee.larkstudiomusic/haps/entry/files/clap-styles'
& $labHdc -t $Device shell "if [ -d $labStyles ]; then mkdir -p $labFiles/clap-styles; for f in $labStyles/*.npy; do [ -f \"`$f\" ] && cp -n \"`$f\" $labFiles/clap-styles/; done; chown -R $labOwner $labFiles/clap-styles; chmod -R a+rX $labFiles/clap-styles; fi"
Write-Output 'New package and models ready. Existing LarkStudio app/data were read only.'
