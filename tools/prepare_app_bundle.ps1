$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$out=Join-Path $projectRoot 'dist\vae-stream-256'
New-Item -ItemType Directory -Path $out -Force > $null
$names=@('fixture.json','vae-256-fp32.mnn','vae-256-fp32.mnn.weight',
  'latent-0.f32','latent-1.f32','latent-2.f32','pcm-0.f32','pcm-1.f32','pcm-2.f32',
  'emit-0.f32','emit-1.f32','emit-2.f32')
foreach($name in $names){
  $source=Join-Path $projectRoot ('models\vae-stream-256\'+$name)
  if(!(Test-Path -LiteralPath $source)){throw ('Missing validated fixture: '+$name)}
  Copy-Item -LiteralPath $source -Destination (Join-Path $out $name) -Force
}
$hap=Join-Path $projectRoot 'app\entry\build\default\outputs\default\entry-default-unsigned.hap'
if(!(Test-Path -LiteralPath $hap)){throw 'Build the diagnostic HAP first'}
Copy-Item -LiteralPath $hap -Destination (Join-Path $projectRoot 'dist\LeVo-Harmony-v0.1.3-unsigned.hap') -Force
Write-Output ('Prepared '+$names.Count+' VAE diagnostic files: '+$out)
Write-Output 'HAP is unsigned. Configure automatic signing in DevEco yourself, then run on your phone.'
