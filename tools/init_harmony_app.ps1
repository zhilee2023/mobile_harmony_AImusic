param([ValidateSet('music-app','stream-app')][string]$App='stream-app',
      [ValidatePattern('^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$')][string]$BundleName,
      [string]$DisplayName)
$ErrorActionPreference='Stop'
$initRoot=Split-Path -Parent $PSScriptRoot
$initProject=Join-Path $initRoot $App
foreach($initRequired in @('build-profile.template.json5','AppScope/app.json5','entry/src/main/module.json5','oh-package.json5','hvigorfile.ts')){
 if(!(Test-Path -LiteralPath (Join-Path $initProject $initRequired))){throw "Incomplete checkout: $App/$initRequired is missing. Clone or download the complete repository."}
}
$initProfile=Join-Path $initProject 'build-profile.json5'
$initMetadataPath=Join-Path $initProject 'AppScope/app.json5'
$initMetadata=Get-Content -LiteralPath $initMetadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
$initChangeBundle=$PSBoundParameters.ContainsKey('BundleName') -and $BundleName -ne $initMetadata.app.bundleName
if($PSBoundParameters.ContainsKey('BundleName') -and ($BundleName.Length -lt 7 -or $BundleName.Length -gt 127)){throw 'BundleName must contain7-127 characters.'}
if($PSBoundParameters.ContainsKey('DisplayName') -and ([string]::IsNullOrWhiteSpace($DisplayName) -or $DisplayName.Length -gt 80)){throw 'DisplayName must contain1-80 nonblank characters.'}
if($initChangeBundle -and (Test-Path -LiteralPath $initProfile)){
 try{$initCurrent=Get-Content -LiteralPath $initProfile -Raw -Encoding UTF8 | ConvertFrom-Json}
 catch{throw 'Cannot inspect local signing config. Use a fresh clone to initialize a new App identity.'}
 if(@($initCurrent.app.signingConfigs).Count -gt 0){throw 'This project already has local signing material. Create a fresh clone for the independent App; existing signed project was not renamed.'}
}
$initResourceEdits=@()
if($PSBoundParameters.ContainsKey('DisplayName')){
 foreach($initResourceRoot in @('AppScope/resources','entry/src/main/resources')){
  $initResourcePath=Join-Path $initProject $initResourceRoot
  if(!(Test-Path -LiteralPath $initResourcePath)){continue}
  foreach($initResource in Get-ChildItem -LiteralPath $initResourcePath -Filter string.json -Recurse -File){
   $initStrings=Get-Content -LiteralPath $initResource.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
   $initChanged=$false
   foreach($initString in $initStrings.string){if($initString.name -in @('app_name','EntryAbility_label')){$initString.value=$DisplayName;$initChanged=$true}}
   if($initChanged){$initResourceEdits+=@(@{path=$initResource.FullName;content=$initStrings})}
  }
 }
 if($initResourceEdits.Count -eq 0){throw 'No app_name resource was found; no identity files changed.'}
}
if(Test-Path -LiteralPath $initProfile){Write-Output 'Kept existing local build/signing configuration.'}
else{
 Copy-Item -LiteralPath (Join-Path $initProject 'build-profile.template.json5') -Destination $initProfile
 Write-Output 'Created unsigned project configuration from the public template.'
}
if($initChangeBundle){
 $initMetadata.app.bundleName=$BundleName
 [IO.File]::WriteAllText($initMetadataPath,($initMetadata | ConvertTo-Json -Depth 30)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
}
foreach($initEdit in $initResourceEdits){[IO.File]::WriteAllText($initEdit.path,($initEdit.content | ConvertTo-Json -Depth 30)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))}
Write-Output "App bundle: $($initMetadata.app.bundleName)"
if($initChangeBundle){Write-Output 'New App identity prepared. Generate its own debug signature in DevEco before installation.'}
Write-Output "Open this folder in DevEco: $initProject"
Write-Output 'Connect the target phone and configure Signing Configs. Native MNN dependencies are configured separately.'
