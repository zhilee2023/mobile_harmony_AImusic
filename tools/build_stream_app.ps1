param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio',
      [string]$MnnRoot='',[string]$MnnLibrary='',[switch]$PrepareOnly)
$ErrorActionPreference='Stop'
$labRoot=Split-Path -Parent $PSScriptRoot;$labProject=Join-Path $labRoot 'stream-app'
$labProfile=Join-Path $labProject 'build-profile.json5'
if(!(Test-Path -LiteralPath $labProfile)){
 Copy-Item -LiteralPath (Join-Path $labProject 'build-profile.template.json5') -Destination $labProfile
 Write-Output 'Created unsigned local profile. Configure automatic signing in DevEco for the connected phone.'
}
$labWorkspace=Split-Path -Parent $labRoot
if(!$MnnRoot){$MnnRoot=Join-Path $labWorkspace 'cosyvoice-harmony/third_party/MNN'}
if(!$MnnLibrary){$MnnLibrary=Join-Path $labWorkspace 'cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a'}
if(!(Test-Path -LiteralPath (Join-Path $MnnRoot 'include/MNN/Interpreter.hpp')) -or !(Test-Path -LiteralPath $MnnLibrary)){
 throw 'MNN dependency missing. Provide -MnnRoot and -MnnLibrary for the compatible HarmonyOS ARM64 build; this is separate from signing.'
}
$labLocal='set(LEVO_MNN_ROOT [=['+$MnnRoot.Replace('\','/')+']=] CACHE PATH "" FORCE)'+[Environment]::NewLine+
          'set(LEVO_MNN_LIBRARY [=['+$MnnLibrary.Replace('\','/')+']=] CACHE FILEPATH "" FORCE)'+[Environment]::NewLine
[IO.File]::WriteAllText((Join-Path $labProject 'entry/src/main/cpp/stream-local.cmake'),$labLocal,[Text.UTF8Encoding]::new($false))
if($PrepareOnly){Write-Output "Prepared $labProject; existing signing configuration was preserved.";return}
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk';$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr';$env:NODE_HOME=Join-Path $DevEcoRoot 'tools/node'
$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
Push-Location $labProject
try{
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/ohpm/bin/pm-cli.js" install --all
 if($LASTEXITCODE -ne 0){throw 'Dependency setup failed'}
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/hvigor/bin/hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
 if($LASTEXITCODE -ne 0){throw 'Build failed'}
}finally{Pop-Location}
