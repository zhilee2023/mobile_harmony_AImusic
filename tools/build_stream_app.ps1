param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio')
$ErrorActionPreference='Stop'
$labRoot=Split-Path -Parent $PSScriptRoot;$labProject=Join-Path $labRoot 'stream-app'
$labMnn='C:/cosyvoice-harmony/third_party/MNN'
$labLibrary='C:/cosyvoice-harmony/app/entry/.cxx/default/default/debug/arm64-v8a/mnn/libMNN.a'
[IO.File]::WriteAllText((Join-Path $labProject 'entry/src/main/cpp/stream-local.cmake'),('set(LEVO_MNN_ROOT "'+$labMnn+'")'+[Environment]::NewLine+'set(LEVO_MNN_LIBRARY "'+$labLibrary+'")'),[Text.UTF8Encoding]::new($false))
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk';$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr';$env:NODE_HOME=Join-Path $DevEcoRoot 'tools/node'
$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
Push-Location $labProject
try{
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/ohpm/bin/pm-cli.js" install --all
 if($LASTEXITCODE -ne 0){throw 'Dependency setup failed'}
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/hvigor/bin/hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
 if($LASTEXITCODE -ne 0){throw 'Build failed'}
}finally{Pop-Location}
