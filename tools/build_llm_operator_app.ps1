param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio',
      [string]$MnnRoot='C:\cosyvoice-harmony\third_party\MNN',
      [string]$MnnLibrary='C:\cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a')
$ErrorActionPreference='Stop'
$operatorProject=Join-Path (Split-Path -Parent $PSScriptRoot) 'llm-operator-app'
$operatorProfile=Join-Path $operatorProject 'build-profile.json5'
if(!(Test-Path -LiteralPath $operatorProfile)){Copy-Item -LiteralPath (Join-Path $operatorProject 'build-profile.template.json5') -Destination $operatorProfile}
if(!(Test-Path -LiteralPath $MnnLibrary) -or !(Test-Path -LiteralPath (Join-Path $MnnRoot 'include/MNN/Interpreter.hpp'))){throw 'Provide the pinned compatible ARM64 MNN source/archive'}
$operatorCmake='set(LEVO_MNN_ROOT [=['+$MnnRoot.Replace('\','/')+']=] CACHE PATH "" FORCE)'+[Environment]::NewLine+
 'set(LEVO_MNN_LIBRARY [=['+$MnnLibrary.Replace('\','/')+']=] CACHE FILEPATH "" FORCE)'+[Environment]::NewLine
[IO.File]::WriteAllText((Join-Path $operatorProject 'entry/src/main/cpp/operator-local.cmake'),$operatorCmake,[Text.UTF8Encoding]::new($false))
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk';$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr';$env:NODE_HOME=Join-Path $DevEcoRoot 'tools/node';$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
Push-Location $operatorProject
try{
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/ohpm/bin/pm-cli.js" install --all
 if($LASTEXITCODE -ne 0){throw 'OHPM dependency setup failed'}
 & "$env:NODE_HOME/node.exe" "$DevEcoRoot/tools/hvigor/bin/hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
 if($LASTEXITCODE -ne 0){throw 'LLM Operator App build failed'}
}finally{Pop-Location}
Write-Output 'Independent package: com.zhilee.llmoperatorlab. Configure its own DevEco debug signing before installation.'
