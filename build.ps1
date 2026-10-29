param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio',
      [switch]$WithMnn,
      [string]$MnnRoot='C:\cosyvoice-harmony\third_party\MNN',
      [string]$MnnStaticLibrary='C:\cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a')
$ErrorActionPreference='Stop'
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk'
$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr'
$env:NODE_HOME=Join-Path $DevEcoRoot 'tools\node'
$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
if($WithMnn){
  if(!(Test-Path -LiteralPath "$MnnRoot\include\MNN\Interpreter.hpp") -or !(Test-Path -LiteralPath $MnnStaticLibrary)){throw 'Compatible MNN source/library missing'}
  $localConfig=Join-Path $PSScriptRoot 'app\entry\src\main\cpp\levo-local.cmake'
  $configText='set(LEVO_MNN_ROOT [=['+$MnnRoot.Replace('\','/')+']=] CACHE PATH "" FORCE)'+[Environment]::NewLine+
              'set(LEVO_MNN_STATIC_LIBRARY [=['+$MnnStaticLibrary.Replace('\','/')+']=] CACHE FILEPATH "" FORCE)'+[Environment]::NewLine
  [System.IO.File]::WriteAllText($localConfig,$configText,[System.Text.UTF8Encoding]::new($false))
}
Push-Location (Join-Path $PSScriptRoot 'app')
try {
  & "$env:NODE_HOME\node.exe" "$DevEcoRoot\tools\hvigor\bin\hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
  if($LASTEXITCODE -ne 0){throw 'Harmony build failed'}
} finally { Pop-Location }
# Unsigned by design. Use DevEco automatic signing for com.zhilee.levoharmony.
# Never commit local signing material or reuse the CosyVoice bundle identity.
