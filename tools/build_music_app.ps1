param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio',
      [string]$MnnRoot='',[string]$MnnLibrary='')
$ErrorActionPreference='Stop'
$musicProject=Join-Path (Split-Path -Parent $PSScriptRoot) 'music-app'
$musicWorkspace=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if(!$MnnRoot){$MnnRoot=Join-Path $musicWorkspace 'cosyvoice-harmony\third_party\MNN'}
if(!$MnnLibrary){$MnnLibrary=Join-Path $musicWorkspace 'cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a'}
if(!(Test-Path -LiteralPath "$MnnRoot\include\MNN\Interpreter.hpp") -or !(Test-Path -LiteralPath $MnnLibrary)){throw 'Provide -MnnRoot and -MnnLibrary for your local pinned MNN build'}
$musicLocal='set(LEVO_MNN_ROOT [=['+$MnnRoot.Replace('\','/')+']=] CACHE PATH "" FORCE)'+[Environment]::NewLine+
            'set(LEVO_MNN_LIBRARY [=['+$MnnLibrary.Replace('\','/')+']=] CACHE FILEPATH "" FORCE)'+[Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $musicProject 'entry\src\main\cpp\music-local.cmake'),$musicLocal,[System.Text.UTF8Encoding]::new($false))
if (!(Test-Path -LiteralPath (Join-Path $musicProject 'build-profile.json5'))) {
  Copy-Item -LiteralPath (Join-Path $musicProject 'build-profile.template.json5') -Destination (Join-Path $musicProject 'build-profile.json5')
}
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk'
$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr'
$env:NODE_HOME=Join-Path $DevEcoRoot 'tools\node'
$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
Push-Location $musicProject
try {
  & "$env:NODE_HOME\node.exe" "$DevEcoRoot\tools\ohpm\bin\pm-cli.js" install --all
  if ($LASTEXITCODE -ne 0) { throw 'Model App dependency setup failed' }
  & "$env:NODE_HOME\node.exe" "$DevEcoRoot\tools\hvigor\bin\hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
  if ($LASTEXITCODE -ne 0) { throw 'Model App build failed' }
} finally { Pop-Location }
