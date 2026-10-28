param([string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio')
$ErrorActionPreference='Stop'
$env:DEVECO_SDK_HOME=Join-Path $DevEcoRoot 'sdk'
$env:JAVA_HOME=Join-Path $DevEcoRoot 'jbr'
$env:NODE_HOME=Join-Path $DevEcoRoot 'tools\node'
$env:PATH="$env:JAVA_HOME\bin;$env:NODE_HOME;$env:PATH"
Push-Location (Join-Path $PSScriptRoot 'app')
try {
  & "$env:NODE_HOME\node.exe" "$DevEcoRoot\tools\hvigor\bin\hvigorw.js" --mode module -p product=default -p module=entry@default -p buildMode=debug assembleHap --no-daemon
  if($LASTEXITCODE -ne 0){throw 'Harmony build failed'}
} finally { Pop-Location }
# Unsigned by design. Use DevEco automatic signing for com.zhilee.levoharmony.
# Never commit local signing material or reuse the CosyVoice bundle identity.
