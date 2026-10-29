param(
  [string]$Python='C:\Users\lizhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe',
  [string]$Hdc='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe'
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$binary=Join-Path $projectRoot 'native-build\levo_block_bench'
if (!(Test-Path -LiteralPath $binary)) { throw 'Build native-bench first.' }
function Invoke-HdcChecked([string[]]$Arguments) {
  & $Hdc @Arguments
  if ($LASTEXITCODE -ne 0) { throw ('HDC command failed: '+($Arguments -join ' ')) }
}
foreach($frames in @(32,256)) {
  $suffix=if($frames -eq 32){''}else{'-'+$frames}
  $remote='/data/local/tmp/levo-block-bench'+$suffix
  $fixture=Join-Path $projectRoot ('models\phone-block'+$suffix)
  $models=Join-Path $projectRoot ('models\dit-block'+$suffix)
  if (!(Test-Path -LiteralPath (Join-Path $fixture 'fixture.json'))) { throw 'Prepare both fixed fixtures first.' }
  Invoke-HdcChecked -Arguments @('shell','mkdir','-p',$remote)
  Invoke-HdcChecked -Arguments @('file','send',$binary,($remote+'/levo_block_bench'))
  foreach($file in Get-ChildItem -LiteralPath $fixture -File) {
    Invoke-HdcChecked -Arguments @('file','send',$file.FullName,($remote+'/'+$file.Name))
  }
  $names=@('real-block0-w32.mnn','real-block0-w8.mnn','real-block0-w4.mnn')
  if($frames -eq 32){$names+=@('w4-g16.mnn')}
  foreach($name in $names){foreach($extension in @('','.weight')){
    Invoke-HdcChecked -Arguments @('file','send',(Join-Path $models ($name+$extension)),($remote+'/'+$name+$extension))
  }}
  Invoke-HdcChecked -Arguments @('shell','chmod','755',($remote+'/levo_block_bench'))
  $profiles=@(@('real-block0-w32.mnn','high','w32-high'),
              @('real-block0-w8.mnn','high','w8-high'),
              @('real-block0-w4.mnn','high','w4-high'),
              @('real-block0-w32.mnn','low','w32-low'))
  if($frames -eq 32){$profiles+=,@('w4-g16.mnn','high','w4-g16-high')}
  foreach($profile in $profiles){
    $logSuffix=if($frames -eq 32){'-verified'}else{'-256'}
    $log=Join-Path $projectRoot ('reports\phone-block-'+$profile[2]+$logSuffix+'.log')
    & $Hdc shell ($remote+'/levo_block_bench '+$remote+' '+$profile[0]+' '+$profile[1]+' 4') *> $log
    if($LASTEXITCODE -ne 0){Get-Content -LiteralPath $log;throw 'Phone block benchmark failed.'}
  }
}
& $Python -X utf8 (Join-Path $PSScriptRoot 'merge_phone_validation.py') --hdc $Hdc
if($LASTEXITCODE -ne 0){throw 'Unified report verification failed.'}
