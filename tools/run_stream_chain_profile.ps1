param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9]+$')][string]$Device,
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,70}$')][string]$RunId,
 [ValidateRange(25,1500)][int]$Frames=375,
 [ValidateSet(0,1)][int]$Model=0,
 [Parameter(Mandatory=$true)][string]$SeedResult,
 [string]$DevEcoRoot='C:\Program Files\Huawei\DevEco Studio'
)
$ErrorActionPreference='Stop'
$profileRoot=Split-Path -Parent $PSScriptRoot
$profileHdc=Join-Path $DevEcoRoot 'sdk/default/openharmony/toolchains/hdc.exe'
$profileCmake=Join-Path $DevEcoRoot 'sdk/default/openharmony/native/build-tools/cmake/bin/cmake.exe'
$profileOutput=Join-Path $profileRoot "reports/local/$RunId"
if(Test-Path -LiteralPath $profileOutput){throw 'Preserve previous measurements; use a new RunId'}
$profileSeed=Get-Content -LiteralPath $SeedResult -Raw -Encoding UTF8 | ConvertFrom-Json
if([string]::IsNullOrWhiteSpace([string]$profileSeed.lyrics)){throw 'SeedResult must contain real App test lyrics'}
$profileFiles='/data/app/el2/100/base/com.zhilee.levostreamlab/haps/entry/files'
$profileBundle=if($Model -eq 0){'larkstudio-small-0.6b-v1'}else{'larkstudio-base-v2'}
$profileRemote="/data/local/tmp/levo-chain-profile-$RunId"
$profileNpy=([string]$profileSeed.npy).Replace('/data/storage/el2/base/haps/entry/files',$profileFiles)
if($profileNpy -notmatch '^/[A-Za-z0-9_./-]+\.npy$' -or $profileNpy.Contains('..')){throw 'Style path must be an existing safe phone path'}
function Invoke-ProfileHdc([string[]]$ProfileArguments){
 $profileLines=& $profileHdc -t $Device @ProfileArguments
 if($LASTEXITCODE -ne 0){throw 'HDC command failed; keep partial output for diagnosis'}
 return ($profileLines -join "`n")
}
$profilePower=Invoke-ProfileHdc @('shell','hidumper -s PowerManagerService -a -a')
if($profilePower -notmatch 'Current State: AWAKE'){throw 'Manually light/unlock the phone before profiling'}
$profileOffBefore=[regex]::Match($profilePower,'Last Screen Off:\s*(\d+)').Groups[1].Value
$profileActive=Invoke-ProfileHdc @('shell',"test -f $profileFiles/model-bundles/$profileBundle/model-bundle.json && test -f $profileNpy && echo PROFILE_INPUTS_PRESENT")
if($profileActive -notmatch 'PROFILE_INPUTS_PRESENT'){throw 'Phone model/style not present'}
& $profileCmake --build (Join-Path $profileRoot 'native-build') --target levo_stream_chain_profile -j 2
if($LASTEXITCODE -ne 0){throw 'Native profiler build failed'}
$null=New-Item -ItemType Directory -Path $profileOutput
[IO.File]::WriteAllText((Join-Path $profileOutput 'power-before.txt'),$profilePower,[Text.UTF8Encoding]::new($false))
$profileCreated=Invoke-ProfileHdc @('shell',"test ! -e $profileRemote && mkdir -p $profileRemote/root && ln -s $profileFiles/model-bundles $profileRemote/root/model-bundles && echo PROFILE_CREATED")
if($profileCreated -notmatch 'PROFILE_CREATED'){throw 'Remote path exists or cannot be prepared; choose a new RunId'}
$profileCrop=if($Model -eq 0){"$profileFiles/experimental/music-head-r1"}else{''}
$profileRequest=@{lyrics=[string]$profileSeed.lyrics;npy=$profileNpy;resident=($Model -eq 0);cfg=$true;ditSteps=2;
 prefetch=$false;lmThreads=4;ditThreads=6;vaeThreads=6;croppedLm=$profileCrop}
$profileRequestFile=Join-Path $profileOutput 'request.json'
[IO.File]::WriteAllText($profileRequestFile,($profileRequest | ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
$profileBinary=Join-Path $profileRoot 'native-build/levo_stream_chain_profile'
$null=Invoke-ProfileHdc @('file','send',$profileBinary,"$profileRemote/probe")
$null=Invoke-ProfileHdc @('file','send',$profileRequestFile,"$profileRemote/request.json")
$null=Invoke-ProfileHdc @('shell',"chmod 755 $profileRemote/probe")
$profileHashes=Invoke-ProfileHdc @('shell',"sha256sum $profileRemote/probe $profileRemote/request.json")
foreach($profileFile in @($profileBinary,$profileRequestFile)){
 $profileHash=(Get-FileHash -LiteralPath $profileFile -Algorithm SHA256).Hash.ToLowerInvariant()
 if(!$profileHashes.Contains($profileHash)){throw 'Transferred profiler/request hash mismatch'}
}
[IO.File]::WriteAllText((Join-Path $profileOutput 'transfer-hashes.txt'),$profileHashes,[Text.UTF8Encoding]::new($false))
$profileMeta=@{schemaVersion=1;runId=$RunId;frames=$Frames;model=$Model;modeOrder=@('baseline','operators');
 modelId=$profileBundle;
 binarySha256=(Get-FileHash -LiteralPath $profileBinary -Algorithm SHA256).Hash.ToLowerInvariant();
 sourceGitHead=(& git -C $profileRoot rev-parse HEAD);sourceWorkingTreeDiff=(& git -C $profileRoot diff --stat -- runtime/include/levo/experimental native-bench/StreamLabChainProfile.cpp);
 scope='Actual CLI chain using App weights, no playback, no foreground App RTF or cold file-cache claim';
 requestSha256=(Get-FileHash -LiteralPath $profileRequestFile -Algorithm SHA256).Hash.ToLowerInvariant()}
$profileMeta.sourceSha256=@{}
foreach($profileSource in @('native-bench/StreamLabChainProfile.cpp','runtime/include/levo/experimental/StreamLab.hpp','runtime/include/levo/experimental/StreamLabProfile.hpp','tools/run_stream_chain_profile.ps1')){
 $profileMeta.sourceSha256[$profileSource]=(Get-FileHash -LiteralPath (Join-Path $profileRoot $profileSource) -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach($profileMode in @('baseline','operators')){
 Write-Output "Running $profileMode, $Frames frames / $($Frames/25) seconds requested"
 $profileLog=Invoke-ProfileHdc @('shell',"timeout 900 $profileRemote/probe $profileRemote/root $Model $Frames $profileRemote/request.json $profileRemote/$profileMode $profileMode")
 [IO.File]::WriteAllText((Join-Path $profileOutput "$profileMode.log"),$profileLog,[Text.UTF8Encoding]::new($false))
 $null=Invoke-ProfileHdc @('file','recv',"$profileRemote/$profileMode",(Join-Path $profileOutput $profileMode))
 $profileResult=Get-Content -LiteralPath (Join-Path $profileOutput "$profileMode/profile.json") -Raw -Encoding UTF8 | ConvertFrom-Json
 if(!$profileResult.success){throw "The $profileMode chain failed; raw results preserved"}
 $profileDuring=Invoke-ProfileHdc @('shell','hidumper -s PowerManagerService -a -a')
 [IO.File]::WriteAllText((Join-Path $profileOutput "power-after-$profileMode.txt"),$profileDuring,[Text.UTF8Encoding]::new($false))
 if($profileDuring -notmatch 'Current State: AWAKE' -or [regex]::Match($profileDuring,'Last Screen Off:\s*(\d+)').Groups[1].Value -ne $profileOffBefore){
  throw 'Phone slept during profiling; results retained but not accepted as awake speed measurements'
 }
 Write-Output "$profileMode completed: $($profileResult.generation.elapsedSeconds) seconds, RSS $($profileResult.generation.rssPeakMiB) MiB"
}
$profileMeta.screenAwakeBeforeAndAfter=$true
$profileMeta.lastScreenOffUnchanged=$true
[IO.File]::WriteAllText((Join-Path $profileOutput 'run-metadata.json'),($profileMeta | ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
Write-Output "Raw paired results: $profileOutput"
