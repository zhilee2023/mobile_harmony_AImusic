param([string]$MnnRoot='C:\cosyvoice-harmony\third_party\MNN',
      [string]$MnnStaticLibrary='C:\cosyvoice-harmony\app\entry\.cxx\default\default\debug\arm64-v8a\mnn\libMNN.a')
$ErrorActionPreference='Stop'
$native='C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\native'
$cmake=Join-Path $native 'build-tools\cmake\bin\cmake.exe'
$options=@('-G','Ninja',
 '-DCMAKE_TOOLCHAIN_FILE=C:\Program Files\Huawei\DevEco Studio\sdk\default\hms\native\build\cmake\hmos.toolchain.bisheng.cmake',
 "-DOHOS_SDK_NATIVE=$native","-DCMAKE_MAKE_PROGRAM=$native\build-tools\cmake\bin\ninja.exe",
 '-DCMAKE_BUILD_TYPE=Release','-DOHOS_ARCH=arm64-v8a','-DOHOS_STL=c++_static',
 "-DMNN_ROOT=$MnnRoot","-DMNN_STATIC_LIBRARY=$MnnStaticLibrary")
& $cmake -S "$PSScriptRoot\native-bench" -B "$PSScriptRoot\native-build" @options
if($LASTEXITCODE-ne 0){throw 'Native benchmark configuration failed'}
& $cmake --build "$PSScriptRoot\native-build" --parallel 4
if($LASTEXITCODE-ne 0){throw 'Native benchmark build failed'}
