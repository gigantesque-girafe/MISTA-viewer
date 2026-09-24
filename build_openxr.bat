@echo off
REM build_openxr.bat - Build SIBR remote viewer targets (v4.2, desktop)
REM
REM Builds only the two viewers needed for the live remote-Gaussian VR/desktop
REM pipeline:
REM   v4.2    - SIBR_remoteGaussianOpenXRv4_2_app (OpenXR stereo VR viewer,
REM             needs a headset + runtime to run)
REM   desktop - SIBR_remoteGaussianDesktopV42_app (mono desktop-window viewer,
REM             same v4.2 CUDA IPC transport and GaussianLiveViewV42, no OpenXR.
REM             Runs against the unchanged Python server render_vr_v4_2.py, port 6012)
REM
REM Single file for both supported GPU generations - pick one with --gpu:
REM   --gpu turing     - compute capability 7.5  (e.g. RTX 20-series)
REM   --gpu blackwell  - compute capability 12.0 (e.g. RTX 50-series)
REM
REM Uses VS BuildTools vcvars64 + CUDA + NMake.
REM --allow-unsupported-compiler bypasses the CUDA/cl.exe version check.
REM
REM Usage:
REM   build_openxr.bat --gpu blackwell                        - build v4.2 + desktop
REM   build_openxr.bat --gpu turing --target v4.2             - build only v4.2
REM   build_openxr.bat --gpu blackwell --target desktop       - build only desktop
REM   build_openxr.bat --gpu blackwell --libtorch C:\libtorch - use a custom LibTorch
REM
REM ---------------------------------------------------------------------------
REM Machine-specific paths - edit these for your setup before building.
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

REM Repo root is auto-detected from this script's own location (%~dp0), so the
REM script works regardless of where the repo is cloned.
set "REPO_ROOT=%~dp0"
set "SIBR_SRC=%REPO_ROOT%submodules\sibr-core"
set "SIBR_BUILD=%REPO_ROOT%submodules\sibr-core\build"
set "SIBR_INSTALL=%REPO_ROOT%submodules\sibr-core\install"

REM CUDA 12.8 can target both Turing (sm_75) and Blackwell (sm_120) from the
REM same toolkit, so one CUDA_DIR covers both --gpu choices below.
set "CUDA_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

set "VS_VCVARS=C:\PROGRA~2\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM sibr_core requires cmake >= 3.22; point this at a cmake new enough
REM (e.g. a conda env's Library\bin, or pip cmake in an env's Scripts dir).
set "CMAKE_BIN=C:\TODO\path\to\cmake\bin"

REM No bundled default - LibTorch must be supplied with --libtorch.
set "DEFAULT_LIBTORCH="

REM ---------------------------------------------------------------------------
REM -- Parse arguments --
REM ---------------------------------------------------------------------------
set "LIBTORCH_PATH="
set "TARGET_FILTER="
set "GPU_CHOICE="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--libtorch" (
    set "LIBTORCH_PATH=%~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--target" (
    set "TARGET_FILTER=%~2"
    shift & shift & goto parse_args
)
if /i "%~1"=="--gpu" (
    set "GPU_CHOICE=%~2"
    shift & shift & goto parse_args
)
echo [WARN] Unknown argument: %~1
shift & goto parse_args
:args_done

REM Fall back to the env's bundled LibTorch when none was supplied.
if "%LIBTORCH_PATH%"=="" set "LIBTORCH_PATH=%DEFAULT_LIBTORCH%"

REM -- Resolve GPU architecture --
if /i "%GPU_CHOICE%"=="turing" (
    set "CUDA_ARCH=75"
    set "TORCH_ARCH=7.5"
) else if /i "%GPU_CHOICE%"=="blackwell" (
    set "CUDA_ARCH=120"
    set "TORCH_ARCH=12.0"
) else (
    echo [ERROR] --gpu ^<turing^|blackwell^> is required.
    echo         turing    = compute capability 7.5  ^(RTX 20-series^)
    echo         blackwell = compute capability 12.0 ^(RTX 50-series^)
    exit /b 1
)

REM -- Decide which targets to build --
set BUILD_V42=0
set BUILD_DESKTOP=0

if "%TARGET_FILTER%"=="" (
    set BUILD_V42=1
    set BUILD_DESKTOP=1
) else if "%TARGET_FILTER%"=="v4.2" ( set BUILD_V42=1
) else if /i "%TARGET_FILTER%"=="desktop" ( set BUILD_DESKTOP=1
) else (
    echo [ERROR] Unknown --target "%TARGET_FILTER%". Use v4.2 or desktop.
    exit /b 1
)

REM Both targets compile GaussianLiveViewV42.cpp (LibTorch Color MLP), so both
REM need LibTorch regardless of OpenXR.
if "%LIBTORCH_PATH%"=="" (
    echo [ERROR] --libtorch ^<path^> is required to build v4.2 / desktop.
    exit /b 1
)

REM -- Validate prerequisites --
if not exist "%VS_VCVARS%" (
    echo [ERROR] VS BuildTools not found: %VS_VCVARS%
    exit /b 1
)
if not exist "%CUDA_DIR%/bin/nvcc.exe" (
    echo [ERROR] CUDA not found: %CUDA_DIR%/bin/nvcc.exe
    exit /b 1
)
if not exist "%LIBTORCH_PATH%\share\cmake\Torch\TorchConfig.cmake" (
    echo [ERROR] LibTorch not found at: %LIBTORCH_PATH%
    echo         Expected: %LIBTORCH_PATH%\share\cmake\Torch\TorchConfig.cmake
    exit /b 1
)

REM -- Load VS environment --
echo [INFO] Loading VS BuildTools environment ...
call "%VS_VCVARS%"
set "PATH=%CMAKE_BIN%;%CUDA_DIR%/bin;%PATH%"

echo [INFO] Compiler versions:
cl 2>&1 | findstr /i "Version"
nvcc --version 2>&1 | findstr /i "release"
cmake --version 2>&1 | findstr /i "version"
echo [INFO] GPU target: %GPU_CHOICE% (compute capability %TORCH_ARCH%)

REM -- Wipe stale cache if generator changed --
if exist "%SIBR_BUILD%\CMakeCache.txt" (
    findstr /i "NMake" "%SIBR_BUILD%\CMakeCache.txt" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [INFO] Removing stale build cache ...
        rmdir /s /q "%SIBR_BUILD%"
    )
)

REM -- CMake configure --
echo.
echo [INFO] Configuring ...
echo.

cmake -S "%SIBR_SRC%" -B "%SIBR_BUILD%" ^
    -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    "-DCMAKE_CUDA_COMPILER=%CUDA_DIR%/bin/nvcc.exe" ^
    "-DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR%" ^
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" ^
    "-DCMAKE_INSTALL_PREFIX=%SIBR_INSTALL%" ^
    -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
    -DTORCH_CUDA_ARCH_LIST=%TORCH_ARCH% ^
    -DBUILD_IBR_BASIC=OFF ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    "-DCMAKE_CXX_FLAGS=/arch:AVX2" ^
    -DCMAKE_PREFIX_PATH="%LIBTORCH_PATH%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] cmake configure failed.
    exit /b %ERRORLEVEL%
)

REM -- Build and install each target --
set BUILT=

if "%BUILD_V42%"=="1" (
    echo.
    echo [INFO] Building v4.2: SIBR_remoteGaussianOpenXRv4_2_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussianOpenXRv4_2_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] v4.2 build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussianOpenXRv4_2_app_install
    set "BUILT=%BUILT% v4.2"
)

if "%BUILD_DESKTOP%"=="1" (
    echo.
    echo [INFO] Building desktop: SIBR_remoteGaussianDesktopV42_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussianDesktopV42_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] desktop build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussianDesktopV42_app_install
    set "BUILT=%BUILT% desktop"
)

echo.
echo ============================================================
echo  Build succeeded: %BUILT%
echo  GPU target: %GPU_CHOICE% (compute capability %TORCH_ARCH%)
echo  Output: %SIBR_INSTALL%\bin\
echo ============================================================

endlocal
