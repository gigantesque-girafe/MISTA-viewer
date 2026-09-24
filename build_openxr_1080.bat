@echo off
REM build_openxr_1080.bat - Build SIBR remote viewer targets (v3, v4, v4.2, desktop, desktopv43)
REM   Machine-local variant of build_openxr.bat for the RTX 1080 workstation
REM   (Pascal, compute capability 6.1) with CUDA 11.8.
REM   The original build_openxr.bat is hardcoded for the RTX 5060 laptop
REM   (Blackwell CC 12.0 / CUDA 12.8) and is left untouched.
REM
REM Differences from build_openxr.bat:
REM   - repo root      C:\3dgs-v2                -> C:\Users\travu\MISTA
REM   - CUDA           v12.8                     -> v11.8
REM   - cmake          conda Library\bin (trvu2600) -> pip cmake in env Scripts (travu)
REM   - GPU arch       120 / 12.0 (Blackwell)    -> 61 / 6.1 (Pascal, RTX 1080)
REM   - LibTorch       required via --libtorch   -> defaults to the env's torch package
REM
REM Uses VS2019 BuildTools vcvars64 + CUDA 11.8 + NMake
REM --allow-unsupported-compiler bypasses the CUDA/cl.exe version check
REM
REM Targets:
REM   v3 / v4 / v4.2 - OpenXR stereo VR viewers (need a headset + runtime to run)
REM   desktop        - mono desktop-window viewer, same v4.2 CUDA IPC transport and
REM                    the same GaussianLiveViewV42, no OpenXR. Runs against the
REM                    unchanged Python server (render_vr_v4_2.py, port 6012).
REM   desktopv43     - distinct target cloned from desktop (same shared view today),
REM                    reserved for the ROMP-driven display path. Same LibTorch needs.
REM
REM Usage:
REM   build_openxr_1080.bat                       - build v3 + v4 + v4.2 + desktop (default LibTorch)
REM   build_openxr_1080.bat --target v4.2         - build only v4.2
REM   build_openxr_1080.bat --target desktop      - build only the desktop viewer
REM   build_openxr_1080.bat --target desktopv43   - build only the v4.3 desktop viewer
REM   build_openxr_1080.bat --libtorch C:\libtorch --target v4.2  - use a custom LibTorch

setlocal enabledelayedexpansion

set "SIBR_SRC=C:\Users\travu\MISTA\submodules\sibr-core"
set "SIBR_BUILD=C:\Users\travu\MISTA\submodules\sibr-core\build"
set "SIBR_INSTALL=C:\Users\travu\MISTA\submodules\sibr-core\install"
set "CUDA_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"

set "VS19_VCVARS=C:\PROGRA~2\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
REM VS2019 BuildTools ships CMake 3.20, but sibr_core requires >= 3.22.
REM Use the pip-installed cmake (4.3.4) in the 3dgs-avatar conda env instead.
set "CMAKE_BIN=C:\Users\travu\AppData\Local\miniconda3\envs\3dgs-avatar\Scripts"

REM Default LibTorch = the C++ distribution shipped inside the env's torch package
REM (torch 2.1.2+cu118). Override with --libtorch to use a standalone LibTorch.
set "DEFAULT_LIBTORCH=C:\Users\travu\AppData\Local\miniconda3\envs\3dgs-avatar\Lib\site-packages\torch"

REM -- Parse arguments --
set "LIBTORCH_PATH="
set "TARGET_FILTER="

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
echo [WARN] Unknown argument: %~1
shift & goto parse_args
:args_done

REM Fall back to the env's bundled LibTorch when none was supplied.
if "%LIBTORCH_PATH%"=="" set "LIBTORCH_PATH=%DEFAULT_LIBTORCH%"

REM -- Decide which targets to build --
set BUILD_V3=0
set BUILD_V4=0
set BUILD_V42=0
set BUILD_DESKTOP=0
set BUILD_DESKTOPV43=0
set BUILD_UI=0

if "%TARGET_FILTER%"=="" (
    set BUILD_V3=1
    set BUILD_UI=1
    if not "%LIBTORCH_PATH%"=="" (
        set BUILD_V4=1
        set BUILD_V42=1
        set BUILD_DESKTOP=1
        set BUILD_DESKTOPV43=1
    )
) else if "%TARGET_FILTER%"=="v3"  ( set BUILD_V3=1
) else if "%TARGET_FILTER%"=="v4"  ( set BUILD_V4=1
) else if "%TARGET_FILTER%"=="v4.2" ( set BUILD_V42=1
) else if /i "%TARGET_FILTER%"=="desktop" ( set BUILD_DESKTOP=1
) else if /i "%TARGET_FILTER%"=="desktopv43" ( set BUILD_DESKTOPV43=1
) else if /i "%TARGET_FILTER%"=="ui" ( set BUILD_UI=1
) else (
    echo [ERROR] Unknown --target "%TARGET_FILTER%". Use v3, v4, v4.2, desktop, desktopv43, or ui.
    exit /b 1
)

if "%BUILD_V4%"=="1" (
    if "%LIBTORCH_PATH%"=="" (
        echo [ERROR] --libtorch ^<path^> required to build v4.
        exit /b 1
    )
)
if "%BUILD_V42%"=="1" (
    if "%LIBTORCH_PATH%"=="" (
        echo [ERROR] --libtorch ^<path^> required to build v4.2.
        exit /b 1
    )
)
REM The desktop viewer compiles GaussianLiveViewV42.cpp (LibTorch Color MLP), so it
REM has exactly the same LibTorch requirement as v4.2 despite not needing OpenXR.
if "%BUILD_DESKTOP%"=="1" (
    if "%LIBTORCH_PATH%"=="" (
        echo [ERROR] --libtorch ^<path^> required to build desktop.
        exit /b 1
    )
)
REM v4.3 desktop compiles the same GaussianLiveViewV42.cpp (LibTorch Color MLP),
REM so it has the same LibTorch requirement as v4.2 / desktop.
if "%BUILD_DESKTOPV43%"=="1" (
    if "%LIBTORCH_PATH%"=="" (
        echo [ERROR] --libtorch ^<path^> required to build desktopv43.
        exit /b 1
    )
)

REM sibr_remote (RemotePointView, used by the ui target) links sibr_basic, which
REM only exists when the basic project is enabled. The OpenXR/desktop targets do
REM not need it, so basic stays OFF unless we are building ui.
set IBR_BASIC=OFF
if "%BUILD_UI%"=="1" set IBR_BASIC=ON

REM -- Validate prerequisites --
if not exist "%VS19_VCVARS%" (
    echo [ERROR] VS2019 BuildTools not found: %VS19_VCVARS%
    exit /b 1
)
if not exist "%CUDA_DIR%/bin/nvcc.exe" (
    echo [ERROR] CUDA not found: %CUDA_DIR%/bin/nvcc.exe
    exit /b 1
)
if not "%LIBTORCH_PATH%"=="" (
    if not exist "%LIBTORCH_PATH%\share\cmake\Torch\TorchConfig.cmake" (
        echo [ERROR] LibTorch not found at: %LIBTORCH_PATH%
        echo         Expected: %LIBTORCH_PATH%\share\cmake\Torch\TorchConfig.cmake
        exit /b 1
    )
)

REM -- Load VS2019 environment --
echo [INFO] Loading VS2019 BuildTools environment ...
call "%VS19_VCVARS%"
set "PATH=%CMAKE_BIN%;%CUDA_DIR%/bin;%PATH%"

echo [INFO] Compiler versions:
cl 2>&1 | findstr /i "Version"
nvcc --version 2>&1 | findstr /i "release"
cmake --version 2>&1 | findstr /i "version"

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

set "CMAKE_PREFIX="
if not "%LIBTORCH_PATH%"=="" set "CMAKE_PREFIX=-DCMAKE_PREFIX_PATH=%LIBTORCH_PATH%"

REM RTX 1080 = Pascal, compute capability 6.1. Set explicitly so LibTorch's
REM Caffe2Config.cmake skips its compile-and-run GPU auto-detection.
cmake -S "%SIBR_SRC%" -B "%SIBR_BUILD%" ^
    -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    "-DCMAKE_CUDA_COMPILER=%CUDA_DIR%/bin/nvcc.exe" ^
    "-DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR%" ^
    "-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler" ^
    "-DCMAKE_INSTALL_PREFIX=%SIBR_INSTALL%" ^
    -DCMAKE_CUDA_ARCHITECTURES=61 ^
    -DTORCH_CUDA_ARCH_LIST=6.1 ^
    -DBUILD_IBR_BASIC=%IBR_BASIC% ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    "-DCMAKE_CXX_FLAGS=/arch:AVX2" ^
    %CMAKE_PREFIX%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] cmake configure failed.
    exit /b %ERRORLEVEL%
)

REM -- Build and install each target --
set BUILT=

if "%BUILD_V3%"=="1" (
    echo.
    echo [INFO] Building v3: SIBR_remoteGaussianOpenXR_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussianOpenXR_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] v3 build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussianOpenXR_app_install
    set "BUILT=%BUILT% v3"
)

if "%BUILD_V4%"=="1" (
    echo.
    echo [INFO] Building v4: SIBR_remoteGaussianOpenXRv4_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussianOpenXRv4_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] v4 build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussianOpenXRv4_app_install
    set "BUILT=%BUILT% v4"
)

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

if "%BUILD_DESKTOPV43%"=="1" (
    echo.
    echo [INFO] Building desktopv43: SIBR_remoteGaussianDesktopV43_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussianDesktopV43_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] desktopv43 build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussianDesktopV43_app_install
    set "BUILT=%BUILT% desktopv43"
)

REM ui = SIBR_remoteGaussian_app (remoteGaussianUI): the ORIGINAL 3DGS remote
REM viewer. It uses RemotePointView (JSON camera in -> RGB image out, over TCP)
REM and does NO rasterization itself, so it needs no LibTorch. Pair it with the
REM render_desktop_v1.py orchestrator + data/dummy_viewer. This is the desktopv1
REM architecture: Python renders full frames with the untouched render.py pipeline.
if "%BUILD_UI%"=="1" (
    echo.
    echo [INFO] Building ui: SIBR_remoteGaussian_app ...
    cmake --build "%SIBR_BUILD%" --target SIBR_remoteGaussian_app
    if !ERRORLEVEL! neq 0 ( echo [ERROR] ui build failed. & exit /b 1 )
    cmake --install "%SIBR_BUILD%" --config RelWithDebInfo --component SIBR_remoteGaussian_app_install
    set "BUILT=%BUILT% ui"
)

echo.
echo ============================================================
echo  Build succeeded: %BUILT%
echo  Output: %SIBR_INSTALL%\bin\
echo ============================================================

endlocal
