@echo off
echo Building Software TPU Project...
where g++ >nul 2>nul
if %errorlevel% equ 0 (
    g++ -std=c++17 -O3 -mavx2 -mfma main.cpp -o software_tpu.exe
    if %errorlevel% equ 0 (
        echo Build successful! Running software_tpu.exe...
        echo ----------------------------------------
        software_tpu.exe
    )
    exit /b %errorlevel%
)

where cl >nul 2>nul
if %errorlevel% equ 0 (
    cl /std:c++17 /O2 /arch:AVX2 /EHsc main.cpp /Fesoftware_tpu.exe
    if %errorlevel% equ 0 (
        echo Build successful! Running software_tpu.exe...
        echo ----------------------------------------
        software_tpu.exe
    )
    exit /b %errorlevel%
)

echo Neither g++ nor cl found in PATH. Please use your preferred C++ compiler to compile main.cpp.
