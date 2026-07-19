@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================
echo   LocateAnything-VideoClipper 依赖安装
echo ============================================
echo.

set PYTHON=embedded_python\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] 未找到 embedded_python\python.exe
    pause
    exit /b 1
)

echo [1/4] 升级 pip...
"%PYTHON%" -m pip install --upgrade pip -q
if %errorlevel% neq 0 (
    echo [ERROR] pip 升级失败
    pause
    exit /b 1
)

echo [2/4] 安装 PyTorch (CUDA 12.4)...
"%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch 安装失败
    pause
    exit /b 1
)

echo [3/4] 安装 transformers 及其他依赖...
"%PYTHON%" -m pip install transformers==4.57.1 opencv-python-headless==4.11.0.86 numpy==1.25.0 Pillow==11.1.0 peft decord==0.6.0 -q
if %errorlevel% neq 0 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)

echo [4/4] 安装 Gradio...
"%PYTHON%" -m pip install gradio -q
if %errorlevel% neq 0 (
    echo [ERROR] Gradio 安装失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo   依赖安装完成！
echo ============================================
echo.
pause
