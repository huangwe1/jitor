@echo off
echo ============================================
echo  All-in-One Quick Training (GOT-10k only)
echo ============================================
echo.

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
set nvcc_path=%CUDA_PATH%\bin\nvcc.exe

cd /d "%~dp0\lib\train"
python run_training_all_in_one.py --script ostrack --config vitb_256_mae_ce_32x4_got10k_quick --save_dir "%~dp0output" --seed 42

pause
