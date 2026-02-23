@echo off
echo ============================================
echo  Installing ALL dependencies for All-in-One
echo ============================================
echo.

D:\python311\python.exe -m pip install ^
    opencv-python ^
    pycocotools ^
    pandas ^
    easydict ^
    pyyaml ^
    Pillow ^
    tensorboard ^
    lmdb ^
    scikit-image ^
    matplotlib ^
    jittor==1.3.8.5

echo.
echo Done!
pause
