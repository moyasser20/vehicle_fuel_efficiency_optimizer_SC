@echo off
echo ========================================
echo Vehicle Fuel Efficiency Optimizer
echo Neural Network Case Study Demo
echo ========================================
echo.

echo Compiling source files...
javac -d production -sourcepath src src/NNMain.java src/case_study/FuelEfficiencyNN.java src/nn/**/*.java src/nn/**/**/*.java 2>nul

if %errorlevel% neq 0 (
    echo Compilation failed. Trying alternative compilation...
    javac -d production -sourcepath src src/NNMain.java
)

echo.
echo Running demo...
echo ========================================
echo.

java -cp production NNMain

echo.
echo ========================================
echo Demo completed!
pause

