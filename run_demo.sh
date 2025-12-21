#!/bin/bash

echo "========================================"
echo "Vehicle Fuel Efficiency Optimizer"
echo "Neural Network Case Study Demo"
echo "========================================"
echo ""

echo "Compiling source files..."
javac -d production -sourcepath src src/NNMain.java src/case_study/FuelEfficiencyNN.java src/nn/**/*.java src/nn/**/**/*.java 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Compilation failed. Trying alternative compilation..."
    javac -d production -sourcepath src src/NNMain.java
fi

echo ""
echo "Running demo..."
echo "========================================"
echo ""

java -cp production NNMain

echo ""
echo "========================================"
echo "Demo completed!"

