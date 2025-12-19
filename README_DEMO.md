# Runnable Demo Script - Vehicle Fuel Efficiency Optimizer

## What is a Runnable Demo Script?

A **runnable demo script** is an executable program or script that demonstrates your case study. It should:
- ✅ Be executable/runable
- ✅ Show the complete case study in action
- ✅ Display results and outputs
- ✅ Be easy to run (one command)

## Your Runnable Demo Script

**Main Demo Script:** `src/NNMain.java`

This is your primary runnable demo script. It contains the `main()` method that executes the complete case study.

## How to Run the Demo

### Option 1: Using the Helper Scripts (Easiest)

**Windows:**
```bash
run_demo.bat
```

**Linux/Mac:**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

### Option 2: Manual Compilation and Execution

**Step 1: Compile**
```bash
javac -d production -sourcepath src src/NNMain.java
```

**Step 2: Run**
```bash
java -cp production NNMain
```

### Option 3: Using IDE

1. Open the project in your IDE (IntelliJ IDEA, Eclipse, etc.)
2. Open `src/NNMain.java`
3. Right-click → Run 'NNMain.main()'

## Expected Output

When you run the demo, you should see:

```
=== Soft Computing Library - Phase 3: Neural Network Demo ===

==========================================
 NEURAL NETWORK CASE STUDY
 Vehicle Fuel Efficiency Prediction
==========================================

Loading dataset from: vehicle-DataSet/verstappen_telemetry_miami_2024.csv
Loaded 20656 samples from dataset
Extracted 3 input features: Speed, Gear, Throttle
Calculated fuel efficiency scores from RPM data
Dataset prepared:
  Training samples: 16525
  Testing samples: 4131
  Input features: 3

Creating neural network...
Network architecture:
  Input layer: 3 neurons (Speed, Gear, Throttle)
  Hidden layer 1: 10 neurons (ReLU)
  Hidden layer 2: 8 neurons (ReLU)
  Output layer: 1 neuron (Linear) - Fuel Efficiency Score
  Loss function: MSE
  Weight initialization: Xavier
  Learning rate: 0.01
  Batch size: 8

Training network...
----------------------------------------
Epoch 1/100 - Average Loss: 0.XXXXXX
Epoch 10/100 - Average Loss: 0.XXXXXX
...
Epoch 100/100 - Average Loss: 0.XXXXXX
----------------------------------------

Evaluating on test set...
Test Loss (MSE): 0.XXXXXX

Sample predictions:
----------------------------------------
Sample 1:
  Input - Speed: 123.0 km/h, Gear: 3.0, Throttle: 85.0%
  Predicted Fuel Efficiency Score: 0.000123
  Actual Fuel Efficiency Score:    0.000125
  Error:                           0.000002
...
```

## What the Demo Shows

1. **Data Loading:** Loads the telemetry dataset from CSV
2. **Preprocessing:** Normalizes data and splits into train/test sets
3. **Network Creation:** Builds the neural network architecture
4. **Training:** Trains the network for 100 epochs
5. **Evaluation:** Tests on held-out test set
6. **Predictions:** Shows sample predictions with actual vs predicted values

## Files Included

- ✅ **`src/NNMain.java`** - Main runnable demo script
- ✅ **`run_demo.bat`** - Windows batch script (helper)
- ✅ **`run_demo.sh`** - Linux/Mac shell script (helper)
- ✅ **`README_DEMO.md`** - This file (instructions)

## For Submission

Include in your ZIP file:
- ✅ `src/NNMain.java` (the main runnable demo script)
- ✅ `src/case_study/FuelEfficiencyNN.java` (case study implementation)
- ✅ `src/nn/` (entire neural network library)
- ✅ `vehicle-DataSet/` (dataset folder)
- ✅ Helper scripts (optional but recommended)

## Troubleshooting

**Error: "Cannot find or load main class NNMain"**
- Make sure you compiled first: `javac -d production -sourcepath src src/NNMain.java`
- Check that the `production` folder contains the compiled classes

**Error: "FileNotFoundException: vehicle-DataSet/verstappen_telemetry_miami_2024.csv"**
- Make sure the dataset file exists in the `vehicle-DataSet/` folder
- Run from the project root directory

**Error: "Package does not exist"**
- Compile all source files: `javac -d production -sourcepath src src/**/*.java`
- Or use the helper scripts which handle compilation automatically
