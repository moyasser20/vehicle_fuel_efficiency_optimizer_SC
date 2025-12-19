package case_study;

import nn.core.Network;
import nn.activation.ReLU;
import nn.activation.Sigmoid;
import nn.activation.Linear;
import nn.initialization.Xavier;
import nn.initialization.He;
import nn.loss.MSE;
import nn.loss.CrossEntropy;
import nn.training.Trainer;
import nn.data.DataUtils;

/**
 * Case Study: Vehicle Fuel Efficiency Prediction using Neural Networks
 * 
 * Problem: Predict fuel efficiency based on driving behavior
 * Dataset: Real telemetry data from Max Verstappen's 2024 Miami Grand Prix
 * 
 * Input Features (3):
 *   - Speed (km/h): Vehicle speed, directly affects fuel consumption
 *   - Gear Number (nGear): Current gear, affects engine efficiency
 *   - Throttle Position (%): Driver input, indicates acceleration/engine load
 * 
 * Output:
 *   - Fuel Efficiency Score: Calculated metric based on RPM, Speed, and Throttle
 *     Higher score = better fuel efficiency (more speed with less engine load)
 * 
 * This is a regression problem, so we use:
 *   - MSE loss function
 *   - Linear activation for output layer
 *   - ReLU for hidden layers
 */
public class FuelEfficiencyNN {
    
    public static void runDemo() {
        System.out.println("==========================================");
        System.out.println(" NEURAL NETWORK CASE STUDY");
        System.out.println(" Vehicle Fuel Efficiency Prediction");
        System.out.println("==========================================");
        System.out.println();
        
        // Load real dataset from CSV
        String csvPath = "vehicle-DataSet/verstappen_telemetry_miami_2024.csv";
        System.out.println("Loading dataset from: " + csvPath);
        
        // Load all relevant columns: RPM (2), Speed (3), nGear (4), Throttle (5)
        // Column indices: 0=index, 1=Date, 2=RPM, 3=Speed, 4=nGear, 5=Throttle, ...
        double[][] allData = DataUtils.loadCSV(csvPath, new int[]{2, 3, 4, 5}, true);
        
        System.out.println("Loaded " + allData.length + " samples from dataset");
        
        // Extract inputs: Speed (index 1), nGear (index 2), Throttle (index 3)
        // Extract RPM (index 0) for calculating fuel efficiency
        double[][] inputs = new double[allData.length][3];
        double[][] outputs = new double[allData.length][1];
        
        for (int i = 0; i < allData.length; i++) {
            double rpm = allData[i][0];
            double speed = allData[i][1];
            double gear = allData[i][2];
            double throttle = allData[i][3];
            
            // Inputs: Speed, Gear, Throttle
            inputs[i][0] = speed;
            inputs[i][1] = gear;
            inputs[i][2] = throttle;
            
            // Calculate fuel efficiency score
            // Higher score = better efficiency (more speed with less engine load)
            // Formula: Speed / (RPM * normalized_throttle)
            // We normalize throttle to avoid division by zero
            double normalizedThrottle = (throttle / 100.0) + 0.1; // Add small value to avoid zero
            double fuelEfficiency = 0.0;
            if (rpm > 0 && normalizedThrottle > 0) {
                fuelEfficiency = speed / (rpm * normalizedThrottle);
            }
            
            outputs[i][0] = fuelEfficiency;
        }
        
        System.out.println("Extracted 3 input features: Speed, Gear, Throttle");
        System.out.println("Calculated fuel efficiency scores from RPM data");
        
        // Normalize inputs
        DataUtils.NormalizationResult inputNorm = DataUtils.normalize(inputs);
        double[][] normalizedInputs = inputNorm.normalizedData;
        
        // Normalize outputs
        DataUtils.NormalizationResult outputNorm = DataUtils.normalize(outputs);
        double[][] normalizedOutputs = outputNorm.normalizedData;
        
        // Split into training and testing sets
        double[][][] split = DataUtils.trainTestSplit(normalizedInputs, normalizedOutputs, 0.2);
        double[][] trainInputs = split[0];
        double[][] trainOutputs = split[1];
        double[][] testInputs = split[2];
        double[][] testOutputs = split[3];
        
        System.out.println("Dataset prepared:");
        System.out.println("  Training samples: " + trainInputs.length);
        System.out.println("  Testing samples: " + testInputs.length);
        System.out.println("  Input features: " + trainInputs[0].length);
        System.out.println();
        
        // Create neural network
        System.out.println("Creating neural network...");
        Network network = new Network(3); // 3 input features: Speed, Gear, Throttle
        
        // Architecture: 3 -> 10 -> 8 -> 1
        network.addLayer(10, new ReLU());  // Hidden layer 1
        network.addLayer(8, new ReLU());   // Hidden layer 2
        network.addLayer(1, new Linear()); // Output layer (regression)
        
        // Set loss function
        network.setLossFunction(new MSE());
        
        // Set weight initializer (Xavier for ReLU layers)
        network.setWeightInitializer(new Xavier());
        
        // Create trainer
        Trainer trainer = new Trainer(network);
        trainer.setLearningRate(0.01);
        trainer.setEpochs(100);
        trainer.setBatchSize(8);
        trainer.setShuffleData(true);
        
        System.out.println("Network architecture:");
        System.out.println("  Input layer: 3 neurons (Speed, Gear, Throttle)");
        System.out.println("  Hidden layer 1: 10 neurons (ReLU)");
        System.out.println("  Hidden layer 2: 8 neurons (ReLU)");
        System.out.println("  Output layer: 1 neuron (Linear) - Fuel Efficiency Score");
        System.out.println("  Loss function: MSE");
        System.out.println("  Weight initialization: Xavier");
        System.out.println("  Learning rate: 0.01");
        System.out.println("  Batch size: 8");
        System.out.println();
        
        // Train the network
        System.out.println("Training network...");
        System.out.println("----------------------------------------");
        trainer.train(trainInputs, trainOutputs);
        System.out.println("----------------------------------------");
        System.out.println();
        
        // Evaluate on test set
        System.out.println("Evaluating on test set...");
        double testLoss = network.evaluate(testInputs, testOutputs);
        System.out.printf("Test Loss (MSE): %.6f%n", testLoss);
        System.out.println();
        
        // Make some predictions
        System.out.println("Sample predictions:");
        System.out.println("----------------------------------------");
        for (int i = 0; i < Math.min(5, testInputs.length); i++) {
            double[] prediction = network.predict(testInputs[i]);
            double[] denormalizedPred = outputNorm.denormalize(prediction);
            double[] denormalizedActual = outputNorm.denormalize(testOutputs[i]);
            
            System.out.printf("Sample %d:%n", i + 1);
            System.out.printf("  Input - Speed: %.1f km/h, Gear: %.0f, Throttle: %.1f%%%n", 
                inputNorm.denormalize(testInputs[i])[0],
                inputNorm.denormalize(testInputs[i])[1],
                inputNorm.denormalize(testInputs[i])[2]);
            System.out.printf("  Predicted Fuel Efficiency Score: %.6f%n", denormalizedPred[0]);
            System.out.printf("  Actual Fuel Efficiency Score:    %.6f%n", denormalizedActual[0]);
            System.out.printf("  Error:                           %.6f%n", Math.abs(denormalizedPred[0] - denormalizedActual[0]));
            System.out.println();
        }
        
        // Show training history
        System.out.println("Training loss history (first 5 and last 5 epochs):");
        var history = network.getTrainingLossHistory();
        for (int i = 0; i < Math.min(5, history.size()); i++) {
            System.out.printf("  Epoch %d: %.6f%n", i + 1, history.get(i));
        }
        if (history.size() > 10) {
            System.out.println("  ...");
            for (int i = history.size() - 5; i < history.size(); i++) {
                System.out.printf("  Epoch %d: %.6f%n", i + 1, history.get(i));
            }
        }
    }
    
}





