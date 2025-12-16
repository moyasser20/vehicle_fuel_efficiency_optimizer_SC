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
 * Problem: Predict fuel efficiency (MPG) based on vehicle characteristics
 * Dataset: Synthetic vehicle data with features like:
 *   - Engine size
 *   - Cylinders
 *   - Horsepower
 *   - Weight
 *   - Acceleration
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
        
        // Generate synthetic dataset
        System.out.println("Generating synthetic vehicle dataset...");
        double[][] inputs = generateVehicleData();
        double[][] outputs = generateFuelEfficiency(inputs);
        
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
        Network network = new Network(5); // 5 input features
        
        // Architecture: 5 -> 10 -> 8 -> 1
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
        System.out.println("  Input layer: 5 neurons");
        System.out.println("  Hidden layer 1: 10 neurons (ReLU)");
        System.out.println("  Hidden layer 2: 8 neurons (ReLU)");
        System.out.println("  Output layer: 1 neuron (Linear)");
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
            System.out.printf("  Predicted MPG: %.2f%n", denormalizedPred[0]);
            System.out.printf("  Actual MPG:    %.2f%n", denormalizedActual[0]);
            System.out.printf("  Error:         %.2f%n", Math.abs(denormalizedPred[0] - denormalizedActual[0]));
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
    
    /**
     * Generate synthetic vehicle data.
     * Features: [engine_size, cylinders, horsepower, weight, acceleration]
     */
    private static double[][] generateVehicleData() {
        int numSamples = 200;
        double[][] data = new double[numSamples][5];
        
        for (int i = 0; i < numSamples; i++) {
            // Engine size (liters): 1.0 - 6.0
            data[i][0] = 1.0 + Math.random() * 5.0;
            
            // Cylinders: 4, 6, or 8
            int[] cylinders = {4, 6, 8};
            data[i][1] = cylinders[(int)(Math.random() * 3)];
            
            // Horsepower: 50 - 400
            data[i][2] = 50 + Math.random() * 350;
            
            // Weight (kg): 1000 - 3000
            data[i][3] = 1000 + Math.random() * 2000;
            
            // Acceleration (0-60 mph in seconds): 5 - 20
            data[i][4] = 5 + Math.random() * 15;
        }
        
        return data;
    }
    
    /**
     * Generate fuel efficiency based on vehicle characteristics.
     * Simple formula: MPG decreases with engine size, weight, and horsepower,
     * and increases with acceleration.
     */
    private static double[][] generateFuelEfficiency(double[][] inputs) {
        double[][] outputs = new double[inputs.length][1];
        
        for (int i = 0; i < inputs.length; i++) {
            double engineSize = inputs[i][0];
            double cylinders = inputs[i][1];
            double horsepower = inputs[i][2];
            double weight = inputs[i][3];
            double acceleration = inputs[i][4];
            
            // Simple formula with some noise
            double mpg = 50.0 
                - engineSize * 3.0
                - (cylinders - 4) * 2.0
                - horsepower / 20.0
                - weight / 100.0
                + (20 - acceleration) * 0.5
                + (Math.random() - 0.5) * 5.0; // Add noise
            
            // Clamp to reasonable range
            mpg = Math.max(10.0, Math.min(50.0, mpg));
            outputs[i][0] = mpg;
        }
        
        return outputs;
    }
}

