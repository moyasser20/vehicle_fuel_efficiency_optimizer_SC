package nn.examples;

import nn.core.Network;
import nn.activation.*;
import nn.initialization.*;
import nn.loss.*;
import nn.training.Trainer;
import nn.data.DataUtils;

/**
 * Simple example demonstrating the neural network library.
 * This example shows how to:
 * 1. Create a neural network
 * 2. Configure activation functions
 * 3. Set weight initialization
 * 4. Train the network
 * 5. Make predictions
 */
public class SimpleExample {
    
    public static void main(String[] args) {
        System.out.println("=== Simple Neural Network Example ===");
        System.out.println();
        
        // Create a simple dataset: XOR problem
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        double[][] outputs = {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        };
        
        // Create network: 2 inputs -> 4 hidden -> 1 output
        Network network = new Network(2);
        network.addLayer(4, new Sigmoid());  // Hidden layer
        network.addLayer(1, new Sigmoid());  // Output layer
        
        // Set loss function (CrossEntropy for binary classification)
        network.setLossFunction(new MSE()); // Using MSE for simplicity
        
        // Set weight initializer
        network.setWeightInitializer(new Xavier());
        
        // Create trainer
        Trainer trainer = new Trainer(network);
        trainer.setLearningRate(0.5);
        trainer.setEpochs(1000);
        trainer.setBatchSize(4);
        
        // Train
        System.out.println("Training network on XOR problem...");
        trainer.train(inputs, outputs);
        
        // Test predictions
        System.out.println("\nPredictions:");
        for (int i = 0; i < inputs.length; i++) {
            double[] prediction = network.predict(inputs[i]);
            System.out.printf("Input: [%.1f, %.1f] -> Output: %.3f (Expected: %.1f)%n",
                inputs[i][0], inputs[i][1], prediction[0], outputs[i][0]);
        }
    }
}

