package nn.training;

import nn.core.Network;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Handles training of neural networks with configurable hyperparameters.
 */
public class Trainer {
    private Network network;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private boolean shuffleData;
    private Random random;
    
    public Trainer(Network network) {
        this.network = network;
        this.learningRate = 0.01;
        this.epochs = 100;
        this.batchSize = 1; // Default: online learning
        this.shuffleData = true;
        this.random = new Random();
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        network.setLearningRate(learningRate);
    }
    
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public void setShuffleData(boolean shuffle) {
        this.shuffleData = shuffle;
    }
    
    /**
     * Train the network on a dataset.
     * @param inputs Input samples
     * @param expectedOutputs Expected output samples
     * @return List of average loss per epoch
     */
    public List<Double> train(double[][] inputs, double[][] expectedOutputs) {
        if (inputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("Input and output arrays must have the same length");
        }
        
        List<Double> epochLosses = new ArrayList<>();
        int numSamples = inputs.length;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Shuffle data if enabled
            if (shuffleData) {
                shuffleData(inputs, expectedOutputs);
            }
            
            double epochLoss = 0.0;
            int batchCount = 0;
            
            // Process in batches
            for (int i = 0; i < numSamples; i += batchSize) {
                int batchEnd = Math.min(i + batchSize, numSamples);
                
                // Reset gradients for batch
                resetGradients();
                
                // Accumulate gradients over batch
                for (int j = i; j < batchEnd; j++) {
                    // Forward pass
                    double[] output = network.forward(inputs[j]);
                    // Calculate loss
                    if (network.getLossFunction() != null) {
                        epochLoss += network.getLossFunction().compute(output, expectedOutputs[j]);
                    }
                    // Backward pass (accumulates gradients)
                    network.backward(expectedOutputs[j]);
                }
                
                // Update weights after processing entire batch
                network.updateWeights();
                
                batchCount++;
            }
            
            double avgLoss = epochLoss / numSamples;
            epochLosses.add(avgLoss);
            network.getTrainingLossHistory().add(avgLoss);
            
            // Print progress every 10 epochs or on last epoch
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d/%d - Average Loss: %.6f%n", epoch + 1, epochs, avgLoss);
            }
        }
        
        return epochLosses;
    }
    
    /**
     * Reset gradients for all layers (for batch training).
     */
    private void resetGradients() {
        for (var layer : network.getLayers()) {
            for (var neuron : layer.getNeurons()) {
                neuron.resetGradients();
            }
        }
    }
    
    /**
     * Shuffle input and output arrays together.
     */
    private void shuffleData(double[][] inputs, double[][] expectedOutputs) {
        for (int i = inputs.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            // Swap inputs
            double[] tempInput = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = tempInput;
            // Swap outputs
            double[] tempOutput = expectedOutputs[i];
            expectedOutputs[i] = expectedOutputs[j];
            expectedOutputs[j] = tempOutput;
        }
    }
}

