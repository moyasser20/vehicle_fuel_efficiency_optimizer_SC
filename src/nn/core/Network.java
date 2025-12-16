package nn.core;

import nn.activation.ActivationFunction;
import nn.initialization.WeightInitializer;
import nn.loss.LossFunction;
import java.util.ArrayList;
import java.util.List;

/**
 * Main neural network class that manages layers and training.
 */
public class Network {
    private List<Layer> layers;
    private LossFunction lossFunction;
    private WeightInitializer weightInitializer;
    private double learningRate;
    private int inputSize;
    
    // Training history
    private List<Double> trainingLossHistory;
    
    public Network(int inputSize) {
        this.inputSize = inputSize;
        this.layers = new ArrayList<>();
        this.trainingLossHistory = new ArrayList<>();
        this.learningRate = 0.01; // Default learning rate
    }
    
    public void addLayer(int outputSize, ActivationFunction activationFunction) {
        int layerInputSize = layers.isEmpty() ? inputSize : layers.get(layers.size() - 1).getOutputSize();
        Layer layer = new Layer(layerInputSize, outputSize, activationFunction);
        layers.add(layer);
    }
    
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    
    public LossFunction getLossFunction() {
        return lossFunction;
    }
    
    public void setWeightInitializer(WeightInitializer initializer) {
        this.weightInitializer = initializer;
        initializeWeights();
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public List<Layer> getLayers() {
        return layers;
    }
    
    public List<Double> getTrainingLossHistory() {
        return trainingLossHistory;
    }
    
    /**
     * Initialize all weights in the network using the specified initializer.
     */
    private void initializeWeights() {
        if (weightInitializer == null) {
            return;
        }
        
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                int inputSize = neuron.getWeights().length;
                double[] weights = weightInitializer.initialize(inputSize, layer.getOutputSize());
                neuron.setWeights(weights);
                neuron.setBias(weightInitializer.initializeBias());
            }
        }
    }
    
    /**
     * Forward propagation through the entire network.
     * @param inputs Input values
     * @return Output values
     */
    public double[] forward(double[] inputs) {
        if (inputs.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch: expected " + inputSize + ", got " + inputs.length);
        }
        
        double[] currentInput = inputs;
        for (Layer layer : layers) {
            currentInput = layer.forward(currentInput);
        }
        return currentInput;
    }
    
    /**
     * Backward propagation through the entire network.
     * @param expectedOutput Expected output values
     */
    public void backward(double[] expectedOutput) {
        if (lossFunction == null) {
            throw new IllegalStateException("Loss function must be set before training");
        }
        
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = outputLayer.getOutputs();
        
        // Calculate output layer deltas using loss function
        double[] outputDeltas = lossFunction.computeGradient(output, expectedOutput);
        
        // Backpropagate through all layers
        double[] currentDeltas = outputDeltas;
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentDeltas = layers.get(i).backward(currentDeltas);
        }
    }
    
    /**
     * Update all weights in the network.
     */
    public void updateWeights() {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }
    
    /**
     * Train the network on a single sample.
     * @param inputs Input values
     * @param expectedOutput Expected output values
     * @return Loss value
     */
    public double trainSample(double[] inputs, double[] expectedOutput) {
        // Forward pass
        double[] output = forward(inputs);
        
        // Calculate loss
        double loss = lossFunction.compute(output, expectedOutput);
        
        // Backward pass
        backward(expectedOutput);
        
        // Update weights
        updateWeights();
        
        return loss;
    }
    
    /**
     * Predict output for a single input.
     * @param inputs Input values
     * @return Predicted output values
     */
    public double[] predict(double[] inputs) {
        return forward(inputs);
    }
    
    /**
     * Predict outputs for a batch of inputs.
     * @param inputsBatch Batch of input values
     * @return Batch of predicted output values
     */
    public double[][] predictBatch(double[][] inputsBatch) {
        double[][] predictions = new double[inputsBatch.length][];
        for (int i = 0; i < inputsBatch.length; i++) {
            predictions[i] = predict(inputsBatch[i]);
        }
        return predictions;
    }
    
    /**
     * Evaluate the network on a dataset.
     * @param inputs Input samples
     * @param expectedOutputs Expected output samples
     * @return Average loss
     */
    public double evaluate(double[][] inputs, double[][] expectedOutputs) {
        if (lossFunction == null) {
            throw new IllegalStateException("Loss function must be set before evaluation");
        }
        
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            double[] output = forward(inputs[i]);
            totalLoss += lossFunction.compute(output, expectedOutputs[i]);
        }
        return totalLoss / inputs.length;
    }
    
    /**
     * Get intermediate values for debugging.
     * @param inputs Input values
     * @return List of layer outputs (including input)
     */
    public List<double[]> getIntermediateValues(double[] inputs) {
        List<double[]> intermediate = new ArrayList<>();
        intermediate.add(inputs.clone());
        
        double[] currentInput = inputs;
        for (Layer layer : layers) {
            currentInput = layer.forward(currentInput);
            intermediate.add(currentInput.clone());
        }
        return intermediate;
    }
}

