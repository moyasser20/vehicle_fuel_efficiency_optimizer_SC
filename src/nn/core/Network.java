package nn.core;

import nn.activation.ActivationFunction;
import nn.initialization.WeightInitializer;
import nn.loss.LossFunction;
import java.util.ArrayList;
import java.util.List;

public class Network {
    private List<Layer> layers;
    private LossFunction lossFunction;
    private WeightInitializer weightInitializer;
    private double learningRate;
    private int inputSize;
    
    private List<Double> trainingLossHistory;
    
    public Network(int inputSize) {
        this.inputSize = inputSize;
        this.layers = new ArrayList<>();
        this.trainingLossHistory = new ArrayList<>();
        this.learningRate = 0.01;
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
    
    public void backward(double[] expectedOutput) {
        if (lossFunction == null) {
            throw new IllegalStateException("Loss function must be set before training");
        }
        
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = outputLayer.getOutputs();
        
        double[] outputDeltas = lossFunction.computeGradient(output, expectedOutput);
        
        double[] currentDeltas = outputDeltas;
        for (int i = layers.size() - 1; i >= 0; i--) {
            currentDeltas = layers.get(i).backward(currentDeltas);
        }
    }
    
    public void updateWeights() {
        for (Layer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }
    
    public double trainSample(double[] inputs, double[] expectedOutput) {
        double[] output = forward(inputs);
        
        double loss = lossFunction.compute(output, expectedOutput);
        
        backward(expectedOutput);
        
        updateWeights();
        
        return loss;
    }
    
    public double[] predict(double[] inputs) {
        return forward(inputs);
    }
    
    public double[][] predictBatch(double[][] inputsBatch) {
        double[][] predictions = new double[inputsBatch.length][];
        for (int i = 0; i < inputsBatch.length; i++) {
            predictions[i] = predict(inputsBatch[i]);
        }
        return predictions;
    }
    
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

