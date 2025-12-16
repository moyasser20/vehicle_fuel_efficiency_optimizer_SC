package nn.core;

/**
 * Represents a single neuron in a neural network layer.
 * Each neuron has weights, bias, and activation function.
 */
public class Neuron {
    private double[] weights;
    private double bias;
    private double output;
    private double delta; // For backpropagation
    private double[] weightGradients; // For gradient accumulation
    private double biasGradient;
    
    public Neuron(int inputSize) {
        this.weights = new double[inputSize];
        this.weightGradients = new double[inputSize];
        this.bias = 0.0;
        this.biasGradient = 0.0;
    }
    
    public double[] getWeights() {
        return weights;
    }
    
    public void setWeights(double[] weights) {
        this.weights = weights;
        this.weightGradients = new double[weights.length];
    }
    
    public double getBias() {
        return bias;
    }
    
    public void setBias(double bias) {
        this.bias = bias;
    }
    
    public double getOutput() {
        return output;
    }
    
    public void setOutput(double output) {
        this.output = output;
    }
    
    public double getDelta() {
        return delta;
    }
    
    public void setDelta(double delta) {
        this.delta = delta;
    }
    
    public double[] getWeightGradients() {
        return weightGradients;
    }
    
    public double getBiasGradient() {
        return biasGradient;
    }
    
    public void resetGradients() {
        for (int i = 0; i < weightGradients.length; i++) {
            weightGradients[i] = 0.0;
        }
        biasGradient = 0.0;
    }
    
    public void accumulateGradient(int weightIndex, double gradient) {
        weightGradients[weightIndex] += gradient;
    }
    
    public void accumulateBiasGradient(double gradient) {
        biasGradient += gradient;
    }
}

