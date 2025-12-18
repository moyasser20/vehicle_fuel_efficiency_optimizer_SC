package nn.core;

import nn.activation.ActivationFunction;
import java.util.ArrayList;
import java.util.List;

public class Layer {
    private List<Neuron> neurons;
    private ActivationFunction activationFunction;
    private int inputSize;
    private int outputSize;
    private double[] inputs;
    private double[] outputs;
    
    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFunction = activationFunction;
        this.neurons = new ArrayList<>();
        
        for (int i = 0; i < outputSize; i++) {
            neurons.add(new Neuron(inputSize));
        }
    }
    
    public List<Neuron> getNeurons() {
        return neurons;
    }
    
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getInputSize() {
        return inputSize;
    }
    
    public int getOutputSize() {
        return outputSize;
    }
    
    public double[] getInputs() {
        return inputs;
    }
    
    public double[] getOutputs() {
        return outputs;
    }
    
    public double[] forward(double[] inputs) {
        if (inputs.length != inputSize) {
            throw new IllegalArgumentException("Input size mismatch: expected " + inputSize + ", got " + inputs.length);
        }
        
        this.inputs = inputs.clone();
        this.outputs = new double[outputSize];
        
        for (int i = 0; i < outputSize; i++) {
            Neuron neuron = neurons.get(i);
            double sum = neuron.getBias();
            
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * neuron.getWeights()[j];
            }
            
            double activated = activationFunction.activate(sum);
            neuron.setOutput(activated);
            outputs[i] = activated;
        }
        
        return outputs;
    }
    
    public double[] backward(double[] deltas) {
        if (deltas.length != outputSize) {
            throw new IllegalArgumentException("Delta size mismatch: expected " + outputSize + ", got " + deltas.length);
        }
        
        double[] prevDeltas = new double[inputSize];
        
        for (int i = 0; i < outputSize; i++) {
            Neuron neuron = neurons.get(i);
            double delta = deltas[i];
            
            double activationDerivative = activationFunction.derivative(neuron.getOutput());
            double neuronDelta = delta * activationDerivative;
            neuron.setDelta(neuronDelta);
            
            for (int j = 0; j < inputSize; j++) {
                double weightGradient = neuronDelta * inputs[j];
                neuron.accumulateGradient(j, weightGradient);
                prevDeltas[j] += neuron.getWeights()[j] * neuronDelta;
            }
            
            neuron.accumulateBiasGradient(neuronDelta);
        }
        
        return prevDeltas;
    }
    
    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            double[] weights = neuron.getWeights();
            double[] gradients = neuron.getWeightGradients();
            for (int i = 0; i < weights.length; i++) {
                weights[i] -= learningRate * gradients[i];
            }
            
            neuron.setBias(neuron.getBias() - learningRate * neuron.getBiasGradient());
            
            neuron.resetGradients();
        }
    }
}


