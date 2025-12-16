package nn.initialization;

/**
 * Interface for weight initialization methods.
 */
public interface WeightInitializer {
    /**
     * Initialize weights for a neuron.
     * @param inputSize Number of inputs to the neuron
     * @param outputSize Number of outputs from the layer (for fan-out)
     * @return Array of initialized weights
     */
    double[] initialize(int inputSize, int outputSize);
    
    /**
     * Initialize bias value.
     * @return Initialized bias
     */
    double initializeBias();
}

