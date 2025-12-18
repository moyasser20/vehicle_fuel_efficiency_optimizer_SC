package nn.initialization;

public interface WeightInitializer {
    double[] initialize(int inputSize, int outputSize);
    
    double initializeBias();
}


