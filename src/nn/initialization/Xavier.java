package nn.initialization;

import java.util.Random;

public class Xavier implements WeightInitializer {
    private Random random;
    
    public Xavier() {
        this.random = new Random();
    }
    
    @Override
    public double[] initialize(int inputSize, int outputSize) {
        double limit = Math.sqrt(6.0 / (inputSize + outputSize));
        double[] weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
        return weights;
    }
    
    @Override
    public double initializeBias() {
        return 0.0;
    }
}


