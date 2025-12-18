package nn.initialization;

import java.util.Random;

public class RandomUniform implements WeightInitializer {
    private Random random;
    private double limit;
    
    public RandomUniform() {
        this(0.1);
    }
    
    public RandomUniform(double limit) {
        this.random = new Random();
        this.limit = limit;
    }
    
    @Override
    public double[] initialize(int inputSize, int outputSize) {
        double[] weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = (random.nextDouble() * 2.0 - 1.0) * limit;
        }
        return weights;
    }
    
    @Override
    public double initializeBias() {
        return (random.nextDouble() * 2.0 - 1.0) * limit;
    }
}


