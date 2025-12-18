package nn.initialization;

import java.util.Random;

public class He implements WeightInitializer {
    private Random random;
    
    public He() {
        this.random = new Random();
    }
    
    @Override
    public double[] initialize(int inputSize, int outputSize) {
        double std = Math.sqrt(2.0 / inputSize);
        double[] weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            double u1 = random.nextDouble();
            double u2 = random.nextDouble();
            double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            weights[i] = z * std;
        }
        return weights;
    }
    
    @Override
    public double initializeBias() {
        return 0.0;
    }
}


