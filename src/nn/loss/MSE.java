package nn.loss;

/**
 * Mean Squared Error (MSE) loss function.
 * MSE = (1/n) * Σ(predicted - expected)²
 */
public class MSE implements LossFunction {
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException("Predicted and expected arrays must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - expected[i];
            sum += error * error;
        }
        return sum / predicted.length;
    }
    
    @Override
    public double[] computeGradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException("Predicted and expected arrays must have the same length");
        }
        
        double[] gradient = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            // Gradient of MSE: 2 * (predicted - expected) / n
            gradient[i] = 2.0 * (predicted[i] - expected[i]) / predicted.length;
        }
        return gradient;
    }
}

