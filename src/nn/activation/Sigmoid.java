package nn.activation;

/**
 * Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * Output range: (0, 1)
 */
public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double x) {
        // Clamp to prevent overflow
        x = Math.max(-500, Math.min(500, x));
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    @Override
    public double derivative(double activatedValue) {
        // Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        return activatedValue * (1.0 - activatedValue);
    }
}

