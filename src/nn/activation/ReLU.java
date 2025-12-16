package nn.activation;

/**
 * Rectified Linear Unit (ReLU) activation function: f(x) = max(0, x)
 * Output range: [0, +∞)
 */
public class ReLU implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.max(0.0, x);
    }
    
    @Override
    public double derivative(double activatedValue) {
        // Derivative of ReLU: 1 if x > 0, else 0
        return activatedValue > 0 ? 1.0 : 0.0;
    }
}

