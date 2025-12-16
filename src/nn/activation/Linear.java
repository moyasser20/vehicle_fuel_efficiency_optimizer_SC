package nn.activation;

/**
 * Linear activation function: f(x) = x
 * Output range: (-∞, +∞)
 */
public class Linear implements ActivationFunction {
    @Override
    public double activate(double x) {
        return x;
    }
    
    @Override
    public double derivative(double activatedValue) {
        // Derivative of linear function is always 1
        return 1.0;
    }
}

