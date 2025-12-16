package nn.activation;

/**
 * Hyperbolic Tangent (Tanh) activation function: f(x) = tanh(x)
 * Output range: (-1, 1)
 */
public class Tanh implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }
    
    @Override
    public double derivative(double activatedValue) {
        // Derivative of tanh: f'(x) = 1 - tanh²(x) = 1 - f(x)²
        return 1.0 - activatedValue * activatedValue;
    }
}

