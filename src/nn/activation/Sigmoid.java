package nn.activation;

public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double x) {
        x = Math.max(-500, Math.min(500, x));
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    @Override
    public double derivative(double activatedValue) {
        return activatedValue * (1.0 - activatedValue);
    }
}


