package nn.activation;

public class Tanh implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }
    
    @Override
    public double derivative(double activatedValue) {
        return 1.0 - activatedValue * activatedValue;
    }
}


