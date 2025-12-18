package nn.activation;

public class Linear implements ActivationFunction {
    @Override
    public double activate(double x) {
        return x;
    }
    
    @Override
    public double derivative(double activatedValue) {
        return 1.0;
    }
}


