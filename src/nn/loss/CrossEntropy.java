package nn.loss;

public class CrossEntropy implements LossFunction {
    private static final double EPSILON = 1e-15;
    
    @Override
    public double compute(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException("Predicted and expected arrays must have the same length");
        }
        
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double p = Math.max(EPSILON, Math.min(1.0 - EPSILON, predicted[i]));
            loss -= expected[i] * Math.log(p);
        }
        return loss;
    }
    
    @Override
    public double[] computeGradient(double[] predicted, double[] expected) {
        if (predicted.length != expected.length) {
            throw new IllegalArgumentException("Predicted and expected arrays must have the same length");
        }
        
        double[] gradient = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            double p = Math.max(EPSILON, Math.min(1.0 - EPSILON, predicted[i]));
            gradient[i] = -expected[i] / p;
        }
        return gradient;
    }
}


