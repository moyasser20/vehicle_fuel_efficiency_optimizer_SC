package nn.loss;

public interface LossFunction {
    double compute(double[] predicted, double[] expected);
    
    double[] computeGradient(double[] predicted, double[] expected);
}


