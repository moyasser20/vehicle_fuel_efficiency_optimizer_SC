package nn.loss;

/**
 * Interface for loss functions.
 */
public interface LossFunction {
    /**
     * Compute loss between predicted and expected outputs.
     * @param predicted Predicted output values
     * @param expected Expected output values
     * @return Loss value
     */
    double compute(double[] predicted, double[] expected);
    
    /**
     * Compute gradient of loss with respect to predicted output.
     * Used in backpropagation.
     * @param predicted Predicted output values
     * @param expected Expected output values
     * @return Gradient array
     */
    double[] computeGradient(double[] predicted, double[] expected);
}

