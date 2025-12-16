package nn.activation;

/**
 * Interface for activation functions.
 * All activation functions must implement activate() and derivative() methods.
 */
public interface ActivationFunction {
    /**
     * Apply activation function to input value.
     * @param x Input value
     * @return Activated value
     */
    double activate(double x);
    
    /**
     * Compute derivative of activation function.
     * @param activatedValue The activated value (output of activate())
     * @return Derivative value
     */
    double derivative(double activatedValue);
}

