package fuzzy.norm;

public class ProductSumNorm implements NormOperator {
    @Override
    public double and(double a, double b) { return a * b; }
    @Override
    public double or(double a, double b) { return a + b - a * b; }
}
