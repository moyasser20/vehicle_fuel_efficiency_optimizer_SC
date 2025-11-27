package fuzzy.norm;

public class MinMaxNorm implements NormOperator {
    @Override
    public double and(double a, double b) { return Math.min(a, b); }
    @Override
    public double or(double a, double b) { return Math.max(a, b); }
}
