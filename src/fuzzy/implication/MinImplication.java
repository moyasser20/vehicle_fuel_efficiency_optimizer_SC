package fuzzy.implication;

public class MinImplication implements ImplicationOperator {
    @Override
    public double imply(double mu, double alpha) { return Math.min(mu, alpha); }
}
