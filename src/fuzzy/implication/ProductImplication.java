package fuzzy.implication;

public class ProductImplication implements ImplicationOperator {
    @Override
    public double imply(double mu, double alpha) { return mu * alpha; }
}
