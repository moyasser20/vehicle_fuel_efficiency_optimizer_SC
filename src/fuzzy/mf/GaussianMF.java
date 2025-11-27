package fuzzy.mf;

public class GaussianMF implements MembershipFunction {
    private final double mean;
    private final double sigma;

    public GaussianMF(double mean, double sigma) {
        this.mean = mean;
        this.sigma = sigma;
    }

    @Override
    public double getMembership(double x) {
        double diff = x - mean;
        return Math.exp(- (diff * diff) / (2 * sigma * sigma));
    }
}
