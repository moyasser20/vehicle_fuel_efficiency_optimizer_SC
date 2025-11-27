package fuzzy.core;

import fuzzy.mf.MembershipFunction;

public class FuzzySet {
    private final String label;
    private final MembershipFunction mf;
    private final double min;
    private final double max;

    public FuzzySet(String label, MembershipFunction mf, double min, double max) {
        this.label = label;
        this.mf = mf;
        this.min = min;
        this.max = max;
    }

    public String getLabel() { return label; }
    public double getMembership(double x) { return mf.getMembership(x); }
    public double getMin() { return min; }
    public double getMax() { return max; }
}
