package fuzzy.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LinguisticVariable {
    private final String name;
    private final List<FuzzySet> sets = new ArrayList<>();

    public LinguisticVariable(String name) {
        this.name = name;
    }

    public String getName() { return name; }

    public void addFuzzySet(FuzzySet set) { sets.add(set); }

    public List<FuzzySet> getFuzzySets() { return sets; }

    public Map<String, Double> fuzzify(double x) {
        Map<String, Double> degrees = new HashMap<>();
        for (FuzzySet s : sets) {
            degrees.put(s.getLabel(), s.getMembership(x));
        }
        return degrees;
    }

    public double getDomainMin() {
        double min = Double.POSITIVE_INFINITY;
        for (FuzzySet s : sets) min = Math.min(min, s.getMin());
        return min == Double.POSITIVE_INFINITY ? 0.0 : min;
    }

    public double getDomainMax() {
        double max = Double.NEGATIVE_INFINITY;
        for (FuzzySet s : sets) max = Math.max(max, s.getMax());
        return max == Double.NEGATIVE_INFINITY ? 0.0 : max;
    }

    public double clamp(double x) {
        double min = getDomainMin();
        double max = getDomainMax();
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    public double getDefault() {
        double min = getDomainMin();
        double max = getDomainMax();
        return (min + max) / 2.0;
    }
}
