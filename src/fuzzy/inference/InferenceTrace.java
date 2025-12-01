package fuzzy.inference;

import java.util.Collections;
import java.util.Map;

public class InferenceTrace {
    private final Map<String, Map<Double, Double>> aggregated;
    private final Map<String, Double> ruleFirings;

    public InferenceTrace(Map<String, Map<Double, Double>> aggregated, Map<String, Double> ruleFirings) {
        this.aggregated = aggregated == null ? Collections.emptyMap() : aggregated;
        this.ruleFirings = ruleFirings == null ? Collections.emptyMap() : ruleFirings;
    }

    public Map<String, Map<Double, Double>> getAggregated() { return aggregated; }
    public Map<String, Double> getRuleFirings() { return ruleFirings; }
}
