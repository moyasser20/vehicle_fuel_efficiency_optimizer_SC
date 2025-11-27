package fuzzy.inference;

import java.util.Map;

public interface InferenceEngine {
    /**
     * Run inference and return a map of output variable name -> aggregated (x->degree) mapping.
     */
    Map<String, java.util.Map<Double, Double>> infer(Map<String, Double> inputs);

    /**
     * Run inference and return an InferenceTrace containing aggregated outputs plus per-rule firings.
     * Default implementation wraps the result of {@link #infer(Map)} with an empty rule-firings map.
     */
    default InferenceTrace inferWithTrace(Map<String, Double> inputs) {
        Map<String, java.util.Map<Double, Double>> agg = infer(inputs);
        return new InferenceTrace(agg, java.util.Collections.emptyMap());
    }
}
