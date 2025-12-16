package fuzzy.inference;

import java.util.Map;

public interface InferenceEngine {
    Map<String, java.util.Map<Double, Double>> infer(Map<String, Double> inputs);

    default InferenceTrace inferWithTrace(Map<String, Double> inputs) {
        Map<String, java.util.Map<Double, Double>> agg = infer(inputs);
        return new InferenceTrace(agg, java.util.Collections.emptyMap());
    }
}

//aggregated output from fuzzy inputs to crisp outputs
