package fuzzy.core;

import fuzzy.inference.InferenceTrace;

import java.util.Collections;
import java.util.Map;

public class EvaluationResult {
    private final Map<String, Map<String, Double>> fuzzifiedInputs;
    private final InferenceTrace inferenceTrace;
    private final Map<String, Double> crispOutputs;

    public EvaluationResult(Map<String, Map<String, Double>> fuzzifiedInputs, InferenceTrace inferenceTrace, Map<String, Double> crispOutputs) {
        this.fuzzifiedInputs = fuzzifiedInputs == null ? Collections.emptyMap() : fuzzifiedInputs;
        this.inferenceTrace = inferenceTrace;
        this.crispOutputs = crispOutputs == null ? Collections.emptyMap() : crispOutputs;
    }

    public Map<String, Map<String, Double>> getFuzzifiedInputs() { return fuzzifiedInputs; }
    public InferenceTrace getInferenceTrace() { return inferenceTrace; }
    public Map<String, Double> getCrispOutputs() { return crispOutputs; }
}
