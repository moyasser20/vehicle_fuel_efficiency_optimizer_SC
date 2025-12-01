package fuzzy.core;
import fuzzy.defuzzification.CentroidDefuzzifier;
import fuzzy.defuzzification.Defuzzifier;
import fuzzy.inference.InferenceEngine;
import fuzzy.inference.InferenceTrace;
import fuzzy.inference.MamdaniEngine;
import fuzzy.implication.MinImplication;
import fuzzy.mf.GaussianMF;
import fuzzy.mf.TriangularMF;
import fuzzy.mf.TrapezoidalMF;
import fuzzy.norm.MinMaxNorm;
import fuzzy.norm.NormOperator;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class FuzzySystem {
	private final Map<String, LinguisticVariable> variables = new HashMap<>();
	private final List<FuzzyRule> rules = new ArrayList<>();
	private InferenceEngine engine;
	private Defuzzifier defuzzifier;

	private NormOperator defaultNorm = new MinMaxNorm();
	private String defaultMFType = "triangular";
	private int defaultNumSets = 3;
	private Defuzzifier defaultDefuzzifier = new CentroidDefuzzifier();
	private String defaultInference = "mamdani";

	public void addVariable(LinguisticVariable v) { variables.put(v.getName(), v); }
	public LinguisticVariable getVariable(String name) { return variables.get(name); }
	public void addRule(FuzzyRule r) { rules.add(r); }

	public void setInferenceEngine(InferenceEngine engine) { this.engine = engine; }
	public void setDefuzzifier(Defuzzifier d) { this.defuzzifier = d; }

	public Map<String, LinguisticVariable> getVariableMap() { return variables; }
	public List<FuzzyRule> getRulesList() { return rules; }

	public Map<String, Double> evaluate(Map<String, Double> inputs) {
		return evaluateWithTrace(inputs).getCrispOutputs();
	}

	public EvaluationResult evaluateWithTrace(Map<String, Double> inputs) {
		if (engine == null || defuzzifier == null) throw new IllegalStateException("Engine/Defuzzifier not set");

		Map<String, Map<String, Double>> fuzzified = new LinkedHashMap<>();
		for (Map.Entry<String, LinguisticVariable> ve : variables.entrySet()) {
			String var = ve.getKey();
			LinguisticVariable lv = ve.getValue();
			Double raw = inputs.get(var);
			double value = (raw == null || raw.isNaN()) ? lv.getDefault() : lv.clamp(raw);
			fuzzified.put(var, lv.fuzzify(value));
		}

		InferenceTrace trace = engine.inferWithTrace(inputs);
		Map<String, Double> crisp = new HashMap<>();
		for (Map.Entry<String, java.util.Map<Double, Double>> e : trace.getAggregated().entrySet()) {
			crisp.put(e.getKey(), defuzzifier.defuzzify(e.getValue()));
		}

		return new EvaluationResult(fuzzified, trace, crisp);
	}

	public RuleBase getRuleBase() { return new RuleBase(new ArrayList<>(rules)); }
	public void setRuleBase(RuleBase rb) { rules.clear(); if (rb != null) rules.addAll(rb.listRules()); }

	public void setDefaultNorm(NormOperator norm) { this.defaultNorm = norm; }
	public NormOperator getDefaultNorm() { return defaultNorm; }

	public void setDefaultMFType(String t) { this.defaultMFType = t == null ? "triangular" : t.toLowerCase(); }
	public String getDefaultMFType() { return defaultMFType; }

	public void setDefaultNumSets(int n) { this.defaultNumSets = Math.max(2, n); }
	public int getDefaultNumSets() { return defaultNumSets; }

	public void setDefaultDefuzzifier(Defuzzifier d) { this.defaultDefuzzifier = d; }
	public Defuzzifier getDefaultDefuzzifier() { return defaultDefuzzifier; }

	public void setDefaultInference(String s) { this.defaultInference = s == null ? "mamdani" : s.toLowerCase(); }
	public String getDefaultInference() { return defaultInference; }

	public LinguisticVariable addVariableWithDefaults(String name, double min, double max) {
		return addVariableWithDefaults(name, min, max, defaultNumSets, defaultMFType);
	}

	public LinguisticVariable addVariableWithDefaults(String name, double min, double max, int n, String mfType) {
		n = Math.max(2, n);
		String mf = mfType == null ? defaultMFType : mfType.toLowerCase();
		LinguisticVariable lv = new LinguisticVariable(name);
		double step = (max - min) / (n - 1);

		String[] labels;
		if (n == 2) labels = new String[]{"Low", "High"};
		else if (n == 3) labels = new String[]{"Low", "Medium", "High"};
		else {
			labels = new String[n];
			for (int i = 0; i < n; i++) labels[i] = "Term" + (i + 1);
		}

		for (int i = 0; i < n; i++) {
			double center = min + i * step;
			String label = labels[Math.min(i, labels.length - 1)];
			if (mf.equals("gaussian")) {
				double sigma = step <= 0 ? 1.0 : step / 2.0;
				lv.addFuzzySet(new FuzzySet(label, new GaussianMF(center, sigma), min, max));
			} else if (mf.equals("trapezoidal")) {
				double a = Math.max(min, center - step);
				double b = Math.max(min, center - step / 2.0);
				double c = Math.min(max, center + step / 2.0);
				double d = Math.min(max, center + step);
				lv.addFuzzySet(new FuzzySet(label, new TrapezoidalMF(a, b, c, d), min, max));
			} else {
				if (i == 0) {
					double a = min;
					double b = min;
					double c = Math.min(max, center + step);
					lv.addFuzzySet(new FuzzySet(label, new TrapezoidalMF(a, b, c, c + 1e-9), min, max));
				} else if (i == n - 1) {
					double a = Math.max(min, center - step);
					double b = max;
					lv.addFuzzySet(new FuzzySet(label, new TrapezoidalMF(a, a + 1e-9, b, b), min, max));
				} else {
					double a = center - step;
					double b = center;
					double c = center + step;
					lv.addFuzzySet(new FuzzySet(label, new TriangularMF(a, b, c), min, max));
				}
			}
		}

		addVariable(lv);
		return lv;
	}

	public void applyDefaultEngine() {
		if ("mamdani".equals(defaultInference)) {
			this.engine = new MamdaniEngine(variables, rules, defaultNorm, new MinImplication());
			this.defuzzifier = defaultDefuzzifier;
		}
	}
}
