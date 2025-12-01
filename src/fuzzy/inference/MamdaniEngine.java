package fuzzy.inference;

import fuzzy.core.FuzzyRule;
import fuzzy.core.FuzzySet;
import fuzzy.core.LinguisticVariable;
import fuzzy.implication.ImplicationOperator;
import fuzzy.implication.MinImplication;
import fuzzy.norm.MinMaxNorm;
import fuzzy.norm.NormOperator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MamdaniEngine implements InferenceEngine {
	private final Map<String, LinguisticVariable> variables;
	private final List<FuzzyRule> rules;
	private final NormOperator norm;
	private final ImplicationOperator implication;
	private final int samples;

	public MamdaniEngine(Map<String, LinguisticVariable> variables, List<FuzzyRule> rules) {
		this(variables, rules, new MinMaxNorm(), new MinImplication(), 101);
	}

	public MamdaniEngine(Map<String, LinguisticVariable> variables, List<FuzzyRule> rules, NormOperator norm, ImplicationOperator implication) {
		this(variables, rules, norm, implication, 101);
	}

	public MamdaniEngine(Map<String, LinguisticVariable> variables, List<FuzzyRule> rules, NormOperator norm, ImplicationOperator implication, int samples) {
		this.variables = variables;
		this.rules = rules;
		this.norm = norm == null ? new MinMaxNorm() : norm;
		this.implication = implication == null ? new MinImplication() : implication;
		this.samples = Math.max(3, samples);
	}

	@Override
	public Map<String, Map<Double, Double>> infer(Map<String, Double> inputs) {
		return inferWithTrace(inputs).getAggregated();
	}

	@Override
	public InferenceTrace inferWithTrace(Map<String, Double> inputs) {
		Map<String, Map<Double, Double>> aggregated = new HashMap<>();
		Map<String, Double> ruleFirings = new HashMap<>();

		for (FuzzyRule rule : rules) {
			Double firing = null;
			for (Map.Entry<String, String> a : rule.getAntecedent().entrySet()) {
				String var = a.getKey();
				String setLabel = a.getValue();
				LinguisticVariable lv = variables.get(var);
				if (lv == null) { firing = 0.0; break; }

				Double crisp = inputs.get(var);
				double value = (crisp == null || crisp.isNaN()) ? lv.getDefault() : lv.clamp(crisp);

				double degree = 0.0;
				for (FuzzySet s : lv.getFuzzySets()) {
					if (s.getLabel().equals(setLabel)) { degree = s.getMembership(value); break; }
				}

				if (firing == null) firing = degree; else firing = this.norm.and(firing, degree);
			}

			if (firing == null || firing <= 0.0) continue;

			if (!rule.isEnabled()) continue;

			firing = firing * rule.getWeight();
			ruleFirings.put(rule.getId(), firing);

			for (Map.Entry<String, String> c : rule.getConsequent().entrySet()) {
				String outVar = c.getKey();
				String outSetLabel = c.getValue();
				LinguisticVariable outLv = variables.get(outVar);
				if (outLv == null) continue;

				double min = outLv.getDomainMin();
				double max = outLv.getDomainMax();
				if (max <= min) continue;

				Map<Double, Double> outMap = aggregated.computeIfAbsent(outVar, k -> new HashMap<>());

				for (int i = 0; i < samples; i++) {
					double x = min + (max - min) * i / (samples - 1);
					double mu = 0.0;
					for (FuzzySet s : outLv.getFuzzySets()) {
						if (s.getLabel().equals(outSetLabel)) { mu = s.getMembership(x); break; }
					}
					double implied = implication.imply(mu, firing);
					Double existing = outMap.get(x);
					double aggregatedVal = existing == null ? implied : this.norm.or(existing, implied);
					outMap.put(x, aggregatedVal);
				}
			}
		}

		return new InferenceTrace(aggregated, ruleFirings);
	}
}
