package fuzzy.inference;

import fuzzy.core.FuzzyRule;
import fuzzy.core.FuzzySet;
import fuzzy.core.LinguisticVariable;
import fuzzy.norm.MinMaxNorm;
import fuzzy.norm.NormOperator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SugenoEngine implements InferenceEngine {
    private final Map<String, LinguisticVariable> variables;
    private final List<FuzzyRule> rules;
    private final NormOperator norm;

    public SugenoEngine(Map<String, LinguisticVariable> variables, List<FuzzyRule> rules) {
        this(variables, rules, new MinMaxNorm());
    }

    public SugenoEngine(Map<String, LinguisticVariable> variables, List<FuzzyRule> rules, NormOperator norm) {
        this.variables = variables;
        this.rules = rules;
        this.norm = norm == null ? new MinMaxNorm() : norm;
    }

    @Override
    public Map<String, Map<Double, Double>> infer(Map<String, Double> inputs) {
        InferenceTrace t = inferWithTrace(inputs);
        return t.getAggregated();
    }

    @Override
    public InferenceTrace inferWithTrace(Map<String, Double> inputs) {
        Map<String, Double> numerators = new HashMap<>();
        Map<String, Double> denominators = new HashMap<>();
        Map<String, Double> ruleFirings = new HashMap<>();

        for (FuzzyRule r : rules) {
            for (String var : r.getSugenoConsequent().keySet()) {
                numerators.putIfAbsent(var, 0.0);
                denominators.putIfAbsent(var, 0.0);
            }
        }

        for (FuzzyRule rule : rules) {
            Double firing = null;
            for (Map.Entry<String, String> a : rule.getAntecedent().entrySet()) {
                String var = a.getKey();
                String setLabel = a.getValue();
                LinguisticVariable lv = variables.get(var);
                if (lv == null) { firing = 0.0; break; }
                Double crisp = inputs.get(var);
                double value;
                if (crisp == null || crisp.isNaN()) {
                    value = lv.getDefault();
                } else {
                    value = lv.clamp(crisp);
                }
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

            for (Map.Entry<String, Double> c : rule.getSugenoConsequent().entrySet()) {
                String outVar = c.getKey();
                double z = c.getValue();
                numerators.put(outVar, numerators.getOrDefault(outVar, 0.0) + firing * z);
                denominators.put(outVar, denominators.getOrDefault(outVar, 0.0) + firing);
            }
        }

        Map<String, Map<Double, Double>> aggregated = new HashMap<>();
        for (String var : numerators.keySet()) {
            double num = numerators.get(var);
            double den = denominators.getOrDefault(var, 0.0);
            double crisp = den == 0.0 ? 0.0 : num / den;
            Map<Double, Double> outMap = new HashMap<>();
            outMap.put(crisp, 1.0);
            aggregated.put(var, outMap);
        }

        return new InferenceTrace(aggregated, ruleFirings);
    }
}
